"""Opt-in #183 LFM measurement: real campaign, fixed recipe, exact wire audit.

Run each stage inside a coordinator-admitted PrismaBuild action. This driver
does not submit jobs, reserve devices, or change the production recipe. The
campaign/build stages run in the known-good producer container; seal/check run
in that environment too. Census/smoke commands are emitted for a separate host
action using the producer's stock-vLLM launcher and PB's Docker shim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
IMAGE = "eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c"
PRODUCER_IMAGE = "sha256:47dd0e9aaa4e7a6575d21cfc661d96a47c0e35e87c64e850631e210bdf04ebc0"
PRODUCER_CONFIG_SHA = "fda47b55fb7105c93e8a0bf99cd633191c198e4033719957734d065a635de31e"
PRODUCER_ROOTFS_SHA = "df0f8207331bd466df86322a178e14501f707f7b765e820a60e7ce9f28d51d71"
TESSERA_COMMIT = "ba582d476a3b6db9057ebd1385dc52926f171451"
STACK = "model.layers.13.feed_forward.experts"
EXPECTED_UNITS = {f"{STACK}.{expert}.{role}" for expert in range(32)
                  for role in ("w1", "w3", "w2")}
SCOPE = ["--tessera-platform", "sm_121", "--tessera-runtime-image", IMAGE,
         "--tessera-execution-mode", "eager", "--tessera-residency", "resident"]

# Actual direct dependencies of tessera_campaign.main/_calibration_tokens and
# the existing cached-export/wire-audit handoff. Importing them performs no
# checkpoint construction, calibration draw, encoding or CUDA work.
PRODUCER_DEPENDENCIES = {
    "datasets": ("load_dataset",),
    "torch": ("Generator", "bfloat16", "cuda"),
    "transformers": ("AutoModelForCausalLM", "AutoTokenizer"),
    "numpy": ("ndarray",),
    "safetensors": ("safe_open",),
    "safetensors.torch": ("load_file", "save_file"),
    "tessera.export": ("ActivationSource", "encode_linear_planes"),
    "tessera.cached_unit": ("make_unit_record", "verify_cached_unit"),
    "tessera.fused": ("parse_fused",),
    "prismaquant.tessera_campaign": ("main", "_calibration_tokens"),
    "prismaquant.tessera_export_lane": ("main",),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run(args, command, label, *, capture=False):
    command = list(map(str, command))
    started = time.time()
    with (args.out / f"{label}.log").open("x") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        if capture:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=log, text=True)
            log.write(result.stdout)
        else:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    write(args.out / f"{label}.command.json", {
        "argv": command, "exit_code": result.returncode,
        "started_unix": started, "ended_unix": time.time(),
    })
    result.check_returncode()
    return result.stdout.strip() if capture else None


def validate_topology(args):
    config = read(args.model / "config.json")
    expected = {"model_type": "lfm2_moe", "num_hidden_layers": 24,
                "num_experts": 32, "num_experts_per_tok": 4,
                "num_dense_layers": 2, "hidden_size": 2048,
                "moe_intermediate_size": 1792}
    require(all(config.get(k) == v for k, v in expected.items()),
            f"this bounded experiment requires the declared LFM topology: {expected}")
    return expected


def payload(args):
    # An owned, locally produced pickle; never accept an untrusted download.
    with (args.out / "cost.pkl").open("rb") as stream:
        return pickle.load(stream)


def campaign(args):
    producer_preflight(args)
    validate_topology(args)
    require(not (args.out / "cost.pkl").exists(), "campaign output exists; use a new attempt")
    run(args, [sys.executable, "-m", "cProfile", "-o", args.out / "campaign.pstats",
               "-m", "prismaquant.tessera_campaign",
               "--model", args.model, "--out", args.out / "cost.pkl",
               "--cache-dir", args.out / "cache", "--menu-mode", "attested",
               "--anchors", 1, "--max-rounds", 1, "--anchor-budget", 1,
               "--nsamples", 32, "--seqlen", 512, "--seed", 0,
               "--max-act-rows", 512, "--layer-stride", 13,
               "--hessian", "require", "--tp-degree", 1, *SCOPE], "campaign")


def producer_preflight(args):
    """Refuse unavailable campaign APIs before constructing the full model."""
    record = {"schema": "prismaquant.pq183-producer-dependencies.v1",
              "checked_unix": time.time(), "modules": {}, "problems": [],
              "checkpoint_construction_attempted": False,
              "calibration_contract": "unchanged tessera_campaign._calibration_tokens"}
    for name, symbols in PRODUCER_DEPENDENCIES.items():
        entry = record["modules"][name] = {"required_symbols": list(symbols)}
        try:
            module = importlib.import_module(name)
            missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
            require(not missing, f"{name}: missing required APIs {missing}")
            entry.update(imported=True, version=getattr(module, "__version__", None),
                         source=getattr(module, "__file__", None))
        except Exception as exc:
            entry.update(imported=False, error=repr(exc))
            record["problems"].append(f"{name}: {exc}")
    record["passed"] = not record["problems"]
    write(args.out / "producer-dependencies.json", record)
    require(record["passed"], "producer dependency preflight failed before model load: "
            + "; ".join(record["problems"]))


def allocate(args):
    """A declared fixed recipe, using the allocator's metadata authorities."""
    from safetensors import safe_open
    from prismaquant.layer_config import LAYER_CONFIG_META_KEY
    from prismaquant.tessera_expert_projection import (
        PROJECTION_KEY, allocation_expert_projection_block, carried_units,
    )
    from prismaquant.tessera_formats import parse_tessera_format_name
    from prismaquant.tessera_menu import assert_uniform_hessian_identity, priced_static_scales

    data = payload(args)
    prov, costs = data["provenance"], data["costs"]
    _, units, stacks = carried_units(prov[PROJECTION_KEY])
    require(set(units) == EXPECTED_UNITS and set(stacks.values()) == {STACK},
            "producer projection must contain exactly 96 layer-13 expert units")
    selected = {}
    for name in sorted(units):
        candidates = []
        for fmt, row in costs.get(name, {}).items():
            parsed = parse_tessera_format_name(fmt)
            if (parsed and parsed[0].payload_grid().name == "E4M3" and parsed[1] == 1024
                    and row.get("output_mse_measured") is True
                    and row.get("hessian_identity", {}).get("applied") is True):
                candidates.append(fmt)
        require(len(candidates) == 1, f"{name}: need one measured H-aware E4M3/q1024 row")
        selected[name] = candidates[0]

    # Every source body matrix is explicit, including omitted layers and
    # profile-pinned matrices. There are no fabricated BF16 cost rows.
    shapes = {}
    for path in sorted(args.model.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                shape = handle.get_slice(name).get_shape()
                if name.startswith("model.layers.") and name.endswith(".weight") and len(shape) == 2:
                    require(name[:-7] not in shapes, f"duplicate source tensor: {name}")
                    shapes[name[:-7]] = shape
    require(set(selected) <= set(shapes), "selected projection absent from source body weights")
    assignment = {name: selected.get(name, "BF16") for name in sorted(shapes)}
    meta = {
        "measurement_recipe": {
            "schema": "prismaquant.pq183-lfm-fixed-recipe.v1",
            "selection": "predeclared_measured_E4M3_q1024_layer13",
            "scope_selection": "lowest_index_ge_13_with_all_experts_observed_before_quantization",
            "coverage_histogram_sha256": "d8ab6d0816a53a715596c0f4ff2ab28cf2895270668822aa93143d96819c1fc3",
            "minimum_observed_routed_rows": 13, "calibration_nsamples": 32,
            "full_rank_hessian_claimed": False,
            "allocator_optimum_claimed": False, "quality_promotion_claimed": False,
            "unpriced_disposition": "explicit_BF16", "cost_sha256": sha(args.out / "cost.pkl"),
        },
        "tessera_hessian": assert_uniform_hessian_identity(costs),
        "tessera_activation_static_scales": priced_static_scales(selected, costs),
        "tessera_serving_scope": prov["tessera_serving_scope"],
        **allocation_expert_projection_block(data, assignment),
    }
    write(args.out / "layer_config.json", {**assignment, LAYER_CONFIG_META_KEY: meta})
    selected_parameters = sum(shapes[name][0] * shapes[name][1] for name in selected)
    priced_wire_bytes = sum(costs[name][fmt]["wire_bytes"] for name, fmt in selected.items())
    write(args.out / "recipe.json", {
        **meta["measurement_recipe"], "topology": validate_topology(args),
        "campaign_population": prov["population"],
        "selected": selected, "explicit_bf16": sorted(set(assignment) - set(selected)),
        "selected_units": len(selected), "selected_parameters": selected_parameters,
        "selected_priced_wire_bytes": priced_wire_bytes,
        "selected_wire_bpp": 8 * priced_wire_bytes / selected_parameters,
        "bpp_scope": "only_selected_quantizable_expert_parameters_excluding_BF16_and_framing",
        "whole_model_bpp": None, "kl": None,
    })


def build(args):
    allocate(args)
    data = payload(args)
    hessian = Path(data["provenance"]["hessian"]["capture_path"])
    build_sha = run(args, [sys.executable, "-m", "prismaquant.tessera_export_lane",
               "--model", args.model, "--assignment", args.out / "layer_config.json",
               "--hessian", hessian, "--write-build-json", args.out / "build.json",
               "--write-cached-expert-units", "--print-build-sha256", *SCOPE],
              "preflight", capture=True)
    require(re.fullmatch(r"[0-9a-f]{64}", build_sha), "preflight did not return its owned build digest")
    export_from_build(args, hessian, build_sha)


def export_from_build(args, hessian, build_sha):
    """The existing plan/export/seal handoff, also used by a frozen continuation."""
    build_record = read(args.out / "build.json")
    require(build_record.get("cached_expert_units"), "preflight did not bind cached expert units")
    run(args, [sys.executable, args.tessera_repo / "experiments/plan_from_layer_config.py",
               args.out / "layer_config.json", args.model, args.out / "plan.json",
               "--cover", "as-allocated", "--prismaquant", REPO], "plan")
    run(args, [sys.executable, args.tessera_repo / "experiments/export_tessera_serving.py",
               args.model, args.out / "exported", "--plan-json", args.out / "plan.json",
               "--priced-inputs", args.out / "build.json",
               "--priced-inputs-sha256", build_sha,
               "--hessian", hessian, "--cached-expert-units", build_record["cached_expert_units"],
               "--device", "cuda"], "export")
    seal(args)


def campaign_input_description(root, *, hash_files=True):
    """The consumed run05 population and input roster, without loading a model."""
    root = Path(root)
    with (root / "cost.pkl").open("rb") as stream:
        data = pickle.load(stream)  # Only the owned campaign's sealed output.
    config = read(root / "layer_config.json")
    meta, costs, prov = config["__prismaquant__"], data["costs"], data["provenance"]
    recipe, command, previous = (read(root / name) for name in
                                 ("recipe.json", "campaign.command.json", "host-status.json"))
    require(command.get("exit_code") == 0, "prior campaign did not exit successfully")
    argv = command["argv"]
    for flag, value in {"--menu-mode": "attested", "--anchors": "1", "--max-rounds": "1",
                        "--anchor-budget": "1", "--nsamples": "32", "--seqlen": "512",
                        "--seed": "0", "--max-act-rows": "512", "--layer-stride": "13",
                        "--hessian": "require", "--tp-degree": "1"}.items():
        require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value,
                f"prior campaign changed its declared calibration/scope flag {flag}")
    require(set(costs) == EXPECTED_UNITS, "prior campaign is not the complete measured 96-unit population")
    receipts = meta["tessera_expert_wires"]
    require(set(receipts) == EXPECTED_UNITS, "prior assignment does not carry exactly 96 priced wires")
    require(meta["tessera_expert_stack_formats"] == {STACK: "TESSERA_E4M3_K1_R1024"},
            "prior assignment differs from the frozen whole-stack recipe")
    for name, rows in costs.items():
        fmt = config[name]
        require(fmt == "TESSERA_E4M3_K1_R1024" and set(rows) == {fmt},
                f"{name}: prior campaign/assignment changed rung")
        row = rows[fmt]
        require(row.get("output_mse_measured") is True
                and row.get("hessian_identity", {}).get("applied") is True
                and row["wire_bytes"] == receipts[name]["blob_bytes"],
                f"{name}: prior row is not the measured H-aware priced wire")
        require(data["tessera_expert_wires"][name][fmt] == receipts[name],
                f"{name}: assignment receipt differs from the measured cost payload")
    require(all(value == "BF16" for name, value in config.items()
                if name not in EXPECTED_UNITS and name != "__prismaquant__"),
            "prior assignment changed an unpriced unit")
    cost_sha = sha(root / "cost.pkl")
    require(recipe.get("selected_units") == 96
            and recipe["cost_sha256"] == meta["measurement_recipe"]["cost_sha256"] == cost_sha,
            "prior recipe is not bound to this complete cost payload")
    target = {"platform": "sm_121", "runtime_image": IMAGE,
              "execution_mode": "eager", "residency": "resident"}
    require(meta["tessera_serving_scope"] == prov["tessera_serving_scope"]
            and meta["tessera_serving_scope"]["target"] == target,
            "prior measured scope differs from the continuation target")
    require(previous["producer_image_id"] == PRODUCER_IMAGE
            and previous["serving_image"] == IMAGE
            and previous["tessera_source"]["commit"] == TESSERA_COMMIT,
            "prior producer/source/serving runtime differs")
    require(re.fullmatch(r"[0-9a-f]{40}", previous["source_snapshot"])
            and previous.get("cleanup") and all(item.get("safe") is True
                                                 for item in previous["cleanup"].values()),
            "prior source snapshot or completed container cleanup is unverified")
    verify_producer_image(read(root / "producer-image.json"), PRODUCER_IMAGE)
    require(IMAGE in read(root / "serving-image.json").get("RepoDigests", []),
            "prior serving image receipt lacks the exact target digest")
    require(read(root / "producer-dependencies.json")["passed"] is True,
            "prior producer dependency preflight failed")
    wire_dir = Path(meta["tessera_expert_wire_dir"])
    require(str(wire_dir) == prov["wire_dir"] and wire_dir.parent.parent == root,
            "prior wire paths do not belong to the declared original campaign root")
    hessian = Path(prov["hessian"]["capture_path"])
    require(hessian.is_relative_to(root), "prior Hessian path escapes the campaign")
    files = {"cost.pkl", "layer_config.json", "recipe.json", "campaign.command.json",
             "host-status.json", "producer-image.json", "serving-image.json", "producer-dependencies.json",
             str(hessian.relative_to(root)), str(hessian.relative_to(root)) + ".provenance.json"}
    wire_files = set()
    for record in receipts.values():
        require(Path(record["file"]).name == record["file"], "priced wire file is not a local leaf")
        rel = str((wire_dir / record["file"]).relative_to(root))
        require(rel not in wire_files, "two priced units share a wire file")
        wire_files.add(rel)
    require(len(wire_files) == 96, "prior wire roster is incomplete")
    files.update(wire_files)
    hashes = {}
    for name in sorted(files):
        path = root / name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
                f"campaign input is not a contained regular file: {name}")
        hashes[name] = sha(path) if hash_files else None
    if hash_files:
        for record in receipts.values():
            path = wire_dir / record["file"]
            require(hashes[str(path.relative_to(root))] == record["blob_sha256"]
                    and path.stat().st_size == record["blob_bytes"], "prior wire differs from its priced receipt")
    return {"schema": "prismaquant.pq183-campaign-inputs.v1", "campaign_root": str(root),
            "campaign_source_snapshot": previous["source_snapshot"],
            "tessera_source": previous["tessera_source"], "producer_image_id": PRODUCER_IMAGE,
            "serving_target": target, "measured_units": 96, "priced_wires": 96,
            "files": hashes}


def verify_campaign_inputs(args):
    require(args.campaign_input and args.campaign_input_manifest and args.campaign_input_manifest_sha256,
            "continuation requires the original campaign root and a sealed input manifest/digest")
    require(args.campaign_input.is_absolute() and args.campaign_input != args.out
            and not args.out.is_relative_to(args.campaign_input), "continuation output must be fresh and outside its input")
    require(sha(args.campaign_input_manifest) == args.campaign_input_manifest_sha256,
            "campaign input manifest changed")
    manifest = read(args.campaign_input_manifest)
    require(manifest.get("schema") == "prismaquant.pq183-campaign-inputs.v1"
            and manifest.get("campaign_root") == str(args.campaign_input), "wrong campaign input seal")
    require(isinstance(manifest.get("files"), dict)
            and {"cost.pkl", "layer_config.json"} <= set(manifest["files"]),
            "campaign seal omits the authoritative cost/assignment inputs")
    for name, digest in manifest["files"].items():
        path = args.campaign_input / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts
                and path.resolve().is_relative_to(args.campaign_input.resolve())
                and not path.is_symlink() and path.is_file(), "campaign input escaped its sealed root")
        require(sha(path) == digest, f"frozen campaign input changed: {name}")
    # The bound pickle is read only AFTER its expected digest was verified.
    described = campaign_input_description(args.campaign_input, hash_files=False)
    require(set(described["files"]) == set(manifest["files"])
            and {key: value for key, value in described.items() if key != "files"}
            == {key: value for key, value in manifest.items() if key != "files"},
            "campaign input roster, source, scope or measured completion differs")
    return manifest


def seal_campaign(args):
    """Offline input preparation, run through PB before continuation admission."""
    require(args.campaign_input and args.campaign_input_manifest,
            "seal-campaign requires --campaign-input and --campaign-input-manifest")
    require(not args.campaign_input_manifest.resolve().is_relative_to(args.campaign_input.resolve()),
            "write the input seal outside the immutable original campaign")
    manifest = campaign_input_description(args.campaign_input)
    write(args.campaign_input_manifest, manifest)
    print(json.dumps({"path": str(args.campaign_input_manifest),
                      "sha256": sha(args.campaign_input_manifest), "measured_units": 96, "priced_wires": 96}))


def continue_export(args):
    """Export existing prices and bytes; never call the campaign or allocator."""
    manifest = verify_campaign_inputs(args)
    # The host resolves the admitted PQ snapshot before launching this image,
    # which intentionally has no Git dependency. Bind its receipt to this phase.
    host_receipt = read(args.out / "host-status.json")
    source_snapshot = host_receipt.get("source_snapshot")
    require(host_receipt.get("schema") == "prismaquant.pq183-host-observation.v1"
            and isinstance(source_snapshot, str) and re.fullmatch(r"[0-9a-f]{40}", source_snapshot)
            and host_receipt.get("campaign_source_snapshot") == manifest["campaign_source_snapshot"]
            and host_receipt.get("phases", {}).get("continue-export", {}).get(
                "campaign_input_manifest_sha256") == args.campaign_input_manifest_sha256,
            "continuation host source receipt is absent, malformed or bound to different inputs")
    producer_preflight(args)
    for name in ("cost.pkl", "layer_config.json", "recipe.json"):
        destination = args.out / name
        require(not destination.exists(), f"continuation output collision: {name}")
        shutil.copyfile(args.campaign_input / name, destination)
        require(sha(destination) == manifest["files"][name], f"copied input changed: {name}")
    data = payload(args)
    hessian = Path(data["provenance"]["hessian"]["capture_path"])
    from prismaquant.tessera_export_lane import preflight, write_cached_expert_units
    from prismaquant.tessera_serving_scope import ServingTarget
    # The shared assignment/Hessian/source/wire gates run before any transport
    # bundle is made. The old cache is read-only; only its writer is deferred.
    report = preflight(args.model, assignment_path=args.out / "layer_config.json",
                       hessian_path=hessian, target=ServingTarget(**manifest["serving_target"]),
                       cached_expert_units=False)
    write(args.out / "continuation-preflight.json", report)
    projection = report["selected_serving_scope"]["expert_projection"]
    require(set(projection["units"]) == EXPECTED_UNITS, "preflight did not validate all 96 priced wires")
    bundle_dir = args.out / "cached-expert-units"
    bundle_dir.mkdir()
    for record in projection["units"].values():
        original = Path(projection["wire_dir"]) / record["file"]
        destination = bundle_dir / record["file"]
        shutil.copyfile(original, destination)
        require(sha(destination) == record["blob_sha256"] == sha(original), "wire transport copy changed")
    cached = write_cached_expert_units({**projection, "wire_dir": str(bundle_dir)})
    build_record = {**report["build"], "cached_expert_units": str(cached)}
    build_bytes = (json.dumps(build_record, indent=2, sort_keys=True) + "\n").encode()
    with (args.out / "build.json").open("xb") as stream:
        stream.write(build_bytes)
    write(args.out / "continuation.json", {
        "input_manifest_sha256": args.campaign_input_manifest_sha256,
        "campaign_source_snapshot": manifest["campaign_source_snapshot"],
        "continuation_source_snapshot": source_snapshot,
        "assignment_unchanged": True, "cost_unchanged": True, "priced_wire_copies": 96,
        "original_input_root": str(args.campaign_input), "original_inputs_mounted_read_only": True,
        "reallocated": False, "requantized": False})
    export_from_build(args, hessian, hashlib.sha256(build_bytes).hexdigest())
    verify_campaign_inputs(args)


def wire_audit(args):
    """Read safetensors payloads, independently of exported sidecar hashes."""
    from safetensors import safe_open
    from tessera.fused import parse_fused
    from prismaquant.tessera_expert_projection import EXPERT_WIRES_KEY, PROJECTION_KEY

    assignment = read(args.out / "layer_config.json")["__prismaquant__"]
    receipts = assignment[EXPERT_WIRES_KEY]
    projection = assignment[PROJECTION_KEY]["producer"]
    units = {unit["tensor"].removesuffix(".weight"): unit
             for stack in projection["stacks"].values() for unit in stack["units"]}
    require(set(receipts) == set(units) == EXPECTED_UNITS, "wire audit roster drift")
    expected_wires = {unit["wire"]: name for name, unit in units.items()}
    seen = {}
    for path in sorted((args.out / "exported").glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for wire in handle.keys():
                if not wire.endswith(".wire"):
                    continue
                require(wire in expected_wires and wire not in seen, f"unexpected/duplicate wire {wire}")
                name = expected_wires[wire]
                tensor = handle.get_tensor(wire)
                require(str(tensor.dtype) == "torch.uint8" and tensor.ndim == 1,
                        f"{wire}: expected flat byte tensor")
                raw = tensor.numpy().tobytes()
                members = parse_fused(raw)
                require(len(members) == 1, f"{wire}: not one projected member")
                member = members[0]
                unit = units[name]
                require(member.name == unit["projection"] and member.rows == unit["rows"],
                        f"{wire}: framed role/geometry mismatch")
                blob_sha = hashlib.sha256(member.blob).hexdigest()
                record = receipts[name]
                cached = Path(assignment["tessera_expert_wire_dir"]) / record["file"]
                require(blob_sha == record["blob_sha256"] == sha(cached),
                        f"{wire}: written payload differs from priced receipt/cache")
                require(member.blob == cached.read_bytes(), f"{wire}: cached bytes differ")
                seen[wire] = {"unit": name, "shard": path.name,
                              "blob_sha256": blob_sha, "blob_bytes": len(member.blob),
                              "container_sha256": hashlib.sha256(raw).hexdigest(),
                              "container_bytes": len(raw)}
    require(set(seen) == set(expected_wires), "missing exported expert wires")
    return {"schema": "prismaquant.pq183-written-wire-audit.v1", "passed": True,
            "units": len(seen), "wires": seen,
            "layer_config_sha256": sha(args.out / "layer_config.json"),
            "cost_sha256": sha(args.out / "cost.pkl")}


def seal(args):
    from tessera.serving_parts import source_identity
    audit = wire_audit(args)
    write(args.out / "wire-audit.json", audit)
    manifest = read(args.out / "exported/tessera_serving_manifest.json")
    write(args.out / "artifact-seal.json", {
        "checkpoint": str(args.out / "exported"),
        "checkpoint_identity": source_identity(args.out / "exported"),
        "export_identity": manifest["export_identity"],
        "wire_audit_sha256": sha(args.out / "wire-audit.json"),
    })


def check(args):
    from tessera.serving_parts import source_identity
    from tessera.serving.contract import derive_smoke_status
    from tessera.serving.build_identity import is_complete, incomplete_reason
    seal_record = read(args.out / "artifact-seal.json")
    require(seal_record["checkpoint"] == str(args.out / "exported"), "seal checkpoint drift")
    require(source_identity(args.out / "exported") == seal_record["checkpoint_identity"],
            "artifact changed after sealing")
    require(sha(args.out / "wire-audit.json") == seal_record["wire_audit_sha256"], "wire audit drift")
    require(wire_audit(args) == read(args.out / "wire-audit.json"), "post-serve wire audit drift")
    census = read(args.out / "census/check.json")
    require(census.get("verdict") == "passed" and census.get("require_attested") is True,
            "census did not attest the actual artifact's complete selected population")
    pair = read(args.out / "smoke/pair.json")
    require(pair.get("contract_record", {}).get("record"), "smoke pair has no derived record")
    smoke_status = derive_smoke_status(pair["contract_record"])
    for arm in ("bf16", "tessera"):
        identity = read(args.out / f"smoke/smoke_{arm}.build.json")
        require(is_complete(identity), incomplete_reason(identity))
        require(identity["identity"]["image"] == IMAGE and identity["identity"]["eager"] is True
                and identity["identity"]["compiled_forward"] is False,
                f"{arm}: wrong serving runtime or execution mode")
        expected_identity = (seal_record["export_identity"]["source"] if arm == "bf16"
                             else seal_record["checkpoint_identity"])
        require(read(args.out / f"smoke/identity_{arm}_before.json") == expected_identity
                and read(args.out / f"smoke/identity_{arm}_after.json") == expected_identity,
                f"{arm}: served checkpoint differs from the sealed export/source")
        if arm == "tessera":
            require(identity["identity"]["serve_mode"] == "resident", "student residency mismatch")
    write(args.out / "artifact-after.json", {"unchanged": True,
          "artifact_seal_sha256": sha(args.out / "artifact-seal.json"),
          "census_check_sha256": sha(args.out / "census/check.json"),
          "smoke_pair_sha256": sha(args.out / "smoke/pair.json"),
          "smoke_status": smoke_status, "accepted": smoke_status == "recorded",
          "quality_promotion_claimed": False, "checked_unix": time.time()})
    require(smoke_status == "recorded", f"bounded serving observation failed: {smoke_status}")


def serving_commands(args):
    """The fixed serving phase graph shared by preview and admitted execution."""
    out, ts = args.out, args.tessera_repo
    name = args.container_prefix
    census_body = shlex.join(["python3", "tools/tessera_route_census.py", str(out / "exported"),
                              "/census/census.json", "--tessera-commit", args.tessera_commit,
                              "--prompt-tokens", "64", "--max-model-len", "512",
                              "--gpu-memory-utilization", "0.35"])
    census_body += ' --runtime-image "$TESSERA_CENSUS_RUNTIME_IMAGE"'
    census = ["env", f"TS={ts}", f"RUNS={out}", f"EXT={out / 'ext'}", f"IMG={IMAGE}",
              str(ts / "experiments/tessera_plugin_run.sh"), "--name", f"{name}-census",
              "--memory=64g", "--memory-swap=64g", "--ipc=host",
              "-e", "TESSERA_SERVE_MODE=resident", "-e", "OMP_NUM_THREADS=1",
              "-e", "MKL_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
              "-e", "MAX_JOBS=4", "-e", "CMAKE_BUILD_PARALLEL_LEVEL=4",
              "-e", "NINJAFLAGS=-j4", "-e", "MAKEFLAGS=-j4",
              "-v", "/mnt/shared:/mnt/shared:ro",
              "-v", f"{out / 'exported'}:{out / 'exported'}:ro",
              "-v", f"{out / 'census'}:/census", "--", census_body]
    gate = [sys.executable, str(ts / "experiments/ts5_census_check.py"),
            "--plan", str(out / "plan.json"), "--checkpoint", str(out / "exported"),
            "--census", str(out / "census/census.json"), "--runtime-image", IMAGE,
            "--require-attested", "--out", str(out / "census/check.json")]
    smoke = ["env", f"TS={ts}", f"IMAGE={IMAGE}", f"EXT={out / 'ext'}",
             "TESSERA_GPU_MEM_UTIL=0.35",
             f"VLLM_CACHE={out / 'vllm-cache'}", f"NAME_PREFIX={name}-smoke", f"PORT={args.port}",
             f"PY={sys.executable}", str(ts / "experiments/moe_greedy_smoke_pair.sh"),
             str(args.model), str(out / "exported"), str(out / "artifact-seal.json"), str(out / "smoke")]
    return {"schema": "prismaquant.pq183-served-commands.v1",
                      "commands": {"census": census, "census_gate": gate, "smoke_pair": smoke},
                      "owned_container_names": [f"{name}-census", f"{name}-smoke-bf16",
                                                f"{name}-smoke-tessera"],
                      "prepare_directories": [str(out / "census"), str(out / "ext"), str(out / "vllm-cache")],
                      "requirements": ["PB admitted exclusive GPU host action; preserve Docker shim",
                                       "refuse preexisting exact owned container names before launch",
                                       "capture both-host Netdata and power/CPU/memory throughout",
                                       "verify cleanup of exact owned container IDs on every exit",
                                       "compare artifact seal before/after each served stage",
                                       "run check stage after census and smoke before releasing evidence"],
                      "producer_image_id": PRODUCER_IMAGE,
                      "serving_image": IMAGE, "serving_mode": "eager", "residency": "resident",
                      "tp_degree": 1}


def commands(args):
    print(json.dumps(serving_commands(args), indent=2))


def verify_tessera_source(root, manifest_path, expected_sha):
    """Bind a complete external source snapshot to the seal in the PB input."""
    require(sha(manifest_path) == expected_sha, "Tessera source manifest digest changed")
    manifest = read(manifest_path)
    require(set(manifest) == {"schema", "commit", "files"}
            and manifest["schema"] == "prismaquant.pq183-tessera-source.v1"
            and manifest["commit"] == TESSERA_COMMIT, "invalid pinned Tessera source seal")
    files = manifest["files"]
    require(isinstance(files, dict) and bool(files), "empty Tessera source seal")
    actual = {}
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if ".git" in rel.parts:
            continue
        require(not path.is_symlink(), f"source symlink is not sealed: {rel}")
        if path.is_file():
            actual[str(rel)] = sha(path)
    require(actual == files, "Tessera source files differ from the complete pinned seal")
    return {"commit": manifest["commit"], "manifest_sha256": expected_sha, "files": len(files)}


def verify_producer_image(inspected, image):
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", image) and inspected["Id"] == image,
            "producer requires the explicit local immutable Docker image ID")
    for key, expected in (("Config", PRODUCER_CONFIG_SHA), ("RootFS", PRODUCER_ROOTFS_SHA)):
        digest = hashlib.sha256(json.dumps(inspected[key], sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
        require(digest == expected, f"producer {key} differs from the verified numerical environment")


def producer(args):
    campaign(args)
    build(args)


def host(args):
    """One deterministic admitted action, with no agent between runtime phases."""
    sys.path.insert(0, str(REPO / "experiments"))
    from pq87_paired_validation import require_container_name_available
    from pq87_physical_ab import _http, cleanup_container, dump, remaining

    owner = os.environ.get("PRISMABUILD_CONTAINER_OWNER")
    require(owner and os.environ.get("CUDA_VISIBLE_DEVICES") != "",
            "host stage requires an admitted PrismaBuild GPU action")
    require(args.tessera_source_manifest and args.tessera_source_manifest_sha256,
            "host stage requires the sealed Tessera source manifest and digest")
    require(len(args.netdata_url) == 2 and len(set(args.netdata_url)) == 2,
            "declare both distinct Netdata URLs")
    require(args.seconds > 0, "deadline must be positive")
    require(args.producer_image, "host stage requires the host's explicit immutable producer image ID")
    continuation_fields = (args.campaign_input, args.campaign_input_manifest,
                           args.campaign_input_manifest_sha256)
    require(all(continuation_fields) or not any(continuation_fields),
            "continuation must declare the original campaign root, input manifest and digest together")
    require(not args.out.exists(), "host output exists; use a fresh attempt")
    args.out.mkdir(parents=True)
    deadline = time.monotonic() + args.seconds
    nonce = uuid.uuid4().hex[:12]
    args.container_prefix += "-" + nonce
    serve = serving_commands(args)
    names = [args.container_prefix + "-producer", *serve["owned_container_names"],
             args.container_prefix + "-check"]
    state = {"schema": "prismaquant.pq183-host-observation.v1", "status": "inconclusive",
             "started_unix": time.time(), "deadline_seconds": args.seconds,
             "source_snapshot": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                         cwd=REPO, text=True).strip(),
             "owner": owner, "names": names, "phases": {}, "observed_containers": {},
             "producer_image_id": args.producer_image, "serving_image": IMAGE,
             "memory_cap_gib": 64,
             "native_thread_policy": {"producer": 1, "census": 1,
                                       "smoke": "unchanged producer launcher; PB four-CPU cpuset bounds all threads"},
             "cpu_affinity": sorted(os.sched_getaffinity(0)), "netdata_urls": args.netdata_url}
    require(len(state["cpu_affinity"]) <= 4, "action affinity exceeds declared four-CPU envelope")
    stop = threading.Event()
    telemetry_ok = {url: False for url in args.netdata_url}
    monitor_errors = []

    def inspect_name(name):
        got = subprocess.run(["docker", "inspect", name], capture_output=True, text=True, timeout=10)
        if got.returncode:
            require(any(f"no such {kind}: {name}" in (got.stdout + got.stderr).lower()
                        for kind in ("object", "container")), f"cannot inspect owned name {name}")
            return None
        record = json.loads(got.stdout)[0]
        labels = record.get("Config", {}).get("Labels") or {}
        require(record["Name"] == "/" + name and labels.get("prismabuild.action") == owner,
                f"container ownership mismatch: {name}")
        cid = record["Id"]
        require(re.fullmatch(r"[0-9a-f]{64}", cid), f"invalid exact container ID: {name}")
        previous = state["observed_containers"].get(name)
        require(previous is None or previous["id"] == cid, f"container identity changed: {name}")
        mask = record["HostConfig"].get("CpusetCpus", "")
        cpus = set()
        for part in mask.split(","):
            if "-" in part:
                first, last = map(int, part.split("-"))
                cpus.update(range(first, last + 1))
            elif part:
                cpus.add(int(part))
        require(cpus and cpus <= set(state["cpu_affinity"]), "container widened or omitted the admitted CPU mask")
        require(0 < record["HostConfig"].get("Memory", 0) <= 64 * 2**30,
                "container memory limit exceeds declared envelope")
        state["observed_containers"][name] = {
            "id": cid, "labels": labels, "image": record["Image"],
            "running": record["State"]["Running"], "cpu_set": record["HostConfig"].get("CpusetCpus"),
            "memory_limit": record["HostConfig"].get("Memory"),
            "host_pid": record["State"].get("Pid"),
            "native_environment": [value for value in record["Config"].get("Env", [])
                                   if value.split("=", 1)[0] in
                                   {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MAX_JOBS"}],
        }
        return record

    def collect():
        with (args.out / "telemetry.jsonl").open("x", buffering=1) as log:
            tick = 0
            while not stop.is_set():
                row = {"time": time.time(), "meminfo": Path("/proc/meminfo").read_text()}
                try:
                    for name in names:
                        inspect_name(name)
                    if tick % 5 == 0:
                        row["gpu_power_w"] = subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                            text=True, timeout=3).strip()
                        row["cpu_stat"] = Path("/proc/stat").read_text()
                        row["container_processes"] = {}
                        for name, observed in list(state["observed_containers"].items()):
                            pid = observed.get("host_pid")
                            if pid:
                                process = row["container_processes"][name] = {"pid": pid}
                                for field in ("status", "io"):
                                    try:
                                        process[field] = Path(f"/proc/{pid}/{field}").read_text()
                                    except OSError as exc:
                                        process[field + "_error"] = repr(exc)
                        for url in args.netdata_url:
                            try:
                                row[url] = _http(url.rstrip("/") + "/api/v1/allmetrics?format=json", timeout=2)
                                require(bool(row[url]), "empty Netdata reply")
                                telemetry_ok[url] = True
                            except Exception as exc:
                                row[url] = {"error": repr(exc)}
                except Exception as exc:
                    row["error"] = repr(exc)
                    monitor_errors.append(repr(exc))
                log.write(json.dumps(row) + "\n")
                tick += 1
                stop.wait(2)

    def bound_source():
        return verify_tessera_source(args.tessera_repo, args.tessera_source_manifest,
                                     args.tessera_source_manifest_sha256)

    def phase(label, argv):
        state["phases"][label] = {"argv": list(map(str, argv)), "source": bound_source(),
                                   "started_unix": time.time()}
        if args.campaign_input:
            original = verify_campaign_inputs(args)
            state["phases"][label]["campaign_input_manifest_sha256"] = args.campaign_input_manifest_sha256
            state["campaign_source_snapshot"] = original["campaign_source_snapshot"]
        dump(args.out / "host-status.json", state)
        proc = None
        try:
            with (args.out / f"host-{label}.log").open("x") as log:
                proc = subprocess.Popen(list(map(str, argv)), stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                code = proc.wait(timeout=remaining(deadline))
            state["phases"][label]["exit_code"] = code
            require(code == 0, f"phase {label} exited {code}")
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
            state["phases"][label]["finished_unix"] = time.time()
            if args.campaign_input:
                verify_campaign_inputs(args)
                state["phases"][label]["campaign_inputs_unchanged_after"] = True
            dump(args.out / "host-status.json", state)

    def producer_command(stage, name):
        return ["docker", "run", "--rm", "--pull=never", "--name", name,
                "--cidfile", str(args.out / f"{name}.cid"),
                "--label", "prismaquant.pq183.campaign=" + nonce,
                "--gpus", "all", "--memory=64g", "--memory-swap=64g", "--cpus=4",
                "--ipc=host", "--network=host", "-v", f"{REPO}:{REPO}:ro",
                "-v", f"{args.tessera_repo}:{args.tessera_repo}:ro",
                "-v", "/mnt/shared:/mnt/shared:ro", "-v", f"{args.model}:{args.model}:ro",
                *(["-v", f"{args.campaign_input}:{args.campaign_input}:ro",
                   "-v", f"{args.campaign_input_manifest}:{args.campaign_input_manifest}:ro"]
                  if args.campaign_input else []),
                "-v", f"{args.out}:{args.out}", "-w", str(REPO),
                "-e", "OMP_NUM_THREADS=1", "-e", "MKL_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
                "-e", "MAX_JOBS=4", "-e", "CMAKE_BUILD_PARALLEL_LEVEL=4",
                "-e", "NINJAFLAGS=-j4", "-e", "MAKEFLAGS=-j4",
                "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONNOUSERSITE=1",
                "-e", "HF_HOME=/opt/pq183-hf-cache", "-e", "HF_HUB_OFFLINE=1",
                "-e", "HF_DATASETS_OFFLINE=1", "-e", f"TORCH_EXTENSIONS_DIR={args.out / 'ext'}",
                "-e", f"TRITON_CACHE_DIR={args.out / 'triton'}", "--entrypoint", "python3",
                args.producer_image, str(Path(__file__).resolve()), stage,
                "--model", str(args.model), "--out", str(args.out),
                "--tessera-repo", str(args.tessera_repo), "--tessera-commit", TESSERA_COMMIT,
                *(["--campaign-input", str(args.campaign_input),
                   "--campaign-input-manifest", str(args.campaign_input_manifest),
                   "--campaign-input-manifest-sha256", args.campaign_input_manifest_sha256]
                  if args.campaign_input else [])]

    def verify_artifact(label):
        from tessera.serving_parts import source_identity
        bound_source()
        seal_record = read(args.out / "artifact-seal.json")
        require(seal_record["checkpoint"] == str(args.out / "exported"), "artifact seal path differs")
        require(source_identity(args.out / "exported") == seal_record["checkpoint_identity"],
                "artifact differs from seal")
        write(args.out / f"{label}-artifact-binding.json", {
            "unchanged": True, "artifact_seal_sha256": sha(args.out / "artifact-seal.json"),
            "checked_unix": time.time()})

    def interrupted(signum, _frame):
        raise TimeoutError(f"host action interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    thread = None
    try:
        state["tessera_source"] = bound_source()
        for name in names:
            require_container_name_available(name)
        for image in (args.producer_image, IMAGE):
            inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", image],
                                                           text=True, timeout=10))[0]
            if image == args.producer_image:
                verify_producer_image(inspected, image)
            else:
                require(image in inspected.get("RepoDigests", []), "serving runtime digest mismatch")
            write(args.out / ("producer-image.json" if image == args.producer_image else "serving-image.json"), inspected)
        for directory in ("census", "ext", "vllm-cache", "triton"):
            (args.out / directory).mkdir()
        # Host tools must import the pinned tree without mutating its seal.
        os.environ.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                          MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        phase("instrument-preflight", [sys.executable, "-c", "import requests, tokenizers, numpy"])
        thread = threading.Thread(target=collect, daemon=True)
        thread.start()
        phase("continue-export" if args.campaign_input else "producer",
              producer_command("continue-export" if args.campaign_input else "producer", names[0]))
        verify_artifact("before-census")
        census = serve["commands"]["census"]
        boundary = census.index("--")
        census[boundary:boundary] = ["--cidfile", str(args.out / f"{names[1]}.cid"),
                                     "--label", "prismaquant.pq183.campaign=" + nonce]
        phase("census", census)
        verify_artifact("after-census")
        phase("census-gate", serve["commands"]["census_gate"])
        verify_artifact("before-smoke")
        phase("smoke", serve["commands"]["smoke_pair"])
        verify_artifact("after-smoke")
        phase("check", producer_command("check", names[-1]))
        require(all(telemetry_ok.values()), "missing both-host Netdata evidence")
        require(not monitor_errors, "container/host telemetry inspection failed")
        state["status"] = "observed"
    except BaseException as exc:
        state["error"] = repr(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stop.set()
        if thread is not None:
            thread.join(timeout=45)
        state["cleanup"] = {}
        for name in names:
            cleanup = {"safe": False}
            try:
                record = inspect_name(name)
                cidfile = args.out / f"{name}.cid"
                cid = (record["Id"] if record else
                       state["observed_containers"].get(name, {}).get("id"))
                if cidfile.exists():
                    recorded_cid = cidfile.read_text().strip()
                    require(re.fullmatch(r"[0-9a-f]{64}", recorded_cid), "invalid Docker cidfile")
                    require(cid is None or cid == recorded_cid, "cidfile differs from observed container")
                    cid = recorded_cid
                cleanup = {"safe": True, "never_created": True}
                if cid is not None:
                    exact = subprocess.run(["docker", "inspect", cid], capture_output=True,
                                           text=True, timeout=10)
                    if exact.returncode:
                        require(any(f"no such {kind}: {cid}" in (exact.stdout + exact.stderr).lower()
                                    for kind in ("object", "container")), "cannot verify exact container absence")
                        cleanup = {"safe": True, "already_absent": True}
                    else:
                        obj = json.loads(exact.stdout)[0]
                        require(obj["Id"] == cid and obj["Name"] == "/" + name
                                and (obj.get("Config", {}).get("Labels") or {}).get("prismabuild.action") == owner,
                                "exact container ownership differs during cleanup")
                        cleanup = cleanup_container(cid)
                cleanup["container_id"] = cid
                if state["status"] == "observed":
                    require(cid is not None, "completed phase lacks exact container ID evidence")
            except Exception as exc:
                cleanup = {"safe": False, "error": repr(exc)}
            state["cleanup"][name] = cleanup
        if not all(item["safe"] for item in state["cleanup"].values()):
            state["status"] = "inconclusive"
        state["telemetry_success"] = telemetry_ok
        state["monitor_errors"] = monitor_errors
        state["finished_unix"] = time.time()
        dump(args.out / "host-status.json", state)
        print(json.dumps({"status": state["status"], "out": str(args.out),
                          "cleanup_safe": all(item["safe"] for item in state["cleanup"].values())}), flush=True)
    return 0 if state["status"] == "observed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("host", "producer", "producer-preflight", "campaign",
                                           "build", "seal", "check", "commands", "seal-campaign", "continue-export"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tessera-repo", type=Path, required=True)
    parser.add_argument("--tessera-commit", required=True)
    parser.add_argument("--container-prefix", default="pq183-lfm-r1")
    parser.add_argument("--port", type=int, default=8196)
    parser.add_argument("--tessera-source-manifest", type=Path)
    parser.add_argument("--tessera-source-manifest-sha256")
    parser.add_argument("--producer-image", help="immutable local Docker ID; Config and RootFS are verified")
    parser.add_argument("--campaign-input", type=Path, help="opt-in continuation of an unchanged completed campaign")
    parser.add_argument("--campaign-input-manifest", type=Path)
    parser.add_argument("--campaign-input-manifest-sha256")
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--netdata-url", action="append", default=[])
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    require(args.tessera_commit == TESSERA_COMMIT, f"requires pinned Tessera {TESSERA_COMMIT}")
    require(args.model.is_absolute() and args.out.is_absolute() and args.tessera_repo.is_absolute(),
            "model, output and producer paths must be absolute")
    require(str(args.out).startswith("/home/rob/tessera-runs/"), "use task-owned local output")
    require(all(c.isalnum() or c in "_-" for c in args.container_prefix), "invalid container prefix")
    if args.stage != "host":
        args.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.tessera_repo / "src"), str(args.tessera_repo)]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(REPO), str(args.tessera_repo / "src"), str(args.tessera_repo),
         os.environ.get("PYTHONPATH", "")])
    os.environ["TESSERA_REPO"] = str(args.tessera_repo)
    os.environ["TESSERA_COMMIT"] = args.tessera_commit
    return globals()[args.stage.replace("-", "_")](args)


if __name__ == "__main__":
    raise SystemExit(main())
