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
import json
import os
from pathlib import Path
import pickle
import re
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
IMAGE = "eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c"
PRODUCER_IMAGE = "sha256:337dae6b15313ff7a46aad56ec200119c6416555fd21c1085661f1c7cbd13b88"
TESSERA_COMMIT = "ba582d47"
STACK = "model.layers.12.feed_forward.experts"
EXPECTED_UNITS = {f"{STACK}.{expert}.{role}" for expert in range(32)
                  for role in ("w1", "w3", "w2")}
SCOPE = ["--tessera-platform", "sm_121", "--tessera-runtime-image", IMAGE,
         "--tessera-execution-mode", "eager", "--tessera-residency", "resident"]


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
    validate_topology(args)
    require(not (args.out / "cost.pkl").exists(), "campaign output exists; use a new attempt")
    run(args, [sys.executable, "-m", "prismaquant.tessera_campaign",
               "--model", args.model, "--out", args.out / "cost.pkl",
               "--cache-dir", args.out / "cache", "--menu-mode", "attested",
               "--anchors", 1, "--max-rounds", 1, "--anchor-budget", 1,
               "--nsamples", 8, "--seqlen", 512, "--seed", 0,
               "--max-act-rows", 512, "--layer-stride", 12,
               "--hessian", "require", "--tp-degree", 1, *SCOPE], "campaign")


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
            "producer projection must contain exactly 96 layer-12 expert units")
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
            "selection": "predeclared_measured_E4M3_q1024_layer12",
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


def commands(args):
    """Print reviewable host commands; execution remains coordinator-owned."""
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
              "-e", "TESSERA_SERVE_MODE=resident", "-v", "/mnt/shared:/mnt/shared:ro",
              "-v", f"{out / 'exported'}:{out / 'exported'}:ro",
              "-v", f"{out / 'census'}:/census", "--", census_body]
    gate = [sys.executable, str(ts / "experiments/ts5_census_check.py"),
            "--plan", str(out / "plan.json"), "--checkpoint", str(out / "exported"),
            "--census", str(out / "census/census.json"), "--runtime-image", IMAGE,
            "--require-attested", "--out", str(out / "census/check.json")]
    smoke = ["env", f"TS={ts}", f"IMAGE={IMAGE}", f"EXT={out / 'ext'}",
             f"VLLM_CACHE={out / 'vllm-cache'}", f"NAME_PREFIX={name}-smoke", f"PORT={args.port}",
             f"PY={sys.executable}", str(ts / "experiments/moe_greedy_smoke_pair.sh"),
             str(args.model), str(out / "exported"), str(out / "artifact-seal.json"), str(out / "smoke")]
    print(json.dumps({"schema": "prismaquant.pq183-served-commands.v1",
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
                      "tp_degree": 1}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("campaign", "build", "seal", "check", "commands"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tessera-repo", type=Path, required=True)
    parser.add_argument("--tessera-commit", required=True)
    parser.add_argument("--container-prefix", default="pq183-lfm-r1")
    parser.add_argument("--port", type=int, default=8196)
    args = parser.parse_args()
    require(args.tessera_commit.startswith(TESSERA_COMMIT), "requires pinned Tessera ba582d47")
    require(args.model.is_absolute() and args.out.is_absolute() and args.tessera_repo.is_absolute(),
            "model, output and producer paths must be absolute")
    require(str(args.out).startswith("/home/rob/tessera-runs/"), "use task-owned local output")
    require(all(c.isalnum() or c in "_-" for c in args.container_prefix), "invalid container prefix")
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.tessera_repo / "src"), str(args.tessera_repo)]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(REPO), str(args.tessera_repo / "src"), str(args.tessera_repo),
         os.environ.get("PYTHONPATH", "")])
    os.environ["TESSERA_REPO"] = str(args.tessera_repo)
    os.environ["TESSERA_COMMIT"] = args.tessera_commit
    globals()[args.stage](args)


if __name__ == "__main__":
    main()
