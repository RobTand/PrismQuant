#!/usr/bin/env python3
"""Deterministic artifact preparation/consumption for one real native panel.

Run GPU preparation through PrismaBuild. This uses the existing source/Hessian
capture and one ProductionWeightCache entry; Tessera's separate producer owns
native runtime preparation and measurement. No allocator prices are invented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import re


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"input artifact SHA256 mismatch: {path}")
    return Path(path)


def dump(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def prepare(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from tessera import cached_unit
    from prismaquant.calibration_data import load_calibration_input
    from prismaquant.native_operator_panel import EXECUTION, prepare_native_inputs
    from prismaquant.production_weight_cache import ProductionWeightCache
    from prismaquant.tessera_export_lane import hessian_capture_sha256
    from prismaquant.tessera_formats import parse_tessera_format_name
    from prismaquant.tessera_hessian import activation_source, encoder_kwargs
    from prismaquant.tessera_render import encode_tessera_unit, rung_accepts_hessian, tessera_wire_recipe

    plan = json.loads(checked(args.plan, args.plan_sha256).read_text())
    if plan["schema"] != "prismaquant.native_dense_preparation.v1":
        raise ValueError("unsupported native preparation plan")
    if re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", plan["runtime_image"]) is None:
        raise ValueError("native preparation needs a full immutable image RepoDigest reference")
    unit, fmt = plan["unit"], plan["format"]
    capture = json.loads(checked(plan["capture"], plan["capture_sha256"]).read_text())
    root = Path(plan["capture"]).parent
    identity = capture["identity"]
    tokens, calibration = load_calibration_input(capture["tokens_file"],
        expected_sha256=capture["tokens_file_sha256"], n_samples=identity["nsamples"], seqlen=identity["seqlen"])
    if calibration["provenance"] != identity:
        raise ValueError("token provenance differs from capture")
    del tokens
    if unit not in capture["captured_units"] or unit in capture.get("excluded_units", []):
        raise ValueError("native panel unit was not captured")
    checked(root / "activations.safetensors", capture["activation_sha256"])
    checked(plan["units"], capture["units_safetensors_sha256"])
    with safe_open(plan["units"], framework="pt", device="cpu") as source:
        weight = source.get_tensor("weight/" + unit).to("cuda")
    with safe_open(str(root / "activations.safetensors"), framework="pt", device="cpu") as source:
        rows = source.get_tensor("activation/" + unit).to(device="cuda", dtype=torch.bfloat16)
    hessians = torch.load(capture["hessian_capture"], map_location="cpu", mmap=True, weights_only=True)
    if hessian_capture_sha256(hessians["H"], hessians["provenance"]) != capture["hessian_capture_sha256"]:
        raise ValueError("captured Hessian values/provenance changed")
    # The input scale is reconstructed from the capture's actual calibrated G
    # by the SAME contract used by QDQ, not from a prefix maximum.
    from prismaquant import format_registry as fr
    spec = fr.get_format(fmt)
    amax = None
    if spec.static_activation_contract is not None:
        from safetensors.torch import load_file
        scales = load_file(capture["static_scales"])
        key = unit + ".input_global_scale"
        if key not in scales:
            raise ValueError("captured static activation scale missing")
        # The capture receipt carries the actual maxima separately. A reciprocal
        # convention cannot be inferred from a scale artifact alone.
        maxima = capture.get("activation_max_abs", {})
        if unit not in maxima:
            raise ValueError("capture must retain actual activation_max_abs for a static native route")
        amax = float(maxima[unit])
        actual_g = spec.static_activation_contract.require_input_global_scale(amax, qname=unit, consumer="native panel")
        if actual_g != float(scales[key].reshape(())):
            raise ValueError("captured scale differs from current activation contract")
    cache = ProductionWeightCache(weights={}, levers={"tessera_hessian_identity": identity},
                                  activation_max_abs={unit: amax} if amax is not None else {})
    family, rung = parse_tessera_format_name(fmt)
    recipe = tessera_wire_recipe(family, rung)
    uses_hessian = rung_accepts_hessian(fmt, recipe)
    activation = activation_source({unit: hessians["H"][unit]}, identity) if uses_hessian else None
    kwargs = (encoder_kwargs(activation, unit, weight.shape[1], weight.device, scale_plane=recipe.scale_plane)
              if activation is not None else None)
    args.out.mkdir(parents=True, exist_ok=False)
    # The numerical policy is persisted before any native preparation/output.
    dump(args.out / "preparation-plan.json", plan)
    with torch.inference_mode():
        rendered, blob = encode_tessera_unit(weight, fmt, activation_kwargs=kwargs,
            hessian_required=uses_hessian, recipe=recipe, verify=True)
    cache.weights[unit, fmt] = rendered
    encoding = cached_unit.encoding_input_identity(weight, unit, family.payload_grid(), int(rung), activation=activation)
    record = cached_unit.make_unit_record(blob, encoding, filename="weight.tessera")
    inputs, tensors = prepare_native_inputs(cache, weight, rows, unit=unit, format_name=fmt,
        calibration_receipt=calibration, wire_blob=blob, wire_record=record, encoding_identity=encoding,
        numerics=plan["numerics"], prefill_rows=plan["prefill_rows"], decode_rows=plan["decode_rows"],
        max_resident_bytes=plan["max_resident_bytes"])
    (args.out / "weight.tessera").write_bytes(blob)
    dump(args.out / "wire-record.json", record)
    # Clone aliases because A16 QDQ may deliberately return the original input.
    save_file({key: tensor.detach().cpu().contiguous().clone() for key, tensor in tensors.items()},
              str(args.out / "tensors.safetensors"))
    cache.weights[unit, fmt] = rendered.detach().cpu()
    with (args.out / "production.pkl").open("wb") as stream:
        pickle.dump(cache, stream)
    inputs["artifacts"] = {name: sha256(args.out / name) for name in
        ("weight.tessera", "wire-record.json", "tensors.safetensors", "production.pkl")}
    inputs["preparation_plan_sha256"] = args.plan_sha256
    inputs["preparation_plan_file"] = "preparation-plan.json"
    inputs["runtime_image"] = plan["runtime_image"]
    inputs["probe_request"] = plan["probe_request"]
    dump(args.out / "inputs.json", inputs)
    dump(args.out / "request.json", {"schema": "tessera.native_dense_request.v1", "unit": unit,
        "format": fmt, "wire_path": "weight.tessera", "wire_record_path": "wire-record.json",
        "tensors_path": "tensors.safetensors", "runtime_image": plan["runtime_image"],
        "input_global_scale": inputs["activation"]["input_global_scale"], "execution": dict(EXECUTION)})
    print(json.dumps({"status": "prepared", "unit": unit, "format": fmt, "shape": inputs["shape"],
        "inputs_sha256": sha256(args.out / "inputs.json"), "request_sha256": sha256(args.out / "request.json"),
        "prefetch": inputs["prefetch"], "hessian_applied": uses_hessian}, sort_keys=True))


def freeze(args):
    from prismaquant.native_operator_panel import freeze_native_panel
    inputs = json.loads(checked(args.inputs, args.inputs_sha256).read_text())
    preflight = json.loads(checked(args.preflight, args.preflight_sha256).read_text())
    with checked(args.cost, args.cost_sha256).open("rb") as stream:
        cost = pickle.load(stream)
    row = cost["costs"][inputs["unit"]][inputs["format"]]
    panel = freeze_native_panel(inputs, preflight, row, cost_sha256=args.cost_sha256)
    if args.out.exists():
        raise ValueError("refuse overwriting frozen native panel")
    dump(args.out, panel)
    print(json.dumps({"status": "frozen", "panel_sha256": sha256(args.out)}, sort_keys=True))


def probe(args):
    """Run the shared CLI on exact inputs after one resident/streamed check."""
    import gc
    import torch
    from transformers import AutoModelForCausalLM
    from prismaquant import aura_cost
    from prismaquant.calibration_data import load_calibration_input
    from prismaquant.joint_aura import validate_joint_aura_entry
    from prismaquant.model_profiles import detect_profile

    inputs = json.loads(checked(args.inputs, args.inputs_sha256).read_text())
    root = args.inputs.parent
    plan = json.loads(checked(root / inputs["preparation_plan_file"], inputs["preparation_plan_sha256"]).read_text())
    capture = json.loads(checked(plan["capture"], plan["capture_sha256"]).read_text())
    request = inputs["probe_request"]
    checked(root / "production.pkl", inputs["artifacts"]["production.pkl"])
    calibration, _ = load_calibration_input(capture["tokens_file"],
        expected_sha256=capture["tokens_file_sha256"], n_samples=capture["identity"]["nsamples"],
        seqlen=capture["identity"]["seqlen"])
    if args.out.exists():
        raise ValueError("refuse overwriting joint probe output")
    args.out.mkdir(parents=True)
    detect_profile(request["source_model"])
    # A new model architecture must agree with its ordinary HF forward before
    # its streamed cotangents can qualify this panel. This is one bounded
    # first-sequence check, not a model-wide parity or performance claim.
    reference_model = AutoModelForCausalLM.from_pretrained(request["source_model"],
        dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, attn_implementation="eager").to("cuda").eval()
    with torch.inference_mode():
        reference = reference_model(calibration[:1].cuda(), use_cache=False).logits.detach().cpu()
    del reference_model
    gc.collect()
    torch.cuda.empty_cache()
    actual_compute = aura_cost.compute_aura_cost_streamed

    def checked_compute(runner, calib, formats, **kwargs):
        runner.require_prefetched_residency = True
        with torch.inference_mode():
            observed = runner(calib[:1]).logits.detach()
        expected = reference.to(device=observed.device)
        delta = (observed.float() - expected.float()).abs()
        limit = plan["numerics"]["atol"] + plan["numerics"]["rtol"] * expected.float().abs()
        finite = bool(torch.isfinite(observed).all() and torch.isfinite(expected).all())
        passed = finite and bool((delta <= limit).all())
        dump(args.out / "streamed-forward-parity.json", {
            "status": "passed" if passed else "failed", "scope": "first calibration sequence only",
            "shape": list(reference.shape), "numerics": plan["numerics"],
            "max_abs_error": float(delta.max()) if finite else None,
            "max_normalized_error": float((delta / limit).max()) if finite else None,
            "require_prefetched_residency": True})
        if not passed:
            raise ValueError("streamed LFM forward differs from resident source reference")
        del observed, expected, delta, limit
        return actual_compute(runner, calib, formats, **kwargs)

    argv = ["--model", request["source_model"], "--dtype", "bfloat16", "--streaming", "--joint-activation",
            "--production-cache", str(root / "production.pkl"), "--require-production-cache",
            "--formats", inputs["format"], "--unit-filter", "^" + re.escape(inputs["unit"]) + "$",
            "--allow-packed-expert-omission", "--n-probes", str(request["n_probes"]),
            "--seed-base", str(request["seed_base"]), "--token-scope", request["token_scope"],
            "--temperature", str(request["temperature"]), "--min-free-gib", "2",
            "--calibration-input", capture["tokens_file"], "--calibration-input-sha256", capture["tokens_file_sha256"],
            "--n-calib-samples", str(capture["identity"]["nsamples"]),
            "--calib-seqlen", str(capture["identity"]["seqlen"]),
            "--checkpoint-dir", str(args.out / "checkpoints"),
            "--streaming-offload-dir", str(args.out / "streaming"), "--output", str(args.out / "joint.pkl")]
    dump(args.out / "probe-request.json", {"inputs_sha256": args.inputs_sha256, "argv": argv,
                                           "probe_request": request, "scope": "one dense unit; no model-wide cost coverage"})
    aura_cost.compute_aura_cost_streamed = checked_compute
    try:
        code = aura_cost.main(argv)
    finally:
        aura_cost.compute_aura_cost_streamed = actual_compute
    if code != 0:
        raise RuntimeError(f"shared joint probe failed with exit status {code}")
    with (args.out / "joint.pkl").open("rb") as stream:
        result = pickle.load(stream)
    row = result["costs"][inputs["unit"]][inputs["format"]]
    if not validate_joint_aura_entry(row):
        raise ValueError("shared probe did not return actual joint currency")
    print(json.dumps({"status": "measured_joint_row", "unit": inputs["unit"], "format": inputs["format"],
        "cost_sha256": sha256(args.out / "joint.pkl"), "predicted_dloss": row["predicted_dloss"],
        "predicted_dloss_stderr": row["predicted_dloss_stderr"], "probe_identity_sha256": row["probe_identity_sha256"],
        "joint_operator_identity_sha256": row["joint_operator_identity_sha256"]}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--plan", required=True, type=Path)
    prep.add_argument("--plan-sha256", required=True)
    prep.add_argument("--out", required=True, type=Path)
    probed = sub.add_parser("probe")
    probed.add_argument("--inputs", required=True, type=Path)
    probed.add_argument("--inputs-sha256", required=True)
    probed.add_argument("--out", required=True, type=Path)
    frozen = sub.add_parser("freeze")
    for name in ("inputs", "preflight", "cost"):
        frozen.add_argument("--" + name, required=True, type=Path)
        frozen.add_argument("--" + name + "-sha256", required=True)
    frozen.add_argument("--out", required=True, type=Path)
    consumed = sub.add_parser("consume")
    for name in ("receipt", "panel"):
        consumed.add_argument("--" + name, required=True, type=Path)
        consumed.add_argument("--" + name + "-sha256", required=True)
    consumed.add_argument("--memory-trace", type=Path)
    consumed.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "probe":
        probe(args)
    elif args.command == "freeze":
        freeze(args)
    else:
        from prismaquant.native_operator_panel import consume_native_receipt
        panel = json.loads(checked(args.panel, args.panel_sha256).read_text())
        result = consume_native_receipt(args.receipt, expected_sha256=args.receipt_sha256,
            expected_panel=panel, memory_trace_path=args.memory_trace)
        if args.out.exists():
            raise ValueError("refuse overwriting native observation")
        dump(args.out, result)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
