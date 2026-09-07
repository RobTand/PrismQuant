#!/usr/bin/env python3
"""Prepare/freeze/consume one complete routed panel from existing PQ artifacts.

Execute preparation through PrismaBuild. This adapter neither renders new
weights nor operates the serving runtime. The caller supplies independently
hashed canonical capture, original wire/PWC artifacts and a versioned plan.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import pickle


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(path, digest):
    path = Path(path).resolve(strict=True)
    if sha256(path) != digest:
        raise ValueError(f"native MoE artifact changed: {path}")
    return path


def dump(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def prepare(args):
    import torch
    import transformers
    from safetensors.torch import save_file
    from tessera import cached_unit
    from prismaquant.calibration_data import load_calibration_input
    from prismaquant.native_moe_panel import (EXECUTION, FORMAT, ROLES, _equal, _member_roster,
        prepare_moe_inputs, routed_boundary_inputs, verified_probe_subset)
    from prismaquant.model_profiles import profile_from_model
    from prismaquant.tessera_calibration_cache import require_capture_contract, prefetch_capture
    from prismaquant.tessera_formats import parse_tessera_format_name
    from prismaquant.tessera_expert_projection import carried_units, source_unit_weight
    from prismaquant.tessera_hessian import activation_source

    plan = json.loads(checked(args.plan, args.plan_sha256).read_text())
    if plan.get("schema") != "prismaquant.native_moe_preparation.v1":
        raise ValueError("native MoE requires its explicit preparation plan")
    if args.out.exists():
        raise ValueError("refuse overwriting native MoE preparation")
    capture = require_capture_contract(plan["capture"], expected_sha256=plan["capture_sha256"])
    census = json.loads(checked(plan["census"], capture["identity"]["census_sha256"]).read_text())
    calibration = capture["identity"]["calibration"]
    tokens, receipt = load_calibration_input(plan["calibration_input"],
        expected_sha256=plan["calibration_input_sha256"], n_samples=calibration["nsamples"], seqlen=calibration["seqlen"])
    probe_calibration, probe_scope = None, None
    subset_keys = {"probe_calibration_input", "probe_calibration_input_sha256"}
    present = subset_keys & plan.keys()
    if present and present != subset_keys:
        raise ValueError("native MoE subset input and independent hash must be paired")
    if present:
        subset_tokens, probe_calibration = load_calibration_input(plan["probe_calibration_input"],
            expected_sha256=plan["probe_calibration_input_sha256"], n_samples=1, seqlen=calibration["seqlen"])
        probe_scope = verified_probe_subset(tokens, subset_tokens, parent_calibration=receipt,
                                            subset_calibration=probe_calibration)
        del subset_tokens
    del tokens
    for key in ("fit_ids_sha256", "text_sha256", "source", "seed", "nsamples", "seqlen"):
        _equal(receipt["provenance"][key], calibration[key], f"capture/token {key}")
    payload = torch.load(checked(plan["routing_boundary"], plan["routing_boundary_sha256"]),
                         map_location="cpu", weights_only=True)
    routed, phase_tensors, bias = routed_boundary_inputs(payload, calibration_receipt=receipt,
                                                        capture_manifest=capture, device="cuda:0")
    unit, shape = routed["unit"], routed["shape"]
    _equal(unit, plan["unit"], "planned routed unit")
    _equal(routed["producer_source"], census["expert_projection"]["producer"]["source"], "boundary/census source")
    source, projected_units, stack_of = carried_units(census["expert_projection"])
    _equal(source, routed["producer_source"], "validated projected source")
    model_root = Path(census["model"]).resolve(strict=True)
    request = plan["probe_request"]
    for key, expected in (("source_model", str(model_root)), ("source_shards", source["files"]),
                          ("source_config_sha256", source["config_sha256"]),
                          ("source_auxiliary_sha256", source["auxiliary_sha256"])):
        _equal(request[key], expected, f"planned probe {key}")
    for name, digest in {**source["files"], **source["auxiliary_sha256"], "config.json": source["config_sha256"]}.items():
        checked(model_root / name, digest)
    for key, value in (("torch", str(torch.__version__)), ("cuda", torch.version.cuda),
                       ("transformers", transformers.__version__)):
        _equal(routed["capture_runtime"][key], value, f"reference runtime {key}")
    members = [{"unit": f"{unit}.{expert}.{role}", "expert": expert, "role": role, "format": FORMAT,
                "shape": ([shape["hidden_size"], shape["intermediate_size"]] if role == "w2"
                          else [shape["intermediate_size"], shape["hidden_size"]])}
               for expert in range(shape["experts"]) for role in ROLES]
    _member_roster(unit, members, shape)
    names = [member["unit"] for member in members]
    if set(plan["wires"]) != set(names):
        raise ValueError("preparation plan must cover exactly all 96 original wires")
    with checked(plan["production_cache"], plan["production_cache_sha256"]).open("rb") as stream:
        cache = pickle.load(stream)
    weights = {}
    for member in members:
        name = member["unit"]
        _equal(stack_of[name], unit, f"{name} producer stack")
        projection = projected_units[name]
        _equal(projection["expert"], member["expert"], f"{name} producer expert")
        _equal([projection["rows"], projection["cols"]], member["shape"], f"{name} producer shape")
        weights[name] = source_unit_weight(model_root, source, projection).to("cuda:0")
    (_, hessians, counts, _), prefetched = prefetch_capture(plan["capture"], expected_sha256=plan["capture_sha256"],
        expected_identity=capture["identity"], census=census, names=names, device="cuda:0")
    activation = activation_source(hessians, calibration)
    family, rung = parse_tessera_format_name(FORMAT)
    # The ordinary routed campaign seals unit_input_identity, which includes
    # the producer projection. Re-derive that exact identity from the canonical
    # census and actual source/H rather than rewrapping the original record.
    encodings = {name: cached_unit.unit_input_identity(weights[name], projected_units[name], family.payload_grid(),
                                                      int(rung), activation=activation) for name in names}
    del hessians, activation
    blobs, records, wire_inputs = {}, {}, []
    for member in members:
        name, artifact = member["unit"], plan["wires"][member["unit"]]
        wire = checked(artifact["wire"], artifact["wire_sha256"])
        record = checked(artifact["record"], artifact["record_sha256"])
        blobs[name], records[name] = wire.read_bytes(), json.loads(record.read_text())
        wire_inputs.append({**{key: member[key] for key in ("unit", "expert", "role", "format")},
                            "wire_path": str(wire), "wire_record_path": str(record)})
    serving_config = checked(plan["serving_config"], plan["serving_config_sha256"])
    # Only the real source class's parameter-free activation behavior is used.
    # A meta skeleton avoids reloading the entire source model for that behavior.
    config = transformers.AutoConfig.from_pretrained(model_root, local_files_only=True, trust_remote_code=True)
    with torch.device("meta"):
        skeleton = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True,
                                                                 attn_implementation="eager")
    experts = skeleton.get_submodule(unit)
    _equal(sha256(inspect.getfile(type(experts))), routed["routing"]["source_protocol"]["router_source_sha256"],
           "source reference implementation")
    inputs, tensors = prepare_moe_inputs(cache, weights, phase_tensors, unit=unit, members=members,
        shape=shape, routing=routed["routing"], calibration_receipt=receipt, routing_capture=routed,
        experts_module=experts, profile=profile_from_model(skeleton), wire_blobs=blobs, wire_records=records,
        encoding_identities=encodings, numerics=plan["numerics"], max_resident_bytes=plan["max_resident_bytes"],
        max_temporary_bytes=plan["max_temporary_bytes"], runtime_image=plan["runtime_image"],
        serving_config_sha256=plan["serving_config_sha256"], probe_request=request,
        probe_calibration_receipt=probe_calibration, probe_scope=probe_scope)
    tensors["routing_bias"] = bias
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / "preparation-plan.json", plan)
    save_file({name: value.detach().cpu().contiguous().clone() for name, value in tensors.items()},
              str(args.out / "tensors.safetensors"))
    inputs["source_capture"] = {"manifest_sha256": plan["capture_sha256"],
        "routing_boundary_sha256": plan["routing_boundary_sha256"], "prefetch": prefetched, "counts": counts}
    inputs["preparation_plan_sha256"] = args.plan_sha256
    inputs["tensors_file_sha256"] = sha256(args.out / "tensors.safetensors")
    dump(args.out / "inputs.json", inputs)
    dump(args.out / "request.json", {"schema": "tessera.native_moe_request.v1", "unit": unit,
        "members": wire_inputs, "shape": shape, "routing": inputs["routing"], "execution": dict(EXECUTION),
        "runtime_image": plan["runtime_image"], "tensors_path": "tensors.safetensors", "profile_role_order": list(ROLES),
        "routing_capture_sha256": inputs["routing_capture_sha256"],
        "phases": {phase: {"transport": inputs["phases"][phase]["transport"]} for phase in ("prefill", "decode")},
        "serving_config_path": str(serving_config)})
    print(json.dumps({"status": "prepared", "inputs_sha256": sha256(args.out / "inputs.json"),
                      "request_sha256": sha256(args.out / "request.json"), "members": len(members)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--plan", required=True, type=Path)
    prep.add_argument("--plan-sha256", required=True)
    prep.add_argument("--out", required=True, type=Path)
    for command, names in (("freeze", ("inputs", "preflight", "cost")), ("consume", ("receipt", "panel"))):
        operation = sub.add_parser(command)
        for name in names:
            operation.add_argument("--" + name, required=True, type=Path)
            operation.add_argument("--" + name + "-sha256", required=True)
        operation.add_argument("--out", required=True, type=Path)
        if command == "consume":
            operation.add_argument("--memory-trace", type=Path)
        else:
            operation.add_argument("--source-execution-qualification", type=Path)
            operation.add_argument("--source-execution-qualification-sha256")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    if args.command == "freeze":
        from prismaquant.native_moe_panel import freeze_moe_panel
        inputs = json.loads(checked(args.inputs, args.inputs_sha256).read_text())
        preflight = json.loads(checked(args.preflight, args.preflight_sha256).read_text())
        cost_path = checked(args.cost, args.cost_sha256)
        if cost_path.suffix == ".json":
            costs = json.loads(cost_path.read_text())
        else:
            with cost_path.open("rb") as stream:
                costs = pickle.load(stream)
        rows = {member["unit"]: costs["costs"][member["unit"]][member["format"]] for member in inputs["members"]}
        result = freeze_moe_panel(inputs, preflight, rows, cost_sha256=args.cost_sha256,
            source_execution_qualification_path=args.source_execution_qualification,
            source_execution_qualification_sha256=args.source_execution_qualification_sha256)
    else:
        from prismaquant.native_moe_panel import consume_moe_receipt
        panel = json.loads(checked(args.panel, args.panel_sha256).read_text())
        result = consume_moe_receipt(args.receipt, expected_sha256=args.receipt_sha256,
                                    expected_panel=panel, memory_trace_path=args.memory_trace)
    dump(args.out, result)
    print(json.dumps({"status": args.command, "output_sha256": sha256(args.out)}))


if __name__ == "__main__":
    main()
