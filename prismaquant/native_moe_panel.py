"""Whole routed-stack preparation and native evidence, without serving imports.

A stack measurement binds every original source/PWC/wire member and the actual
captured routed invocation. It never adds leaf medians or creates a new group
cost. Native loading, fused execution and resource collection belong to the
separate Tessera producer; fixed/full-model resources remain unknown here.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import re

from .joint_aura import identity_sha256, validate_joint_aura_entry
from .measured_runtime_prices import RuntimeBinding
from .native_operator_panel import PHASES, _bytes, _equal, _number, _sha

INPUT_SCHEMA = "prismaquant.native_moe_inputs.v1"
PANEL_SCHEMA = "tessera.native_moe_panel.v1"
FORMAT = "TESSERA_E4M3_K1_R1024"
ROLES = ("w1", "w3", "w2")
EXECUTION = {"owner_kind": "complete_routed_moe", "mode": "resident",
             "execution_mode": "eager", "tensor_parallel": 1, "expert_parallel": 1,
             "include_router": False, "topk_selection": "external", "shared_experts": False,
             "monolithic": False, "bias": False}
ROUTING_FIELDS = {"activation", "scoring_func", "renormalize", "routed_scaling_factor",
                  "apply_router_weight_on_input", "expert_map", "input_dtype",
                  "topk_weights_dtype", "topk_ids_dtype", "device", "weights_contract", "source_protocol"}


def validate_geometry(shape):
    if not isinstance(shape, dict) or set(shape) != {"experts", "hidden_size", "intermediate_size", "top_k"}:
        raise ValueError("native MoE requires complete explicit stack geometry")
    if any(type(value) is not int or value < 1 for value in shape.values()):
        raise ValueError("native MoE geometry must be positive integers")
    if shape["experts"] != 32 or not 1 <= shape["top_k"] <= shape["experts"]:
        raise ValueError("native MoE panel supports one complete 32-expert stack")
    return shape


def validate_routing(routing):
    if not isinstance(routing, dict) or set(routing) != ROUTING_FIELDS:
        raise ValueError("native MoE requires exact captured routing settings")
    if (routing["activation"] != "silu" or routing["scoring_func"] != "sigmoid"
            or type(routing["renormalize"]) is not bool
            or routing["apply_router_weight_on_input"] is not False
            or routing["expert_map"] is not None
            or routing["input_dtype"] != "torch.bfloat16"
            or routing["topk_weights_dtype"] not in ("torch.bfloat16", "torch.float32")
            or routing["topk_ids_dtype"] not in ("torch.int32", "torch.int64")
            or routing["device"] != "cuda:0"
            or routing["weights_contract"] != "post_renormalization_and_routed_scaling"
            or type(routing["routed_scaling_factor"]) not in (int, float)
            or routing["routed_scaling_factor"] != 1):
        raise ValueError("native MoE reference requires the captured sigmoid/SiLU output-weighted unit-scale route")
    source = routing["source_protocol"]
    if (not isinstance(source, dict) or set(source) != {"router_class", "router_source_sha256",
            "selection_bias", "normalization_epsilon", "expert_bias_affects"}
            or not isinstance(source["router_class"], str) or not source["router_class"]
            or source["normalization_epsilon"] != 1e-6
            or source["expert_bias_affects"] != "selection_only"):
        raise ValueError("native MoE requires the actual source router protocol")
    _sha(source["router_source_sha256"], "router source")
    if source["selection_bias"] is not None:
        _sha(source["selection_bias"]["content_sha256"], "source selection bias")
    # These are already the weights passed INTO the actual experts forward.
    # No normalization or scaling is applied to them a second time here.
    # BF16 source normalization does not imply an exact sum of one.
    return routing


def _member_roster(unit, members, shape):
    validate_geometry(shape)
    if not isinstance(unit, str) or re.fullmatch(r"model\.layers\.[0-9]+\.feed_forward\.experts", unit) is None:
        raise ValueError("native MoE panel needs an exact LFM routed stack name")
    expected = [(expert, role, f"{unit}.{expert}.{role}")
                for expert in range(shape["experts"]) for role in ROLES]
    if not isinstance(members, list) or len(members) != len(expected):
        raise ValueError("native MoE panel requires all 96 explicitly ordered source members")
    for member, (expert, role, name) in zip(members, expected):
        if (type(member["expert"]) is not int or member["expert"] != expert
                or member["role"] != role or member["unit"] != name or member["format"] != FORMAT):
            raise ValueError("native MoE expert-role member ordering/format differs")
        geometry = ([shape["hidden_size"], shape["intermediate_size"]] if role == "w2"
                    else [shape["intermediate_size"], shape["hidden_size"]])
        _equal(member["shape"], geometry, f"{name} shape")
    return members


def _calibration_and_capture(calibration, capture, *, unit, shape, routing):
    if calibration.get("schema") != "prismaquant.calibration_input.v1":
        raise ValueError("native MoE needs the exact calibration receipt")
    _sha(calibration["calibration_sha256"], "calibration")
    if capture.get("schema") != "prismaquant.routed_boundary_capture.v1":
        raise ValueError("native MoE needs an actual routed-boundary capture")
    for key, expected in (("unit", unit), ("shape", shape), ("routing", routing),
                          ("calibration_sha256", calibration["calibration_sha256"]),
                          ("calibration_shape", calibration["shape"]),
                          ("calibration_dtype", calibration["dtype"])):
        _equal(capture[key], expected, f"capture {key}")
    from .tessera_expert_projection import _require_source_identity
    source = _require_source_identity(capture["producer_source"])
    _sha(source["config_sha256"], "capture source config")
    for digest in (*source["files"].values(), *source["auxiliary_sha256"].values()):
        _sha(digest, "capture source file")
    if not source["files"] or not source["tensors"] or not isinstance(capture["runtime_config"], dict):
        raise ValueError("native MoE capture needs its full producer source and actual runtime config")
    _sha(capture["capture_source_sha256"], "capture producer")
    from . import validate_pretrained_initialization_contract
    validate_pretrained_initialization_contract(capture["model_load_contract"])
    if capture["attention_implementation"] != "eager":
        raise ValueError("native MoE capture must record the actual eager source backend")
    runtime = capture["capture_runtime"]
    if (not isinstance(runtime, dict) or set(runtime) != {"torch", "cuda", "transformers"}
            or any(not isinstance(value, str) or not value for value in runtime.values())
            or runtime["transformers"] != capture["model_load_contract"]["transformers_version"]):
        raise ValueError("native MoE capture runtime does not match its canonical model load")


def _no_preclip():
    from .memory_management import env_truthy
    if env_truthy("PRISMAQUANT_PROD_ACT_SCALES", default=True):
        raise ValueError("native MoE protocol requires explicit PRISMAQUANT_PROD_ACT_SCALES=0")


def _validate_phase_tensors(x, ids, weights, shape, *, cuda):
    import torch
    if not all(isinstance(value, torch.Tensor) for value in (x, ids, weights)):
        raise ValueError("native MoE requires actual input, top-k IDs and weights tensors")
    if (x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != shape["hidden_size"]
            or x.dtype != torch.bfloat16):
        raise ValueError("native MoE input must be a nonempty 2-D BF16 hidden-state batch")
    if (ids.shape != (x.shape[0], shape["top_k"]) or weights.shape != ids.shape
            or ids.dtype not in (torch.int32, torch.int64)
            or weights.dtype not in (torch.bfloat16, torch.float32)):
        raise ValueError("native MoE top-k requires captured integer IDs and BF16/FP32 weights at matching geometry")
    if len({value.device for value in (x, ids, weights)}) != 1 or (cuda and x.device.type != "cuda"):
        raise ValueError("native MoE routed invocation must be resident on one CUDA device")
    if (not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any()) or bool((ids < 0).any()) or bool((ids >= shape["experts"]).any())):
        raise ValueError("native MoE routed invocation has nonfinite values or invalid assignments")
    if shape["top_k"] > 1 and bool((ids.sort(dim=-1).values.diff(dim=-1) == 0).any()):
        raise ValueError("native MoE capture repeats an expert within a token's top-k")


def packed_reference(experts_module, x, ids, weights, gate_up_weight, down_weight, *, spec, cache):
    """Use the shared packed-expert operation at both existing QDQ boundaries.

    Packing below is transient preparation, not a second residency/cache path.
    The source module supplies its actual activation operation. Real preparation
    requires CUDA; this arithmetic helper can also exercise small CPU fixtures.
    """
    from .measure_quant_cost import _packed_experts_forward_with_weights
    from .perturbed_x_cache import _activation_qdq
    _no_preclip()
    return _packed_experts_forward_with_weights(
        experts_module, x, ids, weights, gate_up_weight, down_weight,
        input_quantize=lambda value: _activation_qdq(value, spec, cache.activation_max_abs or {}, None),
        intermediate_quantize=lambda value: _activation_qdq(value, spec, cache.activation_max_abs or {}, None))


def prepare_moe_inputs(cache, source_weights, phase_tensors, *, unit, members, shape, routing,
                       calibration_receipt, routing_capture, experts_module, profile,
                       wire_blobs, wire_records, encoding_identities, numerics,
                       max_resident_bytes, max_temporary_bytes, runtime_image, serving_config_sha256, probe_request):
    """Prepare one complete routed reference from existing resident PWC data.

    Member records explicitly declare the expert/role order. Source tensors and
    producer encoding identities are independent inputs; they must not be copied
    from an unverified wire record. The caller pins their source artifact bytes.
    """
    import torch
    from tessera.cached_unit import verify_cached_unit, tensor_identity as producer_tensor_identity
    from tessera.unit_artifact import read_unit_artifact
    from . import format_registry as fr
    from .joint_aura import activation_identity, prefetch_joint_cache
    from .production_weight_cache import ProductionWeightCache, _cb_cache_tensor_identity
    _no_preclip()
    _member_roster(unit, members, shape)
    validate_routing(routing)
    _calibration_and_capture(calibration_receipt, routing_capture, unit=unit, shape=shape, routing=routing)
    if set(numerics) != {"atol", "rtol"}:
        raise ValueError("native MoE needs predeclared numerical tolerances")
    for name, value in numerics.items():
        _number(value, name)
    if re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", runtime_image) is None:
        raise ValueError("native MoE requires an immutable image RepoDigest")
    _sha(serving_config_sha256, "serving configuration")
    if not isinstance(cache, ProductionWeightCache):
        raise TypeError("native MoE requires the actual ProductionWeightCache")
    if (profile.name() != "lfm2_moe" or type(experts_module).__name__ not in profile.packed_expert_module_class_names()
            or getattr(experts_module, "num_experts", None) != shape["experts"]):
        raise ValueError("native MoE reference requires the source profile's actual packed LFM experts module")
    names = [member["unit"] for member in members]
    for values in (source_weights, wire_blobs, wire_records, encoding_identities):
        if set(values) != set(names):
            raise ValueError("native MoE source/wire/encoding coverage must equal all declared members")
    if set(phase_tensors) != set(PHASES) or set(routing_capture["phases"]) != set(PHASES):
        raise ValueError("native MoE requires captured prefill and decode invocations")
    for phase in PHASES:
        observed = phase_tensors[phase]
        if set(observed) != {"input", "topk_ids", "topk_weights", "source_topk_ids", "source_topk_weights"}:
            raise ValueError("native MoE phase tensor set differs")
        _validate_phase_tensors(observed["input"], observed["topk_ids"], observed["topk_weights"], shape, cuda=True)
        for key in ("input", "topk_ids", "topk_weights"):
            tensor = observed[key]
            _equal(_cb_cache_tensor_identity(tensor), routing_capture["phases"][phase][key], f"captured {phase} {key}")
            _equal(str(tensor.dtype), routing[{"input": "input_dtype", "topk_ids": "topk_ids_dtype",
                                              "topk_weights": "topk_weights_dtype"}[key]], f"{phase} supplied dtype")
            _equal(str(tensor.device), routing["device"], f"{phase} supplied device")
        validate_transport(observed, routing_capture["phases"][phase]["transport"])
    device = phase_tensors["prefill"]["input"].device
    if phase_tensors["decode"]["input"].device != device:
        raise ValueError("native MoE phases must share the resident reference device")
    packed_bytes = sum(source_weights[name].numel() * source_weights[name].element_size() for name in names)
    # One gate/up+down pack plus the largest transient decoded member. Input,
    # output and GEMM activations are covered separately by PB's action budget.
    required = packed_bytes + max(source_weights[name].numel() * source_weights[name].element_size() for name in names)
    if required > _bytes(max_temporary_bytes, "temporary packing budget"):
        raise ValueError("native MoE reference pack exceeds its explicit temporary budget")
    prefetch = prefetch_joint_cache(cache, names, {name: [FORMAT] for name in names},
                                   max_resident_bytes=max_resident_bytes)
    spec = fr.get_format(FORMAT)
    tensors, actual_members, rendered = {}, [], {}
    for member in members:
        name = member["unit"]
        source, render = source_weights[name], cache.get(name, FORMAT)
        if (source.device != device or render.device != device or source.dtype != torch.bfloat16
                or render.dtype != torch.bfloat16 or list(source.shape) != member["shape"]
                or render.shape != source.shape or not bool(torch.isfinite(source).all())
                or not bool(torch.isfinite(render).all())):
            raise ValueError(f"native MoE source/PWC tensor is not the declared resident BF16 member: {name}")
        _equal(encoding_identities[name]["source"], producer_tensor_identity(source), f"{name} encoder source")
        _equal(encoding_identities[name]["unit"], name, f"{name} encoder unit")
        verify_cached_unit(wire_blobs[name], wire_records[name], encoding_identities[name])
        decoded = read_unit_artifact(wire_blobs[name], device=str(device)).to(torch.bfloat16)
        _equal(_cb_cache_tensor_identity(decoded), _cb_cache_tensor_identity(render), f"{name} wire/PWC")
        del decoded
        activation = activation_identity(spec, cache.activation_max_abs or {}, name)
        if activation["clip_enabled"] or activation["input_global_scale"] is not None:
            raise ValueError("native MoE reference requires the shared dynamic E4M3 activation contract")
        actual_members.append({**member, "source_weight": _cb_cache_tensor_identity(source),
            "rendered_weight": _cb_cache_tensor_identity(render), "activation": activation,
            "wire": {"blob_sha256": hashlib.sha256(wire_blobs[name]).hexdigest(),
                     "blob_bytes": len(wire_blobs[name]), "record": wire_records[name]}})
        tensors["source_weight/" + name], tensors["rendered_weight/" + name] = source, render
        rendered[name] = render
    # Reuse the format/profile's declared gate/up roles. This pack is discarded
    # before the producer's native preparation and never persisted as a cache.
    with torch.inference_mode():
        gate_up = torch.empty(shape["experts"], 2 * shape["intermediate_size"], shape["hidden_size"],
                              dtype=torch.bfloat16, device=device)
        for expert in range(shape["experts"]):
            for role_index, role in enumerate(ROLES[:2]):
                gate_up[expert, role_index * shape["intermediate_size"]:(role_index + 1) * shape["intermediate_size"]].copy_(
                    rendered[f"{unit}.{expert}.{role}"])
        down = torch.stack([rendered[f"{unit}.{expert}.w2"] for expert in range(shape["experts"])])
        phases = {}
        from .perturbed_x_cache import _activation_qdq
        for phase in PHASES:
            values = {key: phase_tensors[phase][key] for key in ("input", "topk_ids", "topk_weights")}
            output = packed_reference(experts_module, values["input"], values["topk_ids"], values["topk_weights"],
                                      gate_up, down, spec=spec, cache=cache)
            qdq = _activation_qdq(values["input"], spec, cache.activation_max_abs or {}, None)
            for key, value in {**values, "reference_qdq": qdq, "reference_output": output}.items():
                tensors[f"{phase}.{key}"] = value
            phases[phase] = {"m": values["input"].shape[0], "transport": routing_capture["phases"][phase]["transport"], **{key: _cb_cache_tensor_identity(tensors[f"{phase}.{key}"])
                for key in ("input", "topk_ids", "topk_weights", "reference_qdq", "reference_output")}}
    del gate_up, down
    reference_file = Path(inspect.getfile(type(experts_module)))
    return {"schema": INPUT_SCHEMA, "unit": unit, "format": FORMAT, "shape": dict(shape),
            "members": actual_members, "profile_role_order": list(ROLES), "routing": dict(routing), "execution": dict(EXECUTION),
            "calibration": calibration_receipt, "routing_capture": routing_capture,
            "routing_capture_sha256": identity_sha256(routing_capture),
            "numerics": dict(numerics), "runtime_image": runtime_image, "serving_config_sha256": serving_config_sha256, "probe_request": probe_request,
            "reference": {"operation": "prismaquant.measure_quant_cost._packed_experts_forward_with_weights",
                          "module_class": f"{type(experts_module).__module__}.{type(experts_module).__qualname__}",
                          "module_source_sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest(),
                          "profile": profile.name(), "temporary_pack_bytes": packed_bytes,
                          "activation_preclip": False},
            "prefetch": prefetch, "phases": phases}, tensors


def validate_transport(values, transport):
    """Check actual raw-to-supplied tensors before their original store is left.

    Source IDs and BF16 router weights are retained by the shared capture. A
    runtime-friendly integer cast or FP32 promotion must preserve every value;
    normalization, reordering and rounded narrowing are not transport.
    """
    import torch
    from .production_weight_cache import _cb_cache_tensor_identity
    if not isinstance(transport, dict) or set(transport) != {"topk_ids", "topk_weights"}:
        raise ValueError("native MoE needs both explicit routed transport records")
    for name in ("topk_ids", "topk_weights"):
        raw, supplied, record = values["source_" + name], values[name], transport[name]
        if (not isinstance(raw, torch.Tensor) or not isinstance(record, dict)
                or set(record) != {"source", "supplied", "operation"}):
            raise ValueError("native MoE routed transport lacks original tensors or exact records")
        _equal(_cb_cache_tensor_identity(raw), record["source"], f"raw {name}")
        _equal(_cb_cache_tensor_identity(supplied), record["supplied"], f"supplied {name}")
        expected_operation = "identity" if raw.dtype == supplied.dtype else "lossless_dtype_conversion"
        _equal(record["operation"], expected_operation, f"{name} transport operation")
        if raw.shape != supplied.shape or raw.device != supplied.device:
            raise ValueError("native MoE routed transport changes shape/device")
        if name == "topk_ids":
            if raw.dtype not in (torch.int32, torch.int64) or supplied.dtype not in (torch.int32, torch.int64):
                raise ValueError("native MoE routed IDs must stay integers")
        elif raw.dtype not in (torch.bfloat16, torch.float32) or supplied.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("native MoE routing weights must preserve BF16/FP32 values")
        if (not torch.equal(raw.to(supplied.dtype), supplied)
                or not torch.equal(supplied.to(raw.dtype), raw)):
            raise ValueError("native MoE routed transport changes actual IDs or weights")


def _transport_identity(phase):
    transport = phase["transport"]
    if not isinstance(transport, dict) or set(transport) != {"topk_ids", "topk_weights"}:
        raise ValueError("native MoE phase requires captured routing transport provenance")
    for name, record in transport.items():
        if not isinstance(record, dict) or set(record) != {"source", "supplied", "operation"}:
            raise ValueError("native MoE routed transport record is incomplete")
        _equal(record["supplied"], phase[name], f"{name} supplied capture identity")
        source, supplied = record["source"], record["supplied"]
        _sha(source["content_sha256"], f"{name} source tensor")
        _equal(source["shape"], supplied["shape"], f"{name} transport shape")
        expected = "identity" if source["dtype"] == supplied["dtype"] else "lossless_dtype_conversion"
        _equal(record["operation"], expected, f"{name} transport operation")
        if expected == "identity":
            _equal(source, supplied, f"{name} unchanged transport")


def _workspace_identity(workspace):
    if (workspace.get("schema") != "tessera.native_moe_workspace.v1"
            or workspace.get("owner") != "vllm.WorkspaceManager"
            or type(workspace.get("num_ubatches")) is not int or workspace["num_ubatches"] != 1
            or type(workspace.get("num_lanes")) is not int or workspace["num_lanes"] != 1
            or workspace.get("locked") is not True or not isinstance(workspace.get("slots"), list)):
        raise ValueError("native MoE workspace is not a frozen single-lane runtime allocation")
    _bytes(workspace["resident_bytes"], "workspace resident")
    seen = set()
    for slot in workspace["slots"]:
        index = slot["index"]
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("native MoE workspace slot identity repeats")
        seen.add(index)
        if "allocation" in slot:
            if set(slot) != {"index", "allocation"} or slot["allocation"] is not None:
                raise ValueError("native MoE empty workspace slot is not an unallocated slot")
            continue
        for name in ("storage_bytes", "logical_bytes", "storage_offset"):
            _bytes(slot[name], f"workspace {name}")
        if (slot["device"] != "cuda:0" or not isinstance(slot["dtype"], str)
                or not isinstance(slot["shape"], list) or not isinstance(slot["stride"], list)
                or len(slot["shape"]) != len(slot["stride"])
                or any(type(value) is not int or value < 0 for value in slot["shape"] + slot["stride"])):
            raise ValueError("native MoE workspace geometry differs from the supported device")
    return workspace


def _native_member_identity(member):
    return {**{key: member[key] for key in ("unit", "expert", "role", "format", "shape", "source_weight", "rendered_weight")},
            "wire_sha256": member["wire"]["blob_sha256"],
            "wire_record_sha256": identity_sha256(member["wire"]["record"])}


def freeze_moe_panel(inputs, preflight, cost_rows, *, cost_sha256):
    """Bind all aligned member rows to one actual whole-stack native operator.

    RuntimeBinding is the existing cost/runtime join. No local or simultaneous
    group cost is computed by this boundary, and no leaf timing is summed.
    """
    from .joint_aura import _validated_assignment
    _sha(cost_sha256, "cost payload")
    if inputs.get("schema") != INPUT_SCHEMA:
        raise ValueError("native MoE input schema unsupported")
    if (preflight.get("schema") != "tessera.native_moe_preflight.v1"
            or preflight.get("status") != "untimed_preparation"):
        raise ValueError("native MoE panel requires untimed producer preparation")
    members = _member_roster(inputs["unit"], inputs["members"], inputs["shape"])
    validate_routing(inputs["routing"])
    _equal(inputs["execution"], EXECUTION, "input execution")
    _equal(inputs["profile_role_order"], list(ROLES), "source profile role order")
    _calibration_and_capture(inputs["calibration"], inputs["routing_capture"], unit=inputs["unit"],
                             shape=inputs["shape"], routing=inputs["routing"])
    _equal(inputs["routing_capture_sha256"], identity_sha256(inputs["routing_capture"]), "routing capture digest")
    rows = _validated_assignment(cost_rows, "additive")  # validation only; no scalar summary
    if set(rows) != {member["unit"] for member in members}:
        raise ValueError("native MoE runtime binding must cover exactly all 96 members")
    first = rows[members[0]["unit"]]
    probe, request = first["probe_identity"], inputs["probe_request"]
    for key in ("n_probes", "seed_base", "token_scope", "temperature", "distribution", "normalization"):
        _equal(probe[key], request[key], f"predeclared probe {key}")
    _equal(probe["source_model"]["source"], request["source_model"], "probe source model")
    _equal({Path(item["path"]).name: item["sha256"] for item in probe["source_model"]["shards"]},
           request["source_shards"], "probe source checkpoint bytes")
    from .cost_streaming import canonical_streamed_model_semantic_config
    source = inputs["routing_capture"]["producer_source"]
    _equal(source["files"], request["source_shards"], "capture/probe full source files")
    _equal(source["config_sha256"], request["source_config_sha256"], "captured source config bytes")
    _equal(source["auxiliary_sha256"], request["source_auxiliary_sha256"], "captured source auxiliary files")
    _equal(source["tensors"], probe["source_model"]["checkpoint_weight_map"], "capture/probe original checkpoint tensor map")
    _equal(canonical_streamed_model_semantic_config(inputs["routing_capture"]["runtime_config"], where="captured config"),
           canonical_streamed_model_semantic_config(probe["source_model"]["config"], where="probe config"),
           "capture/probe runtime config")
    for field, key in (("calibration_sha256", "calibration_sha256"), ("calibration_shape", "shape"), ("calibration_dtype", "dtype")):
        _equal(probe[field], inputs["calibration"][key], f"joint {field}")
    for member in members:
        joint = rows[member["unit"]]["joint_operator_identity"]
        for key, expected in (("qname", member["unit"]), ("format", member["format"]),
                              ("source_weight", member["source_weight"]), ("rendered_weight", member["rendered_weight"]),
                              ("activation", member["activation"])):
            _equal(joint[key], expected, f"{member['unit']} joint {key}")
        if joint["activation"].get("clip_enabled") is not False or joint["activation"].get("input_global_scale") is not None:
            raise ValueError("native MoE refuses clipped or statically scaled member joint rows")
    operator = preflight["operator"]
    _equal(operator["members"], [_native_member_identity(member) for member in members], "native member roster")
    for key, expected in (("shape", inputs["shape"]), ("routing", inputs["routing"]),
                          ("profile_role_order", inputs["profile_role_order"]),
                          ("routing_capture_sha256", inputs["routing_capture_sha256"])):
        _equal(operator[key], expected, f"native {key}")
    _equal(preflight["runtime_sha256"], identity_sha256(preflight["runtime"]), "runtime digest")
    _equal(preflight["native_tensors_sha256"], identity_sha256(operator["native_tensors"]), "native tensors")
    _equal(preflight["scheme_sha256"], identity_sha256(operator["scheme"]), "native scheme")
    _equal(operator["config_sha256"], identity_sha256(operator["config"]), "native MoE config")
    _equal(preflight["runtime"]["execution"], EXECUTION, "native execution")
    _equal(preflight["runtime"]["image"], inputs["runtime_image"], "native image")
    _equal(operator["serving_config_sha256"], _sha(inputs["serving_config_sha256"], "serving configuration"), "native serving config")
    _workspace_identity(preflight["workspace"])
    _equal(preflight["workspace_sha256"], identity_sha256(preflight["workspace"]), "workspace digest")
    route = operator["declared_route"]
    for key, expected in (("kind", "moe"), ("policy", "TESSERA_FP8:resident"),
                          ("decoder", "torch_materialize_stock"), ("contract", "fp8_per_token_dynamic")):
        _equal(route[key], expected, f"native route {key}")
    if not isinstance(route["symbol"], str) or re.fullmatch(r"vllm\.fused_moe\.modular_kernel:.+", route["symbol"]) is None:
        raise ValueError("native MoE route must name the actual modular backend")
    phases = {}
    for phase in PHASES:
        expected = inputs["phases"][phase]
        _transport_identity(expected)
        _equal(operator["phases"][phase]["transport"], expected["transport"], f"native {phase} transport")
        for key in ("input", "topk_ids", "topk_weights"):
            _equal(expected[key], inputs["routing_capture"]["phases"][phase][key], f"{phase} capture {key}")
        phases[phase] = {**expected, "expected_route": route}
    binding = RuntimeBinding(
        {member["unit"]: member["format"] for member in members},
        {name: row["joint_operator_identity_sha256"] for name, row in rows.items()},
        {member["unit"]: tuple(member["shape"]) for member in members}, route["symbol"])
    return json.loads(json.dumps({"schema": PANEL_SCHEMA, "unit": inputs["unit"], "format": FORMAT,
        "shape": inputs["shape"], "members": members, "profile_role_order": list(ROLES),
        "routing": inputs["routing"], "routing_capture_sha256": inputs["routing_capture_sha256"],
        "source_sha256": probe["source_model"]["content_sha256"], "calibration_sha256": probe["calibration_sha256"],
        "cost_sha256": cost_sha256, "serving_config_sha256": inputs["serving_config_sha256"], "probe_identity_sha256": first["probe_identity_sha256"],
        "runtime_binding": binding.as_dict(), "execution": dict(EXECUTION), "runtime": preflight["runtime"],
        "native_tensors_sha256": preflight["native_tensors_sha256"], "scheme_sha256": preflight["scheme_sha256"],
        "config_sha256": operator["config_sha256"], "workspace": preflight["workspace"],
        "workspace_sha256": preflight["workspace_sha256"], "numerics": inputs["numerics"], "phases": phases}, allow_nan=False))


def captured_moe_boundary(module, args, kwargs, coordinates, *, unit, source_model,
                          calibration_receipt, producer_source, prefill_rows=512):
    """Return actual first-sequence routed tensors for the existing PAC writer.

    The generic packed collector supplies the original forward arguments and
    calibration coordinates. No router is recomputed here. Returning None on
    later samples lets the collector retain only this bounded native input.
    """
    import torch
    from . import pretrained_initialization_contract
    from .model_profiles import profile_from_model
    from .production_weight_cache import _cb_cache_tensor_identity
    from .tessera_expert_projection import _require_source_identity
    if (type(prefill_rows) is not int or prefill_rows < 1
            or calibration_receipt.get("schema") != "prismaquant.calibration_input.v1"
            or calibration_receipt["shape"][1] < prefill_rows):
        raise ValueError("native MoE boundary requires an exact calibration and bounded first sequence")
    if (not isinstance(coordinates, torch.Tensor) or coordinates.ndim != 2
            or coordinates.shape[1] != 2 or coordinates.dtype != torch.int64):
        raise ValueError("native MoE collector must provide original int64 sample/token coordinates")
    selected = (coordinates[:, 0] == 0) & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < prefill_rows)
    if not bool(selected.any()):
        return None
    positions = coordinates[selected]
    if (positions.shape[0] != prefill_rows
            or not torch.equal(positions[:, 1], torch.arange(prefill_rows, device=coordinates.device))):
        raise ValueError("native MoE first sequence must arrive completely and in original token order")
    model_load_contract = pretrained_initialization_contract(source_model)
    if getattr(source_model.config, "_attn_implementation", None) != "eager":
        raise ValueError("native MoE capture requires the actual eager source attention backend")
    profile = profile_from_model(source_model)
    if (profile.name() != "lfm2_moe" or type(module).__name__ not in profile.packed_expert_module_class_names()
            or source_model.get_submodule(unit) is not module):
        raise ValueError("native MoE capture target is not the source model's declared LFM experts")
    # Use the class's declared signature: a hook wrapper may expose *args, but
    # that is not permission to guess positional meanings in a generic collector.
    bound = inspect.signature(type(module).forward).bind(module, *args, **kwargs)
    required = ("hidden_states", "top_k_index", "top_k_weights")
    if any(name not in bound.arguments for name in required):
        raise ValueError("source packed forward does not expose the required routed arguments")
    x, ids, weights = (bound.arguments[name] for name in required)
    if any(not isinstance(value, torch.Tensor) or value.shape[0] != coordinates.shape[0]
           for value in (x, ids, weights)):
        raise ValueError("captured router tensors are not aligned with original calibration coordinates")
    # The outer calibration loop owns CPU coordinates. Move its exact selector
    # losslessly for CUDA indexing; retain the original coordinate values/store.
    selected_on_device = selected.to(device=x.device)
    x, ids, weights = (value[selected_on_device].contiguous() for value in (x, ids, weights))
    shape = {"experts": int(module.num_experts), "hidden_size": int(x.shape[1]),
             "intermediate_size": int(module.down_proj.shape[-1]), "top_k": int(ids.shape[1])}
    validate_geometry(shape)
    _validate_phase_tensors(x, ids, weights, shape, cuda=True)
    parent = source_model.get_submodule(unit.rsplit(".", 1)[0])
    router = parent.gate
    bias = getattr(parent, "expert_bias", None)
    if (type(router).__name__ != "Lfm2MoeTopKRouter" or bias is None or bias.dtype != torch.float32
            or list(bias.shape) != [shape["experts"]] or bias.device != x.device
            or not bool(torch.isfinite(bias).all())):
        raise ValueError("native MoE capture requires the actual FP32-biased LFM router")
    source = _require_source_identity(producer_source)
    routing = {"activation": "silu", "scoring_func": "sigmoid", "renormalize": router.norm_topk_prob,
        "routed_scaling_factor": router.routed_scaling_factor, "apply_router_weight_on_input": False,
        "expert_map": None, "input_dtype": str(x.dtype), "topk_weights_dtype": str(weights.dtype),
        "topk_ids_dtype": str(ids.dtype), "device": str(x.device),
        "weights_contract": "post_renormalization_and_routed_scaling",
        "source_protocol": {"router_class": f"{type(router).__module__}.{type(router).__qualname__}",
            "router_source_sha256": hashlib.sha256(Path(inspect.getfile(type(router))).read_bytes()).hexdigest(),
            "selection_bias": _cb_cache_tensor_identity(bias), "normalization_epsilon": 1e-6,
            "expert_bias_affects": "selection_only"}}
    validate_routing(routing)
    tensors = {"inputs": x, "top_k_index": ids, "top_k_weights": weights,
               "coordinates": positions.contiguous(), "expert_bias": bias}
    return {"tensors": tensors, "metadata": {
        "schema": "prismaquant.native_moe_raw_boundary.v1", "unit": unit, "shape": shape,
        "routing": routing, "profile_role_order": list(ROLES),
        "scope": "first calibration sequence; decode uses its first row, not autoregressive generation",
        "calibration_sha256": calibration_receipt["calibration_sha256"],
        "calibration_shape": calibration_receipt["shape"], "calibration_dtype": calibration_receipt["dtype"],
        "producer_source": source, "runtime_config": source_model.config.to_dict(),
        "model_load_contract": model_load_contract,
        "attention_implementation": source_model.config._attn_implementation,
        "capture_runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                            "transformers": __import__("transformers").__version__},
        "capture_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tensors": {name: _cb_cache_tensor_identity(value) for name, value in tensors.items()}}}


def consume_moe_receipt(path, *, expected_sha256, expected_panel, memory_trace_path=None):
    """Retain whole-stack evidence and distinct persistent runtime workspace.

    WorkspaceManager storage is shared persistent runtime state, not per-apply
    scratch or a complete measured model-fixed price. No cross-row composition
    rule for it is invented here; the observation remains inadmissible as a
    complete allocator runtime table.
    """
    from .native_operator_panel import (native_operator_measurement, native_operator_scratch,
                                        validate_native_numerics)
    raw = Path(path).read_bytes()
    _equal(hashlib.sha256(raw).hexdigest(), _sha(expected_sha256, "receipt"), "receipt file")
    receipt = json.loads(raw)
    if (receipt.get("schema") != "tessera.native_moe_operator_receipt.v1"
            or receipt.get("status") != "timing_admissible"):
        raise ValueError("native MoE receipt has no admitted whole-apply observation")
    if expected_panel.get("schema") != PANEL_SCHEMA:
        raise ValueError("native MoE receipt requires its independently frozen panel")
    _equal(receipt["panel"], expected_panel, "receipt panel")
    _equal(receipt["panel_sha256"], identity_sha256(expected_panel), "receipt panel digest")
    _equal(receipt["runtime"], expected_panel["runtime"], "receipt runtime")
    _equal(receipt["runtime_sha256"], identity_sha256(expected_panel["runtime"]), "receipt runtime digest")
    _equal(receipt["workspace"], expected_panel["workspace"], "receipt workspace")
    _equal(receipt["workspace_sha256"], expected_panel["workspace_sha256"], "receipt workspace digest")
    _workspace_identity(receipt["workspace"])
    binding = RuntimeBinding.from_dict(expected_panel["runtime_binding"])
    members = _member_roster(expected_panel["unit"], expected_panel["members"], expected_panel["shape"])
    _equal(dict(binding.member_formats), {member["unit"]: member["format"] for member in members}, "runtime member coverage")
    operator = receipt["operator"]
    _equal(operator["members"], [_native_member_identity(member) for member in members], "receipt native members")
    for key in ("shape", "routing", "profile_role_order", "routing_capture_sha256", "serving_config_sha256"):
        _equal(operator[key], expected_panel[key], f"receipt {key}")
    _equal(identity_sha256(operator["native_tensors"]), expected_panel["native_tensors_sha256"], "receipt native tensors")
    _equal(identity_sha256(operator["scheme"]), expected_panel["scheme_sha256"], "receipt scheme")
    _equal(identity_sha256(operator["config"]), expected_panel["config_sha256"], "receipt MoE config")
    _equal(operator["config_sha256"], expected_panel["config_sha256"], "receipt MoE config digest")
    resources = receipt["resources"]
    complete = resources.get("status") == "complete_operator_bound"
    workspace_bytes = _bytes(resources["workspace_resident_bytes"], "runtime workspace resident")
    _equal(workspace_bytes, expected_panel["workspace"]["resident_bytes"], "workspace accounting")
    _equal(resources["workspace_sha256"], expected_panel["workspace_sha256"], "resource workspace")
    if complete:
        if memory_trace_path is None:
            raise ValueError("complete native MoE resource bound requires its actual memory trace")
        trace = json.loads(Path(memory_trace_path).read_text())
        _equal(identity_sha256(trace), resources["trace_sha256"], "memory trace")
        _equal(trace["capture"]["collector_library_sha256"],
               expected_panel["runtime"]["resource_collector"]["library_sha256"], "resource collector")
    observations = {}
    for phase in PHASES:
        observed, expected = receipt["phases"][phase], expected_panel["phases"][phase]
        for name in ("input", "topk_ids", "topk_weights", "reference_qdq", "reference_output", "transport"):
            _equal(observed[name], expected[name], f"{phase} {name}")
        route = observed["route"]
        _equal({key: route[key] for key in expected["expected_route"]}, expected["expected_route"], f"{phase} route")
        geometry = expected_panel["shape"]
        shape = f"M{expected['m']}:N{2 * geometry['intermediate_size']}:K{geometry['hidden_size']}"
        if route.get("state") != "served" or route.get("reason") is not None or route.get("shape") != shape:
            raise ValueError(f"{phase}: native MoE route state/shape differs")
        for kind in ("numerics", "qdq_numerics"):
            validate_native_numerics(observed[kind], expected_panel["numerics"], phase=phase, kind=kind)
        measurement = native_operator_measurement(observed["measurement"], path=path, expected_sha256=expected_sha256)
        bound = resources["phases"][phase].get("bound")
        scratch = native_operator_scratch(bound, phase=phase) if complete else None
        observations[phase] = {"measurement": measurement.as_dict(), "median_ms": measurement.median_ms,
                              "peak_scratch_bytes": scratch, "resource_bound": bound,
                              "input_bytes": expected["input"]["logical_bytes"],
                              "output_bytes": expected["reference_output"]["logical_bytes"]}
    return {"schema": "prismaquant.native_moe_observation.v1", "status": "operator_evidence",
            "unit": expected_panel["unit"], "format": FORMAT, "runtime_binding": binding.as_dict(),
            "panel_sha256": identity_sha256(expected_panel), "receipt_sha256": expected_sha256,
            "cost_sha256": expected_panel["cost_sha256"], "phases": observations,
            "serialized_unit_bytes": sum(_bytes(member["wire"]["blob_bytes"], "member wire") for member in members),
            "resident_bytes": _bytes(resources["resident_bytes"], "native layer resident"),
            "workspace_resident_bytes": workspace_bytes, "workspace_sha256": expected_panel["workspace_sha256"],
            "full_model_resources": None, "runtime_table_admissible": False,
            "unknown": ["fixed_and_full_model_resources", "cross_operator_workspace_composition"]
                       + ([] if complete else ["native_operator_scratch"])}
