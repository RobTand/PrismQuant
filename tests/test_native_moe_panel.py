"""Whole-stack boundary tests. These synthetic records are never measurements."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from prismaquant.joint_aura import arithmetic_identity, identity_sha256, make_joint_aura_entry
from prismaquant.native_moe_panel import (EXECUTION, FORMAT, INPUT_SCHEMA, ROLES, consume_moe_receipt,
    freeze_moe_panel, packed_reference, validate_routing, validate_transport, _validate_phase_tensors)
from prismaquant.production_weight_cache import _cb_cache_tensor_identity as tensor_id


def routing():
    return {"activation": "silu", "scoring_func": "sigmoid", "renormalize": True,
        "routed_scaling_factor": 1.0, "apply_router_weight_on_input": False, "expert_map": None,
        "input_dtype": "torch.bfloat16", "topk_weights_dtype": "torch.float32",
        "topk_ids_dtype": "torch.int32", "device": "cuda:0",
        "weights_contract": "post_renormalization_and_routed_scaling", "source_protocol": {
            "router_class": "fixture.Lfm2MoeTopKRouter", "router_source_sha256": "d" * 64,
            "selection_bias": tensor_id(torch.zeros(32)), "normalization_epsilon": 1e-6,
            "expert_bias_affects": "selection_only"}}


def phase_tensors():
    raw_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64)
    # BF16 source normalization need not sum exactly to one. Never repair it.
    raw_weights = torch.tensor([[.498046875, .5], [.75, .2490234375]], dtype=torch.bfloat16)
    values = {"input": torch.ones(2, 4, dtype=torch.bfloat16), "topk_ids": raw_ids.int(),
        "topk_weights": raw_weights.float(), "source_topk_ids": raw_ids, "source_topk_weights": raw_weights}
    transport = {name: {"source": tensor_id(values["source_" + name]), "supplied": tensor_id(values[name]),
                       "operation": "lossless_dtype_conversion"} for name in ("topk_ids", "topk_weights")}
    return values, transport


@pytest.fixture
def joined():
    unit = "model.layers.2.feed_forward.experts"
    shape = {"experts": 32, "hidden_size": 4, "intermediate_size": 3, "top_k": 2}
    members = []
    activation = {"schema": "prismaquant.joint_aura.activation.v1", "quantizes_input": True,
                  "activation_max_abs": None, "input_global_scale": None, "clip_enabled": False}
    for expert in range(32):
        for role in ROLES:
            dims = [4, 3] if role == "w2" else [3, 4]
            weight = tensor_id(torch.ones(dims, dtype=torch.bfloat16))
            members.append({"unit": f"{unit}.{expert}.{role}", "expert": expert, "role": role,
                "format": FORMAT, "shape": dims, "source_weight": weight, "rendered_weight": weight,
                "activation": copy.deepcopy(activation), "wire": {"blob_sha256": "3" * 64, "blob_bytes": 42,
                    "record": {"unit": f"{unit}.{expert}.{role}"}}})
    source = {"files": {"fixture.safetensors": "8" * 64}, "config_sha256": "9" * 64,
              "auxiliary_sha256": {"config.json": "9" * 64},
              "tensors": {member["unit"] + ".weight": "fixture.safetensors" for member in members}}
    config = {"model_type": "lfm2_moe", "fixture": True}
    value = {"config": config, "weight_map": {name: name for name in source["tensors"]},
             "checkpoint_weight_map": source["tensors"],
             "shards": [{"path": "/fixture/fixture.safetensors", "size": 1, "sha256": "8" * 64}]}
    model = {"schema": "prismaquant.streamed_model.identity.v1", "source": "/fixture", "resolved_commit": None,
             "content_sha256": identity_sha256(value), **value}
    arithmetic = arithmetic_identity(torch.bfloat16)
    probe = {"schema": "prismaquant.joint_aura.probes.v1", "source_model": model,
        "calibration_sha256": "1" * 64, "calibration_shape": [1, 2], "calibration_dtype": "torch.int64",
        "producer_source_sha256": "2" * 64, "n_probes": 3, "seed_base": 7, "token_scope": "causal",
        "distribution": "rademacher", "normalization": "global_kl_fisher", "temperature": 1.0,
        "arithmetic": arithmetic}
    rows = {}
    for member in members:
        joint = {"schema": "prismaquant.joint_aura.operator.v1", "qname": member["unit"], "format": FORMAT,
            **{key: member[key] for key in ("source_weight", "rendered_weight", "activation")},
            "arithmetic": arithmetic, "probe_identity_sha256": identity_sha256(probe)}
        rows[member["unit"]] = make_joint_aura_entry(operator_identity=joint, probe_identity=probe,
            signed_components=[{"weight": v, "activation": 0., "mixed": 0., "total": v} for v in (.1, -.2, .3)])
    values, transport = phase_tensors()
    tensor_fields = {key: tensor_id(value) for key, value in values.items() if not key.startswith("source_")}
    phases = {phase: {"m": 2, **tensor_fields, "reference_qdq": tensor_fields["input"],
                     "reference_output": tensor_fields["input"], "transport": copy.deepcopy(transport)}
              for phase in ("prefill", "decode")}
    calibration = {"schema": "prismaquant.calibration_input.v1", "calibration_sha256": "1" * 64,
                   "shape": [1, 2], "dtype": "torch.int64"}
    capture = {"schema": "prismaquant.routed_boundary_capture.v1", "unit": unit, "shape": shape,
        "routing": routing(), "calibration_sha256": "1" * 64, "calibration_shape": [1, 2],
        "calibration_dtype": "torch.int64", "producer_source": source, "runtime_config": config,
        "capture_source_sha256": "c" * 64, "phases": phases,
        "model_load_contract": {"schema": "prismaquant.pretrained_initialization.v1", "scope": "checkpoint_missing_state",
                                "status": "completed", "transformers_version": "fixture-transformers"},
        "attention_implementation": "eager", "capture_runtime": {"torch": "fixture-torch", "cuda": "fixture-cuda",
                                                                   "transformers": "fixture-transformers"}}
    inputs = {"schema": INPUT_SCHEMA, "unit": unit, "format": FORMAT, "shape": shape, "members": members,
        "profile_role_order": list(ROLES), "routing": routing(), "execution": dict(EXECUTION),
        "calibration": calibration, "routing_capture": capture, "routing_capture_sha256": identity_sha256(capture),
        "runtime_image": "fixture/image@sha256:" + "a" * 64, "serving_config_sha256": "b" * 64, "numerics": {"atol": .015625, "rtol": .015625},
        "phases": phases, "probe_request": {
            **{key: probe[key] for key in ("n_probes", "seed_base", "token_scope", "temperature", "distribution", "normalization")},
            "source_model": "/fixture", "source_shards": source["files"], "source_config_sha256": source["config_sha256"],
            "source_auxiliary_sha256": source["auxiliary_sha256"]}}
    native_members = [{**{key: member[key] for key in ("unit", "expert", "role", "format", "shape", "source_weight", "rendered_weight")},
        "wire_sha256": member["wire"]["blob_sha256"], "wire_record_sha256": identity_sha256(member["wire"]["record"])} for member in members]
    native = {"members": native_members, "shape": shape, "routing": routing(), "profile_role_order": list(ROLES),
        "routing_capture_sha256": inputs["routing_capture_sha256"], "serving_config_sha256": "b" * 64, "native_tensors": {"fixture": tensor_fields["input"]},
        "scheme": {"fixture": True}, "config": {"fixture": "actual MoE config"},
        "phases": {phase: {"transport": value["transport"]} for phase, value in phases.items()},
        "declared_route": {"kind": "moe", "policy": "TESSERA_FP8:resident",
            "symbol": "vllm.fused_moe.modular_kernel:fixture", "decoder": "torch_materialize_stock",
            "contract": "fp8_per_token_dynamic"}}
    native["config_sha256"] = identity_sha256(native["config"])
    runtime = {"schema": "tessera.native_moe_runtime.v1", "execution": dict(EXECUTION),
        "image": inputs["runtime_image"], "resource_collector": {"library_sha256": "5" * 64}}
    workspace = {"schema": "tessera.native_moe_workspace.v1", "owner": "vllm.WorkspaceManager",
        "num_ubatches": 1, "num_lanes": 1, "locked": True, "slots": [{"index": 0, "shape": [64],
            "dtype": "torch.uint8", "device": "cuda:0", "storage_bytes": 64, "logical_bytes": 64,
            "stride": [1], "storage_offset": 0}], "resident_bytes": 64}
    preflight = {"schema": "tessera.native_moe_preflight.v1", "status": "untimed_preparation", "operator": native,
        "runtime": runtime, "runtime_sha256": identity_sha256(runtime), "workspace": workspace,
        "workspace_sha256": identity_sha256(workspace), "native_tensors_sha256": identity_sha256(native["native_tensors"]),
        "scheme_sha256": identity_sha256(native["scheme"])}
    return inputs, preflight, rows


def test_complete_runtime_binding_has_96_members_and_no_new_group_cost(joined):
    panel = freeze_moe_panel(*joined, cost_sha256="4" * 64)
    assert len(panel["runtime_binding"]["member_operator_identity_sha256"]) == 96
    assert "predicted_dloss" not in panel and "group_cost" not in panel
    assert panel["profile_role_order"] == ["w1", "w3", "w2"]


@pytest.mark.parametrize("change", ["missing", "order", "format", "rows", "probe", "preclip", "wire", "source",
                                    "config", "route", "runtime", "capture", "transport", "workspace"])
def test_changed_or_partial_stack_cannot_freeze(joined, change):
    inputs, preflight, rows = joined
    first = inputs["members"][0]["unit"]
    if change == "missing":
        inputs["members"].pop()
    elif change == "order":
        inputs["members"][0], inputs["members"][1] = inputs["members"][1], inputs["members"][0]
    elif change == "format":
        inputs["members"][0]["format"] = "TESSERA_BF16_K1_R1792"
    elif change == "rows":
        rows.pop(first)
    elif change == "probe":
        inputs["probe_request"]["seed_base"] += 1
    elif change == "preclip":
        inputs["members"][0]["activation"]["clip_enabled"] = True
    elif change == "wire":
        preflight["operator"]["members"][0]["wire_sha256"] = "0" * 64
    elif change == "source":
        inputs["probe_request"]["source_shards"] = {"fixture.safetensors": "0" * 64}
    elif change == "config":
        inputs["routing_capture"]["runtime_config"] = {"model_type": "different"}
        inputs["routing_capture_sha256"] = identity_sha256(inputs["routing_capture"])
    elif change == "route":
        preflight["operator"]["declared_route"]["kind"] = "dense"
    elif change == "runtime":
        preflight["runtime"]["execution"]["expert_parallel"] = 2
        preflight["runtime_sha256"] = identity_sha256(preflight["runtime"])
    elif change == "capture":
        inputs["routing_capture_sha256"] = "0" * 64
    elif change == "transport":
        inputs["phases"]["prefill"]["transport"]["topk_weights"]["operation"] = "renormalized"
    else:
        preflight["workspace"]["locked"] = False
        preflight["workspace_sha256"] = identity_sha256(preflight["workspace"])
    with pytest.raises((ValueError, RuntimeError)):
        freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)


def test_lossless_transport_keeps_real_bf16_rounding():
    values, transport = phase_tensors()
    validate_transport(values, transport)
    _validate_phase_tensors(values["input"], values["topk_ids"], values["topk_weights"],
                           {"hidden_size": 4, "top_k": 2, "experts": 32}, cuda=False)
    assert not torch.equal(values["topk_weights"].sum(-1), torch.ones(2))


@pytest.mark.parametrize("change", ["renormalize", "overflow", "reorder", "source_hash"])
def test_transport_cannot_repair_or_replace_routing(change):
    values, transport = phase_tensors()
    if change == "renormalize":
        values["topk_weights"] /= values["topk_weights"].sum(-1, keepdim=True)
        transport["topk_weights"]["supplied"] = tensor_id(values["topk_weights"])
    elif change == "overflow":
        values["source_topk_ids"][0, 0] = 2**32
        transport["topk_ids"]["source"] = tensor_id(values["source_topk_ids"])
    elif change == "reorder":
        values["topk_ids"] = values["topk_ids"].flip(-1)
        transport["topk_ids"]["supplied"] = tensor_id(values["topk_ids"])
    else:
        transport["topk_weights"]["source"]["content_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_transport(values, transport)


def test_route_rejects_softmax_label_and_input_weighting():
    for key, value in (("scoring_func", "softmax"), ("apply_router_weight_on_input", True), ("routed_scaling_factor", 2.)):
        changed = routing() | {key: value}
        with pytest.raises(ValueError):
            validate_routing(changed)


def test_reference_preserves_external_weights_without_renormalizing(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_PROD_ACT_SCALES", "0")
    from prismaquant import format_registry
    from prismaquant.production_weight_cache import ProductionWeightCache
    values, _ = phase_tensors()
    experts = SimpleNamespace(num_experts=4, act_fn=torch.nn.functional.silu)
    gate_up = torch.arange(4 * 6 * 4, dtype=torch.float32).reshape(4, 6, 4).div(32).to(torch.bfloat16)
    down = torch.ones(4, 4, 3, dtype=torch.bfloat16)
    # Tessera E4M3's registry synthesis delegates its activation QDQ to this
    # existing owner. This reference test exercises that owner without encoding.
    spec, cache = format_registry.get_format("FP8_E4M3"), ProductionWeightCache(weights={}, levers={})
    left = packed_reference(experts, values["input"], values["topk_ids"], values["topk_weights"], gate_up, down, spec=spec, cache=cache)
    right = packed_reference(experts, values["input"], values["topk_ids"], values["topk_weights"] / 2, gate_up, down, spec=spec, cache=cache)
    assert torch.equal(left / 2, right)
    monkeypatch.setenv("PRISMAQUANT_PROD_ACT_SCALES", "1")
    with pytest.raises(ValueError, match="PRISMAQUANT_PROD_ACT_SCALES"):
        packed_reference(experts, values["input"], values["topk_ids"], values["topk_weights"], gate_up, down, spec=spec, cache=cache)


def receipt_fixture(joined, complete=True):
    inputs, preflight, _ = joined
    panel = freeze_moe_panel(*joined, cost_sha256="4" * 64)
    error = {"status": "passed", "finite": True, "max_normalized_error": .25, **panel["numerics"]}
    phases = {phase: {**inputs["phases"][phase], "numerics": dict(error), "qdq_numerics": dict(error),
        "route": {**preflight["operator"]["declared_route"], "state": "served", "reason": None, "shape": "M2:N6:K4"},
        "measurement": {"method": "cuda_events", "sample_unit": "single_apply", "samples_ms": [3., 1., 2.],
                        "warmup_iterations": 4}} for phase in ("prefill", "decode")}
    trace = {"fixture": "trace", "capture": {"collector_library_sha256": "5" * 64}}
    bound = {"status": "complete_operator_bound", "composition": "sum_of_independent_peaks_including_output",
             "full_model_fixed_resources_complete": False, "peak_scratch_bytes": 128,
             "external_native_peak_bytes": 64, "torch_peak_increment_bytes": 64}
    receipt = {"schema": "tessera.native_moe_operator_receipt.v1", "status": "timing_admissible",
        "panel": panel, "panel_sha256": identity_sha256(panel), "operator": preflight["operator"],
        "runtime": preflight["runtime"], "runtime_sha256": preflight["runtime_sha256"],
        "workspace": preflight["workspace"], "workspace_sha256": preflight["workspace_sha256"], "phases": phases,
        "resources": {"status": "complete_operator_bound" if complete else "incomplete", "resident_bytes": 100,
            "workspace_resident_bytes": 64, "workspace_sha256": preflight["workspace_sha256"],
            "trace_sha256": identity_sha256(trace), "phases": {
                phase: {"bound": copy.deepcopy(bound)} if complete else {} for phase in ("prefill", "decode")}}}
    return panel, receipt, trace


def write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("complete", [False, True])
def test_whole_apply_evidence_preserves_workspace_and_full_model_unknown(joined, tmp_path, complete):
    panel, receipt, trace = receipt_fixture(joined, complete)
    path, trace_path = tmp_path / "receipt.json", tmp_path / "trace.json"
    digest = write(path, receipt)
    write(trace_path, trace)
    observed = consume_moe_receipt(path, expected_sha256=digest, expected_panel=panel,
                                   memory_trace_path=trace_path if complete else None)
    assert observed["resident_bytes"] == 100
    assert observed["workspace_resident_bytes"] == 64
    assert observed["phases"]["prefill"]["median_ms"] == 2.
    assert observed["phases"]["prefill"]["peak_scratch_bytes"] == (128 if complete else None)
    assert observed["runtime_table_admissible"] is False and observed["full_model_resources"] is None
    assert "cross_operator_workspace_composition" in observed["unknown"]


@pytest.mark.parametrize("change", ["missing_member", "weights", "workspace", "workspace_bytes", "resource_trace",
                                    "scalar_leaf_sum", "tolerance", "scratch", "route_shape", "config", "raw_identity"])
def test_whole_receipt_cannot_replace_frozen_inputs_or_resources(joined, tmp_path, change):
    panel, receipt, trace = receipt_fixture(joined)
    if change == "missing_member":
        receipt["operator"]["members"].pop()
    elif change == "weights":
        receipt["phases"]["prefill"]["topk_weights"] = dict(receipt["phases"]["prefill"]["topk_weights"], content_sha256="0" * 64)
    elif change == "workspace":
        receipt["workspace"] = copy.deepcopy(receipt["workspace"])
        receipt["workspace"]["slots"][0]["shape"] = [32]
        receipt["workspace_sha256"] = identity_sha256(receipt["workspace"])
    elif change == "workspace_bytes":
        receipt["resources"]["workspace_resident_bytes"] = 0
    elif change == "resource_trace":
        trace["capture"]["collector_library_sha256"] = "0" * 64
        receipt["resources"]["trace_sha256"] = identity_sha256(trace)
    elif change == "scalar_leaf_sum":
        receipt["phases"]["decode"]["measurement"]["sample_unit"] = "sum_of_leaf_medians"
    elif change == "tolerance":
        receipt["phases"]["prefill"]["numerics"]["atol"] *= 2
    elif change == "scratch":
        receipt["resources"]["phases"]["decode"]["bound"]["peak_scratch_bytes"] = 64
    elif change == "route_shape":
        receipt["phases"]["decode"]["route"]["shape"] = "M2:N3:K4"
    elif change == "config":
        receipt["operator"]["config"] = {"different": "MoE config"}
        receipt["operator"]["config_sha256"] = identity_sha256(receipt["operator"]["config"])
    else:
        receipt["phases"]["decode"]["transport"] = copy.deepcopy(receipt["phases"]["decode"]["transport"])
        receipt["phases"]["decode"]["transport"]["topk_ids"]["source"]["content_sha256"] = "0" * 64
    path, trace_path = tmp_path / "receipt.json", tmp_path / "trace.json"
    digest = write(path, receipt)
    write(trace_path, trace)
    with pytest.raises((ValueError, RuntimeError)):
        consume_moe_receipt(path, expected_sha256=digest, expected_panel=panel, memory_trace_path=trace_path)


@pytest.mark.parametrize("change", ["missing_init", "noncanonical_init", "attention", "runtime_version"])
def test_quarantined_capture_cannot_qualify_native_panel(joined, change):
    inputs, preflight, rows = joined
    capture = inputs["routing_capture"]
    if change == "missing_init":
        capture["model_load_contract"] = None
    elif change == "noncanonical_init":
        capture["model_load_contract"]["status"] = "operator_asserted"
    elif change == "attention":
        capture["attention_implementation"] = "sdpa"
    else:
        capture["capture_runtime"]["transformers"] = "different-transformers"
    inputs["routing_capture_sha256"] = identity_sha256(capture)
    with pytest.raises(ValueError):
        freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)


@pytest.fixture
def raw_boundary(joined):
    inputs, _, _ = joined
    values, _ = phase_tensors()
    tensors = {"inputs": values["input"], "top_k_index": values["source_topk_ids"],
        "top_k_weights": values["source_topk_weights"], "coordinates": torch.tensor([[0, 0], [0, 1]]),
        "expert_bias": torch.zeros(32)}
    meta = {key: copy.deepcopy(value) for key, value in inputs["routing_capture"].items() if key != "phases"}
    meta.update(schema="prismaquant.native_moe_raw_boundary.v1", profile_role_order=list(ROLES),
                scope="first calibration sequence; decode uses its first row, not autoregressive generation")
    meta["routing"].update(topk_ids_dtype="torch.int64", topk_weights_dtype="torch.bfloat16")
    meta["tensors"] = {key: tensor_id(value) for key, value in tensors.items()}
    manifest = {"schema": "prismaquant.tessera_calibration_cache.v2", "status": "complete", "identity": {
        **{key: copy.deepcopy(meta[key]) for key in ("model_load_contract", "capture_runtime", "attention_implementation")},
        "source_files": {**meta["producer_source"]["files"], **meta["producer_source"]["auxiliary_sha256"]}}}
    return {"source": "routed_boundary_capture", "boundary_metadata": meta, **tensors}, inputs["calibration"], manifest


def test_raw_boundary_transport_preserves_original_bf16_weights(raw_boundary):
    from prismaquant.native_moe_panel import routed_boundary_inputs
    raw, calibration, manifest = raw_boundary
    capture, phases, bias = routed_boundary_inputs(raw, calibration_receipt=calibration,
                                                  capture_manifest=manifest, device="cpu")
    assert phases["decode"]["input"].shape == (1, 4)
    assert torch.equal(phases["prefill"]["topk_weights"], raw["top_k_weights"].float())
    assert float(phases["prefill"]["topk_weights"][0].sum()) != 1.0
    assert capture["phases"]["prefill"]["transport"]["topk_weights"]["source"] == tensor_id(raw["top_k_weights"])
    assert bias.dtype == torch.float32


@pytest.mark.parametrize("change", ["historical", "coordinates", "weights", "dtype", "bias", "initialization", "source"])
def test_raw_boundary_changes_cannot_enter_native_protocol(raw_boundary, change):
    from prismaquant.native_moe_panel import routed_boundary_inputs
    raw, calibration, manifest = raw_boundary
    if change == "historical":
        manifest["schema"] = "prismaquant.tessera_calibration_cache.v1"
    elif change == "coordinates":
        raw["coordinates"] = raw["coordinates"].flip(0)
        raw["boundary_metadata"]["tensors"]["coordinates"] = tensor_id(raw["coordinates"])
    elif change == "weights":
        raw["top_k_weights"][0, 0] = .5
    elif change == "dtype":
        raw["boundary_metadata"]["routing"]["topk_weights_dtype"] = "torch.float32"
    elif change == "bias":
        raw["expert_bias"] = raw["expert_bias"].bfloat16()
        raw["boundary_metadata"]["tensors"]["expert_bias"] = tensor_id(raw["expert_bias"])
    elif change == "initialization":
        manifest["identity"]["model_load_contract"]["status"] = "skipped"
    else:
        manifest["identity"]["source_files"]["fixture.safetensors"] = "0" * 64
    with pytest.raises(ValueError):
        routed_boundary_inputs(raw, calibration_receipt=calibration, capture_manifest=manifest, device="cpu")


def token_receipt(tokens):
    identity = tensor_id(tokens)
    return {"schema": "prismaquant.calibration_input.v1", "shape": identity["shape"],
            "dtype": identity["dtype"], "calibration_sha256": identity["content_sha256"]}


def test_subset_probe_verifies_actual_ids_and_keeps_full_parent_distinct():
    from prismaquant.native_moe_panel import verified_probe_subset, _probe_calibration
    parent = torch.tensor([[1, 2], [3, 4]])
    subset = parent[:1].clone()
    full_receipt, subset_receipt = token_receipt(parent), token_receipt(subset)
    scope = verified_probe_subset(parent, subset, parent_calibration=full_receipt, subset_calibration=subset_receipt)
    assert scope["parent_calibration_sha256"] != scope["subset_calibration_sha256"]
    assert _probe_calibration({"calibration": full_receipt, "probe_calibration": subset_receipt, "probe_scope": scope}) == subset_receipt
    with pytest.raises(ValueError, match="actual first calibration sequence"):
        verified_probe_subset(parent, parent[1:], parent_calibration=full_receipt, subset_calibration=token_receipt(parent[1:]))
    with pytest.raises(ValueError, match="actual subset calibration_sha256"):
        verified_probe_subset(parent, subset, parent_calibration=full_receipt, subset_calibration=subset_receipt | {"calibration_sha256": "0" * 64})


def test_subset_joint_panel_retains_both_calibration_scopes(joined):
    from prismaquant.native_moe_panel import verified_probe_subset
    inputs, preflight, old_rows = joined
    parent = torch.tensor([[1, 2], [3, 4]])
    inputs["calibration"] = token_receipt(parent)
    inputs["probe_calibration"] = token_receipt(parent[:1])
    inputs["probe_scope"] = verified_probe_subset(parent, parent[:1], parent_calibration=inputs["calibration"],
                                                  subset_calibration=inputs["probe_calibration"])
    for key, value in (("calibration_shape", [2, 2]), ("calibration_sha256", inputs["calibration"]["calibration_sha256"])):
        inputs["routing_capture"][key] = value
    inputs["routing_capture_sha256"] = identity_sha256(inputs["routing_capture"])
    preflight["operator"]["routing_capture_sha256"] = inputs["routing_capture_sha256"]
    rows = {}
    for name, row in old_rows.items():
        probe = copy.deepcopy(row["probe_identity"])
        probe.update(calibration_shape=[1, 2], calibration_sha256=inputs["probe_calibration"]["calibration_sha256"])
        operator = copy.deepcopy(row["joint_operator_identity"])
        operator["probe_identity_sha256"] = identity_sha256(probe)
        rows[name] = make_joint_aura_entry(operator_identity=operator, probe_identity=probe,
            signed_components=[{"weight": v, "activation": 0., "mixed": 0., "total": v} for v in (.1, -.2, .3)])
    panel = freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)
    assert panel["calibration_sha256"] == inputs["probe_calibration"]["calibration_sha256"]
    assert panel["probe_scope"]["parent_calibration_sha256"] == inputs["calibration"]["calibration_sha256"]
    assert panel["probe_scope"]["scope"] == "first_sequence_integration_screen"
    inputs["probe_scope"]["sample_indices"] = [1]
    with pytest.raises(ValueError, match="first-sequence screen"):
        freeze_moe_panel(inputs, preflight, rows, cost_sha256="4" * 64)
