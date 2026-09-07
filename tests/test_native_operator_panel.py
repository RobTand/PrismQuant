"""Boundary fixtures for frozen inputs, real joint currency and unknown resources.

Synthetic receipt records below test intake only; they are never measurements.
The actual producer/GPU qualification is a separate recorded action.
"""
import copy
import hashlib
import json

import pytest
import torch

from prismaquant.joint_aura import arithmetic_identity, identity_sha256, make_joint_aura_entry
from prismaquant.native_operator_panel import (EXECUTION, INPUT_SCHEMA, consume_native_receipt,
                                               freeze_native_panel)
from prismaquant.production_weight_cache import _cb_cache_tensor_identity
from test_streamed_cost_checkpoints import _model_identity


@pytest.fixture
def joined():
    unit, fmt = "fixture.dense", "TESSERA_BF16_K1_R1792"
    weight = _cb_cache_tensor_identity(torch.arange(16, dtype=torch.bfloat16).reshape(4, 4))
    activation = {"schema": "prismaquant.joint_aura.activation.v1", "quantizes_input": False,
                  "activation_max_abs": None, "input_global_scale": None, "clip_enabled": False}
    arithmetic = arithmetic_identity(torch.bfloat16)
    probe = {"schema": "prismaquant.joint_aura.probes.v1", "source_model": _model_identity("panel-fixture"),
             "calibration_sha256": "1" * 64, "calibration_shape": [1, 4], "calibration_dtype": "torch.int64",
             "producer_source_sha256": "2" * 64, "n_probes": 3, "seed_base": 7,
             "token_scope": "causal",
             "distribution": "rademacher", "normalization": "global_kl_fisher", "temperature": 1.0,
             "arithmetic": arithmetic}
    joint = {"schema": "prismaquant.joint_aura.operator.v1", "qname": unit, "format": fmt,
             "source_weight": weight, "rendered_weight": weight, "activation": activation,
             "arithmetic": arithmetic, "probe_identity_sha256": identity_sha256(probe)}
    row = make_joint_aura_entry(operator_identity=joint, probe_identity=probe, signed_components=[
        {"weight": v, "activation": 0., "mixed": 0., "total": v} for v in (.1, -.2, .3)])
    tensor = _cb_cache_tensor_identity(torch.ones(1, 4, dtype=torch.bfloat16))
    wire = {"blob_sha256": "3" * 64, "blob_bytes": 42, "record": {"fixture": "wire-record"}}
    route = {"kind": "dense", "policy": "TESSERA_BF16:resident", "symbol": "torch.mm",
             "decoder": "fixture-window", "contract": "bf16_unquantized"}
    inputs = {"schema": INPUT_SCHEMA, "unit": unit, "format": fmt, "shape": [4, 4],
              "source_weight": weight, "rendered_weight": weight, "activation": activation,
              "calibration": {"calibration_sha256": "1" * 64, "shape": [1, 4], "dtype": "torch.int64"},
              "wire": wire, "numerics": {"atol": .015625, "rtol": .015625},
              "runtime_image": "fixture/image@sha256:" + "9" * 64,
              "probe_request": {**{key: probe[key] for key in ("n_probes", "seed_base", "token_scope",
                                  "temperature", "distribution", "normalization")},
                                "source_model": "panel-fixture", "source_shards": {
                                    "panel-fixture.safetensors": probe["source_model"]["shards"][0]["sha256"]}},
              "phases": {p: {"m": 1, "input": tensor, "reference_qdq": tensor, "reference_output": tensor}
                         for p in ("prefill", "decode")}}
    native = {"source_weight": weight, "rendered_weight": weight, "input_global_scale": None,
              "clip_enabled": False, "wire_sha256": wire["blob_sha256"],
              "wire_record_sha256": identity_sha256(wire["record"]), "native_tensors": {"fixture": weight},
              "scheme": {"fixture": True}, "declared_route": route, "activation_contract": "bf16_unquantized"}
    runtime = {"schema": "tessera.native_dense_runtime.v1", "execution": dict(EXECUTION),
               "image": inputs["runtime_image"],
               "resource_collector": {"library_sha256": "5" * 64}}
    preflight = {"schema": "tessera.native_dense_preflight.v1", "status": "untimed_preparation",
                 "operator": native, "runtime": runtime, "runtime_sha256": identity_sha256(runtime),
                 "native_tensors_sha256": identity_sha256(native["native_tensors"]),
                 "scheme_sha256": identity_sha256(native["scheme"])}
    return inputs, preflight, row


def test_freeze_joins_joint_operator_and_predeclared_tolerance(joined):
    inputs, preflight, row = joined
    panel = freeze_native_panel(*joined, cost_sha256="4" * 64)
    assert panel["joint_operator_identity_sha256"] == row["joint_operator_identity_sha256"]
    assert panel["numerics"] == inputs["numerics"]
    assert panel["phases"]["decode"]["expected_route"] == preflight["operator"]["declared_route"]


@pytest.mark.parametrize("coordinate", ["calibration", "rendered_weight", "wire", "scale", "runtime", "probe_seed", "source_bytes", "image"])
def test_self_consistent_native_facts_cannot_replace_independent_inputs(joined, coordinate):
    inputs, preflight, row = joined
    if coordinate == "calibration":
        inputs["calibration"]["calibration_sha256"] = "0" * 64
    elif coordinate == "rendered_weight":
        inputs["rendered_weight"] = dict(inputs["rendered_weight"], content_sha256="0" * 64)
    elif coordinate == "wire":
        preflight["operator"]["wire_sha256"] = "0" * 64
    elif coordinate == "scale":
        preflight["operator"]["input_global_scale"] = 1.0
    elif coordinate == "runtime":
        preflight["runtime"]["execution"]["tensor_parallel"] = 2
        preflight["runtime_sha256"] = identity_sha256(preflight["runtime"])
    elif coordinate == "probe_seed":
        inputs["probe_request"]["seed_base"] += 1
    elif coordinate == "source_bytes":
        inputs["probe_request"]["source_shards"]["panel-fixture.safetensors"] = "0" * 64
    else:
        preflight["runtime"]["image"] = "another/image@sha256:" + "8" * 64
        preflight["runtime_sha256"] = identity_sha256(preflight["runtime"])
    with pytest.raises(ValueError):
        freeze_native_panel(inputs, preflight, row, cost_sha256="4" * 64)


def test_legacy_cost_never_becomes_joint_currency(joined):
    inputs, preflight, _ = joined
    with pytest.raises(ValueError, match="actual joint"):
        freeze_native_panel(inputs, preflight, {"output_mse": .1}, cost_sha256="4" * 64)


def receipt_fixture(joined, complete=False):
    inputs, preflight, _ = joined
    panel = freeze_native_panel(*joined, cost_sha256="4" * 64)
    numerics = {"status": "passed", "finite": True, "max_normalized_error": .5, **inputs["numerics"]}
    phases = {p: {**inputs["phases"][p], "route": {**preflight["operator"]["declared_route"],
                                                "state": "served", "reason": None, "shape": "M1:N4:K4"},
                  "numerics": dict(numerics), "qdq_numerics": dict(numerics),
                  "measurement": {"method": "cuda_events", "sample_unit": "single_apply",
                                  "samples_ms": [3., 1., 2.], "warmup_iterations": 4}}
              for p in ("prefill", "decode")}
    trace = {"fixture": "trace", "capture": {"collector_library_sha256": "5" * 64}}
    bound = {"status": "complete_operator_bound", "composition": "sum_of_independent_peaks_including_output",
             "full_model_fixed_resources_complete": False, "peak_scratch_bytes": 128,
             "external_native_peak_bytes": 64, "torch_peak_increment_bytes": 64}
    receipt = {"schema": "tessera.native_dense_operator_receipt.v1", "status": "timing_admissible",
               "panel": panel, "panel_sha256": identity_sha256(panel), "operator": preflight["operator"],
               "runtime": preflight["runtime"], "runtime_sha256": preflight["runtime_sha256"], "phases": phases,
               "resources": {"status": "complete_operator_bound" if complete else "incomplete", "resident_bytes": 64,
                             "trace_sha256": identity_sha256(trace),
                             "phases": {p: {"bound": copy.deepcopy(bound)} if complete else {}
                                        for p in ("prefill", "decode")}}}
    return panel, receipt, trace


def write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("complete", [False, True])
def test_observation_keeps_unmeasured_full_model_resources_unknown(joined, tmp_path, complete):
    panel, receipt, trace = receipt_fixture(joined, complete)
    path = tmp_path / "receipt.json"
    digest = write(path, receipt)
    trace_path = tmp_path / "trace.json"
    write(trace_path, trace)
    observation = consume_native_receipt(path, expected_sha256=digest, expected_panel=panel,
                                         memory_trace_path=trace_path if complete else None)
    assert observation["full_model_resources"] is None
    assert observation["runtime_table_admissible"] is False
    assert observation["phases"]["prefill"]["median_ms"] == 2.
    assert observation["phases"]["prefill"]["peak_scratch_bytes"] == (128 if complete else None)


@pytest.mark.parametrize("mutation", ["tolerance", "panel", "native_unknown", "composed_bound", "receipt_hash", "trace"])
def test_receipt_rejects_altered_binding_or_unproved_resource_claim(joined, tmp_path, mutation):
    panel, receipt, trace = receipt_fixture(joined, complete=True)
    if mutation == "tolerance":
        receipt["phases"]["decode"]["numerics"]["atol"] *= 2
    elif mutation == "panel":
        receipt["panel"] = copy.deepcopy(panel)
        receipt["panel"]["cost_sha256"] = "0" * 64
        receipt["panel_sha256"] = identity_sha256(receipt["panel"])
    elif mutation == "native_unknown":
        receipt["resources"]["phases"]["decode"]["bound"]["external_native_peak_bytes"] = None
    elif mutation == "composed_bound":
        receipt["resources"]["phases"]["decode"]["bound"]["peak_scratch_bytes"] = 0
    elif mutation == "trace":
        trace["capture"]["collector_library_sha256"] = "0" * 64
        receipt["resources"]["trace_sha256"] = identity_sha256(trace)
    path, trace_path = tmp_path / "receipt.json", tmp_path / "trace.json"
    digest = write(path, receipt)
    write(trace_path, trace)
    with pytest.raises(ValueError):
        consume_native_receipt(path, expected_sha256="0" * 64 if mutation == "receipt_hash" else digest,
                               expected_panel=panel, memory_trace_path=trace_path)
