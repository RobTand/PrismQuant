"""Frozen PWC inputs and a narrow native operator receipt bridge.

No serving runtime is imported here. Tessera owns native preparation/execution;
PQ supplies the actual production render, shared activation QDQ and joint-cost
identity. An operator observation never invents fixed/full-model resources or
becomes a complete measured-runtime allocation table by itself.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .joint_aura import identity_sha256, validate_joint_aura_entry
from .measured_runtime_prices import OperatorMeasurement

INPUT_SCHEMA = "prismaquant.native_dense_inputs.v1"
PANEL_SCHEMA = "tessera.native_dense_panel.v1"
EXECUTION = {"owner_kind": "single_dense", "mode": "resident",
             "execution_mode": "eager", "tensor_parallel": 1, "bias": False}
PHASES = ("prefill", "decode")


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name}: lowercase SHA256 required")
    return value


def _equal(actual, expected, name):
    if identity_sha256(actual) != identity_sha256(expected):
        raise ValueError(f"native panel {name} differs from independently frozen input")


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}: finite nonnegative number required")
    return value


def _bytes(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: nonnegative integer bytes required")
    return value


def prepare_native_inputs(cache, source_weight, activation_rows, *, unit, format_name,
                          calibration_receipt, wire_blob, wire_record, encoding_identity,
                          numerics, prefill_rows, decode_rows, max_resident_bytes):
    """Prepare independent references from existing resident PWC/activation data.

    ``encoding_identity`` must be derived by the producer from the actual
    source/Hessian/calibration settings, not copied from the wire record. The
    caller pins that preparation input artifact before executing this function.
    Artifact transport and tensor hashing occur outside any timed native apply.
    """
    import torch
    from tessera.cached_unit import verify_cached_unit
    from tessera.unit_artifact import read_unit_artifact
    from . import format_registry as fr
    from .joint_aura import activation_identity, prefetch_joint_cache
    from .perturbed_x_cache import _activation_qdq
    from .production_weight_cache import ProductionWeightCache, _cb_cache_tensor_identity

    if not isinstance(cache, ProductionWeightCache):
        raise TypeError("native panel requires the actual ProductionWeightCache")
    for value, name in ((source_weight, "source"), (activation_rows, "activation rows")):
        if value.ndim != 2 or value.dtype != torch.bfloat16 or value.device.type != "cuda":
            raise ValueError(f"native panel {name} requires resident 2-D CUDA BF16")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"native panel {name} is nonfinite")
    if activation_rows.shape[1] != source_weight.shape[1]:
        raise ValueError("native panel activation/source width differs")
    for rows in (prefill_rows, decode_rows):
        if type(rows) is not int or rows < 1 or rows > activation_rows.shape[0]:
            raise ValueError("native panel phase rows exceed retained calibration activations")
    if set(numerics) != {"atol", "rtol"}:
        raise ValueError("native panel requires explicit predeclared atol and rtol")
    for key, value in numerics.items():
        _number(value, key)
    if calibration_receipt.get("schema") != "prismaquant.calibration_input.v1":
        raise ValueError("native panel requires an exact calibration input receipt")
    _sha(calibration_receipt["calibration_sha256"], "calibration")
    prefetch = prefetch_joint_cache(cache, [unit], {unit: [format_name]},
                                   max_resident_bytes=max_resident_bytes)
    rendered = cache.get(unit, format_name).to(device=source_weight.device)
    if rendered.shape != source_weight.shape or rendered.dtype != torch.bfloat16:
        raise ValueError("native panel PWC source/render dtype or shape differs")
    verify_cached_unit(wire_blob, wire_record, encoding_identity)
    decoded = read_unit_artifact(wire_blob, device=str(rendered.device)).to(rendered.dtype)
    _equal(_cb_cache_tensor_identity(decoded), _cb_cache_tensor_identity(rendered), "wire/PWC decode")
    del decoded
    spec = fr.get_format(format_name)
    activation = activation_identity(spec, cache.activation_max_abs or {}, unit)
    if activation["clip_enabled"]:
        raise ValueError("native operator does not implement PQ's optional activation preclip")
    tensors = {"source_weight": source_weight, "rendered_weight": rendered}
    phases = {}
    with torch.inference_mode():
        for phase, count in (("prefill", prefill_rows), ("decode", decode_rows)):
            x = activation_rows[:count].contiguous()
            qx = (_activation_qdq(x, spec, cache.activation_max_abs or {}, unit)
                  if spec.act_quant_changes_input else x)
            output = torch.nn.functional.linear(qx, rendered)
            for name, value in (("input", x), ("reference_qdq", qx), ("reference_output", output)):
                tensors[f"{phase}.{name}"] = value
            phases[phase] = {"m": count, **{name: _cb_cache_tensor_identity(tensors[f"{phase}.{name}"])
                for name in ("input", "reference_qdq", "reference_output")}}
    return {
        "schema": INPUT_SCHEMA, "unit": unit, "format": format_name,
        "shape": list(source_weight.shape), "source_weight": _cb_cache_tensor_identity(source_weight),
        "rendered_weight": _cb_cache_tensor_identity(rendered), "activation": activation,
        "calibration": calibration_receipt, "activation_rows": _cb_cache_tensor_identity(activation_rows),
        "wire": {"blob_sha256": hashlib.sha256(wire_blob).hexdigest(), "blob_bytes": len(wire_blob),
                 "record": wire_record}, "numerics": dict(numerics), "execution": dict(EXECUTION),
        "phases": phases, "prefetch": prefetch,
    }, tensors


def freeze_native_panel(inputs, preflight, cost_row, *, cost_sha256):
    """Join independently frozen PWC references to untimed native facts and cost.

    A legacy MSE row is refused; only actual validated joint AURA rows can bind
    this panel. Native facts are declarations for subsequent measurement, not
    substituted numerical references or evidence of successful execution.
    """
    _sha(cost_sha256, "cost payload")
    if inputs.get("schema") != INPUT_SCHEMA:
        raise ValueError("native panel input schema unsupported")
    if (preflight.get("schema") != "tessera.native_dense_preflight.v1"
            or preflight.get("status") != "untimed_preparation"):
        raise ValueError("native panel requires untimed producer preparation")
    if not validate_joint_aura_entry(cost_row):
        raise ValueError("native panel requires an actual joint AURA cost row")
    joint = cost_row["joint_operator_identity"]
    operator = preflight["operator"]
    probe = cost_row["probe_identity"]
    request = inputs["probe_request"]
    for key in ("n_probes", "seed_base", "token_scope", "temperature", "distribution", "normalization"):
        _equal(probe[key], request[key], f"predeclared probe {key}")
    _equal(probe["source_model"]["source"], request["source_model"], "probe source model")
    _equal({Path(item["path"]).name: item["sha256"] for item in probe["source_model"]["shards"]},
           request["source_shards"], "probe source checkpoint bytes")
    for key, expected in (("qname", inputs["unit"]), ("format", inputs["format"]),
                          ("source_weight", inputs["source_weight"]),
                          ("rendered_weight", inputs["rendered_weight"]),
                          ("activation", inputs["activation"])):
        _equal(joint[key], expected, f"joint {key}")
    _equal(probe["calibration_sha256"], inputs["calibration"]["calibration_sha256"], "joint calibration")
    _equal(probe["calibration_shape"], inputs["calibration"]["shape"], "joint calibration shape")
    _equal(probe["calibration_dtype"], inputs["calibration"]["dtype"], "joint calibration dtype")
    for key in ("source_weight", "rendered_weight"):
        _equal(operator[key], inputs[key], f"native {key}")
    _equal(operator["input_global_scale"], joint["activation"]["input_global_scale"], "native input scale")
    _equal(operator["clip_enabled"], False, "native clip")
    _equal(operator["wire_sha256"], inputs["wire"]["blob_sha256"], "native wire")
    _equal(operator["wire_record_sha256"], identity_sha256(inputs["wire"]["record"]), "native wire record")
    _equal(preflight["runtime_sha256"], identity_sha256(preflight["runtime"]), "runtime digest")
    _equal(preflight["native_tensors_sha256"], identity_sha256(operator["native_tensors"]), "native tensors")
    _equal(preflight["scheme_sha256"], identity_sha256(operator["scheme"]), "native scheme")
    _equal(preflight["runtime"]["execution"], EXECUTION, "native execution")
    _equal(preflight["runtime"]["image"], inputs["runtime_image"], "native image reference")
    route = operator["declared_route"]
    _equal(route["contract"], operator["activation_contract"], "activation route")
    panel = {
        "schema": PANEL_SCHEMA, "unit": inputs["unit"], "format": inputs["format"],
        "shape": inputs["shape"], "source_sha256": probe["source_model"]["content_sha256"],
        "calibration_sha256": probe["calibration_sha256"], "cost_sha256": cost_sha256,
        "probe_identity_sha256": cost_row["probe_identity_sha256"],
        "joint_operator_identity_sha256": cost_row["joint_operator_identity_sha256"],
        "joint_operator_identity": joint, "wire": inputs["wire"], "execution": dict(EXECUTION),
        "runtime": preflight["runtime"], "native_tensors_sha256": preflight["native_tensors_sha256"],
        "scheme_sha256": preflight["scheme_sha256"], "numerics": inputs["numerics"],
        "phases": {phase: {**inputs["phases"][phase], "expected_route": route} for phase in PHASES},
    }
    return json.loads(json.dumps(panel, allow_nan=False))


def consume_native_receipt(path, *, expected_sha256, expected_panel, memory_trace_path=None):
    """Validate an exact receipt and retain unknown full-model resource fields.

    Returns warmed operator evidence only. The existing runtime table still
    needs independently measured fixed work/KV resources and whole-unit/fused
    coverage; this bridge does not fabricate that table or its missing prices.
    """
    raw = Path(path).read_bytes()
    _equal(hashlib.sha256(raw).hexdigest(), _sha(expected_sha256, "receipt"), "receipt file")
    receipt = json.loads(raw)
    if receipt.get("schema") != "tessera.native_dense_operator_receipt.v1" or receipt.get("status") != "timing_admissible":
        raise ValueError("native receipt has no admitted numerical/timing observation")
    _equal(receipt["panel"], expected_panel, "receipt panel")
    _equal(receipt["panel_sha256"], identity_sha256(expected_panel), "receipt panel digest")
    _equal(receipt["runtime"], expected_panel["runtime"], "receipt runtime")
    _equal(receipt["runtime_sha256"], identity_sha256(expected_panel["runtime"]), "receipt runtime digest")
    operator = receipt["operator"]
    joint = expected_panel["joint_operator_identity"]
    for key in ("source_weight", "rendered_weight"):
        _equal(operator[key], joint[key], f"receipt {key}")
    _equal(operator["input_global_scale"], joint["activation"]["input_global_scale"], "receipt activation scale")
    _equal(operator["clip_enabled"], False, "receipt activation clip")
    _equal(operator["wire_sha256"], expected_panel["wire"]["blob_sha256"], "receipt wire")
    _equal(operator["wire_record_sha256"], identity_sha256(expected_panel["wire"]["record"]), "receipt wire record")
    _equal(identity_sha256(operator["native_tensors"]), expected_panel["native_tensors_sha256"], "receipt native tensors")
    _equal(identity_sha256(operator["scheme"]), expected_panel["scheme_sha256"], "receipt scheme")
    resources = receipt["resources"]
    complete = resources.get("status") == "complete_operator_bound"
    if complete:
        if memory_trace_path is None:
            raise ValueError("complete native resource bound requires its actual memory trace")
        trace = json.loads(Path(memory_trace_path).read_text())
        _equal(identity_sha256(trace), resources["trace_sha256"], "memory trace")
        _equal(trace["capture"]["collector_library_sha256"],
               expected_panel["runtime"]["resource_collector"]["library_sha256"], "resource collector")
    observations = {}
    for phase in PHASES:
        observed, expected = receipt["phases"][phase], expected_panel["phases"][phase]
        for name in ("input", "reference_qdq", "reference_output"):
            _equal(observed[name], expected[name], f"{phase} {name}")
        route = observed["route"]
        _equal({key: route[key] for key in expected["expected_route"]}, expected["expected_route"], f"{phase} route")
        if (route.get("state") != "served" or route.get("reason") is not None
                or route.get("shape") != f"M{expected['m']}:N{expected_panel['shape'][0]}:K{expected_panel['shape'][1]}"):
            raise ValueError(f"{phase}: native route state/shape differs")
        for kind in ("numerics", "qdq_numerics"):
            validate_native_numerics(observed[kind], expected_panel["numerics"], phase=phase, kind=kind)
        measurement = native_operator_measurement(observed["measurement"], path=path, expected_sha256=expected_sha256)
        bound = resources["phases"][phase].get("bound")
        scratch = None
        if complete:
            scratch = native_operator_scratch(bound, phase=phase)
        observations[phase] = {"measurement": measurement.as_dict(), "median_ms": measurement.median_ms,
                               "peak_scratch_bytes": scratch, "resource_bound": bound,
                               "input_bytes": expected["input"]["logical_bytes"],
                               "output_bytes": expected["reference_output"]["logical_bytes"]}
    return {"schema": "prismaquant.native_dense_observation.v1", "status": "operator_evidence",
            "panel_sha256": identity_sha256(expected_panel), "receipt_sha256": expected_sha256,
            "unit": expected_panel["unit"], "format": expected_panel["format"],
            "cost_sha256": expected_panel["cost_sha256"], "phases": observations,
            "serialized_unit_bytes": expected_panel["wire"]["blob_bytes"],
            "resident_bytes": _bytes(resources["resident_bytes"], "resident"),
            "full_model_resources": None, "runtime_table_admissible": False,
            "unknown": ["fixed_and_full_model_resources"] + ([] if complete else ["native_operator_scratch"])}


def validate_native_numerics(error, numerics, *, phase, kind):
    """The shared frozen numerical gate for dense and whole routed operators."""
    if error.get("status") != "passed" or error.get("finite") is not True:
        raise ValueError(f"{phase}: refused native {kind} comparison")
    _equal({key: error[key] for key in ("atol", "rtol")}, numerics, f"{phase} tolerance")
    if _number(error["max_normalized_error"], "normalized numerical error") > 1:
        raise ValueError(f"{phase}: numerical error exceeds frozen tolerance")


def native_operator_measurement(timing, *, path, expected_sha256):
    """Keep exact repeated whole-apply observations through the existing type."""
    if timing.get("sample_unit") != "single_apply":
        raise ValueError("native timing is not repeated individual operator invocations")
    return OperatorMeasurement.from_dict({key: timing[key] for key in
        ("method", "samples_ms", "warmup_iterations")} | {
            "receipt_path": str(path), "receipt_sha256": expected_sha256})


def native_operator_scratch(bound, *, phase):
    """Read the producer's conservative two-domain bound; retain its scope."""
    if (not isinstance(bound, dict) or bound.get("status") != "complete_operator_bound"
            or bound.get("full_model_fixed_resources_complete") is not False
            or bound.get("composition") != "sum_of_independent_peaks_including_output"):
        raise ValueError(f"{phase}: native resource scope/bound mismatch")
    native = _bytes(bound["external_native_peak_bytes"], "external native peak")
    torch_peak = _bytes(bound["torch_peak_increment_bytes"], "torch peak")
    scratch = _bytes(bound["peak_scratch_bytes"], "scratch")
    if scratch != native + torch_peak:
        raise ValueError(f"{phase}: resource peaks do not compose to the declared bound")
    return scratch
