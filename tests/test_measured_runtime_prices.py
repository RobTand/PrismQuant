"""Synthetic contract tests; none of these fixture timings are GPU evidence."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from prismaquant.measured_runtime_prices import (
    CONTEXT_SCHEMA, SCHEMA, RuntimeBinding, RuntimePriceError, RuntimeResources,
    build_runtime_resources, load_measured_runtime_table, load_runtime_context,
    parse_measured_runtime_table, parse_runtime_context,
)

NOW = datetime(2026, 9, 5, tzinfo=timezone.utc)
SHA = "a" * 64


@pytest.fixture
def payload(tmp_path):
    receipt = tmp_path / "synthetic-receipt.txt"
    receipt.write_text("Synthetic CPU test fixture, NOT measured GPU evidence.\n")
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    context = {
        "schema": CONTEXT_SCHEMA,
        "serving_context": {"platform": "sm_121", "structure": "dense", "residency": "resident",
                            "runtime_image": "example.invalid/runtime@sha256:" + SHA,
                            "execution_mode": "eager"},
        "gpu_identity": "synthetic-gpu", "runtime_sha256": SHA, "source_sha256": SHA,
        "calibration_sha256": SHA, "prompt_tokens": 1024, "batch_size": 1,
        "tensor_parallel": 1, "graph_mode": "eager",
        "operator_routes": {"layer": {"W4A4": "synthetic-w4a4", "W4A16": "synthetic-w4a16"}},
    }
    measurement = {"method": "cuda_events", "samples_ms": [1.0, 2.0, 3.0],
                   "warmup_iterations": 3, "receipt_path": receipt.name, "receipt_sha256": digest}
    resources = dict(prefill_ms=2.0, decode_ms=None, serialized_bytes=100,
                     resident_bytes=200, peak_scratch_bytes=40, activation_bytes=80, kv_bytes=0)
    rows = []
    for fmt, time, identity in (("W4A4", 2.0, SHA), ("W4A16", 6.0, "b" * 64)):
        row = {"unit": "layer", "format": fmt,
               "binding": {"member_formats": {"layer": fmt},
                           "member_operator_identity_sha256": {"layer": identity},
                           "member_shapes": {"layer": [8, 16]},
                           "operator_route": context["operator_routes"]["layer"][fmt]},
               "resources": dict(resources, prefill_ms=time),
               "prefill": dict(measurement, samples_ms=[time - 1, time, time + 1]), "decode": None}
        rows.append(row)
    return {"schema": SCHEMA, "table_id": "synthetic-only", "status": "proposal_data",
            "composition": "sequential_operator_sum", "context": context, "cost_sha256": SHA,
            "measured_at": "2026-09-04T00:00:00Z", "valid_until": "2026-09-06T00:00:00Z",
            "fixed_assignment": {"lm_head": "BF16"},
            "fixed_resources": dict(resources, prefill_ms=0.0, serialized_bytes=800,
                                    resident_bytes=1600, activation_bytes=160, kv_bytes=400),
            "fixed_resources_receipt_path": receipt.name, "fixed_resources_receipt_sha256": digest,
            "rows": rows}


def parse(payload, **kwargs):
    return parse_measured_runtime_table(payload,
        expected_context=parse_runtime_context(payload["context"]), expected_cost_sha256=SHA,
        now=NOW, **kwargs)


def candidates():
    return {"layer": [SimpleNamespace(fmt=fmt, memory_bytes=100, member_formats=None)
                      for fmt in ("W4A16", "W4A4")]}


def test_same_bytes_different_activation_family_preserved(payload):
    table = parse(payload)
    bindings = {row.key: row.binding for row in table.rows}
    result = build_runtime_resources(table, candidates(), expected_bindings=bindings)
    assert list(result) == [("layer", "W4A16"), ("layer", "W4A4")]
    assert result["layer", "W4A4"].serialized_bytes == result["layer", "W4A16"].serialized_bytes == 100
    assert result["layer", "W4A4"].prefill_ms == 2
    assert result["layer", "W4A16"].prefill_ms == 6
    assert result["layer", "W4A4"].resident_bytes == 200
    assert table.fixed_resources.kv_bytes == 400
    assert table.identity()["slo_eligible"] is False
    assert table.identity()["status"] == "proposal_data"


def test_roundtrip_order_and_immutability(payload):
    table = parse(payload)
    reordered = copy.deepcopy(payload)
    reordered["rows"].reverse()
    assert parse(reordered).identity() == table.identity()
    assert parse(table.as_dict()) == table
    with pytest.raises(TypeError):
        table.context.operator_routes["layer"]["W4A4"] = "changed"
    with pytest.raises(TypeError):
        table.rows[0].binding.member_formats["layer"] = "BF16"


@pytest.mark.parametrize("field,value", [
    ("runtime_sha256", "b" * 64), ("source_sha256", "b" * 64),
    ("calibration_sha256", "b" * 64), ("gpu_identity", "different-gpu"),
    ("prompt_tokens", 2048), ("batch_size", 2), ("tensor_parallel", 2),
    ("graph_mode", "full"),
])
def test_context_mismatch(payload, field, value):
    expected = parse_runtime_context(payload["context"])
    payload["context"][field] = value
    with pytest.raises(RuntimePriceError, match="context mismatch"):
        parse_measured_runtime_table(payload, expected_context=expected, expected_cost_sha256=SHA, now=NOW)


@pytest.mark.parametrize("field,value", [("measured_at", "2026-09-05T01:00:00Z"),
                                         ("valid_until", "2026-09-05T00:00:00Z"),
                                         ("cost_sha256", "b" * 64)])
def test_stale_or_future_refused(payload, field, value):
    payload[field] = value
    with pytest.raises(RuntimePriceError, match="stale|future"):
        parse(payload)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(status="served_p95_certified"),
    lambda p: p.update(composition="parameter_weighted"),
    lambda p: p["rows"].append(copy.deepcopy(p["rows"][0])),
    lambda p: p["rows"][0]["prefill"].update(method="encoder_seconds"),
    lambda p: p["rows"][0]["prefill"].update(samples_ms=[1, 2]),
    lambda p: p["rows"][0]["prefill"].update(samples_ms=[1, float("nan"), 3]),
    lambda p: p["rows"][0]["prefill"].update(samples_ms=[1, True, 3]),
    lambda p: p["rows"][0]["prefill"].update(warmup_iterations=0),
    lambda p: p["rows"][0]["resources"].update(prefill_ms=99),
    lambda p: p["rows"][0]["resources"].update(decode_ms=1),
    lambda p: p["rows"][0]["resources"].update(resident_bytes=1.5),
    lambda p: p["rows"][0]["resources"].update(kv_bytes=1),
    lambda p: p["rows"][0]["binding"].update(operator_route="unmeasured-route"),
    lambda p: p["rows"][0]["binding"]["member_shapes"].update(layer=[8, 0]),
    lambda p: p["rows"][0]["binding"]["member_operator_identity_sha256"].update(layer="missing"),
    lambda p: p["rows"][0].update(activation_bits_speed_hint=4),
    lambda p: p["context"].update(batch_size=True),
    lambda p: p.update(rows=[]),
])
def test_malformed_or_proxy_prices_refused(payload, mutation):
    mutation(payload)
    with pytest.raises(RuntimePriceError):
        parse(payload)


def test_missing_binding_and_bytes_refused(payload):
    table = parse(payload)
    bindings = {row.key: row.binding for row in table.rows}
    with pytest.raises(RuntimePriceError, match="missing"):
        build_runtime_resources(table, candidates(), expected_bindings={})
    wrong_candidates = candidates()
    wrong_candidates["layer"][0].memory_bytes = 101
    with pytest.raises(RuntimePriceError, match="serialized byte"):
        build_runtime_resources(table, wrong_candidates, expected_bindings=bindings)
    wrong = bindings["layer", "W4A4"].as_dict()
    wrong["member_operator_identity_sha256"]["layer"] = "c" * 64
    bindings["layer", "W4A4"] = RuntimeBinding.from_dict(wrong)
    with pytest.raises(RuntimePriceError, match="binding mismatch"):
        build_runtime_resources(table, candidates(), expected_bindings=bindings)


def test_whole_group_cannot_be_priced_by_sum_of_leaf_rows(payload):
    table = parse(payload)
    group = {"fused": [SimpleNamespace(fmt="group-option", memory_bytes=200,
                                     member_formats={"layer": "W4A4", "other": "W4A16"})]}
    with pytest.raises(RuntimePriceError, match="missing"):
        build_runtime_resources(table, group, expected_bindings={})


def test_explicit_whole_group_row(payload):
    row = payload["rows"][0]
    row["unit"], row["format"] = "fused", "group-option"
    row["binding"]["member_formats"]["other"] = "W4A16"
    row["binding"]["member_operator_identity_sha256"]["other"] = "c" * 64
    row["binding"]["member_shapes"]["other"] = [8, 16]
    row["resources"]["serialized_bytes"] = 200
    payload["context"]["operator_routes"]["fused"] = {"group-option": row["binding"]["operator_route"]}
    table = parse(payload)
    group = {"fused": [SimpleNamespace(fmt="group-option", memory_bytes=200,
                                     member_formats=dict(row["binding"]["member_formats"]))]}
    result = build_runtime_resources(table, group, expected_bindings={r.key: r.binding for r in table.rows})
    assert result["fused", "group-option"].prefill_ms == 2


def test_loader_checks_raw_receipt_content_and_duplicate_json(payload, tmp_path):
    path = tmp_path / "table.json"
    path.write_text(json.dumps(payload))
    context = parse_runtime_context(payload["context"])
    table = load_measured_runtime_table(path, expected_context=context, expected_cost_sha256=SHA, now=NOW)
    assert table.source_path == str(path)
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(payload["context"]))
    assert load_runtime_context(context_path) == context
    (tmp_path / "synthetic-receipt.txt").write_text("tampered")
    with pytest.raises(RuntimePriceError, match="receipt SHA-256 mismatch"):
        load_measured_runtime_table(path, expected_context=context, expected_cost_sha256=SHA, now=NOW)
    path.write_text('{"schema":1,"schema":2}')
    with pytest.raises(RuntimePriceError, match="duplicate JSON"):
        load_measured_runtime_table(path, expected_context=context, expected_cost_sha256=SHA, now=NOW)


def test_resources_do_not_equate_resident_with_serialized_bytes():
    resource = RuntimeResources(1, None, 100, 800, 64, 1024)
    assert resource.serialized_bytes == 100
    assert resource.resident_bytes == 800


def test_decode_samples_remain_separate_from_prefill(payload):
    row = payload["rows"][0]
    row["decode"] = dict(row["prefill"], samples_ms=[0.1, 0.2, 0.3])
    row["resources"]["decode_ms"] = 0.2
    table = parse(payload)
    priced = next(r for r in table.rows if r.fmt == "W4A4")
    assert priced.resources.prefill_ms == 2.0
    assert priced.resources.decode_ms == 0.2


def test_missing_row_is_not_replaced_by_another_family(payload):
    payload["rows"] = payload["rows"][:1]
    table = parse(payload)
    with pytest.raises(RuntimePriceError, match="missing measured runtime row"):
        build_runtime_resources(table, candidates(), expected_bindings={r.key: r.binding for r in table.rows})


@pytest.mark.parametrize("field,value", [("residency", "streamed"), ("execution_mode", "compiled"),
                                         ("runtime_image", "example.invalid/other@sha256:" + "b" * 64)])
def test_serving_context_change_refuses_old_prices(payload, field, value):
    expected = parse_runtime_context(payload["context"])
    payload["context"]["serving_context"][field] = value
    with pytest.raises(RuntimePriceError, match="context mismatch"):
        parse_measured_runtime_table(payload, expected_context=expected, expected_cost_sha256=SHA, now=NOW)


def test_missing_receipt_refuses_load(payload, tmp_path):
    path = tmp_path / "table.json"
    path.write_text(json.dumps(payload))
    (tmp_path / "synthetic-receipt.txt").unlink()
    with pytest.raises(RuntimePriceError, match="cannot read measurement receipt"):
        load_measured_runtime_table(path, expected_context=parse_runtime_context(payload["context"]),
                                    expected_cost_sha256=SHA, now=NOW)


@pytest.mark.parametrize("field,value", [("serialized_bytes", True), ("resident_bytes", -1),
                                         ("activation_bytes", 1.0), ("prefill_ms", float("inf")),
                                         ("decode_ms", -1)])
def test_invalid_resource_values_refused(field, value):
    fields = dict(prefill_ms=1, decode_ms=None, serialized_bytes=100,
                  resident_bytes=200, peak_scratch_bytes=10, activation_bytes=20)
    fields[field] = value
    with pytest.raises(RuntimePriceError):
        RuntimeResources(**fields)
