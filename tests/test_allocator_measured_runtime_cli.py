"""CPU synthetic CLI contracts; fixture timings are not measured GPU evidence."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from prismaquant import allocator
from prismaquant.serve_constraints import (
    ServeConstraintError, ServeSLOs, evaluate_measured_assignment,
)


def _resources(**updates):
    return SimpleNamespace(**(dict(prefill_ms=0.0, decode_ms=0.0,
        serialized_bytes=0, resident_bytes=0, activation_bytes=0,
        peak_scratch_bytes=0, kv_bytes=0) | updates))


def test_expanded_group_and_fixed_auxiliary_resource_accounting():
    verdict = evaluate_measured_assignment(
        {"q": "A", "k": "B", "head": "BF16"},
        option_assignments={("qkv", "mixed"): {"q": "A", "k": "B"}},
        resources={("qkv", "mixed"): _resources(prefill_ms=3, decode_ms=2,
            serialized_bytes=100, resident_bytes=800, activation_bytes=40, peak_scratch_bytes=20)},
        fixed_assignment={"head": "BF16"},
        fixed_resources=_resources(prefill_ms=2, decode_ms=1, resident_bytes=500,
                                  activation_bytes=10, peak_scratch_bytes=30, kv_bytes=50),
        slos=ServeSLOs(p95_ttft_ms=4, p95_itl_ms=3, device_budget_bytes=1400,
                      kv_bytes=5, peak_scratch_bytes=6), table_identity={"synthetic": True},
    )
    assert verdict.predicted == {"operator_sum_prefill_ms": 5,
        "operator_sum_decode_ms": 3, "device_memory_bytes": 1461}
    assert verdict.violation_names() == ("operator_sum_prefill_ms", "device_memory_bytes")
    assert verdict.coverage["units_priced"] == 1
    assert verdict.as_dict()["certifies_p95"] is False
    assert verdict.as_dict()["certifies_end_to_end_slo"] is False


@pytest.mark.parametrize("assignment", [
    {"q": "A", "k": "A", "head": "BF16"},
    {"q": "A", "k": "B", "head": "FP8"},
    {"q": "A", "k": "B", "head": "BF16", "extra": "BF16"},
])
def test_expansion_promotion_and_auxiliary_drift_refuse(assignment):
    with pytest.raises(ServeConstraintError):
        evaluate_measured_assignment(assignment,
            option_assignments={("qkv", "mixed"): {"q": "A", "k": "B"}},
            resources={("qkv", "mixed"): _resources(prefill_ms=1)},
            fixed_assignment={"head": "BF16"}, fixed_resources=_resources(),
            slos=ServeSLOs(p95_ttft_ms=5), table_identity={})


@pytest.mark.parametrize("options,diagnostic", [
    (["--serve-dispatch-table", "old.json"], "mutually exclusive"),
    (["--serve-workload-mix", "prefill:a=1"], "mutually exclusive"),
    (["--slo-decode-p05-tps", "100"], "cannot certify"),
    (["--slo-prefill-p95-ttft-ms", "nan"], "positive and finite"),
])
def test_measured_cli_refuses_ambiguous_configuration(monkeypatch, capsys, options, diagnostic):
    monkeypatch.setattr(sys, "argv", ["allocator", "--probe", "unused.pkl",
        "--costs", "unused.pkl", "--layer-config", "unused-layer.json",
        "--pareto-csv", "unused-pareto.csv", "--measured-runtime-table", "unused.json",
        "--measured-runtime-context", "unused-context.json",
        "--slo-prefill-p95-ttft-ms", "5", *options])
    with pytest.raises(SystemExit, match="2"):
        allocator.main()
    assert diagnostic in capsys.readouterr().err


def _main_fixture(tmp_path, *, fixed_ms=0.0):
    import torch
    from prismaquant.joint_aura import (
        activation_identity, arithmetic_identity, identity_sha256, make_joint_aura_entry,
    )
    from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    from prismaquant.measured_runtime_prices import SCHEMA, CONTEXT_SCHEMA

    name = "model.layers.0.self_attn.o_proj"
    formats = ["FP8_E4M3", "FP8_E5M2"]
    shape = (64, 64)
    source_content = {"config": {"fixture": True}, "weight_map": {"fixture.weight": "fixture.weight"},
        "shards": [{"path": "/fixture/synthetic.safetensors", "size": 1, "sha256": "a" * 64}]}
    source_model = {"schema": STREAMED_MODEL_IDENTITY_SCHEMA, "source": "synthetic",
        "resolved_commit": None,
        "content_sha256": canonical_json_sha256(source_content, where="synthetic CLI fixture"),
        **source_content}
    arithmetic = arithmetic_identity(torch.float32)
    rows = {}
    for fmt, loss in zip(formats, (1.0, 2.0)):
        probe = {"schema": "prismaquant.joint_aura.probes.v2", "seed_base": 0,
                 "n_probes": 3, "calibration_sha256": "c" * 64,
                 "producer_source_sha256": "d" * 64, "source_model": source_model,
                 "distribution": "rademacher", "normalization": "global_kl_fisher",
                 "temperature": 1.0, "arithmetic": arithmetic}
        operator = {"schema": "prismaquant.joint_aura.operator.v2", "qname": name,
                    "format": fmt, "probe_identity_sha256": identity_sha256(probe),
                    "source_weight": {"content_sha256": "a" * 64, "shape": list(shape),
                                      "dtype": "torch.float32", "logical_bytes": 16384},
                    "rendered_weight": {"content_sha256": "b" * 64, "shape": list(shape),
                                        "dtype": "torch.float32", "logical_bytes": 16384},
                    "activation": activation_identity(allocator.fr.get_format(fmt), {}, name),
                    "arithmetic": arithmetic}
        rows[fmt] = make_joint_aura_entry(operator_identity=operator, probe_identity=probe,
            signed_components=[dict(weight=math.sqrt(2 * loss), activation=0.0,
                                    mixed=0.0, total=math.sqrt(2 * loss)) for _ in range(3)])
    probe_path, cost_path = tmp_path / "probe.pkl", tmp_path / "costs.pkl"
    probe_path.write_bytes(pickle.dumps({"stats": {name: {
        "h_trace": 1.0, "n_params": 4096, "in_features": 64, "out_features": 64}},
        "meta": {"model": None}}))
    cost_path.write_bytes(pickle.dumps({"costs": {name: rows},
        "meta": {"formats": formats}, "provenance": {"cost_mode": "aura",
            "joint_activation": True, "cost_currency": "joint_aura_predicted_dloss"}}))
    receipt = tmp_path / "synthetic-receipt.txt"
    receipt.write_text("Synthetic CPU test fixture, not GPU measurement evidence.\n")
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    context = {"schema": CONTEXT_SCHEMA,
        "serving_context": {"platform": "sm_121", "structure": "dense", "residency": "resident",
                            "runtime_image": "fixture@sha256:" + "a" * 64, "execution_mode": "eager"},
        "gpu_identity": "synthetic", "runtime_sha256": "a" * 64,
        "source_sha256": source_model["content_sha256"], "calibration_sha256": "c" * 64,
        "prompt_tokens": 16, "batch_size": 1, "tensor_parallel": 1, "graph_mode": "eager",
        "operator_routes": {name: {fmt: f"synthetic-{fmt}" for fmt in formats}}}
    table_rows = []
    for fmt, milliseconds in zip(formats, (8.0, 2.0)):
        from prismaquant.allocator_candidates import serialized_candidate_payload
        serialized, _, _ = serialized_candidate_payload(
            allocator.fr.get_format(fmt), shape, qname=name, cb_serialization_context=None)
        table_rows.append({"unit": name, "format": fmt,
            "binding": {"member_formats": {name: fmt},
                "member_operator_identity_sha256": {name: rows[fmt]["joint_operator_identity_sha256"]},
                "member_shapes": {name: list(shape)}, "operator_route": context["operator_routes"][name][fmt]},
            "resources": vars(_resources(prefill_ms=milliseconds, decode_ms=None,
                serialized_bytes=serialized, resident_bytes=16384)),
            "prefill": {"method": "cuda_events", "samples_ms": [milliseconds] * 3,
                "warmup_iterations": 3, "receipt_path": receipt.name, "receipt_sha256": receipt_sha},
            "decode": None})
    now = datetime.now(timezone.utc)
    table = {"schema": SCHEMA, "table_id": "synthetic-only", "status": "proposal_data",
        "composition": "sequential_operator_sum", "context": context,
        "cost_sha256": hashlib.sha256(cost_path.read_bytes()).hexdigest(),
        "measured_at": (now - timedelta(days=1)).isoformat(),
        "valid_until": (now + timedelta(days=1)).isoformat(), "fixed_assignment": {},
        "fixed_resources": vars(_resources(prefill_ms=fixed_ms)),
        "fixed_resources_receipt_path": receipt.name,
        "fixed_resources_receipt_sha256": receipt_sha, "rows": table_rows}
    table_path, context_path = tmp_path / "runtime.json", tmp_path / "context.json"
    table_path.write_text(json.dumps(table))
    context_path.write_text(json.dumps(context))
    argv = ["allocator", "--probe", str(probe_path), "--costs", str(cost_path),
        "--formats", ",".join(formats), "--allow-default-profile", "--target-bits", "9",
        "--pareto-targets", "9", "--layer-config", str(tmp_path / "layer.json"),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
        "--pareto-output-dir", str(tmp_path / "seeds"),
        "--measured-runtime-table", str(table_path), "--measured-runtime-context", str(context_path),
        "--slo-prefill-p95-ttft-ms", "5"]
    return name, argv


def test_real_cli_selects_same_byte_faster_alternative(tmp_path, monkeypatch):
    name, argv = _main_fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    allocator.main()
    result = json.loads((tmp_path / "layer.json").read_text())
    from prismaquant.layer_config import load_assignment
    assert load_assignment(tmp_path / "layer.json")[name] == "FP8_E5M2"
    verdict = result["__prismaquant__"]["serve_constraints"]
    assert verdict["predicted"]["operator_sum_prefill_ms"] == 2
    assert verdict["certifies_p95"] is False
    seeds = list((tmp_path / "seeds").glob("allocator_target_*.json"))
    assert seeds
    assert json.loads(seeds[0].read_text())["serve_constraints"]["certifies_end_to_end_slo"] is False


def test_real_cli_charges_fixed_runtime_before_search(tmp_path, monkeypatch):
    _, argv = _main_fixture(tmp_path, fixed_ms=4)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="measured runtime proposal infeasible"):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()


def test_real_cli_without_measured_opt_in_keeps_quality_objective(tmp_path, monkeypatch):
    name, argv = _main_fixture(tmp_path)
    argv = argv[:argv.index("--measured-runtime-table")]
    monkeypatch.setattr(sys, "argv", argv)
    allocator.main()
    from prismaquant.layer_config import load_assignment
    assert load_assignment(tmp_path / "layer.json")[name] == "FP8_E4M3"
    meta = json.loads((tmp_path / "layer.json").read_text())["__prismaquant__"]
    assert "measured_runtime_search" not in meta
    assert "serve_constraints" not in meta


def test_real_cli_refuses_promotion_of_priced_proposal(tmp_path, monkeypatch):
    name, argv = _main_fixture(tmp_path)
    def change_format(assignment, *args, **kwargs):
        return {**assignment, name: "FP8_E4M3"}
    monkeypatch.setattr(allocator, "promote_serving_units", change_format)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="promotion changed a measured runtime proposal"):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()


def test_real_cli_refuses_mismatched_fixed_auxiliary_assignment(tmp_path, monkeypatch):
    _, argv = _main_fixture(tmp_path)
    path = tmp_path / "runtime.json"
    payload = json.loads(path.read_text())
    payload["fixed_assignment"] = {"mtp.head": "BF16"}
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="fixed auxiliary assignment differs"):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()


def test_real_cli_uses_resident_bytes_in_device_constraint(tmp_path, monkeypatch):
    _, argv = _main_fixture(tmp_path)
    # Wire payload is 4352 bytes; terminal resident weights are 16384.
    monkeypatch.setattr(sys, "argv", argv + ["--serve-device-budget-bytes", "10000"])
    with pytest.raises(SystemExit, match="measured runtime proposal infeasible"):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()


@pytest.mark.parametrize("field", ["source_sha256", "calibration_sha256"])
def test_real_cli_binds_expected_context_to_joint_cost_evidence(tmp_path, monkeypatch, field):
    _, argv = _main_fixture(tmp_path)
    table_path, context_path = tmp_path / "runtime.json", tmp_path / "context.json"
    table, context = json.loads(table_path.read_text()), json.loads(context_path.read_text())
    context[field] = "f" * 64
    table["context"] = context
    context_path.write_text(json.dumps(context))
    table_path.write_text(json.dumps(table))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="source/calibration differs from the joint AURA"):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()


def test_real_cli_rejects_cost_bytes_swapped_between_admission_and_parsing(tmp_path, monkeypatch):
    """Restoring admitted bytes after parsing must not authenticate different prices."""
    from prismaquant import measured_runtime_prices
    from prismaquant.joint_aura import make_joint_aura_entry

    name, argv = _main_fixture(tmp_path)
    cost_path = tmp_path / "costs.pkl"
    admitted_bytes = cost_path.read_bytes()
    changed = pickle.loads(admitted_bytes)
    row = changed["costs"][name]["FP8_E5M2"]
    changed["costs"][name]["FP8_E5M2"] = make_joint_aura_entry(
        operator_identity=row["joint_operator_identity"],
        probe_identity=row["probe_identity"],
        signed_components=[dict(weight=6.0, activation=0.0, mixed=0.0, total=6.0)] * 3,
    )
    changed_bytes = pickle.dumps(changed)
    assert hashlib.sha256(changed_bytes).digest() != hashlib.sha256(admitted_bytes).digest()
    real_admit = measured_runtime_prices.load_measured_runtime_table
    real_load = pickle.load

    def admit_then_replace(*args, **kwargs):
        table = real_admit(*args, **kwargs)
        cost_path.write_bytes(changed_bytes)
        return table

    def parse_then_restore(stream, *args, **kwargs):
        value = real_load(stream, *args, **kwargs)
        if getattr(stream, "name", None) == str(cost_path):
            # Simulate an in-place writer restoring the admitted file between
            # the vulnerable parser's first read and its later hash read.
            cost_path.write_bytes(admitted_bytes)
        return value

    monkeypatch.setattr(measured_runtime_prices, "load_measured_runtime_table", admit_then_replace)
    monkeypatch.setattr(pickle, "load", parse_then_restore)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="cost payload changed after measured runtime admission"):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()
