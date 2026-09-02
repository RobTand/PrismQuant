"""The constrained-Pareto axis, end to end through ``allocator.main()`` (P5c).

``docs/lanes/nvfp4-cb/format-speed-policy.md`` §1 specifies quality-minimizing
selection under hard byte AND serving constraints and, until ultraplan P5c,
deferred the serving half. This file pins the shipped behaviour of that half
by driving the REAL ``allocator.main()`` — same harness as
``test_allocator_byte_budget_selection.py``, which pins the byte half — so
what is tested is the code that ships selections.

Three properties:

1. **Absent constraints change nothing.** With no dispatch table and no SLOs,
   every number in ``selection.json`` and every byte of ``layer_config.json``
   must be what the pre-P5c allocator wrote. The only permitted difference is
   a provenance stamp recording that constraints were absent. This is pinned
   by running the same fixture twice — once with the feature entirely absent,
   once with the feature PRESENT BUT UNUSED (a table supplied, no SLO) — and
   comparing.

2. **An SLO is a constraint, not a penalty.** The decisive case: the
   min-Δloss assignment fits the card but misses the prefill SLO, so it is
   INFEASIBLE and a worse-Δloss assignment ships. The rejected assignment and
   the binding constraint must both be recoverable from ``selection.json``.

3. **The objective is untouched.** Among the assignments that satisfy both
   axes, selection is still minimum predicted Δloss with ties to the larger
   footprint, and no λ appears anywhere.
"""
from __future__ import annotations

import json
import pickle
import struct
import sys

import pytest

import prismaquant.allocator as alloc
from prismaquant import footprint as fp
from prismaquant import format_registry as fr
from prismaquant.serve_dispatch_table import SCHEMA as TABLE_SCHEMA

_NAMES = [f"model.layers.{i}.self_attn.o_proj" for i in range(4)]
_OUT = _IN = 256
_NPARAMS = _OUT * _IN
_OVERHEAD_RESERVE = 512
_FLOOR_TENSORS = {
    "model.embed_tokens.weight": ("BF16", (512, 64)),
    "lm_head.weight": ("BF16", (512, 64)),
    "model.norm.weight": ("BF16", (64,)),
}

_PROV = {
    "source": "tests/test_allocator_serve_constraints.py synthetic fixture",
    "date": "2026-08-01",
    "gpu": "synthetic",
    "measured_quantity": "synthetic relative serving cost",
    "units": "dimensionless",
    "derivation": "fixture constant",
}


def _write_safetensors(path, tensors):
    header = {}
    off = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = fp._ST_DTYPE_BYTES[dtype]
        for d in shape:
            nbytes *= d
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * off)


def _fixture(tmp_path, *, nvfp4_dloss, fp8_dloss):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tensors = dict(_FLOOR_TENSORS)
    for n in _NAMES:
        tensors[f"{n}.weight"] = ("BF16", (_OUT, _IN))
    _write_safetensors(model_dir / "model-00001.safetensors", tensors)

    stats = {
        n: {"h_trace": 1.0 + 0.1 * i, "n_params": _NPARAMS,
            "in_features": _IN, "out_features": _OUT}
        for i, n in enumerate(_NAMES)
    }
    probe = {"stats": stats, "meta": {"model": str(model_dir)}}
    costs = {
        "costs": {
            n: {
                "NVFP4": {"weight_mse": nvfp4_dloss, "output_mse": nvfp4_dloss,
                          "output_mse_measured": True,
                          "predicted_dloss": nvfp4_dloss},
                "FP8_E4M3": {"weight_mse": fp8_dloss, "output_mse": fp8_dloss,
                             "output_mse_measured": True,
                             "predicted_dloss": fp8_dloss},
            }
            for n in _NAMES
        },
        "meta": {"formats": ["NVFP4", "FP8_E4M3"]},
    }
    probe_p = tmp_path / "probe.pkl"
    cost_p = tmp_path / "cost.pkl"
    probe_p.write_bytes(pickle.dumps(probe))
    cost_p.write_bytes(pickle.dumps(costs))
    return model_dir, probe_p, cost_p, stats


def _stub_solver(fmt_for_target):
    def solve(stats, candidates, target_bits, format_specs, format_rank,
              bit_precision, **kw):
        fmt = fmt_for_target(float(target_bits))
        assign = {n: fmt for n in candidates}
        total_params = sum(stats[n]["n_params"] for n in assign)
        bits = 0.0
        for n in assign:
            cand = next(c for c in candidates[n] if c.fmt == fmt)
            bits += 8.0 * cand.memory_bytes
        achieved = bits / max(total_params, 1)
        diag = kw.get("diagnostics")
        if diag is not None:
            diag.update({"feasible": True, "achieved_bits": achieved,
                         "predicted_dloss": None, "evals": 1})
        return assign, achieved
    return solve


def _artifact_bytes(model_dir, fmt, stats):
    info = fp.floor_bytes_for_model(str(model_dir), _NAMES, stats)
    body = 0
    for n in _NAMES:
        body += fr.get_format(fmt).memory_bytes_for_shape((_OUT, _IN))
        if fmt == "NVFP4":
            body += fp.nvfp4_global_sidecar_bytes(n, (_OUT, _IN))
    return info["floor_bytes"] + body, info["floor_bytes"]


def _dispatch_table(tmp_path, *, nvfp4_prefill, fp8_prefill,
                    nvfp4_decode=1.0, fp8_decode=1.0, name="table.json"):
    payload = {
        "schema": TABLE_SCHEMA,
        "table_id": "fixture",
        "status": "proposal_data",
        "description": "synthetic fixture",
        "arenas": [
            {"phase": "prefill", "m_regime": "dense", "m": 1400,
             "reference_route": "fixture native", "metric": "ttft_ms",
             "absolute_value": 1000.0, "statistic": "p95",
             "provenance": dict(_PROV)},
            {"phase": "decode", "m_regime": "batch1", "m": 1,
             "reference_route": "fixture native", "metric": "decode_tok_s",
             "absolute_value": 10.0, "statistic": "p05",
             "provenance": dict(_PROV)},
        ],
        "rows": [
            {"format_family": "NVFP4", "phase": "prefill", "m_regime": "dense",
             "lane": "native", "relative_unit_cost": nvfp4_prefill,
             "provenance": dict(_PROV)},
            {"format_family": "FP8_E4M3", "phase": "prefill",
             "m_regime": "dense", "lane": "native",
             "relative_unit_cost": fp8_prefill, "provenance": dict(_PROV)},
            {"format_family": "NVFP4", "phase": "decode",
             "m_regime": "batch1", "lane": "native",
             "relative_unit_cost": nvfp4_decode, "provenance": dict(_PROV)},
            {"format_family": "FP8_E4M3", "phase": "decode",
             "m_regime": "batch1", "lane": "native",
             "relative_unit_cost": fp8_decode, "provenance": dict(_PROV)},
        ],
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2))
    return path


def _run(monkeypatch, tmp_path, probe_p, cost_p, *, disk_gb, fmt_for_target,
         pareto="4.6,8.2", extra_argv=(), out_prefix="run"):
    """Drive the real ``allocator.main()``; outputs land in their own subdir.

    ``selection.json`` is written beside ``--pareto-csv``, so two runs in one
    fixture need two directories rather than two filename prefixes.
    """
    monkeypatch.setattr(alloc, "solve_with_promotion",
                        _stub_solver(fmt_for_target))
    out_dir = tmp_path / out_prefix
    out_dir.mkdir(exist_ok=True)
    lc = out_dir / "layer_config.json"
    csv = out_dir / "pareto.csv"
    argv = [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", "NVFP4,FP8_E4M3",
        "--pareto-targets", pareto,
        "--target-disk-gb", repr(disk_gb),
        "--layer-config", str(lc),
        "--pareto-csv", str(csv),
        "--allow-default-profile",
        "--artifact-overhead-reserve-bytes", str(_OVERHEAD_RESERVE),
        *extra_argv,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    alloc.main()
    selection = json.loads((out_dir / "selection.json").read_text())
    layer_cfg = json.loads(lc.read_text())
    return selection, layer_cfg


# ---------------------------------------------------------------------------
# 1. Absent constraints: byte-identical to the pre-P5c allocator
# ---------------------------------------------------------------------------
def test_no_constraints_selection_is_identical_to_feature_present_but_unused(
        monkeypatch, tmp_path):
    """The compatibility pin.

    Run A supplies none of the new flags. Run B supplies a dispatch table but
    no SLO, so the constraint axis is present and inert. Every key of
    ``selection.json`` must agree except the provenance stamp — and the
    layer_config must agree completely, because the stamp is written there
    only when the axis actually ran.
    """
    model_dir, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    fp8_bytes, _floor = _artifact_bytes(model_dir, "FP8_E4M3", stats)
    table = _dispatch_table(tmp_path, nvfp4_prefill=1.0, fp8_prefill=9.0)

    kw = dict(
        disk_gb=(fp8_bytes + 10_000) / fp.GB,
        fmt_for_target=lambda t: "FP8_E4M3" if t >= 6.0 else "NVFP4",
    )
    sel_a, cfg_a = _run(monkeypatch, tmp_path, probe_p, cost_p,
                        out_prefix="a", **kw)
    sel_b, cfg_b = _run(monkeypatch, tmp_path, probe_p, cost_p,
                        out_prefix="b",
                        extra_argv=["--serve-dispatch-table", str(table)],
                        **kw)

    stamp_a = sel_a.pop("serve_constraints")
    stamp_b = sel_b.pop("serve_constraints")
    # ``solver_seconds`` is a wall-clock float: two runs of the same solve
    # differ in it by construction, so comparing it asserts that the machine
    # was equally busy both times, not that the selection is identical.  It
    # travels in the selection AND inside the layer_config's provenance, so
    # drop it wherever it appears and keep every count.
    def _drop_timings(node):
        if isinstance(node, dict):
            node.pop("solver_seconds", None)
            for value in node.values():
                _drop_timings(value)
        elif isinstance(node, list):
            for value in node:
                _drop_timings(value)
        return node

    _drop_timings(sel_a)
    _drop_timings(sel_b)
    _drop_timings(cfg_a)
    _drop_timings(cfg_b)
    assert sel_a == sel_b, (
        "supplying a dispatch table without an SLO must not change any "
        "selection number")
    assert cfg_a == cfg_b, (
        "the layer_config must be byte-identical when the constraint axis "
        "did not run")

    # Both runs stamp that constraints were absent, and neither claims
    # anything about latency.
    for stamp in (stamp_a, stamp_b):
        assert stamp["active"] is False
        assert stamp["aggregation_model"] is None
        assert "no latency" in stamp["note"]
        assert stamp["solver_contract"] == (
            "additive_candidate_proposal_then_exact_assignment_filter")
    # ...and the only difference between the two stamps is the table identity
    # run B supplied, which is exactly the "modulo a provenance stamp" clause.
    assert stamp_a["dispatch_table"] is None
    assert stamp_b["dispatch_table"]["table_id"] == "fixture"
    assert {k: v for k, v in stamp_a.items() if k != "dispatch_table"} == {
        k: v for k, v in stamp_b.items() if k != "dispatch_table"}


def test_unconstrained_ratchet_trace_rows_gain_no_keys(
        monkeypatch, tmp_path):
    """The trace is a shipped artifact; unused features must not widen it."""
    model_dir, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    fp8_bytes, _floor = _artifact_bytes(model_dir, "FP8_E4M3", stats)
    selection, _cfg = _run(
        monkeypatch, tmp_path, probe_p, cost_p,
        disk_gb=(fp8_bytes + 10_000) / fp.GB,
        fmt_for_target=lambda t: "FP8_E4M3" if t >= 6.0 else "NVFP4")
    for row in selection["ratchet_trace"]:
        assert set(row) == {
            "stage", "target_bits", "achieved_bits", "tensor_payload_gb",
            "whole_artifact_upper_bound_gb", "dloss", "fits", "accepted",
        }
    assert "rejected_assignments" not in selection["serve_constraints"]


# ---------------------------------------------------------------------------
# 2. The decisive case: an SLO flips the selection
# ---------------------------------------------------------------------------
def test_slo_makes_the_min_dloss_assignment_infeasible_and_flips_selection(
        monkeypatch, tmp_path):
    """FP8 is the better allocation on quality AND it fits the card, so the
    unconstrained allocator ships it. Under a prefill SLO it is 9x the
    reference and misses — so it is INFEASIBLE, not "worse-scored", and the
    NVFP4 allocation ships instead. The rejected arm and the constraint that
    rejected it must both be in the artifact."""
    model_dir, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)   # FP8 strictly better
    fp8_bytes, _floor = _artifact_bytes(model_dir, "FP8_E4M3", stats)
    # 1000 ms reference; NVFP4 is 1.0x (1000 ms), FP8 is 1.4x (1400 ms).
    table = _dispatch_table(tmp_path, nvfp4_prefill=1.0, fp8_prefill=1.4)
    kw = dict(
        disk_gb=(fp8_bytes + 10_000) / fp.GB,
        fmt_for_target=lambda t: "FP8_E4M3" if t >= 6.0 else "NVFP4",
    )

    # Baseline: no SLO -> the min-Δloss (FP8) allocation ships.
    unconstrained, cfg_u = _run(
        monkeypatch, tmp_path, probe_p, cost_p, out_prefix="u", **kw)
    assert unconstrained["chosen_target_bits"] == 8.2
    assert {c["data_type"] for n, c in cfg_u.items()
            if n != "__prismaquant__"} == {"fp8_e4m3"}

    # Same inputs, one hard prefill SLO of 1200 ms.
    constrained, cfg_c = _run(
        monkeypatch, tmp_path, probe_p, cost_p, out_prefix="c",
        extra_argv=[
            "--serve-dispatch-table", str(table),
            "--serve-workload-mix", "prefill:dense=1.0",
            "--slo-prefill-p95-ttft-ms", "1200",
        ],
        **kw)

    assert constrained["chosen_target_bits"] == 4.6
    assert {c["data_type"] for n, c in cfg_c.items()
            if n != "__prismaquant__"} == {"nv_fp"}
    # The objective did NOT change: the shipped arm is still worse on Δloss,
    # it simply is the best among the FEASIBLE ones.
    assert constrained["predicted_dloss"] > unconstrained["predicted_dloss"]
    assert constrained["ratchet_objective"] == (
        "min_predicted_dloss__ties_to_larger_footprint")

    serve = constrained["serve_constraints"]
    assert serve["active"] is True
    assert serve["feasible"] is True
    assert serve["lambda_blended_objective"] is False
    assert serve["predicted"]["p95_ttft_ms"] == pytest.approx(1000.0)
    assert serve["binding_constraint_at_optimum"] == "p95_ttft_ms"

    # The rejected assignment is named, with the constraint that rejected it.
    rejected = serve["rejected_assignments"]
    assert rejected, "the FP8 arm must be recorded as rejected, not vanish"
    assert all(r["binding_constraint"] == "p95_ttft_ms" for r in rejected)
    assert any(r["predicted"]["p95_ttft_ms"] == pytest.approx(1400.0)
               for r in rejected)
    assert any(r["violations"][0]["limit"] == 1200.0 for r in rejected)
    assert serve["n_probes_rejected_by_serving_constraints"] == len(rejected)

    # The 8.2 grid rung fitted the CARD and was removed by the SLO alone.
    [fp8_rejection] = [r for r in rejected if r["target_bits"] == 8.2]
    assert fp8_rejection["violated_constraints"] == ["p95_ttft_ms"]
    [fp8_grid_row] = [g for g in constrained["grid"]
                      if g["target_bits"] == 8.2]
    assert fp8_grid_row["fits"], (
        "the byte axis still reports the rung as fitting; the SLO axis is a "
        "SECOND constraint, not a redefinition of the first")

    # The ratchet trace distinguishes the two rejection reasons on every probe
    # it evaluated, and never accepts a serve-infeasible one.
    slow_rows = [r for r in constrained["ratchet_trace"]
                 if r["serve_feasible"] is False]
    assert slow_rows, "the FP8 probes must be traced, not silently skipped"
    for row in slow_rows:
        assert row["fits_bytes"] is True
        assert row["fits"] is False and row["accepted"] is False
        assert row["serve_binding_constraint"] == "p95_ttft_ms"
        assert row["serve_violated_constraints"] == ["p95_ttft_ms"]

    # And the shipped layer_config carries the verdict too.
    meta = cfg_c["__prismaquant__"]["serve_constraints"]
    assert meta["feasible"] is True and meta["active"] is True
    assert meta["global_optimality_claimed"] is False


def test_satisfiable_slo_leaves_the_selection_unchanged(
        monkeypatch, tmp_path):
    """Non-regression: a constraint everything satisfies must not move the
    pick. The axis is feasibility only — it never re-ranks the survivors."""
    model_dir, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    fp8_bytes, _floor = _artifact_bytes(model_dir, "FP8_E4M3", stats)
    table = _dispatch_table(tmp_path, nvfp4_prefill=1.0, fp8_prefill=1.4)
    kw = dict(
        disk_gb=(fp8_bytes + 10_000) / fp.GB,
        fmt_for_target=lambda t: "FP8_E4M3" if t >= 6.0 else "NVFP4",
    )
    loose, _cfg = _run(
        monkeypatch, tmp_path, probe_p, cost_p, out_prefix="l",
        extra_argv=[
            "--serve-dispatch-table", str(table),
            "--serve-workload-mix", "prefill:dense=1.0",
            "--slo-prefill-p95-ttft-ms", "99999",
        ],
        **kw)
    assert loose["chosen_target_bits"] == 8.2
    assert loose["serve_constraints"]["rejected_assignments"] == []
    assert loose["serve_constraints"]["predicted"]["p95_ttft_ms"] == (
        pytest.approx(1400.0))


def test_every_allocation_missing_the_slo_exits_rather_than_relaxing_it(
        monkeypatch, tmp_path):
    """Hard means hard: when nothing is feasible the run stops and says which
    limit bound, instead of shipping the closest miss."""
    model_dir, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    fp8_bytes, _floor = _artifact_bytes(model_dir, "FP8_E4M3", stats)
    table = _dispatch_table(tmp_path, nvfp4_prefill=5.0, fp8_prefill=9.0)
    with pytest.raises(SystemExit, match="hard serving constraint"):
        _run(monkeypatch, tmp_path, probe_p, cost_p, out_prefix="x",
             disk_gb=(fp8_bytes + 10_000) / fp.GB,
             fmt_for_target=lambda t: "FP8_E4M3" if t >= 6.0 else "NVFP4",
             extra_argv=[
                 "--serve-dispatch-table", str(table),
                 "--serve-workload-mix", "prefill:dense=1.0",
                 "--slo-prefill-p95-ttft-ms", "1200",
             ])
    selection = json.loads((tmp_path / "x" / "selection.json").read_text())
    assert selection["serve_constraints_infeasible"] is True
    assert selection["serve_constraints"]["binding_constraints"] == [
        "p95_ttft_ms"]
    assert selection["serve_constraints"]["rejected_assignments"]


def test_device_memory_constraint_rejects_the_denser_allocation(
        monkeypatch, tmp_path):
    """The fourth constraint of policy §1: resident + KV + peak scratch."""
    model_dir, probe_p, cost_p, stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    nvfp4_bytes, _floor = _artifact_bytes(model_dir, "NVFP4", stats)
    fp8_bytes, _ = _artifact_bytes(model_dir, "FP8_E4M3", stats)
    table = _dispatch_table(tmp_path, nvfp4_prefill=1.0, fp8_prefill=1.0)
    kv, scratch = 1000, 2000
    # A device budget that the NVFP4 residency clears and FP8's does not.
    budget = nvfp4_bytes + kv + scratch + 10
    assert budget < fp8_bytes + kv + scratch
    selection, cfg = _run(
        monkeypatch, tmp_path, probe_p, cost_p, out_prefix="m",
        disk_gb=(fp8_bytes + 10_000) / fp.GB,
        fmt_for_target=lambda t: "FP8_E4M3" if t >= 6.0 else "NVFP4",
        extra_argv=[
            "--serve-dispatch-table", str(table),
            "--serve-workload-mix", "prefill:dense=1.0",
            "--serve-device-budget-bytes", str(int(budget)),
            "--serve-kv-bytes", str(kv),
            "--serve-peak-scratch-bytes", str(scratch),
        ])
    assert selection["chosen_target_bits"] == 4.6
    assert {c["data_type"] for n, c in cfg.items()
            if n != "__prismaquant__"} == {"nv_fp"}
    serve = selection["serve_constraints"]
    assert serve["coverage"]["memory"]["kv_bytes"] == kv
    assert serve["coverage"]["memory"]["peak_scratch_bytes"] == scratch
    assert any(r["binding_constraint"] == "device_memory_bytes"
               for r in serve["rejected_assignments"])


# ---------------------------------------------------------------------------
# 3. Operator-facing input errors
# ---------------------------------------------------------------------------
def test_slo_without_a_table_exits_on_the_command_line(monkeypatch, tmp_path):
    _model_dir, probe_p, cost_p, _stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    with pytest.raises(SystemExit, match="serve-dispatch-table"):
        _run(monkeypatch, tmp_path, probe_p, cost_p, out_prefix="e",
             disk_gb=1.0, fmt_for_target=lambda t: "NVFP4",
             extra_argv=["--slo-prefill-p95-ttft-ms", "10"])


def test_malformed_workload_mix_exits_on_the_command_line(
        monkeypatch, tmp_path):
    _model_dir, probe_p, cost_p, _stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    table = _dispatch_table(tmp_path, nvfp4_prefill=1.0, fp8_prefill=1.0)
    with pytest.raises(SystemExit, match="sums to"):
        _run(monkeypatch, tmp_path, probe_p, cost_p, out_prefix="f",
             disk_gb=1.0, fmt_for_target=lambda t: "NVFP4",
             extra_argv=[
                 "--serve-dispatch-table", str(table),
                 "--serve-workload-mix", "prefill:dense=0.5",
                 "--slo-prefill-p95-ttft-ms", "10",
             ])


def test_unreadable_dispatch_table_exits_on_the_command_line(
        monkeypatch, tmp_path):
    _model_dir, probe_p, cost_p, _stats = _fixture(
        tmp_path, nvfp4_dloss=1e-4, fp8_dloss=1e-6)
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "nope"}')
    with pytest.raises(SystemExit, match="schema"):
        _run(monkeypatch, tmp_path, probe_p, cost_p, out_prefix="g",
             disk_gb=1.0, fmt_for_target=lambda t: "NVFP4",
             extra_argv=["--serve-dispatch-table", str(bad)])
