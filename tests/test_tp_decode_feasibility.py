"""CPU-only contract tests for tools/tp_decode_feasibility.py.

These cover the decide-mode arithmetic against hand-checked fixtures: the
all-reduce counts and byte counts are 2*L and H x bf16 per token, and the
Qwen3.8-27B-CB numbers in the task spec (128 collectives/token,
1.31 MB/token, ~1.1 ms wire) are the anchors.  No ranks, no network, no GPU.
"""
from __future__ import annotations

import pytest

from tools.tp_decode_feasibility import (
    BF16_BYTES,
    DEFAULT_SIZES_BYTES,
    MODEL_SPECS,
    VERDICT_LOSE,
    VERDICT_UNKNOWN,
    VERDICT_WIN,
    _resolve_sizes,
    build_decision_rows,
    check_scratch_path,
    collective_math,
    percentile,
    verdict_for,
)


# ---------------------------------------------------------------------------
# Seeded model specs are the task's fixtures; drift here is a bug.


def test_model_specs_seeded_values():
    assert MODEL_SPECS["Qwen3.8-27B-CB"] == {
        "hidden_size": 5120, "layers": 64, "tpot_ms_single": 64.4, "moe": False}
    for name in ("Qwen3.5-122B-A10B", "Mistral-Medium-3.5-128B", "DSv4-Flash"):
        assert MODEL_SPECS[name]["tpot_ms_single"] is None, name
    # MoE flag matters because routing adds collectives beyond 2*L.
    assert MODEL_SPECS["Qwen3.5-122B-A10B"]["moe"] is True
    assert MODEL_SPECS["Mistral-Medium-3.5-128B"]["moe"] is False
    assert MODEL_SPECS["DSv4-Flash"]["moe"] is False


def test_default_sweep_sizes_are_hidden_sizes_times_bf16():
    assert DEFAULT_SIZES_BYTES == (8192, 10240, 12288, 16384)
    hidden = [b // BF16_BYTES for b in DEFAULT_SIZES_BYTES]
    assert hidden == [4096, 5120, 6144, 8192]


def test_resolve_sizes_parses_and_sorts():
    assert _resolve_sizes(type("A", (), {"sizes_bytes": "16384,8192,12288"})) == \
        (8192, 12288, 16384)
    with pytest.raises(SystemExit):
        _resolve_sizes(type("A", (), {"sizes_bytes": "8191"}))  # not bf16-even


# ---------------------------------------------------------------------------
# Hand-checked collective arithmetic.


@pytest.mark.parametrize(
    "hidden,layers,ar_per_tok,bytes_per_tok,wire_ms",
    [
        # Qwen3.8-27B-CB: 128 AR/tok, 1310720 B/tok = 1.31 MB, ~1.05 ms @10GbE
        (5120, 64, 128, 1310720, 1.048576),
        (6144, 80, 160, 1966080, 1.572864),
        (6144, 88, 176, 2162688, 1.7301504),
        (4096, 43, 86, 704512, 0.5636096),
    ],
)
def test_collective_math_hand_checked(hidden, layers, ar_per_tok, bytes_per_tok, wire_ms):
    m = collective_math(hidden, layers, latency_us=30.0)
    assert m["all_reduces_per_token"] == ar_per_tok == 2 * layers
    assert m["bytes_per_allreduce"] == hidden * 2
    assert m["bytes_per_token"] == bytes_per_tok
    assert m["wire_ms_at_link"] == pytest.approx(wire_ms, abs=1e-9)
    assert m["added_ms"] == pytest.approx(ar_per_tok * 30.0 / 1000.0)


def test_added_latency_matches_spec_examples():
    # Spec: Qwen3.8-27B at 30/100/200 us/collective => 3.8/12.8/25.6 ms/token.
    for us, expected_ms in ((30.0, 3.84), (100.0, 12.8), (200.0, 25.6)):
        m = collective_math(5120, 64, latency_us=us)
        assert m["added_ms"] == pytest.approx(expected_ms, rel=1e-12)


def test_qwen_cb_budget_and_crossover_threshold():
    # TPOT 64.4 ms => budget 32.2 ms over 128 collectives => <251.5625 us each.
    tpot = MODEL_SPECS["Qwen3.8-27B-CB"]["tpot_ms_single"]
    budget_ms = tpot / 2.0
    assert budget_ms == pytest.approx(32.2)
    threshold_us = budget_ms * 1000.0 / 128.0
    assert threshold_us == pytest.approx(251.5625)
    # Just under the threshold wins; just over loses.
    assert verdict_for(tpot, 128 * (threshold_us - 1.0) / 1000.0) == VERDICT_WIN
    assert verdict_for(tpot, 128 * (threshold_us + 1.0) / 1000.0) == VERDICT_LOSE


def test_verdict_boundary_is_strict_inequality():
    # Exactly at budget counts as LOSE (no free lunch).
    assert verdict_for(64.4, 32.2) == VERDICT_LOSE
    assert verdict_for(64.4, 32.199999) == VERDICT_WIN


@pytest.mark.parametrize("latency_us", [0.001, 30.0, 250.0, 10_000.0])
def test_unknown_tpot_never_prints_a_verdict(latency_us):
    # The hard rule: no WIN/LOSE without a measured single-box TPOT.
    for name in ("Qwen3.5-122B-A10B", "Mistral-Medium-3.5-128B", "DSv4-Flash"):
        spec = MODEL_SPECS[name]
        m = collective_math(spec["hidden_size"], spec["layers"], latency_us)
        assert verdict_for(None, m["added_ms"]) == VERDICT_UNKNOWN


def test_build_decision_rows_flat_latency():
    rows = {r["model"]: r for r in build_decision_rows(latency_us=30.0)}
    assert len(rows) == 4
    qwen = rows["Qwen3.8-27B-CB"]
    assert qwen["verdict"] == VERDICT_WIN
    assert qwen["budget_us_per_allreduce"] == pytest.approx(251.5625)
    assert rows["DSv4-Flash"]["verdict"] == VERDICT_UNKNOWN
    # Rows with latency_us=None defer the verdict to size-matched percentiles.
    deferred = {r["model"]: r for r in build_decision_rows()}
    assert all(r["verdict"] is None for r in deferred.values())


# ---------------------------------------------------------------------------
# Percentile semantics (numpy 'linear' interpolation), hand-checked.


def test_percentile_linear_interpolation():
    vals = [10.0, 20.0, 30.0, 40.0]          # pos = (n-1)*q/100
    assert percentile(vals, 50) == pytest.approx(25.0)   # idx 1.5
    assert percentile(vals, 90) == pytest.approx(37.0)   # idx 2.7
    assert percentile(vals, 99) == pytest.approx(39.7)   # idx 2.97
    assert percentile([7.0], 99) == 7.0
    assert percentile(list(range(1, 101)), 50) == pytest.approx(50.5)
    with pytest.raises(ValueError):
        percentile([], 50)


# ---------------------------------------------------------------------------
# Scratch-path discipline: host /tmp is never written.


def test_check_scratch_path_rejects_tmp():
    with pytest.raises(ValueError, match="dq-runs"):
        check_scratch_path("/tmp/opencode/whatever.json")
    with pytest.raises(ValueError):
        check_scratch_path("/tmp")


def test_check_scratch_path_accepts_dq_runs():
    p = check_scratch_path("/home/rob/dq-runs/ox-wave3-2026-08-23/tpbench/x.json")
    assert str(p).startswith("/home/rob/dq-runs/")
