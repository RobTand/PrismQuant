"""Exact discrete runtime search: exhaustive oracles, never speed heuristics."""
from __future__ import annotations

import itertools
import random
from types import SimpleNamespace

import pytest

from prismaquant import allocator_solver as solver


def candidate(fmt, memory, loss):
    return solver.Candidate(fmt, memory * 8 / 1000, memory, loss)


def price(c, prefill, decode=1.0, resident=0, scratch=0, activation=0):
    return SimpleNamespace(serialized_bytes=c.memory_bytes, prefill_ms=prefill,
                           decode_ms=decode, resident_bytes=resident,
                           peak_scratch_bytes=scratch, activation_bytes=activation)


def test_same_byte_faster_higher_loss_route_survives():
    a16, a8 = candidate("W4A16", 500, 0.1), candidate("W4A8", 500, 0.3)
    candidates = {"q": [a16, a8]}
    resources = {("q", a16.fmt): price(a16, 9.0),
                 ("q", a8.fmt): price(a8, 2.0)}
    result = solver.solve_runtime_frontier(
        candidates, resources, max_memory_bytes=500, max_prefill_ms=3.0)
    assert [r.assignment for r in result] == [{"q": "W4A8"}]
    assert result[0].predicted_dloss == 0.3
    # Opt-in runtime search does not change the existing quality-only solver.
    legacy, _ = solver.solve_allocation({"q": {"n_params": 1000}},
                                        candidates, target_bits=4.0)
    assert legacy == {"q": "W4A16"}


def test_nonconvex_pocket_is_retained():
    cs = [candidate("cheap", 1, 10.0), candidate("pocket", 2, 9.0),
          candidate("dense", 3, 0.0)]
    resources = {("u", c.fmt): price(c, 1.0) for c in cs}
    frontier = solver.solve_runtime_frontier(
        {"u": cs}, resources, max_memory_bytes=2, max_prefill_ms=1.0)
    assert frontier[0].assignment == {"u": "pocket"}
    assert {r.assignment["u"] for r in frontier} == {"cheap", "pocket"}


def _oracle(candidates, resources, memory, prefill, decode, device, fixed=0):
    names = sorted(candidates)
    rows = []
    for cs in itertools.product(*(candidates[name] for name in names)):
        rs = [resources[(n, c.fmt)] for n, c in zip(names, cs)]
        serialized = sum(c.memory_bytes for c in cs)
        loss = sum(c.predicted_dloss for c in cs)
        pf = sum(r.prefill_ms for r in rs)
        dc = sum(r.decode_ms for r in rs)
        resident = sum(r.resident_bytes for r in rs)
        scratch = max((r.peak_scratch_bytes for r in rs), default=0)
        activation = max((r.activation_bytes for r in rs), default=0)
        if serialized > memory or pf > prefill:
            continue
        if decode is not None and dc > decode:
            continue
        if device is not None and resident + scratch + activation + fixed > device:
            continue
        vector = (serialized, loss, pf)
        if decode is not None:
            vector += (dc,)
        if device is not None:
            vector += (resident, scratch, activation)
        rows.append((vector, tuple(c.fmt for c in cs)))
    # Independent all-pairs dominance, including canonical equivalence ties.
    kept = []
    for vector, formats in rows:
        if any(all(a <= b for a, b in zip(other, vector))
               and (other != vector or other_formats < formats)
               for other, other_formats in rows):
            continue
        kept.append((vector, formats))
    return set(kept)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("decode,device", [(None, None), (10, None),
                                            (None, 40), (10, 40)])
def test_matches_exhaustive_multiresource_frontier(seed, decode, device):
    rng = random.Random(seed)
    candidates, resources = {}, {}
    for n in range(5):
        name = f"unit{n}"
        candidates[name] = []
        for i in range(3):
            c = candidate(f"fmt{i}", rng.randint(1, 8), rng.randint(-2, 10))
            candidates[name].append(c)
            resources[(name, c.fmt)] = price(c, rng.randint(1, 4),
                rng.randint(1, 3), rng.randint(0, 8), rng.randint(0, 12),
                rng.randint(0, 10))
    results = solver.solve_runtime_frontier(candidates, resources,
        max_memory_bytes=25, max_prefill_ms=15, max_decode_ms=decode,
        max_device_bytes=device, fixed_device_bytes=3)
    actual = set()
    for r in results:
        vector = (r.memory_bytes, r.predicted_dloss, r.prefill_ms)
        if decode is not None:
            vector += (r.decode_ms,)
        if device is not None:
            vector += (r.resident_bytes, r.peak_scratch_bytes, r.activation_bytes)
        actual.add((vector, tuple(r.assignment[n] for n in sorted(candidates))))
        assert r.device_bytes == r.resident_bytes + r.peak_scratch_bytes + r.activation_bytes + 3
    assert actual == _oracle(candidates, resources, 25, 15, decode, device, 3)
    assert [r.predicted_dloss for r in results] == sorted(r.predicted_dloss for r in results)


def test_peak_resources_must_not_be_collapsed_before_the_fold():
    # The first option has smaller current total device memory (6 vs 8),
    # but the second becomes better after the next unit's scratch peak (12 vs 8).
    low, scratch, last = candidate("resident", 1, 0), candidate("scratch", 1, 0), candidate("last", 1, 0)
    candidates = {"a": [low, scratch], "b": [last]}
    resources = {("a", low.fmt): price(low, 1, resident=6),
                 ("a", scratch.fmt): price(scratch, 1, scratch=8),
                 ("b", last.fmt): price(last, 1, scratch=6)}
    result = solver.solve_runtime_frontier(candidates, resources,
        max_memory_bytes=2, max_prefill_ms=2, max_device_bytes=8)
    assert result[0].assignment == {"a": "scratch", "b": "last"}
    assert result[0].device_bytes == 8


def test_exact_bytes_above_float_integer_precision():
    small = candidate("small", 2**53, 1)
    large = candidate("large", 2**53 + 1, 0)
    resources = {("a", c.fmt): price(c, 1) for c in (small, large)}
    result = solver.solve_runtime_frontier({"a": [large, small]}, resources,
        max_memory_bytes=2**53, max_prefill_ms=1)
    assert result[0].assignment == {"a": "small"}


def test_whole_group_option_remains_atomic():
    c = candidate("group_option", 100, 2)
    c.member_formats = {"q": "W4A8", "k": "W4A8"}
    resources = {("qkv", c.fmt): price(c, 2, resident=120)}
    result = solver.solve_runtime_frontier({"qkv": [c]}, resources,
        max_memory_bytes=100, max_prefill_ms=3)
    assert result[0].assignment == {"qkv": "group_option"}
    assert result[0].chosen_candidates["qkv"].member_formats == c.member_formats


def test_missing_group_price_does_not_sum_leaf_prices():
    c = candidate("group_option", 100, 2)
    c.member_formats = {"q": "W4A8", "k": "W4A8"}
    resources = {("q", "W4A8"): price(c, 1), ("k", "W4A8"): price(c, 1)}
    with pytest.raises(ValueError, match="qkv.*group_option"):
        solver.solve_runtime_frontier({"qkv": [c]}, resources,
            max_memory_bytes=100, max_prefill_ms=3)


@pytest.mark.parametrize("field,value", [
    ("prefill_ms", None), ("prefill_ms", float("nan")),
    ("prefill_ms", float("inf")), ("prefill_ms", -1),
    ("prefill_ms", True), ("prefill_ms", "1"),
    ("decode_ms", float("nan")), ("decode_ms", -1),
    ("resident_bytes", -1), ("resident_bytes", 1.5),
    ("activation_bytes", True), ("peak_scratch_bytes", -1),
    ("serialized_bytes", 11),
])
def test_invalid_runtime_rows_refuse_even_if_candidate_is_infeasible(field, value):
    c = candidate("a", 10, 1)
    r = price(c, 1)
    setattr(r, field, value)
    with pytest.raises(ValueError, match=field):
        solver.solve_runtime_frontier({"u": [c]}, {("u", "a"): r},
            max_memory_bytes=0, max_prefill_ms=0)


def test_missing_decode_refuses_only_when_constrained():
    c = candidate("a", 1, 1)
    resources = {("u", "a"): price(c, 1, decode=None)}
    assert solver.solve_runtime_frontier({"u": [c]}, resources,
        max_memory_bytes=1, max_prefill_ms=1)[0].decode_ms is None
    with pytest.raises(ValueError, match="decode_ms"):
        solver.solve_runtime_frontier({"u": [c]}, resources,
            max_memory_bytes=1, max_prefill_ms=1, max_decode_ms=5)


@pytest.mark.parametrize("kwargs", [
    {"max_memory_bytes": -1}, {"max_memory_bytes": 1.2},
    {"max_prefill_ms": float("nan")}, {"max_decode_ms": float("inf")},
    {"max_device_bytes": True}, {"fixed_device_bytes": -1},
    {"max_states": 0}, {"max_states": True}, {"max_transitions": -1},
])
def test_invalid_budgets_refuse(kwargs):
    options = dict(max_memory_bytes=1, max_prefill_ms=1)
    options.update(kwargs)
    with pytest.raises(ValueError):
        solver.solve_runtime_frontier({}, {}, **options)


def test_deterministic_ties_and_state_cap():
    cs = [candidate("c", 3, 0), candidate("b", 2, 1), candidate("a", 1, 2)]
    resources = {("u", c.fmt): price(c, 1) for c in cs}
    for options in (cs, list(reversed(cs))):
        diagnostics = {}
        with pytest.raises(solver.RuntimeFrontierLimitError, match="max_states=2"):
            solver.solve_runtime_frontier({"u": options}, resources,
                max_memory_bytes=3, max_prefill_ms=1, max_states=2,
                diagnostics=diagnostics)
        assert diagnostics["complete"] is False
        assert diagnostics["refusal"] == "max_states"
    ties = [candidate("z", 1, 1), candidate("a", 1, 1)]
    resources = {(n, c.fmt): price(c, 1) for n in ("u", "v") for c in ties}
    for options in (ties, list(reversed(ties))):
        result = solver.solve_runtime_frontier({"v": options, "u": options}, resources,
            max_memory_bytes=2, max_prefill_ms=2, max_states=1)
        assert len(result) == 1
        assert result[0].assignment == {"u": "a", "v": "a"}


def test_transition_limit_does_not_return_a_partial_answer():
    c = candidate("a", 1, 1)
    resources = {(n, "a"): price(c, 1) for n in ("u", "v")}
    diagnostics = {}
    with pytest.raises(solver.RuntimeFrontierLimitError, match="max_transitions=1"):
        solver.solve_runtime_frontier({"u": [c], "v": [c]}, resources,
            max_memory_bytes=2, max_prefill_ms=2, max_transitions=1,
            diagnostics=diagnostics)
    assert diagnostics["complete"] is False
    assert diagnostics["refusal"] == "max_transitions"


@pytest.mark.parametrize("peak", ["peak_scratch_bytes", "activation_bytes"])
def test_later_peak_replacement_preserves_canonical_assignment_tie(peak):
    a, z, final = candidate("a", 1, 0), candidate("z", 1, 0), candidate("final", 1, 0)
    resources = {("u", c.fmt): price(c, 1) for c in (a, z)}
    resources[("v", final.fmt)] = price(final, 1)
    setattr(resources[("u", "a")], peak, 8)
    setattr(resources[("u", "z")], peak, 7)
    setattr(resources[("v", "final")], peak, 9)
    result = solver.solve_runtime_frontier({"u": [a, z], "v": [final]}, resources,
        max_memory_bytes=2, max_prefill_ms=2, max_device_bytes=9)
    assert [r.assignment for r in result] == [{"u": "a", "v": "final"}]


def test_float_sum_tie_is_canonical_after_strict_prefix_difference():
    a, z, final = candidate("a", 1, 1.0), candidate("z", 1, 0.0), candidate("final", 1, 2**54)
    resources = {(n, c.fmt): price(c, 1) for n, cs in (("u", [a, z]), ("v", [final])) for c in cs}
    result = solver.solve_runtime_frontier({"u": [a, z], "v": [final]}, resources,
        max_memory_bytes=2, max_prefill_ms=2)
    assert [r.assignment for r in result] == [{"u": "a", "v": "final"}]


def test_no_assignment_and_empty_menu_are_distinct():
    diagnostics = {}
    c = candidate("a", 2, 0)
    assert solver.solve_runtime_frontier({"u": [c]}, {("u", "a"): price(c, 1)},
        max_memory_bytes=1, max_prefill_ms=1, diagnostics=diagnostics) == []
    assert diagnostics["complete"] is True
    assert diagnostics["feasible"] is False
    with pytest.raises(ValueError, match="empty"):
        solver.solve_runtime_frontier({"u": []}, {}, max_memory_bytes=1, max_prefill_ms=1)
    empty = solver.solve_runtime_frontier({}, {}, max_memory_bytes=0, max_prefill_ms=0)
    assert empty[0].assignment == {}


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), True])
def test_invalid_quality_refuses(loss):
    c = candidate("a", 1, loss)
    with pytest.raises(ValueError, match="predicted_dloss"):
        solver.solve_runtime_frontier({"u": [c]}, {("u", "a"): price(c, 1)},
            max_memory_bytes=1, max_prefill_ms=1)


def test_duplicate_formats_refuse_ambiguous_runtime_key():
    c = candidate("a", 1, 1)
    with pytest.raises(ValueError, match="duplicate"):
        solver.solve_runtime_frontier({"u": [c, c]}, {("u", "a"): price(c, 1)},
            max_memory_bytes=1, max_prefill_ms=1)
