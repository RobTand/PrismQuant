"""Runtime search must receive alternatives before byte/loss pruning."""
import hashlib
import itertools

import pytest

from prismaquant import allocator_candidates as ac
from prismaquant import tessera_runtime_contract as trc
from prismaquant.allocator_solver import Candidate


def _candidate(index, size, loss):
    return Candidate(f"TESSERA_E4M3_K1_R{256 + index}", size / 100,
                     size, loss)


@pytest.mark.parametrize("precision", [None, 0.1])
def test_runtime_mode_retains_byte_loss_dominated_routes(precision):
    rows = [_candidate(0, 100, 1.0), _candidate(1, 100, 2.0),
            _candidate(2, 110, 3.0)]
    stats = {"linear": {"n_params": 800}}
    original = {"linear": rows}
    assert len(ac.reduce_continuous_menu(original, stats)["linear"]) == 1
    report = {}
    result = ac.reduce_continuous_menu(
        original, stats, bit_precision=precision, report=report,
        preserve_runtime_frontier=True)
    # Any of these recipes can be the fastest measured route. No latency
    # measurement has reached the reducer, so it cannot discard one.
    assert result["linear"] == rows
    assert report["runtime_frontier_preserved"] is True


def _licence():
    path = trc.contract_path()
    return trc._load_at(str(path), hashlib.sha256(path.read_bytes()).hexdigest(),
                        "runtime-preservation-test").fused_module


def test_runtime_group_fold_preserves_every_coherent_combination():
    members = ["q", "k"]
    rows = {name: [_candidate(0, 100, 1.0), _candidate(1, 100, 2.0)]
            for name in members}
    ordinary = ac.tessera_group_composites(
        members, rows, 1600, licence=_licence())
    assert len(ordinary) == 1
    report = {}
    options = ac.tessera_group_composites(
        members, rows, 1600, licence=_licence(), report=report,
        preserve_runtime_frontier=True)
    expected = {tuple(c.fmt for c in pair)
                for pair in itertools.product(*(rows[m] for m in members))}
    assert {tuple(c.member_formats[m] for m in members) for c in options} == expected
    assert len(options) == 4


def test_runtime_group_fold_refuses_cap_instead_of_dropping_alternatives(monkeypatch):
    monkeypatch.setattr(ac, "_GROUP_FOLD_MAX_PAIRS", 3)
    rows = {name: [_candidate(0, 100, 1.0), _candidate(1, 100, 2.0)]
            for name in ("q", "k")}
    with pytest.raises(AssertionError, match="guard"):
        ac.tessera_group_composites(
            list(rows), rows, 1600, licence=_licence(),
            preserve_runtime_frontier=True)


def test_runtime_aggregation_has_one_option_per_expanded_operator(monkeypatch):
    from types import SimpleNamespace
    from prismaquant import format_registry as fr, tessera_menu

    monkeypatch.setattr(tessera_menu, "fused_module_licence", _licence)
    members = ["layer.q_proj", "layer.k_proj"]
    rows = {name: [_candidate(0, 100, 1.0), _candidate(1, 100, 2.0)]
            for name in members}
    stats = {name: {"n_params": 800, "in_features": 20,
                    "out_features": 40, "h_trace": 1.0} for name in members}
    costs = {name: {c.fmt: {"predicted_dloss": c.predicted_dloss}
                   for c in options} for name, options in rows.items()}
    specs = [fr.get_format(c.fmt) for c in rows[members[0]]]
    profile = SimpleNamespace(fused_sibling_group=lambda name: "layer.qkv")
    grouped_stats, _, grouped = ac.aggregate_fused_siblings(
        stats, costs, specs, rows, profile, preserve_runtime_frontier=True)
    unit, options = next(iter(grouped.items()))
    recipes = [tuple(sorted(ac.expand_fused_sibling_assignment(
        {unit: candidate.fmt}, grouped_stats).items())) for candidate in options]
    # A uniform per-name choice and its composite twin execute the same
    # operator. Retaining both makes final runtime re-binding ambiguous.
    assert len(recipes) == len(set(recipes)) == 4
