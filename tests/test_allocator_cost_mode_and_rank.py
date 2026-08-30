"""UNWIRED_LINKS #7 and #5: the currency gate's input, and promotion's rank.

Two defects, both of the same shape -- a decision made against a value the
code could not actually see.

#7 ``allocator.py``: ``build_candidates`` was called with neither
``cost_mode=`` nor ``trellis_provenance=``.  The first made the trellis
surface's currency gate compare against ``os.environ.get("COST_MODE",
"aura")``, and ``run-pipeline.sh`` assigns ``COST_MODE`` with ``:=`` and never
exports it, so the gate read the default on every pipeline run.  The second
threw away the manifest identity, the anchor currency and the activation
contract the anchors were measured under, instead of letting them travel with
the assignment.

#5 ``allocator_solver.py``: ``promote_serving_units`` looked a format's rank
up with a bare subscript.  It survives today only because aggregation
guarantees a trellis rung reaches promotion as a lone ungrouped Linear, which
promotion skips.
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import pytest

from prismaquant import allocator
from prismaquant.allocator_solver import (
    Candidate,
    FormatRankUnknownError,
    promote_fused,
    promote_serving_units,
)

_REPO = Path(__file__).resolve().parents[1]


# ==========================================================================
# #7(a) -- the run's objective, resolved from data rather than the shell
# ==========================================================================
def test_resolve_run_cost_mode_prefers_the_explicit_flag():
    assert allocator.resolve_run_cost_mode(
        {"costs": {}}, flag="local", costs_path="cost.pkl") == "local"


def test_resolve_run_cost_mode_reads_the_cost_tables_own_stamp():
    """No flag: the objective the numbers in front of the DP were measured in."""

    payload = {"costs": {}, "provenance": {"cost_mode": "production-render-score"}}
    assert allocator.resolve_run_cost_mode(
        payload, flag=None, costs_path="cost.pkl"
    ) == "production-render-score"


def test_resolve_run_cost_mode_refuses_a_flag_that_contradicts_the_stamp():
    """One DP prices in one currency; the allocator does not trust the shell."""

    payload = {"costs": {}, "provenance": {"cost_mode": "local"}}
    with pytest.raises(SystemExit) as exc:
        allocator.resolve_run_cost_mode(
            payload, flag="aura", costs_path="/w/cost.pkl")
    message = str(exc.value)
    assert "'aura'" in message and "'local'" in message
    assert "/w/cost.pkl" in message


def test_resolve_run_cost_mode_is_none_when_nothing_declares_one():
    """A table predating the re-vet R2 stamp, run without the flag."""

    assert allocator.resolve_run_cost_mode(
        {"costs": {}}, flag=None, costs_path="cost.pkl") is None


def test_run_pipeline_hands_the_allocator_the_runs_cost_mode():
    """The plumbing half: COST_MODE is assigned with := and never exported.

    ``: "${COST_MODE:=aura}"`` is an assignment, and one that also PRESERVES
    an inherited export, so what a child sees depends on how the script was
    invoked.  Every other stage is fed the value explicitly; the allocator
    now is too.  If this assertion fails, the currency gate is back to
    reading a default.
    """

    script = (_REPO / "prismaquant/run-pipeline.sh").read_text()
    assert re.search(r'^:\s*"\$\{COST_MODE:=aura\}"', script, re.M), (
        "COST_MODE's default moved; re-check whether it is now exported"
    )
    assert not re.search(r"^\s*export\s+[A-Z_ ]*\bCOST_MODE\b", script, re.M), (
        "COST_MODE is exported now -- the flag is still the contract, but this "
        "test's premise changed and the ledger entry's wording should follow"
    )
    invocation = script.split("python3 -m prismaquant.allocator", 1)
    assert len(invocation) == 2, "the allocator invocation moved"
    # Up to the end of the piped command.
    body = invocation[1].split("logs/allocator.log", 1)[0]
    assert '--cost-mode "$COST_MODE"' in body


def _tiny_run(tmp_path, *, cost_provenance=None, extra_argv=(),
              with_lm_head=False):
    """A minimal probe/cost pair the allocator can actually solve."""

    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
    }))
    names = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    ]
    if with_lm_head:
        names.append("lm_head")
    stats, costs = {}, {}
    for idx, name in enumerate(names):
        stats[name] = {
            "h_trace": float(idx + 1),
            "n_params": 128 * 128,
            "in_features": 128,
            "out_features": 128,
        }
        costs[name] = {
            "NVFP4": {"predicted_dloss": 10.0 + idx},
            "FP8_DYNAMIC": {"predicted_dloss": 1.0 + 0.1 * idx},
            "BF16": {"predicted_dloss": 0.0},
        }
    # The byte-budget path sizes the non-quantizable floor from real shards.
    import torch
    from safetensors.torch import save_file

    save_file(
        {
            **{f"{name}.weight": torch.zeros(128, 128, dtype=torch.bfloat16)
               for name in names},
            "model.embed_tokens.weight": torch.zeros(
                256, 128, dtype=torch.bfloat16),
        },
        str(model_dir / "model.safetensors"),
    )

    probe_path = tmp_path / "probe.pkl"
    cost_path = tmp_path / "cost.pkl"
    with open(probe_path, "wb") as f:
        pickle.dump({"stats": stats, "meta": {"model": str(model_dir)}}, f)
    payload = {"costs": costs, "formats": ["NVFP4", "FP8_DYNAMIC", "BF16"]}
    if cost_provenance is not None:
        payload["provenance"] = cost_provenance
    with open(cost_path, "wb") as f:
        pickle.dump(payload, f)
    return [
        "--probe", str(probe_path),
        "--costs", str(cost_path),
        "--model-override", str(model_dir),
        "--formats", "NVFP4,FP8_DYNAMIC,BF16",
        "--target-bits", "6.0",
        "--bit-precision", "0.1",
        "--layer-config", str(tmp_path / "layer_config.json"),
        "--pareto-csv", str(tmp_path / "pareto.csv"),
        *extra_argv,
    ]


def _spy_on_the_seam(monkeypatch):
    """Record what ``build_candidates`` forwards to the trellis seam."""

    from prismaquant import trellis_menu

    seen: list[dict] = []
    real = trellis_menu.augment_candidates

    def spy(candidates, stats, **kwargs):
        seen.append(dict(kwargs))
        return real(candidates, stats, **kwargs)

    monkeypatch.setattr(trellis_menu, "augment_candidates", spy)
    return seen


def test_the_seam_is_told_the_runs_objective_not_the_environments(
        tmp_path, monkeypatch, capsys):
    """The defect itself: an environment value must not decide the currency.

    ``COST_MODE`` is set in the environment to the WRONG answer.  The cost
    table stamps the right one.  What reaches the seam must be the table's.
    """

    monkeypatch.setenv("COST_MODE", "aura")
    argv = _tiny_run(tmp_path, cost_provenance={"cost_mode": "local"})
    seen = _spy_on_the_seam(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["allocator", *argv])
    allocator.main()
    capsys.readouterr()

    assert seen, "the trellis seam was never reached from build_candidates"
    modes = {call["cost_mode"] for call in seen}
    assert modes == {"local"}, (
        f"the seam priced against {modes}; the environment said 'aura' and the "
        f"cost table said 'local'. The run's objective is the table's."
    )


def test_every_menu_the_run_builds_is_priced_in_the_same_currency(
        tmp_path, monkeypatch, capsys):
    """The body, head, MTP and visual menus all reach the same seam."""

    monkeypatch.setenv("COST_MODE", "aura")
    argv = _tiny_run(
        tmp_path,
        cost_provenance={"cost_mode": "local"},
        # A QUANTIZED head, so the second build_candidates call really runs.
        # A BF16 head is pinned and never builds a menu, which would make this
        # test pass while the head call site read the environment.
        extra_argv=("--lm-head-format", "FP8_DYNAMIC"),
        with_lm_head=True,
    )
    seen = _spy_on_the_seam(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["allocator", *argv])
    allocator.main()
    capsys.readouterr()
    assert len(seen) >= 2, (
        f"only {len(seen)} menu(s) were built; this test has to exercise the "
        f"head call site as well as the body one"
    )
    assert {call["cost_mode"] for call in seen} == {"local"}


def test_the_flag_beats_an_unstamped_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COST_MODE", "aura")
    argv = _tiny_run(tmp_path, extra_argv=("--cost-mode", "local"))
    seen = _spy_on_the_seam(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["allocator", *argv])
    allocator.main()
    capsys.readouterr()
    assert {call["cost_mode"] for call in seen} == {"local"}


def test_a_surface_run_that_cannot_name_its_objective_is_refused(
        tmp_path, monkeypatch):
    """Unstamped table, no flag, surface on: refuse instead of guessing."""

    manifest = tmp_path / "surface.json"
    manifest.write_text("{}")
    monkeypatch.setenv("PRISMAQUANT_TRELLIS_SURFACE", str(manifest))
    monkeypatch.delenv("COST_MODE", raising=False)
    argv = _tiny_run(tmp_path)
    monkeypatch.setattr(sys, "argv", ["allocator", *argv])
    with pytest.raises(SystemExit) as exc:
        allocator.main()
    message = str(exc.value)
    assert "PRISMAQUANT_TRELLIS_SURFACE" in message
    assert "--cost-mode" in message
    assert "provenance['cost_mode']" in message


def test_an_unstamped_table_without_the_surface_is_not_refused(
        tmp_path, monkeypatch, capsys):
    """Principle 6: the refusal is scoped to the consumer that needs an answer."""

    monkeypatch.delenv("PRISMAQUANT_TRELLIS_SURFACE", raising=False)
    monkeypatch.delenv("COST_MODE", raising=False)
    argv = _tiny_run(tmp_path)
    monkeypatch.setattr(sys, "argv", ["allocator", *argv])
    allocator.main()
    capsys.readouterr()
    assert (tmp_path / "layer_config.json").exists()


# ==========================================================================
# #7(b) -- the surface's identity travels with the assignment
# ==========================================================================
_STAMP = {
    "schema": "prismaquant.trellis_menu_provenance.v1",
    "manifest_path": "/campaign/surface.json",
    "cost_mode": "local",
    "currency": "weighted_sse",
    "target_profile": "gridbook_sm121",
    "target_platform": "sm121",
    "anchor_activation_contract": "W*A16",
    "candidates_added": 7,
    "research_only": True,
    "exportable": False,
}


def _stamping_seam(monkeypatch, payload):
    """Make the seam yield a stamp, the way a wired surface would."""

    from prismaquant import trellis_menu

    def stamper(candidates, stats, *, cost_mode, manifest_path=None,
                provenance_out=None):
        if provenance_out is not None:
            provenance_out.update(payload)
        return candidates

    monkeypatch.setattr(trellis_menu, "augment_candidates", stamper)


def test_the_surfaces_identity_and_anchor_contract_reach_the_artifacts(
        tmp_path, monkeypatch, capsys):
    """P12/P14: the numbers must carry what they were measured on.

    A stamp the menu produced used to be discarded at the call site.  It now
    lands in both artifacts a reader or an exporter picks the assignment up
    from.
    """

    argv = _tiny_run(
        tmp_path,
        cost_provenance={"cost_mode": "local"},
        extra_argv=("--target-disk-gb", "0.01",
                    "--artifact-overhead-reserve-bytes", "4096"),
    )
    _stamping_seam(monkeypatch, _STAMP)
    monkeypatch.setattr(sys, "argv", ["allocator", *argv])
    allocator.main()
    capsys.readouterr()

    layer_config = json.loads((tmp_path / "layer_config.json").read_text())
    selection = json.loads((tmp_path / "selection.json").read_text())
    for where, payload in (("layer_config.json", layer_config["__prismaquant__"]),
                           ("selection.json", selection)):
        stamp = payload.get("trellis_surface")
        assert stamp is not None, f"{where} dropped the trellis surface stamp"
        # The three the ledger entry names, by name, as structured values.
        assert stamp["manifest_path"] == _STAMP["manifest_path"], where
        assert stamp["currency"] == _STAMP["currency"], where
        assert (stamp["anchor_activation_contract"]
                == _STAMP["anchor_activation_contract"]), where
        assert stamp["target_platform"] == _STAMP["target_platform"], where


def test_no_surface_means_no_stamp_at_all(tmp_path, monkeypatch, capsys):
    """Principle 6: an unset run's artifacts must not gain an empty key."""

    monkeypatch.delenv("PRISMAQUANT_TRELLIS_SURFACE", raising=False)
    argv = _tiny_run(
        tmp_path,
        cost_provenance={"cost_mode": "local"},
        extra_argv=("--target-disk-gb", "0.01",
                    "--artifact-overhead-reserve-bytes", "4096"),
    )
    monkeypatch.setattr(sys, "argv", ["allocator", *argv])
    allocator.main()
    capsys.readouterr()
    layer_config = json.loads((tmp_path / "layer_config.json").read_text())
    selection = json.loads((tmp_path / "selection.json").read_text())
    assert "trellis_surface" not in layer_config["__prismaquant__"]
    assert "trellis_surface" not in selection


# ==========================================================================
# #5 -- promotion's rank: derived where derivable, refused where not
# ==========================================================================
_QKV = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
]


def test_promotion_refuses_a_format_the_rank_table_does_not_cover():
    """The latent crash links 3/4 are about to make live.

    A fused sibling group holding a rung the table has never heard of has no
    defined 'most expensive member'.  Ranking it anyway silently reorders the
    promotion decision and every member ships in whatever that wrong order
    picked, reporting nothing.
    """

    assignment = {
        _QKV[0]: "TCQ_E2M1_R640",
        _QKV[1]: "NVFP4",
        _QKV[2]: "NVFP4",
    }
    rank = {"NVFP4": 0, "FP8_DYNAMIC": 1, "BF16": 2}
    with pytest.raises(FormatRankUnknownError) as exc:
        promote_serving_units(assignment, rank)
    message = str(exc.value)
    assert "TCQ_E2M1_R640" in message                  # the format
    assert _QKV[1] in message                          # the serving unit
    assert "extend_format_rank_from_candidates" in message   # the missing input
    assert "['NVFP4', 'FP8_DYNAMIC', 'BF16']" in message     # what it did know


def test_promote_fused_refuses_on_the_same_terms():
    """The fused entry point refuses too, via its component pass.

    ``promote_fused``'s own legacy per-group repass also had a bare subscript
    and is guarded now, but it can never be the first to see an unknown
    format: it runs AFTER ``promote_serving_units``, whose connected
    components are supersets of the same groups. So this pins the reachable
    behaviour -- ``promote_fused`` refuses -- and the repass guard stays as
    defence in depth that nothing can reach first.
    """

    assignment = {name: "TCQ_E4M3_R1024" for name in _QKV}
    assignment[_QKV[0]] = "NVFP4"
    with pytest.raises(FormatRankUnknownError) as exc:
        promote_fused(assignment, {"NVFP4": 0, "BF16": 1})
    assert "TCQ_E4M3_R1024" in str(exc.value)


def test_a_lone_ungrouped_unit_is_still_skipped():
    """Why the crash was latent: promotion never ranks a one-member unit."""

    assignment = {"model.layers.0.mlp.down_proj": "TCQ_E2M1_R640"}
    out = promote_serving_units(assignment, {"NVFP4": 0, "BF16": 1})
    assert out == assignment


def _candidate(fmt, *, memory_bytes, bits):
    return Candidate(fmt=fmt, bits_per_param=bits, memory_bytes=memory_bytes,
                     predicted_dloss=0.0)


def test_the_rank_is_derived_from_the_candidates_exact_bytes():
    """Not a sentinel: a rung's rate comes from the bytes it declares.

    Two rungs at 2 and 6 bits per weight over a 128x128 unit must land below
    and above the 4-bit scalar format, from their bytes alone.
    """

    n_params = 128 * 128
    stats = {"u": {"in_features": 128, "out_features": 128, "n_params": n_params}}
    menus = {"u": [
        _candidate("NVFP4", memory_bytes=n_params // 2, bits=4.0),
        _candidate("BF16", memory_bytes=n_params * 2, bits=16.0),
        _candidate("TCQ_E2M1_R512", memory_bytes=n_params // 4, bits=2.0),
        _candidate("TCQ_E4M3_R1536", memory_bytes=(n_params * 6) // 8, bits=6.0),
    ]}
    extended = allocator.extend_format_rank_from_candidates(
        {"NVFP4": 0, "BF16": 1}, {"NVFP4": 4.0, "BF16": 16.0}, menus, stats)
    order = sorted(extended, key=lambda name: extended[name])
    assert order == ["TCQ_E2M1_R512", "NVFP4", "TCQ_E4M3_R1536", "BF16"]


def test_a_derived_rank_makes_promotion_stop_refusing():
    """The two halves meet: the derivation supplies what the refusal names."""

    n_params = 128 * 128
    stats = {name: {"in_features": 128, "out_features": 128,
                    "n_params": n_params} for name in _QKV}
    menus = {name: [
        _candidate("NVFP4", memory_bytes=n_params // 2, bits=4.0),
        _candidate("BF16", memory_bytes=n_params * 2, bits=16.0),
        _candidate("TCQ_E4M3_R1536", memory_bytes=(n_params * 6) // 8, bits=6.0),
    ] for name in _QKV}
    extended = allocator.extend_format_rank_from_candidates(
        {"NVFP4": 0, "BF16": 1}, {"NVFP4": 4.0, "BF16": 16.0}, menus, stats)
    assignment = {_QKV[0]: "TCQ_E4M3_R1536", _QKV[1]: "NVFP4", _QKV[2]: "NVFP4"}
    out = promote_serving_units(assignment, extended)
    # The 6-bit rung outranks the 4-bit scalar, so the whole unit takes it.
    assert set(out.values()) == {"TCQ_E4M3_R1536"}


def test_extending_a_scalar_only_menu_changes_nothing():
    """Principle 6: a run whose menu adds nothing gets the same table back."""

    n_params = 128 * 128
    stats = {"u": {"in_features": 128, "out_features": 128, "n_params": n_params}}
    menus = {"u": [
        _candidate("NVFP4", memory_bytes=n_params // 2, bits=4.0),
        _candidate("BF16", memory_bytes=n_params * 2, bits=16.0),
    ]}
    before = {"NVFP4": 0, "BF16": 1}
    assert allocator.extend_format_rank_from_candidates(
        before, {"NVFP4": 4.0, "BF16": 16.0}, menus, stats) == before


def test_a_disagreement_about_the_existing_order_is_refused():
    """The two orderings must be one measurement, not two."""

    n_params = 128 * 128
    stats = {"u": {"in_features": 128, "out_features": 128, "n_params": n_params}}
    menus = {"u": [_candidate("TCQ_E2M1_R512",
                              memory_bytes=n_params // 4, bits=2.0)]}
    with pytest.raises(SystemExit) as exc:
        allocator.extend_format_rank_from_candidates(
            {"NVFP4": 0, "BF16": 1},
            {"NVFP4": 16.0, "BF16": 4.0},   # rates contradict the ranks
            menus, stats)
    assert "REORDER" in str(exc.value)


def test_a_format_on_units_with_no_readable_shape_is_refused_not_ranked():
    """No parameter count, no rate, no rank -- and no guess."""

    menus = {"u": [_candidate("TCQ_E2M1_R512", memory_bytes=4096, bits=2.0)]}
    with pytest.raises(SystemExit) as exc:
        allocator.extend_format_rank_from_candidates(
            {"NVFP4": 0}, {"NVFP4": 4.0}, menus, {"u": {"n_params": 4096}})
    assert "TCQ_E2M1_R512" in str(exc.value)


def test_a_real_run_extends_its_rank_table_from_the_built_menus(
        tmp_path, monkeypatch, capsys):
    """End to end: the extension runs before anything reads the table.

    Injects one extra rung into the built menu the way a wired surface would,
    and checks the run completes -- i.e. promotion found a rank for it rather
    than raising.
    """

    from prismaquant import allocator_candidates as ac

    real = ac.build_candidates
    n_params = 128 * 128

    def with_an_extra_rung(stats, costs, formats, *args, **kwargs):
        out = real(stats, costs, formats, *args, **kwargs)
        for name, menu in out.items():
            if not name.endswith((".q_proj", ".k_proj", ".v_proj")):
                continue
            menu.append(_candidate("TCQ_E4M3_R1536",
                                   memory_bytes=(n_params * 6) // 8, bits=6.0))
        return out

    monkeypatch.setattr(allocator, "build_candidates", with_an_extra_rung)
    argv = _tiny_run(tmp_path, cost_provenance={"cost_mode": "local"})
    monkeypatch.setattr(sys, "argv", ["allocator", *argv])
    allocator.main()
    out = capsys.readouterr().out
    assert "format rank extended with" in out
    assert "TCQ_E4M3_R1536" in out
    assert (tmp_path / "layer_config.json").exists()
