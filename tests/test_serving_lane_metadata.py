"""Serving-lane metadata on allocator candidates — what survives P5b.

Rewritten 2026-09-02, when the Gridbook codebook (NVFP4-CB / FP8-CB) serving
lane was retired to ``archive/gridbook_lane_2026-09-02/``.

This file used to pin four things, all of them through the ``nvfp4_cb``
serving profile: the two per-grid N-dimension load gates, the fused-mid-M
backed rung set as version-keyed spec data, the route travelling on every
``Candidate``, and the backed/fallback split in the shipped
``selection.json``.  ``prismaquant/serving_profile_specs/nvfp4_cb.json`` was
the ONLY spec in the repo that ever declared a ``serving_lanes`` block, so
the first three lost their subject outright when it went: no surviving
profile (gguf, research, tessera_research_sm121,
vllm_glm5_next_packed_moe, vllm_packed_moe, vllm_qwen3_5_packed_moe) declares
a lane, a
``route_status``, an ``activation_contract`` or a ``fused_mid_m`` set.  That
is a recorded capability loss (docs/ARCHITECTURE.md debt D34), not a bug to
route around, and the deleted tests below say so in place rather than being
re-pointed at a profile that would answer them vacuously.

What still has a subject, and is what this file now pins:

1. ``serving_lane_route`` returns ``None`` for a profile that declares no
   lane — which is today every profile.
2. The end-to-end ``selection.json`` / ``format_applicability.json`` blocks:
   the P5a activation-fair-pricing verdict, the republished cross-family
   ladder verdict, and a serving-lane provenance report in which every
   selected unit is honestly laneless.

The CB format/cost/render plumbing (``cb_layout``, ``nvfp4_cb_formats``,
``format_registry``'s ``*_CB_*`` FormatSpecs) was deliberately kept, so CB
rungs are still registered and still priceable; nothing offers or ships them.
"""
from __future__ import annotations

import json
import pickle
import struct
import sys

import pytest

import prismaquant.allocator as alloc
from prismaquant import footprint as fp
from prismaquant import serving_profiles as sp


# ---------------------------------------------------------------------------
# 1. The N-dimension load gates  — DELETED 2026-09-02
# ---------------------------------------------------------------------------
#
# Eight tests (23 parametrized cases) are gone from here:
#
#   test_fp4_cb_rungs_load_on_out_features_multiple_of_8      (5 params)
#   test_fp4_cb_rungs_are_masked_off_a_non_multiple_of_8      (5 params)
#   test_fp8_cb_rungs_load_on_out_features_multiple_of_16     (4 params)
#   test_fp8_cb_rungs_are_masked_off_a_non_multiple_of_16     (5 params)
#   test_the_two_families_load_gates_really_are_different
#   test_the_load_gates_are_declared_per_grid_from_the_family_table
#   test_the_load_gates_do_not_touch_the_native_carriers
#   test_the_superblock_gate_is_unchanged
#
# They asserted that the fp4-CB family loads only on ``out_features % 8 == 0``
# and the fp8-CB family only on ``out_features % 16 == 0``; that the two gates
# are genuinely different at N=8; that both are derived from
# ``cb_layout.FAMILIES`` rather than hand-listed; that the gates leave the
# native NVFP4 / FP8_DYNAMIC / BF16 carriers alone; and that the older
# ``in_features % 256`` superblock gate was not relaxed by the others.
#
# All three rules --- cb_superblock_shape, cb_fp4_out_features_load_gate,
# cb_fp8_out_features_load_gate --- were declared in
# prismaquant/serving_profile_specs/nvfp4_cb.json and went with it to
# archive/gridbook_lane_2026-09-02/.  No surviving profile declares any of
# them, so there is no gate left to assert about.
#
# The two "loads on a legal N" tests (nine cases) and the native-carrier one
# were still GREEN at deletion --- ten of the thirteen this file still passed
# --- which is exactly why they had to go too:
# ``check_serving_shape`` catches ``FileNotFoundError`` and silently resolves
# an unknown profile id to ``research``, which permits every shape.  They were
# passing on permit-all, not on a gate --- a tautology wearing a gate's
# docstring.
#
# The tail of test_the_load_gates_are_declared_per_grid_from_the_family_table
# also asserted the ladder partition
# (``NVFP4_CB_FORMAT_NAMES | FP8_CB_FORMAT_NAMES == CB_FORMAT_NAMES``, and the
# two disjoint).  That is still covered: tests/test_cb_layout.py pins each
# family against the registry (``list_producer_formats("fp8_cb")`` /
# ``("nvfp4_cb")``) and their union against ``cb_layout.CB_FORMAT_NAMES``.


# ---------------------------------------------------------------------------
# 2. The fused-lane route as versioned spec data  — DELETED 2026-09-02
# ---------------------------------------------------------------------------
#
# Eight tests are gone from here:
#
#   test_backed_fused_mid_m_rungs_are_spec_data_for_the_pinned_runtime
#   test_the_0_6_0_backed_set_is_the_k_mod_4_law_not_a_missing_five
#   test_advancing_the_pin_adds_a_version_key_and_never_edits_an_old_one
#   test_block128_source_lane_is_w8a16_and_direct_g32_remains_w8a8
#   test_the_fp4_opt_in_fused_mid_m_lane_is_available_not_backed
#   test_an_undeclared_runtime_version_backs_nothing
#   test_fp4_cb_declares_the_bf16_bridge_and_no_fused_mid_m_lane
#   test_the_lane_catalog_is_reportable_and_names_its_runtime
#
# Between them they pinned: that FP8_CB's backed fused-mid-M set is version
# keyed and equals ``range(28, 49, 4)`` under every declared Gridbook release
# 0.5.0..0.8.5; that the set is the ``k % 4 == 0`` TMA/sub-table law rather
# than an incomplete instantiation list; that advancing the pin ADDS a version
# key and never edits an old one, so an artifact stays resolvable at the route
# it shipped on; that FP8_BLOCK_UE8M0_SOURCE serves W8A16 while
# MXFP8_UE8M0_G32 serves W8A8; that the fp4-CB fused mid-M kernel is AVAILABLE
# behind PRISMAQUANT_CB_FP4_FUSED_MIDM but not BACKED; that an undeclared
# runtime version backs nothing (fail-closed); that every fp4-CB rung rides
# the ``w4-bf16-bridge`` fallback; and that the whole lane catalog is
# reportable and names its runtime.
#
# Every one of those facts lived in the ``serving_lanes`` block of
# prismaquant/serving_profile_specs/nvfp4_cb.json, the only spec in the repo
# that ever carried one (archive/gridbook_lane_2026-09-02/).  With it gone the
# structured per-lane route_status / activation_contract / fused_mid_m table
# has ZERO live declarations, ``sp.gridbook_runtime_version`` no longer
# exists, and there is nothing left for these to read.  They are deleted
# rather than re-pointed: re-pointing at a profile that declares no lane would
# convert a capability loss into a green assertion.


def test_profiles_without_a_declared_lane_return_none():
    """`research` describes emulation legality, not a served container.

    Since 2026-09-02 this is the general case, not the exception: no serving
    profile declares a lane at all (archive/gridbook_lane_2026-09-02/).  The
    companion assertion that plain NVFP4 has no CB lane under ``nvfp4_cb`` was
    dropped with the profile --- it would now resolve through
    ``serving_lane_route``'s ``FileNotFoundError -> None`` branch and pass for
    the wrong reason.
    """
    assert sp.serving_lane_route("research", "FP8_CB_K36") is None
    assert sp.serving_lane_route("research", "NVFP4") is None


# ---------------------------------------------------------------------------
# 3. The route travels with the candidate  — DELETED 2026-09-02
# ---------------------------------------------------------------------------
#
# Three tests are gone from here:
#
#   test_candidates_carry_the_concrete_serving_lane_route
#   test_selection_provenance_splits_backed_rungs_from_the_fallback
#   test_selection_provenance_reads_the_route_off_the_chosen_candidate
#
# They pinned that ``build_candidates`` attaches a resolved route to every
# Candidate and to the ``_serving_lane_by_format`` side map; that
# ``selection_serving_lane_provenance`` splits the selected rungs into
# backed-fused / fallback / no-declared-lane buckets with a per-format
# breakdown and an activation-contract histogram; and that the provenance
# reads a route off the chosen candidate first, re-resolving from the profile
# only for expanded members of aggregated super items.
#
# All three drove ``target_profile="nvfp4_cb"``.  With no profile declaring a
# lane, every candidate's ``serving_lane`` is ``None`` and every bucket but
# ``units_without_declared_lane`` is empty by construction, so the split these
# tests exist to check has no subject.  The laneless case is still covered
# end to end by test_selection_json_carries_the_p5a_and_p5b_provenance below.
#
# ONE live assertion went with them: test_selection_provenance_reads_the_route
# _off_the_chosen_candidate also pinned the ``activation_pricing_branches``
# stamping for an expanded member with no candidate of its own
# (``branches["unrecorded"] == 1``, total == 3).  That code path
# (prismaquant/allocator_candidates.py, the branch_counts accumulation behind
# "activation_pricing_branches") is profile independent and still works under
# ``research``, so it is RE-HOMED below rather than smuggled into a renamed
# test -- as its own test, against the function directly, with no CB profile
# in the frame.


def test_a_unit_with_no_candidate_is_stamped_unrecorded_not_omitted():
    """``activation_pricing_branches`` must account for every selected unit.

    An expanded member of an aggregated super item has no ``Candidate`` of its
    own, so nothing recorded which estimator priced its activation side.  The
    honest answer is the ``unrecorded`` bucket: a census that silently dropped
    those rows would make the branch histogram sum to fewer units than the
    assignment, and "no row" would read as "no activation cost" -- the same
    absence-as-evidence error ``units_without_declared_lane`` exists to stop.

    This is the assertion that rode the deleted CB test above; it is profile
    independent, so it is pinned here against ``research``.
    """
    from prismaquant.allocator_candidates import (
        selection_serving_lane_provenance,
    )

    assignment = {
        "model.layers.0.mlp.gate_proj": "NVFP4",
        "model.layers.0.mlp.up_proj": "NVFP4",
        "model.layers.0.mlp.down_proj": "FP8_E4M3",
    }
    report = selection_serving_lane_provenance(
        assignment, candidates=None, target_profile="research")

    branches = report["activation_pricing_branches"]
    assert branches == {"unrecorded": 3}
    assert sum(branches.values()) == report["units_total"] == len(assignment)
    # `research` declares no lane, so the route census says so rather than
    # reporting a clean bill.
    assert report["units_without_declared_lane"] == len(assignment)
    assert report["route_status_counts"] == {"no_declared_lane": 3}
    assert report["route_status_attested"] is False


# ---------------------------------------------------------------------------
# 4. End to end: the shipped selection.json
# ---------------------------------------------------------------------------
#
# Same synthetic-checkpoint + stubbed-solver harness as
# tests/test_allocator_byte_budget_selection.py, so what is pinned is the code
# that ships selections rather than a helper beside it.

_NAMES = [f"model.layers.{i}.self_attn.o_proj" for i in range(4)]
_OUT = _IN = 256
_OVERHEAD_RESERVE = 512
_FLOOR_TENSORS = {
    "model.embed_tokens.weight": ("BF16", (512, 64)),
    "lm_head.weight": ("BF16", (512, 64)),
    "model.norm.weight": ("BF16", (64,)),
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


def _fixture(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tensors = dict(_FLOOR_TENSORS)
    for n in _NAMES:
        tensors[f"{n}.weight"] = ("BF16", (_OUT, _IN))
    _write_safetensors(model_dir / "model-00001.safetensors", tensors)
    stats = {
        n: {"h_trace": 1.0 + 0.1 * i, "n_params": _OUT * _IN,
            "in_features": _IN, "out_features": _OUT}
        for i, n in enumerate(_NAMES)
    }
    probe = {"stats": stats, "meta": {"model": str(model_dir)}}
    costs = {
        "costs": {
            n: {
                # Measured dense rows for one family; weight-only rows for
                # the other, i.e. the mixed-estimator shape P5a exists for.
                "NVFP4": {"weight_mse": 1e-4, "output_mse": 4e-4,
                          "output_mse_measured": True},
                "FP8_E4M3": {"weight_mse": 1e-6, "output_mse": 2e-6,
                             "output_mse_measured": True},
            }
            for n in _NAMES
        },
        "meta": {"formats": ["NVFP4", "FP8_E4M3"]},
        "provenance": {
            "cb_ladder_cross_family_verdict": {
                "schema": "prismaquant.cb_ladder.cross_family_verdict.v1",
                "verdict": "fail",
                "cross_family_comparison_publishable": False,
                "detail": "synthetic asymmetric bands",
            }
        },
    }
    probe_p = tmp_path / "probe.pkl"
    cost_p = tmp_path / "cost.pkl"
    probe_p.write_bytes(pickle.dumps(probe))
    cost_p.write_bytes(pickle.dumps(costs))
    return model_dir, probe_p, cost_p, stats


def _stub_solver(fmt):
    def solve(stats, candidates, target_bits, format_specs, format_rank,
              bit_precision, **kw):
        assign = {n: fmt for n in candidates}
        total_params = sum(stats[n]["n_params"] for n in assign)
        bits = sum(
            8.0 * next(c for c in candidates[n] if c.fmt == fmt).memory_bytes
            for n in assign
        )
        achieved = bits / max(total_params, 1)
        diag = kw.get("diagnostics")
        if diag is not None:
            diag.update({"feasible": True, "achieved_bits": achieved,
                         "predicted_dloss": None, "evals": 1})
        return assign, achieved
    return solve


def test_selection_json_carries_the_p5a_and_p5b_provenance(
        monkeypatch, tmp_path):
    """The two P5 verdicts and the serving-lane split must be recoverable
    from the shipped artifact, not from the producer commit that made it."""
    model_dir, probe_p, cost_p, stats = _fixture(tmp_path)
    monkeypatch.setattr(alloc, "solve_with_promotion", _stub_solver("NVFP4"))
    lc = tmp_path / "layer_config.json"
    csv = tmp_path / "pareto.csv"
    monkeypatch.setattr(sys, "argv", [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", "NVFP4,FP8_E4M3",
        "--pareto-targets", "4.6,8.2",
        "--target-disk-gb", repr(1.0),
        "--layer-config", str(lc),
        "--pareto-csv", str(csv),
        "--artifact-overhead-reserve-bytes", str(_OVERHEAD_RESERVE),
        "--allow-default-profile",
    ])
    alloc.main()
    selection = json.loads((tmp_path / "selection.json").read_text())

    pricing = selection["activation_fair_pricing"]
    assert pricing["schema"] == "prismaquant.activation_fair_pricing.v1"
    assert pricing["env_flag"] == "PRISMAQUANT_ACTIVATION_FAIR_PRICING"
    assert pricing["functional_form"].startswith("per_family_multiplicative")
    # Both act-quantizing families have measured rows here, so both calibrate
    # and no weight-only row is left uncorrected.
    assert set(pricing["families"]) == {"nv", "fp"}
    assert pricing["families"]["nv"]["penalty"] == pytest.approx(4.0)
    assert pricing["families"]["fp"]["penalty"] == pytest.approx(2.0)
    assert pricing["uncalibrated_families"] == []

    # The cost run's cross-family verdict is republished verbatim, failure
    # and all — a consumer reading only allocator artifacts can still see it.
    verdict = selection["cb_ladder_cross_family_verdict"]
    assert verdict["verdict"] == "fail"
    assert verdict["cross_family_comparison_publishable"] is False

    lanes = selection["serving_lane_provenance"]
    assert lanes["schema"] == sp.SERVING_LANE_SCHEMA
    assert lanes["units_total"] == len(_NAMES)
    # `research` (the default profile here) declares no CB lane, so every
    # selected unit reports laneless rather than claiming a fast path.
    assert lanes["units_without_declared_lane"] == len(_NAMES)
    assert lanes["selected_rungs_fused_mid_m_backed"] == []
    assert set(lanes["activation_pricing_branches"]) == {
        "measured_output_mse"}


def test_the_applicability_report_carries_the_same_three_blocks(
        monkeypatch, tmp_path):
    """The byte-budget selection is optional; the applicability report is
    written on every run, so the P5 verdicts must live there too."""
    _model_dir, probe_p, cost_p, _stats = _fixture(tmp_path)
    monkeypatch.setattr(alloc, "solve_with_promotion", _stub_solver("NVFP4"))
    lc = tmp_path / "layer_config.json"
    csv = tmp_path / "pareto.csv"
    monkeypatch.setattr(sys, "argv", [
        "allocator",
        "--probe", str(probe_p),
        "--costs", str(cost_p),
        "--formats", "NVFP4,FP8_E4M3",
        "--pareto-targets", "4.6",
        "--target-bits", "4.6",
        "--layer-config", str(lc),
        "--pareto-csv", str(csv),
        "--allow-default-profile",
    ])
    alloc.main()
    report = json.loads(
        (tmp_path / "format_applicability.json").read_text())
    assert report["activation_fair_pricing"]["enabled"] is True
    assert (report["cb_ladder_cross_family_verdict"]["verdict"] == "fail")
    assert report["serving_lanes"]["target_profile"] == "research"


# ---------------------------------------------------------------------------
# The parser that has nothing left to parse (capability loss 3, D34)
# ---------------------------------------------------------------------------
# REGRESSION GUARD, NOT A FAIL-BEFORE. This passes on the commit that retired
# the lane too. It exists because that retirement left the `serving_lanes`
# parser with ZERO live declarations: `serving_profile_specs/nvfp4_cb.json`
# was the only spec that ever carried one, and the parser was deliberately
# kept because it is the shape the Tessera serving profile must declare in.
# A parser with no input and no test is a parser that rots silently and is
# discovered broken by the first profile that needs it. This gives it one
# input, built here rather than shipped as a spec, so nothing in the tree
# claims a lane that no runtime serves.
def test_the_serving_lanes_parser_still_parses_a_declared_lane():
    """One synthetic declaration, exercising every structured field a gate
    reads: the route status source's ``structures`` (principle 9 wants
    ``route_status`` in a field, never in prose), the activation contract,
    the fallback route, and the version-keyed fused mid-M table whose empty
    resolution is the designed fail-closed answer."""
    profile = sp.ServingProfile.from_dict({
        "id": "synthetic_lane_parser_probe",
        "description": "test-local; never written to serving_profile_specs/",
        "emulation_only": True,
        "serving_lanes": [
            {
                "id": "synthetic_native_lane",
                "formats": ["NVFP4"],
                "activation_contract": "w4a4-nvfp4-e2m1-fp8-block16",
                "fallback_route": "expand_and_gemm",
                "detail": "synthetic; asserts the parser, not a runtime",
                "fused_mid_m": {
                    "m_range": [8, 64],
                    "rungs_by_runtime_version": {"9.9.9": [28, 32]},
                },
                "route_status_source": {
                    "attestation": "synthetic.lane_eligibility",
                    "structures": ["dense", "routed_moe"],
                },
            }
        ],
    })

    assert len(profile.serving_lanes) == 1
    lane = profile.serving_lanes[0]
    assert lane.id == "synthetic_native_lane"
    assert lane.formats == ("NVFP4",)
    assert lane.activation_contract == "w4a4-nvfp4-e2m1-fp8-block16"
    assert lane.fallback_route == "expand_and_gemm"
    assert lane.fused_mid_m_range == (8, 64)
    assert lane.fused_mid_m_rungs_by_runtime_version == (("9.9.9", (28, 32)),)
    assert lane.route_status_structures == ("dense", "routed_moe")

    # and the lane resolves for a format it covers, and only for one it covers
    assert lane.covers("NVFP4")
    assert not lane.covers("FP8_DYNAMIC")
    resolved = profile.serving_lane_for("NVFP4", runtime_version="9.9.9")
    assert resolved is not None
    assert resolved.activation_contract == "w4a4-nvfp4-e2m1-fp8-block16"
    # An UNDECLARED runtime version backs nothing -- the fail-closed direction
    # the docstring on ServingLaneSpec calls out.
    other = profile.serving_lane_for("NVFP4", runtime_version="0.0.0")
    assert other is not None
    assert not other.fused_mid_m_backed
