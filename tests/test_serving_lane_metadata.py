"""Gridbook serving eligibility as candidate metadata — ultraplan P5b.

Gridbook's ``docs/audits/ultraplan_perf_2026-08-01.md`` §6 ("Three cost-model
asymmetries", #3): *the producer models exactly one gridbook kernel gate*
(``K % 256`` for CB). "It knows nothing of the ``N % 8`` (FP4) / ``N % 16``
(FP8) load gates, ``n_sub``, the fused mid-M rung set, or LUT residency — it
can legally assign a rung whose fast serving lane does not exist. (The 27B
ladder pricing five unbacked fused-mid-M rungs, gridbook K1.2, is the same
defect seen from the other end.)"

What this file pins:

1. The two N-dimension load gates are declared per GRID and they are
   DIFFERENT (8 for the fp4-CB families, 16 for fp8-CB), including at the
   boundary values where they disagree.
2. The backed fused-mid-M rung set is spec DATA keyed by the pinned Gridbook
   runtime version, and an undeclared version backs nothing (fail-closed).
3. The route — activation contract, backed/unbacked, fallback — is attached
   to every ``Candidate`` and survives aggregation.
4. The shipped ``selection.json`` records which selected rungs ride a backed
   fused lane and which take the expand+GEMM fallback, alongside the P5a
   activation-pricing verdict and the cross-family ladder verdict.

Deliberately NOT pinned here: any latency constraint in the solver. That is
P5c; this item is metadata only and ``solve_allocation``'s DP semantics are
untouched.
"""
from __future__ import annotations

import json
import pickle
import struct
import sys

import pytest

import prismaquant.allocator as alloc
from prismaquant import cb_layout
from prismaquant import footprint as fp
from prismaquant import format_registry as fr
from prismaquant import serving_profiles as sp
from prismaquant.allocator_candidates import (
    build_candidates,
    selection_serving_lane_provenance,
)
from prismaquant.nvfp4_cb_footprint import CBSerializationContext

_CB_CONTEXT = CBSerializationContext.production()
_PROFILE = "nvfp4_cb"


def _shape_decision(fmt, *, out_features, in_features=1024):
    return sp.check_serving_shape(
        _PROFILE, fmt, in_features=in_features, out_features=out_features)


# ---------------------------------------------------------------------------
# 1. The N-dimension load gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("out_features", [8, 16, 24, 512, 4096])
def test_fp4_cb_rungs_load_on_out_features_multiple_of_8(out_features):
    for fmt in ("NVFP4_CB_K12", "NVFP4_CB_K24", "NVFP4_CB_S16"):
        assert _shape_decision(fmt, out_features=out_features).legal


@pytest.mark.parametrize("out_features", [4, 7, 12, 20, 4095])
def test_fp4_cb_rungs_are_masked_off_a_non_multiple_of_8(out_features):
    for fmt in ("NVFP4_CB_K12", "NVFP4_CB_K24"):
        decision = _shape_decision(fmt, out_features=out_features)
        assert not decision.legal
        assert decision.rule == "cb_fp4_out_features_load_gate"
        assert decision.reason == "kernel_shape"
        assert f"out_features={out_features}" in decision.detail


@pytest.mark.parametrize("out_features", [16, 32, 512, 4096])
def test_fp8_cb_rungs_load_on_out_features_multiple_of_16(out_features):
    for fmt in ("FP8_CB_K28", "FP8_CB_K48"):
        assert _shape_decision(fmt, out_features=out_features).legal


@pytest.mark.parametrize("out_features", [8, 12, 24, 40, 4088])
def test_fp8_cb_rungs_are_masked_off_a_non_multiple_of_16(out_features):
    for fmt in ("FP8_CB_K28", "FP8_CB_K48"):
        decision = _shape_decision(fmt, out_features=out_features)
        assert not decision.legal
        assert decision.rule == "cb_fp8_out_features_load_gate"


def test_the_two_families_load_gates_really_are_different():
    """N = 8 is the boundary that separates them: legal for every fp4-CB
    rung, illegal for every fp8-CB rung. A single shared gate would be a
    silently wrong model of one family or the other."""
    assert _shape_decision("NVFP4_CB_K16", out_features=8).legal
    assert not _shape_decision("FP8_CB_K36", out_features=8).legal
    assert _shape_decision("FP8_CB_K36", out_features=16).legal


def test_the_load_gates_are_declared_per_grid_from_the_family_table():
    """Spec data, not a hand-listed set: a new rung inherits its family's
    gate the moment ``cb_layout.FAMILIES`` gains it."""
    profile = sp.load_serving_profile(_PROFILE)
    fp4 = next(r for r in profile.shape_rules
               if r.id == "cb_fp4_out_features_load_gate")
    fp8 = next(r for r in profile.shape_rules
               if r.id == "cb_fp8_out_features_load_gate")
    assert set(fp4.formats) == cb_layout.NVFP4_CB_FORMAT_NAMES
    assert set(fp8.formats) == cb_layout.FP8_CB_FORMAT_NAMES
    assert fp4.out_features_multiple_of == 8
    assert fp8.out_features_multiple_of == 16
    # Together the two per-grid sets are exactly the CB ladder — no rung
    # falls between the gates.
    assert (cb_layout.NVFP4_CB_FORMAT_NAMES
            | cb_layout.FP8_CB_FORMAT_NAMES) == cb_layout.CB_FORMAT_NAMES
    assert not (cb_layout.NVFP4_CB_FORMAT_NAMES
                & cb_layout.FP8_CB_FORMAT_NAMES)


def test_the_load_gates_do_not_touch_the_native_carriers():
    """The mixed container also ships plain NVFP4/FP8_DYNAMIC/BF16; those are
    stock compressed-tensors, not gridbook CB loads, and must keep their own
    (looser) shape rules."""
    for fmt in ("NVFP4", "FP8_DYNAMIC", "BF16"):
        assert _shape_decision(fmt, out_features=12).legal


def test_the_superblock_gate_is_unchanged():
    """P5b adds gates; it does not relax the one that already existed."""
    assert not _shape_decision(
        "NVFP4_CB_K16", out_features=512, in_features=1000).legal
    assert _shape_decision(
        "NVFP4_CB_K16", out_features=512, in_features=1024).legal


# ---------------------------------------------------------------------------
# 2. The fused-lane route is versioned spec data
# ---------------------------------------------------------------------------

def test_backed_fused_mid_m_rungs_are_spec_data_for_the_pinned_runtime():
    """Gridbook 0.8.0 instantiates K in {28,32,36,40,44,48} while production
    permits every K28..K48 — so five of the published 27B ladder's eight
    rungs silently take expand+GEMM (gridbook ROADMAP K1.2). That set is
    DATA, read against the version in gridbook_runtime_pin.json."""
    assert sp.gridbook_runtime_version() == "0.8.0"
    backed = sp.serving_lane_route(_PROFILE, "FP8_CB_K36")
    assert backed.fused_mid_m_backed
    assert backed.fused_mid_m_rungs == (28, 32, 36, 40, 44, 48)
    assert backed.fused_mid_m_range == (9, 128)
    assert backed.activation_contract == "w8a8-dynamic-e4m3"
    assert backed.rungs_source == "serving_profile_spec:0.8.0"

    for k in (37, 38, 39, 41, 47):
        unbacked = sp.serving_lane_route(_PROFILE, f"FP8_CB_K{k}")
        assert not unbacked.fused_mid_m_backed, k
        assert "expand+GEMM" in unbacked.fallback_route


def test_the_0_6_0_backed_set_is_the_k_mod_4_law_not_a_missing_five():
    """Gridbook 0.6.0 RESOLVED K1.2, and the answer was that the set was
    never incomplete. ``gridbook/codec.py`` derives ``FP8_FUSED_KBITS`` from a
    format+TMA law rather than transcribing an instantiation list:
    ``type_size = 4k`` is the packed-B TMA box's contiguous extent and must be
    a 16-byte multiple, and the fused mainloop decodes with ONE sub-table
    width ``CbSubW = k/4`` while the format splits k over ``n_sub = 4``
    raggedly — at k37 the true widths are ``(10,9,9,9)``, so a uniform decode
    would be WRONG, not merely unaligned. Both conditions are ``k % 4 == 0``.

    So the five off-law rungs of the published 27B K36..K47 ladder are
    permanently fallback-served, not pending; the producer must keep pricing
    them on the fallback row forever rather than waiting for coverage that
    cannot arrive. This test states the law, so a future spec edit that
    "completes" the set to every K28..K48 fails here.

    The law is why the surface did not move again at 0.7.0 or 0.8.0: those
    releases carry the deepseek_v4 serving contract (D0.1), the post-0.6.0
    review remediation, and then the source-passthrough loader plus the
    opt-in MXFP8 dense lane -- and ``FP8_FUSED_KBITS`` is still
    ``range(28, 49, 4)`` at the v0.8.0 tag commit."""
    backed = sp.serving_lane_route(_PROFILE, "FP8_CB_K36").fused_mid_m_rungs
    assert backed == tuple(range(28, 49, 4))
    for k in range(28, 49):
        route = sp.serving_lane_route(_PROFILE, f"FP8_CB_K{k}")
        assert route.fused_mid_m_backed == (k % 4 == 0), k


def test_advancing_the_pin_adds_a_version_key_and_never_edits_an_old_one():
    """The lane spec's own rule ("ADD the version key rather than editing an
    existing list"). An artifact produced under the 0.5.0 or 0.6.0 pin must
    stay resolvable at the route it actually shipped on, so every historical
    key survives the 0.8.0 advance — and each answers for itself."""
    lane = next(l for l in sp.load_serving_profile(_PROFILE).serving_lanes
                if l.id == "fp8_cb_fused_mid_m")
    declared = dict(lane.fused_mid_m_rungs_by_runtime_version)
    assert {"0.5.0", "0.6.0", "0.7.0", "0.8.0"} <= set(declared)
    assert (declared["0.5.0"] == declared["0.6.0"] == declared["0.7.0"]
            == declared["0.8.0"] == tuple(range(28, 49, 4)))

    for old_version in ("0.5.0", "0.6.0"):
        old = sp.serving_lane_route(
            _PROFILE, "FP8_CB_K36", runtime_version=old_version)
        assert old.fused_mid_m_backed, old_version
        assert old.rungs_source == f"serving_profile_spec:{old_version}"
    old = sp.serving_lane_route(
        _PROFILE, "FP8_CB_K36", runtime_version="0.5.0")
    # ...and the two resolutions are distinguishable, because the runtime
    # version is part of a route's identity even when the rungs agree.
    assert old.route_key() != sp.serving_lane_route(
        _PROFILE, "FP8_CB_K36").route_key()


def test_the_fp4_opt_in_fused_mid_m_lane_is_available_not_backed():
    """Gridbook 0.6.0 shipped a contract-preserving fp4-CB v2 fused mid-M
    kernel (``csrc/cb_fused_fp4v2_gemm.cu``, dense, 9 <= M <= 128, decoded
    weights bit-identical to ``cb_expand_v2`` at all 13 K12..K24 rungs) — and
    at 0.8.0, the version this release pins, it is STILL OPT-IN behind
    ``PRISMAQUANT_CB_FP4_FUSED_MIDM=1`` pending the served NATIVE-PARITY gate,
    with the flag unset leaving the dispatch byte-for-byte the BF16 bridge.

    This data declares what the DEFAULT contract serves, so the backed set
    stays EMPTY: a lane the operator must set an env flag to reach is
    AVAILABLE, not BACKED, and pricing a rung on a lane the default serve
    never takes is precisely the P5b defect this file exists to prevent. The
    distinction is recorded in the lane's ``detail`` and cites the flag."""
    lane = next(l for l in sp.load_serving_profile(_PROFILE).serving_lanes
                if l.id == "nvfp4_cb_quality_path")
    assert lane.fused_mid_m_rungs_by_runtime_version == ()
    assert "PRISMAQUANT_CB_FP4_FUSED_MIDM" in lane.detail
    for fmt in ("NVFP4_CB_K12", "NVFP4_CB_K24", "NVFP4_CB_S16"):
        route = sp.serving_lane_route(_PROFILE, fmt)
        assert not route.fused_mid_m_backed, fmt
        assert route.rungs_source == "lane_declares_no_fused_mid_m_lane"


def test_an_undeclared_runtime_version_backs_nothing():
    """Fail-closed: assuming a newer runtime backs what an older one did is
    exactly how an unbacked fast path gets priced."""
    future = sp.serving_lane_route(
        _PROFILE, "FP8_CB_K36", runtime_version="9.9.9")
    assert not future.fused_mid_m_backed
    assert future.fused_mid_m_rungs == ()
    assert future.rungs_source == "pinned_runtime_version_not_declared"


def test_fp4_cb_declares_the_bf16_bridge_and_no_fused_mid_m_lane():
    """The DEFAULT fp4-CB contract is not W4A4: dense M > 8 and MoE T > 16
    expand to BF16 and feed an Ampere-schedule grouped GEMM (ultraplan §2
    causes b and c). Every fp4-CB rung therefore rides the fallback."""
    for fmt in ("NVFP4_CB_K12", "NVFP4_CB_K18", "NVFP4_CB_K24"):
        route = sp.serving_lane_route(_PROFILE, fmt)
        assert route.activation_contract == "w4-bf16-bridge"
        assert not route.fused_mid_m_backed
        assert route.fused_mid_m_rungs == ()
        assert route.rungs_source == "lane_declares_no_fused_mid_m_lane"
        assert "cb_bf16_grouped_gemm" in route.fallback_route
    # ...and the two families' contracts are distinguishable, which is the
    # W4A4-vs-W8A8 fact the format name alone never carried.
    assert (sp.serving_lane_route(_PROFILE, "NVFP4_CB_K16").activation_contract
            != sp.serving_lane_route(
                _PROFILE, "FP8_CB_K36").activation_contract)


def test_profiles_without_a_declared_lane_return_none():
    """`research` describes emulation legality, not a served container."""
    assert sp.serving_lane_route("research", "FP8_CB_K36") is None
    # Plain NVFP4 has no CB lane declaration either — it is delegated to
    # stock compressed-tensors, whose route this spec does not own.
    assert sp.serving_lane_route(_PROFILE, "NVFP4") is None


def test_the_lane_catalog_is_reportable_and_names_its_runtime():
    catalog = sp.serving_lane_catalog(_PROFILE)
    assert catalog["schema"] == sp.SERVING_LANE_SCHEMA
    assert catalog["gridbook_runtime_version"] == "0.8.0"
    assert set(catalog["lanes"]) == {
        "nvfp4_cb_quality_path", "fp8_cb_fused_mid_m",
        # The two SOURCE-PASSTHROUGH lanes. They are declared as lanes of
        # their own precisely so P5b never files a passthrough unit under a
        # CB activation contract it does not have.
        "delegated_native_mxfp4", "delegated_native_fp8_block_ue8m0",
        # The RE-QUANTIZED native lane. Declared for the same P5b reason and
        # currently UNBACKED: no released Gridbook runtime carries a loader.
        "mxfp8_ue8m0_g32",
    }
    assert catalog["lanes"]["fp8_cb_fused_mid_m"]["fused_mid_m_rungs"] == [
        28, 32, 36, 40, 44, 48]
    # A passthrough lane backs no fused mid-M rung, and says so explicitly
    # rather than leaving the field absent: there is no decode prologue to
    # fuse, so an empty backed set is the honest state, not a data gap. The
    # unbacked re-quant lane reads the same way and for the same reason.
    for lane_id in ("delegated_native_mxfp4",
                    "delegated_native_fp8_block_ue8m0",
                    "mxfp8_ue8m0_g32"):
        lane = catalog["lanes"][lane_id]
        assert lane["fused_mid_m_rungs"] == []
        assert lane["fused_mid_m_rungs_source"] == (
            "lane_declares_no_fused_mid_m_lane")


# ---------------------------------------------------------------------------
# 3. The route travels with the candidate
# ---------------------------------------------------------------------------

def _cb_tables(menu):
    stats, costs = {}, {}
    for i in range(2):
        name = f"model.layers.{i}.self_attn.o_proj"
        stats[name] = {
            "h_trace": 1.0 + i, "n_params": 1024 * 512,
            "in_features": 1024, "out_features": 512,
        }
        costs[name] = {
            fmt: {"weight_mse": 1e-4, "output_mse": 2e-4,
                  "output_mse_measured": True}
            for fmt in menu
        }
    return stats, costs


def test_candidates_carry_the_concrete_serving_lane_route():
    menu = ["FP8_CB_K36", "FP8_CB_K37", "NVFP4_CB_K16", "BF16"]
    stats, costs = _cb_tables(menu)
    cands = build_candidates(
        stats, costs, [fr.get_format(m) for m in menu],
        target_profile=_PROFILE,
        cb_serialization_context=_CB_CONTEXT)
    by_fmt = {c.fmt: c for c in cands["model.layers.0.self_attn.o_proj"]}
    assert by_fmt["FP8_CB_K36"].serving_lane.fused_mid_m_backed
    assert not by_fmt["FP8_CB_K37"].serving_lane.fused_mid_m_backed
    assert by_fmt["NVFP4_CB_K16"].serving_lane.activation_contract == (
        "w4-bf16-bridge")
    assert by_fmt["BF16"].serving_lane is None
    # Two rungs of the SAME family with different backing have different
    # route keys — a format name is not an execution identity.
    assert (by_fmt["FP8_CB_K36"].serving_lane.route_key()
            != by_fmt["FP8_CB_K37"].serving_lane.route_key())
    # The side map build_candidates injects carries it too, for the paths
    # that read stats rather than candidates.
    lanes = stats["model.layers.0.self_attn.o_proj"]["_serving_lane_by_format"]
    assert lanes["FP8_CB_K36"].fused_mid_m_backed


def test_selection_provenance_splits_backed_rungs_from_the_fallback():
    """"Record in provenance which selected rungs ride a backed fused lane vs
    the expand+GEMM fallback"."""
    assignment = {
        "model.layers.0.self_attn.o_proj": "FP8_CB_K36",   # backed
        "model.layers.1.self_attn.o_proj": "FP8_CB_K37",   # fallback
        "model.layers.2.mlp.down_proj": "NVFP4_CB_K16",    # fallback (bridge)
        "model.layers.3.mlp.down_proj": "BF16",            # no CB lane
    }
    prov = selection_serving_lane_provenance(
        assignment, None, target_profile=_PROFILE)
    assert prov["gridbook_runtime_version"] == "0.8.0"
    assert prov["units_total"] == 4
    assert prov["units_on_backed_fused_mid_m_lane"] == 1
    assert prov["units_on_fallback_route"] == 2
    assert prov["units_without_declared_lane"] == 1
    assert prov["selected_rungs_fused_mid_m_backed"] == [36]
    assert prov["selected_rungs_on_fallback_route"] == [16, 37]
    assert prov["activation_contracts"] == {
        "w4-bf16-bridge": 1, "w8a8-dynamic-e4m3": 2}
    assert prov["by_format"]["FP8_CB_K37"]["units"] == 1
    assert prov["by_format"]["BF16"]["route"] is None


def test_selection_provenance_reads_the_route_off_the_chosen_candidate():
    """The candidate is the object the DP actually saw, so that is the first
    source; expanded members of aggregated super items (which have no
    candidate of their own) re-resolve from the profile and agree."""
    menu = ["FP8_CB_K36", "BF16"]
    stats, costs = _cb_tables(menu)
    cands = build_candidates(
        stats, costs, [fr.get_format(m) for m in menu],
        target_profile=_PROFILE,
        cb_serialization_context=_CB_CONTEXT)
    assignment = {n: "FP8_CB_K36" for n in stats}
    assignment["model.layers.9.mlp.experts.down_proj"] = "FP8_CB_K36"
    prov = selection_serving_lane_provenance(
        assignment, cands, target_profile=_PROFILE)
    assert prov["units_on_backed_fused_mid_m_lane"] == 3
    assert prov["units_without_declared_lane"] == 0
    # Every unit's activation-pricing branch is stamped too — including the
    # expanded member with no candidate, which says so rather than claiming
    # an estimator it never had.
    assert sum(prov["activation_pricing_branches"].values()) == 3
    assert prov["activation_pricing_branches"]["unrecorded"] == 1


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
