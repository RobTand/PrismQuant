"""Stock-vLLM anchored cost lane: ladder, partition, merge, and resume.

Four properties, each of which is a way the lane could be silently wrong
rather than loudly broken:

  * a profile-pinned unit gets a zero-cost passthrough row and *nothing else*
    -- never a quantized rung the runtime has no route for;
  * extrapolation is structurally impossible, not merely unused;
  * the packed-expert empirical half merges as a disjoint union, and an
    absent measurement is named rather than skipped;
  * the durable journal resumes by semantic identity, recomputing nothing and
    trusting nothing.

The legality tests run the *real* serving profile and the *real*
``check_format_applicability`` against synthetic shapes.  Synthetic weights,
real gate: a test that mocked the gate would pass while the campaign failed.
"""
from __future__ import annotations

import pickle

import pytest

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    ANCHORED_AURA_COST_CURRENCY,
    ANCHORED_AURA_COST_SOURCE,
    cost_entry_is_anchored_aura_supersurrogate,
    cost_entry_is_bit_exact,
    cost_entry_is_source_passthrough,
    cost_entry_predicted_dloss,
    serialized_candidate_payload,
)
from prismaquant import anchored_cost as core
from prismaquant import stock_anchored_cost as stock


SERVING_PROFILE = "vllm_glm5_next_packed_moe"
DENSE_QNAME = "model.layers.0.mlp.gate_proj"
DENSE_SHAPE = (12288, 4096)
PACKED_QNAME = "model.layers.3.mlp.experts.gate_up_proj"
PACKED_SHAPE = (8, 512, 256)
PINNED_QNAME = "model.layers.0.self_attn.o_proj"
PINNED_SHAPE = (4096, 4096)


def _n_params(shape):
    total = 1
    for value in shape:
        total *= value
    return total


def _payload(format_name, shape, qname):
    payload, _, _ = serialized_candidate_payload(
        fr.get_format(format_name), shape,
        qname=qname, cb_serialization_context=None,
    )
    return int(payload)


def _declaration(
    qname=DENSE_QNAME, shape=DENSE_SHAPE, *, source_kind="fp8",
    terminal="FP8_SOURCE", unit_class="dense",
):
    return stock.StockUnitDeclaration(
        qname=qname,
        role=qname.rsplit(".", 1)[-1],
        unit_class=unit_class,
        n_params=_n_params(shape),
        source_kind=source_kind,
        costed_format="NVFP4",
        terminal_format=terminal,
        payload_bytes_by_format={
            "NVFP4": _payload("NVFP4", shape, qname),
            terminal: _payload(terminal, shape, qname),
        },
    )


def _plugin(renderer=None):
    return stock.StockAnchoredFormatPlugin(
        arm_identity={"render_levers": dict(stock.RENDER_LEVERS)},
        serving_profile_id=SERVING_PROFILE,
        renderer=renderer,
    )


def _campaign_identity():
    return {
        "model_identity": {"model": "glm5-test"},
        "menu_identity": {"formats": ["NVFP4", "FP8_SOURCE"]},
        "calibration_identity": {"calib_hash": "deadbeef"},
    }


def _measured(units, value=4.0e-6):
    return {
        unit.qname: {
            "NVFP4": {
                "predicted_dloss": value,
                "dw_source": stock.PRODUCTION_RENDER_DW_SOURCE,
                "production_anchor_measured": True,
            },
        }
        for unit in units
    }


# --------------------------------------------------------------------------
# 1. Ladder legality
# --------------------------------------------------------------------------
def test_pinned_unit_gets_a_zero_cost_passthrough_and_nothing_else():
    rows = stock.pinned_passthrough_rows({
        PINNED_QNAME: ("BF16", _payload("BF16", PINNED_SHAPE, PINNED_QNAME)),
    })
    assert list(rows) == [PINNED_QNAME]
    per_format = rows[PINNED_QNAME]
    assert list(per_format) == ["BF16"], (
        "a pinned unit must carry exactly one rung: its source terminal")
    entry = per_format["BF16"]
    assert cost_entry_is_bit_exact(entry, "BF16")
    assert cost_entry_predicted_dloss({"h_trace": 12.5}, entry,
                                      format_name="BF16") == 0.0


def test_pinned_unit_cannot_become_a_unit_spec():
    """The core refuses it, so a pin can never acquire a costed rung."""
    with pytest.raises(ValueError, match="no renderable candidate"):
        core.UnitSpec(
            qname=PINNED_QNAME, role="o_proj", unit_class="dense",
            candidates=(core.CandidateSpec(
                format_name="BF16", bits=16.0, payload_bytes=1024,
                family=stock.SOURCE_TERMINAL_FAMILY,
                equivalence_class=stock.SOURCE_TERMINAL_BASIS,
                shape_features=(), coordinate=0.0, terminal=True,
            ),),
            n_params=1024,
        )


def test_real_gate_refuses_fp8_source_on_routed_experts():
    """The blocker this lane must report, not route around.

    ``vllm_packed_moe``'s inherited ``packed_moe_expert_formats`` allow-list
    does not contain ``FP8_SOURCE``, and it is consulted before the glm5 rule,
    so a routed-expert unit on a native-FP8 checkpoint has no legal
    passthrough terminal at all. That is a serving gap; the lane's job is to
    name it, and this test pins that it is named rather than smoothed over.
    """
    refusals = stock.check_declaration_legality(
        _declaration(
            PACKED_QNAME, PACKED_SHAPE, unit_class="packed_expert",
        ),
        shape=PACKED_SHAPE,
        target_profile=SERVING_PROFILE,
    )
    kinds = {(item.kind, item.format_name, item.reason) for item in refusals}
    assert ("source_terminal", "FP8_SOURCE", "profile_mismatch") in kinds
    assert all(item.kind != "costed_rung" for item in refusals), (
        "NVFP4 itself is legal on routed experts; only the terminal is not")


def test_dense_unit_ladder_is_legal_and_builds():
    declaration = _declaration()
    assert stock.check_declaration_legality(
        declaration, shape=DENSE_SHAPE, target_profile=SERVING_PROFILE,
    ) == ()
    units = stock.build_stock_units([declaration], _plugin())
    assert len(units) == 1
    formats = {candidate.format_name for candidate in units[0].candidates}
    assert formats == {"NVFP4", "FP8_SOURCE"}


def test_terminal_must_be_lossless_and_activation_identity():
    with pytest.raises(stock.StockAnchoredCostError,
                       match="no source-passthrough contract"):
        stock.exact_terminal_cost_entry("NVFP4")


def test_fp8_source_terminal_is_not_spelled_source_passthrough():
    """FP8_SOURCE is not in SOURCE_PASSTHROUGH_FORMATS, so that stamp lies."""
    entry = stock.exact_terminal_cost_entry("FP8_SOURCE")
    assert "cost_source" not in entry
    assert not cost_entry_is_source_passthrough(
        {**entry, "cost_source": "source_passthrough"}, "FP8_SOURCE"), (
        "premise: the source_passthrough stamp does NOT admit FP8_SOURCE")
    assert cost_entry_is_bit_exact(entry, "FP8_SOURCE")


# --------------------------------------------------------------------------
# 2. Extrapolation is structurally impossible
# --------------------------------------------------------------------------
def test_second_costed_rung_is_refused_by_the_plugin():
    declaration = _declaration()
    units = stock.build_stock_units([declaration], _plugin())
    extra = core.CandidateSpec(
        format_name="MXFP8_E4M3", bits=8.0, payload_bytes=10_000_000,
        family=fr.get_format("MXFP8_E4M3").family,
        equivalence_class=stock.SINGLE_RUNG_EQUIVALENCE_CLASS,
        shape_features=stock.SINGLE_RUNG_SHAPE_FEATURES, coordinate=8.0,
    )
    widened = core.UnitSpec(
        qname=units[0].qname, role=units[0].role,
        unit_class=units[0].unit_class,
        candidates=(*units[0].candidates, extra),
        n_params=units[0].n_params,
    )
    plugin = _plugin()
    with pytest.raises(stock.StockAnchoredCostError,
                       match="second costed rung"):
        plugin.describe_candidate(widened, "MXFP8_E4M3")
    with pytest.raises(stock.StockAnchoredCostError):
        stock.assert_single_rung_partition([widened], plugin)


def test_select_anchor_refuses_a_multi_rung_segment():
    declaration = _declaration()
    units = stock.build_stock_units([declaration], _plugin())
    unit = units[0]
    costed = next(c for c in unit.candidates if not c.terminal)
    with pytest.raises(stock.StockAnchoredCostError, match="costed rungs"):
        _plugin().select_anchor(
            unit, unit.segment_for(costed), (costed, costed),
        )


def test_core_shape_fit_cannot_be_built_for_a_single_rung_segment():
    """The core's own refusal, exercised so the guarantee is not just ours.

    ``_fit_currency`` needs two rungs per panel unit; a single-rung ladder can
    never produce a ``ShapeFit``, so ``price_anchored_candidates`` -- which
    hard-requires one -- is unreachable without fabricating panel provenance.
    """
    declaration = _declaration()
    plugin = _plugin()
    units = stock.build_stock_units([declaration], plugin)
    unit = units[0]
    (segment, candidates), = core.candidates_by_segment(unit, plugin).items()
    requests = core.plan_anchor_requests(units, plugin)
    anchors = stock.anchors_from_measured_scalars(
        requests, _measured(units),
        arm_identity=plugin.arm_identity,
        payload_identity=_campaign_identity(),
    )
    anchor = anchors[(unit.qname, segment)]
    observation = core.ShapeObservation(
        unit.qname, segment, candidates[0].format_name,
        anchor.predicted_dloss,
        receipt=core.make_production_render_receipt_from_hashes(
            core.RenderRequest(
                unit.qname, segment, candidates[0].format_name, "panel",
            ),
            core.ScalarRenderResult(anchor.predicted_dloss),
            arm_identity_sha256=anchor.receipt.arm_identity_sha256,
            payload_identity_sha256=anchor.receipt.payload_identity_sha256,
        ),
    )
    with pytest.raises(core.AnchoredCostError, match="fewer than two rungs"):
        core.fit_segment_shape(
            [observation], segment=segment, candidates=candidates,
        )


def test_priced_row_carries_ratio_one_and_admits_to_the_allocator():
    declaration = _declaration()
    plugin = _plugin()
    units = stock.build_stock_units([declaration], plugin)
    requests = core.plan_anchor_requests(units, plugin)
    anchors = stock.anchors_from_measured_scalars(
        requests, _measured(units),
        arm_identity=plugin.arm_identity,
        payload_identity=_campaign_identity(),
    )
    rows = stock.price_single_rung_candidates(units, plugin, anchors)
    entry = rows[DENSE_QNAME]["NVFP4"]
    assert entry["cost_source"] == ANCHORED_AURA_COST_SOURCE
    assert entry["cost_currency"] == ANCHORED_AURA_COST_CURRENCY
    assert entry["fisher_application_count"] == 1
    assert entry["shape_ratio"] == 1.0
    assert entry["predicted_dloss"] == pytest.approx(4.0e-6)
    assert cost_entry_is_anchored_aura_supersurrogate(dict(entry))
    # The allocator reads the projection directly: no second Fisher, no
    # h_trace multiply. A stats row with a large h_trace must not move it.
    assert cost_entry_predicted_dloss(
        {"h_trace": 1.0e6}, dict(entry), format_name="NVFP4",
    ) == pytest.approx(4.0e-6)


def test_anchor_from_a_non_production_render_is_refused():
    declaration = _declaration()
    plugin = _plugin()
    units = stock.build_stock_units([declaration], plugin)
    requests = core.plan_anchor_requests(units, plugin)
    rtn = {
        units[0].qname: {
            "NVFP4": {
                "predicted_dloss": 1.0e-6,
                "dw_source": "rtn",
                "production_anchor_measured": True,
            },
        },
    }
    with pytest.raises(stock.StockAnchoredCostError, match="dw_source"):
        stock.anchors_from_measured_scalars(
            requests, rtn,
            arm_identity=plugin.arm_identity,
            payload_identity=_campaign_identity(),
        )


def test_anchor_from_a_different_production_arm_is_refused():
    declaration = _declaration()
    plugin = _plugin()
    units = stock.build_stock_units([declaration], plugin)
    requests = core.plan_anchor_requests(units, plugin)
    foreign = stock.anchors_from_measured_scalars(
        requests, _measured(units),
        arm_identity={"render_levers": {"gptq": False}},
        payload_identity=_campaign_identity(),
    )
    with pytest.raises(stock.StockAnchoredCostError,
                       match="different production arm"):
        stock.price_single_rung_candidates(units, plugin, foreign)


# --------------------------------------------------------------------------
# 3. Packed-expert merge
# --------------------------------------------------------------------------
def _empirical_payload(qnames, value=9.0e-5):
    return {
        "schema": "prismaquant.expert_empirical_cost.v1",
        "formats": ["NVFP4", "FP8_SOURCE"],
        "stats": {name: {"h_trace": 0.0, "n_params": 1024} for name in qnames},
        "costs": {
            name: {
                "NVFP4": {
                    "predicted_dloss": value,
                    "cost_source": "empirical_unit_kl",
                    "output_mse_measured": False,
                },
            }
            for name in qnames
        },
        "provenance": {"unit_kls": {}},
    }


def test_empirical_rows_keep_their_own_provenance():
    rows = stock.expert_empirical_rows(_empirical_payload([PACKED_QNAME]))
    entry = rows[PACKED_QNAME]["NVFP4"]
    assert entry["cost_source"] == "empirical_unit_kl", (
        "an empirical serving-unit KL is not an AURA projection and must not "
        "be re-stamped into the anchored currency")
    assert not cost_entry_is_anchored_aura_supersurrogate(entry)


def test_empirical_payload_with_a_foreign_schema_is_refused():
    payload = _empirical_payload([PACKED_QNAME])
    payload["schema"] = "prismaquant.something_else.v1"
    with pytest.raises(stock.StockAnchoredCostError, match="schema"):
        stock.expert_empirical_rows(payload)


def test_merge_is_a_disjoint_union():
    declaration = _declaration()
    plugin = _plugin()
    units = stock.build_stock_units([declaration], plugin)
    requests = core.plan_anchor_requests(units, plugin)
    anchors = stock.anchors_from_measured_scalars(
        requests, _measured(units),
        arm_identity=plugin.arm_identity,
        payload_identity=_campaign_identity(),
    )
    anchored = stock.price_single_rung_candidates(units, plugin, anchors)
    pinned = stock.pinned_passthrough_rows({PINNED_QNAME: ("BF16", 4096)})
    empirical = stock.expert_empirical_rows(_empirical_payload([PACKED_QNAME]))
    merged, report = stock.merge_cost_rows(
        anchored=anchored, pinned=pinned, empirical=empirical,
        expected_empirical_units=[PACKED_QNAME],
    )
    assert set(merged) == {DENSE_QNAME, PINNED_QNAME, PACKED_QNAME}
    assert report.anchored_units == 1
    assert report.pinned_units == 1
    assert report.empirical_units == 1
    assert report.total_units == 3
    assert report.missing_empirical == ()


def test_merge_refuses_a_unit_priced_by_two_provenances():
    pinned = stock.pinned_passthrough_rows({PACKED_QNAME: ("BF16", 4096)})
    empirical = stock.expert_empirical_rows(_empirical_payload([PACKED_QNAME]))
    with pytest.raises(stock.StockAnchoredCostError, match="both pinned and"):
        stock.merge_cost_rows(
            anchored={}, pinned=pinned, empirical=empirical,
        )


def test_absent_empirical_measurement_is_named_not_skipped():
    with pytest.raises(stock.StockAnchoredCostError,
                       match="no empirical serving-unit KL"):
        stock.merge_cost_rows(
            anchored={}, pinned={}, empirical=None,
            expected_empirical_units=[PACKED_QNAME],
        )
    merged, report = stock.merge_cost_rows(
        anchored={}, pinned={}, empirical=None,
        expected_empirical_units=[PACKED_QNAME], require_empirical=False,
    )
    assert merged == {}
    assert report.missing_empirical == (PACKED_QNAME,)


def test_probe_coverage_refuses_a_unit_the_allocator_would_drop():
    with pytest.raises(stock.StockAnchoredCostError, match="absent from probe"):
        stock.assert_probe_coverage({DENSE_QNAME: {}}, {PINNED_QNAME: {}})
    unpriced = stock.assert_probe_coverage(
        {DENSE_QNAME: {}}, {DENSE_QNAME: {}, PINNED_QNAME: {}},
    )
    assert unpriced == (PINNED_QNAME,)


# --------------------------------------------------------------------------
# 4. Durable checkpoint resume
# --------------------------------------------------------------------------
class _CountingRenderer:
    """A renderer that records every request it is actually asked for."""

    def __init__(self, value=4.0e-6):
        self.calls: list[str] = []
        self.value = value

    def __call__(self, request):
        self.calls.append(request.request_id)
        return core.ScalarRenderResult(self.value)


def _campaign(units, plugin, checkpoint_dir, *, resume):
    return core.run_scalar_render_campaign(
        core.plan_anchor_requests(units, plugin),
        plugin,
        checkpoint_dir=checkpoint_dir,
        identity=_campaign_identity(),
        resume=resume,
        stage="glm53-stock-anchored-render",
    )


def test_checkpoint_resume_is_idempotent_and_recomputes_nothing(tmp_path):
    renderer = _CountingRenderer()
    plugin = _plugin(renderer)
    units = stock.build_stock_units(
        [_declaration(), _declaration(
            "model.layers.0.mlp.down_proj", (4096, 12288),
        )],
        plugin,
    )
    root = tmp_path / "ckpt"
    first = _campaign(units, plugin, root, resume=False)
    assert len(renderer.calls) == 2

    second = _campaign(units, plugin, root, resume=True)
    assert len(renderer.calls) == 2, (
        "a resumed campaign must recompute nothing")
    assert {k: v.to_dict() for k, v in first.items()} == {
        k: v.to_dict() for k, v in second.items()
    }

    # And the priced table is byte-identical across the resume boundary.
    def _price(results):
        requests = core.plan_anchor_requests(units, plugin)
        return stock.price_single_rung_candidates(
            units, plugin, core.anchors_from_results(requests, results),
        )

    assert _price(first) == _price(second)


def test_resume_refuses_a_changed_campaign_identity(tmp_path):
    plugin = _plugin(_CountingRenderer())
    units = stock.build_stock_units([_declaration()], plugin)
    root = tmp_path / "ckpt"
    _campaign(units, plugin, root, resume=False)

    identity = _campaign_identity()
    identity["calibration_identity"] = {"calib_hash": "0ther"}
    with pytest.raises(RuntimeError, match="refusing reuse or recompute"):
        core.run_scalar_render_campaign(
            core.plan_anchor_requests(units, plugin),
            plugin,
            checkpoint_dir=root,
            identity=identity,
            resume=True,
            stage="glm53-stock-anchored-render",
        )


def test_campaign_without_resume_refuses_an_existing_journal(tmp_path):
    plugin = _plugin(_CountingRenderer())
    units = stock.build_stock_units([_declaration()], plugin)
    root = tmp_path / "ckpt"
    _campaign(units, plugin, root, resume=False)
    with pytest.raises(RuntimeError, match="pass --resume"):
        _campaign(units, plugin, root, resume=False)


def test_renderer_absent_fails_closed_rather_than_falling_back_to_rtn():
    plugin = _plugin(renderer=None)
    units = stock.build_stock_units([_declaration()], plugin)
    request = core.plan_anchor_requests(units, plugin)[0]
    with pytest.raises(stock.StockAnchoredCostError, match="render-free"):
        plugin.render(request)


# --------------------------------------------------------------------------
# 5. Payload shape
# --------------------------------------------------------------------------
def test_payload_has_no_stats_key_and_declares_its_formats():
    declaration = _declaration()
    plugin = _plugin()
    units = stock.build_stock_units([declaration], plugin)
    requests = core.plan_anchor_requests(units, plugin)
    anchors = stock.anchors_from_measured_scalars(
        requests, _measured(units),
        arm_identity=plugin.arm_identity,
        payload_identity=_campaign_identity(),
    )
    merged, report = stock.merge_cost_rows(
        anchored=stock.price_single_rung_candidates(units, plugin, anchors),
        pinned={}, empirical=None,
    )
    payload = stock.build_stock_allocator_cost_payload(
        costs=merged, merge_report=report, plugin=plugin,
        campaign_identity=_campaign_identity(),
        refusals=[stock.LadderRefusal(
            PACKED_QNAME, "FP8_SOURCE", "source_terminal", "profile_mismatch",
        )],
    )
    assert set(payload) == {"schema", "formats", "costs", "provenance", "meta"}
    assert "stats" not in payload, (
        "the allocator reads stats from --probe; a second copy here would be "
        "a silent authority conflict")
    assert payload["formats"] == ["FP8_SOURCE", "NVFP4"]
    provenance = payload["provenance"]
    assert provenance["cost_mode"] == "aura"
    assert provenance["extrapolation_expressible"] is False
    assert provenance["ladder_refusals"][0]["reason"] == "profile_mismatch", (
        "a serving gap travels inside the artifact, not only in a log")
    # Round-trips through the pickle the allocator will read.
    assert pickle.loads(pickle.dumps(payload)) == payload


def test_payload_becomes_allocator_candidates_on_the_intended_branches():
    """End to end through the real ``build_candidates``.

    The schema table in a handover is a claim; this is the evidence. Both
    numbers below are branch-discriminating: if the anchored row fell through
    to the generic weight-only path it would be priced
    ``0.5 * h_trace * weight_mse`` -- exactly 0.0 for a row with no
    ``weight_mse`` -- and the terminal would be indistinguishable from it.
    """
    declaration = _declaration()
    plugin = _plugin()
    units = stock.build_stock_units([declaration], plugin)
    requests = core.plan_anchor_requests(units, plugin)
    anchors = stock.anchors_from_measured_scalars(
        requests, _measured(units),
        arm_identity=plugin.arm_identity,
        payload_identity=_campaign_identity(),
    )
    merged, report = stock.merge_cost_rows(
        anchored=stock.price_single_rung_candidates(units, plugin, anchors),
        pinned={}, empirical=None,
    )
    payload = stock.build_stock_allocator_cost_payload(
        costs=merged, merge_report=report, plugin=plugin,
        campaign_identity=_campaign_identity(),
    )
    from prismaquant.allocator_candidates import build_candidates

    stats = {
        DENSE_QNAME: {
            "h_trace": 1692.48, "n_params": _n_params(DENSE_SHAPE),
            "in_features": DENSE_SHAPE[1], "out_features": DENSE_SHAPE[0],
        },
    }
    candidates = build_candidates(
        stats, payload["costs"],
        [fr.get_format(name) for name in ("NVFP4", "FP8_SOURCE")],
        source_manifest={DENSE_QNAME: "fp8"},
        target_profile=SERVING_PROFILE,
    )
    by_format = {c.fmt: c for c in candidates[DENSE_QNAME]}
    assert set(by_format) == {"NVFP4", "FP8_SOURCE"}
    assert by_format["NVFP4"].predicted_dloss == pytest.approx(4.0e-6, rel=0)
    assert by_format["FP8_SOURCE"].predicted_dloss == 0.0
    assert by_format["NVFP4"].memory_bytes < by_format["FP8_SOURCE"].memory_bytes
