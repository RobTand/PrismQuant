from __future__ import annotations

import math
import json
import os
from pathlib import Path
import sys
from dataclasses import replace

import pytest

from prismaquant.anchored_cost import (
    AURA_CURRENCY,
    AnchorScalar,
    AnchoredCostError,
    AnchoredFormatPlugin,
    CandidateSpec,
    PluginDeclaration,
    ProductionRenderReceipt,
    RenderRequest,
    ScalarRenderResult,
    SegmentKey,
    ShapeFit,
    ShapeObservation,
    UnitSpec,
    anchors_from_results,
    extrapolation_distance_report,
    fit_segment_shape,
    lower_convex_hull,
    make_production_render_receipt,
    plan_anchor_requests,
    price_anchored_candidates,
    run_allocator_once,
    run_scalar_render_campaign,
)
from prismaquant.cost_stage_checkpoint import write_unit


_FAMILY = "synthetic_family"
_ROLE = "projection"
_EQUIVALENCE_A = "direction_alpha"
_EQUIVALENCE_B = "direction_beta"


def _candidate(
    name: str,
    *,
    bits: float,
    payload_bytes: int,
    equivalence_class: str,
    coordinate: float,
    terminal: bool = False,
) -> CandidateSpec:
    return CandidateSpec(
        format_name=name,
        bits=bits,
        payload_bytes=payload_bytes,
        family="source" if terminal else _FAMILY,
        equivalence_class=(
            "passthrough" if terminal else equivalence_class
        ),
        shape_features=() if terminal else (coordinate,),
        coordinate=coordinate,
        terminal=terminal,
    )


def _unit(qname: str) -> UnitSpec:
    return UnitSpec(
        qname=qname,
        role=_ROLE,
        unit_class="synthetic",
        candidates=(
            _candidate(
                "A0", bits=1.0, payload_bytes=100,
                equivalence_class=_EQUIVALENCE_A, coordinate=0,
            ),
            _candidate(
                "A1", bits=2.0, payload_bytes=200,
                equivalence_class=_EQUIVALENCE_A, coordinate=1,
            ),
            _candidate(
                "A2", bits=3.0, payload_bytes=300,
                equivalence_class=_EQUIVALENCE_A, coordinate=2,
            ),
            _candidate(
                "B0", bits=1.25, payload_bytes=125,
                equivalence_class=_EQUIVALENCE_B, coordinate=0,
            ),
            _candidate(
                "B1", bits=2.25, payload_bytes=225,
                equivalence_class=_EQUIVALENCE_B, coordinate=1,
            ),
            _candidate(
                "B2", bits=3.25, payload_bytes=325,
                equivalence_class=_EQUIVALENCE_B, coordinate=2,
            ),
            _candidate(
                "SOURCE", bits=16.0, payload_bytes=1600,
                equivalence_class="passthrough", coordinate=0,
                terminal=True,
            ),
        ),
        n_params=800,
    )


class _SyntheticPlugin:
    """Two-segment executable plugin with no platform-format knowledge."""

    def __init__(
        self,
        *,
        arm_identity: str = "production-arm-a",
        fail_on_call: int | None = None,
        forbidden_render_field: str | None = None,
        zero_anchor: tuple[str, str] | None = None,
    ) -> None:
        self.arm_identity = arm_identity
        self.fail_on_call = fail_on_call
        self.forbidden_render_field = forbidden_render_field
        self.zero_anchor = zero_anchor
        self.rendered: list[RenderRequest] = []

    def plugin_identity(self) -> PluginDeclaration:
        return PluginDeclaration(
            plugin_id="synthetic.anchored",
            plugin_version="1",
            equivalence_contract="direction-id-v1",
        )

    def describe_candidate(
        self,
        unit: UnitSpec,
        format_name: str,
    ) -> CandidateSpec:
        candidate = next(
            candidate
            for candidate in unit.candidates
            if candidate.format_name == format_name
        )
        authoritative = {
            **{f"A{index}": _EQUIVALENCE_A for index in range(3)},
            **{f"B{index}": _EQUIVALENCE_B for index in range(3)},
            "SOURCE": "passthrough",
        }
        if candidate.equivalence_class != authoritative[format_name]:
            raise AnchoredCostError(
                f"{format_name}: unit label differs from plugin partition"
            )
        return candidate

    def select_anchor(
        self,
        unit: UnitSpec,
        segment: SegmentKey,
        candidates: tuple[CandidateSpec, ...],
    ) -> str:
        del unit, segment
        return candidates[len(candidates) // 2].format_name

    def render(
        self,
        request: RenderRequest,
    ) -> ScalarRenderResult | dict[str, float]:
        self.rendered.append(request)
        if self.fail_on_call == len(self.rendered):
            raise RuntimeError("synthetic interruption")
        if self.forbidden_render_field is not None:
            return {
                "predicted_dloss": 1.0,
                self.forbidden_render_field: 2.0,
            }
        if self.zero_anchor == (request.qname, request.format_name):
            return ScalarRenderResult(0.0)
        level = 1.0 + int(request.qname.rsplit(".", 1)[1])
        coordinate = int(request.format_name[1:])
        family_level = 1.0 if request.format_name.startswith("A") else 3.0
        shape = 0.5**coordinate
        return ScalarRenderResult(
            predicted_dloss=level * family_level * shape,
            weight_mse_diagnostic=11.0 * level * shape,
        )

    def provenance_identity_fields(self) -> dict[str, str]:
        return {
            "arm_identity": self.arm_identity,
            "renderer_contract": "scalar-only-v1",
        }


def _segment(equivalence_class: str) -> SegmentKey:
    return SegmentKey(_FAMILY, _ROLE, equivalence_class)


def _receipt_payload_identity(plugin: _SyntheticPlugin) -> dict[str, object]:
    declaration = plugin.plugin_identity()
    return {
        **_identity(),
        "plugin_identity": declaration.to_dict(),
        "plugin_provenance": plugin.provenance_identity_fields(),
        "cost_currency": AURA_CURRENCY,
        "fisher_application_count": 1,
    }


def _receipt(
    request: RenderRequest,
    scalar: ScalarRenderResult,
    plugin: _SyntheticPlugin,
) -> ProductionRenderReceipt:
    return make_production_render_receipt(
        request,
        scalar,
        arm_identity=plugin.provenance_identity_fields()["arm_identity"],
        payload_identity=_receipt_payload_identity(plugin),
    )


def _render_receipts(
    requests: tuple[RenderRequest, ...],
    plugin: _SyntheticPlugin,
) -> dict[str, ProductionRenderReceipt]:
    return {
        request.request_id: _receipt(request, plugin.render(request), plugin)
        for request in requests
    }


def _segment_candidates(
    unit: UnitSpec,
    equivalence_class: str,
) -> tuple[CandidateSpec, ...]:
    return tuple(
        candidate
        for candidate in unit.candidates
        if not candidate.terminal
        and candidate.equivalence_class == equivalence_class
    )


def _observations(
    units: tuple[UnitSpec, ...],
    equivalence_class: str,
    plugin: _SyntheticPlugin | None = None,
) -> tuple[ShapeObservation, ...]:
    plugin = plugin or _SyntheticPlugin()
    prefix = "A" if equivalence_class == _EQUIVALENCE_A else "B"
    family_level = 1.0 if prefix == "A" else 3.0
    segment = _segment(equivalence_class)
    rows = []
    for unit_index, unit in enumerate(units, start=1):
        for coordinate in range(3):
            shape = 0.5**coordinate
            request = RenderRequest(
                unit.qname, segment, f"{prefix}{coordinate}", "panel",
            )
            scalar = ScalarRenderResult(
                unit_index * family_level * shape,
                (10.0 + unit_index) * shape,
            )
            rows.append(ShapeObservation(
                qname=unit.qname,
                segment=segment,
                format_name=f"{prefix}{coordinate}",
                predicted_dloss=scalar.predicted_dloss,
                weight_mse_diagnostic=scalar.weight_mse_diagnostic,
                receipt=_receipt(request, scalar, plugin),
            ))
    return tuple(rows)


def _fits(
    units: tuple[UnitSpec, ...],
    plugin: _SyntheticPlugin | None = None,
) -> dict[SegmentKey, ShapeFit]:
    plugin = plugin or _SyntheticPlugin()
    unit = units[0]
    return {
        segment: fit_segment_shape(
            _observations(units, equivalence_class, plugin),
            segment=segment,
            candidates=_segment_candidates(unit, equivalence_class),
        )
        for equivalence_class in (_EQUIVALENCE_A, _EQUIVALENCE_B)
        for segment in (_segment(equivalence_class),)
    }


def _identity(**updates: str) -> dict[str, str]:
    identity = {
        "model_identity": "synthetic-model",
        "menu_identity": "synthetic-menu",
        "calibration_identity": "synthetic-calibration",
    }
    identity.update(updates)
    return identity


def test_protocol_has_five_methods_and_core_is_platform_vocabulary_free():
    # ``__protocol_attrs__`` is a CPython implementation detail of
    # typing.Protocol that only exists on 3.12+; read the declared members off
    # the class instead so this pins the contract on every supported Python.
    declared = {
        name for name in vars(AnchoredFormatPlugin)
        if not name.startswith("_")
    }
    assert declared == {
        "plugin_identity",
        "describe_candidate",
        "select_anchor",
        "render",
        "provenance_identity_fields",
    }
    assert isinstance(_SyntheticPlugin(), AnchoredFormatPlugin)

    source = (
        Path(__file__).parents[1] / "prismaquant" / "anchored_cost.py"
    ).read_text()
    for platform_word in (
        "NVFP4", "FP8_CB", "codebook", "lattice", "learned",
    ):
        assert platform_word not in source
    assert "from prismaquant.cost_stage_checkpoint import" in source


def test_synthetic_plugin_independently_refuses_a_mislabeled_partition():
    unit = _unit("unit.0")
    candidates = tuple(
        replace(candidate, equivalence_class=_EQUIVALENCE_A)
        if candidate.format_name == "B0" else candidate
        for candidate in unit.candidates
    )
    mislabeled = replace(unit, candidates=candidates)
    with pytest.raises(AnchoredCostError, match="plugin partition"):
        plan_anchor_requests((mislabeled,), _SyntheticPlugin())


def test_anchor_plan_and_renderer_scale_by_unit_equivalence_segment(tmp_path):
    units = (_unit("unit.0"), _unit("unit.1"))
    plugin = _SyntheticPlugin()
    requests = plan_anchor_requests(units, plugin)

    # Two legal equivalence segments per unit, versus six renderable rungs.
    assert len(requests) == len(units) * 2 == 4
    assert sum(not item.terminal for unit in units for item in unit.candidates) == 12
    assert {
        (request.qname, request.segment.equivalence_class)
        for request in requests
    } == {
        (unit.qname, equivalence_class)
        for unit in units
        for equivalence_class in (_EQUIVALENCE_A, _EQUIVALENCE_B)
    }
    assert {request.format_name for request in requests} == {"A1", "B1"}

    results = run_scalar_render_campaign(
        requests,
        plugin,
        checkpoint_dir=tmp_path / "anchors",
        identity=_identity(),
        resume=False,
    )
    assert len(plugin.rendered) == len(requests)
    assert set(results) == {request.request_id for request in requests}
    assert all(
        isinstance(value, ProductionRenderReceipt)
        for value in results.values()
    )
    assert all(value.rendered_weight_persisted is False for value in results.values())
    assert len(anchors_from_results(requests, results)) == 4

    with pytest.raises(AnchoredCostError, match="bare scalar"):
        anchors_from_results(
            requests,
            {
                request.request_id: ScalarRenderResult(1.0)
                for request in requests
            },
        )

    with pytest.raises(AnchoredCostError, match="required fields"):
        run_scalar_render_campaign(
            requests,
            _SyntheticPlugin(),
            checkpoint_dir=tmp_path / "missing-identity",
            identity={},
            resume=False,
        )


def test_fixed_effect_fit_is_identifiable_and_currency_diagnostic_only():
    units = (_unit("unit.0"), _unit("unit.1"))
    fits = _fits(units)

    for fit in fits.values():
        assert fit.design_rank == fit.design_rank_required == 1
        assert fit.shape_fit_currency == AURA_CURRENCY
        assert fit.ratio(
            fit.reference_format[0] + "2", fit.reference_format,
        ) == pytest.approx(0.25)
        diagnostic = fit.aura_vs_weight_diagnostic
        assert diagnostic is not None
        assert diagnostic["currency_invariance_test_only"] is True
        assert diagnostic["max_abs_dex"] == pytest.approx(0.0, abs=1e-12)


def test_fit_and_application_refuse_cross_equivalence_transfer(tmp_path):
    del tmp_path
    units = (_unit("unit.0"), _unit("unit.1"))
    unit = units[0]
    segment_a = _segment(_EQUIVALENCE_A)
    segment_b = _segment(_EQUIVALENCE_B)
    observations_a = _observations(units, _EQUIVALENCE_A)
    observations_b = _observations(units, _EQUIVALENCE_B)

    with pytest.raises(AnchoredCostError, match="may not span"):
        fit_segment_shape(
            (*observations_a, observations_b[0]),
            segment=segment_a,
            candidates=_segment_candidates(unit, _EQUIVALENCE_A),
        )
    with pytest.raises(AnchoredCostError, match="crosses"):
        fit_segment_shape(
            observations_a,
            segment=segment_a,
            candidates=(
                *_segment_candidates(unit, _EQUIVALENCE_A),
                _segment_candidates(unit, _EQUIVALENCE_B)[0],
            ),
        )

    plugin = _SyntheticPlugin()
    requests = plan_anchor_requests(units, plugin)
    results = _render_receipts(requests, plugin)
    anchors = anchors_from_results(requests, results)
    fits = _fits(units)

    with pytest.raises(ValueError, match="receipt request differs"):
        AnchorScalar(
            unit.qname,
            segment_a,
            "B1",
            1.0,
            anchors[(unit.qname, segment_b)].receipt,
        )

    wrong_fit = dict(fits)
    wrong_fit[segment_a] = fits[segment_b]
    with pytest.raises(AnchoredCostError, match="fit lies across"):
        price_anchored_candidates(units, plugin, anchors, wrong_fit)


def test_every_candidate_is_anchor_priced_once_and_h_squared_inputs_refuse(
    tmp_path,
):
    units = (_unit("unit.0"), _unit("unit.1"))
    plugin = _SyntheticPlugin()
    requests = plan_anchor_requests(units, plugin)
    results = _render_receipts(requests, plugin)
    anchors = anchors_from_results(requests, results)
    rows = price_anchored_candidates(
        units, plugin, anchors, _fits(units, plugin)
    )

    assert set(rows) == {unit.qname for unit in units}
    for unit in units:
        cells = rows[unit.qname]
        assert {cell.candidate.format_name for cell in cells} == {
            candidate.format_name for candidate in unit.candidates
        }
        assert len(cells) == len(unit.candidates)
        for cell in cells:
            entry = cell.allocation_entry()
            assert not {"h_trace", "cw_m2", "weight_mse"} & set(entry)
            assert entry["cost_currency"] == AURA_CURRENCY
            if cell.segment is None:
                assert cell.candidate.terminal is True
                continue
            assert cell.cost_source == "anchored_aura_extrapolation"
            assert entry["fisher_application_count"] == 1
            assert cell.anchor_predicted_dloss is not None
            assert cell.shape_ratio is not None

    request = requests[0]
    for field in ("h_trace", "cw_m2", "weight_mse", "activation_mse"):
        bad_plugin = _SyntheticPlugin(forbidden_render_field=field)
        with pytest.raises(AnchoredCostError, match="forbidden.*cost inputs"):
            run_scalar_render_campaign(
                (request,),
                bad_plugin,
                checkpoint_dir=tmp_path / field,
                identity=_identity(),
                resume=False,
            )


def test_terminal_cost_source_is_the_allocator_passthrough_contract():
    """The byte-verbatim terminal must speak the allocator's own vocabulary.

    ``cost_entry_is_source_passthrough`` refuses any other ``cost_source``, so a
    near-miss label still prices correctly while misclassifying provenance and
    the activation branch. Pin the two constants together so they cannot drift.
    """
    from prismaquant.allocator_candidates import SOURCE_PASSTHROUGH_COST_SOURCE

    units = (_unit("unit.0"),)
    plugin = _SyntheticPlugin()
    requests = plan_anchor_requests(units, plugin)
    anchors = anchors_from_results(requests, _render_receipts(requests, plugin))
    rows = price_anchored_candidates(
        units, plugin, anchors, _fits(units, plugin)
    )

    terminals = [cell for cell in rows["unit.0"] if cell.candidate.terminal]
    assert len(terminals) == 1
    entry = terminals[0].allocation_entry()
    assert entry["cost_source"] == SOURCE_PASSTHROUGH_COST_SOURCE
    assert entry["predicted_dloss"] == 0.0

    source = (
        Path(__file__).parents[1] / "prismaquant" / "anchored_cost.py"
    ).read_text()
    assert "exact_source_passthrough_terminal" not in source


def test_zero_measured_anchor_is_retained_and_not_proof_pruned():
    units = (_unit("unit.0"),)
    plugin = _SyntheticPlugin(zero_anchor=("unit.0", "A1"))
    requests = plan_anchor_requests(units, plugin)
    results = _render_receipts(requests, plugin)
    anchors = anchors_from_results(requests, results)
    rows = price_anchored_candidates(units, plugin, anchors, _fits((
        _unit("unit.0"), _unit("unit.1"),
    ), plugin))

    segment_a = _segment(_EQUIVALENCE_A)
    assert anchors[("unit.0", segment_a)].predicted_dloss == 0.0
    a_cells = [
        cell for cell in rows["unit.0"]
        if cell.segment == segment_a
    ]
    assert {cell.candidate.format_name for cell in a_cells} == {
        "A0", "A1", "A2",
    }
    assert {cell.predicted_dloss for cell in a_cells} == {0.0}


def test_interrupt_resume_matches_uninterrupted_and_identity_refuses(tmp_path):
    units = (_unit("unit.0"), _unit("unit.1"))
    planner = _SyntheticPlugin()
    requests = plan_anchor_requests(units, planner)
    checkpoint = tmp_path / "interrupted"

    interrupted = _SyntheticPlugin(fail_on_call=2)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_scalar_render_campaign(
            requests,
            interrupted,
            checkpoint_dir=checkpoint,
            identity=_identity(),
            resume=False,
        )
    assert len(interrupted.rendered) == 2

    resumed_plugin = _SyntheticPlugin()
    resumed = run_scalar_render_campaign(
        requests,
        resumed_plugin,
        checkpoint_dir=checkpoint,
        identity=_identity(),
        resume=True,
    )
    uninterrupted_plugin = _SyntheticPlugin()
    uninterrupted = run_scalar_render_campaign(
        requests,
        uninterrupted_plugin,
        checkpoint_dir=tmp_path / "uninterrupted",
        identity=_identity(),
        resume=False,
    )
    assert resumed == uninterrupted
    assert len(resumed_plugin.rendered) == len(requests) - 1
    assert len(uninterrupted_plugin.rendered) == len(requests)

    mismatch_plugin = _SyntheticPlugin()
    with pytest.raises(RuntimeError, match="identity mismatch.*model_identity"):
        run_scalar_render_campaign(
            requests,
            mismatch_plugin,
            checkpoint_dir=checkpoint,
            identity=_identity(model_identity="other-model"),
            resume=True,
        )
    assert mismatch_plugin.rendered == []

    arm_mismatch_plugin = _SyntheticPlugin(arm_identity="other-arm")
    with pytest.raises(RuntimeError, match="identity mismatch.*arm_identity"):
        run_scalar_render_campaign(
            requests,
            arm_mismatch_plugin,
            checkpoint_dir=checkpoint,
            identity=_identity(),
            resume=True,
        )
    assert arm_mismatch_plugin.rendered == []

    manifest = json.loads((checkpoint / "manifest.json").read_text())
    request = requests[0]
    bad_receipt = dict(resumed[request.request_id].to_dict())
    bad_receipt["fisher_application_count"] = 2
    write_unit(
        checkpoint,
        stage="anchored-production-render",
        qname=request.request_id,
        identity_sha256=manifest["identity_sha256"],
        state={"production_render_receipt": bad_receipt},
    )
    tampered_plugin = _SyntheticPlugin()
    with pytest.raises(
        AnchoredCostError, match="production receipt semantics differ"
    ):
        run_scalar_render_campaign(
            requests,
            tampered_plugin,
            checkpoint_dir=checkpoint,
            identity=_identity(),
            resume=True,
        )
    assert tampered_plugin.rendered == []


def _write_synthetic_allocator(path: Path) -> None:
    path.write_text("""
import json
import os
from pathlib import Path
import sys

output = Path(sys.argv[1])
counter = Path(sys.argv[2])
sentinel = Path(sys.argv[3])
count = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(count))
if os.environ.get("SYNTHETIC_FAIL_ONCE") == "1" and not sentinel.exists():
    (output / "partial-output.txt").write_text("preserve me")
    sentinel.write_text("failed")
    raise SystemExit(17)
(output / "layer_config.json").write_text(json.dumps({"unit": "A1"}))
(output / "selection.json").write_text(json.dumps({"feasible": True}))
""")


def test_allocator_late_resume_reuses_exact_and_refuses_identity_mismatch(
    tmp_path,
):
    script = tmp_path / "allocator.py"
    _write_synthetic_allocator(script)
    output = tmp_path / "allocator-output"
    counter = tmp_path / "counter"
    command = (
        sys.executable, str(script), str(output), str(counter),
        str(tmp_path / "unused-sentinel"),
    )
    provenance = {"cost_currency": AURA_CURRENCY, "budget_bytes": 1000}

    run_allocator_once(
        command=command,
        output_dir=output,
        invocation_provenance=provenance,
        resume=False,
    )
    receipt = (output / "anchored_allocator_invocation.json").read_bytes()
    receipt_payload = json.loads(receipt)
    assert receipt_payload["cost_currency"] == AURA_CURRENCY
    assert receipt_payload["invocation_provenance"] == provenance
    assert counter.read_text() == "1"

    assert run_allocator_once(
        command=command,
        output_dir=output,
        invocation_provenance=provenance,
        resume=True,
    ) == output
    assert counter.read_text() == "1"
    assert (output / "anchored_allocator_invocation.json").read_bytes() == receipt

    with pytest.raises(
        AnchoredCostError, match="allocator invocation identity mismatch",
    ):
        run_allocator_once(
            command=command,
            output_dir=output,
            invocation_provenance={**provenance, "budget_bytes": 999},
            resume=True,
        )
    assert counter.read_text() == "1"

    with pytest.raises(AnchoredCostError, match="refusing overwrite"):
        run_allocator_once(
            command=command,
            output_dir=output,
            invocation_provenance=provenance,
            resume=False,
        )


def test_allocator_child_inherits_explicit_read_descriptor(tmp_path):
    script = tmp_path / "fd-allocator.py"
    script.write_text("""
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
inherited = Path(sys.argv[2])
(output / "layer_config.json").write_text(json.dumps({"unit": "A1"}))
(output / "selection.json").write_text(json.dumps({"feasible": True}))
(output / "inherited.bin").write_bytes(inherited.read_bytes())
""")
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"exact inherited bytes")
    descriptor = os.open(payload, os.O_RDONLY)
    try:
        output = tmp_path / "allocator-output"
        run_allocator_once(
            command=(
                sys.executable, str(script), str(output),
                f"/proc/self/fd/{descriptor}",
            ),
            output_dir=output,
            pass_fds=(descriptor,),
        )
    finally:
        os.close(descriptor)
    assert (output / "inherited.bin").read_bytes() == b"exact inherited bytes"


def test_allocator_late_resume_preserves_interrupted_output_before_retry(
    tmp_path,
):
    script = tmp_path / "allocator.py"
    _write_synthetic_allocator(script)
    output = tmp_path / "allocator-output"
    counter = tmp_path / "counter"
    sentinel = tmp_path / "fail-once"
    command = (
        sys.executable, str(script), str(output), str(counter), str(sentinel),
    )
    environment = {"SYNTHETIC_FAIL_ONCE": "1"}
    provenance = {"campaign": "same-identity"}

    with pytest.raises(AnchoredCostError, match="exit code 17"):
        run_allocator_once(
            command=command,
            output_dir=output,
            environment_updates=environment,
            invocation_provenance=provenance,
            resume=False,
        )
    assert (output / "partial-output.txt").read_text() == "preserve me"

    run_allocator_once(
        command=command,
        output_dir=output,
        environment_updates=environment,
        invocation_provenance=provenance,
        resume=True,
    )
    preserved = list(tmp_path.glob("allocator-output.incomplete-*"))
    assert len(preserved) == 1
    assert (preserved[0] / "partial-output.txt").read_text() == "preserve me"
    assert (preserved[0] / "anchored_allocator_identity.json").is_file()
    assert counter.read_text() == "2"
    assert (output / "anchored_allocator_invocation.json").is_file()


def test_lower_hull_is_computed_and_discards_dominated_points():
    hull = lower_convex_hull({
        "r0": (1.0, 9.0),
        "r1": (2.0, 7.0),  # Above the r0-to-r2 lower chord.
        "r2": (3.0, 3.0),
        "r3": (4.0, 2.0),
        "dominated": (5.0, 4.0),
    })
    assert hull.vertices == ("r0", "r2", "r3")
    assert hull.interior == ("r1", "dominated")

    with pytest.raises(AnchoredCostError, match="strictly positive"):
        lower_convex_hull({"zero": (1.0, 0.0), "other": (2.0, 1.0)})


def test_extrapolation_distance_reports_selected_to_own_segment_anchor():
    units = (_unit("unit.0"), _unit("unit.1"))
    plugin = _SyntheticPlugin()
    requests = plan_anchor_requests(units, plugin)
    results = _render_receipts(requests, plugin)
    anchors = anchors_from_results(requests, results)
    report = extrapolation_distance_report(
        units,
        plugin,
        anchors,
        {"unit.0": "A2", "unit.1": "B0"},
    )

    assert report["unit"] == "plugin_rung_coordinate"
    assert report["count"] == 2
    assert report["distribution"] == [{"distance": 1.0, "unit_count": 2}]
    assert {
        row["segment"]["equivalence_class"] for row in report["rows"]
    } == {_EQUIVALENCE_A, _EQUIVALENCE_B}
    assert all("basis" not in row["segment"] for row in report["rows"])

    with pytest.raises(AnchoredCostError, match="complete unit set"):
        extrapolation_distance_report(
            units, plugin, anchors, {"unit.0": "A2"}
        )


def test_rank_deficient_panel_refuses_instead_of_guessing_shape():
    units = (_unit("unit.0"), _unit("unit.1"))
    segment = _segment(_EQUIVALENCE_A)
    plugin = _SyntheticPlugin()
    observations = []
    for index, unit in enumerate(units):
        request = RenderRequest(unit.qname, segment, "A1", "panel")
        scalar = ScalarRenderResult(float(index + 1))
        for _repeat in range(2):
            observations.append(ShapeObservation(
                qname=unit.qname,
                segment=segment,
                format_name="A1",
                predicted_dloss=scalar.predicted_dloss,
                receipt=_receipt(request, scalar, plugin),
            ))
    with pytest.raises(AnchoredCostError, match="design rank is 0 of 1"):
        fit_segment_shape(
            observations,
            segment=segment,
            candidates=_segment_candidates(units[0], _EQUIVALENCE_A),
        )


def test_rank_detection_is_invariant_to_plugin_feature_units():
    units = (_unit("unit.0"), _unit("unit.1"))
    candidates = tuple(
        replace(
            candidate,
            shape_features=(candidate.coordinate * 1e-9,),
        )
        for candidate in _segment_candidates(units[0], _EQUIVALENCE_A)
    )
    fit = fit_segment_shape(
        _observations(units, _EQUIVALENCE_A),
        segment=_segment(_EQUIVALENCE_A),
        candidates=candidates,
    )
    assert fit.design_rank == fit.design_rank_required == 1
    assert fit.ratio("A2", "A0") == pytest.approx(0.25)
