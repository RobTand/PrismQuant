"""Gate for the continuous trellis rate surface (research-only).

The module turns a handful of measured anchors into a dense, exactly-priced
candidate menu.  The things that can go wrong are all failures of honesty
rather than of arithmetic: a rung silently extrapolated beyond what was
measured, an interpolated value reported as measured, two objectives mixed in
one DP, or a non-monotone anchor set laundered into a smooth-looking cost.
Each of those has a test here.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math

import pytest

from prismaquant.trellis_allocator import build_trellis_allocator_candidate
from prismaquant.trellis_formats import (
    E2M1_FAMILY,
    E4M3_FAMILY,
    get_trellis_family,
    native_code_value,
    LAYOUT_TIGHT_OFFSETS,
    TrellisFormatError,
    validate_schedule,
)
from prismaquant.trellis_rate_surface import (
    FIT_MATCHED_SCALAR_GAIN_DB,
    FIT_RAW_DLOSS_LOG2,
    PROVENANCE_INTERPOLATED,
    PROVENANCE_MEASURED,
    SURFACE_USE_ALLOCATOR_COST,
    SURFACE_USE_CAMPAIGN_PLANNING,
    TrellisAllocationRegretGate,
    TrellisCurveIdentity,
    TrellisDensifiedSurface,
    TrellisHoldoutGrade,
    TrellisMeasuredRateAnchor,
    TrellisRateSurface,
    TrellisScalarBackbonePoint,
    allocation_regret,
    densify_rate_surface,
    fit_rate_surface,
    grade_rate_surface_holdout,
    leave_one_anchor_out,
    rate_surface_solver_menu,
    seal_rate_surface_holdout,
    uniform_column_schedule,
)

UNIT = "model.layers.0.mlp.down_proj"
SHAPE = (512, 512)
CURRENCY = "aura_adjoint"
SHA = "1" * 64


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()


def resealed_regret_gate(
    gate: TrellisAllocationRegretGate, **updates: object
) -> TrellisAllocationRegretGate:
    body = {**gate._body(), **updates}
    return TrellisAllocationRegretGate(
        byte_budget=body["byte_budget"],
        max_regret_pct=body["max_regret_pct"],
        regret_pct=body["regret_pct"],
        assignment_agreement=body["assignment_agreement"],
        interpolated_assignment_fraction=body[
            "interpolated_assignment_fraction"
        ],
        true_loss_deciding_on_interpolated=body[
            "true_loss_deciding_on_interpolated"
        ],
        true_loss_deciding_on_truth=body["true_loss_deciding_on_truth"],
        assignment_interpolated=tuple(
            body["assignment_interpolated"].items()
        ),
        assignment_truth=tuple(body["assignment_truth"].items()),
        surface_bindings=tuple(body["surface_bindings"].items()),
        densified_bindings=tuple(body["densified_bindings"].items()),
        truth_bindings=tuple(body["truth_bindings"].items()),
        holdout_grade_sha256=tuple(
            (unit, tuple(digests))
            for unit, digests in body["holdout_grade_sha256"].items()
        ),
        bracket_agreement=body["bracket_agreement"],
        passed=body["passed"],
        gate_sha256=canonical_sha256(body),
    )


def alphabet(rate: int) -> tuple[int, ...]:
    """A distinct, NaN-free E4M3 alphabet, sorted the way the wire demands.

    The wire requires the alphabet ordered by DECODED VALUE then code, which
    is not byte order: 0x80 is -0.0 and sorts below every positive code.
    """

    spec = get_trellis_family(E4M3_FAMILY)
    required = 1 << (rate + 1)
    codes = [code for code in range(256) if code not in (0x7F, 0xFF)]
    if required > len(codes):
        # Rate 7 wants 256 slots from a 254-value finite grid.  The wire
        # allows exactly this corner by spending both zero byte patterns.
        codes.extend((0x00, 0x80))
        required = len(codes)
    codes.sort(key=lambda code: (native_code_value(spec, code), code))
    # Take a contiguous run around zero so the subset stays value-ordered.
    start = (len(codes) - required) // 2
    return tuple(codes[start:start + required])


def alphabets_for(schedule) -> dict[int, tuple[int, ...]]:
    return {rate: alphabet(rate) for rate in sorted(set(schedule)) if rate < 8}


def anchor(
    rate_q256: int,
    dloss: float,
    stderr: float = 0.0,
    *,
    unit_name: str = UNIT,
):
    schedule = uniform_column_schedule(
        SHAPE[1], rate_q256, family=E4M3_FAMILY,
    )
    return build_trellis_allocator_candidate(
        unit_name,
        SHAPE,
        family=E4M3_FAMILY,
        body_rate_q256=rate_q256,
        layout=LAYOUT_TIGHT_OFFSETS,
        schedule=schedule,
        alphabets=alphabets_for(schedule),
        predicted_dloss=dloss,
        predicted_dloss_stderr=stderr,
        variant_label=PROVENANCE_MEASURED,
    )


def recipe_sha256(rate_q256: int, *, unit_name: str = UNIT) -> str:
    return str(
        anchor(rate_q256, 1.0, unit_name=unit_name).footprint[
            "pre_render_recipe_identity_sha256"
        ]
    )


def curve_identity(**overrides) -> TrellisCurveIdentity:
    values = {
        "wire_family": E4M3_FAMILY,
        "selector": "exact_dp",
        "alphabet_policy": "canonical_fitted",
        "alphabet_fitting_scope": "per_tensor",
        "scale_plane": get_trellis_family(E4M3_FAMILY).scale_contract,
        "scale_coding": "raw_fp32",
        "encode_tier": "max",
        "schedule_policy": "uniform_bresenham",
        "render_path": "rtn",
        "corpus_id": "glm-corpus-v1",
        "importance_id": "weighted-sse-a16-v1",
        "population": "dense",
        "render_recipe_identity_sha256": "5" * 64,
        "codec_closure_sha256": SHA,
    }
    values.update(overrides)
    return TrellisCurveIdentity(**values)


CURVE = curve_identity()


def backbone(
    rate_q256: int,
    dloss: float,
    *,
    identity: TrellisCurveIdentity = CURVE,
    parity: bool = True,
    closure_sha256: str = "2" * 64,
    currency: str = CURRENCY,
) -> TrellisScalarBackbonePoint:
    return TrellisScalarBackbonePoint(
        curve_identity=identity,
        body_rate_q256=rate_q256,
        dloss=dloss,
        currency=currency,
        backbone_closure_sha256=closure_sha256,
        context_parity_verified=parity,
    )


def measured(
    rate_q256: int,
    dloss: float,
    *,
    backbone_dloss: float | None = None,
    stderr: float = 0.0,
    identity: TrellisCurveIdentity = CURVE,
    parity: bool = True,
    unit_name: str = UNIT,
    backbone_closure_sha256: str = "2" * 64,
    currency: str = CURRENCY,
) -> TrellisMeasuredRateAnchor:
    return TrellisMeasuredRateAnchor(
        candidate=anchor(rate_q256, dloss, stderr, unit_name=unit_name),
        curve_identity=identity,
        currency=currency,
        scalar_backbone=(
            backbone(
                rate_q256,
                backbone_dloss,
                identity=identity,
                parity=parity,
                closure_sha256=backbone_closure_sha256,
                currency=currency,
            )
            if backbone_dloss is not None
            else None
        ),
    )


def measured_truth(
    values: dict[int, float],
    controls: dict[int, TrellisScalarBackbonePoint],
    *,
    identity: TrellisCurveIdentity = CURVE,
    unit_name: str = UNIT,
) -> dict[int, TrellisMeasuredRateAnchor]:
    return {
        rate: measured(
            rate,
            dloss,
            backbone_dloss=controls[rate].dloss,
            identity=identity,
            unit_name=unit_name,
            backbone_closure_sha256=(
                controls[rate].backbone_closure_sha256
            ),
        )
        for rate, dloss in values.items()
    }


def surface(anchors=((1024, 1.0), (1280, 0.25), (1536, 0.0625))):
    return fit_rate_surface(
        [measured(rate, dloss) for rate, dloss in anchors],
        currency=CURRENCY,
        curve_identity=CURVE,
        fit_response=FIT_RAW_DLOSS_LOG2,
    )


def gain_surface(
    unit_anchors=((1024, 1.0, 2.0), (1536, 0.25, 0.5)),
    *,
    unit_name: str = UNIT,
    identity: TrellisCurveIdentity = CURVE,
) -> TrellisRateSurface:
    return fit_rate_surface(
        [
            measured(
                rate,
                dloss,
                backbone_dloss=backbone_dloss,
                unit_name=unit_name,
                identity=identity,
            )
            for rate, dloss, backbone_dloss in unit_anchors
        ],
        currency=CURRENCY,
        curve_identity=identity,
        fit_response=FIT_MATCHED_SCALAR_GAIN_DB,
    )


def allocator_ready_bundle(
    *,
    unit_name: str = UNIT,
    identity: TrellisCurveIdentity = CURVE,
    max_regret_pct: float = 0.0,
):
    """One exact gain fit, excluded holdout, regret gate, and dense menu."""

    rd = gain_surface(unit_name=unit_name, identity=identity)
    controls = {
        1024: backbone(1024, 2.0, identity=identity),
        1280: backbone(1280, 1.0, identity=identity),
        1536: backbone(1536, 0.5, identity=identity),
    }
    seal = seal_rate_surface_holdout(
        rd,
        1280,
        pre_render_recipe_identity_sha256=recipe_sha256(
            1280, unit_name=unit_name
        ),
        scalar_backbone=controls[1280],
    )
    holdout = measured(
        1280,
        0.5,
        backbone_dloss=1.0,
        identity=identity,
        unit_name=unit_name,
    )
    grade = grade_rate_surface_holdout(rd, seal, holdout)
    built = densify_rate_surface(
        rd,
        SHAPE,
        q256_values=(1024, 1280, 1536),
        alphabets={rate: alphabet(rate) for rate in range(1, 8)},
        intended_use=SURFACE_USE_ALLOCATOR_COST,
        schedule_policy=identity.schedule_policy,
        scalar_backbone=controls,
    )
    exact_bytes = {
        candidate.body_rate_q256: candidate.footprint["total_bytes"]
        for candidate in built
    }
    gate = allocation_regret(
        {unit_name: rd},
        {
            unit_name: measured_truth(
                {1024: 1.0, 1280: 0.5, 1536: 0.25},
                controls,
                identity=identity,
                unit_name=unit_name,
            )
        },
        densified={unit_name: built},
        scalar_backbones={unit_name: controls},
        holdout_grades={unit_name: (grade,)},
        byte_budget=exact_bytes[1280],
        max_regret_pct=max_regret_pct,
        bracket_agreement=True,
    )
    return rd, built, gate, grade, controls


def misranked_allocator_bundle(*, cross_unit_identity_drift: bool = False):
    """Two units whose measured holdouts expose an allocation misranking."""

    identities = {name: CURVE for name in ("u0", "u1")}
    if cross_unit_identity_drift:
        identities["u1"] = curve_identity(corpus_id="different-corpus")
    surfaces = {
        "u0": gain_surface(
            ((1024, 1.0, 2.0), (1536, 0.8, 1.6)),
            unit_name="u0",
            identity=identities["u0"],
        ),
        "u1": gain_surface(
            ((1024, 1.0, 2.0), (1536, 0.1, 0.2)),
            unit_name="u1",
            identity=identities["u1"],
        ),
    }
    truth_values = {
        "u0": {1024: 1.0, 1280: 0.90, 1536: 0.80},
        "u1": {1024: 1.0, 1280: 0.99, 1536: 0.10},
    }
    control_values = {
        "u0": {1024: 2.0, 1280: 1.8, 1536: 1.6},
        "u1": {1024: 2.0, 1280: 1.0, 1536: 0.2},
    }
    controls = {
        unit: {
            rate: backbone(rate, value, identity=identities[unit])
            for rate, value in values.items()
        }
        for unit, values in control_values.items()
    }
    grades = {}
    densified = {}
    exact_bytes = {}
    for unit in ("u0", "u1"):
        seal = seal_rate_surface_holdout(
            surfaces[unit],
            1280,
            pre_render_recipe_identity_sha256=recipe_sha256(
                1280, unit_name=unit
            ),
            scalar_backbone=controls[unit][1280],
        )
        heldout = measured(
            1280,
            truth_values[unit][1280],
            backbone_dloss=control_values[unit][1280],
            identity=identities[unit],
            unit_name=unit,
        )
        grades[unit] = (
            grade_rate_surface_holdout(surfaces[unit], seal, heldout),
        )
        densified[unit] = densify_rate_surface(
            surfaces[unit],
            SHAPE,
            q256_values=(1024, 1280, 1536),
            alphabets={rate: alphabet(rate) for rate in range(1, 8)},
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            schedule_policy=identities[unit].schedule_policy,
            scalar_backbone=controls[unit],
        )
        exact_bytes[unit] = {
            candidate.body_rate_q256: candidate.footprint["total_bytes"]
            for candidate in densified[unit]
        }
    budget = exact_bytes["u0"][1280] + exact_bytes["u1"][1024]
    gate = allocation_regret(
        surfaces,
        {
            unit: measured_truth(
                truth_values[unit],
                controls[unit],
                identity=identities[unit],
                unit_name=unit,
            )
            for unit in truth_values
        },
        densified=densified,
        scalar_backbones=controls,
        holdout_grades=grades,
        byte_budget=budget,
        max_regret_pct=0.0,
        bracket_agreement=True,
    )
    return surfaces, densified, gate


# --------------------------------------------------------------------------
# The schedule constructor: the rate is interpolated, the BYTES never are.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rate_q256", [1024, 1088, 1152, 1280, 1536, 1792])
def test_uniform_schedule_is_wire_legal_and_hits_the_rate(rate_q256):
    schedule = uniform_column_schedule(512, rate_q256, family=E4M3_FAMILY)
    # The wire itself is the authority on legality, not this test's opinion.
    validate_schedule(
        E4M3_FAMILY, rate_q256, schedule, layout=LAYOUT_TIGHT_OFFSETS,
    )
    assert len(schedule) == 512
    # tight_offsets tolerates less than one physical body bit of drift.
    assert abs(sum(schedule) - rate_q256 * 512 / 256) < 1.0


# E2M1 is where the low-bpp mandate lives, and it is the harder family for
# this constructor: bypass_rate is 4, so a block packed with rate-4 columns
# has zero coded steps and the wire refuses it.  These rates are inside the
# research range [256, 1016], not an unused corner.
@pytest.mark.parametrize("rate_q256", [384, 512, 640, 768, 896, 1008])
def test_uniform_schedule_is_wire_legal_on_e2m1(rate_q256):
    schedule = uniform_column_schedule(512, rate_q256, family=E2M1_FAMILY)
    validate_schedule(
        E2M1_FAMILY, rate_q256, schedule, layout=LAYOUT_TIGHT_OFFSETS,
    )
    assert abs(sum(schedule) - rate_q256 * 512 / 256) < 1.0
    # Every superblock keeps enough coded steps, which is only true because
    # the remainder is spread rather than packed at the front.
    for start in range(0, 512, 256):
        block = schedule[start:start + 256]
        assert sum(value < 4 for value in block) >= 8


def test_remainder_is_spread_not_packed():
    # Packing the dearer columns contiguously would give block 0 all of them.
    schedule = uniform_column_schedule(512, 896, family=E2M1_FAMILY)
    first = sum(schedule[:256])
    second = sum(schedule[256:])
    assert abs(first - second) <= 1


def test_uniform_schedule_is_flat_within_one_bit():
    schedule = uniform_column_schedule(512, 1152, family=E4M3_FAMILY)
    assert max(schedule) - min(schedule) <= 1
    assert set(schedule) == {4, 5}


def test_uniform_schedule_refuses_a_partial_block():
    # A short final block is legal on the wire, but its rate accounting is
    # the caller's to declare; guessing it here would be a silent choice.
    with pytest.raises(TrellisFormatError, match="multiple of 256"):
        uniform_column_schedule(500, 1152, family=E4M3_FAMILY)


def test_uniform_schedule_refuses_rates_needing_illegal_columns():
    with pytest.raises(TrellisFormatError):
        uniform_column_schedule(512, 200, family=E4M3_FAMILY)


# --------------------------------------------------------------------------
# The surface contract.
# --------------------------------------------------------------------------


def test_one_anchor_is_a_point_not_a_surface():
    with pytest.raises(TrellisFormatError, match="at least two"):
        fit_rate_surface(
            [measured(1024, 1.0)],
            currency=CURRENCY,
            curve_identity=CURVE,
            fit_response=FIT_RAW_DLOSS_LOG2,
        )


def test_non_monotone_anchors_are_refused_not_smoothed():
    # A rate that costs MORE loss than a cheaper one is a measurement
    # problem.  Interpolating through it would launder it into a cost.
    with pytest.raises(TrellisFormatError, match="strictly decrease"):
        fit_rate_surface(
            [measured(1024, 0.25), measured(1280, 1.0)],
            currency=CURRENCY,
            curve_identity=CURVE,
            fit_response=FIT_RAW_DLOSS_LOG2,
        )


def test_anchors_may_not_repeat_a_rate():
    with pytest.raises(TrellisFormatError, match="repeat a rate"):
        fit_rate_surface(
            [measured(1024, 1.0), measured(1024, 0.5)],
            currency=CURRENCY,
            curve_identity=CURVE,
            fit_response=FIT_RAW_DLOSS_LOG2,
        )


def test_currency_must_be_declared():
    with pytest.raises(TrellisFormatError, match="currency"):
        fit_rate_surface(
            [measured(1024, 1.0), measured(1280, 0.25)],
            currency="",
            curve_identity=CURVE,
            fit_response=FIT_RAW_DLOSS_LOG2,
        )


def test_measured_anchor_currency_cannot_be_relabelled_at_fit_time():
    with pytest.raises(TrellisFormatError, match="currency drifted"):
        fit_rate_surface(
            [
                measured(1024, 1.0, currency="output_mse"),
                measured(1280, 0.25, currency="output_mse"),
            ],
            currency=CURRENCY,
            curve_identity=CURVE,
            fit_response=FIT_RAW_DLOSS_LOG2,
        )


def test_predict_reproduces_anchors_exactly():
    rd = surface()
    for rate, dloss in ((1024, 1.0), (1280, 0.25), (1536, 0.0625)):
        assert rd.predict(
            rate, intended_use=SURFACE_USE_CAMPAIGN_PLANNING
        ) == pytest.approx(dloss, rel=0, abs=0)
        assert rd.provenance(rate) == PROVENANCE_MEASURED


def test_predict_interpolates_in_log_space():
    rd = surface()
    # Midway between 1.0 and 0.25 in log2 is the geometric mean, 0.5 --
    # not the arithmetic mean 0.625.  Distortion decays geometrically with
    # rate, so a linear-in-D interpolation would be biased high everywhere.
    assert rd.predict(
        1152, intended_use=SURFACE_USE_CAMPAIGN_PLANNING
    ) == pytest.approx(0.5, rel=1e-12)
    assert rd.provenance(1152) == PROVENANCE_INTERPOLATED


def test_extrapolation_is_refused_never_guessed():
    rd = surface()
    for outside in (1023, 1537, 2040):
        with pytest.raises(TrellisFormatError, match="outside the measured"):
            rd.predict(
                outside, intended_use=SURFACE_USE_CAMPAIGN_PLANNING
            )


def test_interpolated_stderr_never_shrinks_by_interpolating():
    rd = fit_rate_surface(
        [measured(1024, 1.0, stderr=0.01), measured(1280, 0.25, stderr=0.4)],
        currency=CURRENCY,
        curve_identity=CURVE,
        fit_response=FIT_RAW_DLOSS_LOG2,
    )
    assert rd.predict_stderr(
        1024, intended_use=SURFACE_USE_CAMPAIGN_PLANNING
    ) == pytest.approx(0.01)
    assert rd.predict_stderr(
        1280, intended_use=SURFACE_USE_CAMPAIGN_PLANNING
    ) == pytest.approx(0.4)
    # Between them it inherits the WIDER bracketing anchor, not the nearer.
    assert rd.predict_stderr(
        1088, intended_use=SURFACE_USE_CAMPAIGN_PLANNING
    ) == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selector", "beam_v2"),
        ("alphabet_policy", "learned_nested"),
        ("alphabet_fitting_scope", "per_block"),
        ("scale_coding", "delta_varint"),
        ("encode_tier", "fast"),
        ("schedule_policy", "importance_ranked"),
        ("render_path", "gptq"),
        ("corpus_id", "other-corpus"),
        ("importance_id", "other-importance"),
        ("population", "routed"),
        ("render_recipe_identity_sha256", "6" * 64),
        ("codec_closure_sha256", "3" * 64),
    ],
)
def test_every_curve_identity_dimension_changes_the_identity_and_refuses_drift(
    field, value
):
    drifted = curve_identity(**{field: value})
    assert drifted.identity_sha256 != CURVE.identity_sha256
    with pytest.raises(TrellisFormatError, match="identity drifted"):
        fit_rate_surface(
            [measured(1024, 1.0), measured(1536, 0.25)],
            currency=CURRENCY,
            curve_identity=drifted,
            fit_response=FIT_RAW_DLOSS_LOG2,
        )


def test_wire_family_and_scale_plane_cross_family_mix_is_refused():
    e2_identity = curve_identity(
        wire_family=E2M1_FAMILY,
        scale_plane=get_trellis_family(E2M1_FAMILY).scale_contract,
    )
    assert e2_identity.identity_sha256 != CURVE.identity_sha256
    with pytest.raises(TrellisFormatError, match="wire family"):
        TrellisMeasuredRateAnchor(
            candidate=anchor(1024, 1.0),
            curve_identity=e2_identity,
            currency=CURRENCY,
        )


def test_bare_legacy_anchors_cannot_be_given_invented_provenance():
    with pytest.raises(TrellisFormatError, match="identity-bound"):
        fit_rate_surface(
            [anchor(1024, 1.0), anchor(1536, 0.25)],
            currency=CURRENCY,
            curve_identity=CURVE,
            fit_response=FIT_RAW_DLOSS_LOG2,
        )


def test_identity_wrapper_cannot_relabel_an_unmarked_candidate_as_measured():
    schedule = uniform_column_schedule(512, 1024, family=E4M3_FAMILY)
    unmarked = build_trellis_allocator_candidate(
        UNIT,
        SHAPE,
        family=E4M3_FAMILY,
        body_rate_q256=1024,
        layout=LAYOUT_TIGHT_OFFSETS,
        schedule=schedule,
        alphabets=alphabets_for(schedule),
        predicted_dloss=1.0,
        predicted_dloss_stderr=0.0,
    )
    with pytest.raises(TrellisFormatError, match="explicitly labelled measured"):
        TrellisMeasuredRateAnchor(
            candidate=unmarked,
            curve_identity=CURVE,
            currency=CURRENCY,
        )


@pytest.mark.parametrize(
    "identity",
    [
        curve_identity(selector="qtip_exact"),
        curve_identity(render_path="ldlq"),
        curve_identity(render_path="rotated"),
    ],
)
def test_qtip_ldlq_and_rotated_cannot_borrow_unbound_zero_anchors(identity):
    with pytest.raises(TrellisFormatError, match="identity-bound"):
        fit_rate_surface(
            [anchor(1024, 1.0), anchor(1536, 0.25)],
            currency=CURRENCY,
            curve_identity=identity,
            fit_response=FIT_RAW_DLOSS_LOG2,
        )


def test_gain_fit_requires_matched_backbone_and_context_parity():
    with pytest.raises(TrellisFormatError, match="missing scalar backbone"):
        fit_rate_surface(
            [measured(1024, 1.0), measured(1536, 0.25)],
            currency=CURRENCY,
            curve_identity=CURVE,
            fit_response=FIT_MATCHED_SCALAR_GAIN_DB,
        )
    with pytest.raises(TrellisFormatError, match="context parity"):
        fit_rate_surface(
            [
                measured(1024, 1.0, backbone_dloss=2.0),
                measured(1536, 0.25, backbone_dloss=0.5, parity=False),
            ],
            currency=CURRENCY,
            curve_identity=CURVE,
            fit_response=FIT_MATCHED_SCALAR_GAIN_DB,
        )


def test_gain_fit_and_prediction_bind_one_scalar_backbone_closure():
    with pytest.raises(TrellisFormatError, match="different backbone closures"):
        fit_rate_surface(
            [
                measured(1024, 1.0, backbone_dloss=2.0),
                measured(
                    1536,
                    0.25,
                    backbone_dloss=0.5,
                    backbone_closure_sha256="4" * 64,
                ),
            ],
            currency=CURRENCY,
            curve_identity=CURVE,
            fit_response=FIT_MATCHED_SCALAR_GAIN_DB,
        )
    rd = gain_surface()
    with pytest.raises(TrellisFormatError, match="closure differs"):
        rd.predict(
            1280,
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            scalar_backbone=backbone(
                1280, 1.0, closure_sha256="4" * 64
            ),
        )


def test_nonmonotone_scalar_backbone_anchors_are_refused():
    with pytest.raises(TrellisFormatError, match="backbone anchor loss"):
        fit_rate_surface(
            [
                measured(1024, 1.0, backbone_dloss=1.0),
                measured(1536, 0.25, backbone_dloss=1.1),
            ],
            currency=CURRENCY,
            curve_identity=CURVE,
            fit_response=FIT_MATCHED_SCALAR_GAIN_DB,
        )


def test_gain_anchor_prediction_requires_its_exact_measured_scalar_point():
    with pytest.raises(TrellisFormatError, match="different measured scalar"):
        gain_surface().predict(
            1024,
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            scalar_backbone=backbone(1024, 2.1),
        )


def test_raw_fit_does_not_silently_discard_a_supplied_backbone():
    with pytest.raises(TrellisFormatError, match="fit their gain"):
        fit_rate_surface(
            [
                measured(1024, 1.0, backbone_dloss=2.0),
                measured(1536, 0.25, backbone_dloss=0.5),
            ],
            currency=CURRENCY,
            curve_identity=CURVE,
            fit_response=FIT_RAW_DLOSS_LOG2,
        )


@pytest.mark.parametrize(
    "forbidden_use", ["menu_verdict", "family_verdict", "publication"]
)
def test_menu_family_and_publication_uses_are_closed(forbidden_use):
    with pytest.raises(TrellisFormatError, match="available only"):
        gain_surface().predict(
            1280,
            intended_use=forbidden_use,
            scalar_backbone=backbone(1280, 1.0),
        )


def test_raw_dloss_is_explicitly_planning_only():
    with pytest.raises(TrellisFormatError, match="planning-only"):
        surface().predict(1152, intended_use=SURFACE_USE_ALLOCATOR_COST)


def test_gain_prediction_requires_the_same_context_complete_backbone():
    rd = gain_surface()
    with pytest.raises(TrellisFormatError, match="requires a matched"):
        rd.predict(1280, intended_use=SURFACE_USE_ALLOCATOR_COST)
    with pytest.raises(TrellisFormatError, match="context parity"):
        rd.predict(
            1280,
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            scalar_backbone=backbone(1280, 1.0, parity=False),
        )
    with pytest.raises(TrellisFormatError, match="identity differs"):
        rd.predict(
            1280,
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            scalar_backbone=backbone(
                1280,
                1.0,
                identity=curve_identity(corpus_id="other-corpus"),
            ),
        )


def test_gain_prediction_refuses_a_scalar_control_in_another_currency():
    with pytest.raises(TrellisFormatError, match="currency differs"):
        gain_surface().predict(
            1280,
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            scalar_backbone=backbone(
                1280, 1.0, currency="output_mse"
            ),
        )


# --------------------------------------------------------------------------
# Densification: exact bytes, honest labels.
# --------------------------------------------------------------------------


def test_densify_prices_every_rung_with_real_bytes():
    rd = surface()
    rungs = (1024, 1088, 1152, 1216, 1280)
    schedule = uniform_column_schedule(512, 1152, family=E4M3_FAMILY)
    built = densify_rate_surface(
        rd,
        SHAPE,
        q256_values=rungs,
        alphabets=alphabets_for(schedule),
        intended_use=SURFACE_USE_CAMPAIGN_PLANNING,
        schedule_policy=CURVE.schedule_policy,
    )
    assert len(built) == len(rungs)
    assert [record.body_rate_q256 for record in built] == list(rungs)
    # Bytes are exact and strictly increase with rate: nothing about the
    # footprint is interpolated, only the loss.
    footprints = [record.footprint["total_bytes"] for record in built]
    assert footprints == sorted(footprints)
    assert len(set(footprints)) == len(footprints)


def test_densify_labels_interpolated_rungs_as_such():
    rd = surface()
    built = densify_rate_surface(
        rd,
        SHAPE,
        q256_values=(1024, 1152, 1280),
        alphabets=alphabets_for(
            uniform_column_schedule(512, 1152, family=E4M3_FAMILY)
        ),
        intended_use=SURFACE_USE_CAMPAIGN_PLANNING,
        schedule_policy=CURVE.schedule_policy,
    )
    labels = {
        record.body_rate_q256: record.variant_label for record in built
    }
    assert labels[1024] == PROVENANCE_MEASURED
    assert labels[1280] == PROVENANCE_MEASURED
    assert labels[1152] == PROVENANCE_INTERPOLATED


def test_densify_refuses_schedule_identity_drift():
    with pytest.raises(TrellisFormatError, match="schedule policy differs"):
        densify_rate_surface(
            surface(),
            SHAPE,
            q256_values=(1024, 1280),
            alphabets={rate: alphabet(rate) for rate in range(1, 8)},
            intended_use=SURFACE_USE_CAMPAIGN_PLANNING,
            schedule_policy="importance_ranked",
        )


def test_densify_refuses_tensor_shape_drift_from_measured_anchors():
    with pytest.raises(TrellisFormatError, match="shape differs"):
        densify_rate_surface(
            surface(),
            (256, 512),
            q256_values=(1024, 1280),
            alphabets={rate: alphabet(rate) for rate in range(1, 8)},
            intended_use=SURFACE_USE_CAMPAIGN_PLANNING,
            schedule_policy=CURVE.schedule_policy,
        )


def test_densify_refuses_target_profile_context_drift():
    with pytest.raises(TrellisFormatError, match="target profile differs"):
        densify_rate_surface(
            surface(),
            SHAPE,
            q256_values=(1024, 1280),
            alphabets={rate: alphabet(rate) for rate in range(1, 8)},
            intended_use=SURFACE_USE_CAMPAIGN_PLANNING,
            schedule_policy=CURVE.schedule_policy,
            target_profile="trellis_research_sm121",
        )


@pytest.mark.parametrize("requested", [(), (1024, 1024)])
def test_densify_refuses_empty_or_duplicate_requested_rungs(requested):
    with pytest.raises(TrellisFormatError, match="at least one|repeat"):
        densify_rate_surface(
            surface(),
            SHAPE,
            q256_values=requested,
            alphabets={rate: alphabet(rate) for rate in range(1, 8)},
            intended_use=SURFACE_USE_CAMPAIGN_PLANNING,
            schedule_policy=CURVE.schedule_policy,
        )


def test_campaign_planning_densification_cannot_enter_solver_menu():
    planned = densify_rate_surface(
        surface(),
        SHAPE,
        q256_values=(1024, 1280),
        alphabets={rate: alphabet(rate) for rate in range(1, 8)},
        intended_use=SURFACE_USE_CAMPAIGN_PLANNING,
        schedule_policy=CURVE.schedule_policy,
    )
    _rd, _built, gate, _grade, _controls = allocator_ready_bundle()
    with pytest.raises(TrellisFormatError, match="allocator-cost research"):
        rate_surface_solver_menu(
            {UNIT: planned},
            currencies={UNIT: CURRENCY},
            intended_use=SURFACE_USE_CAMPAIGN_PLANNING,
            regret_gate=gate,
            byte_budget=gate.byte_budget,
        )


def test_menu_refuses_to_mix_objectives():
    _rd, built, gate, _grade, _controls = allocator_ready_bundle()
    with pytest.raises(TrellisFormatError, match="one currency"):
        rate_surface_solver_menu(
            {"a": built, "b": built},
            currencies={"a": "aura_adjoint", "b": "output_mse"},
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            regret_gate=gate,
            byte_budget=gate.byte_budget,
        )


def test_menu_builds_for_one_declared_currency():
    _rd, built, gate, _grade, _controls = allocator_ready_bundle()
    menu = rate_surface_solver_menu(
        {UNIT: built},
        currencies={UNIT: CURRENCY},
        intended_use=SURFACE_USE_ALLOCATOR_COST,
        regret_gate=gate,
        byte_budget=gate.byte_budget,
    )
    assert set(menu) == {UNIT}
    assert len(menu[UNIT]) == 3


def test_regret_gate_cannot_be_replayed_at_a_different_byte_budget():
    _rd, built, gate, _grade, _controls = allocator_ready_bundle()
    with pytest.raises(TrellisFormatError, match="budget differs"):
        rate_surface_solver_menu(
            {UNIT: built},
            currencies={UNIT: CURRENCY},
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            regret_gate=gate,
            byte_budget=gate.byte_budget + 1,
        )


def test_regret_gate_cannot_authorize_a_different_exact_candidate_menu():
    rd, _built, gate, _grade, controls = allocator_ready_bundle()
    changed = densify_rate_surface(
        rd,
        SHAPE,
        q256_values=(1024, 1536),
        alphabets={rate: alphabet(rate) for rate in range(1, 8)},
        intended_use=SURFACE_USE_ALLOCATOR_COST,
        schedule_policy=CURVE.schedule_policy,
        scalar_backbone={rate: controls[rate] for rate in (1024, 1536)},
    )
    with pytest.raises(TrellisFormatError, match="exact candidate menu differs"):
        rate_surface_solver_menu(
            {UNIT: changed},
            currencies={UNIT: CURRENCY},
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            regret_gate=gate,
            byte_budget=gate.byte_budget,
        )


# --------------------------------------------------------------------------
# Diagnostic vs gate.
# --------------------------------------------------------------------------


def test_loo_is_near_zero_on_a_log_linear_surface():
    # dloss halves every 128 q256: interpolation is exact by construction,
    # so any residual here is the implementation's own error.
    anchors = tuple(
        (1024 + 128 * step, 2.0 ** (-step)) for step in range(5)
    )
    report = leave_one_anchor_out(surface(anchors))
    assert report["interior_anchors"] == 3
    assert report["max_abs_log2_error"] < 1e-12


def test_loo_exposes_saturation_the_way_the_ladder_shows_it():
    # The real E4M3 ladder saturates against the scalar ceiling: ~5.4 dB per
    # bit, then 2.22 for the last step.  A surface shaped like that must
    # report a real interpolation error, not hide it.
    anchors = (
        (1024, 1.0),
        (1280, 0.29),
        (1536, 0.085),
        (1792, 0.051),
    )
    report = leave_one_anchor_out(surface(anchors))
    assert report["max_abs_log2_error"] > 0.1


def test_regret_is_zero_when_the_surface_is_exact():
    _surface, _built, gate, _grade, _controls = allocator_ready_bundle()
    assert gate["regret_pct"] == pytest.approx(0.0)
    assert gate["assignment_agreement"] == 1.0
    assert gate.passed


def test_regret_is_reported_when_interpolation_misranks():
    # u1's true curve has a kink the two anchors cannot see, so the surface
    # overstates the value of its middle rung and the allocator misspends.
    _surfaces, _densified, report = misranked_allocator_bundle()
    # The point is that regret is REPORTED and finite, not that it is large:
    # the one-anchor campaign's lesson is that a noisy surrogate can still
    # make near-omniscient decisions.
    assert report["regret_pct"] >= 0.0
    assert 0.0 <= report["assignment_agreement"] <= 1.0
    assert math.isfinite(report["true_loss_deciding_on_interpolated"])
    assert not report.passed


def test_holdout_must_be_sealed_at_an_excluded_interior_rung():
    rd = gain_surface()
    for rate in (1024, 1536):
        with pytest.raises(TrellisFormatError, match="strictly interior"):
            seal_rate_surface_holdout(
                rd,
                rate,
                pre_render_recipe_identity_sha256=recipe_sha256(rate),
                scalar_backbone=backbone(rate, 2.0),
            )
    three_anchor = fit_rate_surface(
        [
            measured(1024, 1.0, backbone_dloss=2.0),
            measured(1280, 0.5, backbone_dloss=1.0),
            measured(1536, 0.25, backbone_dloss=0.5),
        ],
        currency=CURRENCY,
        curve_identity=CURVE,
        fit_response=FIT_MATCHED_SCALAR_GAIN_DB,
    )
    with pytest.raises(TrellisFormatError, match="fitted anchor"):
        seal_rate_surface_holdout(
            three_anchor,
            1280,
            pre_render_recipe_identity_sha256=recipe_sha256(1280),
            scalar_backbone=backbone(1280, 1.0),
        )


def test_holdout_grade_consumes_the_exact_sealed_backbone_point():
    rd = gain_surface()
    seal = seal_rate_surface_holdout(
        rd,
        1280,
        pre_render_recipe_identity_sha256=recipe_sha256(1280),
        scalar_backbone=backbone(1280, 1.0),
    )
    mismatched = measured(1280, 0.5, backbone_dloss=1.1)
    with pytest.raises(TrellisFormatError, match="different scalar backbone"):
        grade_rate_surface_holdout(rd, seal, mismatched)


def test_self_resealed_holdout_grade_cannot_change_the_surface_prediction():
    rd = gain_surface()
    controls = {
        1024: backbone(1024, 2.0),
        1280: backbone(1280, 1.0),
        1536: backbone(1536, 0.5),
    }
    seal = seal_rate_surface_holdout(
        rd,
        1280,
        pre_render_recipe_identity_sha256=recipe_sha256(1280),
        scalar_backbone=controls[1280],
    )
    measured_holdout = measured(
        1280, 0.6, backbone_dloss=controls[1280].dloss
    )
    grade = grade_rate_surface_holdout(rd, seal, measured_holdout)
    built = densify_rate_surface(
        rd,
        SHAPE,
        q256_values=(1024, 1280, 1536),
        alphabets={rate: alphabet(rate) for rate in range(1, 8)},
        intended_use=SURFACE_USE_ALLOCATOR_COST,
        schedule_policy=CURVE.schedule_policy,
        scalar_backbone=controls,
    )
    changed_prediction = grade.measured_dloss
    seal_body = {
        "schema": "prismaquant.trellis_rate_holdout_seal.v1",
        "surface_sha256": grade.surface_sha256,
        "curve_identity_sha256": grade.curve_identity_sha256,
        "body_rate_q256": grade.body_rate_q256,
        "pre_render_recipe_identity_sha256": (
            grade.pre_render_recipe_identity_sha256
        ),
        "predicted_dloss": changed_prediction,
        "scalar_backbone_point_sha256": (
            grade.scalar_backbone_point_sha256
        ),
        "prediction_precedes_measurement": True,
        "research_only": True,
        "publication_eligible": False,
    }
    forged_body = {
        **grade._body(),
        "seal_sha256": canonical_sha256(seal_body),
        "predicted_dloss": changed_prediction,
        "log2_error": 0.0,
        "relative_error_pct": 0.0,
    }
    forged = TrellisHoldoutGrade(
        surface_sha256=forged_body["surface_sha256"],
        curve_identity_sha256=forged_body["curve_identity_sha256"],
        body_rate_q256=forged_body["body_rate_q256"],
        seal_sha256=forged_body["seal_sha256"],
        measured_anchor_sha256=forged_body["measured_anchor_sha256"],
        pre_render_recipe_identity_sha256=forged_body[
            "pre_render_recipe_identity_sha256"
        ],
        scalar_backbone_point_sha256=forged_body[
            "scalar_backbone_point_sha256"
        ],
        predicted_dloss=forged_body["predicted_dloss"],
        measured_dloss=forged_body["measured_dloss"],
        log2_error=forged_body["log2_error"],
        relative_error_pct=forged_body["relative_error_pct"],
        grade_sha256=canonical_sha256(forged_body),
    )
    exact_bytes = {
        candidate.body_rate_q256: candidate.footprint["total_bytes"]
        for candidate in built
    }
    with pytest.raises(TrellisFormatError, match="prediction differs"):
        allocation_regret(
            {UNIT: rd},
            {
                UNIT: measured_truth(
                    {1024: 1.0, 1280: 0.6, 1536: 0.25}, controls
                )
            },
            densified={UNIT: built},
            scalar_backbones={UNIT: controls},
            holdout_grades={UNIT: (forged,)},
            byte_budget=exact_bytes[1280],
            max_regret_pct=0.0,
            bracket_agreement=True,
        )


def test_holdout_grade_rejects_a_self_resealed_wrong_residual():
    _rd, _built, _gate, grade, _controls = allocator_ready_bundle()
    changed = {**grade._body(), "log2_error": grade.log2_error + 1.0}
    with pytest.raises(TrellisFormatError, match="residuals disagree"):
        TrellisHoldoutGrade(
            surface_sha256=changed["surface_sha256"],
            curve_identity_sha256=changed["curve_identity_sha256"],
            body_rate_q256=changed["body_rate_q256"],
            seal_sha256=changed["seal_sha256"],
            measured_anchor_sha256=changed["measured_anchor_sha256"],
            pre_render_recipe_identity_sha256=changed[
                "pre_render_recipe_identity_sha256"
            ],
            scalar_backbone_point_sha256=changed[
                "scalar_backbone_point_sha256"
            ],
            predicted_dloss=changed["predicted_dloss"],
            measured_dloss=changed["measured_dloss"],
            log2_error=changed["log2_error"],
            relative_error_pct=changed["relative_error_pct"],
            grade_sha256=canonical_sha256(changed),
        )


def test_holdout_grade_refuses_a_different_wire_recipe_than_precommitted():
    rd = gain_surface()
    seal = seal_rate_surface_holdout(
        rd,
        1280,
        pre_render_recipe_identity_sha256=recipe_sha256(1280),
        scalar_backbone=backbone(1280, 1.0),
    )
    schedule = uniform_column_schedule(512, 1280, family=E4M3_FAMILY)
    shifted = alphabet(5)[1:] + (alphabet(5)[-1] + 1,)
    different_recipe = build_trellis_allocator_candidate(
        UNIT,
        SHAPE,
        family=E4M3_FAMILY,
        body_rate_q256=1280,
        layout=LAYOUT_TIGHT_OFFSETS,
        schedule=schedule,
        alphabets={5: shifted},
        predicted_dloss=0.5,
        predicted_dloss_stderr=0.0,
        variant_label=PROVENANCE_MEASURED,
    )
    measured_different_recipe = TrellisMeasuredRateAnchor(
        candidate=different_recipe,
        curve_identity=CURVE,
        currency=CURRENCY,
        scalar_backbone=backbone(1280, 1.0),
    )
    with pytest.raises(TrellisFormatError, match="precommitted wire recipe"):
        grade_rate_surface_holdout(rd, seal, measured_different_recipe)


def test_holdout_measurement_cannot_cross_allocator_units():
    rd = gain_surface()
    seal = seal_rate_surface_holdout(
        rd,
        1280,
        pre_render_recipe_identity_sha256=recipe_sha256(1280),
        scalar_backbone=backbone(1280, 1.0),
    )
    wrong_unit = measured(
        1280,
        0.5,
        backbone_dloss=1.0,
        unit_name="model.layers.1.mlp.down_proj",
    )
    with pytest.raises(TrellisFormatError, match="unit/shape/wire context"):
        grade_rate_surface_holdout(rd, seal, wrong_unit)


def test_holdout_measurement_cannot_drift_target_profile():
    rd = gain_surface()
    seal = seal_rate_surface_holdout(
        rd,
        1280,
        pre_render_recipe_identity_sha256=recipe_sha256(1280),
        scalar_backbone=backbone(1280, 1.0),
    )
    schedule = uniform_column_schedule(512, 1280, family=E4M3_FAMILY)
    different_profile = TrellisMeasuredRateAnchor(
        candidate=build_trellis_allocator_candidate(
            UNIT,
            SHAPE,
            family=E4M3_FAMILY,
            body_rate_q256=1280,
            layout=LAYOUT_TIGHT_OFFSETS,
            schedule=schedule,
            alphabets=alphabets_for(schedule),
            predicted_dloss=0.5,
            predicted_dloss_stderr=0.0,
            target_profile="trellis_research_sm121",
            variant_label=PROVENANCE_MEASURED,
        ),
        curve_identity=CURVE,
        currency=CURRENCY,
        scalar_backbone=backbone(1280, 1.0),
    )
    with pytest.raises(TrellisFormatError, match="unit/shape/wire context"):
        grade_rate_surface_holdout(rd, seal, different_profile)


def test_raw_holdout_cannot_claim_scalar_backbone_provenance():
    with pytest.raises(TrellisFormatError, match="unused scalar backbone"):
        seal_rate_surface_holdout(
            surface(),
            1152,
            pre_render_recipe_identity_sha256=recipe_sha256(1152),
            scalar_backbone=backbone(1152, 0.75),
        )


def test_bracket_disagreement_fails_before_a_regret_gate_exists():
    rd, built, _gate, grade, controls = allocator_ready_bundle()
    exact_bytes = {
        candidate.body_rate_q256: candidate.footprint["total_bytes"]
        for candidate in built
    }
    with pytest.raises(TrellisFormatError, match="bracket disagreement"):
        allocation_regret(
            {UNIT: rd},
            {
                UNIT: measured_truth(
                    {1024: 1.0, 1280: 0.5, 1536: 0.25}, controls
                )
            },
            densified={UNIT: built},
            scalar_backbones={UNIT: controls},
            holdout_grades={UNIT: (grade,)},
            byte_budget=exact_bytes[1280],
            max_regret_pct=0.0,
            bracket_agreement=False,
        )


def test_allocation_gate_requires_a_measured_holdout_for_every_surface():
    rd, built, _gate, _grade, controls = allocator_ready_bundle()
    exact_bytes = {
        candidate.body_rate_q256: candidate.footprint["total_bytes"]
        for candidate in built
    }
    with pytest.raises(TrellisFormatError, match="no measured interior holdout"):
        allocation_regret(
            {UNIT: rd},
            {
                UNIT: measured_truth(
                    {1024: 1.0, 1280: 0.5, 1536: 0.25}, controls
                )
            },
            densified={UNIT: built},
            scalar_backbones={UNIT: controls},
            holdout_grades={UNIT: ()},
            byte_budget=exact_bytes[1280],
            max_regret_pct=0.0,
            bracket_agreement=True,
        )


def test_allocation_truth_cannot_be_unbound_float_claims():
    rd, built, _gate, grade, controls = allocator_ready_bundle()
    exact_bytes = {
        candidate.body_rate_q256: candidate.footprint["total_bytes"]
        for candidate in built
    }
    with pytest.raises(TrellisFormatError, match="identity-bound measured"):
        allocation_regret(
            {UNIT: rd},
            {UNIT: {1024: 1.0, 1280: 0.5, 1536: 0.25}},  # type: ignore[arg-type]
            densified={UNIT: built},
            scalar_backbones={UNIT: controls},
            holdout_grades={UNIT: (grade,)},
            byte_budget=exact_bytes[1280],
            max_regret_pct=0.0,
            bracket_agreement=True,
        )


def test_allocation_truth_must_measure_the_exact_densified_wire_recipe():
    rd, built, _gate, grade, controls = allocator_ready_bundle()
    truth = measured_truth(
        {1024: 1.0, 1280: 0.5, 1536: 0.25}, controls
    )
    schedule = uniform_column_schedule(512, 1280, family=E4M3_FAMILY)
    shifted = alphabet(5)[1:] + (alphabet(5)[-1] + 1,)
    different_recipe = build_trellis_allocator_candidate(
        UNIT,
        SHAPE,
        family=E4M3_FAMILY,
        body_rate_q256=1280,
        layout=LAYOUT_TIGHT_OFFSETS,
        schedule=schedule,
        alphabets={5: shifted},
        predicted_dloss=0.5,
        predicted_dloss_stderr=0.0,
        variant_label=PROVENANCE_MEASURED,
    )
    truth[1280] = TrellisMeasuredRateAnchor(
        candidate=different_recipe,
        curve_identity=CURVE,
        currency=CURRENCY,
        scalar_backbone=controls[1280],
    )
    exact_bytes = {
        candidate.body_rate_q256: candidate.footprint["total_bytes"]
        for candidate in built
    }
    with pytest.raises(TrellisFormatError, match="different exact wire recipe"):
        allocation_regret(
            {UNIT: rd},
            {UNIT: truth},
            densified={UNIT: built},
            scalar_backbones={UNIT: controls},
            holdout_grades={UNIT: (grade,)},
            byte_budget=exact_bytes[1280],
            max_regret_pct=0.0,
            bracket_agreement=True,
        )


def test_failed_regret_gate_cannot_be_consumed_by_solver_menu():
    _surfaces, densified, gate = misranked_allocator_bundle()
    assert not gate.passed
    with pytest.raises(TrellisFormatError, match="exceeds"):
        rate_surface_solver_menu(
            densified,
            currencies={unit: CURRENCY for unit in densified},
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            regret_gate=gate,
            byte_budget=gate.byte_budget,
        )


def test_allocation_regret_refuses_cross_unit_curve_identity_drift():
    with pytest.raises(TrellisFormatError, match="share one curve identity"):
        misranked_allocator_bundle(cross_unit_identity_drift=True)


def test_solver_bridge_rechecks_cross_unit_curve_identity():
    surfaces, densified, failed_gate = misranked_allocator_bundle()
    passed_gate = resealed_regret_gate(
        failed_gate,
        max_regret_pct=failed_gate.regret_pct,
        passed=True,
    )
    drifted_surface = replace(
        surfaces["u1"],
        curve_identity=curve_identity(corpus_id="different-corpus"),
    )
    drifted_densified = {
        **densified,
        "u1": replace(densified["u1"], surface=drifted_surface),
    }
    with pytest.raises(TrellisFormatError, match="share one curve identity"):
        rate_surface_solver_menu(
            drifted_densified,
            currencies={unit: CURRENCY for unit in drifted_densified},
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            regret_gate=passed_gate,
            byte_budget=passed_gate.byte_budget,
        )


def test_allocation_regret_refuses_target_profile_denied_exact_menu():
    rd, built, _gate, grade, controls = allocator_ready_bundle()
    denied_candidates = tuple(
        replace(
            candidate,
            servability=replace(
                candidate.servability,
                capability_legal=False,
                capability_detail="adversarial denied candidate",
            ),
        )
        for candidate in built
    )
    denied = TrellisDensifiedSurface(
        surface=rd,
        intended_use=SURFACE_USE_ALLOCATOR_COST,
        candidates=denied_candidates,
        scalar_backbone_point_sha256=(
            built.scalar_backbone_point_sha256
        ),
    )
    truth = measured_truth(
        {1024: 1.0, 1280: 0.5, 1536: 0.25}, controls
    )
    denied_by_rate = {
        candidate.body_rate_q256: candidate for candidate in denied
    }
    truth = {
        rate: replace(
            anchor_record,
            candidate=replace(
                anchor_record.candidate,
                servability=denied_by_rate[rate].servability,
            ),
        )
        for rate, anchor_record in truth.items()
    }
    exact_bytes = {
        candidate.body_rate_q256: candidate.footprint["total_bytes"]
        for candidate in denied
    }
    with pytest.raises(TrellisFormatError, match="target-profile-denied"):
        allocation_regret(
            {UNIT: rd},
            {UNIT: truth},
            densified={UNIT: denied},
            scalar_backbones={UNIT: controls},
            holdout_grades={UNIT: (grade,)},
            byte_budget=exact_bytes[1280],
            max_regret_pct=0.0,
            bracket_agreement=True,
        )


def test_solver_bridge_cannot_request_post_gate_denied_filtering():
    _rd, built, gate, _grade, _controls = allocator_ready_bundle()
    with pytest.raises(TrellisFormatError, match="cannot silently filter"):
        rate_surface_solver_menu(
            {UNIT: built},
            currencies={UNIT: CURRENCY},
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            regret_gate=gate,
            byte_budget=gate.byte_budget,
            fail_on_denied=False,
        )


def test_regret_gate_rejects_malformed_assignments_and_duplicate_grades():
    _rd, _built, gate, _grade, _controls = allocator_ready_bundle()
    with pytest.raises(TrellisFormatError, match="positive q256 integers"):
        replace(gate, assignment_interpolated=((UNIT, True),))
    grade_digest = gate.holdout_grade_sha256[0][1][0]
    with pytest.raises(TrellisFormatError, match="repeats a holdout"):
        replace(
            gate,
            holdout_grade_sha256=((UNIT, (grade_digest, grade_digest)),),
        )

    _surfaces, _densified, two_unit_gate = misranked_allocator_bundle()
    shared_digest = two_unit_gate.holdout_grade_sha256[0][1][0]
    duplicate_across_units = {
        unit: [shared_digest]
        for unit, _digests in two_unit_gate.holdout_grade_sha256
    }
    with pytest.raises(TrellisFormatError, match="across units"):
        resealed_regret_gate(
            two_unit_gate,
            holdout_grade_sha256=duplicate_across_units,
        )


def test_self_resealed_gate_cannot_change_derived_regret_or_agreement():
    _rd, _built, gate, _grade, _controls = allocator_ready_bundle()
    with pytest.raises(TrellisFormatError, match="agreement disagrees"):
        resealed_regret_gate(gate, assignment_agreement=0.5)
    with pytest.raises(TrellisFormatError, match="percentage disagrees"):
        resealed_regret_gate(
            gate,
            max_regret_pct=1.0,
            regret_pct=0.25,
            passed=True,
        )


def test_solver_bridge_recomputes_assignment_budget_and_mixing_fraction():
    _rd, built, gate, _grade, _controls = allocator_ready_bundle()
    high_rate = max(candidate.body_rate_q256 for candidate in built)
    overspent = resealed_regret_gate(
        gate,
        assignment_interpolated={UNIT: high_rate},
        assignment_truth={UNIT: high_rate},
        assignment_agreement=1.0,
        interpolated_assignment_fraction=0.0,
    )
    with pytest.raises(TrellisFormatError, match="exceeds its exact byte budget"):
        rate_surface_solver_menu(
            {UNIT: built},
            currencies={UNIT: CURRENCY},
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            regret_gate=overspent,
            byte_budget=overspent.byte_budget,
        )

    wrong_mixing = resealed_regret_gate(
        gate,
        interpolated_assignment_fraction=0.0,
    )
    with pytest.raises(TrellisFormatError, match="fraction disagrees"):
        rate_surface_solver_menu(
            {UNIT: built},
            currencies={UNIT: CURRENCY},
            intended_use=SURFACE_USE_ALLOCATOR_COST,
            regret_gate=wrong_mixing,
            byte_budget=wrong_mixing.byte_budget,
        )


def test_equal_byte_plateau_chooses_the_lowest_loss_exact_q256_rung():
    rd = gain_surface(
        ((1025, 1.0, 2.0), (1033, 0.5, 1.0))
    )
    controls = {
        1025: backbone(1025, 2.0),
        1026: backbone(1026, 1.9),
    }
    seal = seal_rate_surface_holdout(
        rd,
        1026,
        pre_render_recipe_identity_sha256=recipe_sha256(1026),
        scalar_backbone=controls[1026],
    )
    grade = grade_rate_surface_holdout(
        rd, seal, measured(1026, 0.95, backbone_dloss=1.9)
    )
    built = densify_rate_surface(
        rd,
        SHAPE,
        q256_values=(1025, 1026),
        alphabets={rate: alphabet(rate) for rate in range(1, 8)},
        intended_use=SURFACE_USE_ALLOCATOR_COST,
        schedule_policy=CURVE.schedule_policy,
        scalar_backbone=controls,
    )
    exact_bytes = {
        candidate.body_rate_q256: candidate.footprint["total_bytes"]
        for candidate in built
    }
    assert exact_bytes[1025] == exact_bytes[1026]
    gate = allocation_regret(
        {UNIT: rd},
        {
            UNIT: measured_truth(
                {1025: 1.0, 1026: 0.95}, controls
            )
        },
        densified={UNIT: built},
        scalar_backbones={UNIT: controls},
        holdout_grades={UNIT: (grade,)},
        byte_budget=exact_bytes[1025],
        max_regret_pct=0.0,
        bracket_agreement=True,
    )
    assert dict(gate.assignment_interpolated)[UNIT] == 1026
    assert gate.interpolated_assignment_fraction == 1.0
