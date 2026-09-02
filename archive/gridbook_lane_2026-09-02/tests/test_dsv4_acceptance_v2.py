from __future__ import annotations

import random

import pytest

from tools.dsv4_afast_burn import (
    ANCHORS,
    AUDIT_MEDIAN_TOLERANCE,
    AUDIT_P95_TOLERANCE,
    BACKSTOP_TOLERANCE,
    EXPERT_COUNT,
    LAYER_COUNT,
    MISSING_RUNGS,
    RUNGS,
    _acceptance,
    _audit_rung,
    _audit_stats,
)
from tools.dsv4_afast_campaign import law_predict

# The v2 acceptance path fits a per-expert level-2 affine log correction
# (a, b) on ANCHORS and predicts the measured audit rung from it.  Building
# the synthetic truth *from that law*, and expressing every perturbation in
# units of the tolerance it is meant to straddle, keeps these tests keyed to
# the current rung domain: a legitimate domain or anchor move is tracked
# automatically instead of re-pinned by hand.

PROJECTION = "gate_proj"


def _level2(expert: int) -> tuple[float, float]:
    """A per-expert (a, b): distinct scale offsets and level-2 tilts."""
    return 0.10 * (expert % 7) - 0.2, 0.004 * ((expert % 5) - 2)


def _law_exact(
    audit_rung: int, projection: str = PROJECTION,
) -> tuple[dict[int, list[float]], list[float]]:
    """Anchor + audit truth that the two-probe law reproduces exactly."""
    anchor_errors: dict[int, list[float]] = {k: [] for k in ANCHORS}
    audit_errors: list[float] = []
    for expert in range(EXPERT_COUNT):
        a, b = _level2(expert)
        for rung in ANCHORS:
            anchor_errors[rung].append(law_predict(projection, rung, a, b))
        audit_errors.append(law_predict(projection, audit_rung, a, b))
    return anchor_errors, audit_errors


def _missed_by(truth: float, relative_error: float) -> float:
    """A measured value the (unchanged) prediction misses by exactly `r`.

    Both gates score ``|prediction - truth| / |truth|``, so scaling a
    law-exact truth by ``1 / (1 - r)`` places the residual at ``r``.  That
    lets each perturbation below be written as a multiple of the tolerance
    under test rather than as a magic constant.
    """
    return truth / (1.0 - relative_error)


def test_v2_audit_draw_is_layer_deterministic() -> None:
    draws = [_audit_rung(layer) for layer in range(LAYER_COUNT)]
    state = random.getstate()
    try:
        for layer, drawn in enumerate(draws):
            # Deterministic per layer, and independent of the global RNG:
            # the burn calls _audit_rung from several stages of a long
            # process, so a draw that moved with ambient random state would
            # not reproduce the rung its own shard was measured at.
            random.seed(977 * layer + 1)
            random.random()
            assert _audit_rung(layer) == drawn
            # Drawn from the priced domain, and never from an anchor: the
            # audit is only evidence if it is a rung the fit did not see.
            assert drawn in MISSING_RUNGS
            assert drawn in RUNGS
            assert drawn not in ANCHORS
            # The registered recipe.  Seed 42 + layer is a contract input,
            # not an expected output: it is stamped into every layer
            # identity as `audit_seed`, and each banked layer shard records
            # the rung it produced under the domain in force when it was
            # measured.  Deriving the pool from MISSING_RUNGS keeps this
            # assertion alive across a domain move while still failing if
            # the draw itself is re-seeded out from under the bank.
            assert drawn == random.Random(42 + layer).choice(MISSING_RUNGS)
    finally:
        random.setstate(state)

    assert MISSING_RUNGS == tuple(k for k in RUNGS if k not in ANCHORS)
    # A constant draw would audit one rung for the whole stack and leave the
    # rest of the interpolated menu uncertified.
    assert len(set(draws)) > 1


def test_v2_backstop_only_rejects_the_gross_audit_rung_outlier() -> None:
    audit_rung = _audit_rung(0)
    anchor_errors, audit_errors = _law_exact(audit_rung)

    accepted, rejected, fit = _acceptance(
        anchor_errors, audit_rung, audit_errors, PROJECTION,
    )

    # Law-exact truth: the two-probe fit reproduces the measured audit rung,
    # so every expert ships interpolated rows.
    assert rejected == []
    assert accepted == list(range(EXPERT_COUNT))
    assert fit["backstop_failed"] == 0
    assert fit["acceptance_rate"] == 1.0
    assert fit["backstop_tolerance"] == BACKSTOP_TOLERANCE
    assert max(
        row["audit_rung_relative_error"] for row in fit["per_slice"]
    ) < 1e-9

    gross, marginal = 0, 1
    perturbed = list(audit_errors)
    perturbed[gross] = _missed_by(
        audit_errors[gross], 2.0 * BACKSTOP_TOLERANCE,
    )
    perturbed[marginal] = _missed_by(
        audit_errors[marginal], 0.5 * BACKSTOP_TOLERANCE,
    )

    accepted, rejected, fit = _acceptance(
        anchor_errors, audit_rung, perturbed, PROJECTION,
    )

    # Only the gross outlier is demoted to measured rows; a slice that
    # misses by half the bar is ordinary law residue and still ships
    # interpolated.
    assert rejected == [gross]
    assert marginal in accepted
    assert len(accepted) == EXPERT_COUNT - 1
    assert fit["backstop_failed"] == 1
    assert fit["backstop_passed"] == EXPERT_COUNT - 1
    assert fit["accepted"] == len(accepted)
    assert fit["rejected"] == len(rejected)
    assert sorted(accepted + rejected) == list(range(EXPERT_COUNT))

    per_slice = {row["expert"]: row for row in fit["per_slice"]}
    assert len(per_slice) == EXPERT_COUNT
    assert all(row["audit_rung"] == audit_rung for row in per_slice.values())
    assert per_slice[gross]["backstop_pass"] is False
    assert per_slice[gross]["audit_rung_relative_error"] > BACKSTOP_TOLERANCE
    assert per_slice[marginal]["backstop_pass"] is True
    assert (
        per_slice[marginal]["audit_rung_relative_error"] <= BACKSTOP_TOLERANCE
    )


def test_v2_audit_gate_scores_the_two_probe_law_prediction() -> None:
    audit_rung = _audit_rung(0)
    anchor_errors, audit_errors = _law_exact(audit_rung)

    clean = _audit_stats(
        anchor_errors, audit_errors, audit_rung, None, PROJECTION,
    )
    assert clean["pass"] is True
    assert clean["rung"] == audit_rung
    assert clean["n"] == EXPERT_COUNT
    assert clean["n_excluded"] == 0
    assert clean["median"] == pytest.approx(0.0, abs=1e-9)
    assert clean["p95"] == pytest.approx(0.0, abs=1e-9)
    assert clean["thresholds"] == {
        "median": AUDIT_MEDIAN_TOLERANCE, "p95": AUDIT_P95_TOLERANCE,
    }

    # Median arm: a miss every slice shares, above the median bar but under
    # the p95 bar, must fail the gate on the median alone.
    broad = (AUDIT_MEDIAN_TOLERANCE + AUDIT_P95_TOLERANCE) / 2.0
    stats = _audit_stats(
        anchor_errors, [_missed_by(t, broad) for t in audit_errors],
        audit_rung, None, PROJECTION,
    )
    assert stats["median"] > AUDIT_MEDIAN_TOLERANCE
    assert stats["p95"] <= AUDIT_P95_TOLERANCE
    assert stats["pass"] is False

    # p95 arm: a tail that leaves the median clean must still fail closed.
    outliers = list(range(EXPERT_COUNT - EXPERT_COUNT // 10, EXPERT_COUNT))
    tailed = list(audit_errors)
    for expert in outliers:
        tailed[expert] = _missed_by(
            audit_errors[expert], 2.0 * AUDIT_P95_TOLERANCE,
        )
    stats = _audit_stats(
        anchor_errors, tailed, audit_rung, None, PROJECTION,
    )
    assert stats["median"] <= AUDIT_MEDIAN_TOLERANCE
    assert stats["p95"] > AUDIT_P95_TOLERANCE
    assert stats["pass"] is False

    # v2 semantics: slices the backstop rejected ship measured rows, so the
    # gate scores only the accepted set -- their residue is not a menu
    # defect, and excluding them restores the pass.
    kept = [e for e in range(EXPERT_COUNT) if e not in outliers]
    scoped = _audit_stats(
        anchor_errors, tailed, audit_rung, kept, PROJECTION,
    )
    assert scoped["pass"] is True
    assert scoped["n"] == len(kept)
    assert scoped["n_excluded"] == len(outliers)
    assert len(scoped["per_slice_relative_error"]) == len(kept)

    # ... but when the backstop rejected everything there is no interpolated
    # row left to certify, and the gate must not pass vacuously.
    empty = _audit_stats(anchor_errors, tailed, audit_rung, [], PROJECTION)
    assert empty["pass"] is False
    assert empty["n"] == 0
    assert empty["n_excluded"] == EXPERT_COUNT
    assert empty["median"] is None
