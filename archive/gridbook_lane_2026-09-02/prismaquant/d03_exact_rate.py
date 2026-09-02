"""D0.3 exact-rate experiment harness — ultraplan P5d.

Runs the two experiments gridbook ``ROADMAP.md`` **D0.3** names, with the
pricing fixes P5a landed and the constraint axis P5c added:

  (i)  at 4.5 bpp, ``FP8_CB_K36`` vs vanilla ``NVFP4`` on DENSE units, at
       matched **exact whole-artifact bytes**; and
  (ii) below 4.5 bpp, byte-neutral sweeps in which vanilla-NVFP4 promotions on
       chosen layers are **funded by cheaper CB rungs elsewhere**.

``docs/lanes/nvfp4-cb/format-speed-policy.md`` §4 states the discipline both
inherit: "At 4.5 bpp, compare native NVFP4 against FP8-CB K36 only after exact
whole-artifact accounting. Below native NVFP4's 4.5-bpp floor, 'same average
rate' is an assignment-level comparison: every NVFP4 promotion must be funded
by lower CB rungs elsewhere. [...] never compare an isolated promoted layer
against an unfunded baseline."

What this harness is
--------------------

It **prepares release-gate evidence. It does not constitute it.** Every arm it
emits is labelled proposal data. D0.3 is "an empirical release gate", and the
gate is the served NATIVE-PARITY protocol: same-session end-to-end streaming
TTFT/ITL/TPS percentiles plus served KL/PPL/tasks on the real artifacts. What
this harness supplies is the part that can be computed offline and exactly —
the assignments, their exact whole-artifact bytes, their predicted Δloss under
activation-fair pricing, and the provenance that says which of those numbers
are comparable at all.

Two refusals are load-bearing:

* **The cross-family verdict.** P5a's cross-family CB-ladder symmetry check
  exists precisely so that "NVFP4-CB vs FP8-CB" is not read off two ladders
  fitted independently and never cross-calibrated. When that band check FAILED
  (``cross_family_comparison_publishable: false``), this harness prints no
  cross-family verdict at all — it prints the refusal and the numbers behind
  it. Suppressing the verdict is the whole purpose of the check; printing it
  with a caveat would defeat it.
* **The matched-bytes gate.** Experiment (i) is only a contest if the two arms
  really are byte-matched. The formal target policy §5 already states is
  ``<= 0.1%`` of whole-artifact bytes (it is the threshold the published 0.6B
  endpoint pair MISSED, at +0.154%). An arm pair outside it is reported with
  ``matched_bytes_gate: false`` and no quality verdict.

Scope exclusion (recorded in every report)
------------------------------------------

**Packed-expert vanilla NVFP4 is excluded from the contest.** The Gridbook
producer profile denies stock NVFP4/FP8 on packed expert stacks (policy §4:
"Packed expert stacks currently deny stock NVFP4/FP8 in the Gridbook producer
profile. The native-versus-CB mixed frontier is therefore feasible only for
dense and shared units"), because no stock-compressed-tensors packed-expert
emit path exists in the container. Building one is out of scope under the
one-payload / no-new-packer rule; unlocking it is gridbook **D0.2** (packed-
expert native delegation), which remains open. The audit's own caveat applies
until it lands: the 295B "offered and never chosen" zero for vanilla NVFP4 is
weak evidence because the expert mass was never a fair contest.

The module is stdlib + the CPU allocator surface; it never imports torch.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import format_registry as fr
from .allocator_candidates import (
    build_candidates,
    calibrate_activation_fair_pricing,
    selection_serving_lane_provenance,
    serialized_candidate_payload,
)
from .allocator_solver import Candidate, _shape_from_stats
from .cb_ladder_cross_family import cross_family_verdict_from_cost_payload
from .cb_layout import parse_format_name
from .footprint import nvfp4_global_sidecar_bytes
from .nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    is_cb_format,
)
from .serve_constraints import (
    RELATIVE_TAX_REFERENCE_RULE,
    ServeConstraintContext,
    ServeConstraintError,
    ServeSLOs,
    WorkloadMix,
    evaluate_assignment as evaluate_serve_constraints,
    fastest_feasible_summary,
)
from .serve_dispatch_table import DispatchTableError, load_dispatch_table
from .serving_profiles import serving_lane_route

SCHEMA = "prismaquant.d03_exact_rate.v1"

#: Policy §5's formal whole-artifact byte-match target. Named there as the
#: threshold the published 0.6B endpoint pair MISSED (+0.154%), so it is a
#: stated target of the record, not a constant invented here.
MATCHED_BYTES_TOLERANCE_FRACTION = 0.001

#: The exclusion recorded in every report. See the module docstring.
PACKED_EXPERT_EXCLUSION = {
    "excluded": "packed_expert_vanilla_nvfp4",
    "reason": (
        "The Gridbook producer serving profile denies stock NVFP4/FP8 on "
        "packed expert stacks: no stock-compressed-tensors packed-expert emit "
        "path exists in the container, and building one is out of scope under "
        "the one-payload / no-new-packer rule. Packed experts are therefore "
        "NOT contestants in either experiment; only dense and shared units "
        "are."
    ),
    "unlocked_by": (
        "gridbook ROADMAP D0.2 — 'Complete packed-expert native delegation if "
        "the assignment needs it.' Its fail-closed clause (delegated_preflight)"
        " is shipped and generalized; the packed-expert loader work itself "
        "remains open."
    ),
    "consequence_for_evidence": (
        "Until D0.2 lands, an 'offered and never chosen' zero for vanilla "
        "NVFP4 on a MoE model is weak evidence — the expert mass was never a "
        "fair contest (gridbook docs/audits/ultraplan_perf_2026-08-01.md §6)."
    ),
}

_EVIDENCE_LABEL = (
    "PROPOSAL DATA. Offline exact-byte accounting and predicted Δloss only. "
    "gridbook ROADMAP D0.3 is an EMPIRICAL RELEASE GATE and this harness does "
    "not satisfy it: promotion requires the served NATIVE-PARITY protocol "
    "(same-session streaming TTFT/ITL/TPS percentiles plus served KL/PPL/"
    "tasks over the representative workload matrix), with format/rung, "
    "layout, activation quantization, concrete backend, GPU/runtime identity, "
    "TP and fallback state recorded per docs/lanes/nvfp4-cb/"
    "format-speed-policy.md §3."
)


class D03Error(RuntimeError):
    """The harness cannot run the requested contest on these inputs."""


# --------------------------------------------------------------------------- #
# Exact whole-artifact byte accounting
# --------------------------------------------------------------------------- #
def exact_payload_bytes(
    assignment: Mapping[str, str],
    stats: Mapping[str, Mapping[str, Any]],
    *,
    cb_serialization_context: CBSerializationContext | None,
) -> dict:
    """Exact serialized tensor payload for an assignment, in bytes.

    The SAME non-additive accounting the allocator's exact-payload filter uses
    (``allocator._assignment_payload_totals``): shared CB codebook sidecars
    are charged ONCE per physical identity rather than per tensor, and NVFP4
    global scale tensors are charged per emitted Linear. Candidate
    ``memory_bytes`` is the additive DP proposal cost and is deliberately not
    used here — it double-charges shared sidecars, which is exactly the error
    a matched-bytes contest cannot afford.
    """
    total = 0
    params = 0
    cb_assignment: dict[str, str] = {}
    cb_shapes: dict[str, tuple[int, ...]] = {}
    missing: list[str] = []
    for name in sorted(assignment):
        fmt = str(assignment[name])
        entry = stats.get(name)
        if not isinstance(entry, Mapping):
            missing.append(name)
            continue
        params += int(entry.get("n_params", 0) or 0)
        shape = _shape_from_stats(entry)
        if is_cb_format(fmt):
            cb_assignment[name] = fmt
            cb_shapes[name] = shape
            continue
        payload_bytes, _identity, _sidecar = serialized_candidate_payload(
            fr.get_format(fmt),
            shape,
            qname=name,
            cb_serialization_context=cb_serialization_context,
        )
        total += int(payload_bytes)
        if fmt == "NVFP4":
            total += int(nvfp4_global_sidecar_bytes(name, shape))
    if missing:
        raise D03Error(
            f"exact byte accounting has no stats for {len(missing)} "
            f"tensor(s): {sorted(missing)[:8]}"
        )
    cb_tensor_bytes = 0
    cb_sidecar_bytes = 0
    if cb_assignment:
        if cb_serialization_context is None:
            raise D03Error(
                "a CB assignment reached exact byte accounting without a "
                "CBSerializationContext; exact CB bytes are undefined without "
                "the producer's scale coding / codebook-source contract"
            )
        breakdown = cb_assignment_payload_breakdown(
            cb_assignment, cb_shapes, context=cb_serialization_context)
        cb_tensor_bytes = int(breakdown["tensor_payload_bytes"])
        cb_sidecar_bytes = int(breakdown["codebook_sidecar_bytes"])
        total += cb_tensor_bytes + cb_sidecar_bytes
    return {
        "exact_bytes": int(total),
        "quantizable_params": int(params),
        "bits_per_param": (8.0 * total / params) if params else 0.0,
        "cb_tensor_bytes": cb_tensor_bytes,
        "cb_shared_sidecar_bytes": cb_sidecar_bytes,
        "accounting": (
            "exact_serialized_tensor_payload__shared_cb_sidecars_charged_once"
        ),
    }


def assignment_predicted_dloss(
    assignment: Mapping[str, str],
    candidates: Mapping[str, Sequence[Candidate]],
) -> float:
    """Sum predicted Δloss, refusing to price an assignment it cannot."""
    total = 0.0
    for name in sorted(assignment):
        fmt = assignment[name]
        chosen = next(
            (c for c in candidates.get(name, ()) if c.fmt == fmt), None)
        if chosen is None:
            raise D03Error(
                f"{name}: no candidate prices format {fmt!r} (available: "
                f"{sorted(c.fmt for c in candidates.get(name, ()))}). Scoring "
                "it at zero Δloss would make an illegal arm look free."
            )
        total += float(chosen.predicted_dloss)
    return total


# --------------------------------------------------------------------------- #
# Arm construction
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    """One contested assignment plus everything needed to judge it."""

    label: str
    assignment: dict[str, str]
    exact: dict = field(default_factory=dict)
    predicted_dloss: float = float("nan")
    serve: Any = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self, *, include_assignment: bool) -> dict:
        counts: dict[str, int] = {}
        for fmt in self.assignment.values():
            counts[fmt] = counts.get(fmt, 0) + 1
        out = {
            "label": self.label,
            "n_units": len(self.assignment),
            "format_counts": dict(sorted(counts.items())),
            "exact_bytes": self.exact.get("exact_bytes"),
            "exact_bits_per_param": self.exact.get("bits_per_param"),
            "cb_shared_sidecar_bytes": self.exact.get("cb_shared_sidecar_bytes"),
            "predicted_dloss": self.predicted_dloss,
            "predicted_dloss_pricing": (
                "activation_fair_pricing_applied__see_activation_fair_pricing"
                "_block"
            ),
            "notes": list(self.notes),
        }
        if self.serve is not None:
            out["serve_constraints"] = self.serve.as_dict()
        if include_assignment:
            out["assignment"] = dict(sorted(self.assignment.items()))
        return out


def _legal(name: str, fmt: str,
           candidates: Mapping[str, Sequence[Candidate]]) -> bool:
    return any(c.fmt == fmt for c in candidates.get(name, ()))


def contest_scope(
    candidates: Mapping[str, Sequence[Candidate]],
    *,
    contenders: Sequence[str],
    exclude_markers: Sequence[str],
) -> tuple[list[str], dict]:
    """Units on which every contender is legal, with the exclusions recorded.

    A unit that cannot legally take BOTH arms' formats is not a contest — it
    would silently become an arm-specific format difference on a row nobody
    compared. Those rows are excluded and counted, never quietly defaulted.
    """
    eligible: list[str] = []
    excluded_marker: list[str] = []
    excluded_illegal: dict[str, list[str]] = {}
    for name in sorted(candidates):
        if any(marker in name for marker in exclude_markers):
            excluded_marker.append(name)
            continue
        missing = [f for f in contenders if not _legal(name, f, candidates)]
        if missing:
            excluded_illegal.setdefault(",".join(sorted(missing)), []).append(
                name)
            continue
        eligible.append(name)
    report = {
        "contenders": list(contenders),
        "n_eligible_units": len(eligible),
        "n_excluded_by_marker": len(excluded_marker),
        "excluded_by_marker_sample": excluded_marker[:8],
        "excluded_marker_patterns": list(exclude_markers),
        "excluded_illegal_counts": {
            key: len(names) for key, names in sorted(excluded_illegal.items())
        },
        "excluded_illegal_sample": {
            key: names[:6] for key, names in sorted(excluded_illegal.items())
        },
        "packed_expert_exclusion": dict(PACKED_EXPERT_EXCLUSION),
    }
    return eligible, report


# --------------------------------------------------------------------------- #
# Experiment (i): matched exact whole-artifact bytes on dense units
# --------------------------------------------------------------------------- #
def run_matched_bytes_experiment(
    candidates: Mapping[str, Sequence[Candidate]],
    stats: Mapping[str, Mapping[str, Any]],
    *,
    cb_format: str,
    native_format: str,
    baseline_format: str | None,
    exclude_markers: Sequence[str],
    cb_serialization_context: CBSerializationContext | None,
    serve_context: ServeConstraintContext,
    serve_lane_for,
) -> dict:
    """``FP8_CB_K36`` vs vanilla ``NVFP4`` at matched exact bytes (D0.3 i)."""
    contenders = [cb_format, native_format]
    eligible, scope = contest_scope(
        candidates, contenders=contenders, exclude_markers=exclude_markers)
    if not eligible:
        raise D03Error(
            f"no unit admits BOTH {cb_format} and {native_format}; there is "
            "no dense contest to run on these inputs. Widen --formats, or "
            "check the serving profile's shape rules "
            "(format_applicability.json names every mask)."
        )
    # Units outside the contest are pinned to ONE format across both arms, so
    # the byte and Δloss difference between the arms is attributable to the
    # contested rows alone.
    pinned: dict[str, str] = {}
    for name in sorted(candidates):
        if name in eligible:
            continue
        if baseline_format and _legal(name, baseline_format, candidates):
            pinned[name] = baseline_format

    arms = []
    for label, fmt in (("cb", cb_format), ("native", native_format)):
        assignment = {name: fmt for name in eligible}
        assignment.update(pinned)
        arm = Arm(label=f"{label}:{fmt}", assignment=assignment)
        arm.exact = exact_payload_bytes(
            assignment, stats,
            cb_serialization_context=cb_serialization_context)
        arm.predicted_dloss = assignment_predicted_dloss(
            assignment, candidates)
        arm.serve = evaluate_serve_constraints(
            assignment, stats, serve_context,
            lane_for=serve_lane_for,
            resident_bytes=int(arm.exact["exact_bytes"]),
        )
        arms.append(arm)

    cb_arm, native_arm = arms
    delta = int(cb_arm.exact["exact_bytes"]) - int(
        native_arm.exact["exact_bytes"])
    denom = max(
        int(cb_arm.exact["exact_bytes"]),
        int(native_arm.exact["exact_bytes"]),
        1,
    )
    fraction = abs(delta) / denom
    matched = fraction <= MATCHED_BYTES_TOLERANCE_FRACTION
    return {
        "experiment": "d03_i_matched_exact_whole_artifact_bytes_dense",
        "roadmap": (
            "gridbook ROADMAP D0.3: 'At 4.5 bpp compare native NVFP4 with "
            "FP8-CB K36 using exact whole-artifact bytes.'"
        ),
        "scope": scope,
        "n_pinned_units_outside_contest": len(pinned),
        "pinned_format_outside_contest": baseline_format,
        "arms": [a.as_dict(include_assignment=True) for a in arms],
        "bytes_delta_bytes": delta,
        "bytes_delta_fraction": fraction,
        "matched_bytes_tolerance_fraction": MATCHED_BYTES_TOLERANCE_FRACTION,
        "matched_bytes_tolerance_source": (
            "docs/lanes/nvfp4-cb/format-speed-policy.md §5 names <=0.1% as "
            "the formal whole-artifact byte-match target (the published 0.6B "
            "endpoint pair MISSED it at +0.154%)."
        ),
        "matched_bytes_gate": bool(matched),
        "dloss_delta_cb_minus_native": (
            cb_arm.predicted_dloss - native_arm.predicted_dloss),
        "evidence_status": _EVIDENCE_LABEL,
    }


# --------------------------------------------------------------------------- #
# Experiment (ii): byte-neutral NVFP4 promotions funded by cheaper CB rungs
# --------------------------------------------------------------------------- #
def _cb_rung(fmt: str) -> int | None:
    parsed = parse_format_name(fmt)
    return None if parsed is None else int(parsed[1])


def _cb_ladder_for(name: str, candidates: Mapping[str, Sequence[Candidate]],
                   grid: str) -> list[Candidate]:
    """This unit's CB rungs of one grid, cheapest (fewest bytes) first."""
    rungs = []
    for cand in candidates.get(name, ()):
        parsed = parse_format_name(cand.fmt)
        if parsed is not None and parsed[0].grid == grid:
            rungs.append(cand)
    return sorted(rungs, key=lambda c: (c.memory_bytes, c.fmt))


def run_byte_neutral_sweep(
    candidates: Mapping[str, Sequence[Candidate]],
    stats: Mapping[str, Mapping[str, Any]],
    *,
    native_format: str,
    baseline_cb_format: str,
    cb_grid: str,
    promote_counts: Sequence[int],
    exclude_markers: Sequence[str],
    cb_serialization_context: CBSerializationContext | None,
    serve_context: ServeConstraintContext,
    serve_lane_for,
) -> dict:
    """Below 4.5 bpw: promotions to vanilla NVFP4, FUNDED, at equal bytes.

    Deterministic and greedy, with the funding rule stated in the output
    rather than implied:

    1. Baseline: every eligible unit on ``baseline_cb_format``.
    2. Promote the ``k`` units with the largest predicted-Δloss REDUCTION per
       extra byte from moving to ``native_format`` (ties by name).
    3. Fund the promotion by demoting other units down their own CB ladder,
       taking the cheapest Δloss increase per byte freed first (ties by name,
       then by rung), until exact bytes are back at or below the baseline.
    4. RECLAIM the overshoot: a CB rung is a discrete step, so step 3 usually
       frees more than it needed and the point would sit BELOW the baseline
       rate — an unfair comparison in the other direction (a sparser
       assignment judged against a denser baseline). Upgrade donors back up
       their ladder, best Δloss reduction per byte spent first, while the
       exact bytes stay within budget. Without this the sweep answers "is a
       funded promotion better than the baseline at a LOWER rate", which is
       not the question.
    5. If the ladder cannot free enough bytes, the sweep point is reported
       ``byte_neutral: false`` and carries no quality claim — policy §4 forbids
       comparing an unfunded promotion against the baseline.

    Every point is re-priced with the EXACT accounting, not the additive
    per-candidate bytes, because the shared CB sidecar set changes as rungs
    leave and enter the assignment.
    """
    contenders = [native_format, baseline_cb_format]
    eligible, scope = contest_scope(
        candidates, contenders=contenders, exclude_markers=exclude_markers)
    if not eligible:
        raise D03Error(
            f"no unit admits BOTH {native_format} and {baseline_cb_format}; "
            "there is no byte-neutral sweep to run on these inputs."
        )

    baseline_assignment = {name: baseline_cb_format for name in eligible}
    baseline_exact = exact_payload_bytes(
        baseline_assignment, stats,
        cb_serialization_context=cb_serialization_context)
    baseline_dloss = assignment_predicted_dloss(
        baseline_assignment, candidates)
    baseline_arm = Arm(
        label=f"baseline:{baseline_cb_format}",
        assignment=dict(baseline_assignment),
        exact=baseline_exact,
        predicted_dloss=baseline_dloss,
    )
    baseline_arm.serve = evaluate_serve_constraints(
        baseline_assignment, stats, serve_context,
        lane_for=serve_lane_for,
        resident_bytes=int(baseline_exact["exact_bytes"]),
    )
    budget = int(baseline_exact["exact_bytes"])

    def _cand(name: str, fmt: str) -> Candidate:
        for c in candidates.get(name, ()):
            if c.fmt == fmt:
                return c
        raise D03Error(f"{name}: no candidate for {fmt!r}")

    # Promotion order: best predicted-Δloss reduction per extra byte.
    promo_rank: list[tuple[float, str]] = []
    for name in eligible:
        base = _cand(name, baseline_cb_format)
        promoted = _cand(name, native_format)
        extra = promoted.memory_bytes - base.memory_bytes
        gain = base.predicted_dloss - promoted.predicted_dloss
        # Only promotions that COST bytes need funding; a promotion that is
        # free or cheaper is ranked ahead of every paid one.
        density = gain / extra if extra > 0 else float("inf")
        promo_rank.append((density, name))
    promo_order = [
        name for _d, name in sorted(promo_rank, key=lambda t: (-t[0], t[1]))
    ]

    points = []
    for k in promote_counts:
        k = int(k)
        if k <= 0 or k > len(promo_order):
            continue
        promoted_names = sorted(promo_order[:k])
        assignment = dict(baseline_assignment)
        for name in promoted_names:
            assignment[name] = native_format
        exact = exact_payload_bytes(
            assignment, stats,
            cb_serialization_context=cb_serialization_context)
        funding_moves: list[dict] = []
        # Fund it: demote non-promoted units down their own CB ladder.
        # Deterministic greedy over (Δloss increase per byte freed, name,
        # rung), recomputing exact bytes after each accepted move because the
        # shared sidecar set is not additive.
        donors = [n for n in eligible if n not in set(promoted_names)]
        while exact["exact_bytes"] > budget:
            best_move = None
            for name in donors:
                current = _cand(name, assignment[name])
                for rung in _cb_ladder_for(name, candidates, cb_grid):
                    if rung.memory_bytes >= current.memory_bytes:
                        continue
                    freed = current.memory_bytes - rung.memory_bytes
                    cost = rung.predicted_dloss - current.predicted_dloss
                    key = (cost / freed, name, rung.fmt)
                    if best_move is None or key < best_move[0]:
                        best_move = (key, name, rung.fmt, freed, cost)
            if best_move is None:
                break
            _key, name, fmt, freed, cost = best_move
            funding_moves.append({
                "unit": name,
                "from": assignment[name],
                "to": fmt,
                "additive_bytes_freed": int(freed),
                "predicted_dloss_increase": float(cost),
            })
            assignment[name] = fmt
            exact = exact_payload_bytes(
                assignment, stats,
                cb_serialization_context=cb_serialization_context)

        # Reclaim the discrete-step overshoot so the point sits AT the budget
        # rather than under it (see the docstring, step 4).
        reclaim_moves: list[dict] = []
        while exact["exact_bytes"] <= budget:
            # Every upgrade available now, best Δloss reduction per byte
            # spent first; ties by name then rung so the walk is
            # reproducible. The FIRST one that still fits is taken — a large
            # upgrade that busts the budget must not hide a small one that
            # does not.
            options: list[tuple[tuple[float, str, str], str, str, int, float]]
            options = []
            for name in donors:
                current = _cand(name, assignment[name])
                for rung in _cb_ladder_for(name, candidates, cb_grid):
                    if rung.memory_bytes <= current.memory_bytes:
                        continue
                    spent = rung.memory_bytes - current.memory_bytes
                    gain = current.predicted_dloss - rung.predicted_dloss
                    options.append(
                        ((-gain / spent, name, rung.fmt), name, rung.fmt,
                         spent, gain))
            options.sort(key=lambda o: o[0])
            applied = False
            for _key, name, fmt, spent, gain in options:
                previous = assignment[name]
                assignment[name] = fmt
                trial = exact_payload_bytes(
                    assignment, stats,
                    cb_serialization_context=cb_serialization_context)
                if trial["exact_bytes"] > budget:
                    assignment[name] = previous
                    continue
                exact = trial
                reclaim_moves.append({
                    "unit": name,
                    "from": previous,
                    "to": fmt,
                    "additive_bytes_spent": int(spent),
                    "predicted_dloss_reduction": float(gain),
                })
                applied = True
                break
            if not applied:
                break

        byte_neutral = exact["exact_bytes"] <= budget
        arm = Arm(
            label=f"promote_{k}:{native_format}",
            assignment=assignment,
            exact=exact,
            predicted_dloss=assignment_predicted_dloss(
                assignment, candidates),
        )
        arm.serve = evaluate_serve_constraints(
            assignment, stats, serve_context,
            lane_for=serve_lane_for,
            resident_bytes=int(exact["exact_bytes"]),
        )
        if not byte_neutral:
            arm.notes.append(
                "NOT byte-neutral: the CB ladder could not free enough bytes "
                "to fund these promotions. Policy §4 forbids comparing an "
                "unfunded promotion against the baseline, so this point "
                "carries no quality claim."
            )
        points.append({
            "n_promoted": k,
            "promoted_units": promoted_names,
            "funding_moves": funding_moves,
            "n_funding_moves": len(funding_moves),
            "reclaim_moves": reclaim_moves,
            "n_reclaim_moves": len(reclaim_moves),
            "byte_neutral": bool(byte_neutral),
            "bytes_vs_baseline": int(exact["exact_bytes"]) - budget,
            "net_dloss_vs_baseline": (
                arm.predicted_dloss - baseline_dloss),
            "arm": arm.as_dict(include_assignment=True),
        })

    return {
        "experiment": "d03_ii_byte_neutral_nvfp4_promotions_funded_by_cb",
        "roadmap": (
            "gridbook ROADMAP D0.3: 'below 4.5 bpp evaluate byte-neutral "
            "assignments whose NVFP4 promotions are funded by lower CB rungs "
            "elsewhere.'"
        ),
        "funding_rule": (
            "greedy, deterministic: promote by max predicted-Δloss reduction "
            "per extra byte; fund by min predicted-Δloss increase per byte "
            "freed down each donor's own CB ladder; re-price EXACTLY after "
            "every accepted move because shared CB sidecars are not additive"
        ),
        "scope": scope,
        "baseline": baseline_arm.as_dict(include_assignment=False),
        "byte_budget_bytes": budget,
        "points": points,
        "evidence_status": _EVIDENCE_LABEL,
    }


# --------------------------------------------------------------------------- #
# Cross-family verdict — the refusal that matters
# --------------------------------------------------------------------------- #
def cross_family_section(cost_payload: Mapping) -> dict:
    """Publish the CB-ladder cross-family verdict, or refuse to.

    P5a's band check exists so a cross-family (NVFP4-CB vs FP8-CB) claim is
    not drawn from two ladders fitted independently and never cross-
    calibrated. When it failed, the ONLY honest output is the refusal plus
    the numbers behind it — printing the verdict "with a caveat" would defeat
    the check, which is its entire purpose.
    """
    verdict = cross_family_verdict_from_cost_payload(cost_payload)
    if verdict is None:
        return {
            "publishable": False,
            "verdict": None,
            "refusal_reason": "no_cross_family_band_check_in_the_cost_payload",
            "detail": (
                "This cost run predates the P5a cross-family symmetry check or "
                "never fitted a CB ladder, so nothing establishes that the two "
                "families' predicted Δloss are on a comparable scale. No "
                "cross-family verdict is printed."
            ),
        }
    publishable = bool(
        verdict.get("cross_family_comparison_publishable", False))
    if not publishable:
        return {
            "publishable": False,
            "verdict": None,
            "refusal_reason": "p5a_cross_family_band_check_failed",
            "detail": (
                "The per-family CB-ladder holdout residuals are NOT in "
                "family-symmetric bands for this run, so a cross-family "
                "comparison drawn from these ladders would be an artefact of "
                "the fits. No cross-family verdict is printed. The band "
                "check's numbers are below so the failure is actionable."
            ),
            "band_check": dict(verdict),
        }
    return {
        "publishable": True,
        "verdict": verdict.get("verdict"),
        "detail": verdict.get("detail", ""),
        "band_check": dict(verdict),
        "still_not_a_promotion": (
            "A publishable band check makes the two families' predicted Δloss "
            "COMPARABLE. It does not make predicted Δloss served quality; the "
            "served KL/PPL/task gate is unchanged."
        ),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(
    *,
    probe_path: str,
    costs_path: str,
    formats: Sequence[str],
    target_profile: str | None,
    cb_format: str,
    native_format: str,
    baseline_format: str | None,
    baseline_cb_format: str,
    cb_grid: str,
    promote_counts: Sequence[int],
    exclude_markers: Sequence[str],
    cb_serialization_context: CBSerializationContext | None,
    serve_context: ServeConstraintContext,
    skip_matched_bytes: bool = False,
    skip_byte_neutral: bool = False,
) -> dict:
    """Run both D0.3 experiments and return one report payload."""
    with open(probe_path, "rb") as fh:
        probe = pickle.load(fh)
    with open(costs_path, "rb") as fh:
        cost_payload = pickle.load(fh)
    stats = dict(probe.get("stats", {}))
    costs = dict(cost_payload.get("costs", {}))
    if not stats or not costs:
        raise D03Error(
            f"probe {probe_path} / costs {costs_path} carry no stats/costs")

    specs = [fr.get_format(name) for name in formats]
    activation_pricing = calibrate_activation_fair_pricing(
        stats, costs, specs)
    candidates = build_candidates(
        stats, costs, specs,
        target_profile=target_profile,
        cb_serialization_context=cb_serialization_context,
        activation_pricing=activation_pricing,
    )
    if not candidates:
        raise D03Error("no Linear carries candidates on these inputs")

    lane_cache: dict[str, Any] = {}

    def serve_lane_for(name: str, fmt: str):
        for cand in candidates.get(name, ()):
            if cand.fmt == fmt:
                return cand.serving_lane
        if fmt not in lane_cache:
            lane_cache[fmt] = serving_lane_route(target_profile, fmt)
        return lane_cache[fmt]

    report: dict = {
        "schema": SCHEMA,
        "evidence_status": _EVIDENCE_LABEL,
        "inputs": {
            "probe": str(probe_path),
            "costs": str(costs_path),
            "formats": list(formats),
            "target_profile": str(target_profile or "research"),
            "cb_format": cb_format,
            "native_format": native_format,
            "baseline_cb_format": baseline_cb_format,
            "cb_grid": cb_grid,
            "exclude_markers": list(exclude_markers),
        },
        "packed_expert_exclusion": dict(PACKED_EXPERT_EXCLUSION),
        "activation_fair_pricing": activation_pricing.as_dict(),
        "cross_family": cross_family_section(cost_payload),
        "serve_constraints_context": (
            serve_context.stamp_inactive() if not serve_context.active
            else {
                "active": True,
                "dispatch_table": serve_context.table.identity(),
                "workload_mix": serve_context.mix.as_dict(),
                "slos": serve_context.slos.as_dict(),
            }
        ),
        "relative_tax_reference_rule": RELATIVE_TAX_REFERENCE_RULE,
        "experiments": {},
    }

    arm_probe_rows: list[dict] = []
    if not skip_matched_bytes:
        matched = run_matched_bytes_experiment(
            candidates, stats,
            cb_format=cb_format,
            native_format=native_format,
            baseline_format=baseline_format,
            exclude_markers=exclude_markers,
            cb_serialization_context=cb_serialization_context,
            serve_context=serve_context,
            serve_lane_for=serve_lane_for,
        )
        report["experiments"]["matched_bytes"] = matched
        for arm in matched["arms"]:
            serve = arm.get("serve_constraints") or {}
            arm_probe_rows.append({
                "label": arm["label"],
                "feasible": bool(serve.get("feasible")),
                "predicted": dict(serve.get("predicted") or {}),
            })

    if not skip_byte_neutral:
        sweep = run_byte_neutral_sweep(
            candidates, stats,
            native_format=native_format,
            baseline_cb_format=baseline_cb_format,
            cb_grid=cb_grid,
            promote_counts=promote_counts,
            exclude_markers=exclude_markers,
            cb_serialization_context=cb_serialization_context,
            serve_context=serve_context,
            serve_lane_for=serve_lane_for,
        )
        report["experiments"]["byte_neutral_sweep"] = sweep
        for point in sweep["points"]:
            serve = point["arm"].get("serve_constraints") or {}
            arm_probe_rows.append({
                "label": point["arm"]["label"],
                "feasible": bool(serve.get("feasible")),
                "predicted": dict(serve.get("predicted") or {}),
            })

    if serve_context.active:
        report["fastest_feasible_reference"] = fastest_feasible_summary(
            arm_probe_rows,
            scope_note=(
                "Scope: the ARMS OF THIS CONTEST only — a hand-built, "
                "deliberately small set — not an enumeration of the globally "
                "feasible assignments under this byte budget. Any relative "
                "tax quoted against it must say so."
            ),
        )

    # Serving-lane provenance for the arms that exist, so the report says
    # which selected rungs ride a backed fused mid-M lane and which take the
    # expand+GEMM fallback (P5b) — the trade the contest is about.
    lanes: dict[str, dict] = {}
    for exp in report["experiments"].values():
        arms = exp.get("arms") or [p["arm"] for p in exp.get("points", ())]
        baseline_arm = exp.get("baseline")
        for arm in list(arms) + ([baseline_arm] if baseline_arm else []):
            assignment = arm.get("assignment")
            if not assignment:
                continue
            lanes[arm["label"]] = selection_serving_lane_provenance(
                assignment, candidates, target_profile)
    report["serving_lane_provenance_by_arm"] = lanes
    return report


def summarize(report: Mapping) -> str:
    """Operator-facing summary. Mirrors the report's refusals exactly."""
    lines: list[str] = []
    lines.append("[d03] " + str(report.get("evidence_status", "")))
    cross = report.get("cross_family", {})
    if cross.get("publishable"):
        lines.append(
            f"[d03] cross-family verdict: {cross.get('verdict')} — "
            f"{cross.get('detail')}")
    else:
        lines.append(
            "[d03] cross-family verdict WITHHELD "
            f"({cross.get('refusal_reason')}): {cross.get('detail')}")
    matched = report.get("experiments", {}).get("matched_bytes")
    if matched:
        arms = matched["arms"]
        lines.append(
            "[d03] (i) matched-bytes dense contest over "
            f"{matched['scope']['n_eligible_units']} unit(s): "
            + "; ".join(
                f"{a['label']} = {a['exact_bytes']} B, "
                f"Δloss {a['predicted_dloss']:.6e}"
                for a in arms
            )
            + f" | Δbytes = {matched['bytes_delta_bytes']} "
            f"({matched['bytes_delta_fraction'] * 100:.4f}%) | "
            f"matched_bytes_gate={matched['matched_bytes_gate']}"
        )
        if not matched["matched_bytes_gate"]:
            lines.append(
                "[d03]     the arms are NOT byte-matched within "
                f"{MATCHED_BYTES_TOLERANCE_FRACTION * 100:.1f}%, so this is "
                "not a same-rate contest and no quality verdict follows.")
    sweep = report.get("experiments", {}).get("byte_neutral_sweep")
    if sweep:
        lines.append(
            f"[d03] (ii) byte-neutral sweep at {sweep['byte_budget_bytes']} B "
            f"({len(sweep['points'])} point(s)):")
        for point in sweep["points"]:
            lines.append(
                f"[d03]     promote {point['n_promoted']:>3} -> "
                f"{point['bytes_vs_baseline']:+d} B, "
                f"net Δloss {point['net_dloss_vs_baseline']:+.6e}, "
                f"{point['n_funding_moves']} funding move(s), "
                f"byte_neutral={point['byte_neutral']}")
    lines.append(
        "[d03] packed-expert vanilla NVFP4 EXCLUDED from the contest: "
        + PACKED_EXPERT_EXCLUSION["reason"]
        + " Unlocked by " + PACKED_EXPERT_EXCLUSION["unlocked_by"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python3 -m prismaquant.d03_exact_rate",
        description=(
            "gridbook ROADMAP D0.3 exact-rate experiments (ultraplan P5d). "
            "Prepares release-gate evidence; does not constitute it."
        ),
    )
    ap.add_argument("--probe", required=True, help="sensitivity_probe pickle")
    ap.add_argument("--costs", required=True, help="measure_quant_cost pickle")
    ap.add_argument("--out", required=True, help="Output JSON report path")
    ap.add_argument("--formats", default="FP8_CB_K36,NVFP4,BF16",
                    help="Comma-separated menu the contest draws from")
    ap.add_argument("--target-profile", default=None,
                    help="Serving profile whose legality + serving lanes gate "
                         "the contest")
    ap.add_argument("--cb-format", default="FP8_CB_K36",
                    help="The CB contender of experiment (i)")
    ap.add_argument("--native-format", default="NVFP4",
                    help="The vanilla/native contender")
    ap.add_argument("--baseline-format", default=None,
                    help="Format pinned on units outside the contest, so the "
                         "arms differ only on contested rows")
    ap.add_argument("--baseline-cb-format", default="FP8_CB_K36",
                    help="Baseline rung of the byte-neutral sweep")
    ap.add_argument("--cb-grid", default="fp8", choices=("fp8", "fp4"),
                    help="Which CB grid's ladder funds the promotions")
    ap.add_argument("--promote-counts", default="1,2,4,8",
                    help="Comma-separated promotion counts to sweep")
    ap.add_argument("--exclude", default=".experts.,.expert_",
                    help="Comma-separated qname substrings excluded from the "
                         "contest. Packed experts are excluded BY DEFAULT: "
                         "vanilla NVFP4 has no packed-expert emit path "
                         "(gridbook D0.2).")
    ap.add_argument("--skip-matched-bytes", action="store_true")
    ap.add_argument("--skip-byte-neutral", action="store_true")
    ap.add_argument("--cb-scale-coding", default="two_tier",
                    choices=("two_tier", "v1"))
    ap.add_argument("--cb-codebook-source", default="lattice")
    ap.add_argument("--cb-scale-sweep", choices=("0", "1"), default="1")
    ap.add_argument("--cb-encode-tier", default="balanced",
                    choices=("fast", "balanced", "max"))
    ap.add_argument("--serve-dispatch-table", default=None)
    ap.add_argument("--serve-workload-mix", default=None)
    ap.add_argument("--slo-prefill-p95-ttft-ms", type=float, default=None)
    ap.add_argument("--slo-decode-p95-itl-ms", type=float, default=None)
    ap.add_argument("--slo-decode-p05-tps", type=float, default=None)
    ap.add_argument("--serve-device-budget-bytes", type=int, default=None)
    ap.add_argument("--serve-kv-bytes", type=int, default=0)
    ap.add_argument("--serve-peak-scratch-bytes", type=int, default=0)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    promote_counts = [
        int(x) for x in str(args.promote_counts).split(",") if x.strip()
    ]
    exclude = [x for x in str(args.exclude).split(",") if x]

    cb_context = None
    if any(is_cb_format(f) for f in formats):
        cb_context = CBSerializationContext(
            scale_coding=args.cb_scale_coding,
            codebook_source=args.cb_codebook_source,
            scale_sweep=args.cb_scale_sweep == "1",
            encode_tier=args.cb_encode_tier,
        )
    try:
        serve_context = ServeConstraintContext(
            table=(
                load_dispatch_table(args.serve_dispatch_table)
                if args.serve_dispatch_table else None
            ),
            mix=WorkloadMix.parse(args.serve_workload_mix),
            slos=ServeSLOs(
                p95_ttft_ms=args.slo_prefill_p95_ttft_ms,
                p95_itl_ms=args.slo_decode_p95_itl_ms,
                p05_tps=args.slo_decode_p05_tps,
                device_budget_bytes=args.serve_device_budget_bytes,
                kv_bytes=int(args.serve_kv_bytes or 0),
                peak_scratch_bytes=int(args.serve_peak_scratch_bytes or 0),
            ),
        )
        serve_context.validate()
        report = run(
            probe_path=args.probe,
            costs_path=args.costs,
            formats=formats,
            target_profile=args.target_profile,
            cb_format=args.cb_format,
            native_format=args.native_format,
            baseline_format=args.baseline_format,
            baseline_cb_format=args.baseline_cb_format,
            cb_grid=args.cb_grid,
            promote_counts=promote_counts,
            exclude_markers=exclude,
            cb_serialization_context=cb_context,
            serve_context=serve_context,
            skip_matched_bytes=args.skip_matched_bytes,
            skip_byte_neutral=args.skip_byte_neutral,
        )
    except (D03Error, DispatchTableError, ServeConstraintError) as exc:
        print(f"[d03] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(summarize(report), flush=True)
    print(f"[d03] report -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
