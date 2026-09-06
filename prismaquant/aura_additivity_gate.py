"""AURA assignment diagnostics with explicit covariance and identity scope.

The additivity residual compares measured end-KL with the sum of unary
quadratic prices. Complete joint rows establish common probe/calibration,
source, currency and per-candidate operator bindings before the empirical
standard error includes covariance. Bare legacy arrays retain their numeric
diagnostic with unverified alignment; equal lengths cannot establish pairing.

Probe sampling error is conditional on fixed calibration. Supplied held-out
sequence uncertainty remains a separate quantity. These diagnostics neither
admit an allocation nor establish generalization or background-independent
unary rankings.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Mapping, Sequence

from prismaquant.joint_aura import (
    ASSIGNMENT_OBJECTIVES, PROBE_UNCERTAINTY_SCOPE, assignment_probe_summary,
    paired_assignment_difference, validate_joint_aura_entry,
)

ZERO_COST_SOURCES = {"aura_passthrough_zero"}


def additivity_gate(
    cost_payload: Mapping,
    assignment: Mapping[str, str],
    measured_kl: float,
    *,
    measured_kl_stderr: float = 0.0,
) -> dict:
    """Compare an assignment's measured end-KL to AURA's additive prediction.

    Returns the predicted sum, empirical stderr, residual, descriptive
    z-score, identity/uncertainty scope, and coverage accounting.
    Uncovered members (assignment entries with no cost row) are LISTED, never
    silently dropped — a large uncovered set invalidates the comparison.
    """
    measured_kl = float(measured_kl)
    measured_kl_stderr = float(measured_kl_stderr)
    if not math.isfinite(measured_kl):
        raise ValueError("measured KL estimate must be finite")
    if not math.isfinite(measured_kl_stderr) or measured_kl_stderr < 0:
        raise ValueError("measured KL standard error must be finite and nonnegative")
    costs = cost_payload["costs"]
    covered: list[tuple[str, str]] = []
    uncovered: list[str] = []
    zero_rows = 0
    per_probe_ok = True
    joint_rows = {}
    n_probes = int(cost_payload.get("n_probes", 0))

    for name, fmt in assignment.items():
        fmt = str(fmt).strip().upper()
        row = costs.get(name, {}).get(fmt)
        if row is None:
            uncovered.append(f"{name}|{fmt}")
            continue
        # Validate joint claims before any zero-cost shortcut or sample use.
        if validate_joint_aura_entry(row):
            if row["joint_operator_identity"]["format"] != fmt:
                raise ValueError(f"joint AURA assignment format identity mismatch: {name}")
            joint_rows[name] = row
        if row.get("cost_source") in ZERO_COST_SOURCES or (
                row.get("predicted_dloss", 0.0) == 0.0
                and "x2_per_probe" not in row):
            zero_rows += 1
            continue
        covered.append((name, fmt))
        if "x2_per_probe" not in row or len(row["x2_per_probe"]) != n_probes:
            per_probe_ok = False

    predicted_sum = sum(
        float(costs[n][f]["predicted_dloss"]) for n, f in covered)

    alignment_verified = False
    probe_identity = None
    if joint_rows:
        if len(joint_rows) != len(covered) or zero_rows:
            raise ValueError("joint AURA diagnostic refuses mixed or unmeasured zero rows")
        summary = assignment_probe_summary(joint_rows, objective="additive")
        if n_probes != len(summary["probe_ids"]):
            raise ValueError("joint AURA payload probe count alignment mismatch")
        predicted_sum = summary["mean"]
        predicted_stderr = summary["standard_error"]
        stderr_method = "per_probe_aligned_empirical"
        alignment_verified = True
        probe_identity = summary["probe_identity_sha256"]
    elif covered and per_probe_ok and n_probes >= 2:
        # Preserve historical arithmetic without certifying a common draw.
        s = [0.0] * n_probes
        for n, f in covered:
            for k, x2 in enumerate(costs[n][f]["x2_per_probe"]):
                s[k] += x2
        mean_s = sum(s) / n_probes
        var_s = sum((v - mean_s) ** 2 for v in s) / (n_probes - 1)
        predicted_stderr = 0.5 * math.sqrt(var_s / n_probes)
        stderr_method = "per_probe_unverified"
    else:
        predicted_stderr = math.sqrt(sum(
            float(costs[n][f].get("predicted_dloss_stderr", 0.0)) ** 2
            for n, f in covered))
        # Ignoring covariance is not generally a lower bound: covariance can
        # be negative. This remains only the historical independence estimate.
        stderr_method = "independence_assumed"

    residual = float(measured_kl) - predicted_sum
    denom = math.hypot(predicted_stderr, measured_kl_stderr)
    z = residual / denom if denom > 0 else float("inf") if residual else 0.0

    return {
        "schema": "prismaquant.aura_additivity_gate.v1",
        "measured_kl": float(measured_kl),
        "measured_kl_stderr": float(measured_kl_stderr),
        "predicted_sum": predicted_sum,
        "predicted_stderr": predicted_stderr,
        "stderr_method": stderr_method,
        "probe_alignment_verified": alignment_verified,
        "probe_identity_sha256": probe_identity,
        "objective": "additive",
        "predicted_uncertainty_scope": (
            PROBE_UNCERTAINTY_SCOPE if alignment_verified
            else "unverified_probe_alignment"),
        "measured_uncertainty_scope": "caller_supplied_heldout_sequence_standard_error",
        "residual_z_scope": "descriptive_independent_probe_and_sequence_errors_assumed",
        "residual": residual,
        "residual_over_measured": (
            residual / measured_kl if measured_kl else 0.0),
        "residual_z": z,
        "n_covered": len(covered),
        "n_zero_cost": zero_rows,
        "uncovered": sorted(uncovered),
        "n_probes": n_probes,
    }


def paired_assignment_report(
    cost_payload: Mapping, assignment_a: Mapping[str, str],
    assignment_b: Mapping[str, str], *, objective: str = "additive",
) -> dict:
    """Compare two complete assignments from one cost artifact, A minus B.

    The signed-probe helper owns numerical/identity validation. Missing units
    and unmeasured BF16 placeholders cannot be invented as zeros for a pair.
    No held-out sequence standard error is inferred from these probe draws.
    """
    def select(assignment):
        rows = {}
        for name, fmt in assignment.items():
            fmt = str(fmt).strip().upper()
            row = cost_payload["costs"].get(name, {}).get(fmt)
            if row is None:
                raise ValueError(f"paired joint AURA missing assignment row: {name}|{fmt}")
            if not validate_joint_aura_entry(row):
                raise ValueError("paired joint AURA requires complete joint rows")
            if row["joint_operator_identity"]["format"] != fmt:
                raise ValueError(f"paired joint AURA format identity mismatch: {name}")
            rows[name] = row
        return rows

    result = paired_assignment_difference(select(assignment_a), select(assignment_b), objective=objective)
    if cost_payload.get("n_probes") != len(result["probe_ids"]):
        raise ValueError("paired joint AURA payload probe count alignment mismatch")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="AURA additivity gate: measured vs Σ predicted end-KL")
    p.add_argument("--costs", required=True, help="AURA cost.pkl")
    p.add_argument("--assignment", required=True,
                   help="layer_config.json (format-name or AutoRound dicts)")
    p.add_argument("--measured-kl", required=True, type=float)
    p.add_argument("--measured-kl-stderr", type=float, default=0.0)
    p.add_argument("--comparison-assignment", help="optional complete assignment B; report A minus B")
    p.add_argument("--paired-objective", choices=ASSIGNMENT_OBJECTIVES, default="additive",
                   help="paired diagnostic only; additivity prediction remains additive")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    from prismaquant.layer_config import load_assignment
    with open(args.costs, "rb") as fh:
        payload = pickle.load(fh)
    assignment = load_assignment(args.assignment)
    result = additivity_gate(
        payload, assignment, args.measured_kl,
        measured_kl_stderr=args.measured_kl_stderr)
    if args.comparison_assignment:
        result["paired_assignment"] = paired_assignment_report(
            payload, assignment, load_assignment(args.comparison_assignment),
            objective=args.paired_objective)
    text = json.dumps(result, indent=1)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
