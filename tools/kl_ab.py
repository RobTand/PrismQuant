#!/usr/bin/env python3
"""Compare two gold-lane result JSONs — and refuse to cross serving stacks.

R15. §7.4's rule ("A/B arms must have identical extension residency; conf-KL
deltas under ~+-20% across differing serving stacks are not evidence") was prose
with nothing enforcing it, and the mechanism is dated and measured: on the 27B,
the same artifact read 0.01134 vs 0.01328 conf-KL (+-17%) keyed purely on whether
one lane's CUDA `.so` was resident during the dump.  (That reading was taken on
the Gridbook lane, retired 2026-09-02 -- archive/gridbook_lane_2026-09-02/ --
but the mechanism belongs to the loader, not to any one lane.)

    python3 tools/kl_ab.py A.json B.json
    python3 tools/kl_ab.py A.json B.json --metric ppl
    python3 tools/kl_ab.py A.json B.json --allow-cross-fingerprint

Current result JSONs carry two deliberately different identities:

* ``performance_stack_fingerprint`` is the comparable stack projection.  It
  intentionally omits arm artifact and live-session identity, and it must match
  before a delta is evidence.
* ``serve_fingerprint`` binds each individual result to its artifact and live
  session.  It is recomputed and validated for each arm, but it is *not* the
  cross-arm equality key and normally differs in a legitimate A/B.

Different performance-stack fingerprints exit 3 without a delta and name the
stack keys that differ. ``--allow-cross-fingerprint`` downgrades the report to
a **range**: it prints the +-20% band, says whether the measured difference
clears it, and never calls a within-band difference a win. A current-looking
record with a missing or stale manifest fingerprint also exits 3. Two genuinely
legacy bare metric JSONs (no manifest or fingerprints) compare as before with a
warning because there is no attestation to replay.

Where one arm maps a pinned native extension the other does not, and the
pinned Tessera runtime publishes a substitute decoder for that absence
(``when_unavailable``), the refusal names it (PrismaQuant #142): that arm
measured the substitute -- or never served -- and the drift band is not the
reading. The note is conditional on the arm having served a Tessera artifact,
which the manifest cannot prove, so it says so rather than asserting it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # package mode (`python -m tools.kl_ab`)
    from .serve_fingerprint import (
        fingerprint,
        native_extension_status,
        performance_stack_fingerprint,
        performance_stack_payload,
    )
except ImportError:  # script mode (`python tools/kl_ab.py`)
    from serve_fingerprint import (  # type: ignore
        fingerprint,
        native_extension_status,
        performance_stack_fingerprint,
        performance_stack_payload,
    )

#: §7.4: below this relative delta, a cross-stack comparison is not evidence.
CROSS_STACK_BAND = 0.20

#: Preference order when the caller does not name a metric.
METRIC_PREFERENCE = (
    "kl_confident_mean",
    "kl_mean",
    "ppl",
    "mean_nll",
    "last_token_kl",
)


def _load(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: not a result object")
    return payload


def _pick_metric(a: Mapping[str, Any], b: Mapping[str, Any],
                 requested: str | None) -> str:
    if requested:
        for name, side in ((a, "A"), (b, "B")):
            if requested not in name:
                raise SystemExit(f"metric {requested!r} missing from {side}")
        return requested
    for key in METRIC_PREFERENCE:
        if _finite(a.get(key)) and _finite(b.get(key)):
            return key
    raise SystemExit(
        "no shared finite metric; pass --metric (looked for "
        f"{list(METRIC_PREFERENCE)})")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _manifest(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    manifest = payload.get("serve_manifest")
    return manifest if isinstance(manifest, dict) else None


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validated_attestation(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str | None, str | None, Mapping[str, Any] | None, str | None]:
    """Return ``(performance, serve, manifest, problem)`` for one arm.

    Fingerprint strings are evidence only after replaying their canonical
    projections from the embedded manifest.  A bare metric JSON with no trace
    of either attestation remains the explicitly supported legacy case.
    """
    manifest = _manifest(payload)
    root_performance = payload.get("performance_stack_fingerprint")
    root_serve = payload.get("serve_fingerprint")
    if manifest is None:
        if root_performance is None and root_serve is None:
            return None, None, None, None
        return None, None, None, (
            f"{label} carries a fingerprint without an embedded serve_manifest; "
            "it cannot be validated"
        )

    recorded_performance = manifest.get("performance_stack_fingerprint")
    recorded_serve = manifest.get("serve_fingerprint")
    if not _sha256(recorded_performance):
        return None, None, manifest, (
            f"{label} serve_manifest has no valid performance_stack_fingerprint"
        )
    if not _sha256(recorded_serve):
        return None, None, manifest, (
            f"{label} serve_manifest has no valid serve_fingerprint"
        )

    try:
        expected_performance = performance_stack_fingerprint(manifest)
        expected_serve = fingerprint(manifest)
    except Exception as exc:
        return None, None, manifest, (
            f"{label} serve_manifest cannot be canonicalized: {exc}"
        )
    if recorded_performance != expected_performance:
        return None, None, manifest, (
            f"{label} performance_stack_fingerprint is stale"
        )
    if recorded_serve != expected_serve:
        return None, None, manifest, f"{label} serve_fingerprint is stale"
    status_problem = _native_extension_status_problem(manifest)
    if status_problem is not None:
        return None, None, manifest, (
            f"{label} {status_problem}"
        )
    if root_performance is not None and root_performance != recorded_performance:
        return None, None, manifest, (
            f"{label} top-level performance_stack_fingerprint differs from "
            "its serve_manifest"
        )
    if root_serve is not None and root_serve != recorded_serve:
        return None, None, manifest, (
            f"{label} top-level serve_fingerprint differs from its serve_manifest"
        )
    return recorded_performance, recorded_serve, manifest, None


def _native_extension_status_problem(
    manifest: Mapping[str, Any],
) -> str | None:
    """Replay the manifest's expected-vs-found extension block, if carried.

    ``native_extension_status`` is a deterministic projection of
    ``resident_extensions`` through the tool-carried rows, so a manifest
    whose block disagrees with the replay is stale or tampered with.  A
    manifest that predates the key carries no block and is simply legacy:
    its fingerprints still validate, because the block is excluded from
    both.  Rows the current tool knows and the manifest does not are
    likewise ignored -- an older manifest beside a newer pin is still
    evidence about the rows it does carry, and the §7.4 refusal below fires
    on ``resident_extensions`` either way.
    """
    if not isinstance(manifest, Mapping):
        return None
    carried = manifest.get("native_extension_status")
    if carried is None:
        return None
    if not isinstance(carried, list):
        return "native_extension_status is not a list"
    try:
        expected = {row["module_name_prefix"]: row
                    for row in native_extension_status(
                        manifest.get("resident_extensions"))}
    except Exception as exc:
        return f"native_extension_status cannot be replayed: {exc}"
    for row in carried:
        if not isinstance(row, Mapping):
            return "native_extension_status carries a non-object row"
        prefix = row.get("module_name_prefix")
        if prefix not in expected:
            return (
                f"native_extension_status carries {prefix!r}, which the "
                "current tool rows do not publish"
            )
        if dict(row) != dict(expected[prefix]):
            return (
                f"native_extension_status for {prefix!r} disagrees with "
                "resident_extensions"
            )
    return None


def _substitute_decoder_notes(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
    *,
    label_a: str,
    label_b: str,
) -> list[str]:
    """Name the substitute decoder for a one-sided missing extension (#142).

    §7.4's rule as written is a comparability warning about
    allocator-address drift ("deltas under ~±20% across differing stacks are
    not evidence").  With the pinned runtime's ``when_unavailable`` block in
    hand, one specific mismatch is categorically stronger: an arm missing a
    library the other arm maps, where the pinned table says a serve without
    it keeps serving on a NAMED substitute decoder, measured nothing about
    the lane at all -- at any delta.

    The reading is conditional on the arm having served a Tessera artifact
    (a stock serve is missing the same library and ran its own decoder), so
    the note says which arm, which library, and what the pinned table
    publishes, rather than asserting which decoder ran.
    """
    blocks: dict[str, dict[str, Any]] = {}
    for label, manifest in ((label_a, left), (label_b, right)):
        carried = (manifest.get("native_extension_status")
                   if isinstance(manifest, Mapping) else None)
        rows: dict[str, Any] = {}
        if isinstance(carried, list):
            for row in carried:
                if (isinstance(row, Mapping)
                        and isinstance(row.get("module_name_prefix"), str)):
                    rows[row["module_name_prefix"]] = row
        blocks[label] = rows
    notes: list[str] = []
    for prefix in sorted(set(blocks[label_a]) | set(blocks[label_b])):
        row_a = blocks[label_a].get(prefix)
        row_b = blocks[label_b].get(prefix)
        resident_a = (row_a.get("resident") if isinstance(row_a, Mapping)
                      else None)
        resident_b = (row_b.get("resident") if isinstance(row_b, Mapping)
                      else None)
        if resident_a is True and resident_b is False:
            missing, row = label_b, row_b if row_b is not None else row_a
        elif resident_a is False and resident_b is True:
            missing, row = label_a, row_a if row_a is not None else row_b
        else:
            continue
        behaviours = (row.get("when_unavailable")
                      if isinstance(row, Mapping) else None)
        if not isinstance(behaviours, Mapping) or not behaviours:
            continue
        rendered = "; ".join(
            f"{mode} -> {behaviour.get('status')}"
            + (f" ({behaviour.get('decoder')})"
               if behaviour.get("decoder") else "")
            for mode, behaviour in sorted(behaviours.items())
            if isinstance(behaviour, Mapping)
        )
        notes.append(
            f"  !! {missing} is missing {prefix} (resident in "
            f"{label_a if missing == label_b else label_b}): the pinned "
            f"Tessera runtime publishes when_unavailable {{{rendered}}}. If "
            f"{missing} was serving a Tessera artifact, it measured nothing "
            "about the lane at all -- it ran the substitute decoder, or "
            "never served -- at any delta.")
    return notes


def _performance_stack_differences(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> list[str]:
    if left is None or right is None:
        return []
    a = performance_stack_payload(left)
    b = performance_stack_payload(right)
    return sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))


def compare(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    metric: str,
    allow_cross_fingerprint: bool = False,
    label_a: str = "A",
    label_b: str = "B",
) -> tuple[int, list[str]]:
    """Return `(exit_code, lines)` — the whole verdict, no printing."""
    va, vb = float(a[metric]), float(b[metric])
    delta = vb - va
    rel = (delta / va) if va else float("nan")

    lines = [
        f"metric: {metric}",
        f"  {label_a}: {va:.6g}   ({a.get('model')})",
        f"  {label_b}: {vb:.6g}   ({b.get('model')})",
    ]

    for label, payload in ((label_a, a), (label_b, b)):
        if payload.get("spec_decode_detected"):
            lines.append(
                f"  !! {label} reports spec_decode_detected=true — that number "
                "is the DRAFT model's, not the artifact's (§7.5)")

    pa, sa, ma, problem_a = _validated_attestation(a, label=label_a)
    pb, sb, mb, problem_b = _validated_attestation(b, label=label_b)
    problems = [problem for problem in (problem_a, problem_b) if problem]
    if problems:
        lines.append("")
        lines.extend(f"INVALID ATTESTATION: {problem}" for problem in problems)
        lines.append(
            "REFUSED: re-create the affected result with the current gold-lane "
            "tool; no A/B delta can be derived from stale or unverifiable "
            "fingerprints.")
        return 3, lines

    legacy_a = ma is None
    legacy_b = mb is None
    if legacy_a != legacy_b:
        lines.append("")
        lines.append(
            "REFUSED: one arm is an unattested legacy metric and the other is "
            "a current attested result. Re-measure both arms under one current "
            "performance stack.")
        return 3, lines

    if pa and pb and pa != pb:
        differing = _performance_stack_differences(ma, mb)
        lines.append("")
        lines.append(f"performance_stack_fingerprint {label_a}: {pa[:16]}")
        lines.append(f"performance_stack_fingerprint {label_b}: {pb[:16]}")
        lines.append(
            "  differing performance-stack keys: "
            + (", ".join(differing) if differing
               else "(manifests not embedded in the result JSONs)"))
        lines.append(f"  serve_fingerprint {label_a}: {sa[:16]} (validated)")
        lines.append(f"  serve_fingerprint {label_b}: {sb[:16]} (validated)")
        lines.extend(_substitute_decoder_notes(
            ma, mb, label_a=label_a, label_b=label_b))
        if not allow_cross_fingerprint:
            lines.append("")
            lines.append(
                "REFUSED: these numbers come from different serving stacks. "
                "Loading any CUDA extension shifts allocator addresses and "
                "flips alignment-sensitive kernel selection — the same 27B "
                "artifact read 0.01134 vs 0.01328 conf-KL (±17%) on that "
                "alone. Re-measure both arms on one stack, or pass "
                "--allow-cross-fingerprint to quote a range instead of a "
                "delta.")
            return 3, lines
        band = CROSS_STACK_BAND * 100.0
        lines.append("")
        lines.append(
            f"CROSS-STACK RANGE (not a delta). §7.4 band: ±{band:.0f}% of "
            f"{label_a} = [{va * (1 - CROSS_STACK_BAND):.6g}, "
            f"{va * (1 + CROSS_STACK_BAND):.6g}]")
        lines.append(f"  measured relative difference: {rel * 100:+.1f}%")
        if abs(rel) <= CROSS_STACK_BAND:
            lines.append(
                "  VERDICT: INSIDE the band — NOT EVIDENCE either way. Quote "
                f"{label_b} as 'within ±{band:.0f}% of {label_a} across "
                "differing serving stacks', never as a win or a regression.")
        else:
            lines.append(
                f"  VERDICT: outside the ±{band:.0f}% band, so the difference "
                "is unlikely to be residency alone — but it is still a "
                "cross-stack comparison; quote it as a range with the "
                "differing keys above, not as a measured delta.")
        return 0, lines

    if legacy_a and legacy_b:
        lines.append("")
        lines.append(
            "WARNING: neither JSON has a serve manifest or fingerprints "
            "(legacy JSONs). Comparing anyway, but nothing verified that "
            "these ran on the same serving stack — the ±17% residency drift "
            "is invisible here. Re-measure with a current tool to get a "
            "checked delta.")
    else:
        lines.append("")
        lines.append(
            f"performance_stack_fingerprint: {pa[:16]} (matched, validated)")
        lines.append(
            f"  serve_fingerprint {label_a}: {sa[:16]} "
            "(validated per-run attestation)")
        lines.append(
            f"  serve_fingerprint {label_b}: {sb[:16]} "
            "(validated per-run attestation)")

    ca = (a.get("git_commit") or "")[:12]
    cb = (b.get("git_commit") or "")[:12]
    if ca and cb and ca != cb:
        lines.append(f"NOTE: different git_commit ({ca} vs {cb}) — same serving "
                     "stack, different measuring code.")

    lines.append("")
    lines.append(f"delta ({label_b} - {label_a}): {delta:+.6g}  "
                 f"({rel * 100:+.2f}%)")
    return 0, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--metric", default=None,
                    help=f"default: first shared finite key in "
                         f"{list(METRIC_PREFERENCE)}")
    ap.add_argument("--allow-cross-fingerprint", action="store_true",
                    help="downgrade a refused cross-stack delta to an honest "
                         "±20% range")
    args = ap.parse_args(argv)

    a, b = _load(args.a), _load(args.b)
    metric = _pick_metric(a, b, args.metric)
    code, lines = compare(
        a, b,
        metric=metric,
        allow_cross_fingerprint=args.allow_cross_fingerprint,
        label_a=Path(args.a).name,
        label_b=Path(args.b).name,
    )
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
