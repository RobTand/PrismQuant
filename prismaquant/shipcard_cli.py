#!/usr/bin/env python3
"""Open, fill and verify the ship record (`exported/shipcard.json`).

R13. `export_native_compressed` opens the card with the build-lane facts and
five empty, required serve-lane slots; this tool closes them and refuses.

    # after the serve lane has run
    python3 -m prismaquant.shipcard_cli show   exported/shipcard.json
    python3 -m prismaquant.shipcard_cli fill   exported/shipcard.json \
        --slot gold.kl --record /home/rob/dq-runs/<run>/kl_student.json
    python3 -m prismaquant.shipcard_cli verify exported/shipcard.json --model-dir exported

`verify` exits 0 only when every slot holds a passing record whose `model_sha`
matches the artifact on disk, and the two `gold.*` records report
`spec_decode_detected: false`. Anything else exits 1 and prints why.

`native_export.*` and `ship_gate` are filled by the validators themselves
(`--shipcard`); `fill` is for the gold lane, whose tools write a result JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from prismaquant.shipcard import (
    GOLD_SLOTS,
    OPTIONAL_SLOTS,
    _verify_gold_record,
    compute_model_sha,
    fill_slot,
    load_shipcard,
    make_record,
    required_slots,
    reattest_weight_stats,
    verify,
)

#: Metrics lifted out of a gold-lane result JSON onto the record, in the order
#: they are searched for a "primary" number.
PRIMARY_METRIC_KEYS = (
    "kl_confident_mean",
    "kl_mean",
    "ppl",
    "mean_nll",
    "last_token_kl",
)

CARRIED_METRIC_KEYS = PRIMARY_METRIC_KEYS + (
    "kl_p99", "kl_max", "n_positions", "n_confident", "n_samples", "seqlen",
    "n_tokens_scored", "per_chunk_mean_nll", "max_chunk_mean_nll",
    "quantization", "score_positions",
    "serve_manifest",
    "teacher_evidence", "mode", "prompt_top_k", "vocab_size",
    "split", "n_tokens_requested", "calibration_contract",
    "calibration_contract_sha256",
)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except Exception:
        return False


def _cmd_show(args: argparse.Namespace) -> int:
    card = load_shipcard(args.shipcard)
    blocking_slots = required_slots(
        card, model_dir=Path(args.shipcard).resolve().parent
    )
    optional_present = tuple(
        slot for slot in OPTIONAL_SLOTS
        if slot in (card.get("slots") or {})
    )
    slots_shown = tuple(dict.fromkeys(blocking_slots + optional_present))
    print(f"shipcard: {args.shipcard}")
    print(f"  model_dir:      {card.get('model_dir')}")
    print(f"  model_sha:      {str(card.get('model_sha'))[:16]}")
    build = card.get("build") or {}
    print(f"  git_commit:     {(build.get('git') or {}).get('commit')}")
    print(f"  achieved_bpp:   {(build.get('achieved_bpp') or {}).get('value')}"
          f"  ({(build.get('achieved_bpp') or {}).get('source')})")
    print(f"  artifact_bytes: {card.get('artifact_bytes')}")
    print("  slots:")
    for slot in slots_shown:
        record = (card.get("slots") or {}).get(slot)
        if not record:
            print(f"    [ ] {slot}  UNFILLED")
            continue
        state = "PASS" if record.get("passed") else "FAIL"
        print(f"    [x] {slot}  {state}  {record.get('tool')}  "
              f"{record.get('filled_at')}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    card = load_shipcard(args.shipcard)
    # The card lives inside the artifact.  Its parent is authoritative when
    # --model-dir is omitted, so moving an intact artifact does not weaken or
    # break verification and a stale path embedded before publication cannot
    # suppress the on-disk identity check.
    model_dir = args.model_dir or str(Path(args.shipcard).resolve().parent)
    blocking_slots = required_slots(card, model_dir=model_dir)
    claimed_slots = tuple(args.require_slot or ())
    slots_required = tuple(dict.fromkeys(blocking_slots + claimed_slots))
    problems = verify(card, model_dir=model_dir, required=slots_required)
    if not problems:
        print(f"[shipcard] OK — {len(slots_required)}/{len(slots_required)} "
              f"slots filled and matching for {card.get('model_dir')}")
        return 0
    print(f"[shipcard] REFUSED — {len(problems)} problem(s) with "
          f"{args.shipcard}:")
    for problem in problems:
        print(f"  - {problem}")
    return 1


def _cmd_reattest(args: argparse.Namespace) -> int:
    model_dir = args.model_dir or str(Path(args.shipcard).resolve().parent)
    reattest_weight_stats(args.shipcard, model_dir)
    print(
        f"[shipcard] exact weight content matches; refreshed stat attestation "
        f"for {model_dir}"
    )
    return 0


def _cmd_fill(args: argparse.Namespace) -> int:
    card = load_shipcard(args.shipcard)
    payload = json.loads(Path(args.record).read_text())

    model_dir = args.model_dir
    if model_dir is None:
        candidate = payload.get("model")
        if candidate and Path(str(candidate)).is_dir():
            model_dir = str(candidate)
    if model_dir is None:
        print("[shipcard] ERROR: cannot resolve the measured artifact "
              "directory — the record JSON has no local 'model' path; pass "
              "--model-dir", file=sys.stderr)
        return 2
    model_sha = compute_model_sha(model_dir)

    spec = payload.get("spec_decode_detected")
    if args.slot in GOLD_SLOTS and spec is not False:
        if not args.allow_spec_decode:
            print(
                f"[shipcard] REFUSED: {args.record} reports "
                f"spec_decode_detected={spec!r}. A gold number measured "
                "against a spec-decode serve is the DRAFT model's NLL, not "
                "the artifact's (§7.5). Re-measure on a no-spec serve, or "
                "pass --allow-spec-decode to record it anyway (verify will "
                "still refuse).",
                file=sys.stderr)
            return 2
        print("[shipcard] WARN recording a spec-decode-tainted gold number on "
              "the card; verify will refuse it", file=sys.stderr)

    metrics = {k: payload[k] for k in CARRIED_METRIC_KEYS if k in payload}
    if args.slot == "gold.kl":
        slot_primary_keys = ("kl_confident_mean", "kl_mean")
        count_ok = any(
            isinstance(payload.get(key), int)
            and not isinstance(payload.get(key), bool)
            and payload.get(key, 0) > 0
            for key in ("n_positions", "n_samples")
        )
    else:
        slot_primary_keys = ("ppl", "mean_nll")
        count_ok = (
            isinstance(payload.get("n_tokens_scored"), int)
            and not isinstance(payload.get("n_tokens_scored"), bool)
            and payload.get("n_tokens_scored", 0) > 0
        )
    slot_primary = next(
        (payload[key] for key in slot_primary_keys if key in payload and _finite(payload[key])),
        None,
    )
    fingerprint = payload.get("serve_fingerprint")
    commit = payload.get("git_commit") or (payload.get("git") or {}).get("commit")
    structurally_valid = (
        slot_primary is not None
        and float(slot_primary) >= 0
        and count_ok
        and isinstance(fingerprint, str)
        and len(fingerprint) == 64
        and all(character in "0123456789abcdef" for character in fingerprint)
        and isinstance(commit, str)
        and len(commit) in {40, 64}
        and all(character in "0123456789abcdef" for character in commit)
    )
    if args.passed is True and not structurally_valid:
        print(
            f"[shipcard] REFUSED: --passed cannot override missing or malformed "
            f"{args.slot} metric/count/fingerprint/git identity",
            file=sys.stderr,
        )
        return 2
    passed = structurally_valid if args.passed is None else args.passed
    if not structurally_valid:
        print(
            f"[shipcard] WARN {args.record} lacks complete {args.slot} "
            "metric/count/fingerprint/git evidence; recording passed=false",
            file=sys.stderr,
        )

    record = make_record(
        slot=args.slot,
        tool=args.tool or f"record:{Path(args.record).name}",
        passed=bool(passed),
        model_sha=model_sha,
        metrics=metrics,
        detail=args.detail or f"from {args.record}",
        spec_decode_detected=spec,
        serve_fingerprint=fingerprint,
        git_commit=commit,
        extra={"record_path": str(Path(args.record).resolve()),
               "measured_model": payload.get("model")},
    )
    # The gold-record replay here must track `verify()`: filling with a
    # stricter contract than publication replays would refuse evidence that
    # ships fine, and filling with a looser one would defer a refusal to the
    # very last gate. Until 2026-09-02 it ran only for the retired Gridbook
    # codebook lane's cards, gated on that lane's extra blocking slots; those
    # slots and their DSv4 release contract went into
    # archive/gridbook_lane_2026-09-02/ with the lane, so the generic replay
    # now runs for every structurally valid record instead of for none.
    replay_problems = (
        _verify_gold_record(
            args.slot,
            record,
            model_dir=model_dir,
            require_current_artifact_path=True,
        )
        if structurally_valid
        else []
    )
    if replay_problems:
        if args.passed is True:
            print(
                "[shipcard] REFUSED: --passed cannot override invalid gold "
                "evidence: " + "; ".join(replay_problems),
                file=sys.stderr,
            )
            return 2
        record["passed"] = False
        print(
            "[shipcard] WARN gold evidence did not replay; "
            "recording passed=false: " + "; ".join(replay_problems),
            file=sys.stderr,
        )
    fill_slot(args.shipcard, args.slot, record)
    print(f"[shipcard] filled {args.slot} from {args.record} "
          f"(passed={record['passed']}, primary={slot_primary})")
    updated_card = load_shipcard(args.shipcard)
    unfilled = [s for s in required_slots(updated_card, model_dir=model_dir)
                if not updated_card["slots"].get(s)]
    print(f"[shipcard] remaining unfilled: {unfilled or 'none'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_show = sub.add_parser("show", help="print slot state")
    p_show.add_argument("shipcard")
    p_show.set_defaults(func=_cmd_show)

    p_verify = sub.add_parser("verify", help="refuse unless every slot is closed")
    p_verify.add_argument("shipcard")
    p_verify.add_argument("--model-dir", default=None,
                          help="recompute model_sha from this directory and "
                               "require the records to match it")
    p_verify.add_argument(
        "--require-slot",
        action="append",
        choices=sorted(OPTIONAL_SLOTS),
        default=None,
        help="also require a named optional claim (repeatable)",
    )
    p_verify.set_defaults(func=_cmd_verify)

    p_reattest = sub.add_parser(
        "reattest",
        help="full-hash a legitimately copied CB artifact and refresh its stat cache",
    )
    p_reattest.add_argument("shipcard")
    p_reattest.add_argument("--model-dir", default=None)
    p_reattest.set_defaults(func=_cmd_reattest)

    p_fill = sub.add_parser("fill", help="close a slot from a result JSON")
    p_fill.add_argument("shipcard")
    # Native/ship-gate slots carry structured evidence and may only be closed
    # by their validators.  The generic record importer is deliberately
    # limited to the two metric slots it owns.
    p_fill.add_argument("--slot", required=True, choices=sorted(GOLD_SLOTS))
    p_fill.add_argument("--record", required=True,
                        help="result JSON written by the serve-lane tool")
    p_fill.add_argument("--model-dir", default=None,
                        help="artifact directory the record was measured on "
                             "(default: the record's own 'model' path)")
    p_fill.add_argument("--tool", default=None)
    p_fill.add_argument("--detail", default=None)
    p_fill.add_argument("--passed", dest="passed", action="store_true",
                        default=None)
    p_fill.add_argument("--failed", dest="passed", action="store_false")
    p_fill.add_argument("--allow-spec-decode", action="store_true",
                        help="record a gold number measured under spec-decode "
                             "anyway (verify still refuses it)")
    p_fill.set_defaults(func=_cmd_fill)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
