#!/usr/bin/env python3
"""Open, fill and verify the ship record (`exported/shipcard.json`).

R13. `export_native_compressed` opens the card with the build-lane facts and
five empty, required serve-lane slots; this tool closes them and refuses.

    # after the serve lane has run
    python3 tools/shipcard.py show   exported/shipcard.json
    python3 tools/shipcard.py fill   exported/shipcard.json \
        --slot gold.kl --record /home/rob/dq-runs/<run>/kl_student.json
    python3 tools/shipcard.py verify exported/shipcard.json --model-dir exported

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.shipcard import (  # noqa: E402
    GOLD_SLOTS,
    REQUIRED_SLOTS,
    compute_model_sha,
    fill_slot,
    load_shipcard,
    make_record,
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
    "n_tokens_scored", "max_chunk_mean_nll", "quantization", "score_positions",
)


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except Exception:
        return False


def _cmd_show(args: argparse.Namespace) -> int:
    card = load_shipcard(args.shipcard)
    print(f"shipcard: {args.shipcard}")
    print(f"  model_dir:      {card.get('model_dir')}")
    print(f"  model_sha:      {str(card.get('model_sha'))[:16]}")
    build = card.get("build") or {}
    print(f"  git_commit:     {(build.get('git') or {}).get('commit')}")
    print(f"  achieved_bpp:   {(build.get('achieved_bpp') or {}).get('value')}"
          f"  ({(build.get('achieved_bpp') or {}).get('source')})")
    print(f"  artifact_bytes: {card.get('artifact_bytes')}")
    print("  slots:")
    for slot in REQUIRED_SLOTS:
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
    problems = verify(card, model_dir=args.model_dir)
    if not problems:
        print(f"[shipcard] OK — {len(REQUIRED_SLOTS)}/{len(REQUIRED_SLOTS)} "
              f"slots filled and matching for {card.get('model_dir')}")
        return 0
    print(f"[shipcard] REFUSED — {len(problems)} problem(s) with "
          f"{args.shipcard}:")
    for problem in problems:
        print(f"  - {problem}")
    return 1


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
    primary = next(
        (payload[k] for k in PRIMARY_METRIC_KEYS
         if k in payload and _finite(payload[k])),
        None,
    )
    passed = primary is not None if args.passed is None else args.passed
    if primary is None and args.passed is None:
        print(f"[shipcard] WARN {args.record} carries no finite metric in "
              f"{PRIMARY_METRIC_KEYS}; recording passed=false", file=sys.stderr)

    record = make_record(
        slot=args.slot,
        tool=args.tool or f"record:{Path(args.record).name}",
        passed=bool(passed),
        model_sha=model_sha,
        metrics=metrics,
        detail=args.detail or f"from {args.record}",
        spec_decode_detected=spec,
        serve_fingerprint=payload.get("serve_fingerprint"),
        git_commit=(payload.get("git_commit")
                    or (payload.get("git") or {}).get("commit")),
        extra={"record_path": str(Path(args.record).resolve()),
               "measured_model": payload.get("model")},
    )
    fill_slot(args.shipcard, args.slot, record)
    print(f"[shipcard] filled {args.slot} from {args.record} "
          f"(passed={bool(passed)}, primary={primary})")
    unfilled = [s for s in REQUIRED_SLOTS
                if not (load_shipcard(args.shipcard)["slots"].get(s))]
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
    p_verify.set_defaults(func=_cmd_verify)

    p_fill = sub.add_parser("fill", help="close a slot from a result JSON")
    p_fill.add_argument("shipcard")
    p_fill.add_argument("--slot", required=True, choices=list(REQUIRED_SLOTS))
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
