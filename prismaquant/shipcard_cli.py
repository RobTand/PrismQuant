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
    ROUTE_CENSUS_SLOT,
    UNIFORM_CONTROL_METRIC_KEYS,
    UNIFORM_CONTROL_SLOT,
    _verify_gold_record,
    assert_weight_stat_attestation,
    compute_model_sha,
    ensure_optional_slot,
    fill_slot,
    load_shipcard,
    make_record,
    make_route_census_record,
    make_uniform_control_record,
    record_uniform_control_override,
    required_slots,
    reattest_weight_stats,
    uniform_control_summary,
    verify,
    write_shipcard,
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
    # Principle 12: the bpp claim and the verdict of the arm that tests
    # whether those bytes were spent well are printed together, always.
    summary = uniform_control_summary(
        card, model_dir=Path(args.shipcard).resolve().parent)
    print(f"  uniform control: {summary['detail']}"
          + ("  [OVERRIDDEN]" if summary["overridden"] else ""))
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


def _cmd_fill_control(args: argparse.Namespace) -> int:
    """Close `uniform_control` from Tessera's block plus the control's own KL."""
    card = load_shipcard(args.shipcard)
    model_dir = args.model_dir or str(Path(args.shipcard).resolve().parent)
    block = json.loads(Path(args.control_block).read_text())
    payload = json.loads(Path(args.control_record).read_text())

    verdict = (block.get("verdict") or {}) if isinstance(block, dict) else {}
    if not verdict.get("measured") and not args.allow_unserved:
        print(
            f"[shipcard] REFUSED: {args.control_block} carries an UNSERVED "
            "verdict — the control was built and priced but neither arm was "
            "served. A built control is not a passed gate. Re-run Tessera's "
            "`experiments/uniform_control.py verify` with both served KLs, or "
            "pass --allow-unserved to record the absence (verify will still "
            "refuse).",
            file=sys.stderr)
        return 2

    control_model_dir = args.control_model_dir
    if control_model_dir is None:
        candidate = payload.get("model")
        if candidate and Path(str(candidate)).is_dir():
            control_model_dir = str(candidate)
    if control_model_dir is None:
        print("[shipcard] ERROR: cannot resolve the CONTROL checkpoint's "
              "directory — the record JSON has no local 'model' path; pass "
              "--control-model-dir", file=sys.stderr)
        return 2

    control_arm = {
        "tool": args.tool or f"record:{Path(args.control_record).name}",
        "model_sha": compute_model_sha(control_model_dir),
        "git_commit": (payload.get("git_commit")
                       or (payload.get("git") or {}).get("commit")),
        "serve_fingerprint": payload.get("serve_fingerprint"),
        "spec_decode_detected": payload.get("spec_decode_detected"),
        "metrics": {k: payload[k] for k in CARRIED_METRIC_KEYS if k in payload},
        "measured_model": payload.get("model"),
        "record_path": str(Path(args.control_record).resolve()),
    }
    record = make_uniform_control_record(
        tool=args.tool or f"uniform_control:{Path(args.control_block).name}",
        model_sha=compute_model_sha(model_dir),
        control_block=block,
        control_arm=control_arm,
        gold_metric_key=args.gold_metric_key,
        git_commit=(payload.get("git_commit")
                    or (payload.get("git") or {}).get("commit")),
    )
    ensure_optional_slot(args.shipcard, UNIFORM_CONTROL_SLOT)
    fill_slot(args.shipcard, UNIFORM_CONTROL_SLOT, record)
    summary = uniform_control_summary(
        load_shipcard(args.shipcard), model_dir=model_dir)
    print(f"[shipcard] filled {UNIFORM_CONTROL_SLOT} from "
          f"{args.control_block} (passed={record['passed']})")
    print(f"[shipcard]   {summary['detail']}")
    return 0


def _cmd_fill_route_census(args: argparse.Namespace) -> int:
    """Close `route.census` from the priced routes and the served records."""
    from prismaquant.tessera_route_receipt import (
        TesseraRouteReceiptError,
        parse_census_json,
        substitute_decoders_from_contract_answer,
    )

    model_dir = args.model_dir or str(Path(args.shipcard).resolve().parent)
    try:
        records = parse_census_json(Path(args.census).read_bytes().decode("utf-8"), where=str(args.census))
    except (OSError, ValueError) as exc:
        print(f"[shipcard] ERROR: cannot read census rows from "
              f"{args.census}: {exc}", file=sys.stderr)
        return 2
    card = load_shipcard(args.shipcard)
    scoped = isinstance(records, dict)
    binding = None
    if scoped:
        try:
            if not args.layer_config:
                raise TesseraRouteReceiptError("v2 census requires --layer-config exact allocation input")
            binding = {"layer_config_json": Path(args.layer_config).read_bytes().decode("utf-8"),
                       "config_json": (Path(model_dir) / "config.json").read_bytes().decode("utf-8"),
                       "manifest_json": (Path(model_dir) / "tessera_serving_manifest.json").read_bytes().decode("utf-8")}
        except (OSError, ValueError) as exc:
            print(f"[shipcard] REFUSED: cannot bind scoped census: {exc}", file=sys.stderr)
            return 2
    substitutes = list(args.substitute_decoder or ())
    if not substitutes and not scoped:
        try:
            from prismaquant import tessera_runtime_contract as trc

            contract = trc.load_tessera_contract()
            if contract is not None:
                substitutes = list(
                    substitute_decoders_from_contract_answer(
                        trc.contract_answer(contract)))
        except trc.TesseraContractError as exc:
            print(f"[shipcard] ERROR: {exc}", file=sys.stderr)
            return 2
    if not substitutes and not scoped:
        print("[shipcard] ERROR: no substitute decoder is known -- pass "
              "--substitute-decoder explicitly (repeatable) or set "
              "PRISMAQUANT_TESSERA_DEV_PIN so the pinned contract answer "
              "can be read. A gate that knows no substitute detects "
              "nothing.", file=sys.stderr)
        return 2
    try:
        record = make_route_census_record(
            tool=args.tool or f"route-census:{Path(args.census).name}",
            model_sha=compute_model_sha(model_dir),
            priced_routes=list(args.priced_route),
            route_records=records,
            substitute_decoders=substitutes,
            binding=binding,
            build=card.get("build"),
            model_dir=model_dir,
        )
    except TesseraRouteReceiptError as exc:
        print(f"[shipcard] REFUSED: {args.census} cannot be a census "
              f"receipt: {exc}", file=sys.stderr)
        return 2
    # Lane-scoped, not optional: the slot exists on cards the lane opened
    # (`lane_shipcard open --lane tessera`).  Filling a card that never
    # opened it is a refusal, not an auto-added key -- a slot the card does
    # not owe is a slot the receipt does not belong on.
    if ROUTE_CENSUS_SLOT not in (card.get("slots") or {}):
        print(f"[shipcard] REFUSED: {args.shipcard} has no "
              f"{ROUTE_CENSUS_SLOT} slot; open a Tessera lane card first "
              f"(python -m prismaquant.lane_shipcard open --lane tessera "
              f"--artifact {model_dir})", file=sys.stderr)
        return 2
    fill_slot(args.shipcard, ROUTE_CENSUS_SLOT, record)
    print(f"[shipcard] filled {ROUTE_CENSUS_SLOT} from {args.census} "
          f"(passed={record['passed']})")
    print(f"[shipcard]   {record['detail']}")
    return 0


def _confirm_artifact_name(model_dir: str, typed: str | None) -> str | None:
    """Re-typing the basename is the confirmation, as `publish_artifact` has it."""
    expected = Path(model_dir).resolve().name
    if typed is None:
        if not sys.stdin.isatty():
            print(
                "[shipcard] REFUSED: override-control needs the artifact "
                f"directory basename re-typed ({expected!r}); no tty, so pass "
                "--confirm-name", file=sys.stderr)
            return None
        typed = input(
            "[shipcard] Type the artifact directory name to ship an "
            f"allocation that LOST to its uniform control ({expected}): ")
    if str(typed).strip() != expected:
        print(f"[shipcard] REFUSED: typed {str(typed).strip()!r} != "
              f"{expected!r}; the confirmation must match the artifact "
              "directory basename", file=sys.stderr)
        return None
    return expected


def _cmd_override_control(args: argparse.Namespace) -> int:
    """Ship an allocation that lost to its control — deliberately, and stamped."""
    card = load_shipcard(args.shipcard)
    model_dir = args.model_dir or str(Path(args.shipcard).resolve().parent)
    summary = uniform_control_summary(card, model_dir=model_dir)
    if not summary["filled"]:
        print("[shipcard] REFUSED: no uniform_control record to override; "
              "fill the slot first (fill-control)", file=sys.stderr)
        return 2
    if not summary["measured"]:
        print("[shipcard] REFUSED: this control was never SERVED. An override "
              "forgives a measured loss; it is not a way to skip the "
              "measurement. Serve the control arm, or publish with "
              "--force-unverified and wear that stamp instead.",
              file=sys.stderr)
        return 2
    if summary["beat_control"]:
        print("[shipcard] REFUSED: this allocation BEAT its uniform control "
              f"({summary['candidate_over_control']!r}x); there is nothing to "
              "override.", file=sys.stderr)
        return 2

    typed = _confirm_artifact_name(model_dir, args.confirm_name)
    if typed is None:
        return 1
    if card.get("weight_stat_attestation") is not None:
        assert_weight_stat_attestation(card, model_dir)
    record_uniform_control_override(
        card,
        reason=args.reason,
        authorized_by=args.authorized_by,
        confirmed_artifact_name=typed,
    )
    write_shipcard(args.shipcard, card)
    print(f"[shipcard] stamped uniform_control_override=true into "
          f"{args.shipcard}: this artifact ships an allocation "
          f"{summary['candidate_over_control']!r}x worse than spending the "
          "same bytes at one rung")
    problems = verify(card, model_dir=model_dir)
    print(f"[shipcard] remaining problems: {problems or 'none'}")
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

    p_control = sub.add_parser(
        "fill-control",
        help="close uniform_control from Tessera's control block plus the "
             "control checkpoint's own served KL",
    )
    p_control.add_argument("shipcard")
    p_control.add_argument(
        "--control-block", required=True,
        help="JSON from tessera.control.control_block() / Tessera's "
             "experiments/uniform_control.py verify")
    p_control.add_argument(
        "--control-record", required=True,
        help="the CONTROL checkpoint's gold KL result JSON, written by the "
             "same tool that filled this card's gold.kl")
    p_control.add_argument(
        "--control-model-dir", default=None,
        help="the control checkpoint's directory (default: the record's own "
             "'model' path)")
    p_control.add_argument("--model-dir", default=None)
    p_control.add_argument(
        "--gold-metric-key", default="kl_mean",
        choices=sorted(UNIFORM_CONTROL_METRIC_KEYS),
        help="which gold KL metric both arms are compared on")
    p_control.add_argument("--tool", default=None)
    p_control.add_argument(
        "--allow-unserved", action="store_true",
        help="record a control that was built but never served (verify still "
             "refuses it)")
    p_control.set_defaults(func=_cmd_fill_control)

    p_override = sub.add_parser(
        "override-control",
        help="ship an allocation that LOST to its byte-matched uniform "
             "control; requires the artifact basename re-typed and stamps "
             "the card",
    )
    p_override.add_argument("shipcard")
    p_override.add_argument("--reason", required=True)
    p_override.add_argument("--authorized-by", required=True)
    p_override.add_argument(
        "--confirm-name", default=None,
        help="re-typed artifact directory basename (required with no tty)")
    p_override.add_argument("--model-dir", default=None)
    p_override.set_defaults(func=_cmd_override_control)

    p_census = sub.add_parser(
        "fill-route-census",
        help="close route.census from the priced routes and the serve's "
             "route records (Tessera lane: priced-vs-served decoder gate)",
    )
    p_census.add_argument("shipcard")
    p_census.add_argument(
        "--census", required=True,
        help="Complete Tessera route_census/2 JSON (scoped), or historical unscoped row array")
    p_census.add_argument("--layer-config", default=None,
                         help="Exact allocation JSON bound by card.build.layer_config_sha; required for v2")
    p_census.add_argument(
        "--priced-route", action="append", default=[],
        help="legacy priced route (repeatable, required for flat rows); optional cross-check for v2")
    p_census.add_argument(
        "--substitute-decoder", action="append", default=[],
        help="a decoder a serve falls back to (repeatable; default: derived "
             "from the pinned Tessera contract answer, which needs "
             "PRISMAQUANT_TESSERA_DEV_PIN)")
    p_census.add_argument("--model-dir", default=None)
    p_census.add_argument("--tool", default=None)
    p_census.set_defaults(func=_cmd_fill_route_census)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
