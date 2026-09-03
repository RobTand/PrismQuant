"""The priced-vs-served route gate for the Tessera lane (PrismaQuant #136).

Tessera's runtime contract publishes, per native extension, what a serve
does when the ``.so`` cannot build (``native_extensions[].when_unavailable``,
contract v7): resident mode keeps serving on a NAMED substitute decoder and
stamps it on every route record.  The field exists so "a receipt must never
claim the native decoder for a serve that took the" fallback -- but nothing
in PrismaQuant read it, so a shipcard could price ``TESSERA_NVFP4`` W4A4
while a serve produced every number on ``torch_materialize_stock`` with
nothing refusing.

This module is the refusing consumer.  It reads route records -- the
``{route, decoder}`` rows Tessera's own ``tools/tessera_route_census.py``
parses out of a serve log, which this repository names rather than vendors
(``lane_specs/tessera.json``) -- and compares the routes the serve actually
emitted against the routes the artifact priced, refusing on:

* no records at all (an absent census is not a clean bill);
* any record without a decoder (a decoder-less row read as native is the
  hole that existed before this gate);
* any record whose decoder is a KNOWN substitute -- derived from the pinned
  contract's ``when_unavailable`` via
  :func:`substitute_decoders_from_contract_answer`, never hardcoded here;
* served-vs-priced route disagreement in either direction (a number whose
  units rode a route other than the one they were priced on is not a
  result).

Stdlib only, no torch: the shipcard replays this at publication through
``prismaquant.shipcard``, which is stdlib-only by contract.

What this does not do: the native decoder's NAME is not published anywhere
this repository reads, so the gate cannot assert "this ran native" -- it
refuses every substitute the pin names and stamps the served decoder set it
observed, which is the positive claim the receipt carries.  Coverage
strictness is uncalibrated against a real serve (nothing has been served
yet on this side): both directions refuse, and relaxing either needs a
measured serve, not an argument.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


class TesseraRouteReceiptError(ValueError):
    """The route census is missing, malformed, or disagrees with the price."""


def substitute_decoders_from_contract_answer(
    answer: Mapping[str, Any],
) -> tuple[str, ...]:
    """Every substitute decoder the pinned answer names, sorted.

    Derived from ``native_extensions[].when_unavailable[].decoder`` (nulls
    dropped: a refused serve has no decoder to detect).  The gate reads this
    set rather than a hardcoded name, so a runtime that renames its fallback
    moves the gate instead of passing silently -- and the dev-pin's answer
    refusal is what keeps the transcription honest.
    """
    try:
        rows = answer["native_extensions"]
    except (KeyError, TypeError) as exc:
        raise TesseraRouteReceiptError(
            "the contract answer carries no native_extensions table"
        ) from exc
    decoders = set()
    for row in rows:
        behaviours = row.get("when_unavailable") if isinstance(
            row, Mapping) else None
        if not isinstance(behaviours, Mapping):
            continue
        for behaviour in behaviours.values():
            decoder = behaviour.get("decoder") if isinstance(
                behaviour, Mapping) else None
            if isinstance(decoder, str) and decoder:
                decoders.add(decoder)
    return tuple(sorted(decoders))


def parse_route_records(
    records: Any,
    *,
    where: str = "route records",
) -> list[dict[str, Any]]:
    """Fail-closed read of census rows into ``{route, decoder, count}``.

    Each row must carry a non-empty ``route`` and a non-empty ``decoder``;
    ``count`` is carried when present (a non-negative integer) and ``None``
    otherwise.  Extra keys are ignored -- the census schema is Tessera's, and
    this gate reads two fields out of it rather than owning it.
    """
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TesseraRouteReceiptError(
            f"{where} must be a sequence of route rows")
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        at = f"{where}[{index}]"
        if not isinstance(row, Mapping):
            raise TesseraRouteReceiptError(f"{at} must be an object")
        route, decoder = row.get("route"), row.get("decoder")
        if not isinstance(route, str) or not route:
            raise TesseraRouteReceiptError(
                f"{at} carries no route name; a row that cannot say which "
                "route it rode is not evidence for any of them")
        if not isinstance(decoder, str) or not decoder:
            raise TesseraRouteReceiptError(
                f"{at} carries no decoder for route {route!r}; a "
                "decoder-less row read as native is the hole this gate "
                "exists to close")
        count = row.get("count")
        if count is not None and (
                isinstance(count, bool) or not isinstance(count, int)
                or count < 0):
            raise TesseraRouteReceiptError(
                f"{at} carries count {count!r}, not a non-negative integer")
        parsed.append({"route": route, "decoder": decoder, "count": count})
    return parsed


def check_route_receipt(
    *,
    priced_routes: Sequence[str],
    route_records: Sequence[Mapping[str, Any]],
    substitute_decoders: Sequence[str],
) -> dict[str, Any]:
    """Priced-vs-served comparison.  Returns the verdict; never raises it.

    Malformed INPUTS (nothing priced, no known substitute, malformed rows)
    raise: a gate constructed so it cannot detect anything must not return a
    verdict.  A well-formed census that DISAGREES returns
    ``passed: False`` with the disagreement itemised.
    """
    priced = [str(r) for r in (priced_routes or ())]
    if not priced or any(not r for r in priced):
        raise TesseraRouteReceiptError(
            "the artifact priced no routes, so there is nothing to compare "
            "a serve against; name the priced routes explicitly")
    substitutes = [str(d) for d in (substitute_decoders or ())]
    if not substitutes or any(not d for d in substitutes):
        raise TesseraRouteReceiptError(
            "the gate knows no substitute decoder, so every serve would "
            "pass; derive the set from the pinned contract answer "
            "(substitute_decoders_from_contract_answer)")
    records = parse_route_records(list(route_records))

    verdict: dict[str, Any] = {
        "priced_routes": sorted(set(priced)),
        "served_routes": sorted({row["route"] for row in records}),
        "served_decoders": sorted({row["decoder"] for row in records}),
        "substitute_hits": sorted(
            {row["decoder"] for row in records
             if row["decoder"] in substitutes}),
        "unserved_priced": sorted(
            set(priced) - {row["route"] for row in records}),
        "unpriced_served": sorted(
            {row["route"] for row in records} - set(priced)),
        "n_records": len(records),
    }
    reasons: list[str] = []
    if not records:
        reasons.append("the serve emitted no route records")
    if verdict["substitute_hits"]:
        reasons.append(
            "served on substitute decoder(s) "
            f"{verdict['substitute_hits']}: the priced routes were not the "
            "routes that ran")
    if verdict["unserved_priced"]:
        reasons.append(
            f"priced route(s) never served: {verdict['unserved_priced']}")
    if verdict["unpriced_served"]:
        reasons.append(
            f"served route(s) never priced: {verdict['unpriced_served']}")
    verdict["passed"] = not reasons
    verdict["detail"] = ("route census agrees: "
                         f"{verdict['served_routes']} on "
                         f"{verdict['served_decoders']}"
                         if verdict["passed"]
                         else "route census REFUSED: " + "; ".join(reasons))
    return verdict


__all__ = [
    "TesseraRouteReceiptError",
    "check_route_receipt",
    "parse_route_records",
    "substitute_decoders_from_contract_answer",
]
