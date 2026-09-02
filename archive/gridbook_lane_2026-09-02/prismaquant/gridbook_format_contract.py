"""Torch-free consumer for Gridbook's versioned CB format declarations.

Gridbook owns the runtime and its packaged ``runtime_contract.json``.  This
module deliberately accepts a decoded contract mapping instead of importing
Gridbook or reading an unpinned installation.  The legacy v4 contract remains a
reader-compatibility input; only v11's explicit ``formats[].producer_rungs``
field can attest a new producer ladder.

DELIBERATELY NOT EXTENDED TO v12 (2026-08-30, pin advance 0.8.11 -> 0.9.1).
The pinned contract is now v12 and it *does* carry ``producer_rungs``, so
accepting it here is one line -- and that one line is a producer-policy change,
not a version bump.  0.9.1 publishes ``FP8_CB_K.producer_rungs = [40, 44, 48]``
and ``NVFP4_CB_K.producer_rungs = [12..24]``.  Binding this reader to v12 would
therefore make ``FP8_CB_K28``/``K32`` -- the rungs the shipped DSv4 routed
experts are built on -- no longer producer-attested, and would trip the
``declared.producer_rungs != local.rungs`` equality in
``validate_gridbook_cb_rung_contract``.  That is Gridbook's own narrowing and
it may well be right, but it is a decision about what PrismaQuant may BUILD; it
does not belong inside a pin commit, and no production caller feeds this module
the materialized pin today (the sole caller,
``rtx4090_qwen38_policy``, passes a contract of its own).  A v12 contract
handed to this reader raises rather than silently re-scoping the ladder --
which is the correct failure -- and closing that gap is its own change, with
its own decision about the DSv4 rungs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any

from .cb_layout import (
    FAMILIES,
    is_producer_format_name,
    parse_format_name,
)


GRIDBOOK_PRODUCER_RUNGS_CONTRACT_SCHEMA = "gridbook.runtime-contract.v11"
_CONTRACT_SCHEMA_RE = re.compile(r"gridbook[.]runtime-contract[.]v([0-9]+)")


class GridbookFormatContractError(ValueError):
    """A Gridbook format declaration is malformed or incompatible."""


@dataclass(frozen=True)
class GridbookFormatRungs:
    family: str
    accepted_rungs: tuple[int, ...]
    producer_rungs: tuple[int, ...] | None
    producer_rungs_attested: bool


def _schema_version(contract: Mapping[str, Any], *, where: str) -> int:
    schema = contract.get("schema")
    match = _CONTRACT_SCHEMA_RE.fullmatch(str(schema))
    if match is None:
        raise GridbookFormatContractError(
            f"{where}: unsupported Gridbook runtime contract schema {schema!r}"
        )
    version = int(match.group(1))
    if version not in {4, 11}:
        raise GridbookFormatContractError(
            f"{where}: unsupported Gridbook runtime contract version {version}; "
            "the accepted schemas are exactly v4 (legacy reader) and v11 "
            "(producer-rung attestation)"
        )
    contract_version = contract.get("contract_version")
    if type(contract_version) is not int or contract_version != version:
        raise GridbookFormatContractError(
            f"{where}: schema names v{version} but contract_version is "
            f"{contract_version!r}; the two version fields must move together"
        )
    return version


def _strict_rungs(value: Any, *, where: str) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise GridbookFormatContractError(
            f"{where}: rungs must be a nonempty JSON array"
        )
    result = tuple(value)
    if not result:
        raise GridbookFormatContractError(f"{where}: rungs cannot be empty")
    if any(type(item) is not int or item <= 0 for item in result):
        raise GridbookFormatContractError(
            f"{where}: rungs must contain only positive JSON integers"
        )
    if result != tuple(sorted(set(result))):
        raise GridbookFormatContractError(
            f"{where}: rungs must be sorted and unique, got {list(result)}"
        )
    return result


def gridbook_format_rungs(
    contract: Mapping[str, Any],
    family: str,
    *,
    where: str = "Gridbook runtime contract",
) -> GridbookFormatRungs:
    """Read one family without treating legacy reader rungs as producers."""

    if not isinstance(contract, Mapping):
        raise GridbookFormatContractError(f"{where}: contract must be an object")
    version = _schema_version(contract, where=where)
    raw_formats = contract.get("formats")
    if (
        not isinstance(raw_formats, Sequence)
        or isinstance(raw_formats, (str, bytes, bytearray))
    ):
        raise GridbookFormatContractError(
            f"{where}.formats must be a JSON array"
        )
    entries: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_formats):
        if not isinstance(raw, Mapping):
            raise GridbookFormatContractError(
                f"{where}.formats[{index}] must be an object"
            )
        declared = raw.get("family")
        if not isinstance(declared, str) or not declared:
            raise GridbookFormatContractError(
                f"{where}.formats[{index}].family must be a nonempty string"
            )
        if declared in entries:
            raise GridbookFormatContractError(
                f"{where}.formats repeats family {declared!r}"
            )
        entries[declared] = raw

    try:
        entry = entries[str(family)]
    except KeyError as exc:
        raise GridbookFormatContractError(
            f"{where}.formats has no {family!r} family"
        ) from exc
    accepted = _strict_rungs(
        entry.get("rungs"), where=f"{where}.formats[{family!r}].rungs"
    )
    has_producer = "producer_rungs" in entry
    if version == 11 and not has_producer:
        raise GridbookFormatContractError(
            f"{where}.formats[{family!r}].producer_rungs is required by v11"
        )
    if version < 11 and has_producer:
        raise GridbookFormatContractError(
            f"{where}.formats[{family!r}].producer_rungs is not attested by "
            f"the v{version} schema"
        )
    producer = None
    if has_producer:
        producer = _strict_rungs(
            entry["producer_rungs"],
            where=f"{where}.formats[{family!r}].producer_rungs",
        )
        extra = sorted(set(producer) - set(accepted))
        if extra:
            raise GridbookFormatContractError(
                f"{where}.formats[{family!r}].producer_rungs contains values "
                f"outside its reader rungs: {extra}"
            )
    return GridbookFormatRungs(
        family=str(family),
        accepted_rungs=accepted,
        producer_rungs=producer,
        producer_rungs_attested=version == 11 and producer is not None,
    )


def validate_gridbook_cb_rung_contract(
    contract: Mapping[str, Any],
    *,
    require_producer_attestation: bool = False,
    where: str = "Gridbook runtime contract",
) -> dict[str, GridbookFormatRungs]:
    """Validate Gridbook's reader/producer rung boundary against PrismaQuant.

    Pre-v11 contracts are accepted only as historical readers, and their rung
    domains may be strict subsets of today's accepted domain.  A v11 contract
    must match both local sets exactly before it can attest production.
    """

    version = _schema_version(contract, where=where)
    result: dict[str, GridbookFormatRungs] = {}
    raw_formats = contract.get("formats")
    if (
        not isinstance(raw_formats, Sequence)
        or isinstance(raw_formats, (str, bytes, bytearray))
    ):
        raise GridbookFormatContractError(f"{where}.formats must be a JSON array")
    # Version the field for the whole format table, including compatibility
    # families PrismaQuant no longer produces (for example legacy signed FP4).
    # Otherwise a partially-upgraded v11 contract could attest the two local
    # families while leaving another entry's producer semantics implicit.
    for index, entry in enumerate(raw_formats):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("family"), str):
            raise GridbookFormatContractError(
                f"{where}.formats[{index}] must name a family"
            )
        gridbook_format_rungs(contract, str(entry["family"]), where=where)
    for local in FAMILIES:
        family_name = local.prefix.removesuffix("_")
        declared = gridbook_format_rungs(
            contract, family_name, where=where
        )
        local_accepted = set(local.accepted_rungs)
        unexpected = sorted(set(declared.accepted_rungs) - local_accepted)
        if unexpected:
            raise GridbookFormatContractError(
                f"{where} {family_name} reader rungs are unknown to "
                f"PrismaQuant: {unexpected}"
            )
        if version == 11:
            if declared.accepted_rungs != local.accepted_rungs:
                raise GridbookFormatContractError(
                    f"{where} {family_name} reader rungs differ from the "
                    "PrismaQuant accepted wire domain"
                )
            if declared.producer_rungs != local.rungs:
                raise GridbookFormatContractError(
                    f"{where} {family_name} producer rungs differ from the "
                    "PrismaQuant producer ladder"
                )
        elif require_producer_attestation:
            raise GridbookFormatContractError(
                f"{where}: producer-rung attestation requires "
                f"{GRIDBOOK_PRODUCER_RUNGS_CONTRACT_SCHEMA}, got v{version}"
            )
        result[family_name] = declared
    return result


def gridbook_contract_attests_producer_format(
    contract: Mapping[str, Any],
    format_name: str,
    *,
    where: str = "Gridbook runtime contract",
) -> bool:
    """Return true only for a local producer rung explicitly attested by v11."""

    parsed = parse_format_name(format_name)
    if parsed is None or not is_producer_format_name(format_name):
        return False
    family, k = parsed
    declarations = validate_gridbook_cb_rung_contract(
        contract,
        require_producer_attestation=True,
        where=where,
    )
    declared = declarations[family.prefix.removesuffix("_")]
    return declared.producer_rungs is not None and k in declared.producer_rungs


__all__ = [
    "GRIDBOOK_PRODUCER_RUNGS_CONTRACT_SCHEMA",
    "GridbookFormatContractError",
    "GridbookFormatRungs",
    "gridbook_contract_attests_producer_format",
    "gridbook_format_rungs",
    "validate_gridbook_cb_rung_contract",
]
