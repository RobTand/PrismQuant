"""Torch-free source of truth for the Gridbook CB serialized layout.

This module owns producer-side facts that used to be repeated by the format
registry, layer-config parser, packer, and exact byte accountant.  Gridbook is
an intentionally independent consumer implementation; cross-repository CI
compares its packaged runtime contract with these values field by field.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


VEC_DIM = 8
SUPERBLOCK = 256
FP4_GROUP = 16
FP4_SCALE_GROUPS_PER_SUPERBLOCK = SUPERBLOCK // FP4_GROUP
CODEWORDS_PER_SUPERBLOCK = SUPERBLOCK // VEC_DIM
INDEX_BYTES_PER_K = CODEWORDS_PER_SUPERBLOCK // 8
INDEX_BIT_ORDER = "lsb_first"
SUBINDEX_SPLIT = "ceil_first"

SCALE_CODING_V1 = "v1"
SCALE_CODING_TWO_TIER = "two_tier"
SCALE_CODINGS = frozenset({SCALE_CODING_V1, SCALE_CODING_TWO_TIER})
LAYOUT_FOR_SCALE_CODING = {
    SCALE_CODING_V1: 1,
    SCALE_CODING_TWO_TIER: 2,
}
SCALE_PLANE_BYTES = {
    ("fp4", SCALE_CODING_V1): 16,
    ("fp4", SCALE_CODING_TWO_TIER): 9,
    ("fp8", SCALE_CODING_V1): 0,
}


@dataclass(frozen=True)
class CBFamily:
    prefix: str
    grid: str
    mode: str
    n_sub: int
    # ``rungs`` is the producer surface: menus, new assignments, and exports
    # may contain only these values. ``accepted_rungs`` is the wider reader
    # surface retained for already-materialized artifacts.  Keeping the two
    # sets on the family itself prevents a permissive parser from accidentally
    # becoming a producer menu.
    rungs: tuple[int, ...]
    accepted_rungs: tuple[int, ...]
    layout_versions: tuple[int, ...]
    moe_layout_versions: tuple[int, ...]

    def name(self, k: int) -> str:
        if int(k) not in self.rungs:
            raise ValueError(f"{self.prefix}{k} is not a producer rung")
        return f"{self.prefix}{int(k)}"

    def accepted_name(self, k: int) -> str:
        """Return a wire name accepted by readers, including legacy rungs."""

        if int(k) not in self.accepted_rungs:
            raise ValueError(f"{self.prefix}{k} is not an accepted rung")
        return f"{self.prefix}{int(k)}"

    def is_producer_rung(self, k: int) -> bool:
        return int(k) in self.rungs


# The unsigned FP4 wire ABI carries one k-bit codeword for every 8 weights.
# The public reader and producer domain stops at K25. K1 has the ceil-first
# split (1, 0); K25 has split (13, 12). Width-14..16 lattice assets and the
# direct uint32 codec endpoint remain available only to explicit low-level
# research code. They have no public format id, registry entry, contract rung,
# chooser candidate, assignment, bundle, or export path.
NVFP4_PRODUCT_RUNGS = tuple(range(1, 26))
NVFP4_ACCEPTED_RUNGS = NVFP4_PRODUCT_RUNGS
FP8_PRODUCT_RUNGS = tuple(range(4, 49, 4))
# Gridbook artifacts produced before the K%4/TMA rule was made a producer
# invariant used every integer rung from K28 through K48.  They remain valid
# serialized inputs and must stay parseable/reportable, but the off-law rungs
# must never re-enter a new producer menu.
FP8_LEGACY_RUNGS = tuple(range(28, 49))
FP8_ACCEPTED_RUNGS = tuple(sorted(set(FP8_PRODUCT_RUNGS) | set(FP8_LEGACY_RUNGS)))

# The signed sign-magnitude family (``NVFP4_CB_S13..S16``, mode="signed",
# n_sub=1) was DELETED on 2026-08-17 (Rob: "we can get entirely rid of the
# signed versions. they are not performant. We don't support them.").
#
# It was already excluded from production allocation by the serving profile's
# format rule -- research-only after losing 78.48% of matched weight-MSE
# comparisons -- so no shipped artifact and no allocation on disk references an
# ``NVFP4_CB_S*`` rung, and removing it changes no assignment.
#
# The durable reason it can never come back without new kernels: Gridbook's
# native FP4 path is written against the UNSIGNED two-tier product layout and
# tests for it exactly -- ``is_fp4 and is_v2 and n_sub == 2 and
# type_size == 4*k + 9`` (gridbook ``linear.py::_require_fp4_v2_product``).
# A signed n_sub=1 rung has decode support in older kernels but "no native
# quality-preserving dense prefill kernel", and it trips the v2 GEMV's own
# ``cb_elems`` check (gridbook ``moe_gemv_select.py:307``). Serving it would
# mean one layer changing numeric implementation with batch size.
FAMILIES = (
    CBFamily(
        prefix="NVFP4_CB_K",
        grid="fp4",
        mode="product",
        n_sub=2,
        rungs=NVFP4_PRODUCT_RUNGS,
        accepted_rungs=NVFP4_ACCEPTED_RUNGS,
        layout_versions=(1, 2),
        moe_layout_versions=(2,),
    ),
    CBFamily(
        prefix="FP8_CB_K",
        grid="fp8",
        mode="product",
        n_sub=4,
        rungs=FP8_PRODUCT_RUNGS,
        accepted_rungs=FP8_ACCEPTED_RUNGS,
        layout_versions=(1,),
        moe_layout_versions=(1,),
    ),
)
FAMILY_BY_PREFIX = {family.prefix: family for family in FAMILIES}
FAMILY_BY_GRID_MODE = {
    (family.grid, family.mode): family for family in FAMILIES
}
CB_FORMATS = tuple(
    family.name(k) for family in FAMILIES for k in family.rungs
)
CB_FORMAT_NAMES = frozenset(CB_FORMATS)
ACCEPTED_CB_FORMATS = tuple(
    family.accepted_name(k)
    for family in FAMILIES
    for k in family.accepted_rungs
)
ACCEPTED_CB_FORMAT_NAMES = frozenset(ACCEPTED_CB_FORMATS)
PRODUCT_CB_FORMATS = tuple(
    family.name(k)
    for family in FAMILIES
    if family.mode == "product"
    for k in family.rungs
)
PRODUCT_CB_FORMAT_NAMES = frozenset(PRODUCT_CB_FORMATS)
# Per-GRID rung sets. Gridbook's load gates are grid-specific — the fp4
# families share one N-dimension packing (`out_features % 8`) and the fp8
# family another (`out_features % 16`) — so a serving profile that has to
# name "every fp4-CB rung" must derive it from the family table rather than
# hand-list 17 names that drift the next time a rung is added.
NVFP4_CB_FORMAT_NAMES = frozenset(
    family.name(k)
    for family in FAMILIES
    if family.grid == "fp4"
    for k in family.rungs
)
FP8_CB_FORMAT_NAMES = frozenset(
    family.name(k)
    for family in FAMILIES
    if family.grid == "fp8"
    for k in family.rungs
)
FP8_ACCEPTED_FORMAT_NAMES = frozenset(
    family.accepted_name(k)
    for family in FAMILIES
    if family.grid == "fp8"
    for k in family.accepted_rungs
)
_FORMAT_RE = re.compile(r"^(NVFP4_CB_K|FP8_CB_K)(\d+)$")


def bit_split(k: int, n_sub: int) -> tuple[int, ...]:
    """Split index bits evenly across subtables, larger partitions first."""

    k = int(k)
    n_sub = int(n_sub)
    if k <= 0 or n_sub <= 0:
        raise ValueError(f"k and n_sub must be positive, got {k}, {n_sub}")
    base, extra = divmod(k, n_sub)
    return tuple(base + (1 if index < extra else 0)
                 for index in range(n_sub))


def family_for(grid: str, mode: str) -> CBFamily:
    """Return the one producer family for a serialized grid/mode pair."""

    key = (str(grid).lower(), str(mode).lower())
    try:
        return FAMILY_BY_GRID_MODE[key]
    except KeyError as exc:
        raise ValueError(f"unknown CB grid/mode {key!r}") from exc


def subtable_bit_widths(
    k: int,
    mode: str,
    n_sub: int,
) -> tuple[int, ...]:
    """Index bits represented by each serialized codebook subtable.

    Product families split all ``k`` bits ceil-first. Full mode has one table
    representing all bits.

    ``signed`` is refused rather than merely absent: it was a real serialized
    mode until 2026-08-17, so a stale caller passing it must fail loudly
    instead of falling through to the product split, which would silently
    produce a different subtable geometry for the same name.
    """

    k = int(k)
    mode = str(mode).lower()
    n_sub = int(n_sub)
    if mode == "signed":
        raise ValueError(
            "signed CB mode was deleted 2026-08-17: Gridbook's native FP4 path "
            "requires the unsigned two-tier product layout (n_sub=2, "
            "type_size=4*k+9) and has no quality-preserving prefill kernel for "
            "n_sub=1. Use an NVFP4_CB_K* product rung."
        )
    if mode == "full":
        if n_sub != 1:
            raise ValueError(f"full CB requires n_sub=1, got {n_sub}")
        return (k,)
    if mode != "product":
        raise ValueError(f"unknown CB mode {mode!r}")
    return bit_split(k, n_sub)


def scale_coding_layout_version(scale_coding: str) -> int:
    try:
        return LAYOUT_FOR_SCALE_CODING[str(scale_coding)]
    except KeyError as exc:
        raise ValueError(f"unknown CB scale coding {scale_coding!r}") from exc


def type_size(
    k: int,
    grid: str,
    scale_coding: str = SCALE_CODING_V1,
) -> int:
    """Serialized bytes per 256-weight superblock."""

    grid = str(grid)
    coding = SCALE_CODING_V1 if grid == "fp8" else str(scale_coding)
    try:
        scale_bytes = SCALE_PLANE_BYTES[(grid, coding)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported CB grid/scale coding {(grid, coding)!r}"
        ) from exc
    return INDEX_BYTES_PER_K * int(k) + scale_bytes


def parse_format_name(name: str) -> tuple[CBFamily, int] | None:
    """Parse any supported CB wire name, including historical reader rungs."""

    match = _FORMAT_RE.fullmatch(str(name).upper())
    if match is None:
        return None
    family = FAMILY_BY_PREFIX[match.group(1)]
    k = int(match.group(2))
    if k not in family.accepted_rungs:
        return None
    return family, k


def parse_producer_format_name(name: str) -> tuple[CBFamily, int] | None:
    """Parse a format only when it is eligible for a newly produced artifact."""

    parsed = parse_format_name(name)
    if parsed is None:
        return None
    family, k = parsed
    return parsed if family.is_producer_rung(k) else None


def is_producer_format_name(name: str) -> bool:
    """Whether ``name`` belongs in a producer menu or new assignment."""

    return parse_producer_format_name(name) is not None


def codebook_subtable_shapes(
    k: int,
    mode: str,
    n_sub: int,
) -> tuple[tuple[int, int], ...]:
    """Exact FP16 sidecar subtable shapes for one CB format."""

    widths = subtable_bit_widths(k, mode, n_sub)
    if mode in {"signed", "full"}:
        return ((1 << widths[0], VEC_DIM),)
    if VEC_DIM % int(n_sub):
        raise ValueError(f"unsupported CB mode/n_sub {(mode, n_sub)!r}")
    sub_dim = VEC_DIM // int(n_sub)
    return tuple((1 << width, sub_dim)
                 for width in widths)


def product_format_menu(*additional_formats: str) -> str:
    """Canonical ordered product-CB menu plus explicit policy suffixes."""

    return ",".join((*PRODUCT_CB_FORMATS,
                     *(str(name) for name in additional_formats)))


__all__ = [
    "ACCEPTED_CB_FORMATS",
    "ACCEPTED_CB_FORMAT_NAMES",
    "CBFamily",
    "CB_FORMATS",
    "CB_FORMAT_NAMES",
    "CODEWORDS_PER_SUPERBLOCK",
    "FAMILIES",
    "FAMILY_BY_GRID_MODE",
    "FAMILY_BY_PREFIX",
    "FP4_GROUP",
    "FP4_SCALE_GROUPS_PER_SUPERBLOCK",
    "FP8_PRODUCT_RUNGS",
    "FP8_ACCEPTED_RUNGS",
    "FP8_ACCEPTED_FORMAT_NAMES",
    "FP8_LEGACY_RUNGS",
    "INDEX_BIT_ORDER",
    "INDEX_BYTES_PER_K",
    "LAYOUT_FOR_SCALE_CODING",
    "NVFP4_ACCEPTED_RUNGS",
    "NVFP4_PRODUCT_RUNGS",
    "PRODUCT_CB_FORMAT_NAMES",
    "PRODUCT_CB_FORMATS",
    "SCALE_CODINGS",
    "SCALE_CODING_TWO_TIER",
    "SCALE_CODING_V1",
    "SCALE_PLANE_BYTES",
    "SUBINDEX_SPLIT",
    "SUPERBLOCK",
    "VEC_DIM",
    "bit_split",
    "codebook_subtable_shapes",
    "family_for",
    "parse_format_name",
    "parse_producer_format_name",
    "is_producer_format_name",
    "product_format_menu",
    "scale_coding_layout_version",
    "subtable_bit_widths",
    "type_size",
]
