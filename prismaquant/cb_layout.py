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
    rungs: tuple[int, ...]
    layout_versions: tuple[int, ...]
    moe_layout_versions: tuple[int, ...]
    source: str | None = "lattice"

    def name(self, k: int) -> str:
        if int(k) not in self.rungs:
            raise ValueError(f"{self.prefix}{k} is not a producer rung")
        return f"{self.prefix}{int(k)}"


NVFP4_PRODUCT_RUNGS = tuple(range(12, 25))
NVFP4_SIGNED_RUNGS = (13, 14, 15, 16)
FP8_PRODUCT_RUNGS = tuple(range(28, 49))
FP8_CBL_PRODUCT_RUNGS = (28, 32, 36, 40, 44)

FAMILIES = (
    CBFamily(
        prefix="NVFP4_CB_K",
        grid="fp4",
        mode="product",
        n_sub=2,
        rungs=NVFP4_PRODUCT_RUNGS,
        layout_versions=(1, 2),
        moe_layout_versions=(2,),
    ),
    CBFamily(
        prefix="NVFP4_CB_S",
        grid="fp4",
        mode="signed",
        n_sub=1,
        rungs=NVFP4_SIGNED_RUNGS,
        layout_versions=(1, 2),
        moe_layout_versions=(2,),
    ),
    CBFamily(
        prefix="FP8_CB_K",
        grid="fp8",
        mode="product",
        n_sub=4,
        rungs=FP8_PRODUCT_RUNGS,
        layout_versions=(1,),
        moe_layout_versions=(1,),
    ),
    CBFamily(
        prefix="FP8_CBL_K",
        grid="fp8",
        mode="product",
        n_sub=4,
        rungs=FP8_CBL_PRODUCT_RUNGS,
        layout_versions=(1,),
        moe_layout_versions=(1,),
        source="learned",
    ),
)
FAMILY_BY_PREFIX = {family.prefix: family for family in FAMILIES}
# ``family_for(grid, mode)`` predates source-bearing names and continues to
# mean the canonical lattice geometry.  Source-bearing callers already have a
# parsed family and must use it directly; silently overwriting this mapping
# with FP8_CBL would reinterpret every historical FP8_CB call site.
FAMILY_BY_GRID_MODE = {
    (family.grid, family.mode): family
    for family in FAMILIES
    if family.source == "lattice"
}
FAMILY_BY_GRID_MODE_SOURCE = {
    (family.grid, family.mode, family.source): family for family in FAMILIES
}
CB_FORMATS = tuple(
    family.name(k) for family in FAMILIES for k in family.rungs
)
CB_FORMAT_NAMES = frozenset(CB_FORMATS)
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
_FORMAT_RE = re.compile(
    rf"^({'|'.join(re.escape(prefix) for prefix in FAMILY_BY_PREFIX)})(\d+)$"
)


def bit_split(k: int, n_sub: int) -> tuple[int, ...]:
    """Split index bits evenly across subtables, larger partitions first."""

    k = int(k)
    n_sub = int(n_sub)
    if k <= 0 or n_sub <= 0:
        raise ValueError(f"k and n_sub must be positive, got {k}, {n_sub}")
    base, extra = divmod(k, n_sub)
    return tuple(base + (1 if index < extra else 0)
                 for index in range(n_sub))


def family_for(
    grid: str,
    mode: str,
    source: str | None = "lattice",
) -> CBFamily:
    """Return the producer family for a serialized grid/mode/source tuple.

    The two-argument spelling remains the historical lattice lookup.  A
    source-bearing format should normally be parsed by name instead, which
    avoids erasing the distinction between FP8_CB and FP8_CBL.
    """

    key = (
        str(grid).lower(),
        str(mode).lower(),
        None if source is None else str(source).lower(),
    )
    try:
        return FAMILY_BY_GRID_MODE_SOURCE[key]
    except KeyError as exc:
        raise ValueError(f"unknown CB grid/mode/source {key!r}") from exc


def subtable_bit_widths(
    k: int,
    mode: str,
    n_sub: int,
) -> tuple[int, ...]:
    """Index bits represented by each serialized codebook subtable.

    Product families split all ``k`` bits ceil-first. Signed families spend
    the low ``VEC_DIM`` bits on signs, so their sole magnitude table represents
    only ``k - VEC_DIM`` bits. Full mode has one table representing all bits.
    """

    k = int(k)
    mode = str(mode).lower()
    n_sub = int(n_sub)
    if mode == "signed":
        if n_sub != 1 or k <= VEC_DIM:
            raise ValueError(
                f"signed CB requires n_sub=1 and k > {VEC_DIM}, got "
                f"{(k, n_sub)!r}"
            )
        return (k - VEC_DIM,)
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
    match = _FORMAT_RE.fullmatch(str(name).upper())
    if match is None:
        return None
    family = FAMILY_BY_PREFIX[match.group(1)]
    k = int(match.group(2))
    if k not in family.rungs:
        return None
    return family, k


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
    "CBFamily",
    "CB_FORMATS",
    "CB_FORMAT_NAMES",
    "CODEWORDS_PER_SUPERBLOCK",
    "FAMILIES",
    "FAMILY_BY_GRID_MODE",
    "FAMILY_BY_GRID_MODE_SOURCE",
    "FAMILY_BY_PREFIX",
    "FP4_GROUP",
    "FP4_SCALE_GROUPS_PER_SUPERBLOCK",
    "FP8_CBL_PRODUCT_RUNGS",
    "FP8_PRODUCT_RUNGS",
    "INDEX_BIT_ORDER",
    "INDEX_BYTES_PER_K",
    "LAYOUT_FOR_SCALE_CODING",
    "NVFP4_PRODUCT_RUNGS",
    "NVFP4_SIGNED_RUNGS",
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
    "product_format_menu",
    "scale_coding_layout_version",
    "subtable_bit_widths",
    "type_size",
]
