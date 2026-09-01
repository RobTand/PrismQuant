"""Tessera families in the shape PrismaQuant's rate-distortion allocator wants.

This module is the seam that replaces ``trellis_formats``.  The story is worth
stating because the diff looks bigger than the change is.

``trellis_allocator`` (1768 lines) and ``trellis_rate_surface`` (876 lines) were
written for the Gridbook rate-256 tail-biting trellis, but neither mentions
Gridbook or TCQ even once: they are exact-marginal pricing, Pareto frontiers,
lambda choice, RD hulls, leave-one-anchor-out and allocation regret -- pricing
machinery that happens to have been pointed at a trellis.  Between them they
consume exactly **five** and **seven** names from ``trellis_formats``, and the
load-bearing one is a nine-field frozen dataclass describing a *family*.

So retiring the Gridbook attempt does not mean deleting the pricing.  It means
pointing that seam at Tessera, which is what this module does.  Nothing here
reimplements Tessera: every number is read from the ``tessera`` package, which
is the authority for its own grammar.  If the two ever disagree, the import
fails rather than drifting.

**This is not building Tessera out of Gridbook's internals.**  The direction is
the opposite one.  The pricing machinery is PrismaQuant's, the format authority
is Tessera's, and the Gridbook-specific vocabulary -- ``trellis_formats``,
``trellis_footprint``, ``trellis_menu`` -- is what gets walled off.

Why the currencies already line up
----------------------------------
Both sides parameterise rate as ``q256``: 256ths of a bit per position, kept as
an integer so persisted identities never carry float spelling.  That is not a
coincidence to be grateful for -- both descend from
``docs/design/embedded_native_weight_coding_2026-08-31.md``, and ``tessera``
ships ``Q256_UNIT``, ``root_from_q256`` and ``bresenham_rate_schedule`` for the
same reason this file needs them.  A Tessera root rate is realised as a mixed
per-column schedule, so its rate space is continuous over the realisable roots
rather than a handful of rungs, which is precisely the problem
``trellis_rate_surface`` was built to solve.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import json

__all__ = [
    "MIN_TRELLIS_STEPS",
    "Q256_UNIT",
    "RATE_SURFACE_ADAPTIVE",
    "RATE_SURFACE_ALL_LEGAL",
    "RATE_SURFACE_DENSE",
    "RATE_SURFACE_MODES",
    "SUPERBLOCK_WEIGHTS",
    "TESSERA_FAMILIES",
    "TesseraFamily",
    "TesseraFormatError",
    "TesseraRateSurface",
    "get_tessera_family",
    "parse_tessera_format_name",
    "validate_body_rate_q256",
]


class TesseraFormatError(ValueError):
    """A Tessera family, rung, or rate schedule is invalid."""


try:  # pragma: no cover - exercised by the import-failure test
    from tessera.alphabet import E2M1_GRID, E4M3_GRID, PayloadGrid, tuple_grid
    from tessera.grammar import Q256_UNIT, root_from_q256
except ImportError as exc:  # pragma: no cover
    raise TesseraFormatError(
        "prismaquant.tessera_formats requires the `tessera` package, which is "
        "the authority for Tessera's grammar.  Install it editable from the "
        "Tessera checkout (`pip install -e /path/to/tessera --no-deps`).  This "
        "module deliberately does not carry its own copy of the constants: a "
        "second copy is a drift bug waiting for a rate to change."
    ) from exc


# Tessera's superblock is 256 positions and its scale plane is quoted per
# superblock, so the accounting unit matches the one the allocator already
# uses.  MIN_TRELLIS_STEPS is the shortest run that still carries a trellis;
# below it the body degenerates to a scalar quantiser and should be priced as
# one of the terminal formats instead.
SUPERBLOCK_WEIGHTS = 256
MIN_TRELLIS_STEPS = 8

RATE_SURFACE_ALL_LEGAL = "all_legal"
RATE_SURFACE_DENSE = "dense"
RATE_SURFACE_ADAPTIVE = "adaptive"
RATE_SURFACE_MODES = frozenset({
    RATE_SURFACE_ALL_LEGAL,
    RATE_SURFACE_DENSE,
    RATE_SURFACE_ADAPTIVE,
})

# The scale plane is S6b: one E8M0 base byte per group of 32 plus a 4-bit
# refinement per half of 16.  That is a flat 0.5 bits per position on top of
# whatever the body costs, and it is the reason a 3.0-bit body ships as a
# 3.5 bpp artifact.  Priced here so the allocator never has to know the layout.
SCALE_PLANE_BITS_Q256 = 128


@dataclass(frozen=True, slots=True)
class TesseraFamily:
    """One (grid, arity) pair, and the rungs it can address.

    ``arity`` is the number of weight positions a single code stands for.  It
    is the field that makes the rate ladder finer than the grid alone allows:
    a code space of ``2^payload_bits`` spends ``rate`` bits per *code*, so the
    per-position body rate is ``rate / arity`` and a k=2 family fills the rungs
    between the k=1 ones.  Everything else here is bookkeeping around that.
    """

    family: str
    grid: str
    payload_bits: int
    arity: int
    scale_contract: str
    terminal_format: str
    minimum_capability_sm: int

    @property
    def rate_cap(self) -> int:
        """Largest legal rate per code: ``|A_R| * |D(a)|`` must close at 2^P."""
        return self.payload_bits - 1

    @property
    def mathematical_q256_bounds(self) -> tuple[int, int]:
        """Inclusive q256 bounds on the *body*, before the scale plane.

        A root rate is legal from 1 bit per code up to the cap, and a code
        covers ``arity`` positions, so the per-position bounds are the code
        bounds divided by arity.  Both ends are exact in q256 because
        ``Q256_UNIT`` is a power of two and arity is 1 or 2.
        """
        lo = Fraction(Q256_UNIT, self.arity)
        hi = Fraction(self.rate_cap * Q256_UNIT, self.arity)
        if lo.denominator != 1 or hi.denominator != 1:
            raise TesseraFormatError(
                f"{self.family}: arity {self.arity} does not divide the q256 "
                f"grid exactly; bounds {lo}..{hi} are not integers"
            )
        return (int(lo), int(hi))

    @property
    def artifact_q256_bounds(self) -> tuple[int, int]:
        """The same bounds including the S6b scale plane -- what ships."""
        lo, hi = self.mathematical_q256_bounds
        return (lo + SCALE_PLANE_BITS_Q256, hi + SCALE_PLANE_BITS_Q256)

    def format_name(self, body_rate_q256: int) -> str:
        validate_body_rate_q256(self, body_rate_q256)
        return f"{self.family}_R{body_rate_q256}"

    def root_rate(self, body_rate_q256: int) -> Fraction:
        """Per-position root rate for a rung, as an exact Fraction."""
        validate_body_rate_q256(self, body_rate_q256)
        return root_from_q256(body_rate_q256)

    def payload_grid(self) -> "PayloadGrid":
        """The live Tessera grid object -- built by tessera, never by us."""
        base = {"E2M1": E2M1_GRID, "E4M3": E4M3_GRID}[self.grid]
        return base if self.arity == 1 else tuple_grid(base, self.arity)


# The three families that are both measured and hardware-representable.  E4M3
# at arity 2 is deliberately absent: 65 536 anchors scored per trellis step is
# the same cost that got k=4 refused, and no measurement asked for it.
TESSERA4 = TesseraFamily(
    family="TESSERA4",
    grid="E2M1",
    payload_bits=4,
    arity=1,
    scale_contract="s6b",
    terminal_format="NVFP4",
    minimum_capability_sm=120,
)
TESSERA4_K2 = TesseraFamily(
    family="TESSERA4K2",
    grid="E2M1",
    payload_bits=8,
    arity=2,
    scale_contract="s6b",
    terminal_format="NVFP4",
    minimum_capability_sm=120,
)
TESSERA8 = TesseraFamily(
    family="TESSERA8",
    grid="E4M3",
    payload_bits=8,
    arity=1,
    scale_contract="s6b",
    terminal_format="FP8_E4M3",
    minimum_capability_sm=89,
)

TESSERA_FAMILIES: Mapping[str, TesseraFamily] = {
    f.family: f for f in (TESSERA4, TESSERA4_K2, TESSERA8)
}


def get_tessera_family(family: "str | TesseraFamily") -> TesseraFamily:
    if isinstance(family, TesseraFamily):
        return family
    try:
        return TESSERA_FAMILIES[family]
    except (KeyError, TypeError):
        legal = ", ".join(sorted(TESSERA_FAMILIES))
        raise TesseraFormatError(
            f"unknown Tessera family {family!r}; legal families are {legal}"
        ) from None


def validate_body_rate_q256(
    family: "str | TesseraFamily",
    body_rate_q256: int,
) -> int:
    spec = get_tessera_family(family)
    if type(body_rate_q256) is not int:
        raise TesseraFormatError("body_rate_q256 must be a JSON integer")
    lower, upper = spec.mathematical_q256_bounds
    if not lower <= body_rate_q256 <= upper:
        raise TesseraFormatError(
            f"{spec.family} body_rate_q256 must be in [{lower}, {upper}], "
            f"got {body_rate_q256}"
        )
    return body_rate_q256


def parse_tessera_format_name(name: object) -> "tuple[TesseraFamily, int] | None":
    """Split ``TESSERA4_R768`` into its family and rung, or return None.

    Returns None rather than raising for anything that is not a Tessera format
    name at all, because every caller is asking "is this one of mine?" about a
    menu that also holds NVFP4, FP8 and BF16.  A name that *is* Tessera-shaped
    but names an illegal rung raises, because that is a real error.
    """
    if not isinstance(name, str):
        return None
    head, sep, tail = name.rpartition("_R")
    if not sep or head not in TESSERA_FAMILIES:
        return None
    if not tail.isdigit():
        raise TesseraFormatError(f"{name!r} has a Tessera family but no integer rung")
    spec = TESSERA_FAMILIES[head]
    return (spec, validate_body_rate_q256(spec, int(tail)))


@dataclass(frozen=True, slots=True)
class TesseraRateSurface:
    """A deterministic, allocator-addressable rate surface over one family.

    Same contract the trellis surface had, and for the same reason: a rung is a
    parameter of one family, not a registry entry and not a promise that some
    separately compiled kernel exists for it.  ``adaptive`` surfaces carry the
    identity of the measured RD hull that proposed their rates, so a surface
    can always be traced back to the measurements that justified it.
    """

    family: str
    mode: str
    bounds_q256: tuple[int, ...]
    step_q256: "int | None" = None
    anchor_q256: tuple[int, ...] = ()
    proposed_q256: tuple[int, ...] = ()
    source_identity_sha256: "str | None" = None

    def __post_init__(self) -> None:
        spec = get_tessera_family(self.family)
        if self.mode not in RATE_SURFACE_MODES:
            legal = ", ".join(sorted(RATE_SURFACE_MODES))
            raise TesseraFormatError(
                f"rate-surface mode {self.mode!r} must be one of {legal}"
            )
        for field in ("bounds_q256", "anchor_q256", "proposed_q256"):
            raw = getattr(self, field)
            if isinstance(raw, (str, bytes, bytearray, Mapping)):
                raise TesseraFormatError(
                    f"rate-surface {field} must be an integer sequence"
                )
            if not isinstance(raw, Sequence):
                raise TesseraFormatError(
                    f"rate-surface {field} must be an integer sequence"
                )
            object.__setattr__(self, field, tuple(raw))
        if len(self.bounds_q256) != 2:
            raise TesseraFormatError("rate-surface bounds_q256 must be a (lo, hi) pair")
        lo, hi = self.bounds_q256
        for rung in (lo, hi, *self.anchor_q256, *self.proposed_q256):
            validate_body_rate_q256(spec, rung)
        if lo > hi:
            raise TesseraFormatError(
                f"rate-surface bounds_q256 must be ordered, got ({lo}, {hi})"
            )
        if self.mode == RATE_SURFACE_ADAPTIVE and not self.source_identity_sha256:
            raise TesseraFormatError(
                "an adaptive rate surface must name the measured RD hull that "
                "proposed its rates; an unattributed proposal is a guess"
            )

    def identity(self) -> str:
        """Canonical JSON of the surface, for provenance stamping."""
        return json.dumps(
            {
                "family": self.family,
                "mode": self.mode,
                "bounds_q256": list(self.bounds_q256),
                "step_q256": self.step_q256,
                "anchor_q256": list(self.anchor_q256),
                "proposed_q256": list(self.proposed_q256),
                "source_identity_sha256": self.source_identity_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
