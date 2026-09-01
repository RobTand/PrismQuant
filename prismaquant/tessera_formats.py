"""Tessera's grid-space, in the shape PrismaQuant's rate-distortion allocator wants.

This module is the seam that replaces ``trellis_formats``.  The story is worth
stating because the diff looks bigger than the change is.

``trellis_allocator`` (1768 lines) and ``trellis_rate_surface`` (876) were
written for the Gridbook rate-256 tail-biting trellis, but neither file mentions
Gridbook or TCQ even once: they are exact-marginal pricing, Pareto frontiers,
lambda choice, RD hulls, leave-one-anchor-out and allocation regret -- pricing
machinery that happens to have been pointed at a trellis.  Between them they
consume exactly **five** and **seven** names from ``trellis_formats``, and the
load-bearing one is a frozen dataclass describing a *family*.

So retiring the Gridbook attempt does not mean deleting the pricing.  It means
pointing that seam at Tessera.

**This is not building Tessera out of Gridbook's internals.**  The direction is
the opposite one: the pricing machinery is PrismaQuant's, the format authority
is Tessera's, and the Gridbook-specific vocabulary is what gets walled off.
Nothing here reimplements Tessera -- every number is read from the ``tessera``
package, and a second copy of a rate constant is a drift bug waiting for a rate
to change.

The space, and why it is continuous
-----------------------------------
A family is a **(base grid, arity)** pair.  A base grid of ``G`` codes at arity
``k`` gives a code space of ``G**k``, so ``payload_bits = k*log2(G)`` and the
rate cap is ``payload_bits - 1`` (``|A_R| * |D(a)| = 2^(R+1) * 2^(cap-R)`` has
to close at ``2^payload_bits`` exactly).  A code covers ``k`` positions, so the
per-position body rate is ``rate/k`` -- which is how ``k=2`` fills the rungs
between the ``k=1`` ones, and how 4.0 bpp becomes addressable at all.

Within a family the rate axis is **continuous at a 1/256-bpp quantum**, not a
handful of rungs.  A root rate is realised as a mixed per-column Bresenham
schedule, and a root is realisable over ``n`` columns exactly when
``(root - floor(root)) * n`` is an integer.  Rates are quoted in q256 -- 256ths
of a bit -- and the superblock is 256 columns, so that product is
``root_q256 mod 256``, an integer by construction.  **The q256 grid is exactly
the realisable set at superblock scale.**  Verified exhaustively: ~9500 rungs
across ten families spanning 1.00 to 8.00 bpp, zero unrealisable
(``tessera/tests/test_grid_space_continuity.py``).

Realisability is not quality.  This module says which rungs can be *encoded*;
which are worth encoding is a measurement, and only four of them have one.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import json
import re

__all__ = [
    "ANCHOR_BUDGET_BITS",
    "MIN_TRELLIS_STEPS",
    "Q256_UNIT",
    "RATE_SURFACE_ADAPTIVE",
    "RATE_SURFACE_ALL_LEGAL",
    "RATE_SURFACE_DENSE",
    "RATE_SURFACE_MODES",
    "SCALE_PLANE_BITS_Q256",
    "SUPERBLOCK_WEIGHTS",
    "TesseraFamily",
    "TesseraFormatError",
    "TesseraRateSurface",
    "artifact_bpp",
    "enumerate_grid_space",
    "get_tessera_family",
    "parse_tessera_format_name",
    "realisable_rungs",
    "tessera_family",
    "validate_body_rate_q256",
]


class TesseraFormatError(ValueError):
    """A Tessera family, rung, or rate schedule is invalid."""


try:  # pragma: no cover - exercised by the import-failure path
    from tessera.alphabet import E2M1_GRID, E4M3_GRID, lloyd_max_grid, tuple_grid
    from tessera.grammar import (
        Q256_UNIT,
        bresenham_rate_schedule,
        root_from_q256,
        superblock_quota_ok,
    )
except ImportError as exc:  # pragma: no cover
    raise TesseraFormatError(
        "prismaquant.tessera_formats requires the `tessera` package, which is "
        "the authority for Tessera's grammar.  Install it editable from the "
        "Tessera checkout (`pip install -e /path/to/tessera --no-deps`).  This "
        "module deliberately carries no copy of the constants."
    ) from exc


#: Tessera's superblock, in positions.  The scale plane is quoted per
#: superblock and the rate quota is kept per superblock, so this is the unit
#: the allocator already accounts in.
SUPERBLOCK_WEIGHTS = 256

#: Shortest run that still carries a trellis.  Below it the body degenerates to
#: a scalar quantiser and should be priced as a terminal format instead.
MIN_TRELLIS_STEPS = 8

#: The S6b scale plane: an E8M0 base byte per group of 32 plus a 4-bit
#: refinement per half of 16.  A flat 0.5 bits per position on top of the body,
#: which is why a 3.0-bit body ships as a 3.5 bpp artifact.  Priced here so the
#: allocator never has to know the layout.
SCALE_PLANE_BITS_Q256 = Q256_UNIT // 2

#: Encoding scores ``2**payload_bits`` anchors at every trellis step, so the
#: code space is a *cost*, not just an addressing choice.  65 536 anchors per
#: step is the level the encoder already refuses -- it is why k=4 over E2M1 is
#: not offered -- so a family that reaches it is refused here too, rather than
#: being handed to a DP that would cheerfully select something nothing can
#: encode.  E4M3 at arity 2 lands exactly on this wall and is excluded by it,
#: which is the intended reading and not an off-by-one.
ANCHOR_BUDGET_BITS = 16

RATE_SURFACE_ALL_LEGAL = "all_legal"
RATE_SURFACE_DENSE = "dense"
RATE_SURFACE_ADAPTIVE = "adaptive"
RATE_SURFACE_MODES = frozenset({
    RATE_SURFACE_ALL_LEGAL,
    RATE_SURFACE_DENSE,
    RATE_SURFACE_ADAPTIVE,
})

#: Hardware grids materialise into a stock format at load, so an artifact over
#: them serves on a runtime that has never heard of Tessera.  Free (Lloyd-Max)
#: grids do not: `lloyd_max_grid` sets ``native=None`` precisely to say so.
_HARDWARE_BASES: Mapping[str, tuple[int, str, int]] = {
    # base -> (size, terminal format it materialises into, minimum SM)
    "E2M1": (16, "NVFP4", 120),
    "E4M3": (256, "FP8_E4M3", 89),
}
_FREE_BASE = re.compile(r"^LM(\d+)$")
_FORMAT_NAME = re.compile(r"^TESSERA_([A-Z0-9]+)_K(\d+)_R(\d+)$")

LANE_STOCK = "stock"
LANE_KERNEL = "kernel"


@dataclass(frozen=True, slots=True)
class TesseraFamily:
    """One (base grid, arity) pair, and the rungs it can address."""

    base: str
    base_size: int
    arity: int

    def __post_init__(self) -> None:
        if self.arity < 1:
            raise TesseraFormatError(f"arity must be >= 1, got {self.arity}")
        if self.base_size < 2 or self.base_size & (self.base_size - 1):
            raise TesseraFormatError(
                f"base grid size must be a power of two >= 2, got {self.base_size}"
            )
        if self.payload_bits >= ANCHOR_BUDGET_BITS:
            raise TesseraFormatError(
                f"{self.name} needs {1 << self.payload_bits} anchors scored per "
                f"trellis step, at or above the {1 << ANCHOR_BUDGET_BITS} wall "
                f"the encoder already refuses. "
                "This is a cost refusal, not a grammar one: the rungs are "
                "legal, nothing can afford to encode them."
            )

    @property
    def name(self) -> str:
        return f"TESSERA_{self.base}_K{self.arity}"

    @property
    def family(self) -> str:
        """Alias for :attr:`name`.

        The RD allocator identifies a family by ``spec.family``; keeping the
        alias means retargeting it is an import swap rather than a rewrite,
        which is the whole point of the seam being three fields wide.
        """
        return self.name

    @property
    def payload_bits(self) -> int:
        """Width of the code space: ``arity * log2(base_size)``."""
        return (self.base_size.bit_length() - 1) * self.arity

    @property
    def rate_cap(self) -> int:
        """Largest legal rate per code: ``|A_R| * |D(a)|`` closes at 2^P."""
        return self.payload_bits - 1

    @property
    def lane(self) -> str:
        """``stock`` if it materialises into a hardware format, else ``kernel``.

        Materialisability is a property of the *base* grid, not the tuple: a
        k=2 code over E2M1 decodes to two E2M1 nibbles, so it materialises into
        an ordinary NVFP4 tensor exactly as the scalar one does.  (``tessera``'s
        ``tuple_grid`` drops ``native``, which understates this; the base is the
        honest place to ask.)
        """
        return LANE_STOCK if self.base in _HARDWARE_BASES else LANE_KERNEL

    @property
    def terminal_format(self) -> "str | None":
        spec = _HARDWARE_BASES.get(self.base)
        return None if spec is None else spec[1]

    @property
    def minimum_capability_sm(self) -> "int | None":
        spec = _HARDWARE_BASES.get(self.base)
        return None if spec is None else spec[2]

    @property
    def mathematical_q256_bounds(self) -> tuple[int, int]:
        """Inclusive per-position q256 bounds on the *body*, before the scales.

        A rate is legal from 1 bit per code up to the cap, and a code covers
        ``arity`` positions.  Both ends are exact because ``Q256_UNIT`` is a
        power of two; a non-dividing arity is refused rather than rounded.
        """
        lo = Fraction(Q256_UNIT, self.arity)
        hi = Fraction(self.rate_cap * Q256_UNIT, self.arity)
        if lo.denominator != 1 or hi.denominator != 1:
            raise TesseraFormatError(
                f"{self.name}: arity {self.arity} does not divide the q256 grid "
                f"exactly; bounds {lo}..{hi} are not integers"
            )
        return (int(lo), int(hi))

    @property
    def artifact_q256_bounds(self) -> tuple[int, int]:
        """The same bounds including the S6b scale plane -- what ships."""
        lo, hi = self.mathematical_q256_bounds
        return (lo + SCALE_PLANE_BITS_Q256, hi + SCALE_PLANE_BITS_Q256)

    def format_name(self, body_rate_q256: int) -> str:
        validate_body_rate_q256(self, body_rate_q256)
        return f"{self.name}_R{body_rate_q256}"

    def root_rate(self, body_rate_q256: int) -> Fraction:
        """Per-*code* root rate for a rung, which is what a schedule quotes."""
        validate_body_rate_q256(self, body_rate_q256)
        return root_from_q256(body_rate_q256 * self.arity)

    def column_schedule(
        self, body_rate_q256: int, n_columns: int = SUPERBLOCK_WEIGHTS
    ) -> tuple[int, ...]:
        """The canonical per-column rates realising a rung.  Raises if it cannot.

        This is the function that makes "continuous" checkable rather than
        asserted: an allocator that selects a rung is selecting something the
        encoder can be handed directly.
        """
        root = self.root_rate(body_rate_q256)
        return bresenham_rate_schedule(root, n_columns, self.rate_cap)

    def payload_grid(self):
        """The live Tessera grid object -- built by tessera, never by us."""
        return _build_grid(self.base, self.base_size, self.arity)


@lru_cache(maxsize=64)
def _build_grid(base: str, base_size: int, arity: int):
    if base in _HARDWARE_BASES:
        scalar = {"E2M1": E2M1_GRID, "E4M3": E4M3_GRID}[base]
    else:
        scalar = lloyd_max_grid(base_size)
    return scalar if arity == 1 else tuple_grid(scalar, arity)


@lru_cache(maxsize=256)
def tessera_family(base: str, arity: int = 1) -> TesseraFamily:
    """Build a family from a base-grid name and an arity.

    ``base`` is ``E2M1``, ``E4M3``, or ``LM<n>`` for a free Lloyd-Max grid of
    ``n`` levels.  Free grids are kernel-lane only, which is a fact about the
    grid and not a policy choice: they materialise into no hardware format.
    """
    hardware = _HARDWARE_BASES.get(base)
    if hardware is not None:
        size = hardware[0]
    else:
        match = _FREE_BASE.match(base or "")
        if match is None:
            legal = ", ".join(sorted(_HARDWARE_BASES)) + ", or LM<n>"
            raise TesseraFormatError(
                f"unknown Tessera base grid {base!r}; legal bases are {legal}"
            )
        size = int(match.group(1))
    return TesseraFamily(base=base, base_size=size, arity=arity)


def get_tessera_family(family: "str | TesseraFamily") -> TesseraFamily:
    """Accept a family, a family name, or a full format name."""
    if isinstance(family, TesseraFamily):
        return family
    if not isinstance(family, str):
        raise TesseraFormatError(f"not a Tessera family: {family!r}")
    parsed = parse_tessera_format_name(family)
    if parsed is not None:
        return parsed[0]
    head = family.removeprefix("TESSERA_")
    base, sep, arity = head.rpartition("_K")
    if not sep or not arity.isdigit():
        raise TesseraFormatError(
            f"unknown Tessera family {family!r}; expected TESSERA_<base>_K<arity>"
        )
    return tessera_family(base, int(arity))


def validate_body_rate_q256(
    family: "str | TesseraFamily", body_rate_q256: int
) -> int:
    spec = get_tessera_family(family)
    if type(body_rate_q256) is not int:
        raise TesseraFormatError("body_rate_q256 must be a JSON integer")
    lower, upper = spec.mathematical_q256_bounds
    if not lower <= body_rate_q256 <= upper:
        raise TesseraFormatError(
            f"{spec.name} body_rate_q256 must be in [{lower}, {upper}], "
            f"got {body_rate_q256}"
        )
    return body_rate_q256


def realisable_rungs(
    family: "str | TesseraFamily", step_q256: int = 1
) -> range:
    """Every addressable rung of a family, as per-position q256.

    ``step_q256=1`` is the true resolution -- 1/256 of a bit per position --
    and every value in it is realisable at superblock scale.  A coarser step is
    a *budget* decision (how many rungs to measure), never a correctness one.
    """
    spec = get_tessera_family(family)
    if step_q256 < 1:
        raise TesseraFormatError(f"step_q256 must be >= 1, got {step_q256}")
    lo, hi = spec.mathematical_q256_bounds
    return range(lo, hi + 1, step_q256)


def artifact_bpp(family: "str | TesseraFamily", body_rate_q256: int) -> Fraction:
    """Bits per position the artifact actually weighs: body plus scale plane."""
    spec = get_tessera_family(family)
    validate_body_rate_q256(spec, body_rate_q256)
    return Fraction(body_rate_q256 + SCALE_PLANE_BITS_Q256, Q256_UNIT)


def enumerate_grid_space(
    bases: "Sequence[str] | None" = None,
    arities: Sequence[int] = (1, 2),
) -> Iterator[TesseraFamily]:
    """Every family the cost budget admits, cheapest code space first.

    Defaults to the two hardware bases plus the free grids that have been
    measured.  Families over the anchor budget are skipped rather than raising,
    because enumerating a space is asking what is *available*.
    """
    if bases is None:
        bases = ("E2M1", "E4M3", "LM8", "LM16", "LM32", "LM64")
    seen = []
    for base in bases:
        for arity in arities:
            try:
                seen.append(tessera_family(base, arity))
            except TesseraFormatError:
                continue
    seen.sort(key=lambda f: (f.payload_bits, f.base, f.arity))
    yield from seen


def parse_tessera_format_name(name: object) -> "tuple[TesseraFamily, int] | None":
    """Split ``TESSERA_E2M1_K2_R896`` into family and rung, or return None.

    Returns None rather than raising for anything that is not Tessera-shaped,
    because every caller is asking "is this one of mine?" about a menu that
    also holds NVFP4, FP8 and BF16.  A name that *is* Tessera-shaped but names
    an illegal rung raises, because that is a real error and silence there
    would put an unpriced format in front of the DP.
    """
    if not isinstance(name, str):
        return None
    match = _FORMAT_NAME.match(name)
    if match is None:
        return None
    base, arity, rung = match.group(1), int(match.group(2)), int(match.group(3))
    spec = tessera_family(base, arity)
    return (spec, validate_body_rate_q256(spec, rung))


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
            if isinstance(raw, (str, bytes, bytearray, Mapping)) or not isinstance(
                raw, Sequence
            ):
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

    def rungs(self) -> range:
        lo, hi = self.bounds_q256
        return range(lo, hi + 1, self.step_q256 or 1)

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
