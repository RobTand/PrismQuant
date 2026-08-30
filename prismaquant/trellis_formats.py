"""Research contracts for the Gridbook rate-256 tail-biting trellis formats.

This module is deliberately torch-free.  It defines the mathematical and wire
admission boundary that probing, accounting, export experiments, and Gridbook
kernel tests share.  It does *not* make a format producer-eligible: promotion
requires a pinned Gridbook wire ABI plus quality and physical serving receipts.

``body_rate_q256`` is the exact number of body bits in a constant-quota
256-weight block.  For example, ``384`` names 1.5 body bits/weight.  Keeping
that integer in persisted identities avoids float spelling and rounding drift.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
import math
import re


TRELLIS_FORMAT_CONTRACT_SCHEMA = "prismaquant.trellis_format_contract.v1"
TRELLIS_RATE_SURFACE_SCHEMA = "prismaquant.trellis_rate_surface.v1"
TRELLIS_WIRE_SCHEMA = "gridbook.trellis.wire.v1"
SUPERBLOCK_WEIGHTS = 256
STATE_COUNT = 256
STATE_MEMORY_BITS = 8
GENERATOR_OCTAL = ("561", "753")
MIN_TRELLIS_STEPS = STATE_MEMORY_BITS
LAYOUT_TIGHT_OFFSETS = "tight_offsets"
LAYOUT_FIXED_QUOTA = "fixed_quota_per_256"
LAYOUTS = frozenset({LAYOUT_TIGHT_OFFSETS, LAYOUT_FIXED_QUOTA})

E2M1_FAMILY = "TCQ_E2M1_R256"
E4M3_FAMILY = "TCQ_E4M3_R256"

RATE_SURFACE_ALL_LEGAL = "all_legal"
RATE_SURFACE_DENSE = "dense"
RATE_SURFACE_ADAPTIVE = "adaptive"
RATE_SURFACE_MODES = frozenset({
    RATE_SURFACE_ALL_LEGAL,
    RATE_SURFACE_DENSE,
    RATE_SURFACE_ADAPTIVE,
})

_FORMAT_NAME = re.compile(r"(TCQ_E2M1|TCQ_E4M3)_R([0-9]+)")


class TrellisFormatError(ValueError):
    """A trellis family, rung, schedule, or alphabet is invalid."""


def _canonical_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _json_integer(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise TrellisFormatError(f"{field} must be a JSON integer")
    return value


@dataclass(frozen=True, slots=True)
class TrellisFamily:
    family: str
    grid: str
    grid_bits: int
    shaped_max_rate: int
    bypass_rate: int
    scale_contract: str
    quality_candidate_q256: tuple[int, ...]
    terminal_format: str
    minimum_capability_sm: int

    @property
    def mathematical_q256_bounds(self) -> tuple[int, int]:
        # At least nu=8 positions must remain in the trellis.  The maximum
        # therefore uses bypass_rate at 248 positions and shaped_max_rate at 8.
        return (
            SUPERBLOCK_WEIGHTS,
            (
                (SUPERBLOCK_WEIGHTS - MIN_TRELLIS_STEPS) * self.bypass_rate
                + MIN_TRELLIS_STEPS * self.shaped_max_rate
            ),
        )

    def format_name(self, body_rate_q256: int) -> str:
        validate_body_rate_q256(self, body_rate_q256)
        prefix = self.family.removesuffix("_R256")
        return f"{prefix}_R{body_rate_q256}"


@dataclass(frozen=True, slots=True)
class TrellisRateSurface:
    """A deterministic, allocator-addressable research rate surface.

    Rates are parameters of one family implementation, not registry entries or
    promises of separately compiled kernels.  ``adaptive`` surfaces carry the
    identity of the measured lower-convex RD hull that proposed new rates.
    """

    family: str
    mode: str
    bounds_q256: tuple[int, int]
    step_q256: int | None = None
    anchor_q256: tuple[int, ...] = ()
    proposed_q256: tuple[int, ...] = ()
    source_identity_sha256: str | None = None

    def __post_init__(self) -> None:
        for field in ("bounds_q256", "anchor_q256", "proposed_q256"):
            raw = getattr(self, field)
            if isinstance(raw, (str, bytes, bytearray, Mapping)):
                raise TrellisFormatError(
                    f"rate-surface {field} must be an integer sequence"
                )
            try:
                normalized = tuple(raw)
            except TypeError as exc:
                raise TrellisFormatError(
                    f"rate-surface {field} must be an integer sequence"
                ) from exc
            object.__setattr__(self, field, normalized)
        spec = get_trellis_family(self.family)
        if self.family != spec.family:
            raise TrellisFormatError("rate-surface family must be canonical")
        if self.mode not in RATE_SURFACE_MODES:
            raise TrellisFormatError(
                f"unknown rate-surface mode {self.mode!r}; expected "
                f"{sorted(RATE_SURFACE_MODES)}"
            )
        if (
            len(self.bounds_q256) != 2
            or any(type(value) is not int for value in self.bounds_q256)
        ):
            raise TrellisFormatError(
                "rate-surface bounds_q256 must be two JSON integers"
            )
        start, stop = self.bounds_q256
        validate_body_rate_q256(spec, start)
        validate_body_rate_q256(spec, stop)
        if start > stop:
            raise TrellisFormatError("rate-surface bounds are reversed")
        for field, values in (
            ("anchor_q256", self.anchor_q256),
            ("proposed_q256", self.proposed_q256),
        ):
            if tuple(sorted(set(values))) != values:
                raise TrellisFormatError(
                    f"rate-surface {field} must be sorted and unique"
                )
            for rate in values:
                validate_body_rate_q256(spec, rate)
                if not start <= rate <= stop:
                    raise TrellisFormatError(
                        f"rate-surface {field} contains {rate} outside "
                        f"requested bounds [{start}, {stop}]"
                    )
        if self.step_q256 is not None and (
            type(self.step_q256) is not int or self.step_q256 <= 0
        ):
            raise TrellisFormatError("step_q256 must be a positive JSON integer")
        if self.mode == RATE_SURFACE_DENSE and self.step_q256 is None:
            raise TrellisFormatError("dense rate surfaces require step_q256")
        if self.mode != RATE_SURFACE_DENSE and self.step_q256 is not None:
            raise TrellisFormatError(
                "step_q256 is defined only for dense rate surfaces"
            )
        if self.mode == RATE_SURFACE_ALL_LEGAL and self.anchor_q256:
            raise TrellisFormatError(
                "all-legal rate surfaces already contain every anchor"
            )
        if self.mode != RATE_SURFACE_ADAPTIVE and self.proposed_q256:
            raise TrellisFormatError(
                "proposed_q256 is defined only for adaptive surfaces"
            )
        if self.mode == RATE_SURFACE_ADAPTIVE:
            if not self.anchor_q256 and not self.proposed_q256:
                raise TrellisFormatError(
                    "adaptive rate surface must contain an anchor or proposal"
                )
            if set(self.anchor_q256).intersection(self.proposed_q256):
                raise TrellisFormatError(
                    "adaptive proposals must not repeat measured anchors"
                )
            identity = self.source_identity_sha256 or ""
            if (
                len(identity) != 64
                or any(character not in "0123456789abcdef" for character in identity)
            ):
                raise TrellisFormatError(
                    "adaptive rate surfaces require a lowercase SHA-256 "
                    "source identity"
                )
        elif self.source_identity_sha256 is not None:
            raise TrellisFormatError(
                "source_identity_sha256 is defined only for adaptive surfaces"
            )

    def iter_rates_q256(self) -> Iterator[int]:
        """Generate rates deterministically without storing a per-unit list."""

        start, stop = self.bounds_q256
        if self.mode == RATE_SURFACE_ADAPTIVE:
            yield from sorted(set(self.anchor_q256) | set(self.proposed_q256))
            return
        step = 1 if self.mode == RATE_SURFACE_ALL_LEGAL else self.step_q256
        assert step is not None
        extras = tuple(sorted(set(self.anchor_q256) | {stop}))
        extra_index = 0
        for base_rate in range(start, stop + 1, step):
            while (
                extra_index < len(extras)
                and extras[extra_index] < base_rate
            ):
                yield extras[extra_index]
                extra_index += 1
            if extra_index < len(extras) and extras[extra_index] == base_rate:
                extra_index += 1
            yield base_rate
        yield from extras[extra_index:]

    @property
    def rates_q256(self) -> "TrellisRateSequence":
        """A re-iterable O(1)-state view of the surface's q256 values."""

        return TrellisRateSequence(self)

    def as_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": TRELLIS_RATE_SURFACE_SCHEMA,
            "family": self.family,
            "mode": self.mode,
            "bounds_q256_inclusive": list(self.bounds_q256),
            "step_q256": self.step_q256,
            "anchor_q256": list(self.anchor_q256),
            "proposed_q256": list(self.proposed_q256),
            "source_identity_sha256": self.source_identity_sha256,
            "rate_count": len(self.rates_q256),
            "rate_generator": (
                "explicit_adaptive_anchors_and_proposals"
                if self.mode == RATE_SURFACE_ADAPTIVE
                else "inclusive_integer_range_with_anchors"
            ),
            "address_fields": [
                "family",
                "body_rate_q256",
                "layout",
                "pre_render_recipe_identity_sha256",
            ],
            "kernel_specialization": "parameterized_rate_no_surface_compile",
            "public_format_registry_entries_created": 0,
            "research_only": True,
            "producer_eligible": False,
        }
        return {**body, "identity_sha256": _canonical_sha256(body)}

    @property
    def identity_sha256(self) -> str:
        return str(self.as_dict()["identity_sha256"])


@dataclass(frozen=True, slots=True)
class TrellisRateSequence(Sequence[int]):
    """Lazy sequence facade over a :class:`TrellisRateSurface`."""

    surface: TrellisRateSurface

    def __iter__(self) -> Iterator[int]:
        return self.surface.iter_rates_q256()

    def __len__(self) -> int:
        start, stop = self.surface.bounds_q256
        if self.surface.mode == RATE_SURFACE_ADAPTIVE:
            return len(set(self.surface.anchor_q256) | set(
                self.surface.proposed_q256
            ))
        step = (
            1
            if self.surface.mode == RATE_SURFACE_ALL_LEGAL
            else self.surface.step_q256
        )
        assert step is not None
        base = range(start, stop + 1, step)
        extras = set(self.surface.anchor_q256) | {stop}
        extra_count = sum(rate not in base for rate in extras)
        return len(base) + extra_count

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step < 0:
                # itertools.islice refuses negative steps.  Slicing is an
                # explicit materialization boundary (the returned value is a
                # tuple), so make one O(R) transient tuple without changing
                # the O(1)-state storage contract of the surface itself.
                return tuple(self)[index]
            return tuple(islice(iter(self), start, stop, step))
        if type(index) is not int:
            raise TypeError("rate sequence indices must be integers or slices")
        normalized = index if index >= 0 else len(self) + index
        if not 0 <= normalized < len(self):
            raise IndexError("rate sequence index out of range")
        return next(islice(iter(self), normalized, normalized + 1))


FAMILIES: Mapping[str, TrellisFamily] = {
    E2M1_FAMILY: TrellisFamily(
        family=E2M1_FAMILY,
        grid="e2m1",
        grid_bits=4,
        shaped_max_rate=3,
        bypass_rate=4,
        scale_contract="group16_fp8_e4m3_0p5_bpw",
        quality_candidate_q256=(384, 512, 640, 768, 896),
        terminal_format="NVFP4",
        minimum_capability_sm=120,
    ),
    E4M3_FAMILY: TrellisFamily(
        family=E4M3_FAMILY,
        grid="e4m3fn",
        grid_bits=8,
        shaped_max_rate=7,
        bypass_rate=8,
        scale_contract="per_output_row_fp32",
        quality_candidate_q256=(1152,),
        terminal_format="FP8_E4M3",
        minimum_capability_sm=89,
    ),
}

# torch.float8_e4m3fn byte encodings 0x7f and 0xff are NaNs.  A wire alphabet
# may repeat legal finite codes (required by the R7 research ceiling), but it
# must never emit either NaN byte.
E4M3FN_NAN_CODES = frozenset({0x7F, 0xFF})
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def native_code_value(
    family: str | TrellisFamily, code: int,
) -> float:
    spec = get_trellis_family(family)
    if type(code) is not int or not 0 <= code < (1 << spec.grid_bits):
        raise TrellisFormatError(
            f"native code must be a {spec.grid_bits}-bit JSON integer"
        )
    if spec.family == E2M1_FAMILY:
        value = _E2M1_MAGNITUDES[code & 7]
        return -value if code & 8 else value
    if code in E4M3FN_NAN_CODES:
        raise TrellisFormatError("E4M3FN trellis alphabets cannot emit NaNs")
    sign = -1.0 if code & 0x80 else 1.0
    exponent = (code >> 3) & 0xF
    mantissa = code & 7
    value = (
        math.ldexp(mantissa / 8.0, 1 - 7)
        if exponent == 0
        else math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    )
    return sign * value


def get_trellis_family(value: str | TrellisFamily) -> TrellisFamily:
    if isinstance(value, TrellisFamily):
        canonical = FAMILIES.get(value.family)
        if canonical != value:
            raise TrellisFormatError("trellis family object is not canonical")
        return value
    try:
        return FAMILIES[str(value)]
    except KeyError as exc:
        raise TrellisFormatError(f"unknown trellis family {value!r}") from exc


def validate_body_rate_q256(
    family: str | TrellisFamily,
    body_rate_q256: int,
) -> int:
    spec = get_trellis_family(family)
    if type(body_rate_q256) is not int:
        raise TrellisFormatError("body_rate_q256 must be a JSON integer")
    lower, upper = spec.mathematical_q256_bounds
    if not lower <= body_rate_q256 <= upper:
        raise TrellisFormatError(
            f"{spec.family} body_rate_q256 must be in [{lower}, {upper}], "
            f"got {body_rate_q256}"
        )
    return body_rate_q256


def trellis_rate_surface(
    family: str | TrellisFamily,
    *,
    mode: str,
    start_q256: int | None = None,
    stop_q256: int | None = None,
    step_q256: int | None = None,
    include_q256: Sequence[int] = (),
) -> TrellisRateSurface:
    """Build an explicit all-legal or deterministic dense research surface.

    ``all_legal`` returns every integer q256 admitted by the mathematical
    family contract. ``dense`` walks from ``start`` by ``step`` and always
    includes the exact stop, even when the final interval is shorter. Optional
    anchors are unioned into the dense grid. The caller must name the mode so a
    sparse quality seed can never silently expand into thousands of candidates.
    """

    spec = get_trellis_family(family)
    legal_start, legal_stop = spec.mathematical_q256_bounds
    start = (
        legal_start
        if start_q256 is None
        else _json_integer(start_q256, field="start_q256")
    )
    stop = (
        legal_stop
        if stop_q256 is None
        else _json_integer(stop_q256, field="stop_q256")
    )
    validate_body_rate_q256(spec, start)
    validate_body_rate_q256(spec, stop)
    if start > stop:
        raise TrellisFormatError("start_q256 must not exceed stop_q256")
    if isinstance(include_q256, (str, bytes, bytearray)):
        raise TrellisFormatError("include_q256 must be an integer sequence")
    raw_anchors = tuple(include_q256)
    for rate in raw_anchors:
        _json_integer(rate, field="include_q256 entry")
        validate_body_rate_q256(spec, rate)
        if not start <= rate <= stop:
            raise TrellisFormatError(
                f"included q256 rate {rate} lies outside [{start}, {stop}]"
            )
    anchors = tuple(sorted(set(raw_anchors)))

    if mode == RATE_SURFACE_ALL_LEGAL:
        if step_q256 is not None:
            raise TrellisFormatError(
                "all_legal rate surfaces do not accept step_q256"
            )
        if anchors:
            raise TrellisFormatError(
                "all_legal rate surfaces already include every anchor"
            )
        return TrellisRateSurface(
            family=spec.family,
            mode=mode,
            bounds_q256=(start, stop),
        )

    if mode != RATE_SURFACE_DENSE:
        if mode == RATE_SURFACE_ADAPTIVE:
            raise TrellisFormatError(
                "adaptive surfaces require measured frontier metadata; use "
                "trellis_allocator.adaptive_trellis_rate_surface"
            )
        raise TrellisFormatError(
            f"unknown rate-surface mode {mode!r}; expected "
            f"{[RATE_SURFACE_ALL_LEGAL, RATE_SURFACE_DENSE]}"
        )
    step = _json_integer(step_q256, field="step_q256")
    if step <= 0:
        raise TrellisFormatError("step_q256 must be positive")
    return TrellisRateSurface(
        family=spec.family,
        mode=mode,
        bounds_q256=(start, stop),
        step_q256=step,
        anchor_q256=anchors,
    )


def parse_trellis_format_name(name: str) -> tuple[TrellisFamily, int] | None:
    match = _FORMAT_NAME.fullmatch(str(name))
    if match is None:
        return None
    family = get_trellis_family(f"{match.group(1)}_R256")
    rate = validate_body_rate_q256(family, int(match.group(2)))
    return family, rate


def quality_candidate_format_names() -> tuple[str, ...]:
    """Return measured quality candidates; this grants no producer authority."""
    return tuple(
        family.format_name(rate)
        for family in FAMILIES.values()
        for rate in family.quality_candidate_q256
    )


def all_legal_trellis_format_names() -> tuple[str, ...]:
    """Every ``TCQ_<grid>_R<q256>`` name the family contracts admit.

    A serving-profile ``allow_formats_from`` needs a closed enum, and the
    trellis rate axis is dense rather than enumerable by hand: the wire's
    resolution is ``SUPERBLOCK_WEIGHTS/columns`` q256, so the *addressable*
    names are every integer q256 in each family's mathematical bounds. This
    grants no producer, render, or serving authority -- it is the vocabulary a
    profile rule can be written against, nothing more.
    """

    names: list[str] = []
    for family in FAMILIES.values():
        low, high = family.mathematical_q256_bounds
        names.extend(family.format_name(rate) for rate in range(low, high + 1))
    return tuple(names)


#: Materialized once; ``serving_profiles`` resolves ``module:ATTR`` references
#: at profile-load time and must see concrete sequences, not callables.  The
#: family splits let the research profile carry each native A=W activation
#: contract and the fail-closed ``unattested`` route status on the candidate
#: without conflating E2M1 W4A4 with E4M3 W8A8.
E2M1_TRELLIS_FORMAT_NAMES: tuple[str, ...] = tuple(
    E2M1_FAMILY.replace("_R256", f"_R{rate}")
    for rate in range(
        FAMILIES[E2M1_FAMILY].mathematical_q256_bounds[0],
        FAMILIES[E2M1_FAMILY].mathematical_q256_bounds[1] + 1,
    )
)
E4M3_TRELLIS_FORMAT_NAMES: tuple[str, ...] = tuple(
    E4M3_FAMILY.replace("_R256", f"_R{rate}")
    for rate in range(
        FAMILIES[E4M3_FAMILY].mathematical_q256_bounds[0],
        FAMILIES[E4M3_FAMILY].mathematical_q256_bounds[1] + 1,
    )
)
ALL_LEGAL_TRELLIS_FORMAT_NAMES: tuple[str, ...] = (
    E2M1_TRELLIS_FORMAT_NAMES + E4M3_TRELLIS_FORMAT_NAMES
)


def _schedule_values(schedule: Sequence[int]) -> tuple[int, ...]:
    if isinstance(schedule, (str, bytes, bytearray)):
        raise TrellisFormatError("trellis schedule must be an integer sequence")
    values = tuple(schedule)
    if any(type(value) is not int for value in values):
        raise TrellisFormatError("trellis schedule values must be JSON integers")
    return values


def validate_schedule(
    family: str | TrellisFamily,
    body_rate_q256: int,
    schedule: Sequence[int],
    *,
    layout: str,
) -> tuple[int, ...]:
    """Validate one tensor-shared per-input-column schedule.

    ``tight_offsets`` permits global reverse-water-filling and therefore
    variable block totals; the tensor-wide total must differ from the declared
    q256 target by less than one physical body bit.  ``fixed_quota_per_256``
    requires every *complete* block to contain exactly ``body_rate_q256`` bits,
    which is the fixed-stride/TMA research variant.  A short final block is
    charged exactly as scheduled.  Both layouts require at least eight
    non-bypass trellis positions in every block, including the short tail.
    """

    spec = get_trellis_family(family)
    target = validate_body_rate_q256(spec, body_rate_q256)
    if layout not in LAYOUTS:
        raise TrellisFormatError(
            f"unknown trellis layout {layout!r}; expected {sorted(LAYOUTS)}"
        )
    values = _schedule_values(schedule)
    if not values:
        raise TrellisFormatError("trellis schedule must not be empty")
    if any(not 1 <= value <= spec.bypass_rate for value in values):
        raise TrellisFormatError(
            f"{spec.family} schedule values must be in [1, {spec.bypass_rate}]"
        )
    blocks = tuple(
        values[start:start + SUPERBLOCK_WEIGHTS]
        for start in range(0, len(values), SUPERBLOCK_WEIGHTS)
    )
    for index, block in enumerate(blocks):
        trellis_steps = sum(value < spec.bypass_rate for value in block)
        if trellis_steps < MIN_TRELLIS_STEPS:
            raise TrellisFormatError(
                f"trellis block {index} has {trellis_steps} coded steps; "
                f"at least {MIN_TRELLIS_STEPS} are required"
            )
        if (
            layout == LAYOUT_FIXED_QUOTA
            and len(block) == SUPERBLOCK_WEIGHTS
            and sum(block) != target
        ):
            raise TrellisFormatError(
                f"fixed-quota block {index} has {sum(block)} body bits; "
                f"expected {target}"
            )
    if (
        layout == LAYOUT_TIGHT_OFFSETS
        and abs(
            sum(values) * SUPERBLOCK_WEIGHTS - target * len(values)
        ) >= SUPERBLOCK_WEIGHTS
    ):
        raise TrellisFormatError(
            "tight-offset schedule differs from its declared q256 target "
            "by at least one physical body bit"
        )
    return values


def validate_alphabets(
    family: str | TrellisFamily,
    schedule: Sequence[int],
    alphabets: Mapping[int, Sequence[int]],
) -> dict[int, tuple[int, ...]]:
    """Validate code-valued, per-rate reconstruction alphabets.

    The mapping stores native grid *codes*, not float spellings.  The kernel
    derives four modulo-four Ungerboeck subsets from the supplied order.
    """

    spec = get_trellis_family(family)
    values = _schedule_values(schedule)
    used = {rate for rate in values if rate < spec.bypass_rate}
    if set(alphabets) != used:
        raise TrellisFormatError(
            f"alphabet rates differ: expected {sorted(used)}, got "
            f"{sorted(alphabets)}"
        )
    result: dict[int, tuple[int, ...]] = {}
    maximum_code = (1 << spec.grid_bits) - 1
    for rate in sorted(used):
        if not 1 <= rate <= spec.shaped_max_rate:
            raise TrellisFormatError(f"invalid shaped rate {rate}")
        raw_codes = alphabets[rate]
        if isinstance(raw_codes, (str, bytes, bytearray)):
            raise TrellisFormatError(f"rate-{rate} alphabet must be an array")
        codes = tuple(raw_codes)
        required = 1 << (rate + 1)
        if len(codes) != required:
            raise TrellisFormatError(
                f"rate-{rate} alphabet needs {required} codes, got {len(codes)}"
            )
        if any(
            type(code) is not int or not 0 <= code <= maximum_code
            for code in codes
        ):
            raise TrellisFormatError(
                f"rate-{rate} alphabet contains a non-{spec.grid_bits}-bit code"
            )
        if spec.family == E4M3_FAMILY and E4M3FN_NAN_CODES.intersection(codes):
            raise TrellisFormatError("E4M3FN trellis alphabets cannot emit NaNs")
        ordered = tuple(sorted(
            codes, key=lambda code: (native_code_value(spec, code), code),
        ))
        if codes != ordered:
            raise TrellisFormatError(
                f"rate-{rate} alphabet must be sorted by decoded value then code"
            )
        duplicates = len(set(codes)) != len(codes)
        canonical_r7_duplicates = (
            spec.family == E4M3_FAMILY
            and rate == 7
            and codes.count(0x00) == 2
            and codes.count(0x80) == 2
            and all(
                codes.count(code) == 1
                for code in range(256)
                if code not in E4M3FN_NAN_CODES | {0x00, 0x80}
            )
        )
        if duplicates and not canonical_r7_duplicates:
            raise TrellisFormatError(
                f"rate-{rate} alphabet contains duplicate native codes"
            )
        result[rate] = codes
    return result


def format_contract_payload() -> dict[str, object]:
    """Return the closed research contract for receipts and documentation."""

    return {
        "schema": TRELLIS_FORMAT_CONTRACT_SCHEMA,
        "wire_schema": TRELLIS_WIRE_SCHEMA,
        "rate_surface_schema": TRELLIS_RATE_SURFACE_SCHEMA,
        "superblock_weights": SUPERBLOCK_WEIGHTS,
        "state_count": STATE_COUNT,
        "state_memory_bits": STATE_MEMORY_BITS,
        "generator_octal": list(GENERATOR_OCTAL),
        "minimum_trellis_steps": MIN_TRELLIS_STEPS,
        "schedule_scope": "tensor_input_column_shared_across_rows",
        "layouts": sorted(LAYOUTS),
        "allocator_rate_surface": {
            "modes": [RATE_SURFACE_ALL_LEGAL, RATE_SURFACE_DENSE],
            "adaptive_mode": RATE_SURFACE_ADAPTIVE,
            "address_fields": [
                "family",
                "body_rate_q256",
                "layout",
                "pre_render_recipe_identity_sha256",
            ],
            "public_format_registry_entries_created": 0,
            "producer_eligible": False,
        },
        "families": [
            {
                "family": family.family,
                "grid": family.grid,
                "grid_bits": family.grid_bits,
                "shaped_rate_range": [1, family.shaped_max_rate],
                "bypass_rate": family.bypass_rate,
                "mathematical_q256_bounds": list(
                    family.mathematical_q256_bounds
                ),
                "quality_candidate_q256": list(
                    family.quality_candidate_q256
                ),
                "research_q256_bounds_inclusive": list(
                    family.mathematical_q256_bounds
                ),
                "scale_contract": family.scale_contract,
                "terminal_format": family.terminal_format,
                "minimum_capability_sm": family.minimum_capability_sm,
                "producer_eligible": False,
            }
            for family in FAMILIES.values()
        ],
    }


__all__ = [
    "E2M1_FAMILY",
    "E2M1_TRELLIS_FORMAT_NAMES",
    "E4M3_FAMILY",
    "E4M3_TRELLIS_FORMAT_NAMES",
    "E4M3FN_NAN_CODES",
    "FAMILIES",
    "GENERATOR_OCTAL",
    "LAYOUT_FIXED_QUOTA",
    "LAYOUT_TIGHT_OFFSETS",
    "LAYOUTS",
    "MIN_TRELLIS_STEPS",
    "RATE_SURFACE_ADAPTIVE",
    "RATE_SURFACE_ALL_LEGAL",
    "RATE_SURFACE_DENSE",
    "RATE_SURFACE_MODES",
    "STATE_COUNT",
    "STATE_MEMORY_BITS",
    "SUPERBLOCK_WEIGHTS",
    "TRELLIS_FORMAT_CONTRACT_SCHEMA",
    "TRELLIS_RATE_SURFACE_SCHEMA",
    "TRELLIS_WIRE_SCHEMA",
    "TrellisFamily",
    "TrellisFormatError",
    "TrellisRateSurface",
    "TrellisRateSequence",
    "format_contract_payload",
    "get_trellis_family",
    "native_code_value",
    "parse_trellis_format_name",
    "quality_candidate_format_names",
    "trellis_rate_surface",
    "validate_alphabets",
    "validate_body_rate_q256",
    "validate_schedule",
]
