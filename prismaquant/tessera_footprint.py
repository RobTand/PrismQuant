"""Exact serialized bytes for one Tessera Linear, from Tessera's own layout.

This replaces ``trellis_footprint`` in the allocator's byte path.  The two are
not ports of each other and must not be: ``trellis_footprint`` prices the
``gridbook.trellis.wire.v1`` layout -- an 88-byte binary header, its own row
alignment and block-offset rules -- and Tessera has a different wire
(``prismaquant.tessera.v1``) with a different plane set.  Re-deriving Tessera's
byte count here from Gridbook's model would produce a number that no exporter
would ever write.

So nothing here counts bytes.  ``tessera.layout`` does, through the same
``build_planes`` / ``build_terminal`` path the artifact writer uses, and this
module arranges the call and reports the result.  That is deliberate: the
allocator's byte budget, the accountant's figures and the exported artifact
have to be one number, and the only way to guarantee that is to have one
implementation of it.

The allocator reads four fields -- ``exact_bpw``, ``format``, ``shape`` and
``total_bytes``.  The rest is provenance, carried so a priced candidate can be
audited back to the planes it was priced from.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
import hashlib
import json

from .tessera_formats import (
    tessera_wire_defaults,
    SUPERBLOCK_WEIGHTS,
    TesseraFamily,
    TesseraFormatError,
    get_tessera_family,
    validate_body_rate_q256,
)

__all__ = [
    "TESSERA_TENSOR_PAYLOAD_SCHEMA",
    "tessera_tensor_payload_breakdown",
    "validate_tessera_tensor_payload_breakdown",
]

TESSERA_TENSOR_PAYLOAD_SCHEMA = "prismaquant.tessera_tensor_payload.v1"

#: Tessera's group/half geometry, from the S6b scale contract: an E8M0 base
#: byte per 32 weights plus a 4-bit refinement per 16.  Carried on the wire
#: rather than assumed, which is why they are named here and not inlined.
GROUP_WEIGHTS = 32
HALF_WEIGHTS = 16

try:  # pragma: no cover
    from tessera.layout import TerminalSpec, build_planes, build_terminal
    from tessera.manifest import Geometry
except ImportError as exc:  # pragma: no cover
    raise TesseraFormatError(
        "prismaquant.tessera_footprint requires the `tessera` package, which "
        "owns the plane layout these byte counts come from."
    ) from exc


#: What the recipe digest covers.  Named because a content address is only
#: meaningful next to its scope: this addresses the *pre-render recipe* -- the
#: family, rung, geometry and plane counts -- and deliberately not the rendered
#: weights, which do not exist yet when a candidate is priced.
_IDENTITY_SCOPE = "family+rung+geometry+planes"


def _recipe_identity(breakdown: Mapping[str, object]) -> str:
    """SHA-256 over canonical JSON of everything but the digest itself.

    A content address, not an authorization signature.  Recomputing it detects
    a report that has been edited or has drifted from the layout that produced
    it -- which is exactly what a downstream price must not be built on.
    """
    body = {k: v for k, v in breakdown.items()
            if k != "pre_render_recipe_identity_sha256"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"),
                         default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _alphabet_bytes(
    spec: TesseraFamily, alphabets: "Mapping[int, Sequence[int]] | None"
) -> int:
    """Bytes for the anchor tables, one entry per code in the grid's width.

    A grid of at most 256 codes indexes in a byte; wider grids need four.  The
    tables are per *rate*, because a rate's anchor set is ``2^(R+1)`` codes and
    a unit that mixes rates carries a table for each rate it uses.
    """
    if not alphabets:
        return 0
    width = 1 if (1 << spec.payload_bits) <= 256 else 4
    total = 0
    for rate, codes in alphabets.items():
        if not 1 <= rate <= spec.rate_cap:
            raise TesseraFormatError(
                f"{spec.name}: alphabet given for rate {rate}, outside 1..{spec.rate_cap}"
            )
        expected = 1 << (rate + 1)
        if len(codes) != expected:
            raise TesseraFormatError(
                f"{spec.name} rate {rate}: alphabet has {len(codes)} anchors, "
                f"|A_R| = 2^(R+1) = {expected}"
            )
        total += len(codes) * width
    return total


def tessera_tensor_payload_breakdown(
    shape: Sequence[int],
    *,
    family: "str | TesseraFamily",
    body_rate_q256: int,
    layout: str = "tight",
    schedule: "Sequence[int] | None" = None,
    alphabets: "Mapping[int, Sequence[int]] | None" = None,
    alphabet_bytes: "int | None" = None,
    sidecar_header_bytes: int = 0,
    completion: "int | None" = 0,
    with_diagonals: bool = False,
    span: "int | None" = None,
    scale_plane: "str | None" = None,
) -> dict[str, object]:
    """Exact serialized bytes for one 2-D Linear weight at one Tessera rung.

    ``span`` and ``scale_plane`` default to the tessera exporter's wire
    (``tessera_wire_defaults``), so a footprint priced here is the footprint
    of the bytes ``encode_linear`` writes.  Both are recorded in the breakdown
    and re-derived by ``validate_tessera_tensor_payload_breakdown``.

    ``schedule`` may be omitted, in which case the canonical Bresenham schedule
    for the rung is used -- the same one the encoder would build.  Passing one
    is how an importance-placed arrangement gets priced; it is checked against
    the rung rather than trusted, because a schedule that does not realise its
    root would price a rung the artifact does not contain.
    """
    spec = get_tessera_family(family)
    rung = validate_body_rate_q256(spec, body_rate_q256)
    default_span, default_plane = tessera_wire_defaults()
    span = default_span if span is None else int(span)
    plane = default_plane if scale_plane is None else str(scale_plane)
    if span < 1:
        raise TesseraFormatError(f"span must be positive, got {span}")
    if plane not in ("s6b", "lut16"):
        raise TesseraFormatError(f"unknown scale plane {plane!r}; s6b or lut16")

    dims = tuple(shape)
    if len(dims) != 2 or any(type(v) is not int or v <= 0 for v in dims):
        raise TesseraFormatError(
            f"Tessera tensor shape must be two positive integers, got {dims}"
        )
    rows, columns = dims
    if columns % SUPERBLOCK_WEIGHTS:
        raise TesseraFormatError(
            f"columns must be a multiple of the {SUPERBLOCK_WEIGHTS}-column "
            f"superblock; a short trailing block has no quota to keep, got {columns}"
        )
    if type(sidecar_header_bytes) is not int or sidecar_header_bytes < 0:
        raise TesseraFormatError("sidecar_header_bytes must be nonnegative")

    rates = (
        spec.column_schedule(rung, columns)
        if schedule is None
        else tuple(int(r) for r in schedule)
    )
    if len(rates) != columns:
        raise TesseraFormatError(
            f"schedule covers {len(rates)} columns for shape {dims}"
        )
    # A schedule is only this rung's schedule if its quota matches.  Checking
    # here is what stops a mispriced candidate reaching the DP.
    quota = sum(rates) * SUPERBLOCK_WEIGHTS
    expected = rung * spec.arity * columns
    if quota * 256 != expected * SUPERBLOCK_WEIGHTS:
        raise TesseraFormatError(
            f"{spec.name}: schedule sums to {sum(rates)} bits over {columns} "
            f"columns, which is not rung {rung} (q256)"
        )
    if any(not 1 <= r <= spec.rate_cap for r in rates):
        raise TesseraFormatError(
            f"{spec.name}: schedule has a rate outside 1..{spec.rate_cap}"
        )

    # --- arity: the code grid is not the weight grid -----------------------
    # A tuple code covers `arity` **consecutive rows** (tessera's `tuple_grid`:
    # codes map onto k consecutive rows, because the trellis runs down columns
    # and a tuple must be contiguous along the trellis axis).  So the body
    # plane has `rows // arity` code-rows, not `rows`.
    #
    # `Geometry.positions` is `rows * columns`, and the scale planes are quoted
    # off it -- but S6b groups *weights*, 32 of them, not codes.  Shrinking the
    # row count alone would undercount the scale plane by exactly `arity`.
    # Scaling the group geometry by the same factor cancels it exactly:
    #     positions // (32 // k) == (weights // k) // (32 // k) == weights // 32
    # which is the honest count, with no second accountant and no correction
    # term applied after the fact.
    if rows % spec.arity:
        raise TesseraFormatError(
            f"{spec.name}: {rows} rows is not a multiple of arity {spec.arity}; "
            "a tuple code spans that many consecutive rows and cannot straddle "
            "the end of the tensor"
        )
    if GROUP_WEIGHTS % spec.arity or HALF_WEIGHTS % spec.arity:
        raise TesseraFormatError(
            f"{spec.name}: arity {spec.arity} does not divide the S6b group "
            f"geometry ({GROUP_WEIGHTS}/{HALF_WEIGHTS})"
        )
    code_rows = rows // spec.arity
    if code_rows % span:
        raise TesseraFormatError(
            f"{spec.name}: {code_rows} code rows is not a whole number of "
            f"span-{span} super-symbols; this shape cannot carry the span"
        )

    # `alphabet_bytes` is the already-counted figure, which is what a recorded
    # footprint carries; `alphabets` is the table itself.  Revalidating a report
    # must use the recorded count, or it re-prices a different unit.
    if alphabet_bytes is None:
        alphabet_bytes = _alphabet_bytes(spec, alphabets)
    elif type(alphabet_bytes) is not int or alphabet_bytes < 0:
        raise TesseraFormatError("alphabet_bytes must be a nonnegative integer")

    geometry = Geometry(
        rows=code_rows,
        columns=columns,
        superblock_columns=SUPERBLOCK_WEIGHTS,
        group_weights=GROUP_WEIGHTS // spec.arity,
        half_weights=HALF_WEIGHTS // spec.arity,
        quantizable_params=rows * columns,
    )
    terminal_spec = TerminalSpec(
        slot_id="alloc",
        # ``completion`` is the second rate axis, and the default is the
        # exporter's: ``encode_linear(completion=0)``.  It briefly defaulted to
        # the cap instead, because ``unit_artifact`` was writing the COMPLETION
        # plane at full width whatever depth the encoder spent -- so the cap
        # was, for a few hours, what the wire really did.  Both defaults have
        # been wrong in opposite directions and by the same amount, and the
        # only defence is that this spec now sizes the *planes* as well as the
        # terminal, so the two cannot describe different artifacts.
        completion_bits=tuple(
            (spec.rate_cap - r) if completion is None
            else min(completion, spec.rate_cap - r)
            for r in rates
        ),
        released_positions=0,
        # A LUT plane has no base plane; its table lives in the manifest
        # (side bytes, outside the plane region this accountant prices).
        with_scale_base=plane == "s6b",
        with_scale_refine=True,
        with_diagonals=with_diagonals,
    )
    planes = build_planes(
        geometry,
        rates,
        bytes(alphabet_bytes),
        b"",
        with_diagonals=with_diagonals,
        cap=spec.rate_cap,
        spec=terminal_spec,
        span=span,
    )
    record = build_terminal(
        geometry, rates, terminal_spec, planes, alphabet_bytes, 0,
        cap=spec.rate_cap, span=span,
    )

    total_bytes = record.exact_bytes + sidecar_header_bytes
    exact_bpw = Fraction(total_bytes * 8, rows * columns)
    breakdown = {
        "schema": TESSERA_TENSOR_PAYLOAD_SCHEMA,
        "wire_schema": "prismaquant.tessera.v1",
        "format": spec.format_name(rung),
        "family": spec.name,
        "grid": spec.base,
        "arity": spec.arity,
        "lane": spec.lane,
        "shape": [rows, columns],
        "layout": layout,
        "body_rate_q256": rung,
        "scale_contract": plane,
        "trellis_span": span,
        "superblock_weights": SUPERBLOCK_WEIGHTS,
        "schedule_bits_per_code_row": sum(rates),
        "code_rows": code_rows,
        "distinct_rates": sorted(set(rates)),
        "alphabet_bytes": alphabet_bytes,
        "sidecar_header_bytes": sidecar_header_bytes,
        "plane_elements": list(record.plane_elements),
        "payload_bytes": record.exact_bytes,
        "total_bytes": total_bytes,
        "exact_bpp_payload": str(record.exact_bpp),
        "exact_bpw": float(exact_bpw),
        "exact_bpw_rational": [exact_bpw.numerator, exact_bpw.denominator],
        # The stock lane materialises into `terminal_format` at load, so the
        # resident cost is that format's, not this one's.  Reported because a
        # byte win that disappears at load is not a byte win.
        "terminal_format": spec.terminal_format,
        "materialises": spec.lane == "stock",
        "pre_render_recipe_identity_scope": _IDENTITY_SCOPE,
    }
    breakdown["pre_render_recipe_identity_sha256"] = _recipe_identity(breakdown)
    return breakdown


def validate_tessera_tensor_payload_breakdown(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Re-check a footprint at an API boundary, and recompute what it claims.

    The arithmetic is re-derived rather than trusted: a report that has been
    edited or has drifted from the layout that produced it is exactly the
    thing a downstream price should not be built on.
    """
    if not isinstance(payload, Mapping):
        raise TesseraFormatError("Tessera footprint must be a mapping")
    copied = dict(payload)
    if copied.get("schema") != TESSERA_TENSOR_PAYLOAD_SCHEMA:
        raise TesseraFormatError(
            f"footprint schema must be {TESSERA_TENSOR_PAYLOAD_SCHEMA}, "
            f"got {copied.get('schema')!r}"
        )
    shape = copied.get("shape")
    if not isinstance(shape, Sequence) or len(shape) != 2:
        raise TesseraFormatError("footprint shape must be a two-element sequence")
    rows, columns = int(shape[0]), int(shape[1])
    recomputed = tessera_tensor_payload_breakdown(
        (rows, columns),
        family=str(copied["family"]),
        body_rate_q256=int(copied["body_rate_q256"]),
        layout=str(copied.get("layout", "tight")),
        alphabet_bytes=int(copied.get("alphabet_bytes", 0)),
        sidecar_header_bytes=int(copied.get("sidecar_header_bytes", 0)),
        # A report written before minor 1 carries neither field and means the
        # wire of its day; one written after names what it priced.
        span=int(copied.get("trellis_span", 1)),
        scale_plane=str(copied.get("scale_contract", "s6b")),
    )
    claimed = copied.get("pre_render_recipe_identity_sha256")
    if claimed != _recipe_identity(copied):
        raise TesseraFormatError(
            "footprint recipe identity does not address its own contents; the "
            "report has been edited or has drifted from its layout"
        )
    for field in ("total_bytes", "payload_bytes", "exact_bpw", "format"):
        if copied.get(field) != recomputed[field]:
            raise TesseraFormatError(
                f"footprint {field} is {copied.get(field)!r}, but the layout "
                f"gives {recomputed[field]!r}"
            )
    return copied
