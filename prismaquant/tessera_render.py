"""Render a Linear through Tessera, in the shape the production cache wants.

The render contract PrismaQuant already has is exactly the one Tessera needs:
``render_production_weight(weight, fmt, qname=...)`` returns a **dequantized
weight of the same shape and dtype**, and every stage downstream -- the AURA
cost, the allocator's per-(Linear, format) price, the real-KL validation, the
exported bytes -- is defined against that tensor.  So a Tessera rung becomes a
first-class allocator candidate the moment encode->decode is reachable from a
format name.  No export path and no serving backend are required to price it,
which is the point: the measurement that decides whether Tessera is worth a
vLLM backend can be taken before the backend exists.

**Rungs are family parameters, not registry entries.**  ``tessera_formats``
states this deliberately -- a family addresses ~9500 rungs at 1/256-bpp
resolution, and materialising those as static ``REGISTRY`` rows would turn a
continuous rate axis into a menu someone has to maintain.  So the specs are
*synthesized on demand* from the name, and ``format_registry.get_format`` falls
back here for anything Tessera-shaped.  Every existing consumer that resolves a
format by name keeps working, unchanged.

**Nothing here reimplements Tessera.**  The grid, the rate schedule, the
forests, the Viterbi and the reconstruction all come from the ``tessera``
package; this module is the adapter and holds no numeric constant of its own.
A second copy of a rate constant is a drift bug waiting for a rate to change.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import torch

from .tessera_formats import (
    TesseraFamily,
    parse_tessera_format_name,
)

__all__ = [
    "TESSERA_CONV_MEMORY",
    "TESSERA_GROUP",
    "TESSERA_HALF",
    "is_tessera_format",
    "render_tessera_weight",
    "synthesize_tessera_spec",
    "tessera_quantize_dequantize",
]

#: The convolutional code the encoder profile commits to.  Memory 6 is the
#: order every measured Tessera figure was produced at; it is not a free
#: parameter here, because changing it changes ``encoder_profile_id`` and so
#: changes which artifacts a reader will accept.
TESSERA_CONV_MEMORY = 6

#: Segment-2b scale geometry.  These are the values ``artifact_bpp`` already
#: prices (``SCALE_PLANE_BITS_Q256``); rendering on different ones would make
#: the surrogate and the accountant disagree about the same artifact.
TESSERA_GROUP = 32
TESSERA_HALF = 16


#: Does a pinned runtime execute Tessera bytes natively?
#:
#: The kernel lane -- where the body stays compressed and is decoded inside the
#: GEMV -- has a Triton decode kernel but no vLLM backend, so nothing serves
#: these bytes today.  Principle 9 makes this a *measured platform fact*, not a
#: preference: flipping it is an attestation that a pinned runtime loads and
#: routes the format on real shapes, and it belongs to that evidence, not to
#: whoever is adding a format next.
_TESSERA_SERVING_LANE_EXISTS = False


def is_tessera_format(name: object) -> bool:
    """True for a Tessera-shaped format name, without raising on others."""
    try:
        return parse_tessera_format_name(name) is not None
    except Exception:
        # A Tessera-shaped name naming an illegal rung raises inside the parser.
        # That is a real error for a caller that means to *use* the format, but
        # this predicate is only ever asking "is this one of mine?", and the
        # answer there is yes -- let the render path raise with the detail.
        return isinstance(name, str) and name.startswith("TESSERA_")


@lru_cache(maxsize=64)
def _grid_for(family: TesseraFamily):
    from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid

    bases = {"E2M1": E2M1_GRID, "E4M3": E4M3_GRID}
    if family.base not in bases:
        raise NotImplementedError(
            f"{family.name}: only hardware base grids render today "
            f"({sorted(bases)}). A free/Lloyd-Max base is measurable but not "
            "serialisable -- its values are fitted to the tensor and no "
            "identifier reproduces them, so it needs a VALUES plane first."
        )
    return tuple_grid(bases[family.base], family.arity)


@lru_cache(maxsize=64)
def tessera_rung_is_serialisable(name: str) -> bool:
    """Can the *wire* carry this rung's bytes at all?

    Distinct from "does a runtime serve it".  A rung can render perfectly and
    still be unserialisable: ``_grid_for`` admits any *hardware* base, but the
    reader resolves a grid by digest against ``SERIALISABLE_GRIDS``, which is a
    permanent wire commitment and a strictly smaller set.  ``E4M3`` is exactly
    that gap -- ``TESSERA_E4M3_K1_R1024`` renders, prices at 4.5000 bpp, and
    then dies in ``alphabet_plane()`` at export time, after the allocation and
    the whole production cache have been built.

    Pricing a rung the exporter cannot write is the menu offering something the
    format cannot deliver, so this predicate exists to be read by a gate rather
    than discovered by a traceback.
    """
    from tessera.alphabet import SERIALISABLE_GRIDS, grid_digest

    parsed = parse_tessera_format_name(name)
    if parsed is None:
        return False
    try:
        grid = _grid_for(parsed[0])
    except NotImplementedError:
        return False           # a free base: no identifier reproduces its values
    return grid_digest(grid) in SERIALISABLE_GRIDS


@lru_cache(maxsize=256)
def _plan(family: TesseraFamily, body_rate_q256: int, n_columns: int):
    """Rate schedule and forests for one (family, rung, width).

    Cached because the forests are an exhaustive per-rate optimisation and are
    identical for every Linear of the same width at the same rung -- which, on
    a 288-expert MoE layer, is 864 units sharing one plan.
    """
    from tessera.alphabet import build_forest
    from tessera.grammar import bresenham_rate_schedule

    grid = _grid_for(family)
    # ``body_rate_q256`` is per POSITION; the trellis spends bits per CODE, and
    # a code covers ``arity`` positions.  Getting this factor wrong is silent:
    # it produces a legal artifact at the wrong rate.
    root = Fraction(body_rate_q256 * family.arity, 256)
    rates = bresenham_rate_schedule(root, n_columns, cap=grid.rate_cap)
    forests = {rate: build_forest(rate, grid=grid) for rate in sorted(set(rates))}
    return grid, rates, forests


def render_tessera_weight(
    weight: torch.Tensor,
    name: str,
    *,
    col_weights: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Encode ``weight`` at the rung ``name`` names and return the reconstruction.

    The returned tensor is what a Tessera artifact at this rung decodes to, so
    it is simultaneously the surrogate's error source, the KL validation's
    weight, and (once an exporter exists) the bytes' meaning -- the rendering
    identity principle 8 requires, established by construction rather than by
    keeping three code paths in step.

    ``weight`` is ``[out_features, in_features]``.  The trellis runs down
    columns and a k-tuple covers ``arity`` consecutive **rows**, so tuples are
    consecutive output channels at one input position, and the segment-2b
    scales run along the input axis in row-major order -- the same axis
    NVFP4's group-16 scales run along.
    """
    from tessera.decode import reconstruct_unit
    from tessera.encode import encode_unit
    from tessera.manifest import RotationState
    from tessera.trellis import ConvCode

    parsed = parse_tessera_format_name(name)
    if parsed is None:
        raise ValueError(f"{name!r} is not a Tessera format name")
    family, rung = parsed

    # Refuse rather than ignore.  The render contract passes ``col_weights``
    # for the imatrix-weighted families; Tessera's encoder does not consume it
    # yet, and silently dropping it would price a lever that was never applied
    # -- the same failure shape as an activation dict keyed by the wrong name.
    if col_weights is not None:
        raise NotImplementedError(
            "Tessera render does not consume col_weights yet; it must not be "
            "silently ignored. Add imatrix weighting to encode_unit first."
        )

    if weight.ndim != 2:
        raise ValueError(
            f"{name}: Tessera renders a 2-D Linear, got shape {tuple(weight.shape)}. "
            "A packed 3-D expert stack is rendered per expert, which is how "
            "every other format already keys MoE units."
        )
    rows, cols = weight.shape
    grid, rates, forests = _plan(family, rung, cols)
    if rows % grid.arity:
        raise ValueError(
            f"{name}: {rows} output features is not a whole number of arity-"
            f"{grid.arity} tuples. The rung is legal; this shape cannot carry it."
        )

    unit = encode_unit(
        weight,
        forests,
        rates,
        ConvCode(memory=TESSERA_CONV_MEMORY),
        rotation=RotationState.NONE,
        with_diagonals=False,
        completion=0,
        group=TESSERA_GROUP,
        half=TESSERA_HALF,
    )
    out = reconstruct_unit(unit, forests, ConvCode(memory=TESSERA_CONV_MEMORY))
    return out.to(dtype=weight.dtype, device=weight.device)


def tessera_quantize_dequantize(name: str):
    """A one-argument RTN-shaped callable, for ``FormatSpec.quantize_dequantize``.

    Tessera *is* the quantizer -- there is no post-hoc error compensation to
    layer on top, which is precisely the "format-first over GPTQ compensation"
    preference: choosing the right ``(format, transform)`` beats correcting a
    wrong one afterwards.  So the registry callable is the whole render.
    """

    def _qdq(w: torch.Tensor) -> torch.Tensor:
        return render_tessera_weight(w, name)

    return _qdq


def synthesize_tessera_spec(name: str):
    """Build a ``FormatSpec`` for a Tessera rung on demand, or return None.

    Returning None rather than raising for a non-Tessera name is what lets
    ``get_format`` use this as a fallback without reordering its own error
    handling: an unknown name must still produce ``get_format``'s KeyError,
    naming the registry, not a Tessera parse failure.
    """
    from . import format_registry as fr
    from .tessera_formats import artifact_bpp

    parsed = parse_tessera_format_name(name)
    if parsed is None:
        return None
    family, rung = parsed

    bpp = artifact_bpp(family, rung)
    return fr.FormatSpec(
        name=name,
        # ``weight_bits`` is the integer field the accountant reads; Tessera's
        # rate is fractional by construction, so the exact value travels in
        # ``exact_bits_per_param`` and this is the ceiling for anything that
        # wants one number. Reporting a floor here would under-count every
        # artifact.
        weight_bits=-(-bpp.numerator // bpp.denominator),
        group_size=TESSERA_GROUP,
        scale_bits=8,
        scale_dtype_name="uint8_e8m0",
        weight_element_dtype=f"tessera_{family.base.lower()}_k{family.arity}",
        act_bits=None,               # W-only: the body decodes to bf16 weights
        act_dtype_name=None,
        family=family.name,
        min_capability_sm=80,
        quantize_dequantize=tessera_quantize_dequantize(name),
        # Producer-eligibility is the AND of two independent gates, and
        # conflating them is how a rung reaches the DP that cannot be written:
        #   (a) the wire can carry it -- the grid's digest is a permanent
        #       commitment in SERIALISABLE_GRIDS, which E4M3 is not;
        #   (b) a pinned runtime executes it -- the kernel lane has no vLLM
        #       backend yet, so this is False for every rung today (principle 9).
        # (a) is per-rung and settled here; (b) is one flag so that flipping it
        # behind an attested route CANNOT silently admit an unwritable rung.
        producer_eligible=(
            _TESSERA_SERVING_LANE_EXISTS and tessera_rung_is_serialisable(name)
        ),
        # The whole rate, body and scale planes together -- which is what
        # ``artifact_bpp`` computes.  Without this the generic accountant
        # charges ``ceil(bpp)`` plus a *second* group-scale term on top of a
        # number that already includes the scales: R896 priced at 4.25 bpp
        # against an artifact the exporter's byte-exact accountant measures at
        # 4.002.  The allocator would have been ranking Tessera against NVFP4
        # on a 6% overcharge it invented.
        exact_bits_per_param=bpp,
    )
