"""Opt-in seam that puts the continuous trellis rate surface in the DP's menu.

WHAT THIS IS
------------
``trellis_formats`` / ``trellis_footprint`` / ``trellis_allocator`` /
``trellis_rate_surface`` are a complete, exactly-priced research surface that
nothing in the pipeline imported.  This module is the ONE place the production
allocator reaches them, and it is off unless ``PRISMAQUANT_TRELLIS_SURFACE``
names a manifest file.  Unset, :func:`augment_candidates` returns its input
unchanged and the run is byte-identical to one built without this module
(principle 6, the ``PRISMAQUANT_FISHER_CAP_MULTIPLIER`` precedent).

STATUS: WIRED FOR ALLOCATION, NEVER EXPORTED
---------------------------------------------
:func:`build_trellis_menu` produces a correctly priced menu and
:func:`augment_candidates` installs it when the manifest flag is set.  Exact
bytes travel in ``_memory_bytes_by_format``; the payload and footprint paths
prefer that map, the rank table is extended from candidates' exact serialized
rates, and the run's attested objective plus surface provenance travel with
the assignment.  Fused and packed aggregation build from their members'
menus; real ``allocator.main()`` tests drive both paths through the real
solver and kill rung-loss, byte-loss and dloss-loss mutations.  Packed parent
bytes use the resolved architecture profile's native-export decomposition:
one rank-2 wire per expert and projection, never a flattened rank-3 guess.
Each packed anchor must additionally carry a typed ``packed_parent`` contract
that declares ``dloss_scope=whole_packed_parent`` and repeats that exact wire
decomposition.  Only bytes multiply by expert/projection count; the measured
parent loss is applied once.  Missing or contradictory scope refuses before a
rung can enter the menu.

The currency entry is different: no plumbing turns a weighted-SSE anchor into
an AURA-priced one, so :func:`_require_run_currency` remains a genuine
dW-supply refusal.  Rendering and export remain unavailable as well.

WHY NO TCQ SPEC ANSWERS FOR BYTES
---------------------------------
The obvious alternative -- register a minimal TCQ ``FormatSpec`` so every
site that resolves a format through the registry just works -- is not
available, and not as a matter of taste.  ``FormatSpec.memory_bytes_for_shape``
is a closed form over ``weight_bits`` / ``scale_bits`` / ``group_size``, while
a rung's exact size needs the layout, the per-column schedule and the alphabet
directory (``trellis_footprint.trellis_tensor_payload_breakdown``: 16-byte row
stride alignment, an 88-byte wire header, a nibble schedule plane, block
offsets under ``tight_offsets``, and a family-specific scale plane).  Those are
per-campaign manifest data, so ``(name, shape)`` does not determine the bytes
and a registered spec could only be plausible and WRONG -- silently consumed
by ``_serialized_format_rates``, ``footprint`` and the payload filter, which
is precisely the failure this module exists to prevent.  It would also expose
TCQ to every ``quantize_dequantize`` consumer with no render behind it -- which
is why that helper still refuses, while ``act_quant_changes_input`` answers,
since the executed contract is a fact about the name.  Exact bytes therefore
ride the ``Candidate`` and ``_memory_bytes_by_format`` -- the repo's existing
single mechanism.  ``fr.get_format('TCQ_...')`` parse-resolves an
exact-or-refuse ``TrellisFormatSpec`` that is never inserted into ``REGISTRY``;
it answers what a rung EXECUTES and refuses what it WEIGHS.  The cost is N
pointed refusals instead of one closed form; each one refuses where a closed
form would have guessed.

WHY A MANIFEST AND NOT A ``FORMATS`` ENUM ENTRY
----------------------------------------------
A trellis rung is not an enum value.  It is ``(family, body_rate_q256,
layout, schedule, alphabets)``, and the wire's rate resolution is
``256/columns`` q256 -- effectively continuous.  What makes a rung *cost*
something is a measured anchor, and anchors are per-campaign data, not a
constant in the source tree.  So the flag names a file of measured anchors and
this module densifies them.  The names the DP and ``layer_config.json`` see
are the closed ``TCQ_{E2M1,E4M3}_R<q256>`` spelling that
``trellis_formats.parse_trellis_format_name`` round-trips.

THE FOUR THINGS THIS REFUSES, AND WHY
-------------------------------------
1. **A profile with no ``target_platform``.**  ``trellis_allocator``'s
   ``_capability_gate`` returns *legal* when the profile declares no exact
   platform (:578-586) -- deliberately, because admission is then the
   experiment's responsibility.  Six of nine shipped serving-profile specs
   take that branch, so wiring the surface to one of them would run the
   allocator against a capability gate that cannot fire.  A vacuous gate is
   worse than no gate: it looks like a check.  So the manifest must name a
   profile that declares ``target_platform``, and E2M1 (SM120+) / E4M3
   (SM89+) then really are compared against it.

2. **An objective the run is not pricing in.**  The manifest declares
   ``cost_mode`` and ``currency``; both must match the run.  A DP that ranks
   an AURA-priced trellis rung against an output-MSE-priced NVFP4 rung is not
   solving any stated objective.

3. **An activation contract the anchors were not measured under.**  The
   hull anchors that produced the measured surface were priced ``W*A16``
   while the native ``_scaled_mm`` routes for both families are ``A=W``
   (W8A8 for E4M3, W4A4 for E2M1).  Rendering identity without execution
   identity is what priced a real A-side at zero once already (NVFP4_CB,
   2026-08-17), so the manifest must state the contract its dloss numbers
   were measured under and it is stamped on every candidate's provenance.
   Where the pinned runtime publishes no attestation table -- which is the
   case at the producer pin today -- the lane resolves ``unattested`` and
   that word travels with the candidate rather than being rounded to
   ``backed``.

4. **A packed parent with no exact architecture-owned wire decomposition and
   no typed parent-level loss scope.**  The trellis footprint describes one
   rank-2 matrix, while packed probe rows are rank 3.  ``allocator.main()``
   threads its resolved ``ModelProfile``; the seam reuses the same
   ``split_packed_experts_for_format`` and
   ``packed_expert_projection_names`` policy native export uses, and prices
   one wire per expert/projection.  The campaign manifest must independently
   bind each packed point to that exact repeated-wire recipe and declare
   ``dloss_scope=whole_packed_parent``.  A missing/default profile, an
   undeclared packed parameter, a non-splitting format, an indivisible
   projection, a missing typed contract, or a per-wire loss declaration raises
   ``TrellisPackedExpertLayoutError``.  Flattening or multiplying one arbitrary
   wire by ``num_experts`` would give different answers at the campaign's
   4.0117-bpw boundary, while reusing a per-wire loss once would underprice
   quality by the wire count, so neither is a fallback.  This is a research
   allocation recipe, not a Gridbook/runtime attestation; export remains
   explicitly disabled below.

WHAT THIS DOES NOT DO
---------------------
It does not render, export, or serve.  ``ProductionWeightCache`` has no
trellis mechanism and ``export_native_compressed`` refuses a TCQ assignment
outright.  This is allocation-time reach only: it lets the DP see the surface,
report what it would choose, and price the choice in exact serialized bytes.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path

from .allocator_solver import Candidate, _shape_from_stats
from .trellis_allocator import TrellisAllocatorCandidate
from .trellis_formats import (
    ALL_LEGAL_TRELLIS_FORMAT_NAMES,
    LAYOUT_TIGHT_OFFSETS,
    LAYOUTS,
    SUPERBLOCK_WEIGHTS,
    TrellisFormatError,
    get_trellis_family,
)

#: Spelling guard for the name the DP and ``layer_config.json`` will carry.
_LEGAL_FORMAT_NAMES = frozenset(ALL_LEGAL_TRELLIS_FORMAT_NAMES)
from .trellis_rate_surface import (
    densify_rate_surface,
    fit_rate_surface,
)

#: Names a JSON manifest of measured anchors.  Unset is the default and is a
#: byte-identical no-op.
TRELLIS_SURFACE_ENV = "PRISMAQUANT_TRELLIS_SURFACE"

TRELLIS_SURFACE_MANIFEST_SCHEMA = "prismaquant.trellis_surface_manifest.v1"
TRELLIS_MENU_PROVENANCE_SCHEMA = "prismaquant.trellis_menu_provenance.v1"
TRELLIS_PACKED_WIRE_RECIPE_SCHEMA = (
    "prismaquant.trellis_packed_wire_recipe.v1"
)
TRELLIS_PACKED_PARENT_ANCHOR_SCHEMA = (
    "prismaquant.trellis_packed_parent_anchor.v1"
)
PACKED_PARENT_DLOSS_SCOPE = "whole_packed_parent"
PACKED_WIRE_DECOMPOSITION = (
    "model_profile_per_expert_per_projection_rank2"
)

#: Rungs densified per unit when the manifest does not say.  The anchors
#: themselves are always included on top of this count.
DEFAULT_RUNGS_PER_UNIT = 16


class TrellisMenuError(RuntimeError):
    """The surface manifest cannot be admitted to this run."""


class TrellisSeamUnwiredError(TrellisMenuError):
    """The production seam is enabled but the DP cannot honour a TCQ rung."""


class TrellisPackedExpertLayoutError(TrellisMenuError):
    """A packed row has no exact architecture-owned 2-D wire decomposition."""


#: The links whose production behaviour or required input remains unwired.
#: This is the live
#: re-enable checklist cited by ``docs/ARCHITECTURE.md`` §4.9.  Delete an entry
#: only when a test drives the entry point and exercises the behaviour it names
#: -- code that merely looks present does not license deletion.
#:
#: Closed entries and the tests that license their deletion:
#:   * registry resolution (#1): ``tests/test_trellis_format_spec.py``;
#:   * exact assignment-payload and target-disk byte paths (#2/#6):
#:     ``tests/test_trellis_byte_budget_path.py``;
#:   * named promotion-rank refusal plus exact candidate-rate extension (#5):
#:     ``tests/test_allocator_cost_mode_and_rank.py``;
#:   * fused-sibling and packed-expert aggregation (#3/#4), including exact
#:     summed bytes/dloss and the architecture-owned packed wire layout:
#:     ``tests/test_trellis_aggregation_entrypoint.py``;
#:   * attested cost mode and surface provenance at the allocator call site
#:     (#7): ``tests/test_allocator_cost_mode_and_rank.py``.
UNWIRED_LINKS: tuple[tuple[str, str], ...] = (
    ("trellis_rate_surface.py:43-52",
     "the anchors' currency is weighted SSE under a per-input-channel "
     "activation second moment -- an output-MSE proxy, explicitly NOT the "
     "AURA KL-adjoint the production DP ranks in; aura_cost's two dW sources "
     "(ProductionWeightCache and fr.get_format(...).quantize_dequantize, "
     ":609-654) both require a registered format, so AURA-priced anchors are "
     "a dW-supply problem, not an objective change"),
)


#: ``COST_MODE`` is a spelling over ``COST_RENDER x COST_OBJECTIVE`` (re-vet
#: R3).  The objective half IS the currency a ``predicted_dloss`` is
#: denominated in, so the run's currency is a definitional function of its
#: attested cost mode -- not a threshold anyone picks (principle 2).  This is
#: the same case block ``run-pipeline.sh`` resolves the axes with (its
#: ``case "$COST_MODE"``); a mode outside it has no declared objective and is
#: refused rather than defaulted.
COST_MODE_OBJECTIVE_CURRENCY: Mapping[str, str] = {
    "local": "weight-recon",
    "production-render-score": "render-score",
    "production-render": "render-score",
    "aura": "aura-adjoint",
}


@dataclass(frozen=True)
class TrellisSurfaceManifest:
    """A campaign's measured trellis anchors, plus what they were measured on."""

    path: Path
    #: SHA-256 of the manifest bytes. The IDENTITY that travels with the
    #: assignment: a path is a name that can be rewritten, moved, or point at
    #: different bytes on the machine that reads the layer_config.
    sha256: str
    cost_mode: str
    currency: str
    target_profile: str
    activation_contract: str
    layout: str
    rungs_per_unit: int
    #: unit name -> {"family", "alphabets": {rate: codes}, "points": [...],
    #:               "packed_parent": {...} when the probe row is packed}
    anchors: Mapping[str, Mapping[str, object]]
    provenance: Mapping[str, object]


def _require(payload: Mapping[str, object], field: str) -> object:
    if field not in payload:
        raise TrellisMenuError(
            f"trellis surface manifest is missing required field {field!r}"
        )
    return payload[field]


def load_manifest(path: str | os.PathLike[str]) -> TrellisSurfaceManifest:
    """Parse and structurally validate a surface manifest.

    Every field is required.  There is no default for ``activation_contract``
    or ``target_profile`` on purpose: a defaulted answer to "what did you
    measure this on" is indistinguishable from a wrong one.
    """

    resolved = Path(path)
    try:
        raw = resolved.read_bytes()
    except FileNotFoundError as exc:
        raise TrellisMenuError(
            f"{TRELLIS_SURFACE_ENV}={resolved} does not exist"
        ) from exc
    try:
        payload = json.loads(raw)
    except FileNotFoundError as exc:
        raise TrellisMenuError(
            f"{TRELLIS_SURFACE_ENV}={resolved} does not exist"
        ) from exc
    except json.JSONDecodeError as exc:
        raise TrellisMenuError(
            f"{TRELLIS_SURFACE_ENV}={resolved} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise TrellisMenuError("trellis surface manifest must be an object")
    schema = payload.get("schema")
    if schema != TRELLIS_SURFACE_MANIFEST_SCHEMA:
        raise TrellisMenuError(
            f"trellis surface manifest schema {schema!r} is not "
            f"{TRELLIS_SURFACE_MANIFEST_SCHEMA!r}"
        )
    layout = str(_require(payload, "layout"))
    if layout not in LAYOUTS:
        raise TrellisMenuError(
            f"layout {layout!r} is not one of {sorted(LAYOUTS)}"
        )
    anchors = _require(payload, "anchors")
    if not isinstance(anchors, dict) or not anchors:
        raise TrellisMenuError("manifest 'anchors' must be a non-empty object")
    rungs = int(payload.get("rungs_per_unit", DEFAULT_RUNGS_PER_UNIT))
    if rungs < 2:
        raise TrellisMenuError("rungs_per_unit must be at least 2")
    return TrellisSurfaceManifest(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        cost_mode=str(_require(payload, "cost_mode")),
        currency=str(_require(payload, "currency")),
        target_profile=str(_require(payload, "target_profile")),
        activation_contract=str(_require(payload, "activation_contract")),
        layout=layout,
        rungs_per_unit=rungs,
        anchors=anchors,
        provenance=dict(payload.get("provenance") or {}),
    )


def _profile_declares_platform(profile_id: str) -> str:
    """Return the profile's exact target platform, or refuse.

    This is the whole reason the manifest names a profile rather than
    inheriting the run's.  ``_capability_gate`` cannot fire without one, and a
    capability gate that cannot fire is provenance, not a gate (principle 9).
    """

    from .serving_profiles import load_serving_profile

    try:
        profile = load_serving_profile(profile_id)
    except FileNotFoundError as exc:
        raise TrellisMenuError(
            f"trellis surface names serving profile {profile_id!r}, which "
            f"does not exist"
        ) from exc
    platform = getattr(profile, "target_platform", None)
    if not platform:
        raise TrellisMenuError(
            f"serving profile {profile_id!r} declares no 'target_platform', "
            f"so trellis_allocator._capability_gate returns legal for every "
            f"family without comparing anything (E2M1 needs SM120+, E4M3 "
            f"SM89+). Refusing to allocate against a gate that cannot fire. "
            f"Name a profile that declares the exact hardware the anchors "
            f"were measured on."
        )
    return str(platform)


def _require_run_currency(manifest: TrellisSurfaceManifest,
                          cost_mode: str) -> str:
    """Refuse a surface whose anchors are not in the run's currency.

    This is the surviving entry of :data:`UNWIRED_LINKS` and the one refusal
    an enabled seam still hits in practice.  It is not plumbing: the ladder's
    measured anchors are weighted SSE under a per-input-channel activation
    second moment -- an output-MSE proxy -- while an ``aura`` run's DP ranks
    the KL-adjoint.  A DP that ranks a weighted-SSE rung against an
    AURA-priced NVFP4 rung is not solving any stated objective, and no amount
    of wiring changes what the anchors measured.  Producing AURA-priced
    anchors is a dW-supply problem (``aura_cost``'s two dW sources both need
    a registered format), owned elsewhere.

    Returns the run's objective currency on success.
    """

    if not cost_mode:
        raise TrellisMenuError(
            "the cost table carries no provenance['cost_mode'] stamp "
            "(re-vet R2), so this run's objective is unknown and the "
            "manifest's declared currency cannot be checked against it. "
            "Re-run the cost stage with --cost-mode; an unstamped table is "
            "refused rather than compared against a default."
        )
    expected = COST_MODE_OBJECTIVE_CURRENCY.get(cost_mode)
    if expected is None:
        raise TrellisMenuError(
            f"COST_MODE={cost_mode!r} names no objective in "
            f"COST_MODE_OBJECTIVE_CURRENCY "
            f"({sorted(COST_MODE_OBJECTIVE_CURRENCY)}), so the currency its "
            f"predicted_dloss is denominated in is undeclared. A trellis "
            f"surface cannot be admitted to a run whose objective has no name."
        )
    if manifest.currency != expected:
        raise TrellisSeamUnwiredError(
            f"trellis surface declares currency {manifest.currency!r}, but "
            f"COST_MODE={cost_mode!r} prices in {expected!r}. This is "
            f"the UNWIRED_LINKS currency entry and it is not a plumbing gap: "
            f"the measured "
            f"trellis anchors are weighted SSE under a per-input-channel "
            f"activation second moment (trellis_rate_surface.py:43-52), an "
            f"output-MSE proxy, NOT the AURA KL-adjoint the production DP "
            f"ranks in. Ranking rungs measured under one objective against "
            f"candidates priced under another solves neither. AURA-priced "
            f"anchors are a dW-supply problem (aura_cost's two dW sources "
            f"both require a registered format), not a flag."
        )
    return expected


def _achievable_q256(columns: int, family, low: int, high: int,
                     count: int) -> list[int]:
    """Rungs a real per-column schedule can hit, spread across the envelope.

    The wire carries one 4-bit rate code per input column shared across rows,
    so the achievable tensor totals are the integers and the rate resolution
    is ``SUPERBLOCK_WEIGHTS/columns`` q256.  Anchors are always kept: they are
    the only rungs that were measured rather than interpolated.
    """

    spec = get_trellis_family(family)
    lowest = max(low, SUPERBLOCK_WEIGHTS)
    highest = min(high, spec.bypass_rate * SUPERBLOCK_WEIGHTS)
    if highest < lowest:
        return []
    if count <= 1:
        return [lowest]
    step = (highest - lowest) / (count - 1)
    return sorted({
        int(round(lowest + index * step)) for index in range(count)
    })


def _unit_candidates(
    unit_name: str,
    shape: Sequence[int],
    entry: Mapping[str, object],
    manifest: TrellisSurfaceManifest,
    *,
    qname: str | None,
    packed_expert: bool | None,
) -> tuple[TrellisAllocatorCandidate, ...]:
    family = str(_require(entry, "family"))
    raw_alphabets = _require(entry, "alphabets")
    if not isinstance(raw_alphabets, dict):
        raise TrellisMenuError(f"{unit_name}: 'alphabets' must be an object")
    alphabets = {
        int(rate): [int(code) for code in codes]
        for rate, codes in raw_alphabets.items()
    }
    points = _require(entry, "points")
    if not isinstance(points, list) or len(points) < 2:
        raise TrellisMenuError(
            f"{unit_name}: a rate surface needs at least two measured anchors; "
            f"one anchor cannot bracket anything and this module refuses to "
            f"extrapolate"
        )
    dims = tuple(int(value) for value in shape)
    if len(dims) != 2:
        raise TrellisMenuError(f"{unit_name}: shape must be rank 2, got {dims}")
    columns = dims[1]
    if columns % SUPERBLOCK_WEIGHTS:
        raise TrellisMenuError(
            f"{unit_name}: {columns} input columns is not a multiple of "
            f"{SUPERBLOCK_WEIGHTS}; a short final superblock is legal on the "
            f"wire but its rate accounting is the campaign's to declare"
        )

    from .trellis_rate_surface import uniform_column_schedule

    anchor_records = []
    from .trellis_allocator import build_trellis_allocator_candidate
    for point in sorted(points, key=lambda p: int(p["q256"])):
        rate = int(point["q256"])
        schedule = uniform_column_schedule(columns, rate, family=family)
        used = {
            value for value in schedule
            if value < get_trellis_family(family).bypass_rate
        }
        anchor_records.append(
            build_trellis_allocator_candidate(
                unit_name,
                dims,
                family=family,
                body_rate_q256=rate,
                layout=manifest.layout,
                schedule=schedule,
                alphabets={key: alphabets[key] for key in sorted(used)},
                predicted_dloss=float(point["dloss"]),
                predicted_dloss_stderr=float(point.get("stderr", 0.0)),
                target_profile=manifest.target_profile,
                qname=qname,
                packed_expert=packed_expert,
                variant_label="measured",
            )
        )
    surface = fit_rate_surface(anchor_records, currency=manifest.currency)
    low, high = surface.q256_range
    rungs = _achievable_q256(
        columns, family, low, high, manifest.rungs_per_unit,
    )
    rungs = sorted(set(rungs) | set(surface.anchor_q256))
    rungs = [rate for rate in rungs if low <= rate <= high]
    return densify_rate_surface(
        surface,
        dims,
        q256_values=rungs,
        alphabets=alphabets,
        target_profile=manifest.target_profile,
        qname=qname,
        packed_expert=packed_expert,
    )


def _packed_expert_wire_plan(
    unit_name: str,
    shape: Sequence[int],
    model_profile,
    packed_param: object = None,
) -> tuple[tuple[int, int], tuple[str, ...], int, str]:
    """Resolve the export-owned physical wires for one packed parent.

    Packed probe rows are rank 3, but the trellis wire is rank 2.  The model
    profile already owns the exact bridge: native export consults
    ``split_packed_experts_for_format`` and
    ``packed_expert_projection_names`` before emitting one 2-D tensor per
    expert and projection.  Reusing that contract keeps allocation bytes and
    the architecture's on-disk decomposition identical.  Flattening the
    parent would erase hundreds of wire headers and alphabet directories and
    is materially wrong at the campaign's 4.0117-bpw boundary.

    The split decision itself is checked per concrete TCQ name after the
    rungs are built.  This helper resolves the shape shared by those rungs.
    """

    dims = tuple(int(value) for value in shape)
    if len(dims) != 3 or min(dims) <= 0:
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: packed-expert trellis pricing requires a positive "
            f"rank-3 source shape, got {dims}"
        )
    profile_name = str(getattr(model_profile, "name", ""))
    if model_profile is None or profile_name == "default":
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: packed-expert source shape {dims} cannot be priced "
            "without a non-default resolved ModelProfile. The trellis wire "
            "is rank 2, and flattening a packed parent or merely multiplying "
            "one wire by num_experts would guess its header/alphabet "
            "multiplicity. Drive allocator.main() with a detectable model "
            "profile or --model-override."
        )

    projection_fn = getattr(
        model_profile, "packed_expert_projection_names", None)
    split_fn = getattr(model_profile, "split_packed_experts_for_format", None)
    param_names_fn = getattr(model_profile, "packed_expert_param_names", None)
    if (
        not callable(projection_fn)
        or not callable(split_fn)
        or not callable(param_names_fn)
    ):
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: model profile "
            f"{getattr(model_profile, 'name', type(model_profile).__name__)!r} "
            "does not expose the packed-expert projection/split contract "
            "native export uses"
        )

    param_name = (
        str(packed_param)
        if isinstance(packed_param, str) and packed_param
        else unit_name.rsplit(".", 1)[-1]
    )
    try:
        declared_params = tuple(str(value) for value in param_names_fn())
    except Exception as exc:
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: model profile could not enumerate packed "
            f"parameters: {type(exc).__name__}: {exc}"
        ) from exc
    if param_name not in declared_params:
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: packed parameter {param_name!r} is not declared by "
            f"model profile {profile_name!r}; declared parameters are "
            f"{declared_params!r}"
        )
    try:
        projections = tuple(str(value) for value in projection_fn(param_name))
    except Exception as exc:
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: model profile could not resolve packed projection "
            f"names for {param_name!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if not projections or any(not value for value in projections):
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: model profile returned no usable projection names "
            f"for packed parameter {param_name!r}"
        )
    if len(set(projections)) != len(projections):
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: model profile returned duplicate packed projection "
            f"names {projections!r}"
        )

    experts, packed_rows, columns = dims
    if packed_rows % len(projections):
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: {packed_rows} packed rows cannot split evenly "
            f"across declared projections {projections!r}; native export "
            "would refuse the same decomposition"
        )
    projection_shape = (packed_rows // len(projections), columns)
    return projection_shape, projections, experts, param_name


def _validate_packed_parent_anchor(
    unit_name: str,
    entry: Mapping[str, object],
    *,
    source_shape: Sequence[int],
    projection_shape: Sequence[int],
    projection_names: Sequence[str],
    experts: int,
    packed_param: str,
    model_profile,
) -> dict[str, object]:
    """Bind packed-parent dloss anchors to the exact repeated-wire recipe.

    The rate-surface point belongs to the WHOLE packed parent exactly once;
    only bytes multiply by expert/projection wire count.  Without this typed
    statement a per-wire ``dloss`` is indistinguishable from a parent-level
    one and would be underpriced by ``E * P``.  The manifest is the campaign's
    measurement contract, so it must also repeat the profile-derived physical
    decomposition and match it field for field.
    """

    raw = entry.get("packed_parent")
    if not isinstance(raw, Mapping):
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: packed manifest anchor is missing the typed "
            f"'packed_parent' contract ({TRELLIS_PACKED_PARENT_ANCHOR_SCHEMA}). "
            f"Its points must declare dloss_scope={PACKED_PARENT_DLOSS_SCOPE!r} "
            "and the exact per-expert/per-projection wire decomposition; "
            "otherwise a per-wire loss would be silently underpriced as one "
            "whole parent."
        )
    expected_fields = {
        "schema",
        "dloss_scope",
        "wire_decomposition",
        "model_profile",
        "source_shape",
        "packed_param",
        "projection_names",
        "projection_shape",
        "wire_count",
    }
    observed_fields = {str(key) for key in raw}
    if observed_fields != expected_fields:
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: packed_parent fields must be exactly "
            f"{sorted(expected_fields)}, got {sorted(observed_fields)}"
        )

    profile_name = str(getattr(model_profile, "name", ""))
    expected: dict[str, object] = {
        "schema": TRELLIS_PACKED_PARENT_ANCHOR_SCHEMA,
        "dloss_scope": PACKED_PARENT_DLOSS_SCOPE,
        "wire_decomposition": PACKED_WIRE_DECOMPOSITION,
        "model_profile": profile_name,
        "source_shape": [int(value) for value in source_shape],
        "packed_param": packed_param,
        "projection_names": [str(value) for value in projection_names],
        "projection_shape": [int(value) for value in projection_shape],
        "wire_count": int(experts) * len(tuple(projection_names)),
    }
    for field, wanted in expected.items():
        observed = raw.get(field)
        if observed != wanted:
            detail = ""
            if field == "dloss_scope":
                detail = (
                    " A per-wire/per-projection point cannot be reused: this "
                    "path applies the point once to the whole packed parent."
                )
            raise TrellisPackedExpertLayoutError(
                f"{unit_name}: packed_parent.{field} must be {wanted!r}, "
                f"got {observed!r}.{detail}"
            )
    return expected


def _packed_solver_candidate(
    record: TrellisAllocatorCandidate,
    *,
    unit_name: str,
    source_shape: Sequence[int],
    projection_shape: Sequence[int],
    projection_names: Sequence[str],
    experts: int,
    packed_param: str,
    model_profile,
) -> Candidate:
    """Price one parent rung as its exact per-expert/projection wire set."""

    fmt = str(record.footprint["format"])
    split_fn = getattr(model_profile, "split_packed_experts_for_format")
    try:
        should_split = split_fn(fmt)
    except Exception as exc:
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: model profile could not decide the packed wire "
            f"layout for {fmt}: {type(exc).__name__}: {exc}"
        ) from exc
    if should_split is not True:
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: model profile "
            f"{getattr(model_profile, 'name', type(model_profile).__name__)!r} "
            f"does not split packed experts for {fmt}. The declared source "
            f"shape {tuple(source_shape)} would remain rank 3, but the "
            "trellis footprint contract is rank 2; refusing rather than "
            "flattening it and guessing the wire multiplicity."
        )

    projection_count = len(tuple(projection_names))
    wire_count = int(experts) * projection_count
    total_bytes = int(record.memory_bytes) * wire_count
    source_params = 1
    for value in source_shape:
        source_params *= int(value)
    component_params = (
        int(projection_shape[0])
        * int(projection_shape[1])
        * wire_count
    )
    if component_params != source_params:
        raise TrellisPackedExpertLayoutError(
            f"{unit_name}: packed wire decomposition covers "
            f"{component_params} params but source shape "
            f"{tuple(source_shape)} contains {source_params}"
        )

    recipe = {
        "schema": TRELLIS_PACKED_WIRE_RECIPE_SCHEMA,
        "format": fmt,
        "family": record.family,
        "body_rate_q256": record.body_rate_q256,
        "layout": record.layout,
        "source_shape": [int(value) for value in source_shape],
        "packed_param": packed_param,
        "projections": [
            {
                "name": str(name),
                "wire_shape": [int(value) for value in projection_shape],
                "wire_count": int(experts),
                "wire_payload_bytes": int(record.memory_bytes),
                "per_wire_pre_render_recipe_identity_sha256": (
                    record.pre_render_recipe_identity_sha256
                ),
            }
            for name in projection_names
        ],
        "experts": int(experts),
        "wire_count": wire_count,
        "total_wire_bytes": total_bytes,
        "identity_scope": (
            "physical per-expert/per-projection pre-render wire recipes and "
            "multiplicity; excludes encoded body and scale values"
        ),
    }
    encoded = json.dumps(
        recipe,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    serialized_identity = hashlib.sha256(encoded).hexdigest()
    base = record.to_solver_candidate()
    return replace(
        base,
        fmt=fmt,
        bits_per_param=8.0 * total_bytes / max(source_params, 1),
        memory_bytes=total_bytes,
        serialized_identity=serialized_identity,
    )


def build_trellis_menu(
    candidates: dict[str, list[Candidate]],
    stats: Mapping[str, Mapping[str, object]],
    *,
    cost_mode: str,
    manifest_path: str | None = None,
    provenance_out: dict | None = None,
    model_profile=None,
) -> dict[str, list[Candidate]]:
    """Add trellis rungs to an already-built menu, or return it unchanged.

    ``manifest_path`` defaults to ``PRISMAQUANT_TRELLIS_SURFACE``.  With
    neither set this is a no-op returning the same object, so a run without
    the flag executes exactly the code path it executed before this module
    existed.

    The DP is untouched: trellis rungs are ordinary multi-choice knapsack
    candidates, priced in exact serialized tensor-payload bytes.  Their
    ``fmt`` is the closed ``TCQ_<grid>_R<q256>`` name -- SHAPE-FREE on
    purpose, because ``aggregate_fused_siblings`` and
    ``aggregate_packed_serving_groups`` intersect member menus BY FORMAT NAME.
    ``TrellisAllocatorCandidate.allocator_key`` embeds the per-tensor
    pre-render recipe digest, which includes the shape, so using it directly
    would give q_proj and k_proj disjoint menus at identical rungs and
    silently collapse every fused group back to individual rows.  The recipe
    digest still travels, on ``serialized_identity``, where per-member layout
    identity belongs.
    """

    resolved_path = manifest_path or os.environ.get(TRELLIS_SURFACE_ENV)
    if not resolved_path:
        return candidates

    manifest = load_manifest(resolved_path)
    if manifest.cost_mode != cost_mode:
        raise TrellisMenuError(
            f"trellis surface was measured under COST_MODE="
            f"{manifest.cost_mode!r} but this run prices in {cost_mode!r}. "
            f"One DP prices in one currency; ranking rungs measured under two "
            f"objectives against each other solves neither."
        )
    run_currency = _require_run_currency(manifest, cost_mode)
    platform = _profile_declares_platform(manifest.target_profile)

    added = 0
    covered: list[str] = []
    skipped: dict[str, str] = {}
    packed_wire_layouts: dict[str, dict[str, object]] = {}
    for unit_name, entry in manifest.anchors.items():
        if unit_name not in candidates:
            skipped[unit_name] = "unit has no priced scalar menu in this run"
            continue
        stat = stats.get(unit_name)
        if not stat:
            skipped[unit_name] = "unit absent from probe stats"
            continue
        # The repo's own shape helper, not a hand-rolled 2-tuple: it returns
        # (num_experts, out, in) for a packed row, and pricing that row as
        # (out, in) underprices it by num_experts (128x on DSv4).  A silent
        # 128x underprice makes a rung look nearly free to the DP and the
        # seam would report it as "0 unit(s) skipped".
        shape = _shape_from_stats(dict(stat))
        if len(shape) < 2 or min(shape) <= 0:
            skipped[unit_name] = f"unusable shape {shape}"
            continue
        packed_expert = _is_packed_expert(stat)
        projection_shape: tuple[int, int] | None = None
        projection_names: tuple[str, ...] = ()
        experts = 0
        packed_param = ""
        packed_wire_layout: dict[str, object] | None = None
        unit_added = False
        candidate_shape = shape
        if packed_expert:
            projection_shape, projection_names, experts, packed_param = (
                _packed_expert_wire_plan(
                    unit_name,
                    shape,
                    model_profile,
                    stat.get("_packed_param"),
                )
            )
            candidate_shape = projection_shape
            declared_params = int(stat.get("n_params", 0) or 0)
            source_params = int(shape[0]) * int(shape[1]) * int(shape[2])
            if declared_params and declared_params != source_params:
                raise TrellisPackedExpertLayoutError(
                    f"{unit_name}: probe declares n_params={declared_params} "
                    f"but packed source shape {shape} contains "
                    f"{source_params}; exact bytes need one physical census"
                )
            packed_anchor = _validate_packed_parent_anchor(
                unit_name,
                entry,
                source_shape=shape,
                projection_shape=projection_shape,
                projection_names=projection_names,
                experts=experts,
                packed_param=packed_param,
                model_profile=model_profile,
            )
            packed_wire_layout = {
                **packed_anchor,
                "experts": experts,
            }
        try:
            records = _unit_candidates(
                unit_name,
                candidate_shape,
                entry,
                manifest,
                qname=unit_name,
                packed_expert=True if packed_expert else None,
            )
        except (TrellisFormatError, TrellisMenuError, KeyError) as exc:
            skipped[unit_name] = f"{type(exc).__name__}: {exc}"
            continue

        seen: set[str] = {cand.fmt for cand in candidates[unit_name]}
        for record in records:
            if not record.servability.legal:
                continue
            fmt = str(record.footprint["format"])
            if fmt not in _LEGAL_FORMAT_NAMES:
                # The name is the cross-module contract: layer_config.json
                # stores it and parse_trellis_format_name must read it back.
                # A drift in TrellisFamily.format_name would otherwise ship a
                # recipe nothing downstream can parse.
                raise TrellisMenuError(
                    f"{unit_name}: {fmt!r} is not in the closed trellis "
                    f"format vocabulary; TrellisFamily.format_name and "
                    f"parse_trellis_format_name have drifted apart"
                )
            if fmt in seen:
                raise TrellisMenuError(
                    f"{unit_name}: duplicate candidate format {fmt!r}; a unit "
                    f"cannot offer one rung twice under one manifest"
                )
            seen.add(fmt)
            if packed_expert:
                assert projection_shape is not None
                cand = _packed_solver_candidate(
                    record,
                    unit_name=unit_name,
                    source_shape=shape,
                    projection_shape=projection_shape,
                    projection_names=projection_names,
                    experts=experts,
                    packed_param=packed_param,
                    model_profile=model_profile,
                )
            else:
                base = record.to_solver_candidate()
                cand = replace(base, fmt=fmt)
            candidates[unit_name].append(cand)
            unit_added = True
            if packed_wire_layout is not None:
                # Stamp only a layout that actually contributed a legal rung.
                # A planned layout whose candidates were skipped must not look
                # like part of the menu in the assignment provenance.
                packed_wire_layouts.setdefault(
                    unit_name, packed_wire_layout)
            # The SAME exact-bytes channel build_candidates writes for a
            # FormatSpec row (allocator_candidates:1950), and the reason no
            # TCQ FormatSpec is needed: a rung's serialized size is not a
            # function of (name, shape) -- it needs the layout, the
            # per-column schedule plane and the alphabet directory this
            # manifest declares (trellis_tensor_payload_breakdown) -- so the
            # bytes travel from where they were computed instead of being
            # recomputed from a closed form that cannot express them. Every
            # downstream byte path already PREFERS this map over the registry
            # (allocator's payload filter, footprint, compute_achieved,
            # kl_measurement, bit attribution), so writing it is what wires
            # them, and one mechanism stays one mechanism (principle 8).
            if isinstance(stat, MutableMapping):
                stat.setdefault(
                    "_memory_bytes_by_format", {})[fmt] = int(cand.memory_bytes)
                if cand.serialized_identity is not None:
                    stat.setdefault(
                        "_serialized_identity_by_format",
                        {})[fmt] = cand.serialized_identity
                    stat.setdefault(
                        "_serialized_sidecar_identity_by_format",
                        {})[fmt] = cand.serialized_sidecar_identity
            else:
                raise TrellisMenuError(
                    f"{unit_name}: stats entry is a read-only "
                    f"{type(stat).__name__}, so the rung's exact bytes cannot "
                    f"be recorded in '_memory_bytes_by_format'. Every "
                    f"downstream byte path reads them from there and no "
                    f"FormatSpec can recompute them; a menu whose bytes "
                    f"cannot be recorded must not be built."
                )
            added += 1
        if unit_added:
            covered.append(unit_name)

    payload = {
        "schema": TRELLIS_MENU_PROVENANCE_SCHEMA,
        "manifest_path": str(manifest.path),
        # IDENTITY, not location: a consumer holding the layer_config can
        # check it has the same anchors, on a machine where the path means
        # nothing (principle 12).
        "manifest_sha256": manifest.sha256,
        "cost_mode": manifest.cost_mode,
        "currency": manifest.currency,
        # The objective this RUN prices in, resolved from the cost table's
        # own provenance stamp. Recorded next to the manifest's declared
        # currency so the equality the gate enforced is auditable from the
        # assignment alone rather than re-derived.
        "run_objective_currency": run_currency,
        "target_profile": manifest.target_profile,
        "target_platform": platform,
        # The contract the anchors' dloss was MEASURED under. The hull that
        # produced these numbers priced W*A16 while both families' native
        # _scaled_mm routes are A=W; stamping it here is what stops a future
        # A=W lane from silently inheriting a W*A16 loss.
        "anchor_activation_contract": manifest.activation_contract,
        "layout": manifest.layout,
        "rungs_per_unit": manifest.rungs_per_unit,
        "units_covered": len(covered),
        "units_in_menu": len(candidates),
        "candidates_added": added,
        "units_skipped": skipped,
        "packed_wire_layouts": packed_wire_layouts,
        "research_only": True,
        "exportable": False,
        "export_note": (
            "export_native_compressed refuses a TCQ assignment: no production "
            "render mechanism exists and the producer pin publishes no "
            "activation-contract attestation table for these lanes."
        ),
        "anchor_provenance": dict(manifest.provenance),
    }
    if provenance_out is not None:
        provenance_out.update(payload)
    print(
        f"[alloc] trellis surface: +{added} rungs on {len(covered)}/"
        f"{len(candidates)} units from {manifest.path.name} "
        f"(profile={manifest.target_profile} platform={platform} "
        f"anchors@{manifest.activation_contract}); "
        f"{len(skipped)} unit(s) skipped",
        flush=True,
    )
    if skipped:
        for unit_name in sorted(skipped)[:5]:
            print(f"[alloc]   skip {unit_name}: {skipped[unit_name]}",
                  flush=True)
    return candidates


def _is_packed_expert(stat: Mapping[str, object]) -> bool:
    """The repo's packed-expert detector, imported late to avoid a cycle.

    ``allocator_candidates`` imports this module, so the import cannot be at
    module scope.  Reading ``stat["packed_expert"]`` instead -- as this seam
    did until 2026-08-29 -- tests a key nothing in the probe-stats path ever
    writes, so the guard was always falsy.
    """

    from .allocator_candidates import _stats_indicates_packed_expert

    return _stats_indicates_packed_expert(dict(stat))


def augment_candidates(
    candidates: dict[str, list[Candidate]],
    stats: Mapping[str, Mapping[str, object]],
    *,
    cost_mode: str,
    manifest_path: str | None = None,
    provenance_out: dict | None = None,
    model_profile=None,
) -> dict[str, list[Candidate]]:
    """The production seam: a no-op when unset, a built menu when set.

    Unset, this returns its input object unchanged and the run is
    byte-identical to one built without this module -- that half was always
    real and is what ships.

    Set, it builds the menu.  Exact-byte, rank, call-site and fused/packed
    aggregation links are closed.  The currency entry remains a genuine
    refusal inside the build because no plumbing changes the anchors'
    objective.
    Two things this still does not do: render and export.
    ``ProductionWeightCache`` has no trellis mechanism and
    ``export_native_compressed`` refuses a TCQ assignment outright.  A
    selected rung is a research result -- an exactly priced report of what
    the DP would choose -- not a shippable artifact.
    """

    resolved_path = manifest_path or os.environ.get(TRELLIS_SURFACE_ENV)
    if not resolved_path:
        return candidates

    return build_trellis_menu(
        candidates,
        stats,
        cost_mode=cost_mode,
        manifest_path=resolved_path,
        provenance_out=provenance_out,
        model_profile=model_profile,
    )


def assignment_has_trellis(assignment: Mapping[str, str]) -> list[str]:
    """Units in an assignment whose selected format is a trellis rung."""

    from .trellis_formats import parse_trellis_format_name

    return sorted(
        name for name, fmt in assignment.items()
        if isinstance(fmt, str) and parse_trellis_format_name(fmt) is not None
    )


__all__ = [
    "DEFAULT_RUNGS_PER_UNIT",
    "TRELLIS_MENU_PROVENANCE_SCHEMA",
    "TRELLIS_PACKED_PARENT_ANCHOR_SCHEMA",
    "TRELLIS_PACKED_WIRE_RECIPE_SCHEMA",
    "PACKED_PARENT_DLOSS_SCOPE",
    "PACKED_WIRE_DECOMPOSITION",
    "TRELLIS_SURFACE_ENV",
    "TRELLIS_SURFACE_MANIFEST_SCHEMA",
    "TrellisMenuError",
    "TrellisPackedExpertLayoutError",
    "TrellisSeamUnwiredError",
    "TrellisSurfaceManifest",
    "assignment_has_trellis",
    "UNWIRED_LINKS",
    "augment_candidates",
    "build_trellis_menu",
    "load_manifest",
]
