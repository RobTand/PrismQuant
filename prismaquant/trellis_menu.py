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

STATUS: WIRED FOR ALLOCATION, REFUSED ON CURRENCY, NEVER EXPORTED
-----------------------------------------------------------------
:func:`build_trellis_menu` produces a correctly priced menu and
:func:`augment_candidates` now installs it.  The first version of this module
(40d3e15) claimed the seam's placement inside ``build_candidates`` meant
trellis rungs "pass the same legality, aggregation and byte accounting every
other candidate does".  That was false on all three counts; the module then
refused as a whole, because the registry gaps crashed loudly while the
aggregation gaps were SILENT and a partial fix trades the loud failure for
the silent one.

Seven of those eight links were wired on 2026-08-29, each with a behaviour
test:

* super-item aggregation builds each menu from the MEMBERS' candidate lists
  (``_super_menu_format_names``), so a rung survives a fused or packed group;
* ``format_rank`` is extended from the candidate menu by exact serialized
  rate, so ``promote_serving_units`` can rank a selected rung;
* exact bytes travel in ``_memory_bytes_by_format`` -- written here, read by
  the payload filter, ``footprint``, ``compute_achieved``, ``kl_measurement``
  and bit attribution -- with a POINTED refusal at each site rather than a
  registry fallback, because no ``FormatSpec`` can express these bytes;
* the run's attested ``COST_MODE`` and the menu's provenance are threaded
  from ``allocator.py`` into the layer-config metadata block.

The eighth is not a link.  It is the anchors' **currency**
(:data:`UNWIRED_LINKS`), and no plumbing turns a weighted-SSE anchor into an
AURA-priced one, so it stays a refusal (:func:`_require_run_currency`).

WHY NO TCQ ``FormatSpec``
-------------------------
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
TCQ to every ``act_quant_changes_input`` / ``quantize_dequantize`` consumer
with no render behind it.  So exact bytes ride the ``Candidate`` and
``_memory_bytes_by_format`` -- the repo's existing single mechanism -- and
``fr.get_format('TCQ_...')`` keeps raising ``KeyError``.  The cost is N
pointed refusals instead of one registry entry; each one refuses where a
closed form would have guessed.

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

THE THREE THINGS THIS REFUSES, AND WHY
--------------------------------------
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

#: Rungs densified per unit when the manifest does not say.  The anchors
#: themselves are always included on top of this count.
DEFAULT_RUNGS_PER_UNIT = 16


class TrellisMenuError(RuntimeError):
    """The surface manifest cannot be admitted to this run."""


class TrellisSeamUnwiredError(TrellisMenuError):
    """The production seam is enabled but the DP cannot honour a TCQ rung."""


#: The links between a built trellis menu and a shipped assignment that still
#: do NOT exist.  This list is the refusal message and the re-enable
#: checklist; it is also what ``docs/ARCHITECTURE.md`` 4.9 cites instead of
#: the claim it used to make.  Delete an entry only when a test exercises the
#: behaviour it names -- not when the code merely looks present.
#:
#: Seven of the original eight were wired on 2026-08-29 and their entries
#: removed together with the tests that exercise them
#: (``tests/test_trellis_menu.py``, ``tests/test_super_item_menu_byte_identity.py``).
#: What remains is the CURRENCY question, which is not a wiring gap at all:
#: no plumbing turns a weighted-SSE anchor into an AURA-priced one.  It is
#: enforced by :func:`_require_run_currency`, so an enabled seam now refuses
#: on the currency mismatch rather than on missing links.
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
    #: unit name -> {"family", "alphabets": {rate: codes}, "points": [...]}
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
            f"UNWIRED_LINKS[0] and it is not a plumbing gap: the measured "
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


def build_trellis_menu(
    candidates: dict[str, list[Candidate]],
    stats: Mapping[str, Mapping[str, object]],
    *,
    cost_mode: str,
    manifest_path: str | None = None,
    provenance_out: dict | None = None,
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
        if _is_packed_expert(stat):
            # Refuse rather than price num_experts x per-expert bytes: that
            # pricing would assert a per-expert trellis render coherent with
            # the packed unit's single-format constraint, which no measurement
            # supports.  Counted, not silent.
            skipped[unit_name] = (
                f"packed-expert row (shape {shape}); no per-expert trellis "
                f"render exists, and pricing one would be an unmeasured claim"
            )
            continue
        try:
            records = _unit_candidates(
                unit_name,
                shape,
                entry,
                manifest,
                qname=unit_name,
                packed_expert=None,   # refused above; never reached packed
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
            base = record.to_solver_candidate()
            cand = replace(base, fmt=fmt)
            candidates[unit_name].append(cand)
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
        if records:
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
) -> dict[str, list[Candidate]]:
    """The production seam: a no-op when unset, a built menu when set.

    Unset, this returns its input object unchanged and the run is
    byte-identical to one built without this module -- that half was always
    real and is what ships.

    Set, it now builds the menu.  Until 2026-08-29 it refused outright,
    because eight links between a built menu and a shipped assignment did not
    exist and they did not fail the same way: the registry gaps crashed
    loudly inside the Pareto sweep, while the aggregation gaps were SILENT --
    they dropped every rung from every fused and packed group and handed back
    a plausible-looking frontier in which only ``o_proj`` and ``down_proj``
    could carry one.  Refusing as a whole was right precisely because a
    partial fix converts the loud failures into that silent one.

    Seven of those links are now wired, each with a test that exercises the
    behaviour rather than the source text.  The eighth is not a link at all:
    the anchors' currency (:data:`UNWIRED_LINKS`).  No plumbing turns a
    weighted-SSE anchor into an AURA-priced one, so that stays a refusal --
    raised by :func:`_require_run_currency` inside the build, against the
    run's ATTESTED cost mode.

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
    "TRELLIS_SURFACE_ENV",
    "TRELLIS_SURFACE_MANIFEST_SCHEMA",
    "TrellisMenuError",
    "TrellisSeamUnwiredError",
    "TrellisSurfaceManifest",
    "assignment_has_trellis",
    "UNWIRED_LINKS",
    "augment_candidates",
    "build_trellis_menu",
    "load_manifest",
]
