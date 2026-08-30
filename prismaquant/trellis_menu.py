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

STATUS: THE MENU IS BUILT, ALLOCATION-TO-RENDER IS NOT WIRED
------------------------------------------------------------
:func:`build_trellis_menu` produces a correctly priced menu.  The production
seam :func:`augment_candidates` **refuses** when the flag is set, because
the links between that menu and a shipped assignment do not exist -- see
:data:`UNWIRED_LINKS`, which is the refusal message and the re-enable
checklist.  The first version of this module (40d3e15) claimed the seam's
placement inside ``build_candidates`` meant trellis rungs "pass the same
legality, aggregation and byte accounting every other candidate does".  That
was false on all three counts and is the reason the refusal exists: the
registry gaps crash loudly, but the aggregation gaps are SILENT, and a partial
fix would trade the loud failure for the silent one.

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
It does not hand a chosen candidate to the renderer, export, or serve.  The
independent encoder/wire writer and ``ProductionWeightCache`` blob path exist,
but require an explicit value-bearing ``TrellisEncodePlan``; this manifest seam
still emits only irreversible schedule/alphabet digests.  The native compressed
exporter is the wrong container, and the pinned Gridbook contract attests no
trellis lane, so export/serve remain ``unattested`` and refused.  This module is
allocation-time reach only: it lets the DP see the surface, report what it
would choose, and price the choice in exact serialized bytes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
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


#: Every link between a built trellis menu and a shipped assignment that does
#: NOT exist yet, verified at 58eb69d.  This list is the refusal message and
#: the re-enable checklist; it is also what ``docs/ARCHITECTURE.md`` 4.9 cites
#: instead of the claim it used to make.  Delete an entry only when a test
#: exercises the behaviour it names -- not when the code merely looks present.
UNWIRED_LINKS: tuple[tuple[str, str], ...] = (
    ("trellis_allocator.py:526-540",
     "serialized candidates retain schedule/alphabet hashes but not their "
     "actual values, so layer_config.json cannot construct the mandatory "
     "value-bearing TrellisEncodePlan. Hashes are deliberately not inverted "
     "or resolved through an ambient campaign path"),
    ("format_registry.py:1267-1272",
     "no TCQ name is a FormatSpec; fr.get_format('TCQ_E2M1_R640') KeyErrors, "
     "so every site that resolves an assigned format through the registry "
     "fails on a selected rung"),
    ("allocator.py:3369-3386",
     "the exact assignment-payload filter finds no '_memory_bytes_by_format' "
     "entry for a TCQ row and falls through to fr.get_format -- the allocator "
     "dies inside the Pareto sweep, before layer_config.json is written; the "
     "pointed refusals in layer_config.canonicalize_format and "
     "export_native_compressed are therefore unreachable"),
    ("allocator_candidates.py:2464",
     "fused-sibling aggregation builds each super-item menu by iterating "
     "FormatSpec objects, so every trellis rung is dropped from every fused "
     "group (probe: members offered TCQ_E2M1_R640, super item offered only "
     "BF16/NVFP4)"),
    ("allocator_candidates.py:2701",
     "packed-expert aggregation has the identical construction, so no MoE "
     "expert group can carry a rung either; between the two, on a dense model "
     "only o_proj and down_proj could ever hold one"),
    ("allocator_solver.py:340-342",
     "promote_serving_units' format_rank lookup does not KeyError today only "
     "because aggregation guarantees a TCQ unit is a lone ungrouped Linear; "
     "fixing aggregation without it makes that crash live"),
    ("footprint.py:1183",
     "the byte-budget (--target-disk-gb) path has its own registry lookup "
     "that KeyErrors on TCQ independently of the payload filter"),
    ("allocator.py:2756",
     "build_candidates is called with neither cost_mode= nor "
     "trellis_provenance=, so the currency gate below compares against "
     "os.environ.get('COST_MODE','aura') -- a variable run-pipeline.sh sets "
     "with := and never exports (:438) -- and the manifest identity, anchor "
     "currency and anchor activation contract are discarded instead of "
     "travelling with the assignment (principles 12 and 14)"),
    ("trellis_rate_surface.py:43-52",
     "the anchors' currency is weighted SSE under a per-input-channel "
     "activation second moment -- an output-MSE proxy, explicitly NOT the "
     "AURA KL-adjoint the production DP ranks in; aura_cost's two dW sources "
     "(ProductionWeightCache and fr.get_format(...).quantize_dequantize, "
     ":609-654) both require a registered format, so AURA-priced anchors are "
     "a dW-supply problem, not an objective change"),
)


@dataclass(frozen=True)
class TrellisSurfaceManifest:
    """A campaign's measured trellis anchors, plus what they were measured on."""

    path: Path
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
        payload = json.loads(resolved.read_text())
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
            candidates[unit_name].append(replace(base, fmt=fmt))
            added += 1
        if records:
            covered.append(unit_name)

    payload = {
        "schema": TRELLIS_MENU_PROVENANCE_SCHEMA,
        "manifest_path": str(manifest.path),
        "cost_mode": manifest.cost_mode,
        "currency": manifest.currency,
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
    """The production seam: a no-op when unset, a REFUSAL when set.

    Unset, this returns its input object unchanged and the run is
    byte-identical to one built without this module -- that half is real and
    is what ships.

    Set, it refuses.  :func:`build_trellis_menu` builds a correctly priced
    menu, but eight links between that menu and a shipped assignment do not
    exist (:data:`UNWIRED_LINKS`), and they do not fail the same way: the
    registry gaps crash loudly inside the Pareto sweep, while the aggregation
    gaps are SILENT -- they would drop every rung from every fused and packed
    group and hand back a plausible-looking frontier in which only o_proj and
    down_proj could carry a rung.  A partial fix that removed only the crashes
    would convert the loud failures into that silent one, which is why this
    refuses as a whole rather than being wired halfway (principle 1: the
    measurement must be right, not the symptom suppressed).

    Enabling the surface therefore means landing those links with tests that
    exercise behaviour, then deleting this refusal -- not passing a flag.
    Until then :func:`build_trellis_menu` is reachable directly, for research
    and for the tests, where a wrong menu cannot reach a shipped artifact.
    """

    resolved_path = manifest_path or os.environ.get(TRELLIS_SURFACE_ENV)
    if not resolved_path:
        return candidates

    links = "\n".join(f"  - {where}: {what}" for where, what in UNWIRED_LINKS)
    raise TrellisSeamUnwiredError(
        f"{TRELLIS_SURFACE_ENV}={resolved_path} was set, but the allocator "
        f"cannot honour a trellis rung end-to-end. Eight links are missing:\n"
        f"{links}\n"
        f"Build the menu directly with trellis_menu.build_trellis_menu() for "
        f"research. Do not remove this refusal to reach a selectable run: the "
        f"aggregation gaps are silent, so the run would look successful and "
        f"allocate wrongly."
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
