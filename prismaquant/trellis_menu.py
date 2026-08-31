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

STATUS: BOTH MANIFEST MENU PATHS REFUSE
--------------------------------------
The legacy v1 manifest lacks the frozen curve identity, matched scalar
backbone, pre-measurement holdout seal, and full-truth allocation-regret record.
Attaching those facts while loading would invent provenance, so
:func:`build_trellis_menu` refuses v1 before mutating a candidate menu.  An
identity-complete v2 manifest parser is intentionally not present yet.  The
in-memory research API in ``trellis_rate_surface`` is the only densification
path.  Its full-truth gate retrospectively validates the exact menu with an
internal greedy allocator; merely materializing that menu does not transfer the
result to a consuming solver.  The production seam also refuses because the
end-to-end links in :data:`UNWIRED_LINKS` remain absent.

WHY A MANIFEST AND NOT A ``FORMATS`` ENUM ENTRY
----------------------------------------------
A trellis rung is not an enum value.  It is ``(family, body_rate_q256,
layout, schedule, alphabets)``, and the wire's rate resolution is
``256/columns`` q256 -- effectively continuous.  What makes a rung *cost*
something is a measured anchor, and anchors are per-campaign data, not a
constant in the source tree.  So the flag names a file of measured anchors and
this module can parse its legacy campaign provenance, but refuses to densify
it.  A future identity-complete v2 record would still describe closed
``TCQ_{E2M1,E4M3}_R<q256>`` wire spellings rather than enum constants.

ADDITIONAL GATES A FUTURE V2 LOADER MUST RETAIN
------------------------------------------------
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
It does not densify a manifest, populate a DP menu, render, export, or serve.
``ProductionWeightCache`` has no trellis mechanism and
``export_native_compressed`` refuses a TCQ assignment outright. The separate
in-memory rate-surface module offers research-only campaign planning and
retrospective internal-greedy validation; this manifest seam exposes neither.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path

from .allocator_solver import Candidate
from .trellis_formats import (
    LAYOUTS,
)

#: Names a JSON manifest of measured anchors.  Unset is the default and is a
#: byte-identical no-op.
TRELLIS_SURFACE_ENV = "PRISMAQUANT_TRELLIS_SURFACE"

TRELLIS_SURFACE_MANIFEST_SCHEMA = "prismaquant.trellis_surface_manifest.v1"
TRELLIS_SURFACE_MANIFEST_SCHEMA_V2 = "prismaquant.trellis_surface_manifest.v2"
TRELLIS_MENU_PROVENANCE_SCHEMA = "prismaquant.trellis_menu_provenance.v1"

#: Rungs densified per unit when the manifest does not say.  The anchors
#: themselves are always included on top of this count.
DEFAULT_RUNGS_PER_UNIT = 16


class TrellisMenuError(RuntimeError):
    """The surface manifest cannot be admitted to this run."""


class TrellisSeamUnwiredError(TrellisMenuError):
    """The production seam is enabled but the DP cannot honour a TCQ rung."""


#: Every link between a built trellis menu and a shipped assignment that does
#: NOT exist yet, verified at 58eb69d and re-verified at WO-A
#: (2026-08-31, muse/wo-a-trellis-format-20260831): two links are now wired
#: — format_registry now registers the five E2M1 candidate rungs as
#: tcq_trellis FormatSpecs derived at import time from the pinned
#: contract, and footprint's exact byte seam now handles TCQ via
#: trellis_footprint.trellis_tensor_payload_breakdown — so they are
#: removed here. This list is the refusal message and the re-enable
#: checklist; it is also what ``docs/ARCHITECTURE.md`` 4.9 cites instead
#: of the claim it used to make. Delete an entry only when a test
#: exercises the behaviour it names -- not when the code merely looks present.
UNWIRED_LINKS: tuple[tuple[str, str], ...] = (
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
    """Parse v1 campaign provenance for read-only planning inspection.

    Every field is required.  There is no default for ``activation_contract``
    or ``target_profile`` on purpose: a defaulted answer to "what did you
    measure this on" is indistinguishable from a wrong one.  Returning this
    record does not validate or authorize interpolation;
    :func:`build_trellis_menu` always refuses it.
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
    if schema == TRELLIS_SURFACE_MANIFEST_SCHEMA_V2:
        raise TrellisMenuError(
            "identity-complete trellis manifest v2 ingestion is not "
            "implemented; use the sealed in-memory trellis_rate_surface "
            "research API rather than discarding v2 identity or gate fields"
        )
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


def build_trellis_menu(
    candidates: dict[str, list[Candidate]],
    stats: Mapping[str, Mapping[str, object]],
    *,
    cost_mode: str,
    manifest_path: str | None = None,
    provenance_out: dict | None = None,
) -> dict[str, list[Candidate]]:
    """Refuse legacy manifest interpolation, or no-op when no path is set.

    ``manifest_path`` defaults to ``PRISMAQUANT_TRELLIS_SURFACE``.  With
    neither set this is a no-op returning the same object, so a run without
    the flag executes exactly the code path it executed before this module
    existed.

    Manifest v1 cannot prove the curve identity or the holdout/regret record.
    It is therefore not a compatibility path.  A future v2 loader must parse
    identity-complete records and consume the same sealed full-truth validation
    context as
    :func:`prismaquant.trellis_rate_surface.rate_surface_solver_menu`. That
    context is not a license for a different consuming allocator; until an
    identity-complete loader exists, direct manifest-to-menu conversion is
    impossible.
    """

    resolved_path = manifest_path or os.environ.get(TRELLIS_SURFACE_ENV)
    if not resolved_path:
        return candidates

    load_manifest(resolved_path)
    raise TrellisMenuError(
        f"legacy manifest schema {TRELLIS_SURFACE_MANIFEST_SCHEMA!r} is "
        "planning provenance only and cannot be densified or admitted to an "
        "allocator menu: it lacks the full frozen curve identity, matched "
        "scalar-backbone context, a pre-measurement holdout seal, and a "
        "full-truth retrospective allocation-regret record. Supply an "
        "identity-complete "
        f"{TRELLIS_SURFACE_MANIFEST_SCHEMA_V2!r} record once a v2 loader "
        "exists; do not infer those fields from v1."
    )


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

    Set, it refuses.  The identity-complete in-memory research API can build
    an exactly priced menu covered by retrospective full-truth validation, but
    that validation does not transfer to another allocator. Six links
    between such a menu and a shipped assignment do not exist
    (:data:`UNWIRED_LINKS`; WO-A wired the two loud registry/footprint
    links, leaving the silent aggregation gaps), and they do not fail the
    same way: the remaining registry gaps crash loudly inside the Pareto
    sweep, while the aggregation gaps are SILENT -- they would drop every
    rung from every fused and packed group and hand back a plausible-
    looking frontier in which only o_proj and down_proj could carry a rung.
    A partial fix that removed only the crashes would convert the loud
    failures into that silent one, which is why this refuses as a whole
    rather than being wired halfway (principle 1: the measurement must be
    right, not the symptom suppressed).

    Enabling the surface therefore means landing those links with tests that
    exercise behaviour, then deleting this refusal -- not passing a flag.
    Legacy :func:`build_trellis_menu` also refuses because its v1 records lack
    the identity and full-truth validation record required even for
    retrospective allocator research.
    """

    resolved_path = manifest_path or os.environ.get(TRELLIS_SURFACE_ENV)
    if not resolved_path:
        return candidates

    links = "\n".join(f"  - {where}: {what}" for where, what in UNWIRED_LINKS)
    raise TrellisSeamUnwiredError(
        f"{TRELLIS_SURFACE_ENV}={resolved_path} was set, but the allocator "
        f"cannot honour a trellis rung end-to-end. Six links are missing:\n"
        f"{links}\n"
        f"Use the identity-bound trellis_rate_surface research API only. Do "
        f"not remove this refusal to reach a selectable run: the "
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
    "TRELLIS_SURFACE_MANIFEST_SCHEMA_V2",
    "TrellisMenuError",
    "TrellisSeamUnwiredError",
    "TrellisSurfaceManifest",
    "assignment_has_trellis",
    "UNWIRED_LINKS",
    "augment_candidates",
    "build_trellis_menu",
    "load_manifest",
]
