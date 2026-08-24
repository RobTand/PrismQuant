"""Anchored CB AURA (+AQUA) for a DENSE model, over the platform-agnostic core.

All pricing math lives in :mod:`prismaquant.anchored_cost` and the CB facts in
:mod:`prismaquant.cb_anchored_cost`.  This module supplies only what a *dense*
campaign owns: the body census (derived from the probe, not hardcoded), the
CB ladders, the panel/validation policy, the anchors, and the byte budget.

WHY THIS EXISTS ALONGSIDE ``dsv4_aura_cb_reprice``
--------------------------------------------------
That module is a frozen DSv4 launcher: routed/dense expert split, a learned
FP8-CB basis, a two-rung routed ladder, a W8A16 readmission lane.  None of it
applies to a dense model, and it is frozen precisely so a shipped campaign can
be replayed byte-for-byte.  This is a sibling caller, not a refactor of it.

WHY ANCHORED RATHER THAN THE STOCK ``COST_MODE=aura`` PATH
-----------------------------------------------------------
``run-pipeline.sh`` [2b/4] renders EVERY non-BF16 format in ``FORMATS`` into a
retained format-menu production cache.  Measured on Qwen3.8-27B, a retained
rung costs 45.5 GB (the cache stores the *rendered* ~bf16 weight, 2.002
B/qparam, K-independent), so the complete 37-rung public CB menu is ~1.68 TB
-- more than the box has -- and the cache has no downstream consumer, because the exporter
re-encodes from source.  Truncating the ladder to fit that budget would be a
render-budget heuristic deciding the allocator's candidate set, which is what
``PRODUCTION_CACHE_UNION`` was archived for (principle 1).

The anchored path renders ONE production anchor per (unit, family) plus a
sampled panel, fits a per-segment shape, and prices the whole ladder from it.
The renders are transient: ``build_production_cache`` sets
``retain_rendered=(render_scope == "assignment")``, so a format-menu CB build
consumes each (qname, fmt) pair into the AURA adjoint and drops it, retaining
receipts instead of weights.  Disk cost is ~0.

WHERE AQUA ENTERS
-----------------
AURA is activation-quant-blind: NVFP4 and NVFP4A16 render weights bit
identically, so a pure AURA table prices a W4A4 rung and a W16A16 rung of the
same weight error the same.  On a CB menu that is the whole
``NVFP4_CB`` (act_bits=4, group 16) vs ``FP8_CB`` (act_bits=8) decision, so an
anchored AURA table without an A-side cannot see the boundary it most needs to.

``allocator_candidates.cost_entry_predicted_dloss`` reads the anchored branch as
``base + cost_entry_act_dloss(cost_entry)`` -- the A-side is ADDED, not
multiplied by P5a's per-family constant, because P5a's multiply is an
estimator TRANSFER from weight space to output space and an anchored projection
is already in the right currency.  So AQUA on this lane is exactly: run
``aqua_activation_cost`` over the anchored payload before allocating.  It needs
no render (``activation_dloss`` reads the dense ``W[o,j]^2``, the card's
``g_sq_sum`` and the format's activation grid), which is why it can be layered
onto a production-rendered W-side with no rendering confound.

The weight-only anchored payload is written and KEPT, so the pre-AQUA
allocation stays reproducible as an A/B arm.  That A/B, on served KL at matched
bpp, is what would promote AQUA-on-CB; this module produces the candidate, not
the verdict.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import sys

from prismaquant import format_registry as fr
from prismaquant.anchored_cost import (
    AURA_CURRENCY,
    UnitSpec,
    extrapolation_distance_report,
    plan_anchor_requests,
    price_anchored_candidates,
    run_allocator_once,
)
from prismaquant.cb_anchored_cost import (
    CBPanelPolicy,
    CBUnitDeclaration,
    CodebookAnchoredFormatPlugin,
    LATTICE_BASIS,
    ROUTE_FLIP_LIMITATION,
    anchors_from_streamed_payload,
    build_cb_allocator_cost_payload,
    build_cb_units,
    build_streamed_cb_render_plan,
    cb_rung,
    fit_all_cb_segments,
    fitted_cb_hull_report,
    heldout_validation_report,
    observations_from_streamed_payload,
    plan_cb_panel_and_validation,
    require_allocator_supersurrogate_support,
    run_streamed_cb_anchor_aura,
    write_cb_cost_payload,
)
from prismaquant.cb_layout import FP8_PRODUCT_RUNGS, NVFP4_PRODUCT_RUNGS
from prismaquant.cost_stage_checkpoint import (
    atomic_write_bytes,
    canonical_json_sha256,
)

CAMPAIGN_SCHEMA = "prismaquant.dense_anchored_cb.campaign.v3"
DEFAULT_TARGET_PROFILE = "qwen38_sm120_cb_validation_only"

# NVFP4-CB is outside the gridbook fused mid-M kernel law (its lane backs no
# fused rungs at any version; every rung rides the fallback route), so the
# campaign follows the authoritative contiguous producer range. The v3
# campaign schema is intentionally incompatible with the old K12..K24 fit.
NVFP4_CB_LADDER = tuple(
    f"NVFP4_CB_K{k}" for k in NVFP4_PRODUCT_RUNGS
)
# FP8-CB follows its authoritative producer law, not the narrower fused-mid-M
# performance subset of the historical Gridbook pin.  K4..K48 step 4 is the
# public producer ladder; a rung without a fused mid-M route can still use the
# correctness bridge.  This validation-only campaign does not turn either
# route into a release or device-qualification claim.
FP8_CB_LADDER = tuple(f"FP8_CB_K{k}" for k in FP8_PRODUCT_RUNGS)

# One anchor per (family, basis), rendered for EVERY unit of that segment, so
# it must be legal everywhere the segment reaches and interior enough to keep
# the worst extrapolation short.
#   nvfp4_cb: K1..K25 -> K13 is the central integer rung.
#   fp8_cb:   K4..K48 step 4 -> K24 is one of the two central rungs (worst
#             distance 24, tied with K28) and is the lower one, nearer the
#             compressed candidates this campaign exists to study.
# Learned codebooks are a measured NULL on Qwen dense (holdout ~1.00 across
# K28-K43), so this lane declares ONE basis and there is no learned segment.
ANCHOR_FORMATS = {
    ("nvfp4_cb", LATTICE_BASIS): "NVFP4_CB_K13",
    ("fp8_cb", LATTICE_BASIS): "FP8_CB_K24",
}

# Panel rungs must (a) stay inside the segment's legal ladder and (b) give the
# shape design full rank after per-unit centering. The expanded NVFP4 design is
# (rung, parity, below-K12 hinge, above-K24 hinge): it fits separate low-band
# and endpoint effects and cannot reuse the old K12..K24 line as endpoint
# evidence. The panel
# brackets both hinges, both parities, and the exact K1/K25 endpoints. K25 is
# the only producer rung above the high hinge, so its coefficient is honestly
# a measured endpoint shoulder rather than an extrapolated high-band slope.
# Every legal FP8-CB rung is k % 4 == 0, so parity is constant there, the
# design drops to (rung,) alone, and any two distinct rungs suffice.  The
# three-point K4/K28/K48 panel spans both endpoints and an interior point;
# this is a measured straight-line proposal whose curvature is challenged by
# four off-panel validation rungs below and above the K24 anchor.
PANEL_RUNGS = {
    "nvfp4_cb": (
        "NVFP4_CB_K1", "NVFP4_CB_K2", "NVFP4_CB_K11", "NVFP4_CB_K12",
        "NVFP4_CB_K23", "NVFP4_CB_K24", "NVFP4_CB_K25",
    ),
    "fp8_cb": ("FP8_CB_K4", "FP8_CB_K28", "FP8_CB_K48"),
}
# Held-out UNITS are the primary generalization axis (the fit is applied to
# every unit, not just panel members).  A validation rung must never be the
# ANCHOR: there the fit reproduces the measurement by construction, so its dex
# is exactly 0.0000 and it dilutes the report with tautological passes.  The
# first Qwen3.8-27B campaign shipped FP8_CB_K36 (its own anchor) in this table
# and 48 of 192 validation cells were therefore vacuous;
# `plan_cb_panel_and_validation` now refuses that configuration on EVERY CB
# driver rather than reporting it.
#
# Being absent from the PANEL is a second, weaker requirement, and it is
# deliberately not enforced: a panel rung on a held-out unit still tests
# whether the fitted shape transfers to that unit, and on a two-rung ladder
# whose second rung is the anchor it is the only validation that can exist at
# all.  This lane has rungs to spare and takes the stronger design anyway; the
# report tags each cell's `held_out_axes` so a reader can tell which was used.
#
# NVFP4-CB validation is off-panel on both sides of the low hinge and within
# the historical band. K25 is deliberately repeated on held-out units because
# there is no second producer rung above K24: this tests transfer of the
# measured endpoint shoulder without pretending that a held-out high-band
# slope exists. The report's ``held_out_axes`` records that distinction.
# FP8-CB validation covers the low shoulder, the compressed side immediately
# below the anchor, and two high-side points.  All four are off-panel and none
# is the anchor, so every cell holds out both unit and rung.
VALIDATION_RUNGS = {
    "nvfp4_cb": (
        "NVFP4_CB_K3", "NVFP4_CB_K10", "NVFP4_CB_K14",
        "NVFP4_CB_K22", "NVFP4_CB_K25",
    ),
    "fp8_cb": (
        "FP8_CB_K8", "FP8_CB_K20", "FP8_CB_K36", "FP8_CB_K44",
    ),
}
# The smallest role on a hybrid-attention dense model is the full-attention
# block count (16 on Qwen3.8-27B), and plan_cb_panel_and_validation refuses a
# role with fewer eligible units than panel+validation.  10+4 leaves slack.
PANEL_UNITS_PER_ROLE = 10
VALIDATION_UNITS_PER_ROLE = 4
PANEL_SEED = 42

# Streaming source-cache policy: fail-closed, no prefetch lookahead, matching
# the DSv4 campaign's contract.  A dense 27B fits comfortably; the bound exists
# so a pressured box pre-evicts instead of OOMing on the layer read.
STREAMING_CACHE_MAX_SLOTS = 2

_HOME_RESERVE_FRACTION = 0.10


class DenseCampaignError(RuntimeError):
    """A census, identity, or orchestration refusal."""


@dataclass(frozen=True)
class PreparedCampaign:
    args: argparse.Namespace
    profile: object
    probe_stats: dict
    probe_meta: dict
    body_qnames: tuple[str, ...]
    roles: Mapping[str, str]
    cb_context: object
    plugin: CodebookAnchoredFormatPlugin
    format_plan: object
    units: tuple[UnitSpec, ...]
    anchor_requests: tuple
    panel_requests: tuple
    validation_requests: tuple
    formats_by_qname: Mapping[str, Sequence[str]]
    purposes_by_qname: Mapping[str, Mapping[str, Sequence[str]]]
    plan_report: dict
    panel_report: dict
    arm_identity: Mapping[str, object]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_work_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve(strict=False)
    if resolved == Path("/tmp") or Path("/tmp") in resolved.parents:
        raise DenseCampaignError("campaign work-dir may not be under /tmp")
    return resolved


def _assert_home_reserve() -> None:
    """Principle: never start a long render that can fill the disk."""
    usage = shutil.disk_usage("/home/rob")
    if usage.free < usage.total * _HOME_RESERVE_FRACTION:
        raise DenseCampaignError(
            f"/home/rob has {usage.free / 1e9:.1f} GB free, below the "
            f"{_HOME_RESERVE_FRACTION:.0%} reserve "
            f"({usage.total * _HOME_RESERVE_FRACTION / 1e9:.1f} GB)"
        )


def _load_probe(args: argparse.Namespace) -> tuple[dict, dict]:
    try:
        with Path(args.probe).open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise DenseCampaignError(f"probe is unreadable: {args.probe}") from exc
    stats = payload.get("stats") if isinstance(payload, Mapping) else None
    meta = payload.get("meta") if isinstance(payload, Mapping) else None
    if not isinstance(stats, Mapping) or not isinstance(meta, Mapping):
        raise DenseCampaignError("probe lacks stats/meta mappings")
    for field, value in (
        ("nsamples", int(args.n_calib_samples)),
        ("seqlen", int(args.calib_seqlen)),
    ):
        if int(meta.get(field, -1)) != value:
            raise DenseCampaignError(
                f"probe calibration {field}={meta.get(field)!r}, "
                f"expected {value}"
            )
    calibration_hash = meta.get("calib_hash")
    if not isinstance(calibration_hash, str) or not calibration_hash:
        raise DenseCampaignError("probe has no exact calibration hash")
    return dict(stats), dict(meta)


def _body_census(
    profile, stats: Mapping[str, Mapping[str, object]], args: argparse.Namespace,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """The units the DP may allocate, and their roles.

    Three classes are excluded and each for a different reason, so they are
    filtered by name rather than by one predicate:

    * routed experts -- there must be none.  This driver's smooth-AURA anchors
      are route-flip-blind, and a model with routed experts needs the empirical
      expert unit-KL hybrid the DSv4 lane carries.  Refuse rather than price
      them wrong.
    * profile-pinned names (``lm_head``) -- gridbook cannot serve a CB head at
      all: the export succeeds and the load dies on ``lm_head.cb_qweight``
      because no method claims ``ParallelLMHead``.
    * MTP / visual sidecars -- stripped from this artifact by campaign
      decision, and pinned BF16 by the allocator regardless.
    """
    from prismaquant.routed_experts import ProfileRoutedExpertClassifier

    classifier = ProfileRoutedExpertClassifier(profile)
    routed = sorted(
        qname for qname in stats if classifier.classify(qname) is not None
    )
    if routed:
        raise DenseCampaignError(
            f"{len(routed)} routed-expert units are declared; this dense "
            f"driver has no empirical expert cost stage. sample={routed[:4]}"
        )
    excluded_markers = tuple(
        marker.strip() for marker in args.exclude_markers.split(",")
        if marker.strip()
    )
    body = tuple(
        qname for qname in sorted(stats)
        if not any(marker in qname for marker in excluded_markers)
    )
    if not body:
        raise DenseCampaignError("the body census is empty after exclusions")
    if len(body) != int(args.expect_body_units):
        raise DenseCampaignError(
            f"body census is {len(body)} units, campaign declares "
            f"{args.expect_body_units}; excluded "
            f"{sorted(set(stats) - set(body))[:8]}"
        )
    roles = {qname: str(qname).rsplit(".", 1)[-1] for qname in body}
    counts = Counter(roles.values())
    smallest = min(counts.values())
    needed = PANEL_UNITS_PER_ROLE + VALIDATION_UNITS_PER_ROLE
    if smallest < needed:
        raise DenseCampaignError(
            f"smallest role has {smallest} units; the panel policy needs "
            f"{needed}. Roles: {dict(sorted(counts.items()))}"
        )
    return body, roles


def _build_declarations(
    *,
    stats: Mapping[str, Mapping[str, object]],
    body: Sequence[str],
    roles: Mapping[str, str],
    format_plan,
    cb_context,
) -> tuple[CBUnitDeclaration, ...]:
    from prismaquant.allocator_candidates import (
        _source_bpp_applicability,
        serialized_candidate_payload,
    )

    serving_group: dict[str, str] = {}
    for members in format_plan.serving_groups:
        representative = min(map(str, members))
        for qname in members:
            serving_group[str(qname)] = representative

    declarations: list[CBUnitDeclaration] = []
    for qname in body:
        row = stats[qname]
        shape = (int(row["out_features"]), int(row["in_features"]))
        if math.prod(shape) != int(row["n_params"]):
            raise DenseCampaignError(f"{qname}: probe shape/n_params differ")
        payloads: dict[str, int] = {}
        for format_name in (*NVFP4_CB_LADDER, *FP8_CB_LADDER):
            spec = fr.get_format(format_name)
            verdict = _source_bpp_applicability(
                shape, spec, qname=qname, source_kind="bf16",
                cb_serialization_context=cb_context,
            )
            if not verdict.legal:
                raise DenseCampaignError(
                    f"{qname}/{format_name}: source-rate gate refused "
                    f"{verdict.reason}: {verdict.detail}"
                )
            payloads[spec.name] = serialized_candidate_payload(
                spec, shape, qname=qname,
                cb_serialization_context=cb_context,
            )[0]
        # BF16 is the terminal: the source IS bf16, so this is a verbatim
        # passthrough and the only W/A identity rung on the ladder.
        payloads["BF16"] = 2 * math.prod(shape)
        declarations.append(CBUnitDeclaration(
            qname=qname,
            role=roles[qname],
            unit_class="nonexpert",
            n_params=math.prod(shape),
            payload_bytes_by_format=payloads,
            terminal_format="BF16",
            serving_group=serving_group.get(qname),
        ))
    return tuple(declarations)


@dataclass(frozen=True)
class DensePlan:
    """The dense analogue of ``SourceClassFormatPlan``.

    ``build_source_class_format_plan`` cannot express this campaign: it splits
    the menu into lower- and higher-source-rate classes and requires the
    expert menu to be a STRICT subset of the nonexpert one. A dense model has
    exactly ONE source-payload class, so there is no split to declare and
    faking one would put a menu in the artifact that no unit can take.

    What that plan supplies and this one must too: the serving-atomic groups
    (fused siblings must share a format or vLLM cannot load the checkpoint),
    the ladders, and one identity digest binding both into provenance.
    """

    ladders: Mapping[str, tuple[str, ...]]
    serving_groups: tuple[tuple[str, ...], ...]
    serving_restrictions: Mapping[str, object]
    target_profile: str
    target_platform: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "prismaquant.dense_anchored_cb.format_plan.v2",
            "ladders": {k: list(v) for k, v in sorted(self.ladders.items())},
            "serving_groups": [list(g) for g in self.serving_groups],
            "serving_restrictions": dict(self.serving_restrictions),
            "target_profile": self.target_profile,
            "target_platform": self.target_platform,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_json_sha256(
            self.to_dict(), where="dense anchored CB format plan"
        )


def _derive_dense_plan(
    profile,
    body: Sequence[str],
    *,
    target_profile: str,
) -> DensePlan:
    """Derive both ladders from the registry and selected target profile.

    Neither ladder is a CLI opinion or a constant taken on trust:

    * the registered family is narrowed by the serving profile's production
      format rule. This used to be what dropped the signed ``NVFP4_CB_S13..S16``
      rungs (research-only after losing 78.48% of matched weight-MSE
      comparisons); that family was DELETED 2026-08-17, since every native
      Gridbook FP4 route requires the unsigned two-tier product layout
      (``n_sub == 2 and type_size == 4*k + 9``), so the narrowing now has no
      signed rung left to drop;
    * the selected profile then closes that producer registry.  Fused-mid-M
      eligibility is deliberately NOT an admission filter: it is a
      performance route for one batch regime, while unfused rungs retain a
      correctness bridge.  Treating the historical fused set as the format
      set was what hid FP8-CB K4..K24 from AQUA despite their public producer
      identity.

    The module constants are asserted against the selected profile's exact
    producer intersection.  This proves structural candidate registration;
    release pinning and device-qualified route evidence remain separate,
    fail-closed shipping gates.
    """
    from prismaquant.serving_profiles import (
        check_serving_format,
        load_serving_profile,
    )
    from prismaquant.source_class_format_plan import _serving_components

    selected = load_serving_profile(target_profile)

    ladders: dict[str, tuple[str, ...]] = {}
    restrictions: dict[str, object] = {}
    for family, expected in (
        ("nvfp4_cb", NVFP4_CB_LADDER), ("fp8_cb", FP8_CB_LADDER),
    ):
        legal = tuple(
            spec.name for spec in fr.list_producer_formats(family)
            if check_serving_format(target_profile, None, spec.name).legal
        )
        if not legal:
            raise DenseCampaignError(
                f"serving profile {target_profile!r} declares no production "
                f"format of family {family!r}"
            )
        if legal != tuple(expected):
            raise DenseCampaignError(
                f"{family}: selected profile {target_profile!r} admits "
                f"{list(legal)}, campaign declares {list(expected)}. Update "
                "the profile and ladder together rather than silently "
                "truncating or widening the candidate set."
            )
        ladders[family] = legal
        restrictions[family] = {
            "profile_id": target_profile,
            "target_platform": selected.target_platform,
            "admission": "producer_registry_intersect_selected_profile",
            "producer_formats": list(legal),
            "fused_mid_m_is_performance_metadata_not_format_admission": True,
            "shipping_gate": (
                "immutable release pin plus exact device-qualified route "
                "contract; not claimed by this validation plan"
            ),
        }

    return DensePlan(
        ladders=ladders,
        serving_groups=_serving_components(profile, tuple(body)),
        serving_restrictions=restrictions,
        target_profile=target_profile,
        target_platform=selected.target_platform,
    )


def _authoritative_source_map(cb_context) -> dict[str, str]:
    """The per-format codebook basis, exactly -- never inferred from a K range.

    ``_normalize_source_map`` refuses "a global source scalar or K-range
    inference" because with a learned bundle in play, only the bundle knows
    which rungs it actually backs. That refusal has one exact complement:
    ``CB_CODEBOOK_SOURCE_SCOPE=none`` is a declaration that NO format takes a
    learned book, so the per-format map is not guessed from the scalar, it is
    entailed by the scope. This builds that map explicitly rather than letting
    a scalar leak through as a default, and refuses every other combination --
    including a learned scope with no bundle, which is the case the core is
    protecting against.
    """
    declared = getattr(cb_context, "codebook_source_by_format", None)
    if isinstance(declared, Mapping) and declared:
        off_basis = sorted(
            name for name in (*NVFP4_CB_LADDER, *FP8_CB_LADDER)
            if str(declared.get(name, LATTICE_BASIS)) != LATTICE_BASIS
        )
        if off_basis:
            raise DenseCampaignError(
                "this driver declares the lattice basis only, but the bundle "
                f"maps {off_basis[:6]} elsewhere. Learned codebooks are a "
                "measured NULL on Qwen dense (holdout ~1.00 across K28-K43); "
                "set CB_CODEBOOK_SOURCE_SCOPE=none."
            )
        return dict(declared)

    scope = str(getattr(cb_context, "codebook_source_scope", "") or "").lower()
    scalar = str(getattr(cb_context, "codebook_source", "") or "").lower()
    if scope != "none" or scalar != LATTICE_BASIS:
        raise DenseCampaignError(
            "CB anchoring needs an authoritative per-format basis map. The "
            f"context declares scope={scope!r} source={scalar!r} with no "
            "bundle, which is not one: only "
            "CB_CODEBOOK_SOURCE_SCOPE=none + CB_CODEBOOK_SOURCE=lattice "
            "entails the map without a bundle."
        )
    return {
        name: LATTICE_BASIS for name in (*NVFP4_CB_LADDER, *FP8_CB_LADDER)
    }


def _panel_policy(roles: Mapping[str, str]) -> CBPanelPolicy:
    """One policy row per (family, role, basis) -- the role is load-bearing.

    The anchor-is-not-a-validation-rung and panel/validation-disjointness
    checks are NOT here: they live in ``plan_cb_panel_and_validation``, which
    every CB driver calls and which reads the anchors off the plugin rather
    than off a module constant.  A copy here would have protected this driver
    only -- and this driver is a ``__main__`` with no importers, so the
    campaign that actually shipped the defect (DSv4, which builds its own
    policy literal) would have gone unguarded.
    """
    all_roles = sorted(set(roles.values()))
    return CBPanelPolicy(
        panel_rungs_by_segment={
            (family, role, LATTICE_BASIS): PANEL_RUNGS[family]
            for family in PANEL_RUNGS
            for role in all_roles
        },
        validation_rungs_by_segment={
            (family, role, LATTICE_BASIS): VALIDATION_RUNGS[family]
            for family in VALIDATION_RUNGS
            for role in all_roles
        },
        panel_units_per_role=PANEL_UNITS_PER_ROLE,
        validation_units_per_role=VALIDATION_UNITS_PER_ROLE,
        seed=PANEL_SEED,
    )


def _require_bf16_body_source_manifest(
    body: Sequence[str],
    source_census: Mapping[str, str],
) -> dict[str, str]:
    """Bind this SM120 campaign to its single maintained source class.

    The generic registry deliberately retains source-FP8/W8A16 compatibility
    for already-published source-quantized artifacts.  This dense Qwen3.8
    campaign is a different product line: its body source is BF16, and admitting
    any source-FP8 unit would silently reopen the legacy W8A16 lane that the
    target profile excludes.  Refuse before the plan, allocator, or GPU work.
    """

    missing_source = sorted(set(body) - set(source_census))
    if missing_source:
        raise DenseCampaignError(
            f"source census misses body units: {missing_source[:8]}"
        )
    off_source = sorted(
        qname for qname in body if str(source_census[qname]) != "bf16"
    )
    if off_source:
        raise DenseCampaignError(
            "this driver declares one bf16 source class; found "
            f"{len(off_source)} others, sample="
            f"{[(q, source_census[q]) for q in off_source[:4]]}"
        )
    return {qname: source_census[qname] for qname in body}


def prepare_campaign(args: argparse.Namespace) -> PreparedCampaign:
    """Every CPU identity/legality check, before any GPU work."""
    work_dir = _safe_work_dir(args.work_dir)
    stats, probe_meta = _load_probe(args)

    from prismaquant.model_profiles import detect_profile
    from prismaquant.allocator_candidates import _scan_source_dtype_manifest
    from prismaquant.nvfp4_cb_footprint import (
        cb_serialization_context_from_env, cb_serialization_context_stamp,
    )

    profile = detect_profile(args.model)
    body, roles = _body_census(profile, stats, args)
    target_profile = str(
        getattr(args, "target_profile", DEFAULT_TARGET_PROFILE) or ""
    ).strip()
    if not target_profile:
        raise DenseCampaignError("target-profile must be a non-empty identity")

    source_census = _scan_source_dtype_manifest(args.model, profile)
    source_manifest = _require_bf16_body_source_manifest(body, source_census)

    cb_context = cb_serialization_context_from_env(
        require_explicit=True, where="dense anchored CB AURA campaign"
    )
    source_map = _authoritative_source_map(cb_context)

    format_plan = _derive_dense_plan(
        profile, body, target_profile=target_profile,
    )
    format_plan_path = work_dir / "checkpoints" / "dense_format_plan.json"
    format_plan_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        format_plan_path,
        json.dumps(format_plan.to_dict(), indent=2, sort_keys=True).encode(),
    )

    declarations = _build_declarations(
        stats=stats, body=body, roles=roles,
        format_plan=format_plan, cb_context=cb_context,
    )
    render_levers = {
        "gptq": True,
        "static_act_order": True,
        "joint_scale_opt": True,
        "weighted_vq": True,
    }
    arm_identity = {
        "campaign_schema": CAMPAIGN_SCHEMA,
        "cb_serialization_context": cb_serialization_context_stamp(
            cb_context, formats=(*NVFP4_CB_LADDER, *FP8_CB_LADDER)
        ),
        "render_levers": render_levers,
        "target_profile": format_plan.target_profile,
        "target_platform": format_plan.target_platform,
        "production_arm_only": True,
        "rtn_anchor_allowed": False,
    }
    plugin = CodebookAnchoredFormatPlugin(
        codebook_source_by_format=source_map,
        arm_identity=arm_identity,
        anchor_formats=ANCHOR_FORMATS,
    )
    units = build_cb_units(declarations, plugin)
    anchor_requests = plan_anchor_requests(units, plugin)
    panel, validation, panel_report = plan_cb_panel_and_validation(
        units, plugin, _panel_policy(roles)
    )
    formats, purposes, union_report = build_streamed_cb_render_plan(
        units, plugin, anchor_requests, panel, validation
    )
    full_menu_cells = len(units) * len(
        (*NVFP4_CB_LADDER, *FP8_CB_LADDER)
    )
    rendered_cells = sum(len(v) for v in formats.values())
    plan_report = {
        **dict(union_report),
        "campaign_schema": CAMPAIGN_SCHEMA,
        "target_profile": format_plan.target_profile,
        "target_platform": format_plan.target_platform,
        "units": len(units),
        "roles": dict(sorted(Counter(roles.values()).items())),
        "anchor_renders": len(anchor_requests),
        "anchor_renders_by_segment": dict(sorted(Counter(
            request.segment.stamp for request in anchor_requests
        ).items())),
        "panel_renders": len(panel),
        "validation_renders": len(validation),
        "rendered_cells": rendered_cells,
        "full_menu_cells": full_menu_cells,
        "render_fraction_of_full_menu": (
            rendered_cells / full_menu_cells if full_menu_cells else None
        ),
        "format_plan_identity_sha256": format_plan.identity_sha256,
        "format_plan_path": str(format_plan_path),
        "route_flip_limitation": ROUTE_FLIP_LIMITATION,
        "streaming_source_cache": {
            "max_cache_slots": STREAMING_CACHE_MAX_SLOTS,
            "effective_prefetch_lookahead": 0,
        },
    }
    return PreparedCampaign(
        args=args, profile=profile, probe_stats=stats, probe_meta=probe_meta,
        body_qnames=body, roles=roles, cb_context=cb_context, plugin=plugin,
        format_plan=format_plan, units=units,
        anchor_requests=anchor_requests, panel_requests=panel,
        validation_requests=validation, formats_by_qname=formats,
        purposes_by_qname=purposes, plan_report=plan_report,
        panel_report=panel_report, arm_identity=arm_identity,
    )


def measure_streamed(prepared: PreparedCampaign) -> dict[str, object]:
    """One streamed adjoint pass, one live layer at a time, renders transient."""
    args = prepared.args
    _assert_home_reserve()
    from prismaquant.gpu_guard import require_cuda_hot_path

    device = require_cuda_hot_path("dense_anchored_cb", "cuda")
    import torch
    from transformers import AutoTokenizer
    from prismaquant.build_production_cache import _load_col_weights
    from prismaquant.cost_streaming import (
        build_streamed_causal_lm, build_streamed_model_identity,
    )
    from prismaquant.measure_quant_cost import ActivationIndex
    from prismaquant.perturbed_x_cache import calibration_data_hash
    from prismaquant.sensitivity_probe import load_calibration

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
        local_files_only=Path(args.model).exists(),
    )
    calibration = load_calibration(
        tokenizer, args.dataset, args.n_calib_samples, args.calib_seqlen,
        calib_seed=args.calib_seed,
    ).to(device)
    observed_hash = calibration_data_hash(calibration)
    if observed_hash != prepared.probe_meta["calib_hash"]:
        raise DenseCampaignError(
            "tokenized calibration hash differs from probe identity: "
            f"{observed_hash} vs {prepared.probe_meta['calib_hash']}"
        )
    col_weights = _load_col_weights(
        args.col_weights, (*NVFP4_CB_LADDER, *FP8_CB_LADDER)
    )
    if not isinstance(col_weights, Mapping):
        raise DenseCampaignError("CB render imatrix did not load")
    missing_col = sorted(set(prepared.body_qnames) - set(col_weights))
    if missing_col:
        raise DenseCampaignError(
            f"CB render imatrix misses body units: {missing_col[:8]}"
        )
    body_stats = {q: prepared.probe_stats[q] for q in prepared.body_qnames}
    activation_index = ActivationIndex(
        Path(args.activation_cache_dir), body_stats
    )
    missing_act = sorted(
        q for q in prepared.body_qnames if q not in activation_index
    )
    if missing_act:
        raise DenseCampaignError(
            "the activation cache misses body units, so their renders would "
            f"be unweighted: {missing_act[:8]}"
        )
    checkpoint_root = Path(args.checkpoint_dir)
    runner = build_streamed_causal_lm(
        args.model, device=device, dtype=torch.bfloat16,
        offload_folder=str(checkpoint_root / "streamed-model-offload"),
        profile=prepared.profile,
        max_cache_slots=STREAMING_CACHE_MAX_SLOTS,
    )
    try:
        model_identity = build_streamed_model_identity(
            runner, args.model,
            identity_cache_path=(
                checkpoint_root / "streamed_model_identity.json"
            ),
        )
        return run_streamed_cb_anchor_aura(
            runner, calibration,
            formats_by_qname=prepared.formats_by_qname,
            legal_formats_by_qname={
                unit.qname: tuple(
                    candidate.format_name for candidate in unit.candidates
                    if not candidate.terminal
                )
                for unit in prepared.units
            },
            purposes_by_qname=prepared.purposes_by_qname,
            activation_index=activation_index,
            render_levers=prepared.arm_identity["render_levers"],
            col_weights=col_weights,
            cb_serialization_context=prepared.cb_context,
            calibration_hash=observed_hash,
            arm_identity=prepared.arm_identity,
            model_identity=model_identity,
            checkpoint_dir=checkpoint_root / "aura",
            resume=bool(args.resume),
            n_probes=int(args.n_probes),
            profile=prepared.profile,
            checkpoint_identity_extra={
                "campaign_schema": CAMPAIGN_SCHEMA,
                "source_format_plan_identity_sha256": (
                    prepared.format_plan.identity_sha256
                ),
                "segment_key_fields": [
                    "family", "role", "equivalence_class",
                ],
                "equivalence_vocabulary_name": "codebook_basis",
                "streaming_source_cache": {
                    "max_cache_slots": STREAMING_CACHE_MAX_SLOTS,
                    "effective_prefetch_lookahead": 0,
                },
            },
        )
    finally:
        runner.shutdown()


def _merge_aqua_a_side(
    prepared: PreparedCampaign, *, weight_only_path: Path, out_path: Path,
) -> dict[str, object]:
    """Price the A-side off the Sensitivity Card and merge it into the rows.

    Written to a NEW artifact: ``weight_only_path`` stays untouched so the
    pre-AQUA anchored allocation remains reproducible as the A/B arm this
    lane's promotion depends on.
    """
    import numpy as np
    from prismaquant.aqua_activation_cost import (
        activation_dloss_table, merge_act_dloss,
        resolve_executed_activation_formats,
    )
    from prismaquant.allocator_candidates import ACT_DLOSS_KEY
    from prismaquant.sensitivity_card import SensitivityCard

    args = prepared.args
    with weight_only_path.open("rb") as handle:
        blob = pickle.load(handle)
    costs = blob["costs"]
    # The card is a SCHEMA, not an npz: its arrays are stored under per-unit
    # prefixed keys (`<code>_g_sq_sum` ...) with the unit map in `__header__`,
    # so a raw np.load hands activation_dloss_table something that KeyErrors on
    # the first qname. Load and validate it exactly as the stage's own CLI does.
    card = SensitivityCard.from_npz(args.aqua_card)
    card.validate()
    formats = sorted({fmt for row in costs.values() for fmt in row})
    # This is the CB export lane. Its runtime decides the A-side, not the
    # format registry -- see LaneActivationContract. Resolving through the lane
    # spec is what makes an A-side of exactly zero say so loudly instead of
    # being priced as if the fused W4A4 kernel were serving.
    executed = resolve_executed_activation_formats(lane_id="nvfp4_cb")
    table, holes, meta = activation_dloss_table(
        card, args.model, formats, device=args.aqua_device,
        names=list(costs), act_dir=args.act_dir,
        executed_activation_formats=executed,
    )
    report = merge_act_dloss(costs, table)
    if report["entries_merged"] == 0:
        raise DenseCampaignError(
            "the AQUA merge wrote no A-side into any row; allocating now "
            "would silently reproduce the weight-only arm"
        )
    provenance = dict(blob.get("provenance") or {})
    provenance["aqua_activation_cost"] = {
        "card_path": os.path.abspath(args.aqua_card),
        "card_sha256": _sha256(args.aqua_card),
        "card_fingerprint": card.provenance.fingerprint(),
        "card_units": len(card),
        "formats_priced": formats,
        "holes": {key: len(value) for key, value in holes.items()},
        "merge_report": report,
        "act_dir": os.path.abspath(args.act_dir) if args.act_dir else None,
        "act_var_source": "measured" if args.act_dir else "modelled",
        "weight_only_arm": str(weight_only_path),
        **meta,
    }
    blob["provenance"] = provenance
    atomic_write_bytes(
        out_path, pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL)
    )
    # What the DP will now see differently, per family. Not a result -- the
    # served KL A/B at matched bpp is -- but enough to catch a merge that
    # technically succeeded and changed nothing.
    ratios: dict[str, list[float]] = defaultdict(list)
    for entry in costs.values():
        for fmt, cell in entry.items():
            if not isinstance(cell, dict) or ACT_DLOSS_KEY not in cell:
                continue
            base = cell.get("predicted_dloss", 0.0)
            if base and base > 0:
                ratios[fr.get_format(fmt).family].append(
                    cell[ACT_DLOSS_KEY] / base
                )
    summary = {
        family: {
            "units": len(values),
            "p10": float(np.percentile(values, 10)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
        }
        for family, values in sorted(ratios.items()) if values
    }
    provenance["aqua_activation_cost"]["a_over_w_ratio"] = summary
    atomic_write_bytes(
        out_path, pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL)
    )
    return {"merge_report": report, "a_over_w_ratio": summary,
            "holes": {k: len(v) for k, v in holes.items()}}


def _allocator_command(
    prepared: PreparedCampaign, *, cost_path: Path, output_dir: Path,
) -> list[str]:
    args = prepared.args
    formats = (
        *prepared.format_plan.ladders["nvfp4_cb"],
        *prepared.format_plan.ladders["fp8_cb"],
        "BF16",
    )
    command = [
        sys.executable, "-m", "prismaquant.allocator",
        "--probe", str(args.probe),
        "--costs", str(cost_path),
        "--model-override", str(args.model),
        "--target-profile", prepared.format_plan.target_profile,
        "--target-disk-gb", f"{args.target_disk_gb:.9f}",
        "--artifact-overhead-reserve-bytes",
        str(int(args.artifact_overhead_reserve_bytes)),
        # The COMPLETE CB render-context flag set. The allocator's identity
        # gates demand the set, not a subset: render-context completeness,
        # codebook_source match, menu-subset-of-measured-menu, and col_weights
        # identity all fire in order, and a missing flag fails the first.
        "--cb-scale-coding", "two_tier",
        "--cb-codebook-source", "lattice",
        "--cb-codebook-source-scope", "none",
        "--cb-scale-sweep", "1",
        "--cb-scale-sweep-scope", "all",
        "--cb-ldlq", "0", "--cb-ldlq-scope", "none",
        "--cb-encode-tier", "balanced",
        "--cb-col-weights", str(args.col_weights),
        "--formats", ",".join(formats),
        "--mtp-format", "BF16", "--visual-format", "BF16",
        "--threads", str(int(args.threads)),
        "--layer-config", str(output_dir / "layer_config.json"),
        "--pareto-csv", str(output_dir / "pareto.csv"),
        "--pareto-output-dir", str(output_dir / "pareto-points"),
        "--applicability-report",
        str(output_dir / "format_applicability.json"),
        "--bit-attribution-json", str(output_dir / "bit_attribution.json"),
        "--bit-attribution-csv", str(output_dir / "bit_attribution.csv"),
    ]
    return command


def _recipe_format(qname: str, value: object) -> str:
    """Recover the selected format from one allocator recipe cell.

    A CB cell is NOT a format string: the allocator emits the full render
    identity (`data_type`, `cb_k`, activation contract, and a
    `cb_serialized_identity` JSON blob) so the exporter reproduces the exact
    encode. The serialized identity is authoritative because it is what the
    export render-identity gate compares against; `data_type`+`cb_k` is the
    fallback, and native/source cells go through the same shared parser the
    exporter uses rather than a second interpretation of the same fields.
    """
    if isinstance(value, str):
        return fr.canonical_format_name(value)
    if not isinstance(value, Mapping):
        raise DenseCampaignError(
            f"allocator recipe cell for {qname} is not an object"
        )
    serialized = value.get("cb_serialized_identity")
    if isinstance(serialized, str):
        try:
            format_name = json.loads(serialized).get("format")
        except (TypeError, ValueError):
            format_name = None
        if format_name:
            return fr.canonical_format_name(str(format_name))
    data_type = str(value.get("data_type", "")).lower()
    if data_type in {"nvfp4_cb", "fp8_cb"}:
        return f"{data_type.upper()}_K{int(value['cb_k'])}"
    try:
        from prismaquant.layer_config import canonicalize_format

        return fr.canonical_format_name(canonicalize_format(value))
    except (KeyError, TypeError, ValueError) as exc:
        raise DenseCampaignError(
            f"cannot recover the selected format for {qname} from "
            f"data_type={data_type!r}"
        ) from exc


def _selected_assignment(
    layer_config: Path, units: Sequence[UnitSpec],
) -> dict[str, str]:
    recipe = json.loads(layer_config.read_text())
    known = {unit.qname for unit in units}
    assignment: dict[str, str] = {}
    for qname, value in recipe.items():
        if qname == "__prismaquant__" or qname not in known:
            continue
        assignment[qname] = _recipe_format(qname, value)
    missing = sorted(known - set(assignment))
    if missing:
        raise DenseCampaignError(
            f"the recipe does not assign {len(missing)} body units: "
            f"{missing[:8]}"
        )
    return assignment


def finish_campaign(
    prepared: PreparedCampaign,
    streamed_payload: Mapping[str, object],
    *,
    artifacts_dir: Path,
    allocator_output: Path,
    report_path: Path,
) -> int:
    """CPU tail: fit -> validate -> price -> AQUA -> one DP -> report."""
    args = prepared.args
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    anchors = anchors_from_streamed_payload(
        prepared.anchor_requests, streamed_payload
    )
    panel_observations = observations_from_streamed_payload(
        prepared.panel_requests, streamed_payload
    )
    fits = fit_all_cb_segments(
        panel_observations, prepared.units, prepared.plugin, anchors=anchors
    )
    validation_observations = observations_from_streamed_payload(
        prepared.validation_requests, streamed_payload
    )
    validation = heldout_validation_report(
        validation_observations, anchors, fits, basis=LATTICE_BASIS,
        panel_requests=prepared.panel_requests,
    )
    hulls = fitted_cb_hull_report(prepared.units, prepared.plugin, fits)
    cells = price_anchored_candidates(
        prepared.units, prepared.plugin, anchors, fits
    )
    cost_payload = build_cb_allocator_cost_payload(
        cells, streamed_payload=streamed_payload, fits=fits,
        hull_report=hulls, validation_report=validation,
    )
    weight_only_path = write_cb_cost_payload(
        artifacts_dir / "cost_aura_anchored.pkl", cost_payload
    )

    aqua_report: dict[str, object] | None = None
    cost_path = weight_only_path
    if args.aqua_card:
        cost_path = artifacts_dir / "cost_aura_anchored_aqua.pkl"
        aqua_report = _merge_aqua_a_side(
            prepared, weight_only_path=weight_only_path, out_path=cost_path,
        )

    command = _allocator_command(
        prepared, cost_path=cost_path, output_dir=allocator_output
    )
    run_allocator_once(
        command=command,
        output_dir=allocator_output,
        resume=bool(args.resume),
        invocation_provenance={
            "cost_currency": AURA_CURRENCY,
            "campaign_schema": CAMPAIGN_SCHEMA,
            "target_profile": prepared.format_plan.target_profile,
            "target_platform": prepared.format_plan.target_platform,
            # P5a's per-family multiply is an estimator TRANSFER from weight
            # space to output space. An anchored projection is already in the
            # right currency, so cost_entry_predicted_dloss reads this table's
            # rows as `base + act_dloss` and never applies the transfer -- the
            # AQUA A-side reaches the DP either way, which is why no kill
            # switch is set here.
            "activation_fair_pricing_applies_to_anchored_rows": False,
            "aqua_a_side_merged": bool(args.aqua_card),
            "cost_payload_sha256": _sha256(cost_path),
            "weight_only_arm_sha256": _sha256(weight_only_path),
            "format_plan_identity_sha256": (
                prepared.format_plan.identity_sha256
            ),
            "production_arm_identity_sha256": canonical_json_sha256(
                prepared.arm_identity,
                where="dense anchored allocator production arm identity",
            ),
        },
    )
    assignment = _selected_assignment(
        allocator_output / "layer_config.json", prepared.units
    )
    exposure = extrapolation_distance_report(
        prepared.units, prepared.plugin, anchors, assignment
    )
    report = {
        "campaign_schema": CAMPAIGN_SCHEMA,
        "plan": prepared.plan_report,
        "panel": prepared.panel_report,
        "heldout_validation": validation,
        "fitted_hulls": hulls,
        "aqua": aqua_report,
        "extrapolation_exposure_of_selection": exposure,
        "selected_format_counts": dict(sorted(
            Counter(assignment.values()).items()
        )),
        "cost_payload": str(cost_path),
        "weight_only_arm": str(weight_only_path),
        "layer_config": str(allocator_output / "layer_config.json"),
        "route_flip_limitation": ROUTE_FLIP_LIMITATION,
    }
    atomic_write_bytes(
        report_path,
        json.dumps(report, indent=2, sort_keys=True, default=str).encode(),
    )
    print(f"[campaign] report: {report_path}")
    print(f"[campaign] selected: {report['selected_format_counts']}")
    print(f"[campaign] held-out validation: "
          f"{validation.get('verdict', validation)}")
    return 0


def run_campaign(args: argparse.Namespace) -> int:
    work_dir = _safe_work_dir(args.work_dir)
    artifacts_dir = work_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_campaign(args)
    require_allocator_supersurrogate_support()
    plan_path = artifacts_dir / "anchored_plan_report.json"
    atomic_write_bytes(
        plan_path,
        json.dumps(prepared.plan_report, indent=2, sort_keys=True,
                   default=str).encode(),
    )
    print(f"[campaign] plan: {plan_path}")
    print(f"[campaign] units={prepared.plan_report['units']} "
          f"anchors={prepared.plan_report['anchor_renders']} "
          f"panel={prepared.plan_report['panel_renders']} "
          f"validation={prepared.plan_report['validation_renders']} "
          f"= {prepared.plan_report['rendered_cells']} cells, "
          f"{prepared.plan_report['render_fraction_of_full_menu']:.1%} of a "
          f"full-menu render")
    if args.plan_only:
        print("[campaign] --plan-only: stopping before any GPU work")
        return 0

    raw_path = artifacts_dir / "streamed_anchor_aura.pkl"
    if raw_path.exists() and args.resume:
        print(f"[campaign] reusing completed streamed payload: {raw_path}")
        with raw_path.open("rb") as handle:
            streamed_payload = pickle.load(handle)
    else:
        streamed_payload = measure_streamed(prepared)
        atomic_write_bytes(
            raw_path,
            pickle.dumps(streamed_payload, protocol=pickle.HIGHEST_PROTOCOL),
        )
    return finish_campaign(
        prepared, streamed_payload,
        artifacts_dir=artifacts_dir,
        allocator_output=work_dir / "allocator-anchored",
        report_path=artifacts_dir / "campaign_report.json",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Anchored CB AURA (+AQUA) campaign for a dense model"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--activation-cache-dir", required=True)
    parser.add_argument("--col-weights", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--expect-body-units", type=int, required=True,
        help="the campaign's declared body census; a mismatch refuses rather "
             "than silently allocating over a different unit set",
    )
    parser.add_argument(
        "--exclude-markers", default="mtp,lm_head,visual",
        help="substrings marking units the DP may not allocate",
    )
    parser.add_argument("--target-disk-gb", type=float, required=True)
    parser.add_argument(
        "--target-profile",
        default=DEFAULT_TARGET_PROFILE,
        help="closed serving-profile identity used for plan derivation and "
             "allocator legality (default: validation-only SM120 candidate "
             "registration; this is not a release qualification)",
    )
    parser.add_argument(
        "--artifact-overhead-reserve-bytes", type=int, default=40_000_000
    )
    parser.add_argument("--n-calib-samples", type=int, default=16)
    parser.add_argument("--calib-seqlen", type=int, default=512)
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--n-probes", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--plan-only", action="store_true",
        help="run every CPU census/legality/plan check and stop; the cheap "
             "abort before hours of render",
    )
    parser.add_argument(
        "--aqua-card",
        help="Sensitivity Card .npz. When given, the AQUA A-side is priced "
             "and merged, and the DP allocates on the merged table. The "
             "weight-only anchored payload is still written, as the A/B arm.",
    )
    parser.add_argument(
        "--act-dir",
        help="cached real activations; makes AQUA's act_var MEASURED rather "
             "than modelled from a per-channel Gaussian fit",
    )
    parser.add_argument("--aqua-device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.act_dir and not args.aqua_card:
        raise SystemExit("--act-dir only applies with --aqua-card")
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
