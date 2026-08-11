"""Frozen-CLI DSv4 configuration and orchestration for anchored CB AURA.

All reusable ladder, equivalence, fitting, render-hook, and artifact behavior
lives in :mod:`prismaquant.cb_anchored_cost`; platform-neutral fitting and
pricing live in :mod:`prismaquant.anchored_cost`.  This module supplies only
the DSv4 census, exact source-class menus, panel policy, byte budget, and the
one-shot stage order consumed by ``tools/run_aura_cb_reprice.sh``.

The current tree deliberately refuses before P0 because the existing
allocator has not yet declared AURA-supersurrogate admission semantics.  No
render-free, RTN, cross-basis, or full-menu fallback exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
    RenderRequest,
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
    LEARNED_BASIS,
    ROUTE_FLIP_LIMITATION,
    anchors_from_streamed_payload,
    basis_segment_dict,
    build_cb_allocator_cost_payload,
    build_cb_units,
    build_streamed_cb_render_plan,
    fit_all_cb_segments,
    fitted_cb_hull_report,
    heldout_validation_report,
    observations_from_streamed_payload,
    plan_cb_panel_and_validation,
    require_allocator_supersurrogate_support,
    run_streamed_cb_anchor_aura,
    write_cb_cost_payload,
    write_exportable_artifacts,
)
from prismaquant.cost_stage_checkpoint import (
    atomic_write_bytes,
    canonical_json,
    canonical_json_sha256,
)


DSV4_BOUNDED_CAMPAIGN_WORKER = True
DSV4_CAMPAIGN_SCHEMA = "prismaquant.dsv4_cb_anchored_aura.v1"
DSV4_TOTAL_UNITS = 33_325
DSV4_EXPERT_UNITS = 43 * 256 * 3
DSV4_NONEXPERT_UNITS = 301
DSV4_EXPECTED_ANCHORS = 99_975
DSV4_BUDGET_BYTES = 112_690_000_000
DSV4_ARTIFACT_RESERVE_BYTES = 268_435_456

NVFP4_FORMATS = tuple(f"NVFP4_CB_K{k}" for k in range(12, 19))
FP8_LATTICE_FORMATS = tuple(f"FP8_CB_K{k}" for k in range(28, 49))
FP8_LEARNED_FORMATS = tuple(
    f"FP8_CBL_K{k}" for k in (28, 32, 36, 40, 44)
)
FP8_NONEXPERT_FORMATS = tuple(sorted(
    (*FP8_LATTICE_FORMATS, *FP8_LEARNED_FORMATS),
    key=lambda name: (
        int(name.rsplit("K", 1)[1]),
        name.startswith("FP8_CBL_"),
    ),
))
FP8_EXPERT_FORMATS = tuple(
    name for name in FP8_NONEXPERT_FORMATS
    if int(name.rsplit("K", 1)[1]) <= 33
)
_ROUTED_LEARNED_EXPERT_RUNGS = (28, 32)
DSV4_ANCHOR_FORMATS = {
    ("nvfp4_cb", LATTICE_BASIS): "NVFP4_CB_K15",
    ("fp8_cb", LEARNED_BASIS): "FP8_CBL_K32",
    # One role/basis segment spans experts and nonexperts. K32 is the highest
    # lattice rung legal for every source class, so it is the common anchor.
    ("fp8_cb", LATTICE_BASIS): "FP8_CB_K32",
}

_ALL_ROLES = (
    "gate_proj", "up_proj", "down_proj",
    "wq_a", "wq_b", "wkv", "wo_b",
)
_EXPECTED_ROLES = frozenset(_ALL_ROLES)
_DSV4_ROLE_UNIT_COUNTS = {
    "gate_proj": 11_051,
    "up_proj": 11_051,
    "down_proj": 11_051,
    "wq_a": 43,
    "wq_b": 43,
    "wkv": 43,
    "wo_b": 43,
}
DSV4_PANEL_POLICY = CBPanelPolicy(
    panel_rungs_by_segment={
        **{
            ("nvfp4_cb", role, LATTICE_BASIS): (
                "NVFP4_CB_K12", "NVFP4_CB_K15",
                "NVFP4_CB_K16", "NVFP4_CB_K18",
            )
            for role in _ALL_ROLES
        },
        **{
            ("fp8_cb", role, LEARNED_BASIS): (
                "FP8_CBL_K28", "FP8_CBL_K32",
                "FP8_CBL_K40", "FP8_CBL_K44",
            )
            for role in _ALL_ROLES
        },
        **{
            ("fp8_cb", role, LATTICE_BASIS): (
                "FP8_CB_K47", "FP8_CB_K48",
            )
            for role in _ALL_ROLES
        },
    },
    validation_rungs_by_segment={
        **{
            ("fp8_cb", role, LEARNED_BASIS): (
                "FP8_CBL_K28", "FP8_CBL_K44",
            )
            for role in _ALL_ROLES
        },
    },
    panel_units_per_role=32,
    validation_units_per_role=4,
    seed=42,
)
_TERMINAL_BY_SOURCE_KIND = {
    "mxfp4": "MXFP4_SOURCE",
    "fp8_ue8m0": "FP8_BLOCK_UE8M0_SOURCE",
    "bf16": "BF16",
}


class DSv4CampaignError(RuntimeError):
    """A DSv4 inventory, identity, or orchestration refusal."""


@dataclass(frozen=True)
class PreparedDSv4Campaign:
    args: argparse.Namespace
    profile: object
    probe_stats: Mapping[str, Mapping[str, object]]
    probe_meta: Mapping[str, object]
    cb_context: object
    plugin: CodebookAnchoredFormatPlugin
    format_plan: object
    units: tuple[UnitSpec, ...]
    anchor_requests: tuple[RenderRequest, ...]
    panel_requests: tuple[RenderRequest, ...]
    validation_requests: tuple[RenderRequest, ...]
    formats_by_qname: Mapping[str, tuple[str, ...]]
    purposes_by_qname: Mapping[str, Mapping[str, list[str]]]
    plan_report: Mapping[str, object]
    panel_report: Mapping[str, object]
    routed_selection_sha256: str
    arm_identity: Mapping[str, object]
    cold_expert_provenance: Mapping[str, object]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_work_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve(strict=False)
    forbidden = (
        Path("/tmp"),
        Path("/home/rob/prismaquant"),
        Path("/home/rob/pq-cbl-export"),
    )
    if any(resolved == root or root in resolved.parents for root in forbidden):
        raise DSv4CampaignError(f"unsafe campaign work directory: {resolved}")
    baseline_artifacts = Path(
        "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p7/artifacts"
    ).resolve(strict=False)
    campaign_artifacts = (resolved / "artifacts").resolve(strict=False)
    if (
        campaign_artifacts == baseline_artifacts
        or baseline_artifacts in campaign_artifacts.parents
    ):
        raise DSv4CampaignError(
            "campaign artifacts would overwrite or nest inside the measured "
            f"Track A baseline: {campaign_artifacts}"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_probe(args: argparse.Namespace) -> tuple[dict, dict]:
    try:
        with Path(args.probe).open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise DSv4CampaignError(f"probe is unreadable: {args.probe}") from exc
    stats = payload.get("stats") if isinstance(payload, Mapping) else None
    meta = payload.get("meta") if isinstance(payload, Mapping) else None
    if not isinstance(stats, Mapping) or not isinstance(meta, Mapping):
        raise DSv4CampaignError("probe lacks stats/meta mappings")
    if len(stats) != DSV4_TOTAL_UNITS:
        raise DSv4CampaignError(
            f"DSv4 probe has {len(stats)} units, expected {DSV4_TOTAL_UNITS}"
        )
    expected = {
        "nsamples": int(args.n_calib_samples),
        "seqlen": int(args.calib_seqlen),
    }
    for field, value in expected.items():
        if int(meta.get(field, -1)) != value:
            raise DSv4CampaignError(
                f"probe calibration {field}={meta.get(field)!r}, expected {value}"
            )
    for field, supplied in (
        ("model", args.model),
        ("dataset", args.dataset),
        ("activation_cache_dir", args.activation_cache_dir),
    ):
        recorded = meta.get(field)
        if not recorded or Path(str(recorded)).resolve(strict=False) != Path(
            supplied
        ).resolve(strict=False):
            raise DSv4CampaignError(
                f"probe {field} identity differs: {recorded!r} vs {supplied!r}"
            )
    calibration_hash = meta.get("calib_hash")
    if not isinstance(calibration_hash, str) or not calibration_hash:
        raise DSv4CampaignError("probe has no exact calibration hash")
    return dict(stats), dict(meta)


def _validate_routed_selection(path: str | Path) -> str:
    from prismaquant.cb_banked_books import load_routed_moe_cbl_selection

    selection = load_routed_moe_cbl_selection(path)
    observed = {
        (cell.layer, cell.projection, cell.rung) for cell in selection.cells
    }
    expected = {
        (layer, projection, rung)
        for layer in range(43)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for rung in _ROUTED_LEARNED_EXPERT_RUNGS
    }
    if observed != expected:
        raise DSv4CampaignError(
            "routed learned-book selection coverage differs: "
            f"missing={sorted(expected - observed)[:8]} "
            f"extra={sorted(observed - expected)[:8]}"
        )
    return str(selection.content_sha256)


def _validate_routed_bundle_selection_identity(
    bundle_path: str | Path,
    selection_sha256: str,
) -> dict[str, object]:
    """Require every routed learned bundle cell to name this selection."""
    from prismaquant.cb_banked_books import (
        BANKED_CBL_ORIGIN_SCHEMA,
        validate_banked_cbl_origin,
    )
    from prismaquant.cb_learned_bundle import load_bundle_cached

    bundle = load_bundle_cached(bundle_path)
    expected = {
        (layer, projection, rung)
        for layer in range(43)
        for projection in ("gate_proj", "up_proj", "down_proj")
        for rung in _ROUTED_LEARNED_EXPERT_RUNGS
    }
    observed: dict[tuple[int, str, int], tuple[str, str]] = {}
    wrong_selection: list[tuple[str, str, str]] = []
    raw_cells = bundle.manifest.get("cells")
    if not isinstance(raw_cells, Mapping):
        raise DSv4CampaignError("learned bundle has no cell manifest")
    for raw_qname, raw_formats in raw_cells.items():
        if not isinstance(raw_formats, Mapping):
            raise DSv4CampaignError(
                f"learned bundle cell map is malformed for {raw_qname!r}"
            )
        for raw_format, raw_cell in raw_formats.items():
            if not isinstance(raw_cell, Mapping):
                raise DSv4CampaignError(
                    f"learned bundle cell is malformed for "
                    f"{raw_qname}/{raw_format}"
                )
            raw_origin = raw_cell.get("pretrained_origin")
            if not isinstance(raw_origin, Mapping) or raw_origin.get(
                "schema"
            ) != BANKED_CBL_ORIGIN_SCHEMA:
                continue
            try:
                origin = validate_banked_cbl_origin(
                    raw_origin,
                    where=(
                        f"DSv4 bundle {raw_qname}/{raw_format} bank origin"
                    ),
                )
            except Exception as exc:
                raise DSv4CampaignError(str(exc)) from None
            coordinate = (
                int(origin["layer"]),
                str(origin["projection"]),
                int(origin["rung"]),
            )
            expected_format = f"FP8_CBL_K{coordinate[2]}"
            if str(raw_format) != expected_format:
                raise DSv4CampaignError(
                    "routed learned bundle origin is attached to a format "
                    "whose name does not declare learned source: "
                    f"{raw_qname}/{raw_format}, expected {expected_format}"
                )
            if str(raw_cell.get("source", "")).strip().lower() != LEARNED_BASIS:
                raise DSv4CampaignError(
                    f"{raw_qname}/{raw_format}: routed learned bundle cell "
                    "does not declare source='learned'"
                )
            if coordinate in observed:
                raise DSv4CampaignError(
                    "learned bundle repeats routed bank coordinate "
                    f"{coordinate}: {observed[coordinate]} and "
                    f"{(str(raw_qname), str(raw_format))}"
                )
            observed[coordinate] = (str(raw_qname), str(raw_format))
            if str(origin["selection_sha256"]) != str(selection_sha256):
                wrong_selection.append((
                    str(raw_qname), str(raw_format),
                    str(origin["selection_sha256"]),
                ))
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra or wrong_selection:
        raise DSv4CampaignError(
            "routed learned bundle is not bound to the supplied book "
            "selection: "
            f"missing={missing[:8]} extra={extra[:8]} "
            f"wrong_selection={wrong_selection[:8]}"
        )
    return {
        "selection_sha256": str(selection_sha256),
        "bundle_content_sha256": str(bundle.bundle_content_sha256),
        "routed_learned_origin_cells": len(observed),
        "expected_coordinates_complete": True,
    }


def _role_and_class_maps(
    profile, qnames: Sequence[str],
) -> tuple[dict[str, str], dict[str, str]]:
    from prismaquant.routed_experts import ProfileRoutedExpertClassifier

    classifier = ProfileRoutedExpertClassifier(profile)
    roles: dict[str, str] = {}
    classes: dict[str, str] = {}
    for qname in qnames:
        match = classifier.classify(qname)
        if match is not None:
            roles[qname] = match.projection_name
            classes[qname] = "profile_declared_routed_expert"
        else:
            role = str(qname).rsplit(".", 1)[-1]
            roles[qname] = role
            classes[qname] = "nonexpert"
    routed = sum(value == "profile_declared_routed_expert" for value in classes.values())
    if routed != DSV4_EXPERT_UNITS:
        raise DSv4CampaignError(
            f"profile declares {routed} routed experts, expected {DSV4_EXPERT_UNITS}"
        )
    if set(roles.values()) != _EXPECTED_ROLES:
        raise DSv4CampaignError(
            f"DSv4 profile role census differs: {sorted(set(roles.values()))}"
        )
    return roles, classes


def _validated_cold_expert_provenance(
    *,
    stats: Mapping[str, Mapping[str, object]],
    unit_classes: Mapping[str, str],
    missing_activations: Sequence[str],
    col_weights_path: str | Path,
) -> dict[str, object]:
    """Bind the renderer's cold branch to probe and imatrix evidence."""
    zero_token_names = sorted(
        qname for qname, row in stats.items()
        if int(row.get("n_tokens_seen", -1)) == 0
    )
    missing = sorted(map(str, missing_activations))
    if missing != zero_token_names:
        raise DSv4CampaignError(
            "activation-cache misses differ from probe-declared never-routed "
            f"units: missing_only="
            f"{sorted(set(missing) - set(zero_token_names))[:8]} "
            f"zero_token_only="
            f"{sorted(set(zero_token_names) - set(missing))[:8]}"
        )
    wrong_class = [
        qname for qname in missing
        if unit_classes.get(qname) != "profile_declared_routed_expert"
    ]
    if wrong_class:
        raise DSv4CampaignError(
            "cold-render declaration contains non-profile-routed units: "
            f"{wrong_class[:8]}"
        )
    sidecar = Path(f"{col_weights_path}.provenance.json")
    try:
        payload = json.loads(sidecar.read_text())
    except Exception as exc:
        raise DSv4CampaignError(
            f"cold-expert imatrix provenance is unreadable: {sidecar}"
        ) from exc
    rule = "unrouted_expert_neutral_prior:layer_routed_mean"
    raw_names = payload.get("names") if isinstance(payload, Mapping) else None
    sidecar_names = (
        sorted(map(str, raw_names))
        if isinstance(raw_names, Sequence)
        and not isinstance(raw_names, (str, bytes))
        else None
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("rule") != rule
        or payload.get("basis") != "probe n_tokens_seen == 0"
        or sidecar_names != missing
    ):
        raise DSv4CampaignError(
            "cold-expert imatrix provenance differs from the exact probe/"
            "activation-cache never-routed set"
        )
    return {
        "rule": rule,
        "basis": "probe n_tokens_seen == 0",
        "names": missing,
        "count": len(missing),
        "profile_declared_routed_experts_only": True,
        "imatrix_provenance_path": str(sidecar.resolve()),
        "imatrix_provenance_sha256": _sha256(sidecar),
    }


def _build_declarations(
    *,
    stats: Mapping[str, Mapping[str, object]],
    source_manifest: Mapping[str, str],
    format_plan,
    roles: Mapping[str, str],
    unit_classes: Mapping[str, str],
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
    for qname, planned in sorted(format_plan.units.items()):
        row = stats[qname]
        shape = (
            int(row["out_features"]), int(row["in_features"])
        )
        if math.prod(shape) != int(row["n_params"]):
            raise DSv4CampaignError(f"{qname}: probe shape/n_params differ")
        source_kind = str(source_manifest[qname])
        terminal = _TERMINAL_BY_SOURCE_KIND.get(source_kind)
        if terminal is None:
            raise DSv4CampaignError(
                f"{qname}: unsupported exact source kind {source_kind!r}"
            )
        formats = (*NVFP4_FORMATS, *format_plan.formats_for(qname))
        payloads: dict[str, int] = {}
        for format_name in formats:
            spec = fr.get_format(format_name)
            verdict = _source_bpp_applicability(
                shape,
                spec,
                qname=qname,
                source_kind=source_kind,
                cb_serialization_context=cb_context,
            )
            if not verdict.legal:
                raise DSv4CampaignError(
                    f"{qname}/{format_name}: source-rate gate refused "
                    f"{verdict.reason}: {verdict.detail}"
                )
            payloads[spec.name] = serialized_candidate_payload(
                spec,
                shape,
                qname=qname,
                cb_serialization_context=cb_context,
            )[0]
        payloads[terminal] = int(planned.source_payload_bytes)
        declarations.append(CBUnitDeclaration(
            qname=qname,
            role=roles[qname],
            unit_class=unit_classes[qname],
            n_params=math.prod(shape),
            payload_bytes_by_format=payloads,
            terminal_format=terminal,
            serving_group=serving_group.get(qname),
        ))
    return tuple(declarations)


def _assert_authoritative_dsv4_basis_map(
    source_map: Mapping[str, str],
) -> None:
    """Require value-bearing cells without letting their map choose source."""

    required = {*NVFP4_FORMATS, *FP8_NONEXPERT_FORMATS}
    missing = sorted(required - set(source_map))
    if missing:
        raise DSv4CampaignError(
            "bundle/context map lacks source-in-name campaign formats: "
            f"missing={missing[:8]}. Map values do not choose learned versus "
            "lattice; FP8_CBL and FP8_CB names do."
        )


def assert_dsv4_anchor_accounting(
    units: Sequence[UnitSpec], plugin: CodebookAnchoredFormatPlugin,
) -> dict[str, int]:
    requests = plan_anchor_requests(units, plugin)
    counts: Counter[str] = Counter(
        f"{request.segment.family}|{request.segment.equivalence_class}"
        for request in requests
    )
    expected = {
        "nvfp4_cb|lattice": 33_325,
        "fp8_cb|learned": 33_325,
        "fp8_cb|lattice": 33_325,
    }
    by_segment: Counter[str] = Counter(
        request.segment.stamp for request in requests
    )
    expected_by_segment = {
        f"{family}|{role}|{basis}": unit_count
        for family, basis, role_counts in (
            ("nvfp4_cb", LATTICE_BASIS, _DSV4_ROLE_UNIT_COUNTS),
            ("fp8_cb", LEARNED_BASIS, _DSV4_ROLE_UNIT_COUNTS),
            ("fp8_cb", LATTICE_BASIS, _DSV4_ROLE_UNIT_COUNTS),
        )
        for role, unit_count in role_counts.items()
    }
    if (
        len(units) != DSV4_TOTAL_UNITS
        or dict(counts) != expected
        or dict(by_segment) != expected_by_segment
    ):
        raise DSv4CampaignError(
            f"DSv4 anchor census differs: units={len(units)}, "
            f"family/equivalence={dict(counts)}, expected={expected}; "
            f"segments={dict(by_segment)}, "
            f"expected_segments={expected_by_segment}"
        )
    if len(requests) != DSV4_EXPECTED_ANCHORS:
        raise DSv4CampaignError(
            f"DSv4 anchor total is not {DSV4_EXPECTED_ANCHORS:,}"
        )
    return dict(counts)


def prepare_dsv4_campaign(args: argparse.Namespace) -> PreparedDSv4Campaign:
    """Complete every CPU identity/legality check before the P0 GPU pass."""
    work_dir = _safe_work_dir(args.work_dir)
    stats, probe_meta = _load_probe(args)
    from prismaquant.model_profiles import detect_profile
    from prismaquant.allocator_candidates import _scan_source_dtype_manifest
    from prismaquant.source_class_format_plan import (
        build_source_class_format_plan,
        write_format_plan,
    )
    from prismaquant.nvfp4_cb_footprint import (
        cb_serialization_context_from_env,
        cb_serialization_context_stamp,
    )

    profile = detect_profile(args.model)
    roles, unit_classes = _role_and_class_maps(profile, tuple(stats))
    from prismaquant.measure_quant_cost import ActivationIndex

    activation_index = ActivationIndex(
        Path(args.activation_cache_dir), stats
    )
    missing_activations = sorted(
        qname for qname in stats if qname not in activation_index
    )
    cold_expert_provenance = _validated_cold_expert_provenance(
        stats=stats,
        unit_classes=unit_classes,
        missing_activations=missing_activations,
        col_weights_path=args.col_weights,
    )
    source_census = _scan_source_dtype_manifest(args.model, profile)
    missing_source = sorted(set(stats) - set(source_census))
    if missing_source:
        raise DSv4CampaignError(
            f"source census misses planned units: {missing_source[:8]}"
        )
    source_manifest = {qname: source_census[qname] for qname in stats}
    cb_context = cb_serialization_context_from_env(
        require_explicit=True, where="DSv4 anchored AURA campaign"
    )
    source_map = cb_context.codebook_source_by_format
    if not isinstance(source_map, Mapping):
        raise DSv4CampaignError(
            "loaded bundle/context has no authoritative source map"
        )
    _assert_authoritative_dsv4_basis_map(source_map)
    routed_sha = _validate_routed_selection(args.routed_book_selection)
    bundle_path = getattr(cb_context, "codebook_bundle_path", None)
    if not bundle_path:
        raise DSv4CampaignError(
            "validated CB context has no value-bearing bundle path"
        )
    routed_bundle_binding = _validate_routed_bundle_selection_identity(
        bundle_path, routed_sha,
    )
    format_plan = build_source_class_format_plan(
        stats,
        source_manifest,
        profile,
        expert_formats=args.expert_formats,
        nonexpert_formats=args.nonexpert_formats,
        cb_serialization_context=cb_context,
    )
    if tuple(format_plan.menus["expert"]) != FP8_EXPERT_FORMATS or tuple(
        format_plan.menus["nonexpert"]
    ) != FP8_NONEXPERT_FORMATS:
        raise DSv4CampaignError(
            "frozen CLI menus differ from the DSv4 source-in-name FP8 "
            "contract"
        )
    format_plan_path = work_dir / "checkpoints" / "source_format_plan.json"
    format_plan_path.parent.mkdir(parents=True, exist_ok=True)
    write_format_plan(format_plan, format_plan_path)

    declarations = _build_declarations(
        stats=stats,
        source_manifest=source_manifest,
        format_plan=format_plan,
        roles=roles,
        unit_classes=unit_classes,
        cb_context=cb_context,
    )
    all_cb_formats = (*NVFP4_FORMATS, *FP8_NONEXPERT_FORMATS)
    render_levers = {
        "gptq": True,
        "static_act_order": True,
        "joint_scale_opt": True,
        "weighted_vq": True,
    }
    arm_identity = {
        "cb_serialization_context": cb_serialization_context_stamp(
            cb_context, formats=all_cb_formats
        ),
        "render_levers": render_levers,
        "routed_book_selection_sha256": routed_sha,
        "routed_bundle_selection_binding": routed_bundle_binding,
        "production_arm_only": True,
        "rtn_anchor_allowed": False,
        "cold_expert_provenance": cold_expert_provenance,
    }
    plugin = CodebookAnchoredFormatPlugin(
        codebook_source_by_format=source_map,
        arm_identity=arm_identity,
        anchor_formats=DSV4_ANCHOR_FORMATS,
    )
    units = build_cb_units(declarations, plugin)
    anchor_requests = plan_anchor_requests(units, plugin)
    anchor_counts = assert_dsv4_anchor_accounting(units, plugin)
    panel, validation, panel_report = plan_cb_panel_and_validation(
        units, plugin, DSV4_PANEL_POLICY
    )
    formats, purposes, union_report = build_streamed_cb_render_plan(
        units, plugin, anchor_requests, panel, validation
    )
    plan_report = {
        **dict(union_report),
        "anchor_renders_by_family_basis": anchor_counts,
        "anchor_renders_by_segment": dict(sorted(Counter(
            request.segment.stamp for request in anchor_requests
        ).items())),
        "anchor_renders": len(anchor_requests),
        "panel_renders": len(panel),
        "validation_renders": len(validation),
        "format_plan_identity_sha256": format_plan.identity_sha256,
        "format_plan_path": str(format_plan_path),
        "cold_expert_render_units": len(missing_activations),
    }
    return PreparedDSv4Campaign(
        args=args,
        profile=profile,
        probe_stats=stats,
        probe_meta=probe_meta,
        cb_context=cb_context,
        plugin=plugin,
        format_plan=format_plan,
        units=units,
        anchor_requests=anchor_requests,
        panel_requests=panel,
        validation_requests=validation,
        formats_by_qname=formats,
        purposes_by_qname=purposes,
        plan_report=plan_report,
        panel_report=panel_report,
        routed_selection_sha256=routed_sha,
        arm_identity=arm_identity,
        cold_expert_provenance=cold_expert_provenance,
    )


def _assert_home_reserve() -> None:
    usage = shutil.disk_usage("/home/rob")
    if usage.free < math.ceil(0.10 * usage.total):
        raise DSv4CampaignError(
            f"/home/rob free space is {usage.free / usage.total:.2%}; "
            "campaign requires >=10%"
        )


def _measure_streamed(prepared: PreparedDSv4Campaign) -> dict[str, object]:
    """P0+bounded renders: one streamed adjoint, one live layer at a time."""
    args = prepared.args
    _assert_home_reserve()
    from prismaquant.gpu_guard import require_cuda_hot_path
    device = require_cuda_hot_path("dsv4_aura_cb_reprice", "cuda")
    import torch
    from transformers import AutoTokenizer
    from prismaquant.build_production_cache import _load_col_weights
    from prismaquant.cost_streaming import (
        build_streamed_causal_lm,
        build_streamed_model_identity,
    )
    from prismaquant.measure_quant_cost import ActivationIndex
    from prismaquant.perturbed_x_cache import calibration_data_hash
    from prismaquant.sensitivity_probe import load_calibration

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=Path(args.model).exists(),
    )
    calibration = load_calibration(
        tokenizer,
        args.dataset,
        args.n_calib_samples,
        args.calib_seqlen,
        calib_seed=args.calib_seed,
    ).to(device)
    observed_calibration_hash = calibration_data_hash(calibration)
    if observed_calibration_hash != prepared.probe_meta["calib_hash"]:
        raise DSv4CampaignError(
            "tokenized calibration hash differs from probe identity"
        )
    col_weights = _load_col_weights(
        args.col_weights,
        (*NVFP4_FORMATS, *FP8_NONEXPERT_FORMATS),
    )
    if not isinstance(col_weights, Mapping):
        raise DSv4CampaignError("CB render imatrix did not load")
    missing_col = sorted(set(prepared.probe_stats) - set(col_weights))
    if missing_col:
        raise DSv4CampaignError(
            f"CB render imatrix misses units: {missing_col[:8]}"
        )
    activation_index = ActivationIndex(
        Path(args.activation_cache_dir), prepared.probe_stats
    )
    checkpoint_root = Path(args.checkpoint_dir)
    runner = build_streamed_causal_lm(
        args.model,
        device=device,
        dtype=torch.bfloat16,
        offload_folder=str(checkpoint_root / "streamed-model-offload"),
        profile=prepared.profile,
    )
    try:
        model_identity = build_streamed_model_identity(
            runner,
            args.model,
            identity_cache_path=checkpoint_root / "streamed_model_identity.json",
        )
        payload = run_streamed_cb_anchor_aura(
            runner,
            calibration,
            formats_by_qname=prepared.formats_by_qname,
            legal_formats_by_qname={
                unit.qname: tuple(
                    candidate.format_name
                    for candidate in unit.candidates
                    if not candidate.terminal
                )
                for unit in prepared.units
            },
            purposes_by_qname=prepared.purposes_by_qname,
            activation_index=activation_index,
            render_levers=prepared.arm_identity["render_levers"],
            col_weights=col_weights,
            cb_serialization_context=prepared.cb_context,
            calibration_hash=observed_calibration_hash,
            arm_identity=prepared.arm_identity,
            model_identity=model_identity,
            checkpoint_dir=checkpoint_root / "aura",
            resume=bool(args.resume),
            n_probes=int(args.n_probes),
            profile=prepared.profile,
            cold_expert_provenance=prepared.cold_expert_provenance,
            checkpoint_identity_extra={
                "campaign_schema": DSV4_CAMPAIGN_SCHEMA,
                "source_format_plan_identity_sha256": (
                    prepared.format_plan.identity_sha256
                ),
                "routed_book_selection_sha256": (
                    prepared.routed_selection_sha256
                ),
                "segment_key_fields": [
                    "family", "role", "equivalence_class"
                ],
                "equivalence_vocabulary_name": "codebook_basis",
            },
        )
    finally:
        runner.shutdown()
    return payload


def _allocator_command(
    prepared: PreparedDSv4Campaign,
    *,
    cost_path: Path,
    output_dir: Path,
) -> list[str]:
    args = prepared.args
    bundle = getattr(prepared.cb_context, "codebook_bundle_path", None)
    if not bundle:
        raise DSv4CampaignError("validated CB context has no bundle path")
    formats = (
        *NVFP4_FORMATS,
        *FP8_NONEXPERT_FORMATS,
        "MXFP4_SOURCE", "FP8_BLOCK_UE8M0_SOURCE", "BF16",
    )
    return [
        sys.executable, "-m", "prismaquant.allocator",
        "--probe", str(args.probe),
        "--costs", str(cost_path),
        "--model-override", str(args.model),
        "--target-profile", "nvfp4_cb",
        "--target-disk-gb", f"{DSV4_BUDGET_BYTES / 1e9:.9f}",
        "--artifact-overhead-reserve-bytes",
        str(DSV4_ARTIFACT_RESERVE_BYTES),
        "--cb-scale-coding", "two_tier",
        "--cb-codebook-source", "learned",
        "--cb-codebook-source-scope", "fp8",
        "--cb-codebook-bundle", str(bundle),
        "--cb-scale-sweep", "1",
        "--cb-scale-sweep-scope", "all",
        "--cb-ldlq", "0", "--cb-ldlq-scope", "none",
        "--cb-encode-tier", "balanced",
        "--cb-col-weights", str(args.col_weights),
        "--formats", ",".join(formats),
        "--mtp-format", "BF16", "--visual-format", "BF16",
        "--threads", "16",
        "--layer-config", str(output_dir / "layer_config.json"),
        "--pareto-csv", str(output_dir / "pareto.csv"),
        "--pareto-output-dir", str(output_dir / "pareto-points"),
        "--applicability-report", str(output_dir / "format_applicability.json"),
        "--bit-attribution-json", str(output_dir / "bit_attribution.json"),
        "--bit-attribution-csv", str(output_dir / "bit_attribution.csv"),
    ]


def _recipe_format(value: object) -> str:
    if isinstance(value, str):
        return fr.canonical_format_name(value)
    if not isinstance(value, Mapping):
        raise DSv4CampaignError("allocator recipe cell is not an object")
    serialized = value.get("cb_serialized_identity")
    if isinstance(serialized, str):
        try:
            format_name = json.loads(serialized).get("format")
            if format_name:
                return fr.canonical_format_name(str(format_name))
        except Exception:
            pass
    data_type = str(value.get("data_type", "")).lower()
    if data_type in {"nvfp4_cb", "fp8_cb", "fp8_cbl"}:
        return f"{data_type.upper()}_K{int(value['cb_k'])}"
    try:
        # Delegate native/source recipe interpretation to the same shared
        # parser export consumes.  In particular, DSv4's source terminal is
        # emitted as data_type=fp8_e4m3 + group_size=128 + scale_fmt=ue8m0,
        # not as the registry name FP8_BLOCK_UE8M0_SOURCE.
        from prismaquant.layer_config import canonicalize_format

        return fr.canonical_format_name(canonicalize_format(value))
    except (KeyError, TypeError, ValueError) as exc:
        raise DSv4CampaignError(
            "cannot recover selected format from allocator recipe "
            f"data_type={data_type!r}"
        ) from exc


def _selected_assignment(
    layer_config_path: Path, units: Sequence[UnitSpec],
) -> dict[str, str]:
    payload = json.loads(layer_config_path.read_text())
    assignment = {
        qname: _recipe_format(payload[qname]) for qname in sorted(
            unit.qname for unit in units
        )
    }
    for unit in units:
        legal = {candidate.format_name for candidate in unit.candidates}
        if assignment[unit.qname] not in legal:
            raise DSv4CampaignError(
                f"allocator selected source-illegal {unit.qname}@"
                f"{assignment[unit.qname]}"
            )
    return assignment


def render_economics_report(
    prepared: PreparedDSv4Campaign,
) -> dict[str, object]:
    """Project the bounded run from measured, parameter-scaled timings.

    The render projection charges every physical ``(qname, format)`` once,
    even when an anchor also belongs to the panel.  Tensor sizes are converted
    to 2048x4096 expert equivalents from the exact probe ``n_params``; counting
    every dense tensor as one expert would materially understate this run.
    """
    reference_nparams = 2048 * 4096
    # Current compiled one-expert measurements and the user-requested
    # conservative E8 pre-optimization baseline.  The latter is also the only
    # supplied same-shape low-rung reference, so NV K12/K15/K16/K18 use it and
    # are labelled as unmatched rather than pretending it is an NV timing.
    profile_seconds = {
        "NVFP4_CB_K12": 0.069821,
        "NVFP4_CB_K15": 0.069821,
        "NVFP4_CB_K16": 0.069821,
        "NVFP4_CB_K18": 0.069821,
        "FP8_CBL_K28": 0.075109,
        # K32 CBL and lattice anchors use the measured K33 result as a
        # conservative adjacent-rung proxy; source stays encoded by each name.
        "FP8_CBL_K32": 0.144363,
        "FP8_CB_K32": 0.144363,
        # Medians of elapsed_encode_seconds in the older 16-expert nested
        # production-shape pilot.  They are real rung timings, but predate the
        # current compiled arm. K41 and K46 conservatively proxy the adjacent
        # legal learned rungs K40 and K44 respectively.
        "FP8_CBL_K40": 1.3973947751555897,
        "FP8_CBL_K44": 3.335986553436669,
        "FP8_CB_K47": 3.8868322437820098,
        "FP8_CB_K48": 4.442083168687532,
    }
    requests_by_cell: dict[tuple[str, str], list[RenderRequest]] = {}
    for request in (
        *prepared.anchor_requests,
        *prepared.panel_requests,
        *prepared.validation_requests,
    ):
        requests_by_cell.setdefault(
            (request.qname, request.format_name), []
        ).append(request)
    physical_pairs = set(requests_by_cell)
    planned_pairs = {
        (qname, format_name)
        for qname, rows in prepared.purposes_by_qname.items()
        for format_name in rows
    }
    if physical_pairs != planned_pairs:
        raise DSv4CampaignError(
            "economics request union differs from streamed render plan"
        )
    missing_timings = sorted({
        format_name for _qname, format_name in physical_pairs
        if format_name not in profile_seconds
    })
    if missing_timings:
        raise DSv4CampaignError(
            f"no measured timing projection for {missing_timings}"
        )

    priority = {"anchor": 0, "panel": 1, "validation": 2}
    encode_seconds_by_charge: Counter[str] = Counter()
    render_count_by_charge: Counter[str] = Counter()
    reference_equivalents_by_charge: Counter[str] = Counter()
    encode_seconds_by_segment: Counter[str] = Counter()
    for (qname, format_name), requests in sorted(requests_by_cell.items()):
        try:
            n_params = int(prepared.probe_stats[qname]["n_params"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DSv4CampaignError(
                f"{qname}: economics lacks exact probe n_params"
            ) from exc
        equivalents = n_params / reference_nparams
        charged = min(requests, key=lambda item: priority[item.purpose])
        seconds = equivalents * profile_seconds[format_name]
        encode_seconds_by_charge[charged.purpose] += seconds
        render_count_by_charge[charged.purpose] += 1
        reference_equivalents_by_charge[charged.purpose] += equivalents
        encode_seconds_by_segment[charged.segment.stamp] += seconds

    encode_seconds = math.fsum(encode_seconds_by_charge.values())
    # Measured DSv4 incremental-probe phase timings.  The central projection
    # scales head and layer backward compute by K, while charging the measured
    # non-backward reverse overhead once because the anchored implementation
    # loads each layer once and runs all K probes inside that window.
    p0_measured = {
        "phase1_forward_seconds": 129.0,
        "phase2_head_backward_seconds_per_probe": 1.8,
        "phase3_reverse_sweep_seconds": 259.6,
        "phase3_backward_compute_seconds_per_probe": 107.8,
        "phase3_load_seconds_once": 10.0,
    }
    probes = int(prepared.args.n_probes)
    p0_non_backward_once = (
        p0_measured["phase3_reverse_sweep_seconds"]
        - p0_measured["phase3_backward_compute_seconds_per_probe"]
    )
    p0_seconds = (
        p0_measured["phase1_forward_seconds"]
        + probes * p0_measured["phase2_head_backward_seconds_per_probe"]
        + probes * p0_measured[
            "phase3_backward_compute_seconds_per_probe"
        ]
        + p0_non_backward_once
    )
    p0_lower_seconds = (
        p0_measured["phase1_forward_seconds"]
        + p0_measured["phase3_load_seconds_once"]
        + probes * (
            p0_measured["phase2_head_backward_seconds_per_probe"]
            + p0_measured["phase3_backward_compute_seconds_per_probe"]
        )
    )
    p0_upper_seconds = (
        p0_measured["phase1_forward_seconds"]
        + probes * (
            p0_measured["phase2_head_backward_seconds_per_probe"]
            + p0_measured["phase3_reverse_sweep_seconds"]
        )
    )
    block = int(os.statvfs(Path(prepared.args.work_dir)).f_frsize)
    candidate_cells = sum(len(unit.candidates) for unit in prepared.units)
    # Deliberately conservative: one filesystem allocation block for every
    # scalar receipt, every full cost cell, and every source-plan unit, plus
    # the one required export copy of the existing imatrix. Rendered weights
    # contribute exactly zero persistent bytes.
    projected_new_disk = (
        (len(physical_pairs) + candidate_cells + len(prepared.units)) * block
        + Path(prepared.args.col_weights).stat().st_size
    )
    return {
        "reference_tensor_nparams": reference_nparams,
        "measured_encode_seconds_per_reference_tensor_by_format": (
            profile_seconds
        ),
        "timing_sources": {
            "current_low_rungs": (
                "docs/results/cb_encode_cuda_profile_2026-08-11.md: "
                "CBL K28=75.109ms; measured K33=144.363ms is the "
                "conservative CBL/CB K32 proxy; conservative E8 "
                "K28=69.821ms/expert"
            ),
            "older_high_rungs": (
                "/home/rob/dq-runs/dsv4-flash-0731/nested-pilot/"
                "raw_records.jsonl: medians of elapsed_encode_seconds for "
                "K41/K46/K47/K48; K41/K46 conservatively proxy CBL "
                "K40/K44"
            ),
            "p0_proxy": (
                "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p7/"
                "logs/probe.log: phase1=129.0s, phase2=1.8s, "
                "phase3=259.6s (107.8s backward, 10.0s load)"
            ),
        },
        "physical_union_render_cells": len(physical_pairs),
        "physical_render_cells_charged_by_purpose": dict(
            render_count_by_charge
        ),
        "reference_tensor_equivalents_by_purpose": dict(
            reference_equivalents_by_charge
        ),
        "encode_projection_seconds_by_purpose": dict(
            encode_seconds_by_charge
        ),
        "encode_projection_seconds_by_segment": dict(sorted(
            encode_seconds_by_segment.items()
        )),
        "encode_projection_seconds": encode_seconds,
        "encode_projection_gpu_hours": encode_seconds / 3600.0,
        "p0_projection_inputs": p0_measured,
        "p0_projection_n_probes": probes,
        "p0_projection_seconds": p0_seconds,
        "p0_projection_gpu_hours": p0_seconds / 3600.0,
        "p0_projection_lower_seconds": p0_lower_seconds,
        "p0_projection_upper_seconds": p0_upper_seconds,
        "total_projected_gpu_hours": (
            encode_seconds + p0_seconds
        ) / 3600.0,
        "total_projected_gpu_hours_lower": (
            encode_seconds + p0_lower_seconds
        ) / 3600.0,
        "total_projected_gpu_hours_upper": (
            encode_seconds + p0_upper_seconds
        ) / 3600.0,
        "projection_limitation": (
            "K12/K15/K16/K18 use the measured 69.821ms/E production-shape "
            "low-rung proxy; CBL/CB K32 uses the measured K33 result; CBL "
            "K40/K44 use older K41/K46 pilot medians; lattice K47/K48 use "
            "their older measured pilot medians. P0 is a measured-phase "
            "scaling proxy, not a timing of this new fused pass. Dot-product, "
            "checkpoint-fsync, allocator, and export-copy wall time are not "
            "separately measured."
        ),
        "projected_peak_new_disk_bytes": projected_new_disk,
        "projected_peak_new_disk_assumptions": (
            "one filesystem allocation block per scalar receipt, legal cost "
            "cell, and source-plan unit, plus one imatrix copy; variable "
            "pickle/JSON payload sizes and allocator Pareto artifacts are not "
            "separately bounded"
        ),
        "disk_projection_block_bytes": block,
        "persistent_rendered_weight_bytes": 0,
        "preexisting_activation_cache_reused_not_counted": str(
            prepared.args.activation_cache_dir
        ),
        "anchor_renders": len(prepared.anchor_requests),
        "anchor_renders_by_family_basis": prepared.plan_report[
            "anchor_renders_by_family_basis"
        ],
        "anchor_renders_by_segment": prepared.plan_report[
            "anchor_renders_by_segment"
        ],
        "panel_renders": len(prepared.panel_requests),
        "validation_renders": len(prepared.validation_requests),
    }


def run_dsv4_anchor_campaign(
    args: argparse.Namespace, *, control_plane: str,
) -> int:
    """CPU validation -> one streamed pass -> fit/price -> one DP -> artifact."""
    if control_plane != __name__:
        raise DSv4CampaignError(
            f"unexpected DSv4 control plane {control_plane!r}"
        )
    prepared = prepare_dsv4_campaign(args)
    # Satisfied since 2026-08-11 (anchored-AURA admission landed in
    # allocator_candidates); kept as a fail-closed re-check that fires before
    # CUDA/P0, so a regression or a downgraded checkout refuses at the gate
    # rather than mid-campaign.  The remaining launch blockers are operator
    # inputs (WORK_DIR, codebook bundle, routed-book selection, receipt).
    require_allocator_supersurrogate_support()

    work_dir = Path(args.work_dir)
    artifacts_dir = work_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    streamed_payload = _measure_streamed(prepared)
    raw_path = artifacts_dir / "streamed_anchor_aura.pkl"
    atomic_write_bytes(
        raw_path,
        pickle.dumps(streamed_payload, protocol=pickle.HIGHEST_PROTOCOL),
    )
    anchors = anchors_from_streamed_payload(
        prepared.anchor_requests, streamed_payload
    )
    panel_observations = observations_from_streamed_payload(
        prepared.panel_requests, streamed_payload
    )
    fits = fit_all_cb_segments(
        panel_observations, prepared.units, prepared.plugin
    )
    validation_observations = observations_from_streamed_payload(
        prepared.validation_requests, streamed_payload
    )
    validation = heldout_validation_report(
        validation_observations, anchors, fits
    )
    hulls = fitted_cb_hull_report(prepared.units, prepared.plugin, fits)
    cells = price_anchored_candidates(
        prepared.units, prepared.plugin, anchors, fits
    )
    cost_payload = build_cb_allocator_cost_payload(
        cells,
        streamed_payload=streamed_payload,
        fits=fits,
        hull_report=hulls,
        validation_report=validation,
    )
    cost_path = write_cb_cost_payload(
        artifacts_dir / "cost_aura_anchored.pkl", cost_payload
    )
    allocator_output = work_dir / "allocator-aura"
    command = _allocator_command(
        prepared, cost_path=cost_path, output_dir=allocator_output
    )
    run_allocator_once(
        command=command,
        output_dir=allocator_output,
        resume=bool(args.resume),
        environment_updates={"PRISMAQUANT_ACTIVATION_FAIR_PRICING": "0"},
        invocation_provenance={
            "cost_currency": AURA_CURRENCY,
            "activation_fair_pricing": False,
            "budget_bytes": DSV4_BUDGET_BYTES,
            "cost_payload_sha256": _sha256(cost_path),
            "format_plan_identity_sha256": (
                prepared.format_plan.identity_sha256
            ),
            "production_arm_identity_sha256": canonical_json_sha256(
                prepared.arm_identity,
                where="DSv4 allocator production arm identity",
            ),
        },
    )
    assignment = _selected_assignment(
        allocator_output / "layer_config.json", prepared.units
    )
    exposure = extrapolation_distance_report(
        prepared.units, prepared.plugin, anchors, assignment
    )
    economics = render_economics_report(prepared)
    report = {
        "schema": DSV4_CAMPAIGN_SCHEMA,
        "cost_currency": AURA_CURRENCY,
        "budget_bytes": DSV4_BUDGET_BYTES,
        "plan": dict(prepared.plan_report),
        "panel": dict(prepared.panel_report),
        "validation": validation,
        "currency_invariance": {
            segment.stamp: fit.aura_vs_weight_diagnostic
            for segment, fit in sorted(fits.items())
        },
        "lower_convex_hull": hulls,
        "extrapolation_distance": exposure,
        "economics": economics,
        "route_flip_limitation": ROUTE_FLIP_LIMITATION,
        "served_metric_is_only_gate": True,
    }
    json.dumps(report, allow_nan=False)
    atomic_write_bytes(
        artifacts_dir / "campaign_report.json",
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode(),
    )
    write_exportable_artifacts(
        artifacts_dir / "exportable-aura",
        allocator_output_dir=allocator_output,
        cb_col_weights_path=args.col_weights,
        provenance=report,
        resume=bool(args.resume),
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded, anchored DSv4 AURA/CB one-shot campaign"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--activation-cache-dir", required=True)
    parser.add_argument("--col-weights", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--expert-formats", required=True)
    parser.add_argument("--nonexpert-formats", required=True)
    parser.add_argument("--routed-book-selection", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--n-calib-samples", type=int, default=16)
    parser.add_argument("--calib-seqlen", type=int, default=512)
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--n-probes", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cost-mode", choices=("aura",), required=True)
    parser.add_argument("--require-production-cache", action="store_true")
    parser.add_argument("--format-plan")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.require_production_cache:
        parser.error("DSv4 anchors require the production render arm")
    if args.format_plan:
        parser.error(
            "DSv4 format plan is derived from the exact source census; "
            "external replacement is not accepted by the frozen launcher"
        )
    if Path(args.work_dir).resolve(strict=False) == Path("/tmp") or Path(
        "/tmp"
    ) in Path(args.work_dir).resolve(strict=False).parents:
        parser.error("campaign work-dir may not be under /tmp")
    return run_dsv4_anchor_campaign(args, control_plane=__name__)


if __name__ == "__main__":  # pragma: no cover - frozen shell entrypoint
    raise SystemExit(main())


__all__ = [
    "DSV4_BOUNDED_CAMPAIGN_WORKER",
    "DSV4_BUDGET_BYTES",
    "DSV4_EXPECTED_ANCHORS",
    "DSV4_EXPERT_UNITS",
    "DSV4_NONEXPERT_UNITS",
    "DSV4_PANEL_POLICY",
    "DSV4_TOTAL_UNITS",
    "DSv4CampaignError",
    "FP8_EXPERT_FORMATS",
    "FP8_NONEXPERT_FORMATS",
    "NVFP4_FORMATS",
    "PreparedDSv4Campaign",
    "assert_dsv4_anchor_accounting",
    "main",
    "prepare_dsv4_campaign",
    "render_economics_report",
    "run_dsv4_anchor_campaign",
]
