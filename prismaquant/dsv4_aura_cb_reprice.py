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
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import sys

from prismaquant import format_registry as fr
from prismaquant.gridbook_runtime_pin import GRIDBOOK_RUNTIME_RELEASE_VERSION
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
    cb_rung,
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
from prismaquant.dsv4_campaign_completion import (
    CampaignCompletionError,
    assert_replay_matches_completion_receipt,
    completion_receipt_path,
    historical_checkpoint_root,
    verify_receipt_for_replay,
)


DSV4_BOUNDED_CAMPAIGN_WORKER = True
DSV4_CAMPAIGN_SCHEMA = "prismaquant.dsv4_cb_anchored_aura.v1"
DSV4_CPU_REPLAY_SCHEMA = "prismaquant.dsv4_cb_anchored_aura.cpu_replay.v1"
DSV4_W8A16_READMISSION_SCHEMA = (
    "prismaquant.dsv4_cb_anchored_aura.w8a16_readmission.v1"
)
DSV4_TOTAL_UNITS = 33_325
DSV4_EXPERT_UNITS = 43 * 256 * 3
DSV4_NONEXPERT_UNITS = 301
DSV4_EXPECTED_ANCHORS = 66_951
# Two budgets that happened to be equal, and must not be spelled the same.
#
# The frozen one is part of an approval RECORD: DSV4_W8A16_APPROVED_SELECTION
# below is attested by sha256, and dsv4_w8a16_export_handoff re-checks the
# stamp's budget_bytes against it. It describes the 112.69 GB artifact that was
# approved and can never change.
#
# The default one is this run's CONSTRAINT, overridable with --budget-bytes.
# They were a single name until 2026-08-16, which meant retargeting the
# campaign to a smaller artifact would have silently rewritten the W8A16
# approval it has nothing to do with.
DSV4_W8A16_APPROVED_BUDGET_BYTES = 112_690_000_000
DSV4_DEFAULT_BUDGET_BYTES = 112_690_000_000
# --target-disk-gb is a whole-artifact hard budget over all regular files
# recursive, so the selector needs an operator reserve for safetensors headers,
# JSON and copied tokenizer files. 256 MiB is what the 112.69 GB artifact ran
# with; it actually consumed 7.1 MB, so the margin is deliberate and cheap.
DSV4_ARTIFACT_RESERVE_BYTES = 268_435_456
# Spark has one unified CPU/GPU memory pool.  The worst DSv4 anchor layer's
# exact FP32 dW plane is 51.3 GiB, so source weights must never retain or
# speculatively load a second decoder layer during that reverse window.
DSV4_STREAMING_CACHE_MAX_SLOTS = 1
# One profile id for the whole campaign: it bounds the source-class plan's
# family to the rungs the pinned runtime's fused mid-M lane instantiates, and
# it is the allocator's --target-profile.  Splitting these would let the
# campaign price a menu the artifact is not allowed to ship.
DSV4_TARGET_PROFILE = "nvfp4_cb"
_FULL_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_FULL_SHA256 = re.compile(r"[0-9a-f]{64}")

# The only historical measurement producer this one-purpose semantic migration
# may consume. Generic replay remains same-snapshot-only.
DSV4_W8A16_LEGACY_PRODUCER = {
    "commit": "58fb335fdb8e4fde35312d40564062c5a44f3457",
    "tree": "55b67ebe49b6df0d8a2d8a57b52dcda7b27fd79e",
    "closure_sha256": (
        "a4f10f5c6a8411f4194eec03ee747f81782762644735c507e0c6a615fda11816"
    ),
}
DSV4_W8A16_LEGACY_RECEIPT_SHA256 = (
    "835a814ccae84bb8082f29e19f484f6c26494bee97c4cfc6c9880ea24f0dfe75"
)
DSV4_W8A16_LEGACY_INVOCATION_ID = "e3f41a3b3ff44dc4b3a8960b8f0cc796"
DSV4_W8A16_LEGACY_FORMAT_PLAN_SHA256 = (
    "f0f4cfa233a6190aea49a706a567d22909719c54366b4437330e13d41286a644"
)
DSV4_W8A16_APPROVED_ASSIGNMENT_SHA256 = (
    "db6d9bc78807efa973ca0a49666c0aa8c4ef198a3e891a4b77f16840af1f5fef"
)
DSV4_W8A16_APPROVED_LAYER_CONFIG_SHA256 = (
    "1b3a3c2ea0e44e42e4e7642730672c0604b300d45c88e8b511d5edf3eb57ef32"
)
DSV4_W8A16_APPROVED_SELECTION_SHA256 = (
    "d38e41133e567bb38e99983bae9ecb2696c53ed56bccc1887b9c24ca83b8fdb8"
)
DSV4_W8A16_APPROVED_CB_COL_WEIGHTS_SHA256 = (
    "df045bde786f7d092e501bfa856984243106a13f05594f4a11fe30270fb09379"
)
DSV4_W8A16_APPROVED_SELECTION = {
    "budget_bytes": DSV4_W8A16_APPROVED_BUDGET_BYTES,
    "chosen_achieved_bits": 2.7555507482797204,
    "predicted_dloss": 0.35597906499701,
    "selection_tensor_payload_bytes": 112_349_756_664,
    "selection_whole_artifact_upper_bound_bytes": 112_618_192_120,
}

# The FP8-CB menus are the on-law rungs only.  gridbook K1.2's fused mid-M
# kernel law admits FP8-CB at k % 4 == 0 and nothing else: type_size = 4k is
# the packed-B TMA box's contiguous extent and must stay a 16-byte multiple,
# and the fused mainloop's single CbSubW = k/4 sub-table width is the format's
# real layout only on those rungs -- a uniform decode at, say, k37 would be
# *wrong*, not merely unaligned.  serving_profile_specs/nvfp4_cb.json backs
# [28, 32, 36, 40, 44, 48] for every runtime 0.5.0..0.9.1 (the pin is 0.9.1,
# version_is_release, and its packaged contract explicitly attests the routed
# per-role LUT ABI -- abi_features is byte-equal to 0.8.11's), so this is the
# served set, not an aspiration.  READ THIS BEFORE A RE-EXPORT: 0.9.1 also
# publishes ``formats[FP8_CB_K].producer_rungs = [40, 44, 48]``, a narrower
# ladder than the reader surface above.  The two are different questions --
# what the runtime DECODES versus what gridbook attests a producer may EMIT --
# and gridbook_format_contract.py is deliberately still bound to v4/v11, so
# nothing enforces the narrowing today.  The K28/K32 expert rungs below are on
# the decode law and were built that way; whether they stay producible is the
# open decision that reader's docstring records. NVFP4-CB
# is outside that law -- its lane backs no fused mid-M rungs at any version --
# so K12..K18 stays contiguous.
NVFP4_FORMATS = tuple(f"NVFP4_CB_K{k}" for k in range(12, 19))
# Routed experts are additionally capped by the byte-exact source-payload
# ceiling at K33, so on-law + legal leaves exactly two rungs.
FP8_EXPERT_FORMATS = ("FP8_CB_K28", "FP8_CB_K32")
FP8_NONEXPERT_FORMATS = (
    "FP8_CB_K28", "FP8_CB_K32", "FP8_CB_K36",
    "FP8_CB_K40", "FP8_CB_K44", "FP8_CB_K48",
)
# Derived so the routed-book coverage contracts and the basis map can never
# drift from the menus above.
FP8_EXPERT_RUNGS = tuple(cb_rung(name) for name in FP8_EXPERT_FORMATS)
FP8_NONEXPERT_RUNGS = tuple(cb_rung(name) for name in FP8_NONEXPERT_FORMATS)

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
_DSV4_NONEXPERT_ROLE_UNIT_COUNTS = {
    role: 43 for role in _ALL_ROLES
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
        # Spans the full dense learned ladder and includes both rungs a routed
        # expert can legally take, so routed cells are priced by interpolation
        # inside the panel range rather than off its end.  FP8-CB lattice
        # declares one legal rung (K48) and is therefore absent by contract:
        # it is priced by its own anchor, and plan_cb_panel_and_validation
        # refuses a panel it could not use.
        **{
            ("fp8_cb", role, LEARNED_BASIS): (
                "FP8_CB_K28", "FP8_CB_K32",
                "FP8_CB_K40", "FP8_CB_K44",
            )
            for role in _ALL_ROLES
        },
    },
    validation_rungs_by_segment={
        # NVFP4-CB shipped with NO held-out validation on the first pass, so
        # 233,275 of the 334,454 legal DP cells -- 69.7% -- were priced by a
        # fit nothing ever checked.  (The 471,065 in the 2026-08-11 handover
        # predates the gridbook k %% 4 fused-kernel law narrowing the FP8
        # ladders; it was right then and is not the live menu.)  K13/K14/K17 are legal on this ladder and used by
        # neither the panel nor the anchor, so two of them are free two-axis
        # hold-outs: K13 sits between panel rungs K12 and K15, K17 between
        # K16 and K18, and together they straddle the K15 anchor.
        **{
            ("nvfp4_cb", role, LATTICE_BASIS): (
                "NVFP4_CB_K13", "NVFP4_CB_K17",
            )
            for role in _ALL_ROLES
        },
        # K36 is the one learned FP8 rung the panel does not contain and the
        # anchor is not, so it is the only two-axis hold-out this segment
        # admits (K48 is lattice, not learned).  K40 is a panel rung and is
        # kept as a unit-axis hold-out: it satisfies the >=2-rungs-per-unit
        # requirement and still tests transfer to held-out units.  K28/K44
        # were both panel rungs, so the previous pair tested only the unit
        # axis and the report could not say so.
        **{
            ("fp8_cb", role, LEARNED_BASIS): (
                "FP8_CB_K36", "FP8_CB_K40",
            )
            for role in _ALL_ROLES
        },
    },
    panel_units_per_role=32,
    validation_units_per_role=4,
    seed=42,
)
# One anchor format per (family, basis) is rendered for *every* unit in that
# segment, so an anchor must be legal everywhere the segment reaches.  Routed
# experts stop at K32 and dense units run to K48, so the FP8 learned anchor
# has to come from the {K28, K32} intersection; K32 is the interior choice,
# halving the worst extrapolation distance on the dense ladder (K32->K44 is
# 12 rungs, K28->K44 is 16) at no cost on the 2-rung routed ladder.  FP8
# lattice declares exactly one legal rung, so its anchor is that rung and
# pricing there is measured rather than extrapolated.
DSV4_ANCHOR_FORMATS = {
    ("nvfp4_cb", LATTICE_BASIS): "NVFP4_CB_K15",
    ("fp8_cb", LEARNED_BASIS): "FP8_CB_K32",
    ("fp8_cb", LATTICE_BASIS): "FP8_CB_K48",
}
_TERMINAL_BY_SOURCE_KIND = {
    "mxfp4": "MXFP4_SOURCE",
    "fp8_ue8m0": "FP8_BLOCK_UE8M0_SOURCE",
    "bf16": "BF16",
}


class DSv4CampaignError(RuntimeError):
    """A DSv4 inventory, identity, or orchestration refusal."""


def _release_runtime_identity() -> dict[str, str]:
    commit = os.environ.get("PRISMAQUANT_IDENTITY_GIT_COMMIT", "")
    if _FULL_GIT_COMMIT.fullmatch(commit) is None:
        raise DSv4CampaignError(
            "activation-safe replay requires the full immutable PrismaQuant "
            "runtime commit identity"
        )
    if os.environ.get("PRISMAQUANT_IDENTITY_GIT_DIRTY") != "0":
        raise DSv4CampaignError(
            "activation-safe replay requires a clean immutable runtime identity"
        )
    if "PYTHONPATH" in os.environ:
        raise DSv4CampaignError(
            "activation-safe replay requires PYTHONPATH to be absent"
        )
    if (
        os.environ.get("PYTHONSAFEPATH") != "1"
        or not sys.flags.safe_path
    ):
        raise DSv4CampaignError(
            "activation-safe replay requires Python safe-path mode"
        )
    if (
        os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
        or not sys.dont_write_bytecode
    ):
        raise DSv4CampaignError(
            "activation-safe replay requires bytecode writes to be disabled"
        )
    if (
        os.environ.get("PYTHONNOUSERSITE") != "1"
        or not sys.flags.no_user_site
    ):
        raise DSv4CampaignError(
            "activation-safe replay requires disabled Python user-site imports"
        )
    root_raw = os.environ.get("PQ_RUNTIME_PRISMAQUANT_ROOT", "")
    tree = os.environ.get("PQ_RUNTIME_PRISMAQUANT_TREE", "")
    closure = os.environ.get(
        "PQ_RUNTIME_PRISMAQUANT_CLOSURE_SHA256", ""
    )
    root = Path(root_raw)
    if (
        not root_raw
        or not root.is_absolute()
        or root.is_symlink()
        or _FULL_GIT_COMMIT.fullmatch(tree) is None
        or _FULL_SHA256.fullmatch(closure) is None
    ):
        raise DSv4CampaignError(
            "activation-safe replay lacks a complete runtime snapshot identity"
        )
    try:
        resolved_root = root.resolve(strict=True)
        expected_module = (
            resolved_root / "prismaquant" / "dsv4_aura_cb_reprice.py"
        ).resolve(strict=True)
        observed_module = Path(__file__).resolve(strict=True)
        verifier_path = (
            resolved_root / "tools" / "prismaquant_runtime_snapshot.py"
        )
        if verifier_path.is_symlink() or not verifier_path.is_file():
            raise ValueError("runtime snapshot verifier is absent or unsafe")
        if observed_module != expected_module:
            raise ValueError(
                "loaded DSv4 replay module is outside the selected snapshot"
            )
        spec = importlib.util.spec_from_file_location(
            "_prismaquant_replay_runtime_snapshot", verifier_path
        )
        if spec is None or spec.loader is None:
            raise ValueError("runtime snapshot verifier cannot be loaded")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        identity = verifier.verify_snapshot(
            resolved_root,
            expected_commit=commit,
            expected_tree=tree,
            expected_closure_sha256=closure,
        )
    except Exception as exc:
        raise DSv4CampaignError(
            f"activation-safe replay runtime snapshot failed verification: {exc}"
        ) from exc
    if (
        identity.get("snapshot") != str(resolved_root)
        or identity.get("commit") != commit
        or identity.get("tree") != tree
        or identity.get("closure_sha256") != closure
    ):
        raise DSv4CampaignError(
            "activation-safe replay runtime snapshot identity differs"
        )
    return {
        "snapshot": str(resolved_root),
        "commit": commit,
        "tree": tree,
        "closure_sha256": closure,
    }


def _release_runtime_commit() -> str:
    """Compatibility projection of the full replay runtime proof."""
    return _release_runtime_identity()["commit"]


def _bind_completion_receipt_to_runtime(
    receipt: Mapping[str, object], runtime: Mapping[str, str]
) -> None:
    producer = receipt.get("producer")
    if not isinstance(producer, Mapping):
        raise DSv4CampaignError(
            "campaign completion receipt lacks its runtime producer identity"
        )
    for key in ("commit", "tree", "closure_sha256"):
        if producer.get(key) != runtime.get(key):
            raise DSv4CampaignError(
                "activation-safe replay runtime differs from the completion "
                f"receipt at producer.{key}"
            )


@dataclass(frozen=True)
class PreparedDSv4Campaign:
    args: argparse.Namespace
    profile: object
    probe_stats: Mapping[str, Mapping[str, object]]
    probe_meta: Mapping[str, object]
    cb_context: object
    plugin: CodebookAnchoredFormatPlugin
    format_plan: object
    measurement_format_plan_identity_sha256: str
    format_plan_migration: Mapping[str, object] | None
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
        # 43 layers x 3 projections x the 2 on-law routed rungs = 258 cells.
        for rung in FP8_EXPERT_RUNGS
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
        # 43 layers x 3 projections x the 2 on-law routed rungs = 258 cells.
        for rung in FP8_EXPERT_RUNGS
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
    expected = {
        **{format_name: LATTICE_BASIS for format_name in NVFP4_FORMATS},
        **{
            f"FP8_CB_K{k}": (
                LEARNED_BASIS if k <= 46 else LATTICE_BASIS
            )
            for k in FP8_NONEXPERT_RUNGS
        },
    }
    missing = sorted(set(expected) - set(source_map))
    mismatched = {
        name: (expected[name], source_map.get(name))
        for name in expected
        if name in source_map and source_map[name] != expected[name]
    }
    if missing or mismatched:
        raise DSv4CampaignError(
            "bundle-authoritative basis map differs from the owner policy: "
            f"missing={missing[:8]} mismatched={list(mismatched.items())[:8]}"
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
        "fp8_cb|lattice": 301,
    }
    by_segment: Counter[str] = Counter(
        request.segment.stamp for request in requests
    )
    expected_by_segment = {
        f"{family}|{role}|{basis}": unit_count
        for family, basis, role_counts in (
            ("nvfp4_cb", LATTICE_BASIS, _DSV4_ROLE_UNIT_COUNTS),
            ("fp8_cb", LEARNED_BASIS, _DSV4_ROLE_UNIT_COUNTS),
            (
                "fp8_cb", LATTICE_BASIS,
                _DSV4_NONEXPERT_ROLE_UNIT_COUNTS,
            ),
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
        raise DSv4CampaignError("DSv4 anchor total is not 66,951")
    return dict(counts)


def _w8a16_format_plan_delta(
    historical_plan: object,
    current_plan: object,
) -> dict[str, object]:
    """Prove the 0.8.4 -> pinned-release plan change is version metadata only.

    The historical side is a recorded artifact fact and stays the 0.8.4
    literal.  The current side is whatever the producer pin names today, so it
    binds to the pin constant: the gate's claim is that only the version
    metadata moved, not that the runtime never advances.  Every non-version
    field is still compared exactly below, so a real plan change still fails.
    """

    historical = historical_plan.to_dict()
    current = current_plan.to_dict()
    if historical.get("identity_sha256") != DSV4_W8A16_LEGACY_FORMAT_PLAN_SHA256:
        raise DSv4CampaignError(
            "W8A16 readmission historical format-plan identity is not allowlisted"
        )
    historical_restriction = historical.get("serving_backed_restriction")
    current_restriction = current.get("serving_backed_restriction")
    if not isinstance(historical_restriction, Mapping) or not isinstance(
        current_restriction, Mapping
    ):
        raise DSv4CampaignError(
            "W8A16 readmission requires old and current serving restrictions"
        )
    allowed = {"runtime_version", "rungs_source"}
    old_delta = {name: historical_restriction.get(name) for name in allowed}
    new_delta = {name: current_restriction.get(name) for name in allowed}
    if old_delta != {
        "runtime_version": "0.8.4",
        "rungs_source": "serving_profile_spec:0.8.4",
    } or new_delta != {
        "runtime_version": GRIDBOOK_RUNTIME_RELEASE_VERSION,
        "rungs_source": (
            f"serving_profile_spec:{GRIDBOOK_RUNTIME_RELEASE_VERSION}"
        ),
    }:
        raise DSv4CampaignError(
            "W8A16 readmission format-plan version delta is not the approved "
            f"0.8.4 -> {GRIDBOOK_RUNTIME_RELEASE_VERSION} transition: "
            f"old={old_delta}, new={new_delta}"
        )
    old_normalized = dict(historical)
    new_normalized = dict(current)
    old_normalized.pop("identity_sha256", None)
    new_normalized.pop("identity_sha256", None)
    old_restriction = dict(historical_restriction)
    new_restriction = dict(current_restriction)
    for name in allowed:
        old_restriction.pop(name, None)
        new_restriction.pop(name, None)
    old_normalized["serving_backed_restriction"] = old_restriction
    new_normalized["serving_backed_restriction"] = new_restriction
    if canonical_json(
        old_normalized, where="historical W8A16 format plan"
    ) != canonical_json(new_normalized, where="current W8A16 format plan"):
        raise DSv4CampaignError(
            "W8A16 readmission format plan changed beyond runtime-version "
            "provenance"
        )
    return {
        "historical_identity_sha256": historical["identity_sha256"],
        "current_identity_sha256": current["identity_sha256"],
        "allowed_changed_fields": [
            "serving_backed_restriction.runtime_version",
            "serving_backed_restriction.rungs_source",
            "identity_sha256",
        ],
        "historical_runtime": old_delta,
        "current_runtime": new_delta,
        "semantic_payload_equal_after_allowed_delta": True,
    }


def prepare_dsv4_campaign(
    args: argparse.Namespace,
    *,
    publish_format_plan: bool = True,
    allow_historical_plan_delta: bool = False,
) -> PreparedDSv4Campaign:
    """Complete every CPU identity/legality check before the P0 GPU pass."""
    work_dir = _safe_work_dir(args.work_dir)
    stats, probe_meta = _load_probe(args)
    from prismaquant.model_profiles import detect_profile
    from prismaquant.allocator_candidates import _scan_source_dtype_manifest
    from prismaquant.source_class_format_plan import (
        build_source_class_format_plan,
        load_format_plan,
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
        # The on-law menu is not a CLI opinion: it is the fused mid-M rung set
        # nvfp4_cb declares for the pinned Gridbook release, resolved through
        # the serving lane.  Passing it here is what lets the planner's
        # complete-family contract and the frozen menus above be the same
        # statement -- without it the planner demands all 21 registered
        # FP8-CB rungs, including the fifteen k % 4 != 0 ones this campaign
        # exists to exclude.  DSV4_TARGET_PROFILE also drives --target-profile
        # on the allocator run below, so producer and consumer read one table.
        serving_backed_profile=DSV4_TARGET_PROFILE,
    )
    restriction = format_plan.serving_backed_restriction or {}
    if tuple(restriction.get("fused_mid_m_rungs", ())) != FP8_NONEXPERT_RUNGS:
        raise DSv4CampaignError(
            "the pinned serving runtime does not back the DSv4 on-law rung "
            f"set: profile={restriction.get('profile_id')!r} "
            f"runtime={restriction.get('runtime_version')!r} "
            f"backs {list(restriction.get('fused_mid_m_rungs', ()))}, "
            f"campaign requires {list(FP8_NONEXPERT_RUNGS)}"
        )
    if tuple(format_plan.menus["expert"]) != FP8_EXPERT_FORMATS or tuple(
        format_plan.menus["nonexpert"]
    ) != FP8_NONEXPERT_FORMATS:
        raise DSv4CampaignError(
            "frozen CLI menus differ from the DSv4 on-law contract "
            f"(experts {list(FP8_EXPERT_FORMATS)}, "
            f"nonexperts {list(FP8_NONEXPERT_FORMATS)})"
        )
    format_plan_path = work_dir / "checkpoints" / "source_format_plan.json"
    measurement_format_plan_identity = format_plan.identity_sha256
    format_plan_migration = None
    if publish_format_plan:
        format_plan_path.parent.mkdir(parents=True, exist_ok=True)
        write_format_plan(format_plan, format_plan_path)
    else:
        # Replay treats the completed campaign as immutable input.  Validate
        # its already-published plan against a fresh derivation, but do not
        # even atomically replace it with equivalent bytes.
        try:
            published_plan = load_format_plan(
                format_plan_path,
                verify_current_serving_restriction=(
                    not allow_historical_plan_delta
                ),
            )
        except Exception as exc:
            raise DSv4CampaignError(
                f"completed source format plan is invalid: {format_plan_path}"
            ) from exc
        if allow_historical_plan_delta:
            format_plan_migration = _w8a16_format_plan_delta(
                published_plan, format_plan
            )
            measurement_format_plan_identity = published_plan.identity_sha256
        elif published_plan.to_dict() != format_plan.to_dict():
            raise DSv4CampaignError(
                "completed source format plan differs from fresh derivation"
            )

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
        "measurement_format_plan_identity_sha256": (
            measurement_format_plan_identity
        ),
        **(
            {"format_plan_migration": format_plan_migration}
            if format_plan_migration is not None else {}
        ),
        "format_plan_path": str(format_plan_path),
        "cold_expert_render_units": len(missing_activations),
        "streaming_source_cache": {
            "max_cache_slots": DSV4_STREAMING_CACHE_MAX_SLOTS,
            "effective_prefetch_lookahead": 0,
        },
    }
    return PreparedDSv4Campaign(
        args=args,
        profile=profile,
        probe_stats=stats,
        probe_meta=probe_meta,
        cb_context=cb_context,
        plugin=plugin,
        format_plan=format_plan,
        measurement_format_plan_identity_sha256=(
            measurement_format_plan_identity
        ),
        format_plan_migration=format_plan_migration,
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
        max_cache_slots=DSV4_STREAMING_CACHE_MAX_SLOTS,
    )
    try:
        if (
            runner.context.max_cache_slots != DSV4_STREAMING_CACHE_MAX_SLOTS
            or runner.prefetch_lookahead != 0
        ):
            raise DSv4CampaignError(
                "DSv4 streamed source-cache policy is not fail-closed "
                f"(max_cache_slots={runner.context.max_cache_slots}, "
                f"prefetch_lookahead={runner.prefetch_lookahead})"
            )
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
                "streaming_source_cache": {
                    "max_cache_slots": DSV4_STREAMING_CACHE_MAX_SLOTS,
                    "effective_prefetch_lookahead": 0,
                },
            },
        )
    finally:
        runner.shutdown()
    return payload


def _normalized_purpose_plan(
    raw: Mapping[str, object],
) -> dict[str, dict[str, list[str]]]:
    normalized: dict[str, dict[str, list[str]]] = {}
    for raw_qname, raw_rows in raw.items():
        qname = str(raw_qname)
        if not isinstance(raw_rows, Mapping):
            raise DSv4CampaignError(
                f"render-purpose plan for {qname!r} is not a mapping"
            )
        rows: dict[str, list[str]] = {}
        for raw_format, raw_purposes in raw_rows.items():
            fmt = fr.canonical_format_name(str(raw_format))
            values = (
                [raw_purposes]
                if isinstance(raw_purposes, str) else list(raw_purposes)
            )
            rows[fmt] = [str(value) for value in values]
        normalized[qname] = rows
    return normalized


def _expected_checkpoint_cost_row(
    qname: str,
    fmt: str,
    state_row: Mapping[str, object],
    *,
    n_probes: int,
    diagnostic_expected: bool,
) -> dict[str, object]:
    try:
        samples = [float(value) for value in state_row["x2_probe"]]
        s2 = float(state_row["s2"])
        s4 = float(state_row["s4"])
        dw_source = str(state_row["dw_src"])
    except Exception as exc:
        raise DSv4CampaignError(
            f"AURA journal state is malformed for {qname}@{fmt}"
        ) from exc
    if len(samples) != n_probes or any(
        not math.isfinite(value) or value < 0.0 for value in samples
    ):
        raise DSv4CampaignError(
            f"AURA journal samples are invalid for {qname}@{fmt}"
        )
    if (
        not math.isfinite(s2)
        or not math.isfinite(s4)
        or s2 < 0.0
        or s4 < 0.0
        or not math.isclose(
            sum(samples), s2, rel_tol=1e-15, abs_tol=0.0,
        )
        or not math.isclose(
            sum(value * value for value in samples),
            s4,
            rel_tol=1e-15,
            abs_tol=0.0,
        )
        or dw_source != "production_render"
    ):
        raise DSv4CampaignError(
            f"AURA journal accumulators differ for {qname}@{fmt}"
        )
    inv = 1.0 / float(n_probes)
    mean_x2 = inv * s2
    variance = (
        max((s4 - n_probes * mean_x2 * mean_x2) / (n_probes - 1), 0.0)
        if n_probes >= 2 else 0.0
    )
    row: dict[str, object] = {
        "predicted_dloss": 0.5 * mean_x2,
        "predicted_dloss_stderr": 0.5 * math.sqrt(variance * inv),
        "x2_per_probe": samples,
        "dw_source": "production_render",
        "output_mse_measured": False,
        "cost_source": "aura",
        "production_anchor_measured": True,
        "production_anchor_zero": mean_x2 == 0.0,
    }
    observed_diagnostic = "weight_mse_diagnostic" in state_row
    if observed_diagnostic != diagnostic_expected:
        raise DSv4CampaignError(
            f"AURA journal diagnostic scope differs for {qname}@{fmt}"
        )
    if diagnostic_expected:
        diagnostic = float(state_row["weight_mse_diagnostic"])
        if not math.isfinite(diagnostic) or diagnostic < 0.0:
            raise DSv4CampaignError(
                f"AURA journal diagnostic is invalid for {qname}@{fmt}"
            )
        row.update({
            "weight_mse_diagnostic": diagnostic,
            "weight_mse_diagnostic_normalization": "mean_per_weight",
            "weight_mse_is_cost_input": False,
        })
    return row


def _load_and_audit_completed_streamed_payload(
    prepared: PreparedDSv4Campaign,
    payload_path: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Replay a completed streamed result from its independent unit journal.

    The monolithic pickle is convenient, not its own authority.  Every
    measured scalar and source-weight binding is reconstructed from the
    checksummed per-unit AURA envelopes before CPU-only fitting is admitted.
    Historical synthetic terminal zeros are accepted only in their exact old
    shape and are quarantined from the replay payload.
    """
    from prismaquant.aura_cost import (
        AURA_CHECKPOINT_IDENTITY_SCHEMA,
        AURA_CHECKPOINT_MANIFEST_SCHEMA,
        AURA_CHECKPOINT_UNIT_SCHEMA,
    )
    from prismaquant.production_weight_cache import (
        _combined_source_weights_sha256,
    )

    expected_payload_path = (
        Path(prepared.args.work_dir) / "artifacts" /
        "streamed_anchor_aura.pkl"
    ).resolve(strict=False)
    source_path = Path(payload_path).resolve(strict=False)
    if source_path != expected_payload_path:
        raise DSv4CampaignError(
            "CPU replay accepts only this campaign's completed "
            f"streamed_anchor_aura.pkl: {expected_payload_path}"
        )
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise DSv4CampaignError(
            f"completed streamed AURA payload is absent: {source_path}"
        )
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    try:
        raw_payload = pickle.loads(source_bytes)
    except Exception as exc:
        raise DSv4CampaignError(
            f"streamed AURA payload is unreadable: {source_path}"
        ) from exc
    if not isinstance(raw_payload, Mapping):
        raise DSv4CampaignError("streamed AURA payload is not a mapping")

    checkpoint_root = Path(prepared.args.checkpoint_dir) / "aura"
    manifest_path = checkpoint_root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except Exception as exc:
        raise DSv4CampaignError(
            f"AURA checkpoint manifest is unreadable: {manifest_path}"
        ) from exc
    if not isinstance(manifest, Mapping) or manifest.get(
        "schema"
    ) != AURA_CHECKPOINT_MANIFEST_SCHEMA:
        raise DSv4CampaignError("AURA checkpoint manifest schema differs")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping) or identity.get(
        "schema"
    ) != AURA_CHECKPOINT_IDENTITY_SCHEMA:
        raise DSv4CampaignError("AURA checkpoint identity is absent or invalid")
    identity_sha256 = canonical_json_sha256(
        identity, where="DSv4 replay AURA checkpoint identity"
    )
    if manifest.get("identity_sha256") != identity_sha256:
        raise DSv4CampaignError("AURA checkpoint manifest checksum differs")

    units_by_name = {unit.qname: unit for unit in prepared.units}
    expected_names = set(units_by_name)
    if len(expected_names) != len(prepared.units):
        raise DSv4CampaignError("prepared replay unit scope is duplicated")
    identity_units = identity.get("units")
    if not isinstance(identity_units, Sequence) or isinstance(
        identity_units, (str, bytes)
    ):
        raise DSv4CampaignError("AURA checkpoint identity units are invalid")
    identity_units_by_name: dict[str, Mapping[str, object]] = {}
    for row in identity_units:
        if not isinstance(row, Mapping):
            raise DSv4CampaignError("AURA checkpoint identity unit is invalid")
        qname = str(row.get("qname", ""))
        if not qname or qname in identity_units_by_name:
            raise DSv4CampaignError("AURA checkpoint identity units duplicate")
        identity_units_by_name[qname] = row
    if set(identity_units_by_name) != expected_names:
        raise DSv4CampaignError(
            "AURA checkpoint identity unit scope differs from preparation"
        )
    for qname, row in identity_units_by_name.items():
        probe = prepared.probe_stats[qname]
        expected_shape = [
            int(probe["out_features"]), int(probe["in_features"])
        ]
        if (
            list(row.get("shape", ())) != expected_shape
            or int(row.get("n_params", -1)) != int(probe["n_params"])
            or row.get("dtype") != "torch.bfloat16"
        ):
            raise DSv4CampaignError(
                f"AURA checkpoint unit identity differs for {qname}"
            )
    if (
        int(identity.get("n_probes", -1)) != int(prepared.args.n_probes)
        or identity.get("collect_col_energy") is not False
        or identity.get("require_production_cache") is not True
        or identity.get("calibration", {}).get("calib_hash")
        != prepared.probe_meta["calib_hash"]
    ):
        raise DSv4CampaignError(
            "AURA checkpoint measurement/calibration identity differs"
        )
    raw_chunks = identity.get("chunks")
    if not isinstance(raw_chunks, Sequence) or isinstance(
        raw_chunks, (str, bytes)
    ):
        raise DSv4CampaignError("AURA checkpoint chunk identity is invalid")
    chunk_names = [str(name) for chunk in raw_chunks for name in chunk]
    if len(chunk_names) != len(set(chunk_names)) or set(
        chunk_names
    ) != expected_names:
        raise DSv4CampaignError("AURA checkpoint chunks do not cover units once")

    extra = identity.get("extra")
    if not isinstance(extra, Mapping):
        raise DSv4CampaignError("AURA checkpoint campaign identity is absent")
    if (
        extra.get("campaign_schema") != DSV4_CAMPAIGN_SCHEMA
        or extra.get("source_format_plan_identity_sha256")
        != prepared.measurement_format_plan_identity_sha256
        or extra.get("routed_book_selection_sha256")
        != prepared.routed_selection_sha256
        or extra.get("include_routed_experts") is not True
    ):
        raise DSv4CampaignError("AURA checkpoint campaign identity differs")
    expected_purposes = _normalized_purpose_plan(
        prepared.purposes_by_qname
    )
    checkpoint_purposes = extra.get("production_anchor_render_purposes")
    if not isinstance(checkpoint_purposes, Mapping) or (
        _normalized_purpose_plan(checkpoint_purposes) != expected_purposes
    ):
        raise DSv4CampaignError("AURA checkpoint render-purpose plan differs")
    expected_render_plan = {
        qname: tuple(rows)
        for qname, rows in expected_purposes.items()
    }
    legacy_zero_formats = {
        "MXFP4_SOURCE", "FP8_BLOCK_UE8M0_SOURCE"
    }
    checkpoint_plan = extra.get("streamed_formats_by_qname")
    if not isinstance(checkpoint_plan, Mapping) or set(
        map(str, checkpoint_plan)
    ) != expected_names:
        raise DSv4CampaignError("AURA checkpoint streamed plan scope differs")
    for qname in expected_names:
        raw_formats = checkpoint_plan[qname]
        if isinstance(raw_formats, (str, bytes)):
            raise DSv4CampaignError(
                f"AURA checkpoint streamed plan is invalid for {qname}"
            )
        canonical = [fr.canonical_format_name(fmt) for fmt in raw_formats]
        expected = set(prepared.formats_by_qname[qname])
        if len(canonical) != len(set(canonical)) or not (
            expected <= set(canonical) <= expected | legacy_zero_formats
        ):
            raise DSv4CampaignError(
                f"AURA checkpoint streamed plan differs for {qname}"
            )
    checkpoint_unmeasured = extra.get(
        "unmeasured_streamed_formats_by_qname"
    )
    if checkpoint_unmeasured is not None:
        expected_unmeasured = {
            unit.qname: [next(
                candidate.format_name for candidate in unit.candidates
                if candidate.terminal
            )]
            for unit in prepared.units
        }
        if canonical_json(
            checkpoint_unmeasured,
            where="DSv4 replay checkpoint unmeasured formats",
        ) != canonical_json(
            expected_unmeasured,
            where="DSv4 replay expected unmeasured formats",
        ):
            raise DSv4CampaignError(
                "AURA checkpoint unmeasured-terminal plan differs"
            )

    manifest_units = manifest.get("units")
    if not isinstance(manifest_units, Sequence) or isinstance(
        manifest_units, (str, bytes)
    ):
        raise DSv4CampaignError("AURA checkpoint manifest unit list is invalid")
    checkpoint_paths: dict[str, Path] = {}
    for row in manifest_units:
        if not isinstance(row, Mapping):
            raise DSv4CampaignError("AURA checkpoint manifest unit is invalid")
        qname = str(row.get("qname", ""))
        expected_file = (
            "units/" + hashlib.sha256(qname.encode()).hexdigest() + ".pkl"
        )
        if (
            qname not in expected_names
            or qname in checkpoint_paths
            or row.get("file") != expected_file
        ):
            raise DSv4CampaignError(
                f"AURA checkpoint manifest unit binding differs for {qname}"
            )
        checkpoint_paths[qname] = checkpoint_root / expected_file
    if set(checkpoint_paths) != expected_names:
        raise DSv4CampaignError("AURA checkpoint manifest unit scope differs")
    actual_files = set((checkpoint_root / "units").glob("*.pkl"))
    if actual_files != set(checkpoint_paths.values()):
        raise DSv4CampaignError(
            "AURA checkpoint unit directory is incomplete or contains extras"
        )

    raw_costs = raw_payload.get("costs")
    raw_stats = raw_payload.get("stats")
    raw_provenance = raw_payload.get("provenance")
    if not all(isinstance(value, Mapping) for value in (
        raw_costs, raw_stats, raw_provenance,
    )):
        raise DSv4CampaignError(
            "streamed AURA payload lacks costs/stats/provenance mappings"
        )
    if (
        raw_payload.get("schema") != "prismaquant.aura_cost.v1"
        or int(raw_payload.get("n_probes", -1))
        != int(prepared.args.n_probes)
        or set(map(str, raw_costs)) != expected_names
        or set(map(str, raw_stats)) != expected_names
        or raw_provenance.get("calib_hash")
        != prepared.probe_meta["calib_hash"]
    ):
        raise DSv4CampaignError("streamed AURA payload scope/identity differs")
    raw_purposes = raw_provenance.get(
        "production_anchor_render_purposes"
    )
    if not isinstance(raw_purposes, Mapping) or (
        _normalized_purpose_plan(raw_purposes) != expected_purposes
    ):
        raise DSv4CampaignError("streamed AURA payload purpose plan differs")

    base_renderer = extra.get("production_anchor_renderer")
    raw_renderer = raw_provenance.get("production_anchor_renderer")
    if not isinstance(base_renderer, Mapping) or not isinstance(
        raw_renderer, Mapping
    ):
        raise DSv4CampaignError("production renderer identity is absent")
    if base_renderer.get("arm_identity") != prepared.arm_identity:
        raise DSv4CampaignError("checkpoint production arm identity differs")
    for renderer, where in (
        (base_renderer, "checkpoint"), (raw_renderer, "payload"),
    ):
        renderer_plan = renderer.get("formats_by_qname")
        if not isinstance(renderer_plan, Mapping) or set(
            map(str, renderer_plan)
        ) != expected_names:
            raise DSv4CampaignError(
                f"{where} production renderer scope differs"
            )
        for qname, expected_formats in expected_render_plan.items():
            raw_formats = renderer_plan[qname]
            if isinstance(raw_formats, (str, bytes)):
                raise DSv4CampaignError(
                    f"{where} production renderer plan is invalid for {qname}"
                )
            canonical_formats = tuple(
                fr.canonical_format_name(fmt) for fmt in raw_formats
            )
            # Renderer order is a performance plan; purpose-map keys are
            # canonical JSON and therefore lexically ordered in the old live
            # manifest.  The pair set, not these two independent orderings,
            # is the load-bearing measurement scope.
            if (
                len(canonical_formats) != len(set(canonical_formats))
                or set(canonical_formats) != set(expected_formats)
            ):
                raise DSv4CampaignError(
                    f"{where} production renderer plan differs for {qname}"
                )
    base_static = dict(base_renderer)
    raw_static = dict(raw_renderer)
    base_cb = base_static.pop("cb_render_identity", None)
    raw_cb = raw_static.pop("cb_render_identity", None)
    raw_static.pop("source_weights", None)
    if raw_static != base_static or not isinstance(base_cb, Mapping) or not (
        isinstance(raw_cb, Mapping)
    ):
        raise DSv4CampaignError(
            "completed production renderer differs from checkpoint identity"
        )
    dynamic_source_fields = {
        "source_weights_complete", "source_weights_shapes",
        "source_weights_content_sha256", "source_weights_sha256",
        "render_scope",
    }
    if (
        {key: value for key, value in base_cb.items()
         if key not in dynamic_source_fields}
        != {key: value for key, value in raw_cb.items()
            if key not in dynamic_source_fields}
    ):
        raise DSv4CampaignError(
            "completed sparse CB identity changed immutable renderer inputs"
        )

    n_probes = int(prepared.args.n_probes)
    passthrough_zero = {
        "predicted_dloss": 0.0,
        "output_mse_measured": False,
        "cost_source": "aura_passthrough_zero",
    }
    safe_costs: dict[str, dict[str, object]] = {}
    source_records: dict[str, dict[str, object]] = {}
    unit_payload_descriptors: list[dict[str, str]] = []
    readmitted_fp8_zero = 0
    quarantined_fp8_zero = 0
    quarantined_cross_terminal_zero = 0
    measured_rows = 0
    for qname in sorted(expected_names):
        path = checkpoint_paths[qname]
        try:
            encoded = path.read_bytes()
            envelope = pickle.loads(encoded)
        except Exception as exc:
            raise DSv4CampaignError(
                f"AURA unit checkpoint is corrupt for {qname}: {path}"
            ) from exc
        if not isinstance(envelope, Mapping):
            raise DSv4CampaignError(
                f"AURA unit checkpoint is not an envelope for {qname}"
            )
        payload_bytes = envelope.get("payload")
        if not isinstance(payload_bytes, bytes):
            raise DSv4CampaignError(
                f"AURA unit checkpoint has no payload for {qname}"
            )
        payload_digest = hashlib.sha256(payload_bytes).hexdigest()
        if (
            envelope.get("schema") != AURA_CHECKPOINT_UNIT_SCHEMA
            or envelope.get("qname") != qname
            or envelope.get("identity_sha256") != identity_sha256
            or envelope.get("payload_sha256") != payload_digest
        ):
            raise DSv4CampaignError(
                f"AURA unit checkpoint envelope differs for {qname}"
            )
        try:
            state = pickle.loads(payload_bytes)
        except Exception as exc:
            raise DSv4CampaignError(
                f"AURA unit checkpoint state is corrupt for {qname}"
            ) from exc
        if not isinstance(state, Mapping) or not isinstance(
            state.get("rows"), Mapping
        ):
            raise DSv4CampaignError(
                f"AURA unit checkpoint state is invalid for {qname}"
            )
        unit_payload_descriptors.append({
            "qname": qname, "payload_sha256": payload_digest,
        })
        rows = state["rows"]
        expected_formats = set(expected_render_plan[qname])
        if set(map(str, rows)) != expected_formats:
            raise DSv4CampaignError(
                f"AURA unit checkpoint render rows differ for {qname}"
            )
        try:
            g_trace = float(state["g_trace"])
        except Exception as exc:
            raise DSv4CampaignError(
                f"AURA unit checkpoint g_trace is invalid for {qname}"
            ) from exc
        if not math.isfinite(g_trace) or g_trace < 0.0:
            raise DSv4CampaignError(
                f"AURA unit checkpoint g_trace is invalid for {qname}"
            )
        if state.get("col_energy") is not None:
            raise DSv4CampaignError(
                f"AURA unit checkpoint unexpectedly has col_energy for {qname}"
            )
        raw_source = state.get("source_weight_identity")
        if not isinstance(raw_source, Mapping):
            raise DSv4CampaignError(
                f"AURA unit checkpoint lacks source identity for {qname}"
            )
        try:
            source_record = {
                "shape": [int(dim) for dim in raw_source["shape"]],
                "sha256": str(raw_source["sha256"]).lower(),
            }
        except Exception as exc:
            raise DSv4CampaignError(
                f"AURA unit source identity is malformed for {qname}"
            ) from exc
        probe = prepared.probe_stats[qname]
        if (
            source_record["shape"] != [
                int(probe["out_features"]), int(probe["in_features"])
            ]
            or len(source_record["sha256"]) != 64
            or any(char not in "0123456789abcdef"
                   for char in source_record["sha256"])
        ):
            raise DSv4CampaignError(
                f"AURA unit source identity differs for {qname}"
            )
        source_records[qname] = source_record

        payload_rows = raw_costs[qname]
        if not isinstance(payload_rows, Mapping):
            raise DSv4CampaignError(
                f"streamed AURA costs are invalid for {qname}"
            )
        safe_row: dict[str, object] = {}
        for fmt in expected_render_plan[qname]:
            expected_row = _expected_checkpoint_cost_row(
                qname,
                fmt,
                rows[fmt],
                n_probes=n_probes,
                diagnostic_expected=(
                    "panel" in expected_purposes[qname][fmt]
                ),
            )
            if payload_rows.get(fmt) != expected_row:
                raise DSv4CampaignError(
                    f"streamed AURA scalar differs from journal for "
                    f"{qname}@{fmt}"
                )
            safe_row[fmt] = expected_row
            measured_rows += 1
        terminal = next(
            candidate for candidate in units_by_name[qname].candidates
            if candidate.terminal
        )
        unexpected_formats = set(map(str, payload_rows)) - expected_formats
        for fmt in sorted(unexpected_formats):
            canonical = fr.canonical_format_name(fmt)
            if (
                canonical not in legacy_zero_formats | {terminal.format_name}
                or payload_rows[fmt] != passthrough_zero
            ):
                raise DSv4CampaignError(
                    f"streamed AURA contains an unaudited row {qname}@{fmt}"
                )
            if canonical == terminal.format_name and (
                terminal.allocator_selectable
            ):
                safe_row[canonical] = dict(passthrough_zero)
                if canonical == "FP8_BLOCK_UE8M0_SOURCE":
                    readmitted_fp8_zero += 1
            elif canonical == "FP8_BLOCK_UE8M0_SOURCE":
                quarantined_fp8_zero += 1
            else:
                quarantined_cross_terminal_zero += 1
        safe_costs[qname] = safe_row

        expected_stats = {
            "h_trace": g_trace / float(n_probes),
            "n_params": int(probe["n_params"]),
            "in_features": int(probe["in_features"]),
            "out_features": int(probe["out_features"]),
            "n_probes": n_probes,
        }
        if raw_stats[qname] != expected_stats:
            raise DSv4CampaignError(
                f"streamed AURA stats differ from journal for {qname}"
            )

    raw_source_binding = raw_renderer.get("source_weights")
    if not isinstance(raw_source_binding, Mapping) or (
        raw_source_binding.get("complete") is not True
        or raw_source_binding.get("scope") != "sparse_anchor_plan"
        or raw_source_binding.get("records") != source_records
        or raw_source_binding.get("identity_sha256")
        != canonical_json_sha256(
            source_records, where="DSv4 replay source weights"
        )
    ):
        raise DSv4CampaignError(
            "completed renderer source-weight binding differs from journal"
        )
    cb_qnames = list(raw_cb.get("cb_formats_by_qname", ()))
    cb_shapes = {name: source_records[name]["shape"] for name in cb_qnames}
    cb_content = {name: source_records[name]["sha256"] for name in cb_qnames}
    if (
        raw_cb.get("source_weights_complete") is not True
        or raw_cb.get("source_weights_shapes") != cb_shapes
        or raw_cb.get("source_weights_content_sha256") != cb_content
        or raw_cb.get("source_weights_sha256")
        != _combined_source_weights_sha256(cb_shapes, cb_content)
        or raw_cb.get("render_scope") != "sparse_production_anchors"
        or raw_provenance.get("production_anchor_sparse_render_identity")
        != raw_cb
    ):
        raise DSv4CampaignError(
            "completed sparse CB source identity differs from journal"
        )
    expanded_cb = raw_provenance.get("cb_render_identity")
    if not isinstance(expanded_cb, Mapping):
        raise DSv4CampaignError(
            "streamed AURA payload lacks expanded CB renderer inputs"
        )
    for field in (
        "source_weights_complete", "source_weights_shapes",
        "source_weights_content_sha256", "source_weights_sha256",
    ):
        if expanded_cb.get(field) != raw_cb.get(field):
            raise DSv4CampaignError(
                f"expanded CB source identity differs at {field}"
            )

    if int(raw_provenance.get("dw_production_anchor_rows", -1)) != measured_rows:
        raise DSv4CampaignError(
            "streamed AURA production-render row count differs from journal"
        )
    replay = {
        "schema": DSV4_CPU_REPLAY_SCHEMA,
        "measurement_invoked": False,
        "source_payload": {
            "path": str(source_path),
            "size_bytes": len(source_bytes),
            "sha256": source_sha256,
        },
        "checkpoint_manifest": {
            "path": str(manifest_path.resolve()),
            "size_bytes": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "identity_sha256": identity_sha256,
            "producer_git_commit": identity.get("git_commit"),
            "producer_source_sha256": identity.get(
                "producer_source_sha256"
            ),
        },
        "source_format_plan_identity_sha256": (
            prepared.measurement_format_plan_identity_sha256
        ),
        "current_source_format_plan_identity_sha256": (
            prepared.format_plan.identity_sha256
        ),
        "production_arm_identity_sha256": canonical_json_sha256(
            prepared.arm_identity,
            where="DSv4 replay production arm identity",
        ),
        "unit_checkpoint_count": len(unit_payload_descriptors),
        "unit_payload_set_sha256": canonical_json_sha256(
            unit_payload_descriptors,
            where="DSv4 replay AURA unit payload set",
        ),
        "measured_render_rows_validated": measured_rows,
        "legacy_fp8_terminal_zero_rows_quarantined": quarantined_fp8_zero,
        "legacy_fp8_terminal_zero_rows_readmitted": readmitted_fp8_zero,
        "legacy_cross_terminal_zero_rows_quarantined": (
            quarantined_cross_terminal_zero
        ),
        "fp8_block_terminal_allocator_selectable": all(
            candidate.allocator_selectable
            for unit in prepared.units
            for candidate in unit.candidates
            if candidate.format_name == "FP8_BLOCK_UE8M0_SOURCE"
        ),
        "no_gpu_measurement_or_render": True,
    }
    replay = dict(canonical_json(replay, where="DSv4 CPU replay provenance"))
    sanitized = dict(raw_payload)
    sanitized["costs"] = safe_costs
    sanitized_provenance = dict(raw_provenance)
    sanitized_provenance["cpu_replay"] = replay
    sanitized["provenance"] = sanitized_provenance
    return sanitized, replay


def run_budget_bytes(args: object) -> int:
    """This run's whole-artifact hard budget, in bytes.

    Distinct from :data:`DSV4_W8A16_APPROVED_BUDGET_BYTES`, which is a frozen
    approval record. Note the scope: `--target-disk-gb` bounds ONE artifact
    directory recursively, so when MTP ships as a separate `/draft` artifact
    (the `dspark_cb_sidecar` topology) the draft's bytes are NOT inside this
    number and the operator must split the total across the two directories.
    """
    value = getattr(args, "budget_bytes", None)
    return DSV4_DEFAULT_BUDGET_BYTES if value is None else int(value)


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
        # Both source carriers are byte-verbatim W/A identity terminals.
        # Gridbook release/route admission remains an independent export gate.
        "MXFP4_SOURCE", "FP8_BLOCK_UE8M0_SOURCE", "BF16",
    )
    # A release replay runs from a loader-less, source-bootstrap package
    # namespace. A bare ``python -m`` child could resolve the image's stale
    # site-package instead of the immutable /pq snapshot. When repository
    # tools are present, re-enter through that exact snapshot's bootstrap and
    # remove PYTHONPATH before its fail-closed source proof. Installed wheels do
    # not package repository-level tools/, so they retain the normal module
    # invocation through their already-selected interpreter distribution.
    source_root = Path(__file__).resolve().parents[1]
    source_bootstrap = source_root / "tools" / "prismaquant_source_bootstrap.py"
    strict_root_raw = os.environ.get("PQ_RUNTIME_PRISMAQUANT_ROOT", "")
    if strict_root_raw:
        strict_root = Path(strict_root_raw)
        try:
            resolved_strict_root = strict_root.resolve(strict=True)
        except OSError as exc:
            raise DSv4CampaignError(
                "strict-snapshot allocator launch has an unreadable runtime "
                "source root"
            ) from exc
        if (
            not strict_root.is_absolute()
            or strict_root.is_symlink()
            or resolved_strict_root != source_root
        ):
            raise DSv4CampaignError(
                "strict-snapshot allocator launch source root differs from "
                "the active PrismaQuant module"
            )
        if source_bootstrap.is_symlink() or not source_bootstrap.is_file():
            raise DSv4CampaignError(
                "strict-snapshot allocator launch requires the exact source "
                "bootstrap"
            )
    bootstrap_available = (
        source_bootstrap.is_file() and not source_bootstrap.is_symlink()
    )
    if bootstrap_available:
        env_program = shutil.which("env")
        if not env_program:
            raise DSv4CampaignError(
                "source-snapshot allocator launch requires the env utility"
            )
        allocator_prefix = [
            env_program, "-u", "PYTHONPATH",
            sys.executable, "-P", str(source_bootstrap), "run-module",
            "--source-root", str(source_root), "prismaquant.allocator",
        ]
    else:
        allocator_prefix = [sys.executable, "-m", "prismaquant.allocator"]
    return [
        *allocator_prefix,
        "--probe", str(args.probe),
        "--costs", str(cost_path),
        "--model-override", str(args.model),
        "--target-profile", DSV4_TARGET_PROFILE,
        "--target-disk-gb", f"{run_budget_bytes(args) / 1e9:.9f}",
        "--artifact-overhead-reserve-bytes",
        str(DSV4_ARTIFACT_RESERVE_BYTES),
        # Deliberately NOT implied by --budget-bytes. Tying the partition to
        # the budget would be the same implicit coupling that let one constant
        # mean both this run's target and a frozen approval: the 112.69 GB
        # approval was priced WITH MTP in the payload, and a retarget that
        # silently changed the partition would rewrite what that number meant.
        *(
            arg
            for prefix in (getattr(args, "exclude_source_prefix", None) or ())
            for arg in ("--exclude-source-prefix", str(prefix))
        ),
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
    if data_type in {"nvfp4_cb", "fp8_cb"}:
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
        legal = {
            candidate.format_name for candidate in unit.candidates
            if candidate.allocator_selectable
        }
        if assignment[unit.qname] not in legal:
            raise DSv4CampaignError(
                f"allocator selected source-illegal or unmeasured "
                f"{unit.qname}@"
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
    measured_seconds = {
        "NVFP4_CB_K12": 0.069821,
        "NVFP4_CB_K15": 0.069821,
        "NVFP4_CB_K16": 0.069821,
        "NVFP4_CB_K18": 0.069821,
        "FP8_CB_K28": 0.075109,
        "FP8_CB_K33": 0.144363,
        # Medians of elapsed_encode_seconds in the older 16-expert nested
        # production-shape pilot.  They are real rung timings, but predate the
        # current compiled arm and are therefore conservative proxies.
        "FP8_CB_K41": 1.3973947751555897,
        "FP8_CB_K46": 3.335986553436669,
        "FP8_CB_K47": 3.8868322437820098,
        "FP8_CB_K48": 4.442083168687532,
    }
    # The on-law FP8 menu (k % 4 == 0) lands on rungs the timing sets never
    # sampled.  Each borrows the next rung UP from the same measurement
    # family: encode time grows with K, so a higher rung is a strict
    # over-estimate.  These are labelled projections and are never presented
    # as timings of the rung they price -- only wall-clock planning reads
    # them, never any cost or quality number.
    onlaw_timing_proxy = {
        "FP8_CB_K32": "FP8_CB_K33",
        "FP8_CB_K40": "FP8_CB_K41",
        "FP8_CB_K44": "FP8_CB_K46",
    }
    profile_seconds = dict(measured_seconds)
    for projected, source in onlaw_timing_proxy.items():
        profile_seconds[projected] = measured_seconds[source]
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
                "K28=75.109ms, K33=144.363ms, conservative E8 "
                "K28=69.821ms/expert"
            ),
            "older_high_rungs": (
                "/home/rob/dq-runs/dsv4-flash-0731/nested-pilot/"
                "raw_records.jsonl: medians of elapsed_encode_seconds for "
                "K41/K46/K47/K48"
            ),
            "onlaw_rung_next_rung_up_proxy": {
                projected: source
                for projected, source in sorted(onlaw_timing_proxy.items())
            },
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
            "low-rung proxy; K41/K46/K47/K48 use older measured pilot "
            "medians. No on-law FP8 rung above K28 was itself timed: K32/K40/"
            "K44 borrow the next measured rung up (K33/K41/K46), which "
            "over-estimates because encode time grows with K. P0 is a "
            "measured-phase scaling proxy, not a timing of this new fused "
            "pass. Dot-product, checkpoint-fsync, allocator, and export-copy "
            "wall time are not separately measured."
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


def _finish_dsv4_campaign(
    prepared: PreparedDSv4Campaign,
    streamed_payload: Mapping[str, object],
    *,
    artifacts_dir: Path,
    allocator_output: Path,
    exportable_output: Path,
    report_path: Path,
    replay_provenance: Mapping[str, object] | None = None,
    approved_assignment: Mapping[str, str] | None = None,
) -> int:
    """Shared CPU fit/price/allocate/publish tail for fresh and replay runs."""
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
        validation_observations, anchors, fits,
        panel_requests=prepared.panel_requests,
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
    from prismaquant.allocator_candidates import cost_entry_is_source_passthrough

    unsafe_rows = sorted(
        qname
        for qname, row in cost_payload["costs"].items()
        if "FP8_BLOCK_UE8M0_SOURCE" in row
        and not cost_entry_is_source_passthrough(
            row["FP8_BLOCK_UE8M0_SOURCE"],
            "FP8_BLOCK_UE8M0_SOURCE",
        )
    )
    if unsafe_rows:
        raise DSv4CampaignError(
            "FP8 block terminal lacks exact source-passthrough provenance; "
            f"sample={unsafe_rows[:8]}"
        )
    cost_path = write_cb_cost_payload(
        artifacts_dir / "cost_aura_anchored.pkl", cost_payload
    )
    command = _allocator_command(
        prepared, cost_path=cost_path, output_dir=allocator_output
    )
    command_formats = command[command.index("--formats") + 1].split(",")
    block_terminal_count = sum(
        1
        for unit in prepared.units
        for candidate in unit.candidates
        if candidate.format_name == "FP8_BLOCK_UE8M0_SOURCE"
    )
    block_selectable_count = sum(
        1
        for unit in prepared.units
        for candidate in unit.candidates
        if candidate.format_name == "FP8_BLOCK_UE8M0_SOURCE"
        and candidate.allocator_selectable
    )
    if block_terminal_count and (
        block_selectable_count != block_terminal_count
        or "FP8_BLOCK_UE8M0_SOURCE" not in command_formats
    ):
        raise DSv4CampaignError(
            "W8A16 FP8 block terminal admission differs between prepared "
            "units and allocator command"
        )
    invocation_replay = (
        dict(replay_provenance) if replay_provenance is not None else None
    )
    run_allocator_once(
        command=command,
        output_dir=allocator_output,
        resume=bool(args.resume),
        environment_updates={"PRISMAQUANT_ACTIVATION_FAIR_PRICING": "0"},
        invocation_provenance={
            "cost_currency": AURA_CURRENCY,
            "activation_fair_pricing": False,
            "budget_bytes": run_budget_bytes(args),
            "cost_payload_sha256": _sha256(cost_path),
            "format_plan_identity_sha256": (
                prepared.format_plan.identity_sha256
            ),
            "production_arm_identity_sha256": canonical_json_sha256(
                prepared.arm_identity,
                where="DSv4 allocator production arm identity",
            ),
            "fp8_block_terminal_allocator_selectable": (
                block_selectable_count == block_terminal_count
                and block_terminal_count > 0
            ),
            **(
                {"cpu_replay": invocation_replay}
                if invocation_replay is not None else {}
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
    unselectable_terminals = Counter(
        candidate.format_name
        for unit in prepared.units
        for candidate in unit.candidates
        if candidate.terminal and not candidate.allocator_selectable
    )
    selected_formats = Counter(assignment.values())
    assignment_attestation = None
    if approved_assignment is not None:
        from prismaquant.nvfp4_cb_footprint import (
            assignment_serialization_sha256,
        )

        approved = {str(name): str(fmt) for name, fmt in approved_assignment.items()}
        differing = sorted(
            name for name in set(assignment) | set(approved)
            if assignment.get(name) != approved.get(name)
        )
        assignment_sha256 = assignment_serialization_sha256(assignment)
        approved_sha256 = assignment_serialization_sha256(approved)
        if (
            differing
            or approved_sha256 != DSV4_W8A16_APPROVED_ASSIGNMENT_SHA256
            or assignment_sha256 != approved_sha256
        ):
            raise DSv4CampaignError(
                "W8A16 readmission assignment differs from approved raw "
                f"allocation: sample={differing[:8]}, current={assignment_sha256}, "
                f"approved={approved_sha256}"
            )
        try:
            selection = json.loads(
                (allocator_output / "selection.json").read_text()
            )
            whole = selection["whole_artifact_budget"]
        except Exception as exc:
            raise DSv4CampaignError(
                "W8A16 readmission allocator selection is unreadable"
            ) from exc
        observed_selection = {
            "budget_bytes": selection.get("budget_bytes"),
            "chosen_achieved_bits": selection.get("chosen_achieved_bits"),
            "predicted_dloss": selection.get("predicted_dloss"),
            "selection_tensor_payload_bytes": whole.get(
                "selection_tensor_payload_bytes"
            ),
            "selection_whole_artifact_upper_bound_bytes": whole.get(
                "selection_whole_artifact_upper_bound_bytes"
            ),
        }
        if observed_selection != DSV4_W8A16_APPROVED_SELECTION or whole.get(
            "selection_assignment_sha256"
        ) != assignment_sha256:
            raise DSv4CampaignError(
                "W8A16 readmission selection metrics differ from approved raw "
                f"allocation: {observed_selection}"
            )
        assignment_attestation = {
            "approved_assignment_sha256": approved_sha256,
            "readmitted_assignment_sha256": assignment_sha256,
            "unit_count": len(assignment),
            "full_qname_format_map_equal": True,
            "selection": observed_selection,
        }
    report = {
        "schema": DSV4_CAMPAIGN_SCHEMA,
        "cost_currency": AURA_CURRENCY,
        "budget_bytes": run_budget_bytes(args),
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
        "terminal_admission": {
            "identity_retained_but_allocator_unselectable": dict(
                sorted(unselectable_terminals.items())
            ),
            "criterion": (
                "exact source terminal is selectable only when its registered "
                "activation path is identity"
            ),
            "fp8_block_terminal_selected_count": selected_formats.get(
                "FP8_BLOCK_UE8M0_SOURCE", 0
            ),
            "allocator_command_formats": command_formats,
        },
        **(
            {"cpu_replay": invocation_replay}
            if invocation_replay is not None else {}
        ),
        **(
            {"approved_raw_assignment_attestation": assignment_attestation}
            if assignment_attestation is not None else {}
        ),
    }
    json.dumps(report, allow_nan=False)
    atomic_write_bytes(
        report_path,
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode(),
    )
    write_exportable_artifacts(
        exportable_output,
        allocator_output_dir=allocator_output,
        cb_col_weights_path=args.col_weights,
        provenance=report,
        resume=bool(args.resume),
    )
    return 0


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
    return _finish_dsv4_campaign(
        prepared,
        streamed_payload,
        artifacts_dir=artifacts_dir,
        allocator_output=work_dir / "allocator-aura",
        exportable_output=artifacts_dir / "exportable-aura",
        report_path=artifacts_dir / "campaign_report.json",
    )


def run_dsv4_anchor_replay(
    args: argparse.Namespace, *, control_plane: str,
) -> int:
    """Validate a completed streamed journal and run only the CPU tail."""
    if control_plane != __name__:
        raise DSv4CampaignError(
            f"unexpected DSv4 control plane {control_plane!r}"
        )
    work_dir = Path(args.work_dir)
    receipt_path = completion_receipt_path(work_dir)
    runtime_identity = _release_runtime_identity()
    try:
        completion_receipt = verify_receipt_for_replay(
            receipt_path,
            work_dir=work_dir,
            historical_root=historical_checkpoint_root(work_dir),
            expected_producer_commit=runtime_identity["commit"],
        )
    except CampaignCompletionError as exc:
        raise DSv4CampaignError(
            "activation-safe replay lacks a valid terminal campaign receipt: "
            f"{exc}"
        ) from exc
    _bind_completion_receipt_to_runtime(
        completion_receipt, runtime_identity
    )
    prepared = prepare_dsv4_campaign(args, publish_format_plan=False)
    require_allocator_supersurrogate_support()
    streamed_payload, replay = _load_and_audit_completed_streamed_payload(
        prepared, args.replay_streamed_payload
    )
    try:
        assert_replay_matches_completion_receipt(replay, completion_receipt)
    except CampaignCompletionError as exc:
        raise DSv4CampaignError(
            f"activation-safe replay differs from terminal receipt: {exc}"
        ) from exc
    replay = dict(replay)
    replay["campaign_completion_receipt"] = {
        "path": str(receipt_path.resolve(strict=True)),
        "receipt_sha256": completion_receipt["receipt_sha256"],
        "producer_commit": completion_receipt["producer"]["commit"],
    }
    replay_artifacts = work_dir / "artifacts" / "replay-activation-safe"
    return _finish_dsv4_campaign(
        prepared,
        streamed_payload,
        artifacts_dir=replay_artifacts,
        allocator_output=work_dir / "allocator-aura-activation-safe",
        exportable_output=(
            work_dir / "artifacts" / "exportable-aura-activation-safe"
        ),
        report_path=replay_artifacts / "campaign_report.json",
        replay_provenance=replay,
    )


def _assert_w8a16_legacy_receipt(
    receipt: Mapping[str, object],
) -> None:
    producer = receipt.get("producer")
    service = receipt.get("service")
    if not isinstance(producer, Mapping) or any(
        producer.get(key) != value
        for key, value in DSV4_W8A16_LEGACY_PRODUCER.items()
    ):
        raise DSv4CampaignError(
            "W8A16 readmission receipt producer is not the exact allowlisted "
            "measurement snapshot"
        )
    if (
        receipt.get("receipt_sha256") != DSV4_W8A16_LEGACY_RECEIPT_SHA256
        or not isinstance(service, Mapping)
        or service.get("result") != "success"
        or service.get("invocation_id") != DSV4_W8A16_LEGACY_INVOCATION_ID
    ):
        raise DSv4CampaignError(
            "W8A16 readmission receipt/service identity is not allowlisted"
        )


def _w8a16_runtime_contract_proof() -> dict[str, object]:
    from prismaquant.gridbook_runtime_pin import (
        GRIDBOOK_RUNTIME_CONTRACT_SCHEMA,
        GRIDBOOK_RUNTIME_RELEASE_VERSION as PINNED_VERSION,
        load_gridbook_runtime_pin,
        require_exact_gridbook_runtime_release,
        supports_source_fp8_block128_w8a16,
    )

    pin = load_gridbook_runtime_pin()
    try:
        require_exact_gridbook_runtime_release(pin)
    except Exception as exc:
        raise DSv4CampaignError(
            f"W8A16 readmission requires a resolved Gridbook release pin: {exc}"
        ) from exc
    if (
        pin.version != PINNED_VERSION
        or pin.version_is_release is not True
        or pin.runtime_contract_schema != GRIDBOOK_RUNTIME_CONTRACT_SCHEMA
        or not supports_source_fp8_block128_w8a16(pin)
    ):
        raise DSv4CampaignError(
            f"W8A16 readmission requires released Gridbook {PINNED_VERSION} "
            f"with {GRIDBOOK_RUNTIME_CONTRACT_SCHEMA} "
            "source_fp8_block128_w8a16=1"
        )
    return {
        "schema": pin.schema,
        "repository": pin.repository,
        "commit": pin.commit,
        "version": pin.version,
        "version_is_release": pin.version_is_release,
        "runtime_contract_schema": pin.runtime_contract_schema,
        "required_abi_features": dict(pin.required_abi_features),
    }


def _load_w8a16_approved_raw_assignment(
    work_dir: Path,
) -> tuple[dict[str, str], dict[str, object]]:
    from prismaquant.layer_config import load_assignment
    from prismaquant.nvfp4_cb_footprint import assignment_serialization_sha256

    publication = work_dir / "artifacts" / "exportable-aura"
    layer_config = publication / "layer_config.json"
    selection_path = publication / "selection.json"
    col_weights = publication / "cb_col_weights.pkl"
    expected_files = {
        layer_config: DSV4_W8A16_APPROVED_LAYER_CONFIG_SHA256,
        selection_path: DSV4_W8A16_APPROVED_SELECTION_SHA256,
        col_weights: DSV4_W8A16_APPROVED_CB_COL_WEIGHTS_SHA256,
    }
    for path, expected_sha256 in expected_files.items():
        try:
            observed = _sha256(path)
        except Exception as exc:
            raise DSv4CampaignError(
                f"approved raw publication input is unreadable: {path}"
            ) from exc
        if observed != expected_sha256:
            raise DSv4CampaignError(
                f"approved raw publication input changed: {path} "
                f"observed={observed} expected={expected_sha256}"
            )
    assignment = load_assignment(layer_config)
    assignment_sha256 = assignment_serialization_sha256(assignment)
    if (
        len(assignment) != DSV4_TOTAL_UNITS
        or assignment_sha256 != DSV4_W8A16_APPROVED_ASSIGNMENT_SHA256
        or Counter(assignment.values()).get("FP8_BLOCK_UE8M0_SOURCE", 0) != 120
    ):
        raise DSv4CampaignError(
            "approved raw publication assignment no longer matches its "
            "allowlisted identity"
        )
    selection = json.loads(selection_path.read_text())
    whole = selection.get("whole_artifact_budget")
    observed_selection = {
        "budget_bytes": selection.get("budget_bytes"),
        "chosen_achieved_bits": selection.get("chosen_achieved_bits"),
        "predicted_dloss": selection.get("predicted_dloss"),
        "selection_tensor_payload_bytes": (
            whole.get("selection_tensor_payload_bytes")
            if isinstance(whole, Mapping) else None
        ),
        "selection_whole_artifact_upper_bound_bytes": (
            whole.get("selection_whole_artifact_upper_bound_bytes")
            if isinstance(whole, Mapping) else None
        ),
    }
    if (
        observed_selection != DSV4_W8A16_APPROVED_SELECTION
        or not isinstance(whole, Mapping)
        or whole.get("selection_assignment_sha256") != assignment_sha256
    ):
        raise DSv4CampaignError(
            "approved raw publication selection stamp changed"
        )
    return assignment, {
        "publication": str(publication.resolve(strict=True)),
        "layer_config_sha256": expected_files[layer_config],
        "selection_sha256": expected_files[selection_path],
        "cb_col_weights_sha256": expected_files[col_weights],
        "assignment_sha256": assignment_sha256,
        "unit_count": len(assignment),
        "fp8_block_terminal_count": 120,
        "selection": observed_selection,
    }


def run_dsv4_w8a16_readmission(
    args: argparse.Namespace, *, control_plane: str,
) -> int:
    """CPU-only, no-clobber reinterpretation of the exact completed journal."""

    if control_plane != __name__:
        raise DSv4CampaignError(
            f"unexpected DSv4 control plane {control_plane!r}"
        )
    if bool(args.resume):
        raise DSv4CampaignError("W8A16 readmission is fresh-only; --resume is invalid")
    work_dir = Path(args.work_dir)
    runtime_identity = _release_runtime_identity()
    runtime_contract = _w8a16_runtime_contract_proof()
    receipt_path = completion_receipt_path(work_dir)
    try:
        completion_receipt = verify_receipt_for_replay(
            receipt_path,
            work_dir=work_dir,
            historical_root=historical_checkpoint_root(work_dir),
            expected_producer_commit=DSV4_W8A16_LEGACY_PRODUCER["commit"],
        )
    except CampaignCompletionError as exc:
        raise DSv4CampaignError(
            f"W8A16 readmission lacks its exact terminal receipt: {exc}"
        ) from exc
    _assert_w8a16_legacy_receipt(completion_receipt)
    approved_assignment, approved_raw = _load_w8a16_approved_raw_assignment(
        work_dir
    )

    replay_artifacts = work_dir / "artifacts" / "replay-w8a16-readmission"
    allocator_output = work_dir / "allocator-aura-w8a16-readmission"
    exportable_output = (
        work_dir / "artifacts" / "exportable-aura-w8a16-readmission"
    )
    occupied = [
        path for path in (replay_artifacts, allocator_output, exportable_output)
        if path.exists()
    ]
    if occupied:
        raise DSv4CampaignError(
            "W8A16 readmission refuses to overwrite prior outputs: "
            + ", ".join(map(str, occupied))
        )

    prepared = prepare_dsv4_campaign(
        args,
        publish_format_plan=False,
        allow_historical_plan_delta=True,
    )
    require_allocator_supersurrogate_support()
    streamed_payload, cpu_replay = _load_and_audit_completed_streamed_payload(
        prepared, args.replay_streamed_payload
    )
    try:
        assert_replay_matches_completion_receipt(
            cpu_replay, completion_receipt
        )
    except CampaignCompletionError as exc:
        raise DSv4CampaignError(
            f"W8A16 readmission differs from terminal receipt: {exc}"
        ) from exc
    readmission = {
        "schema": DSV4_W8A16_READMISSION_SCHEMA,
        "measurement_invoked": False,
        "no_gpu_measurement_or_render": True,
        "legacy_measurement_producer": dict(DSV4_W8A16_LEGACY_PRODUCER),
        "current_interpretation_producer": dict(runtime_identity),
        "campaign_completion_receipt": {
            "path": str(receipt_path.resolve(strict=True)),
            "receipt_sha256": completion_receipt["receipt_sha256"],
            "service_invocation_id": DSV4_W8A16_LEGACY_INVOCATION_ID,
        },
        "format_plan_migration": prepared.format_plan_migration,
        "gridbook_runtime_pin": runtime_contract,
        "approved_raw_publication": approved_raw,
        "cpu_replay": cpu_replay,
    }
    readmission = dict(canonical_json(
        readmission, where="DSv4 W8A16 readmission provenance"
    ))
    return _finish_dsv4_campaign(
        prepared,
        streamed_payload,
        artifacts_dir=replay_artifacts,
        allocator_output=allocator_output,
        exportable_output=exportable_output,
        report_path=replay_artifacts / "campaign_report.json",
        replay_provenance=readmission,
        approved_assignment=approved_assignment,
    )


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
    parser.add_argument(
        "--replay-streamed-payload",
        help=(
            "CPU-only replay of this work-dir's completed "
            "artifacts/streamed_anchor_aura.pkl. Every row is revalidated "
            "against the durable AURA unit journal after a bound terminal "
            "campaign receipt authorizes replay; no model/GPU work runs."
        ),
    )
    parser.add_argument(
        "--budget-bytes",
        type=int,
        default=None,
        help=(
            "whole-artifact hard budget in bytes for THIS run's allocation "
            f"(default {DSV4_DEFAULT_BUDGET_BYTES}, the 112.69 GB artifact). "
            "Scope is one artifact directory measured recursively, so if MTP "
            "ships as a separate /draft artifact its bytes are NOT included "
            "here and the total must be split across the two directories."
        ),
    )
    parser.add_argument(
        "--exclude-source-prefix",
        action="append",
        default=None,
        metavar="PREFIX",
        help=(
            "Passed through to the allocator. Declares that this artifact "
            "does not ship the source tensors under PREFIX because they ship "
            "separately. REQUIRED when MTP goes to its own /draft directory: "
            "the probe has zero MTP rows, so the recipe cannot mention MTP "
            "and --mtp-format is inert, and without this flag MTP's 10.863 GB "
            "is priced into every rung. That under-fills the budget by the "
            "excluded mass, which the exporter's fail-closed recursive stat "
            "cannot catch because the artifact lands UNDER budget."
        ),
    )
    parser.add_argument(
        "--w8a16-readmission",
        action="store_true",
        help=(
            "reinterpret only the exact allowlisted completed DSv4 journal "
            "under the raw-resident Gridbook W8A16 contract; CPU-only and "
            "fresh/no-clobber"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.require_production_cache:
        parser.error("DSv4 anchors require the production render arm")
    if args.budget_bytes is not None and (
        args.budget_bytes <= DSV4_ARTIFACT_RESERVE_BYTES
    ):
        parser.error(
            f"--budget-bytes must exceed the {DSV4_ARTIFACT_RESERVE_BYTES}B "
            "non-tensor reserve; the selection contract is "
            "tensor_payload + reserve <= budget"
        )
    if args.format_plan:
        parser.error(
            "DSv4 format plan is derived from the exact source census; "
            "external replacement is not accepted by the frozen launcher"
        )
    if Path(args.work_dir).resolve(strict=False) == Path("/tmp") or Path(
        "/tmp"
    ) in Path(args.work_dir).resolve(strict=False).parents:
        parser.error("campaign work-dir may not be under /tmp")
    if args.replay_streamed_payload:
        if args.w8a16_readmission:
            return run_dsv4_w8a16_readmission(
                args, control_plane=__name__
            )
        return run_dsv4_anchor_replay(args, control_plane=__name__)
    if args.w8a16_readmission:
        parser.error("--w8a16-readmission requires --replay-streamed-payload")
    return run_dsv4_anchor_campaign(args, control_plane=__name__)


if __name__ == "__main__":  # pragma: no cover - frozen shell entrypoint
    raise SystemExit(main())


__all__ = [
    "DSV4_BOUNDED_CAMPAIGN_WORKER",
    "DSV4_DEFAULT_BUDGET_BYTES",
    "DSV4_W8A16_APPROVED_BUDGET_BYTES",
    "run_budget_bytes",
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
    "run_dsv4_anchor_replay",
    "run_dsv4_w8a16_readmission",
]
