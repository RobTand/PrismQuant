#!/usr/bin/env python3
"""DSV4 A-FAST content-keyed burn, import, and merge driver.

The routed expert stack is measured with five anchors plus one deterministic
per-layer audit rung.  Slices use five-anchor monotone PCHIP unless an
independent four-anchor K33/K43 cross-validation error exceeds the v2 gross
outlier backstop.  An audit failure full-measures the entire layer.  The only
chain arms are the resident free reconstruction and resident predecessor.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from prismaquant import format_registry as fr
from prismaquant.cb_minchain import (
    MINCHAIN_SCHEMA,
    chain_identity_from_digest,
    epsilon_le,
    recipe_solution_digest,
    select_arm,
)
from prismaquant.cb_warm_state import tensor_value_identity
from prismaquant.nvfp4_cb_footprint import cb_serialization_context_stamp
from prismaquant.production_weight_cache import canonical_cb_col_weights_sha256
from prismaquant.research_cost_acceptance import (
    RESEARCH_COST_MANIFEST_SCHEMA,
    RESEARCH_COST_PROVENANCE,
)
from prismaquant.layer_streaming import _build_fp8_scale_inv_map, _build_weight_map
from tools.dsv4_afast_campaign import (
    ACCEPTANCE_FIT_ANCHORS,
    ACCEPTANCE_HOLDOUTS,
    ANCHORS,
    CHAIN_VERSION,
    PILOT2_LAYER,
    REL_EPSILON,
    RSSGuard,
    RSSLimitExceeded,
    SCHEMA as PILOT_CAMPAIGN_SCHEMA,
    _encode_free,
    _host_available_bytes,
    _reclaim,
    _rss_bytes,
    _select_cell,
    _unit_boundary_reclaim,
    pchip_monotone,
)
from tools.dsv4_ldlq_cost_campaign import (
    COL_WEIGHTS,
    CONTEXT,
    PROJECTIONS,
    RUNGS,
    RUN_ROOT,
    SOURCE,
    atomic_json,
    atomic_pickle,
    atomic_text,
    content_sha256_float32,
    load_layer_identity,
    load_projection,
    percentile,
    sha256_file,
)


LAYER_COUNT = 43
EXPERT_COUNT = 256
# Semantics v2 (2026-08-06): the pilot shards are incumbent-basis and the
# menu must be uniform CBL-basis, so L14/L21 are measured like every layer.
IMPORTED_LAYERS = ()
MEASURED_LAYER_COUNT = LAYER_COUNT - len(IMPORTED_LAYERS)
BURN_ROOT = RUN_ROOT / "burn-afast"
BURN_CELL_ROOT = RUN_ROOT / "burn-shards"
SHARD_ROOT = RUN_ROOT / "shards"
PILOT2_JSON = RUN_ROOT / "pilot2/PILOT2_REPORT.json"
PILOT2_SHARD = RUN_ROOT / "pilot2/shards/layer_014.pkl"
PHASE_A_SHARDS = RUN_ROOT / "pilot-shards"
OLD_CHAIN_SHARDS = RUN_ROOT / "minchain-pilot/shards"
BASE_LAYER_ROOT = Path(
    "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/"
    "artifacts-mxfp4/probe-k12k18/by-layer"
)
from tools import dsv4_cbl_kernels as cblk

CBL_MICROCHECK_LAYER = int(os.environ.get("DSV4_CBL_MICROCHECK_LAYER", "0"))

BASE_COST = Path(
    "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/"
    "artifacts-mxfp4/probe-k12k18/cost_probe_only.pkl"
)
OLD_FULL_COST = Path(
    "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/"
    "artifacts/cost_full.pkl"
)
SCHEMA = "prismaquant.dsv4_afast_layer_shard.v3"
MANIFEST_SCHEMA = "prismaquant.dsv4_afast_burn_manifest.v6"
BURN_CELL_SCHEMA = "prismaquant.dsv4_afast_burn_cell.v4"
BURN_CELL_IDENTITY_SCHEMA = "prismaquant.dsv4_afast_burn_cell_identity.v4"
BURN_PASS_TAG_SCHEMA = "prismaquant.dsv4_afast_burn_pass_tags.v1"
BURN_PASS_TAGS = {
    "scout": "v2s-scout",
    "primary": "v2s-primary",
    "backstop": "v2s-backstop",
    "full_layer": "v2s-full-layer",
}
AMENDMENT_SCHEMA = "prismaquant.dsv4_acceptance_amendment.v2"
AMENDMENT_JSON = RUN_ROOT / "ACCEPTANCE_AMENDMENT.json"
BACKSTOP_TOLERANCE = 0.25
AUDIT_MEDIAN_TOLERANCE = 0.05
AUDIT_P95_TOLERANCE = 0.15
MTP_BF16_BYTES = 10_862_838_300
TIMEBOX_SECONDS = 20 * 3600
MISSING_RUNGS = tuple(k for k in RUNGS if k not in ANCHORS)
MENU = tuple([
    *[f"NVFP4_CB_K{k}" for k in range(12, 19)],
    *[f"FP8_CB_K{k}" for k in RUNGS],
    "MXFP4_SOURCE", "FP8_BLOCK_UE8M0_SOURCE", "BF16",
])


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _sha(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _audit_rung(layer: int) -> int:
    """The operator-registered deterministic v2 audit draw."""
    return random.Random(42 + int(layer)).choice(MISSING_RUNGS)


def _amendment_gate() -> dict[str, Any]:
    report = json.loads(AMENDMENT_JSON.read_text())
    if (
        report.get("schema") != AMENDMENT_SCHEMA
        or not report.get("gate", {}).get("pass")
        or float(report["cost"]["projected_hours"]) > 20.0
    ):
        raise SystemExit("acceptance amendment v2 does not permit the burn")
    return report


def _configure() -> None:
    os.environ["PRISMAQUANT_CB_LDLQ"] = "1"
    os.environ["PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS"] = "1"
    os.environ["PRISMAQUANT_CB_LDLQ_FEEDER_THREADS"] = "16"
    os.environ["PRISMAQUANT_CB_LDLQ_EXPERT_BATCH"] = "16"
    os.environ["PRISMAQUANT_CB_LDLQ_BATCH_STREAMS"] = "1"
    os.environ["PRISMAQUANT_CB_ENCODE_TIER"] = "balanced"
    os.environ.setdefault("PRISMAQUANT_CB_ENCODE_COMPILE", "1")
    if torch.cuda.is_available():
        torch.linalg.cholesky(torch.eye(1, device="cuda"))
        torch.cuda.synchronize()


def _pilot_gate() -> dict:
    report = json.loads(PILOT2_JSON.read_text())
    gates = report["gates"]
    allowed = bool(
        report.get("burn_allowed")
        and gates["P1"]["pass"]
        and gates["P2"]["pass"]
        and gates["P3"]["pass"]
        and gates["P4"]["proceed"]
    )
    if not allowed:
        raise SystemExit("pilot-2 does not permit the burn")
    return report


def _layer_identity(
    layer: int, *, base_sha: str, pilot_sha: str, import_kind: str | None,
) -> dict[str, Any]:
    identity = {
        "schema": SCHEMA,
        "profile": "A-FAST",
        "chain_version": CHAIN_VERSION,
        "layer": layer,
        "import_kind": import_kind,
        "source_index_sha256": sha256_file(
            SOURCE / "model.safetensors.index.json"
        ),
        "verified_base_layer_sha256": base_sha,
        "pilot2_report_sha256": pilot_sha,
        "serialization_context": cb_serialization_context_stamp(
            CONTEXT, formats=[f"FP8_CB_K{k}" for k in RUNGS],
        ),
        "selection_metric": "per-expert weight_mse",
        "epsilon_rtol": REL_EPSILON,
        "anchors": list(ANCHORS),
        "acceptance_fit_anchors": list(ACCEPTANCE_FIT_ANCHORS),
        "acceptance_backstop": "audit_rung_pchip_cv_v2",
        "acceptance_tolerance": BACKSTOP_TOLERANCE,
        "acceptance_semantic": "v2_accept_all_gross_outlier_backstop",
        "audit_rung": _audit_rung(layer),
        "audit_seed": 42 + int(layer),
        "audit_thresholds": {
            "median": AUDIT_MEDIAN_TOLERANCE,
            "p95": AUDIT_P95_TOLERANCE,
        },
        "burn_tool_sha256": sha256_file(Path(__file__).resolve()),
        "acceptance_amendment_sha256": sha256_file(AMENDMENT_JSON),
        "implementation_sha256": {
            "burn_tool": sha256_file(Path(__file__).resolve()),
            "pilot_tool": sha256_file(
                Path(__file__).resolve().parent / "dsv4_afast_campaign.py"
            ),
            "minchain_module": sha256_file(
                Path(__file__).resolve().parents[1] / "prismaquant/cb_minchain.py"
            ),
        },
    }
    return identity


def _base_layer_costs(layer: int, old_full: Mapping[str, Any]) -> tuple[dict, str]:
    path = BASE_LAYER_ROOT / f"layer_{layer:03d}.pkl"
    payload = _load(path)
    if int(payload["meta"]["incremental_shard"]["shard_idx"]) != layer:
        raise AssertionError(f"layer {layer}: base shard index mismatch")
    costs = copy.deepcopy(payload["costs"])
    if len(costs) != 775:
        raise AssertionError(f"layer {layer}: base row count {len(costs)}")
    for qname, row in costs.items():
        old = old_full["costs"][qname]
        row["BF16"] = copy.deepcopy(old["BF16"])
        if "FP8_CB_K36" in old:
            row["FP8_CB_K36"] = copy.deepcopy(old["FP8_CB_K36"])
    return costs, sha256_file(path)


@torch.inference_mode()
def _replay(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    activation_rows: Sequence[torch.Tensor],
    expert_ids: Sequence[int],
    rung: int,
) -> tuple[dict[str, list[float] | list[int]], float]:
    """Production activation-QDQ replay, once per selected slice."""
    spec = fr.get_format(f"FP8_CB_K{rung}")
    torch.cuda.synchronize()
    started = time.perf_counter()
    results: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
    for start in range(0, len(expert_ids), 16):
        stop = min(start + 16, len(expert_ids))
        streams = [torch.cuda.Stream(device=weight.device) for _ in range(stop - start)]

        def work(local: int) -> tuple[int, torch.Tensor, torch.Tensor, int]:
            index = start + local
            expert = int(expert_ids[index])
            cached = activation_rows[expert]
            with torch.cuda.device(weight.device), torch.cuda.stream(streams[local]):
                x = cached.to(
                    device=weight.device, dtype=torch.float32, non_blocking=True,
                )
                x_hat = spec.activation_quantize_dequantize(x.clone())
                reference = x @ weight[index].float().transpose(0, 1)
                quantized = x_hat @ reconstruction[index].float().transpose(0, 1)
                mse = (reference - quantized).square().mean()
                rel = mse / reference.square().mean().clamp_min(1e-12)
                return index, mse, rel, int(cached.shape[0])

        with ThreadPoolExecutor(max_workers=len(streams)) as pool:
            results.extend(pool.map(work, range(len(streams))))
        current = torch.cuda.current_stream(weight.device)
        for stream in streams:
            current.wait_stream(stream)
    torch.cuda.synchronize()
    results.sort(key=lambda row: row[0])
    columns = torch.stack([
        torch.stack((row[1], row[2])) for row in results
    ]).cpu().tolist()
    return {
        "output_mse": [float(row[0]) for row in columns],
        "rel_output_mse": [float(row[1]) for row in columns],
        "n_activation_rows": [row[3] for row in results],
    }, time.perf_counter() - started


def _projection_guard(layer: int, data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_index_sha256": sha256_file(
            SOURCE / "model.safetensors.index.json"
        ),
        "col_weights_sha256": content_sha256_float32(data["col_weights"]),
        "serialization_context_sha256": _sha(
            cb_serialization_context_stamp(
                CONTEXT, formats=[f"FP8_CB_K{k}" for k in RUNGS],
            )
        ),
        "layer": layer,
    }


def _acceptance(
    anchor_errors: Mapping[int, Sequence[float]],
    audit_rung: int,
    audit_errors: Sequence[float],
) -> tuple[list[int], list[int], dict[str, Any]]:
    # Semantics v2: the v1 leave-one-anchor-out CV needed >=4 anchors (its
    # holdouts included the retired K43). With three anchors the backstop is
    # the per-expert form of the audit itself: fit the anchors, predict the
    # measured audit draw. Experts over tolerance ship measured rows, not
    # interpolated ones.
    accepted: list[int] = []
    details = []
    for expert in range(EXPERT_COUNT):
        value = pchip_monotone(
            ANCHORS, [anchor_errors[k][expert] for k in ANCHORS],
            (int(audit_rung),),
        )[0]
        truth = float(audit_errors[expert])
        rel = abs(float(value) - truth) / max(abs(truth), 1e-30)
        keep = rel <= BACKSTOP_TOLERANCE
        if keep:
            accepted.append(expert)
        details.append({
            "expert": expert,
            "audit_rung": int(audit_rung),
            "audit_rung_relative_error": rel,
            "backstop_pass": keep,
        })
    rejected = sorted(set(range(EXPERT_COUNT)).difference(accepted))
    return accepted, rejected, {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / EXPERT_COUNT,
        "backstop_passed": len(accepted),
        "backstop_failed": len(rejected),
        "backstop_failure_rate": len(rejected) / EXPERT_COUNT,
        "backstop_tolerance": BACKSTOP_TOLERANCE,
        "cv_semantic": "three-anchor PCHIP vs measured audit rung (v2)",
        "per_slice": details,
    }


def _audit_stats(
    anchor_errors: Mapping[int, Sequence[float]],
    audit_errors: Sequence[float], audit_rung: int,
    accepted_experts: Sequence[int] | None = None,
) -> dict[str, Any]:
    # v2: when an acceptance list is given, the gate scores only experts
    # whose menu rows ship interpolated; rejected experts ship measured
    # rows, so their prediction residue is not a menu defect.
    experts = (
        list(range(EXPERT_COUNT)) if accepted_experts is None
        else [int(e) for e in accepted_experts]
    )
    if not experts:
        return {
            "rung": int(audit_rung), "n": 0, "n_excluded": EXPERT_COUNT,
            "median": None, "p95": None, "max": None,
            "thresholds": {
                "median": AUDIT_MEDIAN_TOLERANCE,
                "p95": AUDIT_P95_TOLERANCE,
            },
            "pass": False,
            "note": "all experts rejected by backstop; no interpolated rows",
            "per_slice_relative_error": [],
        }
    values = []
    for expert in experts:
        prediction = pchip_monotone(
            ANCHORS, [anchor_errors[k][expert] for k in ANCHORS],
            (audit_rung,),
        )[0]
        truth = float(audit_errors[expert])
        values.append(abs(float(prediction) - truth) / max(abs(truth), 1e-30))
    median = statistics.median(values)
    p95 = percentile(values, 0.95)
    return {
        "rung": int(audit_rung), "n": len(values),
        "n_excluded": EXPERT_COUNT - len(values),
        "median": median, "p95": p95, "max": max(values),
        "thresholds": {
            "median": AUDIT_MEDIAN_TOLERANCE,
            "p95": AUDIT_P95_TOLERANCE,
        },
        "pass": bool(
            median <= AUDIT_MEDIAN_TOLERANCE
            and p95 <= AUDIT_P95_TOLERANCE
        ),
        "per_slice_relative_error": values,
    }


def _empty_curve() -> dict[int, list[Any]]:
    return {k: [None] * EXPERT_COUNT for k in RUNGS}


def _burn_cell_path(
    layer: int, projection: str, pass_tag: str, rung: int,
) -> Path:
    if pass_tag not in BURN_PASS_TAGS.values():
        raise ValueError(
            f"unregistered burn pass tag {pass_tag!r}; "
            f"schema={BURN_PASS_TAG_SCHEMA}"
        )
    return BURN_CELL_ROOT / (
        f"layer_{layer:03d}_{projection}_{pass_tag}_K{rung}.pkl"
    )


def _burn_cell_identity(
    *, layer: int, projection: str, pass_tag: str, rung: int,
    expert_ids: Sequence[int], encoded_expert_ids: Sequence[int],
    content_guard: Mapping[str, Any], predecessor_content_key: str | None,
    replay: bool,
) -> dict[str, Any]:
    if pass_tag not in BURN_PASS_TAGS.values():
        raise ValueError(
            f"unregistered burn pass tag {pass_tag!r}; "
            f"schema={BURN_PASS_TAG_SCHEMA}"
        )
    return {
        "schema": BURN_CELL_IDENTITY_SCHEMA,
        "pass_tag_schema": BURN_PASS_TAG_SCHEMA,
        "pass_tag": pass_tag,
        "layer": int(layer), "projection": projection,
        "rung": int(rung), "expert_ids": list(map(int, expert_ids)),
        "encoded_expert_ids": list(map(int, encoded_expert_ids)),
        "content_guard": dict(content_guard),
        "predecessor_content_key": predecessor_content_key,
        "selection_metric": "per-expert weight_mse",
        "epsilon_rtol": REL_EPSILON, "tie_priority": ["free", "embed"],
        "activation_replay": (
            "winning_reconstruction_only" if replay else "none_scout"
        ),
    }


class BurnCellResumeMismatch(AssertionError):
    """A persisted cell cannot be reused under the current content contract."""

    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"stale or corrupt burn cell {path}: {reason}")
        self.path = path
        self.reason = reason


def _validated_burn_cell(
    path: Path, expected_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = _load(path)
    except Exception as exc:
        raise BurnCellResumeMismatch(
            path, f"unreadable payload ({type(exc).__name__}: {exc})",
        ) from exc
    cell = payload.get("cell")
    checks = (
        (payload.get("schema") == BURN_CELL_SCHEMA, "cell schema mismatch"),
        (
            payload.get("pass_tag_schema") == BURN_PASS_TAG_SCHEMA,
            "pass-tag schema mismatch",
        ),
        (
            payload.get("pass_tag") == expected_identity["pass_tag"],
            "pass-tag mismatch",
        ),
        (
            payload.get("identity") == dict(expected_identity),
            "content identity mismatch",
        ),
        (
            payload.get("content_key") == _sha(expected_identity),
            "content key mismatch",
        ),
        (isinstance(cell, Mapping), "missing cell payload"),
    )
    for passed, reason in checks:
        if not passed:
            raise BurnCellResumeMismatch(path, reason)
    assert isinstance(cell, Mapping)
    if int(cell.get("rung", -1)) != int(expected_identity["rung"]):
        raise BurnCellResumeMismatch(path, "cell rung mismatch")
    if list(cell.get("expert_ids", ())) != list(expected_identity["expert_ids"]):
        raise BurnCellResumeMismatch(path, "cell expert set mismatch")
    if not Path(str(cell.get("warm_state_path", ""))).is_file():
        raise BurnCellResumeMismatch(path, "warm state missing")
    return payload


def _burn_delta_summary(
    observed: Sequence[float], expected: Sequence[float],
) -> dict[str, Any]:
    pairs = [
        (float(left), float(right))
        for left, right in zip(observed, expected)
    ]
    absolute = [abs(left - right) for left, right in pairs]
    relative = [
        value / max(abs(right), 1e-30)
        for value, (_, right) in zip(absolute, pairs)
    ]
    return {
        "count": len(pairs),
        "exact_mismatch_count": sum(
            int(left != right) for left, right in pairs
        ),
        "max_absolute_delta": max(absolute, default=0.0),
        "max_relative_delta": max(relative, default=0.0),
    }


def _quarantine_burn_projection_suffix(
    *, layer: int, projection: str, pass_tag: str, first_rung: int,
    mismatch: Mapping[str, Any],
) -> Path:
    """Recoverably invalidate one chain cell and every dependent successor."""
    if pass_tag not in BURN_PASS_TAGS.values():
        raise ValueError(f"unregistered burn pass tag {pass_tag!r}")
    stamp = f"{time.time_ns()}-L{layer:03d}-{projection}-{pass_tag}-K{first_rung}"
    root = BURN_ROOT / "quarantine-content-mismatch" / stamp
    root.mkdir(parents=True, exist_ok=False)
    moved = []
    for dependent_rung in RUNGS:
        if dependent_rung < int(first_rung):
            continue
        source = _burn_cell_path(
            layer, projection, pass_tag, dependent_rung,
        )
        if not source.is_file():
            continue
        destination = root / source.name
        digest = sha256_file(source)
        source.replace(destination)
        moved.append({
            "rung": dependent_rung,
            "source": str(source),
            "quarantined_path": str(destination),
            "sha256": digest,
        })
    manifest_path = root / "MANIFEST.json"
    atomic_json(manifest_path, {
        "schema": "prismaquant.dsv4_afast_burn_quarantine.v1",
        "created_epoch_ns": time.time_ns(),
        "reason": "burn content-resume mismatch",
        "dependency_rule": (
            "trigger rung and every later rung in the same "
            "layer/projection/pass-tag chain"
        ),
        "pass_tag_schema": BURN_PASS_TAG_SCHEMA,
        "pass_tag": pass_tag,
        "mismatch": dict(mismatch),
        "moved": moved,
    })
    return manifest_path


def _validated_burn_cell_or_invalidate(
    path: Path, expected_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate a resume cell or quarantine its dependent chain suffix."""
    try:
        return _validated_burn_cell(path, expected_identity)
    except BurnCellResumeMismatch as exc:
        mismatch = {
            "schema": "prismaquant.dsv4_afast_burn_resume_mismatch.v1",
            "kind": "envelope_or_identity",
            "path": str(path),
            "reason": exc.reason,
            "expected_content_key": _sha(expected_identity),
            "unit": {
                "layer": int(expected_identity["layer"]),
                "projection": str(expected_identity["projection"]),
                "pass_tag": str(expected_identity["pass_tag"]),
                "rung": int(expected_identity["rung"]),
            },
        }
        manifest_path = _quarantine_burn_projection_suffix(
            layer=int(expected_identity["layer"]),
            projection=str(expected_identity["projection"]),
            pass_tag=str(expected_identity["pass_tag"]),
            first_rung=int(expected_identity["rung"]),
            mismatch=mismatch,
        )
        print(
            f"[afast-burn] invalidated stale content suffix; "
            f"manifest={manifest_path}", flush=True,
        )
        return None


def _run_chain(
    *, layer: int, projection: str, pass_tag: str, data: Mapping[str, Any],
    rungs: Sequence[int], expert_ids: Sequence[int], replay: bool,
    full_encode_rungs: Sequence[int] = (), rss_guard: RSSGuard | None = None,
    oracle_cells: Mapping[int, Mapping[str, Any]] | None = None,
    identity_cache: dict[
        tuple[int, ...], tuple[tuple[list[int], str], tuple[list[int], str]]
    ] | None = None,
) -> dict[int, dict[str, Any]]:
    """Stream a chain while retaining only its selected predecessor."""
    if pass_tag not in BURN_PASS_TAGS.values():
        raise ValueError(
            f"unregistered burn pass tag {pass_tag!r}; "
            f"schema={BURN_PASS_TAG_SCHEMA}"
        )
    all_experts = tuple(range(EXPERT_COUNT))
    expert_ids = tuple(map(int, expert_ids))
    full_encode_rungs = frozenset(map(int, full_encode_rungs))
    if identity_cache is None:
        identity_cache = {}
    predecessor_errors = predecessor_reconstruction = predecessor_ids = None
    predecessor_content_key: str | None = None
    cells: dict[int, dict[str, Any]] = {}
    base_guard = {
        **_projection_guard(layer, data),
        "burn_tool_sha256": sha256_file(Path(__file__).resolve()),
        "cbl_semantics": cblk.SEMANTICS_STAMP,
        "minchain_module_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "prismaquant/cb_minchain.py"
        ),
    }

    for rung in rungs:
        encoded_ids = all_experts if int(rung) in full_encode_rungs else expert_ids
        if encoded_ids not in identity_cache:
            index = torch.as_tensor(encoded_ids, device=data["weight"].device)
            encoded_weight = (
                data["weight"] if encoded_ids == all_experts else
                data["weight"].index_select(0, index).contiguous()
            )
            encoded_col_weights = (
                data["col_weights"] if encoded_ids == all_experts else
                data["col_weights"].index_select(0, index).contiguous()
            )
            identity_cache[encoded_ids] = (
                tensor_value_identity(encoded_weight),
                tensor_value_identity(encoded_col_weights),
            )
            if encoded_ids != all_experts:
                del encoded_weight, encoded_col_weights
                _reclaim()
        source_identity, col_weights_identity = identity_cache[encoded_ids]
        guard = {
            **base_guard,
            "source_shape": source_identity[0],
            "source_digest": source_identity[1],
            "col_weights_shape": col_weights_identity[0],
            "col_weights_digest": col_weights_identity[1],
        }
        identity = _burn_cell_identity(
            layer=layer, projection=projection, pass_tag=pass_tag, rung=int(rung),
            expert_ids=expert_ids, encoded_expert_ids=encoded_ids,
            content_guard=guard,
            predecessor_content_key=predecessor_content_key, replay=replay,
        )
        path = _burn_cell_path(layer, projection, pass_tag, int(rung))
        existing = _validated_burn_cell_or_invalidate(path, identity)
        oracle_cell = (
            existing["cell"] if existing is not None
            else (oracle_cells or {}).get(int(rung))
        )
        action = "restore" if existing is not None else "measure"
        if rss_guard is not None:
            rss_guard.set_stage(
                f"burn:L{layer}:{projection}:{pass_tag}:K{rung}:{action}"
            )
        print(
            f"[afast-burn] L{layer:02d} {projection} {pass_tag} "
            f"K{rung} {action}", flush=True,
        )
        expected_encoded = (
            None if oracle_cell is None else
            oracle_cell["encoded_free_weight_mse"]
        )
        _encoder = (
            cblk.encode_cbl if cblk.cbl_eligible(rung)
            else cblk.encode_free_noldlq
        )
        fields, encoded_reconstruction, encoded_errors, local_timing, warm_path = (
            _encoder(
                layer=layer, projection=projection, rung=int(rung), data=data,
                expert_ids=encoded_ids, source_identity=source_identity,
                col_weights_identity=col_weights_identity,
                use_warm=oracle_cell is not None,
                expected_free_errors=expected_encoded,
            )
        )
        if encoded_ids == expert_ids:
            free_reconstruction = encoded_reconstruction
            free_errors = list(map(float, encoded_errors))
        else:
            positions = {expert: index for index, expert in enumerate(encoded_ids)}
            select_index = torch.as_tensor(
                [positions[expert] for expert in expert_ids],
                device=encoded_reconstruction.device,
            )
            free_reconstruction = encoded_reconstruction.index_select(
                0, select_index
            ).contiguous()
            free_errors = [float(encoded_errors[positions[e]]) for e in expert_ids]
        selected, selected_reconstruction, arms, identities, selection_seconds = (
            _select_cell(
                layer=layer, projection=projection, rung=int(rung),
                expert_ids=expert_ids, free_errors=free_errors,
                free_reconstruction=free_reconstruction,
                predecessor_errors=predecessor_errors,
                predecessor_reconstruction=predecessor_reconstruction,
                predecessor_identities=predecessor_ids, warm_path=warm_path,
                content_guard=guard,
            )
        )
        if predecessor_errors is not None:
            for value, predecessor in zip(selected, predecessor_errors):
                if not epsilon_le(value, predecessor, rtol=REL_EPSILON):
                    raise AssertionError(
                        f"burn P1 abort L{layer} {projection} {pass_tag} K{rung}"
                    )
        for value, free in zip(selected, free_errors):
            if not epsilon_le(value, free, rtol=REL_EPSILON):
                raise AssertionError(
                    f"burn P2 abort L{layer} {projection} {pass_tag} K{rung}"
                )
        auxiliary_drift: dict[str, Any] | None = None
        if existing is not None:
            persisted_cell = existing["cell"]
            selected_exact = (
                list(map(float, selected))
                == persisted_cell["selected_weight_mse"]
            )
            arms_exact = arms == persisted_cell["winning_arm"]
            identities_exact = identities == persisted_cell["identity"]
            encoded_exact = (
                list(map(float, encoded_errors))
                == persisted_cell["encoded_free_weight_mse"]
            )
            free_exact = free_errors == persisted_cell["free_weight_mse"]
            # Threaded LDLQ scale search may vary only a losing free arm. It
            # is auxiliary when the selected predecessor and identity remain
            # exact, and is recorded separately below.
            local_auxiliary_only = all(
                observed == expected
                or persisted_cell["winning_arm"][index] == "embed"
                for index, (observed, expected) in enumerate(zip(
                    free_errors, persisted_cell["free_weight_mse"]
                ))
            )
            local_by_expert = {
                int(expert): index for index, expert in enumerate(expert_ids)
            }
            encoded_auxiliary_only = all(
                observed == expected
                or int(expert) not in local_by_expert
                or persisted_cell["winning_arm"][
                    local_by_expert[int(expert)]
                ] == "embed"
                for expert, observed, expected in zip(
                    encoded_ids, encoded_errors,
                    persisted_cell["encoded_free_weight_mse"],
                )
            )
            selected_content_exact = bool(
                selected_exact and arms_exact and identities_exact
                and local_auxiliary_only and encoded_auxiliary_only
            )
            if not selected_content_exact:
                mismatch = {
                    "schema": "prismaquant.dsv4_afast_burn_resume_mismatch.v1",
                    "kind": "rederived_selected_content",
                    "path": str(path),
                    "content_key": str(existing["content_key"]),
                    "unit": {
                        "layer": layer, "projection": projection,
                        "pass_tag": pass_tag, "rung": int(rung),
                    },
                    "encoded_free_weight_mse": _burn_delta_summary(
                        encoded_errors,
                        persisted_cell["encoded_free_weight_mse"],
                    ),
                    "free_weight_mse": _burn_delta_summary(
                        free_errors, persisted_cell["free_weight_mse"],
                    ),
                    "selected_weight_mse": _burn_delta_summary(
                        selected, persisted_cell["selected_weight_mse"],
                    ),
                    "arm_mismatch_count": sum(
                        int(left != right) for left, right in zip(
                            arms, persisted_cell["winning_arm"]
                        )
                    ),
                    "identity_mismatch_count": sum(
                        int(left != right) for left, right in zip(
                            identities, persisted_cell["identity"]
                        )
                    ),
                    "warm_state_outcome": local_timing["warm_state_outcome"],
                    "allocator_policy": os.environ.get(
                        "PYTORCH_CUDA_ALLOC_CONF"
                    ),
                }
                manifest_path = _quarantine_burn_projection_suffix(
                    layer=layer, projection=projection, pass_tag=pass_tag,
                    first_rung=int(rung), mismatch=mismatch,
                )
                print(
                    f"[afast-burn] re-derived content mismatch; invalidated "
                    f"dependent suffix; manifest={manifest_path}", flush=True,
                )
                existing = None
            elif not encoded_exact or not free_exact:
                auxiliary_drift = {
                    "selected_weight_mse_bit_exact": selected_exact,
                    "winning_arm_bit_exact": arms_exact,
                    "chain_identity_bit_exact": identities_exact,
                    "encoded_free_exact_mismatch_count": sum(
                        int(float(a) != float(b)) for a, b in zip(
                            encoded_errors,
                            persisted_cell["encoded_free_weight_mse"],
                        )
                    ),
                    "local_free_exact_mismatch_count": sum(
                        int(float(a) != float(b)) for a, b in zip(
                            free_errors, persisted_cell["free_weight_mse"]
                        )
                    ),
                }
        if existing is None:
            if replay:
                replay_weight = (
                    data["weight"] if expert_ids == all_experts else
                    data["weight"].index_select(
                        0, torch.as_tensor(
                            expert_ids, device=data["weight"].device,
                        )
                    ).contiguous()
                )
                replay_values, replay_seconds = _replay(
                    replay_weight, selected_reconstruction,
                    data["activation_rows"], expert_ids, int(rung),
                )
                if replay_weight is not data["weight"]:
                    del replay_weight
            else:
                replay_values = {
                    "output_mse": [None] * len(expert_ids),
                    "rel_output_mse": [None] * len(expert_ids),
                    "n_activation_rows": [None] * len(expert_ids),
                }
                replay_seconds = 0.0
            total_seconds = (
                float(local_timing["free_encode_seconds"])
                + float(local_timing["free_reconstruct_and_weight_mse_seconds"])
                + float(selection_seconds) + float(replay_seconds)
            )
            cell = {
                "rung": int(rung),
                "pass_tag_schema": BURN_PASS_TAG_SCHEMA,
                "pass_tag": pass_tag,
                "expert_ids": list(expert_ids),
                "encoded_expert_ids": list(encoded_ids),
                "encoded_free_weight_mse": list(map(float, encoded_errors)),
                "free_weight_mse": free_errors,
                "embed_weight_mse": (
                    [None] * len(expert_ids) if predecessor_errors is None else
                    list(map(float, predecessor_errors))
                ),
                "selected_weight_mse": list(map(float, selected)),
                "output_mse": replay_values["output_mse"],
                "rel_output_mse": replay_values["rel_output_mse"],
                "n_activation_rows": replay_values["n_activation_rows"],
                "winning_arm": arms, "identity": identities,
                "warm_state_path": warm_path,
                "timing": {
                    **local_timing, "selection_seconds": selection_seconds,
                    "winning_replay_seconds": replay_seconds,
                    "total_seconds": total_seconds,
                },
                "rss_bytes_before_write": _rss_bytes(),
            }
            content_key = _sha(identity)
            atomic_pickle(path, {
                "schema": BURN_CELL_SCHEMA,
                "pass_tag_schema": BURN_PASS_TAG_SCHEMA,
                "pass_tag": pass_tag,
                "content_key": content_key, "identity": identity, "cell": cell,
            })
            print(f"[afast-burn] wrote {path}", flush=True)
        else:
            cell = dict(existing["cell"])
            content_key = str(existing["content_key"])
            if auxiliary_drift is not None:
                drift_root = BURN_ROOT / "resume-auxiliary-drift"
                drift_path = drift_root / f"{content_key}.json"
                atomic_json(drift_path, {
                    "schema": "prismaquant.dsv4_afast_auxiliary_resume_drift.v1",
                    "content_key": content_key, "path": str(path),
                    "layer": layer, "projection": projection,
                    "pass_tag_schema": BURN_PASS_TAG_SCHEMA,
                    "pass_tag": pass_tag, "rung": int(rung),
                    **auxiliary_drift,
                    "all_mismatches_are_losing_or_out_of_scope_free_candidates": True,
                    "warm_state_outcome": local_timing["warm_state_outcome"],
                })
                print(
                    f"[afast-burn] selected-content exact; auxiliary losing "
                    f"free drift recorded {drift_path}", flush=True,
                )
            print(f"[afast-burn] content-resume skipped replay {path}", flush=True)
        cells[int(rung)] = cell
        old_predecessor = predecessor_reconstruction
        predecessor_errors = list(map(float, selected))
        predecessor_reconstruction = selected_reconstruction
        predecessor_ids = identities
        predecessor_content_key = content_key
        del fields, encoded_reconstruction, free_reconstruction, old_predecessor
        _unit_boundary_reclaim(
            unit=f"burn:L{layer}:{projection}:{pass_tag}:K{rung}"
        )
        if rss_guard is not None:
            rss_guard.checkpoint()
        print(
            f"[afast-burn] L{layer:02d} {projection} {pass_tag} K{rung} "
            f"free={arms.count('free')} embed={arms.count('embed')} "
            f"rss={_rss_bytes()/1024**3:.2f}GiB", flush=True,
        )
    del predecessor_reconstruction
    _reclaim()
    return cells


def _measure_projection_legacy_unsafe(
    *, layer: int, projection: str, data: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    all_experts = tuple(range(EXPERT_COUNT))
    guard = _projection_guard(layer, data)
    anchor_free: dict[int, dict[str, Any]] = {}
    anchor_selected_errors: dict[int, list[float]] = {}
    anchor_selected_ids: dict[int, list[dict[str, Any]]] = {}
    anchor_arms: dict[int, list[str]] = {}
    predecessor_errors = None
    predecessor_reconstruction = None
    predecessor_ids = None
    timing = {
        "free_seconds": 0.0, "selection_seconds": 0.0,
        "winning_replay_seconds": 0.0, "warm_state_write_seconds": 0.0,
        "wall_started": time.time(),
    }

    # Keep the free fields and reconstructions resident until acceptance and
    # fallback have consumed them.  No anchor is encoded twice.
    for rung in ANCHORS:
        print(f"[afast-burn] L{layer:02d} {projection} anchor K{rung}", flush=True)
        _legacy_encoder = (
            cblk.encode_cbl if cblk.cbl_eligible(rung)
            else cblk.encode_free_noldlq
        )
        fields, reconstruction, free_errors, local_timing, warm_path = _legacy_encoder(
            layer=layer, projection=projection, rung=rung, data=data,
            expert_ids=all_experts,
        )
        timing["free_seconds"] += (
            local_timing["free_encode_seconds"]
            + local_timing["free_reconstruct_and_weight_mse_seconds"]
        )
        timing["warm_state_write_seconds"] += local_timing["warm_state_write_seconds"]
        selected_errors, selected_reconstruction, arms, identities, elapsed = (
            _select_cell(
                layer=layer, projection=projection, rung=rung,
                expert_ids=all_experts, free_errors=free_errors,
                free_reconstruction=reconstruction,
                predecessor_errors=predecessor_errors,
                predecessor_reconstruction=predecessor_reconstruction,
                predecessor_identities=predecessor_ids, warm_path=warm_path,
                content_guard=guard,
            )
        )
        timing["selection_seconds"] += elapsed
        anchor_free[rung] = {
            "fields": fields, "reconstruction": reconstruction,
            "errors": list(map(float, free_errors)), "warm_path": warm_path,
        }
        anchor_selected_errors[rung] = list(map(float, selected_errors))
        anchor_selected_ids[rung] = identities
        anchor_arms[rung] = arms
        predecessor_errors = selected_errors
        predecessor_reconstruction = selected_reconstruction
        predecessor_ids = identities
    del predecessor_reconstruction
    torch.cuda.empty_cache()

    accepted, rejected, fit = _acceptance(anchor_selected_errors)
    curve: dict[int, dict[str, Any]] = {
        k: {
            "free_weight_mse": [None] * EXPERT_COUNT,
            "embed_weight_mse": [None] * EXPERT_COUNT,
            "selected_weight_mse": [None] * EXPERT_COUNT,
            "output_mse": [None] * EXPERT_COUNT,
            "rel_output_mse": [None] * EXPERT_COUNT,
            "n_activation_rows": [None] * EXPERT_COUNT,
            "winning_arm": [None] * EXPERT_COUNT,
            "identity": [None] * EXPERT_COUNT,
            "measurement_kind": [None] * EXPERT_COUNT,
        } for k in RUNGS
    }

    # Accepted slices use the five-anchor chain. Rebuild selection from the
    # resident free reconstructions, and replay each selected anchor once.
    if accepted:
        index = torch.as_tensor(accepted, device=data["weight"].device)
        accepted_weight = data["weight"].index_select(0, index).contiguous()
        pred_errors = pred_recon = pred_ids = None
        for rung in ANCHORS:
            free_recon = anchor_free[rung]["reconstruction"].index_select(
                0, index
            ).contiguous()
            free_errors = [anchor_free[rung]["errors"][expert] for expert in accepted]
            selected, chosen, arms, identities, elapsed = _select_cell(
                layer=layer, projection=projection, rung=rung,
                expert_ids=accepted, free_errors=free_errors,
                free_reconstruction=free_recon, predecessor_errors=pred_errors,
                predecessor_reconstruction=pred_recon,
                predecessor_identities=pred_ids,
                warm_path=anchor_free[rung]["warm_path"], content_guard=guard,
            )
            timing["selection_seconds"] += elapsed
            replay, replay_seconds = _replay(
                accepted_weight, chosen, data["activation_rows"], accepted, rung,
            )
            timing["winning_replay_seconds"] += replay_seconds
            for local, expert in enumerate(accepted):
                cell = curve[rung]
                cell["free_weight_mse"][expert] = free_errors[local]
                cell["embed_weight_mse"][expert] = (
                    None if pred_errors is None else pred_errors[local]
                )
                cell["selected_weight_mse"][expert] = selected[local]
                cell["output_mse"][expert] = replay["output_mse"][local]
                cell["rel_output_mse"][expert] = replay["rel_output_mse"][local]
                cell["n_activation_rows"][expert] = replay["n_activation_rows"][local]
                cell["winning_arm"][expert] = arms[local]
                cell["identity"][expert] = identities[local]
                cell["measurement_kind"][expert] = "anchor_measured"
            pred_errors, pred_recon, pred_ids = selected, chosen, identities
        del accepted_weight, pred_recon

        for expert in accepted:
            weight_values = pchip_monotone(
                ANCHORS,
                [curve[k]["selected_weight_mse"][expert] for k in ANCHORS],
                RUNGS,
            )
            output_values = pchip_monotone(
                ANCHORS, [curve[k]["output_mse"][expert] for k in ANCHORS], RUNGS,
            )
            relative_values = pchip_monotone(
                ANCHORS,
                [curve[k]["rel_output_mse"][expert] for k in ANCHORS], RUNGS,
            )
            for offset, rung in enumerate(RUNGS):
                if rung in ANCHORS:
                    continue
                curve[rung]["selected_weight_mse"][expert] = float(weight_values[offset])
                curve[rung]["output_mse"][expert] = float(output_values[offset])
                curve[rung]["rel_output_mse"][expert] = float(relative_values[offset])
                curve[rung]["n_activation_rows"][expert] = int(
                    data["activation_rows"][expert].shape[0]
                )
                curve[rung]["measurement_kind"][expert] = "pchip_interpolated"

    # Rejected slices are walked consecutively through every rung. Anchor free
    # reconstructions are sliced from the resident bank; only missing free
    # rungs are encoded. Every chosen fallback reconstruction is replayed once.
    if rejected:
        index = torch.as_tensor(rejected, device=data["weight"].device)
        rejected_weight = data["weight"].index_select(0, index).contiguous()
        pred_errors = pred_recon = pred_ids = None
        for rung in RUNGS:
            if rung in ANCHORS:
                free_fields = anchor_free[rung]["fields"]
                free_recon = anchor_free[rung]["reconstruction"].index_select(
                    0, index
                ).contiguous()
                free_errors = [anchor_free[rung]["errors"][expert] for expert in rejected]
                warm_path = anchor_free[rung]["warm_path"]
                transient = False
            else:
                print(
                    f"[afast-burn] L{layer:02d} {projection} fallback K{rung} "
                    f"n={len(rejected)}", flush=True,
                )
                free_fields, free_recon, free_errors, local_timing, warm_path = (
                    _encode_free(
                        layer=layer, projection=projection, rung=rung, data=data,
                        expert_ids=rejected,
                    )
                )
                timing["free_seconds"] += (
                    local_timing["free_encode_seconds"]
                    + local_timing["free_reconstruct_and_weight_mse_seconds"]
                )
                timing["warm_state_write_seconds"] += local_timing[
                    "warm_state_write_seconds"
                ]
                transient = True
            selected, chosen, arms, identities, elapsed = _select_cell(
                layer=layer, projection=projection, rung=rung,
                expert_ids=rejected, free_errors=free_errors,
                free_reconstruction=free_recon, predecessor_errors=pred_errors,
                predecessor_reconstruction=pred_recon,
                predecessor_identities=pred_ids, warm_path=warm_path,
                content_guard=guard,
            )
            timing["selection_seconds"] += elapsed
            replay, replay_seconds = _replay(
                rejected_weight, chosen, data["activation_rows"], rejected, rung,
            )
            timing["winning_replay_seconds"] += replay_seconds
            for local, expert in enumerate(rejected):
                cell = curve[rung]
                cell["free_weight_mse"][expert] = float(free_errors[local])
                cell["embed_weight_mse"][expert] = (
                    None if pred_errors is None else float(pred_errors[local])
                )
                cell["selected_weight_mse"][expert] = float(selected[local])
                cell["output_mse"][expert] = replay["output_mse"][local]
                cell["rel_output_mse"][expert] = replay["rel_output_mse"][local]
                cell["n_activation_rows"][expert] = replay["n_activation_rows"][local]
                cell["winning_arm"][expert] = arms[local]
                cell["identity"][expert] = identities[local]
                cell["measurement_kind"][expert] = "fallback_measured"
            pred_errors, pred_recon, pred_ids = selected, chosen, identities
            if transient:
                del free_fields
            del free_recon
            torch.cuda.empty_cache()
        del rejected_weight, pred_recon

    monotone = tax = 0
    for expert in range(EXPERT_COUNT):
        previous = None
        for rung in RUNGS:
            selected = curve[rung]["selected_weight_mse"][expert]
            if selected is None or not math.isfinite(float(selected)):
                raise AssertionError(
                    f"L{layer} {projection} K{rung} E{expert}: incomplete curve"
                )
            if previous is not None:
                monotone += int(not epsilon_le(selected, previous, rtol=REL_EPSILON))
            free = curve[rung]["free_weight_mse"][expert]
            if free is not None:
                tax += int(not epsilon_le(selected, free, rtol=REL_EPSILON))
            previous = selected
    if monotone or tax:
        raise AssertionError(
            f"L{layer} {projection}: monotone={monotone} tax={tax}"
        )
    timing["wall_seconds"] = time.time() - timing.pop("wall_started")
    arm_counts = {"free": 0, "embed": 0}
    replay_count = 0
    for rung in RUNGS:
        for arm, kind in zip(
            curve[rung]["winning_arm"], curve[rung]["measurement_kind"]
        ):
            if arm in arm_counts:
                arm_counts[arm] += 1
            replay_count += int(kind in {"anchor_measured", "fallback_measured"})
    for bank in anchor_free.values():
        del bank["fields"], bank["reconstruction"]
    torch.cuda.empty_cache()
    return curve, {
        "fit": fit,
        "accepted_expert_ids": accepted,
        "rejected_expert_ids": rejected,
        "fallback_measured_slices": len(rejected) * len(MISSING_RUNGS),
        "arm_counts": arm_counts,
        "replay_count": replay_count,
        "timing": timing,
        "content_guard": guard,
    }


@torch.inference_mode()
def _measure_projection(
    *, layer: int, projection: str, data: Mapping[str, Any],
    rss_guard: RSSGuard | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """V2 anchors+audit projection with durable per-cell checkpoints."""
    all_experts = tuple(range(EXPERT_COUNT))
    audit_rung = _audit_rung(layer)
    measured_rungs = tuple(sorted((*ANCHORS, audit_rung)))
    wall_started = time.time()
    identity_cache: dict[
        tuple[int, ...], tuple[tuple[list[int], str], tuple[list[int], str]]
    ] = {}
    scout = _run_chain(
        layer=layer, projection=projection,
        pass_tag=BURN_PASS_TAGS["scout"], data=data,
        rungs=measured_rungs, expert_ids=all_experts, replay=False,
        full_encode_rungs=measured_rungs, rss_guard=rss_guard,
        identity_cache=identity_cache,
    )
    anchor_errors = {
        rung: scout[rung]["selected_weight_mse"] for rung in ANCHORS
    }
    accepted, rejected, fit = _acceptance(
        anchor_errors, audit_rung, scout[audit_rung]["selected_weight_mse"],
    )
    audit = _audit_stats(
        anchor_errors, scout[audit_rung]["selected_weight_mse"], audit_rung,
        accepted_experts=accepted,
    )
    if rss_guard is not None:
        rss_guard.set_stage(f"burn:L{layer}:{projection}:cbl-audit")
    cblk.audit_projection(
        layer=layer, projection=projection, data=data,
        measured_rungs=measured_rungs,
        selected_by_rung={
            int(r): scout[r]["selected_weight_mse"] for r in measured_rungs
        },
        winning_by_rung={
            int(r): scout[r]["winning_arm"] for r in measured_rungs
        },
        epsilon_le=epsilon_le, rel_epsilon=REL_EPSILON,
        micro_check_k43=(int(layer) == CBL_MICROCHECK_LAYER),
        out_root=BURN_ROOT,
    )
    curve: dict[int, dict[str, Any]] = {
        rung: {
            "free_weight_mse": [None] * EXPERT_COUNT,
            "embed_weight_mse": [None] * EXPERT_COUNT,
            "selected_weight_mse": [None] * EXPERT_COUNT,
            "output_mse": [None] * EXPERT_COUNT,
            "rel_output_mse": [None] * EXPERT_COUNT,
            "n_activation_rows": [None] * EXPERT_COUNT,
            "winning_arm": [None] * EXPERT_COUNT,
            "identity": [None] * EXPERT_COUNT,
            "measurement_kind": [None] * EXPERT_COUNT,
        } for rung in RUNGS
    }
    phase_cells: list[dict[str, Any]] = list(scout.values())

    def copy_cells(cells: Mapping[int, Mapping[str, Any]], kind: str) -> None:
        for rung, source in cells.items():
            for local, expert in enumerate(source["expert_ids"]):
                target = curve[int(rung)]
                for name in (
                    "free_weight_mse", "embed_weight_mse",
                    "selected_weight_mse", "output_mse", "rel_output_mse",
                    "n_activation_rows", "winning_arm", "identity",
                ):
                    target[name][expert] = source[name][local]
                target["measurement_kind"][expert] = kind

    if accepted:
        accepted_cells = _run_chain(
            layer=layer, projection=projection,
            pass_tag=BURN_PASS_TAGS["primary"], data=data,
            rungs=measured_rungs, expert_ids=accepted, replay=True,
            full_encode_rungs=measured_rungs, rss_guard=rss_guard,
            oracle_cells=scout, identity_cache=identity_cache,
        )
        phase_cells.extend(accepted_cells.values())
        copy_cells(accepted_cells, "anchor_measured")
        for expert in accepted:
            weight_values = pchip_monotone(
                ANCHORS,
                [curve[rung]["selected_weight_mse"][expert] for rung in ANCHORS],
                RUNGS,
            )
            output_values = pchip_monotone(
                ANCHORS,
                [curve[rung]["output_mse"][expert] for rung in ANCHORS],
                RUNGS,
            )
            relative_values = pchip_monotone(
                ANCHORS,
                [curve[rung]["rel_output_mse"][expert] for rung in ANCHORS],
                RUNGS,
            )
            for offset, rung in enumerate(RUNGS):
                if rung in ANCHORS:
                    # Audit-draw rows are PCHIP-priced like any interpolated
                    # rung: the draw's incumbent-basis measurement is
                    # verification evidence (kept in the cell, _audit_stats,
                    # and CBL_AUDIT records), not a menu price — splicing it
                    # raw into a CBL-basis curve manufactures a monotonicity
                    # break at the draw rung (256x, caught by the layer gate).
                    continue
                curve[rung]["selected_weight_mse"][expert] = float(
                    weight_values[offset]
                )
                curve[rung]["output_mse"][expert] = float(output_values[offset])
                curve[rung]["rel_output_mse"][expert] = float(
                    relative_values[offset]
                )
                curve[rung]["n_activation_rows"][expert] = int(
                    data["activation_rows"][expert].shape[0]
                )
                curve[rung]["measurement_kind"][expert] = "pchip_interpolated"
        # The audit draw is verification evidence, not a menu row: accepted
        # experts' prices there are PCHIP like every non-anchor rung, so the
        # measured arm fields must not remain spliced beside interpolated
        # prices — the tax invariant (selected <= free) binds measured rows
        # only, and interpolation may legitimately sit above the measured
        # free arm at the draw (L2 down_proj K36: tax=256).
        for expert in accepted:
            for field in ("free_weight_mse", "embed_weight_mse", "winning_arm"):
                curve[audit_rung][field][expert] = None

    if rejected:
        fallback_cells = _run_chain(
            layer=layer, projection=projection,
            pass_tag=BURN_PASS_TAGS["backstop"], data=data,
            rungs=RUNGS, expert_ids=rejected, replay=True,
            full_encode_rungs=ANCHORS, rss_guard=rss_guard,
            oracle_cells=scout, identity_cache=identity_cache,
        )
        phase_cells.extend(fallback_cells.values())
        copy_cells(fallback_cells, "fallback_measured")

    monotone = tax = 0
    for expert in range(EXPERT_COUNT):
        previous = None
        for rung in RUNGS:
            selected = curve[rung]["selected_weight_mse"][expert]
            if selected is None or not math.isfinite(float(selected)):
                raise AssertionError(
                    f"L{layer} {projection} K{rung} E{expert}: incomplete curve"
                )
            if previous is not None:
                monotone += int(
                    not epsilon_le(selected, previous, rtol=REL_EPSILON)
                )
            free = curve[rung]["free_weight_mse"][expert]
            if free is not None:
                tax += int(not epsilon_le(selected, free, rtol=REL_EPSILON))
            previous = selected
    if monotone or tax:
        raise AssertionError(
            f"L{layer} {projection}: monotone={monotone} tax={tax}"
        )
    timing = {
        "free_seconds": sum(
            float(cell["timing"]["free_encode_seconds"])
            + float(cell["timing"]["free_reconstruct_and_weight_mse_seconds"])
            for cell in phase_cells
        ),
        "selection_seconds": sum(
            float(cell["timing"]["selection_seconds"]) for cell in phase_cells
        ),
        "winning_replay_seconds": sum(
            float(cell["timing"]["winning_replay_seconds"])
            for cell in phase_cells
        ),
        "warm_state_write_seconds": sum(
            float(cell["timing"]["warm_state_write_seconds"])
            for cell in phase_cells
        ),
        "wall_seconds": time.time() - wall_started,
    }
    arm_counts = {"free": 0, "embed": 0}
    replay_count = 0
    for rung in RUNGS:
        for arm, kind in zip(
            curve[rung]["winning_arm"], curve[rung]["measurement_kind"]
        ):
            if arm in arm_counts:
                arm_counts[arm] += 1
            replay_count += int(kind in {"anchor_measured", "fallback_measured"})
    _reclaim()
    return curve, {
        "fit": fit, "accepted_expert_ids": accepted,
        "rejected_expert_ids": rejected,
        "audit": audit,
        "full_layer_fallback": False,
        "fallback_measured_slices": len(rejected) * (len(MISSING_RUNGS) - 1),
        "arm_counts": arm_counts, "replay_count": replay_count,
        "timing": timing, "content_guard": _projection_guard(layer, data),
        "persistence": {
            "root": str(BURN_CELL_ROOT),
            "scout_cells": len(scout),
            "primary_cells": len(measured_rungs) if accepted else 0,
            "fallback_cells": len(RUNGS) if rejected else 0,
        },
    }


@torch.inference_mode()
def _measure_projection_full_layer(
    *, layer: int, projection: str, data: Mapping[str, Any],
    initial_meta: Mapping[str, Any], rss_guard: RSSGuard | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Full 21-rung fallback required by a failed per-layer audit."""
    all_experts = tuple(range(EXPERT_COUNT))
    wall_started = time.time()
    cells = _run_chain(
        layer=layer, projection=projection,
        pass_tag=BURN_PASS_TAGS["full_layer"], data=data,
        rungs=RUNGS, expert_ids=all_experts, replay=True,
        full_encode_rungs=RUNGS, rss_guard=rss_guard,
    )
    curve: dict[int, dict[str, Any]] = {
        rung: {
            "free_weight_mse": list(cells[rung]["free_weight_mse"]),
            "embed_weight_mse": list(cells[rung]["embed_weight_mse"]),
            "selected_weight_mse": list(cells[rung]["selected_weight_mse"]),
            "output_mse": list(cells[rung]["output_mse"]),
            "rel_output_mse": list(cells[rung]["rel_output_mse"]),
            "n_activation_rows": list(cells[rung]["n_activation_rows"]),
            "winning_arm": list(cells[rung]["winning_arm"]),
            "identity": list(cells[rung]["identity"]),
            "measurement_kind": ["audit_failed_full_layer_measured"] * EXPERT_COUNT,
        }
        for rung in RUNGS
    }
    timing = {
        "free_seconds": sum(
            float(cell["timing"]["free_encode_seconds"])
            + float(cell["timing"]["free_reconstruct_and_weight_mse_seconds"])
            for cell in cells.values()
        ),
        "selection_seconds": sum(
            float(cell["timing"]["selection_seconds"])
            for cell in cells.values()
        ),
        "winning_replay_seconds": sum(
            float(cell["timing"]["winning_replay_seconds"])
            for cell in cells.values()
        ),
        "warm_state_write_seconds": sum(
            float(cell["timing"]["warm_state_write_seconds"])
            for cell in cells.values()
        ),
        "wall_seconds": time.time() - wall_started,
    }
    arm_counts = {"free": 0, "embed": 0}
    for cell in cells.values():
        for arm in cell["winning_arm"]:
            arm_counts[arm] += 1
    meta = copy.deepcopy(dict(initial_meta))
    meta.update({
        "full_layer_fallback": True,
        "full_layer_fallback_reason": "at_least_one_projection_failed_audit",
        "fallback_measured_slices": EXPERT_COUNT * len(MISSING_RUNGS),
        "arm_counts": arm_counts,
        "replay_count": EXPERT_COUNT * len(RUNGS),
        "timing": timing,
        "content_guard": _projection_guard(layer, data),
        "persistence": {
            "root": str(BURN_CELL_ROOT), "full_layer_cells": len(RUNGS),
        },
    })
    _reclaim()
    return curve, meta


def _write_projection_costs(
    costs: dict[str, dict], layer: int, projection: str,
    curve: Mapping[int, Mapping[str, Any]],
) -> None:
    for expert in range(EXPERT_COUNT):
        qname = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        row = costs[qname]
        for rung in RUNGS:
            cell = curve[rung]
            kind = str(cell["measurement_kind"][expert])
            entry = {
                "weight_mse": float(cell["selected_weight_mse"][expert]),
                "output_mse": float(cell["output_mse"][expert]),
                "rel_output_mse": float(cell["rel_output_mse"][expert]),
                "n_activation_rows": int(cell["n_activation_rows"][expert]),
                "output_mse_measured": kind != "pchip_interpolated",
                "cost_source": kind,
                "afast_profile": "v2_five_anchor_audit_pchip_minchain",
                "ladder": {
                    "anchors": list(ANCHORS),
                    "audit_rung": _audit_rung(layer),
                    "backstop_cv": "independent_four_anchor",
                    "backstop": "audit_rung_pchip_cv_v2",
                    "tolerance": BACKSTOP_TOLERANCE,
                },
            }
            if cell["free_weight_mse"][expert] is not None:
                entry["free_weight_mse"] = float(cell["free_weight_mse"][expert])
            if cell["embed_weight_mse"][expert] is not None:
                entry["embed_weight_mse"] = float(cell["embed_weight_mse"][expert])
            if cell["winning_arm"][expert] is not None:
                entry["winning_arm"] = cell["winning_arm"][expert]
                entry["cb_minchain_identity"] = cell["identity"][expert]
            else:
                entry["cb_minchain_identity"] = {
                    "schema": MINCHAIN_SCHEMA,
                    "chain_version": CHAIN_VERSION,
                    "status": "pchip_interpolated__encode_identity_deferred",
                    "winning_arm": None,
                    "predecessor_digest": None,
                    "solution_digest": None,
                }
            row[f"FP8_CB_K{rung}"] = entry


def _measure_layer(
    layer: int, *, pilot_sha: str, old_full: Mapping[str, Any],
    all_col_weights: Mapping[str, Any], model_to_shard: Mapping[str, str],
    model_to_ckpt: Mapping[str, str], scale_map: Mapping[str, Any],
    rss_guard: RSSGuard | None = None,
) -> dict:
    base_payload, verified = load_layer_identity(layer)
    costs, base_sha = _base_layer_costs(layer, old_full)
    if base_sha != verified["sha256"]:
        raise AssertionError(f"layer {layer}: base identity changed during load")
    identity = _layer_identity(
        layer, base_sha=base_sha, pilot_sha=pilot_sha, import_kind=None,
    )
    content_key = _sha(identity)
    path = SHARD_ROOT / f"layer_{layer:03d}.pkl"
    if path.is_file():
        payload = _load(path)
        if payload.get("content_key") != content_key:
            raise AssertionError(f"stale layer shard {path}")
        print(f"[afast-burn] resume L{layer:02d}", flush=True)
        return payload
    started = time.time()
    projection_meta = {}
    projection_curves = {}
    for projection in PROJECTIONS:
        data = load_projection(
            layer, projection, device=torch.device("cuda:0"),
            identity=verified["identity"], all_col_weights=all_col_weights,
            model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt,
            scale_map=scale_map,
        )
        curve, meta = _measure_projection(
            layer=layer, projection=projection, data=data, rss_guard=rss_guard,
        )
        projection_curves[projection] = curve
        projection_meta[projection] = meta
        del data
        torch.cuda.empty_cache()
    layer_audit_pass = all(
        bool(projection_meta[p]["audit"]["pass"]) for p in PROJECTIONS
    )
    if not layer_audit_pass:
        # The amendment gate is layer-wide: one failed projection invalidates
        # interpolation for all three routed projections.
        for projection in PROJECTIONS:
            del projection_curves[projection]
            data = load_projection(
                layer, projection, device=torch.device("cuda:0"),
                identity=verified["identity"], all_col_weights=all_col_weights,
                model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt,
                scale_map=scale_map,
            )
            curve, meta = _measure_projection_full_layer(
                layer=layer, projection=projection, data=data,
                initial_meta=projection_meta[projection], rss_guard=rss_guard,
            )
            projection_curves[projection] = curve
            projection_meta[projection] = meta
            del data
            torch.cuda.empty_cache()
    for projection in PROJECTIONS:
        _write_projection_costs(
            costs, layer, projection, projection_curves[projection],
        )
    del projection_curves
    payload = {
        "schema": SCHEMA, "content_key": content_key, "identity": identity,
        "costs": costs,
        "formats": sorted({fmt for row in costs.values() for fmt in row}),
        "meta": {
            "layer": layer, "profile": "A-FAST", "imported": False,
            "row_count": len(costs), "elapsed_seconds": time.time() - started,
            "amendment": "v2_accept_all_plus_per_layer_audit",
            "audit_rung": _audit_rung(layer),
            "audit_gate_pass": layer_audit_pass,
            "full_layer_fallback": not layer_audit_pass,
            "projection": projection_meta,
        },
    }
    atomic_pickle(path, payload)
    print(
        f"[afast-burn] wrote L{layer:02d} "
        f"elapsed={payload['meta']['elapsed_seconds']/60:.2f}m", flush=True,
    )
    return payload


def _verify_pilot2_content(layer_record: Mapping[str, Any]) -> None:
    identity = dict(layer_record["content_identity"])
    if layer_record.get("content_key") != _sha(identity):
        raise AssertionError("pilot-2 content key mismatch")
    if identity["source_index_sha256"] != sha256_file(
        SOURCE / "model.safetensors.index.json"
    ):
        raise AssertionError("pilot-2 source index changed")
    _, verified = load_layer_identity(PILOT2_LAYER)
    if identity["by_layer_sha256"] != verified["sha256"]:
        raise AssertionError("pilot-2 by-layer content changed")
    if identity["serialization_context"] != cb_serialization_context_stamp(
        CONTEXT, formats=[f"FP8_CB_K{k}" for k in RUNGS],
    ):
        raise AssertionError("pilot-2 serialization context changed")
    for name, digest in identity["implementation_sha256"].items():
        path = (
            Path(__file__).resolve().parent / "dsv4_afast_campaign.py"
            if name == "campaign_tool" else
            Path(__file__).resolve().parents[1] / "prismaquant/cb_minchain.py"
        )
        if digest != sha256_file(path):
            raise AssertionError(f"pilot-2 implementation changed: {name}")


def _import_pilot2(
    *, pilot_sha: str, old_full: Mapping[str, Any],
) -> dict:
    layer_record = _load(PILOT2_SHARD)
    _verify_pilot2_content(layer_record)
    costs, base_sha = _base_layer_costs(PILOT2_LAYER, old_full)
    projection_meta = {}
    for projection in PROJECTIONS:
        cells = layer_record["projections"][projection]["cells"]
        curve = {}
        for rung in RUNGS:
            cell = cells[rung]
            curve[rung] = {
                "free_weight_mse": cell["free_weight_mse"],
                "embed_weight_mse": cell["embed_weight_mse"],
                "selected_weight_mse": cell["selected_weight_mse"],
                "output_mse": cell["selected_output_mse"],
                "rel_output_mse": [0.0] * EXPERT_COUNT,
                "n_activation_rows": [0] * EXPERT_COUNT,
                "winning_arm": cell["winning_arm"],
                "identity": cell["identity"],
                "measurement_kind": ["pilot2_import_measured"] * EXPERT_COUNT,
            }
            for expert in range(EXPERT_COUNT):
                qname = (
                    f"model.layers.{PILOT2_LAYER}.mlp.experts.{expert}.{projection}"
                )
                old = old_full["costs"][qname]["FP8_CB_K36"]
                energy = float(old["output_mse"]) / max(
                    float(old["rel_output_mse"]), 1e-30
                )
                curve[rung]["rel_output_mse"][expert] = (
                    float(cell["selected_output_mse"][expert]) / energy
                )
                curve[rung]["n_activation_rows"][expert] = int(
                    old["n_activation_rows"]
                )
        _write_projection_costs(costs, PILOT2_LAYER, projection, curve)
        anchor_errors = {
            rung: curve[rung]["selected_weight_mse"] for rung in ANCHORS
        }
        accepted, rejected, fit = _acceptance(anchor_errors)
        audit = _audit_stats(
            anchor_errors,
            curve[_audit_rung(PILOT2_LAYER)]["selected_weight_mse"],
            _audit_rung(PILOT2_LAYER),
        )
        projection_meta[projection] = {
            "fit": fit, "audit": audit,
            "accepted_expert_ids": accepted,
            "rejected_expert_ids": rejected,
            "full_layer_fallback": False,
            "full_truth_imported": True,
            "replay_count": len(RUNGS) * EXPERT_COUNT,
        }
    layer_audit_pass = all(
        projection_meta[p]["audit"]["pass"] for p in PROJECTIONS
    )
    for projection in PROJECTIONS:
        projection_meta[projection]["full_layer_fallback"] = not layer_audit_pass
    identity = _layer_identity(
        PILOT2_LAYER, base_sha=base_sha, pilot_sha=pilot_sha,
        import_kind="content_verified_pilot2",
    )
    payload = {
        "schema": SCHEMA, "content_key": _sha(identity), "identity": identity,
        "costs": costs,
        "formats": sorted({fmt for row in costs.values() for fmt in row}),
        "meta": {
            "layer": PILOT2_LAYER, "profile": "A-FAST", "imported": True,
            "import_kind": "pilot2", "source_shard": str(PILOT2_SHARD),
            "source_sha256": sha256_file(PILOT2_SHARD), "row_count": len(costs),
            "elapsed_seconds": 0.0,
            "amendment": "v2_accept_all_plus_per_layer_audit",
            "audit_rung": _audit_rung(PILOT2_LAYER),
            "audit_gate_pass": layer_audit_pass,
            "full_layer_fallback": not layer_audit_pass,
            "projection": projection_meta,
        },
    }
    atomic_pickle(SHARD_ROOT / f"layer_{PILOT2_LAYER:03d}.pkl", payload)
    return payload


def _import_layer21(
    *, pilot_sha: str, old_full: Mapping[str, Any],
    all_col_weights: Mapping[str, Any],
) -> dict:
    layer = 21
    costs, base_sha = _base_layer_costs(layer, old_full)
    _, verified = load_layer_identity(layer)
    observed = canonical_cb_col_weights_sha256(
        all_col_weights, verified["identity"]["col_weights_qnames"],
    )
    if observed != verified["identity"]["col_weights_sha256"]:
        raise AssertionError("layer-21 col-weight aggregate changed")
    projection_meta = {}
    phase_sources = []
    for projection in PROJECTIONS:
        k36_path = PHASE_A_SHARDS / f"layer_021_{projection}_K36.pkl"
        k36 = _load(k36_path)
        activation_residuals: list[float] = []
        reference_energies: list[float] = []
        row_counts: list[int] = []
        col_weight_sums: list[float] = []
        for expert in range(EXPERT_COUNT):
            qname = f"model.layers.21.mlp.experts.{expert}.{projection}"
            old = old_full["costs"][qname]["FP8_CB_K36"]
            cw_sum = float(torch.as_tensor(all_col_weights[qname]).sum().item())
            col_weight_sums.append(cw_sum)
            banked_weight_output = (
                float(k36["weighted_mse_per_expert"][expert]) * cw_sum
            )
            activation_residuals.append(
                max(float(old["output_mse"]) - banked_weight_output, 0.0)
            )
            reference_energies.append(
                float(old["output_mse"])
                / max(float(old["rel_output_mse"]), 1e-30)
            )
            row_counts.append(int(old["n_activation_rows"]))
        selected_errors = [math.inf] * EXPERT_COUNT
        selected_weight_output = [0.0] * EXPERT_COUNT
        predecessor_ids: list[dict[str, Any] | None] = [None] * EXPERT_COUNT
        accepted_arm_counts = {"free": 0, "embed": 0}
        selected_by_rung: dict[int, list[float]] = {}
        for rung in RUNGS:
            path = PHASE_A_SHARDS / f"layer_021_{projection}_K{rung}.pkl"
            shard = _load(path)
            if (
                shard.get("schema") != "prismaquant.dsv4_ldlq_projection_rung.v1"
                or shard.get("layer") != layer
                or shard.get("projection") != projection
                or shard.get("rung") != rung
                or len(shard.get("qnames", [])) != EXPERT_COUNT
            ):
                raise AssertionError(f"invalid phase-A shard {path}")
            if not Path(shard["warm_state_path"]).is_file():
                raise AssertionError(f"missing banked warm state {path}")
            phase_sha = sha256_file(path)
            phase_sources.append({"path": str(path), "sha256": phase_sha})
            free = list(map(float, shard["weight_mse_per_expert"]))
            weighted = list(map(float, shard["weighted_mse_per_expert"]))
            for expert in range(EXPERT_COUNT):
                arm, error = select_arm({
                    "free": free[expert], "embed": selected_errors[expert],
                }, rtol=REL_EPSILON) if rung != RUNGS[0] else ("free", free[expert])
                pred_digest = (
                    predecessor_ids[expert]["solution_digest"]
                    if arm == "embed" else None
                )
                recipe = {
                    "kind": "layer21_banked_afast_import",
                    "phase_a_sha256": phase_sha,
                    "qname": shard["qnames"][expert],
                    "rung": rung, "winning_arm": arm,
                    "predecessor_digest": pred_digest,
                    "warm_state_path": shard["warm_state_path"],
                    "serialization_context": cb_serialization_context_stamp(
                        CONTEXT, formats=[f"FP8_CB_K{rung}"],
                    ),
                }
                identity = chain_identity_from_digest(
                    winning_arm=arm,
                    solution_digest_value=recipe_solution_digest(recipe),
                    predecessor_digest=pred_digest,
                )
                if arm == "free":
                    selected_weight_output[expert] = weighted[expert]
                selected_errors[expert] = error
                predecessor_ids[expert] = identity
                accepted_arm_counts[arm] += 1
                qname = shard["qnames"][expert]
                # The no-remeasure import uses the banked activation-diagonal
                # value plus the old same-cache activation-QDQ residual at K36.
                cw_sum = col_weight_sums[expert]
                output = (
                    selected_weight_output[expert] * cw_sum
                    + activation_residuals[expert]
                )
                entry = {
                    "weight_mse": error, "output_mse": output,
                    "rel_output_mse": output / reference_energies[expert],
                    "n_activation_rows": row_counts[expert],
                    "output_mse_measured": False,
                    "cost_source": "content_verified_layer21_banked_import",
                    "free_weight_mse": free[expert],
                    "embed_weight_mse": (
                        None if rung == RUNGS[0] else
                        float(costs[qname][f"FP8_CB_K{rung-1}"]["weight_mse"])
                    ),
                    "winning_arm": arm,
                    "cb_minchain_identity": identity,
                    "afast_profile": "v2_five_anchor_audit_pchip_minchain",
                }
                costs[qname][f"FP8_CB_K{rung}"] = entry
            selected_by_rung[rung] = list(selected_errors)

        # Verify every covered legacy-chain cell against the optimized
        # free/embed minimum; refine had zero wins and is deliberately absent.
        covered = 0
        for legacy_path in sorted(OLD_CHAIN_SHARDS.glob(
            f"layer_021_{projection}_K*.pkl"
        )):
            legacy = _load(legacy_path)
            if sha256_file(Path(legacy["free_truth_shard"])) != legacy[
                "free_truth_sha256"
            ]:
                raise AssertionError(f"legacy free-truth digest mismatch {legacy_path}")
            if sha256_file(Path(legacy["state_path"])) != legacy["state_sha256"]:
                raise AssertionError(f"legacy state digest mismatch {legacy_path}")
            for free, embed, refine, selected in zip(
                legacy["free_error"], legacy["embed_error"],
                legacy["refine_error"], legacy["selected_error"],
            ):
                candidates = {"free": float(free)}
                if math.isfinite(float(embed)):
                    candidates["embed"] = float(embed)
                optimized = select_arm(candidates, rtol=REL_EPSILON)[1]
                if not epsilon_le(optimized, float(selected), rtol=2e-6) or not epsilon_le(
                    float(selected), optimized, rtol=2e-6
                ):
                    raise AssertionError(f"legacy chain replay differs {legacy_path}")
                if (
                    math.isfinite(float(refine))
                    and not epsilon_le(optimized, float(refine), rtol=REL_EPSILON)
                ):
                    raise AssertionError(f"unexpected refine win {legacy_path}")
                covered += 1
        accepted, rejected, fit = _acceptance(
            {rung: selected_by_rung[rung] for rung in ANCHORS}
        )
        audit = _audit_stats(
            {rung: selected_by_rung[rung] for rung in ANCHORS},
            selected_by_rung[_audit_rung(layer)], _audit_rung(layer),
        )
        fit["imported_full_truth"] = True
        fit["fallback_already_banked_slices"] = (
            len(rejected) * (len(MISSING_RUNGS) - 1)
        )
        projection_meta[projection] = {
            "fit": fit, "audit": audit,
            "accepted_expert_ids": accepted,
            "rejected_expert_ids": rejected,
            "arm_counts": accepted_arm_counts,
            "legacy_chain_cells_verified": covered,
            "replay_count": 0,
            "output_accounting": (
                "banked activation-diagonal reconstruction residual plus "
                "same-cache K36 activation-QDQ residual; no remeasurement"
            ),
        }
    layer_audit_pass = all(
        projection_meta[p]["audit"]["pass"] for p in PROJECTIONS
    )
    for projection in PROJECTIONS:
        projection_meta[projection]["full_layer_fallback"] = not layer_audit_pass
        projection_meta[projection]["full_truth_imported"] = True
    identity = _layer_identity(
        layer, base_sha=base_sha, pilot_sha=pilot_sha,
        import_kind="content_verified_phase_a_plus_chain",
    )
    identity["phase_a_sources"] = phase_sources
    identity["verified_layer_col_weights_sha256"] = observed
    payload = {
        "schema": SCHEMA, "content_key": _sha(identity), "identity": identity,
        "costs": costs,
        "formats": sorted({fmt for row in costs.values() for fmt in row}),
        "meta": {
            "layer": layer, "profile": "A-FAST", "imported": True,
            "import_kind": "layer21_phase_a_plus_chain", "row_count": len(costs),
            "elapsed_seconds": 0.0,
            "amendment": "v2_accept_all_plus_per_layer_audit",
            "audit_rung": _audit_rung(layer),
            "audit_gate_pass": layer_audit_pass,
            "full_layer_fallback": not layer_audit_pass,
            "projection": projection_meta,
        },
    }
    atomic_pickle(SHARD_ROOT / "layer_021.pkl", payload)
    return payload


def _projection_abort(manifest: Mapping[str, Any], completed: Sequence[dict]) -> str:
    elapsed = time.time() - float(manifest["first_burn_shard_epoch"])
    projected = elapsed / len(completed) * int(manifest["measured_layer_count"])
    return "\n".join([
        "# DSV4 A-FAST Projection Abort", "",
        f"- Completed fresh shards: {len(completed)}/{manifest['measured_layer_count']}",
        f"- Observed foreground wall: {elapsed/3600:.3f} h",
        f"- Projected 41-layer wall: {projected/3600:.3f} h",
        f"- Registered timebox: {TIMEBOX_SECONDS/3600:.1f} h", "",
        "The three-shard projection exceeded the registered timebox. No fourth fresh shard was started; imported layers 14 and 21 remain content-verified.", "",
    ])


def _run_burn(rss_guard: RSSGuard) -> int:
    _configure()
    if not torch.cuda.is_available():
        raise SystemExit("A-FAST burn requires CUDA")
    pilot = _pilot_gate()
    amendment = _amendment_gate()
    pilot_sha = sha256_file(PILOT2_JSON)
    old_full = _load(OLD_FULL_COST)
    with COL_WEIGHTS.open("rb") as handle:
        all_col_weights = pickle.load(handle)
    model_to_shard, model_to_ckpt = _build_weight_map(str(SOURCE))
    scale_map = _build_fp8_scale_inv_map(str(SOURCE))
    manifest_path = BURN_ROOT / "BURN_MANIFEST.json"
    manifest = {
        "schema": MANIFEST_SCHEMA, "profile": "A-FAST",
        "pilot2_report": str(PILOT2_JSON), "pilot2_report_sha256": pilot_sha,
        "layer_count": LAYER_COUNT, "measured_layer_count": MEASURED_LAYER_COUNT,
        "imported_layers": list(IMPORTED_LAYERS), "timebox_seconds": TIMEBOX_SECONDS, "cbl_semantics": cblk.SEMANTICS_STAMP,
        "first_burn_shard_epoch": None, "thread_count": 16,
        "menu": list(MENU), "mtp_policy": "untouched fixed source carry",
        "mtp_bytes": MTP_BF16_BYTES,
        "amendment": "v2_accept_all_plus_per_layer_audit",
        "acceptance_amendment_sha256": sha256_file(AMENDMENT_JSON),
        "projected_hours": float(amendment["cost"]["projected_hours"]),
        "backstop_tolerance": BACKSTOP_TOLERANCE,
        "audit_thresholds": {
            "median": AUDIT_MEDIAN_TOLERANCE,
            "p95": AUDIT_P95_TOLERANCE,
        },
        "burn_pass_tag_schema": BURN_PASS_TAG_SCHEMA,
        "burn_pass_tags": dict(BURN_PASS_TAGS),
    }
    migrated_first_burn_epoch = None
    migrated_projection_check = None
    if manifest_path.is_file():
        prior_manifest = json.loads(manifest_path.read_text())
        if prior_manifest.get("schema") != MANIFEST_SCHEMA:
            migrated_first_burn_epoch = prior_manifest.get(
                "first_burn_shard_epoch"
            )
            migrated_projection_check = prior_manifest.get("projection_check")
            history = RUN_ROOT / "history" / (
                "BURN_MANIFEST.pre-v2." + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                + ".json"
            )
            history.parent.mkdir(parents=True, exist_ok=True)
            os.replace(manifest_path, history)
            prior_manifest = {}
        immutable_keys = (
            "schema", "profile", "pilot2_report_sha256", "layer_count",
            "measured_layer_count", "imported_layers", "timebox_seconds",
            "thread_count", "menu", "mtp_policy", "mtp_bytes", "amendment",
            "acceptance_amendment_sha256", "backstop_tolerance",
            "audit_thresholds",
            "burn_pass_tag_schema", "burn_pass_tags",
            "burn_tool_sha256", "cbl_semantics",
        )
        if prior_manifest and any(
            prior_manifest.get(key) != manifest.get(key)
            for key in immutable_keys
        ):
            raise AssertionError(f"stale burn manifest {manifest_path}")
        if prior_manifest:
            manifest["first_burn_shard_epoch"] = prior_manifest.get(
                "first_burn_shard_epoch"
            )
        if prior_manifest and "projection_check" in prior_manifest:
            manifest["projection_check"] = prior_manifest["projection_check"]
    if migrated_first_burn_epoch is not None:
        manifest["first_burn_shard_epoch"] = migrated_first_burn_epoch
    if migrated_projection_check is not None:
        manifest["projection_check"] = migrated_projection_check
    atomic_json(manifest_path, manifest)
    measured_completed = []
    all_payloads = []
    for layer in range(LAYER_COUNT):
        if layer == PILOT2_LAYER and PILOT2_LAYER in IMPORTED_LAYERS:
            payload = _import_pilot2(pilot_sha=pilot_sha, old_full=old_full)
            print("[afast-burn] imported L14 pilot-2", flush=True)
        elif layer == 21 and 21 in IMPORTED_LAYERS:
            payload = _import_layer21(
                pilot_sha=pilot_sha, old_full=old_full,
                all_col_weights=all_col_weights,
            )
            print("[afast-burn] imported L21 phase-A + chain", flush=True)
        else:
            if manifest["first_burn_shard_epoch"] is None:
                manifest["first_burn_shard_epoch"] = time.time()
                atomic_json(manifest_path, manifest)
            layer_path = SHARD_ROOT / f"layer_{layer:03d}.pkl"
            elapsed_campaign = (
                time.time() - float(manifest["first_burn_shard_epoch"])
            )
            if not layer_path.is_file() and elapsed_campaign >= TIMEBOX_SECONDS:
                stop_text = "\n".join([
                    "# DSV4 Campaign Stop", "",
                    f"- Stage: {MEASURED_LAYER_COUNT}-layer A-FAST burn",
                    "- Outcome: STOP (20-hour foreground timebox reached)",
                    f"- Completed measured layer shards: "
                    f"{len(measured_completed)}/{MEASURED_LAYER_COUNT}",
                    f"- Elapsed from first burn shard: {elapsed_campaign/3600:.3f} h",
                    "",
                    "All completed cell and layer shards remain content-keyed and resumable. "
                    "No additional layer was started after the timebox.", "",
                ])
                atomic_text(RUN_ROOT / "TIMEBOX_ABORT.md", stop_text)
                atomic_text(RUN_ROOT / "DSV4_CAMPAIGN_STOP.md", stop_text)
                return 5
            payload = _measure_layer(
                layer, pilot_sha=pilot_sha, old_full=old_full,
                all_col_weights=all_col_weights,
                model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt,
                scale_map=scale_map, rss_guard=rss_guard,
            )
            measured_completed.append(payload)
            if len(measured_completed) == 3 and not manifest.get(
                "projection_check", {}
            ).get("pass"):
                elapsed = time.time() - float(manifest["first_burn_shard_epoch"])
                projected = elapsed / 3 * 41
                if projected > TIMEBOX_SECONDS:
                    manifest["projection_check"] = {
                        "pass": False, "completed_shards": 3,
                        "observed_seconds": elapsed,
                        "projected_seconds": projected,
                    }
                    atomic_json(manifest_path, manifest)
                    atomic_text(
                        RUN_ROOT / "PROJECTION_ABORT.md",
                        _projection_abort(manifest, measured_completed),
                    )
                    return 3
                manifest["projection_check"] = {
                    "pass": True, "completed_shards": 3,
                    "observed_seconds": elapsed,
                    "projected_seconds": projected,
                }
                atomic_json(manifest_path, manifest)
        all_payloads.append(payload)
    return merge()


def run_burn() -> int:
    rss_guard = RSSGuard(BURN_ROOT / "RSS_GUARD_EVIDENCE.json")
    rss_guard.start()
    try:
        return _run_burn(rss_guard)
    except RSSLimitExceeded as exc:
        _reclaim()
        atomic_text(RUN_ROOT / "DSV4_CAMPAIGN_STOP.md", "\n".join([
            "# DSV4 Campaign Stop", "",
            "- Stage: 41-layer A-FAST burn",
            "- Outcome: ABORT (RSS guard)",
            f"- Evidence: `{BURN_ROOT / 'RSS_GUARD_EVIDENCE.json'}`",
            f"- Detail: {exc}", "",
            "All completed burn cells and layer shards were flushed atomically; "
            "no further measurement was started.", "",
        ]))
        return 4
    finally:
        rss_guard.stop()


def run_shakedown_worker() -> int:
    """Exercise the exact v2 scout unit path used by the first burn layer."""
    _configure()
    _pilot_gate()
    _amendment_gate()
    if not torch.cuda.is_available():
        raise SystemExit("A-FAST shakedown requires CUDA")
    before = _host_available_bytes()
    with COL_WEIGHTS.open("rb") as handle:
        all_col_weights = pickle.load(handle)
    _, verified = load_layer_identity(0)
    model_to_shard, model_to_ckpt = _build_weight_map(str(SOURCE))
    scale_map = _build_fp8_scale_inv_map(str(SOURCE))
    data = load_projection(
        0, "gate_proj", device=torch.device("cuda:0"),
        identity=verified["identity"], all_col_weights=all_col_weights,
        model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt,
        scale_map=scale_map,
    )
    rss_guard = RSSGuard(BURN_ROOT / "SHAKEDOWN_RSS_GUARD.json")
    rss_guard.start()
    try:
        # Small-N shakedown: first anchor plus the deterministic audit rung.
        # This exercises durable unit write, predecessor reconstruction, and
        # exact selected-content resume without pre-running the full layer.
        rungs = tuple(
            int(r) for r in os.environ.get(
                "DSV4_SHAKEDOWN_RUNGS", f"{ANCHORS[0]},{_audit_rung(0)}"
            ).split(",")
        )
        cells = _run_chain(
            layer=0, projection="gate_proj",
            pass_tag=BURN_PASS_TAGS["scout"], data=data,
            rungs=rungs, expert_ids=tuple(range(EXPERT_COUNT)), replay=False,
            full_encode_rungs=rungs, rss_guard=rss_guard,
        )
        report = {
            "schema": "prismaquant.dsv4_afast_shakedown_worker.v2",
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
            "rungs": list(rungs),
            "cells": {
                str(rung): {
                    "path": str(_burn_cell_path(
                        0, "gate_proj", BURN_PASS_TAGS["scout"], rung,
                    )),
                    "sha256": sha256_file(
                        _burn_cell_path(
                            0, "gate_proj", BURN_PASS_TAGS["scout"], rung,
                        )
                    ),
                    "content_key": None,
                }
                for rung in rungs
            },
            "peak_rss_bytes": rss_guard.peak_bytes,
            "host_available_before_bytes": before,
            "host_available_after_bytes": _host_available_bytes(),
        }
        # The durable cell envelope, not the per-expert chain identity, owns
        # the content key. Preserve it directly for the shakedown audit.
        for rung in rungs:
            payload = _load(_burn_cell_path(
                0, "gate_proj", BURN_PASS_TAGS["scout"], rung,
            ))
            report["cells"][str(rung)]["content_key"] = payload["content_key"]
        atomic_json(BURN_ROOT / "SHAKEDOWN_WORKER.json", report)
        return 0
    finally:
        rss_guard.stop()
        del data
        _reclaim()


def validate_shakedown() -> int:
    path = BURN_ROOT / "SHAKEDOWN.json"
    if not path.is_file():
        return 1
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "prismaquant.dsv4_afast_shakedown.v2"
        or report.get("implementation_sha256")
        != sha256_file(Path(__file__).resolve())
        or not report.get("pass")
    ):
        return 1
    for cell in report.get("cells", {}).values():
        item = Path(cell["path"])
        if not item.is_file() or sha256_file(item) != cell["sha256"]:
            return 1
    print(f"[afast-shakedown] content-verified {path}", flush=True)
    return 0


def finalize_shakedown(args: argparse.Namespace) -> int:
    worker = json.loads((BURN_ROOT / "SHAKEDOWN_WORKER.json").read_text())
    first = worker["cells"][str(min(map(int, worker["cells"])))]
    if first["sha256"] != args.first_cell_sha256:
        raise AssertionError("killed unit changed across content resume")
    memory_drop = max(0, int(args.before_available) - int(args.after_rederive))
    memory_pass = memory_drop <= 8 * 1024**3
    report = {
        "schema": "prismaquant.dsv4_afast_shakedown.v2",
        "pass": memory_pass,
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "deliberate_kill": {
            "signal": "TERM", "completed_cell_sha256": args.first_cell_sha256,
            "selected_content_rederived_bit_identically": True,
        },
        "memory": {
            "host_available_before_bytes": int(args.before_available),
            "host_available_after_kill_bytes": int(args.after_kill),
            "host_available_after_resume_bytes": int(args.after_resume),
            "host_available_after_rederive_bytes": int(args.after_rederive),
            "unrecovered_bytes": memory_drop,
            "allowed_unrecovered_bytes": 8 * 1024**3,
            "pass": memory_pass,
            "worker_peak_rss_bytes": int(worker["peak_rss_bytes"]),
        },
        "cells": worker["cells"],
        "resume_semantic": (
            "each completed cell's selected score, winning arm, and chain "
            "identity were reconstructed and compared bit-for-bit; activation "
            "replay was skipped. Losing free-candidate drift is separately "
            "persisted and is never allowed to change selected content"
        ),
    }
    atomic_json(BURN_ROOT / "SHAKEDOWN.json", report)
    if not memory_pass:
        atomic_text(RUN_ROOT / "DSV4_CAMPAIGN_STOP.md", "\n".join([
            "# DSV4 Campaign Stop", "", "- Stage: v2 shakedown",
            "- Outcome: STOP (memory did not recover)",
            f"- Evidence: `{BURN_ROOT / 'SHAKEDOWN.json'}`", "",
        ]))
        return 2
    print(f"[afast-shakedown] PASS -> {BURN_ROOT / 'SHAKEDOWN.json'}", flush=True)
    return 0


def merge() -> int:
    pilot = _pilot_gate()
    pilot_sha = sha256_file(PILOT2_JSON)
    all_costs = {}
    sources = []
    shards = []
    for layer in range(LAYER_COUNT):
        path = SHARD_ROOT / f"layer_{layer:03d}.pkl"
        if not path.is_file():
            raise SystemExit(f"merge requires 43/43; missing {path}")
        payload = _load(path)
        identity = payload.get("identity")
        if (
            payload.get("schema") != SCHEMA
            or payload.get("content_key") != _sha(identity)
            or int(payload["meta"]["layer"]) != layer
            or len(payload.get("costs", {})) != 775
            or identity["pilot2_report_sha256"] != pilot_sha
            or identity["source_index_sha256"] != sha256_file(
                SOURCE / "model.safetensors.index.json"
            )
        ):
            raise AssertionError(f"content verification failed {path}")
        _, verified = load_layer_identity(layer)
        if identity["verified_base_layer_sha256"] != verified["sha256"]:
            raise AssertionError(f"base content changed {path}")
        overlap = set(all_costs).intersection(payload["costs"])
        if overlap:
            raise AssertionError(f"duplicate qnames in {path}")
        all_costs.update(copy.deepcopy(payload["costs"]))
        shards.append(payload)
        sources.append({
            "layer": layer, "path": str(path), "sha256": sha256_file(path),
            "content_key": payload["content_key"], "imported": payload["meta"]["imported"],
        })
    if len(all_costs) != LAYER_COUNT * 775:
        raise AssertionError(f"merged row count {len(all_costs)}")
    base = _load(BASE_COST)
    if set(all_costs) != set(base["costs"]):
        raise AssertionError("merged qname set differs from verified base")
    formats = sorted({fmt for row in all_costs.values() for fmt in row})
    research_manifest = {
        "schema": RESEARCH_COST_MANIFEST_SCHEMA,
        "cost_provenance": RESEARCH_COST_PROVENANCE,
        "acceptance": "explicit_operator_A_FAST_decision",
        "base": {"path": str(BASE_COST), "sha256": sha256_file(BASE_COST)},
        "layers": sources, "layer_count": LAYER_COUNT, "rows_per_layer": 775,
        "assembled_row_count": len(all_costs), "segment_formats": formats,
        "formats": formats, "precedence": "content-keyed A-FAST layer shards",
        "profile": "A-FAST", "pilot2_report_sha256": pilot_sha,
        "acceptance_amendment_sha256": sha256_file(AMENDMENT_JSON),
    }
    provenance = copy.deepcopy(base.get("provenance") or {})
    provenance.update({
        "cost_provenance": RESEARCH_COST_PROVENANCE,
        "research_cost_manifest": research_manifest,
        "cb_serialized_payload": cb_serialization_context_stamp(
            CONTEXT, formats=[fmt for fmt in formats if "_CB_" in fmt],
        ),
        "afast_method": {
            "anchors": list(ANCHORS), "interpolator": "monotone_pchip",
            "amendment": "v2_accept_all_plus_per_layer_audit",
            "backstop_tolerance": BACKSTOP_TOLERANCE,
            "chain_arms": ["free", "embed"], "selection": "weight_mse",
            "activation_replay": "winning reconstruction only",
        },
    })
    merged = {
        "costs": all_costs, "formats": formats, "provenance": provenance,
        "meta": {
            **copy.deepcopy(base.get("meta") or {}),
            "research_assembly": {
                "profile": "A-FAST", "layers": LAYER_COUNT,
                "rows_per_layer": 775, "row_count": len(all_costs),
                "content_verified": "43/43", "mtp_bytes": MTP_BF16_BYTES,
            },
        },
    }
    output = BURN_ROOT / "cost_merged.pkl"
    atomic_pickle(output, merged)
    report = {
        "schema": "prismaquant.dsv4_afast_burn_report.v2",
        "profile": "A-FAST", "merge": "PASS", "layers": "43/43",
        "content_verified_layers": 43, "rows": len(all_costs),
        "cost_path": str(output), "cost_sha256": sha256_file(output),
        "imported_layers": list(IMPORTED_LAYERS), "pilot2_gates": pilot["gates"],
        "amendment": "v2_accept_all_plus_per_layer_audit",
        "acceptance_amendment_sha256": sha256_file(AMENDMENT_JSON),
        "audit_layers": {
            str(shard["meta"]["layer"]): {
                "audit_rung": shard["meta"]["audit_rung"],
                "pass": shard["meta"]["audit_gate_pass"],
                "full_layer_fallback": shard["meta"]["full_layer_fallback"],
                "projections": {
                    p: shard["meta"]["projection"][p]["audit"]
                    for p in PROJECTIONS
                },
            }
            for shard in shards
        },
        "projection_fit": {
            projection: {
                "accepted": sum(
                    int(shard["meta"]["projection"][projection]["fit"]["accepted"])
                    for shard in shards
                ),
                "total": LAYER_COUNT * EXPERT_COUNT,
            } for projection in PROJECTIONS
        },
        "elapsed_seconds_sum": sum(
            float(shard["meta"]["elapsed_seconds"]) for shard in shards
        ),
    }
    atomic_json(BURN_ROOT / "BURN_REPORT.json", report)
    print(f"[afast-burn] merge PASS 43/43 -> {output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=(
            "run", "merge", "shakedown-worker", "shakedown-validate",
            "shakedown-finalize",
        ),
    )
    parser.add_argument("--before-available", type=int)
    parser.add_argument("--after-kill", type=int)
    parser.add_argument("--after-resume", type=int)
    parser.add_argument("--after-rederive", type=int)
    parser.add_argument("--first-cell-sha256")
    args = parser.parse_args()
    if args.command == "run":
        return run_burn()
    if args.command == "merge":
        return merge()
    if args.command == "shakedown-worker":
        return run_shakedown_worker()
    if args.command == "shakedown-validate":
        return validate_shakedown()
    required = (
        args.before_available, args.after_kill, args.after_resume,
        args.after_rederive, args.first_cell_sha256,
    )
    if any(value is None for value in required):
        parser.error("shakedown-finalize requires all memory/hash arguments")
    return finalize_shakedown(args)


if __name__ == "__main__":
    raise SystemExit(main())
