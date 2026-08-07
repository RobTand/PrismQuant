#!/usr/bin/env python3
"""DSV4 A-FAST five-anchor monotone min-chain campaign.

This is a study driver, not a production-default encoder.  It extends the
existing content-verified DSV4 layer store and CB warm-state mechanism.  The
free arm is the unchanged production LDLQ encode; the integrated embed arm is
represented by the resident predecessor reconstruction and its exact recorded
weight MSE.  Only the selected reconstruction is activation-replayed.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import hashlib
import json
import math
import os
import pickle
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from prismaquant import format_registry as fr
from prismaquant.cb_minchain import (
    MINCHAIN_SCHEMA,
    chain_identity_from_digest,
    epsilon_le,
    recipe_solution_digest,
    select_arm,
)
from prismaquant.cb_warm_state import (
    CBWarmStateStore,
    build_warm_record,
    tensor_value_identity,
)
from prismaquant.nvfp4_cb_footprint import cb_serialization_context_stamp
from prismaquant.nvfp4_cb_formats import nvfp4_cb_reconstruct
from tools.dsv4_ldlq_cost_campaign import (
    ACT_ROOT,
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
from prismaquant.layer_streaming import _build_fp8_scale_inv_map, _build_weight_map
from prismaquant.nvfp4_cb_footprint import cb_fields_for_context
from prismaquant.production_weight_cache import (
    canonical_cb_col_weights_sha256,
)


SCHEMA = "prismaquant.dsv4_afast_minchain.v1"
PILOT2_SCHEMA = "prismaquant.dsv4_afast_pilot2.v1"
CHAIN_VERSION = "dsv4-afast-5anchor-pchip-minchain-v1"
PILOT2_LAYER = 14
ANCHORS = (29, 35)  # hierarchical-law rev, two-probe form (Rob + operator
# 2026-08-06 evening): level-2 is 2-DoF, so two probes determine it; the
# sweep over all pairs on L14+L0 per-expert truth picks (29,35) — worst
# medians L14 1.2-2.5%, L0 gate/up 4.7/4.8% (pass), L0 down 5.3% (gated to
# fallback, correctly). The audit draw is measured anyway; after the gate
# passes, level-2 refits on anchors+audit (3 points) for the menu fill.
# interpolation is no longer PCHIP. error(K) follows the hierarchical
# rate-distortion law log y = a0 + a1*K + phi_{K%4} (level-1, fitted once
# on the L14 21-rung pilot ladder) plus a per-expert affine log correction
# a + b*K (level-2, fitted from that expert's own anchors). Validated
# out-of-sample per-expert on L0's full CBL ladder: gate 3.1%, up 4.7%
# worst medians at n=3; deviants (L0 down K36-class) are caught fail-closed
# by the detection rule below and ship measured. ERROR_LAW.md /
# CAMPAIGN_V2_PROTOCOL.md in cost-ldlq/interp-diagnosis; muse-spark study,
# independently re-derived by operator before adoption.

LAW_COEF = {
    # a0, a1, phi1, phi2, phi3  (phi0 = 0), fitted on L14 medians K28-38
    "gate_proj": (-6.8249, -0.15897, 0.03594, 0.05849, 0.06637),
    "up_proj": (-6.7873, -0.15964, 0.04179, 0.06700, 0.06039),
    "down_proj": (-6.4734, -0.16092, 0.04078, 0.06231, 0.05702),
}
LAW_DETECT = {"b_abs": 0.09}  # level-2 tilt bound. Evidence (L0 shard +
# L21 offline, 2026-08-06): the scale offset `a` varies legitimately by
# layer (L21 down a=-1.03 with 3.1% interiors; L0 a=+1.4 with ~2% audit
# medians) — flagging on |a| is flagging what level-2 exists to absorb.
# |b| <= 0.09 is the monotone-safety bound (a1 + b + max dphi < 0); actual
# misfit detection is the audit gate, which measures prediction error.


def law_log_G(projection: str, rung: int) -> float:
    a0, a1, p1, p2, p3 = LAW_COEF[str(projection)]
    phi = (0.0, p1, p2, p3)[int(rung) % 4]
    return a0 + a1 * float(rung) + phi


def law_fit_level2(
    anchors: Sequence[int], values: Sequence[float], projection: str,
) -> tuple[float, float]:
    """Per-unit affine log correction (a, b) from anchor measurements."""
    rhs = np.array([
        math.log(max(float(v), 1e-30)) - law_log_G(projection, k)
        for k, v in zip(anchors, values)
    ])
    design = np.array([[1.0, float(k)] for k in anchors])
    (a, b), *_ = np.linalg.lstsq(design, rhs, rcond=None)
    return float(a), float(b)


def law_predict(projection: str, rung: int, a: float, b: float) -> float:
    return math.exp(law_log_G(projection, rung) + a + b * float(rung))

GEOMETRY_OMEGA = {
    # per-projection sub-table weights, estimated from pilot average
    # per-family drops; holdout medians robust to +-20% perturbation.
    "gate_proj": (0.18, 0.22, 0.26, 0.34),
    "up_proj": (0.19, 0.21, 0.25, 0.35),
    "down_proj": (0.20, 0.22, 0.25, 0.33),
}


def geometry_x(rungs: Sequence[int], projection: str) -> list[float]:
    """Map rungs to the coordinate where the error staircase is smooth."""
    from prismaquant.cb_layout import subtable_bit_widths

    weights = GEOMETRY_OMEGA[str(projection)]
    return [
        sum(
            weight * bits
            for weight, bits in zip(
                weights, subtable_bit_widths(int(rung), "product", 4)
            )
        )
        for rung in rungs
    ]
ACCEPTANCE_FIT_ANCHORS = (28, 38, 48)
ACCEPTANCE_HOLDOUTS = (33, 43)
FIT_TOLERANCE = 0.10
REL_EPSILON = 1e-12
PILOT2_ROOT = RUN_ROOT / "pilot2"
PILOT2_CELL_SHARDS = RUN_ROOT / "pilot2-shards"
PILOT2_LAYER_SHARDS = PILOT2_ROOT / "shards"
BURN_SHARDS = RUN_ROOT / "shards"
SEGMENTS = RUN_ROOT / "segments"
LOGS = RUN_ROOT / "logs"
BASE_SEGMENTS = (
    Path("/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2")
    / "artifacts-mxfp4/probe-k12k18/by-layer"
)
PROD_SHARDS = (
    Path("/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2")
    / "work-prod/shards"
)
RSS_LIMIT_BYTES = 60 * 1024**3
RSS_POLL_SECONDS = 2.0
HOST_AVAILABLE_CACHE_RELEASE_BYTES = 40 * 1024**3


class RSSLimitExceeded(RuntimeError):
    """The registered foreground measurement RSS limit was crossed."""


def _rss_bytes() -> int:
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/self/status has no VmRSS")


def _host_available_bytes() -> int:
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable")


def _malloc_trim() -> None:
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError):
        pass


def _reclaim() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    _malloc_trim()


def _unit_boundary_reclaim(*, unit: str) -> dict[str, Any]:
    """Release host garbage and shed CUDA cache when unified memory is low."""
    gc.collect()
    _malloc_trim()
    available = _host_available_bytes()
    released_cuda_cache = available < HOST_AVAILABLE_CACHE_RELEASE_BYTES
    if released_cuda_cache:
        torch.cuda.empty_cache()
    evidence = {
        "unit": str(unit),
        "host_available_bytes": int(available),
        "threshold_bytes": HOST_AVAILABLE_CACHE_RELEASE_BYTES,
        "cuda_cache_released": released_cuda_cache,
    }
    print(
        f"[afast] memory-boundary unit={unit} "
        f"host_available={available / 1024**3:.2f}GiB "
        f"cuda_cache_released={released_cuda_cache}",
        flush=True,
    )
    return evidence


class RSSGuard:
    """Poll process RSS and persist first-crossing evidence immediately."""

    def __init__(self, evidence_path: Path, limit_bytes: int = RSS_LIMIT_BYTES):
        self.evidence_path = Path(evidence_path)
        self.limit_bytes = int(limit_bytes)
        self.peak_bytes = 0
        self.stage = "initializing"
        self._tripped = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="afast-rss-guard", daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=RSS_POLL_SECONDS * 2)

    def set_stage(self, stage: str) -> None:
        self.stage = str(stage)
        self.checkpoint()

    def checkpoint(self) -> None:
        rss = _rss_bytes()
        self.peak_bytes = max(self.peak_bytes, rss)
        if rss > self.limit_bytes:
            self._trip(rss)
        if self._tripped.is_set():
            raise RSSLimitExceeded(
                f"RSS guard crossed at stage={self.stage!r}: "
                f"peak={self.peak_bytes / 1024**3:.3f} GiB > "
                f"limit={self.limit_bytes / 1024**3:.3f} GiB"
            )

    def _trip(self, rss: int) -> None:
        if self._tripped.is_set():
            return
        self.peak_bytes = max(self.peak_bytes, int(rss))
        self._tripped.set()
        atomic_json(self.evidence_path, {
            "schema": "prismaquant.dsv4_rss_guard.v1",
            "created_at": utc_now(),
            "pid": os.getpid(),
            "stage": self.stage,
            "rss_bytes": int(rss),
            "peak_rss_bytes": int(self.peak_bytes),
            "limit_bytes": self.limit_bytes,
            "action": "flush_and_abort_at_next_safe_checkpoint",
        })

    def _run(self) -> None:
        while not self._stop.wait(RSS_POLL_SECONDS):
            try:
                rss = _rss_bytes()
                self.peak_bytes = max(self.peak_bytes, rss)
                if rss > self.limit_bytes:
                    self._trip(rss)
                    return
            except Exception:
                # The foreground checkpoints remain authoritative.  A sampler
                # read failure must not manufacture a memory-limit crossing.
                return


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure() -> None:
    os.environ["PRISMAQUANT_CB_LDLQ"] = "1"
    os.environ["PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS"] = "1"
    os.environ["PRISMAQUANT_CB_LDLQ_FEEDER_THREADS"] = "16"
    os.environ["PRISMAQUANT_CB_LDLQ_EXPERT_BATCH"] = "16"
    os.environ["PRISMAQUANT_CB_LDLQ_BATCH_STREAMS"] = "1"
    os.environ["PRISMAQUANT_CB_ENCODE_TIER"] = "balanced"
    os.environ.setdefault("PRISMAQUANT_CB_ENCODE_COMPILE", "1")
    if torch.cuda.is_available():
        # torch.linalg's CUDA dispatch wrapper is lazily initialized.  Letting
        # 16 LDLQ feeder threads race its first Cholesky call can raise
        # "lazy wrapper should be called at most once" before any arithmetic.
        # Prime dispatch on the foreground thread; this tiny matrix is not
        # part of measurement state.
        torch.linalg.cholesky(torch.eye(1, device="cuda"))
        torch.cuda.synchronize()


def _sha256_text(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _pava_decreasing(y: np.ndarray) -> np.ndarray:
    vals = [-float(v) for v in y]
    means: list[float] = []
    counts: list[int] = []
    for value in vals:
        means.append(value)
        counts.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            count = counts[-2] + counts[-1]
            means[-2] = (
                means[-2] * counts[-2] + means[-1] * counts[-1]
            ) / count
            counts[-2] = count
            means.pop()
            counts.pop()
    return -np.asarray([
        mean for mean, count in zip(means, counts) for _ in range(count)
    ])


def pchip_monotone(
    x: Sequence[float], y: Sequence[float], xq: Sequence[float],
) -> np.ndarray:
    """Fritsch-Carlson PCHIP, matching the registered anchor study."""
    x = np.asarray(x, dtype=float)
    y = _pava_decreasing(np.asarray(y, dtype=float))
    xq = np.asarray(xq, dtype=float)
    h = np.diff(x)
    d = np.diff(y) / h
    n = len(x)
    m = np.zeros(n, dtype=float)
    if n == 2:
        m[:] = d[0]
    else:
        for index in range(1, n - 1):
            if (
                d[index - 1] == 0
                or d[index] == 0
                or np.sign(d[index - 1]) != np.sign(d[index])
            ):
                m[index] = 0.0
            else:
                w1 = 2 * h[index] + h[index - 1]
                w2 = h[index] + 2 * h[index - 1]
                m[index] = (w1 + w2) / (
                    w1 / d[index - 1] + w2 / d[index]
                )

        def endpoint(h0: float, h1: float, d0: float, d1: float) -> float:
            value = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
            if np.sign(value) != np.sign(d0):
                return 0.0
            if np.sign(d0) != np.sign(d1) and abs(value) > abs(3 * d0):
                return 3 * d0
            return value

        m[0] = endpoint(h[0], h[1], d[0], d[1])
        m[-1] = endpoint(h[-1], h[-2], d[-1], d[-2])
    out = np.empty_like(xq)
    for out_index, query in enumerate(xq):
        index = min(max(int(np.searchsorted(x, query) - 1), 0), n - 2)
        t = (query - x[index]) / h[index]
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        out[out_index] = (
            h00 * y[index]
            + h10 * h[index] * m[index]
            + h01 * y[index + 1]
            + h11 * h[index] * m[index + 1]
        )
    return out


def _activation_replay(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    activation_rows: Sequence[torch.Tensor],
    expert_ids: Sequence[int],
    rung: int,
) -> tuple[list[float], float]:
    """Replay the production activation-inclusive cost exactly once."""
    spec = fr.get_format(f"FP8_CB_K{rung}")
    torch.cuda.synchronize()
    started = time.perf_counter()
    results: list[tuple[int, torch.Tensor]] = []
    for batch_start in range(0, len(expert_ids), 16):
        batch_stop = min(batch_start + 16, len(expert_ids))
        streams = [
            torch.cuda.Stream(device=weight.device)
            for _ in range(batch_stop - batch_start)
        ]

        def work(local: int) -> tuple[int, torch.Tensor]:
            index = batch_start + local
            expert = int(expert_ids[index])
            with torch.cuda.device(weight.device), torch.cuda.stream(streams[local]):
                rows = activation_rows[expert].to(
                    device=weight.device, dtype=torch.float32, non_blocking=True,
                )
                quantized_rows = spec.activation_quantize_dequantize(rows.clone())
                reference = rows @ weight[index].to(torch.float32).transpose(0, 1)
                quantized = (
                    quantized_rows
                    @ reconstruction[index].to(torch.float32).transpose(0, 1)
                )
                value = (reference - quantized).square().mean()
            return index, value

        with ThreadPoolExecutor(max_workers=len(streams)) as pool:
            results.extend(pool.map(work, range(len(streams))))
        current = torch.cuda.current_stream(weight.device)
        for stream in streams:
            current.wait_stream(stream)
    torch.cuda.synchronize()
    results.sort(key=lambda item: item[0])
    values = torch.stack([value for _, value in results]).cpu().tolist()
    return list(map(float, values)), time.perf_counter() - started


def _weight_mse(weight: torch.Tensor, reconstruction: torch.Tensor) -> list[float]:
    return (
        (weight - reconstruction).float().square().mean(dim=(1, 2))
        .detach().cpu().tolist()
    )


TIER3_GROUP_COUNTS = (2, 4, 8)  # tier-3 side-band (Rob, 2026-08-06):
# contiguous input-dim blocks, superblock-aligned for every projection shape.


def _weight_group_mse(
    weight: torch.Tensor, reconstruction: torch.Tensor,
    group_counts: Sequence[int] = TIER3_GROUP_COUNTS,
) -> dict[int, list[list[float]]]:
    """Per-column-group MSE decomposition of the same reconstruction.

    Equal-width groups over the input dim, so the mean of a unit's group
    means equals its whole-unit mean — the additivity identity the tier-3
    study verifies. Study-grade side-band: banked in burn cells only, never
    a menu row."""
    squared = (weight - reconstruction).float().square()
    result: dict[int, list[list[float]]] = {}
    for count in group_counts:
        width = int(squared.shape[2]) // int(count)
        result[int(count)] = torch.stack(
            [view.mean(dim=(1, 2)) for view in squared.split(width, dim=2)],
            dim=1,
        ).detach().cpu().tolist()
    del squared
    return result


def _select_experts(value: torch.Tensor, expert_ids: Sequence[int]) -> torch.Tensor:
    """Return the resident full stack without duplicating it when possible."""
    if len(expert_ids) == int(value.shape[0]) and all(
        int(expert) == index for index, expert in enumerate(expert_ids)
    ):
        return value
    index = torch.as_tensor(expert_ids, device=value.device)
    return value.index_select(0, index).contiguous()


def _encode_free(
    *,
    layer: int,
    projection: str,
    rung: int,
    data: Mapping[str, Any],
    expert_ids: Sequence[int],
    source_identity: tuple[list[int], str] | None = None,
    col_weights_identity: tuple[list[int], str] | None = None,
    use_warm: bool = True,
    expected_free_errors: Sequence[float] | None = None,
) -> tuple[dict, torch.Tensor, list[float], dict[str, Any], str]:
    weight = _select_experts(data["weight"], expert_ids)
    col_weights = _select_experts(data["col_weights"], expert_ids)
    activation_rows = tuple(data["activation_rows"][i] for i in expert_ids)
    format_name = f"FP8_CB_K{rung}"
    spec = fr.get_format(format_name)
    if source_identity is None:
        source_identity = tensor_value_identity(weight)
    if col_weights_identity is None:
        col_weights_identity = tensor_value_identity(col_weights)
    logical_qname = f"model.layers.{layer}.mlp.experts.{projection}"
    warm_store = CBWarmStateStore(RUN_ROOT / "warm-state")
    warm_record = (
        warm_store.load_matching(
            qname=logical_qname,
            format_name=format_name,
            source_shape=source_identity[0],
            source_digest=source_identity[1],
            col_weights_shape=col_weights_identity[0],
            col_weights_digest=col_weights_identity[1],
            context=CONTEXT,
        )
        if use_warm else None
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    fields = cb_fields_for_context(
        spec,
        weight,
        context=CONTEXT,
        col_weights=col_weights,
        activation_rows=activation_rows,
        warm_scale_state=(
            None if warm_record is None else dict(warm_record.scale_state)
        ),
    )
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - started
    score_started = time.perf_counter()
    reconstruction = nvfp4_cb_reconstruct(
        fields, rung, grid="fp8", mode="product",
    ).to(weight.dtype)
    errors = _weight_mse(weight, reconstruction)
    warm_outcome = "warm_used_content_oracle_verified"
    if warm_record is None:
        warm_outcome = "cold_no_oracle" if not use_warm else "cold_fallback"
    elif (
        expected_free_errors is None
        or list(map(float, expected_free_errors)) != errors
    ):
        # Warm scale state is an optimization, never truth.  A complete prior
        # content-keyed cell (or the killed run's completed gate projection)
        # is required as the numeric oracle.  Mismatch falls back to the full
        # compiled scale sweep before any shard can be written.
        del fields, reconstruction
        _reclaim()
        fields = cb_fields_for_context(
            spec,
            weight,
            context=CONTEXT,
            col_weights=col_weights,
            activation_rows=activation_rows,
        )
        reconstruction = nvfp4_cb_reconstruct(
            fields, rung, grid="fp8", mode="product",
        ).to(weight.dtype)
        errors = _weight_mse(weight, reconstruction)
        warm_outcome = "warm_oracle_mismatch_fallback_cold"
        if (
            expected_free_errors is not None
            and list(map(float, expected_free_errors)) != errors
        ):
            warm_outcome = "warm_and_prior_oracle_rejected_fresh_cold"
    torch.cuda.synchronize()
    score_seconds = time.perf_counter() - score_started
    warm_started = time.perf_counter()
    warm_path = warm_store.write(
        build_warm_record(
            qname=logical_qname,
            format_name=format_name,
            source_weight=weight,
            col_weights=col_weights,
            context=CONTEXT,
            fields=fields,
            source_identity=source_identity,
            col_weights_identity=col_weights_identity,
        )
    )
    warm_state_write_seconds = time.perf_counter() - warm_started
    return fields, reconstruction, errors, {
        "free_encode_seconds": encode_seconds,
        "free_reconstruct_and_weight_mse_seconds": score_seconds,
        # Serialization is recorded for auditability but excluded from P4:
        # P4 compares integrated GPU work, and warm-state persistence is a
        # campaign checkpoint rather than part of either algorithmic arm.
        "warm_state_write_seconds": warm_state_write_seconds,
        "warm_state_outcome": (
            warm_outcome
        ),
    }, str(warm_path)


def _select_cell(
    *,
    layer: int,
    projection: str,
    rung: int,
    expert_ids: Sequence[int],
    free_errors: Sequence[float],
    free_reconstruction: torch.Tensor,
    predecessor_errors: Sequence[float] | None,
    predecessor_reconstruction: torch.Tensor | None,
    predecessor_identities: Sequence[Mapping[str, Any]] | None,
    warm_path: str,
    content_guard: Mapping[str, Any],
) -> tuple[list[float], torch.Tensor, list[str], list[dict[str, Any]], float]:
    started = time.perf_counter()
    selected_errors: list[float] = []
    arms: list[str] = []
    identities: list[dict[str, Any]] = []
    use_embed: list[bool] = []
    for local, expert in enumerate(expert_ids):
        if predecessor_errors is None:
            arm, error = "free", float(free_errors[local])
        else:
            arm, error = select_arm({
                "free": float(free_errors[local]),
                "embed": float(predecessor_errors[local]),
            }, rtol=REL_EPSILON)
        predecessor_digest = None
        if arm == "embed":
            assert predecessor_identities is not None
            predecessor_digest = str(
                predecessor_identities[local]["solution_digest"]
            )
        recipe = {
            "campaign_schema": SCHEMA,
            "chain_version": CHAIN_VERSION,
            "layer": layer,
            "projection": projection,
            "expert": int(expert),
            "rung": rung,
            "winning_arm": arm,
            "predecessor_digest": predecessor_digest,
            "free_warm_state": warm_path,
            "content_guard": dict(content_guard),
        }
        identities.append(chain_identity_from_digest(
            winning_arm=arm,
            solution_digest_value=recipe_solution_digest(recipe),
            predecessor_digest=predecessor_digest,
        ))
        selected_errors.append(error)
        arms.append(arm)
        use_embed.append(arm == "embed")
    if predecessor_reconstruction is None:
        selected_reconstruction = free_reconstruction
    else:
        mask = torch.as_tensor(
            use_embed, device=free_reconstruction.device, dtype=torch.bool,
        ).view(-1, 1, 1)
        selected_reconstruction = torch.where(
            mask, predecessor_reconstruction, free_reconstruction,
        )
    torch.cuda.synchronize()
    return (
        selected_errors,
        selected_reconstruction,
        arms,
        identities,
        time.perf_counter() - started,
    )


def _cell_shard_path(layer: int, projection: str, rung: int) -> Path:
    return PILOT2_CELL_SHARDS / f"layer_{layer:03d}_{projection}_K{rung}.pkl"


def _cell_identity(
    *, layer: int, projection: str, rung: int,
    expert_ids: Sequence[int], content_guard: Mapping[str, Any],
    predecessor_content_key: str | None,
) -> dict[str, Any]:
    return {
        "schema": "prismaquant.dsv4_afast_cell_identity.v1",
        "campaign_schema": SCHEMA,
        "chain_version": CHAIN_VERSION,
        "layer": int(layer),
        "projection": str(projection),
        "rung": int(rung),
        "expert_ids": list(map(int, expert_ids)),
        "content_guard": dict(content_guard),
        "predecessor_content_key": predecessor_content_key,
        "selection_metric": "per-expert weight_mse",
        "epsilon_rtol": REL_EPSILON,
        "tie_priority": ["free", "embed"],
        "activation_replay": "winning_reconstruction_only",
    }


def _validated_cell_shard(
    path: Path, expected_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    expected_key = _sha256_text(expected_identity)
    cell = payload.get("cell")
    if (
        payload.get("schema") != "prismaquant.dsv4_afast_cell_shard.v1"
        or payload.get("identity") != dict(expected_identity)
        or payload.get("content_key") != expected_key
        or not isinstance(cell, Mapping)
        or int(cell.get("rung", -1)) != int(expected_identity["rung"])
        or len(cell.get("expert_ids", ())) != len(expected_identity["expert_ids"])
    ):
        raise AssertionError(f"stale or corrupt A-FAST cell shard: {path}")
    warm_path = Path(str(cell.get("warm_state_path", "")))
    if not warm_path.is_file():
        raise AssertionError(f"A-FAST cell shard has no warm state: {path}")
    return payload


def _delta_summary(
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


def _quarantine_projection_suffix(
    *, layer: int, projection: str, first_rung: int,
    mismatch: Mapping[str, Any],
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root = PILOT2_ROOT / "quarantine-content-mismatch" / stamp
    root.mkdir(parents=True, exist_ok=False)
    moved = []
    for dependent_rung in RUNGS:
        if dependent_rung < int(first_rung):
            continue
        source = _cell_shard_path(layer, projection, dependent_rung)
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
        "schema": "prismaquant.dsv4_afast_quarantine.v1",
        "created_at": utc_now(),
        "reason": "content-matched restore changed numeric winner",
        "dependency_rule": "trigger rung and every later rung in the projection",
        "mismatch": dict(mismatch),
        "moved": moved,
    })
    return manifest_path


def _measure_projection(
    *,
    layer: int,
    projection: str,
    data: Mapping[str, Any],
    rungs: Sequence[int],
    expert_ids: Sequence[int] = tuple(range(256)),
    rss_guard: RSSGuard | None = None,
    oracle_cells: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    weight = _select_experts(data["weight"], expert_ids)
    col_weights = _select_experts(data["col_weights"], expert_ids)
    if rss_guard is not None:
        rss_guard.set_stage(f"{projection}:content-identity")
    source_identity = tensor_value_identity(weight)
    col_weights_identity = tensor_value_identity(col_weights)
    predecessor_errors: list[float] | None = None
    predecessor_reconstruction: torch.Tensor | None = None
    predecessor_identities: list[dict[str, Any]] | None = None
    predecessor_content_key: str | None = None
    cells: dict[int, dict[str, Any]] = {}
    guard = {
        "source_index_sha256": sha256_file(SOURCE / "model.safetensors.index.json"),
        "source_shape": source_identity[0],
        "source_digest": source_identity[1],
        "col_weights_shape": col_weights_identity[0],
        "col_weights_digest": col_weights_identity[1],
        "serialization_context_sha256": _sha256_text(
            cb_serialization_context_stamp(
                CONTEXT, formats=[f"FP8_CB_K{k}" for k in RUNGS],
            )
        ),
        "minchain_module_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "prismaquant/cb_minchain.py"
        ),
    }
    for rung in rungs:
        identity = _cell_identity(
            layer=layer,
            projection=projection,
            rung=int(rung),
            expert_ids=expert_ids,
            content_guard=guard,
            predecessor_content_key=predecessor_content_key,
        )
        shard_path = _cell_shard_path(layer, projection, int(rung))
        existing = _validated_cell_shard(shard_path, identity)
        oracle_cell = (
            existing["cell"] if existing is not None
            else (oracle_cells or {}).get(int(rung))
        )
        action = "restore" if existing is not None else "measure"
        if rss_guard is not None:
            rss_guard.set_stage(f"{projection}:K{rung}:{action}")
        print(
            f"[afast] layer {layer} {projection} K{rung} {action}", flush=True,
        )
        cell_started = time.perf_counter()
        fields, free_reconstruction, free_errors, timing, warm_path = _encode_free(
            layer=layer,
            projection=projection,
            rung=rung,
            data=data,
            expert_ids=expert_ids,
            source_identity=source_identity,
            col_weights_identity=col_weights_identity,
            use_warm=oracle_cell is not None,
            expected_free_errors=(
                None if oracle_cell is None
                else oracle_cell["free_weight_mse"]
            ),
        )
        oracle_exact = bool(
            oracle_cell is not None
            and list(map(float, free_errors))
            == list(map(float, oracle_cell["free_weight_mse"]))
        )
        if rss_guard is not None:
            rss_guard.checkpoint()
        (
            selected_errors,
            selected_reconstruction,
            arms,
            identities,
            selection_seconds,
        ) = _select_cell(
            layer=layer,
            projection=projection,
            rung=rung,
            expert_ids=expert_ids,
            free_errors=free_errors,
            free_reconstruction=free_reconstruction,
            predecessor_errors=predecessor_errors,
            predecessor_reconstruction=predecessor_reconstruction,
            predecessor_identities=predecessor_identities,
            warm_path=warm_path,
            content_guard=guard,
        )
        embed_errors = (
            [math.nan] * len(expert_ids)
            if predecessor_errors is None
            else list(map(float, predecessor_errors))
        )
        # Construction asserts are runtime gates.  An exception is an ABORT;
        # callers never convert it into a numeric violation count.
        if predecessor_errors is not None:
            for selected, predecessor in zip(selected_errors, predecessor_errors):
                if not epsilon_le(selected, predecessor, rtol=REL_EPSILON):
                    raise AssertionError(
                        f"P1 abort: layer {layer} {projection} K{rung} "
                        f"selected={selected} predecessor={predecessor}"
                    )
        for selected, free in zip(selected_errors, free_errors):
            if not epsilon_le(selected, free, rtol=REL_EPSILON):
                raise AssertionError(
                    f"P2 abort: layer {layer} {projection} K{rung} "
                    f"selected={selected} free={free}"
                )
        if existing is not None:
            cell = dict(existing["cell"])
            content_key = str(existing["content_key"])
            resume_mismatch = bool(
                list(map(float, free_errors)) != cell["free_weight_mse"]
                or list(map(float, selected_errors))
                != cell["selected_weight_mse"]
                or arms != cell["winning_arm"]
                or identities != cell["identity"]
            )
            if resume_mismatch:
                mismatch = {
                    "schema": "prismaquant.dsv4_afast_resume_mismatch.v1",
                    "created_at": utc_now(),
                    "path": str(shard_path),
                    "content_key": content_key,
                    "unit": {
                        "layer": layer, "projection": projection,
                        "rung": int(rung),
                    },
                    "free_weight_mse": _delta_summary(
                        free_errors, cell["free_weight_mse"],
                    ),
                    "selected_weight_mse": _delta_summary(
                        selected_errors, cell["selected_weight_mse"],
                    ),
                    "arm_mismatch_count": sum(
                        int(left != right)
                        for left, right in zip(arms, cell["winning_arm"])
                    ),
                    "identity_mismatch_count": sum(
                        int(left != right)
                        for left, right in zip(identities, cell["identity"])
                    ),
                    "warm_state_outcome": timing["warm_state_outcome"],
                    "allocator_policy": os.environ.get(
                        "PYTORCH_CUDA_ALLOC_CONF"
                    ),
                }
                atomic_json(PILOT2_ROOT / "CONTENT_RESUME_MISMATCH.json", mismatch)
                manifest_path = _quarantine_projection_suffix(
                    layer=layer, projection=projection, first_rung=int(rung),
                    mismatch=mismatch,
                )
                print(
                    f"[afast] quarantined stale suffix from K{rung}; "
                    f"manifest={manifest_path}", flush=True,
                )
                existing = None
                oracle_exact = False

        if existing is None:
            replay_errors, replay_seconds = _activation_replay(
                weight, selected_reconstruction, data["activation_rows"],
                expert_ids, rung,
            )
            optimized_total_seconds = (
                float(timing["free_encode_seconds"])
                + float(timing["free_reconstruct_and_weight_mse_seconds"])
                + selection_seconds
                + replay_seconds
            )
            timing.update({
                "selection_seconds": selection_seconds,
                "winning_replay_seconds": replay_seconds,
                "total_seconds": optimized_total_seconds,
                "cell_wall_seconds": time.perf_counter() - cell_started,
            })
            cell = {
                "rung": int(rung),
                "expert_ids": list(map(int, expert_ids)),
                "free_weight_mse": list(map(float, free_errors)),
                "embed_weight_mse": embed_errors,
                "selected_weight_mse": list(map(float, selected_errors)),
                "selected_output_mse": list(map(float, replay_errors)),
                "winning_arm": arms,
                "identity": identities,
                "warm_state_path": warm_path,
                "timing": timing,
                "replay_count": len(expert_ids),
                "rss_bytes_before_write": _rss_bytes(),
                "numeric_oracle": (
                    "none_cold_measurement"
                    if oracle_cell is None else
                    (
                        "content_matched_completed_gate_projection"
                        if oracle_exact else
                        "prior_oracle_rejected_fresh_cold_authoritative"
                    )
                ),
            }
            content_key = _sha256_text(identity)
            atomic_pickle(shard_path, {
                "schema": "prismaquant.dsv4_afast_cell_shard.v1",
                "created_at": utc_now(),
                "content_key": content_key,
                "identity": identity,
                "cell": cell,
            })
            print(f"[afast] wrote {shard_path}", flush=True)
        else:
            print(f"[afast] content-resume skipped replay {shard_path}", flush=True)
        cells[int(rung)] = cell
        old_predecessor = predecessor_reconstruction
        predecessor_errors = selected_errors
        predecessor_reconstruction = selected_reconstruction
        predecessor_identities = identities
        predecessor_content_key = content_key
        del fields, free_reconstruction, old_predecessor
        _unit_boundary_reclaim(unit=f"L{layer}:{projection}:K{rung}")
        if rss_guard is not None:
            rss_guard.checkpoint()
        print(
            f"[afast] layer {layer} {projection} K{rung} "
            f"free={arms.count('free')} embed={arms.count('embed')} "
            f"total={float(cell['timing']['total_seconds']):.1f}s "
            f"rss={_rss_bytes() / 1024**3:.2f}GiB",
            flush=True,
        )
    del predecessor_reconstruction
    _reclaim()
    return {"projection": projection, "cells": cells, "content_guard": guard}


def _fit_errors(layer_record: Mapping[str, Any]) -> dict[str, Any]:
    nonanchors = [k for k in RUNGS if k not in ANCHORS]
    projections: dict[str, Any] = {}
    for projection in PROJECTIONS:
        cells = layer_record["projections"][projection]["cells"]
        errors: list[float] = []
        per_slice: list[dict[str, float]] = []
        fallback = 0
        for expert in range(256):
            anchor_y = [
                float(cells[k]["selected_weight_mse"][expert]) for k in ANCHORS
            ]
            predicted = pchip_monotone(ANCHORS, anchor_y, RUNGS)
            local = []
            for rung in nonanchors:
                truth = float(cells[rung]["selected_weight_mse"][expert])
                rel = abs(float(predicted[rung - 28]) - truth) / max(abs(truth), 1e-30)
                local.append(rel)
                errors.append(rel)
            acceptance_pred = pchip_monotone(
                ACCEPTANCE_FIT_ANCHORS,
                [float(cells[k]["selected_weight_mse"][expert])
                 for k in ACCEPTANCE_FIT_ANCHORS],
                ACCEPTANCE_HOLDOUTS,
            )
            holdout_errors = [
                abs(float(pred) - float(cells[k]["selected_weight_mse"][expert]))
                / max(abs(float(cells[k]["selected_weight_mse"][expert])), 1e-30)
                for pred, k in zip(acceptance_pred, ACCEPTANCE_HOLDOUTS)
            ]
            accepted = all(value <= FIT_TOLERANCE for value in holdout_errors)
            fallback += int(not accepted)
            per_slice.append({
                "expert": expert,
                "median": statistics.median(local),
                "p95": percentile(local, 0.95),
                "max": max(local),
                "K33_acceptance_error": holdout_errors[0],
                "K43_acceptance_error": holdout_errors[1],
                "accepted": accepted,
            })
        stats = {
            "n": len(errors),
            "median": statistics.median(errors),
            "p95": percentile(errors, 0.95),
            "max": max(errors),
            "gate_pass": (
                statistics.median(errors) <= 0.05
                and percentile(errors, 0.95) <= 0.15
            ),
            "fallback_slices": fallback,
            "fallback_fraction": fallback / 256.0,
            "per_slice": per_slice,
        }
        projections[projection] = stats
    return {
        "anchors": list(ANCHORS),
        "scored_nonanchor_rungs": nonanchors,
        "acceptance_semantics": {
            "fit_anchors": list(ACCEPTANCE_FIT_ANCHORS),
            "held_out": list(ACCEPTANCE_HOLDOUTS),
            "per_holdout_tolerance": FIT_TOLERANCE,
        },
        "projections": projections,
        "gate_pass": all(projections[p]["gate_pass"] for p in PROJECTIONS),
        "fallback_slices": sum(projections[p]["fallback_slices"] for p in PROJECTIONS),
        "fallback_fraction": sum(
            projections[p]["fallback_slices"] for p in PROJECTIONS
        ) / (256 * len(PROJECTIONS)),
    }


def _gate_counts(layer_record: Mapping[str, Any]) -> tuple[int, int, dict[str, float]]:
    p1 = p2 = 0
    free_seconds = total_seconds = 0.0
    for projection in PROJECTIONS:
        previous: list[float] | None = None
        for rung in RUNGS:
            cell = layer_record["projections"][projection]["cells"][rung]
            selected = list(map(float, cell["selected_weight_mse"]))
            free = list(map(float, cell["free_weight_mse"]))
            if previous is not None:
                p1 += sum(
                    not epsilon_le(a, b, rtol=REL_EPSILON)
                    for a, b in zip(selected, previous)
                )
            p2 += sum(
                not epsilon_le(a, b, rtol=REL_EPSILON)
                for a, b in zip(selected, free)
            )
            previous = selected
            timing = cell["timing"]
            free_seconds += (
                float(timing["free_encode_seconds"])
                + float(timing["free_reconstruct_and_weight_mse_seconds"])
            )
            total_seconds += float(timing["total_seconds"])
    return p1, p2, {
        "free_seconds": free_seconds,
        "total_seconds": total_seconds,
        "total_over_free": total_seconds / free_seconds,
    }


def _pilot2_markdown(report: Mapping[str, Any]) -> str:
    gates = report["gates"]
    lines = [
        "# DSV4 A-FAST Pilot-2",
        "",
        f"- Result: **{report['decision']}**",
        f"- Pre-declared layer: {PILOT2_LAYER}",
        "- Method: free/embed min-chain, refine deleted; per-expert weight-MSE selection; one winning-arm production activation-QDQ output-MSE replay per measured cell",
        "- Anchors: K28/K33/K38/K43/K48; monotone PCHIP scored on all 16 non-anchor rungs",
        "- Epsilon: `a <= b + 1e-12*max(abs(a),abs(b))`; epsilon ties select free",
        "- Runtime accounting: an exception/replay failure is ABORT, never a monotonicity violation",
        "",
        "| Gate | Result | Measured | Threshold |",
        "|---|---|---:|---:|",
        f"| P1 monotonicity | {'PASS' if gates['P1']['pass'] else 'FAIL'} | {gates['P1']['violations']} violations; {gates['P1']['aborts']} aborts | 0 / 0 |",
        f"| P2 zero tax vs free | {'PASS' if gates['P2']['pass'] else 'FAIL'} | {gates['P2']['violations']} violations | 0 |",
        f"| P3 five-anchor PCHIP | {'PASS' if gates['P3']['pass'] else 'FAIL'} | per-projection below | median <=5%, p95 <=15% |",
        f"| P4 optimized overhead | {'PASS' if gates['P4']['pass'] else ('PROCEED' if gates['P4']['proceed'] else 'STOP')} | {gates['P4']['total_over_free']:.3f}x | pass <=1.35x; proceed <=1.6x |",
        f"| P5 fallback fraction | INFO | {gates['P5']['fallback_fraction']:.2%} ({gates['P5']['fallback_slices']}/768) | report |",
        "",
        "## PCHIP fit and fallback detail",
        "",
        "| Projection | Median | p95 | Max | P3 | Fallback |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for projection in PROJECTIONS:
        row = gates["P3"]["projections"][projection]
        lines.append(
            f"| {projection} | {row['median']:.2%} | {row['p95']:.2%} | "
            f"{row['max']:.2%} | {'PASS' if row['gate_pass'] else 'FAIL'} | "
            f"{row['fallback_fraction']:.2%} ({row['fallback_slices']}/256) |"
        )
    lines.extend([
        "",
        "Fallback acceptance is the registered held-out semantic: fit K28/K38/K48, score K33 and K43 independently at <=10% each, then fit accepted slices with all five anchors. Pilot P3 remains the stricter proper 16-rung non-anchor score.",
        "",
        "## Timing",
        "",
        f"- Free encode + free reconstruction/weight score: {gates['P4']['free_seconds'] / 60:.2f} min",
        f"- Optimized total including selection and winning replay: {gates['P4']['total_seconds'] / 60:.2f} min",
        "- P4 excludes checkpoint serialization; each cell records that time separately as `warm_state_write_seconds` and records raw `cell_wall_seconds`.",
        "",
        "## Serialization stamp",
        "",
        "```json",
        json.dumps(report["serialization"], indent=2, sort_keys=True),
        "```",
        "",
    ])
    if not report["burn_allowed"]:
        lines.extend([
            "The burn was not started because a mandatory pilot-2 gate or the P4 proceed band failed.",
            "",
        ])
    return "\n".join(lines)


def run_pilot2() -> int:
    _configure()
    if not torch.cuda.is_available():
        raise SystemExit("pilot-2 requires CUDA")
    device = torch.device("cuda:0")
    _, identity_record = load_layer_identity(PILOT2_LAYER)
    with COL_WEIGHTS.open("rb") as handle:
        all_col_weights = pickle.load(handle)
    observed = canonical_cb_col_weights_sha256(
        all_col_weights, identity_record["identity"]["col_weights_qnames"],
    )
    if observed != identity_record["identity"]["col_weights_sha256"]:
        raise AssertionError("layer-14 aggregate col-weight identity mismatch")
    model_to_shard, model_to_ckpt = _build_weight_map(str(SOURCE))
    scale_map = _build_fp8_scale_inv_map(str(SOURCE))
    report_path = RUN_ROOT / "PILOT2_REPORT.md"
    serialization_stamp = cb_serialization_context_stamp(
        CONTEXT, formats=[f"FP8_CB_K{k}" for k in RUNGS],
    )
    legacy_layer_path = (
        PILOT2_LAYER_SHARDS / f"layer_{PILOT2_LAYER:03d}.pkl"
    )
    gate_oracle = None
    gate_oracle_sha256 = None
    if legacy_layer_path.is_file():
        with legacy_layer_path.open("rb") as handle:
            legacy_layer = pickle.load(handle)
        legacy_identity = legacy_layer.get("content_identity") or {}
        legacy_gate = (legacy_layer.get("projections") or {}).get("gate_proj")
        if (
            legacy_identity.get("source_index_sha256")
            != sha256_file(SOURCE / "model.safetensors.index.json")
            or legacy_identity.get("col_weights_sha256") != observed
            or legacy_identity.get("by_layer_sha256") != identity_record["sha256"]
            or legacy_identity.get("serialization_context") != serialization_stamp
            or not isinstance(legacy_gate, Mapping)
            or set(legacy_gate.get("cells", {})) != set(RUNGS)
        ):
            raise AssertionError("surviving gate-projection oracle is not content matched")
        gate_oracle = legacy_gate["cells"]
        gate_oracle_sha256 = sha256_file(legacy_layer_path)
    content_identity = {
        "schema": SCHEMA,
        "profile": "A-FAST",
        "chain_version": CHAIN_VERSION,
        "layer": PILOT2_LAYER,
        "source": str(SOURCE.resolve()),
        "source_index_sha256": sha256_file(
            SOURCE / "model.safetensors.index.json"
        ),
        "col_weights": str(COL_WEIGHTS.resolve()),
        "col_weights_sha256": observed,
        "by_layer_path": identity_record["path"],
        "by_layer_sha256": identity_record["sha256"],
        "serialization_context": serialization_stamp,
        "implementation_sha256": {
            "campaign_tool": sha256_file(Path(__file__).resolve()),
            "minchain_module": sha256_file(
                Path(__file__).resolve().parents[1] / "prismaquant/cb_minchain.py"
            ),
        },
        "anchors": list(ANCHORS),
        "selection_metric": "per-expert weight_mse",
        "replay_metric": "production activation-QDQ output_mse",
        "epsilon_rtol": REL_EPSILON,
        "gate_numeric_oracle": (
            None if gate_oracle is None else {
                "path": str(legacy_layer_path),
                "sha256_before_run": gate_oracle_sha256,
                "semantics": "remeasure_then_require_exact_numeric_arrays",
            }
        ),
    }
    layer_record: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "layer": PILOT2_LAYER,
        "profile": "A-FAST",
        "chain_version": CHAIN_VERSION,
        "rungs": list(RUNGS),
        "projections": {},
        "content_key": _sha256_text(content_identity),
        "content_identity": content_identity,
    }
    PILOT2_CELL_SHARDS.mkdir(parents=True, exist_ok=True)
    rss_guard = RSSGuard(PILOT2_ROOT / "RSS_GUARD_EVIDENCE.json")
    rss_guard.start()
    try:
        for projection in PROJECTIONS:
            rss_guard.set_stage(f"{projection}:load-projection")
            data = load_projection(
                PILOT2_LAYER,
                projection,
                device=device,
                identity=identity_record["identity"],
                all_col_weights=all_col_weights,
                model_to_shard=model_to_shard,
                model_to_ckpt=model_to_ckpt,
                scale_map=scale_map,
            )
            layer_record["projections"][projection] = _measure_projection(
                layer=PILOT2_LAYER,
                projection=projection,
                data=data,
                rungs=RUNGS,
                rss_guard=rss_guard,
                oracle_cells=(
                    gate_oracle if projection == "gate_proj" else None
                ),
            )
            del data
            _reclaim()
            rss_guard.checkpoint()
            atomic_pickle(
                PILOT2_LAYER_SHARDS / f"layer_{PILOT2_LAYER:03d}.pkl",
                layer_record,
            )
    except Exception as error:
        rss_guard.stop()
        _reclaim()
        stop = {
            "schema": PILOT2_SCHEMA,
            "created_at": utc_now(),
            "decision": "ABORT",
            "burn_allowed": False,
            "error": f"{type(error).__name__}: {error}",
            "rss_guard": {
                "limit_bytes": RSS_LIMIT_BYTES,
                "peak_rss_bytes": rss_guard.peak_bytes,
                "evidence_path": str(PILOT2_ROOT / "RSS_GUARD_EVIDENCE.json"),
            },
            "gates": {
                "P1": {"pass": False, "violations": 0, "aborts": 1},
                "P2": {"pass": False, "violations": 0},
            },
        }
        atomic_json(PILOT2_ROOT / "PILOT2_REPORT.json", stop)
        atomic_text(report_path, "\n".join([
            "# DSV4 A-FAST Pilot-2", "", "- Result: **ABORT**",
            "- P1 numeric violation count: 0",
            "- P1 abort count: 1",
            f"- Error: `{type(error).__name__}: {error}`", "",
            "An abort is not counted as a monotonicity violation. The burn was not started.", "",
        ]))
        atomic_text(RUN_ROOT / "DSV4_CAMPAIGN_STOP.md", "\n".join([
            "# DSV4 Campaign Stop", "",
            f"- Stage: pilot-2 ({rss_guard.stage})",
            f"- Result: ABORT ({type(error).__name__})",
            f"- Error: `{error}`",
            f"- RSS peak: {rss_guard.peak_bytes / 1024**3:.3f} GiB",
            f"- Incremental cell shards: {len(list(PILOT2_CELL_SHARDS.glob('layer_014_*_K*.pkl')))}/63",
            "",
            "All completed cells were atomically persisted before the stop. The burn was not started.", "",
        ]))
        raise
    rss_guard.stop()
    fit = _fit_errors(layer_record)
    p1, p2, timing = _gate_counts(layer_record)
    p4_pass = timing["total_over_free"] <= 1.35
    p4_proceed = timing["total_over_free"] <= 1.60
    mandatory = p1 == 0 and p2 == 0 and fit["gate_pass"] and p4_proceed
    serialization = {
        "chain_schema": MINCHAIN_SCHEMA,
        "campaign_schema": SCHEMA,
        "chain_version": CHAIN_VERSION,
        "base_context": serialization_stamp,
        "selection_metric": "per-expert weight_mse",
        "interpolator": "monotone_pchip_fritsch_carlson",
        "anchors": list(ANCHORS),
        "epsilon_rtol": REL_EPSILON,
    }
    report = {
        "schema": PILOT2_SCHEMA,
        "created_at": utc_now(),
        "decision": "PASS" if mandatory else "STOP",
        "burn_allowed": mandatory,
        "gates": {
            "P1": {"pass": p1 == 0, "violations": p1, "aborts": 0},
            "P2": {"pass": p2 == 0, "violations": p2},
            "P3": {"pass": fit["gate_pass"], **fit},
            "P4": {
                "pass": p4_pass,
                "proceed": p4_proceed,
                "threshold_pass": 1.35,
                "threshold_proceed": 1.60,
                **timing,
            },
            "P5": {
                "fallback_slices": fit["fallback_slices"],
                "fallback_fraction": fit["fallback_fraction"],
            },
        },
        "serialization": serialization,
        "layer_shard": str(
            PILOT2_LAYER_SHARDS / f"layer_{PILOT2_LAYER:03d}.pkl"
        ),
        "incremental_shard_root": str(PILOT2_CELL_SHARDS),
        "incremental_shard_count": len(list(
            PILOT2_CELL_SHARDS.glob("layer_014_*_K*.pkl")
        )),
        "rss_guard": {
            "limit_bytes": RSS_LIMIT_BYTES,
            "peak_rss_bytes": rss_guard.peak_bytes,
        },
    }
    layer_record["fit"] = fit
    layer_record["pilot2_report"] = report
    atomic_pickle(
        PILOT2_LAYER_SHARDS / f"layer_{PILOT2_LAYER:03d}.pkl", layer_record,
    )
    atomic_json(PILOT2_ROOT / "PILOT2_REPORT.json", report)
    atomic_text(report_path, _pilot2_markdown(report))
    print(
        f"[afast] pilot2 {report['decision']} overhead="
        f"{timing['total_over_free']:.3f}x fallback={fit['fallback_fraction']:.2%}",
        flush=True,
    )
    if not mandatory:
        atomic_text(RUN_ROOT / "DSV4_CAMPAIGN_STOP.md", "\n".join([
            "# DSV4 Campaign Stop", "", "- Stage: pilot-2 gates",
            f"- Result: {report['decision']}",
            f"- P1: {report['gates']['P1']}",
            f"- P2: {report['gates']['P2']}",
            f"- P3 pass: {report['gates']['P3']['pass']}",
            f"- P4: {report['gates']['P4']['total_over_free']:.6f}x",
            "", "The registered pilot outcome does not authorize the burn.", "",
        ]))
    return 0 if mandatory else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("pilot2",))
    args = parser.parse_args()
    if args.command == "pilot2":
        return run_pilot2()
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
