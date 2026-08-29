"""Deterministic CB imatrix values derived from sensitivity-probe marginals.

The activation snapshot cache is intentionally row-capped because its job is
replay, not population statistics.  The sensitivity probe already accumulates
the full calibration-corpus second moment for every input column, so learned
codebook training should consume that resident-GPU reduction rather than
re-estimating it from sampled replay rows.

This module creates no cache.  It only turns the existing ``act_sq_sum`` and
``n_tokens_seen`` fields (or their packed-expert counterparts) into the same
qname -> tensor mapping every CB renderer already accepts.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Mapping

import torch


CB_IMATRIX_FROM_PROBE_SCHEMA = (
    "prismaquant.cb_imatrix.probe_act_sq_sum_over_tokens.v1"
)
CB_IMATRIX_VALUE_HASH_SCHEMA = "prismaquant.cb_imatrix_values.v1"


def _frame(digest: "hashlib._Hash", payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="little", signed=False))
    digest.update(payload)


def canonical_imatrix_sha256(
    col_weights: Mapping[str, torch.Tensor],
) -> str:
    """Hash sorted qnames, shapes, and canonical little-endian FP32 values."""

    if not isinstance(col_weights, Mapping) or not col_weights:
        raise ValueError("CB imatrix values must be a nonempty mapping")
    digest = hashlib.sha256()
    digest.update((CB_IMATRIX_VALUE_HASH_SCHEMA + "\0").encode("utf-8"))
    if any(not isinstance(name, str) or not name for name in col_weights):
        raise ValueError("CB imatrix qnames must be nonempty strings")
    for name in sorted(col_weights):
        value = torch.as_tensor(col_weights[name]).detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        if value.numel() == 0 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name}: CB imatrix must be finite and nonempty")
        if bool((value < 0).any()):
            raise ValueError(f"{name}: CB imatrix cannot contain negative values")
        shape = json.dumps(
            [int(dim) for dim in value.shape], separators=(",", ":")
        ).encode("ascii")
        raw = value.numpy().astype("<f4", copy=False).tobytes(order="C")
        _frame(digest, name.encode("utf-8"))
        _frame(digest, shape)
        _frame(digest, raw)
    return digest.hexdigest()


def _positive_count(value: object, *, where: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be a positive integer")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be a positive integer") from exc
    try:
        exact = float(value) == count
    except (TypeError, ValueError):
        exact = False
    if count <= 0 or not exact:
        raise ValueError(f"{where} must be positive, got {count}")
    return count


def _normalized_second_moment(
    numerator: object,
    denominator: object,
    *,
    where: str,
) -> torch.Tensor:
    count = _positive_count(denominator, where=f"{where} denominator")
    value = torch.as_tensor(numerator, dtype=torch.float32).detach().cpu()
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{where} must be finite and nonempty")
    if bool((value < 0).any()):
        raise ValueError(f"{where} cannot contain negative second moments")
    normalized = value / float(count)
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError(f"{where} normalization produced non-finite values")
    return normalized.contiguous()


def imatrix_from_probe_stats(
    stats: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Build full-corpus imatrix values from existing probe accumulators.

    Dense entries use ``act_sq_sum / n_tokens_seen``.  Packed populations use
    ``expert_act_sq_sum / expert_tokens`` independently for every expert and
    retain the encoder's accepted ``[experts, 1, in_features]`` spelling.
    A routed expert with zero observed tokens needs the repository's existing
    neutral-prior synthesis before this function; silently dividing it by one
    would turn missing calibration into evidence.
    """

    if not isinstance(stats, Mapping) or not stats:
        raise ValueError("probe stats must be a nonempty mapping")
    result: dict[str, torch.Tensor] = {}
    dense_entries = 0
    packed_entries = 0
    skipped_missing = 0
    for raw_name in sorted(stats):
        name = str(raw_name)
        raw = stats[raw_name]
        if not isinstance(raw, Mapping):
            continue
        expert_sum = raw.get("expert_act_sq_sum")
        expert_tokens = raw.get("expert_tokens")
        if expert_sum is not None or expert_tokens is not None:
            if expert_sum is None or expert_tokens is None:
                raise ValueError(
                    f"{name}: packed imatrix needs both expert_act_sq_sum "
                    "and expert_tokens"
                )
            sums = torch.as_tensor(expert_sum, dtype=torch.float32).detach().cpu()
            tokens = torch.as_tensor(expert_tokens).detach().cpu()
            if sums.ndim != 2 or tokens.ndim != 1 or sums.shape[0] != tokens.shape[0]:
                raise ValueError(
                    f"{name}: packed imatrix shapes differ: "
                    f"expert_act_sq_sum={tuple(sums.shape)}, "
                    f"expert_tokens={tuple(tokens.shape)}"
                )
            if tokens.dtype == torch.bool or not bool(torch.isfinite(tokens).all()):
                raise ValueError(f"{name}: expert_tokens must be finite integers")
            if not torch.equal(tokens, tokens.to(torch.int64).to(tokens.dtype)):
                raise ValueError(f"{name}: expert_tokens must be integers")
            if bool((tokens <= 0).any()):
                missing = torch.nonzero(tokens <= 0, as_tuple=False).flatten().tolist()
                raise ValueError(
                    f"{name}: packed imatrix has unrouted expert(s) {missing[:8]}; "
                    "apply the existing neutral-prior synthesis first"
                )
            if not bool(torch.isfinite(sums).all()) or bool((sums < 0).any()):
                raise ValueError(
                    f"{name}: expert_act_sq_sum must be finite and nonnegative"
                )
            result[name] = (
                sums / tokens.to(torch.float32).unsqueeze(-1)
            ).unsqueeze(1).contiguous()
            packed_entries += 1
            continue

        act_sum = raw.get("act_sq_sum")
        if act_sum is None:
            skipped_missing += 1
            continue
        value = _normalized_second_moment(
            act_sum,
            raw.get("n_tokens_seen"),
            where=f"{name}.act_sq_sum",
        ).reshape(-1)
        expected = int(raw.get("in_features", value.numel()) or value.numel())
        if value.numel() != expected:
            raise ValueError(
                f"{name}: act_sq_sum has {value.numel()} entries, expected "
                f"in_features={expected}"
            )
        result[name] = value
        dense_entries += 1

    if not result:
        raise ValueError("probe contains no usable act_sq_sum imatrix entries")
    provenance: dict[str, object] = {
        "schema": CB_IMATRIX_FROM_PROBE_SCHEMA,
        "dense_entries": dense_entries,
        "packed_entries": packed_entries,
        "skipped_missing_entries": skipped_missing,
        "value_sha256": canonical_imatrix_sha256(result),
    }
    return result, provenance


def imatrix_from_probe_file(
    path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Read an existing trusted ``probe.pkl`` and derive its imatrix values."""

    probe_path = Path(path)
    with probe_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{probe_path}: probe must be a mapping")
    raw_stats = payload.get("stats", payload)
    if not isinstance(raw_stats, Mapping):
        raise ValueError(f"{probe_path}: probe stats must be a mapping")
    stats = {
        str(name): value
        for name, value in raw_stats.items()
        if name != "meta" and isinstance(value, Mapping)
    }
    values, provenance = imatrix_from_probe_stats(stats)
    meta = payload.get("meta", {}) if isinstance(payload, Mapping) else {}
    calibration_hash = None
    if isinstance(meta, Mapping):
        for key in ("calib_hash", "calibration_hash", "calib_sha256"):
            if meta.get(key):
                calibration_hash = str(meta[key])
                break
    provenance = {
        **provenance,
        "probe_path": str(probe_path.resolve()),
        **(
            {"calibration_hash": calibration_hash}
            if calibration_hash is not None
            else {}
        ),
    }
    return values, provenance


__all__ = [
    "CB_IMATRIX_FROM_PROBE_SCHEMA",
    "CB_IMATRIX_VALUE_HASH_SCHEMA",
    "canonical_imatrix_sha256",
    "imatrix_from_probe_file",
    "imatrix_from_probe_stats",
]
