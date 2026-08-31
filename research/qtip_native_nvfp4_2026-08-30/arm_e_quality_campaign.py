"""Matched Arm-E quality campaign for canonical E2M1 trellis wires.

This is an explicit research runner, not a format registration, exporter, or
serving path.  It compares rotated BlockLDL plus the canonical trellis wire
against scalar native NVFP4 and stock Arm C on either the accepted Qwen
one-Linear activation contract or the finalized GLM BF16/importance corpus.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import socket
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from prismaquant import trellis_bf16_corpus as CORPUS_MODULE
from prismaquant import cb_imatrix as IMATRIX_MODULE
from prismaquant import trellis_encoder as ENCODER_MODULE
from prismaquant import trellis_formats as FORMATS_MODULE
from prismaquant import trellis_footprint as FOOTPRINT_MODULE
from prismaquant import trellis_producer as TRELLIS_PRODUCER_MODULE
from prismaquant import trellis_rate_surface as RATE_MODULE
from prismaquant import trellis_wire as WIRE_MODULE
from prismaquant.trellis_bf16_corpus import FinalizedBF16Corpus
from prismaquant.trellis_formats import (
    E2M1_FAMILY,
    LAYOUT_TIGHT_OFFSETS,
    TRELLIS_WIRE_SCHEMA,
    get_trellis_family,
    native_code_value,
)
from prismaquant.trellis_footprint import trellis_tensor_payload_breakdown
from prismaquant.trellis_rate_surface import uniform_column_schedule
from prismaquant.trellis_wire import TrellisWire, decode_values_torch


NATIVE = importlib.import_module(
    "research.qtip_native_nvfp4_2026-08-30.native_nvfp4_ldlq"
)
ARM_E = importlib.import_module(
    "research.qtip_native_nvfp4_2026-08-30."
    "trellis_online_hadamard_producer"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ATOMIC_PATH = (
    _REPO_ROOT
    / "research/trellis_e2m1_highrate_2026-08-30/atomic_publication.py"
)
_ATOMIC_SPEC = importlib.util.spec_from_file_location(
    "arm_e_numeric_atomic_publication", _ATOMIC_PATH
)
if _ATOMIC_SPEC is None or _ATOMIC_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import atomic publication helpers from {_ATOMIC_PATH}")
ATOMIC = importlib.util.module_from_spec(_ATOMIC_SPEC)
_ATOMIC_SPEC.loader.exec_module(ATOMIC)


MANIFEST_SCHEMA = "prismaquant.research.arm_e_quality_campaign_manifest.v1"
RESULT_SCHEMA = "prismaquant.research.arm_e_quality_tensor_result.v1"
RECEIPT_SCHEMA = "prismaquant.research.arm_e_quality_campaign_receipt.v1"
PREFLIGHT_SCHEMA = "prismaquant.research.arm_e_quality_campaign_preflight.v1"
CLAIM_SCHEMA = "prismaquant.research.arm_e_quality_campaign_claim.v1"
QWEN_MODE = "qwen_one_linear"
GLM_MODE = "glm_corpus"
MODES = frozenset({QWEN_MODE, GLM_MODE})
RATE_POLICY = "largest_q256_complete_wire_not_above_min_native_A_C_bytes_v1"
SCHEDULE_POLICY = "uniform_column_schedule_bresenham_v1"
ALPHABET_POLICY = "canonical_full_e2m1_value_order_rate3_v1"
INPUT_BLOCK_POLICY = "largest_power_of_two_divisor_capped_v1"
GLM_ACTIVATION_UNAVAILABLE_REASON = (
    "finalized trellis.bf16_corpus.v2 contains BF16 weights and activation "
    "second moments, not activation rows; synthetic algebra witnesses are "
    "excluded from quality metrics"
)
DENSE_BLOCKLDL_MAX_COLUMNS = 4096
QWEN_ACTIVATION_AVAILABLE_REASON = (
    "measured from the same preprocessed calibration activation rows used "
    "to construct the matched Hessian"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CAMPAIGN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_ROOT_FIELDS = frozenset({
    "schema", "campaign_id", "mode", "research_opt_in", "execution",
    "output", "input", "recipe", "seeds",
})
_EXECUTION_FIELDS = frozenset({
    "device", "host", "container_identity", "prismaquant_checkout",
    "prismaquant_commit",
})
_OUTPUT_FIELDS = frozenset({"root", "durable_root_uri"})
_RECIPE_FIELDS = frozenset({
    "family", "layout", "rate_policy", "schedule_policy",
    "alphabet_policy", "scale_rule", "terminal_metric_mode",
    "input_block_policy", "max_input_block_size", "output_block_size",
    "sb_chunk", "determinism_mode", "tailbite_candidates", "backend",
    "point_route", "buffer_blocks", "damp_fraction",
    "glm_algebra_witness_rows",
})
_SEED_FIELDS = frozenset({"label", "input_seed", "output_seed"})
_QWEN_INPUT_FIELDS = frozenset({
    "kind", "model_id", "weight_path", "weight_key", "activations_path",
    "activations_key", "calibration_manifest",
})
_GLM_INPUT_FIELDS = frozenset({
    "kind", "corpus_manifest", "selected_tensors", "limit",
})
_TENSOR_RESULT_BODY_FIELDS = frozenset({
    "schema", "status", "campaign_claim_identity_sha256",
    "source_closure_identity_sha256", "mode", "tensor", "hessian_contract",
    "metric_availability", "rate_plan", "transform_geometry", "controls",
    "arm_e_by_seed", "feasibility_telemetry", "format_registry_entries_created",
    "runtime_pin_changed", "production_contract_changed", "producer_eligible",
})
_CAMPAIGN_RECEIPT_BODY_FIELDS = frozenset({
    "schema", "status", "campaign_claim_identity_sha256", "manifest", "mode",
    "input_provenance", "source_closure", "execution", "acceptance_contract",
    "summary", "published_members", "publication", "claim_boundary",
})

_FIXED_RECIPE = {
    "family": E2M1_FAMILY,
    "layout": LAYOUT_TIGHT_OFFSETS,
    "rate_policy": RATE_POLICY,
    "schedule_policy": SCHEDULE_POLICY,
    "alphabet_policy": ALPHABET_POLICY,
    "scale_rule": "static_6",
    "input_block_policy": INPUT_BLOCK_POLICY,
    "output_block_size": 256,
    "determinism_mode": "on",
    "tailbite_candidates": 4,
    "point_route": "full",
    "damp_fraction": 1.0,
}

ACCEPTANCE_CONTRACT: Mapping[str, object] = {
    "schema": "prismaquant.research.arm_e_quality_acceptance.v1",
    "correctness": {
        "canonical_same_byte_reopen_reserialize_decode": True,
        "complete_wire_bytes_not_above_both_native_controls": True,
        "cross_terminal_feedback_nonzero_where_declared": True,
        "serving_or_performance_claim_permitted": False,
    },
    "qwen": {
        "advance": (
            "primary Arm E activation-output and regularized-H NSSE are "
            "both <= stock Arm C at lower exact bpw"
        ),
        "robust": (
            "three predeclared seeds: median deltas versus C >= 0 dB and "
            "worst activation delta >= -0.1 dB"
        ),
        "promising": (
            "both metrics >= +1 dB over scalar and within 0.25 dB of C"
        ),
    },
    "glm": {
        "metric": "raw_importance_weighted_snr_db",
        "minimum": (
            "separately dense/routed: median E-A and E-C > 0 dB; wins "
            ">= 7/9 dense and >= 18/24 routed"
        ),
        "strong": (
            "separately dense/routed: both medians >= +0.25 dB, both "
            "one-sided paired-bootstrap 95% lower bounds > 0, wins >= "
            "8/9 dense and >= 20/24 routed"
        ),
        "model_quality_claim_permitted": False,
    },
    "feasibility_pilot": {
        "peak_allocated_bytes_max": 24 * 1024**3,
        "largest_tensor_seconds_max": 1800.0,
        "projected_full_corpus_seconds_max": 6 * 3600.0,
        "cpu_factorization_fallback_permitted": False,
        "in_process_profile_required": True,
        "both_host_netdata_required": True,
        "gb10_power_envelope_watts": 140.0,
        "gpu_utilization_is_diagnostic": False,
    },
}


def _canonical_bytes(value: object, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            indent=2 if pretty else None,
            sort_keys=True,
            separators=None if pretty else (",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _identity_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON member {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_object)
    except Exception as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _require_exact_fields(
    value: Any, expected: frozenset[str], *, where: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"{where} has missing={missing}, unknown={unknown}")
    return value


def _plain_int(value: Any, *, where: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{where} must be a JSON integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{where} must be >= {minimum}")
    return value


def _nonempty_string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a nonempty string")
    if any(character in value for character in ("\0", "\n", "\r")):
        raise ValueError(f"{where} must be one line")
    return value.strip()


def validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the closed campaign manifest."""

    root = _require_exact_fields(value, _ROOT_FIELDS, where="manifest")
    if root["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"manifest.schema must be {MANIFEST_SCHEMA!r}")
    campaign_id = _nonempty_string(root["campaign_id"], where="campaign_id")
    if not _CAMPAIGN_ID_RE.fullmatch(campaign_id):
        raise ValueError("campaign_id has non-portable characters")
    mode = root["mode"]
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    if root["research_opt_in"] != ARM_E.RESEARCH_OPT_IN:
        raise ValueError(
            f"research_opt_in must equal {ARM_E.RESEARCH_OPT_IN!r}"
        )

    execution = dict(_require_exact_fields(
        root["execution"], _EXECUTION_FIELDS, where="execution"
    ))
    for field in ("device", "host", "prismaquant_checkout"):
        execution[field] = _nonempty_string(execution[field], where=f"execution.{field}")
    execution["container_identity"] = NATIVE.validate_container_identity(
        _nonempty_string(
            execution["container_identity"], where="execution.container_identity"
        )
    )
    if not isinstance(execution["prismaquant_commit"], str) or not _COMMIT_RE.fullmatch(
        execution["prismaquant_commit"]
    ):
        raise ValueError("execution.prismaquant_commit must be 40 lowercase hex")

    output = dict(_require_exact_fields(
        root["output"], _OUTPUT_FIELDS, where="output"
    ))
    output["root"] = _nonempty_string(output["root"], where="output.root")
    output["durable_root_uri"] = _nonempty_string(
        output["durable_root_uri"], where="output.durable_root_uri"
    ).rstrip("/")

    recipe = dict(_require_exact_fields(
        root["recipe"], _RECIPE_FIELDS, where="recipe"
    ))
    for field, expected in _FIXED_RECIPE.items():
        if type(recipe[field]) is not type(expected) or recipe[field] != expected:
            raise ValueError(f"recipe.{field} must be exactly {expected!r}")
    if recipe["terminal_metric_mode"] not in {"diag_block_D", "qtip_frobenius"}:
        raise ValueError("recipe.terminal_metric_mode is not implemented")
    maximum = _plain_int(
        recipe["max_input_block_size"],
        where="recipe.max_input_block_size",
        minimum=256,
    )
    if maximum & (maximum - 1):
        raise ValueError("recipe.max_input_block_size must be a power of two")
    for field in ("sb_chunk", "buffer_blocks", "glm_algebra_witness_rows"):
        recipe[field] = _plain_int(recipe[field], where=f"recipe.{field}", minimum=1)
    if recipe["backend"] not in {"eager", "triton"}:
        raise ValueError("recipe.backend must be 'eager' or 'triton'")

    seeds_raw = root["seeds"]
    if not isinstance(seeds_raw, list) or not 1 <= len(seeds_raw) <= 3:
        raise ValueError("seeds must contain one to three predeclared seeds")
    seeds: list[dict[str, Any]] = []
    labels: set[str] = set()
    pairs: set[tuple[int, int]] = set()
    for index, raw in enumerate(seeds_raw):
        seed = dict(_require_exact_fields(
            raw, _SEED_FIELDS, where=f"seeds[{index}]"
        ))
        label = _nonempty_string(seed["label"], where=f"seeds[{index}].label")
        if not _CAMPAIGN_ID_RE.fullmatch(label) or label in labels:
            raise ValueError("seed labels must be unique portable identifiers")
        pair = (
            _plain_int(seed["input_seed"], where=f"seeds[{index}].input_seed", minimum=0),
            _plain_int(seed["output_seed"], where=f"seeds[{index}].output_seed", minimum=0),
        )
        if pair in pairs:
            raise ValueError("seed pairs must be unique")
        labels.add(label)
        pairs.add(pair)
        seeds.append({"label": label, "input_seed": pair[0], "output_seed": pair[1]})

    if mode == QWEN_MODE:
        input_value = dict(_require_exact_fields(
            root["input"], _QWEN_INPUT_FIELDS, where="input"
        ))
        if input_value["kind"] != QWEN_MODE:
            raise ValueError(f"input.kind must equal {QWEN_MODE!r}")
        for field in (
            "model_id", "weight_path", "activations_path", "calibration_manifest"
        ):
            input_value[field] = _nonempty_string(
                input_value[field], where=f"input.{field}"
            )
        for field in ("weight_key", "activations_key"):
            if input_value[field] is not None:
                input_value[field] = _nonempty_string(
                    input_value[field], where=f"input.{field}"
                )
    else:
        input_value = dict(_require_exact_fields(
            root["input"], _GLM_INPUT_FIELDS, where="input"
        ))
        if input_value["kind"] != GLM_MODE:
            raise ValueError(f"input.kind must equal {GLM_MODE!r}")
        input_value["corpus_manifest"] = _nonempty_string(
            input_value["corpus_manifest"], where="input.corpus_manifest"
        )
        selected = input_value["selected_tensors"]
        if not isinstance(selected, list) or any(
            not isinstance(name, str) or not name for name in selected
        ):
            raise ValueError("input.selected_tensors must be a string array")
        if len(set(selected)) != len(selected):
            raise ValueError("input.selected_tensors contains duplicates")
        input_value["selected_tensors"] = list(selected)
        limit = input_value["limit"]
        if limit is not None:
            input_value["limit"] = _plain_int(limit, where="input.limit", minimum=1)

    return {
        "schema": MANIFEST_SCHEMA,
        "campaign_id": campaign_id,
        "mode": mode,
        "research_opt_in": ARM_E.RESEARCH_OPT_IN,
        "execution": execution,
        "output": output,
        "input": input_value,
        "recipe": recipe,
        "seeds": seeds,
    }


def load_manifest(path: str | Path) -> tuple[dict[str, Any], dict[str, object]]:
    source = Path(path).resolve(strict=True)
    normalized = validate_manifest(_read_json(source))
    provenance = {
        "path": str(source),
        "file_sha256": _file_sha256(source),
        "semantic_identity_sha256": _identity_sha256(normalized),
    }
    return normalized, provenance


def native_payload_bytes(shape: Sequence[int]) -> int:
    rows, columns = map(int, shape)
    if rows <= 0 or columns <= 0 or columns % 16:
        raise ValueError("native NVFP4 shape must be positive and group-16 aligned")
    return rows * (columns // 2) + rows * (columns // 16) + 8


def canonical_highrate_alphabets(schedule: Sequence[int]) -> dict[int, tuple[int, ...]]:
    """Return the one non-learned high-rate E2M1 alphabet."""

    shaped = {int(rate) for rate in schedule if int(rate) < 4}
    if shaped != {3}:
        raise ValueError(
            "the exact native-byte frontier left the canonical rate-3/4 "
            f"campaign domain; shaped rates={sorted(shaped)}"
        )
    ordered = tuple(sorted(
        range(16),
        key=lambda code: (native_code_value(E2M1_FAMILY, code), code),
    ))
    return {3: ordered}


def exact_native_budget_frontier(
    shape: Sequence[int], *, arm_a_bytes: int, arm_c_bytes: int
) -> dict[str, object]:
    """Select the highest complete-wire q256 fitting both native controls."""

    rows, columns = map(int, shape)
    budget = min(
        _plain_int(arm_a_bytes, where="arm_a_bytes", minimum=1),
        _plain_int(arm_c_bytes, where="arm_c_bytes", minimum=1),
    )
    family = get_trellis_family(E2M1_FAMILY)
    lower, upper = family.mathematical_q256_bounds
    for q256 in range(upper, lower - 1, -1):
        schedule = uniform_column_schedule(columns, q256, family=family)
        try:
            alphabets = canonical_highrate_alphabets(schedule)
        except ValueError:
            continue
        footprint = trellis_tensor_payload_breakdown(
            (rows, columns),
            family=family,
            body_rate_q256=q256,
            layout=LAYOUT_TIGHT_OFFSETS,
            schedule=schedule,
            alphabets=alphabets,
        )
        if int(footprint["total_bytes"]) <= budget:
            return {
                "schema": "prismaquant.research.arm_e_exact_byte_frontier.v1",
                "policy": RATE_POLICY,
                "shape": [rows, columns],
                "arm_a_bytes": int(arm_a_bytes),
                "arm_c_bytes": int(arm_c_bytes),
                "budget_bytes": budget,
                "selected_body_rate_q256": q256,
                "schedule_policy": SCHEDULE_POLICY,
                "schedule": list(schedule),
                "alphabet_policy": ALPHABET_POLICY,
                "alphabets": {
                    str(rate): list(codes)
                    for rate, codes in sorted(alphabets.items())
                },
                "footprint": footprint,
            }
    raise ValueError("no canonical high-rate trellis wire fits the native budget")


def automatic_input_block_size(columns: int, cap: int) -> int:
    columns = _plain_int(columns, where="columns", minimum=256)
    cap = _plain_int(cap, where="cap", minimum=256)
    candidate = 1 << min(columns, cap).bit_length() - 1
    while columns % candidate:
        candidate //= 2
    if candidate < 256:
        raise ValueError("input width has no supported >=256 Hadamard divisor")
    return candidate


def dense_producer_feasibility(shape: Sequence[int]) -> dict[str, object]:
    """Plan the current dense-H producer without allocating its matrices.

    The transformed diagonal GLM Hessian is exactly block diagonal at the
    declared input-Hadamard geometry, but the current producer still accepts
    and factors one dense K-by-K tensor.  K>4096 therefore fails closed until
    the separately reviewed block-structured implementation exists.
    """

    rows, columns = map(int, shape)
    hessian_bytes = 4 * columns * columns
    weight_bytes = 4 * rows * columns
    # Planning estimates only: enough to expose scale, never a measured peak
    # or runtime prediction.
    conservative_dense_working_set_bytes = 10 * hessian_bytes + 3 * weight_bytes
    cholesky_flop_order = columns**3 // 3
    executable = columns <= DENSE_BLOCKLDL_MAX_COLUMNS
    return {
        "schema": "prismaquant.research.arm_e_dense_producer_feasibility.v1",
        "shape": [rows, columns],
        "dense_fp32_hessian_bytes": hessian_bytes,
        "conservative_dense_working_set_bytes": conservative_dense_working_set_bytes,
        "cholesky_flop_order_estimate": cholesky_flop_order,
        "estimate_is_measurement": False,
        "current_dense_producer_max_columns": DENSE_BLOCKLDL_MAX_COLUMNS,
        "current_dense_producer_executable": executable,
        "refusal_reason": None if executable else (
            "K exceeds the declared dense-H/whole-matrix BlockLDL feasibility "
            "boundary; transformed diagonal H is block diagonal by the input "
            "Hadamard block, but that exact structure-aware factorization is "
            "not implemented in the current producer"
        ),
    }


def require_dense_producer_feasible(shape: Sequence[int]) -> dict[str, object]:
    plan = dense_producer_feasibility(shape)
    if not plan["current_dense_producer_executable"]:
        raise ValueError(
            f"refusing dense Arm E producer for shape {tuple(shape)}: "
            f"{plan['refusal_reason']}"
        )
    return plan


def regularized_glm_diagonal(
    importance: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, object]]:
    value = importance.detach().float().reshape(-1).clone()
    if not value.numel() or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError("GLM importance must be finite, nonnegative, and nonempty")
    dead = value <= 0
    alive = ~dead
    mean = (
        value[alive].mean() if bool(alive.any()) else value.new_ones(())
    ).clamp_min(1.0e-12)
    if bool(dead.any()):
        value[dead] = 1.0
    realized = float(mean.item())
    value.add_(realized)
    return value, {
        "construction": (
            "activation_second_moment_diagonal_with_production_dead_channel_"
            "convention_and_damp_1p0_times_alive_mean"
        ),
        "damp_fraction": 1.0,
        "realized_diagonal_damping": realized,
        "dead_channel_count": int(dead.sum().item()),
        "activation_rows_reconstructed_or_fabricated": False,
    }


def _snr(nsse: float) -> float:
    return -10.0 * math.log10(max(nsse, 1.0e-300))


def quality_metrics(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    *,
    regularized_hessian: torch.Tensor | None = None,
    regularized_diagonal: torch.Tensor | None = None,
    activations: torch.Tensor | None = None,
    raw_importance: torch.Tensor | None = None,
) -> dict[str, object]:
    """Score in original basis and make metric availability literal."""

    w = weight.float()
    q = reconstruction.to(device=w.device, dtype=torch.float32)
    if q.shape != w.shape:
        raise ValueError("reconstruction shape differs from weight")
    error = w - q
    weight_num = float(error.square().sum().item())
    weight_den = float(w.square().sum().item())
    weight_nsse = weight_num / max(weight_den, 1.0e-30)
    result: dict[str, object] = {
        "weight": {
            "nsse": weight_nsse,
            "snr_db": _snr(weight_nsse),
            "numerator": weight_num,
            "denominator": weight_den,
        }
    }
    if (regularized_hessian is None) == (regularized_diagonal is None):
        raise ValueError("supply exactly one regularized Hessian representation")
    if regularized_diagonal is not None:
        diagonal = regularized_diagonal.to(device=w.device, dtype=torch.float32)
        hnum = float((error.square() * diagonal).sum().item())
        hden = float((w.square() * diagonal).sum().item())
        representation = "diagonal"
    else:
        hessian = regularized_hessian.to(device=w.device, dtype=torch.float32)
        hnum = float(((error @ hessian) * error).sum().item())
        hden = float(((w @ hessian) * w).sum().item())
        representation = "dense"
    hnsse = hnum / max(hden, 1.0e-30)
    result["regularized_hessian_proxy"] = {
        "available": True,
        "basis": "untransformed_original_linear",
        "representation": representation,
        "nsse": hnsse,
        "snr_db": _snr(hnsse),
        "numerator": hnum,
        "denominator": hden,
    }
    if activations is None:
        result["activation_output"] = {
            "available": False,
            "reason": GLM_ACTIVATION_UNAVAILABLE_REASON,
        }
    else:
        x = activations.to(device=w.device, dtype=torch.float32)
        output_error = x @ error.T
        output = x @ w.T
        numerator = float(output_error.square().sum().item())
        denominator = float(output.square().sum().item())
        nsse = numerator / max(denominator, 1.0e-30)
        result["activation_output"] = {
            "available": True,
            "reason": QWEN_ACTIVATION_AVAILABLE_REASON,
            "nsse": nsse,
            "snr_db": _snr(nsse),
            "numerator": numerator,
            "denominator": denominator,
        }
    if raw_importance is not None:
        raw = raw_importance.to(device=w.device, dtype=torch.float32).reshape(-1)
        numerator = float((error.square() * raw).sum().item())
        denominator = float((w.square() * raw).sum().item())
        nsse = numerator / max(denominator, 1.0e-30)
        result["raw_importance_weighted"] = {
            "available": True,
            "nsse": nsse,
            "snr_db": _snr(nsse),
            "numerator": numerator,
            "denominator": denominator,
        }
    else:
        result["raw_importance_weighted"] = {
            "available": False,
            "reason": "raw importance is defined only by the GLM corpus contract",
        }
    return result


def _source_paths() -> dict[str, Path]:
    return {
        "research/qtip_native_nvfp4_2026-08-30/arm_e_quality_campaign.py": Path(__file__).resolve(),
        "research/qtip_native_nvfp4_2026-08-30/native_nvfp4_ldlq.py": Path(NATIVE.__file__).resolve(),
        "research/qtip_native_nvfp4_2026-08-30/trellis_online_hadamard_producer.py": Path(ARM_E.__file__).resolve(),
        "research/trellis_e2m1_highrate_2026-08-30/atomic_publication.py": _ATOMIC_PATH.resolve(),
        "prismaquant/trellis_encoder.py": Path(ENCODER_MODULE.__file__).resolve(),
        "prismaquant/trellis_formats.py": Path(FORMATS_MODULE.__file__).resolve(),
        "prismaquant/trellis_wire.py": Path(WIRE_MODULE.__file__).resolve(),
        "prismaquant/trellis_footprint.py": Path(FOOTPRINT_MODULE.__file__).resolve(),
        "prismaquant/trellis_rate_surface.py": Path(RATE_MODULE.__file__).resolve(),
        "prismaquant/trellis_producer.py": Path(TRELLIS_PRODUCER_MODULE.__file__).resolve(),
        "prismaquant/trellis_bf16_corpus.py": Path(CORPUS_MODULE.__file__).resolve(),
        "prismaquant/cb_imatrix.py": Path(IMATRIX_MODULE.__file__).resolve(),
    }


_IMPORTED_SOURCE_SHA256 = {
    relative: _file_sha256(path) for relative, path in _source_paths().items()
}


def require_source_closure(checkout: str | Path) -> dict[str, object]:
    root = Path(checkout).resolve()
    current_paths = _source_paths()
    for relative, path in current_paths.items():
        expected_path = (root / relative).resolve()
        if path != expected_path:
            raise ValueError(f"imported source is outside checkout: {relative}={path}")
        current = _file_sha256(path)
        if current != _IMPORTED_SOURCE_SHA256[relative]:
            raise ValueError(f"source changed since module import: {relative}")
    pin_path = root / "prismaquant/gridbook_runtime/gridbook_runtime_pin.json"
    pin = _read_json(pin_path)
    required_features = pin.get("required_abi_features")
    if not isinstance(required_features, dict):
        raise ValueError("Gridbook pin has no required_abi_features object")
    transform_abi = any(
        any(token in str(key).lower() for token in ("hadamard", "rotation", "qtip"))
        for key in required_features
    )
    body: dict[str, object] = {
        "source_sha256": dict(sorted(_IMPORTED_SOURCE_SHA256.items())),
        "qtip_source_audit": {
            "repository": ARM_E.QTIP_REPOSITORY,
            "commit": ARM_E.QTIP_PINNED_COMMIT,
            "source_sha256": dict(sorted(ARM_E.QTIP_SOURCE_FILES.items())),
            "runtime_or_wire_imported": False,
        },
        "gridbook_runtime_pin": {
            "path": "prismaquant/gridbook_runtime/gridbook_runtime_pin.json",
            "file_sha256": _file_sha256(pin_path),
            "content": pin,
            "online_transform_abi_declared": transform_abi,
            "runtime_executed": False,
        },
    }
    return {**body, "identity_sha256": _identity_sha256(body)}


def validated_source_closure(
    checkout: str | Path, expected_commit: str
) -> dict[str, object]:
    """Bind the clean checkout and the import-time source closure together."""

    checkout_record = NATIVE.validate_prismaquant_checkout(
        checkout, expected_commit
    )
    closure = dict(require_source_closure(checkout))
    closure.pop("identity_sha256")
    body = {**closure, "checkout": checkout_record}
    return {**body, "identity_sha256": _identity_sha256(body)}


@dataclass(frozen=True)
class InputUnit:
    name: str
    population: str
    shape: tuple[int, int]
    source_weight_sha256: str | None


@dataclass(frozen=True)
class PreflightInputs:
    mode: str
    units: tuple[InputUnit, ...]
    provenance: Mapping[str, object]
    corpus: FinalizedBF16Corpus | None = None


def _selected_glm_entries(
    corpus: FinalizedBF16Corpus, selected: Sequence[str], limit: int | None
):
    entries = list(corpus.entries)
    if selected:
        by_name = {entry.name: entry for entry in entries}
        unknown = sorted(set(selected) - set(by_name))
        if unknown:
            raise ValueError(f"unknown selected GLM tensors: {unknown}")
        entries = [by_name[name] for name in selected]
    if limit is not None:
        entries = entries[:limit]
    if not entries:
        raise ValueError("GLM selection is empty")
    return tuple(entries)


def preflight_inputs(manifest: Mapping[str, Any]) -> PreflightInputs:
    source = manifest["input"]
    if manifest["mode"] == QWEN_MODE:
        weight_path = Path(source["weight_path"]).resolve(strict=True)
        activations_path = Path(source["activations_path"]).resolve(strict=True)
        calibration_path = Path(source["calibration_manifest"]).resolve(strict=True)
        weight = NATIVE._load(str(weight_path), source["weight_key"])
        activations = NATIVE._load(str(activations_path), source["activations_key"])
        if weight.ndim != 2 or activations.ndim < 2:
            raise ValueError("Qwen inputs must be weight [out,in] and activations [...,in]")
        rows, columns = map(int, weight.shape)
        if columns % 256 or rows % 256 or int(activations.shape[-1]) != columns:
            raise ValueError("Qwen Arm E shape must be row/input 256 aligned")
        calibration = NATIVE.validate_calibration_manifest(calibration_path)
        provenance = {
            "kind": QWEN_MODE,
            "model_id": source["model_id"],
            "weight": {
                "path": str(weight_path),
                "key": source["weight_key"],
                "file_sha256": _file_sha256(weight_path),
                "tensor_sha256": NATIVE.tensor_sha256(weight),
            },
            "activations": {
                "path": str(activations_path),
                "key": source["activations_key"],
                "file_sha256": _file_sha256(activations_path),
                "tensor_sha256": NATIVE.tensor_sha256(activations),
            },
            "calibration": calibration,
        }
        unit = InputUnit(
            name=str(source["weight_key"] or weight_path.name),
            population="qwen_one_linear",
            shape=(rows, columns),
            source_weight_sha256=None,
        )
        return PreflightInputs(QWEN_MODE, (unit,), provenance)

    corpus = CORPUS_MODULE.load_finalized_bf16_corpus(source["corpus_manifest"])
    entries = _selected_glm_entries(
        corpus, source["selected_tensors"], source["limit"]
    )
    units = tuple(InputUnit(
        name=entry.name,
        population=entry.population,
        shape=entry.source_weight_shape,
        source_weight_sha256=entry.source_weight_sha256,
    ) for entry in entries)
    provenance = {
        "kind": GLM_MODE,
        "manifest_path": str(corpus.manifest_path),
        "manifest_file_sha256": _file_sha256(corpus.manifest_path),
        "artifact_path": str(corpus.artifact_path),
        "artifact_file_sha256": str(corpus.manifest["file_sha256"]),
        "calibration": corpus.manifest["calibration"],
        "importance_identity": corpus.manifest["importance_identity"],
        "selected_tensors": [unit.name for unit in units],
        "full_census_selected": len(units) == len(corpus.entries),
    }
    return PreflightInputs(GLM_MODE, units, provenance, corpus)


def preflight_report(
    manifest: Mapping[str, Any],
    manifest_provenance: Mapping[str, object],
    inputs: PreflightInputs,
    source_closure: Mapping[str, object],
) -> dict[str, object]:
    plans = []
    for shape in sorted(set(unit.shape for unit in inputs.units)):
        native_bytes = native_payload_bytes(shape)
        plan = exact_native_budget_frontier(
            shape, arm_a_bytes=native_bytes, arm_c_bytes=native_bytes
        )
        plans.append({
            "shape": list(shape),
            "native_control_bytes": native_bytes,
            "selected_body_rate_q256": plan["selected_body_rate_q256"],
            "wire_bytes": plan["footprint"]["total_bytes"],
            "wire_bpw": plan["footprint"]["exact_bpw"],
            "input_block_size": automatic_input_block_size(
                shape[1], manifest["recipe"]["max_input_block_size"]
            ),
            "dense_producer_feasibility": dense_producer_feasibility(shape),
        })
    refused = [
        plan for plan in plans
        if not plan["dense_producer_feasibility"]["current_dense_producer_executable"]
    ]
    full_glm_requested = bool(
        inputs.mode == GLM_MODE
        and inputs.provenance.get("full_census_selected") is True
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "validated_no_quality_campaign_executed",
        "execution_readiness": (
            "refused_current_dense_producer_shape"
            if refused else "shape_contract_ready"
        ),
        "manifest": dict(manifest_provenance),
        "mode": manifest["mode"],
        "selected_tensor_count": len(inputs.units),
        "input_provenance": inputs.provenance,
        "shape_plans": plans,
        "refused_shape_plans": refused,
        "full_glm_census_requested": full_glm_requested,
        "full_glm_census_executable": full_glm_requested and not refused,
        "source_closure_identity_sha256": source_closure["identity_sha256"],
        "claim_boundary": {
            "quality_measured": False,
            "gridbook_runtime_executed": False,
            "served": False,
            "performance_claim": False,
            "completed_campaign": False,
        },
    }


def _slug(name: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")[-64:]
    return f"{readable or 'tensor'}-{hashlib.sha256(name.encode()).hexdigest()[:12]}"


def _safe_output(root: Path, relative: str) -> Path:
    target = (root / relative).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    canonical = target.parent.resolve(strict=True) / target.name
    try:
        canonical.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"output escapes campaign root: {relative}") from exc
    return canonical


def _publish_or_verify_bytes(destination: Path, payload: bytes) -> str:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(f"existing output is not a regular file: {destination}")
        if destination.read_bytes() != payload:
            raise ValueError(f"existing output differs under the same claim: {destination}")
        return "resumed_identical_existing"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        ATOMIC.publish_file_no_replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "published_no_replace"


def _publish_or_verify_json(destination: Path, value: Mapping[str, object]) -> str:
    return _publish_or_verify_bytes(destination, _canonical_bytes(value, pretty=True))


def _artifact_member(root: Path, path: Path, kind: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    relative = resolved.relative_to(root.resolve()).as_posix()
    return {
        "kind": kind,
        "relative_path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def verify_published_wire(
    path: Path,
    *,
    expected_blob: bytes,
    expected_decoded: torch.Tensor,
    expected_footprint_bytes: int,
) -> dict[str, object]:
    """Reopen the publication boundary and decode only those exact bytes."""

    blob = path.read_bytes()
    if blob != expected_blob:
        raise AssertionError("published wire differs from producer bytes")
    if len(blob) != expected_footprint_bytes:
        raise AssertionError("published wire length differs from exact footprint")
    parsed = TrellisWire.from_bytes(blob)
    if parsed.to_bytes() != blob:
        raise AssertionError("published wire did not canonical-reserialize")
    decoded = decode_values_torch(
        blob,
        device=expected_decoded.device,
        dtype=expected_decoded.dtype,
    )
    if not torch.equal(decoded, expected_decoded):
        raise AssertionError("published same-byte decode differs from producer decode")
    return {
        "schema": TRELLIS_WIRE_SCHEMA,
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "same_byte_reopen_verified": True,
        "same_byte_reserialize_verified": True,
        "same_byte_decode_verified": True,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def require_gpu_campaign_execution(manifest: Mapping[str, Any]) -> torch.device:
    """Keep manifest validation/preflight CPU-safe but quality execution GPU-only."""

    device = torch.device(manifest["execution"]["device"])
    if device.type != "cuda":
        raise ValueError("quality campaign execution requires a declared CUDA device")
    if not torch.cuda.is_available():
        raise ValueError("quality campaign execution requires available CUDA")
    if manifest["recipe"]["backend"] != "triton":
        raise ValueError("quality campaign execution requires the Triton encoder backend")
    return device


@contextlib.contextmanager
def _measure_phase(
    name: str, device: torch.device, timings: dict[str, float]
) -> Iterator[None]:
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.nvtx.range_push(name)
    start = time.perf_counter()
    try:
        yield
    finally:
        _synchronize(device)
        timings[name] = time.perf_counter() - start
        if device.type == "cuda":
            torch.cuda.nvtx.range_pop()


def _native_arm_record(
    arm: Any, metrics: Mapping[str, object]
) -> dict[str, object]:
    accounting = NATIVE.payload_accounting(arm.fields)
    return {
        "carrier": "compressed_tensors_native_nvfp4_fields",
        "fields_sha256": NATIVE.fields_sha256(arm.fields),
        "payload": accounting,
        "metrics": metrics,
    }


def _algebra_witness(rows: int, columns: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(0xA4E20260830)
    values = torch.randint(0, 2, (rows, columns), generator=generator)
    return values.mul_(2).sub_(1).to(device=device, dtype=torch.float32)


def _metric_snr(metrics: Mapping[str, object], name: str) -> float:
    value = metrics[name]
    if not isinstance(value, Mapping) or value.get("available") is False:
        raise ValueError(f"metric {name} is unavailable")
    return float(value["snr_db"])


def _validate_resumed_metrics(value: Any, *, mode: str, where: str) -> None:
    metrics = _require_exact_fields(
        value,
        frozenset({
            "weight", "regularized_hessian_proxy", "activation_output",
            "raw_importance_weighted",
        }),
        where=where,
    )
    weight = _require_exact_fields(
        metrics["weight"],
        frozenset({"nsse", "snr_db", "numerator", "denominator"}),
        where=f"{where}.weight",
    )
    hessian = _require_exact_fields(
        metrics["regularized_hessian_proxy"],
        frozenset({
            "available", "basis", "representation", "nsse", "snr_db",
            "numerator", "denominator",
        }),
        where=f"{where}.regularized_hessian_proxy",
    )
    if hessian["available"] is not True or hessian["basis"] != "untransformed_original_linear":
        raise ValueError(f"{where}: regularized Hessian metric domain differs")
    expected_representation = "dense" if mode == QWEN_MODE else "diagonal"
    if hessian["representation"] != expected_representation:
        raise ValueError(f"{where}: regularized Hessian representation differs")
    activation = metrics["activation_output"]
    raw = metrics["raw_importance_weighted"]
    if mode == QWEN_MODE:
        _require_exact_fields(
            activation,
            frozenset({
                "available", "reason", "nsse", "snr_db", "numerator",
                "denominator",
            }),
            where=f"{where}.activation_output",
        )
        if activation["available"] is not True or activation["reason"] != QWEN_ACTIVATION_AVAILABLE_REASON:
            raise ValueError(f"{where}: activation metric domain differs")
        if raw != {
            "available": False,
            "reason": "raw importance is defined only by the GLM corpus contract",
        }:
            raise ValueError(f"{where}: raw-importance metric domain differs")
        numeric_objects = (weight, hessian, activation)
    else:
        if activation != {
            "available": False,
            "reason": GLM_ACTIVATION_UNAVAILABLE_REASON,
        }:
            raise ValueError(f"{where}: activation refusal differs")
        _require_exact_fields(
            raw,
            frozenset({
                "available", "nsse", "snr_db", "numerator", "denominator",
            }),
            where=f"{where}.raw_importance_weighted",
        )
        if raw["available"] is not True:
            raise ValueError(f"{where}: raw-importance metric is unavailable")
        numeric_objects = (weight, hessian, raw)
    for item in numeric_objects:
        for field in ("nsse", "snr_db", "numerator", "denominator"):
            number = item[field]
            if type(number) not in {int, float} or not math.isfinite(float(number)):
                raise ValueError(f"{where}: {field} must be finite numeric evidence")
        if float(item["nsse"]) < 0 or float(item["numerator"]) < 0 or float(item["denominator"]) <= 0:
            raise ValueError(f"{where}: metric signs are invalid")


def _qwen_seed_verdict(
    e_metrics: Mapping[str, object],
    a_metrics: Mapping[str, object],
    c_metrics: Mapping[str, object],
    *,
    e_bpw: float,
    c_bpw: float,
) -> dict[str, object]:
    names = ("activation_output", "regularized_hessian_proxy")
    deltas_a = {name: _metric_snr(e_metrics, name) - _metric_snr(a_metrics, name) for name in names}
    deltas_c = {name: _metric_snr(e_metrics, name) - _metric_snr(c_metrics, name) for name in names}
    advance = all(float(e_metrics[name]["nsse"]) <= float(c_metrics[name]["nsse"]) for name in names) and e_bpw < c_bpw
    promising = all(deltas_a[name] >= 1.0 and deltas_c[name] >= -0.25 for name in names)
    return {
        "delta_snr_db_vs_scalar": deltas_a,
        "delta_snr_db_vs_stock_arm_c": deltas_c,
        "advance": advance,
        "promising_fallback": promising,
    }


def _load_unit_tensors(
    manifest: Mapping[str, Any], inputs: PreflightInputs, unit: InputUnit
) -> tuple[torch.Tensor, torch.Tensor | None]:
    source = manifest["input"]
    if inputs.mode == QWEN_MODE:
        return (
            NATIVE._load(source["weight_path"], source["weight_key"]),
            NATIVE._load(source["activations_path"], source["activations_key"]),
        )
    assert inputs.corpus is not None
    weight, importance = inputs.corpus.load_tensor(unit.name)
    return weight, importance


def run_unit(
    manifest: Mapping[str, Any],
    inputs: PreflightInputs,
    unit: InputUnit,
    *,
    index: int,
    output_root: Path,
    claim_identity_sha256: str,
    source_closure_identity_sha256: str,
) -> tuple[dict[str, object], Path]:
    device = torch.device(manifest["execution"]["device"])
    recipe = manifest["recipe"]
    feasibility_plan = require_dense_producer_feasible(unit.shape)
    timings: dict[str, float] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with _measure_phase("load_inputs", device, timings):
        weight_cpu, auxiliary_cpu = _load_unit_tensors(manifest, inputs, unit)
        weight = weight_cpu.to(device=device, dtype=torch.float32)
        auxiliary = auxiliary_cpu.to(device=device, dtype=torch.float32) if auxiliary_cpu is not None else None
    rows, columns = map(int, weight.shape)
    if (rows, columns) != unit.shape or rows % 256 or columns % 256:
        raise ValueError(f"{unit.name}: loaded shape differs or is not 256 aligned")

    if inputs.mode == QWEN_MODE:
        assert auxiliary is not None
        with _measure_phase("construct_qwen_hessian", device, timings):
            x, hessian, realized_damp = NATIVE.damped_hessian(
                auxiliary, columns, device
            )
        regularized_diagonal = None
        raw_importance = None
        algebra_rows = x
        hessian_contract = {
            "construction": "matched preprocessed X.T@X plus damp",
            "damp_fraction": 1.0,
            "realized_diagonal_damping": realized_damp,
            "activation_output_metric_available": True,
        }
    else:
        assert auxiliary is not None
        raw_importance = auxiliary
        with _measure_phase("construct_glm_diagonal_hessian", device, timings):
            regularized_diagonal, hessian_contract = regularized_glm_diagonal(
                raw_importance
            )
            hessian = torch.diag(regularized_diagonal)
        x = None
        algebra_rows = _algebra_witness(
            recipe["glm_algebra_witness_rows"], columns, device
        )
        hessian_contract = {
            **hessian_contract,
            "activation_output_metric_available": False,
            "activation_output_metric_unavailable_reason": GLM_ACTIVATION_UNAVAILABLE_REASON,
            "algebra_probe": {
                "kind": "synthetic_matrix_orientation_witness",
                "rows": int(algebra_rows.shape[0]),
                "tensor_sha256": NATIVE.tensor_sha256(algebra_rows),
                "included_in_quality_metrics": False,
            },
        }

    with torch.inference_mode():
        with _measure_phase("arm_a_scalar_native", device, timings):
            arm_a = NATIVE.rtn_arm(weight)
        with _measure_phase("arm_c_stock_blockldl_native", device, timings):
            arm_c = NATIVE.qtip_native_arm_from_hessian(weight, hessian)
    a_accounting = NATIVE.payload_accounting(arm_a.fields)
    c_accounting = NATIVE.payload_accounting(arm_c.fields)
    expected_native_bytes = native_payload_bytes((rows, columns))
    if a_accounting["bytes"] != expected_native_bytes or c_accounting["bytes"] != expected_native_bytes:
        raise AssertionError("native control payload differs from shape-derived exact bytes")
    rate_plan = exact_native_budget_frontier(
        (rows, columns),
        arm_a_bytes=int(a_accounting["bytes"]),
        arm_c_bytes=int(c_accounting["bytes"]),
    )
    q256 = int(rate_plan["selected_body_rate_q256"])
    schedule = tuple(int(value) for value in rate_plan["schedule"])
    alphabets = {
        int(rate): tuple(int(code) for code in codes)
        for rate, codes in rate_plan["alphabets"].items()
    }
    input_block = automatic_input_block_size(
        columns, recipe["max_input_block_size"]
    )

    metric_kwargs = {
        "regularized_hessian": hessian if inputs.mode == QWEN_MODE else None,
        "regularized_diagonal": regularized_diagonal,
        "activations": x,
        "raw_importance": raw_importance,
    }
    with _measure_phase("score_native_controls", device, timings):
        a_metrics = quality_metrics(weight, arm_a.reconstruction, **metric_kwargs)
        c_metrics = quality_metrics(weight, arm_c.reconstruction, **metric_kwargs)

    unit_dir = _safe_output(output_root, f"tensors/{index:03d}-{_slug(unit.name)}")
    unit_dir.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict[str, object]] = []
    for seed in manifest["seeds"]:
        label = seed["label"]
        with torch.inference_mode():
            with _measure_phase(f"arm_e_prepare_{label}", device, timings):
                prepared = ARM_E.prepare_one_linear_scaffold(
                    weight,
                    hessian,
                    body_rate_q256=q256,
                    input_block_size=input_block,
                    output_block_size=recipe["output_block_size"],
                    input_seed=seed["input_seed"],
                    output_seed=seed["output_seed"],
                    research_opt_in=ARM_E.RESEARCH_OPT_IN,
                )
            with _measure_phase(f"arm_e_encode_{label}", device, timings):
                artifact = ARM_E.require_blockldl_trellis_wire_round_trip(
                    prepared,
                    algebra_rows,
                    body_rate_q256=q256,
                    schedule=schedule,
                    layout=recipe["layout"],
                    alphabets=alphabets,
                    scale_rule=recipe["scale_rule"],
                    sb_chunk=recipe["sb_chunk"],
                    determinism_mode=recipe["determinism_mode"],
                    tailbite_candidates=recipe["tailbite_candidates"],
                    backend=recipe["backend"],
                    point_route=recipe["point_route"],
                    terminal_metric_mode=recipe["terminal_metric_mode"],
                    buffer_blocks=recipe["buffer_blocks"],
                    research_opt_in=ARM_E.RESEARCH_OPT_IN,
                )
            feedback_count = int(
                artifact.receipt["block_ldl"][
                    "cross_block_feedback_nonzero_count"
                ]
            )
            if columns > 256 and feedback_count <= 0:
                raise AssertionError(
                    "Arm E campaign requires nonzero cross-terminal feedback; "
                    "the selected Hessian/transform reduced to independent terminals"
                )
            original_q = ARM_E.decoded_weight_in_original_basis(
                artifact.decoded_transformed_weight, artifact.online_transform
            )
            with _measure_phase(f"arm_e_score_{label}", device, timings):
                e_metrics = quality_metrics(weight, original_q, **metric_kwargs)

        wire_path = unit_dir / f"{label}.trellis"
        publication_status = _publish_or_verify_bytes(wire_path, artifact.wire_bytes)
        wire_verification = verify_published_wire(
            wire_path,
            expected_blob=artifact.wire_bytes,
            expected_decoded=artifact.decoded_transformed_weight,
            expected_footprint_bytes=int(rate_plan["footprint"]["total_bytes"]),
        )
        exact_bpw = 8.0 * len(artifact.wire_bytes) / (rows * columns)
        if exact_bpw != float(rate_plan["footprint"]["exact_bpw"]):
            raise AssertionError("wire-derived bpw differs from exact footprint")
        seed_record: dict[str, object] = {
            "seed": dict(seed),
            "online_transform": artifact.online_transform,
            "wire": {
                **wire_verification,
                "relative_path": wire_path.relative_to(output_root).as_posix(),
                "publication_status": publication_status,
                "exact_bpw_over_quantizable_weights": exact_bpw,
                "quantizable_weight_count": rows * columns,
            },
            "metrics": e_metrics,
            "producer_receipt": artifact.receipt,
            "claim_boundary": {
                "quality_only": True,
                "gridbook_runtime_executed": False,
                "served": False,
                "performance_claim": False,
                "producer_eligible": False,
                "runtime_pin_changed": False,
            },
        }
        if inputs.mode == QWEN_MODE:
            seed_record["verdict"] = _qwen_seed_verdict(
                e_metrics,
                a_metrics,
                c_metrics,
                e_bpw=exact_bpw,
                c_bpw=float(c_accounting["bits_per_weight"]),
            )
        seed_results.append(seed_record)

    peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    tensor_body: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "status": "matched_quality_isolate_complete",
        "campaign_claim_identity_sha256": claim_identity_sha256,
        "source_closure_identity_sha256": source_closure_identity_sha256,
        "mode": inputs.mode,
        "tensor": {
            "name": unit.name,
            "population": unit.population,
            "shape": [rows, columns],
            "dtype_loaded_for_optimization": "torch.float32",
            "tensor_sha256": NATIVE.tensor_sha256(weight),
            "corpus_raw_weight_sha256": unit.source_weight_sha256,
        },
        "hessian_contract": hessian_contract,
        "metric_availability": {
            "activation_output": inputs.mode == QWEN_MODE,
            "raw_importance_weighted": inputs.mode == GLM_MODE,
            "regularized_hessian_proxy": True,
        },
        "rate_plan": rate_plan,
        "transform_geometry": {
            "input_block_size": input_block,
            "output_block_size": recipe["output_block_size"],
            "physical_blockldl_terminal_columns": 256,
        },
        "controls": {
            "A_scalar_native_nvfp4": _native_arm_record(arm_a, a_metrics),
            "C_stock_blockldl_native_nvfp4": _native_arm_record(arm_c, c_metrics),
        },
        "arm_e_by_seed": seed_results,
        "feasibility_telemetry": {
            "scope": "offline_quality_campaign_noncomparative",
            "preflight_plan": feasibility_plan,
            "phase_seconds": dict(sorted(timings.items())),
            "total_measured_phase_seconds": sum(timings.values()),
            "torch_cuda_peak_allocated_bytes": peak,
            "cpu_factorization_fallback_observed": device.type != "cuda",
            "gpu_utilization_used_as_diagnostic": False,
            "serving_or_throughput_claim": False,
        },
        "format_registry_entries_created": 0,
        "runtime_pin_changed": False,
        "production_contract_changed": False,
        "producer_eligible": False,
    }
    tensor_result = {
        **tensor_body,
        "identity_sha256": _identity_sha256(tensor_body),
    }
    result_path = unit_dir / "result.json"
    _publish_or_verify_json(result_path, tensor_result)
    return tensor_result, result_path


def _bootstrap_lower_95(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("bootstrap requires observations")
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(20260830)
    indices = torch.randint(
        0, tensor.numel(), (10_000, tensor.numel()), generator=generator
    )
    medians = tensor[indices].median(dim=1).values
    return float(torch.quantile(medians, 0.05).item())


def aggregate_results(
    mode: str, results: Sequence[Mapping[str, object]], seeds: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    if mode == QWEN_MODE:
        if len(results) != 1:
            raise ValueError("Qwen campaign must contain exactly one tensor")
        seed_results = results[0]["arm_e_by_seed"]
        primary = seed_results[0]
        summary: dict[str, object] = {
            "mode": QWEN_MODE,
            "primary_seed": primary["seed"]["label"],
            "primary_advance": primary["verdict"]["advance"],
            "primary_promising_fallback": primary["verdict"]["promising_fallback"],
            "robust_gate_evaluable": len(seed_results) == 3,
        }
        if len(seed_results) == 3:
            activation = [float(item["verdict"]["delta_snr_db_vs_stock_arm_c"]["activation_output"]) for item in seed_results]
            hessian = [float(item["verdict"]["delta_snr_db_vs_stock_arm_c"]["regularized_hessian_proxy"]) for item in seed_results]
            summary["robust_gate"] = {
                "median_activation_delta_db": statistics.median(activation),
                "median_hessian_delta_db": statistics.median(hessian),
                "worst_activation_delta_db": min(activation),
                "pass": statistics.median(activation) >= 0 and statistics.median(hessian) >= 0 and min(activation) >= -0.1,
            }
        return summary

    primary_label = str(seeds[0]["label"])
    populations: dict[str, object] = {}
    expected = {"dense": 9, "routed": 24}
    for population in ("dense", "routed"):
        selected = [result for result in results if result["tensor"]["population"] == population]
        delta_a: list[float] = []
        delta_c: list[float] = []
        for result in selected:
            arm_e = next(
                item for item in result["arm_e_by_seed"]
                if item["seed"]["label"] == primary_label
            )
            e = _metric_snr(arm_e["metrics"], "raw_importance_weighted")
            a = _metric_snr(result["controls"]["A_scalar_native_nvfp4"]["metrics"], "raw_importance_weighted")
            c = _metric_snr(result["controls"]["C_stock_blockldl_native_nvfp4"]["metrics"], "raw_importance_weighted")
            delta_a.append(e - a)
            delta_c.append(e - c)
        complete = len(selected) == expected[population]
        record: dict[str, object] = {
            "tensor_count": len(selected),
            "required_full_census_count": expected[population],
            "full_census_gate_evaluable": complete,
        }
        if selected:
            wins_required_min = 7 if population == "dense" else 18
            wins_required_strong = 8 if population == "dense" else 20
            median_a = statistics.median(delta_a)
            median_c = statistics.median(delta_c)
            wins_a = sum(value > 0 for value in delta_a)
            wins_c = sum(value > 0 for value in delta_c)
            lower_a = _bootstrap_lower_95(delta_a)
            lower_c = _bootstrap_lower_95(delta_c)
            record.update({
                "median_delta_db_vs_scalar": median_a,
                "median_delta_db_vs_stock_arm_c": median_c,
                "wins_vs_scalar": wins_a,
                "wins_vs_stock_arm_c": wins_c,
                "bootstrap_one_sided_95_lower_db_vs_scalar": lower_a,
                "bootstrap_one_sided_95_lower_db_vs_stock_arm_c": lower_c,
                "minimum_gate_pass": complete and median_a > 0 and median_c > 0 and wins_a >= wins_required_min and wins_c >= wins_required_min,
                "strong_gate_pass": complete and median_a >= 0.25 and median_c >= 0.25 and lower_a > 0 and lower_c > 0 and wins_a >= wins_required_strong and wins_c >= wins_required_strong,
            })
        populations[population] = record
    return {
        "mode": GLM_MODE,
        "primary_seed": primary_label,
        "activation_output_metric_available": False,
        "model_quality_claim_permitted": False,
        "populations": populations,
    }


def _verify_complete_receipt(
    path: Path,
    *,
    claim_identity_sha256: str,
    source_closure: Mapping[str, object],
    manifest_provenance: Mapping[str, object],
    input_provenance: Mapping[str, object],
    expected_members: Mapping[str, str],
    mode: str,
    root: Path,
) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("existing campaign receipt has the wrong schema")
    body = dict(value)
    identity = body.pop("identity_sha256", None)
    if identity != _identity_sha256(body):
        raise ValueError("existing campaign receipt identity mismatch")
    _require_exact_fields(
        body, _CAMPAIGN_RECEIPT_BODY_FIELDS, where="campaign receipt"
    )
    if body.get("campaign_claim_identity_sha256") != claim_identity_sha256:
        raise ValueError("existing campaign receipt belongs to another claim")
    if body.get("mode") != mode or body.get("status") != "quality_campaign_complete":
        raise ValueError("existing campaign receipt mode/status differs")
    if body.get("manifest") != manifest_provenance:
        raise ValueError("existing campaign manifest provenance differs")
    if body.get("input_provenance") != input_provenance:
        raise ValueError("existing campaign input provenance differs")
    source = body.get("source_closure")
    if source != source_closure:
        raise ValueError("existing campaign source closure differs")
    if body.get("acceptance_contract") != ACCEPTANCE_CONTRACT:
        raise ValueError("existing campaign acceptance contract differs")
    if body.get("claim_boundary") != {
        "quality_only": True,
        "activation_output_model_quality": False,
        "gridbook_runtime_executed": False,
        "served": False,
        "performance_claim": False,
        "producer_eligible": False,
        "runtime_pin_changed": False,
        "production_contract_changed": False,
    }:
        raise ValueError("existing campaign claim boundary differs")
    members = body.get("published_members")
    if not isinstance(members, list) or not members:
        raise ValueError("existing campaign has no published members")
    paths = [member.get("relative_path") for member in members if isinstance(member, Mapping)]
    if len(paths) != len(members) or len(set(paths)) != len(paths):
        raise ValueError("existing campaign member census is invalid")
    if set(paths) != set(expected_members):
        raise ValueError("existing campaign member census differs from manifest")
    for member in members:
        _require_exact_fields(
            member,
            frozenset({"kind", "relative_path", "bytes", "sha256"}),
            where="campaign published member",
        )
        if member["kind"] != expected_members[member["relative_path"]]:
            raise ValueError("existing campaign member kind differs")
        target = _safe_output(root, member["relative_path"])
        if target.stat().st_size != member["bytes"] or _file_sha256(target) != member["sha256"]:
            raise ValueError(f"existing campaign member differs: {target}")
    return value


def _resume_unit_result(
    root: Path,
    *,
    index: int,
    unit: InputUnit,
    mode: str,
    claim_identity_sha256: str,
    source_closure_identity_sha256: str,
    expected_seeds: Sequence[Mapping[str, object]],
    recipe: Mapping[str, object],
) -> tuple[dict[str, Any], Path] | None:
    result_path = _safe_output(
        root, f"tensors/{index:03d}-{_slug(unit.name)}/result.json"
    )
    if not result_path.exists():
        return None
    value = _read_json(result_path)
    body = dict(value)
    identity = body.pop("identity_sha256", None)
    if identity != _identity_sha256(body):
        raise ValueError(f"resumed tensor receipt identity mismatch: {result_path}")
    _require_exact_fields(
        body, _TENSOR_RESULT_BODY_FIELDS, where="resumed tensor receipt"
    )
    expected = {
        "schema": RESULT_SCHEMA,
        "status": "matched_quality_isolate_complete",
        "campaign_claim_identity_sha256": claim_identity_sha256,
        "source_closure_identity_sha256": source_closure_identity_sha256,
        "mode": mode,
    }
    for field, expected_value in expected.items():
        if body.get(field) != expected_value:
            raise ValueError(f"resumed tensor receipt {field} mismatch: {result_path}")
    expected_availability = {
        "activation_output": mode == QWEN_MODE,
        "raw_importance_weighted": mode == GLM_MODE,
        "regularized_hessian_proxy": True,
    }
    if body.get("metric_availability") != expected_availability:
        raise ValueError(f"resumed tensor metric availability differs: {result_path}")
    for field, expected_value in {
        "format_registry_entries_created": 0,
        "runtime_pin_changed": False,
        "production_contract_changed": False,
        "producer_eligible": False,
    }.items():
        if type(body.get(field)) is not type(expected_value) or body.get(field) != expected_value:
            raise ValueError(f"resumed tensor {field} differs: {result_path}")
    tensor = body.get("tensor")
    if not isinstance(tensor, Mapping) or (
        tensor.get("name") != unit.name
        or tensor.get("population") != unit.population
        or tensor.get("shape") != list(unit.shape)
        or tensor.get("corpus_raw_weight_sha256") != unit.source_weight_sha256
    ):
        raise ValueError(f"resumed tensor identity mismatch: {result_path}")
    expected_native_bytes = native_payload_bytes(unit.shape)
    expected_rate_plan = exact_native_budget_frontier(
        unit.shape,
        arm_a_bytes=expected_native_bytes,
        arm_c_bytes=expected_native_bytes,
    )
    if body.get("rate_plan") != expected_rate_plan:
        raise ValueError(f"resumed tensor exact-byte frontier differs: {result_path}")
    expected_geometry = {
        "input_block_size": automatic_input_block_size(
            unit.shape[1], int(recipe["max_input_block_size"])
        ),
        "output_block_size": recipe["output_block_size"],
        "physical_blockldl_terminal_columns": 256,
    }
    if body.get("transform_geometry") != expected_geometry:
        raise ValueError(f"resumed tensor transform geometry differs: {result_path}")
    controls = _require_exact_fields(
        body.get("controls"),
        frozenset({
            "A_scalar_native_nvfp4", "C_stock_blockldl_native_nvfp4",
        }),
        where="resumed tensor controls",
    )
    for name, control in controls.items():
        record = _require_exact_fields(
            control,
            frozenset({"carrier", "fields_sha256", "payload", "metrics"}),
            where=f"resumed tensor controls.{name}",
        )
        if record["carrier"] != "compressed_tensors_native_nvfp4_fields" or not _SHA256_RE.fullmatch(str(record["fields_sha256"])):
            raise ValueError(f"resumed native control identity differs: {name}")
        payload = record["payload"]
        if (
            not isinstance(payload, Mapping)
            or payload.get("bytes") != expected_native_bytes
            or payload.get("n_weights") != math.prod(unit.shape)
            or payload.get("bits_per_weight") != 8.0 * expected_native_bytes / math.prod(unit.shape)
        ):
            raise ValueError(f"resumed native control accounting differs: {name}")
        _validate_resumed_metrics(
            record["metrics"], mode=mode, where=f"resumed controls.{name}.metrics"
        )
    seeds = body.get("arm_e_by_seed")
    if not isinstance(seeds, list) or len(seeds) != len(expected_seeds):
        raise ValueError(f"resumed tensor seed census differs: {result_path}")
    if [item.get("seed") for item in seeds if isinstance(item, Mapping)] != [dict(seed) for seed in expected_seeds]:
        raise ValueError(f"resumed tensor seed order/domain differs: {result_path}")
    for expected_seed, seed in zip(expected_seeds, seeds, strict=True):
        seed_fields = {
            "seed", "online_transform", "wire", "metrics", "producer_receipt",
            "claim_boundary",
        }
        if mode == QWEN_MODE:
            seed_fields.add("verdict")
        _require_exact_fields(
            seed, frozenset(seed_fields), where=f"resumed arm_e.{expected_seed['label']}"
        )
        if seed.get("claim_boundary") != {
            "quality_only": True,
            "gridbook_runtime_executed": False,
            "served": False,
            "performance_claim": False,
            "producer_eligible": False,
            "runtime_pin_changed": False,
        }:
            raise ValueError(f"resumed Arm E claim boundary differs: {result_path}")
        wire = seed.get("wire") if isinstance(seed, Mapping) else None
        if not isinstance(wire, Mapping):
            raise ValueError(f"resumed tensor wire record is invalid: {result_path}")
        expected_relative = (
            f"tensors/{index:03d}-{_slug(unit.name)}/"
            f"{expected_seed['label']}.trellis"
        )
        if wire.get("relative_path") != expected_relative:
            raise ValueError(f"resumed tensor wire path differs: {result_path}")
        path = _safe_output(root, expected_relative)
        if path.stat().st_size != wire.get("bytes") or _file_sha256(path) != wire.get("sha256"):
            raise ValueError(f"resumed tensor wire differs: {path}")
        blob = path.read_bytes()
        parsed = TrellisWire.from_bytes(blob)
        if parsed.to_bytes() != blob:
            raise ValueError(f"resumed tensor wire is not canonical: {path}")
        footprint = expected_rate_plan["footprint"]
        if (
            parsed.rows != unit.shape[0]
            or parsed.columns != unit.shape[1]
            or parsed.body_rate_q256 != expected_rate_plan["selected_body_rate_q256"]
            or parsed.layout != LAYOUT_TIGHT_OFFSETS
            or list(parsed.schedule) != expected_rate_plan["schedule"]
            or {str(rate): list(codes) for rate, codes in parsed.alphabets.items()} != expected_rate_plan["alphabets"]
            or len(blob) != footprint["total_bytes"]
            or wire.get("exact_bpw_over_quantizable_weights") != footprint["exact_bpw"]
            or wire.get("quantizable_weight_count") != math.prod(unit.shape)
            or wire.get("same_byte_reopen_verified") is not True
            or wire.get("same_byte_reserialize_verified") is not True
            or wire.get("same_byte_decode_verified") is not True
        ):
            raise ValueError(f"resumed tensor wire recipe/accounting differs: {path}")
        transform = ARM_E.validate_online_transform(
            seed.get("online_transform"), rows=unit.shape[0], columns=unit.shape[1]
        )
        if (
            transform["input"]["seed"] != expected_seed["input_seed"]
            or transform["output"]["seed"] != expected_seed["output_seed"]
        ):
            raise ValueError(f"resumed tensor transform seeds differ: {path}")
        _validate_resumed_metrics(
            seed.get("metrics"), mode=mode, where=f"resumed arm_e.{expected_seed['label']}.metrics"
        )
        producer = seed.get("producer_receipt")
        if not isinstance(producer, Mapping):
            raise ValueError(f"resumed producer receipt is missing: {path}")
        producer_body = dict(producer)
        producer_identity = producer_body.pop("identity_sha256", None)
        block_ldl = producer_body.get("block_ldl")
        decoded_weight = decode_values_torch(
            blob, device=torch.device("cpu"), dtype=torch.float32
        )
        decoded_codes = WIRE_MODULE.decode_codes_torch(
            blob, device=torch.device("cpu")
        )
        if (
            producer_identity != ARM_E._canonical_sha256(producer_body)
            or producer_body.get("schema") != ARM_E.BLOCKLDL_COMBINED_ARTIFACT_SCHEMA
            or producer_body.get("wire_bytes") != len(blob)
            or producer_body.get("wire_identity_sha256") != wire["sha256"]
            or producer_body.get("decoded_weight_sha256") != ARM_E._tensor_sha256(decoded_weight)
            or producer_body.get("decoded_codes_sha256") != ARM_E._tensor_sha256(decoded_codes)
            or producer_body.get("same_byte_reparse_verified") is not True
            or producer_body.get("producer_eligible") is not False
            or (
                unit.shape[1] > 256
                and (
                    not isinstance(block_ldl, Mapping)
                    or type(block_ldl.get("cross_block_feedback_nonzero_count")) is not int
                    or block_ldl["cross_block_feedback_nonzero_count"] <= 0
                )
            )
        ):
            raise ValueError(f"resumed producer receipt differs: {path}")
        if mode == QWEN_MODE:
            expected_verdict = _qwen_seed_verdict(
                seed["metrics"],
                controls["A_scalar_native_nvfp4"]["metrics"],
                controls["C_stock_blockldl_native_nvfp4"]["metrics"],
                e_bpw=float(wire["exact_bpw_over_quantizable_weights"]),
                c_bpw=float(controls["C_stock_blockldl_native_nvfp4"]["payload"]["bits_per_weight"]),
            )
            if seed.get("verdict") != expected_verdict:
                raise ValueError(f"resumed Qwen verdict differs: {path}")
    return value, result_path


def run_campaign(
    manifest: Mapping[str, Any],
    manifest_provenance: Mapping[str, object],
    inputs: PreflightInputs,
    source_closure: Mapping[str, object],
    *,
    command: Sequence[str],
) -> dict[str, object]:
    require_gpu_campaign_execution(manifest)
    if validated_source_closure(
        manifest["execution"]["prismaquant_checkout"],
        manifest["execution"]["prismaquant_commit"],
    ) != source_closure:
        raise ValueError("source closure changed before campaign claim")
    output_root = Path(manifest["output"]["root"]).absolute()
    # Refuse the complete selection before creating a claim or publishing a
    # prefix.  In particular, a full GLM manifest containing K=12288 cannot
    # silently begin a dense O(K^3) factorization.
    for unit in inputs.units:
        require_dense_producer_feasible(unit.shape)
    output_root.mkdir(parents=True, exist_ok=True)
    output_root = output_root.resolve(strict=True)
    final_path = _safe_output(output_root, "receipt.json")
    claim_body: dict[str, object] = {
        "schema": CLAIM_SCHEMA,
        "manifest": dict(manifest_provenance),
        "manifest_semantics": manifest,
        "input_provenance": inputs.provenance,
        "source_closure_identity_sha256": source_closure["identity_sha256"],
        "output_root": str(output_root),
    }
    claim_identity_sha256 = _identity_sha256(claim_body)
    with ATOMIC.exclusive_publication_claim(final_path, identity=claim_body):
        if final_path.exists():
            expected_members: dict[str, str] = {}
            for index, unit in enumerate(inputs.units):
                directory = f"tensors/{index:03d}-{_slug(unit.name)}"
                expected_members[f"{directory}/result.json"] = (
                    "tensor_result_commit_marker"
                )
                for seed in manifest["seeds"]:
                    expected_members[f"{directory}/{seed['label']}.trellis"] = (
                        "canonical_trellis_wire"
                    )
            value = _verify_complete_receipt(
                final_path,
                claim_identity_sha256=claim_identity_sha256,
                source_closure=source_closure,
                manifest_provenance=manifest_provenance,
                input_provenance=inputs.provenance,
                expected_members=expected_members,
                mode=manifest["mode"],
                root=output_root,
            )
            resumed_results: list[dict[str, object]] = []
            actual_members: list[dict[str, object]] = []
            for index, unit in enumerate(inputs.units):
                resumed = _resume_unit_result(
                    output_root,
                    index=index,
                    unit=unit,
                    mode=inputs.mode,
                    claim_identity_sha256=claim_identity_sha256,
                    source_closure_identity_sha256=str(
                        source_closure["identity_sha256"]
                    ),
                    expected_seeds=manifest["seeds"],
                    recipe=manifest["recipe"],
                )
                if resumed is None:
                    raise ValueError("completed campaign is missing a tensor result")
                result, result_path = resumed
                resumed_results.append(result)
                actual_members.append(_artifact_member(
                    output_root, result_path, "tensor_result_commit_marker"
                ))
                for seed_result in result["arm_e_by_seed"]:
                    wire_path = _safe_output(
                        output_root, seed_result["wire"]["relative_path"]
                    )
                    actual_members.append(_artifact_member(
                        output_root, wire_path, "canonical_trellis_wire"
                    ))
            expected_summary = aggregate_results(
                manifest["mode"], resumed_results, manifest["seeds"]
            )
            if value["summary"] != expected_summary:
                raise ValueError("completed campaign summary differs from tensor results")
            if value["published_members"] != sorted(
                actual_members, key=lambda item: item["relative_path"]
            ):
                raise ValueError("completed campaign member records differ")
            if validated_source_closure(
                manifest["execution"]["prismaquant_checkout"],
                manifest["execution"]["prismaquant_commit"],
            ) != source_closure:
                raise ValueError("source closure changed during receipt resume")
            return value
        results: list[dict[str, object]] = []
        result_paths: list[Path] = []
        for index, unit in enumerate(inputs.units):
            resumed = _resume_unit_result(
                output_root,
                index=index,
                unit=unit,
                mode=inputs.mode,
                claim_identity_sha256=claim_identity_sha256,
                source_closure_identity_sha256=str(source_closure["identity_sha256"]),
                expected_seeds=manifest["seeds"],
                recipe=manifest["recipe"],
            )
            if resumed is None:
                result, path = run_unit(
                    manifest,
                    inputs,
                    unit,
                    index=index,
                    output_root=output_root,
                    claim_identity_sha256=claim_identity_sha256,
                    source_closure_identity_sha256=str(source_closure["identity_sha256"]),
                )
            else:
                result, path = resumed
            results.append(result)
            result_paths.append(path)
        closing_sources = validated_source_closure(
            manifest["execution"]["prismaquant_checkout"],
            manifest["execution"]["prismaquant_commit"],
        )
        if closing_sources != source_closure:
            raise ValueError("source closure changed during campaign")
        summary = aggregate_results(manifest["mode"], results, manifest["seeds"])
        wire_paths = [
            _safe_output(output_root, item["wire"]["relative_path"])
            for result in results for item in result["arm_e_by_seed"]
        ]
        members = [
            *(_artifact_member(output_root, path, "tensor_result_commit_marker") for path in result_paths),
            *(_artifact_member(output_root, path, "canonical_trellis_wire") for path in wire_paths),
        ]
        device = torch.device(manifest["execution"]["device"])
        receipt_body: dict[str, object] = {
            "schema": RECEIPT_SCHEMA,
            "status": "quality_campaign_complete",
            "campaign_claim_identity_sha256": claim_identity_sha256,
            "manifest": dict(manifest_provenance),
            "mode": manifest["mode"],
            "input_provenance": inputs.provenance,
            "source_closure": source_closure,
            "execution": {
                "declared_host": manifest["execution"]["host"],
                "observed_hostname": socket.gethostname(),
                "container_identity": manifest["execution"]["container_identity"],
                "device": NATIVE.device_identity(device),
                "torch_version": torch.__version__,
                "cuda_toolkit_version": torch.version.cuda,
                "command": list(command),
            },
            "acceptance_contract": ACCEPTANCE_CONTRACT,
            "summary": summary,
            "published_members": sorted(members, key=lambda item: item["relative_path"]),
            "publication": {
                "semantics": "persistent_identity_claim_per_tensor_result_markers_receipt_last",
                "durable_root_uri": manifest["output"]["durable_root_uri"],
                "resume": (
                    "a complete receipt is revalidated and returned; identical "
                    "per-tensor wires/results under the same persistent claim "
                    "are structurally revalidated against the manifest seed, "
                    "shape, rate, accounting, metric-domain, transform, and "
                    "producer-receipt contracts before reuse; conflicting or "
                    "incomplete prefixes fail closed"
                ),
                "resume_assurance": (
                    "structural crash-resume validation, not an external "
                    "signature over locally writable evidence"
                ),
                "commit_marker_relative_path": "receipt.json",
            },
            "claim_boundary": {
                "quality_only": True,
                "activation_output_model_quality": False,
                "gridbook_runtime_executed": False,
                "served": False,
                "performance_claim": False,
                "producer_eligible": False,
                "runtime_pin_changed": False,
                "production_contract_changed": False,
            },
        }
        receipt = {
            **receipt_body,
            "identity_sha256": _identity_sha256(receipt_body),
        }
        if validated_source_closure(
            manifest["execution"]["prismaquant_checkout"],
            manifest["execution"]["prismaquant_commit"],
        ) != source_closure:
            raise ValueError("source closure changed before receipt publication")
        _publish_or_verify_json(final_path, receipt)
        return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate the closed contract and exact byte frontier without encoding",
    )
    args = parser.parse_args(argv)
    actual_command = [str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    manifest, manifest_provenance = load_manifest(args.manifest)
    checkout = manifest["execution"]["prismaquant_checkout"]
    source_closure = validated_source_closure(
        checkout, manifest["execution"]["prismaquant_commit"]
    )
    inputs = preflight_inputs(manifest)
    if args.preflight_only:
        report = preflight_report(
            manifest, manifest_provenance, inputs, source_closure
        )
    else:
        device = torch.device(manifest["execution"]["device"])
        if manifest["recipe"]["backend"] == "triton" and device.type != "cuda":
            raise ValueError("the manifest requests Triton on a non-CUDA device")
        report = run_campaign(
            manifest,
            manifest_provenance,
            inputs,
            source_closure,
            command=actual_command,
        )
    print(_canonical_bytes(report, pretty=True).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
