"""Exact in-process vLLM contract for DSv4 Gridbook gold measurements."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from prismaquant.gridbook_assignment import artifact_requires_moe_backend_marlin
from prismaquant.gridbook_environment import (
    apply_canonical_gold_environment,
    attest_canonical_gold_environment,
)


DSV4_GRIDBOOK_CONTRACT_SCHEMA = "prismaquant.dsv4_gridbook_llm_contract/1"
DSV4_MAX_MODEL_LEN = 8192
DSV4_MAX_NUM_BATCHED_TOKENS = 512
DSV4_KV_CACHE_MEMORY_BYTES = 1_073_741_824
DSV4_GPU_MEMORY_UTILIZATION = 0.84
DSV4_MAX_LOGPROBS = 248_320
_FORBIDDEN_RUNTIME_PIN_OVERRIDES = (
    "PQ_GRIDBOOK_RUNTIME_COMMIT",
    "PQ_GRIDBOOK_RUNTIME_VERSION",
    "PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256",
)


def requires_moe_backend_marlin(model: str | os.PathLike) -> bool:
    """Whether the finalized artifact contains native MXFP4 expert payloads."""
    root = Path(model)
    try:
        payload = json.loads((root / "quant_config.json").read_text(
            encoding="utf-8"
        ))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"DSv4 Gridbook contract cannot read {root / 'quant_config.json'}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("quant_method") != "gridbook":
        raise ValueError("DSv4 Gridbook gold measurement requires a Gridbook artifact")
    return artifact_requires_moe_backend_marlin(payload)


def exact_llm_contract(model: str | os.PathLike) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return exact vLLM kwargs plus a JSON-safe receipt for one Spark."""
    root = Path(model)
    if not root.is_dir():
        raise ValueError(
            "DSv4 Gridbook gold measurement requires a local artifact directory"
        )
    requires_marlin = requires_moe_backend_marlin(root)
    # Runtime-pin overrides are launcher provenance, not Gridbook execution
    # knobs.  Gold receives its immutable pin separately and must not inherit
    # an ambient override.  The Gridbook helper then clears *every* known 0.8.4
    # environment input before installing the complete canonical state.  This
    # happens before the measurement tool imports Gridbook or vLLM.
    for name in _FORBIDDEN_RUNTIME_PIN_OVERRIDES:
        os.environ.pop(name, None)
    apply_canonical_gold_environment()
    environment_receipt = attest_canonical_gold_environment()
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": DSV4_GPU_MEMORY_UTILIZATION,
        "max_logprobs": DSV4_MAX_LOGPROBS,
        "quantization": "gridbook",
        "kv_cache_dtype": "fp8",
        "tokenizer_mode": "deepseek_v4",
        "generation_config": "vllm",
        "enable_prefix_caching": False,
        "max_model_len": DSV4_MAX_MODEL_LEN,
        "max_num_seqs": 1,
        "max_num_batched_tokens": DSV4_MAX_NUM_BATCHED_TOKENS,
        "kv_cache_memory_bytes": DSV4_KV_CACHE_MEMORY_BYTES,
        "seed": 0,
        "enforce_eager": True,
        "disable_log_stats": True,
    }
    if requires_marlin:
        kwargs["moe_backend"] = "marlin"
    receipt = {
        "schema": DSV4_GRIDBOOK_CONTRACT_SCHEMA,
        "artifact_dir": str(root.resolve()),
        "requires_moe_backend_marlin": requires_marlin,
        "llm_kwargs": dict(kwargs),
        # Null means the variable was affirmatively cleared.  Keeping those
        # rows in the receipt prevents an inherited retired/debug/schedule
        # variable from being indistinguishable from the canonical default.
        "environment": dict(environment_receipt["environment"]),
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "gpu_memory_utilization": DSV4_GPU_MEMORY_UTILIZATION,
        "disable_log_stats": True,
        "speculative_decoding": False,
    }
    return kwargs, receipt
