#!/usr/bin/env python3
"""Direct selected-assignment export for the unreleasable GB10 lane.

This handoff starts after allocation.  It consumes one completed
``layer_config.json``, its exact value-bound CB column weights, the original
source checkpoint, and a Gridbook v11 compile-only SM89 contract.  It invokes
the existing streaming exporter directly: no probe, cost, retained format-menu
cache, frontier selection, or stock pipeline stage is rerun.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

import torch

from .layer_config import load_assignment
from .nvfp4_cb_footprint import (
    whole_artifact_budget_from_assignment_payload,
)
from .production_weight_cache import validate_cb_render_identity_metadata
from .rtx4090_qwen38_policy import (
    RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES,
    RTX4090_VALIDATION_ONLY_POLICY_ID,
    require_rtx4090_compile_only_runtime_contract,
    validate_qwen38_dense_config,
    validate_rtx4090_assignment,
)


DIRECT_VALIDATION_EXPORT_SCHEMA = (
    "prismaquant.rtx4090_validation_only_direct_export.v1"
)
DEFAULT_SHARD_BYTES = 1024**3


class RTX4090DirectValidationExportError(ValueError):
    """The completed allocation is not an exact direct-export handoff."""


def _load_json_object(path: Path, *, where: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RTX4090DirectValidationExportError(
            f"{where} cannot be read as UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RTX4090DirectValidationExportError(f"{where} must be an object")
    return payload


def preflight_direct_validation_export(
    *,
    model_dir: str | Path,
    layer_config: str | Path,
    col_weights: str | Path,
    runtime_contract: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Validate the completed handoff before constructing a GPU command."""

    model = Path(model_dir).resolve()
    recipe_path = Path(layer_config).resolve()
    col_path = Path(col_weights).resolve()
    contract_path = Path(runtime_contract).resolve()
    output = Path(out_dir).resolve()
    for path, label in (
        (model / "config.json", "source config"),
        (recipe_path, "completed layer_config"),
        (col_path, "exact col_weights"),
        (contract_path, "Gridbook v11 runtime contract"),
    ):
        if not path.is_file():
            raise RTX4090DirectValidationExportError(
                f"{label} must be an existing regular file: {path}"
            )
    if output.exists():
        raise RTX4090DirectValidationExportError(
            f"direct export output must not already exist: {output}"
        )

    source_config = _load_json_object(
        model / "config.json", where="direct-export source config"
    )
    validate_qwen38_dense_config(
        source_config, where="direct-export source model"
    )
    recipe = _load_json_object(
        recipe_path, where="direct-export layer_config"
    )
    try:
        assignment = validate_rtx4090_assignment(
            load_assignment(recipe_path), where="direct-export selected assignment"
        )
    except Exception as exc:
        raise RTX4090DirectValidationExportError(str(exc)) from exc
    budget = whole_artifact_budget_from_assignment_payload(
        recipe,
        where="direct-export layer_config",
        assignment=assignment,
    )
    if budget is None:
        raise RTX4090DirectValidationExportError(
            "direct export requires the allocator's exact whole_artifact_budget stamp"
        )
    budget_bytes = int(budget["budget_bytes"])
    if not 0 < budget_bytes <= RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES:
        raise RTX4090DirectValidationExportError(
            "direct export budget must be positive and no greater than "
            f"{RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES} bytes"
        )
    if budget.get("excluded_source_prefixes") not in (None, []):
        raise RTX4090DirectValidationExportError(
            "validation-only full artifact cannot exclude source namespaces"
        )

    contract = _load_json_object(
        contract_path, where="direct-export Gridbook v11 contract"
    )
    attestation = require_rtx4090_compile_only_runtime_contract(
        contract,
        tuple(assignment.values()),
        where="direct-export Gridbook compile-only contract",
    )

    meta = recipe.get("__prismaquant__")
    render_identity = recipe.get("cb_render_identity")
    if render_identity is None and isinstance(meta, Mapping):
        render_identity = meta.get("cb_render_identity")
    if not isinstance(render_identity, Mapping):
        raise RTX4090DirectValidationExportError(
            "completed layer_config has no value-bearing cb_render_identity"
        )
    try:
        with col_path.open("rb") as handle:
            raw_col_weights = pickle.load(handle)
        if not isinstance(raw_col_weights, Mapping):
            raise ValueError("col_weights pickle must contain a mapping")
        col_values = {
            str(name): torch.as_tensor(value)
            for name, value in raw_col_weights.items()
        }
        selected_cb = {
            name: (fmt,)
            for name, fmt in assignment.items()
            if fmt.startswith("FP8_CB_K")
        }
        validate_cb_render_identity_metadata(
            render_identity,
            expected_formats_by_qname=selected_cb,
            col_weights=col_values,
            require_source_complete=True,
            where="direct-export exact col_weights/render identity",
        )
    except Exception as exc:
        raise RTX4090DirectValidationExportError(str(exc)) from exc

    return {
        "schema": DIRECT_VALIDATION_EXPORT_SCHEMA,
        "model_dir": str(model),
        "layer_config": str(recipe_path),
        "col_weights": str(col_path),
        "runtime_contract": str(contract_path),
        "out_dir": str(output),
        "budget_bytes": budget_bytes,
        "assignment_units": len(assignment),
        "selected_fp8_cb_units": len(selected_cb),
        "runtime_contract_sha256": attestation["runtime_contract_sha256"],
    }


def build_direct_validation_export_command(
    handoff: Mapping[str, Any],
    *,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    python_executable: str = sys.executable,
) -> tuple[list[str], dict[str, str]]:
    """Construct the sole selected-assignment renderer invocation."""

    if handoff.get("schema") != DIRECT_VALIDATION_EXPORT_SCHEMA:
        raise RTX4090DirectValidationExportError("invalid direct-export handoff")
    if isinstance(shard_bytes, bool) or int(shard_bytes) <= 0:
        raise RTX4090DirectValidationExportError("shard_bytes must be positive")
    command = [
        str(python_executable),
        "-m",
        "prismaquant.export_nvfp4_cb_streaming",
        "--model-dir",
        str(handoff["model_dir"]),
        "--layer-config",
        str(handoff["layer_config"]),
        "--col-weights",
        str(handoff["col_weights"]),
        "--out",
        str(handoff["out_dir"]),
        "--shard-bytes",
        str(int(shard_bytes)),
        "--codebook-source",
        "lattice",
        "--scale-coding",
        "v1",
        "--device",
        "cuda",
        "--producer-policy",
        RTX4090_VALIDATION_ONLY_POLICY_ID,
        "--producer-runtime-contract",
        str(handoff["runtime_contract"]),
    ]
    environment = dict(os.environ)
    environment.update({
        "CB_CODEBOOK_SOURCE_SCOPE": "none",
        "CB_CODEBOOK_SOURCE": "lattice",
        "CB_ACTIVATION_SCOPE": "none",
        "CB_SCALE_SWEEP": "1",
        "CB_SCALE_SWEEP_SCOPE": "fp8",
        "PRISMAQUANT_CB_LDLQ": "0",
        "PRISMAQUANT_CB_MINCHAIN": "0",
        "PRISMAQUANT_CB_ENCODE_TIER": "balanced",
    })
    return command, environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--layer-config", required=True)
    parser.add_argument("--col-weights", required=True)
    parser.add_argument("--runtime-contract", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    args = parser.parse_args(argv)

    handoff = preflight_direct_validation_export(
        model_dir=args.model_dir,
        layer_config=args.layer_config,
        col_weights=args.col_weights,
        runtime_contract=args.runtime_contract,
        out_dir=args.out,
    )
    command, environment = build_direct_validation_export_command(
        handoff, shard_bytes=args.shard_bytes
    )
    print("[rtx4090-direct-validation] selected-assignment streaming export")
    print("[rtx4090-direct-validation] " + " ".join(command))
    subprocess.run(command, check=True, env=environment)

    from .validate_rtx4090_fp8_cb_validation_only import (
        validate_rtx4090_validation_only_artifact,
    )

    result = validate_rtx4090_validation_only_artifact(
        handoff["out_dir"], runtime_contract=handoff["runtime_contract"]
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
