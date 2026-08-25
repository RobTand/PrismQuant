#!/usr/bin/env python3
"""Structural validator for unreleasable GB10-built RTX4090 FP8-CB artifacts.

This command consumes Gridbook v11 ``compile_only`` SM89 cells.  It validates
the same FP8-only wire layout, source census, finalized tensor census, and
content ledgers as the strict producer, but it can never emit serving or ship
evidence and its immutable policy stamp is rejected by all release tooling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .rtx4090_qwen38_policy import (
    RTX4090_VALIDATION_ONLY_DISPOSITION,
    is_rtx4090_validation_only_policy,
    load_rtx4090_runtime_contract,
    validate_qwen38_dense_config,
    validate_rtx4090_quant_config_manifest,
)
from .shipcard import compute_model_sha


VALIDATION_ONLY_PACKAGE_RECEIPT_SCHEMA = (
    "prismaquant.rtx4090_validation_only_package_receipt.v1"
)


class RTX4090ValidationOnlyArtifactError(RuntimeError):
    """The artifact is not the closed compile-only validation artifact."""


def validate_rtx4090_validation_only_artifact(
    model_dir: str | Path,
    *,
    runtime_contract: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Replay the finalized structural contract without claiming servability."""

    root = Path(model_dir)
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        quant = json.loads(
            (root / "quant_config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RTX4090ValidationOnlyArtifactError(
            f"validation-only artifact JSON cannot be read: {exc}"
        ) from exc
    provenance = quant.get("provenance") if isinstance(quant, Mapping) else None
    stamp = provenance.get("producer_policy") if isinstance(
        provenance, Mapping
    ) else None
    if not is_rtx4090_validation_only_policy(stamp):
        raise RTX4090ValidationOnlyArtifactError(
            "artifact is not immutably stamped UNRELEASABLE_VALIDATION_ONLY"
        )
    contract = load_rtx4090_runtime_contract(
        runtime_contract, where="validation-only Gridbook v11 contract"
    )
    try:
        validate_qwen38_dense_config(
            config, where="validation-only Qwen3.8 config"
        )
        result = validate_rtx4090_quant_config_manifest(
            quant,
            runtime_contract=contract,
            allow_unreleasable_validation_only=True,
            artifact_dir=root,
            where="validation-only finalized RTX4090 manifest",
        )
    except Exception as exc:
        raise RTX4090ValidationOnlyArtifactError(str(exc)) from exc
    from .cluster_campaign_contract import gridbook_runtime_contract_sha256

    return {
        **result,
        "model_sha": compute_model_sha(root),
        "runtime_contract_sha256": gridbook_runtime_contract_sha256(contract),
        "artifact_disposition": RTX4090_VALIDATION_ONLY_DISPOSITION,
        "release_eligible": False,
        "serving_evidence_emitted": False,
    }


def build_validation_only_package_receipt(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the local structural qualification without creating ship evidence."""

    if (
        result.get("artifact_disposition")
        != RTX4090_VALIDATION_ONLY_DISPOSITION
        or result.get("release_eligible") is not False
        or result.get("serving_evidence_emitted") is not False
    ):
        raise RTX4090ValidationOnlyArtifactError(
            "validation-only package result has an unsafe disposition"
        )
    source_census = result.get("source_census")
    runtime_contract_sha256 = result.get("runtime_contract_sha256")
    if not isinstance(source_census, Mapping):
        raise RTX4090ValidationOnlyArtifactError(
            "validation-only package result has no finalized source census"
        )
    if (
        not isinstance(runtime_contract_sha256, str)
        or len(runtime_contract_sha256) != 64
        or any(
            char not in "0123456789abcdef"
            for char in runtime_contract_sha256
        )
    ):
        raise RTX4090ValidationOnlyArtifactError(
            "validation-only package result has no runtime contract identity"
        )
    from .cluster_campaign_contract import canonical_sha256

    body: dict[str, Any] = {
        "schema": VALIDATION_ONLY_PACKAGE_RECEIPT_SCHEMA,
        "artifact_disposition": RTX4090_VALIDATION_ONLY_DISPOSITION,
        "release_eligible": False,
        "serving_evidence_emitted": False,
        "runtime_contract_sha256": runtime_contract_sha256,
        "validation": dict(result),
    }
    return {**body, "identity_sha256": canonical_sha256(body)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("--runtime-contract", required=True)
    parser.add_argument(
        "--receipt",
        help="durably write the sealed validation-only package receipt",
    )
    args = parser.parse_args(argv)
    result = validate_rtx4090_validation_only_artifact(
        args.model_dir, runtime_contract=args.runtime_contract
    )
    if args.receipt:
        from .cluster_transport import write_exact_bytes_no_clobber

        receipt = build_validation_only_package_receipt(result)
        payload = (
            json.dumps(
                receipt,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        write_exact_bytes_no_clobber(Path(args.receipt), payload)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
