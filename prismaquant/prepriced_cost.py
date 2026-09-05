"""Read-only preflight for an explicitly supplied pipeline cost table.

This is an intake gate, not a new cost producer or an acceptance override.
The existing schema, currency, research-provenance and Tessera Hessian owners
decide their own contracts. A model reference is checked exactly as recorded;
it is not a checkpoint-content attestation. Allocator and export gates still
own per-unit coverage, renderer identity and serving eligibility.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any


REPORT_SCHEMA = "prismaquant.prepriced_cost_input.v1"


def _input_path(path: str | os.PathLike) -> Path:
    supplied = Path(path)
    try:
        resolved = supplied.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError("not a regular file")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"prepriced cost input {supplied}: {exc}") from exc
    return resolved


def _sha256(path: Path) -> str:
    from .shipcard import file_sha256

    digest = file_sha256(path)
    if digest is None:
        raise ValueError(f"prepriced cost input {path}: cannot read sha256")
    return digest


def _require_sha256(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{where}: sha256 must be a full lowercase SHA-256 digest")
    return value


def _model_binding(payload: Mapping, model: str) -> dict[str, Any]:
    """Follow the metadata edges written by existing cost producers only.

    Monolithic/incremental cost writes ``meta.model``; AURA, the Tessera
    campaign and empirical expert cost write ``provenance.model``. Incremental
    merge retains each shard's metadata in ``meta.shards`` (including its
    ``incremental_shard`` stamp). Production render wraps its baseline metadata
    in ``meta.baseline_meta``. No model-name or source-path heuristic is used.
    """
    if not isinstance(model, str) or not model:
        raise ValueError("requested model must be an explicit nonempty model reference")
    fields: dict[str, str] = {}
    active: set[int] = set()

    def visit(meta: Any, where: str) -> None:
        if not isinstance(meta, Mapping):
            raise ValueError(f"{where}: model metadata must be a mapping")
        if id(meta) in active:
            raise ValueError(f"{where}: model metadata contains a cycle")
        active.add(id(meta))
        try:
            if "model" in meta:
                reference = meta["model"]
                if not isinstance(reference, str) or not reference:
                    raise ValueError(f"{where}.model must be a nonempty model reference")
                if reference != model:
                    raise ValueError(
                        f"{where}.model={reference!r} does not match requested "
                        f"model={model!r}; model references must match exactly")
                fields[f"{where}.model"] = reference
            for key in ("baseline_meta", "incremental_shard"):
                if meta.get(key) is not None:
                    visit(meta[key], f"{where}.{key}")
            if "shards" in meta:
                shards = meta["shards"]
                if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
                    raise ValueError(f"{where}.shards: model metadata must be a sequence")
                for index, shard in enumerate(shards):
                    visit(shard, f"{where}.shards[{index}]")
        finally:
            active.remove(id(meta))

    for key in ("meta", "provenance"):
        if key in payload:
            visit(payload[key], key)
    if not fields:
        raise ValueError(
            "model reference is unstamped: require the producer's meta.model "
            "or provenance.model (including retained baseline/shard metadata); "
            "no checkpoint identity is inferred from the cost file path")
    return {
        "kind": "exact_model_reference",
        "model": model,
        "fields": fields,
        "checkpoint_content_attested": False,
    }


def validate_prepriced_cost(
    path: str | os.PathLike, *, cost_mode: str, model: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an external pickle without changing its bytes or run defaults.

    This accepts the project's trusted cost-pickle input, not untrusted pickle
    data. The exact requested mode must match its existing producer stamp,
    including legacy spelling: currency equivalence does not rewrite identity.
    """
    from .cost_currency import COST_MODE_OBJECTIVE_CURRENCY, require_run_currency
    from .research_cost_acceptance import accepted_cost_provenance
    from .schemas import validate_cost_payload
    from .tessera_menu import assert_uniform_hessian_identity

    input_path = Path(path).absolute()
    source = _input_path(input_path)
    before = _sha256(source)
    if expected_sha256 is not None:
        expected = _require_sha256(expected_sha256, where=f"prepriced cost input {source}")
        if before != expected:
            raise ValueError(
                f"prepriced cost input {source}: sha256 mismatch: expected "
                f"{expected}, read {before}")
    try:
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        validate_cost_payload(payload, str(source))
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance.cost_mode is unstamped; an explicit override requires its producer's mode")
        stamped = provenance.get("cost_mode")
        if not isinstance(stamped, str) or stamped not in COST_MODE_OBJECTIVE_CURRENCY:
            raise ValueError(
                f"provenance.cost_mode={stamped!r} is missing or unknown; "
                f"expected a producer stamp from {sorted(COST_MODE_OBJECTIVE_CURRENCY)}")
        if not isinstance(cost_mode, str) or cost_mode not in COST_MODE_OBJECTIVE_CURRENCY:
            raise ValueError(f"requested COST_MODE={cost_mode!r} names no owned objective")
        if stamped != cost_mode:
            raise ValueError(
                f"provenance.cost_mode={stamped!r} does not match requested "
                f"COST_MODE={cost_mode!r}; reprice for the requested mode or "
                "explicitly select the matching mode, not a rewritten stamp")
        currency = require_run_currency(payload)
        if accepted_cost_provenance(payload) is not None:
            raise ValueError(
                "research-stamped cost input requires the allocator's explicit "
                "research acknowledgement; the pipeline override cannot accept it")
        hessian = assert_uniform_hessian_identity(payload["costs"])
        binding = _model_binding(payload, model)
        usable = sum(
            "error" not in row
            for rows in payload["costs"].values() for row in rows.values())
        if not usable:
            raise ValueError("costs contains no usable cost rows; an external override cannot supply measurements")
        formats = sorted({fmt for rows in payload["costs"].values() for fmt in rows})
        declared = sorted(set(payload.get("formats") or []))
    except Exception as exc:
        raise ValueError(f"prepriced cost input {source}: {exc}") from exc
    after = _sha256(source)
    if _input_path(input_path) != source:
        raise ValueError(f"prepriced cost input {input_path}: resolved path changed during validation")
    if after != before:
        raise ValueError(
            f"prepriced cost input {source}: sha256 changed during validation: "
            f"before {before}, after {after}")
    return {
        "schema": REPORT_SCHEMA,
        "input_path": str(input_path),
        "path": str(source),
        "sha256": before,
        "cost_mode": stamped,
        "formats": formats,
        "declared_formats": declared,
        "units": len(payload["costs"]),
        "usable_rows": usable,
        "currency": currency,
        "tessera_hessian_identity": hessian,
        "model_binding": binding,
    }


def verify_prepriced_cost_report(report_path: str | os.PathLike) -> dict[str, Any]:
    """Recheck the original file immediately before allocator consumption.

    The receipt records preflight, and this step verifies byte stability. It
    does not promote the table to a serving or checkpoint-content attestation.
    """
    path = Path(report_path)
    try:
        report = json.loads(path.read_text())
        if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
            raise ValueError("missing or unsupported report schema")
        reference = report.get("path")
        if not isinstance(reference, str) or not Path(reference).is_absolute():
            raise ValueError("report path must name the original absolute input path")
        original = report.get("input_path")
        if not isinstance(original, str) or not Path(original).is_absolute():
            raise ValueError("report input_path must retain the original absolute argument path")
        expected = _require_sha256(report.get("sha256"), where="report")
        source = _input_path(original)
        if str(source) != reference:
            raise ValueError(
                f"input path {original} resolves to {source}, not the preflight "
                f"path {reference}; the original argument changed after preflight")
        observed = _sha256(source)
        if _input_path(original) != source:
            raise ValueError(f"input path {original} changed during sha256 verification")
        if observed != expected:
            raise ValueError(
                f"input {source}: sha256 mismatch: expected {expected}, read {observed}; "
                "input changed after preflight")
    except Exception as exc:
        raise ValueError(f"prepriced cost report {path}: {exc}") from exc
    return report


def _write_report(path: Path, report: Mapping) -> None:
    source = Path(report["path"])
    # Reject both path aliases and hardlinks before touching the receipt.
    if path.resolve() == source or (path.exists() and path.samefile(source)):
        raise ValueError(f"report {path} aliases the supplied cost input {source}; refusing to overwrite it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", help="existing trusted prepriced cost pickle; never rewritten")
    parser.add_argument("--cost-mode", help="explicit run COST_MODE, matched to the producer stamp")
    parser.add_argument("--model", help="exact source model reference recorded by the cost producer")
    parser.add_argument("--report", help="write a separate local JSON receipt")
    parser.add_argument("--expected-sha256", help="optionally require a previously measured input digest")
    parser.add_argument("--verify-report", help="verify the receipt's input sha256 immediately before allocation")
    args = parser.parse_args(argv)
    if args.verify_report:
        if any(value is not None for value in (args.path, args.cost_mode, args.model, args.report, args.expected_sha256)):
            parser.error("--verify-report cannot be combined with validation or report-writing arguments")
    elif any(value is None for value in (args.path, args.cost_mode, args.model)):
        parser.error("validation requires --path, --cost-mode and --model")
    try:
        if args.verify_report:
            report = verify_prepriced_cost_report(args.verify_report)
        else:
            report = validate_prepriced_cost(
                args.path, cost_mode=args.cost_mode, model=args.model,
                expected_sha256=args.expected_sha256)
            if args.report:
                _write_report(Path(args.report), report)
        print(json.dumps(report, sort_keys=True))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"[prepriced-cost] ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
