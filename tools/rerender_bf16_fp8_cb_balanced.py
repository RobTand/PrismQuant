#!/usr/bin/env python3
"""Re-render the bf16 W8A8 ladder's FP8-CB arms at production tier.

This is the one-variable correction for ``bf16_ladder_lloyd.json``.  It
imports the provenance-locked ``bf16_ladder.py`` (which itself imports
``fp8_ladder.py``), routes the existing ``cb_arm_fp8`` object with the sole
override ``encode_tier="balanced"``, and renders only the six ``fp8_cb@*``
arms.  The Lloyd trellis and scalar comparison arms are copied from the
baseline with every existing field unchanged.  Every output arm receives an
explicit ``encode_tier`` stamp: ``balanced`` for FP8-CB and
``not_applicable`` for comparison families that do not consume that option.

Interpretation of the completed measurement
-------------------------------------------

* If balanced beats max, the 6.0 bpw FP8-CB verdict strengthens.
* If balanced loses to max, the verdict weakens and must be restated at the
  production tier.
* If there is no material difference, the confound is closed and the verdict
  stands as written.

``--dry-run`` reads only small JSON/source metadata: it resolves and verifies
every input, prints the exact tensor and arm plan, and derives a conservative
wall-clock estimate from the historical arms' ``encode_seconds``.  It does not
import torch, open the safetensors corpus, or touch CUDA.

The output path is created by one final atomic rename after every tensor and
every invariant has passed.  No checkpoint is published at that path, so its
existence means the full run completed.

Usage (the same CUDA venv as ``run_bf16_ladder.sh``)::

    /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -u \
      tools/rerender_bf16_fp8_cb_balanced.py --dry-run
    /home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -u \
      tools/rerender_bf16_fp8_cb_balanced.py
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Sequence


RUN_DIR = Path("/home/rob/dq-runs/trellis-bf16-20260829")
LOCKED_LADDER_DIR = Path("/home/rob/dq-runs/trellis-hull-20260828")
BF16_LADDER = LOCKED_LADDER_DIR / "bf16_ladder.py"
FP8_LADDER = LOCKED_LADDER_DIR / "fp8_ladder.py"
MANIFEST = RUN_DIR / "bf16_inputs_manifest.json"
BASELINE = RUN_DIR / "bf16_ladder_lloyd.json"
TIMING_REFERENCE = RUN_DIR / "bf16_ladder_exact_dp.json"
DEFAULT_OUT = RUN_DIR / "bf16_ladder_lloyd_balanced_fp8_cb.json"

SCHEMA = "trellis.bf16_ladder.v1"
CORPUS_SCHEMA = "trellis.bf16_corpus.v1"
ENCODE_TIER = "balanced"
LEGACY_ENCODE_TIER = "max"
NOT_APPLICABLE = "not_applicable"
PRODUCTION_SCALE = "production_row_fp32"
EXPECTED_CALIBRATION_SEED = 20260829
EXPECTED_STAGE6_SEED = 20260824
EXPECTED_LEARNED_CB_SEED = 0
EXPECTED_CORPUS_SHA256 = (
    "05bedda657e42897cad1cbec867c0a3aaa5f3e3a8fd412951437b6994cbca311"
)
EXPECTED_RENDER_ARMS = (
    "fp8_cb@28",
    "fp8_cb@32",
    "fp8_cb@36",
    "fp8_cb@40",
    "fp8_cb@44",
    "fp8_cb@48",
)
EXPECTED_COMPARISON_ARMS = (
    "tcq_e4m3@1.0",
    "tcq_e4m3@2.0",
    "tcq_e4m3@3.0",
    "tcq_e4m3@4.0",
    "tcq_e4m3@5.0",
    "tcq_e4m3@6.0",
    "tcq_e4m3@7.0",
    "scalar_rtn_e4m3@8",
)
EXPECTED_SHA256 = {
    "bf16_ladder": (
        "ba51fb5ca6be916fd26a9b4b86d75a629f396227851919e50be196c1fd6d12f3"
    ),
    "fp8_ladder": (
        "f9c5167905b98fe98a3389a9471cb9bea06e6ced9a1288329ce1b0fb6a92d2a3"
    ),
    "manifest": (
        "fd4aa43670077d879e7c009dfe06dcfa8c613cb43ccad66de118c488a811cfa4"
    ),
    "baseline": (
        "08db0c87220398812026f4c0a8ead8c82a36751322cb263dfe43ff8abba4b1c4"
    ),
    "timing_reference": (
        "2ee6c7cdcb3003f6fd382e3b7f01d4ba049e139558768b2560e2a6d88bb50984"
    ),
}


class DriverError(RuntimeError):
    """A provenance or measurement contract was not satisfied."""


@dataclass(frozen=True)
class Preflight:
    paths: Mapping[str, Path]
    hashes: Mapping[str, str]
    manifest: Mapping[str, Any]
    baseline: Mapping[str, Any]
    tensor_names: tuple[str, ...]
    render_arms: tuple[str, ...]
    comparison_arms: tuple[str, ...]
    historical_encode_seconds: float
    output: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DriverError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DriverError(f"{path}: expected a JSON object")
    return payload


def _resolve_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise DriverError(f"{label} does not resolve: {path}: {exc}") from exc
    if not resolved.is_file():
        raise DriverError(f"{label} is not a regular file: {resolved}")
    return resolved


def _arm_plan(
    report: Mapping[str, Any], tensor_names: Sequence[str], source: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    cells = report.get("cells")
    if not isinstance(cells, dict):
        raise DriverError(f"{source}: cells must be an object")
    if tuple(cells) != tuple(tensor_names):
        raise DriverError(
            f"{source}: cell order/domain differs from the corpus manifest"
        )

    first_arms: tuple[str, ...] | None = None
    for tensor_name in tensor_names:
        cell = cells.get(tensor_name)
        arms = cell.get("arms") if isinstance(cell, dict) else None
        if not isinstance(arms, dict):
            raise DriverError(f"{source}: {tensor_name}.arms must be an object")
        names = tuple(arms)
        if first_arms is None:
            first_arms = names
        elif names != first_arms:
            raise DriverError(f"{source}: arm order/domain varies by tensor")
        for arm_name, row in arms.items():
            if not isinstance(row, dict):
                raise DriverError(
                    f"{source}: {tensor_name}.arms[{arm_name!r}] is not an object"
                )

    if first_arms is None:
        raise DriverError(f"{source}: no tensor cells")
    rendered = tuple(name for name in first_arms if name.startswith("fp8_cb@"))
    comparison = tuple(name for name in first_arms if name not in rendered)
    if rendered != EXPECTED_RENDER_ARMS:
        raise DriverError(
            f"{source}: FP8-CB arm drift: {rendered!r} != "
            f"{EXPECTED_RENDER_ARMS!r}"
        )
    if comparison != EXPECTED_COMPARISON_ARMS:
        raise DriverError(
            f"{source}: comparison arm drift: {comparison!r} != "
            f"{EXPECTED_COMPARISON_ARMS!r}"
        )
    return rendered, comparison


def _assert_timing_reference_identity(
    baseline: Mapping[str, Any],
    timing_reference: Mapping[str, Any],
    tensor_names: Sequence[str],
    render_arms: Sequence[str],
) -> float:
    """Return one estimate; the exact-DP file contains seeded copies."""
    base_cells = baseline["cells"]
    timing_cells = timing_reference.get("cells")
    if timing_reference.get("schema") != SCHEMA:
        raise DriverError("timing reference schema differs from baseline")
    if timing_reference.get("corpus_file_sha256") != EXPECTED_CORPUS_SHA256:
        raise DriverError("timing reference corpus differs from baseline")
    if not isinstance(timing_cells, dict) or set(timing_cells) != set(base_cells):
        raise DriverError("timing reference tensor domain differs from baseline")

    total = 0.0
    for tensor_name in tensor_names:
        timing_arms = timing_cells[tensor_name].get("arms", {})
        for arm_name in render_arms:
            base_row = base_cells[tensor_name]["arms"][arm_name]
            timing_row = timing_arms.get(arm_name)
            if timing_row != base_row:
                raise DriverError(
                    "exact-DP timing reference does not contain the baseline's "
                    f"seeded FP8-CB row: {tensor_name} {arm_name}"
                )
            seconds = base_row.get("encode_seconds")
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(float(seconds))
                or float(seconds) < 0.0
            ):
                raise DriverError(
                    f"invalid encode_seconds at {tensor_name} {arm_name}: "
                    f"{seconds!r}"
                )
            total += float(seconds)
    return total


def _preflight(output_arg: Path) -> Preflight:
    resolved: dict[str, Path] = {
        "bf16_ladder": _resolve_file(BF16_LADDER, "locked bf16 ladder"),
        "fp8_ladder": _resolve_file(FP8_LADDER, "locked fp8 ladder"),
        "manifest": _resolve_file(MANIFEST, "corpus manifest"),
        "baseline": _resolve_file(BASELINE, "Lloyd baseline"),
        "timing_reference": _resolve_file(
            TIMING_REFERENCE, "exact-DP timing reference"
        ),
    }
    hashes: dict[str, str] = {}
    for label, path in resolved.items():
        actual = _sha256_file(path)
        expected = EXPECTED_SHA256[label]
        if actual != expected:
            raise DriverError(
                f"{label} hash drift: {actual} != provenance key {expected}"
            )
        hashes[label] = actual

    manifest = _read_json(resolved["manifest"])
    if manifest.get("schema") != CORPUS_SCHEMA:
        raise DriverError(f"unexpected corpus schema: {manifest.get('schema')!r}")
    if manifest.get("file_sha256") != EXPECTED_CORPUS_SHA256:
        raise DriverError("corpus manifest carries an unexpected file_sha256")
    calibration = manifest.get("calibration")
    if (
        not isinstance(calibration, dict)
        or calibration.get("seed") != EXPECTED_CALIBRATION_SEED
    ):
        raise DriverError("calibration seed differs from the bf16 corpus contract")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise DriverError("corpus manifest has no entries")
    tensor_names = tuple(str(entry.get("name")) for entry in entries)
    if any(not name or name == "None" for name in tensor_names):
        raise DriverError("corpus manifest contains an invalid tensor name")
    if len(set(tensor_names)) != len(tensor_names):
        raise DriverError("corpus manifest contains duplicate tensor names")

    corpus_file = manifest.get("file")
    if not isinstance(corpus_file, str) or not corpus_file:
        raise DriverError("corpus manifest has no file")
    corpus = _resolve_file(resolved["manifest"].parent / corpus_file, "corpus")
    resolved["corpus"] = corpus

    baseline = _read_json(resolved["baseline"])
    if baseline.get("schema") != SCHEMA:
        raise DriverError(f"unexpected baseline schema: {baseline.get('schema')!r}")
    if baseline.get("corpus_file_sha256") != EXPECTED_CORPUS_SHA256:
        raise DriverError("baseline and corpus manifest hashes differ")
    baseline_corpus = baseline.get("corpus")
    if not isinstance(baseline_corpus, str):
        raise DriverError("baseline has no corpus path")
    if Path(baseline_corpus).resolve() != corpus:
        raise DriverError("baseline and manifest resolve to different corpus files")
    if baseline.get("alphabet_mode") != "lloyd":
        raise DriverError("baseline is not the Lloyd alphabet bracket")
    if baseline.get("trellis_scale_coding") != PRODUCTION_SCALE:
        raise DriverError("baseline is not on the production row-FP32 plane")
    if "encode_tier" in baseline:
        raise DriverError(
            "the pinned baseline unexpectedly has an encode_tier stamp; "
            "re-audit its identity before using this correction driver"
        )

    render_arms, comparison_arms = _arm_plan(
        baseline, tensor_names, resolved["baseline"]
    )
    timing_reference = _read_json(resolved["timing_reference"])
    historical_seconds = _assert_timing_reference_identity(
        baseline, timing_reference, tensor_names, render_arms
    )

    output = output_arg.expanduser().resolve()
    try:
        output.parent.resolve(strict=True)
    except OSError as exc:
        raise DriverError(f"output parent does not resolve: {output.parent}") from exc
    if not output.parent.is_dir():
        raise DriverError(f"output parent is not a directory: {output.parent}")
    if output.suffix != ".json":
        raise DriverError(f"output must be a .json file: {output}")
    if output.exists():
        raise DriverError(
            f"output already exists; refusing to overwrite a completion receipt: {output}"
        )
    if output in resolved.values():
        raise DriverError("output aliases an input path")

    return Preflight(
        paths=resolved,
        hashes=hashes,
        manifest=manifest,
        baseline=baseline,
        tensor_names=tensor_names,
        render_arms=render_arms,
        comparison_arms=comparison_arms,
        historical_encode_seconds=historical_seconds,
        output=output,
    )


def _duration(seconds: float) -> str:
    rounded = int(math.ceil(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {secs:02d}s"


def _print_plan(preflight: Preflight, *, dry_run: bool) -> None:
    mode = "DRY RUN (metadata only; torch/CUDA not imported)" if dry_run else "RUN"
    print(f"mode: {mode}")
    for label in (
        "bf16_ladder",
        "fp8_ladder",
        "manifest",
        "corpus",
        "baseline",
        "timing_reference",
    ):
        path = preflight.paths[label]
        digest = preflight.hashes.get(label)
        suffix = f"  sha256={digest}" if digest else "  (not read or hashed)"
        print(f"input {label:16s}: {path}{suffix}")
    print(f"output                 : {preflight.output}")
    print(
        f"corpus_file_sha256      : {EXPECTED_CORPUS_SHA256} "
        "(baseline == manifest; payload hash deferred to queued run)"
    )
    print(f"calibration seed        : {EXPECTED_CALIBRATION_SEED}")
    print(f"locked Stage-6 seed     : {EXPECTED_STAGE6_SEED}")
    print(f"locked learned-CB seed  : {EXPECTED_LEARNED_CB_SEED} (arm not rendered)")
    print("fixed-lattice FP8 seed  : not_applicable (deterministic arm)")
    print(
        f"tensors                 : {len(preflight.tensor_names)} "
        "(manifest order; no subset)"
    )
    for name in preflight.tensor_names:
        print(f"  tensor: {name}")
    print(
        f"render arms per tensor  : {len(preflight.render_arms)} "
        f"at encode_tier={ENCODE_TIER!r}"
    )
    for name in preflight.render_arms:
        print(f"  render: {name}")
    print(
        f"comparison arms copied  : {len(preflight.comparison_arms)} "
        f"with encode_tier={NOT_APPLICABLE!r}"
    )
    for name in preflight.comparison_arms:
        print(f"  copy:   {name}")
    total_renders = len(preflight.tensor_names) * len(preflight.render_arms)
    print(f"total GPU renders       : {total_renders}")
    print(
        "historical wall estimate: "
        f"{preflight.historical_encode_seconds:.3f}s "
        f"({_duration(preflight.historical_encode_seconds)}) for {total_renders} "
        "max-tier FP8-CB arms"
    )
    print(
        "estimate provenance     : sum of arms[*].encode_seconds in the Lloyd "
        "JSON, verified byte-for-byte against its seeded copies in exact-DP; "
        "setup/final-write overhead excluded"
    )
    if dry_run:
        print("tensor payload loaded   : no")
        print("GPU touched             : no")


def _load_locked_bf16(preflight: Preflight) -> ModuleType:
    os.environ["E4M3_ALPHABET"] = "lloyd"
    locked_dir = str(preflight.paths["bf16_ladder"].parent)
    sys.path.insert(0, locked_dir)
    spec = importlib.util.spec_from_file_location(
        "_prismaquant_locked_bf16_ladder", preflight.paths["bf16_ladder"]
    )
    if spec is None or spec.loader is None:
        raise DriverError("cannot construct the locked bf16_ladder import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_runtime(
    ladder: ModuleType, preflight: Preflight,
) -> tuple[Mapping[str, Any], tuple[str, ...], Mapping[str, Mapping[str, Any]]]:
    if Path(ladder.__file__).resolve() != preflight.paths["bf16_ladder"]:
        raise DriverError("bf16_ladder imported from an unexpected path")
    imported_fp8 = Path(ladder.F.__file__).resolve()
    if imported_fp8 != preflight.paths["fp8_ladder"]:
        raise DriverError(f"fp8_ladder imported from {imported_fp8}")
    if ladder.SCHEMA != SCHEMA:
        raise DriverError(f"runtime schema drift: {ladder.SCHEMA!r}")
    if Path(ladder.INPUT).resolve() != preflight.paths["corpus"]:
        raise DriverError("runtime bf16 ladder resolves a different corpus")
    if Path(ladder.MANIFEST).resolve() != preflight.paths["manifest"]:
        raise DriverError("runtime bf16 ladder resolves a different manifest")
    if ladder.PRODUCTION_SCALE != PRODUCTION_SCALE:
        raise DriverError("runtime production scale contract drifted")
    runtime_arms = tuple(f"fp8_cb@{k}" for k in ladder.FP8_CB_RUNGS)
    if runtime_arms != preflight.render_arms:
        raise DriverError(f"runtime FP8-CB menu drift: {runtime_arms!r}")
    if ladder.F.CB_ENCODE_TIER != LEGACY_ENCODE_TIER:
        raise DriverError(
            "locked fp8 ladder no longer identifies the historical tier as max"
        )
    if ladder.F.H.SEED != EXPECTED_STAGE6_SEED:
        raise DriverError(f"locked Stage-6 seed drift: {ladder.F.H.SEED!r}")
    if ladder.F.H.LEARNED_CB_SEED != EXPECTED_LEARNED_CB_SEED:
        raise DriverError(
            f"locked learned-CB seed drift: {ladder.F.H.LEARNED_CB_SEED!r}"
        )
    if ladder.cb_arm_fp8 is not ladder.F.cb_arm_fp8:
        raise DriverError("bf16_ladder no longer imports the fp8 arm object directly")
    signature = inspect.signature(ladder.F.cb_arm_fp8)
    if "encode_tier" not in signature.parameters:
        raise DriverError("locked fp8 arm has no explicit encode_tier keyword")

    # This validates the frozen Stage-6 source closure against its canary.  It
    # is deliberately the imported driver's existing check, not a local copy.
    ladder.F.H.source_hashes()

    if not ladder.torch.cuda.is_available():
        raise DriverError("CUDA is required; refusing the CPU measurement path")
    if ladder.free_gib(ladder.CORPUS) < 20.0:
        raise DriverError("below the locked bf16 driver's 20 GiB disk floor")

    manifest, names, entries = ladder.load_corpus()
    names_tuple = tuple(names)
    if manifest != preflight.manifest:
        raise DriverError("runtime manifest parse differs from dry preflight")
    if names_tuple != preflight.tensor_names:
        raise DriverError("runtime tensor order/domain differs from preflight")
    return manifest, names_tuple, entries


def _cell_metadata(cell: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: value for key, value in cell.items() if key != "arms"}


def _git_head(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DriverError(f"cannot resolve driver Git identity: {exc}") from exc


def _prepare_report(preflight: Preflight) -> dict[str, Any]:
    report = copy.deepcopy(preflight.baseline)
    report["source_baseline_generated"] = report["generated"]
    report["source_baseline_host"] = report["host"]
    report["source_baseline_prismaquant_commit"] = report["prismaquant_commit"]
    report["encode_tier"] = ENCODE_TIER
    report["encode_tier_by_arm_family"] = {
        "fp8_cb": ENCODE_TIER,
        "tcq_e4m3": NOT_APPLICABLE,
        "scalar_rtn_e4m3": NOT_APPLICABLE,
    }
    report["source_baseline"] = str(preflight.paths["baseline"])
    report["source_baseline_sha256"] = preflight.hashes["baseline"]
    report["source_baseline_fp8_cb_encode_tier"] = LEGACY_ENCODE_TIER
    report["timing_reference"] = str(preflight.paths["timing_reference"])
    report["timing_reference_sha256"] = preflight.hashes["timing_reference"]
    report["bf16_ladder"] = str(preflight.paths["bf16_ladder"])
    report["bf16_ladder_sha256"] = preflight.hashes["bf16_ladder"]
    report["fp8_ladder"] = str(preflight.paths["fp8_ladder"])
    report["fp8_ladder_sha256"] = preflight.hashes["fp8_ladder"]
    report["corpus_manifest"] = str(preflight.paths["manifest"])
    report["corpus_manifest_sha256"] = preflight.hashes["manifest"]
    report["seeds"] = {
        "calibration": EXPECTED_CALIBRATION_SEED,
        "locked_stage6": EXPECTED_STAGE6_SEED,
        "locked_learned_cb": EXPECTED_LEARNED_CB_SEED,
        "fixed_lattice_fp8_cb": NOT_APPLICABLE,
    }
    report["rerendered_arms"] = list(preflight.render_arms)
    report["copied_comparison_arms"] = list(preflight.comparison_arms)
    report["historical_fp8_cb_encode_seconds"] = (
        preflight.historical_encode_seconds
    )
    report["contention_note"] = (
        "encode_seconds are recorded for queue sizing only; no performance "
        "claim is made without paired in-process and box telemetry"
    )

    for tensor_name in preflight.tensor_names:
        arms = report["cells"][tensor_name]["arms"]
        for arm_name in preflight.comparison_arms:
            arms[arm_name]["encode_tier"] = NOT_APPLICABLE
    return report


def _render(preflight: Preflight) -> dict[str, Any]:
    # Dry-run intentionally avoids this 1.2 GB read.  The funded run verifies
    # the artifact-level identity before importing torch or touching CUDA;
    # per-tensor weight and importance hashes are then replayed by load_tensor.
    corpus_sha256 = _sha256_file(preflight.paths["corpus"])
    if corpus_sha256 != EXPECTED_CORPUS_SHA256:
        raise DriverError(
            f"corpus payload hash drift: {corpus_sha256} != "
            f"{EXPECTED_CORPUS_SHA256}"
        )
    print(f"verified corpus payload sha256={corpus_sha256}", flush=True)
    ladder = _load_locked_bf16(preflight)
    _, names, entries = _validate_runtime(ladder, preflight)
    report = _prepare_report(preflight)

    # B.measure resolves this imported global at call time.  Partial preserves
    # the locked arm implementation and changes exactly its one keyword.
    locked_cb_arm = ladder.cb_arm_fp8
    ladder.cb_arm_fp8 = partial(locked_cb_arm, encode_tier=ENCODE_TIER)

    def forbid_comparison_render(*_args: Any, **_kwargs: Any) -> None:
        raise DriverError("comparison-arm renderer was invoked")

    # The skip set is the primary selection contract.  These sentinels make
    # "render only FP8-CB" fail closed if the locked loop ever violates it.
    ladder.trellis_arm_e4m3 = forbid_comparison_render
    ladder.scalar_rtn_e4m3 = forbid_comparison_render
    args = SimpleNamespace(cuda=True, trellis_scale_coding=PRODUCTION_SCALE)
    skip = set(preflight.comparison_arms)

    started = time.monotonic()
    for index, tensor_name in enumerate(names, 1):
        print(f"[{index}/{len(names)}] {tensor_name}", flush=True)
        measured = ladder.measure(tensor_name, entries[tensor_name], args, skip)
        baseline_cell = preflight.baseline["cells"][tensor_name]
        if _cell_metadata(measured) != _cell_metadata(baseline_cell):
            differing = sorted(
                key
                for key in set(measured) | set(baseline_cell)
                if key != "arms" and measured.get(key) != baseline_cell.get(key)
            )
            raise DriverError(
                f"{tensor_name}: non-arm measurement context drifted: {differing}"
            )
        measured_arms = measured.get("arms")
        if not isinstance(measured_arms, dict):
            raise DriverError(f"{tensor_name}: measured arms are not an object")
        if tuple(measured_arms) != preflight.render_arms:
            raise DriverError(
                f"{tensor_name}: rendered arm set/order drifted: "
                f"{tuple(measured_arms)!r}"
            )

        output_arms = report["cells"][tensor_name]["arms"]
        for arm_name in preflight.render_arms:
            row = measured_arms[arm_name]
            baseline_row = baseline_cell["arms"][arm_name]
            if row.get("arm") != baseline_row.get("arm"):
                raise DriverError(f"{tensor_name} {arm_name}: arm family drifted")
            if row.get("footprint") != baseline_row.get("footprint"):
                raise DriverError(f"{tensor_name} {arm_name}: footprint drifted")
            stamped = dict(row)
            stamped["encode_tier"] = ENCODE_TIER
            output_arms[arm_name] = stamped

    report["generated"] = datetime.now(timezone.utc).isoformat()
    report["host"] = socket.gethostname()
    report["prismaquant_commit"] = _git_head(
        Path(__file__).resolve().parents[1]
    )
    report["driver"] = str(Path(__file__).resolve())
    report["driver_sha256"] = _sha256_file(Path(__file__).resolve())
    report["corpus_payload_sha256_verified"] = corpus_sha256
    report["rerender_elapsed_seconds"] = time.monotonic() - started
    return report


def _validate_complete_report(
    report: Mapping[str, Any], preflight: Preflight,
) -> None:
    if report.get("schema") != SCHEMA:
        raise DriverError("completed report schema drifted")
    if report.get("encode_tier") != ENCODE_TIER:
        raise DriverError("completed report lost its top-level encode_tier")
    if report.get("corpus") != preflight.baseline.get("corpus"):
        raise DriverError("completed report changed the corpus path")
    if report.get("corpus_file_sha256") != EXPECTED_CORPUS_SHA256:
        raise DriverError("completed report changed the corpus_file_sha256")
    if report.get("corpus_payload_sha256_verified") != EXPECTED_CORPUS_SHA256:
        raise DriverError("completed report lost the verified corpus payload hash")
    cells = report.get("cells")
    if not isinstance(cells, dict) or tuple(cells) != preflight.tensor_names:
        raise DriverError("completed report changed the tensor domain/order")

    for tensor_name in preflight.tensor_names:
        output_cell = cells[tensor_name]
        baseline_cell = preflight.baseline["cells"][tensor_name]
        if _cell_metadata(output_cell) != _cell_metadata(baseline_cell):
            raise DriverError(f"{tensor_name}: output cell metadata drifted")
        output_arms = output_cell.get("arms")
        baseline_arms = baseline_cell["arms"]
        if not isinstance(output_arms, dict) or tuple(output_arms) != tuple(
            baseline_arms
        ):
            raise DriverError(f"{tensor_name}: output arm domain/order drifted")
        for arm_name in preflight.render_arms:
            row = output_arms[arm_name]
            if row.get("encode_tier") != ENCODE_TIER:
                raise DriverError(f"{tensor_name} {arm_name}: missing balanced stamp")
            if row.get("arm") != baseline_arms[arm_name].get("arm"):
                raise DriverError(f"{tensor_name} {arm_name}: arm family changed")
            if row.get("footprint") != baseline_arms[arm_name].get("footprint"):
                raise DriverError(f"{tensor_name} {arm_name}: footprint changed")
        for arm_name in preflight.comparison_arms:
            row = dict(output_arms[arm_name])
            if row.pop("encode_tier", None) != NOT_APPLICABLE:
                raise DriverError(
                    f"{tensor_name} {arm_name}: missing not-applicable tier stamp"
                )
            if row != baseline_arms[arm_name]:
                raise DriverError(
                    f"{tensor_name} {arm_name}: copied comparison arm changed"
                )


def _publish_as_final_action(
    output: Path, serialized: str,
) -> None:
    """Publish atomically; the final rename is intentionally the last action."""
    temporary = output.with_name(f".{output.name}.{os.getpid()}.incomplete")
    if temporary.exists():
        raise DriverError(f"temporary output already exists: {temporary}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    print(f"all invariants passed; final action will publish {output}", flush=True)
    os.replace(temporary, output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"completion JSON (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify/print metadata-only plan; never import torch or open corpus",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        preflight = _preflight(args.out)
        _print_plan(preflight, dry_run=args.dry_run)
        if args.dry_run:
            return 0
        report = _render(preflight)
        _validate_complete_report(report, preflight)
        serialized = json.dumps(
            report,
            indent=1,
            allow_nan=False,
        ) + "\n"
        # No print, read, validation, or cleanup may follow this call.
        _publish_as_final_action(preflight.output, serialized)
        return 0
    except DriverError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
