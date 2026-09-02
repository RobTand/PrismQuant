#!/usr/bin/env python3
"""CPU-only preflight for the AURA-on-CB re-price campaigns.

This module never imports transformers or a CUDA entrypoint. Optional immutable
bundle/selection validation may load the repository's CPU tensor validators,
but ``CUDA_VISIBLE_DEVICES`` is cleared before argument handling. It inventories
the immutable inputs and checks whether the repository exposes the
streamed/checkpointed capabilities required by the launch driver. Missing
output artifacts are reported as BUILD items; unsafe or unsupported execution
paths are BLOCK items.

The current DSv4 source is much larger than the Spark's unified-memory pool.
Consequently a whole-model fallback is never an acceptable way to make this
preflight pass.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
from typing import Iterable, Sequence


DSV4_RUN_ROOT = Path("/home/rob/dq-runs/dsv4-flash-0731")
DEFAULT_DATASET = Path("/home/rob/dq-runs/calibration/diverse-v1.jsonl")
DEFAULT_GPU_LOCK = Path("/home/rob/dq-runs/gpu.lock")

DSV4_TOTAL_UNITS = 33_325
DSV4_EXPERT_UNITS = 43 * 256 * 3
DSV4_NONEXPERT_UNITS = DSV4_TOTAL_UNITS - DSV4_EXPERT_UNITS
DSV4_EXPERT_2048X4096_UNITS = 43 * 256 * 2
DSV4_EXPERT_4096X2048_UNITS = 43 * 256
DSV4_NVFP4_RUNGS = tuple(range(12, 19))
# FP8-CB is legal only at k % 4 == 0 (gridbook K1.2 fused mid-M kernel law:
# type_size = 4k must stay a 16-byte-multiple TMA extent and CbSubW = k/4 is
# the real sub-table width only there).  serving_profile_specs/nvfp4_cb.json
# backs [28, 32, 36, 40, 44, 48] for runtimes 0.5.0..0.8.4. Routed experts
# are additionally capped at K33 by the byte-exact source-payload ceiling,
# leaving two rungs.  NVFP4-CB is outside the law and stays contiguous.
DSV4_EXPERT_FP8_RUNGS = (28, 32)
DSV4_NONEXPERT_FP8_RUNGS = (28, 32, 36, 40, 44, 48)
DSV4_FP8_LEARNED_RUNGS = tuple(k for k in DSV4_NONEXPERT_FP8_RUNGS if k <= 46)
DSV4_FP8_LATTICE_RUNGS = tuple(k for k in DSV4_NONEXPERT_FP8_RUNGS if k > 46)
DSV4_ROLE_UNITS = (
    ("gate_proj", 11_051),
    ("up_proj", 11_051),
    ("down_proj", 11_051),
    ("wq_a", 43),
    ("wq_b", 43),
    ("wkv", 43),
    ("wo_b", 43),
)
DSV4_NONEXPERT_ROLE_UNITS = tuple((role, 43) for role, _ in DSV4_ROLE_UNITS)
DENSE_FP8_RUNGS = tuple(range(28, 49))


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    path: str | None = None


def fp8_format(rung: int) -> str:
    return f"FP8_CB_K{int(rung)}"


def nvfp4_format(rung: int) -> str:
    return f"NVFP4_CB_K{int(rung)}"


def campaign_accounting() -> dict[str, object]:
    """Return the exact DSv4 legal menu and bounded anchor render domain."""

    nvfp4_formats = [nvfp4_format(k) for k in DSV4_NVFP4_RUNGS]
    expert_fp8_formats = [fp8_format(k) for k in DSV4_EXPERT_FP8_RUNGS]
    nonexpert_fp8_formats = [
        fp8_format(k) for k in DSV4_NONEXPERT_FP8_RUNGS
    ]
    expert_source_format = "MXFP4_SOURCE"
    nonexpert_source_format = "FP8_BLOCK_UE8M0_SOURCE"

    nvfp4_cells = DSV4_TOTAL_UNITS * len(DSV4_NVFP4_RUNGS)
    expert_fp8_cells = DSV4_EXPERT_UNITS * len(DSV4_EXPERT_FP8_RUNGS)
    nonexpert_fp8_cells = DSV4_NONEXPERT_UNITS * len(
        DSV4_NONEXPERT_FP8_RUNGS
    )
    fp8_learned_cells = (
        expert_fp8_cells
        + DSV4_NONEXPERT_UNITS * len(DSV4_FP8_LEARNED_RUNGS)
    )
    fp8_lattice_cells = (
        DSV4_NONEXPERT_UNITS * len(DSV4_FP8_LATTICE_RUNGS)
    )
    source_terminal_cells = DSV4_TOTAL_UNITS
    expert_cells = DSV4_EXPERT_UNITS * (
        len(DSV4_NVFP4_RUNGS) + len(DSV4_EXPERT_FP8_RUNGS) + 1
    )
    nonexpert_cells = DSV4_NONEXPERT_UNITS * (
        len(DSV4_NVFP4_RUNGS) + len(DSV4_NONEXPERT_FP8_RUNGS) + 1
    )

    anchor_segments = [
        {
            "family": family,
            "role": role,
            "equivalence_class": basis,
            # CB plugin vocabulary retained for campaign accounting/reporting;
            # the generic anchored-cost core sees only equivalence_class.
            "basis": basis,
            "units": units,
            "renders": units,
        }
        for family, basis, role_units in (
            ("NVFP4_CB", "lattice", DSV4_ROLE_UNITS),
            ("FP8_CB", "learned", DSV4_ROLE_UNITS),
            ("FP8_CB", "lattice", DSV4_NONEXPERT_ROLE_UNITS),
        )
        for role, units in role_units
    ]
    anchor_renders = sum(int(row["renders"]) for row in anchor_segments)
    return {
        "total_units": DSV4_TOTAL_UNITS,
        "expert_units": DSV4_EXPERT_UNITS,
        "expert_2048x4096_units": DSV4_EXPERT_2048X4096_UNITS,
        "expert_4096x2048_units": DSV4_EXPERT_4096X2048_UNITS,
        "nonexpert_units": DSV4_NONEXPERT_UNITS,
        "nvfp4_formats": nvfp4_formats,
        "expert_fp8_formats": expert_fp8_formats,
        "nonexpert_fp8_formats": nonexpert_fp8_formats,
        "expert_source_format": expert_source_format,
        "nonexpert_source_format": nonexpert_source_format,
        "expert_formats": [
            *nvfp4_formats,
            *expert_fp8_formats,
            expert_source_format,
        ],
        "nonexpert_formats": [
            *nvfp4_formats,
            *nonexpert_fp8_formats,
            nonexpert_source_format,
        ],
        "expert_cells": expert_cells,
        "nonexpert_cells": nonexpert_cells,
        "nvfp4_cells": nvfp4_cells,
        "expert_fp8_cells": expert_fp8_cells,
        "nonexpert_fp8_cells": nonexpert_fp8_cells,
        "fp8_learned_cells": fp8_learned_cells,
        "fp8_lattice_cells": fp8_lattice_cells,
        "fp8_cells": expert_fp8_cells + nonexpert_fp8_cells,
        "source_terminal_cells": source_terminal_cells,
        "candidate_cells": expert_cells + nonexpert_cells,
        "segment_key_fields": ["family", "role", "equivalence_class"],
        "plugin_equivalence_vocabulary": "codebook_basis",
        "anchor_segments": anchor_segments,
        "anchor_renders": anchor_renders,
        "encode_seconds_formula": (
            "sum(t_anchor[qname,family,role,equivalence_class] for 66,951 "
            "legal unit-family-equivalence anchors) + t_panel + t_validation"
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_check(
    name: str,
    path: Path,
    *,
    size: int | None = None,
    sha256: str | None = None,
    verify_hashes: bool = False,
) -> Check:
    if not path.is_file():
        return Check(name, "BLOCK", "required file is missing", str(path))
    observed_size = path.stat().st_size
    if size is not None and observed_size != size:
        return Check(
            name,
            "BLOCK",
            f"size drift: expected {size:,} bytes, found {observed_size:,}",
            str(path),
        )
    detail = f"{observed_size:,} bytes"
    if sha256 is not None:
        if verify_hashes:
            observed_hash = _sha256(path)
            if observed_hash != sha256:
                return Check(
                    name,
                    "BLOCK",
                    f"sha256 drift: expected {sha256}, found {observed_hash}",
                    str(path),
                )
            detail += f", sha256={observed_hash}"
        else:
            detail += f", pinned sha256={sha256} (hash read skipped)"
    return Check(name, "PASS", detail, str(path))


def _directory_check(name: str, path: Path) -> Check:
    if not path.is_dir():
        return Check(name, "BLOCK", "required directory is missing", str(path))
    return Check(name, "PASS", "directory exists", str(path))


def _learned_bundle_check(repo: Path, name: str, path: Path) -> Check:
    """CPU-load the immutable bundle and verify its internal digest graph."""

    if not path.is_file():
        return Check(name, "BLOCK", "required file is missing", str(path))
    try:
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        from prismaquant.cb_learned_bundle import load_bundle

        bundle = load_bundle(path)
        return Check(
            name,
            "PASS",
            f"schema/digests/rung policy verified; cells="
            f"{len(bundle.manifest['cells']):,}; "
            f"bundle_sha256={bundle.bundle_content_sha256}",
            str(path),
        )
    except Exception as exc:
        return Check(name, "BLOCK", f"immutable bundle validation failed: {exc}", str(path))


def _routed_selection_check(repo: Path, path: Path) -> Check:
    """Validate the explicit DSv4 on-law routed-book selection (K28/K32)."""

    if not path.is_file():
        return Check(
            "DSv4 routed-book selection", "BLOCK", "required file is missing", str(path)
        )
    try:
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        from prismaquant.cb_banked_books import load_routed_moe_cbl_selection

        selection = load_routed_moe_cbl_selection(path)
        observed = {
            (cell.layer, cell.projection, cell.rung)
            for cell in selection.cells
        }
        expected = {
            (layer, projection, rung)
            for layer in range(43)
            for projection in ("gate_proj", "up_proj", "down_proj")
            for rung in DSV4_EXPERT_FP8_RUNGS
        }
        if observed != expected:
            missing = sorted(expected - observed)[:8]
            extra = sorted(observed - expected)[:8]
            return Check(
                "DSv4 routed-book selection",
                "BLOCK",
                f"coverage differs: missing={missing}, extra={extra}",
                str(path),
            )
        return Check(
            "DSv4 routed-book selection",
            "PASS",
            f"{len(observed):,} cells; sha256={selection.content_sha256}",
            str(path),
        )
    except Exception as exc:
        return Check(
            "DSv4 routed-book selection",
            "BLOCK",
            f"selection validation failed: {exc}",
            str(path),
        )


def _bundle_selection_binding_check(
    repo: Path, bundle_path: Path, selection_path: Path,
) -> Check:
    """Bind routed bank origins in the bundle to the supplied selection."""
    name = "DSv4 bundle/selection identity"
    if not bundle_path.is_file() or not selection_path.is_file():
        return Check(
            name,
            "BLOCK",
            "bundle and routed selection must both exist before identity "
            "binding can be verified",
        )
    try:
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        from prismaquant.cb_banked_books import (
            load_routed_moe_cbl_selection,
        )
        from prismaquant.dsv4_aura_cb_reprice import (
            _validate_routed_bundle_selection_identity,
        )

        selection = load_routed_moe_cbl_selection(selection_path)
        report = _validate_routed_bundle_selection_identity(
            bundle_path, selection.content_sha256,
        )
        return Check(
            name,
            "PASS",
            f"{report['routed_learned_origin_cells']:,} routed learned "
            f"bundle cells bind selection sha256={selection.content_sha256}",
            str(bundle_path),
        )
    except Exception as exc:
        return Check(
            name,
            "BLOCK",
            f"bundle/selection identity differs: {exc}",
            str(bundle_path),
        )


def _source_checkpoint_checks(source: Path) -> list[Check]:
    checks: list[Check] = [
        _directory_check("source checkpoint", source),
        _file_check("source config", source / "config.json"),
    ]
    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        single = source / "model.safetensors"
        checks.append(_file_check("source safetensors", single))
        return checks
    checks.append(_file_check("source index", index_path))
    try:
        payload = json.loads(index_path.read_text())
        weight_map = payload.get("weight_map") or {}
        shards = sorted(set(map(str, weight_map.values())))
        missing = [name for name in shards if not (source / name).is_file()]
        empty = [
            name
            for name in shards
            if (source / name).is_file() and (source / name).stat().st_size == 0
        ]
        if missing or empty:
            checks.append(
                Check(
                    "source shard coverage",
                    "BLOCK",
                    f"missing={missing[:8]}, empty={empty[:8]}",
                    str(index_path),
                )
            )
        else:
            indexed = payload.get("metadata", {}).get("total_size")
            checks.append(
                Check(
                    "source shard coverage",
                    "PASS",
                    f"{len(shards)} shards, {len(weight_map):,} indexed tensors, "
                    f"metadata.total_size={indexed}",
                    str(index_path),
                )
            )
    except (OSError, ValueError, TypeError) as exc:
        checks.append(
            Check("source shard coverage", "BLOCK", f"unreadable index: {exc}")
        )
    return checks


def _probe_check(path: Path) -> Check:
    base = _file_check(
        "CE sensitivity probe",
        path,
        size=7_383_774,
        sha256=(
            "a0fdbb62c075fdc2d3fa3518e22ef87226aa4f619989d2e8162a2e3f9eda0535"
        ),
    )
    if base.status != "PASS":
        return base
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        stats = payload.get("stats") or {}
        meta = payload.get("meta") or {}
        if len(stats) != DSV4_TOTAL_UNITS:
            return Check(
                base.name,
                "BLOCK",
                f"expected {DSV4_TOTAL_UNITS:,} stats, found {len(stats):,}",
                str(path),
            )
        expert_shapes = Counter(
            (row.get("out_features"), row.get("in_features"))
            for name, row in stats.items()
            if ".experts." in str(name)
        )
        expected_shapes = Counter(
            {
                (2048, 4096): DSV4_EXPERT_2048X4096_UNITS,
                (4096, 2048): DSV4_EXPERT_4096X2048_UNITS,
            }
        )
        if expert_shapes != expected_shapes:
            return Check(
                base.name,
                "BLOCK",
                f"expert orientation census differs: {dict(expert_shapes)}",
                str(path),
            )
        detail = (
            f"{len(stats):,} stats; {meta.get('nsamples')}x"
            f"{meta.get('seqlen')}; device_map={meta.get('device_map')}; "
            "experts=22,016x(2048x4096)+11,008x(4096x2048); "
            "CE empirical-Fisher/h_trace only (not KL-adjoint)"
        )
        return Check(base.name, "PASS", detail, str(path))
    except Exception as exc:  # local trusted artifact; surface schema drift
        return Check(base.name, "BLOCK", f"cannot read probe: {exc}", str(path))


def _activation_cache_check(path: Path) -> Check:
    if not path.is_dir():
        return Check(
            "activation cache", "BLOCK", "required directory is missing", str(path)
        )
    count = 0
    nbytes = 0
    for item in path.iterdir():
        if item.is_file() and item.suffix == ".pt":
            count += 1
            nbytes += item.stat().st_size
    expected_count = 33_274
    expected_bytes = 25_292_991_482
    status = (
        "PASS"
        if (count, nbytes) == (expected_count, expected_bytes)
        else "BLOCK"
    )
    return Check(
        "activation cache",
        status,
        f"{count:,} .pt files / {nbytes:,} bytes; expected "
        f"{expected_count:,} / {expected_bytes:,} (51 never-routed rows absent)",
        str(path),
    )


def _col_weight_provenance_check(path: Path) -> Check:
    if not path.is_file():
        return Check(
            "column-weight provenance", "BLOCK", "required sidecar is missing", str(path)
        )
    try:
        payload = json.loads(path.read_text())
        neutral = payload.get("names") or []
        rule = payload.get("rule")
        status = (
            "PASS"
            if len(neutral) == 51
            and rule == "unrouted_expert_neutral_prior:layer_routed_mean"
            else "BLOCK"
        )
        return Check(
            "column-weight provenance",
            status,
            f"neutral_fills={len(neutral)}, rule={rule!r}",
            str(path),
        )
    except (OSError, ValueError, TypeError) as exc:
        return Check(
            "column-weight provenance", "BLOCK", f"unreadable JSON: {exc}", str(path)
        )


def _repo_text(repo: Path, relative: str) -> str:
    path = repo / relative
    try:
        return path.read_text()
    except OSError:
        return ""


def _concrete_top_level_function(
    source: str,
    name: str,
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, str]:
    """Return a non-placeholder top-level function without importing it.

    Preflight must remain CPU-only, so importing the campaign provider merely
    to ask whether an attribute is callable is not acceptable.  Parsing also
    lets this check reject the prior false-positive state where the CLI shim
    existed but delegated to a symbol that had never been implemented.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return None, f"provider is not valid Python: {exc}"
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ),
        None,
    )
    if function is None:
        return None, f"provider does not define top-level {name}"

    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    placeholder_nodes = (ast.Pass, ast.Raise)
    placeholder = not body or all(
        isinstance(node, placeholder_nodes)
        or (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and node.value.value is Ellipsis
        )
        for node in body
    )
    if placeholder or not any(
        isinstance(node, ast.Call) for node in ast.walk(function)
    ):
        return None, f"provider {name} is only a placeholder, not orchestration"
    return function, ""


def _function_accepts_call(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    call: ast.Call,
) -> bool:
    """Statically check the ordinary arguments used by the thin CLI shim."""

    positional = [*function.args.posonlyargs, *function.args.args]
    if any(isinstance(argument, ast.Starred) for argument in call.args):
        return False
    if len(call.args) > len(positional) and function.args.vararg is None:
        return False
    known_keywords = {
        argument.arg for argument in (*positional, *function.args.kwonlyargs)
    }
    supplied_keywords = {
        keyword.arg for keyword in call.keywords if keyword.arg is not None
    }
    if (
        any(keyword.arg is None for keyword in call.keywords)
        or supplied_keywords - known_keywords
    ) and function.args.kwarg is None:
        return False

    supplied_positionally = {
        argument.arg for argument in positional[: len(call.args)]
    }
    required_positional = {
        argument.arg
        for argument in positional[: len(positional) - len(function.args.defaults)]
    }
    if required_positional - supplied_positionally - supplied_keywords:
        return False
    required_keyword_only = {
        argument.arg
        for argument, default in zip(
            function.args.kwonlyargs, function.args.kw_defaults
        )
        if default is None
    }
    return not (required_keyword_only - supplied_keywords)


def _dsv4_campaign_worker_check(repo: Path) -> Check:
    """Require a wired, concrete DSv4 orchestration seam, not a filename."""

    worker = repo / "prismaquant/dsv4_aura_cb_reprice.py"
    if not worker.is_file():
        return Check(
            "DSv4 bounded campaign worker",
            "BLOCK",
            "driver seam prismaquant.dsv4_aura_cb_reprice is not implemented; "
            "refusing to substitute resident entrypoints",
            str(worker),
        )
    worker_source = _repo_text(repo, "prismaquant/dsv4_aura_cb_reprice.py")
    try:
        tree = ast.parse(worker_source)
    except SyntaxError as exc:
        return Check(
            "DSv4 bounded campaign worker",
            "BLOCK",
            f"driver seam is not valid Python: {exc}",
            str(worker),
        )

    main = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "main"
        ),
        None,
    )
    if main is None:
        return Check(
            "DSv4 bounded campaign worker",
            "BLOCK",
            "driver has no top-level main entrypoint",
            str(worker),
        )

    imports: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        module_name = node.module
        if node.level == 1:
            module_name = f"prismaquant.{module_name}"
        elif node.level:
            continue
        for alias in node.names:
            if alias.name == "run_dsv4_anchor_campaign":
                imports[alias.asname or alias.name] = (module_name, alias.name)
    calls = {
        node.func.id: node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    failures: list[str] = []
    local_call = calls.get("run_dsv4_anchor_campaign")
    if local_call is not None:
        function, failure = _concrete_top_level_function(
            worker_source, "run_dsv4_anchor_campaign"
        )
        if function is None:
            failures.append(f"local driver provider: {failure}")
        elif not _function_accepts_call(function, local_call):
            failures.append(
                "local run_dsv4_anchor_campaign does not accept the driver call"
            )
        else:
            return Check(
                "DSv4 bounded campaign worker",
                "PASS",
                "driver invokes concrete local run_dsv4_anchor_campaign; "
                "external receipt remains the behavioral launch gate",
                str(worker),
            )

    wired = [
        (local_name, imports[local_name], calls[local_name])
        for local_name in sorted(imports)
        if local_name in calls
    ]
    if not wired:
        return Check(
            "DSv4 bounded campaign worker",
            "BLOCK",
            "; ".join(failures)
            or "driver main does not define/import and invoke "
            "run_dsv4_anchor_campaign",
            str(worker),
        )

    for _local_name, (module_name, function_name), call in wired:
        if not module_name.startswith("prismaquant."):
            failures.append(f"unsupported provider module {module_name}")
            continue
        provider = repo.joinpath(*module_name.split(".")).with_suffix(".py")
        if not provider.is_file():
            failures.append(f"provider module is missing: {module_name}")
            continue
        provider_source = provider.read_text()
        function, failure = _concrete_top_level_function(
            provider_source, function_name
        )
        if function is None:
            failures.append(f"{module_name}: {failure}")
            continue
        if not _function_accepts_call(function, call):
            failures.append(
                f"{module_name}.{function_name} does not accept the driver call"
            )
            continue
        return Check(
            "DSv4 bounded campaign worker",
            "PASS",
            f"driver invokes concrete {module_name}.{function_name}; external "
            "receipt remains the behavioral launch gate",
            str(worker),
        )

    return Check(
        "DSv4 bounded campaign worker",
        "BLOCK",
        "; ".join(failures) or "no concrete orchestration provider is wired",
        str(worker),
    )


def _implementation_receipt_check(
    repo: Path, target: str, receipt_path: Path | None
) -> Check:
    """Require an explicit, commit-bound review receipt before GPU launch.

    Source-text feature sniffing below is useful diagnostics, but it is not a
    behavioral proof.  This receipt is deliberately absent in the preparation
    commit.  The implementation change must add it only after the named
    interruption/resume and hybrid integration tests pass.
    """

    if receipt_path is None:
        return Check(
            "commit-bound implementation receipt",
            "BLOCK",
            "AURA_CB_LAUNCH_RECEIPT/--implementation-receipt is required",
        )
    path = receipt_path.resolve(strict=False)
    repo_resolved = repo.resolve(strict=False)
    forbidden = (Path("/tmp"), Path("/home/rob/prismaquant"))
    if (
        path == repo_resolved
        or repo_resolved in path.parents
        or any(path == root or root in path.parents for root in forbidden)
    ):
        return Check(
            "commit-bound implementation receipt",
            "BLOCK",
            "receipt must be an external immutable run input outside the repo, "
            "/tmp, and /home/rob/prismaquant",
            str(path),
        )
    if not path.is_file():
        return Check(
            "commit-bound implementation receipt",
            "BLOCK",
            "reviewed execution receipt is absent; source-string diagnostics "
            "alone may not authorize CUDA",
            str(path),
        )
    required = {
        "identity_bound_cache_resume",
        "checkpointed_aura_per_unit",
        "input_identity_validation",
        "calibration_contract_16x512",
    }
    if target == "dsv4":
        required |= {
            "streamed_format_menu",
            "streamed_expert_unit_kl",
            "split_source_format_plan",
            "hybrid_allocator_keyspace",
        }
    else:
        required |= {
            "dense_profile_resolution",
            "cache_capacity_with_overhead",
        }
    try:
        payload = json.loads(path.read_text())
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "prismaquant",
                "tools",
                "tests",
                "docs/ARCHITECTURE.md",
            ],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        capabilities = payload.get("capabilities") or {}
        targets = set(map(str, payload.get("targets") or []))
        tests = payload.get("tests") or []
        missing = sorted(name for name in required if capabilities.get(name) is not True)
        valid = (
            payload.get("schema") == "prismaquant.aura_cb_reprice_launch_receipt.v1"
            and payload.get("git_commit") == head
            and not dirty
            and target in targets
            and not missing
            and isinstance(tests, list)
            and bool(tests)
            and all(isinstance(item, str) and item.strip() for item in tests)
        )
        return Check(
            "commit-bound implementation receipt",
            "PASS" if valid else "BLOCK",
            (
                f"receipt matches HEAD {head[:12]} and target={target}; "
                f"recorded_tests={len(tests)}"
                if valid
                else f"receipt is incomplete/stale: head={head}, dirty={bool(dirty)}, "
                f"receipt_commit={payload.get('git_commit')}, "
                f"targets={sorted(targets)}, missing_capabilities={missing}"
            ),
            str(path),
        )
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        return Check(
            "commit-bound implementation receipt",
            "BLOCK",
            f"receipt cannot be verified: {exc}",
            str(path),
        )


def repository_capability_checks(
    repo: Path, target: str, receipt_path: Path | None = None
) -> list[Check]:
    """Fail closed on the currently missing execution capabilities.

    These checks intentionally name behavioral contracts instead of treating
    a CLI that merely starts as proof.  When the underlying implementation is
    added, update this preflight and its tests in the same change.
    """

    build_cache = _repo_text(repo, "prismaquant/build_production_cache.py")
    cache = _repo_text(repo, "prismaquant/production_weight_cache.py")
    aura = _repo_text(repo, "prismaquant/aura_cost.py")
    expert = _repo_text(repo, "prismaquant/expert_empirical_cost.py")
    production_render = _repo_text(
        repo, "prismaquant/production_render_cost.py"
    )
    pipeline = _repo_text(repo, "prismaquant/run-pipeline.sh")
    vendored = _repo_text(repo, "prismaquant/vendored/__init__.py")
    candidates = _repo_text(repo, "prismaquant/allocator_candidates.py")
    learned = _repo_text(repo, "prismaquant/cb_learned_bundle.py")
    footprint = _repo_text(repo, "prismaquant/nvfp4_cb_footprint.py")
    checks: list[Check] = []

    source_rate_ready = (
        "SOURCE_BPP_EXCEEDED_REASON" in candidates
        and "exact integer bytes with no tolerance" in candidates
        and (repo / "tests/test_allocator_source_bpp_legality.py").is_file()
    )
    checks.append(
        Check(
            "exact source-bpp candidate gate",
            "PASS" if source_rate_ready else "BLOCK",
            (
                "byte-exact source-payload ceiling and K33/K34 tests are present"
                if source_rate_ready
                else "source-rate legality implementation/tests are incomplete"
            ),
            str(repo / "prismaquant/allocator_candidates.py"),
        )
    )

    learned_policy_ready = (
        re.search(r"46:\s*\{\s*\"enabled\": True", learned) is not None
        and re.search(r"47:\s*\{\s*\"enabled\": False", learned) is not None
        and "CBL_RUNG_POLICY.get" in footprint
    )
    checks.append(
        Check(
            "measured CBL rung policy",
            "PASS" if learned_policy_ready else "BLOCK",
            (
                "K46 is learned-enabled and K47 is lattice fallback"
                if learned_policy_ready
                else "measured K46/K47 learned/lattice dispatch is not wired"
            ),
            str(repo / "prismaquant/cb_learned_bundle.py"),
        )
    )

    streaming_menu_blocked = (
        "streaming a full format menu is out " in build_cache
        and "--render-scope assignment" in build_cache
    )
    streamed_status = (
        "BLOCK" if streaming_menu_blocked and target == "dsv4" else "PASS"
    )
    checks.append(
        Check(
            "streamed cached-menu render",
            streamed_status,
            (
                "build_production_cache --streaming rejects format-menu; "
                "this is a DSv4 blocker"
                if streaming_menu_blocked and target == "dsv4"
                else "dense resident format-menu does not require model "
                "streaming (disk capacity remains a separate launch gate)"
                if streaming_menu_blocked
                else "streaming format-menu rejection is absent; implementation still "
                "requires its targeted tests"
            ),
            str(repo / "prismaquant/build_production_cache.py"),
        )
    )

    cb_resume_blocked = "production CB cache resume is disabled" in cache
    checks.append(
        Check(
            "CB cache per-unit resume",
            "BLOCK" if cb_resume_blocked else "PASS",
            (
                "pre-existing CB pair shards are explicitly rejected"
                if cb_resume_blocked
                else "explicit CB-resume rejection is absent; identity-bound resume "
                "tests must still pass"
            ),
            str(repo / "prismaquant/production_weight_cache.py"),
        )
    )

    aura_checkpointed = (
        '"--checkpoint-dir"' in aura
        and '"--unit-filter"' in aura
        and (target != "dsv4" or '"--streaming"' in aura)
    )
    checks.append(
        Check(
            "checkpointed KL-adjoint",
            "PASS" if aura_checkpointed else "BLOCK",
            (
                "AURA exposes the required unit filtering/checkpoint contract"
                if aura_checkpointed
                else (
                    "aura_cost fully loads the model and writes only its final "
                    "pickle; no streaming/unit checkpoint interface exists"
                    if target == "dsv4"
                    else "aura_cost writes only its final pickle; no durable "
                    "per-unit checkpoint interface exists"
                )
            ),
            str(repo / "prismaquant/aura_cost.py"),
        )
    )

    if target == "dsv4":
        aura_supersurrogate_ready = (
            "AURA_SUPERSURROGATE_ALLOCATOR_SEMANTICS = True" in candidates
            and "def cost_entry_is_anchored_aura_supersurrogate" in candidates
            and candidates.count(
                "cost_entry_is_anchored_aura_supersurrogate("
            ) >= 3
            and "anchored_aura_extrapolation" in candidates
        )
        checks.append(
            Check(
                "AURA supersurrogate allocator semantics",
                "PASS" if aura_supersurrogate_ready else "BLOCK",
                (
                    "allocator has an explicit anchored-AURA branch: reads the "
                    "projection directly, keeps it out of the P5a sample, and "
                    "admits a measured zero. NOTE: this is a CURRENCY claim, "
                    "not an activation error model -- AURA is activation-"
                    "weighted but activation-quantization-BLIND (constant "
                    "across K within a CB family, so it moves only the "
                    "family-choice margin); limitation reported, served A/B "
                    "arbitrates"
                    if aura_supersurrogate_ready
                    else "allocator still classifies explicit predicted_dloss "
                    "as weight-only and can mask a measured zero AURA CB cell "
                    "as activation_cost_unmeasured; refusing GPU launch until "
                    "both the pricing branch and the zero-admission branch "
                    "name the anchored AURA supersurrogate explicitly"
                ),
                str(repo / "prismaquant/allocator_candidates.py"),
            )
        )
        checks.append(_dsv4_campaign_worker_check(repo))
        split_menu = (
            '"--format-plan"' in build_cache
            and '"--format-plan"' in aura
            and '"--format-plan"' in expert
            and '"--format-plan"' in production_render
            and "format_plan_identity_sha256" in production_render
            and "planned_scope" in production_render
        )
        checks.append(
            Check(
                "source-class split format plan",
                "PASS" if split_menu else "BLOCK",
                (
                    "identity-bound per-unit format plan is wired through "
                    "cache, AURA, expert KL, and production-render pricing"
                    if split_menu
                    else "one global FORMATS menu cannot express experts "
                    "K28/K32 and nonexperts K28..K48 step 4 across cache, "
                    "AURA, "
                    "expert KL, and production-render pricing without "
                    "illegal work or truncation"
                ),
            )
        )
        empirical_checkpointed = (
            '"--checkpoint-dir"' in expert and '"--streaming"' in expert
        )
        checks.append(
            Check(
                "checkpointed streamed expert unit-KL",
                "PASS" if empirical_checkpointed else "BLOCK",
                (
                    "expert empirical path is streamed and checkpointed"
                    if empirical_checkpointed
                    else "expert_empirical_cost fully loads BF16 and writes only its "
                    "final merged pickle"
                ),
                str(repo / "prismaquant/expert_empirical_cost.py"),
            )
        )
        unfolded = "enable_per_expert_experts" in vendored
        aura_omits_only_packed = "OMITTED_EXPERTS" in pipeline
        dsv4_hybrid_ready = (
            not unfolded
            or (
                "unpacked_expert" in aura
                and "unpacked_expert" in expert
                and "--col-weights" in pipeline[pipeline.find("OMITTED_EXPERTS") :]
            )
        )
        checks.append(
            Check(
                "DSv4 hybrid key-space",
                "PASS" if dsv4_hybrid_ready and aura_omits_only_packed else "BLOCK",
                (
                    "unfolded DSv4 expert rows reach empirical serving-unit KL"
                    if dsv4_hybrid_ready and aura_omits_only_packed
                    else "vendored DSv4 experts are per-expert nn.Linear rows, while "
                    "the AURA hybrid trigger/empirical helper recognize packed 3-D "
                    "units; current execution would skip the required hybrid"
                ),
            )
        )
    checks.append(_implementation_receipt_check(repo, target, receipt_path))
    return checks


def dsv4_checks(
    repo: Path,
    run_root: Path,
    work_dir: Path | None,
    dataset: Path,
    gpu_lock: Path,
    bundle: Path | None,
    routed_selection: Path | None,
    implementation_receipt: Path | None,
    *,
    verify_hashes: bool,
) -> list[Check]:
    source = run_root / "source"
    cal = run_root / "prod-cal-0p7"
    artifacts = cal / "artifacts"
    checks = _source_checkpoint_checks(source)
    checks.extend(
        [
            _file_check(
                "source index identity",
                source / "model.safetensors.index.json",
                size=5_602_871,
                sha256=(
                    "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
                ),
                verify_hashes=verify_hashes,
            ),
            _file_check(
                "source config identity",
                source / "config.json",
                size=1_888,
                sha256=(
                    "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
                ),
                verify_hashes=verify_hashes,
            ),
            _file_check(
                "calibration corpus",
                dataset,
                size=4_597_898,
                sha256=(
                    "e09a138a4903c4af66a3bf2f9367185f3432224391f1dfe8c94ccc29d99315ba"
                ),
                verify_hashes=verify_hashes,
            ),
            _probe_check(artifacts / "probe.pkl"),
            _activation_cache_check(cal / "act"),
            _file_check(
                "CB column weights",
                artifacts / "cb_col_weights.pkl",
                size=466_388_371,
                sha256=(
                    "df045bde786f7d092e501bfa856984243106a13f05594f4a11fe30270fb09379"
                ),
                verify_hashes=verify_hashes,
            ),
            _col_weight_provenance_check(
                artifacts / "cb_col_weights.pkl.provenance.json"
            ),
            _file_check(
                "incremental CE precompute",
                cal / "work/work/precomputed.pt",
                size=12_082_888_305,
            ),
            _file_check(
                "legacy local cost",
                artifacts / "cost.pkl",
                size=63_497_938,
                sha256=(
                    "08db119fe4e57da4c457106bea498ad1c4a1a4b5d6370777455beb9e29f79a9a"
                ),
                verify_hashes=verify_hashes,
            ),
            _file_check("GPU mutex", gpu_lock, size=0),
        ]
    )
    if work_dir is None:
        checks.append(Check(
            "DSv4 work directory", "BLOCK", "WORK_DIR is required",
        ))
    else:
        resolved = work_dir.resolve(strict=False)
        campaign_artifacts = (resolved / "artifacts").resolve(strict=False)
        baseline_artifacts = (
            run_root / "prod-cal-0p7" / "artifacts"
        ).resolve(strict=False)
        forbidden = (
            Path("/tmp"),
            Path("/home/rob/prismaquant"),
            repo.resolve(strict=False),
        )
        unsafe_root = any(
            resolved == root or root in resolved.parents for root in forbidden
        )
        baseline_overlap = (
            campaign_artifacts == baseline_artifacts
            or baseline_artifacts in campaign_artifacts.parents
        )
        bad = unsafe_root or baseline_overlap
        checks.append(Check(
            "DSv4 work directory",
            "BLOCK" if bad else "PASS",
            (
                f"resolved path={resolved}; campaign artifacts="
                f"{campaign_artifacts}"
                + (
                    "; would overwrite/nest inside the measured Track A "
                    "baseline"
                    if baseline_overlap else " is forbidden"
                    if unsafe_root else " is isolated from the baseline"
                )
            ),
            str(work_dir),
        ))
    if bundle is None:
        checks.append(
            Check(
                "DSv4 learned-codebook bundle",
                "BLOCK",
                "CB_CODEBOOK_BUNDLE is required; the research bucket books are "
                "not an immutable production .pqcb bundle",
            )
        )
    else:
        checks.append(
            _learned_bundle_check(
                repo, "DSv4 learned-codebook bundle", bundle
            )
        )
    if routed_selection is None:
        checks.append(
            Check(
                "DSv4 routed-book selection",
                "BLOCK",
                "CB_ROUTED_MOE_BOOK_SELECTION is required for fail-closed "
                "routed learned-book loading",
            )
        )
    else:
        checks.append(_routed_selection_check(repo, routed_selection))
    if bundle is not None and routed_selection is not None:
        checks.append(_bundle_selection_binding_check(
            repo, bundle, routed_selection,
        ))
    checks.extend(
        repository_capability_checks(repo, "dsv4", implementation_receipt)
    )
    checks.extend(
        [
            Check(
                "KL-adjoint output",
                "BUILD",
                "no DSv4 KL-adjoint/AURA result exists; the CE probe cannot "
                "substitute for it",
                str(run_root / "aura-cb-reprice/artifacts/cost_aura.pkl"),
            ),
            Check(
                "strict production cached-menu",
                "BUILD",
                "no ProductionWeightCache exists under the run root",
                str(run_root / "aura-cb-reprice/cache/production_weight_cache.pkl"),
            ),
            Check(
                "empirical expert unit-KL",
                "BUILD",
                "no routed-expert empirical unit-KL result exists",
                str(run_root / "aura-cb-reprice/artifacts/expert_unit_kl.pkl"),
            ),
        ]
    )
    return checks


def _config_is_dense(config: dict[str, object]) -> tuple[bool, str]:
    nested = config.get("text_config")
    configs = [config]
    if isinstance(nested, dict):
        configs.append(nested)
    archs: list[str] = []
    expert_values: list[tuple[str, object]] = []
    for cfg in configs:
        archs.extend(map(str, cfg.get("architectures") or []))
        for name in (
            "n_routed_experts",
            "num_experts",
            "num_local_experts",
            "num_experts_per_tok",
        ):
            value = cfg.get(name)
            if value not in (None, 0, "", False):
                expert_values.append((name, value))
    if any("moe" in arch.lower() for arch in archs) or expert_values:
        return False, f"architectures={archs}, expert_fields={expert_values}"
    return True, f"architectures={archs}, no routed-expert fields"


def _dense_body_header_census(model: Path) -> tuple[int, dict[str, int]]:
    """Count rank-2 decoder weights and their safetensors dtypes, header-only."""

    from safetensors import safe_open

    index_path = model / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text())
        weight_map = payload.get("weight_map") or {}
    else:
        weight_map = {}
        for shard in sorted(model.glob("*.safetensors")):
            with safe_open(str(shard), framework="np") as handle:
                for key in handle.keys():
                    weight_map[key] = shard.name
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        lowered = str(key).lower()
        if not str(key).endswith(".weight") or ".layers." not in str(key):
            continue
        if any(
            token in lowered
            for token in (
                ".visual.",
                ".vision_",
                ".audio_",
                "layernorm",
                ".norm.",
            )
        ):
            continue
        by_shard.setdefault(str(shard), []).append(str(key))
    nparams = 0
    dtypes: Counter[str] = Counter()
    for shard, keys in by_shard.items():
        path = model / shard
        with safe_open(str(path), framework="np") as handle:
            for key in keys:
                shape = tuple(int(dim) for dim in handle.get_slice(key).get_shape())
                if len(shape) != 2 or min(shape) < 16:
                    continue
                dtypes[str(handle.get_slice(key).get_dtype()).upper()] += 1
                count = 1
                for dim in shape:
                    count *= dim
                nparams += count
    return nparams, dict(sorted(dtypes.items()))


def _available_bytes(path: Path) -> int:
    probe = path.resolve(strict=False)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    stats = os.statvfs(probe)
    return int(stats.f_bavail) * int(stats.f_frsize)


def dense_checks(
    repo: Path,
    model: Path | None,
    work_dir: Path | None,
    dataset: Path,
    gpu_lock: Path,
    bundle: Path | None,
    col_weights: Path | None,
    implementation_receipt: Path | None,
) -> list[Check]:
    checks: list[Check] = []
    if model is None:
        checks.append(
            Check(
                "dense model",
                "BLOCK",
                "MODEL_PATH is required; no local Qwen3.8-27B checkpoint was found",
            )
        )
    else:
        checks.extend(_source_checkpoint_checks(model))
        config_path = model / "config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text())
                dense, detail = _config_is_dense(config)
                checks.append(
                    Check(
                        "dense architecture census",
                        "PASS" if dense else "BLOCK",
                        detail,
                        str(config_path),
                    )
                )
                model_type = str(config.get("model_type") or "")
                archs = list(map(str, config.get("architectures") or []))
                known = (
                    model_type in {"qwen3", "qwen3_5", "qwen3_6"}
                    or any(
                        arch.startswith(("Qwen3For", "Qwen3_5For", "Qwen3_6For"))
                        for arch in archs
                    )
                )
                checks.append(
                    Check(
                        "registered Qwen producer profile",
                        "PASS" if known else "BLOCK",
                        (
                            f"model_type={model_type!r}, architectures={archs}; "
                            + (
                                "matches an existing Qwen profile"
                                if known
                                else "Qwen3.8 metadata is not registered on this branch; "
                                "DefaultProfile is not acceptable without a compatibility proof"
                            )
                        ),
                        str(config_path),
                    )
                )
                qcfg = config.get("quantization_config")
                if not isinstance(qcfg, dict) and isinstance(
                    config.get("text_config"), dict
                ):
                    qcfg = config["text_config"].get("quantization_config")
                quant_text = json.dumps(qcfg or {}).lower()
                source_fp8 = "fp8" in quant_text or "e4m3" in quant_text
                checks.append(
                    Check(
                        "dense FP8 source declaration",
                        "PASS" if source_fp8 else "BLOCK",
                        (
                            f"quantization_config={qcfg}"
                            if source_fp8
                            else "config does not declare native FP8; the 8-bpp "
                            "whole-ladder premise needs a header/source census"
                        ),
                        str(config_path),
                    )
                )
            except (OSError, ValueError, TypeError) as exc:
                checks.append(
                    Check(
                        "dense config",
                        "BLOCK",
                        f"unreadable config: {exc}",
                        str(config_path),
                    )
                )
    if work_dir is None:
        checks.append(Check("dense work directory", "BLOCK", "WORK_DIR is required"))
    else:
        resolved = work_dir.resolve(strict=False)
        forbidden = (Path("/tmp"), Path("/home/rob/prismaquant"))
        bad = any(resolved == root or root in resolved.parents for root in forbidden)
        checks.append(
            Check(
                "dense work directory",
                "BLOCK" if bad else "PASS",
                f"resolved path={resolved}" + (" is forbidden" if bad else ""),
                str(work_dir),
            )
        )
    if model is not None and model.is_dir() and work_dir is not None:
        try:
            nparams, dtype_counts = _dense_body_header_census(model)
            fp8_headers = (
                bool(dtype_counts)
                and all(dtype.startswith("F8") for dtype in dtype_counts)
            )
            checks.append(
                Check(
                    "dense FP8 tensor-header census",
                    "PASS" if fp8_headers else "BLOCK",
                    f"rank-2 decoder weight dtypes={dtype_counts}; every body "
                    "Linear must be native FP8",
                    str(model),
                )
            )
            # ProductionWeightCache stores one BF16 rendered tensor for each
            # nonzero format. The requested dense FP8-CB ladder has 21 rungs.
            payload_bytes = nparams * 2 * len(DENSE_FP8_RUNGS)
            available = _available_bytes(work_dir)
            # Pair-file serialization, manifests, checkpoints, logs, and a
            # free-space floor are not part of the raw tensor payload.  Keep
            # the launch gate conservative until the eventual implementation
            # can report an exact planned-file census.
            overhead_reserve = max(
                (payload_bytes + 19) // 20, 20 * 1024**3
            )
            required_bytes = payload_bytes + overhead_reserve
            capacity_ok = nparams > 0 and available >= required_bytes
            checks.append(
                Check(
                    "dense cached-menu disk capacity",
                    "PASS" if capacity_ok else "BLOCK",
                    f"rank-2 body params={nparams:,}; current BF16 cache payload "
                    f"for 21 rungs={payload_bytes:,} bytes; overhead/free-space "
                    f"reserve={overhead_reserve:,}; required={required_bytes:,}; "
                    f"filesystem available={available:,} bytes",
                    str(work_dir),
                )
            )
        except Exception as exc:
            checks.append(
                Check(
                    "dense cached-menu disk capacity",
                    "BLOCK",
                    f"could not prove capacity from tensor headers: {exc}",
                    str(work_dir),
                )
            )
    checks.extend(
        [
            _file_check("calibration corpus", dataset),
            _file_check("GPU mutex", gpu_lock, size=0),
        ]
    )
    if bundle is None:
        checks.append(
            Check(
                "dense learned-codebook bundle",
                "BLOCK",
                "CB_CODEBOOK_BUNDLE is required for the on-law CBL rungs "
                "K28/K32/K36/K40/K44",
            )
        )
    else:
        checks.append(
            _learned_bundle_check(repo, "dense learned-codebook bundle", bundle)
        )
    if col_weights is None:
        checks.append(
            Check(
                "dense CB column weights",
                "BLOCK",
                "CB_COL_WEIGHTS is required and must come from the same "
                "calibration contract as the bundle/cache",
            )
        )
    else:
        checks.append(_file_check("dense CB column weights", col_weights))
    checks.extend(
        repository_capability_checks(repo, "dense", implementation_receipt)
    )
    checks.extend(
        [
            Check(
                "dense strict production cached-menu",
                "BUILD",
                "render FP8_CB_K28..K48 with current imatrix/bundle identity",
                str((work_dir or Path("<WORK_DIR>")) / "cache/production_weight_cache.pkl"),
            ),
            Check(
                "dense AURA cost",
                "BUILD",
                "straight KL-adjoint AURA; no expert hybrid on a dense model",
                str((work_dir or Path("<WORK_DIR>")) / "artifacts/cost_aura.pkl"),
            ),
        ]
    )
    return checks


def _print_human(target: str, checks: Iterable[Check]) -> None:
    print(f"AURA-on-CB preflight: target={target}")
    for check in checks:
        suffix = f" [{check.path}]" if check.path else ""
        print(f"{check.status:>5}  {check.name}: {check.detail}{suffix}")
    accounting = campaign_accounting()
    print("\nSource-rate-derived legal menu and bounded render domain:")
    if target == "dense":
        print("  Dense native-FP8 units: FP8_CB_K28..K48 (21 rungs)")
        print("  Unit count and cache bytes must come from the supplied headers")
        return
    print(
        f"  NVFP4 lattice: {accounting['total_units']:,} x K12..K18 = "
        f"{accounting['nvfp4_cells']:,} legal cells"
    )
    print(
        f"  FP8 experts: {accounting['expert_units']:,} x K28/K32 = "
        f"{accounting['expert_fp8_cells']:,} legal cells; orientations: "
        f"{accounting['expert_2048x4096_units']:,} x 2048x4096 + "
        f"{accounting['expert_4096x2048_units']:,} x 4096x2048"
    )
    print(
        f"  FP8 nonexperts: {accounting['nonexpert_units']:,} x K28..K48 step 4 = "
        f"{accounting['nonexpert_fp8_cells']:,} legal cells"
    )
    print(
        f"  Exact source terminals: {accounting['source_terminal_cells']:,}; "
        f"total legal DP cells: {accounting['candidate_cells']:,}"
    )
    for segment in accounting["anchor_segments"]:
        print(
            f"  Anchors {segment['family']}/{segment['role']}/"
            f"{segment['basis']}: "
            f"{segment['renders']:,} production renders"
        )
    print(
        f"  Anchor total: {accounting['anchor_renders']:,} + bounded panel + "
        "validation renders (no full-menu materialization)"
    )
    print(f"  T_encode = {accounting['encode_seconds_formula']} seconds")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("dsv4", "dense"))
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--run-root", type=Path, default=DSV4_RUN_ROOT)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--cb-codebook-bundle", type=Path, default=None)
    parser.add_argument("--cb-col-weights", type=Path, default=None)
    parser.add_argument("--routed-book-selection", type=Path, default=None)
    parser.add_argument("--implementation-receipt", type=Path, default=None)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="inventory/report mode: print blockers but return success",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Defense in depth: this process is inventory-only and must never see a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = _build_parser().parse_args(argv)
    if args.target == "dsv4":
        checks = dsv4_checks(
            args.repo,
            args.run_root,
            args.work_dir,
            args.dataset,
            args.gpu_lock,
            args.cb_codebook_bundle,
            args.routed_book_selection,
            args.implementation_receipt,
            verify_hashes=args.verify_hashes,
        )
    else:
        checks = dense_checks(
            args.repo,
            args.model,
            args.work_dir,
            args.dataset,
            args.gpu_lock,
            args.cb_codebook_bundle,
            args.cb_col_weights,
            args.implementation_receipt,
        )
    if args.json:
        print(
            json.dumps(
                {
                    "target": args.target,
                    "checks": [asdict(check) for check in checks],
                    "accounting": campaign_accounting(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        _print_human(args.target, checks)
    blocked = any(check.status == "BLOCK" for check in checks)
    return 0 if args.allow_blocked or not blocked else 2


if __name__ == "__main__":
    sys.exit(main())
