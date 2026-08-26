#!/usr/bin/env python3
"""Deterministic command line interface for PrismaSnap source preparation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.prismasnap import PrismaSnapSearchConfig  # noqa: E402
from prismaquant.prismasnap_checkpoint import (  # noqa: E402
    bind_legacy_text_probe,
    load_plan,
    materialize_checkpoint,
    materialize_checkpoint_part,
    merge_checkpoint_parts,
    merge_plans,
    plan_dense_checkpoint,
    realize_bf16_plan,
    scan_tensor_metadata_manifest,
)
from prismaquant.prismasnap_validation import attest_fold_fidelity


def _layers(raw: str) -> list[int] | None:
    value = raw.strip()
    if not value or value == "all":
        return None
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise argparse.ArgumentTypeError("empty layer selector")
        if "-" in part:
            first, last = part.split("-", 1)
            try:
                start, stop = int(first), int(last)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid layer range {part!r}"
                ) from exc
            if start < 0 or stop < start:
                raise argparse.ArgumentTypeError(f"invalid layer range {part!r}")
            result.update(range(start, stop + 1))
        else:
            try:
                index = int(part)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid layer index {part!r}"
                ) from exc
            if index < 0:
                raise argparse.ArgumentTypeError("layer indices must be non-negative")
            result.add(index)
    return sorted(result)


def _search_config(args: argparse.Namespace) -> PrismaSnapSearchConfig:
    return PrismaSnapSearchConfig(
        alphas=tuple(args.alphas),
        max_rounds=args.max_rounds,
        stage=True,
        polish=True,
        polish_top=args.polish_top,
        polish_pool=args.polish_pool,
        scale_rule=args.nvfp4_scale_rule,
        snapped_scale_scoring=False,
    )


def _shards_file(raw: str) -> list[str]:
    path = Path(raw).resolve(strict=True)
    shards: list[str] = []
    for number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        name = raw_line.strip()
        if not name:
            continue
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise argparse.ArgumentTypeError(
                f"{path}:{number}: expected a safetensors shard basename"
            )
        if name in shards:
            raise argparse.ArgumentTypeError(
                f"{path}:{number}: duplicate shard basename {name!r}"
            )
        shards.append(name)
    if not shards:
        raise argparse.ArgumentTypeError(f"{path}: shard list is empty")
    return shards


def _print(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PrismaSnap is an additive BF16 source-checkpoint pre-pass; it "
            "does not alter run-pipeline.sh or any serving runtime."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bind_probe = sub.add_parser(
        "bind-legacy-text-probe",
        help="normalize and content-bind a frozen legacy text probe",
    )
    bind_probe.add_argument("--source", required=True)
    bind_probe.add_argument("--source-identity", required=True)
    bind_probe.add_argument("--probe", required=True)
    bind_probe.add_argument("--output", required=True)
    bind_probe.add_argument("--resume", action="store_true")

    scan_headers = sub.add_parser(
        "scan-tensor-metadata",
        help="publish a content-bound full safetensors header census",
    )
    scan_headers.add_argument("--source", required=True)
    scan_headers.add_argument("--source-identity", required=True)
    scan_headers.add_argument("--output", required=True)
    scan_headers.add_argument("--resume", action="store_true")

    plan = sub.add_parser("plan-dense", help="plan dense/hybrid body seams")
    plan.add_argument("--source", required=True)
    plan.add_argument("--probe", required=True)
    plan.add_argument("--source-identity", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--tensor-metadata-manifest", default=None)
    plan.add_argument("--device", default="cuda")
    plan.add_argument("--layers", type=_layers, default=None)
    plan.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.125, 0.25, 0.375, 0.5],
    )
    plan.add_argument("--resume", action="store_true")
    plan.add_argument("--max-rounds", type=int, default=4)
    plan.add_argument("--polish-top", type=int, default=8)
    plan.add_argument("--polish-pool", type=int, default=16)
    plan.add_argument(
        "--nvfp4-scale-rule",
        default="static_6",
        choices=("static_6", "four_over_six_mse", "joint_mse"),
    )
    plan.add_argument(
        "--skip-source-content-verification",
        action="store_true",
        help=(
            "Development-only: trust the supplied identity's shard digests. "
            "Production campaign manifests must not set this."
        ),
    )

    merge = sub.add_parser("merge-plans", help="exact-union worker layer plans")
    merge.add_argument("--input", action="append", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--resume", action="store_true")

    realize = sub.add_parser(
        "realize-bf16",
        help="derive a cast-aware executable v2 plan from a full merged v1 plan",
    )
    realize.add_argument("--source", required=True)
    realize.add_argument("--plan", required=True)
    realize.add_argument("--output", required=True)
    realize.add_argument("--device", default="cuda")
    realize.add_argument("--resume", action="store_true")

    materialize = sub.add_parser(
        "materialize", help="stream and atomically commit a snapped checkpoint"
    )
    materialize.add_argument("--source", required=True)
    materialize.add_argument("--plan", required=True)
    materialize.add_argument("--output", required=True)
    materialize.add_argument("--device", default="cuda")
    materialize.add_argument("--resume", action="store_true")

    materialize_part = sub.add_parser(
        "materialize-part",
        help="materialize a content-bound proper subset of source shards",
    )
    materialize_part.add_argument("--source", required=True)
    materialize_part.add_argument("--plan", required=True)
    materialize_part.add_argument("--output", required=True)
    materialize_part.add_argument("--shards-file", type=_shards_file, required=True)
    materialize_part.add_argument("--device", default="cuda")
    materialize_part.add_argument("--resume", action="store_true")

    merge_parts = sub.add_parser(
        "merge-checkpoint-parts",
        help="verify and exact-union independently materialized shard parts",
    )
    merge_parts.add_argument("--source", required=True)
    merge_parts.add_argument("--plan", required=True)
    merge_parts.add_argument("--part", action="append", required=True)
    merge_parts.add_argument("--output", required=True)
    merge_parts.add_argument("--resume", action="store_true")
    merge_parts.add_argument(
        "--require-hardlinks",
        action="store_true",
        help=(
            "Fail closed unless every verified part shard can be hardlinked "
            "into the output staging tree on the same filesystem."
        ),
    )

    attest = sub.add_parser(
        "attest-fold-fidelity",
        help="transition a materialized checkpoint to VERIFIED from served BF16 KL",
    )
    attest.add_argument("--checkpoint", required=True)
    attest.add_argument("--student-result", required=True)
    attest.add_argument("--teacher-meta", required=True)
    attest.add_argument(
        "--source-identity",
        required=True,
        help="full streamed-model identity for the original BF16 source",
    )

    inspect = sub.add_parser("inspect", help="verify and summarize a plan")
    inspect.add_argument("--plan", required=True)

    args = parser.parse_args(argv)
    if args.command == "bind-legacy-text-probe":
        result = bind_legacy_text_probe(
            args.source,
            args.source_identity,
            args.probe,
            args.output,
            resume=args.resume,
            production=True,
        )
        _print(result)
        return 0
    if args.command == "scan-tensor-metadata":
        result = scan_tensor_metadata_manifest(
            args.source,
            args.source_identity,
            args.output,
            resume=args.resume,
            production=True,
        )
        _print(result)
        return 0
    if args.command == "plan-dense":
        result = plan_dense_checkpoint(
            args.source,
            args.probe,
            args.source_identity,
            args.output,
            device=args.device,
            layers=args.layers,
            search_config=_search_config(args),
            tensor_metadata_manifest_path=args.tensor_metadata_manifest,
            verify_source_content=not args.skip_source_content_verification,
            resume=args.resume,
            production=True,
        )
        _print(
            {
                "plan": str(Path(args.output).resolve()),
                "plan_sha256": result["plan_sha256"],
                "layers": result["model"]["planned_layers"],
                "seams": len(result["seams"]),
            }
        )
        return 0
    if args.command == "merge-plans":
        result = merge_plans(args.input, args.output, resume=args.resume)
        _print(
            {
                "plan": str(Path(args.output).resolve()),
                "plan_sha256": result["plan_sha256"],
                "layers": result["model"]["planned_layers"],
                "workers": result["workers"],
            }
        )
        return 0
    if args.command == "realize-bf16":
        result = realize_bf16_plan(
            args.source,
            args.plan,
            args.output,
            device=args.device,
            resume=args.resume,
            production=True,
        )
        _print(
            {
                "plan": str(Path(args.output).resolve()),
                "plan_sha256": result["plan_sha256"],
                "parent_plan_sha256": result["derivation"]["parent"][
                    "plan_sha256"
                ],
                "layers": result["model"]["planned_layers"],
                "seams": len(result["seams"]),
            }
        )
        return 0
    if args.command == "materialize":
        result = materialize_checkpoint(
            args.source,
            args.plan,
            args.output,
            device=args.device,
            resume=args.resume,
            production=True,
        )
        _print(result)
        return 0
    if args.command == "materialize-part":
        result = materialize_checkpoint_part(
            args.source,
            args.plan,
            args.output,
            args.shards_file,
            device=args.device,
            resume=args.resume,
            production=True,
        )
        _print(result)
        return 0
    if args.command == "merge-checkpoint-parts":
        result = merge_checkpoint_parts(
            args.source,
            args.plan,
            args.part,
            args.output,
            resume=args.resume,
            require_hardlinks=args.require_hardlinks,
        )
        _print(result)
        return 0
    if args.command == "attest-fold-fidelity":
        result = attest_fold_fidelity(
            args.checkpoint,
            args.student_result,
            args.teacher_meta,
            args.source_identity,
        )
        _print(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "state": result["state"],
                "provenance_sha256": result["provenance_sha256"],
                "fold_fidelity": result["fold_fidelity"],
            }
        )
        return 0
    if args.command == "inspect":
        result, scales = load_plan(args.plan)
        _print(
            {
                "schema": result["schema"],
                "plan_sha256": result["plan_sha256"],
                "profile": result["profile"],
                "layers": result["model"]["planned_layers"],
                "seams": len(result["seams"]),
                "transforms": len(result["transforms"]),
                "scale_vectors": sorted(scales),
                "verification": result["verification"],
            }
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
