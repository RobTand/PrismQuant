#!/usr/bin/env python3
"""Verify and print the fixed DSv4-Flash W8A16 pre-export receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from prismaquant.dsv4_w8a16_export_handoff import (
    W8A16ExportHandoffError,
    verify_dsv4_w8a16_export_handoff,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "fail closed unless the exact DSv4 W8A16 readmission and all "
            "immutable export inputs are ready"
        )
    )
    parser.add_argument("--publication", required=True)
    parser.add_argument("--approved-raw-publication", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source-identity", required=True)
    parser.add_argument("--codebook-bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve(strict=True).parent.parent),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = verify_dsv4_w8a16_export_handoff(
            publication_dir=args.publication,
            approved_raw_publication_dir=args.approved_raw_publication,
            source_model_dir=args.model_dir,
            source_identity_path=args.source_identity,
            codebook_bundle_path=args.codebook_bundle,
            output_path=args.out,
            repo_root=args.repo_root,
        )
    except W8A16ExportHandoffError as exc:
        raise SystemExit(f"REFUSE: DSv4 W8A16 export handoff failed: {exc}")
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
