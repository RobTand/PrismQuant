#!/usr/bin/env python3
"""Emit an exact no-codebook byte-budget infeasibility certificate."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Keep direct ``python tools/...py`` invocation consistent with the repo's
# other operator tools without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.native_baseline_feasibility import (
    certify_native_baseline_from_model,
    write_certificate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--model-identity-cache", required=True)
    parser.add_argument(
        "--upgrade-partial-model-identity-cache",
        action="store_true",
        help=(
            "atomically extend an old decoder-only cache to the complete "
            "checkpoint, reusing unchanged shard hashes"
        ),
    )
    parser.add_argument("--budget-bytes", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-profile", default="nvfp4_cb")
    parser.add_argument("--lane", default="nvfp4_cb")
    args = parser.parse_args()
    certificate = certify_native_baseline_from_model(
        model_path=args.model,
        probe_path=args.probe,
        model_identity_cache_path=args.model_identity_cache,
        budget_bytes=args.budget_bytes,
        target_profile=args.target_profile,
        lane_id=args.lane,
        upgrade_partial_identity_cache=args.upgrade_partial_model_identity_cache,
    )
    write_certificate(certificate, args.output)
    accounting = certificate["accounting"]
    print(
        f"infeasible: lower_bound={accounting['all_native_lower_bound_bytes']} "
        f"budget={accounting['budget_bytes']} excess={accounting['excess_bytes']} "
        f"sha256={certificate['certificate_sha256']}"
    )


if __name__ == "__main__":
    main()
