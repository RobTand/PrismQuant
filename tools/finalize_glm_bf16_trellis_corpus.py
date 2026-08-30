#!/usr/bin/env python3
"""Create the immutable, importance-complete GLM BF16 trellis corpus.

The source artifact and its ``v0-INCOMPLETE`` manifest remain untouched.  The
new artifact is written with a pread-only byte copy and will not overwrite any
existing path.  ``--dry-run`` validates and prints the exact population and
importance plan without writing output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
from pathlib import Path

from prismaquant.trellis_bf16_corpus import (
    adapt_glm_importance_from_probe,
    finalize_glm_bf16_corpus,
    validate_incomplete_glm_source,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incomplete-manifest", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--prismaquant-commit", required=True)
    parser.add_argument("--generated", required=True,
                        help="explicit ISO-8601 provenance timestamp")
    parser.add_argument("--host", default=socket.gethostname())
    parser.add_argument("--output-artifact", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    calibration = json.loads(args.calibration_json.read_text())
    if not isinstance(calibration, dict):
        raise SystemExit("calibration JSON must be an object")
    importance, identity = adapt_glm_importance_from_probe(
        args.incomplete_manifest, args.probe
    )
    source_contract = validate_incomplete_glm_source(
        args.incomplete_manifest, args.source_artifact
    )
    populations = {
        name: sum(
            (".experts.0." in target) == (name == "routed")
            for target in importance
        )
        for name in ("dense", "routed")
    }
    plan = {
        "status": "validated_no_write" if args.dry_run else "ready_to_finalize",
        "source_artifact": str(args.source_artifact.resolve()),
        "source_artifact_sha256": source_contract["file_sha256"],
        "source_payload_bytes": source_contract["payload_bytes"],
        "probe": str(args.probe.resolve()),
        "importance_identity": identity,
        "populations": populations,
        "output_artifact": str(args.output_artifact.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    completed = finalize_glm_bf16_corpus(
        incomplete_manifest_path=args.incomplete_manifest,
        source_artifact_path=args.source_artifact,
        importance=importance,
        importance_identity=identity,
        output_artifact_path=args.output_artifact,
        output_manifest_path=args.output_manifest,
        calibration=calibration,
        model_config_sha256=_sha256(args.model_config),
        prismaquant_commit=args.prismaquant_commit,
        generated=args.generated,
        host=args.host,
    )
    print(json.dumps({**plan, "status": "finalized", "manifest": str(completed)},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
