"""Command-line verifier for immutable artifact-collection record graphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from prismaquant.artifact_collection import ArtifactCollectionError, load_record
from prismaquant.artifact_collection_records import verify_collection_graph


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument(
        "records",
        nargs="+",
        metavar="RECORD.json",
        help="complete closed set of semantic records to reconcile",
    )
    args = parser.parse_args(argv)
    try:
        records = [load_record(Path(path)) for path in args.records]
        result = verify_collection_graph(records)
    except ArtifactCollectionError as exc:
        parser.exit(2, f"artifact-collection: REFUSED: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
