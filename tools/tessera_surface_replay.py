#!/usr/bin/env python3
"""Replay trusted local Tessera campaign measurements; emit research report only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prismaquant.cost_stage_checkpoint import atomic_write_bytes
from prismaquant.tessera_anchored_surface import replay_campaign


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costs", required=True, help="trusted campaign cost pickle")
    parser.add_argument("--checkpoint", required=True, help="existing campaign checkpoint manifest")
    parser.add_argument("--plan", required=True, help="explicit JSON replay plan")
    parser.add_argument("--out", required=True, help="research JSON report")
    args = parser.parse_args(argv)
    try:
        plan = json.loads(Path(args.plan).read_text())
        report = replay_campaign(args.costs, args.checkpoint, plan)
        encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        output = Path(args.out)
        if output.exists() and output.read_bytes() != encoded:
            raise ValueError("output already contains a different replay; use another --out")
        if not output.exists():
            atomic_write_bytes(output, encoded)
    except (ValueError, RuntimeError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"replay refused: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
