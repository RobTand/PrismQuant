#!/usr/bin/env python3
"""Cross-repository verifier: decode the prismaquant golden vectors with gridbook.trellis.

Run this from a Gridbook checkout (the one pinned at 0.9.1, commit 227420f):

  python /path/to/prismaquant/tools/verify_trellis_golden_gridbook.py \
      --fixtures /path/to/prismaquant/tests/fixtures/trellis_wire_golden

Or, if you are inside the Gridbook repo and the fixtures are at ../prismaquant:

  python tools/verify_trellis_golden_gridbook.py  # auto-finds fixtures relative to script

Each fixture was produced by prismaquant.trellis_wire.pack_planes and its
expected decoded weight is stored as a .pt. This script reads the same wire
bytes with gridbook.trellis.TrellisWire.from_bytes and decodes via
gridbook.trellis.decode_values, then asserts the two floating tensors are
bit-exact (torch.equal). A mismatch is a wire-contract drift between the two
repositories.

Do NOT import prismaquant from this script; the test suite proves
prismaquant's side, this script proves gridbook's side, and the shared bytes
are the only contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

# gridbook is only available inside a Gridbook checkout / wheel install
try:
    from gridbook import trellis as gridbook_trellis
except ImportError as exc:  # pragma: no cover
    print(f"cannot import gridbook.trellis: {exc}", file=sys.stderr)
    print("Install the pinned wheel dist-ci/gridbook-0.9.1-py3-none-any.whl or run from a Gridbook checkout", file=sys.stderr)
    sys.exit(2)


def _find_fixtures(arg: Path | None) -> Path:
    if arg and arg.exists():
        return arg
    # auto-find relative to this script: tools/verify_... -> ../../tests/fixtures/...
    # When run from Gridbook checkout, fixtures are elsewhere; require --fixtures
    candidates = [
        Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "trellis_wire_golden",
        Path(__file__).resolve().parent / "tests" / "fixtures" / "trellis_wire_golden",
        Path.cwd() / "tests" / "fixtures" / "trellis_wire_golden",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("cannot locate trellis_wire_golden fixtures; pass --fixtures")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, default=None, help="path to tests/fixtures/trellis_wire_golden")
    args = ap.parse_args()
    fixture_dir = _find_fixtures(args.fixtures)
    bins = sorted(fixture_dir.glob("*.bin"))
    if not bins:
        print(f"no .bin fixtures in {fixture_dir}", file=sys.stderr)
        return 2
    errors = 0
    for blob_path in bins:
        blob = blob_path.read_bytes()
        meta = json.loads(blob_path.with_suffix(".json").read_text())
        # verify blob sha
        sha = hashlib.sha256(blob).hexdigest()
        if sha != meta["wire_sha256"]:
            print(f"{blob_path.name}: wire sha mismatch {sha} != {meta['wire_sha256']}")
            errors += 1
            continue
        # decode with gridbook
        try:
            wire = gridbook_trellis.TrellisWire.from_bytes(blob)
            wire.validate()
        except Exception as exc:
            print(f"{blob_path.name}: gridbook parse failed: {exc}")
            errors += 1
            continue
        # gridbook decode_values returns list[list[float]]; convert to tensor
        # Use gridbook's decode_values then torch tensor for comparison, to avoid
        # relying on gridbook's torch path if not available
        try:
            values = gridbook_trellis.decode_values(wire)
            decoded_gb = torch.tensor(values, dtype=torch.float32)
        except Exception as exc:
            # fallback to torch helper if present
            try:
                decoded_gb = gridbook_trellis.decode_values_torch(wire, device="cpu")
            except Exception as exc2:
                print(f"{blob_path.name}: gridbook decode failed: {exc} / {exc2}")
                errors += 1
                continue
        # load expected (prismaquant's decode)
        pt_path = blob_path.with_suffix(".pt")
        expected = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        if decoded_gb.shape != expected.shape:
            print(f"{blob_path.name}: shape mismatch gridbook {decoded_gb.shape} vs prismaquant {expected.shape}")
            errors += 1
            continue
        if not torch.equal(decoded_gb, expected):
            # find first diff
            diff = (decoded_gb != expected)
            idx = diff.nonzero(as_tuple=False)[0].tolist() if diff.any() else []
            print(f"{blob_path.name}: decoded NOT bit-exact (first diff at {idx})")
            # show values
            if idx:
                r, c = idx[0], idx[1] if len(idx)>1 else 0
                print(f"  gridbook={decoded_gb[r,c].item()} prismaquant={expected[r,c].item()}")
            errors += 1
            continue
        print(f"{blob_path.name}: OK {decoded_gb.shape} sha {meta['decoded_sha256'][:16]}")
    if errors:
        print(f"FAILED: {errors}/{len(bins)} fixtures mismatched", file=sys.stderr)
        return 1
    print(f"PASSED: {len(bins)} fixtures bit-exact between prismaquant and gridbook")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
