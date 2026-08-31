"""Golden-vector cross-repo proof: prismaquant.trellis_wire vs stored decode.

The fixtures under tests/fixtures/trellis_wire_golden/ were produced by
scratch/gen_golden.py using prismaquant.trellis_wire. This test proves the
decoder is deterministic and bit-exact against the stored expected weight.

A second script tools/verify_trellis_golden_gridbook.py lets a Gridbook
checkout run the same fixtures through gridbook.trellis and compare — the
AGENTS.md:38 forbids importing gridbook here, so agreement is proved via
shared bytes, not a shared import.
"""
from pathlib import Path
import hashlib
import json

import torch
from prismaquant.trellis_wire import TrellisWire, decode_values_torch

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "trellis_wire_golden"


def _wire_paths():
    return sorted(FIXTURE_DIR.glob("*.bin"))


def test_golden_vector_count_and_coverage():
    bins = _wire_paths()
    assert len(bins) >= 8, f"need N>=8 golden wires, got {len(bins)}"
    families = set()
    layouts = set()
    rates = set()
    has_non_multiple_256 = False
    has_non_multiple_16_rows = False
    for p in bins:
        meta = json.loads((p.with_suffix(".json")).read_text())
        families.add(meta["family"])
        layouts.add(meta["layout"])
        rates.add(meta["body_rate_q256"])
        if meta["columns"] % 256 != 0:
            has_non_multiple_256 = True
        if meta["rows"] % 16 != 0:
            has_non_multiple_16_rows = True
    assert len(families) >= 2, f"must span both families, got {families}"
    assert len(layouts) >= 2, f"must span both layouts, got {layouts}"
    assert len({r for r in rates if r in (384,512,640,768,896)}) >= 3, f"need >=3 of five E2M1 candidate rungs, got {rates}"
    assert has_non_multiple_256, "need a non-multiple-of-256 column count"
    assert has_non_multiple_16_rows, "need a non-multiple-of-16 row count"


def test_golden_wires_decode_bit_exact():
    bins = _wire_paths()
    assert bins, f"no fixtures in {FIXTURE_DIR}"
    for blob_path in bins:
        blob = blob_path.read_bytes()
        # sha256 of blob must match meta
        meta_path = blob_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        sha = hashlib.sha256(blob).hexdigest()
        assert sha == meta["wire_sha256"], f"{blob_path.name}: wire sha mismatch"
        # decode via prismaquant
        wire = TrellisWire.from_bytes(blob)
        wire.validate()
        decoded = decode_values_torch(wire, device="cpu", dtype=torch.float32)
        # load expected
        pt_path = blob_path.with_suffix(".pt")
        expected = torch.load(str(pt_path), map_location="cpu", weights_only=True)
        # bit-exact: torch.equal on float32 bit pattern
        assert decoded.shape == expected.shape, f"{blob_path.name}: shape {decoded.shape} != {expected.shape}"
        assert torch.equal(decoded, expected), f"{blob_path.name}: decoded weight not bit-exact"
        # also verify wire_bytes roundtrip
        assert wire.to_bytes() == blob, f"{blob_path.name}: to_bytes roundtrip mismatch"
        # sha of decoded
        decoded_sha = hashlib.sha256(decoded.numpy().tobytes()).hexdigest()
        assert decoded_sha == meta["decoded_sha256"], f"{blob_path.name}: decoded sha mismatch"
