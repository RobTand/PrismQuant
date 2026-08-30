"""Independent wire-v1 agreement gate; this test never imports Gridbook."""
from __future__ import annotations

import hashlib

import torch

from prismaquant.trellis_formats import E2M1_FAMILY
from prismaquant.trellis_wire import TrellisWire, decode_codes_torch, pack_planes


def _unpack_lsb_bits(value: str) -> torch.Tensor:
    packed = bytes.fromhex(value)
    return torch.tensor([
        (packed[index // 8] >> (index & 7)) & 1
        for index in range(len(packed) * 8)
    ], dtype=torch.uint8)


def test_gridbook_stage5_frozen_golden_vector_matches_bit_exactly():
    """Gridbook froze this real eager-Viterbi output at seed 20260825."""
    u = _unpack_lsb_bits(
        "8070d48f68c8eb1d131b04c20f1f418f"
        "a1280d304d965f9d3bd4e9fd70fb91d6"
    )
    points = _unpack_lsb_bits(
        "e165bed867bba5cf94f2d1f78f1bb4a0"
        "c31f7de7fb9a54f7a7fba64dbd73c665"
    )
    wire = pack_planes(
        family=E2M1_FAMILY,
        body_rate_q256=512,
        schedule=[2] * 256,
        layout="fixed_quota_per_256",
        u_bits=u.reshape(1, 256),
        point_indices=points.reshape(1, 256),
        bypass_codes=torch.zeros(1, 256, dtype=torch.uint8),
        alphabets={2: (15, 13, 11, 9, 8, 2, 4, 7)},
        scale_blob=bytes([0x38] * 16),
    )
    blob = wire.to_bytes()
    assert len(blob) == 307
    assert TrellisWire.from_bytes(blob).to_bytes() == blob
    decoded = decode_codes_torch(blob).numpy().tobytes()
    assert hashlib.sha256(decoded).hexdigest() == (
        "289f28f80580fcd7565401b9b1f0d9c"
        "79451a31b7f46e76d98dbc4b1c5b78e61"
    )
