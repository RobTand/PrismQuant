from __future__ import annotations

import hashlib

import pytest
import torch

from prismaquant.trellis_formats import E2M1_FAMILY
from prismaquant.trellis_encoder import TrellisEncoderError
from prismaquant.trellis_producer import encode_trellis_one_linear
from prismaquant.trellis_wire import TrellisWire, TrellisWireError, decode_values_torch


_ALPHABET = {2: (15, 13, 11, 9, 8, 2, 4, 7)}


def _encode(global_scale_real_override=None):
    generator = torch.Generator().manual_seed(20260830)
    weight = torch.randn(1, 256, generator=generator)
    importance = torch.linspace(0.5, 1.5, 256)
    return encode_trellis_one_linear(
        weight,
        importance,
        family=E2M1_FAMILY,
        body_rate_q256=512,
        schedule=[2] * 256,
        layout="fixed_quota_per_256",
        alphabets=_ALPHABET,
        scale_rule="static_6",
        sb_chunk=1,
        determinism_mode="on",
        tailbite_candidates=4,
        backend="eager",
        point_route="full",
        global_scale_real_override=global_scale_real_override,
    )


def test_shared_producer_binds_and_decodes_the_same_immutable_bytes():
    artifact = _encode()
    blob = artifact.wire_bytes
    parsed = TrellisWire.from_bytes(blob)
    assert parsed.to_bytes() == blob
    assert torch.equal(
        artifact.decoded_weight,
        decode_values_torch(blob, dtype=artifact.decoded_weight.dtype),
    )
    assert artifact.receipt["wire_identity_sha256"] == hashlib.sha256(
        blob
    ).hexdigest()
    assert artifact.receipt["same_byte_reparse_verified"] is True
    assert artifact.receipt["encoder_reconstruction_bf16_equal"] is True
    assert artifact.receipt["producer_eligible"] is False


def test_wire_parser_refuses_truncation_and_body_mutation_is_observable():
    artifact = _encode()
    with pytest.raises(TrellisWireError):
        TrellisWire.from_bytes(artifact.wire_bytes[:-1])

    changed = bytearray(artifact.wire_bytes)
    changed[-16] ^= 1
    parsed = TrellisWire.from_bytes(changed)
    assert parsed.to_bytes() == bytes(changed)
    assert not torch.equal(
        artifact.decoded_weight,
        decode_values_torch(changed, dtype=artifact.decoded_weight.dtype),
    )


def test_explicit_shared_global_reproduces_the_default_one_block_wire():
    default = _encode()
    fixed = _encode(TrellisWire.from_bytes(default.wire_bytes).global_scale_real)
    assert fixed.wire_bytes == default.wire_bytes


def test_explicit_shared_global_must_be_positive_and_finite():
    with pytest.raises(TrellisEncoderError, match="finite and positive"):
        _encode(-1.0)
