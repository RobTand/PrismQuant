"""The encoder must compose its reconstruction the way the WIRE decoder does.

WHY THIS FILE EXISTS.  ``_scale_context`` folds the E2M1 global into the
per-group scale, so the natural spelling of the reconstruction is
``code * (e4m3 * global)``.  ``trellis_wire.decode_values_torch`` -- the bytes
the served runtime executes -- evaluates ``(code * e4m3) * global``.  Floating
multiplication is not associative, the two differ by at most one fp32 ULP, and
that ULP lands on a DIFFERENT bf16 value whenever the product sits near a tie.

``encode_trellis_one_linear`` then raises ``TrellisProducerError`` on its own
same-byte invariant, so the failure is a hard abort in the middle of a render,
not a silent quality loss.  It was found on 2026-08-31 by encoding real
Qwen3-4B weights: ``layers.31.self_attn.v_proj`` disagreed on 149,261 of
2,621,440 values (5.69%), every one exactly one bf16 ULP apart, identically
under the eager and the Triton backend -- roughly one tensor in thirty, and the
1x256 random fixture in ``test_trellis_producer.py`` never reaches it because
its scale regime does not produce ties.

The fix is not a tolerance.  Relaxing the invariant to "within one ULP" would
abandon the byte-identity evidence boundary the producer exists to hold
(principle 8: the surrogate, the KL validation and the exported bytes are ONE
rendering).  The decoder is the shipped contract, so the encoder adopts its
association.
"""
from __future__ import annotations

import pytest
import torch

from prismaquant.trellis_encoder import (
    E2M1_SCALE_FLOOR,
    encode_trellis_planes,
)
from prismaquant.trellis_formats import E2M1_FAMILY, native_code_value
from prismaquant.trellis_producer import encode_trellis_one_linear
from prismaquant.trellis_wire import (
    TrellisWire,
    decode_codes_torch,
    decode_values_torch,
    pack_planes,
)

_ALPHABET = {2: (15, 13, 11, 9, 8, 2, 4, 7)}
#: Found by search over (rows, seed, scale): the smallest input on which the
#: two associations disagree at the bf16 boundary. 14 of 256 values.
_TIE_SEED = 15
_TIE_SCALE = 0.2
_TIE_COLUMNS = 256


def _tie_inputs(device: str = "cpu"):
    generator = torch.Generator().manual_seed(_TIE_SEED)
    weight = (
        torch.randn(1, _TIE_COLUMNS, generator=generator) * _TIE_SCALE
    ).to(torch.bfloat16).to(device)
    col_weights = (
        torch.rand(_TIE_COLUMNS, generator=generator).add(0.1).to(device)
    )
    return weight, col_weights


def _encode(weight, col_weights, backend="eager"):
    columns = int(weight.shape[1])
    return encode_trellis_planes(
        weight,
        col_weights,
        family=E2M1_FAMILY,
        schedule=[2] * columns,
        alphabets=_ALPHABET,
        scale_rule="static_6",
        sb_chunk=int(weight.shape[0]),
        determinism_mode="on",
        tailbite_candidates=4,
        backend=backend,
        point_route="full",
    )


def _pack(encoded, columns):
    return pack_planes(
        family=E2M1_FAMILY,
        body_rate_q256=512,
        schedule=[2] * columns,
        layout="fixed_quota_per_256",
        u_bits=encoded.u_bits,
        point_indices=encoded.point_indices,
        bypass_codes=encoded.bypass_codes,
        alphabets=_ALPHABET,
        scale_blob=encoded.scale_blob,
        global_scale_real=encoded.global_scale_real,
    ).to_bytes()


def _planes(blob, device):
    """The normalized code values, the raw e4m3 plane and the global."""
    wire = TrellisWire.from_bytes(blob)
    table = torch.tensor(
        [native_code_value(E2M1_FAMILY, code) for code in range(16)],
        dtype=torch.float32,
        device=device,
    )
    codes = decode_codes_torch(blob, device=device).to(torch.int64)
    scale_codes = torch.frombuffer(
        bytearray(wire.scale_blob), dtype=torch.uint8
    ).reshape(wire.rows, -1).to(device)
    e4m3 = scale_codes.view(torch.float8_e4m3fn).to(torch.float32)
    e4m3 = e4m3.repeat_interleave(16, dim=1)[:, : wire.columns]
    return table[codes], e4m3, float(wire.global_scale_real)


def test_the_two_associations_really_do_disagree_on_this_input():
    """Guard against a vacuous regression test.

    If a future change makes these two spellings agree everywhere, this test
    fails and the one below stops proving anything -- at which point the tie
    input must be re-derived, not the assertion deleted.
    """
    weight, col_weights = _tie_inputs()
    blob = _pack(_encode(weight, col_weights), _TIE_COLUMNS)
    normalized, e4m3, global_real = _planes(blob, "cpu")
    decoder_order = ((normalized * e4m3) * global_real).to(torch.bfloat16)
    folded_order = (
        normalized * (e4m3 * global_real).clamp_min(E2M1_SCALE_FLOOR)
    ).to(torch.bfloat16)
    disagreements = int((decoder_order != folded_order).sum())
    assert disagreements > 0, (
        "the tie input no longer exercises the association difference; "
        "re-derive it rather than weakening the test"
    )


def test_reconstruction_is_bit_identical_to_the_wire_decode():
    weight, col_weights = _tie_inputs()
    encoded = _encode(weight, col_weights)
    blob = _pack(encoded, _TIE_COLUMNS)
    decoded = decode_values_torch(blob, device="cpu", dtype=torch.float32)
    assert torch.equal(decoded, encoded.reconstruction.float())


def test_the_producer_accepts_the_tie_input():
    """Pre-fix this raised TrellisProducerError and aborted the render."""
    weight, col_weights = _tie_inputs()
    artifact = encode_trellis_one_linear(
        weight,
        col_weights,
        family=E2M1_FAMILY,
        body_rate_q256=512,
        schedule=[2] * _TIE_COLUMNS,
        layout="fixed_quota_per_256",
        alphabets=_ALPHABET,
        scale_rule="static_6",
        sb_chunk=1,
        determinism_mode="on",
        tailbite_candidates=4,
        backend="eager",
        point_route="full",
    )
    assert artifact.receipt["encoder_reconstruction_bf16_equal"] is True
    assert artifact.receipt["encoder_reconstruction_fp32_max_abs"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_both_backends_agree_with_the_decoder_on_the_tie_input():
    weight, col_weights = _tie_inputs("cuda")
    for backend in ("eager", "triton"):
        encoded = _encode(weight, col_weights, backend=backend)
        blob = _pack(encoded, _TIE_COLUMNS)
        decoded = decode_values_torch(blob, device="cuda", dtype=torch.float32)
        assert torch.equal(decoded, encoded.reconstruction.float()), backend
