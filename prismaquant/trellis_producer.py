"""Versioned one-Linear producer for canonical Gridbook trellis wire bytes.

This module is the narrow repository-owned boundary shared by research
producers.  It does not register a format or integrate a cache/export path.
All value-bearing encoder choices are explicit, and the returned decoded
matrix is derived only by parsing the immutable bytes just produced.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import torch

from .trellis_encoder import encode_trellis_planes, encoder_source_sha256
from .trellis_formats import TRELLIS_WIRE_SCHEMA
from .trellis_wire import (
    TrellisWire,
    decode_codes_torch,
    decode_values_torch,
    pack_planes,
)


TRELLIS_ONE_LINEAR_PRODUCER_SCHEMA = (
    "prismaquant.trellis_one_linear_producer.v1"
)


class TrellisProducerError(RuntimeError):
    """An exact encode/pack/parse/decode invariant did not hold."""


@dataclass(frozen=True, slots=True)
class TrellisOneLinearArtifact:
    """Immutable wire plus reference views decoded from those exact bytes."""

    wire_bytes: bytes
    decoded_codes: torch.Tensor
    decoded_weight: torch.Tensor
    receipt: Mapping[str, object]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def encode_trellis_one_linear(
    weight: torch.Tensor,
    col_weights: torch.Tensor,
    *,
    family: str,
    body_rate_q256: int,
    schedule: Sequence[int],
    layout: str,
    alphabets: Mapping[int, Sequence[int]],
    scale_rule: str,
    sb_chunk: int,
    determinism_mode: str,
    tailbite_candidates: int,
    backend: str,
    point_route: str,
) -> TrellisOneLinearArtifact:
    """Encode, serialize, reparse, and decode one dense Linear.

    The decoder is never allowed to consume the in-memory encoder result.
    This makes the byte identity, rather than a caller-supplied tensor, the
    evidence boundary.
    """

    encoded = encode_trellis_planes(
        weight,
        col_weights,
        family=family,
        schedule=schedule,
        alphabets=alphabets,
        scale_rule=scale_rule,
        sb_chunk=sb_chunk,
        determinism_mode=determinism_mode,
        tailbite_candidates=tailbite_candidates,
        backend=backend,
        point_route=point_route,
    )
    wire = pack_planes(
        family=family,
        body_rate_q256=body_rate_q256,
        schedule=schedule,
        layout=layout,
        u_bits=encoded.u_bits,
        point_indices=encoded.point_indices,
        bypass_codes=encoded.bypass_codes,
        alphabets=alphabets,
        scale_blob=encoded.scale_blob,
        global_scale_real=encoded.global_scale_real,
    )
    blob = wire.to_bytes()
    reparsed = TrellisWire.from_bytes(blob)
    if reparsed.to_bytes() != blob:
        raise TrellisProducerError(
            "canonical wire did not reserialize byte-identically"
        )

    decoded_codes = decode_codes_torch(blob, device=weight.device)
    decoded_weight = decode_values_torch(
        blob, device=weight.device, dtype=weight.dtype
    )
    expected = encoded.reconstruction.to(weight.dtype)
    decoded_bf16 = decoded_weight.to(torch.bfloat16)
    expected_bf16 = expected.to(torch.bfloat16)
    if not torch.equal(decoded_bf16, expected_bf16):
        mismatch = int((decoded_bf16 != expected_bf16).sum().item())
        raise TrellisProducerError(
            "same-byte reference decode differs from the encoder "
            f"reconstruction at the BF16 validation boundary at {mismatch} values"
        )
    fp32_max_abs = float(
        (decoded_weight.float() - expected.float()).abs().max().item()
    )

    recipe = {
        "family": reparsed.family,
        "body_rate_q256": reparsed.body_rate_q256,
        "layout": reparsed.layout,
        "schedule": list(reparsed.schedule),
        "alphabets": {
            str(rate): list(codes)
            for rate, codes in sorted(reparsed.alphabets.items())
        },
        "scale_rule": scale_rule,
        "sb_chunk": int(sb_chunk),
        "determinism_mode": determinism_mode,
        "tailbite_candidates": int(tailbite_candidates),
        "backend": backend,
        "point_route": point_route,
        "encoder_source_sha256": encoder_source_sha256(),
    }
    receipt_body: dict[str, object] = {
        "schema": TRELLIS_ONE_LINEAR_PRODUCER_SCHEMA,
        "status": "same_byte_encode_pack_parse_decode_verified",
        "scope": "shared_one_linear_reference_unregistered",
        "wire_schema": TRELLIS_WIRE_SCHEMA,
        "shape": [int(weight.shape[0]), int(weight.shape[1])],
        "source_weight_sha256": _tensor_sha256(weight),
        "col_weights_sha256": _tensor_sha256(col_weights),
        "recipe": recipe,
        "recipe_identity_sha256": _canonical_sha256(recipe),
        "wire_bytes": len(blob),
        "wire_identity_sha256": hashlib.sha256(blob).hexdigest(),
        "decoded_codes_sha256": _tensor_sha256(decoded_codes),
        "decoded_weight_sha256": _tensor_sha256(decoded_weight),
        "same_byte_reparse_verified": True,
        "encoder_reconstruction_bf16_equal": True,
        "encoder_reconstruction_fp32_max_abs": fp32_max_abs,
        "format_registry_entries_created": 0,
        "runtime_pin_changed": False,
        "production_contract_changed": False,
        "producer_eligible": False,
    }
    receipt = {
        **receipt_body,
        "identity_sha256": _canonical_sha256(receipt_body),
    }
    return TrellisOneLinearArtifact(
        wire_bytes=blob,
        decoded_codes=decoded_codes.contiguous(),
        decoded_weight=decoded_weight.contiguous(),
        receipt=receipt,
    )


__all__ = [
    "TRELLIS_ONE_LINEAR_PRODUCER_SCHEMA",
    "TrellisOneLinearArtifact",
    "TrellisProducerError",
    "encode_trellis_one_linear",
]
