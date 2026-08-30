"""Fail-closed producer scaffold for QTIP-style transforms plus E2M1 TCQ.

This module closes the deterministic basis and metadata halves of the proposed
one-Linear experiment.  It deliberately does *not* serialize a trellis wire:
PrismaQuant currently exposes the E2M1 rate/schedule/alphabet/accounting
contract, but no repository-owned Viterbi-output -> ``gridbook.trellis.wire.v1``
packer/decoder seam.  Importing Gridbook or copying its runtime is forbidden at
this repository boundary, and QTIP's bitshift wire is a different format.

``require_combined_wire_round_trip`` therefore always refuses.  The function
is intentional executable debt: a caller cannot mistake transformed floats
and a valid Gridbook sidecar for a physical, decoded Gridbook artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from typing import Any, Mapping

import torch

from prismaquant.trellis_formats import (
    E2M1_FAMILY,
    TRELLIS_WIRE_SCHEMA,
    get_trellis_family,
    validate_body_rate_q256,
)


SCAFFOLD_SCHEMA = (
    "prismaquant.research.qtip_trellis_online_hadamard_one_linear.v1"
)
RESEARCH_OPT_IN = "qtip_trellis_online_hadamard_one_linear_v1"
TRANSFORM_SCHEMA = "gridbook.qtip-online-hadamard.v1"
TRANSFORM_ALGORITHM = "block_walsh_hadamard"
TRANSFORM_NORMALIZATION = "orthonormal"
TRANSFORM_PADDING = "none"
SIGN_GENERATOR = "sha256_counter_rademacher"

_ROOT_PAYLOAD_FIELDS = frozenset({
    "schema", "algorithm", "normalization", "padding", "input", "output",
})
_ROOT_FIELDS = _ROOT_PAYLOAD_FIELDS | {"transform_sha256"}
_SIDE_FIELDS = frozenset({
    "dimension", "block_size", "seed", "sign_generator", "sign_sha256",
})
_SIGN_DOMAIN = (TRANSFORM_SCHEMA + "/signs\0").encode("ascii")


class MissingTrellisProducerSeam(RuntimeError):
    """The physical Gridbook wire cannot be produced or replayed in-tree."""


@dataclass(frozen=True)
class PreparedOneLinear:
    """Basis-transformed inputs plus an unregistered, non-artifact receipt."""

    transformed_weight: torch.Tensor
    transformed_hessian: torch.Tensor
    online_transform: Mapping[str, object]
    receipt: Mapping[str, object]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(
        list(contiguous.shape), separators=(",", ":")
    ).encode("ascii"))
    digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _plain_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be a JSON integer")
    return value


def _sign_bit_bytes(role: str, dimension: int, seed: int) -> bytes:
    """Reproduce Gridbook v1's language-independent Rademacher stream."""

    if role not in ("input", "output"):
        raise ValueError(f"role must be 'input' or 'output', got {role!r}")
    dimension = _plain_int(dimension, "dimension")
    seed = _plain_int(seed, "seed")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if not 0 <= seed < (1 << 64):
        raise ValueError("seed must fit uint64")
    prefix = (
        _SIGN_DOMAIN
        + role.encode("ascii") + b"\0"
        + dimension.to_bytes(8, "little")
        + seed.to_bytes(8, "little")
    )
    needed = (dimension + 7) // 8
    output = bytearray()
    counter = 0
    while len(output) < needed:
        output.extend(hashlib.sha256(
            prefix + counter.to_bytes(8, "little")
        ).digest())
        counter += 1
    del output[needed:]
    if dimension % 8:
        output[-1] &= (1 << (dimension % 8)) - 1
    return bytes(output)


def seeded_sign_digest(role: str, dimension: int, seed: int) -> str:
    return hashlib.sha256(_sign_bit_bytes(role, dimension, seed)).hexdigest()


def seeded_signs(
    role: str,
    dimension: int,
    seed: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    packed = _sign_bit_bytes(role, dimension, seed)
    values = [
        -1.0 if ((packed[index // 8] >> (index % 8)) & 1) else 1.0
        for index in range(dimension)
    ]
    return torch.tensor(values, device=device, dtype=dtype)


def online_transform_digest(value: Mapping[str, Any]) -> str:
    """Hash exactly the fields hashed by Gridbook's v1 producer helper."""

    return _canonical_sha256({
        field: value[field] for field in _ROOT_PAYLOAD_FIELDS
    })


def _validate_side(
    value: Any,
    *,
    role: str,
    dimension: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"online_transform.{role} must be an object")
    missing = sorted(_SIDE_FIELDS - set(value))
    unknown = sorted(set(value) - _SIDE_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"online_transform.{role} has missing={missing}, unknown={unknown}"
        )
    declared = _plain_int(
        value["dimension"], f"online_transform.{role}.dimension"
    )
    if declared != dimension:
        raise ValueError(
            f"online_transform.{role}.dimension={declared} does not match "
            f"the Linear dimension {dimension}"
        )
    block = _plain_int(
        value["block_size"], f"online_transform.{role}.block_size"
    )
    if block <= 0 or block & (block - 1):
        raise ValueError(
            f"online_transform.{role}.block_size must be a positive power of two"
        )
    if dimension % block:
        raise ValueError(
            f"online_transform.{role}.block_size={block} must divide "
            f"dimension={dimension}"
        )
    seed = _plain_int(value["seed"], f"online_transform.{role}.seed")
    if not 0 <= seed < (1 << 64):
        raise ValueError(f"online_transform.{role}.seed must fit uint64")
    if value["sign_generator"] != SIGN_GENERATOR:
        raise ValueError(
            f"online_transform.{role}.sign_generator must be {SIGN_GENERATOR!r}"
        )
    digest = value["sign_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            f"online_transform.{role}.sign_sha256 must be lowercase SHA-256"
        )
    expected = seeded_sign_digest(role, dimension, seed)
    if not hmac.compare_digest(digest, expected):
        raise ValueError(
            f"online_transform.{role}.sign_sha256 does not bind its seed"
        )
    return {
        "dimension": dimension,
        "block_size": block,
        "seed": seed,
        "sign_generator": SIGN_GENERATOR,
        "sign_sha256": digest,
    }


def validate_online_transform(
    value: Any,
    *,
    rows: int,
    columns: int,
) -> dict[str, object]:
    """Validate the closed Gridbook v1 sidecar without importing Gridbook."""

    if not isinstance(value, Mapping):
        raise ValueError("online_transform must be an object")
    missing = sorted(_ROOT_FIELDS - set(value))
    unknown = sorted(set(value) - _ROOT_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"online_transform has missing={missing}, unknown={unknown}"
        )
    fixed = {
        "schema": TRANSFORM_SCHEMA,
        "algorithm": TRANSFORM_ALGORITHM,
        "normalization": TRANSFORM_NORMALIZATION,
        "padding": TRANSFORM_PADDING,
    }
    for field, expected in fixed.items():
        if value[field] != expected:
            raise ValueError(
                f"online_transform.{field} must be {expected!r}, got "
                f"{value[field]!r}"
            )
    normalized = {
        **fixed,
        "input": _validate_side(
            value["input"], role="input", dimension=columns
        ),
        "output": _validate_side(
            value["output"], role="output", dimension=rows
        ),
    }
    digest = value["transform_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("online_transform.transform_sha256 must be lowercase SHA-256")
    expected = online_transform_digest(normalized)
    if not hmac.compare_digest(digest, expected):
        raise ValueError(
            "online_transform.transform_sha256 does not bind its semantics"
        )
    normalized["transform_sha256"] = digest
    return normalized


def build_online_transform(
    *,
    rows: int,
    columns: int,
    input_block_size: int,
    output_block_size: int,
    input_seed: int,
    output_seed: int,
) -> dict[str, object]:
    def side(role: str, dimension: int, block_size: int, seed: int):
        return {
            "dimension": dimension,
            "block_size": block_size,
            "seed": seed,
            "sign_generator": SIGN_GENERATOR,
            "sign_sha256": seeded_sign_digest(role, dimension, seed),
        }

    value: dict[str, object] = {
        "schema": TRANSFORM_SCHEMA,
        "algorithm": TRANSFORM_ALGORITHM,
        "normalization": TRANSFORM_NORMALIZATION,
        "padding": TRANSFORM_PADDING,
        "input": side("input", columns, input_block_size, input_seed),
        "output": side("output", rows, output_block_size, output_seed),
    }
    value["transform_sha256"] = online_transform_digest(value)
    return validate_online_transform(value, rows=rows, columns=columns)


def _normalized_block_hadamard_rows(
    value: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Gridbook's transparent FP32 block-Sylvester reference operation."""

    if value.ndim != 2:
        raise ValueError("Hadamard input must be rank two")
    dimension = int(value.shape[1])
    if dimension % block_size:
        raise ValueError("Hadamard block size must divide its dimension")
    work = value.float().reshape(
        value.shape[0], dimension // block_size, block_size
    )
    stride = 1
    while stride < block_size:
        paired = work.reshape(*work.shape[:-1], -1, 2, stride)
        left = paired[..., 0, :]
        right = paired[..., 1, :]
        work = torch.stack((left + right, left - right), dim=-2).reshape(
            *work.shape
        )
        stride *= 2
    return work.mul_(block_size ** -0.5).reshape(value.shape)


def _side_spec(contract: Mapping[str, object], role: str) -> Mapping[str, object]:
    value = contract[role]
    if not isinstance(value, Mapping):
        raise ValueError(f"online_transform.{role} must be an object")
    return value


def _input_basis_rows(
    value: torch.Tensor,
    contract: Mapping[str, object],
    role: str,
) -> torch.Tensor:
    side = _side_spec(contract, role)
    signs = seeded_signs(
        role,
        int(side["dimension"]),
        int(side["seed"]),
        device=value.device,
        dtype=value.dtype,
    )
    return _normalized_block_hadamard_rows(
        value * signs, int(side["block_size"])
    )


def _inverse_basis_rows(
    value: torch.Tensor,
    contract: Mapping[str, object],
    role: str,
) -> torch.Tensor:
    side = _side_spec(contract, role)
    signs = seeded_signs(
        role,
        int(side["dimension"]),
        int(side["seed"]),
        device=value.device,
        dtype=torch.float32,
    )
    return _normalized_block_hadamard_rows(
        value, int(side["block_size"])
    ) * signs


def transform_weight_and_hessian(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    online_transform: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``R_out W R_in.T`` and ``R_in H R_in.T`` in FP32."""

    if weight.ndim != 2:
        raise ValueError("weight must be rank two")
    rows, columns = map(int, weight.shape)
    contract = validate_online_transform(
        online_transform, rows=rows, columns=columns
    )
    if hessian.shape != (columns, columns):
        raise ValueError("hessian must be square over the Linear input dimension")
    if not bool(torch.isfinite(weight.float()).all()):
        raise ValueError("weight must be finite")
    if not bool(torch.isfinite(hessian.float()).all()):
        raise ValueError("hessian must be finite")
    hessian32 = hessian.float()
    if not torch.allclose(hessian32, hessian32.T, rtol=0.0, atol=1e-6):
        raise ValueError("hessian must be symmetric")

    # Row notation: x D H is the transpose of H D x_col.
    weight_right = _input_basis_rows(weight.float(), contract, "input")
    transformed_weight = _input_basis_rows(
        weight_right.T, contract, "output"
    ).T.contiguous()
    hessian_right = _input_basis_rows(hessian32, contract, "input")
    transformed_hessian = _input_basis_rows(
        hessian_right.T, contract, "input"
    ).T.contiguous()
    return transformed_weight, transformed_hessian


def transformed_activations(
    activations: torch.Tensor,
    online_transform: Mapping[str, object],
    *,
    runtime_bf16_boundary: bool = False,
) -> torch.Tensor:
    """Apply Gridbook's pre-native-FP4 row transform reference."""

    input_spec = _side_spec(online_transform, "input")
    output_spec = _side_spec(online_transform, "output")
    contract = validate_online_transform(
        online_transform,
        rows=int(output_spec["dimension"]),
        columns=int(activations.shape[1]) if activations.ndim == 2 else -1,
    )
    input_spec = _side_spec(contract, "input")
    if activations.ndim != 2 or activations.shape[1] != input_spec["dimension"]:
        raise ValueError("activations do not match online_transform.input")
    transformed = _input_basis_rows(activations, contract, "input")
    return transformed.to(torch.bfloat16) if runtime_bf16_boundary else transformed


def inverse_transformed_outputs(
    outputs: torch.Tensor,
    online_transform: Mapping[str, object],
    *,
    runtime_bf16_boundary: bool = False,
) -> torch.Tensor:
    """Apply Gridbook's post-``_scaled_mm`` row inverse reference."""

    input_spec = _side_spec(online_transform, "input")
    output_spec = _side_spec(online_transform, "output")
    contract = validate_online_transform(
        online_transform,
        rows=int(outputs.shape[1]) if outputs.ndim == 2 else -1,
        columns=int(input_spec["dimension"]),
    )
    output_spec = _side_spec(contract, "output")
    if outputs.ndim != 2 or outputs.shape[1] != output_spec["dimension"]:
        raise ValueError("outputs do not match online_transform.output")
    transformed = _inverse_basis_rows(outputs, contract, "output")
    return transformed.to(torch.bfloat16) if runtime_bf16_boundary else transformed


def decoded_weight_in_original_basis(
    decoded_transformed_weight: torch.Tensor,
    online_transform: Mapping[str, object],
) -> torch.Tensor:
    """Map a decoded ``Q`` to ``R_out.T Q R_in`` for reference scoring."""

    rows, columns = map(int, decoded_transformed_weight.shape)
    contract = validate_online_transform(
        online_transform, rows=rows, columns=columns
    )
    right = _inverse_basis_rows(
        decoded_transformed_weight.float(), contract, "input"
    )
    return _inverse_basis_rows(right.T, contract, "output").T.contiguous()


def verify_post_decode_serve_algebra(
    decoded_transformed_weight: torch.Tensor,
    activations: torch.Tensor,
    online_transform: Mapping[str, object],
) -> dict[str, object]:
    """Verify the serve algebra after, but not the identity of, a wire decode.

    The caller-supplied decoded tensor is intentionally *not* treated as proof
    of any wire.  This gate closes only the matrix orientation around a future
    exact decoder seam.
    """

    rows, columns = map(int, decoded_transformed_weight.shape)
    contract = validate_online_transform(
        online_transform, rows=rows, columns=columns
    )
    if activations.ndim != 2 or int(activations.shape[1]) != columns:
        raise ValueError("activations have the wrong input dimension")
    if not bool(torch.isfinite(decoded_transformed_weight.float()).all()):
        raise ValueError("decoded transformed weight must be finite")
    x_rot = transformed_activations(activations.float(), contract)
    transformed_output = x_rot @ decoded_transformed_weight.float().T
    served = inverse_transformed_outputs(transformed_output, contract)
    original_weight = decoded_weight_in_original_basis(
        decoded_transformed_weight, contract
    )
    expected = activations.float() @ original_weight.T
    error = (served - expected).abs()
    tolerance = 2e-5
    maximum = float(error.max()) if error.numel() else 0.0
    if not torch.allclose(served, expected, rtol=tolerance, atol=tolerance):
        raise AssertionError(
            f"online transform serve algebra mismatch; max_abs={maximum}"
        )
    return {
        "status": "post_decode_matrix_algebra_verified",
        "max_abs_error": maximum,
        "rtol": tolerance,
        "atol": tolerance,
        "wire_identity_verified": False,
        "runtime_fp4_activation_quantization_emulated": False,
        "claim_boundary": (
            "caller-supplied decoded tensor only; no Gridbook wire was parsed"
        ),
    }


def prepare_one_linear_scaffold(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    body_rate_q256: int,
    input_block_size: int,
    output_block_size: int,
    input_seed: int,
    output_seed: int,
    research_opt_in: str,
) -> PreparedOneLinear:
    """Prepare the exact basis and sidecar, but never claim a wire artifact."""

    if research_opt_in != RESEARCH_OPT_IN:
        raise ValueError(
            f"research_opt_in must equal {RESEARCH_OPT_IN!r}"
        )
    if weight.ndim != 2:
        raise ValueError("weight must be rank two")
    rows, columns = map(int, weight.shape)
    if rows <= 0 or columns <= 0 or columns % 16:
        raise ValueError("E2M1 trellis weight must be nonempty and group-16 aligned")
    family = get_trellis_family(E2M1_FAMILY)
    rate = validate_body_rate_q256(family, body_rate_q256)
    contract = build_online_transform(
        rows=rows,
        columns=columns,
        input_block_size=input_block_size,
        output_block_size=output_block_size,
        input_seed=input_seed,
        output_seed=output_seed,
    )
    transformed_weight, transformed_hessian = transform_weight_and_hessian(
        weight, hessian, contract
    )
    receipt_body: dict[str, object] = {
        "schema": SCAFFOLD_SCHEMA,
        "status": "blocked_missing_exact_trellis_encoder_decoder_seam",
        "scope": "research_only_one_linear_unregistered_contract_scaffold",
        "research_opt_in": RESEARCH_OPT_IN,
        "shape": {"rows": rows, "columns": columns},
        "source": {
            "weight": {
                "dtype": str(weight.dtype),
                "sha256": _tensor_sha256(weight),
            },
            "hessian": {
                "dtype": str(hessian.dtype),
                "sha256": _tensor_sha256(hessian),
            },
        },
        "transformed": {
            "weight": {
                "dtype": str(transformed_weight.dtype),
                "sha256": _tensor_sha256(transformed_weight),
            },
            "hessian": {
                "dtype": str(transformed_hessian.dtype),
                "sha256": _tensor_sha256(transformed_hessian),
            },
        },
        "basis": {
            "weight": "R_out W R_in.T",
            "hessian": "R_in H R_in.T",
            "row_input": "x D_in H_in",
            "row_output_inverse": "y H_out D_out",
            "weight_dtype": "torch.float32",
            "hessian_dtype": "torch.float32",
        },
        "online_transform": contract,
        "wire": {
            "schema": TRELLIS_WIRE_SCHEMA,
            "family": E2M1_FAMILY,
            "body_rate_q256": rate,
            "terminal_grid": "E2M1",
            "scale_contract": family.scale_contract,
            "qtip_bitshift_wire_allowed": False,
            "wire_bytes": None,
            "wire_identity_sha256": None,
            "encoder_invoked": False,
            "decoder_invoked": False,
        },
        "missing_seam": {
            "present_in_prismaquant": [
                "trellis family/rate/schedule/alphabet validation",
                "pre-render footprint accounting",
                "QTIP-style BlockLDLQ research isolate",
            ],
            "absent_from_prismaquant_repository_api": [
                "tail-biting Viterbi path planes",
                "gridbook.trellis.wire.v1 packer returning physical bytes",
                "wire-v1 reference parser/decoder bound to those bytes",
            ],
            "forbidden_substitutions": [
                "QTIP bitshift wire",
                "vendored or imported Gridbook runtime",
                "unparsed caller-asserted decoded weights",
            ],
        },
        "format_registry_entries_created": 0,
        "runtime_pin_changed": False,
        "production_contract_changed": False,
        "producer_eligible": False,
    }
    receipt = {
        **receipt_body,
        "identity_sha256": _canonical_sha256(receipt_body),
    }
    return PreparedOneLinear(
        transformed_weight=transformed_weight,
        transformed_hessian=transformed_hessian,
        online_transform=contract,
        receipt=receipt,
    )


def require_combined_wire_round_trip(_prepared: PreparedOneLinear) -> None:
    """Refuse until PrismaQuant has one exact encoder+packer+decoder seam."""

    raise MissingTrellisProducerSeam(
        "combined QTIP-basis/E2M1-trellis publication is unavailable: "
        "PrismaQuant has no repository API that returns tail-biting path "
        "planes, packs them as gridbook.trellis.wire.v1, and reference-"
        "decodes the same bytes; refusing QTIP bitshift, a Gridbook import, "
        "or a caller-asserted decoded tensor"
    )


__all__ = [
    "MissingTrellisProducerSeam",
    "PreparedOneLinear",
    "RESEARCH_OPT_IN",
    "SCAFFOLD_SCHEMA",
    "SIGN_GENERATOR",
    "TRANSFORM_ALGORITHM",
    "TRANSFORM_NORMALIZATION",
    "TRANSFORM_PADDING",
    "TRANSFORM_SCHEMA",
    "build_online_transform",
    "decoded_weight_in_original_basis",
    "inverse_transformed_outputs",
    "online_transform_digest",
    "prepare_one_linear_scaffold",
    "require_combined_wire_round_trip",
    "seeded_sign_digest",
    "seeded_signs",
    "transform_weight_and_hessian",
    "transformed_activations",
    "validate_online_transform",
    "verify_post_decode_serve_algebra",
]
