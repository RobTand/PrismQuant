"""Research producer for QTIP-style transforms plus PrismaQuant E2M1 TCQ.

The online-transform contract is independent of Gridbook and the physical
wire remains PrismaQuant's existing ``gridbook.trellis.wire.v1`` carrier.
The combined entry point is explicit, opt-in, and unregistered; it encodes,
packs, reparses, and reference-decodes the same immutable bytes before it can
report a successful one-Linear algebra check.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from typing import Any, Callable, Mapping, Sequence

import torch

from prismaquant.trellis_formats import (
    E2M1_FAMILY,
    TRELLIS_WIRE_SCHEMA,
    get_trellis_family,
    validate_body_rate_q256,
)
from prismaquant.trellis_encoder import (
    EncodedTrellisPlanes,
    encode_trellis_planes,
    encoder_source_sha256,
)
from prismaquant.trellis_producer import (
    TrellisOneLinearArtifact,
    encode_trellis_one_linear,
)
from prismaquant.trellis_wire import (
    TrellisWire,
    decode_codes_torch,
    decode_values_torch,
    pack_planes,
)


SCAFFOLD_SCHEMA = (
    "prismaquant.research.qtip_trellis_online_hadamard_one_linear.v1"
)
COMBINED_ARTIFACT_SCHEMA = (
    "prismaquant.research.qtip_trellis_online_hadamard_artifact.v1"
)
BLOCKLDL_COMBINED_ARTIFACT_SCHEMA = (
    "prismaquant.research.qtip_blockldl_trellis_hadamard_artifact.v1"
)
TRELLIS_FEEDBACK_BLOCK_SIZE = 256
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


@dataclass(frozen=True)
class PreparedOneLinear:
    """Basis-transformed inputs plus an unregistered, non-artifact receipt."""

    transformed_weight: torch.Tensor
    transformed_hessian: torch.Tensor
    online_transform: Mapping[str, object]
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class CombinedOneLinearArtifact:
    """Physical trellis bytes and their transformed-basis decoded view."""

    wire_bytes: bytes
    decoded_transformed_weight: torch.Tensor
    decoded_codes: torch.Tensor
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
        "status": "prepared_exact_trellis_wire_seam_available",
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
        "wire_seam": {
            "available_repository_api": [
                "tail-biting Viterbi path planes",
                "gridbook.trellis.wire.v1 immutable byte packer",
                "same-byte canonical parser and reference decoder",
            ],
            "excluded_substitutions": [
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


def _validate_prepared_one_linear(
    prepared: PreparedOneLinear,
    *,
    body_rate_q256: int,
) -> dict[str, object]:
    """Reauthenticate the prepared receipt at the encode trust boundary."""

    if not isinstance(prepared, PreparedOneLinear):
        raise ValueError("prepared must be a PreparedOneLinear")
    if not isinstance(prepared.receipt, Mapping):
        raise ValueError("prepared receipt must be an object")
    body = dict(prepared.receipt)
    identity = body.pop("identity_sha256", None)
    if not isinstance(identity, str) or not hmac.compare_digest(
        identity, _canonical_sha256(body)
    ):
        raise ValueError("prepared receipt identity mismatch")
    if body.get("schema") != SCAFFOLD_SCHEMA:
        raise ValueError("prepared receipt schema mismatch")
    if body.get("research_opt_in") != RESEARCH_OPT_IN:
        raise ValueError("prepared receipt research opt-in mismatch")
    rows, columns = map(int, prepared.transformed_weight.shape)
    if body.get("shape") != {"rows": rows, "columns": columns}:
        raise ValueError("prepared receipt shape mismatch")
    transformed = body.get("transformed")
    if not isinstance(transformed, Mapping):
        raise ValueError("prepared receipt transformed identity is missing")
    expected = {
        "weight": {
            "dtype": str(prepared.transformed_weight.dtype),
            "sha256": _tensor_sha256(prepared.transformed_weight),
        },
        "hessian": {
            "dtype": str(prepared.transformed_hessian.dtype),
            "sha256": _tensor_sha256(prepared.transformed_hessian),
        },
    }
    if transformed != expected:
        raise ValueError("prepared transformed tensor identity mismatch")
    wire = body.get("wire")
    if not isinstance(wire, Mapping):
        raise ValueError("prepared receipt wire contract is missing")
    if (
        wire.get("schema") != TRELLIS_WIRE_SCHEMA
        or wire.get("family") != E2M1_FAMILY
        or wire.get("body_rate_q256") != body_rate_q256
        or wire.get("qtip_bitshift_wire_allowed") is not False
    ):
        raise ValueError("prepared receipt wire contract mismatch")
    contract = validate_online_transform(
        body.get("online_transform"), rows=rows, columns=columns
    )
    if contract != validate_online_transform(
        prepared.online_transform, rows=rows, columns=columns
    ):
        raise ValueError("prepared online-transform metadata mismatch")
    return body


def qtip_block_ldl_factors(
    hessian: torch.Tensor,
    *,
    block_size: int = TRELLIS_FEEDBACK_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pinned-QTIP block feedback ``L`` and dense diagonal blocks.

    ``feedback_lower`` has zero diagonal blocks because only
    ``L[k, j], k > j`` enters the reverse recurrence. ``diagonal_blocks``
    retains the complete dense ``D_j`` factors; callers must not mislabel its
    diagonal as the full local Hessian objective.
    """

    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Hessian must be square")
    columns = int(hessian.shape[0])
    if columns < block_size or columns % block_size:
        raise ValueError(
            f"Hessian width must be divisible by block_size={block_size}"
        )
    value = hessian.float()
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("Hessian must be finite")
    if not torch.allclose(value, value.T, rtol=0.0, atol=1.0e-5):
        raise ValueError("Hessian must be symmetric")
    value = ((value + value.T) * 0.5).contiguous()
    try:
        chol = torch.linalg.cholesky(value)
    except RuntimeError as exc:
        raise ValueError("Hessian must be positive definite") from exc
    unit_lower = chol.clone()
    diagonal_blocks = []
    for first in range(0, columns, block_size):
        last = first + block_size
        diagonal_factor = chol[first:last, first:last]
        diagonal_blocks.append(diagonal_factor @ diagonal_factor.T)
        unit_lower[:, first:last] = torch.linalg.solve_triangular(
            diagonal_factor.T,
            chol[:, first:last].T,
            upper=True,
        ).T
        unit_lower[first:last, first:last] = torch.eye(
            block_size, dtype=value.dtype, device=value.device
        )
    feedback_lower = unit_lower.clone()
    for first in range(0, columns, block_size):
        feedback_lower[first:first + block_size, first:first + block_size] = 0
    return feedback_lower.contiguous(), torch.stack(diagonal_blocks)


def reverse_block_feedback_reference(
    weight: torch.Tensor,
    feedback_lower: torch.Tensor,
    terminal: Callable[[int, torch.Tensor], torch.Tensor],
    *,
    block_size: int = TRELLIS_FEEDBACK_BLOCK_SIZE,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Unbuffered mathematical oracle for the reverse BlockLDL recurrence."""

    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if weight.ndim != 2 or int(weight.shape[1]) % block_size:
        raise ValueError("weight width must be block aligned")
    if feedback_lower.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("feedback_lower shape differs from weight width")
    q = torch.zeros_like(weight)
    targets: list[torch.Tensor | None] = [None] * (
        int(weight.shape[1]) // block_size
    )
    for first in range(int(weight.shape[1]) - block_size, -1, -block_size):
        last = first + block_size
        target = weight[:, first:last].clone()
        if last < weight.shape[1]:
            target += (weight[:, last:] - q[:, last:]) @ feedback_lower[
                last:, first:last
            ]
        block_index = first // block_size
        decoded = terminal(block_index, target)
        if decoded.shape != target.shape or not bool(
            torch.isfinite(decoded).all().item()
        ):
            raise ValueError("terminal returned an invalid decoded block")
        q[:, first:last] = decoded
        targets[block_index] = target
    return q, tuple(value for value in targets if value is not None)


def reverse_block_feedback_buffered(
    weight: torch.Tensor,
    feedback_lower: torch.Tensor,
    terminal: Callable[[int, torch.Tensor], torch.Tensor],
    *,
    block_size: int = TRELLIS_FEEDBACK_BLOCK_SIZE,
    buffer_blocks: int = 1,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Pinned-QTIP buffered recurrence generalized to 256-column terminals."""

    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if weight.ndim != 2 or int(weight.shape[1]) % block_size:
        raise ValueError("weight width must be block aligned")
    columns = int(weight.shape[1])
    if feedback_lower.shape != (columns, columns):
        raise ValueError("feedback_lower shape differs from weight width")
    if type(buffer_blocks) is not int or buffer_blocks < 1:
        raise ValueError("buffer_blocks must be a positive integer")
    block_count = columns // block_size
    q = torch.zeros_like(weight)
    prod_cache = torch.zeros_like(weight)
    targets: list[torch.Tensor | None] = [None] * block_count
    for end_block in range(block_count, 0, -int(buffer_blocks)):
        first_block = max(0, end_block - int(buffer_blocks))
        buffer_first = first_block * block_size
        buffer_last = end_block * block_size
        for block_index in range(end_block - 1, first_block - 1, -1):
            first = block_index * block_size
            last = first + block_size
            target = weight[:, first:last] + prod_cache[:, first:last]
            if last < buffer_last:
                target = target + (
                    weight[:, last:buffer_last] - q[:, last:buffer_last]
                ) @ feedback_lower[last:buffer_last, first:last]
            decoded = terminal(block_index, target)
            if decoded.shape != target.shape or not bool(
                torch.isfinite(decoded).all().item()
            ):
                raise ValueError("terminal returned an invalid decoded block")
            q[:, first:last] = decoded
            targets[block_index] = target
        prod_cache += (
            weight[:, buffer_first:buffer_last]
            - q[:, buffer_first:buffer_last]
        ) @ feedback_lower[buffer_first:buffer_last, :]
    return q, tuple(value for value in targets if value is not None)


def require_combined_wire_round_trip(
    prepared: PreparedOneLinear,
    activations: torch.Tensor,
    *,
    body_rate_q256: int,
    schedule: tuple[int, ...] | list[int],
    layout: str,
    alphabets: Mapping[int, tuple[int, ...] | list[int]],
    scale_rule: str,
    sb_chunk: int,
    determinism_mode: str,
    tailbite_candidates: int,
    backend: str,
    point_route: str,
    research_opt_in: str,
) -> CombinedOneLinearArtifact:
    """Produce and verify the exact combined research artifact.

    The promoted encoder's objective is column-diagonal.  Consequently this
    seam consumes exactly ``diag(H_tilde)`` and records that limitation; it
    does not claim the encoder applies full BlockLDLQ off-diagonal feedback.
    """

    if research_opt_in != RESEARCH_OPT_IN:
        raise ValueError(f"research_opt_in must equal {RESEARCH_OPT_IN!r}")
    prepared_receipt = _validate_prepared_one_linear(
        prepared, body_rate_q256=body_rate_q256
    )
    rows, columns = map(int, prepared.transformed_weight.shape)
    contract = validate_online_transform(
        prepared.online_transform, rows=rows, columns=columns
    )
    if prepared.transformed_hessian.shape != (columns, columns):
        raise ValueError("prepared transformed Hessian has the wrong shape")
    objective = torch.diagonal(prepared.transformed_hessian).contiguous()
    if not bool(torch.isfinite(objective).all().item()) or bool(
        (objective < 0).any().item()
    ):
        raise ValueError(
            "diag(transformed_hessian) must be finite and nonnegative"
        )
    if float(objective.sum().item()) <= 0.0:
        raise ValueError("diag(transformed_hessian) must contain positive mass")

    trellis: TrellisOneLinearArtifact = encode_trellis_one_linear(
        prepared.transformed_weight,
        objective,
        family=E2M1_FAMILY,
        body_rate_q256=body_rate_q256,
        schedule=schedule,
        layout=layout,
        alphabets=alphabets,
        scale_rule=scale_rule,
        sb_chunk=sb_chunk,
        determinism_mode=determinism_mode,
        tailbite_candidates=tailbite_candidates,
        backend=backend,
        point_route=point_route,
    )
    serve = dict(verify_post_decode_serve_algebra(
        trellis.decoded_weight, activations, contract
    ))
    serve.update({
        "wire_identity_verified": True,
        "wire_identity_sha256": trellis.receipt["wire_identity_sha256"],
        "claim_boundary": (
            "decoded by PrismaQuant's reference decoder from the same "
            "canonical gridbook.trellis.wire.v1 bytes"
        ),
    })
    receipt_body: dict[str, object] = {
        "schema": COMBINED_ARTIFACT_SCHEMA,
        "status": "physical_wire_and_serve_algebra_verified",
        "scope": "research_only_one_linear_unregistered",
        "research_opt_in": RESEARCH_OPT_IN,
        "shape": {"rows": rows, "columns": columns},
        "online_transform": contract,
        "prepared_receipt_identity_sha256": prepared.receipt["identity_sha256"],
        "prepared_receipt_status": prepared_receipt["status"],
        "trellis_objective": {
            "kind": "diag(transformed_hessian)",
            "sha256": _tensor_sha256(objective),
            "full_off_diagonal_blockldlq_applied": False,
        },
        "trellis": dict(trellis.receipt),
        "serve_algebra": serve,
        "qtip_bitshift_wire_allowed": False,
        "format_registry_entries_created": 0,
        "runtime_pin_changed": False,
        "production_contract_changed": False,
        "producer_eligible": False,
    }
    receipt = {
        **receipt_body,
        "identity_sha256": _canonical_sha256(receipt_body),
    }
    return CombinedOneLinearArtifact(
        wire_bytes=trellis.wire_bytes,
        decoded_transformed_weight=trellis.decoded_weight,
        decoded_codes=trellis.decoded_codes,
        online_transform=contract,
        receipt=receipt,
    )


def require_blockldl_trellis_wire_round_trip(
    prepared: PreparedOneLinear,
    activations: torch.Tensor,
    *,
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
    terminal_metric_mode: str,
    buffer_blocks: int,
    research_opt_in: str,
) -> CombinedOneLinearArtifact:
    """Run 256-column reverse BlockLDL feedback with exact trellis terminals.

    This is a QTIP-derived recurrence, not QTIP's 16-by-16 codebook geometry.
    Each Gridbook physical row-by-256 tail-biting cycle is one indivisible LDL
    block.  A single precommitted tensor-global scale is shared by every block
    so the resulting planes can be represented by one canonical wire.
    """

    if research_opt_in != RESEARCH_OPT_IN:
        raise ValueError(f"research_opt_in must equal {RESEARCH_OPT_IN!r}")
    if terminal_metric_mode == "dense_block_D":
        raise ValueError(
            "dense_block_D is unsupported: its cross-coordinate residual "
            "terms are not representable by the current additive 256-state "
            "trellis dynamic program"
        )
    if terminal_metric_mode not in {"diag_block_D", "qtip_frobenius"}:
        raise ValueError(
            "terminal_metric_mode must be 'diag_block_D', "
            "'qtip_frobenius', or the explicitly refused 'dense_block_D'"
        )
    prepared_receipt = _validate_prepared_one_linear(
        prepared, body_rate_q256=body_rate_q256
    )
    weight = prepared.transformed_weight.float()
    rows, columns = map(int, weight.shape)
    if columns % TRELLIS_FEEDBACK_BLOCK_SIZE:
        raise ValueError("trellis BlockLDL weight width must be divisible by 256")
    if len(schedule) != columns:
        raise ValueError("schedule length differs from transformed weight width")
    contract = validate_online_transform(
        prepared.online_transform, rows=rows, columns=columns
    )
    feedback_lower, diagonal_blocks = qtip_block_ldl_factors(
        prepared.transformed_hessian,
        block_size=TRELLIS_FEEDBACK_BLOCK_SIZE,
    )

    group_scales = weight.reshape(rows, columns // 16, 16).abs().amax(-1)
    group_scales = group_scales.clamp_min(1.0e-12) / 6.0
    shared_global_tensor = (
        group_scales.amax() / 448.0
    ).clamp_min(1.0e-12).to(torch.float32)
    shared_global = float(shared_global_tensor.item())

    encoded_blocks: dict[int, EncodedTrellisPlanes] = {}
    block_receipts: dict[int, dict[str, object]] = {}

    def terminal(block_index: int, target: torch.Tensor) -> torch.Tensor:
        first = block_index * TRELLIS_FEEDBACK_BLOCK_SIZE
        last = first + TRELLIS_FEEDBACK_BLOCK_SIZE
        block_schedule = tuple(int(value) for value in schedule[first:last])
        local_body_rate_q256 = sum(block_schedule)
        bypass_rate = get_trellis_family(E2M1_FAMILY).bypass_rate
        block_rates = sorted({
            rate for rate in block_schedule if rate < bypass_rate
        })
        try:
            block_alphabets = {
                rate: alphabets[rate] for rate in block_rates
            }
        except KeyError as exc:
            raise ValueError(
                f"missing alphabet for block rate {int(exc.args[0])}"
            ) from exc
        dense_d = diagonal_blocks[block_index]
        metric = (
            dense_d.diagonal().contiguous()
            if terminal_metric_mode == "diag_block_D"
            else torch.ones(
                TRELLIS_FEEDBACK_BLOCK_SIZE,
                dtype=torch.float32,
                device=target.device,
            )
        )
        encoded = encode_trellis_planes(
            target,
            metric,
            family=E2M1_FAMILY,
            schedule=block_schedule,
            alphabets=block_alphabets,
            scale_rule=scale_rule,
            sb_chunk=sb_chunk,
            determinism_mode=determinism_mode,
            tailbite_candidates=tailbite_candidates,
            backend=backend,
            point_route=point_route,
            global_scale_real_override=shared_global,
        )
        terminal_wire = pack_planes(
            family=E2M1_FAMILY,
            body_rate_q256=local_body_rate_q256,
            schedule=block_schedule,
            layout=layout,
            u_bits=encoded.u_bits,
            point_indices=encoded.point_indices,
            bypass_codes=encoded.bypass_codes,
            alphabets=block_alphabets,
            scale_blob=encoded.scale_blob,
            global_scale_real=shared_global,
        )
        terminal_blob = terminal_wire.to_bytes()
        if TrellisWire.from_bytes(terminal_blob).to_bytes() != terminal_blob:
            raise AssertionError("terminal wire did not reserialize exactly")
        decoded_terminal = decode_values_torch(
            terminal_blob, device=target.device, dtype=target.dtype
        )
        if not torch.equal(
            decoded_terminal.to(torch.bfloat16),
            encoded.reconstruction.to(torch.bfloat16),
        ):
            raise AssertionError(
                "terminal same-byte decode differs from encoder reconstruction"
            )
        target_group_scales = target.reshape(rows, 16, 16).abs().amax(-1) / 6.0
        clipped = int(
            (target_group_scales > shared_global_tensor * 448.0).sum().item()
        )
        encoded_blocks[block_index] = encoded
        block_receipts[block_index] = {
            "block_index": block_index,
            "first_column": first,
            "last_column_exclusive": last,
            "local_body_rate_q256": local_body_rate_q256,
            "feedback_target_sha256": _tensor_sha256(target),
            "terminal_wire_identity_sha256": hashlib.sha256(
                terminal_blob
            ).hexdigest(),
            "decoded_terminal_sha256": _tensor_sha256(decoded_terminal),
            "dense_D_sha256": _tensor_sha256(dense_d),
            "terminal_metric_sha256": _tensor_sha256(metric),
            "terminal_metric_mode": terminal_metric_mode,
            "dense_D_consumed_by_terminal": False,
            "clipped_group_count_at_fixed_global": clipped,
        }
        return decoded_terminal

    recurrence_q, buffered_targets = reverse_block_feedback_buffered(
        weight,
        feedback_lower,
        terminal,
        block_size=TRELLIS_FEEDBACK_BLOCK_SIZE,
        buffer_blocks=buffer_blocks,
    )
    block_count = columns // TRELLIS_FEEDBACK_BLOCK_SIZE
    if set(encoded_blocks) != set(range(block_count)):
        raise AssertionError("BlockLDL recurrence did not encode every block")

    def decoded_replay(block_index: int, _target: torch.Tensor) -> torch.Tensor:
        first = block_index * TRELLIS_FEEDBACK_BLOCK_SIZE
        return recurrence_q[:, first:first + TRELLIS_FEEDBACK_BLOCK_SIZE]

    oracle_q, oracle_targets = reverse_block_feedback_reference(
        weight,
        feedback_lower,
        decoded_replay,
        block_size=TRELLIS_FEEDBACK_BLOCK_SIZE,
    )
    if not torch.equal(oracle_q, recurrence_q):
        raise AssertionError("buffered recurrence decoded blocks changed")
    target_max_abs = max(
        float((buffered - oracle).abs().max().item())
        for buffered, oracle in zip(buffered_targets, oracle_targets, strict=True)
    )
    if not all(
        torch.allclose(buffered, oracle, rtol=3.0e-5, atol=3.0e-5)
        for buffered, oracle in zip(
            buffered_targets, oracle_targets, strict=True
        )
    ):
        raise AssertionError(
            "buffered BlockLDL targets differ from the unbuffered oracle; "
            f"max_abs={target_max_abs}"
        )

    ordered = [encoded_blocks[index] for index in range(block_count)]
    scale_planes = [
        torch.frombuffer(bytearray(value.scale_blob), dtype=torch.uint8).reshape(
            rows, TRELLIS_FEEDBACK_BLOCK_SIZE // 16
        )
        for value in ordered
    ]
    scale_blob = (
        torch.cat(scale_planes, dim=1).contiguous().numpy().tobytes()
    )
    wire = pack_planes(
        family=E2M1_FAMILY,
        body_rate_q256=body_rate_q256,
        schedule=schedule,
        layout=layout,
        u_bits=torch.cat([value.u_bits for value in ordered], dim=1),
        point_indices=torch.cat(
            [value.point_indices for value in ordered], dim=1
        ),
        bypass_codes=torch.cat(
            [value.bypass_codes for value in ordered], dim=1
        ),
        alphabets=alphabets,
        scale_blob=scale_blob,
        global_scale_real=shared_global,
    )
    blob = wire.to_bytes()
    if TrellisWire.from_bytes(blob).to_bytes() != blob:
        raise AssertionError("BlockLDL trellis wire did not reserialize exactly")
    decoded_codes = decode_codes_torch(blob, device=weight.device)
    decoded_weight = decode_values_torch(
        blob, device=weight.device, dtype=weight.dtype
    )
    if not torch.equal(decoded_weight, recurrence_q):
        raise AssertionError(
            "same-byte wire decode differs from recurrence terminal decodes"
        )

    unit_lower = feedback_lower.clone()
    for first in range(0, columns, TRELLIS_FEEDBACK_BLOCK_SIZE):
        unit_lower[
            first:first + TRELLIS_FEEDBACK_BLOCK_SIZE,
            first:first + TRELLIS_FEEDBACK_BLOCK_SIZE,
        ] = torch.eye(
            TRELLIS_FEEDBACK_BLOCK_SIZE,
            dtype=unit_lower.dtype,
            device=unit_lower.device,
        )
    dense_d = torch.block_diag(*diagonal_blocks.unbind(0))
    reconstructed_hessian = unit_lower @ dense_d @ unit_lower.T
    factor_max_abs = float(
        (reconstructed_hessian - prepared.transformed_hessian.float())
        .abs()
        .max()
        .item()
    )
    if not torch.allclose(
        reconstructed_hessian,
        prepared.transformed_hessian.float(),
        rtol=3.0e-5,
        atol=3.0e-5,
    ):
        raise AssertionError(
            "BlockLDL factors do not reconstruct the transformed Hessian; "
            f"max_abs={factor_max_abs}"
        )
    error = weight - decoded_weight.float()
    transformed_error = error @ unit_lower
    decomposed_proxy = sum(
        (
            transformed_error[
                :, index * TRELLIS_FEEDBACK_BLOCK_SIZE:
                (index + 1) * TRELLIS_FEEDBACK_BLOCK_SIZE
            ]
            @ diagonal_blocks[index]
        )
        .mul(
            transformed_error[
                :, index * TRELLIS_FEEDBACK_BLOCK_SIZE:
                (index + 1) * TRELLIS_FEEDBACK_BLOCK_SIZE
            ]
        )
        .sum()
        for index in range(block_count)
    )
    direct_proxy = ((error @ prepared.transformed_hessian.float()) * error).sum()
    proxy_abs = float((decomposed_proxy - direct_proxy).abs().item())
    if not torch.allclose(
        decomposed_proxy, direct_proxy, rtol=3.0e-5, atol=3.0e-5
    ):
        raise AssertionError(
            "BlockLDL quadratic decomposition mismatch; "
            f"abs_error={proxy_abs}"
        )

    serve = dict(verify_post_decode_serve_algebra(
        decoded_weight, activations, contract
    ))
    wire_sha256 = hashlib.sha256(blob).hexdigest()
    serve.update({
        "wire_identity_verified": True,
        "wire_identity_sha256": wire_sha256,
        "claim_boundary": (
            "decoded from the one canonical BlockLDL-composed "
            "gridbook.trellis.wire.v1 payload"
        ),
    })
    factor = {
        "algorithm": "pinned_qtip_block_LDL_reverse_feedback",
        "block_size": TRELLIS_FEEDBACK_BLOCK_SIZE,
        "buffer_blocks": int(buffer_blocks),
        "feedback_lower_sha256": _tensor_sha256(feedback_lower),
        "diagonal_blocks_sha256": _tensor_sha256(diagonal_blocks),
        "factorization_max_abs_error": factor_max_abs,
        "hessian_symmetrization": (
            "fp32 (H+H.T)/2 after max asymmetry <= 1e-5"
        ),
        "buffered_oracle_target_max_abs_error": target_max_abs,
        "quadratic_decomposition_abs_error": proxy_abs,
        "full_cross_block_feedback_matrix_consumed": block_count > 1,
        "cross_block_feedback_nonzero_count": int(
            torch.count_nonzero(feedback_lower).item()
        ),
        "terminal_dense_D_consumed": False,
        "terminal_dense_D_exact": False,
        "terminal_metric_mode": terminal_metric_mode,
        "atomic_terminal_geometry": "one_output_row_by_256_input_columns",
        "qtip_16_by_16_terminal_geometry_claimed": False,
    }
    wire_recipe = {
        "schema": TRELLIS_WIRE_SCHEMA,
        "family": E2M1_FAMILY,
        "body_rate_q256": int(body_rate_q256),
        "layout": layout,
        "schedule": [int(value) for value in schedule],
        "alphabets": {
            str(rate): [int(code) for code in codes]
            for rate, codes in sorted(alphabets.items())
        },
        "scale_rule": scale_rule,
        "global_scale_real": shared_global,
        "global_scale_selection": "pre_feedback_transformed_weight_static_6",
        "encoder_source_sha256": encoder_source_sha256(),
        "backend": backend,
        "point_route": point_route,
        "sb_chunk": int(sb_chunk),
        "determinism_mode": determinism_mode,
        "tailbite_candidates": int(tailbite_candidates),
    }
    receipt_body: dict[str, object] = {
        "schema": BLOCKLDL_COMBINED_ARTIFACT_SCHEMA,
        "status": "blockldl_feedback_physical_wire_and_algebra_verified",
        "scope": "research_only_one_linear_unregistered",
        "research_opt_in": RESEARCH_OPT_IN,
        "shape": {"rows": rows, "columns": columns},
        "online_transform": contract,
        "prepared_receipt_identity_sha256": prepared.receipt["identity_sha256"],
        "prepared_receipt_status": prepared_receipt["status"],
        "block_ldl": factor,
        "terminal_blocks": [block_receipts[index] for index in range(block_count)],
        "wire_recipe": wire_recipe,
        "wire_recipe_identity_sha256": _canonical_sha256(wire_recipe),
        "wire_bytes": len(blob),
        "wire_identity_sha256": wire_sha256,
        "decoded_codes_sha256": _tensor_sha256(decoded_codes),
        "decoded_weight_sha256": _tensor_sha256(decoded_weight),
        "same_byte_reparse_verified": True,
        "serve_algebra": serve,
        "qtip_bitshift_wire_allowed": False,
        "format_registry_entries_created": 0,
        "runtime_pin_changed": False,
        "production_contract_changed": False,
        "producer_eligible": False,
    }
    receipt = {
        **receipt_body,
        "identity_sha256": _canonical_sha256(receipt_body),
    }
    return CombinedOneLinearArtifact(
        wire_bytes=blob,
        decoded_transformed_weight=decoded_weight,
        decoded_codes=decoded_codes,
        online_transform=contract,
        receipt=receipt,
    )


__all__ = [
    "BLOCKLDL_COMBINED_ARTIFACT_SCHEMA",
    "CombinedOneLinearArtifact",
    "COMBINED_ARTIFACT_SCHEMA",
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
    "qtip_block_ldl_factors",
    "require_blockldl_trellis_wire_round_trip",
    "require_combined_wire_round_trip",
    "reverse_block_feedback_buffered",
    "reverse_block_feedback_reference",
    "seeded_sign_digest",
    "seeded_signs",
    "transform_weight_and_hessian",
    "transformed_activations",
    "validate_online_transform",
    "verify_post_decode_serve_algebra",
]
