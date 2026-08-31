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
import math
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, cast

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
    require_encoder_source_unchanged as require_encoder_module_unchanged,
    snap_e2m1_scale_codes,
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
from prismaquant.trellis_scale_grid import (
    propose_e2m1_scale_plane,
    require_scale_grid_source_unchanged as require_scale_grid_module_unchanged,
    scale_grid_source_sha256,
)


SCAFFOLD_SCHEMA = (
    "prismaquant.research.qtip_trellis_online_hadamard_one_linear.v1"
)
COMBINED_ARTIFACT_SCHEMA = (
    "prismaquant.research.qtip_trellis_online_hadamard_artifact.v1"
)
BLOCKLDL_COMBINED_ARTIFACT_SCHEMA = (
    "prismaquant.research.qtip_blockldl_trellis_hadamard_artifact.v2"
)
DIAGONAL_HESSIAN_SCAFFOLD_SCHEMA = (
    "prismaquant.research.qtip_trellis_online_hadamard_diagonal_hessian.v1"
)
TRELLIS_FEEDBACK_BLOCK_SIZE = 256
RESEARCH_OPT_IN = "qtip_trellis_online_hadamard_one_linear_v1"
TRANSFORM_SCHEMA = "gridbook.qtip-online-hadamard.v1"
TRANSFORM_ALGORITHM = "block_walsh_hadamard"
TRANSFORM_NORMALIZATION = "orthonormal"
TRANSFORM_PADDING = "none"
SIGN_GENERATOR = "sha256_counter_rademacher"
QTIP_REPOSITORY = "https://github.com/Cornell-RelaxML/qtip"
QTIP_PINNED_COMMIT = "e90c6688c8dfae326a3a81b5eb032db7c6680ec0"
QTIP_SOURCE_FILES = {
    "lib/algo/finetune.py": (
        "0a1021d9bffa3e6a1a86f537096a072779c759417b87624f1eef669a1df2c1a4"
    ),
    "lib/algo/ldlq.py": (
        "793e364fbe91e5b28740d0fc81a6e8618daa6a6a8ce5adbf9b877ba2e46e5bbe"
    ),
    "lib/codebook/bitshift.py": (
        "a299ae97d2ccc80a142095c3c16ed619b435b68736fd52702ab396bc37218531"
    ),
    "lib/utils/math_utils.py": (
        "65d50936e87b2c266806de201dea89b2d74a2ed38e33ef462bd8c3aafb333844"
    ),
}

_ROOT_PAYLOAD_FIELDS = frozenset({
    "schema", "algorithm", "normalization", "padding", "input", "output",
})
_ROOT_FIELDS = _ROOT_PAYLOAD_FIELDS | {"transform_sha256"}
_SIDE_FIELDS = frozenset({
    "dimension", "block_size", "seed", "sign_generator", "sign_sha256",
})
_SIGN_DOMAIN = (TRANSFORM_SCHEMA + "/signs\0").encode("ascii")
_PREPARED_ROOT_FIELDS = frozenset({
    "schema",
    "status",
    "scope",
    "research_opt_in",
    "shape",
    "source",
    "transformed",
    "basis",
    "online_transform",
    "wire",
    "wire_seam",
    "format_registry_entries_created",
    "runtime_pin_changed",
    "production_contract_changed",
    "producer_eligible",
})
_PREPARED_SOURCE_FIELDS = frozenset({"authority", "weight", "hessian"})
_PREPARED_SOURCE_KINDS = frozenset({"weight", "hessian"})
_PREPARED_SOURCE_AUTHORITY_FIELDS = frozenset({
    "status", "reauthenticated_at_encode", "reason",
})
_PREPARED_TENSOR_IDENTITY_FIELDS = frozenset({"dtype", "sha256"})
_PREPARED_TRANSFORMED_KINDS = frozenset({"weight", "hessian"})
_PREPARED_SHAPE_FIELDS = frozenset({"rows", "columns"})
_PREPARED_BASIS = {
    "weight": "R_out W R_in.T",
    "hessian": "R_in H R_in.T",
    "row_input": "x D_in H_in",
    "row_output_inverse": "y H_out D_out",
    "weight_dtype": "torch.float32",
    "hessian_dtype": "torch.float32",
}
_PREPARED_WIRE_FIELDS = frozenset({
    "schema",
    "family",
    "body_rate_q256",
    "terminal_grid",
    "scale_contract",
    "qtip_bitshift_wire_allowed",
    "wire_bytes",
    "wire_identity_sha256",
    "encoder_invoked",
    "decoder_invoked",
})
_PREPARED_WIRE_SEAM_FIELDS = frozenset({
    "available_repository_api", "excluded_substitutions",
})
_PREPARED_AVAILABLE_REPOSITORY_API = [
    "tail-biting Viterbi path planes",
    "gridbook.trellis.wire.v1 immutable byte packer",
    "same-byte canonical parser and reference decoder",
]
_PREPARED_EXCLUDED_SUBSTITUTIONS = [
    "QTIP bitshift wire",
    "vendored or imported Gridbook runtime",
    "unparsed caller-asserted decoded weights",
]
_PREPARED_SOURCE_AUTHORITY = {
    "status": "preparation_time_provenance_only",
    "reauthenticated_at_encode": False,
    "reason": (
        "original source tensors are not retained; transformed tensor "
        "identities are reauthenticated at encode"
    ),
}
_PRODUCER_SOURCE_PATH = Path(__file__).resolve()
_IMPORTED_PRODUCER_SOURCE_SHA256 = hashlib.sha256(
    _PRODUCER_SOURCE_PATH.read_bytes()
).hexdigest()
_IMPORTED_ENCODER_SOURCE_SHA256 = encoder_source_sha256()
_IMPORTED_SCALE_GRID_SOURCE_SHA256 = scale_grid_source_sha256()


@dataclass(frozen=True)
class PreparedOneLinear:
    """Basis-transformed inputs plus an unregistered, non-artifact receipt."""

    transformed_weight: torch.Tensor
    transformed_hessian: torch.Tensor
    online_transform: Mapping[str, object]
    receipt: Mapping[str, object]


@dataclass(frozen=True)
class PreparedDiagonalHessianOneLinear:
    """Transformed weight plus an exact retained diagonal-H contract.

    The input transform is block diagonal.  A diagonal source Hessian
    therefore becomes an exact block-diagonal transformed Hessian without a
    dense ``K x K`` allocation.  The source diagonal remains live so the
    encode boundary can reauthenticate and derive every block itself.
    """

    transformed_weight: torch.Tensor
    source_hessian_diagonal: torch.Tensor
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


@dataclass(frozen=True)
class BlockLDLFactorGroup:
    """One independently factorable block of an exact block-diagonal H."""

    first_column: int
    last_column_exclusive: int
    transformed_hessian: torch.Tensor
    feedback_lower: torch.Tensor
    diagonal_blocks: torch.Tensor


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def producer_source_sha256() -> str:
    """Return the producer source identity captured once at module import."""

    return _IMPORTED_PRODUCER_SOURCE_SHA256


def _current_producer_source_sha256() -> str:
    return hashlib.sha256(_PRODUCER_SOURCE_PATH.read_bytes()).hexdigest()


def _require_producer_source_unchanged() -> str:
    current = _current_producer_source_sha256()
    if not hmac.compare_digest(current, _IMPORTED_PRODUCER_SOURCE_SHA256):
        raise ValueError(
            "BlockLDL producer source changed since module import; refusing "
            "to publish a receipt for a mixed code closure"
        )
    return _IMPORTED_PRODUCER_SOURCE_SHA256


def _require_encoder_source_unchanged() -> str:
    try:
        module_identity = require_encoder_module_unchanged()
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    current = encoder_source_sha256()
    if (
        not hmac.compare_digest(current, module_identity)
        or not hmac.compare_digest(current, _IMPORTED_ENCODER_SOURCE_SHA256)
    ):
        raise ValueError(
            "trellis encoder source changed since module import; refusing "
            "to publish a receipt for a mixed code closure"
        )
    return _IMPORTED_ENCODER_SOURCE_SHA256


def _require_scale_grid_source_unchanged() -> str:
    module_identity = require_scale_grid_module_unchanged()
    current = scale_grid_source_sha256()
    if (
        not hmac.compare_digest(current, module_identity)
        or not hmac.compare_digest(current, _IMPORTED_SCALE_GRID_SOURCE_SHA256)
    ):
        raise ValueError(
            "trellis scale-grid source changed since module import; refusing "
            "to execute or bind a mixed selector closure"
        )
    return _IMPORTED_SCALE_GRID_SOURCE_SHA256


def _require_implementation_sources_unchanged() -> tuple[str, str]:
    return (
        _require_producer_source_unchanged(),
        _require_encoder_source_unchanged(),
    )


def _checked_blockldl_render_inputs(
    *,
    body_rate_q256: object,
    schedule: object,
    layout: object,
    alphabets: object,
    scale_rule: object,
    sb_chunk: object,
    determinism_mode: object,
    tailbite_candidates: object,
    backend: object,
    point_route: object,
    terminal_metric_mode: object,
    buffer_blocks: object,
    research_opt_in: object,
    scale_grid_selection_scope: object,
) -> dict[str, object]:
    for name, value in {
        "body_rate_q256": body_rate_q256,
        "sb_chunk": sb_chunk,
        "tailbite_candidates": tailbite_candidates,
        "buffer_blocks": buffer_blocks,
    }.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"Arm E {name} must be a positive plain int")
    for name, value in {
        "layout": layout,
        "scale_rule": scale_rule,
        "determinism_mode": determinism_mode,
        "backend": backend,
        "point_route": point_route,
        "terminal_metric_mode": terminal_metric_mode,
        "research_opt_in": research_opt_in,
    }.items():
        if type(value) is not str or not value:
            raise ValueError(f"Arm E {name} must be a nonempty plain string")
    if (
        scale_grid_selection_scope is not None
        and (
            type(scale_grid_selection_scope) is not str
            or not scale_grid_selection_scope
        )
    ):
        raise ValueError(
            "Arm E scale_grid_selection_scope must be a plain string or None"
        )
    try:
        frozen_schedule = tuple(schedule)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("Arm E schedule must be an integer sequence") from exc
    if not frozen_schedule or any(type(value) is not int for value in frozen_schedule):
        raise ValueError("Arm E schedule must contain only plain ints")
    if not isinstance(alphabets, Mapping):
        raise ValueError("Arm E alphabets must be a mapping")
    frozen_alphabets: dict[int, tuple[int, ...]] = {}
    for rate, codes in dict(alphabets).items():
        if type(rate) is not int:
            raise ValueError("Arm E alphabet rates must be plain ints")
        try:
            frozen_codes = tuple(codes)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Arm E alphabet values must be integer sequences"
            ) from exc
        if not frozen_codes or any(type(code) is not int for code in frozen_codes):
            raise ValueError(
                "Arm E alphabet values must contain only plain ints"
            )
        frozen_alphabets[rate] = frozen_codes
    return {
        "body_rate_q256": body_rate_q256,
        "schedule": frozen_schedule,
        "layout": layout,
        "alphabets": frozen_alphabets,
        "scale_rule": scale_rule,
        "sb_chunk": sb_chunk,
        "determinism_mode": determinism_mode,
        "tailbite_candidates": tailbite_candidates,
        "backend": backend,
        "point_route": point_route,
        "terminal_metric_mode": terminal_metric_mode,
        "buffer_blocks": buffer_blocks,
        "research_opt_in": research_opt_in,
        "scale_grid_selection_scope": scale_grid_selection_scope,
    }


def _require_exact_fields(
    value: Any,
    expected: frozenset[str],
    *,
    where: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"{where} has missing={missing}, unknown={unknown}")
    return value


def _require_sha256(value: Any, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{where} must be lowercase SHA-256")
    return value


def _json_exact_equal(actual: Any, expected: Any) -> bool:
    """Compare closed JSON semantics without Python's bool/int aliasing."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _json_exact_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_exact_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


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


def transform_weight(
    weight: torch.Tensor,
    online_transform: Mapping[str, object],
) -> torch.Tensor:
    """Return ``R_out W R_in.T`` without constructing any Hessian."""

    if weight.ndim != 2:
        raise ValueError("weight must be rank two")
    rows, columns = map(int, weight.shape)
    contract = validate_online_transform(
        online_transform, rows=rows, columns=columns
    )
    value = weight.float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("weight must be finite")
    weight_right = _input_basis_rows(value, contract, "input")
    return _input_basis_rows(
        weight_right.T, contract, "output"
    ).T.contiguous()


def _validated_positive_hessian_diagonal(
    diagonal: torch.Tensor,
    *,
    dimension: int,
) -> torch.Tensor:
    # Refuse shape before detach/dtype conversion.  In particular, a forged
    # K-by-K BF16 tensor must not allocate a global FP32 K-by-K temporary on
    # the structured path that exists specifically to avoid that geometry.
    if diagonal.ndim != 1 or int(diagonal.numel()) != dimension:
        raise ValueError(
            "structured Hessian source must be one rank-one diagonal vector; "
            "dense or off-diagonal source matrices are not accepted"
        )
    value = diagonal.detach().float()
    if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
        raise ValueError(
            "structured Hessian diagonal must be finite and strictly positive"
        )
    return value.contiguous()


def transformed_diagonal_hessian_block(
    source_diagonal_block: torch.Tensor,
) -> torch.Tensor:
    """Construct exactly ``H diag(d) H`` for one Sylvester block.

    Input-side Rademacher signs cancel on a diagonal source Hessian.  This
    helper intentionally accepts only a rank-one diagonal vector, making an
    off-block or otherwise dense source impossible to smuggle into the
    structured producer.
    """

    if source_diagonal_block.ndim != 1:
        raise ValueError(
            "source_diagonal_block must be rank one; off-block input refused"
        )
    block_size = int(source_diagonal_block.numel())
    if block_size < 1 or block_size & (block_size - 1):
        raise ValueError("diagonal-H transform block must be a positive power of two")
    diagonal = _validated_positive_hessian_diagonal(
        source_diagonal_block, dimension=block_size
    )
    dense_block = torch.diag(diagonal)
    right = _normalized_block_hadamard_rows(dense_block, block_size)
    transformed = _normalized_block_hadamard_rows(
        right.T.contiguous(), block_size
    ).T.contiguous()
    return ((transformed + transformed.T) * 0.5).contiguous()


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
            "authority": dict(_PREPARED_SOURCE_AUTHORITY),
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


def _diagonal_hessian_structure(
    diagonal: torch.Tensor,
    contract: Mapping[str, object],
) -> dict[str, object]:
    input_spec = _side_spec(contract, "input")
    dimension = int(input_spec["dimension"])
    block_size = int(input_spec["block_size"])
    value = _validated_positive_hessian_diagonal(
        diagonal, dimension=dimension
    )
    body: dict[str, object] = {
        "schema": "prismaquant.research.block_diagonal_hessian_structure.v1",
        "construction": "block_walsh_hadamard_similarity_of_positive_diagonal",
        "dimension": dimension,
        "transform_block_size": block_size,
        "transform_block_count": dimension // block_size,
        "source_diagonal_sha256": _tensor_sha256(value),
        "ordered_blocks": [
            {
                "index": index,
                "first_column": first,
                "last_column_exclusive": first + block_size,
                "columns": block_size,
                "source_diagonal_sha256": _tensor_sha256(
                    value[first:first + block_size]
                ),
            }
            for index, first in enumerate(range(0, dimension, block_size))
        ],
        "input_rademacher_signs_cancel_exactly": True,
        "off_block_entries_zero_by_construction": True,
        "dense_k_by_k_materialized": False,
    }
    return {**body, "identity_sha256": _canonical_sha256(body)}


def prepare_one_linear_diagonal_hessian_scaffold(
    weight: torch.Tensor,
    hessian_diagonal: torch.Tensor,
    *,
    body_rate_q256: int,
    input_block_size: int,
    output_block_size: int,
    input_seed: int,
    output_seed: int,
    research_opt_in: str,
) -> PreparedDiagonalHessianOneLinear:
    """Prepare a diagonal-H Arm E input without ever allocating dense KxK."""

    if research_opt_in != RESEARCH_OPT_IN:
        raise ValueError(f"research_opt_in must equal {RESEARCH_OPT_IN!r}")
    if weight.ndim != 2:
        raise ValueError("weight must be rank two")
    rows, columns = map(int, weight.shape)
    if rows <= 0 or columns <= 0 or columns % 16:
        raise ValueError("E2M1 trellis weight must be nonempty and group-16 aligned")
    if (
        type(input_block_size) is not int
        or input_block_size < TRELLIS_FEEDBACK_BLOCK_SIZE
        or input_block_size % TRELLIS_FEEDBACK_BLOCK_SIZE
    ):
        raise ValueError(
            "structured input_block_size must be an integer multiple of 256"
        )
    diagonal = _validated_positive_hessian_diagonal(
        hessian_diagonal, dimension=columns
    )
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
    transformed_weight = transform_weight(weight, contract)
    structure = _diagonal_hessian_structure(diagonal, contract)
    source_authority = {
        "status": "retained_positive_diagonal_reauthenticated_at_encode",
        "weight_reauthenticated_at_encode": False,
        "hessian_diagonal_reauthenticated_at_encode": True,
        "reason": (
            "the original weight is represented by the transformed-weight "
            "identity; the retained diagonal is hashed again before every "
            "structured factorization"
        ),
    }
    basis = {
        **_PREPARED_BASIS,
        "hessian_source": "diag(retained_positive_diagonal)",
        "hessian_transformed_representation": (
            "exact_block_diagonal_without_dense_k_by_k_materialization"
        ),
    }
    receipt_body: dict[str, object] = {
        "schema": DIAGONAL_HESSIAN_SCAFFOLD_SCHEMA,
        "status": "prepared_exact_block_diagonal_hessian_wire_seam_available",
        "scope": "research_only_one_linear_unregistered_contract_scaffold",
        "research_opt_in": RESEARCH_OPT_IN,
        "shape": {"rows": rows, "columns": columns},
        "source": {
            "authority": source_authority,
            "weight": {
                "dtype": str(weight.dtype),
                "sha256": _tensor_sha256(weight),
            },
            "hessian_diagonal": {
                "dtype": str(diagonal.dtype),
                "sha256": _tensor_sha256(diagonal),
            },
        },
        "transformed": {
            "weight": {
                "dtype": str(transformed_weight.dtype),
                "sha256": _tensor_sha256(transformed_weight),
            },
            "hessian_structure": structure,
        },
        "basis": basis,
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
            "available_repository_api": _PREPARED_AVAILABLE_REPOSITORY_API,
            "excluded_substitutions": _PREPARED_EXCLUDED_SUBSTITUTIONS,
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
    return PreparedDiagonalHessianOneLinear(
        transformed_weight=transformed_weight,
        source_hessian_diagonal=diagonal,
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
    _require_sha256(identity, where="prepared receipt identity_sha256")
    if not hmac.compare_digest(identity, _canonical_sha256(body)):
        raise ValueError("prepared receipt identity mismatch")
    _require_exact_fields(
        body, _PREPARED_ROOT_FIELDS, where="prepared receipt",
    )
    expected_root_constants = {
        "schema": SCAFFOLD_SCHEMA,
        "status": "prepared_exact_trellis_wire_seam_available",
        "scope": "research_only_one_linear_unregistered_contract_scaffold",
        "research_opt_in": RESEARCH_OPT_IN,
        "format_registry_entries_created": 0,
        "runtime_pin_changed": False,
        "production_contract_changed": False,
        "producer_eligible": False,
    }
    for field, expected_value in expected_root_constants.items():
        if not _json_exact_equal(body[field], expected_value):
            raise ValueError(f"prepared receipt {field} mismatch")
    rows, columns = map(int, prepared.transformed_weight.shape)
    _require_exact_fields(
        body["shape"], _PREPARED_SHAPE_FIELDS, where="prepared receipt shape",
    )
    if not _json_exact_equal(
        body["shape"], {"rows": rows, "columns": columns}
    ):
        raise ValueError("prepared receipt shape mismatch")
    source = _require_exact_fields(
        body["source"],
        _PREPARED_SOURCE_FIELDS,
        where="prepared receipt source",
    )
    _require_exact_fields(
        source["authority"],
        _PREPARED_SOURCE_AUTHORITY_FIELDS,
        where="prepared receipt source.authority",
    )
    if not _json_exact_equal(
        source["authority"], _PREPARED_SOURCE_AUTHORITY
    ):
        raise ValueError("prepared receipt source.authority mismatch")
    for kind in sorted(_PREPARED_SOURCE_KINDS):
        source_identity = _require_exact_fields(
            source[kind],
            _PREPARED_TENSOR_IDENTITY_FIELDS,
            where=f"prepared receipt source.{kind}",
        )
        dtype_text = source_identity["dtype"]
        dtype_value = (
            getattr(torch, dtype_text.removeprefix("torch."), None)
            if isinstance(dtype_text, str) and dtype_text.startswith("torch.")
            else None
        )
        if not isinstance(dtype_value, torch.dtype):
            raise ValueError(
                f"prepared receipt source.{kind}.dtype must be a torch dtype"
            )
        _require_sha256(
            source_identity["sha256"],
            where=f"prepared receipt source.{kind}.sha256",
        )
    transformed = _require_exact_fields(
        body["transformed"],
        _PREPARED_TRANSFORMED_KINDS,
        where="prepared receipt transformed",
    )
    for kind in sorted(_PREPARED_TRANSFORMED_KINDS):
        _require_exact_fields(
            transformed[kind],
            _PREPARED_TENSOR_IDENTITY_FIELDS,
            where=f"prepared receipt transformed.{kind}",
        )
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
    if not _json_exact_equal(body["basis"], _PREPARED_BASIS):
        raise ValueError("prepared receipt basis mismatch")
    wire = _require_exact_fields(
        body["wire"], _PREPARED_WIRE_FIELDS, where="prepared receipt wire",
    )
    family = get_trellis_family(E2M1_FAMILY)
    expected_wire = {
        "schema": TRELLIS_WIRE_SCHEMA,
        "family": E2M1_FAMILY,
        "body_rate_q256": body_rate_q256,
        "terminal_grid": "E2M1",
        "scale_contract": family.scale_contract,
        "qtip_bitshift_wire_allowed": False,
        "wire_bytes": None,
        "wire_identity_sha256": None,
        "encoder_invoked": False,
        "decoder_invoked": False,
    }
    if not _json_exact_equal(wire, expected_wire):
        raise ValueError("prepared receipt wire contract mismatch")
    wire_seam = _require_exact_fields(
        body["wire_seam"],
        _PREPARED_WIRE_SEAM_FIELDS,
        where="prepared receipt wire_seam",
    )
    if not _json_exact_equal(
        wire_seam,
        {
            "available_repository_api": _PREPARED_AVAILABLE_REPOSITORY_API,
            "excluded_substitutions": _PREPARED_EXCLUDED_SUBSTITUTIONS,
        },
    ):
        raise ValueError("prepared receipt wire_seam mismatch")
    contract = validate_online_transform(
        body.get("online_transform"), rows=rows, columns=columns
    )
    if contract != validate_online_transform(
        prepared.online_transform, rows=rows, columns=columns
    ):
        raise ValueError("prepared online-transform metadata mismatch")
    return body


def _validate_prepared_diagonal_hessian_one_linear(
    prepared: PreparedDiagonalHessianOneLinear,
    *,
    body_rate_q256: int,
) -> dict[str, object]:
    """Reauthenticate the retained diagonal and its exact structure claim."""

    if not isinstance(prepared, PreparedDiagonalHessianOneLinear):
        raise ValueError(
            "prepared must be a PreparedDiagonalHessianOneLinear"
        )
    if not isinstance(prepared.receipt, Mapping):
        raise ValueError("prepared receipt must be an object")
    body = dict(prepared.receipt)
    identity = body.pop("identity_sha256", None)
    _require_sha256(identity, where="prepared receipt identity_sha256")
    if not hmac.compare_digest(identity, _canonical_sha256(body)):
        raise ValueError("prepared receipt identity mismatch")
    _require_exact_fields(body, _PREPARED_ROOT_FIELDS, where="prepared receipt")
    expected_root = {
        "schema": DIAGONAL_HESSIAN_SCAFFOLD_SCHEMA,
        "status": "prepared_exact_block_diagonal_hessian_wire_seam_available",
        "scope": "research_only_one_linear_unregistered_contract_scaffold",
        "research_opt_in": RESEARCH_OPT_IN,
        "format_registry_entries_created": 0,
        "runtime_pin_changed": False,
        "production_contract_changed": False,
        "producer_eligible": False,
    }
    for field, expected in expected_root.items():
        if not _json_exact_equal(body[field], expected):
            raise ValueError(f"prepared receipt {field} mismatch")

    if prepared.transformed_weight.ndim != 2:
        raise ValueError("prepared transformed weight must be rank two")
    rows, columns = map(int, prepared.transformed_weight.shape)
    shape = _require_exact_fields(
        body["shape"], _PREPARED_SHAPE_FIELDS, where="prepared receipt shape"
    )
    if not _json_exact_equal(shape, {"rows": rows, "columns": columns}):
        raise ValueError("prepared receipt shape mismatch")
    diagonal = _validated_positive_hessian_diagonal(
        prepared.source_hessian_diagonal, dimension=columns
    )
    contract = validate_online_transform(
        body.get("online_transform"), rows=rows, columns=columns
    )
    input_block_size = int(_side_spec(contract, "input")["block_size"])
    if (
        input_block_size < TRELLIS_FEEDBACK_BLOCK_SIZE
        or input_block_size % TRELLIS_FEEDBACK_BLOCK_SIZE
    ):
        raise ValueError(
            "prepared structured input transform block must be 256-aligned"
        )
    if contract != validate_online_transform(
        prepared.online_transform, rows=rows, columns=columns
    ):
        raise ValueError("prepared online-transform metadata mismatch")

    source = _require_exact_fields(
        body["source"],
        frozenset({"authority", "weight", "hessian_diagonal"}),
        where="prepared receipt source",
    )
    expected_authority = {
        "status": "retained_positive_diagonal_reauthenticated_at_encode",
        "weight_reauthenticated_at_encode": False,
        "hessian_diagonal_reauthenticated_at_encode": True,
        "reason": (
            "the original weight is represented by the transformed-weight "
            "identity; the retained diagonal is hashed again before every "
            "structured factorization"
        ),
    }
    if not _json_exact_equal(source["authority"], expected_authority):
        raise ValueError("prepared receipt source.authority mismatch")
    source_weight = _require_exact_fields(
        source["weight"],
        _PREPARED_TENSOR_IDENTITY_FIELDS,
        where="prepared receipt source.weight",
    )
    source_dtype = source_weight["dtype"]
    if not (
        isinstance(source_dtype, str)
        and source_dtype.startswith("torch.")
        and isinstance(
            getattr(torch, source_dtype.removeprefix("torch."), None),
            torch.dtype,
        )
    ):
        raise ValueError("prepared receipt source.weight.dtype must be a torch dtype")
    _require_sha256(
        source_weight["sha256"], where="prepared receipt source.weight.sha256"
    )
    expected_diagonal = {
        "dtype": str(diagonal.dtype),
        "sha256": _tensor_sha256(diagonal),
    }
    if not _json_exact_equal(source["hessian_diagonal"], expected_diagonal):
        raise ValueError("prepared retained Hessian diagonal identity mismatch")

    transformed = _require_exact_fields(
        body["transformed"],
        frozenset({"weight", "hessian_structure"}),
        where="prepared receipt transformed",
    )
    expected_weight = {
        "dtype": str(prepared.transformed_weight.dtype),
        "sha256": _tensor_sha256(prepared.transformed_weight),
    }
    if not _json_exact_equal(transformed["weight"], expected_weight):
        raise ValueError("prepared transformed weight identity mismatch")
    expected_structure = _diagonal_hessian_structure(diagonal, contract)
    if not _json_exact_equal(
        transformed["hessian_structure"], expected_structure
    ):
        raise ValueError("prepared transformed Hessian structure mismatch")
    expected_basis = {
        **_PREPARED_BASIS,
        "hessian_source": "diag(retained_positive_diagonal)",
        "hessian_transformed_representation": (
            "exact_block_diagonal_without_dense_k_by_k_materialization"
        ),
    }
    if not _json_exact_equal(body["basis"], expected_basis):
        raise ValueError("prepared receipt basis mismatch")

    wire = _require_exact_fields(
        body["wire"], _PREPARED_WIRE_FIELDS, where="prepared receipt wire"
    )
    family = get_trellis_family(E2M1_FAMILY)
    expected_wire = {
        "schema": TRELLIS_WIRE_SCHEMA,
        "family": E2M1_FAMILY,
        "body_rate_q256": body_rate_q256,
        "terminal_grid": "E2M1",
        "scale_contract": family.scale_contract,
        "qtip_bitshift_wire_allowed": False,
        "wire_bytes": None,
        "wire_identity_sha256": None,
        "encoder_invoked": False,
        "decoder_invoked": False,
    }
    if not _json_exact_equal(wire, expected_wire):
        raise ValueError("prepared receipt wire contract mismatch")
    wire_seam = _require_exact_fields(
        body["wire_seam"],
        _PREPARED_WIRE_SEAM_FIELDS,
        where="prepared receipt wire_seam",
    )
    if not _json_exact_equal(
        wire_seam,
        {
            "available_repository_api": _PREPARED_AVAILABLE_REPOSITORY_API,
            "excluded_substitutions": _PREPARED_EXCLUDED_SUBSTITUTIONS,
        },
    ):
        raise ValueError("prepared receipt wire_seam mismatch")
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


def iter_transformed_diagonal_block_ldl_factors(
    source_diagonal: torch.Tensor,
    online_transform: Mapping[str, object],
    *,
    block_size: int = TRELLIS_FEEDBACK_BLOCK_SIZE,
) -> Iterator[BlockLDLFactorGroup]:
    """Yield exact factor groups without constructing a dense global Hessian.

    The source contract is deliberately narrower than arbitrary block-sparse
    input: only a retained positive diagonal plus the already validated
    block-local orthogonal transform is accepted.  Thus off-block zeros are a
    theorem of the construction rather than an unchecked caller assertion.
    """

    input_spec = _side_spec(online_transform, "input")
    dimension = int(input_spec["dimension"])
    transform_block_size = int(input_spec["block_size"])
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if transform_block_size % block_size:
        raise ValueError(
            "input transform block must be divisible by the feedback block"
        )
    diagonal = _validated_positive_hessian_diagonal(
        source_diagonal, dimension=dimension
    )
    expected_diagonal_sha256 = _tensor_sha256(diagonal)
    for first in range(0, dimension, transform_block_size):
        if not hmac.compare_digest(
            _tensor_sha256(diagonal), expected_diagonal_sha256
        ):
            raise ValueError(
                "structured Hessian diagonal changed during factorization"
            )
        last = first + transform_block_size
        transformed_hessian = transformed_diagonal_hessian_block(
            diagonal[first:last]
        )
        feedback_lower, diagonal_blocks = qtip_block_ldl_factors(
            transformed_hessian, block_size=block_size
        )
        yield BlockLDLFactorGroup(
            first_column=first,
            last_column_exclusive=last,
            transformed_hessian=transformed_hessian,
            feedback_lower=feedback_lower,
            diagonal_blocks=diagonal_blocks,
        )
    if not hmac.compare_digest(
        _tensor_sha256(diagonal), expected_diagonal_sha256
    ):
        raise ValueError("structured Hessian diagonal changed during factorization")


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
    prepared: PreparedOneLinear | PreparedDiagonalHessianOneLinear,
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
    scale_grid_multipliers: Sequence[float] | None = None,
    scale_grid_selection_scope: str | None = None,
) -> CombinedOneLinearArtifact:
    """Run 256-column reverse BlockLDL feedback with exact trellis terminals.

    This is a QTIP-derived recurrence, not QTIP's 16-by-16 codebook geometry.
    Each Gridbook physical row-by-256 tail-biting cycle is one indivisible LDL
    block.  A single precommitted tensor-global scale is shared by every block
    so the resulting planes can be represented by one canonical wire.
    """

    checked = _checked_blockldl_render_inputs(
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
        terminal_metric_mode=terminal_metric_mode,
        buffer_blocks=buffer_blocks,
        research_opt_in=research_opt_in,
        scale_grid_selection_scope=scale_grid_selection_scope,
    )
    body_rate_q256 = cast(int, checked["body_rate_q256"])
    schedule = cast(tuple[int, ...], checked["schedule"])
    layout = cast(str, checked["layout"])
    alphabets = cast(dict[int, tuple[int, ...]], checked["alphabets"])
    scale_rule = cast(str, checked["scale_rule"])
    sb_chunk = cast(int, checked["sb_chunk"])
    determinism_mode = cast(str, checked["determinism_mode"])
    tailbite_candidates = cast(int, checked["tailbite_candidates"])
    backend = cast(str, checked["backend"])
    point_route = cast(str, checked["point_route"])
    terminal_metric_mode = cast(str, checked["terminal_metric_mode"])
    buffer_blocks = cast(int, checked["buffer_blocks"])
    research_opt_in = cast(str, checked["research_opt_in"])
    scale_grid_selection_scope = cast(
        str | None, checked["scale_grid_selection_scope"]
    )
    if research_opt_in != RESEARCH_OPT_IN:
        raise ValueError(f"research_opt_in must equal {RESEARCH_OPT_IN!r}")
    scale_grid_enabled = scale_grid_multipliers is not None
    if scale_grid_enabled:
        scale_grid_raw = tuple(scale_grid_multipliers)
        if any(
            type(value) not in {int, float}
            for value in scale_grid_raw
        ):
            raise ValueError(
                "Arm E scale-grid multipliers must be plain numeric values"
            )
        scale_grid_menu = tuple(float(value) for value in scale_grid_raw)
        if not scale_grid_menu or scale_grid_menu[0] != 1.0:
            raise ValueError("Arm E scale-grid candidate zero must be exactly 1.0")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in scale_grid_menu
        ):
            raise ValueError("Arm E scale-grid multipliers must be finite/positive")
        if len(set(scale_grid_menu)) != len(scale_grid_menu):
            raise ValueError("Arm E scale-grid multipliers must be unique")
        if scale_grid_selection_scope != "row_factor_group":
            raise ValueError(
                "Arm E scale-grid selection must use row_factor_group; "
                "splicing inside a coupled recurrence is forbidden"
            )
        scale_grid_source_at_start = _require_scale_grid_source_unchanged()
    else:
        scale_grid_menu = None
        scale_grid_source_at_start = None
        if scale_grid_selection_scope is not None:
            raise ValueError(
                "Arm E scale_grid_selection_scope requires an explicit multiplier menu"
            )
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
    structured_diagonal = isinstance(
        prepared, PreparedDiagonalHessianOneLinear
    )
    if structured_diagonal:
        prepared_receipt = _validate_prepared_diagonal_hessian_one_linear(
            prepared, body_rate_q256=body_rate_q256
        )
    else:
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
    if not structured_diagonal and prepared.transformed_hessian.shape != (
        columns, columns
    ):
        raise ValueError("prepared transformed Hessian has the wrong shape")

    group_scales = weight.reshape(rows, columns // 16, 16).abs().amax(-1)
    group_scales = group_scales.clamp_min(1.0e-12) / 6.0
    shared_global_tensor = (
        group_scales.amax() / 448.0
    ).clamp_min(1.0e-12).to(torch.float32)
    shared_global = float(shared_global_tensor.item())

    encoded_blocks: dict[int, EncodedTrellisPlanes] = {}
    identity_encoded_blocks: dict[int, EncodedTrellisPlanes] = {}
    block_receipts: dict[int, dict[str, object]] = {}
    block_count = columns // TRELLIS_FEEDBACK_BLOCK_SIZE
    recurrence_q = torch.zeros_like(weight)
    buffered_targets: list[torch.Tensor | None] = [None] * block_count
    factor_group_records: list[dict[str, object]] = []
    target_errors: list[float] = []
    factor_errors: list[float] = []
    proxy_errors: list[float] = []
    feedback_nonzero_count = 0
    decomposed_proxy_total = weight.new_zeros(())
    direct_proxy_total = weight.new_zeros(())

    def factor_groups() -> Iterator[BlockLDLFactorGroup]:
        if structured_diagonal:
            assert isinstance(prepared, PreparedDiagonalHessianOneLinear)
            yield from iter_transformed_diagonal_block_ldl_factors(
                prepared.source_hessian_diagonal,
                contract,
                block_size=TRELLIS_FEEDBACK_BLOCK_SIZE,
            )
            return
        assert isinstance(prepared, PreparedOneLinear)
        dense_hessian = prepared.transformed_hessian.float()
        feedback, diagonal = qtip_block_ldl_factors(
            dense_hessian, block_size=TRELLIS_FEEDBACK_BLOCK_SIZE
        )
        yield BlockLDLFactorGroup(
            first_column=0,
            last_column_exclusive=columns,
            transformed_hessian=dense_hessian,
            feedback_lower=feedback,
            diagonal_blocks=diagonal,
        )

    scale_grid_candidate_win_rows = 0
    scale_grid_total_rows = 0

    for group_index, group in enumerate(factor_groups()):
        group_first = group.first_column
        group_last = group.last_column_exclusive
        group_columns = group_last - group_first
        group_first_block = group_first // TRELLIS_FEEDBACK_BLOCK_SIZE
        group_block_count = group_columns // TRELLIS_FEEDBACK_BLOCK_SIZE

        group_weight = weight[:, group_first:group_last]
        arm_encoded: dict[str, dict[int, EncodedTrellisPlanes]] = {}
        arm_receipts: dict[str, dict[int, dict[str, object]]] = {}

        def run_trajectory(
            arm: str,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], float]:
            if arm not in {"identity", "candidate"}:
                raise AssertionError("unknown BlockLDL scale-grid arm")
            if arm == "candidate" and not scale_grid_enabled:
                raise AssertionError("candidate trajectory requested while grid is off")
            trajectory_encoded: dict[int, EncodedTrellisPlanes] = {}
            trajectory_receipts: dict[int, dict[str, object]] = {}

            def terminal(
                local_block_index: int, target: torch.Tensor
            ) -> torch.Tensor:
                block_index = group_first_block + local_block_index
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
                dense_d = group.diagonal_blocks[local_block_index]
                metric = (
                    dense_d.diagonal().contiguous()
                    if terminal_metric_mode == "diag_block_D"
                    else torch.ones(
                        TRELLIS_FEEDBACK_BLOCK_SIZE,
                        dtype=torch.float32,
                        device=target.device,
                    )
                )
                scale_plane_override = None
                proposal_record: dict[str, object] | None = None
                if arm == "candidate":
                    assert scale_grid_menu is not None
                    target_scales = (
                        target.float().reshape(rows, 16, 16)
                        .abs().amax(-1).clamp_min(1.0e-12) / 6.0
                    )
                    identity_codes = snap_e2m1_scale_codes(
                        target_scales,
                        shared_global,
                        multiplier=1.0,
                        floor_to_min_positive=True,
                    )
                    proposal = propose_e2m1_scale_plane(
                        target,
                        metric,
                        global_scale_real=shared_global,
                        identity_scale_codes=identity_codes,
                        multipliers=scale_grid_menu,
                        floor_to_min_positive=True,
                    )
                    scale_plane_override = proposal.scale_codes
                    proposal_record = {
                        "multiplier_indices_sha256": _tensor_sha256(
                            proposal.multiplier_indices
                        ),
                        "scale_codes_sha256": _tensor_sha256(
                            proposal.scale_codes
                        ),
                        "masked_candidate_cells": (
                            proposal.masked_candidate_cells
                        ),
                        "clipped_candidate_cells": (
                            proposal.clipped_candidate_cells
                        ),
                        "rtn_floor_nonregression_verified": True,
                    }
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
                    scale_plane_override=scale_plane_override,
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
                target_group_scales = (
                    target.reshape(rows, 16, 16).abs().amax(-1) / 6.0
                )
                clipped = int(
                    (
                        target_group_scales
                        > shared_global_tensor * 448.0
                    ).sum().item()
                )
                trajectory_encoded[block_index] = encoded
                trajectory_receipts[block_index] = {
                    "arm": arm,
                    "block_index": block_index,
                    "factor_group_index": group_index,
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
                    "dense_D_terminal_consumption": {
                        "diagonal_consumed": (
                            terminal_metric_mode == "diag_block_D"
                        ),
                        "off_diagonal_consumed": False,
                        "full_matrix_consumed": False,
                        "exact_dense_objective": False,
                    },
                    "clipped_group_count_at_fixed_global": clipped,
                    "scale_proposal": proposal_record,
                }
                return decoded_terminal

            trajectory_q, trajectory_targets = reverse_block_feedback_buffered(
                group_weight,
                group.feedback_lower,
                terminal,
                block_size=TRELLIS_FEEDBACK_BLOCK_SIZE,
                buffer_blocks=buffer_blocks,
            )

            def decoded_replay(
                local_block_index: int, _target: torch.Tensor
            ) -> torch.Tensor:
                first = local_block_index * TRELLIS_FEEDBACK_BLOCK_SIZE
                return trajectory_q[
                    :, first:first + TRELLIS_FEEDBACK_BLOCK_SIZE
                ]

            oracle_q, oracle_targets = reverse_block_feedback_reference(
                group_weight,
                group.feedback_lower,
                decoded_replay,
                block_size=TRELLIS_FEEDBACK_BLOCK_SIZE,
            )
            if not torch.equal(oracle_q, trajectory_q):
                raise AssertionError(
                    f"{arm} buffered recurrence decoded blocks changed"
                )
            trajectory_target_max_abs = max(
                float((buffered - oracle).abs().max().item())
                for buffered, oracle in zip(
                    trajectory_targets, oracle_targets, strict=True
                )
            )
            if not all(
                torch.allclose(buffered, oracle, rtol=3.0e-5, atol=3.0e-5)
                for buffered, oracle in zip(
                    trajectory_targets, oracle_targets, strict=True
                )
            ):
                raise AssertionError(
                    f"{arm} buffered BlockLDL targets differ from the "
                    f"unbuffered oracle; max_abs={trajectory_target_max_abs}"
                )
            arm_encoded[arm] = trajectory_encoded
            arm_receipts[arm] = trajectory_receipts
            return trajectory_q, trajectory_targets, trajectory_target_max_abs

        identity_q, identity_targets, identity_target_error = run_trajectory(
            "identity"
        )
        if scale_grid_enabled:
            candidate_q, candidate_targets, candidate_target_error = run_trajectory(
                "candidate"
            )
        else:
            candidate_q = identity_q
            candidate_targets = identity_targets
            candidate_target_error = identity_target_error

        def row_proxy_fp64(reconstruction: torch.Tensor) -> torch.Tensor:
            error = group_weight.to(torch.float64) - reconstruction.to(torch.float64)
            hessian_fp64 = group.transformed_hessian.to(torch.float64)
            return ((error @ hessian_fp64) * error).sum(dim=1).contiguous()

        identity_row_proxy = row_proxy_fp64(identity_q)
        candidate_row_proxy = row_proxy_fp64(candidate_q)
        candidate_wins = (
            candidate_row_proxy < identity_row_proxy
            if scale_grid_enabled
            else torch.zeros(rows, dtype=torch.bool, device=weight.device)
        )
        row_mask = candidate_wins.reshape(rows, 1)
        group_q = torch.where(row_mask, candidate_q, identity_q).contiguous()
        group_targets = tuple(
            torch.where(row_mask, candidate, identity).contiguous()
            for identity, candidate in zip(
                identity_targets, candidate_targets, strict=True
            )
        )
        final_row_proxy = row_proxy_fp64(group_q)
        if not torch.equal(
            final_row_proxy,
            torch.minimum(identity_row_proxy, candidate_row_proxy),
        ):
            raise AssertionError(
                "Arm E row-factor-group Cf is not exactly min(C0, C1)"
            )
        if bool((final_row_proxy > identity_row_proxy).any().item()):
            raise AssertionError("Arm E row-factor-group Cf exceeded identity C0")
        if scale_grid_enabled:
            def selected_terminal(
                local_block_index: int, _target: torch.Tensor
            ) -> torch.Tensor:
                first = local_block_index * TRELLIS_FEEDBACK_BLOCK_SIZE
                return group_q[
                    :, first:first + TRELLIS_FEEDBACK_BLOCK_SIZE
                ]

            selected_oracle_q, selected_oracle_targets = (
                reverse_block_feedback_reference(
                    group_weight,
                    group.feedback_lower,
                    selected_terminal,
                    block_size=TRELLIS_FEEDBACK_BLOCK_SIZE,
                )
            )
            if not torch.equal(selected_oracle_q, group_q):
                raise AssertionError(
                    "selected Arm E recurrence changed decoded blocks"
                )
            selected_target_error = max(
                float((selected - oracle).abs().max().item())
                for selected, oracle in zip(
                    group_targets, selected_oracle_targets, strict=True
                )
            )
            if not all(
                torch.allclose(selected, oracle, rtol=3.0e-5, atol=3.0e-5)
                for selected, oracle in zip(
                    group_targets, selected_oracle_targets, strict=True
                )
            ):
                raise AssertionError(
                    "selected Arm E targets differ from the unbuffered "
                    f"recurrence; max_abs={selected_target_error}"
                )
        else:
            selected_target_error = identity_target_error
        scale_grid_candidate_win_rows += int(candidate_wins.sum().item())
        scale_grid_total_rows += rows
        group_target_max_abs = max(
            identity_target_error, candidate_target_error, selected_target_error
        )
        target_errors.append(group_target_max_abs)

        recurrence_q[:, group_first:group_last] = group_q
        for local_index, target in enumerate(group_targets):
            buffered_targets[group_first_block + local_index] = target

        for local_index in range(group_block_count):
            block_index = group_first_block + local_index
            identity_encoded = arm_encoded["identity"][block_index]
            identity_encoded_blocks[block_index] = identity_encoded
            if scale_grid_enabled:
                candidate_encoded = arm_encoded["candidate"][block_index]
                column_mask = row_mask.expand(
                    rows, TRELLIS_FEEDBACK_BLOCK_SIZE
                )
                scale_mask = row_mask.expand(
                    rows, TRELLIS_FEEDBACK_BLOCK_SIZE // 16
                )
                identity_scale = identity_encoded.scale_codes
                candidate_scale = candidate_encoded.scale_codes
                if (
                    identity_scale is None
                    or candidate_scale is None
                    or identity_scale.device != weight.device
                    or candidate_scale.device != weight.device
                    or tuple(identity_scale.shape) != (rows, 16)
                    or tuple(candidate_scale.shape) != (rows, 16)
                ):
                    raise AssertionError(
                        "Arm E terminal lost its resident E2M1 scale-code plane"
                    )
                final_scale = torch.where(
                    scale_mask, candidate_scale, identity_scale
                )
                final_encoded = EncodedTrellisPlanes(
                    reconstruction=group_q[
                        :, local_index * TRELLIS_FEEDBACK_BLOCK_SIZE:
                        (local_index + 1) * TRELLIS_FEEDBACK_BLOCK_SIZE
                    ].contiguous(),
                    u_bits=torch.where(
                        column_mask,
                        candidate_encoded.u_bits,
                        identity_encoded.u_bits,
                    ).contiguous(),
                    point_indices=torch.where(
                        column_mask,
                        candidate_encoded.point_indices,
                        identity_encoded.point_indices,
                    ).contiguous(),
                    bypass_codes=torch.where(
                        column_mask,
                        candidate_encoded.bypass_codes,
                        identity_encoded.bypass_codes,
                    ).contiguous(),
                    scale_blob=(
                        final_scale.detach().cpu().contiguous().numpy().tobytes()
                    ),
                    global_scale_real=identity_encoded.global_scale_real,
                    scale_codes=final_scale.contiguous(),
                )
            else:
                final_encoded = identity_encoded
            encoded_blocks[block_index] = final_encoded

            identity_receipt = arm_receipts["identity"][block_index]
            if not scale_grid_enabled:
                legacy_receipt = dict(identity_receipt)
                legacy_receipt.pop("arm")
                legacy_receipt.pop("scale_proposal")
                block_receipts[block_index] = legacy_receipt
                continue
            candidate_receipt = arm_receipts["candidate"][block_index]
            first = block_index * TRELLIS_FEEDBACK_BLOCK_SIZE
            last = first + TRELLIS_FEEDBACK_BLOCK_SIZE
            block_schedule = tuple(int(value) for value in schedule[first:last])
            block_rates = sorted({
                rate for rate in block_schedule
                if rate < get_trellis_family(E2M1_FAMILY).bypass_rate
            })
            block_alphabets = {rate: alphabets[rate] for rate in block_rates}
            final_terminal_wire = pack_planes(
                family=E2M1_FAMILY,
                body_rate_q256=sum(block_schedule),
                schedule=block_schedule,
                layout=layout,
                u_bits=final_encoded.u_bits,
                point_indices=final_encoded.point_indices,
                bypass_codes=final_encoded.bypass_codes,
                alphabets=block_alphabets,
                scale_blob=final_encoded.scale_blob,
                global_scale_real=shared_global,
            )
            final_terminal_blob = final_terminal_wire.to_bytes()
            if TrellisWire.from_bytes(
                final_terminal_blob
            ).to_bytes() != final_terminal_blob:
                raise AssertionError(
                    "spliced Arm E terminal wire did not reserialize exactly"
                )
            final_terminal_decoded = decode_values_torch(
                final_terminal_blob, device=weight.device, dtype=weight.dtype
            )
            expected_block = group_q[
                :, local_index * TRELLIS_FEEDBACK_BLOCK_SIZE:
                (local_index + 1) * TRELLIS_FEEDBACK_BLOCK_SIZE
            ]
            if not torch.equal(final_terminal_decoded, expected_block):
                raise AssertionError(
                    "spliced Arm E terminal decode differs from selected trajectory"
                )
            final_target = group_targets[local_index]
            target_group_scales = (
                final_target.reshape(rows, 16, 16).abs().amax(-1) / 6.0
            )
            block_receipts[block_index] = {
                **{
                    key: value for key, value in identity_receipt.items()
                    if key not in {
                        "arm", "scale_proposal", "feedback_target_sha256",
                        "terminal_wire_identity_sha256",
                        "decoded_terminal_sha256",
                        "clipped_group_count_at_fixed_global",
                    }
                },
                "feedback_target_sha256": _tensor_sha256(final_target),
                "terminal_wire_identity_sha256": hashlib.sha256(
                    final_terminal_blob
                ).hexdigest(),
                "decoded_terminal_sha256": _tensor_sha256(
                    final_terminal_decoded
                ),
                "clipped_group_count_at_fixed_global": int(
                    (
                        target_group_scales
                        > shared_global_tensor * 448.0
                    ).sum().item()
                ),
                "scale_selection": {
                    "mode": "e4m3_grid_gated_v1",
                    "scope": "row_factor_group",
                    "candidate_win_rows": int(candidate_wins.sum().item()),
                    "identity_win_or_tie_rows": int(
                        rows - candidate_wins.sum().item()
                    ),
                    "identity_arm": identity_receipt,
                    "candidate_arm": candidate_receipt,
                },
            }

        unit_lower = group.feedback_lower.clone()
        for first in range(0, group_columns, TRELLIS_FEEDBACK_BLOCK_SIZE):
            unit_lower[
                first:first + TRELLIS_FEEDBACK_BLOCK_SIZE,
                first:first + TRELLIS_FEEDBACK_BLOCK_SIZE,
            ] = torch.eye(
                TRELLIS_FEEDBACK_BLOCK_SIZE,
                dtype=unit_lower.dtype,
                device=unit_lower.device,
            )
        dense_d = torch.block_diag(*group.diagonal_blocks.unbind(0))
        reconstructed_hessian = unit_lower @ dense_d @ unit_lower.T
        group_factor_max_abs = float(
            (reconstructed_hessian - group.transformed_hessian)
            .abs().max().item()
        )
        factor_errors.append(group_factor_max_abs)
        if not torch.allclose(
            reconstructed_hessian,
            group.transformed_hessian,
            rtol=3.0e-5,
            atol=3.0e-5,
        ):
            raise AssertionError(
                "BlockLDL factors do not reconstruct their transformed "
                f"Hessian group; max_abs={group_factor_max_abs}"
            )
        def decomposition_evidence(
            arm_name: str, reconstruction: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, float]:
            group_error = group_weight - reconstruction.float()
            transformed_error = group_error @ unit_lower
            group_decomposed = sum(
                (
                    transformed_error[
                        :, index * TRELLIS_FEEDBACK_BLOCK_SIZE:
                        (index + 1) * TRELLIS_FEEDBACK_BLOCK_SIZE
                    ] @ group.diagonal_blocks[index]
                ).mul(
                    transformed_error[
                        :, index * TRELLIS_FEEDBACK_BLOCK_SIZE:
                        (index + 1) * TRELLIS_FEEDBACK_BLOCK_SIZE
                    ]
                ).sum()
                for index in range(group_block_count)
            )
            group_direct = (
                (group_error @ group.transformed_hessian) * group_error
            ).sum()
            absolute = float((group_decomposed - group_direct).abs().item())
            if not torch.allclose(
                group_decomposed,
                group_direct,
                rtol=3.0e-5,
                atol=3.0e-5,
            ):
                raise AssertionError(
                    f"{arm_name} BlockLDL quadratic decomposition mismatch; "
                    f"abs_error={absolute}"
                )
            return group_decomposed, group_direct, absolute

        identity_decomposed, identity_direct, identity_proxy_abs = (
            decomposition_evidence("identity", identity_q)
        )
        if scale_grid_enabled:
            candidate_decomposed, candidate_direct, candidate_proxy_abs = (
                decomposition_evidence("candidate", candidate_q)
            )
        else:
            candidate_decomposed = identity_decomposed
            candidate_direct = identity_direct
            candidate_proxy_abs = identity_proxy_abs
        group_decomposed_proxy, group_direct_proxy, group_proxy_abs = (
            decomposition_evidence("selected", group_q)
        )
        proxy_errors.append(group_proxy_abs)
        decomposed_proxy_total += group_decomposed_proxy
        direct_proxy_total += group_direct_proxy
        group_feedback_nonzero = int(
            torch.count_nonzero(group.feedback_lower).item()
        )
        feedback_nonzero_count += group_feedback_nonzero
        factor_group_record: dict[str, object] = {
            "index": group_index,
            "first_column": group_first,
            "last_column_exclusive": group_last,
            "columns": group_columns,
            "source_diagonal_sha256": (
                _tensor_sha256(
                    prepared.source_hessian_diagonal[group_first:group_last]
                )
                if structured_diagonal else None
            ),
            "transformed_hessian_sha256": _tensor_sha256(
                group.transformed_hessian
            ),
            "feedback_lower_sha256": _tensor_sha256(group.feedback_lower),
            "diagonal_blocks_sha256": _tensor_sha256(group.diagonal_blocks),
            "feedback_nonzero_count": group_feedback_nonzero,
            "factorization_max_abs_error": group_factor_max_abs,
            "buffered_oracle_target_max_abs_error": group_target_max_abs,
            "quadratic_decomposition_abs_error": group_proxy_abs,
        }
        if scale_grid_enabled:
            factor_group_record["scale_selection"] = {
                "mode": "e4m3_grid_gated_v1",
                "scope": "row_factor_group",
                "full_recurrence_arms": 2,
                "identity_row_proxy_fp64_sha256": _tensor_sha256(
                    identity_row_proxy
                ),
                "candidate_row_proxy_fp64_sha256": _tensor_sha256(
                    candidate_row_proxy
                ),
                "final_row_proxy_fp64_sha256": _tensor_sha256(final_row_proxy),
                "candidate_win_mask_sha256": _tensor_sha256(candidate_wins),
                "candidate_win_rows": int(candidate_wins.sum().item()),
                "identity_win_or_tie_rows": int(
                    rows - candidate_wins.sum().item()
                ),
                "cf_exact_minimum": True,
                "cf_le_c0": True,
                "selected_recurrence_reverified": True,
                "selected_oracle_target_max_abs_error": selected_target_error,
                "identity_decomposition_abs_error": identity_proxy_abs,
                "candidate_decomposition_abs_error": candidate_proxy_abs,
                "selected_decomposition_abs_error": group_proxy_abs,
                "identity_decomposed_proxy_fp32_hex": float(
                    identity_decomposed.item()
                ).hex(),
                "candidate_decomposed_proxy_fp32_hex": float(
                    candidate_decomposed.item()
                ).hex(),
                "identity_direct_proxy_fp32_hex": float(
                    identity_direct.item()
                ).hex(),
                "candidate_direct_proxy_fp32_hex": float(
                    candidate_direct.item()
                ).hex(),
            }
        factor_group_records.append(factor_group_record)

    if set(encoded_blocks) != set(range(block_count)):
        raise AssertionError("BlockLDL recurrence did not encode every block")
    if set(identity_encoded_blocks) != set(range(block_count)):
        raise AssertionError("identity BlockLDL recurrence did not encode every block")
    if any(target is None for target in buffered_targets):
        raise AssertionError("BlockLDL recurrence omitted a feedback target")
    target_max_abs = max(target_errors)
    factor_max_abs = max(factor_errors)
    proxy_abs = float((decomposed_proxy_total - direct_proxy_total).abs().item())
    if not torch.allclose(
        decomposed_proxy_total,
        direct_proxy_total,
        rtol=3.0e-5,
        atol=3.0e-5,
    ):
        raise AssertionError(
            "structured BlockLDL total quadratic decomposition mismatch; "
            f"abs_error={proxy_abs}"
        )

    def compose_wire(
        blocks: Mapping[int, EncodedTrellisPlanes]
    ) -> TrellisWire:
        ordered = [blocks[index] for index in range(block_count)]
        scale_planes = [
            torch.frombuffer(
                bytearray(value.scale_blob), dtype=torch.uint8
            ).reshape(rows, TRELLIS_FEEDBACK_BLOCK_SIZE // 16)
            for value in ordered
        ]
        scale_blob = torch.cat(
            scale_planes, dim=1
        ).contiguous().numpy().tobytes()
        return pack_planes(
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

    identity_wire = compose_wire(identity_encoded_blocks)
    identity_blob = identity_wire.to_bytes()
    if TrellisWire.from_bytes(identity_blob).to_bytes() != identity_blob:
        raise AssertionError("identity BlockLDL wire did not reserialize exactly")
    wire = compose_wire(encoded_blocks)
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
    if len(blob) != len(identity_blob):
        raise AssertionError("Arm E scale grid changed the exact wire byte length")
    if scale_grid_candidate_win_rows == 0 and blob != identity_blob:
        raise AssertionError(
            "Arm E no-win scale-grid result is not byte-identical to identity"
        )
    if (
        scale_grid_enabled
        and _require_scale_grid_source_unchanged() != scale_grid_source_at_start
    ):
        raise AssertionError("scale-grid selector source changed during Arm E encode")

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
    logical_feedback_sha256 = _canonical_sha256({
        "representation": (
            "block_diagonal_factor_groups"
            if structured_diagonal else "dense_whole_matrix"
        ),
        "groups": [
            {
                "index": item["index"],
                "first_column": item["first_column"],
                "last_column_exclusive": item["last_column_exclusive"],
                "sha256": item["feedback_lower_sha256"],
            }
            for item in factor_group_records
        ],
    })
    logical_diagonal_blocks_sha256 = _canonical_sha256({
        "terminal_block_size": TRELLIS_FEEDBACK_BLOCK_SIZE,
        "groups": [
            {
                "index": item["index"],
                "first_column": item["first_column"],
                "last_column_exclusive": item["last_column_exclusive"],
                "sha256": item["diagonal_blocks_sha256"],
            }
            for item in factor_group_records
        ],
    })
    factor = {
        "algorithm": "pinned_qtip_block_LDL_reverse_feedback",
        "block_size": TRELLIS_FEEDBACK_BLOCK_SIZE,
        "buffer_blocks": int(buffer_blocks),
        "factorization_strategy": (
            "exact_block_diagonal_from_retained_source_diagonal_v1"
            if structured_diagonal else "dense_whole_matrix_v1"
        ),
        "factor_groups": factor_group_records,
        "factor_group_count": len(factor_group_records),
        "largest_factor_group_columns": max(
            int(item["columns"]) for item in factor_group_records
        ),
        "dense_k_by_k_materialized": not structured_diagonal,
        "off_block_coupling": (
            "identically_zero_by_block_local_transform_of_diagonal_source"
            if structured_diagonal else "represented_in_dense_factor"
        ),
        "feedback_lower_sha256": logical_feedback_sha256,
        "diagonal_blocks_sha256": logical_diagonal_blocks_sha256,
        "factorization_max_abs_error": factor_max_abs,
        "hessian_symmetrization": (
            "fp32 (H+H.T)/2 after max asymmetry <= 1e-5"
        ),
        "buffered_oracle_target_max_abs_error": target_max_abs,
        "quadratic_decomposition_abs_error": proxy_abs,
        "quadratic_decomposition_group_max_abs_error": max(proxy_errors),
        "full_cross_block_feedback_matrix_consumed": (
            block_count > 1 and feedback_nonzero_count > 0
        ),
        "full_cross_output_rows_processed": True,
        "cross_block_feedback_nonzero_count": feedback_nonzero_count,
        "dense_D_terminal_consumption": {
            "diagonal_consumed": terminal_metric_mode == "diag_block_D",
            "off_diagonal_consumed": False,
            "full_matrix_consumed": False,
            "exact_dense_objective": False,
        },
        "terminal_metric_mode": terminal_metric_mode,
        "atomic_terminal_geometry": "one_output_row_by_256_input_columns",
        "qtip_16_by_16_terminal_geometry_claimed": False,
    }
    if scale_grid_enabled:
        assert scale_grid_menu is not None
        factor["scale_selection"] = {
            "mode": "e4m3_grid_gated_v1",
            "scope": "row_factor_group",
            "full_recurrence_arms": 2,
            "multipliers": list(scale_grid_menu),
            "multipliers_sha256": _canonical_sha256(list(scale_grid_menu)),
            "identity_index": 0,
            "candidate_win_rows": scale_grid_candidate_win_rows,
            "identity_win_or_tie_rows": (
                scale_grid_total_rows - scale_grid_candidate_win_rows
            ),
            "total_row_factor_groups": scale_grid_total_rows,
            "immutable_global_scale": True,
            "cf_exact_minimum_per_row_factor_group": True,
            "cf_le_c0": True,
            "selected_recurrence_reverified": True,
            "same_length": True,
            "no_win_byte_identical": scale_grid_candidate_win_rows == 0,
            "identity_wire_bytes": len(identity_blob),
            "final_wire_bytes": len(blob),
            "identity_wire_sha256": hashlib.sha256(identity_blob).hexdigest(),
            "final_wire_sha256": wire_sha256,
            "wire_byte_delta": 0,
            "delta_bpw_q256": 0,
            "selector_source_sha256": scale_grid_source_at_start,
        }
    if structured_diagonal:
        assert isinstance(prepared, PreparedDiagonalHessianOneLinear)
        _validate_prepared_diagonal_hessian_one_linear(
            prepared, body_rate_q256=body_rate_q256
        )
    else:
        assert isinstance(prepared, PreparedOneLinear)
        _validate_prepared_one_linear(
            prepared, body_rate_q256=body_rate_q256
        )
    (
        implementation_source_sha256,
        implementation_encoder_sha256,
    ) = _require_implementation_sources_unchanged()
    if scale_grid_enabled:
        _require_scale_grid_source_unchanged()
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
        "encoder_source_sha256": implementation_encoder_sha256,
        "backend": backend,
        "point_route": point_route,
        "sb_chunk": int(sb_chunk),
        "determinism_mode": determinism_mode,
        "tailbite_candidates": int(tailbite_candidates),
        "scale_selection": (
            {
                "mode": "e4m3_grid_gated_v1",
                "scope": "row_factor_group",
                "multipliers": list(scale_grid_menu),
                "multipliers_sha256": _canonical_sha256(list(scale_grid_menu)),
                "selector_source_sha256": scale_grid_source_at_start,
            }
            if scale_grid_enabled else {"mode": "off"}
        ),
    }
    expected_wire_alphabets = {
        int(rate): tuple(codes) for rate, codes in alphabets.items()
    }
    for arm_name, arm_blob in (
        ("identity", identity_blob),
        ("selected", blob),
    ):
        parsed_arm = TrellisWire.from_bytes(arm_blob)
        if (
            parsed_arm.family != wire_recipe["family"]
            or parsed_arm.body_rate_q256 != wire_recipe["body_rate_q256"]
            or list(parsed_arm.schedule) != wire_recipe["schedule"]
            or parsed_arm.layout != wire_recipe["layout"]
            or dict(parsed_arm.alphabets) != expected_wire_alphabets
            or parsed_arm.global_scale_real != wire_recipe["global_scale_real"]
        ):
            raise AssertionError(
                f"{arm_name} Arm E wire differs from the bound render recipe"
            )
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
        "implementation_provenance": {
            "producer_source": {
                "path": (
                    "research/qtip_native_nvfp4_2026-08-30/"
                    "trellis_online_hadamard_producer.py"
                ),
                "sha256": implementation_source_sha256,
            },
            "encoder_source": {
                "path": "prismaquant/trellis_encoder.py",
                "sha256": implementation_encoder_sha256,
            },
            "scale_grid_selector_source": (
                {
                    "path": "prismaquant/trellis_scale_grid.py",
                    "sha256": scale_grid_source_at_start,
                }
                if scale_grid_enabled else None
            ),
            "qtip_source_audit": {
                "repository": QTIP_REPOSITORY,
                "commit": QTIP_PINNED_COMMIT,
                "source_sha256": dict(sorted(QTIP_SOURCE_FILES.items())),
                "runtime_or_wire_imported": False,
            },
        },
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
    _require_implementation_sources_unchanged()
    if scale_grid_enabled:
        _require_scale_grid_source_unchanged()
    return CombinedOneLinearArtifact(
        wire_bytes=blob,
        decoded_transformed_weight=decoded_weight,
        decoded_codes=decoded_codes,
        online_transform=contract,
        receipt=receipt,
    )


def require_blockldl_trellis_artifact_replay(
    expected: CombinedOneLinearArtifact,
    prepared: PreparedOneLinear | PreparedDiagonalHessianOneLinear,
    activations: torch.Tensor,
    **recipe: object,
) -> CombinedOneLinearArtifact:
    """Rerun both configured trajectories and require exact artifact identity.

    The combined receipt's self-digest detects accidental mutation but cannot
    authenticate opaque trajectory hashes by itself. This explicit research
    replay is the authority boundary: caller-owned prepared tensors and the
    full recipe are revalidated, the producer reruns, and canonical bytes,
    decoded tensors, transform metadata, and all receipt semantics must match.
    """

    if not isinstance(expected, CombinedOneLinearArtifact):
        raise ValueError("expected replay artifact must be CombinedOneLinearArtifact")
    if not isinstance(expected.receipt, Mapping):
        raise ValueError("expected replay receipt must be an object")
    expected_body = dict(expected.receipt)
    expected_identity = expected_body.pop("identity_sha256", None)
    _require_sha256(
        expected_identity,
        where="expected replay receipt identity_sha256",
    )
    if not hmac.compare_digest(
        expected_identity,
        _canonical_sha256(expected_body),
    ):
        raise ValueError("expected replay receipt identity mismatch")
    replay = require_blockldl_trellis_wire_round_trip(
        prepared,
        activations,
        **recipe,
    )
    if expected.wire_bytes != replay.wire_bytes:
        raise ValueError("BlockLDL artifact replay differs at canonical wire bytes")
    if not torch.equal(
        expected.decoded_transformed_weight,
        replay.decoded_transformed_weight,
    ):
        raise ValueError("BlockLDL artifact replay differs at decoded weight")
    if not torch.equal(expected.decoded_codes, replay.decoded_codes):
        raise ValueError("BlockLDL artifact replay differs at decoded codes")
    if not _json_exact_equal(
        expected.online_transform,
        replay.online_transform,
    ):
        raise ValueError("BlockLDL artifact replay differs at transform metadata")
    if not _json_exact_equal(expected.receipt, replay.receipt):
        raise ValueError("BlockLDL artifact replay differs at receipt semantics")
    return replay


__all__ = [
    "BLOCKLDL_COMBINED_ARTIFACT_SCHEMA",
    "BlockLDLFactorGroup",
    "CombinedOneLinearArtifact",
    "COMBINED_ARTIFACT_SCHEMA",
    "DIAGONAL_HESSIAN_SCAFFOLD_SCHEMA",
    "PreparedDiagonalHessianOneLinear",
    "PreparedOneLinear",
    "QTIP_PINNED_COMMIT",
    "QTIP_REPOSITORY",
    "QTIP_SOURCE_FILES",
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
    "iter_transformed_diagonal_block_ldl_factors",
    "producer_source_sha256",
    "prepare_one_linear_diagonal_hessian_scaffold",
    "prepare_one_linear_scaffold",
    "qtip_block_ldl_factors",
    "require_blockldl_trellis_artifact_replay",
    "require_blockldl_trellis_wire_round_trip",
    "require_combined_wire_round_trip",
    "reverse_block_feedback_buffered",
    "reverse_block_feedback_reference",
    "seeded_sign_digest",
    "seeded_signs",
    "transform_weight",
    "transform_weight_and_hessian",
    "transformed_diagonal_hessian_block",
    "transformed_activations",
    "validate_online_transform",
    "verify_post_decode_serve_algebra",
]
