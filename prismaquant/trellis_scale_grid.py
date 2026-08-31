"""Fail-closed research scale-grid gate for E2M1 trellis wires.

The multiplier search in this module is only a proposal generator. Safety is
established by two complete, independent trellis encodes, scoring the
reconstructions decoded from each canonical wire in fp64, and splicing only
at the proven independent ``(row, 256-column superblock)`` scope. This module
is research scaffolding: it does not register a format or alter a runtime ABI.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Mapping, Sequence, cast

import torch

from .trellis_encoder import (
    EncodedTrellisPlanes,
    decode_e2m1_scale_codes,
    encode_trellis_planes,
    encoder_source_sha256,
    iter_snapped_e2m1_scale_codes,
    require_encoder_source_unchanged,
    snap_e2m1_scale_codes,
)
from .trellis_formats import E2M1_FAMILY, SUPERBLOCK_WEIGHTS
from .trellis_wire import (
    TrellisWire,
    decode_codes_torch,
    decode_values_torch,
    pack_planes,
)


SCALE_GRID_SCHEMA = "prismaquant.research.trellis_scale_grid_two_arm.v2"
SCALE_GRID_RENDER_RECIPE_SCHEMA = (
    "prismaquant.research.trellis_scale_grid_render_recipe.v2"
)
SCALE_GRID_IMPLEMENTATION_CLOSURE_SCHEMA = (
    "prismaquant.research.trellis_scale_grid_implementation_closure.v1"
)
E2M1_LEVELS = (
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
)
E4M3_MAX = 448.0
E2M1_GROUP_SIZE = 16
SCALE_PLANE_RATE_Q256 = 128


class ScaleGridError(RuntimeError):
    """A scale-grid proof obligation was not satisfied."""


def scale_grid_multipliers() -> tuple[float, ...]:
    """Identity first, then the 33 preregistered log-spaced proposals."""

    probes = torch.logspace(
        math.log10(0.55), math.log10(1.30), 33, dtype=torch.float64
    ).tolist()
    return (1.0, *(sorted(float(value) for value in probes if value != 1.0)))


SCALE_GRID_MULTIPLIERS = scale_grid_multipliers()
_SCALE_GRID_SOURCE_PATH = Path(__file__).resolve()
_TRELLIS_ENCODER_SOURCE_PATH = _SCALE_GRID_SOURCE_PATH.with_name(
    "trellis_encoder.py"
)
_TRELLIS_WIRE_SOURCE_PATH = _SCALE_GRID_SOURCE_PATH.with_name("trellis_wire.py")
_TRELLIS_FORMATS_SOURCE_PATH = _SCALE_GRID_SOURCE_PATH.with_name(
    "trellis_formats.py"
)
_IMPORTED_SCALE_GRID_SOURCE_SHA256 = hashlib.sha256(
    _SCALE_GRID_SOURCE_PATH.read_bytes()
).hexdigest()
_IMPORTED_ENCODER_SOURCE_SHA256 = encoder_source_sha256()
_IMPORTED_WIRE_SOURCE_SHA256 = hashlib.sha256(
    _TRELLIS_WIRE_SOURCE_PATH.read_bytes()
).hexdigest()
_IMPORTED_FORMATS_SOURCE_SHA256 = hashlib.sha256(
    _TRELLIS_FORMATS_SOURCE_PATH.read_bytes()
).hexdigest()


@dataclass(frozen=True, slots=True)
class ScalePlaneProposal:
    """A legal candidate plane proposed by a non-authoritative RTN floor."""

    scale_codes: torch.Tensor
    multiplier_indices: torch.Tensor
    multipliers: tuple[float, ...]
    group_sse: torch.Tensor
    identity_group_sse: torch.Tensor
    masked_candidate_cells: int
    clipped_candidate_cells: int


@dataclass(frozen=True, slots=True)
class ScaleGridSelection:
    """Canonical two-arm wire and the evidence for its diagonal gate."""

    encoded_planes: EncodedTrellisPlanes
    wire_bytes: bytes
    decoded_codes: torch.Tensor
    decoded_weight: torch.Tensor
    identity_wire_bytes: bytes
    identity_tile_sse: torch.Tensor
    candidate_tile_sse: torch.Tensor
    final_tile_sse: torch.Tensor
    candidate_wins: torch.Tensor
    proposal: ScalePlaneProposal
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


def scale_grid_source_sha256() -> str:
    """Return the selector source identity captured at module import."""

    return _IMPORTED_SCALE_GRID_SOURCE_SHA256


def _current_scale_grid_source_sha256() -> str:
    return hashlib.sha256(_SCALE_GRID_SOURCE_PATH.read_bytes()).hexdigest()


def _current_wire_source_sha256() -> str:
    return hashlib.sha256(_TRELLIS_WIRE_SOURCE_PATH.read_bytes()).hexdigest()


def _current_formats_source_sha256() -> str:
    return hashlib.sha256(_TRELLIS_FORMATS_SOURCE_PATH.read_bytes()).hexdigest()


def require_scale_grid_source_unchanged() -> str:
    current = _current_scale_grid_source_sha256()
    if not hmac.compare_digest(current, _IMPORTED_SCALE_GRID_SOURCE_SHA256):
        raise ScaleGridError(
            "trellis scale-grid source changed since module import; refusing "
            "to execute or bind a mixed selector closure"
        )
    return _IMPORTED_SCALE_GRID_SOURCE_SHA256


def require_scale_grid_encoder_source_unchanged() -> str:
    try:
        current = require_encoder_source_unchanged()
    except RuntimeError as exc:
        raise ScaleGridError(str(exc)) from exc
    if not hmac.compare_digest(current, _IMPORTED_ENCODER_SOURCE_SHA256):
        raise ScaleGridError(
            "trellis encoder source changed since scale-grid import; refusing "
            "to execute or bind a mixed encoder closure"
        )
    return _IMPORTED_ENCODER_SOURCE_SHA256


def require_scale_grid_implementation_unchanged() -> tuple[str, str]:
    """Revalidate every source and declared callable in the execution closure.

    The two-element return is retained for callers written against v1.  The
    complete closure is available from ``scale_grid_implementation_closure``.
    """

    if _scale_grid_execution_gateway is not _BOUND_SCALE_GRID_EXECUTION_GATEWAY:
        raise ScaleGridError("trellis scale-grid execution gateway was substituted")
    selector, encoder, _closure = _BOUND_SCALE_GRID_EXECUTION_GATEWAY()
    return selector, encoder


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_nonnegative_hex(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        decoded = float.fromhex(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(decoded) and decoded >= 0.0


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_scale_grid_receipt(value: Mapping[str, object]) -> Mapping[str, object]:
    """Validate the closed JSON structure and its internal self-consistency.

    Opaque artifact hashes require ``require_scale_grid_selection_replay``;
    this mapping-only check is deliberately not an authority boundary.
    """

    if _scale_grid_execution_gateway is not _BOUND_SCALE_GRID_EXECUTION_GATEWAY:
        raise ScaleGridError("trellis scale-grid execution gateway was substituted")
    if not isinstance(value, Mapping):
        raise ScaleGridError("scale-grid receipt must be an object")
    body = dict(value)
    identity = body.pop("identity_sha256", None)
    if (
        not _is_sha256(identity)
        or not hmac.compare_digest(identity, _canonical_sha256(body))
    ):
        raise ScaleGridError("scale-grid receipt identity mismatch")
    expected_root = {
        "schema", "status", "scope", "mode", "selection_scope", "snap_path",
        "multipliers", "multipliers_sha256", "multiplier_count", "identity_index",
        "selector_source_sha256", "encoder_source_sha256",
        "implementation_closure",
        "render_recipe", "render_recipe_identity_sha256",
        "global_scale_real_hex", "pricing", "arms", "final", "proof",
        "format_registry_entries_created", "runtime_pin_changed",
        "production_contract_changed", "producer_eligible",
    }
    if set(body) != expected_root:
        raise ScaleGridError("scale-grid receipt fields differ from the closed schema")
    constants = {
        "schema": SCALE_GRID_SCHEMA,
        "status": "two_full_arms_canonical_decode_gate_verified",
        "scope": "research_only_unregistered",
        "mode": "e4m3_grid_gated_v1",
        "selection_scope": "row_superblock",
        "snap_path": "fixed_global_override_floor_2pow9",
        "identity_index": 0,
        "format_registry_entries_created": 0,
        "runtime_pin_changed": False,
        "production_contract_changed": False,
        "producer_eligible": False,
    }
    if any(body.get(field) != expected for field, expected in constants.items()):
        raise ScaleGridError("scale-grid receipt constant mismatch")
    menu = body["multipliers"]
    count = body["multiplier_count"]
    if (
        not isinstance(menu, list)
        or not _nonnegative_int(count)
        or len(menu) != count
        or any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in menu
        )
    ):
        raise ScaleGridError("scale-grid receipt multiplier menu is malformed")
    _checked_multipliers(menu)
    if body["multipliers_sha256"] != _canonical_sha256(menu):
        raise ScaleGridError("scale-grid receipt multiplier identity mismatch")
    current_selector, current_encoder, current_closure = (
        _BOUND_SCALE_GRID_EXECUTION_GATEWAY()
    )
    if body["implementation_closure"] != current_closure:
        raise ScaleGridError("scale-grid receipt implementation closure mismatch")
    if (
        not _is_sha256(body["selector_source_sha256"])
        or body["selector_source_sha256"] != current_selector
    ):
        raise ScaleGridError("scale-grid receipt selector source mismatch")
    if (
        not _is_sha256(body["encoder_source_sha256"])
        or body["encoder_source_sha256"] != current_encoder
    ):
        raise ScaleGridError("scale-grid receipt encoder source mismatch")
    render = body["render_recipe"]
    render_fields = {
        "schema", "family", "body_rate_q256", "schedule", "layout",
        "alphabets", "scale_rule", "sb_chunk", "determinism_mode",
        "tailbite_candidates", "backend", "point_route",
        "global_scale_real_override_hex", "encoder_source_sha256",
        "implementation_closure_identity_sha256", "scale_selection",
    }
    render_selection_fields = {
        "mode", "scope", "snap_path", "multipliers", "multipliers_sha256",
        "identity_index", "selector_source_sha256",
    }
    if (
        not isinstance(render, Mapping)
        or set(render) != render_fields
        or body["render_recipe_identity_sha256"] != _canonical_sha256(render)
        or render["schema"] != SCALE_GRID_RENDER_RECIPE_SCHEMA
        or render["family"] != E2M1_FAMILY
        or render["encoder_source_sha256"] != current_encoder
        or render["implementation_closure_identity_sha256"]
        != current_closure["identity_sha256"]
        or not _nonnegative_int(render["body_rate_q256"])
        or render["body_rate_q256"] <= 0
        or not isinstance(render["schedule"], list)
        or not render["schedule"]
        or any(not _nonnegative_int(item) for item in render["schedule"])
        or not isinstance(render["alphabets"], Mapping)
        or any(
            not isinstance(rate, str)
            or not rate.isdigit()
            or not isinstance(codes, list)
            or any(not _nonnegative_int(code) for code in codes)
            for rate, codes in render["alphabets"].items()
        )
        or any(
            not isinstance(render[field], str) or not render[field]
            for field in (
                "layout", "scale_rule", "determinism_mode", "backend",
                "point_route",
            )
        )
        or not _nonnegative_int(render["sb_chunk"])
        or render["sb_chunk"] <= 0
        or not _nonnegative_int(render["tailbite_candidates"])
        or render["tailbite_candidates"] <= 0
        or not isinstance(render["scale_selection"], Mapping)
        or set(render["scale_selection"]) != render_selection_fields
    ):
        raise ScaleGridError("scale-grid receipt render recipe is malformed")
    render_selection = render["scale_selection"]
    if (
        render_selection["mode"] != body["mode"]
        or render_selection["scope"] != body["selection_scope"]
        or render_selection["snap_path"] != body["snap_path"]
        or render_selection["multipliers"] != menu
        or render_selection["multipliers_sha256"]
        != body["multipliers_sha256"]
        or render_selection["identity_index"] != body["identity_index"]
        or render_selection["selector_source_sha256"] != current_selector
    ):
        raise ScaleGridError("scale-grid receipt render recipe identity mismatch")
    override_hex = render["global_scale_real_override_hex"]
    if override_hex is not None:
        try:
            override = float.fromhex(override_hex)
        except (OverflowError, TypeError, ValueError):
            override = float("nan")
        if not math.isfinite(override) or override <= 0.0:
            raise ScaleGridError("scale-grid receipt render override is malformed")
    try:
        global_scale = float.fromhex(body["global_scale_real_hex"])
    except (OverflowError, TypeError, ValueError):
        global_scale = float("nan")
    if not math.isfinite(global_scale) or global_scale <= 0.0:
        raise ScaleGridError("scale-grid receipt global scale is malformed")
    pricing = body["pricing"]
    pricing_fields = {
        "body_rate_q256", "scale_plane_rate_q256", "identity_wire_bytes",
        "candidate_wire_bytes", "final_wire_bytes", "wire_byte_delta",
        "delta_bpw_q256",
    }
    if (
        not isinstance(pricing, Mapping)
        or set(pricing) != pricing_fields
        or any(not _nonnegative_int(pricing[field]) for field in pricing_fields)
        or pricing["body_rate_q256"] <= 0
        or pricing["identity_wire_bytes"] <= 0
        or pricing["identity_wire_bytes"] != pricing["candidate_wire_bytes"]
        or pricing["identity_wire_bytes"] != pricing["final_wire_bytes"]
        or pricing["wire_byte_delta"] != 0
        or pricing["delta_bpw_q256"] != 0
        or pricing["scale_plane_rate_q256"] != SCALE_PLANE_RATE_Q256
        or pricing["body_rate_q256"] != render["body_rate_q256"]
    ):
        raise ScaleGridError("scale-grid receipt exact-byte pricing mismatch")
    proof = body["proof"]
    proof_fields = {
        "full_identity_encode", "full_candidate_encode",
        "canonical_pack_reparse_decode_each_arm", "cf_exact_minimum", "cf_le_c0",
        "same_length", "identity_candidate_byte_equal",
        "strict_candidate_wins_tie_identity", "no_win_byte_identical",
    }
    required_true = proof_fields - {"no_win_byte_identical"}
    if (
        not isinstance(proof, Mapping)
        or set(proof) != proof_fields
        or any(proof[field] is not True for field in required_true)
        or not isinstance(proof["no_win_byte_identical"], bool)
    ):
        raise ScaleGridError("scale-grid receipt proof obligation is not true")
    arms = body["arms"]
    identity_arm_fields = {
        "objective_fp64_hex", "wire_sha256", "decoded_weight_sha256",
    }
    candidate_arm_fields = identity_arm_fields | {
        "masked_candidate_cells", "clipped_candidate_cells",
    }
    if (
        not isinstance(arms, Mapping)
        or set(arms) != {"identity", "candidate"}
        or not isinstance(arms["identity"], Mapping)
        or not isinstance(arms["candidate"], Mapping)
        or set(arms["identity"]) != identity_arm_fields
        or set(arms["candidate"]) != candidate_arm_fields
    ):
        raise ScaleGridError("scale-grid receipt arm records are malformed")
    for arm_name in ("identity", "candidate"):
        arm = arms[arm_name]
        if (
            not _finite_nonnegative_hex(arm["objective_fp64_hex"])
            or not _is_sha256(arm["wire_sha256"])
            or not _is_sha256(arm["decoded_weight_sha256"])
        ):
            raise ScaleGridError("scale-grid receipt arm evidence is malformed")
    candidate_arm = arms["candidate"]
    if (
        not _nonnegative_int(candidate_arm["masked_candidate_cells"])
        or not _nonnegative_int(candidate_arm["clipped_candidate_cells"])
    ):
        raise ScaleGridError("scale-grid receipt candidate census is malformed")
    final = body["final"]
    final_fields = {
        "objective_fp64_hex", "wire_sha256", "tiles", "candidate_win_tiles",
        "identity_win_tiles", "tie_tiles", "max_tile_regression",
        "tile_sse_sha256",
    }
    if (
        not isinstance(final, Mapping)
        or set(final) != final_fields
        or not _finite_nonnegative_hex(final["objective_fp64_hex"])
        or not _is_sha256(final["wire_sha256"])
        or not _is_sha256(final["tile_sse_sha256"])
        or any(
            not _nonnegative_int(final[field])
            for field in (
                "tiles", "candidate_win_tiles", "identity_win_tiles", "tie_tiles"
            )
        )
        or final["tiles"] <= 0
        or (
            final["candidate_win_tiles"]
            + final["identity_win_tiles"]
            + final["tie_tiles"]
            != final["tiles"]
        )
        or isinstance(final["max_tile_regression"], bool)
        or final["max_tile_regression"] != 0.0
        or proof["no_win_byte_identical"]
        != (final["candidate_win_tiles"] == 0)
    ):
        raise ScaleGridError("scale-grid receipt reports a tile regression")
    return body


def _checked_multipliers(values: Sequence[float]) -> tuple[float, ...]:
    raw = tuple(values)
    if any(
        type(value) not in {int, float}
        for value in raw
    ):
        raise ScaleGridError("scale-grid multipliers must be plain numeric values")
    result = tuple(float(value) for value in raw)
    if not result or result[0] != 1.0:
        raise ScaleGridError("scale-grid candidate zero must be exactly 1.0")
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ScaleGridError("scale-grid multipliers must be finite and positive")
    if len(set(result)) != len(result):
        raise ScaleGridError("scale-grid multipliers must be unique")
    return result


def _checked_render_inputs(
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
    global_scale_real_override: object,
    selection_scope: object,
) -> dict[str, object]:
    integer_fields = {
        "body_rate_q256": body_rate_q256,
        "sb_chunk": sb_chunk,
        "tailbite_candidates": tailbite_candidates,
    }
    for name, value in integer_fields.items():
        if type(value) is not int or value <= 0:
            raise ScaleGridError(f"scale-grid {name} must be a positive plain int")
    string_fields = {
        "layout": layout,
        "scale_rule": scale_rule,
        "determinism_mode": determinism_mode,
        "backend": backend,
        "point_route": point_route,
        "selection_scope": selection_scope,
    }
    for name, value in string_fields.items():
        if type(value) is not str or not value:
            raise ScaleGridError(f"scale-grid {name} must be a nonempty plain string")
    try:
        frozen_schedule = tuple(schedule)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ScaleGridError("scale-grid schedule must be an integer sequence") from exc
    if not frozen_schedule or any(type(value) is not int for value in frozen_schedule):
        raise ScaleGridError("scale-grid schedule must contain only plain ints")
    if not isinstance(alphabets, Mapping):
        raise ScaleGridError("scale-grid alphabets must be a mapping")
    frozen_alphabets: dict[int, tuple[int, ...]] = {}
    for rate, codes in dict(alphabets).items():
        if type(rate) is not int:
            raise ScaleGridError("scale-grid alphabet rates must be plain ints")
        try:
            frozen_codes = tuple(codes)
        except (TypeError, ValueError) as exc:
            raise ScaleGridError(
                "scale-grid alphabet values must be integer sequences"
            ) from exc
        if not frozen_codes or any(type(code) is not int for code in frozen_codes):
            raise ScaleGridError(
                "scale-grid alphabet values must contain only plain ints"
            )
        frozen_alphabets[rate] = frozen_codes
    if global_scale_real_override is None:
        frozen_override = None
    else:
        if type(global_scale_real_override) not in {int, float}:
            raise ScaleGridError(
                "scale-grid global_scale_real_override must be a plain number"
            )
        frozen_override = float(global_scale_real_override)
        if not math.isfinite(frozen_override) or frozen_override <= 0.0:
            raise ScaleGridError(
                "scale-grid global_scale_real_override must be finite and positive"
            )
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
        "global_scale_real_override": frozen_override,
        "selection_scope": selection_scope,
    }


def _scale_grid_render_recipe(
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
    multipliers: tuple[float, ...],
    global_scale_real_override: float | None,
    selection_scope: str,
    selector_source_sha256: str,
    encoder_source_identity_sha256: str,
    implementation_closure_identity_sha256: str,
) -> dict[str, object]:
    return {
        "schema": SCALE_GRID_RENDER_RECIPE_SCHEMA,
        "family": E2M1_FAMILY,
        "body_rate_q256": body_rate_q256,
        "schedule": list(schedule),
        "layout": layout,
        "alphabets": {
            str(rate): list(codes)
            for rate, codes in sorted(alphabets.items())
        },
        "scale_rule": scale_rule,
        "sb_chunk": sb_chunk,
        "determinism_mode": determinism_mode,
        "tailbite_candidates": tailbite_candidates,
        "backend": backend,
        "point_route": point_route,
        "global_scale_real_override_hex": (
            None
            if global_scale_real_override is None
            else float(global_scale_real_override).hex()
        ),
        "encoder_source_sha256": encoder_source_identity_sha256,
        "implementation_closure_identity_sha256": (
            implementation_closure_identity_sha256
        ),
        "scale_selection": {
            "mode": "e4m3_grid_gated_v1",
            "scope": selection_scope,
            "snap_path": "fixed_global_override_floor_2pow9",
            "multipliers": list(multipliers),
            "multipliers_sha256": _canonical_sha256(list(multipliers)),
            "identity_index": 0,
            "selector_source_sha256": selector_source_sha256,
        },
    }


def _require_wire_matches_render_recipe(
    wire: TrellisWire,
    render_recipe: Mapping[str, object],
) -> None:
    expected_alphabets = {
        int(rate): tuple(cast(Sequence[int], codes))
        for rate, codes in cast(
            Mapping[str, Sequence[int]], render_recipe["alphabets"]
        ).items()
    }
    if (
        wire.family != render_recipe["family"]
        or wire.body_rate_q256 != render_recipe["body_rate_q256"]
        or list(wire.schedule) != render_recipe["schedule"]
        or wire.layout != render_recipe["layout"]
        or dict(wire.alphabets) != expected_alphabets
    ):
        raise ScaleGridError(
            "canonical trellis wire differs from the bound render recipe"
        )


def _validated_weight(weight: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(weight)
    if value.ndim != 2 or not value.is_floating_point():
        raise ScaleGridError("weight must be a rank-two floating tensor")
    if value.shape[0] < 1 or value.shape[1] < 1 or value.shape[1] % E2M1_GROUP_SIZE:
        raise ScaleGridError("weight must be nonempty and group-16 aligned")
    if not bool(torch.isfinite(value).all().item()):
        raise ScaleGridError("weight must be finite")
    return value


def _metric_vector(metric_weight: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    columns = int(weight.shape[1])
    metric = torch.as_tensor(
        metric_weight, dtype=torch.float32, device=weight.device
    ).reshape(-1)
    if metric.numel() != columns:
        raise ScaleGridError(f"metric_weight must have exactly {columns} values")
    if not bool(torch.isfinite(metric).all().item()) or bool((metric < 0).any().item()):
        raise ScaleGridError("metric_weight must be finite and nonnegative")
    if float(metric.sum().item()) <= 0.0:
        raise ScaleGridError("metric_weight must contain positive mass")
    return metric.clamp_min(1.0e-12).contiguous()


def _real_group_scales(weight: torch.Tensor) -> torch.Tensor:
    rows, columns = map(int, weight.shape)
    return (
        weight.detach().to(torch.float32)
        .reshape(rows, columns // E2M1_GROUP_SIZE, E2M1_GROUP_SIZE)
        .abs().amax(dim=-1).clamp_min(1.0e-12) / 6.0
    ).contiguous()


def _resident_scale_codes(
    encoded: EncodedTrellisPlanes,
    *,
    rows: int,
    groups: int,
    device: torch.device,
) -> torch.Tensor:
    codes = encoded.scale_codes
    if (
        codes is None
        or codes.dtype != torch.uint8
        or tuple(codes.shape) != (rows, groups)
        or codes.device != device
    ):
        raise ScaleGridError(
            "E2M1 encoder did not retain its exact resident scale-code plane"
        )
    return codes


def _nearest_e2m1_levels(value: torch.Tensor) -> torch.Tensor:
    levels = torch.tensor(E2M1_LEVELS, dtype=torch.float32, device=value.device)
    insertion = torch.bucketize(value.contiguous(), levels)
    low = levels[(insertion - 1).clamp(0, levels.numel() - 1)]
    high = levels[insertion.clamp(max=levels.numel() - 1)]
    return torch.where((high - value).abs() < (value - low).abs(), high, low)


def propose_e2m1_scale_plane(
    weight: torch.Tensor,
    metric_weight: torch.Tensor,
    *,
    global_scale_real: float,
    identity_scale_codes: torch.Tensor,
    multipliers: Sequence[float] = SCALE_GRID_MULTIPLIERS,
    floor_to_min_positive: bool = True,
) -> ScalePlaneProposal:
    """Propose a legal plane; this RTN score is never an acceptance gate."""

    value = _validated_weight(weight).detach().to(torch.float32)
    metric = _metric_vector(metric_weight, value)
    rows, columns = map(int, value.shape)
    groups = columns // E2M1_GROUP_SIZE
    identity = torch.as_tensor(identity_scale_codes)
    if identity.dtype != torch.uint8 or tuple(identity.shape) != (rows, groups):
        raise ScaleGridError(
            f"identity_scale_codes must be uint8 with shape {(rows, groups)}"
        )
    if identity.device != value.device:
        raise ScaleGridError("identity_scale_codes must reside on the weight device")
    menu = _checked_multipliers(multipliers)
    real_scale = _real_group_scales(value)
    expected_identity = _BOUND_SNAP_E2M1_SCALE_CODES(
        real_scale,
        global_scale_real,
        multiplier=1.0,
        floor_to_min_positive=floor_to_min_positive,
    )
    if not torch.equal(expected_identity, identity):
        raise ScaleGridError(
            "candidate-zero scale codes do not byte-equal the current identity plane"
        )

    grouped = value.reshape(rows, groups, E2M1_GROUP_SIZE)
    grouped_metric = metric.reshape(1, groups, E2M1_GROUP_SIZE)
    best_codes = identity.clone()
    best_indices = torch.zeros((rows, groups), dtype=torch.int64, device=value.device)
    best_cost: torch.Tensor | None = None
    identity_cost: torch.Tensor | None = None
    masked_cells_tensor = torch.zeros((), dtype=torch.int64, device=value.device)
    clipped_cells_tensor = torch.zeros((), dtype=torch.int64, device=value.device)
    code_planes = _BOUND_ITER_SNAPPED_E2M1_SCALE_CODES(
        real_scale,
        global_scale_real,
        menu,
        floor_to_min_positive=floor_to_min_positive,
    )
    global_tensor = torch.tensor(
        float(global_scale_real), dtype=torch.float32, device=value.device
    )
    for index, (multiplier, codes) in enumerate(zip(menu, code_planes, strict=True)):
        decoded_raw = codes.contiguous().view(torch.float8_e4m3fn).to(torch.float32)
        legal = torch.isfinite(decoded_raw) & (decoded_raw > 0)
        if index == 0 and not bool(legal.all().item()):
            raise ScaleGridError("identity scale plane contains an illegal E4M3 cell")
        if index > 0:
            masked_cells_tensor += (~legal).sum()
        safe_decoded = torch.where(
            legal,
            decoded_raw,
            identity.contiguous().view(torch.float8_e4m3fn).to(torch.float32),
        )
        effective = (safe_decoded * global_tensor).clamp_min(1.0e-12)
        normalized = grouped / effective.unsqueeze(-1)
        reconstruction = _nearest_e2m1_levels(normalized) * effective.unsqueeze(-1)
        cost = (
            (grouped.to(torch.float64) - reconstruction.to(torch.float64)).square()
            * grouped_metric.to(torch.float64)
        ).sum(dim=-1)
        cost = torch.where(legal, cost, torch.full_like(cost, float("inf")))
        ratio = real_scale / global_tensor
        if multiplier != 1.0:
            ratio = ratio * multiplier
        clipped_cells_tensor += (ratio > E4M3_MAX).sum()
        if index == 0:
            identity_cost = cost.clone()
            best_cost = cost.clone()
            continue
        assert best_cost is not None
        improves = cost < best_cost
        best_cost = torch.where(improves, cost, best_cost)
        best_codes = torch.where(improves, codes, best_codes)
        best_indices = torch.where(improves, index, best_indices)

    assert best_cost is not None and identity_cost is not None
    if bool((best_cost > identity_cost).any().item()):
        raise AssertionError("proposal generator regressed its RTN identity floor")
    _BOUND_DECODE_E2M1_SCALE_CODES(best_codes, global_scale_real)
    masked_cells = int(masked_cells_tensor.item())
    clipped_cells = int(clipped_cells_tensor.item())
    return ScalePlaneProposal(
        scale_codes=best_codes.contiguous(),
        multiplier_indices=best_indices.contiguous(),
        multipliers=menu,
        group_sse=best_cost.contiguous(),
        identity_group_sse=identity_cost.contiguous(),
        masked_candidate_cells=masked_cells,
        clipped_candidate_cells=clipped_cells,
    )


def _pack_reparse_decode(
    encoded: EncodedTrellisPlanes,
    *,
    body_rate_q256: int,
    schedule: Sequence[int],
    layout: str,
    alphabets: Mapping[int, Sequence[int]],
    device: torch.device,
) -> tuple[bytes, TrellisWire, torch.Tensor, torch.Tensor]:
    wire = _BOUND_PACK_PLANES(
        family=E2M1_FAMILY,
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
    reparsed = _BOUND_TRELLIS_WIRE_FROM_BYTES(blob)
    if reparsed.to_bytes() != blob:
        raise ScaleGridError("canonical scale-grid wire did not reserialize exactly")
    decoded_codes = _BOUND_DECODE_CODES_TORCH(blob, device=device)
    decoded_weight = _BOUND_DECODE_VALUES_TORCH(
        blob, device=device, dtype=torch.float32
    )
    if not torch.equal(
        decoded_weight.to(torch.bfloat16),
        encoded.reconstruction.to(torch.bfloat16),
    ):
        raise ScaleGridError(
            "canonical arm decode differs from the encoder reconstruction"
        )
    return blob, reparsed, decoded_codes, decoded_weight


def score_realized_tiles_fp64(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    metric_weight: torch.Tensor,
) -> torch.Tensor:
    """Score realized bytes at the independent row/superblock scope."""

    value = _validated_weight(weight)
    if tuple(reconstruction.shape) != tuple(value.shape):
        raise ScaleGridError("realized reconstruction shape differs from weight")
    if not bool(torch.isfinite(reconstruction).all().item()):
        raise ScaleGridError("realized reconstruction must be finite")
    rows, columns = map(int, value.shape)
    if columns % SUPERBLOCK_WEIGHTS:
        raise ScaleGridError("diagonal scale-grid gate requires 256-column alignment")
    metric = _metric_vector(metric_weight, value)
    error = value.to(torch.float64) - reconstruction.to(torch.float64)
    weighted = error.square() * metric.to(torch.float64).reshape(1, columns)
    return weighted.reshape(rows, columns // SUPERBLOCK_WEIGHTS, SUPERBLOCK_WEIGHTS).sum(
        dim=-1
    ).contiguous()


def _splice_encoded_planes(
    identity: EncodedTrellisPlanes,
    candidate: EncodedTrellisPlanes,
    candidate_wins: torch.Tensor,
) -> EncodedTrellisPlanes:
    if identity.u_bits.shape != candidate.u_bits.shape:
        raise ScaleGridError("two scale-grid arms have different plane shapes")
    rows, columns = map(int, identity.u_bits.shape)
    blocks = columns // SUPERBLOCK_WEIGHTS
    if tuple(candidate_wins.shape) != (rows, blocks) or candidate_wins.dtype != torch.bool:
        raise ScaleGridError("candidate_wins must be bool at row/superblock scope")
    if candidate_wins.device != identity.u_bits.device:
        raise ScaleGridError("candidate_wins must reside with the encoded planes")
    if candidate.u_bits.device != identity.u_bits.device:
        raise ScaleGridError("two scale-grid arms must reside on one device")
    if identity.global_scale_real != candidate.global_scale_real:
        raise ScaleGridError("scale-grid arms changed the immutable global scale")

    column_mask = candidate_wins.repeat_interleave(SUPERBLOCK_WEIGHTS, dim=1)
    group_mask = candidate_wins.repeat_interleave(
        SUPERBLOCK_WEIGHTS // E2M1_GROUP_SIZE, dim=1
    )
    groups = columns // E2M1_GROUP_SIZE
    identity_scale = _resident_scale_codes(
        identity, rows=rows, groups=groups, device=identity.u_bits.device
    )
    candidate_scale = _resident_scale_codes(
        candidate, rows=rows, groups=groups, device=identity.u_bits.device
    )
    final_scale = torch.where(group_mask, candidate_scale, identity_scale)
    return EncodedTrellisPlanes(
        reconstruction=torch.where(
            column_mask, candidate.reconstruction, identity.reconstruction
        ).contiguous(),
        u_bits=torch.where(column_mask, candidate.u_bits, identity.u_bits).contiguous(),
        point_indices=torch.where(
            column_mask, candidate.point_indices, identity.point_indices
        ).contiguous(),
        bypass_codes=torch.where(
            column_mask, candidate.bypass_codes, identity.bypass_codes
        ).contiguous(),
        scale_blob=final_scale.detach().cpu().contiguous().numpy().tobytes(),
        global_scale_real=identity.global_scale_real,
        scale_codes=final_scale.contiguous(),
    )


def _callable_descriptor(value: object) -> dict[str, str]:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise RuntimeError("scale-grid closure member lacks a stable callable identity")
    return {"module": module, "qualname": qualname}


def _live_scale_grid_callables() -> dict[str, object]:
    return {
        "EncodedTrellisPlanes": EncodedTrellisPlanes,
        "TrellisWire": TrellisWire,
        "TrellisWire.from_bytes": TrellisWire.from_bytes.__func__,
        "_canonical_sha256": _canonical_sha256,
        "_checked_multipliers": _checked_multipliers,
        "_checked_render_inputs": _checked_render_inputs,
        "_metric_vector": _metric_vector,
        "_pack_reparse_decode": _pack_reparse_decode,
        "_real_group_scales": _real_group_scales,
        "_require_wire_matches_render_recipe": _require_wire_matches_render_recipe,
        "_resident_scale_codes": _resident_scale_codes,
        "_scale_grid_render_recipe": _scale_grid_render_recipe,
        "_splice_encoded_planes": _splice_encoded_planes,
        "_tensor_sha256": _tensor_sha256,
        "_validated_weight": _validated_weight,
        "decode_codes_torch": decode_codes_torch,
        "decode_e2m1_scale_codes": decode_e2m1_scale_codes,
        "decode_values_torch": decode_values_torch,
        "encode_trellis_planes": encode_trellis_planes,
        "iter_snapped_e2m1_scale_codes": iter_snapped_e2m1_scale_codes,
        "pack_planes": pack_planes,
        "propose_e2m1_scale_plane": propose_e2m1_scale_plane,
        "require_encoder_source_unchanged": require_encoder_source_unchanged,
        "score_realized_tiles_fp64": score_realized_tiles_fp64,
        "snap_e2m1_scale_codes": snap_e2m1_scale_codes,
        "validate_scale_grid_receipt": validate_scale_grid_receipt,
    }


_IMPORTED_SCALE_GRID_CALLABLES = tuple(_live_scale_grid_callables().items())
_BOUND_ENCODE_TRELLIS_PLANES = encode_trellis_planes
_BOUND_DECODE_CODES_TORCH = decode_codes_torch
_BOUND_DECODE_E2M1_SCALE_CODES = decode_e2m1_scale_codes
_BOUND_DECODE_VALUES_TORCH = decode_values_torch
_BOUND_ITER_SNAPPED_E2M1_SCALE_CODES = iter_snapped_e2m1_scale_codes
_BOUND_PACK_PLANES = pack_planes
_BOUND_PACK_REPARSE_DECODE = _pack_reparse_decode
_BOUND_PROPOSE_E2M1_SCALE_PLANE = propose_e2m1_scale_plane
_BOUND_REQUIRE_WIRE_MATCHES_RENDER_RECIPE = _require_wire_matches_render_recipe
_BOUND_SCORE_REALIZED_TILES_FP64 = score_realized_tiles_fp64
_BOUND_SPLICE_ENCODED_PLANES = _splice_encoded_planes
_BOUND_SNAP_E2M1_SCALE_CODES = snap_e2m1_scale_codes
_BOUND_TRELLIS_WIRE_FROM_BYTES = TrellisWire.from_bytes


def require_scale_grid_callable_closure_unchanged() -> str:
    """Refuse module-level callable substitution at the research boundary."""

    expected = dict(_IMPORTED_SCALE_GRID_CALLABLES)
    live = _live_scale_grid_callables()
    changed = sorted(
        name for name, original in expected.items() if live.get(name) is not original
    )
    if changed:
        raise ScaleGridError(
            "trellis scale-grid callable closure changed since module import; "
            f"refusing substituted callables {changed}"
        )
    return _canonical_sha256({
        name: _callable_descriptor(value)
        for name, value in _IMPORTED_SCALE_GRID_CALLABLES
    })


def scale_grid_implementation_closure() -> Mapping[str, object]:
    """Return the import-bound source/callable closure named by v2 receipts."""

    callable_descriptors = {
        name: _callable_descriptor(value)
        for name, value in _IMPORTED_SCALE_GRID_CALLABLES
    }
    body: dict[str, object] = {
        "schema": SCALE_GRID_IMPLEMENTATION_CLOSURE_SCHEMA,
        "sources": {
            "selector": {
                "path": "prismaquant/trellis_scale_grid.py",
                "sha256": _IMPORTED_SCALE_GRID_SOURCE_SHA256,
            },
            "encoder": {
                "path": "prismaquant/trellis_encoder.py",
                "sha256": _IMPORTED_ENCODER_SOURCE_SHA256,
            },
            "wire": {
                "path": "prismaquant/trellis_wire.py",
                "sha256": _IMPORTED_WIRE_SOURCE_SHA256,
            },
            "formats": {
                "path": "prismaquant/trellis_formats.py",
                "sha256": _IMPORTED_FORMATS_SOURCE_SHA256,
            },
        },
        "callables": callable_descriptors,
        "callable_identity_sha256": _canonical_sha256(callable_descriptors),
    }
    return {**body, "identity_sha256": _canonical_sha256(body)}


_BOUND_PUBLIC_REQUIRE_IMPLEMENTATION = require_scale_grid_implementation_unchanged
_BOUND_PUBLIC_IMPLEMENTATION_CLOSURE = scale_grid_implementation_closure
_BOUND_PUBLIC_VALIDATE_RECEIPT = validate_scale_grid_receipt
_BOUND_REQUIRE_SELECTOR_SOURCE = require_scale_grid_source_unchanged
_BOUND_REQUIRE_ENCODER_SOURCE = require_scale_grid_encoder_source_unchanged
_BOUND_CURRENT_SELECTOR_SOURCE = _current_scale_grid_source_sha256
_BOUND_CURRENT_WIRE_SOURCE = _current_wire_source_sha256
_BOUND_CURRENT_FORMATS_SOURCE = _current_formats_source_sha256
_BOUND_LIVE_SCALE_GRID_CALLABLES = _live_scale_grid_callables
_BOUND_CALLABLE_DESCRIPTOR = _callable_descriptor


def _scale_grid_execution_gateway(
) -> tuple[str, str, Mapping[str, object]]:
    """Authenticate the helpers used to authenticate the executable closure."""

    guarded_helpers = {
        "_scale_grid_execution_gateway": (
            _scale_grid_execution_gateway,
            _BOUND_SCALE_GRID_EXECUTION_GATEWAY,
        ),
        "require_scale_grid_implementation_unchanged": (
            require_scale_grid_implementation_unchanged,
            _BOUND_PUBLIC_REQUIRE_IMPLEMENTATION,
        ),
        "scale_grid_implementation_closure": (
            scale_grid_implementation_closure,
            _BOUND_PUBLIC_IMPLEMENTATION_CLOSURE,
        ),
        "validate_scale_grid_receipt": (
            validate_scale_grid_receipt,
            _BOUND_PUBLIC_VALIDATE_RECEIPT,
        ),
        "require_scale_grid_source_unchanged": (
            require_scale_grid_source_unchanged,
            _BOUND_REQUIRE_SELECTOR_SOURCE,
        ),
        "require_scale_grid_encoder_source_unchanged": (
            require_scale_grid_encoder_source_unchanged,
            _BOUND_REQUIRE_ENCODER_SOURCE,
        ),
        "_current_scale_grid_source_sha256": (
            _current_scale_grid_source_sha256,
            _BOUND_CURRENT_SELECTOR_SOURCE,
        ),
        "_current_wire_source_sha256": (
            _current_wire_source_sha256,
            _BOUND_CURRENT_WIRE_SOURCE,
        ),
        "_current_formats_source_sha256": (
            _current_formats_source_sha256,
            _BOUND_CURRENT_FORMATS_SOURCE,
        ),
        "_live_scale_grid_callables": (
            _live_scale_grid_callables,
            _BOUND_LIVE_SCALE_GRID_CALLABLES,
        ),
        "_callable_descriptor": (
            _callable_descriptor,
            _BOUND_CALLABLE_DESCRIPTOR,
        ),
    }
    substituted = sorted(
        name for name, (live, bound) in guarded_helpers.items() if live is not bound
    )
    expected_callables = dict(_IMPORTED_SCALE_GRID_CALLABLES)
    execution_aliases = {
        "_BOUND_ENCODE_TRELLIS_PLANES": (
            _BOUND_ENCODE_TRELLIS_PLANES,
            expected_callables["encode_trellis_planes"],
        ),
        "_BOUND_DECODE_CODES_TORCH": (
            _BOUND_DECODE_CODES_TORCH,
            expected_callables["decode_codes_torch"],
        ),
        "_BOUND_DECODE_E2M1_SCALE_CODES": (
            _BOUND_DECODE_E2M1_SCALE_CODES,
            expected_callables["decode_e2m1_scale_codes"],
        ),
        "_BOUND_DECODE_VALUES_TORCH": (
            _BOUND_DECODE_VALUES_TORCH,
            expected_callables["decode_values_torch"],
        ),
        "_BOUND_ITER_SNAPPED_E2M1_SCALE_CODES": (
            _BOUND_ITER_SNAPPED_E2M1_SCALE_CODES,
            expected_callables["iter_snapped_e2m1_scale_codes"],
        ),
        "_BOUND_PACK_PLANES": (
            _BOUND_PACK_PLANES,
            expected_callables["pack_planes"],
        ),
        "_BOUND_PACK_REPARSE_DECODE": (
            _BOUND_PACK_REPARSE_DECODE,
            expected_callables["_pack_reparse_decode"],
        ),
        "_BOUND_PROPOSE_E2M1_SCALE_PLANE": (
            _BOUND_PROPOSE_E2M1_SCALE_PLANE,
            expected_callables["propose_e2m1_scale_plane"],
        ),
        "_BOUND_REQUIRE_WIRE_MATCHES_RENDER_RECIPE": (
            _BOUND_REQUIRE_WIRE_MATCHES_RENDER_RECIPE,
            expected_callables["_require_wire_matches_render_recipe"],
        ),
        "_BOUND_SCORE_REALIZED_TILES_FP64": (
            _BOUND_SCORE_REALIZED_TILES_FP64,
            expected_callables["score_realized_tiles_fp64"],
        ),
        "_BOUND_SPLICE_ENCODED_PLANES": (
            _BOUND_SPLICE_ENCODED_PLANES,
            expected_callables["_splice_encoded_planes"],
        ),
        "_BOUND_SNAP_E2M1_SCALE_CODES": (
            _BOUND_SNAP_E2M1_SCALE_CODES,
            expected_callables["snap_e2m1_scale_codes"],
        ),
        "_BOUND_TRELLIS_WIRE_FROM_BYTES": (
            getattr(_BOUND_TRELLIS_WIRE_FROM_BYTES, "__func__", None),
            expected_callables["TrellisWire.from_bytes"],
        ),
        "_BOUND_PUBLIC_IMPLEMENTATION_CLOSURE": (
            _BOUND_PUBLIC_IMPLEMENTATION_CLOSURE,
            scale_grid_implementation_closure,
        ),
        "_BOUND_PUBLIC_VALIDATE_RECEIPT": (
            _BOUND_PUBLIC_VALIDATE_RECEIPT,
            validate_scale_grid_receipt,
        ),
        "_BOUND_REQUIRE_SELECTOR_SOURCE": (
            _BOUND_REQUIRE_SELECTOR_SOURCE,
            require_scale_grid_source_unchanged,
        ),
        "_BOUND_REQUIRE_ENCODER_SOURCE": (
            _BOUND_REQUIRE_ENCODER_SOURCE,
            require_scale_grid_encoder_source_unchanged,
        ),
        "_BOUND_CURRENT_SELECTOR_SOURCE": (
            _BOUND_CURRENT_SELECTOR_SOURCE,
            _current_scale_grid_source_sha256,
        ),
        "_BOUND_CURRENT_WIRE_SOURCE": (
            _BOUND_CURRENT_WIRE_SOURCE,
            _current_wire_source_sha256,
        ),
        "_BOUND_CURRENT_FORMATS_SOURCE": (
            _BOUND_CURRENT_FORMATS_SOURCE,
            _current_formats_source_sha256,
        ),
        "_BOUND_LIVE_SCALE_GRID_CALLABLES": (
            _BOUND_LIVE_SCALE_GRID_CALLABLES,
            _live_scale_grid_callables,
        ),
    }
    substituted.extend(
        name for name, (live, original) in execution_aliases.items()
        if live is not original
    )
    substituted.sort()
    if substituted:
        raise ScaleGridError(
            "trellis scale-grid gateway changed since module import; refusing "
            f"substituted helpers {substituted}"
        )
    selector = _BOUND_REQUIRE_SELECTOR_SOURCE()
    encoder = _BOUND_REQUIRE_ENCODER_SOURCE()
    current_wire = _BOUND_CURRENT_WIRE_SOURCE()
    if not hmac.compare_digest(current_wire, _IMPORTED_WIRE_SOURCE_SHA256):
        raise ScaleGridError(
            "trellis wire source changed since scale-grid import; refusing "
            "to execute or publish a mixed implementation closure"
        )
    current_formats = _BOUND_CURRENT_FORMATS_SOURCE()
    if not hmac.compare_digest(current_formats, _IMPORTED_FORMATS_SOURCE_SHA256):
        raise ScaleGridError(
            "trellis format source changed since scale-grid import; refusing "
            "to execute or publish a mixed implementation closure"
        )
    live_callables = _BOUND_LIVE_SCALE_GRID_CALLABLES()
    changed = sorted(
        name
        for name, original in expected_callables.items()
        if live_callables.get(name) is not original
    )
    if changed:
        raise ScaleGridError(
            "trellis scale-grid callable closure changed since module import; "
            f"refusing substituted callables {changed}"
        )
    closure = _BOUND_PUBLIC_IMPLEMENTATION_CLOSURE()
    return selector, encoder, closure


_BOUND_SCALE_GRID_EXECUTION_GATEWAY = _scale_grid_execution_gateway


def encode_e2m1_scale_grid_two_arm(
    weight: torch.Tensor,
    metric_weight: torch.Tensor,
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
    multipliers: Sequence[float] = SCALE_GRID_MULTIPLIERS,
    global_scale_real_override: float | None = None,
    selection_scope: str = "row_superblock",
) -> ScaleGridSelection:
    """Run the two full encodes and prove exact diagonal non-regression."""

    if _scale_grid_execution_gateway is not _BOUND_SCALE_GRID_EXECUTION_GATEWAY:
        raise ScaleGridError("trellis scale-grid execution gateway was substituted")
    selector_source, encoder_source, implementation_closure = (
        _BOUND_SCALE_GRID_EXECUTION_GATEWAY()
    )
    checked = _checked_render_inputs(
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
        global_scale_real_override=global_scale_real_override,
        selection_scope=selection_scope,
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
    global_scale_real_override = cast(
        float | None, checked["global_scale_real_override"]
    )
    selection_scope = cast(str, checked["selection_scope"])
    if selection_scope != "row_superblock":
        raise ScaleGridError(
            "diagonal scale-grid selection scope must be row_superblock; "
            "BlockLDL requires the separate row_factor_group gate"
        )
    value = _validated_weight(weight)
    metric = _metric_vector(metric_weight, value)
    rows, columns = map(int, value.shape)
    if columns % SUPERBLOCK_WEIGHTS:
        raise ScaleGridError("two-arm trellis scale grid requires 256-column alignment")
    menu = _checked_multipliers(multipliers)
    render_recipe = _scale_grid_render_recipe(
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
        multipliers=menu,
        global_scale_real_override=global_scale_real_override,
        selection_scope=selection_scope,
        selector_source_sha256=selector_source,
        encoder_source_identity_sha256=encoder_source,
        implementation_closure_identity_sha256=cast(
            str, implementation_closure["identity_sha256"]
        ),
    )
    encoder_kwargs = {
        "family": E2M1_FAMILY,
        "schedule": schedule,
        "alphabets": alphabets,
        "scale_rule": scale_rule,
        "sb_chunk": sb_chunk,
        "determinism_mode": determinism_mode,
        "tailbite_candidates": tailbite_candidates,
        "backend": backend,
        "point_route": point_route,
    }
    identity = _BOUND_ENCODE_TRELLIS_PLANES(
        value,
        metric,
        **encoder_kwargs,
        global_scale_real_override=global_scale_real_override,
    )
    identity_blob, identity_wire, _identity_codes, identity_decoded = (
        _BOUND_PACK_REPARSE_DECODE(
            identity,
            body_rate_q256=body_rate_q256,
            schedule=schedule,
            layout=layout,
            alphabets=alphabets,
            device=value.device,
        )
    )
    _BOUND_REQUIRE_WIRE_MATCHES_RENDER_RECIPE(identity_wire, render_recipe)
    identity_scale_codes = _resident_scale_codes(
        identity,
        rows=rows,
        groups=columns // E2M1_GROUP_SIZE,
        device=value.device,
    )
    proposal = _BOUND_PROPOSE_E2M1_SCALE_PLANE(
        value,
        metric,
        global_scale_real=identity.global_scale_real,
        identity_scale_codes=identity_scale_codes,
        multipliers=menu,
        floor_to_min_positive=True,
    )
    candidate = _BOUND_ENCODE_TRELLIS_PLANES(
        value,
        metric,
        **encoder_kwargs,
        global_scale_real_override=identity.global_scale_real,
        scale_plane_override=proposal.scale_codes,
    )
    if candidate.global_scale_real != identity.global_scale_real:
        raise ScaleGridError("candidate arm changed the immutable global scale")
    candidate_blob, candidate_wire, _candidate_codes, candidate_decoded = (
        _BOUND_PACK_REPARSE_DECODE(
            candidate,
            body_rate_q256=body_rate_q256,
            schedule=schedule,
            layout=layout,
            alphabets=alphabets,
            device=value.device,
        )
    )
    _BOUND_REQUIRE_WIRE_MATCHES_RENDER_RECIPE(candidate_wire, render_recipe)
    if candidate_wire.global_scale_real != identity_wire.global_scale_real:
        raise ScaleGridError("candidate arm changed the immutable global scale")
    invariant_fields = (
        "family", "layout", "rows", "columns", "body_rate_q256", "schedule",
        "block_offsets_bits", "alphabets", "row_body_bits",
        "row_stride_bytes",
    )
    if any(
        getattr(identity_wire, field) != getattr(candidate_wire, field)
        for field in invariant_fields
    ):
        raise ScaleGridError("candidate arm changed a fixed wire/recipe field")
    if len(identity_blob) != len(candidate_blob):
        raise ScaleGridError("candidate arm changed the exact wire byte length")

    identity_cost = _BOUND_SCORE_REALIZED_TILES_FP64(
        value, identity_decoded, metric
    )
    candidate_cost = _BOUND_SCORE_REALIZED_TILES_FP64(
        value, candidate_decoded, metric
    )
    candidate_wins = candidate_cost < identity_cost
    final_encoded = _BOUND_SPLICE_ENCODED_PLANES(
        identity, candidate, candidate_wins
    )
    final_blob, final_wire, final_codes, final_decoded = _BOUND_PACK_REPARSE_DECODE(
        final_encoded,
        body_rate_q256=body_rate_q256,
        schedule=schedule,
        layout=layout,
        alphabets=alphabets,
        device=value.device,
    )
    _BOUND_REQUIRE_WIRE_MATCHES_RENDER_RECIPE(final_wire, render_recipe)
    if len(final_blob) != len(identity_blob):
        raise ScaleGridError("spliced wire changed the exact byte length")
    if final_wire.global_scale_real != identity_wire.global_scale_real:
        raise ScaleGridError("spliced wire changed the immutable global scale")
    final_cost = _BOUND_SCORE_REALIZED_TILES_FP64(value, final_decoded, metric)
    exact_minimum = torch.minimum(identity_cost, candidate_cost)
    if not torch.equal(final_cost, exact_minimum):
        raise ScaleGridError("spliced Cf is not exactly min(C0, C1) per tile")
    if bool((final_cost > identity_cost).any().item()):
        raise ScaleGridError("spliced Cf exceeded the identity C0")

    no_wins = not bool(candidate_wins.any().item())
    if no_wins and final_blob != identity_blob:
        raise ScaleGridError("no-win scale-grid result is not byte-identical to identity")
    expected_memory_reconstruction = torch.where(
        candidate_wins.repeat_interleave(SUPERBLOCK_WEIGHTS, dim=1),
        candidate.reconstruction,
        identity.reconstruction,
    )
    if not torch.equal(
        final_decoded.to(torch.bfloat16),
        expected_memory_reconstruction.to(torch.bfloat16),
    ):
        raise ScaleGridError(
            "same-byte final decode differs from the spliced encoder reconstruction"
        )

    final_encoded = EncodedTrellisPlanes(
        reconstruction=final_decoded.contiguous(),
        u_bits=final_encoded.u_bits,
        point_indices=final_encoded.point_indices,
        bypass_codes=final_encoded.bypass_codes,
        scale_blob=final_encoded.scale_blob,
        global_scale_real=final_encoded.global_scale_real,
        scale_codes=final_encoded.scale_codes,
    )
    ties = candidate_cost == identity_cost
    receipt_body: dict[str, object] = {
        "schema": SCALE_GRID_SCHEMA,
        "status": "two_full_arms_canonical_decode_gate_verified",
        "scope": "research_only_unregistered",
        "mode": "e4m3_grid_gated_v1",
        "selection_scope": "row_superblock",
        "snap_path": "fixed_global_override_floor_2pow9",
        "multipliers": list(menu),
        "multipliers_sha256": _canonical_sha256(list(menu)),
        "multiplier_count": len(menu),
        "identity_index": 0,
        "selector_source_sha256": selector_source,
        "encoder_source_sha256": encoder_source,
        "implementation_closure": implementation_closure,
        "render_recipe": render_recipe,
        "render_recipe_identity_sha256": _canonical_sha256(render_recipe),
        "global_scale_real_hex": float(identity_wire.global_scale_real).hex(),
        "pricing": {
            "body_rate_q256": int(body_rate_q256),
            "scale_plane_rate_q256": SCALE_PLANE_RATE_Q256,
            "identity_wire_bytes": len(identity_blob),
            "candidate_wire_bytes": len(candidate_blob),
            "final_wire_bytes": len(final_blob),
            "wire_byte_delta": 0,
            "delta_bpw_q256": 0,
        },
        "arms": {
            "identity": {
                "objective_fp64_hex": float(identity_cost.sum().item()).hex(),
                "wire_sha256": hashlib.sha256(identity_blob).hexdigest(),
                "decoded_weight_sha256": _tensor_sha256(identity_decoded),
            },
            "candidate": {
                "objective_fp64_hex": float(candidate_cost.sum().item()).hex(),
                "wire_sha256": hashlib.sha256(candidate_blob).hexdigest(),
                "decoded_weight_sha256": _tensor_sha256(candidate_decoded),
                "masked_candidate_cells": proposal.masked_candidate_cells,
                "clipped_candidate_cells": proposal.clipped_candidate_cells,
            },
        },
        "final": {
            "objective_fp64_hex": float(final_cost.sum().item()).hex(),
            "wire_sha256": hashlib.sha256(final_blob).hexdigest(),
            "tiles": int(final_cost.numel()),
            "candidate_win_tiles": int(candidate_wins.sum().item()),
            "identity_win_tiles": int((candidate_cost > identity_cost).sum().item()),
            "tie_tiles": int(ties.sum().item()),
            "max_tile_regression": 0.0,
            "tile_sse_sha256": _tensor_sha256(final_cost),
        },
        "proof": {
            "full_identity_encode": True,
            "full_candidate_encode": True,
            "canonical_pack_reparse_decode_each_arm": True,
            "cf_exact_minimum": True,
            "cf_le_c0": True,
            "same_length": True,
            "no_win_byte_identical": no_wins,
            "identity_candidate_byte_equal": True,
            "strict_candidate_wins_tie_identity": True,
        },
        "format_registry_entries_created": 0,
        "runtime_pin_changed": False,
        "production_contract_changed": False,
        "producer_eligible": False,
    }
    receipt = {**receipt_body, "identity_sha256": _canonical_sha256(receipt_body)}
    _BOUND_PUBLIC_VALIDATE_RECEIPT(receipt)
    _BOUND_SCALE_GRID_EXECUTION_GATEWAY()
    return ScaleGridSelection(
        encoded_planes=final_encoded,
        wire_bytes=final_blob,
        decoded_codes=final_codes.contiguous(),
        decoded_weight=final_decoded.contiguous(),
        identity_wire_bytes=identity_blob,
        identity_tile_sse=identity_cost,
        candidate_tile_sse=candidate_cost,
        final_tile_sse=final_cost,
        candidate_wins=candidate_wins.contiguous(),
        proposal=proposal,
        receipt=receipt,
    )


def require_scale_grid_selection_replay(
    expected: ScaleGridSelection,
    weight: torch.Tensor,
    metric_weight: torch.Tensor,
    **recipe: object,
) -> ScaleGridSelection:
    """Rerun a selection and require exact artifact-and-receipt identity.

    ``validate_scale_grid_receipt`` validates a closed, self-consistent JSON
    record; it cannot authenticate hashes without their artifacts. This
    replay boundary is the authoritative check: it reruns both complete arms
    from the caller-supplied tensors and recipe, then compares every retained
    plane, wire, score, proposal, winner, and receipt exactly.
    """

    if not isinstance(expected, ScaleGridSelection):
        raise ScaleGridError("expected replay artifact must be ScaleGridSelection")
    receipt = _BOUND_PUBLIC_VALIDATE_RECEIPT(expected.receipt)
    required_recipe_fields = {
        "body_rate_q256", "schedule", "layout", "alphabets", "scale_rule",
        "sb_chunk", "determinism_mode", "tailbite_candidates", "backend",
        "point_route",
    }
    optional_recipe_fields = {
        "multipliers", "global_scale_real_override", "selection_scope",
    }
    recipe_fields = set(recipe)
    if (
        not required_recipe_fields.issubset(recipe_fields)
        or not recipe_fields.issubset(required_recipe_fields | optional_recipe_fields)
    ):
        raise ScaleGridError(
            "scale-grid replay recipe fields differ from the closed schema"
        )
    menu = _checked_multipliers(cast(
        Sequence[float], recipe.get("multipliers", SCALE_GRID_MULTIPLIERS)
    ))
    checked = _checked_render_inputs(
        body_rate_q256=recipe["body_rate_q256"],
        schedule=recipe["schedule"],
        layout=recipe["layout"],
        alphabets=recipe["alphabets"],
        scale_rule=recipe["scale_rule"],
        sb_chunk=recipe["sb_chunk"],
        determinism_mode=recipe["determinism_mode"],
        tailbite_candidates=recipe["tailbite_candidates"],
        backend=recipe["backend"],
        point_route=recipe["point_route"],
        global_scale_real_override=recipe.get("global_scale_real_override"),
        selection_scope=recipe.get("selection_scope", "row_superblock"),
    )
    selector_source, encoder_source, implementation_closure = (
        _BOUND_SCALE_GRID_EXECUTION_GATEWAY()
    )
    replay_recipe = _scale_grid_render_recipe(
        body_rate_q256=cast(int, checked["body_rate_q256"]),
        schedule=cast(tuple[int, ...], checked["schedule"]),
        layout=cast(str, checked["layout"]),
        alphabets=cast(dict[int, tuple[int, ...]], checked["alphabets"]),
        scale_rule=cast(str, checked["scale_rule"]),
        sb_chunk=cast(int, checked["sb_chunk"]),
        determinism_mode=cast(str, checked["determinism_mode"]),
        tailbite_candidates=cast(int, checked["tailbite_candidates"]),
        backend=cast(str, checked["backend"]),
        point_route=cast(str, checked["point_route"]),
        multipliers=menu,
        global_scale_real_override=cast(
            float | None, checked["global_scale_real_override"]
        ),
        selection_scope=cast(str, checked["selection_scope"]),
        selector_source_sha256=selector_source,
        encoder_source_identity_sha256=encoder_source,
        implementation_closure_identity_sha256=cast(
            str, implementation_closure["identity_sha256"]
        ),
    )
    if (
        replay_recipe != receipt["render_recipe"]
        or _canonical_sha256(replay_recipe)
        != receipt["render_recipe_identity_sha256"]
    ):
        raise ScaleGridError("scale-grid replay recipe identity mismatch")
    replay = encode_e2m1_scale_grid_two_arm(
        weight,
        metric_weight,
        **checked,
        multipliers=menu,
    )

    def require_tensor(name: str, left: torch.Tensor, right: torch.Tensor) -> None:
        if not torch.equal(left, right):
            raise ScaleGridError(f"scale-grid replay differs at {name}")

    def require_optional_tensor(
        name: str,
        left: torch.Tensor | None,
        right: torch.Tensor | None,
    ) -> None:
        if left is None or right is None:
            if left is not right:
                raise ScaleGridError(f"scale-grid replay differs at {name}")
            return
        require_tensor(name, left, right)

    left_encoded = expected.encoded_planes
    right_encoded = replay.encoded_planes
    for field in ("reconstruction", "u_bits", "point_indices", "bypass_codes"):
        require_tensor(
            f"encoded_planes.{field}",
            getattr(left_encoded, field),
            getattr(right_encoded, field),
        )
    if (
        left_encoded.scale_blob != right_encoded.scale_blob
        or left_encoded.global_scale_real != right_encoded.global_scale_real
    ):
        raise ScaleGridError("scale-grid replay differs at encoded scale context")
    require_optional_tensor(
        "encoded_planes.scale_codes",
        left_encoded.scale_codes,
        right_encoded.scale_codes,
    )
    for field in (
        "decoded_codes", "decoded_weight", "identity_tile_sse",
        "candidate_tile_sse", "final_tile_sse", "candidate_wins",
    ):
        require_tensor(field, getattr(expected, field), getattr(replay, field))
    if (
        expected.wire_bytes != replay.wire_bytes
        or expected.identity_wire_bytes != replay.identity_wire_bytes
    ):
        raise ScaleGridError("scale-grid replay differs at canonical wire bytes")

    left_proposal = expected.proposal
    right_proposal = replay.proposal
    for field in (
        "scale_codes", "multiplier_indices", "group_sse", "identity_group_sse"
    ):
        require_tensor(
            f"proposal.{field}",
            getattr(left_proposal, field),
            getattr(right_proposal, field),
        )
    if (
        left_proposal.multipliers != right_proposal.multipliers
        or left_proposal.masked_candidate_cells
        != right_proposal.masked_candidate_cells
        or left_proposal.clipped_candidate_cells
        != right_proposal.clipped_candidate_cells
    ):
        raise ScaleGridError("scale-grid replay differs at proposal evidence")
    if expected.receipt != replay.receipt:
        raise ScaleGridError("scale-grid replay differs at receipt semantics")
    return replay


def select_e2m1_scale_grid(*_args: object, **_kwargs: object) -> ScaleGridSelection:
    """Refuse the retired independent RTN-only group selector."""

    raise ScaleGridError(
        "independent per-group RTN scale selection is retired because shaped "
        "trellis positions are path-coupled; use encode_e2m1_scale_grid_two_arm"
    )


__all__ = [
    "E2M1_LEVELS",
    "SCALE_GRID_MULTIPLIERS",
    "SCALE_GRID_IMPLEMENTATION_CLOSURE_SCHEMA",
    "SCALE_GRID_RENDER_RECIPE_SCHEMA",
    "SCALE_GRID_SCHEMA",
    "SCALE_PLANE_RATE_Q256",
    "ScaleGridError",
    "ScaleGridSelection",
    "ScalePlaneProposal",
    "encode_e2m1_scale_grid_two_arm",
    "propose_e2m1_scale_plane",
    "require_scale_grid_encoder_source_unchanged",
    "require_scale_grid_callable_closure_unchanged",
    "require_scale_grid_implementation_unchanged",
    "require_scale_grid_selection_replay",
    "require_scale_grid_source_unchanged",
    "scale_grid_multipliers",
    "scale_grid_implementation_closure",
    "scale_grid_source_sha256",
    "score_realized_tiles_fp64",
    "select_e2m1_scale_grid",
    "validate_scale_grid_receipt",
]
