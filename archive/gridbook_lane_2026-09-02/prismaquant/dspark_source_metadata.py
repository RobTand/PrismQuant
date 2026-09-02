"""Producer-owned metadata bridge for DeepSeek-V4 DSpark source weights.

DeepSeek-V4-Flash stores the three DSpark draft stages under the physical
checkpoint namespace ``mtp.{0,1,2}.*``.  vLLM 0.26 constructs those stages as
ordinary DeepSeek decoder blocks with prefixes ``model.layers.{L+stage}.*``
(``L == num_hidden_layers``), then its DSpark loader maps the physical names
into the registered three-layer module list while streaming the checkpoint.

That gives one artifact three relevant namespaces:

* physical checkpoint: ``mtp.0.attn.wq_a``;
* quant-method construction: ``model.layers.43.attn.fused_wqa_wkv``;
* registered parameter: ``model.layers.0.attn.fused_wqa_wkv``.

Gridbook must dispatch source-format methods in the *construction* namespace;
the architecture's own loader must continue to receive the *physical* names.
This module describes exactly that producer-side overlay.  It never rewrites a
tensor and deliberately has no vLLM or Gridbook import.

The overlay is intentionally narrow.  It activates only for a DeepSeek-V4
config carrying the released DSpark marker and exactly three complete source
stages.  An unfamiliar scale-bearing leaf, a missing stage/projection/expert,
or a geometry mismatch fails before export instead of emitting metadata that
could select the wrong native decoder.
"""
from __future__ import annotations

from copy import deepcopy
import ctypes
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Protocol


DSPARK_OVERLAY_SCHEMA = "prismaquant.dspark_source_overlay.v1"
DSPARK_TARGET_BRIDGE_SCHEMA = "gridbook.dspark-target-bridge.v1"
DSPARK_STAGE_COUNT = 3
MXFP4_SOURCE_FORMAT = "MXFP4_SOURCE"
FP8_BLOCK_UE8M0_SOURCE_FORMAT = "FP8_BLOCK_UE8M0_SOURCE"

_ROUTED_EXPERT_RE = re.compile(
    r"^mtp\.(?P<stage>\d+)\.ffn\.experts\.(?P<expert>\d+)\."
    r"(?P<leaf>w1|w2|w3)$"
)
_PHYSICAL_RE = re.compile(r"^mtp\.(?P<stage>\d+)\.(?P<rest>.+)$")
_CONSTRUCTION_RE = re.compile(
    r"^model[.]layers[.](?P<layer>\d+)[.](?P<rest>.+)$"
)
_CB_EXPERT_RECIPE_RE = re.compile(
    r"^ffn[.]experts[.](?P<expert>\d+)[.]"
    r"(?P<projection>gate_proj|up_proj|down_proj)$"
)
_CB_DIRECT_OUTPUT_RE = re.compile(
    r"^(?:attn[.](?:wq_a|wkv|wq_b|wo_a|wo_b)|"
    r"ffn[.]shared_experts[.](?:w1|w2|w3))$"
)
_CB_PACKED_EXPERT_OUTPUTS = frozenset({
    "ffn.experts.gate_up_proj",
    "ffn.experts.down_proj",
})
_DSPARK_CB_HYBRID_TAILS = (
    "attn.wkv",
    "attn.wo_b",
    "attn.wq_a",
    "attn.wq_b",
    "ffn.experts.down_proj",
    "ffn.experts.gate_up_proj",
    "ffn.shared_experts.w1",
    "ffn.shared_experts.w2",
    "ffn.shared_experts.w3",
)
_CB_EXPERT_SOURCE_PROJECTION = {
    "gate_proj": "w1",
    "up_proj": "w3",
    "down_proj": "w2",
}
_CB_EXPERT_OUTPUT_PROJECTION = {
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
    "down_proj": "down_proj",
}

_FP8_REST_TO_CONSTRUCTION = {
    "attn.wq_a": "attn.fused_wqa_wkv",
    "attn.wkv": "attn.fused_wqa_wkv",
    "attn.wq_b": "attn.wq_b",
    "attn.wo_a": "attn.wo_a",
    "attn.wo_b": "attn.wo_b",
    "ffn.shared_experts.w1": "ffn.shared_experts.gate_up_proj",
    "ffn.shared_experts.w3": "ffn.shared_experts.gate_up_proj",
    "ffn.shared_experts.w2": "ffn.shared_experts.down_proj",
}
_EXPECTED_STAGE_UNITS = frozenset({
    "attn.fused_wqa_wkv",
    "attn.wq_b",
    "attn.wo_a",
    "attn.wo_b",
    "ffn.shared_experts.gate_up_proj",
    "ffn.shared_experts.down_proj",
    "ffn.experts",
})


class _Skeleton(Protocol):
    def keys(self): ...
    def get_shape(self, name: str) -> tuple[int, ...]: ...
    def get_dtype(self, name: str): ...


@dataclass(frozen=True)
class DSparkSourceOverlay:
    """The complete metadata-only overlay for one three-stage checkpoint."""

    num_hidden_layers: int
    n_mtp_layers: int
    physical_targets: Mapping[str, str]
    construction_units: Mapping[str, str]
    physical_to_construction_unit: Mapping[str, str]

    def provenance(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for format_name in self.physical_targets.values():
            counts[format_name] = counts.get(format_name, 0) + 1
        return {
            "schema": DSPARK_OVERLAY_SCHEMA,
            "physical_namespace": "mtp.{stage}",
            "construction_namespace": (
                "model.layers.{num_hidden_layers+stage}"
            ),
            "num_hidden_layers": self.num_hidden_layers,
            "n_mtp_layers": self.n_mtp_layers,
            "physical_stage_ids": list(range(self.n_mtp_layers)),
            "construction_layer_ids": [
                self.num_hidden_layers + stage
                for stage in range(self.n_mtp_layers)
            ],
            "physical_target_counts": dict(sorted(counts.items())),
            "construction_unit_count": len(self.construction_units),
            "tensor_bytes_rewritten": 0,
        }


@dataclass(frozen=True)
class _TensorContract:
    """One exact tensor in the released DSpark checkpoint layout."""

    shape: tuple[int, ...]
    dtypes: frozenset[str]


def _dtype_code(value: Any) -> str:
    text = str(value)
    aliases = {
        "torch.int8": "I8",
        "torch.uint8": "U8",
        "torch.bfloat16": "BF16",
        "torch.float32": "F32",
        "torch.float8_e4m3fn": "F8_E4M3",
        "torch.float8_e8m0fnu": "F8_E8M0",
    }
    return aliases.get(text, text.upper())


def _config_is_released_dspark(config: Mapping[str, Any]) -> bool:
    model_type = str(config.get("model_type", "")).replace("-", "_").lower()
    architectures = tuple(str(value) for value in config.get("architectures", ()))
    return (
        model_type == "deepseek_v4"
        and any(name == "DeepseekV4ForCausalLM" for name in architectures)
        and config.get("dspark_block_size") is not None
    )


def _dspark_cb_topology(config: Mapping[str, Any]) -> tuple[int, int]:
    """Return the released DSpark ``(body layers, draft stages)`` topology.

    The source checkpoint predates the emitted ``n_mtp_layers`` field, so its
    three-stage fact is recoverable from ``dspark_target_layer_ids``.  A
    materialized sidecar carries ``n_mtp_layers`` directly.  If both spellings
    are present they must agree; neither is guessed from observed target names.
    """

    if not isinstance(config, Mapping):
        raise ValueError("DSpark CB namespace requires a source config object")

    def integer(name: str, raw: object) -> int:
        if isinstance(raw, bool):
            raise ValueError(f"DSpark config {name} must be an integer")
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"DSpark config {name} must be an integer"
            ) from exc
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"DSpark config {name} must be an integer")
        if isinstance(raw, str) and raw.strip() != str(value):
            raise ValueError(f"DSpark config {name} must be an integer")
        return value

    try:
        body_layers = integer(
            "num_hidden_layers", config["num_hidden_layers"]
        )
    except KeyError as exc:
        raise ValueError(
            "DSpark CB namespace requires num_hidden_layers"
        ) from exc
    if body_layers <= 0:
        raise ValueError(
            "DSpark config num_hidden_layers must be positive, got "
            f"{body_layers}"
        )

    target_ids_raw = config.get("dspark_target_layer_ids")
    target_ids: tuple[int, ...] | None = None
    if target_ids_raw is not None:
        if isinstance(target_ids_raw, (str, bytes)):
            raise ValueError(
                "DSpark config dspark_target_layer_ids must be an integer list"
            )
        try:
            target_ids = tuple(
                integer("dspark_target_layer_ids entry", raw)
                for raw in target_ids_raw
            )
        except TypeError as exc:
            raise ValueError(
                "DSpark config dspark_target_layer_ids must be an integer list"
            ) from exc
        if (
            len(target_ids) != DSPARK_STAGE_COUNT
            or len(set(target_ids)) != DSPARK_STAGE_COUNT
            or any(target < 0 or target >= body_layers for target in target_ids)
        ):
            raise ValueError(
                "DSpark CB namespace requires exactly three distinct target "
                f"layer ids in [0, {body_layers}); got {target_ids}"
            )

    stages_raw = config.get("n_mtp_layers")
    if stages_raw is None:
        if target_ids is None:
            raise ValueError(
                "DSpark CB namespace requires n_mtp_layers or the released "
                "three-entry dspark_target_layer_ids"
            )
        stages = len(target_ids)
    else:
        stages = integer("n_mtp_layers", stages_raw)
    if stages != DSPARK_STAGE_COUNT:
        raise ValueError(
            "DSpark CB namespace supports exactly three draft stages, got "
            f"{stages}"
        )
    if target_ids is not None and len(target_ids) != stages:
        raise ValueError(
            "DSpark n_mtp_layers disagrees with dspark_target_layer_ids"
        )
    return body_layers, stages


def _dspark_physical_parts(
    name: str,
    *,
    n_mtp_layers: int,
) -> tuple[int, str]:
    if not isinstance(name, str):
        raise ValueError("DSpark physical target must be a string")
    match = _PHYSICAL_RE.fullmatch(name)
    if match is None or any(not part for part in name.split(".")):
        raise ValueError(
            f"{name!r}: expected physical DSpark base mtp.{{stage}}.<tail>"
        )
    stage = int(match.group("stage"))
    if not 0 <= stage < n_mtp_layers:
        raise ValueError(
            f"{name!r}: DSpark stage must be in [0, {n_mtp_layers})"
        )
    rest = match.group("rest")
    serialized_suffixes = (
        ".weight",
        ".scale",
        ".cb_qweight",
        ".weight_scale",
        ".weight_global_scale",
        ".input_global_scale",
    )
    if rest.endswith(serialized_suffixes):
        raise ValueError(
            f"{name!r}: DSpark namespace helpers require a tensor base, not "
            "a serialized tensor leaf"
        )
    return stage, rest


def _dspark_cb_output_tail(rest: str, *, where: str) -> str:
    if rest == "main_proj":
        raise ValueError(
            f"{where}: main_proj is DSpark glue, not a contracted CB target"
        )
    if _CB_DIRECT_OUTPUT_RE.fullmatch(rest) or rest in _CB_PACKED_EXPERT_OUTPUTS:
        return rest
    raise ValueError(
        f"{where}: unsupported DSpark CB output tail {rest!r}"
    )


def dspark_cb_physical_source_for_recipe_target(
    recipe_target: str,
    config: Mapping[str, Any],
) -> str:
    """Resolve one CB recipe target to the physical source tensor base.

    Dense attention and shared-expert recipes already use checkpoint spelling.
    Routed-expert recipes use the allocator's logical projection vocabulary;
    the released checkpoint stores those members as ``w1/w3/w2``.
    """

    _body_layers, stages = _dspark_cb_topology(config)
    stage, rest = _dspark_physical_parts(
        recipe_target, n_mtp_layers=stages
    )
    expert = _CB_EXPERT_RECIPE_RE.fullmatch(rest)
    if expert is not None:
        projection = _CB_EXPERT_SOURCE_PROJECTION[expert.group("projection")]
        return (
            f"mtp.{stage}.ffn.experts.{int(expert.group('expert'))}."
            f"{projection}"
        )
    if _CB_DIRECT_OUTPUT_RE.fullmatch(rest):
        return recipe_target
    if rest in _CB_PACKED_EXPERT_OUTPUTS:
        raise ValueError(
            f"{recipe_target!r}: a packed DSpark CB output has no single "
            "checkpoint source member"
        )
    _dspark_cb_output_tail(rest, where=recipe_target)
    raise AssertionError("unreachable DSpark CB source classification")


def dspark_cb_physical_output_for_recipe_target(
    recipe_target: str,
    config: Mapping[str, Any],
) -> str:
    """Resolve a direct/member recipe target to its physical CB output base."""

    _body_layers, stages = _dspark_cb_topology(config)
    stage, rest = _dspark_physical_parts(
        recipe_target, n_mtp_layers=stages
    )
    expert = _CB_EXPERT_RECIPE_RE.fullmatch(rest)
    if expert is not None:
        projection = _CB_EXPERT_OUTPUT_PROJECTION[expert.group("projection")]
        return f"mtp.{stage}.ffn.experts.{projection}"
    _dspark_cb_output_tail(rest, where=recipe_target)
    return recipe_target


def dspark_cb_construction_target_for_physical_output(
    physical_output: str,
    config: Mapping[str, Any],
) -> str:
    """Map a serialized physical CB base to vLLM construction namespace."""

    body_layers, stages = _dspark_cb_topology(config)
    stage, rest = _dspark_physical_parts(
        physical_output, n_mtp_layers=stages
    )
    _dspark_cb_output_tail(rest, where=physical_output)
    return f"model.layers.{body_layers + stage}.{rest}"


def dspark_cb_physical_output_for_construction_target(
    construction_target: str,
    config: Mapping[str, Any],
) -> str:
    """Invert the same-tail DSpark construction mapping, fail closed."""

    body_layers, stages = _dspark_cb_topology(config)
    if not isinstance(construction_target, str):
        raise ValueError("DSpark construction target must be a string")
    match = _CONSTRUCTION_RE.fullmatch(construction_target)
    if match is None or any(not part for part in construction_target.split(".")):
        raise ValueError(
            f"{construction_target!r}: expected DSpark construction target "
            "model.layers.{L+stage}.<tail>"
        )
    layer = int(match.group("layer"))
    stage = layer - body_layers
    if not 0 <= stage < stages:
        raise ValueError(
            f"{construction_target!r}: construction layer must be in "
            f"[{body_layers}, {body_layers + stages})"
        )
    rest = match.group("rest")
    _dspark_cb_output_tail(rest, where=construction_target)
    return f"mtp.{stage}.{rest}"


def dspark_cb_expected_physical_targets(
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the exact hybrid sidecar CB bases in physical namespace.

    DSpark ``wo_a`` is a grouped BMM, not an ordinary dense Linear.  Gridbook's
    generic CB Linear method does not implement that algebra, so all three
    ``wo_a`` bases remain on their released source-FP8 W8A16 route.  The CB
    surface is consequently nine physical bases per stage (27 total for the
    released three-stage topology).
    """

    _body_layers, stages = _dspark_cb_topology(config)
    return tuple(sorted(
        f"mtp.{stage}.{tail}"
        for stage in range(stages)
        for tail in _DSPARK_CB_HYBRID_TAILS
    ))


def dspark_cb_source_passthrough_mapping(
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Return exact physical -> construction W8A16 routes for the sidecar."""

    body_layers, stages = _dspark_cb_topology(config)
    physical = ["mtp.0.main_proj"] + [
        f"mtp.{stage}.attn.wo_a" for stage in range(stages)
    ]
    mapping: dict[str, str] = {}
    for target in sorted(physical):
        construction = dspark_construction_unit_for_physical_target(
            target,
            num_hidden_layers=body_layers,
            n_mtp_layers=stages,
        )
        if construction is None:
            raise AssertionError(
                f"internal DSpark hybrid source target has no construction "
                f"mapping: {target}"
            )
        mapping[target] = construction
    return mapping


def _unique_string_names(raw: object, *, where: str) -> tuple[str, ...]:
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"{where} must be a sequence of target names")
    try:
        values = tuple(raw)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{where} must be a sequence of target names") from exc
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{where} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{where} must not contain duplicate target names")
    return values


def _execution_contract_target_names(
    execution_contract: Mapping[str, Any],
) -> tuple[str, ...]:
    if not isinstance(execution_contract, Mapping):
        raise ValueError("DSpark target bridge execution contract must be an object")
    if "target_names" not in execution_contract:
        raise ValueError(
            "DSpark target bridge execution contract has no target_names"
        )
    names = _unique_string_names(
        execution_contract["target_names"],
        where="execution-contract target_names",
    )
    try:
        target_count = int(execution_contract["target_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "DSpark target bridge execution contract requires target_count"
        ) from exc
    if target_count != len(names):
        raise ValueError(
            "execution-contract target_count disagrees with target_names: "
            f"{target_count} != {len(names)}"
        )
    return names


def _expected_dspark_target_bridge(
    config: Mapping[str, Any],
    *,
    contracted_cb_construction_targets: object,
    activation_execution_contract: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    construction = _unique_string_names(
        () if contracted_cb_construction_targets is None
        else contracted_cb_construction_targets,
        where="contracted CB construction targets",
    )
    if activation_execution_contract is None:
        if construction:
            raise ValueError(
                "contracted DSpark CB targets require an activation execution "
                "contract"
            )
        return None
    physical = _execution_contract_target_names(activation_execution_contract)
    if not construction or not physical:
        raise ValueError(
            "DSpark target bridge must be wholly present or absent; contracted "
            "targets and execution-contract target_names must both be nonempty"
        )

    body_layers, stages = _dspark_cb_topology(config)
    mapping = {
        target: dspark_cb_physical_output_for_construction_target(target, config)
        for target in construction
    }
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(
            "DSpark construction-to-physical mapping must be one-to-one"
        )
    physical_set = set(physical)
    if set(mapping.values()) != physical_set:
        raise ValueError(
            "DSpark bridge physical values must exactly equal execution-contract "
            "target_names: missing="
            f"{sorted(physical_set - set(mapping.values()))[:8]}, extra="
            f"{sorted(set(mapping.values()) - physical_set)[:8]}"
        )

    observed_stages = {
        _dspark_physical_parts(name, n_mtp_layers=stages)[0]
        for name in physical
    }
    expected_stages = set(range(stages))
    if observed_stages != expected_stages:
        raise ValueError(
            "DSpark target bridge requires the complete three-stage set: "
            f"expected={sorted(expected_stages)}, got={sorted(observed_stages)}"
        )
    return {
        "schema": DSPARK_TARGET_BRIDGE_SCHEMA,
        "num_hidden_layers": body_layers,
        "n_mtp_layers": stages,
        "construction_to_physical": dict(sorted(mapping.items())),
    }


def build_dspark_target_bridge(
    config: Mapping[str, Any],
    *,
    contracted_cb_construction_targets: object = (),
    activation_execution_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build the optional Gridbook DSpark activation-namespace bridge.

    Absence is legal only when both the contracted target set and activation
    execution contract are absent.  A present record covers all three stages,
    maps exact same-tail CB targets one-to-one, and never admits ``main_proj``.
    """

    return _expected_dspark_target_bridge(
        config,
        contracted_cb_construction_targets=contracted_cb_construction_targets,
        activation_execution_contract=activation_execution_contract,
    )


def validate_dspark_target_bridge(
    bridge: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    *,
    contracted_cb_construction_targets: object = (),
    activation_execution_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate and canonicalize a top-level ``dspark_target_bridge`` record."""

    expected = _expected_dspark_target_bridge(
        config,
        contracted_cb_construction_targets=contracted_cb_construction_targets,
        activation_execution_contract=activation_execution_contract,
    )
    if expected is None:
        if bridge is not None:
            raise ValueError(
                "dspark_target_bridge must be absent without contracted targets"
            )
        return None
    if not isinstance(bridge, Mapping):
        raise ValueError("dspark_target_bridge must be an object")
    required = {
        "schema",
        "num_hidden_layers",
        "n_mtp_layers",
        "construction_to_physical",
    }
    if set(bridge) != required:
        raise ValueError(
            "dspark_target_bridge keys must be exactly "
            f"{sorted(required)}, got {sorted(str(key) for key in bridge)}"
        )
    if bridge.get("schema") != DSPARK_TARGET_BRIDGE_SCHEMA:
        raise ValueError(
            "dspark_target_bridge must use schema "
            f"{DSPARK_TARGET_BRIDGE_SCHEMA!r}"
        )
    mapping = bridge.get("construction_to_physical")
    if not isinstance(mapping, Mapping):
        raise ValueError(
            "dspark_target_bridge.construction_to_physical must be an object"
        )
    if any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or not key
        or not value
        for key, value in mapping.items()
    ):
        raise ValueError(
            "dspark_target_bridge construction mapping requires nonempty "
            "string keys and values"
        )
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(
            "dspark_target_bridge construction mapping must be one-to-one"
        )
    if dict(bridge) != expected:
        raise ValueError(
            "dspark_target_bridge does not exactly match the source topology, "
            "contracted CB targets, and execution-contract target_names"
        )
    return expected


def dspark_construction_unit_for_physical_target(
    physical_target: str,
    *,
    num_hidden_layers: int,
    n_mtp_layers: int = DSPARK_STAGE_COUNT,
) -> str | None:
    """Map one scale-bearing physical base to vLLM's construction prefix.

    ``None`` means the name is not a source-quantized DSpark target this
    contract knows.  Callers that are classifying an observed scale-bearing
    tensor must turn that ``None`` into a refusal; artifact readers may use it
    as an ordinary "not this namespace" result.
    """

    expert = _ROUTED_EXPERT_RE.fullmatch(str(physical_target))
    if expert is not None:
        stage = int(expert.group("stage"))
        if 0 <= stage < int(n_mtp_layers):
            return (
                f"model.layers.{int(num_hidden_layers) + stage}.ffn.experts"
            )
        return None

    match = _PHYSICAL_RE.fullmatch(str(physical_target))
    if match is None:
        return None
    stage = int(match.group("stage"))
    if not 0 <= stage < int(n_mtp_layers):
        return None
    rest = match.group("rest")
    if rest == "main_proj" and stage == 0:
        return "model.main_proj"
    construction_rest = _FP8_REST_TO_CONSTRUCTION.get(rest)
    if construction_rest is None:
        return None
    return (
        f"model.layers.{int(num_hidden_layers) + stage}."
        f"{construction_rest}"
    )


def parse_dspark_overlay_provenance(
    quant_config: Mapping[str, Any],
) -> tuple[int, int] | None:
    """Return ``(num_hidden_layers, n_mtp_layers)`` for a stamped artifact."""

    raw = (quant_config.get("provenance") or {}).get("dspark_source_overlay")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or raw.get("schema") != DSPARK_OVERLAY_SCHEMA:
        raise ValueError(
            "provenance.dspark_source_overlay must use schema "
            f"{DSPARK_OVERLAY_SCHEMA!r}"
        )
    try:
        num_hidden_layers = int(raw["num_hidden_layers"])
        n_mtp_layers = int(raw["n_mtp_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "provenance.dspark_source_overlay requires integer "
            "num_hidden_layers and n_mtp_layers"
        ) from exc
    if num_hidden_layers <= 0 or n_mtp_layers != DSPARK_STAGE_COUNT:
        raise ValueError(
            "provenance.dspark_source_overlay describes unsupported topology: "
            f"num_hidden_layers={num_hidden_layers}, "
            f"n_mtp_layers={n_mtp_layers}"
        )
    return num_hidden_layers, n_mtp_layers


def apply_dspark_overlay_to_model_config(
    config: Mapping[str, Any], overlay: DSparkSourceOverlay | None
) -> dict[str, Any]:
    """Copy ``config`` and stamp the stage count only when an overlay exists."""

    out = dict(config)
    if overlay is None:
        return out
    existing = out.get("n_mtp_layers")
    if existing is not None and int(existing) != overlay.n_mtp_layers:
        raise ValueError(
            "source config n_mtp_layers disagrees with the three physical "
            f"DSpark stages: {existing!r} != {overlay.n_mtp_layers}"
        )
    out["n_mtp_layers"] = overlay.n_mtp_layers
    return out


def _released_dspark_tensor_layout(
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    q_lora_rank: int,
    o_groups: int,
    o_lora_rank: int,
    moe_intermediate_size: int,
    num_experts: int,
    vocab_size: int,
    markov_rank: int,
    target_layer_count: int,
    hc_mult: int,
    fp8_block: tuple[int, int],
) -> tuple[dict[str, _TensorContract], dict[str, str]]:
    """Build the closed 4,705-tensor released layout from model config.

    The returned source-target map is keyed by physical Linear base.  The
    tensor map additionally covers every unscaled tensor vLLM's DSpark loader
    consumes: six BF16 glue matrices, fourteen BF16 norms, and twenty-seven
    F32 sink/router/hyper-connection/head tensors.
    """

    tensors: dict[str, _TensorContract] = {}
    physical_targets: dict[str, str] = {}

    def add_tensor(
        name: str, shape: tuple[int, ...], *dtypes: str
    ) -> None:
        if name in tensors:
            raise AssertionError(f"duplicate internal DSpark tensor {name}")
        tensors[name] = _TensorContract(
            tuple(int(value) for value in shape), frozenset(dtypes)
        )

    def add_fp8(base: str, shape: tuple[int, int]) -> None:
        add_tensor(base + ".weight", shape, "F8_E4M3")
        add_tensor(
            base + ".scale",
            tuple(
                -(-dimension // block_size)
                for dimension, block_size in zip(shape, fp8_block)
            ),
            "F8_E8M0",
        )
        physical_targets[base] = FP8_BLOCK_UE8M0_SOURCE_FORMAT

    def add_mxfp4(base: str, logical_shape: tuple[int, int]) -> None:
        out_features, in_features = logical_shape
        add_tensor(
            base + ".weight", (out_features, in_features // 2), "I8", "U8"
        )
        add_tensor(
            base + ".scale", (out_features, in_features // 32), "F8_E8M0"
        )
        physical_targets[base] = MXFP4_SOURCE_FORMAT

    attention_shapes = {
        "attn.wq_a": (q_lora_rank, hidden_size),
        "attn.wkv": (head_dim, hidden_size),
        "attn.wq_b": (num_heads * head_dim, q_lora_rank),
        "attn.wo_a": (
            o_groups * o_lora_rank,
            num_heads * head_dim // o_groups,
        ),
        "attn.wo_b": (hidden_size, o_groups * o_lora_rank),
    }
    shared_expert_shapes = {
        "ffn.shared_experts.w1": (moe_intermediate_size, hidden_size),
        "ffn.shared_experts.w2": (hidden_size, moe_intermediate_size),
        "ffn.shared_experts.w3": (moe_intermediate_size, hidden_size),
    }
    expert_shapes = {
        "w1": (moe_intermediate_size, hidden_size),
        "w2": (hidden_size, moe_intermediate_size),
        "w3": (moe_intermediate_size, hidden_size),
    }
    mix_hc = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * hidden_size

    for stage in range(DSPARK_STAGE_COUNT):
        stage_prefix = f"mtp.{stage}."
        for rest, shape in (*attention_shapes.items(), *shared_expert_shapes.items()):
            add_fp8(stage_prefix + rest, shape)
        for expert_id in range(num_experts):
            for leaf, shape in expert_shapes.items():
                add_mxfp4(
                    f"{stage_prefix}ffn.experts.{expert_id}.{leaf}", shape
                )

        add_tensor(stage_prefix + "attn.q_norm.weight", (q_lora_rank,), "BF16")
        add_tensor(stage_prefix + "attn.kv_norm.weight", (head_dim,), "BF16")
        add_tensor(stage_prefix + "attn_norm.weight", (hidden_size,), "BF16")
        add_tensor(stage_prefix + "ffn_norm.weight", (hidden_size,), "BF16")

        add_tensor(stage_prefix + "ffn.gate.weight", (num_experts, hidden_size), "BF16")
        add_tensor(stage_prefix + "ffn.gate.bias", (num_experts,), "F32")
        add_tensor(stage_prefix + "attn.attn_sink", (num_heads,), "F32")
        for branch in ("attn", "ffn"):
            add_tensor(
                stage_prefix + f"hc_{branch}_fn", (mix_hc, hc_dim), "F32"
            )
            add_tensor(
                stage_prefix + f"hc_{branch}_base", (mix_hc,), "F32"
            )
            add_tensor(stage_prefix + f"hc_{branch}_scale", (3,), "F32")

    add_fp8(
        "mtp.0.main_proj",
        (hidden_size, hidden_size * target_layer_count),
    )
    add_tensor("mtp.0.main_norm.weight", (hidden_size,), "BF16")
    add_tensor("mtp.2.norm.weight", (hidden_size,), "BF16")
    add_tensor(
        "mtp.2.confidence_head.proj.weight",
        (1, hidden_size + markov_rank),
        "BF16",
    )
    for leaf in ("markov_w1", "markov_w2"):
        add_tensor(
            f"mtp.2.markov_head.{leaf}.weight",
            (vocab_size, markov_rank),
            "BF16",
        )
    add_tensor("mtp.2.hc_head_fn", (hc_mult, hc_dim), "F32")
    add_tensor("mtp.2.hc_head_base", (hc_mult,), "F32")
    add_tensor("mtp.2.hc_head_scale", (1,), "F32")

    return tensors, physical_targets


def _validate_released_dspark_tensor_layout(
    skeleton: _Skeleton,
    expected: Mapping[str, _TensorContract],
    physical_targets: Mapping[str, str],
) -> None:
    """Validate the closed released layout without reading tensor payloads."""

    keys = {str(name) for name in skeleton.keys() if str(name).startswith("mtp.")}
    expected_keys = set(expected)
    missing = expected_keys - keys
    unknown = keys - expected_keys

    # Name pair corruption deserves the most actionable message: this is the
    # only route whose scale plane would otherwise be silently discarded.
    for base in sorted(physical_targets):
        weight_present = base + ".weight" in keys
        scale_present = base + ".scale" in keys
        if weight_present != scale_present:
            raise ValueError(
                f"{base}: DSpark source target requires both .weight and .scale"
            )
    if missing or unknown:
        raise ValueError(
            "released DSpark essential tensor layout mismatch: "
            f"missing={sorted(missing)[:8]}, unknown={sorted(unknown)[:8]}"
        )

    for name, contract in expected.items():
        actual_dtype = _dtype_code(skeleton.get_dtype(name))
        actual_shape = tuple(int(value) for value in skeleton.get_shape(name))
        if actual_dtype not in contract.dtypes or actual_shape != contract.shape:
            expected_dtype = (
                next(iter(contract.dtypes))
                if len(contract.dtypes) == 1
                else sorted(contract.dtypes)
            )
            raise ValueError(
                f"{name}: released DSpark tensor must be {expected_dtype} "
                f"{contract.shape}, got {actual_dtype} {actual_shape}"
            )


def discover_dspark_source_overlay(
    skeleton: _Skeleton,
    config: Mapping[str, Any],
) -> DSparkSourceOverlay | None:
    """Discover and validate the released three-stage source-format payload.

    Non-DSpark configs return ``None`` without inspecting shapes.  A positive
    DSpark marker is a promise, so any incomplete/unknown payload raises.
    """

    if not _config_is_released_dspark(config):
        return None
    try:
        num_hidden_layers = int(config["num_hidden_layers"])
        num_experts = int(config["n_routed_experts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "released DSpark metadata requires positive num_hidden_layers "
            "and n_routed_experts"
        ) from exc
    if num_hidden_layers <= 0 or num_experts <= 0:
        raise ValueError(
            "released DSpark metadata requires positive num_hidden_layers "
            f"and n_routed_experts, got {num_hidden_layers}/{num_experts}"
        )
    try:
        block_size = int(config["dspark_block_size"])
        markov_rank = int(config["dspark_markov_rank"])
        target_layer_ids = tuple(
            int(value) for value in config["dspark_target_layer_ids"]
        )
        nextn_layers = int(config["num_nextn_predict_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "released DSpark config requires integer dspark_block_size, "
            "dspark_markov_rank, dspark_target_layer_ids, and "
            "num_nextn_predict_layers"
        ) from exc
    if (
        block_size <= 0
        or markov_rank <= 0
        or len(target_layer_ids) != DSPARK_STAGE_COUNT
        or nextn_layers != 1
    ):
        raise ValueError(
            "released DSpark topology requires positive block/Markov sizes, "
            "three target layer ids, and num_nextn_predict_layers=1; got "
            f"block={block_size}, markov_rank={markov_rank}, "
            f"targets={target_layer_ids}, nextn={nextn_layers}"
        )
    if (
        len(set(target_layer_ids)) != len(target_layer_ids)
        or any(
            layer_id < 0 or layer_id >= num_hidden_layers
            for layer_id in target_layer_ids
        )
    ):
        raise ValueError(
            "released DSpark target layer ids must be distinct and each in "
            f"[0, {num_hidden_layers}); got {target_layer_ids}"
        )
    existing_layers = config.get("n_mtp_layers")
    if existing_layers is not None and int(existing_layers) != DSPARK_STAGE_COUNT:
        raise ValueError(
            "source config n_mtp_layers disagrees with the released DSpark "
            f"three-stage contract: {existing_layers!r}"
        )
    if str(config.get("expert_dtype", "")).lower() not in {
        "fp4", "mxfp4", "mx_fp4"
    }:
        raise ValueError(
            "released DSpark source overlay requires expert_dtype='fp4'"
        )
    quantization = config.get("quantization_config") or {}
    if not isinstance(quantization, Mapping):
        raise ValueError("released DSpark quantization_config must be an object")
    raw_block = quantization.get("weight_block_size")
    raw_scale_format = quantization.get("scale_fmt")
    is_gridbook_artifact = (
        quantization.get("quant_method") == "gridbook"
        and quantization.get("config_file") == "quant_config.json"
    )
    # A source checkpoint states the block geometry here.  A PrismaQuant
    # artifact has replaced that source record with its Gridbook pointer, but
    # the exact geometry remains independently provable from every serialized
    # weight/scale pair below.  Accept only that known pointer form; an absent
    # or unfamiliar source contract still fails closed.
    if raw_block is None and is_gridbook_artifact:
        block = (128, 128)
    else:
        try:
            block = tuple(int(value) for value in (raw_block or ()))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "released DSpark weight_block_size must contain integers"
            ) from exc
    if block != (128, 128):
        raise ValueError(
            "released DSpark block-FP8 overlay requires "
            f"weight_block_size [128, 128], got {block!r}"
        )
    if raw_scale_format is None and is_gridbook_artifact:
        # The required F8_E8M0 dtype is checked on every physical scale plane.
        pass
    elif str(raw_scale_format).lower() not in {"ue8m0", "e8m0"}:
        raise ValueError(
            "released DSpark source overlay requires an E8M0 scale_fmt"
        )

    required_dimension_names = (
        "hidden_size",
        "num_attention_heads",
        "head_dim",
        "q_lora_rank",
        "o_groups",
        "o_lora_rank",
        "moe_intermediate_size",
        "n_shared_experts",
        "vocab_size",
    )
    try:
        dimensions = {
            name: int(config[name]) for name in required_dimension_names
        }
        hc_mult = int(config.get("hc_mult", 4))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "released DSpark config requires integer layout dimensions "
            f"{required_dimension_names} and optional hc_mult"
        ) from exc
    non_positive = {
        name: value
        for name, value in (*dimensions.items(), ("hc_mult", hc_mult))
        if value <= 0
    }
    if non_positive:
        raise ValueError(
            "released DSpark layout dimensions must be positive; got "
            f"{non_positive}"
        )
    if dimensions["n_shared_experts"] != 1:
        raise ValueError(
            "released DSpark layout requires n_shared_experts=1, got "
            f"{dimensions['n_shared_experts']}"
        )
    if (
        dimensions["hidden_size"] % 32
        or dimensions["moe_intermediate_size"] % 32
        or (
            dimensions["num_attention_heads"] * dimensions["head_dim"]
        )
        % dimensions["o_groups"]
    ):
        raise ValueError(
            "released DSpark MXFP4/output-group dimensions are not divisible "
            "by their serialized packing groups"
        )

    tensor_layout, physical_targets = _released_dspark_tensor_layout(
        hidden_size=dimensions["hidden_size"],
        num_heads=dimensions["num_attention_heads"],
        head_dim=dimensions["head_dim"],
        q_lora_rank=dimensions["q_lora_rank"],
        o_groups=dimensions["o_groups"],
        o_lora_rank=dimensions["o_lora_rank"],
        moe_intermediate_size=dimensions["moe_intermediate_size"],
        num_experts=num_experts,
        vocab_size=dimensions["vocab_size"],
        markov_rank=markov_rank,
        target_layer_count=len(target_layer_ids),
        hc_mult=hc_mult,
        fp8_block=block,
    )
    _validate_released_dspark_tensor_layout(
        skeleton, tensor_layout, physical_targets
    )

    construction_units: dict[str, str] = {}
    physical_to_unit: dict[str, str] = {}
    for base, format_name in sorted(physical_targets.items()):
        construction = dspark_construction_unit_for_physical_target(
            base,
            num_hidden_layers=num_hidden_layers,
            n_mtp_layers=DSPARK_STAGE_COUNT,
        )
        if construction is None:  # expert path is expected to map too
            raise AssertionError(f"internal DSpark namespace gap for {base}")
        previous = construction_units.setdefault(construction, format_name)
        if previous != format_name:
            raise ValueError(
                f"{construction}: fused DSpark unit mixes {previous} and "
                f"{format_name} source formats"
            )
        physical_to_unit[base] = construction

    for stage in range(DSPARK_STAGE_COUNT):
        stage_prefix = f"model.layers.{num_hidden_layers + stage}."
        observed_units = {
            unit[len(stage_prefix):]
            for unit in construction_units
            if unit.startswith(stage_prefix)
        }
        if observed_units != _EXPECTED_STAGE_UNITS:
            raise ValueError(
                f"DSpark construction layer {num_hidden_layers + stage} "
                f"requires units {sorted(_EXPECTED_STAGE_UNITS)}, got "
                f"{sorted(observed_units)}"
            )

    if construction_units.get("model.main_proj") != FP8_BLOCK_UE8M0_SOURCE_FORMAT:
        raise ValueError("released DSpark stage 0 requires block-FP8 model.main_proj")
    if len(physical_targets) != DSPARK_STAGE_COUNT * num_experts * 3 + 25:
        raise AssertionError(
            "unexpected released DSpark source target count: "
            f"{len(physical_targets)}"
        )
    if len(construction_units) != DSPARK_STAGE_COUNT * 7 + 1:
        raise AssertionError(
            "unexpected released DSpark construction-unit count: "
            f"{len(construction_units)}"
        )

    return DSparkSourceOverlay(
        num_hidden_layers=num_hidden_layers,
        n_mtp_layers=DSPARK_STAGE_COUNT,
        physical_targets=dict(sorted(physical_targets.items())),
        construction_units=dict(sorted(construction_units.items())),
        physical_to_construction_unit=dict(sorted(physical_to_unit.items())),
    )


class ArtifactHeaderSkeleton:
    """The discovery protocol backed only by safetensors JSON headers."""

    def __init__(self, artifact_dir: str | Path):
        from prismaquant.artifact_completeness import read_artifact_header

        self.artifact_dir = Path(artifact_dir)
        self.header = read_artifact_header(self.artifact_dir)

    def keys(self):
        return self.header.keys()

    def get_shape(self, name: str) -> tuple[int, ...]:
        return tuple(int(value) for value in self.header[name]["shape"])

    def get_dtype(self, name: str) -> str:
        return str(self.header[name]["dtype"])


def discover_dspark_source_overlay_from_artifact(
    artifact_dir: str | Path,
) -> DSparkSourceOverlay | None:
    """Discover the overlay from an existing artifact without tensor reads."""

    root = Path(artifact_dir)
    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"DSpark artifact has no config.json: {root}")
    config = json.loads(config_path.read_text())
    if not isinstance(config, Mapping):
        raise ValueError(f"{config_path}: model config must be an object")
    return discover_dspark_source_overlay(
        ArtifactHeaderSkeleton(root), config
    )


def _explicit_target(name: str) -> str:
    """Match the producer's existing anchored compressed-tensors spelling."""

    return f"re:^{name.replace('.', '[.]')}$"


def _target_claims(target: object, unit: str) -> bool:
    """Conservatively decide whether one config target can claim ``unit``."""

    text = str(target)
    if text.startswith("re:"):
        try:
            return re.match(text[len("re:"):], unit) is not None
        except re.error as exc:
            raise ValueError(f"invalid config-group regex target {text!r}") from exc
    return text == unit or unit.startswith(text + ".")


def _literal_explicit_target(target: object) -> str | None:
    """Decode only the exact anchored spelling emitted by this producer."""

    text = str(target)
    if not text.startswith("re:^") or not text.endswith("$"):
        return None
    candidate = text[len("re:^"):-1].replace("[.]", ".")
    return candidate if _explicit_target(candidate) == text else None


def _source_group_template(format_name: str) -> dict[str, Any]:
    # Lazy import keeps header discovery torch-free.  The template is producer
    # owned; duplicating its wire id or layout here would create a second
    # contract that could silently drift.
    from prismaquant.cb_export_config import source_passthrough_config_group

    return source_passthrough_config_group(format_name)


def _group_matches_template(
    group: Mapping[str, Any], template: Mapping[str, Any]
) -> bool:
    return all(group.get(key) == value for key, value in template.items())


def _next_group_key(groups: Mapping[str, Any]) -> str:
    used = set(str(key) for key in groups)
    index = 0
    while f"group_{index}" in used:
        index += 1
    return f"group_{index}"


def _assert_body_metadata_unchanged(
    old_config: Mapping[str, Any],
    old_quant: Mapping[str, Any],
    new_config: Mapping[str, Any],
    new_quant: Mapping[str, Any],
) -> None:
    """Prove the overlay only added MTP/draft metadata."""

    old_model = deepcopy(dict(old_config))
    new_model = deepcopy(dict(new_config))
    old_model.pop("n_mtp_layers", None)
    new_model.pop("n_mtp_layers", None)
    if old_model != new_model:
        raise AssertionError("DSpark overlay changed body model configuration")

    allowed_top = {"config_groups", "ignore", "source_passthrough", "provenance"}
    old_top = {key: value for key, value in old_quant.items() if key not in allowed_top}
    new_top = {key: value for key, value in new_quant.items() if key not in allowed_top}
    if old_top != new_top:
        raise AssertionError("DSpark overlay changed the body quantization contract")

    old_ignore = {str(value) for value in old_quant.get("ignore") or ()}
    new_ignore = {str(value) for value in new_quant.get("ignore") or ()}
    if {value for value in old_ignore if not value.startswith("mtp.")} != {
        value for value in new_ignore if not value.startswith("mtp.")
    }:
        raise AssertionError("DSpark overlay changed a non-MTP ignore entry")

    old_units = dict((old_quant.get("source_passthrough") or {}).get("units") or {})
    new_units = dict((new_quant.get("source_passthrough") or {}).get("units") or {})
    for unit, wire in old_units.items():
        if new_units.get(unit) != wire:
            raise AssertionError(
                f"DSpark overlay changed body passthrough unit {unit!r}"
            )

    old_groups = old_quant.get("config_groups") or {}
    new_groups = new_quant.get("config_groups") or {}
    if not isinstance(old_groups, Mapping) or not isinstance(new_groups, Mapping):
        raise AssertionError("config_groups must remain objects")
    for key, old_group in old_groups.items():
        if key not in new_groups:
            raise AssertionError(f"DSpark overlay removed config group {key!r}")
        new_group = new_groups[key]
        if not isinstance(old_group, Mapping) or not isinstance(new_group, Mapping):
            if old_group != new_group:
                raise AssertionError(f"DSpark overlay changed config group {key!r}")
            continue
        old_fields = {name: value for name, value in old_group.items() if name != "targets"}
        new_fields = {name: value for name, value in new_group.items() if name != "targets"}
        if old_fields != new_fields:
            raise AssertionError(f"DSpark overlay changed config group {key!r}")
        old_targets = set(str(value) for value in old_group.get("targets") or ())
        new_targets = set(str(value) for value in new_group.get("targets") or ())
        if not old_targets <= new_targets:
            raise AssertionError(
                f"DSpark overlay removed targets from config group {key!r}"
            )
        added = new_targets - old_targets
        if any("mtp[.]" not in target for target in added):
            raise AssertionError(
                f"DSpark overlay added a non-MTP target to config group {key!r}"
            )

    old_provenance = deepcopy(dict(old_quant.get("provenance") or {}))
    new_provenance = deepcopy(dict(new_quant.get("provenance") or {}))
    for record in (old_provenance, new_provenance):
        record.pop("dspark_source_overlay", None)
        record.pop("source_passthrough_targets", None)
    if old_provenance != new_provenance:
        raise AssertionError("DSpark overlay changed unrelated provenance")


def apply_dspark_overlay_to_quant_config(
    quant_config: Mapping[str, Any],
    overlay: DSparkSourceOverlay | None,
) -> dict[str, Any]:
    """Return a validated metadata-only ``quant_config.json`` overlay.

    Existing groups are extended rather than replaced.  Every newly declared
    physical target must either be in ``ignore`` (the expected pre-overlay
    artifact) or already have the identical source group (idempotent replay).
    Any other prior ownership fails closed.
    """

    out = deepcopy(dict(quant_config))
    if overlay is None:
        return out
    if out.get("quant_method") != "gridbook":
        raise ValueError("DSpark sidecar overlay requires quant_method='gridbook'")
    groups = out.get("config_groups")
    if not isinstance(groups, dict):
        raise ValueError("DSpark sidecar overlay requires config_groups object")
    ignore_raw = out.get("ignore")
    if not isinstance(ignore_raw, list) or not all(
        isinstance(value, str) for value in ignore_raw
    ):
        raise ValueError("DSpark sidecar overlay requires a string ignore list")
    ignored = set(ignore_raw)

    group_for_format: dict[str, tuple[str, dict[str, Any]]] = {}
    for format_name in sorted(set(overlay.physical_targets.values())):
        template = _source_group_template(format_name)
        wire_id = template["source_passthrough_id"]
        malformed = [
            key for key, group in groups.items()
            if isinstance(group, Mapping)
            and group.get("source_passthrough_id") == wire_id
            and not _group_matches_template(group, template)
        ]
        if malformed:
            raise ValueError(
                f"source group(s) {malformed} claim {wire_id!r} with a "
                "different tensor-layout contract"
            )
        candidates = [
            (str(key), group) for key, group in groups.items()
            if isinstance(group, dict) and _group_matches_template(group, template)
        ]
        if len(candidates) > 1:
            raise ValueError(
                f"multiple source groups claim format {format_name}: "
                f"{[key for key, _ in candidates]}"
            )
        if candidates:
            group_for_format[format_name] = candidates[0]
        else:
            key = _next_group_key(groups)
            group = deepcopy(template)
            group["targets"] = []
            groups[key] = group
            group_for_format[format_name] = (key, group)

    claims_by_physical: dict[str, list[tuple[str, Mapping[str, Any]]]] = {
        physical: [] for physical in overlay.physical_targets
    }
    physical_set = set(claims_by_physical)
    for key, group in groups.items():
        if not isinstance(group, Mapping):
            continue
        for target in group.get("targets") or ():
            literal = _literal_explicit_target(target)
            if literal is not None:
                if literal in physical_set:
                    claims_by_physical[literal].append((str(key), group))
                continue
            # Broad/custom regexes are uncommon.  They are evaluated against
            # the closed 2,329-target set once here rather than recompiling all
            # 10k exact body/expert targets for every DSpark base.
            for physical in physical_set:
                if _target_claims(target, physical):
                    claims_by_physical[physical].append((str(key), group))

    for physical, format_name in overlay.physical_targets.items():
        claims = claims_by_physical[physical]
        expected_key, expected_group = group_for_format[format_name]
        wrong = [key for key, group in claims if group is not expected_group]
        if wrong:
            raise ValueError(
                f"{physical}: already claimed by incompatible config group(s) {wrong}"
            )
        if claims:
            if physical in ignored:
                raise ValueError(
                    f"{physical}: is both source-declared and ignored; refusing "
                    "to repair an ambiguous partial overlay"
                )
        elif physical not in ignored:
            raise ValueError(
                f"{physical}: is neither ignored nor already source-declared; "
                "the sidecar does not have the expected pre-overlay ownership"
            )
        target = _explicit_target(physical)
        targets = expected_group.setdefault("targets", [])
        if not isinstance(targets, list):
            raise ValueError(f"config group {expected_key} targets must be a list")
        if target not in targets:
            targets.append(target)

    for _key, group in group_for_format.values():
        group["targets"] = sorted(set(str(value) for value in group["targets"]))
    out["ignore"] = sorted(ignored - set(overlay.physical_targets))

    from prismaquant.cb_export_config import (
        build_source_passthrough_declaration,
        parse_source_passthrough_declaration,
    )

    current_units = dict((out.get("source_passthrough") or {}).get("units") or {})
    overlay_declaration = build_source_passthrough_declaration(
        overlay.construction_units
    )
    for unit, wire_id in overlay_declaration["units"].items():
        previous = current_units.setdefault(unit, wire_id)
        if previous != wire_id:
            raise ValueError(
                f"{unit}: existing delegated route {previous!r} conflicts "
                f"with DSpark overlay {wire_id!r}"
            )
    out["source_passthrough"] = {
        "version": overlay_declaration["version"],
        "units": dict(sorted(current_units.items())),
    }

    provenance = out.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("quant_config provenance must be an object")
    previous_overlay = provenance.get("dspark_source_overlay")
    stamp = overlay.provenance()
    if previous_overlay not in (None, stamp):
        raise ValueError("existing DSpark overlay provenance conflicts with headers")
    provenance["dspark_source_overlay"] = stamp
    counts: dict[str, int] = {}
    for group in groups.values():
        if not isinstance(group, Mapping):
            continue
        format_name = group.get("source_format")
        if group.get("format") == "source-passthrough" and isinstance(
            format_name, str
        ):
            counts[str(format_name)] = counts.get(str(format_name), 0) + len(
                group.get("targets") or ()
            )
    provenance["source_passthrough_targets"] = dict(sorted(counts.items()))

    from prismaquant.allocator_candidates import ROUTE_PENDING_PASSTHROUGH_FORMATS

    acknowledged = set(
        provenance.get("route_pending_passthrough_acknowledged") or ()
    )
    pending = set(overlay.construction_units.values()) & set(
        ROUTE_PENDING_PASSTHROUGH_FORMATS
    )
    if not pending <= acknowledged:
        raise ValueError(
            "DSpark overlay contains route-pending format(s) without the "
            "artifact's prior ship acknowledgement: "
            f"{sorted(pending - acknowledged)}"
        )

    parse_source_passthrough_declaration(out)
    return out


def build_dspark_sidecar_overlay(
    config: Mapping[str, Any],
    quant_config: Mapping[str, Any],
    overlay: DSparkSourceOverlay | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build both JSON sidecars and assert body metadata is identical."""

    new_config = apply_dspark_overlay_to_model_config(config, overlay)
    new_quant = apply_dspark_overlay_to_quant_config(quant_config, overlay)
    _assert_body_metadata_unchanged(
        config, quant_config, new_config, new_quant
    )
    return new_config, new_quant


def _cb_output_tensor_names(header: Mapping[str, Mapping[str, Any]]) -> list[str]:
    names = set(header)
    out: set[str] = set()
    for name in names:
        if not name.endswith(".cb_qweight"):
            continue
        out.add(name)
        base = name[: -len(".cb_qweight")]
        for suffix in (".weight_scale", ".input_global_scale"):
            companion = base + suffix
            if companion in names:
                out.add(companion)
    return sorted(out)


def _hardlink_artifact_tree(source: Path, staging: Path) -> None:
    """Clone an artifact structurally without copying any tensor bytes."""

    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = staging / relative
        if path.is_symlink():
            raise ValueError(
                f"refusing DSpark staging through artifact symlink {path}"
            )
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            raise ValueError(f"unsupported artifact entry {path}")
        if relative.as_posix() in {"config.json", "quant_config.json"}:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Linux atomic directory publication that never replaces a target."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(
            "atomic DSpark publication requires Linux renameat2(RENAME_NOREPLACE)"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error,
            "refusing to replace existing DSpark output artifact",
            str(destination),
        )
    raise OSError(error, os.strerror(error), str(destination))


def apply_dspark_sidecar_overlay(
    artifact_dir: str | Path,
    output_artifact_dir: str | Path,
) -> DSparkSourceOverlay:
    """Atomically publish a sidecar-updated hardlink artifact.

    ``output_artifact_dir`` must be an absent sibling of ``artifact_dir``.  A
    hidden staging tree hardlinks every tensor/container, validates the two new
    sidecars against that complete tree, then becomes visible with one atomic
    directory rename.  A launcher can therefore observe only "output absent"
    or the complete new artifact; there is no two-sidecar crash window.  The
    source artifact is never modified.
    """

    root = Path(artifact_dir).resolve()
    output = Path(output_artifact_dir).resolve()
    if output.parent != root.parent:
        raise ValueError(
            "DSpark output must be an absent sibling of the source artifact "
            "so hardlinks and the final directory rename stay on one filesystem"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"refusing to replace existing DSpark output artifact: {output}"
        )
    config_path = root / "config.json"
    quant_path = root / "quant_config.json"
    config = json.loads(config_path.read_bytes())
    quant_config = json.loads(quant_path.read_bytes())
    overlay = discover_dspark_source_overlay(
        ArtifactHeaderSkeleton(root), config
    )
    if overlay is None:
        raise ValueError(f"{root}: not a released DeepSeek-V4 DSpark artifact")
    new_config, new_quant = build_dspark_sidecar_overlay(
        config, quant_config, overlay
    )

    header = ArtifactHeaderSkeleton(root).header
    model_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.safetensors")
        if path.is_file()
    )
    model_snapshot = {
        name: (
            (root / name).stat().st_ino,
            (root / name).stat().st_size,
            (root / name).stat().st_mtime_ns,
        )
        for name in model_files
    }
    provenance = new_quant.get("provenance") or {}
    serialized_payload = provenance.get("serialized_payload")
    if not isinstance(serialized_payload, Mapping):
        raise ValueError(
            "DSpark sidecar overlay requires producer serialized_payload "
            "provenance to recompute the artifact inventory"
        )
    inventory = provenance.get("artifact_inventory") or {}
    whole_budget = inventory.get("whole_artifact_budget_bytes")

    with tempfile.TemporaryDirectory(
        dir=root.parent, prefix=f".{output.name}.dspark-overlay."
    ) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        _hardlink_artifact_tree(root, staging)
        (staging / "config.json").write_text(
            json.dumps(new_config, indent=2) + "\n"
        )
        from prismaquant.nvfp4_cb_footprint import (
            finalize_cb_export_artifact_inventory,
        )

        finalize_cb_export_artifact_inventory(
            staging,
            new_quant,
            serialized_payload=serialized_payload,
            cb_tensor_names=_cb_output_tensor_names(header),
            codebook_file=new_quant.get("codebook_file"),
            expected_model_files=model_files,
            whole_artifact_budget_bytes=(
                int(whole_budget) if whole_budget is not None else None
            ),
        )
        # Discovery + completeness are rerun against the staged, consumer-
        # visible artifact.  With an overlay, mtp.* is no longer an exempt
        # verbatim namespace: its physical source groups must claim every
        # scale-bearing target.
        staged_overlay = discover_dspark_source_overlay_from_artifact(staging)
        if staged_overlay != overlay:
            raise AssertionError("staged DSpark overlay differs from source headers")
        from prismaquant.artifact_completeness import assert_artifact_complete

        assert_artifact_complete(staging, verbatim_prefixes=())

        for name, before in model_snapshot.items():
            stat = (root / name).stat()
            after = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if after != before:
                raise RuntimeError(
                    f"artifact model file changed during sidecar staging: {name}"
                )

        # Output did not exist at the preflight above.  Renaming this complete
        # directory is the single publication point: no consumer can observe
        # one new sidecar without the other.
        _rename_directory_noreplace(staging, output)

    for name, before in model_snapshot.items():
        stat = (root / name).stat()
        after = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if after != before:
            raise RuntimeError(
                f"artifact model file changed while applying sidecars: {name}"
            )
    return overlay


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Validate or apply the producer-owned metadata-only DeepSeek-V4 "
            "DSpark source overlay"
        )
    )
    parser.add_argument("artifact")
    parser.add_argument(
        "--output-artifact", metavar="PATH",
        help=(
            "atomically publish a complete sibling artifact whose model files "
            "are hardlinks and whose two JSON sidecars carry the overlay"
        ),
    )
    args = parser.parse_args(argv)
    root = Path(args.artifact)
    if args.output_artifact:
        overlay = apply_dspark_sidecar_overlay(root, args.output_artifact)
    else:
        config = json.loads((root / "config.json").read_text())
        quant = json.loads((root / "quant_config.json").read_text())
        overlay = discover_dspark_source_overlay(
            ArtifactHeaderSkeleton(root), config
        )
        if overlay is None:
            raise ValueError(f"{root}: not a released DeepSeek-V4 DSpark artifact")
        build_dspark_sidecar_overlay(config, quant, overlay)
        print(
            "validated only; no artifact files changed "
            "(pass --output-artifact PATH to publish)"
        )
    print(json.dumps(overlay.provenance(), indent=2, sort_keys=True))
    return 0


__all__ = [
    "ArtifactHeaderSkeleton",
    "DSPARK_OVERLAY_SCHEMA",
    "DSPARK_STAGE_COUNT",
    "DSPARK_TARGET_BRIDGE_SCHEMA",
    "DSparkSourceOverlay",
    "apply_dspark_overlay_to_model_config",
    "apply_dspark_overlay_to_quant_config",
    "apply_dspark_sidecar_overlay",
    "build_dspark_target_bridge",
    "build_dspark_sidecar_overlay",
    "discover_dspark_source_overlay",
    "discover_dspark_source_overlay_from_artifact",
    "dspark_cb_construction_target_for_physical_output",
    "dspark_cb_physical_output_for_construction_target",
    "dspark_cb_physical_output_for_recipe_target",
    "dspark_cb_physical_source_for_recipe_target",
    "dspark_construction_unit_for_physical_target",
    "parse_dspark_overlay_provenance",
    "validate_dspark_target_bridge",
]


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke
    raise SystemExit(main())
