"""Schema-separated, research-only checkpoint planner for PrismaSnap MoE.

The dense Qwen3.8-27B release treatment is frozen in
``prismasnap_checkpoint.plan_dense_checkpoint``.  This module extends the
same source identity, tensor-header, probe, producer, atomic plan, streaming
materialization, and multi-worker collation machinery without changing dense
defaults.  MoE plans and materialized receipts have distinct schemas and are
categorically ineligible for production admission until a real MoE checkpoint
supplies fold-fidelity and served-KL evidence.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import math
import os
from pathlib import Path
import pickle
import shutil
from typing import Any

import numpy as np
from safetensors.torch import save_file
import torch

from .cost_stage_checkpoint import canonical_json_sha256
from .model_profiles import detect_profile
from .prismasnap import (
    PrismaSnapConsumer,
    PrismaSnapSearchConfig,
    apply_diagonal_transform,
    search_diagonal_scale,
)
from .prismasnap_moe import (
    PRISMASNAP_MOE_ALGORITHM,
    PRISMASNAP_MOE_PROMOTION,
    PackedDown,
    PackedGateUp,
    apply_packed_expert_slice_transform,
    fp64_router_and_expert_invariance,
    moe_search_contract,
    packed_post_norm_consumers,
    search_packed_up_down_scales,
)


MOE_PLAN_SCHEMA = "prismaquant.prismasnap.moe_plan.v1"
MOE_PLAN_SET_SCHEMA = "prismaquant.prismasnap.moe_plan_set.v1"
MOE_PROVENANCE_SCHEMA = "prismaquant.prismasnap.moe_provenance.v1"
MOE_PROFILE_SCHEMA = "prismaquant.prismasnap.moe_layer_profile.v1"

_PLAN_KEYS = frozenset(
    {
        "schema",
        "state",
        "algorithm",
        "producer",
        "profile",
        "source",
        "probe",
        "model",
        "search",
        "tensor_metadata",
        "tensor_metadata_binding",
        "scales",
        "seams",
        "transforms",
        "verification",
        "promotion",
        "plan_sha256",
    }
)
_PLAN_SET_KEYS = _PLAN_KEYS | {"workers"}
_MODEL_KEYS = frozenset(
    {
        "hidden_size",
        "layer_count",
        "planned_layers",
        "excluded_prefixes",
        "expert_counts",
        "routed_layouts",
    }
)
_PROFILE_KEYS = frozenset(
    {
        "schema",
        "layer",
        "input_norm",
        "post_attention_norm",
        "mlp_prefix",
        "router",
        "packed_routed",
        "per_expert_routed",
        "shared_experts",
        "bias_policy",
    }
)
_PACKED_PROFILE_KEYS = frozenset(
    {
        "gate_up",
        "down",
        "expert_axis",
        "row_axis",
        "input_axis",
        "gate_rows",
        "up_rows",
    }
)
_PER_EXPERT_PROFILE_KEYS = frozenset(
    {"root", "gate_projection", "up_projection", "down_projection"}
)
_SHARED_PROFILE_KEYS = frozenset({"output_gate", "gate", "up", "down"})
_STATS_KEYS = frozenset(
    {
        "algorithm",
        "error_baseline",
        "error_final",
        "improvement_fraction",
        "groups",
        "groups_moved",
        "rounds",
        "candidate_count",
        "fell_back",
        "polish_pool",
        "polished",
        "variant",
    }
)
_NORM_KEYS = frozenset(
    {
        "layer",
        "kind",
        "vector",
        "norm",
        "norm_parameter_offset",
        "consumers",
        "stats",
        "graph_sha256",
    }
)
_POST_KEYS = frozenset(
    {
        "layer",
        "kind",
        "vector",
        "norm",
        "norm_parameter_offset",
        "objective_consumers",
        "compensation_only_consumers",
        "router",
        "routed_layout",
        "stats",
        "graph_sha256",
    }
)
_PACKED_ROUTED_KEYS = frozenset(
    {
        "layer",
        "kind",
        "vector",
        "layout",
        "experts",
        "gate_up",
        "down",
        "expert_axis",
        "row_axis",
        "input_axis",
        "gate_rows",
        "up_rows",
        "stats",
        "graph_sha256",
    }
)
_PER_EXPERT_ROUTED_KEYS = frozenset(
    {
        "layer",
        "kind",
        "vector",
        "layout",
        "experts",
        "roles",
        "stats",
        "graph_sha256",
    }
)
_SHARED_KEYS = frozenset(
    {
        "layer",
        "kind",
        "shared_index",
        "vector",
        "output_gate",
        "gate",
        "up",
        "down",
        "stats",
        "graph_sha256",
    }
)
_PER_EXPERT_ROLE_KEYS = frozenset({"expert", "gate", "up", "down"})
_VERIFICATION_KEYS = frozenset(
    {
        "fp64_invariance_max_abs",
        "router_logit_max_abs",
        "route_weight_max_abs",
        "routed_output_max_abs",
        "routing_changed",
        "threshold",
        "domain",
        "required_bf16_fold_kl_max",
        "real_moe_fold_kl_evidence",
    }
)


def _checkpoint_module():
    # Lazy import avoids a module cycle when the common checkpoint loader
    # dispatches a MoE schema back into this module.
    from . import prismasnap_checkpoint as checkpoint

    return checkpoint


def _moe_producer_identity() -> dict[str, object]:
    """Extend the common receipt with the schema-separated MoE code bytes."""
    checkpoint = _checkpoint_module()
    producer = checkpoint._producer_identity()
    repository = Path(__file__).resolve().parents[1]
    files = dict(producer["source_files"])
    for relative in (
        "prismaquant/prismasnap_moe.py",
        "prismaquant/prismasnap_moe_checkpoint.py",
    ):
        files[relative] = checkpoint._sha256_file(repository / relative)
    files = dict(sorted(files.items()))
    return {
        **producer,
        "source_files": files,
        "source_sha256": canonical_json_sha256(
            files, where="PrismaSnap MoE producer source files"
        ),
    }


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], where: str) -> None:
    actual = set(value)
    if actual != set(expected):
        raise RuntimeError(
            f"{where} fields differ: missing={sorted(set(expected) - actual)} "
            f"extra={sorted(actual - set(expected))}"
        )


def _source_name(profile, recipe_name: str) -> str:
    result = profile.source_tensor_name(recipe_name)
    if not isinstance(result, str) or not result:
        raise RuntimeError(
            f"profile {profile.name!r} returned malformed source name for "
            f"{recipe_name!r}"
        )
    return result


def _validate_profile_contract(
    raw: object,
    *,
    layer: int,
    profile,
) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise RuntimeError(
            f"profile {profile.name!r} does not declare PrismaSnap MoE layer {layer}"
        )
    _exact_keys(raw, _PROFILE_KEYS, "PrismaSnap MoE profile contract")
    if (
        raw.get("schema") != MOE_PROFILE_SCHEMA
        or raw.get("layer") != layer
        or raw.get("bias_policy") != "reject_projection_biases"
    ):
        raise RuntimeError("PrismaSnap MoE profile contract header is invalid")
    for key in ("input_norm", "post_attention_norm", "mlp_prefix"):
        if not isinstance(raw.get(key), str) or not raw.get(key):
            raise RuntimeError(f"PrismaSnap MoE profile {key} is malformed")
    router = raw.get("router")
    if not isinstance(router, str) or not router:
        raise RuntimeError("PrismaSnap MoE typed router declaration is malformed")
    packed = raw.get("packed_routed")
    per_expert = raw.get("per_expert_routed")
    if not isinstance(packed, Mapping) or not isinstance(per_expert, Mapping):
        raise RuntimeError("PrismaSnap MoE routed layout declarations are malformed")
    _exact_keys(packed, _PACKED_PROFILE_KEYS, "PrismaSnap packed profile")
    _exact_keys(per_expert, _PER_EXPERT_PROFILE_KEYS, "PrismaSnap per-expert profile")
    if (
        any(not isinstance(packed.get(key), str) or not packed.get(key) for key in ("gate_up", "down"))
        or packed.get("expert_axis") != 0
        or packed.get("row_axis") != 1
        or packed.get("input_axis") != 2
        or packed.get("gate_rows") != "first_half"
        or packed.get("up_rows") != "second_half"
        or any(
            not isinstance(per_expert.get(key), str) or not per_expert.get(key)
            for key in _PER_EXPERT_PROFILE_KEYS
        )
    ):
        raise RuntimeError("PrismaSnap MoE routed layout contract is invalid")
    shared = raw.get("shared_experts")
    if not isinstance(shared, (tuple, list)):
        raise RuntimeError("PrismaSnap shared-expert declaration is malformed")
    normalized_shared: list[dict[str, str]] = []
    for index, value in enumerate(shared):
        if not isinstance(value, Mapping):
            raise RuntimeError(f"PrismaSnap shared expert {index} is malformed")
        _exact_keys(value, _SHARED_PROFILE_KEYS, f"PrismaSnap shared expert {index}")
        if any(not isinstance(value.get(key), str) or not value.get(key) for key in _SHARED_PROFILE_KEYS):
            raise RuntimeError(f"PrismaSnap shared expert {index} roles are malformed")
        normalized_shared.append({key: str(value[key]) for key in sorted(_SHARED_PROFILE_KEYS)})
    declared_role_names = [
        str(value)
        for role in normalized_shared
        for value in role.values()
    ]
    if router in declared_role_names or len(set(declared_role_names)) != len(
        declared_role_names
    ):
        raise RuntimeError("PrismaSnap MoE router/shared roles overlap")

    # Keep this research declaration cross-bound to the profile's production
    # allocation/export vocabulary instead of creating a second family-name
    # table that can silently drift.
    packed_gate_leaf = str(packed["gate_up"]).rsplit(".", 1)[-1]
    packed_down_leaf = str(packed["down"]).rsplit(".", 1)[-1]
    declared_packed = profile.packed_expert_param_names()
    if {packed_gate_leaf, packed_down_leaf} - set(declared_packed):
        raise RuntimeError(
            "PrismaSnap MoE packed roles are absent from profile vocabulary"
        )
    if tuple(profile.packed_expert_projection_names(packed_gate_leaf)) != (
        str(per_expert["gate_projection"]),
        str(per_expert["up_projection"]),
    ) or tuple(profile.packed_expert_projection_names(packed_down_leaf)) != (
        str(per_expert["down_projection"]),
    ):
        raise RuntimeError(
            "PrismaSnap MoE packed/per-expert projection roles disagree"
        )
    packed_groups = {
        profile.packed_expert_format_group(str(packed["gate_up"])),
        profile.packed_expert_format_group(str(packed["down"])),
    }
    if None in packed_groups or len(packed_groups) != 1:
        raise RuntimeError(
            "PrismaSnap MoE packed roles do not share one format group"
        )
    return {
        **dict(raw),
        "router": str(router),
        "packed_routed": dict(packed),
        "per_expert_routed": dict(per_expert),
        "shared_experts": tuple(normalized_shared),
    }


def _load_moe_probe(
    path: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, object], str]:
    """Read the existing dense+packed probe contract without weakening dense."""
    checkpoint = _checkpoint_module()
    probe_bytes = path.read_bytes()
    try:
        payload = pickle.loads(probe_bytes)  # noqa: S301 - trusted local receipt
    except Exception as exc:
        raise RuntimeError("PrismaSnap MoE probe is unreadable") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("stats"), dict):
        raise RuntimeError("PrismaSnap MoE probe has no stats mapping")
    if not isinstance(payload.get("meta"), dict):
        raise RuntimeError("PrismaSnap MoE probe has no metadata")
    stats = payload["stats"]
    for qname, row in stats.items():
        if not isinstance(qname, str) or not isinstance(row, dict):
            raise RuntimeError("PrismaSnap MoE probe stats are malformed")
        if checkpoint._BODY_LAYER.match(qname) is None:
            continue
        for field in ("in_features", "out_features"):
            if type(row.get(field)) is not int or int(row[field]) <= 0:
                raise RuntimeError(f"PrismaSnap MoE probe row {qname!r} lacks {field}")
        if "expert_act_sq_sum" in row:
            experts = row.get("num_experts")
            importance = np.asarray(row["expert_act_sq_sum"], dtype=np.float32)
            tokens = np.asarray(row.get("expert_tokens"), dtype=np.float64)
            if (
                type(experts) is not int
                or experts <= 0
                or importance.shape != (experts, int(row["in_features"]))
                or tokens.shape != (experts,)
                or not np.isfinite(importance).all()
                or np.any(importance < 0)
                or not np.isfinite(tokens).all()
                or np.any(tokens <= 0)
                or np.any(importance.sum(axis=1) <= 0)
                or not isinstance(row.get("_packed_param"), str)
            ):
                raise RuntimeError(
                    f"PrismaSnap MoE packed probe row {qname!r} lacks complete "
                    "per-expert routed importance; a new probe is required"
                )
        else:
            importance = np.asarray(row.get("act_sq_sum"), dtype=np.float32)
            if (
                importance.shape != (int(row["in_features"]),)
                or not np.isfinite(importance).all()
                or np.any(importance < 0)
                or not np.any(importance > 0)
            ):
                raise RuntimeError(f"PrismaSnap MoE probe row {qname!r} has invalid importance")
    return stats, payload["meta"], hashlib.sha256(probe_bytes).hexdigest()


def _dense_importance(row: Mapping[str, object], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(row["act_sq_sum"], dtype=np.float32), device=device)


def _packed_importance(row: Mapping[str, object], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        np.asarray(row["expert_act_sq_sum"], dtype=np.float32), device=device
    )


def _normalized_stats(value: Mapping[str, object]) -> dict[str, object]:
    result = {key: value[key] for key in _STATS_KEYS}
    result["algorithm"] = PRISMASNAP_MOE_ALGORITHM
    return result


def _stats_key_for_weight(profile, source_weight: str) -> str:
    """Map a source weight back to the recipe probe key, closed by caller."""
    suffix = ".weight"
    if not source_weight.endswith(suffix):
        raise RuntimeError(f"PrismaSnap MoE weight lacks .weight: {source_weight}")
    # Profile contracts originate as recipe names, so graph discovery retains
    # an explicit source->recipe map. This function is only a guard against a
    # malformed call and is not used to reverse an arbitrary namespace.
    return source_weight[: -len(suffix)]


def _moe_plan_graph_sha256(
    *,
    layer: int,
    input_norm: str,
    norm_parameter_offset: float,
    input_consumers: Sequence[str],
    post_norm: str,
    router: str,
    routed_layout: str,
    experts: int,
    intermediate: int,
    packed_gate_up: str | None,
    packed_down: str | None,
    per_expert_roles: Sequence[Mapping[str, object]],
    shared_roles: Sequence[Mapping[str, object]],
) -> str:
    """Digest only serialized, executable, typed MoE graph roles."""
    if routed_layout == "packed_3d":
        routed: dict[str, object] = {
            "layout": routed_layout,
            "experts": experts,
            "gate_up": packed_gate_up,
            "down": packed_down,
            "expert_axis": 0,
            "row_axis": 1,
            "input_axis": 2,
            "gate_rows": [0, intermediate],
            "up_rows": [intermediate, 2 * intermediate],
        }
    elif routed_layout == "per_expert_2d":
        routed = {
            "layout": routed_layout,
            "experts": experts,
            "roles": [
                {
                    "expert": int(role["expert"]),
                    "gate": str(role["gate"]),
                    "up": str(role["up"]),
                    "down": str(role["down"]),
                }
                for role in per_expert_roles
            ],
        }
    else:
        raise RuntimeError(f"unsupported PrismaSnap MoE routed layout {routed_layout!r}")
    return canonical_json_sha256(
        {
            "layer": layer,
            "input_norm": input_norm,
            "post_attention_norm": post_norm,
            "norm_parameter_offset": norm_parameter_offset,
            "input_consumers": list(input_consumers),
            "router": router,
            "routed": routed,
            "shared_experts": [
                {
                    "shared_index": int(role["shared_index"]),
                    "output_gate": str(role["output_gate"]),
                    "gate": str(role["gate"]),
                    "up": str(role["up"]),
                    "down": str(role["down"]),
                }
                for role in shared_roles
            ],
        },
        where=f"PrismaSnap MoE layer {layer} executable graph",
    )


def _layer_source_graph(
    *,
    layer: int,
    hidden_size: int,
    expected_experts: int,
    stats: Mapping[str, dict[str, object]],
    profile,
    source,
    tensor_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    contract = _validate_profile_contract(
        profile.prismasnap_moe_layer_contract(layer),
        layer=layer,
        profile=profile,
    )
    recipe_to_source: dict[str, str] = {}

    def source_key(recipe: str) -> str:
        key = _source_name(profile, recipe)
        prior = recipe_to_source.setdefault(recipe, key)
        if prior != key:
            raise RuntimeError("PrismaSnap profile source mapping is nondeterministic")
        return key

    input_norm = source_key(str(contract["input_norm"]))
    post_norm = source_key(str(contract["post_attention_norm"]))
    for name in (input_norm, post_norm):
        row = tensor_rows.get(name)
        if row is None or tuple(row["shape"]) != (hidden_size,) or row["dtype"] != "BF16":
            raise RuntimeError(f"layer {layer}: invalid BF16 RMSNorm tensor {name}")

    router = source_key(str(contract["router"]))
    router_row = tensor_rows.get(router)
    if (
        router_row is None
        or tuple(router_row["shape"]) != (expected_experts, hidden_size)
        or router_row["dtype"] != "BF16"
    ):
        raise RuntimeError(f"layer {layer}: E-way router shape/dtype failed: {router}")

    packed_decl = contract["packed_routed"]
    packed_gate_up_recipe = str(packed_decl["gate_up"])
    packed_down_recipe = str(packed_decl["down"])
    packed_gate_leaf = packed_gate_up_recipe.rsplit(".", 1)[-1]
    packed_down_leaf = packed_down_recipe.rsplit(".", 1)[-1]
    packed_gate_up = source_key(packed_gate_up_recipe)
    packed_down = source_key(packed_down_recipe)
    packed_presence = (packed_gate_up in tensor_rows, packed_down in tensor_rows)

    per_decl = contract["per_expert_routed"]
    per_root = str(per_decl["root"])
    projection_names = (
        str(per_decl["gate_projection"]),
        str(per_decl["up_projection"]),
        str(per_decl["down_projection"]),
    )
    per_roles: list[dict[str, object]] = []
    per_any = False
    per_complete = True
    for expert in range(expected_experts):
        names = {
            role: source_key(f"{per_root}.{expert}.{projection}.weight")
            for role, projection in zip(("gate", "up", "down"), projection_names)
        }
        present = {role: name in tensor_rows for role, name in names.items()}
        per_any = per_any or any(present.values())
        per_complete = per_complete and all(present.values())
        per_roles.append({"expert": expert, **names})
    if any(packed_presence) and not all(packed_presence):
        raise RuntimeError(f"layer {layer}: packed expert pair is partial")
    if all(packed_presence) and per_any:
        raise RuntimeError(f"layer {layer}: mixed packed and per-expert routed source")
    if not all(packed_presence) and not (per_any and per_complete):
        raise RuntimeError(f"layer {layer}: no complete profile-declared routed source layout")

    if all(packed_presence):
        gu_shape = tuple(tensor_rows[packed_gate_up]["shape"])
        down_shape = tuple(tensor_rows[packed_down]["shape"])
        if (
            len(gu_shape) != 3
            or gu_shape[0] != expected_experts
            or gu_shape[2] != hidden_size
            or gu_shape[1] % 2
            or down_shape != (expected_experts, hidden_size, gu_shape[1] // 2)
            or tensor_rows[packed_gate_up]["dtype"] != "BF16"
            or tensor_rows[packed_down]["dtype"] != "BF16"
        ):
            raise RuntimeError(f"layer {layer}: packed routed shape/dtype contract failed")
        routed_layout = "packed_3d"
        intermediate = gu_shape[1] // 2
    else:
        first = per_roles[0]
        gate_shape = tuple(tensor_rows[str(first["gate"])]["shape"])
        up_shape = tuple(tensor_rows[str(first["up"])]["shape"])
        down_shape = tuple(tensor_rows[str(first["down"])]["shape"])
        if len(gate_shape) != 2 or gate_shape[1] != hidden_size or up_shape != gate_shape:
            raise RuntimeError(f"layer {layer}: per-expert gate/up shapes are invalid")
        intermediate = gate_shape[0]
        if down_shape != (hidden_size, intermediate):
            raise RuntimeError(f"layer {layer}: per-expert down shape is invalid")
        for role in per_roles:
            shapes = {
                name: tuple(tensor_rows[str(role[name])]["shape"])
                for name in ("gate", "up", "down")
            }
            if (
                shapes["gate"] != gate_shape
                or shapes["up"] != up_shape
                or shapes["down"] != down_shape
                or any(tensor_rows[str(role[name])]["dtype"] != "BF16" for name in ("gate", "up", "down"))
            ):
                raise RuntimeError(f"layer {layer}: per-expert shape/dtype drift")
        routed_layout = "per_expert_2d"

    shared_roles: list[dict[str, object]] = []
    for index, role in enumerate(contract["shared_experts"]):
        names = {
            name: source_key(str(role[name]))
            for name in ("output_gate", "gate", "up", "down")
        }
        if any(name not in tensor_rows for name in names.values()):
            raise RuntimeError(f"layer {layer}: shared expert {index} is incomplete")
        output_gate_shape = tuple(tensor_rows[names["output_gate"]]["shape"])
        gate_shape = tuple(tensor_rows[names["gate"]]["shape"])
        up_shape = tuple(tensor_rows[names["up"]]["shape"])
        down_shape = tuple(tensor_rows[names["down"]]["shape"])
        if (
            output_gate_shape != (1, hidden_size)
            or tensor_rows[names["output_gate"]]["dtype"] != "BF16"
            or len(gate_shape) != 2
            or gate_shape[1] != hidden_size
            or up_shape != gate_shape
            or down_shape != (hidden_size, gate_shape[0])
            or any(tensor_rows[names[name]]["dtype"] != "BF16" for name in names)
        ):
            raise RuntimeError(f"layer {layer}: shared expert {index} shape/dtype failed")
        shared_roles.append({"shared_index": index, **names})
    shared_output_gates = [str(role["output_gate"]) for role in shared_roles]
    compensation_consumers = [router, *shared_output_gates]

    projection_weights: set[str] = set()
    if routed_layout == "packed_3d":
        projection_weights.update((packed_gate_up, packed_down))
    else:
        for role in per_roles:
            projection_weights.update(str(role[name]) for name in ("gate", "up", "down"))
    for role in shared_roles:
        projection_weights.update(str(role[name]) for name in ("gate", "up", "down"))
    for weight in projection_weights:
        bias = weight.removesuffix(".weight") + ".bias"
        if bias in tensor_rows:
            raise RuntimeError(
                f"layer {layer}: expert projection bias is unsupported and must "
                f"fail closed: {bias}"
            )

    # Closed hidden-input census inside the profile-declared MLP. Unknown
    # matrices that could consume the post-norm state are never silently left
    # uncompensated.
    source_mlp_prefix = _source_name(profile, str(contract["mlp_prefix"])) + "."
    expected_hidden_consumers = set(compensation_consumers)
    if routed_layout == "packed_3d":
        expected_hidden_consumers.add(packed_gate_up)
    else:
        for role in per_roles:
            expected_hidden_consumers.update((str(role["gate"]), str(role["up"])))
    for role in shared_roles:
        expected_hidden_consumers.update((str(role["gate"]), str(role["up"])))
    observed_hidden_consumers = {
        name
        for name, row in tensor_rows.items()
        if name.startswith(source_mlp_prefix)
        and len(row["shape"]) in {2, 3}
        and int(row["shape"][-1]) == hidden_size
    }
    if observed_hidden_consumers != expected_hidden_consumers:
        raise RuntimeError(
            f"layer {layer}: profile does not close every MLP hidden-input "
            f"consumer; missing={sorted(observed_hidden_consumers - expected_hidden_consumers)} "
            f"declared_but_absent={sorted(expected_hidden_consumers - observed_hidden_consumers)}"
        )

    # Token-mixer consumers are the one remaining hidden-width activation
    # equivalence class outside the MLP. The profile supplies ordering through
    # its fused groups; no attention projection names are guessed here.
    recipe_prefix = f"model.layers.{layer}."
    layer_stats = {name: row for name, row in stats.items() if name.startswith(recipe_prefix)}
    input_candidates = {
        name: row
        for name, row in layer_stats.items()
        if not name.startswith(f"model.layers.{layer}.mlp.")
        and "act_sq_sum" in row
        and int(row["in_features"]) == hidden_size
    }
    clusters: dict[str, list[str]] = defaultdict(list)
    checkpoint = _checkpoint_module()
    for name, row in input_candidates.items():
        clusters[checkpoint._importance_digest(row["act_sq_sum"])].append(name)

    # Output projections such as attention o_proj also have input width H but
    # consume a different runtime tensor. Select the one activation class that
    # closes at least one complete profile-declared fused *input* group. This
    # keeps hybrid qkvz+ba groups profile-driven and leaves unrelated output
    # classes explicitly outside the norm seam.
    fused_groups: dict[str, list[str]] = defaultdict(list)
    for name in input_candidates:
        group = profile.fused_sibling_group(name)
        if group is not None:
            fused_groups[str(group)].append(name)
    leaf_mapping = profile.fused_sibling_leaf_mapping()
    complete_groups: set[str] = set()
    for group, members in fused_groups.items():
        expected_leaves = leaf_mapping.get(group.rsplit(".", 1)[-1])
        observed_leaves = {name.rsplit(".", 1)[-1] for name in members}
        if expected_leaves is None or observed_leaves != set(expected_leaves):
            raise RuntimeError(
                f"layer {layer}: token-mixer fused group is partial/unknown: "
                f"{sorted(members)}"
            )
        member_digests = {
            checkpoint._importance_digest(input_candidates[name]["act_sq_sum"])
            for name in members
        }
        if len(member_digests) != 1:
            raise RuntimeError(
                f"layer {layer}: token-mixer fused siblings do not share input"
            )
        complete_groups.add(group)
    candidate_digests = {
        checkpoint._importance_digest(input_candidates[members[0]]["act_sq_sum"])
        for group, members in fused_groups.items()
        if group in complete_groups and len(members) >= 2
    }
    if len(candidate_digests) != 1:
        raise RuntimeError(
            f"layer {layer}: expected one profile-closed token-mixer input "
            f"class, got {[sorted(clusters[digest]) for digest in candidate_digests]}"
        )
    input_digest = next(iter(candidate_digests))
    input_recipe = checkpoint._ordered_dense_consumers(
        clusters[input_digest], profile
    )
    if len(input_recipe) < 2:
        raise RuntimeError(f"layer {layer}: token-mixer consumer closure is too small")
    input_source = [source_key(f"{name}.weight") for name in input_recipe]
    for recipe, name in zip(input_recipe, input_source):
        row = tensor_rows.get(name)
        expected = (int(stats[recipe]["out_features"]), hidden_size)
        if row is None or tuple(row["shape"]) != expected or row["dtype"] != "BF16":
            raise RuntimeError(f"layer {layer}: token-mixer source/probe mismatch: {name}")

    # Probe topology is owned by the live model, not by the physical source
    # checkpoint layout. Qwen loads either source representation into packed
    # [E,...] Parameters and the production probe emits the same canonical
    # packed gate_up/down rows in both cases. Per-expert source tensors select
    # row e from these arrays; they do not require a different estimator.
    packed_probe_name = packed_gate_up_recipe.removesuffix(".weight")
    down_probe_name = packed_down_recipe.removesuffix(".weight")
    packed_probe_rows = (
        (packed_probe_name, hidden_size, packed_gate_leaf),
        (down_probe_name, intermediate, packed_down_leaf),
    )
    token_vectors: list[np.ndarray] = []
    for name, width, expected_param in packed_probe_rows:
        row = stats.get(name)
        tokens = np.asarray(row.get("expert_tokens")) if isinstance(row, Mapping) else np.asarray(None)
        if (
            not isinstance(row, Mapping)
            or np.asarray(row.get("expert_act_sq_sum")).shape
            != (expected_experts, width)
            or tokens.shape != (expected_experts,)
            or row.get("_packed_param") != expected_param
            or row.get("num_experts") != expected_experts
        ):
            raise RuntimeError(
                f"layer {layer}: packed live-model probe contract is missing {name}"
            )
        token_vectors.append(tokens.astype(np.float64, copy=False))
    if not np.array_equal(token_vectors[0], token_vectors[1]):
        raise RuntimeError(
            f"layer {layer}: gate_up/down routed token vectors disagree"
        )
    routed_probe = {
        "topology": "packed_live_model",
        "gate_up": packed_probe_name,
        "down": down_probe_name,
        "expert_tokens": token_vectors[0].tolist(),
    }
    shared_probe: list[dict[str, object]] = []
    for index, role in enumerate(contract["shared_experts"]):
        names = {name: str(role[name]).removesuffix(".weight") for name in ("gate", "up", "down")}
        for name in names.values():
            if not isinstance(stats.get(name), Mapping) or "act_sq_sum" not in stats[name]:
                raise RuntimeError(f"layer {layer}: shared expert probe row is missing: {name}")
        shared_probe.append({"shared_index": index, **names})

    norm_offset = profile.rms_norm_parameter_offset()
    if norm_offset is None or not math.isfinite(float(norm_offset)):
        raise RuntimeError(f"layer {layer}: profile has no finite RMSNorm encoding")
    graph: dict[str, object] = {
        "layer": layer,
        "input_norm": input_norm,
        "post_norm": post_norm,
        "norm_parameter_offset": float(norm_offset),
        "input_recipe_consumers": input_recipe,
        "input_consumers": input_source,
        "router": router,
        "shared_output_gates": shared_output_gates,
        "compensation_consumers": compensation_consumers,
        "routed_layout": routed_layout,
        "experts": expected_experts,
        "intermediate": intermediate,
        "packed_gate_up": packed_gate_up if routed_layout == "packed_3d" else None,
        "packed_down": packed_down if routed_layout == "packed_3d" else None,
        "per_expert_roles": per_roles if routed_layout == "per_expert_2d" else [],
        "shared_roles": shared_roles,
        "routed_probe": routed_probe,
        "shared_probe": shared_probe,
        "source_recipe_map": dict(sorted(recipe_to_source.items())),
    }
    graph["graph_sha256"] = _moe_plan_graph_sha256(
        layer=layer,
        input_norm=input_norm,
        norm_parameter_offset=float(norm_offset),
        input_consumers=input_source,
        post_norm=post_norm,
        router=router,
        routed_layout=routed_layout,
        experts=expected_experts,
        intermediate=intermediate,
        packed_gate_up=(packed_gate_up if routed_layout == "packed_3d" else None),
        packed_down=(packed_down if routed_layout == "packed_3d" else None),
        per_expert_roles=(per_roles if routed_layout == "per_expert_2d" else []),
        shared_roles=shared_roles,
    )
    return graph


def _mean_importance(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise RuntimeError("PrismaSnap MoE objective has no importance vectors")
    return torch.stack([value.to(torch.float64) for value in values], dim=0).mean(dim=0)


def _consumer(
    *,
    name: str,
    weight: torch.Tensor,
    importance: torch.Tensor,
    mode: str,
    codec_group: str | None = None,
) -> PrismaSnapConsumer:
    return PrismaSnapConsumer(
        name=name,
        weight=weight,
        importance=importance,
        mode=mode,  # type: ignore[arg-type]
        codec_group_name=codec_group,
    )


def _plan_layer(
    graph: Mapping[str, object],
    *,
    source,
    stats: Mapping[str, dict[str, object]],
    device: torch.device,
    config: PrismaSnapSearchConfig,
    top_k: int,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    layer = int(graph["layer"])
    weights_to_load = set(str(name) for name in graph["input_consumers"])
    weights_to_load.update(str(name) for name in graph["compensation_consumers"])
    if graph["routed_layout"] == "packed_3d":
        weights_to_load.update((str(graph["packed_gate_up"]), str(graph["packed_down"])))
    else:
        for role in graph["per_expert_roles"]:
            weights_to_load.update(str(role[name]) for name in ("gate", "up", "down"))
    for role in graph["shared_roles"]:
        weights_to_load.update(str(role[name]) for name in ("gate", "up", "down"))
    weights = {name: source.load(name, device) for name in sorted(weights_to_load)}
    input_norm = source.load(str(graph["input_norm"]), device)
    post_norm = source.load(str(graph["post_norm"]), device)

    input_consumers: list[PrismaSnapConsumer] = []
    input_importances: list[torch.Tensor] = []
    for recipe, source_name in zip(graph["input_recipe_consumers"], graph["input_consumers"]):
        importance = _dense_importance(stats[str(recipe)], device)
        input_importances.append(importance)
        input_consumers.append(
            _consumer(name=str(source_name), weight=weights[str(source_name)], importance=importance, mode="column_inverse")
        )
    input_scale, input_stats = search_diagonal_scale(
        input_consumers, _mean_importance(input_importances), config=config
    )

    post_consumers: list[PrismaSnapConsumer] = []
    post_importances: list[torch.Tensor] = []
    packed_gate_up: PackedGateUp | None = None
    packed_down: PackedDown | None = None
    per_expert_objectives: list[
        tuple[
            str,
            str,
            str,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = []
    if graph["routed_layout"] == "packed_3d":
        probe = graph["routed_probe"]
        gate_up_importance = _packed_importance(stats[str(probe["gate_up"])], device)
        down_importance = _packed_importance(stats[str(probe["down"])], device)
        packed_gate_up = PackedGateUp(
            name=str(graph["packed_gate_up"]),
            weight=weights[str(graph["packed_gate_up"])],
            importance=gate_up_importance,
            gate_rows=(0, int(graph["intermediate"])),
            up_rows=(int(graph["intermediate"]), 2 * int(graph["intermediate"])),
        )
        packed_down = PackedDown(
            name=str(graph["packed_down"]),
            weight=weights[str(graph["packed_down"])],
            importance=down_importance,
        )
        post_consumers.extend(
            packed_post_norm_consumers(packed_gate_up, config=config)
        )
        post_importances.extend(gate_up_importance[index] for index in range(int(graph["experts"])))
    else:
        probe = graph["routed_probe"]
        packed_gate_up_importance = _packed_importance(
            stats[str(probe["gate_up"])], device
        )
        packed_down_importance = _packed_importance(
            stats[str(probe["down"])], device
        )
        for source_role in graph["per_expert_roles"]:
            expert = int(source_role["expert"])
            # Both physical gate/up Linears consume the same routed hidden
            # rows represented by one packed live-model probe slice.
            gate_imp = packed_gate_up_importance[expert]
            up_imp = packed_gate_up_importance[expert]
            down_imp = packed_down_importance[expert]
            gate_name, up_name, down_name = (str(source_role[name]) for name in ("gate", "up", "down"))
            logical_gate_up = f"layer:{layer}:routed:{source_role['expert']}:gate_up"
            post_consumers.extend(
                [
                    _consumer(
                        name=gate_name,
                        weight=weights[gate_name],
                        importance=gate_imp,
                        mode="column_inverse",
                        codec_group=logical_gate_up,
                    ),
                    _consumer(
                        name=up_name,
                        weight=weights[up_name],
                        importance=up_imp,
                        mode="column_inverse",
                        codec_group=logical_gate_up,
                    ),
                ]
            )
            post_importances.extend((gate_imp, up_imp))
            per_expert_objectives.append(
                (
                    gate_name,
                    up_name,
                    down_name,
                    gate_imp,
                    up_imp,
                    down_imp,
                    weights[gate_name],
                    weights[up_name],
                )
            )

    shared_objectives: list[
        tuple[
            str,
            str,
            str,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = []
    for source_role, probe_role in zip(graph["shared_roles"], graph["shared_probe"]):
        gate_imp = _dense_importance(stats[str(probe_role["gate"])], device)
        up_imp = _dense_importance(stats[str(probe_role["up"])], device)
        down_imp = _dense_importance(stats[str(probe_role["down"])], device)
        gate_name, up_name, down_name = (str(source_role[name]) for name in ("gate", "up", "down"))
        logical_gate_up = f"layer:{layer}:shared:{source_role['shared_index']}:gate_up"
        post_consumers.extend(
            [
                _consumer(
                    name=gate_name,
                    weight=weights[gate_name],
                    importance=gate_imp,
                    mode="column_inverse",
                    codec_group=logical_gate_up,
                ),
                _consumer(
                    name=up_name,
                    weight=weights[up_name],
                    importance=up_imp,
                    mode="column_inverse",
                    codec_group=logical_gate_up,
                ),
            ]
        )
        post_importances.extend((gate_imp, up_imp))
        shared_objectives.append(
            (
                gate_name,
                up_name,
                down_name,
                gate_imp,
                up_imp,
                down_imp,
                weights[gate_name],
                weights[up_name],
            )
        )

    post_scale, post_stats = search_diagonal_scale(
        post_consumers, _mean_importance(post_importances), config=config
    )

    # The expert seam is composed after the post-norm fold. Search against
    # the exact sequential-BF16 intermediate instead of independently scoring
    # source bytes with a stale gate/up global.
    if packed_gate_up is not None and packed_down is not None:
        routed_scales, raw_routed_stats = search_packed_up_down_scales(
            packed_gate_up,
            packed_down,
            config=config,
            post_norm_scale=post_scale,
        )
        routed_stats = [_normalized_stats(row) for row in raw_routed_stats]
    else:
        per_scales: list[torch.Tensor] = []
        routed_stats = []
        for (
            gate_name,
            up_name,
            down_name,
            gate_imp,
            up_imp,
            down_imp,
            gate_weight,
            up_weight,
        ) in per_expert_objectives:
            logical_gate_up = f"layer:{layer}:routed:{len(per_scales)}:gate_up"
            folded_gate = (
                gate_weight.to(torch.float64)
                / post_scale.to(device=device, dtype=torch.float64).view(1, -1)
            ).to(gate_weight.dtype)
            folded_up = (
                up_weight.to(torch.float64)
                / post_scale.to(device=device, dtype=torch.float64).view(1, -1)
            ).to(up_weight.dtype)
            folded_up_imp = (
                up_imp.to(torch.float64)
                * post_scale.to(device=device, dtype=torch.float64).square()
            ).to(up_imp.dtype)
            folded_gate_imp = (
                gate_imp.to(torch.float64)
                * post_scale.to(device=device, dtype=torch.float64).square()
            ).to(gate_imp.dtype)
            scale, row = search_diagonal_scale(
                [
                    _consumer(
                        name=gate_name,
                        weight=folded_gate,
                        importance=folded_gate_imp,
                        mode="stationary",
                        codec_group=logical_gate_up,
                    ),
                    _consumer(
                        name=down_name,
                        weight=weights[down_name],
                        importance=down_imp,
                        mode="column_inverse",
                    ),
                    _consumer(
                        name=up_name,
                        weight=folded_up,
                        importance=folded_up_imp,
                        mode="row",
                        codec_group=logical_gate_up,
                    ),
                ],
                down_imp,
                config=config,
            )
            per_scales.append(scale)
            routed_stats.append(_normalized_stats(row))
        routed_scales = torch.stack(per_scales, dim=0)

    shared_scales: list[torch.Tensor] = []
    shared_stats: list[dict[str, object]] = []
    for (
        gate_name,
        up_name,
        down_name,
        gate_imp,
        up_imp,
        down_imp,
        gate_weight,
        up_weight,
    ) in shared_objectives:
        logical_gate_up = f"layer:{layer}:shared:{len(shared_scales)}:gate_up"
        folded_gate = (
            gate_weight.to(torch.float64)
            / post_scale.to(device=device, dtype=torch.float64).view(1, -1)
        ).to(gate_weight.dtype)
        folded_up = (
            up_weight.to(torch.float64)
            / post_scale.to(device=device, dtype=torch.float64).view(1, -1)
        ).to(up_weight.dtype)
        folded_up_imp = (
            up_imp.to(torch.float64)
            * post_scale.to(device=device, dtype=torch.float64).square()
        ).to(up_imp.dtype)
        folded_gate_imp = (
            gate_imp.to(torch.float64)
            * post_scale.to(device=device, dtype=torch.float64).square()
        ).to(gate_imp.dtype)
        scale, row = search_diagonal_scale(
            [
                _consumer(
                    name=gate_name,
                    weight=folded_gate,
                    importance=folded_gate_imp,
                    mode="stationary",
                    codec_group=logical_gate_up,
                ),
                _consumer(
                    name=down_name,
                    weight=weights[down_name],
                    importance=down_imp,
                    mode="column_inverse",
                ),
                _consumer(
                    name=up_name,
                    weight=folded_up,
                    importance=folded_up_imp,
                    mode="row",
                    codec_group=logical_gate_up,
                ),
            ],
            down_imp,
            config=config,
        )
        shared_scales.append(scale)
        shared_stats.append(_normalized_stats(row))

    vector_names = {
        "input": f"layer_{layer:05d}_input",
        "post": f"layer_{layer:05d}_post",
        "routed": f"layer_{layer:05d}_routed_updown",
    }
    scale_tensors = {
        vector_names["input"]: input_scale.cpu(),
        vector_names["post"]: post_scale.cpu(),
        vector_names["routed"]: routed_scales.cpu(),
    }
    for index, scale in enumerate(shared_scales):
        scale_tensors[f"layer_{layer:05d}_shared_{index:03d}_updown"] = scale.cpu()

    graph_sha = str(graph["graph_sha256"])
    if graph["routed_layout"] == "packed_3d":
        post_objective_names = [str(graph["packed_gate_up"])]
    else:
        post_objective_names = [
            str(role[name])
            for role in graph["per_expert_roles"]
            for name in ("gate", "up")
        ]
    for role in graph["shared_roles"]:
        post_objective_names.extend((str(role["gate"]), str(role["up"])))

    seams: list[dict[str, object]] = [
        {
            "layer": layer,
            "kind": "input_norm",
            "vector": vector_names["input"],
            "norm": graph["input_norm"],
            "norm_parameter_offset": graph["norm_parameter_offset"],
            "consumers": list(graph["input_consumers"]),
            "stats": _normalized_stats(input_stats),
            "graph_sha256": graph_sha,
        },
        {
            "layer": layer,
            "kind": "post_attention_norm",
            "vector": vector_names["post"],
            "norm": graph["post_norm"],
            "norm_parameter_offset": graph["norm_parameter_offset"],
            "objective_consumers": post_objective_names,
            "compensation_only_consumers": list(graph["compensation_consumers"]),
            "router": graph["router"],
            "routed_layout": graph["routed_layout"],
            "stats": _normalized_stats(post_stats),
            "graph_sha256": graph_sha,
        },
    ]
    if graph["routed_layout"] == "packed_3d":
        routed_seam: dict[str, object] = {
            "layer": layer,
            "kind": "routed_up_down",
            "vector": vector_names["routed"],
            "layout": "packed_3d",
            "experts": int(graph["experts"]),
            "gate_up": graph["packed_gate_up"],
            "down": graph["packed_down"],
            "expert_axis": 0,
            "row_axis": 1,
            "input_axis": 2,
            "gate_rows": [0, int(graph["intermediate"])],
            "up_rows": [int(graph["intermediate"]), 2 * int(graph["intermediate"])],
            "stats": routed_stats,
            "graph_sha256": graph_sha,
        }
    else:
        routed_seam = {
            "layer": layer,
            "kind": "routed_up_down",
            "vector": vector_names["routed"],
            "layout": "per_expert_2d",
            "experts": int(graph["experts"]),
            "roles": [dict(role) for role in graph["per_expert_roles"]],
            "stats": routed_stats,
            "graph_sha256": graph_sha,
        }
    seams.append(routed_seam)
    for index, (role, stat) in enumerate(zip(graph["shared_roles"], shared_stats)):
        seams.append(
            {
                "layer": layer,
                "kind": "shared_up_down",
                "shared_index": index,
                "vector": f"layer_{layer:05d}_shared_{index:03d}_updown",
                "output_gate": role["output_gate"],
                "gate": role["gate"],
                "up": role["up"],
                "down": role["down"],
                "stats": stat,
                "graph_sha256": graph_sha,
            }
        )

    transforms: list[dict[str, object]] = []
    for norm, vector in ((graph["input_norm"], vector_names["input"]), (graph["post_norm"], vector_names["post"])):
        transforms.append(
            {
                "tensor": norm,
                "vector": vector,
                "operation": "affine_multiply",
                "axis": 0,
                "order": 0,
                "parameter_offset": graph["norm_parameter_offset"],
            }
        )
    transforms.extend(
        {
            "tensor": name,
            "vector": vector_names["input"],
            "operation": "divide",
            "axis": 1,
            "order": 0,
        }
        for name in graph["input_consumers"]
    )
    post_terms = [
        *seams[1]["objective_consumers"],
        *graph["compensation_consumers"],
    ]
    for name in post_terms:
        shape = weights[str(name)].shape
        transforms.append(
            {
                "tensor": name,
                "vector": vector_names["post"],
                "operation": "divide",
                "axis": 2 if len(shape) == 3 else 1,
                "order": 0,
            }
        )
    if graph["routed_layout"] == "packed_3d":
        transforms.extend(
            [
                {
                    "tensor": graph["packed_gate_up"],
                    "vector": vector_names["routed"],
                    "operation": "multiply",
                    "axis": 1,
                    "order": 1,
                    "expert_axis": 0,
                    "channel_start": int(graph["intermediate"]),
                    "channel_stop": 2 * int(graph["intermediate"]),
                },
                {
                    "tensor": graph["packed_down"],
                    "vector": vector_names["routed"],
                    "operation": "divide",
                    "axis": 2,
                    "order": 0,
                    "expert_axis": 0,
                    "channel_start": 0,
                    "channel_stop": int(graph["intermediate"]),
                },
            ]
        )
    else:
        for role in graph["per_expert_roles"]:
            expert = int(role["expert"])
            transforms.extend(
                [
                    {
                        "tensor": role["up"],
                        "vector": vector_names["routed"],
                        "operation": "multiply",
                        "axis": 0,
                        "order": 1,
                        "vector_index": expert,
                    },
                    {
                        "tensor": role["down"],
                        "vector": vector_names["routed"],
                        "operation": "divide",
                        "axis": 1,
                        "order": 0,
                        "vector_index": expert,
                    },
                ]
            )
    for index, role in enumerate(graph["shared_roles"]):
        vector = f"layer_{layer:05d}_shared_{index:03d}_updown"
        transforms.extend(
            [
                {
                    "tensor": role["up"],
                    "vector": vector,
                    "operation": "multiply",
                    "axis": 0,
                    "order": 1,
                },
                {
                    "tensor": role["down"],
                    "vector": vector,
                    "operation": "divide",
                    "axis": 1,
                    "order": 0,
                },
            ]
        )

    if graph["routed_layout"] == "packed_3d":
        gate_up_for_gate = weights[str(graph["packed_gate_up"])]
        down_for_gate = weights[str(graph["packed_down"])]
    else:
        # Keep source-native per-expert tensors as views/references. Stacking
        # them would duplicate the complete routed bank solely for a gate that
        # touches at most T*top_k deterministic slices.
        gate_up_for_gate = [
            (weights[str(role["gate"])], weights[str(role["up"])])
            for role in graph["per_expert_roles"]
        ]
        down_for_gate = [
            weights[str(role["down"])] for role in graph["per_expert_roles"]
        ]
    # Deterministic, bounded input rows. RMS normalization's scalar denominator
    # is independent of gamma, so x*gamma is sufficient for this scale algebra.
    hidden_size = int(input_scale.numel())
    base = torch.linspace(-0.75, 0.75, hidden_size, device=device, dtype=torch.float64)
    hidden = torch.stack((base, torch.sin(base * 3.0), torch.cos(base * 5.0)), dim=0)
    router_weight = weights[str(graph["router"])]
    invariance = fp64_router_and_expert_invariance(
        hidden=hidden,
        norm_gamma=post_norm.to(torch.float64) + float(graph["norm_parameter_offset"]),
        norm_scale=post_scale,
        router_weight=router_weight,
        gate_up=gate_up_for_gate,
        down=down_for_gate,
        expert_scales=routed_scales,
        top_k=top_k,
        shared_gate_up=[
            torch.cat(
                (weights[str(role["gate"])], weights[str(role["up"])]),
                dim=0,
            )
            for role in graph["shared_roles"]
        ],
        shared_down=[
            weights[str(role["down"])] for role in graph["shared_roles"]
        ],
        shared_output_gate_weight=[
            weights[str(name)] for name in graph["shared_output_gates"]
        ],
        shared_expert_scales=shared_scales,
    )

    # Every additional router-like direct consumer and every token-mixer/input
    # consumer is checked independently; shared up/down products are checked by
    # sampled exact products just like dense v1.
    worst = max(invariance.values())
    gamma_in = input_norm.to(torch.float64) + float(graph["norm_parameter_offset"])
    for norm_gamma, scale, names in (
        (gamma_in, input_scale, graph["input_consumers"]),
        (
            post_norm.to(torch.float64) + float(graph["norm_parameter_offset"]),
            post_scale,
            graph["compensation_consumers"],
        ),
    ):
        folded_gamma = norm_gamma * scale
        for name in names:
            weight = weights[str(name)].to(torch.float64)
            rows = torch.linspace(0, weight.shape[0] - 1, min(7, weight.shape[0]), device=device).round().long()
            cols = torch.linspace(0, weight.shape[1] - 1, min(17, weight.shape[1]), device=device).round().long()
            before = weight[rows][:, cols] * norm_gamma[cols]
            after = (weight[rows][:, cols] / scale[cols]) * folded_gamma[cols]
            worst = max(worst, float((before - after).abs().max().item()))
    for role, scale in zip(graph["shared_roles"], shared_scales):
        up = weights[str(role["up"])].to(torch.float64)
        down = weights[str(role["down"])].to(torch.float64)
        before = down[:, : min(17, down.shape[1])].unsqueeze(2) * up[: min(17, up.shape[0]), : min(7, up.shape[1])].unsqueeze(0)
        after = (down[:, : min(17, down.shape[1])] / scale[: min(17, scale.numel())]).unsqueeze(2) * (up[: min(17, up.shape[0]), : min(7, up.shape[1])] * scale[: min(17, scale.numel())].view(-1, 1)).unsqueeze(0)
        worst = max(worst, float((before - after).abs().max().item()))
    if not math.isfinite(worst) or worst > 1e-10:
        raise RuntimeError(f"PrismaSnap MoE layer {layer} fp64 gate failed: {worst}")
    invariance["fp64_invariance_max_abs"] = worst
    return scale_tensors, seams, transforms, invariance


def _seam_sort_key(row: Mapping[str, object]) -> tuple[int, int, int]:
    order = {"input_norm": 0, "post_attention_norm": 1, "routed_up_down": 2, "shared_up_down": 3}
    return (int(row["layer"]), order[str(row["kind"])], int(row.get("shared_index", -1)))


def plan_moe_checkpoint(
    source_dir: str | Path,
    probe_path: str | Path,
    source_identity_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
    layers: Sequence[int] | None = None,
    search_config: PrismaSnapSearchConfig | None = None,
    tensor_metadata_manifest_path: str | Path | None = None,
    verify_source_content: bool = True,
    resume: bool = False,
    production: bool = False,
) -> dict[str, object]:
    """Build one opt-in MoE research plan from streamed layer tensors."""
    if production:
        raise RuntimeError(
            "PrismaSnap MoE is research-only and cannot be production-promoted "
            "without real MoE fold-fidelity and served-KL evidence"
        )
    checkpoint = _checkpoint_module()
    source = checkpoint._Checkpoint(Path(source_dir), require_all_shards=False)
    identity, portable, identity_file_sha256 = checkpoint._validate_source_identity(
        source,
        Path(source_identity_path).resolve(strict=True),
        verify_content=verify_source_content,
    )
    requested_probe = Path(probe_path)
    if requested_probe.is_symlink() or not requested_probe.is_file():
        raise RuntimeError("PrismaSnap MoE probe must be a regular local file")
    probe_file = requested_probe.resolve(strict=True)
    stats, probe_meta, probe_sha256 = _load_moe_probe(probe_file)
    checkpoint._validate_probe_source_contract(probe_meta, source)
    config_payload = checkpoint._load_json(source.root / "config.json", where="model config")
    text_config = config_payload.get("text_config")
    cfg_payload = text_config if isinstance(text_config, Mapping) else config_payload
    hidden_size = int(cfg_payload.get("hidden_size", config_payload.get("hidden_size", 0)))
    layer_count = int(cfg_payload.get("num_hidden_layers", config_payload.get("num_hidden_layers", 0)))
    expected_experts = int(
        cfg_payload.get("num_experts", cfg_payload.get("num_local_experts", 0))
    )
    top_k = int(cfg_payload.get("num_experts_per_tok", cfg_payload.get("num_experts_per_token", 0)))
    if hidden_size <= 0 or layer_count <= 0 or expected_experts <= 0 or not 1 <= top_k <= expected_experts:
        raise RuntimeError("PrismaSnap MoE config lacks hidden/layer/expert/top-k contract")
    requested_layers = list(range(layer_count)) if layers is None else sorted(set(layers))
    if not requested_layers or requested_layers[0] < 0 or requested_layers[-1] >= layer_count:
        raise ValueError("PrismaSnap MoE requested layers are empty/out of range")
    profile = detect_profile(str(source.root))
    if profile.prismasnap_moe_layer_contract(requested_layers[0]) is None:
        raise RuntimeError(f"profile {profile.name!r} has no PrismaSnap MoE contract")
    config = search_config or PrismaSnapSearchConfig()
    execution_device = torch.device(device)
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PrismaSnap MoE planning requested unavailable CUDA")
    producer = _moe_producer_identity()

    if tensor_metadata_manifest_path is None:
        tensor_metadata = checkpoint._scan_checkpoint_tensor_metadata(source)
        tensor_binding: dict[str, object] = {
            "mode": "inline_full_header_scan",
            "manifest_sha256": None,
            "tensor_metadata_sha256": tensor_metadata["sha256"],
        }
    else:
        manifest = checkpoint._validate_tensor_metadata_manifest(
            Path(tensor_metadata_manifest_path).resolve(strict=True),
            identity=identity,
            portable=portable,
            # Header scanning is common dense/MoE infrastructure. Its own
            # receipt binds the common producer closure; the MoE plan then
            # embeds the validated metadata and binds the extended producer.
            producer=checkpoint._producer_identity(),
        )
        tensor_metadata = dict(manifest["tensor_metadata"])
        tensor_binding = {
            "mode": "external_manifest",
            "manifest_sha256": manifest["manifest_sha256"],
            "tensor_metadata_sha256": tensor_metadata["sha256"],
        }
    tensor_rows = tensor_metadata["tensors"]

    output = Path(output_dir)
    if os.path.lexists(output):
        if resume and not output.is_symlink() and output.is_dir():
            existing, _ = checkpoint.load_plan(output)
            if (
                existing.get("schema") not in {MOE_PLAN_SCHEMA, MOE_PLAN_SET_SCHEMA}
                or existing.get("producer") != producer
                or existing.get("source", {}).get("portable_identity") != portable
                or existing.get("probe", {}).get("sha256") != probe_sha256
                or existing.get("model", {}).get("planned_layers") != requested_layers
            ):
                raise RuntimeError("resumed PrismaSnap MoE plan belongs to different inputs")
            return existing
        raise RuntimeError(f"PrismaSnap MoE plan output exists: {output}")
    staging = output.with_name(output.name + ".prismasnap-plan-incomplete")
    if os.path.lexists(staging):
        if not resume or staging.is_symlink() or not staging.is_dir():
            raise RuntimeError(f"stale PrismaSnap MoE plan staging exists: {staging}")
        if (staging / checkpoint.PLAN_JSON).is_file() and (staging / checkpoint.PLAN_SCALES).is_file():
            existing, _ = checkpoint.load_plan(staging)
            os.replace(staging, output)
            checkpoint._fsync_dir(output.parent)
            return existing
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)

    scale_tensors: dict[str, torch.Tensor] = {}
    seams: list[dict[str, object]] = []
    transforms: list[dict[str, object]] = []
    expert_counts: dict[str, int] = {}
    routed_layouts: dict[str, str] = {}
    aggregate_invariance = {
        "fp64_invariance_max_abs": 0.0,
        "router_logit_max_abs": 0.0,
        "route_weight_max_abs": 0.0,
        "routed_output_max_abs": 0.0,
        "routing_changed": 0.0,
    }
    for ordinal, layer in enumerate(requested_layers, start=1):
        print(
            f"[prismasnap-moe-plan] layer {layer} "
            f"({ordinal}/{len(requested_layers)}) graph/load/search",
            flush=True,
        )
        graph = _layer_source_graph(
            layer=layer,
            hidden_size=hidden_size,
            expected_experts=expected_experts,
            stats=stats,
            profile=profile,
            source=source,
            tensor_rows=tensor_rows,
        )
        layer_scales, layer_seams, layer_transforms, invariance = _plan_layer(
            graph,
            source=source,
            stats=stats,
            device=execution_device,
            config=config,
            top_k=top_k,
        )
        if set(scale_tensors) & set(layer_scales):
            raise RuntimeError("PrismaSnap MoE scale vector collision")
        scale_tensors.update(layer_scales)
        seams.extend(layer_seams)
        transforms.extend(layer_transforms)
        expert_counts[str(layer)] = int(graph["experts"])
        routed_layouts[str(layer)] = str(graph["routed_layout"])
        for key in aggregate_invariance:
            aggregate_invariance[key] = max(
                aggregate_invariance[key], float(invariance.get(key, 0.0))
            )

    save_file(
        {name: value.contiguous() for name, value in sorted(scale_tensors.items())},
        str(staging / checkpoint.PLAN_SCALES),
        metadata={"format": "pt", "algorithm": PRISMASNAP_MOE_ALGORITHM},
    )
    with (staging / checkpoint.PLAN_SCALES).open("rb") as handle:
        os.fsync(handle.fileno())
    plan: dict[str, object] = {
        "schema": MOE_PLAN_SCHEMA,
        "state": "PLANNED",
        "algorithm": PRISMASNAP_MOE_ALGORITHM,
        "producer": producer,
        "profile": profile.name,
        "source": {
            "identity": identity,
            "portable_identity": portable,
            "identity_file_sha256": identity_file_sha256,
        },
        "probe": {
            "path": str(probe_file),
            "sha256": probe_sha256,
            **{
                key: probe_meta.get(key)
                for key in (
                    "calib_hash",
                    "dataset",
                    "nsamples",
                    "seqlen",
                    "calibration_modality",
                    "model",
                    "dtype",
                    "device_map",
                    "execution_device",
                )
            },
            "legacy_text_binding": None,
        },
        "model": {
            "hidden_size": hidden_size,
            "layer_count": layer_count,
            "planned_layers": requested_layers,
            "excluded_prefixes": ["model.visual.", "mtp."],
            "expert_counts": expert_counts,
            "routed_layouts": routed_layouts,
        },
        "search": moe_search_contract(config),
        "tensor_metadata": tensor_metadata,
        "tensor_metadata_binding": tensor_binding,
        "scales": {
            "file": checkpoint.PLAN_SCALES,
            "sha256": checkpoint._sha256_file(staging / checkpoint.PLAN_SCALES),
            "vectors": len(scale_tensors),
        },
        "seams": sorted(seams, key=_seam_sort_key),
        "transforms": sorted(transforms, key=lambda row: (str(row["tensor"]), int(row["order"]))),
        "verification": {
            **aggregate_invariance,
            "threshold": 1e-10,
            "domain": "pre_cast_fp64_router_routing_and_expert_algebra",
            "required_bf16_fold_kl_max": 5e-4,
            "real_moe_fold_kl_evidence": None,
        },
        "promotion": PRISMASNAP_MOE_PROMOTION,
    }
    if _moe_producer_identity() != producer:
        raise RuntimeError("PrismaSnap MoE producer changed while planning")
    source.verify_stable()
    plan["plan_sha256"] = checkpoint._plan_digest(plan)
    checkpoint._atomic_json(staging / checkpoint.PLAN_JSON, plan)
    validate_moe_plan_semantics(plan, scale_tensors)
    checkpoint._fsync_dir(staging)
    os.replace(staging, output)
    checkpoint._fsync_dir(output.parent)
    return plan


def plan_scale_vector_names(plan: Mapping[str, object]) -> set[str]:
    result = {
        str(row["vector"])
        for row in plan.get("seams", [])
        if isinstance(row, Mapping) and isinstance(row.get("vector"), str)
    }
    return result


def _validate_stat(
    value: object,
    *,
    channels: int,
    search: Mapping[str, object],
    where: str,
) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{where} is malformed")
    _exact_keys(value, _STATS_KEYS, where)
    if value.get("algorithm") != PRISMASNAP_MOE_ALGORITHM:
        raise RuntimeError(f"{where} algorithm differs")
    for key in ("error_baseline", "error_final", "improvement_fraction"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise RuntimeError(f"{where}.{key} is not finite")
    baseline = float(value["error_baseline"])
    final = float(value["error_final"])
    improvement = float(value["improvement_fraction"])
    expected = 0.0 if baseline == 0 else (baseline - final) / baseline
    ints = ("groups", "groups_moved", "rounds", "candidate_count", "polish_pool", "polished")
    if any(type(value.get(key)) is not int for key in ints):
        raise RuntimeError(f"{where} integer fields are malformed")
    if (
        baseline < 0
        or final < 0
        or final > baseline
        or not math.isclose(improvement, expected, rel_tol=1e-12, abs_tol=1e-15)
        or int(value["groups"]) * int(search["group_size"]) != channels
        or not 0 <= int(value["groups_moved"]) <= int(value["groups"])
        or not 1 <= int(value["rounds"]) <= int(search["max_rounds"])
        or int(value["candidate_count"]) != len(search["alphas"])
        or type(value.get("fell_back")) is not bool
        or value.get("variant") != search["variant"]
    ):
        raise RuntimeError(f"{where} violates the MoE search contract")


def _validate_search(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError("PrismaSnap MoE search is malformed")
    extras = {
        "expert_global_scope",
        "routed_gate_up_global_scope",
        "shared_gate_up_global_scope",
        "down_global_scope",
        "packed_expert_axis",
        "router_codec_objective",
        "expert_seam_scope",
        "expert_coverage_policy",
        "promotion",
    }
    checkpoint = _checkpoint_module()
    dense_part = {key: item for key, item in value.items() if key not in extras}
    dense_part["algorithm"] = checkpoint.PRISMASNAP_ALGORITHM
    canonical_dense = checkpoint._validate_search_contract(dense_part)
    config = PrismaSnapSearchConfig(
        group_size=int(canonical_dense["group_size"]),
        alphas=tuple(float(v) for v in canonical_dense["alphas"]),
        max_rounds=int(canonical_dense["max_rounds"]),
        stage="stage" in canonical_dense["variant"],
        polish="polish" in canonical_dense["variant"],
        polish_top=int(canonical_dense["polish_top"]),
        polish_pool=int(canonical_dense["polish_pool"]),
        scale_rule=str(canonical_dense["nvfp4_scale_rule"]),
        snapped_scale_scoring=bool(canonical_dense["nvfp4_snapped_scale_scoring"]),
        joint_scale_levels=tuple(float(v) for v in canonical_dense["nvfp4_joint_scale_levels"]),
    )
    canonical = moe_search_contract(config)
    if dict(value) != canonical:
        raise RuntimeError("PrismaSnap MoE search is noncanonical")
    return canonical


def _expected_moe_transforms(
    seams: Sequence[Mapping[str, object]],
    tensor_rows: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    by_layer: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for seam in seams:
        by_layer[int(seam["layer"])].append(seam)
    transforms: list[dict[str, object]] = []
    for layer in sorted(by_layer):
        rows = by_layer[layer]
        input_row = next(row for row in rows if row["kind"] == "input_norm")
        post_row = next(row for row in rows if row["kind"] == "post_attention_norm")
        routed = next(row for row in rows if row["kind"] == "routed_up_down")
        for row in (input_row, post_row):
            transforms.append(
                {
                    "tensor": row["norm"],
                    "vector": row["vector"],
                    "operation": "affine_multiply",
                    "axis": 0,
                    "order": 0,
                    "parameter_offset": row["norm_parameter_offset"],
                }
            )
        transforms.extend(
            {
                "tensor": name,
                "vector": input_row["vector"],
                "operation": "divide",
                "axis": 1,
                "order": 0,
            }
            for name in input_row["consumers"]
        )
        for name in [*post_row["objective_consumers"], *post_row["compensation_only_consumers"]]:
            transforms.append(
                {
                    "tensor": name,
                    "vector": post_row["vector"],
                    "operation": "divide",
                    "axis": 2 if len(tensor_rows[str(name)]["shape"]) == 3 else 1,
                    "order": 0,
                }
            )
        if routed["layout"] == "packed_3d":
            transforms.extend(
                [
                    {
                        "tensor": routed["gate_up"],
                        "vector": routed["vector"],
                        "operation": "multiply",
                        "axis": routed["row_axis"],
                        "order": 1,
                        "expert_axis": routed["expert_axis"],
                        "channel_start": routed["up_rows"][0],
                        "channel_stop": routed["up_rows"][1],
                    },
                    {
                        "tensor": routed["down"],
                        "vector": routed["vector"],
                        "operation": "divide",
                        "axis": routed["input_axis"],
                        "order": 0,
                        "expert_axis": routed["expert_axis"],
                        "channel_start": 0,
                        "channel_stop": routed["up_rows"][1] - routed["up_rows"][0],
                    },
                ]
            )
        else:
            for role in routed["roles"]:
                transforms.extend(
                    [
                        {
                            "tensor": role["up"],
                            "vector": routed["vector"],
                            "operation": "multiply",
                            "axis": 0,
                            "order": 1,
                            "vector_index": role["expert"],
                        },
                        {
                            "tensor": role["down"],
                            "vector": routed["vector"],
                            "operation": "divide",
                            "axis": 1,
                            "order": 0,
                            "vector_index": role["expert"],
                        },
                    ]
                )
        for row in (row for row in rows if row["kind"] == "shared_up_down"):
            transforms.extend(
                [
                    {
                        "tensor": row["up"],
                        "vector": row["vector"],
                        "operation": "multiply",
                        "axis": 0,
                        "order": 1,
                    },
                    {
                        "tensor": row["down"],
                        "vector": row["vector"],
                        "operation": "divide",
                        "axis": 1,
                        "order": 0,
                    },
                ]
            )
    return sorted(transforms, key=lambda row: (str(row["tensor"]), int(row["order"])))


def validate_moe_plan_semantics(
    plan: Mapping[str, object],
    scales: Mapping[str, torch.Tensor],
) -> None:
    schema = plan.get("schema")
    _exact_keys(plan, _PLAN_KEYS if schema == MOE_PLAN_SCHEMA else _PLAN_SET_KEYS, "PrismaSnap MoE plan")
    if (
        schema not in {MOE_PLAN_SCHEMA, MOE_PLAN_SET_SCHEMA}
        or plan.get("state") != "PLANNED"
        or plan.get("algorithm") != PRISMASNAP_MOE_ALGORITHM
        or plan.get("promotion") != PRISMASNAP_MOE_PROMOTION
        or not isinstance(plan.get("producer"), Mapping)
        or not isinstance(plan.get("profile"), str)
    ):
        raise RuntimeError("PrismaSnap MoE plan header is malformed")
    model = plan.get("model")
    if not isinstance(model, Mapping):
        raise RuntimeError("PrismaSnap MoE model contract is malformed")
    _exact_keys(model, _MODEL_KEYS, "PrismaSnap MoE model")
    hidden = model.get("hidden_size")
    layers = model.get("planned_layers")
    layer_count = model.get("layer_count")
    if (
        type(hidden) is not int
        or hidden <= 0
        or type(layer_count) is not int
        or layer_count <= 0
        or not isinstance(layers, list)
        or not layers
        or layers != sorted(set(layers))
        or any(type(layer) is not int or not 0 <= layer < layer_count for layer in layers)
        or model.get("excluded_prefixes") != ["model.visual.", "mtp."]
        or not isinstance(model.get("expert_counts"), dict)
        or set(model["expert_counts"]) != {str(layer) for layer in layers}
        or not isinstance(model.get("routed_layouts"), dict)
        or set(model["routed_layouts"]) != {str(layer) for layer in layers}
    ):
        raise RuntimeError("PrismaSnap MoE model contract is invalid")
    search = _validate_search(plan.get("search"))
    checkpoint = _checkpoint_module()
    tensors = checkpoint._validate_tensor_metadata_contract(plan)
    scale_meta = plan.get("scales")
    if (
        not isinstance(scale_meta, Mapping)
        or scale_meta.get("file") != checkpoint.PLAN_SCALES
        or type(scale_meta.get("vectors")) is not int
        or scale_meta.get("vectors") != len(scales)
        or set(scales) != plan_scale_vector_names(plan)
    ):
        raise RuntimeError("PrismaSnap MoE scale census is malformed")
    seams = plan.get("seams")
    if not isinstance(seams, list) or not seams:
        raise RuntimeError("PrismaSnap MoE seam census is empty")
    if seams != sorted(seams, key=_seam_sort_key):
        raise RuntimeError("PrismaSnap MoE seam order is noncanonical")
    by_layer: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for index, row in enumerate(seams):
        if not isinstance(row, Mapping) or type(row.get("layer")) is not int or row["layer"] not in layers:
            raise RuntimeError(f"PrismaSnap MoE seam {index} is malformed")
        kind = row.get("kind")
        expected_keys = {
            "input_norm": _NORM_KEYS,
            "post_attention_norm": _POST_KEYS,
            "shared_up_down": _SHARED_KEYS,
        }.get(str(kind))
        if kind == "routed_up_down":
            expected_keys = _PACKED_ROUTED_KEYS if row.get("layout") == "packed_3d" else _PER_EXPERT_ROUTED_KEYS
        if expected_keys is None:
            raise RuntimeError(f"PrismaSnap MoE seam {index} has unknown kind")
        _exact_keys(row, expected_keys, f"PrismaSnap MoE seam {index}")
        by_layer[int(row["layer"])].append(row)
    for layer in layers:
        rows = by_layer[layer]
        fixed = [row for row in rows if row["kind"] != "shared_up_down"]
        if [row["kind"] for row in fixed] != ["input_norm", "post_attention_norm", "routed_up_down"]:
            raise RuntimeError(f"PrismaSnap MoE layer {layer} lacks exact core seams")
        shared_rows = [row for row in rows if row["kind"] == "shared_up_down"]
        if [row["shared_index"] for row in shared_rows] != list(
            range(len(shared_rows))
        ):
            raise RuntimeError(
                f"PrismaSnap MoE layer {layer} shared expert order differs"
            )
        graph_hashes = {row["graph_sha256"] for row in rows}
        if len(graph_hashes) != 1 or any(not isinstance(v, str) or len(v) != 64 for v in graph_hashes):
            raise RuntimeError(f"PrismaSnap MoE layer {layer} graph binding differs")
        input_row, post_row, routed = fixed
        expected_vectors = {
            "input": f"layer_{layer:05d}_input",
            "post": f"layer_{layer:05d}_post",
            "routed": f"layer_{layer:05d}_routed_updown",
        }
        if (
            input_row["vector"] != expected_vectors["input"]
            or post_row["vector"] != expected_vectors["post"]
            or routed["vector"] != expected_vectors["routed"]
            or any(
                row["vector"]
                != f"layer_{layer:05d}_shared_{index:03d}_updown"
                for index, row in enumerate(shared_rows)
            )
        ):
            raise RuntimeError(f"PrismaSnap MoE layer {layer} vector binding differs")
        input_vector = scales[str(input_row["vector"])]
        post_vector = scales[str(post_row["vector"])]
        if input_vector.ndim != 1 or post_vector.ndim != 1 or input_vector.numel() != hidden or post_vector.numel() != hidden:
            raise RuntimeError(f"PrismaSnap MoE layer {layer} norm scale shapes differ")
        for row in (input_row, post_row):
            norm = str(row["norm"])
            if norm not in tensors or tuple(tensors[norm]["shape"]) != (hidden,) or tensors[norm]["dtype"] != "BF16":
                raise RuntimeError(f"PrismaSnap MoE layer {layer} norm tensor differs")
            checkpoint._require_tensor_layer(
                norm, layer, where="PrismaSnap MoE norm role"
            )
            _validate_stat(row["stats"], channels=hidden, search=search, where=f"PrismaSnap MoE layer {layer} {row['kind']} stats")
        input_consumers = input_row["consumers"]
        if (
            not isinstance(input_consumers, list)
            or len(input_consumers) < 2
            or any(not isinstance(name, str) for name in input_consumers)
            or len(set(input_consumers)) != len(input_consumers)
        ):
            raise RuntimeError(
                f"PrismaSnap MoE layer {layer} input roles are malformed"
            )
        for name in input_consumers:
            if name not in tensors:
                raise RuntimeError(f"PrismaSnap MoE input role is absent: {name}")
            checkpoint._require_tensor_layer(
                name, layer, where="PrismaSnap MoE input role"
            )
            shape = tuple(tensors[name]["shape"])
            if (
                len(shape) != 2
                or shape[1] != hidden
                or tensors[name]["dtype"] != "BF16"
            ):
                raise RuntimeError(f"PrismaSnap MoE input role shape differs: {name}")
        input_offset = input_row["norm_parameter_offset"]
        post_offset = post_row["norm_parameter_offset"]
        if (
            isinstance(input_offset, bool)
            or not isinstance(input_offset, (int, float))
            or not math.isfinite(float(input_offset))
            or input_offset != post_offset
        ):
            raise RuntimeError(
                f"PrismaSnap MoE layer {layer} norm encodings disagree"
            )
        objective = post_row["objective_consumers"]
        compensation = post_row["compensation_only_consumers"]
        if (
            not isinstance(objective, list)
            or not objective
            or not isinstance(compensation, list)
            or not compensation
            or any(not isinstance(name, str) for name in [*objective, *compensation])
            or set(objective) & set(compensation)
            or len(set([*objective, *compensation])) != len([*objective, *compensation])
        ):
            raise RuntimeError(f"PrismaSnap MoE layer {layer} post roles are malformed")
        experts = model["expert_counts"].get(str(layer))
        if type(experts) is not int or experts <= 0 or routed.get("experts") != experts:
            raise RuntimeError(f"PrismaSnap MoE layer {layer} expert count differs")
        routed_scale = scales[str(routed["vector"])]
        if routed_scale.ndim != 2 or routed_scale.shape[0] != experts:
            raise RuntimeError(f"PrismaSnap MoE layer {layer} routed scale must be [E,I]")
        routed_stats = routed["stats"]
        if not isinstance(routed_stats, list) or len(routed_stats) != experts:
            raise RuntimeError(f"PrismaSnap MoE layer {layer} routed stats differ")
        for expert, stat in enumerate(routed_stats):
            _validate_stat(stat, channels=int(routed_scale.shape[1]), search=search, where=f"PrismaSnap MoE layer {layer} expert {expert} stats")
        if routed["layout"] == "packed_3d":
            if (
                model["routed_layouts"].get(str(layer)) != "packed_3d"
                or routed["expert_axis"] != 0
                or routed["row_axis"] != 1
                or routed["input_axis"] != 2
                or routed["gate_rows"] != [0, int(routed_scale.shape[1])]
                or routed["up_rows"] != [int(routed_scale.shape[1]), 2 * int(routed_scale.shape[1])]
            ):
                raise RuntimeError(f"PrismaSnap MoE layer {layer} packed axes/slices differ")
            packed_names = (str(routed["gate_up"]), str(routed["down"]))
            if any(name not in tensors for name in packed_names):
                raise RuntimeError(f"PrismaSnap MoE layer {layer} packed role is absent")
            if tuple(tensors[packed_names[0]]["shape"]) != (experts, 2 * routed_scale.shape[1], hidden) or tuple(tensors[packed_names[1]]["shape"]) != (experts, hidden, routed_scale.shape[1]):
                raise RuntimeError(f"PrismaSnap MoE layer {layer} packed tensor shapes differ")
            for name in packed_names:
                checkpoint._require_tensor_layer(
                    name, layer, where="PrismaSnap MoE packed role"
                )
                if tensors[name]["dtype"] != "BF16":
                    raise RuntimeError(f"PrismaSnap MoE packed role is not BF16: {name}")
        else:
            if model["routed_layouts"].get(str(layer)) != "per_expert_2d":
                raise RuntimeError(f"PrismaSnap MoE layer {layer} layout stamp differs")
            roles = routed["roles"]
            if not isinstance(roles, list) or len(roles) != experts:
                raise RuntimeError(f"PrismaSnap MoE layer {layer} per-expert role census differs")
            for expert, role in enumerate(roles):
                if not isinstance(role, Mapping):
                    raise RuntimeError("PrismaSnap per-expert role is malformed")
                _exact_keys(role, _PER_EXPERT_ROLE_KEYS, "PrismaSnap per-expert role")
                if role["expert"] != expert:
                    raise RuntimeError("PrismaSnap per-expert role order differs")
                role_names = [str(role[name]) for name in ("gate", "up", "down")]
                if any(name not in tensors for name in role_names):
                    raise RuntimeError("PrismaSnap per-expert role is absent")
                if tuple(tensors[role_names[0]]["shape"]) != (routed_scale.shape[1], hidden) or tuple(tensors[role_names[1]]["shape"]) != (routed_scale.shape[1], hidden) or tuple(tensors[role_names[2]]["shape"]) != (hidden, routed_scale.shape[1]):
                    raise RuntimeError("PrismaSnap per-expert role shapes differ")
                for name in role_names:
                    checkpoint._require_tensor_layer(
                        name, layer, where="PrismaSnap MoE per-expert role"
                    )
                    if tensors[name]["dtype"] != "BF16":
                        raise RuntimeError(
                            f"PrismaSnap MoE per-expert role is not BF16: {name}"
                        )
        for row in shared_rows:
            vector = scales[str(row["vector"])]
            role_names = {
                name: str(row[name])
                for name in ("output_gate", "gate", "up", "down")
            }
            if any(name not in tensors for name in role_names.values()):
                raise RuntimeError("PrismaSnap shared-expert role is absent")
            if vector.ndim != 1 or tuple(tensors[role_names["output_gate"]]["shape"]) != (1, hidden) or tuple(tensors[role_names["gate"]]["shape"]) != (vector.numel(), hidden) or tuple(tensors[role_names["up"]]["shape"]) != (vector.numel(), hidden) or tuple(tensors[role_names["down"]]["shape"]) != (hidden, vector.numel()):
                raise RuntimeError("PrismaSnap shared-expert shapes differ")
            for name in role_names.values():
                checkpoint._require_tensor_layer(
                    name, layer, where="PrismaSnap MoE shared-expert role"
                )
                if tensors[name]["dtype"] != "BF16":
                    raise RuntimeError(
                        f"PrismaSnap shared-expert role is not BF16: {name}"
                    )
            _validate_stat(row["stats"], channels=int(vector.numel()), search=search, where=f"PrismaSnap MoE layer {layer} shared stats")

        router = post_row.get("router")
        if not isinstance(router, str) or router not in tensors:
            raise RuntimeError(f"PrismaSnap MoE layer {layer} router role is absent")
        checkpoint._require_tensor_layer(
            router, layer, where="PrismaSnap MoE router role"
        )
        if (
            tuple(tensors[router]["shape"]) != (experts, hidden)
            or tensors[router]["dtype"] != "BF16"
        ):
            raise RuntimeError(f"PrismaSnap MoE layer {layer} router shape differs")
        if routed["layout"] == "packed_3d":
            expected_objective = [str(routed["gate_up"])]
            packed_gate_up = str(routed["gate_up"])
            packed_down = str(routed["down"])
            per_expert_roles: Sequence[Mapping[str, object]] = []
        else:
            expected_objective = [
                str(role[name])
                for role in routed["roles"]
                for name in ("gate", "up")
            ]
            packed_gate_up = None
            packed_down = None
            per_expert_roles = routed["roles"]
        for row in shared_rows:
            expected_objective.extend((str(row["gate"]), str(row["up"])))
        expected_compensation = [
            router,
            *(str(row["output_gate"]) for row in shared_rows),
        ]
        if objective != expected_objective or compensation != expected_compensation:
            raise RuntimeError(
                f"PrismaSnap MoE layer {layer} post objective/compensation census differs"
            )
        all_weight_roles = [
            *input_consumers,
            router,
            *(
                [packed_gate_up, packed_down]
                if routed["layout"] == "packed_3d"
                else [
                    str(role[name])
                    for role in per_expert_roles
                    for name in ("gate", "up", "down")
                ]
            ),
            *(
                str(row[name])
                for row in shared_rows
                for name in ("output_gate", "gate", "up", "down")
            ),
        ]
        norms = {str(input_row["norm"]), str(post_row["norm"])}
        if (
            len(norms) != 2
            or len(set(all_weight_roles)) != len(all_weight_roles)
            or norms & set(all_weight_roles)
        ):
            raise RuntimeError(f"PrismaSnap MoE layer {layer} roles overlap")
        expected_graph_sha = _moe_plan_graph_sha256(
            layer=layer,
            input_norm=str(input_row["norm"]),
            norm_parameter_offset=float(input_offset),
            input_consumers=[str(name) for name in input_consumers],
            post_norm=str(post_row["norm"]),
            router=router,
            routed_layout=str(routed["layout"]),
            experts=experts,
            intermediate=int(routed_scale.shape[1]),
            packed_gate_up=packed_gate_up,
            packed_down=packed_down,
            per_expert_roles=per_expert_roles,
            shared_roles=shared_rows,
        )
        if graph_hashes != {expected_graph_sha}:
            raise RuntimeError(
                f"PrismaSnap MoE layer {layer} graph digest is not role-bound"
            )
    expected_transforms = _expected_moe_transforms(seams, tensors)
    if plan.get("transforms") != expected_transforms:
        raise RuntimeError("PrismaSnap MoE transform program is not seam-derived")
    verification = plan.get("verification")
    if not isinstance(verification, Mapping):
        raise RuntimeError("PrismaSnap MoE verification is malformed")
    _exact_keys(verification, _VERIFICATION_KEYS, "PrismaSnap MoE verification")
    numeric = ("fp64_invariance_max_abs", "router_logit_max_abs", "route_weight_max_abs", "routed_output_max_abs", "routing_changed", "threshold", "required_bf16_fold_kl_max")
    if any(isinstance(verification.get(key), bool) or not isinstance(verification.get(key), (int, float)) or not math.isfinite(float(verification[key])) for key in numeric):
        raise RuntimeError("PrismaSnap MoE verification numbers are malformed")
    if (
        float(verification["threshold"]) != 1e-10
        or max(float(verification[key]) for key in numeric[:4]) > 1e-10
        or float(verification["routing_changed"]) != 0.0
        or float(verification["required_bf16_fold_kl_max"]) != 5e-4
        or verification["domain"] != "pre_cast_fp64_router_routing_and_expert_algebra"
        or verification["real_moe_fold_kl_evidence"] is not None
    ):
        raise RuntimeError("PrismaSnap MoE research verification gate differs")
    if schema == MOE_PLAN_SET_SCHEMA:
        workers = plan.get("workers")
        if not isinstance(workers, list) or len(workers) < 2:
            raise RuntimeError("PrismaSnap MoE merged workers are malformed")


def validate_moe_materialization_plan(
    plan: Mapping[str, object],
    source,
    scales: Mapping[str, torch.Tensor],
) -> None:
    """Replay MoE semantic and transform shape contracts before any write."""
    validate_moe_plan_semantics(plan, scales)
    model = plan["model"]
    if model["planned_layers"] != list(range(int(model["layer_count"]))):
        raise RuntimeError("PrismaSnap MoE materialization requires full layer coverage")
    # Reuse all content/producer/header/config checks, but avoid the dense-only
    # transform branch in the common helper.
    checkpoint = _checkpoint_module()
    source_meta = plan["source"]
    identity = checkpoint.validate_streamed_model_identity(
        source_meta["identity"], where="PrismaSnap MoE materialization source"
    )
    portable = source_meta.get("portable_identity")
    if (
        not isinstance(portable, Mapping)
        or checkpoint.portable_streamed_model_content_identity(identity) != portable
    ):
        raise RuntimeError(
            "PrismaSnap MoE portable identity does not derive from local identity"
        )
    if identity["checkpoint_weight_map"] != source.weight_map:
        raise RuntimeError("PrismaSnap MoE source index differs from plan")
    tensors = checkpoint._validate_tensor_metadata_contract(plan)
    for tensor_name, owner in source.weight_map.items():
        if owner not in source.available_shards:
            continue
        shape, dtype = source.metadata(tensor_name)
        planned = tensors[tensor_name]
        if tuple(planned["shape"]) != shape or planned["dtype"] != dtype:
            raise RuntimeError(
                "PrismaSnap MoE materialization source header differs from plan: "
                f"{tensor_name}"
            )
    checkpoint._validate_config_semantics(source.root, identity)
    if plan["producer"] != _moe_producer_identity():
        raise RuntimeError("PrismaSnap MoE producer differs between plan/materialize")
    for transform in plan["transforms"]:
        name = str(transform["tensor"])
        if name not in source.weight_map or name not in tensors:
            raise RuntimeError("PrismaSnap MoE transform references unknown tensor")
        if tensors[name]["dtype"] != "BF16":
            raise RuntimeError(f"PrismaSnap MoE transform source is not BF16: {name}")


def apply_moe_materialization_transform(
    tensor: torch.Tensor,
    vector: torch.Tensor,
    transform: Mapping[str, object],
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Execute one already-semantic-validated MoE transform."""
    operation = str(transform["operation"])
    axis = int(transform["axis"])
    if "expert_axis" in transform:
        return apply_packed_expert_slice_transform(
            tensor,
            vector,
            expert_axis=int(transform["expert_axis"]),
            channel_axis=axis,
            expert=None,
            channel_start=int(transform["channel_start"]),
            channel_stop=int(transform["channel_stop"]),
            operation=operation,
            output_dtype=output_dtype,
        )
    selected = vector
    if "vector_index" in transform:
        selected = vector[int(transform["vector_index"])]
    return apply_diagonal_transform(
        tensor,
        selected,
        operation,  # type: ignore[arg-type]
        axis,
        parameter_offset=float(transform.get("parameter_offset", 0.0)),
        output_dtype=output_dtype,
    )


__all__ = [
    "MOE_PLAN_SCHEMA",
    "MOE_PLAN_SET_SCHEMA",
    "MOE_PROVENANCE_SCHEMA",
    "apply_moe_materialization_transform",
    "plan_moe_checkpoint",
    "plan_scale_vector_names",
    "validate_moe_materialization_plan",
    "validate_moe_plan_semantics",
]
