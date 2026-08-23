"""Declarative model-structure specs and graph extraction.

This module is the bridge between architecture-specific profile knowledge and
the format/cache/allocator core.  It intentionally does not replace
``ModelProfile`` yet; instead it records the name/decomposition contract in a
portable shape and can build a typed graph from an already-loaded model.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import torch.nn as nn


SCHEMA = "prismaquant.model_structure.v1"

# ---------------------------------------------------------------------------
# Detection vocabulary
# ---------------------------------------------------------------------------
#
# `match` is the declarative form of `ModelProfile.matches()`.  It is
# deliberately tiny: the nine in-tree profiles are predicates over
# `(model_type in set, architecture glob)`, and the two non-prefix cases
# (`qwen3.py`'s exact architectures and `qwen3_5_dense.py`'s "not a Moe" veto)
# need exactly the glob and the negative list below.  Anything richer belongs
# in Python, not in JSON.
MATCH_KEYS = frozenset({
    "model_type",             # exact strings
    "architectures",          # fnmatch globs (an exact name is a valid glob)
    "architectures_exclude",  # fnmatch globs; any hit vetoes the whole match
})

# `priority` orders detection: LOWER is consulted first, like a sort rank.
# Built-in profiles declare 100..199 in `registry.py`'s historical list order;
# `ModelProfile.priority` defaults to 0 so a third-party `register_profile()`
# still wins outright (the documented insert-at-front contract).  A spec that
# declares no priority resolves after every Python profile.
SPEC_DEFAULT_PRIORITY = 1000

# ---------------------------------------------------------------------------
# Export lanes (EXPORT_CONTAINER vocabulary)
# ---------------------------------------------------------------------------
EXPORT_LANES = ("compressed-tensors", "nvfp4_cb", "gguf")
DEFAULT_EXPORT_LANE = "compressed-tensors"
# The serving-profile side spells the native lane with an underscore
# (`serving_profile_specs/vllm_packed_moe.json` -> export_lane.id
# "compressed_tensors"); EXPORT_CONTAINER uses the hyphen. One alias, declared.
_EXPORT_LANE_ALIASES = {"compressed_tensors": "compressed-tensors"}


def canonical_export_lane(name: str) -> str:
    """Canonical EXPORT_CONTAINER spelling for a lane id.

    Raises on an unknown lane: a typo in a spec must fail at parse time, not
    silently declare an architecture eligible for a lane that does not exist.
    """
    lane = _EXPORT_LANE_ALIASES.get(str(name), str(name))
    if lane not in EXPORT_LANES:
        raise ValueError(
            f"unknown export lane {name!r}; known lanes: {list(EXPORT_LANES)}"
        )
    return lane


@dataclass(frozen=True)
class SpecMatch:
    """Declarative detection predicate — the spec form of ``matches()``.

    Verdict: an ``architectures_exclude`` hit vetoes unconditionally; otherwise
    the profile claims the config when ``model_type`` matches exactly OR any
    declared architecture glob matches any declared architecture.
    """

    model_type: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()
    architectures_exclude: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "SpecMatch":
        payload = payload or {}
        unknown = sorted(set(payload) - MATCH_KEYS)
        if unknown:
            raise ValueError(
                f"unsupported model-structure match keys: {unknown}; "
                f"vocabulary is {sorted(MATCH_KEYS)}"
            )
        return cls(
            model_type=tuple(str(v) for v in payload.get("model_type", ())),
            architectures=tuple(str(v) for v in payload.get("architectures", ())),
            architectures_exclude=tuple(
                str(v) for v in payload.get("architectures_exclude", ())
            ),
        )

    @property
    def declared(self) -> bool:
        return bool(self.model_type or self.architectures)

    def claims(
        self,
        model_type: str | None,
        architectures: Iterable[str] | None,
    ) -> bool:
        archs = [str(a) for a in (architectures or ())]
        for pattern in self.architectures_exclude:
            if any(fnmatchcase(arch, pattern) for arch in archs):
                return False
        if model_type and str(model_type) in self.model_type:
            return True
        for pattern in self.architectures:
            if any(fnmatchcase(arch, pattern) for arch in archs):
                return True
        return False


@dataclass(frozen=True)
class NameRewriteRule:
    """One declarative name rewrite.

    Supported forms:
      - exact replacement: ``{"exact": "lm_head", "replace": "head"}``
      - prefix replacement: ``{"prefix": "model.", "replace": ""}``
      - strip/add prefix: ``{"strip_prefix": "a.", "add_prefix": "b."}``
      - regex substitution: ``{"regex": "...", "replace": "..."}``

    Rules are applied in file order.  ``stop`` ends the rule chain after a
    match, which is useful for top-level names that should not flow through
    later generic regexes.
    """

    exact: str | None = None
    prefix: str | None = None
    strip_prefix: str | None = None
    regex: str | None = None
    replace: str = ""
    add_prefix: str = ""
    stop: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NameRewriteRule":
        return cls(
            exact=_optional_str(payload.get("exact")),
            prefix=_optional_str(payload.get("prefix")),
            strip_prefix=_optional_str(payload.get("strip_prefix")),
            regex=_optional_str(payload.get("regex")),
            replace=str(payload.get("replace", "")),
            add_prefix=str(payload.get("add_prefix", "")),
            stop=bool(payload.get("stop", False)),
        )

    def apply(self, name: str) -> tuple[str, bool]:
        if self.exact is not None:
            if name != self.exact:
                return name, False
            return self.replace, True
        if self.prefix is not None:
            if not name.startswith(self.prefix):
                return name, False
            return self.replace + name[len(self.prefix):], True
        if self.strip_prefix is not None:
            if not name.startswith(self.strip_prefix):
                return name, False
            return self.add_prefix + name[len(self.strip_prefix):], True
        if self.regex is not None:
            new_name, n_subs = re.subn(self.regex, self.replace, name)
            return new_name, n_subs > 0
        return name, False


@dataclass(frozen=True)
class FusedGroupSpec:
    target_suffix: str
    member_suffixes: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FusedGroupSpec":
        members = payload.get("members") or ()
        return cls(
            target_suffix=str(payload["target"]),
            member_suffixes=tuple(str(member) for member in members),
        )


@dataclass(frozen=True)
class PackedExpertSpec:
    param_names: tuple[str, ...] = ()
    module_class_names: tuple[str, ...] = ()
    split_for_formats: tuple[str, ...] = ()
    projection_splits: tuple[tuple[str, tuple[str, ...]], ...] = ()
    format_groups: tuple[tuple[str, ...], ...] = ()
    declared: bool = False

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        declared: bool = False,
    ) -> "PackedExpertSpec":
        return cls(
            param_names=tuple(str(v) for v in payload.get("param_names", ())),
            module_class_names=tuple(
                str(v) for v in payload.get("module_class_names", ())
            ),
            split_for_formats=tuple(
                str(v) for v in payload.get("split_for_formats", ())
            ),
            projection_splits=tuple(
                (str(param_name), tuple(str(v) for v in projections))
                for param_name, projections in (
                    payload.get("projection_splits") or {}
                ).items()
            ),
            format_groups=tuple(
                tuple(str(member) for member in group)
                for group in payload.get("format_groups", ())
            ),
            declared=declared,
        )


@dataclass(frozen=True)
class PackageRequirement:
    module: str
    package: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PackageRequirement":
        return cls(
            module=str(payload["module"]),
            package=str(payload.get("package") or payload["module"]),
        )


@dataclass(frozen=True)
class NamingVariant:
    """A naming block that applies only to some of a family's architectures.

    Several families ship a multimodal wrapper class and a text-only carve-out
    that share every structural rule *except* the namespace their serving
    runtime uses.  Qwen3.5/3.6 is the canonical case: the wrapper builds the
    body under `language_model.model.` and the head at `language_model.lm_head`,
    while `Qwen3_5ForCausalLM` builds `model.` and a bare `lm_head` (its
    `hf_to_vllm_mapper` strips `model.language_model.` instead of adding it).
    One `naming` block cannot be right for both, and picking the wrong one
    emits `config_groups` targets that match no module: the unit loads
    unquantized and its orphaned scale kills the load.

    Only the keys a variant declares are overridden; the rest inherit the base
    `naming` block, so a variant states the difference and nothing else.
    """

    when: SpecMatch = field(default_factory=SpecMatch)
    live_to_recipe: tuple[NameRewriteRule, ...] | None = None
    recipe_to_source: tuple[NameRewriteRule, ...] | None = None
    recipe_to_vllm: tuple[NameRewriteRule, ...] | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NamingVariant":
        unknown = sorted(set(payload) - {"when", "naming"})
        if unknown:
            raise ValueError(
                f"unsupported naming_variants keys: {unknown}; "
                "vocabulary is ['when', 'naming']"
            )
        when = SpecMatch.from_dict(payload.get("when"))
        if not when.declared:
            raise ValueError(
                "a naming variant must declare `when.model_type` or "
                "`when.architectures`; an unconditional variant is just `naming`"
            )
        naming = payload.get("naming") or {}
        unknown_naming = sorted(
            set(naming) - {"live_to_recipe", "recipe_to_source", "recipe_to_vllm"}
        )
        if unknown_naming:
            raise ValueError(
                f"unsupported naming keys in a variant: {unknown_naming}"
            )
        if not naming:
            raise ValueError("a naming variant must declare at least one map")
        return cls(
            when=when,
            live_to_recipe=(
                _rules(naming["live_to_recipe"])
                if "live_to_recipe" in naming else None
            ),
            recipe_to_source=(
                _rules(naming["recipe_to_source"])
                if "recipe_to_source" in naming else None
            ),
            recipe_to_vllm=(
                _rules(naming["recipe_to_vllm"])
                if "recipe_to_vllm" in naming else None
            ),
        )


@dataclass(frozen=True)
class ModelStructureSpec:
    """Declarative model decomposition contract for one architecture family."""

    id: str
    schema: str = SCHEMA
    match: SpecMatch = field(default_factory=SpecMatch)
    priority: int = SPEC_DEFAULT_PRIORITY
    supported_lanes: tuple[str, ...] = ()
    preferred_lane: str | None = None
    live_to_recipe: tuple[NameRewriteRule, ...] = ()
    recipe_to_source: tuple[NameRewriteRule, ...] = ()
    recipe_to_vllm: tuple[NameRewriteRule, ...] = ()
    naming_variants: tuple[NamingVariant, ...] = ()
    fused_groups: tuple[FusedGroupSpec, ...] = ()
    packed_experts: PackedExpertSpec = field(default_factory=PackedExpertSpec)
    unpacked_expert_projection_names: tuple[str, ...] = ()
    per_expert_moe_regex: str | None = None
    per_expert_mtp_regex: str | None = None
    default_serving_profile: str | None = None
    bypass_hf_fp8_module_rewrite: bool = False
    fast_kernel_requirements: tuple[PackageRequirement, ...] = ()
    probe_skip_module_class_names: tuple[str, ...] = ()
    probe_grouped_module_class_names: tuple[str, ...] = ()
    passthrough_prefixes: tuple[str, ...] = ()
    pinned_names: tuple[str, ...] = ("lm_head",)
    stage_text_only_strip_keys: tuple[str, ...] | None = None
    stage_text_only_promote_inner_model_type: bool | None = None
    body_layer_prefix: str | None = None
    mtp_layer_prefix: str | None = None
    mtp_source_prefix: str | None = None
    mtp_extra_linear_names: tuple[str, ...] = ()
    visual_layer_prefix: str | None = None
    visual_config_key: str | None = None
    lm_head_name: str | None = None
    embedding_name: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelStructureSpec":
        schema = str(payload.get("schema", SCHEMA))
        if schema != SCHEMA:
            raise ValueError(f"unsupported model-structure schema: {schema!r}")
        naming = payload.get("naming") or {}
        moe = payload.get("moe") or {}
        packed_payload = payload.get("packed_experts")
        runtime = payload.get("runtime_requirements") or {}
        probe = payload.get("probe") or {}
        staging = payload.get("staging") or {}
        shard_regexes = payload.get("shard_regexes") or {}
        supported_lanes = tuple(
            canonical_export_lane(v) for v in payload.get("supported_lanes", ())
        )
        preferred_lane = _optional_str(payload.get("preferred_lane"))
        if preferred_lane is not None:
            preferred_lane = canonical_export_lane(preferred_lane)
            if supported_lanes and preferred_lane not in supported_lanes:
                raise ValueError(
                    f"{payload['id']}: preferred_lane {preferred_lane!r} is not "
                    f"in supported_lanes {list(supported_lanes)}"
                )
        return cls(
            id=str(payload["id"]),
            schema=schema,
            match=SpecMatch.from_dict(payload.get("match")),
            priority=int(payload.get("priority", SPEC_DEFAULT_PRIORITY)),
            supported_lanes=supported_lanes,
            preferred_lane=preferred_lane,
            live_to_recipe=_rules(naming.get("live_to_recipe")),
            recipe_to_source=_rules(naming.get("recipe_to_source")),
            recipe_to_vllm=_rules(naming.get("recipe_to_vllm")),
            naming_variants=tuple(
                NamingVariant.from_dict(entry)
                for entry in payload.get("naming_variants", ())
            ),
            fused_groups=tuple(
                FusedGroupSpec.from_dict(entry)
                for entry in payload.get("fused_groups", ())
            ),
            packed_experts=PackedExpertSpec.from_dict(
                packed_payload or {},
                declared=packed_payload is not None,
            ),
            unpacked_expert_projection_names=tuple(
                str(v) for v in moe.get("unpacked_expert_projection_names", ())
            ),
            per_expert_moe_regex=_optional_str(moe.get("per_expert_regex")),
            per_expert_mtp_regex=_optional_str(moe.get("per_expert_mtp_regex")),
            default_serving_profile=_optional_str(payload.get("default_serving_profile")),
            bypass_hf_fp8_module_rewrite=bool(
                staging.get("bypass_hf_fp8_module_rewrite", False)
            ),
            fast_kernel_requirements=tuple(
                PackageRequirement.from_dict(entry)
                for entry in runtime.get("fast_kernel_packages", ())
            ),
            probe_skip_module_class_names=tuple(
                str(v) for v in probe.get("skip_module_class_names", ())
            ),
            probe_grouped_module_class_names=tuple(
                str(v) for v in probe.get("grouped_module_class_names", ())
            ),
            passthrough_prefixes=tuple(
                str(v) for v in payload.get("passthrough_prefixes", ())
            ),
            pinned_names=tuple(str(v) for v in payload.get("pinned_names", ("lm_head",))),
            stage_text_only_strip_keys=_optional_str_tuple(
                staging.get("text_only_strip_keys")
                if "text_only_strip_keys" in staging else None
            ),
            stage_text_only_promote_inner_model_type=(
                bool(staging["promote_inner_model_type"])
                if "promote_inner_model_type" in staging else None
            ),
            body_layer_prefix=_optional_str(shard_regexes.get("body_layer_prefix")),
            mtp_layer_prefix=_optional_str(shard_regexes.get("mtp_layer_prefix")),
            mtp_source_prefix=_optional_str(shard_regexes.get("mtp_source_prefix")),
            mtp_extra_linear_names=tuple(
                str(v) for v in shard_regexes.get("mtp_extra_linear_names", ())
            ),
            visual_layer_prefix=_optional_str(shard_regexes.get("visual_layer_prefix")),
            visual_config_key=_optional_str(shard_regexes.get("visual_config_key")),
            lm_head_name=_optional_str(shard_regexes.get("lm_head_name")),
            embedding_name=_optional_str(shard_regexes.get("embedding_name")),
        )

    def for_config(
        self,
        model_type: str | None,
        architectures: Iterable[str] | None,
    ) -> "ModelStructureSpec":
        """Return the spec specialized to one checkpoint's declared config.

        Variants are consulted in declaration order; the first whose `when`
        claims the config wins.  With no variants — every spec but
        `qwen3_5_dense` today — this is `self`, so the specialization is a
        no-op everywhere it is not explicitly declared.
        """
        if not self.naming_variants:
            return self
        for variant in self.naming_variants:
            if not variant.when.claims(model_type, architectures):
                continue
            overrides: dict[str, Any] = {}
            for attr in ("live_to_recipe", "recipe_to_source", "recipe_to_vllm"):
                rules = getattr(variant, attr)
                if rules is not None:
                    overrides[attr] = rules
            return replace(self, naming_variants=(), **overrides)
        return self

    def rewrite_live_to_recipe(self, name: str) -> str:
        return apply_name_rewrites(name, self.live_to_recipe)

    def rewrite_recipe_to_source(self, name: str) -> str:
        return apply_name_rewrites(name, self.recipe_to_source)

    def rewrite_recipe_to_vllm(self, name: str) -> str:
        return apply_name_rewrites(name, self.recipe_to_vllm)

    def fused_group_for(self, linear_qname: str) -> str | None:
        for group in self.fused_groups:
            for member in group.member_suffixes:
                if linear_qname == member:
                    return group.target_suffix
                if linear_qname.endswith("." + member):
                    return (
                        linear_qname[: -len(member)]
                        + group.target_suffix
                    )
        return None

    def split_packed_experts_for_format(self, fmt: str) -> bool | None:
        rules = self.packed_experts.split_for_formats
        if not rules:
            return None
        fmt_upper = str(fmt).upper()
        return any(rule == "*" or rule.upper() == fmt_upper for rule in rules)

    def packed_expert_projection_names(self, param_name: str) -> tuple[str, ...]:
        for packed_name, projections in self.packed_experts.projection_splits:
            if packed_name == param_name:
                return projections
        return (str(param_name),)

    def packed_expert_parent_for_projection(self, projection_name: str) -> str | None:
        for packed_name, projections in self.packed_experts.projection_splits:
            if projection_name in projections:
                return packed_name
        packed_names = set(self.packed_experts.param_names)
        if projection_name in packed_names:
            return projection_name
        return None

    def packed_expert_format_group(self, qname: str) -> str | None:
        """Return the serving-format group key for a packed expert tensor.

        ``format_groups`` describes projection names that the serving runtime
        must load under one quantization scheme.  The matcher handles both the
        packed recipe form (``...experts.gate_up_proj``) and the split
        per-expert export form (``...experts.7.gate_proj``).
        """
        groups = self.packed_experts.format_groups
        if not groups:
            return None
        parts = str(qname).split(".")
        try:
            experts_idx = len(parts) - 1 - list(reversed(parts)).index("experts")
        except ValueError:
            return None
        tail = parts[experts_idx + 1:]
        split_per_expert = False
        if len(tail) == 1:
            leaf = tail[0]
            parent = ".".join(parts[:experts_idx + 1])
        elif len(tail) == 2 and tail[0].isdigit():
            leaf = tail[1]
            parent = ".".join(parts[:experts_idx + 1])
            split_per_expert = True
        else:
            return None
        for group in groups:
            if leaf in group:
                if not self._packed_expert_group_matches_representation(
                    group,
                    split_per_expert=split_per_expert,
                ):
                    continue
                return f"{parent}::__packed_format__:{','.join(group)}"
        return None

    def _packed_expert_group_matches_representation(
        self,
        group: tuple[str, ...],
        *,
        split_per_expert: bool,
    ) -> bool:
        packed_names = set(self.packed_experts.param_names)
        if split_per_expert:
            return not any(
                member in packed_names
                and self.packed_expert_projection_names(member) != (member,)
                for member in group
            )
        return not any(
            member not in packed_names
            and self.packed_expert_parent_for_projection(member) is not None
            for member in group
        )


@dataclass(frozen=True)
class ModelTensor:
    """One named tensor as seen by the model/pipeline/export boundary."""

    role: str
    live_name: str
    recipe_name: str
    source_name: str
    vllm_name: str
    export_name: str
    shape: tuple[int, ...]
    block: str | None = None
    group: str | None = None
    quantizable: bool = False
    pinned: bool = False
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptimizationUnit:
    """An independently optimizable unit in the model graph.

    A unit can be a single Linear, a fused-sibling group that must share a
    serving format, or a packed expert group whose projections should be
    optimized together.
    """

    id: str
    scope: str
    members: tuple[str, ...]
    constraints: tuple[str, ...] = ()
    block: str | None = None


@dataclass(frozen=True)
class ModelGraph:
    """Resolved model structure consumed by pipeline stages."""

    profile_name: str
    tensors: tuple[ModelTensor, ...]
    spec_id: str | None = None

    def __post_init__(self) -> None:
        _assert_unique("recipe_name", (t.recipe_name for t in self.tensors))
        _assert_unique("live_name", (t.live_name for t in self.tensors))

    def quantizable_tensors(self) -> tuple[ModelTensor, ...]:
        return tuple(t for t in self.tensors if t.quantizable and not t.pinned)

    def by_recipe_name(self) -> dict[str, ModelTensor]:
        return {t.recipe_name: t for t in self.tensors}

    def by_live_name(self) -> dict[str, ModelTensor]:
        return {t.live_name: t for t in self.tensors}

    def optimization_units(
        self,
        *,
        include_pinned: bool = False,
    ) -> tuple[OptimizationUnit, ...]:
        grouped: dict[str, list[ModelTensor]] = {}
        scopes: dict[str, str] = {}
        for tensor in self.tensors:
            if not tensor.quantizable:
                if not include_pinned:
                    continue
                if not tensor.pinned:
                    continue
            unit_id, scope = _optimization_unit_key(tensor)
            grouped.setdefault(unit_id, []).append(tensor)
            scopes[unit_id] = scope

        units: list[OptimizationUnit] = []
        for unit_id, tensors in grouped.items():
            constraints = tuple(sorted({
                constraint
                for tensor in tensors
                for constraint in tensor.constraints
                if constraint != "pinned" or include_pinned
            }))
            blocks = {tensor.block for tensor in tensors if tensor.block is not None}
            units.append(OptimizationUnit(
                id=unit_id,
                scope=scopes[unit_id],
                members=_ordered_unit_members(scopes[unit_id], tensors),
                constraints=constraints,
                block=next(iter(blocks)) if len(blocks) == 1 else None,
            ))
        units.sort(key=lambda unit: unit.id)
        return tuple(units)


def load_structure_spec(profile_name: str) -> ModelStructureSpec | None:
    """Load ``model_profiles/specs/<profile_name>.json`` if it exists."""

    resource = (
        Path(__file__).resolve().parent
        / "specs"
        / f"{profile_name}.json"
    )
    try:
        text = resource.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return ModelStructureSpec.from_dict(json.loads(text))


def structure_spec_ids() -> tuple[str, ...]:
    """Every ``specs/<id>.json`` shipped with the package, sorted by id."""

    specs_dir = Path(__file__).resolve().parent / "specs"
    out: list[str] = []
    for entry in specs_dir.iterdir():
        name = entry.name
        if name.endswith(".json") and not name.startswith("_"):
            out.append(name[: -len(".json")])
    return tuple(sorted(out))


def iter_structure_specs() -> tuple[ModelStructureSpec, ...]:
    """Load every shipped structure spec, sorted by id."""

    specs = []
    for spec_id in structure_spec_ids():
        spec = load_structure_spec(spec_id)
        if spec is not None:
            specs.append(spec)
    return tuple(specs)


def apply_name_rewrites(name: str, rules: Iterable[NameRewriteRule]) -> str:
    out = str(name)
    for rule in rules:
        out_next, matched = rule.apply(out)
        out = out_next
        if matched and rule.stop:
            break
    return out


def build_model_graph(
    model,
    profile=None,
    *,
    spec: ModelStructureSpec | None = None,
) -> ModelGraph:
    """Build a graph from a live transformers model.

    The graph uses existing ``ModelProfile`` methods as the source of truth so
    this can land without changing production behavior.  Specs are attached for
    provenance and can be used by tests/tools to compare declarative and
    executable profile behavior.
    """

    if profile is None:
        from .registry import profile_from_model

        profile = profile_from_model(model)
    if spec is None:
        spec = load_structure_spec(profile.name)

    packed_names = set(profile.packed_expert_param_names())
    pinned_names = set(profile.pinned_names())
    modules_by_qname = dict(model.named_modules())
    tensors: list[ModelTensor] = []
    for full_name, param in model.named_parameters():
        shape = tuple(int(dim) for dim in getattr(param, "shape", ()))
        qname, attr = _split_param_name(full_name)
        owner = modules_by_qname.get(qname)
        recipe_qname = profile.live_to_recipe_name(qname)
        live_param_name = full_name
        recipe_param_name = f"{recipe_qname}.{attr}" if attr else recipe_qname
        source_name = profile.source_tensor_name(live_param_name)
        export_name = profile.export_tensor_name(recipe_param_name)
        vllm_name = profile.to_vllm_internal_name(recipe_qname)
        group = profile.fused_sibling_group(recipe_qname)
        role = _infer_role(attr, shape, qname, owner, packed_names)
        pinned = _is_pinned(recipe_qname, recipe_param_name, pinned_names)
        quantizable = role in {"linear_weight", "packed_expert_weight"} and not pinned
        constraints = _constraints_for(role, group, pinned)
        if attr and not vllm_name.endswith(f".{attr}"):
            vllm_param_name = f"{vllm_name}.{attr}"
        else:
            vllm_param_name = vllm_name
        tensors.append(
            ModelTensor(
                role=role,
                live_name=live_param_name,
                recipe_name=recipe_param_name,
                source_name=source_name,
                vllm_name=vllm_param_name,
                export_name=export_name,
                shape=shape,
                block=_block_id(recipe_qname),
                group=group,
                quantizable=quantizable,
                pinned=pinned,
                constraints=constraints,
            )
        )

    tensors.sort(key=lambda t: t.recipe_name)
    return ModelGraph(
        profile_name=str(profile.name),
        spec_id=spec.id if spec is not None else None,
        tensors=tuple(tensors),
    )


def _rules(payload: object) -> tuple[NameRewriteRule, ...]:
    if not payload:
        return ()
    if not isinstance(payload, list):
        raise ValueError("name rewrite rules must be a list")
    return tuple(NameRewriteRule.from_dict(entry) for entry in payload)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_str_tuple(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(str(v) for v in value)


def _assert_unique(label: str, values: Iterable[str]) -> None:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    if dupes:
        sample = ", ".join(sorted(dupes)[:5])
        raise ValueError(f"model graph has duplicate {label}: {sample}")


def _split_param_name(full_name: str) -> tuple[str, str]:
    if "." not in full_name:
        return full_name, ""
    qname, attr = full_name.rsplit(".", 1)
    return qname, attr


def _infer_role(
    attr: str,
    shape: tuple[int, ...],
    qname: str,
    owner: nn.Module | None,
    packed_names: set[str],
) -> str:
    leaf = qname.rsplit(".", 1)[-1]
    if attr in packed_names and len(shape) == 3:
        return "packed_expert_weight"
    if isinstance(owner, nn.Linear) and attr == "weight" and len(shape) == 2:
        return "linear_weight"
    if isinstance(owner, nn.Embedding) and attr == "weight":
        return "embedding_weight"
    if leaf in packed_names and len(shape) == 3:
        return "packed_expert_weight"
    if attr == "bias":
        return "bias"
    return "parameter"


def _is_pinned(qname: str, full_name: str, pinned_names: set[str]) -> bool:
    candidates = {qname, full_name}
    return any(name in candidates or qname.endswith("." + name) for name in pinned_names)


def _constraints_for(role: str, group: str | None, pinned: bool) -> tuple[str, ...]:
    out: list[str] = []
    if pinned:
        out.append("pinned")
    if group is not None:
        out.append("fused_sibling_format")
    if role == "packed_expert_weight":
        out.append("packed_expert_decomposition")
    return tuple(out)


def _block_id(qname: str) -> str | None:
    parts = qname.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers" and idx + 1 < len(parts):
            try:
                int(parts[idx + 1])
            except ValueError:
                continue
            return ".".join(parts[:idx + 2])
    return None


def _optimization_unit_key(tensor: ModelTensor) -> tuple[str, str]:
    if tensor.group is not None:
        return f"fused:{tensor.group}", "fused_sibling_group"
    if "packed_expert_decomposition" in tensor.constraints:
        parent = tensor.recipe_name.rsplit(".", 1)[0]
        return f"packed_expert:{parent}", "packed_expert_group"
    return f"tensor:{tensor.recipe_name}", "tensor"


def _ordered_unit_members(
    scope: str,
    tensors: list[ModelTensor],
) -> tuple[str, ...]:
    if scope != "packed_expert_group":
        return tuple(sorted(t.recipe_name for t in tensors))
    priority = {
        "gate_up_proj": 0,
        "gate_proj": 0,
        "w1": 0,
        "up_proj": 1,
        "w3": 1,
        "down_proj": 2,
        "w2": 2,
    }

    def key(tensor: ModelTensor) -> tuple[int, str]:
        leaf = tensor.recipe_name.rsplit(".", 1)[-1]
        return priority.get(leaf, 99), tensor.recipe_name

    return tuple(t.recipe_name for t in sorted(tensors, key=key))
