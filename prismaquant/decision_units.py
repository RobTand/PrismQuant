"""Fused-sibling decision-unit helpers shared by production paths.

These types originally lived in the Block-CLADO module.  The production stack
still needs them for measured polish and production-cache sibling grouping even
when cross-layer CLADO machinery is archived.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import format_registry as fr
from .allocator_candidates import check_format_applicability
from .name_projection import strip_weight_leaf
from .serving_profiles import resolve_target_profile


SCHEMA = "prismaquant.block_clado.v1"


@dataclass(frozen=True)
class FormatCost:
    """One format choice for one fused-group decision unit."""

    fmt: str
    omega_ii: float
    bits_per_param: float
    memory_bytes: int

    @property
    def bits_total(self) -> float:
        return float(self.memory_bytes) * 8.0


@dataclass(frozen=True)
class DecisionUnit:
    """A fused-sibling group treated as a single allocator decision."""

    name: str
    block_id: str
    member_qnames: tuple[str, ...]
    options: tuple[FormatCost, ...]


@dataclass(frozen=True)
class BlockPair:
    """A measured intra-block interaction between two decision units."""

    unit_a: str
    unit_b: str
    block_id: str
    omega_ij: dict[tuple[str, str], float]


def block_id_from_qname(qname: str) -> str:
    """Return the architectural block ID a quantizable Linear belongs to."""
    parts = qname.split(".")
    for idx, token in enumerate(parts):
        if token == "layers" and idx + 1 < len(parts):
            try:
                int(parts[idx + 1])
            except ValueError:
                continue
            return ".".join(parts[: idx + 2])
    return qname


def fused_group_key(profile, qname: str) -> str:
    """Profile-aware fused-sibling group key, or the bare qname if none."""
    try:
        if profile is not None:
            group = profile.fused_sibling_group(qname)
            if group:
                return group
    except Exception:
        pass
    return qname


def incomplete_fused_group_members(names: Iterable[str], profile) -> set[str]:
    """Present members of fused-sibling groups that are MISSING a sibling.

    vLLM loads a fused projection (e.g. ``qkv_proj`` = q/k/v, ``gate_up_proj`` =
    gate/up) as a single quantized unit. If a group is incomplete — a sibling is
    absent from the checkpoint, e.g. Gemma4 ``k_eq_v`` full-attention layers
    synthesize ``v = k`` and ship no ``v_proj`` (and vLLM never creates a
    ``v_scale`` param) — the present members CANNOT be quantized: the fused load
    would look up a scale param that doesn't exist (``KeyError`` on
    ``k_proj.weight_scale``). Those present members must ship BF16 instead.

    This is a generic vLLM fused-load invariant, not model-specific: it is
    driven entirely by the profile's ``fused_sibling_group`` +
    ``fused_sibling_leaf_mapping``. Returns the set of qnames to force to BF16.
    """
    if profile is None:
        return set()
    try:
        leaf_map = profile.fused_sibling_leaf_mapping() or {}
    except Exception:
        leaf_map = {}
    if not leaf_map:
        return set()
    groups: dict[str, set[str]] = defaultdict(set)
    for n in names:
        groups[fused_group_key(profile, n)].add(n)
    pinned: set[str] = set()
    for gk, present in groups.items():
        fused_leaf = gk.rsplit(".", 1)[-1]
        members = leaf_map.get(fused_leaf)
        if not members:
            continue  # not a recognized fused group (bare qname, o_proj, ...)
        prefix = gk[: len(gk) - len(fused_leaf)]
        expected = {prefix + m for m in members}
        if expected - present:            # at least one expected sibling absent
            pinned |= present & expected  # → present siblings must ship BF16
    return pinned


def _enumerate_quantizable_linears(model, profile=None) -> list[str]:
    from .build_rtn_cache import iter_quantizable_tensors

    names: list[str] = []
    for full_name, _module, _attr in iter_quantizable_tensors(model, profile):
        names.append(strip_weight_leaf(full_name))
    return sorted(set(names))


def _shape_of_param(model, qname: str, profile=None) -> tuple[int, ...] | None:
    from .build_rtn_cache import iter_quantizable_tensors

    target = qname
    for full_name, module, attr in iter_quantizable_tensors(model, profile):
        if strip_weight_leaf(full_name) == target:
            param = getattr(module, attr, None)
            if param is None:
                return None
            return tuple(int(v) for v in param.shape)
    return None


def _prod(seq: Iterable[int]) -> int:
    out = 1
    for v in seq:
        out *= int(v)
    return out


def discover_units(
    model,
    profile,
    formats: Sequence["fr.FormatSpec"],
    *,
    target_profile: str | None = None,
) -> tuple[dict[str, list[DecisionUnit]], list[DecisionUnit], dict[str, int]]:
    """Build fused-sibling decision units directly from the model.

    This generic API has no artifact-wide CB serialization context, so it
    cannot deduplicate codebooks or choose FP4 layout v1/v2.  Reject CB menus
    rather than silently falling back to ``FormatSpec.memory_bytes_for_shape``
    (the stale legacy-v1 approximation).  CB allocation must use the main
    allocator's context-bound candidate path.
    """
    cb_formats = [
        spec.name
        for spec in formats
        if spec.family in {"nvfp4_cb", "fp8_cb"}
    ]
    if cb_formats:
        raise ValueError(
            "decision_units.discover_units cannot price CB formats without "
            "an artifact-wide CBSerializationContext and shared-sidecar "
            f"accounting: {sorted(set(cb_formats))}. Use prismaquant.allocator."
        )
    graph_units = _discover_units_from_model_graph(
        model,
        profile,
        formats,
        target_profile=target_profile,
    )
    if graph_units is not None:
        return graph_units

    qnames = _enumerate_quantizable_linears(model, profile)
    groups: dict[str, list[str]] = defaultdict(list)
    for qname in qnames:
        groups[fused_group_key(profile, qname)].append(qname)

    blocks: dict[str, list[DecisionUnit]] = defaultdict(list)
    singletons: list[DecisionUnit] = []
    n_params_by_unit: dict[str, int] = {}

    for group_name, members in sorted(groups.items()):
        members = sorted(set(members))
        block_ids = [block_id_from_qname(m) for m in members]
        block_id = max(set(block_ids), key=block_ids.count) if block_ids else group_name
        member_shapes_by_name: dict[str, tuple[int, ...]] = {}
        for member in members:
            shape = _shape_of_param(model, member, profile)
            if shape is not None:
                member_shapes_by_name[member] = shape
        if not member_shapes_by_name:
            continue

        n_params_unit = sum(int(_prod(s)) for s in member_shapes_by_name.values())
        n_params_by_unit[group_name] = n_params_unit
        options = []
        for spec in formats:
            spec_canon = fr.canonical_format_name(spec.name)
            if not all(
                check_format_applicability(
                    shape,
                    spec,
                    qname=member,
                    target_profile=_target_profile_id(profile, target_profile),
                ).legal
                for member, shape in member_shapes_by_name.items()
            ):
                continue
            mem_bytes = sum(
                spec.memory_bytes_for_shape(s)
                for s in member_shapes_by_name.values()
            )
            options.append(FormatCost(
                fmt=spec_canon,
                omega_ii=0.0,
                bits_per_param=8.0 * mem_bytes / max(n_params_unit, 1),
                memory_bytes=int(mem_bytes),
            ))
        if not options:
            continue
        options.sort(key=lambda opt: (opt.bits_per_param, opt.fmt))
        unit = DecisionUnit(
            name=group_name,
            block_id=block_id,
            member_qnames=tuple(members),
            options=tuple(options),
        )
        if block_id == group_name and ".layers." not in block_id:
            singletons.append(unit)
        else:
            blocks[block_id].append(unit)

    pruned_blocks: dict[str, list[DecisionUnit]] = {}
    for block_id, units in blocks.items():
        if len(units) == 1 and ".layers." not in block_id:
            singletons.append(units[0])
        else:
            pruned_blocks[block_id] = units
    return pruned_blocks, singletons, n_params_by_unit


def _discover_units_from_model_graph(
    model,
    profile,
    formats: Sequence["fr.FormatSpec"],
    *,
    target_profile: str | None = None,
) -> tuple[dict[str, list[DecisionUnit]], list[DecisionUnit], dict[str, int]] | None:
    if profile is None or not hasattr(profile, "build_model_graph"):
        return None
    try:
        graph = profile.build_model_graph(model)
    except Exception:
        return None

    shapes_by_name = {
        strip_weight_leaf(tensor.recipe_name): tensor.shape
        for tensor in graph.quantizable_tensors()
    }
    if not shapes_by_name:
        return None

    blocks: dict[str, list[DecisionUnit]] = defaultdict(list)
    singletons: list[DecisionUnit] = []
    n_params_by_unit: dict[str, int] = {}
    serving_profile = _target_profile_id(profile, target_profile)
    for opt_unit in graph.optimization_units():
        unit_name = _decision_unit_name_from_graph_unit(opt_unit.id)
        members = tuple(
            strip_weight_leaf(member)
            for member in opt_unit.members
            if strip_weight_leaf(member) in shapes_by_name
        )
        if not members:
            continue
        member_shapes_by_name = {
            member: shapes_by_name[member]
            for member in members
        }
        n_params_unit = sum(int(_prod(s)) for s in member_shapes_by_name.values())
        n_params_by_unit[unit_name] = n_params_unit
        options = []
        for spec in formats:
            spec_canon = fr.canonical_format_name(spec.name)
            if not all(
                check_format_applicability(
                    shape,
                    spec,
                    qname=member,
                    target_profile=serving_profile,
                ).legal
                for member, shape in member_shapes_by_name.items()
            ):
                continue
            mem_bytes = sum(
                spec.memory_bytes_for_shape(s)
                for s in member_shapes_by_name.values()
            )
            options.append(FormatCost(
                fmt=spec_canon,
                omega_ii=0.0,
                bits_per_param=8.0 * mem_bytes / max(n_params_unit, 1),
                memory_bytes=int(mem_bytes),
            ))
        if not options:
            continue
        options.sort(key=lambda opt: (opt.bits_per_param, opt.fmt))
        block_id = opt_unit.block or block_id_from_qname(members[0])
        unit = DecisionUnit(
            name=unit_name,
            block_id=block_id,
            member_qnames=members,
            options=tuple(options),
        )
        if block_id == unit_name and ".layers." not in block_id:
            singletons.append(unit)
        else:
            blocks[block_id].append(unit)

    if not blocks and not singletons:
        return None
    pruned_blocks: dict[str, list[DecisionUnit]] = {}
    for block_id, units in blocks.items():
        if len(units) == 1 and ".layers." not in block_id:
            singletons.append(units[0])
        else:
            pruned_blocks[block_id] = units
    return pruned_blocks, singletons, n_params_by_unit


def _target_profile_id(profile, target_profile: str | None) -> str:
    return resolve_target_profile(profile, target_profile)


def _decision_unit_name_from_graph_unit(unit_id: str) -> str:
    for prefix in ("fused:", "packed_expert:", "tensor:"):
        if unit_id.startswith(prefix):
            return strip_weight_leaf(unit_id[len(prefix):])
    return strip_weight_leaf(unit_id)


def _parse_pair_key(key: str) -> tuple[str, str]:
    if "__" not in key:
        raise ValueError(f"invalid pair key: {key!r}")
    a, b = key.split("__", 1)
    return fr.canonical_format_name(a), fr.canonical_format_name(b)


def parse_payload(payload: Mapping) -> tuple[
    dict[str, list[DecisionUnit]],
    list[DecisionUnit],
    dict[str, list[BlockPair]],
]:
    """Re-hydrate units, singletons, and intra-block pairs from JSON."""
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported decision-unit schema: {payload.get('schema')!r}")
    blocks: dict[str, list[DecisionUnit]] = {}
    pairs_by_block: dict[str, list[BlockPair]] = {}
    for block_id, block_payload in (payload.get("blocks") or {}).items():
        units: list[DecisionUnit] = []
        for unit_name, unit_payload in (block_payload.get("units") or {}).items():
            options = []
            for fmt, entry in unit_payload["options"].items():
                options.append(FormatCost(
                    fmt=fr.canonical_format_name(fmt),
                    omega_ii=float(entry["omega_ii"]),
                    bits_per_param=float(entry["bits_per_param"]),
                    memory_bytes=int(entry["memory_bytes"]),
                ))
            options.sort(key=lambda opt: (opt.bits_per_param, opt.fmt))
            units.append(DecisionUnit(
                name=str(unit_name),
                block_id=str(block_id),
                member_qnames=tuple(
                    unit_payload.get("members")
                    or unit_payload.get("member_qnames")
                    or [unit_name]
                ),
                options=tuple(options),
            ))
        units.sort(key=lambda unit: unit.name)
        blocks[str(block_id)] = units

        pairs: list[BlockPair] = []
        for pair_payload in block_payload.get("pairs") or []:
            omega = {}
            for key, value in (pair_payload.get("omega_ij") or {}).items():
                fmt_a, fmt_b = _parse_pair_key(str(key))
                omega[(fmt_a, fmt_b)] = float(value)
            pairs.append(BlockPair(
                unit_a=str(pair_payload["unit_a"]),
                unit_b=str(pair_payload["unit_b"]),
                block_id=str(block_id),
                omega_ij=omega,
            ))
        pairs_by_block[str(block_id)] = pairs

    singletons: list[DecisionUnit] = []
    for unit_name, unit_payload in (payload.get("singletons") or {}).items():
        options = []
        for fmt, entry in unit_payload["options"].items():
            options.append(FormatCost(
                fmt=fr.canonical_format_name(fmt),
                omega_ii=float(entry["omega_ii"]),
                bits_per_param=float(entry["bits_per_param"]),
                memory_bytes=int(entry["memory_bytes"]),
            ))
        options.sort(key=lambda opt: (opt.bits_per_param, opt.fmt))
        singletons.append(DecisionUnit(
            name=str(unit_name),
            block_id=str(unit_payload.get("block_id") or unit_name),
            member_qnames=tuple(
                unit_payload.get("members")
                or unit_payload.get("member_qnames")
                or [unit_name]
            ),
            options=tuple(options),
        ))
    singletons.sort(key=lambda unit: unit.name)
    return blocks, singletons, pairs_by_block


def load_payload(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def total_param_count(payload: Mapping) -> int:
    """Sum n_params across all units in a decision-unit payload."""
    total = 0
    for block in (payload.get("blocks") or {}).values():
        for unit in (block.get("units") or {}).values():
            options = unit.get("options") or {}
            ref = options.get("BF16")
            if ref is None:
                ref = next(iter(options.values()))
            mem = int(ref["memory_bytes"])
            bpp = float(ref["bits_per_param"])
            if bpp > 0:
                total += int(round(mem * 8.0 / bpp))
    for unit in (payload.get("singletons") or {}).values():
        options = unit.get("options") or {}
        ref = options.get("BF16") or next(iter(options.values()))
        mem = int(ref["memory_bytes"])
        bpp = float(ref["bits_per_param"])
        if bpp > 0:
            total += int(round(mem * 8.0 / bpp))
    return total
