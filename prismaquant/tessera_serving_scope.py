"""Explicit runtime intake shared by Tessera campaign, allocation and export.

The target describes the requested runtime, not the model. Structure is read
separately for every unit from probe/discovery facts, checked against the model
profile's declarations. Neither a model name nor the calibration device is a
serving target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Mapping

from .lane_eligibility import ServingContext, STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE


@dataclass(frozen=True)
class ServingTarget:
    platform: str
    runtime_image: str
    execution_mode: str
    residency: str

    def __post_init__(self):
        # Reuse the owner's validation; this does not classify any model unit.
        self.context(STRUCTURE_DENSE)

    def context(self, structure: str) -> ServingContext:
        return ServingContext(structure=structure, **self.as_dict())

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def add_serving_scope_arguments(parser) -> None:
    """No defaults: a legacy context-free call remains context-free."""
    parser.add_argument("--tessera-platform", default=None,
                        help="Exact serving platform; otherwise the selected serving profile's target_platform")
    parser.add_argument("--tessera-runtime-image", default=None,
                        help="Exact serving repository@sha256 digest, never a mutable tag")
    parser.add_argument("--tessera-execution-mode", default=None,
                        help="Explicit serving execution mode: eager or compiled")
    parser.add_argument("--tessera-residency", default=None,
                        help="Explicit serving residency: resident or streamed")


def serving_target_from_args(args, *, target_platform: str | None = None) -> ServingTarget | None:
    fields = ("platform", "runtime_image", "execution_mode", "residency")
    values = {field: getattr(args, "tessera_" + field, None) for field in fields}
    if all(value is None for value in values.values()):
        return None
    explicit_platform = values["platform"]
    if explicit_platform is not None and target_platform and explicit_platform != target_platform:
        raise ValueError(
            f"tessera platform conflict: --tessera-platform={explicit_platform!r} "
            f"but the selected serving profile declares {target_platform!r}")
    if explicit_platform is None:
        values["platform"] = target_platform
    for field, value in values.items():
        if value is None:
            raise ValueError(f"tessera serving target requires explicit {field}; "
                             f"supply --tessera-{field.replace('_', '-')}")
    return ServingTarget(**values)


def unit_structure_from_profile(qname: str, profile) -> str:
    """Classify a known checkpoint unit with the profile's declared grammar.

    Export first proves the unit exists in the source headers. This function
    does not make a missing source tensor or unknown profile into a dense one.
    """
    if profile is None or profile.structure_spec() is None:
        raise ValueError(f"{qname}: explicit Tessera scope needs a declared model profile")
    if profile.packed_expert_format_group(qname) is not None:
        return STRUCTURE_ROUTED_MOE
    for rule in (profile.per_expert_moe_regex(), profile.per_expert_mtp_regex()):
        if rule and re.fullmatch(rule.removeprefix("re:"), profile.to_vllm_internal_name(qname)):
            return STRUCTURE_ROUTED_MOE
    return STRUCTURE_DENSE


def unit_structure_from_stats(qname: str, row: Mapping, profile) -> str:
    """Use owned probe facts, never tensor shape or a model-wide MoE guess."""
    declared_structure = unit_structure_from_profile(qname, profile)
    packed_module = row.get("_packed_experts_module")
    count = row.get("num_experts")
    if packed_module is not None or count is not None:
        if not isinstance(packed_module, str) or not packed_module or isinstance(count, bool) \
                or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{qname}: ambiguous packed expert topology; "
                             "need _packed_experts_module and positive num_experts")
        structure = STRUCTURE_ROUTED_MOE
    else:
        if "router_path" not in row or "expert_id" not in row:
            raise ValueError(f"{qname}: missing per-unit router_path/expert_id topology")
        router, expert = row["router_path"], row["expert_id"]
        if router is None and expert is None:
            structure = STRUCTURE_DENSE
        elif isinstance(router, str) and router and expert is not None \
                and not isinstance(expert, bool) and str(expert).isdigit():
            structure = STRUCTURE_ROUTED_MOE
        else:
            raise ValueError(f"{qname}: conflicting router_path/expert_id topology")
    if declared_structure == STRUCTURE_ROUTED_MOE and structure != STRUCTURE_ROUTED_MOE:
        raise ValueError(f"{qname}: probe topology conflicts with the profile's routed expert declaration")
    return structure


def context_by_unit_from_stats(target: ServingTarget | None, stats: Mapping[str, Mapping], profile
                               ) -> dict[str, ServingContext] | None:
    if target is None:
        return None
    return {name: target.context(unit_structure_from_stats(name, row, profile))
            for name, row in stats.items()}


def scope_provenance(target: ServingTarget | None,
                     contexts: Mapping[str, ServingContext] | None) -> dict:
    if target is None:
        return {}
    return {"target": target.as_dict(),
            "by_unit": {name: context.as_dict() for name, context in sorted((contexts or {}).items())}}
