"""Small, stdlib-only readers for finalized Gridbook assignment declarations."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


MXFP4_MARLIN_WIRE_ID = "mxfp4_e2m1_ue8m0_g32"
_CB_WIRE_ID = re.compile(r"^(?:NVFP4_CB|FP8_CB)_K[0-9]+$")
_DELEGATED_NATIVE_WIRE_IDS = {
    MXFP4_MARLIN_WIRE_ID,
    "fp8_e4m3_ue8m0_block128",
    "mxfp8_e4m3_e8m0_g32",
}


def artifact_requires_moe_backend_marlin(
    quant_config: Mapping[str, Any],
) -> bool:
    """Read only the live serving assignment, never menus or provenance.

    DSv4Flash can route MXFP4 through either the top-level delegated-native
    unit map or the per-expert format partition.  A string elsewhere in the
    config is merely metadata and must not select a serving backend.
    """
    delegated = quant_config.get("source_passthrough")
    delegated_wires: list[str] = []
    if delegated is not None:
        if not isinstance(delegated, Mapping) or set(delegated) != {
            "version", "units"
        } or delegated.get("version") != 1:
            raise ValueError("source_passthrough is not the closed version-1 declaration")
        units = delegated.get("units")
        if not isinstance(units, Mapping) or not units:
            raise ValueError("source_passthrough carries no live units")
        for unit, wire in units.items():
            if (
                not isinstance(unit, str)
                or not unit
                or unit != unit.strip()
                or not isinstance(wire, str)
                or wire not in _DELEGATED_NATIVE_WIRE_IDS
            ):
                raise ValueError("source_passthrough contains a malformed live route")
            delegated_wires.append(wire)

    per_expert = quant_config.get("per_expert_format_groups")
    expert_wires: list[str] = []
    if per_expert is not None:
        if not isinstance(per_expert, Mapping) or set(per_expert) != {
            "version", "layers"
        } or per_expert.get("version") != 1:
            raise ValueError(
                "per_expert_format_groups is not the closed version-1 declaration"
            )
        layers = per_expert.get("layers")
        if not isinstance(layers, Mapping) or not layers:
            raise ValueError("per_expert_format_groups carries no live layers")
        for layer, families in layers.items():
            if (
                not isinstance(layer, str)
                or not layer.isdigit()
                or not isinstance(families, Mapping)
                or set(families) != {"w13", "w2"}
            ):
                raise ValueError("per-expert layer/family declaration is malformed")
            for family in ("w13", "w2"):
                entries = families.get(family)
                if not isinstance(entries, list) or not entries:
                    raise ValueError("per-expert family carries no live format groups")
                seen_experts: set[int] = set()
                for entry in entries:
                    if not isinstance(entry, Mapping) or set(entry) != {
                        "format_wire_id", "expert_ids", "tensor_prefix"
                    }:
                        raise ValueError("per-expert format group is malformed")
                    wire = entry.get("format_wire_id")
                    expert_ids = entry.get("expert_ids")
                    prefix = entry.get("tensor_prefix")
                    if (
                        not isinstance(wire, str)
                        or (
                            wire != MXFP4_MARLIN_WIRE_ID
                            and _CB_WIRE_ID.fullmatch(wire) is None
                        )
                        or not isinstance(prefix, str)
                        or not prefix
                        or not isinstance(expert_ids, list)
                        or not expert_ids
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < 0
                            for value in expert_ids
                        )
                        or expert_ids != sorted(set(expert_ids))
                        or seen_experts.intersection(expert_ids)
                    ):
                        raise ValueError("per-expert format group values are malformed")
                    seen_experts.update(expert_ids)
                    expert_wires.append(wire)

    return MXFP4_MARLIN_WIRE_ID in delegated_wires + expert_wires
