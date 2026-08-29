"""Byte-identity gate on the two super-item aggregators (principle 6).

``aggregate_fused_siblings`` and ``aggregate_packed_serving_groups`` are
DEFAULT-PATH code: every production allocation runs both.  This module pins
their output -- every super item's whole menu, byte for byte and float for
float -- against a golden digest committed BEFORE the trellis-seam wiring
touched either function, so a change to those functions that perturbs an
existing scalar-only menu fails here instead of silently re-allocating a
27B artifact.

What is pinned, per scenario:

* every super item's candidate list, IN ORDER, with ``fmt``,
  ``bits_per_param``, ``memory_bytes``, ``predicted_dloss``,
  ``serialized_identity``, ``serialized_sidecar_identity``,
  ``activation_pricing`` and ``serving_lane`` -- floats by ``repr`` so the
  comparison is exact rather than tolerance-based;
* ``stats_ext[super]["_memory_bytes_by_format"]`` (the exact-bytes map every
  downstream byte path prefers) and the whole ``costs_ext[super]`` row,
  including ``predicted_dloss_stderr`` and the applied-pricing marker;
* the pass-through rows' names, so a change cannot quietly move a Linear
  into or out of a group.

The scenarios deliberately cover the branches the wiring work touches: both
aggregators, ``PRISMAQUANT_COST_UCB_Z`` at 0 and non-zero (the hedge
conversion), calibrated gains, activation fair pricing, and the role-split
packed profile.

Regenerate ONLY when a behaviour change is intended and argued::

    PYTHONPATH=. python tests/test_super_item_menu_byte_identity.py --regen
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest import mock

from prismaquant import allocator_candidates as alloc_cand
from prismaquant import format_registry as fr
from prismaquant.activation_fair_pricing import (
    REASON_CALIBRATED,
    ActivationFairPricing,
    FamilyCalibration,
)
from prismaquant.allocator_candidates import (
    _FUSED_SIBLING_MARKER,
    _PACKED_GROUP_MARKER,
    aggregate_fused_siblings,
    aggregate_packed_serving_groups,
    build_candidates,
    packed_role_split_profile,
)

GOLDEN_PATH = Path(__file__).with_name("fixtures") / "super_item_menu_golden.json"

MENU = ("NVFP4", "FP8_E4M3", "BF16")


# ---------------------------------------------------------------------------
# Fixtures: a dense attention+MLP block and a packed-MoE layer
# ---------------------------------------------------------------------------
class _DenseProfile:
    def fused_sibling_group(self, name: str) -> str | None:
        if name.endswith((".q_proj", ".k_proj", ".v_proj")):
            return name.rsplit(".", 1)[0] + ".qkv_proj"
        if name.endswith((".gate_proj", ".up_proj")):
            return name.rsplit(".", 1)[0] + ".gate_up_proj"
        return None


class _PackedProfile:
    """Two MoE layers of 3 experts x {w1, w3, w2}, grouped per layer."""

    def packed_expert_format_group(self, name: str) -> str | None:
        if ".experts." not in name:
            return None
        return name.split(".experts.")[0] + ".experts"

    def packed_expert_role_group(self, name: str) -> str | None:
        if name.endswith((".w1", ".w3")):
            return "gate_up_proj"
        if name.endswith(".w2"):
            return "down_proj"
        return None


def _row(d_out: int, d_in: int, h_trace: float, mse: float,
         stderr: float) -> tuple[dict, dict]:
    stats = {
        "h_trace": h_trace,
        "h_trace_raw": h_trace * 1.5,
        "h_w2_sum": h_trace * 0.25,
        "w_max_abs": 0.75,
        "w_norm_sq": float(d_out * d_in) * 1e-4,
        "n_params": d_out * d_in,
        "in_features": d_in,
        "out_features": d_out,
        "n_tokens_seen": 4096,
    }
    costs = {
        "NVFP4": {
            "weight_mse": mse,
            "predicted_dloss": 0.5 * h_trace * mse,
            "predicted_dloss_stderr": stderr,
        },
        "FP8_E4M3": {
            "weight_mse": mse * 0.05,
            "predicted_dloss": 0.5 * h_trace * mse * 0.05,
            "predicted_dloss_stderr": stderr * 0.05,
        },
        "BF16": {
            "weight_mse": 0.0,
            "predicted_dloss": 0.0,
            "predicted_dloss_stderr": 0.0,
        },
    }
    return stats, costs


def _dense_inputs() -> tuple[dict, dict]:
    layer = "model.layers.0"
    shapes = {
        "self_attn.q_proj": (4096, 4096),
        "self_attn.k_proj": (1024, 4096),
        "self_attn.v_proj": (1024, 4096),
        "self_attn.o_proj": (4096, 4096),
        "mlp.gate_proj": (11008, 4096),
        "mlp.up_proj": (11008, 4096),
        "mlp.down_proj": (4096, 11008),
    }
    h = {
        "self_attn.q_proj": 0.5, "self_attn.k_proj": 0.3,
        "self_attn.v_proj": 0.7, "self_attn.o_proj": 0.4,
        "mlp.gate_proj": 0.8, "mlp.up_proj": 0.6, "mlp.down_proj": 0.9,
    }
    mse = {
        "self_attn.q_proj": 0.021, "self_attn.k_proj": 0.017,
        "self_attn.v_proj": 0.033, "self_attn.o_proj": 0.011,
        "mlp.gate_proj": 0.041, "mlp.up_proj": 0.029, "mlp.down_proj": 0.013,
    }
    stderr = {
        "self_attn.q_proj": 0.0031, "self_attn.k_proj": 0.0011,
        "self_attn.v_proj": 0.0052, "self_attn.o_proj": 0.0007,
        "mlp.gate_proj": 0.0063, "mlp.up_proj": 0.0044, "mlp.down_proj": 0.0019,
    }
    stats: dict = {}
    costs: dict = {}
    for leaf, shape in shapes.items():
        name = f"{layer}.{leaf}"
        stats[name], costs[name] = _row(
            shape[0], shape[1], h[leaf], mse[leaf], stderr[leaf])
    return stats, costs


def _packed_inputs() -> tuple[dict, dict]:
    stats: dict = {}
    costs: dict = {}
    for layer in (0, 1):
        for expert in range(3):
            for leaf, (d_out, d_in) in (
                ("w1", (768, 2048)),
                ("w3", (768, 2048)),
                ("w2", (2048, 768)),
            ):
                name = f"model.layers.{layer}.mlp.experts.{expert}.{leaf}"
                seed = layer * 7 + expert * 3 + len(leaf)
                stats[name], costs[name] = _row(
                    d_out, d_in,
                    0.2 + 0.05 * seed,
                    0.01 + 0.002 * seed,
                    0.0004 * (seed + 1),
                )
    # One dense row that must pass through untouched.
    stats["model.layers.0.self_attn.o_proj"], \
        costs["model.layers.0.self_attn.o_proj"] = _row(
            4096, 4096, 0.4, 0.011, 0.0007)
    return stats, costs


def _pricing() -> ActivationFairPricing:
    def fit(family: str, penalty: float) -> FamilyCalibration:
        return FamilyCalibration(
            family=family,
            n_rows=8,
            penalty=penalty,
            log2_penalty=0.0,
            log2_stdev=0.0,
            log2_stderr=0.0,
            log2_residual_min=0.0,
            log2_residual_max=0.0,
            formats=(),
            per_format_log2_penalty=(),
            rung_dependence_log2_range=0.0,
            sample=(),
            rows_digest="test",
        )

    return ActivationFairPricing(
        enabled=True,
        reason=REASON_CALIBRATED,
        families={"nv": fit("nv", 1.37), "fp": fit("fp", 1.09)},
        measured_rows_by_family={"nv": 8, "fp": 8},
        weight_only_rows_by_family={"nv": 4, "fp": 4},
        uncalibrated_families=(),
    )


GAINS = {"NVFP4": 1.23, "FP8_E4M3": 0.88}


def _specs() -> list:
    return [fr.REGISTRY[name] for name in MENU]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def _run_fused(ucb_z: str, *, gains, pricing):
    stats, costs = _dense_inputs()
    specs = _specs()
    with mock.patch.dict(os.environ, {"PRISMAQUANT_COST_UCB_Z": ucb_z}):
        cands = build_candidates(
            stats, costs, specs, calibrated_gains=gains,
            activation_pricing=pricing)
        return aggregate_fused_siblings(
            stats, costs, specs, cands, _DenseProfile(),
            calibrated_gains=gains, activation_pricing=pricing)


def _run_packed(ucb_z: str, *, gains, pricing, role_split: bool):
    stats, costs = _packed_inputs()
    specs = _specs()
    profile = _PackedProfile()
    if role_split:
        profile = packed_role_split_profile(profile)
    with mock.patch.dict(os.environ, {"PRISMAQUANT_COST_UCB_Z": ucb_z}):
        cands = build_candidates(
            stats, costs, specs, calibrated_gains=gains,
            activation_pricing=pricing)
        return aggregate_packed_serving_groups(
            stats, costs, specs, cands, profile,
            calibrated_gains=gains, activation_pricing=pricing)


SCENARIOS = {
    "fused_plain": lambda: _run_fused("0", gains=None, pricing=None),
    "fused_hedged_gained_priced": lambda: _run_fused(
        "1.5", gains=GAINS, pricing=_pricing()),
    "packed_plain": lambda: _run_packed(
        "0", gains=None, pricing=None, role_split=False),
    "packed_hedged_gained_priced": lambda: _run_packed(
        "1.5", gains=GAINS, pricing=_pricing(), role_split=False),
    "packed_role_split_hedged": lambda: _run_packed(
        "1.5", gains=GAINS, pricing=_pricing(), role_split=True),
}


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------
def _num(value):
    """Exact serialization of a float: repr round-trips, str() may not."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _candidate_payload(cand) -> dict:
    return {
        "fmt": cand.fmt,
        "bits_per_param": _num(cand.bits_per_param),
        "memory_bytes": int(cand.memory_bytes),
        "predicted_dloss": _num(cand.predicted_dloss),
        "serialized_identity": cand.serialized_identity,
        "serialized_sidecar_identity": cand.serialized_sidecar_identity,
        "activation_pricing": (
            None if cand.activation_pricing is None
            else str(cand.activation_pricing)
        ),
        "serving_lane": (
            None if cand.serving_lane is None else repr(cand.serving_lane)
        ),
    }


def _scenario_payload(triple) -> dict:
    stats_ext, costs_ext, cands_ext = triple
    supers = sorted(
        n for n in cands_ext
        if _FUSED_SIBLING_MARKER in n or _PACKED_GROUP_MARKER in n
    )
    passthrough = sorted(set(cands_ext) - set(supers))
    payload = {
        "super_items": {},
        "passthrough_rows": passthrough,
    }
    for name in supers:
        payload["super_items"][name] = {
            # IN ORDER: menu order is the DP's tie-break, so a reordering is
            # a behaviour change even when every field matches as a set.
            "menu": [_candidate_payload(c) for c in cands_ext[name]],
            "memory_bytes_by_format": {
                fmt: int(nbytes) for fmt, nbytes
                in sorted(stats_ext[name]["_memory_bytes_by_format"].items())
            },
            "cost_rows": {
                fmt: {k: _num(v) for k, v in sorted(row.items())}
                for fmt, row in sorted(costs_ext[name].items())
            },
            "n_params": int(stats_ext[name]["n_params"]),
            "members": sorted(
                stats_ext[name].get("_fused_siblings")
                or stats_ext[name].get("_packed_group_members")
                or []
            ),
        }
    return payload


def _build_payload() -> dict:
    return {
        "schema": "prismaquant.super_item_menu_golden.v1",
        "menu": list(MENU),
        "scenarios": {
            name: _scenario_payload(build())
            for name, build in sorted(SCENARIOS.items())
        },
    }


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def digest(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_super_item_menus_are_byte_identical_to_the_golden():
    assert GOLDEN_PATH.exists(), (
        f"{GOLDEN_PATH} is missing. It is the pre-change record of what the "
        "two aggregators produce; regenerate it only with an argued "
        "behaviour change."
    )
    golden = json.loads(GOLDEN_PATH.read_text())
    current = _build_payload()
    for name in sorted(golden["scenarios"]):
        assert name in current["scenarios"], f"scenario {name} disappeared"
        assert current["scenarios"][name] == golden["scenarios"][name], (
            f"super-item menu changed in scenario {name!r}: the aggregators "
            "are default-path code and this is the principle-6 gate"
        )
    assert digest(current) == golden["digest"], (
        "aggregate digest changed even though every recorded scenario "
        "matched -- a scenario was added or removed"
    )


def test_the_golden_actually_exercises_both_aggregators_and_the_hedge():
    """A golden that recorded nothing would pass the test above forever."""
    payload = _build_payload()
    fused = payload["scenarios"]["fused_plain"]["super_items"]
    packed = payload["scenarios"]["packed_plain"]["super_items"]
    assert len(fused) == 2, sorted(fused)
    assert all(_FUSED_SIBLING_MARKER in n for n in fused)
    assert len(packed) == 2, sorted(packed)
    assert all(_PACKED_GROUP_MARKER in n for n in packed)
    # Every super item offers the whole 3-rung menu, so the pin covers the
    # per-format loop rather than a single surviving format.
    for group in (fused, packed):
        for entry in group.values():
            assert [c["fmt"] for c in entry["menu"]] == list(MENU)
            assert sorted(entry["memory_bytes_by_format"]) == sorted(MENU)

    # The hedge scenario must actually differ from the unhedged one, or the
    # UCB conversion is untested.
    plain = payload["scenarios"]["fused_plain"]["super_items"]
    hedged = payload["scenarios"]["fused_hedged_gained_priced"]["super_items"]
    assert set(plain) == set(hedged)
    differs = [
        name for name in plain
        if plain[name]["menu"] != hedged[name]["menu"]
    ]
    assert differs, "UCB z / gains / activation pricing changed nothing"
    for name in plain:
        row = hedged[name]["cost_rows"]["NVFP4"]
        assert float(row["predicted_dloss_stderr"]) > 0.0

    # Role-split really splits: 2 layers x 2 roles = 4 units, not 2.
    split = payload["scenarios"]["packed_role_split_hedged"]["super_items"]
    assert len(split) == 4, sorted(split)


def test_the_golden_is_load_bearing():
    """A passing golden is not evidence until a driver change makes it fail.

    Mutate the aggregators' menu-order helper at runtime -- same SET of
    formats, different DP order, the minimal perturbation an order-blind
    record would miss -- and the digest must move.  Restoring it must bring
    the recorded digest back, so the check is the fixture's, not the run's.
    """
    golden = json.loads(GOLDEN_PATH.read_text())
    assert digest(_build_payload()) == golden["digest"]

    original = alloc_cand._super_menu_format_names

    def reordered(formats, member_menu_intersection):
        return tuple(sorted(original(formats, member_menu_intersection)))

    with mock.patch.object(
        alloc_cand, "_super_menu_format_names", reordered
    ):
        mutated = digest(_build_payload())
    assert mutated != golden["digest"], (
        "reordering every super item's menu did not change the recorded "
        "payload -- the golden does not pin what it claims to pin"
    )
    assert digest(_build_payload()) == golden["digest"], (
        "the golden did not return to its recorded value after the mutation "
        "was undone"
    )


def test_passthrough_rows_are_not_swallowed_by_either_aggregator():
    payload = _build_payload()
    fused = payload["scenarios"]["fused_plain"]
    assert fused["passthrough_rows"] == [
        "model.layers.0.mlp.down_proj",
        "model.layers.0.self_attn.o_proj",
    ]
    packed = payload["scenarios"]["packed_plain"]
    assert packed["passthrough_rows"] == ["model.layers.0.self_attn.o_proj"]


if __name__ == "__main__":  # pragma: no cover - regeneration entry point
    import sys

    if "--regen" not in sys.argv:
        raise SystemExit("pass --regen to rewrite the golden")
    _payload = _build_payload()
    _payload["digest"] = digest(_payload)
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(_payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH} digest={_payload['digest']}")
