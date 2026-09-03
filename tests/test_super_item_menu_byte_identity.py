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
  ``activation_pricing`` and ``serving_lane`` -- floats by ``repr``, which
  round-trips exactly, so the fixture records the full float64 value;
* ``stats_ext[super]["_memory_bytes_by_format"]`` (the exact-bytes map every
  downstream byte path prefers) and the whole ``costs_ext[super]`` row,
  including ``predicted_dloss_stderr`` and the applied-pricing marker;
* the pass-through rows' names, so a change cannot quietly move a Linear
  into or out of a group.

The scenarios deliberately cover the branches the wiring work touches: both
aggregators, ``PRISMAQUANT_COST_UCB_Z`` at 0 and non-zero (the hedge
conversion), calibrated gains, activation fair pricing, and the role-split
packed profile.

How each field is compared
--------------------------

Everything that decides BYTES or DP STRUCTURE is compared EXACTLY: menu order,
``fmt``, ``memory_bytes``, ``_memory_bytes_by_format``, ``n_params``, the
member lists, both serialized identities, ``activation_pricing``,
``serving_lane``, the applied-pricing marker and the pass-through row names.
Those are ints, bools, ``None`` and identity strings -- there is no float
question to ask about them, and they are what the exported artifact is made of.

The four ACCUMULATED float fields (``_FLOAT_FIELDS``) are compared to float64
precision instead.  They must be compared, not dropped: the DP ranks on
``predicted_dloss``, so a structure-only gate would miss a real cost regression
entirely.  But pinning them to the LAST bit pins the reduction implementation
rather than the aggregation:

    a 9-member packed group's ``predicted_dloss`` is one builtin ``sum()``
    over floats, and CPython 3.12 gave that ``sum()`` Neumaier compensated
    summation (gh-100425) where 3.11 sums naively left-to-right.  Same
    aggregation, same inputs, a fraction of a ulp apart.  Measured on this
    interpreter (3.12.3, where ``sum([1e16, 1.0, -1e16])`` returns ``1.0``,
    so the compensation is live) by shadowing ``sum`` inside
    ``allocator_candidates`` while the payload is built: of the 48 float
    reductions the five scenarios run (lengths 2, 3, 6 and 9), 6 differ from
    the naive left-to-right accumulation of the same terms, the worst by
    0.833 ulps of ``eps * |value|`` -- against the 16-ulp bound below.  No
    numpy takes part in this call chain, so the interpreter's reduction is
    the whole of the difference.

An exact pin on those four therefore asserted summation order, not the
property this module exists to protect -- the same defect, and the same fix,
as ``tests/test_math_reunderwrite_pins.py`` (PR #90).

Regenerate ONLY when a behaviour change is intended and argued -- and the
argument is required, not requested: ``--regen`` refuses without it and
stamps it into the fixture beside the digest::

    PRISMAQUANT_REGEN_GOLDEN_REASON="why this change is intended" \\
        PYTHONPATH=. python tests/test_super_item_menu_byte_identity.py --regen
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

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
    build_candidates,
    packed_role_split_profile,
)

GOLDEN_PATH = Path(__file__).with_name("fixtures") / "super_item_menu_golden.json"

MENU = ("NVFP4", "FP8_E4M3", "BF16")

# Fields ``_num`` renders as a float ``repr``.  Every OTHER leaf in the payload
# is an int, a bool, ``None`` or an identity string and is compared exactly.
_FLOAT_FIELDS = frozenset({
    "bits_per_param",
    "predicted_dloss",
    "predicted_dloss_stderr",
    "weight_mse",
})

# Principle 2: the bound comes from the dtype, not from what made the test
# pass.  These are Python floats, i.e. IEEE-754 binary64, so the unit is
# ``sys.float_info.epsilon`` (2.22e-16).
#
# 16 ulps covers the longest chain any pinned field accumulates over.  The
# worst case is a packed group's hedged ``predicted_dloss``: per member, the
# cost-entry product chain (0.5 * h_trace * mse, then the calibrated gain and
# the per-family activation penalty) is ~5 roundings; those 9 member terms are
# then summed (a 9-term reduction, <= 9 roundings naive, ~2 with the 3.12
# compensated sum); the linear hedge is subtracted and ``z * sqrt(sum stderr^2)``
# added, another ~4.  ~18 roundings at <= 0.5 ulp each, over an all-positive
# sum whose condition number is 1, is a bound near 9 ulps; 16 is the next
# power of two above it and matches the multiplier PR #90 derived the same way.
#
# Measured against the naive left-to-right reduction 3.11's ``sum()`` used
# (the module docstring records the check), the worst divergence across all
# five scenarios is 0.833 ulps in these units, against a bound of 16 -- so the
# tolerance is neither tuned to the observed noise nor a headroom guess.
# ``test_the_golden_is_load_bearing`` records how large a driver perturbation
# the bound still catches.
_FLOAT_TOLERANCE_ULPS = 16.0
_FLOAT64_EPS = sys.float_info.epsilon


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
        return alloc_cand.aggregate_fused_siblings(
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
        return alloc_cand.aggregate_packed_serving_groups(
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


REGEN_REASON_ENV = "PRISMAQUANT_REGEN_GOLDEN_REASON"


def regen_reason(env) -> str:
    """The argued reason for rewriting the golden, or a refusal.

    ``--regen`` can turn any regression in default-path code into a green
    suite, so it refuses unless the argument travels with it.  The reason is
    stamped beside the ``digest`` -- provenance, not compared payload, so the
    gate itself is unaffected.
    """
    reason = (env.get(REGEN_REASON_ENV) or "").strip()
    if not reason:
        raise SystemExit(
            f"refusing to rewrite {GOLDEN_PATH.name}: set "
            f'{REGEN_REASON_ENV}="<why this behaviour change is intended>" '
            "alongside --regen. A golden regenerated to make a red run green "
            "is not a gate."
        )
    return reason


def digest(payload: dict) -> str:
    """Provenance stamp written by ``--regen``.

    NOT the gate.  It hashes the float ``repr``s, so it moves on a 1-ulp
    reduction difference; ``compare_to_golden`` is what the tests assert.
    """
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _float_mismatch(current: str, golden: str) -> str | None:
    """Compare one accumulated float field to float64 precision.

    Returns ``None`` when they agree, else a description.  A golden of exactly
    ``0.0`` gets a tolerance of exactly 0.0 -- a bit-exact rung must stay
    bit-exact, and this falls out of the relative form rather than needing a
    special case.
    """
    cur = float(current)
    gold = float(golden)
    if cur == gold:
        return None
    tol = _FLOAT_TOLERANCE_ULPS * _FLOAT64_EPS * abs(gold)
    delta = abs(cur - gold)
    if delta <= tol:
        return None
    ulps = delta / max(abs(gold) * _FLOAT64_EPS, 5e-324)
    return (f"{current} != {golden} (delta {delta:.6g} = {ulps:.1f} ulps, "
            f"bound {tol:.6g} = {_FLOAT_TOLERANCE_ULPS:.0f} ulps)")


def _walk(current, golden, path: str, field: str, out: list[str]) -> None:
    if isinstance(golden, dict):
        if not isinstance(current, dict):
            out.append(f"{path}: expected an object, got {type(current).__name__}")
            return
        # Both directions: a key ADDED by the driver is a behaviour change too.
        for key in sorted(set(golden) - set(current)):
            out.append(f"{path}/{key}: missing (golden has {golden[key]!r})")
        for key in sorted(set(current) - set(golden)):
            out.append(f"{path}/{key}: added (value {current[key]!r})")
        for key in sorted(set(golden) & set(current)):
            _walk(current[key], golden[key], f"{path}/{key}", key, out)
        return
    if isinstance(golden, list):
        if not isinstance(current, list):
            out.append(f"{path}: expected a list, got {type(current).__name__}")
            return
        if len(current) != len(golden):
            out.append(
                f"{path}: length {len(current)} != {len(golden)} -- the menu "
                "order and membership are pinned exactly")
            return
        for i, (c, g) in enumerate(zip(current, golden)):
            _walk(c, g, f"{path}[{i}]", field, out)
        return
    if (field in _FLOAT_FIELDS
            and isinstance(current, str) and isinstance(golden, str)):
        problem = _float_mismatch(current, golden)
        if problem is not None:
            out.append(f"{path}: {problem}")
        return
    if current != golden:
        out.append(f"{path}: {current!r} != {golden!r}")


def compare_to_golden(current: dict, golden: dict) -> list[str]:
    """Every mismatch between a built payload and the fixture, or ``[]``.

    THE gate.  Exact on every structural field; float64-precision on the four
    accumulated float fields.
    """
    out: list[str] = []
    _walk(current["scenarios"], golden["scenarios"], "", "", out)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_super_item_menus_match_the_golden():
    assert GOLDEN_PATH.exists(), (
        f"{GOLDEN_PATH} is missing. It is the pre-change record of what the "
        "two aggregators produce; regenerate it only with an argued "
        "behaviour change."
    )
    golden = json.loads(GOLDEN_PATH.read_text())
    current = _build_payload()

    # What the digest assertion this replaces actually claimed to catch. Both
    # directions, so an ADDED scenario is caught as well as a removed one.
    assert sorted(current["scenarios"]) == sorted(golden["scenarios"]), (
        "the scenario set changed: "
        f"added {sorted(set(current['scenarios']) - set(golden['scenarios']))}, "
        f"removed {sorted(set(golden['scenarios']) - set(current['scenarios']))}"
    )

    problems = compare_to_golden(current, golden)
    assert not problems, (
        "super-item menu changed -- the aggregators are default-path code and "
        "this is the principle-6 gate:\n  " + "\n  ".join(problems)
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

    Two teeth, one per half of the comparison, and both mutate the DRIVER --
    never the fixture, which would only prove the comparator can read its own
    output.

    STRUCTURAL: reorder every super item's menu (same SET of formats, different
    DP order -- the minimal perturbation an order-blind record would miss).
    No float tolerance can absorb this, and none should.  The perturbation is
    equivalent to editing the aggregators' menu loop to ``for spec in
    reversed(formats):``; that edit was run against this fixture and the
    comparison reported ``menu[0]/fmt: 'BF16' != 'NVFP4'`` and the matching
    ``memory_bytes`` / ``bits_per_param`` / ``predicted_dloss`` swaps on every
    super item of all five scenarios.

    NUMERIC: scale every member's predicted dloss by 1 + 1e-12.  This is the
    tooth the tolerance could blunt, so it is asserted in-tree rather than
    only recorded in a commit message.  1e-12 relative sits ~280x above the
    3.55e-15 bound and far above the largest reduction difference measured
    (0.833 ulps), so it can neither flake nor pass.  Bisected on this tree:
    1e-14 is CAUGHT (56 mismatches, as is 1e-12), 1e-15 is NOT -- the gate
    resolves a
    driver change nine orders of magnitude finer than the 1e-5 / 1e-6
    threshold PR #90 settled for in float32.
    """
    golden = json.loads(GOLDEN_PATH.read_text())
    assert not compare_to_golden(_build_payload(), golden)

    # The order-deriving step is the aggregators' own ``for spec in formats:``
    # loop over the candidate list (``allocator_candidates.py`` :2925 fused,
    # :3270 packed).  Reversing the sequence those loops walk is exactly the
    # one-token driver edit ``for spec in reversed(formats):``, reached without
    # editing the module under test.  ``reversed`` rather than a sort so the
    # order is guaranteed to differ for any menu of two or more distinct rungs,
    # whatever MENU happens to hold.
    assert len(set(MENU)) >= 2, "a one-rung menu has no order to perturb"
    originals = {
        name: getattr(alloc_cand, name)
        for name in ("aggregate_fused_siblings", "aggregate_packed_serving_groups")
    }

    def _reversed_menu_order(original):
        def aggregate(stats, costs, formats, *args, **kwargs):
            return original(stats, costs, list(reversed(formats)), *args, **kwargs)
        return aggregate

    with mock.patch.multiple(
        alloc_cand,
        **{
            name: _reversed_menu_order(original)
            for name, original in originals.items()
        },
    ):
        assert compare_to_golden(_build_payload(), golden), (
            "reordering every super item's menu did not fail the comparison "
            "-- the golden does not pin what it claims to pin"
        )

    original_dloss = alloc_cand.cost_entry_predicted_dloss

    def perturbed(*args, **kwargs):
        return original_dloss(*args, **kwargs) * (1.0 + 1e-12)

    with mock.patch.object(
        alloc_cand, "cost_entry_predicted_dloss", perturbed
    ):
        problems = compare_to_golden(_build_payload(), golden)
    assert problems, (
        "a 1e-12 relative change to every member's predicted dloss did not "
        "fail the comparison -- the float tolerance is too loose to be a gate"
    )

    assert not compare_to_golden(_build_payload(), golden), (
        "the golden did not compare clean again after the mutations were "
        "undone"
    )


def test_regeneration_refuses_without_an_argued_reason():
    """``--regen`` rewrites the gate's own record, so it must be argued.

    The docstring has always said "regenerate ONLY when a behaviour change is
    intended and argued".  Nothing enforced it, and an unenforced rule loses
    to the next red run: rewriting the golden turns any regression into a
    green suite.  The reason is now required, and stamped into the fixture
    beside the digest.
    """
    for env in ({}, {REGEN_REASON_ENV: "   "}):
        with pytest.raises(SystemExit) as caught:
            regen_reason(env)
        assert REGEN_REASON_ENV in str(caught.value), str(caught.value)
    assert regen_reason(
        {REGEN_REASON_ENV: "  the packed hedge changed on purpose  "}
    ) == "the packed hedge changed on purpose"


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
    _reason = regen_reason(os.environ)
    _payload = _build_payload()
    _payload["digest"] = digest(_payload)
    _payload["regen_reason"] = _reason
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(_payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH} digest={_payload['digest']} reason={_reason!r}")
