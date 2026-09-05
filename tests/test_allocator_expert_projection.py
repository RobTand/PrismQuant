"""The allocation carries what the campaign priced the expert population under.

PrismaQuant #183: the campaign prices the profile-declared per-expert units
under the producer's projection and seals a priced-wire receipt per unit and
rung.  The allocator selects one rung per executed stack; the allocation must
say which population was priced, under which projection, and carry the
receipt of exactly the rung it selected for every projected unit -- so the
export lane can hand the exporter the priced bytes and nothing else.  A
selected rung with no receipt, or a projected unit the allocation does not
place, is refused by name before the layer config is written.

The real ``allocator.main()`` is driven here (the block is main()'s wiring,
not a helper); Tessera admission uses the same v5 contract fixture as
``test_tessera_scope_endpoints.py``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import sys

import pytest

from prismaquant import tessera_expert_projection as tep
from prismaquant.tessera_expert_projection import (
    EXPERT_WIRES_KEY,
    POPULATION_KEY,
    POPULATION_SCHEMA,
    PROJECTION_KEY,
    ExpertProjectionError,
    allocation_expert_projection_block,
    bind_expert_projection,
    carried_projection,
    stack_plan_request,
)

IMAGE = "example/runtime@sha256:" + "a" * 64
STACK = "model.layers.2.feed_forward.experts"
ROUTER = "model.layers.2.feed_forward.gate"
DENSE = "model.layers.0.self_attn.out_proj"
FMT = "TESSERA_E4M3_K1_R1024"
SHARD = "model.safetensors"
N = 256  # every synthetic unit is N x N: one Tessera superblock square


def _units():
    return [f"{STACK}.{expert}.{role}" for expert in range(2) for role in ("w1", "w3", "w2")]


def _producer_projection() -> dict:
    """A ``tessera.expert_projection.v1`` answer for one two-expert stack."""
    units, tensors = [], {}
    for expert in range(2):
        for role, projection, group in (("w1", "gate_proj", "w13"), ("w3", "up_proj", "w13"),
                                        ("w2", "down_proj", "w2")):
            tensor = f"{STACK}.{expert}.{role}.weight"
            units.append({
                "tensor": tensor, "wire": f"{STACK}.{expert}.{role}.wire",
                "source_tensor": tensor, "source_layout": tep.SOURCE_LAYOUT_UNPACKED,
                "source_slice": {"expert": expert, "selector": "whole", "transpose": False},
                "expert": expert, "projection": projection, "group": group,
                "rows": N, "cols": N,
            })
            tensors[tensor] = SHARD
    return {
        "schema": tep.PROJECTION_SCHEMA,
        "stacks": {STACK: {"source_layout": tep.SOURCE_LAYOUT_UNPACKED, "grid": "E4M3",
                           "q256": 1024, "experts": 2, "units": units}},
        "source": {"config_sha256": "c" * 64, "auxiliary_sha256": {},
                   "files": {SHARD: "f" * 64}, "tensors": tensors},
    }


def _carried() -> dict:
    producer = _producer_projection()
    declared = {STACK: {name: (N, N) for name in _units()}}
    return carried_projection(producer, bind_expert_projection(producer, declared=declared),
                              request=stack_plan_request({STACK: ("E4M3", 1024)}), tool="t")


def _receipt(name: str, unit: dict, fmt: str = FMT) -> dict:
    """A producer ``make_unit_record`` receipt for ``name`` at ``fmt``."""
    from prismaquant.tessera_formats import parse_tessera_format_name
    family, q256 = parse_tessera_format_name(fmt)
    blob = hashlib.sha256(f"{name}:{fmt}".encode()).digest() * 4
    return {
        "file": name.replace(".", "__") + f"__{fmt}.tessera",
        "blob_sha256": hashlib.sha256(blob).hexdigest(), "blob_bytes": len(blob),
        "identity": {
            "schema": "tessera.cached_unit_inputs.v1", "unit": name,
            "recipe": {"grid": family.payload_grid().name, "q256": int(q256)},
            "projection": {key: unit[key] for key in tep.UNIT_IDENTITY_KEYS},
        },
    }


def _population() -> dict:
    return {
        "schema": POPULATION_SCHEMA, "layer_stride": 1,
        "priced": {"dense": [DENSE], "routed_experts": _units(),
                   "packed_parameters": {f"{STACK}.gate_up_proj": [2, 2 * N, N],
                                         f"{STACK}.down_proj": [2, N, N]},
                   "stacks": [STACK]},
        "omitted": {"dense_outside_layer_stride": [], "packed_outside_layer_stride": {},
                    "pinned": [ROUTER]},
        "counts": {"dense_priced": 1, "routed_experts_priced": 6, "dense_omitted": 0,
                   "packed_omitted": 0, "pinned": 1},
    }


def _cost_payload(tmp_path, *, formats=(FMT,)) -> dict:
    from prismaquant.cost_currency import RENDER_SCORE_COST_MODE
    from prismaquant.tessera_campaign import CURRENCY

    carried = _carried()
    units = carried["stacks"][STACK]
    row = {"weight_mse": 1e-4, "output_mse": 4e-4, "output_mse_measured": True,
           "currency": CURRENCY}
    return {
        "costs": {name: {fmt: dict(row) for fmt in formats} for name in (DENSE, *_units())},
        "formats": list(formats),
        "provenance": {"cost_mode": RENDER_SCORE_COST_MODE, "wire_dir": str(tmp_path / "wire"),
                       POPULATION_KEY: _population(), PROJECTION_KEY: carried},
        EXPERT_WIRES_KEY: {name: {fmt: _receipt(name, units[name], fmt) for fmt in formats}
                           for name in _units()},
    }


def _v5_contract(monkeypatch):
    from prismaquant import tessera_menu as menu
    from prismaquant import tessera_runtime_contract as contract
    from prismaquant.lane_eligibility import LANE_ELIGIBILITY_SCHEMAS
    payload = json.loads(contract.contract_path().read_text())
    block = payload["lane_eligibility"]
    if block.get("schema") not in LANE_ELIGIBILITY_SCHEMAS:
        # The importable producer publishes a lane table this checkout's
        # reader does not accept yet (the re-pin is PrismaQuant #192); the
        # allocator cannot admit any Tessera rung here, so main()'s wiring
        # is exercised at the pinned producer and the block-level tests
        # below carry the refusals on every checkout.
        pytest.skip(f"packaged lane table {block.get('schema')!r} is not readable by this "
                    "checkout's lane_eligibility (PrismaQuant #192)")
    block["schema"] = "tessera.lane-eligibility.v5"
    for cell in block["cells"]:
        cell["runtime"] = {"image": IMAGE, "execution_modes": ["eager"]}
    extra = copy.deepcopy(block["cells"])
    for cell in extra:
        cell["id"] += "_expert_fixture"
        cell["structure"] = "routed_moe"
    block["cells"].extend(extra)
    block["structures"].append("routed_moe")
    parsed = contract._parse(payload, commit="fixture", sha="fixture", path="fixture")
    monkeypatch.setattr(menu, "tessera_runtime_contract", lambda: parsed)
    monkeypatch.setenv("PRISMAQUANT_TESSERA_MENU", "attested")


def _allocator_argv(tmp_path, payload, *, drop_probe_unit=None):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "lfm2_moe", "architectures": ["Lfm2MoeForCausalLM"]}))
    stats = {DENSE: dict(router_path=None, expert_id=None)}
    for name in _units():
        stats[name] = dict(router_path=ROUTER, expert_id=name.split(".")[-2])
    for row in stats.values():
        row.update(h_trace=1.0, n_params=N * N, in_features=N, out_features=N)
    if drop_probe_unit is not None:
        del stats[drop_probe_unit]
    probe = tmp_path / "probe.pkl"
    costs = tmp_path / "cost.pkl"
    probe.write_bytes(pickle.dumps({"stats": stats, "meta": {"model": str(model_dir)}}))
    costs.write_bytes(pickle.dumps(payload))
    return ["allocator", "--probe", str(probe), "--costs", str(costs), "--formats", FMT,
            "--allow-legacy-fisher-norm",
            "--target-profile", "tessera_research_sm121", "--target-bits", "16",
            "--pareto-targets", "16", "--bit-precision", "0.1",
            "--layer-config", str(tmp_path / "layer.json"),
            "--pareto-csv", str(tmp_path / "pareto.csv"),
            "--tessera-runtime-image", IMAGE, "--tessera-execution-mode", "eager",
            "--tessera-residency", "resident", "--tessera-platform", "sm_121"]


# ---------------------------------------------------------------------------
# main()'s wiring
# ---------------------------------------------------------------------------
def test_main_carries_projection_population_and_selected_receipts(tmp_path, monkeypatch):
    from prismaquant import allocator
    _v5_contract(monkeypatch)
    payload = _cost_payload(tmp_path)
    monkeypatch.setattr(sys, "argv", _allocator_argv(tmp_path, payload))
    allocator.main()
    from prismaquant.layer_config import load_assignment
    placed = load_assignment(tmp_path / "layer.json")
    assert {name: placed[name] for name in _units()} == {name: FMT for name in _units()}
    meta = json.loads((tmp_path / "layer.json").read_text())["__prismaquant__"]

    # The producer's projection and the campaign's population statement travel
    # unchanged; the receipts are exactly the selected rung's, one per unit.
    assert meta[PROJECTION_KEY] == payload["provenance"][PROJECTION_KEY]
    assert meta[POPULATION_KEY] == payload["provenance"][POPULATION_KEY]
    assert meta[EXPERT_WIRES_KEY] == {name: payload[EXPERT_WIRES_KEY][name][FMT]
                                      for name in _units()}
    assert meta["tessera_expert_stack_formats"] == {STACK: FMT}
    assert meta["tessera_expert_wire_dir"] == str(tmp_path / "wire")


def test_main_refuses_a_selected_rung_with_no_priced_wire(tmp_path, monkeypatch):
    from prismaquant import allocator
    _v5_contract(monkeypatch)
    payload = _cost_payload(tmp_path)
    del payload[EXPERT_WIRES_KEY][f"{STACK}.1.w2"][FMT]
    monkeypatch.setattr(sys, "argv", _allocator_argv(tmp_path, payload))
    with pytest.raises(SystemExit, match=rf"expert projection: {STACK}\.1\.w2: selected {FMT} "
                                          "has no priced wire receipt"):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()


def test_main_refuses_a_projected_unit_the_allocation_does_not_place(tmp_path, monkeypatch):
    from prismaquant import allocator
    _v5_contract(monkeypatch)
    payload = _cost_payload(tmp_path)
    monkeypatch.setattr(sys, "argv", _allocator_argv(
        tmp_path, payload, drop_probe_unit=f"{STACK}.0.w3"))
    with pytest.raises(SystemExit, match=rf"expert projection: .*not in the assignment.*{STACK}\.0\.w3"):
        allocator.main()
    assert not (tmp_path / "layer.json").exists()


def test_main_adds_nothing_for_a_table_without_a_population(tmp_path, monkeypatch):
    from prismaquant import allocator
    _v5_contract(monkeypatch)
    payload = _cost_payload(tmp_path)
    for key in (POPULATION_KEY, PROJECTION_KEY):
        del payload["provenance"][key]
    del payload[EXPERT_WIRES_KEY]
    monkeypatch.setattr(sys, "argv", _allocator_argv(tmp_path, payload))
    allocator.main()
    meta = json.loads((tmp_path / "layer.json").read_text())["__prismaquant__"]
    assert not {PROJECTION_KEY, POPULATION_KEY, EXPERT_WIRES_KEY,
                "tessera_expert_stack_formats", "tessera_expert_wire_dir"} & set(meta)


# ---------------------------------------------------------------------------
# The block itself: the refusals main() cannot reach through the real solver
# ---------------------------------------------------------------------------
def _assignment(fmt=FMT):
    return {DENSE: fmt, **{name: fmt for name in _units()}}


def test_block_refuses_a_role_split_stack_and_a_receipt_for_another_rung(tmp_path):
    payload = _cost_payload(tmp_path, formats=(FMT, "TESSERA_E4M3_K1_R768"))
    split = _assignment()
    split[f"{STACK}.0.w2"] = "TESSERA_E4M3_K1_R768"
    with pytest.raises(ExpertProjectionError, match="rungs differ across the stack"):
        allocation_expert_projection_block(payload, split)
    # A receipt filed under the selected rung but sealed for another one.
    swapped = copy.deepcopy(payload)
    swapped[EXPERT_WIRES_KEY][f"{STACK}.0.w1"][FMT] = \
        payload[EXPERT_WIRES_KEY][f"{STACK}.0.w1"]["TESSERA_E4M3_K1_R768"]
    with pytest.raises(ExpertProjectionError, match=rf"{STACK}\.0\.w1: .*not the selected rung"):
        allocation_expert_projection_block(swapped, _assignment())
    # A stack kept whole at a non-Tessera format needs no receipt and says so.
    block = allocation_expert_projection_block(payload, _assignment("BF16"))
    assert block["tessera_expert_stack_formats"] == {STACK: "BF16"}
    assert block[EXPERT_WIRES_KEY] == {}


@pytest.mark.parametrize("damage,match", [
    (lambda p: p.pop(EXPERT_WIRES_KEY), "no priced expert wires"),
    (lambda p: p["provenance"].pop("wire_dir"), "names no wire_dir"),
    (lambda p: p["provenance"].pop(PROJECTION_KEY), "priced expert wires but no producer projection"),
    (lambda p: p["provenance"][POPULATION_KEY].update(schema="other"), "population block"),
    (lambda p: p["provenance"][PROJECTION_KEY]["stacks"][STACK].pop(f"{STACK}.1.w1"),
     r"1\.w1: carried unit record disagrees|coverage|experts"),
])
def test_block_refuses_a_table_that_cannot_say_what_it_priced(tmp_path, damage, match):
    payload = _cost_payload(tmp_path)
    damage(payload)
    with pytest.raises(ExpertProjectionError, match=match):
        allocation_expert_projection_block(payload, _assignment())


def test_block_is_empty_for_a_stock_table(tmp_path):
    assert allocation_expert_projection_block({"costs": {}, "formats": []}, {}) == {}
    assert allocation_expert_projection_block(
        {"costs": {}, "formats": [], "provenance": {"cost_mode": "x"}}, {}) == {}
