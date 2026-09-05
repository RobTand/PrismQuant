"""The export lane binds selected routed units to the producer's projection.

PrismaQuant #183: the allocation carries the producer's expert projection,
the receipt of the rung it selected for every projected unit, and the
campaign's wire directory.  Before the external translator runs, the scope
gate re-binds every selected routed unit to THAT projection -- the producer's
record, not a source-member guess, attests the executed unit's geometry, so a
predicated cell resolves on it -- checks the priced bytes against their
receipts where they are about to be handed over, and the CLI writes the
producer's ``tessera.cached_units.v1`` bundle into the wire directory for the
exporter's ``--cached-expert-units`` intake.  A selected routed unit the
allocation carries no projection for stays refused by name.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

from prismaquant import tessera_export_lane as export
from prismaquant import tessera_expert_projection as tep
from prismaquant import tessera_serving_runtime_pin as pin
from prismaquant.tessera_expert_projection import (
    EXPERT_WIRES_KEY,
    POPULATION_KEY,
    POPULATION_SCHEMA,
    PROJECTION_KEY,
    STACK_FORMATS_KEY,
    WIRE_DIR_KEY,
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
N = 64


@dataclass(frozen=True)
class Target:
    platform: str = "sm_121"
    runtime_image: str = IMAGE
    execution_mode: str = "eager"
    residency: str = "resident"

    def as_dict(self):
        return asdict(self)


def _units():
    return [f"{STACK}.{expert}.{role}" for expert in range(2) for role in ("w1", "w3", "w2")]


def _context(structure="dense"):
    return {**Target().as_dict(), "structure": structure}


def _payload(predicates=()):
    family = "TESSERA_E4M3_K1"
    cells = []
    for structure in ("dense", "routed_moe"):
        for regime in ("decode", "batch"):
            cells.append({
                "id": f"{structure}_{regime}", "platform": "sm_121",
                "family": family, "structure": structure, "regime": regime,
                "rungs_q256": [1024], "activation_contract": "fp8_per_token_dynamic",
                "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
                "requires_plugin": "tessera",
                "predicates": [dict(p) for p in predicates] if structure == "routed_moe" else [],
                "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
                "executes": [{"symbol": "torch.bmm", "decoder": "torch_window"}],
                "runtime": {"image": IMAGE, "execution_modes": ["eager"]},
            })
    return {
        "formats": [{"family": family, "kind": "tessera_wire",
                     "name_pattern": family + "_R{k}",
                     "reader_rate_range_q256": [256, 2048],
                     "residency_modes": ["resident", "streamed"]}],
        "lane_eligibility": {
            "schema": "tessera.lane-eligibility.v5",
            "platforms": {"sm_121": {}}, "regimes": ["decode", "batch"],
            "structures": ["dense", "routed_moe"], "cells": cells,
        },
    }


def _producer_projection() -> dict:
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


def _blob(name: str, fmt: str = FMT) -> bytes:
    return hashlib.sha256(f"{name}:{fmt}".encode()).digest() * 4


def _receipt(name: str, unit: dict, fmt: str = FMT) -> dict:
    from prismaquant.tessera_formats import parse_tessera_format_name
    family, q256 = parse_tessera_format_name(fmt)
    blob = _blob(name, fmt)
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


def _hessian_block():
    return {
        "supplied": False, "identity_schema": "modern",
        "text_sha": "b" * 64, "token_count": 4096,
        "kwarg": ["ldl", "refit_metric"],
        "text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64,
        "fit_tokens": 4096, "stamped_rows": 7, "legacy_rows": 0,
        "unstamped_rows": 0,
    }


@pytest.fixture
def case(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "lfm2_moe", "architectures": ["Lfm2MoeForCausalLM"],
        "num_experts": 2,
    }))
    # Headers only: the gate never reads weights.
    header = json.dumps({
        name + ".weight": {"dtype": "BF16", "shape": [N, N],
                            "data_offsets": [index * N * N * 2, (index + 1) * N * N * 2]}
        for index, name in enumerate((DENSE, *_units()))
    }).encode()
    (model / SHARD).write_bytes(struct.pack("<Q", len(header)) + header)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(_payload()))
    monkeypatch.setattr(export, "packaged_contract_path", lambda: contract)
    wire_dir = tmp_path / "wire"
    wire_dir.mkdir()
    carried = _carried()
    units = carried["stacks"][STACK]
    receipts = {name: _receipt(name, units[name]) for name in _units()}
    for name, record in receipts.items():
        (wire_dir / record["file"]).write_bytes(_blob(name))
    assignment = tmp_path / "layer_config.json"
    payload = {
        name: {"data_type": "tessera", "bits": 4, "tessera_format": FMT}
        for name in (DENSE, *_units())
    }
    payload[ROUTER] = "BF16"
    payload["__prismaquant__"] = {
        "tessera_serving_scope": {
            "target": Target().as_dict(),
            "by_unit": {DENSE: _context(), **{name: _context("routed_moe") for name in _units()}},
        },
        "tessera_hessian": _hessian_block(),
        POPULATION_KEY: _population(),
        PROJECTION_KEY: carried,
        EXPERT_WIRES_KEY: receipts,
        STACK_FORMATS_KEY: {STACK: FMT},
        WIRE_DIR_KEY: str(wire_dir),
    }
    assignment.write_text(json.dumps(payload))
    return SimpleNamespace(model=model, assignment=assignment, payload=payload,
                           contract=contract, wire_dir=wire_dir, receipts=receipts,
                           units=units)


def _save(case):
    case.assignment.write_text(json.dumps(case.payload))


def _meta(case):
    return case.payload["__prismaquant__"]


def _scope(case):
    return export.require_assignment_scope(case.model, case.assignment, target=Target())


def test_scope_binds_selected_routed_units_to_the_carried_projection(case):
    report = _scope(case)
    assert set(report["by_unit"]) == {DENSE, *_units()}
    for name in _units():
        route = report["by_unit"][name]
        assert route["structure"] == "routed_moe"
        assert route["route_status"] == "backed_with_serve_flag"
        assert (route["out_features"], route["in_features"]) == (N, N)
    projection = report["expert_projection"]
    assert projection["stacks"] == {STACK: FMT}
    assert projection["wire_dir"] == str(case.wire_dir)
    assert projection["source"] == _carried()["producer"]["source"]
    assert set(projection["units"]) == set(_units())
    for name, record in projection["units"].items():
        assert record == case.receipts[name]


def test_scope_attests_a_predicated_cell_on_the_producers_geometry(case):
    case.contract.write_text(json.dumps(_payload(
        predicates=[{"fact": "out_features", "op": "equals", "value": N}])))
    report = _scope(case)
    assert all(report["by_unit"][name]["route_status"] == "backed_with_serve_flag"
               for name in _units())
    case.contract.write_text(json.dumps(_payload(
        predicates=[{"fact": "out_features", "op": "equals", "value": 2 * N}])))
    with pytest.raises(export.TesseraExportLaneError, match="regime|unattested"):
        _scope(case)


def test_scope_refuses_priced_wires_bound_to_no_carried_projection(case):
    # The projection is the only thing that binds a receipt to an executed
    # unit. An allocation with it stripped out, but still naming priced wires,
    # is not the pre-#183 lane -- it is a receipt that attests nothing.
    _meta(case).pop(PROJECTION_KEY)
    _save(case)
    with pytest.raises(export.TesseraExportLaneError,
                       match="no producer expert projection"):
        _scope(case)


def test_scope_without_a_projection_is_the_unchanged_pre_bridge_lane(case):
    # A carried projection is an unlock, not a new requirement: an allocation
    # from a campaign that never priced a packed population still exports its
    # routed units on their source-member shapes, exactly as before #183.
    for key in (PROJECTION_KEY, EXPERT_WIRES_KEY, STACK_FORMATS_KEY, WIRE_DIR_KEY):
        _meta(case).pop(key)
    _save(case)
    report = _scope(case)
    assert set(report["by_unit"]) == {DENSE, *_units()}
    assert "expert_projection" not in report


def test_the_receipt_names_which_path_produced_the_routed_bytes(case):
    # PrismaQuant #222.  The fallback above is legitimate and stays -- but it
    # was SILENT: an allocation with all four keys stripped re-encodes its
    # routed units from source, and nothing in the export's own output said the
    # bytes about to ship were not the bytes the campaign priced.  The two
    # paths are now distinguishable in the scope receipt, derived where the
    # choice is made, without changing what either path refuses or encodes.
    priced = _scope(case)
    assert priced[export.ROUTED_EXPERT_BYTES_KEY] == export.ROUTED_EXPERT_BYTES_PRICED_WIRES
    assert set(priced["expert_projection"]["units"]) == set(_units())

    # The INCOHERENT case is #220's refusal and must not weaken: priced wires
    # bound to no executed unit are still refused by name, never stamped as a
    # fallback and shipped.
    _meta(case).pop(PROJECTION_KEY)
    _save(case)
    with pytest.raises(export.TesseraExportLaneError,
                       match="no producer expert projection"):
        _scope(case)

    # All four gone: the pre-#183 lane, which still succeeds and still resolves
    # every unit exactly as the priced run did.  Only the receipt is different.
    for key in (EXPERT_WIRES_KEY, STACK_FORMATS_KEY, WIRE_DIR_KEY):
        _meta(case).pop(key)
    _save(case)
    fallback = _scope(case)
    assert fallback[export.ROUTED_EXPERT_BYTES_KEY] == export.ROUTED_EXPERT_BYTES_REENCODED
    assert "expert_projection" not in fallback
    assert fallback["by_unit"] == priced["by_unit"]


def test_an_export_that_ships_no_routed_bytes_is_not_stamped_as_a_re_encode(case):
    # The stamp names what produced the ROUTED bytes, so an export that ships
    # none must not read as the fallback -- a dense allocation re-encodes no
    # routed unit because it selects none.  True on both sides of the carried
    # projection, since neither run hands a routed unit to the exporter.
    for name in _units():
        case.payload[name] = "BF16"
        _meta(case)["tessera_serving_scope"]["by_unit"].pop(name)
    _save(case)
    assert _scope(case)[export.ROUTED_EXPERT_BYTES_KEY] == export.ROUTED_EXPERT_BYTES_NONE
    for key in (PROJECTION_KEY, EXPERT_WIRES_KEY, STACK_FORMATS_KEY, WIRE_DIR_KEY):
        _meta(case).pop(key)
    _save(case)
    assert _scope(case)[export.ROUTED_EXPERT_BYTES_KEY] == export.ROUTED_EXPERT_BYTES_NONE


def test_scope_without_a_projection_still_refuses_a_predicated_cell(case):
    # And the refusal that made #183 necessary stays: without the producer's
    # record, a source-member dimension does not attest a fused execution unit.
    for key in (PROJECTION_KEY, EXPERT_WIRES_KEY, STACK_FORMATS_KEY, WIRE_DIR_KEY):
        _meta(case).pop(key)
    _save(case)
    case.contract.write_text(json.dumps(_payload(
        predicates=[{"fact": "out_features", "op": "equals", "value": N}])))
    with pytest.raises(export.TesseraExportLaneError, match="predicate|projection"):
        _scope(case)


def test_scope_refuses_a_routed_unit_the_producer_did_not_project(case):
    other = f"{STACK}.5.w1"
    case.payload[other] = case.payload[f"{STACK}.0.w1"]
    _meta(case)["tessera_serving_scope"]["by_unit"][other] = _context("routed_moe")
    _save(case)
    # The header carries no such tensor either; the projection refusal comes
    # first only for a unit that IS in the source, so add it there too.
    header = json.dumps({
        name + ".weight": {"dtype": "BF16", "shape": [N, N],
                            "data_offsets": [index * N * N * 2, (index + 1) * N * N * 2]}
        for index, name in enumerate((DENSE, *_units(), other))
    }).encode()
    (case.model / SHARD).write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(export.TesseraExportLaneError, match=f"{other}.*projection"):
        _scope(case)


def test_scope_refuses_a_partly_selected_stack(case):
    case.payload[f"{STACK}.1.w2"] = "BF16"
    _meta(case)["tessera_serving_scope"]["by_unit"].pop(f"{STACK}.1.w2")
    _save(case)
    with pytest.raises(export.TesseraExportLaneError, match="stack whole|not.*selected"):
        _scope(case)


def test_scope_refuses_a_source_shard_the_producer_did_not_hash(case):
    tensor = f"{STACK}.0.w1.weight"
    source = _meta(case)[PROJECTION_KEY]["producer"]["source"]
    # A self-consistent producer roster -- the other shard IS one the producer
    # hashed -- so what is refused is the disagreement with the shard THIS
    # checkpoint holds the tensor in, not a malformed source identity.
    source["files"]["other.safetensors"] = "e" * 64
    source["tensors"][tensor] = "other.safetensors"
    _save(case)
    with pytest.raises(export.TesseraExportLaneError, match="other.safetensors|shard"):
        _scope(case)


def test_scope_refuses_a_stack_format_stamp_that_disagrees_with_the_selection(case):
    _meta(case)[STACK_FORMATS_KEY] = {STACK: "TESSERA_E4M3_K1_R512"}
    _save(case)
    with pytest.raises(export.TesseraExportLaneError, match="stack.*format|disagree"):
        _scope(case)


@pytest.mark.parametrize("damage", ["bytes", "missing", "receipt"])
def test_scope_checks_the_priced_bytes_against_their_receipts(case, damage):
    name = f"{STACK}.1.w3"
    path = case.wire_dir / case.receipts[name]["file"]
    if damage == "bytes":
        path.write_bytes(b"\0" * case.receipts[name]["blob_bytes"])
    elif damage == "missing":
        path.unlink()
    else:
        _meta(case)[EXPERT_WIRES_KEY].pop(name)
        _save(case)
    with pytest.raises(export.TesseraExportLaneError, match=f"{name}.*(receipt|wire)"):
        _scope(case)


def _isolate_other_gates(monkeypatch):
    monkeypatch.setattr(export, "require_declared_structure", lambda model: "routed_moe")
    monkeypatch.setattr(export, "require_executes_derived_from_contract", lambda: ())
    monkeypatch.setattr(export, "require_producer_tools", lambda: ())
    monkeypatch.setattr(export, "require_release_pin", lambda: None)
    monkeypatch.setattr(pin, "load_tessera_serving_runtime_pin",
                        lambda: SimpleNamespace(version="fixture", commit="f" * 40))


def _cli(case, tmp_path, *extra):
    return export.main([
        "--model", str(case.model), "--assignment", str(case.assignment),
        "--write-build-json", str(tmp_path / "build.json"),
        "--tessera-platform", "sm_121", "--tessera-runtime-image", IMAGE,
        "--tessera-execution-mode", "eager", "--tessera-residency", "resident",
        *extra,
    ])


def test_cli_writes_the_producers_cached_units_bundle_into_the_wire_dir(case, tmp_path,
                                                                        monkeypatch):
    cached_unit = pytest.importorskip(
        "tessera.cached_unit",
        reason="this checkout's tessera has no cached_unit bundle API (PrismaQuant #192)")
    _isolate_other_gates(monkeypatch)
    assert _cli(case, tmp_path, "--write-cached-expert-units") == 0
    build = json.loads((tmp_path / "build.json").read_text())
    manifest = Path(build["cached_expert_units"])
    assert manifest.parent == case.wire_dir
    bundle = json.loads(manifest.read_text())
    assert bundle["schema"] == cached_unit.CACHE_SCHEMA
    assert bundle["source"] == _carried()["producer"]["source"]
    assert bundle["units"] == case.receipts
    assert build["tessera_expert_stack_formats"] == {STACK: FMT}


def test_cli_refuses_to_bundle_without_the_producers_schema(case, tmp_path, monkeypatch,
                                                            capsys):
    _isolate_other_gates(monkeypatch)
    monkeypatch.setitem(sys.modules, "tessera.cached_unit", None)
    assert _cli(case, tmp_path, "--write-cached-expert-units") == 2
    assert "cached_unit" in capsys.readouterr().err
    assert not (tmp_path / "build.json").exists()
    assert not list(case.wire_dir.glob("*.json"))


def test_cli_writes_no_bundle_for_an_allocation_without_a_projection(case, tmp_path,
                                                                     monkeypatch):
    # The control below covers the STOCK-table shape: no projection metadata at
    # all.  It is not the shape a #220 allocator emits, which is why it never
    # caught #229 -- see the next test.
    _isolate_other_gates(monkeypatch)
    for name in _units():
        case.payload[name] = "BF16"
        _meta(case)["tessera_serving_scope"]["by_unit"].pop(name)
    for key in (PROJECTION_KEY, EXPERT_WIRES_KEY, STACK_FORMATS_KEY, WIRE_DIR_KEY):
        _meta(case).pop(key)
    _save(case)
    assert _cli(case, tmp_path, "--write-cached-expert-units") == 0
    build = json.loads((tmp_path / "build.json").read_text())
    assert "cached_expert_units" not in build
    assert not list(case.wire_dir.glob("*.json"))


def _allocator_metadata(case, assignment):
    """The projection block a REAL allocation carries, from the allocator's helper.

    ``allocation_expert_projection_block`` is what `allocator.main` stamps, and
    it is the producer side of the seam #229 was filed against: hand-built
    metadata cannot show that the allocator and the export lane disagree about
    what a retained projection means.  Shaped like the campaign cost table the
    allocator reads -- population, projection and wire directory in
    ``provenance``, the priced receipts keyed by rung at the top level.
    """
    meta = _meta(case)
    payload = {
        "provenance": {
            POPULATION_KEY: meta[POPULATION_KEY],
            PROJECTION_KEY: meta[PROJECTION_KEY],
            "wire_dir": str(case.wire_dir),
        },
        EXPERT_WIRES_KEY: {name: {FMT: receipt}
                           for name, receipt in case.receipts.items()},
    }
    return tep.allocation_expert_projection_block(payload, assignment)


def test_cli_bundles_nothing_when_the_allocation_keeps_every_expert_in_bf16(
        case, tmp_path, monkeypatch):
    # PrismaQuant #229 (P1).  Selecting Tessera for a dense Linear while every
    # routed expert stays BF16 is a decision the allocator is DESIGNED to
    # produce: `allocation_expert_projection_block` keeps the population and
    # the projection, records the stack as BF16, and carries no wire receipts.
    # The export lane then filtered the assignment to Tessera rows, found no
    # routed unit, and still handed the empty bundle to the writer -- so the
    # driver, which always passes --write-cached-expert-units, refused a valid
    # allocation with exit 2 and wrote no build anchor.
    _isolate_other_gates(monkeypatch)
    block = _allocator_metadata(
        case, {DENSE: FMT, **{name: "BF16" for name in _units()}})
    assert block[EXPERT_WIRES_KEY] == {}
    assert block[STACK_FORMATS_KEY] == {STACK: "BF16"}
    _meta(case).update(block)
    for name in _units():
        case.payload[name] = "BF16"
        _meta(case)["tessera_serving_scope"]["by_unit"].pop(name)
    _save(case)

    # #222's third value is what draws the line: a carried projection that no
    # SELECTED unit rides ships no routed byte, so there is nothing to bundle
    # and nothing was re-encoded either.
    scope = _scope(case)
    assert set(scope["by_unit"]) == {DENSE}
    assert scope[export.ROUTED_EXPERT_BYTES_KEY] == export.ROUTED_EXPERT_BYTES_NONE

    assert _cli(case, tmp_path, "--write-cached-expert-units") == 0
    build = json.loads((tmp_path / "build.json").read_text())
    assert "cached_expert_units" not in build
    assert not list(case.wire_dir.glob("*.json"))
    # The provenance the allocator carried is KEPT, not discarded to get here.
    assert build[export.BUILD_ROUTED_EXPERT_BYTES_KEY] == export.ROUTED_EXPERT_BYTES_NONE
    carried = json.loads(case.assignment.read_text())["__prismaquant__"]
    assert carried[PROJECTION_KEY] == _meta(case)[PROJECTION_KEY]
    assert carried[POPULATION_KEY] == _meta(case)[POPULATION_KEY]
    assert carried[STACK_FORMATS_KEY] == {STACK: "BF16"}


def test_the_build_anchor_carries_which_path_produced_the_routed_bytes(case, tmp_path,
                                                                       monkeypatch):
    # The build anchor is the only thing the CLI serialises, and
    # `lane_shipcard open --build-json` stamps it whole onto the artifact's
    # ship record -- so this is where a consumer of the SHIPPED artifact reads
    # which path produced its routed bytes (#222).  One derivation: the anchor
    # copies the scope receipt's answer, it does not recompute it.
    _isolate_other_gates(monkeypatch)
    assert _cli(case, tmp_path) == 0
    build = json.loads((tmp_path / "build.json").read_text())
    assert (build[export.BUILD_ROUTED_EXPERT_BYTES_KEY]
            == export.ROUTED_EXPERT_BYTES_PRICED_WIRES)
    assert build["tessera_expert_stack_formats"] == {STACK: FMT}
    for key in (PROJECTION_KEY, EXPERT_WIRES_KEY, STACK_FORMATS_KEY, WIRE_DIR_KEY):
        _meta(case).pop(key)
    _save(case)
    assert _cli(case, tmp_path) == 0
    build = json.loads((tmp_path / "build.json").read_text())
    assert (build[export.BUILD_ROUTED_EXPERT_BYTES_KEY]
            == export.ROUTED_EXPERT_BYTES_REENCODED)
    assert "tessera_expert_stack_formats" not in build


def test_shell_hands_the_exporter_the_bundle_the_preflight_wrote():
    driver = (Path(__file__).parents[1] / "prismaquant" / "run-pipeline.sh").read_text()
    translator = driver.index('python3 "${TESSERA_REPO%/}/experiments/plan_from_layer_config.py"')
    gate = driver.rfind("python3 -m prismaquant.tessera_export_lane", 0, translator)
    invocation = driver[gate:driver.index("; then", gate)]
    assert "--write-cached-expert-units" in invocation
    exporter = driver.index('python3 "${TESSERA_REPO%/}/experiments/export_tessera_serving.py"')
    encode = driver[exporter:driver.index("tee", exporter)]
    assert '"${TESSERA_CACHED_UNIT_ARGS[@]}"' in encode
    assert "--cached-expert-units" in driver[translator:exporter]
