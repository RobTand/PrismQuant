"""The real export endpoint binds selected units to their priced v5 scope."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

from prismaquant import tessera_export_lane as export
from prismaquant import tessera_serving_runtime_pin as pin
from prismaquant.pipeline import stage_settings_projection


IMAGE = "example/runtime@sha256:" + "a" * 64
OTHER_IMAGE = "example/other@sha256:" + "b" * 64
DENSE = "model.layers.0.self_attn.o_proj"
EXPERT = "model.layers.0.mlp.experts.0.down_proj"
FORMAT = "TESSERA_E4M3_K1_R1024"


@dataclass(frozen=True)
class Target:
    platform: str = "sm_121"
    runtime_image: str = IMAGE
    execution_mode: str = "eager"
    residency: str = "resident"

    def as_dict(self):
        return asdict(self)


def _context(structure="dense", target=None):
    return {**(target or Target()).as_dict(), "structure": structure}


def _payload():
    family = "TESSERA_E4M3_K1"
    cells = []
    for structure in ("dense", "routed_moe"):
        for regime in ("decode", "batch"):
            cells.append({
                "id": f"{structure}_{regime}", "platform": "sm_121",
                "family": family, "structure": structure, "regime": regime,
                "rungs_q256": [1024], "activation_contract": "fp8_per_token_dynamic",
                "route_status": "backed_with_serve_flag", "qualification": "device_qualified",
                "requires_plugin": "tessera", "predicates": [],
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


@pytest.fixture
def case(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "architectures": ["Qwen3MoeForCausalLM"],
        "num_experts": 2,
    }))
    # The endpoint reads headers only; no weight materialization is needed.
    header = json.dumps({
        name + ".weight": {"dtype": "BF16", "shape": [64, 64],
                            "data_offsets": [index * 8192, (index + 1) * 8192]}
        for index, name in enumerate((DENSE, EXPERT))
    }).encode()
    (model / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(_payload()))
    monkeypatch.setattr(export, "packaged_contract_path", lambda: contract)
    assignment = tmp_path / "layer_config.json"
    payload = {
        name: {"data_type": "tessera", "bits": 4, "tessera_format": FORMAT}
        for name in (DENSE, EXPERT)
    }
    payload["__prismaquant__"] = {"tessera_serving_scope": {
        "target": Target().as_dict(),
        "by_unit": {DENSE: _context(), EXPERT: _context("routed_moe")},
    }}
    assignment.write_text(json.dumps(payload))
    return SimpleNamespace(model=model, assignment=assignment, payload=payload,
                           contract=contract)


def _save(case):
    case.assignment.write_text(json.dumps(case.payload))


def _scope(case):
    return case.payload["__prismaquant__"]["tessera_serving_scope"]


def test_selected_export_units_resolve_their_own_structure(case):
    report = export.require_assignment_scope(case.model, case.assignment, target=Target())
    assert set(report["by_unit"]) == {DENSE, EXPERT}
    assert report["by_unit"][DENSE]["structure"] == "dense"
    assert report["by_unit"][EXPERT]["structure"] == "routed_moe"
    for route in report["by_unit"].values():
        assert route["route_status"] == "backed_with_serve_flag"
        assert {row["regime"] for row in route["regime_routes"]} == {"decode", "batch"}
        assert all(row["runtime_image"] == IMAGE for row in route["regime_routes"])


@pytest.mark.parametrize("field,value", [
    ("runtime_image", OTHER_IMAGE), ("execution_mode", "compiled"),
    ("residency", "streamed"), ("platform", "sm_120"),
])
def test_export_target_must_equal_the_allocation_target(case, field, value):
    with pytest.raises(export.TesseraExportLaneError, match="allocation.*target|target.*allocation"):
        export.require_assignment_scope(case.model, case.assignment,
                                        target=replace(Target(), **{field: value}))


@pytest.mark.parametrize("missing", ["scope", "target", "unit"])
def test_v5_export_refuses_missing_recorded_scope(case, missing):
    if missing == "scope":
        case.payload["__prismaquant__"].pop("tessera_serving_scope")
    elif missing == "target":
        _scope(case).pop("target")
    else:
        _scope(case)["by_unit"].pop(EXPERT)
    _save(case)
    with pytest.raises(export.TesseraExportLaneError, match="scope|context|target"):
        export.require_assignment_scope(case.model, case.assignment, target=Target())


def test_v5_export_refuses_an_unbound_target_even_when_allocation_is_bound(case):
    with pytest.raises(export.TesseraExportLaneError, match="explicit.*target|target.*required"):
        export.require_assignment_scope(case.model, case.assignment)


def test_export_rechecks_topology_instead_of_trusting_forged_dense_scope(case):
    _scope(case)["by_unit"][EXPERT]["structure"] = "dense"
    _save(case)
    with pytest.raises(export.TesseraExportLaneError, match="structure|context"):
        export.require_assignment_scope(case.model, case.assignment, target=Target())


def test_export_refuses_selected_source_tensor_that_does_not_exist(case):
    unknown = "model.layers.99.self_attn.o_proj"
    case.payload[unknown] = case.payload.pop(DENSE)
    _scope(case)["by_unit"][unknown] = _scope(case)["by_unit"].pop(DENSE)
    _save(case)
    with pytest.raises(export.TesseraExportLaneError, match="source|checkpoint|shape"):
        export.require_assignment_scope(case.model, case.assignment, target=Target())


def test_export_cannot_join_partial_regimes_from_another_runtime(case):
    payload = _payload()
    next(row for row in payload["lane_eligibility"]["cells"]
         if row["id"] == "routed_moe_batch")["runtime"]["image"] = OTHER_IMAGE
    case.contract.write_text(json.dumps(payload))
    with pytest.raises(export.TesseraExportLaneError, match="regime|unattested"):
        export.require_assignment_scope(case.model, case.assignment, target=Target())


def test_export_shape_predicates_use_checkpoint_dimensions(case):
    payload = _payload()
    for row in payload["lane_eligibility"]["cells"]:
        row["predicates"] = [{"fact": "in_features", "op": "equals", "value": 128}]
    case.contract.write_text(json.dumps(payload))
    with pytest.raises(export.TesseraExportLaneError, match="regime|unattested"):
        export.require_assignment_scope(case.model, case.assignment, target=Target())


def test_matching_shape_predicate_cannot_claim_the_producers_fused_unit(case):
    payload = _payload()
    for row in payload["lane_eligibility"]["cells"]:
        row["predicates"] = [{"fact": "out_features", "op": "equals", "value": 64}]
    case.contract.write_text(json.dumps(payload))
    with pytest.raises(export.TesseraExportLaneError, match="predicate|projection"):
        export.require_assignment_scope(case.model, case.assignment, target=Target())


def test_export_does_not_change_plain_bf16_assignments(case):
    case.payload[DENSE] = "BF16"
    _scope(case)["by_unit"].pop(DENSE)
    _save(case)
    report = export.require_assignment_scope(case.model, case.assignment, target=Target())
    assert set(report["by_unit"]) == {EXPERT}


def test_legacy_unscoped_export_keeps_its_existing_behavior(case):
    payload = _payload()
    payload["lane_eligibility"]["schema"] = "tessera.lane-eligibility.v4"
    for row in payload["lane_eligibility"]["cells"]:
        row.pop("runtime")
    case.contract.write_text(json.dumps(payload))
    case.payload.pop("__prismaquant__")
    _save(case)
    assert export.require_assignment_scope(case.model, case.assignment) is None


def _isolate_other_gates(monkeypatch):
    monkeypatch.setattr(export, "require_declared_structure", lambda model: "routed_moe")
    monkeypatch.setattr(export, "require_executes_derived_from_contract", lambda: ())
    monkeypatch.setattr(export, "require_producer_tools", lambda: ())
    monkeypatch.setattr(export, "require_release_pin", lambda: None)
    monkeypatch.setattr(pin, "load_tessera_serving_runtime_pin",
                        lambda: SimpleNamespace(version="fixture", commit="f" * 40))


def test_early_preflight_refuses_v5_without_an_explicit_target(case, monkeypatch):
    _isolate_other_gates(monkeypatch)
    with pytest.raises(export.TesseraExportLaneError, match="explicit.*target|target.*required"):
        export.preflight(case.model)


def test_later_preflight_calls_selected_unit_gate(case, monkeypatch):
    _isolate_other_gates(monkeypatch)
    _scope(case)["by_unit"][EXPERT]["structure"] = "dense"
    _save(case)
    with pytest.raises(export.TesseraExportLaneError, match="structure|context"):
        export.preflight(case.model, target=Target(), assignment_path=case.assignment)


def test_cli_passes_explicit_target_and_assignment_to_preflight(case, monkeypatch):
    calls = []

    def preflight(model, **kwargs):
        calls.append((model, kwargs))
        raise export.TesseraExportLaneError("fixture stop after observing boundary")

    monkeypatch.setattr(export, "preflight", preflight)
    assert export.main([
        "--model", str(case.model), "--assignment", str(case.assignment),
        "--tessera-platform", "sm_121", "--tessera-runtime-image", IMAGE,
        "--tessera-execution-mode", "eager", "--tessera-residency", "resident",
    ]) == 2
    assert len(calls) == 1
    assert calls[0][1]["target"].as_dict() == Target().as_dict()
    assert str(calls[0][1]["assignment_path"]) == str(case.assignment)


def test_shell_gates_selected_scope_before_the_external_translator():
    driver = (Path(__file__).parents[1] / "prismaquant" / "run-pipeline.sh").read_text()
    translator = driver.index('python3 "${TESSERA_REPO%/}/experiments/plan_from_layer_config.py"')
    gate = driver.rfind("python3 -m prismaquant.tessera_export_lane", 0, translator)
    invocation = driver[gate:driver.index("; then", gate)]
    assert '--assignment "${WORK_DIR}/artifacts/layer_config.json"' in invocation
    assert '"${TESSERA_SCOPE_ARGS[@]}"' in invocation
    for flag in ("platform", "runtime-image", "execution-mode", "residency"):
        assert f"--tessera-{flag}" in driver


@pytest.mark.parametrize("key,value", [
    ("TESSERA_PLATFORM", "sm_121"), ("TESSERA_RUNTIME_IMAGE", IMAGE),
    ("TESSERA_EXECUTION_MODE", "eager"), ("TESSERA_RESIDENCY", "resident"),
])
def test_tessera_plan_cache_identity_changes_with_serving_target(key, value):
    legacy = {"MODEL_PATH": "model", "TESSERA_PLAN_COVER": "as-allocated"}
    before, missing = stage_settings_projection("tessera-plan", legacy)
    assert missing == []
    after, _ = stage_settings_projection("tessera-plan", {**legacy, key: value})
    assert after != before
    assert value in after.values()


def _run_head_policy(monkeypatch, capsys, *, body=FORMAT, head="BF16",
                     scoped=True, serialisable=True, cost_override="cost.pkl"):
    """Execute the driver's real inline policy under CPU-only mocked formats."""
    from prismaquant import format_registry as fr
    from prismaquant import tessera_render
    from prismaquant import model_profiles
    from prismaquant.model_profiles.qwen3 import Qwen3Profile

    monkeypatch.setattr(model_profiles, "detect_profile", lambda _: Qwen3Profile())
    for key, value in {
        "PQ_MODEL_PATH": "fixture", "PQ_ALLOW_PINNED": "",
        "PQ_LM_HEAD_FORMAT": head, "PQ_BODY_FORMATS": body,
        "PQ_COST_PATH_OVERRIDE": cost_override,
    }.items():
        monkeypatch.setenv(key, value)

    def get_format(value):
        if value == "TESSERA":
            raise KeyError("menu token is not a shape-free format")
        return SimpleNamespace(name=value)

    monkeypatch.setattr(fr, "get_format", get_format)
    monkeypatch.setattr(fr, "format_is_producer_eligible", lambda name, **kwargs:
                        name == "BF16" or bool(kwargs.get("context_by_unit")))
    monkeypatch.setattr(tessera_render, "tessera_rung_is_serialisable", lambda _: serialisable)
    args = ["fixture", "--target-profile", "tessera_research_sm121"]
    if scoped:
        args += ["--tessera-platform", "sm_121", "--tessera-runtime-image", IMAGE,
                 "--tessera-execution-mode", "eager", "--tessera-residency", "resident"]
    monkeypatch.setattr(sys, "argv", args)
    driver = (Path(__file__).parents[1] / "prismaquant" / "run-pipeline.sh").read_text()
    block = driver[driver.index('if ! LM_HEAD_POLICY_TEXT="$('):]
    program = block.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    exec(compile(program, "run-pipeline.sh:head-policy", "exec"), {"__name__": "__main__"})
    return capsys.readouterr().out.splitlines()


def test_actual_shell_body_defers_scoped_tessera_admission_to_per_unit_gate(monkeypatch, capsys):
    lines = _run_head_policy(monkeypatch, capsys)
    assert lines[0] == "BF16"
    assert lines[4] == FORMAT


def test_actual_shell_head_uses_its_explicit_context(monkeypatch, capsys):
    lines = _run_head_policy(monkeypatch, capsys, body="BF16", head=FORMAT)
    assert lines[0] == FORMAT
    assert lines[4] == "BF16," + FORMAT


def test_actual_shell_preserves_the_prepriced_tessera_menu_token(monkeypatch, capsys):
    lines = _run_head_policy(monkeypatch, capsys, body="TESSERA,BF16")
    assert lines[4] == "TESSERA,BF16"


@pytest.mark.parametrize("kwargs", [
    {"scoped": False}, {"serialisable": False},
    {"body": "TESSERA", "cost_override": ""},
])
def test_actual_shell_keeps_unbound_unwritable_and_uncosted_refusals(monkeypatch, capsys, kwargs):
    with pytest.raises(SystemExit) as error:
        _run_head_policy(monkeypatch, capsys, **kwargs)
    assert error.value.code == 2


def test_export_cli_writes_existing_build_anchor_from_exact_raw_assignment(case, tmp_path, monkeypatch):
    _isolate_other_gates(monkeypatch)
    raw = json.dumps(case.payload, indent=3) + "\n\n"
    case.assignment.write_text(raw)
    output = tmp_path / "tessera_build.json"
    assert export.main([
        "--model", str(case.model), "--assignment", str(case.assignment),
        "--tessera-platform", "sm_121", "--tessera-runtime-image", IMAGE,
        "--tessera-execution-mode", "eager", "--tessera-residency", "resident",
        "--write-build-json", str(output),
    ]) == 0
    build = json.loads(output.read_text())
    assert build["layer_config_sha"] == hashlib.sha256(raw.encode()).hexdigest()
    assert build["tessera_serving_scope"] == _scope(case)
    assert build["layer_config"] == str(case.assignment)
    assert build["source_model"] == str(case.model)


def test_export_build_anchor_requires_an_assignment(case, tmp_path):
    output = tmp_path / "tessera_build.json"
    assert export.main(["--model", str(case.model), "--write-build-json", str(output)]) == 2
    assert not output.exists()


def test_shell_passes_the_bound_build_json_to_lane_shipcard():
    driver = (Path(__file__).parents[1] / "prismaquant" / "run-pipeline.sh").read_text()
    assert '--write-build-json "$TESSERA_BUILD_JSON"' in driver
    opening = driver[driver.index("python3 -m prismaquant.lane_shipcard open"):]
    assert '--build-json "$TESSERA_BUILD_JSON"' in opening.split("; then", 1)[0]
