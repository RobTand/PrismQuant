"""Producer-shaped census v2 remains bound to price, artifact and actual runtime."""
from copy import deepcopy
import hashlib
import json

import pytest

from prismaquant import lane_eligibility as lane
from prismaquant import shipcard
from prismaquant import tessera_route_receipt as receipt


IMAGE = "example/runtime@sha256:" + "a" * 64
FORMAT = "TESSERA_E4M3_K1_R1024"
FAMILY = "TESSERA_E4M3_K1"
OWNER = "model.layers.0.feed_forward.experts"


def _text(value):
    return json.dumps(value, indent=2) + "\n"


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture(tmp_path, monkeypatch, *, structure="routed_moe"):
    owner = OWNER if structure == "routed_moe" else "model.layers.0.self_attn.o_proj"
    target = {"platform": "sm_121", "runtime_image": IMAGE,
              "execution_mode": "eager", "residency": "resident"}
    context = {**target, "structure": structure}
    roles = []
    groups = {"w13": {"rows": 128, "columns": 128, "q256": 1024,
                      "roles": [["gate_proj", 64], ["up_proj", 64]], "wire_stride": 4096},
              "w2": {"rows": 128, "columns": 64, "q256": 1024,
                     "roles": [["down_proj", 128]], "wire_stride": 4096}}
    if structure == "routed_moe":
        for expert in range(2):
            for group, row in groups.items():
                for name, rows in row["roles"]:
                    source = f"{owner}.{expert}.{name}.weight"
                    roles.append({"tensor": source, "source_tensor": source,
                                  "source_layout": "unpacked_per_expert",
                                  "source_slice": {"expert": expert, "selector": "whole", "transpose": False},
                                  "expert": expert, "group": group, "role": name,
                                  "rows": rows, "cols": row["columns"], "q256": 1024,
                                  "grid": "E4M3", "family": "TESSERA_FP8"})
        scheme = {"family": "TESSERA_FP8", "structure": structure, "grid": "E4M3",
                  "body": "WINDOW", "plane": "CHANNEL", "experts": 2, "groups": groups}
        symbol, decoder, kind = "vllm.fused_moe.modular_kernel:TRITON", "torch_materialize_stock", "moe"
    else:
        roles = [{"tensor": owner + ".weight", "role": "o_proj", "rows": 128,
                  "cols": 128, "q256": 1024, "grid": "E4M3", "family": "TESSERA_FP8"}]
        scheme = {"family": "TESSERA_FP8", "structure": "dense", "grid": "E4M3",
                  "body": "WINDOW", "plane": "CHANNEL", "q256": 1024,
                  "rows": 128, "columns": 128, "roles": [["o_proj", 128]], "wire_bytes": 8192}
        symbol, decoder, kind = "torch._scaled_mm", "torch_window", "dense"
    units = [role["tensor"].removesuffix(".weight") for role in roles]
    scope = {"target": target, "by_unit": {unit: context for unit in units}}
    scope["by_unit"]["model.embed_tokens"] = {**target, "structure": "dense"}
    assignment = {unit: {"data_type": "tessera", "tessera_format": FORMAT} for unit in units}
    assignment["model.embed_tokens"] = "BF16"
    assignment["__prismaquant__"] = {"tessera_serving_scope": scope}
    config = {"architectures": ["Lfm2MoeForCausalLM"], "model_type": "lfm2_moe",
              "quantization_config": {"quant_method": "tessera", "config_groups": {
                  "group_0": {"format": "TESSERA", "targets": [owner], "scheme": scheme}}}}
    manifest = {"modules": {owner: {"structure": structure, "family": "TESSERA_FP8",
                                   "grid": "E4M3", "q256": 1024, "roles": roles}}}
    if structure == "routed_moe":
        manifest["modules"][owner]["experts"] = 2
    binding = {"layer_config_json": _text(assignment), "config_json": _text(config),
               "manifest_json": _text(manifest)}
    build = {"layer_config_sha": _sha(binding["layer_config_json"]), "tessera_serving_scope": scope}
    model_dir = tmp_path / structure
    model_dir.mkdir()
    (model_dir / "config.json").write_text(binding["config_json"])
    (model_dir / "tessera_serving_manifest.json").write_text(binding["manifest_json"])
    (model_dir / "model.safetensors").write_bytes(b"fixture-weight-bytes")
    module = owner + ".routed_experts" if kind == "moe" else owner
    census = {"schema": "tessera.serving.route_census/2", "checkpoint": "/model",
              "runtime": {"image": IMAGE, "execution_mode": "eager"}, "compiled": False,
              "env": {"TESSERA_SERVE_MODE": "resident"}, "device": {"capability": [12, 1]},
              "verdict": "served", "problems": [], "declared_name_mapping": None,
              "declared_names_mapped_to_module_space": False,
              "checkpoint_sidecars": {"config.json": _sha(binding["config_json"]),
                                      "tessera_serving_manifest.json": _sha(binding["manifest_json"])},
              "records": {}, "record_owner": {}}
    for phase, m in (("decode", 1), ("prefill", 64)):
        census["records"][phase] = {module: {"kind": kind, "policy": "TESSERA_FP8:resident",
                                            "symbol": symbol, "decoder": decoder,
                                            "contract": "fp8_per_token_dynamic", "state": "served",
                                            "shape": f"M{m}:N128:K128"}}
        census["record_owner"][phase] = {module: owner}
    formats = {FAMILY: {"family": FAMILY, "kind": "tessera_wire", "grid": "E4M3",
                        "name_pattern": "TESSERA_E4M3_K1_R{k}", "reader_rate_range_q256": [256, 2048],
                        "residency_modes": ["resident", "streamed"]}}
    cells = [{"id": f"{structure}_{regime}", "platform": "sm_121", "family": FAMILY,
              "structure": structure, "regime": regime, "rungs_q256": [1024],
              "activation_contract": "fp8_per_token_dynamic", "route_status": "backed_with_serve_flag",
              "qualification": "device_qualified", "requires_plugin": "tessera", "predicates": [],
              "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
              "executes": [{"symbol": symbol.split(":", 1)[0], "decoder": decoder}],
              "runtime": {"image": IMAGE, "execution_modes": ["eager"],
                          "vllm": "0.0.0+fixture", "torch": "0.0.0+fixture"},
              # The smallest evidence block the CURRENT lane grammar (v8)
              # reads; this fixture is about the census binding, not the
              # evidence, and a not_recorded smoke with no control and no
              # artifact is the honest empty value of each field.
              "evidence": {"grade": "route_only", "kl": [],
                           "smoke": {"status": "not_recorded", "receipt": None,
                                     "attribution": "unattributed", "control": None},
                           "artifact": None}}
             for regime in ("decode", "batch")]
    block = {"schema": lane.LANE_ELIGIBILITY_SCHEMA_TESSERA, "platforms": {"sm_121": {}},
             "structures": [structure], "regimes": ["decode", "batch"], "cells": cells}
    # No extension launch in this fixture (torch / vLLM symbols only), so no
    # lane row is needed for the reader to bind launches to.
    table = lane._parse_table(block, list(formats.values()), "fixture", "fixture", "fixture",
                              native_extensions=[])
    monkeypatch.setattr(receipt, "_current_scoped_contract", lambda: (table, formats), raising=False)
    return census, binding, build, model_dir, units


def make(data):
    census, binding, build, model_dir, _units = data
    return shipcard.make_route_census_record(tool="test", model_sha=shipcard.compute_model_sha(model_dir),
        route_records=census, priced_routes=[], substitute_decoders=[],
        binding=binding, build=build, model_dir=model_dir)


@pytest.mark.parametrize("structure", ["dense", "routed_moe"])
def test_raw_v2_positive_retains_runtime_phase_owner_and_exact_price(tmp_path, monkeypatch, structure):
    data = fixture(tmp_path, monkeypatch, structure=structure)
    record = make(data)
    assert record["passed"] is True
    assert record["route_census"] == data[0]
    assert record["census_binding"] == data[1]
    assert set(record["scoped_verdict"]["by_unit"]) == set(data[4])
    assert record["scoped_verdict"]["target"] == data[2]["tessera_serving_scope"]["target"]
    if structure == "routed_moe":
        assert record["served_decoders"] == ["torch_materialize_stock"]
        assert ":TRITON" in next(iter(record["route_census"]["records"]["decode"].values()))["symbol"]
    assert shipcard._verify_route_census_record("route.census", record,
        card={"build": data[2]}, model_dir=data[3]) == []


@pytest.mark.parametrize("mutation", ["missing_image", "wrong_image", "wrong_mode", "compiled_disagrees",
    "wrong_residency", "wrong_platform", "missing_phase", "extra_phase", "missing_owner", "extra_owner",
    "forged_owner", "wrong_kind", "wrong_policy", "wrong_decoder", "wrong_symbol", "wrong_activation",
    "wrong_state", "missing_seal", "wrong_seal", "failed_verdict", "malformed_shape", "wrong_regime",
    "nonidentity_mapping"])
def test_raw_v2_refuses_unbound_or_contradictory_observations(tmp_path, monkeypatch, mutation):
    data = fixture(tmp_path, monkeypatch)
    raw = data[0]
    row = next(iter(raw["records"]["decode"].values()))
    if mutation == "missing_image": del raw["runtime"]["image"]
    elif mutation == "wrong_image": raw["runtime"]["image"] = IMAGE.replace("a" * 64, "b" * 64)
    elif mutation == "wrong_mode": raw["runtime"]["execution_mode"] = "compiled"
    elif mutation == "compiled_disagrees": raw["compiled"] = True
    elif mutation == "wrong_residency": raw["env"]["TESSERA_SERVE_MODE"] = "streamed"
    elif mutation == "wrong_platform": raw["device"]["capability"] = [9, 0]
    elif mutation == "missing_phase": del raw["records"]["prefill"]
    elif mutation == "extra_phase": raw["records"]["unknown"] = {}
    elif mutation == "missing_owner": raw["records"]["decode"] = {}
    elif mutation == "extra_owner": raw["records"]["decode"]["unexpected"] = deepcopy(row)
    elif mutation == "forged_owner": raw["record_owner"]["decode"][OWNER + ".routed_experts"] = "forged"
    elif mutation == "wrong_kind": row["kind"] = "fp8"
    elif mutation == "wrong_policy": row["policy"] = "TESSERA_FP8:streamed"
    elif mutation == "wrong_decoder": row["decoder"] = "unpublished"
    elif mutation == "wrong_symbol": row["symbol"] = "torch.mm"
    elif mutation == "wrong_activation": row["contract"] = "other"
    elif mutation == "wrong_state": row["state"] = "prepared"
    elif mutation == "missing_seal": del raw["checkpoint_sidecars"]
    elif mutation == "wrong_seal": raw["checkpoint_sidecars"]["config.json"] = "0" * 64
    elif mutation == "failed_verdict": raw["verdict"] = "REFUSED"
    elif mutation == "malformed_shape": row["shape"] = "1:128:128"
    elif mutation == "wrong_regime": row["shape"] = "M2:N128:K128"
    elif mutation == "nonidentity_mapping":
        raw["declared_name_mapping"] = {OWNER: "new_owner"}
        raw["declared_names_mapped_to_module_space"] = True
    with pytest.raises(receipt.TesseraRouteReceiptError): make(data)


@pytest.mark.parametrize("mutation", ["recipe", "target", "structure", "projection", "price_rung",
                                      "missing_build", "sidecar_on_disk"])
def test_independent_build_and_artifact_anchors_reject_rewritten_binding(tmp_path, monkeypatch, mutation):
    data = fixture(tmp_path, monkeypatch)
    census, binding, build, model_dir, units = data
    if mutation == "recipe": binding["layer_config_json"] += " "
    elif mutation == "target": build["tessera_serving_scope"] = {"target": {}, "by_unit": {}}
    elif mutation in {"structure", "price_rung"}:
        recipe = json.loads(binding["layer_config_json"])
        if mutation == "structure": recipe["__prismaquant__"]["tessera_serving_scope"]["by_unit"][units[0]]["structure"] = "dense"
        else: recipe[units[0]] = {"data_type": "tessera", "tessera_format": "TESSERA_E4M3_K1_R1025"}
        binding["layer_config_json"] = _text(recipe)
        build["layer_config_sha"] = _sha(binding["layer_config_json"])
        build["tessera_serving_scope"] = recipe["__prismaquant__"]["tessera_serving_scope"]
    elif mutation == "projection":
        manifest = json.loads(binding["manifest_json"])
        manifest["modules"][OWNER]["roles"][0]["source_tensor"] = "forged.weight"
        binding["manifest_json"] = _text(manifest)
        census["checkpoint_sidecars"]["tessera_serving_manifest.json"] = _sha(binding["manifest_json"])
        (model_dir / "tessera_serving_manifest.json").write_text(binding["manifest_json"])
    elif mutation == "missing_build": build.clear()
    elif mutation == "sidecar_on_disk": (model_dir / "config.json").write_text("{}")
    with pytest.raises(receipt.TesseraRouteReceiptError): make(data)


def test_offline_replay_rejects_edited_scope_raw_record_and_stale_verdict(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    record = make(data)
    record["route_census"]["runtime"]["image"] = "example/other@sha256:" + "b" * 64
    record["scoped_verdict"]["target"]["runtime_image"] = record["route_census"]["runtime"]["image"]
    assert shipcard._verify_route_census_record("route.census", record,
        card={"build": data[2]}, model_dir=data[3])


def test_dense_undeclared_materialized_fallback_is_not_moe_permission(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch, structure="dense")
    for rows in data[0]["records"].values(): next(iter(rows.values()))["decoder"] = "torch_materialize_stock"
    with pytest.raises(receipt.TesseraRouteReceiptError): make(data)


def test_legacy_flat_parser_never_discards_explicit_runtime_context():
    with pytest.raises(receipt.TesseraRouteReceiptError, match="scope|runtime"):
        receipt.parse_route_records([{"route": "TESSERA_FP8", "decoder": "torch_window",
                                      "runtime_image": IMAGE, "execution_mode": "eager"}])


def test_scoped_card_cannot_replay_flat_legacy_receipt(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    # A historical flat receipt: filled where no current scoped table existed
    # to refuse it (fill applies the same rule as verify, #214), then carried
    # onto a scoped card.
    with monkeypatch.context() as historical:
        def absent():
            raise ModuleNotFoundError("No module named 'tessera'", name="tessera")
        historical.setattr(receipt, "_current_scoped_contract", absent)
        legacy = shipcard.make_route_census_record(tool="legacy", model_sha="fixture",
            priced_routes=["TESSERA_FP8"], route_records=[{"route": "TESSERA_FP8", "decoder": "native"}],
            substitute_decoders=["fallback"])
    assert shipcard._verify_route_census_record("route.census", legacy, card={"build": data[2]})
    # Filling that same flat list where the scoped table IS current refuses
    # by name, before any card sees it.
    with pytest.raises(receipt.TesseraRouteReceiptError, match="cannot attest an unbound legacy flat census"):
        shipcard.make_route_census_record(tool="legacy", model_sha="fixture",
            priced_routes=["TESSERA_FP8"], route_records=[{"route": "TESSERA_FP8", "decoder": "native"}],
            substitute_decoders=["fallback"])


def test_scoped_verification_requires_independent_artifact_files(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    record = make(data)
    assert shipcard._verify_route_census_record("route.census", record, card={"build": data[2]})


def test_missing_packaged_contract_is_typed_refusal(tmp_path, monkeypatch):
    data = fixture(tmp_path, monkeypatch)
    def unavailable():
        raise ModuleNotFoundError("no packaged Tessera")
    monkeypatch.setattr(receipt, "_current_scoped_contract", unavailable)
    with pytest.raises(receipt.TesseraRouteReceiptError, match="packaged Tessera"):
        make(data)


@pytest.mark.parametrize("missing_recipe", [False, True])
def test_cli_retains_full_v2_and_replays_against_card_and_model(tmp_path, monkeypatch, missing_recipe):
    from prismaquant.shipcard_cli import main
    data = fixture(tmp_path, monkeypatch)
    census, binding, build, model_dir, _units = data
    card_path = model_dir / "shipcard.json"
    shipcard.write_shipcard(card_path, shipcard.build_shipcard(model_dir, build=build, lane="tessera"))
    census_path = tmp_path / "census.json"
    census_path.write_text(_text(census))
    recipe_path = tmp_path / "layer_config.json"
    recipe_path.write_text(binding["layer_config_json"])
    args = ["fill-route-census", str(card_path), "--census", str(census_path), "--model-dir", str(model_dir)]
    if not missing_recipe:
        args += ["--layer-config", str(recipe_path)]
    assert main(args) == (2 if missing_recipe else 0)
    card = shipcard.load_shipcard(card_path)
    if missing_recipe:
        assert card["slots"]["route.census"] is None
    else:
        assert card["slots"]["route.census"]["route_census"] == census
        assert not [p for p in shipcard.verify(card, model_dir=model_dir) if p.startswith("route.census:")]
        card["build"]["layer_config_sha"] = "0" * 64
        assert any(p.startswith("route.census:") for p in shipcard.verify(card, model_dir=model_dir))


def test_duplicate_json_keys_do_not_replace_price_units():
    with pytest.raises(receipt.TesseraRouteReceiptError, match="duplicate"):
        receipt.parse_census_json('{"unit": "BF16", "unit": "BF16"}', where="fixture")


def test_unknown_raw_schema_and_legacy_table_are_not_v5_evidence(tmp_path, monkeypatch):
    from dataclasses import replace
    data = fixture(tmp_path, monkeypatch)
    data[0]["schema"] = "tessera.serving.route_census/unknown"
    with pytest.raises(receipt.TesseraRouteReceiptError, match="schema"):
        make(data)
    data[0]["schema"] = receipt.SCOPED_CENSUS_SCHEMA
    table, formats = receipt._current_scoped_contract()
    monkeypatch.setattr(receipt, "_current_scoped_contract", lambda: (replace(table, schema=lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V4), formats))
    with pytest.raises(receipt.TesseraRouteReceiptError, match="v5|legacy"):
        make(data)


def test_member_dimensions_cannot_attest_executed_shape_predicates(tmp_path, monkeypatch):
    from dataclasses import replace
    data = fixture(tmp_path, monkeypatch, structure="dense")
    table, formats = receipt._current_scoped_contract()
    cells = tuple(replace(cell, predicates=(("out_features", "equals", 128),)) for cell in table.cells)
    monkeypatch.setattr(receipt, "_current_scoped_contract", lambda: (replace(table, cells=cells), formats))
    with pytest.raises(receipt.TesseraRouteReceiptError, match="predicate|executed"):
        make(data)


@pytest.mark.parametrize("structure,combined", [("routed_moe", False), ("routed_moe", True), ("dense", False)])
def test_compiled_scope_needs_single_attributable_moe_launch(tmp_path, monkeypatch, structure, combined):
    from dataclasses import replace
    data = fixture(tmp_path, monkeypatch, structure=structure)
    raw, binding, build, _model_dir, _units = data
    recipe = json.loads(binding["layer_config_json"])
    scope = recipe["__prismaquant__"]["tessera_serving_scope"]
    scope["target"]["execution_mode"] = "compiled"
    for context in scope["by_unit"].values(): context["execution_mode"] = "compiled"
    build["tessera_serving_scope"] = scope
    binding["layer_config_json"] = _text(recipe)
    build["layer_config_sha"] = _sha(binding["layer_config_json"])
    raw["runtime"]["execution_mode"], raw["compiled"] = "compiled", True
    for rows in raw["records"].values():
        for row in rows.values():
            row["shape"] = "M*:N128:K128"
            if combined: row["symbol"] = "torch.bmm+vllm.fused_moe.modular_kernel:TRITON"
    table, formats = receipt._current_scoped_contract()
    cells = tuple(replace(cell, execution_modes=("eager", "compiled")) for cell in table.cells)
    monkeypatch.setattr(receipt, "_current_scoped_contract", lambda: (replace(table, cells=cells), formats))
    if structure == "dense" or combined:
        with pytest.raises(receipt.TesseraRouteReceiptError): make(data)
    else:
        record = make(data)
        assert record["passed"] is True
        assert record["scoped_verdict"]["target"]["execution_mode"] == "compiled"
