"""CPU-only contract for the direct validation-only export handoff."""
from __future__ import annotations

import json
from pathlib import Path
import pickle

import pytest

import prismaquant.rtx4090_validation_export as direct


def _handoff(tmp_path: Path) -> dict:
    return {
        "schema": direct.DIRECT_VALIDATION_EXPORT_SCHEMA,
        "model_dir": str(tmp_path / "source"),
        "layer_config": str(tmp_path / "layer_config.json"),
        "col_weights": str(tmp_path / "cb_col_weights.pkl"),
        "runtime_contract": str(tmp_path / "runtime_contract.json"),
        "out_dir": str(tmp_path / "exported"),
        "budget_bytes": 18_000_000_000,
    }


def test_direct_command_rerenders_selected_assignment_without_pipeline(tmp_path):
    command, environment = direct.build_direct_validation_export_command(
        _handoff(tmp_path), python_executable="/known/python"
    )

    assert command[:3] == [
        "/known/python", "-m", "prismaquant.export_nvfp4_cb_streaming"
    ]
    assert "run-pipeline.sh" not in " ".join(command)
    assert "production_weight_cache" not in " ".join(command)
    assert command[command.index("--layer-config") + 1].endswith(
        "layer_config.json"
    )
    assert command[command.index("--col-weights") + 1].endswith(
        "cb_col_weights.pkl"
    )
    assert command[command.index("--producer-policy") + 1] == (
        "qwen38_27b_rtx4090_fp8_cb_validation_only"
    )
    assert command[command.index("--codebook-source") + 1] == "lattice"
    assert command[command.index("--scale-coding") + 1] == "v1"
    assert "--allow-unbacked-route" not in command
    assert "--allow-unstamped-research" not in command
    assert environment["CB_CODEBOOK_SOURCE_SCOPE"] == "none"
    assert environment["CB_ACTIVATION_SCOPE"] == "none"
    assert environment["PRISMAQUANT_CB_LDLQ"] == "0"


def _preflight_files(tmp_path: Path):
    model = tmp_path / "source"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    recipe = tmp_path / "layer_config.json"
    recipe.write_text(json.dumps({
        "model.layers.0.mlp.down_proj": "FP8_CB_K20",
        "__prismaquant__": {"cb_render_identity": {"schema": "identity"}},
    }), encoding="utf-8")
    col = tmp_path / "cb_col_weights.pkl"
    with col.open("wb") as handle:
        pickle.dump({"model.layers.0.mlp.down_proj": [1.0, 2.0]}, handle)
    contract = tmp_path / "runtime_contract.json"
    contract.write_text("{}", encoding="utf-8")
    return model, recipe, col, contract


def _patch_preflight_policy(monkeypatch, *, budget_bytes=18_000_000_000):
    monkeypatch.setattr(direct, "validate_qwen38_dense_config", lambda *a, **k: {})
    monkeypatch.setattr(
        direct,
        "load_assignment",
        lambda _path: {"model.layers.0.mlp.down_proj": "FP8_CB_K20"},
    )
    monkeypatch.setattr(
        direct,
        "validate_rtx4090_assignment",
        lambda assignment, **_kwargs: dict(assignment),
    )
    monkeypatch.setattr(
        direct,
        "whole_artifact_budget_from_assignment_payload",
        lambda *_args, **_kwargs: {"budget_bytes": budget_bytes},
    )
    monkeypatch.setattr(
        direct,
        "require_rtx4090_compile_only_runtime_contract",
        lambda *_args, **_kwargs: {"runtime_contract_sha256": "a" * 64},
    )


def test_preflight_binds_exact_col_weights_and_selected_format(monkeypatch, tmp_path):
    model, recipe, col, contract = _preflight_files(tmp_path)
    _patch_preflight_policy(monkeypatch)
    observed = {}

    def validate_identity(identity, **kwargs):
        observed["identity"] = identity
        observed.update(kwargs)

    monkeypatch.setattr(
        direct, "validate_cb_render_identity_metadata", validate_identity
    )
    handoff = direct.preflight_direct_validation_export(
        model_dir=model,
        layer_config=recipe,
        col_weights=col,
        runtime_contract=contract,
        out_dir=tmp_path / "exported",
    )

    assert handoff["budget_bytes"] == 18_000_000_000
    assert handoff["selected_fp8_cb_units"] == 1
    assert observed["expected_formats_by_qname"] == {
        "model.layers.0.mlp.down_proj": ("FP8_CB_K20",)
    }
    assert observed["require_source_complete"] is True
    assert set(observed["col_weights"]) == {
        "model.layers.0.mlp.down_proj"
    }


def test_preflight_refuses_budget_above_18_decimal_gb(monkeypatch, tmp_path):
    model, recipe, col, contract = _preflight_files(tmp_path)
    _patch_preflight_policy(monkeypatch, budget_bytes=18_000_000_001)
    monkeypatch.setattr(
        direct, "validate_cb_render_identity_metadata", lambda *a, **k: None
    )

    with pytest.raises(
        direct.RTX4090DirectValidationExportError,
        match="no greater than 18000000000",
    ):
        direct.preflight_direct_validation_export(
            model_dir=model,
            layer_config=recipe,
            col_weights=col,
            runtime_contract=contract,
            out_dir=tmp_path / "exported",
        )


def test_shell_wrapper_is_direct_and_never_calls_stock_pipeline():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "export_qwen38_rtx4090_fp8_cb_validation_only.sh"
    ).read_text(encoding="utf-8")
    assert "prismaquant.rtx4090_validation_export" in script
    assert "run-pipeline.sh" not in script
    assert "LAYER_CONFIG" in script
    assert "CB_COL_WEIGHTS" in script

