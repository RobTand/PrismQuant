from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import cb_banked_books as bank
from prismaquant import cb_learned_bundle as learned
from prismaquant import nvfp4_cb_formats as cb
from prismaquant.build_cb_learned_bundle import build_bundle_from_model
from prismaquant.cb_layout import codebook_subtable_shapes, family_for
from prismaquant.cb_warm_state import tensor_value_identity
from prismaquant.export_nvfp4_cb_streaming import _LazySkeleton


@pytest.fixture(autouse=True)
def _pretend_gridbook_supports_routed_lut(monkeypatch):
    """Supply the routed per-role LUT capability explicitly.

    The banked builder writes routed-MoE learned cells, gated on
    ``GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION``. The shipped pin
    is the released 0.8.2, which correctly refuses that path — no released
    Gridbook carries the ABI yet. Reading the shipped pin here would let a pin
    move decide whether the builder's own tests run their subject.
    """
    from prismaquant import gridbook_runtime_pin as runtime_pin

    supported = runtime_pin.parse_gridbook_runtime_pin({
        "schema": runtime_pin.GRIDBOOK_RUNTIME_PIN_SCHEMA,
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "a" * 40,
        "version": (
            runtime_pin.GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION),
        "version_is_release": False,
    })
    monkeypatch.setattr(
        learned, "load_gridbook_runtime_pin", lambda: supported)


_LAYER = 7
_RUNG = 28
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_SOURCE_LEAF = {
    "gate_proj": "w1",
    "up_proj": "w3",
    "down_proj": "w2",
}


def _pool_sha(tables: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for table in tables:
        digest.update(
            table.to(torch.float32).numpy().astype("<f4", copy=False).tobytes()
        )
    return digest.hexdigest()


def _role_book(projection: str) -> tuple[torch.Tensor, ...]:
    family = family_for("fp8", "product")
    shapes = codebook_subtable_shapes(_RUNG, family.mode, family.n_sub)
    role_index = _PROJECTIONS.index(projection)
    tables = []
    for subtable_index, shape in enumerate(shapes):
        raw = torch.linspace(
            -1.0 + 0.03125 * (role_index + subtable_index),
            1.0 - 0.03125 * role_index,
            steps=shape[0] * shape[1],
            dtype=torch.float32,
        ).reshape(shape)
        tables.append(
            cb._snap_to_grid(raw, "fp8", positive=False).to(torch.float16)
        )
    return tuple(tables)


def _write_model(model_dir: Path) -> dict[str, torch.Tensor]:
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
    }))
    tensors: dict[str, torch.Tensor] = {}
    for expert in range(2):
        for projection in _PROJECTIONS:
            leaf = _SOURCE_LEAF[projection]
            role_index = _PROJECTIONS.index(projection)
            value = torch.arange(256 * 256, dtype=torch.float32).reshape(256, 256)
            value = (
                value / 65536.0
                + expert * 0.25
                + role_index * 0.0625
            ).to(torch.float16)
            tensors[
                f"layers.{_LAYER}.ffn.experts.{expert}.{leaf}.weight"
            ] = value
    save_file(tensors, model_dir / "model.safetensors")
    return tensors


def _col_weights() -> dict[str, torch.Tensor]:
    result = {}
    for expert in range(2):
        for role_index, projection in enumerate(_PROJECTIONS):
            result[
                f"model.layers.{_LAYER}.mlp.experts.{expert}.{projection}"
            ] = torch.linspace(
                0.5 + role_index * 0.125 + expert * 0.03125,
                1.5 + role_index * 0.125 + expert * 0.03125,
                256,
                dtype=torch.float32,
            )
    # Normal production imatrix augmentation carries this redundant packed
    # spelling for down.  It must agree with, not displace, the per-expert rows
    # that define the completed burn's role identity.
    result[
        f"model.layers.{_LAYER}.mlp.experts.down_proj"
    ] = torch.stack([
        result[
            f"model.layers.{_LAYER}.mlp.experts.{expert}.down_proj"
        ].reshape(1, -1)
        for expert in range(2)
    ])
    return result


def _write_bank_cell(
    *,
    root: Path,
    projection: str,
    weight: torch.Tensor,
    col_weights: torch.Tensor,
) -> tuple[Path, tuple[torch.Tensor, ...]]:
    source_shape, source_digest = tensor_value_identity(weight)
    col_shape, col_digest = tensor_value_identity(col_weights)
    semantics_schema = "prismaquant.dsv4_cbl_measurement_semantics.v2"
    train = dict(bank.BANKED_CBL_TRAIN_STAMP)
    book_key = bank._legacy_book_key(
        semantics_schema=semantics_schema,
        layer=_LAYER,
        projection=projection,
        rung=_RUNG,
        source_digest=source_digest,
        col_weights_digest=col_digest,
        train=train,
    )
    tables = _role_book(projection)
    book_sha = _pool_sha(tables)
    book_path = root / book_key[:2] / f"{book_key}.safetensors"
    book_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": bank.BANKED_CBL_BOOK_SCHEMA,
        "book_key": book_key,
        "book_sha256": book_sha,
        "n_sub": len(tables),
        "layer": _LAYER,
        "projection": projection,
        "rung": _RUNG,
        "source_digest": source_digest,
        "col_weights_digest": col_digest,
        "train": train,
        "device_class": "synthetic-cpu",
    }
    save_file(
        {f"sub{index}": table for index, table in enumerate(tables)},
        book_path,
        metadata={
            bank.BANKED_CBL_BOOK_METADATA_KEY: json.dumps(metadata)
        },
    )

    pass_tag = "v2s-primary"
    expert_ids = [0, 1]
    identity = {
        "schema": bank.BURN_CELL_IDENTITY_SCHEMA,
        "pass_tag_schema": bank.BURN_PASS_TAG_SCHEMA,
        "pass_tag": pass_tag,
        "layer": _LAYER,
        "projection": projection,
        "rung": _RUNG,
        "expert_ids": expert_ids,
        "encoded_expert_ids": expert_ids,
        "content_guard": {
            "source_shape": source_shape,
            "source_digest": source_digest,
            "col_weights_shape": col_shape,
            "col_weights_digest": col_digest,
            "col_weights_sha256": col_digest,
            "cbl_semantics": {
                "schema": semantics_schema,
                "adopted_encoder": "cbl_poolb",
                "ldlq_in_measurement": False,
                "book_train": train,
            },
        },
        "predecessor_content_key": None,
        "selection_metric": "per-expert weight_mse",
        "epsilon_rtol": 1e-12,
        "tie_priority": ["free", "embed"],
        "activation_replay": "winning_reconstruction_only",
    }
    cell = {
        "rung": _RUNG,
        "pass_tag_schema": bank.BURN_PASS_TAG_SCHEMA,
        "pass_tag": pass_tag,
        "expert_ids": expert_ids,
        "encoded_expert_ids": expert_ids,
        "warm_state_path": str(book_path),
        "timing": {
            "measurement_semantics": {
                "schema": semantics_schema,
                "encoder": "cbl_poolb",
                "ldlq": False,
                "scale_policy": "one_shot_cand0",
                "book_sha256": book_sha,
                "book_path": str(book_path),
            }
        },
    }
    payload = {
        "schema": bank.BURN_CELL_SCHEMA,
        "pass_tag_schema": bank.BURN_PASS_TAG_SCHEMA,
        "pass_tag": pass_tag,
        "content_key": bank._burn_content_key(identity),
        "identity": identity,
        "cell": cell,
    }
    shard = root.parent / f"accepted-{projection}.pkl"
    with shard.open("wb") as handle:
        pickle.dump(payload, handle)
    return shard, tables


def _fixture(tmp_path: Path):
    model_dir = tmp_path / "model"
    _write_model(model_dir)
    col = _col_weights()
    source = _LazySkeleton(model_dir)
    root = tmp_path / "bucket-books"
    root.mkdir()
    expected = {}
    shards = {}
    for projection in _PROJECTIONS:
        leaf = _SOURCE_LEAF[projection]
        weight = torch.stack([
            source.dequant_weight(
                f"layers.{_LAYER}.ffn.experts.{expert}.{leaf}.weight"
            )
            for expert in range(2)
        ])
        role_col = torch.stack([
            col[
                f"model.layers.{_LAYER}.mlp.experts.{expert}.{projection}"
            ].reshape(1, -1)
            for expert in range(2)
        ])
        shards[projection], expected[projection] = _write_bank_cell(
            root=root,
            projection=projection,
            weight=weight,
            col_weights=role_col,
        )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({
        "schema": bank.ROUTED_MOE_CBL_SELECTION_SCHEMA,
        "book_root": str(root),
        "cells": [
            {
                "layer": _LAYER,
                "projection": projection,
                "rung": _RUNG,
                "burn_shard": str(shards[projection]),
            }
            for projection in _PROJECTIONS
        ],
    }))
    return model_dir, col, selection_path, expected, shards


def test_builder_copies_distinct_gate_up_down_books_and_never_trains(
    tmp_path, monkeypatch
):
    model_dir, col, selection, expected, _shards = _fixture(tmp_path)

    def forbidden_train(*_args, **_kwargs):
        raise AssertionError("routed bundle must not retrain pooled-Lloyd books")

    monkeypatch.setattr(learned, "learn_pool", forbidden_train)
    bundle = build_bundle_from_model(
        model_dir=model_dir,
        col_weights=col,
        formats=["FP8_CBL_K28"],
        output=tmp_path / "bundle.pqcb",
        device="cpu",
        routed_moe_book_selection=selection,
    )

    observed_digests = []
    for projection in _PROJECTIONS:
        qname = f"model.layers.{_LAYER}.mlp.experts.{projection}"
        cell = bundle.cell(qname, "FP8_CBL_K28")
        actual = tuple(
            bundle.sidecar_tensors[ref]
            for ref in cell["codebook_ref"]
        )
        assert all(
            torch.equal(got, want)
            for got, want in zip(actual, expected[projection], strict=True)
        )
        observed_digests.append(tuple(cell["content_sha256"]))
        origin = bank.validate_banked_cbl_origin(
            cell["pretrained_origin"],
            where=f"{qname}/FP8_CBL_K28 test origin",
        )
        assert origin["selection_sha256"] == hashlib.sha256(
            selection.read_bytes()
        ).hexdigest()
        assert origin["burn_shard"] == str(_shards[projection])
        assert origin["layer"] == _LAYER
        assert origin["projection"] == projection
        assert origin["rung"] == _RUNG
        assert origin["subtable_content_sha256"] == cell["content_sha256"]
        assert origin["source_digest"] == bundle.manifest["inputs"][qname][
            "source_weight"
        ]["sha256"]
        assert origin["col_weights_digest"] == bundle.manifest["inputs"][
            qname
        ]["col_weights"]["sha256"]
        for expert in range(2):
            alias = (
                f"model.layers.{_LAYER}.mlp.experts.{expert}.{projection}"
            )
            assert bundle.manifest["aliases"][alias]["cell_qname"] == qname
    assert len(set(observed_digests)) == 3
    assert set(bundle.manifest["cells"]) == {
        f"model.layers.{_LAYER}.mlp.experts.{projection}"
        for projection in _PROJECTIONS
    }


def test_builder_refuses_stale_current_imatrix_identity(tmp_path):
    model_dir, col, selection, _expected, _shards = _fixture(tmp_path)
    stale = dict(col)
    key = f"model.layers.{_LAYER}.mlp.experts.0.gate_proj"
    stale[key] = stale[key].clone()
    stale[key][0] += 1.0

    with pytest.raises(bank.BankedCBLBookError, match="col_weights digest"):
        build_bundle_from_model(
            model_dir=model_dir,
            col_weights=stale,
            formats=["FP8_CBL_K28"],
            output=tmp_path / "stale.pqcb",
            device="cpu",
            routed_moe_book_selection=selection,
        )


def test_builder_refuses_conflicting_redundant_packed_down_imatrix(tmp_path):
    model_dir, col, selection, _expected, _shards = _fixture(tmp_path)
    conflicting = dict(col)
    key = f"model.layers.{_LAYER}.mlp.experts.down_proj"
    conflicting[key] = conflicting[key].clone()
    conflicting[key][0, 0, 0] += 1.0

    with pytest.raises(ValueError, match="redundant packed role col_weights"):
        build_bundle_from_model(
            model_dir=model_dir,
            col_weights=conflicting,
            formats=["FP8_CBL_K28"],
            output=tmp_path / "conflicting.pqcb",
            device="cpu",
            routed_moe_book_selection=selection,
        )


def test_selection_is_hashed_and_missing_shard_fails_closed(tmp_path):
    _model_dir, _col, selection_path, _expected, shards = _fixture(tmp_path)
    selection = bank.load_routed_moe_cbl_selection(selection_path)
    assert selection.content_sha256 == hashlib.sha256(
        selection_path.read_bytes()
    ).hexdigest()
    assert {cell.rung for cell in selection.cells} == {_RUNG}

    shards["up_proj"].unlink()
    with pytest.raises(bank.BankedCBLBookError, match="accepted burn shard"):
        bank.load_routed_moe_cbl_selection(selection_path)


def test_selection_refuses_rungs_outside_routed_k28_k33(tmp_path):
    root = tmp_path / "books"
    root.mkdir()
    shard = tmp_path / "cell.pkl"
    shard.write_bytes(b"not read by selection validation")
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({
        "schema": bank.ROUTED_MOE_CBL_SELECTION_SCHEMA,
        "book_root": str(root),
        "cells": [{
            "layer": 0,
            "projection": "gate_proj",
            "rung": 34,
            "burn_shard": str(shard),
        }],
    }))
    with pytest.raises(bank.BankedCBLBookError, match="K28-K33"):
        bank.load_routed_moe_cbl_selection(manifest)
