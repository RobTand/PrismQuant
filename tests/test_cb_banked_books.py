from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import pickle

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import cb_banked_books as bank
from prismaquant.cb_layout import codebook_subtable_shapes, family_for


def _historical_pool_sha(tables):
    digest = hashlib.sha256()
    for table in tables:
        digest.update(table.to(torch.float32).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _fixture(tmp_path: Path):
    layer = 7
    projection = "gate_proj"
    rung = 32
    source_digest = "a" * 64
    col_digest = "b" * 64
    source_shape = (2, 4, 256)
    col_shape = (2, 1, 256)
    expert_ids = [0, 1]
    semantics_schema = "prismaquant.dsv4_cbl_measurement_semantics.v2"
    train = dict(bank.BANKED_CBL_TRAIN_STAMP)
    book_root = tmp_path / "bucket-books"
    book_root.mkdir()
    key = bank._legacy_book_key(
        semantics_schema=semantics_schema,
        layer=layer,
        projection=projection,
        rung=rung,
        source_digest=source_digest,
        col_weights_digest=col_digest,
        train=train,
    )
    book_path = book_root / key[:2] / f"{key}.safetensors"
    book_path.parent.mkdir()
    family = family_for("fp8", "product")
    shapes = codebook_subtable_shapes(rung, family.mode, family.n_sub)
    tables = tuple(
        torch.linspace(
            -1.0 + index / 16.0,
            1.0 - index / 16.0,
            steps=rows * cols,
            dtype=torch.float16,
        ).reshape(rows, cols)
        for index, (rows, cols) in enumerate(shapes)
    )
    book_sha = _historical_pool_sha(tables)
    metadata = {
        "schema": bank.BANKED_CBL_BOOK_SCHEMA,
        "book_key": key,
        "book_sha256": book_sha,
        "n_sub": len(tables),
        "layer": layer,
        "projection": projection,
        "rung": rung,
        "source_digest": source_digest,
        "col_weights_digest": col_digest,
        "train": train,
        "device_class": "synthetic-cpu",
    }
    save_file(
        {f"sub{index}": table for index, table in enumerate(tables)},
        book_path,
        metadata={bank.BANKED_CBL_BOOK_METADATA_KEY: json.dumps(metadata)},
    )

    pass_tag = "v2s-primary"
    identity = {
        "schema": bank.BURN_CELL_IDENTITY_SCHEMA,
        "pass_tag_schema": bank.BURN_PASS_TAG_SCHEMA,
        "pass_tag": pass_tag,
        "layer": layer,
        "projection": projection,
        "rung": rung,
        "expert_ids": expert_ids,
        "encoded_expert_ids": expert_ids,
        "content_guard": {
            "source_shape": list(source_shape),
            "source_digest": source_digest,
            "col_weights_shape": list(col_shape),
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
        "rung": rung,
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
    shard_path = tmp_path / "accepted-cell.pkl"
    with shard_path.open("wb") as handle:
        pickle.dump(payload, handle)
    request = bank.BankedCBLBookRequest(
        burn_shard_path=shard_path,
        layer=layer,
        projection=projection,
        rung=rung,
        source_digest=source_digest,
        col_weights_digest=col_digest,
        source_shape=source_shape,
        col_weights_shape=col_shape,
    )
    return {
        "request": request,
        "book_root": book_root,
        "book_path": book_path,
        "tables": tables,
        "metadata": metadata,
        "payload": payload,
        "shard_path": shard_path,
    }


def _rewrite_shard(fixture, payload):
    payload["content_key"] = bank._burn_content_key(payload["identity"])
    with fixture["shard_path"].open("wb") as handle:
        pickle.dump(payload, handle)


def _rewrite_book(fixture, *, tables=None, metadata=None):
    tables = fixture["tables"] if tables is None else tables
    metadata = fixture["metadata"] if metadata is None else metadata
    save_file(
        {f"sub{index}": table for index, table in enumerate(tables)},
        fixture["book_path"],
        metadata={bank.BANKED_CBL_BOOK_METADATA_KEY: json.dumps(metadata)},
    )


def test_loads_exact_fp16_book_from_explicit_accepted_shard(tmp_path):
    fixture = _fixture(tmp_path)
    resolved = bank.load_banked_cbl_book(
        fixture["request"], book_root=fixture["book_root"]
    )

    assert (resolved.layer, resolved.projection, resolved.rung) == (
        7,
        "gate_proj",
        32,
    )
    assert resolved.format_name == "FP8_CBL_K32"
    assert resolved.encoded_expert_ids == (0, 1)
    assert resolved.book_path == fixture["book_path"]
    assert resolved.book_file_sha256 == hashlib.sha256(
        fixture["book_path"].read_bytes()
    ).hexdigest()
    assert all(table.dtype == torch.float16 for table in resolved.subtables)
    assert all(
        torch.equal(observed, expected)
        for observed, expected in zip(
            resolved.subtables, fixture["tables"], strict=True
        )
    )
    assert len(resolved.subtable_content_sha256) == 4


@pytest.mark.parametrize(
    ("request_update", "message"),
    [
        ({"layer": 8}, "burn cell"),
        ({"projection": "down_proj"}, "burn cell"),
        ({"rung": 28}, "burn cell"),
        ({"source_digest": "c" * 64}, "source digest"),
        ({"col_weights_digest": "d" * 64}, "col_weights digest"),
        ({"source_shape": (2, 5, 256)}, "source shape"),
        ({"col_weights_shape": (2, 256)}, "col_weights shape"),
    ],
)
def test_refuses_requested_cell_or_input_identity_mismatch(
    tmp_path, request_update, message
):
    fixture = _fixture(tmp_path)
    request = replace(fixture["request"], **request_update)
    with pytest.raises(bank.BankedCBLBookError, match=message):
        bank.load_banked_cbl_book(request, book_root=fixture["book_root"])


def test_refuses_non_cbl_poolb_or_changed_trainer_stamp(tmp_path):
    fixture = _fixture(tmp_path)
    payload = copy.deepcopy(fixture["payload"])
    payload["cell"]["timing"]["measurement_semantics"]["encoder"] = (
        "incumbent_sweep_noldlq"
    )
    _rewrite_shard(fixture, payload)
    with pytest.raises(bank.BankedCBLBookError, match="certified cbl_poolb"):
        bank.load_banked_cbl_book(
            fixture["request"], book_root=fixture["book_root"]
        )

    payload = copy.deepcopy(fixture["payload"])
    payload["identity"]["content_guard"]["cbl_semantics"]["book_train"][
        "iters"
    ] = 5
    _rewrite_shard(fixture, payload)
    with pytest.raises(bank.BankedCBLBookError, match="trainer identity differs"):
        bank.load_banked_cbl_book(
            fixture["request"], book_root=fixture["book_root"]
        )


def test_refuses_path_substitution_without_searching_bank(tmp_path):
    fixture = _fixture(tmp_path)
    payload = copy.deepcopy(fixture["payload"])
    wrong = fixture["book_root"] / "decoy" / "decoy.safetensors"
    payload["cell"]["warm_state_path"] = str(wrong)
    payload["cell"]["timing"]["measurement_semantics"]["book_path"] = str(
        wrong
    )
    _rewrite_shard(fixture, payload)

    with pytest.raises(bank.BankedCBLBookError, match="content-addressed key"):
        bank.load_banked_cbl_book(
            fixture["request"], book_root=fixture["book_root"]
        )
    assert not wrong.exists()
    assert fixture["book_path"].is_file()


def test_refuses_missing_book_without_training_or_lattice_fallback(tmp_path):
    fixture = _fixture(tmp_path)
    moved = fixture["book_path"].with_suffix(".held")
    fixture["book_path"].rename(moved)

    with pytest.raises(
        bank.BankedCBLBookError,
        match="missing; refusing retraining or lattice fallback",
    ):
        bank.load_banked_cbl_book(
            fixture["request"], book_root=fixture["book_root"]
        )
    assert not fixture["book_path"].exists()
    assert moved.is_file()


def test_refuses_book_metadata_hash_or_payload_tamper(tmp_path):
    fixture = _fixture(tmp_path)
    metadata = copy.deepcopy(fixture["metadata"])
    metadata["book_sha256"] = "e" * 64
    _rewrite_book(fixture, metadata=metadata)
    with pytest.raises(bank.BankedCBLBookError, match="metadata hashes differ"):
        bank.load_banked_cbl_book(
            fixture["request"], book_root=fixture["book_root"]
        )

    altered = list(fixture["tables"])
    altered[0] = altered[0].clone()
    altered[0][0, 0] += torch.tensor(0.5, dtype=torch.float16)
    _rewrite_book(fixture, tables=tuple(altered))
    with pytest.raises(bank.BankedCBLBookError, match="do not reproduce book_sha256"):
        bank.load_banked_cbl_book(
            fixture["request"], book_root=fixture["book_root"]
        )


def test_explicit_set_rejects_duplicates_and_empty_input(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(bank.BankedCBLBookError, match="duplicate accepted"):
        bank.load_banked_cbl_books(
            [fixture["request"], fixture["request"]],
            book_root=fixture["book_root"],
        )
    with pytest.raises(bank.BankedCBLBookError, match="no accepted"):
        bank.load_banked_cbl_books([], book_root=fixture["book_root"])
