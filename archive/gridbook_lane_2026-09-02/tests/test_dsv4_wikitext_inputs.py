"""Strict offline token-input contract for DSv4 gold measurements."""
from __future__ import annotations

import copy
import json

import pytest
import torch

from tools import dsv4_wikitext_inputs as inputs
from tools import measure_vllm_wikitext_ppl as ppl


def _small_payload(monkeypatch) -> tuple[dict, dict]:
    files = {
        "tokenizer.json": {
            "bytes": 1,
            "sha256": "a" * 64,
        }
    }
    tokenizer = {
        "schema": "prismaquant.tokenizer_identity/1",
        "content_sha256": inputs.canonical_sha256({"files": files}),
        "files": files,
    }
    windows = [[1, 2, 3], [4, 5, 6]]
    ppl_ids = [7, 8, 9, 10]
    monkeypatch.setattr(
        inputs, "TOKENIZER_IDENTITY_SHA256", tokenizer["content_sha256"]
    )
    monkeypatch.setattr(inputs, "TOKENIZER_VOCAB_SIZE", 32)
    monkeypatch.setattr(inputs, "FULL_KL_DATASET_FINGERPRINT", "train-fp")
    monkeypatch.setattr(inputs, "FULL_KL_CORPUS_SHA256", "b" * 64)
    monkeypatch.setattr(inputs, "FULL_KL_TOTAL_TOKENS", 7)
    monkeypatch.setattr(inputs, "FULL_KL_N_SAMPLES", 2)
    monkeypatch.setattr(inputs, "FULL_KL_SEQLEN", 3)
    monkeypatch.setattr(inputs, "FULL_KL_STARTS", (0, 3))
    monkeypatch.setattr(
        inputs,
        "FULL_KL_TOKEN_IDS_TENSOR_SHA256",
        inputs._tensor_sha256(torch.tensor(windows, dtype=torch.long)),
    )
    monkeypatch.setattr(inputs, "PPL_DATASET_FINGERPRINT", "test-fp")
    monkeypatch.setattr(inputs, "PPL_CORPUS_SHA256", "c" * 64)
    monkeypatch.setattr(inputs, "PPL_TOTAL_TOKENS", 12)
    monkeypatch.setattr(inputs, "PPL_N_TOKENS", 4)
    monkeypatch.setattr(
        inputs, "PPL_TOKEN_IDS_SHA256", inputs.canonical_sha256(ppl_ids)
    )
    payload = inputs.seal_dsv4_wikitext_inputs({
        "schema": inputs.DSV4_WIKITEXT_INPUTS_SCHEMA,
        "datasets_distribution": {
            "name": inputs.DATASETS_DISTRIBUTION,
            "version": inputs.DATASETS_VERSION,
        },
        "corpus_construction": dict(inputs.CORPUS_CONSTRUCTION),
        "tokenizer": tokenizer,
        "full_kl": {
            "dataset": inputs._expected_dataset(split=inputs.FULL_KL_SPLIT),
            "selection": {
                "sampler": (
                    "python.random.Random(seed).sample(range(max_start), "
                    "n_samples)/v1"
                ),
                "window_seed": inputs.FULL_KL_WINDOW_SEED,
                "n_samples": 2,
                "seqlen": 3,
                "starts": [0, 3],
            },
            "token_ids": windows,
            "token_ids_tensor_sha256": (
                inputs.FULL_KL_TOKEN_IDS_TENSOR_SHA256
            ),
        },
        "ppl": {
            "dataset": inputs._expected_dataset(split=inputs.PPL_SPLIT),
            "selection": {
                "strategy": (
                    "contiguous_prefix_after_full_corpus_tokenization/v1"
                ),
                "n_tokens": 4,
            },
            "token_ids": ppl_ids,
            "token_ids_sha256": inputs.PPL_TOKEN_IDS_SHA256,
        },
    }, expected_tokenizer_identity=tokenizer)
    return payload, tokenizer


def test_offline_inputs_round_trip_and_refuse_token_tampering(
    monkeypatch, tmp_path
):
    payload, tokenizer = _small_payload(monkeypatch)
    path = tmp_path / "wikitext-inputs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = inputs.load_dsv4_wikitext_inputs(
        path, expected_tokenizer_identity=tokenizer
    )
    assert loaded["full_kl"]["token_ids"] == [[1, 2, 3], [4, 5, 6]]
    assert loaded["ppl"]["token_ids"] == [7, 8, 9, 10]

    tampered = copy.deepcopy(payload)
    tampered["ppl"]["token_ids"][0] = 11
    tampered["semantic_sha256"] = inputs.canonical_sha256({
        key: value for key, value in tampered.items()
        if key != "semantic_sha256"
    })
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(inputs.DSv4WikiTextInputsError, match="token values"):
        inputs.load_dsv4_wikitext_inputs(
            path, expected_tokenizer_identity=tokenizer
        )


def test_ppl_dsv4_loader_uses_offline_payload_without_a_tokenizer(monkeypatch):
    attestation = {"content_sha256": "tokenizer"}
    observed = {}

    def load(path, *, expected_tokenizer_identity):
        observed.update(
            path=path,
            expected_tokenizer_identity=expected_tokenizer_identity,
        )
        return {
            "ppl": {
                "dataset": {
                    "fingerprint": "fingerprint",
                    "corpus_sha256": "d" * 64,
                    "total_tokens": 12,
                },
                "token_ids": [1, 2, 3, 4],
            }
        }

    monkeypatch.setattr(ppl, "load_dsv4_wikitext_inputs", load)
    args = type("Args", (), {
        "dsv4_gridbook_contract": True,
        "wikitext_inputs": "/offline/inputs.json",
    })()

    ids, evidence = ppl._load_measurement_ids(
        args, tokenizer_attestation=attestation
    )

    assert ids == [1, 2, 3, 4]
    assert evidence["fingerprint"] == "fingerprint"
    assert observed == {
        "path": "/offline/inputs.json",
        "expected_tokenizer_identity": attestation,
    }


def test_ppl_dsv4_loader_requires_offline_payload():
    args = type("Args", (), {
        "dsv4_gridbook_contract": True,
        "wikitext_inputs": None,
    })()
    with pytest.raises(ValueError, match="requires --wikitext-inputs"):
        ppl._load_measurement_ids(args, tokenizer_attestation={})


def test_offline_inputs_reject_duplicate_members_and_oversize(
    tmp_path, monkeypatch
):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    with pytest.raises(
        inputs.DSv4WikiTextInputsError, match="duplicate object member 'schema'"
    ):
        inputs.load_dsv4_wikitext_inputs(
            duplicate, expected_tokenizer_identity={}
        )

    oversized = tmp_path / "oversized.json"
    monkeypatch.setattr(inputs, "DSV4_WIKITEXT_INPUTS_MAX_BYTES", 8)
    oversized.write_text('{"schema":"too-large"}', encoding="utf-8")
    with pytest.raises(inputs.DSv4WikiTextInputsError, match="size is outside"):
        inputs.load_dsv4_wikitext_inputs(
            oversized, expected_tokenizer_identity={}
        )
