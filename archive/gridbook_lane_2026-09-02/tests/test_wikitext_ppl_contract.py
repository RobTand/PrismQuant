"""Immutable evidence for the release WikiText perplexity workload."""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
import types

import pytest
import torch

from tools import measure_vllm_wikitext_ppl as ppl


class _Dataset(list):
    _fingerprint = "immutable-test-fingerprint"


class _Tokenizer:
    def __call__(self, text, *, return_tensors, add_special_tokens):
        assert text == "first\n\nsecond"
        assert return_tensors == "pt"
        assert add_special_tokens is False
        return types.SimpleNamespace(
            input_ids=torch.arange(12, dtype=torch.long).reshape(1, 12)
        )


def test_load_ids_pins_dataset_revision_and_value_identity(monkeypatch):
    observed = {}

    def load_dataset(name, config, **kwargs):
        observed.update(name=name, config=config, **kwargs)
        return _Dataset([
            {"text": "first"},
            {"text": "  "},
            {"text": "second"},
        ])

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset)
    )
    ids, evidence = ppl._load_ids(
        _Tokenizer(), cache_dir="/cache", split="test", n_tokens=8
    )

    assert ids == list(range(8))
    assert observed == {
        "name": ppl.WIKITEXT_DATASET,
        "config": ppl.WIKITEXT_CONFIG,
        "split": "test",
        "cache_dir": "/cache",
        "revision": ppl.WIKITEXT_REVISION,
    }
    assert evidence == {
        "fingerprint": "immutable-test-fingerprint",
        "corpus_sha256": hashlib.sha256(
            b"first\n\nsecond"
        ).hexdigest(),
        "total_tokens": 12,
    }


def test_ppl_contract_binds_token_prefix_and_nonoverlapping_scoring():
    args = argparse.Namespace(
        split="test", n_tokens=8, seqlen=4,
    )
    ids = list(range(8))
    contract = ppl._ppl_calibration_contract(
        args=args,
        ids=ids,
        dataset_evidence={
            "fingerprint": "dataset-fingerprint",
            "corpus_sha256": "c" * 64,
            "total_tokens": 12,
        },
        tokenizer_identity_sha256="d" * 64,
        chunks=[ids[:4], ids[4:]],
        n_tokens_scored=6,
    )

    assert contract["schema"] == ppl.WIKITEXT_PPL_CALIBRATION_SCHEMA
    assert contract["dataset"]["revision"] == ppl.WIKITEXT_REVISION
    assert contract["token_selection"] == {
        "strategy": "contiguous_prefix_after_full_corpus_tokenization/v1",
        "n_tokens_requested": 8,
        "n_tokens_available": 12,
        "selected_token_count": 8,
        "token_ids_sha256": ppl._canonical_sha256(ids),
        "digest_encoding": "canonical_json_integer_array/v1",
    }
    assert contract["scoring"] == {
        "chunking": "nonoverlapping_contiguous/v1",
        "seqlen": 4,
        "chunk_starts": [0, 4],
        "chunk_token_counts": [4, 4],
        "positions": "within_each_chunk_positions_1_through_N_minus_1",
        "n_tokens_scored": 6,
        "prompt_logprobs": 1,
        "temperature": 0.0,
        "max_tokens": 1,
        "detokenize": False,
    }


def test_load_ids_refuses_a_shorter_than_requested_prefix(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *args, **kwargs: _Dataset([
            {"text": "first"},
            {"text": "second"},
        ])),
    )

    with pytest.raises(RuntimeError, match="requested exact prefix"):
        ppl._load_ids(
            _Tokenizer(), cache_dir="/cache", split="test", n_tokens=13
        )


@pytest.mark.parametrize("name,value", [
    ("n_tokens", 1),
    ("n_tokens", True),
    ("seqlen", 1),
    ("seqlen", False),
])
def test_workload_preflight_requires_at_least_two_tokens(name, value):
    args = argparse.Namespace(
        split="test",
        n_tokens=8,
        seqlen=4,
        dsv4_gridbook_contract=False,
    )
    setattr(args, name, value)

    with pytest.raises(ValueError, match="integer >= 2"):
        ppl._validate_workload_args(args)


@pytest.mark.parametrize("name,value", [
    ("split", "train"),
    ("n_tokens", 8191),
    ("seqlen", 511),
])
def test_dsv4_workload_preflight_is_closed(name, value):
    args = argparse.Namespace(
        split=ppl.DSV4_WIKITEXT_SPLIT,
        n_tokens=ppl.DSV4_WIKITEXT_N_TOKENS,
        seqlen=ppl.DSV4_WIKITEXT_SEQLEN,
        dsv4_gridbook_contract=True,
    )
    setattr(args, name, value)

    with pytest.raises(ValueError, match="DSv4 Gridbook PPL requires"):
        ppl._validate_workload_args(args)


def test_contract_refuses_nonlossless_or_one_token_chunk_partitions():
    args = argparse.Namespace(
        split="test",
        n_tokens=8,
        seqlen=4,
        dsv4_gridbook_contract=False,
    )
    evidence = {
        "fingerprint": "dataset-fingerprint",
        "corpus_sha256": "c" * 64,
        "total_tokens": 12,
    }
    ids = list(range(8))

    with pytest.raises(RuntimeError, match="exact chunk law"):
        ppl._ppl_calibration_contract(
            args=args,
            ids=ids,
            dataset_evidence=evidence,
            tokenizer_identity_sha256="d" * 64,
            chunks=[ids[:4], ids[5:]],
            n_tokens_scored=5,
        )
    with pytest.raises(RuntimeError, match="at least two tokens"):
        ppl._build_chunks(list(range(9)), seqlen=4)


def test_dsv4_contract_pins_all_value_bearing_identities(monkeypatch):
    args = argparse.Namespace(
        split=ppl.DSV4_WIKITEXT_SPLIT,
        n_tokens=ppl.DSV4_WIKITEXT_N_TOKENS,
        seqlen=ppl.DSV4_WIKITEXT_SEQLEN,
        dsv4_gridbook_contract=True,
    )
    ids = list(range(ppl.DSV4_WIKITEXT_N_TOKENS))
    chunks = ppl._build_chunks(ids, seqlen=ppl.DSV4_WIKITEXT_SEQLEN)
    monkeypatch.setattr(
        ppl,
        "_canonical_sha256",
        lambda value: ppl.DSV4_WIKITEXT_SELECTED_TOKEN_IDS_SHA256,
    )
    contract = ppl._ppl_calibration_contract(
        args=args,
        ids=ids,
        dataset_evidence={
            "fingerprint": ppl.DSV4_WIKITEXT_DATASET_FINGERPRINT,
            "corpus_sha256": ppl.DSV4_WIKITEXT_CORPUS_SHA256,
            "total_tokens": ppl.DSV4_WIKITEXT_TOTAL_TOKENS,
        },
        tokenizer_identity_sha256=ppl.DSV4_TOKENIZER_IDENTITY_SHA256,
        chunks=chunks,
        n_tokens_scored=8176,
    )

    assert contract["dataset"]["fingerprint"] == "7ccd6deaa4fc56e5"
    assert contract["dataset"]["corpus_sha256"] == (
        "c5b5caea5bd655cb221545a484f2f0f59d35092a17a66840d7b9513d0b99687d"
    )
    assert contract["token_selection"]["n_tokens_available"] == 287_597
    assert contract["token_selection"]["token_ids_sha256"] == (
        "6c23cefbd78c327d6edac566a5c6b419871021b6cf9890ec830713c1de704961"
    )
    assert contract["tokenizer"]["identity_sha256"] == (
        "9f7ee7cb93b58bf30f278965547e7584b89c848e76c3adfeb92c070a88492de0"
    )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 1e-9])
def test_logprob_value_refuses_nonfinite_or_positive_values(value):
    with pytest.raises(ValueError, match="logprob"):
        ppl._logprob_value({7: value}, 7)


def test_result_json_refuses_nan():
    with pytest.raises(ValueError, match="JSON compliant"):
        ppl._strict_json({"mean_nll": math.nan})
