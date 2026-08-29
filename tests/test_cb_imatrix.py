from __future__ import annotations

import pickle

import pytest
import torch

from prismaquant.cb_imatrix import (
    CB_IMATRIX_FROM_PROBE_SCHEMA,
    canonical_imatrix_sha256,
    imatrix_from_probe_file,
    imatrix_from_probe_stats,
)


def test_probe_marginals_become_dense_and_per_expert_imatrix_values():
    values, provenance = imatrix_from_probe_stats({
        "dense": {
            "act_sq_sum": torch.tensor([4.0, 8.0, 12.0]),
            "n_tokens_seen": 4,
            "in_features": 3,
        },
        "experts": {
            "expert_act_sq_sum": torch.tensor([[2.0, 4.0], [9.0, 12.0]]),
            "expert_tokens": torch.tensor([2, 3]),
        },
        "no_marginal": {"other": torch.ones(1)},
    })

    assert torch.equal(values["dense"], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(
        values["experts"],
        torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
    )
    assert provenance == {
        "schema": CB_IMATRIX_FROM_PROBE_SCHEMA,
        "dense_entries": 1,
        "packed_entries": 1,
        "skipped_missing_entries": 1,
        "value_sha256": canonical_imatrix_sha256(values),
    }


def test_imatrix_value_hash_is_qname_sorted_and_value_sensitive():
    first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    second = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    assert canonical_imatrix_sha256(first) == canonical_imatrix_sha256(second)
    second["b"] = torch.tensor([3.0])
    assert canonical_imatrix_sha256(first) != canonical_imatrix_sha256(second)


def test_unrouted_expert_requires_existing_neutral_prior_synthesis():
    with pytest.raises(ValueError, match="unrouted expert.*neutral-prior"):
        imatrix_from_probe_stats({
            "experts": {
                "expert_act_sq_sum": torch.ones(2, 4),
                "expert_tokens": torch.tensor([1, 0]),
            },
        })


def test_probe_file_preserves_calibration_identity(tmp_path):
    path = tmp_path / "probe.pkl"
    with path.open("wb") as handle:
        pickle.dump({
            "stats": {
                "dense": {
                    "act_sq_sum": torch.tensor([2.0, 6.0]),
                    "n_tokens_seen": 2,
                },
            },
            "meta": {"calib_hash": "calibration-a"},
        }, handle)
    values, provenance = imatrix_from_probe_file(path)
    assert torch.equal(values["dense"], torch.tensor([1.0, 3.0]))
    assert provenance["calibration_hash"] == "calibration-a"
    assert provenance["probe_path"] == str(path.resolve())
