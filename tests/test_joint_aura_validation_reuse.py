"""Allocator metadata reuse must preserve mutation and digest rejection."""
import copy

import pytest

from prismaquant import joint_aura as joint
from test_joint_aura_assignment_diagnostics import _row, UNIT_A


def _prepare(row):
    payload = {"costs": {UNIT_A: {"FP8_E4M3": row}}}
    joint.prepare_joint_aura_identities(payload)
    return row


def test_shared_identity_validated_once_and_detached(monkeypatch):
    from prismaquant import cost_streaming
    row = _row(UNIT_A, [1, 2, 3])
    original = row["probe_identity"]
    other = dict(row)
    calls = []
    validate = cost_streaming.validate_streamed_model_identity
    def counted(*args, **kwargs):
        calls.append(1)
        return validate(*args, **kwargs)
    monkeypatch.setattr(cost_streaming, "validate_streamed_model_identity", counted)
    joint.prepare_joint_aura_identities({"costs": {UNIT_A: {"a": row, "b": other}}})
    for entry in (row, other, row):
        assert joint.validate_joint_aura_entry(entry)
    assert len(calls) == 1
    assert row["probe_identity"] is other["probe_identity"]
    original["source_model"]["shards"][0]["sha256"] = "f" * 64
    assert joint.validate_joint_aura_entry(row)
    detached = row["probe_identity"]["source_model"]
    detached["shards"][0]["sha256"] = "e" * 64
    assert joint.validate_joint_aura_entry(row)
    with pytest.raises(TypeError):
        row["probe_identity"]["temperature"] = 2


@pytest.mark.parametrize("field", ["predicted_dloss", "probe_identity_sha256", "joint_operator_identity_sha256"])
def test_row_mutation_still_rejected(field):
    row = _prepare(_row(UNIT_A, [1, 2, 3]))
    row[field] = 999 if field == "predicted_dloss" else "e" * 64
    with pytest.raises(ValueError):
        joint.validate_joint_aura_entry(row)


def test_replaced_probe_is_validated():
    row = _row(UNIT_A, [1, 2, 3])
    replacement = copy.deepcopy(row["probe_identity"])
    _prepare(row)
    replacement["source_model"]["config"]["fixture"] = "changed"
    row["probe_identity"] = replacement
    row["probe_identity_sha256"] = joint.identity_sha256(replacement)
    row["joint_operator_identity"]["probe_identity_sha256"] = row["probe_identity_sha256"]
    row["joint_operator_identity_sha256"] = joint.identity_sha256(row["joint_operator_identity"])
    with pytest.raises(ValueError, match="content_sha256"):
        joint.validate_joint_aura_entry(row)


def test_corrupt_source_cannot_be_prepared():
    row = _row(UNIT_A, [1, 2, 3])
    row["probe_identity"]["source_model"]["shards"][0]["sha256"] = "f" * 64
    with pytest.raises((RuntimeError, ValueError), match="content_sha256"):
        _prepare(row)


def test_serialization_does_not_preserve_validation_trust():
    import pickle
    row = _prepare(_row(UNIT_A, [1, 2, 3]))
    restored = pickle.loads(pickle.dumps(row))
    assert type(restored["probe_identity"]) is dict
    assert joint.validate_joint_aura_entry(restored)
    restored["probe_identity"]["n_probes"] = True
    with pytest.raises(ValueError, match="at least two probes"):
        joint.validate_joint_aura_entry(restored)


def test_distinct_probe_objects_do_not_share_digest_trust():
    good = _row(UNIT_A, [1, 2, 3])
    bad = copy.deepcopy(good)
    bad["probe_identity"]["temperature"] = 2.0
    joint.prepare_joint_aura_identities({"costs": {UNIT_A: {"a": good, "b": bad}}})
    assert joint.validate_joint_aura_entry(good)
    with pytest.raises(ValueError, match="probe identity mismatch"):
        joint.validate_joint_aura_entry(bad)


def test_preparation_keeps_original_source_type_rejection():
    row = _row(UNIT_A, [1, 2, 3])
    source = row["probe_identity"]["source_model"]
    source["shards"] = tuple(source["shards"])
    # JSON hashes are unchanged by list -> tuple; source admission is stricter.
    with pytest.raises(ValueError, match="source shard"):
        joint.validate_joint_aura_entry(row)
    with pytest.raises((RuntimeError, ValueError), match="source shard"):
        _prepare(row)
