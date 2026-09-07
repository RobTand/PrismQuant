"""A campaign merge must preserve the journal's existing trust boundary."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def _journal(tmp_path):
    from prismaquant.cost_stage_checkpoint import prepare_journal, write_unit, unit_path

    row = tmp_path / "row"
    manifest = row / "cost.anchors.json"
    parts, digest, _ = prepare_journal(
        row / "cost.anchors.json.parts", stage="Tessera campaign", resume=True,
        identity={"units": {"a": {"weight": "wa"}}}, qnames=["a"],
        manifest_path=manifest)
    write_unit(parts, stage="Tessera campaign", qname="a", identity_sha256=digest,
               state={"measured": {"rung": 1.0}})
    return row, manifest, unit_path(parts, "a")


@pytest.mark.parametrize("field,value", [
    ("payload_sha256", "0" * 64),
    ("identity_sha256", "0" * 64),
    ("qname", "another-unit"),
    ("stage", "another-stage"),
    ("schema", "another-schema"),
])
def test_merge_refuses_a_shard_the_journal_itself_refuses(tmp_path, field, value):
    import dispatch_tessera_campaign as dispatch

    row, _, shard = _journal(tmp_path)
    envelope = pickle.loads(shard.read_bytes())
    envelope[field] = value
    shard.write_bytes(pickle.dumps(envelope))
    out = tmp_path / "merged" / "cost.anchors.json"
    with pytest.raises(RuntimeError, match=field):
        dispatch.merge_checkpoint({"row-0000": str(row)}, out)
    assert not out.exists()


def test_merge_refuses_a_manifest_with_a_wrong_identity_digest(tmp_path):
    import dispatch_tessera_campaign as dispatch

    row, manifest, _ = _journal(tmp_path)
    value = json.loads(manifest.read_text())
    value["identity_sha256"] = "0" * 64
    manifest.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="identity_sha256"):
        dispatch.merge_checkpoint({"row-0000": str(row)}, tmp_path / "merged.json")


def test_merge_refuses_a_manifest_that_drops_an_identity_unit(tmp_path):
    import dispatch_tessera_campaign as dispatch

    row, manifest, _ = _journal(tmp_path)
    value = json.loads(manifest.read_text())
    value["units"] = []
    manifest.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="units"):
        dispatch.merge_checkpoint({"row-0000": str(row)}, tmp_path / "merged.json")


def test_valid_merged_journal_resumes_through_the_owned_reader(tmp_path):
    import dispatch_tessera_campaign as dispatch
    from prismaquant.cost_stage_checkpoint import prepare_journal

    row, _, _ = _journal(tmp_path)
    out = tmp_path / "merged" / "cost.anchors.json"
    merged = dispatch.merge_checkpoint({"row-0000": str(row)}, out)
    _, _, states = prepare_journal(
        out.with_name(out.name + ".parts"), stage="Tessera campaign", resume=True,
        identity=merged["identity"], qnames=["a"], manifest_path=out)
    assert states == {"a": {"measured": {"rung": 1.0}}}


def test_refusal_order_does_not_depend_on_anchor_insertion_order():
    from test_tessera_stack_sample_cost import _anchor
    from prismaquant import tessera_campaign as campaign

    def payload(names):
        return campaign.campaign_cost_payload(
            {name: {"TESSERA_BF16_K1": [
                _anchor(campaign, name, "TESSERA_BF16_K1", "TESSERA_BF16_K1_R1792",
                        1792, 1.0)]} for name in names}, {}, loo={}, provenance={})

    assert payload(["b", "a"])["non_interpolable"] == payload(["a", "b"])["non_interpolable"]


def test_merge_accepts_stack_incomplete_rung_refusal_without_family():
    import dispatch_tessera_campaign as dispatch
    from test_tessera_campaign_fanout import _payloads

    payloads = _payloads()
    refusal = {"qname": "a", "format_name": "TESSERA_BF16_K1_R1792",
               "reason": "stack_rung_incomplete_over_sample", "missing_experts": [2]}
    payloads["row-0000"]["non_interpolable"] = [refusal]
    merged = dispatch.merge_payloads(payloads, census={"counts": {}}, capture_sha256="merged")
    assert merged["non_interpolable"] == [refusal]
