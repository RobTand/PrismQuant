"""A file-oriented stage reuses the journal's existing identity envelope."""
import json

import pytest

from prismaquant.cost_stage_checkpoint import (
    MANIFEST_SCHEMA, prepare_journal, write_unit,
)


def test_explicit_manifest_path_preserves_same_identity_resume(tmp_path):
    root = tmp_path / "cost.anchors.json.parts"
    manifest = tmp_path / "cost.anchors.json"
    identity = {"source": "one", "units": ["unit"]}
    journal, digest, state = prepare_journal(
        root, stage="tessera campaign", resume=True, identity=identity,
        qnames=["unit"], manifest_path=manifest,
    )
    assert journal == root and state == {}
    assert json.loads(manifest.read_text())["schema"] == MANIFEST_SCHEMA
    assert not (root / "manifest.json").exists()
    write_unit(root, stage="tessera campaign", qname="unit",
               identity_sha256=digest, state={"anchors": [1]})
    assert prepare_journal(
        root, stage="tessera campaign", resume=True, identity=identity,
        qnames=["unit"], manifest_path=manifest,
    )[2] == {"unit": {"anchors": [1]}}
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        prepare_journal(
            root, stage="tessera campaign", resume=True,
            identity={"source": "changed", "units": ["unit"]},
            qnames=["unit"], manifest_path=manifest,
        )


def test_explicit_manifest_path_refuses_legacy_without_overwrite(tmp_path):
    manifest = tmp_path / "cost.anchors.json"
    manifest.write_text(json.dumps({"schema": "old campaign", "anchors": []}))
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        prepare_journal(
            tmp_path / "parts", stage="tessera campaign", resume=True,
            identity={"source": "current"}, qnames=["unit"], manifest_path=manifest,
        )
    assert json.loads(manifest.read_text())["schema"] == "old campaign"
