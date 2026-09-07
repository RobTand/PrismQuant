"""Old source captures remain reproducible history, never fresh preparation."""
import hashlib
import json

import pytest

from experiments.pq267_native_panel import load_canonical_capture


def test_original_preparation_plan_cannot_reuse_quarantined_source():
    with pytest.raises(ValueError, match="canonical capture plan v2"):
        load_canonical_capture({"schema": "prismaquant.native_dense_preparation.v1"})


def test_renaming_plan_cannot_promote_old_capture(tmp_path):
    capture = tmp_path / "capture.json"
    capture.write_text(json.dumps({"schema": "prismaquant.tessera_calibration_cache.v1",
                                  "status": "complete", "canonical": True}))
    with pytest.raises(ValueError, match="historical unqualified"):
        load_canonical_capture({"schema": "prismaquant.native_dense_preparation.v2", "capture": str(capture),
                                "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest()})
