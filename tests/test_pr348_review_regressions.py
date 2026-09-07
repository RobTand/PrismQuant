"""Independent regression for PR348's orphaned export-cache admission."""
from pathlib import Path
import pytest
import torch
from test_export_resume_source_identity import (
    _exporter, _fill_cache, _write_source, _WEIGHT_A, _WEIGHT_B, _BIAS_A,
)


@pytest.mark.parametrize("change_source", [False, True])
def test_missing_manifest_never_admits_orphaned_layer_payloads(tmp_path, monkeypatch, change_source):
    run, source = _exporter(tmp_path, monkeypatch)
    cache = tmp_path / "resume"
    _fill_cache(run, source, cache, weight=_WEIGHT_A, bias=_BIAS_A)
    (cache / "manifest.json").unlink()
    expected = _WEIGHT_B if change_source else _WEIGHT_A
    if change_source:
        _write_source(source, weight=expected, bias=_BIAS_A)
    replays = []
    emitted = run(cache, replay_log=replays)
    got = emitted["model.layers.0.proj.weight"]
    assert not replays and torch.equal(got, expected), {
        "orphaned_layers_replayed": replays,
        "actual_emitted_weight": got.tolist(),
        "expected_source_weight": expected.tolist(),
    }
