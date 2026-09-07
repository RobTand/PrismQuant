"""Exact token artifacts preserve the draw rather than re-running a sampler."""
import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file


def artifact(tmp_path, *, dtype=torch.int64, provenance=None):
    ids = torch.arange(32).reshape(4, 8).to(dtype)
    identity = {"source": "fixture", "fit_tokens": 32, "nsamples": 4, "seqlen": 8,
                "fit_ids_sha256": hashlib.sha256(ids.int().numpy().tobytes()).hexdigest()}
    identity.update(provenance or {})
    path = tmp_path / "tokens.safetensors"
    save_file({"calibration_ids": ids}, str(path),
              metadata={"calibration_provenance": json.dumps(identity)})
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), ids


def test_exact_calibration_preserves_ids_and_both_hash_conventions(tmp_path):
    from prismaquant.calibration_data import load_calibration_input
    path, sha, ids = artifact(tmp_path)
    actual, receipt = load_calibration_input(path, expected_sha256=sha, n_samples=4, seqlen=8)
    assert torch.equal(actual, ids)
    assert receipt["artifact_sha256"] == sha
    assert receipt["calibration_sha256"] == hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    assert receipt["provenance"]["fit_ids_sha256"] != receipt["calibration_sha256"]


@pytest.mark.parametrize("change", ["file_hash", "shape", "draw_hash", "dtype", "missing_hash"])
def test_exact_calibration_refuses_different_contract(tmp_path, change):
    from prismaquant.calibration_data import load_calibration_input
    path, sha, _ = artifact(tmp_path, dtype=torch.int32 if change == "dtype" else torch.int64,
                            provenance={"fit_ids_sha256": "0" * 64} if change == "draw_hash" else None)
    with pytest.raises(ValueError):
        load_calibration_input(path, expected_sha256=None if change == "missing_hash" else
                               "0" * 64 if change == "file_hash" else sha,
                               n_samples=3 if change == "shape" else 4, seqlen=8)


@pytest.mark.parametrize("extra", [
    ["--calibration-input", "tokens.safetensors"],
    ["--calibration-input-sha256", "a" * 64],
    ["--calibration-input", "tokens.safetensors", "--calibration-input-sha256", "a" * 64,
     "--dataset", "wikitext"],
])
def test_calibration_cli_refuses_ambiguous_inputs_before_model_loading(extra):
    from prismaquant.aura_cost import main
    with pytest.raises(SystemExit) as failure:
        main(["--model", "/nonexistent", "--output", "/unused", *extra])
    assert failure.value.code == 2
