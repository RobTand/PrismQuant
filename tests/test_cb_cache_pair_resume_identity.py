from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import prismaquant.production_weight_cache as pwc
from prismaquant.nvfp4_cb_footprint import CBSerializationContext


class _TinyDense(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(32, 32, bias=False)
        self.forward_calls = 0
        self.fail_forward = False

    def forward(self, input_ids, use_cache=False):
        del use_cache
        self.forward_calls += 1
        if self.fail_forward:
            raise AssertionError("identity-admitted resume ran model.forward")
        batch, seqlen = input_ids.shape
        x = torch.ones((batch, seqlen, 32), dtype=self.l1.weight.dtype)
        return SimpleNamespace(logits=self.l1(x))


def _fill(model: _TinyDense, cache_dir, *, max_act_rows: int = 8):
    return pwc.fill_production_weight_cache(
        model,
        torch.tensor([[0, 1, 2]], dtype=torch.long),
        qnames=["l1"],
        formats=["FP8_CB_K28"],
        cache_dir=cache_dir,
        levers={"gptq": False},
        max_act_rows=max_act_rows,
        col_weights={"l1": torch.arange(1, 33, dtype=torch.float32)},
        cb_serialization_context=CBSerializationContext.production(
            codebook_source="lattice",
            ldlq_scope="none",
        ),
        progress=False,
    )


def test_dense_cb_pair_resume_skips_encoder_and_forward(tmp_path, monkeypatch):
    model = _TinyDense()
    calls = {"render": 0}

    def render(weight, _fmt, **_kwargs):
        calls["render"] += 1
        return weight.detach().float() + 0.125

    monkeypatch.setattr(pwc, "render_production_weight", render)
    monkeypatch.setattr(pwc, "_production_cache_git_commit", lambda: "a" * 40)
    first = _fill(model, tmp_path)
    assert calls["render"] == 1

    sidecar_path = tmp_path / pwc._cache_pair_identity_filename(
        "l1", "FP8_CB_K28"
    )
    sidecar = json.loads(sidecar_path.read_text())
    required = {
        "codebook_source",
        "codebook_bundle_sha256",
        "col_weights_sha256",
        "scale_coding",
        "scale_sweep_scope",
        "ldlq_scope",
        "layout_version",
        "rung",
        "format",
        "calibration_hash",
        "git_commit",
    }
    assert required.issubset(sidecar["identity"])
    assert sidecar["identity"]["codebook_source"] == "lattice"
    assert sidecar["identity"]["rung"] == 28
    assert sidecar["tensor"]["content_sha256"]
    assert sidecar["render_score"]["qname"] == "l1"

    monkeypatch.setattr(
        pwc,
        "render_production_weight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity-admitted resume reran encoder")
        ),
    )
    model.fail_forward = True
    resumed = _fill(model, tmp_path)

    assert model.forward_calls == 1
    assert first.metadata["cb_cache_pair_identity"] == (
        resumed.metadata["cb_cache_pair_identity"]
    )
    assert resumed.metadata["render_scores"]["records"] == (
        first.metadata["render_scores"]["records"]
    )


def test_dense_cb_pair_resume_names_mismatched_identity_field(
    tmp_path, monkeypatch
):
    model = _TinyDense()
    monkeypatch.setattr(
        pwc,
        "render_production_weight",
        lambda weight, _fmt, **_kwargs: weight.detach().float() + 0.125,
    )
    monkeypatch.setattr(pwc, "_production_cache_git_commit", lambda: "b" * 40)
    _fill(model, tmp_path)

    sidecar_path = tmp_path / pwc._cache_pair_identity_filename(
        "l1", "FP8_CB_K28"
    )
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["identity"]["col_weights_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))

    with pytest.raises(RuntimeError, match="field 'col_weights_sha256' differs"):
        _fill(model, tmp_path)


def test_dense_cb_pair_resume_refuses_replaced_shard(tmp_path, monkeypatch):
    model = _TinyDense()
    monkeypatch.setattr(
        pwc,
        "render_production_weight",
        lambda weight, _fmt, **_kwargs: weight.detach().float() + 0.125,
    )
    monkeypatch.setattr(pwc, "_production_cache_git_commit", lambda: "c" * 40)
    _fill(model, tmp_path)

    shard_path = tmp_path / pwc._cache_weight_filename("l1", "FP8_CB_K28")
    original = torch.load(shard_path, map_location="cpu", weights_only=True)
    torch.save(torch.zeros_like(original), shard_path)

    with pytest.raises(RuntimeError, match="tensor.content_sha256"):
        resumed = _fill(model, tmp_path)
        resumed.get("l1", "FP8_CB_K28")


def test_dense_cb_pair_resume_binds_activation_sampling_budget(
    tmp_path, monkeypatch
):
    model = _TinyDense()
    monkeypatch.setattr(
        pwc,
        "render_production_weight",
        lambda weight, _fmt, **_kwargs: weight.detach().float() + 0.125,
    )
    monkeypatch.setattr(pwc, "_production_cache_git_commit", lambda: "d" * 40)
    _fill(model, tmp_path, max_act_rows=8)

    # #146: the directory-level render-identity guard refuses first, naming
    # the same field, before the per-pair contract is even consulted.
    with pytest.raises(ValueError, match=r"max_act_rows") as exc_info:
        _fill(model, tmp_path, max_act_rows=9)
    assert "rebuild this directory" in str(exc_info.value)

    # The per-pair contract still binds the budget underneath: admit the
    # directory identity and the CB resume gate refuses on the same field.
    sidecar_path = tmp_path / pwc.RENDER_IDENTITY_SIDECAR_FILENAME
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["max_act_rows"] = 9
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    with pytest.raises(
        RuntimeError,
        match=r"render_input_contract\.max_act_rows.*differs",
    ):
        _fill(model, tmp_path, max_act_rows=9)


def test_dense_cb_pair_resume_refuses_mutated_surrogate_score(
    tmp_path, monkeypatch
):
    model = _TinyDense()
    monkeypatch.setattr(
        pwc,
        "render_production_weight",
        lambda weight, _fmt, **_kwargs: weight.detach().float() + 0.125,
    )
    monkeypatch.setattr(pwc, "_production_cache_git_commit", lambda: "e" * 40)
    _fill(model, tmp_path)

    sidecar_path = tmp_path / pwc._cache_pair_identity_filename(
        "l1", "FP8_CB_K28"
    )
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["render_score"]["score"] = 999.0
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))

    with pytest.raises(RuntimeError, match="render_score_sha256.*differs"):
        _fill(model, tmp_path)


def test_producer_source_digest_binds_every_package_file(tmp_path):
    package_root = tmp_path / "prismaquant"
    nested = package_root / "data"
    bytecode = package_root / "__pycache__"
    nested.mkdir(parents=True)
    bytecode.mkdir()
    (package_root / "producer.py").write_text("VALUE = 1\n")
    lattice = nested / "lattice.pt"
    lattice.write_bytes(b"lattice-v1")
    ignored = bytecode / "producer.pyc"
    ignored.write_bytes(b"runtime-cache-v1")

    original = pwc._production_cache_source_sha256(package_root)
    lattice.write_bytes(b"lattice-v2")
    assert pwc._production_cache_source_sha256(package_root) != original

    lattice.write_bytes(b"lattice-v1")
    ignored.write_bytes(b"runtime-cache-v2")
    assert pwc._production_cache_source_sha256(package_root) == original


@pytest.mark.parametrize("length", [39, 41, 63, 65])
def test_producer_git_override_rejects_non_object_id_lengths(
    monkeypatch, length
):
    monkeypatch.setenv("PRISMAQUANT_IDENTITY_GIT_COMMIT", "a" * length)
    with pytest.raises(RuntimeError, match="40- or 64-character"):
        pwc._production_cache_git_commit()
