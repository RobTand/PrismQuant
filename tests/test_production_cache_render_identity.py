"""Render-identity guard for the production cache directory (#146).

Resume admits a unit as done by file presence alone, so a ``--cache-dir``
resumed under a different ``--render-scope``, a different
``--include-qnames-file``, a different lever string or a different
calibration would silently mix units rendered under different conditions.
``fill_production_weight_cache`` now writes its render identity into
``render_identity.json`` on first use and refuses on mismatch. A missing
sidecar means a pre-guard directory: warn, do not refuse.
"""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from prismaquant.production_weight_cache import fill_production_weight_cache


class _TinyTwoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(4, 32)
        self.l1 = nn.Linear(32, 32, bias=False)
        self.l2 = nn.Linear(32, 32, bias=False)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[0, 0] = 1.0
            self.embed.weight[1, 1] = 2.0
            self.l1.weight.copy_(torch.eye(32))
            self.l2.weight.fill_(1.0)

    def forward(self, input_ids, use_cache=False):
        x = self.embed(input_ids)
        x = self.l1(x)
        return self.l2(x)


def _fake_render_env(monkeypatch):
    import prismaquant.export_native_compressed as enc
    import prismaquant.production_weight_cache as pwc

    monkeypatch.setattr(
        enc,
        "_compute_nvfp4_joint_global",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        pwc,
        "render_production_weight",
        lambda weight, fmt, **_kwargs: weight.detach().to(torch.float32),
    )


def _fill(model, calib_ids, cache_dir, **kwargs):
    params = {
        "qnames": ["l1", "l2"],
        "formats": ["NVFP4"],
        "levers": {"gptq": False, "scale_sweep": False},
        "max_act_rows": 8,
        "cache_dir": cache_dir,
        "progress": False,
    }
    params.update(kwargs)
    return fill_production_weight_cache(model, calib_ids, **params)


def _sidecar(cache_dir):
    import prismaquant.production_weight_cache as pwc

    return cache_dir / pwc.RENDER_IDENTITY_SIDECAR_FILENAME


def test_resume_refuses_different_render_scope(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    with pytest.raises(ValueError, match="render_scope") as exc_info:
        _fill(
            _TinyTwoLinear(),
            calib_ids,
            tmp_path,
            render_assignment={"l1": "NVFP4"},
        )

    assert "rebuild this directory" in str(exc_info.value)


def test_resume_refuses_different_render_subset(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    with pytest.raises(ValueError, match="rendered_pairs") as exc_info:
        _fill(
            _TinyTwoLinear(),
            calib_ids,
            tmp_path,
            render_qnames=["l1"],
        )

    assert "rebuild this directory" in str(exc_info.value)


def test_resume_refuses_different_hooked_enumeration(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    with pytest.raises(ValueError, match="hooked_qnames") as exc_info:
        _fill(
            _TinyTwoLinear(),
            calib_ids,
            tmp_path,
            qnames=["l1", "l2", "lm_head"],
        )

    assert "rebuild this directory" in str(exc_info.value)


def test_resume_refuses_different_levers(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    with pytest.raises(ValueError, match="levers") as exc_info:
        _fill(
            _TinyTwoLinear(),
            calib_ids,
            tmp_path,
            levers={"gptq": True, "scale_sweep": False},
        )

    assert "rebuild this directory" in str(exc_info.value)


def test_resume_refuses_different_calibration(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    _fill(_TinyTwoLinear(), torch.tensor([[0, 1]], dtype=torch.long), tmp_path)

    with pytest.raises(ValueError, match="calib_hash") as exc_info:
        _fill(
            _TinyTwoLinear(),
            torch.tensor([[2, 3]], dtype=torch.long),
            tmp_path,
        )

    assert "rebuild this directory" in str(exc_info.value)


def test_resume_with_identical_identity_reuses_shards(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    first = _fill(_TinyTwoLinear(), calib_ids, tmp_path)
    second = _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    assert set(second.weights) == set(first.weights)
    assert not second.failed


def test_fresh_directory_writes_validated_sidecar(tmp_path, monkeypatch):
    import prismaquant.production_weight_cache as pwc

    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)

    sidecar = _sidecar(tmp_path)
    assert sidecar.is_file()
    identity = pwc.validate_production_cache_render_identity(
        json.loads(sidecar.read_text())
    )
    assert identity["schema"] == pwc.RENDER_IDENTITY_SCHEMA
    assert identity["render_scope"] == "format-menu"
    assert identity["requested_formats"] == ["NVFP4"]
    assert identity["hooked_qnames_sha256"] == pwc._qname_set_sha256(
        ["l1", "l2"]
    )
    assert identity["rendered_pairs"] == ["l1|NVFP4", "l2|NVFP4"]


def test_preguard_directory_warns_and_adopts_identity(
    tmp_path, monkeypatch, capsys
):
    import prismaquant.production_weight_cache as pwc

    _fake_render_env(monkeypatch)
    model = _TinyTwoLinear()
    torch.save(
        model.l1.weight.detach().to(torch.float32),
        tmp_path / pwc._cache_weight_filename("l1", "NVFP4"),
    )
    assert not _sidecar(tmp_path).exists()

    cache = _fill(model, torch.tensor([[0, 1]], dtype=torch.long), tmp_path)

    out = capsys.readouterr().out
    assert "no render_identity.json" in out
    assert "pre-guard directory" in out
    assert _sidecar(tmp_path).is_file()
    assert ("l1", "NVFP4") in cache.weights


def test_corrupt_sidecar_refuses_fail_closed(tmp_path, monkeypatch):
    _fake_render_env(monkeypatch)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)
    _sidecar(tmp_path).write_text('{"schema": "bogus"')

    with pytest.raises(ValueError, match="rebuild this directory"):
        _fill(_TinyTwoLinear(), calib_ids, tmp_path)
