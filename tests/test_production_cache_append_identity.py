"""Append-identity guard for the production cache directory (#170).

#146's ``render_identity.json`` guards ``fill_production_weight_cache`` only.
Two other writers stream further shards into the SAME ``--cache-dir`` after
that fill returns — ``fill_profile_mtp_production_cache`` and
``fill_packed_expert_cache_entries`` — under their own value-bearing inputs
(reservoir budgets, the packed-expert gate corpus, the MTP activation
source). Each append now compare-or-writes its own section of the sidecar:
the first append records it, every later append compares and refuses on the
first differing field.
"""
from __future__ import annotations

import json

import pytest
import torch

import prismaquant.production_weight_cache as pwc
from prismaquant.production_weight_cache import (
    ProductionWeightCache,
    fill_packed_expert_cache_entries,
)

# The section key is owned by production_weight_cache; the literal fallback
# keeps this test importable on the pre-fix tree (where the guard — and the
# constant — does not exist yet) so the refusal tests can be shown failing.
PACKED_APPEND_SIDECAR_KEY = getattr(
    pwc, "PACKED_APPEND_SIDECAR_KEY", "packed_expert_append"
)

from test_mtp_production_cache import (
    _TinyMtpProfile,
    _empty_cache,
    _patch_auto_config,
    _write_activation,
)
from test_packed_expert_cross_domain_gate import ASSIGNMENT, TinyLM
from test_production_cache_render_identity import _TinyTwoLinear


def _packed_cache(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ProductionWeightCache(
        weights={},
        levers={"gptq": True},
        activation_max_abs={},
        failed={},
        cache_dir=str(cache_dir),
        metadata={},
    )


def _packed_append(cache, model, calib, cache_dir, **kwargs):
    params = {
        "render_assignment": dict(ASSIGNMENT),
        "levers": {"gptq": True},
        "profile": None,
        "module_token_budget": 4096,
        "eval_rows_per_expert": 8,
        "cache_dir": cache_dir,
        "progress": False,
    }
    params.update(kwargs)
    return fill_packed_expert_cache_entries(cache, model, calib, **params)


def _mtp_append(cache, activation_dir, cache_dir, **kwargs):
    from prismaquant.mtp_production_cache import (
        fill_profile_mtp_production_cache,
    )

    params = {
        "profile": _TinyMtpProfile(),
        "activation_cache_dir": activation_dir,
        "formats": ("FP8_E4M3",),
        "cache_dir": cache_dir,
        "device": "cpu",
        "dtype": torch.float32,
        "max_act_rows": 2,
        "progress": False,
    }
    params.update(kwargs)
    return fill_profile_mtp_production_cache(cache, "/fake/model", **params)


def _sidecar(cache_dir):
    import prismaquant.production_weight_cache as pwc

    return cache_dir / pwc.RENDER_IDENTITY_SIDECAR_FILENAME


def _packed_rows():
    return torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75],
            [-0.25, 0.5, 1.0, -1.0],
            [0.125, 0.25, -0.5, 0.75],
        ],
        dtype=torch.float32,
    )


def test_packed_append_refuses_different_module_token_budget(tmp_path):
    torch.manual_seed(11)
    model = TinyLM().eval()
    calib = torch.randint(0, 32, (2, 64))
    cache_dir = tmp_path / "weights"
    cache = _packed_cache(cache_dir)

    _packed_append(cache, model, calib, cache_dir, module_token_budget=4096)

    with pytest.raises(ValueError, match="module_token_budget") as exc_info:
        _packed_append(cache, model, calib, cache_dir, module_token_budget=512)

    assert "instead of appending to it" in str(exc_info.value)


def test_packed_append_refuses_added_gate_corpus(tmp_path):
    torch.manual_seed(11)
    model = TinyLM().eval()
    calib = torch.randint(0, 32, (2, 64))
    gate_calib = torch.randint(0, 32, (2, 64))
    cache_dir = tmp_path / "weights"
    cache = _packed_cache(cache_dir)

    _packed_append(cache, model, calib, cache_dir)

    with pytest.raises(
        ValueError, match="gate_calibration_hash"
    ) as exc_info:
        _packed_append(cache, model, calib, cache_dir, gate_calib_ids=gate_calib)

    assert "instead of appending to it" in str(exc_info.value)


def test_packed_append_with_identical_config_reuses_shards(tmp_path):
    torch.manual_seed(11)
    model = TinyLM().eval()
    calib = torch.randint(0, 32, (2, 64))
    cache_dir = tmp_path / "weights"
    cache = _packed_cache(cache_dir)

    first = _packed_append(cache, model, calib, cache_dir)
    second = _packed_append(cache, model, calib, cache_dir)

    # The first call renders the assignment; the repeat call finds every
    # shard, scale and score already done and renders nothing new — but the
    # entries stay resolvable and the recorded budget is the first call's.
    assert set(first) == set(ASSIGNMENT)
    assert set(second) <= set(ASSIGNMENT)
    assert all(
        cache.resolve_key(qname, "NVFP4") is not None for qname in ASSIGNMENT
    )
    sidecar = json.loads(_sidecar(cache_dir).read_text())
    assert (
        sidecar[PACKED_APPEND_SIDECAR_KEY]["module_token_budget"] == 4096
    )


def test_mtp_append_refuses_different_max_act_rows(tmp_path, monkeypatch):
    _patch_auto_config(monkeypatch)
    activation_dir = tmp_path / "activations"
    cache_dir = tmp_path / "weights"
    _write_activation(activation_dir, "mtp.fc", _packed_rows())
    _write_activation(activation_dir, "mtp.proj", _packed_rows().flip(0))
    cache = _empty_cache(cache_dir)

    _mtp_append(cache, activation_dir, cache_dir, max_act_rows=2)

    with pytest.raises(ValueError, match="max_act_rows") as exc_info:
        _mtp_append(cache, activation_dir, cache_dir, max_act_rows=1)

    assert "instead of appending to it" in str(exc_info.value)


def test_mtp_append_refuses_different_activation_source(
    tmp_path, monkeypatch
):
    _patch_auto_config(monkeypatch)
    first_act = tmp_path / "activations-a"
    second_act = tmp_path / "activations-b"
    cache_dir = tmp_path / "weights"
    _write_activation(first_act, "mtp.fc", _packed_rows())
    _write_activation(first_act, "mtp.proj", _packed_rows().flip(0))
    _write_activation(second_act, "mtp.fc", _packed_rows() + 0.5)
    _write_activation(second_act, "mtp.proj", _packed_rows().flip(0) - 0.5)
    cache = _empty_cache(cache_dir)

    _mtp_append(cache, first_act, cache_dir)

    with pytest.raises(
        ValueError, match="activation_source_hash"
    ) as exc_info:
        _mtp_append(cache, second_act, cache_dir)

    assert "instead of appending to it" in str(exc_info.value)


def test_mtp_append_with_identical_config_reuses_shards(
    tmp_path, monkeypatch
):
    _patch_auto_config(monkeypatch)
    activation_dir = tmp_path / "activations"
    cache_dir = tmp_path / "weights"
    _write_activation(activation_dir, "mtp.fc", _packed_rows())
    _write_activation(activation_dir, "mtp.proj", _packed_rows().flip(0))
    cache = _empty_cache(cache_dir)

    first = _mtp_append(cache, activation_dir, cache_dir)
    second = _mtp_append(cache, activation_dir, cache_dir)

    assert first == second == 2
    assert cache.resolve_key("mtp.fc", "FP8_E4M3") is not None


def test_base_fill_resume_ignores_append_sections(tmp_path, monkeypatch):
    """A base resume after an append still compares the base identity only.

    The append section rides in the same sidecar without disturbing the
    base fill's guard in either direction.
    """
    import prismaquant.production_weight_cache as pwc
    from test_production_cache_render_identity import (
        _fake_render_env,
        _fill,
    )

    _fake_render_env(monkeypatch)
    torch.manual_seed(11)
    calib_ids = torch.tensor([[0, 1]], dtype=torch.long)
    _fill(_TinyTwoLinear(), calib_ids, tmp_path)
    base_sidecar = json.loads(_sidecar(tmp_path).read_text())
    assert PACKED_APPEND_SIDECAR_KEY not in base_sidecar

    model = TinyLM().eval()
    packed_calib = torch.randint(0, 32, (2, 64))
    cache = pwc.ProductionWeightCache(
        weights={},
        levers={"gptq": True},
        activation_max_abs={},
        failed={},
        cache_dir=str(tmp_path),
        metadata={},
    )
    fill_packed_expert_cache_entries(
        cache,
        model,
        packed_calib,
        render_assignment=dict(ASSIGNMENT),
        levers={"gptq": True},
        profile=None,
        module_token_budget=4096,
        eval_rows_per_expert=8,
        cache_dir=tmp_path,
        progress=False,
    )
    merged = json.loads(_sidecar(tmp_path).read_text())
    assert merged["render_scope"] == base_sidecar["render_scope"]
    assert merged["calib_hash"] == base_sidecar["calib_hash"]
    assert merged[PACKED_APPEND_SIDECAR_KEY]["module_token_budget"] == 4096

    resumed = _fill(_TinyTwoLinear(), calib_ids, tmp_path)
    assert ("l1", "NVFP4") in resumed.weights
    assert (
        json.loads(_sidecar(tmp_path).read_text())[PACKED_APPEND_SIDECAR_KEY][
            "module_token_budget"
        ]
        == 4096
    )


def test_corrupt_append_section_refuses_fail_closed(tmp_path):
    torch.manual_seed(11)
    model = TinyLM().eval()
    calib = torch.randint(0, 32, (2, 64))
    cache_dir = tmp_path / "weights"
    cache = _packed_cache(cache_dir)

    _packed_append(cache, model, calib, cache_dir)

    sidecar_path = _sidecar(cache_dir)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar[PACKED_APPEND_SIDECAR_KEY]["module_token_budget"] = "lots"
    sidecar_path.write_text(json.dumps(sidecar))

    with pytest.raises(ValueError, match="instead of appending to it"):
        _packed_append(cache, model, calib, cache_dir)
