from __future__ import annotations

import builtins
import json

import torch
import torch.nn as nn

from prismaquant.sensitivity_probe import discover_moe_routers, load_calibration


class _ToyTokenizer:
    eos_token_id = 0

    def __call__(self, text, return_tensors="pt", truncation=False):
        del return_tensors, truncation
        ids = [max(1, (ord(ch) % 31)) for ch in text]
        return type("Tokenized", (), {"input_ids": torch.tensor([ids])})


def test_local_jsonl_calibration_does_not_require_datasets(monkeypatch, tmp_path):
    path = tmp_path / "calib.jsonl"
    path.write_text(json.dumps({"text": "abcdefghijklmnopqrstuvwxyz"}) + "\n")

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "datasets":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    calib = load_calibration(_ToyTokenizer(), str(path), n_samples=1, seqlen=8)

    assert calib.shape == (1, 8)


def test_extensionless_bind_mounted_jsonl_stays_local(monkeypatch, tmp_path):
    path = tmp_path / "dataset"
    path.write_text(json.dumps({"text": "abcdefghijklmnopqrstuvwxyz"}) + "\n")

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "datasets":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    calib = load_calibration(_ToyTokenizer(), str(path), n_samples=1, seqlen=8)

    assert calib.shape == (1, 8)


class _PackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(7, 6, 4))
        self.down_proj = nn.Parameter(torch.empty(7, 4, 3))


class _PackedMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 7, bias=False)
        self.experts = _PackedExperts()


class _PackedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _PackedMlp()


class _PackedProfile:
    def packed_expert_param_names(self):
        return ("gate_up_proj", "down_proj")


def test_packed_moe_router_discovery_does_not_require_linear_expert_leaves():
    assert discover_moe_routers(
        _PackedModel(), profile=_PackedProfile()) == {"mlp.gate": 7}
