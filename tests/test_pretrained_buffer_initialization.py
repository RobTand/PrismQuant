"""Ordinary checkpoint loads must initialize tensors absent from the wire."""
import pytest
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from prismaquant import (genuine_weight_initialization,
    pretrained_initialization_contract, validate_pretrained_initialization_contract)


class _Config(PretrainedConfig):
    model_type = 'pq_nonpersistent_buffer_probe'


class _Model(PreTrainedModel):
    config_class = _Config
    base_model_prefix = 'probe'
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(2, 2, bias=False)
        self.register_buffer('derived_scale', torch.tensor([3.], dtype=torch.float32), persistent=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, _Model):
            module.derived_scale.fill_(3.)
        elif isinstance(module, nn.Linear):
            module.weight.fill_(2.)


def _checkpoint(tmp_path):
    with genuine_weight_initialization():
        source = _Model(_Config())
    with torch.no_grad():
        source.linear.weight.fill_(7.)
    source.save_pretrained(tmp_path)


def test_checkpoint_load_initializes_nonpersistent_buffers_without_replacing_weights(tmp_path, monkeypatch):
    _checkpoint(tmp_path)
    original = torch.empty_like
    def poisoned_empty(value, *args, **kwargs):
        result = original(value, *args, **kwargs)
        if result.shape == (1,) and result.is_floating_point() and not result.is_meta:
            result.fill_(-777.)
        return result
    monkeypatch.setattr(torch, 'empty_like', poisoned_empty)
    loaded = _Model.from_pretrained(tmp_path, dtype=torch.bfloat16)
    assert torch.equal(loaded.derived_scale, torch.tensor([3.]))
    assert torch.equal(loaded.linear.weight, torch.full((2, 2), 7., dtype=torch.bfloat16))


def test_checkpoint_initializer_failure_does_not_enable_later_skeleton_initialization(tmp_path, monkeypatch):
    _checkpoint(tmp_path)
    def fail(module_self, module):
        raise RuntimeError('initialization sentinel')
    monkeypatch.setattr(_Model, '_init_weights', fail)
    with pytest.raises(RuntimeError, match='initialization sentinel'):
        _Model.from_pretrained(tmp_path)
    # The pre-existing no-init from-config contract remains scoped outside
    # checkpoint finalization, including after an initializer exception.
    _Model(_Config())


def test_initialization_contract_is_emitted_only_by_completed_checkpoint_load(tmp_path):
    _checkpoint(tmp_path)
    with pytest.raises(ValueError, match='initialization contract'):
        pretrained_initialization_contract(_Model(_Config()))
    loaded = _Model.from_pretrained(tmp_path)
    contract = pretrained_initialization_contract(loaded)
    assert contract['scope'] == 'checkpoint_missing_state'
    assert contract['status'] == 'completed'
    assert contract['transformers_version']
    contract['status'] = 'changed'
    assert pretrained_initialization_contract(loaded)['status'] == 'completed'
    with pytest.raises(ValueError, match='initialization contract'):
        validate_pretrained_initialization_contract(contract)


def test_checkpoint_initialization_does_not_enable_another_thread(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    _checkpoint(tmp_path)
    real_init = _Model._init_weights
    with ThreadPoolExecutor(max_workers=1) as pool:
        def check_thread(self, module):
            if isinstance(module, _Model):
                # A global toggle would recursively enter this callback in
                # the worker thread. Context-local permission leaves its
                # ordinary from-config construction on the no-init path.
                pool.submit(_Model, _Config()).result(timeout=10)
            return real_init(self, module)
        monkeypatch.setattr(_Model, '_init_weights', check_thread)
        loaded = _Model.from_pretrained(tmp_path)
    assert torch.equal(loaded.derived_scale, torch.tensor([3.]))
