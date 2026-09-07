"""Canonical census provenance comes from the loaded model, not CLI labels."""
import importlib.metadata
import json
from types import SimpleNamespace

import pytest
import torch
import prismaquant


def fixture(monkeypatch,tmp_path):
    from test_tessera_campaign_resume import _main_fixture
    tc,_,argv,model,inputs = _main_fixture(monkeypatch,tmp_path)
    model.config = SimpleNamespace(_attn_implementation='eager')
    contract = dict(schema='prismaquant.pretrained_initialization.v1',
        scope='checkpoint_missing_state',status='completed',
        transformers_version=importlib.metadata.version('transformers'))
    calls = []
    def getter(actual):
        assert actual is model
        calls.append(actual)
        return dict(contract)
    monkeypatch.setattr(prismaquant,'pretrained_initialization_contract',getter)
    path = tmp_path/'census.json'
    return tc,argv+['--census-out',str(path)],model,contract,calls,path


def test_census_binds_actual_model_initialization_and_backend(monkeypatch,tmp_path):
    tc,argv,model,contract,calls,path = fixture(monkeypatch,tmp_path)
    assert tc.main([*argv,'--attention-implementation','eager']) == 0
    census = json.loads(path.read_text())
    assert calls == [model]
    assert census['model_load_contract'] == contract
    assert census['attention_implementation'] == 'eager'
    assert census['capture_runtime'] == dict(torch=torch.__version__,cuda=torch.version.cuda,
        transformers=importlib.metadata.version('transformers'))


def test_census_refuses_unrequested_backend(monkeypatch,tmp_path):
    tc,argv,model,contract,calls,path = fixture(monkeypatch,tmp_path)
    with pytest.raises(SystemExit,match='2'):
        tc.main(argv)
    assert not calls and not path.exists()
    model.config._attn_implementation = 'sdpa'
    with pytest.raises(RuntimeError,match='backend differs'):
        tc.main([*argv,'--attention-implementation','eager'])
    assert not path.exists()


def test_census_refuses_model_without_initialization_evidence(monkeypatch,tmp_path):
    tc,argv,model,contract,calls,path = fixture(monkeypatch,tmp_path)
    def refuse(model):
        raise ValueError('Missing or invalid pretrained initialization contract')
    monkeypatch.setattr(prismaquant,'pretrained_initialization_contract',refuse)
    with pytest.raises(ValueError,match='initialization contract'):
        tc.main([*argv,'--attention-implementation','eager'])
    assert not path.exists()
