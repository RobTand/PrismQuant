"""A reusable calibration keeps full-H/prefix-X bytes and refuses stale inputs."""
import copy
import importlib.metadata
import prismaquant
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from prismaquant import tessera_calibration_cache as cc


def canonical_fields():
    return dict(model_load_contract=dict(schema='prismaquant.pretrained_initialization.v1',
        scope='checkpoint_missing_state',status='completed',
        transformers_version=importlib.metadata.version('transformers')),
        attention_implementation='eager',capture_runtime=dict(torch=torch.__version__,
        cuda=torch.version.cuda,transformers=importlib.metadata.version('transformers')))


def identity(path,calibration,max_act_rows):
    fields = canonical_fields()
    return cc.capture_identity(path,calibration=calibration,max_act_rows=max_act_rows,
        model_load_contract=fields['model_load_contract'],attention_implementation='eager')


@pytest.fixture
def capture(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    (source/'config.json').write_text('{}')
    (source/'model.safetensors').write_bytes(b'bounded source fixture')
    census = dict(model=str(source),counts={'a':5,'b':7},max_abs={'a':4.,'b':8.},
                  unit_shapes={'a':[3,2],'b':[3,2]},layer_stride=1,
                  anchor_groups={'u:a':['a'],'u:b':['b']},**canonical_fields())
    path = tmp_path/'census.json'
    path.write_text(json.dumps(census))
    capture_id = identity(path,calibration={'fit_ids_sha256':'draw'},max_act_rows=2)
    acts = {'a':torch.tensor([[1.,2.],[3.,4.]]),'b':torch.ones(2,2)}
    hessians = {'a':torch.eye(2)*13,'b':torch.eye(2)*25}
    root = tmp_path/'capture'
    record = cc.publish_capture(root,census_path=path,identity=capture_id,acts=acts,
        hessians=hessians,counts=census['counts'],maxima=census['max_abs'])
    return root,path,census,capture_id,acts,hessians,record


def test_prefetch_only_selected_and_preserves_full_h_and_prefix_precision(capture):
    root,path,census,identity,acts,hessians,record = capture
    (root/'inputs/b.pt').unlink()  # unrelated units are not transferred by this quantum
    values,_ = cc.prefetch_capture(record['path'],expected_identity=identity,
        census=census,names=['a'],device='cpu',expected_sha256=record['sha256'])
    assert set(values[0]) == {'a'}
    assert torch.equal(values[0]['a'],acts['a'])
    assert torch.equal(values[1]['a'],hessians['a'])
    assert values[2] == {'a':5}


@pytest.mark.parametrize('change',['artifact','manifest','source','draw','geometry','scope'])
def test_prefetch_refuses_drift(capture,change):
    root,path,census,identity,acts,hessians,record = capture
    expected = copy.deepcopy(identity)
    if change == 'artifact':
        with (root/'inputs/a.pt').open('ab') as f:
            f.write(b'changed')
    elif change == 'manifest':
        with open(record['path'],'a') as f:
            f.write(' ')
    elif change == 'source':
        (path.parent/'source/model.safetensors').write_bytes(b'changed')
        expected = globals()['identity'](path,calibration=identity['calibration'],max_act_rows=2)
    elif change == 'draw':
        expected['calibration']['fit_ids_sha256'] = 'other'
    elif change == 'geometry':
        expected['units']['a'] = [4,2]
    else:
        expected['units'].pop('b')
    with pytest.raises(RuntimeError,match='checksum|identity|manifest changed'):
        cc.prefetch_capture(record['path'],expected_identity=expected,census=census,
            names=['a'],device='cpu',expected_sha256=record['sha256'])


def test_completed_journal_revalidates_artifacts(capture):
    root,path,census,identity,acts,hessians,record = capture
    again = cc.publish_capture(root,census_path=path,identity=identity,acts=acts,
        hessians=hessians,counts=census['counts'],maxima=census['max_abs'])
    assert again == record
    (root/'inputs/a.pt').unlink()
    with pytest.raises(FileNotFoundError):
        cc.publish_capture(root,census_path=path,identity=identity,acts=acts,
            hessians=hessians,counts=census['counts'],maxima=census['max_abs'])


def test_layer_writer_requires_full_scope_and_actual_witness(capture, tmp_path):
    _root, path, census, identity, acts, hessians, _record = capture
    root = tmp_path/'layer-writer'
    writer = cc.CaptureWriter(root, census_path=path, identity=identity)
    def write(name):
        writer.write(acts={name: acts[name]}, hessians={name: hessians[name]},
            counts={name: census['counts'][name]}, maxima={name: census['max_abs'][name]})
    write('a')
    assert (root/'inputs/a.pt').is_file()
    assert not (root/'capture_manifest.json').exists()
    with pytest.raises(RuntimeError, match='full census'):
        writer.finish(model_load_contract=identity['model_load_contract'])
    write('b')
    wrong = dict(identity['model_load_contract'], transformers_version='wrong-runtime')
    with pytest.raises(RuntimeError, match='actual capture initialization'):
        writer.finish(model_load_contract=wrong)
    assert not (root/'capture_manifest.json').exists()
    record = writer.finish(model_load_contract=identity['model_load_contract'])
    assert cc.require_capture_contract(record['path'])['status'] == 'complete'


def test_layer_writer_replay_refuses_changed_hessian(capture, tmp_path):
    _root, path, census, identity, acts, hessians, _record = capture
    root = tmp_path/'interrupted'
    writer = cc.CaptureWriter(root, census_path=path, identity=identity)
    writer.write(acts={'a':acts['a']}, hessians={'a':hessians['a']},
                 counts={'a':5}, maxima={'a':4.})
    resumed = cc.CaptureWriter(root, census_path=path, identity=identity)
    with pytest.raises(RuntimeError, match='replayed capture differs'):
        resumed.write(acts={'a':acts['a']}, hessians={'a':hessians['a']+torch.eye(2)},
                      counts={'a':5}, maxima={'a':4.})
    assert not (root/'capture_manifest.json').exists()


def test_layer_writer_refuses_insufficient_disk(capture, tmp_path, monkeypatch):
    import shutil
    _root, path, census, identity, acts, hessians, _record = capture
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    with pytest.raises(RuntimeError, match='additional disk bytes'):
        cc.CaptureWriter(tmp_path/'no-space', census_path=path, identity=identity)


def test_cli_capture_then_reuse_never_repeats_forward(monkeypatch,tmp_path):
    from test_tessera_campaign_resume import _main_fixture,UNIT
    tc,_,argv,model,inputs = _main_fixture(monkeypatch,tmp_path)
    model.config = SimpleNamespace(_attn_implementation='eager')
    monkeypatch.setattr(prismaquant,'pretrained_initialization_contract',lambda model:canonical_fields()['model_load_contract'])
    argv += ['--attention-implementation','eager']
    source = tmp_path/'source'
    source.mkdir()
    (source/'config.json').write_text('{}')
    (source/'model.safetensors').write_bytes(b'fixture')
    argv[argv.index('--model')+1] = str(source)
    argv[argv.index('--hessian')+1] = 'require'
    calibration = tc.th.calibration_identity(inputs['text'],inputs['tokens'],fit_tokens=4,
        source='wikitext-2-raw-v1/train',split_role='calibration',model=str(source),
        seed=0,nsamples=32,seqlen=512,fit_tokens_min=4)
    census = tc.calibration_census({UNIT:4},{UNIT:3.},args=SimpleNamespace(model=str(source),
        nsamples=32,seqlen=512,seed=0,layer_stride=1),groups={'u:'+UNIT:[UNIT]},
        dense_targets=[UNIT],expert_targets=[],shapes={UNIT:[32,256]},identity=calibration,**canonical_fields())
    census_path = tmp_path/'census.json'
    census_path.write_text(json.dumps(census))
    argv += ['--nsamples','32','--seqlen','512','--layer-stride','1',
             '--calibration-census',str(census_path)]
    root = tmp_path/'full-capture'
    assert tc.main([*argv,'--capture-calibration-out',str(root)]) == 0
    monkeypatch.setattr(tc,'_collect_activations',lambda *a,**k: pytest.fail('repeated forward'))
    assert tc.main([*argv,'--capture-calibration-out',str(root)]) == 0
    assert tc.main([*argv,'--calibration-cache',str(root/'capture_manifest.json')]) == tc.EXIT_EMPTY_MENU


def test_driver_capture_and_plan_bind_one_complete_capture(capture):
    from tools import dispatch_tessera_campaign as dispatch
    root,path,census,identity,acts,hessians,record = capture
    spec = path.parent/'spec.json'
    spec.write_text(json.dumps(dict(model=census['model'],campaign_argv=[],
        cwd=str(path.parent),python='python3',env={},cpus=1,headroom_gb=2)))
    common = ['--spec',str(spec),'--workspace',str(path.parent)]
    assert dispatch.main(['capture',*common]) == 0
    quantum = json.loads((path.parent/'capture-manifest.json').read_text())
    assert len(quantum) == 1
    assert '--capture-calibration-out' in quantum[0]['argv']
    assert '--units' not in quantum[0]['argv']
    assert dispatch.main(['plan',*common,'--calibration-cache',record['path']]) == 0
    rows = json.loads((path.parent/'manifest.json').read_text())
    for row in rows:
        argv = row['argv']
        assert argv[argv.index('--calibration-cache')+1] == record['path']
        assert argv[argv.index('--calibration-cache-sha256')+1] == record['sha256']
    manifest = json.loads(Path(record['path']).read_text())
    manifest['status'] = 'partial'
    Path(record['path']).write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match='complete.*capture'):
        dispatch.main(['plan',*common,'--calibration-cache',record['path']])


def test_merge_refuses_disagreeing_capture_bindings():
    from test_tessera_campaign_fanout import _payloads
    from tools import dispatch_tessera_campaign as dispatch
    payloads = _payloads()
    for index,payload in enumerate(payloads.values()):
        payload['provenance']['calibration_cache'] = dict(path='/capture.json',sha256=str(index))
    with pytest.raises(dispatch.MergeRefused,match='calibration_cache'):
        dispatch.merge_payloads(payloads,census={'counts':{'a':16384,'b':16384}},capture_sha256='merged')


@pytest.mark.parametrize('field',['model_load_contract','attention_implementation','capture_runtime'])
def test_identity_refuses_legacy_or_changed_census(capture,field):
    root,path,census,capture_id,acts,hessians,record = capture
    census.pop(field)
    path.write_text(json.dumps(census))
    with pytest.raises((RuntimeError,ValueError),match='initialization|runtime|attention'):
        identity(path,calibration=capture_id['calibration'],max_act_rows=2)


@pytest.mark.parametrize('field',['model_load_contract','attention_implementation','capture_runtime','schema'])
def test_downstream_contract_refuses_unqualified_identity(capture,field):
    root,path,census,capture_id,acts,hessians,record = capture
    manifest = json.loads(Path(record['path']).read_text())
    manifest['identity'].pop(field)
    Path(record['path']).write_text(json.dumps(manifest))
    with pytest.raises((RuntimeError,ValueError),match='initialization|runtime'):
        cc.require_capture_contract(record['path'])


@pytest.mark.parametrize('kwargs_route',[False,True])
def test_boundary_callback_preserves_raw_arguments_and_reservoir(kwargs_route):
    from test_tessera_campaign_packed import _RoutedModel,EXPERT_PREFIX
    from prismaquant.production_weight_cache import _PackedExpertActivationCollector
    model = _RoutedModel().to(torch.bfloat16)
    module = model.model.layers[2].feed_forward.experts
    x = torch.arange(32,dtype=torch.bfloat16).reshape(8,4)/32
    indices = torch.arange(8).remainder(2).reshape(8,1)
    weights = torch.ones(8,1,dtype=torch.bfloat16)
    snapshots = []
    for callback in (None,lambda name,actual,args,kwargs:snapshots.append((name,actual,args,kwargs))):
        collector = _PackedExpertActivationCollector(model,{EXPERT_PREFIX},module_token_budget=3,
            store_device='cpu',boundary_consumer=callback)
        collector.install()
        try:
            if kwargs_route:
                module(x,indices=indices,weights=weights)
            else:
                module(x,indices,weights)
        finally:
            collector.remove()
        sampled = torch.cat(collector.activations[EXPERT_PREFIX])
        if callback is None:
            original = sampled
        else:
            assert torch.equal(original,sampled)
    name,actual,args,kwargs = snapshots[0]
    assert name == EXPERT_PREFIX and actual is module and args[0] is x
    assert (kwargs['indices'] if kwargs_route else args[1]) is indices
    assert (kwargs['weights'] if kwargs_route else args[2]) is weights
    assert not module._forward_pre_hooks


def test_campaign_boundary_coordinates_follow_batched_calibration_ids():
    from test_tessera_campaign_packed import _RoutedModel,EXPERT_PREFIX
    from prismaquant import tessera_campaign as tc
    class TokenModel(_RoutedModel):
        def forward(self,ids):
            hidden = torch.stack((ids.float()*2-1,ids.float(),ids.float()+1,ids.float()-1),dim=-1)
            return super().forward(hidden)
    model = TokenModel()
    tokens = [torch.tensor([[0,1],[1,0]]),torch.tensor([[0,1]])]
    captures = []
    values = tc._collect_activations(model,[EXPERT_PREFIX+'.0.w1',EXPERT_PREFIX+'.1.w1'],
        tokens,2,'cpu',want_hessian=True,
        boundary_consumer=lambda name,module,args,kwargs,coords:captures.append((args[0].clone(),coords.clone())))
    assert len(captures) == 2
    assert captures[0][1].tolist() == [[0,0],[0,1],[1,0],[1,1]]
    assert captures[1][1].tolist() == [[2,0],[2,1]]
    assert captures[0][0][:,0].tolist() == [-1,1,1,-1]
    assert all(count == 3 for count in values[2].values())
