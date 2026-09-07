"""Selection is non-exportable; materialization preserves chosen prices and bytes."""
import copy
import hashlib
import json
import pickle
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import tessera_materialization as tm
from prismaquant import tessera_campaign as tc
from prismaquant import tessera_expert_projection as tep
from prismaquant.layer_config import canonicalize_assignment, load_assignment
from test_tessera_packed_export_scope import _pack
from test_tessera_export_projection import case, _meta, _units, FMT, STACK, N, DENSE


@pytest.fixture
def selection(case, tmp_path):
    _pack(case)
    # This materialization fixture checks actual source bytes as well as headers.
    source = case.payload['__prismaquant__'][tep.PROJECTION_KEY]['producer']['source']
    source['files'] = {name:tm._sha(case.model/name) for name in source['files']}
    source['config_sha256'] = tm._sha(case.model/'config.json')
    config = copy.deepcopy(case.payload)
    assignment = canonicalize_assignment(config)
    meta = config['__prismaquant__']
    for record in case.receipts.values():
        record['identity']['encoder_source_sha256'] = 'producer-source'
    census_identity = dict(text_sha256='a'*64, fit_ids_sha256='b'*64, fit_tokens=8,
        model=str(case.model), seed=0, nsamples=2, seqlen=4, fit_tokens_min=8,
        source='wikitext-2-raw-v1/train', split_role='calibration')
    costs = {name: {FMT: {'output_mse': 0.123}} for name in meta[tep.POPULATION_KEY]['priced']['dense']
             + meta[tep.POPULATION_KEY]['priced']['routed_experts']}
    provenance = dict(model=str(case.model), nsamples=2, seqlen=4, layer_stride=1, max_act_rows=4,
        wire_dir=str(case.wire_dir),
        hessian=dict(supplied=False, calibration_identity=census_identity, recipe={'fixture': True}),
        activation_static_scales=dict(policy='fixture', units={n: float(i+1) for i,n in enumerate(_units())}))
    provenance.update({key: meta[key] for key in (tep.POPULATION_KEY, tep.PROJECTION_KEY)})
    cost = dict(costs=costs, provenance=provenance,
                tessera_expert_wires={n: {FMT: r} for n,r in case.receipts.items()})
    # One missing unsampled expert demonstrates the actual selection/finalization gap.
    missing = f'{STACK}.1.w2'
    del cost[tep.EXPERT_WIRES_KEY][missing]
    packed = f'{STACK}.down_proj'
    cost['provenance'][tep.POPULATION_KEY]['stack_decisions'][packed]['sampled_members'] = [f'{STACK}.0.w2']
    cost_path = tmp_path / 'cost.pkl'
    cost_path.write_bytes(pickle.dumps(cost))
    request_path = tmp_path / 'request.json'
    output = tmp_path / 'final-layer-config.json'
    tm.write_selection_request(request_path, layer_config=config, assignment=assignment,
                               cost_path=cost_path, cost_payload=cost, output_path=output)
    census = dict(schema=tc.CENSUS_SCHEMA, model=str(case.model), nsamples=2, seqlen=4, seed=0,
        layer_stride=1, text_sha256='a'*64, fit_ids_sha256='b'*64,
        counts={n:8 for n in _units()}, max_abs={n:1.0 for n in _units()},
        anchor_groups={'s:'+STACK: _units()}, unit_shapes={n:[N,N] for n in _units()},
        expert_projection=cost['provenance'][tep.PROJECTION_KEY])
    census_path = tmp_path / 'census.json'
    census_path.write_text(json.dumps(census))
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps(dict(model=str(case.model), campaign_argv=[], cwd=str(tmp_path),
        python='python', env={}, cpus=1, headroom_gb=3)))
    return SimpleNamespace(case=case, config=config, assignment=assignment, cost=cost,
        cost_path=cost_path, request=request_path, output=output, missing=missing,
        census=census_path, spec=spec, workspace=tmp_path/'materialization')


def test_selection_does_not_publish_allocation_or_invent_missing_receipts(selection):
    s = selection
    assert not s.output.exists()
    request, cost, _source, _units, expanded = tm._request(s.request)
    assert request['status'] == 'selection_only_nonexportable'
    assert s.missing not in request['layer_config']['__prismaquant__'][tep.EXPERT_WIRES_KEY]
    assert expanded[s.missing] == FMT
    with pytest.raises(tep.ExpertProjectionError, match='no priced wire receipt'):
        tep.allocation_expert_projection_block(cost, s.assignment)
    with pytest.raises(ValueError):
        load_assignment(s.request)


@pytest.mark.parametrize('defect', ['cost', 'assignment', 'projection'])
def test_request_refuses_input_drift(selection, defect):
    s = selection
    if defect == 'cost':
        s.cost_path.write_bytes(s.cost_path.read_bytes() + b'changed')
    else:
        request = json.loads(s.request.read_text())
        if defect == 'assignment':
            request['assignment'][DENSE] = 'BF16'
        else:
            request['layer_config']['__prismaquant__'][tep.PROJECTION_KEY]['tool'] = 'changed'
        s.request.write_text(json.dumps(request))
    with pytest.raises(RuntimeError, match='changed'):
        tm._request(s.request)


def _plan(selection, monkeypatch):
    from tools import dispatch_tessera_campaign as dispatcher
    calls = []
    def row(spec, argv, **kwargs):
        calls.append((argv, kwargs))
        return dict(argv=argv, **kwargs)
    monkeypatch.setattr(dispatcher, '_row', row)
    result = tm.plan(selection.request, selection.census, selection.workspace, selection.spec)
    return selection.workspace / 'plan.json', result, calls


def test_plan_uses_census_quanta_and_pb_row_sizing(selection, monkeypatch):
    path, result, calls = _plan(selection, monkeypatch)
    assert len(result['groups']) == len(calls) == 1
    assert result['groups'][0]['assignment'] == {n:FMT for n in _units()}
    assert calls[0][1]['module'] == 'prismaquant.tessera_materialization'
    assert calls[0][1]['mem_gb'] >= 3
    assert (selection.workspace / 'manifest.json').is_file()
    assert not selection.output.exists()


def test_plan_refuses_different_calibration(selection, monkeypatch):
    census = json.loads(selection.census.read_text())
    census['fit_ids_sha256'] = 'different'
    selection.census.write_text(json.dumps(census))
    with pytest.raises(RuntimeError, match='different draws'):
        _plan(selection, monkeypatch)


class ReceiptAPI:
    def encoder_source_sha256(self):
        return 'producer-source'

    def unit_input_identity(self, weight, unit, grid, rung, activation=None):
        return dict(schema='tessera.cached_unit_inputs.v1', unit=unit['tensor'][:-7],
            encoder_source_sha256=self.encoder_source_sha256(),
            recipe=dict(grid=grid.name, q256=rung),
            projection={key:unit[key] for key in tep.UNIT_IDENTITY_KEYS})

    def verify_cached_unit(self, blob, record, identity):
        assert record['identity'] == identity
        assert hashlib.sha256(blob).hexdigest() == record['blob_sha256']
        assert len(blob) == record['blob_bytes']


def _completed(selection, monkeypatch, *, hessian=False):
    from prismaquant.cost_stage_checkpoint import write_unit
    from prismaquant import tessera_hessian as th
    if hessian:
        s = selection
        calibration = s.cost['provenance']['hessian']['calibration_identity']
        original = s.workspace / 'original'
        original.mkdir(parents=True)
        capture, _scales, digest = tc.write_export_inputs(original,
            hessians={s.missing:torch.eye(N)}, hessian_rows={s.missing:8},
            hessian_identity=calibration, static_scales={}, static_scale_policy='fixture')
        s.cost['provenance']['hessian'].update(supplied=True, capture_path=str(capture), capture_sha256=digest)
        s.config['__prismaquant__']['tessera_hessian'].update(supplied=True,
            **calibration, capture_sha256=digest)
        s.cost_path.write_bytes(pickle.dumps(s.cost))
        tm.write_selection_request(s.request, layer_config=s.config, assignment=s.assignment,
            cost_path=s.cost_path, cost_payload=s.cost, output_path=s.output)
    path, data, _calls = _plan(selection, monkeypatch)
    monkeypatch.setattr(tc, '_checkpoint_identity_api', lambda:ReceiptAPI())
    monkeypatch.setattr(th, 'encoder_recipe', lambda:{'fixture':True})
    monkeypatch.setattr(tm, '_verify_source', lambda *args:None)
    monkeypatch.setattr(tep, 'source_unit_weight', lambda *args:torch.ones(N,N))
    monkeypatch.setattr(th, 'activation_source', lambda *args:None)
    from prismaquant import tessera_render
    monkeypatch.setattr(tessera_render, 'rung_accepts_hessian', lambda *args:True)
    *_, identity, group = tm._inputs(path, 0)
    root, (journal, digest, _completed) = tm._journal(data, identity, 0, sorted(_units()))
    wires = root/'wire'
    wires.mkdir()
    for name, record in selection.case.receipts.items():
        (wires/record['file']).write_bytes((selection.case.wire_dir/record['file']).read_bytes())
        write_unit(journal, stage=tm.STAGE, qname=name, identity_sha256=digest,
                   state=dict(format=FMT, record=record, anchor=None))
    calibration = selection.cost['provenance']['hessian']['calibration_identity']
    capture_digest = None
    if hessian:
        _capture, _scales, capture_digest = tc.write_export_inputs(root,
            hessians={n:torch.eye(N) for n in _units()}, hessian_rows={n:8 for n in _units()},
            hessian_identity=calibration, static_scales={}, static_scale_policy='fixture')
    tm._json(root/'complete.json', dict(identity=identity, group=0, units=sorted(_units()),
        calibration_identity=calibration, capture_sha256=capture_digest))
    return path, root


def test_finalize_preserves_packed_decisions_and_reuses_selected_wire_bytes(selection, monkeypatch):
    path, root = _completed(selection, monkeypatch)
    report = tm.finalize(path)
    assert report['verified_source_units'] == len(_units())
    assert load_assignment(selection.output) == selection.assignment
    meta = json.loads(selection.output.read_text())['__prismaquant__']
    assert set(meta[tep.EXPERT_WIRES_KEY]) == set(_units())
    assert meta['tessera_selected_wire_materialization']['selection_prices'] == 'unchanged_sampled_stack_estimates'
    assert pickle.loads(selection.cost_path.read_bytes()) == selection.cost
    for name, record in selection.case.receipts.items():
        assert (selection.workspace/'final'/'wire'/record['file']).read_bytes() == (
            selection.case.wire_dir/record['file']).read_bytes()
    assert tm.finalize(path) == report


@pytest.mark.parametrize('defect', ['wire', 'completion', 'producer', 'recipe', 'missing_unit'])
def test_finalize_refuses_incomplete_or_contradictory_evidence(selection, monkeypatch, defect):
    from prismaquant.cost_stage_checkpoint import unit_path
    path, root = _completed(selection, monkeypatch)
    if defect == 'wire':
        record = selection.case.receipts[selection.missing]
        (root/'wire'/record['file']).write_bytes(b'corrupt')
    elif defect == 'completion':
        data = json.loads((root/'complete.json').read_text())
        data['units'].remove(selection.missing)
        tm._json(root/'complete.json', data)
    elif defect == 'producer':
        monkeypatch.setattr(ReceiptAPI, 'encoder_source_sha256', lambda self:'changed')
    elif defect == 'recipe':
        monkeypatch.setattr(tc.th, 'encoder_recipe', lambda:{'fixture':False})
    else:
        unit_path(root/'journal', selection.missing).unlink()
    with pytest.raises((RuntimeError, AssertionError), match='changed|incomplete|match|recipe'):
        tm.finalize(path)
    assert not selection.output.exists()


def test_finalize_unions_hessians_and_requires_exact_priced_overlap(selection, monkeypatch):
    path, root = _completed(selection, monkeypatch, hessian=True)
    result = tm.finalize(path)
    capture = torch.load(result['hessian'], map_location='cpu', weights_only=False)
    assert set(capture['H']) == set(_units())
    assert set(capture['counts']) == set(_units())
    selection.output.unlink()
    group_capture = root/'hessian_capture.pt'
    payload = torch.load(group_capture, weights_only=False)
    payload['H'][selection.missing] += 1
    _capture, _scales, digest = tc.write_export_inputs(root, hessians=payload['H'],
        hessian_rows=payload['counts'], hessian_identity=payload['provenance'],
        static_scales={}, static_scale_policy='fixture')
    complete = json.loads((root/'complete.json').read_text())
    complete['capture_sha256'] = digest
    tm._json(root/'complete.json', complete)
    with pytest.raises(RuntimeError, match='disagrees with priced overlap'):
        tm.finalize(path)
    assert not selection.output.exists()


def test_complete_selected_stack_needs_no_materialization_quantum(selection, monkeypatch):
    s = selection
    s.cost[tep.EXPERT_WIRES_KEY][s.missing] = {FMT:s.case.receipts[s.missing]}
    s.cost_path.write_bytes(pickle.dumps(s.cost))
    tm.write_selection_request(s.request, layer_config=s.config, assignment=s.assignment,
        cost_path=s.cost_path, cost_payload=s.cost, output_path=s.output)
    _path, result, calls = _plan(s, monkeypatch)
    assert result['groups'] == calls == []


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('reuse', [False, True])
def test_run_encodes_only_missing_selection_and_resume_verifies_existing_bytes(selection, monkeypatch, batch_size, reuse):
    from transformers import AutoModelForCausalLM
    from prismaquant import tessera_hessian as th
    s = selection
    expected_missing = [s.missing]
    if batch_size == 2:
        expected_missing.append(f'{STACK}.1.w3')
        del s.cost[tep.EXPERT_WIRES_KEY][expected_missing[-1]]
        s.cost_path.write_bytes(pickle.dumps(s.cost))
        tm.write_selection_request(s.request, layer_config=s.config, assignment=s.assignment,
            cost_path=s.cost_path, cost_payload=s.cost, output_path=s.output)
    if reuse:
        from prismaquant import tessera_calibration_cache as cc
        census = json.loads(s.census.read_text())
        from test_tessera_calibration_cache import canonical_fields
        import prismaquant
        census.update(canonical_fields())
        s.census.write_text(json.dumps(census))
        monkeypatch.setattr(prismaquant,'pretrained_initialization_contract',lambda model:canonical_fields()['model_load_contract'])
        cache_identity = cc.capture_identity(s.census,
            calibration=s.cost['provenance']['hessian']['calibration_identity'],max_act_rows=4,
            model_load_contract=canonical_fields()['model_load_contract'],attention_implementation='eager')
        record = cc.publish_capture(s.workspace/'capture',census_path=s.census,
            identity=cache_identity,acts={n:torch.ones(4,N) for n in _units()},
            hessians={n:torch.eye(N) for n in _units()},counts=census['counts'],maxima=census['max_abs'])
        s.cost['provenance']['calibration_cache'] = record
        s.cost_path.write_bytes(pickle.dumps(s.cost))
        tm.write_selection_request(s.request,layer_config=s.config,assignment=s.assignment,
            cost_path=s.cost_path,cost_payload=s.cost,output_path=s.output)
    path, _data, _calls = _plan(s, monkeypatch)
    api = ReceiptAPI()
    api.make_unit_record = lambda blob, identity, filename: dict(file=filename,
        blob_bytes=len(blob), blob_sha256=hashlib.sha256(blob).hexdigest(), identity=identity)
    monkeypatch.setattr(tc, '_checkpoint_identity_api', lambda:api)
    monkeypatch.setattr(th, 'encoder_recipe', lambda:{'fixture':True})
    monkeypatch.setattr(tm, '_verify_source', lambda *args:None)
    weight = torch.ones(N,N)
    monkeypatch.setattr(tep, 'source_unit_weight', lambda *args:weight)
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained',
                        lambda *args, **kwargs:SimpleNamespace(eval=lambda:None,config=SimpleNamespace(_attn_implementation="eager")))
    monkeypatch.setattr(tc, '_require_campaign_population', lambda *args:SimpleNamespace(
        members=[SimpleNamespace(qname=n, weight=weight) for n in _units()]))
    monkeypatch.setattr(tc, '_calibration_tokens', lambda *args:(torch.ones(2,4), 'fixture'))
    calibration = s.cost['provenance']['hessian']['calibration_identity']
    monkeypatch.setattr(th, 'calibration_identity', lambda *args, **kwargs:calibration)
    captures = []
    def collect(model, targets, tokens, max_rows, device, **kwargs):
        assert not reuse, 'materialization repeated the calibration forward'
        captures.append((list(targets), max_rows))
        return ({n:torch.ones(4,N) for n in targets}, {}, {n:8 for n in targets}, {n:1.0 for n in targets})
    monkeypatch.setattr(tc, '_collect_activations', collect)
    monkeypatch.setattr(tc, '_static_input_scales', lambda *args, **kwargs:(
        s.cost['provenance']['activation_static_scales']['units'], 'fixture'))
    measured = []
    def measure(**kwargs):
        name = kwargs['qname']
        measured.append((name, kwargs['format_name']))
        record = s.case.receipts[name]
        (kwargs['wire_dir']/record['file']).write_bytes((s.case.wire_dir/record['file']).read_bytes())
        return tc.CampaignAnchor(qname=name, family='TESSERA_E4M3_K1', format_name=FMT,
            body_rate_q256=1024, dloss=0.1, dloss_stderr=0.0, memory_bytes=128,
            bits_per_param=4, activation_contract='fixture', activation_quantized=True,
            wire_bytes=record['blob_bytes'], seconds=0.0)
    monkeypatch.setattr(tc, '_measure_anchor', measure)
    batches = []
    def batch(**kwargs):
        batches.append(kwargs['qnames'])
        return [measure(qname=n, format_name=kwargs['format_name'], wire_dir=kwargs['wire_dir'])
                for n in kwargs['qnames']]
    monkeypatch.setattr(tc, '_measure_anchor_batch', batch)
    from prismaquant import tessera_render
    monkeypatch.setattr(tessera_render, 'require_tessera_batch_encoder', lambda:None)
    tm.run(path, 0, anchor_batch_size=batch_size)
    assert measured == [(n,FMT) for n in expected_missing]
    assert batches == ([] if batch_size == 1 else [expected_missing])
    assert captures == ([] if reuse else [(sorted(_units()),4)])
    assert not s.output.exists()
    tm.run(path, 0, anchor_batch_size=1)
    assert measured == [(n,FMT) for n in expected_missing]
    report = tm.finalize(path)
    assert report['verified_source_units'] == len(_units())


    if reuse:
        with open(record['path'],'a') as handle:
            handle.write(' ')
        with pytest.raises(RuntimeError,match='priced calibration capture manifest changed'):
            tm.run(path,0)

def test_real_producer_wire_materialization_and_translator_handoff(tmp_path, monkeypatch):
    """Real source/capture/encoder/receipt/translator; no served-route scoring."""
    import test_tessera_campaign_packed as packed_fixture
    from test_tessera_campaign_packed import _bridge_main_fixture, RUNG, STACK as REAL_STACK
    monkeypatch.setattr(packed_fixture, 'EXPERTS', 3)
    monkeypatch.setattr(packed_fixture._Router, 'forward', lambda self, hidden:
        torch.nn.functional.one_hot(torch.arange(len(hidden)) % 3, 3).to(hidden.dtype))
    from prismaquant import tessera_export_lane as export
    from prismaquant import tessera_hessian as th
    campaign, argv, model, encoded = _bridge_main_fixture(monkeypatch, tmp_path)
    measured_anchors = {}
    real_measure = campaign._measure_anchor
    def remember(**kwargs):
        anchor = real_measure(**kwargs)
        measured_anchors.setdefault(anchor.qname, {}).setdefault(anchor.family, []).append(anchor)
        return anchor
    monkeypatch.setattr(campaign, '_measure_anchor', remember)
    assert campaign.main(argv) == 0
    census_path = tmp_path/'census.json'
    assert campaign.main([*argv, '--census-out', str(census_path)]) == 0
    cost_path = tmp_path/'cost.pkl'
    cost = pickle.loads(cost_path.read_bytes())
    # The existing CPU bridge fixture replaces only the hessian-status report.
    cost['provenance']['hessian']['recipe'] = th.encoder_recipe()
    from test_tessera_stack_sample_cost import _packed_probe_row
    from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile
    profile = Lfm2MoeProfile()
    population = campaign._require_campaign_population(model, profile, 1)
    samples = {}
    for packed, shape in population.packed_in_scope.items():
        stats = _packed_probe_row(shape[0], [1.0]*shape[0],
            packed_param=packed.rsplit('.',1)[-1], out_features=shape[1], in_features=shape[2])
        stats['_packed_experts_module'] = REAL_STACK
        samples[packed] = campaign.stack_sample_from_probe(packed, stats, profile,
            sampled_experts=[0,1], inclusion_prob={0:2/3, 1:2/3, 2:2/3}, seed=0)
    retained = {n for sample in samples.values() for members in sample.members.values() for n in members}
    dense = 'model.layers.0.attention'
    measured = {n:rows for n,rows in measured_anchors.items() if n == dense or n in retained}
    prices = campaign.campaign_cost_payload(measured, {}, loo={},
        provenance={'provenance':cost['provenance']}, stack_samples=samples)
    prices['provenance'][tep.POPULATION_KEY] = campaign.campaign_population_block(
        dense_targets=[dense], expert_targets=population.qnames, dense_all=[dense], pinned=[],
        population=population, layer_stride=1, costs=prices['costs'],
        menus={n:[RUNG] for n in prices['costs']}, stack_samples=samples, profile=profile)
    missing = f'{REAL_STACK}.2.w2'
    original_record = cost[tep.EXPERT_WIRES_KEY][missing][RUNG]
    missing_names = set(cost[tep.EXPERT_WIRES_KEY]) - retained
    prices[tep.EXPERT_WIRES_KEY] = {n:rows for n,rows in cost[tep.EXPERT_WIRES_KEY].items() if n in retained}
    cost = prices
    cost_path.write_bytes(pickle.dumps(cost))
    assignment = {name:RUNG for name in cost['costs']}
    config = {name:dict(data_type='tessera', bits=4, tessera_format=fmt)
              for name,fmt in assignment.items()}
    config['__prismaquant__'] = {'tessera_hessian':{'supplied':False}}
    request = tmp_path/'request.json'
    output = tmp_path/'allocation.json'
    tm.write_selection_request(request, layer_config=config, assignment=assignment,
        cost_path=cost_path, cost_payload=cost, output_path=output)
    spec = tmp_path/'spec.json'
    spec.write_text(json.dumps(dict(model=argv[1], campaign_argv=[], cwd=str(tmp_path),
        python='python', env={}, cpus=1, headroom_gb=3)))
    workspace = tmp_path/'materialization'
    tm.plan(request, census_path, workspace, spec)
    before = len(encoded)
    tm.run(workspace/'plan.json', 0)
    assert len(encoded) == before+len(missing_names)
    assert {row[0] for row in encoded[before:]} == missing_names
    tm.finalize(workspace/'plan.json')
    metadata = json.loads(output.read_text())['__prismaquant__']
    assert metadata[tep.EXPERT_WIRES_KEY][missing] == original_record
    assert load_assignment(output) == assignment
    import os
    import runpy
    from pathlib import Path
    from safetensors import safe_open
    shapes = {}
    for shard in Path(argv[1]).glob('*.safetensors'):
        with safe_open(str(shard), framework='pt', device='cpu') as handle:
            shapes.update({name:tuple(handle.get_slice(name).get_shape()) for name in handle.keys()})
    producer = runpy.run_path(str(Path(os.environ['TESSERA_REPO']) / 'experiments/plan_from_layer_config.py'))
    derived = export._write_plan_assignment(output, expected_sha256=tm._sha(output))
    assert load_assignment(output) == assignment
    producer_config = json.loads(Path(derived['plan_assignment']).read_text())
    translated, _provenance = producer['build'](producer_config, shapes,
        cover='as-allocated', allow_disagreement=False, prismaquant=None, with_control=False)
    assert translated[missing + '.weight'] == {'grid':'E4M3', 'q256':1024}


def test_real_allocator_routes_opt_in_request_before_strict_wire_gate(tmp_path, monkeypatch):
    """Use the established scoped packed solve to exercise the CLI branch."""
    import sys
    from pathlib import Path
    from prismaquant import allocator
    from test_tessera_stack_driver_integration import test_a_stack_payload_allocates_through_the_real_allocator
    original_main = allocator.main
    seen = []
    def record(path, **kwargs):
        assert canonicalize_assignment(kwargs['layer_config']) == kwargs['assignment']
        assert Path(kwargs['cost_path']).is_file()
        assert not Path(kwargs['output_path']).exists()
        seen.append((path, kwargs))
    monkeypatch.setattr(tm, 'write_selection_request', record)
    def both_paths():
        original_argv = list(sys.argv)
        sys.argv += ['--tessera-materialization-plan', str(tmp_path/'selection-request.json')]
        original_main()
        assert len(seen) == 1
        assert not (tmp_path/'layer.json').exists()
        sys.argv[:] = original_argv
        original_main()
    monkeypatch.setattr(allocator, 'main', both_paths)
    test_a_stack_payload_allocates_through_the_real_allocator(tmp_path, monkeypatch)
    assert seen[0][0] == str(tmp_path/'selection-request.json')


@pytest.mark.parametrize('field', ['model.safetensors', 'config.json', 'tokenizer.json'])
def test_source_verification_refuses_changed_checkpoint_files(tmp_path, field):
    source = {'files':{}, 'config_sha256':'', 'auxiliary_sha256':{}}
    for name in ('model.safetensors', 'config.json', 'tokenizer.json'):
        (tmp_path/name).write_bytes(name.encode())
    source['files']['model.safetensors'] = tm._sha(tmp_path/'model.safetensors')
    source['config_sha256'] = tm._sha(tmp_path/'config.json')
    source['auxiliary_sha256']['tokenizer.json'] = tm._sha(tmp_path/'tokenizer.json')
    tm._verify_source(tmp_path, source)
    (tmp_path/field).write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='changed'):
        tm._verify_source(tmp_path, source)


def test_plan_refuses_missing_geometry_before_pb_memory_sizing(selection, monkeypatch):
    census = json.loads(selection.census.read_text())
    del census['unit_shapes'][selection.missing]
    selection.census.write_text(json.dumps(census))
    with pytest.raises(RuntimeError, match='shape does not match'):
        _plan(selection, monkeypatch)


def test_finalize_refuses_original_capture_count_drift(selection, monkeypatch):
    path, _root = _completed(selection, monkeypatch, hessian=True)
    original = selection.cost['provenance']['hessian']['capture_path']
    capture = torch.load(original, weights_only=False)
    capture['counts'][selection.missing] += 1
    torch.save(capture, original)
    with pytest.raises(RuntimeError, match='census disagrees'):
        tm.finalize(path)
    assert not selection.output.exists()


def test_materialization_refuses_unbound_producer_source(selection, monkeypatch):
    s = selection
    s.cost[tep.EXPERT_WIRES_KEY] = {}
    s.cost_path.write_bytes(pickle.dumps(s.cost))
    tm.write_selection_request(s.request, layer_config=s.config, assignment=s.assignment,
        cost_path=s.cost_path, cost_payload=s.cost, output_path=s.output)
    path, _data, _calls = _plan(s, monkeypatch)
    monkeypatch.setattr(tc, '_checkpoint_identity_api', lambda:ReceiptAPI())
    with pytest.raises(RuntimeError, match='no priced producer receipt'):
        tm._inputs(path)
