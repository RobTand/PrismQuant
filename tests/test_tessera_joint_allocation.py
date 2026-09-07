"""CPU identity oracles for the completed-joint-table handoff; no GPU work."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from test_joint_aura_assignment_diagnostics import _row, _rebuild


def fixture(names=None):
    from prismaquant.tessera_joint_aura import PREPARED_SCHEMA

    fmt = 'TESSERA_E4M3_K1_R1024'
    names = names or ['model.layers.0.mlp.down_proj', 'model.layers.1.mlp.down_proj']
    costs, verified, cells, units = {}, {}, {}, {}
    for name in names:
        row = _row(name, [1., -2., 3., -4.], fmt=fmt)
        bf16 = _row(name, [0., 0., 0., 0.], fmt='BF16')
        bf16 = _rebuild(bf16, operator_change=lambda op: op.update(rendered_weight=copy.deepcopy(op['source_weight'])))
        def probe_context(probe):
            probe.update(calibration_shape=[2, 4], calibration_dtype='torch.int64')
            probe['source_model']['source'] = '/fixture/model'
        row = _rebuild(row, probe_change=probe_context)
        bf16 = _rebuild(bf16, probe_change=probe_context)
        costs[name] = {fmt: row, 'BF16': bf16}
        op = row['joint_operator_identity']
        record = {'blob_sha256': 'a'*64, 'identity': {'unit': name,
                  'source': {'shape': [2, 2], 'dtype': 'float32', 'sha256': op['source_weight']['content_sha256']}}}
        units[name] = {'weight': copy.deepcopy(record['identity']['source'])}
        verified[name, fmt] = {**{k: copy.deepcopy(op[k]) for k in ('source_weight', 'rendered_weight', 'activation')},
                               'wire_sha256': record['blob_sha256'], 'encoding_identity_sha256': 'b'*64}
        cells[name, fmt] = {'record': record}
    probe = costs[names[0]][fmt]['probe_identity']
    prepared_binding = {'path': '/fixture/prepared.json', 'sha256': 'c'*64}
    inputs = {'merged_cost': {'path': '/fixture/anchors.pkl', 'sha256': 'd'*64}}
    hessian = {'supplied': True, 'capture_sha256': 'e'*64,
               'calibration_identity': {'fit_ids_sha256': 'f'*64, 'seed': 0, 'nsamples': 2, 'seqlen': 4}}
    anchor_rows = {n: {fmt: {'hessian_identity': {'supplied': True, 'applied': True},
                  'tessera_family': 'TESSERA_E4M3_K1', 'tessera_body_rate_q256': 1024,
                  'wire_bytes': 16, 'input_global_scale': None, 'activation_contract': 'fp8_e4m3',
                  'activation_quantized': True, 'output_mse': 999.}} for n in names}
    provenance = {'model': '/fixture/model', 'nsamples': 2, 'seqlen': 4, 'layer_stride': 1,
                  'max_act_rows': 4, 'hessian': hessian, 'activation_static_scales': {'units': {}, 'policy': 'fixture'},
                  'wire_dir': '/fixture/wire', 'calibration_cache': {'path': '/fixture/capture', 'sha256': '1'*64}}
    data = SimpleNamespace(inputs=inputs, formats_by_qname={n: (fmt, 'BF16') for n in names},
        cells=cells, payload={'costs': anchor_rows, 'provenance': provenance},
        census={'unit_shapes': {n: [2, 2] for n in names}}, manifest={'identity': {'units': units}})
    calibration = {'calibration_sha256': probe['calibration_sha256'], 'shape': [2, 4],
                   'provenance': copy.deepcopy(hessian['calibration_identity'])}
    prepared = {'schema': PREPARED_SCHEMA, 'status': 'complete', 'plan_sha256': '2'*64,
                'source_model_identity': probe['source_model'], 'calibration_input': calibration,
                'formats_by_qname': {n: list(fmts) for n, fmts in data.formats_by_qname.items()},
                'measured_cells': len(cells), 'reader_identity': {'fixture': 'reader'},
                'projection_backend': probe['arithmetic']['projection_backend']}
    metadata = {'schema': PREPARED_SCHEMA, 'inputs': inputs, 'verified_cells': verified,
                'reader_identity': prepared['reader_identity'], 'projection_backend': prepared['projection_backend']}
    joint = {'schema': 'prismaquant.aura_cost.v1', 'stats': {n: {'h_trace': 1., 'n_params': 4,
                    'in_features': 2, 'out_features': 2} for n in names}, 'costs': costs,
             'formats': [fmt, 'BF16'], 'provenance': {'cost_mode': 'aura', 'joint_activation': True,
                'cost_currency': 'joint_aura_predicted_dloss', 'tessera_joint_anchors': {
                    'plan_sha256': prepared['plan_sha256'], 'prepared': prepared_binding,
                    'inputs': inputs, 'calibration_input': calibration, 'measured_cells': len(cells)}}}
    return joint, data, prepared, metadata, {'plan_sha256': prepared['plan_sha256'], 'prepared_binding': prepared_binding}


def test_handoff_preserves_all_joint_prices_and_identities():
    from prismaquant.tessera_joint_allocation import bind_allocation_payload
    joint, data, prepared, metadata, kwargs = fixture()
    before = copy.deepcopy(joint)
    result = bind_allocation_payload(joint, data, prepared, metadata, **kwargs)
    assert joint == before
    assert result['stats'] == before['stats']
    for name, rows in before['costs'].items():
        for fmt, row in rows.items():
            assert {key: result['costs'][name][fmt][key] for key in row} == row
            assert 'output_mse' not in result['costs'][name][fmt]
    assert result['provenance']['cost_mode'] == 'aura'
    assert result['provenance']['hessian'] == data.payload['provenance']['hessian']
    assert result['provenance']['tessera_joint_allocation']['status'] == 'research_metadata_handoff'


@pytest.mark.parametrize('mutation', ['missing_unit', 'extra_format', 'source', 'render', 'activation',
    'wire', 'calibration', 'prepared', 'shape', 'overwritten_metadata', 'nonzero_bf16'])
def test_handoff_refuses_changed_evidence(mutation):
    from prismaquant.tessera_joint_allocation import bind_allocation_payload
    joint, data, prepared, metadata, kwargs = fixture()
    name = next(iter(joint['costs'])); fmt = next(f for f in joint['costs'][name] if f != 'BF16')
    row = joint['costs'][name][fmt]
    if mutation == 'missing_unit': joint['costs'].pop(name)
    elif mutation == 'extra_format': joint['costs'][name]['FP8_E4M3'] = copy.deepcopy(row)
    elif mutation in ('source', 'render'):
        field = 'source_weight' if mutation == 'source' else 'rendered_weight'
        joint['costs'][name][fmt] = _rebuild(row, operator_change=lambda op: op[field].update(content_sha256='9'*64))
    elif mutation == 'activation': metadata['verified_cells'][name, fmt]['activation']['clip_enabled'] = not row['joint_operator_identity']['activation']['clip_enabled']
    elif mutation == 'wire': metadata['verified_cells'][name, fmt]['wire_sha256'] = '8'*64
    elif mutation == 'calibration': prepared['calibration_input']['calibration_sha256'] = '7'*64
    elif mutation == 'prepared': kwargs['prepared_binding'] = {**kwargs['prepared_binding'], 'sha256': '6'*64}
    elif mutation == 'shape': joint['stats'][name]['n_params'] = 5
    elif mutation == 'overwritten_metadata': row['wire_bytes'] = 999
    elif mutation == 'nonzero_bf16':
        from prismaquant.joint_aura import make_joint_aura_entry
        old = joint['costs'][name]['BF16']
        joint['costs'][name]['BF16'] = make_joint_aura_entry(
            operator_identity=old['joint_operator_identity'], probe_identity=old['probe_identity'],
            signed_components=[dict(weight=1., activation=0., mixed=0., total=1.) for _ in range(4)])
    from prismaquant.cost_currency import CostCurrencyError
    with pytest.raises((ValueError, CostCurrencyError)):
        bind_allocation_payload(joint, data, prepared, metadata, **kwargs)


def test_handoff_carries_complete_original_projection_and_measured_wire_roster():
    from prismaquant import tessera_expert_projection as tep
    from prismaquant.tessera_joint_allocation import bind_allocation_payload
    from test_tessera_expert_projection import _projection, _declared, STACK

    declared = _declared(experts=(0,), n=2, k=2)
    joint, data, prepared, metadata, kwargs = fixture(list(declared[STACK]))
    projection = _projection(experts=(0,), n=2, k=2)
    carried = tep.carried_projection(projection, tep.bind_expert_projection(projection, declared=declared),
        request=tep.stack_plan_request({STACK: ('E4M3', 1024)}), tool='/fixture/producer')
    data.payload['provenance'][tep.PROJECTION_KEY] = carried
    data.payload[tep.EXPERT_WIRES_KEY] = {
        name: {fmt: copy.deepcopy(cell['record']) for (unit, fmt), cell in data.cells.items() if unit == name}
        for name in joint['costs']}
    result = bind_allocation_payload(joint, data, prepared, metadata, **kwargs)
    assert result['provenance'][tep.PROJECTION_KEY] == carried
    assert result[tep.EXPERT_WIRES_KEY] == data.payload[tep.EXPERT_WIRES_KEY]
    name = next(iter(result[tep.EXPERT_WIRES_KEY]))
    data.payload[tep.EXPERT_WIRES_KEY][name] = {}
    with pytest.raises(ValueError, match='projected wire receipt'):
        bind_allocation_payload(joint, data, prepared, metadata, **kwargs)


@pytest.mark.parametrize('mutation', [None, 'joint_bytes', 'cache_bytes', 'render_path', 'output_exists', 'receipt_exists'])
def test_file_handoff_authenticates_owned_bytes_and_preserves_original(tmp_path, monkeypatch, mutation):
    import hashlib
    import json
    import pickle
    from prismaquant.production_weight_cache import ProductionWeightCache
    from prismaquant import tessera_joint_aura as bridge
    from prismaquant.tessera_joint_allocation import handoff

    joint, data, prepared, metadata, _kwargs = fixture()
    def write_bound(name, raw):
        path = tmp_path / name
        path.write_bytes(raw)
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    plan = {'schema': bridge.SCHEMA, 'inputs': data.inputs,
            'calibration_input': {'sha256': '3'*64}}
    plan_binding = write_bound('plan.json', json.dumps(plan).encode())
    prepared['plan_sha256'] = plan_binding['sha256']
    prepared['calibration_input']['artifact_sha256'] = '3'*64
    for pair, cell in data.cells.items():
        cell['render'] = '/fixture/' + pair[0] + '.pt'
    weights = {pair: cell['render'] for pair, cell in data.cells.items()}
    if mutation == 'render_path': weights[next(iter(weights))] = '/changed.pt'
    cache = ProductionWeightCache(weights=weights, levers={}, metadata=metadata)
    prepared['production_cache'] = write_bound('production.pkl', pickle.dumps(cache))
    prepared_binding = write_bound('prepared.json', json.dumps(prepared).encode())
    evidence = joint['provenance']['tessera_joint_anchors']
    evidence.update(plan_sha256=plan_binding['sha256'], prepared=prepared_binding,
                    calibration_input=prepared['calibration_input'])
    joint_raw = pickle.dumps(joint)
    joint_binding = write_bound('joint.pkl', joint_raw)
    def load_inputs(inputs, *, verify_payloads):
        assert inputs == data.inputs
        assert verify_payloads is False
        return data
    monkeypatch.setattr(bridge, 'load_measured_anchor_input', load_inputs)
    output = tmp_path / 'allocation.pkl'
    receipt_path = tmp_path / 'allocation.pkl.receipt.json'
    if mutation == 'joint_bytes': (tmp_path / 'joint.pkl').write_bytes(joint_raw + b'changed')
    elif mutation == 'cache_bytes': (tmp_path / 'production.pkl').write_bytes(b'changed')
    elif mutation == 'output_exists': output.write_bytes(b'existing')
    elif mutation == 'receipt_exists': receipt_path.write_bytes(b'existing')
    if mutation is not None:
        with pytest.raises(ValueError):
            handoff(joint_binding=joint_binding, plan_binding=plan_binding, output_path=output)
        assert output.read_bytes() == b'existing' if mutation == 'output_exists' else not output.exists()
        return
    receipt = handoff(joint_binding=joint_binding, plan_binding=plan_binding, output_path=output)
    assert (tmp_path / 'joint.pkl').read_bytes() == joint_raw
    assert hashlib.sha256(output.read_bytes()).hexdigest() == receipt['output']['sha256']
    assert json.loads(receipt_path.read_text()) == receipt
    result = pickle.loads(output.read_bytes())
    assert result['stats'] == joint['stats']
    assert receipt['joint_fields_unchanged'] and receipt['research_only']
