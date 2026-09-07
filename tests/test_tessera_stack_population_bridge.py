"""Packed campaign decisions retain honest source coverage and priced receipts."""
from __future__ import annotations

import copy
import json
import pickle
import sys

import pytest

from prismaquant import tessera_campaign as campaign
from prismaquant import tessera_expert_projection as tep
from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile
from test_allocator_expert_projection import (
    _receipt, _v5_contract, _allocator_argv, DENSE, FMT, N, STACK, SHARD,
)


def _fixture(tmp_path, *, sampled=False):
    profile = Lfm2MoeProfile()
    parameters = {f'{STACK}.gate_up_proj': (4, 2*N, N),
                  f'{STACK}.down_proj': (4, N, N)}
    declared, tensors, producer_units = {}, {}, []
    for expert in range(4):
        for role, projection, group in (('w1', 'gate_proj', 'w13'),
                                        ('w3', 'up_proj', 'w13'),
                                        ('w2', 'down_proj', 'w2')):
            name = f'{STACK}.{expert}.{role}'
            tensor = name + '.weight'
            declared[name] = (N, N)
            tensors[tensor] = SHARD
            producer_units.append(dict(tensor=tensor, source_tensor=tensor,
                                       source_layout=tep.SOURCE_LAYOUT_UNPACKED,
                                       source_slice=dict(expert=expert, selector='whole', transpose=False),
                                       expert=expert, projection=projection, group=group,
                                       rows=N, cols=N))
    producer = dict(schema=tep.PROJECTION_SCHEMA,
                    stacks={STACK: dict(source_layout=tep.SOURCE_LAYOUT_UNPACKED, grid='E4M3',
                                       q256=1024, experts=4, units=producer_units)},
                    source=dict(config_sha256='c'*64, auxiliary_sha256={},
                                files={SHARD: 'f'*64}, tensors=tensors))
    carried = tep.carried_projection(producer,
        tep.bind_expert_projection(producer, declared={STACK: declared}),
        request=tep.stack_plan_request({STACK: ('E4M3', 1024)}), tool='fixture')
    population = campaign.ExpertPopulation(members=(), declared={STACK: declared},
        packed_in_scope=parameters, omitted_outside_layer_stride={})
    stats, samples = {}, {}
    for packed, shape in parameters.items():
        row = dict(h_trace=4.0, h_trace_per_expert=[1.0]*4, num_experts=4,
                   _packed_experts_module=STACK, _packed_param=packed.rsplit('.', 1)[1],
                   out_features=shape[1], in_features=shape[2], n_params=4*shape[1]*shape[2],
                   router_path=None, expert_id=None)
        stats[packed] = row
        samples[packed] = campaign.stack_sample_from_probe(packed, row, profile,
            sampled_experts=(0, 1) if sampled else range(4),
            inclusion_prob={e: .5 if sampled else 1.0 for e in range(4)}, seed=0)
    costs = {name: {FMT: dict(output_mse=1e-4, output_mse_measured=True,
                             currency=campaign.CURRENCY)} for name in parameters}
    menus = {name: [FMT] for name in declared}
    payload = dict(costs=costs, formats=[FMT],
                   provenance=dict(cost_mode='production-render-score', wire_dir=str(tmp_path/'wire'),
                                   tessera_expert_projection=carried),
                   tessera_expert_wires={name: {FMT: _receipt(name, record)}
                       for name, record in carried['stacks'][STACK].items()
                       if not sampled or record['expert'] < 2})
    kwargs = dict(dense_targets=[], expert_targets=list(declared), dense_all=[], pinned=[],
                  population=population, layer_stride=1, costs=costs, menus=menus,
                  stack_samples=samples, profile=profile)
    return payload, kwargs, stats


def _with_population(tmp_path, **kwargs):
    payload, inputs, stats = _fixture(tmp_path, **kwargs)
    payload['provenance'][tep.POPULATION_KEY] = campaign.campaign_population_block(**inputs)
    return payload, stats


def test_population_prices_packed_decisions_and_records_full_sample_frame(tmp_path):
    payload, inputs, _ = _fixture(tmp_path, sampled=True)
    block = campaign.campaign_population_block(**inputs)
    assert set(block['priced']['routed_experts']) == set(payload['costs'])
    assert block['unpriced']['routed_experts'] == {}
    assert block['priced']['stacks'] == [STACK]
    decisions = block['stack_decisions']
    assert sum(len(d['members']) for d in decisions.values()) == 12
    assert sum(len(d['sampled_members']) for d in decisions.values()) == 6
    assert set(block['enumerated']['routed_experts']) == set(decisions)


def test_census_packed_assignment_carries_all_selected_source_receipts(tmp_path):
    payload, stats = _with_population(tmp_path)
    assignment = {q: FMT for q in stats}
    original = copy.deepcopy(assignment)
    block = tep.allocation_expert_projection_block(payload, assignment)
    assert assignment == original
    assert block[tep.STACK_FORMATS_KEY] == {STACK: FMT}
    assert block[tep.EXPERT_WIRES_KEY] == {n: formats[FMT] for n, formats in payload[tep.EXPERT_WIRES_KEY].items()}


def test_sampled_tessera_selection_refuses_missing_unsampled_wire(tmp_path):
    payload, stats = _with_population(tmp_path, sampled=True)
    with pytest.raises(tep.ExpertProjectionError, match=r'2\.w1: selected .*no priced wire'):
        tep.allocation_expert_projection_block(payload, {q: FMT for q in stats})


def test_sampled_stack_retained_at_bf16_needs_no_missing_wire(tmp_path):
    payload, stats = _with_population(tmp_path, sampled=True)
    block = tep.allocation_expert_projection_block(payload, {q: 'BF16' for q in stats})
    assert block[tep.EXPERT_WIRES_KEY] == {}
    assert block[tep.STACK_FORMATS_KEY] == {STACK: 'BF16'}


def test_population_refuses_double_pricing_a_member_and_its_packed_decision(tmp_path):
    payload, inputs, _ = _fixture(tmp_path)
    inputs['costs'][f'{STACK}.0.w1'] = {FMT: {'output_mse': .1}}
    with pytest.raises(campaign.StackSampleError, match='both|double'):
        campaign.campaign_population_block(**inputs)


def test_bridge_refuses_mapping_a_member_to_two_packed_decisions(tmp_path):
    payload, stats = _with_population(tmp_path)
    mapping = payload['provenance'][tep.POPULATION_KEY]['stack_decisions']
    names = sorted(mapping)
    mapping[names[0]]['members'].append(mapping[names[1]]['members'][0])
    with pytest.raises(tep.ExpertProjectionError, match='more than one|multiple'):
        tep.allocation_expert_projection_block(payload, {q: FMT for q in stats})


def test_main_packed_census_maps_to_the_producer_projection(tmp_path, monkeypatch):
    from prismaquant import allocator
    _v5_contract(monkeypatch)
    payload, stats = _with_population(tmp_path)
    argv = _allocator_argv(tmp_path, payload)
    probe_path = tmp_path/'probe.pkl'
    probe = pickle.loads(probe_path.read_bytes())
    probe['stats'] = stats
    probe_path.write_bytes(pickle.dumps(probe))
    monkeypatch.setattr(sys, 'argv', argv)
    allocator.main()
    result = json.loads((tmp_path/'layer.json').read_text())
    meta = result['__prismaquant__']
    assert meta[tep.STACK_FORMATS_KEY] == {STACK: FMT}
    assert len(meta[tep.EXPERT_WIRES_KEY]) == 12
