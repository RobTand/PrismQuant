"""Packed decisions use the same source receipts and scope gates as a census."""

import pytest

from prismaquant import tessera_campaign as campaign
from prismaquant import tessera_export_lane as export
from prismaquant import tessera_expert_projection as tep
from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile
from test_tessera_export_projection import (
    case, _scope, _save, _meta, _units, _context, DENSE, FMT, N, STACK,
)


def _pack(case):
    profile = Lfm2MoeProfile()
    parameters = {f'{STACK}.gate_up_proj': (2, 2*N, N),
                  f'{STACK}.down_proj': (2, N, N)}
    samples = {}
    for name in parameters:
        stats = dict(h_trace=2.0, h_trace_per_expert=[1.0, 1.0], num_experts=2,
                     _packed_param=name.rsplit('.', 1)[1], _packed_experts_module=STACK)
        samples[name] = campaign.stack_sample_from_probe(name, stats, profile,
            sampled_experts=[0, 1], inclusion_prob={0: 1.0, 1: 1.0}, seed=0)
    costs = {name: {FMT: {'output_mse': 1e-4}} for name in [DENSE, *parameters]}
    population = campaign.campaign_population_block(
        dense_targets=[DENSE], expert_targets=_units(), dense_all=[DENSE], pinned=[],
        population=campaign.ExpertPopulation(members=(),
            declared={STACK: {name: (N, N) for name in _units()}},
            packed_in_scope=parameters, omitted_outside_layer_stride={}),
        layer_stride=1, costs=costs, menus={n: [FMT] for n in costs},
        stack_samples=samples, profile=profile)
    source = dict(costs=costs,
        provenance={tep.POPULATION_KEY: population, tep.PROJECTION_KEY: _meta(case)[tep.PROJECTION_KEY],
                    'wire_dir': str(case.wire_dir)},
        tessera_expert_wires={n: {FMT: receipt} for n, receipt in case.receipts.items()})
    assignment = {n: FMT for n in costs}
    _meta(case).update(tep.allocation_expert_projection_block(source, assignment))
    for name in _units():
        del case.payload[name]
        _meta(case)['tessera_serving_scope']['by_unit'].pop(name)
    for name in parameters:
        case.payload[name] = {'data_type': 'tessera', 'bits': 4, 'tessera_format': FMT}
        _meta(case)['tessera_serving_scope']['by_unit'][name] = _context('routed_moe')
    _save(case)
    return parameters


def test_packed_census_has_the_same_scope_as_expanded_receipt_backed_assignment(case):
    expanded = _scope(case)
    _pack(case)
    original = case.assignment.read_bytes()
    assert _scope(case) == expanded
    assert case.assignment.read_bytes() == original


def test_packed_scope_refuses_a_missing_unsampled_wire(case):
    _pack(case)
    name = f'{STACK}.1.w2'
    for decision in _meta(case)[tep.POPULATION_KEY]['stack_decisions'].values():
        decision['sampled_members'] = [n for n in decision['members'] if '.0.' in n]
    del _meta(case)[tep.EXPERT_WIRES_KEY][name]
    _save(case)
    with pytest.raises(export.TesseraExportLaneError, match=r'1\.w2: selected .*no priced wire'):
        _scope(case)


def test_packed_scope_refuses_wire_bytes_that_do_not_match_the_receipt(case):
    _pack(case)
    record = case.receipts[f'{STACK}.1.w2']
    (case.wire_dir/record['file']).write_bytes(b'wrong bytes')
    with pytest.raises(export.TesseraExportLaneError, match='does not match its receipt'):
        _scope(case)


def test_packed_scope_refuses_missing_source_ownership(case):
    _pack(case)
    decision = _meta(case)[tep.POPULATION_KEY]['stack_decisions'][f'{STACK}.down_proj']
    decision['members'].remove(f'{STACK}.1.w2')
    decision['sampled_members'].remove(f'{STACK}.1.w2')
    _save(case)
    with pytest.raises(export.TesseraExportLaneError, match='not in the assignment|partial stack'):
        _scope(case)


def test_packed_scope_refuses_conflicting_explicit_source_assignment(case):
    _pack(case)
    case.payload[f'{STACK}.1.w2'] = 'BF16'
    _save(case)
    with pytest.raises(export.TesseraExportLaneError, match='source assignment disagrees'):
        _scope(case)


def test_packed_scope_refuses_conflicting_source_context(case):
    _pack(case)
    _meta(case)['tessera_serving_scope']['by_unit'][f'{STACK}.1.w2'] = _context('dense')
    _save(case)
    with pytest.raises(export.TesseraExportLaneError, match='context disagrees'):
        _scope(case)


def test_packed_static_route_still_requires_its_priced_scales(case):
    parameters = _pack(case)
    for name in parameters:
        case.payload[name]['tessera_format'] = 'TESSERA_E2M1_K2_R896'
    _save(case)
    with pytest.raises(export.TesseraExportLaneError, match='static activation contract.*no --input-scales'):
        export.require_priced_export_inputs(case.assignment)
