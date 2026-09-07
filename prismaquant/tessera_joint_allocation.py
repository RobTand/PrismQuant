"""Bind completed joint prices to their original Tessera export metadata.

This is an opt-in metadata handoff, not a probe, renderer or export gate. The
original joint table stays immutable. Current wire bytes are checked by the
existing exporter; this handoff authenticates the recorded preparation that
proved which original wire produced each tensor measured by joint AURA.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle

from .cluster_campaign import _atomic_write_new_bytes as atomic_write_bytes
from .cost_stage_checkpoint import canonical_json_sha256
from .tessera_joint_aura import PREPARED_SCHEMA, SCHEMA, _require, _same

HANDOFF_SCHEMA = 'prismaquant.tessera_joint_allocation.v1'
ROW_FIELDS = ('hessian_identity', 'tessera_family', 'tessera_body_rate_q256',
              'wire_bytes', 'input_global_scale', 'activation_contract', 'activation_quantized')
PROVENANCE_FIELDS = ('model', 'nsamples', 'seqlen', 'layer_stride', 'max_act_rows',
                     'hessian', 'activation_static_scales', 'wire_dir', 'calibration_cache',
                     'population', 'tessera_expert_projection')


def _add(target, key, value, where):
    if key in target:
        _same(target[key], value, f'{where}: existing {key}')
    else:
        target[key] = copy.deepcopy(value)


def bind_allocation_payload(joint, data, prepared, cache_metadata, *, plan_sha256, prepared_binding):
    """Join already-authenticated artifacts without replacing any joint field."""
    from .cost_currency import require_run_currency
    from .schemas import validate_cost_payload, validate_probe_payload
    from . import tessera_expert_projection as tep

    validate_cost_payload(joint)
    validate_probe_payload(joint)
    currency = require_run_currency(joint)
    _require(currency.get('joint_aura_rows', 0) > 0, 'a complete joint table is required')
    evidence = joint['provenance'].get('tessera_joint_anchors', {})
    _same(evidence.get('plan_sha256'), plan_sha256, 'joint plan')
    _same(evidence.get('prepared'), prepared_binding, 'joint prepared binding')
    _same(evidence.get('inputs'), data.inputs, 'joint original anchor inputs')
    _same(prepared.get('schema'), PREPARED_SCHEMA, 'prepared schema')
    _same(prepared.get('status'), 'complete', 'prepared completion')
    _same(prepared.get('plan_sha256'), plan_sha256, 'prepared plan')
    _same(prepared.get('calibration_input'), evidence.get('calibration_input'), 'prepared calibration')
    roster = {name: list(formats) for name, formats in data.formats_by_qname.items()}
    _same(prepared.get('formats_by_qname'), roster, 'prepared candidate roster')
    _same(set(joint['costs']), set(roster), 'joint unit roster')
    _same(set(joint['stats']), set(roster), 'joint statistic roster')
    for item in (prepared, evidence):
        _same(item.get('measured_cells'), len(data.cells), 'measured anchor count')
    _same(cache_metadata.get('schema'), PREPARED_SCHEMA, 'prepared cache schema')
    _same(cache_metadata.get('inputs'), data.inputs, 'prepared cache inputs')
    for key in ('reader_identity', 'projection_backend'):
        _same(cache_metadata.get(key), prepared.get(key), f'prepared cache {key}')
    verified = cache_metadata.get('verified_cells', {})
    _same(set(verified), set(data.cells), 'prepared tensor receipt roster')
    original = data.payload['provenance']
    calibration = prepared['calibration_input']
    original_draw = original['hessian']['calibration_identity']
    _same(calibration['provenance'], {key: original_draw.get(key) for key in calibration['provenance']},
          'anchor calibration draw')
    source_by_unit = {}
    result = copy.deepcopy(joint)
    for name, formats in roster.items():
        _same(set(joint['costs'][name]), set(formats), f'{name}: measured candidate roster')
        shape = data.census['unit_shapes'][name]
        stats = joint['stats'][name]
        _same([stats.get('out_features'), stats.get('in_features')], shape, f'{name}: statistics shape')
        _same(stats['n_params'], math.prod(shape), f'{name}: parameter count')
        for fmt in formats:
            row = joint['costs'][name][fmt]
            operator, probe = row['joint_operator_identity'], row['probe_identity']
            _same(probe['source_model'], prepared['source_model_identity'], f'{name}: source model')
            _same(probe['source_model']['source'], original['model'], f'{name}: source model path')
            _same(probe['calibration_sha256'], calibration['calibration_sha256'], f'{name}: probe calibration')
            _same(probe.get('calibration_shape'), calibration['shape'], f'{name}: calibration shape')
            _same(operator['arithmetic']['projection_backend'], prepared['projection_backend'], f'{name}: projection backend')
            _same(operator['source_weight']['shape'], shape, f'{name}: source shape')
            source = operator['source_weight']
            _same(source, source_by_unit.setdefault(name, source), f'{name}: source changed across formats')
            if fmt == 'BF16':
                _same(operator['rendered_weight'], source, f'{name}: BF16 source passthrough')
                _require(not operator['activation']['quantizes_input'] and
                         all(value == 0 for value in row['x2_per_probe']), f'{name}: BF16 is not measured zero')
                continue
            pair = (name, fmt)
            receipt = verified[pair]
            for key in ('source_weight', 'rendered_weight', 'activation'):
                _same(operator[key], receipt[key], f'{name}@{fmt}: prepared {key}')
            record = data.cells[pair]['record']
            _same(receipt['wire_sha256'], record['blob_sha256'], f'{name}@{fmt}: original wire')
            source_record = data.manifest['identity']['units'][name]['weight']
            # Tessera hashes dtype/shape plus values; PWC hashes raw values.
            # Preparation verified the original encoding identity against the
            # actual source tensor and retained BOTH owner-defined receipts.
            # Join through that authenticated encoding receipt, never compare
            # hashes from different grammars or manufacture a conversion.
            _same(receipt['encoding_identity_sha256'],
                  canonical_json_sha256(record['identity'], where='original encoding'),
                  f'{name}@{fmt}: prepared encoding identity')
            _same(record['identity']['source'], source_record, f'{name}: original source identity')
            _same(source['shape'], source_record['shape'], f'{name}: original source dimensions')
            _same(source['dtype'].removeprefix('torch.'), source_record['dtype'].removeprefix('torch.'), f'{name}: original source dtype')
            anchor_row = data.payload['costs'][name][fmt]
            _same(operator['activation']['input_global_scale'], anchor_row.get('input_global_scale'), f'{name}@{fmt}: original static scale')
            for key in ROW_FIELDS:
                if key in anchor_row:
                    _add(result['costs'][name][fmt], key, anchor_row[key], f'{name}@{fmt}')
    for key in PROVENANCE_FIELDS:
        if key in original:
            _add(result['provenance'], key, original[key], 'joint provenance')
    if tep.PROJECTION_KEY in original:
        _source, units, _stacks = tep.carried_units(original[tep.PROJECTION_KEY])
        _require(set(units) <= set(roster), 'original projection exceeds joint roster')
        original_wires = data.payload.get(tep.EXPERT_WIRES_KEY, {})
        wires = {}
        for name in units:
            wires[name] = {}
            for fmt in roster[name]:
                if fmt == 'BF16':
                    continue
                record = data.cells[name, fmt]['record']
                _same(original_wires.get(name, {}).get(fmt), record, f'{name}@{fmt}: projected wire receipt')
                wires[name][fmt] = copy.deepcopy(record)
        _add(result, tep.EXPERT_WIRES_KEY, wires, 'joint payload')
        # The existing population/projection validator checks coverage without
        # choosing a quantization: BF16 is only an inert receipt-validation arm.
        tep.allocation_expert_projection_block(result, {name: 'BF16' for name in roster})
    _add(result['provenance'], 'tessera_joint_allocation', {
        'schema': HANDOFF_SCHEMA, 'status': 'research_metadata_handoff',
        'plan_sha256': plan_sha256, 'prepared': prepared_binding,
        'units': len(roster), 'measured_cells': len(data.cells),
        'cost_fields': 'all_original_joint_fields_unchanged',
        'wire_validation': 'historical_prepared_identity; current_bytes_require_export_gate',
    }, 'joint provenance')
    _same(require_run_currency(result), currency, 'unchanged joint currency')
    return result


def _read_bound(record, label):
    _require(isinstance(record, dict) and set(record) == {'path', 'sha256'}, f'{label}: bound path and SHA256 required')
    raw = Path(record['path']).read_bytes()
    _same(hashlib.sha256(raw).hexdigest(), record['sha256'], f'{label}: owned bytes')
    return raw


def handoff(*, joint_binding, plan_binding, output_path):
    """Authenticate inputs and publish a new table, then its success receipt.

    Both publications refuse concurrent destinations atomically. If receipt
    publication fails, the incomplete own table remains for diagnosis; no
    success receipt is returned and no competing bytes are replaced.
    """
    from .production_weight_cache import ProductionWeightCache
    from .tessera_joint_aura import load_measured_anchor_input

    joint = pickle.loads(_read_bound(joint_binding, 'joint cost'))
    plan = json.loads(_read_bound(plan_binding, 'joint plan'))
    _same(plan.get('schema'), SCHEMA, 'joint plan schema')
    evidence = joint['provenance']['tessera_joint_anchors']
    _same(evidence['plan_sha256'], plan_binding['sha256'], 'joint plan binding')
    _same(evidence['inputs'], plan['inputs'], 'joint plan original inputs')
    prepared_binding = evidence['prepared']
    prepared = json.loads(_read_bound(prepared_binding, 'prepared completion'))
    _same(prepared['calibration_input']['artifact_sha256'], plan['calibration_input']['sha256'], 'planned calibration artifact')
    cache = pickle.loads(_read_bound(prepared['production_cache'], 'prepared cache'))
    _require(isinstance(cache, ProductionWeightCache), 'prepared cache owner is not ProductionWeightCache')
    # This joins historical identities; it does not re-read all decoded model
    # tensors or weaken the exporter's current-byte verification.
    data = load_measured_anchor_input(plan['inputs'], verify_payloads=False)
    _same(cache.weights, {pair: cell['render'] for pair, cell in data.cells.items()}, 'prepared render paths')
    result = bind_allocation_payload(joint, data, prepared, cache.metadata,
        plan_sha256=plan_binding['sha256'], prepared_binding=prepared_binding)
    result['provenance']['tessera_joint_allocation']['original_joint_cost'] = dict(joint_binding)
    output = Path(output_path)
    _require(not output.exists() and output.resolve() != Path(joint_binding['path']).resolve(), 'handoff output must be new')
    receipt_path = output.with_suffix(output.suffix + '.receipt.json')
    _require(not receipt_path.exists(), 'handoff receipt must be new')
    raw = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    receipt = {'schema': HANDOFF_SCHEMA, 'original_joint_cost': joint_binding, 'plan': plan_binding,
               'prepared': prepared_binding, 'prepared_cache': prepared['production_cache'],
               'output': {'path': str(output.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()},
               'units': len(result['costs']), 'joint_fields_unchanged': True, 'research_only': True}
    atomic_write_bytes(output, raw)
    atomic_write_bytes(receipt_path,
                       (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode())
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joint-cost', required=True)
    parser.add_argument('--joint-cost-sha256', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(handoff(joint_binding={'path': args.joint_cost, 'sha256': args.joint_cost_sha256},
        plan_binding={'path': args.plan, 'sha256': args.plan_sha256}, output_path=args.output)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
