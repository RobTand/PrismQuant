"""Materialize selected expert wires between sampled allocation and export.

A selection request is deliberately not a layer config. Each census anchor
stack is one independent PB action, using campaign calibration, render cache,
producer identities and the shared journal. Only finalization can publish the
packed allocation, after the ordinary receipt and priced-input gates pass.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

from . import tessera_expert_projection as tep
from .cost_stage_checkpoint import atomic_write_bytes

REQUEST_SCHEMA = "prismaquant.tessera_selected_wire_request.v1"
PLAN_SCHEMA = "prismaquant.tessera_selected_wire_plan.v1"
STAGE = "tessera_selected_wire"


def _sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _json(path, value):
    atomic_write_bytes(Path(path), (json.dumps(value, indent=2, sort_keys=True,
                                             allow_nan=False) + '\n').encode())


def write_selection_request(path, *, layer_config, assignment, cost_path,
                            cost_payload, output_path):
    """Persist decisions and existing evidence without claiming export readiness."""
    from .layer_config import canonicalize_assignment
    if canonicalize_assignment(layer_config) != dict(assignment):
        raise RuntimeError('selection layer config disagrees with resolved assignment')
    block = tep.selection_expert_projection_block(cost_payload, assignment)
    if not block.get(tep.PROJECTION_KEY):
        raise RuntimeError('selected-wire materialization requires a producer projection')
    config = copy.deepcopy(layer_config)
    config['__prismaquant__'].update(block)
    request = dict(schema=REQUEST_SCHEMA, status='selection_only_nonexportable',
                   layer_config=config, assignment=dict(assignment),
                   cost_path=str(Path(cost_path).resolve()), cost_sha256=_sha(cost_path),
                   output_path=str(Path(output_path).resolve()))
    _json(path, request)
    return request


def _request(path):
    from .layer_config import canonicalize_assignment
    request = json.loads(Path(path).read_text())
    if (request.get('schema') != REQUEST_SCHEMA or
            request.get('status') != 'selection_only_nonexportable'):
        raise RuntimeError('not a non-exportable selected-wire request')
    if _sha(request['cost_path']) != request['cost_sha256']:
        raise RuntimeError('selected-wire cost input changed')
    with Path(request['cost_path']).open('rb') as handle:
        cost = pickle.load(handle)
    assignment = canonicalize_assignment(request['layer_config'])
    if assignment != request['assignment']:
        raise RuntimeError('selected-wire assignment changed')
    block = tep.selection_expert_projection_block(cost, assignment)
    for key, value in block.items():
        if request['layer_config']['__prismaquant__'].get(key) != value:
            raise RuntimeError(f'selected-wire request changed carried {key}')
    source, units, stacks = tep.carried_units(block[tep.PROJECTION_KEY])
    expanded, _owners = tep.expand_stack_decision_assignment(
        assignment, block.get(tep.POPULATION_KEY), units=units, stack_of=stacks,
        costs=cost.get('costs', {}))
    return request, cost, source, units, expanded


def _check_census(path, cost, source, units):
    from . import tessera_campaign as tc
    provenance = {**cost['provenance']['hessian']['calibration_identity'], **cost['provenance']}
    args = SimpleNamespace(**{field:provenance[field]
        for field in ('model', 'nsamples', 'seqlen', 'seed', 'layer_stride')})
    census = tc.load_calibration_census(path, args=args)
    tc.require_census_draw(census, provenance['hessian']['calibration_identity'],
                           where='selected-wire planning')
    census_source, census_units, _stacks = tep.carried_units(census.get('expert_projection'))
    if census_source != source or census_units != units:
        raise RuntimeError('materialization census producer projection differs from priced projection')
    for name, unit in units.items():
        if census.get('unit_shapes', {}).get(name) != [unit['rows'], unit['cols']]:
            raise RuntimeError(f'{name}: census shape does not match projected geometry')
    return census


def _planned_groups(census, selected, cost):
    groups, covered = [], set()
    for key, members in sorted(census['anchor_groups'].items()):
        names = sorted(set(members) & set(selected))
        if covered.intersection(names):
            raise RuntimeError('census groups overlap selected units')
        covered.update(names)
        if names and any(cost.get(tep.EXPERT_WIRES_KEY, {}).get(n, {}).get(selected[n]) is None
                         for n in names):
            groups.append(dict(key=key, assignment={n:selected[n] for n in names}))
    if covered != set(selected):
        raise RuntimeError('census groups do not cover selected source units')
    return groups


def plan(request_path, census_path, workspace, spec_path, *, anchor_batch_size=1):
    from tools import dispatch_tessera_campaign as dispatcher
    if anchor_batch_size < 1:
        raise ValueError('anchor batch size must be positive')
    request, cost, source, units, expanded = _request(request_path)
    provenance = cost['provenance']
    census = _check_census(census_path, cost, source, units)
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    spec = dispatcher.load_spec(Path(spec_path))
    if spec['model'] != provenance['model']:
        raise RuntimeError('materialization spec model differs from priced model')
    selected = {n: expanded[n] for n in units if expanded[n].startswith('TESSERA_')}
    groups = _planned_groups(census, selected, cost)
    result = dict(schema=PLAN_SCHEMA, request_path=str(Path(request_path).resolve()),
                  request_sha256=_sha(request_path), census_path=str(Path(census_path).resolve()),
                  census_sha256=_sha(census_path), workspace=str(workspace), groups=groups)
    plan_path = workspace / 'plan.json'
    _json(plan_path, result)
    rows = []
    for index, group in enumerate(groups):
        row = dispatcher._row(spec, ['run', '--plan', str(plan_path), '--group', str(index),
                                     '--anchor-batch-size', str(anchor_batch_size)],
            mem_gb=dispatcher._row_memory_gb({**spec, 'max_act_rows':provenance['max_act_rows']},
                                            list(group['assignment']), census),
            timeout_s=int(spec.get('timeout_s', 86400)), module='prismaquant.tessera_materialization')
        rows.append(row)
    _json(workspace / 'manifest.json', rows)
    return result


def _inputs(plan_path, group_index=None):
    from . import tessera_campaign as tc
    plan_data = json.loads(Path(plan_path).read_text())
    if plan_data.get('schema') != PLAN_SCHEMA:
        raise RuntimeError('not a selected-wire plan')
    for field in ('request', 'census'):
        if _sha(plan_data[field + '_path']) != plan_data[field + '_sha256']:
            raise RuntimeError(f'selected-wire {field} changed after planning')
    request, cost, source, units, expanded = _request(plan_data['request_path'])
    expected = {n: expanded[n] for n in units if expanded[n].startswith('TESSERA_')}
    census = _check_census(plan_data['census_path'], cost, source, units)
    if plan_data['groups'] != _planned_groups(census, expected, cost):
        raise RuntimeError('materialization plan differs from selected census groups')
    api = tc._checkpoint_identity_api()
    source_sha = api.encoder_source_sha256()
    # Every measured receipt must describe the same producer source.
    priced_receipts = 0
    for name, rows in cost.get(tep.EXPERT_WIRES_KEY, {}).items():
        for record in rows.values():
            priced_receipts += 1
            if record['identity'].get('encoder_source_sha256') != source_sha:
                raise RuntimeError(f'{name}: priced wire producer source changed')
    if expected and not priced_receipts:
        raise RuntimeError('selected-wire request has no priced producer receipt to bind encoder source')
    if tc.th.encoder_recipe() != cost['provenance']['hessian']['recipe']:
        raise RuntimeError('selected-wire encoder recipe differs from priced recipe')
    from .production_weight_cache import _production_cache_source_sha256
    identity = dict(plan_sha256=_sha(plan_path), encoder_source_sha256=source_sha,
                    prismaquant_source_sha256=_production_cache_source_sha256(),
                    recipe=tc.th.encoder_recipe())
    if group_index is not None and not 0 <= group_index < len(plan_data['groups']):
        raise ValueError('materialization group index is outside the plan')
    group = None if group_index is None else plan_data['groups'][group_index]
    return plan_data, request, cost, source, units, expanded, census, identity, group


def _verify_source(model, source):
    for filename, expected in source['files'].items():
        if _sha(Path(model) / filename) != expected:
            raise RuntimeError(f'selected-wire source checkpoint changed: {filename}')
    if _sha(Path(model) / 'config.json') != source['config_sha256']:
        raise RuntimeError('selected-wire source model config changed')
    for filename, expected in source.get('auxiliary_sha256', {}).items():
        if _sha(Path(model) / filename) != expected:
            raise RuntimeError(f'selected-wire auxiliary source changed: {filename}')


def _journal(plan_data, identity, index, names):
    from .cost_stage_checkpoint import prepare_journal
    root = Path(plan_data['workspace']) / 'groups' / str(index)
    root.mkdir(parents=True, exist_ok=True)
    return root, prepare_journal(root / 'journal', stage=STAGE, resume=True,
        identity={**identity, 'group': index}, qnames=names)


def run(plan_path, group_index, *, anchor_batch_size=1):
    """Execute inside one admitted action; never submit recursively."""
    import functools
    from dataclasses import asdict
    import torch
    from transformers import AutoModelForCausalLM
    from . import tessera_campaign as tc
    from . import tessera_hessian as th
    from .cost_stage_checkpoint import write_unit
    from .model_profiles import detect_profile
    from .production_weight_cache import ProductionWeightCache
    from .tessera_formats import parse_tessera_format_name, tessera_wire_recipe
    from .tessera_render import rung_accepts_hessian

    if anchor_batch_size < 1:
        raise ValueError('anchor batch size must be positive')
    if anchor_batch_size > 1:
        from .tessera_render import require_tessera_batch_encoder
        require_tessera_batch_encoder()
    data, request, cost, source, units, _expanded, census, identity, group = _inputs(plan_path, group_index)
    names = sorted(group['assignment'])
    root, (journal, journal_sha, completed) = _journal(data, identity, group_index, names)
    provenance = {**cost['provenance']['hessian']['calibration_identity'], **cost['provenance']}
    model_path = provenance['model']
    _verify_source(model_path, source)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    profile = detect_profile(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device)
    model.eval()
    population = tc._require_campaign_population(model, profile, provenance['layer_stride'])
    members = {member.qname: member for member in population.members}
    if set(names) - set(members):
        raise RuntimeError('selected source units absent from profile population')
    tokens, corpus = tc._calibration_tokens(model_path, provenance['nsamples'],
                                           provenance['seqlen'], provenance['seed'])
    want_h = provenance['hessian']['supplied']
    acts, hessians, counts, maxima = tc._collect_activations(
        model, names, tokens, provenance['max_act_rows'], device,
        want_hessian=want_h, profile=profile)
    hi, lo = tc.census_token_counts(census, counts)
    calibration = th.calibration_identity(corpus, tokens, fit_tokens=hi,
        source='wikitext-2-raw-v1/train', split_role='calibration', model=str(model_path),
        seed=provenance['seed'], nsamples=provenance['nsamples'],
        seqlen=provenance['seqlen'], fit_tokens_min=lo)
    tc.require_census_draw(census, calibration, where='selected-wire capture')
    if calibration != provenance['hessian']['calibration_identity']:
        raise RuntimeError('selected-wire capture calibration differs from priced capture')
    scales, scale_policy = tc._static_input_scales(tc.census_max_abs(census, maxima), profile=profile)
    stamped_scales = provenance['activation_static_scales']
    if scale_policy != stamped_scales['policy'] or scales != stamped_scales['units']:
        raise RuntimeError('selected-wire static scales differ from priced census scales')
    weights = {name: members[name].weight.detach() for name in names}
    for name, weight in weights.items():
        original = tep.source_unit_weight(model_path, source, units[name])
        if not torch.equal(original, weight.cpu()):
            raise RuntimeError(f'{name}: live projection differs from source checkpoint')
    del model, members, population
    if device == 'cuda':
        torch.cuda.empty_cache()
    activation = th.activation_source(hessians, calibration) if want_h else None
    @functools.lru_cache(maxsize=None)
    def kwargs(name, plane):
        return th.encoder_kwargs(activation, name, weights[name].shape[1], device, scale_plane=plane)
    wire_dir = root / 'wire'
    wire_dir.mkdir(exist_ok=True)
    cache = ProductionWeightCache(weights={}, levers={'tessera_campaign': True},
        cache_dir=str(root), metadata={'schema': REQUEST_SCHEMA})
    api = tc._checkpoint_identity_api()
    missing, expected_by_name = [], {}
    for name in names:
        fmt = group['assignment'][name]
        family, rung = parse_tessera_format_name(fmt)
        active = (activation if activation is not None and
                  rung_accepts_hessian(fmt, tessera_wire_recipe(family, rung)) else None)
        expected = api.unit_input_identity(weights[name], units[name], family.payload_grid(), rung,
                                           activation=active)
        wire_path = tc._wire_path(wire_dir, name, fmt)
        state = completed.get(name)
        record = None if state is None else state['record']
        if record is None:
            record = cost.get(tep.EXPERT_WIRES_KEY, {}).get(name, {}).get(fmt)
            if record is not None:
                tc._link_seed_wire(Path(provenance['wire_dir']), wire_dir, record['file'])
        if record is not None:
            tep.verify_expert_wire_record(record, name=name, unit=units[name], q256=rung,
                grid=family.payload_grid().name, wire_dir=wire_dir)
            api.verify_cached_unit(wire_path.read_bytes(), record, expected)
            if record['file'] != wire_path.name:
                raise RuntimeError(f'{name}: wire filename differs from selected rung')
            anchor = None if state is None else state.get('anchor')
        else:
            if wire_path.exists():
                raise RuntimeError(f'{name}: unjournaled wire exists; refusing overwrite')
            missing.append((name, family.name, rung))
            expected_by_name[name] = expected
            continue
        write_unit(journal, stage=STAGE, qname=name, identity_sha256=journal_sha,
            state=dict(format=fmt, record=record, anchor=anchor))
    for batch in tc._anchor_batches(missing, weights=weights, expert_members=units,
                                    batch_size=anchor_batch_size):
        batch_names = [item[0] for item in batch]
        fmt = group['assignment'][batch_names[0]]
        common = dict(format_name=fmt, cache=cache, wire_dir=wire_dir,
                      activation_kwargs_for=kwargs, hessian_required=want_h)
        if len(batch) == 1:
            name = batch_names[0]
            anchors = [tc._measure_anchor(qname=name, weight=weights[name].to(device),
                activations=acts[name].to(device), static_input_scale=scales.get(name), **common)]
        else:
            anchors = tc._measure_anchor_batch(qnames=batch_names,
                weights=[weights[name].to(device) for name in batch_names],
                activations=[acts[name].to(device) for name in batch_names],
                static_input_scales=scales, **common)
        for anchor in anchors:
            name = anchor.qname
            record = tc._checkpoint_wire_record(anchor, wire_dir, expected_by_name[name])
            write_unit(journal, stage=STAGE, qname=name, identity_sha256=journal_sha,
                state=dict(format=anchor.format_name, record=record, anchor=asdict(anchor)))
    capture, scale_file, digest = tc.write_export_inputs(root, hessians=hessians if want_h else None,
        hessian_rows=counts, hessian_identity=calibration, static_scales=scales,
        static_scale_policy=scale_policy)
    _json(root / 'complete.json', dict(identity=identity, group=group_index,
        capture_path=None if capture is None else str(capture), capture_sha256=digest,
        input_scales_path=None if scale_file is None else str(scale_file),
        units=names, calibration_identity=calibration))


def finalize(plan_path):
    """Publish only a fully verified packed allocation and its exact export inputs."""
    import torch
    from . import tessera_campaign as tc
    from . import tessera_hessian as th
    from . import tessera_export_lane as export
    from .tessera_formats import parse_tessera_format_name, tessera_wire_recipe
    from .tessera_render import rung_accepts_hessian

    data, request, cost, source, units, expanded, census, identity, _group = _inputs(plan_path)
    provenance = {**cost['provenance']['hessian']['calibration_identity'], **cost['provenance']}
    _verify_source(provenance['model'], source)
    root = Path(data['workspace']) / 'final'
    root.mkdir(parents=True, exist_ok=True)
    wire_dir = root / 'wire'
    wire_dir.mkdir(exist_ok=True)
    want_h = provenance['hessian']['supplied']
    calibration = provenance['hessian']['calibration_identity']
    hessians, counts = {}, {}
    if want_h:
        original_path = Path(provenance['hessian']['capture_path'])
        hessians, original_identity, digest = export._bound_hessian_capture(original_path)
        if (digest != provenance['hessian']['capture_sha256'] or
                original_identity != {**calibration, 'hessian_role': 'fit'}):
            raise RuntimeError('original priced capture changed before materialization')
        counts = dict(torch.load(original_path, map_location='cpu', weights_only=False)['counts'])
        tc.census_token_counts(census, counts)
        if set(counts) != set(hessians):
            raise RuntimeError('original priced capture counts do not cover its Hessians')
    receipts = copy.deepcopy(cost.get(tep.EXPERT_WIRES_KEY, {}))
    api = tc._checkpoint_identity_api()
    completed_names = set()
    for index, group in enumerate(data['groups']):
        names = sorted(group['assignment'])
        group_root, (_journal_root, _digest, completed) = _journal(data, identity, index, names)
        completion = json.loads((group_root / 'complete.json').read_text())
        if (completion['identity'] != identity or completion['group'] != index or
                completion['units'] != names or completion['calibration_identity'] != calibration or
                set(completed) != set(names)):
            raise RuntimeError(f'materialization group {index} is incomplete or changed')
        group_h = {}
        if want_h:
            capture = group_root / 'hessian_capture.pt'
            group_h, capture_identity, digest = export._bound_hessian_capture(capture)
            if (digest != completion['capture_sha256'] or
                    capture_identity != {**calibration, 'hessian_role': 'fit'} or
                    set(group_h) != set(names)):
                raise RuntimeError(f'materialization group {index} capture changed')
            group_counts = torch.load(capture, map_location='cpu', weights_only=False)['counts']
            tc.census_token_counts(census, group_counts)
            if set(group_counts) != set(names):
                raise RuntimeError('materialization capture counts do not cover exactly its selected units')
            for name, hessian in group_h.items():
                if name in hessians and (not torch.equal(hessians[name], hessian) or
                                        counts[name] != group_counts[name]):
                    raise RuntimeError(f'{name}: materialized Hessian disagrees with priced overlap')
                hessians[name], counts[name] = hessian, group_counts[name]
        for name in names:
            fmt = group['assignment'][name]
            state = completed[name]
            if state['format'] != fmt:
                raise RuntimeError(f'{name}: materialization changed selected format')
            record = state['record']
            previous = receipts.setdefault(name, {}).get(fmt)
            if previous is not None and previous != record:
                raise RuntimeError(f'{name}: materialization replaced an existing selected receipt')
            receipts[name][fmt] = record
            completed_names.add(name)
    # Existing complete stacks need no new calibration quantum. Verify their
    # receipts under the original capture, alongside every completed group.
    activation = th.activation_source(hessians, calibration) if want_h else None
    for name in sorted(units):
        fmt = expanded[name]
        if not fmt.startswith('TESSERA_'):
            continue
        record = receipts.get(name, {}).get(fmt)
        if record is None:
            raise RuntimeError(f'{name}: selected wire is still missing')
        source_dir = Path(provenance['wire_dir'])
        for index, group in enumerate(data['groups']):
            if name in group['assignment']:
                source_dir = Path(data['workspace']) / 'groups' / str(index) / 'wire'
                break
        family, rung = parse_tessera_format_name(fmt)
        active = (activation if activation is not None and
                  rung_accepts_hessian(fmt, tessera_wire_recipe(family, rung)) else None)
        weight = tep.source_unit_weight(provenance['model'], source, units[name])
        expected = api.unit_input_identity(weight, units[name], family.payload_grid(), rung,
                                           activation=active)
        checked = tep.verify_expert_wire_record(record, name=name, unit=units[name], q256=rung,
            grid=family.payload_grid().name, wire_dir=source_dir)
        api.verify_cached_unit((source_dir / checked['file']).read_bytes(), checked, expected)
        tc._link_seed_wire(source_dir, wire_dir, checked['file'])
        api.verify_cached_unit((wire_dir / checked['file']).read_bytes(), checked, expected)
        completed_names.add(name)
    scales = provenance['activation_static_scales']
    capture, scale_path, digest = tc.write_export_inputs(root,
        hessians=hessians if want_h else None, hessian_rows=counts,
        hessian_identity=calibration, static_scales=scales['units'],
        static_scale_policy=scales['policy'])
    completed_cost = {**cost, tep.EXPERT_WIRES_KEY: receipts,
                      'provenance': {**provenance, 'wire_dir': str(wire_dir)}}
    config = copy.deepcopy(request['layer_config'])
    meta = config['__prismaquant__']
    meta.update(tep.allocation_expert_projection_block(completed_cost, request['assignment']))
    meta['tessera_activation_static_scales'] = dict(schema=export.PRICED_STATIC_SCALES_SCHEMA,
                                                   units=dict(scales['units']))
    if want_h:
        meta['tessera_hessian'] = {**meta['tessera_hessian'], 'capture_sha256': digest,
                                   'capture_path': str(capture)}
    meta['tessera_selected_wire_materialization'] = dict(
        schema='prismaquant.tessera_selected_wire_completion.v1',
        request_sha256=data['request_sha256'], plan_sha256=_sha(plan_path),
        original_cost_sha256=request['cost_sha256'],
        original_capture_sha256=provenance['hessian'].get('capture_sha256'),
        capture_sha256=digest, verified_source_units=sorted(completed_names),
        selection_prices='unchanged_sampled_stack_estimates')
    candidate = root / 'verified-layer-config.json'
    _json(candidate, config)
    export.require_priced_export_inputs(candidate, hessian_path=capture,
                                       input_scales_path=scale_path)
    atomic_write_bytes(Path(request['output_path']), candidate.read_bytes())
    report = dict(layer_config=request['output_path'], hessian=None if capture is None else str(capture),
                  input_scales=None if scale_path is None else str(scale_path),
                  wire_dir=str(wire_dir), verified_source_units=len(completed_names))
    _json(root / 'result.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    planning = commands.add_parser('plan')
    planning.add_argument('--request', required=True)
    planning.add_argument('--census', required=True)
    planning.add_argument('--workspace', required=True)
    planning.add_argument('--spec', required=True)
    planning.add_argument('--anchor-batch-size', type=int, default=1)
    running = commands.add_parser('run')
    running.add_argument('--plan', required=True)
    running.add_argument('--group', type=int, required=True)
    running.add_argument('--anchor-batch-size', type=int, default=1)
    finalizing = commands.add_parser('finalize')
    finalizing.add_argument('--plan', required=True)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        result = plan(args.request, args.census, args.workspace, args.spec,
                      anchor_batch_size=args.anchor_batch_size)
    elif args.command == 'run':
        result = run(args.plan, args.group, anchor_batch_size=args.anchor_batch_size)
    else:
        result = finalize(args.plan)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
