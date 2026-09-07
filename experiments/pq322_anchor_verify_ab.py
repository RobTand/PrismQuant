"""PB-only equal-work A/B of original versus bound, namespaced anchor verification."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle
import socket
import sys
import statistics
import time
from types import SimpleNamespace


def main():
    import torch
    from safetensors import safe_open
    from prismaquant import tessera_calibration_cache as cc, tessera_campaign as tc, tessera_hessian as th
    from prismaquant.calibration_data import load_calibration_input
    from prismaquant.cost_stage_checkpoint import _load_unit, canonical_json_sha256, unit_path
    from prismaquant.joint_aura import prefetch_joint_cache
    from prismaquant.model_profiles import detect_profile
    from prismaquant.tessera_reader import load_declared_reader, _source_tree
    from prismaquant.production_weight_cache import ProductionWeightCache, _cache_weight_filename
    from prismaquant.tessera_joint_aura import calibrated_maxima, verify_anchor_render

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--pairs', type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 3:
        parser.error('at least three interleaved measured pairs are required')
    if cc.sha256(args.plan) != args.plan_sha256:
        raise ValueError('qualification plan changed')
    plan = json.loads(args.plan.read_text())
    root = Path(plan['out']); root.mkdir(parents=True, exist_ok=False)
    for key in ('cost', 'checkpoint', 'census', 'capture', 'tokens'):
        if cc.sha256(plan[key]['path']) != plan[key]['sha256']:
            raise ValueError(f'{key} changed')
    payload = pickle.loads(Path(plan['cost']['path']).read_bytes())
    checkpoint = Path(plan['checkpoint']['path'])
    manifest = json.loads(checkpoint.read_text())
    identity = manifest['identity']
    if canonical_json_sha256(identity, where='wire qualification') != manifest['identity_sha256']:
        raise ValueError('journal identity changed')
    names = sorted(identity['units'])
    if names != plan['units']:
        raise ValueError('exact bounded qualification roster changed')
    census = json.loads(Path(plan['census']['path']).read_text())
    profile = detect_profile(census['model'])
    maxima, scales = calibrated_maxima(SimpleNamespace(census=census, payload=payload), profile)
    capture = cc.require_capture_contract(plan['capture']['path'], expected_sha256=plan['capture']['sha256'])
    expected = cc.capture_identity(plan['census']['path'],
        calibration=payload['provenance']['hessian']['calibration_identity'],
        max_act_rows=capture['identity']['max_act_rows'],
        model_load_contract=census['model_load_contract'], attention_implementation=census['attention_implementation'])
    if expected != capture['identity']:
        raise ValueError('actual source/capture identity changed')
    (acts, hessians, _counts, _maxima), _ = cc.prefetch_capture(plan['capture']['path'],
        expected_sha256=plan['capture']['sha256'], expected_identity=expected, census=census, names=names, device='cuda')
    calibration_source = th.activation_source(hessians, expected['calibration'])
    _ids, calibration = load_calibration_input(plan['tokens']['path'], expected_sha256=plan['tokens']['sha256'],
                                               n_samples=512, seqlen=512)
    weights = {}
    with safe_open(str(Path(census['model']) / 'model.safetensors'), framework='pt', device='cpu') as stream:
        for name in names:
            weights[name] = stream.get_tensor(name + '.weight').cuda()
    cells = {}
    parts = checkpoint.with_name(checkpoint.name + '.parts')
    for name in names:
        state = _load_unit(unit_path(parts, name), stage='Tessera campaign', qname=name,
                           identity_sha256=manifest['identity_sha256'])
        for anchor in state['anchors']:
            fmt = anchor['format_name']; record = state['wire_records'][fmt]
            render = Path(payload['provenance']['cache_dir']) / _cache_weight_filename(name, fmt)
            cells[name, fmt] = dict(anchor=anchor, record=record,
                wire=str(Path(payload['provenance']['wire_dir']) / record['file']),
                render=str(render), render_file_sha256=cc.sha256(render))
    cache = ProductionWeightCache(weights={pair: cell['render'] for pair, cell in cells.items()},
                                   levers={'tessera_campaign': True}, activation_max_abs=maxima)
    cache.enable_lru(plan['max_render_bytes'])
    formats = {name: sorted(fmt for qname, fmt in cells if qname == name) for name in names}
    prefetched = prefetch_joint_cache(cache, names, formats, max_resident_bytes=plan['max_render_bytes'])
    if len(names) != 2 or len(cells) != 14:
        raise ValueError('qualification requires original two-unit, fourteen-cell roster')
    # Load the producer API before the reader so module preservation is observable.
    api = tc._checkpoint_identity_api()
    import tessera
    from tessera.unit_artifact import read_unit_artifact
    primary_root = Path(tessera.__file__).resolve().parent
    producer_digest, producer_files = _source_tree(primary_root)
    primary_modules = {name: module for name, module in sys.modules.items()
                       if name == 'tessera' or name.startswith('tessera.')}
    reader = load_declared_reader(plan['reader'])
    if reader is None:
        raise ValueError('optimized arm requires a separately bound reader')
    result = {'schema': 'prismaquant.anchor_verify_ab.v1', 'passed': False,
        'plan_sha256': args.plan_sha256, 'pairs': args.pairs,
        'env': {'started_epoch': time.time(), 'host': socket.gethostname(),
            'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(), 'cpu_affinity': sorted(__import__('os').sched_getaffinity(0))},
        'producer': {'path': str(primary_root), 'source_sha256': producer_digest},
        'reader': reader.identity, 'phases': [], 'comparisons': [],
        'negative_cases': [], 'prefetch': prefetched,
        'workload': 'two original dense units, fourteen original wires/renders; same verify_anchor_render and PWC transfer in each arm',
        'scope': 'resident-prefetched verification only; excludes whole-model streaming, capture intake, and file-hash intake',
        'first_pair_policy': 'retained separately; first parse, not claimed disk-cold'}
    anchors = {name: [tc.CampaignAnchor(**cells[name, fmt]['anchor']) for fmt in formats[name]]
               for name in names}
    kwargs = dict(calibration_source=calibration_source, projected_unit=None, static_scales=scales)

    def before_verify():
        records = {}
        for name in names:
            for fmt in formats[name]:
                cell = cells[name, fmt]
                if cc.sha256(cell['render']) != cell['render_file_sha256']:
                    raise ValueError('original render changed')
                records[name, fmt] = verify_anchor_render(cell, weights[name], cache.get(name, fmt).cuda(), **kwargs)
        return records

    def after_verify():
        records = {}
        for name in names:
            # Binding, source/H hashing and finite checks are inside the timed arm.
            with tc.bind_checkpoint_unit_identity(anchors[name], source_weight=weights[name], **kwargs) as bound:
                for fmt in formats[name]:
                    cell = cells[name, fmt]
                    if cc.sha256(cell['render']) != cell['render_file_sha256']:
                        raise ValueError('original render changed')
                    records[name, fmt] = verify_anchor_render(cell, weights[name], cache.get(name, fmt).cuda(),
                                                             bound_unit=bound, reader=reader, **kwargs)
        return records

    reference = None
    def phase(arm, kind, index, *, profile=False):
        nonlocal reference
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        fn = before_verify if arm == 'before' else after_verify
        name = f'{kind}-{index:02d}-{arm}'
        start_epoch = time.time(); start = time.perf_counter()
        if profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                 torch.profiler.ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as profiler:
                value = fn(); torch.cuda.synchronize()
            elapsed = time.perf_counter() - start; end_epoch = time.time()
            profiler.export_chrome_trace(str(root / (name + '.trace.json')))
            (root / (name + '.operators.txt')).write_text(profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=60))
        else:
            value = fn(); torch.cuda.synchronize()
            elapsed = time.perf_counter() - start; end_epoch = time.time()
        if reference is None:
            reference = value
        if value != reference:
            raise ValueError(f'{name}: original/optimized verification records differ')
        result['phases'].append({'phase': name, 'arm': arm, 'pair': index, 'kind': kind,
            'start_epoch': start_epoch, 'end_epoch': end_epoch, 'seconds': elapsed,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
            'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(), 'equal_records': len(value)})
        print(json.dumps(result['phases'][-1]), flush=True)

    try:
        for arm in ('before', 'after'):
            phase(arm, 'first_parse', 0)
        for index in range(args.pairs):
            for arm in (('before', 'after') if index % 2 == 0 else ('after', 'before')):
                phase(arm, 'measured', index)
        for arm in ('before', 'after'):
            phase(arm, 'profile', 0, profile=True)
        result['comparisons'] = [{'unit': name, 'format': fmt, 'equal': True, **record}
                                 for (name, fmt), record in sorted(reference.items())]
        pair = next(pair for pair in sorted(cells) if 'E4M3' in pair[1])
        name, fmt = pair; cell = cells[pair]
        for arm in ('before', 'after'):
            for kind in ('render', 'source', 'settings', 'wire'):
                changed = copy.deepcopy(cell)
                source, render = weights[name], cache.get(name, fmt).cuda()
                if kind == 'render':
                    render = render.clone(); render[0, 0] += 1
                elif kind == 'source':
                    source = source.clone(); source[0, 0] += 1
                elif kind == 'settings':
                    changed['record']['identity']['recipe']['q256'] += 1
                else:
                    altered = bytearray(Path(cell['wire']).read_bytes()); altered[-1] ^= 1
                    corrupt = root / f'intentional-corrupt-wire-{arm}.bin'; corrupt.write_bytes(altered)
                    changed['wire'] = str(corrupt)
                try:
                    if arm == 'before':
                        verify_anchor_render(changed, source, render, **kwargs)
                    else:
                        with tc.bind_checkpoint_unit_identity(anchors[name], source_weight=source, **kwargs) as bound:
                            verify_anchor_render(changed, source, render, bound_unit=bound, reader=reader, **kwargs)
                except (ValueError, RuntimeError) as exc:
                    result['negative_cases'].append({'arm': arm, 'case': kind, 'refused': True, 'error': str(exc)})
                else:
                    raise AssertionError(f'{arm}: {kind} corruption was accepted')
        after_digest, after_files = _source_tree(primary_root)
        if (after_digest, after_files) != (producer_digest, producer_files):
            raise ValueError('primary producer source files changed')
        if any(sys.modules.get(name) is not module for name, module in primary_modules.items()):
            raise ValueError('primary producer module object changed')
        result['producer']['preserved_modules'] = {name: str(Path(module.__file__).resolve())
            for name, module in primary_modules.items() if getattr(module, '__file__', None)}
        result['producer']['files_unchanged'] = result['producer']['module_objects_unchanged'] = True
        medians = {arm: statistics.median(p['seconds'] for p in result['phases']
                    if p['kind'] == 'measured' and p['arm'] == arm) for arm in ('before', 'after')}
        result['measured'] = {'median_seconds': medians, 'ratio_before_over_after': medians['before'] / medians['after']}
        result['units'], result['cells'] = len(names), len(cells)
        result['passed'] = True
    except BaseException as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        result['env']['finished_epoch'] = time.time()
        (root / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'passed': result['passed'], 'result_sha256': cc.sha256(root / 'results.json')}), flush=True)


if __name__ == '__main__':
    main()
