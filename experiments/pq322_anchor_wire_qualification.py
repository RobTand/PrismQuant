"""PB-only original dense-anchor verification; never a full-model cost run."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle
import socket
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
    from prismaquant.native_operator_panel import prepare_native_inputs
    from prismaquant.production_weight_cache import ProductionWeightCache, _cache_weight_filename
    from prismaquant.tessera_joint_aura import calibrated_maxima, verify_anchor_render

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
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
    result = {'schema': 'prismaquant.anchor_wire_qualification.v1', 'passed': False,
        'plan_sha256': args.plan_sha256, 'env': {'started_epoch': time.time(), 'host': socket.gethostname(),
            'torch': str(torch.__version__), 'cuda': torch.version.cuda},
        'phases': [], 'comparisons': [], 'negative_cases': [], 'prefetch': prefetched}

    def phase(name, fn):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.time()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
             torch.profiler.ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as profiler:
            value = fn()
            torch.cuda.synchronize()
        end = time.time()
        profiler.export_chrome_trace(str(root / (name + '.trace.json')))
        (root / (name + '.operators.txt')).write_text(profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=60))
        result['phases'].append({'phase': name, 'kind': 'profile', 'start_epoch': start, 'end_epoch': end,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved()})
        return value

    def baseline():
        refs = {}
        for (name, fmt), cell in sorted(cells.items()):
            anchor = tc.CampaignAnchor(**cell['anchor'])
            encoding = tc._checkpoint_anchor_identity(anchor, weights=weights,
                menus={name: [SimpleNamespace(format_name=fmt)]}, calibration_source=calibration_source,
                static_scales=scales)
            record, tensors = prepare_native_inputs(cache, weights[name], acts[name].to(torch.bfloat16),
                unit=name, format_name=fmt, calibration_receipt=calibration,
                wire_blob=Path(cell['wire']).read_bytes(), wire_record=cell['record'],
                encoding_identity=encoding, numerics={'atol': 0.0, 'rtol': 0.0},
                prefill_rows=4, decode_rows=1, max_resident_bytes=plan['max_render_bytes'])
            refs[name, fmt] = {key: record[key] for key in ('source_weight', 'rendered_weight', 'activation')}
            del tensors, record
        return refs

    refs = phase('existing_native_input_bridge', baseline)
    def after():
        for (name, fmt), cell in sorted(cells.items()):
            record = verify_anchor_render(cell, weights[name], cache.get(name, fmt).cuda(),
                calibration_source=calibration_source, projected_unit=None, static_scales=scales)
            for key in ('source_weight', 'rendered_weight'):
                if record[key] != refs[name, fmt][key]:
                    raise ValueError(f'{name}@{fmt}: original/native source or render differs')
            result['comparisons'].append({'unit': name, 'format': fmt, 'equal': True, **record})
    phase('checked_anchor_bridge', after)
    pair = next(pair for pair in sorted(cells) if 'E4M3' in pair[1])
    name, fmt = pair; cell = cells[pair]
    kwargs = dict(calibration_source=calibration_source, projected_unit=None, static_scales=scales)
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
            original = Path(cell['wire']).read_bytes(); altered = bytearray(original); altered[-1] ^= 1
            corrupt = root / 'intentional-corrupt-wire.bin'; corrupt.write_bytes(altered)
            changed['wire'] = str(corrupt)
        try:
            verify_anchor_render(changed, source, render, **kwargs)
        except (ValueError, RuntimeError) as exc:
            result['negative_cases'].append({'case': kind, 'refused': True, 'error': str(exc)})
        else:
            raise AssertionError(f'{kind} corruption was accepted')
    result['env']['finished_epoch'] = time.time(); result['passed'] = True
    result['units'], result['cells'] = len(names), len(cells)
    (root / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'units': len(names), 'cells': len(cells),
                      'result_sha256': cc.sha256(root / 'results.json')}))


if __name__ == '__main__':
    main()
