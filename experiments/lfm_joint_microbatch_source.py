"""PB-only resident/streamed, same-size source and packed-gradient controls."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import socket
import time
from pathlib import Path

from experiments.lfm_packed_joint_screen import file_sha


def main():
    import torch
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    from prismaquant.aura_cost import compute_aura_cost_streamed
    from prismaquant.cost_streaming import build_streamed_causal_lm, build_streamed_model_identity
    from prismaquant.joint_aura import source_execution_identity
    from prismaquant.kl_fisher import fisher_probe_scalar
    from prismaquant.model_profiles import detect_profile
    from prismaquant.production_weight_cache import ProductionWeightCache

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/mnt/shared/models/LFM2.5-8B-A1B-BF16')
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--tokens-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if file_sha(args.tokens) != args.tokens_sha256:
        raise ValueError('canonical calibration artifact differs')
    ids = load_file(str(args.tokens))['calibration_ids'][:3].contiguous()
    if ids.shape != (3, 512) or ids.dtype != torch.int64:
        raise ValueError('expected three canonical int64 512-token sequences')
    result = {'schema': 'prismaquant.joint_aura.microbatch_source_control.v1',
        'env': {'host': socket.gethostname(), 'started_epoch': time.time(),
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'cuda': torch.version.cuda},
        'calibration': {'artifact': str(args.tokens), 'sha256': args.tokens_sha256,
                        'rows': [0, 1, 2], 'shape': list(ids.shape),
                        'tensor_sha256': hashlib.sha256(ids.numpy().tobytes()).hexdigest()},
        'cross_partition': {}, 'same_size': [], 'phases': []}
    def save():
        (args.out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    def digest(t):
        return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
    def compare(a, b):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('comparison geometry/dtype differs')
        record = {'shape': list(a.shape), 'dtype': str(a.dtype),
                  'equal': torch.equal(a, b), 'a_sha256': digest(a), 'b_sha256': digest(b)}
        if record['equal']:
            return dict(record, max_abs=0.0, relative_l2=0.0, different=0)
        af, bf = a.reshape(-1), b.reshape(-1)
        error2 = norm2 = 0.0
        maximum = 0.0
        different = 0
        for start in range(0, a.numel(), 1 << 20):
            x, y = af[start:start + (1 << 20)].float(), bf[start:start + (1 << 20)].float()
            d = y - x
            error2 += float(d.double().square().sum())
            norm2 += float(x.double().square().sum())
            maximum = max(maximum, float(d.abs().max()))
            different += int(torch.count_nonzero(d))
        return dict(record, max_abs=maximum, relative_l2=math.sqrt(error2 / max(norm2, 1e-100)), different=different)
    unit = 'model.layers.2.feed_forward.experts'
    leaf_names = ('gate_up_proj', 'down_proj')
    expected_grads = {}
    expected_logits = {}
    expected_boundaries = {}
    expected_routes = {}
    current = None
    reference = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=True, attn_implementation='eager').cuda().eval()
    result['reference_source_execution'] = source_execution_identity(reference)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    parameters = {name: reference.get_parameter(f'{unit}.{name}') for name in leaf_names}
    for parameter in parameters.values():
        parameter.requires_grad_(True)
    def boundary_hook(depth):
        def record(module, inputs, output):
            if current[2] == 0:
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                expected_boundaries[(*current[:2], depth)] = tensor.detach().cpu()
        return record
    handles = [layer.register_forward_hook(boundary_hook(depth))
               for depth, layer in enumerate(reference.model.layers)]
    def route_hook(module, inputs):
        if current[2] == 0:
            expected_routes[current[:2]] = tuple(x.detach().cpu() for x in inputs)
    handles.append(reference.get_submodule(unit).register_forward_pre_hook(route_hook))
    started = time.time()
    for batch_size in (3, 1):
        for probe_index in range(4):
            for offset in range(0, 3, batch_size):
                current = (batch_size, offset, probe_index)
                logits = reference(ids[offset:offset + batch_size].cuda()).logits
                if probe_index == 0:
                    expected_logits[current[:2]] = logits.detach().cpu()
                scalar = fisher_probe_scalar(logits, seed=7000 + probe_index,
                    token_scope='all', distribution='rademacher',
                    token_count_override=1536, global_row_offset=offset)
                scalar.backward()
                for name, parameter in parameters.items():
                    if parameter.grad is None:
                        raise AssertionError('resident packed source produced no gradient')
                    expected_grads[(*current, name)] = parameter.grad.detach().cpu()
                    parameter.grad = None
                del scalar, logits
            save()
    result['phases'].append({'phase': 'resident', 'kind': 'correctness', 'start_epoch': started, 'end_epoch': time.time()})
    for handle in handles:
        handle.remove()
    parameters.clear()
    parameter = handle = None
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    result['cross_partition']['logits'] = compare(expected_logits[(3, 0)], torch.cat([expected_logits[(1, i)] for i in range(3)]))
    result['cross_partition']['boundaries'] = [dict(layer=depth, **compare(
        expected_boundaries[(3, 0, depth)], torch.cat([expected_boundaries[(1, i, depth)] for i in range(3)])))
        for depth in range(24)]
    result['cross_partition']['routes'] = [dict(tensor=index, **compare(
        expected_routes[(3, 0)][index], torch.cat([expected_routes[(1, i)][index] for i in range(3)])))
        for index in range(len(expected_routes[(3, 0)]))]
    result['cross_partition']['gradients'] = []
    for probe_index in range(4):
        for name in leaf_names:
            full = expected_grads[(3, 0, probe_index, name)].float()
            partitioned = sum(expected_grads[(1, i, probe_index, name)].float() for i in range(3))
            result['cross_partition']['gradients'].append(dict(probe=probe_index, parameter=name, **compare(full, partitioned)))
            del full, partitioned
    save()
    for batch_size in (3, 1):
        profile = detect_profile(args.model)
        runner = build_streamed_causal_lm(args.model, device=torch.device('cuda'), dtype=torch.bfloat16,
            offload_folder=str(args.out / f'offload-{batch_size}'), profile=profile,
            attn_implementation='eager', max_cache_slots=24, prefetch_workers=4,
            prefetch_lookahead=4, cache_headroom_gb=4, prefetch_min_available_gb=2,
            require_prefetched_residency=True)
        model_identity = build_streamed_model_identity(runner, args.model,
            identity_cache_path=args.out / 'model_identity.json')
        result['source_model_identity'] = model_identity
        if source_execution_identity(runner.model) != result['reference_source_execution']:
            raise AssertionError('streamed source backend differs from resident')
        for depth in range(runner.num_layers):
            runner.context.schedule_prefetch(depth)
        record = {'batch_size': batch_size, 'boundaries': [], 'routes': [], 'logits': [], 'gradients': []}
        result['same_size'].append(record)
        original_capture, original_install, original_tail = runner.capture_boundaries, runner.context.install, runner.tail_logits
        capture_count = 0
        tail_count = 0
        grad_count = {name: 0 for name in leaf_names}
        offsets = list(range(0, 3, batch_size))
        def captured(calibration):
            nonlocal capture_count
            offset = offsets[capture_count]
            route_values = []
            def observe_routes(module, values):
                route_values.append(tuple(x.detach().cpu() for x in values))
            route_handle = runner.model.get_submodule(unit).register_forward_pre_hook(observe_routes)
            try:
                batch = original_capture(calibration)
            finally:
                route_handle.remove()
            if len(route_values) != 1:
                raise AssertionError('expected one original packed route boundary')
            for index, value in enumerate(route_values[0]):
                record['routes'].append(dict(offset=offset, tensor=index, **compare(expected_routes[(batch_size, offset)][index], value)))
            for depth in range(24):
                record['boundaries'].append(dict(offset=offset, layer=depth, **compare(
                    expected_boundaries[(batch_size, offset, depth)], batch.activations_cpu[depth + 1])))
            capture_count += 1
            save()
            return batch
        def tail(batch, hidden):
            nonlocal tail_count
            logits = original_tail(batch, hidden)
            if tail_count % 4 == 0:
                offset = offsets[tail_count // 4]
                record['logits'].append(dict(offset=offset, **compare(expected_logits[(batch_size, offset)], logits.detach().cpu())))
            tail_count += 1
            return logits
        def install(depth, **kwargs):
            installed = original_install(depth, **kwargs)
            if depth == 2 and capture_count == len(offsets):
                for name in leaf_names:
                    parameter = runner.model.get_parameter(f'{unit}.{name}')
                    parameter.requires_grad_(True)
                    def observe(value, name=name):
                        index = grad_count[name]
                        grad_count[name] += 1
                        probe_index, part = divmod(index, len(offsets))
                        offset = offsets[part]
                        expected = expected_grads.pop((batch_size, offset, probe_index, name))
                        actual = value.grad.detach().cpu()
                        record['gradients'].append(dict(offset=offset, probe=probe_index, parameter=name, **compare(expected, actual)))
                        save()
                    parameter.register_post_accumulate_grad_hook(observe)
            return installed
        runner.capture_boundaries, runner.tail_logits, runner.context.install = captured, tail, install
        names = [f'{unit}.{expert}.{role}' for expert in range(32) for role in ('w1', 'w3', 'w2')]
        started = time.time()
        compute_aura_cost_streamed(runner, ids, ['BF16'], n_probes=4, seed_base=7000,
            token_scope='all', production_cache=ProductionWeightCache(weights={}, levers={}),
            min_free_gib=0, joint_activation=True, include_routed_experts=True,
            model_identity=model_identity, formats_by_qname={name: ['BF16'] for name in names},
            probe_microbatch=batch_size, profile=profile)
        result['phases'].append({'phase': f'streamed-{batch_size}', 'kind': 'correctness', 'start_epoch': started, 'end_epoch': time.time()})
        if grad_count != {name: 4 * len(offsets) for name in leaf_names}:
            raise AssertionError('incomplete packed gradient comparisons')
        runner.context.shutdown()
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    result['passed'] = not expected_grads and all(
        value['equal'] for record in result['same_size']
        for kind in ('boundaries', 'routes', 'logits', 'gradients') for value in record[kind])
    result['env']['finished_epoch'] = time.time()
    result['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
    save()
    print(json.dumps({'passed': result['passed'], 'same_size_records': len(result['same_size'])}))
    if not result['passed']:
        raise AssertionError('same-size streamed source differs from resident oracle')


if __name__ == '__main__':
    main()
