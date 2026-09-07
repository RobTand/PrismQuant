"""PB-only canonical source/packed-joint screen on an explicit token subset."""
from __future__ import annotations
import argparse
import cProfile
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import time


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    import torch
    import transformers
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM
    from prismaquant import pretrained_initialization_contract
    from prismaquant.aura_cost import compute_aura_cost_streamed
    from prismaquant.cost_streaming import build_streamed_causal_lm, build_streamed_model_identity
    from prismaquant.joint_aura import validate_joint_aura_entry, source_execution_identity
    from prismaquant.kl_fisher import fisher_probe_scalar
    from prismaquant.model_profiles import detect_profile
    from prismaquant.production_weight_cache import ProductionWeightCache

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['source', 'cost'], required=True)
    parser.add_argument('--model', default='/mnt/shared/models/LFM2.5-8B-A1B-BF16')
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--tokens-sha256', required=True)
    parser.add_argument('--row', type=int, default=0)
    parser.add_argument('--rows', type=int, default=1)
    parser.add_argument('--probe-microbatch', type=int, default=0)
    parser.add_argument('--profile-tool', choices=['cprofile', 'torch', 'none'], default='cprofile')
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--subset-artifact', type=Path)
    parser.add_argument('--subset-artifact-sha256')
    parser.add_argument('--qualify-boundary', type=Path)
    parser.add_argument('--qualify-boundary-sha256')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--cache-sha256')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if file_sha(args.tokens) != args.tokens_sha256:
        raise ValueError('token artifact hash differs')
    all_ids = load_file(str(args.tokens))['calibration_ids']
    if all_ids.dtype != torch.int64 or all_ids.ndim != 2 or all_ids.shape[1] != 512:
        raise ValueError('screen needs the canonical int64 512-token draw')
    ids_cpu = all_ids[args.row:args.row + args.rows].contiguous()
    if args.row < 0 or args.rows < 1 or ids_cpu.shape != (args.rows, 512):
        raise ValueError('requested canonical calibration rows are absent')
    subset_path = args.out / 'calibration_subset.safetensors'
    if args.subset_artifact is not None:
        subset_path = args.subset_artifact
        if file_sha(subset_path) != args.subset_artifact_sha256 or not torch.equal(
                load_file(str(subset_path))['calibration_ids'], ids_cpu):
            raise ValueError('published operator subset differs from actual source tokens')
    else:
        from safetensors import safe_open
        with safe_open(str(args.tokens), framework='pt', device='cpu') as source:
            metadata = dict(source.metadata() or {})
        calibration_provenance = json.loads(metadata.get('calibration_provenance', '{}'))
        calibration_provenance.update(nsamples=args.rows, fit_tokens=ids_cpu.numel(),
            fit_ids_sha256=hashlib.sha256(ids_cpu.to(torch.int32).numpy().tobytes()).hexdigest())
        metadata['calibration_provenance'] = json.dumps(calibration_provenance, sort_keys=True)
        metadata['operator_screen_parent'] = json.dumps({
            'artifact': str(args.tokens), 'sha256': args.tokens_sha256,
            'row_start': args.row, 'rows': args.rows})
        save_file({'calibration_ids': ids_cpu}, str(subset_path), metadata=metadata)
    ids = ids_cpu.cuda()
    unit = 'model.layers.2.feed_forward.experts'
    names = [f'{unit}.{expert}.{role}' for expert in range(32) for role in ('w1', 'w3', 'w2')]
    policy = {'n_probes': 4, 'seed_base': 7000, 'token_scope': 'all',
              'temperature': 1.0, 'distribution': 'rademacher', 'normalization': 'global_kl_fisher'}
    result = {'schema': 'prismaquant.packed_joint_screen.v1', 'mode': args.mode,
        'env': {'host': socket.gethostname(), 'started_epoch': time.time(),
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'cuda': torch.version.cuda, 'affinity': sorted(os.sched_getaffinity(0)),
                'production_act_scales': os.environ.get('PRISMAQUANT_PROD_ACT_SCALES')},
        'calibration_subset': {'artifact': str(args.tokens), 'artifact_sha256': args.tokens_sha256,
            'subset_artifact': str(subset_path), 'subset_artifact_sha256': file_sha(subset_path),
            'full_shape': list(all_ids.shape), 'row': args.row, 'shape': list(ids_cpu.shape),
            'dtype': str(ids_cpu.dtype), 'sha256': hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest(),
            'scope': ('full_canonical_calibration' if args.row == 0 and args.rows == len(all_ids)
                      else 'canonical_sequence_subset')},
        'execution': {'probe_microbatch': args.probe_microbatch,
                      'profile_tool': args.profile_tool},
        'memory_observations': {'captured_boundary_bytes': 0, 'capture_batch_rows': [],
                                'tail_shapes': [], 'tail_calls': 0},
        'probe_policy': policy, 'unit_names': names, 'phases': [], 'gradient_comparisons': []}
    def save():
        (args.out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    def phase(label, call):
        torch.cuda.synchronize()
        started = time.time()
        try:
            if args.profile_tool == 'cprofile':
                profiler = cProfile.Profile()
                try:
                    value = profiler.runcall(call)
                finally:
                    profiler.dump_stats(str(args.out / (label + '.pstats')))
            elif args.profile_tool == 'torch':
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA],
                    profile_memory=True, record_shapes=False,
                ) as profiler:
                    value = call()
                profiler.export_chrome_trace(str(args.out / (label + '.trace.json')))
                (args.out / (label + '.profile.txt')).write_text(
                    profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=100))
            else:
                value = call()
            torch.cuda.synchronize()
        finally:
            result['phases'].append({'phase': label, 'kind': 'profile' if args.profile_tool != 'none' else 'correctness',
                                    'start_epoch': started, 'end_epoch': time.time()})
            save()
        return value
    def digest(tensor):
        return hashlib.sha256(memoryview(tensor.detach().cpu().contiguous().view(torch.uint8).numpy())).hexdigest()
    expected_gradients = {}
    expected_logits = None
    if args.mode == 'source':
        reference = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
            trust_remote_code=True, local_files_only=True, attn_implementation='eager').cuda().eval()
        result['model_load_contract'] = pretrained_initialization_contract(reference)
        result['reference_source_execution_identity'] = source_execution_identity(reference)
        raw_boundary = None
        if args.qualify_boundary is not None:
            if file_sha(args.qualify_boundary) != args.qualify_boundary_sha256:
                raise ValueError('retained boundary artifact hash differs')
            raw_boundary = torch.load(args.qualify_boundary, map_location='cpu', weights_only=False)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        source_parameters = {name: reference.get_parameter(f'{unit}.{name}')
                             for name in ('gate_up_proj', 'down_proj')}
        for parameter in source_parameters.values():
            parameter.requires_grad_(True)
        experts = reference.get_submodule(unit)
        parent = reference.get_submodule(unit.rsplit('.', 1)[0])
        original_experts_forward = experts.forward
        def inspect_source(hidden_states, top_k_index, top_k_weights):
            if raw_boundary is not None:
                coordinates = torch.stack((torch.zeros(512, dtype=torch.int64), torch.arange(512)), dim=1)
                actual_tensors = {'inputs': hidden_states, 'top_k_index': top_k_index,
                                  'top_k_weights': top_k_weights, 'expert_bias': parent.expert_bias,
                                  'coordinates': coordinates}
                checks = {}
                for name, actual in actual_tensors.items():
                    expected = raw_boundary[name]
                    cpu = actual.detach().cpu()
                    equal = cpu.dtype == expected.dtype and torch.equal(cpu, expected)
                    checks[name] = {'equal': equal, 'shape': list(cpu.shape), 'dtype': str(cpu.dtype),
                                    'actual_sha256': digest(cpu), 'captured_sha256': digest(expected)}
                    if not equal:
                        raise AssertionError(f'retained canonical boundary differs: {name}')
                result['retained_boundary_qualification'] = {
                    'schema': 'prismaquant.packed_source_boundary_qualification.v1',
                    'unit_qname': unit, 'artifact': str(args.qualify_boundary), 'artifact_sha256': args.qualify_boundary_sha256,
                    'boundary_metadata': raw_boundary['boundary_metadata'],
                    'source_execution_identity': result['reference_source_execution_identity'],
                    'calibration_subset': result['calibration_subset'], 'tensor_comparisons': checks}
            from prismaquant.measure_quant_cost import derive_per_expert_activations
            from torch.nn import functional as F
            derived = derive_per_expert_activations(experts, hidden_states, parent)
            # The shared derivation orders rows by top-k slot then token;
            # grouped_mm sorts the flattened token-major route list.
            flat = top_k_index.reshape(-1)
            _, permutation = torch.sort(flat)
            grouped_token = permutation // top_k_index.shape[1]
            grouped_slot = permutation % top_k_index.shape[1]
            expected_down = []
            for expert in range(experts.num_experts):
                slots, tokens = torch.where(top_k_index.T == expert)
                keys = tokens * top_k_index.shape[1] + slots
                selected = flat[permutation] == expert
                grouped_keys = grouped_token[selected] * top_k_index.shape[1] + grouped_slot[selected]
                order = torch.argsort(keys)
                expected_down.append(derived['down'][expert][order[torch.searchsorted(keys[order], grouped_keys)]])
            expected_down = torch.cat(expected_down)
            original_grouped = F.grouped_mm
            def grouped(inputs, weight, *, offs, **kwargs):
                if weight._base is source_parameters['down_proj']:
                    delta = (inputs.float() - expected_down.float()).abs()
                    result['derived_down_comparison'] = {'equal': torch.equal(inputs, expected_down),
                        'max_abs': float(delta.max()), 'different_elements': int(torch.count_nonzero(delta)),
                        'shape': list(inputs.shape), 'actual_sha256': digest(inputs),
                        'derived_sha256': digest(expected_down), 'scope': 'layer2_canonical_row0',
                        'source_backend': 'grouped_mm', 'derivation_backend': 'shared_F.linear'}
                return original_grouped(inputs, weight, offs=offs, **kwargs)
            F.grouped_mm = grouped
            try:
                return original_experts_forward(hidden_states, top_k_index, top_k_weights)
            finally:
                F.grouped_mm = original_grouped
        experts.forward = inspect_source
        def reference_probes():
            nonlocal expected_logits
            for index in range(policy['n_probes']):
                reference.zero_grad(set_to_none=True)
                logits = reference(ids, use_cache=False).logits
                if index == 0:
                    experts.forward = original_experts_forward
                    expected_logits = logits.detach().cpu()
                probe = fisher_probe_scalar(logits, seed=policy['seed_base'] + index,
                    token_scope='all', temperature=1.0, distribution='rademacher')
                probe.backward()
                for name, parameter in source_parameters.items():
                    if parameter.grad is None:
                        raise RuntimeError(f'reference gradient absent: {name}')
                    expected_gradients[(name, index)] = parameter.grad.detach().cpu()
                del logits, probe
        phase('resident_reference_forward_backward', reference_probes)
        reference.zero_grad(set_to_none=True)
        source_parameters.clear()
        experts = parent = original_experts_forward = None
        parameter = None
        del reference
        gc.collect()
        torch.cuda.empty_cache()

    profile = detect_profile(args.model)
    runner = build_streamed_causal_lm(args.model, device=torch.device('cuda'), dtype=torch.bfloat16,
        offload_folder=str(args.out / 'offload'), profile=profile, attn_implementation='eager',
        max_cache_slots=24, prefetch_workers=4, prefetch_lookahead=4, cache_headroom_gb=4,
        prefetch_min_available_gb=2, require_prefetched_residency=True)
    model_identity = build_streamed_model_identity(runner, args.model,
        identity_cache_path=args.out / 'model_identity.json')
    result['source_model_identity'] = model_identity
    result['streamed_backend'] = runner.model.config._attn_implementation
    result['source_execution'] = {'attention': runner.model.config._attn_implementation,
                                  'experts': runner.model.config._experts_implementation}
    result['streamed_source_execution_identity'] = source_execution_identity(runner.model)
    if args.mode == 'source':
        assert result['streamed_source_execution_identity'] == result['reference_source_execution_identity']
        if 'retained_boundary_qualification' in result:
            result['retained_boundary_qualification'].update(
                source_model_identity=model_identity, runtime=result['env'],
                streamed_source_execution_identity=result['streamed_source_execution_identity'])
    # Prefetch remains owned by the existing streaming context/cache.
    for layer in range(runner.num_layers):
        runner.context.schedule_prefetch(layer)
    if args.mode == 'source':
        with torch.inference_mode():
            actual_logits = runner(ids).logits.cpu()
        difference = (actual_logits.float() - expected_logits.float()).abs()
        result['source_forward'] = {'shape': list(actual_logits.shape),
            'equal': torch.equal(actual_logits, expected_logits), 'max_abs': float(difference.max())}
        del actual_logits, expected_logits, difference
        assert result['source_forward']['equal'], 'canonical full-vocabulary source forward differs'
        observed_counts = {'gate_up_proj': 0, 'down_proj': 0}
        original_install = runner.context.install
        def install(layer, **kwargs):
            installed = original_install(layer, **kwargs)
            if layer == 2:
                for name in observed_counts:
                    parameter = runner.model.get_parameter(f'{unit}.{name}')
                    parameter.requires_grad_(True)
                    def observe(value, name=name):
                        index = observed_counts[name]
                        observed_counts[name] += 1
                        actual = value.grad.detach().cpu()
                        expected = expected_gradients.pop((name, index))
                        equal = torch.equal(actual, expected)
                        delta = (actual.float() - expected.float()).abs()
                        result['gradient_comparisons'].append({'parameter': name, 'probe': index,
                            'shape': list(actual.shape), 'equal': equal, 'max_abs': float(delta.max()),
                            'actual_sha256': digest(actual), 'expected_sha256': digest(expected)})
                        save()
                        assert equal, f'packed source gradient differs: {name}, probe {index}'
                    parameter.register_post_accumulate_grad_hook(observe)
            return installed
        runner.context.install = install
        cache = ProductionWeightCache(weights={}, levers={})
        formats = ['BF16']
    else:
        if args.cache is None or not args.cache_sha256 or file_sha(args.cache) != args.cache_sha256:
            raise ValueError('actual original-wire production cache hash is required')
        with args.cache.open('rb') as handle:
            cache = pickle.load(handle)
        if not isinstance(cache, ProductionWeightCache):
            raise TypeError('expected the existing ProductionWeightCache pickle')
        result['production_cache'] = {'path': str(args.cache), 'sha256': args.cache_sha256}
        formats = ['TESSERA_E4M3_K1_R1024', 'BF16']
    original_capture = runner.capture_boundaries
    original_tail = runner.tail_logits
    def capture(ids):
        batch = original_capture(ids)
        observation = result['memory_observations']
        observation['capture_batch_rows'].append(len(ids))
        observation['captured_boundary_bytes'] += sum(x.numel() * x.element_size() for x in batch.activations_cpu)
        save()
        return batch
    def tail(batch, hidden):
        logits = original_tail(batch, hidden)
        observation = result['memory_observations']
        observation['tail_calls'] += 1
        shape = list(logits.shape)
        if shape not in observation['tail_shapes']:
            observation['tail_shapes'].append(shape)
        return logits
    runner.capture_boundaries = capture
    runner.tail_logits = tail
    def cost():
        return compute_aura_cost_streamed(runner, ids, formats, n_probes=4, seed_base=7000,
            **({'probe_microbatch': args.probe_microbatch} if args.probe_microbatch else {}),
            checkpoint_dir=args.checkpoint_dir, resume=args.resume,
            token_scope='all', temperature=1.0, min_free_gib=0, production_cache=cache,
            joint_activation=True, include_routed_experts=True, profile=profile,
            model_identity=model_identity, formats_by_qname={name: formats for name in names},
            checkpoint_identity_extra={'operator_screen_subset': result['calibration_subset'],
                                       'source_execution': result['source_execution']})
    payload = phase('streamed_packed_joint', cost)
    assert set(payload['costs']) == set(names)
    for name, rows in payload['costs'].items():
        assert set(rows) == set(formats)
        for row in rows.values():
            assert validate_joint_aura_entry(row)
            assert row['joint_operator_identity']['qname'] == name
            assert row['probe_ids'] == [7000, 7001, 7002, 7003]
    payload['provenance']['operator_screen_subset'] = result['calibration_subset']
    (args.out / 'joint-cost.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    if args.mode == 'source':
        assert not expected_gradients and observed_counts == {'gate_up_proj': 4, 'down_proj': 4}
    result['passed'] = True
    result['env']['finished_epoch'] = time.time()
    result['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
    result['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved()
    result['cost_sha256'] = file_sha(args.out / 'joint-cost.json')
    save()
    runner.context.shutdown()
    print(json.dumps({'passed': True, 'mode': args.mode, 'units': len(names),
                      'cost_sha256': result['cost_sha256']}), flush=True)


if __name__ == '__main__':
    main()
