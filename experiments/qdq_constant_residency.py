"""Bounded, admitted A/B of the served QDQ owner in actual joint projections.

Uses one retained canonical unit and original measured PWC renders. The four
fixed random cotangents qualify projection arithmetic, not model Fisher costs.
"""
import argparse
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import resource
import socket
import time

import torch
from safetensors import safe_open

from prismaquant import format_registry as fr
from prismaquant import nvfp4_activation_contract as owner
from prismaquant.joint_aura import SignedJointProjectionLease, activation_identity, prefetch_joint_cache
from prismaquant.production_weight_cache import _cb_cache_tensor_identity
from prismaquant.tessera_calibration_cache import prefetch_capture, require_capture_contract


ROOT = Path('/mnt/shared/tessera-measurements/first-model-20260907')
NAME = 'model.layers.23.feed_forward.experts.0.w1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def io_counters():
    return {key: int(value) for key, value in
            (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


def main(*, variant_controller=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-seconds', type=float, default=35)
    parser.add_argument('--qualification-only', action='store_true')
    parser.add_argument('--profile-qualified', action='store_true')
    parser.add_argument('--source-receipt', type=Path)
    parser.add_argument('--source-sha256')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if variant_controller is not None:
        variant_controller.persist_binary(args.output)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    receipt = {'schema': 'pq.qdq_constant_residency.v1', 'passed': False, 'phases': [],
               'env': {'started_epoch': time.time(), 'host': socket.gethostname(),
                       'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                       'device': torch.cuda.get_device_name(), 'affinity': sorted(os.sched_getaffinity(0))}}
    if variant_controller is not None:
        receipt.update(schema=variant_controller.schema, variant=variant_controller.identity)

    def save():
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))

    save()
    if variant_controller is None:
        baseline_path = ROOT / 'qdq-residency/frozen-source/baseline_nvfp4_activation_contract.py'
        baseline_text = baseline_path.read_text()
        tree = ast.parse(baseline_text)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'nvfp4_activation_qdq_served')
        baseline_function = ast.get_source_segment(baseline_text, function)
        namespace = dict(owner.__dict__)
        exec(compile(baseline_function, str(baseline_path), 'exec'), namespace)
        before = namespace['nvfp4_activation_qdq_served']
        after = owner.nvfp4_activation_qdq_served
        after_function = inspect.getsource(after)

        def activate(function):
            owner.nvfp4_activation_qdq_served = function
    else:
        baseline_path = variant_controller.baseline_path
        baseline_function = variant_controller.before_source
        after_function = variant_controller.after_source
        before, after = variant_controller.before, variant_controller.after
        activate = variant_controller.activate
    (args.output / 'before.py').write_text(baseline_function)
    (args.output / 'after.py').write_text(after_function)
    prepared_path = ROOT / 'full-model-joint-aura/run-02/prepare/prepared.json'
    assert sha(prepared_path) == '69a56bff2d29beaf38759b309dee1ac94ca9e091e50f846590d8b51cb0ba5908'
    prepared = json.loads(prepared_path.read_text())
    cache_path = Path(prepared['production_cache']['path'])
    assert sha(cache_path) == prepared['production_cache']['sha256']
    cache = pickle.loads(cache_path.read_bytes())
    formats = [fmt for fmt in prepared['formats_by_qname'][NAME] if fmt != 'BF16']
    render_paths = {fmt: Path(cache.weights[NAME, fmt]) for fmt in formats}
    for fmt, path in render_paths.items():
        assert sha(path) == cache.metadata['verified_cells'][NAME, fmt]['render_file_sha256']
    manifest_path = ROOT / 'capture-reuse/canonical/capture_manifest.json'
    manifest_sha = 'db3cd996ee8a3ac82d62c6e7e2f23cdb995874b831adcf9360acdae682654823'
    manifest = require_capture_contract(manifest_path, expected_sha256=manifest_sha)
    census_path = ROOT / 'canonical-census-512/census.json'
    assert sha(census_path) == manifest['identity']['census_sha256']
    census = json.loads(census_path.read_text())
    capture = manifest['entries'][NAME]
    x_path = manifest_path.parent / capture['path']
    assert sha(x_path) == capture['sha256'] == '87ffaceab268f0747aaa325aeb872cb358cd1de1dbba6b6f44044833e05d71ba'
    (acts, hessians, counts, maxima), capture_receipt = prefetch_capture(
        manifest_path, expected_identity=manifest['identity'], census=census,
        names=[NAME], device='cuda', expected_sha256=manifest_sha)
    captured = acts[NAME]
    assert isinstance(captured, torch.Tensor) and captured.shape == (512, 2048)
    with safe_open('/mnt/shared/models/LFM2.5-8B-A1B-BF16/model.safetensors', framework='pt', device='cpu') as model:
        source = model.get_tensor(NAME + '.weight')
    assert source.dtype == torch.bfloat16 and source.shape == (1792, 2048)
    source_identity = _cb_cache_tensor_identity(source)
    for fmt in formats:
        assert source_identity == cache.metadata['verified_cells'][NAME, fmt]['source_weight']
    prefetch = prefetch_joint_cache(cache, [NAME], {NAME: formats}, max_resident_bytes=256 * 2**20)
    source = source.cuda()
    renders = {fmt: cache.get(NAME, fmt).cuda() for fmt in formats}
    for fmt, value in renders.items():
        assert _cb_cache_tensor_identity(value) == cache.metadata['verified_cells'][NAME, fmt]['rendered_weight']
    deltas = {(NAME, fmt): value.float() - source.float() for fmt, value in renders.items()}
    specs = {NAME: {fmt: fr.get_format(fmt) for fmt in formats}}
    for fmt in formats:
        assert activation_identity(specs[NAME][fmt], cache.activation_max_abs, NAME) == cache.metadata['verified_cells'][NAME, fmt]['activation']
    layer = torch.nn.Linear(2048, 1792, bias=False, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(source)
    x = captured.to(device='cuda', dtype=torch.bfloat16).requires_grad_()
    assert torch.equal(x.float(), captured)
    generator = torch.Generator(device='cuda').manual_seed(7000)
    gradients = ([torch.randn((512, 1792), generator=generator, device='cuda', dtype=torch.bfloat16) for _ in range(4)]
                 if args.source_receipt is None else [])
    probe_inputs = [[(x, gradient)] for gradient in gradients]
    source_capture = None
    if args.source_receipt is not None:
        assert args.source_sha256 and sha(args.source_receipt) == args.source_sha256
        source_capture = json.loads(args.source_receipt.read_text())
        assert source_capture['schema'] == 'pq.actual_routed_projection_capture.v1' and source_capture['passed'] is True
        assert source_capture['unit'] == NAME and source_capture['global_token_count'] == 262144
        assert source_capture['source_model_identity'] == prepared['source_model_identity']
        assert source_capture['source_execution'] == prepared['source_execution']
        assert source_capture['calibration'] == prepared['calibration_input']
        assert source_capture['source_weight'] == source_identity
        payload_path = Path(source_capture['payload']['path'])
        assert sha(payload_path) == source_capture['payload']['sha256']
        payload = torch.load(payload_path, map_location='cpu', weights_only=True)
        assert _cb_cache_tensor_identity(payload['source_weight']) == source_identity
        probe_inputs = [[] for _ in range(4)]
        for entry, bound in zip(payload['records'], source_capture['records'], strict=True):
            assert entry['probe'] == bound['probe'] and entry['row'] == bound['row']
            assert tensor_sha(entry['x']) == bound['x_sha256'] and tensor_sha(entry['g']) == bound['g_sha256']
            probe_inputs[entry['probe']].append((entry['x'].cuda().requires_grad_(), entry['g'].cuda()))
        assert [len(values) for values in probe_inputs] == [4, 4, 4, 4]
        actual_prefix = torch.cat([inputs.detach() for inputs, _ in probe_inputs[0]])
        assert len(actual_prefix) <= len(captured)
        order_path = ROOT / 'qdq-residency/capture-order-diagnosis.json'
        order_proof = json.loads(order_path.read_text())
        assert order_proof['source_receipt_sha256'] == args.source_sha256
        assert order_proof['canonical_sha256'] == capture['sha256']
        order = order_proof['actual_to_canonical_row']
        assert sorted(order) == list(range(len(actual_prefix)))
        start = 0
        for inputs, _ in probe_inputs[0]:
            end = start + len(inputs)
            assert sorted(order[start:end]) == list(range(start, end))
            start = end
        reordered_canonical = captured[torch.tensor(order, device=captured.device)]
        assert torch.equal(actual_prefix.float(), reordered_canonical), 'actual B1 routed input bytes differ from original canonical rows'
    receipt['inputs'] = {'unit': NAME, 'rows': 512, 'formats': formats, 'probe_count': 4,
                         'cotangent_kind': 'fixed random BF16; not model-output Fisher probes',
                         'seed': 7000, 'prepared': {'path': str(prepared_path), 'sha256': sha(prepared_path)},
                         'cache_sha256': sha(cache_path), 'capture_sha256': sha(x_path),
                         'manifest_sha256': manifest_sha, 'census_sha256': sha(census_path),
                         'capture_receipt': capture_receipt, 'capture_count': counts[NAME],
                         'capture_maximum': maxima[NAME], 'source_identity': source_identity,
                         'render_file_sha256': {fmt: sha(path) for fmt, path in render_paths.items()},
                         'x_sha256': tensor_sha(x), 'gradient_sha256': [tensor_sha(g) for g in gradients],
                         'prefetch': prefetch, 'before_module_sha256': sha(baseline_path),
                         'before_function_sha256': sha(args.output / 'before.py'),
                         'after_function_sha256': sha(args.output / 'after.py')}
    if source_capture is not None:
        receipt['inputs'].update(cotangent_kind='actual source-tail Fisher; unchanged B1 rows 0–3',
            rows=None, actual_source_rows=source_capture['rows'],
            canonical_prefix_permutation_equal_rows=len(actual_prefix),
            canonical_order_diagnosis_sha256=sha(order_path),
            source_capture={'path': str(args.source_receipt), 'sha256': args.source_sha256},
            actual_invocations=source_capture['records'], rows_per_probe=[sum(len(v[0]) for v in values) for values in probe_inputs],
            projection_scope='four actual per-expert invocations accumulated per probe; not full-model cost')
    save()
    lease_factory = (variant_controller.make_lease if variant_controller is not None
                     and hasattr(variant_controller, 'make_lease') else SignedJointProjectionLease)
    lease = lease_factory({NAME: layer}, specs, deltas, activation_max_abs=cache.activation_max_abs)

    def cycle(check=False):
        if variant_controller is not None:
            variant_controller.checking = check
        results = []
        for values in probe_inputs:
            lease.begin_probe()
            hashes = []
            for inputs, gradient in values:
                layer.weight.grad = None
                inputs.grad = None
                y = layer(inputs)
                y.backward(gradient)
                if check:
                    hashes.append({'output': tensor_sha(y), 'input_grad': tensor_sha(inputs.grad),
                                   'weight_grad': tensor_sha(layer.weight.grad)})
            terms = lease.finish_probe()
            if check:
                results.append({'terms': {fmt: terms[NAME, fmt] for fmt in formats}, 'invocations': hashes})
        return results

    with lease:
        activate(before)
        expected = cycle(check=True)
        activate(after)
        actual = cycle(check=True)
        assert actual == expected, 'baseline/output/backward/signed projection bits differ'
        receipt['qualification'] = {'exact_projection_and_forward_backward': True, 'probes': actual}
        if variant_controller is not None:
            receipt['qualification']['reductions'] = variant_controller.reduction_checks
        if not args.qualification_only:
            timings = []
            for function in (before, after):
                activate(function)
                cycle()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(3):
                    cycle()
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) / 3)
            count = max(1, math.ceil(args.target_seconds / min(timings)))
            assert count <= 5000, count
            receipt['calibration'] = {'seconds_per_cycle': timings, 'cycles_per_arm': count}
            save()
            for index, (label, function) in enumerate((('before', before), ('after', after), ('after', after), ('before', before))):
                activate(function)
                receipt['active_phase'] = {'phase': f'{index}-{label}', 'cycles': count, 'started_epoch': time.time()}
                save()
                print(json.dumps({'starting': receipt['active_phase']}), flush=True)
                torch.cuda.synchronize()
                start, counters = time.time(), io_counters()
                for _ in range(count):
                    cycle()
                torch.cuda.synchronize()
                end = time.time()
                after_counters = io_counters()
                phase = {'phase': f'{index}-{label}', 'kind': 'measured', 'arm': label,
                         'start_epoch': start, 'end_epoch': end, 'seconds': end - start,
                         'cycles': count, 'projection_terms': count * 4 * len(formats),
                         'io_delta': {key: after_counters[key] - value for key, value in counters.items()}}
                receipt['phases'].append(phase)
                receipt.pop('active_phase', None)
                save()
                print(json.dumps(phase), flush=True)
        if not args.qualification_only or args.profile_qualified:
            for label, function in (('before', before), ('after', after)):
                activate(function)
                torch.cuda.synchronize()
                start = time.time()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True, profile_memory=True, with_stack=True) as profiler:
                    for _ in range(3):
                        cycle()
                torch.cuda.synchronize()
                end = time.time()
                profiler.export_chrome_trace(str(args.output / f'{label}-trace.json'))
                events = profiler.key_averages()
                (args.output / f'{label}-cpu.txt').write_text(events.table(sort_by='self_cpu_time_total', row_limit=100))
                (args.output / f'{label}-cuda.txt').write_text(events.table(sort_by='self_device_time_total', row_limit=100))
                receipt['phases'].append({'phase': label + '-profile', 'kind': 'profile', 'start_epoch': start, 'end_epoch': end})
                save()
    activate(after)
    receipt.update(passed=True, telemetry=lease.telemetry, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   cuda_max_allocated=torch.cuda.max_memory_allocated(), cuda_max_reserved=torch.cuda.max_memory_reserved())
    receipt['env']['finished_epoch'] = time.time()
    save()
    print(json.dumps({'passed': True, 'receipt': str(args.output / 'receipt.json')}), flush=True)


if __name__ == '__main__':
    main()
