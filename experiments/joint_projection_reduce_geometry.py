"""Numerical qualification on retained source weights, captured X and wire rungs.

Cotangents are explicitly seeded local operands, not source-model Fisher probes.
Each invocation qualifies one original model unit; PrismaBuild owns placement.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import socket
import time

import torch
from safetensors import safe_open

from experiments.qdq_constant_residency import ROOT, sha, tensor_sha
from experiments.joint_projection_reduce_bench import Candidate
from prismaquant import format_registry as fr
from prismaquant.joint_aura import SignedJointProjectionLease, activation_identity, prefetch_joint_cache
from prismaquant.kernels import joint_projection_reduce as kernel
from prismaquant.production_weight_cache import _cb_cache_tensor_identity
from prismaquant.tessera_calibration_cache import prefetch_capture, require_capture_contract

BINARY = ROOT / 'joint-fused-projection/actual-ab-01/pq_joint_projection_reduce_17d100e93d552b85.so'
BINARY_SHA = '9305c183c5214dc5ff1f73382963f8275eb1b6197cb8d173c6a93bffd700c115'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    assert sha(BINARY) == BINARY_SHA
    spec = importlib.util.spec_from_file_location(BINARY.stem, BINARY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel.load_backend = lambda: module  # Research replay of the already measured binary.
    candidate = Candidate(args.output)
    candidate.persist_binary(args.output)
    prepared_path = ROOT / 'full-model-joint-aura/run-02/prepare/prepared.json'
    assert sha(prepared_path) == '69a56bff2d29beaf38759b309dee1ac94ca9e091e50f846590d8b51cb0ba5908'
    prepared = json.loads(prepared_path.read_text())
    cache_path = Path(prepared['production_cache']['path'])
    assert sha(cache_path) == prepared['production_cache']['sha256']
    cache = pickle.loads(cache_path.read_bytes())
    name = args.unit
    formats = [fmt for fmt in prepared['formats_by_qname'][name] if fmt != 'BF16']
    assert len(formats) == 7
    for fmt in formats:
        assert sha(Path(cache.weights[name, fmt])) == cache.metadata['verified_cells'][name, fmt]['render_file_sha256']
    manifest_path = ROOT / 'capture-reuse/canonical/capture_manifest.json'
    manifest_sha = 'db3cd996ee8a3ac82d62c6e7e2f23cdb995874b831adcf9360acdae682654823'
    manifest = require_capture_contract(manifest_path, expected_sha256=manifest_sha)
    census_path = ROOT / 'canonical-census-512/census.json'
    assert sha(census_path) == manifest['identity']['census_sha256']
    census = json.loads(census_path.read_text())
    (acts, _, counts, maxima), capture_receipt = prefetch_capture(
        manifest_path, expected_identity=manifest['identity'], census=census,
        names=[name], device='cuda', expected_sha256=manifest_sha)
    with safe_open('/mnt/shared/models/LFM2.5-8B-A1B-BF16/model.safetensors', framework='pt', device='cpu') as model:
        source = model.get_tensor(name + '.weight')
    source_identity = _cb_cache_tensor_identity(source)
    for fmt in formats:
        assert source_identity == cache.metadata['verified_cells'][name, fmt]['source_weight']
    prefetch = prefetch_joint_cache(cache, [name], {name: formats}, max_resident_bytes=1024 * 2**20)
    source = source.cuda()
    renders = {fmt: cache.get(name, fmt).cuda() for fmt in formats}
    for fmt, value in renders.items():
        assert _cb_cache_tensor_identity(value) == cache.metadata['verified_cells'][name, fmt]['rendered_weight']
    deltas = {(name, fmt): value.float() - source.float() for fmt, value in renders.items()}
    specs = {name: {fmt: fr.get_format(fmt) for fmt in formats}}
    for fmt in formats:
        assert activation_identity(specs[name][fmt], cache.activation_max_abs, name) == cache.metadata['verified_cells'][name, fmt]['activation']
    layer = torch.nn.Linear(source.shape[1], source.shape[0], bias=False, device='cuda', dtype=torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(source)
    x = acts[name].to(dtype=torch.bfloat16).requires_grad_()
    assert torch.equal(x.float(), acts[name])
    generator = torch.Generator(device='cuda').manual_seed(7000)
    gradients = [torch.randn((len(x), source.shape[0]), generator=generator, device='cuda', dtype=torch.bfloat16) for _ in range(4)]
    receipt = {'schema': 'pq.joint_projection_geometry_qualification.v1', 'passed': False,
               'unit': name, 'source_weight': source_identity, 'x_sha256': tensor_sha(x),
               'x_shape': list(x.shape), 'formats': formats, 'prefetch': prefetch,
               'prepared': {'path': str(prepared_path), 'sha256': sha(prepared_path)},
               'capture_receipt': capture_receipt, 'capture_count': counts[name], 'capture_maximum': maxima[name],
               'cotangents': {'kind': 'seeded BF16 qualification operands; not source Fisher', 'seed': 7000,
                              'sha256': [tensor_sha(g) for g in gradients]},
               'candidate': candidate.identity,
               'env': {'host': socket.gethostname(), 'started_epoch': time.time()}}
    (args.output / 'before.py').write_text(candidate.before_source)
    (args.output / 'after.py').write_text(candidate.after_source)
    lease = SignedJointProjectionLease({name: layer}, specs, deltas, activation_max_abs=cache.activation_max_abs)

    def cycle():
        result = []
        for gradient in gradients:
            lease.begin_probe()
            x.grad = layer.weight.grad = None
            output = layer(x)
            output.backward(gradient)
            terms = lease.finish_probe()
            result.append({'signed': {fmt: terms[name, fmt] for fmt in formats}, 'output': tensor_sha(output),
                           'dx': tensor_sha(x.grad), 'dw': tensor_sha(layer.weight.grad)})
        return result

    try:
        with lease:
            candidate.activate(candidate.before)
            expected = cycle()
            candidate.activate(candidate.after)
            candidate.checking = True
            actual = cycle()
        assert actual == expected, 'signed components or forward/backward hashes changed'
        assert len(candidate.reduction_checks) == 52 and all(x['equal'] for x in candidate.reduction_checks)
        receipt.update(passed=True, qualification={'probes': actual, 'reductions': candidate.reduction_checks})
    finally:
        candidate.activate(candidate.before)
        receipt['env']['finished_epoch'] = time.time()
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'passed': receipt['passed'], 'unit': name, 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
