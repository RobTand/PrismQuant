"""PB-only fixed-logit CUDA control for the row-indexed probe layout."""
from pathlib import Path
import argparse
import json
import socket
import time


def main():
    import torch
    from prismaquant.kl_fisher import fisher_probe_scalar
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'prismaquant.joint_aura.cuda_probe_partition.v1',
              'env': {'host': socket.gethostname(), 'started_epoch': time.time(),
                      'torch': torch.__version__, 'cuda': torch.version.cuda},
              'shape': [3, 512, 128000], 'comparisons': []}
    torch.manual_seed(971)
    logits = torch.randn(3, 512, 128000, device='cuda', dtype=torch.bfloat16).requires_grad_(True)
    for seed in range(7000, 7004):
        kwargs = dict(seed=seed, token_scope='all', distribution='rademacher',
                      token_count_override=1536)
        scalar = fisher_probe_scalar(logits, global_row_offset=0, **kwargs)
        expected, = torch.autograd.grad(scalar, logits)
        for size in (1, 2):
            actual = torch.empty_like(expected)
            for offset in range(0, 3, size):
                local = logits[offset:offset + size].detach().requires_grad_(True)
                scalar = fisher_probe_scalar(local, global_row_offset=offset, **kwargs)
                actual[offset:offset + size], = torch.autograd.grad(scalar, local)
            delta = actual.float() - expected.float()
            result['comparisons'].append({'seed': seed, 'batch_rows': size,
                'equal': torch.equal(actual, expected), 'max_abs': float(delta.abs().max()),
                'relative_l2': float(delta.norm() / expected.float().norm())})
    result['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
    result['env']['finished_epoch'] = time.time()
    result['passed'] = all(x['equal'] for x in result['comparisons'])
    (args.out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if not result['passed']:
        raise AssertionError('fixed-logit CUDA Fisher probes depend on the partition')


if __name__ == '__main__':
    main()
