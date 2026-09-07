"""Research-only fused product/reduction, not wired into production projections.

The candidate reuses the running PyTorch release's native CUDA reduction tree.
Eligibility keeps its indexing/alignment identical to an allocated FP32 product;
other layouts, dtypes, empty inputs and CPU inputs use the unchanged expression.
The caller owns all input residency. This module caches compiled code, no tensors.
"""
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import torch


CUDA_FLAGS = ['-O3', '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true', '-lineinfo']
CPP_FLAGS = ['-O3']


def _source_digest():
    digest = hashlib.sha256()
    for suffix in ('.cpp', '.cu'):
        digest.update(Path(__file__).with_suffix(suffix).read_bytes())
    digest.update(str((torch.__version__, torch.version.cuda, CUDA_FLAGS, CPP_FLAGS)).encode())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_backend():
    """Compile/load before the measured projection lease is entered."""
    from torch.utils.cpp_extension import load

    return load(name='pq_joint_projection_reduce_' + _source_digest()[:16],
                sources=[str(Path(__file__).with_suffix(suffix)) for suffix in ('.cpp', '.cu')],
                extra_cflags=CPP_FLAGS, extra_cuda_cflags=CUDA_FLAGS,
                with_cuda=True, verbose=True)


def fast_path_eligible(left, right):
    return (left.device.type == 'cuda' and left.device == right.device
            and left.dtype == right.dtype == torch.float32 and left.shape == right.shape
            and left.is_contiguous() and right.is_contiguous()
            and 0 < left.numel() < (2**31 // 4)
            and left.data_ptr() % 16 == 0 and right.data_ptr() % 16 == 0)


def fused_product_sum(left, right):
    """Opt-in experimental equivalent of ``(left * right).sum()``."""
    if not fast_path_eligible(left, right):
        return (left * right).sum()
    if torch.is_grad_enabled() and (left.requires_grad or right.requires_grad):
        raise RuntimeError('joint projection reduction has no autograd registration; use under no_grad')
    return load_backend().mul_sum(left, right)


def build_identity():
    """Bind the actual loaded binary, compiler flags and reduction headers."""
    module = load_backend()
    include = Path(torch.__file__).parent / 'include'
    names = ['ATen/native/cuda/Reduce.cuh', 'ATen/native/cuda/MemoryAccess.cuh',
             'ATen/native/cuda/thread_constants.h', 'ATen/cuda/DeviceUtils.cuh',
             'ATen/TensorIterator.h']
    return {'source_sha256': _source_digest(),
            'python_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'torch': str(torch.__version__),
            'torch_git': torch.version.git_version, 'cuda': torch.version.cuda,
            'arch_list': os.environ.get('TORCH_CUDA_ARCH_LIST'),
            'cpp_flags': CPP_FLAGS, 'cuda_flags': CUDA_FLAGS,
            'binary_path': module.__file__,
            'binary_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            'headers': {name: hashlib.sha256((include / name).read_bytes()).hexdigest() for name in names}}
