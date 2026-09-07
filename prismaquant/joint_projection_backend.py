"""Explicit joint-projection backend admission and prewarm, owning no tensors.

The default remains the native torch expression. The opt-in fused backend loads
only a prebuilt binary with the packaged numerical qualification; compilation
belongs to the separately admitted build/qualification workflow.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import platform
import re
import subprocess

import torch

from .kernels import joint_projection_reduce as kernel

SCHEMA = 'prismaquant.joint_projection_backend.v1'
FUSED_NAME = 'fused_fp32_v1'
QUALIFICATION_PATH = Path(__file__).with_name('kernels') / 'joint_projection_reduce_qualification.json'
REFERENCE_IDENTITY = {'schema': SCHEMA, 'name': 'torch', 'expression': '(left * right).sum()'}
_PREWARM_SEAL = object()
# Code modules only: all resident input tensors remain owned by the caller.
_LOADED_MODULES = {}
_WARMED_DEVICES = set()


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=1)
def _qualification():
    # Immutable package metadata, like the loaded code; row admission must not
    # reopen the qualification file for every format/probe result.
    data = QUALIFICATION_PATH.read_bytes()
    value = json.loads(data)
    if value.get('schema') != 'prismaquant.joint_projection_qualification.v1' or value.get('status') != 'qualified':
        raise RuntimeError('joint projection backend has no qualified packaged runtime')
    return value, hashlib.sha256(data).hexdigest()


def normalize_projection_backend(config=None):
    """Validate the complete plan selector without accessing CUDA or a compiler."""
    if config is None:
        return {'name': 'torch'}
    if not isinstance(config, Mapping):
        raise ValueError('joint projection backend must be an explicit mapping')
    config = deepcopy(dict(config))
    if config == {'name': 'torch'}:
        return config
    if set(config) != {'name', 'binary'} or config['name'] != FUSED_NAME:
        raise ValueError('unsupported joint projection backend or selector fields')
    binary = config['binary']
    if (not isinstance(binary, dict) or set(binary) != {'path', 'sha256'}
            or not isinstance(binary['path'], str) or not binary['path']
            or not isinstance(binary['sha256'], str) or re.fullmatch('[a-f0-9]{64}', binary['sha256']) is None):
        raise ValueError('fused joint projection requires an independently bound binary path/SHA256')
    qualification, _ = _qualification()
    if binary['sha256'] != qualification['build']['binary_sha256']:
        raise ValueError('joint projection binary is outside the packaged qualification')
    return config


def _runtime_identity(device):
    """Read the runtime/compiler/header identity once, before any lease."""
    props = torch.cuda.get_device_properties(device)
    include = Path(torch.__file__).parent / 'include'
    qualification, _ = _qualification()
    return {'torch': str(torch.__version__), 'torch_git': torch.version.git_version,
            'cuda': torch.version.cuda, 'machine': platform.machine(),
            'device': {'name': props.name, 'major': props.major, 'minor': props.minor,
                       'multi_processor_count': props.multi_processor_count},
            'headers': {name: _sha(include / name) for name in qualification['runtime']['headers']},
            'compiler': {
                'nvcc_version': subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True),
                'cxx_version': subprocess.check_output(['c++', '--version'], text=True)}}


def _require_runtime(actual, expected):
    if actual != expected:
        changed = sorted(key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key))
        raise RuntimeError('joint projection unqualified runtime identity: ' + ', '.join(changed))


def validate_projection_backend_identity(identity):
    """Validate the serialized arithmetic contract without loading GPU code."""
    if identity == REFERENCE_IDENTITY:
        return
    qualification, digest = _qualification()
    expected = {'schema': SCHEMA, 'name': FUSED_NAME, 'qualification_sha256': digest,
                'build': qualification['build'], 'runtime': qualification['runtime'],
                'qualified_shapes': qualification['qualified_shapes'],
                'ineligible_layout': 'torch_reference'}
    if identity != expected:
        raise ValueError('joint projection arithmetic has an unqualified backend identity')


class _TorchProjection:
    @property
    def identity(self):
        return deepcopy(REFERENCE_IDENTITY)

    def require_device(self, device):
        pass

    @staticmethod
    def product_sum(left, right):
        return (left * right).sum()


class _FusedProjection:
    def __init__(self, module, device, identity, *, seal):
        if seal is not _PREWARM_SEAL:
            raise RuntimeError('joint projection fused backend requires explicit prewarm')
        self._module = module
        self._device = device
        self._identity = deepcopy(identity)
        self._shapes = frozenset(tuple(shape) for shape in identity['qualified_shapes'])

    @property
    def identity(self):
        return deepcopy(self._identity)

    def require_device(self, device):
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if device != self._device:
            raise RuntimeError('joint projection fused backend was not prewarmed for this device')

    def product_sum(self, left, right):
        self.require_device(left.device)
        self.require_device(right.device)
        if tuple(left.shape) not in self._shapes or left.shape != right.shape:
            raise RuntimeError('joint projection matrix shape is outside the packaged qualification')
        if torch.is_grad_enabled() and (left.requires_grad or right.requires_grad):
            raise RuntimeError('joint projection reduction has no autograd registration; use under no_grad')
        if not kernel.fast_path_eligible(left, right):
            return (left * right).sum()
        return self._module.mul_sum(left, right)


def require_prewarmed_projection(backend, *, device):
    """Lease admission does no compilation, file reads, or implicit prewarm."""
    if backend is None:
        return _TorchProjection()
    if type(backend) not in (_TorchProjection, _FusedProjection):
        raise RuntimeError('joint projection backend must be explicitly prewarmed before the lease')
    backend.require_device(device)
    return backend


def prewarm_projection_backend(config=None, *, device):
    """Verify/load a qualified binary before source/cotangent hot execution."""
    if type(config) in (_TorchProjection, _FusedProjection):
        return require_prewarmed_projection(config, device=device)
    config = normalize_projection_backend(config)
    if config['name'] == 'torch':
        return _TorchProjection()
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('qualified joint projection requires a CUDA device')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    qualification, digest = _qualification()
    runtime = _runtime_identity(device)
    _require_runtime(runtime, qualification['runtime'])
    build = qualification['build']
    if (kernel._source_digest() != build['source_sha256'] or kernel.CPP_FLAGS != build['cpp_flags']
            or kernel.CUDA_FLAGS != build['cuda_flags']):
        raise RuntimeError('joint projection kernel source/compiler flags differ from qualification')
    path = Path(config['binary']['path']).resolve(strict=True)
    if _sha(path) != build['binary_sha256']:
        raise RuntimeError('joint projection binary bytes differ from qualification')
    # The qualified extension name is part of its Python initialization ABI.
    name = build['module_name']
    key = (str(path), build['binary_sha256'])
    module = _LOADED_MODULES.get(key)
    if module is None:
        loader = importlib.machinery.ExtensionFileLoader(name, str(path))
        spec = importlib.util.spec_from_file_location(name, path, loader=loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        if Path(module.__file__).resolve() != path or _sha(module.__file__) != build['binary_sha256']:
            raise RuntimeError('joint projection actually loaded a different binary')
        _LOADED_MODULES[key] = module
    warm_key = (key, device.index)
    if warm_key not in _WARMED_DEVICES:
        # Temporary zero operands load the qualified kernel on the current
        # stream before a lease; only the code/device marker survives.
        with torch.no_grad():
            zeros = torch.zeros(qualification['qualified_shapes'][0], device=device, dtype=torch.float32)
            actual = module.mul_sum(zeros, zeros)
            expected = (zeros * zeros).sum()
            if actual.view(torch.int32).item() != expected.view(torch.int32).item():
                raise RuntimeError('joint projection prewarm changed reference reduction bits')
        del zeros, actual, expected
        _WARMED_DEVICES.add(warm_key)
    identity = {'schema': SCHEMA, 'name': FUSED_NAME, 'qualification_sha256': digest,
                'build': deepcopy(build), 'runtime': runtime,
                'qualified_shapes': deepcopy(qualification['qualified_shapes']),
                'ineligible_layout': 'torch_reference'}
    validate_projection_backend_identity(identity)
    return _FusedProjection(module, device, identity, seal=_PREWARM_SEAL)
