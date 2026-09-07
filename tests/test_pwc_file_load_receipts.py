"""Integrity receipts belong to the PWC's actual tensor load and lifetime."""
from pathlib import Path
import hashlib
import pickle

import pytest
import torch

from prismaquant.production_weight_cache import ProductionWeightCache


FMT = 'TESSERA_E4M3_K1_R1024'


def make_cache(tmp_path, count=1, *, budget=1024):
    paths = {}
    tensors = {}
    for index in range(count):
        key = (f'unit{index}', FMT)
        tensors[key] = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4) + index
        paths[key] = tmp_path / f'unit{index}.pt'
        torch.save(tensors[key], paths[key])
    cache = ProductionWeightCache(weights={k: str(v) for k, v in paths.items()}, levers={})
    cache.enable_lru(budget)
    return cache, paths, tensors


def test_receipt_hashes_exact_single_read_and_tensor(tmp_path, monkeypatch):
    cache, paths, tensors = make_cache(tmp_path)
    key, path = next(iter(paths.items()))
    blob = path.read_bytes()
    calls = []
    original = Path.open
    class Counted:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return self.stream.__exit__(*args)
        def read(self, *args):
            raw = self.stream.read(*args); calls.append(('read', len(raw))); return raw
        def __getattr__(self, name):
            return getattr(self.stream, name)
    def counted(candidate, *args, **kwargs):
        stream = original(candidate, *args, **kwargs)
        if candidate == path:
            calls.append(('open', None)); return Counted(stream)
        return stream
    monkeypatch.setattr(Path, 'open', counted)
    cache.enable_file_load_receipts(max_file_bytes=len(blob))
    assert cache.prefetch([key], max_workers=1) == 1
    tensor = cache.get(*key)
    receipt = cache.file_load_receipt(key, tensor)
    assert torch.equal(tensor, tensors[key])
    assert receipt == {'path': str(path), 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()}
    assert calls == [('open', None), ('read', len(blob))]
    receipt['sha256'] = '0' * 64
    assert cache.file_load_receipt(key, tensor)['sha256'] == hashlib.sha256(blob).hexdigest()


@pytest.mark.parametrize('mutation', ['tensor', 'replacement', 'file'])
def test_receipt_refuses_drift_after_load(tmp_path, mutation):
    cache, paths, _ = make_cache(tmp_path)
    key, path = next(iter(paths.items()))
    cache.enable_file_load_receipts(max_file_bytes=path.stat().st_size)
    cache.prefetch([key], max_workers=1)
    tensor = cache.get(*key)
    if mutation == 'tensor':
        tensor[0, 0] += 1
    elif mutation == 'replacement':
        cache.weights[key] = tensor.clone()
    else:
        path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match='receipt|changed'):
        cache.file_load_receipt(key, tensor)


def test_receipt_dies_at_eviction_and_compaction(tmp_path):
    cache, paths, _ = make_cache(tmp_path, 2, budget=32)
    a, b = paths
    cache.enable_file_load_receipts(max_file_bytes=max(p.stat().st_size for p in paths.values()))
    old = cache.get(*a)
    cache.file_load_receipt(a, old)
    cache.get(*b)
    with pytest.raises(RuntimeError, match='receipt|resident'):
        cache.file_load_receipt(a, old)
    new = cache.get(*a)
    assert new is not old
    cache.file_load_receipt(a, new)
    cache.compact_for_pickle()
    with pytest.raises(RuntimeError, match='receipt|resident'):
        cache.file_load_receipt(a, new)
    cache.disable_file_load_receipts()
    assert pickle.loads(pickle.dumps(cache)).weights == {k: str(v) for k, v in paths.items()}


def test_read_buffer_bound_and_regular_file_required(tmp_path):
    cache, paths, _ = make_cache(tmp_path)
    key, path = next(iter(paths.items()))
    cache.enable_file_load_receipts(max_file_bytes=path.stat().st_size - 1)
    with pytest.raises((ValueError, RuntimeError), match='budget|bound'):
        cache.prefetch([key], max_workers=1)
    assert isinstance(cache.weights[key], str)
    cache.disable_file_load_receipts()
    target = path.with_suffix('.original'); path.rename(target); path.symlink_to(target)
    cache.enable_file_load_receipts(max_file_bytes=target.stat().st_size)
    with pytest.raises((ValueError, RuntimeError), match='regular|symlink'):
        cache.prefetch([key], max_workers=1)


def test_mutation_during_read_refuses_without_registering_tensor(tmp_path, monkeypatch):
    cache, paths, _ = make_cache(tmp_path)
    key, path = next(iter(paths.items()))
    cache.enable_file_load_receipts(max_file_bytes=path.stat().st_size + 20)
    original = Path.open
    class Mutating:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def read(self, *args):
            raw = self.stream.read(*args)
            with original(path, 'ab') as writer: writer.write(b'drift')
            return raw
        def __getattr__(self, name): return getattr(self.stream, name)
    def altered(candidate, *args, **kwargs):
        stream = original(candidate, *args, **kwargs)
        return Mutating(stream) if candidate == path else stream
    monkeypatch.setattr(Path, 'open', altered)
    with pytest.raises(RuntimeError, match='changed'):
        cache.prefetch([key], max_workers=1)
    assert isinstance(cache.weights[key], str)


def test_receipt_capture_requires_unloaded_entries(tmp_path):
    cache, paths, _ = make_cache(tmp_path)
    key, path = next(iter(paths.items()))
    cache.get(*key)
    with pytest.raises(RuntimeError, match='resident|unloaded'):
        cache.enable_file_load_receipts(max_file_bytes=path.stat().st_size)


def test_small_file_read_request_ignores_large_global_bound(tmp_path, monkeypatch):
    cache, paths, _ = make_cache(tmp_path)
    key, path = next(iter(paths.items()))
    size = path.stat().st_size
    cache.enable_file_load_receipts(max_file_bytes=size * 1000)
    original = Path.open
    class LimitedRead:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def read(self, requested):
            assert requested == size + 1, 'small file inherited the whole-cache read bound'
            return self.stream.read(requested)
        def __getattr__(self, name): return getattr(self.stream, name)
    def limited(candidate, *args, **kwargs):
        stream = original(candidate, *args, **kwargs)
        return LimitedRead(stream) if candidate == path else stream
    monkeypatch.setattr(Path, 'open', limited)
    assert cache.prefetch([key], max_workers=1) == 1
