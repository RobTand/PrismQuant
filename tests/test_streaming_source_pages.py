"""CPU byte/range checks; CUDA lifecycle is explicitly mocked here."""
import json
import os
import stat
import threading
import weakref
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prismaquant import layer_streaming as ls


def checkpoint(tmp_path):
    page = os.sysconf('SC_PAGE_SIZE')
    tensors = {name: torch.arange(page * 3 + 7, dtype=torch.int32)
               for name in ('adjacent_before', 'selected', 'unread_after')}
    path = tmp_path / 'weights.safetensors'
    save_file(tensors, path)
    with path.open('rb') as f:
        size = int.from_bytes(f.read(8), 'little')
        header = json.loads(f.read(size))
    return path, tensors, header, 8 + size, page


def test_release_only_complete_selected_payload_pages(tmp_path, monkeypatch):
    path, tensors, header, base, page = checkpoint(tmp_path)
    calls = []
    real_advice = os.posix_fadvise

    def advise(fd, a, n, advice):
        calls.append((a, n, advice))
        real_advice(fd, a, n, advice)

    monkeypatch.setattr(os, 'posix_fadvise', advise)
    ls._advise_consumed_safetensors_pages(str(path), ['selected'])
    begin, end = (base + x for x in header['selected']['data_offsets'])
    expected_begin = ((begin + page - 1) // page) * page
    expected_end = (end // page) * page
    assert calls == [(expected_begin, expected_end - expected_begin, os.POSIX_FADV_DONTNEED)]
    assert expected_begin >= begin >= base and expected_end <= end
    with safe_open(path, framework='pt') as f:
        for name, value in tensors.items():
            assert torch.equal(f.get_tensor(name), value)


@pytest.mark.parametrize('kind', ['fifo', 'character', 'missing_api'])
def test_unsupported_source_never_advised(tmp_path, monkeypatch, kind):
    path, *_ = checkpoint(tmp_path)
    calls = []
    monkeypatch.setattr(os, 'posix_fadvise', lambda *args: calls.append(args))
    if kind == 'fifo':
        path.unlink()
        os.mkfifo(path)
    elif kind == 'character':
        original = os.fstat
        monkeypatch.setattr(os, 'fstat', lambda fd: SimpleNamespace(st_mode=stat.S_IFCHR, st_size=original(fd).st_size))
    else:
        monkeypatch.delattr(os, 'posix_fadvise')
    ls._advise_consumed_safetensors_pages(str(path), ['selected'])
    assert calls == []


def test_no_complete_page_no_advice(tmp_path, monkeypatch):
    path = tmp_path / 'tiny.safetensors'
    save_file({'tiny': torch.arange(3)}, path)
    calls = []
    monkeypatch.setattr(os, 'posix_fadvise', lambda *args: calls.append(args))
    ls._advise_consumed_safetensors_pages(str(path), ['tiny'])
    assert calls == []


@pytest.mark.parametrize('enabled', [False, True])
def test_cpu_mapping_stays_live_and_is_never_advised(tmp_path, monkeypatch, enabled):
    path, tensors, *_ = checkpoint(tmp_path)
    monkeypatch.setenv('PRISMAQUANT_RELEASE_SOURCE_PAGES', '1' if enabled else '0')
    calls = []
    monkeypatch.setattr(os, 'posix_fadvise', lambda *args: calls.append(args))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: pytest.fail('CPU read synchronized CUDA'))
    out = ls._read_layer_to_device('', {'selected': str(path)}, {'selected': 'selected'},
                                   torch.int32, torch.device('cpu'))
    assert torch.equal(out['selected'], tensors['selected'])
    assert calls == []


class FakeCudaTensor:
    """Actual CPU payload copied at mocked CUDA transfer boundary."""
    device = torch.device('cuda:0')

    def __init__(self, value):
        self.value = value.clone()

    def __getattr__(self, key):
        return getattr(self.value, key)


def mocked_cuda_reader(path, monkeypatch, *, direct=False, failure=None,
                       wait_for_first_advice=False):
    """Track each reader context separately; no real CUDA operation occurs."""
    chunks = []
    current = threading.local()
    first_advised = threading.Event()
    real_to = torch.Tensor.to

    def copy(value, target, *args, **kwargs):
        if isinstance(target, torch.device) and target.type == 'cuda':
            current.chunk['events'].append('copy')
            current.chunk['staged'].append(weakref.ref(value))
            return FakeCudaTensor(value)
        return real_to(value, target, *args, **kwargs)

    class Context:
        def __enter__(self):
            self.ctx = safe_open(path, framework='pt')
            self.file = self.ctx.__enter__()
            self.chunk = dict(events=['open'], staged=[], keys=[])
            current.chunk = self.chunk
            chunks.append(self.chunk)
            return self

        def get_tensor(self, name):
            chunk = self.chunk
            chunk['keys'].append(name)
            number = int(name.split('_')[-1])
            if wait_for_first_advice and number == 5:
                assert first_advised.wait(3), 'finished chunk pages retained behind sibling read'
            if failure == 'read' and number == 2:
                raise RuntimeError('source read failed')
            value = self.file.get_tensor('selected')
            if direct:
                chunk['events'].append('copy')
                return FakeCudaTensor(value)
            return value

        def __exit__(self, *args):
            self.ctx.__exit__(*args)
            self.chunk['events'].append('close')

    class Event:
        def record(self, stream):
            assert stream == threading.get_ident()
            self.chunk = current.chunk
            self.chunk['events'].append('record')
            if failure == 'event_record' and self.chunk['keys'][0] == 'selected_0':
                raise RuntimeError('event record failed')

        def synchronize(self):
            chunk = self.chunk
            assert 'close' in chunk['events']
            assert all(ref() is not None for ref in chunk['staged'])
            chunk['events'].append('sync')
            if failure == 'event_sync' and chunk['keys'][0] == 'selected_0':
                raise RuntimeError('event sync failed')

    def advise(shard, keys, expected_stat):
        chunk = next(chunk for chunk in chunks if chunk['keys'] == keys)
        assert all(ref() is None for ref in chunk['staged'])
        chunk['events'].append('advise')
        if keys[0] == 'selected_0':
            first_advised.set()

    monkeypatch.setattr(torch.Tensor, 'to', copy)
    monkeypatch.setattr(ls, 'safe_open', lambda *args, **kwargs: Context())
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args: threading.get_ident())
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: pytest.fail('device-wide synchronization'))
    monkeypatch.setattr(ls, '_advise_consumed_safetensors_pages', advise)
    return chunks


def read_mock_layer(path, count):
    names = [f'selected_{i}' for i in range(count)]
    return ls._read_layer_to_device('', {name: str(path) for name in names},
                                    {name: name for name in names},
                                    torch.int32, torch.device('cuda:0'))


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('threads', [1, 4])
def test_cuda_release_follows_each_chunks_copies_and_mapping_close(tmp_path, monkeypatch, direct, enabled, threads):
    path, tensors, *_ = checkpoint(tmp_path)
    monkeypatch.setenv('PRISMAQUANT_RELEASE_SOURCE_PAGES', '1' if enabled else '0')
    monkeypatch.setenv('PRISMAQUANT_DIRECT_CUDA_LOAD', '1' if direct else '0')
    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', str(threads))
    chunks = mocked_cuda_reader(path, monkeypatch, direct=direct)
    out = read_mock_layer(path, 20 if threads > 1 else 1)
    assert all(torch.equal(value.value, tensors['selected']) for value in out.values())
    for chunk in chunks:
        events = chunk['events']
        if enabled:
            assert events[-4:] == ['close', 'record', 'sync', 'advise']
            assert events.count('record') == events.count('sync') == 1
        else:
            assert 'advise' not in events and 'sync' not in events


def test_finished_chunk_releases_pages_while_sibling_is_still_reading(tmp_path, monkeypatch):
    path, *_ = checkpoint(tmp_path)
    monkeypatch.setenv('PRISMAQUANT_RELEASE_SOURCE_PAGES', '1')
    monkeypatch.setenv('PRISMAQUANT_DIRECT_CUDA_LOAD', '0')
    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '4')
    chunks = mocked_cuda_reader(path, monkeypatch, wait_for_first_advice=True)
    assert len(read_mock_layer(path, 20)) == 20
    assert len(chunks) == 4


@pytest.mark.parametrize('failure', ['read', 'event_record', 'event_sync'])
def test_failed_parallel_chunk_drains_copies_without_advising_its_pages(tmp_path, monkeypatch, failure):
    path, *_ = checkpoint(tmp_path)
    monkeypatch.setenv('PRISMAQUANT_RELEASE_SOURCE_PAGES', '1')
    monkeypatch.setenv('PRISMAQUANT_DIRECT_CUDA_LOAD', '0')
    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '4')
    chunks = mocked_cuda_reader(path, monkeypatch, failure=failure)
    with pytest.raises(RuntimeError, match='source read failed|event .* failed') as caught:
        read_mock_layer(path, 20)
    assert len(chunks) == 4
    for chunk in chunks:
        events = chunk['events']
        assert 'close' in events and events.count('record') == 1
        failed = chunk['keys'][0] == 'selected_0'
        assert events.count('sync') == (0 if failed and failure == 'event_record' else 1)
        assert ('advise' in events) is not failed
        if failed and failure.startswith('event'):
            assert caught.value.__traceback__ is not None
            assert all(ref() is not None for ref in chunk['staged'])


def test_unexpected_cpu_backed_output_is_not_advised(tmp_path, monkeypatch):
    path, tensors, *_ = checkpoint(tmp_path)
    monkeypatch.setenv('PRISMAQUANT_RELEASE_SOURCE_PAGES', '1')
    monkeypatch.setenv('PRISMAQUANT_DIRECT_CUDA_LOAD', '1')
    monkeypatch.setattr(ls, 'safe_open', lambda *args, **kwargs: safe_open(path, framework='pt'))
    monkeypatch.setattr(os, 'posix_fadvise', lambda *args: pytest.fail('live CPU output advised'))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: pytest.fail('CPU output synchronized'))
    out = ls._read_layer_to_device('', {'selected': str(path)}, {'selected': 'selected'},
                                   torch.int32, torch.device('cuda:0'))
    assert torch.equal(out['selected'], tensors['selected'])


def test_changed_source_identity_refuses_advice(tmp_path, monkeypatch):
    path, *_ = checkpoint(tmp_path)
    expected = path.stat()
    replacement = tmp_path / 'replacement.safetensors'
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    monkeypatch.setattr(os, 'posix_fadvise', lambda *args: pytest.fail('replacement source advised'))
    with pytest.raises(RuntimeError, match='source changed'):
        ls._advise_consumed_safetensors_pages(str(path), ['selected'], expected)
