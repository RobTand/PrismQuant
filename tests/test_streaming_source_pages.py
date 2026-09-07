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


@pytest.mark.parametrize('direct', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('threads', [1, 4])
def test_cuda_release_follows_copies_and_all_mapping_closes(tmp_path, monkeypatch, direct, enabled, threads):
    path, tensors, *_ = checkpoint(tmp_path)
    monkeypatch.setenv('PRISMAQUANT_RELEASE_SOURCE_PAGES', '1' if enabled else '0')
    monkeypatch.setenv('PRISMAQUANT_DIRECT_CUDA_LOAD', '1' if direct else '0')
    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', str(threads))
    events = []
    staged = []
    real_to = torch.Tensor.to

    def copy(value, target, *args, **kwargs):
        if isinstance(target, torch.device) and target.type == 'cuda':
            events.append('copy')
            staged.append(weakref.ref(value))
            return FakeCudaTensor(value)
        return real_to(value, target, *args, **kwargs)

    class Context:
        def __enter__(self):
            self.ctx = safe_open(path, framework='pt')
            self.file = self.ctx.__enter__()
            events.append('open')
            return self

        def get_tensor(self, name):
            value = self.file.get_tensor(name)
            if direct:
                events.append('copy')
                return FakeCudaTensor(value)
            return value

        def __exit__(self, *args):
            self.ctx.__exit__(*args)
            events.append('close')

    monkeypatch.setattr(torch.Tensor, 'to', copy)
    monkeypatch.setattr(ls, 'safe_open', lambda *args, **kwargs: Context())

    class Event:
        def record(self, stream):
            assert stream == threading.get_ident()
            events.append('record')

        def synchronize(self):
            assert all(ref() is not None for ref in staged)
            events.append('sync')

    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args: threading.get_ident())
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: pytest.fail('device-wide synchronization'))
    monkeypatch.setattr(os, 'posix_fadvise', lambda *args: events.append('advise'))
    names = [f'selected_{i}' for i in range(20 if threads > 1 else 1)]
    out = ls._read_layer_to_device('', {name: str(path) for name in names},
                                   {name: 'selected' for name in names},
                                   torch.int32, torch.device('cuda:0'))
    assert all(torch.equal(out[name].value, tensors['selected']) for name in names)
    if enabled:
        assert events.index('advise') > max(i for i, event in enumerate(events) if event == 'close')
        assert events.index('advise') > events.index('sync') > max(i for i, event in enumerate(events) if event == 'copy')
    else:
        assert 'advise' not in events and 'sync' not in events


@pytest.mark.parametrize('failure', ['read', 'event_record', 'event_sync'])
def test_failed_parallel_read_drains_chunk_copies_without_advice(tmp_path, monkeypatch, failure):
    path, *_ = checkpoint(tmp_path)
    monkeypatch.setenv('PRISMAQUANT_RELEASE_SOURCE_PAGES', '1')
    monkeypatch.setenv('PRISMAQUANT_DIRECT_CUDA_LOAD', '0')
    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '4')
    events, staged = [], []
    real_to = torch.Tensor.to

    def copy(value, target, *args, **kwargs):
        if isinstance(target, torch.device) and target.type == 'cuda':
            staged.append(weakref.ref(value))
            return FakeCudaTensor(value)
        return real_to(value, target, *args, **kwargs)

    class Context:
        def __enter__(self):
            self.ctx = safe_open(path, framework='pt')
            self.file = self.ctx.__enter__()
            return self.file

        def __exit__(self, *args):
            self.ctx.__exit__(*args)
            events.append('close')

    class Event:
        def record(self, stream):
            assert stream == threading.get_ident()
            events.append('record')
            if failure == 'event_record' and events.count('record') == 1:
                raise RuntimeError('event record failed')

        def synchronize(self):
            assert events.count('close') == 4
            assert all(ref() is not None for ref in staged)
            events.append('sync')
            if failure == 'event_sync' and events.count('sync') == 1:
                raise RuntimeError('event sync failed')

    monkeypatch.setattr(torch.Tensor, 'to', copy)
    monkeypatch.setattr(ls, 'safe_open', lambda *args, **kwargs: Context())
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args: threading.get_ident())
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: pytest.fail('device-wide synchronization'))
    monkeypatch.setattr(os, 'posix_fadvise', lambda *args: pytest.fail('failed gather advised source pages'))
    names = [f'selected_{i}' for i in range(20)]
    with pytest.raises(Exception, match='missing|event .* failed') as caught:
        ls._read_layer_to_device('', {name: str(path) for name in names},
                                 {name: 'missing' if i == 2 and failure == 'read' else 'selected'
                                  for i, name in enumerate(names)},
                                 torch.int32, torch.device('cuda:0'))
    assert events.count('record') == 4
    assert events.count('sync') == (3 if failure == 'event_record' else 4)
    if failure.startswith('event'):
        assert caught.value.__traceback__ is not None
        assert all(ref() is not None for ref in staged)


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
