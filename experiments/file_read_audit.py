"""Count selected file reads in one admitted, synchronous measurement scope."""
from __future__ import annotations

import builtins
from contextlib import AbstractContextManager
import io
import os
import threading


class _ReadFile:
    def __init__(self, stream, audit, path):
        self.stream, self.audit, self.path = stream, audit, path

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def read(self, *args):
        raw = self.stream.read(*args)
        self.audit.record(self.path, 'read_calls', 1)
        self.audit.record(self.path, 'read_bytes', len(raw))
        return raw

    def readinto(self, buffer):
        count = self.stream.readinto(buffer)
        self.audit.record(self.path, 'read_calls', 1)
        self.audit.record(self.path, 'read_bytes', count or 0)
        return count


class FileReadAudit(AbstractContextManager):
    """Observe real Python/C++-backed file adapters without changing their bytes.

    Path.open uses io.open; torch.serialization uses builtins.open and passes
    its handle into the C++ reader. Wrapping both records the read/readinto
    calls made by those handles. No payload is retained. Scope owners must join
    their file-loader threads before exiting, as PWC.prefetch already does.
    """
    def __init__(self, files):
        self.files = {os.path.abspath(os.fspath(path)): kind for path, kind in files.items()}
        self.counts = {path: {'kind': kind, 'opens': 0, 'read_calls': 0, 'read_bytes': 0}
                       for path, kind in self.files.items()}
        self.lock = threading.Lock()

    def record(self, path, key, amount):
        with self.lock:
            self.counts[path][key] += amount

    def _wrapper(self, original):
        def opened(file, *args, **kwargs):
            stream = original(file, *args, **kwargs)
            path = os.path.abspath(os.fsdecode(file)) if isinstance(file, (str, bytes, os.PathLike)) else None
            if path in self.files:
                self.record(path, 'opens', 1)
                return _ReadFile(stream, self, path)
            return stream
        return opened

    def __enter__(self):
        self.original_builtin, self.original_io = builtins.open, io.open
        builtins.open = self._wrapper(self.original_builtin)
        io.open = self._wrapper(self.original_io)
        return self

    def __exit__(self, *_args):
        builtins.open, io.open = self.original_builtin, self.original_io

    def summary(self):
        by_kind = {}
        for record in self.counts.values():
            target = by_kind.setdefault(record['kind'], {'files': 0, 'opens': 0, 'read_calls': 0, 'read_bytes': 0})
            target['files'] += 1
            for key in ('opens', 'read_calls', 'read_bytes'):
                target[key] += record[key]
        return {'by_kind': by_kind, 'files': self.counts}
