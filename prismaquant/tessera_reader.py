"""An explicitly bound Tessera consumer beside an unchanged primary producer.

The primary ``tessera`` package derives original source/H/encoder inputs. This
reader consumes those independently derived identities and original bytes;
its own source digest is separate provenance, never substituted into a producer
receipt. No serving runtime is imported and no module is monkeypatched.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import re
import sys
from types import SimpleNamespace

SOURCE_SUFFIXES = {'.py', '.cu', '.cuh', '.cpp', '.h'}


def _source_tree(root):
    """The owner's encoder_source_sha256 framing, read without importing it."""
    digest = hashlib.sha256()
    files = {}
    for path in sorted(p for p in root.rglob('*') if p.suffix in SOURCE_SUFFIXES):
        if path.is_symlink() or not path.is_file():
            raise ValueError('reader source must be regular files in its declared package')
        raw = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode() + b'\0'); digest.update(raw); digest.update(b'\0')
        files[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
    if not files or not (root / '__init__.py').is_file():
        raise ValueError('reader source must name a complete package directory')
    return digest.hexdigest(), files


class _ReaderSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, expected):
        super().__init__(fullname, path)
        self.expected = expected

    def get_code(self, fullname):
        # Always read the bound source, including after another interpreter
        # wrote bytecode. Every imported file must still match the package seal.
        raw = Path(self.path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.expected:
            raise ImportError('reader source changed after its package checksum')
        tree = ast.parse(raw, filename=self.path)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or '']
            elif isinstance(node, ast.Call) and node.args:
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', '')
                if name in {'import_module', '__import__', 'files'} and isinstance(node.args[0], ast.Constant):
                    names = [node.args[0].value]
            if any(isinstance(name, str) and (name == 'tessera' or name.startswith('tessera.')) for name in names):
                raise ImportError(f'{fullname} contains an absolute tessera import; reader namespace would escape')
        return compile(tree, self.path, 'exec', dont_inherit=True)


class _ReaderFinder(importlib.abc.MetaPathFinder):
    def __init__(self, namespace, root, files):
        self.namespace, self.root, self.files = namespace, root, files

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.namespace + '.'):
            return None
        relative = fullname[len(self.namespace) + 1:]
        if relative.split('.')[0] in {'serving', 'stock', 'kernel_window_gemv'}:
            raise ImportError('the bound reader does not import serving or serving-kernel modules')
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.origin is None:
            raise ImportError(f'bound reader module is absent: {fullname}')
        origin = str(Path(spec.origin).resolve())
        if origin not in self.files or not origin.endswith('.py'):
            raise ImportError(f'bound reader module escapes declared source: {fullname}')
        spec.loader = _ReaderSourceLoader(fullname, origin, self.files[origin])
        return spec


def load_declared_reader(record):
    """Import a hash-bound consumer without replacing any producer module."""
    if record is None:
        return None
    if not isinstance(record, dict) or set(record) != {'path', 'source_sha256'}:
        raise ValueError('reader requires an explicit package path and source_sha256')
    if not re.fullmatch(r'[0-9a-f]{64}', str(record['source_sha256'])):
        raise ValueError('reader requires an exact source SHA256')
    root = Path(record['path']).resolve()
    digest, files = _source_tree(root)
    if digest != record['source_sha256']:
        raise ValueError('reader package checksum changed')
    namespace = 'tessera_reader_' + digest
    if namespace not in sys.modules:
        finder = _ReaderFinder(namespace, root, files)
        spec = importlib.util.spec_from_file_location(namespace, root / '__init__.py',
            submodule_search_locations=[str(root)],
            loader=_ReaderSourceLoader(namespace, str(root / '__init__.py'), files[str(root / '__init__.py')]))
        module = importlib.util.module_from_spec(spec)
        sys.meta_path.insert(0, finder)
        sys.modules[namespace] = module
        try:
            spec.loader.exec_module(module)
            importlib.import_module(namespace + '.cached_unit')
            importlib.import_module(namespace + '.unit_artifact')
        except BaseException:
            for name in tuple(sys.modules):
                if name == namespace or name.startswith(namespace + '.'):
                    del sys.modules[name]
            sys.meta_path.remove(finder)
            raise
    else:
        actual = Path(sys.modules[namespace].__file__).resolve().parent
        if actual != root:
            raise ValueError('reader namespace is already bound to a different source path')
    cached = importlib.import_module(namespace + '.cached_unit')
    artifact = importlib.import_module(namespace + '.unit_artifact')
    return SimpleNamespace(identity={'schema': 'prismaquant.tessera_reader.v1',
        'path': str(root), 'source_sha256': digest, 'namespace': namespace},
        verify_cached_unit=cached.verify_cached_unit, read_unit_artifact=artifact.read_unit_artifact)
