"""A declared reader cannot replace or import the primary producer package."""
import hashlib
import sys
from pathlib import Path

import pytest


def package(tmp_path, *, escaping=False):
    root = tmp_path / 'tessera'; root.mkdir()
    (root / '__init__.py').write_text('MARKER = "reader"\n')
    (root / 'cached_unit.py').write_text('def verify_cached_unit(*args): return __name__, args\n')
    (root / 'unit_artifact.py').write_text(
        ('import tessera\n' if escaping else '') +
        'def read_unit_artifact(*args, **kwargs): return __name__, args, kwargs\n')
    digest = hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0')
        digest.update(path.read_bytes()); digest.update(b'\0')
    return {'path': str(root), 'source_sha256': digest.hexdigest()}


def test_reader_namespace_preserves_primary_and_binds_actual_sources(tmp_path):
    import tessera
    from prismaquant.tessera_reader import load_declared_reader
    primary = dict((name, module) for name, module in sys.modules.items()
                   if name == 'tessera' or name.startswith('tessera.'))
    declared = package(tmp_path)
    reader = load_declared_reader(declared)
    assert reader.identity['source_sha256'] == declared['source_sha256']
    assert reader.read_unit_artifact(b'x')[0].startswith('tessera_reader_')
    assert reader.verify_cached_unit(b'x', {}, {})[0].startswith('tessera_reader_')
    assert sys.modules['tessera'] is tessera
    assert all(sys.modules.get(name) is module for name, module in primary.items())
    reader.identity['source_sha256'] = '0' * 64
    assert load_declared_reader(declared).identity['source_sha256'] == declared['source_sha256']


def test_reader_refuses_absolute_producer_import(tmp_path):
    from prismaquant.tessera_reader import load_declared_reader
    with pytest.raises(ImportError, match='absolute.*tessera'):
        load_declared_reader(package(tmp_path, escaping=True))


def test_reader_refuses_changed_source(tmp_path):
    from prismaquant.tessera_reader import load_declared_reader
    declared = package(tmp_path)
    (Path(declared['path']) / '__init__.py').write_text('CHANGED = True\n')
    with pytest.raises(ValueError, match='checksum'):
        load_declared_reader(declared)
