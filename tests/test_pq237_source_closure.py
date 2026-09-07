"""The PQ237 seal must be an exact source closure, not just a hash of a list.

Every check here drives the public guard, `verify_source_manifest`, and the
shadowing case first proves that Python's real `PathFinder` resolves the
unsealed addition. Comparing our own two lists to each other would only prove
the arithmetic; it would not prove the import escape is closed.
"""
import hashlib
from importlib.machinery import PathFinder
import json
from pathlib import Path
import py_compile
import sys

import pytest

from experiments.pq237_joint_aura_streamed import verify_source_manifest


def _seal(root, manifest_path, *, extra=None):
    """Seal every regular file under `root` and write the manifest."""
    receipt = {"commit": "a" * 40, "files": {}, "symlinks": {}}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            receipt["symlinks"][relative] = str(path.readlink())
        elif path.is_file():
            receipt["files"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt.update(extra or {})
    manifest_path.write_text(json.dumps(receipt, sort_keys=True))
    return receipt


@pytest.fixture
def sealed(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "measurement.py").write_text("VALUE = 'sealed'\n")
    manifest = tmp_path / "manifest.json"
    _seal(root, manifest)
    verify_source_manifest(root, manifest)
    return root, manifest


def test_unlisted_package_shadows_a_sealed_module_and_is_refused(sealed):
    root, manifest = sealed
    package = root / "measurement"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 'unsealed'\n")
    # The escape is real before it is refused: the finder resolves the package.
    spec = PathFinder.find_spec("measurement", [str(root)])
    assert Path(spec.origin) == package / "__init__.py"
    with pytest.raises(ValueError, match="exact file closure"):
        verify_source_manifest(root, manifest)


def test_bare_directory_does_not_shadow_and_is_not_refused(sealed):
    """The closure is narrower than "any new directory", and stays correct."""
    root, manifest = sealed
    (root / "measurement").mkdir()
    spec = PathFinder.find_spec("measurement", [str(root)])
    assert Path(spec.origin) == root / "measurement.py"
    verify_source_manifest(root, manifest)


def test_unlisted_sourceless_bytecode_is_refused(tmp_path, sealed):
    """A bare `name.pyc` in a path entry imports on its own, cache or not."""
    root, manifest = sealed
    donor = tmp_path / "donor.py"
    donor.write_text("VALUE = 'unsealed'\n")
    py_compile.compile(str(donor), cfile=str(root / "extra.pyc"), doraise=True)
    spec = PathFinder.find_spec("extra", [str(root)])
    assert Path(spec.origin) == root / "extra.pyc"
    with pytest.raises(ValueError, match="exact file closure"):
        verify_source_manifest(root, manifest)


def test_unlisted_extension_module_is_refused(sealed):
    root, manifest = sealed
    (root / "accelerator.so").write_bytes(b"\x7fELF not a real module")
    with pytest.raises(ValueError, match="exact file closure"):
        verify_source_manifest(root, manifest)


def test_sealed_file_replaced_by_a_symlink_is_refused(sealed):
    """file_sha256 reads through a link, so the kind must be checked first."""
    root, manifest = sealed
    elsewhere = root.parent / "elsewhere.py"
    elsewhere.write_text("VALUE = 'sealed'\n")
    (root / "measurement.py").unlink()
    (root / "measurement.py").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="no longer a regular file"):
        verify_source_manifest(root, manifest)


def test_sealed_symlink_replaced_by_a_file_is_refused(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "measurement.py").write_text("VALUE = 'sealed'\n")
    (root / "alias.py").symlink_to(root / "measurement.py")
    manifest = tmp_path / "manifest.json"
    _seal(root, manifest)
    verify_source_manifest(root, manifest)
    (root / "alias.py").unlink()
    (root / "alias.py").write_text("VALUE = 'unsealed'\n")
    with pytest.raises(ValueError, match="no longer a symlink"):
        verify_source_manifest(root, manifest)


def test_declared_excluded_symlinks_are_verified_not_merely_tolerated(tmp_path):
    """`excluded_symlinks` says the target is outside the seal, not unchecked."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "measurement.py").write_text("VALUE = 'sealed'\n")
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 'outside'\n")
    (root / "calibration.py").symlink_to(outside)
    manifest = tmp_path / "manifest.json"
    receipt = _seal(root, manifest)
    excluded = {"calibration.py": receipt["symlinks"].pop("calibration.py")}
    receipt["excluded_symlinks"] = excluded
    manifest.write_text(json.dumps(receipt, sort_keys=True))
    verify_source_manifest(root, manifest)
    (root / "calibration.py").unlink()
    (root / "calibration.py").write_text("VALUE = 'unsealed'\n")
    with pytest.raises(ValueError, match="no longer a symlink"):
        verify_source_manifest(root, manifest)


def test_in_tree_bytecode_cache_is_refused_unless_the_reader_redirects(
        tmp_path, sealed, monkeypatch):
    """A timestamp-validated .pyc runs instead of the sealed source it names."""
    root, manifest = sealed
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / f"measurement.cpython-{sys.version_info[0]}{sys.version_info[1]}.pyc").write_bytes(b"")
    with pytest.raises(ValueError, match="bytecode caches"):
        verify_source_manifest(root, manifest)
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "pycache"))
    verify_source_manifest(root, manifest)


def test_additions_under_a_dot_directory_are_unreachable_and_not_refused(sealed):
    """The one structural exclusion: no module name component starts with '.'."""
    root, manifest = sealed
    hidden = root / ".vendored"
    hidden.mkdir()
    (hidden / "__init__.py").write_text("VALUE = 'unreachable'\n")
    assert PathFinder.find_spec(".vendored", [str(root)]) is None
    verify_source_manifest(root, manifest)


def test_a_listed_entry_under_a_dot_directory_is_still_verified(tmp_path):
    """Excluding a directory governs unlisted additions, never listed entries."""
    root = tmp_path / "source"
    (root / ".github").mkdir(parents=True)
    (root / "measurement.py").write_text("VALUE = 'sealed'\n")
    (root / ".github" / "ci.yml").write_text("on: push\n")
    manifest = tmp_path / "manifest.json"
    _seal(root, manifest)
    verify_source_manifest(root, manifest)
    (root / ".github" / "ci.yml").write_text("on: pull_request\n")
    with pytest.raises(ValueError, match="source changed after manifest"):
        verify_source_manifest(root, manifest)
