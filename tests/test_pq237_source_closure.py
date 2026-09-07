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
import shutil
import subprocess
import sys

import pytest

from experiments import pq237_joint_aura_streamed
from experiments.pq237_joint_aura_streamed import verify_source_manifest

EXPERIMENTS = Path(pq237_joint_aura_streamed.__file__).resolve().parent


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
    """`excluded_symlinks` says the target is outside the seal, not unchecked.

    The link is a data link, named as the three in the historical receipts are
    (`calibration/diverse-v1.jsonl` and its siblings). A `.py` name here would
    be a different claim -- that unsealed *code* outside the root is tolerated
    -- and the test below shows it is refused.
    """
    root = tmp_path / "source"
    (root / "calibration").mkdir(parents=True)
    (root / "measurement.py").write_text("VALUE = 'sealed'\n")
    outside = tmp_path / "diverse-v1.jsonl"
    outside.write_text('{"text": "outside"}\n')
    (root / "calibration" / "diverse-v1.jsonl").symlink_to(outside)
    manifest = tmp_path / "manifest.json"
    receipt = _seal(root, manifest)
    relative = "calibration/diverse-v1.jsonl"
    receipt["excluded_symlinks"] = {relative: receipt["symlinks"].pop(relative)}
    manifest.write_text(json.dumps(receipt, sort_keys=True))
    verify_source_manifest(root, manifest)
    (root / relative).unlink()
    (root / relative).write_text('{"text": "unsealed"}\n')
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


# --- The bootstrap: the policy must be checked before it is trusted -----------
#
# Everything above drives `verify_source_manifest` in this process, where the
# real helper is already imported. That cannot see the bootstrap escape, which
# is about *which* helper a fresh interpreter loads, so these tests run the
# actual entry point in a subprocess whose cwd is the root under test.

ENTRY = """
from experiments.pq237_joint_aura_streamed import verify_source_manifest
import sys
verify_source_manifest(sys.argv[1], sys.argv[2])
import experiments.pq237_source_closure as helper
print('VERIFIER ACCEPTED ROOT:', helper.__file__)
"""

SHADOW = """def verify_source_closure(*args):
    print("UNLISTED HELPER RAN")
"""


def _sealed_experiment_root(tmp_path, *, listed=("pq237_joint_aura_streamed.py",
                                                 "pq237_source_closure.py")):
    """Copy the real entry point and its policy into a root, and seal them."""
    root = tmp_path / "root"
    (root / "experiments").mkdir(parents=True)
    files = {}
    for name in ("pq237_joint_aura_streamed.py", "pq237_source_closure.py"):
        shutil.copy2(EXPERIMENTS / name, root / "experiments" / name)
        if name in listed:
            digest = hashlib.sha256((EXPERIMENTS / name).read_bytes()).hexdigest()
            files[f"experiments/{name}"] = digest
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps({"files": files, "symlinks": {}}))
    return root, manifest


def _run_entry(root, manifest):
    return subprocess.run(
        [sys.executable, "-B", "-c", ENTRY, str(root), str(manifest)],
        cwd=root, text=True, capture_output=True)


def test_the_sealed_entry_point_verifies_its_own_root(tmp_path):
    """The positive twin: loading the policy by path still works in-root."""
    root, manifest = _sealed_experiment_root(tmp_path)
    result = _run_entry(root, manifest)
    assert result.returncode == 0, result.stderr
    assert "VERIFIER ACCEPTED ROOT:" in result.stdout


def test_an_unlisted_package_cannot_supply_the_closure_policy(tmp_path):
    """A shadowing package answered `verify_source_closure` with a no-op."""
    root, manifest = _sealed_experiment_root(tmp_path)
    shadow = root / "experiments" / "pq237_source_closure"
    shadow.mkdir()
    (shadow / "__init__.py").write_text(SHADOW)
    (root / "unlisted_module.py").write_text('VALUE = "unsealed"\n')
    # The escape is real before it is refused: normal resolution finds the
    # package, not the sealed module beside it.
    spec = PathFinder.find_spec("pq237_source_closure", [str(root / "experiments")])
    assert Path(spec.origin) == shadow / "__init__.py"
    result = _run_entry(root, manifest)
    assert result.returncode != 0
    assert "UNLISTED HELPER RAN" not in result.stdout
    assert "exact file closure" in result.stderr


def test_a_policy_under_the_root_is_refused_before_it_executes(tmp_path):
    """Ordering, not just origin: unlisted bytes must not run at all."""
    root, manifest = _sealed_experiment_root(
        tmp_path, listed=("pq237_joint_aura_streamed.py",))
    helper = root / "experiments" / "pq237_source_closure.py"
    helper.write_text(helper.read_text() + '\nprint("HELPER EXECUTED")\n')
    result = _run_entry(root, manifest)
    assert result.returncode != 0
    assert "HELPER EXECUTED" not in result.stdout
    assert "does not list it" in result.stderr


def test_a_changed_policy_under_the_root_is_refused_before_it_executes(tmp_path):
    """The hash is checked against the bytes that are then compiled and run."""
    root, manifest = _sealed_experiment_root(tmp_path)
    helper = root / "experiments" / "pq237_source_closure.py"
    helper.write_text(helper.read_text() + '\nprint("HELPER EXECUTED")\n')
    result = _run_entry(root, manifest)
    assert result.returncode != 0
    assert "HELPER EXECUTED" not in result.stdout
    assert "source changed after manifest" in result.stderr


# --- __pycache__ is a readable directory, not just a cache -------------------


@pytest.fixture
def external_cache(tmp_path, monkeypatch):
    """Redirect cache lookups outside the root, as the runbook's serve does."""
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "external-cache"))


def test_unlisted_source_inside_the_bytecode_cache_is_refused(
        sealed, external_cache):
    """`sys.pycache_prefix` moves cache *lookups*; the directory still imports."""
    root, manifest = sealed
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "injected.py").write_text("VALUE = 'unsealed'\n")
    package = PathFinder.find_spec("__pycache__", [str(root)])
    injected = PathFinder.find_spec(
        "__pycache__.injected", list(package.submodule_search_locations))
    assert Path(injected.origin) == cache / "injected.py"
    with pytest.raises(ValueError, match="exact file closure"):
        verify_source_manifest(root, manifest)


def test_unlisted_sourceless_bytecode_inside_the_cache_is_refused(
        tmp_path, sealed, external_cache):
    """A bare `name.pyc` is importable wherever it sits, cache directory or not."""
    root, manifest = sealed
    donor = tmp_path / "donor.py"
    donor.write_text("VALUE = 'unsealed'\n")
    cache = root / "__pycache__"
    cache.mkdir()
    py_compile.compile(str(donor), cfile=str(cache / "extra.pyc"), doraise=True)
    with pytest.raises(ValueError, match="exact file closure"):
        verify_source_manifest(root, manifest)


def test_unlisted_package_below_the_bytecode_cache_is_refused(
        sealed, external_cache):
    root, manifest = sealed
    nested = root / "__pycache__" / "sub"
    nested.mkdir(parents=True)
    (nested / "__init__.py").write_text("VALUE = 'unsealed'\n")
    with pytest.raises(ValueError, match="exact file closure"):
        verify_source_manifest(root, manifest)


def test_conventional_cache_files_are_still_not_sealed(sealed, external_cache):
    """The carve-out stays narrow: a tagged stem no import can name.

    Sealing these would pin the manifest to one interpreter on one box, which
    is why the directory is walked but its cache files are not enumerated.
    """
    root, manifest = sealed
    cache = root / "__pycache__"
    cache.mkdir()
    tag = f"cpython-{sys.version_info[0]}{sys.version_info[1]}"
    (cache / f"measurement.{tag}.pyc").write_bytes(b"")
    (cache / f"measurement.{tag}.opt-1.pyc").write_bytes(b"")
    assert PathFinder.find_spec(f"measurement.{tag}", [str(cache)]) is None
    verify_source_manifest(root, manifest)


# --- A symlink pins a target string; the bytes behind it must be sealed ------


def test_a_declared_package_symlink_to_outside_code_is_refused(tmp_path):
    """Sealing the link's target *string* left its code free to change."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "measurement.py").write_text("VALUE = 'sealed'\n")
    target = tmp_path / "external-package"
    target.mkdir()
    (target / "__init__.py").write_text("VALUE = 'original'\n")
    (root / "linked_package").symlink_to(target, target_is_directory=True)
    manifest = tmp_path / "manifest.json"
    _seal(root, manifest)
    # The escape is real: the link resolves as a package the seal never hashed.
    spec = PathFinder.find_spec("linked_package", [str(root)])
    assert Path(spec.origin) == root / "linked_package" / "__init__.py"
    with pytest.raises(ValueError, match="does not seal"):
        verify_source_manifest(root, manifest)


def test_a_declared_module_symlink_to_outside_code_is_refused(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "measurement.py").write_text("VALUE = 'sealed'\n")
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 'outside'\n")
    (root / "calibration.py").symlink_to(outside)
    manifest = tmp_path / "manifest.json"
    _seal(root, manifest)
    with pytest.raises(ValueError, match="does not seal"):
        verify_source_manifest(root, manifest)


def test_declaring_outside_code_as_an_excluded_symlink_does_not_admit_it(tmp_path):
    """`excluded_symlinks` exempts data from the seal, never code."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "measurement.py").write_text("VALUE = 'sealed'\n")
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 'outside'\n")
    (root / "calibration.py").symlink_to(outside)
    manifest = tmp_path / "manifest.json"
    receipt = _seal(root, manifest)
    receipt["excluded_symlinks"] = {
        "calibration.py": receipt["symlinks"].pop("calibration.py")}
    manifest.write_text(json.dumps(receipt, sort_keys=True))
    with pytest.raises(ValueError, match="does not seal"):
        verify_source_manifest(root, manifest)


def test_a_symlink_into_a_dot_directory_is_refused(tmp_path):
    """The discriminating case: in-root, but past the one structural exclusion.

    `.vendored/__init__.py` is unreachable under its own name and so is never
    enumerated -- but `package/__init__.py` reaches it, and nothing hashed it.
    """
    root = tmp_path / "source"
    (root / ".vendored").mkdir(parents=True)
    (root / "measurement.py").write_text("VALUE = 'sealed'\n")
    (root / ".vendored" / "__init__.py").write_text("VALUE = 'unsealed'\n")
    (root / "package").symlink_to(root / ".vendored", target_is_directory=True)
    manifest = tmp_path / "manifest.json"
    receipt = _seal(root, manifest)
    # The writer seals the tracked roster plus the closure, and an untracked
    # file under a dot directory is in neither.
    receipt["files"].pop(".vendored/__init__.py")
    manifest.write_text(json.dumps(receipt, sort_keys=True))
    spec = PathFinder.find_spec("package", [str(root)])
    assert Path(spec.origin) == root / "package" / "__init__.py"
    with pytest.raises(ValueError, match="does not seal"):
        verify_source_manifest(root, manifest)


def test_a_symlink_to_sealed_in_root_source_is_accepted(tmp_path):
    """Refusal is about unsealed bytes, not about symlinks."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "measurement.py").write_text("VALUE = 'sealed'\n")
    (root / "alias.py").symlink_to(root / "measurement.py")
    manifest = tmp_path / "manifest.json"
    _seal(root, manifest)
    verify_source_manifest(root, manifest)


def test_a_symlink_to_an_outside_data_directory_is_accepted(tmp_path):
    """The calibration links must keep working without a data-file closure."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "measurement.py").write_text("VALUE = 'sealed'\n")
    corpus = tmp_path / "calibration"
    corpus.mkdir()
    (corpus / "diverse-v1.jsonl").write_text('{"text": "outside"}\n')
    (root / "calibration").symlink_to(corpus, target_is_directory=True)
    # Written by hand rather than through `_seal`: the point of this test is
    # that the data behind the link is *not* sealed, so the manifest must not
    # quietly acquire it by walking through the link.
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "files": {"measurement.py": hashlib.sha256(
            (root / "measurement.py").read_bytes()).hexdigest()},
        "symlinks": {"calibration": str(corpus)}}, sort_keys=True))
    verify_source_manifest(root, manifest)
    # Changing data behind the link is not a source-closure question.
    (corpus / "diverse-v1.jsonl").write_text('{"text": "rewritten"}\n')
    verify_source_manifest(root, manifest)
