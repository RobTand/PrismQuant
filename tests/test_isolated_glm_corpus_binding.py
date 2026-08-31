from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/isolated_glm_corpus.py"
)
_SPEC = importlib.util.spec_from_file_location("isolated_glm_binding_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _encoded(value: str) -> bytes:
    return json.dumps(
        {"identity": value, "padding": value * ((1 << 20) + 128)},
        sort_keys=True,
    ).encode("utf-8")


def test_bound_json_hashes_and_parses_one_open_file_during_atomic_swap(
    tmp_path, monkeypatch,
):
    target = tmp_path / "manifest.json"
    replacement = tmp_path / "replacement.json"
    displaced = tmp_path / "displaced.json"
    original_bytes = _encoded("a")
    replacement_bytes = _encoded("b")
    target.write_bytes(original_bytes)
    replacement.write_bytes(replacement_bytes)

    real_read = _MODULE.os.read
    swapped = False

    def swap_after_first_read(descriptor, amount):
        nonlocal swapped
        chunk = real_read(descriptor, amount)
        if chunk and not swapped:
            swapped = True
            target.replace(displaced)
            replacement.replace(target)
        return chunk

    monkeypatch.setattr(_MODULE.os, "read", swap_after_first_read)
    document, binding = _MODULE.read_bound_json(target)

    assert swapped
    assert document["identity"] == "a"
    assert binding["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert target.read_bytes() == replacement_bytes


def test_bound_corpus_loader_rejects_manifest_swap_between_parse_and_load(
    tmp_path, monkeypatch,
):
    manifest = tmp_path / "manifest.json"
    replacement = tmp_path / "replacement.json"
    displaced = tmp_path / "displaced.json"
    first = {"schema": "A", "identity": "first"}
    second = {"schema": "B", "identity": "second"}
    manifest.write_text(json.dumps(first))
    replacement.write_text(json.dumps(second))

    def swapped_loader(_repo_root, _manifest):
        manifest.replace(displaced)
        replacement.replace(manifest)
        return SimpleNamespace(manifest=second)

    monkeypatch.setattr(_MODULE, "load_active_glm_corpus", swapped_loader)
    with pytest.raises(RuntimeError, match="changed between bound read"):
        _MODULE.load_active_glm_corpus_bound(tmp_path, manifest)
