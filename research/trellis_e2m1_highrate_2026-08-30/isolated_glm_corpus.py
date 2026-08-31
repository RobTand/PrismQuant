"""Load the active GLM corpus reader beside a frozen PrismaQuant codec.

The locked hull scripts intentionally bind the canonical ``prismaquant``
package name to their historical Stage-6 snapshot.  The finalized GLM corpus
reader is newer.  This helper gives only that reader a private package name so
its relative imports resolve from the active checkout without changing the
codec modules used by a campaign.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import types


_PACKAGE_NAME = "_prismaquant_active_glm_corpus"


def read_bound_json(path: Path) -> tuple[dict[str, object], dict[str, str]]:
    """Parse and hash the exact bytes read from one pinned file description."""

    candidate = Path(path).resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        chunks = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)

    def object_from_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value}")

    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{candidate}: expected one JSON object")
    return value, {
        "path": str(candidate),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def load_active_glm_corpus(repo_root: Path, manifest: Path):
    package_root = (repo_root / "prismaquant").resolve(strict=True)
    package = sys.modules.get(_PACKAGE_NAME)
    if package is None:
        package = types.ModuleType(_PACKAGE_NAME)
        package.__file__ = str(package_root / "__init__.py")
        package.__package__ = _PACKAGE_NAME
        package.__path__ = [str(package_root)]
        sys.modules[_PACKAGE_NAME] = package
    else:
        bound_roots = [Path(path).resolve() for path in package.__path__]
        if bound_roots != [package_root]:
            raise RuntimeError(
                "active GLM corpus alias is already bound to another checkout: "
                f"{bound_roots} != {[package_root]}"
            )

    module = importlib.import_module(f"{_PACKAGE_NAME}.trellis_bf16_corpus")
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(package_root):
        raise RuntimeError(
            f"active GLM corpus loader escaped checkout: {module_path}"
        )
    return module.load_finalized_bf16_corpus(manifest)


def load_active_glm_corpus_bound(repo_root: Path, manifest: Path):
    """Load a corpus only when its parsed manifest matches bound input bytes."""

    document, binding = read_bound_json(manifest)
    corpus = load_active_glm_corpus(repo_root, manifest)
    if corpus.manifest != document:
        raise RuntimeError(
            "GLM manifest changed between bound read and corpus load"
        )
    return corpus, binding
