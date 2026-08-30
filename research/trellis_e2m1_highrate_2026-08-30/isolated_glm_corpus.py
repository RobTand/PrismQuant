"""Load the active GLM corpus reader beside a frozen PrismaQuant codec.

The locked hull scripts intentionally bind the canonical ``prismaquant``
package name to their historical Stage-6 snapshot.  The finalized GLM corpus
reader is newer.  This helper gives only that reader a private package name so
its relative imports resolve from the active checkout without changing the
codec modules used by a campaign.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types


_PACKAGE_NAME = "_prismaquant_active_glm_corpus"


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
