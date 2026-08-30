"""Shared pytest fixtures.

Deliberately minimal: this repo had no conftest before 2026-08-30, and the
only thing added here is one NON-autouse fixture, so collection semantics for
the other ~5600 tests are unchanged.
"""
from __future__ import annotations

import pytest

# Sibling import: pytest puts this conftest's own directory on sys.path
# (rootdir/prepend import mode), and tests/ is not a package.
from cb_synthetic_target import declare_synthetic_cb_target


@pytest.fixture
def synthetic_cb_target():
    """Opt-in: this module's CB bodies are fixtures, not served artifacts.

    See ``tests/cb_synthetic_target`` for why this is a declaration and not a
    gate override.  Not autouse -- a module asks for it by name.
    """
    with declare_synthetic_cb_target() as reason:
        yield reason
