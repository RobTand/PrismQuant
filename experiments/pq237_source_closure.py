#!/usr/bin/env python3
"""One source-closure policy for the PQ237 seal, shared by writer and reader.

The seal's claim is an *exact* file closure: between sealing and use, nothing
under the root changed, and nothing the Python import system can consume is
unaccounted for. Creation and validation therefore must agree on what "nothing
else" means. They agree by calling `iter_source_closure` here, rather than by
keeping two lists that happen to match today.

Scope. The closure covers what `importlib.machinery.PathFinder` can resolve
from the root when the root is a `sys.path` entry. That is the escape this
policy closes: an unlisted `measurement/__init__.py` next to a sealed
`measurement.py` makes the finder resolve the package instead of the sealed
module, without changing any listed file.

Not in scope. Data files under the root are pinned only if the manifest lists
them; listed entries are always verified. This module widens *what must be
listed*, it never weakens verification of what is listed.
"""
from __future__ import annotations

import importlib.machinery
import os
from pathlib import Path
import sys

# Stamped by the writer so a reader can tell which closure policy produced a
# manifest. The reader's closure check is deliberately *not* gated on it:
# an older, schema-less manifest is checked by exactly the same policy.
SCHEMA = "prismaquant.pq237.source_closure.v2"

# What a `sys.path` entry's FileFinder will load as a module. Taken from the
# interpreter rather than written out, so the set follows the interpreter that
# will actually do the importing. `EXTENSION_SUFFIXES` is interpreter-specific
# (it carries entries such as `.cpython-312-aarch64-linux-gnu.so`), but it
# always contains the plain `.so`, so a seal written under one interpreter
# still enumerates every extension module a different one could load.
# `.pyc` is included because a bare `name.pyc` in a path entry is importable
# on its own through `SourcelessFileLoader`; it is not merely a cache.
IMPORTABLE_SUFFIXES = tuple(sorted(set(
    importlib.machinery.SOURCE_SUFFIXES
    + importlib.machinery.BYTECODE_SUFFIXES
    + importlib.machinery.EXTENSION_SUFFIXES)))

# The bytecode cache directory. Excluded from the closure because its contents
# are interpreter- and machine-specific, so hashing them would make a seal
# unusable anywhere but the box that wrote it. Excluding it is only safe
# because `require_bytecode_cache_is_unreadable` refuses its presence unless
# the interpreter is provably reading its cache from outside the root: a
# timestamp-validated `.pyc` whose header repeats the sealed source's mtime and
# size executes *instead of* the sealed source, so a silent exclusion here
# would reopen the hole this module exists to close.
BYTECODE_CACHE_DIRECTORY = "__pycache__"

# Why a dot-prefixed directory is not part of the closure: a dotted module name
# is split on ".", so no component of an importable name can itself begin with
# a ".". `PathFinder.find_spec` therefore cannot descend into `.git`, `.venv`,
# `.pytest_cache` or `.github` from the root. This is a structural property of
# name resolution, not a list of directories we happen to recognise, so it does
# not grow stale. Entries under such a directory that the manifest *does* list
# are still verified; the exclusion governs unlisted additions only.
DOT_DIRECTORY_REASON = (
    "a module name component cannot begin with '.', so PathFinder cannot "
    "resolve anything inside a dot-prefixed directory")


def excluded_directory_reason(name):
    """Return why `name` is not a source directory, or None if it is one."""
    if name.startswith("."):
        return DOT_DIRECTORY_REASON
    if name == BYTECODE_CACHE_DIRECTORY:
        return ("cached bytecode, not source; presence is refused separately "
                "unless the interpreter reads its cache from outside the root")
    return None


def iter_source_closure(root):
    """Yield `(relative_posix_path, kind)` for every entry the closure covers.

    `kind` is "symlink" or "file". A symlink of any name is covered because a
    symlink can stand in for a module file *or* for a package directory, and
    because swapping a symlink for a regular file at the same path changes the
    content without changing the path. A regular file is covered when its name
    ends in an importable suffix.

    A plain directory is not itself a closure entry: `FileFinder` prefers
    `name.py` over a same-named directory that has no `__init__`, and
    `PathFinder` walks past such a namespace portion to later `sys.path`
    entries, so a bare directory cannot shadow a sealed module. A directory
    that *can* shadow one has an `__init__` with an importable suffix, and that
    file is a closure entry by the rule above.
    """
    root = Path(root)
    for parent, directories, names in os.walk(root, followlinks=False):
        parent = Path(parent)
        kept = []
        for name in directories:
            if (parent / name).is_symlink():
                # Not descended into (followlinks=False), so record it here:
                # a directory symlink can supply a whole package.
                yield (parent / name).relative_to(root).as_posix(), "symlink"
            elif excluded_directory_reason(name) is None:
                kept.append(name)
        directories[:] = kept
        for name in names:
            path = parent / name
            if path.is_symlink():
                yield path.relative_to(root).as_posix(), "symlink"
            elif name.endswith(IMPORTABLE_SUFFIXES):
                yield path.relative_to(root).as_posix(), "file"


def find_bytecode_cache_directories(root):
    """Return the relative paths of every `__pycache__` directory under root."""
    root = Path(root)
    found = []
    for parent, directories, _ in os.walk(root, followlinks=False):
        parent = Path(parent)
        kept = []
        for name in directories:
            if (parent / name).is_symlink() or name.startswith("."):
                continue
            if name == BYTECODE_CACHE_DIRECTORY:
                found.append((parent / name).relative_to(root).as_posix())
            else:
                kept.append(name)
        directories[:] = kept
    return sorted(found)


def require_bytecode_cache_is_unreadable(root):
    """Refuse in-tree bytecode caches the interpreter would still read.

    `sys.pycache_prefix` redirects `importlib.util.cache_from_source`, so with
    it set outside the root the `__pycache__` directories under the root are
    provably not consulted and need not be sealed. Without it, a forged
    timestamp-validated `.pyc` runs in place of the sealed source.
    """
    root = Path(root).resolve()
    prefix = sys.pycache_prefix
    if prefix and not Path(prefix).resolve().is_relative_to(root):
        return
    found = find_bytecode_cache_directories(root)
    if found:
        raise ValueError(
            "source closure cannot seal in-tree bytecode caches that this "
            "interpreter would read: " + ", ".join(found[:8])
            + (f" (+{len(found) - 8} more)" if len(found) > 8 else "")
            + "; set PYTHONPYCACHEPREFIX outside the root, or remove them")


def verify_source_closure(root, declared_files, declared_symlinks):
    """Refuse any closure entry the manifest does not declare, and kind drift.

    `declared_files` and `declared_symlinks` are the sets of relative paths the
    manifest claims as regular files and as symlinks. Both the writer and the
    reader derive the closure from `iter_source_closure`, so the two sides
    cannot drift apart.
    """
    require_bytecode_cache_is_unreadable(root)
    declared_files = set(declared_files)
    declared_symlinks = set(declared_symlinks)
    unlisted, wrong_kind = [], []
    for relative, kind in iter_source_closure(root):
        declared = declared_symlinks if kind == "symlink" else declared_files
        other = declared_files if kind == "symlink" else declared_symlinks
        if relative in declared:
            continue
        (wrong_kind if relative in other else unlisted).append((relative, kind))
    if unlisted:
        raise ValueError(
            "source manifest is not an exact file closure; unlisted importable "
            "entries under the root: "
            + ", ".join(f"{path} ({kind})" for path, kind in sorted(unlisted)[:8])
            + (f" (+{len(unlisted) - 8} more)" if len(unlisted) > 8 else ""))
    if wrong_kind:
        raise ValueError(
            "source entry kind differs from the manifest: "
            + ", ".join(f"{path} is now a {kind}" for path, kind in sorted(wrong_kind)))
