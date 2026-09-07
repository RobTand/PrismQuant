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

The trust boundary is one sentence: *the bytes hashed at seal time are the only
bytes Python may execute from the root*. Every rule below is that sentence
applied to one way the interpreter can reach other bytes -- a directory that
shadows a module, a bare `.pyc`, an entry inside `__pycache__`, or a symlink
whose target the root walk never reads.

Not in scope. Data files under the root are pinned only if the manifest lists
them; listed entries are always verified. This module widens *what must be
listed*, it never weakens verification of what is listed. A symlink to data --
`calibration/xdom-fit-v1.jsonl` and its two siblings in the historical
receipts -- names nothing the import system can load, so it stays a declared
target string and needs no data closure.
"""
from __future__ import annotations

import importlib.machinery
import os
from pathlib import Path
import sys

# Stamped by the writer so a reader can tell which closure policy produced a
# manifest. The reader's closure check is deliberately *not* gated on it:
# an older, schema-less manifest is checked by exactly the same policy.
SCHEMA = "prismaquant.pq237.source_closure.v3"

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

# The bytecode cache directory. It is *not* excluded from the closure: it is an
# ordinary directory whose name is an ordinary module-name component, so
# `import __pycache__.injected` loads `__pycache__/injected.py` like any other
# source file. What is excluded is the conventional cache *file* inside it,
# whose contents are interpreter- and machine-specific. The two are separated
# by `is_conventional_cache_file`, so the directory stays walkable while the
# caches stay unsealed.
#
# `require_bytecode_cache_is_unreadable` remains a second, independent refusal
# for those caches: a timestamp-validated `.pyc` whose header repeats the
# sealed source's mtime and size executes *instead of* the sealed source.
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
    """Return why `name` is not a source directory, or None if it is one.

    The dot rule is the only exclusion. `__pycache__` used to be excluded here
    and that was an escape: `sys.pycache_prefix` redirects where the
    interpreter *looks up* a cache, it does not make the directory unreadable,
    so an unlisted `__pycache__/injected.py` imported and ran.
    """
    if name.startswith("."):
        return DOT_DIRECTORY_REASON
    return None


def is_conventional_cache_file(name):
    """True when `name` is a compiler-written cache file, not a loadable module.

    `FileFinder` resolves a module by testing `tail + suffix` against the
    directory listing, where `tail` is one dotted-name component and therefore
    can never contain a ".". A cache file carries the interpreter tag in its
    stem (`measurement.cpython-312.pyc`, `measurement.cpython-312.opt-1.pyc`),
    so no import can name it, and hashing it would pin a seal to one box. A
    bare `measurement.pyc` has no tag, is importable through
    `SourcelessFileLoader`, and is deliberately not covered here.

    Only bytecode suffixes are tested. An extension module's tag sits in a
    suffix `EXTENSION_SUFFIXES` spells out, and that suffix set is specific to
    the running interpreter, so a dotted stem there may still be importable
    under the interpreter that reads the seal.
    """
    return any(name.endswith(suffix) and "." in name[:-len(suffix)]
               for suffix in importlib.machinery.BYTECODE_SUFFIXES)


def is_importable_file_name(name):
    """True when a `sys.path` entry's FileFinder could load `name` as a module."""
    return name.endswith(IMPORTABLE_SUFFIXES) and not is_conventional_cache_file(name)


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

    Declaring a symlink pins its target *string*. That is not the same as
    pinning the bytes behind it, which is why `verify_source_closure` also
    resolves every importable symlink; see `importable_symlink_sources`.
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
            elif is_importable_file_name(name):
                yield path.relative_to(root).as_posix(), "file"


def importable_symlink_sources(root, relative):
    """Return `(must_be_sealed, escapes)` for the symlink at `root/relative`.

    A symlink is importable-capable in exactly two ways: its own name resolves
    as a module, or it resolves to a directory, which supplies a package or a
    namespace portion from which further modules load. Either way the bytes
    Python executes are the *target's*, and the root walk never reads them --
    it does not follow links. Declaring the link therefore pins a string while
    the code behind it stays free to change, which is the escape this closes.

    `must_be_sealed` holds root-relative paths whose sealed hash is what an
    import through the link would actually execute; the caller requires each to
    be a declared file, so the link is allowed only where it re-exposes bytes
    the seal already covers. `escapes` holds reasons the link exposes code no
    sealed entry can cover at all: a target outside the root, a nested symlink
    the walk below deliberately does not chase, or an unresolvable target.

    A link to data returns two empty lists and is left to the target-string
    check: `calibration/xdom-fit-v1.jsonl` names nothing loadable, and a link
    to a directory holding only data has nothing importable beneath it.
    """
    root = Path(root).resolve()
    link = Path(root) / relative
    targets, escapes = [], []
    # `is_dir` follows the link, which is the question being asked: can this
    # name supply a package?
    if link.is_dir():
        for parent, directories, names in os.walk(link.resolve(), followlinks=False):
            parent = Path(parent)
            kept = []
            for name in directories:
                if (parent / name).is_symlink():
                    escapes.append(f"{relative}/... reaches a further symlink "
                                   f"at {(parent / name).name}")
                elif excluded_directory_reason(name) is None:
                    kept.append(name)
            directories[:] = kept
            for name in names:
                child = parent / name
                if child.is_symlink():
                    if is_importable_file_name(name) or child.is_dir():
                        escapes.append(f"{relative}/... reaches a further "
                                       f"symlink at {name}")
                elif is_importable_file_name(name):
                    targets.append(child)
    elif is_importable_file_name(link.name):
        target = link.resolve()
        if target.is_file():
            targets.append(target)
        else:
            escapes.append(f"{relative} does not resolve to a regular file")
    must_be_sealed = []
    for target in targets:
        if target.is_relative_to(root):
            must_be_sealed.append(target.relative_to(root).as_posix())
        else:
            escapes.append(f"{relative} exposes {target}, outside the root")
    return must_be_sealed, escapes


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
    unlisted, wrong_kind, symlinks = [], [], []
    for relative, kind in iter_source_closure(root):
        if kind == "symlink":
            symlinks.append(relative)
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
    require_symlinked_code_is_sealed(root, symlinks, declared_files)


def require_symlinked_code_is_sealed(root, symlinks, declared_files):
    """Refuse a symlink that can execute code the manifest never hashed.

    Declaring a link records its target string. The target's *content* is
    outside the root walk, so without this a sealed `linked_package ->
    ../external-package` verifies while its `__init__.py` is rewritten between
    seal and use. Code reached through a link is admitted only when it is the
    same sealed byte the manifest already pins; anything else is refused rather
    than enumerated, because there is no honest hash to compare it against.
    """
    escapes = []
    for relative in sorted(symlinks):
        must_be_sealed, reasons = importable_symlink_sources(root, relative)
        escapes.extend(reasons)
        escapes.extend(f"{relative} exposes unsealed source {path}"
                       for path in must_be_sealed if path not in declared_files)
    if escapes:
        raise ValueError(
            "source symlink exposes code the manifest does not seal: "
            + ", ".join(sorted(escapes)[:8])
            + (f" (+{len(escapes) - 8} more)" if len(escapes) > 8 else ""))
