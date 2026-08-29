"""Fail-closed output-path preflights shared by checkpoint exporters.

Exporters write large artifacts over many minutes or hours. Reusing a
non-empty destination can mix files from different source checkpoints, while
an output symlink (including a broken one) can redirect an owned write outside
the requested artifact. These helpers establish the production contract: the
source and output must resolve to different paths, directory outputs must be
new or empty, and file outputs must not already exist in any filesystem form.
"""
from __future__ import annotations

import os
import inspect
import shutil
import stat
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path


_DIRECTORY_PUBLICATION_CONTEXT: ContextVar[
    tuple[Path, Path] | None
] = ContextVar("prismaquant_directory_publication_context", default=None)


def directory_publication_target(staged_output: str | Path) -> Path:
    """Return the final publish path for an active staged directory.

    Transactional exporters receive an owned sibling directory instead of
    the operator-requested destination.  Value-bearing metadata written while
    the transaction is active must still name the final artifact, not the
    private ``.tmp-*`` staging root.  Outside the matching transaction this is
    an identity operation over the resolved path.
    """
    staged_resolved = _resolved_output(
        staged_output,
        where="directory_publication_target",
    )
    context = _DIRECTORY_PUBLICATION_CONTEXT.get()
    if context is None:
        return staged_resolved
    active_staged, final_target = context
    if staged_resolved == active_staged:
        return final_target
    return staged_resolved


def _resolved_source(source: str | Path, *, where: str) -> Path:
    path = Path(source)
    try:
        return path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"{where}: source path cannot be resolved safely: {path}"
        ) from exc


def _resolved_output(output: str | Path, *, where: str) -> Path:
    path = Path(output)
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"{where}: output path cannot be resolved safely: {path}"
        ) from exc


def _reject_source_output_alias(
    source: str | Path,
    output: str | Path,
    *,
    where: str,
) -> tuple[Path, Path]:
    source_resolved = _resolved_source(source, where=where)
    output_resolved = _resolved_output(output, where=where)
    if source_resolved == output_resolved:
        raise RuntimeError(
            f"{where}: source and output resolve to the same path "
            f"({source_resolved}); refusing an in-place export"
        )
    return source_resolved, output_resolved


def validate_fresh_export_directory(
    source: str | Path,
    output: str | Path,
    *,
    where: str,
) -> Path:
    """Validate one fresh directory output without creating it.

    An existing real directory is accepted only when it has no entries at
    all. ``os.scandir`` and ``lstat`` semantics make hidden files and broken
    symlinks visible without following them. Existing output symlinks are
    always rejected, even when they point at an empty directory.
    """
    source_resolved, output_resolved = _reject_source_output_alias(
        source,
        output,
        where=where,
    )
    if (
        source_resolved in output_resolved.parents
        or output_resolved in source_resolved.parents
    ):
        raise RuntimeError(
            f"{where}: source directory {source_resolved} and output "
            f"directory {output_resolved} overlap as ancestor/descendant; "
            "use a separate output tree"
        )
    out = Path(output)

    if os.path.lexists(out):
        try:
            mode = out.lstat().st_mode
        except OSError as exc:
            raise RuntimeError(
                f"{where}: cannot inspect existing output target {out}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise RuntimeError(
                f"{where}: output directory target {out} is a symlink; "
                "use a fresh real directory"
            )
        if not stat.S_ISDIR(mode):
            raise RuntimeError(
                f"{where}: output directory target {out} already exists and "
                "is not a directory"
            )
        try:
            with os.scandir(out) as entries:
                existing = sorted(entry.name for entry in entries)
        except OSError as exc:
            raise RuntimeError(
                f"{where}: cannot inspect output directory {out}"
            ) from exc
        if existing:
            raise RuntimeError(
                f"{where}: output directory {out} is not empty; existing "
                f"entries include {existing[:12]}. Use a fresh/empty output "
                "directory."
            )
        return out

    return out


def prepare_fresh_export_directory(
    source: str | Path,
    output: str | Path,
    *,
    where: str,
) -> Path:
    """Validate and create/accept one fresh directory output."""
    out = validate_fresh_export_directory(
        source,
        output,
        where=where,
    )
    if os.path.lexists(out):
        return out

    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        # A target appearing between lexists() and mkdir() is not trusted: it
        # may be a symlink or a concurrent export. Do not re-open the race by
        # accepting it after the fact.
        raise RuntimeError(
            f"{where}: output target {out} appeared during preflight; "
            "refusing a concurrent or redirected export"
        ) from exc
    return out


def prepare_fresh_export_file(
    source: str | Path,
    output: str | Path,
    *,
    where: str,
) -> Path:
    """Validate one new file output without opening or creating it."""
    _reject_source_output_alias(source, output, where=where)
    out = Path(output)
    if os.path.lexists(out):
        try:
            mode = out.lstat().st_mode
        except OSError as exc:
            raise RuntimeError(
                f"{where}: cannot inspect existing output target {out}"
            ) from exc
        kind = "symlink" if stat.S_ISLNK(mode) else "filesystem entry"
        raise RuntimeError(
            f"{where}: output file target {out} already exists as a {kind}; "
            "refusing to overwrite it"
        )
    return out


def _directory_identity(path: Path, *, where: str) -> tuple[int, int]:
    """Return one real directory's stable filesystem identity."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{where}: cannot inspect directory {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"{where}: {path} is not a real directory")
    return int(info.st_dev), int(info.st_ino)


def _assert_empty_directory_identity(
    path: Path,
    identity: tuple[int, int],
    *,
    where: str,
) -> None:
    """Fail if an empty directory reservation was replaced or populated."""
    observed = _directory_identity(path, where=where)
    if observed != identity:
        raise RuntimeError(
            f"{where}: directory {path} changed identity during export; "
            "refusing to publish over a replaced destination"
        )
    try:
        with os.scandir(path) as entries:
            existing = sorted(entry.name for entry in entries)
    except OSError as exc:
        raise RuntimeError(f"{where}: cannot inspect directory {path}") from exc
    if existing:
        raise RuntimeError(
            f"{where}: directory {path} was populated during export; entries "
            f"include {existing[:12]}. Refusing to replace concurrent output."
        )


def _remove_owned_empty_directory(
    path: Path,
    identity: tuple[int, int],
    *,
    where: str,
) -> None:
    """Remove only the unchanged empty directory this transaction created."""
    if not os.path.lexists(path):
        return
    _assert_empty_directory_identity(path, identity, where=where)
    try:
        path.rmdir()
    except OSError as exc:
        raise RuntimeError(
            f"{where}: could not remove owned destination reservation {path}"
        ) from exc


def _regular_file_identity(path: Path, *, where: str) -> tuple[int, int]:
    """Return one real regular file's stable filesystem identity."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{where}: cannot inspect file {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"{where}: {path} is not a real regular file")
    return int(info.st_dev), int(info.st_ino)


def _remove_owned_file_link(
    path: Path,
    identity: tuple[int, int],
    *,
    where: str,
) -> None:
    """Unlink only a path that still names the transaction's staged inode."""
    if not os.path.lexists(path):
        return
    observed = _regular_file_identity(path, where=where)
    if observed != identity:
        raise RuntimeError(
            f"{where}: published path {path} changed identity; refusing to "
            "unlink an untrusted replacement"
        )
    try:
        path.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"{where}: could not roll back published path {path}"
        ) from exc


def _remove_owned_temp_root(
    root: Path,
    identity: tuple[int, int],
    *,
    where: str,
) -> None:
    """Recursively remove only the unchanged temporary root we created."""
    if not os.path.lexists(root):
        return
    observed = _directory_identity(root, where=where)
    if observed != identity:
        raise RuntimeError(
            f"{where}: owned temporary root {root} changed identity; refusing "
            "recursive cleanup of an untrusted replacement"
        )
    try:
        shutil.rmtree(root)
    except OSError as exc:
        raise RuntimeError(
            f"{where}: could not clean owned temporary root {root}"
        ) from exc


def _owned_sibling_temp_root(output: Path, *, where: str) -> tuple[Path, tuple[int, int]]:
    """Create an inode-tracked sibling root on the destination filesystem."""
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"{where}: cannot create output parent {output.parent}"
        ) from exc
    prefix_name = output.name[:80] or "export"
    try:
        root = Path(tempfile.mkdtemp(
            prefix=f".{prefix_name}.tmp-",
            dir=output.parent,
        ))
    except OSError as exc:
        raise RuntimeError(
            f"{where}: cannot create an owned sibling temporary root beside "
            f"{output}"
        ) from exc
    return root, _directory_identity(root, where=where)


@contextmanager
def transactional_export_directory(
    source: str | Path,
    output: str | Path,
    *,
    where: str,
    require_absent: bool = False,
):
    """Build a directory in an owned sibling and publish it after success.

    The final destination is never populated incrementally. A previously
    absent destination is reserved with an exclusive ``mkdir`` only after the
    caller has completed successfully, then replaced by the staged directory
    on the same filesystem. An already-existing empty directory remains
    supported: its inode and emptiness must be unchanged at publication time.
    """
    out = validate_fresh_export_directory(source, output, where=where)
    if require_absent:
        if os.path.lexists(out):
            raise RuntimeError(
                f"{where}: output destination {out} must remain absent for "
                "this export"
            )
        # Deliberately keep this None after the check.  If any filesystem
        # entry appears before publication, the exclusive mkdir below must
        # fail instead of adopting a concurrent empty directory.
        existing_identity = None
    else:
        existing_identity = (
            _directory_identity(out, where=where)
            if os.path.lexists(out)
            else None
        )
    temp_root, temp_identity = _owned_sibling_temp_root(out, where=where)
    primary_error: BaseException | None = None
    try:
        publication_token = _DIRECTORY_PUBLICATION_CONTEXT.set((
            _resolved_output(temp_root, where=where),
            _resolved_output(out, where=where),
        ))
        try:
            yield temp_root
        finally:
            _DIRECTORY_PUBLICATION_CONTEXT.reset(publication_token)

        reservation_identity = existing_identity
        owns_reservation = False
        if reservation_identity is None:
            try:
                # Deliberately omit an explicit mode: the caller's umask must
                # define the published directory exactly as a normal mkdir
                # would. The staging root starts at mkdtemp's private 0700 and
                # is changed to this effective mode immediately before rename.
                out.mkdir(exist_ok=False)
            except FileExistsError as exc:
                raise RuntimeError(
                    f"{where}: output destination {out} appeared during export; "
                    "refusing to replace concurrent output"
                ) from exc
            except OSError as exc:
                raise RuntimeError(
                    f"{where}: cannot reserve final output directory {out}"
                ) from exc
            reservation_identity = _directory_identity(out, where=where)
            owns_reservation = True

        try:
            _assert_empty_directory_identity(
                out,
                reservation_identity,
                where=where,
            )
            if _directory_identity(temp_root, where=where) != temp_identity:
                raise RuntimeError(
                    f"{where}: owned temporary root {temp_root} changed "
                    "identity; refusing to publish an untrusted replacement"
                )
            final_mode = stat.S_IMODE(out.lstat().st_mode)
            temp_root.chmod(final_mode)
            os.replace(temp_root, out)
        except BaseException:
            if owns_reservation:
                _remove_owned_empty_directory(
                    out,
                    reservation_identity,
                    where=where,
                )
            raise
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if primary_error is not None:
            # PRESERVE the partial temp root on failure.
            #
            # Gates run at very different depths. `assert_artifact_complete`
            # runs AFTER every tensor is written, and can fail on METADATA
            # alone: measured 2026-08-08 on DSv4-Flash, a 21-tensor `ignore`
            # mis-declaration (`attn.indexer.wq_b`) discarded ~6 hours of
            # byte-identical tensor writes, because this `finally` deleted them
            # unconditionally and `--reuse-prior` therefore had nothing to
            # reuse. Deleting work that the retry will reproduce bit-for-bit is
            # pure waste.
            #
            # This cannot be mistaken for a finished artifact: the publish path
            # is untouched, the name is dot-prefixed `.tmp-<token>`, and it
            # carries no completeness stamp — `--reuse-prior` still re-verifies
            # what it adopts. Size is proportional to work actually done, so an
            # early preflight failure preserves ~nothing.
            print(
                f"{where}: PRESERVED partial export at {temp_root}\n"
                f"{where}:   resume:  --reuse-prior {temp_root} --reuse-verify\n"
                f"{where}:   discard: rm -rf {temp_root}",
                flush=True,
            )
        else:
            try:
                _remove_owned_temp_root(
                    temp_root,
                    temp_identity,
                    where=where,
                )
            except BaseException as cleanup_exc:
                if primary_error is None:
                    raise
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note(
                        f"transaction cleanup also failed: {cleanup_exc}"
                    )


@contextmanager
def transactional_export_file(
    source: str | Path,
    output: str | Path,
    *,
    where: str,
):
    """Build one file under an owned sibling root and hard-link-publish it.

    ``os.link`` is an atomic no-clobber publication on the same filesystem: the
    final path appears with the complete staged inode or the call fails with
    ``EEXIST``. There is no check/open window and no visible empty placeholder.
    """
    out = prepare_fresh_export_file(source, output, where=where)
    temp_root, temp_identity = _owned_sibling_temp_root(out, where=where)
    staged = temp_root / (out.name or "artifact")
    primary_error: BaseException | None = None
    try:
        yield staged
        if _directory_identity(temp_root, where=where) != temp_identity:
            raise RuntimeError(
                f"{where}: owned temporary root {temp_root} changed identity; "
                "refusing to publish an untrusted replacement"
            )
        try:
            staged_identity = _regular_file_identity(staged, where=where)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{where}: staged output file was not produced: {staged}"
            ) from exc
        published = False
        try:
            os.link(staged, out, follow_symlinks=False)
            published = True
        except (TypeError, NotImplementedError) as exc:
            raise RuntimeError(
                f"{where}: this platform cannot perform fail-closed no-follow "
                "hard-link publication"
            ) from exc
        except FileExistsError as exc:
            raise RuntimeError(
                f"{where}: output destination {out} appeared during export; "
                "refusing to overwrite concurrent output"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"{where}: atomic no-clobber publication failed for {out}"
            ) from exc
        try:
            if _regular_file_identity(out, where=where) != staged_identity:
                raise RuntimeError(
                    f"{where}: published file {out} does not name the staged "
                    "inode"
                )
            staged.unlink()
            temp_root.rmdir()
        except BaseException:
            if published:
                _remove_owned_file_link(
                    out,
                    staged_identity,
                    where=where,
                )
            raise
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            _remove_owned_temp_root(
                temp_root,
                temp_identity,
                where=where,
            )
        except BaseException as cleanup_exc:
            if primary_error is None:
                raise
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(
                    f"transaction cleanup also failed: {cleanup_exc}"
                )


def transactional_directory_output(
    *,
    source_parameter: str,
    output_parameter: str,
    where: str,
    require_absent_parameter: str | None = None,
):
    """Decorate an exporter so its output parameter receives a staging root."""
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            source = bound.arguments[source_parameter]
            output = bound.arguments[output_parameter]
            require_absent = (
                require_absent_parameter is not None
                and bound.arguments.get(require_absent_parameter) is not None
            )
            with transactional_export_directory(
                source,
                output,
                where=where,
                require_absent=require_absent,
            ) as staged:
                bound.arguments[output_parameter] = staged
                return function(*bound.args, **bound.kwargs)

        return wrapped
    return decorate


def transactional_file_output(
    *,
    source_parameter: str,
    output_parameter: str,
    where: str,
):
    """Decorate an exporter so its output parameter receives a staged file."""
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            source = bound.arguments[source_parameter]
            output = bound.arguments[output_parameter]
            with transactional_export_file(
                source,
                output,
                where=where,
            ) as staged:
                bound.arguments[output_parameter] = staged
                return function(*bound.args, **bound.kwargs)

        return wrapped
    return decorate
