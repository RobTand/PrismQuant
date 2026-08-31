#!/usr/bin/env python3
"""Run one numeric driver under py-spy and verify both publications.

The public grammar is deliberately closed::

    numeric_profiled_launcher.py --profile ABSOLUTE_PATH -- DRIVER... \
        --out ABSOLUTE_PATH ...

``profiler_argv`` is an injection seam for isolated tests and reviewed
launchers.  It replaces only ``/usr/local/bin/py-spy record``; this wrapper
still owns the unique output flag, speedscope format, delimiter, and driver
argv.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import stat
import subprocess
import sys


DEFAULT_PROFILER_ARGV = ("/usr/local/bin/py-spy", "record")
_RESERVED_PROFILER_ARGUMENTS = frozenset(
    {"--", "--format", "--output", "-f", "-o"}
)


class ProfiledLaunchError(RuntimeError):
    """The launch grammar or a required publication is invalid."""


class ProfiledProcessError(ProfiledLaunchError):
    """The profiler/driver subprocess returned a nonzero status."""

    def __init__(self, returncode: int):
        self.returncode = returncode
        super().__init__(
            f"profiler/driver subprocess exited with status {returncode}"
        )


@dataclass(frozen=True)
class LaunchSpec:
    profile: Path
    output: Path
    driver_argv: tuple[str, ...]


@dataclass(frozen=True)
class _TargetGuard:
    path: Path
    parent_chain: tuple[tuple[str, int, int], ...]


def _absolute_new_file_path(raw: str, *, where: str) -> Path:
    if type(raw) is not str or not raw or "\x00" in raw:
        raise ProfiledLaunchError(f"{where} must be a nonempty path string")
    path = Path(raw)
    if not path.is_absolute() or path == Path("/"):
        raise ProfiledLaunchError(f"{where} must be a non-root absolute path")
    if ".." in path.parts:
        raise ProfiledLaunchError(f"{where} must not contain parent traversal")
    return path


def parse_launch_argv(argv: Sequence[str]) -> LaunchSpec:
    """Parse only ``--profile PATH -- DRIVER...`` and one driver ``--out``."""

    if isinstance(argv, (str, bytes)) or any(type(token) is not str for token in argv):
        raise ProfiledLaunchError("launcher argv must be a sequence of strings")
    tokens = list(argv)
    if len(tokens) < 5 or tokens[0] != "--profile" or tokens[2] != "--":
        raise ProfiledLaunchError(
            "launcher grammar is exactly --profile PATH -- DRIVER..."
        )
    if tokens.count("--") != 1:
        raise ProfiledLaunchError("launcher argv must contain exactly one -- delimiter")
    if any(
        token == "--profile" or token.startswith("--profile=")
        for token in tokens[1:]
    ):
        raise ProfiledLaunchError("launcher argv contains a duplicate --profile flag")

    profile = _absolute_new_file_path(tokens[1], where="profile path")
    driver = tokens[3:]
    if not driver or not driver[0] or "\x00" in driver[0] or driver[0].startswith("-"):
        raise ProfiledLaunchError("DRIVER must begin with an executable argv token")
    if any("\x00" in token for token in driver):
        raise ProfiledLaunchError("driver argv must not contain NUL bytes")
    if any(token.startswith("--out=") for token in driver):
        raise ProfiledLaunchError("driver output must use the exact --out PATH form")
    out_positions = [index for index, token in enumerate(driver) if token == "--out"]
    if len(out_positions) != 1:
        raise ProfiledLaunchError("driver argv must contain exactly one --out PATH")
    out_index = out_positions[0]
    if out_index + 1 >= len(driver):
        raise ProfiledLaunchError("driver --out flag is missing its path")
    output = _absolute_new_file_path(
        driver[out_index + 1], where="driver output path"
    )
    if profile == output:
        raise ProfiledLaunchError("profile and driver output paths must differ")
    return LaunchSpec(
        profile=profile,
        output=output,
        driver_argv=tuple(driver),
    )


def _directory_chain_identity(path: Path, *, where: str) -> tuple[tuple[str, int, int], ...]:
    """Return the no-symlink identity of every component through ``path``."""

    if not path.is_absolute():  # pragma: no cover - callers normalize first
        raise AssertionError("directory-chain validation requires an absolute path")
    current = Path(path.anchor)
    chain: list[tuple[str, int, int]] = []
    for part in (None, *path.parts[1:]):
        if part is not None:
            current /= part
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise ProfiledLaunchError(
                f"{where} parent directory is unavailable: {current}"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ProfiledLaunchError(
                f"{where} parent is not a real directory: {current}"
            )
        chain.append((str(current), info.st_dev, info.st_ino))
    return tuple(chain)


def _prepare_target(path: Path, *, where: str) -> _TargetGuard:
    if os.path.lexists(path):
        raise ProfiledLaunchError(f"{where} already exists: {path}")
    return _TargetGuard(
        path=path,
        parent_chain=_directory_chain_identity(path.parent, where=where),
    )


def _verify_parent_unchanged(guard: _TargetGuard, *, where: str) -> None:
    observed = _directory_chain_identity(guard.path.parent, where=where)
    if observed != guard.parent_chain:
        raise ProfiledLaunchError(
            f"{where} parent directory changed during profiler execution"
        )


def _verify_new_regular_nonempty(guard: _TargetGuard, *, where: str) -> os.stat_result:
    _verify_parent_unchanged(guard, where=where)
    try:
        info = os.lstat(guard.path)
    except OSError as exc:
        raise ProfiledLaunchError(f"{where} was not created: {guard.path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ProfiledLaunchError(
            f"{where} is not a regular non-symlink file: {guard.path}"
        )
    if info.st_size <= 0:
        raise ProfiledLaunchError(f"{where} is empty: {guard.path}")
    return info


def _normalize_profiler_argv(value: Sequence[str] | None) -> tuple[str, ...]:
    raw = DEFAULT_PROFILER_ARGV if value is None else value
    if isinstance(raw, (str, bytes)) or not raw:
        raise ProfiledLaunchError("profiler argv must be a nonempty string sequence")
    if any(type(token) is not str or not token or "\x00" in token for token in raw):
        raise ProfiledLaunchError(
            "profiler argv must contain only nonempty strings without NUL bytes"
        )
    for token in raw:
        if (
            token in _RESERVED_PROFILER_ARGUMENTS
            or token.startswith("--output=")
            or token.startswith("--format=")
            or (token.startswith("-o") and not token.startswith("--"))
            or (token.startswith("-f") and not token.startswith("--"))
        ):
            raise ProfiledLaunchError(
                "profiler argv must not override output, format, or -- delimiter"
            )
    return tuple(raw)


def run_profiled(
    argv: Sequence[str],
    *,
    profiler_argv: Sequence[str] | None = None,
) -> int:
    """Run the profiler and accept only two new, nonempty regular files."""

    spec = parse_launch_argv(argv)
    profiler = _normalize_profiler_argv(profiler_argv)
    profile_guard = _prepare_target(spec.profile, where="profile output")
    output_guard = _prepare_target(spec.output, where="driver output")
    # Narrow the preflight-to-exec window and catch a competing creator after
    # either parent walk. Publication-capable campaign directories still need
    # external single-principal/ACL ownership; a child opens these pathnames.
    if os.path.lexists(spec.profile) or os.path.lexists(spec.output):
        raise ProfiledLaunchError("an output path appeared during launch preflight")

    command = [
        *profiler,
        "--output",
        str(spec.profile),
        "--format",
        "speedscope",
        "--",
        *spec.driver_argv,
    ]
    try:
        completed = subprocess.run(command, check=False, shell=False)
    except OSError as exc:
        raise ProfiledLaunchError(f"cannot execute profiler argv: {profiler!r}") from exc
    if completed.returncode != 0:
        raise ProfiledProcessError(completed.returncode)

    profile_info = _verify_new_regular_nonempty(
        profile_guard, where="profile output"
    )
    output_info = _verify_new_regular_nonempty(
        output_guard, where="driver output"
    )
    if (profile_info.st_dev, profile_info.st_ino) == (
        output_info.st_dev,
        output_info.st_ino,
    ):
        raise ProfiledLaunchError(
            "profile and driver output must be distinct filesystem objects"
        )
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    profiler_argv: Sequence[str] | None = None,
) -> int:
    """CLI adapter: preserve child status; use status 2 for contract refusal."""

    try:
        return run_profiled(
            sys.argv[1:] if argv is None else argv,
            profiler_argv=profiler_argv,
        )
    except ProfiledProcessError as exc:
        print(f"numeric_profiled_launcher: {exc}", file=sys.stderr)
        return exc.returncode
    except ProfiledLaunchError as exc:
        print(f"numeric_profiled_launcher: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
