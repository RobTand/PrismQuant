from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/numeric_profiled_launcher.py"
)
_SPEC = importlib.util.spec_from_file_location("numeric_profiled_launcher", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
L = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = L
_SPEC.loader.exec_module(L)


_DRIVER_SUCCESS = """
from pathlib import Path
import sys
index = sys.argv.index('--out')
Path(sys.argv[index + 1]).write_bytes(b'driver-result\\n')
"""

_FAKE_PROFILER = """
from pathlib import Path
import subprocess
import sys
args = sys.argv[1:]
if args[:1] != ['--output'] or args[2:5] != ['--format', 'speedscope', '--']:
    raise SystemExit(91)
completed = subprocess.run(args[5:], check=False)
if completed.returncode == 0:
    Path(args[1]).write_bytes(b'speedscope-profile\\n')
raise SystemExit(completed.returncode)
"""


def _argv(profile: Path, output: Path, program: str = _DRIVER_SUCCESS) -> list[str]:
    return [
        "--profile",
        str(profile),
        "--",
        sys.executable,
        "-c",
        program,
        "--out",
        str(output),
    ]


def _fake_profiler_argv(program: str = _FAKE_PROFILER) -> list[str]:
    return [sys.executable, "-c", program]


def test_supplied_profiler_runs_driver_and_requires_both_publications(tmp_path: Path):
    profile = tmp_path / "profile.speedscope.json"
    output = tmp_path / "result.json"

    assert L.run_profiled(
        _argv(profile, output), profiler_argv=_fake_profiler_argv()
    ) == 0
    assert profile.read_bytes() == b"speedscope-profile\n"
    assert output.read_bytes() == b"driver-result\n"
    assert not profile.is_symlink()
    assert not output.is_symlink()


def test_constructed_profiler_argv_is_exact_and_shell_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    profile = tmp_path / "profile.json"
    output = tmp_path / "result.json"
    observed: list[list[str]] = []

    def fake_run(
        command: list[str], *, check: bool, shell: bool
    ) -> subprocess.CompletedProcess[bytes]:
        observed.append(command)
        assert check is False
        assert shell is False
        profile.write_bytes(b"profile\n")
        output.write_bytes(b"result\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(L.subprocess, "run", fake_run)
    profiler = ["/reviewed/profiler", "record", "--rate", "99"]
    driver = ["/usr/bin/python3", "-B", "/driver.py", "--out", str(output)]
    assert L.run_profiled(
        ["--profile", str(profile), "--", *driver],
        profiler_argv=profiler,
    ) == 0
    assert observed == [[
        *profiler,
        "--output",
        str(profile),
        "--format",
        "speedscope",
        "--",
        *driver,
    ]]


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--profile"],
        ["--profile", "/tmp/p", "/usr/bin/python3", "--out", "/tmp/o"],
        ["extra", "--profile", "/tmp/p", "--", "/bin/true", "--out", "/tmp/o"],
        ["--profile", "/tmp/p", "--", "--out", "/tmp/o"],
        ["--profile", "/tmp/p", "--", "/bin/true"],
        ["--profile", "/tmp/p", "--", "/bin/true", "--out"],
        ["--profile", "/tmp/p", "--", "/bin/true", "--out=/tmp/o"],
        [
            "--profile", "/tmp/p", "--", "/bin/true",
            "--out", "/tmp/o", "--out", "/tmp/o2",
        ],
        [
            "--profile", "/tmp/p", "--", "/bin/true",
            "--out", "/tmp/o", "--profile", "/tmp/p2",
        ],
        [
            "--profile", "/tmp/p", "--", "/bin/true",
            "--out", "/tmp/o", "--",
        ],
    ],
)
def test_exact_launcher_grammar_rejects_ambiguity(argv: list[str]):
    with pytest.raises(L.ProfiledLaunchError):
        L.parse_launch_argv(argv)


@pytest.mark.parametrize(
    ("profile", "output"),
    [
        ("relative-profile", "/tmp/output"),
        ("/tmp/profile", "relative-output"),
        ("/tmp/a/../profile", "/tmp/output"),
        ("/tmp/profile", "/tmp/a/../output"),
        ("/tmp/same", "/tmp/same"),
    ],
)
def test_output_paths_must_be_unambiguous_absolute_new_names(
    profile: str, output: str
):
    with pytest.raises(L.ProfiledLaunchError):
        L.parse_launch_argv(
            ["--profile", profile, "--", "/bin/true", "--out", output]
        )


@pytest.mark.parametrize("target", ["profile", "output"])
@pytest.mark.parametrize("kind", ["regular", "dangling_symlink"])
def test_preexisting_or_symlink_output_refuses_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    kind: str,
):
    profile = tmp_path / "profile.json"
    output = tmp_path / "output.json"
    path = profile if target == "profile" else output
    if kind == "regular":
        path.write_bytes(b"incumbent\n")
    else:
        path.symlink_to(tmp_path / "missing-target")

    monkeypatch.setattr(
        L.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid target must refuse before subprocess")
        ),
    )
    assert L.main(_argv(profile, output), profiler_argv=["/bin/true"]) == 2
    assert os.path.lexists(path)


def test_symlinked_parent_refuses_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    profile = linked_parent / "profile.json"
    output = tmp_path / "output.json"
    monkeypatch.setattr(
        L.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("symlinked parent must refuse before subprocess")
        ),
    )
    assert L.main(_argv(profile, output), profiler_argv=["/bin/true"]) == 2


@pytest.mark.parametrize(
    "profiler",
    [
        [],
        ["/bin/true", "--"],
        ["/bin/true", "--output", "/tmp/other"],
        ["/bin/true", "--output=/tmp/other"],
        ["/bin/true", "--format", "raw"],
        ["/bin/true", "--format=raw"],
        ["/bin/true", "-o", "/tmp/other"],
        ["/bin/true", "-o/tmp/other"],
        ["/bin/true", "-f", "raw"],
        ["/bin/true", "-fspeedscope"],
    ],
)
def test_supplied_profiler_cannot_override_owned_protocol(profiler: list[str]):
    with pytest.raises(L.ProfiledLaunchError):
        L._normalize_profiler_argv(profiler)


def test_nonzero_driver_status_is_preserved_and_never_reported_as_success(
    tmp_path: Path,
):
    profile = tmp_path / "profile.json"
    output = tmp_path / "output.json"
    failing_driver = "raise SystemExit(7)"
    argv = _argv(profile, output, failing_driver)
    with pytest.raises(L.ProfiledProcessError) as caught:
        L.run_profiled(argv, profiler_argv=_fake_profiler_argv())
    assert caught.value.returncode == 7
    assert L.main(argv, profiler_argv=_fake_profiler_argv()) == 7
    assert not profile.exists()
    assert not output.exists()


def test_masked_child_failure_cannot_pass_without_final_result_marker(
    tmp_path: Path,
):
    profile = tmp_path / "profile.json"
    output = tmp_path / "output.json"
    masking_profiler = """
from pathlib import Path
import subprocess
import sys
args = sys.argv[1:]
if args[:1] != ['--output'] or args[2:5] != ['--format', 'speedscope', '--']:
    raise SystemExit(91)
completed = subprocess.run(args[5:], check=False)
if completed.returncode == 0:
    raise SystemExit(92)
Path(args[1]).write_bytes(b'masked-child-failure-profile\\n')
raise SystemExit(0)
"""
    assert L.main(
        _argv(profile, output, "raise SystemExit(23)"),
        profiler_argv=_fake_profiler_argv(masking_profiler),
    ) == 2
    assert profile.read_bytes() == b"masked-child-failure-profile\n"
    assert not output.exists()


def test_zero_profiler_without_publications_is_refused(tmp_path: Path):
    profile = tmp_path / "profile.json"
    output = tmp_path / "output.json"
    assert L.main(_argv(profile, output), profiler_argv=["/bin/true"]) == 2


def test_unavailable_profiler_is_an_explicit_contract_failure(tmp_path: Path):
    profile = tmp_path / "profile.json"
    output = tmp_path / "output.json"
    assert L.main(
        _argv(profile, output),
        profiler_argv=[str(tmp_path / "missing-profiler")],
    ) == 2
    assert not profile.exists()
    assert not output.exists()


def test_parent_replacement_during_subprocess_refuses_publications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    parent = tmp_path / "campaign"
    parent.mkdir()
    parked = tmp_path / "parked-campaign"
    profile = parent / "profile.json"
    output = parent / "output.json"

    def swap_and_publish(
        command: list[str], *, check: bool, shell: bool
    ) -> subprocess.CompletedProcess[bytes]:
        parent.rename(parked)
        parent.mkdir()
        profile.write_bytes(b"replacement-profile\n")
        output.write_bytes(b"replacement-result\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(L.subprocess, "run", swap_and_publish)
    assert L.main(_argv(profile, output), profiler_argv=["/reviewed/profiler"]) == 2


@pytest.mark.parametrize("bad_target", ["profile", "output"])
def test_zero_status_empty_publication_is_refused(
    tmp_path: Path, bad_target: str
):
    profile = tmp_path / "profile.json"
    output = tmp_path / "output.json"
    program = f"""
from pathlib import Path
import sys
args = sys.argv[1:]
profile = Path(args[1])
driver = args[5:]
out = Path(driver[driver.index('--out') + 1])
profile.write_bytes(b'' if {bad_target!r} == 'profile' else b'profile\\n')
out.write_bytes(b'' if {bad_target!r} == 'output' else b'output\\n')
"""
    assert L.main(_argv(profile, output), profiler_argv=_fake_profiler_argv(program)) == 2


@pytest.mark.parametrize("bad_target", ["profile", "output"])
def test_zero_status_symlink_publication_is_refused(
    tmp_path: Path, bad_target: str
):
    profile = tmp_path / "profile.json"
    output = tmp_path / "output.json"
    backing = tmp_path / "backing"
    backing.write_bytes(b"backing\n")
    program = f"""
from pathlib import Path
import sys
args = sys.argv[1:]
profile = Path(args[1])
driver = args[5:]
out = Path(driver[driver.index('--out') + 1])
profile.symlink_to(Path({str(backing)!r})) if {bad_target!r} == 'profile' else profile.write_bytes(b'profile\\n')
out.symlink_to(Path({str(backing)!r})) if {bad_target!r} == 'output' else out.write_bytes(b'output\\n')
"""
    assert L.main(_argv(profile, output), profiler_argv=_fake_profiler_argv(program)) == 2


def test_profile_and_result_cannot_be_two_names_for_one_inode(tmp_path: Path):
    profile = tmp_path / "profile.json"
    output = tmp_path / "output.json"
    program = """
from pathlib import Path
import os
import sys
args = sys.argv[1:]
profile = Path(args[1])
driver = args[5:]
out = Path(driver[driver.index('--out') + 1])
out.write_bytes(b'not-two-artifacts\\n')
os.link(out, profile)
"""
    assert L.main(_argv(profile, output), profiler_argv=_fake_profiler_argv(program)) == 2
