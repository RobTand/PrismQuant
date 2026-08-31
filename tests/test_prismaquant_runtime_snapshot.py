"""Content-addressed PrismaQuant runtime snapshot boundary."""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "prismaquant_runtime_snapshot.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("runtime_snapshot_under_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args: object):
    return subprocess.run(
        [sys.executable, str(TOOL), *(str(value) for value in args)],
        check=False,
        capture_output=True,
        text=True,
    )


def _repository(root: Path) -> tuple[Path, str]:
    source = root / "source"
    (source / "prismaquant").mkdir(parents=True)
    (source / "tools").mkdir()
    (source / "prismaquant" / "__init__.py").write_text("VALUE = 1\n")
    (source / "tools" / "container_runtime_identity.py").write_text("# tool\n")
    (source / "tools" / "prismaquant_runtime_snapshot.py").write_text("# self\n")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git", "-C", str(source), "-c", "user.name=test",
            "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return source, commit


def test_materialize_is_content_addressed_and_ignores_live_worktree(tmp_path):
    source, commit = _repository(tmp_path)
    cache = tmp_path / "cache"
    first = _run(
        "materialize", "--source-root", source,
        "--cache-root", cache, "--commit", commit,
    )
    assert first.returncode == 0, first.stderr
    payload = json.loads(first.stdout)
    snapshot = Path(payload["snapshot"])
    assert snapshot.is_dir()
    assert payload["commit"] == commit
    assert (snapshot / "prismaquant" / "__init__.py").read_text() == "VALUE = 1\n"

    (source / "prismaquant" / "__init__.py").write_text("VALUE = 2\n")
    repeated = _run(
        "materialize", "--source-root", source,
        "--cache-root", cache, "--commit", commit,
    )
    assert repeated.returncode == 0, repeated.stderr
    assert json.loads(repeated.stdout) == payload
    assert (snapshot / "prismaquant" / "__init__.py").read_text() == "VALUE = 1\n"


def test_verify_refuses_mutation_extra_files_and_wrong_transport_hash(tmp_path):
    source, commit = _repository(tmp_path)
    created = _run(
        "materialize", "--source-root", source,
        "--cache-root", tmp_path / "cache", "--commit", commit,
    )
    assert created.returncode == 0, created.stderr
    payload = json.loads(created.stdout)
    verify_args = (
        "verify", "--snapshot", payload["snapshot"],
        "--expected-commit", commit,
        "--expected-tree", payload["tree"],
        "--expected-closure-sha256", payload["closure_sha256"],
    )
    exact = _run(*verify_args)
    assert exact.returncode == 0, exact.stderr

    wrong = list(verify_args)
    wrong[-1] = "0" * 64
    refused_hash = _run(*wrong)
    assert refused_hash.returncode == 2
    assert "caller-attested" in refused_hash.stderr

    snapshot = Path(payload["snapshot"])
    (snapshot / "untracked.txt").write_text("unexpected\n")
    refused_extra = _run(*verify_args)
    assert refused_extra.returncode == 2
    assert "files differ" in refused_extra.stderr


def test_materialize_refuses_abbreviated_commit(tmp_path):
    source, commit = _repository(tmp_path)
    result = _run(
        "materialize", "--source-root", source,
        "--cache-root", tmp_path / "cache", "--commit", commit[:12],
    )
    assert result.returncode == 2
    assert "full lowercase 40-hex" in result.stderr


def test_materialize_preserves_absolute_git_symlink_without_following_it(tmp_path):
    source, _ = _repository(tmp_path)
    outside = tmp_path / "outside-calibration.jsonl"
    outside.write_text("must not be copied into the snapshot\n")
    calibration = source / "calibration"
    calibration.mkdir()
    link = calibration / "diverse-v1.jsonl"
    os.symlink(str(outside.resolve()), link)
    subprocess.run(["git", "-C", str(source), "add", "calibration"], check=True)
    subprocess.run(
        [
            "git", "-C", str(source), "-c", "user.name=test",
            "-c", "user.email=test@example.invalid", "commit", "-qm",
            "absolute calibration symlink",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    created = _run(
        "materialize", "--source-root", source,
        "--cache-root", tmp_path / "cache", "--commit", commit,
    )

    assert created.returncode == 0, created.stderr
    payload = json.loads(created.stdout)
    snapshot_link = Path(payload["snapshot"]) / "calibration" / "diverse-v1.jsonl"
    assert snapshot_link.is_symlink()
    assert os.readlink(snapshot_link) == str(outside.resolve())
    manifest = json.loads(
        (Path(payload["snapshot"]) / ".prismaquant-runtime-snapshot.json").read_text()
    )
    entry = next(
        row for row in manifest["entries"]
        if row["path"] == "calibration/diverse-v1.jsonl"
    )
    assert entry == {
        "path": "calibration/diverse-v1.jsonl",
        "type": "symlink",
        "target": str(outside.resolve()),
    }
    verified = _run(
        "verify", "--snapshot", payload["snapshot"],
        "--expected-commit", commit,
        "--expected-tree", payload["tree"],
        "--expected-closure-sha256", payload["closure_sha256"],
    )
    assert verified.returncode == 0, verified.stderr


@pytest.mark.parametrize("member_name", ["../escape", "/absolute", "a//b", "a/./b"])
def test_git_archive_extractor_refuses_non_normalized_member_paths(
    tmp_path, member_name
):
    snapshot_tool = _load_tool_module()
    archive = tmp_path / "hostile.tar"
    with tarfile.open(archive, mode="w") as handle:
        member = tarfile.TarInfo(member_name)
        member.size = 1
        handle.addfile(member, io.BytesIO(b"x"))
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(snapshot_tool.SnapshotError, match="normalized relative"):
        snapshot_tool._extract_git_archive(archive, destination)

    assert list(destination.iterdir()) == []


def test_git_archive_extractor_refuses_descendant_under_symlink_before_extract(
    tmp_path,
):
    snapshot_tool = _load_tool_module()
    archive = tmp_path / "hostile.tar"
    outside = tmp_path / "outside"
    outside.mkdir()
    with tarfile.open(archive, mode="w") as handle:
        link = tarfile.TarInfo("calibration")
        link.type = tarfile.SYMTYPE
        link.linkname = str(outside.resolve())
        handle.addfile(link)
        child = tarfile.TarInfo("calibration/escaped.jsonl")
        child.size = 7
        handle.addfile(child, io.BytesIO(b"escaped"))
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(snapshot_tool.SnapshotError, match="descends through symlink"):
        snapshot_tool._extract_git_archive(archive, destination)

    assert list(destination.iterdir()) == []
    assert list(outside.iterdir()) == []


def test_git_archive_extractor_refuses_undeclared_directory_ancestor(tmp_path):
    snapshot_tool = _load_tool_module()
    archive = tmp_path / "hostile.tar"
    with tarfile.open(archive, mode="w") as handle:
        child = tarfile.TarInfo("missing/child.txt")
        child.size = 1
        handle.addfile(child, io.BytesIO(b"x"))
    destination = tmp_path / "destination"
    destination.mkdir()

    with pytest.raises(snapshot_tool.SnapshotError, match="undeclared directory"):
        snapshot_tool._extract_git_archive(archive, destination)

    assert list(destination.iterdir()) == []


def test_git_archive_extractor_refuses_hardlink_and_special_members(tmp_path):
    snapshot_tool = _load_tool_module()
    for kind, type_code in (("hardlink", tarfile.LNKTYPE), ("device", tarfile.CHRTYPE)):
        archive = tmp_path / f"{kind}.tar"
        with tarfile.open(archive, mode="w") as handle:
            member = tarfile.TarInfo(kind)
            member.type = type_code
            member.linkname = "target"
            handle.addfile(member)
        destination = tmp_path / f"{kind}-destination"
        destination.mkdir()

        with pytest.raises(snapshot_tool.SnapshotError, match="unsupported type"):
            snapshot_tool._extract_git_archive(archive, destination)

        assert list(destination.iterdir()) == []


def test_snapshot_directory_publication_never_replaces_an_incumbent(tmp_path):
    snapshot_tool = _load_tool_module()
    candidate = tmp_path / "candidate"
    incumbent = tmp_path / "destination"
    candidate.mkdir()
    incumbent.mkdir()
    (candidate / "identity").write_text("candidate\n")
    (incumbent / "identity").write_text("incumbent\n")

    won = snapshot_tool._rename_directory_noreplace(candidate, incumbent)

    assert won is False
    assert (candidate / "identity").read_text() == "candidate\n"
    assert (incumbent / "identity").read_text() == "incumbent\n"
