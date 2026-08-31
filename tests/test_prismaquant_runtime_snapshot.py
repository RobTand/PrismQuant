"""Content-addressed PrismaQuant runtime snapshot boundary."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
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
    executable = source / "tools" / "tracked_executable.py"
    executable.write_text("#!/usr/bin/env python3\n")
    executable.chmod(0o755)
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


def test_materialized_snapshot_has_exact_modes_and_import_cannot_write(tmp_path):
    source, commit = _repository(tmp_path)
    created = _run(
        "materialize", "--source-root", source,
        "--cache-root", tmp_path / "cache", "--commit", commit,
    )
    assert created.returncode == 0, created.stderr
    payload = json.loads(created.stdout)
    snapshot = Path(payload["snapshot"])

    assert snapshot.stat().st_uid == os.geteuid()
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o555
    assert stat.S_IMODE((snapshot / "prismaquant").stat().st_mode) == 0o555
    assert stat.S_IMODE(
        (snapshot / "prismaquant" / "__init__.py").stat().st_mode
    ) == 0o444
    assert stat.S_IMODE(
        (snapshot / "tools" / "tracked_executable.py").stat().st_mode
    ) == 0o555
    assert stat.S_IMODE(
        (snapshot / ".prismaquant-runtime-snapshot.json").stat().st_mode
    ) == 0o444
    if os.geteuid() == 0:
        pytest.skip("uid 0 bypasses POSIX write permission checks")

    environment = os.environ.copy()
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    environment.pop("PYTHONPYCACHEPREFIX", None)
    used = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pathlib, sys; "
                "assert not sys.dont_write_bytecode; "
                "root = pathlib.Path(sys.argv[1]); "
                "sys.path.insert(0, str(root)); "
                "import prismaquant; "
                "assert prismaquant.VALUE == 1; "
                "probe = root / 'write-probe'; "
                "\ntry:\n probe.write_text('forbidden\\n')"
                "\nexcept OSError:\n pass"
                "\nelse:\n raise SystemExit('snapshot write unexpectedly succeeded')"
            ),
            str(snapshot),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert used.returncode == 0, used.stderr
    assert not (snapshot / "write-probe").exists()
    assert list(snapshot.rglob("__pycache__")) == []


@pytest.mark.parametrize(
    ("relative", "writable_mode"),
    [
        (".", 0o755),
        ("prismaquant", 0o755),
        ("prismaquant/__init__.py", 0o644),
        ("tools/tracked_executable.py", 0o755),
        (".prismaquant-runtime-snapshot.json", 0o644),
    ],
)
def test_verify_refuses_every_writable_snapshot_inode(
    tmp_path, relative, writable_mode
):
    source, commit = _repository(tmp_path)
    created = _run(
        "materialize", "--source-root", source,
        "--cache-root", tmp_path / "cache", "--commit", commit,
    )
    assert created.returncode == 0, created.stderr
    payload = json.loads(created.stdout)
    snapshot = Path(payload["snapshot"])
    target = snapshot if relative == "." else snapshot / relative
    target.chmod(writable_mode)

    refused = _run(
        "verify", "--snapshot", snapshot,
        "--expected-commit", commit,
        "--expected-tree", payload["tree"],
        "--expected-closure-sha256", payload["closure_sha256"],
    )

    assert refused.returncode == 2
    assert "mode must be" in refused.stderr


def test_verify_refuses_untracked_read_only_pycache(tmp_path):
    source, commit = _repository(tmp_path)
    created = _run(
        "materialize", "--source-root", source,
        "--cache-root", tmp_path / "cache", "--commit", commit,
    )
    assert created.returncode == 0, created.stderr
    payload = json.loads(created.stdout)
    snapshot = Path(payload["snapshot"])
    package = snapshot / "prismaquant"
    package.chmod(0o755)
    pycache = package / "__pycache__"
    pycache.mkdir()
    bytecode = pycache / "untracked.pyc"
    bytecode.write_bytes(b"untracked bytecode")
    bytecode.chmod(0o444)
    pycache.chmod(0o555)
    package.chmod(0o555)

    refused = _run(
        "verify", "--snapshot", snapshot,
        "--expected-commit", commit,
        "--expected-tree", payload["tree"],
        "--expected-closure-sha256", payload["closure_sha256"],
    )

    assert refused.returncode == 2
    assert "files differ" in refused.stderr


def test_verify_refuses_untracked_read_only_empty_directory(tmp_path):
    source, commit = _repository(tmp_path)
    created = _run(
        "materialize", "--source-root", source,
        "--cache-root", tmp_path / "cache", "--commit", commit,
    )
    assert created.returncode == 0, created.stderr
    payload = json.loads(created.stdout)
    snapshot = Path(payload["snapshot"])
    snapshot.chmod(0o755)
    extra = snapshot / "untracked-empty"
    extra.mkdir()
    extra.chmod(0o555)
    snapshot.chmod(0o555)

    refused = _run(
        "verify", "--snapshot", snapshot,
        "--expected-commit", commit,
        "--expected-tree", payload["tree"],
        "--expected-closure-sha256", payload["closure_sha256"],
    )

    assert refused.returncode == 2
    assert "directories differ" in refused.stderr


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
    snapshot.chmod(0o755)
    (snapshot / "untracked.txt").write_text("unexpected\n")
    snapshot.chmod(0o555)
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
    outside.chmod(0o640)
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
    assert stat.S_IMODE(outside.stat().st_mode) == 0o640
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


def _publication_candidate(root: Path, identity: str) -> Path:
    root.mkdir()
    nested = root / "nested"
    nested.mkdir(mode=0o750)
    nested.chmod(0o750)
    payload = nested / "payload.bin"
    payload.write_bytes(identity.encode())
    payload.chmod(0o540)
    os.symlink("/absolute/calibration.jsonl", root / "calibration.jsonl")
    (root / ".prismaquant-runtime-snapshot.json").write_text(
        json.dumps({"identity": identity}) + "\n"
    )
    payload.chmod(0o555)
    (root / ".prismaquant-runtime-snapshot.json").chmod(0o444)
    nested.chmod(0o555)
    root.chmod(0o555)
    return root


def test_snapshot_tree_fallback_never_replaces_an_incumbent(tmp_path):
    snapshot_tool = _load_tool_module()
    candidate = _publication_candidate(tmp_path / "candidate", "candidate")
    incumbent = tmp_path / "destination"
    incumbent.mkdir()
    (incumbent / "identity").write_text("incumbent\n")

    won = snapshot_tool._populate_snapshot_tree_noreplace(candidate, incumbent)

    assert won is False
    assert (candidate / "nested" / "payload.bin").read_bytes() == b"candidate"
    assert (incumbent / "identity").read_text() == "incumbent\n"


def test_snapshot_tree_fallback_preserves_modes_links_and_manifest_last(tmp_path):
    snapshot_tool = _load_tool_module()
    candidate = _publication_candidate(tmp_path / "candidate", "exact")
    destination = tmp_path / "destination"
    phases = []

    def observe(phase, relative):
        phases.append((phase, relative))
        if phase == "directory_published":
            assert relative is not None
            assert stat.S_IMODE((destination / relative).stat().st_mode) == 0o700
        elif phase == "directory_finalized":
            assert relative is not None
            assert stat.S_IMODE((destination / relative).stat().st_mode) == 0o555
        elif phase == "before_manifest":
            assert stat.S_IMODE(destination.stat().st_mode) == 0o700
            assert not (
                destination / ".prismaquant-runtime-snapshot.json"
            ).exists()
        elif phase == "manifest_published":
            assert stat.S_IMODE(destination.stat().st_mode) == 0o700
            assert stat.S_IMODE(
                (destination / ".prismaquant-runtime-snapshot.json").stat().st_mode
            ) == 0o444
        elif phase == "root_finalized":
            assert stat.S_IMODE(destination.stat().st_mode) == 0o555

    won = snapshot_tool._populate_snapshot_tree_noreplace(
        candidate,
        destination,
        fault_inject=observe,
    )

    assert won is True
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert stat.S_IMODE((destination / "nested").stat().st_mode) == 0o555
    assert stat.S_IMODE(
        (destination / "nested" / "payload.bin").stat().st_mode
    ) == 0o555
    assert stat.S_IMODE(
        (destination / ".prismaquant-runtime-snapshot.json").stat().st_mode
    ) == 0o444
    assert (
        os.readlink(destination / "calibration.jsonl")
        == "/absolute/calibration.jsonl"
    )
    assert phases[0] == ("destination_claimed", None)
    assert phases[-1] == ("root_finalized", None)
    assert phases.index(("before_manifest", None)) < phases.index(
        ("manifest_published", None)
    )
    assert (destination / ".prismaquant-runtime-snapshot.json").is_file()


@pytest.mark.parametrize(
    "fault_phase",
    [
        "destination_claimed",
        "directory_published",
        "leaf_published",
        "directory_finalized",
        "before_manifest",
        "manifest_published",
    ],
)
def test_snapshot_tree_fallback_fault_leaves_unadoptable_incomplete_destination(
    tmp_path, monkeypatch, fault_phase
):
    snapshot_tool = _load_tool_module()
    source, commit = _repository(tmp_path)
    tree = subprocess.run(
        ["git", "-C", str(source), "rev-parse", f"{commit}^{{tree}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cache = tmp_path / "cache"
    destination = cache / f"{commit}-{tree[:12]}"
    original_populate = snapshot_tool._populate_snapshot_tree_noreplace

    def injected(source_path, destination_path, *, fault_inject=None):
        fired = False

        def fault(phase, relative):
            nonlocal fired
            if phase == fault_phase and not fired:
                fired = True
                raise RuntimeError(f"injected {phase}")

        return original_populate(
            source_path, destination_path, fault_inject=fault
        )

    monkeypatch.setattr(
        snapshot_tool, "_try_rename_directory_noreplace", lambda *_: None
    )
    monkeypatch.setattr(
        snapshot_tool, "_populate_snapshot_tree_noreplace", injected
    )
    with pytest.raises(RuntimeError, match=fault_phase):
        snapshot_tool.materialize_snapshot(source, cache, commit=commit)

    assert destination.is_dir()
    manifest_exists = (
        destination / ".prismaquant-runtime-snapshot.json"
    ).exists()
    assert manifest_exists is (fault_phase == "manifest_published")
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert list(cache.glob(f".{commit[:12]}.tmp-*")) == []
    assert list(cache.glob(f".{commit[:12]}.archive-*")) == []
    monkeypatch.setattr(
        snapshot_tool, "_populate_snapshot_tree_noreplace", original_populate
    )
    expected_error = "directory mode" if manifest_exists else "manifest"
    with pytest.raises(snapshot_tool.SnapshotError, match=expected_error):
        snapshot_tool.materialize_snapshot(source, cache, commit=commit)


def test_snapshot_tree_fallback_crash_after_root_freeze_is_exact_but_not_returned(
    tmp_path, monkeypatch
):
    snapshot_tool = _load_tool_module()
    source, commit = _repository(tmp_path)
    tree = subprocess.run(
        ["git", "-C", str(source), "rev-parse", f"{commit}^{{tree}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cache = tmp_path / "cache"
    destination = cache / f"{commit}-{tree[:12]}"
    original_populate = snapshot_tool._populate_snapshot_tree_noreplace

    def injected(source_path, destination_path, *, fault_inject=None):
        def fault(phase, relative):
            if phase == "root_finalized":
                raise RuntimeError("injected root_finalized")

        return original_populate(
            source_path, destination_path, fault_inject=fault
        )

    monkeypatch.setattr(
        snapshot_tool, "_try_rename_directory_noreplace", lambda *_: None
    )
    monkeypatch.setattr(
        snapshot_tool, "_populate_snapshot_tree_noreplace", injected
    )
    with pytest.raises(RuntimeError, match="root_finalized"):
        snapshot_tool.materialize_snapshot(source, cache, commit=commit)

    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert list(cache.glob(f".{commit[:12]}.tmp-*")) == []
    assert list(cache.glob(f".{commit[:12]}.archive-*")) == []
    monkeypatch.setattr(
        snapshot_tool, "_populate_snapshot_tree_noreplace", original_populate
    )
    adopted = snapshot_tool.materialize_snapshot(source, cache, commit=commit)
    assert adopted["snapshot"] == str(destination.resolve())


def test_snapshot_tree_fallback_manifest_window_never_verifies(
    tmp_path, monkeypatch
):
    snapshot_tool = _load_tool_module()
    source, commit = _repository(tmp_path)
    cache = tmp_path / "cache"
    original_populate = snapshot_tool._populate_snapshot_tree_noreplace
    observed = []

    def instrumented(source_path, destination_path, *, fault_inject=None):
        manifest = json.loads(
            (source_path / ".prismaquant-runtime-snapshot.json").read_text()
        )

        def observe(phase, relative):
            if phase == "manifest_published":
                with pytest.raises(
                    snapshot_tool.SnapshotError, match="directory mode"
                ):
                    snapshot_tool.verify_snapshot(
                        destination_path,
                        expected_commit=manifest["commit"],
                        expected_tree=manifest["tree"],
                        expected_closure_sha256=manifest["closure_sha256"],
                    )
                observed.append("manifest_rejected")
            elif phase == "root_finalized":
                snapshot_tool.verify_snapshot(
                    destination_path,
                    expected_commit=manifest["commit"],
                    expected_tree=manifest["tree"],
                    expected_closure_sha256=manifest["closure_sha256"],
                )
                observed.append("root_accepted")

        return original_populate(
            source_path, destination_path, fault_inject=observe
        )

    monkeypatch.setattr(
        snapshot_tool, "_try_rename_directory_noreplace", lambda *_: None
    )
    monkeypatch.setattr(
        snapshot_tool, "_populate_snapshot_tree_noreplace", instrumented
    )

    created = snapshot_tool.materialize_snapshot(source, cache, commit=commit)

    assert created["commit"] == commit
    assert observed == ["manifest_rejected", "root_accepted"]


def test_atomic_rename_receives_only_a_fully_immutable_candidate(
    tmp_path, monkeypatch
):
    snapshot_tool = _load_tool_module()
    source, commit = _repository(tmp_path)
    observed = []

    def inspect_then_rename(candidate, destination):
        manifest = json.loads(
            (candidate / ".prismaquant-runtime-snapshot.json").read_text()
        )
        snapshot_tool.verify_snapshot(
            candidate,
            expected_commit=manifest["commit"],
            expected_tree=manifest["tree"],
            expected_closure_sha256=manifest["closure_sha256"],
        )
        assert stat.S_IMODE(candidate.stat().st_mode) == 0o555
        assert stat.S_IMODE(
            (candidate / ".prismaquant-runtime-snapshot.json").stat().st_mode
        ) == 0o444
        observed.append("frozen_before_rename")
        candidate.rename(destination)
        return True

    monkeypatch.setattr(
        snapshot_tool, "_try_rename_directory_noreplace", inspect_then_rename
    )

    created = snapshot_tool.materialize_snapshot(
        source, tmp_path / "cache", commit=commit
    )

    assert created["commit"] == commit
    assert observed == ["frozen_before_rename"]


def test_materialize_lost_atomic_race_cleans_frozen_candidate_only(
    tmp_path, monkeypatch
):
    snapshot_tool = _load_tool_module()
    source, commit = _repository(tmp_path)
    cache = tmp_path / "cache"
    original_populate = snapshot_tool._populate_snapshot_tree_noreplace

    def publish_other_winner_then_lose(candidate, destination):
        assert original_populate(candidate, destination) is True
        return False

    monkeypatch.setattr(
        snapshot_tool,
        "_try_rename_directory_noreplace",
        publish_other_winner_then_lose,
    )

    created = snapshot_tool.materialize_snapshot(source, cache, commit=commit)
    snapshot = Path(created["snapshot"])

    assert list(cache.glob(f".{commit[:12]}.tmp-*")) == []
    assert list(cache.glob(f".{commit[:12]}.archive-*")) == []
    leaf = snapshot / "prismaquant" / "__init__.py"
    assert stat.S_IMODE(leaf.stat().st_mode) == 0o444
    assert leaf.stat().st_nlink == 1
    snapshot_tool.verify_snapshot(
        snapshot,
        expected_commit=commit,
        expected_tree=created["tree"],
        expected_closure_sha256=created["closure_sha256"],
    )


def test_private_cleanup_never_follows_candidate_or_nested_symlinks(tmp_path):
    snapshot_tool = _load_tool_module()
    cache = tmp_path / "cache"
    cache.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o750)
    outside.chmod(0o750)
    outside_payload = outside / "keep.bin"
    outside_payload.write_bytes(b"keep")
    outside_payload.chmod(0o640)

    root_link = cache / ".candidate-root-link"
    os.symlink(outside, root_link)
    with pytest.raises(snapshot_tool.SnapshotError, match="not one real directory"):
        snapshot_tool._remove_private_snapshot_tree(
            root_link, trusted_parent=cache
        )
    assert root_link.is_symlink()

    candidate = cache / ".candidate-real"
    candidate.mkdir()
    os.symlink(outside, candidate / "nested-link")
    candidate.chmod(0o555)
    snapshot_tool._remove_private_snapshot_tree(candidate, trusted_parent=cache)

    assert not candidate.exists()
    assert outside_payload.read_bytes() == b"keep"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o750
    assert stat.S_IMODE(outside_payload.stat().st_mode) == 0o640


def test_snapshot_tree_fallback_concurrent_claim_has_one_winner(tmp_path):
    snapshot_tool = _load_tool_module()
    first = _publication_candidate(tmp_path / "first", "same")
    second = _publication_candidate(tmp_path / "second", "same")
    destination = tmp_path / "destination"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                snapshot_tool._populate_snapshot_tree_noreplace,
                candidate,
                destination,
            )
            for candidate in (first, second)
        ]
    outcomes = [future.result() for future in futures]

    assert sorted(outcomes) == [False, True]
    assert (destination / "nested" / "payload.bin").read_bytes() == b"same"
    assert (
        os.readlink(destination / "calibration.jsonl")
        == "/absolute/calibration.jsonl"
    )
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert stat.S_IMODE((destination / "nested").stat().st_mode) == 0o555
    assert stat.S_IMODE(
        (destination / "nested" / "payload.bin").stat().st_mode
    ) == 0o555
