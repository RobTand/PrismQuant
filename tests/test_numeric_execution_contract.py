from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/numeric_execution_contract.py"
)
_SPEC = importlib.util.spec_from_file_location("numeric_execution_contract", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CONTRACT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONTRACT)


def _current_environment(*, host="sparky", device="NVIDIA GB10"):
    return {
        "torch": "2.13.0+cu130",
        "python": "3.12.3",
        "host": host,
        "device": device,
        "triton": "3.7.1",
        "container_image": _CONTRACT.CAMPAIGN_IMAGE_DIGEST,
    }


def _inspect(attestation_path: Path, *, physical_host="sparky"):
    container_id = "c" * 64
    repo_root = attestation_path.parent / "repo"
    git_common_dir = attestation_path.parent / "git-common"
    return {
        "Id": container_id,
        "Image": _CONTRACT.CAMPAIGN_IMAGE_DIGEST,
        "Config": {
            "Hostname": container_id[:12],
            "Image": _CONTRACT.CAMPAIGN_IMAGE_REFERENCE,
            "User": "1000:1000",
            "Env": [
                f"HULL_PHYSICAL_HOST={physical_host}",
                f"HULL_CONTAINER_IMAGE={_CONTRACT.CAMPAIGN_IMAGE_DIGEST}",
                f"HULL_LAUNCH_ATTESTATION={attestation_path}",
                f"HULL_REPO_ROOT={repo_root}",
                f"HULL_GIT_COMMON_DIR={git_common_dir}",
            ],
        },
        "HostConfig": {
            "UTSMode": "host",
            "NetworkMode": "none",
            "IpcMode": "private",
            "DeviceRequests": [{
                "Driver": "",
                "Count": -1,
                "DeviceIDs": None,
                "Capabilities": [["gpu"]],
            }],
        },
        "State": {"Status": "created"},
        "Mounts": [{
            "Destination": str(attestation_path.parent),
            "RW": False,
        }, {
            "Destination": str(repo_root),
            "RW": False,
        }, {
            "Destination": str(git_common_dir),
            "RW": False,
        }, {
            "Destination": "/output",
            "RW": True,
        }],
    }


def _write_attestation(path: Path, *, physical_host="sparky"):
    (path.parent / "repo").mkdir(exist_ok=True)
    (path.parent / "git-common").mkdir(exist_ok=True)
    record = _CONTRACT.build_launch_attestation(
        _inspect(path, physical_host=physical_host), physical_host
    )
    path.write_text(json.dumps(record), encoding="utf-8")
    return record


def _process_environment(path: Path, *, host="sparky"):
    return {
        "HULL_PHYSICAL_HOST": host,
        "HULL_CONTAINER_IMAGE": _CONTRACT.CAMPAIGN_IMAGE_DIGEST,
        "HULL_LAUNCH_ATTESTATION": str(path),
        "HULL_REPO_ROOT": str(path.parent / "repo"),
        "HULL_GIT_COMMON_DIR": str(path.parent / "git-common"),
        "HOSTNAME": "c" * 12,
    }


def _clean_git(monkeypatch):
    def fake_git(_root, *args):
        if args == ("rev-parse", "HEAD"):
            return "b" * 40
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        if args == (
            "rev-parse", "--path-format=absolute", "--git-common-dir"
        ):
            return str(_root.parent / "git-common")
        raise AssertionError(args)

    monkeypatch.setattr(_CONTRACT, "_git", fake_git)
    monkeypatch.setattr(
        _CONTRACT, "require_repo_tree_matches_head", lambda _root: None
    )


def _live_sparky(monkeypatch):
    monkeypatch.setattr(_CONTRACT.socket, "gethostname", lambda: "sparky")
    monkeypatch.setattr(
        _CONTRACT,
        "_live_gpu_identity",
        lambda: ("GPU-e76c7efc-c157-b1f4-1348-83e4eb5092f4", "NVIDIA GB10"),
    )


def test_execution_identity_joins_host_inspection_live_gpu_and_clean_source(
    tmp_path, monkeypatch,
):
    attestation_path = tmp_path / "launch" / "attestation.json"
    attestation_path.parent.mkdir()
    _write_attestation(attestation_path)
    _clean_git(monkeypatch)
    _live_sparky(monkeypatch)

    result = _CONTRACT.require_numeric_execution_environment(
        attestation_path.parent / "repo",
        _current_environment(),
        _process_environment(attestation_path),
        require_cuda=True,
    )

    assert result == {
        "schema": "trellis.numeric_execution.v2",
        "physical_host": "sparky",
        "uts_hostname": "sparky",
        "gpu_uuid": "GPU-e76c7efc-c157-b1f4-1348-83e4eb5092f4",
        "container_image_reference": _CONTRACT.CAMPAIGN_IMAGE_REFERENCE,
        "container_image_digest": _CONTRACT.CAMPAIGN_IMAGE_DIGEST,
        "container_image_id": _CONTRACT.CAMPAIGN_IMAGE_DIGEST,
        "container_image_evidence": "host_docker_daemon_inspect_before_start",
        "container_image_in_process_verification": "not_available",
        "container_user": "1000:1000",
        "ipc_mode": "private",
        "repo_root": str(attestation_path.parent / "repo"),
        "source_mount_evidence": (
            "host_docker_daemon_inspect_readonly_repo_and_git"
        ),
        "repo_git_commit": "b" * 40,
        "repo_tree_clean": True,
        "python": "3.12.3",
        "torch": "2.13.0+cu130",
        "triton": "3.7.1",
        "device": "NVIDIA GB10",
    }
    _CONTRACT.validate_numeric_execution_record(result)


def test_spoofed_uts_hostname_on_wrong_physical_gpu_fails_closed(
    tmp_path, monkeypatch,
):
    path = tmp_path / "attestation.json"
    _write_attestation(path)
    _clean_git(monkeypatch)
    monkeypatch.setattr(_CONTRACT.socket, "gethostname", lambda: "sparky")
    monkeypatch.setattr(
        _CONTRACT,
        "_live_gpu_identity",
        lambda: ("GPU-b1eceeea-fec7-371e-2cf3-cd10f2e7b705", "NVIDIA GB10"),
    )

    with pytest.raises(
        _CONTRACT.NumericExecutionContractError,
        match="GPU UUID/name does not match",
    ):
        _CONTRACT.require_numeric_execution_environment(
            path.parent / "repo",
            _current_environment(),
            _process_environment(path),
            require_cuda=True,
        )


def test_sparklina_logical_host_maps_to_exact_uts_and_gpu(tmp_path, monkeypatch):
    path = tmp_path / "attestation.json"
    _write_attestation(path, physical_host="sparklina")
    _clean_git(monkeypatch)
    monkeypatch.setattr(_CONTRACT.socket, "gethostname", lambda: "gx10-6b77")
    monkeypatch.setattr(
        _CONTRACT,
        "_live_gpu_identity",
        lambda: ("GPU-b1eceeea-fec7-371e-2cf3-cd10f2e7b705", "NVIDIA GB10"),
    )

    result = _CONTRACT.require_numeric_execution_environment(
        path.parent / "repo",
        _current_environment(host="gx10-6b77"),
        _process_environment(path, host="sparklina"),
        require_cuda=True,
    )
    assert result["physical_host"] == "sparklina"
    assert result["uts_hostname"] == "gx10-6b77"
    assert result["gpu_uuid"] == "GPU-b1eceeea-fec7-371e-2cf3-cd10f2e7b705"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value["Config"].__setitem__("Image", "stage6:latest"),
         "allowed campaign image reference"),
        (lambda value: value["Config"].__setitem__("User", "0:0"),
         "non-root --user 1000:1000"),
        (lambda value: value.__setitem__("Image", "sha256:" + "e" * 64),
         "allowed campaign image-ID set"),
        (lambda value: value["State"].__setitem__("Status", "running"),
         "not-yet-started"),
        (lambda value: value["HostConfig"].__setitem__("UTSMode", ""),
         "--uts=host"),
        (lambda value: value["HostConfig"].__setitem__("NetworkMode", "bridge"),
         "--network=none"),
        (lambda value: value["HostConfig"].__setitem__("IpcMode", "host"),
         "--ipc=private"),
        (lambda value: value["HostConfig"].__setitem__("DeviceRequests", []),
         "request exactly one GPU"),
        (lambda value: value["HostConfig"]["DeviceRequests"][0].__setitem__(
            "Driver", "cdi-unknown"
        ), "request exactly one GPU"),
        (lambda value: value["Config"]["Env"].append("HOSTNAME=sparky"),
         "must not override"),
        (lambda value: value["Mounts"][0].__setitem__("RW", True),
         "read-only Docker mount"),
        (lambda value: value["Mounts"][1].__setitem__("RW", True),
         "repository root.*read-only"),
        (lambda value: value["Mounts"].append({
            "Destination": value["Mounts"][1]["Destination"] + "/shadow",
            "RW": True,
        }), "writable mount overlaps"),
        (lambda value: value["Mounts"].append({
            "Destination": value["Mounts"][2]["Destination"] + "/objects",
            "RW": True,
        }), "writable mount overlaps"),
    ],
)
def test_host_side_inspector_rejects_inexact_docker_launch(
    tmp_path, mutate, match,
):
    inspected = _inspect(tmp_path / "attestation.json")
    mutate(inspected)
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match=match):
        _CONTRACT.build_launch_attestation(inspected, "sparky")


def test_approved_local_image_id_under_unapproved_reference_is_refused(tmp_path):
    inspected = _inspect(tmp_path / "attestation.json")
    assert inspected["Image"] == _CONTRACT.CAMPAIGN_IMAGE_DIGEST
    inspected["Config"]["Image"] = "stage6-encoder:58862b"
    with pytest.raises(
        _CONTRACT.NumericExecutionContractError,
        match="allowed campaign image reference",
    ):
        _CONTRACT.build_launch_attestation(inspected, "sparky")


def test_declared_digest_cannot_substitute_for_host_inspected_image(
    tmp_path, monkeypatch,
):
    path = tmp_path / "attestation.json"
    record = _write_attestation(path)
    record["image_id"] = "sha256:" + "e" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    _clean_git(monkeypatch)
    _live_sparky(monkeypatch)
    with pytest.raises(
        _CONTRACT.NumericExecutionContractError,
        match="identity digest mismatch",
    ):
        _CONTRACT.require_numeric_execution_environment(
            path.parent / "repo",
            _current_environment(),
            _process_environment(path),
            require_cuda=True,
        )


def test_live_container_id_must_match_inspected_container(tmp_path, monkeypatch):
    path = tmp_path / "attestation.json"
    _write_attestation(path)
    _clean_git(monkeypatch)
    _live_sparky(monkeypatch)
    process = _process_environment(path)
    process["HOSTNAME"] = "e" * 12
    with pytest.raises(
        _CONTRACT.NumericExecutionContractError,
        match="container-ID HOSTNAME",
    ):
        _CONTRACT.require_numeric_execution_environment(
            path.parent / "repo", _current_environment(), process, require_cuda=True
        )


def test_execution_record_is_closed_and_cannot_overclaim_image_verification(
    tmp_path, monkeypatch,
):
    path = tmp_path / "attestation.json"
    _write_attestation(path)
    _clean_git(monkeypatch)
    _live_sparky(monkeypatch)
    record = _CONTRACT.require_numeric_execution_environment(
        path.parent / "repo",
        _current_environment(),
        _process_environment(path),
        require_cuda=True,
    )
    for mutation in (
        {**record, "anonymous": True},
        {**record, "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000"},
        {**record, "container_image_in_process_verification": "cryptographic"},
        {**record, "container_user": "0:0"},
        {**record, "ipc_mode": "host"},
    ):
        with pytest.raises(_CONTRACT.NumericExecutionContractError):
            _CONTRACT.validate_numeric_execution_record(mutation)


def test_execution_identity_refuses_dirty_tree(tmp_path, monkeypatch):
    path = tmp_path / "attestation.json"
    _write_attestation(path)
    _live_sparky(monkeypatch)

    def dirty_git(_root, *args):
        if args == ("rev-parse", "HEAD"):
            return "b" * 40
        if args == (
            "rev-parse", "--path-format=absolute", "--git-common-dir"
        ):
            return str(_root.parent / "git-common")
        return " M research/driver.py"

    monkeypatch.setattr(_CONTRACT, "_git", dirty_git)
    with pytest.raises(
        _CONTRACT.NumericExecutionContractError,
        match="clean repository worktree",
    ):
        _CONTRACT.require_numeric_execution_environment(
            path.parent / "repo",
            _current_environment(),
            _process_environment(path),
            require_cuda=True,
        )


def test_gpu_optional_preflight_cannot_construct_publication_receipt(tmp_path):
    with pytest.raises(
        _CONTRACT.NumericExecutionContractError,
        match="preflight must not call",
    ):
        _CONTRACT.require_numeric_execution_environment(
            tmp_path, {}, {}, require_cuda=False
        )


def test_live_root_process_cannot_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(_CONTRACT.os, "getuid", lambda: 0)
    monkeypatch.setattr(_CONTRACT.os, "getgid", lambda: 0)
    with pytest.raises(
        _CONTRACT.NumericExecutionContractError,
        match="non-root UID/GID 1000:1000",
    ):
        _CONTRACT.require_numeric_execution_environment(
            tmp_path,
            _current_environment(),
            {},
            require_cuda=True,
        )


def test_host_attestation_writer_is_no_clobber(tmp_path, monkeypatch):
    path = tmp_path / "launch" / "attestation.json"
    inspected = _inspect(Path("/run/pq-launch/attestation.json"))
    monkeypatch.setattr(_CONTRACT, "_docker_inspect", lambda _container: inspected)
    fsync_calls = []
    monkeypatch.setattr(_CONTRACT.os, "fsync", fsync_calls.append)
    _CONTRACT.write_launch_attestation("c" * 64, "sparky", path)
    assert len(fsync_calls) == 2
    with pytest.raises(
        _CONTRACT.NumericExecutionContractError,
        match="without clobbering",
    ):
        _CONTRACT.write_launch_attestation("c" * 64, "sparky", path)


def test_repo_commit_never_degrades_to_none(tmp_path, monkeypatch):
    monkeypatch.setattr(_CONTRACT, "_git", lambda *_args: "unknown")
    with pytest.raises(
        _CONTRACT.NumericExecutionContractError,
        match="not a full lowercase commit",
    ):
        _CONTRACT.require_repo_commit(tmp_path)


def test_tracked_bytes_check_defeats_git_assume_unchanged(tmp_path):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "-q")
    git("config", "user.name", "Numeric Contract Test")
    git("config", "user.email", "numeric-contract@example.invalid")
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    git("add", "tracked.py")
    git("commit", "-qm", "fixture")
    git("update-index", "--assume-unchanged", "tracked.py")
    tracked.write_text("value = 2\n", encoding="utf-8")
    assert git("status", "--porcelain").stdout == ""

    with pytest.raises(
        _CONTRACT.NumericExecutionContractError,
        match="tracked source bytes differ from HEAD",
    ):
        _CONTRACT.require_repo_tree_matches_head(tmp_path)


def test_tracked_tree_check_rejects_ignored_importable_code(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "-q")
    git("config", "user.name", "Numeric Contract Test")
    git("config", "user.email", "numeric-contract@example.invalid")
    (tmp_path / ".gitignore").write_text("sitecustomize.py\n", encoding="utf-8")
    (tmp_path / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    git("add", ".gitignore", "tracked.py")
    git("commit", "-qm", "fixture")
    (tmp_path / "sitecustomize.py").write_text(
        "raise RuntimeError('shadowed runtime')\n", encoding="utf-8"
    )

    with pytest.raises(
        _CONTRACT.NumericExecutionContractError,
        match="ignored executable source/code",
    ):
        _CONTRACT.require_repo_tree_matches_head(tmp_path)
