from __future__ import annotations

import copy
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
_PRODUCTION_ENVIRONMENT_DIGEST = _CONTRACT.CAMPAIGN_IMAGE_ENVIRONMENT_SHA256


@pytest.fixture(autouse=True)
def _isolated_campaign_storage(tmp_path, monkeypatch):
    storage = tmp_path / "campaign-storage"
    storage.mkdir()
    monkeypatch.setattr(_CONTRACT, "CAMPAIGN_STORAGE_ROOT", str(storage))
    fixture_image_environment = {
        "PATH": "/workspace/vllm:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
        "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    }
    monkeypatch.setattr(
        _CONTRACT,
        "CAMPAIGN_IMAGE_ENVIRONMENT_SHA256",
        _CONTRACT._identity_sha256(fixture_image_environment),
    )


def _launch_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    storage = Path(_CONTRACT.CAMPAIGN_STORAGE_ROOT)
    launch = storage / "launch"
    launch.mkdir(exist_ok=True)
    attestation = launch / "attestation.json"
    repo = tmp_path / "immutable-repo"
    driver_dir = repo / "research/trellis_e2m1_highrate_2026-08-30"
    driver_dir.mkdir(parents=True)
    driver = driver_dir / "fp8_cb_tcq_glm.py"
    driver.write_text("# fixture\n", encoding="utf-8")
    git_common = tmp_path / "git-common"
    git_common.mkdir()
    return attestation, repo, git_common, driver


def _current_environment(*, host="sparky", device="NVIDIA GB10"):
    return {
        "torch": "2.13.0+cu130",
        "python": "3.12.3",
        "host": host,
        "device": device,
        "triton": "3.7.1",
        "container_image": _CONTRACT.CAMPAIGN_IMAGE_DIGEST,
    }


def _inspect(
    attestation_path: Path,
    repo_root: Path,
    git_common_dir: Path,
    driver: Path,
    *,
    physical_host="sparky",
):
    container_id = "c" * 64
    storage = Path(_CONTRACT.CAMPAIGN_STORAGE_ROOT)
    command = [
        "/usr/local/bin/py-spy", "record",
        "--output", str(storage / "profiles" / "probe.speedscope.json"),
        "--format", "speedscope", "--", "/usr/bin/python3", "-B",
        str(driver), "--manifest", str(storage / "manifest.json"),
        "--out", str(storage / "result.json"),
    ]
    return {
        "Id": container_id,
        "Image": _CONTRACT.PHYSICAL_HOSTS[physical_host]["image_id"],
        "Config": {
            "Hostname": container_id[:12],
            "Image": _CONTRACT.CAMPAIGN_IMAGE_REFERENCE,
            "User": "1000:1000",
            "Entrypoint": list(_CONTRACT.CAMPAIGN_ENTRYPOINT),
            "WorkingDir": str(repo_root),
            "Cmd": command,
            "Healthcheck": None,
            "Shell": None,
            "Env": [
                f"HULL_PHYSICAL_HOST={physical_host}",
                f"HULL_CONTAINER_IMAGE={_CONTRACT.CAMPAIGN_IMAGE_DIGEST}",
                f"HULL_LAUNCH_ATTESTATION={attestation_path}",
                f"HULL_REPO_ROOT={repo_root}",
                f"HULL_GIT_COMMON_DIR={git_common_dir}",
                "PYTHONNOUSERSITE=1",
                "PYTHONDONTWRITEBYTECODE=1",
                "CUDA_CACHE_PATH=/tmp/cuda-cache",
                "TMPDIR=/tmp",
                "TORCH_EXTENSIONS_DIR=/tmp/torch-extensions",
                "TRITON_CACHE_DIR=/tmp/triton-cache",
                "XDG_CACHE_HOME=/tmp/cache",
                "PATH=/workspace/vllm:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
                "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas",
                "NVIDIA_VISIBLE_DEVICES=all",
                "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            ],
        },
        "HostConfig": {
            "UTSMode": "host",
            "NetworkMode": "none",
            "IpcMode": "private",
            "Privileged": False,
            "CapAdd": None,
            "CapDrop": ["ALL"],
            "GroupAdd": None,
            "Devices": [],
            "PidMode": "",
            "UsernsMode": "",
            "CgroupnsMode": "private",
            "SecurityOpt": ["no-new-privileges:true"],
            "ReadonlyRootfs": True,
            "Runtime": "runc",
            "Tmpfs": dict(_CONTRACT.CAMPAIGN_TMPFS),
            "Binds": None,
            "VolumesFrom": None,
            "Cgroup": "",
            "CgroupParent": "",
            "DeviceCgroupRules": None,
            "Isolation": "",
            "Init": None,
            "Sysctls": None,
            "DeviceRequests": [{
                "Driver": "",
                "Count": -1,
                "DeviceIDs": None,
                "Capabilities": [["gpu"]],
                "Options": {},
            }],
        },
        "State": {"Status": "created"},
        "Mounts": [{
            "Type": "bind", "Source": str(attestation_path.parent),
            "Destination": str(attestation_path.parent), "Mode": "",
            "RW": False, "Propagation": "rprivate",
        }, {
            "Type": "bind", "Source": str(repo_root),
            "Destination": str(repo_root), "Mode": "",
            "RW": False, "Propagation": "rprivate",
        }, {
            "Type": "bind", "Source": str(git_common_dir),
            "Destination": str(git_common_dir), "Mode": "",
            "RW": False, "Propagation": "rprivate",
        }, {
            "Type": "bind", "Source": str(storage),
            "Destination": str(storage), "Mode": "",
            "RW": True, "Propagation": "rprivate",
        }],
    }


def _fixture(tmp_path: Path, *, physical_host="sparky"):
    attestation, repo, git_common, driver = _launch_paths(tmp_path)
    inspected = _inspect(
        attestation, repo, git_common, driver, physical_host=physical_host
    )
    return attestation, repo, git_common, driver, inspected


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(_CONTRACT._canonical_json(value) + b"\n")


def _write_attestation(tmp_path: Path, *, physical_host="sparky"):
    path, repo, git_common, driver, inspected = _fixture(
        tmp_path, physical_host=physical_host
    )
    record = _CONTRACT.build_launch_attestation(
        inspected, physical_host, host_output_path=path, rootfs_changes=()
    )
    _write_canonical(path, record)
    return path, repo, git_common, driver, inspected, record


def _process_environment(path: Path, repo: Path, git_common: Path, *, host="sparky"):
    return {
        "HULL_PHYSICAL_HOST": host,
        "HULL_CONTAINER_IMAGE": _CONTRACT.CAMPAIGN_IMAGE_DIGEST,
        "HULL_LAUNCH_ATTESTATION": str(path),
        "HULL_REPO_ROOT": str(repo),
        "HULL_GIT_COMMON_DIR": str(git_common),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_CACHE_PATH": "/tmp/cuda-cache",
        "TMPDIR": "/tmp",
        "TORCH_EXTENSIONS_DIR": "/tmp/torch-extensions",
        "TRITON_CACHE_DIR": "/tmp/triton-cache",
        "XDG_CACHE_HOME": "/tmp/cache",
        "PATH": "/workspace/vllm:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
        "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
        "HOME": "/home/ubuntu",
        "LC_CTYPE": "C.UTF-8",
        "NVIDIA_CTK_LIBCUDA_DIR": "/usr/lib/aarch64-linux-gnu",
        "PWD": str(repo),
        "SHLVL": "0",
        "HOSTNAME": "c" * 12,
    }


def _clean_git(monkeypatch, git_common: Path):
    def fake_git(_root, *args):
        if args == ("rev-parse", "HEAD"):
            return "b" * 40
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        if args == ("rev-parse", "--path-format=absolute", "--git-common-dir"):
            return str(git_common)
        raise AssertionError(args)

    monkeypatch.setattr(_CONTRACT, "_git", fake_git)
    monkeypatch.setattr(
        _CONTRACT, "require_repo_tree_matches_head", lambda _root: None
    )


def _live_host(monkeypatch, *, host="sparky"):
    spec = _CONTRACT.PHYSICAL_HOSTS[host]
    monkeypatch.setattr(_CONTRACT.os, "getuid", lambda: 1000)
    monkeypatch.setattr(_CONTRACT.os, "getgid", lambda: 1000)
    monkeypatch.setattr(_CONTRACT.os, "getgroups", lambda: [1000])
    monkeypatch.setattr(_CONTRACT.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(_CONTRACT.socket, "gethostname", lambda: spec["uts_hostname"])
    monkeypatch.setattr(
        _CONTRACT,
        "_live_gpu_identity",
        lambda: (spec["gpu_uuid"], spec["gpu_name"]),
    )


def _require(tmp_path, monkeypatch, *, physical_host="sparky"):
    path, repo, git_common, driver, inspected, _record = _write_attestation(
        tmp_path, physical_host=physical_host
    )
    _clean_git(monkeypatch, git_common)
    _live_host(monkeypatch, host=physical_host)
    host = _CONTRACT.PHYSICAL_HOSTS[physical_host]["uts_hostname"]
    process = _process_environment(path, repo, git_common, host=physical_host)
    driver_argv = [str(driver), *inspected["Config"]["Cmd"][10:]]
    return _CONTRACT.require_numeric_execution_environment(
        repo,
        _current_environment(host=host),
        process,
        require_cuda=True,
        driver_path=driver,
        driver_argv=driver_argv,
    )


def test_execution_identity_joins_inspection_gpu_clean_source_and_segment(
    tmp_path, monkeypatch,
):
    record, segment = _require(tmp_path, monkeypatch)
    assert record["physical_host"] == "sparky"
    assert record["container_image_id"] == _CONTRACT.PHYSICAL_HOSTS["sparky"]["image_id"]
    assert record["repo_git_commit"] == "b" * 40
    assert segment["container_id"] == "c" * 64
    _CONTRACT.validate_numeric_execution_record(record)
    _CONTRACT.validate_numeric_execution_segment(segment, environment=record)


def test_pinned_image_environment_baseline_digest_is_exact():
    assert _PRODUCTION_ENVIRONMENT_DIGEST == (
        "347c08481fb3c7a5207a04f2e402add7a180ed5d28f9529ed1f306924a408e04"
    )


def test_sparklina_uses_its_exact_local_image_id(tmp_path, monkeypatch):
    record, segment = _require(tmp_path, monkeypatch, physical_host="sparklina")
    assert record["physical_host"] == "sparklina"
    assert record["uts_hostname"] == "gx10-6b77"
    assert record["container_image_id"] == (
        "sha256:ac631d27c1514ec3f838299d424c98892a0ba854fa642002df4c8f576bbfe9fa"
    )
    assert segment["image_id"] == record["container_image_id"]


def test_spoofed_uts_hostname_on_wrong_physical_gpu_fails_closed(
    tmp_path, monkeypatch,
):
    path, repo, git_common, driver, inspected, _ = _write_attestation(tmp_path)
    _clean_git(monkeypatch, git_common)
    _live_host(monkeypatch)
    monkeypatch.setattr(
        _CONTRACT,
        "_live_gpu_identity",
        lambda: (_CONTRACT.PHYSICAL_HOSTS["sparklina"]["gpu_uuid"], "NVIDIA GB10"),
    )
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="GPU UUID/name"):
        _CONTRACT.require_numeric_execution_environment(
            repo, _current_environment(),
            _process_environment(path, repo, git_common),
            require_cuda=True, driver_path=driver,
            driver_argv=[str(driver), *inspected["Config"]["Cmd"][10:]],
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda v: v["Config"].__setitem__("Image", "stage6:latest"), "image reference"),
        (lambda v: v["Config"].__setitem__("User", "0:0"), "non-root"),
        (lambda v: v.__setitem__("Image", "sha256:" + "e" * 64), "physical host's"),
        (lambda v: v["State"].__setitem__("Status", "running"), "not-yet-started"),
        (lambda v: v["HostConfig"].__setitem__("UTSMode", ""), "--uts=host"),
        (lambda v: v["HostConfig"].__setitem__("NetworkMode", "bridge"), "--network=none"),
        (lambda v: v["HostConfig"].__setitem__("IpcMode", "host"), "--ipc=private"),
        (lambda v: v["HostConfig"].__setitem__("DeviceRequests", []), "--gpus all"),
        (lambda v: v["HostConfig"].__setitem__("Privileged", True), "Privileged"),
        (lambda v: v["HostConfig"].__setitem__("CapAdd", ["SYS_ADMIN"]), "CapAdd"),
        (lambda v: v["HostConfig"].__setitem__("GroupAdd", ["0"]), "GroupAdd"),
        (lambda v: v["HostConfig"].__setitem__("Devices", [{"PathOnHost": "/dev/mem"}]), "Devices"),
        (lambda v: v["HostConfig"].__setitem__("PidMode", "host"), "PidMode"),
        (lambda v: v["HostConfig"].__setitem__("CgroupnsMode", "host"), "CgroupnsMode"),
        (lambda v: v["HostConfig"].__setitem__("SecurityOpt", []), "SecurityOpt"),
        (lambda v: v["HostConfig"].__setitem__("ReadonlyRootfs", False), "ReadonlyRootfs"),
        (lambda v: v["HostConfig"].__setitem__("Tmpfs", None), "Tmpfs"),
        (lambda v: v["HostConfig"].__setitem__("CgroupParent", "/attacker"), "CgroupParent"),
        (lambda v: v["Config"]["Env"].append("HOSTNAME=sparky"), "must not override"),
        (lambda v: v["Config"]["Env"].append("LD_PRELOAD=/evil.so"), "LD_PRELOAD"),
        (lambda v: v["Config"]["Env"].append("PYTHONPATH=/overlay"), "PYTHONPATH"),
        (lambda v: v["Config"]["Env"].append("CUBLAS_WORKSPACE_CONFIG=:16:8"), "environment schema"),
        (lambda v: v["Config"]["Env"].append("NVIDIA_TF32_OVERRIDE=1"), "environment schema"),
        (lambda v: v["Config"]["Env"].append("CUDA_LAUNCH_BLOCKING=1"), "environment schema"),
        (lambda v: v["Config"]["Env"].append("TORCHINDUCTOR_CACHE_DIR=/tmp/evil"), "environment schema"),
        (lambda v: v["Config"]["Env"].append("GENERIC_UNKNOWN_NUMERIC_KNOB=1"), "environment schema"),
        (lambda v: v["Config"].__setitem__("Entrypoint", ["/bin/sh"]), "entrypoint"),
        (lambda v: v["Config"].__setitem__("Healthcheck", {"Test": ["CMD", "/evil"]}), "healthcheck"),
        (lambda v: v["Config"]["Cmd"].__setitem__(0, "/usr/bin/true"), "py-spy"),
        (lambda v: v["Mounts"][0].__setitem__("RW", True), "allowlist"),
        (lambda v: v["Mounts"][0].__setitem__("Source", "/tmp/spoof"), "allowlist"),
        (lambda v: v["Mounts"].append({
            "Type": "bind", "Source": "/usr/bin", "Destination": "/usr/bin",
            "Mode": "", "RW": False, "Propagation": "rprivate",
        }), "allowlist"),
        (lambda v: v["Mounts"].append({
            "Type": "bind", "Source": "/tmp/site-packages",
            "Destination": "/usr/local/lib/python3.12/site-packages",
            "Mode": "", "RW": False, "Propagation": "rprivate",
        }), "allowlist"),
        (lambda v: v["Mounts"].append({
            "Type": "bind", "Source": "/tmp/ld.so.preload",
            "Destination": "/etc/ld.so.preload", "Mode": "", "RW": False,
            "Propagation": "rprivate",
        }), "allowlist"),
    ],
)
def test_host_inspector_rejects_inexact_or_injectable_launch(
    tmp_path, mutate, match,
):
    path, _repo, _git, _driver, inspected = _fixture(tmp_path)
    mutate(inspected)
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match=match):
        _CONTRACT.build_launch_attestation(
            inspected, "sparky", host_output_path=path, rootfs_changes=()
        )


def test_host_output_source_is_bound_to_requested_path(tmp_path):
    path, _repo, _git, _driver, inspected = _fixture(tmp_path)
    other = path.with_name("other.json")
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="must equal"):
        _CONTRACT.build_launch_attestation(
            inspected, "sparky", host_output_path=other, rootfs_changes=()
        )


def test_named_repo_and_git_mounts_cannot_hide_inside_writable_storage(tmp_path):
    path, repo, _git, driver, inspected = _fixture(tmp_path)
    storage = Path(_CONTRACT.CAMPAIGN_STORAGE_ROOT)
    nested_repo = storage / "hidden-repository"
    inspected["Config"]["Env"] = [
        f"HULL_REPO_ROOT={nested_repo}" if item.startswith("HULL_REPO_ROOT=") else item
        for item in inspected["Config"]["Env"]
    ]
    inspected["Config"]["WorkingDir"] = str(nested_repo)
    inspected["Config"]["Cmd"][9] = str(
        nested_repo / driver.relative_to(repo)
    )
    inspected["Mounts"][1].update({
        "Source": str(nested_repo), "Destination": str(nested_repo),
    })
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="must not overlap"):
        _CONTRACT.build_launch_attestation(
            inspected, "sparky", host_output_path=path, rootfs_changes=()
        )

    path, repo, _git, _driver, inspected = _fixture(tmp_path / "second")
    nested_git = repo / "nested-git-common"
    inspected["Config"]["Env"] = [
        f"HULL_GIT_COMMON_DIR={nested_git}"
        if item.startswith("HULL_GIT_COMMON_DIR=") else item
        for item in inspected["Config"]["Env"]
    ]
    inspected["Mounts"][2].update({
        "Source": str(nested_git), "Destination": str(nested_git),
    })
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="must not overlap"):
        _CONTRACT.build_launch_attestation(
            inspected, "sparky", host_output_path=path, rootfs_changes=()
        )


def test_direct_python_launch_is_limited_to_nonpublishing_preflight(tmp_path):
    path, _repo, _git, _driver, inspected = _fixture(tmp_path)
    inspected["Config"]["Cmd"] = [
        *inspected["Config"]["Cmd"][7:], "--preflight-only"
    ]
    _CONTRACT.build_launch_attestation(
        inspected, "sparky", host_output_path=path, rootfs_changes=()
    )
    inspected["Config"]["Cmd"].pop()
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="nonpublishing preflight"):
        _CONTRACT.build_launch_attestation(
            inspected, "sparky", host_output_path=path, rootfs_changes=()
        )


def test_cross_host_local_image_id_is_refused(tmp_path):
    path, _repo, _git, _driver, inspected = _fixture(tmp_path)
    inspected["Image"] = _CONTRACT.PHYSICAL_HOSTS["sparklina"]["image_id"]
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="physical host's"):
        _CONTRACT.build_launch_attestation(
            inspected, "sparky", host_output_path=path, rootfs_changes=()
        )


def test_noncanonical_raw_attestation_is_refused(tmp_path):
    path, _repo, _git, _driver, _inspected, record = _write_attestation(tmp_path)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="canonical JSON"):
        _CONTRACT._read_launch_attestation(str(path))


def test_launch_attestation_symlink_is_refused_at_open(tmp_path):
    path, _repo, _git, _driver, _inspected, _record = _write_attestation(tmp_path)
    link = path.with_name("attestation-link.json")
    link.symlink_to(path)
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="cannot read strict"):
        _CONTRACT._read_launch_attestation(str(link))


def test_resealed_attestation_image_id_spoof_is_refused(tmp_path, monkeypatch):
    path, repo, git_common, driver, inspected, record = _write_attestation(tmp_path)
    spoof = copy.deepcopy(record)
    spoof["image_id"] = "sha256:" + "e" * 64
    unsigned = {key: value for key, value in spoof.items() if key != "attestation_sha256"}
    spoof["attestation_sha256"] = _CONTRACT._identity_sha256(unsigned)
    _write_canonical(path, spoof)
    _clean_git(monkeypatch, git_common)
    _live_host(monkeypatch)
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="image ID"):
        _CONTRACT.require_numeric_execution_environment(
            repo, _current_environment(),
            _process_environment(path, repo, git_common),
            require_cuda=True, driver_path=driver,
            driver_argv=[str(driver), *inspected["Config"]["Cmd"][10:]],
        )


def test_live_container_id_and_driver_argv_must_match_inspection(tmp_path, monkeypatch):
    path, repo, git_common, driver, inspected, _ = _write_attestation(tmp_path)
    _clean_git(monkeypatch, git_common)
    _live_host(monkeypatch)
    process = _process_environment(path, repo, git_common)
    process["HOSTNAME"] = "e" * 12
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="container-ID HOSTNAME"):
        _CONTRACT.require_numeric_execution_environment(
            repo, _current_environment(), process, require_cuda=True,
            driver_path=driver,
            driver_argv=[str(driver), *inspected["Config"]["Cmd"][10:]],
        )
    process["HOSTNAME"] = "c" * 12
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="driver argv"):
        _CONTRACT.require_numeric_execution_environment(
            repo, _current_environment(), process, require_cuda=True,
            driver_path=driver, driver_argv=[str(driver), "--different"],
        )


@pytest.mark.parametrize("key", ["LD_PRELOAD", "PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"])
def test_live_external_import_environment_is_refused(tmp_path, monkeypatch, key):
    path, repo, git_common, driver, inspected, _ = _write_attestation(tmp_path)
    _clean_git(monkeypatch, git_common)
    _live_host(monkeypatch)
    process = _process_environment(path, repo, git_common)
    process[key] = "/attacker"
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match=key):
        _CONTRACT.require_numeric_execution_environment(
            repo, _current_environment(), process, require_cuda=True,
            driver_path=driver,
            driver_argv=[str(driver), *inspected["Config"]["Cmd"][10:]],
        )


@pytest.mark.parametrize("key", [
    "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
    "CUDA_LAUNCH_BLOCKING", "TORCHINDUCTOR_CACHE_DIR",
    "GENERIC_UNKNOWN_NUMERIC_KNOB",
])
def test_live_unreviewed_numeric_environment_is_refused(tmp_path, monkeypatch, key):
    path, repo, git_common, driver, inspected, _ = _write_attestation(tmp_path)
    _clean_git(monkeypatch, git_common)
    _live_host(monkeypatch)
    process = _process_environment(path, repo, git_common)
    process[key] = "1"
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="environment schema"):
        _CONTRACT.require_numeric_execution_environment(
            repo, _current_environment(), process, require_cuda=True,
            driver_path=driver,
            driver_argv=[str(driver), *inspected["Config"]["Cmd"][10:]],
        )


def test_execution_record_and_segment_are_closed(tmp_path, monkeypatch):
    record, segment = _require(tmp_path, monkeypatch)
    attacks = (
        {**record, "anonymous": True},
        {**record, "container_image_id": _CONTRACT.PHYSICAL_HOSTS["sparklina"]["image_id"]},
        {**record, "container_image_in_process_verification": "cryptographic"},
        {**record, "container_user": "0:0"},
        {**record, "ipc_mode": "host"},
    )
    for mutation in attacks:
        with pytest.raises(_CONTRACT.NumericExecutionContractError):
            _CONTRACT.validate_numeric_execution_record(mutation)
    bad = {**segment, "image_id": _CONTRACT.PHYSICAL_HOSTS["sparklina"]["image_id"]}
    unsigned = {key: value for key, value in bad.items() if key != "segment_sha256"}
    bad["segment_sha256"] = _CONTRACT._identity_sha256(unsigned)
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="stable environment"):
        _CONTRACT.validate_numeric_execution_segment(bad, environment=record)


def test_execution_identity_refuses_dirty_tree(tmp_path, monkeypatch):
    path, repo, git_common, driver, inspected, _ = _write_attestation(tmp_path)
    _live_host(monkeypatch)

    def dirty_git(_root, *args):
        if args == ("rev-parse", "HEAD"):
            return "b" * 40
        if args == ("rev-parse", "--path-format=absolute", "--git-common-dir"):
            return str(git_common)
        return " M research/driver.py"

    monkeypatch.setattr(_CONTRACT, "_git", dirty_git)
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="clean repository"):
        _CONTRACT.require_numeric_execution_environment(
            repo, _current_environment(),
            _process_environment(path, repo, git_common), require_cuda=True,
            driver_path=driver,
            driver_argv=[str(driver), *inspected["Config"]["Cmd"][10:]],
        )


def test_live_supplementary_group_cannot_publish(tmp_path, monkeypatch):
    path, repo, git_common, driver, inspected, _ = _write_attestation(tmp_path)
    _clean_git(monkeypatch, git_common)
    _live_host(monkeypatch)
    monkeypatch.setattr(_CONTRACT.os, "getgroups", lambda: [1000, 44])
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="UID/GID/groups"):
        _CONTRACT.require_numeric_execution_environment(
            repo, _current_environment(),
            _process_environment(path, repo, git_common), require_cuda=True,
            driver_path=driver,
            driver_argv=[str(driver), *inspected["Config"]["Cmd"][10:]],
        )


def test_gpu_optional_preflight_cannot_construct_publication_receipt(tmp_path):
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="preflight must not call"):
        _CONTRACT.require_numeric_execution_environment(
            tmp_path, {}, {}, require_cuda=False,
            driver_path=tmp_path / "driver.py", driver_argv=[],
        )


def test_live_root_process_cannot_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(_CONTRACT.os, "getuid", lambda: 0)
    monkeypatch.setattr(_CONTRACT.os, "getgid", lambda: 0)
    monkeypatch.setattr(_CONTRACT.os, "getgroups", lambda: [0])
    monkeypatch.setattr(_CONTRACT.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(
        _CONTRACT, "_require_safe_environment", lambda *_args, **_kwargs: None
    )
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="non-root UID/GID/groups"):
        _CONTRACT.require_numeric_execution_environment(
            tmp_path, _current_environment(), {
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            }, require_cuda=True, driver_path=tmp_path / "driver.py",
            driver_argv=["driver.py"],
        )


def test_host_attestation_writer_is_no_clobber_and_fsyncs(tmp_path, monkeypatch):
    path, _repo, _git, _driver, inspected = _fixture(tmp_path)
    monkeypatch.setattr(_CONTRACT, "_docker_inspect", lambda _container: inspected)
    monkeypatch.setattr(_CONTRACT, "_docker_diff", lambda _container: ())
    fsync_calls = []
    monkeypatch.setattr(_CONTRACT.os, "fsync", fsync_calls.append)
    _CONTRACT.write_launch_attestation("c" * 64, "sparky", path)
    assert len(fsync_calls) == 2
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="without clobbering"):
        _CONTRACT.write_launch_attestation("c" * 64, "sparky", path)


def test_host_attestation_rejects_prestart_rootfs_mutation(tmp_path):
    path, _repo, _git, _driver, inspected = _fixture(tmp_path)
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="writable-layer"):
        _CONTRACT.build_launch_attestation(
            inspected, "sparky", host_output_path=path,
            rootfs_changes=("C /usr/bin/python3",),
        )


def test_host_attestation_two_phase_inspection_detects_toctou(tmp_path, monkeypatch):
    path, _repo, _git, _driver, inspected = _fixture(tmp_path)
    changed = copy.deepcopy(inspected)
    changed["HostConfig"]["ReadonlyRootfs"] = False
    inspections = iter((inspected, changed))
    monkeypatch.setattr(_CONTRACT, "_docker_inspect", lambda _container: next(inspections))
    monkeypatch.setattr(_CONTRACT, "_docker_diff", lambda _container: ())
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="changed during"):
        _CONTRACT.write_launch_attestation("c" * 64, "sparky", path)
    assert not path.exists()

    path, _repo, _git, _driver, inspected = _fixture(tmp_path / "diff-race")
    monkeypatch.setattr(_CONTRACT, "_docker_inspect", lambda _container: inspected)
    diffs = iter(((), ("C /usr/bin/python3",)))
    monkeypatch.setattr(_CONTRACT, "_docker_diff", lambda _container: next(diffs))
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="rootfs changed"):
        _CONTRACT.write_launch_attestation("c" * 64, "sparky", path)
    assert not path.exists()


def test_repo_commit_never_degrades_to_none(tmp_path, monkeypatch):
    monkeypatch.setattr(_CONTRACT, "_git", lambda *_args: "unknown")
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="not a full lowercase commit"):
        _CONTRACT.require_repo_commit(tmp_path)


def _git_fixture(tmp_path: Path, ignored: str | None = None) -> None:
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True,
            capture_output=True, text=True,
        )

    git("init", "-q")
    git("config", "user.name", "Numeric Contract Test")
    git("config", "user.email", "numeric-contract@example.invalid")
    if ignored is not None:
        (tmp_path / ".gitignore").write_text(ignored + "\n", encoding="utf-8")
    (tmp_path / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "fixture")


def test_tracked_bytes_check_defeats_git_assume_unchanged(tmp_path):
    _git_fixture(tmp_path)
    subprocess.run(
        ["git", "-C", str(tmp_path), "update-index", "--assume-unchanged", "tracked.py"],
        check=True,
    )
    (tmp_path / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="tracked source bytes"):
        _CONTRACT.require_repo_tree_matches_head(tmp_path)


@pytest.mark.parametrize("suffix", [".py", ".zip", ".egg", ".whl"])
def test_tracked_tree_rejects_ignored_importable_code_archives(tmp_path, suffix):
    name = "shadow" + suffix
    _git_fixture(tmp_path, name)
    (tmp_path / name).write_bytes(b"attacker import payload")
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="ignored executable source/code"):
        _CONTRACT.require_repo_tree_matches_head(tmp_path)


def test_tracked_tree_rejects_ignored_symlink_import_path(tmp_path):
    _git_fixture(tmp_path, "shadow_package")
    external = tmp_path.parent / (tmp_path.name + "-external-package")
    external.mkdir()
    (external / "__init__.py").write_text("value = 'attacker'\n", encoding="utf-8")
    (tmp_path / "shadow_package").symlink_to(external, target_is_directory=True)
    with pytest.raises(_CONTRACT.NumericExecutionContractError, match="ignored executable source/code"):
        _CONTRACT.require_repo_tree_matches_head(tmp_path)
