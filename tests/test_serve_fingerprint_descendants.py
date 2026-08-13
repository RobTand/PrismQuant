"""Ancestry-scoped evidence for in-process vLLM gold measurements."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.serve_fingerprint as serve_fingerprint
from prismaquant.gridbook_environment import GRIDBOOK_ENVIRONMENT_ALLOWLIST


def test_server_environment_allowlist_is_the_full_gridbook_registry():
    assert serve_fingerprint.SERVER_ENV_ALLOWLIST == (
        "PQ_GRIDBOOK_RUNTIME_COMMIT",
        "PQ_GRIDBOOK_RUNTIME_VERSION",
        "PYTHONPATH",
        "PYTHONSAFEPATH",
        *GRIDBOOK_ENVIRONMENT_ALLOWLIST,
    )


def _write_status(root: Path, pid: int, text: str) -> None:
    directory = root / str(pid)
    directory.mkdir()
    (directory / "status").write_text(text, encoding="utf-8")


def test_descendant_census_is_transitive_and_ppid_fail_closed(tmp_path):
    _write_status(tmp_path, 100, "Name:\tparent\nPPid:\t1\n")
    _write_status(tmp_path, 101, "Name:\thelper\nPPid:\t100\n")
    _write_status(tmp_path, 102, "Name:\tengine\nPPid:\t101\n")
    _write_status(tmp_path, 103, "Name:\tunrelated-vllm\nPPid:\t1\n")
    _write_status(tmp_path, 104, "Name:\tmalformed\nPPid:\tnot-a-pid\n")
    _write_status(tmp_path, 105, "PPid:\t100\nPPid:\t1\n")
    _write_status(tmp_path, 106, "Name:\tself-cycle\nPPid:\t106\n")

    assert serve_fingerprint.descendant_process_pids(
        100, proc_root=tmp_path
    ) == [101, 102]


def test_self_manifest_censuses_parent_and_all_descendants_only(monkeypatch):
    parent = os.getpid()
    helper = 8_100_001
    engine = 8_100_002
    unrelated_engine = 8_100_003
    parents = {
        parent: 1,
        helper: parent,
        engine: helper,
        unrelated_engine: 1,
    }
    argv = {
        helper: ["python", "multiprocessing-helper"],
        engine: ["VLLM::EngineCore"],
        unrelated_engine: ["VLLM::EngineCore"],
    }
    queried_cmdlines: list[int] = []
    inspected: dict[str, list[int]] = {}

    monkeypatch.setattr(
        serve_fingerprint,
        "_proc_pids",
        lambda *, proc_root="/proc": sorted(parents),
    )
    monkeypatch.setattr(
        serve_fingerprint,
        "_read_process_ppid",
        lambda pid, *, proc_root="/proc": parents.get(pid),
    )

    def read_cmdline(pid):
        queried_cmdlines.append(int(pid))
        return argv.get(int(pid), ["python", "measurement.py"])

    def scan_maps(pids):
        inspected["maps"] = list(pids)
        return ["_gridbook_C.so"], list(pids), []

    def scan_environment(pids, names=serve_fingerprint.SERVER_ENV_ALLOWLIST):
        inspected["environment"] = list(pids)
        rows = [
            {"pid": pid, "values": {}, "sha256": "a" * 64}
            for pid in pids
        ]
        return {
            "schema": "prismaquant.server_process_environment/1",
            "allowlist": sorted(set(names)),
            "readable_pids": list(pids),
            "unreadable_pids": [],
            "consistent": True,
            "values": {},
            "processes": rows,
        }

    def identities(pids, *, boot_id):
        inspected["identities"] = list(pids)
        return [
            {
                "pid": pid,
                "argv": argv.get(pid, ["python", "measurement.py"]),
                "cmdline": "test",
                "start_time_ticks": pid,
                "pid_namespace": "pid:[1]",
                "executable": "/usr/bin/python",
                "identity_sha256": f"{pid:064x}",
            }
            for pid in pids
        ]

    def listeners(pids):
        inspected["listeners"] = list(pids)
        return {
            "schema": "prismaquant.server_tcp_listeners/1",
            "tables_readable": True,
            "unreadable_pids": [],
            "listeners": [],
        }

    monkeypatch.setattr(serve_fingerprint, "_read_cmdline", read_cmdline)
    monkeypatch.setattr(serve_fingerprint, "residency_scan", scan_maps)
    monkeypatch.setattr(
        serve_fingerprint, "server_environment_snapshot", scan_environment
    )
    monkeypatch.setattr(serve_fingerprint, "process_identities", identities)
    monkeypatch.setattr(serve_fingerprint, "process_tcp_listeners", listeners)
    monkeypatch.setattr(
        serve_fingerprint,
        "host_identity",
        lambda: {
            "hostname": "test-host",
            "boot_id": "boot",
            "machine_id_sha256": "b" * 64,
            "pid_namespace": "pid:[1]",
        },
    )
    monkeypatch.setattr(
        serve_fingerprint,
        "gpu_identity",
        lambda: {
            "gpu_name": "NVIDIA GB10",
            "gpu_uuid": "GPU-test",
            "driver_version": "test",
            "gpu_count": 1,
        },
    )
    monkeypatch.setattr(serve_fingerprint, "package_versions", lambda: {})
    monkeypatch.setattr(serve_fingerprint, "gridbook_runtime_pin", lambda: None)

    manifest = serve_fingerprint.self_manifest(
        require_engine_descendant=True
    )

    expected = [parent, helper, engine]
    assert manifest["measurement_parent_pid"] == parent
    assert manifest["engine_descendant_pids"] == [engine]
    assert [row["pid"] for row in manifest["processes"]] == expected
    assert inspected == {
        "maps": expected,
        "identities": expected,
        "environment": expected,
        "listeners": expected,
    }
    assert unrelated_engine not in queried_cmdlines
    assert unrelated_engine not in expected


def test_required_engine_descendant_fails_closed(monkeypatch):
    parent = os.getpid()
    child = 8_200_001
    monkeypatch.setattr(
        serve_fingerprint,
        "_proc_pids",
        lambda *, proc_root="/proc": [parent, child],
    )
    monkeypatch.setattr(
        serve_fingerprint,
        "_read_process_ppid",
        lambda pid, *, proc_root="/proc": {parent: 1, child: parent}.get(pid),
    )
    monkeypatch.setattr(
        serve_fingerprint,
        "_read_cmdline",
        lambda pid: ["python", "ordinary-worker.py"],
    )
    monkeypatch.setattr(
        serve_fingerprint,
        "collect_manifest",
        lambda **kwargs: pytest.fail("missing engine must fail before collection"),
    )

    with pytest.raises(ValueError, match="no live EngineCore/VLLM engine"):
        serve_fingerprint.self_manifest(require_engine_descendant=True)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["VLLM::EngineCore"], True),
        (["python", "-m", "vllm.v1.engine.core"], True),
        (["VLLM", "engine"], True),
        (["vllm", "serve", "/model"], False),
        (["python", "unrelated-worker.py"], False),
    ],
)
def test_engine_witness_requires_engine_argv(argv, expected):
    assert serve_fingerprint.argv_identifies_vllm_engine(argv) is expected


def _fake_gridbook_distribution(
    tmp_path: Path,
    pin: dict,
    *,
    transport_url: str | None = None,
    direct_url: dict | None = None,
):
    source_names = (
        "gridbook/__init__.py",
        "gridbook/cuda_ext.py",
        "gridbook/plugin.py",
        "gridbook/runtime_contract.json",
        "gridbook/source_passthrough.py",
        "gridbook/fp8_source_w8a16.py",
        "gridbook/csrc/cb_gemv.cu",
        "gridbook/csrc/fp8_source_w8a16.cu",
        "gridbook/csrc/mxfp8_dense_gemm.cu",
    )
    dist_info = "gridbook-0.8.4.dist-info"
    direct_name = f"{dist_info}/direct_url.json"
    metadata_name = f"{dist_info}/METADATA"
    record_name = f"{dist_info}/RECORD"
    direct = direct_url or {
        "url": transport_url or pin["repository"],
        "vcs_info": {
            "vcs": "git",
            "requested_revision": pin["commit"],
            "commit_id": pin["commit"],
        },
    }
    values = {
        **{name: f"source:{name}\n".encode() for name in source_names},
        direct_name: json.dumps(direct, sort_keys=True).encode(),
        metadata_name: b"Name: gridbook\nVersion: 0.8.4\n",
    }
    rows = []
    for name, data in values.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        digest = base64.urlsafe_b64encode(
            hashlib.sha256(data).digest()
        ).decode().rstrip("=")
        rows.append(f"{name},sha256={digest},{len(data)}\n")
    rows.append(f"{record_name},,\n")
    record_path = tmp_path / record_name
    record_path.write_text("".join(rows), encoding="utf-8")

    class FakeDistribution:
        metadata = {"Name": "gridbook"}
        version = "0.8.4"
        files = [
            serve_fingerprint.importlib_metadata.PackagePath(name)
            for name in (*values, record_name)
        ]

        @staticmethod
        def locate_file(item):
            return tmp_path / str(item)

    init_path = tmp_path / "gridbook" / "__init__.py"
    module = SimpleNamespace(
        __version__="0.8.4",
        __file__=str(init_path),
        __path__=[str(init_path.parent)],
        __spec__=SimpleNamespace(origin=str(init_path)),
    )
    return FakeDistribution(), direct_name, module


def test_gridbook_distribution_attestation_binds_vcs_record_and_sources(
    tmp_path, monkeypatch,
):
    pin = {
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "5" * 40,
        "version": "0.8.4",
    }
    distribution, _direct_name, module = _fake_gridbook_distribution(tmp_path, pin)
    monkeypatch.setattr(
        serve_fingerprint.importlib_metadata,
        "distribution",
        lambda name: distribution,
    )
    monkeypatch.setattr(
        serve_fingerprint.importlib,
        "import_module",
        lambda name: module,
    )

    evidence = serve_fingerprint.gridbook_distribution_provenance(pin)
    assert evidence["repository"] == pin["repository"]
    assert evidence["direct_url"]["vcs_info"]["commit_id"] == pin["commit"]
    assert evidence["record_identity"]["sha256"]
    assert evidence["schema"] == "prismaquant.installed_gridbook_distribution/2"
    assert evidence["import_origin"]["module_file"].endswith(
        "/gridbook/__init__.py"
    )
    assert evidence["import_origin"]["identity_sha256"]
    assert evidence["source_files_sha256"] == serve_fingerprint._canonical_sha256(
        evidence["source_files"]
    )


def test_gridbook_distribution_accepts_exact_local_vcs_transport(
    tmp_path, monkeypatch,
):
    pin = {
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "5" * 40,
        "version": "0.8.4",
    }
    distribution, _direct_name, module = _fake_gridbook_distribution(
        tmp_path,
        pin,
        transport_url="file:///tmp/gridbook-runtime-555555555555",
    )
    monkeypatch.setattr(
        serve_fingerprint.importlib_metadata,
        "distribution",
        lambda name: distribution,
    )
    monkeypatch.setattr(
        serve_fingerprint.importlib,
        "import_module",
        lambda name: module,
    )

    evidence = serve_fingerprint.gridbook_distribution_provenance(pin)
    assert evidence["repository"] == pin["repository"]
    assert evidence["direct_url"]["url"].startswith("file:///tmp/")
    assert evidence["direct_url"]["vcs_info"] == {
        "vcs": "git",
        "requested_revision": pin["commit"],
        "commit_id": pin["commit"],
    }


def test_gridbook_distribution_accepts_exact_release_wheel(
    tmp_path, monkeypatch,
):
    wheel_sha256 = "6" * 64
    pin = {
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "5" * 40,
        "version": "0.8.4",
        "wheel_sha256": wheel_sha256,
    }
    direct_url = {
        "url": "file:///opt/gridbook-install/gridbook-0.8.4-py3-none-any.whl",
        "archive_info": {
            "hash": f"sha256={wheel_sha256}",
            "hashes": {"sha256": wheel_sha256},
        },
    }
    distribution, _direct_name, module = _fake_gridbook_distribution(
        tmp_path, pin, direct_url=direct_url
    )
    monkeypatch.setattr(
        serve_fingerprint.importlib_metadata,
        "distribution",
        lambda name: distribution,
    )
    monkeypatch.setattr(
        serve_fingerprint.importlib,
        "import_module",
        lambda name: module,
    )

    evidence = serve_fingerprint.gridbook_distribution_provenance(pin)
    assert evidence["direct_url"] == direct_url
    assert serve_fingerprint.validate_gridbook_pep610_direct_url(
        direct_url, pin
    ) == "wheel"


def test_gridbook_distribution_rejects_wrong_release_wheel_digest(
    tmp_path, monkeypatch,
):
    pin = {
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "5" * 40,
        "version": "0.8.4",
        "wheel_sha256": "6" * 64,
    }
    direct_url = {
        "url": "file:///opt/gridbook-install/gridbook-0.8.4-py3-none-any.whl",
        "archive_info": {"hashes": {"sha256": "7" * 64}},
    }
    distribution, _direct_name, module = _fake_gridbook_distribution(
        tmp_path, pin, direct_url=direct_url
    )
    monkeypatch.setattr(
        serve_fingerprint.importlib_metadata,
        "distribution",
        lambda name: distribution,
    )
    monkeypatch.setattr(
        serve_fingerprint.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(ValueError, match="exact pinned release wheel"):
        serve_fingerprint.gridbook_distribution_provenance(pin)


def test_gridbook_distribution_attestation_rejects_non_vcs_direct_url(
    tmp_path, monkeypatch,
):
    pin = {
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "5" * 40,
        "version": "0.8.4",
    }
    distribution, direct_name, module = _fake_gridbook_distribution(tmp_path, pin)
    (tmp_path / direct_name).write_text(
        json.dumps({"url": "file:///tmp/gridbook", "dir_info": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        serve_fingerprint.importlib_metadata,
        "distribution",
        lambda name: distribution,
    )
    monkeypatch.setattr(
        serve_fingerprint.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(ValueError, match="PEP 610"):
        serve_fingerprint.gridbook_distribution_provenance(pin)


def test_gridbook_distribution_rejects_stale_pythonpath_shadow(
    tmp_path, monkeypatch,
):
    pin = {
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "5" * 40,
        "version": "0.8.4",
    }
    installed = tmp_path / "installed"
    distribution, _direct_name, _module = _fake_gridbook_distribution(
        installed, pin
    )
    stale_init = tmp_path / "stale-pythonpath" / "gridbook" / "__init__.py"
    stale_init.parent.mkdir(parents=True)
    stale_init.write_text('__version__ = "0.8.4"\n', encoding="utf-8")
    stale_module = SimpleNamespace(
        __version__="0.8.4",
        __file__=str(stale_init),
        __path__=[str(stale_init.parent)],
        __spec__=SimpleNamespace(origin=str(stale_init)),
    )
    monkeypatch.setattr(
        serve_fingerprint.importlib_metadata,
        "distribution",
        lambda name: distribution,
    )
    monkeypatch.setattr(
        serve_fingerprint.importlib,
        "import_module",
        lambda name: stale_module,
    )

    with pytest.raises(ValueError, match="CWD/PYTHONPATH shadow"):
        serve_fingerprint.gridbook_distribution_provenance(pin)


def test_gridbook_distribution_rejects_extra_import_path_outside_distribution(
    tmp_path, monkeypatch,
):
    pin = {
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "5" * 40,
        "version": "0.8.4",
    }
    installed = tmp_path / "installed"
    distribution, _direct_name, module = _fake_gridbook_distribution(
        installed, pin
    )
    stale_path = tmp_path / "stale-pythonpath" / "gridbook"
    stale_path.mkdir(parents=True)
    module.__path__.append(str(stale_path))
    monkeypatch.setattr(
        serve_fingerprint.importlib_metadata,
        "distribution",
        lambda name: distribution,
    )
    monkeypatch.setattr(
        serve_fingerprint.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(ValueError, match="escapes the selected installed"):
        serve_fingerprint.gridbook_distribution_provenance(pin)


def test_gridbook_distribution_rejects_imported_version_mismatch(
    tmp_path, monkeypatch,
):
    pin = {
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "5" * 40,
        "version": "0.8.4",
    }
    distribution, _direct_name, module = _fake_gridbook_distribution(
        tmp_path, pin
    )
    module.__version__ = "0.1.0"
    monkeypatch.setattr(
        serve_fingerprint.importlib_metadata,
        "distribution",
        lambda name: distribution,
    )
    monkeypatch.setattr(
        serve_fingerprint.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(ValueError, match="imported Gridbook version"):
        serve_fingerprint.gridbook_distribution_provenance(pin)
