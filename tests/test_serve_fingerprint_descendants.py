"""Ancestry-scoped evidence for in-process vLLM gold measurements."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import pytest

import tools.serve_fingerprint as serve_fingerprint


@pytest.mark.parametrize("argv,name,expected", [
    (["python3", "measure_boundary_control.py", "--image", "eugr/spark-vllm@sha256:abc"], "python3", False),
    (["python3", "client.py", "--model", "/models/vllm-test"], "python3", False),
    (["bash", "-lc", "vllm serve /model"], "bash", False),
    (["python3", "/usr/local/bin/vllm", "serve", "/model"], "python3", True),
    (["vllm", "serve", "/model"], "vllm", True),
    (["python3", "-m", "vllm.entrypoints.openai.api_server"], "python3", True),
    (["VLLM::EngineCore"], "VLLM::EngineCor", True),
    (["python3", "-c", "multiprocessing.spawn"], "VLLM::Worker_TP0", True),
])
def test_server_discovery_uses_process_identity_not_payload_substrings(monkeypatch, argv, name, expected):
    monkeypatch.setattr(serve_fingerprint, "_read_cmdline", lambda _pid: argv)
    monkeypatch.setattr(serve_fingerprint, "_read_process_name", lambda _pid: name)
    assert serve_fingerprint._looks_like_vllm_process(123) is expected


def test_stable_fingerprint_excludes_phase_but_binds_runtime_stack_fields():
    pre = {
        "attestation_phase": "pre",
        "created": "2026-08-13T00:00:00Z",
        "resident_extensions": ["prismaquant/kernels/nvfp4_fused.so"],
        "package_versions": {"vllm": "0.10.2"},
        "image": "vllm-node:latest",
        "vllm_compilation_provenance": {
            "torch_compile_wrapper": "unwrapped",
        },
    }
    pre["serve_fingerprint"] = serve_fingerprint.fingerprint(pre)
    post = deepcopy(pre)
    post["attestation_phase"] = "post"
    post["created"] = "2026-08-13T00:02:00Z"
    post["serve_fingerprint"] = serve_fingerprint.fingerprint(post)
    assert post["serve_fingerprint"] == pre["serve_fingerprint"]

    mutations = []
    resident = deepcopy(post)
    resident["resident_extensions"].append("unexpected_extension.so")
    mutations.append(resident)
    package = deepcopy(post)
    package["package_versions"]["vllm"] = "0.10.3"
    mutations.append(package)
    image = deepcopy(post)
    image["image"] = "vllm-node:2026-07-01"
    mutations.append(image)
    compilation = deepcopy(post)
    compilation["vllm_compilation_provenance"]["torch_compile_wrapper"] = (
        "wrapped"
    )
    mutations.append(compilation)
    for mutated in mutations:
        mutated["serve_fingerprint"] = serve_fingerprint.fingerprint(mutated)
        assert mutated["serve_fingerprint"] != pre["serve_fingerprint"]


def test_server_environment_allowlist_is_path_vars_plus_the_pinned_residency_knob():
    """Two interpreter-path variables that must be ABSENT, plus one serving
    runtime knob whose VALUE is recorded.

    The first two prove the serving process resolved its imports from the
    installed distribution rather than a working-directory shadow, which is
    what the fingerprint is for. The third is the Tessera plugin's one
    operator knob: a serve that omits it serves a different residency than the
    pin's receipts covered, so its value rides the performance-stack
    fingerprint. The name is the pin's
    (`prismaquant/tessera_runtime/tessera_serving_runtime_pin.json`), not this
    file's opinion -- a serving-lane env belongs to another runtime.
    """
    import json
    from pathlib import Path

    pin = json.loads((
        Path(__file__).resolve().parents[1] / "prismaquant" / "tessera_runtime"
        / "tessera_serving_runtime_pin.json"
    ).read_text(encoding="utf-8"))

    assert serve_fingerprint.SERVER_ENV_ALLOWLIST == (
        "PYTHONPATH",
        "PYTHONSAFEPATH",
        pin["serving_residency_env"],
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
        return ["prismaquant/kernels/nvfp4_fused.so"], list(pids), []

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
