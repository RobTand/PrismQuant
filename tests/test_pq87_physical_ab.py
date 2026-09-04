"""Resource and frozen-artifact guards for the opt-in physical instrument."""
import importlib

import pytest


def test_server_has_explicit_small_memory_population():
    driver = importlib.import_module("experiments.pq87_physical_ab")
    argv = driver.server_argv("nonce")
    assert argv[argv.index("--kv-cache-memory-bytes") + 1] == str(1 << 30)
    assert argv[argv.index("--max-model-len") + 1] == "4096"
    assert argv[argv.index("--max-num-seqs") + 1] == "1"
    assert "--gpu-memory-utilization" not in argv


def test_prelaunch_content_must_match_client_capture():
    driver = importlib.import_module("experiments.pq87_physical_ab")
    driver.require_frozen_content({"weight": "a"}, {"weight": "a"})
    with pytest.raises(ValueError, match="pre-launch"):
        driver.require_frozen_content({"weight": "a"}, {"weight": "b"})


def test_elapsed_deadline_refuses_further_work(monkeypatch):
    driver = importlib.import_module("experiments.pq87_physical_ab")
    monkeypatch.setattr(driver.time, "monotonic", lambda: 100)
    assert driver.remaining(110) == 10
    with pytest.raises(TimeoutError, match="deadline"):
        driver.remaining(100)


def test_health_endpoint_accepts_successful_empty_body(monkeypatch):
    import io
    driver = importlib.import_module("experiments.pq87_physical_ab")
    monkeypatch.setattr(driver.urllib.request, "urlopen", lambda *_a, **_kw: io.BytesIO(b""))
    assert driver._http("http://fixture/health") is None


@pytest.mark.parametrize("running, expected", [(True, False), (False, True)])
def test_cleanup_verifies_live_container_state(monkeypatch, running, expected):
    from types import SimpleNamespace
    driver = importlib.import_module("experiments.pq87_physical_ab")
    monkeypatch.setattr(driver.subprocess, "run", lambda argv, **_kwargs: SimpleNamespace(
        returncode=0, stdout='[{"State": {"Running": ' + str(running).lower() + '}}]'
        if argv[1] == "inspect" else "", stderr=""))
    assert driver.cleanup_container("exact-owned-name")["safe"] is expected


def test_cleanup_accepts_actual_lowercase_docker_absence_diagnostic(monkeypatch):
    from types import SimpleNamespace
    driver = importlib.import_module("experiments.pq87_physical_ab")
    monkeypatch.setattr(driver.subprocess, "run", lambda argv, **_kwargs: SimpleNamespace(
        returncode=1, stdout="[]\nerror: no such object: exact-owned-name\n"))
    assert driver.cleanup_container("exact-owned-name")["safe"] is True
