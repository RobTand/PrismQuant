"""The physical validation is a frozen sequential state machine, not a launch loop."""
import copy
import importlib
import json
from types import SimpleNamespace
from pathlib import Path

import pytest


def _driver():
    return importlib.import_module("experiments.pq87_paired_validation")


def _manifest():
    return json.loads((Path(__file__).parents[1] / "experiments" /
                       "pq87_validation_manifest.json").read_text())


def test_registered_manifest_is_disjoint_and_has_both_roles():
    result = _driver().validate_manifest(_manifest())
    assert set(result["models"]) == {"control", "candidate"}
    screen = {p["text"] for p in result["contracts"]["screen"]["prompts"]}
    heldout = {p["text"] for p in result["contracts"]["heldout"]["prompts"]}
    assert not screen & heldout
    assert result["contracts"]["screen"] == json.loads((Path(__file__).parents[1] /
        "experiments" / "pq87_boundary_contract.json").read_text())


@pytest.mark.parametrize("mutation", ["overlap", "image", "deadline", "same_model", "same_telemetry"])
def test_manifest_refuses_ambiguous_or_unbounded_campaign(mutation):
    manifest = copy.deepcopy(_manifest())
    if mutation == "overlap":
        manifest["contracts"]["heldout"]["prompts"][0]["text"] = manifest["contracts"]["screen"]["prompts"][0]["text"]
    elif mutation == "image":
        manifest["image"] = "runtime:latest"
    elif mutation == "deadline":
        manifest["deadline_seconds"] = 0
    elif mutation == "same_model":
        manifest["models"]["candidate"] = manifest["models"]["control"]
    else:
        manifest["netdata_urls"] *= 0
    with pytest.raises(ValueError):
        _driver().validate_manifest(manifest)


def test_both_model_arms_reuse_the_existing_bounded_server_configuration():
    driver = _driver()
    from tools.serve_fingerprint import normalize_performance_argv
    control = driver.server_argv("nonce", "/models/control")
    candidate = driver.server_argv("nonce", "/models/candidate")
    assert normalize_performance_argv(control) == normalize_performance_argv(candidate)
    assert control[control.index("--kv-cache-memory-bytes") + 1] == str(1 << 30)
    assert control[control.index("--max-num-seqs") + 1] == "1"
    assert "--quantization" not in control + candidate


def test_candidate_refusal_is_an_observation_but_missing_measurement_is_not():
    driver = _driver()
    assert driver.completed_candidate({"decision": {"verdict": "refused"}}, 2)
    assert driver.completed_candidate({"decision": {"verdict": "accepted"}}, 0)
    assert not driver.completed_candidate({"decision": {"verdict": "accepted"}}, 2)
    assert not driver.completed_candidate({}, 2)
    assert not driver.completed_candidate({"decision": {"verdict": "inconclusive"}}, 2)


def test_cpu_only_pool_action_cannot_start_the_physical_controller(monkeypatch):
    driver = _driver()
    monkeypatch.setenv("PRISMABUILD_CONTAINER_OWNER", "cpu-action")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(driver.instrument, "bootstrap", lambda: pytest.fail("CPU action reached physical preparation"))
    with pytest.raises(ValueError, match="GPU"):
        driver.host(SimpleNamespace(manifest="unused", out="unused"))


@pytest.mark.parametrize("field,value", [("residency_readable", False), ("serve_session_id", None)])
def test_ppl_binding_requires_observed_live_residency(field, value):
    before = {"residency_readable": True, "serve_session_id": "live",
              "model": "/models/candidate", "models_endpoint_binding": {"model": {"id": "nonce"}}}
    before[field] = value
    with pytest.raises(ValueError, match="residency|session"):
        _driver().require_ppl_binding(before, "/models/candidate", "nonce")


def test_prelaunch_failure_never_cleans_a_basename_container(tmp_path, monkeypatch):
    driver = _driver()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest()))
    monkeypatch.setenv("PRISMABUILD_CONTAINER_OWNER", "a" * 64)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(driver.instrument, "bootstrap", lambda: (
        tmp_path, None, SimpleNamespace(digest=lambda value: "digest")))
    commands = []
    def check_output(command, **kwargs):
        commands.append(command)
        if command[0] == "git":
            return "source\n"
        raise RuntimeError("prelaunch image missing")
    monkeypatch.setattr(driver.subprocess, "check_output", check_output)
    monkeypatch.setattr(driver.instrument, "cleanup_container", lambda name: pytest.fail(
        f"cleaned uncreated container {name}"))
    assert driver.host(SimpleNamespace(manifest=str(manifest), out=str(tmp_path / "shared-basename"))) == 0
    assert not any(command[1] in {"stop", "rm"} for command in commands)


def test_container_name_collision_refuses_without_cleanup(monkeypatch):
    driver = _driver()
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout='[{"Id":"existing"}]')
    monkeypatch.setattr(driver.subprocess, "run", run)
    with pytest.raises(ValueError, match="already exists"):
        driver.require_container_name_available("already-owned")
    assert commands == [["docker", "inspect", "already-owned"]]


@pytest.mark.parametrize("mismatch", ["action", "nonce", "id"])
def test_cleanup_refuses_an_unowned_container_id(tmp_path, monkeypatch, mismatch):
    driver = _driver()
    cid = "c" * 64
    cidfile = tmp_path / "container.cid"
    cidfile.write_text(cid + "\n")
    observed = {"Id": cid, "Name": "/our-name", "Config": {"Labels": {
        "prismabuild.action": "a" * 64, "prismaquant.pq87.campaign": "nonce"}}}
    if mismatch == "id":
        observed["Id"] = "b" * 64
    else:
        observed["Config"]["Labels"]["prismabuild.action" if mismatch == "action"
                                     else "prismaquant.pq87.campaign"] = "other"
    monkeypatch.setattr(driver.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps([observed])))
    monkeypatch.setattr(driver.instrument, "cleanup_container", lambda name: pytest.fail(
        f"cleaned unowned container {name}"))
    result = driver.cleanup_owned_container(cidfile, "our-name", "a" * 64, "nonce", attempted=True)
    assert not result["safe"]
