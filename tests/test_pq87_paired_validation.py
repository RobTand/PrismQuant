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
