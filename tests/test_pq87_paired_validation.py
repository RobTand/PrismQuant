"""The physical validation is a frozen sequential state machine, not a launch loop."""
import copy
import importlib
import io
import json
import sys
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


@pytest.mark.parametrize("role", ["control", "candidate"])
def test_paired_server_startup_budget_fits_the_reserved_container(role):
    # A fixed 1 GiB KV cache does not bypass vLLM's device-startup fraction
    # check. The image's 0.92 default requests 111.9 GiB on our GB10, far
    # outside this campaign's 28 GiB container and 32 GiB fleet reservation.
    argv = _driver().server_argv("nonce", f"/models/{role}")
    flag = "--gpu-memory-utilization"
    assert argv.count(flag) == 1, "paired server must declare its startup memory budget"
    fraction = float(argv[argv.index(flag) + 1])
    assert fraction == 0.2
    assert (1 << 30) < fraction * 121.63 * (1 << 30) < 28 * (1 << 30)


def _arm_receipts(monkeypatch, *, control_texts=None, candidate_texts):
    """The two `prismaquant.boundary_campaign_arm/1` files the driver reads back."""
    from prismaquant import boundary_control as bc
    from test_boundary_policy import CLEAN, _pair
    control, candidate = _pair(monkeypatch, control_texts or (CLEAN, CLEAN), candidate_texts)
    decision = bc.decide_no_new_failures(control, candidate)
    return ({"measurement": control}, {"measurement": candidate, "decision": decision},
            bc.replay_no_new_failures)


def test_candidate_refusal_is_an_observation_but_missing_measurement_is_not(monkeypatch):
    from test_boundary_policy import CLEAN, ZERO
    driver = _driver()
    control, accepted, replay = _arm_receipts(monkeypatch, candidate_texts=(CLEAN, CLEAN))
    _, refused, _ = _arm_receipts(monkeypatch, candidate_texts=(ZERO, CLEAN))
    assert refused["decision"]["verdict"] == "refused"
    assert driver.completed_candidate(refused, 2, control, replay)
    assert driver.completed_candidate(accepted, 0, control, replay)
    # The exit status and the receipt must agree; either alone is not the observation.
    assert not driver.completed_candidate(accepted, 2, control, replay)
    assert not driver.completed_candidate(refused, 0, control, replay)
    assert not driver.completed_candidate({}, 2, control, replay)
    assert not driver.completed_candidate({"measurement": accepted["measurement"]}, 0, control, replay)


def test_stored_candidate_verdict_is_replayed_against_the_control_not_trusted(monkeypatch):
    """`decision.verdict` in the receipt is a claim; the driver recomputes it.

    The receipt is written by a subprocess the driver launched seconds earlier,
    which is exactly why trusting the stored bit would go unnoticed: nothing
    else between the write and the read re-derives the verdict, and the
    offline verifier `replay_no_new_failures` otherwise has no caller outside
    the tests (pq-190 review, side-finding 3).
    """
    from test_boundary_policy import CLEAN, ZERO
    driver = _driver()
    control, receipt, replay = _arm_receipts(monkeypatch, candidate_texts=(ZERO, CLEAN))
    assert receipt["decision"]["verdict"] == "refused"
    forged = copy.deepcopy(receipt)
    forged["decision"]["verdict"] = "accepted"
    with pytest.raises(ValueError, match="boundary decision differs from replayed paired policy"):
        driver.completed_candidate(forged, 0, control, replay)
    # A verdict that replays only against some other control is not this campaign's.
    other_control, _, _ = _arm_receipts(monkeypatch, control_texts=(ZERO, ZERO),
                                        candidate_texts=(ZERO, CLEAN))
    assert other_control != control
    with pytest.raises(ValueError, match="boundary decision differs from replayed paired policy"):
        driver.completed_candidate(receipt, 2, other_control, replay)


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


def _ppl_result():
    from prismaquant.validate_quantized_model import CheckResult, EVAL_PROMPTS
    return CheckResult(name="perplexity", passed=False, detail="measured threshold refusal", metrics={
        "perplexity": 22026.465794806718, "mean_nll_per_tok": 10.0,
        "p99_nll_per_tok": 10.0, "max_nll_per_tok": 10.0,
        "per_prompt_avg_nll": [10.0] * len(EVAL_PROMPTS),
        "n_tokens": 100, "spec_decode_detected": False})


@pytest.mark.parametrize("mutation", ["empty", "skipped", "unknown_spec", "speculative",
                                      "nonfinite", "n_tokens", "partial"])
def test_ppl_missing_measurement_is_not_completion(mutation):
    from prismaquant.validate_quantized_model import EVAL_PROMPTS
    result = _ppl_result()
    if mutation == "empty":
        result.metrics.clear()
    elif mutation in {"skipped", "unknown_spec", "speculative"}:
        result.metrics.clear()
        result.metrics.update(skipped=True, spec_decode_detected={
            "skipped": False, "unknown_spec": None, "speculative": True}[mutation])
    elif mutation == "nonfinite":
        result.metrics["mean_nll_per_tok"] = float("nan")
    elif mutation == "n_tokens":
        result.metrics["n_tokens"] = 0
    else:
        result.metrics["per_prompt_avg_nll"].pop()
    with pytest.raises(ValueError, match="PPL"):
        _driver().require_ppl_measurement(result, len(EVAL_PROMPTS))


def test_ppl_measured_threshold_refusal_is_a_complete_observation():
    from prismaquant.validate_quantized_model import EVAL_PROMPTS
    result = _ppl_result()
    assert _driver().require_ppl_measurement(result, len(EVAL_PROMPTS)) is result


@pytest.mark.parametrize("metrics", [{}, {"skipped": True, "spec_decode_detected": None}])
def test_actual_ppl_phase_refuses_missing_numeric_observation(monkeypatch, metrics):
    driver = _driver()
    from prismaquant import boundary_control as bc
    result = bc.vqm.CheckResult(name="perplexity", passed=False, detail="missing logprobs", metrics=metrics)
    content = {"frozen": "content"}
    client = SimpleNamespace(_capture_artifact=lambda model: (content, {"stable": True}))
    before = {"residency_readable": True, "serve_session_id": "live",
              "performance_stack_fingerprint": "stack", "model": "/models/control",
              "models_endpoint_binding": {"model": {"id": "nonce"}}}
    manifest = _manifest()
    monkeypatch.setattr(driver.instrument, "bootstrap", lambda: (Path("/repo"), client, bc))
    monkeypatch.setitem(sys.modules, "serve_fingerprint", SimpleNamespace(collect_manifest=lambda **kwargs: before))
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: json.dumps(
        manifest if str(self) == "/run/manifest.json" else {"content": content}))
    monkeypatch.setattr(driver, "open", lambda *args, **kwargs: io.StringIO(), raising=False)
    monkeypatch.setattr(driver.instrument, "dump", lambda *args, **kwargs: None)
    monkeypatch.setattr(bc.vqm, "check_perplexity", lambda *args: result)
    monkeypatch.setattr(bc, "artifact_content_id", lambda value: "artifact")
    # client_phase wraps these attributes for journalling; restore them after the fixture.
    monkeypatch.setattr(bc.vqm, "_post_json", bc.vqm._post_json)
    monkeypatch.setattr(bc, "measure_step", bc.measure_step)
    with pytest.raises(ValueError, match="PPL"):
        driver.client_phase(SimpleNamespace(role="control", population="ppl", nonce="nonce"))
