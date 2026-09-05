"""The opt-in verdict is paired: repaired samples cannot buy new failures."""
from __future__ import annotations

import copy
import json

import pytest

from prismaquant import boundary_control as bc
from test_boundary_control import _binding, _contract, _response
from test_measure_boundary_control_identity import campaign


CLEAN = "<think>work</think>81"
ZERO = "81"
STUTTER = "<think>work</think></think>81"


def _pair(monkeypatch, control_texts=(CLEAN, CLEAN), candidate_texts=(CLEAN, CLEAN),
          *, candidate_finish="stop"):
    contract = {**_contract(), "initial_max_tokens": 16}
    texts = iter(control_texts)
    monkeypatch.setattr(bc.vqm, "_post_json", lambda _url, body:
                        _response(body["max_tokens"], text=next(texts)))
    control = bc.measure_control("http://fixture", "control", contract, _binding())
    texts = iter(candidate_texts)
    monkeypatch.setattr(bc.vqm, "_post_json", lambda _url, body:
                        _response(body["max_tokens"], text=next(texts), finish=candidate_finish))
    candidate = bc.measure_candidate("http://fixture", "candidate", control, _binding("quant"))
    return control, candidate


def test_clean_paired_candidate_is_accepted_only_by_the_opt_in_policy(monkeypatch):
    control, candidate = _pair(monkeypatch)
    assert "verdict" not in bc.compare(control, candidate)
    result = bc.decide_no_new_failures(control, candidate)
    assert result["schema"] == "prismaquant.boundary_decision/1"
    assert result["policy"] == "prismaquant.no_new_boundary_failures/1"
    assert result["advisory_only"] is True
    assert result["verdict"] == "accepted"
    assert result["pair_count"] == 2
    assert result["violations"] == []
    assert result["control_sha256"] == bc.digest(control)
    assert result["candidate_sha256"] == bc.digest(candidate)
    assert bc.replay_no_new_failures(result, control, candidate) == result


def test_repaired_pair_cannot_offset_new_failure_even_in_same_stratum(monkeypatch):
    control, candidate = _pair(monkeypatch, (ZERO, CLEAN), (CLEAN, ZERO))
    assert bc.compare(control, candidate)["strata"]["numeric"]["delta_defects"] == 0
    result = bc.decide_no_new_failures(control, candidate)
    assert result["verdict"] == "refused"
    assert result["newly_broken_pairs"] == 1
    assert result["new_defect_kind_pairs"] == 0
    assert result["violations"] == [{
        "prompt_index": 0, "seed": 47, "stratum": "numeric",
        "control_defects": [], "candidate_defects": ["zero_tag"],
        "new_defects": ["zero_tag"],
    }]


def test_new_kind_on_already_broken_pair_refuses(monkeypatch):
    control, candidate = _pair(monkeypatch, (ZERO, CLEAN), (STUTTER, CLEAN))
    result = bc.decide_no_new_failures(control, candidate)
    assert result["verdict"] == "refused"
    assert result["newly_broken_pairs"] == 0
    assert result["new_defect_kind_pairs"] == 1
    assert result["violations"][0]["new_defects"] == ["think_stutter"]


def test_existing_defects_and_repairs_are_permitted_without_new_kinds(monkeypatch):
    control, candidate = _pair(monkeypatch, (ZERO, STUTTER), (ZERO, CLEAN))
    result = bc.decide_no_new_failures(control, candidate)
    assert result["verdict"] == "accepted"
    assert result["violations"] == []


def test_candidate_truncation_is_a_new_defect_not_an_inconclusive_control(monkeypatch):
    control, candidate = _pair(monkeypatch, candidate_finish="length")
    result = bc.decide_no_new_failures(control, candidate)
    assert result["verdict"] == "refused"
    assert all(row["new_defects"] == ["cap_truncation"] for row in result["violations"])


@pytest.mark.parametrize("mutation", ["missing_control", "missing_candidate", "binding", "score", "schedule", "control_sha"])
def test_missing_incomparable_or_forged_evidence_is_inconclusive(monkeypatch, mutation):
    control, candidate = _pair(monkeypatch)
    if mutation == "missing_control":
        control = None
    elif mutation == "missing_candidate":
        candidate = None
    elif mutation == "binding":
        candidate["binding"]["serve_fingerprint"] = "d" * 64
    elif mutation == "score":
        candidate["step"]["outcomes"][0]["score"]["defects"] = ["zero_tag"]
    elif mutation == "schedule":
        candidate["step"]["outcomes"].reverse()
    else:
        candidate["control_sha256"] = "d" * 64
    result = bc.decide_no_new_failures(control, candidate)
    assert result["verdict"] == "inconclusive"
    assert result["reason"]
    assert result["pair_count"] == 0


def test_censored_control_remains_inconclusive(monkeypatch):
    contract = {**_contract(), "max_steps": 1}
    monkeypatch.setattr(bc.vqm, "_post_json", lambda _url, body:
                        _response(body["max_tokens"], finish="length"))
    control = bc.measure_control("http://fixture", "control", contract, _binding())
    result = bc.decide_no_new_failures(control, None)
    assert result["verdict"] == "inconclusive"


@pytest.mark.parametrize("field,value", [
    ("verdict", "accepted"), ("policy", "other/1"),
    ("violations", []), ("newly_broken_pairs", 0), ("pair_count", 200),
    ("advisory_only", False),
])
def test_offline_replay_recomputes_every_decision_field(monkeypatch, field, value):
    control, candidate = _pair(monkeypatch, candidate_texts=(ZERO, CLEAN))
    result = copy.deepcopy(bc.decide_no_new_failures(control, candidate))
    result[field] = value
    with pytest.raises(ValueError, match="decision"):
        bc.replay_no_new_failures(result, control, candidate)


@pytest.mark.parametrize("field,value", [("pair_count", 2.0), ("newly_broken_pairs", False)])
def test_replay_does_not_coerce_serialized_counter_types(monkeypatch, field, value):
    control, candidate = _pair(monkeypatch)
    result = bc.decide_no_new_failures(control, candidate)
    result[field] = value
    with pytest.raises(ValueError, match="decision"):
        bc.replay_no_new_failures(result, control, candidate)


def test_default_measurement_records_no_decision_and_no_control_self_certification(campaign, monkeypatch):
    """Without the flag the instrument stays a measurement: no verdict, no exit change."""
    from tools import measure_boundary_control as tool

    model, _weight, run, post, _requests = campaign
    run("control.json")
    def candidate_post(url, body):
        if url.endswith("/tokenize"):
            return post(url, body)
        return _response(body["max_tokens"], text=ZERO)
    monkeypatch.setattr(bc.vqm, "_post_json", candidate_post)
    output = model.parent / "candidate.json"
    argv = ["candidate", "--base-url", "http://fixture", "--model-name", "nonce",
            "--model-dir", str(model), "--image", "fixture@sha256:" + "a" * 64,
            "--control", str(model.parent / "control.json"), "--campaign-id", "same-campaign",
            "--out", str(output)]
    assert tool.main(argv) == 0
    arm = json.loads(output.read_text())
    assert "decision" not in arm
    assert "verdict" not in arm["comparison"]
    # A control arm can never grade itself: the policy needs a candidate.
    with pytest.raises(SystemExit):
        tool.main(["control", "--base-url", "http://fixture", "--model-name", "nonce",
                   "--model-dir", str(model), "--image", "fixture@sha256:" + "a" * 64,
                   "--contract", str(model.parent / "contract.json"),
                   "--campaign-id", "same-campaign", "--out", str(model.parent / "self.json"),
                   "--decision-policy", "no-new-failures"])


@pytest.mark.parametrize("broken", [False, True])
def test_real_measurement_cli_emits_opt_in_decision_and_preserves_exit_status(campaign, monkeypatch, broken):
    from tools import measure_boundary_control as tool

    model, _weight, run, post, _requests = campaign
    run("control.json")
    if broken:
        def candidate_post(url, body):
            if url.endswith("/tokenize"):
                return post(url, body)
            return _response(body["max_tokens"], text=ZERO)
        monkeypatch.setattr(bc.vqm, "_post_json", candidate_post)
    output = model.parent / "candidate.json"
    status = tool.main([
        "candidate", "--base-url", "http://fixture", "--model-name", "nonce",
        "--model-dir", str(model), "--image", "fixture@sha256:" + "a" * 64,
        "--control", str(model.parent / "control.json"), "--campaign-id", "same-campaign",
        "--out", str(output), "--decision-policy", "no-new-failures",
    ])
    assert status == (2 if broken else 0)
    arm = json.loads(output.read_text())
    assert arm["decision"]["verdict"] == ("refused" if broken else "accepted")
    assert arm["comparison"]["advisory_only"] is True
