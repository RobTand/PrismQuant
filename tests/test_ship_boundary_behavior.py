"""The sampled boundary-behavior ship gate (issue #87).

KL/PPL (distribution distance) and greedy-smoke (argmax agreement) are
structurally blind to boundary-token distribution defects that only manifest
under sampling: three DSV4-Flash quants within ~3% PPL spanned a 6x
behavioral gap (14/180 to 83/180) on the frozen battery. These tests pin the
added fifth check — sampled terse prompts scored mechanically for
`</think>` stutter/loop, zero-tag runaway, and cap-truncation — and the
shipcard replay that refuses a receipt without it.

The ledger key set is derived from the producer that owns it (the same shape
as `test_shipcard._producer_check_names`): a sixth check added to the
producer appears here on its own; a roster restated from today's five names
would not.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import prismaquant.validate_quantized_model as vqm

_REFUSED_URL = "http://127.0.0.1:1"


def _producer_check_names() -> frozenset:
    """The ship-gate ledger's key set, derived from the producer that owns it.

    Each check constructor names its own CheckResult without needing a live
    server: against a refused localhost port every probe fails fast and
    returns its (failing) verdict, whose name is what `run_validation` files
    under.
    """
    return frozenset({
        vqm.check_serve_ready(_REFUSED_URL).name,
        vqm.check_generation_sanity(
            _REFUSED_URL, "probe", vqm.DEFAULT_MIN_GEN_LEN).name,
        vqm.check_perplexity(
            _REFUSED_URL, "probe",
            vqm.DEFAULT_MAX_PPL, vqm.DEFAULT_MAX_P99_NLL,
            vqm.DEFAULT_MAX_MEAN_NLL).name,
        vqm.check_mtp_acceptance(
            _REFUSED_URL, vqm.DEFAULT_MIN_MTP_ACCEPT_P0).name,
        vqm.check_boundary_behavior(_REFUSED_URL, "probe").name,
    })


def _current_thresholds() -> dict:
    """The filed threshold contract, read off the producer's own defaults."""
    return {
        "max_ppl": vqm.DEFAULT_MAX_PPL,
        "max_mean_nll": vqm.DEFAULT_MAX_MEAN_NLL,
        "max_p99_nll": vqm.DEFAULT_MAX_P99_NLL,
        "min_gen_len": vqm.DEFAULT_MIN_GEN_LEN,
        "min_mtp_accept_p0": vqm.DEFAULT_MIN_MTP_ACCEPT_P0,
        "max_boundary_defects": vqm.DEFAULT_MAX_BOUNDARY_DEFECTS,
        "boundary_temperature": vqm.DEFAULT_BOUNDARY_TEMPERATURE,
        "boundary_max_tokens": vqm.DEFAULT_BOUNDARY_MAX_TOKENS,
        "boundary_reps": vqm.DEFAULT_BOUNDARY_REPS,
    }


# ---------------------------------------------------------------------------
# Pure scorer
# ---------------------------------------------------------------------------

def test_scorer_passes_a_clean_single_tag_completion():
    scored = vqm.score_boundary_text("<think>12</think> 12", "stop")
    assert scored["think_tag_count"] == 1
    assert scored["defects"] == []


def test_scorer_flags_zero_tag_runaway():
    scored = vqm.score_boundary_text("The answer is 12.", "stop")
    assert scored["think_tag_count"] == 0
    assert scored["defects"] == ["zero_tag"]


def test_scorer_flags_think_stutter():
    scored = vqm.score_boundary_text(
        "<think>12</think> <think>12</think> 12", "stop")
    assert scored["think_tag_count"] == 2
    assert scored["defects"] == ["think_stutter"]


def test_scorer_flags_cap_truncation():
    scored = vqm.score_boundary_text("<think>12</think> 12", "length")
    assert scored["defects"] == ["cap_truncation"]


def test_length_sanity_is_blind_to_stutter_the_boundary_scorer_catches():
    """The structural gap in one assertion: a long stuttering completion
    clears the old generation-sanity length rule (>= 30 chars) while the
    boundary scorer fails it. PPL/KL average the same near-tie into noise."""
    stutter = "<think>9</think> <think>9</think> 81 " * 4
    assert len(stutter) >= vqm.DEFAULT_MIN_GEN_LEN
    scored = vqm.score_boundary_text(stutter, "stop")
    assert scored["defects"] == ["think_stutter"]


# ---------------------------------------------------------------------------
# Live check against a scripted fake serve
# ---------------------------------------------------------------------------

class _BoundaryHandler(BaseHTTPRequestHandler):
    mode: str = "clean"  # "clean" | "broken"

    def log_message(self, *a, **kw):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            return
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"vllm:request_success_total 1.0\n")
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n))
        if self.mode == "clean":
            body = {"choices": [{"text": "<think>12</think> 12",
                                 "finish_reason": "stop"}]}
        else:
            # The broken shape from the issue: no clean </think> under
            # sampling (T10 failing 6/6 on the broken artifact).
            body = {"choices": [{"text": "12 12 12 12 12 12",
                                 "finish_reason": "stop"}]}
        out = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def boundary_server():
    _BoundaryHandler.mode = "clean"
    srv = HTTPServer(("127.0.0.1", 0), _BoundaryHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def test_boundary_check_passes_a_clean_serve(boundary_server):
    _BoundaryHandler.mode = "clean"
    r = vqm.check_boundary_behavior(
        boundary_server, "any", prompts=("9²",), reps=2)
    assert r.passed, r.detail
    assert r.metrics["n_generations"] == 2
    assert r.metrics["n_defects"] == 0


def test_boundary_check_fails_a_broken_serve(boundary_server):
    _BoundaryHandler.mode = "broken"
    r = vqm.check_boundary_behavior(
        boundary_server, "any", prompts=("9²",), reps=2)
    assert not r.passed
    assert r.metrics["n_defects"] == 2
    assert r.metrics["defects_by_kind"]["zero_tag"] == 2


def test_boundary_check_refuses_a_greedy_temperature():
    """Temp 0 is the argmax path where the boundary token still wins — the
    defect is invisible there by construction, so a temp-0 "sampled" check
    is the old blind gate wearing the new name."""
    r = vqm.check_boundary_behavior(
        _REFUSED_URL, "probe", temperature=0.0, prompts=("9²",), reps=1)
    assert not r.passed
    assert "not sampling" in r.detail


def test_boundary_check_samples_every_prompt_six_times(boundary_server):
    """The filed evidence shape: the full default battery is 5 prompts × 6
    reps = 30 sampled generations, the published battery's replication
    count — not a number tuned here."""
    _BoundaryHandler.mode = "clean"
    r = vqm.check_boundary_behavior(boundary_server, "any")
    assert r.passed, r.detail
    assert r.metrics["n_prompts"] == len(vqm.BOUNDARY_PROMPTS)
    assert r.metrics["reps"] == vqm.DEFAULT_BOUNDARY_REPS == 6
    assert r.metrics["n_generations"] == len(vqm.BOUNDARY_PROMPTS) * 6
    assert r.metrics["temperature"] == vqm.DEFAULT_BOUNDARY_TEMPERATURE > 0


def test_boundary_prompts_cover_the_issue_strata():
    """The three verbatim prompts from the issue must stay in the battery;
    anything else is a same-strata companion, never a replacement."""
    assert "144÷12" in vqm.BOUNDARY_PROMPTS
    assert "9²" in vqm.BOUNDARY_PROMPTS
    assert "How many legs does a spider have?" in vqm.BOUNDARY_PROMPTS


# ---------------------------------------------------------------------------
# Shipcard replay
# ---------------------------------------------------------------------------

_FAKE_FINGERPRINT = "f" * 64
_FAKE_COMMIT = "a" * 40


def _artifact(tmp_path, *, name="exported"):
    from prismaquant.shipcard import compute_model_sha

    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    return model_dir, compute_model_sha(model_dir)


def _ship_gate_record(model_sha, *, source, boundary_defects=0,
                      boundary_passed=True, drop_boundary=False):
    from prismaquant.shipcard import make_record

    ledger = {name: {"passed": True} for name in _producer_check_names()}
    ledger["perplexity"] = {
        "passed": True,
        "perplexity": 8.33,
        "mean_nll_per_tok": 2.12,
        "max_nll_per_tok": 4.50,
        "n_tokens": 8192,
        "spec_decode_detected": False,
    }
    ledger["boundary_behavior"] = {
        "passed": boundary_passed,
        "n_prompts": len(vqm.BOUNDARY_PROMPTS),
        "reps": vqm.DEFAULT_BOUNDARY_REPS,
        "n_generations": len(vqm.BOUNDARY_PROMPTS)
        * vqm.DEFAULT_BOUNDARY_REPS,
        "n_defects": boundary_defects,
        "max_defects": vqm.DEFAULT_MAX_BOUNDARY_DEFECTS,
        "temperature": vqm.DEFAULT_BOUNDARY_TEMPERATURE,
        "max_tokens": vqm.DEFAULT_BOUNDARY_MAX_TOKENS,
        "defects_by_kind": {"zero_tag": boundary_defects,
                            "think_stutter": 0, "cap_truncation": 0},
        "failing_examples": [],
    }
    if drop_boundary:
        del ledger["boundary_behavior"]
    return make_record(
        slot="ship_gate", tool="validate_quantized_model.py",
        passed=boundary_passed, model_sha=model_sha, metrics=ledger,
        detail="boundary battery verdict",
        spec_decode_detected=False, git_commit=_FAKE_COMMIT,
        extra={
            "base_url": "http://127.0.0.1:8000",
            "served_model_name": "probe-artifact",
            "thresholds": _current_thresholds(),
            "model_sha_source": source,
        },
    )


def _full_card(tmp_path, **record_kwargs):
    from prismaquant.shipcard import (
        REQUIRED_SLOTS, build_shipcard, fill_slot, make_record,
        write_shipcard,
    )

    model_dir, sha = _artifact(tmp_path)
    path = model_dir / "shipcard.json"
    write_shipcard(path, build_shipcard(model_dir, build={}))
    for slot in REQUIRED_SLOTS:
        if slot == "ship_gate":
            fill_slot(path, slot, _ship_gate_record(
                sha, source=str(model_dir), **record_kwargs))
        elif slot.startswith("native_export."):
            arm = slot.split(".")[1]
            fill_slot(path, slot, make_record(
                slot=slot, tool="validate_native_export.py", passed=True,
                model_sha=sha,
                metrics={"arm": arm, "enforce_eager": arm == "eager",
                         "generated_chars": 128, "max_new_tokens": 16},
                detail=f"{arm} smoke", git_commit=_FAKE_COMMIT))
        elif slot == "gold.kl":
            fill_slot(path, slot, make_record(
                slot=slot, tool="test", passed=True, model_sha=sha,
                spec_decode_detected=False,
                metrics={"kl_mean": 0.0151, "kl_confident_mean": 0.0143,
                         "n_positions": 4088, "n_samples": 8, "seqlen": 512,
                         "score_positions": "all"},
                serve_fingerprint=_FAKE_FINGERPRINT,
                git_commit=_FAKE_COMMIT))
        else:
            fill_slot(path, slot, make_record(
                slot=slot, tool="test", passed=True, model_sha=sha,
                spec_decode_detected=False,
                metrics={"ppl": 8.33, "mean_nll": 2.12,
                         "n_tokens_scored": 8192},
                serve_fingerprint=_FAKE_FINGERPRINT,
                git_commit=_FAKE_COMMIT))
    return path, model_dir


def test_producer_names_boundary_behavior_as_its_own_check():
    """The fix is a check the producer owns, not a roster restated here:
    membership, not equality — a sixth check must not require a test edit."""
    assert "boundary_behavior" in _producer_check_names()


def test_verify_refuses_a_ledger_missing_boundary(tmp_path):
    """The pre-fix shape: four passing checks, no boundary evidence. It
    verified clean before the gate existed; it must not now."""
    from prismaquant.shipcard import load_shipcard, verify

    path, model_dir = _full_card(tmp_path, drop_boundary=True)
    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any("ledger is incomplete" in p for p in problems), problems


def test_verify_refuses_boundary_defects_over_zero(tmp_path):
    """A receipt that sampled defects and passed anyway is exactly the
    certified-broken artifact from the issue."""
    from prismaquant.shipcard import load_shipcard, verify

    path, model_dir = _full_card(
        tmp_path, boundary_defects=3, boundary_passed=True)
    problems = verify(load_shipcard(path), model_dir=model_dir)
    assert any("boundary check reports 3 defective" in p for p in problems), \
        problems


def test_verify_accepts_a_clean_boundary_receipt(tmp_path):
    from prismaquant.shipcard import load_shipcard, verify

    path, model_dir = _full_card(tmp_path)
    assert verify(load_shipcard(path), model_dir=model_dir) == []
