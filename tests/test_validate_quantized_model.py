"""Tests for the pre-ship quality validator.

Goal: pin the bimodal-failure detection logic. The 27B shipped with
mean NLL below threshold but max per-prompt avg NLL was ~9 —
threshold catches it.

Uses an inline HTTPServer so we can script fake vLLM responses and
assert both pass- and fail-paths end-to-end without a real 35B load.
"""
from __future__ import annotations

import json
import math
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import prismaquant.validate_quantized_model as vqm
from prismaquant.validate_quantized_model import (
    check_generation_sanity,
    check_perplexity,
    check_mtp_acceptance,
    run_validation,
    EVAL_PROMPTS,
)


class _FakeVLLMHandler(BaseHTTPRequestHandler):
    # Class-level state so the test can configure per-test.
    mode: str = "healthy"       # "healthy" | "bimodal" | "one_outlier" | ...
    metrics_payload: str = ""
    requests: list[dict] = []

    def log_message(self, *a, **kw):
        pass  # silence

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            return
        if self.path == "/metrics":
            payload = self.metrics_payload.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        req = json.loads(raw)
        self.requests.append(req)
        prompt = req.get("prompt", "")

        if self.path == "/v1/completions":
            mt = int(req.get("max_tokens") or 1)
            echo = bool(req.get("echo", False))
            want_logprobs = req.get("logprobs") is not None

            if echo and want_logprobs:
                # Fake logprobs: pretend every token had logprob -1.5
                # (healthy) or -10 on 80% of prompts (bimodal broken).
                n_tokens = max(1, min(64, len(prompt.split())))
                if self.mode == "healthy":
                    per_tok = -1.5
                elif self.mode == "bimodal":
                    # 2 of every 10 prompts get healthy logprobs, rest broken
                    idx = EVAL_PROMPTS.index(prompt) if prompt in EVAL_PROMPTS else 0
                    per_tok = -1.5 if idx in (3, 7) else -9.5
                elif self.mode == "one_outlier":
                    idx = EVAL_PROMPTS.index(prompt) if prompt in EVAL_PROMPTS else 0
                    per_tok = -20.0 if idx == 0 else -1.5
                elif self.mode == "broken":
                    per_tok = -12.0
                else:
                    per_tok = float("nan")
                token_logprobs = [None] + [per_tok] * n_tokens
                body = {
                    "choices": [{
                        "text": "",
                        "logprobs": {"token_logprobs": token_logprobs},
                    }]
                }
            else:
                # Plain generation request: return a short stub completion.
                # Boundary-stressing prompts get a clean boundary shape
                # (exactly one </think>, finish stop) so the healthy arm is
                # healthy on every check; anything else gets generic prose.
                if prompt in vqm.BOUNDARY_PROMPTS:
                    body = {"choices": [{"text": "<think>12</think> 12",
                                         "finish_reason": "stop"}]}
                else:
                    txt = ("Pretend this is a coherent completion about the topic "
                           "that goes on for enough characters.")
                    body = {"choices": [{"text": txt}]}

            out = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)
            return
        self.send_response(404)
        self.end_headers()


@pytest.fixture
def fake_server():
    _FakeVLLMHandler.mode = "healthy"
    _FakeVLLMHandler.metrics_payload = ""
    _FakeVLLMHandler.requests = []
    srv = HTTPServer(("127.0.0.1", 0), _FakeVLLMHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


# ----------------------------------------------------------------
# Perplexity check — bimodal failure detection
# ----------------------------------------------------------------

def test_perplexity_healthy_passes(fake_server):
    _FakeVLLMHandler.mode = "healthy"
    r = check_perplexity(fake_server, "any", max_ppl=25, max_p99_nll=6, max_mean_nll=3)
    assert r.passed, f"healthy artifact should pass: {r.detail}"
    assert r.metrics["perplexity"] < 25


def test_perplexity_catches_uniformly_broken(fake_server):
    _FakeVLLMHandler.mode = "broken"
    r = check_perplexity(fake_server, "any", max_ppl=25, max_p99_nll=6, max_mean_nll=3)
    assert not r.passed
    # Any of the threshold violations is acceptable reason — but expect mean.
    assert "mean_nll" in r.detail or "ppl" in r.detail


def test_perplexity_catches_bimodal_broken(fake_server):
    """The 27B failure mode: 2/10 prompts normal, rest catastrophic.
    Mean NLL can fall under threshold but the max per-prompt avg NLL
    must flag the bimodality."""
    _FakeVLLMHandler.mode = "bimodal"
    r = check_perplexity(fake_server, "any", max_ppl=1000, max_p99_nll=6,
                         max_mean_nll=100)
    assert not r.passed, (
        f"bimodal failure must trip tail NLL even when mean / ppl slack: {r.detail}"
    )
    assert "bimodal" in r.detail or "p99" in r.detail or "max(per-prompt" in r.detail


def test_perplexity_generous_thresholds_passes_bimodal(fake_server):
    """If thresholds are deliberately generous, bimodal data passes
    through. Confirms threshold wiring works both directions."""
    _FakeVLLMHandler.mode = "bimodal"
    r = check_perplexity(fake_server, "any", max_ppl=1e9, max_p99_nll=100,
                         max_mean_nll=100)
    assert r.passed


def test_perplexity_reports_true_p99_and_max_tail_nll(fake_server):
    _FakeVLLMHandler.mode = "one_outlier"
    r = check_perplexity(fake_server, "any", max_ppl=1e9, max_p99_nll=100,
                         max_mean_nll=100)

    assert r.passed
    assert r.metrics["max_nll_per_tok"] == pytest.approx(20.0)
    assert r.metrics["p99_nll_per_tok"] < r.metrics["max_nll_per_tok"]
    assert r.metrics["p99_nll_per_tok"] > r.metrics["mean_nll_per_tok"]


# ----------------------------------------------------------------
# Generation sanity
# ----------------------------------------------------------------

def test_generation_sanity_passes_when_outputs_long_enough(fake_server):
    _FakeVLLMHandler.mode = "healthy"
    r = check_generation_sanity(fake_server, "any", min_gen_len=30)
    assert r.passed


# ----------------------------------------------------------------
# MTP acceptance
# ----------------------------------------------------------------

def test_mtp_acceptance_passes_above_threshold(fake_server):
    _FakeVLLMHandler.metrics_payload = (
        '# HELP vllm:spec_decode_num_drafts_total\n'
        'vllm:spec_decode_num_drafts_total{engine="0"} 100.0\n'
        '# HELP vllm:spec_decode_num_accepted_tokens_per_pos_total\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 80.0\n'
    )
    r = check_mtp_acceptance(fake_server, min_p0=0.6)
    assert r.passed
    assert r.metrics["accept_rate_p0"] == 0.8


def test_mtp_acceptance_fails_below_threshold(fake_server):
    _FakeVLLMHandler.metrics_payload = (
        'vllm:spec_decode_num_drafts_total{engine="0"} 100.0\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 30.0\n'
    )
    r = check_mtp_acceptance(fake_server, min_p0=0.6)
    assert not r.passed


def test_mtp_acceptance_skipped_when_no_drafts(fake_server):
    """Model without spec-decode should pass with 'skipped' rather
    than fail — the validator shouldn't force spec-decode on every run."""
    _FakeVLLMHandler.metrics_payload = (
        'vllm:spec_decode_num_drafts_total{engine="0"} 0.0\n'
    )
    r = check_mtp_acceptance(fake_server, min_p0=0.6)
    assert r.passed
    assert "skipping" in r.detail.lower()


# ----------------------------------------------------------------
# Spec-decode detection (prevents the "validator returns draft-model
# NLL and false-fails a healthy model" trap).
# ----------------------------------------------------------------

def test_perplexity_refuses_to_run_when_spec_decode_detected(fake_server):
    """If spec-decode is on, /v1/completions echo+logprobs returns
    draft-model logprobs, not target. The validator must refuse to
    produce a verdict on those numbers — it must fail with a clear
    instruction rather than silently mis-reporting."""
    # Healthy logprobs under the hood — the validator should NOT even
    # consult them, because spec-decode detection short-circuits.
    _FakeVLLMHandler.mode = "healthy"
    _FakeVLLMHandler.metrics_payload = (
        '# HELP vllm:spec_decode_num_drafts_total ...\n'
        'vllm:spec_decode_num_drafts_total{engine="0"} 42.0\n'
    )
    r = check_perplexity(fake_server, "any", max_ppl=25, max_p99_nll=6,
                         max_mean_nll=3)
    assert not r.passed, "spec-decode on must fail the perplexity check"
    assert "spec-decode" in r.detail.lower() or "speculative" in r.detail.lower()
    assert r.metrics.get("spec_decode_detected") is True


def test_perplexity_runs_normally_without_spec_decode(fake_server):
    """Sanity: when /metrics has no vllm:spec_decode_* entries, the
    perplexity check runs its usual logic (healthy data passes)."""
    _FakeVLLMHandler.mode = "healthy"
    _FakeVLLMHandler.metrics_payload = (
        '# some other vllm metric\n'
        'vllm:request_success_total 10.0\n'
    )
    r = check_perplexity(fake_server, "any", max_ppl=25, max_p99_nll=6,
                         max_mean_nll=3)
    assert r.passed, f"healthy+no-spec-decode should pass: {r.detail}"


def test_perplexity_bos_token_disables_server_special_tokens(fake_server):
    _FakeVLLMHandler.mode = "healthy"
    _FakeVLLMHandler.metrics_payload = ""

    r = check_perplexity(
        fake_server,
        "any",
        max_ppl=25,
        max_p99_nll=6,
        max_mean_nll=3,
        bos_token="<bos>",
        add_special_tokens=True,
    )

    assert r.passed
    echo_reqs = [req for req in _FakeVLLMHandler.requests if req.get("echo")]
    assert len(echo_reqs) == len(EVAL_PROMPTS)
    assert all(req["prompt"].startswith("<bos>") for req in echo_reqs)
    assert {req["add_special_tokens"] for req in echo_reqs} == {False}


def test_perplexity_refuses_even_when_metrics_mostly_empty(fake_server):
    """Just the /metrics endpoint containing a single spec_decode
    counter is enough to suspend the check — don't require non-zero
    values or any other context."""
    _FakeVLLMHandler.mode = "healthy"
    _FakeVLLMHandler.metrics_payload = (
        'vllm:spec_decode_num_drafts_created{engine="0"} 1776869756.0\n'
    )
    r = check_perplexity(fake_server, "any", max_ppl=25, max_p99_nll=6,
                         max_mean_nll=3)
    assert not r.passed


# ----------------------------------------------------------------
# End-to-end
# ----------------------------------------------------------------

def test_end_to_end_healthy_report_no_spec_decode(fake_server):
    """The no-spec-decode pass: perplexity check runs + passes,
    mtp_acceptance is cleanly skipped (no drafts recorded)."""
    _FakeVLLMHandler.mode = "healthy"
    _FakeVLLMHandler.metrics_payload = (
        '# healthy serve without spec-decode\n'
        'vllm:request_success_total 0.0\n'
    )
    rep = run_validation(fake_server, "any", wait_seconds=5)
    assert rep.passed
    names = [c.name for c in rep.checks]
    assert "perplexity" in names and "mtp_acceptance" in names
    # Spec-decode check skipped cleanly
    mtp = next(c for c in rep.checks if c.name == "mtp_acceptance")
    assert mtp.passed and "skipping" in mtp.detail.lower()


def test_run_validation_forwards_bos_controls(fake_server):
    _FakeVLLMHandler.mode = "healthy"
    _FakeVLLMHandler.metrics_payload = (
        '# healthy serve without spec-decode\n'
        'vllm:request_success_total 0.0\n'
    )

    rep = run_validation(
        fake_server,
        "any",
        wait_seconds=5,
        bos_token="<bos>",
        add_special_tokens=True,
    )

    assert rep.passed
    assert rep.thresholds["bos_token"] == "<bos>"
    echo_reqs = [req for req in _FakeVLLMHandler.requests if req.get("echo")]
    assert echo_reqs
    assert all(req["prompt"].startswith("<bos>") for req in echo_reqs)
    assert {req["add_special_tokens"] for req in echo_reqs} == {False}


def test_cli_exposes_bos_controls(fake_server, monkeypatch, capsys):
    _FakeVLLMHandler.mode = "healthy"
    _FakeVLLMHandler.metrics_payload = (
        '# healthy serve without spec-decode\n'
        'vllm:request_success_total 0.0\n'
    )
    monkeypatch.setattr(sys, "argv", [
        "validate_quantized_model",
        "--base-url", fake_server,
        "--model-name", "any",
        "--wait-seconds", "5",
        "--bos-token", "<bos>",
        "--no-add-special-tokens",
    ])

    assert vqm.main() == 0
    capsys.readouterr()
    echo_reqs = [req for req in _FakeVLLMHandler.requests if req.get("echo")]
    assert echo_reqs
    assert all(req["prompt"].startswith("<bos>") for req in echo_reqs)
    assert {req["add_special_tokens"] for req in echo_reqs} == {False}


def test_end_to_end_spec_decode_forces_perplexity_skip(fake_server):
    """The spec-decode pass: perplexity check refuses with a
    diagnostic (it'd return draft-model NLL), but mtp_acceptance
    runs normally. Caller is expected to have run the no-spec-decode
    pass separately for perplexity."""
    _FakeVLLMHandler.mode = "healthy"
    _FakeVLLMHandler.metrics_payload = (
        'vllm:spec_decode_num_drafts_total{engine="0"} 100.0\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 85.0\n'
    )
    rep = run_validation(fake_server, "any", wait_seconds=5)
    # Perplexity SHOULD fail because spec-decode was detected.
    assert not rep.passed
    ppl = next(c for c in rep.checks if c.name == "perplexity")
    assert not ppl.passed
    assert ppl.metrics.get("spec_decode_detected") is True
    # MTP acceptance still runs and passes.
    mtp = next(c for c in rep.checks if c.name == "mtp_acceptance")
    assert mtp.passed


def test_end_to_end_bimodal_fails(fake_server):
    _FakeVLLMHandler.mode = "bimodal"
    _FakeVLLMHandler.metrics_payload = ""
    rep = run_validation(fake_server, "any", wait_seconds=5)
    assert not rep.passed
    ppl_check = next(c for c in rep.checks if c.name == "perplexity")
    assert not ppl_check.passed


# ----------------------------------------------------------------
# base_url spelling — the 2026-08-14 ship-gate hang
# ----------------------------------------------------------------
# Every test above passes the BARE SERVER ROOT, which is why none of them
# caught this: the CLI default is also bare, but the compressed_tensors lane
# spec published `http://127.0.0.1:8000/v1`, and that spelling made the gate
# poll `/v1/health` (404) for its entire 900 s timeout without ever sending a
# prompt. The bug lived in the gap between the two conventions, so these tests
# drive the `/v1` spelling specifically.
def test_server_root_strips_only_a_trailing_v1():
    assert vqm._server_root("http://h:8000/v1") == "http://h:8000"
    assert vqm._server_root("http://h:8000/v1/") == "http://h:8000"
    assert vqm._server_root("http://h:8000") == "http://h:8000"
    # A serve behind --root-path keeps its prefix; only the API segment goes.
    assert vqm._server_root("http://h:8000/foo/v1") == "http://h:8000/foo"
    # Not a trailing segment => untouched (no substring mangling).
    assert vqm._server_root("http://h:8000/v1beta") == "http://h:8000/v1beta"


def test_openai_root_spelling_still_reaches_health_and_completions(fake_server):
    """`--base-url .../v1` must work, not hang."""
    _FakeVLLMHandler.mode = "healthy"
    _FakeVLLMHandler.metrics_payload = ""
    assert vqm._health_ok(f"{fake_server}/v1") is True
    rep = run_validation(f"{fake_server}/v1", "any", wait_seconds=5)
    assert rep.passed, [c.detail for c in rep.checks if not c.passed]
    # It normalized rather than merely tolerating: no /v1/v1/... was issued.
    assert rep.base_url == fake_server


def test_spec_decode_guard_fails_closed_when_metrics_unreachable():
    """An unreadable /metrics must REFUSE, never read as 'spec-decode off'.

    This guard exists to stop the DRAFT model's NLL being published as the
    target's. Returning False on a fetch error made the one check that
    prevents a silent mis-report issue a confident all-clear.
    """
    dead = "http://127.0.0.1:9"  # discard port: connections refused
    with pytest.raises(vqm.SpecDecodeUndetermined):
        vqm._spec_decode_on(dead)
    res = check_perplexity(dead, "any", max_ppl=100.0, max_p99_nll=100.0,
                           max_mean_nll=100.0)
    assert not res.passed
    assert res.metrics.get("spec_decode_detected") is None
    assert "refusing" in res.detail
