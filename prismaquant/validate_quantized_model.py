"""Pre-ship quality validator for PrismaQuant artifacts.

Designed to catch the class of failure that shipped a broken 27B
checkpoint to HF in this session: predicted Δloss said the artifact
was *better* than its predecessor (13.5% lower), but the actual
model produced ~10,000× worse perplexity because the allocator's
fused-sibling sum-aggregation under-weighted asymmetric sensitivity.

The predicted-Δloss heuristic is not enough. Every artifact must
pass *measured* quality gates before upload.

Checks, in order:

  1. **Serve check** — vLLM actually starts the model (load, MTP
     wrapper, CUDA graph capture) with the recipe's flags.
   2. **Generation sanity** — small set of prompts must produce
      coherent outputs. Filters obvious catastrophic breakage
      (NaN/repetition loops/nonsense) before wasting on stats.
   3. **Boundary behavior** — sampled (temperature > 0) generations over
      terse boundary-stressing prompts (`BOUNDARY_PROMPTS`), scored
      mechanically for `</think>` stutter/loop, zero-tag runaway, and
      cap-truncation-before-answer. The chat endpoint is mandatory: raw
      completions do not apply the model's reasoning template. Zero defects
      under 64 tokens remains a fail-closed historical default pending #87's
      paired-control policy; it is not a calibrated universal claim. This
      is the axis KL/PPL (distribution distance) and greedy-smoke (argmax
      agreement) cannot see at any threshold: three DSV4-Flash quants
      within ~3% PPL spanned a 6x behavioral gap (14/180 to 83/180).
   4. **Perplexity / NLL** — logprobs over a diverse held-out
      prompt suite. Hard thresholds: `ppl < MAX_PPL` and
      worst per-prompt NLL < `MAX_P99_NLL` (legacy flag name).
      The worst-prompt guard catches the 27B failure mode where
      80% of prompts scored NLL~10 while 2/10 scored normally.
   5. **MTP acceptance** — if spec-decode is on, per-position
      acceptance > `MIN_MTP_ACCEPT_P0` at position 0.

Use from CI or pre-ship hook:

    python -m prismaquant.validate_quantized_model \\
        --artifact /path/to/exported \\
        --baseline rdtand/<previous-known-good>     # optional
        --report /path/to/report.md

Exit 0 = all checks passed. Exit 1 = at least one check failed
(prints a report to stdout + writes `--report` markdown).

**Workflow for artifacts with speculative decoding (MTP / Eagle):**
run the validator twice, against two different serves.

  1. Serve WITHOUT `--speculative-config`, run validator → the
     perplexity check produces target-model NLL and a meaningful
     verdict. MTP acceptance is skipped because no drafts fire.

  2. Re-serve WITH `--speculative-config`, run validator → the
     perplexity check will refuse to run (it detects spec-decode
     via /metrics and fails with a diagnostic rather than silently
     return draft-model NLL). MTP acceptance runs and reports
     position-0 accept rate.

The model is "ship-ready" only if both passes succeed. This is
awkward but honest: vLLM offers no target-model logprob path
while spec-decode is active, and faking perplexity from the draft
model has already burned one false-FAIL in the session that spawned
this file.

Design notes:
  - We use vLLM's OpenAI API for logprobs via /v1/completions
    with `echo=True`, which echoes the prompt tokens with their
    prior logprobs.
  - Model serving is run in a subprocess / docker exec so we
    can tear it down cleanly between runs.
  - Prompts span science, history, CS, economics, everyday prose
    to catch domain-specific failure. Short enough to keep a full
    suite under ~30s at spec-decode speed.
  - Thresholds are calibrated: MAX_PPL=25 catches catastrophic
    breakage but tolerates normal 4-bit quant degradation
    (BF16 baseline ~3-5, 4-bit ~4-8). MAX_P99_NLL=6 is ~2σ above
    BF16 average; the implementation also reports the actual p99
    separately from the max prompt NLL.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field, asdict


# -----------------------------------------------------------------
# Prompt suite — diverse, held-out, deliberately off the beaten path
# -----------------------------------------------------------------
EVAL_PROMPTS: list[str] = [
    "The mitochondrion is an organelle found in most eukaryotic cells. It is the site of cellular respiration and ATP production. Mitochondria have their own DNA and are thought to have originated from ancient prokaryotes.",
    "In modern cryptography, a hash function maps data of arbitrary size to a fixed-size output. Good cryptographic hashes are deterministic, collision-resistant, and exhibit the avalanche property: small input changes produce large output changes.",
    "The French Revolution began in 1789 and ended in the late 1790s. It fundamentally reshaped European political thought, ending the monarchy in France and introducing ideals of liberty, equality, and fraternity that would influence democratic movements for centuries.",
    "A binary search tree is a data structure in which each node has at most two children, and for every node the left subtree contains keys less than the node's key and the right subtree contains keys greater. Lookups, insertions, and deletions take O(log n) on average.",
    "Keynesian economics argues that aggregate demand drives economic output, especially during recessions. Governments can stimulate demand through fiscal policy — spending more or cutting taxes — when private consumption and investment fall short.",
    "Photosynthesis converts light energy, primarily from the sun, into chemical energy stored in glucose. It occurs mainly in the chloroplasts of plant cells, where chlorophyll absorbs photons and drives the reduction of carbon dioxide into carbohydrates.",
    "In compilers, a lexer breaks source code into tokens, and a parser groups tokens into a syntax tree according to grammar rules. Semantic analysis then annotates the tree with type and scope information, ready for code generation or interpretation.",
    "The Great Pyramid of Giza was built around 2560 BC as a tomb for the Fourth Dynasty Egyptian pharaoh Khufu. It held the record for the tallest man-made structure for nearly 4000 years, until the completion of Lincoln Cathedral in 1311.",
    "Operating systems manage hardware resources and provide services to applications. Key abstractions include processes, virtual memory, file systems, and a scheduler that decides which process runs when on the available CPU cores.",
    "Neural networks are loosely inspired by biological neurons. In a feedforward network, input activations pass through layers of weighted sums and nonlinear functions. Training uses backpropagation to compute gradients of a loss with respect to each weight.",
    "A sauce Bechamel is one of the five French mother sauces. Its base is a roux of butter and flour cooked gently, to which warm milk is gradually whisked in until the mixture thickens into a smooth white sauce seasoned with salt, nutmeg, and pepper.",
    "The theory of plate tectonics explains the movement of large sections of Earth's lithosphere. Plates move atop the semi-fluid asthenosphere, driven by convection currents in the mantle. Their interactions cause earthquakes, volcanoes, and mountain building.",
]

# Generation-sanity prompts — short, expect coherent short continuation.
# Unlike the perplexity prompts, we actually SAMPLE here (non-zero temp
# at small max_tokens) and only assert the output looks like English.
GEN_PROMPTS: list[str] = [
    "The best approach to learning a new programming language is",
    "When cooking rice in a pot, the most common mistake is",
    "A sensible way to explain gravitational time dilation to a non-physicist is",
    "The single most important fact about photosynthesis is",
]

# Boundary-behavior prompts — terse, boundary-token-stressing inputs scored
# under SAMPLING for `</think>` stutter/loop, zero-tag runaway, and
# cap-truncation-before-answer (issue #87).
#
# Why this gate exists: quantized DSV4-Flash artifacts stutter or fail to emit
# a clean `</think>` on ultra-short numeric prompts under sampling while the
# answer stays correct. Greedy takes the argmax path where the boundary token
# still wins, and KL/PPL average a per-token near-tie at one boundary position
# into noise — so no argmax-agreement or distributional-distance gate can see
# the defect at any threshold, while under a token cap the model never reaches
# its answer. Three independently built DSV4-Flash quants sat within ~3% PPL
# of each other while the behavioral battery spanned 14/180 → 83/180.
#
# Strata (proposal: ultra-short numeric, terse QA, short recall). The first
# three prompts are verbatim from the issue report (`144÷12`, `9²`, and the
# spider-legs terse QA that zero-tag-ran 5/6 on the broken artifact); the last
# two are same-strata companions (one more ultra-short numeric, one short
# recall) so each stratum is exercised more than once.
BOUNDARY_PROMPTS: tuple[str, ...] = (
    "144÷12",
    "9²",
    "How many legs does a spider have?",
    "What is 7×8?",
    "Name the capital of France.",
)

#: Closed defect vocabulary for one sampled generation. `zero_tag` (no
#: `</think>` emitted — the runaway shape), `think_stutter` (more than one
#: `</think>` — the stutter/loop shape), and `cap_truncation` (the server
#: stopped on `length`). A cap hit describes censored output; healthy thinking
#: models can also exhaust the historical cap, so it alone does not establish
#: an artifact defect (issue #87).
BOUNDARY_DEFECTS: tuple[str, ...] = (
    "zero_tag",
    "think_stutter",
    "cap_truncation",
)

THINK_CLOSE_TAG = "</think>"

# The boundary gate must exercise the model's chat template.  Raw
# ``/v1/completions`` continues the literal user string and therefore never
# enters a thinking model's reasoning scaffold.  These values are also filed
# in the shipcard metrics so offline replay can reject a clean-looking count
# produced under the old, structurally blind request.
BOUNDARY_ENDPOINT = "/v1/chat/completions"
BOUNDARY_REQUEST_SCHEMA = "prismaquant.boundary_chat_request/1"
BOUNDARY_RESPONSE_SCHEMA = "prismaquant.boundary_chat_response/1"
BOUNDARY_CHAT_TEMPLATE_KWARGS = {
    "thinking": True,
    "enable_thinking": True,
}


# -----------------------------------------------------------------
# Default thresholds (tune via CLI if needed)
# -----------------------------------------------------------------
DEFAULT_MAX_PPL = 25.0
DEFAULT_MAX_P99_NLL = 6.0
DEFAULT_MAX_MEAN_NLL = 3.0
DEFAULT_MIN_GEN_LEN = 30               # chars in each generated completion
DEFAULT_MIN_MTP_ACCEPT_P0 = 0.60       # position-0 accept fraction
# Boundary-behavior gate (issue #87).  Temperature 1.0 is the unmodified
# distribution and `REPS` 6 is the published battery's own replication count.
# The zero bound and 64-token cap are retained as fail-closed historical
# defaults only while the paired-control policy is unresolved: a first real
# endpoint audit found healthy DSV4 at 7/30 defects under 64 tokens and stock
# Qwen3-8B at 10/15 even under 600.  They are not calibrated universal claims
# and must not be used to close #87 or promote an artifact without the pending
# same-session control decision.
DEFAULT_MAX_BOUNDARY_DEFECTS = 0
DEFAULT_BOUNDARY_TEMPERATURE = 1.0
DEFAULT_BOUNDARY_MAX_TOKENS = 64
DEFAULT_BOUNDARY_REPS = 6


# -----------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------
class SpecDecodeUndetermined(RuntimeError):
    """The spec-decode guard could not read /metrics.

    Distinct from "spec-decode is off": the perplexity check refuses on this
    rather than proceeding, because the failure mode it guards against
    (publishing the DRAFT model's NLL as the target's) is silent.
    """


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    artifact: str
    base_url: str
    model_name: str
    thresholds: dict
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


# -----------------------------------------------------------------
# HTTP helpers (no extra deps — urllib is stdlib)
# -----------------------------------------------------------------
def _post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _server_root(base_url: str) -> str:
    """Server root for the operational endpoints, from the OpenAI API root.

    `/health` and `/metrics` are mounted at the SERVER root; the completions
    endpoints live under `/v1`.  ``--base-url`` names the latter, so appending
    `/health` to it yields `/v1/health`, which vLLM answers with 404.

    Measured 2026-08-14 on the Qwen3.8-27B ship gate: `wait_for_ready` polled
    `/v1/health` for 11 minutes and would have timed out at 900 s without
    sending a single prompt, while `/metrics` failed the same way and made the
    spec-decode guard fail OPEN (see `_spec_decode_on`).

    Only a trailing `/v1` is stripped, so a serve behind `--root-path /foo`
    (`http://h:8000/foo/v1`) resolves to `http://h:8000/foo` rather than being
    flattened to the bare host.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def _health_ok(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(
            f"{_server_root(base_url)}/health", timeout=5.0
        ) as r:
            return r.status == 200
    except Exception:
        return False


def _spec_decode_on(base_url: str) -> bool:
    """True iff the vLLM serve was launched with --speculative-config.

    Detection: vLLM registers the `vllm:spec_decode_*` Prometheus
    counters + gauges at startup whenever spec-decode is configured,
    even before any drafts run. Their literal presence in /metrics is
    a config-time signal.

    Critical for the perplexity check: with spec-decode on, vLLM
    routes /v1/completions echo+logprobs through the DRAFT model, so
    the NLL values returned are the 1-layer MTP head's logprobs,
    NOT the target model's. Those are not usable for target-model
    perplexity measurement. Detecting the condition lets the
    validator refuse to silently mis-report.

    THIS GUARD FAILS CLOSED.  It used to swallow every fetch error and
    return False, which reads as "spec-decode is off" — so an unreachable
    `/metrics` produced a confident all-clear from the one check whose job is
    to stop a draft-model NLL being published.  Until 2026-08-14 the URL was
    also wrong (`/v1/metrics`, 404), so on the standard `--base-url .../v1`
    invocation the guard could never fire at all.  An indeterminate answer is
    now an exception, not a False.
    """
    try:
        text = _get_text(f"{_server_root(base_url)}/metrics")
    except Exception as exc:  # noqa: BLE001 - re-raised as a refusal below
        raise SpecDecodeUndetermined(
            f"cannot read {_server_root(base_url)}/metrics ({exc}); refusing "
            "to certify perplexity, because a guard that cannot see the "
            "server must not report 'no spec-decode'"
        ) from exc
    return "vllm:spec_decode" in text


def wait_for_ready(base_url: str, max_seconds: float = 900.0,
                   poll_interval: float = 5.0) -> bool:
    """Block until the vLLM server responds to /health with 200."""
    t0 = time.time()
    while time.time() - t0 < max_seconds:
        if _health_ok(base_url):
            return True
        time.sleep(poll_interval)
    return False


# -----------------------------------------------------------------
# Individual checks
# -----------------------------------------------------------------
def check_serve_ready(base_url: str) -> CheckResult:
    ok = _health_ok(base_url)
    return CheckResult(
        name="serve_ready",
        passed=ok,
        detail="/health returned 200" if ok else "/health did NOT return 200 (server not up)",
    )


def check_generation_sanity(base_url: str, model_name: str,
                            min_gen_len: int) -> CheckResult:
    """Sample a few short completions — fail if any are empty or
    clearly-not-English. Catches crashes that return empty responses,
    and gross breakage that returns pure noise."""
    short_outputs = []
    for i, prompt in enumerate(GEN_PROMPTS, 1):
        try:
            r = _post_json(
                f"{base_url}/v1/completions",
                {
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": 40,
                    "temperature": 0.3,
                    "top_p": 0.95,
                },
            )
            text = (r["choices"][0].get("text") or "").strip()
        except Exception as e:
            return CheckResult(
                name="generation_sanity",
                passed=False,
                detail=f"request {i} failed: {type(e).__name__}: {e}",
            )
        if len(text) < min_gen_len:
            short_outputs.append((i, len(text), text))
    if short_outputs:
        return CheckResult(
            name="generation_sanity",
            passed=False,
            detail=(f"{len(short_outputs)}/{len(GEN_PROMPTS)} completions "
                    f"shorter than {min_gen_len} chars: "
                    f"{[(i, n) for i, n, _ in short_outputs]}"),
            metrics={"short_outputs": short_outputs},
        )
    return CheckResult(
        name="generation_sanity",
        passed=True,
        detail=f"all {len(GEN_PROMPTS)} completions ≥ {min_gen_len} chars",
    )


def score_boundary_text(
    text: str,
    finish_reason: str | None = None,
) -> dict:
    """Score one sampled generation for boundary-token defects (pure).

    Stdlib-only and server-free, so the gate's decision rule is unit-testable
    without a serve: the same function the live check calls is what the tests
    pin. Returns `{"think_tag_count": int, "defects": [...]}` with defects
    drawn from :data:`BOUNDARY_DEFECTS`.
    """
    body = text if isinstance(text, str) else ""
    count = body.count(THINK_CLOSE_TAG)
    defects: list[str] = []
    if count == 0:
        defects.append("zero_tag")
    elif count > 1:
        defects.append("think_stutter")
    if finish_reason == "length":
        defects.append("cap_truncation")
    return {"think_tag_count": count, "defects": defects}


def _boundary_text_from_chat_choice(choice: Mapping) -> tuple[str, str]:
    """Recover boundary semantics from one chat-completion choice.

    vLLM without a reasoning parser returns the raw generated text in
    ``message.content``.  With a parser, it consumes the first ``</think>`` and
    returns the two sides as ``message.reasoning`` (current spelling) or
    ``message.reasoning_content`` (older OpenAI-compatible spelling).  In the
    structured case we synthesize exactly that one consumed delimiter only
    when both reasoning-side and answer-side content are non-empty.  A second
    delimiter remains in content and is still scored as stutter; an empty
    side remains zero-tag/cap-truncation rather than receiving an invented
    close token.

    The accepted shapes are deliberately closed.  An ambiguous pair of
    non-null reasoning fields or a non-string field is a malformed response,
    not a clean generation.
    """
    if not isinstance(choice, Mapping):
        raise TypeError("chat choice is not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise TypeError("chat choice is missing message object")

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise TypeError("chat message.content is not a string or null")

    structured: list[tuple[str, str]] = []
    for field_name in ("reasoning", "reasoning_content"):
        value = message.get(field_name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(
                f"chat message.{field_name} is not a string or null")
        structured.append((field_name, value))
    if len(structured) > 1:
        raise ValueError(
            "chat message carries both reasoning and reasoning_content")

    if structured:
        field_name, reasoning = structured[0]
        if reasoning and content:
            return reasoning + THINK_CLOSE_TAG + content, field_name
        if not reasoning:
            return reasoning, f"{field_name}_empty"
        return reasoning, f"{field_name}_without_content"
    if content is None:
        raise TypeError(
            "chat message has neither content nor structured reasoning")
    return content, "content"


def check_boundary_behavior(
    base_url: str,
    model_name: str,
    max_defects: int = DEFAULT_MAX_BOUNDARY_DEFECTS,
    *,
    temperature: float = DEFAULT_BOUNDARY_TEMPERATURE,
    max_tokens: int = DEFAULT_BOUNDARY_MAX_TOKENS,
    reps: int = DEFAULT_BOUNDARY_REPS,
    prompts: tuple[str, ...] | list[str] = BOUNDARY_PROMPTS,
) -> CheckResult:
    """Sample chat-templated prompts and score `</think>` behavior.

    Each prompt is sampled `reps` times at `temperature > 0` (sampling, not
    the argmax path greedy-smoke takes) with a small `max_tokens` cap, and
    every generation is scored by :func:`score_boundary_text`.  The request
    uses ``/v1/chat/completions`` with a ``messages`` body; raw completions do
    not apply the model's chat template and cannot exercise this boundary.
    Reasoning-parser responses are recovered through
    :func:`_boundary_text_from_chat_choice` before scoring. Fails when
    total flagged generations exceed `max_defects`. The historical 64-token
    cap and zero bound remain fail-closed pending #87's paired-control policy;
    neither is a calibrated universal artifact-quality threshold. Runs
    alongside KL/PPL, not replacing them.
    """
    invalid = []
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        invalid.append(f"boundary_temperature={temperature!r} is not sampling; must be finite and > 0")
    if type(reps) is not int or reps <= 0:
        invalid.append(f"boundary_reps={reps!r} must be a positive integer")
    if type(max_tokens) is not int or max_tokens <= 0:
        invalid.append(f"boundary_max_tokens={max_tokens!r} must be a positive integer")
    if type(max_defects) is not int or max_defects < 0:
        invalid.append(f"max_boundary_defects={max_defects!r} must be a non-negative integer")
    if (not isinstance(prompts, (tuple, list)) or not prompts
            or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)):
        invalid.append("boundary_prompts must be a nonempty sequence of nonempty strings")
    if invalid:
        return CheckResult(
            name="boundary_behavior",
            passed=False,
            detail="invalid sampling contract: " + "; ".join(invalid),
        )
    n_defects = 0
    by_kind: dict[str, int] = {kind: 0 for kind in BOUNDARY_DEFECTS}
    response_modes: dict[str, int] = {}
    failing_examples: list[dict] = []
    n_generations = 0
    for prompt in prompts:
        for _rep in range(reps):
            try:
                r = _post_json(
                    f"{_server_root(base_url)}{BOUNDARY_ENDPOINT}",
                    {
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "include_reasoning": True,
                        "skip_special_tokens": False,
                        "chat_template_kwargs": dict(
                            BOUNDARY_CHAT_TEMPLATE_KWARGS),
                    },
                )
                choice = r["choices"][0]
                text, response_mode = _boundary_text_from_chat_choice(choice)
                response_modes[response_mode] = (
                    response_modes.get(response_mode, 0) + 1)
                finish = choice.get("finish_reason")
                if finish is not None and not isinstance(finish, str):
                    raise TypeError(
                        "chat choice.finish_reason is not a string or null")
            except Exception as e:
                return CheckResult(
                    name="boundary_behavior",
                    passed=False,
                    detail=f"request failed on {prompt!r}: "
                           f"{type(e).__name__}: {e}",
                )
            scored = score_boundary_text(text, finish)
            n_generations += 1
            for kind in scored["defects"]:
                by_kind[kind] = by_kind.get(kind, 0) + 1
            if scored["defects"]:
                n_defects += 1
                if len(failing_examples) < 5:
                    failing_examples.append({
                        "prompt": prompt,
                        "defects": list(scored["defects"]),
                        "think_tag_count": scored["think_tag_count"],
                        "finish_reason": finish,
                        "excerpt": text[:200],
                    })
    metrics = {
        "endpoint": BOUNDARY_ENDPOINT,
        "request_schema": BOUNDARY_REQUEST_SCHEMA,
        "response_schema": BOUNDARY_RESPONSE_SCHEMA,
        "response_modes": response_modes,
        "n_prompts": len(list(prompts)),
        "reps": reps,
        "n_generations": n_generations,
        "n_defects": n_defects,
        "max_defects": max_defects,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "defects_by_kind": by_kind,
        "failing_examples": failing_examples,
    }
    passed = n_defects <= max_defects
    return CheckResult(
        name="boundary_behavior",
        passed=passed,
        detail=(f"{n_defects}/{n_generations} boundary-defective generations "
                f"(≤ {max_defects} allowed)"
                if not passed else
                f"all {n_generations} sampled generations clean "
                f"({len(list(prompts))} prompts × {reps} reps)"),
        metrics=metrics,
    )


def check_perplexity(base_url: str, model_name: str,
                     max_ppl: float, max_p99_nll: float,
                     max_mean_nll: float,
                     bos_token: str | None = None,
                     add_special_tokens: bool = True) -> CheckResult:
    """Compute per-token NLL across the eval prompt suite.

    **BOS sensitivity (Gemma et al.):** models that key on a leading BOS
    return ~ln(vocab_size) (uniform-random) per-token NLL when the prompt is
    teacher-forced without one. Some exports ship ``add_bos_token=false`` in
    their tokenizer_config, so ``add_special_tokens=True`` alone does NOT add
    it — pass ``bos_token`` (e.g. ``"<bos>"``) to prepend it explicitly. When
    ``bos_token`` is given the request uses ``add_special_tokens=False`` to
    avoid a double-BOS. NOTE: raw-text PPL is also a weak quant-quality signal
    on heavily instruction-tuned models (the off-distribution penalty swamps
    the quantization delta); prefer KL-vs-BF16 for quant A/Bs there.

    Hard fails when mean NLL exceeds threshold OR when the worst
    per-prompt average NLL exceeds threshold. The max guard catches
    bimodal-failure where the model has "quality pockets" (see 27B
    session: 2/10 prompts normal, 8/10 catastrophic at NLL~10). Mean
    alone would have flagged, but the tail prompt is the more
    diagnostic signal.

    **Hard-fails with a diagnostic if spec-decode is detected on the
    serve.** vLLM routes /v1/completions echo+logprobs through the
    draft model when speculative decoding is configured — the NLL
    numbers you'd get back are the 1-layer MTP head's logprobs, not
    the target model's. Running perplexity checks against those is
    like measuring a book's quality by its typos in the copyright
    page. The only reliable fix today is a separate vLLM serve
    without --speculative-config; see the module docstring for the
    standard two-serve workflow.
    """
    try:
        spec_on = _spec_decode_on(base_url)
    except SpecDecodeUndetermined as exc:
        return CheckResult(
            name="perplexity",
            passed=False,
            detail=str(exc),
            metrics={"spec_decode_detected": None, "skipped": True},
        )
    if spec_on:
        return CheckResult(
            name="perplexity",
            passed=False,
            detail=("spec-decode is configured on this vLLM serve — /v1/"
                    "completions echo+logprobs would return DRAFT model "
                    "NLL, not target. Re-serve WITHOUT --speculative-config "
                    "for the perplexity check (use a second serve for MTP "
                    "acceptance). Reports from a spec-decode-on eval have "
                    "false-failed healthy models in the past."),
            metrics={"spec_decode_detected": True, "skipped": True},
        )
    per_prompt_avg_nll: list[float] = []
    total_tokens = 0
    total_nll = 0.0
    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        req_prompt = (bos_token + prompt) if bos_token else prompt
        try:
            r = _post_json(
                f"{base_url}/v1/completions",
                {
                    "model": model_name,
                    "prompt": req_prompt,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": 1,
                    "echo": True,
                    # manual BOS already prepended -> don't let the server add another
                    "add_special_tokens": False if bos_token else add_special_tokens,
                },
            )
        except Exception as e:
            return CheckResult(
                name="perplexity",
                passed=False,
                detail=f"prompt {i}: {type(e).__name__}: {e}",
            )
        lp = r["choices"][0]["logprobs"]
        token_logprobs = lp.get("token_logprobs") or []
        valid = [x for x in token_logprobs if x is not None]
        if not valid:
            return CheckResult(
                name="perplexity",
                passed=False,
                detail=f"prompt {i}: no token_logprobs returned",
            )
        nll = -sum(valid)
        total_nll += nll
        total_tokens += len(valid)
        per_prompt_avg_nll.append(nll / len(valid))

    mean_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(mean_nll)
    per_prompt_avg_nll.sort()
    if len(per_prompt_avg_nll) == 1:
        p99 = per_prompt_avg_nll[0]
    else:
        rank = 0.99 * (len(per_prompt_avg_nll) - 1)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        frac = rank - lo
        p99 = (
            per_prompt_avg_nll[lo] * (1.0 - frac)
            + per_prompt_avg_nll[hi] * frac
        )
    max_nll = per_prompt_avg_nll[-1]

    metrics = {
        "perplexity": ppl,
        "mean_nll_per_tok": mean_nll,
        "p99_nll_per_tok": p99,
        "max_nll_per_tok": max_nll,
        "per_prompt_avg_nll": per_prompt_avg_nll,
        "n_tokens": total_tokens,
        "spec_decode_detected": False,
    }

    reasons = []
    if ppl > max_ppl:
        reasons.append(f"ppl={ppl:.2f} > {max_ppl}")
    if mean_nll > max_mean_nll:
        reasons.append(f"mean_nll={mean_nll:.3f} > {max_mean_nll}")
    if max_nll > max_p99_nll:
        reasons.append(f"max(per-prompt avg NLL)={max_nll:.3f} > {max_p99_nll} "
                       f"(bimodal failure)")
    return CheckResult(
        name="perplexity",
        passed=len(reasons) == 0,
        detail=("OK" if not reasons else "; ".join(reasons)),
        metrics=metrics,
    )


def check_mtp_acceptance(base_url: str, min_p0: float) -> CheckResult:
    """Scrape /metrics for spec-decode acceptance rates. Passes if
    position-0 acceptance fraction exceeds `min_p0`. If no spec-decode
    metrics are exposed (spec-decode not enabled), passes with 'skipped'."""
    try:
        text = _get_text(f"{_server_root(base_url)}/metrics")
    except Exception as e:
        return CheckResult(
            name="mtp_acceptance",
            passed=False,
            detail=f"/metrics unreachable: {type(e).__name__}: {e}",
        )
    drafts = accepted_p0 = None
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            drafts = float(line.split()[-1])
        elif 'spec_decode_num_accepted_tokens_per_pos_total' in line and 'position="0"' in line:
            accepted_p0 = float(line.split()[-1])
    if drafts is None or drafts <= 0:
        return CheckResult(
            name="mtp_acceptance",
            passed=True,
            detail="no spec-decode drafts recorded — skipping (spec-decode off?)",
            metrics={"drafts": drafts or 0},
        )
    frac = (accepted_p0 or 0) / drafts
    metrics = {"drafts": drafts, "accepted_p0": accepted_p0, "accept_rate_p0": frac}
    return CheckResult(
        name="mtp_acceptance",
        passed=(frac >= min_p0),
        detail=(f"pos-0 acceptance = {frac:.1%} "
                f"({'≥' if frac >= min_p0 else '<'} {min_p0:.0%} threshold)"),
        metrics=metrics,
    )


# -----------------------------------------------------------------
# Top-level runner
# -----------------------------------------------------------------
def run_validation(
    base_url: str,
    model_name: str,
    *,
    max_ppl: float = DEFAULT_MAX_PPL,
    max_mean_nll: float = DEFAULT_MAX_MEAN_NLL,
    max_p99_nll: float = DEFAULT_MAX_P99_NLL,
    min_gen_len: int = DEFAULT_MIN_GEN_LEN,
    min_mtp_accept_p0: float = DEFAULT_MIN_MTP_ACCEPT_P0,
    max_boundary_defects: int = DEFAULT_MAX_BOUNDARY_DEFECTS,
    boundary_temperature: float = DEFAULT_BOUNDARY_TEMPERATURE,
    boundary_max_tokens: int = DEFAULT_BOUNDARY_MAX_TOKENS,
    boundary_reps: int = DEFAULT_BOUNDARY_REPS,
    wait_seconds: float = 900.0,
    bos_token: str | None = None,
    add_special_tokens: bool = True,
) -> ValidationReport:
    # `base_url` is the SERVER root, not the OpenAI API root: this module
    # appends `/v1/completions` itself and reads `/health` and `/metrics` off
    # the root. Callers naturally pass the OpenAI root instead (the lane spec
    # published `http://127.0.0.1:8000/v1`), which silently yields
    # `/v1/v1/completions` and `/v1/health` — all 404, so the run waits out its
    # full 900 s timeout having sent no prompt. Normalizing here rather than at
    # each call site keeps the two spellings from diverging again.
    base_url = _server_root(base_url)
    rep = ValidationReport(
        artifact=model_name,
        base_url=base_url,
        model_name=model_name,
        thresholds={
            "max_ppl": max_ppl,
            "max_mean_nll": max_mean_nll,
            "max_p99_nll": max_p99_nll,
            "min_gen_len": min_gen_len,
            "min_mtp_accept_p0": min_mtp_accept_p0,
            "max_boundary_defects": max_boundary_defects,
            "boundary_temperature": boundary_temperature,
            "boundary_max_tokens": boundary_max_tokens,
            "boundary_reps": boundary_reps,
            "bos_token": bos_token,
            "add_special_tokens": add_special_tokens,
        },
    )

    # Wait for server first (don't race probes).
    ok = wait_for_ready(base_url, max_seconds=wait_seconds)
    if not ok:
        rep.checks.append(CheckResult(
            name="serve_ready",
            passed=False,
            detail=f"vLLM /health did not reach 200 within {wait_seconds}s",
        ))
        return rep

    rep.checks.append(check_serve_ready(base_url))
    rep.checks.append(check_generation_sanity(base_url, model_name, min_gen_len))
    rep.checks.append(check_boundary_behavior(
        base_url, model_name,
        max_boundary_defects,
        temperature=boundary_temperature,
        max_tokens=boundary_max_tokens,
        reps=boundary_reps,
    ))
    rep.checks.append(check_perplexity(
        base_url, model_name,
        max_ppl=max_ppl, max_p99_nll=max_p99_nll, max_mean_nll=max_mean_nll,
        bos_token=bos_token,
        add_special_tokens=add_special_tokens,
    ))
    rep.checks.append(check_mtp_acceptance(base_url, min_mtp_accept_p0))
    return rep


def format_report_md(rep: ValidationReport) -> str:
    status = "✅ PASS" if rep.passed else "❌ FAIL"
    lines = [
        f"# PrismaQuant Validation Report — {status}",
        "",
        f"- **artifact:** `{rep.artifact}`",
        f"- **endpoint:** {rep.base_url}",
        f"- **thresholds:** {json.dumps(rep.thresholds, indent=None)}",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for c in rep.checks:
        icon = "✅" if c.passed else "❌"
        lines.append(f"| {c.name} | {icon} | {c.detail} |")
    # Metrics detail
    for c in rep.checks:
        if c.metrics:
            lines.append("")
            lines.append(f"### {c.name} metrics")
            lines.append("```json")
            lines.append(json.dumps(c.metrics, indent=2, default=str))
            lines.append("```")
    return "\n".join(lines)


# -----------------------------------------------------------------
# Ship record (R13)
# -----------------------------------------------------------------
def _resolve_artifact_dir(args, card_model_dir: str | None) -> str | None:
    """Which directory this verdict is about.

    The validator drives an HTTP endpoint, so it cannot see what the server
    loaded; the honest fallback order is explicit flag, then --model-name if it
    happens to be a local path, then the directory the shipcard was opened on.
    """
    if args.artifact_dir:
        return args.artifact_dir
    if args.model_name and os.path.isdir(args.model_name):
        return args.model_name
    return card_model_dir


def _fill_shipcard(args, rep: "ValidationReport") -> None:
    if not getattr(args, "shipcard", None):
        return
    from .shipcard import (
        compute_model_sha, fill_if_requested, git_provenance, load_shipcard,
        make_record,
    )

    try:
        card = load_shipcard(args.shipcard)
    except Exception as exc:
        print(f"[shipcard] WARN {args.shipcard} unreadable: {exc!r}")
        return
    model_dir = _resolve_artifact_dir(args, card.get("model_dir"))
    try:
        model_sha = compute_model_sha(model_dir) if model_dir else None
    except Exception:
        model_sha = None

    ppl_check = next((c for c in rep.checks if c.name == "perplexity"), None)
    spec_detected = None
    if ppl_check is not None:
        spec_detected = bool(ppl_check.metrics.get("spec_decode_detected", False))
    metrics = {c.name: {"passed": c.passed, **c.metrics} for c in rep.checks}
    perplexity = metrics.get("perplexity")
    if isinstance(perplexity, dict):
        # Make the replayable evidence self-contained. ValidationReport already
        # owns the threshold decision; the shipcard additionally needs the
        # exact number of scored tokens to reject a fabricated empty pass.
        if "n_tokens" not in perplexity:
            perplexity["n_tokens"] = 0
    record = make_record(
        slot="ship_gate",
        tool="validate_quantized_model.py",
        passed=bool(rep.passed),
        model_sha=model_sha,
        metrics=metrics,
        detail="; ".join(
            f"{c.name}={'pass' if c.passed else 'FAIL'}" for c in rep.checks),
        spec_decode_detected=spec_detected,
        git_commit=git_provenance().get("commit"),
        extra={
            "base_url": rep.base_url,
            "served_model_name": rep.model_name,
            "thresholds": rep.thresholds,
            "model_sha_source": model_dir,
        },
    )
    fill_if_requested(args.shipcard, "ship_gate", record)


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
                description="Pre-ship quality validator for PrismaQuant artifacts. "
                    "Hits a running vLLM endpoint and runs serve / generation "
                    "sanity / perplexity / MTP acceptance / boundary-behavior checks.")
    ap.add_argument("--base-url", default=os.environ.get("VLLM_URL",
                                                         "http://localhost:8000"),
                    help="vLLM OpenAI-compatible server URL")
    ap.add_argument("--model-name", required=True,
                    help="Model name as reported by vLLM (local path like /exported "
                         "or HF repo id like org/name)")
    ap.add_argument("--max-ppl", type=float, default=DEFAULT_MAX_PPL)
    ap.add_argument("--max-mean-nll", type=float, default=DEFAULT_MAX_MEAN_NLL)
    ap.add_argument("--max-p99-nll", type=float, default=DEFAULT_MAX_P99_NLL)
    ap.add_argument("--min-gen-len", type=int, default=DEFAULT_MIN_GEN_LEN)
    ap.add_argument("--min-mtp-accept-p0", type=float,
                    default=DEFAULT_MIN_MTP_ACCEPT_P0)
    ap.add_argument("--max-boundary-defects", type=int,
                    default=DEFAULT_MAX_BOUNDARY_DEFECTS,
                    help="Max sampled boundary-defective generations allowed "
                         "(stutter/zero-tag/cap-truncation on terse prompts)")
    ap.add_argument("--boundary-temperature", type=float,
                    default=DEFAULT_BOUNDARY_TEMPERATURE,
                    help="Sampling temperature for the boundary check; must "
                         "stay > 0 (the defect is invisible at temp 0)")
    ap.add_argument("--boundary-max-tokens", type=int,
                    default=DEFAULT_BOUNDARY_MAX_TOKENS)
    ap.add_argument("--boundary-reps", type=int,
                    default=DEFAULT_BOUNDARY_REPS,
                    help="Sampled repetitions per boundary prompt")
    ap.add_argument("--bos-token", default=None,
                    help="Optional literal BOS string to prepend before "
                         "perplexity prompts for BOS-sensitive tokenizers "
                         "(for example '<bos>'). When set, the server "
                         "request disables add_special_tokens to avoid a "
                         "double BOS.")
    ap.add_argument("--no-add-special-tokens", dest="add_special_tokens",
                    action="store_false", default=True,
                    help="Pass add_special_tokens=false on perplexity "
                         "requests when --bos-token is not used.")
    ap.add_argument("--wait-seconds", type=float, default=900.0,
                    help="Max time to wait for /health 200 before giving up")
    ap.add_argument("--report", default=None,
                    help="Optional path to write the markdown report")
    ap.add_argument("--shipcard", default=None,
                    help="Path to the artifact's shipcard.json; this run's "
                         "verdict is appended to the ship_gate slot "
                         "(see python -m prismaquant.shipcard_cli).")
    ap.add_argument("--artifact-dir", default=None,
                    help="Local directory of the artifact being served, used "
                         "to stamp model_sha on the shipcard record. Defaults "
                         "to --model-name when that is a directory, then to "
                         "the shipcard's own model_dir.")
    args = ap.parse_args()

    rep = run_validation(
        args.base_url, args.model_name,
        max_ppl=args.max_ppl,
        max_mean_nll=args.max_mean_nll,
        max_p99_nll=args.max_p99_nll,
        min_gen_len=args.min_gen_len,
        min_mtp_accept_p0=args.min_mtp_accept_p0,
        max_boundary_defects=args.max_boundary_defects,
        boundary_temperature=args.boundary_temperature,
        boundary_max_tokens=args.boundary_max_tokens,
        boundary_reps=args.boundary_reps,
        wait_seconds=args.wait_seconds,
        bos_token=args.bos_token,
        add_special_tokens=args.add_special_tokens,
    )
    md = format_report_md(rep)
    print(md)
    if args.report:
        with open(args.report, "w") as f:
            f.write(md)
    _fill_shipcard(args, rep)
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
