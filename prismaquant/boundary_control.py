"""Opt-in paired boundary measurement; deliberately not a shipping verdict.

The BF16 control determines a finishing budget for a frozen prompt/seed
schedule. A token or iteration bound is an inconclusive stop, never proof of
convergence. Candidate and control are then compared at the identical cap.
This module reuses the shipping probe's HTTP, response and scoring contracts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

from prismaquant import validate_quantized_model as vqm

SCHEMA = "prismaquant.boundary_control/1"
SCORER = "prismaquant.boundary_close_count/1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def artifact_content_id(content):
    """Bind exact safetensors bytes plus the existing model metadata scope."""
    from prismaquant.shipcard import _closed_weight_content_manifest

    if (not isinstance(content, dict)
            or set(content) != {"model_sha", "weight_content_manifest"}
            or not isinstance(content["model_sha"], str)
            or re.fullmatch(r"[0-9a-f]{64}", content["model_sha"]) is None):
        raise ValueError("artifact_content requires model_sha and exact weight content manifest")
    _closed_weight_content_manifest(content["weight_content_manifest"],
                                    where="boundary artifact_content")
    return digest(content)


def _positive(value, field):
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def require_bf16_control(config, live_dtype, live_quantization, *, launch_argv=None):
    if launch_argv is not None:
        dtypes = []
        override_flags = ("--quantization", "--quantization-config", "--hf-overrides",
                          "--hf-config-path")
        for i, token in enumerate(launch_argv):
            flag, separator, value = token.partition("=")
            if (flag.startswith("-q") and not flag.startswith("--")) or any(
                flag == option or flag.startswith(option + ".") for option in override_flags
            ):
                raise ValueError(f"BF16 control refuses live override {flag}")
            if flag == "--dtype":
                if not separator:
                    value = launch_argv[i + 1] if i + 1 < len(launch_argv) else ""
                dtypes.append(value)
        if len(dtypes) > 1:
            raise ValueError("BF16 control refuses repeated --dtype")
        live_dtype = dtypes[0] if dtypes else None
    source_dtype = config.get("dtype", config.get("torch_dtype"))
    if (source_dtype not in ("bfloat16", "torch.bfloat16")
            or config.get("quantization_config")
            or live_dtype not in (None, "auto", "bfloat16")
            or live_quantization is not None):
        raise ValueError("BF16 control requires BF16 source and observed BF16/auto unquantized serve")


def _validate(contract, binding):
    if not isinstance(contract, dict) or set(contract) != {
        "prompts", "seeds", "temperature", "initial_max_tokens", "max_steps"
    }:
        raise ValueError("measurement contract fields differ from schema")
    prompts, seeds = contract["prompts"], contract["seeds"]
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be non-empty")
    for row in prompts:
        if (not isinstance(row, dict) or set(row) != {"text", "stratum"}
                or any(not isinstance(v, str) or not v.strip() for v in row.values())):
            raise ValueError("prompt requires text and stratum")
    if len({row["text"] for row in prompts}) != len(prompts):
        raise ValueError("prompts must be unique")
    if (not isinstance(seeds, list) or not seeds
            or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct non-negative integers")
    temp = contract["temperature"]
    if type(temp) not in (int, float) or not math.isfinite(temp) or temp <= 0:
        raise ValueError("temperature must be finite and positive")
    _positive(contract["initial_max_tokens"], "initial_max_tokens")
    _positive(contract["max_steps"], "max_steps")
    for field in ("campaign_id", "artifact_id", "serve_session_id",
                  "serve_fingerprint", "host_boot_id", "producer_source_sha256"):
        if not isinstance(binding.get(field), str) or not binding[field].strip():
            raise ValueError(f"missing {field}")
    if binding["artifact_id"] != artifact_content_id(binding.get("artifact_content")):
        raise ValueError("artifact_id differs from exact artifact_content")
    context = _positive(binding.get("model_context_tokens"), "model_context_tokens")
    counts = binding.get("prompt_tokens")
    if not isinstance(counts, list) or len(counts) != len(prompts):
        raise ValueError("prompt_tokens does not cover the prompt schedule")
    for n in counts:
        _positive(n, "prompt_tokens")
        if n >= context:
            raise ValueError("prompt_tokens exhaust model context")
    tokens = binding.get("prompt_token_ids")
    if (not isinstance(tokens, list) or len(tokens) != len(counts)
            or any(not isinstance(row, list) or len(row) != n
                   or any(type(token) is not int or token < 0 for token in row)
                   for row, n in zip(tokens, counts))):
        raise ValueError("prompt_tokens and prompt_token_ids disagree or are malformed")
    return min(context - n for n in counts)


def _summary(outcomes):
    by_kind = {kind: 0 for kind in vqm.BOUNDARY_DEFECTS}
    n_defects = 0
    for row in outcomes:
        score = vqm.score_boundary_text(row["text"], row["finish_reason"])
        if row["score"] != score:
            raise ValueError("outcome score disagrees with the frozen scorer")
        n_defects += bool(score["defects"])
        for kind in score["defects"]:
            by_kind[kind] += 1
    return {"n_generations": len(outcomes), "n_defects": n_defects,
            "defects_by_kind": by_kind}


def measure_step(base_url, model_name, contract, max_tokens, *, raw=False):
    outcomes = []
    for index, prompt in enumerate(contract["prompts"]):
        for seed in contract["seeds"]:
            body = {"model": model_name, "max_tokens": max_tokens,
                    "temperature": contract["temperature"], "seed": seed,
                    "top_p": 1.0, "skip_special_tokens": False}
            if raw:
                body["prompt"] = prompt["text"]
                endpoint = "/v1/completions"
            else:
                body.update(messages=[{"role": "user", "content": prompt["text"]}],
                            include_reasoning=True,
                            chat_template_kwargs=dict(vqm.BOUNDARY_CHAT_TEMPLATE_KWARGS))
                endpoint = vqm.BOUNDARY_ENDPOINT
            response = vqm._post_json(vqm._server_root(base_url) + endpoint, body)
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("response must contain exactly one choice")
            choice = choices[0]
            if raw:
                text, mode = choice.get("text"), "raw_completion"
                if not isinstance(text, str):
                    raise ValueError("raw completion text must be a string")
            else:
                text, mode = vqm._boundary_text_from_chat_choice(choice)
            finish = choice.get("finish_reason")
            if finish not in ("stop", "length"):
                raise ValueError(f"unsupported finish_reason: {finish!r}")
            tokens = response.get("usage", {}).get("completion_tokens")
            _positive(tokens, "completion_tokens")
            if tokens > max_tokens:
                raise ValueError("completion_tokens exceeds requested cap")
            outcomes.append({"prompt_index": index, "seed": seed, "text": text,
                             "response_mode": mode, "finish_reason": finish,
                             "completion_tokens": tokens,
                             "score": vqm.score_boundary_text(text, finish)})
    return {"max_tokens": max_tokens, "endpoint": endpoint,
            "outcomes": outcomes, **_summary(outcomes)}


def _receipt(contract, binding, role):
    return {"schema": SCHEMA, "scorer": SCORER,
            "request_schema": vqm.BOUNDARY_REQUEST_SCHEMA,
            "response_schema": vqm.BOUNDARY_RESPONSE_SCHEMA,
            "promptset_sha256": digest(contract["prompts"]),
            "contract": copy.deepcopy(contract), "binding": copy.deepcopy(binding),
            "role": role, "advisory_only": True}


def measure_control(base_url, model_name, contract, binding):
    bound = _validate(contract, binding)
    cap = min(contract["initial_max_tokens"], bound)
    receipt = _receipt(contract, binding, "bf16_control")
    receipt.update(steps=[], fixed_point=False, stop_reason="iteration_backstop")
    for _ in range(contract["max_steps"]):
        step = measure_step(base_url, model_name, contract, cap)
        receipt["steps"].append(step)
        if not step["defects_by_kind"]["cap_truncation"]:
            receipt.update(fixed_point=True, stop_reason="uncensored_control")
            break
        if cap == bound:
            receipt["stop_reason"] = "context_bound"
            break
        cap = min(2 * cap, bound)
    replay_control(receipt)
    return receipt


def _replay_step(step, contract, cap):
    if step.get("max_tokens") != cap or step.get("endpoint") != vqm.BOUNDARY_ENDPOINT:
        raise ValueError("step cap or endpoint differs from measurement contract")
    schedule = [(i, seed) for i in range(len(contract["prompts"]))
                for seed in contract["seeds"]]
    outcomes = step.get("outcomes")
    if not isinstance(outcomes, list) or [
        (row.get("prompt_index"), row.get("seed")) for row in outcomes
    ] != schedule:
        raise ValueError("outcomes do not match exact prompt/seed schedule")
    for row in outcomes:
        if not isinstance(row.get("text"), str) or row.get("finish_reason") not in ("stop", "length"):
            raise ValueError("malformed outcome text or finish_reason")
        if _positive(row.get("completion_tokens"), "completion_tokens") > cap:
            raise ValueError("outcome completion_tokens exceeds cap")
    for key, value in _summary(outcomes).items():
        if step.get(key) != value:
            raise ValueError(f"step {key} disagrees with outcomes")
    return not step["defects_by_kind"]["cap_truncation"]


def _replay_header(receipt):
    expected = {"schema": SCHEMA, "scorer": SCORER,
                "request_schema": vqm.BOUNDARY_REQUEST_SCHEMA,
                "response_schema": vqm.BOUNDARY_RESPONSE_SCHEMA,
                "advisory_only": True}
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"receipt {field} differs from measurement contract")
    contract, binding = receipt["contract"], receipt["binding"]
    bound = _validate(contract, binding)
    if receipt.get("promptset_sha256") != digest(contract["prompts"]):
        raise ValueError("promptset_sha256 does not bind measured prompts")
    return contract, bound


def replay_control(receipt):
    contract, bound = _replay_header(receipt)
    steps = receipt.get("steps")
    if receipt.get("role") != "bf16_control" or not isinstance(steps, list) or not steps:
        raise ValueError("control requires BF16 role and nonempty steps")
    if len(steps) > contract["max_steps"]:
        raise ValueError("control exceeded its iteration backstop")
    cap = min(contract["initial_max_tokens"], bound)
    finished = False
    for i, step in enumerate(steps):
        finished = _replay_step(step, contract, cap)
        if i < len(steps) - 1 and (finished or cap == bound):
            raise ValueError("control continued after its stopping condition")
        if i < len(steps) - 1:
            cap = min(2 * cap, bound)
    reason = ("uncensored_control" if finished else "context_bound" if cap == bound
              else "iteration_backstop")
    if not finished and cap < bound and len(steps) != contract["max_steps"]:
        raise ValueError("control stopped before its fixed point or backstop")
    if receipt.get("fixed_point") is not finished or receipt.get("stop_reason") != reason:
        raise ValueError("control fixed point is not supported by its outcomes")
    return cap


def _paired(control, binding):
    cap = replay_control(control)
    if not control["fixed_point"]:
        raise ValueError("BF16 control did not reach a finishing fixed point")
    _validate(control["contract"], binding)
    for field in ("campaign_id", "host_boot_id", "serve_fingerprint",
                  "model_context_tokens", "prompt_tokens", "prompt_token_ids",
                  "producer_source_sha256"):
        if binding[field] != control["binding"][field]:
            raise ValueError(f"paired {field} differs")
    if (binding["artifact_id"] != control["binding"]["artifact_id"]
            and binding["serve_session_id"] == control["binding"]["serve_session_id"]):
        raise ValueError("distinct artifacts cannot share one serve_session_id")
    return cap


def measure_candidate(base_url, model_name, control, binding):
    cap = _paired(control, binding)
    receipt = _receipt(control["contract"], binding, "candidate")
    receipt["control_sha256"] = digest(control)
    receipt["step"] = measure_step(base_url, model_name, control["contract"], cap)
    return receipt


def compare(control, candidate):
    cap = _paired(control, candidate["binding"])
    _replay_header(candidate)
    if (candidate.get("role") != "candidate" or candidate["contract"] != control["contract"]
            or candidate.get("control_sha256") != digest(control)):
        raise ValueError("candidate does not bind the exact control and contract")
    _replay_step(candidate["step"], control["contract"], cap)
    strata = {}
    for prompt in control["contract"]["prompts"]:
        strata.setdefault(prompt["stratum"], {"control_defects": 0, "candidate_defects": 0,
                                               "n_generations_per_arm": 0})
    for label, step in (("control", control["steps"][-1]), ("candidate", candidate["step"])):
        for outcome in step["outcomes"]:
            stratum = control["contract"]["prompts"][outcome["prompt_index"]]["stratum"]
            strata[stratum][f"{label}_defects"] += bool(outcome["score"]["defects"])
            if label == "control":
                strata[stratum]["n_generations_per_arm"] += 1
    for row in strata.values():
        row["delta_defects"] = row["candidate_defects"] - row["control_defects"]
    return {"schema": "prismaquant.boundary_comparison/1", "advisory_only": True,
            "control_sha256": digest(control), "candidate_sha256": digest(candidate),
            "max_tokens": cap, "strata": strata}
