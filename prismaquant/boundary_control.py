"""Paired boundary measurement and an opt-in, non-default acceptance policy.

The BF16 control determines a finishing budget for a frozen prompt/seed
schedule. A token or iteration bound is an inconclusive stop, never proof of
convergence. Candidate and control are then compared at the identical cap.
This module reuses the shipping probe's HTTP, response and scoring contracts.
The separately versioned no-new-failures decision rejects new defect kinds
on each matched pair. It does not replace the production shipping gate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

from prismaquant import validate_quantized_model as vqm

SCHEMA = "prismaquant.boundary_control/1"
DECLARED_STACK_SCHEMA = "prismaquant.boundary_control/2"
SCORER = "prismaquant.boundary_close_count/1"
DECISION_SCHEMA = "prismaquant.boundary_decision/1"
NO_NEW_FAILURES_POLICY = "prismaquant.no_new_boundary_failures/1"


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


def validate_stack_contract(contract):
    """An explicit image/artifact/role declaration, never a residency wildcard."""
    if (not isinstance(contract, dict) or set(contract) != {"schema", "image", "roles"}
            or contract["schema"] != "prismaquant.boundary_stack_contract/1"
            or not isinstance(contract["image"], str)
            or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", contract["image"])):
        raise ValueError("invalid paired stack contract")
    roles = contract["roles"]
    if not isinstance(roles, dict) or set(roles) != {"bf16_control", "candidate"}:
        raise ValueError("paired stack contract requires both treatment roles")
    for row in roles.values():
        if (not isinstance(row, dict) or set(row) != {"artifact_id", "resident_extensions"}
                or not isinstance(row["artifact_id"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["artifact_id"])):
            raise ValueError("paired stack contract requires exact artifact_id")
        extensions = row["resident_extensions"]
        if (not isinstance(extensions, list)
                or any(not isinstance(name, str) or not re.fullmatch(
                    r"[A-Za-z0-9_+.-]+\.so(?:\.[0-9]+)*", name) for name in extensions)
                or extensions != sorted(set(extensions))):
            raise ValueError("paired resident_extensions must be exact sorted unique basenames")
    if roles["bf16_control"]["artifact_id"] == roles["candidate"]["artifact_id"]:
        raise ValueError("paired stack contract requires distinct treatment artifacts")
    return contract


def _paired_stack_payload(binding, role):
    from tools import serve_fingerprint as sf

    proof = binding.get("paired_stack")
    if (not isinstance(proof, dict) or set(proof) != {"schema", "role", "contract", "manifest"}
            or proof["schema"] != "prismaquant.boundary_stack_binding/1"
            or proof["role"] != role or role not in {"bf16_control", "candidate"}):
        raise ValueError("paired stack binding role or schema differs")
    contract = validate_stack_contract(proof["contract"])
    expected = contract["roles"][role]
    if binding["artifact_id"] != expected["artifact_id"]:
        raise ValueError("paired stack treatment artifact_id differs")
    manifest = proof["manifest"]
    if not isinstance(manifest, dict) or manifest.get("residency_readable") is not True:
        raise ValueError("paired stack requires readable residency")
    raw = sf.performance_stack_fingerprint(manifest)
    if raw != binding["serve_fingerprint"] or raw != manifest.get("performance_stack_fingerprint"):
        raise ValueError("paired raw serve fingerprint differs from observed manifest")
    if (manifest.get("serve_session_id") != binding["serve_session_id"]
            or (manifest.get("host_identity") or {}).get("boot_id") != binding["host_boot_id"]):
        raise ValueError("paired stack live session or host binding differs")
    payload = sf.performance_stack_payload(manifest)
    if payload["image"] != contract["image"]:
        raise ValueError("paired stack image differs from treatment declaration")
    if payload["resident_extensions"] != expected["resident_extensions"]:
        raise ValueError("paired resident_extensions differ from exact role declaration")
    # The complete raw fingerprint remains in the binding. Only this explicitly
    # declared treatment coordinate is projected out of the paired comparison.
    return {key: value for key, value in payload.items() if key != "resident_extensions"}


def bind_stack_contract(binding, contract, manifest, role):
    proof = {"schema": "prismaquant.boundary_stack_binding/1", "role": role,
             "contract": copy.deepcopy(contract), "manifest": copy.deepcopy(manifest)}
    _paired_stack_payload({**binding, "paired_stack": proof}, role)
    return proof


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


def validate_contract(contract):
    """Validate the frozen schedule before a controller starts any server."""
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
    return contract


def _validate(contract, binding):
    validate_contract(contract)
    prompts = contract["prompts"]
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
    if "paired_stack" in binding:
        _paired_stack_payload(binding, role)
    return {"schema": DECLARED_STACK_SCHEMA if "paired_stack" in binding else SCHEMA, "scorer": SCORER,
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
    declared = "paired_stack" in receipt.get("binding", {})
    expected = {"schema": DECLARED_STACK_SCHEMA if declared else SCHEMA, "scorer": SCORER,
                "request_schema": vqm.BOUNDARY_REQUEST_SCHEMA,
                "response_schema": vqm.BOUNDARY_RESPONSE_SCHEMA,
                "advisory_only": True}
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"receipt {field} differs from measurement contract")
    contract, binding = receipt["contract"], receipt["binding"]
    bound = _validate(contract, binding)
    if declared:
        _paired_stack_payload(binding, receipt.get("role"))
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
    reference = control["binding"]
    declared = "paired_stack" in reference or "paired_stack" in binding
    if declared:
        control_payload = _paired_stack_payload(reference, "bf16_control")
        candidate_payload = _paired_stack_payload(binding, "candidate")
        if reference["paired_stack"]["contract"] != binding["paired_stack"]["contract"]:
            raise ValueError("paired stack treatment declaration differs")
        for field in control_payload:
            if digest(control_payload[field]) != digest(candidate_payload[field]):
                raise ValueError(f"paired serve stack {field} differs")
    elif binding["serve_fingerprint"] != reference["serve_fingerprint"]:
        raise ValueError("paired serve_fingerprint differs")
    for field in ("campaign_id", "host_boot_id",
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


def decide_no_new_failures(control, candidate):
    """Apply the opt-in paired rule only after replaying comparable evidence.

    A repair never offsets a new failure, even within one stratum. Existing
    BF16 defects may persist or disappear; introducing a new kind on that
    same prompt/seed refuses. A censored candidate is a measured new defect,
    whereas a censored control supplies no valid comparison and is
    inconclusive. This policy is not an estimator of a population failure
    probability and does not change a production default.
    """
    result = {"schema": DECISION_SCHEMA, "policy": NO_NEW_FAILURES_POLICY,
              "advisory_only": True, "verdict": "inconclusive", "reason": "",
              "control_sha256": None, "candidate_sha256": None,
              "pair_count": 0, "max_tokens": None, "newly_broken_pairs": 0,
              "new_defect_kind_pairs": 0, "violations": []}
    try:
        if not isinstance(control, dict) or not isinstance(candidate, dict):
            raise ValueError("both control and candidate measurement receipts are required")
        comparison = compare(control, candidate)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
        # Malformed serialized evidence is no measurement. Do not coerce a
        # missing count to zero or accept an asserted fixed-point flag.
        result["reason"] = f"paired evidence is incomplete or incomparable: {exc}"
        return result
    result.update(control_sha256=comparison["control_sha256"],
                  candidate_sha256=comparison["candidate_sha256"],
                  max_tokens=comparison["max_tokens"])
    controls = control["steps"][-1]["outcomes"]
    candidates = candidate["step"]["outcomes"]
    result["pair_count"] = len(controls)
    for reference, observed in zip(controls, candidates):
        before, after = set(reference["score"]["defects"]), set(observed["score"]["defects"])
        introduced = after - before
        if not introduced:
            continue
        result["newly_broken_pairs" if not before else "new_defect_kind_pairs"] += 1
        result["violations"].append({
            "prompt_index": observed["prompt_index"], "seed": observed["seed"],
            "stratum": control["contract"]["prompts"][observed["prompt_index"]]["stratum"],
            "control_defects": sorted(before), "candidate_defects": sorted(after),
            "new_defects": sorted(introduced),
        })
    result["verdict"] = "refused" if result["violations"] else "accepted"
    result["reason"] = ("candidate introduced new defects on matched prompt/seed pairs"
                        if result["violations"] else
                        "no new defect kind on any matched prompt/seed pair")
    return result


def replay_no_new_failures(decision, control, candidate):
    """Recompute the complete decision; a stored passed/verdict bit is not evidence."""
    expected = decide_no_new_failures(control, candidate)
    # Python equality coerces 0/False and 2/2.0. A versioned JSON receipt must
    # retain the actual counter/boolean types as well as their values.
    try:
        matches = digest(decision) == digest(expected)
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise ValueError("boundary decision differs from replayed paired policy")
    return expected
