#!/usr/bin/env python3
"""Measure the opt-in #87 instrument inside an already running serve container.

Use one PrismaBuild action for sequential BF16-control/candidate serves. This
client starts no server and allocates no GPU context. The same immutable image,
source snapshot, prompt contract and action/campaign ID must cover both arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

# Reuse the source bootstrap's stdlib namespace, avoiding package __init__ and
# its torch import in the serving container. The exact directory is this tool's.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prismaquant_source_bootstrap import (  # noqa: E402
    activate_prismaquant_source, _install_exact_package_namespace,
)


def _capture_artifact(model_dir):
    """Hash native source weights, refusing mutation during the hash itself."""
    from prismaquant.shipcard import (
        build_weight_content_manifest, compute_model_sha, weight_stat_attestation,
    )

    before = weight_stat_attestation(model_dir)
    content = {"model_sha": compute_model_sha(model_dir),
               "weight_content_manifest": build_weight_content_manifest(model_dir)}
    after = weight_stat_attestation(model_dir)
    if before != after:
        raise ValueError("artifact changed during measurement identity capture")
    return content, after


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("control", "candidate"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True, help="immutable repository@sha256 digest")
    parser.add_argument("--contract", help="frozen prompt/seed/cap contract JSON (control)")
    parser.add_argument("--control", help="completed BF16 control receipt (candidate)")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--legacy-ab", action="store_true",
                        help="also measure historical raw request at initial cap on control")
    parser.add_argument("--decision-policy", choices=("no-new-failures",),
                        help="opt-in candidate decision; leaves the production gate unchanged")
    args = parser.parse_args(argv)
    if "@sha256:" not in args.image:
        parser.error("--image must be immutable")
    output = Path(args.out)
    if output.exists():
        parser.error("--out already exists; use a fresh receipt path")
    root = activate_prismaquant_source()
    _install_exact_package_namespace(root)
    from prismaquant import boundary_control as bc
    import serve_fingerprint as sf

    if args.role == "control":
        if not args.contract or args.control or args.decision_policy:
            parser.error("control requires --contract and no --control/--decision-policy")
        contract = json.loads(Path(args.contract).read_text())
        model_config = json.loads((Path(args.model_dir) / "config.json").read_text())
        control = None
    else:
        if not args.control or args.contract or args.legacy_ab:
            parser.error("candidate requires --control, no --contract/--legacy-ab")
        control = json.loads(Path(args.control).read_text())["measurement"]
        bc.replay_control(control)
        contract = control["contract"]

    artifact_pre, weight_stats_pre = _capture_artifact(args.model_dir)
    before = sf.collect_manifest(image=args.image, base_url=args.base_url,
                                 attestation_phase="pre")
    if not before["residency_readable"] or not before.get("serve_session_id"):
        raise ValueError("serve process/residency attestation is incomplete")
    model = before["models_endpoint_binding"]["model"]
    if model["id"] != args.model_name:
        raise ValueError("requested model is not the observed live serve")
    if Path(before["model"]).resolve() != Path(args.model_dir).resolve():
        raise ValueError("model-dir differs from the observed server argv")
    if args.role == "control":
        bc.require_bf16_control(model_config,
            sf._flag_value(before["launch_argv"], "--dtype"), before["quantization"],
            launch_argv=before["launch_argv"])
    counts = []
    token_ids = []
    for prompt in contract["prompts"]:
        tokenized = bc.vqm._post_json(bc.vqm._server_root(args.base_url) + "/tokenize", {
            "model": args.model_name,
            "messages": [{"role": "user", "content": prompt["text"]}],
            "add_generation_prompt": True,
            "chat_template_kwargs": dict(bc.vqm.BOUNDARY_CHAT_TEMPLATE_KWARGS),
        })
        counts.append(tokenized["count"])
        token_ids.append(tokenized["tokens"])
    source_hashes = {path: bc.hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in ("prismaquant/boundary_control.py",
                     "prismaquant/validate_quantized_model.py", "prismaquant/shipcard.py",
                     "tools/measure_boundary_control.py", "tools/serve_fingerprint.py",
                     "tools/prismaquant_source_bootstrap.py",
                     "prismaquant/tessera_runtime/tessera_serving_runtime_pin.json")}
    binding = {
        "campaign_id": args.campaign_id,
        "artifact_id": bc.artifact_content_id(artifact_pre),
        "artifact_content": artifact_pre,
        "serve_session_id": before["serve_session_id"],
        "serve_fingerprint": before["performance_stack_fingerprint"],
        "host_boot_id": before["host_identity"]["boot_id"],
        "model_context_tokens": model["max_model_len"],
        "prompt_tokens": counts,
        "prompt_token_ids": token_ids,
        "producer_source_sha256": bc.digest(source_hashes),
    }
    context_bound = bc._validate(contract, binding)
    started = time.time()
    historical = None
    if args.legacy_ab:
        historical = bc.measure_step(args.base_url, args.model_name, contract,
                                     min(contract["initial_max_tokens"], context_bound), raw=True)
    if control is None:
        measurement = bc.measure_control(args.base_url, args.model_name, contract, binding)
        comparison = None
    else:
        measurement = bc.measure_candidate(args.base_url, args.model_name, control, binding)
        comparison = bc.compare(control, measurement)
    decision = (bc.decide_no_new_failures(control, measurement)
                if args.decision_policy else None)
    finished = time.time()
    after = sf.collect_manifest(image=args.image, base_url=args.base_url,
                                attestation_phase="post")
    artifact_post, weight_stats_post = _capture_artifact(args.model_dir)
    if artifact_pre != artifact_post or weight_stats_pre != weight_stats_post:
        raise ValueError("artifact changed during measurement")
    for field in ("serve_session_id", "performance_stack_fingerprint", "models_endpoint_binding"):
        if before[field] != after[field]:
            raise ValueError(f"serve changed during measurement: {field}")
    receipt = {"schema": "prismaquant.boundary_campaign_arm/1",
               "measurement": measurement, "legacy_raw": historical,
               "comparison": comparison, "started_unix": started,
               "finished_unix": finished, "serve_pre": before, "serve_post": after,
               "artifact_pre": artifact_pre, "artifact_post": artifact_post,
               "weight_stats_pre": weight_stats_pre, "weight_stats_post": weight_stats_post,
               "source_sha256": source_hashes}
    if decision is not None:
        receipt["decision"] = decision
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"receipt": str(output), "sha256": bc.digest(receipt),
                      "comparison": comparison, "decision": decision,
                      "fixed_point": measurement.get("fixed_point")}))
    if decision is not None:
        return 0 if decision["verdict"] == "accepted" else 2
    return 0 if measurement.get("fixed_point", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
