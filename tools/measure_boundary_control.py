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
    args = parser.parse_args(argv)
    if "@sha256:" not in args.image:
        parser.error("--image must be immutable")
    output = Path(args.out)
    if output.exists():
        parser.error("--out already exists; use a fresh receipt path")
    root = activate_prismaquant_source()
    _install_exact_package_namespace(root)
    from prismaquant import boundary_control as bc
    from prismaquant.shipcard import compute_model_sha
    import serve_fingerprint as sf

    if args.role == "control":
        if not args.contract or args.control:
            parser.error("control requires --contract and no --control")
        contract = json.loads(Path(args.contract).read_text())
        model_config = json.loads((Path(args.model_dir) / "config.json").read_text())
        if model_config.get("quantization_config") or model_config.get(
            "dtype", model_config.get("torch_dtype")
        ) not in ("bfloat16", "torch.bfloat16"):
            parser.error("BF16 control requires an unquantized bfloat16 model config")
        control = None
    else:
        if not args.control or args.contract or args.legacy_ab:
            parser.error("candidate requires --control, no --contract/--legacy-ab")
        control = json.loads(Path(args.control).read_text())["measurement"]
        bc.replay_control(control)
        contract = control["contract"]

    before = sf.collect_manifest(image=args.image, base_url=args.base_url,
                                 attestation_phase="pre")
    if not before["residency_readable"] or not before.get("serve_session_id"):
        raise ValueError("serve process/residency attestation is incomplete")
    model = before["models_endpoint_binding"]["model"]
    if model["id"] != args.model_name:
        raise ValueError("requested model is not the observed live serve")
    if Path(before["model"]).resolve() != Path(args.model_dir).resolve():
        raise ValueError("model-dir differs from the observed server argv")
    counts = []
    for prompt in contract["prompts"]:
        tokenized = bc.vqm._post_json(bc.vqm._server_root(args.base_url) + "/tokenize", {
            "model": args.model_name,
            "messages": [{"role": "user", "content": prompt["text"]}],
            "add_generation_prompt": True,
            "chat_template_kwargs": dict(bc.vqm.BOUNDARY_CHAT_TEMPLATE_KWARGS),
        })
        counts.append(tokenized["count"])
    binding = {
        "campaign_id": args.campaign_id,
        "artifact_id": compute_model_sha(args.model_dir),
        "serve_session_id": before["serve_session_id"],
        "serve_fingerprint": before["performance_stack_fingerprint"],
        "host_boot_id": before["host_identity"]["boot_id"],
        "model_context_tokens": model["max_model_len"],
        "prompt_tokens": counts,
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
    finished = time.time()
    after = sf.collect_manifest(image=args.image, base_url=args.base_url,
                                attestation_phase="post")
    for field in ("serve_session_id", "performance_stack_fingerprint", "models_endpoint_binding"):
        if before[field] != after[field]:
            raise ValueError(f"serve changed during measurement: {field}")
    receipt = {"schema": "prismaquant.boundary_campaign_arm/1",
               "measurement": measurement, "legacy_raw": historical,
               "comparison": comparison, "started_unix": started,
               "finished_unix": finished, "serve_pre": before, "serve_post": after,
               "source_sha256": {path: bc.hashlib.sha256((root / path).read_bytes()).hexdigest()
                   for path in ("prismaquant/boundary_control.py",
                                "prismaquant/validate_quantized_model.py",
                                "tools/measure_boundary_control.py",
                                "tools/serve_fingerprint.py")}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"receipt": str(output), "sha256": bc.digest(receipt),
                      "comparison": comparison, "fixed_point": measurement.get("fixed_point")}))
    return 0 if measurement.get("fixed_point", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
