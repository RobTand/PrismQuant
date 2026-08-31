#!/usr/bin/env python3
"""validate_native_export.py — load a compressed-tensors checkpoint via
vLLM and do a single forward + greedy decode. Binary check: either
vLLM accepts the format and produces tokens, or it doesn't.

Usage (from inside a vllm-node container):
    python -m prismaquant.validate_native_export \\
        --model dq-runs-new/qwen36-fresh/exported \\
        --prompt "The capital of France is" \\
        --max-new-tokens 16

The script can optionally upgrade the container's flashinfer to a
serving-profile-pinned version before loading; this is needed for some vLLM builds
that ship with a flashinfer that can't dispatch the NVFP4 MoE backend
on Blackwell. Pass `--no-flashinfer-upgrade` to skip.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


_DEFAULT_FLASHINFER_PACKAGES = ("flashinfer-python", "flashinfer-cubin")


def maybe_upgrade_flashinfer(
    version: str,
    *,
    package_names: tuple[str, ...] = _DEFAULT_FLASHINFER_PACKAGES,
    env: dict[str, str] | None = None,
) -> None:
    """Raise flashinfer-python and flashinfer-cubin to at least `version` and
    set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass the AOT-cache pin
    that lags behind PyPI. No-op if the installed version is already >= target.

    The profile version is a FLOOR, never an exact pin, and this comparison is
    the whole reason why. It was `== version`, which made a newer install look
    wrong and pip-installed the older one over it. Measured 2026-08-14 on the
    Qwen3.8-27B native-export gate: `gridbook:0.8.6-clean-dde15e0` ships
    flashinfer 0.6.18, the `vllm_packed_moe` profile pins 0.6.8.post1, so the
    gate DOWNGRADED a working container and the engine died with
    `ImportError: cannot import name 'set_autotune_process_group' from
    'flashinfer.autotuner'` — vLLM 0.26 needs the newer API. The pin's original
    purpose was the opposite problem (images too old to dispatch the NVFP4 MoE
    backend on Blackwell), and a floor satisfies that intent without being able
    to break a container that was already fine.
    """
    for key, value in (env or {"FLASHINFER_DISABLE_VERSION_CHECK": "1"}).items():
        os.environ.setdefault(key, value)

    def _parts(v: str) -> tuple[int, ...]:
        # "0.6.8.post1" -> (0, 6, 8); trailing .postN/.devN are ignored, so a
        # post-release never reads as older than its own base version.
        out = []
        for chunk in str(v).split("."):
            digits = "".join(c for c in chunk if c.isdigit())
            if not digits or not chunk[:1].isdigit():
                break
            out.append(int(digits))
        return tuple(out)

    try:
        import flashinfer
        installed = getattr(flashinfer, "__version__", "0.0")
        if installed == version or _parts(installed) >= _parts(version):
            print(f"[validate] flashinfer {installed} already satisfies the "
                  f"{version} floor — not touching it", flush=True)
            return
    except ImportError:
        pass
    package_specs = [f"{name}=={version}" for name in package_names]
    print(f"[validate] upgrading {', '.join(package_names)} to {version}",
          flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--upgrade", "-q",
        *package_specs,
    ])


def _flashinfer_runtime_package(target_profile: str):
    from .serving_profiles import load_serving_profile

    profile = load_serving_profile(target_profile)
    spec = profile.runtime_package("flashinfer")
    if spec is None:
        return (
            "0.6.8.post1",
            _DEFAULT_FLASHINFER_PACKAGES,
            {"FLASHINFER_DISABLE_VERSION_CHECK": "1"},
        )
    return (
        spec.version or "0.6.8.post1",
        spec.pip_packages or _DEFAULT_FLASHINFER_PACKAGES,
        spec.env_dict() or {"FLASHINFER_DISABLE_VERSION_CHECK": "1"},
    )


def _resolve_validation_target_profile(
    model_dir: str | Path,
    requested: str | None,
) -> str:
    """Resolve the serving profile for an exported checkpoint smoke."""
    from .model_profiles.registry import detect_profile
    from .serving_profiles import resolve_target_profile

    try:
        profile = detect_profile(str(model_dir))
    except Exception:
        profile = None
    return resolve_target_profile(
        profile,
        requested,
        default="vllm_packed_moe",
    )


def summarize_quantization_config(cfg_path: Path) -> None:
    cfg = json.load(open(cfg_path))
    qc = cfg.get("quantization_config", {})
    print(f"[validate] quant_method: {qc.get('quant_method', '<missing>')}")
    print(f"[validate] format:       {qc.get('format', '<missing>')}")
    for gn, g in qc.get("config_groups", {}).items():
        w = g.get("weights", {})
        print(f"[validate]   {gn}: bits={w.get('num_bits')} "
              f"strategy={w.get('strategy')} group={w.get('group_size')} "
              f"format={g.get('format')} n_targets={len(g.get('targets', []))}")
    print(f"[validate]   ignore: {len(qc.get('ignore', []))} entries")


def _speculative_config_uses_embedded_mtp(spec: dict) -> bool:
    method = str(spec.get("method") or "").lower()
    return method == "mtp" or method.endswith("_mtp")


def _run_arm(args, model_dir: Path, spec: dict | None, *,
             enforce_eager: bool) -> dict:
    """One load+generate smoke. Returns a shipcard-shaped verdict block."""
    arm = "eager" if enforce_eager else "graph"
    print(f"[validate] starting vLLM ({arm} arm) ...", flush=True)
    from vllm import LLM, SamplingParams

    llm = None
    try:
        llm = LLM(
            model=str(model_dir),
            quantization="compressed-tensors",
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            speculative_config=spec,
        )
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
        out = llm.generate([args.prompt], sp)
        print(f"[validate] generated ({arm}):", flush=True)
        texts = []
        for o in out:
            text = o.outputs[0].text
            texts.append(text)
            print(f"  prompt: {o.prompt!r}", flush=True)
            print(f"  output: {text!r}", flush=True)
        produced = sum(len(t) for t in texts)
        base_passed = produced > 0
        base_detail = (f"{arm}: generated {produced} chars"
                       if produced else f"{arm}: generated NOTHING")
        base_metrics = {"arm": arm, "generated_chars": produced,
                        "enforce_eager": enforce_eager,
                        "max_new_tokens": args.max_new_tokens}
        # --- WO-D D3: leg 2 — priced vs served route must agree ---
        # This is principle 14's second leg. The CB seam is
        # nvfp4_activation_contract.read_route / emit_route; trellis lanes
        # currently set gridbook_activation_contract but do not emit_route,
        # so served telemetry is absent and the gate must refuse for lack of
        # evidence rather than pass (finding recorded in WO-D-FINDINGS.md).
        try:
            priced = _load_priced_trellis_histograms(model_dir)
            served = _collect_served_trellis_histograms(llm)
            route_problems = verify_trellis_priced_vs_served(priced, served)
            if route_problems:
                print(f"[validate] trellis route leg2 FAILED ({arm}): {route_problems[0]}", flush=True)
                return {
                    "passed": False,
                    "detail": f"{arm}: trellis route mismatch: {route_problems[0]}",
                    "metrics": {**base_metrics, "trellis_route_problems": route_problems,
                                "priced_histograms": priced, "served_histograms": served},
                }
            # Also attach histograms on success for audit
            if priced:
                base_metrics["priced_histograms"] = priced
                base_metrics["served_histograms"] = served
        except Exception as exc:
            # Verification of the verification must not be assumed; fail closed
            print(f"[validate] trellis route check error ({arm}): {exc!r}", flush=True)
            return {
                "passed": False,
                "detail": f"{arm}: trellis route check error: {exc}",
                "metrics": {**base_metrics, "trellis_route_error": str(exc)},
            }
        return {
            "passed": base_passed,
            "detail": base_detail,
            "metrics": base_metrics,
        }
    except Exception as exc:
        print(f"[validate] {arm} arm FAILED: {exc!r}", flush=True)
        return {
            "passed": False,
            "detail": f"{arm}: {type(exc).__name__}: {exc}",
            "metrics": {"arm": arm, "enforce_eager": enforce_eager},
        }
    finally:
        # --both-arms holds two engines in one process otherwise; on a
        # unified-memory box the second load then profiles against a pool the
        # first one still owns.
        if llm is not None:
            try:
                del llm
                import gc

                gc.collect()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def _load_priced_trellis_histograms(model_dir: Path) -> dict:
    """Load the priced activation-contract / route histogram (D1) for trellis.

    The priced histogram is the artifact's claim about which routes it will
    take: ``selection_serving_lane_provenance.activation_contracts`` and the
    trellis-specific ``trellis_route_histogram``. It is the first leg of
    principle 14's attestation. The file may live in several places; we try
    them in order and return the first found.

    Returns a dict with at least ``activation_contracts`` and
    ``trellis_route_histogram`` keys, or empty dict if no priced trellis claim
    exists (non-trellis artifact).
    """
    candidates: list[Path] = [
        model_dir / "selection.json",
        model_dir / "quant_config.json",
        model_dir / "shipcard.json",
    ]
    # Also check parent work dir for selection.json (common layout)
    if model_dir.parent != model_dir:
        candidates.append(model_dir.parent / "selection.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        # Direct selection.json shape
        if isinstance(payload, dict) and "activation_contracts" in payload:
            # Heuristic: this looks like a serving_lane_provenance dict
            return {
                "activation_contracts": dict(payload.get("activation_contracts") or {}),
                "trellis_route_histogram": dict(payload.get("trellis_route_histogram") or {}),
                "trellis_units": list(payload.get("trellis_units") or []),
                "source": str(path),
            }
        # selection.json wrapped form: {"serving_lane_provenance": {...}}
        if isinstance(payload, dict) and isinstance(payload.get("serving_lane_provenance"), dict):
            prov = payload["serving_lane_provenance"]
            return {
                "activation_contracts": dict(prov.get("activation_contracts") or {}),
                "trellis_route_histogram": dict(prov.get("trellis_route_histogram") or {}),
                "trellis_units": list(prov.get("trellis_units") or []),
                "source": str(path),
            }
        # quant_config.json provenance
        if isinstance(payload, dict) and isinstance(payload.get("provenance"), dict):
            prov = payload["provenance"]
            # trellis_route_status lifted from selection
            if isinstance(prov.get("trellis_route_status"), dict):
                tr = prov["trellis_route_status"]
                return {
                    "activation_contracts": dict(tr.get("activation_contracts") or tr.get("trellis_route_histogram") or {}),
                    "trellis_route_histogram": dict(tr.get("trellis_route_histogram") or tr.get("route_histogram") or {}),
                    "trellis_units": list(tr.get("trellis_units") or []),
                    "source": str(path),
                }
            if isinstance(prov.get("selection_serving_lane_provenance"), dict):
                sel = prov["selection_serving_lane_provenance"]
                return {
                    "activation_contracts": dict(sel.get("activation_contracts") or {}),
                    "trellis_route_histogram": dict(sel.get("trellis_route_histogram") or {}),
                    "trellis_units": list(sel.get("trellis_units") or []),
                    "source": str(path),
                }
        # shipcard.json trellis_route_status
        if isinstance(payload, dict) and isinstance(payload.get("trellis_route_status"), dict):
            tr = payload["trellis_route_status"]
            return {
                "activation_contracts": dict(tr.get("activation_contracts") or {}),
                "trellis_route_histogram": dict(tr.get("trellis_route_histogram") or {}),
                "trellis_units": list(tr.get("trellis_units") or []),
                "source": str(path),
            }
        # Direct priced file with trellis_route_histogram top-level
        if isinstance(payload, dict) and isinstance(payload.get("trellis_route_histogram"), dict):
            return {
                "activation_contracts": dict(payload.get("activation_contracts") or {}),
                "trellis_route_histogram": dict(payload["trellis_route_histogram"]),
                "trellis_units": list(payload.get("trellis_units") or []),
                "source": str(path),
            }
    return {}


def _find_torch_model_from_llm(llm):
    """Best-effort walk from a vLLM LLM instance to its underlying nn.Module."""
    if llm is None:
        return None
    # Try direct attributes
    seen: set[int] = set()
    stack = [llm]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        # If this object itself has route telemetry, treat it as model
        try:
            import torch.nn as nn
            if isinstance(obj, nn.Module):
                # Check if any submodule has trellis/CB route state
                has_route = False
                for _, mod in obj.named_modules():
                    if hasattr(mod, "gridbook_activation_contract") or hasattr(mod, "_cb_route_state") or hasattr(mod, "_cb_route_kind"):
                        has_route = True
                        break
                if has_route:
                    return obj
        except Exception:
            pass
        for attr in ("model_executor", "driver_worker", "worker", "model_runner", "model", "engine_core", "engine", "llm_engine", "core"):
            try:
                child = getattr(obj, attr, None)
            except Exception:
                child = None
            if child is not None and not isinstance(child, (str, int, float, bool, bytes)):
                stack.append(child)
    return None


def _collect_served_trellis_histograms(llm) -> dict | None:
    """Collect the *served* activation-contract histogram via emit_route telemetry.

    Reuses the CB seam (``nvfp4_activation_contract.emit_route`` / ``read_route``)
    and the trellis lane's ``gridbook_activation_contract`` attribute set by
    ``trellis_e2m1_lane`` / ``trellis_e4m3_lane``. Both are plain Python scalars
    written onto the layer at dispatch time, so this is tensor-free and cannot
    perturb execution.

    Returns a dict like ``{"e2m1_group16_ue4m3_static": 2}`` or ``None`` if no
    trellis route telemetry exists at all (missing seam). The caller must treat
    ``None`` as lack of evidence and refuse closed, not as an empty histogram.
    """
    if llm is None:
        return None
    model = _find_torch_model_from_llm(llm)
    if model is None:
        # No model found → no telemetry
        return None
    served: dict[str, int] = {}
    found_any = False
    # Try CB-style read_route first (covers CB; trellis may also use it in future)
    try:
        from gridbook.nvfp4_activation_contract import read_route as _read_route
    except Exception:
        _read_route = None  # type: ignore
    for _, mod in model.named_modules():
        # Trellis lane telemetry: gridbook_activation_contract set at create_weights
        contract = getattr(mod, "gridbook_activation_contract", None)
        if isinstance(contract, str) and contract:
            found_any = True
            served[contract] = served.get(contract, 0) + 1
            continue
        if _read_route is not None:
            try:
                rec = _read_route(mod)
            except Exception:
                rec = None
            if rec is not None:
                found_any = True
                c = rec.get("contract")
                if isinstance(c, str) and c:
                    served[c] = served.get(c, 0) + 1
                else:
                    served["unknown"] = served.get("unknown", 0) + 1
    if not found_any:
        return None
    return served


def verify_trellis_priced_vs_served(
    priced: dict,
    served: dict | None,
) -> list[str]:
    """Compare priced (selection) vs served (emit_route) trellis histograms.

    This is principle 14's second leg: the priced contract and the served
    contract must be the same object. A disagreement is the 2026-08-17 defect
    (73.7% of a 92 GB body rode arch::Sm80 fallback recorded in selection.json
    and refused by nothing) reproduced for trellis.

    Returns a list of refusal reasons (empty = OK). A ``served is None`` when
    priced contains trellis units is itself a refusal: the comparison cannot be
    faked with an assumption.
    """
    problems: list[str] = []
    # Determine if priced contains any trellis claim.
    priced_hist = priced.get("trellis_route_histogram") if isinstance(priced, dict) else None
    priced_units = priced.get("trellis_units") if isinstance(priced, dict) else None
    has_priced_trellis = bool(priced_hist) or bool(priced_units)
    # Fallback: activation_contracts may contain trellis contracts
    if not has_priced_trellis and isinstance(priced, dict):
        act = priced.get("activation_contracts") or {}
        for k in act:
            if isinstance(k, str) and ("e2m1" in k or "fp8_per_token" in k):
                has_priced_trellis = True
                break
    if not has_priced_trellis:
        return []
    if served is None:
        # WO-D D3 finding: trellis lanes emit no route telemetry at all.
        problems.append(
            "trellis served route telemetry absent: the artifact's priced trellis_route_histogram "
            f"{priced_hist} has no corresponding emit_route records from the serve; this is a finding, "
            "not a pass — the comparison cannot be faked with an assumption, so the gate refuses for lack "
            "of evidence (WO-D D3 second leg). Trellis lanes set gridbook_activation_contract but do not "
            "call nvfp4_activation_contract.emit_route; the CB seam (read_route) is therefore empty for trellis."
        )
        return problems
    # Compare activation_contracts histograms.
    # For trellis, priced_hist keys are "family:contract:route_status", we extract contract counts.
    # Also compare direct activation_contracts dict if present.
    priced_contracts = priced.get("activation_contracts") if isinstance(priced, dict) else {}
    if isinstance(priced_contracts, dict) and priced_contracts:
        # served is already by contract
        if priced_contracts != served:
            problems.append(
                f"trellis priced vs served activation_contracts disagree: priced {dict(sorted(priced_contracts.items()))} "
                f"vs served {dict(sorted(served.items()))} — principle 14 leg 2 refuses (WO-D D3)"
            )
    elif priced_hist:
        # Derive priced contract counts from histogram keys
        from collections import Counter
        derived: Counter[str] = Counter()
        for key, cnt in priced_hist.items():
            # key format "family:contract:status"
            parts = str(key).split(":")
            if len(parts) >= 2:
                contract = parts[1]
            else:
                contract = str(key)
            if contract:
                derived[contract] += int(cnt)
        if dict(derived) != served:
            problems.append(
                f"trellis priced vs served histogram disagree: priced trellis_route_histogram {dict(sorted(priced_hist.items()))} "
                f"implies contracts {dict(derived)} vs served {dict(sorted(served.items()))}"
            )
    return problems


def _record_arm(args, model_dir: Path, spec: dict | None, verdict: dict) -> None:
    """Close the matching `native_export.*` shipcard slot."""
    if not args.shipcard:
        return
    from .shipcard import (
        compute_model_sha, fill_if_requested, git_provenance, make_record,
    )

    arm = verdict["metrics"]["arm"]
    try:
        model_sha = compute_model_sha(model_dir)
    except Exception:
        model_sha = None
    record = make_record(
        slot=f"native_export.{arm}",
        tool="validate_native_export.py",
        passed=bool(verdict["passed"]),
        model_sha=model_sha,
        metrics=verdict["metrics"],
        detail=verdict["detail"],
        spec_decode_detected=spec is not None,
        git_commit=git_provenance().get("commit"),
    )
    fill_if_requested(args.shipcard, f"native_export.{arm}", record)


def main():
    from .serving_profiles import serving_profile_names

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Compressed-tensors checkpoint directory.")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--target-profile", default=None,
                    choices=serving_profile_names(),
                    help="Serving profile whose runtime package pins should "
                         "be used for validation preflight. Defaults to the "
                         "exported model profile's configured serving "
                         "profile, then vllm_packed_moe.")
    ap.add_argument("--flashinfer-version", default=None,
                    help="Override the serving profile's flashinfer package "
                         "version before loading vLLM.")
    ap.add_argument("--no-flashinfer-upgrade", action="store_true",
                    help="Skip the flashinfer pre-flight upgrade.")
    ap.add_argument("--speculative-config", default=None,
                    help="JSON string for vLLM SpeculativeConfig. Use this "
                         "to exercise MTP heads, e.g. "
                         "'{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": 3, "
                         "\"model\": \"<same model dir>\"}'.")
    ap.add_argument("--no-enforce-eager", action="store_true",
                    help="Allow vLLM compile/CUDA-graph execution instead of "
                         "forcing eager mode. Use after the eager smoke passes.")
    ap.add_argument("--both-arms", action="store_true",
                    help="Run the eager arm AND the graph arm in one "
                         "invocation and fill both shipcard slots. The "
                         "two-arm rule used to live only in this help text; "
                         "this is it in code.")
    ap.add_argument("--shipcard", default=None,
                    help="Path to the artifact's shipcard.json; the arm's "
                         "verdict is appended to native_export.<arm> "
                         "(see python -m prismaquant.shipcard_cli).")
    args = ap.parse_args()

    model_dir = Path(args.model)
    target_profile = _resolve_validation_target_profile(
        model_dir,
        args.target_profile,
    )
    print(f"[validate] target profile: {target_profile}", flush=True)

    if not args.no_flashinfer_upgrade:
        version, package_names, env = _flashinfer_runtime_package(target_profile)
        if args.flashinfer_version:
            version = args.flashinfer_version
        maybe_upgrade_flashinfer(version, package_names=package_names, env=env)

    summarize_quantization_config(model_dir / "config.json")

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    spec = None
    if args.speculative_config:
        spec = json.loads(args.speculative_config)
        # If caller omitted "model", default to the same checkpoint for
        # draft-model style configs. MTP-family methods are the exception:
        # their extra heads travel with the target checkpoint, and vLLM expects
        # model to be absent/null so it can take the embedded-MTP path.
        if "model" not in spec and not _speculative_config_uses_embedded_mtp(spec):
            spec["model"] = str(model_dir)
        print(f"[validate] speculative config: {spec}", flush=True)

    if args.both_arms:
        arms = [True, False]
    else:
        arms = [not args.no_enforce_eager]

    verdicts = []
    for enforce_eager in arms:
        verdict = _run_arm(args, model_dir, spec, enforce_eager=enforce_eager)
        _record_arm(args, model_dir, spec, verdict)
        verdicts.append(verdict)

    failed = [v for v in verdicts if not v["passed"]]
    for verdict in verdicts:
        print(f"[validate] {'PASS' if verdict['passed'] else 'FAIL'} "
              f"{verdict['detail']}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
