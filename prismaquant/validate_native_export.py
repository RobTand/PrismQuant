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

#: The quantization this gate loads with (``LLM(..., quantization=...)`` in
#: `_run_arm`). A module constant so the lane check below reads the same value
#: the load uses, rather than a second spelling of it that can drift.
NATIVE_QUANTIZATION = "compressed-tensors"


def _artifact_quant_method(model_dir: str | Path) -> str | None:
    """The artifact's declared ``quantization_config.quant_method``, if any.

    Reads the top-level ``config.json`` entry and the multimodal nesting
    under ``text_config`` -- the same two places the streaming loader reads
    (``layer_streaming``) -- so a Tessera ``text_config`` cannot walk past.
    """
    try:
        cfg = json.loads(Path(model_dir, "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cfg, dict):
        return None
    candidates = [cfg.get("quantization_config")]
    nested = cfg.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested.get("quantization_config"))
    for qc in candidates:
        if isinstance(qc, dict) and qc.get("quant_method"):
            return str(qc["quant_method"])
    return None


def require_native_lane_artifact(model_dir: str | Path) -> str | None:
    """Refuse an artifact this gate cannot load, naming the lane that can.

    This is the NATIVE lane's load gate, not a lane-generic one -- exactly as
    the GGUF lane has its own ``llama-completion`` smoke instead
    (RobTand/prismaquant#119). Loading a foreign-lane artifact with
    ``quantization="compressed-tensors"`` dies inside vLLM on bytes it was
    never going to dispatch; refusing up front says which gate to run
    instead. Returns the declared method on pass (``None`` when the artifact
    declares none, preserving the dense-smoke behavior).
    """
    found = _artifact_quant_method(model_dir)
    if found is not None and found != NATIVE_QUANTIZATION:
        raise SystemExit(
            f"[validate] REFUSE: {model_dir} declares "
            f"quantization_config.quant_method={found!r}, but this gate loads "
            f"with quantization={NATIVE_QUANTIZATION!r}: it is the "
            f"compressed-tensors lane's load smoke, not a lane-generic one. "
            f"A {found!r} artifact is served by its own lane and its own "
            f"gates (see `python -m prismaquant.lane_spec --help`); running "
            f"it here would fail inside vLLM on bytes this quantization "
            f"was never going to dispatch.")
    return found


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
    """Resolve the serving profile for an exported checkpoint smoke.

    A profile this build cannot name falls back to the `default=` target below.
    `DeadVendoredOverrideError` does not (#201): it says the architecture is
    known and its modelling path is dead, and a validator that answers by
    fallback on a checkpoint it was just told it cannot reason about is
    validating the wrong thing.
    """
    from .model_profiles.registry import DeadVendoredOverrideError, detect_profile
    from .serving_profiles import resolve_target_profile

    try:
        profile = detect_profile(str(model_dir))
    except DeadVendoredOverrideError:
        raise
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
            quantization=NATIVE_QUANTIZATION,
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
        return {
            "passed": produced > 0,
            "detail": (f"{arm}: generated {produced} chars"
                       if produced else f"{arm}: generated NOTHING"),
            "metrics": {"arm": arm, "generated_chars": produced,
                        "enforce_eager": enforce_eager,
                        "max_new_tokens": args.max_new_tokens},
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

    # The lane check runs before any vLLM import: a foreign-lane artifact
    # must refuse here, not inside the loader.
    require_native_lane_artifact(model_dir)

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
