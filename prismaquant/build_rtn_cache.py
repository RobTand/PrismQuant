#!/usr/bin/env python3
"""
Build an RTN cache (FP4 and FP8 weights) for a given model, without AutoRound.

For models that are already quantization-friendly at the RTN level (observed
to be the case for 27B+ where RTN-FP4 KL ≈ AutoRound-FP4 KL), AutoRound
provides negligible quality improvement and costs hours of compute + memory
for wrapper state that leaks.

This script:
  1. Loads the BF16 model
  2. Measures RTN-FP4 baseline KL (for manifest)
  3. For each Linear, computes RTN-FP4 and RTN-FP8 round-trips
  4. Saves both to the cache dir as sharded safetensors files
  5. Writes cache_manifest.json compatible with dpq_autoround_first.py's cache format

Memory discipline: extract and save in chunks to avoid holding the full 2×
weight dict in memory at once. For 27B this means we never exceed ~60GB unified
memory usage even during the save.

Usage:
    python3 build_rtn_cache.py \\
        --model /models/Qwen3.5-27B-bf16 \\
        --cache-dir /tmp/dpq_cache/qwen35-27b-rtn \\
        --n-calib-samples 8 --calib-seqlen 512
"""
import argparse
import gc
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.sensitivity_probe import (
    _is_packed_experts_module,
    _packed_experts_param_names,
)


def _fp8_round(weight: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 round-trip with per-output-channel scale."""
    max_abs = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 448.0
    return ((weight / scale).to(torch.float8_e4m3fn).to(weight.dtype)) * scale


def _nvfp4_round_rtn(weight: torch.Tensor, group_size: int = 16) -> torch.Tensor:
    """Per-group NVFP4 (E2M1) round-to-nearest."""
    out_f, in_f = weight.shape
    n_groups = (in_f + group_size - 1) // group_size
    pad = n_groups * group_size - in_f
    w = F.pad(weight, (0, pad)) if pad > 0 else weight
    grouped = w.view(out_f, n_groups, group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = grouped / scales * 6.0
    abs_n = normalized.abs()
    sign = normalized.sign()
    q = torch.where(abs_n <= 0.25, torch.zeros_like(abs_n),
        torch.where(abs_n <= 0.75, torch.full_like(abs_n, 0.5),
        torch.where(abs_n <= 1.25, torch.full_like(abs_n, 1.0),
        torch.where(abs_n <= 1.75, torch.full_like(abs_n, 1.5),
        torch.where(abs_n <= 2.5,  torch.full_like(abs_n, 2.0),
        torch.where(abs_n <= 3.5,  torch.full_like(abs_n, 3.0),
        torch.where(abs_n <= 5.0,  torch.full_like(abs_n, 4.0),
                                   torch.full_like(abs_n, 6.0))))))))
    dequant = sign * q / 6.0 * scales
    dequant = dequant.view(out_f, n_groups * group_size)
    if pad > 0:
        dequant = dequant[:, :in_f]
    return dequant.to(weight.dtype)


def stage_multimodal(model_path: str):
    """Stage a model for CausalLM loading.

    Handles:
    - VL models (Qwen3.5, Gemma 4): strips vision config, remaps tensor names
    - FP8 pre-quantized models (MiniMax): strips quantization_config so the
      model loads as raw tensors without requiring a GPU quantizer

    Returns (staged_path, cleanup_path).  If no staging needed,
    returns (model_path, None).
    """
    src_cfg_path = Path(model_path) / "config.json"
    if not src_cfg_path.exists():
        return model_path, None
    with open(src_cfg_path) as f:
        cfg = json.load(f)

    needs_staging = (
        "vision_config" in cfg
        or "text_config" in cfg
        or "quantization_config" in cfg
    )
    if not needs_staging:
        return model_path, None

    # Strip quantization_config (e.g. fp8) so the model loads as raw tensors
    if "quantization_config" in cfg:
        qc = cfg.pop("quantization_config")
        print(f"[stage] stripped quantization_config: {qc.get('quant_method', '?')}")

    if "vision_config" not in cfg and "text_config" not in cfg:
        # Just needed to strip quant config — no VL remapping
        staged = tempfile.mkdtemp(prefix="rtn_staged_")
        for p in Path(model_path).iterdir():
            if p.name == "config.json":
                continue
            (Path(staged) / p.name).symlink_to(p.resolve())
        with open(Path(staged) / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        return staged, staged

    # --- Build CausalLM config from the VL config ---
    for k in ["vision_config", "image_token_id", "video_token_id",
              "vision_start_token_id", "vision_end_token_id",
              "audio_config", "audio_token_id", "boa_token_id",
              "boi_token_id", "eoa_token_id", "eoa_token_index",
              "eoi_token_id", "vision_soft_tokens_per_image"]:
        cfg.pop(k, None)
    if "text_config" in cfg:
        text_cfg = cfg.pop("text_config")
        for k, v in text_cfg.items():
            if k not in cfg:
                cfg[k] = v
        if "model_type" in text_cfg:
            cfg["model_type"] = text_cfg["model_type"]
    archs = cfg.get("architectures", [])
    if archs:
        cfg["architectures"] = [
            a.replace("ForConditionalGeneration", "ForCausalLM") for a in archs
        ]

    staged = tempfile.mkdtemp(prefix="rtn_staged_")

    # --- Remap safetensors index: strip language_model prefix, drop vision ---
    idx_path = Path(model_path) / "model.safetensors.index.json"
    if idx_path.exists():
        with open(idx_path) as f:
            idx = json.load(f)

        # Detect the prefix pattern used for the language model
        # Common patterns: "model.language_model." (Qwen3.5, Gemma4)
        lm_prefix = None
        for name in idx["weight_map"]:
            if "language_model." in name:
                # e.g. "model.language_model.layers.0..." → prefix is "model.language_model."
                lm_prefix = name[:name.index("language_model.")]  + "language_model."
                break

        if lm_prefix:
            new_weight_map = {}
            for name, shard in idx["weight_map"].items():
                if name.startswith(lm_prefix):
                    # model.language_model.layers.0.X → model.layers.0.X
                    new_name = "model." + name[len(lm_prefix):]
                    new_weight_map[new_name] = shard
                elif "vision" not in name and "visual" not in name and "embed_vision" not in name:
                    # Keep non-vision, non-language_model tensors (e.g. lm_head)
                    new_weight_map[name] = shard
            idx["weight_map"] = new_weight_map

            # Write remapped index
            with open(Path(staged) / "model.safetensors.index.json", "w") as f:
                json.dump(idx, f, indent=2)

            # Symlink everything EXCEPT config.json and the index
            for p in Path(model_path).iterdir():
                if p.name in ("config.json", "model.safetensors.index.json"):
                    continue
                (Path(staged) / p.name).symlink_to(p.resolve())
        else:
            # No language_model prefix — just symlink everything
            for p in Path(model_path).iterdir():
                if p.name == "config.json":
                    continue
                (Path(staged) / p.name).symlink_to(p.resolve())
    else:
        for p in Path(model_path).iterdir():
            if p.name == "config.json":
                continue
            (Path(staged) / p.name).symlink_to(p.resolve())

    with open(Path(staged) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    return staged, staged


# Minimal skip list: ONLY things that are genuinely not weight matrices.
# Everything that IS a weight matrix (including lm_head, MTP heads, MoE
# routers) is a candidate for quantization. HAWQ will find the right
# bit allocation — sensitive params get more bits, insensitive get fewer.
ALWAYS_SKIP_PATTERNS = [
    "embed_tokens",     # nn.Embedding, not nn.Linear — different semantics
    "norm",             # RMSNorm/LayerNorm — not a weight matrix
    "A_log",            # state-space scalar parameter
    "dt_bias",          # state-space bias
    "conv1d",           # Conv1d — needs different quantization approach
    "visual", "vision", # vision tower — stripped from ForCausalLM anyway
]


def should_always_skip(name: str) -> bool:
    import re
    for pattern in ALWAYS_SKIP_PATTERNS:
        if pattern.endswith("$"):
            if re.search(pattern, name):
                return True
        elif pattern in name:
            return True
    return False


def is_fused_moe_experts(module, profile=None) -> bool:
    """Legacy fallback for profile-declared fused-experts containers."""
    getter = getattr(profile, "packed_expert_module_class_names", None)
    if not callable(getter):
        return False
    try:
        names = getter()
    except Exception:
        return False
    return type(module).__name__ in set(str(name) for name in names)


def _packed_expert_param_names_for_module(module, profile=None) -> list[str]:
    names = _packed_experts_param_names(module, profile)
    if names:
        return names
    if not is_fused_moe_experts(module, profile):
        return []
    if profile is None:
        try:
            from prismaquant.model_profiles import DefaultProfile
            profile = DefaultProfile()
        except Exception:
            profile = None
    if profile is not None:
        try:
            return sorted(profile.packed_expert_param_names())
        except Exception:
            pass
    return []


def iter_quantizable_tensors(model, profile=None):
    """Yield (full_name, module, param_attr) for each quantizable weight.
    Caller uses module.{param_attr}.data to read, and can REPLACE the data
    attribute to free storage. Handles nn.Linear and 3D packed MoE experts."""
    if profile is None:
        try:
            from prismaquant.model_profiles import profile_from_model
            profile = profile_from_model(model)
        except Exception:
            profile = None
    for name, mod in model.named_modules():
        # Regular nn.Linear
        if isinstance(mod, nn.Linear) and mod.weight.numel() >= 1000:
            if should_always_skip(name):
                continue
            yield (f"{name}.weight", mod, "weight")
        # Packed MoE experts. The profile/spec owns the accepted parameter
        # names; legacy class-name fallback only exists for old runtimes.
        elif (
            _is_packed_experts_module(mod, profile)
            or is_fused_moe_experts(mod, profile)
        ):
            param_names = _packed_expert_param_names_for_module(mod, profile)
            for param_name in param_names:
                if not hasattr(mod, param_name):
                    continue
                param = getattr(mod, param_name)
                if not isinstance(param, torch.nn.Parameter):
                    continue
                if param.numel() < 1000:
                    continue
                full_name = f"{name}.{param_name}"
                if should_always_skip(full_name):
                    continue
                yield (full_name, mod, param_name)


def rtn_fp4_any_shape(tensor: torch.Tensor) -> torch.Tensor:
    """RTN-FP4 for either 2D (nn.Linear) or 3D (fused MoE experts) weights.
    For 3D (E, out, in), flattens experts into rows: (E*out, in)."""
    if tensor.dim() == 2:
        return _nvfp4_round_rtn(tensor)
    elif tensor.dim() == 3:
        E, out_f, in_f = tensor.shape
        flat = tensor.reshape(E * out_f, in_f)
        quant = _nvfp4_round_rtn(flat)
        return quant.reshape(E, out_f, in_f)
    raise ValueError(f"unexpected weight ndim={tensor.dim()}, shape={tensor.shape}")


def rtn_fp8_any_shape(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return _fp8_round(tensor)
    elif tensor.dim() == 3:
        E, out_f, in_f = tensor.shape
        flat = tensor.reshape(E * out_f, in_f)
        quant = _fp8_round(flat)
        return quant.reshape(E, out_f, in_f)
    raise ValueError(f"unexpected weight ndim={tensor.dim()}, shape={tensor.shape}")


# ---------------------------------------------------------------------------
# Calibration + KL measurement
# ---------------------------------------------------------------------------

def load_wikitext_calibration(
    tokenizer,
    n_samples: int,
    seqlen: int,
    *,
    split: str = "train",
    seed: int = 42,
) -> torch.Tensor:
    from .calibration_data import load_wikitext2_raw
    ds = load_wikitext2_raw(split=split)
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    enc = tokenizer(text, return_tensors="pt", truncation=False).input_ids
    total = enc.size(1)
    max_start = total - seqlen
    import random
    rng = random.Random(int(seed))
    starts = rng.sample(range(max_start), n_samples) if max_start >= n_samples else \
             [i * (max_start // n_samples) for i in range(n_samples)]
    batches = torch.stack([enc[0, s:s + seqlen] for s in starts], dim=0)
    return batches


# NOTE: must cast logits to fp32 BEFORE log_softmax — bf16 has catastrophic
# precision loss in log_softmax (small probs round to 0, log(0) = -inf,
# KL blows up to ~50).
@torch.no_grad()
def cache_reference_log_probs(
    model,
    calib_ids,
    device,
    *,
    kl_scope: str = "full_sequence",
):
    if kl_scope not in {"last_token", "full_sequence"}:
        raise ValueError(
            "kl_scope must be 'last_token' or 'full_sequence', "
            f"got {kl_scope!r}"
        )
    log_probs = []
    for i in range(calib_ids.size(0)):
        batch = calib_ids[i:i + 1].to(device)
        logits = model(batch).logits
        if kl_scope == "last_token":
            logits = logits[:, -1:, :]
        log_probs.append(F.log_softmax(logits.float(), dim=-1))  # fp32!
    return log_probs


def kl_divergence(student_logits, teacher_log_probs):
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)  # fp32!
    teacher_probs = teacher_log_probs.exp()
    kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return kl.mean()


@torch.no_grad()
def measure_kl(model, calib_ids, ref_log_probs, device) -> float:
    kls = []
    for i in range(calib_ids.size(0)):
        batch = calib_ids[i:i + 1].to(device)
        logits = model(batch).logits
        kls.append(kl_divergence(logits, ref_log_probs[i]).item())
    return sum(kls) / len(kls)


# ---------------------------------------------------------------------------
# Chunked save — the KEY to avoiding OOM during save
# ---------------------------------------------------------------------------

def save_weights_chunked(out_path: Path, weights: Dict[str, torch.Tensor],
                         chunk_size_gb: float = 8.0):
    """Save a weight dict as multiple .safetensors files, streaming one chunk
    at a time. Frees each chunk's tensors from the input dict after saving
    so peak memory never exceeds (remaining_dict + one_chunk)."""
    from safetensors.torch import save_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Partition keys into size-balanced chunks
    chunk_bytes = int(chunk_size_gb * 1024 ** 3)
    keys = list(weights.keys())
    shards: List[List[str]] = []
    current: List[str] = []
    current_bytes = 0
    for k in keys:
        sz = weights[k].numel() * weights[k].element_size()
        if current and current_bytes + sz > chunk_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(k)
        current_bytes += sz
    if current:
        shards.append(current)

    n_shards = len(shards)
    print(f"  saving {len(keys)} tensors across {n_shards} shards", flush=True)
    for i, shard_keys in enumerate(shards):
        shard_path = out_path.parent / f"{out_path.stem}-{i:02d}-of-{n_shards:02d}.safetensors"
        chunk = {k: weights[k] for k in shard_keys}
        save_file(chunk, str(shard_path))
        # Free chunk's tensors from both the chunk dict AND the parent dict
        for k in shard_keys:
            del weights[k]
        del chunk
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  shard {i+1}/{n_shards} written: {shard_path.name}", flush=True)


def load_weights_chunked(path_prefix: str) -> Dict[str, torch.Tensor]:
    """Load a sharded cache (new format) or single-file cache (legacy)."""
    from safetensors.torch import load_file
    p = Path(path_prefix)
    parent = p.parent
    stem = p.stem  # e.g. "fp4_weights"
    shards = sorted(parent.glob(f"{stem}-*-of-*.safetensors"))
    if shards:
        merged: Dict[str, torch.Tensor] = {}
        for s in shards:
            merged.update(load_file(str(s)))
        return merged
    if p.exists():
        return load_file(str(p))
    raise FileNotFoundError(f"no shards or single file at {path_prefix}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--n-calib-samples", type=int, default=8)
    parser.add_argument("--calib-seqlen", type=int, default=512)
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--calib-seed", type=int, default=42)
    parser.add_argument("--chunk-size-gb", type=float, default=8.0,
                        help="Max size per safetensors shard (GB)")
    args = parser.parse_args()

    t_start = time.time()

    staged, cleanup = stage_multimodal(args.model)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[rtn-cache] loading {staged}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            staged, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True)
        device = next(model.parameters()).device
        try:
            from prismaquant.model_profiles import detect_profile
            profile = detect_profile(args.model)
        except Exception:
            profile = None
        print(f"[rtn-cache]   {sum(p.numel() for p in model.parameters()):,} params", flush=True)

        # Stage 1: reference log probs
        calib_ids = load_wikitext_calibration(
            tokenizer,
            args.n_calib_samples,
            args.calib_seqlen,
            split=args.calib_split,
            seed=args.calib_seed,
        )
        print(f"[rtn-cache] computing BF16 reference log_probs", flush=True)
        ref_log_probs = cache_reference_log_probs(model, calib_ids, device)

        def run_pass(quantizer_name, quantize_fn):
            """Apply quantizer, measure KL, then stream-extract: build one
            chunk's worth of weights at a time, save to disk, free, repeat.
            Peak dict size never exceeds chunk_size_gb (~6GB) instead of
            the full model size."""
            from safetensors.torch import save_file
            nonlocal model
            print(f"[rtn-cache] RTN-{quantizer_name} pass: quantize in place", flush=True)
            n_quantized = 0
            for full_name, mod, attr in iter_quantizable_tensors(model, profile):
                t = getattr(mod, attr).data
                q = quantize_fn(t)
                getattr(mod, attr).data.copy_(q)
                n_quantized += 1
            print(f"[rtn-cache]   quantized {n_quantized} tensors", flush=True)

            kl = measure_kl(model, calib_ids, ref_log_probs, device)
            print(f"[rtn-cache]   RTN-{quantizer_name} KL vs BF16 = {kl:.6f}", flush=True)

            # Streaming extract-and-save. For each tensor:
            #   (a) clone its data to CPU dict
            #   (b) replace model's param.data with tiny placeholder (frees old)
            # When the dict hits chunk_size_gb, flush to a shard file and clear.
            print(f"[rtn-cache]   streaming extract+save", flush=True)
            cache_root = Path(args.cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            shard_dict: Dict[str, torch.Tensor] = {}
            shard_bytes = 0
            chunk_bytes = int(args.chunk_size_gb * 1024 ** 3)
            shard_idx = 0
            tmp_shard_paths: List[Path] = []
            base_name = f"{quantizer_name.lower()}_weights"

            def flush_shard():
                nonlocal shard_dict, shard_bytes, shard_idx
                if not shard_dict:
                    return
                # Use a temporary name — we'll rename at the end once we know
                # total shard count.
                tmp_path = cache_root / f"{base_name}-tmp-{shard_idx:02d}.safetensors"
                save_file(shard_dict, str(tmp_path))
                print(f"  shard {shard_idx} written ({len(shard_dict)} tensors, {shard_bytes / 1e9:.1f} GB): {tmp_path.name}",
                      flush=True)
                tmp_shard_paths.append(tmp_path)
                shard_dict.clear()
                shard_bytes = 0
                shard_idx += 1
                gc.collect()
                torch.cuda.empty_cache()

            for full_name, mod, attr in iter_quantizable_tensors(model, profile):
                param = getattr(mod, attr)
                t = param.data.cpu().clone()
                shard_dict[full_name] = t
                shard_bytes += t.numel() * t.element_size()
                # REPLACE param.data to drop refcount on original storage
                param.data = torch.zeros(1, dtype=param.dtype, device=param.device)

                if shard_bytes >= chunk_bytes:
                    flush_shard()

            flush_shard()  # final partial shard

            # Rename shards with the correct N-of-M suffix
            n_shards = len(tmp_shard_paths)
            for i, tmp in enumerate(tmp_shard_paths):
                final = cache_root / f"{base_name}-{i:02d}-of-{n_shards:02d}.safetensors"
                tmp.rename(final)
            print(f"[rtn-cache]   {quantizer_name} cache saved across {n_shards} shards", flush=True)

            # Now model is mostly empty — delete before reload
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return kl

        # Stage 2a: RTN-FP4
        rtn_fp4_kl = run_pass("fp4", rtn_fp4_any_shape)

        # Stage 2b: RTN-FP8 — reload model fresh, then same flow
        print(f"[rtn-cache] reloading fresh model for FP8 pass", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            staged, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
        )
        print(f"[rtn-cache]   {sum(p.numel() for p in model.parameters()):,} params", flush=True)
        rtn_fp8_kl = run_pass("fp8", rtn_fp8_any_shape)

        # Write manifest
        manifest = {
            "source_model": args.model,
            "cache_type": "rtn",
            "quantizer": "rtn (no autoround)",
            "rtn_fp4_baseline_kl": rtn_fp4_kl,
            "autoround_fp4_kl": rtn_fp4_kl,  # aliased so apply_role_recipe can read both
            "autoround_fp8_kl": rtn_fp8_kl,
            "elapsed_sec": time.time() - t_start,
        }
        with open(Path(args.cache_dir) / "cache_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"[rtn-cache] done in {time.time() - t_start:.0f}s", flush=True)
        print(f"[rtn-cache]   fp4 KL = {rtn_fp4_kl:.6f}", flush=True)
        print(f"[rtn-cache]   fp8 KL = {rtn_fp8_kl:.6f}", flush=True)
    finally:
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)


if __name__ == "__main__":
    main()
