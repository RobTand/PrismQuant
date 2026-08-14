"""Aura cost: KL-adjoint per-Linear sensitivity surrogate.

Produces an allocator-compatible ``cost.pkl`` whose per-(Linear, format)
``predicted_dloss`` is the second-order KL contribution of quantizing that
Linear, measured against the **KL/Gauss-Newton Fisher** (not the CE empirical
Fisher) and the **production-rendered** weight error:

    predicted_dloss[i, f] = 0.5 * mean_k ( <gW_i^(k), dW_{i,f}> )^2

    gW_i^(k) = d/dW_i [ fisher_probe_scalar(logits; seed=k) ]   (kl_fisher probe;
               E_k[gW_i gW_i^T] = the layer Fisher w.r.t. the model KL)
    dW_{i,f} = Q_f(W_i) - W_i  (production-rendered error from ProductionWeight
               Cache when available, else the format-registry RTN error)

Why this is the right cost (rung-0 validated, 2026-06-04):
  * end-KL is locally a Fisher quadratic in the logit displacement, and the
    per-Linear unary KLs are **additive in fp32** (cross-terms ~0), so summing
    these per-Linear costs is a faithful end-KL surrogate -- the additive
    knapsack is sound once each per-Linear term is the KL-Fisher quantity.
  * <gW_i^(k), dW> = r_k . (J_i dY_i) is the probe projection of the propagated
    logit displacement; 0.5*mean_k(.)^2 is the unbiased estimator of
    0.5 * dY_i^T (J_i^T F J_i) dY_i = the unary KL contribution.
  * This is the analytic O(N) generalization of the validated 35B serving-unit
    propagated-sensitivity win (no hand-tuned scale, covers all Linears).

Reuses kl_fisher (probe), ProductionWeightCache (dW), format_registry (RTN
fallback), schemas (cost.pkl contract). Sets output_mse_measured=False so
allocator_candidates.cost_entry_predicted_dloss consumes predicted_dloss
directly. Measurement defaults to fp32 (the precision the additivity result
requires); memory-safe (one autograd graph at a time, watchdog-gated).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

import prismaquant.format_registry as fr
from prismaquant.kl_fisher import (
    fisher_probe_scalar,
    select_token_scope,
    token_count_for_logits,
)
from prismaquant.perturbed_x_cache import calibration_data_hash
from prismaquant.nvfp4_cb_footprint import (
    cb_cost_provenance,
    is_cb_format,
)
from prismaquant.production_render_cost import (
    _cache_render_score_records,
    _lookup_record as _lookup_render_score_record,
)
from prismaquant.render_score import persisted_cell_score_fields

SCHEMA = "prismaquant.aura_cost.v1"


def _git_commit() -> str | None:
    """Best-effort commit of the prismaquant tree this cost was computed by."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None

# Passthrough formats -> zero predicted_dloss. This is the *passthrough rule*
# (see allocator_candidates.PASSTHROUGH_SOURCE_REQUIREMENTS): zero cost is
# correct only when the source weight already has the target precision --
#   BF16        is lossless iff the source weight dtype is bf16 (or lower);
#   FP8_SOURCE  is lossless iff the source weight is native fp8 (verbatim copy).
# Production models load bf16, so BF16 here is a true passthrough (0 error) and
# the zero-cost is exact. The only unsafe case is an fp32-source model loaded
# with --dtype float32: then BF16 is a *downcast* (~half a bf16-ulp of error),
# not a passthrough, and the unconditional zero would let the allocator pick
# BF16 as "free" when it is not. That case is opt-in guarded by
# compute_aura_cost(assert_bf16_passthrough=True); the default stays a no-op so
# the documented bit-identical regression output is unchanged. FP8_SOURCE has
# no source tensor in a bf16/fp32-loaded model, so its legality is gated by the
# allocator's passthrough-integrity check, not here; aura only declines to
# double-count it.
_ZERO_COST_FORMATS = {"BF16", "FP8_SOURCE"}


def _resolve_auto_dtype(
    staged: str | Path,
    min_free_gib: float,
    available_bytes: int | None = None,
) -> str:
    """Pick float32 when the fp32-resident model fits, else bfloat16.

    fp32 is the additivity-preferred cost regime (per-Linear KLs add in
    fp32; cross-terms vanish), but the model loads FULLY RESIDENT here, so
    on a unified-memory box the choice must be sized, not assumed: a 35B at
    fp32 is ~140 GiB against a 121 GiB pool — an OOM-kill mid-pipeline.
    Sizing is from the checkpoint itself: bytes/param inferred from the
    index (fp8 sources carry weight_scale_inv sidecars and are 1 byte/param;
    bf16/fp16 are 2), headroom is the caller's --min-free-gib knob.
    """
    import json as _json

    src = Path(staged)
    total_bytes = 0
    bytes_per_param = 2.0
    idx = src / "model.safetensors.index.json"
    if idx.is_file():
        try:
            payload = _json.loads(idx.read_text())
            total_bytes = int(payload.get("metadata", {}).get("total_size", 0))
            if any(
                k.endswith(".weight_scale_inv")
                for k in payload.get("weight_map", {})
            ):
                bytes_per_param = 1.0
        except Exception:
            total_bytes = 0
    if not total_bytes:
        total_bytes = sum(
            f.stat().st_size for f in src.glob("*.safetensors"))
    if not total_bytes:
        _log("--dtype auto: could not size the checkpoint; keeping float32")
        return "float32"
    approx_params = total_bytes / bytes_per_param
    fp32_need = approx_params * 4
    if available_bytes is None:
        available_bytes = 0
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        available_bytes = int(line.split()[1]) * 1024
                        break
        except Exception:
            pass
    fits = (
        available_bytes > 0
        and fp32_need + min_free_gib * 1024**3 <= available_bytes
    )
    choice = "float32" if fits else "bfloat16"
    _log(
        f"--dtype auto: fp32-resident needs ~{fp32_need / 1024**3:.0f} GiB "
        f"(+{min_free_gib:.0f} GiB headroom) vs "
        f"{available_bytes / 1024**3:.0f} GiB available -> {choice}")
    return choice


def _log(msg: str) -> None:
    print(f"[aura {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _free_gib() -> float:
    """Reclaimable-inclusive free memory in GiB.

    On the GB10/DGX Spark unified-memory box, CUDA and host share one physical
    pool, and clean page cache (model safetensors, cache shards) counts as
    'used' in ``torch.cuda.mem_get_info()`` even though the kernel reclaims it
    on demand. ``/proc/meminfo`` ``MemAvailable`` is the true 'can still
    allocate' headroom and is what should gate the watchdog -- gating on CUDA
    free aborts spuriously whenever a large file was just read. Fall back to
    the CUDA figure off-Linux / if /proc is unreadable."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)  # kB -> GiB
    except Exception:
        pass
    try:
        return torch.cuda.mem_get_info()[0] / (1024 ** 3)
    except Exception:
        return float("inf")


def _target_linears(
    model: nn.Module, *, include_lm_head: bool = False,
) -> dict[str, nn.Linear]:
    """Quantizable nn.Linear targets. lm_head is EXCLUDED by default (the
    profile pins it BF16). include_lm_head adds it so Aura can MEASURE its
    KL-sensitivity and let the allocator choose its format as a budget
    decision rather than a hardcoded pin -- the KL probe gradient flows
    directly into lm_head (it produces the logits), so its cost is
    measured the same way as any body Linear."""
    out: dict[str, nn.Linear] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "lm_head" in name and not include_lm_head:
            continue
        if mod.weight.dim() == 2 and min(mod.weight.shape) >= 16:
            out[name] = mod
    return out


def _delta_w(
    name: str,
    fmt: str,
    weight: torch.Tensor,
    cache: object | None,
    *,
    strict: bool = False,
) -> tuple[torch.Tensor, str] | None:
    """Q_f(W)-W plus its provenance: ``(delta, "rendered"|"rtn")``.

    "rendered" = production-rendered error from the cache (the bytes export
    ships); "rtn" = format-registry RTN fallback. The distinction is recorded
    per cost row because it is result-changing: RTN-vs-rendered dW moved FP8
    allocations by +36% served KL (2026-06 A/B). ``strict``
    (require_production_cache): when a cache is supplied but lacks the rendered
    (name, fmt), fail fast with a clear coverage error instead of silently
    falling back to RTN -- so a 'production-faithful' run cannot quietly mix
    RTN deltas into the cost. Default off preserves the RTN fallback used by
    non-production ablations."""
    if cache is not None:
        try:
            rendered = cache.get(name, fr.canonical_format_name(fmt))
        except Exception:
            rendered = None
        if rendered is not None:
            delta = rendered.to(weight.device, torch.float32) - weight.float()
            return delta, "rendered"
        if strict:
            raise RuntimeError(
                f"require_production_cache: production-rendered weight missing "
                f"for ({name!r}, {fmt!r}); refusing silent RTN fallback. Build the "
                f"cache for this (Linear, format) or drop --require-production-cache.")
    spec = fr.get_format(fmt)
    if is_cb_format(spec.name):
        raise RuntimeError(
            f"{name}={spec.name}: AURA CB delta requires a production-cache "
            "render with the production col_weights/codebook contract; "
            "refusing the unweighted direct fallback"
        )
    qdq = getattr(spec, "quantize_dequantize", None)
    if qdq is None:
        return None
    try:
        rendered = qdq(weight.float())
        return rendered - weight.float(), "rtn"
    except Exception:
        return None


def _auto_n_chunks(
    linears: dict[str, nn.Linear],
    names: Sequence[str],
    min_free_gib: float,
    *,
    n_nonzero_fmts: int = 1,
    dw_bytes: int = 2,
    accurate_chunk_bytes: bool = False,
    hook_harvest: bool = False,
) -> int:
    """Pick the number of Linear chunks so peak memory stays under budget.

    Per chunk we hold dW_chunk (one bf16 delta per *nonzero* format, ~W/G each)
    + retained grads (one per weight at the model's param dtype, ~W/G) on top of
    the resident model, where W is the chunk's target-weight footprint. We size
    G so the per-chunk peak fits in (free - headroom), headroom covering the
    autograd graph and the watchdog floor. G=1 reproduces the legacy
    single-pass path exactly.

    ``_free_gib`` reads ``/proc/meminfo`` ``MemAvailable``, the correct 'can
    still allocate' signal on this GB10/DGX Spark *unified*-memory box (CUDA and
    host share one physical pool). On a *discrete* GPU MemAvailable is host RAM
    only and says nothing about VRAM headroom -- this sizing would be wrong
    there and would have to gate on ``torch.cuda.mem_get_info`` instead.

    Legacy (default) accounting hardcodes 2 bytes/weight and a single ~W/G dW
    term -- it silently assumes a bf16 model with one nonzero format, and
    under-counts by ~2x on the default fp32 load (4-byte weights+grads) or with
    multiple nonzero formats (one bf16 dW each), picking too few chunks and
    tripping the watchdog mid-run. ``accurate_chunk_bytes`` switches to the real
    footprint: grad bytes from the model param ``element_size()`` (4 for fp32,
    2 for bf16) plus ``n_nonzero_fmts * dw_bytes`` for the per-format bf16
    deltas. It only changes how many memory-bounded passes are taken; the
    numerical payload is bit-identical for any G, so it is purely an opt-in
    safety knob and never perturbs the cost output."""
    free = _free_gib()
    if free == float("inf"):
        return 1
    import math
    numel = sum(linears[n].weight.numel() for n in names)
    # Headroom: 12 GiB covers a stored autograd graph + slack (the legacy
    # regime). With hook-harvest the graph is gone (checkpointing) and grads
    # are freed inside the backward, so the transient is ~one param's fp32
    # grad + logits buffers — 4 GiB suffices and the budget roughly triples
    # on a 90%-occupied box.
    headroom = 4.0 if hook_harvest else 12.0
    budget = max(free - (min_free_gib + headroom), 4.0)
    if not accurate_chunk_bytes and not hook_harvest:
        # Legacy path, preserved bit-for-bit: 2 bytes/weight, peak ~ 2*W/G.
        wgib = numel * 2 / (1024 ** 3)
        return max(1, min(math.ceil(2.0 * wgib / budget), len(names)))
    # Accurate: grad/weight footprint follows the model param dtype; dW is one
    # bf16 (``dw_bytes``) delta per nonzero format. Peak over the resident model
    # per chunk = numel/G * (grad_bytes + n_nonzero_fmts * dw_bytes); with
    # hook-harvest the chunk-wide grad term drops out entirely.
    grad_bytes = (
        next(iter(linears.values())).weight.element_size() if linears else 4
    )
    if hook_harvest:
        grad_bytes = 0
    per_weight_bytes = grad_bytes + max(1, n_nonzero_fmts) * max(1, dw_bytes)
    peak_gib = numel * per_weight_bytes / (1024 ** 3)
    return max(1, min(math.ceil(peak_gib / budget), len(names)))


def _packed_expert_targets(model: nn.Module, profile=None) -> list[str]:
    from prismaquant.build_rtn_cache import iter_quantizable_tensors

    out: list[str] = []
    for name, mod, attr in iter_quantizable_tensors(model, profile):
        if isinstance(mod, nn.Linear):
            continue
        param = getattr(mod, attr, None)
        if isinstance(param, nn.Parameter) and param.dim() == 3:
            out.append(str(name))
    return sorted(set(out))


def _guard_packed_expert_coverage(
    model: nn.Module,
    profile=None,
    *,
    allow_omission: bool = False,
) -> list[str]:
    packed = _packed_expert_targets(model, profile)
    if packed and not allow_omission:
        sample = ", ".join(packed[:6])
        raise RuntimeError(
            "Aura cost does not yet implement packed-MoE expert costs; "
            f"found {len(packed)} packed expert tensor(s), sample={sample}. "
            "Use the empirical packed-expert cost path or pass "
            "--allow-packed-expert-omission only for an explicit research/debug "
            "run that accepts experts being omitted from the AURA cost payload."
        )
    if packed:
        print(
            "[aura-cost] WARNING: omitting packed-MoE experts from Aura cost "
            f"by explicit request: {len(packed)} tensors, sample={packed[:6]}",
            flush=True,
        )
    return packed


def compute_aura_cost(
    model: nn.Module,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    *,
    n_probes: int = 16,
    token_scope: str = "all",
    temperature: float = 1.0,
    production_cache: object | None = None,
    min_free_gib: float = 20.0,
    seed_base: int = 7000,
    n_linear_chunks: int = 0,
    assert_bf16_passthrough: bool = False,
    accurate_chunk_bytes: bool = False,
    require_production_cache: bool = False,
    dw_dtype: str = "bfloat16",
    include_lm_head: bool = False,
    hook_harvest: bool = False,
    allow_packed_expert_omission: bool = False,
    probe_microbatch: int = 0,
    collect_col_energy: bool = False,
) -> dict:
    """Return a cost.pkl payload dict (stats + costs) for the allocator.

    ``n_linear_chunks`` bounds peak memory for large resident models: the
    target Linears are partitioned into G groups, and dW + retained grads are
    held for only one group at a time (peak ~ model + 2*model/G instead of
    3*model). The probe seeds and forwards are deterministic, so the per-Linear
    gradient a Linear receives is identical regardless of which group it lands
    in -- the chunked result is bit-identical to the single-pass (G=1) path,
    just computed in G memory-bounded passes. 0 = auto-size from free memory."""
    if n_probes < 1:
        raise ValueError(f"n_probes must be >= 1, got {n_probes!r}")
    omitted_packed_experts = _guard_packed_expert_coverage(
        model, allow_omission=allow_packed_expert_omission)
    _dw_torch_dtype = torch.float32 if str(dw_dtype) == "float32" else torch.bfloat16
    device = next(model.parameters()).device
    linears = _target_linears(model, include_lm_head=include_lm_head)
    if include_lm_head:
        # Tied-embeddings guard: with tie_word_embeddings the lm_head Parameter
        # IS the input embedding. The retained probe gradient on the shared
        # tensor then includes the embedding-path contribution, so the measured
        # cost prices quantizing BOTH uses -- while export ships only the
        # quantized lm_head view. That cost is wrong for the decision the
        # allocator actually makes; fail fast instead of silently mis-costing.
        embed = None
        get_embed = getattr(model, "get_input_embeddings", None)
        if callable(get_embed):
            try:
                embed = get_embed()
            except Exception:
                embed = None
        tied = [
            n for n, mod in linears.items()
            if "lm_head" in n and embed is not None
            and mod.weight is embed.weight
        ]
        if tied:
            raise RuntimeError(
                f"include_lm_head: {tied!r} shares its Parameter with the "
                f"input embedding (tie_word_embeddings). The probe gradient "
                f"includes the embedding-path contribution, so this cost would "
                f"not measure the lm_head-only decision the allocator prices. "
                f"Drop --include-lm-head for tied models.")
    names = list(linears.keys())
    fmts = [fr.canonical_format_name(f) for f in formats]
    nonzero_fmts = [f for f in fmts if f not in _ZERO_COST_FORMATS]
    render_score_records = _cache_render_score_records(production_cache)
    cb_provenance: dict[str, object] = {}
    if any(is_cb_format(fmt) for fmt in fmts):
        if production_cache is None:
            raise RuntimeError(
                "AURA CB cost requires a ProductionWeightCache with a "
                "persisted CB render identity"
            )
        from prismaquant.production_weight_cache import (
            production_cache_cb_render_provenance,
        )

        cb_provenance = production_cache_cb_render_provenance(
            production_cache,
            require_for_formats=fmts,
            where="AURA production cache",
        )
    # Passthrough-rule guard (opt-in; default off keeps the output byte-for-byte
    # identical). BF16 zero-cost is only valid when the source weight is already
    # bf16/fp16 -- on an fp32-source model loaded as fp32, casting W to BF16 is a
    # real downcast and the unconditional zero-cost is wrong. Catch that here
    # rather than silently mis-cost the format. (fp8 source can't be loaded as a
    # plain Linear weight, so an fp32 resident dtype never legitimizes BF16
    # zero-cost.)
    if assert_bf16_passthrough and "BF16" in fmts:
        src_dtype = next(model.parameters()).dtype
        if src_dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(
                f"assert_bf16_passthrough: BF16 zero-cost requires a bf16/fp16 "
                f"source weight (passthrough rule), but model params are "
                f"{src_dtype}. Loading as float32 makes BF16 a downcast, not a "
                f"passthrough -- drop BF16 from --formats or load the model as "
                f"bfloat16.")
    if n_linear_chunks <= 0:
        n_linear_chunks = _auto_n_chunks(
            linears, names, min_free_gib,
            n_nonzero_fmts=len(nonzero_fmts),
            dw_bytes=_dw_torch_dtype.itemsize,
            accurate_chunk_bytes=accurate_chunk_bytes,
            hook_harvest=hook_harvest,
        )
    n_linear_chunks = max(1, min(n_linear_chunks, len(names)))
    _log(f"targets={len(names)} formats={fmts} probes={n_probes} "
         f"dtype={next(model.parameters()).dtype} chunks={n_linear_chunks} "
         f"free={_free_gib():.1f}")

    for p in model.parameters():
        p.requires_grad_(False)

    # Partition Linears into G contiguous chunks. For each chunk we enable grad
    # on that chunk only, precompute its dW, run all K probes, project, free.
    chunks: list[list[str]] = [
        names[i::n_linear_chunks] for i in range(n_linear_chunks)
    ]
    chunks = [c for c in chunks if c]
    s2: dict[tuple[str, str], float] = {}
    s4: dict[tuple[str, str], float] = {}  # Σ(x²)² for the per-row stderr
    # Per-probe x² samples per row. Rows share the same K probes, so their
    # errors are CORRELATED — any sum of rows (an assignment's predicted KL)
    # needs the per-probe joint samples for an honest stderr; √Σσ² would
    # understate it. K floats per row (~256KB for a 500-Linear model).
    x2_probe: dict[tuple[str, str], list[float]] = {}
    dw_src: dict[tuple[str, str], str] = {}  # "rendered" | "rtn" per row
    g_trace: dict[str, float] = {}  # KL-Fisher weight-grad energy
    # Opt-in per-column KL-Fisher energy: the SAME grad energy g_trace sums to a
    # scalar, reduced over output rows to a length-in_features vector instead
    # (col_energy[n][j] = Σ_probes Σ_out (∂probe/∂W)[:,j]²). Default OFF keeps
    # the harvest arithmetic and payload bit-identical to today; when ON, its
    # per-row sum equals g_trace by construction (same summand, coarser
    # reduction). Feeds fisher_col_weights (exp 4: Fisher vs imatrix weighting).
    col_energy: dict[str, torch.Tensor] = {}
    inv = 1.0 / float(n_probes)

    for ci, chunk in enumerate(chunks):
        for n in chunk:
            linears[n].weight.requires_grad_(True)
        # Precompute dW_{i,f} (fp32 delta, stored bf16) for this chunk only.
        dW: dict[tuple[str, str], torch.Tensor] = {}
        with torch.no_grad():
            for f in nonzero_fmts:
                for n in chunk:
                    res = _delta_w(n, f, linears[n].weight.data,
                                   production_cache,
                                   strict=require_production_cache)
                    if res is not None:
                        d, src = res
                        dW[(n, f)] = d.to(_dw_torch_dtype)  # dot upcasts to fp32
                        dw_src[(n, f)] = src
        for key in dW:
            s2.setdefault(key, 0.0)
            s4.setdefault(key, 0.0)
            x2_probe.setdefault(key, [])
        for n in chunk:
            g_trace.setdefault(n, 0.0)
        # dW is now materialized for this chunk; the cache's LRU-resident
        # rendered weights are no longer needed. Evict them (back to disk
        # paths) so they don't accumulate across chunks -- otherwise the
        # cache LRU holds chunk 1+2+3's weights on top of the model and the
        # watchdog trips by the last chunk. compact_for_pickle() resets the
        # LRU; empty_cache returns the freed segments to the OS pool.
        compact = getattr(production_cache, "compact_for_pickle", None)
        if callable(compact):
            try:
                compact()
            except Exception:
                pass
        elif production_cache is not None and ci == 0:
            # No disk-backed eviction (in-memory cache): rendered tensors the
            # cache holds in RAM persist across chunks, so the per-chunk memory
            # bound is NOT guaranteed. Warn once; a --cache-dir-backed cache is
            # required for large resident models.
            _log("WARNING: production cache has no compact_for_pickle "
                 "(in-memory); cross-chunk memory bound not guaranteed -- use a "
                 "disk-backed (--cache-dir) cache for large resident models.")
        torch.cuda.empty_cache()
        if len(chunks) > 1:
            _log(f"chunk {ci+1}/{len(chunks)}: {len(chunk)} Linears, "
                 f"dW pairs={len(dW)}; free={_free_gib():.1f}")

        # K probe backward passes; one autograd graph alive at a time (fresh
        # forward per probe). Two harvest modes:
        #  * legacy: grads retained for the whole chunk, harvested after
        #    backward (chunk memory = grads + dW);
        #  * hook_harvest: post-accumulate-grad hooks project each grad the
        #    moment it lands and free it inside the backward — chunk memory
        #    is dW only, so chunks are ~3-4x larger and total backwards
        #    proportionally fewer. Per-(key,probe) values are identical
        #    (same reductions, just earlier).

        def _harvest_grad(name: str, g: torch.Tensor) -> None:
            """Project one fully-accumulated probe gradient into the running
            sums. Single reduction shared by all three harvest sites (hook,
            post-backward straggler sweep, legacy loop) so they are
            arithmetically identical by construction."""
            with torch.no_grad():
                gf = g.float()
                g_trace[name] += float((gf * gf).sum().item())
                if collect_col_energy:
                    # Reduce over output rows (dim 0) -> length-in_features
                    # vector; kept resident (GPU-first) and summed across
                    # probes/chunks. Targets are 2D nn.Linear ([out, in]);
                    # packed 3D experts are guarded out of this harvest.
                    ce = (gf * gf).sum(dim=0)
                    prev = col_energy.get(name)
                    col_energy[name] = ce if prev is None else prev + ce
                for f in nonzero_fmts:
                    key = (name, f)
                    if key in dW:
                        x2 = float(
                            (gf * dW[key].float()).sum().item()) ** 2
                        s2[key] += x2
                        s4[key] += x2 * x2
                        x2_probe[key].append(x2)

        hook_handles = []
        # Probe micro-batching (opt-in): at production calib volume the
        # vocab-shaped tensors of a monolithic forward dominate memory
        # (logits 32x1024x152k fp32 ~ 20 GiB, plus probe temps + grad-of-
        # logits). The probe scalar is a token-sum, so backward over
        # micro-batches accumulates EXACTLY the same total gradient; the
        # harvest hooks must fire only once the accumulation is complete,
        # and params absent from the final micro-batch's graph are picked
        # up by the post-backward straggler sweep (see below).
        # Probe noise is seeded per (probe, micro-batch), so results are
        # statistically equivalent to monolithic, not bit-identical —
        # except probe_microbatch=0/>=B (single batch), which is the
        # unchanged legacy path.
        _harvest_gate = {"on": True}
        # Names already harvested for the CURRENT probe. The hook only fires
        # for params in the FINAL micro-batch's autograd graph; a param that
        # participated only in earlier micro-batches (data-dependent routing)
        # still holds its real accumulated grad, which the post-backward
        # straggler sweep below harvests instead (audit 2026-07-02 M5 —
        # previously that grad was silently discarded → predicted_dloss 0.0).
        # This set guards against double-harvest between the two sites.
        _harvested: set[str] = set()
        if hook_harvest:
            def _make_hook(name: str):
                def _hook(param: torch.Tensor) -> None:
                    if not _harvest_gate["on"]:
                        return  # mid-accumulation: keep the partial grad
                    g = param.grad
                    if g is None or name in _harvested:
                        return
                    _harvest_grad(name, g)
                    _harvested.add(name)
                    param.grad = None
                return _hook
            for n in chunk:
                hook_handles.append(
                    linears[n].weight.register_post_accumulate_grad_hook(
                        _make_hook(n)))
        for k in range(n_probes):
            if _free_gib() < min_free_gib:
                raise RuntimeError(
                    f"free UMA {_free_gib():.1f} < floor {min_free_gib}; abort")
            for n in chunk:
                linears[n].weight.grad = None
            _harvested.clear()
            _B = calib_ids.size(0)
            _mb = int(probe_microbatch) if int(probe_microbatch) > 0 else _B
            _starts = list(range(0, _B, _mb))
            # Global selected-token count for the FULL calibration batch.
            # Each micro-batch normalizes its probe by this (not its own
            # slice's count), so the gradient summed across micro-batches
            # matches the monolithic-scale probe exactly (vs sqrt(M)-inflated
            # if each slice used its own count). Computed via a meta tensor
            # so the real scope logic decides the count with no allocation.
            _global_tc = None
            if len(_starts) > 1:
                _shape_probe = torch.zeros(
                    _B, calib_ids.size(1), 1, device="meta")
                _global_tc = token_count_for_logits(
                    select_token_scope(_shape_probe, token_scope))
            for _mi, _s0 in enumerate(_starts):
                _harvest_gate["on"] = (_mi == len(_starts) - 1)
                logits = model(calib_ids[_s0:_s0 + _mb]).logits
                probe = fisher_probe_scalar(
                    logits,
                    seed=(seed_base + k if len(_starts) == 1
                          else seed_base + k * 1000003 + _mi),
                    token_scope=token_scope,
                    temperature=temperature, distribution="rademacher",
                    token_count_override=_global_tc,
                )
                probe.backward()
                del logits, probe
            _harvest_gate["on"] = True
            logits = probe = None
            if hook_harvest:
                # Straggler sweep (M5): harvest any param the hook did NOT
                # fire for this probe but that holds a non-None accumulated
                # grad (i.e. it participated only in non-final micro-batches).
                # The accumulated .grad IS the monolithic-scale gradient:
                # every micro-batch's probe is normalized by the GLOBAL
                # selected-token count (token_count_override=_global_tc), so
                # backward accumulation across micro-batches is a plain sum
                # with factor 1 — no renormalization needed here.
                for n in chunk:
                    if n in _harvested:
                        continue
                    g = linears[n].weight.grad
                    if g is None:
                        continue
                    _harvest_grad(n, g)
                    _harvested.add(n)
                    linears[n].weight.grad = None
            else:
                for n in chunk:
                    g = linears[n].weight.grad
                    if g is None:
                        continue
                    _harvest_grad(n, g)
                    linears[n].weight.grad = None
            torch.cuda.empty_cache()
            if (k + 1) % 8 == 0:
                _log(f"  chunk {ci+1}/{len(chunks)} probe {k+1}/{n_probes}; "
                     f"free={_free_gib():.1f}")
        for h in hook_handles:
            h.remove()
        # Release this chunk's dW + grad enablement before the next chunk.
        del dW
        for n in chunk:
            linears[n].weight.grad = None
            linears[n].weight.requires_grad_(False)
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Assemble payload.
    inv = 1.0 / float(n_probes)
    stats: dict[str, dict] = {}
    costs: dict[str, dict] = {}
    for n in names:
        mod = linears[n]
        stats[n] = {
            "h_trace": g_trace[n] * inv,  # KL-Fisher weight-grad energy
            "n_params": int(mod.weight.numel()),
            "in_features": int(getattr(mod, "in_features", mod.weight.shape[1])),
            "out_features": int(getattr(mod, "out_features", mod.weight.shape[0])),
            "n_probes": int(n_probes),
        }
        if collect_col_energy and n in col_energy:
            # Per-column KL-Fisher energy, mean over probes (× inv) so its sum
            # matches h_trace (= g_trace × inv). fp32 CPU vector, length
            # in_features. Key present ONLY when collection is enabled, so the
            # default payload is byte-identical to today's.
            stats[n]["fisher_col"] = (
                col_energy[n].float() * inv).detach().cpu()
        costs[n] = {}
        for f in fmts:
            if f in _ZERO_COST_FORMATS:
                # Passthrough rule: zero error iff the source already has this
                # precision (bf16 source for BF16, fp8 source for FP8_SOURCE).
                # See _ZERO_COST_FORMATS and the assert_bf16_passthrough guard
                # above for the fp32-source downcast caveat.
                costs[n][f] = {
                    "predicted_dloss": 0.0,
                    "output_mse_measured": False,
                    "cost_source": "aura_passthrough_zero",
                }
                continue
            key = (n, f)
            if key not in s2:
                continue  # format illegal / no dW for this Linear
            # predicted_dloss = 0.5·mean_k(x²); its sampling stderr over the K
            # probes is 0.5·std(x²)/√K. This is the row's *risk*, free from the
            # same projections -- it feeds 'are K probes enough' introspection
            # and the additivity-gate threshold without seed-sweeping.
            # std uses the SAMPLE (1/(K−1)) variance, matching the additivity
            # gate's per-probe stderr (audit 2026-07-02 §3.13: the earlier
            # population 1/K form understated it by √(K/(K−1)), ~1.6% at
            # K=32, feeding the opt-in UCB charge). K<2 → stderr 0.0.
            mean_x2 = inv * s2[key]
            if n_probes >= 2:
                var_x2 = max(
                    (s4[key] - n_probes * mean_x2 * mean_x2)
                    / (n_probes - 1), 0.0)
            else:
                var_x2 = 0.0
            entry = {
                "predicted_dloss": 0.5 * mean_x2,
                "predicted_dloss_stderr": 0.5 * math.sqrt(var_x2 * inv),
                # raw per-probe x² samples (predicted_dloss = 0.5·mean of
                # these). Probe-aligned across rows — the additivity gate sums
                # them per probe for the exact correlated-sum stderr.
                "x2_per_probe": x2_probe[key],
                "dw_source": dw_src[key],
                "output_mse_measured": False,
                "cost_source": "aura",
            }
            render_record = _lookup_render_score_record(
                render_score_records,
                n,
                f,
            )
            entry.update(persisted_cell_score_fields(render_record))
            costs[n][f] = entry
    n_rendered = sum(1 for v in dw_src.values() if v == "rendered")
    n_rtn = sum(1 for v in dw_src.values() if v == "rtn")
    return {
        "schema": SCHEMA,
        "n_probes": n_probes,
        "formats": fmts,
        "token_scope": token_scope,
        "stats": stats,
        "costs": costs,
        # Reproducibility provenance (CLAUDE.md §5: an irreproducible number
        # is quarantined). seed_base is result-changing (allocation is
        # probe-seed-noisy); the rendered/RTN dW split is result-changing
        # (+36% served KL at FP8). main() adds model/calib identity on top.
        "provenance": {
            "seed_base": int(seed_base),
            "temperature": float(temperature),
            "dw_dtype": str(dw_dtype),
            "measurement_dtype": str(next(model.parameters()).dtype),
            "include_lm_head": bool(include_lm_head),
            "n_linear_chunks": int(n_linear_chunks),
            "calib_shape": list(calib_ids.shape),
            "calib_sha256": hashlib.sha256(
                calib_ids.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
            # R14: the canonical cross-stage calibration identity
            # (perturbed_x_cache.calibration_data_hash). validate_assignments_kl
            # intersects its own calib_repeat_hashes against these and refuses
            # to select on text the cost stage already saw. calib_sha256 above
            # is a different construction and is kept for continuity with
            # existing artifacts.
            "calib_hash": calibration_data_hash(calib_ids),
            "calib_hashes": [calibration_data_hash(calib_ids)],
            "omitted_packed_experts": omitted_packed_experts,
            "dw_rendered_rows": n_rendered,
            "dw_rtn_fallback_rows": n_rtn,
            "git_commit": _git_commit(),
            **(
                cb_provenance
                if cb_provenance
                else cb_cost_provenance(fmts)
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aura KL-adjoint allocator cost")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--formats", default="NVFP4,FP8_DYNAMIC,BF16")
    p.add_argument("--production-cache", default=None,
                   help="ProductionWeightCache pickle for production-faithful dW")
    p.add_argument("--n-probes", type=int, default=16)
    p.add_argument("--n-calib-samples", type=int, default=4)
    p.add_argument("--calib-seqlen", type=int, default=256)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-seed", type=int, default=42,
                   help="Seed for the calibration-window DRAW (which token "
                        "windows are sampled), distinct from --seed-base "
                        "(the probe directions). Vary this to measure "
                        "calibration-resampling variance of the cost.")
    p.add_argument("--dataset", default=None,
                   help="Optional calibration source (HF dataset id, .jsonl, "
                        "or .txt) via sensitivity_probe.load_calibration, so "
                        "the cost draws from the same corpus as the pipeline "
                        "probe/render. Default keeps the historical WikiText "
                        "windowed loader (--calib-split/--calib-seed).")
    p.add_argument("--token-scope", default="all")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--dtype", default="float32",
        choices=["float32", "bfloat16", "auto"],
        help="Resident model dtype. float32 (historical default) is the "
        "additivity-preferred cost regime but needs params x 4 bytes "
        "resident — ~140 GiB on a 35B, an OOM-kill on the 121 GiB box. "
        "'auto' sizes the checkpoint and picks float32 only when it fits "
        "with --min-free-gib headroom, else bfloat16 (the setting the 35B "
        "arm-E hybrid cost ran under).")
    p.add_argument("--n-linear-chunks", type=int, default=0,
                   help="Partition Linears into G memory-bounded groups "
                        "(peak ~ model + 2*model/G). 0 = auto-size from free "
                        "UMA. G=1 is the legacy single-pass path. Required >1 "
                        "for large resident models (e.g. 27B on a 121GB box).")
    p.add_argument("--min-free-gib", type=float, default=20.0)
    p.add_argument("--seed-base", type=int, default=7000,
                   help="Base seed for the Rademacher KL probes. Vary it "
                        "(same calibration) to test probe-direction stability "
                        "of the allocation -- i.e. whether K probes suffice.")
    p.add_argument("--assert-bf16-passthrough", action="store_true",
                   help="Fail fast if BF16 is in --formats but the model is "
                        "loaded fp32 (BF16 would be a downcast, not a lossless "
                        "passthrough, so its zero-cost would be wrong). Off by "
                        "default; current behavior is unchanged when omitted.")
    p.add_argument("--accurate-chunk-bytes", action="store_true",
                   help="Size --n-linear-chunks=0 auto-chunking from the real "
                        "per-weight footprint: grad bytes from the model param "
                        "element_size() (4 for fp32, 2 for bf16) + one bf16 dW "
                        "per nonzero format. The legacy default assumes 2 "
                        "bytes/weight and a single dW, under-counting ~2x on the "
                        "default fp32 load and tripping the watchdog. Off by "
                        "default; only changes the pass count, never the output "
                        "(bit-identical for any G).")
    p.add_argument("--require-production-cache", action="store_true",
                   help="Fail fast if the production cache lacks a rendered "
                        "(Linear, format); refuse silent RTN fallback. Off by "
                        "default. Use for production-faithful cost runs.")
    p.add_argument("--dw-dtype", default="bfloat16",
                   choices=["bfloat16", "float32"],
                   help="Storage dtype for the dW=Q_f(W)-W error vector. Default "
                        "bfloat16 (validated: bf16-vs-fp32 Aura Spearman 0.997); "
                        "float32 for exact fidelity at 2x dW memory.")
    p.add_argument("--include-lm-head", action="store_true",
                   help="Also measure lm_head (normally pinned BF16) so the "
                        "allocator can choose its format by budget-value rather "
                        "than a hardcoded pin. dW falls back to RTN if the cache "
                        "lacks a rendered lm_head.")
    p.add_argument("--hook-harvest", action="store_true",
                   help="Project each gradient onto dW inside the backward "
                        "(post-accumulate-grad hooks) and free it immediately. "
                        "Chunk memory becomes dW-only, so chunks grow ~3-4x "
                        "and total backwards shrink proportionally. Identical "
                        "per-probe values; pair with --gradient-checkpointing "
                        "for large fp32 models.")
    p.add_argument("--gradient-checkpointing", action="store_true",
                   help="Recompute activations during the probe backward "
                        "instead of storing the graph. Required for fp32 "
                        "measurement of ~27B models on the 121GB box: the "
                        "resident model (~108GB) + a stored 4x256 graph "
                        "(~10-15GB) OOM-kills between watchdog checks "
                        "(observed 2026-06-10). ~30% slower; numerically "
                        "identical recompute in fp32.")
    p.add_argument("--probe-microbatch", type=int, default=0,
                   help="Forward the calibration in groups of this many "
                        "samples per probe, accumulating gradients (memory "
                        "control for production calib volume; the monolithic "
                        "forward's vocab-shaped tensors are ~20 GiB at "
                        "32x1024). 0 = single batch (legacy, bit-identical). "
                        ">0 changes probe-noise draws: statistically "
                        "equivalent, not bit-identical to monolithic.")
    p.add_argument("--allow-packed-expert-omission", action="store_true",
                   help="Explicit research/debug escape: allow AURA to omit "
                        "packed-MoE expert tensors from the cost payload. "
                        "Default is fail-fast because packed experts need an "
                        "empirical/hybrid expert-cost path, not silent omission.")
    p.add_argument("--collect-col-energy", action="store_true",
                   default=os.environ.get("PRISMAQUANT_FISHER_COL_WEIGHTS") == "1",
                   help="Also emit a per-Linear per-column KL-Fisher energy "
                        "vector (stats[name]['fisher_col'], length in_features) "
                        "for Fisher-weighted codeword/scale search (nvfp4-cb "
                        "exp 4). Additive: the rest of the payload is unchanged. "
                        "Env: PRISMAQUANT_FISHER_COL_WEIGHTS=1.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cost-mode", default="",
                   help="Pipeline COST_MODE stamped into "
                        "provenance['cost_mode'] (re-vet R2).")
    args = p.parse_args(argv)
    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("aura_cost", args.device)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prismaquant.calibration_data import load_wikitext_calibration_windowed
    # Reuse the pipeline's text-only stager so multimodal (VL) checkpoints
    # (e.g. Qwen3.6-27B Qwen3_5ForConditionalGeneration) load as a CausalLM
    # with tensor names that match the production cache keys. No-op on
    # pure-text checkpoints.
    from prismaquant.build_rtn_cache import stage_multimodal

    staged, _cleanup = stage_multimodal(args.model)
    if args.dtype == "auto":
        args.dtype = _resolve_auto_dtype(staged, args.min_free_gib)
    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16
    local_only = Path(staged).exists()
    _log(f"loading {args.model} (staged={staged}) dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(
        staged, trust_remote_code=True, local_files_only=local_only)
    load_kwargs = dict(
        dtype=dt, trust_remote_code=True, local_files_only=local_only,
        attn_implementation="eager",
    )
    if args.device.startswith("cuda"):
        load_kwargs["device_map"] = args.device
    try:
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
    except ValueError as exc:
        if "accelerate" not in str(exc):
            raise
        load_kwargs.pop("device_map", None)
        model = AutoModelForCausalLM.from_pretrained(staged, **load_kwargs)
        model.to(args.device)
    model.eval()
    if args.gradient_checkpointing:
        # transformers gates checkpointing on self.training — in eval() the
        # checkpointed path is silently bypassed and the full graph is stored
        # (observed OOM 2026-06-10). train() arms it; that is numerically
        # identical to eval() ONLY when no dropout/batchnorm is active, so
        # refuse otherwise instead of silently measuring under noise.
        for mod_name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Dropout) and mod.p > 0:
                raise RuntimeError(
                    f"--gradient-checkpointing needs train() mode, but "
                    f"{mod_name} has dropout p={mod.p} — train() would not "
                    f"be eval-equivalent on this architecture.")
            if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                                torch.nn.BatchNorm3d)):
                raise RuntimeError(
                    f"--gradient-checkpointing needs train() mode, but "
                    f"{mod_name} is BatchNorm — train() would update "
                    f"running stats.")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        _log("gradient checkpointing ON (non-reentrant, train-mode armed, "
             "no active dropout/batchnorm)")
    if args.dataset:
        from prismaquant.sensitivity_probe import load_calibration
        calib = load_calibration(
            tok, args.dataset, args.n_calib_samples, args.calib_seqlen,
            calib_seed=args.calib_seed,
        ).to(args.device)
    else:
        calib = load_wikitext_calibration_windowed(
            tok, args.n_calib_samples, args.calib_seqlen,
            split=args.calib_split, seed=args.calib_seed,
        ).to(args.device)

    cache = None
    if args.production_cache:
        with open(args.production_cache, "rb") as fh:
            cache = pickle.load(fh)
        _log(f"loaded production cache: {args.production_cache}")

    payload = compute_aura_cost(
        model, calib, [f.strip() for f in args.formats.split(",") if f.strip()],
        n_probes=args.n_probes, token_scope=args.token_scope,
        temperature=args.temperature, production_cache=cache,
        min_free_gib=args.min_free_gib, n_linear_chunks=args.n_linear_chunks,
        seed_base=args.seed_base,
        assert_bf16_passthrough=args.assert_bf16_passthrough,
        accurate_chunk_bytes=args.accurate_chunk_bytes,
        require_production_cache=args.require_production_cache,
        dw_dtype=args.dw_dtype,
        include_lm_head=args.include_lm_head,
        hook_harvest=args.hook_harvest,
        allow_packed_expert_omission=args.allow_packed_expert_omission,
        probe_microbatch=args.probe_microbatch,
        collect_col_energy=args.collect_col_energy,
    )
    payload["provenance"].update({
        "model": str(args.model),
        "dtype": str(args.dtype),
        "calib_source": (
            str(args.dataset) if args.dataset
            else f"wikitext:{args.calib_split}"),
        "n_calib_samples": int(args.n_calib_samples),
        "calib_seqlen": int(args.calib_seqlen),
        "calib_seed": int(args.calib_seed),
        "production_cache": str(args.production_cache or ""),
        # re-vet R2 precondition (i): which pipeline COST_MODE produced this.
        "cost_mode": str(args.cost_mode or ""),
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as fh:
        pickle.dump(payload, fh)
    nz = sum(1 for n in payload["costs"] for f in payload["costs"][n]
             if payload["costs"][n][f].get("predicted_dloss", 0.0) > 0)
    prov = payload["provenance"]
    _log(f"wrote {args.output}: {len(payload['costs'])} Linears, {nz} non-zero "
         f"cost entries (dW rendered={prov['dw_rendered_rows']} "
         f"rtn={prov['dw_rtn_fallback_rows']}, seed_base={prov['seed_base']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
