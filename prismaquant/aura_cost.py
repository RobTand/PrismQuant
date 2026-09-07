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
from collections.abc import Mapping, MutableMapping
import hashlib
import json
import math
import os
import pickle
import re
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
from prismaquant.routed_experts import (
    PackedExpertProjection,
    profile_declared_packed_expert_projections,
    profile_declared_routed_expert_targets,
    profile_declared_unpacked_expert_linears,
    refresh_packed_expert_projections,
    resolve_routed_expert_profile,
)

SCHEMA = "prismaquant.aura_cost.v1"
AURA_CHECKPOINT_IDENTITY_SCHEMA = "prismaquant.aura_checkpoint.identity.v1"
AURA_CHECKPOINT_MANIFEST_SCHEMA = "prismaquant.aura_checkpoint.manifest.v1"
AURA_CHECKPOINT_UNIT_SCHEMA = "prismaquant.aura_checkpoint.unit.v1"
AURA_PRODUCTION_ANCHOR_DELTA_CONSUMER_IDENTITY = {
    "schema": "prismaquant.aura.production_anchor_delta_consumer.v1",
    "canonical_input": "production_weight_cache_canonical_cpu_tensor",
    "operation": "fp32_subtract_then_store",
    "subtraction_dtype": "torch.float32",
    "storage_dtype": "torch.bfloat16",
    "output_residency": "source_weight_device",
}


def _git_commit() -> str | None:
    """Best-effort commit of the prismaquant tree this cost was computed by."""
    override = str(os.environ.get(
        "PRISMAQUANT_IDENTITY_GIT_COMMIT", ""
    )).strip().lower()
    if override:
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", override) is None:
            raise RuntimeError(
                "PRISMAQUANT_IDENTITY_GIT_COMMIT must be a full 40- or "
                "64-character hexadecimal commit id"
            )
        return override
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None


def _checkpoint_git_commit() -> str:
    commit = _git_commit()
    if commit is None:
        raise RuntimeError("AURA checkpoint identity cannot resolve git commit")
    if os.environ.get("PRISMAQUANT_IDENTITY_GIT_COMMIT"):
        return commit
    repo_root = Path(__file__).resolve().parents[1]
    clean = subprocess.run(
        ["git", "diff", "--quiet", commit, "--", "prismaquant/aura_cost.py"],
        cwd=repo_root,
        check=False,
        timeout=10,
    )
    if clean.returncode != 0:
        raise RuntimeError(
            "AURA checkpoint git identity is not exact: aura_cost.py differs "
            f"from commit {commit}; commit it before checkpoint/resume"
        )
    return commit


def _aura_source_sha256() -> str:
    """Bind AURA to the complete producer package, including dependencies."""
    from prismaquant.production_weight_cache import (
        _production_cache_source_sha256,
    )

    return _production_cache_source_sha256()


def _canonical_json(value: object, *, where: str) -> object:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is not canonical JSON data") from exc
    return json.loads(encoded)


def _canonical_json_sha256(value: object, *, where: str) -> str:
    canonical = _canonical_json(value, where=where)
    return hashlib.sha256(json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    # fsync the containing directory so the rename itself is durable across a
    # host reset. Failure is load-bearing and must not be mistaken for a
    # durable checkpoint.
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _aura_unit_checkpoint_path(checkpoint_dir: Path, qname: str) -> Path:
    digest = hashlib.sha256(str(qname).encode("utf-8")).hexdigest()
    return checkpoint_dir / "units" / f"{digest}.pkl"


def _raise_checkpoint_identity_mismatch(
    *,
    field: str,
    stored: object,
    expected: object,
) -> None:
    from prismaquant.production_weight_cache import identity_value_for_error

    raise RuntimeError(
        f"AURA checkpoint identity mismatch at {field}: "
        f"stored={identity_value_for_error(stored)} "
        f"current={identity_value_for_error(expected)}; refusing reuse or "
        "recompute"
    )


def _write_aura_checkpoint_manifest(
    checkpoint_dir: Path,
    identity: Mapping[str, object],
    names: Sequence[str],
) -> str:
    identity_sha256 = _canonical_json_sha256(
        identity,
        where="AURA checkpoint identity",
    )
    manifest = {
        "schema": AURA_CHECKPOINT_MANIFEST_SCHEMA,
        "identity_sha256": identity_sha256,
        "identity": dict(identity),
        "units": [
            {
                "qname": str(name),
                "file": str(
                    _aura_unit_checkpoint_path(checkpoint_dir, name).relative_to(
                        checkpoint_dir
                    )
                ),
            }
            for name in names
        ],
    }
    encoded = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    _atomic_write_bytes(checkpoint_dir / "manifest.json", encoded)
    return identity_sha256


def _load_aura_checkpoint_manifest(
    checkpoint_dir: Path,
    expected_identity: Mapping[str, object],
) -> str:
    path = checkpoint_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except Exception as exc:
        _raise_checkpoint_identity_mismatch(
            field="manifest_json",
            stored="<invalid>",
            expected="<valid canonical JSON>",
        )
        raise AssertionError("unreachable") from exc
    if not isinstance(manifest, Mapping):
        _raise_checkpoint_identity_mismatch(
            field="manifest",
            stored=manifest,
            expected="<object>",
        )
    if manifest.get("schema") != AURA_CHECKPOINT_MANIFEST_SCHEMA:
        _raise_checkpoint_identity_mismatch(
            field="manifest.schema",
            stored=manifest.get("schema"),
            expected=AURA_CHECKPOINT_MANIFEST_SCHEMA,
        )
    stored_identity = manifest.get("identity")
    from prismaquant.production_weight_cache import first_identity_difference

    difference = first_identity_difference(stored_identity, expected_identity)
    if difference is not None:
        field, stored, expected = difference
        _raise_checkpoint_identity_mismatch(
            field=field,
            stored=stored,
            expected=expected,
        )
    expected_digest = _canonical_json_sha256(
        expected_identity,
        where="AURA checkpoint identity",
    )
    if manifest.get("identity_sha256") != expected_digest:
        _raise_checkpoint_identity_mismatch(
            field="manifest.identity_sha256",
            stored=manifest.get("identity_sha256"),
            expected=expected_digest,
        )
    return expected_digest


def _write_aura_unit_checkpoint(
    checkpoint_dir: Path,
    *,
    qname: str,
    identity_sha256: str,
    state: Mapping[str, object],
) -> None:
    state_bytes = pickle.dumps(dict(state), protocol=pickle.HIGHEST_PROTOCOL)
    envelope = {
        "schema": AURA_CHECKPOINT_UNIT_SCHEMA,
        "qname": str(qname),
        "identity_sha256": str(identity_sha256),
        "payload_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "payload": state_bytes,
    }
    encoded = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_bytes(
        _aura_unit_checkpoint_path(checkpoint_dir, qname),
        encoded,
    )


def _load_aura_unit_checkpoint(
    path: Path,
    *,
    qname: str,
    identity_sha256: str,
) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            envelope = pickle.load(handle)
    except Exception as exc:
        raise RuntimeError(
            f"AURA unit checkpoint {path} is corrupt for {qname}; refusing "
            "reuse or recompute"
        ) from exc
    if not isinstance(envelope, Mapping):
        raise RuntimeError(
            f"AURA unit checkpoint {path} is not an envelope for {qname}; "
            "refusing reuse or recompute"
        )
    for field, expected in (
        ("schema", AURA_CHECKPOINT_UNIT_SCHEMA),
        ("qname", str(qname)),
        ("identity_sha256", str(identity_sha256)),
    ):
        if envelope.get(field) != expected:
            _raise_checkpoint_identity_mismatch(
                field=f"unit[{qname}].{field}",
                stored=envelope.get(field),
                expected=expected,
            )
    payload = envelope.get("payload")
    if not isinstance(payload, bytes):
        raise RuntimeError(
            f"AURA unit checkpoint {path} has no byte payload for {qname}; "
            "refusing reuse or recompute"
        )
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if envelope.get("payload_sha256") != payload_sha256:
        raise RuntimeError(
            f"AURA unit checkpoint {path} payload_sha256 differs for {qname}; "
            "refusing reuse or recompute"
        )
    try:
        state = pickle.loads(payload)
    except Exception as exc:
        raise RuntimeError(
            f"AURA unit checkpoint {path} state is corrupt for {qname}; "
            "refusing reuse or recompute"
        ) from exc
    if not isinstance(state, Mapping):
        raise RuntimeError(
            f"AURA unit checkpoint {path} state is not an object for {qname}; "
            "refusing reuse or recompute"
        )
    return dict(state)


def _prepare_aura_checkpoints(
    checkpoint_dir: str | Path,
    *,
    resume: bool,
    identity: Mapping[str, object],
    names: Sequence[str],
) -> tuple[Path, str, dict[str, dict[str, object]]]:
    root = Path(checkpoint_dir)
    if root.exists() and not root.is_dir():
        raise RuntimeError(f"AURA checkpoint path is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise RuntimeError(
                f"AURA checkpoint manifest already exists at {manifest_path}; "
                "pass --resume to validate and reuse it"
            )
        identity_sha256 = _load_aura_checkpoint_manifest(root, identity)
    else:
        existing_units = sorted((root / "units").glob("*.pkl"))
        if existing_units:
            raise RuntimeError(
                "AURA checkpoint units exist without a manifest; refusing "
                f"name-gated reuse or recompute. sample={existing_units[:8]}"
            )
        identity_sha256 = _write_aura_checkpoint_manifest(
            root,
            identity,
            names,
        )

    expected_paths = {
        _aura_unit_checkpoint_path(root, name): str(name)
        for name in names
    }
    unexpected = sorted(
        path for path in (root / "units").glob("*.pkl")
        if path not in expected_paths
    )
    if unexpected:
        _raise_checkpoint_identity_mismatch(
            field="units.unexpected",
            stored=[str(path.name) for path in unexpected[:8]],
            expected=[],
        )
    completed: dict[str, dict[str, object]] = {}
    for path, name in expected_paths.items():
        if not path.is_file():
            continue
        completed[name] = _load_aura_unit_checkpoint(
            path,
            qname=name,
            identity_sha256=identity_sha256,
        )
    return root, identity_sha256, completed

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
_ZERO_COST_FORMATS = {
    "BF16",
    "FP8_SOURCE",
    "MXFP4_SOURCE",
}


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


def _release_streamed_anchor_allocator_cache(device: object) -> None:
    """Return consumed production anchors to Spark unified memory.

    Dropping the final tensor reference only moves its CUDA allocation into
    PyTorch's reusable cache.  On the single-GPU GB10 path that memory still
    overlaps the much larger FP32 adjoint deltas unless the cache is returned
    to the unified host/device pool before backward begins.  This is once per
    streamed layer, outside the probe loop; non-CUDA test/research paths stay a
    no-op.
    """
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return
    torch.cuda.synchronize(resolved)
    torch.cuda.empty_cache()


def _stored_production_anchor_delta(
    rendered: torch.Tensor,
    source: torch.Tensor,
    *,
    storage_dtype: torch.dtype,
) -> torch.Tensor:
    """Subtract production anchors in FP32, then store the configured dW.

    The projection consumer always upcasts dW to FP32. Keeping every routed
    layer's two anchor deltas in FP32 therefore doubles residency without
    changing the accumulator. Subtract in FP32 to preserve cancellation, then
    use the same validated BF16 storage contract as ordinary cached-menu AURA.
    ``copy=True`` prevents an FP32 injected/test render from being mutated by
    the in-place subtraction.
    """
    delta_fp32 = rendered.to(
        device=source.device,
        dtype=torch.float32,
        copy=True,
    )
    delta_fp32.sub_(source)
    if storage_dtype == torch.float32:
        return delta_fp32
    return delta_fp32.to(dtype=storage_dtype)


def _target_linears(
    model: nn.Module,
    *,
    include_lm_head: bool = False,
    include_routed_experts: bool = False,
    profile=None,
) -> dict[str, nn.Linear]:
    """Quantizable nn.Linear targets. lm_head is EXCLUDED by default (the
    profile pins it BF16). include_lm_head adds it so Aura can MEASURE its
    KL-sensitivity and let the allocator choose its format as a budget
    decision rather than a hardcoded pin -- the KL probe gradient flows
    directly into lm_head (it produces the logits), so its cost is
    measured the same way as any body Linear."""
    # Routed-expert membership is a serving/model-profile property, not a
    # tensor-rank property.  In particular DSv4's probe model exposes every
    # expert projection as a 2-D Linear, which otherwise looks exactly like a
    # smooth AURA target.  Resolve the same profile-owned predicate used by
    # the coverage guard below so exclusion and enforcement cannot diverge.
    profile = resolve_routed_expert_profile(model, profile)
    unpacked_expert_names = {
        member.qname
        for member in profile_declared_unpacked_expert_linears(model, profile)
    }
    out: dict[str, nn.Linear] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name in unpacked_expert_names and not include_routed_experts:
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
    # Ahead of ``get_format``, which imports the ``tessera`` package to
    # synthesize a Tessera spec: the refusal is about the name, not the spec.
    if fr.is_tessera_format_name(fmt):
        raise RuntimeError(
            f"{name}={fmt}: AURA Tessera delta requires a production-cache "
            "render. The registry fallback is a weights-only reconstruction, "
            "not the decoded wire and not the H-aware encode that ships, so "
            "it would price a different dW under the same format name -- and "
            "``strict`` defaults off, so nothing else would say so."
        )
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
    """Profile-declared routed expert targets in either physical layout.

    The historical name is retained for artifact/provenance compatibility,
    but rank is intentionally absent: packed 3-D Parameters and unpacked 2-D
    expert Linears are the same route-sensitive serving class.
    """
    return profile_declared_routed_expert_targets(model, profile)


def _guard_packed_expert_coverage(
    model: nn.Module,
    profile=None,
    *,
    allow_omission: bool = False,
) -> list[str]:
    routed = _packed_expert_targets(model, profile)
    if routed and not allow_omission:
        sample = ", ".join(routed[:6])
        raise RuntimeError(
            "Aura cost does not yet implement packed-MoE expert costs in its "
            "smooth sweep; "
            f"found {len(routed)} profile-declared routed expert target(s), "
            f"sample={sample}. Use the empirical routed-expert cost path or pass "
            "--allow-packed-expert-omission only for an explicit research/debug "
            "run that accepts experts being omitted from the AURA cost payload."
        )
    if routed:
        print(
            "[aura-cost] WARNING: omitting profile-declared routed experts "
            "from Aura cost by explicit request: "
            f"{len(routed)} targets, sample={routed[:6]}",
            flush=True,
        )
    return routed


def _build_aura_checkpoint_identity(
    *,
    model: nn.Module,
    calib_ids: torch.Tensor,
    names: Sequence[str],
    linears: Mapping[str, nn.Linear],
    formats: Sequence[str],
    chunks: Sequence[Sequence[str]],
    n_probes: int,
    token_scope: str,
    temperature: float,
    seed_base: int,
    dw_dtype: str,
    include_lm_head: bool,
    hook_harvest: bool,
    allow_packed_expert_omission: bool,
    probe_microbatch: int,
    collect_col_energy: bool,
    require_production_cache: bool,
    production_cache: object,
    cb_provenance: Mapping[str, object],
    git_commit: str,
    extra_identity: Mapping[str, object] | None,
) -> dict[str, object]:
    raw_calib_sha256 = hashlib.sha256(
        calib_ids.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    cache_metadata = getattr(production_cache, "metadata", None)
    cache_metadata = cache_metadata if isinstance(cache_metadata, Mapping) else {}
    identity = {
        "schema": AURA_CHECKPOINT_IDENTITY_SCHEMA,
        "git_commit": str(git_commit),
        "producer_source_sha256": _aura_source_sha256(),
        "calibration": {
            "shape": [int(dim) for dim in calib_ids.shape],
            "dtype": str(calib_ids.dtype),
            "sha256": raw_calib_sha256,
            "calib_hash": calibration_data_hash(calib_ids),
        },
        "formats": [str(fmt) for fmt in formats],
        "units": [
            {
                "qname": str(name),
                "shape": [int(dim) for dim in linears[name].weight.shape],
                "dtype": str(linears[name].weight.dtype),
                "n_params": int(linears[name].weight.numel()),
            }
            for name in names
        ],
        "chunks": [[str(name) for name in chunk] for chunk in chunks],
        "n_probes": int(n_probes),
        "token_scope": str(token_scope),
        "temperature": float(temperature),
        "seed_base": int(seed_base),
        "measurement_dtype": str(next(model.parameters()).dtype),
        "dw_dtype": str(dw_dtype),
        "include_lm_head": bool(include_lm_head),
        "hook_harvest": bool(hook_harvest),
        "allow_packed_expert_omission": bool(allow_packed_expert_omission),
        "probe_microbatch": int(probe_microbatch),
        "collect_col_energy": bool(collect_col_energy),
        "require_production_cache": bool(require_production_cache),
        "production_cache_calib_hash": cache_metadata.get("calib_hash"),
        "production_cache_pair_identity": cache_metadata.get(
            "cb_cache_pair_identity"
        ),
        # The complete value-bearing identity is intentionally embedded, not
        # reduced to a cache filename. It binds codebooks, source/column
        # weights, scale/layout/sweep/LDLQ, and every qname/rung format scope.
        "cb_render_identity": cb_provenance.get("cb_render_identity"),
        "extra": dict(extra_identity or {}),
    }
    return _canonical_json(identity, where="AURA checkpoint identity")


def _validate_aura_checkpoint_cache_identity(production_cache: object) -> None:
    metadata = getattr(production_cache, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise RuntimeError(
            "AURA durable checkpointing requires production-cache metadata; "
            "refusing model/cache name-gated resume"
        )
    calibration_hash = metadata.get("calib_hash")
    if not isinstance(calibration_hash, str) or not calibration_hash:
        raise RuntimeError(
            "AURA durable checkpointing requires the production cache's exact "
            "calib_hash; refusing model/cache name-gated resume"
        )
    pair_set = metadata.get("cb_cache_pair_identity")
    if not isinstance(pair_set, Mapping):
        raise RuntimeError(
            "AURA durable checkpointing requires identity-bound CB pair "
            "artifacts; refusing model/cache name-gated resume"
        )
    if pair_set.get("schema") != (
        "prismaquant.production_weight_cache.cb_pair_set.v1"
    ):
        raise RuntimeError(
            "AURA durable checkpointing found an unsupported CB pair identity "
            f"schema {pair_set.get('schema')!r}"
        )
    try:
        entries = int(pair_set["entries"])
        published_entries = int(pair_set["published_entries"])
    except Exception as exc:
        raise RuntimeError(
            "AURA durable checkpointing found malformed CB pair entry counts"
        ) from exc
    if entries < 1 or published_entries != entries:
        raise RuntimeError(
            "AURA durable checkpointing requires every identity-bound CB pair "
            f"artifact to be published; entries={entries} "
            f"published_entries={published_entries}"
        )
    for field in ("identity_sha256", "artifact_sha256"):
        digest = str(pair_set.get(field, "")).lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(
                "AURA durable checkpointing found an invalid CB pair "
                f"{field}"
            )
    calibration_hashes = pair_set.get("calibration_hashes")
    if (
        not isinstance(calibration_hashes, Sequence)
        or isinstance(calibration_hashes, (str, bytes))
        or list(calibration_hashes) != [calibration_hash]
    ):
        raise RuntimeError(
            "AURA durable checkpointing found a production-cache calibration "
            "hash that differs from its CB pair artifacts"
        )
    commits = pair_set.get("git_commits")
    if (
        not isinstance(commits, Sequence)
        or isinstance(commits, (str, bytes))
        or len(commits) != 1
        or re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
            str(commits[0]).lower(),
        ) is None
    ):
        raise RuntimeError(
            "AURA durable checkpointing requires one exact CB pair producer "
            "git commit"
        )
    source_digests = pair_set.get("producer_source_sha256")
    if (
        not isinstance(source_digests, Sequence)
        or isinstance(source_digests, (str, bytes))
        or len(source_digests) != 1
        or re.fullmatch(
            r"[0-9a-f]{64}", str(source_digests[0]).lower()
        ) is None
    ):
        raise RuntimeError(
            "AURA durable checkpointing requires one exact CB renderer "
            "source SHA-256 identity"
        )


def _aura_unit_state(
    name: str,
    nonzero_formats: Sequence[str],
    *,
    s2: Mapping[tuple[str, str], float],
    s4: Mapping[tuple[str, str], float],
    x2_probe: Mapping[tuple[str, str], list[float]],
    dw_src: Mapping[tuple[str, str], str],
    g_trace: Mapping[str, float],
    col_energy: Mapping[str, torch.Tensor],
    weight_mse_diagnostic: Mapping[tuple[str, str], float] | None = None,
    source_weight_identity: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    for fmt in nonzero_formats:
        key = (name, fmt)
        if key not in s2:
            continue
        rows[fmt] = {
            "s2": float(s2[key]),
            "s4": float(s4[key]),
            "x2_probe": list(x2_probe[key]),
            "dw_src": str(dw_src[key]),
        }
        if (
            weight_mse_diagnostic is not None
            and key in weight_mse_diagnostic
        ):
            rows[fmt]["weight_mse_diagnostic"] = float(
                weight_mse_diagnostic[key]
            )
    state = {
        "g_trace": float(g_trace[name]),
        "rows": rows,
        "col_energy": (
            col_energy[name].detach().to(device="cpu", dtype=torch.float32)
            if name in col_energy
            else None
        ),
    }
    if source_weight_identity is not None and name in source_weight_identity:
        state["source_weight_identity"] = dict(
            source_weight_identity[name]
        )
    return state


def _restore_aura_unit_state(
    name: str,
    state: Mapping[str, object],
    *,
    nonzero_formats: Sequence[str],
    n_probes: int,
    collect_col_energy: bool,
    s2: dict[tuple[str, str], float],
    s4: dict[tuple[str, str], float],
    x2_probe: dict[tuple[str, str], list[float]],
    dw_src: dict[tuple[str, str], str],
    g_trace: dict[str, float],
    col_energy: dict[str, torch.Tensor],
    diagnostic_weight_mse_pairs: set[tuple[str, str]] | None = None,
    weight_mse_diagnostic: dict[tuple[str, str], float] | None = None,
    require_source_weight_identity: bool = False,
    source_weight_identity: dict[str, dict[str, object]] | None = None,
) -> None:
    try:
        g_trace[name] = float(state["g_trace"])
    except Exception as exc:
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} has invalid g_trace; "
            "refusing reuse or recompute"
        ) from exc
    rows = state.get("rows")
    if not isinstance(rows, Mapping):
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} has invalid rows; "
            "refusing reuse or recompute"
        )
    unexpected = sorted(set(str(fmt) for fmt in rows) - set(nonzero_formats))
    if unexpected:
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} has unexpected formats "
            f"{unexpected}; refusing reuse or recompute"
        )
    for raw_fmt, raw_row in rows.items():
        fmt = str(raw_fmt)
        if not isinstance(raw_row, Mapping):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name}@{fmt} is invalid; "
                "refusing reuse or recompute"
            )
        samples = raw_row.get("x2_probe")
        if (
            not isinstance(samples, Sequence)
            or isinstance(samples, (str, bytes))
            or len(samples) != int(n_probes)
        ):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name}@{fmt} has "
                f"x2_probe length {len(samples) if isinstance(samples, Sequence) else '<invalid>'}; "
                f"expected {n_probes}; refusing reuse or recompute"
            )
        key = (name, fmt)
        try:
            s2[key] = float(raw_row["s2"])
            s4[key] = float(raw_row["s4"])
            x2_probe[key] = [float(value) for value in samples]
            dw_src[key] = str(raw_row["dw_src"])
        except Exception as exc:
            raise RuntimeError(
                f"AURA unit checkpoint state for {name}@{fmt} is malformed; "
                "refusing reuse or recompute"
            ) from exc
        expected_diagnostic = (
            diagnostic_weight_mse_pairs is not None
            and key in diagnostic_weight_mse_pairs
        )
        observed_diagnostic = "weight_mse_diagnostic" in raw_row
        if observed_diagnostic != expected_diagnostic:
            raise RuntimeError(
                f"AURA unit checkpoint state for {name}@{fmt} diagnostic "
                "weight-MSE scope differs; refusing reuse or recompute"
            )
        if expected_diagnostic:
            if weight_mse_diagnostic is None:
                raise RuntimeError(
                    "AURA diagnostic weight-MSE restore has no destination"
                )
            try:
                weight_mse_diagnostic[key] = float(
                    raw_row["weight_mse_diagnostic"]
                )
            except Exception as exc:
                raise RuntimeError(
                    f"AURA unit checkpoint state for {name}@{fmt} has an "
                    "invalid diagnostic weight-MSE"
                ) from exc
    stored_col_energy = state.get("col_energy")
    if collect_col_energy:
        if stored_col_energy is not None and not isinstance(
            stored_col_energy, torch.Tensor
        ):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name} has invalid "
                "col_energy; refusing reuse or recompute"
            )
        if isinstance(stored_col_energy, torch.Tensor):
            col_energy[name] = stored_col_energy.detach().to(
                device="cpu", dtype=torch.float32
            )
    elif stored_col_energy is not None:
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} unexpectedly carries "
            "col_energy; refusing reuse or recompute"
        )
    raw_source = state.get("source_weight_identity")
    if require_source_weight_identity:
        if not isinstance(raw_source, Mapping):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name} has no production "
                "anchor source-weight identity; refusing reuse or recompute"
            )
        try:
            shape = [int(dim) for dim in raw_source["shape"]]
            digest = str(raw_source["sha256"]).lower()
        except Exception as exc:
            raise RuntimeError(
                f"AURA unit checkpoint state for {name} has malformed "
                "source-weight identity"
            ) from exc
        if (
            len(shape) != 2
            or any(dim <= 0 for dim in shape)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise RuntimeError(
                f"AURA unit checkpoint state for {name} has invalid "
                "source-weight shape/hash"
            )
        if source_weight_identity is None:
            raise RuntimeError(
                "AURA source-weight identity restore has no destination"
            )
        source_weight_identity[name] = {
            "shape": shape,
            "sha256": digest,
        }
    elif raw_source is not None:
        raise RuntimeError(
            f"AURA unit checkpoint state for {name} unexpectedly carries a "
            "production-anchor source identity"
        )


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
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    unit_filter: str | None = None,
    checkpoint_identity_extra: Mapping[str, object] | None = None,
    profile=None,
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
    if resume and checkpoint_dir is None:
        raise ValueError("resume=True requires checkpoint_dir")
    profile = resolve_routed_expert_profile(model, profile)
    omitted_packed_experts = _guard_packed_expert_coverage(
        model, profile, allow_omission=allow_packed_expert_omission)
    _dw_torch_dtype = torch.float32 if str(dw_dtype) == "float32" else torch.bfloat16
    device = next(model.parameters()).device
    linears = _target_linears(
        model, include_lm_head=include_lm_head, profile=profile)
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
    if unit_filter:
        try:
            unit_pattern = re.compile(str(unit_filter))
        except re.error as exc:
            raise ValueError(f"invalid unit_filter regex {unit_filter!r}") from exc
        linears = {
            name: mod for name, mod in linears.items()
            if unit_pattern.search(name)
        }
        if not linears:
            raise ValueError(
                f"unit_filter {unit_filter!r} matched no AURA Linear qnames"
            )
    names = list(linears.keys())
    fmts = [fr.canonical_format_name(f) for f in formats]
    nonzero_fmts = [f for f in fmts if f not in _ZERO_COST_FORMATS]
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
    if checkpoint_dir is not None and not cb_provenance:
        raise RuntimeError(
            "AURA durable checkpointing requires a value-bearing CB "
            "ProductionWeightCache identity; refusing model/cache name-gated "
            "resume"
        )
    if checkpoint_dir is not None:
        _validate_aura_checkpoint_cache_identity(production_cache)
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

    checkpoint_root: Path | None = None
    checkpoint_identity_sha256: str | None = None
    completed_checkpoint_units: set[str] = set()
    checkpoint_git_commit: str | None = None
    if checkpoint_dir is not None:
        checkpoint_git_commit = _checkpoint_git_commit()
        checkpoint_identity = _build_aura_checkpoint_identity(
            model=model,
            calib_ids=calib_ids,
            names=names,
            linears=linears,
            formats=fmts,
            chunks=chunks,
            n_probes=n_probes,
            token_scope=token_scope,
            temperature=temperature,
            seed_base=seed_base,
            dw_dtype=dw_dtype,
            include_lm_head=include_lm_head,
            hook_harvest=hook_harvest,
            allow_packed_expert_omission=allow_packed_expert_omission,
            probe_microbatch=probe_microbatch,
            collect_col_energy=collect_col_energy,
            require_production_cache=require_production_cache,
            production_cache=production_cache,
            cb_provenance=cb_provenance,
            git_commit=checkpoint_git_commit,
            extra_identity=checkpoint_identity_extra,
        )
        checkpoint_root, checkpoint_identity_sha256, completed_states = (
            _prepare_aura_checkpoints(
                checkpoint_dir,
                resume=resume,
                identity=checkpoint_identity,
                names=names,
            )
        )
        for name in names:
            state = completed_states.get(name)
            if state is None:
                continue
            _restore_aura_unit_state(
                name,
                state,
                nonzero_formats=nonzero_fmts,
                n_probes=n_probes,
                collect_col_energy=collect_col_energy,
                s2=s2,
                s4=s4,
                x2_probe=x2_probe,
                dw_src=dw_src,
                g_trace=g_trace,
                col_energy=col_energy,
            )
            completed_checkpoint_units.add(name)
        if completed_checkpoint_units:
            _log(
                f"checkpoint resume: validated {len(completed_checkpoint_units)}/"
                f"{len(names)} completed Linear units"
            )

    for ci, original_chunk in enumerate(chunks):
        pending_chunk = [
            name for name in original_chunk
            if name not in completed_checkpoint_units
        ]
        if not pending_chunk:
            if checkpoint_root is not None:
                _log(f"chunk {ci+1}/{len(chunks)}: fully checkpointed; skip")
            continue
        # A process can die between the atomic publication of two unit shards
        # from the same completed chunk. Re-arm the complete original chunk so
        # each pending unit sees the identical autograd topology/kernel work it
        # saw uninterrupted; already-published siblings are harvested and then
        # discarded, never accumulated twice.
        chunk = list(original_chunk)
        pending_names = set(pending_chunk)
        for n in chunk:
            linears[n].weight.requires_grad_(True)
        # Precompute dW_{i,f} (fp32 delta, stored bf16) for this chunk only.
        dW: dict[tuple[str, str], torch.Tensor] = {}
        with torch.no_grad():
            for f in nonzero_fmts:
                for n in pending_chunk:
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
        for n in pending_chunk:
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
                 f"pending={len(pending_chunk)}/{len(chunk)} Linears, "
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
            if name not in pending_names:
                return
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
        if checkpoint_root is not None:
            assert checkpoint_identity_sha256 is not None
            # Publish one durable accumulator shard per completed Linear only
            # after all of its probes have been harvested. A kill between unit
            # renames loses at most the unpublished units in this chunk.
            for n in pending_chunk:
                state = _aura_unit_state(
                    n,
                    nonzero_fmts,
                    s2=s2,
                    s4=s4,
                    x2_probe=x2_probe,
                    dw_src=dw_src,
                    g_trace=g_trace,
                    col_energy=col_energy,
                )
                _write_aura_unit_checkpoint(
                    checkpoint_root,
                    qname=n,
                    identity_sha256=checkpoint_identity_sha256,
                    state=state,
                )
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
            costs[n][f] = {
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
            "git_commit": (
                checkpoint_git_commit
                if checkpoint_git_commit is not None
                else _git_commit()
            ),
            **(
                cb_provenance
                if cb_provenance
                else cb_cost_provenance(fmts)
            ),
        },
    }


def _assemble_streamed_aura_payload(
    *,
    linears: Mapping[str, nn.Linear],
    names: Sequence[str],
    formats: Sequence[str],
    formats_by_qname: Mapping[str, Sequence[str]],
    unmeasured_formats_by_qname: Mapping[str, Sequence[str]],
    n_probes: int,
    token_scope: str,
    seed_base: int,
    temperature: float,
    dw_dtype: str,
    measurement_dtype: torch.dtype,
    n_linear_chunks: int,
    calib_ids: torch.Tensor,
    omitted_packed_experts: Sequence[str],
    cb_provenance: Mapping[str, object],
    checkpoint_git_commit: str | None,
    collect_col_energy: bool,
    s2: Mapping[tuple[str, str], float],
    s4: Mapping[tuple[str, str], float],
    x2_probe: Mapping[tuple[str, str], list[float]],
    dw_src: Mapping[tuple[str, str], str],
    g_trace: Mapping[str, float],
    col_energy: Mapping[str, torch.Tensor],
    weight_mse_diagnostic: Mapping[tuple[str, str], float],
) -> dict:
    inv = 1.0 / float(n_probes)
    stats: dict[str, dict] = {}
    costs: dict[str, dict] = {}
    for name in names:
        mod = linears[name]
        stats[name] = {
            "h_trace": g_trace[name] * inv,
            "n_params": int(mod.weight.numel()),
            "in_features": int(getattr(mod, "in_features", mod.weight.shape[1])),
            "out_features": int(getattr(mod, "out_features", mod.weight.shape[0])),
            "n_probes": int(n_probes),
        }
        if collect_col_energy and name in col_energy:
            stats[name]["fisher_col"] = (
                col_energy[name].float() * inv
            ).detach().cpu()
        costs[name] = {}
        for fmt in formats_by_qname[name]:
            if fmt in _ZERO_COST_FORMATS:
                costs[name][fmt] = {
                    "predicted_dloss": 0.0,
                    "output_mse_measured": False,
                    "cost_source": "aura_passthrough_zero",
                }
                continue
            if fmt in unmeasured_formats_by_qname.get(name, ()):
                # Identity-bearing terminals can be semantically real yet
                # outside this measurement currency.  Retain them in the
                # exact streamed plan, but never invent a scalar row.
                continue
            key = (name, fmt)
            if key not in s2:
                continue
            mean_x2 = inv * s2[key]
            variance = (
                max(
                    (s4[key] - n_probes * mean_x2 * mean_x2)
                    / (n_probes - 1),
                    0.0,
                )
                if n_probes >= 2
                else 0.0
            )
            row = {
                "predicted_dloss": 0.5 * mean_x2,
                "predicted_dloss_stderr": 0.5 * math.sqrt(variance * inv),
                "x2_per_probe": x2_probe[key],
                "dw_source": dw_src[key],
                "output_mse_measured": False,
                "cost_source": "aura",
            }
            if dw_src[key] == "production_render":
                row.update({
                    "production_anchor_measured": True,
                    "production_anchor_zero": mean_x2 == 0.0,
                })
            if key in weight_mse_diagnostic:
                row.update({
                    # Diagnostic only: AURA remains the allocator's sole cost
                    # currency. This same-render value tests within-basis
                    # currency-ratio invariance on fitting-panel cells.
                    "weight_mse_diagnostic": float(
                        weight_mse_diagnostic[key]
                    ),
                    "weight_mse_diagnostic_normalization": "mean_per_weight",
                    "weight_mse_is_cost_input": False,
                })
            costs[name][fmt] = row

    calib_cpu = calib_ids.detach().cpu().contiguous()
    return {
        "schema": SCHEMA,
        "n_probes": n_probes,
        "formats": list(formats),
        "token_scope": token_scope,
        "stats": stats,
        "costs": costs,
        "provenance": {
            "seed_base": int(seed_base),
            "temperature": float(temperature),
            "dw_dtype": str(dw_dtype),
            "measurement_dtype": str(measurement_dtype),
            "include_lm_head": False,
            "n_linear_chunks": int(n_linear_chunks),
            "calib_shape": list(calib_ids.shape),
            "calib_sha256": hashlib.sha256(
                calib_cpu.numpy().tobytes()
            ).hexdigest(),
            "calib_hash": calibration_data_hash(calib_ids),
            "calib_hashes": [calibration_data_hash(calib_ids)],
            "omitted_packed_experts": list(omitted_packed_experts),
            "dw_rendered_rows": int(sum(
                value == "rendered" for value in dw_src.values()
            )),
            "dw_production_anchor_rows": int(sum(
                value == "production_render" for value in dw_src.values()
            )),
            "dw_rtn_fallback_rows": int(sum(
                value == "rtn" for value in dw_src.values()
            )),
            "git_commit": checkpoint_git_commit or _git_commit(),
            "streaming": True,
            "streamed_gradient_harvest": (
                "post_accumulate_per_parameter"
            ),
            "streamed_cotangent_rollover": "in_place_per_probe",
            "streamed_boundary_release": "progressive_reverse",
            **(
                dict(cb_provenance)
                if cb_provenance
                else cb_cost_provenance(formats)
            ),
        },
    }


def compute_aura_cost_streamed(
    runner,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    *,
    n_probes: int = 16,
    probe_microbatch: int = 0,
    token_scope: str = "all",
    temperature: float = 1.0,
    production_cache: object | None = None,
    min_free_gib: float = 20.0,
    seed_base: int = 7000,
    assert_bf16_passthrough: bool = False,
    require_production_cache: bool = False,
    dw_dtype: str = "bfloat16",
    include_lm_head: bool = False,
    allow_packed_expert_omission: bool = False,
    collect_col_energy: bool = False,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    unit_filter: str | None = None,
    model_identity: Mapping[str, object] | None = None,
    checkpoint_identity_extra: Mapping[str, object] | None = None,
    formats_by_qname: Mapping[str, Sequence[str]] | None = None,
    unmeasured_formats_by_qname: Mapping[str, Sequence[str]] | None = None,
    anchor_renderer: object | None = None,
    include_routed_experts: bool = False,
    diagnostic_weight_mse_pairs: Sequence[tuple[str, str]] | None = None,
    joint_activation: bool = False,
    joint_projection_backend=None,
    profile=None,
) -> dict:
    """Layer-streamed KL-adjoint with identity-bound per-Linear shards.

    The source forward stores only decoder-boundary activations.  Its reverse
    pass then installs one decoder layer, recomputes that layer for every KL
    probe, projects each weight gradient onto production ``dW`` in fp32, and
    unloads the layer.  Thus source weights, gradients, and rendered deltas
    are bounded by one decoder layer; no autograd graph can retain the whole
    model. ``probe_microbatch > 0`` opts joint AURA into complete-sequence
    partitions with versioned global-row probes. Local signed terms and weight
    gradients sum across all partitions before squaring; full-vocabulary GPU
    tensors are bounded by the partition. CPU boundaries and shared pass state
    still cover the full calibration. Checkpoints bind the execution partition
    because floating-point kernels may round differently with batch shape.
    The resident :func:`compute_aura_cost` path is deliberately unchanged.
    """
    if type(probe_microbatch) is not int or probe_microbatch < 0:
        raise ValueError("probe_microbatch must be a nonnegative integer")
    if calib_ids.ndim != 2 or min(calib_ids.shape) < 1:
        raise ValueError("streamed AURA needs nonempty [rows, sequence] calibration")
    if joint_projection_backend is not None and not joint_activation:
        raise ValueError("joint_projection_backend requires joint_activation")
    if probe_microbatch and not joint_activation:
        raise ValueError("streamed probe_microbatch currently requires joint_activation")
    batch_rows = min(probe_microbatch or len(calib_ids), len(calib_ids))
    row_offsets = list(range(0, len(calib_ids), batch_rows))
    probe_layout = None
    if probe_microbatch:
        from prismaquant.kl_fisher import ROW_PROBE_LAYOUT
        sequence_length = int(calib_ids.shape[1])
        selected_tokens = {"all": sequence_length, "last": 1,
                           "causal": sequence_length - 1}.get(token_scope, 0)
        if selected_tokens < 1:
            raise ValueError("invalid token scope or sequence length for streamed probes")
        probe_layout = {
            "schema": ROW_PROBE_LAYOUT,
            "global_rows": len(calib_ids),
            "sequence_length": sequence_length,
            "selected_tokens_per_row": selected_tokens,
            "vocab_size": int(runner._head().weight.shape[0]),
            "token_scope": token_scope,
            "global_token_count": len(calib_ids) * selected_tokens,
        }
    execution_partition = {
        "schema": "prismaquant.aura.streamed_microbatch.v1",
        "requested_rows": probe_microbatch,
        "effective_rows": batch_rows,
        "partition_count": len(row_offsets),
        "row_order": "contiguous_complete_sequences",
        "gradient_diagnostics": "sum_fp32_gradients_before_norm",
    } if probe_microbatch else None
    if n_probes < 1:
        raise ValueError(f"n_probes must be >= 1, got {n_probes!r}")
    if resume and checkpoint_dir is None:
        raise ValueError("resume=True requires checkpoint_dir")
    if joint_activation:
        if production_cache is None and anchor_renderer is None:
            raise ValueError("joint AURA requires production-rendered weights")
        if type(n_probes) is not int or n_probes < 2:
            raise ValueError("joint AURA requires at least two probes for sampling uncertainty")
        if model_identity is None:
            raise ValueError("joint AURA requires exact model_identity for its cotangents")
        # Rounding dW to BF16 drops part of the requested local residual.
        dw_dtype = "float32"
    profile = resolve_routed_expert_profile(
        runner.model, profile or runner.profile
    )
    packed_projections = []
    if include_routed_experts:
        routed_targets = set(_packed_expert_targets(runner.model, profile))
        unpacked_targets = {
            member.qname
            for member in profile_declared_unpacked_expert_linears(
                runner.model, profile
            )
        }
        if joint_activation:
            packed_projections = profile_declared_packed_expert_projections(runner.model, profile)
        packed_targets = {member.packed_qname for member in packed_projections}
        unsupported = sorted(routed_targets - unpacked_targets - packed_targets)
        if unsupported:
            raise RuntimeError(
                "streamed AURA can include routed experts only when the "
                "profile exposes them as per-expert nn.Linear modules; "
                f"unsupported profile-declared targets={unsupported[:8]}"
            )
        omitted_packed_experts: list[str] = []
    else:
        omitted_packed_experts = _guard_packed_expert_coverage(
            runner.model,
            profile,
            allow_omission=allow_packed_expert_omission,
        )
    if include_lm_head:
        raise RuntimeError(
            "streamed AURA currently requires the profile-pinned lm_head to "
            "remain resident BF16; --include-lm-head is unsupported"
        )
    linears = _target_linears(
        runner.model,
        include_lm_head=False,
        include_routed_experts=include_routed_experts,
        profile=profile,
    )
    if packed_projections:
        if anchor_renderer is not None:
            raise RuntimeError(
                "packed joint AURA requires decoded ProductionWeightCache entries; "
                "the on-demand production anchor renderer supports only nn.Linear modules"
            )
        linears.update({member.qname: member for member in packed_projections})
    fmts = list(dict.fromkeys(
        fr.canonical_format_name(fmt) for fmt in formats
    ))
    raw_plan: dict[str, tuple[str, ...]] | None = None
    if formats_by_qname is not None:
        raw_plan = {}
        for raw_name, raw_formats in formats_by_qname.items():
            name = str(raw_name)
            if isinstance(raw_formats, (str, bytes)):
                raise TypeError(
                    f"streamed AURA format plan {name!r} must be a sequence"
                )
            planned = tuple(dict.fromkeys(
                fr.canonical_format_name(fmt) for fmt in raw_formats
            ))
            if not planned:
                raise ValueError(
                    f"streamed AURA format plan has no formats for {name}"
                )
            unknown_formats = sorted(set(planned) - set(fmts))
            if unknown_formats:
                raise ValueError(
                    f"streamed AURA format plan for {name} contains formats "
                    f"outside the requested union: {unknown_formats}"
                )
            raw_plan[name] = planned
        unknown_names = sorted(set(raw_plan) - set(linears))
        if unknown_names:
            raise ValueError(
                "streamed AURA format plan contains qnames that are not "
                f"eligible live Linears; sample={unknown_names[:8]}"
            )
        linears = {
            name: linears[name] for name in raw_plan
        }
    raw_unmeasured: dict[str, tuple[str, ...]] = {}
    if unmeasured_formats_by_qname is not None:
        if raw_plan is None:
            raise ValueError(
                "unmeasured streamed formats require formats_by_qname"
            )
        for raw_name, raw_formats in unmeasured_formats_by_qname.items():
            name = str(raw_name)
            if name not in raw_plan:
                raise ValueError(
                    f"unmeasured streamed format plan has unknown qname {name!r}"
                )
            if isinstance(raw_formats, (str, bytes)):
                raise TypeError(
                    f"unmeasured streamed formats for {name!r} must be a sequence"
                )
            values = tuple(dict.fromkeys(
                fr.canonical_format_name(fmt) for fmt in raw_formats
            ))
            unexpected = sorted(set(values) - set(raw_plan[name]))
            if unexpected:
                raise ValueError(
                    f"unmeasured streamed formats for {name} are outside "
                    f"its exact plan: {unexpected}"
                )
            raw_unmeasured[name] = values
    if unit_filter:
        try:
            pattern = re.compile(str(unit_filter))
        except re.error as exc:
            raise ValueError(f"invalid unit_filter regex {unit_filter!r}") from exc
        linears = {
            name: mod for name, mod in linears.items() if pattern.search(name)
        }
        if not linears:
            raise ValueError(
                f"unit_filter {unit_filter!r} matched no AURA Linear qnames"
            )
    names = list(linears)
    if not names:
        raise RuntimeError("streamed AURA found no smooth Linear targets")
    names_by_layer: dict[int, list[str]] = {}
    for name in names:
        layer = runner.layer_index_for_qname(name)
        names_by_layer.setdefault(layer, []).append(name)

    def _refresh_packed_layer_views(layer):
        members = [linears[name] for name in names_by_layer.get(layer, ())
                   if isinstance(linears[name], PackedExpertProjection)]
        if members:
            linears.update({member.qname: member for member in
                            refresh_packed_expert_projections(members, profile)})

    def _source_parameter(target):
        return target.parameter if isinstance(target, PackedExpertProjection) else target.weight

    def _source_gradient(target, gradient):
        return target.gradient_view(gradient) if isinstance(target, PackedExpertProjection) else gradient

    unit_formats: dict[str, tuple[str, ...]] = {}
    render_formats: dict[str, tuple[str, ...]] = {}
    for name in names:
        planned = raw_plan[name] if raw_plan is not None else tuple(fmts)
        unmeasured = set(raw_unmeasured.get(name, ()))
        measured = tuple(
            fmt for fmt in planned
            if fmt not in _ZERO_COST_FORMATS and fmt not in unmeasured
        )
        unit_formats[name] = tuple(planned)
        render_formats[name] = measured
    nonzero_fmts = list(dict.fromkeys(
        fmt for name in names for fmt in render_formats[name]
    ))

    joint_probe_identity = None
    joint_run_identity = None
    joint_rows: dict[str, dict[str, dict]] = {}
    joint_components: dict[tuple[str, str], list[dict]] = {}
    joint_operators: dict[tuple[str, str], dict] = {}
    joint_source_tensors: dict[str, dict] = {}
    joint_cache_renders: dict[str, dict[str, dict]] = {}
    joint_prefetch_stats: list[dict] = []
    if joint_activation:
        from prismaquant.joint_projection_backend import prewarm_projection_backend
        joint_projection_backend = prewarm_projection_backend(joint_projection_backend, device=runner.device)
        from prismaquant.cost_streaming import validate_streamed_model_identity
        from prismaquant.joint_aura import (
            SignedJointProjectionLease, activation_identity, arithmetic_identity,
            identity_sha256, make_joint_aura_entry, prefetch_joint_cache, squared_signed,
            source_execution_identity,
            validate_joint_aura_entry,
        )
        from prismaquant.production_weight_cache import _cb_cache_tensor_identity

        joint_probe_identity = {
            "schema": "prismaquant.joint_aura.probes.v2",
            "source_model": validate_streamed_model_identity(model_identity, where="joint AURA"),
            "calibration_sha256": hashlib.sha256(calib_ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            "calibration_shape": list(calib_ids.shape),
            "calibration_dtype": str(calib_ids.dtype),
            "n_probes": n_probes, "seed_base": seed_base,
            "token_scope": token_scope, "temperature": temperature,
            "distribution": "rademacher", "normalization": "global_kl_fisher",
            "producer_source_sha256": _aura_source_sha256(),
            "source_execution": source_execution_identity(runner.model),
            "arithmetic": arithmetic_identity(runner.dtype, joint_projection_backend),
        }
        if probe_layout is not None:
            joint_probe_identity["noise_layout"] = probe_layout
            # RNG row coordinates are partition independent; the source
            # arithmetic is not. Existing paired/assignment consumers compare
            # this complete probe identity and must refuse mixed batch shapes.
            joint_probe_identity["arithmetic"]["execution_partition"] = execution_partition
        # Hash actual decoded production outputs before checkpoint admission,
        # in layer-bounded prefetch windows. This is identity preparation,
        # outside the cotangent/projection hot path; no tensor copy is retained.
        if production_cache is not None:
            for layer_names in names_by_layer.values():
                joint_prefetch_stats.append(prefetch_joint_cache(
                    production_cache, layer_names, render_formats,
                    max_resident_bytes=max(0, int((_free_gib() - min_free_gib) * 1024**3)),
                ))
                for name in layer_names:
                    joint_cache_renders[name] = {
                        fmt: _cb_cache_tensor_identity(production_cache.get(name, fmt))
                        for fmt in render_formats[name]
                    }
                production_cache.compact_for_pickle()
        joint_run_identity = {
            "schema": "prismaquant.joint_aura.run.v2",
            "probe_identity": joint_probe_identity,
            "cached_rendered_weights": joint_cache_renders,
            "activation_contracts": ({
                name: {fmt: activation_identity(fr.get_format(fmt), production_cache.activation_max_abs or {}, name)
                       for fmt in unit_formats[name]}
                for name in names
            } if production_cache is not None else None),
        }

    anchor_identity: Mapping[str, object] | None = None
    if anchor_renderer is not None:
        if production_cache is not None:
            raise ValueError(
                "streamed production anchors cannot also consume a "
                "materialized production_cache"
            )
        if formats_by_qname is None:
            raise ValueError(
                "streamed production anchors require an exact "
                "formats_by_qname anchor plan"
            )
        if str(dw_dtype) not in {"bfloat16", "float32"}:
            raise ValueError(
                "streamed production-anchor AURA requires dw_dtype to be "
                "'bfloat16' or 'float32'"
            )
        empty = sorted(name for name in names if not render_formats[name])
        if empty:
            raise ValueError(
                "every streamed production-anchor unit needs a real rendered "
                f"format; passthrough-only sample={empty[:8]}"
            )
        renderer_plan = getattr(anchor_renderer, "formats_by_qname", None)
        expected_plan = {
            name: tuple(render_formats[name]) for name in names
        }
        observed_plan = (
            {
                str(name): tuple(
                    fr.canonical_format_name(fmt) for fmt in values
                )
                for name, values in renderer_plan.items()
            }
            if isinstance(renderer_plan, Mapping)
            else None
        )
        if observed_plan != expected_plan:
            raise ValueError(
                "streamed production anchor renderer plan differs from the "
                "AURA qname->format plan"
            )
        anchor_identity = getattr(anchor_renderer, "identity", None)
        if not isinstance(anchor_identity, Mapping):
            raise ValueError(
                "streamed production anchor renderer has no exact identity"
            )
    diagnostic_pairs = {
        (str(name), fr.canonical_format_name(fmt))
        for name, fmt in (diagnostic_weight_mse_pairs or ())
    }
    if diagnostic_pairs:
        if anchor_renderer is None:
            raise ValueError(
                "diagnostic weight-MSE is available only from streamed "
                "production-anchor renders"
            )
        expected_anchor_pairs = {
            (name, fmt)
            for name in names
            for fmt in render_formats[name]
        }
        unexpected = sorted(diagnostic_pairs - expected_anchor_pairs)
        if unexpected:
            raise ValueError(
                "diagnostic weight-MSE requests unrendered anchor cells; "
                f"sample={unexpected[:8]}"
            )
    cb_provenance: dict[str, object] = {}
    if any(is_cb_format(fmt) for fmt in nonzero_fmts):
        if anchor_renderer is not None:
            cb_provenance = {
                "cb_cost_provenance_schema": (
                    "prismaquant.aura.production_anchor.v1"
                ),
                "cb_render_identity": anchor_identity.get(
                    "cb_render_identity"
                ),
                "production_anchor_renderer": dict(anchor_identity),
            }
        elif production_cache is None:
            raise RuntimeError(
                "streamed AURA CB cost requires an identity-bound "
                "ProductionWeightCache"
            )
        else:
            from prismaquant.production_weight_cache import (
                production_cache_cb_render_provenance,
            )

            cb_provenance = production_cache_cb_render_provenance(
                production_cache,
                require_for_formats=fmts,
                where="streamed AURA production cache",
            )
    if checkpoint_dir is not None:
        # The checkpoint identity must embed a value-bearing render identity.
        # Two sources qualify: CB provenance (CB menus), or the production-
        # anchor renderer's exact identity (bound below as
        # extra["production_anchor_renderer"], with the qname->format plan
        # asserted equal above). A non-CB menu has no CB identity to bear, so
        # an anchored non-CB run checkpoints on the anchor identity alone.
        if not cb_provenance and anchor_identity is None and not joint_activation:
            raise RuntimeError(
                "streamed AURA durable checkpointing requires a value-bearing "
                "render identity: a CB ProductionWeightCache identity or a "
                "production-anchor renderer with exact identity"
            )
        if anchor_renderer is None and not joint_activation:
            _validate_aura_checkpoint_cache_identity(production_cache)
    if assert_bf16_passthrough and "BF16" in fmts:
        if runner.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(
                "assert_bf16_passthrough: BF16 zero-cost requires a bf16/fp16 "
                f"streamed source, got {runner.dtype}"
            )
    dw_torch_dtype = (
        torch.float32 if str(dw_dtype) == "float32" else torch.bfloat16
    )

    for parameter in runner.model.parameters():
        parameter.requires_grad_(False)

    s2: dict[tuple[str, str], float] = {}
    s4: dict[tuple[str, str], float] = {}
    x2_probe: dict[tuple[str, str], list[float]] = {}
    dw_src: dict[tuple[str, str], str] = {}
    g_trace: dict[str, float] = {}
    col_energy: dict[str, torch.Tensor] = {}
    weight_mse_diagnostic: dict[tuple[str, str], float] = {}
    source_weight_identity: dict[str, dict[str, object]] = {}
    completed_checkpoint_units: set[str] = set()
    checkpoint_root: Path | None = None
    checkpoint_identity_sha256: str | None = None
    checkpoint_git_commit: str | None = None

    ordered_layer_chunks = [
        list(names_by_layer[layer]) for layer in sorted(names_by_layer)
    ]
    if checkpoint_dir is not None:
        checkpoint_git_commit = _checkpoint_git_commit()
        if model_identity is None:
            raise RuntimeError(
                "streamed AURA checkpointing requires exact model_identity; "
                "refusing model-name-gated resume"
            )
        from prismaquant.cost_streaming import (
            validate_streamed_model_identity,
        )

        exact_model_identity = validate_streamed_model_identity(
            model_identity, where="streamed AURA checkpointing"
        )
        extra = dict(checkpoint_identity_extra or {})
        reserved = {
            "joint_aura",
            "streamed_formats_by_qname",
            "unmeasured_streamed_formats_by_qname",
            "production_anchor_renderer",
            "include_routed_experts",
            "diagnostic_weight_mse_pairs",
            "streamed_gradient_harvest",
            "streamed_cotangent_rollover",
            "streamed_boundary_release",
            "streamed_microbatch",
        } & set(extra)
        if reserved:
            raise ValueError(
                "checkpoint_identity_extra attempts to override streamed "
                f"AURA identity fields: {sorted(reserved)}"
            )
        extra["streaming"] = True
        if execution_partition is not None:
            extra["streamed_microbatch"] = execution_partition
        if joint_activation:
            extra["joint_aura"] = joint_run_identity
        extra["streamed_model_identity"] = exact_model_identity
        extra["streamed_formats_by_qname"] = {
            name: list(unit_formats[name]) for name in names
        }
        extra["unmeasured_streamed_formats_by_qname"] = {
            name: list(raw_unmeasured.get(name, ())) for name in names
        }
        extra["include_routed_experts"] = bool(include_routed_experts)
        extra["diagnostic_weight_mse_pairs"] = [
            [name, fmt] for name, fmt in sorted(diagnostic_pairs)
        ]
        extra["streamed_gradient_harvest"] = (
            "post_accumulate_per_parameter"
        )
        extra["streamed_cotangent_rollover"] = "in_place_per_probe"
        extra["streamed_boundary_release"] = "progressive_reverse"
        if anchor_identity is not None:
            extra["production_anchor_renderer"] = dict(anchor_identity)
        identity = _build_aura_checkpoint_identity(
            model=runner.model,
            calib_ids=calib_ids,
            names=names,
            linears=linears,
            formats=fmts,
            chunks=ordered_layer_chunks,
            n_probes=n_probes,
            token_scope=token_scope,
            temperature=temperature,
            seed_base=seed_base,
            dw_dtype=dw_dtype,
            include_lm_head=False,
            hook_harvest=True,
            allow_packed_expert_omission=allow_packed_expert_omission,
            probe_microbatch=probe_microbatch,
            collect_col_energy=collect_col_energy,
            require_production_cache=require_production_cache,
            production_cache=production_cache,
            cb_provenance=cb_provenance,
            git_commit=checkpoint_git_commit,
            extra_identity=extra,
        )
        checkpoint_root, checkpoint_identity_sha256, completed_states = (
            _prepare_aura_checkpoints(
                checkpoint_dir,
                resume=resume,
                identity=identity,
                names=names,
            )
        )
        for name in names:
            state = completed_states.get(name)
            if state is None:
                continue
            _restore_aura_unit_state(
                name,
                state,
                nonzero_formats=render_formats[name],
                n_probes=n_probes,
                collect_col_energy=collect_col_energy,
                s2=s2,
                s4=s4,
                x2_probe=x2_probe,
                dw_src=dw_src,
                g_trace=g_trace,
                col_energy=col_energy,
                diagnostic_weight_mse_pairs=diagnostic_pairs,
                weight_mse_diagnostic=weight_mse_diagnostic,
                require_source_weight_identity=anchor_renderer is not None,
                source_weight_identity=source_weight_identity,
            )
            if joint_activation:
                rows = state.get("joint_aura_rows")
                expected_formats = set(unit_formats[name]) - set(raw_unmeasured.get(name, ()))
                if not isinstance(rows, Mapping) or set(rows) != expected_formats:
                    raise RuntimeError(f"joint AURA checkpoint row scope mismatch for {name}")
                for fmt, row in rows.items():
                    try:
                        if not validate_joint_aura_entry(row):
                            raise ValueError("not a joint row")
                        operator = row["joint_operator_identity"]
                        if row["probe_identity"] != joint_probe_identity or operator["qname"] != name or operator["format"] != fmt:
                            raise ValueError("probe/operator alignment mismatch")
                        if production_cache is not None:
                            if operator["activation"] != joint_run_identity["activation_contracts"][name][fmt]:
                                raise ValueError("activation identity mismatch")
                            if fmt in render_formats[name] and operator["rendered_weight"] != joint_cache_renders[name][fmt]:
                                raise ValueError("actual rendered-weight identity mismatch")
                        if fmt in render_formats[name] and row["x2_per_probe"] != x2_probe[(name, fmt)]:
                            raise ValueError("legacy/joint sample alignment mismatch")
                    except (ValueError, KeyError, TypeError) as exc:
                        raise RuntimeError(f"joint AURA checkpoint identity mismatch for {name}@{fmt}: {exc}") from exc
                joint_rows[name] = dict(rows)
            completed_checkpoint_units.add(name)

    def _finish_streamed_payload() -> dict:
        payload = _assemble_streamed_aura_payload(
            linears=linears,
            names=names,
            formats=fmts,
            formats_by_qname=unit_formats,
            unmeasured_formats_by_qname=raw_unmeasured,
            n_probes=n_probes,
            token_scope=token_scope,
            seed_base=seed_base,
            temperature=temperature,
            dw_dtype=dw_dtype,
            measurement_dtype=runner.dtype,
            n_linear_chunks=len(ordered_layer_chunks),
            calib_ids=calib_ids,
            omitted_packed_experts=omitted_packed_experts,
            cb_provenance=cb_provenance,
            checkpoint_git_commit=checkpoint_git_commit,
            collect_col_energy=collect_col_energy,
            s2=s2,
            s4=s4,
            x2_probe=x2_probe,
            dw_src=dw_src,
            g_trace=g_trace,
            col_energy=col_energy,
            weight_mse_diagnostic=weight_mse_diagnostic,
        )
        if execution_partition is not None:
            payload["provenance"]["streamed_microbatch"] = execution_partition
        if joint_activation:
            if set(joint_rows) != set(names):
                raise RuntimeError("joint AURA incomplete unit coverage")
            payload["costs"] = joint_rows
            payload["provenance"].update({
                "cost_mode": "aura", "joint_activation": True,
                "cost_currency": "joint_aura_predicted_dloss",
                "joint_aura_identity": joint_run_identity,
                "joint_aura_identity_sha256": identity_sha256(joint_run_identity),
                "probe_identity": joint_probe_identity,
                "probe_identity_sha256": identity_sha256(joint_probe_identity),
                "joint_prefetch": joint_prefetch_stats,
                "measurement_status": "research",
                "uncertainty_scope": "probe_sampling_conditional_on_fixed_calibration",
            })
        if anchor_renderer is not None:
            missing_source_identity = sorted(
                set(names) - set(source_weight_identity)
            )
            if missing_source_identity:
                raise RuntimeError(
                    "streamed production-anchor source identity is "
                    f"incomplete; sample={missing_source_identity[:8]}"
                )
            bind_source = getattr(
                anchor_renderer,
                "bind_completed_source_weight_identities",
                None,
            )
            completed_renderer_identity = (
                bind_source(source_weight_identity)
                if callable(bind_source)
                else {
                    **dict(anchor_identity or {}),
                    "source_weights": {
                        "complete": True,
                        "records": {
                            name: dict(source_weight_identity[name])
                            for name in sorted(source_weight_identity)
                        },
                    },
                }
            )
            payload["provenance"]["production_anchor_renderer"] = (
                completed_renderer_identity
            )
            completed_cb_identity = completed_renderer_identity.get(
                "cb_render_identity"
            )
            if isinstance(completed_cb_identity, Mapping):
                payload["provenance"]["cb_render_identity"] = dict(
                    completed_cb_identity
                )
            expected_renders = sum(
                len(render_formats[name]) for name in names
            )
            rendered_now = int(getattr(anchor_renderer, "render_count", 0))
            payload["provenance"].update({
                "production_anchor_expected_renders": expected_renders,
                "production_anchor_rendered_this_invocation": rendered_now,
                "production_anchor_restored_renders": (
                    expected_renders - rendered_now
                ),
                "production_anchor_max_live_rendered": int(getattr(
                    anchor_renderer, "max_live_rendered", 0
                )),
                "production_anchor_no_full_menu_materialization": True,
                "production_anchor_cost_currency": "aura_only",
                "weight_mse_diagnostic_rows": len(
                    weight_mse_diagnostic
                ),
                "weight_mse_diagnostic_is_cost_input": False,
            })
        return payload

    if completed_checkpoint_units == set(names):
        _log(f"checkpoint resume: validated all {len(names)} streamed units; "
             "skip forward/reverse")
        return _finish_streamed_payload()

    def _record_joint_operator(name, fmt, source, rendered):
        if not joint_activation:
            return
        cache_owner = production_cache if production_cache is not None else getattr(anchor_renderer, "cache", None)
        scales = getattr(cache_owner, "activation_max_abs", None) or {}
        activation = activation_identity(fr.get_format(fmt), scales, name)
        rendered_identity = _cb_cache_tensor_identity(rendered)
        if production_cache is not None and fmt in render_formats[name]:
            if rendered_identity != joint_cache_renders[name][fmt] or activation != joint_run_identity["activation_contracts"][name][fmt]:
                raise RuntimeError(f"joint AURA actual render/activation identity changed for {name}@{fmt}")
        if name not in joint_source_tensors:
            joint_source_tensors[name] = _cb_cache_tensor_identity(source)
        joint_operators[(name, fmt)] = {
            "schema": "prismaquant.joint_aura.operator.v2",
            "qname": name, "format": fmt,
            "source_weight": joint_source_tensors[name],
            "rendered_weight": rendered_identity,
            "activation": activation,
            "arithmetic": joint_probe_identity["arithmetic"],
            "probe_identity_sha256": identity_sha256(joint_probe_identity),
        }
        joint_components[(name, fmt)] = []

    # Identity validation above is intentionally before the first model
    # forward: a mismatched resume is a refusal, never a recomputation.
    _log(f"boundary capture: calib {tuple(calib_ids.shape)} in "
         f"{len(row_offsets)} partition(s) across {runner.num_layers} layers ...")
    capture_started = time.time()
    batches = []
    for offset in row_offsets:
        available_gib = _free_gib()
        if available_gib < min_free_gib:
            raise RuntimeError(f"free UMA {available_gib:.1f} < floor {min_free_gib:.1f}; "
                               f"abort before calibration row {offset}")
        batches.append(runner.capture_boundaries(calib_ids[offset:offset + batch_rows]))
    _log(f"boundary capture done in {(time.time() - capture_started) / 60:.1f} "
         f"min; starting {n_probes}-probe tail cotangents")
    device = runner.device
    dtype = runner.dtype

    # Existing boundary and shared-state mechanisms, one instance per complete
    # sequence partition. Host boundary/cotangent storage still scales with the
    # full calibration; only GPU activations and full-vocabulary tensors are
    # bounded by batch_rows. No second residency or spill cache is introduced.
    from prismaquant.sensitivity_probe import (
        SharedStateCotangents,
        kv_cotangent_path_enabled,
    )
    cotangents = [[SharedStateCotangents(enabled=kv_cotangent_path_enabled())
                  for _ in batches] for _ in range(n_probes)]
    grad_outs: list[list[torch.Tensor]] = [[] for _ in range(n_probes)]
    for batch_index, batch in enumerate(batches):
        for probe_index in range(n_probes):
            tail = batch.activations_cpu[-1].to(
                device=device, dtype=dtype
            ).detach().requires_grad_(True)
            logits = runner.tail_logits(batch, tail)
            if probe_layout is not None and list(logits.shape) != [
                len(batch.input_ids), int(calib_ids.shape[1]), probe_layout["vocab_size"]
            ]:
                raise RuntimeError("streamed AURA tail differs from bound probe geometry")
            probe = fisher_probe_scalar(
                logits, seed=seed_base + probe_index, token_scope=token_scope,
                temperature=temperature, distribution="rademacher",
                **({"token_count_override": probe_layout["global_token_count"],
                    "global_row_offset": row_offsets[batch_index]}
                   if probe_layout is not None else {}),
            )
            probe.backward()
            if tail.grad is None:
                raise RuntimeError("streamed AURA tail produced no cotangent")
            grad_outs[probe_index].append(tail.grad.detach().to("cpu"))
            del logits, probe, tail
        batch.activations_cpu[-1] = torch.empty(0)

    reverse_started = time.time()
    reverse_layers_done = 0
    for layer in reversed(range(runner.num_layers)):
        runner.context.install(
            layer,
            require_prefetched=runner.require_prefetched_residency,
        )
        _refresh_packed_layer_views(layer)
        # Forward boundary capture leaves the final lookahead window hot.
        # Keep that pipeline moving in the direction of this traversal: the
        # existing StreamingContext/LayerCache loads the next reverse layer
        # while the current layer performs its expensive anchor render and
        # adjoint probes.  Without this call, every layer after the retained
        # tail window falls through ensure_loaded()'s synchronous cold path.
        runner.schedule_reverse_prefetch(layer)
        pending = [
            name for name in names_by_layer.get(layer, [])
            if name not in completed_checkpoint_units
        ]
        d_weights: dict[tuple[str, str], torch.Tensor] = {}
        parameter_members = {}
        parameters = {}
        try:
            for name in pending:
                parameter = _source_parameter(linears[name])
                parameter.requires_grad_(True)
                parameters[id(parameter)] = parameter
                parameter_members.setdefault(id(parameter), []).append(name)
                g_trace[name] = 0.0
            with torch.no_grad():
                if anchor_renderer is not None and pending:
                    render_layer = getattr(
                        anchor_renderer, "render_layer", None
                    )
                    if not callable(render_layer):
                        raise TypeError(
                            "streamed production anchor renderer has no "
                            "callable render_layer()"
                        )
                    layer_modules = {
                        name: linears[name]
                        for name in names_by_layer.get(layer, [])
                    }
                    expected_pairs = {
                        (name, fmt)
                        for name in pending
                        for fmt in render_formats[name]
                    }
                    transient_render = getattr(
                        anchor_renderer, "render_layer_transient", None
                    )

                    def _consume_anchor(
                        *, qname, fmt, reference_weight, rendered_weight,
                        render_score,
                    ):
                        del render_score
                        name = str(qname)
                        canonical_fmt = fr.canonical_format_name(fmt)
                        key = (name, canonical_fmt)
                        if key not in expected_pairs:
                            raise RuntimeError(
                                "streamed production anchor consumer received "
                                f"unexpected pair {key}"
                            )
                        if key in d_weights:
                            raise RuntimeError(
                                "streamed production anchor consumer received "
                                f"duplicate pair {key}"
                            )
                        if not isinstance(rendered_weight, torch.Tensor):
                            raise TypeError(
                                "streamed production anchor renderer returned "
                                f"{type(rendered_weight).__name__} for "
                                f"{name}@{canonical_fmt}, expected Tensor"
                            )
                        if tuple(rendered_weight.shape) != tuple(
                            reference_weight.shape
                        ):
                            raise RuntimeError(
                                "streamed production anchor shape differs for "
                                f"{name}@{canonical_fmt}: rendered="
                                f"{tuple(rendered_weight.shape)} source="
                                f"{tuple(reference_weight.shape)}"
                            )
                        _record_joint_operator(name, canonical_fmt, reference_weight, rendered_weight)
                        d_weights[key] = _stored_production_anchor_delta(
                            rendered_weight,
                            reference_weight,
                            storage_dtype=dw_torch_dtype,
                        )
                        if key in diagnostic_pairs:
                            weight_mse_diagnostic[key] = float(
                                d_weights[key].float().pow(2).mean().item()
                            )
                        dw_src[key] = "production_render"
                        s2[key] = 0.0
                        s4[key] = 0.0
                        x2_probe[key] = []
                        return {
                            "operation": "fp32_subtract_then_store",
                            "storage_dtype": str(dw_torch_dtype),
                        }

                    if callable(transient_render):
                        observed_pairs = {
                            (str(name), fr.canonical_format_name(fmt))
                            for name, fmt in transient_render(
                                layer=layer,
                                modules=layer_modules,
                                formats_by_qname={
                                    name: render_formats[name]
                                    for name in pending
                                },
                                consume_render=_consume_anchor,
                                consumer_identity=(
                                    {**AURA_PRODUCTION_ANCHOR_DELTA_CONSUMER_IDENTITY,
                                     "storage_dtype": "torch.float32"}
                                    if joint_activation else AURA_PRODUCTION_ANCHOR_DELTA_CONSUMER_IDENTITY
                                ),
                            )
                        }
                    else:
                        rendered_anchors = render_layer(
                            layer=layer,
                            modules=layer_modules,
                            formats_by_qname={
                                name: render_formats[name]
                                for name in pending
                            },
                        )
                        if not isinstance(rendered_anchors, Mapping):
                            raise TypeError(
                                "streamed production anchor render_layer() "
                                "must return a Mapping"
                            )
                        observed_pairs = {
                            (str(name), fr.canonical_format_name(fmt))
                            for name, fmt in rendered_anchors
                        }
                        # Compatibility for injected/research renderers. The
                        # production renderer above consumes each pair before
                        # the next pair is materialized.
                        rendered_anchor_pool = (
                            rendered_anchors
                            if isinstance(rendered_anchors, MutableMapping)
                            else dict(rendered_anchors)
                        )
                        del rendered_anchors
                        for name in pending:
                            weight = linears[name].weight.data
                            for fmt in render_formats[name]:
                                key = (name, fmt)
                                rendered = rendered_anchor_pool.pop(key)
                                _consume_anchor(
                                    qname=name,
                                    fmt=fmt,
                                    reference_weight=weight,
                                    rendered_weight=rendered,
                                    render_score={},
                                )
                                del rendered
                        del rendered_anchor_pool
                    if observed_pairs != expected_pairs:
                        raise RuntimeError(
                            "streamed production anchor renderer returned a "
                            "different pair set: "
                            f"missing={sorted(expected_pairs - observed_pairs)[:8]} "
                            f"unexpected={sorted(observed_pairs - expected_pairs)[:8]}"
                        )
                    for name in pending:
                        weight = linears[name].weight.data
                        source_lookup = getattr(
                            anchor_renderer,
                            "source_weight_identity_for",
                            None,
                        )
                        if callable(source_lookup):
                            source_weight_identity[name] = dict(
                                source_lookup(name)
                            )
                        else:
                            # Test/injected renderers may not own a CB cache.
                            # Production uses the renderer's already-computed
                            # source binding and therefore never hashes twice.
                            from prismaquant.production_weight_cache import (
                                _source_weight_value_identity,
                            )

                            source_shape, source_sha256 = (
                                _source_weight_value_identity(weight)
                            )
                            source_weight_identity[name] = {
                                "shape": source_shape,
                                "sha256": source_sha256,
                            }
                    _release_streamed_anchor_allocator_cache(device)
                else:
                    if joint_activation and pending:
                        joint_prefetch_stats.append(prefetch_joint_cache(
                            production_cache, pending, render_formats,
                            max_resident_bytes=max(0, int((_free_gib() - min_free_gib) * 1024**3)),
                        ))
                    for name in pending:
                        for fmt in render_formats[name]:
                            if joint_activation:
                                weight = linears[name].weight.data
                                rendered = production_cache.get(name, fmt)
                                _record_joint_operator(name, fmt, weight, rendered)
                                result = (rendered.to(weight.device, torch.float32) - weight.float(), "rendered")
                                del rendered
                            else:
                                result = _delta_w(
                                    name, fmt, linears[name].weight.data,
                                    production_cache, strict=require_production_cache,
                                )
                            if result is None:
                                continue
                            delta, source = result
                            key = (name, fmt)
                            d_weights[key] = delta.to(dw_torch_dtype)
                            dw_src[key] = source
                            s2[key] = 0.0
                            s4[key] = 0.0
                            x2_probe[key] = []
                if joint_activation:
                    for name in pending:
                        for fmt in unit_formats[name]:
                            if fmt in _ZERO_COST_FORMATS:
                                weight = linears[name].weight.data
                                _record_joint_operator(name, fmt, weight, weight)
            if anchor_renderer is None:
                compact = getattr(production_cache, "compact_for_pickle", None)
                if callable(compact):
                    compact()

            # Project each fully accumulated parameter gradient from its
            # post-accumulate hook, then clear ``param.grad`` immediately.
            # A routed layer can otherwise retain another complete 12-GiB
            # BF16 parameter plane on top of its source weights and dW menu.
            # Each reduction below is per parameter, so performing it when
            # that parameter's AccumulateGrad node completes is numerically
            # identical to the old post-backward qname loop.
            harvested: set[str] = set()
            accumulated_gradients: dict[str, torch.Tensor] = {}

            def _consume_streamed_gradient(name, gradient):
                if probe_microbatch:
                    with torch.no_grad():
                        if name in accumulated_gradients:
                            accumulated_gradients[name].add_(gradient)
                        else:
                            accumulated_gradients[name] = gradient.to(torch.float32, copy=True)
                else:
                    _harvest_streamed_gradient(name, gradient)


            def _harvest_streamed_gradient(
                name: str, gradient: torch.Tensor
            ) -> None:
                if name in harvested:
                    raise RuntimeError(
                        "streamed AURA harvested a parameter twice in one "
                        f"probe: {name}"
                    )
                with torch.no_grad():
                    gradient_fp32 = gradient.float()
                    g_trace[name] += float(
                        (gradient_fp32 * gradient_fp32).sum().item()
                    )
                    if collect_col_energy:
                        energy = (
                            gradient_fp32 * gradient_fp32
                        ).sum(dim=0)
                        previous = col_energy.get(name)
                        col_energy[name] = (
                            energy if previous is None else previous + energy
                        )
                    for fmt in render_formats[name]:
                        key = (name, fmt)
                        if key not in d_weights:
                            continue
                        if joint_activation:
                            # Output hooks retain all signed local components;
                            # this parameter hook owns diagnostics only.
                            continue
                        projection = float(
                            (
                                gradient_fp32
                                * d_weights[key].float()
                            ).sum().item()
                        )
                        value = projection ** 2
                        s2[key] += value
                        s4[key] += value * value
                        x2_probe[key].append(value)
                harvested.add(name)

            def _make_streamed_gradient_hook(member_names):
                def _hook(parameter: torch.Tensor) -> None:
                    gradient = parameter.grad
                    if gradient is None:
                        return
                    for name in member_names:
                        if name not in harvested:
                            _consume_streamed_gradient(name, _source_gradient(linears[name], gradient))
                    parameter.grad = None

                return _hook

            hook_handles = []
            joint_lease = None
            # Completed layers still propagate cotangents to pending earlier
            # layers, but have no target device or projections to lease.
            if joint_activation and pending:
                cache_owner = production_cache if production_cache is not None else getattr(anchor_renderer, "cache", None)
                joint_lease = SignedJointProjectionLease(
                    {name: linears[name] for name in pending},
                    {name: {fmt: fr.get_format(fmt) for fmt in render_formats[name]} for name in pending},
                    d_weights, activation_max_abs=getattr(cache_owner, "activation_max_abs", None),
                    projection_backend=joint_projection_backend,
                )
            try:
                if joint_lease is not None:
                    joint_lease.__enter__()
                for parameter_id, parameter in parameters.items():
                    hook_handles.append(
                        parameter.register_post_accumulate_grad_hook(
                            _make_streamed_gradient_hook(parameter_members[parameter_id])
                        )
                    )
                for probe_index in range(n_probes):
                    harvested.clear()
                    accumulated_gradients.clear()
                    if joint_lease is not None:
                        joint_lease.begin_probe()
                    for batch_index, batch in enumerate(batches):
                        available_gib = _free_gib()
                        if available_gib < min_free_gib:
                            raise RuntimeError(
                                f"free UMA {available_gib:.1f} < floor "
                                f"{min_free_gib:.1f}; abort before streamed "
                                f"layer {layer} probe {probe_index + 1}"
                            )
                        incoming_grad = grad_outs[probe_index][batch_index].to(device)
                        x_in = batch.activations_cpu[layer].to(
                            device=device, dtype=dtype
                        ).detach().requires_grad_(True)
                        isolated = profile.isolated_layer_pass_state(
                            batch.shared_pass_state, runner.layers[layer]
                        )
                        isolated = cotangents[probe_index][batch_index].graft(isolated)
                        out = runner.isolated_layer(
                            batch, layer, x_in, pass_state=isolated
                        )
                        roots, root_grads = cotangents[probe_index][batch_index].produced_roots()
                        if roots:
                            torch.autograd.backward(
                                [out, *roots],
                                [incoming_grad, *root_grads],
                            )
                        else:
                            out.backward(incoming_grad)
                        cotangents[probe_index][batch_index].harvest()
                        if x_in.grad is None:
                            raise RuntimeError(
                                f"streamed AURA layer {layer} produced no input "
                                "cotangent"
                            )
                        # Replace this probe's consumed incoming cotangent now.
                        # The former next_grad_outs list retained all 32 incoming
                        # CPU tensors while growing a second complete outgoing
                        # plane. In-place rollover bounds the CPU plane to 32
                        # tensors plus the one result currently being copied.
                        grad_outs[probe_index][batch_index] = x_in.grad.detach().to("cpu")
                        for parameter_id, parameter in parameters.items():
                            gradient = parameter.grad
                            if gradient is not None:
                                # Defensive straggler path for a backend that did
                                # not invoke the post-accumulate hook. It performs
                                # the identical reduction and still frees the
                                # gradient before the next probe.
                                for name in parameter_members[parameter_id]:
                                    if name not in harvested:
                                        _consume_streamed_gradient(name, _source_gradient(linears[name], gradient))
                                parameter.grad = None
                        del (out, x_in, incoming_grad, isolated, roots, root_grads)
                    for name in list(accumulated_gradients):
                        _harvest_streamed_gradient(name, accumulated_gradients.pop(name))
                    for name in pending:
                        if name in harvested:
                            continue
                        # A routed expert not selected by this probe has an
                        # exact zero projection. Record the sample explicitly
                        # so route-sparse and never-routed experts retain the
                        # same K-probe rows as the legacy post-backward loop.
                        for fmt in render_formats[name]:
                            key = (name, fmt)
                            if key in d_weights and not joint_activation:
                                x2_probe[key].append(0.0)
                    if joint_lease is not None:
                        for key, terms in joint_lease.finish_probe().items():
                            joint_components[key].append(terms)
                            # Same squaring as make_joint_aura_entry: the
                            # checkpoint reload compares both lists exactly.
                            value = squared_signed(terms["total"])
                            s2[key] += value
                            s4[key] += value * value
                            x2_probe[key].append(value)
            finally:
                accumulated_gradients.clear()
                if joint_lease is not None:
                    joint_lease.__exit__(None, None, None)
                    joint_lease = None
                for handle in hook_handles:
                    handle.remove()

            if joint_activation:
                if source_execution_identity(runner.model) != joint_probe_identity["source_execution"]:
                    raise RuntimeError("joint AURA source execution backend changed during measurement")
                for name in pending:
                    joint_rows[name] = {}
                    for fmt in unit_formats[name]:
                        if fmt in raw_unmeasured.get(name, ()):
                            continue
                        key = (name, fmt)
                        components = joint_components[key]
                        if fmt in _ZERO_COST_FORMATS:
                            components = [{"weight": 0.0, "activation": 0.0, "mixed": 0.0, "total": 0.0} for _ in range(n_probes)]
                        joint_rows[name][fmt] = make_joint_aura_entry(
                            operator_identity=joint_operators[key],
                            probe_identity=joint_probe_identity,
                            signed_components=components,
                        )

            # This layer's input boundary will not be read again in the reverse
            # sweep (the next iteration consumes boundary ``layer - 1``).
            # Release it progressively instead of retaining all 44 DSv4
            # hc_mult=4 boundary snapshots to the end.
            for batch in batches:
                batch.activations_cpu[layer] = torch.empty(0)

            if checkpoint_root is not None:
                assert checkpoint_identity_sha256 is not None
                for name in pending:
                    _write_aura_unit_checkpoint(
                        checkpoint_root,
                        qname=name,
                        identity_sha256=checkpoint_identity_sha256,
                        state={**_aura_unit_state(
                            name,
                            render_formats[name],
                            s2=s2,
                            s4=s4,
                            x2_probe=x2_probe,
                            dw_src=dw_src,
                            g_trace=g_trace,
                            col_energy=col_energy,
                            weight_mse_diagnostic=weight_mse_diagnostic,
                            source_weight_identity=source_weight_identity,
                        ), **({"joint_aura_rows": joint_rows[name]} if joint_activation else {})},
                    )
            # Closed-loop observability: a reverse layer is minutes of silent
            # render+adjoint work at streamed scale, so each one reports its
            # own rate and the sweep ETA the moment it lands.
            reverse_layers_done += 1
            if pending:
                elapsed = time.time() - reverse_started
                rate = reverse_layers_done / max(elapsed, 1e-9)
                remaining = runner.num_layers - reverse_layers_done
                _log(
                    f"reverse layer {layer} done "
                    f"({reverse_layers_done}/{runner.num_layers}, "
                    f"{len(pending)} unit(s), {elapsed / 60:.1f} min elapsed, "
                    f"ETA {remaining / rate / 60:.1f} min)"
                )
        finally:
            for parameter in parameters.values():
                parameter.grad = None
                parameter.requires_grad_(False)
            # Loop locals and the lease otherwise retain the last layer's
            # delta plane through the next source install/render window.
            d_weights.clear()
            result = delta = weight = None
            del d_weights
            parameters.clear()
            parameter = None
            runner.context.unload(layer)
            _refresh_packed_layer_views(layer)

    return _finish_streamed_payload()


def run_streamed_production_anchor_aura(
    runner,
    calib_ids: torch.Tensor,
    *,
    formats_by_qname: Mapping[str, Sequence[str]],
    render_purposes_by_qname: Mapping[
        str, Mapping[str, str | Sequence[str]]
    ] | None = None,
    unmeasured_formats_by_qname: Mapping[str, Sequence[str]] | None = None,
    activation_index,
    render_levers: Mapping[str, object],
    col_weights: Mapping[str, torch.Tensor],
    cb_serialization_context,
    calibration_hash: str,
    arm_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
    checkpoint_dir: str | Path,
    resume: bool,
    n_probes: int,
    probe_microbatch: int = 0,
    token_scope: str = "all",
    temperature: float = 1.0,
    seed_base: int = 7000,
    cold_expert_provenance: Mapping[str, object] | None = None,
    max_act_rows: int = 512,
    h_detail_dir: str | Path | None = None,
    checkpoint_identity_extra: Mapping[str, object] | None = None,
    include_routed_experts: bool = True,
    allow_packed_expert_omission: bool = False,
    collect_col_energy: bool = False,
    joint_activation: bool = False,
    joint_projection_backend=None,
    profile=None,
) -> dict:
    """Run one streamed KL adjoint over an exact production-anchor plan.

    This is the concrete campaign API. ``formats_by_qname`` may contain
    several measured CB anchors for a unit (anchor, fitting-panel, and held-out
    validation cells) plus an exact identity-bearing terminal.  A terminal
    declared in ``unmeasured_formats_by_qname`` is never sent to the renderer
    and receives no synthetic scalar unless it is independently covered by
    the exact zero-cost passthrough contract.  Every other listed cell is
    rendered in the fixed production arm exactly once unless its
    identity-bound per-unit AURA shard is resumed.

    The returned ordinary AURA payload carries scalar ``predicted_dloss`` rows
    and exact renderer/plan provenance.  Rendered weights and ``dW`` live only
    for the current reverse layer and are discarded before it is unloaded.

    On a routed-expert model whose packed experts are priced by the empirical
    unit-KL path, pass ``include_routed_experts=False`` together with
    ``allow_packed_expert_omission=True``; the payload then carries no
    routed-expert rows and the caller must merge the empirical rows before
    allocation (the packed-expert coverage guard raises otherwise).
    """
    if checkpoint_dir is None:
        raise ValueError(
            "production-anchor AURA requires a durable checkpoint_dir"
        )
    profile = resolve_routed_expert_profile(
        runner.model, profile or runner.profile
    )
    canonical_plan: dict[str, tuple[str, ...]] = {}
    render_plan: dict[str, tuple[str, ...]] = {}
    canonical_unmeasured: dict[str, tuple[str, ...]] = {}
    format_union: list[str] = []
    for raw_name, raw_formats in formats_by_qname.items():
        name = str(raw_name)
        if isinstance(raw_formats, (str, bytes)):
            raise TypeError(
                f"production-anchor formats for {name} must be a sequence"
            )
        formats = tuple(dict.fromkeys(
            fr.canonical_format_name(fmt) for fmt in raw_formats
        ))
        if not formats:
            raise ValueError(
                f"production-anchor formats are empty for {name}"
            )
        raw_unmeasured = (
            unmeasured_formats_by_qname.get(name, ())
            if isinstance(unmeasured_formats_by_qname, Mapping) else ()
        )
        if isinstance(raw_unmeasured, (str, bytes)):
            raise TypeError(
                f"unmeasured production-anchor formats for {name} must be a sequence"
            )
        retained = tuple(dict.fromkeys(
            fr.canonical_format_name(fmt) for fmt in raw_unmeasured
        ))
        unexpected_retained = sorted(set(retained) - set(formats))
        if unexpected_retained:
            raise ValueError(
                f"unmeasured production-anchor formats for {name} are "
                f"outside its exact plan: {unexpected_retained}"
            )
        measured = tuple(
            fmt for fmt in formats
            if fmt not in _ZERO_COST_FORMATS and fmt not in set(retained)
        )
        if not measured:
            raise ValueError(
                f"production-anchor unit {name} has no real rendered anchor"
            )
        canonical_plan[name] = formats
        render_plan[name] = measured
        canonical_unmeasured[name] = retained
        for fmt in formats:
            if fmt not in format_union:
                format_union.append(fmt)
    if isinstance(unmeasured_formats_by_qname, Mapping):
        unexpected_names = sorted(
            set(unmeasured_formats_by_qname) - set(canonical_plan)
        )
        if unexpected_names:
            raise ValueError(
                "unmeasured production-anchor formats contain unplanned "
                f"qnames; sample={unexpected_names[:8]}"
            )

    allowed_purposes = frozenset({"anchor", "panel", "validation"})
    canonical_purposes: dict[str, dict[str, list[str]]] = {}
    raw_purposes = render_purposes_by_qname
    for name, formats in render_plan.items():
        rows: dict[str, list[str]] = {}
        supplied = (
            raw_purposes.get(name, {})
            if isinstance(raw_purposes, Mapping) else {}
        )
        if not isinstance(supplied, Mapping):
            raise TypeError(
                f"production-anchor purposes for {name} must be a mapping"
            )
        for fmt in formats:
            raw = supplied.get(fmt, "anchor")
            values = [raw] if isinstance(raw, str) else list(raw)
            purposes = list(dict.fromkeys(str(value) for value in values))
            invalid = sorted(set(purposes) - allowed_purposes)
            if not purposes or invalid:
                raise ValueError(
                    f"production-anchor purposes for {name}@{fmt} are "
                    f"invalid: {purposes}; allowed={sorted(allowed_purposes)}"
                )
            rows[fmt] = purposes
        unexpected = sorted(set(str(fmt) for fmt in supplied) - set(formats))
        if unexpected:
            raise ValueError(
                f"production-anchor purposes contain unrendered cells for "
                f"{name}: {unexpected}"
            )
        canonical_purposes[name] = rows
    if isinstance(raw_purposes, Mapping):
        unexpected_names = sorted(set(raw_purposes) - set(render_plan))
        if unexpected_names:
            raise ValueError(
                "production-anchor purposes contain unplanned qnames; "
                f"sample={unexpected_names[:8]}"
            )

    from prismaquant.streaming_production_cache import (
        StreamedProductionAnchorRenderer,
    )

    renderer = StreamedProductionAnchorRenderer(
        runner.model,
        act_index=activation_index,
        formats_by_qname=render_plan,
        levers=render_levers,
        profile=profile,
        device=runner.device,
        col_weights=col_weights,
        cb_serialization_context=cb_serialization_context,
        calibration_hash=calibration_hash,
        arm_identity=arm_identity,
        model_identity=model_identity,
        cold_expert_provenance=cold_expert_provenance,
        max_act_rows=max_act_rows,
        h_detail_dir=h_detail_dir,
        transient_consumer_identity=(
            {**AURA_PRODUCTION_ANCHOR_DELTA_CONSUMER_IDENTITY,
             "storage_dtype": "torch.float32"}
            if joint_activation else AURA_PRODUCTION_ANCHOR_DELTA_CONSUMER_IDENTITY
        ),
    )
    extra = dict(checkpoint_identity_extra or {})
    reserved_extra = {
        "production_anchor_render_purposes",
        "production_anchor_unmeasured_formats_by_qname",
    } & set(extra)
    if reserved_extra:
        raise ValueError(
            "checkpoint_identity_extra cannot override production-anchor "
            f"plan identity fields: {sorted(reserved_extra)}"
        )
    extra["production_anchor_render_purposes"] = canonical_purposes
    extra["production_anchor_unmeasured_formats_by_qname"] = {
        name: list(values)
        for name, values in sorted(canonical_unmeasured.items())
    }
    payload = compute_aura_cost_streamed(
        runner,
        calib_ids,
        format_union,
        n_probes=n_probes,
        probe_microbatch=probe_microbatch,
        token_scope=token_scope,
        temperature=temperature,
        production_cache=None,
        seed_base=seed_base,
        require_production_cache=True,
        # DSv4's worst routed layer carries two full anchor planes. FP32
        # subtraction + BF16 storage retains the validated AURA contract
        # (the projection below upcasts) while keeping Spark above the host's
        # 3-GiB guardian floor.
        dw_dtype="bfloat16",
        include_lm_head=False,
        allow_packed_expert_omission=allow_packed_expert_omission,
        collect_col_energy=collect_col_energy,
        joint_activation=joint_activation,
        joint_projection_backend=joint_projection_backend,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        model_identity=model_identity,
        checkpoint_identity_extra=extra,
        formats_by_qname=canonical_plan,
        unmeasured_formats_by_qname=canonical_unmeasured,
        anchor_renderer=renderer,
        include_routed_experts=include_routed_experts,
        diagnostic_weight_mse_pairs=[
            (name, fmt)
            for name, rows in canonical_purposes.items()
            for fmt, purposes in rows.items()
            if "panel" in purposes
        ],
        profile=profile,
    )
    counts = {
        purpose: sum(
            purpose in purposes
            for rows in canonical_purposes.values()
            for purposes in rows.values()
        )
        for purpose in sorted(allowed_purposes)
    }
    payload["provenance"].update({
        "production_anchor_render_purposes": canonical_purposes,
        "production_anchor_unmeasured_formats_by_qname": {
            name: list(values)
            for name, values in sorted(canonical_unmeasured.items())
        },
        "production_anchor_purpose_counts": counts,
        "production_anchor_union_render_count": sum(
            len(rows) for rows in canonical_purposes.values()
        ),
    })
    return payload


def _stage_aura_model(model_path: str) -> str:
    """Return AURA's canonical text-only execution view of a checkpoint.

    Wrapper checkpoints must execute the nested ``text_config`` values, while
    already-flattened sources stay on their original path.  The shared stager
    also owns process-exit cleanup of any temporary staged tree, keeping it
    alive for the complete resident or streaming AURA run.
    """
    from prismaquant.sensitivity_probe import stage_text_only

    return stage_text_only(model_path)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aura KL-adjoint allocator cost")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--formats", default="NVFP4,FP8_DYNAMIC,BF16")
    p.add_argument(
        "--format-plan",
        default=None,
        help="Identity-bound source-class format plan. In streaming mode the "
        "requested family is intersected per qname, so source-rate-illegal "
        "cells are neither rendered nor priced.",
    )
    p.add_argument("--production-cache", default=None,
                   help="ProductionWeightCache pickle for production-faithful dW")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Durable per-Linear AURA checkpoint directory. The "
                        "manifest value-binds calibration, cache rendering, "
                        "formats, execution knobs, and producer commit.")
    p.add_argument("--resume", action="store_true",
                   help="Validate and reuse exact per-Linear checkpoints. An "
                        "identity mismatch refuses both reuse and recompute.")
    p.add_argument("--unit-filter", default="",
                   help="Optional regex selecting exact Linear qnames; the "
                        "resolved ordered unit scope is identity-bound.")
    p.add_argument(
        "--streaming", action="store_true",
        help="Stream decoder layers through the existing prefetch/cache "
             "context and run a boundary-activation KL adjoint. Required for "
             "models whose expanded source cannot be resident.")
    p.add_argument(
        "--joint-activation", action="store_true",
        help="Research-only joint activation/weight AURA; requires --streaming "
             "and production renders. Keeps aligned signed probes and uses FP32 dW.")
    p.add_argument(
        "--streaming-offload-dir", default=None,
        help="Streaming model work directory. Defaults beneath the AURA "
             "checkpoint directory (or the output directory); never /tmp.")
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
    p.add_argument("--calibration-input", default=None,
                   help="Exact safetensors calibration_ids draw; bypasses dataset sampling. "
                        "Requires --calibration-input-sha256 and matching sample/sequence counts.")
    p.add_argument("--calibration-input-sha256", default=None,
                   help="Independent SHA256 of --calibration-input.")
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
                        "Streamed joint AURA uses versioned row-indexed "
                        "draws independent of the partition; compares against "
                        "an explicit full-size >0 reference. Resident draws "
                        "retain the legacy microbatch policy. Checkpoints "
                        "bind the execution batch size.")
    p.add_argument("--allow-packed-expert-omission", action="store_true",
                   help="Explicit research/debug escape: allow AURA to omit "
                        "profile-declared routed expert targets from the cost "
                        "payload. The historical flag name is retained for "
                        "artifact compatibility. Default is fail-fast because "
                        "routed experts need an empirical/hybrid expert-cost "
                        "path, not silent omission.")
    p.add_argument(
        "--include-routed-experts",
        action="store_true",
        help="Include profile-declared expert targets in streamed AURA. "
        "Packed source Parameters require --joint-activation and decoded "
        "production-cache renders; the local derivative remains route-flip-blind.",
    )
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
    if bool(args.calibration_input) != bool(args.calibration_input_sha256):
        p.error("--calibration-input and --calibration-input-sha256 are required together")
    if args.calibration_input and args.dataset:
        p.error("--calibration-input and --dataset are mutually exclusive")
    if args.joint_activation and not args.streaming:
        p.error("--joint-activation requires --streaming")
    if args.resume and not args.checkpoint_dir:
        p.error("--resume requires --checkpoint-dir")
    if args.streaming and args.gradient_checkpointing:
        p.error("--gradient-checkpointing is resident-only; --streaming uses "
                "an explicit one-layer reverse recomputation")
    if args.format_plan and not args.streaming:
        p.error("--format-plan requires --streaming")
    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("aura_cost", args.device)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prismaquant.calibration_data import load_wikitext_calibration_windowed
    from prismaquant.model_profiles import detect_profile

    staged = _stage_aura_model(args.model)
    # Profile detection installs architecture-owned vendored modelling where
    # required and is also the fail-closed authority for routed-expert
    # membership.  Resolve it before model construction and reuse that exact
    # instance for both the guard and smooth-target enumeration.
    profile = detect_profile(staged)
    if args.dtype == "auto":
        args.dtype = _resolve_auto_dtype(staged, args.min_free_gib)
    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16
    local_only = Path(staged).exists()
    _log(f"loading {args.model} (staged={staged}) dtype={args.dtype}")
    tok = AutoTokenizer.from_pretrained(
        staged, trust_remote_code=True, local_files_only=local_only)
    model = None
    streamed_runner = None
    if args.streaming:
        from prismaquant.cost_streaming import build_streamed_causal_lm

        offload_dir = args.streaming_offload_dir
        if not offload_dir:
            anchor = Path(args.checkpoint_dir or args.output).parent
            offload_dir = str(anchor / "aura-streaming-offload")
        streamed_runner = build_streamed_causal_lm(
            staged,
            device=torch.device(args.device),
            dtype=dt,
            offload_folder=offload_dir,
            profile=profile,
            attn_implementation="eager",
        )
    else:
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
        assert model is not None
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
    calibration_input_receipt = None
    if args.calibration_input:
        from prismaquant.calibration_data import load_calibration_input
        calib, calibration_input_receipt = load_calibration_input(
            args.calibration_input, expected_sha256=args.calibration_input_sha256,
            n_samples=args.n_calib_samples, seqlen=args.calib_seqlen,
        )
        calib = calib.to(args.device)
    elif args.dataset:
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

    requested_formats = [
        f.strip() for f in args.formats.split(",") if f.strip()
    ]
    source_format_plan = None
    planned_formats_by_qname = None
    if args.format_plan:
        from prismaquant.source_class_format_plan import load_format_plan

        source_format_plan = load_format_plan(args.format_plan)
        allowed_by_qname = source_format_plan.formats_by_qname()
        planned_universe = {
            fmt for values in allowed_by_qname.values() for fmt in values
        }
        canonical_requested = [
            fr.canonical_format_name(fmt) for fmt in requested_formats
        ]
        planned_formats_by_qname = {
            qname: tuple(
                fmt for fmt in canonical_requested
                if fmt not in planned_universe or fmt in allowed
            )
            for qname, allowed in allowed_by_qname.items()
        }
    if streamed_runner is not None:
        try:
            streamed_model_identity = None
            if args.checkpoint_dir or args.joint_activation:
                from prismaquant.cost_streaming import (
                    build_streamed_model_identity,
                )

                streamed_model_identity = build_streamed_model_identity(
                    streamed_runner,
                    args.model,
                    identity_cache_path=(
                        Path(args.checkpoint_dir or Path(args.output).parent)
                        / "streamed_model_identity.json"
                    ),
                )
            payload = compute_aura_cost_streamed(
                streamed_runner,
                calib,
                requested_formats,
                n_probes=args.n_probes,
                probe_microbatch=args.probe_microbatch,
                token_scope=args.token_scope,
                temperature=args.temperature,
                production_cache=cache,
                min_free_gib=args.min_free_gib,
                seed_base=args.seed_base,
                assert_bf16_passthrough=args.assert_bf16_passthrough,
                require_production_cache=args.require_production_cache,
                dw_dtype=args.dw_dtype,
                include_lm_head=args.include_lm_head,
                allow_packed_expert_omission=args.allow_packed_expert_omission,
                collect_col_energy=args.collect_col_energy,
                joint_activation=args.joint_activation,
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume,
                unit_filter=(args.unit_filter or None),
                model_identity=streamed_model_identity,
                checkpoint_identity_extra={
                    "streaming": True,
                    **(
                        {"source_format_plan_identity_sha256": (
                            source_format_plan.identity_sha256
                        )}
                        if source_format_plan is not None else {}
                    ),
                },
                formats_by_qname=planned_formats_by_qname,
                include_routed_experts=args.include_routed_experts,
                profile=profile,
            )
        finally:
            streamed_runner.shutdown()
    else:
        assert model is not None
        payload = compute_aura_cost(
            model, calib, requested_formats,
            n_probes=args.n_probes, token_scope=args.token_scope,
            temperature=args.temperature, production_cache=cache,
            min_free_gib=args.min_free_gib,
            n_linear_chunks=args.n_linear_chunks,
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
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            unit_filter=(args.unit_filter or None),
            checkpoint_identity_extra={
                "gradient_checkpointing": bool(args.gradient_checkpointing),
            },
            profile=profile,
        )
    payload["provenance"].update({
        "model": str(args.model),
        "dtype": str(args.dtype),
        "calib_source": (
            str(args.calibration_input) if args.calibration_input else str(args.dataset) if args.dataset
            else f"wikitext:{args.calib_split}"),
        "n_calib_samples": int(args.n_calib_samples),
        "calib_seqlen": int(args.calib_seqlen),
        "calib_seed": (calibration_input_receipt["provenance"].get("seed")
                       if calibration_input_receipt is not None else int(args.calib_seed)),
        "production_cache": str(args.production_cache or ""),
        # re-vet R2 precondition (i): which pipeline COST_MODE produced this.
        "cost_mode": str(args.cost_mode or ""),
    })
    if calibration_input_receipt is not None:
        payload["provenance"]["calibration_input"] = calibration_input_receipt
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
