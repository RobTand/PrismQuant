"""Packed-expert imatrix synthesis — CHECKPOINT-based, no model load.

The activation cache holds only each experts MODULE's input, so the harvested
imatrix (``export_gguf.build_imatrix_from_act_cache``) can never contain:

* ``<qn>.gate_up_proj`` — the harvest emits the MODULE name, while the CB
  cost/export look up the packed-param name;
* ``<qn>.down_proj``   — its input is the PER-EXPERT intermediate, which is
  never cached anywhere.

Both the exporter (hard-fails, "no silent RTN") and the local packed-expert
cost (which would otherwise render down_proj unweighted while the export
ships weighted bytes — the rendering-confound class) need these entries, from
ONE shared source. This module synthesizes them by replaying the routed
forward directly from the CHECKPOINT tensors (router weight + per-expert
gate/up) on the cached module inputs: route -> per-expert gate/up ->
activation -> intermediate, mean-square pooled per expert. The
model-loaded twin of this replay lives in
``expert_empirical_cost.ensure_unit_col_weights``; keep semantics in
lockstep.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from safetensors import safe_open


@dataclass(frozen=True)
class RoutedActivationSamples:
    """Value-bearing routed down-projection inputs plus sampling identity.

    ``values`` is ordered by the deterministic token-major route sample.  The
    companion vectors make that ordering auditable: calibration cannot silently
    turn into one equally sized bucket per expert (which would destroy the
    observed routing distribution), or reuse the gate/up rows for down-proj.
    """

    values: torch.Tensor
    cache_row_indices: torch.Tensor
    source_row_indices: torch.Tensor
    expert_indices: torch.Tensor
    route_slots: torch.Tensor
    route_weights: torch.Tensor

    def validate(self) -> None:
        if not isinstance(self.values, torch.Tensor) or self.values.ndim != 2:
            raise ValueError("routed activation values must be a rank-2 tensor")
        rows = int(self.values.shape[0])
        metadata = {
            "cache_row_indices": self.cache_row_indices,
            "source_row_indices": self.source_row_indices,
            "expert_indices": self.expert_indices,
            "route_slots": self.route_slots,
            "route_weights": self.route_weights,
        }
        for name, tensor in metadata.items():
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
                raise ValueError(
                    f"routed activation {name} must be a rank-1 tensor"
                )
            if int(tensor.numel()) != rows:
                raise ValueError(
                    f"routed activation {name} has {tensor.numel()} rows, "
                    f"expected {rows}"
                )
        if rows == 0:
            raise ValueError("routed activation sample is empty")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("routed activation values contain non-finite data")
        if not bool(torch.isfinite(self.route_weights).all()):
            raise ValueError("routed activation weights contain non-finite data")
        if bool((self.route_weights < 0).any()):
            raise ValueError("routed activation weights must be non-negative")


def _weight_map(model_path: Path) -> dict[str, str]:
    idx = model_path / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    st = model_path / "model.safetensors"
    if st.exists():
        with safe_open(str(st), framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise FileNotFoundError(f"no safetensors index under {model_path}")


_WEIGHT_BLOCK_CACHE: dict[str, tuple[int, int]] = {}


def _weight_block_size(model_path: Path) -> tuple[int, int]:
    """The checkpoint's declared FP8 block tiling.

    One checkpoint, one contract: this defers to the streaming loader's
    `_declared_weight_block_size`, which REFUSES a checkpoint that pairs
    fp8 weights with scale tensors but declares no `weight_block_size`
    rather than inferring the grid from the scale-plane shape. Inference
    is unsafe even when it divides exactly -- a 200-row weight against a
    2-row scale plane divides exactly at 100 and is equally a 128-block
    tiling with a partial trailing block, and the two dequants differ on
    every row from 128 up. Two mechanisms that disagree about one
    checkpoint is the defect this avoids; the packed-expert replay and
    the streaming load must read the same grid.

    Cached per path because `_load_tensors` asks once per FP8 tensor.
    """
    from .layer_streaming import _declared_weight_block_size

    key = str(model_path)
    if key not in _WEIGHT_BLOCK_CACHE:
        _WEIGHT_BLOCK_CACHE[key] = _declared_weight_block_size(key)
    return _WEIGHT_BLOCK_CACHE[key]


def _load_tensors(
    model_path: Path,
    weight_map: dict[str, str],
    keys: list[str],
    dtype=None,
) -> dict[str, torch.Tensor]:
    by_shard: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        by_shard[weight_map[k]].append(k)
    out: dict[str, torch.Tensor] = {}
    for shard, ks in by_shard.items():
        with safe_open(str(model_path / shard), framework="pt") as f:
            for k in ks:
                tensor = f.get_tensor(k)
                if str(tensor.dtype).startswith("torch.float8"):
                    # Serialized scale contract: block-wise FP8 checkpoints
                    # (fmt e4m3) carry a `<name>_scale_inv` companion whose
                    # shape tiles the weight; dequantize exactly with it.
                    # The refusal stands for tensors without the companion.
                    scale_key = k + "_scale_inv"
                    if scale_key not in weight_map:
                        raise ValueError(
                            f"{k}: packed-expert replay cannot decode an FP8 "
                            "checkpoint tensor without its serialized scale "
                            "contract; refusing approximate routing/calibration"
                        )
                    sshard = weight_map[scale_key]
                    if sshard == shard:
                        scale = f.get_tensor(scale_key)
                    else:
                        with safe_open(
                            str(model_path / sshard), framework="pt"
                        ) as sf:
                            scale = sf.get_tensor(scale_key)
                    o, i = tensor.shape
                    so, si = scale.shape
                    # Raises when the checkpoint declares no block size --
                    # the grid is read from the checkpoint, never guessed.
                    b0, b1 = _weight_block_size(model_path)
                    if (-(-o // b0), -(-i // b1)) != (so, si):
                        raise ValueError(
                            f"{k}: scale plane {so}x{si} does not tile "
                            f"weight {o}x{i} at the checkpoint's declared "
                            f"weight_block_size {(b0, b1)}"
                        )
                    tensor = (
                        tensor.to(torch.float32)
                        * scale.to(torch.float32)
                        .repeat_interleave(b0, 0)[:o]
                        .repeat_interleave(b1, 1)[:i]
                    )
                out[k] = tensor if dtype is None else tensor.to(dtype)
    return out


# The probe's activation-cache filename transform, mirrored EXACTLY (see
# `measure_quant_cost.ActivationIndex`): one definition would be better, but
# that one lives behind a class that also wants probe-stat metadata.
_ACT_FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")

# Packed parameter roles whose stage is `w13` (the fused gate/up half). Mirrors
# `nvfp4_activation_contract._PACKED_LEAF_ROLES` without importing it at module
# scope; anything else pools to the `w2` (down) stage.
_PACKED_GATE_UP_LEAVES = frozenset(
    {"gate_up_proj", "gate_proj", "up_proj", "w1", "w3"}
)


def _load_act_entry(
    p: Path,
) -> tuple[str, torch.Tensor | None, torch.Tensor | None]:
    """Load the act-cache schema used by ``build_imatrix_from_act_cache``.

    Returns ``(module name, input rows, source-row ids)``.
    """
    blob = torch.load(p, map_location="cpu", weights_only=False)
    inputs = blob.get("inputs") if isinstance(blob, dict) else None
    row_indices = blob.get("row_indices") if isinstance(blob, dict) else None
    name = (blob.get("name") if isinstance(blob, dict) else None) or (
        p.stem.replace("__", "."))
    if inputs is None or inputs.ndim != 2:
        return name, None, None
    if not isinstance(row_indices, torch.Tensor) \
            or row_indices.ndim != 1 \
            or int(row_indices.numel()) != int(inputs.shape[0]):
        row_indices = None
    return name, inputs.float(), row_indices


@torch.no_grad()
def synthesize_packed_expert_col_weights(
    model_path: str | Path,
    act_dir: str | Path,
    col_weights: dict,
    profile=None,
    *,
    max_rows: int = 4096,
    device: str | None = None,
    activation_samples: dict[
        str, torch.Tensor | RoutedActivationSamples
    ] | None = None,
    target_names: set[str] | None = None,
    write_col_weights: bool = True,
) -> list[str]:
    """Fill missing ``<experts_qn>.gate_up_proj`` / ``.down_proj`` imatrix
    entries in ``col_weights`` IN PLACE from the checkpoint + act cache.

    Returns the names added. Loud failure over silent omission: an experts
    module whose router/per-expert tensors can't be resolved raises (the
    exporter would hard-fail later anyway — better here, with the cause).
    """
    model_path = Path(model_path)
    act_dir = Path(act_dir)
    if profile is None:
        from prismaquant.model_profiles import detect_profile_with_warning
        profile = detect_profile_with_warning(
            str(model_path), entrypoint="moe-imatrix")
    wm = _weight_map(model_path)
    cfg = json.loads((model_path / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    top_k = int(tc.get(
        "num_experts_per_tok",
        tc.get("num_experts_per_token", 8),
    ))
    norm_topk = bool(tc.get("norm_topk_prob", True))
    model_type = str(tc.get("model_type", cfg.get("model_type", ""))).lower()
    default_score_function = (
        "sigmoid"
        if model_type in {
            "laguna",
            "lfm2_moe",
            "deepseek_v3",
            "deepseek_v4",
            "hy_v3",
        } or bool(tc.get("use_expert_bias", False))
        else "softmax"
    )
    score_function = str(
        tc.get(
            "scoring_func",
            tc.get(
                "router_score_function",
                default_score_function,
            ),
        )
    ).lower()
    topk_method = str(tc.get("topk_method", "greedy")).lower()
    router_softcap = float(tc.get("moe_router_logit_softcapping", 0.0) or 0.0)
    route_weight_scale = float(tc.get(
        "routed_scaling_factor",
        tc.get("moe_routed_scaling_factor", 1.0),
    ) or 1.0)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    added: list[str] = []
    for p in sorted(act_dir.glob("*.pt")):
        qn, X, source_row_indices = _load_act_entry(p)
        src = profile.source_tensor_name(qn)
        split_gate0 = f"{src}.0.gate_proj.weight"
        packed_gate_up_key = next(
            (
                cand for cand in (
                    f"{src}.gate_up_proj",
                    f"{src}.gate_up_proj.weight",
                )
                if cand in wm
            ),
            None,
        )
        if split_gate0 not in wm and packed_gate_up_key is None:
            continue                    # not a supported experts module
        gu_name, dn_name = f"{qn}.gate_up_proj", f"{qn}.down_proj"
        requested = None if target_names is None else set(target_names)
        need_gu_sample = (
            activation_samples is not None
            and (requested is None or gu_name in requested)
            and gu_name not in activation_samples
        )
        need_dn_sample = (
            activation_samples is not None
            and (requested is None or dn_name in requested)
            and dn_name not in activation_samples
        )
        need_gu_weight = (
            write_col_weights
            and (requested is None or gu_name in requested)
            and gu_name not in col_weights
        )
        need_dn_weight = (
            write_col_weights
            and (requested is None or dn_name in requested)
            and dn_name not in col_weights
        )
        if not any((need_gu_sample, need_dn_sample,
                    need_gu_weight, need_dn_weight)):
            continue
        if X is None:
            raise ValueError(f"{qn}: activation cache entry unreadable — "
                             f"cannot synthesize the packed-expert imatrix")
        X = X[:max_rows].to(dev)

        if need_gu_sample:
            activation_samples[gu_name] = X.detach().to("cpu").contiguous()
        if need_gu_weight:
            col_weights[gu_name] = (
                X.pow(2).mean(dim=0).reshape(1, 1, -1).cpu())
            added.append(gu_name)
        if need_dn_weight or need_dn_sample:
            if need_dn_sample and bool(
                tc.get("moe_apply_router_weight_on_input", False)
            ):
                raise ValueError(
                    f"{qn}: exact routed down-projection activation replay "
                    "does not implement moe_apply_router_weight_on_input; "
                    "production activation calibration fails closed"
                )
            # Router weight naming varies per family (Qwen3.5-MoE:
            # <parent>.gate.weight; hy_v3: <parent>.router.gate.weight).
            src_parent = src.rsplit(".", 1)[0]
            gate_key = None
            for cand in (f"{src_parent}.gate.weight",
                         f"{src_parent}.router.gate.weight",
                         f"{src_parent}.router.weight"):
                if cand in wm:
                    gate_key = cand
                    break
            if gate_key is None:
                raise ValueError(
                    f"{qn}: router weight not in checkpoint (tried "
                    f"{src_parent} .gate/.router.gate/.router .weight) — "
                    f"cannot replay routing for the down_proj imatrix")
            keys = [gate_key]
            if packed_gate_up_key is not None:
                keys.append(packed_gate_up_key)
            else:
                E = 0
                while f"{src}.{E}.gate_proj.weight" in wm:
                    E += 1
                if E == 0:
                    raise ValueError(f"{qn}: no per-expert gate_proj tensors")
                for e in range(E):
                    keys += [f"{src}.{e}.gate_proj.weight",
                             f"{src}.{e}.up_proj.weight"]
            t = _load_tensors(model_path, wm, keys)
            packed_gate_up = None
            if packed_gate_up_key is not None:
                packed_gate_up = t[packed_gate_up_key]
                if packed_gate_up.ndim != 3:
                    raise ValueError(
                        f"{qn}: packed gate_up tensor must be rank 3, got "
                        f"{tuple(packed_gate_up.shape)}"
                    )
                E = int(packed_gate_up.shape[0])
                if int(packed_gate_up.shape[1]) % 2:
                    raise ValueError(
                        f"{qn}: packed gate_up output dimension must be even, "
                        f"got {int(packed_gate_up.shape[1])}"
                    )
            Wg = t[gate_key].to(dev)
            if int(Wg.shape[0]) != E:
                raise ValueError(
                    f"{qn}: router has {int(Wg.shape[0])} experts but "
                    f"expert weights have {E}"
                )
            native_logits = X.to(Wg.dtype) @ Wg.t()
            logits = native_logits.float()
            if router_softcap > 0.0:
                logits = torch.tanh(logits / router_softcap) * router_softcap
            if need_dn_sample and topk_method not in {"", "greedy"}:
                raise ValueError(
                    f"{qn}: exact routed down-projection activation replay "
                    f"does not implement topk_method={topk_method!r}; "
                    "production activation calibration fails closed"
                )
            if score_function == "sigmoid":
                # LFM2-MoE applies sigmoid in the router tensor dtype; Laguna
                # explicitly promotes logits to F32 first.  Preserve that
                # distinction because near-tie TOP-K membership determines
                # which value-bearing down rows exist at all.
                scores = torch.sigmoid(
                    native_logits
                    if model_type == "lfm2_moe" and router_softcap == 0.0
                    else logits
                )
            elif score_function == "softmax":
                scores = torch.softmax(logits, dim=-1)
            elif need_dn_sample:
                raise ValueError(
                    f"{qn}: exact routed down-projection activation replay "
                    f"does not implement scoring_func={score_function!r}; "
                    "production activation calibration fails closed"
                )
            else:
                # Retain the legacy imatrix approximation only when no fused
                # activation contract is being produced.
                scores = torch.softmax(logits, dim=-1)
            # Selection bias (Laguna/DeepSeek/Hy-style no-aux routing) changes
            # TOP-K membership but never the unbiased returned route weight.
            bias = None
            for cand in (f"{src_parent}.gate.e_score_correction_bias",
                         f"{src}.e_score_correction_bias",
                         f"{src_parent}.expert_bias",
                         f"{src_parent}.router.e_score_correction_bias"):
                if cand in wm:
                    bias = _load_tensors(
                        model_path,
                        wm,
                        [cand],
                        dtype=torch.float32,
                    )[cand].to(dev)
                    break
            sel = scores if bias is None else scores + bias
            _, topi = torch.topk(sel, top_k, dim=-1)
            topv = torch.gather(scores, -1, topi)
            if norm_topk:
                denominator = topv.sum(dim=-1, keepdim=True)
                if model_type == "lfm2_moe":
                    denominator = denominator + 1e-6
                else:
                    denominator = denominator.clamp_min(1e-12)
                topv = topv / denominator
            topv = topv * route_weight_scale
            inter = (
                int(packed_gate_up.shape[1]) // 2
                if packed_gate_up is not None
                else int(t[f"{src}.0.gate_proj.weight"].shape[0])
            )
            out = torch.zeros(E, inter, dtype=torch.float32, device=dev)
            hit = torch.zeros(E, dtype=torch.bool)
            if need_dn_sample:
                total_routes = int(topi.numel())
                sample_rows = min(int(max_rows), total_routes)
                if sample_rows <= 0:
                    raise ValueError(
                        f"{qn}: routed replay has no down_proj routes"
                    )
                if sample_rows == total_routes:
                    sampled_flat = torch.arange(total_routes, device=dev)
                else:
                    # Deterministic uniform route reservoir.  A regular stride
                    # aliases badly with top-k slots (for example every second
                    # entry selects only slot zero at top-k=2), while an
                    # expert-by-expert cap destroys observed expert frequency.
                    seed = int.from_bytes(
                        hashlib.sha256(qn.encode("utf-8")).digest()[:8],
                        "little",
                    ) & ((1 << 63) - 1)
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(seed)
                    sampled_flat = torch.randperm(
                        total_routes,
                        generator=generator,
                        device="cpu",
                    )[:sample_rows].sort().values.to(dev)
                sampled_cache_rows = torch.div(
                    sampled_flat,
                    top_k,
                    rounding_mode="floor",
                )
                sampled_slots = sampled_flat.remainder(top_k)
                sampled_experts = topi[
                    sampled_cache_rows,
                    sampled_slots,
                ]
                sampled_weights = topv[
                    sampled_cache_rows,
                    sampled_slots,
                ]
                sampled_values = torch.empty(
                    sample_rows,
                    inter,
                    dtype=torch.float32,
                    device=dev,
                )
            for e in range(E):
                tok = (topi == e).any(dim=-1).nonzero(as_tuple=True)[0]
                if tok.numel() == 0:
                    continue
                if packed_gate_up is not None:
                    gate_weight = packed_gate_up[e, :inter].to(dev)
                    up_weight = packed_gate_up[e, inter:].to(dev)
                else:
                    gate_weight = t[f"{src}.{e}.gate_proj.weight"].to(dev)
                    up_weight = t[f"{src}.{e}.up_proj.weight"].to(dev)
                g = X[tok].to(gate_weight.dtype) @ gate_weight.t()
                u = X[tok].to(up_weight.dtype) @ up_weight.t()
                intermediate = F.silu(g) * u
                intermediate_float = intermediate.float()
                out[e] = intermediate_float.pow(2).mean(dim=0)
                hit[e] = True
                if need_dn_sample:
                    routed_positions = (sampled_experts == e).nonzero(
                        as_tuple=True
                    )[0]
                    if routed_positions.numel() > 0:
                        # ``tok`` is sorted and every expert occurs at most once
                        # per token, so searchsorted maps sampled token routes to
                        # the already-computed expert-local activation rows.
                        local_rows = torch.searchsorted(
                            tok,
                            sampled_cache_rows[routed_positions],
                        )
                        if not torch.equal(
                            tok[local_rows],
                            sampled_cache_rows[routed_positions],
                        ):
                            raise AssertionError(
                                f"{qn}: routed sample/token replay mismatch"
                            )
                        sampled_values[routed_positions] = intermediate_float[
                            local_rows
                        ]
            if bool(hit.any()) and not bool(hit.all()):
                out[~hit] = out[hit].mean(dim=0)
            elif not bool(hit.any()):
                out[:] = 1.0
            if need_dn_weight:
                col_weights[dn_name] = out.reshape(E, 1, inter).cpu()
                added.append(dn_name)
            if need_dn_sample:
                source_rows = (
                    source_row_indices[: int(X.shape[0])].to(dev)[
                        sampled_cache_rows
                    ]
                    if source_row_indices is not None
                    else sampled_cache_rows
                )
                routed_sample = RoutedActivationSamples(
                    values=sampled_values.detach().to("cpu").contiguous(),
                    cache_row_indices=sampled_cache_rows.detach().to(
                        "cpu"
                    ).to(torch.int64).contiguous(),
                    source_row_indices=source_rows.detach().to(
                        "cpu"
                    ).to(torch.int64).contiguous(),
                    expert_indices=sampled_experts.detach().to(
                        "cpu"
                    ).to(torch.int64).contiguous(),
                    route_slots=sampled_slots.detach().to(
                        "cpu"
                    ).to(torch.int64).contiguous(),
                    route_weights=sampled_weights.detach().to(
                        "cpu"
                    ).to(torch.float32).contiguous(),
                )
                routed_sample.validate()
                activation_samples[dn_name] = routed_sample
            del t
    return added


def synthesize_packed_expert_activation_samples(
    model_path: str | Path,
    act_dir: str | Path,
    targets: set[str],
    profile=None,
    *,
    max_rows: int = 4096,
    device: str | None = None,
) -> dict[str, torch.Tensor | RoutedActivationSamples]:
    """Replay only packed-expert inputs absent from the probe cache.

    This is the same checkpoint routing/gate/up replay used by the imatrix
    harvester, so activation-scale calibration and weighted export cannot drift
    onto duplicate MoE semantics.
    """

    samples: dict[str, torch.Tensor | RoutedActivationSamples] = {}
    synthesize_packed_expert_col_weights(
        model_path,
        act_dir,
        {},
        profile,
        max_rows=max_rows,
        device=device,
        activation_samples=samples,
        target_names=set(targets),
        write_col_weights=False,
    )
    return samples


def synthesize_unrouted_expert_col_weights(
    probe_stats: Mapping[str, Mapping[str, Any]],
    col_weights: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    """Give never-routed per-expert Linears a neutral-prior imatrix, IN PLACE.

    Architectures that expose routed experts as PER-EXPERT ``nn.Linear``s
    (DeepSeek-V4-Flash) build col-weights straight from the activation cache,
    so an expert that received zero calibration tokens has no entry at all.
    That is not a measurement the run can go and get: the expert is simply not
    on the calibration distribution. It is also not optional to resolve, because
    BOTH ends fail closed on it — ``measure_quant_cost`` refuses to price a CB
    row without exact col-weights, and the exporter refuses to ship a CB target
    without them ("no silent RTN"). The artifact cannot be built while the hole
    exists.

    The rule, stated once here rather than left to be an accident of missing
    rows: an expert with ``n_tokens_seen == 0`` inherits the MEAN of that same
    layer's routed experts' vectors for the same projection. This is exactly the
    convention the packed path already ships for the same situation
    (``_replay_down_proj_col_weights`` assigns ``out[~hit] = out[hit].mean(0)``
    for partially-routed stacks), so per-expert and packed checkpoints are
    treated alike.

    It is honest for the same reason it is safe: a never-routed expert's
    measured sensitivity is exactly zero (the probe records ``h_trace == 0.0``
    and ``h_w2_sum == 0.0``), so the allocator will hand it the cheapest legal
    rung and it consumes no budget it did not earn. The neutral prior decides
    only HOW its bytes are rendered, never how many it gets.

    Returns ``{"names": [...], "rule": ...}`` for the caller to stamp into
    provenance — a synthesized entry must never be indistinguishable from a
    measured one.
    """
    by_layer_proj: dict[tuple[str, str], list[str]] = {}
    unrouted: list[str] = []
    for qname, stat in probe_stats.items():
        if ".mlp.experts." not in qname:
            continue
        head, _, proj = qname.rpartition(".")
        layer = head.split(".mlp.experts.")[0]
        if int(stat.get("n_tokens_seen", 0) or 0) > 0:
            if qname in col_weights:
                by_layer_proj.setdefault((layer, proj), []).append(qname)
        else:
            unrouted.append(qname)

    added: list[str] = []
    for qname in sorted(unrouted):
        if qname in col_weights:
            continue
        head, _, proj = qname.rpartition(".")
        layer = head.split(".mlp.experts.")[0]
        donors = by_layer_proj.get((layer, proj)) or []
        if not donors:
            raise ValueError(
                f"{qname}: no routed sibling expert in {layer} for projection "
                f"{proj}; cannot form a neutral prior. The calibration reached "
                f"none of this layer's experts — re-probe rather than invent a "
                f"vector for an entire layer.")
        stack = torch.stack([col_weights[d].to(torch.float32) for d in donors])
        col_weights[qname] = stack.mean(dim=0)
        added.append(qname)
    return {
        "names": added,
        "rule": "unrouted_expert_neutral_prior:layer_routed_mean",
        "basis": "probe n_tokens_seen == 0",
    }


@torch.no_grad()
def per_expert_stage_activation_calibration(
    act_dir: str | Path,
    members_by_target: Mapping[str, Mapping[tuple[str, int], str]],
    *,
    policy: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Calibrate packed routed-MoE stages from a PER-EXPERT activation cache.

    ``synthesize_packed_expert_activation_samples`` replays routing from the
    checkpoint because the packed topology caches only the experts MODULE's
    input, so a ``down_proj`` stage has no measured routed intermediate. A
    checkpoint that exposes routed experts as per-expert ``nn.Linear``s
    (DeepSeek-V4-Flash) is the opposite case: the probe cached EVERY expert's
    own inputs, so both stages are already measured and the replay's entry
    condition (``moe_imatrix``'s ``f"{src}.0.gate_proj.weight" in weight_map``)
    never fires. Nothing needs replaying here — only pooling.

    Returns ``(samples, max_abs)`` keyed by the PACKED stage qname:

    * ``samples`` covers the fused gate/up stage (``w13``). Each row is one
      expert's column-wise max |x| over that expert's cached rows, so the
      tensor's ``abs().max()`` is the EXACT pooled module-input max-abs while
      residency stays at one row per expert rather than one row per routed
      token (43 layers x 256 experts x 64 rows x 4096 cols would be ~11 GB
      resident, and the contract builds every stage's sample up front).
    * ``max_abs`` covers the down stage (``w2``). Its inputs are the routed
      INTERMEDIATE, measured directly — not a module input and not a replay —
      which is exactly the separation the stage attestation exists to publish.

    Because the pooling is a max reduction it is exact for every amax-derived
    policy and meaningless for a distribution-fitting one, so
    ``mse_grid_calibrated.v1`` is refused rather than silently mis-fitted.
    """
    from prismaquant.nvfp4_activation_contract import (
        MSE_GRID_INPUT_GLOBAL_SCALE_POLICY,
        resolve_input_global_scale_policy,
    )

    if policy is not None and (
        resolve_input_global_scale_policy(policy)
        == MSE_GRID_INPUT_GLOBAL_SCALE_POLICY
    ):
        raise ValueError(
            "per-expert routed-MoE stage calibration pools each expert's "
            "cached rows to a column-wise max, which is exact for an "
            "amax-derived policy and carries no distribution for "
            f"{MSE_GRID_INPUT_GLOBAL_SCALE_POLICY!r}; re-probe with a packed "
            "experts-module cache entry or choose an amax policy"
        )

    act_dir = Path(act_dir)
    samples: dict[str, torch.Tensor] = {}
    max_abs: dict[str, float] = {}
    unrouted: dict[str, int] = {}
    for packed_qname in sorted(members_by_target):
        member_qnames = members_by_target[packed_qname]
        experts = sorted({e for _proj, e in member_qnames})
        rows: list[torch.Tensor] = []
        absent = 0
        for e in experts:
            per_expert: list[torch.Tensor] = []
            for (proj, expert_id), member in sorted(member_qnames.items()):
                if expert_id != e:
                    continue
                path = act_dir / (
                    _ACT_FNAME_SUB.sub("__", member) + ".pt")
                if not path.exists():
                    # A NEVER-ROUTED expert. The probe writes a cache entry per
                    # Linear it actually saw, so an expert off the calibration
                    # distribution has no file at all (DSv4-Flash 16x512:
                    # 5,505 of 33,153 across 40 layers). It contributes no
                    # observed activation, so it contributes nothing to a max
                    # — the same reading `synthesize_unrouted_expert_col_weights`
                    # takes of the same zero. Counted, never silent.
                    absent += 1
                    continue
                _name, inputs, _row_ids = _load_act_entry(path)
                if inputs is None or inputs.numel() == 0:
                    raise ValueError(
                        f"{packed_qname}: activation cache entry for "
                        f"{member!r} has no value-bearing rows")
                per_expert.append(inputs.abs().amax(dim=0))
            if not per_expert:
                continue
            widths = {int(v.numel()) for v in per_expert}
            if len(widths) != 1:
                raise ValueError(
                    f"{packed_qname}: expert {e} cached inputs disagree on "
                    f"in_features across the fused projections: {widths}")
            rows.append(torch.stack(per_expert).amax(dim=0)
                        if len(per_expert) > 1 else per_expert[0])
        if not rows:
            raise ValueError(
                f"{packed_qname}: NO expert in this stack has an activation "
                f"cache entry ({len(member_qnames)} members, all absent), so "
                "the stage has no calibrated input at all; re-probe rather "
                "than ship an uncalibrated static W4A4 scalar")
        if absent:
            unrouted[packed_qname] = absent
        pooled = torch.stack(rows).contiguous()
        value = float(pooled.max().item())
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{packed_qname}: pooled routed-MoE stage max-abs is {value!r}")
        if len(_PACKED_GATE_UP_LEAVES & {packed_qname.rsplit('.', 1)[1]}):
            samples[packed_qname] = pooled
        else:
            max_abs[packed_qname] = value
    if unrouted:
        total = sum(unrouted.values())
        worst = max(unrouted, key=unrouted.get)
        print(
            f"[export-cb-stream] routed-MoE stage calibration skipped {total} "
            f"never-routed expert projection(s) across {len(unrouted)} "
            f"stack(s) (worst: {worst} with {unrouted[worst]})",
            flush=True,
        )
    return samples, max_abs
