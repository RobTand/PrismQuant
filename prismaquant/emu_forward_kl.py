"""Whole-model emulated forward KL-vs-BF16 (NVFP4-CB Phase-0 gold metric).

There is no NVFP4-CB kernel in Phase 0, so nothing can be served. The strongest
available gate is a *weight-level bit-faithful* emulation: load the model bf16
resident, swap each target Linear's weight for its format's emulation
reconstruction (registry ``quantize_dequantize``; GGUF/IQ via
``gguf_quantize_dequantize``; weighted formats take ``col_weights``), optionally
emulate the served W4A4 activation bucket with a forward-pre-hook, and compute
full-vocab KL against the un-swapped bf16 model on held-out text.

Two model copies won't co-reside for 4B on a 128 GB unified-memory box, so this
runs **two passes over one resident model**: pass 1 buffers bf16 teacher
log-probs (fp32, CPU) chunk-by-chunk; pass 2 swaps the weights in place and
computes KL against the buffer. Original weights are restored after.

KL / confident-position / top-1 convention is the repo's served metric
(``KL(P_bf16 || Q_quant) = Σ_v p·(log p −
log q)`` per position); it matches the historical served-scoring script's
convention. A position is **BF16-confident** when the teacher top-1
probability strictly exceeds ``_CONFIDENT_PROB`` (0.5 — the majority
threshold, derived next to the constant); top-1 agreement is
``argmax P == argmax Q``. Here it is computed full-vocab in fp32 (bf16 KL
differencing is a known measurement floor in this repo), not from served top-K.
Every emitted result stamps the cut that produced it as
``confident_prob_cut``, so a ``kl_confident`` number always carries the
convention it was computed under and cannot silently disagree with another
tool's number.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
from pathlib import Path
from typing import Mapping

import torch

from . import format_registry as fr
from .allocator_candidates import PASSTHROUGH_SOURCE_REQUIREMENTS
from .measure_quant_cost import canonical_linear_name

# Teacher-confident threshold: teacher top-1 prob must strictly exceed this
# for a position to enter the clean confident lane.
#
# The value 0.5 is the majority threshold, not an intuition number, and the
# derivation is the cut itself: a top-1 probability strictly above 0.5 means
# the top-1 token holds more mass than all other tokens combined, so the
# argmax is invariant to ANY redistribution of the remaining mass. No smaller
# cut has that property (at top-1 <= 0.5 the runner-up mass can tie or beat
# the leader under some redistribution), and no larger cut is needed for it.
# That invariance is exactly what "the model has actually decided" means for
# a lane whose job is to separate decided positions from undecided ones, so
# 0.5 is the unique cut with the required property. Do not retune it without
# re-deriving the lane; the value stamped into each result
# (``confident_prob_cut``) names the convention that produced the number.
_CONFIDENT_PROB = 0.5


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Weight / activation emulation
# ---------------------------------------------------------------------------

# Families whose served path quantizes activations (W4A4 / W8A8 style).
# GGUF/IQ serve weight-only (dequant to fp16), so they get NO activation
# emulation — this asymmetry is deliberate: each format is measured with its
# served activation behaviour. fp8_cb decodes to FP8 and serves FP8-dynamic
# activations (registered spec: act_bits=8, fp8_e4m3).
_ACT_EMULATION_FAMILIES = {"nv", "mx", "nvfp4_cb", "fp8_cb"}


def _qdq_accepts_col_weights(spec: fr.FormatSpec) -> bool:
    try:
        sig = inspect.signature(spec.quantize_dequantize)
    except (TypeError, ValueError):
        return False
    return "col_weights" in sig.parameters


def weighted_quantize_dequantize(
    spec: fr.FormatSpec,
    w: torch.Tensor,
    col_weights: torch.Tensor | None,
    *,
    qname: str | None = None,
) -> torch.Tensor:
    """THE single weighted-render definition: reconstruct ``w`` in ``spec``'s
    format, applying the per-input-column imatrix when the family's exporter
    does.

    One function, three callers by design (principle #8, one render): this
    emulation path, the inline cost render
    (``measure_quant_cost._cost_render_uses_imatrix``), and — since re-vet R3
    / CB Milestone C — ``production_weight_cache.render_production_weight``.
    ``col_weights=None`` reproduces the unweighted registry render exactly, so
    every non-weighted family is bit-identical whether or not a caller has a
    vector in hand.
    """
    if spec.family == "gguf":
        from .gguf_formats import gguf_quantize_dequantize
        cw = None if col_weights is None else col_weights.to(w.device)
        return gguf_quantize_dequantize(w.clone(), spec.name, col_weights=cw)
    if spec.family in {"nvfp4_cb", "fp8_cb"}:
        from .nvfp4_cb_footprint import (
            cb_quantize_dequantize_for_context,
            cb_serialization_context_from_env,
        )

        return cb_quantize_dequantize_for_context(
            spec,
            w.clone(),
            context=cb_serialization_context_from_env(),
            qname=qname,
            col_weights=(
                None if col_weights is None else col_weights.to(w.device)
            ),
        )
    if col_weights is not None and _qdq_accepts_col_weights(spec):
        return spec.quantize_dequantize(w.clone(), col_weights=col_weights.to(w.device))
    return spec.quantize_dequantize(w.clone())


# Historical name, kept for this module's own call sites.
_render_weight = weighted_quantize_dequantize


def _wants_act_emulation(spec: fr.FormatSpec) -> bool:
    # "Does serving quantize the input activations?" has exactly one
    # definition (FormatSpec.act_quant_changes_input); never re-derive it from
    # act_bits here, or an A16 rung declared as act_bits=16 would be emulated
    # as a W-and-A format while the allocator prices it as passthrough.
    return (spec.act_quant_changes_input
            and spec.family in _ACT_EMULATION_FAMILIES)


class _WeightSwapper:
    """Context manager: swap target Linear weights + hook activations, restore."""

    def __init__(self, model, targets, *, act_emulation: bool):
        # targets: list of (qname, module, spec, col_weights, smooth_scale)
        self._targets = targets
        self._act_emulation = act_emulation
        self._orig: list[tuple[object, torch.Tensor]] = []
        self._handles: list = []
        # (qname, format_name) -> number of forwards where activation
        # emulation failed and the raw input was used instead. NEVER silent:
        # measure_emulated_kl raises on any nonzero count unless the caller
        # explicitly allowed the weight-only fallback.
        self.fallback_counts: dict[tuple[str, str], int] = {}

    def __enter__(self):
        for qname, mod, spec, cw, smooth in self._targets:
            if spec.family in {"nvfp4_cb", "fp8_cb"} and cw is None:
                raise RuntimeError(
                    f"{qname}={spec.name}: direct CB KL render has no "
                    "production col_weights. CB export is imatrix-weighted; "
                    "use a production cache/materialized render instead of "
                    "validating an unweighted fallback."
                )
            w = mod.weight.data
            # SmoothQuant fold: quantize W' = W·diag(s) (s over the input dim);
            # the inverse activation scale x→x/s is applied in the act hook so
            # the effective compute (x/s)@(W diag(s))^T = x@W^T is preserved and
            # only the *rounded* weight/activation buckets shift. col_weights
            # (E[x'^2]=E[x^2]/s^2) are recomputed in lockstep by the caller.
            w_in = w if smooth is None else w * smooth.to(w.device, w.dtype)
            w_hat = _render_weight(
                spec, w_in, cw, qname=qname
            ).to(dtype=w.dtype, device=w.device)
            self._orig.append((mod, w))  # keep original tensor (still resident)
            mod.weight.data = w_hat
            if self._act_emulation and _wants_act_emulation(spec):
                self._handles.append(
                    mod.register_forward_pre_hook(
                        _make_act_hook(spec, qname, self.fallback_counts,
                                       smooth_scale=smooth)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        for mod, w in self._orig:
            mod.weight.data = w
        self._handles.clear()
        self._orig.clear()
        return False


def _make_act_hook(
    spec: fr.FormatSpec,
    qname: str,
    fallback_counts: dict[tuple[str, str], int],
    smooth_scale: torch.Tensor | None = None,
):
    aqdq = spec.activation_quantize_dequantize
    key = (qname, spec.name)

    def _hook(module, args):
        if not args:
            return args
        x = args[0]
        if smooth_scale is not None:
            x = x / smooth_scale.to(x.device, x.dtype)
        try:
            x_hat = aqdq(x)
        except Exception:
            # Activation qdq may require a shape multiple (e.g. last dim
            # % 16). A silent weight-only fallback would be a measurement
            # confound (plausible-but-wrong KL), so record it here and let
            # measure_emulated_kl fail fast unless explicitly allowed.
            fallback_counts[key] = fallback_counts.get(key, 0) + 1
            return args
        return (x_hat,) + tuple(args[1:])

    return _hook


def _collect_targets(model, format_map: Mapping[str, dict]):
    """Resolve format_map (qname -> {'format', 'col_weights'}) to live modules.

    Passthrough entries (BF16 / FP8_SOURCE / identity qdq with no activation
    emulation) are skipped — swapping them is a no-op and keeping them out
    guarantees an all-passthrough map yields KL == 0 exactly.
    """
    import torch.nn as nn

    targets = []
    matched = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        qname = canonical_linear_name(name)
        entry = format_map.get(qname)
        if entry is None:
            continue
        matched.add(qname)
        fmt = entry["format"] if isinstance(entry, Mapping) else entry
        spec = fr.get_format(fmt)
        cw = entry.get("col_weights") if isinstance(entry, Mapping) else None
        smooth = entry.get("smooth_scale") if isinstance(entry, Mapping) else None
        # Only the SOURCE-PASSTHROUGH family is passthrough-only (CLAUDE.md
        # principle 11); every other format — including real FP8_E4M3 — is
        # rendered. Read from the one table rather than a name tuple: this
        # list was ("BF16", "FP8_SOURCE") when those were the only two, and a
        # newly censused native format silently missing from it would be
        # RE-RENDERED here — emulating quantization error for bytes the
        # exporter copies verbatim, i.e. reporting a KL the artifact does not
        # have.
        skip_weight = spec.name in PASSTHROUGH_SOURCE_REQUIREMENTS
        if skip_weight and not _wants_act_emulation(spec):
            continue
        targets.append((qname, mod, spec, cw, smooth))
    return targets, matched


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _load_chunks(tokenizer, dataset_path, *, seqlen: int, max_tokens: int):
    """Tokenize a raw-text file into ``seqlen`` chunks (BOS-prefixed)."""
    text = Path(dataset_path).read_text(errors="ignore")
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[: int(max_tokens)]
    bos = tokenizer.bos_token_id
    chunks = []
    for i in range(0, len(ids), int(seqlen)):
        block = ids[i:i + int(seqlen)]
        if not block:
            continue
        if bos is not None:
            block = [bos] + block
        chunks.append(torch.tensor(block, dtype=torch.long).unsqueeze(0))
    return chunks


# ---------------------------------------------------------------------------
# KL reduction (kl_tool.py convention, full-vocab fp32)
# ---------------------------------------------------------------------------

class _KLAccumulator:
    def __init__(self):
        self.kl_sum = 0.0
        self.kl_conf_sum = 0.0
        self.n = 0
        self.n_conf = 0
        self.agree_all = 0
        self.agree_conf = 0

    def add(self, teacher_lp: torch.Tensor, student_lp: torch.Tensor):
        # teacher_lp, student_lp: [P, V] fp32 log-probs.
        p = teacher_lp.exp()
        kl = (p * (teacher_lp - student_lp)).sum(dim=-1)  # [P]
        t_max_lp, t_arg = teacher_lp.max(dim=-1)
        s_arg = student_lp.argmax(dim=-1)
        conf = t_max_lp.exp() > _CONFIDENT_PROB
        agree = t_arg == s_arg

        self.kl_sum += float(kl.sum().item())
        self.n += int(kl.numel())
        self.agree_all += int(agree.sum().item())
        self.kl_conf_sum += float(kl[conf].sum().item())
        self.n_conf += int(conf.sum().item())
        self.agree_conf += int((agree & conf).sum().item())

    def result(self) -> dict:
        n = max(self.n, 1)
        if self.n_conf == 0:
            # An empty confident lane has no mean: dividing by
            # max(n_conf, 1) would emit kl_confident = 0.0 — the best possible
            # score, computed over nothing — and report a win on an empty set.
            # Refuse instead; the only caller (measure_emulated_kl) already
            # carries ValueError for empty/ambiguous inputs, so fail closed
            # there too rather than emitting a number no consumer can trust.
            raise ValueError(
                "emu_forward_kl: confident lane is empty "
                f"(confident_prob_cut={_CONFIDENT_PROB!r}, "
                f"n_positions={int(self.n)}): no teacher top-1 probability "
                "strictly exceeded the cut, so kl_confident has no mean. "
                "Refusing rather than emitting 0.0 over nothing."
            )
        nc = self.n_conf
        return {
            "kl_all": self.kl_sum / n,
            "kl_confident": self.kl_conf_sum / nc,
            "top1_agreement": self.agree_conf / nc,
            "top1_agreement_all": self.agree_all / n,
            "n_positions": int(self.n),
            "n_confident": int(self.n_conf),
            # Stamp the cut that produced kl_confident/top1_agreement so any
            # consumer of the number can see which convention it came from.
            "confident_prob_cut": _CONFIDENT_PROB,
        }


def _logits(model, ids: torch.Tensor) -> torch.Tensor:
    out = model(ids)
    return out.logits if hasattr(out, "logits") else out[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_emulated_kl(
    model_path: str,
    format_map: Mapping[str, dict],
    dataset_path: str,
    *,
    device: str = "cuda",
    seqlen: int = 512,
    max_tokens: int = 8192,
    act_emulation: bool = True,
    allow_act_fallback: bool = False,
    allow_missing_targets: bool = False,
    cache_dir: str | None = None,
) -> dict:
    """Whole-model emulated forward KL-vs-BF16.

    Args:
      model_path: HF model dir / id (loaded bf16 resident).
      format_map: ``{qname: {"format": str, "col_weights": Tensor|None}}``.
        A plain ``{qname: format_str}`` is also accepted.
      dataset_path: raw-text file (held-out WikiText).
      allow_act_fallback: a failed activation emulation silently degrades an
        arm to weight-only — a measurement confound. By default any fallback
        RAISES; pass True to tolerate it, in which case
        ``act_fallback_counts`` is reported in the result.
      allow_missing_targets: format_map entries that match no live Linear
        (e.g. a qname typo) yield a false-clean KL. By default they RAISE;
        pass True to tolerate, in which case ``n_targets_missing`` and
        ``missing_targets`` are reported in the result.
    Returns ``{kl_all, kl_confident, top1_agreement, n_positions, ...,
    confident_prob_cut, provenance{git_commit, assignment_sha256,
    dataset_sha256}}``. ``confident_prob_cut`` stamps the teacher top-1
    probability cut the confident lane was computed under. Raises ValueError
    when the confident lane is empty (no teacher top-1 strictly exceeded the
    cut): an empty lane has no mean, and emitting 0.0 over nothing would
    report a win on an empty set.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    dev = torch.device(device)

    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
        **({"cache_dir": cache_dir} if cache_dir else {}))
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        **({"cache_dir": cache_dir} if cache_dir else {}))
    model.to(dev).eval()

    chunks = _load_chunks(tok, dataset_path, seqlen=seqlen, max_tokens=max_tokens)
    if not chunks:
        raise ValueError("emu_forward_kl: dataset produced no token chunks")

    targets, matched = _collect_targets(model, format_map)
    missing = sorted(set(format_map) - matched)
    if missing and not allow_missing_targets:
        preview = ", ".join(missing[:20])
        raise ValueError(
            f"emu_forward_kl: {len(missing)} format_map entries matched no "
            f"live Linear (qname typo → false-clean KL): {preview}"
            + (" …" if len(missing) > 20 else "")
            + ". Pass allow_missing_targets=True to tolerate.")

    with torch.no_grad():
        # Pass 1 — bf16 teacher log-probs, buffered fp32 on CPU.
        teacher_lp = []
        for ids in chunks:
            lp = torch.log_softmax(_logits(model, ids.to(dev)).float(), dim=-1)
            teacher_lp.append(lp[0].to("cpu"))

        # Pass 2 — swap weights + emulate activations, KL against the buffer.
        acc = _KLAccumulator()
        swapper = _WeightSwapper(model, targets, act_emulation=act_emulation)
        with swapper:
            for ids, t_lp in zip(chunks, teacher_lp):
                s_lp = torch.log_softmax(
                    _logits(model, ids.to(dev)).float(), dim=-1)[0]
                acc.add(t_lp.to(dev), s_lp)

    fallbacks = {
        f"{q} [{fmt}]": n
        for (q, fmt), n in sorted(swapper.fallback_counts.items())
    }
    if fallbacks and not allow_act_fallback:
        raise RuntimeError(
            "emu_forward_kl: activation emulation failed and fell back to "
            "the raw input on "
            f"{len(fallbacks)} layer(s) — the arm would be measured "
            "weight-only (measurement confound): "
            + ", ".join(list(fallbacks)[:20])
            + (" …" if len(fallbacks) > 20 else "")
            + ". Pass allow_act_fallback=True to tolerate.")

    result = acc.result()
    if fallbacks:
        result["act_fallback_counts"] = fallbacks
    if missing:
        result["n_targets_missing"] = len(missing)
        result["missing_targets"] = missing
    assignment = {
        q: (e["format"] if isinstance(e, Mapping) else e)
        for q, e in format_map.items()
    }
    tok_bytes = b"".join(
        c.to(torch.int32).numpy().tobytes() for c in chunks)
    result["n_targets_swapped"] = len(targets)
    result["n_targets_matched"] = len(matched)
    result["provenance"] = {
        "git_commit": _git_commit(),
        "assignment_sha256": _sha256(
            json.dumps(assignment, sort_keys=True).encode()),
        "dataset_sha256": _sha256(tok_bytes),
        "seqlen": int(seqlen),
        "max_tokens": int(max_tokens),
        "act_emulation": bool(act_emulation),
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _expand_uniform(model, fmt: str, col_weights: dict) -> dict:
    import torch.nn as nn
    fmap = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            q = canonical_linear_name(name)
            fmap[q] = {"format": fmt, "col_weights": col_weights.get(q)}
    return fmap


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Whole-model emulated forward KL-vs-BF16.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="raw-text held-out file")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--assignment", help="layer_config.json path")
    g.add_argument("--uniform-format", help="apply one format to all Linears")
    ap.add_argument("--col-weights-pkl", default=None,
                    help="pickle dict qname->tensor")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--no-act-emulation", action="store_true")
    ap.add_argument("--allow-act-fallback", action="store_true",
                    help="tolerate (and report) failed activation emulation "
                         "instead of raising")
    ap.add_argument("--allow-missing-targets", action="store_true",
                    help="tolerate (and report) format_map entries that "
                         "match no live Linear instead of raising")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    col_weights = {}
    if args.col_weights_pkl:
        import pickle
        with open(args.col_weights_pkl, "rb") as f:
            col_weights = pickle.load(f)

    if args.uniform_format:
        # Need the model to enumerate Linears for a uniform map.
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, trust_remote_code=True)
        format_map = _expand_uniform(model, args.uniform_format, col_weights)
        del model
    else:
        from .layer_config import load_assignment
        assignment = load_assignment(args.assignment)
        format_map = {
            q: {"format": fmt, "col_weights": col_weights.get(q)}
            for q, fmt in assignment.items()
        }

    result = measure_emulated_kl(
        args.model, format_map, args.dataset,
        device=args.device, seqlen=args.seqlen, max_tokens=args.max_tokens,
        act_emulation=not args.no_act_emulation,
        allow_act_fallback=args.allow_act_fallback,
        allow_missing_targets=args.allow_missing_targets,
        cache_dir=args.cache_dir,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
