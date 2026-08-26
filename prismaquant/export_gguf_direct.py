"""Streaming direct-from-HF GGUF exporter (no llama.cpp skeleton).

For architectures llama.cpp cannot convert (e.g. Tencent Hy3 / hy_v3),
the artifact targets vLLM's GGUF path (vllm-gguf-plugin + a small
weights adapter), so tokenizer embedding and llama.cpp arch tables are
unnecessary. Tensor names in the output are the HF qnames verbatim —
the vLLM adapter maps 1:1 — with MoE experts stacked into one 3-D
tensor per (layer, projection) (the GGUF/ggml fused-MoE layout; one
quant type per stacked tensor = experts uniform per layer).

Streaming discipline (GPU-first, bounded memory):
  - safetensors shards are opened lazily; tensors are read on demand.
  - FP8 sources are dequantized per tensor (``w.float() * weight_scale``
    for the per-tensor-scale scheme Hy3-FP8 ships).
  - quantization runs on CUDA; packed bytes stream straight into the
    GGUF via the incremental writer API (add_tensor_info for all, then
    write_tensor_data one by one) — neither the source nor the artifact
    is ever memory-resident.

Note on the emulation==bytes contract: it holds PER DEVICE. FP8-dequant
values are heavily tie-degenerate (256 codes x one scale), and CPU vs
CUDA reduction order flips ~0.06% of near-tie scale picks. The pipeline
measures cost and exports on CUDA, so the shipped bytes match the
measured emulation; do not mix a CPU-measured cost with a GPU export.

Assignment: a layer_config.json in the usual HF-qname space, where the
stacked expert tensors use the qname
``model.layers.N.mlp.experts.{gate,up,down}_proj`` (one entry per
stacked tensor). Anything absent keeps source precision (BF16/F16
passthrough for norms etc.); an explicit --default-* recipe covers the
uniform hand-recipe case without an allocator run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType as QT
from safetensors import safe_open

from prismaquant.gguf_formats import GGUF_BLOCK_BYTES, gguf_pack
from prismaquant.layer_config import load_assignment
from prismaquant.model_profiles import detect_profile
from prismaquant.prismasnap_contract import refuse_prismasnap_for_unvalidated_lane

_EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def build_direct_imatrix(act_dir: str | Path) -> dict[str, torch.Tensor]:
    """Per-column mean squared activation per cached module, from the
    probe's activation cache. Dense Linears key by their recipe qname;
    packed-experts module snapshots key by the experts module qname
    (``…mlp.experts``) — the exact input of gate/up projections. Ops are
    IDENTICAL to the cost path's derivation (full rows, fp32, mean over
    dim 0): measured cost and shipped bytes stay in lockstep."""
    out: dict[str, torch.Tensor] = {}
    for p in sorted(Path(act_dir).glob("*.pt")):
        blob = torch.load(p, map_location="cpu", weights_only=False)
        inputs = blob.get("inputs") if isinstance(blob, dict) else None
        if inputs is None or inputs.ndim != 2:
            continue
        name = (blob.get("name") if isinstance(blob, dict) else None) or (
            p.stem.replace("__", ".")
        )
        out[name] = inputs.float().pow(2).mean(dim=0)
    return out


def _imatrix_vector_for(
    imatrix: dict[str, torch.Tensor],
    out_name: str,
    kind: str,
    in_features: int,
    recipe_qname_fn,
) -> torch.Tensor | None:
    """Resolve the importance vector for one output tensor, or None.

    Stacked experts use the experts-module snapshot when the column count
    matches (gate/up: module input = their input; down_proj's input is
    the uncached per-expert intermediate → stays unweighted, matching
    the cost path's shape guard)."""
    base = out_name.removesuffix(".weight")
    if kind == "experts":
        module_qname = base.rsplit(".experts.", 1)[0] + ".experts"
        qw = imatrix.get(module_qname)
    else:
        recipe = recipe_qname_fn(base)
        qw = imatrix.get(recipe) if recipe else None
    if qw is not None and qw.numel() != in_features:
        return None
    return qw


class _ShardIndex:
    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        idx = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = idx["weight_map"]
        self._open: dict[str, object] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self._open:
            self._open[shard] = safe_open(
                self.dir / shard, framework="pt", device="cpu",
            )
        return self._open[shard].get_tensor(name)

    def dequant(self, base: str) -> torch.Tensor:
        """Weight tensor in float32, applying an FP8 per-tensor scale if
        the checkpoint carries one."""
        w = self.get(base + ".weight")
        scale_name = base + ".weight_scale"
        if w.dtype == torch.float8_e4m3fn and scale_name in self.weight_map:
            scale = self.get(scale_name).float()
            if scale.numel() != 1:
                raise ValueError(
                    f"{base}: expected a per-tensor weight_scale, got "
                    f"shape {tuple(scale.shape)} — wire the block-scale "
                    f"dequant before exporting this checkpoint"
                )
            return w.float() * scale
        return w.float()


def _plan_tensors(shards: _ShardIndex) -> list[tuple[str, str, list[str]]]:
    """Return (output_name, kind, source_names) in a stable order.

    kind: "linear" (2-D weight, maybe FP8), "experts" (stack per-expert
    weights into 3-D), "raw" (norms, biases, embeddings — passthrough).
    """
    experts: dict[tuple[str, str], dict[int, str]] = {}
    plan: list[tuple[str, str, list[str]]] = []
    seen_bases: set[str] = set()

    for name in sorted(shards.weight_map):
        if name.endswith((".weight_scale", ".input_scale")):
            continue
        m = _EXPERT_RE.match(name)
        if m:
            prefix, idx, proj = m.group(1), int(m.group(2)), m.group(3)
            experts.setdefault((prefix, proj), {})[idx] = name[: -len(".weight")]
            continue
        if name.endswith(".weight"):
            base = name[: -len(".weight")]
            seen_bases.add(base)
            plan.append((name, "linear", [base]))
        else:
            plan.append((name, "raw", [name]))

    for (prefix, proj), members in sorted(experts.items()):
        n = len(members)
        if sorted(members) != list(range(n)):
            raise ValueError(f"{prefix}.experts.{proj}: non-contiguous expert ids")
        plan.append((
            f"{prefix}.experts.{proj}.weight", "experts",
            [members[i] for i in range(n)],
        ))
    return plan


def export_gguf_direct(
    model_dir: str | Path,
    out_path: str | Path,
    layer_config_path: str | Path | None = None,
    default_expert_format: str | None = None,
    default_linear_format: str | None = None,
    token_embedding_format: str | None = None,
    output_format: str | None = None,
    device: str | None = None,
    arch_name: str = "prismaquant-direct",
    exclude: tuple[str, ...] = (),
    imatrix: dict[str, torch.Tensor] | None = None,
) -> dict[str, int]:
    refuse_prismasnap_for_unvalidated_lane(
        model_dir, lane="direct GGUF/vLLM-GGUF export"
    )
    if imatrix is not None and not imatrix:
        raise ValueError(
            "imatrix requested but empty — act-cache dir missing or "
            "unexpected blob schema; unweighted bytes would diverge from "
            "the imatrix-weighted cost measurement"
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    shards = _ShardIndex(model_dir)
    profile = detect_profile(str(model_dir))
    assignment: dict[str, str] = {}
    if layer_config_path is not None:
        assignment = load_assignment(layer_config_path)

    def _recipe_qname(checkpoint_base: str) -> str | None:
        """Checkpoint tensor base -> the recipe/live qname the allocator's
        layer_config uses (e.g. mlp.shared_mlp -> mlp.shared_experts)."""
        live = profile.checkpoint_to_live_name(checkpoint_base + ".weight")
        if live is None:
            return None
        return live.removesuffix(".weight")

    def _fmt_for(qname: str, kind: str, proj: str | None = None) -> str | None:
        if kind == "experts":
            # The allocator keys stacked experts by the LIVE packed param
            # (…experts.gate_up_proj / …experts.down_proj); the exporter
            # plans per source projection. gate/up share the fused packed
            # entry's format by construction.
            prefix = qname.rsplit(".experts.", 1)[0] + ".experts"
            packed = "down_proj" if proj == "down_proj" else "gate_up_proj"
            hit = assignment.get(f"{prefix}.{packed}")
            if hit is not None:
                return hit
            return default_expert_format
        if qname in assignment:
            return assignment[qname]
        if qname == "model.embed_tokens":
            return token_embedding_format
        if qname == "lm_head":
            return output_format
        return default_linear_format

    plan = _plan_tensors(shards)
    cfg = json.loads((Path(model_dir) / "config.json").read_text())

    writer = gguf.GGUFWriter(str(out_path), arch_name)
    writer.add_block_count(int(cfg.get("num_hidden_layers", 0)))
    writer.add_key_value("prismaquant.source_model",
                         str(model_dir), gguf.GGUFValueType.STRING)

    counts: Counter[str] = Counter()
    tensor_formats: dict[str, str] = {}
    staged: list[tuple[str, str, list[str], str | None]] = []

    # Pass 1: tensor metadata only (shapes/types), no data in memory.
    for out_name, kind, sources in plan:
        if any(re.search(p, out_name) for p in exclude):
            counts["excluded"] += 1
            continue
        base = out_name.removesuffix(".weight")
        fmt = None
        explicit = False
        if kind == "experts":
            proj = base.rsplit(".", 1)[-1]
            fmt = _fmt_for(base, kind, proj=proj)
            prefix = base.rsplit(".experts.", 1)[0] + ".experts"
            packed = "down_proj" if proj == "down_proj" else "gate_up_proj"
            explicit = f"{prefix}.{packed}" in assignment
        elif kind == "linear":
            recipe = _recipe_qname(base)
            if recipe is not None:
                fmt = _fmt_for(recipe, kind)
                explicit = recipe in assignment
        if kind == "experts":
            first = shards.get(sources[0] + ".weight")
            shape = (len(sources), *first.shape)
        else:
            shape = tuple(shards.get(
                sources[0] + (".weight" if kind == "linear" else "")
            ).shape)
        wants_quant = (
            fmt is not None and fmt in GGUF_BLOCK_BYTES
            and len(shape) >= 2 and shape[-1] % GGUF_BLOCK_BYTES[fmt][0] == 0
        )
        if fmt is not None and fmt in GGUF_BLOCK_BYTES and not wants_quant:
            if explicit:
                # Explicit allocator assignment must never silently ship at
                # source precision (same contract as export_gguf).
                raise ValueError(
                    f"{out_name}: assigned {fmt} but shape {shape} fails "
                    f"the block constraint"
                )
            counts[f"default_skip({fmt})"] += 1
            fmt = None
        if wants_quant:
            block, type_size = GGUF_BLOCK_BYTES[fmt]
            n_elem = int(np.prod(shape))
            # add_tensor_info with a quantized raw_dtype expects the BYTE
            # shape (it derives the logical shape itself).
            byte_shape = list(shape[:-1]) + [shape[-1] // block * type_size]
            writer.add_tensor_info(
                out_name, byte_shape,
                np.dtype(np.uint8), n_elem // block * type_size,
                getattr(QT, fmt),
            )
            staged.append((out_name, kind, sources, fmt))
            counts[fmt] += 1
            tensor_formats[out_name] = fmt
        elif len(shape) == 1:
            # 1-D passthrough at F32 (norms, expert_bias — routing-critical;
            # matches the GGUF convention and transformers'
            # _keep_in_fp32_modules_strict for e_score_correction_bias).
            writer.add_tensor_info(
                out_name, list(shape), np.dtype(np.float32),
                int(shape[0]) * 4, QT.F32,
            )
            staged.append((out_name, kind, sources, "F32"))
            counts["F32"] += 1
            tensor_formats[out_name] = "F32"
        else:
            # 2-D+ passthrough at F16 (tensors the recipe skips).
            n_elem = int(np.prod(shape))
            writer.add_tensor_info(
                out_name, list(shape), np.dtype(np.float16),
                n_elem * 2, QT.F16,
            )
            staged.append((out_name, kind, sources, None))
            counts["F16"] += 1
            tensor_formats[out_name] = "F16"

    writer.add_key_value("prismaquant.tensor_formats",
                         json.dumps(tensor_formats, sort_keys=True),
                         gguf.GGUFValueType.STRING)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    # Pass 2: stream data tensor by tensor.
    for i, (out_name, kind, sources, fmt) in enumerate(staged):
        if kind == "experts":
            w = torch.stack([shards.dequant(s) for s in sources]).to(device)
        elif kind == "linear":
            w = shards.dequant(sources[0]).to(device)
        else:
            w = shards.get(sources[0]).float().to(device)
        if fmt == "F32":
            data = w.to(torch.float32).cpu().numpy()
        elif fmt is not None:
            qw = None
            if imatrix:
                qw = _imatrix_vector_for(
                    imatrix, out_name, kind, int(w.shape[-1]),
                    _recipe_qname,
                )
                if qw is not None:
                    counts["imatrix_weighted"] += 1
                else:
                    counts["imatrix_fallback"] += 1
            data = gguf_pack(w, fmt, col_weights=qw)
        else:
            data = w.to(torch.float16).cpu().numpy()
        writer.write_tensor_data(data)
        del w, data
        if (i + 1) % 50 == 0 or i + 1 == len(staged):
            print(f"[export-direct] {i + 1}/{len(staged)} tensors", flush=True)

    writer.close()
    return dict(counts)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF snapshot dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer-config", default=None)
    ap.add_argument("--default-expert-format", default=None)
    ap.add_argument("--default-linear-format", default=None)
    ap.add_argument("--token-embedding-format", default=None)
    ap.add_argument("--output-format", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--exclude", action="append", default=[],
                    help="regex of output tensor names to drop (e.g. MTP)")
    ap.add_argument(
        "--imatrix-from-act-cache", default=None,
        help="probe activation-cache dir; applies per-column importance "
        "weighting to k-quant scale selection (dense Linears exact; "
        "stacked experts pooled from the experts-module snapshot for "
        "gate/up, down_proj unweighted — matches the cost path)",
    )
    args = ap.parse_args(argv)
    imatrix = None
    if args.imatrix_from_act_cache:
        imatrix = build_direct_imatrix(args.imatrix_from_act_cache)
        print(f"imatrix: {len(imatrix)} modules from act cache")
    counts = export_gguf_direct(
        args.model, args.out,
        layer_config_path=args.layer_config,
        default_expert_format=args.default_expert_format,
        default_linear_format=args.default_linear_format,
        token_embedding_format=args.token_embedding_format,
        output_format=args.output_format,
        device=args.device,
        exclude=tuple(args.exclude),
        imatrix=imatrix,
    )
    size = Path(args.out).stat().st_size / 1e9
    print(f"wrote {args.out} ({size:.2f} GB)")
    for fmt, n in sorted(counts.items()):
        print(f"  {fmt}: {n}")


if __name__ == "__main__":
    main()
