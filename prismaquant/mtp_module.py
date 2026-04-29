#!/usr/bin/env python3
"""mtp_module.py — reusable MTP (multi-token prediction) module helpers
for Qwen3.5/3.6 MoE.

Transformers v5 ships no MTP module for these models (the top-level
PreTrainedModel has `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`,
so MTP weights are silently dropped on load). MTP is a vLLM-only runtime
feature. To get real Fisher stats / cost measurements / export on MTP
Linears we synthesize one here from HF primitives.

This file holds only the reusable building blocks that several
PrismaQuant stages (incremental probe, incremental cost, export) share:

  - `MtpModule`           — the single-layer MTP decoder, mirroring
                            vLLM's `Qwen3_5MultiTokenPredictor.forward`.
  - `_build_single_layer_config`
                          — produce a 1-layer text config for MtpModule.
  - `_load_mtp_state_dict`
                          — pull `mtp.*` tensors out of safetensors.
  - `_load_into_mtp`      — map checkpoint keys onto MtpModule layout,
                            including packed-expert gate_up/down_proj.

The MTP probe/cost orchestration used to live in separate
`mtp_probe.py` / `mtp_cost.py` entrypoints; those have been folded into
`incremental_probe.py` / `incremental_measure_quant_cost.py` as a
built-in shard, so this file intentionally contains no drivers or
argparse.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import safe_open


# ---------------------------------------------------------------------------
# MTP module
# ---------------------------------------------------------------------------

def _build_single_layer_config(text_config):
    """Return a `Qwen3_5MoeTextConfig` (or compatible) with exactly one
    decoder layer of type 'full_attention'. This matches vLLM's MTP:
    one full-attention decoder block per MTP step.

    `copy.deepcopy` is used so the body's config is untouched and
    gradient checkpointing state on the original model doesn't leak."""
    cfg = copy.deepcopy(text_config)
    cfg.layer_types = ["full_attention"]
    cfg.num_hidden_layers = 1
    return cfg


class MtpModule(nn.Module):
    """Mirrors `vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MultiTokenPredictor`
    but built on HF primitives so Fisher hooks and autograd work normally.

    Dense vs MoE is selected from the config at construction time:
    `Qwen3_5MoeDecoderLayer.__init__` eagerly reads `num_experts_per_tok`,
    which dense configs don't define, so we must route to
    `Qwen3_5DecoderLayer` for Qwen3.5/3.6 dense checkpoints."""

    def __init__(self, text_config):
        super().__init__()
        mtp_cfg = _build_single_layer_config(text_config)
        hidden = mtp_cfg.hidden_size
        eps = mtp_cfg.rms_norm_eps

        is_moe = (
            getattr(mtp_cfg, "num_experts", 0)
            or getattr(mtp_cfg, "num_local_experts", 0)
            or getattr(mtp_cfg, "num_experts_per_tok", 0)
        )
        if is_moe:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeDecoderLayer as _DecoderLayer,
                Qwen3_5MoeRMSNorm as _RMSNorm,
            )
        else:
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5DecoderLayer as _DecoderLayer,
                Qwen3_5RMSNorm as _RMSNorm,
            )

        self.fc = nn.Linear(hidden * 2, hidden, bias=False)
        self.layers = nn.ModuleList([_DecoderLayer(mtp_cfg, layer_idx=0)])
        self.norm = _RMSNorm(hidden, eps=eps)
        self.pre_fc_norm_hidden = _RMSNorm(hidden, eps=eps)
        self.pre_fc_norm_embedding = _RMSNorm(hidden, eps=eps)

    def forward(self,
                inputs_embeds: torch.Tensor,
                body_hidden_states: torch.Tensor,
                position_embeddings,
                causal_mask,
                position_ids):
        e = self.pre_fc_norm_embedding(inputs_embeds)
        h = self.pre_fc_norm_hidden(body_hidden_states)
        h = torch.cat([e, h], dim=-1)
        h = self.fc(h)
        h = self.layers[0](
            hidden_states=h,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )
        if isinstance(h, tuple):
            h = h[0]
        h = self.norm(h)
        return h


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_mtp_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Return every tensor whose key starts with `mtp.`, stripped of
    that prefix so it matches `MtpModule`'s module layout. We do not
    materialize all shards; only the ones that actually hold MTP keys."""
    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    if not idx_path.exists():
        raise RuntimeError(f"no safetensors index at {idx_path}")
    with open(idx_path) as f:
        wm = json.load(f)["weight_map"]
    mtp_files = sorted({wm[k] for k in wm if k.startswith("mtp.")})
    if not mtp_files:
        # Some Qwen3.5/3.6 finetunes inherit num_nextn_predict_layers
        # from the base config but strip the MTP weights. Return empty
        # rather than raising — the runner detects empty and writes an
        # empty shard pickle. See PR #1.
        print("[mtp] no mtp.* weights in safetensors index; "
              "returning empty state dict", flush=True)
        return {}
    out: dict[str, torch.Tensor] = {}
    for fn in mtp_files:
        with safe_open(str(src / fn), framework="pt") as sf:
            for key in sf.keys():
                if not key.startswith("mtp."):
                    continue
                t = sf.get_tensor(key)
                out[key[len("mtp."):]] = t
    return out


def _load_into_mtp(mtp: MtpModule, raw: dict[str, torch.Tensor]):
    """Map checkpoint keys (mtp.* stripped) onto MtpModule's layout.

    Checkpoint layout            -> MtpModule path
      fc.weight                  -> fc.weight
      pre_fc_norm_embedding...   -> pre_fc_norm_embedding.weight
      pre_fc_norm_hidden...      -> pre_fc_norm_hidden.weight
      norm.weight                -> norm.weight
      layers.0.<rest>            -> layers.0.<rest>

    The HF `Qwen3_5MoeDecoderLayer` stores packed experts as 3D
    `mlp.experts.gate_up_proj` / `down_proj`, matching the checkpoint."""
    sd = mtp.state_dict()
    params = dict(mtp.named_parameters())
    mapped: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    loaded_module_keys: set[str] = set()
    packed_pat = re.compile(
        r"^(layers\.\d+\.mlp\.experts)\.(\d+)\."
        r"(gate_proj|up_proj|down_proj)\.weight$"
    )

    for k, v in raw.items():
        if k in sd:
            mapped[k] = v
            loaded_module_keys.add(k)
            continue

        m = packed_pat.match(k)
        if m is None:
            missing.append(k)
            continue

        prefix, expert_id_s, proj = m.groups()
        expert_id = int(expert_id_s)
        if proj == "down_proj":
            packed_name = f"{prefix}.down_proj"
            packed = params.get(packed_name)
            if packed is None:
                missing.append(k)
                continue
            packed.data[expert_id].copy_(v.to(device=packed.device, dtype=packed.dtype))
            loaded_module_keys.add(packed_name)
            continue

        packed_name = f"{prefix}.gate_up_proj"
        packed = params.get(packed_name)
        if packed is None:
            missing.append(k)
            continue
        rows = v.shape[0]
        start = 0 if proj == "gate_proj" else rows
        packed.data[expert_id, start:start + rows].copy_(
            v.to(device=packed.device, dtype=packed.dtype)
        )
        loaded_module_keys.add(packed_name)

    # Load exact-name tensors through state_dict for everything that isn't
    # a packed expert tensor we filled manually above.
    mtp.load_state_dict(mapped, strict=False)
    extra = [k for k in sd if k not in loaded_module_keys]
    return missing, extra
