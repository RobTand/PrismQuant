"""Qwen3.5 / Qwen3.6 MoE profile.

Covers:
  - Qwen3_5MoeForConditionalGeneration (multimodal, MoE)
  - Qwen3_5MoeForCausalLM (text-only MoE)
  - Qwen3_5MoeTextModel  (headless)

The canonical ``Qwen/Qwen3.6-35B-A3B`` checkpoint deliberately belongs to
this producer family: its outer config is ``qwen3_5_moe`` with architecture
``Qwen3_5MoeForConditionalGeneration``.  Its routed experts are already
packed as ``gate_up_proj`` / ``down_proj`` tensors (256 experts, top-8), while
the one shared expert per layer remains split as gate/up/down Linears.  Keep
the producer id ``qwen3_5``: that is the id declared by Gridbook's serving
contract, and inventing a release-name ``qwen3_6`` id would make an otherwise
supported checkpoint fail closed at the repository boundary.

The naming conventions PrismaQuant must juggle:

  | where                    | body                                         |
  |--------------------------|----------------------------------------------|
  | HF multimodal source     | model.language_model.layers.X.*              |
  | vLLM multimodal dispatch | language_model.model.layers.X.*              |
  | HF native-causal source  | model.layers.X.*                              |
  | vLLM native dispatch     | model.layers.X.*                              |
  | HF text-only / lm_head   | lm_head                                       |
  | vLLM wrapper lm_head     | language_model.lm_head                        |
  | MTP source               | mtp.layers.0.*   (mtp.fc, mtp.norm, ...)     |
  | vLLM MTP scheme-dispatch | mtp.layers.0.*   (IDENTITY — mtp. → model.   |
  |                          |                    remap only at weight-load)|

Visual encoder blocks pass through as BF16 (no real calibration yet).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .base import ModelProfile


class Qwen3_5Profile(ModelProfile):

    # Detection priority (lower = consulted first): the 3.5/3.6 MoE catch-all.
    priority = 110

    #: Exact entrypoints whose checkpoint and vLLM module trees are rooted at
    #: ``model.layers``.  Qwen3.8 is a release name; its official 2.4T proxy
    #: declares the Qwen3.5 implementation class, so do not invent a Qwen3.8
    #: class spelling from that proxy.  New spellings must be verified against
    #: the released config, safetensors index, and serving runtime before they
    #: are admitted here.
    NATIVE_CAUSAL_ARCHITECTURES = frozenset({
        "Qwen3_5MoeForCausalLM",
        "Qwen3_6MoeForCausalLM",
        "Qwen3.5MoeForCausalLM",
        "Qwen3.6MoeForCausalLM",
        "Qwen3_5MoeTextModel",
        "Qwen3_6MoeTextModel",
        "Qwen3.5MoeTextModel",
        "Qwen3.6MoeTextModel",
    })

    #: Exact multimodal wrapper entrypoints.  These keep the historical
    #: ``model.language_model`` source namespace and vLLM's
    #: ``language_model.model`` live namespace.
    WRAPPER_ARCHITECTURES = frozenset({
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_6MoeForConditionalGeneration",
        "Qwen3.5MoeForConditionalGeneration",
        "Qwen3.6MoeForConditionalGeneration",
    })

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            "qwen3_5",
            "qwen3_6_moe",
            "qwen3_6_moe_text",
            "qwen3_6",
        }:
            return True
        for arch in architectures:
            if arch.startswith("Qwen3_5") or arch.startswith("Qwen3.5") \
                    or arch.startswith("Qwen3_6") or arch.startswith("Qwen3.6"):
                return True
        return False

    @property
    def name(self) -> str:
        return "qwen3_5"

    def _declared_moe_layout(self) -> str:
        """Classify the checkpoint's declared MoE entrypoint exactly.

        A hand-built profile has no declaration and keeps the historical
        wrapper answer.  Once a real config has been declared, however, an
        absent, unknown, or mixed entrypoint is not evidence for either
        namespace and must not silently inherit the wrapper mapping.
        """
        archs = self.declared_architectures()
        if not archs:
            if self._declared_model_type is None:
                return "wrapper"
            raise RuntimeError(
                "Qwen3.5/Qwen3.8 MoE config declares no architecture; "
                "cannot choose native-causal versus wrapper naming"
            )

        native = [a for a in archs if a in self.NATIVE_CAUSAL_ARCHITECTURES]
        wrapper = [a for a in archs if a in self.WRAPPER_ARCHITECTURES]
        known = set(native) | set(wrapper)
        unknown = [a for a in archs if a not in known]
        if unknown:
            raise RuntimeError(
                "unsupported Qwen3.5/Qwen3.8 MoE architecture declaration "
                f"{unknown!r}; refusing to guess its checkpoint/vLLM namespace"
            )
        if native and wrapper:
            raise RuntimeError(
                "mixed native-causal and wrapper Qwen3.5/Qwen3.8 MoE "
                f"architectures {list(archs)!r}; namespace is ambiguous"
            )
        return "native_causal" if native else "wrapper"

    def vllm_architecture_class(self) -> str | None:
        """vLLM class to read `packed_modules_mapping` +
        `hf_to_vllm_mapper` from. The base class auto-derives
        `fused_sibling_group()` and the body-part of
        `to_vllm_internal_name()` from these two attributes. We only
        override `to_vllm_internal_name()` below to handle the MTP
        prefix specially."""
        if self._declared_moe_layout() == "native_causal":
            return "Qwen3_5MoeForCausalLM"
        return "Qwen3_5MoeForConditionalGeneration"

    def _checkpoint_source_layout(self) -> str:
        """Resolve direct-vs-wrapper source keys from the real index when set.

        ``stage_text_only`` rewrites a wrapper config to the causal entrypoint
        but symlinks its original ``model.language_model.*`` tensors. Thus the
        architecture declaration is authoritative for the live/vLLM tree but
        is not, by itself, authoritative for source lookup. Path-based profile
        detection supplies the index census; config-only profiles use the
        exact architecture declaration (the official native-release case).
        """
        serving_layout = self._declared_moe_layout()
        cached = getattr(self, "_qwen_checkpoint_source_layout", None)
        if cached is not None:
            return cached

        root = self._declared_model_path
        if root is None:
            return serving_layout
        index_path = root / "model.safetensors.index.json"
        if not index_path.is_file():
            # Small single-file checkpoints may legitimately have no index.
            # Their exact entrypoint remains the available declaration.
            return serving_layout
        try:
            import json

            payload = json.loads(index_path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"cannot inspect Qwen MoE source namespace in {index_path}: {exc}"
            ) from exc
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(
                f"Qwen MoE safetensors index {index_path} has no nonempty "
                "weight_map; refusing to guess its source namespace"
            )
        names = tuple(str(name) for name in weight_map)
        direct = any(name.startswith("model.layers.") for name in names)
        wrapper = any(
            name.startswith("model.language_model.layers.") for name in names
        )
        if direct == wrapper:
            detail = "both" if direct else "neither"
            raise RuntimeError(
                f"Qwen MoE safetensors index {index_path} contains {detail} "
                "direct model.layers.* and wrapper "
                "model.language_model.layers.* evidence; refusing to guess"
            )
        resolved = "native_causal" if direct else "wrapper"
        if serving_layout == "wrapper" and resolved != "wrapper":
            raise RuntimeError(
                f"Qwen MoE wrapper architecture has direct model.layers.* "
                f"source tensors in {index_path}; refusing the inconsistent "
                "config/index layout"
            )
        self._qwen_checkpoint_source_layout = resolved
        return resolved

    def source_tensor_name(self, model_qname: str) -> str:
        # A causal *serving* class may still point at a staged wrapper source.
        # Reuse the unspecialized spec's wrapper map in that one case; native
        # released checkpoints take the declarative causal variant below.
        if (self.name == "qwen3_5"
                and self._checkpoint_source_layout() == "wrapper"):
            from .structure import load_structure_spec

            spec = load_structure_spec("qwen3_5")
            if spec is None:
                raise RuntimeError("missing qwen3_5 model-structure spec")
            return spec.rewrite_recipe_to_source(model_qname)
        return super().source_tensor_name(model_qname)

    def structure_spec(self):
        # All spec-backed naming/grouping accessors pass through here.  Guard
        # them as well as vLLM metadata lookup so an unknown future release
        # cannot quietly consume the base wrapper map merely because its
        # model_type still matches this family.  The dense subclass has its
        # own entrypoint classifier and spec, so it is deliberately excluded.
        if self.name == "qwen3_5":
            self._declared_moe_layout()
        return super().structure_spec()

    def rms_norm_parameter_offset(self) -> float:
        """Qwen3.5/Qwen3.8 execute RMSNorm with ``1 + weight``."""
        return 1.0

    def prismasnap_moe_layer_contract(
        self,
        layer_index: int,
    ) -> dict[str, object]:
        """Closed MoE seam declaration for the Qwen3.5 producer family.

        Names are recipe names here and cross into the source checkpoint only
        through :meth:`source_tensor_name`.  That distinction is load-bearing:
        a native-causal serving config may point at a staged wrapper-namespaced
        source index.  The planner accepts either the packed 3-D expert pair or
        the declared per-expert alternative, never a mixed census.
        """
        if type(layer_index) is not int or layer_index < 0:
            raise ValueError("PrismaSnap MoE layer index must be non-negative")
        prefix = f"model.layers.{layer_index}"
        mlp = f"{prefix}.mlp"
        return {
            "schema": "prismaquant.prismasnap.moe_layer_profile.v1",
            "layer": layer_index,
            "input_norm": f"{prefix}.input_layernorm.weight",
            "post_attention_norm": f"{prefix}.post_attention_layernorm.weight",
            "mlp_prefix": mlp,
            # These roles are deliberately typed. `gate` produces E-way
            # routing logits while `shared_expert_gate` produces one sigmoid
            # scalar; tuple order must never decide which algebra applies.
            # Both are exact compensation terms and excluded from the codec
            # objective.
            "router": f"{mlp}.gate.weight",
            "packed_routed": {
                # Packed expert banks are direct Parameters on the experts
                # module, not nn.Linear children; released safetensors keys
                # therefore have no trailing `.weight`.
                "gate_up": f"{mlp}.experts.gate_up_proj",
                "down": f"{mlp}.experts.down_proj",
                "expert_axis": 0,
                "row_axis": 1,
                "input_axis": 2,
                "gate_rows": "first_half",
                "up_rows": "second_half",
            },
            "per_expert_routed": {
                "root": f"{mlp}.experts",
                "gate_projection": "gate_proj",
                "up_projection": "up_proj",
                "down_projection": "down_proj",
            },
            "shared_experts": (
                {
                    "output_gate": f"{mlp}.shared_expert_gate.weight",
                    "gate": f"{mlp}.shared_expert.gate_proj.weight",
                    "up": f"{mlp}.shared_expert.up_proj.weight",
                    "down": f"{mlp}.shared_expert.down_proj.weight",
                },
            ),
            "bias_policy": "reject_projection_biases",
        }

    # ------------------------------------------------------------
    # MTP
    # ------------------------------------------------------------
    def has_mtp(self) -> bool:
        return True

    def mtp_layer_count(self, cfg: dict) -> int:
        # Use base implementation first.
        n = super().mtp_layer_count(cfg)
        if n > 0:
            return n
        # Qwen3.6 uses `mtp_num_hidden_layers` on text_config. Covered
        # above. If still zero, scan safetensors as a last resort.
        return 0  # caller can scan safetensors separately if desired

    def build_mtp_module(self, text_config) -> nn.Module:
        """Return the Qwen3.5/3.6 MTP replica. See `MtpModule` below;
        `Qwen3_5DenseProfile` inherits this unchanged because
        `MtpModule` picks the dense-vs-MoE decoder class from the
        config at construction time."""
        return MtpModule(text_config)

    def mtp_objective_example(self) -> str:
        return ("CE(lm_head(MTP(embed_{t+1}, body_hidden_t)), ids_{t+2}) — "
                "the aux-loss Qwen3.5/3.6 MTP was trained under.")


# ---------------------------------------------------------------------------
# MTP module
#
# Transformers v5 ships no MTP module for these models (the top-level
# PreTrainedModel has `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`,
# so MTP weights are silently dropped on load). MTP is a vLLM-only runtime
# feature. To get real Fisher stats / cost measurements / export on MTP
# Linears we synthesize one here from HF primitives.
#
# Moved verbatim out of the former top-level `prismaquant/mtp_module.py`
# (deleted 2026-07-30, audit R12) so MTP construction goes through the
# profile like every other architecture-specific decision. The generic
# halves — reading `mtp.*` out of safetensors and loading them into the
# module, including the packed-expert fold — now live on `ModelProfile`
# as `read_mtp_source_state_dict()` / `load_mtp_state_dict()`.
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

    Satisfies `ModelProfile.build_mtp_module`'s naming contract: wrapped
    in a parent named `mtp`, its parameters come out as `mtp.fc.weight`,
    `mtp.layers.0.self_attn.q_proj.weight`, ... — the recipe names.

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
