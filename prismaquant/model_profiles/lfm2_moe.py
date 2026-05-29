"""LFM2.5 MoE profile (Liquid AI's hybrid conv/attention MoE family).

Covers:
  - Lfm2MoeForCausalLM (LFM2.5-8B-A1B and siblings, model_type=lfm2_moe)

LFM2.5-8B-A1B is a 24-layer hybrid:

  - Every layer has a pre-mixer ``operator_norm`` (RMSNorm), a mixer, a
    pre-FFN ``ffn_norm`` (RMSNorm), and a feed-forward block.
  - The mixer is a short depthwise convolution (``conv``) on most layers
    and GQA attention (``self_attn``) on layers {2,6,10,14,18,21}
    (``config.layer_types``).
  - The FFN is a dense SwiGLU MLP on the first ``num_dense_layers`` (=2)
    layers {0,1} and a sparse 32-expert top-4 MoE on layers {2..23}.

Two naming/representation gaps PrismaQuant must bridge:

  | where                         | conv mixer Linears                              |
  |-------------------------------|-------------------------------------------------|
  | HF checkpoint + live module   | model.layers.X.conv.{in_proj,out_proj}          |
  | vLLM scheme-dispatch / runtime | model.layers.X.short_conv.{in_proj,out_proj}    |

  vLLM renames ``.conv.`` -> ``.short_conv.`` (its ``ShortConv`` wrapper
  carries an inner depthwise ``conv`` child, so the outer attribute is
  ``short_conv`` to avoid a LoRA regex collision). vLLM exposes this as a
  *substring* ``WeightsMapper`` (``orig_to_new_substr``), which the base
  class's prefix-only auto-derivation cannot see — so we apply the
  rewrite ourselves in ``to_vllm_internal_name`` (and declare it in the
  spec's ``recipe_to_vllm`` rules for graph/provenance parity).

  | where                         | MoE experts                                     |
  |-------------------------------|-------------------------------------------------|
  | HF safetensors checkpoint     | ...feed_forward.experts.E.w{1,2,3}.weight (2D)  |
  | live transformers module      | ...feed_forward.experts.{gate_up_proj,down_proj}|
  |                               |   (3D packed nn.Parameter, Lfm2MoeExperts)      |
  | vLLM runtime FusedMoE         | ...feed_forward.experts.{w13,w2}_weight (fused) |

  The PrismaQuant probe/cache run on the LIVE packed form
  (``gate_up_proj`` [E, 2*moe_inter, hidden], ``down_proj``
  [E, hidden, moe_inter]). The streaming loader bridges the on-disk
  per-expert layout into this packed form automatically at layer-load —
  driven by this profile's packed-experts spec (param names + projection
  splits + per-expert regex), via the generic ``_pack_per_expert_into_packed``
  in ``layer_streaming`` — so the raw HF checkpoint loads directly with no
  out-of-band pre-pack. Export SPLITS the packed params back into per-expert
  2D tensors. The on-disk leaf names must be ``w1``/``w3`` (from
  ``gate_up_proj``, in that order: ``gate, up = lin(x).chunk(2)``) and
  ``w2`` (from ``down_proj``), because vLLM's loader keys on
  ``make_expert_params_mapping(ckpt_gate_proj_name="w1",
  ckpt_down_proj_name="w2", ckpt_up_proj_name="w3")``.

Fused-sibling groups, the dense-FFN ``w13`` (= w1 + w3) fusion, and the
``in_proj`` self-map are read directly from vLLM's
``Lfm2MoeForCausalLM.packed_modules_mapping``
(``{qkv_proj: [q,k,v], w13: [w1,w3], in_proj: [in_proj]}``) — no
hand-coded patterns to drift.

Pins: the tied embedding (``embed_tokens`` / ``lm_head``), every RMSNorm
(``operator_norm``, ``ffn_norm``, ``q_layernorm``, ``k_layernorm``,
``embedding_norm``), the depthwise ``conv.conv`` Conv1d (stateful, not a
Linear), the tiny MoE router ``feed_forward.gate``, and the F32
``feed_forward.expert_bias`` routing-correction buffer
(``use_expert_bias=True``; vLLM loads it as
``gate.e_score_correction_bias``).

The short-conv mixer Linears ``conv.in_proj`` / ``conv.out_proj`` are also
pinned (BF16 passthrough). vLLM builds the mixer via
``ShortConv(... prefix=".conv")`` WITHOUT a ``quant_config``
(vllm/model_executor/models/lfm2_moe.py), so those Linears are always
unquantized at serving time — emitting compressed-tensors scales for them
makes the artifact fail to load (``KeyError: …short_conv.out_proj.
input_global_scale``). Pinning keeps the export aligned with what vLLM can
actually consume. They are a small share of params (the MoE experts hold
~93%), so the bpp cost is negligible.

Text-only family: no multimodal umbrella and no MTP head, so
``live_to_recipe_name`` / ``source_tensor_name`` are identity and
``has_mtp`` stays False.
"""
from __future__ import annotations

from .base import ModelProfile


class Lfm2MoeProfile(ModelProfile):

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {"lfm2_moe", "lfm2-moe"}:
            return True
        for arch in architectures:
            if arch.startswith("Lfm2Moe"):
                return True
        return False

    @property
    def name(self) -> str:
        return "lfm2_moe"

    def vllm_architecture_class(self) -> str | None:
        # `Lfm2MoeForCausalLM` is in the live vLLM registry and exposes:
        #   packed_modules_mapping = {qkv_proj:[q,k,v], w13:[w1,w3],
        #                             in_proj:[in_proj]}
        #   hf_to_vllm_mapper      = WeightsMapper(
        #                              orig_to_new_substr={".conv.":".short_conv."})
        # Fused-sibling promotion auto-derives from packed_modules_mapping.
        # The name remap is substring-based, so the base prefix-only
        # auto-derivation is empty — `to_vllm_internal_name` below applies
        # the conv -> short_conv substring rewrite explicitly.
        return "Lfm2MoeForCausalLM"

    # ------------------------------------------------------------
    # Naming remap: HF checkpoint/recipe -> vLLM scheme-dispatch
    # ------------------------------------------------------------
    def to_vllm_internal_name(self, checkpoint_name: str) -> str:
        """Rewrite ``.conv.`` -> ``.short_conv.`` so quant-scheme targets
        match vLLM's runtime ``ShortConv`` module names at dispatch.

        vLLM's ``hf_to_vllm_mapper`` is a *substring* mapper, which the
        base class (prefix-map only) cannot consume — so we apply it here.
        The base would otherwise return the conv Linears unchanged, and
        their compressed-tensors ``config_groups`` targets would never
        match vLLM's ``model.layers.X.short_conv.{in_proj,out_proj}``.

        We first defer to the declarative spec's ``recipe_to_vllm`` rules
        (which encode the same conv rewrite for graph/provenance parity),
        then fall back to the base prefix mapper (a no-op here)."""
        mapped = super().to_vllm_internal_name(checkpoint_name)
        if ".conv." in mapped:
            mapped = mapped.replace(".conv.", ".short_conv.")
        return mapped

    # `source_tensor_name`, `export_tensor_name`, and `live_to_recipe_name`
    # all stay identity (inherited): LFM2.5 is text-only with a flat
    # `model.layers.X.*` tree and no `language_model` infix. The export
    # path emits recipe-form keys (`...conv.in_proj`, `...experts.E.w1`);
    # vLLM's own `load_weights` applies the `.conv.` -> `.short_conv.` and
    # per-expert `w{1,2,3}` -> fused `w13`/`w2` remaps at load time.

    # `packed_expert_param_names`, `split_packed_experts_for_format`,
    # `packed_expert_projection_names`, `packed_expert_parent_for_projection`,
    # `per_expert_moe_regex`, `pinned_names`, and `fused_sibling_group`
    # all resolve from specs/lfm2_moe.json + the vLLM class via the base
    # implementations. The spec declares:
    #   - packed_experts.projection_splits: gate_up_proj -> [w1, w3],
    #                                        down_proj    -> [w2]
    #   - per_expert_regex matching ...experts.E.(w1|w2|w3)
    #   - pins for embed/lm_head/all norms/conv.conv/gate/expert_bias
    # No further overrides are required.

    # ------------------------------------------------------------
    # Always-resident head pieces specific to LFM2.5's naming.
    # ------------------------------------------------------------
    def head_resident_extra_prefixes(self, root) -> list[str]:
        """Keep LFM2.5's final norm (``embedding_norm``) and rotary
        (``pos_emb``) resident with the always-on head batch — both hang
        off the base model, reached as ``root.model`` (text-only, flat
        tree). The generic streaming loader pins ``norm``/``rotary_emb``;
        these architecture-specific names are contributed here rather than
        hardcoded in the loader. Gated on attribute presence, so this is a
        no-op for any model that lacks them."""
        base = getattr(root, "model", None)
        if base is None:
            return []
        return [f"model.{attr}." for attr in ("embedding_norm", "pos_emb")
                if hasattr(base, attr)]

    # ------------------------------------------------------------
    # MTP — LFM2.5 has no Multi-Token-Prediction head.
    # ------------------------------------------------------------
    def has_mtp(self) -> bool:
        return False
