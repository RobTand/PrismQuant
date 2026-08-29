r"""GLM-5.3-Flash (``Glm5NextForConditionalGeneration``, model_type=glm5_next).

Scaffolding profile, built 2026-08-26 against the real checkpoint at
``/mnt/shared/models/GLM-5.3-Flash`` and transformers 5.16.1
(``/home/rob/dq-runs/venvs/prismaquant-tf516``). Every naming claim below is
cited to modeling code ``file:line`` or to a checkpoint index key; nothing is
carried over from a sibling architecture. The companion structure spec is
``specs/glm5_next.json``; the reconnaissance memo (full tensor inventory,
FP8/BF16 source map, open blockers, registration snippet) is
``/home/rob/dq-runs/coordination/glm5next-structure-recon-2026-08-26.md``.

Shape of the model (config.json ``text_config``)
------------------------------------------------
45 decoder layers + one body-indexed nextn/MTP block at index 45.

* ``layer_types``: 34 ``linear_attention`` (KDA) layers and 11
  ``deepseek_sparse_attention`` (MLA + indexer) layers at 3, 7, 11, ... 43.
  Layer 45 (MTP) is also MLA.
* ``mlp_layer_types``: layers 0-2 dense (``first_k_dense_replace: 3``),
  layers 3-44 (and 45) MoE with 288 routed experts top-8 plus one shared
  expert.
* MLA is NoPE (``qk_rope_head_dim: 0``, ``mla_use_nope: true``) with
  ``q_lora_rank 1536`` / ``kv_lora_rank 512``.
* Manifold-constrained hyper-connections (``mhc: true``, ``hc_mult: 4``) put
  two ``Glm5NextTextHyperConnection`` mixers on every body layer.

Per-layer-kind Linear inventory (modeling_glm5_next.py)
-------------------------------------------------------
KDA / ``Glm5NextTextLinearAttention`` (:584) — 9 Linears + 1 Conv1d:
  ``q_proj`` (:603), ``k_proj`` (:604), ``v_proj`` (:605),
  ``conv1d`` (:608, ``nn.Conv1d``, depthwise),
  ``forget_gate.f_a_proj``/``f_b_proj`` (:312-313),
  ``b_proj`` (:618), ``g_a_proj`` (:620), ``g_b_proj`` (:621),
  ``o_proj`` (:623).

MLA / ``Glm5NextTextAttention`` (:1064) — 5 Linears + a 3-Linear indexer:
  ``q_a_proj`` (:1098), ``q_b_proj`` (:1106),
  ``kv_a_proj_with_mqa`` (:1111), ``kv_b_proj`` (:1117), ``o_proj`` (:1123);
  ``indexer`` (:1131 -> :736) owns ``wq_b`` (:761), ``wk`` (:762),
  ``weights_proj`` (:764) plus two bare Parameters,
  ``index_kpool_compress_ape`` (:770) and ``index_kpool_compress_gate``
  (:771).

MLP — dense ``Glm5NextTextMLP`` (:86) ``gate_proj``/``up_proj``/``down_proj``
(:92-94); MoE ``Glm5NextTextMoE`` (:186) = ``experts``
(``Glm5NextTextExperts`` :108, 3-D ``gate_up_proj`` :116 / ``down_proj``
:117), ``gate`` (``Glm5NextTextTopkRouter`` :145, a bare
``nn.Parameter`` :151, not an ``nn.Linear``), and ``shared_experts``
(a ``Glm5NextTextMLP`` :196).

Three namespaces
----------------
* **source** — safetensors keys: ``model.language_model.layers.N.…``,
  ``model.visual.…``, ``lm_head.weight``.
* **live** — ``model.named_parameters()``. The only causal entrypoint is the
  multimodal wrapper ``Glm5NextForConditionalGeneration`` (:2063); there is
  **no** ``Glm5NextTextForCausalLM``, and ``Glm5NextModel.__init__`` (:1828)
  builds the vision tower unconditionally. So live names keep the
  ``model.language_model.`` infix in *both* staging modes, and
  ``checkpoint_to_live_name`` below deliberately does **not** strip it (the
  base implementation, base.py:1172, would).
* **recipe** — allocator keys: source spellings with the wrapper infix
  collapsed to ``model.`` (``live_to_recipe`` in the spec), matching the
  qwen3_5 convention.

Checkpoint -> live renames come from transformers' own conversion table
(``transformers/conversion_mapping.py``, ``"glm5_next"`` entry): the forget
gate gains an infix, and the hyper-connection parameters are re-spelled.
MLA and indexer projections are **not** renamed — unlike DeepSeek-V4, whose
entry maps ``wq_a`` -> ``q_a_proj``, glm5_next already ships live spellings.

FP8 source (see the spec's ``_verified_source_layout``)
-------------------------------------------------------
``quantization_config``: ``quant_method fp8``, ``fmt e4m3``,
``weight_block_size [128, 128]``, F32 ``.weight_scale_inv`` siblings — the
MiniMax/Qwen convention the **base** discovery path already handles. This
profile therefore does **not** override ``fp8_scale_pairs``; DSv4 only
overrides it because DSv4 ships 1-byte E8M0 ``.scale`` siblings instead.

The source dtype map is **per tensor, not per leaf name**: ``o_proj`` is
BF16 on the 34 KDA layers and FP8 on the 12 MLA layers (12 of 46
``o_proj.weight_scale_inv`` keys in the index). Since BF16 and FP8_SOURCE
are passthrough-only in this codebase, any leaf-keyed legality map would let
the allocator pick FP8_SOURCE for a BF16 KDA ``o_proj``. Keying on the
presence of the ``.weight_scale_inv`` sibling is the only correct rule here.

MTP / nextn
-----------
``num_nextn_predict_layers: 1``; the block is body-indexed at
``model.language_model.layers.45.`` and carries ``eh_proj``, ``enorm``,
``hnorm``, ``shared_head.norm`` plus a full MLA + MoE stack. transformers
5.16 refuses it outright (:1359 ``_keys_to_ignore_on_load_unexpected =
[r"layers\\.45\\.", r"layers\\.\\d+\\.shared_head\\."]``), so the skeleton
never instantiates it. This is the hy_v3 layout, not the ``mtp.*``-keyed
sidecar layout: ``has_mtp()`` stays False and the block ships VERBATIM via
``passthrough_prefixes``.

Quantizable vs pinned (principle 9 gate input)
-----------------------------------------------
The only vLLM serving path is PR #53906 (``ZJY0516/vllm@933876c``), and in
that model file **only routed experts, shared experts and dense MLPs are
constructed with a ``quant_config``**. MLA (``model.py:329
quant_config=None``), KDA (``kda.py:157-160`` nulls it for the whole
module), the indexer (``attention.py:251/:255``), the router
(``model.py:181 GateLinear`` takes no ``quant_config`` at all) and the
vision tower (``model.py:1032-1042``) are hardcoded unquantized. So the
quantizable set for artifact v1 is expert + shared-expert + dense-MLP
Linears; everything else is pinned or passthrough. Full per-class citations
live in ``specs/glm5_next.json``
``_verified_source_layout.serving_restriction``.

Mass split, measured over all 76,108 index keys and every shard header, in
the THREE buckets this profile creates (a ``.weight_scale_inv`` counts in
its weight's bucket):

===========================  ================  ==================
bucket                       params            source
===========================  ================  ==================
quantizable (BODY experts +
shared + dense MLP)          305,915,756,544   100% FP8
passthrough (layer 45
nextn/MTP + vision)            7,996,219,424   FP8 7.37e9
                                               BF16 0.62e9
pinned (everything else)       7,411,055,422   FP8 1.11e9
                                               BF16 6.30e9
total                        321,323,031,390
===========================  ================  ==================

95.21% of the parameter mass is quantizable. **Layer 45 is not**: it is a
full MoE block (288 routed + 1 shared expert, 7,272,923,136 params) that a
naive ``layers.\d+.mlp.`` regex would count as quantizable, but this
profile drops it in :meth:`checkpoint_to_live_name` and ships it verbatim at
FP8 source precision via ``passthrough_prefixes``. Whether the vLLM PR wires
``quant_config`` into ``Glm5NextMTP``'s experts is unattested (principle
14), so passthrough is the only defensible v1 line.

The immutable floor (pinned + passthrough at source precision) is
**20.80 GiB** = 12.77 pinned + 8.03 passthrough. It is a floor, not a budget
knob: 1.11e9 pinned and 7.37e9 passthrough params are FP8 on disk and
principle 11 forbids synthesizing BF16 from them. Whole-artifact weight
bytes ~= ``quantizable * bpp / 8 + 20.80 GiB`` (181.1 GiB at 4.50 bpp,
190.0 GiB at 4.75 bpp).

**Contradiction to resolve before export:** the PR's own comment at
``model.py:329`` reads *"MLA projections are BF16 in checkpoint"*. That is
false for this checkpoint — ``q_a_proj``, ``q_b_proj``,
``kv_a_proj_with_mqa`` and ``o_proj`` carry ``.weight_scale_inv`` on all 12
MLA layers and are ``F8_E4M3`` on disk; only ``kv_b_proj`` is BF16. Pinning
them therefore means FP8_SOURCE passthrough (~8.002 bpp), not BF16.

The KDA short convolution: a 3->1 concat, declared not hardcoded
-----------------------------------------------------------------
``self_attn.{q,k,v}_conv1d.weight`` -> ``self_attn.conv1d.weight`` is a 3->1
``Concatenate(dim=0)`` in the conversion table (34 layers x 3 = 102 source
keys, all BF16). ``checkpoint_to_live_name`` is a 1:1-or-drop contract and
cannot express a merge, so it passes the three source keys through unchanged
— exactly as it does for the per-expert MoE keys — and the merge itself is
declared in ``specs/glm5_next.json`` under ``concat_merges`` and executed by
the generic N->1 bridge ``layer_streaming._merge_concat_sources``, which
carries no architecture names. Order is load-bearing (q, k, v) and comes from
the conversion table; the bridge shape-checks the result against the live
parameter and refuses a dtype mix, so a wrong order or a missing source fails
loud rather than mis-packing. Left unbridged this was a hard blocker: the KDA
short convolution would have run on uninitialised weights.

Open, deliberately unimplemented (report, don't hack)
-----------------------------------------------------
1. ``vllm_architecture_class()`` returns None (DSv4/hy_v3 precedent) and
   ``fused_groups`` is empty. See the spec's ``open_todos.fused_groups``.
2. ``kv_b_proj`` is BF16 in the source while ``q_a``/``q_b``/``kv_a`` are
   FP8 — flagged in the memo as a serving-decision question, not decided
   here.
"""
from __future__ import annotations

import re

from .base import ModelProfile

# Body-indexed nextn/MTP block. transformers refuses these keys
# (modeling_glm5_next.py:1359), so the skeleton has no home for them.
_MTP_LAYER_RE = re.compile(r"^model\.language_model\.layers\.45\.")
_SHARED_HEAD_RE = re.compile(r"\.shared_head\.")

# 1:1 WeightRenaming rules, transformers/conversion_mapping.py "glm5_next".
_FORGET_GATE_RE = re.compile(
    r"\.self_attn\.(f_a_proj\.|f_b_proj\.|dt_bias|A_log)")
_HC_ATTN_RE = re.compile(r"\.hc_attn_(fn|base|scale)$")
_HC_FFN_RE = re.compile(r"\.hc_ffn_(fn|base|scale)$")

_VISUAL_PREFIX = "model.visual."


class Glm5NextProfile(ModelProfile):
    """Zhipu GLM-5.3-Flash / glm5_next family."""

    # Detection priority (lower = consulted first). 210 follows qwen4_exp
    # (200), which follows Laguna (190); disjoint from every other family's
    # match. It MUST equal `specs/glm5_next.json`'s `priority`
    # (tests/test_spec_match_profile.py asserts the two agree).
    priority = 210

    @classmethod
    def matches(cls, model_type: str | None,
                architectures: list[str] | None) -> bool:
        if model_type in ("glm5_next", "glm5_next_text"):
            return True
        return any(a.startswith("Glm5Next") for a in architectures or ())

    @property
    def name(self) -> str:
        return "glm5_next"

    def requires_multimodal_skeleton(self) -> bool:
        # transformers 5.16 registers no Glm5NextForCausalLM: neither
        # Glm5NextConfig nor Glm5NextTextConfig is in the CausalLM auto
        # mapping (verified against the real checkpoint 2026-08-26), so a
        # text-only skeleton is unresolvable. Every streaming construction
        # must instantiate the declared Glm5NextForConditionalGeneration
        # via the multimodal path; text-only staging would fail closed.
        return True

    # ------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------
    def vllm_architecture_class(self):
        # TODO(serving): unknown to this tree. No vLLM class for glm5_next
        # is importable from the pinned serving stacks, so nothing here can
        # be attested (principle 14) and every fused-sibling group / vLLM
        # internal name must come from the spec, not from an assumption.
        # DSv4 and hy_v3 both sit in exactly this state.
        return None

    def to_vllm_internal_name(self, checkpoint_name: str) -> str:
        # Attested from the pinned serving image
        # vllm-glm5next:pr53906-933876c, models/glm5next/nvidia/model.py
        # :988-993 (weight-prefix mapping inherited from
        # Glm4vForConditionalGeneration): ``model.visual.`` -> ``visual.``,
        # ``model.language_model.`` -> ``language_model.model.``,
        # ``lm_head.`` -> ``language_model.lm_head.``. compressed-tensors
        # scheme dispatch (find_matched_target / get_moe_method) compares
        # these INTERNAL names — recipe- or checkpoint-namespace targets
        # match nothing and every module silently falls through to
        # unquantized BF16 (2026-08-27: instantiated ~153G/rank and
        # OOM-killed both Sparks before the load could even fail).
        #
        # Accept both producer namespaces: allocator assignments arrive
        # recipe-spaced (``model.layers.X``), export-walk/bf16-passthrough
        # names arrive checkpoint-spaced (``model.language_model.X``).
        name = checkpoint_name
        if name.startswith("model.layers."):
            name = "model.language_model." + name[len("model."):]
        if name.startswith("model.language_model."):
            return "language_model.model." + name[len("model.language_model."):]
        if name.startswith(_VISUAL_PREFIX):
            return "visual." + name[len(_VISUAL_PREFIX):]
        if name == "lm_head" or name.startswith("lm_head."):
            return "language_model." + name
        if name.startswith("model."):
            # Recipe-space non-layer members (embed_tokens, norm) live
            # under ``model.`` and map like the body does.
            return "language_model." + name
        return name

    _RUNTIME_SOURCE_FP8_RE = re.compile(
        r"\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)$")

    def runtime_loads_source_fp8(self, module_name: str) -> bool:
        # Attested from the pinned serving image: the MLA (DSA-layer)
        # attention projections are constructed with quant_config=None
        # (model.py:329, "MLA projections are BF16 in checkpoint") and
        # their FP8 source bytes are dequantized to BF16 AT LOAD by
        # _try_load_fp8_attn_proj (model.py:1139-1180, _FP8_ATTN_PROJS),
        # keyed on SOURCE-style keys ``weight`` + ``weight_scale_inv``.
        # These modules are therefore not compressed-tensors-served: the
        # export must copy their source keys verbatim (scale name
        # included) and keep them OUT of config_groups. Suffix-matched so
        # it holds in live, recipe and checkpoint namespaces; KDA layers
        # have none of these modules.
        return bool(self._RUNTIME_SOURCE_FP8_RE.search(module_name))

    # ------------------------------------------------------------
    # MTP
    # ------------------------------------------------------------
    def has_mtp(self) -> bool:
        # Body-indexed at model.language_model.layers.45 rather than
        # mtp.*-keyed, so the mtp_source_prefix / build_mtp_module sidecar
        # machinery does not apply (hy_v3 route). checkpoint_to_live_name
        # drops the keys so probe/cost/allocator never see them, and the
        # compressed-tensors lane ships the block VERBATIM through
        # passthrough_prefixes for a future spec-decode serve.
        #
        # `mtp_source_prefix()` is intentionally left at the base default
        # ("mtp."), which matches no key in this checkpoint. Returning ""
        # would make every `startswith(prefix)` test in the tree true.
        return False

    # ------------------------------------------------------------
    # Streamed forward: manifold-constrained hyper-connections (mHC)
    # ------------------------------------------------------------
    #
    # glm5_next runs its decoder stack on `hc_mult` PARALLEL residual
    # streams, not one. `Glm5NextTextModel.forward` expands the embedding
    # to `[B, T, hc_mult, H]` before the loop and collapses it with
    # `hc_head` (an unweighted mean over the stream axis) before the final
    # norm + lm_head. Both are mirrored below from that forward, so the
    # streamed pass reproduces it rather than restating it. Without these
    # the loop feeds `[B, T, H]` into `Glm5NextTextHyperConnection`, whose
    # first op is `hidden_streams.flatten(start_dim=2)` — a silent shape
    # error at layer 0.
    #
    # Source: transformers 5.16.1
    # `models/glm5_next/modeling_glm5_next.py`
    #   :1477  hidden_states = inputs_embeds.unsqueeze(2)
    #                              .expand(-1, -1, self.config.hc_mult, -1)
    #                              .contiguous()
    #   :1493  hidden_states = self.norm(self.hc_head(hidden_states))
    #   Glm5NextTextHyperHead.forward -> hidden_streams.mean(dim=2)

    def expand_hidden_for_layers(self, hidden, base_model):
        """Expand `[B, T, H]` to the `hc_mult` residual streams the mHC
        blocks consume. Read off the live config, never a constant."""
        if hidden.ndim != 3:
            raise RuntimeError(
                "glm5_next streamed forward expects a rank-3 hidden state "
                f"before stream expansion; got shape {tuple(hidden.shape)}"
            )
        config = getattr(base_model, "config", None)
        hc_mult = getattr(config, "hc_mult", None)
        if hc_mult is None:
            raise RuntimeError(
                "glm5_next streamed forward cannot expand residual streams: "
                "the live config declares no hc_mult"
            )
        return hidden.unsqueeze(2).expand(-1, -1, int(hc_mult), -1).contiguous()

    def collapse_hidden_after_layers(self, hidden, base_model):
        """Collapse the mHC streams with the model's own `hc_head`.

        Resolved from the module, not re-derived: `hc_head` is an
        unweighted mean today, and a checkpoint that ever ships a weighted
        one must not silently keep the mean.
        """
        head = getattr(base_model, "hc_head", None)
        if head is None:
            raise RuntimeError(
                "glm5_next streamed forward cannot collapse residual "
                "streams: the live model exposes no hc_head"
            )
        if hidden.ndim != 4:
            raise RuntimeError(
                "glm5_next streamed forward expects a rank-4 hidden state "
                f"after the layer loop; got shape {tuple(hidden.shape)}"
            )
        return head(hidden)

    # ------------------------------------------------------------
    # Probe
    # ------------------------------------------------------------
    def probe_linear_exclude_extra(self) -> str:
        # Every Linear outside the serving contract's quantizable set. The
        # exporter ships their source bytes on the immutable floor, so the
        # probe must not inventory them: on a source-dtype-masked menu they
        # would carry zero legal candidates and trip the allocator's
        # coverage refusal.
        #
        # The set is dictated by vLLM PR #53906 (ZJY0516/vllm@933876c),
        # which builds ONLY routed experts, shared experts and dense MLPs
        # with a quant_config — see specs/glm5_next.json
        # `_verified_source_layout.serving_restriction` for the per-class
        # code citations. This is principle 9's measured-platform-fact
        # carve-out, not a taste-based format ban.
        return (
            r"(?:self_attn\.(?:q_proj|k_proj|v_proj|b_proj"
            r"|f_a_proj|f_b_proj|g_a_proj|g_b_proj|o_proj"
            r"|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj)"
            r"|self_attn\.indexer\."
            r"|mlp\.gate\."
            r"|eh_proj"
            # The vision tower: hardcoded BF16 at model.py:1032-1042 and
            # excluded from bpp accounting by principle 12. It ships
            # verbatim via passthrough_prefixes. Note that
            # `ModelGraph.quantizable` does NOT consult
            # passthrough_prefixes, so without this the probe would
            # inventory 124 visual Linears (see the recon memo, "caveats").
            r"|^model\.visual\.)"
        )

    # ------------------------------------------------------------
    # Discovery walk
    # ------------------------------------------------------------
    def walk_claim_rules(self):
        from prismaquant.model_walk import ClaimRule

        rules = [
            ClaimRule(
                "pin",
                "attention projection: the pinned serving runtime (vLLM PR "
                "#53906) constructs every KDA, MLA and indexer Linear with "
                "quant_config=None, so no quantized bytes here have a "
                "native route; held at source precision",
                name_regex=(
                    r"self_attn\.(?:q_proj|k_proj|v_proj|b_proj"
                    r"|f_a_proj|f_b_proj|g_a_proj|g_b_proj|o_proj"
                    r"|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj"
                    r"|indexer)"
                ),
            ),
            ClaimRule(
                "exclude",
                "KDA depthwise short convolution (nn.Conv1d, kernel 4): a "
                "convolution filter, never a GEMM multiplicand; the "
                "exporter ships its source bytes",
                name_regex=r"self_attn\.conv1d\.weight$",
            ),
            ClaimRule(
                "exclude",
                "vision tower: outside the text graph this artifact "
                "serves; shipped verbatim via passthrough_prefixes",
                name_regex=r"^model\.visual\.",
            ),
        ]
        # Base rules then pin the three probe-skipped module classes
        # declared in specs/glm5_next.json (Glm5NextTextTopkRouter's bare
        # `weight` :151, Glm5NextTextHyperConnection's `fn` :259, and the
        # indexer's `index_kpool_compress_*` :770-771) — each is a 2-D bare
        # nn.Parameter fed to F.linear (:160, :278, :800) that neither the
        # `max_ndim=1` exclude nor the `module_class="Linear"` decide rule
        # would otherwise claim.
        rules.extend(super().walk_claim_rules())
        return rules

    # ------------------------------------------------------------
    # Naming
    # ------------------------------------------------------------
    def checkpoint_to_live_name(self, ckpt_key: str, *,
                                multimodal: bool = False) -> str | None:
        if ckpt_key.endswith(".weight_scale_inv"):
            # Consumed by the FP8 scale map (base fp8_scale_pairs discovery).
            return None
        if _MTP_LAYER_RE.match(ckpt_key) or _SHARED_HEAD_RE.search(ckpt_key):
            return None
        if ckpt_key.startswith(_VISUAL_PREFIX):
            return ckpt_key if multimodal else None
        # NOTE: `self_attn.{q,k,v}_conv1d.weight` are deliberately NOT
        # dropped. They are the three sources of a 3->1 Concatenate(dim=0)
        # into the live `self_attn.conv1d.weight`; this map is 1:1-or-drop, so
        # the merge lives in the spec's `concat_merges` block and runs in
        # `layer_streaming._merge_concat_sources`, which needs the source keys
        # to survive into the loader's tensor dict (the per-expert MoE keys
        # ride the same convention).
        name = _FORGET_GATE_RE.sub(r".self_attn.forget_gate.\1", ckpt_key)
        name = _HC_ATTN_RE.sub(r".attn_hc.\1", name)
        name = _HC_FFN_RE.sub(r".ffn_hc.\1", name)
        # NOTE: no `model.language_model.` strip. The live module tree is
        # the multimodal wrapper's in both staging modes (see docstring),
        # so the infix is part of the live name; the recipe namespace
        # collapses it via the spec's live_to_recipe rules instead.
        return name

    # ------------------------------------------------------------
    # Source keys this profile knowingly cannot bridge (none)
    # ------------------------------------------------------------
    def unbridged_source_keys(self) -> tuple[str, ...]:
        """Source-key patterns dropped by `checkpoint_to_live_name` that a
        live forward genuinely NEEDS.

        Not part of the `ModelProfile` contract — read by the smoke script so
        any such gap is re-stated on every run instead of decaying into a
        comment. **Empty**: the one gap this profile ever had, the 3->1
        `{q,k,v}_conv1d` -> `conv1d` concat, is now declared in
        `specs/glm5_next.json` `concat_merges` and executed by
        `layer_streaming._merge_concat_sources`. Everything else
        `checkpoint_to_live_name` drops (`.weight_scale_inv` siblings, the
        layer-45 nextn block, the vision tower under text-only staging) is
        consumed by another path or shipped verbatim.
        """
        return ()
