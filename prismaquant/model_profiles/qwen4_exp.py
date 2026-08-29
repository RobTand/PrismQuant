"""Qwen4Exp (``model_type: qwen4_exp``) — Qwen3.8-Flash-Next family profile.

Scaffolding for the 177B checkpoint at ``/mnt/shared/models/Qwen3.8-Flash-Next``
(48 text layers, 512 routed experts + 1 shared expert, gated-DeltaNet linear
attention with a full-attention layer every 4th position, a QSA sparse-attention
indexer inside each full-attention layer, hyper-connections with ``hc_count=4``
residual streams, one PLE n-gram-embedding layer, an MTP sidecar, and a vision
tower).

Every naming claim below is traceable to one of two sources, cited inline:

  * ``transformers/models/qwen4_exp/modeling_qwen4_exp.py`` (transformers
    5.16.1, in ``/home/rob/dq-runs/venvs/prismaquant-tf516``) — the module and
    parameter names the live model exposes;
  * ``/mnt/shared/models/Qwen3.8-Flash-Next/model.safetensors.index.json`` —
    the on-disk source keys.

**There is no vLLM class for this architecture in this environment** (``import
vllm`` raises ``ModuleNotFoundError`` in the tf516 venv, and no vLLM release
ships a ``Qwen4Exp*`` model). Tier-1 auto-derivation via
``packed_modules_mapping``/``hf_to_vllm_mapper`` (``base.py`` ``_ensure_vllm_class``)
is therefore *impossible*, not merely unused: ``vllm_architecture_class()``
returns ``None`` and every structural fact is declared by hand in
``specs/qwen4_exp.json``. Consequently the spec deliberately declares **no**
``recipe_to_vllm`` naming map, no ``moe.per_expert_regex``, no
``default_serving_profile`` and no ``supported_lanes``: each of those is a
statement about what a serving runtime does, and CLAUDE.md principle 14 says
such a statement is attested or refused — never asserted from the producer
side. They are owed once a vLLM (or Gridbook) class exists to attest them.

This profile is **not yet registered** in ``registry.py`` (the scaffolding task
was scoped to new files only). Until the registration snippet in
``/home/rob/dq-runs/coordination/qwen4exp-structure-recon-2026-08-26.md`` is
applied, ``detect_profile()`` resolves this architecture to a ``SpecMatchProfile``
built from ``specs/qwen4_exp.json`` — which carries the naming/fusion/packing
declarations but *not* the Python-side walk rules or the RMSNorm offset below.
"""
from __future__ import annotations

from .base import ModelProfile


class Qwen4ExpProfile(ModelProfile):
    """Model profile for the ``qwen4_exp`` (Qwen3.8-Flash-Next) family."""

    # Detection priority (lower = consulted first). 200 sits after every
    # currently-registered profile (Laguna is 190); nothing else claims
    # `qwen4_exp`, so the ordering is only a tie-break placeholder. It MUST
    # equal `specs/qwen4_exp.json`'s `priority`
    # (tests/test_spec_match_profile.py asserts the two agree).
    priority = 200

    @property
    def name(self) -> str:
        return "qwen4_exp"

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        """Claim the multimodal wrapper, the text carve-out, and the inner
        text config.

        `config.json` of the 177B checkpoint declares
        ``model_type: "qwen4_exp"`` with ``architectures:
        ["Qwen4ExpForConditionalGeneration"]``; its ``text_config`` declares
        ``model_type: "qwen4_exp_text"``. A staged text-only checkpoint
        promotes the inner type (see `staging.promote_inner_model_type` in the
        spec), so both must be claimed.
        """
        mt = (model_type or "").lower()
        if mt in {"qwen4_exp", "qwen4_exp_text"}:
            return True
        return any(str(a).startswith("Qwen4Exp") for a in (architectures or []))

    # ------------------------------------------------------------
    # vLLM: absent by measurement, not by omission
    # ------------------------------------------------------------
    def vllm_architecture_class(self) -> str | None:
        """None — no vLLM class exists for ``Qwen4Exp*``.

        Returning a name here would make `base.py`'s auto-derivation import a
        class that does not exist and, worse, would encode an unattested claim
        about vLLM's fused/packed mapping. The fused-sibling groups and packed
        expert layout are declared manually in the spec instead, read from HF
        modelling code.
        """
        return None

    # ------------------------------------------------------------
    # Numerics
    # ------------------------------------------------------------
    def rms_norm_parameter_offset(self) -> float | None:
        """1.0 — this family executes ``gamma = 1 + weight``.

        `Qwen4ExpTextRMSNorm.__init__` initialises ``self.weight`` to *zeros*
        (modeling_qwen4_exp.py:162) and its forward computes
        ``output * (1.0 + self.weight.float())`` (modeling_qwen4_exp.py:177),
        exactly like Qwen3.5/3.6.

        Caveat for any future function-preserving transform: the *gated*
        variant used inside the DeltaNet block, `Qwen4ExpTextRMSNormGated`,
        is a different encoding — ones-init (modeling_qwen4_exp.py:187) and a
        plain ``self.weight * hidden_states`` (modeling_qwen4_exp.py:196), i.e.
        offset 0.0. This method returns the family-wide value for the plain
        norm; a transform that touches ``linear_attn.norm`` must special-case
        it rather than inherit this answer.
        """
        return 1.0

    # ------------------------------------------------------------
    # MTP
    # ------------------------------------------------------------
    def has_mtp(self) -> bool:
        """False for now — transformers ships no MTP module for this arch.

        The checkpoint *does* carry a full sidecar under ``mtp.`` (see
        `mtp_source_prefix`), but `Qwen4ExpForCausalLM` drops it on load
        (``_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]``,
        modeling_qwen4_exp.py:1535) and no reference forward exists in HF or
        vLLM. Qwen3.5's `build_mtp_module()` could be written because its MTP
        block is an ordinary decoder layer assembled from HF primitives; a
        qwen4_exp replica would have to re-implement the hyper-connection
        decoder, the QSA indexer and the fused-expert MoE block from scratch —
        that is a reimplementation, not scaffolding, and it would be unattested.

        TODO(qwen4_exp-mtp): flip to True and implement `build_mtp_module()`
        once a reference MTP forward exists (HF module or vLLM class). The
        sidecar's inventory is recorded in the recon memo; the generic,
        index-driven `read_mtp_source_state_dict()` inherited from base already
        works against the ``mtp.`` prefix.
        """
        return False

    def mtp_source_prefix(self) -> str | None:
        """``"mtp."`` — the sidecar's source-key namespace.

        Index keys: ``mtp.fc_embedding.weight``, ``mtp.fc_hidden.weight``,
        ``mtp.pre_fc_norm_embedding.weight``, ``mtp.pre_fc_norm_hidden.weight``,
        ``mtp.hyper_connection_mixer.*``, ``mtp.layers.0.*``.

        Note the divergence from Qwen3.5/3.6, which store a single ``mtp.fc``:
        this family has **two** projection Linears (``fc_embedding`` and
        ``fc_hidden``), declared in the spec's `mtp_extra_linear_names`.

        Declared even though `has_mtp()` is False, so the walk's MTP exclusion
        rule and the source-passthrough prefix both resolve correctly and the
        sidecar's bytes are never silently folded into the quantizable body.
        """
        return "mtp."

    # ------------------------------------------------------------
    # Discovery-walk claims
    # ------------------------------------------------------------
    def walk_claim_rules(self):
        """Two matmul-fed / non-GEMM families the base rules cannot claim.

        `base.walk_claim_rules()` rule 9 only decides ``nn.Linear`` weights;
        an unclaimed *matmul-fed* node fails the walk by design
        (`model_walk.py:894-912` — the ``wo_a`` failure class). qwen4_exp has
        one such node and one adjacent case:

        1. **The MoE router.** `Qwen4ExpTextTopKRouter` holds a bare
           ``nn.Parameter`` of shape ``[num_experts, hidden]``
           (modeling_qwen4_exp.py:905) and calls ``F.linear(hidden_states,
           self.weight)`` in its forward — matmul-fed, 2-D, but **not** an
           ``nn.Linear``, so no base rule matches it. Pinned, following the
           DSv4 precedent (`deepseek_v4.py:450-460`): a route flip is not a
           smooth cost, so no surrogate in this codebase can price it.
        2. **The DeltaNet / PLE short depthwise convolutions.** ``conv1d.weight``
           is 3-D (e.g. ``[10240, 1, 4]`` for ``linear_attn.conv1d``) and is
           consumed by a convolution rather than a GEMM, so it is neither
           claimed by rule 9 nor excluded by rule 8's ``max_ndim=1``. Pinned
           explicitly rather than left unclaimed, so its bytes are named on the
           immutable floor instead of being silently uncategorised.

        Everything else falls to the base rules: the packed 3-D expert
        parameters are handled by the spec's `packed_experts` declaration, the
        QSA indexer projection and the shared-expert sigmoid gate are ordinary
        Linears pinned via `pinned_names()` (base rule 2), the PLE n-gram
        embedding shards are ``nn.Embedding`` (base rule 5), and the DeltaNet
        ``A_log``/``dt_bias`` parameters are 1-D (base rule 8).
        """
        from prismaquant.model_walk import ClaimRule

        rules = [
            ClaimRule(
                "pin",
                "MoE router gate: a bare nn.Parameter fed to F.linear "
                "(Qwen4ExpTextTopKRouter), matmul-fed but never priced — a "
                "route flip is not a smooth cost; held at source precision",
                module_class="Qwen4ExpTextTopKRouter",
            ),
            ClaimRule(
                "decide",
                "packed MoE expert stack: a 3-D nn.Parameter sliced per "
                "expert and fed to F.linear (Qwen4ExpTextExperts) — "
                "matmul-fed, not an nn.Linear, and 97% of this "
                "architecture's quantizable bytes; the allocator's domain",
                module_class="Qwen4ExpTextExperts",
            ),
            ClaimRule(
                "pin",
                "short depthwise convolution kernel (gated-DeltaNet / PLE): "
                "consumed by a convolution, not a GEMM; held at source "
                "precision on the immutable floor",
                name_regex=r"(?:^|\.)conv1d\.weight$",
            ),
        ]
        return rules + super().walk_claim_rules()
