"""Architecture profile — PrismaQuant's adapter layer between a model
family's checkpoint conventions and the format-agnostic core pipeline.

Each profile captures three kinds of knowledge:

  1. **Naming**: how checkpoint parameter names map to vLLM's internal
     Linear qnames at compressed-tensors scheme dispatch, and the regex
     patterns vLLM uses for per-expert MoE loading.

  2. **Structure**: which Linear groups are fused siblings (q/k/v,
     gate/up, etc.), what 3D Parameters represent packed MoE experts,
     whether the architecture has MTP heads.

  3. **MTP construction**: how to stand up an HF-module replica of the
     architecture's MTP forward (for Fisher probing), and how to load
     `mtp.*` safetensors into it.

Profiles are picked per-run by `registry.detect_profile(model_path)`
from HF config + architectures. Unknown architectures fall back to
`DefaultProfile` which runs the generic path (common fused-sibling
groups, no MTP support, plain `model.layers.*` naming).
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn


class ModelProfile(ABC):
    """Base class for all PrismaQuant architecture profiles.

    Where possible, default implementations auto-derive their return
    values from the vLLM model class registered for this architecture
    (`vllm_architecture_class()`). That way, adding a new architecture
    typically only requires `matches()`, `vllm_architecture_class()`,
    and an optional `build_mtp_module()` — the rest comes from vLLM's
    `packed_modules_mapping` and `hf_to_vllm_mapper` class attributes.
    """

    def __init__(self) -> None:
        # Lazy-compiled derivations from the vLLM class. Computed on
        # first access so profile construction stays cheap.
        self._vllm_cls = None
        self._vllm_cls_loaded = False
        self._fused_matcher = None
        self._name_remapper = None
        self._structure_spec = None
        self._structure_spec_loaded = False

    # ------------------------------------------------------------
    # Identity + match
    # ------------------------------------------------------------
    @classmethod
    @abstractmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        """Return True if this profile claims responsibility for the
        given HF `model_type` / `architectures`."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Profile identifier (e.g. 'qwen3_5', 'default')."""

    def vllm_architecture_class(self) -> str | None:
        """Return the HF `architectures[0]` string whose vLLM class
        PrismaQuant should read `packed_modules_mapping` and
        `hf_to_vllm_mapper` from. Profiles that don't have a vLLM
        counterpart (dev-only architectures) can return None and
        override the dependent methods manually."""
        return None

    def _ensure_vllm_class(self):
        if self._vllm_cls_loaded:
            return
        self._vllm_cls_loaded = True
        arch = self.vllm_architecture_class()
        if arch is None:
            return
        from .vllm_registry import vllm_class_for_architecture
        self._vllm_cls = vllm_class_for_architecture(arch)

    # ------------------------------------------------------------
    # Fused-sibling promotion (allocator.py)
    # ------------------------------------------------------------
    def fused_sibling_group(self, linear_qname: str) -> str | None:
        """Return a canonical 'group key' if this Linear belongs to a
        fused-sibling group (q/k/v/o, gate/up, etc.), otherwise None.

        Default implementation derives sibling groups from the vLLM
        class's `packed_modules_mapping` attribute. Profiles can
        override to add arch-specific groups vLLM doesn't know about,
        or to bypass the vLLM lookup entirely.

        Example (Qwen3.5 via vLLM's `Qwen3_5MoeForConditionalGeneration`):
          model.layers.3.self_attn.q_proj -> 'model.layers.3.self_attn.qkv_proj'
          model.layers.3.self_attn.k_proj -> 'model.layers.3.self_attn.qkv_proj'
          model.layers.3.mlp.gate_proj    -> 'model.layers.3.mlp.gate_up_proj'
        """
        if self._fused_matcher is None:
            self._ensure_vllm_class()
            from .vllm_registry import (
                fused_sibling_matcher_from_packed_mapping,
                packed_modules_mapping_from_class,
            )
            pm = packed_modules_mapping_from_class(self._vllm_cls)
            if not pm:
                spec = self.structure_spec()
                if spec is not None and spec.fused_groups:
                    self._fused_matcher = spec.fused_group_for
                else:
                    self._fused_matcher = lambda _qname: None
            else:
                self._fused_matcher = fused_sibling_matcher_from_packed_mapping(pm)
        return self._fused_matcher(linear_qname)

    def fused_sibling_leaf_mapping(self) -> dict[str, tuple[str, ...]]:
        """Return fused-module leaf names to their member leaf names.

        This is the structured form of ``fused_sibling_group`` for call sites
        that need to resolve sidecar artifacts such as h-detail row weights.
        Prefer vLLM metadata when available, then the declarative
        model-structure spec.
        """
        try:
            self._ensure_vllm_class()
            from .vllm_registry import packed_modules_mapping_from_class

            mapping = packed_modules_mapping_from_class(self._vllm_cls)
            if mapping:
                return {
                    str(fused): tuple(str(member) for member in members)
                    for fused, members in mapping.items()
                }
        except Exception:
            pass

        spec = self.structure_spec()
        if spec is None:
            return {}
        out: dict[str, tuple[str, ...]] = {}
        for group in getattr(spec, "fused_groups", ()):
            target_suffix = str(group.target_suffix)
            if "." not in target_suffix:
                continue
            target_parent, target_leaf = target_suffix.rsplit(".", 1)
            members: list[str] = []
            valid = True
            for member in group.member_suffixes:
                member = str(member)
                if "." not in member:
                    valid = False
                    break
                member_parent, member_leaf = member.rsplit(".", 1)
                if member_parent != target_parent:
                    valid = False
                    break
                members.append(member_leaf)
            if valid and members:
                out[target_leaf] = tuple(members)
        return out

    # ------------------------------------------------------------
    # MoE packing
    # ------------------------------------------------------------
    def packed_expert_param_names(self) -> frozenset[str]:
        """Parameter attribute names (on a `*Experts` module) that hold
        3D packed MoE weight tensors. Union across all known architectures
        is a safe default; specific profiles can narrow."""
        spec = self.structure_spec()
        if spec is not None and spec.packed_experts.declared:
            return frozenset(spec.packed_experts.param_names)
        return frozenset({
            "gate_up_proj", "down_proj",   # Qwen3.5 / 3.6
            "w1", "w2", "w3",              # Mixtral
            "gate_proj", "up_proj",        # some HF layouts
        })

    def packed_expert_module_class_names(self) -> frozenset[str]:
        """Legacy packed-expert container class names accepted by this profile.

        Most current architectures expose profile-declared 3D parameters and
        never need class-name fallback. Specs can list older wrapper classes
        when parameter discovery needs a second hint.
        """
        spec = self.structure_spec()
        if spec is not None and spec.packed_experts.declared:
            return frozenset(spec.packed_experts.module_class_names)
        return frozenset()

    def pinned_names(self) -> tuple[str, ...]:
        """Recipe/module names that must remain unquantized for this profile."""
        spec = self.structure_spec()
        if spec is not None:
            return tuple(spec.pinned_names)
        return ("lm_head",)

    def is_pinned_name(self, qname: str) -> bool:
        """Return True when ``qname`` is covered by this profile's pins."""
        name = str(qname)
        module_name = name[:-7] if name.endswith(".weight") else name
        for pinned in self.pinned_names():
            pin = str(pinned)
            pin_module = pin[:-7] if pin.endswith(".weight") else pin
            if module_name == pin_module or module_name.endswith("." + pin_module):
                return True
            if name == pin or name.endswith("." + pin):
                return True
        return False

    def fast_kernel_requirements(self) -> tuple[tuple[str, str], ...]:
        """Required Python modules for production-speed forwards.

        Returns ``(module_name, install/display_name)`` pairs. Profiles use
        this for fail-fast guards around architecture-specific optimized
        kernels without making callers parse model names.
        """
        spec = self.structure_spec()
        if spec is None:
            return ()
        return tuple(
            (req.module, req.package)
            for req in spec.fast_kernel_requirements
        )

    def per_expert_moe_regex(self) -> str | None:
        """Regex matching vLLM's per-expert Linear qnames at scheme
        dispatch time. Added to the config_groups catch-all so every
        per-expert per-projection tensor picks up the catch-all format
        without ~30k explicit targets."""
        spec = self.structure_spec()
        if spec is not None and spec.per_expert_moe_regex:
            return spec.per_expert_moe_regex
        return None

    # ------------------------------------------------------------
    # MTP
    # ------------------------------------------------------------
    def has_mtp(self) -> bool:
        """True if this architecture has Multi-Token-Prediction heads
        in its checkpoint (`mtp.*` tensors) that PrismaQuant can probe
        and quantize."""
        return False

    def build_mtp_module(self, text_config) -> nn.Module | None:
        """Construct an HF-module replica of the MTP forward (mirrors
        what vLLM's MTP class does at inference time). Return None if
        `has_mtp()` is False.

        The returned module must be wrappable — after `load_state_dict`
        with the stripped-prefix MTP weights it should forward a hidden
        state + next-token embed into the MTP block exactly as vLLM does."""
        return None

    def load_mtp_state_dict(self, mtp_module: nn.Module,
                            raw: dict) -> tuple[list[str], list[str]]:
        """Load raw `mtp.*` tensors (with `mtp.` stripped) into
        `mtp_module`. Return `(unmatched_keys, module_params_without_weight)`.

        Default implementation uses `mtp_module.load_state_dict(raw, strict=False)`."""
        mapped: dict = {}
        for k, v in raw.items():
            mapped[k] = v
        sd = mtp_module.state_dict()
        mapped_filtered = {k: v for k, v in mapped.items() if k in sd}
        missing = [k for k in mapped if k not in sd]
        extra = [k for k in sd if k not in mapped_filtered]
        mtp_module.load_state_dict(mapped_filtered, strict=False)
        return missing, extra

    def mtp_objective_example(self) -> str:
        """One-line description of the MTP training objective for the
        probe's metadata. Generic fallback is fine for most architectures."""
        return "MTP auxiliary loss (predict token t+k given hidden_t)"

    def per_expert_mtp_regex(self) -> str | None:
        """Regex matching MTP per-expert Linear qnames at scheme dispatch.
        Returns None if no MoE MTP in this architecture."""
        spec = self.structure_spec()
        if spec is not None and spec.per_expert_mtp_regex:
            return spec.per_expert_mtp_regex
        return None

    # ------------------------------------------------------------
    # Naming remap for compressed-tensors
    # ------------------------------------------------------------
    def to_vllm_internal_name(self, checkpoint_name: str) -> str:
        """Remap a checkpoint parameter name (as stored in safetensors)
        to the vLLM-internal module qname that `find_matched_target`
        compares against at scheme dispatch.

        Default implementation uses the vLLM class's `hf_to_vllm_mapper`
        (specifically its `orig_to_new_prefix` dict). Matches vLLM's
        own weight-loader remap, so the allocator's config_groups
        targets and the runtime scheme-dispatch names stay in sync
        without PrismaQuant duplicating the mapping.

        Profiles override when: (a) there's no vLLM class for this
        arch, (b) the vLLM mapper is regex/substring-based (we only
        consume the prefix form), or (c) there are arch-specific
        quirks like MTP that need special handling beyond the simple
        prefix rewrite."""
        if self._name_remapper is None:
            self._ensure_vllm_class()
            from .vllm_registry import (
                hf_to_vllm_prefix_map_from_class,
                name_remapper_from_prefix_map,
            )
            prefix = hf_to_vllm_prefix_map_from_class(self._vllm_cls)
            self._name_remapper = name_remapper_from_prefix_map(prefix)
        spec = self.structure_spec()
        if spec is not None and spec.recipe_to_vllm:
            mapped = spec.rewrite_recipe_to_vllm(checkpoint_name)
            if mapped != checkpoint_name:
                return mapped
        return self._name_remapper(checkpoint_name)

    def source_tensor_name(self, model_qname: str) -> str:
        """Rewrite an in-memory HF module qname (from `named_parameters`)
        to the name that should land on disk in the exported
        safetensors. For multimodal HF checkpoints loaded via
        AutoModelForCausalLM, the module tree is flat (`model.layers.X.*`)
        but the source safetensors use the multimodal convention
        (`model.language_model.layers.X.*`) that vLLM expects.

        Default: identity. Multimodal architectures override."""
        spec = self.structure_spec()
        if spec is not None and spec.recipe_to_source:
            return spec.rewrite_recipe_to_source(model_qname)
        return model_qname

    def export_tensor_name(self, model_qname: str) -> str:
        """Rewrite an emitted tensor key to the checkpoint key to write.

        This usually matches ``source_tensor_name``. Profiles may override
        when source-checkpoint lookup and export-load naming intentionally
        differ because the serving runtime performs its own loader remap.
        """
        return self.source_tensor_name(model_qname)

    def live_to_recipe_name(self, live_qname: str) -> str:
        """Map a live HF-module qname (from `named_modules()` on the
        loaded export-time model) to the allocator-recipe qname (from
        the probe's text-only staged model).

        Multimodal architectures where AutoModelForCausalLM returns
        the `ForConditionalGeneration` sibling class get live names
        like `model.language_model.layers.X.*`, but the probe ran
        on a text-only staging that produced recipe keys like
        `model.layers.X.*`. This method strips the language_model
        infix so the allocator's assignment dict lookups succeed.

        Default: identity. Multimodal architectures override."""
        spec = self.structure_spec()
        if spec is not None and spec.live_to_recipe:
            return spec.rewrite_live_to_recipe(live_qname)
        return live_qname

    def on_disk_expert_qname(self, live_hf_qname: str) -> str:
        """Reserved for future profile-specific expert-tensor name
        rewrites. Default: identity. Currently unused by the export
        path (vLLM's architecture-specific weight-loaders handle
        `.moe.` insertion themselves via substring remaps in their
        own `load_weights` code), but kept as an extension point for
        architectures where vLLM's own remap is absent."""
        return live_hf_qname

    def split_packed_experts_for_format(self, fmt: str) -> bool:
        """Whether to split packed MoE experts into per-expert
        per-projection 2D tensors on disk for the given format.

        vLLM's MoE weight loaders vary:

          - Qwen 3.5/3.6 + compressed-tensors NVFP4: expects per-expert
            per-projection 2D tensors with compressed suffixes
            (`experts.0.gate_proj.weight_packed` etc.). We must split.

          - Gemma 4 + BF16: expects 3D packed checkpoint tensors
            (`experts.gate_up_proj`, `experts.down_proj`) and its own
            `_weight_iterator` explodes them into per-expert shards
            for FusedMoE. We must NOT split — a pre-split checkpoint
            lands under a name (`...experts.0.gate_proj`) that vLLM's
            remap turns into `...moe.experts.0.gate_proj`, which then
            misses the 3D-only explode path and fails to route onto
            the fused `w13_weight` / `w2_weight` params.

        Default: split for every non-BF16 format (NVFP4, MXFP8_E4M3, etc.)
        and keep packed for BF16. Profiles can override when their
        vLLM loader has different expectations — for instance, Qwen
        3.5/3.6 would be free to split even at BF16, though there's
        no known quality or compatibility reason to.

        When False, the exporter emits a single 3D tensor named by
        the packed param's live HF qname (e.g.
        `model.language_model.layers.0.experts.gate_up_proj`). vLLM's
        own remap inserts `.moe.` and explodes.

        When True, the exporter splits along the row dim (gate/up
        halves for `gate_up_proj`) and emits per-expert 2D tensors
        named `<parent>.{expert_id}.{proj_name}.weight[.suffix]`."""
        spec = self.structure_spec()
        if spec is not None:
            decision = spec.split_packed_experts_for_format(fmt)
            if decision is not None:
                return decision
        return fmt != "BF16"

    def packed_expert_projection_names(self, param_name: str) -> tuple[str, ...]:
        """Per-expert projection names emitted when a packed 3D parameter
        is split on disk.

        Declarative specs own the model-specific decomposition. The legacy
        fallback keeps older profiles working until they are migrated.
        """
        spec = self.structure_spec()
        if spec is not None and spec.packed_experts.declared:
            return spec.packed_expert_projection_names(param_name)
        if param_name == "gate_up_proj":
            return ("gate_proj", "up_proj")
        return (str(param_name),)

    def packed_expert_parent_for_projection(
        self,
        projection_name: str,
    ) -> str | None:
        """Inverse of :meth:`packed_expert_projection_names` for
        per-expert source keys such as ``experts.7.gate_proj``.
        """
        spec = self.structure_spec()
        if spec is not None and spec.packed_experts.declared:
            return spec.packed_expert_parent_for_projection(projection_name)
        if projection_name in {"gate_proj", "up_proj"}:
            return "gate_up_proj"
        if projection_name == "down_proj":
            return "down_proj"
        if projection_name in self.packed_expert_param_names():
            return projection_name
        return None

    def vllm_fused_moe_scheme_projection_names(
        self, param_name: str
    ) -> tuple[str, ...]:
        """Per-expert projection names vLLM's FusedMoE scheme detection
        (`get_moe_method`) and ignore-matching probe at load time.

        vLLM builds synthetic per-expert names ``experts.0.gate_proj`` /
        ``up_proj`` / ``down_proj`` to look up the FusedMoE quant scheme,
        regardless of the checkpoint's actual projection names. So
        compressed-tensors ``config_groups`` targets and ``ignore`` regexes
        for packed experts must use THESE canonical names — not
        :meth:`packed_expert_projection_names`, which names the on-disk
        weights (e.g. LFM2.5's ``w1``/``w3``/``w2``). Using the on-disk
        names makes vLLM mis-resolve the scheme (it loses the input-
        activation spec → builds the weight-only NVFP4A16 variant, or marks
        BF16 experts un-ignored) and the artifact fails to load. The weights
        themselves still load via the model's expert mapping
        (``gate_proj``=w1, ``up_proj``=w3, ``down_proj``=w2)."""
        if param_name == "gate_up_proj":
            return ("gate_proj", "up_proj")
        if param_name == "down_proj":
            return ("down_proj",)
        if param_name in ("gate_proj", "up_proj", "down_proj"):
            return (param_name,)
        # Unknown packed param: fall back to the on-disk projection names.
        return self.packed_expert_projection_names(param_name)

    def unpacked_expert_projection_names(self) -> tuple[str, ...]:
        """Per-expert *module attribute* names for UNPACKED MoE experts.

        Applies only to architectures where each routed expert is its own
        ``nn.Module`` exposing per-projection ``nn.Linear`` attributes (the
        MiniMax-M2 / Qwen3 / Qwen3.5 MoE layout, e.g. ``.w1``/``.w2``/``.w3``).
        The batched-Fisher MoE-block detector and the fast-MoE forward swap in
        the probes use these names to recognize an expert container; if the
        names don't match, those optimizations silently no-op (probe speed
        only — per-Linear Fisher still accumulates via the regular hooks).

        Packed-expert architectures (DeepSeek-V4: 3D ``gate_up_proj`` /
        ``down_proj`` tensors, no per-expert modules) have no such attributes
        and never match the consumers of this accessor, so the default is
        harmless for them. A declarative structure spec may override via an
        ``unpacked_expert_projection_names`` field; otherwise the default is
        the Qwen3/Qwen3.5 standard ``('w1', 'w2', 'w3')``. Profiles whose
        unpacked experts use different attribute names should override this.
        """
        spec = self.structure_spec()
        declared = getattr(spec, "unpacked_expert_projection_names", None)
        if declared:
            names = declared() if callable(declared) else declared
            if names:
                return tuple(names)
        return ("w1", "w2", "w3")

    def _fallback_packed_expert_format_groups(self) -> tuple[tuple[str, ...], ...]:
        """Common legacy packed-MoE coupling groups for profiles without specs.

        This keeps pre-spec profiles working while preserving the boundary:
        the solver asks the profile for groups; it does not parse model names
        itself. New model families should declare ``packed_experts`` format
        groups in the JSON structure spec instead of depending on this fallback.
        """
        return (
            ("gate_up_proj", "down_proj"),
            ("gate_proj", "up_proj", "down_proj"),
            ("w1", "w2", "w3"),
        )

    def _packed_expert_group_matches_representation(
        self,
        group: tuple[str, ...],
        *,
        split_per_expert: bool,
    ) -> bool:
        packed_names = set(self.packed_expert_param_names())
        if split_per_expert:
            return not any(
                member in packed_names
                and self.packed_expert_projection_names(member) != (member,)
                for member in group
            )
        return not any(
            member not in packed_names
            and self.packed_expert_parent_for_projection(member) is not None
            for member in group
        )

    def packed_expert_format_group(self, qname: str) -> str | None:
        """Return a group key for packed-expert projections that must
        share one serving format.

        This is the model/profile side of vLLM FusedMoE scheme coupling.
        The allocator asks the profile for this key instead of hardcoding
        Qwen/Gemma expert path regexes in the solver.
        """
        spec = self.structure_spec()
        if spec is not None:
            return spec.packed_expert_format_group(qname)
        parts = str(qname).split(".")
        try:
            experts_idx = len(parts) - 1 - list(reversed(parts)).index("experts")
        except ValueError:
            return None
        tail = parts[experts_idx + 1:]
        if len(tail) == 1:
            parent = ".".join(parts[:experts_idx + 1])
            leaf = tail[0]
            split_per_expert = False
        elif len(tail) == 2 and tail[0].isdigit():
            parent = ".".join(parts[:experts_idx + 2])
            leaf = tail[1]
            split_per_expert = True
        else:
            return None

        for group in self._fallback_packed_expert_format_groups():
            if leaf not in group:
                continue
            if not self._packed_expert_group_matches_representation(
                group,
                split_per_expert=split_per_expert,
            ):
                continue
            return f"{parent}::__packed_format__:{','.join(group)}"
        return None

    # ------------------------------------------------------------
    # Source passthrough + text-only staging
    # ------------------------------------------------------------
    def source_passthrough_prefixes(self) -> tuple[str, ...]:
        """Prefixes of checkpoint keys that should be copied from the
        source checkpoint as-is (typically visual encoder + MTP when
        not being quantized)."""
        spec = self.structure_spec()
        if spec is not None and spec.passthrough_prefixes:
            return spec.passthrough_prefixes
        return ()

    def serving_profile_id(self) -> str | None:
        """Default serving/backend constraint profile for this model family."""
        spec = self.structure_spec()
        if spec is not None:
            return spec.default_serving_profile
        return None

    def stage_text_only_strip_keys(self) -> tuple[str, ...]:
        """HF config keys to drop when creating a text-only staged
        config for probe/cost model loading (e.g. `vision_config` on
        multimodal models so `AutoModelForCausalLM` can load)."""
        spec = self.structure_spec()
        if spec is not None and spec.stage_text_only_strip_keys is not None:
            return spec.stage_text_only_strip_keys
        return ("vision_config", "audio_config", "speech_config")

    def stage_text_only_promote_inner_model_type(self) -> bool:
        """When lifting `text_config` keys to top-level during
        text-only staging, should `text_config.model_type` (e.g.
        `gemma4_text`) shadow the outer `model_type` (e.g. `gemma4`)?

        This depends on which HF config class the family's
        `<Arch>ForCausalLM` expects:

        - Gemma 4: `Gemma4ForCausalLM.config: Gemma4TextConfig` — the
          text-specific config class. We must promote `gemma4_text`
          so `AutoConfig` loads `Gemma4TextConfig` and the flat text
          schema's `hidden_size` / `num_hidden_layers` etc. all line
          up with the text checkpoint tensors.

        - Qwen 3.5 MoE: `Qwen3_5MoeForCausalLM.config: Qwen3_5MoeConfig`
          — the multimodal-umbrella config class (with nested
          `text_config`). We must KEEP the outer `qwen3_5_moe` so
          `AutoConfig` loads `Qwen3_5MoeConfig` and the nested
          text_config gets wired in normally.

        Default False (Qwen-like). Families that take a standalone
        text config class override to True."""
        spec = self.structure_spec()
        if (
            spec is not None
            and spec.stage_text_only_promote_inner_model_type is not None
        ):
            return bool(spec.stage_text_only_promote_inner_model_type)
        return False

    # ------------------------------------------------------------
    # Extended shard regexes (incremental_probe)
    # ------------------------------------------------------------
    def extended_shard_regexes(self, model_path: str,
                               layers_per_shard: int,
                               *, include_body: bool = True,
                               include_mtp: bool = True,
                               include_visual: bool = True,
                               include_lm_head: bool = True) -> list[str]:
        """Return the list of Linear-name regexes covering every shard
        of the probe — body, MTP, visual, lm_head.

        Reads the SOURCE config (not a staged copy) so vision/MTP
        metadata that text-only staging might strip remains visible."""
        src_cfg_path = Path(model_path) / "config.json"
        with open(src_cfg_path) as f:
            cfg = json.load(f)
        text_cfg = cfg.get("text_config", cfg)

        regexes: list[str] = []
        if include_body:
            n_body = int(text_cfg.get("num_hidden_layers",
                                       cfg.get("num_hidden_layers", 0)))
            regexes.extend(
                _build_layer_shard_regexes(n_body, layers_per_shard,
                                           layer_prefix=self.body_layer_prefix()))
        if include_mtp and self.has_mtp():
            n_mtp = int(self.mtp_layer_count(cfg) or 0)
            if n_mtp > 0:
                mtp_regexes = (
                    _build_layer_shard_regexes(n_mtp, layers_per_shard,
                                               layer_prefix=self.mtp_layer_prefix()))
                if mtp_regexes and self.mtp_extra_linear_names():
                    extra = "|".join(
                        re.escape(name) for name in self.mtp_extra_linear_names()
                    )
                    mtp_regexes[0] = rf"(?:{extra}|{mtp_regexes[0]})"
                regexes.extend(mtp_regexes)
        visual_key = self.visual_config_key()
        if include_visual and visual_key:
            vis_cfg = cfg.get(visual_key, {})
            n_vis = int(
                vis_cfg.get("depth") or vis_cfg.get("num_hidden_layers") or 0
            )
            if n_vis > 0:
                regexes.extend(
                    _build_layer_shard_regexes(n_vis,
                                               max(layers_per_shard, 4),
                                               layer_prefix=self.visual_layer_prefix()))
        if include_lm_head:
            regexes.append(rf"^{re.escape(self.lm_head_name())}$")
        return regexes

    def body_layer_prefix(self) -> str:
        """Prefix used for body-layer names in the checkpoint (before
        the numeric index)."""
        spec = self.structure_spec()
        if spec is not None and spec.body_layer_prefix is not None:
            return spec.body_layer_prefix
        return "model.layers"

    def mtp_layer_prefix(self) -> str:
        """Prefix used for MTP-layer names in the checkpoint."""
        spec = self.structure_spec()
        if spec is not None and spec.mtp_layer_prefix is not None:
            return spec.mtp_layer_prefix
        return "mtp.layers"

    def mtp_extra_linear_names(self) -> tuple[str, ...]:
        """Top-level MTP Linear qnames to include in the first MTP shard."""
        spec = self.structure_spec()
        if spec is not None:
            return tuple(spec.mtp_extra_linear_names)
        return ("mtp.fc",)

    def visual_layer_prefix(self) -> str | None:
        """Prefix used for visual-encoder block names, or None if this
        model has no visual encoder."""
        spec = self.structure_spec()
        if spec is not None and spec.visual_layer_prefix is not None:
            return spec.visual_layer_prefix
        return None

    def visual_config_key(self) -> str | None:
        """Top-level HF config key under which the vision_config dict
        lives, or None if this model has no visual encoder."""
        spec = self.structure_spec()
        if spec is not None and spec.visual_config_key is not None:
            return spec.visual_config_key
        return None

    def lm_head_name(self) -> str:
        """Qualified name of the lm_head Linear in the checkpoint."""
        spec = self.structure_spec()
        if spec is not None and spec.lm_head_name is not None:
            return spec.lm_head_name
        return "lm_head"

    def mtp_layer_count(self, cfg: dict) -> int:
        """Count of MTP layers from the HF config. Fall back to
        scanning the safetensors index via `_count_mtp_layers_from_safetensors`
        in subclasses when the config doesn't report it."""
        text = cfg.get("text_config", cfg)
        return int(
            text.get("num_nextn_predict_layers")
            or cfg.get("num_nextn_predict_layers")
            or text.get("num_mtp_layers")
            or cfg.get("num_mtp_layers")
            or text.get("mtp_num_hidden_layers")
            or cfg.get("mtp_num_hidden_layers")
            or 0
        )

    # ------------------------------------------------------------
    # Streaming probe adapters (DSv4 generalization, refactor #32)
    #
    # Profiles override these to teach prismaquant about an architecture's
    # idiosyncrasies WITHOUT touching layer_streaming / streaming_model /
    # incremental_probe core paths. Default implementations preserve the
    # behavior the existing codebase had before the refactor — MiniMax,
    # Qwen3.5/3.6, Gemma4 and similar architectures all use the defaults.
    # ------------------------------------------------------------

    def checkpoint_to_live_name(self, ckpt_key: str, *,
                                multimodal: bool = False) -> str | None:
        """Map a checkpoint key (as found in the safetensors index) to
        the live transformers module qname (as found by
        `model.named_parameters()`). Return None to drop the key from
        the body weight map (it is then either ignored, or consumed
        via a sibling path like the FP8 dequant scale map).

        Default: drop visual/audio/MTP keys, drop `.weight_scale_inv`
        (those go through the FP8 scale map), pass everything else
        through unchanged. The multimodal-umbrella branch strips the
        `model.language_model.` infix so probe-side text-only staging
        and the source checkpoint line up.

        DSv4 overrides this to handle its flat naming convention
        (`embed.weight`, `layers.5.attn.wkv.weight` → standard
        transformers names)."""
        if ckpt_key.endswith(".weight_scale_inv"):
            return None
        if (ckpt_key.startswith("model.visual.")
                or ckpt_key.startswith("model.audio_tower.")
                or ckpt_key.startswith("model.vision_tower.")
                or ckpt_key.startswith("model.embed_vision.")
                or ckpt_key.startswith("model.embed_audio.")
                or ckpt_key.startswith("mtp.")):
            return None if not multimodal else (
                # Multimodal staging keeps visual/audio prefixes verbatim
                # but still drops MTP (handled by the MTP synthesis path).
                None if ckpt_key.startswith("mtp.") else ckpt_key)
        if not multimodal and ckpt_key.startswith("model.language_model."):
            return "model." + ckpt_key[len("model.language_model."):]
        return ckpt_key

    def fp8_scale_pairs(self, model_path: str
                        ) -> dict[str, tuple[str, str]] | None:
        """Return `{model_weight_key: (scale_shard_path, scale_ckpt_key)}`
        for every native-FP8 weight tensor in this checkpoint. Returns
        None to fall through to the default `.weight_scale_inv`
        sibling discovery path. Returns `{}` to indicate "no FP8 dequant
        applies to this model". Returns a populated dict to fully
        override the discovery (e.g. DSv4 uses `.scale` siblings).

        Default: None (use the legacy `.weight_scale_inv` discovery)."""
        return None

    def head_resident_extra_prefixes(self, root) -> list[str]:
        """Extra prefixes (rooted under the base model where possible)
        to load with the head-resident batch (embed/norm/lm_head/rotary).
        DSv4 returns `["hc_head."]` so its multi-stream collapse can
        run with real weights at end-of-phase-1.

        Default: empty."""
        return []

    def init_rotaries(self, rotary, cfg, device, dtype) -> bool:
        """Optionally populate rotary buffers on a meta-built skeleton.
        Return True if the profile fully handled init (the caller skips
        its default path), or False to fall through to the standard
        single-rope flow.

        DSv4 / Gemma3 return True after registering per-layer-type
        `<name>_inv_freq` buffers (the rotary has a `layer_types` tuple
        like `("main", "compress")`).

        Default: False (single-rope path)."""
        return False

    def expand_hidden_for_layers(self, hidden, base_model):
        """Optionally reshape the post-embedding hidden state before
        the per-layer forward loop. DSv4 expands single-stream
        `[B, T, H]` to multi-stream `[B, T, hc_mult, H]` (mirrors
        `DeepseekV4Model.forward`). Default: passthrough."""
        return hidden

    def collapse_hidden_after_layers(self, hidden, base_model):
        """Inverse of `expand_hidden_for_layers`: collapse the post-loop
        hidden state back to standard `[B, T, H]` before the final
        norm + lm_head. DSv4 calls `base_model.hc_head(hidden)`.
        Default: passthrough."""
        return hidden

    def extra_layer_kwargs(self, *, input_ids=None) -> dict:
        """Extra kwargs to pass to `layer(...)` during phase-1/3.
        DSv4 hash-routed layers consume `input_ids` for the `tid2eid`
        lookup; other architectures ignore it. Default: empty dict
        (which the layer's `**kwargs` absorbs)."""
        return {}

    def should_probe_linear(self, name: str, mod) -> bool:
        """Whether to register Fisher hooks on this Linear module.
        DSv4 returns False for `DeepseekV4GroupedLinear` (its weight
        shape doesn't match the per-token Hessian-trace effective
        output dim, so the chunk_h * w.pow(2) accumulator can't
        broadcast). Default: True for any nn.Linear instance.

        Profiles may also use this to skip e.g. router gates that
        shouldn't carry Fisher info."""
        import torch.nn as _nn
        if not isinstance(mod, _nn.Linear):
            return False
        spec = self.structure_spec()
        if spec is not None:
            skipped = set(spec.probe_skip_module_class_names)
            if type(mod).__name__ in skipped:
                return False
        return True

    def register_vendored_modeling(self) -> None:
        """Called once when this profile is instantiated by
        `detect_profile()`. Profiles that vendor transformers modeling
        code (DSv4) use this to install monkey-patches and register
        with AutoConfig / AutoModelForCausalLM. Default: no-op."""
        pass

    # ------------------------------------------------------------
    # Declarative structure graph
    # ------------------------------------------------------------
    def structure_spec(self):
        """Return this profile's declarative structure spec, if present.

        The spec is an additive, no-behavior-change description of naming,
        grouping, passthrough, and decomposition rules.  Existing production
        paths continue to use the executable profile methods until call sites
        explicitly opt into a ``ModelGraph``.
        """
        from .structure import load_structure_spec

        if not self._structure_spec_loaded:
            self._structure_spec = load_structure_spec(self.name)
            self._structure_spec_loaded = True
        return self._structure_spec

    def build_model_graph(self, model):
        """Build a typed graph from a live model using this profile.

        This is intentionally not called from hot paths yet.  It provides a
        single graph artifact for future allocator/cache/export migration while
        preserving the current cache and prefetch implementations.
        """
        from .structure import build_model_graph

        return build_model_graph(model, self, spec=self.structure_spec())


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
def _build_layer_shard_regexes(num_layers: int,
                               layers_per_shard: int,
                               *, layer_prefix: str) -> list[str]:
    out: list[str] = []
    for start in range(0, num_layers, layers_per_shard):
        end = min(start + layers_per_shard, num_layers)
        if end - start == 1:
            body = rf"{re.escape(layer_prefix)}\.{start}\."
        else:
            idxs = "|".join(str(i) for i in range(start, end))
            body = rf"{re.escape(layer_prefix)}\.(?:{idxs})\."
        out.append(body)
    return out
