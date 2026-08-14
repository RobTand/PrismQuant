"""Qwen3 MoE profile.

Covers:
  - Qwen3MoeForCausalLM

This is the original Qwen3 MoE family, not the Qwen3.5/3.6 multimodal/MTP
family.  It keeps Qwen3's plain text naming while adding packed expert tensors
(`experts.gate_up_proj`, `experts.down_proj`) and the vLLM per-expert target
regex needed for compressed-tensors scheme dispatch.
"""
from __future__ import annotations

from .qwen3 import Qwen3Profile


class Qwen3MoeProfile(Qwen3Profile):

    # Detection priority (lower = consulted first): must precede Qwen3Profile — `qwen3_moe` archs start with the qwen3 prefix.
    priority = 120

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type == "qwen3_moe":
            return True
        return any(arch.startswith("Qwen3Moe") for arch in architectures)

    @property
    def name(self) -> str:
        return "qwen3_moe"

    def vllm_architecture_class(self) -> str | None:
        return "Qwen3MoeForCausalLM"

    def register_vendored_modeling(self) -> None:
        # The vendored dense Qwen3 modeling patch is intentionally scoped to
        # Qwen3ForCausalLM.  Qwen3-MoE uses the installed transformers class.
        pass
