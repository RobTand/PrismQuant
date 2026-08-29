"""Qwen4-Exp / Qwen3.8-Flash-Next producer profile.

The profile is grounded in the official Transformers ``qwen4_exp`` model
and vLLM's dedicated Qwen3.8-Flash-Next implementation (PR #53896, head
``d4d0f73``).  The source checkpoint is a multimodal wrapper while the
calibration model is staged as the text-only CausalLM:

  | where                    | body namespace                    |
  |--------------------------|-----------------------------------|
  | HF multimodal source     | model.language_model.layers.X.*  |
  | PrismaQuant recipe       | model.layers.X.*                 |
  | vLLM multimodal runtime  | language_model.model.layers.X.*  |

Qwen4's MTP is a different architecture from the Qwen3.5 sidecar.  Until a
faithful probe module exists here, ``mtp.*`` is copied verbatim and is not
advertised as probeable.  The vision tower is handled the same way.  PLE and
hyperconnection weights stay on the immutable source-precision floor: PLE's
~95 GiB sharded n-gram embedding has a dedicated runtime loader, and vLLM
constructs the hyperconnection Linears with ``quant_config=None``.
"""
from __future__ import annotations

from .base import ModelProfile


class Qwen4ExpProfile(ModelProfile):
    """Architecture adapter for official Qwen4-Exp checkpoints."""

    # Distinct from Qwen3, but keep the family next to it in detection order.
    priority = 130

    TEXT_ONLY_ARCHITECTURES = frozenset({"Qwen4ExpForCausalLM"})

    @classmethod
    def matches(cls, model_type: str, architectures: list[str]) -> bool:
        if model_type in {"qwen4_exp", "qwen4_exp_text"}:
            return True
        return any(arch.startswith("Qwen4Exp") for arch in architectures)

    @property
    def name(self) -> str:
        return "qwen4_exp"

    def _is_text_only(self) -> bool:
        if self._declared_model_type == "qwen4_exp_text":
            return True
        archs = self.declared_architectures()
        return bool(archs) and all(
            arch in self.TEXT_ONLY_ARCHITECTURES for arch in archs
        )

    def vllm_architecture_class(self) -> str | None:
        if self._is_text_only():
            return "Qwen4ExpForCausalLM"
        return "Qwen4ExpForConditionalGeneration"

    def has_mtp(self) -> bool:
        """Keep Qwen4 MTP verbatim until its exact module is implemented."""
        return False
