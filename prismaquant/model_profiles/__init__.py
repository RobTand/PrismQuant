"""PrismaQuant model profiles — architecture-specific adapters.

Exports:
  - ModelProfile: abstract base class
  - DefaultProfile: generic fallback
  - Qwen3Profile: covers Qwen3 dense (0.6B-32B)
  - Qwen3MoeProfile: covers Qwen3 MoE
  - Qwen3_5Profile: covers Qwen3.5 and Qwen3.6 MoE (w/ MTP)
  - Gemma4Profile: covers Gemma 4 dense + MoE multimodal
  - detect_profile(model_path): auto-detect profile from HF config
  - register_profile(cls): register a custom profile at runtime
  - ModelGraph / ModelStructureSpec: typed model decomposition artifacts

MiniMaxM2Profile was archived 2026-04-24; see archive/minimax_m2p7/README.md.
"""
from .base import ModelProfile
from .default import DefaultProfile
from .deepseek_v4 import DeepseekV4Profile
from .gemma4 import Gemma4Profile
from .lfm2_moe import Lfm2MoeProfile
from .qwen3 import Qwen3Profile
from .qwen3_5 import Qwen3_5Profile
from .qwen3_5_dense import Qwen3_5DenseProfile
from .qwen3_moe import Qwen3MoeProfile
from .registry import (
    detect_profile,
    profile_from_config,
    profile_from_model,
    register_profile,
)
from .structure import (
    ModelGraph,
    ModelStructureSpec,
    ModelTensor,
    OptimizationUnit,
    build_model_graph,
    load_structure_spec,
)

__all__ = [
    "ModelProfile",
    "DefaultProfile",
    "DeepseekV4Profile",
    "Qwen3Profile",
    "Qwen3MoeProfile",
    "Qwen3_5Profile",
    "Qwen3_5DenseProfile",
    "Gemma4Profile",
    "Lfm2MoeProfile",
    "detect_profile",
    "profile_from_config",
    "profile_from_model",
    "register_profile",
    "ModelGraph",
    "ModelStructureSpec",
    "ModelTensor",
    "OptimizationUnit",
    "build_model_graph",
    "load_structure_spec",
]
