"""PrismaQuant model profiles — architecture-specific adapters.

Exports:
  - ModelProfile: abstract base class
  - DefaultProfile: generic fallback
  - Qwen3Profile: covers original Qwen3 dense and routed-MoE text models
  - Qwen3_5Profile: covers Qwen3.5 and Qwen3.6 MoE (w/ MTP)
  - Gemma4Profile: covers Gemma 4 dense + MoE multimodal
  - detect_profile(model_path): auto-detect profile from HF config
  - register_profile(cls): register a custom profile at runtime
  - ModelGraph / ModelStructureSpec: typed model decomposition artifacts

MiniMaxM2Profile was archived 2026-04-24; see archive/minimax_m2p7_2026-04-24/README.md.
"""
from .base import ModelProfile
from .default import DefaultProfile
from .deepseek_v4 import DeepseekV4Profile
from .gemma4 import Gemma4Profile
from .hy_v3 import HyV3Profile
from .laguna import LagunaProfile
from .lfm2_moe import Lfm2MoeProfile
from .qwen3 import Qwen3Profile
from .qwen3_5 import Qwen3_5Profile
from .qwen3_5_dense import Qwen3_5DenseProfile
from .qwen4_exp import Qwen4ExpProfile
from .registry import (
    detect_profile,
    detect_profile_with_warning,
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
    "Qwen3_5Profile",
    "Qwen3_5DenseProfile",
    "Qwen4ExpProfile",
    "Gemma4Profile",
    "HyV3Profile",
    "LagunaProfile",
    "Lfm2MoeProfile",
    "detect_profile",
    "detect_profile_with_warning",
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
