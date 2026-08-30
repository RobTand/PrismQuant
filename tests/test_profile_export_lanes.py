"""R6 (reader half) — export-lane eligibility as model configuration.

`EXPORT_CONTAINER` is an operator env var with no relationship to whether the
architecture is wired for that lane. Nothing stops `EXPORT_CONTAINER=nvfp4_cb`
on an arch whose gridbook CB expert loader is a TODO: the run completes, the
artifact serves, and the FusedMoE reads uninitialised memory — coherent-looking
garbage, not a crash (commit `9a79963`, Laguna, 93% of parameters). The honest
CB-eligible set is six producer profiles and until now nothing in the tree said
so.

This file pins the spec fields, profile accessors, and the preflight helper now
wired by `run-pipeline.sh`. Cross-repository CI separately compares the exact
producer set with Gridbook's packaged contract.
"""
from __future__ import annotations

import pytest

from prismaquant.model_profiles import detect_profile  # noqa: F401  (API shape)
from prismaquant.model_profiles import registry as _registry
from prismaquant.model_profiles.structure import (
    DEFAULT_EXPORT_LANE,
    EXPORT_LANES,
    ModelStructureSpec,
    SCHEMA,
    canonical_export_lane,
    load_structure_spec,
)
from prismaquant.serving_profiles import (
    load_serving_profile,
    require_lane_supported,
    require_profile_export_lane,
)

# Gridbook support is intentionally not duplicated here. The separately pinned
# runtime publishes its supported producer IDs in runtime_contract.json, and
# tests/test_gridbook_runtime_contract.py compares declarations to that one
# machine-readable table in the dedicated integration job.
GGUF_WIRED = {"hy_v3"}

PROFILE_CLASSES = list(_registry._REGISTERED)
PROFILE_IDS = [c.__name__ for c in PROFILE_CLASSES]


def _profile(cls):
    return cls()


# ------------------------------------------------------------------ vocabulary


def test_lane_vocabulary_is_the_export_container_vocabulary():
    assert EXPORT_LANES == ("compressed-tensors", "nvfp4_cb", "gguf")
    assert DEFAULT_EXPORT_LANE == "compressed-tensors"
    # One declared alias: the serving-profile side spells the native lane with
    # an underscore (`export_lane.id == "compressed_tensors"`).
    assert canonical_export_lane("compressed_tensors") == "compressed-tensors"
    with pytest.raises(ValueError, match="unknown export lane"):
        canonical_export_lane("nvfp4-cb")


def test_preferred_lane_must_be_supported():
    with pytest.raises(ValueError, match="preferred_lane"):
        ModelStructureSpec.from_dict({
            "schema": SCHEMA,
            "id": "lane_test",
            "supported_lanes": ["compressed-tensors"],
            "preferred_lane": "gguf",
        })


# --------------------------------------------------------------- declarations


@pytest.mark.parametrize("cls", PROFILE_CLASSES, ids=PROFILE_IDS)
def test_declared_lanes_include_native_and_match_local_gguf_wiring(cls):
    """Locally owned lanes are checked here; Gridbook is checked externally."""
    profile = _profile(cls)
    lanes = set(profile.supported_export_lanes())
    assert DEFAULT_EXPORT_LANE in lanes, (
        f"{profile.name}: every architecture ships through the native lane")
    assert ("gguf" in lanes) == (profile.name in GGUF_WIRED), (
        f"{profile.name}: gguf declaration disagrees with {sorted(GGUF_WIRED)}")


@pytest.mark.parametrize("cls", PROFILE_CLASSES, ids=PROFILE_IDS)
def test_preferred_lane_is_supported_and_defaults_to_native(cls):
    profile = _profile(cls)
    preferred = profile.preferred_export_lane()
    assert preferred in profile.supported_export_lanes()
    spec = load_structure_spec(profile.name)
    if spec is None or not spec.preferred_lane:
        assert preferred == DEFAULT_EXPORT_LANE


def test_the_two_script_driven_lanes_are_declared():
    """`scripts/run_*_prod_nvfp4cb.sh` and the GGUF lane are tribal knowledge
    today; these two are the archs whose shipped artifacts came off them."""
    assert load_structure_spec("hy_v3").preferred_lane == "gguf"
    assert load_structure_spec("laguna").preferred_lane == "nvfp4_cb"


# ------------------------------------------------------------------- preflight


def test_require_lane_supported_accepts_declared_lanes():
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile
    from prismaquant.model_profiles.laguna import LagunaProfile
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    assert require_lane_supported(Qwen3_5Profile(), "nvfp4_cb") == "nvfp4_cb"
    assert require_lane_supported(DeepseekV4Profile(), "nvfp4_cb") == "nvfp4_cb"
    assert require_lane_supported(LagunaProfile(), None) == "compressed-tensors"
    assert require_lane_supported(
        Qwen3_5Profile(), "compressed_tensors") == "compressed-tensors"


def test_require_lane_supported_refuses_an_undeclared_lane():
    from prismaquant.model_profiles.gemma4 import Gemma4Profile

    with pytest.raises(SystemExit) as excinfo:
        require_lane_supported(Gemma4Profile(), "nvfp4_cb")
    message = str(excinfo.value)
    assert "gemma4" in message
    assert "compressed-tensors" in message      # names the declared set
    assert "garbage" in message                 # names the failure mode


def test_require_lane_supported_refuses_an_unknown_lane():
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    with pytest.raises(SystemExit, match="unknown export lane"):
        require_lane_supported(Qwen3_5Profile(), "nvfp4-cb")


def test_require_lane_supported_is_inert_without_the_accessor():
    """Duck-typed like `resolve_target_profile`: a profile object that predates
    the accessor must not break a run."""
    class _Legacy:
        name = "legacy"

    assert require_lane_supported(_Legacy(), "gguf") == "gguf"


def test_narrow_4090_profile_inherits_cb_serializer_by_lane_identity():
    profile = load_serving_profile("qwen38_rtx4090_fp8_cb")
    assert profile.export_lane is not None
    assert profile.export_lane.id == "nvfp4_cb"
    assert profile.target_platform == "sm_89"
    assert profile.producer_policy == "qwen38_27b_rtx4090_fp8_cb"
    assert require_profile_export_lane(profile.id, "nvfp4_cb") == "nvfp4_cb"
    with pytest.raises(ValueError, match="not requested container"):
        require_profile_export_lane(profile.id, "compressed-tensors")

    validation = load_serving_profile(
        "qwen38_rtx4090_fp8_cb_validation_only"
    )
    assert validation.export_lane is not None
    assert validation.export_lane.id == "nvfp4_cb"
    assert validation.target_platform == "sm_89"
    assert validation.producer_policy == (
        "qwen38_27b_rtx4090_fp8_cb_validation_only"
    )


def test_no_shipped_lane_run_becomes_illegal():
    """Non-regression: every in-tree launch script's (arch, EXPORT_CONTAINER)
    pair must still pass the preflight."""
    cases = [
        ("qwen3_5", "nvfp4_cb"),        # run_27b_prod_nvfp4cb, run_35b_prod_nvfp4cb
        ("qwen3_5_dense", "nvfp4_cb"),
        ("hy_v3", "nvfp4_cb"),          # run_hy3_prod_nvfp4cb / _joint
        ("hy_v3", "gguf"),
        ("laguna", "nvfp4_cb"),         # run_laguna_s21_prod
        ("qwen3_5", "compressed-tensors"),
        ("gemma4", "compressed-tensors"),
        ("lfm2_moe", "compressed-tensors"),
        ("minimax_m2", "compressed-tensors"),
        ("deepseek_v4", "compressed-tensors"),
        ("deepseek_v4", "nvfp4_cb"),
    ]
    by_name = {c().name: c() for c in PROFILE_CLASSES}
    for name, lane in cases:
        assert require_lane_supported(by_name[name], lane) == lane
