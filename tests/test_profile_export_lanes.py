"""R6 (reader half) — export-lane eligibility as model configuration.

`EXPORT_CONTAINER` is an operator env var with no relationship to whether the
architecture is wired for that lane. Nothing stopped `EXPORT_CONTAINER=<lane>`
on an arch whose expert loader for that lane is a TODO: the run completes, the
artifact serves, and the FusedMoE reads uninitialised memory — coherent-looking
garbage, not a crash (commit `9a79963`, Laguna, 93% of parameters).

This file pins the spec fields, profile accessors, and the preflight helper
wired by `run-pipeline.sh`.

The lane vocabulary lost `nvfp4_cb` on 2026-09-02 when the Gridbook codebook
lane was retired (`archive/gridbook_lane_2026-09-02/`). The cases below were
re-pointed at the lanes that remain rather than deleted: the *mechanism* — an
architecture declares its lanes, and the preflight refuses an undeclared one —
is exactly what the retirement leaves standing, and it is what the Tessera
lane will be admitted through.
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

# A serving runtime's supported producer IDs are never duplicated here: the
# pinned runtime publishes them in its own runtime_contract.json and the
# comparison is made against that one machine-readable table (AGENTS.md
# principle 5 / CLAUDE.md principle 14). The Gridbook half of that comparison
# retired with its lane on 2026-09-02.
GGUF_WIRED = {"hy_v3"}

PROFILE_CLASSES = list(_registry._REGISTERED)
PROFILE_IDS = [c.__name__ for c in PROFILE_CLASSES]


def _profile(cls):
    return cls()


# ------------------------------------------------------------------ vocabulary


def test_lane_vocabulary_is_the_export_container_vocabulary():
    assert EXPORT_LANES == ("compressed-tensors", "gguf")
    assert DEFAULT_EXPORT_LANE == "compressed-tensors"
    # One declared alias: the serving-profile side spells the native lane with
    # an underscore (`export_lane.id == "compressed_tensors"`).
    assert canonical_export_lane("compressed_tensors") == "compressed-tensors"
    with pytest.raises(ValueError, match="unknown export lane"):
        canonical_export_lane("nvfp4-cb")
    # Retired 2026-09-02 with the Gridbook lane. The vocabulary is the
    # EXPORT_CONTAINER vocabulary, so a stale driver or spec naming the lane
    # now fails loudly at the preflight instead of writing bytes nothing
    # serves.
    with pytest.raises(ValueError, match="unknown export lane"):
        canonical_export_lane("nvfp4_cb")


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


def test_the_script_driven_lane_is_declared():
    """The GGUF lane is tribal knowledge today; hy_v3 is the arch whose
    shipped artifact came off it.

    Laguna's `preferred_lane` was `nvfp4_cb` until 2026-09-02 and is now the
    default native lane: its shipped artifact came off the retired codebook
    lane, and it has no other declared preference.
    """
    assert load_structure_spec("hy_v3").preferred_lane == "gguf"
    assert load_structure_spec("laguna").preferred_lane == "compressed-tensors"


# ------------------------------------------------------------------- preflight


def test_require_lane_supported_accepts_declared_lanes():
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile
    from prismaquant.model_profiles.hy_v3 import HyV3Profile
    from prismaquant.model_profiles.laguna import LagunaProfile
    from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile

    assert require_lane_supported(HyV3Profile(), "gguf") == "gguf"
    assert require_lane_supported(
        DeepseekV4Profile(), "compressed-tensors") == "compressed-tensors"
    assert require_lane_supported(LagunaProfile(), None) == "compressed-tensors"
    assert require_lane_supported(
        Qwen3_5Profile(), "compressed_tensors") == "compressed-tensors"


def test_require_lane_supported_refuses_an_undeclared_lane():
    from prismaquant.model_profiles.gemma4 import Gemma4Profile

    # `gguf` since 2026-09-02 — `nvfp4_cb` is no longer *undeclared*, it is
    # unknown, and that is a different refusal (below).
    with pytest.raises(SystemExit) as excinfo:
        require_lane_supported(Gemma4Profile(), "gguf")
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


# `test_narrow_4090_profile_inherits_cb_serializer_by_lane_identity` was
# deleted on 2026-09-02: it loaded `qwen38_rtx4090_fp8_cb`, one of the six CB
# producer profiles, all of which went to archive/gridbook_lane_2026-09-02/
# with the lane. `require_profile_export_lane` itself is still exercised by
# `test_serving_profiles.py`; what is gone is the only lane whose serving
# profile pinned a non-default `export_lane.id`.


def test_no_shipped_lane_run_becomes_illegal():
    """Non-regression: every in-tree launch script's (arch, EXPORT_CONTAINER)
    pair must still pass the preflight.

    The `nvfp4_cb` pairs — qwen3_5, qwen3_5_dense, hy_v3, laguna, deepseek_v4 —
    were removed on 2026-09-02 along with their launch scripts. Those runs ARE
    now illegal, deliberately: `canonical_export_lane` refuses the container
    before the profile is consulted (pinned above).
    """
    cases = [
        ("hy_v3", "gguf"),
        ("qwen3_5", "compressed-tensors"),
        ("gemma4", "compressed-tensors"),
        ("lfm2_moe", "compressed-tensors"),
        ("minimax_m2", "compressed-tensors"),
        ("deepseek_v4", "compressed-tensors"),
    ]
    by_name = {c().name: c() for c in PROFILE_CLASSES}
    for name, lane in cases:
        assert require_lane_supported(by_name[name], lane) == lane
