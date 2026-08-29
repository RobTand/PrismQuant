from __future__ import annotations

import re
from pathlib import Path

from prismaquant.fixed_head import (
    allow_pinned_lifts_lm_head,
    is_lm_head_name,
    remaining_profile_pins,
)


ROOT = Path(__file__).resolve().parents[1]


class _AliasedHeadProfile:
    def pinned_names(self):
        return ("lm_head", "head", "router")

    def lm_head_name(self):
        return "head"


def test_fixed_head_lifts_all_head_aliases_but_not_other_profile_pins():
    profile = _AliasedHeadProfile()

    assert is_lm_head_name("language_model.lm_head", profile)
    assert is_lm_head_name("head.weight", profile)
    assert not is_lm_head_name("router", profile)
    assert remaining_profile_pins(
        profile,
        fixed_lm_head_quantized=True,
    ) == ("router",)


def test_allow_pinned_head_is_the_independent_dp_policy():
    profile = _AliasedHeadProfile()

    assert allow_pinned_lifts_lm_head(profile, "language_model.lm_head")
    assert remaining_profile_pins(
        profile,
        allow_pinned="lm_head",
    ) == ("router",)
    assert remaining_profile_pins(
        profile,
        allow_pinned="router",
    ) == ("lm_head", "head")


def test_exporter_metadata_does_not_reforce_fixed_or_dp_head_to_bf16():
    from prismaquant.export_native_compressed import (
        _bf16_passthrough_for_assignment,
    )

    profile = _AliasedHeadProfile()
    assert _bf16_passthrough_for_assignment(None, profile, {}) == {
        "lm_head", "head", "router"
    }
    assert _bf16_passthrough_for_assignment(
        None,
        profile,
        {"lm_head_mode": "fixed", "lm_head_format": "FP8_E4M3"},
    ) == {"router"}
    assert _bf16_passthrough_for_assignment(
        None,
        profile,
        {"lm_head_mode": "dp", "lm_head_format": "BF16"},
    ) == {"router"}
    # An explicit exporter override remains authoritative.
    assert _bf16_passthrough_for_assignment(
        ["head", "router"],
        profile,
        {"lm_head_mode": "fixed", "lm_head_format": "FP8_E4M3"},
    ) == {"head", "router"}


def test_run_pipeline_wires_head_policy_to_every_native_cache_and_export():
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()

    assert ': "${LM_HEAD_FORMAT:=BF16}"' in script
    assert '--lm-head-format "$LM_HEAD_FORMAT_CANONICAL"' in script
    assert 'LM_HEAD_BASE_COST_ARGS=(--include-lm-head)' in script
    assert 'LM_HEAD_AURA_ARGS=(--include-lm-head)' in script
    assert 'EXPORT_PIN_ARGS=(--ignore "${REMAINING_PROFILE_PINS[@]}")' in script
    assert '"${EXPORT_PIN_ARGS[@]}"' in script

    starts = [
        match.start()
        for match in re.finditer(
            r"python3 -m prismaquant[.]build_production_cache \\",
            script,
        )
    ]
    assert len(starts) == 5
    for start in starts:
        block = script[start:script.find("2>&1 | tee", start)]
        assert '"${PRODUCTION_CACHE_PIN_ARGS[@]}"' in block
        assert '--activation-cache-dir "${WORK_DIR}/act"' in block


def test_head_policy_is_part_of_persisted_cost_and_cache_identity():
    from prismaquant.pipeline import STAGE_SETTINGS_KEYS

    affected = {
        "base-cost",
        "render-cost-cache",
        "render-cost",
        "aura-dw-cache",
        "aura-cost",
        "aura-hybrid-cost",
        "frontier-cache",
        "frontier-recache",
        "production-cache-recached",
        "production-cache-raw",
    }
    for stage in affected:
        sources = {source for _manifest, source in STAGE_SETTINGS_KEYS[stage]}
        assert {
            "LM_HEAD_FORMAT",
            "LM_HEAD_RENDER_ACTIVE",
            "LM_HEAD_DP_UNPINNED",
        } <= sources
