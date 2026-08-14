import re
from pathlib import Path


def _run_pipeline_script() -> str:
    return (
        Path(__file__).resolve().parent.parent / "prismaquant" / "run-pipeline.sh"
    ).read_text()


def _shell_default(script: str, name: str) -> str:
    match = re.search(rf'^\s*:\s*"\$\{{{re.escape(name)}:=([^}}]*)\}}"\s*$',
                      script, re.MULTILINE)
    assert match is not None, f"missing shell default for {name}"
    return match.group(1)


def test_production_recache_default_enabled_after_smoke_ladder():
    script = _run_pipeline_script()

    assert "PRODUCTION_CACHE:=1" in script
    assert "PRODUCTION_RECACHE:=1" in script
    assert "PRODUCTION_RECACHE=0" in script
    assert "PIPELINE_SPEC_PATH:=${WORK_DIR}/artifacts/pipeline_spec.json" in script
    # COST_MODE flipped to `aura` 2026-07-30 (re-vet R2); the flip itself is
    # pinned in tests/test_architecture_doc.py alongside the doc it must match.
    assert "COST_MODE:=aura" in script
    assert "PRODUCTION_CACHE_LEVERS:=gptq,static_act_order,joint_scale_opt" in script
    assert "includes static_act_order" not in script
    assert "production-render-staged|production-render-tail" in script  # exit-2 gate arm
    assert "python3 -m prismaquant.pipeline" in script
    assert "--write-default-production" in script
    # re-vet R11: the spec invocation records the RESOLVED profile, and the
    # allocator only receives --target-profile when one was requested.
    assert "--target-profile \"$TARGET_PROFILE_RESOLVED\"" in script
    assert "--target-profile-default \"$TARGET_PROFILE_DEFAULT\"" in script
    assert "require_lane_supported" in script
    assert ': "${HADAMARD_DUQUANT' not in script
    assert "HADAMARD_DUQUANT:-" in script
    assert "archive/hdq_2026-05-14" in script
    assert "H_DETAIL_DIR" not in script
    assert "LEVER_CACHE_TAG" not in script
    assert "PROD_H_DETAIL_ARGS" not in script


def test_production_render_staged_is_archived_and_blocked():
    """COST_MODE=production-render-staged fails fast (re-vet R17): its own 27B
    result doc improved the last-token-KL screen (0.0232 vs 0.0280) while
    direct WikiText PPL regressed (10.83 vs 8.33) — "Do not ship". See
    archive/production_render_staged_2026-07-30/README.md."""
    script = _run_pipeline_script()

    assert "archive/production_render_staged_2026-07-30" in script
    assert "10.83 vs 8.33" in script
    assert (
        "COST_MODE must be local, production-render-score, or aura" in script
    )
    # The staged execution stages and their knobs are gone.
    assert "--select-tail-output" not in script
    assert "--promotion-qnames-file" not in script
    assert "--bf16-policy" not in script
    assert "PRODUCTION_RENDER_COST_PROMOTE_FRACTION" not in script
    assert "PRODUCTION_RENDER_COST_TAIL_QNAMES" not in script


def test_multi_shot_passes_is_archived_and_blocked():
    """MULTI_SHOT_PASSES>1 fails fast with a pointer to the archive after the
    cross-layer-interaction work landed null. See
    archive/multi_shot_2026-05-19/README.md for the validation record."""
    script = _run_pipeline_script()

    assert "MULTI_SHOT_PASSES" in script
    assert "archive/multi_shot_2026-05-19" in script
    assert ': "${MULTI_SHOT_PASSES' not in script  # no opt-in default; user must explicitly opt out of vanilla


def test_grouped_kl_is_archived_and_blocked():
    """COST_MODE=grouped-kl fails fast with a pointer to the archive after it
    lost the shipped vLLM A/B on Qwen3.6-27B. See
    archive/grouped_kl_2026-05-28/README.md for the validation record."""
    script = _run_pipeline_script()

    # grouped-kl is now a fail-fast dispatch arm pointing at the archive.
    assert "archive/grouped_kl_2026-05-28" in script
    # It is no longer advertised as a valid COST_MODE in the catch-all error.
    assert (
        "COST_MODE must be local, production-render-score, or aura"
        in script
    )
    assert "grouped-kl" not in script.split("COST_MODE must be", 1)[1].split(
        "\n", 1)[0]
    # The grouped-kl measurement invocation and its env knobs are gone.
    assert "prismaquant.grouped_kl_cost" not in script
    assert "GROUPED_KL_NSAMPLES" not in script
    assert "GROUPED_KL_MAX_LANES" not in script
    # production-render-score remains an ACCEPTED mode (it is the explicit /
    # legacy spelling since the R2 default flip), which is what makes the
    # grouped-kl arm's "use production-render-score" advice honest.
    assert "production-render-score|production-render)" in script


def test_mse_promotion_is_archived_and_blocked():
    """MSE_PROMOTION fails fast with a pointer to the archive (re-vet R18).
    The post-frontier local-MSE rewrite lost to both the shipped 4.75 artifact
    and the 5.16 kneedle on 35B and is superseded by the AURA cost. See
    archive/mse_promotion_2026-07-30/README.md."""
    script = _run_pipeline_script()

    assert "archive/mse_promotion_2026-07-30" in script
    assert ': "${MSE_PROMOTION' not in script  # no opt-in default survives
    assert "build_mse_promotion_assignment" not in script
    assert "layer_config_before_mse_promotion" in script  # cited in the gate text
    assert "MSE_PROMOTION_TARGET_BPP" not in script


def test_production_cache_union_is_archived_and_blocked():
    """PRODUCTION_CACHE_UNION fails fast with a pointer to the archive
    (re-vet R18): the smart-union render pre-decided which Linears deserved an
    FP8 rung from a surrogate percentile. See
    archive/union_cache_2026-07-30/README.md."""
    script = _run_pipeline_script()

    assert "archive/union_cache_2026-07-30" in script
    assert ': "${PRODUCTION_CACHE_UNION' not in script
    assert "tools.build_union_cache" not in script


def test_core_recipe_defaults_are_pinned():
    script = _run_pipeline_script()

    assert _shell_default(script, "FORMATS") == "NVFP4,FP8_DYNAMIC,BF16"
    assert _shell_default(script, "TARGET_BITS") == "4.75"
    # re-vet R1: SELECTION_MODE is conditional — surrogate by default,
    # validated-surrogate under a byte budget (an explicit value wins).
    assert ': "${SELECTION_MODE:=surrogate}"' in script
    assert ': "${SELECTION_MODE:=validated-surrogate}"' in script
    assert ': "${TARGET_DISK_GB:=}"' in script
    assert ': "${ARTIFACT_OVERHEAD_RESERVE_BYTES:=}"' in script
    assert '--artifact-overhead-reserve-bytes "$ARTIFACT_OVERHEAD_RESERVE_BYTES"' in script
    assert "TARGET_DISK_GB requires ARTIFACT_OVERHEAD_RESERVE_BYTES" in script
    assert ': "${VALIDATED_FRONTIER_PICK:=budget}"' in script


def test_learned_cb_pipeline_is_blocked_before_production_stages():
    script = _run_pipeline_script()

    gate = "learned CB is research-only until one immutable value-bearing"
    assert gate in script
    keyed_gate = "per-family CB_CODEBOOK_SOURCE is prerequisite-only"
    assert keyed_gate in script
    assert script.index(keyed_gate) < script.index("python3 -m prismaquant.allocator")
    assert script.index(gate) < script.index("python3 -m prismaquant.allocator")
    assert "Learned production is accepted" not in script
