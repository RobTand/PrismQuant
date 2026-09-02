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


def test_production_render_score_is_unlicensed_on_a_cb_menu():
    """COST_MODE=production-render-score fails fast on any CB/CBL menu.

    Its score field is `weight_mse` (audit M6), and the per-unit factorization
    mse(e,K) ~= s_e * g(K) FAILS in weight currency across a codebook-basis
    change: CV over experts of weight_mse_CBL/weight_mse_lattice is monotone in
    rung, 0.088 (K28) -> 0.224 (K48), 8 of 10 rung-pairs breaching the 0.10
    bar, while lattice->lattice on the same planes passes at 0.067/0.056.
    Allocating a CB menu on that estimator allocates in the currency that does
    not transfer.
    """
    script = _run_pipeline_script()

    assert "unlicensed on a CB/CBL menu" in script
    # The evidence travels with the guard, so the refusal is auditable.
    assert "0.088" in script and "0.224" in script
    # The escape hatch stays honest: it is still valid off CB menus.
    assert "reproducing pre-CB artifacts on non-CB menus" in script


# RETIRED 2026-09-02 with the Gridbook codebook lane
# (archive/gridbook_lane_2026-09-02/): two tests that drove the CB export gate
# (`test_cb_export_gate_accepts_inherited_lane_and_wires_strict_producer_policy`,
# `test_cb_activation_scope_is_validated_exported_and_stage_bound`). Their
# subject -- `EXPORT_CONTAINER=nvfp4_cb`'s lane inheritance, strict producer
# policy and CB_ACTIVATION_SCOPE plumbing -- no longer exists; that container
# now `exit 2`s. The guard below survives because its FORMATS limb does.


# RETIRED 2026-09-02 with the Gridbook codebook lane
# (archive/gridbook_lane_2026-09-02/). Every gate these tests executed --
# the CB learned-bundle trainer-version enum and its four v2 preconditions,
# CB_ACTIVATION_SCOPE, the three `EXPORT_CONTAINER=nvfp4_cb` preconditions,
# and the three CB producer-policy resolutions -- lived behind that container,
# which now `exit 2`s before any of them is reached. They are deleted, not
# skipped: a gate test for a gate that cannot be reached asserts nothing.


def test_cb_unlicensed_guard_actually_fires():
    """Execute the guard's real predicate; a gate never seen firing is not a gate.

    Text assertions alone would only prove the string exists -- the exact
    guard-scope failure this repo keeps paying for. So pull the condition out
    of the shipped script and evaluate it under both CB signals and both
    non-CB controls.
    """
    import subprocess

    path = (
        Path(__file__).resolve().parent.parent / "prismaquant" / "run-pipeline.sh"
    )
    cond = None
    for line in path.read_text().splitlines():
        if 'FORMATS:-}" == *_CB_*' in line:
            cond = line.strip().removeprefix("if ").removesuffix("; then")
            break
    assert cond is not None, "CB-unlicensed guard condition not found in script"

    def fires(export_container: str, formats: str) -> bool:
        proc = subprocess.run(
            ["bash", "-c", f"if {cond}; then exit 7; else exit 0; fi"],
            env={
                "PATH": "/usr/bin:/bin",
                "EXPORT_CONTAINER": export_container,
                "FORMATS": formats,
            },
            check=False,
        )
        assert proc.returncode in (0, 7), proc.returncode
        return proc.returncode == 7

    # The FORMATS signal must trip it. This is the limb that still matters:
    # the CB format/cost/render plumbing outlived the Gridbook lane (D34), so a
    # `*_CB_*` menu is still nameable and still mis-priced by this estimator.
    assert fires("compressed-tensors", "FP8_CB_K28,FP8_CB_K43")
    # The EXPORT_CONTAINER signal is GONE from this guard as of 2026-09-02, and
    # its absence is the correct state, not drift: `EXPORT_CONTAINER=nvfp4_cb`
    # is now refused outright by the container gate before any cost mode is
    # considered (archive/gridbook_lane_2026-09-02/). A second refusal for the
    # same input would be dead code pretending to be a gate.
    assert not fires("nvfp4_cb", "NVFP4,FP8_DYNAMIC,BF16")
    # ...and neither control may.
    assert not fires("compressed-tensors", "NVFP4,FP8_DYNAMIC,BF16")
    assert not fires("", "")

    # So prove the container gate is what refuses it now, by executing that
    # gate's own predicate the same way.
    container_cond = None
    for line in path.read_text().splitlines():
        if '"$EXPORT_CONTAINER" == "nvfp4_cb"' in line and line.strip().startswith("if "):
            container_cond = line.strip().removeprefix("if ").removesuffix("; then")
            break
    assert container_cond is not None, "retired-container gate not found in script"
    proc = subprocess.run(
        ["bash", "-c", f"if {container_cond}; then exit 7; else exit 0; fi"],
        env={"PATH": "/usr/bin:/bin", "EXPORT_CONTAINER": "nvfp4_cb"},
        check=False,
    )
    assert proc.returncode == 7


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


def test_aura_streaming_identity_path_is_explicit_and_stage_bound():
    from prismaquant.pipeline import STAGE_SETTINGS_KEYS

    script = _run_pipeline_script()

    assert _shell_default(script, "AURA_COST_STREAMING") == "0"
    assert _shell_default(script, "AURA_COST_CHECKPOINT_DIR") == ""
    assert "AURA_COST_STREAMING=1 requires an absolute" in script
    assert '--checkpoint-dir "$AURA_COST_CHECKPOINT_DIR"' in script
    assert '"AURA_COST_STREAMING=$AURA_COST_STREAMING"' in script
    assert '"AURA_COST_CHECKPOINT_DIR=$AURA_COST_CHECKPOINT_DIR"' in script
    aura_cost_sources = {
        source for _manifest, source in STAGE_SETTINGS_KEYS["aura-cost"]
    }
    assert "AURA_COST_STREAMING" in aura_cost_sources
    assert "AURA_COST_CHECKPOINT_DIR" in aura_cost_sources


def test_aura_streaming_also_streams_and_resumes_empirical_expert_tail():
    from prismaquant.pipeline import STAGE_SETTINGS_KEYS

    script = _run_pipeline_script()
    expert_block = script.split("AURA_EXPERT_EXECUTION_ARGS=(", 1)[1].split(
        "# [2b]", 1
    )[0]
    invocation = script.split(
        "python3 -m prismaquant.expert_empirical_cost", 1
    )[1].split('tee "${WORK_DIR}/logs/expert_empirical_cost.log"', 1)[0]

    assert "--streaming" in expert_block
    assert (
        '--checkpoint-dir "${AURA_COST_CHECKPOINT_DIR%/}/'
        'expert-empirical-cost"'
    ) in expert_block
    assert "--resume" in expert_block
    assert "AURA_EXPERT_EXECUTION_ARGS" in invocation
    hybrid_sources = {
        source for _manifest, source in STAGE_SETTINGS_KEYS["aura-hybrid-cost"]
    }
    assert "AURA_COST_STREAMING" in hybrid_sources
    assert "AURA_COST_CHECKPOINT_DIR" in hybrid_sources


def test_large_or_moe_validated_frontier_has_pre_gpu_inplace_gate():
    script = _run_pipeline_script()
    gate = script.split(
        'if [[ "$SELECTION_MODE" == "validated-surrogate" ]]', 1
    )[1].split("# COST_MODE=aura settings", 1)[0]

    assert "--check-frontier-materialization" in gate
    assert '--frontier-materialization "$VALIDATED_FRONTIER_MATERIALIZATION"' in gate
    assert "|| exit $?" in gate


def test_prismasnap_source_is_additive_native_only_and_non_native_fails_closed():
    script = _run_pipeline_script()
    gate = script.split(': "${EXPORT_CONTAINER:=compressed-tensors}"', 1)[1].split(
        'if [[ "$EXPORT_CONTAINER" == "nvfp4_cb" ]]', 1
    )[0]
    assert "prismasnap_provenance.json" in gate
    assert '!= "compressed-tensors"' in gate
    assert "exit 2" in gate


