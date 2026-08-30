from pathlib import Path


DRIVER = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/e2m1_highrate.py"
)


def test_active_corpus_loader_is_isolated_from_locked_hull_package():
    source = DRIVER.read_text()

    assert "load_active_glm_corpus(REPO_ROOT, args.glm_manifest)" in source
    assert "from prismaquant.trellis_bf16_corpus import" not in source


def test_glm_high_rate_plan_is_explicit_and_does_not_replace_default():
    source = DRIVER.read_text()

    assert '"scaffold": BF16_RATES' in source
    assert '"high": NEW_RATES' in source
    assert 'default="scaffold"' in source
    assert 'rate_plan = GLM_RATE_PLANS[args.glm_rate_plan]' in source
    assert '"glm_rate_plan": args.glm_rate_plan' in source
