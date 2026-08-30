from pathlib import Path


DRIVER = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/e2m1_highrate.py"
)


def test_active_corpus_loader_is_isolated_from_locked_hull_package():
    source = DRIVER.read_text()

    assert "load_active_glm_corpus(REPO_ROOT, args.glm_manifest)" in source
    assert "from prismaquant.trellis_bf16_corpus import" not in source
