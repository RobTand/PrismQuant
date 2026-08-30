from pathlib import Path


DRIVER = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/e2m1_highrate.py"
)


def test_active_corpus_loader_is_isolated_from_locked_hull_package():
    source = DRIVER.read_text()

    assert "def _load_active_glm_corpus(manifest: Path):" in source
    assert 'package_name = "_prismaquant_active_glm_corpus"' in source
    assert "module.load_finalized_bf16_corpus(manifest)" in source
    assert "from prismaquant.trellis_bf16_corpus import" not in source
