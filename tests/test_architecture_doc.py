"""Mechanical half of the ARCHITECTURE.md maintenance contract (CLAUDE.md §4
principle 13, AGENTS.md rule 10): the master document's defaults table must
match `prismaquant/run-pipeline.sh`, and its structural anchors must exist.

The judgment half — prose describing behavior that changed — cannot be tested;
this file only makes silent drift of the enumerable facts impossible.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _doc() -> str:
    return (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")


def _pipeline() -> str:
    return (ROOT / "prismaquant" / "run-pipeline.sh").read_text(encoding="utf-8")


def _shell_default(script: str, name: str) -> str:
    m = re.search(rf"\$\{{{name}:=([^}}]*)\}}", script)
    assert m, f"{name} has no ':=' default in run-pipeline.sh"
    return m.group(1)


def test_architecture_doc_exists_with_provenance_stamp():
    doc = _doc()
    assert doc.startswith("# PrismaQuant Architecture")
    assert re.search(r"As of: \d{4}-\d{2}-\d{2}", doc), "provenance stamp missing"
    assert "## 0. Maintenance contract" in doc


def test_defaults_table_matches_run_pipeline():
    doc, script = _doc(), _pipeline()
    for var in (
        "FORMATS",
        "TARGET_BITS",
        "COST_MODE",
        "NSAMPLES",
        "SEQLEN",
        "PRODUCTION_CACHE_LEVERS",
    ):
        val = _shell_default(script, var)
        assert f"{var}={val}" in doc, (
            f"ARCHITECTURE.md §3.3 is stale: run-pipeline.sh has {var}={val}. "
            "Update the defaults table in the same commit as the default change."
        )


def test_target_profile_has_no_shell_default():
    """re-vet R11: TARGET_PROFILE must stay UNSET so the architecture's own
    `spec.default_serving_profile` can win. A `:=` default here silently beat
    every spec (measured: 226 Hy3 FP8 Linears -> BF16, 2026-07-11), so this
    pins the absence of one and requires the doc to say how it resolves."""
    script, doc = _pipeline(), _doc()
    assert re.search(r'\$\{TARGET_PROFILE:=\}', script), (
        "TARGET_PROFILE must have an EMPTY ':=' default in run-pipeline.sh; "
        "an architecture's spec.default_serving_profile can never win against "
        "an explicit request (serving_profiles.resolve_target_profile)."
    )
    assert f"TARGET_PROFILE_DEFAULT={_shell_default(script, 'TARGET_PROFILE_DEFAULT')}" in doc
    assert "spec-resolved" in doc, (
        "ARCHITECTURE.md §3.3 must document that TARGET_PROFILE is "
        "spec-resolved rather than shell-defaulted."
    )


def test_selection_mode_default_documented():
    """SELECTION_MODE is no longer a single ':=' default — it is surrogate,
    or validated-surrogate under a byte budget (re-vet R1)."""
    script, doc = _pipeline(), _doc()
    assert 'SELECTION_MODE:=validated-surrogate' in script
    assert 'SELECTION_MODE:=surrogate' in script
    assert "SELECTION_MODE=surrogate" in doc
    assert "TARGET_DISK_GB" in doc


def test_cost_mode_default_is_aura_with_the_legacy_mode_still_reachable():
    """re-vet R2: the default flipped to `aura` on 2026-07-30. Both halves are
    pinned — the flip itself (so it cannot silently revert) and the continued
    reachability of `production-render-score`, which is how every pre-flip
    artifact reproduces."""
    script, doc = _pipeline(), _doc()
    assert _shell_default(script, "COST_MODE") == "aura"
    assert "production-render-score|production-render)" in script, (
        "production-render-score must stay an accepted COST_MODE: it is the "
        "explicit spelling that reproduces every pre-2026-07-30 artifact."
    )
    assert "COST_MODE=aura" in doc and "explicit/legacy" in doc


def test_cost_axes_are_declared_with_back_compat_aliases():
    """re-vet R3: COST_MODE is a spelling over (COST_RENDER x COST_OBJECTIVE),
    and the three documented values keep their exact meanings."""
    script, doc = _pipeline(), _doc()
    for pair, mode in (
        ('"inline|weight-recon")', "local"),
        ('"cached-menu|render-score")', "production-render-score"),
        ('"cached-menu|aura-adjoint")', "aura"),
    ):
        assert pair in script and mode in script
    # The two unimplemented pairs must stop with a reason, not fall through.
    assert '"inline|aura-adjoint")' in script
    assert '"cached-menu|weight-recon")' in script
    assert "COST_RENDER" in doc and "COST_OBJECTIVE" in doc


def test_additivity_gate_default_is_measure():
    """Ruled 2026-07-30 (R2 residue): every AURA-default run measures a residual.

    `auto` reported only from a KL the run happened to have, so under
    `SELECTION_MODE=surrogate` an artifact carried a prediction and no residual
    — AURA's structural assumption stayed a two-model memory. `measure` costs
    one bounded end-KL eval and buys a per-artifact number.
    """
    script, doc = _pipeline(), _doc()
    assert _shell_default(script, "AURA_ADDITIVITY_GATE") == "measure"
    assert "AURA_ADDITIVITY_GATE=measure" in doc
    # auto and off must stay selectable (the report is never mandatory GPU work
    # for someone who explicitly does not want it).
    assert '"$AURA_ADDITIVITY_GATE" != "0"' in script
    assert '"$AURA_ADDITIVITY_GATE" == "measure"' in script


def test_tail_veto_default_on_with_kl_max_is_documented():
    """R9/D1 ruled 2026-07-30: default-on, `kl_max` contract, derived eta."""
    from prismaquant.select_validated_frontier import (
        DEFAULT_TAIL_ETA,
        DEFAULT_TAIL_VETO,
    )
    doc = _doc()
    assert DEFAULT_TAIL_VETO == "kl_max"
    assert DEFAULT_TAIL_ETA == "auto"
    assert "DEFAULT-ON, contract statistic `kl_max`" in doc
    assert "--tail-eta` defaults to `auto`" in doc


def test_cb_defaults_match_the_shipped_drivers():
    """D15: a default no shipped driver uses documents an unvalidated path.
    Pinned against the drivers themselves so the two cannot drift again."""
    script = _pipeline()
    assert _shell_default(script, "CB_EXPERT_EMPIRICAL") == "0"
    assert _shell_default(script, "CB_SCALE_CODING") == "two_tier"
    # PRISMAQUANT_CB_LDLQ default is now via truth table (neither set -> 0/none),
    # factored into prismaquant/cb_ldlq_normalize.sh — the single source of truth
    # that run-pipeline.sh sources. Check the helper for the canonical defaults,
    # and that run-pipeline.sh actually sources it.
    helper = (ROOT / "prismaquant" / "cb_ldlq_normalize.sh").read_text(encoding="utf-8")
    assert 'PRISMAQUANT_CB_LDLQ=0' in helper
    assert 'PRISMAQUANT_CB_LDLQ_SCOPE="none"' in helper
    assert "cb_ldlq_normalize.sh" in script and "normalize_cb_ldlq_vars" in script
    assert _shell_default(script, "PRISMAQUANT_CB_MINCHAIN") == "0"
    assert "PRISMAQUANT_CB_LDLQ=0" in _doc()
    assert "PRISMAQUANT_CB_MINCHAIN=0" in _doc()
    drivers = [
        ROOT / "scripts" / name for name in (
            "run_hy3_prod_nvfp4cb.sh",
            "run_hy3_prod_joint.sh",
            "run_35b_prod_nvfp4cb.sh",
            "run_laguna_s21_prod.sh",
        )
    ]
    for driver in drivers:
        text = driver.read_text(encoding="utf-8")
        assert "export CB_EXPERT_EMPIRICAL=0" in text, driver.name


def test_three_diagrams_present():
    assert _doc().count("```mermaid") == 3


def test_docs_index_leads_with_architecture():
    readme = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "ARCHITECTURE.md" in readme.split("\n\n")[0] or "ARCHITECTURE.md" in readme[:500]


def test_normative_rule_files_reference_the_contract():
    for name in ("CLAUDE.md", "AGENTS.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "docs/ARCHITECTURE.md" in text, f"{name} lost the doc-sync rule"


def test_every_archive_wall_has_a_banner_readme():
    walls = [p for p in (ROOT / "archive").iterdir() if p.is_dir()]
    assert walls
    missing = [w.name for w in walls if not (w / "README.md").exists()]
    assert not missing, f"archive walls without a banner README: {missing}"
