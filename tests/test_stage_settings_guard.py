"""pipeline.py as the settings-hash authority (re-vet R5, debt D6).

The guard's contract, in one place:
  * artifact absent            -> record this stage's key set, exit 0
  * recorded projection equal  -> exit 0
  * recorded projection differs-> exit 2, naming every diff
  * no record for this stage   -> WARN and record (pre-guard artifacts are
                                  never invalidated)
Plus the property that made R5 worth doing: WHICH keys an artifact depends on
is declared once, in `STAGE_SETTINGS_KEYS`, not re-decided at every call site.
"""

import json
import re
from pathlib import Path

from prismaquant import pipeline

ROOT = Path(__file__).resolve().parents[1]


def _document(**settings):
    return pipeline.stage_settings_document(settings)


def _full_settings():
    """Every source name any stage declares, with a placeholder value."""
    sources = {
        source
        for keys in pipeline.STAGE_SETTINGS_KEYS.values()
        for _manifest_key, source in keys
    }
    return {name: f"<{name}>" for name in sources}


def test_every_declared_stage_projects_cleanly_from_the_full_settings():
    doc = _document(**_full_settings())
    assert doc["unresolved"] == {}
    assert set(doc["artifacts"]) == set(pipeline.STAGE_SETTINGS_KEYS)
    for stage, keys in pipeline.STAGE_SETTINGS_KEYS.items():
        assert set(doc["artifacts"][stage]) == {mk for mk, _ in keys}


def test_run_pipeline_supplies_every_declared_setting():
    """The shell must supply a value for every source pipeline.py declares,
    either in STAGE_SETTINGS_ENV or as a late override at the call site. A
    declared-but-unsupplied key is what the guard hard-fails on, so pin it."""
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    block = script.split("STAGE_SETTINGS_ENV=(", 1)[1].split("\n)", 1)[0]
    supplied = set(re.findall(r'"([A-Z0-9_]+)=', block))
    # RENDER_ENV_SETTINGS is spliced in by reference.
    render_block = script.split("RENDER_ENV_SETTINGS=(", 1)[1].split("\n)", 1)[0]
    supplied |= set(re.findall(r'"([A-Z0-9_]+)=', render_block))
    # Late-computed values passed as overrides at their call sites.
    supplied |= set(re.findall(r'require_stage_settings [^\n]*\\?\n?\s*"([A-Z0-9_]+)=',
                               script))
    declared = {
        source
        for keys in pipeline.STAGE_SETTINGS_KEYS.values()
        for _mk, source in keys
    }
    missing = sorted(declared - supplied)
    assert not missing, (
        f"run-pipeline.sh supplies no value for {missing}; the guard would "
        "exit 2 on a partial key set")


def test_every_guarded_stage_id_in_the_shell_is_declared():
    script = (ROOT / "prismaquant" / "run-pipeline.sh").read_text()
    used = set(re.findall(r'require_stage_settings "[^"]+" ([a-z0-9-]+)', script))
    assert used, "no guard call sites found"
    unknown = sorted(used - set(pipeline.STAGE_SETTINGS_KEYS))
    assert not unknown, f"undeclared settings-hash stages: {unknown}"


def test_absent_artifact_records_then_matches(tmp_path):
    doc = _document(MODEL_PATH="m", DATASET="d", NSAMPLES="8", SEQLEN="512",
                    CALIBRATION_MODALITY="text-only")
    artifact = tmp_path / "probe.pkl"
    code, _ = pipeline.check_stage_settings(artifact, "probe", doc)
    assert code == 0
    manifest = json.loads((tmp_path / "probe.pkl.settings.json").read_text())
    assert manifest["stages"]["probe"]["NSAMPLES"] == "8"

    artifact.write_bytes(b"x")
    code, messages = pipeline.check_stage_settings(artifact, "probe", doc)
    assert code == 0 and messages == []


def test_changed_setting_is_exit_2_naming_the_diff(tmp_path):
    base = dict(MODEL_PATH="m", DATASET="d", NSAMPLES="8", SEQLEN="512",
                CALIBRATION_MODALITY="text-only")
    artifact = tmp_path / "probe.pkl"
    pipeline.check_stage_settings(artifact, "probe", _document(**base))
    artifact.write_bytes(b"x")

    changed = dict(base, NSAMPLES="32")
    code, messages = pipeline.check_stage_settings(
        artifact, "probe", _document(**changed))
    assert code == 2
    joined = "\n".join(messages)
    assert "NSAMPLES" in joined and "'8'" in joined and "'32'" in joined
    assert "refusing silent reuse" in joined


def test_missing_manifest_only_warns(tmp_path):
    artifact = tmp_path / "probe.pkl"
    artifact.write_bytes(b"x")
    code, messages = pipeline.check_stage_settings(
        artifact, "probe", _document(MODEL_PATH="m", DATASET="d", NSAMPLES="8",
                                     SEQLEN="512", CALIBRATION_MODALITY="t"))
    assert code == 0
    assert "WARNING" in messages[0] and "no settings manifest" in messages[0]


def test_legacy_flat_manifest_still_guards_its_stage(tmp_path):
    """Pre-R5 manifests are a flat {key: value} dict. Same key set -> still
    compared (a changed setting must still fail); different stage -> not
    invalidated."""
    artifact = tmp_path / "probe.pkl"
    artifact.write_bytes(b"x")
    (tmp_path / "probe.pkl.settings.json").write_text(json.dumps({
        "MODEL_PATH": "m", "DATASET": "d", "NSAMPLES": "8", "SEQLEN": "512",
        "CALIBRATION_MODALITY": "text-only",
    }))
    same = _document(MODEL_PATH="m", DATASET="d", NSAMPLES="8", SEQLEN="512",
                     CALIBRATION_MODALITY="text-only")
    assert pipeline.check_stage_settings(artifact, "probe", same)[0] == 0
    drift = _document(MODEL_PATH="m", DATASET="OTHER", NSAMPLES="8",
                      SEQLEN="512", CALIBRATION_MODALITY="text-only")
    assert pipeline.check_stage_settings(artifact, "probe", drift)[0] == 2


def test_two_stages_can_own_one_path(tmp_path):
    """COST_MODE=aura + validated-surrogate points the AURA dW cache and the
    frontier cache at the SAME file; both key sets must coexist."""
    settings = _full_settings()
    doc = _document(**settings)
    artifact = tmp_path / "cache.pkl"
    assert pipeline.check_stage_settings(artifact, "aura-dw-cache", doc)[0] == 0
    artifact.write_bytes(b"x")
    code, messages = pipeline.check_stage_settings(artifact, "frontier-cache", doc)
    assert code == 0 and any("predates this stage's" in m for m in messages)
    stored = json.loads((tmp_path / "cache.pkl.settings.json").read_text())
    assert set(stored["stages"]) == {"aura-dw-cache", "frontier-cache"}
    # …and both keep guarding independently.
    assert pipeline.check_stage_settings(artifact, "aura-dw-cache", doc)[0] == 0


def test_unsupplied_declared_key_is_a_hard_stop(tmp_path):
    doc = _document(MODEL_PATH="m")  # DATASET/NSAMPLES/... missing
    code, messages = pipeline.check_stage_settings(
        tmp_path / "probe.pkl", "probe", doc)
    assert code == 2
    assert "no value was supplied" in messages[0]


def test_late_override_resolves_a_missing_key(tmp_path):
    doc = _document(MODEL_PATH="m", DATASET="d", NSAMPLES="8", SEQLEN="512",
                    FORMATS="NVFP4", TARGET_BITS="4.75",
                    LM_HEAD_FORMAT="BF16", LM_HEAD_RENDER_ACTIVE="0",
                    LM_HEAD_DP_UNPINNED="0",
                    PRISMAQUANT_NVFP4_SCALE_RULE="", PRISMAQUANT_GPTQ_DAMP_SWEEP="0",
                    PRISMAQUANT_GPTQ_DAMP="", PRISMAQUANT_ACT_CLIP_QUANTILE="0.999",
                    PRODUCTION_CACHE_LEVERS="gptq", PRODUCTION_CACHE_DISABLE_LEVERS="",
                    CB_SCALE_CODING="", CB_CODEBOOK_SOURCE="",
                    CB_CODEBOOK_SOURCE_SCOPE="", CB_CODEBOOK_BUNDLE="",
                    CB_ROUTED_MOE_BOOK_SELECTION="",
                    CB_ROUTED_MOE_BOOK_SELECTION_SHA256="",
                    CB_SCALE_SWEEP="1", CB_SCALE_SWEEP_SCOPE="",
                    PRISMAQUANT_CB_LDLQ="0",
                    PRISMAQUANT_CB_MINCHAIN="0",
                    PRISMAQUANT_CB_MINCHAIN_ANCHORS="",
                    PRISMAQUANT_CB_MINCHAIN_HOLDBACKS="",
                    PRISMAQUANT_CB_MINCHAIN_AUDIT_SEED="42",
                    PRISMAQUANT_CB_MINCHAIN_BACKSTOP="0.25",
                    PRISMAQUANT_CB_MINCHAIN_AUDIT_MEDIAN="0.05",
                    PRISMAQUANT_CB_MINCHAIN_AUDIT_P95="0.15",
                    PRISMAQUANT_CB_ENCODE_TIER="balanced")
    artifact = tmp_path / "recached.pkl"
    assert doc["unresolved"]["production-cache-recached"] == ["ASSIGNMENT_DIGEST"]
    code, _ = pipeline.check_stage_settings(
        artifact, "production-cache-recached", doc,
        overrides={"ASSIGNMENT_DIGEST": "deadbeef"})
    assert code == 0
    stored = json.loads((tmp_path / "recached.pkl.settings.json").read_text())
    assert stored["stages"]["production-cache-recached"]["ASSIGNMENT_DIGEST"] == "deadbeef"


def test_approved_resource_owners_name_real_implementations():
    """D10: two of the three owner names were never implemented anywhere."""
    owners = {o for names in pipeline.APPROVED_RESOURCE_OWNERS.values()
              for o in names}
    assert owners == {"ProductionWeightCache", "PerturbedActivationCache",
                      "LayerCache"}
    for owner in owners:
        hits = list((ROOT / "prismaquant").rglob("*.py"))
        assert any(f"class {owner}" in p.read_text(encoding="utf-8")
                   for p in hits), f"{owner} has no implementation in the tree"
