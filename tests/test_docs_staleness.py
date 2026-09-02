import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_sample_parallel_runbook_is_fail_closed_and_cross_host_portable():
    doc = _read("docs/design/sample_parallel_probe.md")
    assert "set -euo pipefail" in doc
    assert "SP_IMAGE_DIGEST=\"${SP_IMAGE_REF##*@}\"" in doc
    assert "{{range .RepoDigests}}{{println .}}{{end}}" in doc
    assert "SP_IMAGE_ID" not in doc
    assert "--entrypoint /bin/bash \"$SP_IMAGE_REF\"" in doc
    assert "--user \"$SP_HOST_UID:$SP_HOST_GID\"" in doc
    assert "--workdir /worker-state/tmp" in doc
    assert "--expected-closure-sha256 \"$SP_CLOSURE\" >/dev/null || return 1" in doc
    assert "PRISMAQUANT_CB_ENCODE_COMPILE=1" in doc
    assert "PRISMAQUANT_CB_ATOM_COMPILE=1" in doc
    assert "PRISMAQUANT_CB_COMPILE_FAIL_CLOSED=1" in doc
    for mount in ("/model", "/dataset", "/run", "/worker-state"):
        assert mount in doc


def test_runtime_flags_doc_owns_policy_without_mirroring_plugin_flags():
    """Renamed from `..._without_mirroring_gridbook_flags` on 2026-09-02.

    The rule is not about Gridbook: a serving runtime's own selector names and
    defaults are documented by that runtime and pinned by its immutable pin,
    and PrismaQuant's flag doc must point at the pin rather than mirror the
    flags (AGENTS.md principle 5). The Gridbook pin retired with its lane
    (archive/gridbook_lane_2026-09-02/); the Tessera pin now carries the same
    duty, so the assertion moves rather than disappearing. The
    `PRISMAQUANT_CB_*` runtime selectors below stay in the forbidden list for
    the same reason they always were: they belong to a plugin, not to us.
    """
    doc = _read("docs/design/runtime_flags.md")
    assert "There is deliberately no hand-maintained exhaustive flag list" in doc
    assert "prismaquant/tessera_runtime/tessera_serving_runtime_pin.json" in doc
    assert "rg -o 'PRISMAQUANT_[A-Z0-9_]+' prismaquant scripts tools" in doc
    for external_runtime_flag in (
        "PRISMAQUANT_CB_DECODE",
        "PRISMAQUANT_CB_DISPATCH",
        "PRISMAQUANT_CB_EXPAND",
        "PRISMAQUANT_CB_PREFILL",
        "PRISMAQUANT_PRELOAD_FUSED",
    ):
        assert external_runtime_flag not in doc
    assert "| `PRODUCTION_RENDER_COST_SCORE_FIELD` | `weight_mse` |" in doc
    assert "`joint_mse` is the production JSO scale rule" in doc
    assert "H_DETAIL_DIR" not in doc


def test_package_readme_entrypoints_resolve_to_live_modules():
    text = _read("prismaquant/README.md")
    start = text.index("## CLI entrypoints")
    end = text.index("## Archive")
    modules = [
        f"prismaquant.{name}"
        for name in re.findall(r"`([a-z][a-z0-9_]+)`", text[start:end])
    ]
    assert modules
    assert "prismaquant.polish_from_assignment" not in modules
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    assert not missing
    assert "dated `archive/` walls" in text


def test_root_readme_architecture_status_matches_in_tree_profiles():
    text = _read("README.md")
    assert "DeepSeek-V4-Flash" in text and "vendored transformer" in text
    assert "**Gemma4**" in text
    assert "**LFM2.5**" in text
    assert "GLM-4" not in text
    assert "waiting on `transformers` class" not in text
    assert "blocked on transformers" not in text


def test_audit_notes_are_not_root_level_and_scratch_is_local_only():
    assert not (ROOT / "audit_findings.md").exists()
    assert not (ROOT / "audit_questions.md").exists()
    assert (ROOT / "docs/audits/audit_findings_2026-05-22.md").exists()
    assert (ROOT / "docs/audits/audit_questions_2026-05-22.md").exists()
    assert not (ROOT / "scratch/smoke_graph_memory.py").exists()
    assert (ROOT / "tools/smoke_graph_memory.py").exists()


def test_claude_does_not_overstate_pipeline_enforcement():
    text = _read("CLAUDE.md")

    assert "structurally enforced\n   (`pipeline.py` `APPROVED_RESOURCE_OWNERS`)" not in text
    assert "declarative spec + owner validation, not executor" in text
    assert "runtime enforcement lives in the stage code" in text
