"""Doc/driver consistency for issues #164, #166, #167.

Each test derives the fact from the code that owns it (run-pipeline.sh, the
archive walls, the Tessera lane arm) and then asserts the document says the
same thing — never a re-typed roster either side could outgrow.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _shell_assign_defaults(script: str, name: str) -> list[str]:
    """Every `: "${NAME:=default}"` value in file order (usually one; the
    driver legitimately assigns SELECTION_MODE twice, once per branch)."""
    return re.findall(rf': "\$\{{{name}:=([^}}]*)\}}"', script)


def _defaults_block_claims(doc: str) -> dict[str, str]:
    """VAR=value claims from CLAUDE.md's shell-defaults block.

    Only backticked spans inside the block, whitespace-split (the block packs
    `NSAMPLES=32 SEQLEN=1024` into one span). Spans without `=` (`WORK_DIR`,
    `validated-surrogate`, `${VAR:-}`) are not claims and are ignored. The two
    anchors only locate the parse region — they assert nothing about wording;
    if either ever goes missing the test errors loudly instead of passing.
    """
    start = doc.index("Defaults in `run-pipeline.sh` today:")
    end = doc.index("The archived cost modes", start)
    claims: dict[str, str] = {}
    for span in re.findall(r"`([^`]+)`", doc[start:end]):
        for token in span.split():
            m = re.fullmatch(r"([A-Z][A-Z0-9_]*)=(\S+)", token)
            if m:
                claims[m.group(1)] = m.group(2)
    assert claims, "CLAUDE.md shell-defaults block has no parseable VAR=value claims"
    return claims


def test_164_claude_shell_defaults_match_driver():
    """#164: CLAUDE.md must not promise TARGET_PROFILE / CB shell defaults the
    driver removed. TARGET_PROFILE is deliberately unset (re-vet R11); the CB
    knobs lost shell defaults with the Gridbook lane (2026-09-02)."""
    script = _read("prismaquant/run-pipeline.sh")
    doc = _read("CLAUDE.md")

    # Driver side, derived — not re-typed.
    assert _shell_assign_defaults(script, "TARGET_PROFILE") == [""], (
        "driver changed: TARGET_PROFILE is no longer deliberately unset; "
        "update this test and both docs"
    )
    assert _shell_assign_defaults(script, "TARGET_PROFILE_DEFAULT") == ["vllm_packed_moe"]
    assert _shell_assign_defaults(script, "CB_EXPERT_EMPIRICAL") == [], (
        "driver changed: CB_EXPERT_EMPIRICAL regained a shell default"
    )
    assert _shell_assign_defaults(script, "CB_SCALE_CODING") == [], (
        "driver changed: CB_SCALE_CODING regained a shell default"
    )
    assert "CB_SCALE_CODING=${CB_SCALE_CODING:-}" in script

    # Doc side: the block's own VAR=value claims must match the shell. An
    # `<unset...>` claim requires no non-empty driver default; any other claim
    # must be one of the driver's assigned values. A wrong claim in new words
    # fails here; the fixed wrong strings below are the regression tooth.
    for var, claimed in _defaults_block_claims(doc).items():
        assigned = _shell_assign_defaults(script, var)
        if claimed.startswith("<unset"):
            assert not [v for v in assigned if v != ""], (
                f"CLAUDE.md claims {var} is {claimed} but run-pipeline.sh "
                f"assigns {assigned!r}"
            )
        else:
            assert claimed in assigned, (
                f"CLAUDE.md claims {var}={claimed} but run-pipeline.sh "
                f"assigns {assigned!r}"
            )
    assert "`TARGET_PROFILE=vllm_packed_moe`" not in doc, (
        "CLAUDE.md still pins TARGET_PROFILE=vllm_packed_moe as a shell "
        "default; the driver leaves TARGET_PROFILE unset (R11)"
    )
    assert "`CB_EXPERT_EMPIRICAL=0`" not in doc, (
        "CLAUDE.md still claims a CB_EXPERT_EMPIRICAL=0 shell default; "
        "the driver has none since 2026-09-02"
    )
    assert "`CB_SCALE_CODING=two_tier`" not in doc, (
        "CLAUDE.md still claims a CB_SCALE_CODING=two_tier shell default; "
        "the driver keeps it as a settings-hash entry only"
    )


def test_166_claude_polish_guarantee_is_historical():
    """#166: coordinate-descent polish is archived (2026-05-15), not a live
    shipping guarantee. CLAUDE.md must scope it to the past and stop listing
    polish among live GPU-bound production hot paths."""
    script = _read("prismaquant/run-pipeline.sh")
    doc = _read("CLAUDE.md")

    # Code side, derived: the archive wall exists and says the driver never
    # invokes polish; no live module imports it; the driver's only polish
    # mentions are a comment and an archived-lane error string.
    banner = _read("archive/polish_2026-05-15/README.md")
    assert "does not invoke polish" in banner
    live_importers = [
        p.name
        for p in (ROOT / "prismaquant").glob("*.py")
        if re.search(r"^\s*(import|from)\s+.*polish", p.read_text(encoding="utf-8"), re.M)
        and "tessera" not in p.name
        and p.name != "polish.py"
    ]
    assert not live_importers, f"live polish importers: {live_importers}"
    invocations = [
        line for line in script.splitlines()
        if "polish" in line.lower() and not line.strip().startswith("#")
        and "L3-polish-of-many" not in line
    ]
    assert not invocations, f"driver invokes polish: {invocations}"

    # Doc side: the guarantee paragraph must be past-tense and walled.
    guarantee = next(
        line for line in doc.splitlines() if "guarantee" in line.lower()
    )
    assert "archived" in doc[doc.index(guarantee):doc.index(guarantee) + 800], (
        "CLAUDE.md 'guarantee' paragraph must say polish is archived "
        "at archive/polish_2026-05-15/"
    )
    assert "archive/polish_2026-05-15" in doc
    # No present-tense acceptance promise for the archived mechanism.
    assert "accepts only single-unit flips" not in doc, (
        "CLAUDE.md still sells archived coord-descent polish as a live "
        "accept-only-improving-flips guarantee"
    )
    # Hot-path list must not sell polish as a live production stage.
    hot = next(
        line for line in doc.splitlines() if "Every production hot path" in line
    )
    assert "polish" not in hot or "archived" in hot or "historical" in hot, (
        "CLAUDE.md still lists polish among live GPU-bound production hot paths"
    )


def test_167_diagram2_names_tessera_container_fail_closed():
    """#167: the driver has grown a real EXPORT_CONTAINER=tessera arm, so
    DIAGRAM-2 must draw the Tessera container (fail-closed on the release pin)
    instead of claiming no exporter exists behind the lane."""
    script = _read("prismaquant/run-pipeline.sh")
    doc = _read("docs/ARCHITECTURE.md")

    # Code side, derived: the tessera arm exists in the driver.
    assert 'EXPORT_CONTAINER" == "tessera"' in script
    assert "prismaquant.tessera_export_lane" in script

    # Doc side: locate the DIAGRAM-2 block and assert it no longer denies an
    # exporter while asserting the fail-closed posture and drawing the box.
    start = doc.index("DIAGRAM-2")
    block = doc[start:start + 3000]
    assert "no PrismaQuant exporter writes those bytes" not in block, (
        "DIAGRAM-2 still claims no exporter writes Tessera bytes, "
        "contradicting §9.4 and the driver's tessera arm"
    )
    assert "stays deliberately absent" not in block, (
        "DIAGRAM-2 still keeps the Tessera lane deliberately absent"
    )
    assert "tessera" in block.lower(), "DIAGRAM-2 must name the Tessera lane"
    mermaid = doc[doc.index("```mermaid", start):doc.index("```", doc.index("```mermaid", start) + 3) + 3]
    assert "essera" in mermaid, (
        "DIAGRAM-2's mermaid diagram must draw the Tessera container"
    )
    assert "fail-closed" in block or "fail_closed" in block or "PENDING" in block, (
        "DIAGRAM-2 must say the Tessera container is fail-closed on the "
        "PENDING release pin"
    )
