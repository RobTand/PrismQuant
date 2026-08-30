"""Mechanical half of the ARCHITECTURE.md maintenance contract (CLAUDE.md §4
principle 13, AGENTS.md rule 10): the master document's defaults table must
match `prismaquant/run-pipeline.sh`, and its structural anchors must exist.

The judgment half — prose describing behavior that changed — cannot be tested;
this file only makes silent drift of the enumerable facts impossible.
"""

import json
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


def _plain_markdown_cell(cell: str) -> str:
    return cell.replace("**", "").replace("`", "").strip()


def _architecture_84_rows() -> list[dict[str, str]]:
    match = re.search(
        r"^### 8\.4 Conformance matrix\s*$\n(.*?)(?=^### 8\.5 )",
        _doc(),
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, "ARCHITECTURE.md §8.4 conformance matrix is missing"

    lines = match.group(1).splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("| Arch |")),
        None,
    )
    assert header_index is not None, "ARCHITECTURE.md §8.4 table header is missing"

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    headers = [_plain_markdown_cell(cell) for cell in cells(lines[header_index])]
    assert headers == [
        "Arch",
        "profile",
        "prio",
        "structure spec",
        "default_serving_profile",
        "supported_lanes (preferred)",
        "gridbook opt-in",
        "MTP",
    ], f"ARCHITECTURE.md §8.4 table headers drifted: {headers!r}"
    separator = lines[header_index + 1]
    assert re.fullmatch(r"\|(?:\s*:?-+:?\s*\|)+", separator), (
        f"ARCHITECTURE.md §8.4 table separator is invalid: {separator!r}"
    )

    rows: list[dict[str, str]] = []
    for line in lines[header_index + 2:]:
        if not line.startswith("|"):
            break
        values = cells(line)
        assert len(values) == len(headers), (
            "ARCHITECTURE.md §8.4 table row has "
            f"{len(values)} cells, expected {len(headers)}: {line}"
        )
        rows.append(dict(zip(headers, values, strict=True)))
    assert rows, "ARCHITECTURE.md §8.4 table has no profile rows"
    return rows


def _assert_84_cell(
    *,
    row: str,
    column: str,
    documented: object,
    actual: object,
    matches: bool,
) -> None:
    assert matches, (
        f"ARCHITECTURE.md §8.4 row {row!r}, column {column!r} drifted: "
        f"documented value={documented!r}; actual value={actual!r}"
    )


def _documented_profile_file(row: dict[str, str]) -> str:
    cell = row["profile"]
    match = re.fullmatch(r"`([^`/]+\.py)`", cell)
    _assert_84_cell(
        row=_plain_markdown_cell(row["Arch"]),
        column="profile",
        documented=cell,
        actual="one backticked model-profile module filename",
        matches=match is not None,
    )
    assert match is not None
    return match.group(1)


def _documented_serving_profile(
    row_name: str,
    cell: str,
) -> tuple[str | None, tuple[str, ...] | None]:
    identifiers = re.findall(r"`([^`]+)`", cell)
    if identifiers:
        extends_match = re.search(
            r"\(extends\s+(`[^`]+`(?:\s*,\s*`[^`]+`)*)\)",
            cell,
        )
        documented_extends = (
            tuple(re.findall(r"`([^`]+)`", extends_match.group(1)))
            if extends_match else None
        )
        expected_identifiers = 1 + len(documented_extends or ())
        _assert_84_cell(
            row=row_name,
            column="default_serving_profile",
            documented=cell,
            actual=(
                "one serving-profile ID plus an optional "
                "'(extends `id`, ...)' claim"
            ),
            matches=len(identifiers) == expected_identifiers,
        )
        return identifiers[0], documented_extends
    plain = _plain_markdown_cell(cell).lower()
    is_none = "spec declares none" in plain or plain.startswith("—")
    _assert_84_cell(
        row=row_name,
        column="default_serving_profile",
        documented=cell,
        actual="a backticked serving-profile ID or an explicit none marker",
        matches=is_none,
    )
    return None, None


def _documented_lane_claim(
    row_name: str,
    cell: str,
) -> tuple[tuple[str, ...], str, bool]:
    plain = _plain_markdown_cell(cell).replace("⚠", "").strip()
    claims_no_declaration = "spec declares none" in plain.lower()
    accessor_default = re.search(
        r"accessor default\s+(.+)$",
        plain,
        flags=re.IGNORECASE,
    )
    lane_text = accessor_default.group(1).strip() if accessor_default else plain
    match = re.fullmatch(r"(.+?)(?:\s+\(([^()]*)\))?", lane_text)
    _assert_84_cell(
        row=row_name,
        column="supported_lanes (preferred)",
        documented=cell,
        actual="a comma-separated lane list with an optional preferred lane",
        matches=match is not None,
    )
    assert match is not None

    aliases = {
        "CT": "compressed-tensors",
        "compressed-tensors": "compressed-tensors",
        "nvfp4_cb": "nvfp4_cb",
        "gguf": "gguf",
    }
    documented_lanes = tuple(part.strip() for part in match.group(1).split(","))
    unknown = tuple(lane for lane in documented_lanes if lane not in aliases)
    _assert_84_cell(
        row=row_name,
        column="supported_lanes (preferred)",
        documented=cell,
        actual=f"known lane names {sorted(aliases)!r}",
        matches=not unknown,
    )
    lanes = tuple(aliases[lane] for lane in documented_lanes)

    preferred_text = match.group(2)
    if preferred_text is None or preferred_text.lower() == "default":
        preferred = lanes[0] if len(lanes) == 1 else ""
    else:
        _assert_84_cell(
            row=row_name,
            column="supported_lanes (preferred)",
            documented=cell,
            actual=f"a preferred lane in {sorted(aliases)!r}",
            matches=preferred_text in aliases,
        )
        preferred = aliases[preferred_text]
    return lanes, preferred, claims_no_declaration


def _documented_has_mtp(row_name: str, cell: str) -> bool:
    plain = _plain_markdown_cell(cell)
    declares_true = "build_mtp_module" in plain
    declares_false = plain == "none" or "has_mtp → False" in plain
    _assert_84_cell(
        row=row_name,
        column="MTP",
        documented=cell,
        actual="an explicit build_mtp_module or no-MTP claim",
        matches=declares_true != declares_false,
    )
    return declares_true


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


def test_model_profile_conformance_table_matches_registered_profiles():
    from prismaquant.model_profiles.default import DefaultProfile
    from prismaquant.model_profiles.registry import _REGISTERED
    from prismaquant.model_profiles.structure import load_structure_spec

    registered = []
    for profile_class in [*_REGISTERED, DefaultProfile]:
        profile = profile_class()
        registered.append((profile.name, profile_class, profile))
    registered_names = [name for name, _, _ in registered]
    assert len(registered_names) == len(set(registered_names)), (
        f"model-profile registry has duplicate names: {registered_names!r}"
    )
    profiles = {
        name: (profile_class, profile)
        for name, profile_class, profile in registered
    }

    rows_by_name: dict[str, dict[str, str]] = {}
    for row in _architecture_84_rows():
        profile_file = _documented_profile_file(row)
        row_name = Path(profile_file).stem
        _assert_84_cell(
            row=row_name,
            column="profile",
            documented=row["profile"],
            actual="one table row per profile",
            matches=row_name not in rows_by_name,
        )
        rows_by_name[row_name] = row

    # This is deliberately bidirectional: an omitted new profile and an old
    # row whose profile was removed are independent documentation failures.
    for name, profile_class, _ in registered:
        expected_file = f"{profile_class.__module__.rsplit('.', 1)[-1]}.py"
        _assert_84_cell(
            row=name,
            column="profile",
            documented=(
                rows_by_name[name]["profile"]
                if name in rows_by_name else "<missing row>"
            ),
            actual=expected_file,
            matches=name in rows_by_name,
        )
    for name, row in rows_by_name.items():
        _assert_84_cell(
            row=name,
            column="profile",
            documented=row["profile"],
            actual=(
                f"{profiles[name][0].__module__.rsplit('.', 1)[-1]}.py"
                if name in profiles else "<no profile subclass exists>"
            ),
            matches=name in profiles,
        )

    serving_specs: dict[str, dict[str, object]] = {}
    for path in (ROOT / "prismaquant" / "serving_profile_specs").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        serving_id = str(payload["id"])
        assert serving_id not in serving_specs, (
            f"serving-profile specs have duplicate id {serving_id!r}"
        )
        serving_specs[serving_id] = payload

    for name, (profile_class, profile) in profiles.items():
        row = rows_by_name[name]
        documented_file = _documented_profile_file(row)
        actual_file = f"{profile_class.__module__.rsplit('.', 1)[-1]}.py"
        _assert_84_cell(
            row=name,
            column="profile",
            documented=documented_file,
            actual=actual_file,
            matches=documented_file == actual_file,
        )

        spec = load_structure_spec(name)
        structure_cell = row["structure spec"]
        documented_has_spec = "✅" in structure_cell
        documented_has_no_spec = "n/a" in _plain_markdown_cell(
            structure_cell
        ).lower()
        _assert_84_cell(
            row=name,
            column="structure spec",
            documented=structure_cell,
            actual="exactly one of spec present or n/a",
            matches=documented_has_spec != documented_has_no_spec,
        )
        _assert_84_cell(
            row=name,
            column="structure spec",
            documented=structure_cell,
            actual="present" if spec is not None else "n/a",
            matches=documented_has_spec == (spec is not None),
        )
        if spec is not None:
            _assert_84_cell(
                row=name,
                column="structure spec",
                documented=structure_cell,
                actual=f"spec id {spec.id!r}",
                matches=spec.id == name,
            )

        priority_cell = row["prio"]
        priority_text = _plain_markdown_cell(priority_cell)
        if name == "default":
            _assert_84_cell(
                row=name,
                column="prio",
                documented=priority_cell,
                actual="terminal fallback",
                matches=(
                    priority_text.startswith("—") and "terminal" in priority_text
                ),
            )
        else:
            priority_match = re.fullmatch(r"\d+", priority_text)
            _assert_84_cell(
                row=name,
                column="prio",
                documented=priority_cell,
                actual=profile_class.priority,
                matches=priority_match is not None,
            )
            assert priority_match is not None
            documented_priority = int(priority_match.group())
            _assert_84_cell(
                row=name,
                column="prio",
                documented=documented_priority,
                actual=profile_class.priority,
                matches=documented_priority == profile_class.priority,
            )
            if spec is not None:
                _assert_84_cell(
                    row=name,
                    column="prio",
                    documented=documented_priority,
                    actual=spec.priority,
                    matches=documented_priority == spec.priority,
                )

        serving_cell = row["default_serving_profile"]
        documented_serving, documented_extends = _documented_serving_profile(
            name,
            serving_cell,
        )
        actual_serving = profile.serving_profile_id()
        _assert_84_cell(
            row=name,
            column="default_serving_profile",
            documented=documented_serving,
            actual=actual_serving,
            matches=documented_serving == actual_serving,
        )
        if spec is not None:
            _assert_84_cell(
                row=name,
                column="default_serving_profile",
                documented=documented_serving,
                actual=spec.default_serving_profile,
                matches=documented_serving == spec.default_serving_profile,
            )
        if actual_serving is not None:
            actual_serving_spec = serving_specs.get(actual_serving)
            _assert_84_cell(
                row=name,
                column="default_serving_profile",
                documented=serving_cell,
                actual=(
                    actual_serving
                    if actual_serving_spec is not None
                    else f"{actual_serving} (missing serving-profile spec)"
                ),
                matches=actual_serving_spec is not None,
            )
            if documented_extends is not None:
                assert actual_serving_spec is not None
                actual_extends = tuple(actual_serving_spec.get("extends") or ())
                _assert_84_cell(
                    row=name,
                    column="default_serving_profile",
                    documented=f"extends={documented_extends!r}",
                    actual=f"extends={actual_extends!r}",
                    matches=documented_extends == actual_extends,
                )

        lane_cell = row["supported_lanes (preferred)"]
        documented_lanes, documented_preferred, claims_no_lanes = (
            _documented_lane_claim(name, lane_cell)
        )
        _assert_84_cell(
            row=name,
            column="supported_lanes (preferred)",
            documented=lane_cell,
            actual="a lane set without duplicates",
            matches=len(documented_lanes) == len(set(documented_lanes)),
        )
        actual_lanes = profile.supported_export_lanes()
        actual_preferred = profile.preferred_export_lane()
        _assert_84_cell(
            row=name,
            column="supported_lanes (preferred)",
            documented=(
                f"supported={documented_lanes!r}, preferred={documented_preferred!r}"
            ),
            actual=f"supported={actual_lanes!r}, preferred={actual_preferred!r}",
            matches=(
                set(documented_lanes) == set(actual_lanes)
                and documented_preferred == actual_preferred
            ),
        )
        if claims_no_lanes:
            _assert_84_cell(
                row=name,
                column="supported_lanes (preferred)",
                documented=lane_cell,
                actual=None if spec is None else spec.supported_lanes,
                matches=spec is not None and not spec.supported_lanes,
            )

        mtp_cell = row["MTP"]
        documented_has_mtp = _documented_has_mtp(name, mtp_cell)
        actual_has_mtp = profile.has_mtp()
        _assert_84_cell(
            row=name,
            column="MTP",
            documented=documented_has_mtp,
            actual=actual_has_mtp,
            matches=documented_has_mtp == actual_has_mtp,
        )
        source_prefix = re.search(
            r'mtp_source_prefix\s+"([^"]+)"',
            _plain_markdown_cell(mtp_cell),
        )
        if source_prefix:
            _assert_84_cell(
                row=name,
                column="MTP",
                documented=source_prefix.group(1),
                actual=profile.mtp_source_prefix(),
                matches=source_prefix.group(1) == profile.mtp_source_prefix(),
            )
        if "in passthrough_prefixes" in _plain_markdown_cell(mtp_cell):
            prefix = profile.mtp_source_prefix()
            passthrough_prefixes = profile.source_passthrough_prefixes()
            _assert_84_cell(
                row=name,
                column="MTP",
                documented=mtp_cell,
                actual=passthrough_prefixes,
                matches=(
                    prefix is not None and prefix in passthrough_prefixes
                ),
            )

        # `Arch` is a human label containing aliases and smoke examples; no
        # profile/spec field canonically serializes that prose. `gridbook
        # opt-in` combines an external consumer contract with dated validation
        # status and loader commentary, none of which is derivable from the
        # model or serving specs. The MTP capability, explicit source prefix,
        # and passthrough claim above are checkable; class names, routes,
        # evidence tags, and historical notes elsewhere in that cell are not.
        # The R22/L1 annotations and "all 8 overrides" count are audit history,
        # not schema fields; their underlying spec presence and IDs are pinned.


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
    assert _shell_default(script, "PRISMAQUANT_CB_LDLQ") == "0"
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
