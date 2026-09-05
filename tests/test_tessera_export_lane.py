"""The Tessera EXPORT lane: vocabulary, declaration, and three fail-closed gates.

Rob's decision of 2026-09-02 made the sanctioned serving lanes
compressed-tensors, GGUF and Tessera.  ``run-pipeline.sh`` implemented the
retirement half the same day (``EXPORT_CONTAINER=nvfp4_cb`` fails closed); the
addition half is what this module pins.

The shape being pinned matters more than the values.  Being *in* the
``EXPORT_CONTAINER`` vocabulary is not permission to build: an architecture
still has to declare the lane, and the lane still has to clear a preflight
whose three gates are each read from the pinned runtime's own published table.
Today the third gate -- the release pin -- refuses every run, and it is the
ONLY thing that does.  That is the difference this file protects: the refusal
is the pin's, spoken where an operator can act on it, rather than "unknown
export lane" from a vocabulary check three layers up.
"""
from __future__ import annotations

import json

import pytest

from prismaquant import tessera_export_lane as tel
from prismaquant.model_profiles.structure import (
    DEFAULT_EXPORT_LANE,
    EXPORT_LANES,
    canonical_export_lane,
)
from prismaquant.serving_profiles import require_lane_supported
from prismaquant import tessera_serving_runtime_pin as pin_module
from prismaquant.tessera_serving_runtime_pin import (
    TESSERA_SERVING_RESIDENCY_ENV,
    TESSERA_SERVING_RUNTIME_PIN_SCHEMA,
    TESSERA_SERVING_RUNTIME_REPOSITORY,
    require_pinned_tessera_runtime,
)

#: The same non-real commit ``test_tessera_lane_admission`` uses: it exists to
#: prove the machinery and the constants are monkeypatched to match it, so no
#: fixture here can ever be mistaken for an attestation.
FIXTURE_COMMIT = "0" * 39 + "1"
FIXTURE_VERSION = "0.1.0"


@pytest.fixture()
def released_pin(tmp_path, monkeypatch):
    """A RELEASED Tessera pin, built through the real reader and gate."""
    payload = {
        "schema": TESSERA_SERVING_RUNTIME_PIN_SCHEMA,
        "repository": TESSERA_SERVING_RUNTIME_REPOSITORY,
        "commit": FIXTURE_COMMIT,
        "version": FIXTURE_VERSION,
        "version_is_release": True,
        "contract_sha256": pin_module.installed_tessera_contract_sha256(),
        "runtime_contract_schema": "tessera.runtime-contract.v1",
        "plugin_entry_point": "tessera = tessera.serving:register",
        "serving_residency_env": TESSERA_SERVING_RESIDENCY_ENV,
        "serving_native_extensions": [
            {"module_name_prefix": "tessera_nvfp4_",
             "filename_glob": "tessera_nvfp4_*.so",
             "match": "basename_fnmatch",
             # A synthetic released pin, not a transcription of the runtime:
             # the release gate under test reads commit/version, never this
             # block.  It carries the contracted shape so the fixture parses.
             "when_unavailable": {
                 "resident": {"status": "substituted",
                              "decoder": "torch_materialize_stock"},
                 "streamed": {"status": "refused", "decoder": None}}},
        ],
    }
    path = tmp_path / "released_pin.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        pin_module, "TESSERA_SERVING_RUNTIME_PINNED_COMMIT", FIXTURE_COMMIT)
    monkeypatch.setattr(
        pin_module, "TESSERA_SERVING_RUNTIME_PINNED_VERSION", FIXTURE_VERSION)
    monkeypatch.setattr(
        pin_module, "TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256",
        payload["contract_sha256"])
    pin = pin_module.load_tessera_serving_runtime_pin(path)
    monkeypatch.setattr(
        pin_module, "load_tessera_serving_runtime_pin", lambda *a, **k: pin)
    require_pinned_tessera_runtime(pin)   # the real gate, satisfied
    return pin


@pytest.fixture
def tessera_repo(tmp_path, monkeypatch):
    """A TESSERA_REPO holding exactly the tools the lane spec declares.

    Gate 4 resolves each declared `producer_tools` entry through the env var
    the declaration names, so a preflight run with no TESSERA_REPO refuses --
    correctly, and it is why this fixture exists rather than the test reaching
    into the real checkout: the gate under test must be satisfied by
    something, and a hermetic something is what keeps the pin the only
    variable.
    """
    from prismaquant.lane_spec import load_lane_spec

    from prismaquant import tessera_export_lane as _tel

    root = tmp_path / "tessera-checkout"
    for tool in load_lane_spec("tessera").producer_tools:
        path = root / tool.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    # Gate 5 requires this checkout to BE the pinned Tessera, so the fixture
    # packages the very bytes the pin names -- copied from the installed
    # contract rather than synthesized, which is what makes it the same
    # object rather than a lookalike.
    packaged = _tel.packaged_contract_path()
    contract = root.joinpath("src", *packaged.parts[-3:])
    contract.parent.mkdir(parents=True, exist_ok=True)
    contract.write_bytes(packaged.read_bytes())
    monkeypatch.setenv("TESSERA_REPO", str(root))
    return root


def _dense_target():
    """The serving target a SCOPED (v5/v6) contract requires, derived from it.

    A scoped table admits only under one complete context, so a context-free
    preflight is refused -- correctly.  Every value here is read off the
    contract's own dense cell and its family's ``residency_modes`` rather than
    typed, so a runtime that re-scopes the lane re-stales this helper instead
    of quietly testing a context the runtime no longer publishes.
    """
    from prismaquant.tessera_serving_scope import ServingTarget

    contract = json.loads(
        tel.packaged_contract_path().read_text(encoding="utf-8"))
    cell = next(c for c in contract["lane_eligibility"]["cells"]
                if c["structure"] == "dense")
    return ServingTarget(
        platform=cell["platform"],
        runtime_image=cell["runtime"]["image"],
        execution_mode=cell["runtime"]["execution_modes"][0],
        residency=next(row["residency_modes"][0]
                       for row in contract["formats"]
                       if row["family"] == cell["family"]),
    )


def _model_dir(tmp_path, **config):
    directory = tmp_path / "model"
    directory.mkdir(exist_ok=True)
    payload = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
    payload.update(config)
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------
def test_tessera_is_in_the_vocabulary_and_the_retired_lane_is_not():
    """This file owns the TESSERA half; it does not re-type the roster.

    `assert EXPORT_LANES == (<three literals>)` lived here and in
    `tests/test_profile_export_lanes.py`, two copies of one roster, each of
    which had to be edited to ADD the lane the issue was asking for. The
    vocabulary-as-a-whole properties (closure against `lane_specs/*.json`, the
    retired set, per-lane round-trip) are in that file, quantified over
    `EXPORT_LANES`; what belongs here is the one membership this file's subject
    depends on, and the assurance that gaining it relaxed nothing.
    """
    assert "tessera" in EXPORT_LANES
    assert canonical_export_lane("tessera") == "tessera"
    assert DEFAULT_EXPORT_LANE == "compressed-tensors"
    # The retired lane stays unknown -- adding one is not relaxing the check.
    with pytest.raises(ValueError, match="unknown export lane"):
        canonical_export_lane("nvfp4_cb")


def test_an_architecture_that_declares_the_lane_passes_the_r6_preflight():
    """The lane-NAME refusal is gone for a declared architecture.

    `qwen3` is the profile whose fused groups Tessera's exporter stacks and
    whose 0.6B checkpoint was served from a PrismaQuant allocation on the
    plugin; nothing else declares the lane, so nothing else is admitted.
    """
    from prismaquant.model_profiles.gemma4 import Gemma4Profile
    from prismaquant.model_profiles.qwen3 import Qwen3Profile

    assert require_lane_supported(Qwen3Profile(), "tessera") == "tessera"
    with pytest.raises(SystemExit, match="not a declared lane"):
        require_lane_supported(Gemma4Profile(), "tessera")


# ---------------------------------------------------------------------------
# Gate 1 -- the release pin, which is what refuses today
# ---------------------------------------------------------------------------
def test_the_preflight_refuses_a_tessera_that_is_not_the_pinned_one(monkeypatch):
    """Admission is fail-closed BY THE PIN, and the message is actionable.

    Until 2026-09-04 the refusal fired on PENDING sentinels, because no
    Tessera release tag existed. Rob retired the tag, so what refuses now is
    the digest: a Tessera on ``PYTHONPATH`` whose ``runtime_contract.json`` is
    not the pinned bytes. Same fail-closed answer, a checkable fact.
    """
    monkeypatch.setattr(pin_module, "installed_tessera_contract_sha256",
                        lambda: "d" * 64)
    with pytest.raises(tel.TesseraExportLaneError) as excinfo:
        tel.require_release_pin()
    message = str(excinfo.value)
    assert "the installed Tessera is not the pinned Tessera" in message
    assert "tessera_serving_runtime_pin.json" in message   # names the fix
    assert "ONE reviewed commit" in message                # names the shape


def test_under_the_pinned_runtime_the_whole_preflight_passes(
        released_pin, tmp_path, tessera_repo):
    """The other half of "refused by the pin".

    With ONLY the pin boundary satisfied -- the real packaged contract, the
    real lane spec, the real structure read -- a dense checkpoint clears every
    gate.  That is what proves the refusal above is the pin's and not an
    artefact of an unreadable table or an undeclared structure.

    The target is not optional here: a scoped (v5/v6) contract admits under
    one complete serving context, so the context-free call is refused first
    and the pin is never reached.  Both halves are asserted, because "passes"
    and "passes for the reason claimed" are different facts.
    """
    with pytest.raises(tel.TesseraExportLaneError, match="scoped"):
        tel.preflight(_model_dir(tmp_path))
    report = tel.preflight(_model_dir(tmp_path), target=_dense_target())
    assert report["structure"] == "dense"
    assert report["quant_method"] == "tessera"
    assert report["executes"] == sorted(
        r["name_pattern"].replace("{k}", "*")
        for r in json.loads(tel.packaged_contract_path().read_text())["formats"]
    )
    target = _dense_target()
    assert tel.main([
        "--model", str(_model_dir(tmp_path)),
        "--tessera-platform", target.platform,
        "--tessera-runtime-image", target.runtime_image,
        "--tessera-execution-mode", target.execution_mode,
        "--tessera-residency", target.residency,
    ]) == 0


# ---------------------------------------------------------------------------
# Gate 2 -- principle 14
# ---------------------------------------------------------------------------
def test_executes_is_derived_from_the_packaged_contract():
    """The value a gate reads is a READ of the runtime's table, not a copy."""
    from prismaquant.lane_spec import load_lane_spec

    # The literal is DERIVED here too, by a second implementation, rather than
    # typed: a hand-written tuple re-stales every time the runtime publishes a
    # family (it did, within a day, when TESSERA_BF16_K1 landed), and the
    # maintenance answer to that is to stop hand-writing it -- not to loosen
    # the check.  What is still pinned is the RULE: one glob per published
    # row, `{k}` -> `*`, sorted.
    import json

    rows = json.loads(tel.packaged_contract_path().read_text())["formats"]
    expected = tuple(sorted(r["name_pattern"].replace("{k}", "*") for r in rows))
    derived = tel.derive_executes()
    assert derived == expected
    assert len(derived) == len(rows) and derived, "one glob per published family"
    spec = load_lane_spec("tessera")
    assert tuple(sorted(spec.served_activation_quantization.executes)) == derived
    assert tel.require_executes_derived_from_contract() == derived


def test_a_lane_spec_that_disagrees_with_the_runtime_is_refused(monkeypatch):
    """PRINCIPLE 14 in the field that asserts what the runtime executes.

    A runtime that adds a family, retires one, or renames a pattern makes the
    lane spec wrong; the fix is to re-read the table, and the refusal says so
    rather than letting a stale glob price an A side that is not executed.
    """
    monkeypatch.setattr(
        tel, "derive_executes",
        lambda *a, **k: ("TESSERA_E2M1_K2_R*", "TESSERA_E9M9_K9_R*"))
    with pytest.raises(tel.TesseraExportLaneError, match="PRINCIPLE 14"):
        tel.require_executes_derived_from_contract()


def test_a_formats_row_with_no_rate_placeholder_is_refused():
    """A pattern the derivation cannot read is refused, never guessed at."""
    with pytest.raises(tel.TesseraExportLaneError, match="rate placeholder"):
        tel.derive_executes({"TESSERA_X": {"name_pattern": "TESSERA_X"}})


def test_an_empty_formats_table_is_unattested_not_a_clean_bill():
    with pytest.raises(tel.TesseraExportLaneError, match="UNATTESTED"):
        tel.derive_executes({})


# ---------------------------------------------------------------------------
# Gate 3 -- the structures the contract declares
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config,expected", [
    ({}, "dense"),
    ({"num_experts": 0}, "dense"),
    ({"num_experts": 128}, "routed_moe"),
    ({"n_routed_experts": 160}, "routed_moe"),
    ({"num_local_experts": 8}, "routed_moe"),
    ({"text_config": {"num_experts": 64}}, "routed_moe"),
])
def test_the_structure_is_read_from_the_checkpoint_not_the_family(
        tmp_path, config, expected):
    """`qwen3` claims both Qwen3ForCausalLM and Qwen3MoeForCausalLM, and only
    one of them is a structure the contract declares — so the fact has to come
    from the artifact being built, not from the architecture family."""
    assert tel.model_structure(_model_dir(tmp_path, **config)) == expected


def test_a_routed_moe_checkpoint_is_decided_by_the_contracts_own_evidence(
        released_pin, tessera_repo, tmp_path, monkeypatch):
    """The gate reads ADMISSIBLE cells, and admissibility is what was measured.

    Contract v16 declared `dense` only, so a MoE checkpoint was refused by the
    structure vocabulary.  v17 (Tessera PR #176) DECLARED `routed_moe` and
    carried two routed-MoE cells publishing ``evidence.smoke.status:
    "repetitive"`` -- a greedy smoke that degenerated -- so from v17 through
    v20 this gate refused a MoE checkpoint on the cells' own evidence rather
    than admitting it because the vocabulary grew a word.  v21 (Tessera #313)
    re-measured that smoke through the checkpoint's own chat template and both
    cells now publish ``"recorded"``; the same gate admits the checkpoint with
    no change here.  Promoting routed-MoE past the menu is still a decision on
    evidence and it is Rob's (principle 9, prismaquant #198); what this test
    pins is that PrismaQuant makes neither move by widening or narrowing a
    gate -- both halves are read off the cells.
    """
    directory = _model_dir(tmp_path, num_experts=128,
                           architectures=["Qwen3MoeForCausalLM"])
    contract = json.loads(
        tel.packaged_contract_path().read_text(encoding="utf-8"))
    cells = [c for c in contract["lane_eligibility"]["cells"]
             if c["structure"] == "routed_moe"]
    assert cells, "the packaged contract no longer covers routed_moe at all"

    # The installed contract (v21): the smoke is recorded, the gate admits.
    assert all(c["evidence"]["smoke"]["status"] == "recorded" for c in cells)
    assert tel.require_declared_structure(directory) == "routed_moe"

    # The contract v17 through v20 published: the same cells with the
    # degenerate smoke transplanted back, and the same gate refuses -- naming
    # the structure, the contract's own word, and whose decision it is not.
    historical = json.loads(json.dumps(contract))
    for cell in historical["lane_eligibility"]["cells"]:
        if cell["structure"] == "routed_moe":
            cell["evidence"]["smoke"]["status"] = "repetitive"
            cell["evidence"]["smoke"]["receipt"] = (
                "docs/measurements/tessera-lfm-campaign-2026-09-04.md")
    historical_path = tmp_path / "runtime_contract_v20_shape.json"
    historical_path.write_text(json.dumps(historical), encoding="utf-8")
    monkeypatch.setattr(tel, "packaged_contract_path", lambda: historical_path)
    with pytest.raises(tel.TesseraExportLaneError) as excinfo:
        tel.require_declared_structure(directory)
    message = str(excinfo.value)
    assert "routed_moe" in message
    assert "repetitive" in message          # the contract's own word
    assert "belongs to Rob" in message      # the decision it does not make
    assert tel.main(["--model", str(directory)]) == 2


def test_a_checkpoint_with_no_config_is_refused_rather_than_assumed_dense(
        tmp_path):
    empty = tmp_path / "no_config"
    empty.mkdir()
    with pytest.raises(tel.TesseraExportLaneError, match="refuses rather than"):
        tel.model_structure(empty)


# ---------------------------------------------------------------------------
# The driver arm
# ---------------------------------------------------------------------------
def _run_pipeline_text():
    from pathlib import Path

    import prismaquant

    return (Path(prismaquant.__file__).resolve().parent
            / "run-pipeline.sh").read_text(encoding="utf-8")


def test_the_driver_has_a_real_tessera_arm_that_names_tesseras_own_tools():
    """`tessera` in the vocabulary is only honest if the driver can act on it.

    The arm NAMES Tessera's plan translator and exporter rather than copying
    either: a wire recipe with two homes is how the two halves of one format
    drift apart, and the lane spec already uses this boundary for the serve
    script and the route census.
    """
    text = _run_pipeline_text()
    assert 'if [[ "$EXPORT_CONTAINER" == "tessera" ]]; then' in text
    assert "python3 -m prismaquant.tessera_export_lane --model" in text
    assert "experiments/plan_from_layer_config.py" in text
    assert "experiments/export_tessera_serving.py" in text
    # No codec in this repository: the lane must not grow a second encoder.
    assert "export_native_compressed" not in text.split(
        'if [[ "$EXPORT_CONTAINER" == "tessera" ]]; then')[1].split(
            'if [[ "$EXPORT_CONTAINER" == "gguf" ]]; then')[0]


def test_the_driver_declares_the_residency_knob_and_validates_it():
    """TESSERA_SERVE_MODE changes the footprint AND vLLM's compile-cache key,
    so it is declared rather than defaulted silently, and a third value is a
    refusal rather than a serve nobody receipted."""
    text = _run_pipeline_text()
    assert ': "${TESSERA_SERVE_MODE:=resident}"' in text
    assert "resident|streamed) ;;" in text


def test_the_arm_invokes_exactly_the_declared_producer_tools():
    """#119 half 2, in-repo half: the arm shells out to Tessera's tools by
    the path the lane DECLARATION names -- no second spelling in either
    direction. A tidy-up in the Tessera repository that moves one script, or
    an arm edit that calls a new one, fails here instead of dying at export
    time (or worse: resolving to a different file than the declaration
    advertises).

    Both sides are derived from the code that owns them -- the declaration's
    `producer_tools` and the driver's `${TESSERA_REPO}` command invocations
    (not its banner echoes, which name the serve script and the census for
    the operator) -- so a third tool needs no test edit, only a declaration
    the arm honors."""
    import re

    from prismaquant.lane_spec import load_lane_spec

    text = _run_pipeline_text()
    declared = {
        f"${{TESSERA_REPO%/}}/{tool.path}"
        for tool in load_lane_spec("tessera").producer_tools
    }
    assert declared, "the lane declares no producer tools at all"
    invoked = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("python3 ") or stripped.startswith("bash ")):
            continue
        invoked.update(
            "${TESSERA_REPO%/}/" + path
            for path in re.findall(r"\$\{TESSERA_REPO%/\}/([^\s\"]+)", line)
        )
    for tool in sorted(declared):
        assert tool in invoked, (
            f"the arm never invokes the declared producer tool {tool}")
    undeclared = {path for path in invoked if path not in declared}
    assert not undeclared, (
        "the arm shells out to Tessera-repository tools the lane does not "
        f"declare: {sorted(undeclared)}")


# ---------------------------------------------------------------------------
# Gate 5 -- the tools that WRITE the bytes come from the Tessera the pin attests
# ---------------------------------------------------------------------------
def test_the_checkout_that_encodes_must_be_the_tessera_the_pin_attests(
        tessera_repo, released_pin, monkeypatch):
    """Principle 8: the attested runtime and the encoding tools are one object.

    Gate 1 hashes the ``tessera`` package this *process imports*.  Gate 4
    resolves the two encoder scripts through ``$TESSERA_REPO``.  Nothing bound
    those two together, so a run could satisfy the pin with one Tessera on
    ``PYTHONPATH`` while a *different* checkout wrote the wire -- exactly the
    rendering/execution split-brain principle 8 exists to stop, and a hole the
    move from a release tag to a commit pin OPENS rather than closes (before
    it, the lane could not build at all).

    The check is the same predicate, applied to the checkout: the contract
    that checkout packages must hash to ``pin.contract_sha256``.
    """
    from pathlib import Path

    packaged = tel.packaged_contract_path()
    # Derived, not typed: the repo-relative location comes from the installed
    # package's own path tail, so a layout change is visible here.
    suffix = Path(*packaged.parts[-3:])
    good = tessera_repo / "src" / suffix
    assert good.is_file(), (
        "the fixture checkout must package the contract the pin names")
    assert tel.require_producer_repo_is_pinned() == (str(tessera_repo),)

    # A different Tessera under $TESSERA_REPO is refused, even though the
    # importable one still satisfies gate 1.
    good.write_text(good.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(tel.TesseraExportLaneError) as excinfo:
        tel.require_producer_repo_is_pinned()
    message = str(excinfo.value)
    assert "TESSERA_REPO" in message
    assert "pinned" in message

    # ...and so is a checkout that packages no contract at all: absence is not
    # read as agreement.
    good.unlink()
    with pytest.raises(tel.TesseraExportLaneError) as excinfo:
        tel.require_producer_repo_is_pinned()
    assert "packages no" in str(excinfo.value)


def test_the_preflight_runs_the_producer_repo_gate(
        tmp_path, tessera_repo, released_pin, monkeypatch):
    """The gate is wired, not merely defined: a stray checkout stops a build."""
    target = _dense_target()
    calls = []
    monkeypatch.setattr(
        tel, "require_producer_repo_is_pinned",
        lambda *a, **k: calls.append(True) or ("checked",))
    tel.preflight(_model_dir(tmp_path), target=target)
    assert calls, "preflight did not run the producer-repo gate"
