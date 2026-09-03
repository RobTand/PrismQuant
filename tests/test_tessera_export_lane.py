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
    require_exact_tessera_runtime_release,
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
        pin_module, "TESSERA_SERVING_RUNTIME_RELEASE_COMMIT", FIXTURE_COMMIT)
    monkeypatch.setattr(
        pin_module, "TESSERA_SERVING_RUNTIME_RELEASE_VERSION", FIXTURE_VERSION)
    pin = pin_module.load_tessera_serving_runtime_pin(path)
    monkeypatch.setattr(
        pin_module, "load_tessera_serving_runtime_pin", lambda *a, **k: pin)
    require_exact_tessera_runtime_release(pin)   # the real gate, satisfied
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

    root = tmp_path / "tessera-checkout"
    for tool in load_lane_spec("tessera").producer_tools:
        path = root / tool.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    monkeypatch.setenv("TESSERA_REPO", str(root))
    return root


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
def test_the_preflight_refuses_on_the_pending_pin_and_says_so(tmp_path):
    """Admission is fail-closed BY THE PIN, and the message is actionable."""
    with pytest.raises(tel.TesseraExportLaneError) as excinfo:
        tel.require_release_pin()
    message = str(excinfo.value)
    assert "PENDING" in message
    assert "RobTand/tessera#17" in message          # names the blocker
    assert "tessera_serving_runtime_pin.json" in message   # names the fix

    # ... and it is the gate the whole preflight ends on.
    assert tel.main(["--model", str(_model_dir(tmp_path))]) == 2


def test_under_a_released_pin_the_whole_preflight_passes(
        released_pin, tmp_path, tessera_repo):
    """The other half of "refused by the pin".

    With ONLY the release boundary satisfied -- the real packaged contract,
    the real lane spec, the real structure read -- a dense checkpoint clears
    every gate.  That is what proves the refusal above is the pin's and not an
    artefact of an unreadable table or an undeclared structure.
    """
    report = tel.preflight(_model_dir(tmp_path))
    assert report["structure"] == "dense"
    assert report["quant_method"] == "tessera"
    assert report["executes"] == sorted(
        r["name_pattern"].replace("{k}", "*")
        for r in json.loads(tel.packaged_contract_path().read_text())["formats"]
    )
    assert tel.main(["--model", str(_model_dir(tmp_path))]) == 0


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


def test_a_routed_moe_checkpoint_is_refused_by_the_contracts_own_field(
        released_pin, tmp_path):
    """Absence is the honest state: no served measurement covers routed
    experts, so the contract declares `dense` and nothing here widens it."""
    directory = _model_dir(tmp_path, num_experts=128,
                           architectures=["Qwen3MoeForCausalLM"])
    with pytest.raises(tel.TesseraExportLaneError) as excinfo:
        tel.require_declared_structure(directory)
    message = str(excinfo.value)
    assert "routed_moe" in message and "['dense']" in message
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
