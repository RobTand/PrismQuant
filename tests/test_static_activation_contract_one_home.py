"""One home for "does this route execute a static activation contract" (#221).

PR #213 (#205) made "which activation quantiser does this spec serve" a
property of the spec -- ``FormatSpec.static_activation_contract``, reached
through the registry row that owns it -- and moved the assignment-KL hooks and
the production cache scorer onto it.  Three call sites kept answering the same
question by comparing the route's *source-format name* against ``"NVFP4"``:
``tessera_campaign._measure_anchor``, ``tessera_campaign``'s own
``_format_executes_static_nvfp4``, and ``tessera_export_lane``'s
``--input-scales`` gate.

Nothing moves today, and the first test below is the proof: over all 80
registry rows the two predicates coincide exactly, and ``NVFP4`` is the only
row carrying a contract.  The rest of the file is the failure that is one
registry line away.  ``StaticActivationContract`` is a general dataclass with
``execution``/``group_size`` fields, and ``FormatSpec`` types the field as a
spec-level property rather than an NVFP4 flag -- so the day a SECOND row gets
one, a name compare says "not NVFP4" while the row says "static contract", and
the two halves of #204/#205 disagree about the same rung: the scorer and the
KL hooks refuse to price it, or the export ships a unit whose static A-side
scale nothing ever bound.

``fp8_gains_a_static_contract`` builds exactly that world -- the issue's own
worked example, ``FP8_E4M3`` gaining a static per-tensor scale -- so every
assertion here is a real regression rather than a pin.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

torch = pytest.importorskip("torch")

from prismaquant import format_registry as fr
from prismaquant import nvfp4_activation_contract as owner
from prismaquant.tessera_formats import (
    parse_tessera_format_name,
    route_static_activation_contract,
    tessera_serving_route,
    tessera_wire_recipe,
)

#: An E2M1 rung over a per-16 block plane: the NVFP4 A side, the one static
#: contract that exists today.
W4A4 = "TESSERA_E2M1_K2_R896"
#: An E4M3 rung over a CHANNEL plane: dynamic W8A8 today, and the rung the
#: issue's worked example turns into a static-contract route.
W8A8 = "TESSERA_E4M3_K1_R1024"
#: A BF16 rung: no A-side registry row at all, so no contract either way.
A16 = "TESSERA_BF16_K1_R4096"

UNIT = "model.layers.0.self_attn.o_proj"
TRIPLE = {"text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64,
          "fit_tokens": 4096}


def _route(format_name: str):
    family, rung = parse_tessera_format_name(format_name)
    return tessera_serving_route(family, tessera_wire_recipe(family, rung), rung)


@pytest.fixture
def fp8_gains_a_static_contract(monkeypatch):
    """The hazard #221 names, made real: a SECOND row gets a contract.

    One ``dataclasses.replace`` in a distant file -- which is the point.  The
    execution string is deliberately NOT an NVFP4 one, so a refusal that still
    says "NVFP4" is visibly restating a name instead of reading the contract.
    ``group_size`` stays 16 only so the contract's oracle remains callable;
    nothing here asserts anything about FP8 numerics.
    """
    row = fr.REGISTRY["FP8_E4M3"]
    assert row.static_activation_contract is None, (
        "fixture precondition: FP8_E4M3 must not already carry a contract")
    contract = owner.StaticActivationContract(
        execution="e4m3_per_tensor_static", group_size=16)
    monkeypatch.setitem(
        fr.REGISTRY, "FP8_E4M3",
        dataclasses.replace(row, static_activation_contract=contract))
    return contract


# --------------------------------------------------------------- nothing moves

def test_the_two_predicates_coincide_over_the_whole_registry_today():
    """The auditor's proof, kept as a pin: this fix changes no resolution.

    If a second row ever gains a contract this test is the one that must be
    updated -- deliberately, by whoever adds it -- and by then every consumer
    already reads the row rather than the name.
    """
    rows = sorted(fr.REGISTRY)
    assert len(rows) == 80, (
        f"the registry grew to {len(rows)} rows; re-derive the equivalence "
        "below before trusting it")
    by_name = {n for n in rows if fr.canonical_format_name(n) == "NVFP4"}
    by_contract = {n for n in rows
                   if fr.REGISTRY[n].static_activation_contract is not None}
    assert by_name == by_contract == {"NVFP4"}


def test_the_live_routes_are_unchanged_by_the_derivation():
    assert route_static_activation_contract(_route(W4A4)) is not None
    assert route_static_activation_contract(_route(W8A8)) is None
    assert route_static_activation_contract(_route(A16)) is None
    # A16's route names no A-side row at all; the accessor must not go looking.
    assert _route(A16).activation_source_format is None


def test_the_accessor_reads_the_row_the_route_names():
    """Not a hardcoded table: the contract returned IS the row's own object."""
    route = _route(W4A4)
    assert route.activation_source_format == "NVFP4"
    assert (route_static_activation_contract(route)
            is fr.REGISTRY["NVFP4"].static_activation_contract)


# ------------------------------------------------- one home, three consumers

def test_the_spec_answer_and_the_route_answer_are_the_same_answer(
        fp8_gains_a_static_contract):
    """The campaign asks the spec, the export lane asks the route.

    Two spellings are allowed only because one is computed from the other:
    ``synthesize_tessera_spec`` derives the spec's contract by calling
    ``route_static_activation_contract``.  This pins that they cannot drift --
    in both worlds, including the one where a second row has a contract.
    """
    for name in (W4A4, W8A8, A16):
        spec_says = fr.get_format(name).static_activation_contract is not None
        route_says = route_static_activation_contract(_route(name)) is not None
        assert spec_says == route_says, name


def test_a_second_contract_reaches_the_synthesized_spec(
        fp8_gains_a_static_contract):
    """The render half already read the row, so this passed before #221 --
    which is precisely why the other three sites disagreeing was a split."""
    contract = fr.get_format(W8A8).static_activation_contract
    assert contract is not None
    assert contract.execution == "e4m3_per_tensor_static"
    # A Tessera rung has no dynamic serving path, so the served oracle is the
    # measurement whatever the source row's own screen policy is.
    assert contract.measured_as_served is True
    assert fr.REGISTRY["FP8_E4M3"].static_activation_contract.measured_as_served is False


def test_the_campaign_predicate_answers_every_rung_as_the_name_compare_did():
    """Replacing the name compare moved no answer -- it only widened.

    Checked against the old body verbatim, because "nothing resolves
    differently" is the claim this change rests on.  On every input the helper
    is actually called with (``anchor.format_name``, always a Tessera rung) the
    two agree.  Where they differ, the old one *raised* and the new one
    answers: the old body unpacked ``parse_tessera_format_name`` without
    checking for None, so it died with ``TypeError`` on any non-Tessera name;
    a plain registry row now gets the row's own answer.  A widening of the
    domain, not a changed answer -- and after #218 made ``canonical_format_name``
    resolve case-insensitively, that includes ``INT4_W4A16_g128``, the one
    mixed-case registered row, which must classify False rather than raise.
    """
    from prismaquant.tessera_campaign import (
        _format_executes_static_activation_contract as new,
    )

    def old(format_name):
        family, rung = parse_tessera_format_name(format_name)
        return tessera_serving_route(
            family, tessera_wire_recipe(family, rung), rung
        ).activation_source_format == "NVFP4"

    for name in (W4A4, W8A8, A16):
        assert new(name) == old(name), name

    # Non-Tessera rows: the old body raised, the new one answers.
    for name, expected in (("NVFP4", True), ("FP8_E4M3", False),
                           ("BF16", False), ("INT4_W4A16_g128", False),
                           ("INT4_W4A16_G128", False)):
        assert new(name) is expected, name
        with pytest.raises(TypeError):
            old(name)


def test_the_campaign_predicate_follows_the_contract_not_the_name(
        fp8_gains_a_static_contract):
    from prismaquant.tessera_campaign import (
        _format_executes_static_activation_contract as executes,
    )

    assert executes(W4A4) is True
    assert executes(A16) is False
    # The name compare answers False here: "FP8_E4M3" is not "NVFP4".
    assert executes(W8A8) is True


def test_the_campaign_refuses_a_second_contract_without_its_scale(
        monkeypatch, tmp_path, fp8_gains_a_static_contract):
    """The #205 refusal, at a format whose name contains no "NVFP4".

    Without this the campaign scores the rung under the registry's dynamic
    quantiser and stamps no ``input_global_scale``, while the cache scorer and
    the KL hooks -- which already read the row -- refuse the same unit.  The
    refusal must also name the contract it read, not a format it did not.
    """
    from types import SimpleNamespace

    from prismaquant import tessera_campaign as campaign

    def _must_not_encode(*a, **k):
        raise AssertionError("the refusal must precede the encode")

    monkeypatch.setattr(campaign, "_encode_and_render", _must_not_encode)
    with pytest.raises(campaign.ActivationScaleContractError,
                       match="e4m3_per_tensor_static"):
        campaign._measure_anchor(
            qname="m.q_proj",
            weight=torch.randn(32, 256, dtype=torch.bfloat16),
            activations=torch.randn(8, 256),
            format_name=W8A8,
            cache=SimpleNamespace(weights={}, cache_dir=None),
            wire_dir=tmp_path, hessian_required=False,
            static_input_scale=None)


def test_the_resume_guard_follows_the_contract(fp8_gains_a_static_contract):
    """A resumed row of a second-contract rung is a pre-contract price too."""
    from prismaquant.tessera_campaign import (
        ActivationScaleContractError, CampaignAnchor, _require_resumable_anchor,
    )

    anchor = CampaignAnchor(
        qname="m.up", family="TESSERA_E4M3_K1", format_name=W8A8,
        body_rate_q256=1024, dloss=1e-3, dloss_stderr=0.0,
        memory_bytes=1000, bits_per_param=4.0,
        activation_contract="w8a8-dynamic-e4m3-channel",
        activation_quantized=True, wire_bytes=1000, seconds=1.0)
    with pytest.raises(ActivationScaleContractError,
                       match="pre-served-.*contract"):
        _require_resumable_anchor(anchor, {"m.up": 71.68})
    _require_resumable_anchor(
        dataclasses.replace(anchor, input_global_scale=71.68), {"m.up": 71.68})


def _assignment(tmp_path, format_name):
    """The minimal weights-only allocation the export gate will read."""
    path = tmp_path / "layer_config.json"
    path.write_text(json.dumps({
        UNIT: {"data_type": "tessera", "bits": 4,
               "tessera_format": format_name},
        "__prismaquant__": {
            "tessera_hessian": {"supplied": False, **TRIPLE}},
    }))
    return path


def test_the_export_gate_follows_the_contract_not_the_name(
        tmp_path, fp8_gains_a_static_contract):
    """Otherwise the export ships a unit whose static A-side scale nothing
    bound -- the failure #205 was opened to close, at a different name."""
    from prismaquant import tessera_export_lane as export

    with pytest.raises(export.TesseraExportLaneError,
                       match="static activation contract.*--input-scales"):
        export.require_priced_export_inputs(_assignment(tmp_path, W8A8))


def test_the_export_gate_still_ignores_a_route_with_no_contract(tmp_path):
    """No fixture: FP8_E4M3 has no contract, so the E4M3 rung needs no file."""
    from prismaquant import tessera_export_lane as export

    report = export.require_priced_export_inputs(_assignment(tmp_path, W8A8))
    assert report["input_scales_required"] is False
    assert report["static_activation_contract_units"] == 0


def test_the_export_gate_does_not_import_the_tessera_package(tmp_path):
    """``tessera_export_lane`` gates without importing ``tessera`` (its own
    module docstring), so the gate must not reach ``synthesize_tessera_spec``.

    ``route_static_activation_contract`` reads a plain registry row
    (``NVFP4``/``FP8_E4M3``) and never a synthesized Tessera spec, which is
    why the export lane uses it rather than ``fr.get_format(<rung>)``.
    """
    import prismaquant.tessera_render as tr

    def _must_not_synthesize(*a, **k):
        raise AssertionError(
            "the export gate reached synthesize_tessera_spec; that pulls the "
            "tessera package into a preflight built not to need it")

    original = tr.synthesize_tessera_spec
    tr.synthesize_tessera_spec = _must_not_synthesize
    try:
        from prismaquant import tessera_export_lane as export

        report = export.require_priced_export_inputs(_assignment(tmp_path, W8A8))
    finally:
        tr.synthesize_tessera_spec = original
    assert report["static_activation_contract_units"] == 0
