"""The trellis families' DECLARED activation contract, and the refusal that
makes an unpriced A side visible instead of free.

Two halves, matching the two defects:

  * the declaration -- ``TrellisFamily`` now states ``act_bits`` /
    ``act_dtype_name`` / ``act_group_size`` in ``FormatSpec``'s own spelling,
    and those values must equal the terminal format's registry entry. Both
    Gridbook trellis lanes execute A=W (``TrellisE2M1LinearMethod.apply`` and
    ``TrellisE4M3LinearMethod.apply`` each call ``torch._scaled_mm`` with both
    operands in the family's grid dtype; neither has a BF16-activation route).
    That kernel fact travels here as a declaration plus this equality test --
    AGENTS.md:38 forbids importing the runtime to ask it, and nothing in this
    file imports gridbook.
  * the refusal -- a format that declares ``act_cost_required`` and is handed a
    row with no activation price refuses, rather than contributing 0.0 and
    being selected at a discount to the route the runtime executes.
"""
from __future__ import annotations

import dataclasses

import pytest

from prismaquant import allocator_candidates as ac
from prismaquant import format_registry as fr
from prismaquant import trellis_formats as tf


# --------------------------------------------------------------------------
# (A) the declaration
# --------------------------------------------------------------------------
def test_every_trellis_family_declares_an_activation_grid():
    """No family may be silent about its A side: both lanes are A=W."""
    assert tf.FAMILIES
    for family in tf.FAMILIES.values():
        assert family.act_bits is not None, family.family
        assert family.act_bits < 16, family.family
        assert family.quantizes_activations is True, family.family
        assert family.act_dtype_name, family.family


@pytest.mark.parametrize("family_name", sorted(tf.FAMILIES))
def test_declared_activation_grid_equals_the_terminal_formats(family_name):
    """The rung is served on its TERMINAL format's activation grid.

    A trellis rung spends fractional bits per weight on the wire but decodes
    onto the family's grid, and it is the grid that reaches ``_scaled_mm``. So
    the declaration is not free to drift from the registry entry for the
    terminal format -- if it did, the same activation path would be priced two
    ways depending on which module was asked.
    """
    family = tf.FAMILIES[family_name]
    spec = fr.get_format(family.terminal_format)
    assert (family.act_bits, family.act_dtype_name, family.act_group_size) == (
        spec.act_bits, spec.act_dtype_name, spec.act_group_size)
    assert family.quantizes_activations == spec.act_quant_changes_input


def test_served_activation_contract_is_w4a4_and_w8a8():
    """The established lane fact, pinned as a value."""
    assert tf.FAMILIES[tf.E2M1_FAMILY].served_activation_contract == "W4A4"
    assert tf.FAMILIES[tf.E4M3_FAMILY].served_activation_contract == "W8A8"


def test_the_contract_payload_carries_the_activation_declaration():
    """The declaration must reach the receipt, or it is not readable."""
    payload = tf.format_contract_payload()
    families = {entry["family"]: entry for entry in payload["families"]}
    assert families
    for name, entry in families.items():
        declared = tf.FAMILIES[name]
        assert entry["act_bits"] == declared.act_bits
        assert entry["act_dtype_name"] == declared.act_dtype_name
        assert entry["act_group_size"] == declared.act_group_size
        assert entry["quantizes_activations"] is True
        assert entry["act_cost_required"] is True
        assert entry["served_activation_contract"] == (
            declared.served_activation_contract)


# --------------------------------------------------------------------------
# (B) the refusal
# --------------------------------------------------------------------------
QNAME = "model.layers.0.self_attn.q_proj"

#: A row priced the way AURA prices one: a positive weight-side number, no
#: measured output_mse. This is the ordinary case -- NOT the exactly-0.0
#: corner ``cost_entry_prices_unmeasured_activation_at_zero`` already owns.
WEIGHT_ONLY_ROW = {"predicted_dloss": 1e-3, "output_mse_measured": False}


def _stats(h_trace: float = 4.0):
    return {QNAME: {"h_trace": h_trace, "shape": [512, 512]}}


def _declaring_spec(name: str = "NVFP4") -> fr.FormatSpec:
    """A real registry spec, re-declared as requiring an A-side price."""
    return dataclasses.replace(fr.get_format(name), act_cost_required=True)


def _build(spec, cost_row, stats=None, mask_records=None, extra=()):
    """Build a menu for one unit.

    ``extra`` adds further (spec, row) pairs so the unit is not STARVED when
    the format under test is refused -- starvation is its own fail-closed
    path and is asserted separately.
    """
    specs = [spec, *(s for s, _ in extra)]
    rows = {spec.name: dict(cost_row)}
    for other, other_row in extra:
        rows[other.name] = dict(other_row)
    return ac.build_candidates(
        stats if stats is not None else _stats(),
        {QNAME: rows},
        specs,
        mask_records=mask_records,
    )


def _patch_registry(monkeypatch, spec):
    monkeypatch.setattr(fr, "get_format", lambda n: spec if n == spec.name
                        else fr.REGISTRY[n])


def test_declaring_format_without_an_activation_price_refuses(monkeypatch):
    """The load-bearing check: declared A side + no act_dloss -> no candidate.

    The refusal must NAME the unit and the format, and say what measurement is
    missing -- it is a measurement gap made visible, not a format ban.
    """
    spec = _declaring_spec()
    _patch_registry(monkeypatch, spec)
    # A second, non-declaring rung keeps the unit off the starvation path so
    # this test observes the MASK rather than the raise.
    other = fr.get_format("FP8_E4M3")
    records: list[dict] = []
    out = _build(spec, WEIGHT_ONLY_ROW, mask_records=records,
                 extra=((other, WEIGHT_ONLY_ROW),))

    assert spec.name not in {c.fmt for c in out[QNAME]}
    assert other.name in {c.fmt for c in out[QNAME]}

    hit = [r for r in records
           if r["reason"] == ac.ACTIVATION_COST_UNDECLARED_REASON]
    assert len(hit) == 1, records
    detail = hit[0]["detail"]
    assert hit[0]["qname"] == QNAME
    assert hit[0]["format"] == spec.name
    assert spec.name in detail
    assert "act_dloss" in detail
    assert "aqua_activation_cost" in detail
    # It must read as a missing measurement, not a policy ban.
    assert "MISSING" in detail and "not a ban" in detail
    # The full remedy, and the prose limitation this gate is the machine
    # half of -- named so the two cannot drift apart.
    assert "merge_act_dloss" in detail
    assert "activation_blindness_limitation" in detail


def test_refusing_the_only_rung_starves_the_unit_loudly(monkeypatch):
    """A refused unit is never silently dropped from the menu.

    A name absent from the returned dict never reaches the DP, so its bits and
    bytes vanish from the bpp/footprint accounting while the export still
    emits the tensor. The existing starvation raise owns that, and it must
    name THIS reason's remedy rather than the other refusal's.
    """
    spec = _declaring_spec()
    _patch_registry(monkeypatch, spec)
    with pytest.raises(AssertionError) as exc:
        _build(spec, WEIGHT_ONLY_ROW)
    message = str(exc.value)
    assert QNAME in message
    assert ac.ACTIVATION_COST_UNDECLARED_REASON in message
    assert "aqua_activation_cost" in message
    assert "merge_act_dloss" in message
    assert "activation_blindness_limitation" in message
    # The OTHER refusal's remedy is not offered for this gap.
    assert "PRISMAQUANT_EXPERT_COST_SAMPLE" not in message


def test_a_measured_activation_price_is_admitted(monkeypatch):
    """The same row, priced, is a legal candidate -- and the A side is added."""
    spec = _declaring_spec()
    monkeypatch.setattr(fr, "get_format", lambda n: spec if n == spec.name
                        else fr.REGISTRY[n])
    priced = dict(WEIGHT_ONLY_ROW, act_dloss=2e-3)
    records: list[dict] = []
    out = _build(spec, priced, mask_records=records)

    assert [r for r in records
            if r["reason"] == ac.ACTIVATION_COST_UNDECLARED_REASON] == []
    assert out[QNAME], "a priced row must stay on the menu"
    cand = out[QNAME][0]
    assert cand.fmt == spec.name
    # The A side is summed into the price, not dropped on the floor.
    weight_only = ac.cost_entry_weight_only_dloss(
        _stats()[QNAME], dict(WEIGHT_ONLY_ROW))
    assert cand.predicted_dloss == pytest.approx(weight_only + 2e-3)


def test_a_measured_activation_price_of_zero_is_an_answer(monkeypatch):
    """Key presence, not truthiness.

    ``cost_entry_act_dloss`` reads ``.get(KEY, 0.0) or 0.0``, which collapses a
    measured 0.0 with an absent key. Those are opposite statements and the
    refusal must tell them apart, or measuring an honestly-zero A side would
    look identical to never measuring it.
    """
    spec = _declaring_spec()
    monkeypatch.setattr(fr, "get_format", lambda n: spec if n == spec.name
                        else fr.REGISTRY[n])
    records: list[dict] = []
    out = _build(spec, dict(WEIGHT_ONLY_ROW, act_dloss=0.0),
                 mask_records=records)
    assert [r for r in records
            if r["reason"] == ac.ACTIVATION_COST_UNDECLARED_REASON] == []
    assert out[QNAME]


def test_zero_sensitivity_rows_stay_exempt(monkeypatch):
    """``h_trace == 0`` prices BOTH sides at zero through one Fisher expansion.

    Same exemption ``cost_entry_prices_unmeasured_activation_at_zero`` carries:
    a zero-token expert at thin calibration must stay free to take the
    cheapest format rather than being forced onto a passthrough.
    """
    spec = _declaring_spec()
    monkeypatch.setattr(fr, "get_format", lambda n: spec if n == spec.name
                        else fr.REGISTRY[n])
    records: list[dict] = []
    out = _build(spec, WEIGHT_ONLY_ROW, stats=_stats(h_trace=0.0),
                 mask_records=records)
    assert [r for r in records
            if r["reason"] == ac.ACTIVATION_COST_UNDECLARED_REASON] == []
    assert out[QNAME]


def test_a_measured_output_mse_row_is_not_refused(monkeypatch):
    """That branch already saw the activation path.

    ``measure_quant_cost`` and ``incremental_measure_quant_cost`` both push the
    batch through ``spec.activation_quantize_dequantize`` before measuring, so
    a measured ``output_mse`` is activation-inclusive and a separate A term
    would double-count it.
    """
    spec = _declaring_spec()
    monkeypatch.setattr(fr, "get_format", lambda n: spec if n == spec.name
                        else fr.REGISTRY[n])
    records: list[dict] = []
    out = _build(spec, {"output_mse": 5e-4, "output_mse_measured": True},
                 mask_records=records)
    assert [r for r in records
            if r["reason"] == ac.ACTIVATION_COST_UNDECLARED_REASON] == []
    assert out[QNAME]


# --------------------------------------------------------------------------
# (C) no regression: a format that does NOT declare behaves exactly as before
# --------------------------------------------------------------------------
def test_no_registered_format_declares_act_cost_required_today():
    """The stricter contract is opt-in, and nothing has opted in yet.

    This is what keeps every pre-AQUA cost artifact reproducible (principle 6):
    ``run-pipeline.sh`` runs no AQUA stage, so on the default recipe every
    NVFP4/FP8 row lacks ``act_dloss``, and keying the refusal on
    ``act_quant_changes_input`` alone would empty the default menu. Flipping a
    production format's flag is the enforcement step of the no-A16 direction
    and must land WITH an A-side measurement -- if this test starts failing,
    that is the change, and it needs its own served evidence.
    """
    declaring = sorted(name for name, spec in fr.REGISTRY.items()
                       if getattr(spec, "act_cost_required", False))
    assert declaring == []


@pytest.mark.parametrize("fmt", ["NVFP4", "FP8_E4M3"])
def test_activation_quantizing_format_without_the_flag_is_unchanged(fmt):
    """The exact shape of the pre-existing menu, byte for byte.

    NVFP4 and FP8_E4M3 both quantize activations and both are priced from
    AURA rows with no ``act_dloss``. Without the opt-in flag they must be
    admitted at exactly their weight-only price, as today.
    """
    spec = fr.get_format(fmt)
    assert spec.act_quant_changes_input is True
    assert spec.act_cost_required is False
    records: list[dict] = []
    out = _build(spec, WEIGHT_ONLY_ROW, mask_records=records)
    assert records == []
    assert out[QNAME]
    weight_only = ac.cost_entry_weight_only_dloss(
        _stats()[QNAME], dict(WEIGHT_ONLY_ROW))
    assert out[QNAME][0].predicted_dloss == pytest.approx(weight_only)


def test_predicate_ignores_unregistered_and_undeclaring_formats():
    """Unit-level: the predicate itself, without the menu around it."""
    stats_entry = _stats()[QNAME]
    # Unknown format name -> not our business.
    assert ac.cost_entry_omits_declared_activation_cost(
        stats_entry, dict(WEIGHT_ONLY_ROW), "NOT_A_REAL_FORMAT") is False
    # Declared nothing -> legacy 0.0 tolerance, unchanged.
    assert ac.cost_entry_omits_declared_activation_cost(
        stats_entry, dict(WEIGHT_ONLY_ROW), "NVFP4") is False
    # No format name at all -> unchanged.
    assert ac.cost_entry_omits_declared_activation_cost(
        stats_entry, dict(WEIGHT_ONLY_ROW), None) is False


# --------------------------------------------------------------------------
# (D) the trellis seam: the anchor/serving mismatch is a gate input, not a log
# --------------------------------------------------------------------------
def test_the_unwired_ledger_names_the_unpriced_activation_side():
    """The ledger is the re-enable checklist AND the text of the refusal.

    Before this entry the ledger's eight items were all about wiring; none
    said that the anchors price a weight-only loss while both lanes execute
    A=W. Deleting the other eight would then have enabled a seam that sells
    every rung at a discount to the route the runtime runs. The entry is what
    stops that, so it must exist and must be reachable in the refusal text.
    """
    from prismaquant import trellis_menu as tm

    entries = {where: what for where, what in tm.UNWIRED_LINKS}
    hits = [what for where, what in tm.UNWIRED_LINKS
            if "A=W" in what and "activation" in what]
    assert hits, sorted(entries)
    what = hits[0]
    assert "act_cost_required" in what
    assert "cost_entry_omits_declared_activation_cost" in what


def test_the_seam_refusal_repeats_the_activation_entry(tmp_path, monkeypatch):
    """A ledger entry nothing reads would be the very defect it describes."""
    from prismaquant import trellis_menu as tm
    from test_trellis_menu import (  # tests/ is on sys.path under pytest
        UNIT_A, scalar_menu, stats_for, write_manifest)

    units = {UNIT_A: (1024, 512)}
    manifest = write_manifest(tmp_path, units)
    monkeypatch.setenv(tm.TRELLIS_SURFACE_ENV, str(manifest))
    with pytest.raises(tm.TrellisSeamUnwiredError) as exc:
        tm.augment_candidates(scalar_menu(units), stats_for(units),
                              cost_mode="aura")
    message = str(exc.value)
    assert "A=W" in message
    assert f"{len(tm.UNWIRED_LINKS)} links are missing" in message


def test_the_menu_receipt_carries_both_activation_contracts(tmp_path):
    """Anchor contract AND served contract, as values a gate can compare.

    They disagree today -- the anchors are W8A16 and the E4M3 lane serves
    W8A8 -- and that disagreement is exactly the size of the discount an
    un-refused seam would hand the DP. Stamping only the anchor half, as the
    seam did, leaves the comparison to whoever remembers to make it.
    """
    from prismaquant import trellis_menu as tm
    from test_trellis_menu import (  # tests/ is on sys.path under pytest
        UNIT_A, scalar_menu, stats_for, write_manifest)

    units = {UNIT_A: (1024, 512)}
    manifest = write_manifest(tmp_path, units)
    prov: dict = {}
    tm.build_trellis_menu(scalar_menu(units), stats_for(units),
                          cost_mode="aura", manifest_path=str(manifest),
                          provenance_out=prov)
    assert prov["anchor_activation_contract"] == "W8A16"
    assert prov["served_activation_contracts"] == {tf.E4M3_FAMILY: "W8A8"}
    assert prov["anchor_activation_contract_matches_serving"] is False


def test_an_unresolvable_family_does_not_crash_the_receipt(tmp_path):
    """A malformed anchor is a counted per-unit skip, not a menu-wide crash.

    The receipt resolves each anchor's declared serving contract. Doing that
    eagerly over raw manifest rows would let one bad `family` string raise from
    inside the stamp -- promoting a tolerated skip into a whole-menu failure
    and taking the good units down with it.
    """
    from prismaquant import trellis_menu as tm
    from test_trellis_menu import (  # tests/ is on sys.path under pytest
        UNIT_A, manifest_payload, scalar_menu, stats_for)
    import json

    units = {UNIT_A: (1024, 512)}
    payload = manifest_payload(units)
    payload["anchors"]["model.layers.0.mlp.down_proj"] = {
        "family": "TCQ_NOT_A_FAMILY", "alphabets": {}, "points": []}
    path = tmp_path / "surface.json"
    path.write_text(json.dumps(payload))

    prov: dict = {}
    tm.build_trellis_menu(scalar_menu(units), stats_for(units),
                          cost_mode="aura", manifest_path=str(path),
                          provenance_out=prov)
    # The unknown family is left out rather than raising...
    assert "TCQ_NOT_A_FAMILY" not in prov["served_activation_contracts"]
    assert prov["served_activation_contracts"] == {tf.E4M3_FAMILY: "W8A8"}
    # ...and an unresolved family cannot be reported as agreeing.
    assert prov["anchor_activation_contract_matches_serving"] is False
