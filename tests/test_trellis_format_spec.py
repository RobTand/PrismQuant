"""UNWIRED_LINKS #1: ``get_format`` resolves trellis TCQ rungs by parsing.

The contract under test is exact-or-refuse (principles 1 and 2):

* the fields a consumer can trust — the activation contract, the capability
  floor, the trellis-native identity — resolve exactly;
* every RTN-shaped field with no trellis referent raises
  ``TrellisSpecFieldRefused`` at attribute READ, loudly enough to defeat the
  two silent-swallow layers real consumers wrap around format specs
  (``getattr(spec, ..., None)`` plus ``except Exception`` around the call —
  ``aura_cost._delta_w`` is the archetype).

A test that asserts a refusal fires is worth more than one asserting the
spec constructs: the silent-zero failure class (identity
``quantize_dequantize`` pricing reconstruction error at exactly 0.0) has
bitten this project twice already.
"""
from __future__ import annotations

import pytest

from prismaquant import format_registry as fr
from prismaquant.trellis_formats import (
    FAMILIES,
    TrellisFormatError,
    get_trellis_family,
)

E2M1 = "TCQ_E2M1_R640"     # 2.5 body bits/weight — deliberately non-integer
E4M3 = "TCQ_E4M3_R1152"    # 4.5 body bits/weight

REFUSED_FIELDS = (
    "weight_bits",
    "group_size",
    "scale_bits",
    "scale_dtype_name",
    "scale_block_shape",
    "autoround_config",
    "quantize_dequantize",
    "activation_quantize_dequantize",
    "effective_bits",
)

REFUSED_SHAPE_METHODS = (
    "memory_bytes_for_shape",
    "effective_bits_for_shape",
    "scale_count_for_shape",
)


# ---------------------------------------------------------------------------
# Resolution: a parse, not a registry insertion
# ---------------------------------------------------------------------------

def test_get_format_resolves_tcq_names_without_registry_entries():
    for name in (E2M1, E4M3):
        spec = fr.get_format(name)
        assert isinstance(spec, fr.TrellisFormatSpec)
        assert isinstance(spec, fr.FormatSpec)   # duck-type seam holds
        assert spec.name == name
        # Never inserted: enumeration is impossible (the rate axis is dense)
        # and registry membership would read as producer authority.
        assert name not in fr.REGISTRY
    assert fr.get_format(E2M1) is fr.get_format(E2M1)  # memoized


def test_trellis_identity_fields_are_exact():
    spec = fr.get_format(E2M1)
    assert spec.body_rate_q256 == 640
    assert spec.trellis_family is get_trellis_family("TCQ_E2M1_R256")
    assert spec.scale_contract == spec.trellis_family.scale_contract
    assert spec.family == "tcq"
    assert spec.weight_element_dtype == "trellis_e2m1"


def test_out_of_law_rate_raises_pointed_error_not_keyerror():
    # A well-spelled TCQ name with an illegal rate is a corrupt input, not an
    # unknown format; the family bounds live in the error.
    with pytest.raises(TrellisFormatError):
        fr.get_format("TCQ_E2M1_R100")
    with pytest.raises(TrellisFormatError):
        fr.get_format("TCQ_E2M1_R99999")
    # Non-TCQ spellings keep the historical KeyError contract.
    with pytest.raises(KeyError):
        fr.get_format("TCQ_BOGUS_R640")
    with pytest.raises(KeyError):
        fr.get_format("TCQ_E2M1_R")


# ---------------------------------------------------------------------------
# Fields that MUST be exact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family_key", sorted(FAMILIES))
def test_activation_contract_matches_the_terminal_format(family_key):
    """A=W is forced, not chosen: _scaled_mm on the declared targets
    dispatches only fp4xfp4 and fp8xfp8, so the A-side operand contract is
    exactly the terminal scalar format's. Asserted against the registry
    entry, not literals, so the two cannot silently drift apart."""
    family = FAMILIES[family_key]
    low, high = family.mathematical_q256_bounds
    spec = fr.get_format(family.format_name(low))
    terminal = fr.get_format(family.terminal_format)
    assert spec.act_bits == terminal.act_bits
    assert spec.act_dtype_name == terminal.act_dtype_name
    assert spec.act_group_size == terminal.act_group_size
    # The single predicate every consumer must use answers True: trellis has
    # no BF16-activation route, so a trellis rung is never "weight-only".
    assert spec.act_quant_changes_input is True


def test_capability_floor_comes_from_the_family():
    assert fr.get_format(E2M1).min_capability_sm == 120   # E2M1: SM120+
    assert fr.get_format(E4M3).min_capability_sm == 89    # E4M3: SM89+
    for key, family in FAMILIES.items():
        low, _ = family.mathematical_q256_bounds
        spec = fr.get_format(family.format_name(low))
        assert spec.min_capability_sm == family.minimum_capability_sm


def test_trellis_rungs_are_not_producer_eligible():
    """No ProductionWeightCache mechanism exists and export refuses TCQ, so
    the generic producer surface must refuse too."""
    spec = fr.get_format(E2M1)
    assert spec.producer_eligible is False
    assert fr.format_is_producer_eligible(E2M1) is False
    with pytest.raises(ValueError):
        fr.require_producer_formats([E2M1])


# ---------------------------------------------------------------------------
# Fields that MUST refuse — and refuse loudly enough
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", REFUSED_FIELDS)
def test_rtn_shaped_fields_refuse_at_attribute_read(field):
    spec = fr.get_format(E2M1)
    with pytest.raises(fr.TrellisSpecFieldRefused):
        getattr(spec, field)


@pytest.mark.parametrize("method", REFUSED_SHAPE_METHODS)
def test_byte_and_bpp_helpers_refuse(method):
    spec = fr.get_format(E2M1)
    with pytest.raises(fr.TrellisSpecFieldRefused):
        getattr(spec, method)((256, 512))


@pytest.mark.parametrize("field", REFUSED_FIELDS)
def test_getattr_with_default_cannot_swallow_the_refusal(field):
    """The money test. ``aura_cost._delta_w`` does
    ``getattr(spec, "quantize_dequantize", None)`` and treats None as "no
    delta"; an AttributeError-shaped refusal would become a silent skip and
    an identity default would become a silent zero. The refusal must
    therefore be a non-AttributeError raised at the READ."""
    spec = fr.get_format(E2M1)
    with pytest.raises(fr.TrellisSpecFieldRefused):
        getattr(spec, field, None)
    assert not issubclass(fr.TrellisSpecFieldRefused, AttributeError)
    assert not issubclass(fr.TrellisSpecFieldRefused, KeyError)


def test_repr_and_eq_do_not_touch_refused_fields():
    # The dataclass __repr__/__eq__ walk every field; the overrides must not.
    spec = fr.get_format(E2M1)
    assert "TCQ_E2M1_R640" in repr(spec)
    assert spec == fr.get_format(E2M1)
    assert spec != fr.get_format(E4M3)
    assert spec != fr.get_format("NVFP4")
    assert fr.get_format("NVFP4") != spec


# ---------------------------------------------------------------------------
# Real consumers, exercised end-to-end
# ---------------------------------------------------------------------------

def test_aura_delta_w_rtn_fallback_raises_instead_of_returning_none():
    """aura_cost._delta_w's RTN fallback wraps the qdq CALL in
    ``except Exception: return None`` — a raising callable would be
    swallowed into "no delta" and the rung silently dropped from pricing.
    The refusal at attribute access must propagate out instead."""
    torch = pytest.importorskip("torch")
    from prismaquant.aura_cost import _delta_w

    weight = torch.randn(32, 32)
    with pytest.raises(fr.TrellisSpecFieldRefused):
        _delta_w("layer.q_proj", E2M1, weight, cache=None)


def test_payload_breakdown_refuses_the_formatspec_byte_formula():
    """Pins the NEW shape of UNWIRED_LINKS #6 (footprint byte-budget lookup)
    and the fall-through half of #2 (the allocator's exact payload filter
    calls serialized_candidate_payload -> format_tensor_payload_breakdown):
    no KeyError anymore, and no plausible ceil(n*bits/8) number either — a
    pointed refusal naming where exact trellis bytes live, mirroring the CB
    precedent in the same function."""
    from prismaquant.footprint import format_tensor_payload_breakdown

    spec = fr.get_format(E2M1)
    with pytest.raises(fr.TrellisSpecFieldRefused) as exc:
        format_tensor_payload_breakdown(spec, (256, 512), qname="layer.q_proj")
    assert "trellis_tensor_payload_breakdown" in str(exc.value)

    from prismaquant.allocator_candidates import serialized_candidate_payload

    with pytest.raises(fr.TrellisSpecFieldRefused):
        serialized_candidate_payload(
            spec, (256, 512), qname="layer.q_proj",
            cb_serialization_context=None,
        )


def test_bit_exact_short_circuit_stays_false_for_the_right_reason():
    """Before: False because get_format KeyError'd. After: False because a
    trellis rung quantizes activations (A=W), so even a bit-identical weight
    render changes the layer output through the A side."""
    from prismaquant.allocator_candidates import (
        _format_act_quant_changes_input,
        cost_entry_is_bit_exact,
    )

    assert cost_entry_is_bit_exact({"weight_mse": 0.0}, E2M1) is False
    # ...and the A-side visibility this change buys: the predicate used to
    # return False via the KeyError fallback, which was WRONG for trellis.
    assert _format_act_quant_changes_input(E2M1) is True
    assert _format_act_quant_changes_input(E4M3) is True


def test_batched_rtn_render_cannot_round_a_trellis_rung():
    """measure_quant_cost's batched render dispatches on
    weight_element_dtype; 'trellis_*' is deliberately unknown to every
    RTN/codebook/int branch so an integer-RTN render of a trellis rung is
    impossible rather than silently wrong."""
    torch = pytest.importorskip("torch")
    from prismaquant.measure_quant_cost import _batched_quantize

    spec = fr.get_format(E2M1)
    stacked = torch.randn(2, 32, 32)
    with pytest.raises((ValueError, fr.TrellisSpecFieldRefused)):
        _batched_quantize(spec, stacked)
