"""The trellis render seam: mandatory wire sink, and RTN paths untouched.

Design: /home/rob/dq-runs/trellis-render-design.md

These tests pin the two halves of the seam's contract:

* A trellis rung cannot render without a wire sink, and cannot render at all
  today -- the encoder is an unpinned research dependency and the refusal says
  so by name instead of substituting an RTN render.
* Every existing format renders byte-identically whether or not the new
  keyword exists, which is the non-negotiable precondition for promoting
  trellis at all.

The byte-identity test mirrors ``tests/test_col_weights_render_identity.py``,
which pins the same property for ``col_weights``.
"""
from __future__ import annotations

import pytest
import torch

from prismaquant.production_weight_cache import render_production_weight
from prismaquant.trellis_render import (
    EXECUTED_ACTIVATION_CONTRACT,
    TrellisEncoderUnavailableError,
    TrellisRenderError,
    TrellisRenderRecipe,
    TrellisWireSink,
    TrellisWireSinkError,
    build_trellis_pair_identity,
)

TRELLIS_FMT = "TCQ_E2M1_R512"
_H = "0" * 64
_H2 = "1" * 64


def _recipe(**overrides) -> TrellisRenderRecipe:
    base = dict(
        family="TCQ_E2M1_R256",
        body_rate_q256=512,
        layout="tight_offsets",
        schedule_identity_sha256=_H,
        alphabet_identity_sha256=_H,
        pre_render_recipe_identity_sha256=_H,
        encoder_source_sha256=_H,
        encoder_sb_chunk=24576,
        encoder_determinism_mode="off",
        encoder_tailbite_candidates=4,
        encoder_backend="eager",
    )
    base.update(overrides)
    return TrellisRenderRecipe(**base)


def _levers() -> dict[str, bool]:
    return {"gptq": False, "static_act_order": False, "joint_scale_opt": False}


# --------------------------------------------------------------------------
# The seam refuses, loudly and by name
# --------------------------------------------------------------------------

def test_trellis_render_without_a_wire_sink_is_refused():
    """A wire-less trellis render is not a cheaper render.

    It is a render whose artifact was discarded: the schedule, alphabets and
    scale plane exist only in the wire, so the returned tensor cannot be
    turned back into shippable bytes.
    """
    weight = torch.randn(64, 256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires an explicit wire sink"):
        render_production_weight(
            weight,
            TRELLIS_FMT,
            qname="model.layers.0.mlp.down_proj",
            activations={},
            levers=_levers(),
        )


def test_a_wire_sink_on_a_non_trellis_format_is_refused_not_ignored():
    """Ignoring the sink would leave the caller holding an empty carrier."""
    weight = torch.randn(64, 256, dtype=torch.bfloat16)
    sink = TrellisWireSink(
        qname="q", fmt=TRELLIS_FMT, expected_bytes=4096,
    )
    with pytest.raises(ValueError, match="non-trellis format"):
        render_production_weight(
            weight,
            "NVFP4",
            qname="model.layers.0.mlp.down_proj",
            activations={},
            levers=_levers(),
            trellis_wire_out=sink,
        )


def test_trellis_render_names_the_missing_encoder_instead_of_substituting():
    """The refusal must identify the dependency, not degrade to RTN."""
    weight = torch.randn(64, 256, dtype=torch.bfloat16)
    sink = TrellisWireSink(qname="q", fmt=TRELLIS_FMT, expected_bytes=4096)
    with pytest.raises(TrellisEncoderUnavailableError) as excinfo:
        render_production_weight(
            weight,
            TRELLIS_FMT,
            qname="q",
            activations={},
            levers=_levers(),
            col_weights=torch.ones(256),
            trellis_wire_out=sink,
        )
    message = str(excinfo.value)
    assert "stage5_encoder" in message
    assert "AGENTS.md:48-49" in message
    assert not sink.filled


def test_trellis_render_requires_col_weights():
    """The encoder objective is importance-weighted; an unweighted entry is
    not the artifact.  Same refusal shape as the CB lane's."""
    weight = torch.randn(64, 256, dtype=torch.bfloat16)
    sink = TrellisWireSink(qname="q", fmt=TRELLIS_FMT, expected_bytes=4096)
    with pytest.raises(TrellisRenderError, match="no col_weights"):
        render_production_weight(
            weight,
            TRELLIS_FMT,
            qname="q",
            activations={},
            levers=_levers(),
            col_weights=None,
            trellis_wire_out=sink,
        )


# --------------------------------------------------------------------------
# Every existing render path is untouched
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["BF16", "FP8_E4M3", "MXFP4"])
def test_existing_formats_render_byte_identically_without_the_new_keyword(fmt):
    """The promotion is a non-starter if it perturbs an RTN render.

    The trellis branch is keyed on ``parse_trellis_format_name``, which returns
    None for every registered format name, so these paths execute two
    comparisons more than before and nothing else.
    """
    torch.manual_seed(20260830)
    weight = torch.randn(64, 256, dtype=torch.bfloat16)
    qname = "model.layers.0.mlp.down_proj"
    first = render_production_weight(
        weight, fmt, qname=qname, activations={}, levers=_levers(),
    )
    second = render_production_weight(
        weight, fmt, qname=qname, activations={}, levers=_levers(),
        trellis_wire_out=None,
    )
    assert torch.equal(first, second)
    assert first.shape == weight.shape
    assert first.dtype == weight.dtype


# --------------------------------------------------------------------------
# The sink's own contract
# --------------------------------------------------------------------------

def test_sink_refuses_a_blob_that_disagrees_with_the_priced_bytes():
    """trellis_footprint's total_bytes is exact by construction, so a
    length mismatch means the DP was solved against a byte count the render
    cannot honour."""
    sink = TrellisWireSink(qname="q", fmt=TRELLIS_FMT, expected_bytes=128)
    with pytest.raises(TrellisWireSinkError, match="priced at 128"):
        sink.accept(b"\x00" * 127, recipe=_recipe())


def test_sink_is_single_assignment():
    """A reused sink would attribute one Linear's bytes to another's identity."""
    sink = TrellisWireSink(qname="q", fmt=TRELLIS_FMT, expected_bytes=8)
    sink.accept(b"\x01" * 8, recipe=_recipe())
    with pytest.raises(TrellisWireSinkError, match="already holds a blob"):
        sink.accept(b"\x02" * 8, recipe=_recipe())


def test_sink_refuses_a_wire_without_its_recipe():
    sink = TrellisWireSink(qname="q", fmt=TRELLIS_FMT, expected_bytes=8)
    with pytest.raises(TrellisWireSinkError, match="render recipe"):
        sink.accept(b"\x01" * 8, recipe=None)  # type: ignore[arg-type]


def test_sink_refuses_a_non_trellis_format():
    with pytest.raises(TrellisWireSinkError, match="not a trellis rung"):
        TrellisWireSink(qname="q", fmt="NVFP4", expected_bytes=8)


def test_rendered_wire_identity_is_the_hash_of_the_exact_bytes():
    import hashlib

    blob = bytes(range(64))
    sink = TrellisWireSink(qname="q", fmt=TRELLIS_FMT, expected_bytes=64)
    sink.accept(blob, recipe=_recipe())
    assert sink.rendered_wire_identity_sha256 == hashlib.sha256(blob).hexdigest()
    assert torch.equal(
        sink.as_wire_tensor(),
        torch.frombuffer(bytearray(blob), dtype=torch.uint8),
    )


# --------------------------------------------------------------------------
# The recipe is the identity, and sb_chunk is part of it
# --------------------------------------------------------------------------

def test_sb_chunk_changes_the_recipe_identity():
    """The chunked encoder normalizes branch costs by the CHUNK's mean
    (stage5_encoder.py:173 inside the loop at :744-748), and the campaign's
    phase6 provenance gate already refuses an arm whose sb_chunk differs.
    A cache hit that ignored it would not be provably the same blob."""
    assert _recipe().identity_sha256 != _recipe(
        encoder_sb_chunk=6144
    ).identity_sha256


def test_encoder_source_and_determinism_are_identity_bearing():
    assert _recipe().identity_sha256 != _recipe(
        encoder_source_sha256=_H2
    ).identity_sha256
    assert _recipe().identity_sha256 != _recipe(
        encoder_determinism_mode="on"
    ).identity_sha256


def test_determinism_mode_has_no_default():
    """The canonical campaign runs encode off and eval on; a defaulted answer
    to 'which mode produced this' is indistinguishable from a wrong one."""
    with pytest.raises(TrellisRenderError, match="determinism_mode"):
        _recipe(encoder_determinism_mode="auto")


def test_recipe_carries_the_executed_activation_contract():
    """A=W is forced on both lanes: torch._scaled_mm dispatches only fp4xfp4
    and fp8xfp8 on GB10, and neither lane has a BF16 fallback.  Making the
    contract a required field of the render recipe is what stops a W*A16
    measurement from being installed under an A=W route."""
    assert _recipe().activation_contract == "e2m1_group16_ue4m3_static"
    assert (
        _recipe(family="TCQ_E4M3_R256").activation_contract
        == "fp8_per_token_dynamic"
    )
    assert set(EXECUTED_ACTIVATION_CONTRACT) == {
        "TCQ_E2M1_R256",
        "TCQ_E4M3_R256",
    }
    with pytest.raises(TrellisRenderError, match="unknown trellis family"):
        _recipe(family="TCQ_E3M2_R256")


# --------------------------------------------------------------------------
# The pair identity disambiguates a shape-free, recipe-free cache key
# --------------------------------------------------------------------------

def _identity(**overrides):
    base = dict(
        qname="model.layers.0.mlp.down_proj",
        fmt=TRELLIS_FMT,
        shape=(64, 256),
        recipe=_recipe(),
        source_weight_sha256=_H,
        source_weight_dtype="torch.bfloat16",
        col_weights_sha256=_H,
        col_weights_shape=(256,),
        calibration_hash="calib",
        git_commit="deadbeef",
        producer_source_sha256=_H,
    )
    base.update(overrides)
    return build_trellis_pair_identity(**base)


def test_two_schedules_at_one_rung_are_distinguishable_on_one_cache_key():
    """The TCQ spelling is deliberately shape-free and recipe-free so fused
    aggregation can intersect member menus by name (trellis_menu.py:396-403).
    The cache key therefore cannot disambiguate two manifests; the pair
    identity must."""
    a = _identity()
    b = _identity(recipe=_recipe(schedule_identity_sha256=_H2))
    assert a["qname"] == b["qname"] and a["format"] == b["format"]
    assert a["recipe_identity_sha256"] != b["recipe_identity_sha256"]


def test_col_weights_are_identity_bearing():
    assert (
        _identity()["col_weights_sha256"]
        != _identity(col_weights_sha256=_H2)["col_weights_sha256"]
    )


def test_pair_identity_refuses_a_non_rank_2_shape():
    with pytest.raises(TrellisRenderError, match="rank-2 shape"):
        _identity(shape=(8, 64, 256))
