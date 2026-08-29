"""CPU acceptance gates for one learned/lattice FP8-CB rung menu."""

from __future__ import annotations

import copy

import pytest
import torch

from prismaquant import cb_learned_bundle as learned_bundle
from prismaquant import format_registry as fr
from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_layout import parse_format_name, subtable_bit_widths
from prismaquant.cb_export_config import build_quant_config
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_assignment_payload_breakdown,
    cb_cost_provenance,
    cb_payload_summary,
    cb_quantize_dequantize_for_context,
    cb_serialization_context_from_env,
    cb_serialization_context_stamp,
    codebook_source_for_format,
    validate_cb_cost_provenance,
    validate_cb_serialization_context_stamp,
)


QNAME = "model.layers.0.self_attn.q_proj"
QNAME_K47 = "model.layers.0.self_attn.k_proj"
FORMATS = tuple(f"FP8_CB_K{rung}" for rung in range(28, 49))


def _fast_distinct_learn_pool(weight, col_weights, rung):
    """Valid grid books whose render cannot be confused with the lattice."""

    del weight, col_weights
    parsed = parse_format_name(f"FP8_CB_K{int(rung)}")
    assert parsed is not None
    family, k = parsed
    widths = subtable_bit_widths(k, family.mode, family.n_sub)
    return tuple(
        torch.zeros_like(cb.fixed_lattice(bits, "fp8", 2))
        for bits in widths
    )


@pytest.fixture
def policy_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(
        learned_bundle, "learn_pool", _fast_distinct_learn_pool
    )
    weight = torch.linspace(
        -1.75, 1.75, 2 * 256, dtype=torch.float32
    ).reshape(2, 256)
    col_weights = torch.linspace(0.25, 1.25, 256, dtype=torch.float32)
    bundle = learned_bundle.train_and_save_bundle_streaming(
        tmp_path / "fp8-k28-k48.pqcb",
        qnames=(QNAME,),
        weight_provider=lambda qname: weight,
        col_weights={QNAME: col_weights},
        formats=FORMATS,
    )
    return bundle, weight, col_weights


def _bundle_context(bundle):
    return cb_serialization_context_from_env({
        "CB_CODEBOOK_SOURCE": "learned",
        "CB_CODEBOOK_SOURCE_SCOPE": "fp8",
        "CB_CODEBOOK_BUNDLE": str(bundle.path),
        "CB_SCALE_SWEEP_SCOPE": "all",
    })


def test_policy_bundle_carries_learned_k28_k46_and_lattice_k47_k48(
    policy_bundle,
):
    bundle, _weight, _col_weights = policy_bundle
    expected_sources = {
        format_name: (
            "learned"
            if learned_bundle.CBL_RUNG_POLICY[
                int(format_name.rsplit("K", 1)[1])
            ]["enabled"]
            else "lattice"
        )
        for format_name in FORMATS
    }
    assert expected_sources == {
        **{f"FP8_CB_K{rung}": "learned" for rung in range(28, 47)},
        "FP8_CB_K47": "lattice",
        "FP8_CB_K48": "lattice",
    }
    assert bundle.codebook_source_by_format == expected_sources
    assert set(bundle.codebook_refs_by_cell[QNAME]) == set(FORMATS)

    for format_name, expected_source in expected_sources.items():
        cell = bundle.cell(QNAME, format_name)
        assert cell["source"] == expected_source
        refs = bundle.codebook_refs_by_cell[QNAME][format_name]
        if expected_source == "learned":
            assert all(QNAME in ref for ref in refs)
        else:
            assert all("cb_codebook.lattice." in ref for ref in refs)


def test_k47_bundle_render_is_bitwise_the_no_bundle_lattice_render(
    policy_bundle, monkeypatch
):
    bundle, weight, col_weights = policy_bundle
    context = _bundle_context(bundle)
    assert codebook_source_for_format("FP8_CB_K47", context) == "lattice"

    # Freeze-proof the authority: changing the process-global policy after
    # the bundle context exists must not change this artifact's source map.
    changed_policy = dict(learned_bundle.CBL_RUNG_POLICY[47])
    changed_policy["enabled"] = True
    monkeypatch.setitem(learned_bundle.CBL_RUNG_POLICY, 47, changed_policy)
    assert codebook_source_for_format("FP8_CB_K47", context) == "lattice"

    from_bundle = cb_quantize_dequantize_for_context(
        fr.get_format("FP8_CB_K47"),
        weight,
        context=context,
        qname=QNAME,
        col_weights=col_weights,
    )
    without_bundle = cb_quantize_dequantize_for_context(
        fr.get_format("FP8_CB_K47"),
        weight,
        context=CBSerializationContext.production(),
        qname=QNAME,
        col_weights=col_weights,
    )
    assert torch.equal(from_bundle, without_bundle)


def test_k43_bundle_render_resolves_the_exact_learned_values(policy_bundle):
    bundle, weight, col_weights = policy_bundle
    context = _bundle_context(bundle)
    exact_book = bundle.codebook_for(
        QNAME,
        "FP8_CB_K43",
        weight=weight,
        col_weights=col_weights,
    )
    automatic = cb_quantize_dequantize_for_context(
        fr.get_format("FP8_CB_K43"),
        weight,
        context=context,
        qname=QNAME,
        col_weights=col_weights,
    )
    explicit = cb_quantize_dequantize_for_context(
        fr.get_format("FP8_CB_K43"),
        weight,
        context=context,
        qname=QNAME,
        col_weights=col_weights,
        codebook=exact_book,
    )
    lattice = cb_quantize_dequantize_for_context(
        fr.get_format("FP8_CB_K43"),
        weight,
        context=CBSerializationContext.production(),
        qname=QNAME,
        col_weights=col_weights,
    )
    assert torch.equal(automatic, explicit)
    assert not torch.equal(automatic, lattice)


def test_cost_and_export_identity_refuse_an_induced_per_rung_mismatch(
    policy_bundle,
):
    bundle, _weight, _col_weights = policy_bundle
    context = _bundle_context(bundle)
    payload = {"provenance": cb_cost_provenance(FORMATS, context=context)}
    validate_cb_cost_provenance(
        payload, FORMATS, context=context, where="matching source map"
    )

    stamp = payload["provenance"]["cb_serialized_payload"]
    assert stamp["codebook_source_by_format"] == (
        bundle.codebook_source_by_format
    )
    for mutation in ("changed", "missing"):
        mismatched = copy.deepcopy(payload)
        source_map = mismatched["provenance"]["cb_serialized_payload"][
            "codebook_source_by_format"
        ]
        if mutation == "changed":
            source_map["FP8_CB_K43"] = "lattice"
        else:
            source_map.pop("FP8_CB_K43")
        with pytest.raises(
            ValueError,
            match=r"per-rung codebook source map differs.*(changed|missing)",
        ):
            validate_cb_cost_provenance(
                mismatched,
                FORMATS,
                context=context,
                where=f"induced {mutation} source mismatch",
            )

    # The compact payload and actual quant_config provenance both carry the
    # complete frozen bundle map, including both sides selected here.
    breakdown = cb_assignment_payload_breakdown(
        {
            QNAME: "FP8_CB_K43",
            QNAME_K47: "FP8_CB_K47",
        },
        {
            QNAME: (2, 256),
            QNAME_K47: (2, 256),
        },
        context=context,
    )
    serialized_summary = cb_payload_summary(breakdown)
    expected_export_sources = bundle.codebook_source_by_format
    assert expected_export_sources["FP8_CB_K43"] == "learned"
    assert expected_export_sources["FP8_CB_K47"] == "lattice"
    assert serialized_summary["context"][
        "codebook_source_by_format"
    ] == expected_export_sources
    quant_config = build_quant_config(
        assignment={},
        cb_targets={},
        source_targets=(),
        stock_targets={},
        by_group={},
        codebooks={},
        col_weights={},
        codebook_tensors_by_name={},
        ignore=(),
        codebook_file=None,
        scale_coding=context.scale_coding,
        codebook_source=context.codebook_source,
        serialized_payload_summary=serialized_summary,
        serialization_context=context,
        cb_render_identity={"schema": "test.per_rung_render.v1"},
        git_commit="test",
    )
    assert quant_config["provenance"]["serialized_payload"]["context"][
        "codebook_source_by_format"
    ] == expected_export_sources


def test_codebook_for_remains_strict_on_a_lattice_cell(policy_bundle):
    bundle, _weight, _col_weights = policy_bundle
    with pytest.raises(
        ValueError,
        match=r"FP8_CB_K47 is lattice .*not learned",
    ):
        bundle.codebook_for(QNAME, "FP8_CB_K47")


@pytest.mark.parametrize("source_scope", ("fp8", "all"))
def test_policy_scope_with_only_lattice_rungs_needs_no_learned_digests(
    source_scope,
):
    context = CBSerializationContext.production(
        codebook_source_scope=source_scope,
        codebook_source_by_format={
            "FP8_CB_K47": "lattice",
            "FP8_CB_K48": "lattice",
        },
    )
    stamp = cb_serialization_context_stamp(context, formats=FORMATS[-2:])

    assert stamp["codebook_source"] == "lattice"
    assert stamp["codebook_source_scope"] == source_scope
    assert "codebook_content_sha256" not in stamp
    validate_cb_serialization_context_stamp(
        stamp,
        context,
        where="all-lattice policy bundle",
        formats=FORMATS[-2:],
    )

    contradictory = copy.deepcopy(stamp)
    contradictory["codebook_source"] = "learned"
    with pytest.raises(ValueError, match="artifact-wide ANY"):
        validate_cb_serialization_context_stamp(
            contradictory,
            context,
            where="contradictory all-lattice scalar",
            formats=FORMATS[-2:],
        )

    # The policy-only compatibility path must obey the same scalar/map ANY
    # rule even though it has no immutable bundle map to freeze.
    legacy_context = CBSerializationContext.production(
        codebook_source_scope=source_scope,
    )
    legacy_stamp = cb_serialization_context_stamp(
        legacy_context,
        formats=("FP8_CB_K47",),
    )
    assert legacy_stamp["codebook_source"] == "lattice"
    assert legacy_stamp["codebook_source_by_format"] == {
        "FP8_CB_K47": "lattice"
    }
    validate_cb_serialization_context_stamp(
        legacy_stamp,
        legacy_context,
        where="legacy policy-only lattice rung",
        formats=("FP8_CB_K47",),
    )

    with pytest.raises(ValueError, match="cannot carry learned bundle cells"):
        CBSerializationContext.production(
            codebook_source_scope="none",
            codebook_source_by_format={"FP8_CB_K43": "learned"},
        )
        with pytest.raises(ValueError, match="learned NVFP4 bundle cells"):
            CBSerializationContext.production(
                codebook_source_scope="fp8",
                codebook_source_by_format={"NVFP4_CB_K24": "learned"},
            )
