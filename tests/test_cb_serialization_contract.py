"""Cross-stage tests for the authoritative CB serialization contract."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import pickle
import struct

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant import nvfp4_cb_formats as cb
from prismaquant.allocator import (
    _serialized_format_rates,
    _sort_specs_by_serialized_rate,
)
from prismaquant.kl_measurement import assignment_bit_total
from prismaquant.measure_quant_cost import _cb_cost_quantize_dequantize
from prismaquant.incremental_measure_quant_cost import merge_cost_pickles
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    _safetensors_data_spans,
    _safetensors_tensor_payload_sha256,
    cb_assignment_payload_breakdown,
    cb_assignment_serialization_stamps,
    cb_tensor_serialization_stamp,
    codebook_subtable_shapes,
    cb_export_artifact_inventory,
    cb_payload_summary,
    cb_serialization_context_from_env,
    cb_serialization_context_from_stamp,
    cb_serialization_context_stamp,
    enforce_whole_artifact_budget,
    finalize_cb_export_artifact_inventory,
    load_cb_codebook_digest_manifest,
    lattice_codebook_content_sha256,
    whole_artifact_budget_stamp,
    validate_cb_cost_provenance,
)
from prismaquant.validate_assignments_kl import _assignment_bpp_details


def test_fp8_only_activation_scope_uses_no_activation_schema_and_bytes():
    context = cb_serialization_context_from_env({
        "CB_SCALE_CODING": "v1",
        "CB_CODEBOOK_SOURCE": "lattice",
        "CB_CODEBOOK_SOURCE_SCOPE": "none",
        "CB_SCALE_SWEEP": "1",
        "CB_SCALE_SWEEP_SCOPE": "fp8",
        "CB_ACTIVATION_SCOPE": "none",
        "PRISMAQUANT_CB_LDLQ": "0",
        "PRISMAQUANT_CB_MINCHAIN": "0",
        "PRISMAQUANT_CB_ENCODE_TIER": "balanced",
    })
    assert context.activation_contract is None
    assert context.activation_execution is None

    stamp = cb_serialization_context_stamp(
        context, formats=["FP8_CB_K4"]
    )
    assert stamp["schema"] == "prismaquant.cb_serialized_payload.v2"
    assert "activation_contract" not in stamp
    assert "activation_execution" not in stamp
    restored = cb_serialization_context_from_stamp(stamp, where="FP8 unit")
    assert restored.activation_contract is None
    assert restored.activation_execution is None
    assert cb_serialization_context_stamp(
        restored, formats=["FP8_CB_K4"]
    ) == stamp

    breakdown = cb_assignment_payload_breakdown(
        {"model.layers.0.mlp.down_proj": "FP8_CB_K4"},
        {"model.layers.0.mlp.down_proj": (2, 256)},
        context=restored,
    )
    summary = cb_payload_summary(breakdown)
    assert summary["schema"] == "prismaquant.cb_serialized_payload.v2"
    assert "activation_contract" not in summary["context"]
    assert "activation_execution" not in summary["context"]
    assert summary["fp4_scale_bytes"] == 0
    assert summary["input_global_scale_bytes"] == 0


def test_cb_activation_scope_default_is_historical_and_unknown_is_refused():
    historical = cb_serialization_context_from_env({})
    assert historical.activation_contract is not None
    assert historical.activation_execution is not None
    assert cb_serialization_context_stamp(historical)["schema"] == (
        "prismaquant.cb_serialized_payload.v3"
    )

    with pytest.raises(ValueError, match="CB_ACTIVATION_SCOPE"):
        cb_serialization_context_from_env({"CB_ACTIVATION_SCOPE": "fp8"})


@pytest.mark.parametrize("mode,k", [("product", 12)])
@pytest.mark.parametrize("shape", [(2, 256), (2, 2, 256)])
@pytest.mark.parametrize("scale_coding", ["v1", "two_tier"])
def test_cost_qdq_matches_serialized_pack_unpack(
    monkeypatch, mode, k, shape, scale_coding,
):
    """Cost reconstruction is exactly what the selected writer serializes."""
    monkeypatch.setenv("CB_SCALE_CODING", scale_coding)
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    monkeypatch.setenv("CB_SCALE_SWEEP", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ", "0")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "fast")
    torch.manual_seed(1000 + k)
    weight = torch.randn(*shape) * 0.2
    col_weights = torch.rand(*shape[:-2], 1, shape[-1]) + 0.05
    if len(shape) == 2:
        col_weights = col_weights.reshape(-1)
    spec = fr.get_format(f"NVFP4_CB_K{k}")

    measured = _cb_cost_quantize_dequantize(
        spec,
        weight.clone(),
        col_weights=col_weights,
    )
    packed, fields = cb.nvfp4_cb_pack(
        weight.clone(),
        k,
        grid="fp4",
        mode=mode,
        col_weights=col_weights,
        scale_coding=scale_coding,
        encode_tier="fast",
    )
    unpacked = cb.nvfp4_cb_unpack(
        packed,
        k,
        "fp4",
        mode,
        tuple(weight.shape),
        codebook=fields["codebook"],
        scale_coding=scale_coding,
    )
    serialized = cb.nvfp4_cb_reconstruct(
        unpacked,
        k,
        grid="fp4",
        mode=mode,
    ).to(weight.dtype)
    assert torch.equal(measured, serialized)


@pytest.mark.parametrize("shape", [(2, 256), (2, 2, 256)])
def test_fp8_cost_qdq_matches_serialized_pack_unpack(monkeypatch, shape):
    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    monkeypatch.setenv("CB_SCALE_SWEEP", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ", "0")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "fast")
    torch.manual_seed(1044)
    weight = torch.randn(*shape) * 0.2
    col_weights = torch.rand(*shape[:-2], 1, shape[-1]) + 0.05
    if len(shape) == 2:
        col_weights = col_weights.reshape(-1)
    spec = fr.get_format("FP8_CB_K36")

    measured = _cb_cost_quantize_dequantize(
        spec, weight.clone(), col_weights=col_weights
    )
    packed, fields = cb.nvfp4_cb_pack(
        weight.clone(),
        36,
        grid="fp8",
        mode="product",
        col_weights=col_weights,
        encode_tier="fast",
    )
    unpacked = cb.nvfp4_cb_unpack(
        packed,
        36,
        "fp8",
        "product",
        tuple(weight.shape),
        codebook=fields["codebook"],
        scales=fields["scales"],
    )
    serialized = cb.nvfp4_cb_reconstruct(
        unpacked, 36, grid="fp8", mode="product"
    ).to(weight.dtype)
    assert torch.equal(measured, serialized)


def test_cost_cache_identity_missing_or_mismatched_fails_closed():
    digests = {"materialized": "a" * 64}
    production = CBSerializationContext.production(
        codebook_source="learned",
        codebook_content_digests=digests,
    )
    formats = ["NVFP4_CB_K16", "BF16"]
    with pytest.raises(ValueError, match="no serialized-payload identity"):
        validate_cb_cost_provenance(
            {"provenance": {}},
            formats,
            context=production,
            where="unit cost",
        )
    stale = {
        "provenance": {
            "cb_serialized_payload": cb_serialization_context_stamp(
                CBSerializationContext.legacy_v1(
                    codebook_source="learned",
                    codebook_content_digests=digests,
                )
            )
        }
    }
    with pytest.raises(ValueError, match="differs from allocator recipe"):
        validate_cb_cost_provenance(
            stale,
            formats,
            context=production,
            where="unit cost",
        )


def test_lattice_identity_persists_canonical_content_and_roundtrips_refs():
    fmt = "NVFP4_CB_K16"
    qname = "layer.q_proj"
    refs = (
        f"cb_codebook.lattice.{fmt}.sub0",
        f"cb_codebook.lattice.{fmt}.sub1",
    )
    context = CBSerializationContext.production(
        codebook_refs={qname: refs},
    )
    stamp = cb_serialization_context_stamp(context, formats=[fmt])
    assert stamp["lattice_codebook_sha256_by_format"] == {
        fmt: list(lattice_codebook_content_sha256(fmt)),
    }
    assert stamp["codebook_refs"][qname] == list(refs)
    restored = cb_serialization_context_from_stamp(stamp, where="unit")
    assert restored.codebook_refs == {qname: refs}

    payload = cb_assignment_payload_breakdown(
        {qname: fmt},
        {qname: (2, 256)},
        context=restored,
    )
    assert payload["sidecars"][0]["content_sha256"] == list(
        lattice_codebook_content_sha256(fmt)
    )


def test_lattice_identity_rejects_noncanonical_materialized_bytes():
    fmt = "NVFP4_CB_K16"
    qname = "layer.q_proj"
    refs = (
        f"cb_codebook.lattice.{fmt}.sub0",
        f"cb_codebook.lattice.{fmt}.sub1",
    )
    context = CBSerializationContext.production(
        codebook_refs={qname: refs},
        codebook_content_digests={ref: "f" * 64 for ref in refs},
    )
    with pytest.raises(ValueError, match="canonical lattice identity"):
        cb_assignment_payload_breakdown(
            {qname: fmt},
            {qname: (2, 256)},
            context=context,
        )


def test_learned_context_validation_allows_unused_menu_digest_superset():
    selected = CBSerializationContext.production(
        codebook_source="learned",
        codebook_content_digests={"selected": "a" * 64},
    )
    menu = CBSerializationContext.production(
        codebook_source="learned",
        codebook_content_digests={
            "selected": "a" * 64,
            "unused_candidate": "b" * 64,
        },
    )
    stamp = cb_serialization_context_stamp(menu)
    from prismaquant.nvfp4_cb_footprint import (
        validate_cb_serialization_context_stamp,
    )

    validate_cb_serialization_context_stamp(stamp, selected, where="export")
    wrong = CBSerializationContext.production(
        codebook_source="learned",
        codebook_content_digests={"selected": "c" * 64},
    )
    with pytest.raises(ValueError, match="mismatched"):
        validate_cb_serialization_context_stamp(stamp, wrong, where="export")


def test_incremental_merge_rejects_a_stale_cb_shard(tmp_path, monkeypatch):
    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")
    monkeypatch.setenv("CB_SCALE_SWEEP", "1")
    monkeypatch.setenv("PRISMAQUANT_CB_LDLQ", "0")
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_TIER", "balanced")
    fresh = tmp_path / "fresh.pkl"
    stale = tmp_path / "stale.pkl"
    output = tmp_path / "merged.pkl"
    common = {"formats": ["NVFP4_CB_K16"], "meta": {}}
    col_weights = {
        "layer.0": torch.ones(256),
        "layer.1": torch.ones(256),
    }
    col_path = tmp_path / "col.pkl"
    col_path.write_bytes(pickle.dumps(col_weights))
    monkeypatch.setenv("PRISMAQUANT_CB_COL_WEIGHTS", str(col_path))
    import prismaquant.measure_quant_cost as measure_cost
    monkeypatch.setattr(measure_cost, "_CB_CW_CACHE", None)
    from prismaquant.production_weight_cache import (
        bind_cb_render_identity_source_weights,
        build_production_cache_cb_render_identity,
    )

    def identity(name, context):
        value = build_production_cache_cb_render_identity(
            {name: common["formats"]},
            cb_serialization_context=context,
            col_weights=col_weights,
            render_levers={"weighted_vq": True},
            render_mechanism_plan=[],
        )
        return bind_cb_render_identity_source_weights(
            value,
            {name: torch.ones(2, 256)},
        )

    fresh_identity = identity(
        "layer.0", CBSerializationContext.production()
    )
    stale_identity = identity(
        "layer.1", CBSerializationContext.legacy_v1()
    )
    fresh.write_bytes(pickle.dumps({
        **common,
        "costs": {"layer.0": {"NVFP4_CB_K16": {"output_mse": 1.0}}},
        "provenance": {
            "cb_serialized_payload": fresh_identity[
                "cb_serialized_payload"
            ],
            "cb_render_identity": fresh_identity,
        },
    }))
    stale.write_bytes(pickle.dumps({
        **common,
        "costs": {"layer.1": {"NVFP4_CB_K16": {"output_mse": 1.0}}},
        "provenance": {
            "cb_serialized_payload": stale_identity[
                "cb_serialized_payload"
            ],
            "cb_render_identity": stale_identity,
        },
    }))
    with pytest.raises(ValueError, match="differs from allocator recipe"):
        merge_cost_pickles([fresh, stale], output)
    assert not output.exists()


def _stats(shape):
    return {
        "n_params": math.prod(shape),
        "out_features": shape[-2],
        "in_features": shape[-1],
        **({"num_experts": shape[0]} if len(shape) == 3 else {}),
    }


def _learned_digests(fmt, *roles):
    count = len(codebook_subtable_shapes(fmt))
    return {
        f"cb_codebook.{role}.{fmt}.sub{index}": f"{index + 1:064x}"
        for role in roles
        for index in range(count)
    }


@pytest.mark.parametrize("scale_coding", ["v1", "two_tier"])
def test_assignment_bits_prices_fp4_layout_and_shared_role_once(scale_coding):
    names = ("model.layers.0.q_proj", "model.layers.1.q_proj")
    shape = (4, 512)
    fmt = "NVFP4_CB_K16"
    assignment = {name: fmt for name in names}
    shapes = {name: shape for name in names}
    stats = {name: _stats(shape) for name in names}
    context = CBSerializationContext(
        scale_coding=scale_coding,
        codebook_source="learned",
        codebook_content_digests=_learned_digests(fmt, "q_proj"),
    )
    stamps = cb_assignment_serialization_stamps(
        assignment, shapes, context=context
    )
    breakdown = cb_assignment_payload_breakdown(
        assignment, shapes, context=context
    )
    assert len(breakdown["sidecars"]) == 1
    bits = assignment_bit_total(
        stats,
        assignment,
        {fmt: fr.get_format(fmt)},
        cb_serialization_context=context,
        cb_serialization_stamps=stamps,
    )
    assert bits == 8 * breakdown["total_bytes"]
    bytes_per_superblock = 4 * 16 + (16 if scale_coding == "v1" else 9)
    assert breakdown["tensor_payload_bytes"] == 2 * 4 * 2 * bytes_per_superblock


def test_explicit_codebook_refs_reject_partial_subtable_sharing():
    fmt = "NVFP4_CB_K16"
    assignment = {"layer.a": fmt, "layer.b": fmt}
    refs = {
        "layer.a": ("shared", "a-only"),
        "layer.b": ("shared", "b-only"),
    }
    context = CBSerializationContext.production(
        codebook_source="learned",
        codebook_refs=refs,
        codebook_content_digests={
            "shared": "1" * 64,
            "a-only": "2" * 64,
            "b-only": "3" * 64,
        },
    )
    with pytest.raises(ValueError, match="partially shared or reused"):
        cb_assignment_payload_breakdown(
            assignment,
            {name: (2, 256) for name in assignment},
            context=context,
        )


def test_explicit_codebook_refs_reject_duplicate_ref_within_table_set():
    fmt = "NVFP4_CB_K16"
    context = CBSerializationContext.production(
        codebook_source="learned",
        codebook_refs={"layer.a": ("duplicate", "duplicate")},
        codebook_content_digests={"duplicate": "1" * 64},
    )
    with pytest.raises(ValueError, match="repeats a physical codebook ref"):
        cb_assignment_payload_breakdown(
            {"layer.a": fmt},
            {"layer.a": (2, 256)},
            context=context,
        )


def test_explicit_codebook_refs_reject_conflicting_physical_identity():
    refs = ("same-sub0", "same-sub1")
    assignment = {
        "layer.a": "NVFP4_CB_K16",
        "layer.b": "NVFP4_CB_K17",
    }
    context = CBSerializationContext.production(
        codebook_source="learned",
        codebook_refs={name: refs for name in assignment},
        codebook_content_digests={ref: "4" * 64 for ref in refs},
    )
    with pytest.raises(ValueError, match="partially shared or reused"):
        cb_assignment_payload_breakdown(
            assignment,
            {name: (2, 256) for name in assignment},
            context=context,
        )


@pytest.mark.parametrize("in_features", [512, 5120])
def test_assignment_bits_prices_fp8_row_scales_at_real_shape(in_features):
    shape = (3, in_features)
    fmt = "FP8_CB_K36"
    assignment = {"model.layers.0.q_proj": fmt}
    shapes = {name: shape for name in assignment}
    stats = {name: _stats(shape) for name in assignment}
    context = CBSerializationContext.production()
    stamps = cb_assignment_serialization_stamps(
        assignment, shapes, context=context
    )
    breakdown = cb_assignment_payload_breakdown(
        assignment, shapes, context=context
    )
    assert breakdown["fp8_row_scale_bytes"] == 4 * shape[0]
    assert assignment_bit_total(
        stats,
        assignment,
        {fmt: fr.get_format(fmt)},
        cb_serialization_context=context,
        cb_serialization_stamps=stamps,
    ) == 8 * breakdown["total_bytes"]


def test_assignment_bits_includes_native_nvfp4_global_scale_tensors():
    name = "model.layers.0.self_attn.o_proj"
    shape = (4, 256)
    spec = fr.get_format("NVFP4")
    assert assignment_bit_total(
        {name: _stats(shape)},
        {name: "NVFP4"},
        {"NVFP4": spec},
    ) == 8 * (spec.memory_bytes_for_shape(shape) + 8)


def test_assignment_bpp_details_uses_exact_cb_assignment_payload():
    fmt = "FP8_CB_K36"
    assignment = {
        "model.layers.0.q_proj": fmt,
        "model.layers.1.q_proj": fmt,
    }
    shapes = {name: (3, 512) for name in assignment}
    stats = {name: _stats(shape) for name, shape in shapes.items()}
    context = CBSerializationContext.production(
        codebook_source="learned",
        codebook_content_digests=_learned_digests(fmt, "q_proj"),
    )
    stamps = cb_assignment_serialization_stamps(
        assignment, shapes, context=context
    )
    payload = cb_assignment_payload_breakdown(
        assignment, shapes, context=context
    )
    details = _assignment_bpp_details(
        stats,
        assignment,
        {fmt: fr.get_format(fmt)},
        cb_serialization_context=context,
        cb_serialization_stamps=stamps,
        where="direct bpp contract test",
    )
    assert details["bpp"] == pytest.approx(
        8 * payload["total_bytes"]
        / sum(item["n_params"] for item in stats.values())
    )


def test_assignment_bits_requires_every_matching_per_layer_stamp():
    fmt = "NVFP4_CB_K16"
    shape = (4, 256)
    assignment = {"model.layers.0.q_proj": fmt}
    stats = {name: _stats(shape) for name in assignment}
    context = CBSerializationContext.production()
    specs = {fmt: fr.get_format(fmt)}
    with pytest.raises(ValueError, match="missing per-layer"):
        assignment_bit_total(
            stats,
            assignment,
            specs,
            cb_serialization_context=context,
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        assignment_bit_total(
            stats,
            assignment,
            specs,
            cb_serialization_context=context,
            cb_serialization_stamps={next(iter(assignment)): "stale"},
        )


def test_assignment_stamp_validation_rejects_every_extra_identity():
    from prismaquant.nvfp4_cb_footprint import (
        validate_cb_assignment_serialization_stamps,
    )

    fmt = "NVFP4_CB_K16"
    name = "model.layers.0.q_proj"
    shape = (4, 256)
    assignment = {name: fmt}
    shapes = {name: shape}
    context = CBSerializationContext.production()
    stamps = cb_assignment_serialization_stamps(
        assignment, shapes, context=context
    )
    stamps["model.layers.9.stale_proj"] = stamps[name]
    with pytest.raises(ValueError, match="extra=.*stale_proj"):
        validate_cb_assignment_serialization_stamps(
            assignment,
            shapes,
            context=context,
            stamps=stamps,
            where="exact stamp regression",
        )


def test_tensor_stamp_binds_shape_and_every_byte_component():
    context = CBSerializationContext.production()
    a = cb_tensor_serialization_stamp(
        "FP8_CB_K36", (3, 512), qname="layer.q_proj", context=context
    )
    b = cb_tensor_serialization_stamp(
        "FP8_CB_K36", (6, 256), qname="layer.q_proj", context=context
    )
    assert a != b, "equal n_params must not hide row/block/scale shape drift"
    parsed = json.loads(a)
    assert parsed["shape"] == [3, 512]
    assert parsed["output_rows"] == 3
    assert parsed["superblocks_per_row"] == 2
    assert parsed["fp8_row_scale_bytes"] == 12
    assert parsed["tensor_payload_bytes"] == (
        parsed["packed_weight_bytes"] + parsed["fp8_row_scale_bytes"]
    )


def test_learned_identity_requires_materialized_content_digests():
    context = CBSerializationContext.production(codebook_source="learned")
    with pytest.raises(ValueError, match="materialized SHA-256"):
        cb_assignment_payload_breakdown(
            {"layer.q_proj": "NVFP4_CB_K16"},
            {"layer.q_proj": (2, 256)},
            context=context,
        )
    with pytest.raises(ValueError, match="codebook_content_digests"):
        cb_serialization_context_stamp(context)


def test_codebook_digest_manifest_accepts_inline_json_and_rejects_duplicates():
    assert load_cb_codebook_digest_manifest(
        '{"sidecar":"' + ("a" * 64) + '"}', where="unit"
    ) == {"sidecar": "a" * 64}
    with pytest.raises(AssertionError, match="duplicate JSON object key"):
        load_cb_codebook_digest_manifest(
            '{"sidecar":"' + ("a" * 64) + '","sidecar":"'
            + ("b" * 64) + '"}',
            where="unit",
        )


def test_serialized_format_order_is_input_order_independent():
    stats = {
        f"model.layers.{index}.q_proj": _stats((512, 512))
        for index in range(64)
    }
    names = ["NVFP4", "NVFP4_CB_K24", "FP8_CB_K36", "BF16"]
    context = CBSerializationContext.production()
    expected = None
    for permutation in itertools.permutations(names):
        ordered, rates = _sort_specs_by_serialized_rate(
            [fr.get_format(name) for name in permutation],
            stats,
            context,
        )
        observed = [spec.name for spec in ordered]
        expected = observed if expected is None else expected
        assert observed == expected
        # Production K24 uses 4k+9, not FormatSpec's stale 4k+16 rate.
        assert rates["NVFP4_CB_K24"] < fr.get_format(
            "NVFP4_CB_K24"
        ).effective_bits


def _write_safetensors(path, entries, *, data_bytes=None):
    header = {
        name: {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": list(offsets),
        }
        for name, (dtype, shape, offsets) in entries.items()
    }
    raw = json.dumps(header).encode()
    extent = max((offsets[1] for _dtype, _shape, offsets in entries.values()),
                 default=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<Q", len(raw)) + raw
        + (b"\0" * extent if data_bytes is None else data_bytes)
    )


def _write_canonical_lattice_sidecar(path, payload):
    from safetensors.torch import save_file

    tensors = {}
    for sidecar in payload["sidecars"]:
        fmt = sidecar["format"]
        family, rung = fmt.split("_CB_", 1)
        grid = "fp4" if family == "NVFP4" else "fp8"
        mode = "product"
        k = int(rung[1:])
        codebook = cb._resolve_codebook(
            k, grid, mode, None, torch.device("cpu")
        )
        tables = codebook if isinstance(codebook, tuple) else (codebook,)
        assert len(tables) == len(sidecar["codebook_ref"])
        tensors.update({
            ref: table.to(torch.float16).cpu().contiguous()
            for ref, table in zip(
                sidecar["codebook_ref"], tables, strict=True
            )
        })
    save_file(tensors, str(path))


def _write_raw_safetensors(path, raw_header: str, data: bytes = b""):
    raw = raw_header.encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


@pytest.mark.parametrize(
    "dtype,shape,span",
    [
        ("F8_E8M0", (3,), 3),
        ("F4", (3,), 2),
        ("F4_E2M1", (4,), 2),
    ],
)
def test_safetensors_span_parser_understands_packed_float_widths(
    tmp_path, dtype, shape, span,
):
    path = tmp_path / "packed.safetensors"
    _write_safetensors(path, {"a": (dtype, shape, (0, span))})
    assert _safetensors_data_spans(path) == {"a": span}


@pytest.mark.parametrize(
    "raw_header,match",
    [
        (
            '{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},'
            '"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}',
            "duplicate JSON object key",
        ),
        ('{"__metadata__":{"bad":NaN}}', "non-finite JSON constant"),
    ],
)
def test_safetensors_span_parser_rejects_ambiguous_json(
    tmp_path, raw_header, match,
):
    path = tmp_path / "bad-json.safetensors"
    _write_raw_safetensors(path, raw_header, b"\0")
    with pytest.raises(AssertionError, match=match):
        _safetensors_data_spans(path)


@pytest.mark.parametrize(
    "entries,match",
    [
        (
            {
                "a": ("U8", (2,), (0, 2)),
                "b": ("U8", (2,), (3, 5)),
            },
            "leaves a gap",
        ),
        ({"a": ("F16", (2,), (0, 2))}, "requires 4B"),
    ],
)
def test_safetensors_span_parser_rejects_malformed_layout(tmp_path, entries, match):
    path = tmp_path / "bad.safetensors"
    _write_safetensors(path, entries)
    with pytest.raises(AssertionError, match=match):
        _safetensors_data_spans(path)


def test_recursive_inventory_reaches_a_stable_quant_config_fixed_point(tmp_path):
    context = CBSerializationContext.production()
    assignment = {"layer.q_proj": "NVFP4_CB_K16"}
    shapes = {"layer.q_proj": (1, 256)}
    payload = cb_assignment_payload_breakdown(
        assignment, shapes, context=context
    )
    tensor_bytes = int(payload["tensor_payload_bytes"])
    sidecar_bytes = int(payload["codebook_sidecar_bytes"])
    _write_safetensors(
        tmp_path / "nested" / "model.safetensors",
        {"layer.q_proj.cb_qweight": ("U8", (tensor_bytes,), (0, tensor_bytes))},
    )
    _write_canonical_lattice_sidecar(
        tmp_path / "cb_codebooks.pqcb", payload
    )
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "extra.txt").write_text("abc")
    config = {"provenance": {}}
    inventory = finalize_cb_export_artifact_inventory(
        tmp_path,
        config,
        serialized_payload=payload,
        cb_tensor_names=["layer.q_proj.cb_qweight"],
        codebook_file="cb_codebooks.pqcb",
    )
    assert "nested/model.safetensors" in inventory["file_bytes"]
    assert "tokenizer/extra.txt" in inventory["file_bytes"]
    on_disk = json.loads((tmp_path / "quant_config.json").read_text())
    assert on_disk["provenance"]["artifact_inventory"] == inventory
    assert cb_export_artifact_inventory(
        tmp_path,
        serialized_payload=payload,
        cb_tensor_names=["layer.q_proj.cb_qweight"],
        codebook_file="cb_codebooks.pqcb",
    ) == inventory


def _materialize_minimal_cb_export(tmp_path, *, context=None):
    context = context or CBSerializationContext.production()
    assignment = {"layer.q_proj": "NVFP4_CB_K16"}
    payload = cb_assignment_payload_breakdown(
        assignment, {"layer.q_proj": (1, 256)}, context=context
    )
    tensor_bytes = int(payload["tensor_payload_bytes"])
    _write_safetensors(
        tmp_path / "model.safetensors",
        {"layer.q_proj.cb_qweight": (
            "U8", (tensor_bytes,), (0, tensor_bytes)
        )},
    )
    sidecar_entries = {}
    offset = 0
    for sidecar in payload["sidecars"]:
        for ref, shape in zip(
            sidecar["codebook_ref"], sidecar["subtable_shapes"]
        ):
            nbytes = math.prod(shape) * 2
            sidecar_entries[ref] = ("F16", tuple(shape), (offset, offset + nbytes))
            offset += nbytes
    if context.codebook_source == "lattice":
        _write_canonical_lattice_sidecar(
            tmp_path / "cb_codebooks.pqcb", payload
        )
    else:
        _write_safetensors(tmp_path / "cb_codebooks.pqcb", sidecar_entries)
    return payload


def test_inventory_verifies_learned_codebook_content_digest(tmp_path):
    fmt = "NVFP4_CB_K16"
    refs = [f"cb_codebook.q_proj.{fmt}.sub{index}" for index in range(2)]
    digest = hashlib.sha256(b"\0" * (256 * 4 * 2)).hexdigest()
    context = CBSerializationContext.production(
        codebook_source="learned",
        codebook_content_digests={ref: digest for ref in refs},
    )
    payload = _materialize_minimal_cb_export(tmp_path, context=context)
    inventory = cb_export_artifact_inventory(
        tmp_path,
        serialized_payload=payload,
        cb_tensor_names=["layer.q_proj.cb_qweight"],
        codebook_file="cb_codebooks.pqcb",
        expected_model_files=["model.safetensors"],
    )
    assert inventory["cb_codebook_content_sha256"] == {
        ref: digest for ref in refs
    }

    codebook = tmp_path / "cb_codebooks.pqcb"
    raw = bytearray(codebook.read_bytes())
    raw[-1] ^= 1
    codebook.write_bytes(raw)
    with pytest.raises(AssertionError, match="differ from their content identity"):
        cb_export_artifact_inventory(
            tmp_path,
            serialized_payload=payload,
            cb_tensor_names=["layer.q_proj.cb_qweight"],
            codebook_file="cb_codebooks.pqcb",
            expected_model_files=["model.safetensors"],
        )


def test_nonzero_fp16_digest_matches_exact_safetensors_payload(tmp_path):
    from safetensors.torch import save_file

    tensor = (torch.arange(48, dtype=torch.float16) - 17).reshape(6, 8)
    path = tmp_path / "learned.pqcb"
    save_file({"cb_codebook.role.fmt": tensor}, str(path))
    expected = hashlib.sha256(
        tensor.cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    assert _safetensors_tensor_payload_sha256(
        path, ["cb_codebook.role.fmt"]
    ) == {"cb_codebook.role.fmt": expected}


def test_inventory_rejects_stale_model_shards(tmp_path):
    payload = _materialize_minimal_cb_export(tmp_path)
    _write_safetensors(tmp_path / "stale-00002.safetensors", {})
    with pytest.raises(AssertionError, match="fresh export plan"):
        cb_export_artifact_inventory(
            tmp_path,
            serialized_payload=payload,
            cb_tensor_names=["layer.q_proj.cb_qweight"],
            codebook_file="cb_codebooks.pqcb",
            expected_model_files=["model.safetensors"],
        )


def test_inventory_rejects_stale_model_index_beside_single_file(tmp_path):
    payload = _materialize_minimal_cb_export(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 1},
        "weight_map": {"old.weight": "missing-old-shard.safetensors"},
    }))
    with pytest.raises(AssertionError, match="stale model.safetensors.index"):
        cb_export_artifact_inventory(
            tmp_path,
            serialized_payload=payload,
            cb_tensor_names=["layer.q_proj.cb_qweight"],
            codebook_file="cb_codebooks.pqcb",
            expected_model_files=["model.safetensors"],
        )


def test_inventory_rejects_stale_codebook_sidecars(tmp_path):
    payload = _materialize_minimal_cb_export(tmp_path)
    _write_safetensors(tmp_path / "stale-codebooks.pqcb", {})
    with pytest.raises(AssertionError, match="stale CB codebook sidecar"):
        cb_export_artifact_inventory(
            tmp_path,
            serialized_payload=payload,
            cb_tensor_names=["layer.q_proj.cb_qweight"],
            codebook_file="cb_codebooks.pqcb",
            expected_model_files=["model.safetensors"],
        )


def test_inventory_rejects_unexpected_cb_suffix_tensors(tmp_path):
    payload = _materialize_minimal_cb_export(tmp_path)
    tensor_bytes = int(payload["tensor_payload_bytes"])
    _write_safetensors(
        tmp_path / "model.safetensors",
        {
            "layer.q_proj.cb_qweight": (
                "U8", (tensor_bytes,), (0, tensor_bytes)
            ),
            "layer.q_proj.weight_scale": (
                "F32", (1,), (tensor_bytes, tensor_bytes + 4)
            ),
        },
    )
    with pytest.raises(AssertionError, match="unexpected/stale CB tensors"):
        cb_export_artifact_inventory(
            tmp_path,
            serialized_payload=payload,
            cb_tensor_names=["layer.q_proj.cb_qweight"],
            codebook_file="cb_codebooks.pqcb",
            expected_model_files=["model.safetensors"],
        )


def test_final_inventory_hard_fails_actual_recursive_size_over_budget(tmp_path):
    payload = _materialize_minimal_cb_export(tmp_path)
    (tmp_path / "tokenizer.json").write_text("{}")
    with pytest.raises(RuntimeError, match="exact recursive export size"):
        finalize_cb_export_artifact_inventory(
            tmp_path,
            {"provenance": {}},
            serialized_payload=payload,
            cb_tensor_names=["layer.q_proj.cb_qweight"],
            codebook_file="cb_codebooks.pqcb",
            expected_model_files=["model.safetensors"],
            whole_artifact_budget_bytes=1,
        )


@pytest.mark.parametrize("expected_headroom", [9, 99])
def test_final_inventory_budget_digit_boundaries_converge(
    tmp_path, expected_headroom
):
    probe = tmp_path / "probe"
    probe.mkdir()
    probe_payload = _materialize_minimal_cb_export(probe)
    probe_inventory = finalize_cb_export_artifact_inventory(
        probe,
        {"provenance": {}},
        serialized_payload=probe_payload,
        cb_tensor_names=["layer.q_proj.cb_qweight"],
        codebook_file="cb_codebooks.pqcb",
        expected_model_files=["model.safetensors"],
        whole_artifact_budget_bytes=9000,
    )
    target_budget = (
        int(probe_inventory["export_directory_bytes"]) + expected_headroom
    )
    # Both budgets have the same decimal width, so the embedded budget field
    # occupies exactly the same number of bytes in the fresh target export.
    assert 1000 <= target_budget <= 9999

    target = tmp_path / "target"
    target.mkdir()
    target_payload = _materialize_minimal_cb_export(target)
    inventory = finalize_cb_export_artifact_inventory(
        target,
        {"provenance": {}},
        serialized_payload=target_payload,
        cb_tensor_names=["layer.q_proj.cb_qweight"],
        codebook_file="cb_codebooks.pqcb",
        expected_model_files=["model.safetensors"],
        whole_artifact_budget_bytes=target_budget,
    )
    assert target_budget - inventory["export_directory_bytes"] == expected_headroom
    assert "within_whole_artifact_budget" not in inventory
    assert "whole_artifact_budget_headroom_bytes" not in inventory
    on_disk = json.loads((target / "quant_config.json").read_text())
    assert on_disk["provenance"]["artifact_inventory"] == inventory


def test_generic_export_budget_gate_measures_files_recursively(tmp_path):
    artifact = tmp_path / "artifact"
    (artifact / "nested").mkdir(parents=True)
    (artifact / "a.bin").write_bytes(b"a" * 7)
    (artifact / "nested" / "b.bin").write_bytes(b"b" * 5)
    assignment = {"layer.q_proj": "BF16"}
    payload = {
        "whole_artifact_budget": whole_artifact_budget_stamp(
            budget_bytes=12,
            selection_tensor_payload_bytes=8,
            selection_non_tensor_reserve_bytes=4,
            selection_assignment=assignment,
        ),
    }
    attestation = enforce_whole_artifact_budget(
        artifact, payload, where="unit export", assignment=assignment
    )
    assert attestation["actual_bytes"] == 12
    assert attestation["within_budget"]

    payload["whole_artifact_budget"] = whole_artifact_budget_stamp(
        budget_bytes=11,
        selection_tensor_payload_bytes=8,
        selection_non_tensor_reserve_bytes=3,
        selection_assignment=assignment,
    )
    with pytest.raises(RuntimeError, match="exact completed artifact size"):
        enforce_whole_artifact_budget(
            artifact, payload, where="unit export", assignment=assignment
        )


def test_whole_artifact_budget_rejects_assignment_drift(tmp_path):
    assignment = {"layer.q_proj": "BF16"}
    payload = {
        "whole_artifact_budget": whole_artifact_budget_stamp(
            budget_bytes=1,
            selection_tensor_payload_bytes=1,
            selection_non_tensor_reserve_bytes=0,
            selection_assignment=assignment,
        ),
    }
    (tmp_path / "artifact.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="assignment being consumed"):
        enforce_whole_artifact_budget(
            tmp_path,
            payload,
            where="unit export",
            assignment={"layer.q_proj": "NVFP4"},
        )


def test_fp8_rate_helper_reflects_in_features_row_scale_amortization():
    spec = fr.get_format("FP8_CB_K36")
    context = CBSerializationContext.production()
    narrow = _serialized_format_rates(
        [spec], {"w": _stats((512, 512))}, context
    )[spec.name]
    wide = _serialized_format_rates(
        [spec], {"w": _stats((512, 5120))}, context
    )[spec.name]
    assert narrow > wide


# --- rate vs identity at the source-payload gate ---------------------------
#
# A learned book is banked only for the rungs a unit may actually use, so the
# legality probe evaluates cells whose book will never exist: a routed expert
# against K36 when the byte-exact source ceiling already stops it at K33. That
# verdict is a byte count and must not depend on which books happen to have
# been built -- otherwise the same menu answers differently before and after a
# bundle rebuild. Identity is still required of every candidate that survives.


def _routed_expert_learned_context(fmt, *roles):
    return CBSerializationContext(
        scale_coding="two_tier",
        codebook_source="learned",
        codebook_source_scope="fp8",
        codebook_content_digests=_learned_digests(fmt, *roles),
    )


def test_source_rate_gate_prices_an_unbanked_rung_as_exceeded():
    from prismaquant.allocator_candidates import (
        SOURCE_BPP_EXCEEDED_REASON,
        _source_bpp_applicability,
    )

    qname = "model.layers.0.mlp.experts.0.down_proj"
    shape = (2048, 4096)
    # Books exist for the on-law rungs only; K36 was deliberately never learned.
    context = _routed_expert_learned_context("FP8_CB_K32", "down_proj")

    banked = _source_bpp_applicability(
        shape,
        fr.get_format("FP8_CB_K32"),
        qname=qname,
        source_kind="mxfp4",
        cb_serialization_context=context,
    )
    assert banked.legal

    unbanked = _source_bpp_applicability(
        shape,
        fr.get_format("FP8_CB_K36"),
        qname=qname,
        source_kind="mxfp4",
        cb_serialization_context=context,
    )
    assert not unbanked.legal
    assert unbanked.reason == SOURCE_BPP_EXCEEDED_REASON


def test_source_rate_gate_still_requires_identity_for_a_legal_cell():
    from prismaquant.allocator_candidates import _source_bpp_applicability

    qname = "model.layers.0.self_attn.o_proj"
    shape = (8192, 4096)
    # A dense row can legally reach K36, so an unbanked book here is a real
    # defect and must fail at the gate rather than first at export.
    context = _routed_expert_learned_context("FP8_CB_K32", "o_proj")
    with pytest.raises(ValueError, match="missing materialized SHA-256"):
        _source_bpp_applicability(
            shape,
            fr.get_format("FP8_CB_K36"),
            qname=qname,
            source_kind="fp8_ue8m0",
            cb_serialization_context=context,
        )


def test_sizing_mode_matches_banked_bytes_and_marks_identity_unproven():
    from prismaquant.nvfp4_cb_footprint import (
        cb_tensor_payload_breakdown,
        codebook_sidecar_payload_bytes,
    )

    qname = "model.layers.0.mlp.experts.0.down_proj"
    shape = (2048, 4096)
    fmt = "FP8_CB_K32"
    context = _routed_expert_learned_context(fmt, "down_proj")

    strict = cb_tensor_payload_breakdown(
        fmt, shape, qname=qname, context=context
    )
    sized = cb_tensor_payload_breakdown(
        fmt, shape, qname=qname, context=context,
        require_materialized_codebook_identity=False,
    )
    # A banked cell is byte-identical in both modes, identity included.
    assert sized == strict

    unbanked = cb_tensor_payload_breakdown(
        "FP8_CB_K36", shape, qname=qname, context=context,
        require_materialized_codebook_identity=False,
    )
    assert unbanked["tensor_payload_bytes"] > 0
    assert unbanked["sidecar_identity"]["content_sha256"] is None
    assert unbanked["sidecar_identity"]["materialized_identity"] is False
    # The sidecar's size is a function of the rung, which is exactly why the
    # rate question is answerable without the book.
    assert unbanked["sidecar_payload_bytes"] == codebook_sidecar_payload_bytes(
        "FP8_CB_K36"
    )
    assert unbanked["sidecar_payload_bytes"] > strict["sidecar_payload_bytes"]


# ---------------------------------------------------------------------------
# Namespace exclusion is one statement made in two places: the allocator
# declines to CHARGE for a namespace (handing those bytes to the body) and the
# exporter declines to WRITE it. Making only one of them was silent in the
# direction that costs quality, so the stamp carries the price's exclusion set
# and the exporter is required to match it.
# ---------------------------------------------------------------------------


def _stamp(assignment, *, excluded=()):
    from prismaquant.nvfp4_cb_footprint import whole_artifact_budget_stamp

    return whole_artifact_budget_stamp(
        budget_bytes=1000,
        selection_tensor_payload_bytes=8,
        selection_non_tensor_reserve_bytes=4,
        selection_assignment=assignment,
        excluded_source_prefixes=excluded,
    )


def test_budget_stamp_without_exclusions_is_byte_identical():
    """A run that excludes nothing must write exactly the stamp it always did."""
    assignment = {"layer.q_proj": "BF16"}
    assert _stamp(assignment) == _stamp(assignment, excluded=())
    assert "excluded_source_prefixes" not in _stamp(assignment)
    # Blank/whitespace entries are not exclusions.
    assert "excluded_source_prefixes" not in _stamp(assignment, excluded=["", "  "])


def test_budget_stamp_records_and_dedupes_exclusions():
    from prismaquant.nvfp4_cb_footprint import budget_stamp_excluded_prefixes

    stamp = _stamp({"layer.q_proj": "BF16"}, excluded=["mtp.", " mtp.", "visual."])
    assert stamp["excluded_source_prefixes"] == ["mtp.", "visual."]
    assert budget_stamp_excluded_prefixes(stamp) == ("mtp.", "visual.")


def test_exclusions_must_match_the_price_that_bought_them():
    from prismaquant.nvfp4_cb_footprint import (
        assert_exclusions_match_budget_stamp,
    )

    assignment = {"layer.q_proj": "BF16"}
    priced = _stamp(assignment, excluded=["mtp."])

    # Agreement in both spellings of "nothing" and in the real case.
    assert_exclusions_match_budget_stamp(priced, ["mtp."], where="unit")
    assert_exclusions_match_budget_stamp(
        _stamp(assignment), [], where="unit")
    # No stamp is no claim: exclusion stands on its own without a budget.
    assert_exclusions_match_budget_stamp(None, ["mtp."], where="unit")

    # OVERSHOOT: priced without mtp, but the export writes it anyway.
    with pytest.raises(ValueError, match="overshoots"):
        assert_exclusions_match_budget_stamp(priced, [], where="unit")

    # UNDERSHOOT -- the direction nothing else catches: the price charged for
    # mtp, the export drops it, and the artifact silently ships under budget.
    with pytest.raises(ValueError, match="under budget"):
        assert_exclusions_match_budget_stamp(
            _stamp(assignment), ["mtp."], where="unit")


def test_a_malformed_exclusion_record_is_loud():
    from prismaquant.nvfp4_cb_footprint import (
        whole_artifact_budget_from_assignment_payload,
    )

    assignment = {"layer.q_proj": "BF16"}
    stamp = dict(_stamp(assignment, excluded=["mtp."]))
    stamp["excluded_source_prefixes"] = "mtp."      # a string, not a list
    with pytest.raises(ValueError, match="excluded_source_prefixes"):
        whole_artifact_budget_from_assignment_payload(
            {"whole_artifact_budget": stamp},
            where="unit", assignment=assignment)


def test_exporter_refuses_an_export_that_contradicts_its_price():
    """The guard is wired into the exporter's own validation, not just free-standing."""
    from prismaquant.export_nvfp4_cb_streaming import (
        _validate_namespace_exclusions,
    )

    assignment = {"layer.q_proj": "BF16"}
    priced = _stamp(assignment, excluded=["mtp."])

    assert _validate_namespace_exclusions(
        ["mtp."], assignment=assignment, profile=None,
        budget_stamp=priced) == ("mtp.",)

    with pytest.raises(ValueError, match="overshoots"):
        _validate_namespace_exclusions(
            [], assignment=assignment, profile=None, budget_stamp=priced)

    with pytest.raises(ValueError, match="under budget"):
        _validate_namespace_exclusions(
            ["mtp."], assignment=assignment, profile=None,
            budget_stamp=_stamp(assignment))

    # Unbudgeted exports keep working exactly as before.
    assert _validate_namespace_exclusions(
        ["mtp."], assignment=assignment, profile=None) == ("mtp.",)
