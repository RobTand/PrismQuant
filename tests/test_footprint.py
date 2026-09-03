"""Unit tests for prismaquant.footprint (exact artifact-GB accounting).

Validates the residual-floor identity
    artifact = (source_total - Σ_reencoded n_params·src_bpp)
               + Σ_reencoded memory_bytes_for_shape(shape, fmt)
against a hand-computed synthetic checkpoint, and the safetensors header reader
against a synthetic shard (header-only; no torch). The end-to-end 0.00% match vs
real 27B exports is covered by the verification pass, not here.
"""
from __future__ import annotations

import json
import struct

import pytest

import prismaquant.name_projection as npx
from prismaquant import footprint as fp
from prismaquant import format_registry as fr
from prismaquant.model_profiles.base import ModelProfile
from prismaquant.model_profiles.qwen3 import Qwen3Profile
from prismaquant.nvfp4_cb_footprint import CBSerializationContext
def _write_safetensors(path, tensors):
    """Write a minimal valid .safetensors file. tensors: {name: (dtype, shape)}.

    Data is zero-filled; only the header (dtype/shape/data_offsets) matters for
    the byte accounting, which reads spans, not values.
    """
    header = {}
    off = 0
    for name, (dtype, shape) in tensors.items():
        nbytes = fp._ST_DTYPE_BYTES[dtype]
        for d in shape:
            nbytes *= d
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + nbytes]}
        off += nbytes
    blob = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\x00" * off)


def test_source_checkpoint_bytes_reads_spans(tmp_path):
    _write_safetensors(tmp_path / "model-00001.safetensors", {
        "embed.weight": ("BF16", (100, 8)),      # 100*8*2 = 1600
        "layer.w.weight": ("BF16", (4, 8)),      # 4*8*2 = 64
    })
    _write_safetensors(tmp_path / "model-00002.safetensors", {
        "lm_head.weight": ("BF16", (100, 8)),    # 1600
    })
    total, by_dtype = fp.source_checkpoint_bytes(str(tmp_path))
    assert total == 1600 + 64 + 1600
    assert by_dtype == {"BF16": 1600 + 64 + 1600}
    assert fp.dominant_source_bytes_per_param(by_dtype) == 2


def test_source_checkpoint_bytes_no_shards(tmp_path):
    with pytest.raises(FileNotFoundError):
        fp.source_checkpoint_bytes(str(tmp_path))


def test_dominant_source_bytes_per_param_fp8():
    # native-fp8 source dominated by F8_E4M3 -> 1 byte/param
    assert fp.dominant_source_bytes_per_param({"F8_E4M3": 80, "BF16": 5}) == 1
    assert fp.dominant_source_bytes_per_param({}) == 2          # default bf16
    assert fp.dominant_source_bytes_per_param({"WEIRD": 9}) == 2  # unknown -> 2


def test_source_regime_robust_to_large_vocab_fp8():
    # The whole point: a large-vocab fp8 model where bf16 embed+lm_head OUTMASS
    # the fp8 body fools dominant-by-mass (-> bf16) but source_regime keys off
    # the *presence* of fp8 (which only the body has) -> correctly 'fp8'.
    by = {"BF16": 16000, "F8_E4M3": 10000, "F32": 4}
    assert fp.dominant_source_bytes_per_param(by) == 2   # mass says bf16 (wrong)
    assert fp.source_regime(by) == "fp8"                 # presence says fp8 (right)
    assert fp.source_regime({"BF16": 999}) == "bf16"
    assert fp.source_regime({}) == "bf16"


def test_assignment_artifact_bytes_residual_floor():
    # One body Linear (4x8, 32 params) re-encoded NVFP4; everything else is the
    # floor (kept at source precision). source_total carries a 1600-byte embed +
    # 1600-byte lm_head + the body's own 64 source bytes = 3264.
    stats = {"layer.w": {"n_params": 32, "in_features": 8, "out_features": 4}}
    source_total = 3264
    r = fp.assignment_artifact_bytes(
        {"layer.w": "NVFP4"}, stats,
        source_total_bytes=source_total, regime="bf16",
        source_manifest=None,   # explicit: regime-wide approximation
    )
    # + 8 B fp32 NVFP4 global sidecars (weight_global_scale +
    # input_global_scale) the export emits per 2-D Linear (§3.14 fix).
    body_q = fr.get_format("NVFP4").memory_bytes_for_shape((4, 8)) + 8
    # floor = source_total - reencoded_source = 3264 - 32*2 = 3200 (embed+lm_head)
    assert r["floor_bytes"] == 3200
    assert r["body_quant_bytes"] == body_q
    assert r["artifact_bytes"] == 3200 + body_q
    assert r["reencoded_source_bytes"] == 64
    assert r["n_reencoded"] == 1
    assert r["n_missing_stats"] == 0
    assert r["regime"] == "bf16"


def test_assignment_artifact_bytes_fp8_source_removes_scale_inv():
    # fp8-native source: each re-encoded Linear ships fp8 weight + fp32 128x128
    # weight_scale_inv. The floor must remove BOTH (regime='fp8'), else the
    # source scale_inv is double-counted (the old scalar-src_bpp bug).
    stats = {"layer.w": {"n_params": 65536, "in_features": 256, "out_features": 256}}
    src_weight = 65536                                    # fp8 weight bytes
    src_scale_inv = fr.get_format("FP8_SOURCE").memory_bytes_for_shape((256, 256)) - src_weight
    embed = 1600
    source_total = src_weight + src_scale_inv + embed
    r = fp.assignment_artifact_bytes(
        {"layer.w": "NVFP4"}, stats,
        source_total_bytes=source_total, regime="fp8",
        source_manifest=None,   # explicit: regime-wide approximation
    )
    # floor must be exactly the embed; the fp8 weight AND its scale_inv are removed
    assert r["floor_bytes"] == embed
    assert r["reencoded_source_bytes"] == src_weight + src_scale_inv
    assert r["body_quant_bytes"] == (
        fr.get_format("NVFP4").memory_bytes_for_shape((256, 256)) + 8)
    assert r["artifact_bytes"] == embed + r["body_quant_bytes"]
    # the old scalar (n_params*1) would have left src_scale_inv in the floor:
    old_floor_bug = source_total - src_weight  # = embed + src_scale_inv
    assert old_floor_bug == embed + src_scale_inv and r["floor_bytes"] < old_floor_bug


def test_assignment_artifact_bytes_bf16_passthrough_is_floor_equivalent():
    # Re-encoding a tensor to BF16 must equal leaving it in the (bf16) floor:
    # body_quant(BF16) == source bytes, so artifact == source_total exactly.
    stats = {"layer.w": {"n_params": 32, "in_features": 8, "out_features": 4}}
    r = fp.assignment_artifact_bytes(
        {"layer.w": "BF16"}, stats,
        source_total_bytes=3264, regime="bf16", source_manifest=None,
    )
    assert r["artifact_bytes"] == 3264


def test_assignment_artifact_bytes_cb_uses_exact_tensor_and_shared_sidecar():
    names = ["layer.0.q_proj", "layer.1.q_proj"]
    stats = {
        name: {
            "n_params": 64 * 256,
            "in_features": 256,
            "out_features": 64,
        }
        for name in names
    }
    source_total = 2 * 64 * 256 * 2 + 1234
    assignment = {name: "NVFP4_CB_K16" for name in names}
    with pytest.raises(ValueError, match="CBSerializationContext"):
        fp.assignment_artifact_bytes(
            assignment,
            stats,
            source_total_bytes=source_total,
            regime="bf16",
            source_manifest=None,
        )
    result = fp.assignment_artifact_bytes(
        assignment,
        stats,
        source_total_bytes=source_total,
        regime="bf16",
        source_manifest=None,
        cb_serialization_context=CBSerializationContext.production(),
    )
    # Static fused-W4A4 contract adds one canonical F32
    # input_global_scale scalar per FP4-CB target.
    tensor_bytes = 2 * 64 * (4 * 16 + 9) + 2 * 4
    sidecar_bytes = 2 * 256 * 4 * 2
    assert result["floor_bytes"] == 1234
    assert result["cb_tensor_payload_bytes"] == tensor_bytes
    assert result["cb_codebook_sidecar_bytes"] == sidecar_bytes
    assert result["body_quant_bytes"] == tensor_bytes + sidecar_bytes
    assert result["artifact_bytes"] == 1234 + tensor_bytes + sidecar_bytes
    assert result["artifact_payload_bytes"] == result["artifact_bytes"]
    assert result["artifact_byte_scope"] == "safetensors_tensor_data_spans"
    assert result["export_directory_bytes"] is None
    assert result["cb_serialized_payload"]["global_scale_bytes"] == 0
    assert result["cb_serialized_payload"][
        "input_global_scale_bytes"
    ] == 8


def test_per_expert_assignment_prices_physical_substacks_before_export():
    prefix = "model.layers.0.mlp.experts"
    assignment = {}
    stats = {}
    for expert_id in range(4):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            qname = f"{prefix}.{expert_id}.{projection}"
            if projection in ("gate_proj", "up_proj"):
                fmt = "NVFP4_CB_K16" if expert_id < 2 else "FP8_CB_K28"
            else:
                fmt = (
                    "NVFP4_CB_K16", "FP8_CB_K28",
                    "MXFP4_SOURCE", "MXFP4_SOURCE",
                )[expert_id]
            assignment[qname] = fmt
            stats[qname] = {
                "n_params": 256 * 256,
                "in_features": 256,
                "out_features": 256,
            }
    source_total = sum(2 * row["n_params"] for row in stats.values())
    result = fp.assignment_artifact_bytes(
        {}, stats,
        source_total_bytes=source_total,
        source_manifest=None,
        regime="bf16",
        cb_serialization_context=CBSerializationContext.production(),
        per_expert_assignment=assignment,
    )
    payload = result["per_expert_format_group_payload"]
    assert len(payload["groups"]) == 5  # w13:2, w2:3
    assert sorted(
        len(group["expert_ids"]) for group in payload["groups"].values()
    ) == [1, 1, 2, 2, 2]
    cb_groups = [
        group for group in payload["groups"].values()
        if group["format"].endswith(("K16", "K28"))
    ]
    assert len(cb_groups) == 4
    # Same formats in w13 and w2 have distinct physical sub-stacks and hence
    # one sidecar charge each; no expert row pays a sidecar by itself.
    assert payload["codebook_sidecar_bytes"] == sum(
        group["codebook_sidecar_bytes"] for group in cb_groups
    )
    assert all(group["codebook_sidecar_bytes"] > 0 for group in cb_groups)
    assert result["body_quant_bytes"] == payload["total_bytes"]


def test_uniform_per_expert_footprint_stays_on_legacy_shared_stack_path():
    prefix = "model.layers.0.mlp.experts"
    assignment = {
        f"{prefix}.{expert_id}.{projection}": "NVFP4_CB_K16"
        for expert_id in range(3)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    stats = {
        qname: {
            "n_params": 256 * 256,
            "in_features": 256,
            "out_features": 256,
        }
        for qname in assignment
    }
    common = dict(
        source_total_bytes=sum(2 * row["n_params"] for row in stats.values()),
        source_manifest=None,
        regime="bf16",
        cb_serialization_context=CBSerializationContext.production(),
    )
    legacy = fp.assignment_artifact_bytes(assignment, stats, **common)
    uniform = fp.assignment_artifact_bytes(
        {}, stats, per_expert_assignment=assignment, **common
    )
    assert uniform["artifact_bytes"] == legacy["artifact_bytes"]
    assert uniform["per_expert_format_group_payload"]["groups"] == {}


def test_assignment_artifact_bytes_missing_stats_stay_in_floor():
    # A name absent from stats is not subtracted from the floor (stays at source
    # precision) and is counted as missing — the total is still well-defined.
    stats = {"layer.w": {"n_params": 32, "in_features": 8, "out_features": 4}}
    r = fp.assignment_artifact_bytes(
        {"layer.w": "NVFP4", "ghost.w": "NVFP4"}, stats,
        source_total_bytes=3264, regime="bf16", source_manifest=None,
    )
    assert r["n_missing_stats"] == 1
    assert r["n_reencoded"] == 1


def test_assignment_artifact_gb_matches_bytes():
    stats = {"layer.w": {"n_params": 32, "in_features": 8, "out_features": 4}}
    kw = dict(source_total_bytes=3264, regime="bf16", source_manifest=None)
    gb = fp.assignment_artifact_gb({"layer.w": "NVFP4"}, stats, **kw)
    b = fp.assignment_artifact_bytes({"layer.w": "NVFP4"}, stats, **kw)["artifact_bytes"]
    assert gb == pytest.approx(b / fp.GB)


def test_floor_bytes_for_model(tmp_path):
    _write_safetensors(tmp_path / "m.safetensors", {
        "embed.weight": ("BF16", (100, 8)),      # 1600 floor
        "layer.w.weight": ("BF16", (4, 8)),      # 64 reencoded
    })
    stats = {"layer.w": {"n_params": 32, "in_features": 8, "out_features": 4}}
    info = fp.floor_bytes_for_model(str(tmp_path), ["layer.w"], stats)
    assert info["source_total_bytes"] == 1664
    assert info["regime"] == "bf16"
    assert info["source_bytes_per_param"] == 2
    assert info["reencoded_source_bytes"] == 64
    assert info["floor_bytes"] == 1600


def test_nvfp4_global_sidecar_bytes_dense_and_packed():
    """§3.14 (2026-07-02 audit): the export emits fp32 weight_global_scale +
    input_global_scale per NVFP4 2-D Linear (8 B, verified against shipped
    safetensors headers), and per expert × on-disk projection for packed 3-D
    tensors (gate_up_proj splits into gate_proj + up_proj per expert)."""
    assert fp.nvfp4_global_sidecar_bytes("model.layers.0.self_attn.q_proj",
                                         (128, 64)) == 8
    # down_proj: one projection per expert -> 8·E
    assert fp.nvfp4_global_sidecar_bytes(
        "model.layers.0.mlp.experts.down_proj", (256, 32, 64)) == 8 * 256
    # gate_up_proj: two on-disk projections per expert -> 8·E·2
    assert fp.nvfp4_global_sidecar_bytes(
        "model.layers.0.mlp.experts.gate_up_proj", (256, 128, 32)) == 16 * 256
    # Text-calibration-excluded stock targets are W4A16: one F32 weight
    # global per emitted Linear, with no input_global_scale tensor.
    assert fp.nvfp4_global_sidecar_bytes(
        "model.visual.blocks.0.mlp.fc1",
        (128, 64),
        weight_only=True,
    ) == 4
    assert fp.nvfp4_global_sidecar_bytes(
        "model.layers.0.mlp.experts.gate_up_proj",
        (256, 128, 32),
        weight_only=True,
    ) == 8 * 256


def test_assignment_artifact_bytes_honors_weight_only_nvfp4_marker():
    name = "model.visual.blocks.0.mlp.fc1"
    shape = (4, 16)
    stats = {
        name: {
            "n_params": 64,
            "in_features": 16,
            "out_features": 4,
            fp.NVFP4_WEIGHT_ONLY_STATS_KEY: True,
        },
    }
    result = fp.assignment_artifact_bytes(
        {name: "NVFP4"},
        stats,
        source_total_bytes=128,
        source_manifest=None,
        regime="bf16",
    )
    assert result["artifact_bytes"] == (
        fr.get_format("NVFP4").memory_bytes_for_shape(shape) + 4
    )

    from prismaquant.kl_measurement import assignment_bit_total

    assert assignment_bit_total(
        stats,
        {name: "NVFP4"},
        {"NVFP4": fr.get_format("NVFP4")},
    ) == 8.0 * (
        fr.get_format("NVFP4").memory_bytes_for_shape(shape) + 4
    )


def test_assignment_artifact_bytes_packed_nvfp4_counts_per_expert_globals():
    stats = {
        "layer.experts.gate_up_proj": {
            "n_params": 4 * 128 * 32, "in_features": 32,
            "out_features": 128, "num_experts": 4,
        },
    }
    r = fp.assignment_artifact_bytes(
        {"layer.experts.gate_up_proj": "NVFP4"}, stats,
        source_total_bytes=4 * 128 * 32 * 2, regime="bf16",
        source_manifest=None,
    )
    expected = (
        fr.get_format("NVFP4").memory_bytes_for_shape((4, 128, 32))
        + 8 * 4 * 2  # per-expert weight_global + input_global, gate+up
    )
    assert r["body_quant_bytes"] == expected


# ---------------------------------------------------------------------------
# Per-tensor source-byte manifest (mixed-precision sources)
# ---------------------------------------------------------------------------

def _mixed_source_checkpoint(tmp_path):
    """Two re-encoded tensors on a mixed source: an MXFP4-packed expert
    (I8 nibble weights, 0.5 B/param on disk, + E8M0 group scales) and an
    fp8 attention Linear (+ fp32 weight_scale_inv), plus a BF16 floor
    tensor. The fp8 dtype flips the regime detector to "fp8", so the
    regime path charges the packed expert 1 B/LOGICAL-param — about twice
    its actual on-disk bytes."""
    _write_safetensors(tmp_path / "m.safetensors", {
        # logical (64, 256) packed 2-per-byte -> stored (64, 128) I8 = 8192 B
        "layers.0.experts.0.down_proj.weight": ("I8", (64, 128)),
        # E8M0 scale per 32-element group: (64, 8) U8 = 512 B
        "layers.0.experts.0.down_proj.scale": ("U8", (64, 8)),
        "layers.0.self_attn.q_proj.weight": ("F8_E4M3", (32, 32)),   # 1024 B
        "layers.0.self_attn.q_proj.weight_scale_inv": ("F32", (1, 1)),  # 4 B
        "embed.weight": ("BF16", (100, 8)),                         # 1600 B
    })
    stats = {
        "layers.0.experts.0.down_proj": {
            "n_params": 64 * 256, "in_features": 256, "out_features": 64},
        "layers.0.self_attn.q_proj": {
            "n_params": 32 * 32, "in_features": 32, "out_features": 32},
    }
    reencoded = list(stats)
    return stats, reencoded


def test_source_tensor_bytes_manifest_charges_actual_spans(tmp_path):
    _mixed_source_checkpoint(tmp_path)
    manifest = fp.source_tensor_bytes_manifest(str(tmp_path))
    assert manifest["layers.0.experts.0.down_proj"] == 8192 + 512
    assert manifest["layers.0.self_attn.q_proj"] == 1024 + 4
    assert manifest["embed"] == 1600


def test_manifest_floor_non_negative_where_regime_path_goes_negative(tmp_path):
    stats, reencoded = _mixed_source_checkpoint(tmp_path)
    total, by_dtype = fp.source_checkpoint_bytes(str(tmp_path))
    regime = fp.source_regime(by_dtype)
    assert regime == "fp8"

    # Regime-wide accounting: the packed expert's LOGICAL params are charged
    # 1 B each -> more bytes removed than the tensor (or checkpoint) holds.
    regime_removed = {
        q: fp.reencoded_source_bytes_for_shape(
            (stats[q]["out_features"], stats[q]["in_features"]), regime)
        for q in reencoded
    }
    regime_floor = total - sum(regime_removed.values())
    assert regime_floor < 0
    with pytest.raises(ValueError, match="negative non-quantizable floor"):
        fp.check_floor_non_negative(
            regime_floor, total, regime_removed, context="unit")

    # Manifest accounting: actual header spans can never exceed the total.
    manifest = fp.source_tensor_bytes_manifest(str(tmp_path))
    manifest_floor = total - sum(manifest[q] for q in reencoded)
    assert manifest_floor == 1600  # exactly the BF16 floor tensor
    fp.check_floor_non_negative(
        manifest_floor, total, {q: manifest[q] for q in reencoded},
        context="unit")  # does not raise


def test_floor_bytes_for_model_uses_manifest_on_mixed_source(tmp_path):
    stats, reencoded = _mixed_source_checkpoint(tmp_path)
    info = fp.floor_bytes_for_model(str(tmp_path), reencoded, stats)
    assert info["regime"] == "fp8"
    assert info["reencoded_source_bytes"] == (8192 + 512) + (1024 + 4)
    assert info["floor_bytes"] == 1600


def test_assignment_artifact_bytes_manifest_agrees_with_floor_model(tmp_path):
    # PR-15 review: assignment_artifact_bytes must use the SAME manifest
    # accounting as floor_bytes_for_model (the regime path merely raised on
    # this mixed source instead of pricing it).
    stats, reencoded = _mixed_source_checkpoint(tmp_path)
    info = fp.floor_bytes_for_model(str(tmp_path), reencoded, stats)
    r = fp.assignment_artifact_bytes(
        {q: "NVFP4" for q in reencoded}, stats,
        source_total_bytes=info["source_total_bytes"],
        regime=info["regime"],
        source_manifest=info["source_manifest"],
    )
    assert r["source_accounting"] == "per_tensor_manifest"
    assert r["floor_bytes"] == info["floor_bytes"] == 1600
    assert r["reencoded_source_bytes"] == info["reencoded_source_bytes"]
    # The regime path on the same mixed source cannot agree — it goes
    # negative and is rejected, never silently shipped.
    with pytest.raises(ValueError, match="negative non-quantizable floor"):
        fp.assignment_artifact_bytes(
            {q: "NVFP4" for q in reencoded}, stats,
            source_total_bytes=info["source_total_bytes"], regime="fp8",
            source_manifest=None,
        )


# ---------------------------------------------------------------------------
# Packed-MoE expert name resolution (PR-15 review: the manifest must price
# packed allocator names on BOTH on-disk layouts)
# ---------------------------------------------------------------------------

# hidden=32, intermediate=16, 2 experts. Per-expert projections on disk:
# gate/up (16, 32) each, down (32, 16). The allocator names the PACKED live
# params: gate_up_proj = output-axis gate+up fusion -> (2, 32, 32), and
# down_proj -> (2, 32, 16).
_PACKED_STATS = {
    "layers.0.mlp.experts.gate_up_proj": {
        "n_params": 2 * 32 * 32, "in_features": 32, "out_features": 32,
        "num_experts": 2},
    "layers.0.mlp.experts.down_proj": {
        "n_params": 2 * 32 * 16, "in_features": 16, "out_features": 32,
        "num_experts": 2},
}
_PACKED_NAMES = list(_PACKED_STATS)


def _case_a_per_expert_disk(tmp_path):
    """CASE A: per-expert 2-D tensors on disk; allocator names are packed."""
    _write_safetensors(tmp_path / "m.safetensors", {
        "layers.0.mlp.experts.0.gate_proj.weight": ("BF16", (16, 32)),  # 1024
        "layers.0.mlp.experts.0.up_proj.weight": ("BF16", (16, 32)),    # 1024
        "layers.0.mlp.experts.0.down_proj.weight": ("BF16", (32, 16)),  # 1024
        "layers.0.mlp.experts.1.gate_proj.weight": ("BF16", (16, 32)),
        "layers.0.mlp.experts.1.up_proj.weight": ("BF16", (16, 32)),
        "layers.0.mlp.experts.1.down_proj.weight": ("BF16", (32, 16)),
        "embed.weight": ("BF16", (100, 8)),                             # 1600
    })


def test_packed_expert_alias():
    assert fp.packed_expert_alias("layers.0.mlp.experts.7.gate_proj") == \
        "layers.0.mlp.experts.gate_up_proj"
    assert fp.packed_expert_alias("layers.0.mlp.experts.7.up_proj") == \
        "layers.0.mlp.experts.gate_up_proj"
    assert fp.packed_expert_alias("layers.0.mlp.experts.7.down_proj") == \
        "layers.0.mlp.experts.down_proj"
    # not per-expert / unknown projection / non-expert container -> None
    assert fp.packed_expert_alias("layers.0.mlp.experts.gate_up_proj") is None
    assert fp.packed_expert_alias("layers.0.mlp.experts.7.w1") is None
    assert fp.packed_expert_alias("layers.0.self_attn.q_proj") is None
    # profile callable wins over the fallback
    assert fp.packed_expert_alias(
        "layers.0.mlp.experts.3.w1",
        lambda p: "w1_w3" if p in ("w1", "w3") else None,
    ) == "layers.0.mlp.experts.w1_w3"


def test_manifest_case_a_per_expert_disk_aggregates_to_packed(tmp_path):
    _case_a_per_expert_disk(tmp_path)
    m = fp.source_tensor_bytes_manifest(str(tmp_path))
    # gate+up fuse into gate_up across both experts; down aggregates 1:1.
    assert m["layers.0.mlp.experts.gate_up_proj"] == 4 * 1024
    assert m["layers.0.mlp.experts.down_proj"] == 2 * 1024
    # per-expert entries are kept for per-expert-named allocations
    assert m["layers.0.mlp.experts.0.gate_proj"] == 1024
    assert m["layers.0.mlp.experts.1.down_proj"] == 1024

    info = fp.floor_bytes_for_model(str(tmp_path), _PACKED_NAMES, _PACKED_STATS)
    assert info["reencoded_source_bytes"] == 6 * 1024
    assert info["floor_bytes"] == 1600  # exactly the embed

    # end-to-end: artifact = floor + quantized expert bodies, floor exact
    r = fp.assignment_artifact_bytes(
        {q: "NVFP4" for q in _PACKED_NAMES}, _PACKED_STATS,
        source_total_bytes=info["source_total_bytes"],
        regime=info["regime"], source_manifest=info["source_manifest"],
    )
    body = (fr.get_format("NVFP4").memory_bytes_for_shape((2, 32, 32))
            + 8 * 2 * 2   # gate_up: per-expert globals x 2 projections
            + fr.get_format("NVFP4").memory_bytes_for_shape((2, 32, 16))
            + 8 * 2)      # down: per-expert globals x 1 projection
    assert r["floor_bytes"] == 1600
    assert r["artifact_bytes"] == 1600 + body


def test_manifest_case_b_packed_3d_without_weight_suffix(tmp_path):
    """CASE B: packed 3-D checkpoint keys carry NO '.weight' suffix (the
    transformers packed-experts convention: the param IS the key). They must
    not be skipped, and sidecars still attach to the suffix-less base."""
    _write_safetensors(tmp_path / "m.safetensors", {
        "layers.0.mlp.experts.gate_up_proj": ("F8_E4M3", (2, 32, 32)),   # 2048
        "layers.0.mlp.experts.gate_up_proj.scale": ("U8", (2, 32, 1)),   # 64
        "layers.0.mlp.experts.down_proj": ("F8_E4M3", (2, 32, 16)),      # 1024
        "embed.weight": ("BF16", (100, 8)),                              # 1600
    })
    m = fp.source_tensor_bytes_manifest(str(tmp_path))
    assert m["layers.0.mlp.experts.gate_up_proj"] == 2048 + 64
    assert m["layers.0.mlp.experts.down_proj"] == 1024
    # standalone sidecar keys never become their own manifest entries
    assert "layers.0.mlp.experts.gate_up_proj.scale" not in m

    info = fp.floor_bytes_for_model(str(tmp_path), _PACKED_NAMES, _PACKED_STATS)
    assert info["reencoded_source_bytes"] == 2048 + 64 + 1024
    assert info["floor_bytes"] == 1600


def test_manifest_keeps_tensors_the_live_name_map_declines(tmp_path):
    """A live-graph mapper returning None must not delete source bytes.

    Measured regression (round-2 review, real Qwen3.6-27B checkpoint): the MTP
    sidecar ships as `mtp.*` tensors on disk, but transformers v5 dropped the
    module, so `Qwen3_5DenseProfile.checkpoint_to_live_name("mtp.fc.weight")`
    is None. The exporter still re-encodes `mtp.*` from exactly those bytes and
    the allocator assigns them under their raw names, so dropping them from the
    manifest made every `--target-disk-gb` run on an MTP-carrying model die in
    `resolve_reencoded_source_bytes`. Raw-key fallback resolves them, and stays
    inert for tensors nothing re-encodes (the floor subtracts only resolved
    re-encoded spans).
    """
    _write_safetensors(tmp_path / "m.safetensors", {
        "model.layers.0.mlp.gate_proj.weight": ("BF16", (16, 8)),  # 256
        "mtp.fc.weight": ("BF16", (8, 8)),                         # 128
        "mtp.norm.weight": ("BF16", (8,)),                         # 16
        "model.visual.blocks.0.attn.qkv.weight": ("BF16", (8, 8)),  # 128
        "embed.weight": ("BF16", (100, 8)),                        # 1600
    })
    # Both live classes the Qwen profiles decline: the MTP sidecar and, on a
    # multimodal probe, the visual tower.
    declines = lambda k: (  # noqa: E731
        None if k.startswith(("mtp.", "model.visual.")) else k)

    m = fp.source_tensor_bytes_manifest(str(tmp_path), name_map=declines)
    assert m["mtp.fc"] == 128, "declined key must survive under its raw name"
    assert m["model.visual.blocks.0.attn.qkv"] == 128
    assert m["model.layers.0.mlp.gate_proj"] == 256

    stats = {n: {"n_params": 64, "in_features": 8, "out_features": 8}
             for n in ("model.layers.0.mlp.gate_proj", "mtp.fc")}
    # Re-encoding the MTP Linear removes its span from the floor...
    info = fp.floor_bytes_for_model(
        str(tmp_path), ["model.layers.0.mlp.gate_proj", "mtp.fc"], stats,
        name_map=declines)
    assert info["reencoded_source_bytes"] == 256 + 128
    # embed 1600 + mtp.norm 16 + the un-re-encoded visual tensor 128
    assert info["floor_bytes"] == 1600 + 16 + 128
    # ...and a declined tensor NOT re-encoded stays in the floor untouched:
    # the extra manifest entries cannot move it.
    body_only = fp.floor_bytes_for_model(
        str(tmp_path), ["model.layers.0.mlp.gate_proj"], stats,
        name_map=declines)
    assert body_only["floor_bytes"] == 1600 + 16 + 128 + 128


def test_unresolved_reencoded_name_is_a_hard_error(tmp_path):
    # PR-15 review: a re-encoded Linear the manifest cannot resolve must be
    # a hard error NAMING the tensor, raised before the numbers are consumed
    # — not a post-hoc warning after a fatal "below the floor" exit.
    with pytest.raises(ValueError, match=r"ghost\.proj"):
        fp.resolve_reencoded_source_bytes(
            {"layer.w": 64}, ["layer.w", "ghost.proj"], context="unit")

    _case_a_per_expert_disk(tmp_path)
    with pytest.raises(ValueError, match=r"not\.in\.checkpoint"):
        fp.floor_bytes_for_model(
            str(tmp_path), _PACKED_NAMES + ["not.in.checkpoint"], _PACKED_STATS)

    stats = dict(_PACKED_STATS)
    stats["not.in.checkpoint"] = {
        "n_params": 32, "in_features": 32, "out_features": 1}
    with pytest.raises(ValueError, match=r"not\.in\.checkpoint"):
        fp.assignment_artifact_bytes(
            {"not.in.checkpoint": "NVFP4"}, stats,
            source_total_bytes=10_000, regime="bf16",
            source_manifest={"layer.w": 64})


# ---------------------------------------------------------------------------
# Double-charged source spans (issue #23): the manifest stores a per-expert
# Linear BOTH under its own name and inside its packed-parent aggregate, so
# either naming scheme resolves. Charging both subtracts the same on-disk
# bytes twice — the floor is under-counted by the whole expert mass and an
# over-budget artifact "fits". That was guarded only by a docstring
# convention; the manifest now carries per-entry span provenance so the
# resolver can reject it structurally.
# ---------------------------------------------------------------------------

def test_manifest_carries_span_provenance(tmp_path):
    _case_a_per_expert_disk(tmp_path)
    m = fp.source_tensor_bytes_manifest(str(tmp_path))
    assert isinstance(m, fp.SourceByteManifest)
    # A per-expert entry covers exactly its own checkpoint tensor...
    assert m.spans["layers.0.mlp.experts.0.gate_proj"] == frozenset(
        {"layers.0.mlp.experts.0.gate_proj"})
    # ...and the packed aggregate covers every per-expert tensor it summed,
    # which is exactly why the two overlap.
    assert m.spans["layers.0.mlp.experts.gate_up_proj"] == frozenset({
        "layers.0.mlp.experts.0.gate_proj", "layers.0.mlp.experts.0.up_proj",
        "layers.0.mlp.experts.1.gate_proj", "layers.0.mlp.experts.1.up_proj",
    })
    assert m.spans["embed"] == frozenset({"embed"})
    # Byte totals are untouched by provenance bookkeeping.
    assert m["layers.0.mlp.experts.gate_up_proj"] == 4 * 1024
    assert m["layers.0.mlp.experts.0.gate_proj"] == 1024


def test_both_naming_schemes_resolve_on_their_own(tmp_path):
    """The legitimate cases must keep working: a re-encoded-name list may be
    ALL packed names or ALL per-expert names. Only mixing them is the bug."""
    _case_a_per_expert_disk(tmp_path)
    m = fp.source_tensor_bytes_manifest(str(tmp_path))

    packed = fp.resolve_reencoded_source_bytes(
        m, _PACKED_NAMES, context="unit")
    assert sum(packed.values()) == 6 * 1024

    per_expert_names = [
        f"layers.0.mlp.experts.{i}.{proj}"
        for i in (0, 1) for proj in ("gate_proj", "up_proj", "down_proj")
    ]
    per_expert = fp.resolve_reencoded_source_bytes(
        m, per_expert_names, context="unit")
    # Same underlying source mass, reached through the other naming scheme.
    assert sum(per_expert.values()) == 6 * 1024

    # Requesting the SAME name twice is idempotent, not a double charge: the
    # result is keyed by name, so those bytes are only summed once.
    dup = fp.resolve_reencoded_source_bytes(
        m, _PACKED_NAMES + _PACKED_NAMES, context="unit")
    assert dup == packed

    # ...and the floor is identical whichever scheme names the tensors.
    info_packed = fp.floor_bytes_for_model(
        str(tmp_path), _PACKED_NAMES, _PACKED_STATS)
    info_per_expert = fp.floor_bytes_for_model(
        str(tmp_path), per_expert_names, _PACKED_STATS)
    assert info_packed["floor_bytes"] == info_per_expert["floor_bytes"] == 1600


def test_double_charged_span_is_rejected(tmp_path):
    _case_a_per_expert_disk(tmp_path)
    m = fp.source_tensor_bytes_manifest(str(tmp_path))

    # Mixing schemes for the SAME underlying tensor: the packed aggregate and
    # one of the per-expert Linears it already contains.
    with pytest.raises(ValueError, match="charged twice") as exc:
        fp.resolve_reencoded_source_bytes(
            m,
            ["layers.0.mlp.experts.gate_up_proj",
             "layers.0.mlp.experts.0.gate_proj"],
            context="unit")
    msg = str(exc.value)
    assert "layers.0.mlp.experts.gate_up_proj" in msg
    assert "layers.0.mlp.experts.0.gate_proj" in msg

    # ...through both public consumers, not just the helper.
    with pytest.raises(ValueError, match="charged twice"):
        fp.floor_bytes_for_model(
            str(tmp_path),
            _PACKED_NAMES + ["layers.0.mlp.experts.1.down_proj"],
            _PACKED_STATS)

    stats = dict(_PACKED_STATS)
    stats["layers.0.mlp.experts.1.down_proj"] = {
        "n_params": 32 * 16, "in_features": 16, "out_features": 32}
    with pytest.raises(ValueError, match="charged twice"):
        fp.assignment_artifact_bytes(
            {q: "NVFP4" for q in
             _PACKED_NAMES + ["layers.0.mlp.experts.1.down_proj"]},
            stats,
            source_total_bytes=m["embed"] + 6 * 1024,
            source_manifest=m)


def test_double_charge_would_have_under_counted_the_floor(tmp_path):
    """Why it matters: the double charge is silent under the old guard.

    check_floor_non_negative only fires when the over-subtraction exceeds the
    ENTIRE floor. Here it does not — the floor merely shrinks by the expert
    mass, so the artifact reads ~2 KB smaller than it is and an over-budget
    allocation would 'fit' a byte budget."""
    _case_a_per_expert_disk(tmp_path)
    m = fp.source_tensor_bytes_manifest(str(tmp_path))
    total, _by_dtype = fp.source_checkpoint_bytes(str(tmp_path))
    honest = total - 6 * 1024
    assert honest == 1600

    doubled_names = _PACKED_NAMES + ["layers.0.mlp.experts.1.down_proj"]
    doubled = total - sum(m[q] for q in doubled_names)
    assert doubled == honest - 1024 < honest
    # Still non-negative, so the pre-existing guard says nothing at all.
    fp.check_floor_non_negative(
        doubled, total, {q: m[q] for q in doubled_names}, context="unit")
    # The provenance check is what catches it.
    with pytest.raises(ValueError, match="charged twice"):
        fp.resolve_reencoded_source_bytes(m, doubled_names, context="unit")


def test_plain_dict_manifest_skips_the_overlap_check_but_still_resolves(tmp_path):
    """A hand-built manifest carries no provenance: the byte lookup still
    works, the overlap check is simply not available (and an explicit `spans`
    argument re-enables it)."""
    plain = {"a.w": 64, "b.w": 64}
    assert fp.resolve_reencoded_source_bytes(
        plain, ["a.w", "b.w"], context="unit") == {"a.w": 64, "b.w": 64}
    with pytest.raises(ValueError, match="charged twice"):
        fp.resolve_reencoded_source_bytes(
            plain, ["a.w", "b.w"], context="unit",
            spans={"a.w": ["shard0.t0"], "b.w": ["shard0.t0"]})


def test_source_manifest_is_a_required_keyword():
    """Issue #23: the regime-wide accounting is a legacy approximation exact
    only on a uniform source. It must be reachable only by an explicit
    `source_manifest=None`, never by forgetting the kwarg."""
    stats = {"layer.w": {"n_params": 32, "in_features": 8, "out_features": 4}}
    with pytest.raises(TypeError, match="source_manifest"):
        fp.assignment_artifact_bytes(
            {"layer.w": "NVFP4"}, stats, source_total_bytes=3264)
    with pytest.raises(TypeError, match="source_manifest"):
        fp.assignment_artifact_gb(
            {"layer.w": "NVFP4"}, stats, source_total_bytes=3264)
    # ...and the explicit form reports which accounting actually ran.
    assert fp.assignment_artifact_bytes(
        {"layer.w": "NVFP4"}, stats, source_total_bytes=3264,
        source_manifest=None)["source_accounting"] == "regime"
    assert fp.assignment_artifact_bytes(
        {"layer.w": "NVFP4"}, stats, source_total_bytes=3264,
        source_manifest={"layer.w": 64},
    )["source_accounting"] == "per_tensor_manifest"


# ---------------------------------------------------------------------------
# partitioned_source_total_bytes — an artifact that ships only PART of the
# checkpoint (DSv4-Flash: the draft head is its own directory).
# ---------------------------------------------------------------------------


def _partitioned_disk(tmp_path):
    """A body plus a draft head that ships as a SEPARATE artifact."""
    _write_safetensors(tmp_path / "m.safetensors", {
        "layers.0.self_attn.q_proj.weight": ("BF16", (16, 32)),  # 1024
        "layers.0.mlp.gate_proj.weight": ("BF16", (16, 32)),     # 1024
        "embed.weight": ("BF16", (100, 8)),                      # 1600
        "mtp.0.attn.wo_a.weight": ("BF16", (16, 32)),            # 1024
        "mtp.0.mlp.gate_proj.weight": ("BF16", (16, 32)),        # 1024
    })


def test_partition_removes_exactly_the_excluded_spans(tmp_path):
    _partitioned_disk(tmp_path)
    total, _ = fp.source_checkpoint_bytes(str(tmp_path))
    m = fp.source_tensor_bytes_manifest(str(tmp_path))
    assert total == 5696

    part = fp.partitioned_source_total_bytes(m, total, ["mtp."], context="unit")
    assert part["excluded_source_bytes"] == 2048
    assert part["n_excluded"] == 2
    assert part["source_total_bytes"] == 5696 - 2048
    assert part["excluded_prefixes"] == ("mtp.",)


def test_partition_lowers_the_floor_by_the_excluded_mass(tmp_path):
    """The whole point: a rung must not be priced heavier than it ships."""
    _partitioned_disk(tmp_path)
    total, _ = fp.source_checkpoint_bytes(str(tmp_path))
    m = fp.source_tensor_bytes_manifest(str(tmp_path))
    stats = {"layers.0.mlp.gate_proj": {
        "n_params": 512, "in_features": 32, "out_features": 16}}
    assignment = {"layers.0.mlp.gate_proj": "NVFP4"}

    full = fp.assignment_artifact_bytes(
        assignment, stats, source_total_bytes=total, source_manifest=m,
        context="unit")
    part = fp.partitioned_source_total_bytes(m, total, ["mtp."], context="unit")
    partitioned = fp.assignment_artifact_bytes(
        assignment, stats, source_total_bytes=part["source_total_bytes"],
        source_manifest=m, context="unit")

    # Exactly the excluded mass, and nothing else moved: the body's quantized
    # bytes are identical, only the floor shrank.
    assert full["artifact_payload_bytes"] - partitioned[
        "artifact_payload_bytes"] == 2048
    assert full["body_quant_bytes"] == partitioned["body_quant_bytes"]
    assert full["reencoded_source_bytes"] == partitioned[
        "reencoded_source_bytes"]


def test_partition_prefix_matching_nothing_is_a_hard_error(tmp_path):
    """A typo'd prefix excludes zero bytes, under-fills the budget by the
    whole excluded mass, and leaves every downstream number self-consistent —
    the artifact still 'fits'. It must never be a silent no-op."""
    _partitioned_disk(tmp_path)
    total, _ = fp.source_checkpoint_bytes(str(tmp_path))
    m = fp.source_tensor_bytes_manifest(str(tmp_path))
    with pytest.raises(ValueError, match="matched no tensor"):
        fp.partitioned_source_total_bytes(m, total, ["mtp_"], context="unit")
    # One good prefix does not excuse a bad one.
    with pytest.raises(ValueError, match="matched no tensor"):
        fp.partitioned_source_total_bytes(
            m, total, ["mtp.", "draft."], context="unit")


def test_partition_refuses_a_prefix_whose_matches_overlap(tmp_path):
    """Packed-expert spans are stored twice on purpose. A prefix that catches
    both namings must refuse, not subtract the same bytes twice — an
    over-budget artifact would otherwise read as fitting."""
    _case_a_per_expert_disk(tmp_path)
    total, _ = fp.source_checkpoint_bytes(str(tmp_path))
    m = fp.source_tensor_bytes_manifest(str(tmp_path))
    with pytest.raises(ValueError, match="charged twice"):
        fp.partitioned_source_total_bytes(
            m, total, ["layers.0.mlp.experts."], context="unit")


def test_partition_with_no_prefixes_is_an_exact_passthrough(tmp_path):
    _partitioned_disk(tmp_path)
    total, _ = fp.source_checkpoint_bytes(str(tmp_path))
    m = fp.source_tensor_bytes_manifest(str(tmp_path))
    part = fp.partitioned_source_total_bytes(m, total, [], context="unit")
    assert part["source_total_bytes"] == total
    assert part["excluded_source_bytes"] == 0
    assert part["n_excluded"] == 0


def test_partition_refuses_to_empty_the_checkpoint(tmp_path):
    _partitioned_disk(tmp_path)
    total, _ = fp.source_checkpoint_bytes(str(tmp_path))
    m = fp.source_tensor_bytes_manifest(str(tmp_path))
    with pytest.raises(ValueError, match="leaving nothing to price"):
        fp.partitioned_source_total_bytes(m, total, [""], context="unit")


# --- exclusion is only legal for the FLOOR -----------------------------------
# Exclusion and re-encoding both subtract from the same floor, independently:
# `assignment_artifact_bytes` computes `source_total - reencoded`, and the
# partition has already removed the excluded mass from `source_total`. A name
# in both is subtracted twice, every rung is priced too cheap, and the artifact
# overshoots. The exporter enforces this same invariant on --exclude-namespace.


def test_partition_refuses_to_exclude_an_allocatable_namespace(tmp_path):
    _partitioned_disk(tmp_path)
    total, _ = fp.source_checkpoint_bytes(str(tmp_path))
    m = fp.source_tensor_bytes_manifest(str(tmp_path))

    with pytest.raises(ValueError, match="allocator can assign"):
        fp.partitioned_source_total_bytes(
            m, total, ["mtp."], context="unit",
            assigned_names={"mtp.0.attn.wo_a": {}})


def test_partition_catches_the_other_name_vintage(tmp_path):
    """`mtp.` is the checkpoint spelling; a recipe says `model.mtp.`.

    A prefix that missed only on spelling would leave the double-subtraction
    in place while looking checked -- the same trap that made --mtp-format an
    inert no-op on this probe.
    """
    _partitioned_disk(tmp_path)
    total, _ = fp.source_checkpoint_bytes(str(tmp_path))
    m = fp.source_tensor_bytes_manifest(str(tmp_path))

    with pytest.raises(ValueError, match="allocator can assign"):
        fp.partitioned_source_total_bytes(
            m, total, ["mtp."], context="unit",
            assigned_names={"model.mtp.0.attn.wo_a": {}})


def test_partition_allows_the_shipping_shape(tmp_path):
    """The DSv4 case: mtp is pure floor, the body is what gets allocated."""
    _partitioned_disk(tmp_path)
    total, _ = fp.source_checkpoint_bytes(str(tmp_path))
    m = fp.source_tensor_bytes_manifest(str(tmp_path))

    part = fp.partitioned_source_total_bytes(
        m, total, ["mtp."], context="unit",
        assigned_names={
            "model.layers.0.self_attn.q_proj": {},
            "model.layers.0.mlp.gate_proj": {},
        })
    assert part["excluded_source_bytes"] == 2048
    # Unchanged from the no-universe call: the guard refuses, it never reprices.
    assert part == fp.partitioned_source_total_bytes(
        m, total, ["mtp."], context="unit")


# ---------------------------------------------------------------------------
# Name derivation routes through the shared projection layer (R5,
# walker/consumer-footprint): footprint keeps no private name mapping.
# The leaf rule and the packed-expert alias ARE
# prismaquant.name_projection's; these pins hold the projection-object
# path byte-identical to the raw-accessor path and hold refusals loud.
# ---------------------------------------------------------------------------


class _ExplodingCheckpointProfile(ModelProfile):
    """A profile whose checkpoint_to_live_name is a declaration bug."""

    name = "exploding-checkpoint-stub"

    @classmethod
    def matches(cls, model_type, architectures):
        return False

    def structure_spec(self):
        return None

    def checkpoint_to_live_name(self, ckpt_key, *, multimodal=False):
        raise RuntimeError("spec exploded")


def test_footprint_holds_no_private_name_mapping():
    """The leaf rule and the packed-expert parser are THE layer's.

    ``footprint`` re-exports them for the historic import paths; it does
    not reimplement them."""
    assert fp.strip_weight_leaf is npx.strip_weight_leaf
    assert fp.packed_expert_alias is npx.packed_expert_alias
    # The span-identity helper is a thin alias over the layer's leaf rule.
    assert fp.source_span_identity("a.b.weight") == npx.strip_weight_leaf(
        "a.b.weight")
    assert fp.source_span_identity("a.b") == "a.b"


def _projection_parity_checkpoint(tmp_path):
    """Per-expert-on-disk MoE spans, an fp8 scale sibling with no base
    tensor, a declined MTP tensor, and the BF16 floor — everything the
    manifest's name mapping has to survive identically under both forms."""
    _write_safetensors(tmp_path / "m.safetensors", {
        "model.layers.0.mlp.experts.0.gate_proj.weight": ("BF16", (16, 32)),
        "model.layers.0.mlp.experts.0.up_proj.weight": ("BF16", (16, 32)),
        "model.layers.0.mlp.experts.0.down_proj.weight": ("BF16", (32, 16)),
        "model.layers.0.mlp.experts.1.gate_proj.weight": ("BF16", (16, 32)),
        "model.layers.0.mlp.experts.1.up_proj.weight": ("BF16", (16, 32)),
        "model.layers.0.mlp.experts.1.down_proj.weight": ("BF16", (32, 16)),
        "model.layers.0.self_attn.q_proj.weight_scale_inv": ("F32", (1, 1)),
        "mtp.fc.weight": ("BF16", (8, 8)),                          # 128
        "embed.weight": ("BF16", (100, 8)),                         # 1600
    })


def test_manifest_via_projection_matches_the_raw_accessor_path(tmp_path):
    _projection_parity_checkpoint(tmp_path)
    profile = Qwen3Profile()
    legacy = fp.source_tensor_bytes_manifest(
        str(tmp_path),
        name_map=profile.checkpoint_to_live_name,
        expert_parent_for_projection=(
            profile.packed_expert_parent_for_projection),
    )
    via_layer = fp.source_tensor_bytes_manifest(
        str(tmp_path), projection=npx.NameProjection(profile))

    # Byte-for-byte identical manifest, provenance included.
    assert dict(via_layer) == dict(legacy)
    assert via_layer.spans == legacy.spans

    # The interesting branches really fired on this fixture:
    # per-expert spans aggregate into the packed allocator names ...
    assert via_layer["model.layers.0.mlp.experts.gate_up_proj"] == 4 * 1024
    assert via_layer["model.layers.0.mlp.experts.down_proj"] == 2 * 1024
    # ... a DECLARED drop keeps its raw checkpoint spelling (MTP) ...
    assert via_layer["mtp.fc"] == 128
    # ... and a standalone sidecar gets no entry of its own.
    assert "model.layers.0.self_attn.q_proj.weight_scale_inv" not in via_layer


def test_floor_bytes_for_model_accepts_a_projection(tmp_path):
    _case_a_per_expert_disk(tmp_path)
    profile = Qwen3Profile()
    via_layer = fp.floor_bytes_for_model(
        str(tmp_path), _PACKED_NAMES, _PACKED_STATS,
        projection=npx.NameProjection(profile))
    legacy = fp.floor_bytes_for_model(
        str(tmp_path), _PACKED_NAMES, _PACKED_STATS,
        name_map=profile.checkpoint_to_live_name,
        expert_parent_for_projection=profile.packed_expert_parent_for_projection)
    assert via_layer["floor_bytes"] == legacy["floor_bytes"] == 1600
    assert via_layer["reencoded_source_bytes"] == \
        legacy["reencoded_source_bytes"] == 6 * 1024
    assert dict(via_layer["source_manifest"]) == dict(legacy["source_manifest"])
    assert via_layer["source_manifest"].spans == legacy["source_manifest"].spans


def test_projection_and_raw_accessors_are_mutually_exclusive(tmp_path):
    _partitioned_disk(tmp_path)
    proj = npx.NameProjection(Qwen3Profile())
    with pytest.raises(ValueError, match="never both"):
        fp.source_tensor_bytes_manifest(
            str(tmp_path), name_map=lambda k: k, projection=proj)
    with pytest.raises(ValueError, match="never both"):
        fp.floor_bytes_for_model(
            str(tmp_path), ["layers.0.mlp.gate_proj"], {},
            name_map=lambda k: k, projection=proj)


def test_layer_refusals_propagate_out_of_the_manifest(tmp_path):
    """Requirement: a refused name propagates as a refusal — it must not be
    swallowed into a skip/None/zero (silent-zero is how wo_a stayed
    invisible). A raising or malformed profile accessor is a structured
    NameProjectionError out of source_tensor_bytes_manifest."""
    _write_safetensors(tmp_path / "m.safetensors",
                       {"a.weight": ("BF16", (2, 2))})
    proj = npx.NameProjection(_ExplodingCheckpointProfile())
    with pytest.raises(npx.NameProjectionError) as raising:
        fp.source_tensor_bytes_manifest(str(tmp_path), projection=proj)
    assert raising.value.code == "profile_accessor_failed"
    assert raising.value.name == "a.weight"

    class _Malformed(ModelProfile):
        name = "malformed-checkpoint-stub"

        @classmethod
        def matches(cls, model_type, architectures):
            return False

        def structure_spec(self):
            return None

        def checkpoint_to_live_name(self, ckpt_key, *, multimodal=False):
            return 42

    with pytest.raises(npx.NameProjectionError) as malformed:
        fp.source_tensor_bytes_manifest(
            str(tmp_path), projection=npx.NameProjection(_Malformed()))
    assert malformed.value.code == "malformed_profile_result"
