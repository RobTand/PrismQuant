"""Source-passthrough family: pricing, byte exactness, legality, allocation.

The family's claim is narrow and total: for a unit whose stored bytes ALREADY
are format F, shipping them unchanged costs exactly zero and occupies exactly
the source slice. These tests pin both halves of that claim, the source-kind
gate that decides where it applies, and the end-to-end consequence — that the
allocator will spend a sensitive layer's budget on a passthrough and record the
lane it rides.

See docs/lanes/nvfp4-cb/source-passthrough.md.
"""
from __future__ import annotations

import json
import struct

import pytest
import torch

import prismaquant.format_registry as fr
from prismaquant.activation_fair_pricing import BRANCH_SOURCE_PASSTHROUGH
from prismaquant.allocator_candidates import (
    PASSTHROUGH_SOURCE_REQUIREMENTS,
    PASSTHROUGH_WIRE_FORMAT_IDS,
    ROUTE_STATUS_BACKED,
    ROUTE_STATUS_BLOCKED,
    passthrough_serving_notes,
    ROUTE_PENDING_PASSTHROUGH_FORMATS,
    SOURCE_PASSTHROUGH_CONTRACTS,
    SOURCE_PASSTHROUGH_COST_SOURCE,
    SOURCE_PASSTHROUGH_FORMATS,
    build_candidates,
    check_format_applicability,
    cost_entry_activation_pricing_branch,
    cost_entry_is_exact_by_construction,
    cost_entry_is_source_passthrough,
    cost_entry_predicted_dloss,
    cost_entry_source,
    cost_entry_uses_measured_output_mse,
    selection_serving_lane_provenance,
    synthesized_source_passthrough_cost_entry,
)
from prismaquant.serving_profiles import check_serving_format, serving_lane_route

# The DSv4-Flash-0731 routed-expert and body shapes, and the byte counts the
# checkpoint's own safetensors headers report for them. These are measurements,
# not derivations: if the format arithmetic stops reproducing them, the artifact
# budget is false and the floor accounting no longer cancels.
EXPERT_W13_SHAPE = (2048, 4096)
EXPERT_W13_BYTES = 4_194_304 + 262_144
EXPERT_W2_SHAPE = (4096, 2048)
EXPERT_W2_BYTES = 4_194_304 + 262_144
BODY_WO_A_SHAPE = (8192, 4096)
BODY_WO_A_BYTES = 33_554_432 + 2_048
EXPERT_LAYER_BYTES = 3_422_552_064
N_EXPERTS = 256


# ---------------------------------------------------------------------------
# 1. Registry + byte exactness
# ---------------------------------------------------------------------------

def test_every_contract_names_a_registered_identity_format():
    """A contract may only name a format that is real, and that is a no-op.

    The zero cost is justified ONLY by the exporter copying bytes. A format
    whose weight or activation path actually transforms anything would be
    priced at zero on a false premise, so the identity is checked here rather
    than assumed by the pricing code.
    """
    probe = torch.randn(8, 128)
    for name, contract in SOURCE_PASSTHROUGH_CONTRACTS.items():
        spec = fr.get_format(name)
        assert spec.name == name
        assert contract.source_kind
        assert contract.serving_route
        torch.testing.assert_close(spec.quantize_dequantize(probe), probe)
        torch.testing.assert_close(
            spec.activation_quantize_dequantize(probe), probe)
        # act_bits absent/>=16 is the dtype-level fact the bit-exact and
        # P5a machinery both key off.
        assert not spec.act_quant_changes_input, name


def test_derived_views_agree_with_the_contract_table():
    assert PASSTHROUGH_SOURCE_REQUIREMENTS == {
        n: c.source_kind for n, c in SOURCE_PASSTHROUGH_CONTRACTS.items()}
    assert SOURCE_PASSTHROUGH_FORMATS == {
        n for n, c in SOURCE_PASSTHROUGH_CONTRACTS.items()
        if c.zero_cost_by_construction}
    assert ROUTE_PENDING_PASSTHROUGH_FORMATS == {
        n for n, c in SOURCE_PASSTHROUGH_CONTRACTS.items()
        if not c.route_backed}


@pytest.mark.parametrize("fmt,shape,expected", [
    ("MXFP4_SOURCE", EXPERT_W13_SHAPE, EXPERT_W13_BYTES),
    ("MXFP4_SOURCE", EXPERT_W2_SHAPE, EXPERT_W2_BYTES),
    ("FP8_BLOCK_UE8M0_SOURCE", BODY_WO_A_SHAPE, BODY_WO_A_BYTES),
])
def test_footprint_reproduces_the_checkpoints_own_bytes(fmt, shape, expected):
    assert fr.get_format(fmt).memory_bytes_for_shape(shape) == expected


def test_mxfp4_expert_layer_totals_match_the_checkpoint():
    """One DSv4 expert layer is 256 experts x {w1, w3, w2}."""
    per_expert = 2 * EXPERT_W13_BYTES + EXPERT_W2_BYTES
    assert N_EXPERTS * per_expert == EXPERT_LAYER_BYTES
    assert 43 * EXPERT_LAYER_BYTES == 147_169_738_752


def test_the_two_fp8_block_formats_are_not_interchangeable():
    """FP8_SOURCE and FP8_BLOCK_UE8M0_SOURCE differ in the scale plane.

    Same E4M3 elements and same 128x128 block, but FP32 scales vs one-byte
    UE8M0 exponents. Treating them as one format charges 4x the scale bytes
    and — worse on the export side — widens a UE8M0 plane to FP32, emitting a
    tensor the model's own loader does not expect.
    """
    fp32_scaled = fr.get_format("FP8_SOURCE")
    ue8m0 = fr.get_format("FP8_BLOCK_UE8M0_SOURCE")
    assert fp32_scaled.scale_block_shape == ue8m0.scale_block_shape == (128, 128)
    assert fp32_scaled.scale_bits == 32 and ue8m0.scale_bits == 8
    assert (fp32_scaled.memory_bytes_for_shape(BODY_WO_A_SHAPE)
            > ue8m0.memory_bytes_for_shape(BODY_WO_A_SHAPE))
    assert ue8m0.memory_bytes_for_shape(BODY_WO_A_SHAPE) == BODY_WO_A_BYTES
    assert PASSTHROUGH_SOURCE_REQUIREMENTS["FP8_SOURCE"] != (
        PASSTHROUGH_SOURCE_REQUIREMENTS["FP8_BLOCK_UE8M0_SOURCE"])


# ---------------------------------------------------------------------------
# 2. Pricing: exact by construction, and not by measurement
# ---------------------------------------------------------------------------

def test_synthesized_entry_prices_at_zero_with_passthrough_provenance():
    entry = synthesized_source_passthrough_cost_entry("MXFP4_SOURCE")
    stats = {"h_trace": 12345.0, "n_params": 1 << 20}
    assert entry["cost_source"] == SOURCE_PASSTHROUGH_COST_SOURCE
    assert cost_entry_is_source_passthrough(entry, "MXFP4_SOURCE")
    assert cost_entry_is_exact_by_construction(entry, "MXFP4_SOURCE")
    # Zero even against a large h_trace: the price is not an estimate that
    # happens to be small, it is the identity transform on the reference.
    assert cost_entry_predicted_dloss(
        stats, entry, format_name="MXFP4_SOURCE") == 0.0
    # Neither measured nor weight-only.
    assert not cost_entry_uses_measured_output_mse(
        stats, entry, "MXFP4_SOURCE")
    assert cost_entry_source(stats, entry, "MXFP4_SOURCE") == (
        SOURCE_PASSTHROUGH_COST_SOURCE)
    assert cost_entry_activation_pricing_branch(
        stats, entry, "MXFP4_SOURCE") == BRANCH_SOURCE_PASSTHROUGH


def test_passthrough_provenance_cannot_be_forged_for_another_format():
    """The cost_source string alone proves nothing.

    A hand-written or stale cost table must not be able to claim exactness for
    an activation-quantizing rung by writing the magic string into it — that
    would price a 4-bit W4A4 rung at the DP's global minimum on no evidence,
    the exact defect `cost_entry_prices_unmeasured_activation_at_zero` exists
    to prevent.
    """
    forged = {"cost_source": SOURCE_PASSTHROUGH_COST_SOURCE,
              "predicted_dloss": 0.0}
    assert not cost_entry_is_source_passthrough(forged, "NVFP4_CB_K14")
    assert not cost_entry_is_exact_by_construction(forged, "NVFP4_CB_K14")
    assert not cost_entry_is_source_passthrough(forged, "NOT_A_FORMAT")
    # ...and a passthrough format still needs the provenance to claim it.
    assert not cost_entry_is_source_passthrough(
        {"weight_mse": 0.0}, "MXFP4_SOURCE")


# ---------------------------------------------------------------------------
# 3. Legality: the source-kind gate, in both directions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt,kind,legal", [
    ("MXFP4_SOURCE", "mxfp4", True),
    ("MXFP4_SOURCE", "fp8_ue8m0", False),
    ("MXFP4_SOURCE", "bf16", False),
    ("FP8_BLOCK_UE8M0_SOURCE", "fp8_ue8m0", True),
    ("FP8_BLOCK_UE8M0_SOURCE", "mxfp4", False),
    ("FP8_BLOCK_UE8M0_SOURCE", "fp8", False),
    ("BF16", "bf16", True),
    ("BF16", "mxfp4", False),
])
def test_passthrough_is_legal_exactly_where_the_source_is_that_format(
        fmt, kind, legal):
    verdict = check_format_applicability(
        EXPERT_W13_SHAPE if fmt == "MXFP4_SOURCE" else BODY_WO_A_SHAPE,
        fmt,
        qname=("model.layers.3.mlp.experts.0.gate_proj"
               if fmt == "MXFP4_SOURCE" else "model.layers.3.self_attn.wq_a"),
        source_kind=kind,
        target_profile="nvfp4_cb",
    )
    assert verdict.legal is legal, (fmt, kind, verdict.reason, verdict.detail)
    if not legal:
        assert verdict.reason == "source_dtype_mismatch"


def test_production_profile_allows_mxfp4_on_experts_and_denies_it_elsewhere():
    """Ship-intent: the production nvfp4_cb profile permits the rung."""
    assert check_serving_format(
        "nvfp4_cb", "model.layers.39.mlp.experts.0.gate_proj",
        "MXFP4_SOURCE").legal
    # The aggregated packed super item is still an expert unit. This is the
    # trap the rule's not_regex form exists to avoid: filter_candidates_for_
    # profile re-checks with packed_expert=None after aggregation, and a
    # scope="dense" rule would have stripped the format from the very unit it
    # belongs on.
    assert check_serving_format(
        "nvfp4_cb",
        "model.layers.39.mlp.experts.__packed_serving__.gate_up_proj",
        "MXFP4_SOURCE").legal
    denied = check_serving_format(
        "nvfp4_cb", "model.layers.39.self_attn.wq_a", "MXFP4_SOURCE")
    assert not denied.legal and denied.reason == "runtime_unsupported"


def test_lane_metadata_declares_a_distinct_delegated_native_route():
    lane = serving_lane_route("nvfp4_cb", "MXFP4_SOURCE").as_dict()
    assert lane["lane_id"] == "delegated_native_mxfp4"
    assert lane["rung"] is None
    assert lane["fused_mid_m_backed"] is False
    # NOT a CB contract — that is the whole point of a separate lane.
    assert "cb" not in lane["activation_contract"]
    cb_lane = serving_lane_route("nvfp4_cb", "NVFP4_CB_K14").as_dict()
    assert cb_lane["lane_id"] != lane["lane_id"]
    assert cb_lane["activation_contract"] != lane["activation_contract"]


def test_route_status_follows_the_measurement_not_the_intuition():
    """The measured sm121 verdicts, which came out the opposite way round.

    "The released checkpoint already serves this way, so the route exists"
    is false for the BODY passthrough and true for the EXPERT one — and only
    off the default backend. Pinning both directions here keeps a plausible
    guess from quietly replacing the measurement.
    """
    mxfp4 = SOURCE_PASSTHROUGH_CONTRACTS["MXFP4_SOURCE"]
    body = SOURCE_PASSTHROUGH_CONTRACTS["FP8_BLOCK_UE8M0_SOURCE"]
    assert mxfp4.route_status == ROUTE_STATUS_BACKED
    assert mxfp4.route_backed
    # BACKED only WITH a requirement; a backed route whose requirement is
    # unmet serves no better than a blocked one, so it must be declared.
    assert mxfp4.route_requirement == "vllm --moe-backend marlin"
    assert body.route_status == ROUTE_STATUS_BLOCKED
    assert not body.route_backed
    # Both carry the evidence for their verdict.
    assert mxfp4.route_evidence and body.route_evidence


def test_export_gate_set_is_derived_from_route_status_not_transcribed():
    """The MECHANISM, stated table-agnostically.

    Deliberately separate from the verdict pin above. When the serving side
    unblocks a route the verdict pin SHOULD fail — someone must consciously
    update it and its evidence — but this must not, because the rule "anything
    not measurably backed gates the export" does not change when a single
    format's verdict does. Asserting today's membership here instead would
    make one edit look like two failures and invite fixing it by transcribing
    the new answer.
    """
    assert ROUTE_PENDING_PASSTHROUGH_FORMATS == {
        name for name, contract in SOURCE_PASSTHROUGH_CONTRACTS.items()
        if contract.route_status != ROUTE_STATUS_BACKED
    }
    # ...and `backed` is the ONLY status that ships without an override, so a
    # future third non-backed status inherits the gate rather than escaping it.
    for name, contract in SOURCE_PASSTHROUGH_CONTRACTS.items():
        gated = name in ROUTE_PENDING_PASSTHROUGH_FORMATS
        assert gated is (not contract.route_backed), name


def test_wire_format_ids_are_the_closed_cross_repo_enum():
    """Registry names are ours; wire ids are a contract with the loader."""
    assert PASSTHROUGH_WIRE_FORMAT_IDS == {
        "MXFP4_SOURCE": "mxfp4_e2m1_ue8m0_g32",
        "FP8_BLOCK_UE8M0_SOURCE": "fp8_e4m3_ue8m0_block128",
    }
    # Every format the allocator can SYNTHESIZE must be shippable, i.e. must
    # have a wire id — otherwise the DP can select something the artifact
    # cannot declare.
    for name in SOURCE_PASSTHROUGH_FORMATS:
        assert name in PASSTHROUGH_WIRE_FORMAT_IDS, name


def test_serving_notes_carry_requirement_and_evidence():
    notes = passthrough_serving_notes()
    assert notes["MXFP4_SOURCE"]["requirement"] == "vllm --moe-backend marlin"
    assert notes["MXFP4_SOURCE"]["route_status"] == ROUTE_STATUS_BACKED
    assert notes["FP8_BLOCK_UE8M0_SOURCE"]["route_status"] == (
        ROUTE_STATUS_BLOCKED)
    for entry in notes.values():
        assert entry["evidence"]


# ---------------------------------------------------------------------------
# 4. Allocation integration
# ---------------------------------------------------------------------------

def _expert_tables(n_layers=2, n_experts=2):
    """A miniature DSv4-shaped MoE: per-expert 2-D Linears, mxfp4 source.

    Layer 1 is made far more sensitive than layer 0 (h_trace 1000x), so a DP
    that respects cost must protect it.
    """
    stats, costs, manifest = {}, {}, {}
    for layer in range(n_layers):
        for expert in range(n_experts):
            for proj, shape in (("gate_proj", EXPERT_W13_SHAPE),
                                ("up_proj", EXPERT_W13_SHAPE),
                                ("down_proj", EXPERT_W2_SHAPE)):
                name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
                stats[name] = {
                    "h_trace": 1.0 if layer == 0 else 1000.0,
                    "n_params": shape[0] * shape[1],
                    "out_features": shape[0], "in_features": shape[1],
                }
                costs[name] = {
                    "NVFP4_CB_K14": {"weight_mse": 1e-3, "output_mse": 2e-3,
                                     "output_mse_measured": True},
                    "FP8_CB_K36": {"weight_mse": 1e-5, "output_mse": 2e-5,
                                   "output_mse_measured": True},
                }
                manifest[name] = "mxfp4"
    return stats, costs, manifest


def test_allocator_synthesizes_a_zero_cost_passthrough_candidate():
    """The cost table has NO MXFP4_SOURCE column and never will."""
    stats, costs, manifest = _expert_tables()
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    ctx = CBSerializationContext(
        scale_coding="two_tier", codebook_source="lattice",
        scale_sweep=True, encode_tier="balanced")
    specs = [fr.get_format(f) for f in
             ("NVFP4_CB_K14", "FP8_CB_K36", "MXFP4_SOURCE")]
    cands = build_candidates(
        stats, costs, specs, source_manifest=manifest,
        target_profile="nvfp4_cb", cb_serialization_context=ctx)
    assert cands, "no candidates built"
    for name, rows in cands.items():
        by_fmt = {c.fmt: c for c in rows}
        assert "MXFP4_SOURCE" in by_fmt, name
        passthrough = by_fmt["MXFP4_SOURCE"]
        assert passthrough.predicted_dloss == 0.0
        assert passthrough.activation_pricing == BRANCH_SOURCE_PASSTHROUGH
        assert passthrough.serving_lane.lane_id == "delegated_native_mxfp4"
        # Exact source bytes for the unit, weight + E8M0 scale plane.
        expected = (EXPERT_W2_BYTES if name.endswith("down_proj")
                    else EXPERT_W13_BYTES)
        assert passthrough.memory_bytes == expected
        # ...and it strictly dominates the costlier lossy rung.
        k36 = by_fmt["FP8_CB_K36"]
        assert passthrough.memory_bytes < k36.memory_bytes
        assert passthrough.predicted_dloss < k36.predicted_dloss


def test_passthrough_is_not_offered_where_the_source_is_something_else():
    stats, costs, manifest = _expert_tables()
    manifest = {name: "bf16" for name in manifest}
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    ctx = CBSerializationContext(
        scale_coding="two_tier", codebook_source="lattice",
        scale_sweep=True, encode_tier="balanced")
    specs = [fr.get_format(f) for f in ("NVFP4_CB_K14", "MXFP4_SOURCE")]
    masks: list[dict] = []
    cands = build_candidates(
        stats, costs, specs, source_manifest=manifest,
        target_profile="nvfp4_cb", cb_serialization_context=ctx,
        mask_records=masks)
    for rows in cands.values():
        assert "MXFP4_SOURCE" not in {c.fmt for c in rows}
    assert masks and all(
        m["reason"] == "source_dtype_mismatch"
        for m in masks if m["format"] == "MXFP4_SOURCE")


def test_selection_records_the_delegated_native_lane():
    """A shipped selection must name the lane every unit rides."""
    assignment = {
        "model.layers.0.mlp.experts.0.gate_proj": "NVFP4_CB_K14",
        "model.layers.1.mlp.experts.0.gate_proj": "MXFP4_SOURCE",
    }
    prov = selection_serving_lane_provenance(
        assignment, None, "nvfp4_cb")
    assert prov["units_total"] == 2
    assert "MXFP4_SOURCE" in prov["by_format"]
    route = prov["by_format"]["MXFP4_SOURCE"]["route"]
    assert route["lane_id"] == "delegated_native_mxfp4"
    # The passthrough unit is NOT counted under a CB activation contract.
    assert set(prov["activation_contracts"]) == {
        route["activation_contract"],
        prov["by_format"]["NVFP4_CB_K14"]["route"]["activation_contract"],
    }
