"""Source-passthrough family: pricing, byte exactness, legality, allocation.

The storage claim is narrow and total: for a unit whose stored bytes ALREADY
are format F, shipping its weight plane unchanged occupies exactly the source
slice. A zero end-to-end cost additionally requires an identity activation
path. These tests pin the distinction, the source-kind gate, and the
allocation consequence for terminals that satisfy both halves.

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
    SOURCE_BPP_EXCEEDED_REASON,
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

def test_every_contract_names_a_registered_weight_identity_format():
    """Every contract copies W exactly; only zero-cost contracts copy A too.

    The zero cost is justified only when both the exported source bytes and
    the serving activation path are identities.
    """
    probe = torch.randn(8, 128)
    for name, contract in SOURCE_PASSTHROUGH_CONTRACTS.items():
        spec = fr.get_format(name)
        assert spec.name == name
        assert contract.source_kind
        assert contract.serving_route
        torch.testing.assert_close(spec.quantize_dequantize(probe), probe)
        if contract.zero_cost_by_construction:
            torch.testing.assert_close(
                spec.activation_quantize_dequantize(probe), probe)
            assert not spec.act_quant_changes_input, name

    block = fr.get_format("FP8_BLOCK_UE8M0_SOURCE")
    assert block.act_bits is None
    assert block.act_group_size is None
    assert block.act_dtype_name is None
    assert not block.act_quant_changes_input
    torch.testing.assert_close(block.activation_quantize_dequantize(probe), probe)
    assert block.autoround_config()["act_bits"] == 16
    assert block.autoround_config()["act_data_type"] == "float"


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
# 3. Route status: what each passthrough's bytes actually execute on
# ---------------------------------------------------------------------------
#
# The legality cases that used to live here drove `build_candidates` and
# `selection_serving_lane_provenance` through `target_profile="nvfp4_cb"`.
# That profile went with the Gridbook codebook lane on 2026-09-02
# (archive/gridbook_lane_2026-09-02/), and MXFP4_SOURCE had no other one: it
# is the lane's orphan -- a rung with a real stock-vLLM Marlin route, no
# serving profile that offers it, and no exporter that writes it, since the
# CB container was its only writer. Those tests are deleted rather than
# re-pointed at `vllm_packed_moe`, which denies the rung outright and would
# have turned a real capability loss into a green assertion.

def test_route_status_is_the_verdict_each_route_actually_earned():
    """Two passthroughs, two different verdicts, both from measurements.

    Updated 2026-09-02, when the Gridbook codebook lane was retired
    (archive/gridbook_lane_2026-09-02/). The block-128 UE8M0 body route was
    that plugin's `Fp8SourceW8A16LinearMethod` and nothing else executes those
    bytes, so its verdict flipped to BLOCKED -- by the runtime going away, not
    by an opinion. MXFP4_SOURCE never rode Gridbook: it serves on stock vLLM
    Marlin MoE, so its verdict is untouched. (Its *writer* was the CB
    container and it has none today; that is an exporter fact, not a route
    fact, and this test deliberately does not conflate them.)
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
    # The evidence is KEPT verbatim and stays scoped to the runtime it was
    # measured on: it records what was true, not what is.
    assert "source_fp8_block128_w8a16=1" in body.route_requirement
    assert "e992e5980c96333a48149f96392d6cff56ae9e3f" in body.route_requirement
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
    assert "source_fp8_block128_w8a16=1" in (
        notes["FP8_BLOCK_UE8M0_SOURCE"]["requirement"]
    )
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


