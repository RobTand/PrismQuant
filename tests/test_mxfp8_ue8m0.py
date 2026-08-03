"""MXFP8_UE8M0_G32 — the saturating-ceil MX-FP8 rung.

Six things are pinned here, in the order they matter:

  1. the registry declaration and its byte accounting (exactly 8.25 bpw);
  2. the encoder itself — the ceil rule, subnormals, zeros, the 448 rail and
     round-to-nearest-even;
  3. THE EXACTNESS PROPERTY: a block-scaled FP8 source re-encodes with zero
     WEIGHT error, which is the whole reason this format exists separately
     from MXFP8_E4M3 — together with the reason that does NOT make the layer
     output exact, since the lane is W8A8;
  4. menu legality — a re-quantization rung, not a source-gated passthrough,
     and one whose unmeasured A side must leave the menu rather than be
     priced at the DP's global minimum;
  5. the wire contract: id, config group, and the emitted tensor pair;
  6. cost-measurement wiring — no codebook machinery, and the batched render
     agreeing with the unbatched one.
"""
import math
from pathlib import Path

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant import serving_profiles as sp
from prismaquant.allocator_candidates import (
    PASSTHROUGH_WIRE_FORMAT_IDS,
    REQUANT_WIRE_FORMAT_IDS,
    SOURCE_PASSTHROUGH_CONTRACTS,
    WIRE_FORMAT_IDS,
    check_format_applicability,
)
from prismaquant.layer_config import canonicalize_format
from prismaquant.mx_formats import (
    E4M3_MAX,
    E8M0_BIAS,
    E8M0_MIN_EXP,
    mxfp8_ue8m0_qdq,
    mxfp8_ue8m0_shared_exponent,
)

FMT = "MXFP8_UE8M0_G32"
GROUP = 32
WIRE_ID = "mxfp8_e4m3_e8m0_g32"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _finite_e4m3_codes() -> torch.Tensor:
    """Every finite E4M3 value, as float32. 254 of them (two NaN codes)."""
    codes = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)
    vals = codes.float()
    return vals[torch.isfinite(vals)]


def _block128_source(n=256, k=256, block=128, seed=0):
    """A tensor in the DeepSeek ``fp8_e4m3_ue8m0_block128`` body convention.

    E4M3 codes times ONE shared power of two per [128, 128] block — which is
    what that checkpoint stores — dequantized to float32. Built from raw code
    bytes rather than by rounding random floats so every element really is on
    the E4M3 grid, which is the precondition the exactness claim is about.
    """
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(
        0, 256, (n, k), dtype=torch.uint8, generator=g
    ).view(torch.float8_e4m3fn).float()
    codes = torch.nan_to_num(codes, nan=0.0)
    block_exp = torch.randint(
        -30, 31, (n // block, k // block), generator=g
    )
    scale = torch.pow(2.0, block_exp.float())
    dense = scale.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return codes * dense, block_exp


def _exponents(result) -> torch.Tensor:
    return result.scale.view(torch.uint8).to(torch.int32) - E8M0_BIAS


# ---------------------------------------------------------------------------
# 1. registry declaration + byte accounting
# ---------------------------------------------------------------------------

def test_the_registry_declares_the_mx_group32_e4m3_contract():
    spec = fr.get_format(FMT)
    assert spec.weight_bits == 8
    assert spec.group_size == GROUP
    assert spec.scale_bits == 8
    assert spec.scale_dtype_name == "uint8_e8m0"
    assert spec.weight_element_dtype == "fp8_e4m3"
    assert spec.family == "mx"
    assert spec.scale_block_shape is None
    # W8A8, matching the Gridbook lane: activations are quantized dynamically
    # to the SAME per-32 E8M0 grid. This is what makes the cost stage measure
    # a real A-side error, and it is why the exactness claim below is about
    # the WEIGHT plane rather than the layer output.
    assert spec.act_bits == 8
    assert spec.act_dtype_name == "fp8_e4m3"
    assert spec.act_group_size == 32
    assert spec.act_quant_changes_input is True
    cfg = spec.autoround_config()
    assert cfg["act_bits"] == 8 and cfg["act_group_size"] == 32
    assert cfg["act_dynamic"] is True and cfg["act_data_type"] == "mx_fp"
    assert spec.autoround_config()["sym"] is True


def test_the_bits_per_weight_is_exactly_eight_and_a_quarter():
    """8 element bits + one 8-bit scale per 32 = 8.25, with no rounding slack."""
    spec = fr.get_format(FMT)
    assert spec.effective_bits == 8.25

    for shape in [(2048, 4096), (128, 128), (5120, 1536), (7168, 2048)]:
        n_params = math.prod(shape)
        expected_bytes = n_params + n_params // GROUP
        assert spec.memory_bytes_for_shape(shape) == expected_bytes
        assert spec.effective_bits_for_shape(shape) == 8.25
        assert spec.scale_count_for_shape(shape) == n_params // GROUP


def test_it_is_dearer_than_the_block128_source_it_can_absorb():
    """The scale plane is the whole difference, and it is 32x denser."""
    exact = fr.get_format(FMT).effective_bits_for_shape((2048, 4096))
    block = fr.get_format(
        "FP8_BLOCK_UE8M0_SOURCE").effective_bits_for_shape((2048, 4096))
    assert exact == 8.25
    assert block < exact
    # 8 bits per 32 weights vs 8 bits per 128x128 block.
    assert exact - block == pytest.approx(8 / 32 - 8 / (128 * 128))


# ---------------------------------------------------------------------------
# 2. the encoder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amax,expected", [
    (448.0, 0),      # the rail itself: 448 / 2^0 == 448, exactly admissible
    (449.0, 1),      # one ulp over forces the next exponent up
    (256.0, 0),
    (240.0, 0),      # 240 / 2^-1 = 480 > 448, so -1 is NOT admissible
    (224.0, -1),     # 224 / 2^-1 = 448, exactly admissible
    (1.0, -8),
    (2.0 ** -9, -17),
])
def test_the_shared_exponent_is_the_smallest_that_does_not_clip(amax, expected):
    got = int(mxfp8_ue8m0_shared_exponent(torch.tensor([amax]))[0])
    assert got == expected
    # the defining property, restated as the two inequalities it is made of
    assert amax / 2.0 ** got <= E4M3_MAX
    assert amax / 2.0 ** (got - 1) > E4M3_MAX


def test_the_ceil_rule_never_lets_an_element_reach_the_e4m3_rail():
    g = torch.Generator().manual_seed(3)
    for power in (-140, -60, -8, 0, 8, 60, 118):
        x = torch.randn(16, GROUP, generator=g) * (2.0 ** power)
        e = mxfp8_ue8m0_shared_exponent(x.abs().amax(dim=-1))
        assert bool((torch.ldexp(x, (-e).unsqueeze(-1)).abs() <= E4M3_MAX).all())


def test_the_exponent_is_computed_exactly_not_through_log2():
    """Guards the frexp implementation against a 'simplifying' rewrite.

    ``ceil(log2(amax / 448))`` is the obvious spelling and it is wrong on a
    small set of float32 inputs: when ``amax / 448`` rounds to exactly a power
    of two, ``log2`` lands on an integer, ``ceil`` returns it unchanged, and
    the resulting exponent is one too small — so the group's own maximum
    saturates at 448, which is the one thing the rule exists to prevent.

    This is not hypothetical; it is where the Gridbook consumer's reference
    quantizer currently sits, so the divergence is recorded rather than
    guessed at.
    """
    def via_log2(amax):
        return torch.clamp(
            torch.ceil(torch.log2(amax / E4M3_MAX)), min=-127.0, max=127.0,
        ).to(torch.int32)

    # A value where the two disagree, found by sweep and pinned by hex so it
    # cannot drift: amax / 448 rounds to exactly 2**81 in float32.
    amax = torch.tensor([0x6C600004], dtype=torch.int32).view(torch.float32)
    exact = int(mxfp8_ue8m0_shared_exponent(amax)[0])
    naive = int(via_log2(amax)[0])
    assert exact == naive + 1
    # float64 arbiter: only the exact rule keeps the group inside the grid.
    scaled_exact = amax.double() / 2.0 ** exact
    scaled_naive = amax.double() / 2.0 ** naive
    assert float(scaled_exact) <= E4M3_MAX
    assert float(scaled_naive) > E4M3_MAX

    # And the rule holds over a broad sweep, which the log2 form does not.
    g = torch.Generator().manual_seed(23)
    probe = (torch.rand(200_000, generator=g)
             * torch.pow(2.0, torch.randint(-120, 120, (200_000,),
                                            generator=g).float()))
    probe = probe[probe > 0]
    e = mxfp8_ue8m0_shared_exponent(probe)
    assert bool((probe.double() / torch.pow(torch.tensor(2.0, dtype=torch.float64),
                                            e.double()) <= E4M3_MAX).all())


def test_zero_groups_pin_to_the_bottom_of_the_e8m0_range():
    """A zero group constrains nothing; it must still be deterministic."""
    result = mxfp8_ue8m0_qdq(torch.zeros(4, GROUP))
    assert bool((result.dequant == 0).all())
    assert result.scale.dtype is torch.float8_e8m0fnu
    assert set(_exponents(result).flatten().tolist()) == {E8M0_MIN_EXP}


def test_subnormal_elements_survive_the_round_trip():
    """E4M3 subnormals are exact when the group max lets them be."""
    x = torch.zeros(1, GROUP)
    x[0, 0] = 2.0 ** -9         # min positive subnormal
    x[0, 1] = 3 * 2.0 ** -9
    x[0, 2] = -(5 * 2.0 ** -9)
    x[0, 3] = 2.0 ** -6         # min normal
    result = mxfp8_ue8m0_qdq(x)
    assert torch.equal(result.dequant, x)


def test_the_element_cast_rounds_ties_to_even():
    """400 sits exactly between E4M3 codes 384 (even) and 416 (odd)."""
    x = torch.zeros(1, GROUP)
    x[0, 0] = 448.0             # pins the group exponent to 0
    x[0, 1] = 400.0
    assert float(mxfp8_ue8m0_qdq(x).dequant[0, 1]) == 384.0


def test_an_out_of_range_group_clamps_instead_of_emitting_a_nan_code():
    """The E8M0 rail is the only way to overflow, and it must clamp."""
    g = torch.Generator().manual_seed(9)
    for power in range(-140, 121, 10):
        result = mxfp8_ue8m0_qdq(torch.randn(4, GROUP, generator=g)
                                 * (2.0 ** power))
        raw = result.scale.view(torch.uint8)
        # 255 is the E8M0 NaN code; emitting it would poison the group.
        assert int(raw.max()) <= 254
        assert not bool(torch.isnan(result.quant.float()).any())


def test_a_ragged_last_group_is_zero_padded_not_rejected():
    """Zeros cannot move a max-abs, so the pad is exact for real columns."""
    x = torch.randn(4, GROUP + 5)
    result = mxfp8_ue8m0_qdq(x)
    assert result.dequant.shape == x.shape
    assert result.scale.shape == (4, 2)
    head = mxfp8_ue8m0_qdq(x[:, :GROUP])
    assert torch.equal(result.dequant[:, :GROUP], head.dequant)


def test_the_scale_plane_is_the_native_e8m0_float_dtype():
    result = mxfp8_ue8m0_qdq(torch.randn(8, 4 * GROUP))
    assert result.scale.dtype is torch.float8_e8m0fnu
    assert result.scale.shape == (8, 4)
    assert result.quant.dtype is torch.float8_e4m3fn
    # The decoded scale really is the power of two the exponent names.
    decoded = result.scale.float()
    assert torch.equal(decoded, torch.ldexp(
        torch.ones_like(decoded), _exponents(result)))


def test_the_dequant_is_the_quantized_elements_times_the_decoded_scale():
    """No hidden third quantity: what ships is what was priced."""
    result = mxfp8_ue8m0_qdq(torch.randn(16, 8 * GROUP) * 2.5)
    rebuilt = (
        result.quant.float().reshape(16, 8, GROUP)
        * result.scale.float().unsqueeze(-1)
    ).reshape(16, 8 * GROUP)
    assert torch.equal(rebuilt, result.dequant)


# ---------------------------------------------------------------------------
# 3. THE EXACTNESS PROPERTY
# ---------------------------------------------------------------------------

def test_a_block128_fp8_source_re_encodes_with_zero_weight_error():
    """The property the format exists for: weight_mse is EXACTLY 0.0.

    Any tensor whose stored form is E4M3 codes times a shared power of two —
    the DeepSeek ``fp8_e4m3_ue8m0_block128`` body convention — is exactly
    representable here, because the ceil rule can never pick a group exponent
    above the block's own, so every element is an E4M3 code scaled DOWN by a
    power of two. 128 = 4*32, so a chunk never straddles a block boundary.

    Stated on the WEIGHT plane deliberately. The served lane is W8A8, so a
    weight-lossless unit still has real activation-side error; see
    ``test_zero_weight_error_does_not_make_the_layer_output_exact``.
    """
    w, _ = _block128_source()
    result = mxfp8_ue8m0_qdq(w)
    assert torch.equal(result.dequant, w)
    weight_mse = float(((result.dequant - w) ** 2).mean())
    assert weight_mse == 0.0

    # and the same through the registry closure the cost stage actually calls
    rendered = fr.get_format(FMT).quantize_dequantize(w)
    assert torch.equal(rendered, w)
    assert float(((rendered - w) ** 2).mean()) == 0.0


def test_zero_weight_error_does_not_make_the_layer_output_exact():
    """W' == W silences the WEIGHT side only; the A side is still 8-bit.

    This is the honesty the allocator gates depend on: a block-FP8 source on
    this rung must NOT look free.
    """
    from prismaquant.allocator_candidates import cost_entry_is_bit_exact

    spec = fr.get_format(FMT)
    w, _ = _block128_source(n=128, k=128, block=64)
    torch.manual_seed(4)
    x = torch.randn(64, 128)

    w_hat = spec.quantize_dequantize(w.clone())
    x_hat = spec.activation_quantize_dequantize(x.clone())
    assert torch.equal(w_hat, w)                      # weight side: exact
    assert not torch.equal(x_hat, x)                  # A side: not identity

    output_mse = float((((x @ w.T) - (x_hat @ w_hat.T)) ** 2).mean())
    assert output_mse > 0.0

    # ... and the allocator refuses to short-circuit such an entry to dloss 0.
    assert not cost_entry_is_bit_exact({"weight_mse": 0.0}, FMT)


def test_the_activation_path_is_the_same_dynamic_per_32_ceil_rule():
    """The A side must be the lane's A side, not a stand-in."""
    spec = fr.get_format(FMT)
    torch.manual_seed(7)
    x = torch.randn(16, 4 * GROUP) * 3.0
    assert torch.equal(
        spec.activation_quantize_dequantize(x.clone()),
        mxfp8_ue8m0_qdq(x).dequant,
    )
    # Dynamic: the scale follows the batch, so a rescaled input reuses the
    # same relative grid rather than a fitted static one.
    assert torch.equal(
        spec.activation_quantize_dequantize(x * 4.0),
        spec.activation_quantize_dequantize(x) * 4.0,
    )


def test_group_exponents_stay_at_or_below_their_block_exponent():
    """RTN may pick a SMALLER exponent for a low-magnitude group.

    That is still exact, so the assertion is an inequality plus reconstruction
    equality — not exponent equality, which would be a false requirement.
    """
    w, block_exp = _block128_source()
    result = mxfp8_ue8m0_qdq(w)
    chosen = _exponents(result)
    per_group = block_exp.repeat_interleave(128, 0).repeat_interleave(
        128 // GROUP, 1)
    assert chosen.shape == per_group.shape
    assert bool((chosen <= per_group).all())
    # the interesting half of the claim: it really does go strictly lower
    assert bool((chosen < per_group).any())
    assert torch.equal(result.dequant, w)


def test_exactness_holds_for_every_e4m3_code_as_the_group_maximum():
    """Exhaustive over the element grid, not a sample of it.

    Each finite E4M3 code is driven as the group max, alongside the values
    most likely to fall off the grid when a group is rescaled (the min normal
    and the min subnormal), across the block exponents a real checkpoint uses.
    """
    codes = _finite_e4m3_codes()
    n = codes.numel()
    for block_exp in range(-24, 25, 4):
        scale = 2.0 ** block_exp
        # (a) each code as the max, with the fragile small values beside it
        probe = torch.zeros(n, GROUP)
        probe[:, 0] = codes
        probe[:, 1] = 1.125 * 2.0 ** -6
        probe[:, 2] = 2.0 ** -9
        probe[:, 3] = -codes
        probe = probe * scale
        assert torch.equal(mxfp8_ue8m0_qdq(probe).dequant, probe)

        # (b) the full code set swept through the group, max pinned at 448
        paired = torch.zeros(n, GROUP)
        paired[:, 0] = 448.0
        paired[:, 1] = codes
        paired = paired * scale
        assert torch.equal(mxfp8_ue8m0_qdq(paired).dequant, paired)


def test_the_stock_mxfp8_encoder_is_the_one_that_cannot_do_this():
    """Guards the reason two MX-FP8 formats exist, so it cannot be forgotten.

    The compressed-tensors rule rounds the group amax to a power of two, which
    can scale a group UP; a group holding both 448 and a small normal then
    loses the small one off the E4M3 subnormal ladder. If this ever starts
    passing, MXFP8_E4M3 has changed and the two formats should be reconsidered.
    """
    from prismaquant.mx_formats import mxfp8_e4m3_qdq

    x = torch.zeros(1, GROUP)
    x[0, 0] = 448.0
    x[0, 1] = 1.125 * 2.0 ** -6
    assert not torch.equal(mxfp8_e4m3_qdq(x).dequant, x)
    assert torch.equal(mxfp8_ue8m0_qdq(x).dequant, x)


# ---------------------------------------------------------------------------
# 4. menu legality
# ---------------------------------------------------------------------------

def _menu_tables(cost_entry):
    """One dense Linear with a real sensitivity, priced by ``cost_entry``."""
    name = "model.layers.0.self_attn.q_proj"
    stats = {name: {"h_trace": 2.0, "n_params": 512 * 1024,
                    "in_features": 1024, "out_features": 512}}
    costs = {name: {FMT: dict(cost_entry),
                    "BF16": {"weight_mse": 0.0, "output_mse": 0.0,
                             "output_mse_measured": True}}}
    return name, stats, costs


def test_an_unmeasured_activation_side_is_masked_not_admitted():
    """A W8A8 rung with no measured output_mse must LEAVE the menu.

    On a block-FP8 source the weight error really is 0.0, so an entry lacking
    activation evidence would otherwise be priced at the DP's global minimum
    on no evidence at all — the unbeatable argmin at every budget, for an
    assignment whose served activations are still 8-bit.
    """
    from prismaquant.allocator_candidates import (
        ACTIVATION_COST_UNMEASURED_REASON,
        build_candidates,
        cost_entry_prices_unmeasured_activation_at_zero,
    )

    # The bare predicate: weight-lossless, no measured output side, priced 0.
    assert cost_entry_prices_unmeasured_activation_at_zero(
        {"h_trace": 2.0}, {"weight_mse": 0.0}, 0.0, FMT)

    name, stats, costs = _menu_tables(
        {"weight_mse": 0.0, "output_mse": 0.0, "output_mse_measured": False})
    masks: list[dict] = []
    cands = build_candidates(
        stats, costs,
        [fr.get_format(FMT), fr.get_format("BF16")],
        target_profile="nvfp4_cb", mask_records=masks)
    assert FMT not in {c.fmt for c in cands[name]}
    assert any(m["format"] == FMT
               and m["reason"] == ACTIVATION_COST_UNMEASURED_REASON
               for m in masks), masks


def test_a_measured_activation_side_passes_menu_admission():
    """The same row WITH real W8A8 evidence is admitted and priced above zero."""
    from prismaquant.allocator_candidates import build_candidates

    name, stats, costs = _menu_tables(
        {"weight_mse": 0.0, "output_mse": 3.5e-4,
         "output_mse_measured": True})
    cands = build_candidates(
        stats, costs,
        [fr.get_format(FMT), fr.get_format("BF16")],
        target_profile="nvfp4_cb")
    chosen = {c.fmt: c for c in cands[name]}
    assert FMT in chosen
    assert chosen[FMT].predicted_dloss > 0.0
    assert chosen[FMT].bits_per_param == 8.25
    assert chosen[FMT].serving_lane.lane_id == "mxfp8_ue8m0_g32"
    assert chosen[FMT].serving_lane.fused_mid_m_backed is False
    # Priced from the real A-side measurement, not short-circuited: the
    # weight side is exactly lossless here and would otherwise read as free.
    assert chosen[FMT].activation_pricing == "measured_output_mse"
    assert chosen["BF16"].activation_pricing == "bit_exact"


def test_it_is_a_requantization_rung_not_a_source_gated_passthrough():
    """Legal on ANY source dtype — the whole difference from the *_SOURCE family."""
    assert FMT not in SOURCE_PASSTHROUGH_CONTRACTS
    spec = fr.get_format(FMT)
    # It has a real encoder, which is precisely what disqualifies it from the
    # passthrough table (whose members must be the identity).
    probe = torch.randn(8, 4 * GROUP)
    assert not torch.equal(spec.quantize_dequantize(probe), probe)

    shape = (256, 256)
    for source_kind in ("bf16", "fp8", "fp8_ue8m0", "mxfp4", "other", None):
        verdict = check_format_applicability(
            shape, FMT, source_kind=source_kind,
            target_profile="nvfp4_cb",
            qname="model.layers.0.self_attn.q_proj")
        assert verdict.legal, (source_kind, verdict.reason, verdict.detail)


def test_the_body_and_attention_menus_carry_it_and_packed_experts_do_not():
    dense = "model.layers.0.self_attn.q_proj"
    expert = "model.layers.0.mlp.experts.gate_up_proj"

    assert sp.check_serving_format("nvfp4_cb", dense, FMT).legal
    assert sp.check_serving_format(
        "nvfp4_cb", "model.layers.0.mlp.down_proj", FMT).legal

    denied = sp.check_serving_format(
        "nvfp4_cb", expert, FMT, packed_expert=True)
    assert not denied.legal
    # Denied by the EXISTING container rule for stock-delegated dense-only
    # emission, not by a format-specific special case.
    assert denied.rule == "cb_packed_expert_stock_ct_unsupported"
    assert denied.reason == "runtime_unsupported"


def test_the_expert_cost_menu_is_enumerated_and_excludes_it_by_bpw():
    """The <= 4.25 bpw expert menu is an ENUMERATION, not a numeric cap.

    There is no bpw-cap predicate anywhere in the codebase; the routed-expert
    ladder is the explicit MENU list in the cost driver. This pins the
    invariant that makes 'MXFP8 is naturally excluded' true: every rung that
    menu measures is at or under 4.25 bpw, so an 8.25 bpw rung cannot be on it.
    """
    driver = Path(__file__).resolve().parents[1] / (
        "scripts/run_dsv4_mxfp4_cost.sh")
    menu_line = next(
        line for line in driver.read_text().splitlines()
        if line.startswith("MENU=")
    )
    menu = menu_line.split("{MENU:-", 1)[1].rstrip('}"').split(",")
    assert len(menu) > 1
    for name in menu:
        assert fr.get_format(name).effective_bits <= 4.25, name
    assert FMT not in menu
    assert fr.get_format(FMT).effective_bits > 4.25


def test_it_is_masked_where_the_group_does_not_divide_the_input():
    verdict = check_format_applicability(
        (256, 100), FMT, source_kind="bf16",
        target_profile="nvfp4_cb",
        qname="model.layers.0.self_attn.q_proj")
    assert not verdict.legal
    assert verdict.reason == "group_divisibility"


def test_the_recipe_spellings_round_trip():
    assert canonicalize_format(FMT) == FMT
    assert canonicalize_format("mxfp8_ue8m0_g32") == FMT
    assert canonicalize_format({
        "data_type": "fp8_e4m3", "bits": 8,
        "group_size": 32, "scale_fmt": "ue8m0",
    }) == FMT
    # Without the UE8M0 scale_fmt this stays the stock compressed-tensors rung.
    assert canonicalize_format({
        "data_type": "fp8_e4m3", "bits": 8, "group_size": 32,
    }) == "MXFP8_E4M3"
    # And the historical short alias is untouched.
    assert canonicalize_format("mxfp8") == "MXFP8_E4M3"
    assert fr.canonical_format_name("MXFP8") == "MXFP8_E4M3"


# ---------------------------------------------------------------------------
# 5. the wire contract
# ---------------------------------------------------------------------------

def test_the_wire_id_is_pinned_and_globally_unique():
    assert REQUANT_WIRE_FORMAT_IDS == {FMT: WIRE_ID}
    # Re-quant ids are a SEPARATE table from the passthrough ids: membership of
    # the latter is a claim of byte-verbatim shipping this rung cannot make.
    assert FMT not in PASSTHROUGH_WIRE_FORMAT_IDS
    assert WIRE_ID not in PASSTHROUGH_WIRE_FORMAT_IDS.values()
    # But the consumer dispatches on the id alone, so the union must be 1:1.
    assert WIRE_FORMAT_IDS == {**PASSTHROUGH_WIRE_FORMAT_IDS,
                               **REQUANT_WIRE_FORMAT_IDS}
    assert len(set(WIRE_FORMAT_IDS.values())) == len(WIRE_FORMAT_IDS)
    for name in WIRE_FORMAT_IDS:
        fr.get_format(name)          # every id names a registered format


def test_the_serving_lane_is_declared_and_honestly_unbacked():
    lane = sp.serving_lane_route("nvfp4_cb", FMT)
    assert lane is not None
    assert lane.lane_id == "mxfp8_ue8m0_g32"
    assert lane.rung is None
    assert lane.fused_mid_m_backed is False
    assert lane.fused_mid_m_rungs == ()
    # An ABSENT fused_mid_m key, not an empty declared one: there is no decode
    # prologue to fuse, and this is the fail-closed spelling.
    assert lane.rungs_source == "lane_declares_no_fused_mid_m_lane"
    # Not filed under any CB activation contract.
    assert "cb" not in lane.activation_contract
    assert lane.activation_contract == "w8a8-dynamic-mxfp8-e4m3-group32-ue8m0"


def test_the_lane_only_becomes_backed_when_a_runtime_version_declares_it():
    """rungs_by_runtime_version precedent: a pin with no key backs nothing."""
    spec = sp.load_serving_profile("nvfp4_cb")
    lane = next(l for l in spec.serving_lanes if l.id == "mxfp8_ue8m0_g32")
    assert lane.fused_mid_m_rungs_by_runtime_version == ()
    for version in ("0.7.0", "0.8.0", ""):
        rungs, source = lane.backed_rungs(version)
        assert rungs == ()
        assert source == "lane_declares_no_fused_mid_m_lane"


def test_the_config_group_declares_the_w8a8_activation_contract():
    from prismaquant.cb_export_config import (
        REQUANT_NATIVE_GROUP_FORMAT,
        requant_native_config_group,
        requant_wire_id,
    )

    assert requant_wire_id(FMT) == WIRE_ID
    assert requant_wire_id("mxfp8_ue8m0_g32") == WIRE_ID
    with pytest.raises(ValueError, match="not a declared re-quantization"):
        requant_wire_id("MXFP8_E4M3")

    group = requant_native_config_group(FMT)
    assert group["format"] == REQUANT_NATIVE_GROUP_FORMAT
    assert group["source_format"] == FMT
    assert group["wire_format_id"] == WIRE_ID
    acts = group["input_activations"]
    assert acts is not None, "a W8A8 lane must not declare a weight-only group"
    assert acts["num_bits"] == 8
    assert acts["element_dtype"] == "fp8_e4m3"
    assert acts["scale_dtype"] == "uint8_e8m0"
    assert acts["dynamic"] is True
    assert acts["strategy"] == "group" and acts["group_size"] == GROUP
    weights = group["weights"]
    assert weights["num_bits"] == 8
    assert weights["element_dtype"] == "fp8_e4m3"
    assert weights["scale_dtype"] == "uint8_e8m0"
    assert weights["strategy"] == "group"
    assert weights["group_size"] == GROUP
    # The producer WROTE these bytes; saying otherwise would let the consumer
    # believe they came from the checkpoint unchanged.
    assert weights["source_passthrough"] is False


def test_the_exporter_plans_a_weight_and_scale_pair_at_the_declared_dtypes():
    from prismaquant.export_nvfp4_cb_streaming import (
        _requant_output_specs,
        _requant_pack,
    )

    specs = _requant_output_specs(FMT, (256, 512))
    assert specs == [
        ("weight", torch.float8_e4m3fn, (256, 512)),
        ("weight_scale", torch.float8_e8m0fnu, (256, 512 // GROUP)),
    ]
    # The planned header must match what the packer actually produces, or the
    # streaming writer sizes the file wrong.
    packed = _requant_pack(FMT, torch.randn(256, 512))
    assert sorted(packed) == ["weight", "weight_scale"]
    for suffix, dtype, shape in specs:
        assert packed[suffix].dtype is dtype
        assert tuple(packed[suffix].shape) == shape

    with pytest.raises(ValueError, match="not a multiple"):
        _requant_output_specs(FMT, (256, 100))


def test_the_exporter_packer_is_the_same_codec_the_registry_prices():
    """Emulation and shipped bytes are one rendering, not two that agree."""
    from prismaquant.export_nvfp4_cb_streaming import _requant_pack

    w = torch.randn(64, 4 * GROUP) * 1.7
    packed = _requant_pack(FMT, w)
    rebuilt = (
        packed["weight"].float().reshape(64, 4, GROUP)
        * packed["weight_scale"].float().unsqueeze(-1)
    ).reshape(64, 4 * GROUP)
    assert torch.equal(rebuilt, fr.get_format(FMT).quantize_dequantize(w))


# ---------------------------------------------------------------------------
# 6. cost measurement wiring
# ---------------------------------------------------------------------------

def test_cost_measurement_needs_no_codebook_machinery():
    """A non-CB rung takes the simple render path: no CB env stamps, no context."""
    from prismaquant import measure_quant_cost as mqc

    spec = fr.get_format(FMT)
    assert spec.family not in mqc._CB_COST_FAMILIES
    # cost_payload_provenance binds a CBSerializationContext only for CB menus;
    # a menu of this rung alone must not demand one (it would raise without the
    # CB_* env stamps set).
    provenance = mqc.cost_payload_provenance([spec])
    assert provenance.get("cb_serialization") in (None, {})


def test_the_batched_render_is_the_same_codec_as_the_unbatched_one():
    """The batched path keys on element dtype, which this rung SHARES.

    ``weight_element_dtype`` is "fp8_e4m3" here and on MXFP8_E4M3/FP8_CB, so a
    fall-through would render this rung with the local codebook replica's E8M0
    snap instead of its own saturating-ceil rule — pricing the batched path
    with a different codec than the exporter writes.
    """
    from prismaquant import measure_quant_cost as mqc

    spec = fr.get_format(FMT)
    assert FMT in mqc._EXPORT_ALIGNED_BATCH_FORMATS

    torch.manual_seed(1)
    stacked = torch.randn(3, 32, 4 * GROUP)
    batched = mqc._batched_quantize(spec, stacked)
    unbatched = torch.stack([
        spec.quantize_dequantize(stacked[i].clone()) for i in range(3)])
    assert torch.equal(batched, unbatched)

    # and the property survives the batched path too
    src, _ = _block128_source(n=128, k=128, block=64)
    stacked_src = src.unsqueeze(0).repeat(2, 1, 1)
    assert torch.equal(mqc._batched_quantize(spec, stacked_src), stacked_src)


def test_a_small_layer_prices_through_the_shared_render_helper():
    """Smoke: the rung produces a finite weight cost on a synthetic layer."""
    from prismaquant import measure_quant_cost as mqc

    torch.manual_seed(0)
    w = torch.randn(64, 4 * GROUP, dtype=torch.float32)
    rendered = mqc._render_weight_for_spec(fr.get_format(FMT), w) \
        if hasattr(mqc, "_render_weight_for_spec") \
        else fr.get_format(FMT).quantize_dequantize(w.clone())
    weight_mse = float(((rendered - w) ** 2).mean())
    assert math.isfinite(weight_mse)
    assert weight_mse > 0.0          # generic gaussian input is NOT on the grid

    # ... and exactly zero once the input is a block-FP8 source.
    src, _ = _block128_source(n=64, k=128, block=64)
    assert float(
        ((fr.get_format(FMT).quantize_dequantize(src) - src) ** 2).mean()
    ) == 0.0
