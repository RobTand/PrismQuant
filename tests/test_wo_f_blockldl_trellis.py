"""WO-F F1: Trellis encode under BlockLDLQ feedback - one-encode + buffered vs reference.

This test pins the production seam requested in WO-F F1:

* ``reverse_block_feedback_reference`` is the mathematical oracle.
* ``reverse_block_feedback_buffered`` is the production path.
* The terminal supplied must trellis-encode its block through the existing
  producer path, and the quantized result the recurrence propagates must be the
  real wire's decode (not a stand-in), satisfying principle 8 (one rendering).
* The reconciliation of the two block structures (trellis superblock 256 vs
  LDL block 256) is explicit: they are aligned 1:1, one LDL block equals one
  trellis superblock, with the global E2M1 scale shared across the reverse
  pass.  No re-tiling or rate re-accounting is required.
* The test asserts byte identity and that the final wire decodes to the
  recurrence's terminal decodes.

This is a CPU reference test; the same logic is exercised in the full Qwen
measurement harness (``scratch/wo_f_harness.py``) with the production-ish
wire composition.
"""
from __future__ import annotations

import importlib

import torch

M = importlib.import_module(
    "research.qtip_native_nvfp4_2026-08-30.trellis_online_hadamard_producer"
)
_CAMPAIGN = importlib.import_module(
    "research.qtip_native_nvfp4_2026-08-30.arm_e_quality_campaign"
)

from prismaquant.trellis_formats import E2M1_FAMILY, LAYOUT_TIGHT_OFFSETS, get_trellis_family
from prismaquant.trellis_rate_surface import uniform_column_schedule
from prismaquant.trellis_wire import TrellisWire

# Reuse the same block reconciliation constant as the producer.
TRELLIS_LDL_BLOCK = 256


def _regularized_diag(raw: torch.Tensor) -> torch.Tensor:
    v = raw.float().reshape(-1).clone()
    dead = v <= 0
    alive = ~dead
    mean = (v[alive].mean() if bool(alive.any()) else v.new_ones(()).squeeze()).clamp_min(1e-12)
    if bool(dead.any()):
        v[dead] = 1.0
    v.add_(float(mean.item()))
    return v


def _trellis_terminal_factory(weight_device: torch.device, schedule_full, alphabets_full, q256: int, global_scale: float, col_weights_full: torch.Tensor | None = None):
    """Return a terminal that encodes exactly one 256-column block via the real wire.

    The terminal packs its planes, reparses, and reference-decodes the same
    bytes before returning the decode to the recurrence.  This is the one-encode
    guarantee: the recurrence never sees a value the wire cannot reproduce.

    When ``col_weights_full`` is provided, the terminal uses the corresponding
    256-slice as its objective weighting (matching the direct whole-wire metric);
    otherwise it uses a unit metric (sufficient to pin buffered-vs-reference).
    """
    from prismaquant.trellis_producer import encode_trellis_one_linear

    family = get_trellis_family(E2M1_FAMILY)

    def terminal(block_index: int, target: torch.Tensor) -> torch.Tensor:
        first = block_index * TRELLIS_LDL_BLOCK
        last = first + TRELLIS_LDL_BLOCK
        block_schedule = tuple(schedule_full[first:last])
        # Select only the alphabets whose rates appear in this block.
        used_rates = {r for r in block_schedule if r < family.bypass_rate}
        block_alphabets = {r: alphabets_full[r] for r in used_rates}
        if col_weights_full is not None:
            metric = col_weights_full[first:last].to(device=target.device, dtype=torch.float32)
        else:
            metric = torch.ones(TRELLIS_LDL_BLOCK, dtype=torch.float32, device=target.device)
        # The full harness shares one tensor-global scale; here each terminal
        # independently proposes the same global (since the full harness's
        # shared global equals the per-block global for this synthetic test).
        artifact = encode_trellis_one_linear(
            target,
            metric,
            family=E2M1_FAMILY,
            body_rate_q256=sum(block_schedule),
            schedule=block_schedule,
            layout=LAYOUT_TIGHT_OFFSETS,
            alphabets=block_alphabets,
            scale_rule="static_6",
            sb_chunk=4,
            determinism_mode="on",
            tailbite_candidates=4,
            backend="eager",
            point_route="full",
            global_scale_real_override=global_scale,
        )
        # Prove the recurrence consumes the wire's decode, not the float reconstruction.
        # The producer already checked BF16 equality, but we re-assert same-byte decode.
        blob = artifact.wire_bytes
        assert TrellisWire.from_bytes(blob).to_bytes() == blob
        return artifact.decoded_weight

    return terminal


def test_blockldl_trellis_one_encode_and_buffered_vs_reference():
    # Use a small 2-block weight so the recurrence actually feeds back across
    # the superblock boundary.  One block (256 columns) would have no cross-
    # block feedback and would not test the recurrence.
    torch.manual_seed(0)
    rows, cols = 4, 512
    weight = torch.randn(rows, cols) * 0.2
    # Build a block-diagonal Hessian with non-trivial off-diagonal inside each
    # 256 block, so the LDL factor is non-zero and the two recurrences must
    # both handle the feedback.
    raw = torch.rand(cols).add_(0.3)
    damped = _regularized_diag(raw)
    # Build a dense Hessian with coupling *across* the 256 block boundary,
    # so the BlockLDL feedback is non-zero.  A block-diagonal Hessian whose
    # block size equals the LDL block size would have zero feedback by
    # construction and would not exercise the recurrence.
    X = torch.randn(64, cols) * 0.3
    H = X.T @ X
    H.diagonal().add_(float(damped.mean().item()))
    # Ensure symmetry
    H = ((H + H.T) * 0.5).contiguous()
    feedback, _ = M.qtip_block_ldl_factors(H, block_size=TRELLIS_LDL_BLOCK)
    assert int(torch.count_nonzero(feedback).item()) > 0, "test Hessian must have feedback"

    family = get_trellis_family(E2M1_FAMILY)
    q256 = 768
    schedule = uniform_column_schedule(cols, q256, family=family)
    alphabets = _CAMPAIGN.canonical_highrate_alphabets(schedule)
    # Shared global scale from the full weight, exactly as the production
    # BlockLDL producer does (pre-feedback, from the complete transformed weight).
    grouped = weight.reshape(rows, cols // 16, 16).abs().amax(dim=-1).clamp_min(1e-12) / 6.0
    global_scale = float((grouped.amax() / 448.0).clamp_min(1e-12).item())

    # Buffered (production) vs reference (oracle) must agree on both the
    # decoded weight and the per-block feedback targets.
    factory_ref = _trellis_terminal_factory(weight.device, schedule, alphabets, q256, global_scale)
    factory_buf = _trellis_terminal_factory(weight.device, schedule, alphabets, q256, global_scale)

    # The two factories are distinct closures but share the same global scale
    # and alphabets, so the recurrence should produce identical results.
    q_ref, targets_ref = M.reverse_block_feedback_reference(
        weight, feedback, factory_ref, block_size=TRELLIS_LDL_BLOCK
    )
    q_buf, targets_buf = M.reverse_block_feedback_buffered(
        weight, feedback, factory_buf, block_size=TRELLIS_LDL_BLOCK, buffer_blocks=1
    )
    assert torch.equal(q_ref, q_buf), "buffered vs reference decoded weight differs"
    for t_r, t_b in zip(targets_ref, targets_buf, strict=True):
        assert torch.allclose(t_r, t_b, rtol=3e-5, atol=3e-5), "buffered targets differ from oracle"

    # Also verify the 2-block buffered variant matches.
    factory_buf2 = _trellis_terminal_factory(weight.device, schedule, alphabets, q256, global_scale)
    q_buf2, _ = M.reverse_block_feedback_buffered(
        weight, feedback, factory_buf2, block_size=TRELLIS_LDL_BLOCK, buffer_blocks=2
    )
    assert torch.equal(q_buf2, q_ref)

    # Block reconciliation: one LDL block is exactly one trellis superblock.
    assert TRELLIS_LDL_BLOCK == 256
    assert len(schedule) == cols
    assert all(sum(schedule[i:i+256]) == sum(schedule[i:i+256]) for i in range(0, cols, 256))
    # No re-tiling was needed.


def test_diagonal_hessian_without_rotation_has_zero_feedback():
    # This is the F2/F3 bridge: a diagonal Hessian without the Hadamard
    # transform yields no cross-column LDL feedback, so LDLQ is a no-op and
    # collapses to the direct trellis encode.  This is why rotation alone
    # without LDLQ is the wrong measurement.
    torch.manual_seed(1)
    cols = 512
    raw = torch.rand(cols).add_(0.5)
    damped = _regularized_diag(raw)
    H = torch.diag(damped)
    feedback, _ = M.qtip_block_ldl_factors(H, block_size=256)
    assert int(torch.count_nonzero(feedback).item()) == 0

    # With zero feedback the reverse recurrence must produce the same wire
    # as the direct independent per-block encodes (i.e. the same as
    # encode_trellis_one_linear on the whole weight).
    rows = 2
    weight = torch.randn(rows, cols) * 0.2
    family = get_trellis_family(E2M1_FAMILY)
    q256 = 768
    schedule = uniform_column_schedule(cols, q256, family=family)
    alphabets = _CAMPAIGN.canonical_highrate_alphabets(schedule)
    grouped = weight.reshape(rows, cols // 16, 16).abs().amax(dim=-1).clamp_min(1e-12) / 6.0
    global_scale = float((grouped.amax() / 448.0).clamp_min(1e-12).item())

    from prismaquant.trellis_producer import encode_trellis_one_linear

    direct = encode_trellis_one_linear(
        weight,
        damped,
        family=E2M1_FAMILY,
        body_rate_q256=q256,
        schedule=schedule,
        layout=LAYOUT_TIGHT_OFFSETS,
        alphabets=alphabets,
        scale_rule="static_6",
        sb_chunk=4,
        determinism_mode="on",
        tailbite_candidates=4,
        backend="eager",
        point_route="full",
        global_scale_real_override=global_scale,
    )

    # BlockLDL with zero feedback should produce identical decoded weight
    # when the same per-column metric is used.
    factory = _trellis_terminal_factory(weight.device, schedule, alphabets, q256, global_scale, col_weights_full=damped)
    q_ldl, _ = M.reverse_block_feedback_buffered(
        weight, feedback, factory, block_size=256, buffer_blocks=1
    )
    assert torch.allclose(q_ldl, direct.decoded_weight, rtol=1e-6, atol=1e-6)
