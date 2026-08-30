"""Acceptance tests for the Sensitivity Card / format-cost seam.

The load-bearing test is `test_scalar_model_reproduces_allocator_solver`: the
card must reproduce today's allocator cost *exactly*, because it is the same
formula fed the same numbers. Any drift there is a bug, not a modelling choice.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from prismaquant.allocator_solver import predicted_dloss
from prismaquant.format_cost_protocol import (
    CostModel,
    FormatDescriptor,
    activation_dloss,
    analytic_act_quant_variance,
    block_scaled_act_quant_variance,
    price,
    uniform_act_quant_variance,
    weight_dloss_marginal,
    weight_dloss_scalar,
)
from prismaquant.sensitivity_card import (
    CardProvenance,
    Currency,
    RenderBasis,
    SensitivityCard,
    SensitivityUnit,
    UnitTopology,
)
from prismaquant.sensitivity_card_build import card_from_probe

OUT, IN, TOKENS = 8, 6, 100


def _synthetic_unit(seed: int = 0, *, with_vectors: bool = True,
                    name: str = "model.layers.0.self_attn.q_proj",
                    source_dtype: str = "bfloat16") -> SensitivityUnit:
    """Build a unit whose vectors are the TRUE marginals of a real H.

    H is constructed the way the probe constructs it -- a sum over tokens of
    outer(g^2, x^2) -- so the marginal identities are exact by construction and
    the test checks our arithmetic, not our assumptions.
    """
    rng = np.random.default_rng(seed)
    g_sq = rng.random((TOKENS, OUT)) + 0.1
    x_sq = rng.random((TOKENS, IN)) + 0.1
    H = g_sq.T @ x_sq                      # [OUT, IN], exactly the probe's chunk_h
    w = rng.standard_normal((OUT, IN))

    kwargs = {}
    if with_vectors:
        kwargs = dict(
            fisher_row=H.sum(axis=1),
            fisher_col=H.sum(axis=0),
            act_sq_sum=x_sq.sum(axis=0),
            g_sq_sum=g_sq.sum(axis=0),
            act_absmax=np.sqrt(x_sq).max(axis=0),
        )

    return SensitivityUnit(
        topology=UnitTopology(name=name, layer_index=0, role="q",
                              fused_group="L0.attn", source_dtype=source_dtype),
        out_features=OUT, in_features=IN, n_params=OUT * IN, n_tokens=TOKENS,
        h_trace_raw=float(H.sum()),
        h_w2_sum_raw=float((H * w ** 2).sum()),
        w_norm_sq=float((w ** 2).sum()), w_max_abs=float(np.abs(w).max()),
        **kwargs,
    )


def _provenance() -> CardProvenance:
    return CardProvenance(
        model_id="test/model", calib_hash="deadbeef" * 8, n_calib_samples=8,
        seq_len=512, probe_commit="abc1234", render_basis=RenderBasis.RTN)


# ----------------------------------------------------------------- invariants


def test_marginal_identity_holds():
    """sum(row) == sum(col) == h_trace_raw. Free consistency check."""
    u = _synthetic_unit()
    assert np.isclose(u.fisher_row.sum(), u.h_trace_raw)
    assert np.isclose(u.fisher_col.sum(), u.h_trace_raw)
    u.validate()


def test_validate_rejects_inconsistent_marginals():
    u = _synthetic_unit()
    bad = SensitivityUnit(**{**u.__dict__, "fisher_row": u.fisher_row * 2.0})
    with pytest.raises(ValueError, match="does not match h_trace_raw"):
        bad.validate()


def test_validate_rejects_wrong_shape():
    u = _synthetic_unit()
    bad = SensitivityUnit(**{**u.__dict__, "fisher_col": np.ones(IN + 1)})
    with pytest.raises(ValueError, match="expected"):
        bad.validate()


def test_card_builder_can_bind_the_probe_receipt_commit(tmp_path):
    """A card must not call its own checkout the Fisher producer."""
    probe = tmp_path / "probe.pkl"
    with probe.open("wb") as handle:
        pickle.dump({
            "meta": {
                "model": "test/model",
                "calib_hash": "a" * 32,
                "nsamples": 8,
                "seqlen": 512,
            },
            "stats": {
                "model.layers.0.mlp.down_proj": {
                    "out_features": 2,
                    "in_features": 3,
                    "n_params": 6,
                    "n_tokens_seen": 8,
                    "h_trace_raw": 1.0,
                    "h_w2_sum_raw": 0.5,
                    "w_norm_sq": 2.0,
                    "w_max_abs": 1.0,
                },
            },
        }, handle)

    commit = "0123456789abcdef" * 2 + "01234567"
    card = card_from_probe(str(probe), probe_commit=commit)
    assert card.provenance.probe_commit == commit


# ------------------------------------------------- backward-compatibility gate


def test_scalar_model_reproduces_allocator_solver():
    """The card's scalar path IS allocator_solver.predicted_dloss, exactly."""
    u = _synthetic_unit()
    for weight_mse in (1e-6, 3.25e-4, 0.5):
        for gain in (1.0, 0.75):
            assert weight_dloss_scalar(u, weight_mse, gain) == predicted_dloss(
                u.h_trace, weight_mse, gain)


def test_scalar_path_survives_a_card_without_vectors():
    """A vector-less card degrades to today's behaviour, it does not break."""
    u = _synthetic_unit(with_vectors=False)
    assert not u.has_vectors
    dw_sq = np.full((OUT, IN), 4e-4)
    got = weight_dloss_marginal(u, dw_sq)
    assert got == predicted_dloss(u.h_trace, float(dw_sq.mean()))


# --------------------------------------------------------- the marginal model


def test_marginal_model_matches_exact_fisher_on_rank1():
    """When H is genuinely rank-1, the marginal model is EXACT.

    This is the sharpest available check that the quadratic form and its
    normalization are right: build a single-token H (hence exactly rank-1) and
    compare against the full 0.5 * sum H*dW^2 computed elementwise.
    """
    rng = np.random.default_rng(7)
    g_sq = rng.random((1, OUT)) + 0.1
    x_sq = rng.random((1, IN)) + 0.1
    H = g_sq.T @ x_sq
    u = SensitivityUnit(
        topology=UnitTopology(name="u", source_dtype="bfloat16"),
        out_features=OUT, in_features=IN, n_params=OUT * IN, n_tokens=1,
        h_trace_raw=float(H.sum()), h_w2_sum_raw=0.0,
        w_norm_sq=1.0, w_max_abs=1.0,
        fisher_row=H.sum(axis=1), fisher_col=H.sum(axis=0),
    )
    dw_sq = rng.random((OUT, IN)) * 1e-3
    exact = 0.5 * float((H * dw_sq).sum())
    assert np.isclose(weight_dloss_marginal(u, dw_sq), exact, rtol=1e-10)


def test_marginal_model_differs_from_scalar_when_sensitivity_is_uneven():
    """The point of the marginals: they see structure the scalar cannot.

    Concentrate the weight error on the LOW-sensitivity output channels and the
    marginal model must price it below the scalar model, which averages.
    """
    u = _synthetic_unit(seed=3)
    order = np.argsort(u.fisher_row)
    dw_sq = np.zeros((OUT, IN))
    dw_sq[order[:2], :] = 1e-3            # error only on the 2 least sensitive
    cheap = weight_dloss_marginal(u, dw_sq)

    dw_sq2 = np.zeros((OUT, IN))
    dw_sq2[order[-2:], :] = 1e-3          # same magnitude, most sensitive
    dear = weight_dloss_marginal(u, dw_sq2)

    assert cheap < dear
    # The scalar model is blind to the difference: identical mean error.
    assert np.isclose(dw_sq.mean(), dw_sq2.mean())
    assert weight_dloss_scalar(u, float(dw_sq.mean())) == \
        weight_dloss_scalar(u, float(dw_sq2.mean()))


# ------------------------------------------------------------------ AQUA-AURA


def test_activation_dloss_returns_none_without_g_sq_sum():
    """A card lacking g_sq_sum returns None -- never 0.0.

    An unmeasured activation cost must not read as a free one.
    """
    u = _synthetic_unit()
    stripped = SensitivityUnit(**{**u.__dict__, "g_sq_sum": None})
    w = np.ones((OUT, IN))
    var = np.full(IN, 1e-4)
    assert activation_dloss(stripped, w, var) is None
    assert activation_dloss(u, w, var) is not None


def test_activation_dloss_uses_g_sq_sum_not_fisher_row():
    """The A-side sensitivity must be g_sq_sum (output space), NOT fisher_row.

    This pins a CURRENCY distinction, and the test above does not pin it: that
    one only checks the None contract, so swapping the body to `fisher_row`
    while keeping the `g_sq_sum is None` guard would leave it green.

    The two vectors are easy to confuse because both are indexed by output
    channel, but they are not the same quantity::

        fisher_row[o] = sum_i H[o,i] = sum_t g[t,o]^2 * (sum_i x[t,i]^2)
        g_sq_sum[o]   =                sum_t g[t,o]^2

    fisher_row already carries a per-token input-energy factor. Since
    activation_dloss multiplies the sensitivity by W^2 @ act_var -- itself an
    activation-derived quantity -- using fisher_row would double-count input
    energy and inflate exactly those units whose inputs are largest. That is the
    same failure signature as the divide-by-token-fraction bug removed in PR #14.

    Constructed so the two CANNOT coincide: per-token input energy is swept over
    three orders of magnitude, which makes fisher_row a strongly reweighted
    version of g_sq_sum rather than a near-multiple of it.
    """
    rng = np.random.default_rng(20260814)
    g_sq = rng.random((TOKENS, OUT)) + 0.1
    x_sq = rng.random((TOKENS, IN)) + 0.1
    # Sweep per-token input energy hard, so sum_i x[t,i]^2 is NOT ~constant in t.
    x_sq *= np.logspace(-1.5, 1.5, TOKENS)[:, None]
    H = g_sq.T @ x_sq
    w = rng.standard_normal((OUT, IN))

    u = SensitivityUnit(
        topology=UnitTopology(name="model.layers.0.mlp.down_proj", layer_index=0,
                              role="down", fused_group=None,
                              source_dtype="bfloat16"),
        out_features=OUT, in_features=IN, n_params=OUT * IN, n_tokens=TOKENS,
        h_trace_raw=float(H.sum()), h_w2_sum_raw=float((H * w ** 2).sum()),
        w_norm_sq=float((w ** 2).sum()), w_max_abs=float(np.abs(w).max()),
        fisher_row=H.sum(axis=1), fisher_col=H.sum(axis=0),
        act_sq_sum=x_sq.sum(axis=0), g_sq_sum=g_sq.sum(axis=0),
        act_absmax=np.sqrt(x_sq).max(axis=0),
    )

    var = rng.random(IN) * 1e-3
    per_out = (w ** 2) @ var

    got = activation_dloss(u, w, var)
    want_g = 0.5 * float(u.g_sq_sum @ per_out) / TOKENS
    want_row = 0.5 * float(u.fisher_row @ per_out) / TOKENS

    # The fixture must actually separate the two, else the assertion is vacuous.
    assert want_row > 10.0 * want_g, (
        "fixture failed to separate the two sensitivities; the discrimination "
        f"below would be vacuous (row={want_row:.3e} vs g={want_g:.3e})")

    # Tolerance is float32 eps, not an arbitrary epsilon. `_weighted_row_sum`
    # forms the square and the product in float32 on device and accumulates the
    # reduction in float64, so each product carries ~1 float32 ulp. Every term
    # is a square times a variance -- non-negative, so nothing cancels
    # catastrophically -- and the observed aggregate error on this fixture is
    # 3.4e-9 relative. Do NOT tighten this back to 1e-9: that pins the pre-GPU
    # host-float64 path and fails wherever CUDA is present. The discrimination
    # is untouched -- want_row sits 10x away, five orders outside this band.
    assert got == pytest.approx(want_g, rel=8 * float(np.finfo(np.float32).eps))
    assert got != pytest.approx(want_row, rel=1e-3)


def test_w4a4_and_w4a8_are_distinct_candidates():
    """The whole point of AQUA-AURA: A4 must cost more than A8.

    Under CostModel.AQUA the two formats differ ONLY in act_bits, so any
    difference in predicted loss is the activation term doing its job. Under the
    old scalar model they are indistinguishable.
    """
    u = _synthetic_unit()
    w = np.random.default_rng(11).standard_normal((OUT, IN))

    class _Plug:
        def __init__(self, desc):
            self.descriptor = desc

        def weight_error(self, unit, weight):
            return np.full((unit.out_features, unit.in_features), 4e-4)

    a4 = _Plug(FormatDescriptor(name="W4A4", weight_bits=4.0, act_bits=4,
                                     quantizes_activations=True))
    a8 = _Plug(FormatDescriptor(name="W4A8", weight_bits=4.0, act_bits=8,
                                     quantizes_activations=True))

    c4 = price(u, w, a4, render_basis=RenderBasis.RTN, model=CostModel.AQUA)
    c8 = price(u, w, a8, render_basis=RenderBasis.RTN, model=CostModel.AQUA)

    assert c4.act_dloss is not None and c8.act_dloss is not None
    assert c4.act_dloss > c8.act_dloss, "4-bit activations must cost more than 8-bit"
    assert c4.to_predicted_dloss() > c8.to_predicted_dloss()
    # Same weight side -- the difference is entirely the A-side.
    assert c4.weight_dloss == c8.weight_dloss
    # And the two are identical under the weight-only model, which is the bug
    # AQUA-AURA exists to fix.
    s4 = price(u, w, a4, render_basis=RenderBasis.RTN, model=CostModel.MARGINAL)
    s8 = price(u, w, a8, render_basis=RenderBasis.RTN, model=CostModel.MARGINAL)
    assert s4.to_predicted_dloss() == s8.to_predicted_dloss()


def test_act_quant_variance_shrinks_with_bits():
    u = _synthetic_unit()
    v4 = uniform_act_quant_variance(u, 4)
    v8 = uniform_act_quant_variance(u, 8)
    # Each extra bit halves the step, so variance drops 4x per bit.
    assert np.allclose(v4 / v8, 4.0 ** 4)


# ------------------------------------------------------- legality and currency


def test_passthrough_is_refused_when_source_dtype_mismatches():
    """BF16/FP8_SOURCE are passthrough-only; never synthesize them."""
    u = _synthetic_unit(source_dtype="float8_e4m3fn")
    w = np.zeros((OUT, IN))

    class _Plug:
        descriptor = FormatDescriptor(
            name="BF16", weight_bits=16.0, passthrough=True,
            requires_source_dtype="bfloat16")

        def weight_error(self, unit, weight):
            return np.zeros((unit.out_features, unit.in_features))

    assert price(u, w, _Plug(), render_basis=RenderBasis.RTN) is None

    u_bf16 = _synthetic_unit(source_dtype="bfloat16")
    assert price(u_bf16, w, _Plug(), render_basis=RenderBasis.RTN) is not None


def test_costs_leave_only_in_delta_loss_currency():
    u = _synthetic_unit()
    w = np.zeros((OUT, IN))

    class _Plug:
        descriptor = FormatDescriptor(name="NVFP4", weight_bits=4.25)

        def weight_error(self, unit, weight):
            return np.full((unit.out_features, unit.in_features), 1e-4)

    c = price(u, w, _Plug(), render_basis=RenderBasis.RTN)
    c.assert_currency(Currency.DELTA_LOSS)
    with pytest.raises(ValueError, match="Mixing bases"):
        c.assert_currency(Currency.WEIGHT_MSE)


# ------------------------------------------------------------ card-level rules


def test_calibration_is_identity():
    a = SensitivityCard(_provenance(), [_synthetic_unit()])
    other = CardProvenance(**{**_provenance().__dict__, "calib_hash": "f" * 64})
    b = SensitivityCard(other, [_synthetic_unit()])
    with pytest.raises(ValueError, match="different\n?\\s*calibrations|Calibration is identity"):
        a.assert_compatible(b)


def test_render_basis_mismatch_is_refused():
    a = SensitivityCard(_provenance(), [_synthetic_unit()])
    other = CardProvenance(
        **{**_provenance().__dict__, "render_basis": RenderBasis.COMPENSATED})
    b = SensitivityCard(other, [_synthetic_unit()])
    with pytest.raises(ValueError, match="render basis mismatch"):
        a.assert_compatible(b)


def test_roundtrip_npz_preserves_everything(tmp_path):
    card = SensitivityCard(_provenance(), [
        _synthetic_unit(0, name="a"), _synthetic_unit(1, name="b")])
    card.validate()
    path = str(tmp_path / "card.npz")
    card.to_npz(path)
    back = SensitivityCard.from_npz(path)
    back.validate()

    assert len(back) == 2
    assert back.provenance == card.provenance
    for name in card.names():
        u0, u1 = card[name], back[name]
        assert u0.topology == u1.topology
        assert u0.h_trace_raw == pytest.approx(u1.h_trace_raw)
        for field in ("fisher_row", "fisher_col", "act_sq_sum", "g_sq_sum"):
            assert np.allclose(getattr(u0, field), getattr(u1, field))


def test_npz_is_loadable_without_pickle(tmp_path):
    """A shareable artifact must not execute arbitrary objects on load."""
    card = SensitivityCard(_provenance(), [_synthetic_unit()])
    path = str(tmp_path / "card.npz")
    card.to_npz(path)
    with np.load(path, allow_pickle=False) as data:   # must not raise
        assert "__header__" in data


def test_structure_is_carried_but_not_policy():
    """Sibling identity travels; must-share-format does not."""
    card = SensitivityCard(_provenance(), [
        _synthetic_unit(0, name="model.layers.0.self_attn.q_proj"),
        _synthetic_unit(1, name="model.layers.0.self_attn.k_proj"),
    ])
    groups = card.fused_groups()
    assert set(groups["L0.attn"]) == {
        "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"}
    # The card exposes no serving policy at all -- that is the consumer's job.
    assert not hasattr(card, "must_share_format")


# ------------------------------------------- AQUA-1: block-scaled A-side model
#
# The uniform model sets the activation step from act_absmax -- the largest
# value a channel reached over the WHOLE calibration. Every activation-
# quantizing format in the shipped registry is block scaled (NVFP4 G=16, MX
# G=32) and rescales per token per block, so that step is wrong for 100% of the
# formats the fallback can fire on, and wrong in the direction that penalises
# W4A4. These tests hold the replacement to a simulated quantizer.

A_BITS, A_GROUP = 4, 16
A_IN, A_OUT, A_TOK = 128, 8, 512


def _unit_from_activations(x: np.ndarray, seed: int = 0) -> SensitivityUnit:
    """A unit whose A-side vectors are the TRUE marginals of activations ``x``."""
    rng = np.random.default_rng(seed)
    n_tok, n_in = x.shape
    g_sq = rng.random((n_tok, A_OUT)) + 0.1
    x_sq = x ** 2
    H = g_sq.T @ x_sq
    w = rng.standard_normal((A_OUT, n_in))
    return SensitivityUnit(
        topology=UnitTopology(name="model.layers.0.mlp.down_proj", layer_index=0,
                              role="down", fused_group=None,
                              source_dtype="bfloat16"),
        out_features=A_OUT, in_features=n_in, n_params=A_OUT * n_in,
        n_tokens=n_tok,
        h_trace_raw=float(H.sum()), h_w2_sum_raw=float((H * w ** 2).sum()),
        w_norm_sq=float((w ** 2).sum()), w_max_abs=float(np.abs(w).max()),
        fisher_row=H.sum(axis=1), fisher_col=H.sum(axis=0),
        act_sq_sum=x_sq.sum(axis=0), g_sq_sum=g_sq.sum(axis=0),
        act_absmax=np.abs(x).max(axis=0),
    )


def _simulate_block_act_error_var(x: np.ndarray, bits: int = A_BITS,
                                  group: int = A_GROUP) -> np.ndarray:
    """Ground truth: per-channel error variance of a dynamic per-token,
    per-block absmax grid -- what an NVFP4/MX kernel actually does."""
    n_in = x.shape[1]
    err = np.empty_like(x, dtype=np.float64)
    for start in range(0, n_in, group):
        blk = x[:, start:start + group].astype(np.float64)
        absmax = np.abs(blk).max(axis=1, keepdims=True)
        step = 2.0 * absmax / float(2 ** bits)
        step = np.where(step > 0.0, step, 1.0)
        err[:, start:start + group] = np.round(blk / step) * step - blk
    return (err ** 2).mean(axis=0)


def _hetero_activations(seed: int = 7, *, massive: bool = False) -> np.ndarray:
    """Activations with heterogeneous per-channel scales.

    ``massive=True`` additionally gives 2% of tokens a 30x magnitude spike --
    the "massive activation" phenomenon real LLMs exhibit at delimiter/BOS
    positions. That is the regime where a global-absmax step and a per-token
    block step diverge, so it is what the non-vacuity test uses. It was chosen
    for being representative and STABLE (uniform lands at 19.7-22.6x across
    seeds 7/8/9); an earlier fixture that spiked 6 individual (token, channel)
    cells varied between 3x and 30x depending on where the spikes fell, which
    would have made the threshold a coin flip rather than a property.
    """
    rng = np.random.default_rng(seed)
    scales = np.exp(rng.normal(0.0, 0.8, size=A_IN))
    x = rng.standard_normal((A_TOK, A_IN)) * scales
    if massive:
        n_spiked = max(1, int(0.02 * A_TOK))
        x[rng.choice(A_TOK, n_spiked, replace=False), :] *= 30.0
    return x


@pytest.mark.parametrize("massive", [False, True])
def test_block_model_tracks_a_simulated_block_quantizer(massive):
    """The analytic block model must track the real quantizer's error."""
    x = _hetero_activations(massive=massive)
    u = _unit_from_activations(x)

    analytic = block_scaled_act_quant_variance(u, A_BITS, A_GROUP)
    sim = _simulate_block_act_error_var(x)

    ratio = float(np.median(analytic / sim))
    # Measured at authoring time: 1.04 clean, 1.17 with massive tokens. The
    # residual is the step^2/12 quantizer model, not the order statistic --
    # E[blockmax^2] itself matches simulation to within 0.1%.
    assert 0.7 < ratio < 1.5, (
        f"block model median ratio {ratio:.3f} vs simulated quantizer")


def test_block_model_beats_the_global_absmax_model_it_replaces():
    """NON-VACUITY: the fix must actually move the number.

    Without this, a block model that silently equalled the uniform one would
    still pass the tracking test above on a benign fixture. Activation outliers
    are the regime that separates them, and the regime real LLMs live in.
    """
    x = _hetero_activations(massive=True)
    u = _unit_from_activations(x)
    sim = _simulate_block_act_error_var(x)

    block = block_scaled_act_quant_variance(u, A_BITS, A_GROUP)
    uniform = uniform_act_quant_variance(u, A_BITS)

    uniform_err = float(uniform.mean() / sim.mean())
    block_err = float(block.mean() / sim.mean())

    # Measured at authoring time: uniform overstates by 22.6x, block by 1.08x
    # (19.7-22.6x / 1.08-1.23x across seeds 7/8/9).
    assert uniform_err > 10.0, (
        f"fixture is not discriminating: uniform only off by {uniform_err:.1f}x")
    assert block_err < 3.0, f"block model off by {block_err:.1f}x"
    assert block_err < uniform_err / 5.0


def test_block_model_needs_act_sq_sum_and_says_so():
    """A per-channel SCALE is required; act_absmax alone would smuggle the
    global maximum back in, which is the error being removed."""
    u = _unit_from_activations(_hetero_activations())
    stripped = SensitivityUnit(**{**u.__dict__, "act_sq_sum": None})
    assert block_scaled_act_quant_variance(stripped, A_BITS, A_GROUP) is None


def test_block_model_handles_a_ragged_final_block():
    """in_features need not be a multiple of the group size."""
    x = _hetero_activations()[:, :A_IN - 5]            # 123 channels, G=16
    u = _unit_from_activations(x)
    var = block_scaled_act_quant_variance(u, A_BITS, A_GROUP)
    assert var.shape == (A_IN - 5,)
    assert np.all(np.isfinite(var)) and np.all(var > 0.0)


def test_analytic_dispatch_follows_the_declared_activation_grid():
    """A block-scaled descriptor must get the block model, not the uniform one.

    Dispatch is on declared metadata, so an arbitrary downstream format is
    priced correctly without editing the protocol module.
    """
    u = _unit_from_activations(_hetero_activations(massive=True))
    blocked = FormatDescriptor(name="W4A4_BLOCK", weight_bits=4.0, act_bits=A_BITS,
                               quantizes_activations=True, act_group_size=A_GROUP)
    per_channel = FormatDescriptor(name="W4A4_FLAT", weight_bits=4.0,
                                   act_bits=A_BITS, quantizes_activations=True)

    got_block = analytic_act_quant_variance(u, blocked)
    got_flat = analytic_act_quant_variance(u, per_channel)

    assert np.allclose(
        got_block, block_scaled_act_quant_variance(u, A_BITS, A_GROUP))
    assert np.allclose(got_flat, uniform_act_quant_variance(u, A_BITS))
    # And they must genuinely differ, or the dispatch proves nothing.
    assert got_flat.mean() > 5.0 * got_block.mean()


def test_registry_descriptor_carries_the_activation_group_size():
    """The A-side grouping must survive the FormatSpec -> descriptor hop.

    It was being dropped, which is what made the analytic fallback wrong for
    every format it could fire on.
    """
    from prismaquant.format_cost_registry import descriptor_for
    from prismaquant.format_registry import REGISTRY

    nvfp4 = descriptor_for(REGISTRY["NVFP4"], shape=(A_OUT, A_IN))
    assert nvfp4.act_group_size == 16
    assert nvfp4.group_size == 16

    mx = descriptor_for(REGISTRY["MXFP4"], shape=(A_OUT, A_IN))
    assert mx.act_group_size == 32

    # A format that does not touch activations carries no activation grid.
    a16 = descriptor_for(REGISTRY["NVFP4A16"], shape=(A_OUT, A_IN))
    assert a16.quantizes_activations is False
    assert a16.act_group_size is None
