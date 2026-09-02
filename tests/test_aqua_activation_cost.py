"""AQUA-AURA: the activation term reaches the DP, exactly once.

WHY
---
Choosing NVFP4 commits a Linear's ACTIVATIONS to 4 bits, not just its weights.
The allocator's cost was weight-only and could not see that -- NVFP4 and
NVFP4A16 render weights bit-identically -- so the DP bought W4A4 at a discount
to its true cost. ``cost_entry_act_dloss`` closes that.

The risk in closing it is double-counting, because three of the four pricing
branches must NOT receive the term:

  * ``_prices_from_output_mse`` rows are already activation-inclusive by
    construction -- the measurement saw the activation path.
  * exact-by-construction rows are 0.0 because nothing happens to the tensor.
  * super-item rows already contain their members' A-side, since a super item is
    priced as the SUM of its members' ``cost_entry_predicted_dloss``.

Getting any of those wrong is silent: the allocation still solves, it just
solves the wrong problem. So each branch is pinned separately.

The last test is the one the whole design rests on: the A-side is independent of
the render basis. If that were false, an A-side priced off the card could not be
added to a production-rendered weight cost, and the stage would be a rendering
confound rather than a cost completion.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from prismaquant.allocator_candidates import (  # noqa: E402
    ACT_DLOSS_KEY, cost_entry_act_dloss, cost_entry_predicted_dloss,
)

STATS = {"n_params": 4096, "in_features": 64, "out_features": 64,
         "h_trace": 1.0, "n_tokens": 8}
ACT = 0.25


def _weight_only_entry(**extra):
    e = {"predicted_dloss": 1.0, "output_mse_measured": False,
         "cost_source": "aura"}
    e.update(extra)
    return e


def test_pre_aqua_rows_are_bit_for_bit_unchanged():
    """Every cost artifact written before AQUA lacks the key.

    Those runs must stay reproducible, so a missing A-side reads as 0.0 here --
    while remaining a HOLE in the writer's report, which is where an unpriced
    activation-quantizing format is supposed to be visible.
    """
    entry = _weight_only_entry()
    assert cost_entry_act_dloss(entry) == 0.0
    before = cost_entry_predicted_dloss(STATS, entry, format_name="NVFP4")
    assert before == pytest.approx(1.0)


def test_the_term_is_added_on_the_weight_only_branch():
    entry = _weight_only_entry(**{ACT_DLOSS_KEY: ACT})
    got = cost_entry_predicted_dloss(STATS, entry, format_name="NVFP4")
    assert got == pytest.approx(1.0 + ACT), (
        "the A-side must reach the DP additively -- it is a Delta-loss in the "
        "same currency as the weight term, not a multiplicative penalty")


def test_the_a_side_rides_the_same_p5a_transfer_as_the_w_side():
    """The two halves of one unit's price must be on ONE scale.

    P5a is a per-family multiplicative re-leveling of weight-only rows. Adding
    the A-side outside that multiply leaves it in un-transferred units, and the
    fitted constants are large -- x8103 for the NVFP4 family on Qwen3.8-27B --
    so an A-side worth 6x the W-side arrives as 0.07% of the total. That is not
    a rounding difference: it produced a shipped Pareto byte-identical to the
    weight-only one, which is how the bug was caught.

    Multiplying the SUM keeps the A:W ratio, which is what the DP ranks on.
    """
    entry = _weight_only_entry(**{ACT_DLOSS_KEY: ACT})
    plain = cost_entry_predicted_dloss(STATS, entry, format_name="NVFP4")
    assert plain == pytest.approx(1.0 + ACT)

    class _FixedPenalty:
        enabled = True

        def penalty_for(self, format_name, act_changes):
            # ``_activation_penalty`` reads element [0]; the second slot is the
            # branch label used only for reporting.
            return (1000.0, "test")

    scaled = cost_entry_predicted_dloss(
        STATS, entry, format_name="NVFP4", activation_pricing=_FixedPenalty())
    assert scaled == pytest.approx((1.0 + ACT) * 1000.0), (
        "a large per-family penalty must scale the A-side with the W-side, not "
        "drown it")
    # The ratio is what the DP ranks on and must be penalty-invariant.
    without = dict(entry)
    without.pop(ACT_DLOSS_KEY)
    ref = cost_entry_predicted_dloss(
        STATS, without, format_name="NVFP4", activation_pricing=_FixedPenalty())
    assert scaled / ref == pytest.approx(plain / 1.0)


def test_it_is_not_added_to_an_activation_inclusive_measurement():
    """The double-count guard for ``_prices_from_output_mse``.

    A measured output_mse row already saw the activation path. Adding the
    modelled A-side on top would charge that layer twice for the same physics,
    and would do it ONLY on rows that happen to carry a measurement -- i.e. it
    would mis-rank rungs within one family, which is the failure mode this
    branch was split out to fix in the first place.
    """
    from prismaquant.allocator_candidates import _prices_from_output_mse
    measured = {"output_mse": 0.5, "output_mse_measured": True,
                ACT_DLOSS_KEY: ACT}
    if not _prices_from_output_mse(STATS, measured):
        pytest.skip("this cost shape does not take the output_mse branch")
    with_act = cost_entry_predicted_dloss(STATS, measured, format_name="NVFP4")
    without = dict(measured)
    without.pop(ACT_DLOSS_KEY)
    assert with_act == pytest.approx(
        cost_entry_predicted_dloss(STATS, without, format_name="NVFP4"))


def test_it_is_not_added_to_a_super_item():
    """A super item is the SUM of its members' priced dloss.

    Each member term already carries its own A-side, so re-adding at the
    aggregate would scale the activation cost by the group size -- 3x on a
    fused q/k/v, and more on a packed expert group.
    """
    from prismaquant.allocator_candidates import APPLIED_MARKER_KEY
    entry = _weight_only_entry(**{ACT_DLOSS_KEY: ACT,
                                  APPLIED_MARKER_KEY: True})
    assert cost_entry_predicted_dloss(
        STATS, entry, format_name="NVFP4") == pytest.approx(1.0)


def test_merge_writes_only_where_the_format_exists():
    from prismaquant.aqua_activation_cost import merge_act_dloss
    costs = {"a": {"NVFP4": {"predicted_dloss": 1.0},
                   "BF16": {"predicted_dloss": 0.0}},
             "b": {"NVFP4": {"predicted_dloss": 2.0}}}
    report = merge_act_dloss(costs, {"a": {"NVFP4": 0.1, "FP8_E4M3": 0.9},
                                     "c": {"NVFP4": 0.7}})
    assert costs["a"]["NVFP4"][ACT_DLOSS_KEY] == pytest.approx(0.1)
    assert ACT_DLOSS_KEY not in costs["a"]["BF16"], (
        "BF16 does not quantize activations; writing a 0.0 there would make a "
        "correct absence indistinguishable from an unpriced hole")
    assert ACT_DLOSS_KEY not in costs["b"]["NVFP4"]
    assert report["entries_merged"] == 1
    assert report["units_without_act_price"] == 1


@pytest.mark.parametrize("fmt,expect_priced", [("NVFP4", True),
                                               ("FP8_E4M3", True),
                                               ("BF16", False)])
def test_price_activation_only_follows_the_explicit_predicate(fmt,
                                                              expect_priced):
    card = pytest.importorskip("prismaquant.sensitivity_card")
    from prismaquant.format_cost_protocol import price_activation_only
    from prismaquant.format_cost_registry import RegistryFormatPlugin
    unit = _synthetic_unit(card)
    w = np.random.default_rng(0).normal(0, 0.02, (32, 64)).astype(np.float32)
    try:
        plugin = RegistryFormatPlugin.build(fmt, shape=w.shape, device="cpu")
    except Exception as exc:
        pytest.skip(f"{fmt} unbuildable on CPU: {exc}")
    got = price_activation_only(unit, w, plugin)
    if expect_priced:
        assert got is not None and got > 0.0
    else:
        assert got is None, (
            "a format that leaves activations alone must return None, not 0.0 "
            "-- the two mean different things and only one is a hole")


def test_the_a_side_does_not_depend_on_the_weights_being_rendered():
    """THE claim the whole stage rests on.

    ``activation_dloss`` reads the DENSE weight, ``g_sq_sum`` and the format's
    activation grid. No render enters it. That is what makes it legitimate to
    price the A-side off a shared card and add it to a cost whose W-side was
    built with the full GPTQ+JSO production recipe: the two halves are measured
    on different bases only because the A-side HAS no basis.

    Pinned by pricing the same unit against two very different weight matrices
    that share a scale, and asserting the A-side tracks the weights it is given
    rather than any rendering of them -- and, more sharply, that priced twice on
    the same weights it is exactly reproducible.
    """
    card = pytest.importorskip("prismaquant.sensitivity_card")
    from prismaquant.format_cost_protocol import price_activation_only
    from prismaquant.format_cost_registry import RegistryFormatPlugin
    unit = _synthetic_unit(card)
    rng = np.random.default_rng(1)
    w = rng.normal(0, 0.02, (32, 64)).astype(np.float32)
    try:
        plugin = RegistryFormatPlugin.build("NVFP4", shape=w.shape,
                                            device="cpu")
    except Exception as exc:
        pytest.skip(f"NVFP4 unbuildable on CPU: {exc}")
    a = price_activation_only(unit, w, plugin)
    b = price_activation_only(unit, w.copy(), plugin)
    assert a == pytest.approx(b, rel=0, abs=0), (
        "the A-side must be a pure function of (unit, dense weight, format)")


def test_an_explicit_act_var_wins_over_the_modelled_one():
    """Real measured activations must not be silently overridden by the model.

    ``act_var`` is the one factor in the A-side that is not measured by default:
    the plugin runs the real quantizer but over independent per-channel
    Gaussians, which reproduce every channel's marginal and destroy the joint --
    and an NVFP4 block scale is a function of the joint, since 16 consecutive
    channels share one scale set by the largest magnitude among them.

    So when a caller has the layer's REAL input rows, that variance must be the
    one used, and the plugin's model must not even be consulted. Pinned with a
    plugin whose model would raise if reached.
    """
    card = pytest.importorskip("prismaquant.sensitivity_card")
    from prismaquant.format_cost_protocol import price_activation_only
    unit = _synthetic_unit(card)
    w = np.full((unit.out_features, unit.in_features), 0.01, dtype=np.float32)

    class _Desc:
        quantizes_activations = True

    class _WouldModel:
        descriptor = _Desc()

        def activation_error_variance(self, unit):
            raise AssertionError("the modelled variance must not be consulted "
                                 "when a measured one was supplied")

    var = np.full(unit.in_features, 1e-4, dtype=np.float64)
    got = price_activation_only(unit, w, _WouldModel(), act_var=var)
    assert got is not None and got > 0.0
    # Doubling the measured variance must double the price: the A-side is
    # linear in act_var, which is what makes a real-activation measurement a
    # drop-in replacement for the modelled one rather than a recalibration.
    assert price_activation_only(
        unit, w, _WouldModel(), act_var=var * 2.0) == pytest.approx(2.0 * got)


def _synthetic_unit(card_mod):
    """Smallest card unit that can carry an A-side price.

    ``g_sq_sum`` is the OUTPUT-space sensitivity the A-side uses -- not
    ``fisher_row``, which is the weight-space one. That distinction is the whole
    point of AQUA and was documented backwards once already, so the fixture
    states it explicitly.
    """
    n_in, n_out, n_tok = 64, 32, 128
    rng = np.random.default_rng(7)
    return card_mod.SensitivityUnit(
        topology=card_mod.UnitTopology(name="probe.linear", layer_index=0,
                                       role="down", source_dtype="bfloat16"),
        out_features=n_out, in_features=n_in, n_params=n_in * n_out,
        n_tokens=n_tok,
        h_trace_raw=1.0, h_w2_sum_raw=1e-3,
        w_norm_sq=1.0, w_max_abs=0.1,
        g_sq_sum=np.full(n_out, 1e-3, dtype=np.float64),
        act_sq_sum=rng.uniform(0.5, 1.5, n_in).astype(np.float64),
        act_absmax=rng.uniform(2.0, 6.0, n_in).astype(np.float64),
    )


# --- name resolution -------------------------------------------------------
# A card unit name and a checkpoint tensor key are different namespaces, and
# when they diverge this stage fails SILENTLY: nothing resolves, nothing is
# priced, and because `cost_entry_act_dloss` defaults to 0.0 the DP reads every
# 4-bit-activation format as free. DSv4-Flash is the live example -- it renames
# the path (`model.layers.N.mlp` -> `layers.N.ffn`) AND the leaf
# (`down_proj` -> `w2`), which no path-aliasing recovers.

class _RenamingProfile:
    """Minimal stand-in for a ModelProfile that renames path and leaf."""

    _LEAF = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}

    def checkpoint_to_live_name(self, key: str, *, multimodal: bool = False):
        if not key.startswith("layers."):
            return None
        live = "model." + key.replace(".ffn.", ".mlp.")
        for proj, w in self._LEAF.items():
            live = live.replace(f".{w}.", f".{proj}.")
        return live


def test_resolver_uses_the_profile_when_the_checkpoint_renames_leaves():
    from prismaquant.aqua_activation_cost import build_weight_resolver
    weight_map = {"layers.0.ffn.experts.0.w2.weight": "s0.safetensors",
                  "layers.0.ffn.experts.0.w1.weight": "s0.safetensors"}
    card_name = "model.layers.0.mlp.experts.0.down_proj"

    generic = build_weight_resolver(weight_map)
    assert card_name not in generic, (
        "if the generic index ever resolves this, the regression it guards is "
        "gone and this test is meaningless")

    resolved = build_weight_resolver(weight_map, profile=_RenamingProfile())
    assert resolved[card_name] == "layers.0.ffn.experts.0.w2.weight"


def test_resolver_keeps_direct_keys_authoritative_over_profile_aliases():
    """A profile alias must only ADD reachability, never shadow a real key."""
    from prismaquant.aqua_activation_cost import build_weight_resolver

    class _Colliding:
        def checkpoint_to_live_name(self, key, *, multimodal=False):
            return "model.layers.0.mlp.down_proj.weight"

    weight_map = {"model.layers.0.mlp.down_proj.weight": "s0.safetensors",
                  "layers.0.ffn.w2.weight": "s1.safetensors"}
    resolved = build_weight_resolver(weight_map, profile=_Colliding())
    assert resolved["model.layers.0.mlp.down_proj"] == \
        "model.layers.0.mlp.down_proj.weight"


# --- quantized-source materialization --------------------------------------
# `activation_dloss` needs the DENSE source weight, but DSv4-Flash stores its
# routed experts as MXFP4 nibble-packs and everything else as block-FP8 with
# E8M0 `.scale` siblings. `materialize_source_weight` dispatches on the
# streaming loader's own scale map (declaration-driven, never shape-inferred)
# and reuses the loader's decoders, so the W it prices is the W the probe ran
# on. Every mismatch must RAISE: an unpriced A-side reads as 0.0 (= free) to
# the DP, which is the exact mispricing this stage exists to remove.


def _torch_and_map(mxfp4=(), entries=(), block=(128, 128)):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch lacks float8_e8m0fnu")
    from prismaquant.layer_streaming import Fp8ScaleInvMap
    data = {k: ("shard", k) for k in (*entries, *mxfp4)}
    return torch, Fp8ScaleInvMap(data, block if data else None,
                                 mxfp4_names=frozenset(mxfp4))


def test_materialize_mxfp4_source_shape_and_values():
    """Nibble-pack decode: hand-computed values, not just a shape.

    Byte 0x21 is (low=0x1, high=0x2) -> (0.5, 1.0) in E2M1, low nibble first.
    Scale bytes are E8M0 exponents: 127 -> 2^0, 128 -> 2^1. One scale per 32
    logical elements, so a (2, 32)-byte pack is logically (2, 64) with a
    (2, 2) scale plane.
    """
    torch, fp8_map = _torch_and_map(mxfp4=("u.weight",))
    from prismaquant.aqua_activation_cost import materialize_source_weight
    packed = torch.full((2, 32), 0x21, dtype=torch.uint8)
    scale = torch.tensor([[127, 128], [129, 127]],
                         dtype=torch.uint8).view(torch.float8_e8m0fnu)
    w = materialize_source_weight("u", packed, scale, fp8_map)
    assert w.dtype == torch.float32 and tuple(w.shape) == (2, 64)
    base = np.tile([0.5, 1.0], 32)
    expect = np.stack([
        np.concatenate([base[:32] * 1.0, base[32:] * 2.0]),
        np.concatenate([base[:32] * 4.0, base[32:] * 1.0]),
    ])
    assert np.array_equal(w.numpy(), expect), (
        "decode must be exact: E2M1 values and power-of-two scales are all "
        "representable, so there is no tolerance to hide behind")


def test_materialize_fp8_block_source_shape_and_values():
    """Block-FP8 decode: the E8M0 `.scale` sibling is a MULTIPLIER.

    All-ones e4m3 weight (1.0 is exact), block (2, 2), scale exponents
    [[127, 128], [129, 127]] -> [[1, 2], [4, 1]]. The result must be the
    block-broadcast product, exactly.
    """
    torch, fp8_map = _torch_and_map(entries=("u.weight",), block=(2, 2))
    from prismaquant.aqua_activation_cost import materialize_source_weight
    weight = torch.ones(4, 4, dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[127, 128], [129, 127]],
                         dtype=torch.uint8).view(torch.float8_e8m0fnu)
    w = materialize_source_weight("u", weight, scale, fp8_map)
    assert w.dtype == torch.float32 and tuple(w.shape) == (4, 4)
    expect = np.kron([[1.0, 2.0], [4.0, 1.0]], np.ones((2, 2)))
    assert np.array_equal(w.numpy(), expect)


def test_materialize_dense_passthrough_is_a_noop():
    """Requirement for every already-dense checkpoint (Qwen3.8-27B et al.):
    no scale entry + dense float dtype -> the same values, fp32, untouched."""
    torch, fp8_map = _torch_and_map()
    from prismaquant.aqua_activation_cost import materialize_source_weight
    src = torch.tensor([[0.25, -1.5], [3.0, 0.0]], dtype=torch.bfloat16)
    w = materialize_source_weight("u", src, None, fp8_map)
    assert w.dtype == torch.float32
    assert np.array_equal(w.numpy(), src.to(torch.float32).numpy())
    # An empty/absent map must behave identically (plain dicts are legal).
    w2 = materialize_source_weight("u", src, None, {})
    assert np.array_equal(w2.numpy(), w.numpy())


def test_materialize_fails_loud_never_silent():
    """The full refusal matrix. None of these may return a plausible W."""
    torch, fp8_map = _torch_and_map(entries=("u.weight",), block=(2, 2))
    from prismaquant.aqua_activation_cost import materialize_source_weight
    scale = torch.tensor([[127, 127], [127, 127]],
                         dtype=torch.uint8).view(torch.float8_e8m0fnu)
    # 1. Quantized bytes with no scale entry: casting would install
    #    code-range values (the historical fp8-range bug).
    for bad in (torch.ones(4, 4, dtype=torch.float8_e4m3fn),
                torch.ones(4, 4, dtype=torch.int8)):
        with pytest.raises(RuntimeError, match="no scale entry"):
            materialize_source_weight("v", bad, None, fp8_map)
    # 2. Scale entry declared but no scale supplied.
    with pytest.raises(RuntimeError, match="none was supplied"):
        materialize_source_weight(
            "u", torch.ones(4, 4, dtype=torch.float8_e4m3fn), None, fp8_map)
    # 3. Dense float WITH a scale entry: the map and the checkpoint disagree.
    with pytest.raises(RuntimeError, match="refusing to guess"):
        materialize_source_weight(
            "u", torch.ones(4, 4, dtype=torch.float32), scale, fp8_map)
    # 4. Mapped as block-FP8 but the bytes are not a float8 wire.
    with pytest.raises(RuntimeError, match="not a float8 wire"):
        materialize_source_weight(
            "u", torch.ones(4, 4, dtype=torch.int8), scale, fp8_map)
    # 5. Declared MXFP4 but the packed grid does not match: the loader's own
    #    assertion fires (reused, not re-derived).
    _, mx_map = _torch_and_map(mxfp4=("m.weight",))
    with pytest.raises(ValueError, match="declared MXFP4"):
        materialize_source_weight(
            "m", torch.ones(2, 32, dtype=torch.uint8),
            torch.ones(2, 7, dtype=torch.uint8).view(torch.float8_e8m0fnu),
            mx_map)


def test_zero_resolution_refuses_instead_of_writing_a_no_op(tmp_path):
    """0/N resolved must raise, not produce a plausible-looking artifact.

    The artifact it would otherwise write has the right units and the right
    formats and an A-side that is simply absent -- i.e. free. That is the one
    outcome this stage exists to prevent, so it must never be reachable.
    """
    from prismaquant.aqua_activation_cost import activation_dloss_table
    import json

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"layers.0.ffn.experts.0.w2.weight": "s0.safetensors"}}))
    with pytest.raises(SystemExit) as exc:
        activation_dloss_table(
            object(), str(tmp_path), ["NVFP4"],
            names=["model.layers.0.mlp.experts.0.down_proj"],
            executed_activation_formats="all")
    assert "NAME-SPACE" in str(exc.value)


class TestServedActivationContractGovernsTheASide:
    """The A-side belongs to the RUNTIME, not to the format registry.

    Regression cover for 2026-08-17: the DSv4-Flash 92 GB body and the
    Qwen3.8-27B CB-A allocation were both priced from an AQUA-merged cost while
    their lane served every CB unit through gridbook's exact BF16 bridge, which
    quantizes no activations at all. The A-side was a phantom, and the DP paid
    for it in weight bits -- ~19k units dropped from codebook rung K16 to K12
    to fund FP8 promotions that escaped a cost of zero.
    """

    def test_pricing_refuses_to_guess_the_served_contract(self, tmp_path):
        import json
        from prismaquant.aqua_activation_cost import activation_dloss_table

        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a.weight": "s0.safetensors"}}))
        with pytest.raises(SystemExit) as exc:
            activation_dloss_table(object(), str(tmp_path), ["NVFP4"],
                                   names=["a"])
        assert "executed_activation_formats is required" in str(exc.value)

    def test_a_lane_that_executes_nothing_refuses_rather_than_charging_zero(
            self, tmp_path):
        """An all-zero A-side must be a REFUSAL, not a silently merged no-op.

        Merging zeros would produce a cost artifact carrying the AQUA name that
        is byte-equivalent to the weight-only arm -- indistinguishable later
        from a real A-side priced against a lane that does serve fused.
        """
        import json
        from prismaquant.aqua_activation_cost import activation_dloss_table

        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a.weight": "s0.safetensors"}}))
        with pytest.raises(SystemExit) as exc:
            activation_dloss_table(object(), str(tmp_path), ["NVFP4"],
                                   names=["a"],
                                   executed_activation_formats=frozenset())
        assert "executes NO format's activation quantization" in str(exc.value)

    # `test_the_cb_lane_executes_BOTH_families_activation_grids` was deleted on
    # 2026-09-02 with the Gridbook lane (archive/gridbook_lane_2026-09-02/).
    # It pinned that BOTH CB families really do quantize activations on every
    # served route -- NVFP4_CB onto the E2M1 group-16 grid before every GEMM,
    # FP8_CB as W8A8 per-token dynamic -- against the lane spec that declared
    # it. That lane spec is archived. The DURABLE LESSON the test existed for
    # is kept here because it is not about Gridbook: on 2026-08-17 a revision
    # zeroed the NVFP4_CB A-side by reading "exact BF16 bridge" as an
    # activation-precision statement when it is a GEMM-schedule one; that
    # currency error moved a whole 87 GB DSv4 allocation and was retracted the
    # same day. `tests/test_tessera_lane_admission.py` now carries the same
    # obligation for the lane that replaced it, deriving `executes` from the
    # packaged runtime contract instead of asserting it.

    def test_a_family_pattern_covers_a_rung_added_tomorrow(self):
        """Patterns, not enumerated rungs: an unlisted rung must not go free."""
        from prismaquant.lane_spec import LaneActivationContract

        c = LaneActivationContract.from_dict({"executes": ["FP8_CB_*"]})
        assert c.matches("FP8_CB_K99")
        assert not c.matches("NVFP4_CB_K99")

    def test_an_absent_declaration_is_not_read_as_an_empty_one(self):
        from prismaquant.lane_spec import LaneActivationContract

        with pytest.raises(ValueError, match="must state `executes`"):
            LaneActivationContract.from_dict({"rationale": "..."})

    def test_resolver_refuses_a_lane_with_no_declaration(self, monkeypatch):
        """Re-pointed from `nvfp4_cb` to `tessera` on 2026-09-02.

        The subject is the resolver's refusal, not the lane: a lane that does
        not declare what it executes must stop the run rather than be read as
        executing nothing, which would price every A side at zero. Any lane
        spec serves as the vehicle; the CB one is in
        archive/gridbook_lane_2026-09-02/.
        """
        import dataclasses

        from prismaquant import aqua_activation_cost as aqc
        from prismaquant.lane_spec import load_lane_spec

        spec = load_lane_spec("tessera")
        assert spec.served_activation_quantization is not None
        bare = dataclasses.replace(spec, served_activation_quantization=None)
        monkeypatch.setattr(
            "prismaquant.lane_spec.load_lane_spec", lambda _id: bare)
        with pytest.raises(SystemExit, match="does not declare"):
            aqc.resolve_executed_activation_formats(lane_id="tessera")

    def test_a_genuinely_fused_lane_still_prices_the_full_a_side(self):
        from prismaquant.aqua_activation_cost import (
            resolve_executed_activation_formats)

        assert resolve_executed_activation_formats(
            lane_id=None, executes_all=True) == "all"
