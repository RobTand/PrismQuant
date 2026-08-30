"""Packed routed-expert A-side pricing: assembly, exact chunking, honest holes.

WHY
---
A packed decision unit is ONE ``[E, out, in]`` live parameter, but many
checkpoints serialize it as ``E * len(projections)`` separate 2-D tensors.
``build_weight_resolver`` maps a unit to exactly one checkpoint key, so it
cannot express that fan-out and every such unit dropped out at resolution --
SILENTLY, because an absent ``act_dloss`` reads as 0.0 (free) to the DP. On
GLM-5.3-Flash that was the entire routed body: 84 of 288 units, and precisely
the ones whose menu has a live NVFP4-vs-FP8 choice for the A-side to move.

Three properties are load-bearing and none of them is visible in a passing
production run, which is why they are pinned here:

1. Pricing a 19.3 GiB unit in expert chunks is EXACT, not an approximation.
   ``_activation_dloss_packed`` sums over experts and then applies a
   normalization linear in that sum, so per-chunk prices sum to the whole. If
   that identity broke, the number would still look plausible.
2. An incomplete expert set REFUSES. Assembling a short tensor would price a
   real A-side too low -- the silent under-charge this stage exists to remove.
3. ``holes`` reports what is missing, exactly once, with the true reason. A
   unit that never reached the pricing loop used to be absent from the report
   entirely, so the stage wrote ``holes: {}`` while whole classes of unit were
   priced at zero by the DP.
"""
from __future__ import annotations

import dataclasses
import json
import os

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from safetensors.torch import save_file  # noqa: E402

from prismaquant.aqua_activation_cost import (  # noqa: E402
    _slice_experts, activation_dloss_table, build_packed_expert_plan,
)
from prismaquant.format_cost_protocol import price_activation_only  # noqa: E402
from prismaquant.format_cost_registry import RegistryFormatPlugin  # noqa: E402
from prismaquant.sensitivity_card import (  # noqa: E402
    SensitivityUnit, UnitTopology,
)

FORMATS = ("NVFP4", "FP8_E4M3")
PACKED = "m.layers.0.mlp.experts.gate_up_proj"
DENSE = "m.layers.0.mlp.gate_proj"
PROJS = ("gate_proj", "up_proj")
# E is deliberately coprime with every chunk size exercised below, so a
# chunking bug cannot hide behind an even split.
E, OUT_PER_PROJ, IN = 11, 24, 64
OUT = OUT_PER_PROJ * len(PROJS)


def _packed_unit(seed: int = 7) -> SensitivityUnit:
    rng = np.random.default_rng(seed)
    return SensitivityUnit(
        topology=UnitTopology(name=PACKED), out_features=OUT, in_features=IN,
        n_params=E * OUT * IN, n_tokens=4096, h_trace_raw=1.0,
        h_w2_sum_raw=1.0, w_norm_sq=1.0, w_max_abs=1.0,
        expert_g_sq_sum=rng.random((E, OUT)) * 1e-3,
        expert_act_sq_sum=rng.random((E, IN)) * 10.0,
        expert_act_absmax=rng.random((E, IN)) * 3.0 + 0.5,
        expert_tokens=rng.integers(1, 500, size=E).astype(np.float64))


def _dense_unit(seed: int = 11) -> SensitivityUnit:
    rng = np.random.default_rng(seed)
    return SensitivityUnit(
        topology=UnitTopology(name=DENSE), out_features=OUT, in_features=IN,
        n_params=OUT * IN, n_tokens=4096, h_trace_raw=1.0, h_w2_sum_raw=1.0,
        w_norm_sq=1.0, w_max_abs=1.0,
        g_sq_sum=rng.random(OUT) * 1e-3,
        act_sq_sum=rng.random(IN) * 10.0,
        act_absmax=rng.random(IN) * 3.0 + 0.5)


class _Card:
    def __init__(self, units):
        self._u = {u.topology.name: u for u in units}

    def __getitem__(self, name):
        return self._u[name]

    def units(self):
        return list(self._u.values())


class _Profile:
    """The structural declarations the stage is required to read.

    Nothing here is a name pattern the stage guessed: the parent name and the
    projection order both come from the profile (principle 2).
    """

    def packed_expert_param_names(self):
        return ["gate_up_proj"]

    def packed_expert_projection_names(self, parent):
        assert parent == "gate_up_proj", parent
        return list(PROJS)

    def checkpoint_to_live_name(self, key):
        return key


def _write_checkpoint(root, *, per_expert=True, dense=False, drop=None):
    """A real safetensors checkpoint whose routed experts are unfused."""
    tensors, weight_map = {}, {}
    if per_expert:
        for e in range(E):
            for p in PROJS:
                if drop is not None and (e, p) == drop:
                    continue
                k = f"m.layers.0.mlp.experts.{e}.{p}.weight"
                tensors[k] = torch.randn(OUT_PER_PROJ, IN,
                                         dtype=torch.float32) * 0.02
                weight_map[k] = "s0.safetensors"
    if dense:
        k = DENSE + ".weight"
        tensors[k] = torch.randn(OUT, IN, dtype=torch.float32) * 0.02
        weight_map[k] = "s0.safetensors"
    save_file(tensors, os.path.join(root, "s0.safetensors"))
    (root_index := os.path.join(root, "model.safetensors.index.json"))
    with open(root_index, "w") as fh:
        json.dump({"weight_map": weight_map}, fh)
    return weight_map


# --------------------------------------------------------------------------
# 1. chunk-exactness against the canonical pricing function
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("chunk", [1, 3, 5, E])
def test_expert_chunked_price_equals_whole_tensor_price(fmt, chunk):
    """Summing per-chunk prices must reproduce the single whole-tensor price.

    This is what lets a 19.3 GiB packed unit be priced in ~1 GiB slabs instead
    of not at all. It holds because ``_activation_dloss_packed`` sums over
    experts BEFORE a normalization that is linear in that sum; if a future
    change moved any per-expert term behind a nonlinearity the sum would
    silently stop meaning the same quantity.
    """
    import dataclasses

    unit = _packed_unit()
    w = (np.random.default_rng(3).standard_normal((E, OUT, IN))
         * 0.02).astype(np.float32)
    plugin = RegistryFormatPlugin.build(fmt, shape=(OUT, IN), device="cpu")
    whole = price_activation_only(unit, w, plugin)
    assert whole is not None and whole > 0.0

    total = 0.0
    for lo in range(0, E, chunk):
        ids = list(range(lo, min(lo + chunk, E)))
        sub = dataclasses.replace(
            unit,
            expert_g_sq_sum=_slice_experts(unit.expert_g_sq_sum, ids),
            expert_act_sq_sum=_slice_experts(unit.expert_act_sq_sum, ids),
            expert_act_absmax=_slice_experts(unit.expert_act_absmax, ids),
            expert_tokens=_slice_experts(unit.expert_tokens, ids))
        total += price_activation_only(sub, w[ids], plugin)
    rel = abs(total - whole) / whole
    assert rel < 1e-12, (
        f"{fmt} chunk={chunk}: chunked {total!r} vs whole {whole!r} "
        f"(rel {rel:.3e}); the per-chunk sum must BE the whole-tensor price, "
        f"not approximate it")


def test_slice_experts_passes_none_through():
    """A card row a model does not carry must stay absent, not become empty."""
    assert _slice_experts(None, [0, 1]) is None
    got = _slice_experts(np.arange(12).reshape(4, 3), [1, 3])
    assert got.tolist() == [[3, 4, 5], [9, 10, 11]]


# --------------------------------------------------------------------------
# 2. the estimator is named, not inferred, per chunk
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", FORMATS)
def test_every_chunk_is_priced_by_the_same_named_estimator(fmt):
    """Chunks of one unit priced by two estimators do not sum to anything.

    ``price_activation_only`` picks its variance source by fall-through
    (measured rows, then the plugin's per-expert fit, then the analytic grid
    model). The chunk loop therefore names ``expert_activation_error_variance``
    explicitly. Pinning it here means a plugin that stops offering that method
    fails this test rather than silently mixing two quantities across the
    chunks of one unit.
    """
    import dataclasses

    unit = _packed_unit()
    w = (np.random.default_rng(3).standard_normal((E, OUT, IN))
         * 0.02).astype(np.float32)
    plugin = RegistryFormatPlugin.build(fmt, shape=(OUT, IN), device="cpu")
    measure = getattr(plugin, "expert_activation_error_variance", None)
    assert callable(measure), (
        f"{fmt}: the packed chunk loop requires this estimator; without it "
        f"the loop must produce a HOLE, never a partial total")

    whole = price_activation_only(unit, w, plugin, act_var=measure(unit))
    total = 0.0
    for lo in range(0, E, 4):
        ids = list(range(lo, min(lo + 4, E)))
        sub = dataclasses.replace(
            unit,
            expert_g_sq_sum=_slice_experts(unit.expert_g_sq_sum, ids),
            expert_act_sq_sum=_slice_experts(unit.expert_act_sq_sum, ids),
            expert_act_absmax=_slice_experts(unit.expert_act_absmax, ids),
            expert_tokens=_slice_experts(unit.expert_tokens, ids))
        total += price_activation_only(sub, w[ids], plugin,
                                       act_var=measure(sub))
    assert abs(total - whole) / whole < 1e-12


# --------------------------------------------------------------------------
# 3. the plan comes from the profile, and an incomplete set refuses
# --------------------------------------------------------------------------
def test_plan_uses_the_profiles_declared_projection_order(tmp_path):
    weight_map = _write_checkpoint(str(tmp_path))
    plan = build_packed_expert_plan(
        weight_map, _Card([_packed_unit()]), [PACKED], _Profile())
    assert set(plan) == {PACKED}
    assert len(plan[PACKED]) == E
    for e in range(E):
        assert list(plan[PACKED][e]) == list(PROJS), (
            "projection order must be the profile's, not dict insertion order "
            "-- it decides the output-axis concatenation of the slab")


def test_no_plan_without_a_profile(tmp_path):
    """Withholding the profile removes the aliases AND the assembly."""
    weight_map = _write_checkpoint(str(tmp_path))
    assert build_packed_expert_plan(
        weight_map, _Card([_packed_unit()]), [PACKED], None) == {}


def test_incomplete_expert_set_raises_instead_of_assembling_short(tmp_path):
    """A packed weight missing experts prices a real A-side too low.

    That is the silent under-charge this whole stage exists to remove, so the
    only acceptable outcome is a refusal that names the unit and the expert.
    """
    weight_map = _write_checkpoint(str(tmp_path), drop=(5, "up_proj"))
    with pytest.raises(RuntimeError) as exc:
        build_packed_expert_plan(
            weight_map, _Card([_packed_unit()]), [PACKED], _Profile())
    msg = str(exc.value)
    assert PACKED in msg and "expert 5" in msg, msg


# --------------------------------------------------------------------------
# 4. holes: not under-reported, and not over-reported either
# --------------------------------------------------------------------------
def test_a_unit_that_never_resolved_is_recorded_in_holes(tmp_path):
    """The bite: withhold the profile so the packed unit cannot resolve.

    A passing run does not test this, because with the assembly path every
    unit resolves. Before the fix such a unit never reached the per-format
    loop, so nothing recorded it: the stage wrote ``holes: {}`` while the DP
    read its A-side as 0.0 (free). Mutate the input to force the failure --
    a bad input proves the check runs.
    """
    _write_checkpoint(str(tmp_path), dense=True)
    table, holes, _meta = activation_dloss_table(
        _Card([_packed_unit(), _dense_unit()]), str(tmp_path), list(FORMATS),
        device="cpu", names=[PACKED, DENSE], profile=None,
        executed_activation_formats="all")

    assert PACKED not in table, "the packed unit must be unresolvable here"
    assert DENSE in table, "the dense unit must still price"
    for fmt in FORMATS:
        hits = [h for h in holes[fmt] if PACKED in h]
        assert len(hits) == 1, (fmt, hits)
        assert "unresolved in checkpoint" in hits[0], hits


def test_a_unit_that_reached_the_loop_is_not_recorded_twice(tmp_path):
    """The over-report half: one failure, one entry, its own reason.

    ``table[name]`` is only set when a unit's row is non-empty, so counting
    ``wanted - table`` would record a unit that DID reach the loop and failed
    there a second time, under a reason that is false for it.
    """
    _write_checkpoint(str(tmp_path), dense=True)
    # A menu of exactly one format that cannot be built for this unit: the
    # unit RESOLVES and enters the loop, records its own hole there, and ends
    # with an empty row -- so `table` never gets it. That is the case the
    # `wanted - table` formulation double-counted.
    bogus = "NOT_A_REAL_FORMAT_XYZ"
    table, holes, _meta = activation_dloss_table(
        _Card([_dense_unit()]), str(tmp_path), [bogus],
        device="cpu", names=[DENSE], profile=None,
        executed_activation_formats="all")

    assert DENSE not in table, "an all-formats-failed unit has no row"
    hits = [h for h in holes[bogus] if DENSE in h]
    assert len(hits) == 1, (
        f"a unit that reached the pricing loop must be recorded once, with "
        f"its own reason; got {hits}")
    assert "unbuildable" in hits[0], hits
    assert "unresolved in checkpoint" not in hits[0], (
        "this unit DID resolve -- labelling it unresolved is a false reason")


# --------------------------------------------------------------------------
# 5. end to end: assembly through the stage, chunk-size independent
# --------------------------------------------------------------------------
def test_packed_unit_prices_end_to_end_and_is_chunk_size_independent(tmp_path):
    """Covers ``build_packed_expert_plan`` + ``_assemble_expert_slab`` + the
    chunk loop together, on a real per-expert safetensors checkpoint.

    The chunk sizes straddle E (11) so no split is even, and 48 exercises the
    single-chunk path. Equality is asserted at the objective level: any drift
    here is a mispriced A-side reaching the DP.
    """
    _write_checkpoint(str(tmp_path))
    card = _Card([_packed_unit()])
    seen = {}
    for chunk in (1, 3, 16, 48):
        table, holes, _meta = activation_dloss_table(
            card, str(tmp_path), list(FORMATS), device="cpu", names=[PACKED],
            profile=_Profile(), executed_activation_formats="all",
            experts_per_chunk=chunk)
        assert PACKED in table, (chunk, holes)
        assert not any(PACKED in h for v in holes.values() for h in v)
        seen[chunk] = dict(table[PACKED])

    ref = seen[1]
    for fmt in FORMATS:
        assert ref[fmt] > 0.0
        for chunk, got in seen.items():
            rel = abs(got[fmt] - ref[fmt]) / ref[fmt]
            assert rel < 1e-12, (
                f"{fmt}: chunk={chunk} gave {got[fmt]!r} vs chunk=1 "
                f"{ref[fmt]!r} (rel {rel:.3e}); float64 addition order is the "
                f"only difference the chunk size is allowed to make")


def test_packed_var_source_is_the_per_expert_fit(tmp_path):
    """Every packed chunk must be counted as ``modelled_per_expert``.

    On GLM the readout was ``{measured: 408, modelled_per_expert: 168}`` and
    168 is exactly 84 units x 2 formats -- the evidence that no chunk fell
    through to the dense measure or the analytic model.
    """
    _write_checkpoint(str(tmp_path))
    table, _holes, meta = activation_dloss_table(
        _Card([_packed_unit()]), str(tmp_path), list(FORMATS), device="cpu",
        names=[PACKED], profile=_Profile(),
        executed_activation_formats="all", experts_per_chunk=3)
    assert PACKED in table
    src = meta.get("act_var_source") or meta.get("var_source") or {}
    # Unconditional: `if src:` made this fail-OPEN -- deleting the counter
    # entirely would have satisfied the test it was written to enforce.
    assert src, (
        f"the stage must report which estimator priced each row; meta carried "
        f"no var-source census at all ({sorted(meta)})")
    assert src.get("modelled_per_expert") == len(FORMATS), src
    assert not src.get("measured"), src


# --------------------------------------------------------------------------
# 6. hardening: the stage price against an INDEPENDENT whole-tensor oracle
# --------------------------------------------------------------------------
#
# Added after an adversarial review of this file found that sections 1-5 pin
# chunk settings against EACH OTHER but never against the canonical price. A
# corruption applied once to the production total -- `total *= 2`, a dropped
# `gain`, a wrong `n_tokens` -- is identical at every chunk size, so every
# assertion above stays green while the number the DP reads is wrong. These
# tests supply the missing oracle by assembling the slab through a path that
# does not call the code under test.


def _oracle_slab(root, order):
    """[E, OUT, IN] float32, assembled WITHOUT the stage's assembly code.

    Deliberately duplicates the layout rule (concatenate projections on the
    output axis in the given order, stack experts on a new leading axis) rather
    than importing ``_assemble_expert_slab``: an oracle that calls the code
    under test cannot falsify it.
    """
    from safetensors.torch import load_file
    blob = load_file(os.path.join(root, "s0.safetensors"))
    return torch.stack([
        torch.cat([blob[f"m.layers.0.mlp.experts.{e}.{p}.weight"]
                   for p in order], dim=0)
        for e in range(E)], dim=0).to(torch.float32)


def _oracle_price(root, unit, fmt, order=PROJS):
    w = _oracle_slab(root, order).numpy()
    plugin = RegistryFormatPlugin.build(
        fmt, shape=(unit.out_features, unit.in_features), device="cpu")
    var = plugin.expert_activation_error_variance(unit)
    assert var is not None, "the per-expert estimator must be available here"
    return price_activation_only(unit, w, plugin, act_var=var)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("chunk", [1, 3, E])
def test_stage_price_equals_an_independent_whole_tensor_oracle(tmp_path, fmt,
                                                               chunk):
    """The number the stage writes IS the whole-tensor price, not merely a
    chunk-stable one.

    This is the test that fails if the packed total is scaled, if `gain` is
    dropped, or if `n_tokens` is sliced per chunk -- none of which the
    chunk-vs-chunk comparisons can see.
    """
    _write_checkpoint(str(tmp_path))
    unit = _packed_unit()
    table, holes, _meta = activation_dloss_table(
        _Card([unit]), str(tmp_path), [fmt], device="cpu", names=[PACKED],
        profile=_Profile(), executed_activation_formats="all",
        experts_per_chunk=chunk)
    assert PACKED in table, (chunk, holes)
    want = _oracle_price(str(tmp_path), unit, fmt)
    got = table[PACKED][fmt]
    assert want > 0.0
    rel = abs(got - want) / want
    assert rel < 1e-12, (
        f"{fmt} chunk={chunk}: stage wrote {got!r}, independent whole-tensor "
        f"oracle says {want!r} (rel {rel:.3e}). The stage price must BE the "
        f"canonical price, not just stable across chunk sizes.")


# --------------------------------------------------------------------------
# 7. hardening: the profile's declared order is what assembly obeys
# --------------------------------------------------------------------------
class _ReversedProfile(_Profile):
    """Declares the projections in the opposite order to the checkpoint's.

    The original order test used ``PROJS`` as both the checkpoint insertion
    order and the profile's declared order, so an implementation that hardcoded
    ``gate_proj, up_proj`` -- or that simply used dict insertion order -- passed
    it. Reversing the declaration separates the two.
    """

    def packed_expert_projection_names(self, parent):
        assert parent == "gate_up_proj", parent
        return list(PROJS[::-1])


def test_assembly_follows_the_profiles_order_not_the_checkpoints(tmp_path):
    """Permuting the declared order must permute the assembled output rows.

    Output rows are not interchangeable: ``_activation_dloss_packed`` pairs
    ``g_sq[e, o]`` with ``W[e, o, :]``, so concatenating the projections in the
    wrong order silently misprices every expert. The two orders must therefore
    give DIFFERENT prices, and each must match its own oracle.
    """
    _write_checkpoint(str(tmp_path))
    unit = _packed_unit()
    fmt = FORMATS[0]
    fwd = activation_dloss_table(
        _Card([unit]), str(tmp_path), [fmt], device="cpu", names=[PACKED],
        profile=_Profile(), executed_activation_formats="all",
        experts_per_chunk=3)[0][PACKED][fmt]
    rev = activation_dloss_table(
        _Card([unit]), str(tmp_path), [fmt], device="cpu", names=[PACKED],
        profile=_ReversedProfile(), executed_activation_formats="all",
        experts_per_chunk=3)[0][PACKED][fmt]
    assert fwd != rev, (
        "reversing the profile's declared projection order changed nothing -- "
        "the stage is not reading the profile's order (a hardcoded or "
        "insertion order would behave exactly like this)")
    for got, order in ((fwd, PROJS), (rev, PROJS[::-1])):
        want = _oracle_price(str(tmp_path), unit, fmt, order=order)
        assert abs(got - want) / want < 1e-12, (order, got, want)


# --------------------------------------------------------------------------
# 8. hardening: a wrong-shaped slab refuses (the other half of "fail loud")
# --------------------------------------------------------------------------
def test_wrong_shaped_assembled_slab_raises_instead_of_pricing(tmp_path):
    """Deleting the slab shape check must not go unnoticed.

    The incomplete-expert-set refusal was covered; this one was not, so the
    check in ``_assemble_expert_slab`` could be removed with every test green.
    A card whose ``out_features`` disagrees with what the checkpoint leaves
    actually concatenate to is the realistic form of this: a stale probe, or a
    profile declaring the wrong projection count.
    """
    _write_checkpoint(str(tmp_path))
    unit = _packed_unit()
    bad = dataclasses.replace(unit, out_features=OUT + 8)
    with pytest.raises(RuntimeError) as ei:
        activation_dloss_table(
            _Card([bad]), str(tmp_path), [FORMATS[0]], device="cpu",
            names=[PACKED], profile=_Profile(),
            executed_activation_formats="all", experts_per_chunk=3)
    msg = str(ei.value)
    assert "shape" in msg, msg
    assert str(OUT + 8) in msg or str(OUT) in msg, msg


# --------------------------------------------------------------------------
# 9. hardening: the holes exclusion covers the PACKED half too
# --------------------------------------------------------------------------
def test_a_packed_unit_that_reached_the_loop_is_not_recorded_twice(tmp_path):
    """The dense half of this was covered; the packed half was not.

    With only ``set(resolvable)`` in ``reached_pricing_loop`` -- i.e. omitting
    ``packed_plan`` -- a packed unit that failed INSIDE the loop would be
    recorded twice: once with its true reason, once as "unresolved in
    checkpoint", which is false for a unit that resolved by assembly.
    """
    _write_checkpoint(str(tmp_path))
    bogus = "NOT_A_REAL_FORMAT_XYZ"
    table, holes, _meta = activation_dloss_table(
        _Card([_packed_unit()]), str(tmp_path), [bogus], device="cpu",
        names=[PACKED], profile=_Profile(),
        executed_activation_formats="all", experts_per_chunk=3)
    assert PACKED not in table, "an all-formats-failed unit has no row"
    hits = [h for h in holes[bogus] if PACKED in h]
    assert len(hits) == 1, (
        f"a packed unit that reached the pricing loop must be recorded once; "
        f"got {hits}")
    assert "unbuildable" in hits[0], hits
    assert "unresolved in checkpoint" not in hits[0], (
        "this unit DID resolve, by per-expert assembly -- labelling it "
        "unresolved is a false reason")


# --------------------------------------------------------------------------
# 10. the variance estimator itself is exactly sliceable over experts
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fmt", FORMATS)
def test_the_variance_estimator_is_exactly_sliceable_over_experts(fmt):
    """The chunked estimator's OUTPUT equals the whole call's rows, elementwise.

    Sections 1-8 pin the chunking identity at the PRICE level, which is the
    property the DP consumes. This pins the input to that price one level
    down: ``expert_activation_error_variance`` must return, for a sliced unit,
    exactly the rows the whole-unit call returns for those experts.

    It is not free. The estimator batches experts internally, so a different
    chunk boundary is a different internal batch shape; and it synthesizes its
    rows from a pseudo-random draw, so an estimator that advanced a SHARED RNG
    per call would hand chunk 2 a different sample than the whole call did.
    The production loop's per-chunk prices would still sum to something
    self-consistent and still look plausible. They would not be the price of
    this tensor.

    Both halves are asserted: the rows match, and the call leaves torch's
    global RNG untouched (the estimator owns a seeded generator, so callers
    cannot perturb it and it cannot perturb them).
    """
    unit = _packed_unit()
    plugin = RegistryFormatPlugin.build(fmt, shape=(OUT, IN), device="cpu")
    whole = plugin.expert_activation_error_variance(unit)
    assert whole is not None and whole.shape == (E, IN), (
        f"{fmt}: the per-expert estimator must return [E, in_features]; "
        f"got {None if whole is None else whole.shape}")

    rng_before = torch.random.get_rng_state()
    seen = 0
    for chunk in (1, 3, 4, E):
        for lo in range(0, E, chunk):
            ids = list(range(lo, min(lo + chunk, E)))
            sub = dataclasses.replace(
                unit,
                expert_g_sq_sum=_slice_experts(unit.expert_g_sq_sum, ids),
                expert_act_sq_sum=_slice_experts(unit.expert_act_sq_sum, ids),
                expert_act_absmax=_slice_experts(unit.expert_act_absmax, ids),
                expert_tokens=_slice_experts(unit.expert_tokens, ids))
            part = plugin.expert_activation_error_variance(sub)
            assert part is not None, (
                f"{fmt}: estimator returned None for experts {ids} but not "
                f"for the whole unit; a chunk loop would turn one unit into a "
                f"hole depending only on the chunk size")
            np.testing.assert_array_equal(
                part, whole[ids],
                err_msg=(f"{fmt}: experts {ids} priced differently when "
                         f"measured in a chunk of {chunk} than when measured "
                         f"with all {E}. The chunked total is then not the "
                         f"whole-tensor quantity."))
            seen += 1
    assert seen == E + 4 + 3 + 1, seen

    assert torch.equal(torch.random.get_rng_state(), rng_before), (
        f"{fmt}: the estimator advanced torch's GLOBAL rng. Its draw would "
        f"then depend on how many chunks preceded it, and on whatever else "
        f"the process had drawn -- neither of which is a property of the "
        f"tensor being priced.")
