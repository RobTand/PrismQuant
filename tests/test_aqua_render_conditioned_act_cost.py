"""The render-conditioned A-side: opt-in, byte-identical off, real when on.

WHY THE LEVER EXISTS
--------------------
The local error of a quantized Linear decomposes as::

    W_hat x_hat - W x  =  dW x  +  W_hat dx

so the activation term belongs on the format's RENDERED ``W_hat_f``, not on the
source ``W``. Evaluating it there absorbs the ``dW dx`` contribution that the
source-weight form drops. What stays dropped either way -- and what these tests
therefore do NOT claim to cover -- is the cross-correlation between ``dW x`` and
``W_hat dx``, downstream nonlinearities, and MoE routing interactions.

WHAT THESE TESTS PIN
--------------------
Behaviour, never source text. The three things that can silently go wrong:

  1. The default path moves. AQUA's numbers feed the DP, and a lever that
     perturbs them when unset would re-price every future artifact. Pinned
     against values captured BEFORE the lever existed.
  2. The lever claims a basis it did not use. A ``compensated`` run that quietly
     re-rendered RTN, or a cache that was loaded and ignored, would stamp a
     production basis onto a number that never saw the production recipe --
     the rendering confound the one-cache rule exists to prevent. Pinned by
     planting a DISTINCTIVE tensor in the cache: only a run that actually read
     the cache can reproduce its price.
  3. A miss reads as free. ``cost_entry_act_dloss`` defaults to 0.0, so an
     unpriced activation-quantizing format is indistinguishable from a free one
     unless the miss surfaces. Pinned on the hole/`render_misses` reporting and
     on the zero-coverage refusal.
"""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from prismaquant import aqua_activation_cost as aqc  # noqa: E402
from prismaquant import sensitivity_card as sc  # noqa: E402
from prismaquant.format_cost_protocol import (  # noqa: E402
    activation_dloss, price_activation_only)
from prismaquant.format_cost_registry import RegistryFormatPlugin  # noqa: E402

N_TOK = 128

# Captured from `scratch/aqua_render_basis/baseline.py` on the commit BEFORE
# the lever existed (DP-visible sha256
# 37e13dbe77e2d8ef308b74e8d6478c1bd6fb4a4afa88ab2078f60f17707aa1ad). These are
# the numbers that reach the DP; if the default path ever moves, they move.
BASELINE_ACT_DLOSS = {
    "model.layers.0.self_attn.q_proj": {
        "NVFP4": 2.8745833578739385e-09,
        "FP8_E4M3": 2.0717665164388262e-10},
    "model.layers.0.mlp.gate_proj": {
        "NVFP4": 8.90666699881625e-09,
        "FP8_E4M3": 6.339622982796412e-10},
    "model.layers.0.mlp.down_proj": {
        "NVFP4": 4.8619221175661174e-09,
        "FP8_E4M3": 3.705180376353717e-10},
    "model.layers.1.mlp.up_proj": {
        "NVFP4": 5.843507952618265e-09,
        "FP8_E4M3": 4.370690597404145e-10},
}
BASELINE_SHAPES = {
    "model.layers.0.self_attn.q_proj": (64, 64),
    "model.layers.0.mlp.gate_proj": (128, 64),
    "model.layers.0.mlp.down_proj": (64, 128),
    "model.layers.1.mlp.up_proj": (96, 64),
}
BASELINE_FORMATS = ["NVFP4", "FP8_E4M3", "BF16"]


def _baseline_fixture(tmp_path):
    """The exact fixture the pre-lever baseline was captured on.

    Seeded and shape-pinned: the assertion is on absolute floats, so the
    generator, the outlier injection and the dtype are all load-bearing.
    """
    from safetensors.torch import save_file

    rng = np.random.default_rng(11)
    model = tmp_path / "model"
    model.mkdir(parents=True, exist_ok=True)
    tensors, units = {}, []
    for i, (name, (o, n)) in enumerate(BASELINE_SHAPES.items()):
        w = rng.normal(0.0, 0.02, (o, n)).astype(np.float32)
        w[0, :4] *= 25.0
        tensors[name + ".weight"] = torch.from_numpy(w).to(torch.bfloat16)
        units.append(sc.SensitivityUnit(
            topology=sc.UnitTopology(name=name, layer_index=i // 3,
                                     role="down", source_dtype="bfloat16"),
            out_features=o, in_features=n, n_params=o * n, n_tokens=N_TOK,
            h_trace_raw=1.0 + i, h_w2_sum_raw=1e-3,
            w_norm_sq=float((w ** 2).sum()), w_max_abs=float(abs(w).max()),
            g_sq_sum=rng.uniform(1e-4, 1e-2, o).astype(np.float64),
            act_sq_sum=rng.uniform(0.5, 1.5, n).astype(np.float64),
            act_absmax=rng.uniform(2.0, 6.0, n).astype(np.float64),
        ))
    save_file(tensors, str(model / "model.safetensors"))
    (model / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model.safetensors" for k in tensors}}))
    prov = sc.CardProvenance(model_id="fixture", calib_hash="d" * 64,
                             n_calib_samples=1, seq_len=N_TOK,
                             probe_commit="0" * 40,
                             render_basis=sc.RenderBasis.RTN)
    return sc.SensitivityCard(prov, units), str(model)


def _unit(n_in=64, n_out=32, seed=7):
    rng = np.random.default_rng(seed)
    return sc.SensitivityUnit(
        topology=sc.UnitTopology(name="probe.linear", layer_index=0,
                                 role="down", source_dtype="bfloat16"),
        out_features=n_out, in_features=n_in, n_params=n_in * n_out,
        n_tokens=N_TOK, h_trace_raw=1.0, h_w2_sum_raw=1e-3,
        w_norm_sq=1.0, w_max_abs=0.1,
        g_sq_sum=rng.uniform(1e-4, 1e-2, n_out).astype(np.float64),
        act_sq_sum=rng.uniform(0.5, 1.5, n_in).astype(np.float64),
        act_absmax=rng.uniform(2.0, 6.0, n_in).astype(np.float64),
    )


def _plugin(fmt, unit):
    try:
        return RegistryFormatPlugin.build(
            fmt, shape=(unit.out_features, unit.in_features), device="cpu")
    except Exception as exc:                                  # pragma: no cover
        pytest.skip(f"{fmt} unbuildable on CPU: {exc}")


# --------------------------------------------------------------- P6: unset

def test_unset_reproduces_the_pre_lever_numbers(tmp_path, monkeypatch):
    """The default path is byte-identical to before the lever existed.

    Absolute floats, not a self-consistency check: a self-comparison would pass
    while every number moved together.
    """
    monkeypatch.delenv(aqc.ACT_WEIGHT_BASIS_ENV, raising=False)
    card, model = _baseline_fixture(tmp_path)
    table, holes, meta = aqc.activation_dloss_table(
        card, model, BASELINE_FORMATS, device="cpu",
        names=list(BASELINE_SHAPES), executed_activation_formats="all")
    assert not holes
    assert meta["act_weight_basis"] == aqc.ACT_WEIGHT_BASIS_SOURCE
    assert meta["act_weight_basis_is_render_conditioned"] is False
    for name, want in BASELINE_ACT_DLOSS.items():
        assert set(table[name]) == set(want), (
            "the set of priced formats is part of the default path too")
        for fmt, value in want.items():
            assert table[name][fmt] == pytest.approx(value, rel=0, abs=0), (
                f"{name}/{fmt} moved: the A-side default path must be "
                f"byte-identical to the pre-lever behaviour")


def test_price_activation_only_is_unchanged_without_the_new_argument():
    """The seam's default must not depend on the new keyword existing."""
    unit = _unit()
    w = np.random.default_rng(0).normal(0, 0.02, (32, 64)).astype(np.float32)
    plugin = _plugin("NVFP4", unit)
    assert price_activation_only(unit, w, plugin) == pytest.approx(
        price_activation_only(unit, w, plugin, rendered_weight=None),
        rel=0, abs=0)


# ------------------------------------------------- the term actually changes

def test_rendered_weight_is_the_reduction_the_formula_names():
    """E_a on W_hat equals the hand-computed sum, not merely 'some number'.

    ``0.5/T * sum_o g[o] * sum_j W_hat[o,j]^2 * nu[j]`` -- computed here from
    the same nu the plugin measured, so the only thing under test is which
    weight entered the reduction.
    """
    unit = _unit()
    rng = np.random.default_rng(5)
    w = rng.normal(0.0, 0.05, (unit.out_features, unit.in_features))
    w = w.astype(np.float32)
    plugin = _plugin("NVFP4", unit)
    var = plugin.activation_error_variance(unit)
    assert var is not None

    w_hat = plugin.render(w).to(torch.float32).numpy()
    want = 0.5 / unit.n_tokens * float(
        np.asarray(unit.g_sq_sum, dtype=np.float64)
        @ ((w_hat.astype(np.float64) ** 2) @ np.asarray(var, dtype=np.float64)))
    got = price_activation_only(unit, w, plugin, act_var=var,
                                rendered_weight=w_hat)
    assert got == pytest.approx(want, rel=1e-9)

    # ... and it is NOT the source-weight number, i.e. the basis is real.
    src = price_activation_only(unit, w, plugin, act_var=var)
    assert got != src
    # A tiny relative move is the honest expectation (W_hat ~ W); the point of
    # the assertion is that it moved at all and in the size the render implies.
    assert abs(got / src - 1.0) < 0.25


def test_the_variance_is_still_fitted_on_the_source_weight_path():
    """nu is a property of the INPUT distribution, never of the weights.

    Pinned with a plugin whose variance model would raise if it were handed the
    rendered tensor instead of the unit.
    """
    unit = _unit()
    w = np.full((unit.out_features, unit.in_features), 0.01, dtype=np.float32)
    seen = []

    class _Desc:
        quantizes_activations = True

    class _Recording:
        descriptor = _Desc()

        def activation_error_variance(self, u):
            seen.append(u.topology.name)
            return np.full(u.in_features, 1e-4, dtype=np.float64)

    got = price_activation_only(unit, w, _Recording(),
                                rendered_weight=w * 2.0)
    assert seen == [unit.topology.name]
    # The reduction is quadratic in the weight it is given: doubling W_hat
    # quadruples the term, which is the arithmetic signature of "W_hat entered
    # the sum" rather than "the number changed somehow".
    assert got == pytest.approx(
        4.0 * price_activation_only(unit, w, _Recording()), rel=1e-12)


def test_plugin_render_is_the_formats_own_quantizer():
    """``render`` must BE the registry render, not a parallel model of it.

    Asserted against ``weight_error``: the squared difference between the
    source and the rendered tensor has to reproduce, elementwise, the error the
    plugin reports -- which it computes through its own ``quantize_dequantize``.
    """
    unit = _unit()
    w = np.random.default_rng(9).normal(
        0.0, 0.05, (unit.out_features, unit.in_features)).astype(np.float32)
    for fmt in ("NVFP4", "FP8_E4M3"):
        plugin = _plugin(fmt, unit)
        w_hat = plugin.render(w).to(torch.float32).numpy()
        w_bf16 = torch.as_tensor(w, dtype=torch.bfloat16).float().numpy()
        assert np.allclose((w_bf16 - w_hat) ** 2, plugin.weight_error(unit, w),
                           rtol=0, atol=0), (
            f"{fmt}: render() and weight_error() disagree, so one of them is "
            f"not the format's own quantizer")


def test_a_passthrough_format_renders_exactly_its_input():
    """Lossless by construction: float noise must not decide a comparison."""
    unit = _unit()
    w = np.random.default_rng(2).normal(
        0.0, 0.05, (unit.out_features, unit.in_features)).astype(np.float32)
    plugin = _plugin("BF16", unit)
    got = plugin.render(w).to(torch.float32).numpy()
    assert np.array_equal(got, torch.as_tensor(
        w, dtype=torch.bfloat16).float().numpy())


# ------------------------------------------------------- the lever's contract

def test_env_lever_selects_the_basis(tmp_path, monkeypatch):
    monkeypatch.setenv(aqc.ACT_WEIGHT_BASIS_ENV, "rtn")
    card, model = _baseline_fixture(tmp_path)
    table, holes, meta = aqc.activation_dloss_table(
        card, model, BASELINE_FORMATS, device="cpu",
        names=list(BASELINE_SHAPES), executed_activation_formats="all")
    assert meta["act_weight_basis"] == "rtn"
    assert meta["act_weight_basis_is_render_conditioned"] is True
    assert meta["render_hits"] == {"NVFP4": 4, "FP8_E4M3": 4}
    assert meta["render_misses"] == {}
    assert not holes
    moved = [n for n in BASELINE_ACT_DLOSS
             if table[n]["NVFP4"] != BASELINE_ACT_DLOSS[n]["NVFP4"]]
    assert moved, ("the rtn basis must actually change the priced A-side; "
                   "identical numbers would mean the render never ran")


@pytest.mark.parametrize("bad", ["", "   "])
def test_an_empty_lever_is_the_default_not_an_error(bad, monkeypatch):
    monkeypatch.setenv(aqc.ACT_WEIGHT_BASIS_ENV, bad)
    assert aqc.resolve_act_weight_basis() == aqc.ACT_WEIGHT_BASIS_SOURCE


@pytest.mark.parametrize("bad", ["gptq", "production", "RTN2", "1"])
def test_a_malformed_lever_is_a_hard_error(bad, monkeypatch):
    """Never a silent fall back: a typo must not report a basis it did not use."""
    monkeypatch.setenv(aqc.ACT_WEIGHT_BASIS_ENV, bad)
    with pytest.raises(SystemExit, match="act-weight-basis"):
        aqc.resolve_act_weight_basis()


def test_compensated_without_a_cache_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv(aqc.ACT_WEIGHT_BASIS_ENV, raising=False)
    card, model = _baseline_fixture(tmp_path)
    with pytest.raises(SystemExit, match="production-cache"):
        aqc.activation_dloss_table(
            card, model, BASELINE_FORMATS, device="cpu",
            names=list(BASELINE_SHAPES), executed_activation_formats="all",
            act_weight_basis="compensated")


def test_a_cache_on_a_non_compensated_basis_refuses(tmp_path, monkeypatch):
    """A loaded-and-ignored cache would stamp a production path on an RTN number."""
    monkeypatch.delenv(aqc.ACT_WEIGHT_BASIS_ENV, raising=False)
    from prismaquant.production_weight_cache import ProductionWeightCache

    card, model = _baseline_fixture(tmp_path)
    with pytest.raises(SystemExit, match="never reads it"):
        aqc.activation_dloss_table(
            card, model, BASELINE_FORMATS, device="cpu",
            names=list(BASELINE_SHAPES), executed_activation_formats="all",
            act_weight_basis="rtn",
            render_cache=ProductionWeightCache(weights={}, levers={}))


# ------------------------------------------------- the compensated cache path

def _planted_cache(card, names, fmt="NVFP4"):
    """A cache holding a tensor NOTHING could re-derive from the source.

    That is the whole point: if the priced number matches this tensor's
    reduction, the run read the cache. If it matches any render of the source
    weight, it did not.
    """
    from prismaquant.production_weight_cache import ProductionWeightCache

    weights = {}
    for i, name in enumerate(names):
        u = card[name]
        planted = torch.full((u.out_features, u.in_features),
                             0.125 * (i + 1), dtype=torch.float32)
        weights[(name, fmt)] = planted
    return ProductionWeightCache(weights=weights, levers={})


def test_the_compensated_basis_reads_the_cache_it_was_given(tmp_path,
                                                            monkeypatch):
    monkeypatch.delenv(aqc.ACT_WEIGHT_BASIS_ENV, raising=False)
    card, model = _baseline_fixture(tmp_path)
    names = list(BASELINE_SHAPES)
    cache = _planted_cache(card, names)
    table, holes, meta = aqc.activation_dloss_table(
        card, model, ["NVFP4"], device="cpu", names=names,
        executed_activation_formats="all",
        act_weight_basis="compensated", render_cache=cache)

    assert meta["act_weight_basis"] == "compensated"
    assert meta["render_hits"] == {"NVFP4": len(names)}
    assert not holes
    plugin = _plugin("NVFP4", card[names[0]])
    for name in names:
        unit = card[name]
        pl = RegistryFormatPlugin.build(
            "NVFP4", shape=(unit.out_features, unit.in_features), device="cpu")
        var = pl.activation_error_variance(unit)
        planted = cache.get(name, "NVFP4").numpy().astype(np.float64)
        want = activation_dloss(unit, planted, var)
        assert table[name]["NVFP4"] == pytest.approx(want, rel=1e-9), (
            f"{name}: the priced A-side does not match a reduction over the "
            f"PLANTED cache tensor, so the cache was not what was used")
    del plugin


def test_a_cache_miss_is_a_reported_hole_not_a_free_row(tmp_path, monkeypatch):
    """An unpriced A-side reads as 0.0 to the DP, so a miss must surface."""
    monkeypatch.delenv(aqc.ACT_WEIGHT_BASIS_ENV, raising=False)
    card, model = _baseline_fixture(tmp_path)
    names = list(BASELINE_SHAPES)
    cache = _planted_cache(card, names[:2])
    table, holes, meta = aqc.activation_dloss_table(
        card, model, ["NVFP4"], device="cpu", names=names,
        executed_activation_formats="all",
        act_weight_basis="compensated", render_cache=cache)
    assert meta["render_hits"] == {"NVFP4": 2}
    assert meta["render_misses"] == {"NVFP4": 2}
    assert len(holes["NVFP4"]) == 2
    for missing in names[2:]:
        assert missing not in table, (
            "a unit whose production render is missing must carry NO A-side "
            "rather than one silently re-rendered on another basis")


def test_zero_render_coverage_refuses_instead_of_writing_a_no_op(tmp_path,
                                                                monkeypatch):
    monkeypatch.delenv(aqc.ACT_WEIGHT_BASIS_ENV, raising=False)
    from prismaquant.production_weight_cache import ProductionWeightCache

    card, model = _baseline_fixture(tmp_path)
    with pytest.raises(SystemExit, match="rendered 0"):
        aqc.activation_dloss_table(
            card, model, ["NVFP4"], device="cpu",
            names=list(BASELINE_SHAPES), executed_activation_formats="all",
            act_weight_basis="compensated",
            render_cache=ProductionWeightCache(weights={}, levers={}))


def test_a_wrong_shaped_cache_entry_raises(tmp_path, monkeypatch):
    """Two different tensors under one name is not a basis question."""
    monkeypatch.delenv(aqc.ACT_WEIGHT_BASIS_ENV, raising=False)
    from prismaquant.production_weight_cache import ProductionWeightCache

    card, model = _baseline_fixture(tmp_path)
    name = list(BASELINE_SHAPES)[0]
    cache = ProductionWeightCache(
        weights={(name, "NVFP4"): torch.ones(8, 8, dtype=torch.float32)},
        levers={})
    with pytest.raises(RuntimeError, match="different tensors"):
        aqc.activation_dloss_table(
            card, model, ["NVFP4"], device="cpu", names=[name],
            executed_activation_formats="all",
            act_weight_basis="compensated", render_cache=cache)


# ------------------------------------------------------------- the provenance

def test_the_modelled_variance_is_stamped_as_an_approximation(tmp_path,
                                                              monkeypatch):
    """Second moments cannot represent a clipping tail, and must say so.

    The stage has two variance paths and only one of them is a measurement;
    a provenance that did not distinguish them would let a Gaussian fit be
    cited as a measured activation cost.
    """
    monkeypatch.delenv(aqc.ACT_WEIGHT_BASIS_ENV, raising=False)
    card, model = _baseline_fixture(tmp_path)
    _, _, meta = aqc.activation_dloss_table(
        card, model, BASELINE_FORMATS, device="cpu",
        names=list(BASELINE_SHAPES), executed_activation_formats="all")
    assert meta["act_var_source"] == {"modelled": 8}, (
        "no cached activations were supplied, so every variance is modelled")
    assert meta["act_var_paths"]["modelled"].startswith("APPROXIMATION")
    assert meta["act_var_paths"]["modelled_per_expert"].startswith(
        "APPROXIMATION")
    assert not meta["act_var_paths"]["measured"].startswith("APPROXIMATION")
    assert "cross-correlation" in meta["act_weight_basis_dropped_terms"]
    assert "routing" in meta["act_weight_basis_dropped_terms"]


def test_the_measured_path_wins_and_is_counted(tmp_path, monkeypatch):
    """Real cached rows must replace the Gaussian fit, and be reported as such.

    The QDQ replay is the only path that can see a clipping tail or the joint
    across the channels sharing one block scale, so which path a run took is a
    provenance-grade fact rather than a log line.
    """
    monkeypatch.delenv(aqc.ACT_WEIGHT_BASIS_ENV, raising=False)
    card, model = _baseline_fixture(tmp_path)
    act_dir = tmp_path / "act"
    act_dir.mkdir()
    rng = np.random.default_rng(4)
    names = list(BASELINE_SHAPES)
    for name in names:
        n_in = card[name].in_features
        rows = torch.from_numpy(
            rng.standard_normal((64, n_in)).astype(np.float32))
        # A hard outlier in one channel: exactly the structure a second moment
        # averages away and a QDQ replay sees.
        rows[:, 0] *= 50.0
        torch.save({"inputs": rows},
                   aqc.cached_act_path(str(act_dir), name))
    _, _, meta = aqc.activation_dloss_table(
        card, model, ["NVFP4"], device="cpu", names=names,
        act_dir=str(act_dir), executed_activation_formats="all")
    assert meta["act_var_source"] == {"measured": len(names)}, (
        "cached rows were present for every unit, so nothing may be modelled")


# ------------------------------------------------------------ packed experts

def _packed_fixture(tmp_path, n_e=3, o=32, n=64):
    """A packed [E, M, N] routed-expert unit: ONE decision unit, E matrices.

    The packed parameter is a bare 3-D tensor with NO ``.weight`` suffix, which
    is how a real checkpoint stores it and is exactly the shape both rendered
    bases have to handle without collapsing the experts together.
    """
    from safetensors.torch import save_file

    rng = np.random.default_rng(21)
    name = "model.layers.0.mlp.experts.down_proj"
    w = rng.normal(0.0, 0.02, (n_e, o, n)).astype(np.float32)
    w[:, 0, :4] *= 20.0
    model = tmp_path / "packed"
    model.mkdir(parents=True, exist_ok=True)
    save_file({name: torch.from_numpy(w).to(torch.bfloat16)},
              str(model / "model.safetensors"))
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors"}}))
    unit = sc.SensitivityUnit(
        topology=sc.UnitTopology(name=name, layer_index=0, role="down",
                                 source_dtype="bfloat16"),
        out_features=o, in_features=n, n_params=n_e * o * n, n_tokens=N_TOK,
        h_trace_raw=1.0, h_w2_sum_raw=1e-3,
        w_norm_sq=float((w ** 2).sum()), w_max_abs=float(abs(w).max()),
        expert_g_sq_sum=rng.uniform(1e-4, 1e-2, (n_e, o)).astype(np.float64),
        expert_act_sq_sum=rng.uniform(0.5, 1.5, (n_e, n)).astype(np.float64),
        expert_act_absmax=rng.uniform(2.0, 6.0, (n_e, n)).astype(np.float64),
        expert_tokens=np.array([40.0, 60.0, 28.0], dtype=np.float64),
    )
    prov = sc.CardProvenance(model_id="fixture", calib_hash="e" * 64,
                             n_calib_samples=1, seq_len=N_TOK,
                             probe_commit="0" * 40,
                             render_basis=sc.RenderBasis.RTN)
    # bf16 round-trip: the stage prices the checkpoint tensor, not the fp32
    # draw, so hand the tests the same array it will see.
    stored = torch.from_numpy(w).to(torch.bfloat16).float().numpy()
    return sc.SensitivityCard(prov, [unit]), str(model), name, stored


def test_a_packed_unit_renders_per_expert_on_the_rtn_basis(tmp_path,
                                                           monkeypatch):
    """E matrices, E renders -- never one collapsed render of the stack.

    The exporter derives a per-tensor scale per expert, so rendering the pack
    as one [E*M, N] matrix would price a rendering nothing ships. Pinned by
    reproducing the priced value from an explicitly per-expert render, and by
    showing the collapsed render is a different number.
    """
    monkeypatch.delenv(aqc.ACT_WEIGHT_BASIS_ENV, raising=False)
    card, model, name, w = _packed_fixture(tmp_path)
    table, holes, meta = aqc.activation_dloss_table(
        card, model, ["NVFP4"], device="cpu", names=[name],
        executed_activation_formats="all", act_weight_basis="rtn")
    assert not holes
    assert meta["render_hits"] == {"NVFP4": 1}

    unit = card[name]
    assert unit.n_experts == w.shape[0]
    plugin = _plugin("NVFP4", unit)
    var = plugin.expert_activation_error_variance(unit)
    assert var is not None
    per_expert = np.stack([
        plugin.render(w[e]).to(torch.float32).numpy() for e in range(len(w))])
    want = activation_dloss(unit, per_expert, var)
    assert table[name]["NVFP4"] == pytest.approx(want, rel=1e-9)

    collapsed = plugin.render(w.reshape(-1, unit.in_features))
    collapsed = collapsed.to(torch.float32).numpy().reshape(w.shape)
    assert activation_dloss(unit, collapsed, var) != pytest.approx(
        want, rel=1e-12), (
        "a collapsed render of the pack must not be what gets priced")


def test_a_packed_unit_takes_the_cache_entry_whole_on_the_compensated_basis(
        tmp_path, monkeypatch):
    """The production cache holds the pack as one [E, M, N] entry.

    Asserted, not assumed: a planted 3-D tensor must be what the price is
    reduced over, and a 2-D entry under the same name must raise rather than be
    broadcast into something plausible.
    """
    monkeypatch.delenv(aqc.ACT_WEIGHT_BASIS_ENV, raising=False)
    from prismaquant.production_weight_cache import ProductionWeightCache

    card, model, name, w = _packed_fixture(tmp_path)
    unit = card[name]
    planted = (torch.arange(w.size, dtype=torch.float32).reshape(w.shape)
               * 1e-4)
    cache = ProductionWeightCache(weights={(name, "NVFP4"): planted},
                                  levers={})
    table, holes, meta = aqc.activation_dloss_table(
        card, model, ["NVFP4"], device="cpu", names=[name],
        executed_activation_formats="all",
        act_weight_basis="compensated", render_cache=cache)
    assert not holes and meta["render_hits"] == {"NVFP4": 1}

    plugin = _plugin("NVFP4", unit)
    var = plugin.expert_activation_error_variance(unit)
    want = activation_dloss(unit, planted.numpy(), var)
    assert table[name]["NVFP4"] == pytest.approx(want, rel=1e-9)

    flat = ProductionWeightCache(
        weights={(name, "NVFP4"): torch.ones(unit.out_features,
                                             unit.in_features)}, levers={})
    with pytest.raises(RuntimeError, match="different tensors"):
        aqc.activation_dloss_table(
            card, model, ["NVFP4"], device="cpu", names=[name],
            executed_activation_formats="all",
            act_weight_basis="compensated", render_cache=flat)


def test_a_packed_unit_is_unchanged_on_the_default_basis(tmp_path,
                                                         monkeypatch):
    """The packed default path prices the SOURCE pack -- no render anywhere."""
    monkeypatch.delenv(aqc.ACT_WEIGHT_BASIS_ENV, raising=False)
    card, model, name, w = _packed_fixture(tmp_path)
    unit = card[name]
    table, _, meta = aqc.activation_dloss_table(
        card, model, ["NVFP4"], device="cpu", names=[name],
        executed_activation_formats="all")
    assert meta["act_var_source"] == {"modelled_per_expert": 1}
    assert meta["act_weight_basis_is_render_conditioned"] is False
    plugin = _plugin("NVFP4", unit)
    var = plugin.expert_activation_error_variance(unit)
    assert table[name]["NVFP4"] == pytest.approx(
        activation_dloss(unit, w, var), rel=1e-9)
