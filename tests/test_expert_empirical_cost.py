"""Empirical packed-expert unit-KL cost (the AURA-MoE hybrid's expert leg).

Pins: unit KL measured per serving unit and split across members by
n_params; FP8 stays in the menu (measured, not banned); BF16 rows are
passthrough-zero; weights restored after measurement; hybrid merge refuses
double-costed names; backfill only adds missing rows.
"""
from __future__ import annotations

from copy import deepcopy
import types

import pytest
import torch
import torch.nn as nn

from prismaquant.expert_empirical_cost import (
    backfill_missing_from_base,
    measure_expert_unit_costs,
    merge_cost_payloads,
)

from test_packed_expert_cross_domain_gate import TinyLM


class TinyCausal(nn.Module):
    """TinyLM + head so the KL harness sees ``.logits``."""

    def __init__(self, vocab: int = 32, hidden: int = 16):
        super().__init__()
        self.inner = TinyLM(vocab=vocab, hidden=hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.vocab = vocab

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        h = self.inner(input_ids)
        logits = self.head(h).reshape(input_ids.shape[0], -1, self.vocab)
        return types.SimpleNamespace(logits=logits)


EXPERT_NAMES = {
    "inner.mlp.experts.gate_up_proj",
    "inner.mlp.experts.down_proj",
}


def test_measure_unit_costs_on_tiny_packed_moe():
    torch.manual_seed(7)
    model = TinyCausal().eval()
    calib = torch.randint(0, 32, (2, 24))
    before = {
        n: getattr(model.inner.mlp.experts, a).detach().clone()
        for n, a in (("gate_up", "gate_up_proj"), ("down", "down_proj"))
    }

    stats, costs, unit_kls = measure_expert_unit_costs(
        model, None, calib, ["NVFP4", "FP8_DYNAMIC", "BF16"],
        expert_chunk=1, progress=False,
        cell_scores={
            name: {
                "NVFP4": {
                    "activation_output_mse": 0.25,
                    "activation_output_mse_by_codebook_source": {
                        "lattice": 0.25,
                        "learned": 0.125,
                    },
                },
            }
            for name in EXPERT_NAMES
        })

    assert set(stats) == EXPERT_NAMES
    assert set(costs) == EXPERT_NAMES
    (unit,) = unit_kls.values()
    assert unit["NVFP4"] > 0.0
    assert unit["FP8_E4M3"] >= 0.0
    # FP8 error should be well below NVFP4 on the same unit.
    assert unit["FP8_E4M3"] < unit["NVFP4"]
    for name in EXPERT_NAMES:
        row = costs[name]
        assert set(row) == {"NVFP4", "FP8_E4M3", "BF16"}
        assert row["BF16"]["predicted_dloss"] == 0.0
        assert row["NVFP4"]["cost_source"] == "empirical_unit_kl"
        assert row["NVFP4"]["activation_output_mse"] == 0.25
        assert row["NVFP4"][
            "activation_output_mse_by_codebook_source"
        ] == {"lattice": 0.25, "learned": 0.125}
        assert stats[name]["h_trace"] == 0.0
    # Member shares re-assemble exactly one unit KL per format.
    for fmt in ("NVFP4", "FP8_E4M3"):
        total = sum(costs[n][fmt]["predicted_dloss"] for n in EXPERT_NAMES)
        assert total == pytest.approx(unit[fmt], rel=1e-6)
    # In-place quantize/restore left the model untouched.
    assert torch.equal(
        model.inner.mlp.experts.gate_up_proj.detach(), before["gate_up"])
    assert torch.equal(
        model.inner.mlp.experts.down_proj.detach(), before["down"])


def test_merge_refuses_double_costed_names():
    base = {"stats": {"a": {}}, "costs": {"a": {"NVFP4": {}}}}
    with pytest.raises(RuntimeError, match="collision"):
        merge_cost_payloads(
            base, {"a": {}}, {"a": {"NVFP4": {}}}, formats=["NVFP4", "BF16"])


def test_replace_experts_preserves_optional_cell_scores_from_base():
    scores = {
        "activation_output_mse": 0.25,
        "activation_output_mse_by_codebook_source": {
            "lattice": 0.25,
            "learned": 0.125,
        },
    }
    base = {
        "stats": {"experts.down_proj": {"h_trace": 1.0}},
        "costs": {
            "experts.down_proj": {
                "NVFP4": {"weight_mse": 1.0, **scores},
            },
        },
    }
    empirical = {
        "experts.down_proj": {
            "NVFP4": {"predicted_dloss": 0.1},
        },
    }
    empirical_before = deepcopy(empirical)

    merged = merge_cost_payloads(
        base,
        {"experts.down_proj": {"h_trace": 0.0}},
        empirical,
        formats=["NVFP4"],
        replace_experts=True,
    )

    row = merged["costs"]["experts.down_proj"]["NVFP4"]
    assert row["predicted_dloss"] == 0.1
    assert row["activation_output_mse"] == 0.25
    assert row["activation_output_mse_by_codebook_source"] == {
        "lattice": 0.25,
        "learned": 0.125,
    }
    assert empirical == empirical_before
    assert row is not empirical["experts.down_proj"]["NVFP4"]


def test_merge_and_backfill():
    base = {
        "stats": {"lin": {"h_trace": 1.0}},
        "costs": {"lin": {"NVFP4": {"predicted_dloss": 0.5}}},
    }
    merged = merge_cost_payloads(
        base,
        {"experts.down_proj": {"h_trace": 0.0}},
        {"experts.down_proj": {"NVFP4": {"predicted_dloss": 0.1}}},
        formats=["NVFP4", "FP8_DYNAMIC", "BF16"],
    )
    assert set(merged["costs"]) == {"lin", "experts.down_proj"}
    assert merged["formats"] == ["NVFP4", "FP8_E4M3", "BF16"]

    base_cost = {
        "stats": {"mtp.fc": {"h_trace": 2.0}},
        "costs": {
            "mtp.fc": {"NVFP4": {"predicted_dloss": 9.0}},
            "lin": {"NVFP4": {"predicted_dloss": 777.0}},  # must NOT override
        },
    }
    added = backfill_missing_from_base(merged, base_cost)
    assert added == ["mtp.fc"]
    assert merged["costs"]["lin"]["NVFP4"]["predicted_dloss"] == 0.5
    assert merged["costs"]["mtp.fc"]["NVFP4"]["predicted_dloss"] == 9.0
    assert merged["stats"]["mtp.fc"]["h_trace"] == 2.0


# --------------------------------------------------------------------------
# Audit 2026-07-02 §3.5: NV formats derive one per-TENSOR global scale from
# the slice they are given, so chunk-batched expert quantization shared one
# global across the chunk and made the measured unit KL depend on the
# --expert-chunk knob. Export ships per-expert globals; the measurement must
# quantize per expert slice.
# --------------------------------------------------------------------------
def _spread_expert_model(num_experts: int = 16):
    """TinyCausal with ``num_experts`` experts and a 4x per-expert magnitude
    spread (a chunk-shared NVFP4 global visibly distorts the small ones)."""
    from test_packed_expert_cross_domain_gate import (
        TinyPackedExperts,
        TinyRouter,
    )

    model = TinyCausal().eval()
    model.inner.mlp.gate = TinyRouter(hidden_size=16, num_experts=num_experts)
    model.inner.mlp.experts = TinyPackedExperts(num_experts=num_experts)
    with torch.no_grad():
        for e in range(num_experts):
            scale = 1.0 + 3.0 * e / (num_experts - 1)
            model.inner.mlp.experts.gate_up_proj[e] *= scale
            model.inner.mlp.experts.down_proj[e] *= scale
    return model


def test_nvfp4_unit_kl_is_expert_chunk_invariant():
    from prismaquant import format_registry as fr
    from prismaquant.expert_empirical_cost import (
        _baseline_logprobs,
        _unit_kl,
    )

    torch.manual_seed(3)
    model = _spread_expert_model(16)
    mod = model.inner.mlp.experts
    pnames = ["gate_up_proj", "down_proj"]

    # Premise: on this tensor the chunk-shared global genuinely differs from
    # per-expert globals (otherwise the test could not discriminate).
    spec = fr.get_format("NVFP4")
    w = mod.gate_up_proj.detach().float()
    per_expert = torch.stack(
        [spec.quantize_dequantize(w[e].clone()) for e in range(16)])
    shared = spec.quantize_dequantize(w.clone())
    assert not torch.equal(per_expert, shared)

    calib = torch.randint(0, 32, (2, 24))
    baseline = _baseline_logprobs(model, calib)
    kls = {
        chunk: _unit_kl(model, calib, baseline, mod, pnames, "NVFP4",
                        expert_chunk=chunk)
        for chunk in (1, 4, 16)
    }
    # Per-expert quantization: the chunk knob cannot change the measurement.
    assert kls[4] == kls[1]
    assert kls[16] == kls[1]
    assert kls[1] > 0.0


def test_fp8_weight_qdq_is_chunk_invariant_on_packed_tensors():
    """FP8_E4M3 weight qdq reshapes to (-1, in) with an independent scale per
    output row, so chunk-batching experts is exact — the verified basis for
    keeping the batched path for non-NV formats in ``_unit_kl``."""
    from prismaquant import format_registry as fr
    from prismaquant.expert_empirical_cost import (
        _baseline_logprobs,
        _unit_kl,
    )

    torch.manual_seed(4)
    spec = fr.get_format("FP8_E4M3")
    w = torch.randn(16, 8, 16)
    w *= torch.linspace(1.0, 4.0, 16).view(-1, 1, 1)
    full = spec.quantize_dequantize(w.clone())
    per = torch.stack(
        [spec.quantize_dequantize(w[e].clone()) for e in range(16)])
    assert torch.equal(full, per)

    model = _spread_expert_model(16)
    mod = model.inner.mlp.experts
    calib = torch.randint(0, 32, (2, 24))
    baseline = _baseline_logprobs(model, calib)
    kls = {
        chunk: _unit_kl(model, calib, baseline, mod,
                        ["gate_up_proj", "down_proj"], "FP8_E4M3",
                        expert_chunk=chunk)
        for chunk in (4, 16)
    }
    assert kls[4] == kls[16]


# --------------------------------------------------------------------------- #
# Expert sampling + ladder generalization (encode-speed workstream 2026-07-19)
# --------------------------------------------------------------------------- #
def _wide_moe(num_experts: int = 8, seed: int = 11) -> "TinyCausal":
    from test_packed_expert_cross_domain_gate import (
        TinyPackedExperts,
        TinyRouter,
    )
    torch.manual_seed(seed)
    model = TinyCausal().eval()
    model.inner.mlp.gate = TinyRouter(hidden_size=16, num_experts=num_experts)
    model.inner.mlp.experts = TinyPackedExperts(num_experts=num_experts)
    return model


def test_expert_sampling_restores_and_annotates():
    model = _wide_moe()
    calib = torch.randint(0, 32, (2, 24))
    before = {
        n: getattr(model.inner.mlp.experts, a).detach().clone()
        for n, a in (("gate_up", "gate_up_proj"), ("down", "down_proj"))
    }
    stats, costs, unit_kls = measure_expert_unit_costs(
        model, None, calib, ["NVFP4", "FP8_DYNAMIC", "BF16"],
        expert_chunk=1, progress=False, expert_sample=3)
    (unit,) = unit_kls.values()
    samp = unit["_sampling"]
    assert samp["num_experts"] == 8 and samp["sampled"] == 3
    assert samp["scale"] == pytest.approx(8.0 / 3.0, rel=1e-3)
    assert unit["NVFP4"] > 0.0
    # Restoration exact even under slice-wise clone/restore.
    assert torch.equal(
        model.inner.mlp.experts.gate_up_proj.detach(), before["gate_up"])
    assert torch.equal(
        model.inner.mlp.experts.down_proj.detach(), before["down"])
    # Member shares still re-assemble the (scaled) unit KL.
    total = sum(costs[n]["NVFP4"]["predicted_dloss"] for n in EXPERT_NAMES)
    assert total == pytest.approx(unit["NVFP4"], rel=1e-6)


def test_expert_sampling_estimates_full_unit_kl():
    """Sampling all-but-none vs full: at S == E the sample path must equal the
    full path exactly (scale 1, same experts); at S < E the estimate should
    land within a loose factor of the full measurement on the tiny fixture
    (cross-expert additivity is exact only in fp32; this is a sanity band,
    the real gate is the 35B validation)."""
    model = _wide_moe(seed=13)
    calib = torch.randint(0, 32, (2, 24))
    _, _, full = measure_expert_unit_costs(
        model, None, calib, ["NVFP4", "BF16"], expert_chunk=1,
        progress=False)
    _, _, samp = measure_expert_unit_costs(
        model, None, calib, ["NVFP4", "BF16"], expert_chunk=1,
        progress=False, expert_sample=4)
    (kf,) = full.values()
    (ks,) = samp.values()
    assert 0.2 * kf["NVFP4"] < ks["NVFP4"] < 5.0 * kf["NVFP4"]


def test_max_units_and_filter():
    model = _wide_moe()
    calib = torch.randint(0, 32, (2, 24))
    _, _, unit_kls = measure_expert_unit_costs(
        model, None, calib, ["NVFP4", "BF16"], expert_chunk=1,
        progress=False, unit_filter="no-such-unit")
    assert unit_kls == {}
    _, _, unit_kls = measure_expert_unit_costs(
        model, None, calib, ["NVFP4", "BF16"], expert_chunk=1,
        progress=False, max_units=1)
    assert len(unit_kls) == 1


def test_ladder_split_pays_at_four_fp8_rungs():
    from prismaquant.expert_empirical_cost import (
        _cb_ladder_fit,
        _cb_ladder_rate_factor,
        _cb_ladder_split,
    )
    fmts = ["FP8_CB_K36", "FP8_CB_K40", "FP8_CB_K44", "FP8_CB_K48"]
    split = _cb_ladder_split(fmts)
    assert split is not None and len(split) == 1
    kmap, anchors, holdout, predicted = split[0]
    assert set(anchors) == {"FP8_CB_K36", "FP8_CB_K48"}
    assert len(predicted) == 1 and holdout not in predicted
    # R20 (2026-07-30): the expert chain runs the SAME law as the dense
    # chain, whose leading branch is the split-aware floored linear law
    # D = F + C*R(k) in the exact ceil-first rate factor. Exact on its own
    # model -> holdout accepts, prediction exact.
    kls = {f: 0.002 + 1.5 * _cb_ladder_rate_factor(f, kmap[f]) for f in fmts}
    pred, rel, tol = _cb_ladder_fit(
        kls, kmap, anchors, holdout, predicted, 0.10)
    assert pred is not None and rel < 1e-9
    assert tol == 0.10          # no window datum supplied -> the floor
    (pf,) = predicted
    assert pred[pf] == pytest.approx(kls[pf], rel=1e-9)
    # A FREE-rate exponential (decay != R's) is NOT the law's model. Before
    # R20 the expert chain was log-linear with no R(k) term and fitted this
    # exactly, shipping an interpolated cost; now the holdout gate rejects
    # and the caller measures — the same guarantee the dense chain has had
    # since 5184892.
    expo = {f: 2.0 ** (3.0 - 0.4 * kmap[f]) for f in fmts}
    pred_e, rel_e, _ = _cb_ladder_fit(
        expo, kmap, anchors, holdout, predicted, 0.10)
    assert pred_e is None and rel_e > 0.10
    # Broken law -> holdout rejects.
    kls_bad = dict(kls)
    kls_bad[holdout] *= 3.0
    pred_bad, rel_bad, _ = _cb_ladder_fit(
        kls_bad, kmap, anchors, holdout, predicted, 0.10)
    assert pred_bad is None and rel_bad > 0.10


def test_signed_ladder_rate_uses_only_magnitude_table_bits():
    from prismaquant.expert_empirical_cost import _cb_ladder_rate_factor

    # S13 spends eight of its thirteen codeword bits on signs. Its one
    # magnitude subtable therefore has 2**5 entries of dimension eight.
    assert _cb_ladder_rate_factor("NVFP4_CB_S13", 13) == pytest.approx(
        2.0 ** (-2.0 * 5.0 / 8.0)
    )


def test_ladder_split_too_short():
    from prismaquant.expert_empirical_cost import _cb_ladder_split
    assert _cb_ladder_split(["FP8_CB_K36", "FP8_CB_K44", "NVFP4"]) is None


def test_ladder_split_explicit_campaign_plan(monkeypatch):
    from prismaquant.expert_empirical_cost import _cb_ladder_split

    fmts = [f"FP8_CB_K{k}" for k in range(28, 49)]
    monkeypatch.setenv(
        "PRISMAQUANT_CB_LADDER_ANCHORS",
        "FP8_CB_K28,FP8_CB_K38,FP8_CB_K48",
    )
    monkeypatch.setenv("PRISMAQUANT_CB_LADDER_HOLDOUT", "FP8_CB_K33")
    (kmap, anchors, holdout, predicted), = _cb_ladder_split(fmts)
    assert set(kmap) == set(fmts)
    assert anchors == ["FP8_CB_K28", "FP8_CB_K38", "FP8_CB_K48"]
    assert holdout == "FP8_CB_K33"
    assert set(predicted) == set(fmts) - set(anchors) - {holdout}


def test_ladder_split_rejects_incomplete_explicit_plan(monkeypatch):
    from prismaquant.expert_empirical_cost import _cb_ladder_split

    monkeypatch.setenv("PRISMAQUANT_CB_LADDER_ANCHORS", "FP8_CB_K28,FP8_CB_K48")
    monkeypatch.delenv("PRISMAQUANT_CB_LADDER_HOLDOUT", raising=False)
    with pytest.raises(ValueError, match="requires both"):
        _cb_ladder_split([f"FP8_CB_K{k}" for k in range(28, 49)])


def test_imatrix_synthesis_for_cb_units():
    """CB menus self-synthesize missing packed-expert imatrix entries:
    gate_up = pooled module-input second moment; down_proj = the routed
    per-expert intermediate REPLAY (its input is never activation-cached —
    the latent 35B blocker found 2026-07-19). Shapes and coverage pinned."""
    from prismaquant.expert_empirical_cost import (
        _baseline_logprobs,
        ensure_unit_col_weights,
    )
    model = _wide_moe(seed=17)
    calib = torch.randint(0, 32, (2, 24))
    units = [("inner.mlp.experts", model.inner.mlp.experts)]
    _, unit_x = _baseline_logprobs(model, calib, capture_units=units)
    assert "inner.mlp.experts" in unit_x
    cw: dict = {}
    added = ensure_unit_col_weights(model, units, cw, unit_x)
    assert set(added) == {"inner.mlp.experts.gate_up_proj",
                          "inner.mlp.experts.down_proj"}
    gu = cw["inner.mlp.experts.gate_up_proj"]
    dn = cw["inner.mlp.experts.down_proj"]
    E = model.inner.mlp.experts.num_experts
    inter = model.inner.mlp.experts.down_proj.shape[-1]
    hidden = model.inner.mlp.experts.gate_up_proj.shape[-1]
    assert gu.shape == (1, 1, hidden) and bool((gu > 0).all())
    assert dn.shape == (E, 1, inter)
    # Every expert row is positive (routed rows measured; unrouted rows get
    # the routed mean — never zero, which would zero the VQ objective).
    assert bool((dn > 0).all())
    # Existing entries are respected, not recomputed.
    before = dn.clone()
    added2 = ensure_unit_col_weights(model, units, cw, unit_x)
    assert added2 == [] and torch.equal(cw["inner.mlp.experts.down_proj"],
                                        before)


def test_floor_law_fit_exact_and_degenerate():
    """The 3-anchor floor law D = F + C*2^(-b*k): exact recovery on
    floor-law data (where pure log-linear would miss), None on non-monotone
    anchors (caller falls back to log-linear, holdout still gates)."""
    from prismaquant.expert_empirical_cost import (
        _cb_ladder_fit,
        _fit_floor_law,
    )
    F0, C0, b0 = 0.004, 3.0, 0.35
    ks = [28.0, 40.0, 48.0]
    ds = [F0 + C0 * 2.0 ** (-b0 * k) for k in ks]
    fl = _fit_floor_law(ks, ds)
    assert fl is not None
    F, C, b = fl
    assert F == pytest.approx(F0, rel=1e-6)
    assert C == pytest.approx(C0, rel=1e-6)
    assert b == pytest.approx(b0, rel=1e-6)
    # Non-monotone -> None.
    assert _fit_floor_law(ks, [1.0, 2.0, 0.5]) is None
    # End-to-end through _cb_ladder_fit. Since R20 the shared chain LEADS
    # with the floored-linear-in-R(k) branch, so this floor law is reached
    # only when that branch declines (C <= 0 / degenerate anchors). On
    # floor-law data the R branch is a good but not exact fit: it misses the
    # holdout by 5.2%, inside the 10% gate, and lands the predictions within
    # ~7%. The gate — not the branch order — is what makes either safe.
    fmts = {f"FP8_CB_K{k}": k for k in (28, 32, 36, 40, 44, 48)}
    kls = {f: F0 + C0 * 2.0 ** (-b0 * kk) for f, kk in fmts.items()}
    anchors = ["FP8_CB_K28", "FP8_CB_K40", "FP8_CB_K48"]
    pred, rel, tol = _cb_ladder_fit(kls, fmts, anchors, "FP8_CB_K36",
                                    ["FP8_CB_K32", "FP8_CB_K44"], 0.10)
    assert pred is not None and rel == pytest.approx(0.0516, abs=5e-4)
    assert tol == 0.10
    for f, v in pred.items():
        assert v == pytest.approx(kls[f], rel=0.07)
