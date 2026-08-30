"""Grouped-BMM Fisher: the `wo_a` shape gets a real, correctly-priced row.

DSv4's `attn.wo_a` is an `nn.Linear` subclass whose forward consumes its
`[G*R, D]` weight plane as `W[g, r, d]` through a view + bmm. The dense
accumulator cannot price it (its chunk_h comes out [R, D] against a
[G*R, D] plane), which is why the class used to live in the probe's skip
list and the walk pinned it as unpriced debt. These tests pin the
replacement:

  - the grouped chunk math is EXACT (the elementwise Fisher is
    block-diagonal in g) and hand-checkable on all-ones inputs;
  - the marginals satisfy sum(fisher_row) == sum(fisher_col) == h_trace_raw
    BY CONSTRUCTION — the wiring identity the card validates;
  - normalization matches the dense convention exactly: rows stay RAW
    token sums until `finalize_fisher_stats` divides by the GLOBAL
    calibration token count — never per-routed-token, never /G, never
    /route_prob (that convention was deliberately removed);
  - `n_tokens_seen` counts TOKENS, not token-group pairs;
  - dispatch is explicit (profile-declared classes only) and fails fast
    when a declared class lacks its group count.

The synthetic `GroupedBMMLinear` mirrors the vendored
`DeepseekV4GroupedLinear.forward` line for line; the slow test at the
bottom runs the REAL vendored class end to end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from prismaquant.sensitivity_probe import (
    FisherAccumulator,
    finalize_fisher_stats,
    grouped_linear_fisher_chunk,
    grouped_linear_groups,
    grouped_linear_stats_entry,
)

# Tiny but non-trivial: G does not divide R or D, so a mis-shapen
# accumulator cannot pass by symmetry.
G, R, D, T = 2, 3, 4, 4


class GroupedBMMLinear(nn.Linear):
    """Mirror of DeepseekV4GroupedLinear: standard [out, in] storage,
    per-group bmm consumption."""

    def __init__(self, in_features_per_group: int, out_features: int,
                 n_groups: int):
        super().__init__(in_features_per_group, out_features, bias=False)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-2]
        d_in = x.shape[-1]
        out_per_group = self.out_features // self.n_groups
        w = self.weight.view(self.n_groups, out_per_group, d_in)
        xx = x.reshape(-1, self.n_groups, d_in).permute(1, 0, 2)
        y = torch.bmm(xx, w.transpose(-1, -2)).permute(1, 0, 2)
        return y.reshape(*batch_shape, self.n_groups, out_per_group)


class _DeclaringProfile:
    """Minimal profile stand-in: declares GroupedBMMLinear, nothing else."""

    def probe_grouped_module_class_names(self):
        return ("GroupedBMMLinear",)


class _SilentProfile:
    def probe_grouped_module_class_names(self):
        return ()


def test_grouped_chunk_matches_the_hand_computed_fisher():
    """All-ones inputs make every term countable by hand.

    grad_t W[g,r,d] = gy[t,g,r]·x[t,g,d], so with ones:
      h_trace_raw   = Σ_{t,g,r,d} 1 = T*G*R*D          = 4*2*3*4 = 96
      fisher_row[g*R+r] = Σ_{t,d} 1 = T*D              = 16   (6 entries)
      fisher_col[d]     = Σ_{t,g,r} 1 = T*G*R          = 24   (4 entries)
      g_sq_sum[g*R+r]   = Σ_t gy² = T                  = 4
      act_sq_sum[d]     = Σ_{t,g} x² = T*G             = 8
      trace_per_group[g] = T*R*D                       = 48
      h_w2 with W = w0 everywhere: w0² · T*G*R*D        = 0.25·96 = 24
    """
    x = torch.ones(T, G, D)
    gy = torch.ones(T, G, R)
    weight = torch.full((G * R, D), 0.5)

    pieces = grouped_linear_fisher_chunk(x, gy, G, weight)

    assert float(pieces["h_trace"]) == pytest.approx(T * G * R * D)
    assert pieces["fisher_row"].shape == (G * R,)
    assert torch.allclose(
        pieces["fisher_row"], torch.full((G * R,), float(T * D)))
    assert pieces["fisher_col"].shape == (D,)
    assert torch.allclose(pieces["fisher_col"], torch.full((D,), float(T * G * R)))
    assert torch.allclose(pieces["g_sq_sum"], torch.full((G * R,), float(T)))
    assert torch.allclose(pieces["act_sq_sum"], torch.full((D,), float(T * G)))
    assert torch.allclose(pieces["act_absmax"], torch.ones(D))
    assert pieces["trace_per_group"].shape == (G,)
    assert torch.allclose(
        pieces["trace_per_group"],
        torch.full((G,), float(T * R * D)))
    assert float(pieces["h_w2"]) == pytest.approx(0.25 * T * G * R * D)


def test_grouped_chunk_marginals_satisfy_the_wiring_identity():
    """sum(fisher_row) == sum(fisher_col) == h_trace, on random inputs,
    because every marginal reduces the SAME fp32 chunk_h. This is the
    identity SensitivityCard.validate() enforces downstream."""
    torch.manual_seed(7)
    x = torch.randn(6, G, D)
    gy = torch.randn(6, G, R)
    pieces = grouped_linear_fisher_chunk(x, gy, G, None)
    total = float(pieces["h_trace"])
    assert float(pieces["fisher_row"].sum()) == pytest.approx(total, rel=1e-5)
    assert float(pieces["fisher_col"].sum()) == pytest.approx(total, rel=1e-5)


def test_grouped_chunk_is_exact_vs_brute_force_per_group():
    """The per-group bmm reproduces the brute-force per-token gradient
    norm sum exactly (block-diagonal in g — no approximation)."""
    torch.manual_seed(11)
    x = torch.randn(5, G, D)
    gy = torch.randn(5, G, R)
    xf = x.double()
    gyf = gy.double()

    # Brute force: materialize each token's full [G,R,D] gradient.
    expected = 0.0
    for t in range(5):
        grad_t = gyf[t].unsqueeze(-1) * xf[t].unsqueeze(-2)  # [G,R,D]
        expected += float(grad_t.pow(2).sum())

    pieces = grouped_linear_fisher_chunk(x, gy, G, None)
    assert float(pieces["h_trace"].double()) == pytest.approx(expected, rel=1e-6)


def test_grouped_row_normalizes_like_a_dense_row_end_to_end():
    """THE normalization test. A dense row and a grouped row measured in
    the same run must share ONE denominator: the GLOBAL calibration token
    count passed to finalize(). With ones everywhere:

      dense   raw = T * R * D       = 48 -> h_trace = 48 / 4 = 12
      grouped raw = T * G * R * D   = 96 -> h_trace = 96 / 4 = 24

    Any other denominator on the grouped side (per-routed-token, /G,
    /route_prob) would break the equality with this arithmetic — that is
    the inverted-importance failure mode `finalize_fisher_stats` exists
    to prevent. `n_tokens_seen` stays RAW metadata and counts TOKENS:
    the grouped row saw T tokens through G slices each, NOT T*G."""
    model = nn.Module()
    model.dense = nn.Linear(D, R, bias=False)
    model.grouped = GroupedBMMLinear(D, G * R, G)
    with torch.no_grad():
        model.dense.weight.fill_(1.0)
        model.grouped.weight.fill_(0.5)
    for p in model.parameters():
        p.requires_grad_(False)

    acc = FisherAccumulator(model, ["dense", "grouped"], {},
                            model_profile=_DeclaringProfile())
    x_g = torch.ones(T, G, D, requires_grad=True)
    x_d = torch.ones(T, D, requires_grad=True)
    loss = (model.grouped(x_g) * 1.0).sum() + (model.dense(x_d) * 1.0).sum()
    loss.backward()
    acc.finalize(None, global_tokens=T)
    acc.remove_hooks()

    drow, grow = acc.stats["dense"], acc.stats["grouped"]

    # Same denominator, each against its own hand value.
    assert drow["h_trace"] == pytest.approx(float(R * D))          # 48/4
    assert grow["h_trace"] == pytest.approx(float(G * R * D))      # 96/4
    assert drow["h_trace_norm_tokens"] == T
    assert grow["h_trace_norm_tokens"] == T

    # Raw accumulators were token-SUMMED (pre-normalization values).
    assert grow["h_trace_raw"] == pytest.approx(float(T * G * R * D))

    # Token accounting: TOKENS, not token-group pairs.
    assert grow["n_tokens_seen"] == T
    assert drow["n_tokens_seen"] == T

    # The unit is the whole logical tensor, in flat-plane coordinates.
    assert grow["num_groups"] == G
    assert grow["out_features"] == G * R
    assert grow["in_features"] == D
    assert grow["n_params"] == G * R * D

    # Weight-aware proxy shares the raw/global convention.
    assert grow["h_w2_sum"] == pytest.approx(0.25 * float(G * R * D))


def test_finalize_normalizes_the_per_group_decomposition():
    """`h_trace_per_group_raw` is a [G] decomposition of h_trace_raw; it
    must divide by the same global count as its scalar."""
    stats = {"grouped": {
        "h_trace_raw": 96.0, "h_w2_sum_raw": 24.0,
        "h_trace_per_group_raw": [40.0, 56.0],
        "n_tokens_seen": 4,
    }}
    finalize_fisher_stats(stats, global_tokens=4)
    s = stats["grouped"]
    assert s["h_trace"] == pytest.approx(24.0)
    assert s["h_trace_norm_tokens"] == 4
    assert s["h_trace_per_group"] == pytest.approx([10.0, 14.0])
    assert sum(s["h_trace_per_group"]) == pytest.approx(s["h_trace"])


def test_stats_entry_schema_distinguishes_groups_from_experts():
    """`num_groups` — never `num_experts`, which downstream reads as a
    packed expert stack (`_shape_from_stats`,
    `_stats_indicates_packed_expert`)."""
    mod = GroupedBMMLinear(D, G * R, G)
    entry = grouped_linear_stats_entry(mod, G, w_max_abs=0.1, w_norm_sq=2.0)
    assert entry["num_groups"] == G
    assert "num_experts" not in entry
    assert entry["in_features"] == D
    assert entry["out_features"] == G * R
    assert entry["n_params"] == G * R * D


def test_dispatch_is_explicit_and_fails_fast():
    mod = GroupedBMMLinear(D, G * R, G)
    plain = nn.Linear(D, R)
    assert grouped_linear_groups(mod, _DeclaringProfile()) == G
    # Undeclared class -> dense path (explicit declaration only).
    assert grouped_linear_groups(mod, _SilentProfile()) is None
    assert grouped_linear_groups(mod, None) is None
    assert grouped_linear_groups(plain, _DeclaringProfile()) is None
    # Declared class without its group count must fail LOUD, not fall
    # through to numbers the dense flatten would get wrong.
    bare = type("GroupedBMMLinear", (nn.Linear,), {})(D, G * R)
    with pytest.raises(ValueError, match="n_groups"):
        grouped_linear_groups(bare, _DeclaringProfile())


@pytest.mark.slow
def test_real_dsv4_wo_a_gets_a_priced_probe_row():
    """End to end on the REAL vendored DSv4 modeling code (toy dims): the
    probe tracks `DeepseekV4GroupedLinear` now, and a backward through it
    lands a priced stats row with the right group structure."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_model_walk import _shrunken_dsv4

    profile, model = _shrunken_dsv4()
    tracked = sorted(
        n for n, m in model.named_modules()
        if type(m).__name__ == "DeepseekV4GroupedLinear")
    assert tracked, "shrunken DSv4 must contain wo_a modules"
    for name in tracked:
        assert profile.should_probe_linear(name, model.get_submodule(name))

    o_groups, o_lora_rank = 4, 32
    heads_x_head_dim = 8 * 32
    d_in = heads_x_head_dim // o_groups
    acc = FisherAccumulator(model, tracked, {}, model_profile=profile)
    # Drive each tracked module DIRECTLY rather than backward through the
    # whole toy stack: the shrunken random-init model's compressed-attention
    # backward produces NaN cotangents on some draws/processes (its own
    # masking paths, not the accumulator), which would make this plumbing
    # test flaky. The dispatch and accumulation under test are identical.
    torch.manual_seed(0)
    x = torch.randn(2, o_groups, d_in)
    for name in tracked:
        out = model.get_submodule(name)(x)
        assert out.shape == (2, o_groups, o_lora_rank)
        out.float().pow(2).sum().backward()
        model.zero_grad(set_to_none=True)
    acc.finalize(None, global_tokens=2)
    acc.remove_hooks()

    for name in tracked:
        row = acc.stats[name]
        assert row["num_groups"] == o_groups
        assert row["out_features"] == o_groups * o_lora_rank
        assert row["in_features"] == d_in
        assert row["n_params"] == o_groups * o_lora_rank * d_in
        assert row["h_trace"] > 0.0
        assert row["n_tokens_seen"] == 2
        h_full = acc._h_full[name]
        assert tuple(h_full.shape) == (o_groups * o_lora_rank, d_in)
        # Wiring identity holds on the collected plane too.
        assert float(h_full.sum()) == pytest.approx(row["h_trace_raw"],
                                                    rel=1e-4)
