"""First unit coverage for aura_cost (the paper-spine allocation cost).

Pins the two load-bearing claims the module makes about itself:
  (a) 0.5·mean_k⟨g_k, dW⟩² estimates the exact Fisher quadratic
      (fisher_quadratic_form of the true logit displacement);
  (b) chunked execution (G>1) is bit-identical to single-pass (G=1).
Plus the guards: passthrough zero rows, strict cache mode, the
tied-embeddings include_lm_head guard, and the stderr/provenance fields.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant.aura_cost import _delta_w, compute_aura_cost
from prismaquant.kl_fisher import fisher_quadratic_form


class TinyLM(nn.Module):
    """embed -> body Linear -> relu -> head Linear -> logits.

    Logits are affine in the head weight, so a head-weight perturbation has an
    exactly computable logit displacement — the ground truth for test (a).
    """

    def __init__(self, vocab: int = 64, hidden: int = 32, tie: bool = False):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.body = nn.Linear(hidden, hidden, bias=False)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        if tie:
            self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids):
        h = torch.relu(self.body(self.embed(input_ids)))
        return SimpleNamespace(logits=self.lm_head(h))


class _FakeCache:
    """Production-cache stand-in: returns rendered = W + dW for chosen keys."""

    def __init__(self, rendered: dict[tuple[str, str], torch.Tensor]):
        self._rendered = rendered

    def get(self, name: str, fmt: str):
        return self._rendered.get((name, fmt))


def _ids(batch=2, seqlen=8, vocab=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seqlen), generator=g)


def test_aura_cb_delta_refuses_unweighted_direct_fallback():
    with pytest.raises(RuntimeError, match="requires a production-cache render"):
        _delta_w(
            "layer.q_proj",
            "NVFP4_CB_K16",
            torch.randn(2, 256),
            cache=None,
        )


def test_aura_cb_provenance_comes_from_cache_not_current_env(monkeypatch):
    from prismaquant.nvfp4_cb_footprint import (
        CBSerializationContext,
        cb_serialization_context_stamp,
    )
    from prismaquant.production_weight_cache import (
        ProductionWeightCache,
        bind_cb_render_identity_source_weights,
        build_production_cache_cb_render_identity,
    )

    torch.manual_seed(9)
    model = TinyLM().eval()
    fmt = "NVFP4_CB_K16"
    context = CBSerializationContext.legacy_v1()
    col_weights = {"body": torch.ones(model.body.in_features)}
    identity = build_production_cache_cb_render_identity(
        {"body": (fmt,)},
        cb_serialization_context=context,
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    identity = bind_cb_render_identity_source_weights(
        identity,
        {"body": model.body.weight.detach()},
    )
    cache = ProductionWeightCache(
        weights={
            ("body", fmt): model.body.weight.detach().clone() + 0.03125,
        },
        levers={"weighted_vq": True},
        metadata={"cb_render_identity": identity},
    )
    monkeypatch.setenv("CB_SCALE_CODING", "two_tier")
    monkeypatch.setenv("CB_CODEBOOK_SOURCE", "lattice")

    payload = compute_aura_cost(
        model,
        _ids(batch=1, seqlen=4),
        [fmt],
        n_probes=1,
        n_linear_chunks=1,
        production_cache=cache,
        require_production_cache=True,
        min_free_gib=0.0,
    )

    assert payload["provenance"]["cb_serialized_payload"] == (
        identity["cb_serialized_payload"]
    )
    assert payload["provenance"]["cb_render_identity"] == identity


def test_estimator_matches_exact_fisher_quadratic():
    torch.manual_seed(3)
    model = TinyLM().eval()
    ids = _ids()

    # A known head-weight perturbation, exactly bf16-representable so the
    # bf16 dW storage path is lossless for it.
    dw = (torch.randn_like(model.lm_head.weight) * 0.25).to(torch.bfloat16)
    dw = dw.float() * 0.03125  # power-of-two scale keeps bf16 exactness
    cache = _FakeCache({
        ("lm_head", "NVFP4"): model.lm_head.weight.detach() + dw,
    })

    # Exact: logits are affine in the head weight, so the displacement of a
    # +dw perturbation is computable in closed form via a second forward.
    with torch.no_grad():
        teacher = model(ids).logits
        model.lm_head.weight += dw
        student = model(ids).logits
        model.lm_head.weight -= dw
    exact = float(fisher_quadratic_form(
        teacher, student - teacher, token_scope="all"))

    payload = compute_aura_cost(
        model, ids, ["NVFP4"],
        n_probes=4096, production_cache=cache,
        min_free_gib=0.0, n_linear_chunks=1, include_lm_head=True,
    )
    row = payload["costs"]["lm_head"]["NVFP4"]
    est = row["predicted_dloss"]
    assert row["dw_source"] == "rendered"
    # K=4096 Rademacher probes -> ~2-3% sampling error on the mean; 15% is a
    # flake-proof bound that still rejects any normalization mistake (which
    # would be off by a factor of T, V, or 2).
    assert abs(est - exact) <= 0.15 * exact, (est, exact)
    # stderr should be a plausible scale for the sampling error
    assert 0 < row["predicted_dloss_stderr"] < 0.25 * exact


def test_chunked_is_bit_identical_to_single_pass():
    torch.manual_seed(5)
    model = TinyLM().eval()
    ids = _ids(seed=1)
    kw = dict(n_probes=8, min_free_gib=0.0)

    one = compute_aura_cost(model, ids, ["NVFP4"], n_linear_chunks=1, **kw)
    three = compute_aura_cost(model, ids, ["NVFP4"], n_linear_chunks=3, **kw)

    assert one["costs"].keys() == three["costs"].keys()
    for n in one["costs"]:
        for f in one["costs"][n]:
            a, b = one["costs"][n][f], three["costs"][n][f]
            assert a["predicted_dloss"] == b["predicted_dloss"], (n, f)
            assert a["predicted_dloss_stderr"] == b["predicted_dloss_stderr"]
    for n in one["stats"]:
        assert one["stats"][n]["h_trace"] == three["stats"][n]["h_trace"]


def test_passthrough_formats_emit_zero_cost_rows():
    model = TinyLM().eval()
    payload = compute_aura_cost(
        model, _ids(), ["NVFP4", "BF16"],
        n_probes=2, min_free_gib=0.0, n_linear_chunks=1,
    )
    for n, rows in payload["costs"].items():
        assert rows["BF16"]["predicted_dloss"] == 0.0
        assert rows["BF16"]["cost_source"] == "aura_passthrough_zero"
        assert rows["NVFP4"]["predicted_dloss"] >= 0.0
        assert rows["NVFP4"]["dw_source"] == "rtn"


def test_require_production_cache_refuses_silent_rtn():
    model = TinyLM().eval()
    with pytest.raises(RuntimeError, match="require_production_cache"):
        compute_aura_cost(
            model, _ids(), ["NVFP4"],
            n_probes=2, min_free_gib=0.0, n_linear_chunks=1,
            production_cache=_FakeCache({}),
            require_production_cache=True,
        )


def test_tied_lm_head_guard_fires():
    model = TinyLM(tie=True).eval()
    with pytest.raises(RuntimeError, match="tie_word_embeddings"):
        compute_aura_cost(
            model, _ids(), ["NVFP4"],
            n_probes=2, min_free_gib=0.0, n_linear_chunks=1,
            include_lm_head=True,
        )
    # Without include_lm_head the tied model is fine (lm_head excluded).
    payload = compute_aura_cost(
        model, _ids(), ["NVFP4"],
        n_probes=2, min_free_gib=0.0, n_linear_chunks=1,
    )
    assert "lm_head" not in payload["costs"]


def test_provenance_records_seed_and_dw_split():
    model = TinyLM().eval()
    ids = _ids()
    cache = _FakeCache({
        ("body", "NVFP4"): model.body.weight.detach() * 1.001,
    })
    payload = compute_aura_cost(
        model, ids, ["NVFP4"],
        n_probes=2, min_free_gib=0.0, n_linear_chunks=1,
        production_cache=cache, seed_base=12345,
    )
    prov = payload["provenance"]
    assert prov["seed_base"] == 12345
    assert prov["dw_rendered_rows"] == 1   # body via the cache
    assert prov["dw_rtn_fallback_rows"] == 0  # lm_head excluded by default
    assert prov["calib_shape"] == list(ids.shape)
    assert len(prov["calib_sha256"]) == 64
    assert payload["costs"]["body"]["NVFP4"]["dw_source"] == "rendered"


def test_per_probe_samples_align_and_reproduce_mean():
    model = TinyLM().eval()
    payload = compute_aura_cost(
        model, _ids(), ["NVFP4"],
        n_probes=6, min_free_gib=0.0, n_linear_chunks=2,
    )
    for n, rows in payload["costs"].items():
        row = rows["NVFP4"]
        xs = row["x2_per_probe"]
        assert len(xs) == 6
        assert abs(0.5 * sum(xs) / 6 - row["predicted_dloss"]) < 1e-12


def test_additivity_gate_legacy_correlated_stderr_has_unverified_alignment():
    import math
    from prismaquant.aura_additivity_gate import additivity_gate

    model = TinyLM().eval()
    payload = compute_aura_cost(
        model, _ids(), ["NVFP4"],
        n_probes=8, min_free_gib=0.0, n_linear_chunks=1,
    )
    assignment = {n: "NVFP4" for n in payload["costs"]}
    # Preserve the legacy arithmetic, but array lengths are not probe identity.
    K = 8
    sums = [0.0] * K
    for n in payload["costs"]:
        for k, x2 in enumerate(payload["costs"][n]["NVFP4"]["x2_per_probe"]):
            sums[k] += x2
    mean_s = sum(sums) / K
    var_s = sum((v - mean_s) ** 2 for v in sums) / (K - 1)
    expected_stderr = 0.5 * math.sqrt(var_s / K)
    expected_sum = 0.5 * mean_s

    out = additivity_gate(payload, assignment, measured_kl=expected_sum * 1.1)
    assert out["stderr_method"] == "per_probe_unverified"
    assert out["probe_alignment_verified"] is False
    assert abs(out["predicted_sum"] - expected_sum) < 1e-12
    assert abs(out["predicted_stderr"] - expected_stderr) < 1e-12
    assert abs(out["residual"] - 0.1 * expected_sum) < 1e-9
    assert out["n_covered"] == len(assignment)
    assert out["uncovered"] == []


def test_additivity_gate_reports_uncovered_and_passthrough():
    from prismaquant.aura_additivity_gate import additivity_gate

    model = TinyLM().eval()
    payload = compute_aura_cost(
        model, _ids(), ["NVFP4", "BF16"],
        n_probes=4, min_free_gib=0.0, n_linear_chunks=1,
    )
    assignment = {
        "body": "BF16",                      # passthrough -> zero-cost row
        "model.layers.99.fake": "NVFP4",     # no cost row -> uncovered
    }
    out = additivity_gate(payload, assignment, measured_kl=0.0)
    assert out["n_zero_cost"] == 1
    assert out["uncovered"] == ["model.layers.99.fake|NVFP4"]
    assert out["n_covered"] == 0


def test_cost_ucb_z_charges_stderr(monkeypatch):
    from prismaquant.allocator_candidates import cost_entry_predicted_dloss

    stats = {"h_trace": 1.0}
    row = {"predicted_dloss": 0.010, "predicted_dloss_stderr": 0.002,
           "output_mse_measured": False}
    assert cost_entry_predicted_dloss(stats, row) == 0.010  # default: identical
    monkeypatch.setenv("PRISMAQUANT_COST_UCB_Z", "2")
    assert abs(cost_entry_predicted_dloss(stats, row) - 0.014) < 1e-15
    # rows without stderr (old payloads) are unaffected even with z set
    old = {"predicted_dloss": 0.010, "output_mse_measured": False}
    assert cost_entry_predicted_dloss(stats, old) == 0.010


def test_hook_harvest_matches_legacy_and_frees_grads():
    torch.manual_seed(21)
    model = TinyLM().eval()
    ids = _ids(seed=2)
    kw = dict(n_probes=6, min_free_gib=0.0, n_linear_chunks=2)

    legacy = compute_aura_cost(model, ids, ["NVFP4"], **kw)
    hooked = compute_aura_cost(model, ids, ["NVFP4"], hook_harvest=True, **kw)

    for n in legacy["costs"]:
        a, b = legacy["costs"][n]["NVFP4"], hooked["costs"][n]["NVFP4"]
        assert a["predicted_dloss"] == b["predicted_dloss"], n
        assert a["x2_per_probe"] == b["x2_per_probe"], n
    for n in legacy["stats"]:
        assert legacy["stats"][n]["h_trace"] == hooked["stats"][n]["h_trace"]
    # hooks removed cleanly: a fresh legacy run still matches
    again = compute_aura_cost(model, ids, ["NVFP4"], **kw)
    assert again["costs"].keys() == legacy["costs"].keys()
    for n in legacy["costs"]:
        assert (again["costs"][n]["NVFP4"]["predicted_dloss"]
                == legacy["costs"][n]["NVFP4"]["predicted_dloss"])

class _RoutedToy(nn.Module):
    """Data-dependent MoE routing: token id < vocab//2 -> expert_a, else
    expert_b. With probe_microbatch=1 and sample 0 = low ids only, expert_a
    is ABSENT from the final micro-batch's autograd graph — the audit
    2026-07-02 M5 repro shape (scratchpad hook_mb_repro.py)."""

    def __init__(self, vocab: int = 61, hidden: int = 32):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, hidden)
        self.expert_a = nn.Linear(hidden, hidden, bias=False)
        self.expert_b = nn.Linear(hidden, hidden, bias=False)
        self.dense = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, ids):
        x = self.emb(ids)
        xf = x.reshape(-1, x.size(-1))
        lo = ids.reshape(-1) < self.vocab // 2
        out = torch.zeros_like(xf)
        if lo.any():
            out[lo] = self.expert_a(xf[lo])
        if (~lo).any():
            out[~lo] = self.expert_b(xf[~lo])
        xf = xf + out
        x = xf.reshape(ids.size(0), ids.size(1), -1)
        x = x + torch.tanh(self.dense(x))
        return SimpleNamespace(logits=self.head(x))


def test_hook_harvest_microbatch_harvests_routed_stragglers():
    """Audit 2026-07-02 M5: with hook_harvest + probe_microbatch, a param
    routed only in NON-final micro-batches used to have its accumulated grad
    discarded -> predicted_dloss 0.0 with 0 probe samples, silently."""
    torch.manual_seed(11)
    model = _RoutedToy().eval()
    g = torch.Generator().manual_seed(7)
    lo = torch.randint(0, 30, (1, 8), generator=g)   # sample 0: expert_a only
    hi = torch.randint(30, 61, (1, 8), generator=g)  # sample 1: expert_b only
    ids = torch.cat([lo, hi], dim=0)
    K = 256
    kw = dict(n_probes=K, min_free_gib=0.0, n_linear_chunks=1)

    mono = compute_aura_cost(model, ids, ["NVFP4"], **kw)
    legacy_mb = compute_aura_cost(
        model, ids, ["NVFP4"], probe_microbatch=1, **kw)
    hooked_mb = compute_aura_cost(
        model, ids, ["NVFP4"], hook_harvest=True, probe_microbatch=1, **kw)

    # The straggler expert gets a full complement of probe samples and a
    # nonzero cost (the audit repro showed 0 samples / 0.0 for every format).
    row = hooked_mb["costs"]["expert_a"]["NVFP4"]
    assert len(row["x2_per_probe"]) == K
    assert row["predicted_dloss"] > 0.0

    # hook+microbatch == legacy+microbatch bit-for-bit: same probe seeds,
    # same accumulated grads, shared projection code.
    for n in legacy_mb["costs"]:
        a = legacy_mb["costs"][n]["NVFP4"]
        b = hooked_mb["costs"][n]["NVFP4"]
        assert a["x2_per_probe"] == b["x2_per_probe"], n
        assert a["predicted_dloss"] == b["predicted_dloss"], n
    for n in legacy_mb["stats"]:
        assert (legacy_mb["stats"][n]["h_trace"]
                == hooked_mb["stats"][n]["h_trace"]), n

    # vs the monolithic path the match is statistical, not bit-exact (each
    # micro-batch draws its own Rademacher vector): the grad-accumulation
    # normalization is factor-1 via token_count_override, so any slip there
    # would show as a ~2x+ offset; 0.5 rel is flake-proof at K=256.
    a = mono["costs"]["expert_a"]["NVFP4"]["predicted_dloss"]
    b = hooked_mb["costs"]["expert_a"]["NVFP4"]["predicted_dloss"]
    assert a > 0.0
    assert abs(a - b) <= 0.5 * max(a, b), (a, b)


def test_row_stderr_is_sample_variance_over_probes():
    """Audit 2026-07-02 §3.13: the per-row stderr must use the SAMPLE
    (1/(K-1)) variance, matching aura_additivity_gate, not population 1/K."""
    import math

    model = TinyLM().eval()
    K = 4
    payload = compute_aura_cost(
        model, _ids(), ["NVFP4"],
        n_probes=K, min_free_gib=0.0, n_linear_chunks=1,
    )
    checked = 0
    for n, rows in payload["costs"].items():
        xs = rows["NVFP4"]["x2_per_probe"]
        assert len(xs) == K
        mean = sum(xs) / K
        var = sum((x - mean) ** 2 for x in xs) / (K - 1)  # by hand, 1/(K-1)
        expected = 0.5 * math.sqrt(var / K)
        got = rows["NVFP4"]["predicted_dloss_stderr"]
        assert got == pytest.approx(expected, rel=1e-6), n
        if var > 0:
            # the old population form is strictly smaller -> would fail
            pop = 0.5 * math.sqrt(
                max(sum(x * x for x in xs) / K - mean * mean, 0.0) / K)
            assert got > pop, n
            checked += 1
    assert checked > 0  # the population-vs-sample distinction was exercised

    # K=1: sample variance undefined -> stderr 0.0 (previous convention too)
    p1 = compute_aura_cost(
        model, _ids(), ["NVFP4"],
        n_probes=1, min_free_gib=0.0, n_linear_chunks=1,
    )
    for n, rows in p1["costs"].items():
        assert rows["NVFP4"]["predicted_dloss_stderr"] == 0.0


# ---- grafted from codex/review-batch (packed-expert guard, auto-chunk, delta_w provenance) ----
from prismaquant.aura_cost import (
    _guard_packed_expert_coverage,
    _target_linears,
)
class _PackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 32, 16))
        self.down_proj = nn.Parameter(torch.zeros(2, 32, 32))


class _PackedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.experts = _PackedExperts()


class _Dsv4UnpackedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(32, 32, bias=False)
        self.up_proj = nn.Linear(32, 32, bias=False)
        self.down_proj = nn.Linear(32, 32, bias=False)


class _Dsv4UnpackedModel(nn.Module):
    """DSv4 live/probe shape: routed experts are per-expert Linears."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(32, 32, bias=False)
        layer.mlp = nn.Module()
        layer.mlp.experts = nn.ModuleList(
            [_Dsv4UnpackedExpert(), _Dsv4UnpackedExpert()]
        )


def test_aura_guard_rejects_packed_experts_by_default():
    with pytest.raises(RuntimeError, match="packed-MoE expert costs"):
        _guard_packed_expert_coverage(_PackedModel())


def test_aura_guard_requires_explicit_omission_for_packed_experts():
    omitted = _guard_packed_expert_coverage(
        _PackedModel(),
        allow_omission=True,
    )

    assert omitted == [
        "model.layers.0.mlp.experts.down_proj",
        "model.layers.0.mlp.experts.gate_up_proj",
    ]


def test_aura_guard_allows_dense_only_models():
    model = nn.Sequential(nn.Linear(16, 16, bias=False))

    assert _guard_packed_expert_coverage(model) == []


def test_aura_guard_rejects_profile_declared_dsv4_unpacked_experts():
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

    with pytest.raises(RuntimeError, match="expert costs"):
        _guard_packed_expert_coverage(
            _Dsv4UnpackedModel(),
            DeepseekV4Profile(),
        )


def test_aura_smooth_targets_exclude_dsv4_unpacked_experts():
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

    model = _Dsv4UnpackedModel()
    profile = DeepseekV4Profile()
    omitted = _guard_packed_expert_coverage(
        model,
        profile,
        allow_omission=True,
    )

    assert omitted == [
        f"model.layers.0.mlp.experts.{expert_id}.{projection}"
        for expert_id in range(2)
        for projection in ("down_proj", "gate_proj", "up_proj")
    ]
    assert list(_target_linears(model, profile=profile)) == [
        "model.layers.0.self_attn.q_proj"
    ]


def test_aura_guard_fails_closed_when_profile_cannot_classify_experts():
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

    class BrokenProfile(DeepseekV4Profile):
        def packed_expert_format_group(self, qname: str):
            raise LookupError(f"classification unavailable for {qname}")

    with pytest.raises(RuntimeError, match="could not determine routed-expert"):
        _guard_packed_expert_coverage(_Dsv4UnpackedModel(), BrokenProfile())
