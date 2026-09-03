"""The packed-expert activation collector must hook the call's full module set.

Dense Linears were closed by #130/#135: the collector hooks the caller's whole
``eligible_qnames`` enumeration and ``render_assignment``/``render_qnames``
narrow only the render, so a stripe, an assignment scope and a resume all
reproduce the whole run's bytes. ``_PackedExpertActivationCollector`` is a
separate class with a separate shared generator and a separate hook set, and
nothing in that change touches it (#145).

The mechanism is the same: one shared ``torch.Generator`` feeds every hooked
packed-experts module's priority reservoir, so the rows a module keeps are a
function of how many rows every earlier hook consumed. The hook set is built
render-narrowed -- a module is added only when at least one of its tensors is
non-BF16 in ``render_assignment``
(``production_weight_cache.py`` ``fill_packed_expert_cache_entries``) -- while
``force_format`` mode (format-menu frontier build, eager NVFP4) hooks every
module because it ignores the assignment. A format-menu frontier build and an
assignment-scoped export build therefore hook different module sets, and the
sampled module tokens -- hence each expert's GPTQ Hessian, hence the bytes --
differ.

The invariant under test, mirroring the dense one:

    the bytes of a packed (tensor, fmt) pair are a function of the visible
    packed-experts modules and the calibration, never of which subset of them
    a call renders.

Arms: one frozen two-layer MoE, one frozen calibration, a ``force_format``
arm (the frontier build) and an assignment-scoped arm whose assignment BF16s
layer 0's experts (the export build). Behind two gates that make the matrix
readable at all -- the module reservoir must be selecting (budget < routed
tokens) and the render must be deterministic.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from prismaquant.production_weight_cache import (
    ProductionWeightCache,
    fill_packed_expert_cache_entries,
)


HIDDEN = 16
INTER = 16
EXPERTS = 2
LAYERS = 2
VOCAB = 32
N_CALIB = 4
SEQLEN = 32
MODULE_TOKEN_BUDGET = 32  # < N_CALIB * SEQLEN routed tokens, so it must select
FORMAT = "NVFP4"
LEVERS = {"gptq": True}


class _TinyRouter(nn.Module):
    def __init__(self, hidden: int = HIDDEN, num_experts: int = EXPERTS):
        super().__init__()
        self.top_k = 1
        self.weight = nn.Parameter(torch.randn(num_experts, hidden))

    def forward(self, hidden_states: torch.Tensor):
        logits = F.linear(hidden_states, self.weight)
        scores, indices = torch.topk(
            torch.softmax(logits.float(), dim=-1), 1, dim=-1)
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return logits, scores.to(hidden_states.dtype), indices


class _TinyPackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = EXPERTS
        self.act_fn = F.silu
        self.gate_up_proj = nn.Parameter(
            torch.randn(EXPERTS, 2 * INTER, HIDDEN))
        self.down_proj = nn.Parameter(
            torch.randn(EXPERTS, HIDDEN, INTER))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            mask = mask.permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for ei in hit:
            ei = ei[0]
            pos, tok = torch.where(mask[ei])
            gate, up = F.linear(
                hidden_states[tok], self.gate_up_proj[ei]).chunk(2, dim=-1)
            cur = F.linear(self.act_fn(gate) * up, self.down_proj[ei])
            cur = cur * top_k_weights[tok, pos, None]
            final.index_add_(0, tok, cur.to(final.dtype))
        return final


class _MoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _TinyRouter()
        self.experts = _TinyPackedExperts()


class TwoLayerMoe(nn.Module):
    """Minimal two-MoE-layer model so hook order matters.

    With one experts module there is no "earlier hook" to shift the shared
    stream; with two, BF16-ing layer 0's experts out of the hook set moves
    layer 1's draws, exactly the production shape (one layer's experts BF16d
    in the export assignment, all hooked in the frontier build).
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([_MoeBlock() for _ in range(LAYERS)])

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        del use_cache
        h = self.embed(input_ids).reshape(-1, HIDDEN)
        for block in self.layers:
            _logits, weights, indices = block.gate(h)
            h = block.experts(h, indices, weights)
        return h


def all_tensor_qnames() -> list[str]:
    return [
        f"layers.{i}.experts.{pn}"
        for i in range(LAYERS)
        for pn in ("gate_up_proj", "down_proj")
    ]


def all_module_qnames() -> set[str]:
    return {f"layers.{i}.experts" for i in range(LAYERS)}


def frozen_state() -> dict:
    torch.manual_seed(0)
    return {k: v.clone() for k, v in TwoLayerMoe().state_dict().items()}


def frozen_calib() -> torch.Tensor:
    torch.manual_seed(0)
    _ = TwoLayerMoe()  # consume the same stream prefix frozen_state() does
    return torch.randint(0, VOCAB, (N_CALIB, SEQLEN), dtype=torch.long)


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def run_arm(
    state: dict,
    calib: torch.Tensor,
    *,
    force_format: str | None = None,
    render_assignment: dict | None = None,
) -> dict:
    """Render packed experts one way; digest the rendered bytes per tensor."""
    model = TwoLayerMoe()
    model.load_state_dict(state)
    model.eval()
    cache = ProductionWeightCache(weights={}, levers=dict(LEVERS))
    coverage = fill_packed_expert_cache_entries(
        cache, model, calib,
        render_assignment=render_assignment,
        force_format=force_format,
        levers=dict(LEVERS),
        profile=None,
        module_token_budget=MODULE_TOKEN_BUDGET,
        progress=False,
    )
    return {
        "weights": {
            q: digest(t) for (q, f), t in cache.weights.items() if f == FORMAT
        },
        "coverage": coverage,
    }


def compare(got: dict, ref: dict) -> list[str]:
    return [
        q for q in sorted(got["weights"])
        if got["weights"][q] != ref["weights"].get(q)
    ]


def test_the_module_reservoir_is_actually_selecting(monkeypatch):
    """Without selection pressure both arms agree for an unrelated reason."""
    import prismaquant.measure_quant_cost as mqc

    seen_budgets: list[int] = []
    real_derive = mqc.derive_per_expert_activations

    def spy(experts_mod, X, parent_mod, **kwargs):
        seen_budgets.append(int(X.shape[0]))
        return real_derive(experts_mod, X, parent_mod, **kwargs)

    monkeypatch.setattr(mqc, "derive_per_expert_activations", spy)
    state, calib = frozen_state(), frozen_calib()
    run_arm(state, calib, force_format=FORMAT)
    total_tokens = int(calib.numel())
    assert seen_budgets, "no packed module activations reached the render"
    assert max(seen_budgets) == MODULE_TOKEN_BUDGET, (
        f"module inputs capped at {max(seen_budgets)}, not the "
        f"module_token_budget={MODULE_TOKEN_BUDGET}; the reservoir is not "
        "the binding constraint"
    )
    assert MODULE_TOKEN_BUDGET < total_tokens, (
        "every routed token fits in the budget; the reservoir never selects "
        "and an arm agreement would prove nothing"
    )


def test_the_packed_render_is_deterministic():
    """Without determinism no arm comparison is readable."""
    state, calib = frozen_state(), frozen_calib()
    reference = run_arm(state, calib, force_format=FORMAT)
    control = run_arm(state, calib, force_format=FORMAT)
    assert control["weights"] == reference["weights"]


def test_assignment_scoped_export_keeps_the_frontier_bytes():
    """#145: BF16-ing one layer's experts must not move the other's bytes.

    The frontier build hooks every packed-experts module (``force_format``
    ignores the assignment); the export build renders an assignment that
    BF16s layer 0's experts. Same visible modules, same calibration -- the
    layer-1 bytes must be identical. Under the pre-fix hook set the export
    arm hooks only layer 1, its reservoir draws at a different stream offset,
    and the Hessians (hence the bytes) differ.
    """
    state, calib = frozen_state(), frozen_calib()
    reference = run_arm(state, calib, force_format=FORMAT)
    assert set(reference["weights"]) == set(all_tensor_qnames())

    layer1 = [q for q in all_tensor_qnames() if q.startswith("layers.1.")]
    assigned = {
        q: (FORMAT if q in layer1 else "BF16") for q in all_tensor_qnames()
    }
    arm = run_arm(state, calib, render_assignment=assigned)
    assert set(arm["weights"]) == set(layer1)
    differing = compare(
        arm,
        {"weights": {q: reference["weights"][q] for q in layer1}},
    )
    assert not differing, (
        f"assignment-scoped packed render moved {len(differing)}/"
        f"{len(layer1)} layer-1 expert tensors vs the force_format "
        f"frontier build: {differing}. The hook set is narrowed by the "
        "render again (#145)."
    )


def test_the_hook_set_is_the_full_visible_module_set(monkeypatch):
    """Pin the fix behaviorally: both arms hook every visible module.

    The expected set is derived from the model (every packed-experts module
    the call can see), not restated -- a test that names today's modules
    would pass by restating the roster.
    """
    import prismaquant.production_weight_cache as pwc

    seen: list[set[str]] = []
    real_collector = pwc._PackedExpertActivationCollector

    def spy(model, experts_qnames, **kwargs):
        seen.append(set(experts_qnames))
        return real_collector(model, experts_qnames, **kwargs)

    monkeypatch.setattr(
        pwc, "_PackedExpertActivationCollector", spy)
    state, calib = frozen_state(), frozen_calib()

    from prismaquant.sensitivity_probe import _is_packed_experts_module

    probe = TwoLayerMoe()
    visible = {
        qname for qname, mod in probe.named_modules()
        if _is_packed_experts_module(mod, None)
    }
    assert len(visible) == LAYERS

    run_arm(state, calib, force_format=FORMAT)
    layer1 = [q for q in all_tensor_qnames() if q.startswith("layers.1.")]
    assigned = {
        q: (FORMAT if q in layer1 else "BF16") for q in all_tensor_qnames()
    }
    run_arm(state, calib, render_assignment=assigned)

    assert len(seen) == 2, f"expected two collector constructions, saw {seen}"
    for hooked in seen:
        assert hooked == visible, (
            f"collector hooked {sorted(hooked)}, not the full visible set "
            f"{sorted(visible)}; the render narrowed the hook set (#145)"
        )
