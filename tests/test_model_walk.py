"""Discovery-walker acceptance and unit tests (R5 design contract §5).

The acceptance clients:

  a. A toy model with a bare-Parameter grouped einsum (the ``wo_a`` shape,
     ``[g, r, d]``): the edge is discovered, the walk fails while the
     parameter is unclaimed, and passes once a profile-style rule pins it
     with a reason. A second toy covers the ``nn.Linear`` decide path.
  b. Tiny vendored Qwen3 on meta under ``FakeTensorMode``: every
     ``nn.Linear`` weight appears as an edge, ``Qwen3Profile``'s claim rules
     cover 100% of nodes, and two runs serialize identically.
  c. DSv4 meta-load: attempted when the source checkpoint's config is on
     disk; skipped with the recorded reason otherwise (a+b carry acceptance).

Plus the unit tests the contract names: storage identity through a view and
a slice, an unresolved floating matmul operand failing the walk, and trace
coverage recording an unexecuted module.
"""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from prismaquant.model_walk import (
    ClaimRule,
    WalkError,
    walk_model,
)

WO_A_PIN_REASON = "grouped einsum operand: probe cannot price it yet"


class GroupedEinsumToy(nn.Module):
    """The wo_a shape: a bare Parameter consumed by a grouped einsum on a
    module class no Linear-keyed enumeration would visit."""

    def __init__(self, groups=4, rank=8, dim=16):
        super().__init__()
        self.proj = nn.Linear(groups * dim, groups * dim, bias=False)
        self.wo_a = nn.Parameter(torch.empty(groups, rank, dim))
        self.groups, self.dim = groups, dim

    def forward(self, x):
        y = self.proj(x)
        y = y.view(*y.shape[:-1], self.groups, self.dim)
        return torch.einsum("bsgd,grd->bsgr", y, self.wo_a)


def _meta(model_cls, *args, **kwargs):
    with torch.device("meta"):
        model = model_cls(*args, **kwargs)
    return model.eval()


def _toy_inputs(features=64, device="meta"):
    return (torch.zeros(1, 2, features, device=device),)


LINEAR_DECIDE = ClaimRule(
    "decide", "nn.Linear weight: allocator decision",
    module_class="Linear", leaf="weight")


# ------------------------------------------------------------ acceptance a


def test_grouped_einsum_edge_is_discovered():
    result = walk_model(
        _meta(GroupedEinsumToy), _toy_inputs(), strict=False)
    edges = result.edges_for("wo_a")
    assert edges, "the bare-Parameter einsum operand was not discovered"
    edge = edges[0]
    assert edge.op == "einsum"
    assert edge.equation == "bsgd,grd->bsgr"
    assert edge.role == "multiplicand"
    assert edge.operand_shape == (4, 8, 16)


def test_unclaimed_grouped_einsum_fails_the_walk_with_name_and_op():
    with pytest.raises(WalkError) as excinfo:
        walk_model(
            _meta(GroupedEinsumToy), _toy_inputs(),
            claim_rules=[LINEAR_DECIDE])
    message = str(excinfo.value)
    assert "wo_a" in message
    assert "einsum" in message
    assert "bsgd,grd->bsgr" in message
    result = excinfo.value.result
    assert any(f.node == "wo_a" and f.kind == "unclaimed"
               for f in result.failures)


def test_pinned_grouped_einsum_passes_with_the_reason_recorded():
    rules = [
        ClaimRule("pin", WO_A_PIN_REASON, name_regex=r"^wo_a$"),
        LINEAR_DECIDE,
    ]
    result = walk_model(_meta(GroupedEinsumToy), _toy_inputs(),
                        claim_rules=rules)
    assert result.ok
    claim = result.claims["wo_a"]
    assert claim.disposition == "pin"
    assert claim.reason == WO_A_PIN_REASON


def test_linear_weight_is_claimed_decide_and_edged():
    result = walk_model(
        _meta(GroupedEinsumToy), _toy_inputs(),
        claim_rules=[
            ClaimRule("pin", WO_A_PIN_REASON, name_regex=r"^wo_a$"),
            LINEAR_DECIDE,
        ])
    assert result.claims["proj.weight"].disposition == "decide"
    (edge,) = result.edges_for("proj.weight")
    assert edge.op == "linear"
    assert edge.role == "multiplicand"
    assert edge.stored_bytes == 64 * 64 * 4


# ------------------------------------- storage identity: views and slices


class ViewConsumer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.empty(8, 8))

    def forward(self, x):
        return x @ self.w.view(8, 8).transpose(0, 1)


class SliceConsumer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Parameter(torch.empty(4, 8, 8))

    def forward(self, x):
        return x @ self.stack[2]


def test_view_resolves_to_the_parent_parameter():
    result = walk_model(
        _meta(ViewConsumer), _toy_inputs(features=8),
        claim_rules=[ClaimRule("decide", "test", name_regex=r"^w$")])
    (edge,) = result.edges_for("w")
    assert edge.op == "matmul"
    # `via` records the hop that crossed the parameter->trace boundary; the
    # in-trace transpose aliases the same storage, so it adds no hop.
    assert "view" in edge.via


def test_slice_resolves_to_the_parent_parameter():
    result = walk_model(
        _meta(SliceConsumer), _toy_inputs(features=8),
        claim_rules=[ClaimRule("decide", "test", name_regex=r"^stack$")])
    (edge,) = result.edges_for("stack")
    assert edge.op == "matmul"
    assert edge.operand_shape == (8, 8)      # the slice, not the stack
    assert result.node("stack").shape == (4, 8, 8)
    assert "__getitem__" in edge.via


# --------------------------------------------- unresolved floating operand


class ShadowWeight(nn.Module):
    """A weight reconstructed at init time: `.to()` broke storage identity
    before the walk could index it. The walk must report it, not guess."""

    def __init__(self):
        super().__init__()
        source = nn.Parameter(torch.empty(8, 8))
        self.shadow = source.detach().to(torch.float32)  # a plain attribute

    def forward(self, x):
        return x @ self.shadow


def test_unresolved_floating_matmul_operand_fails():
    with pytest.raises(WalkError) as excinfo:
        walk_model(_meta(ShadowWeight), _toy_inputs(features=8))
    assert "cannot name" in str(excinfo.value)
    result = excinfo.value.result
    assert any(f.kind == "unresolved" and f.op == "matmul"
               for f in result.failures)
    (operand,) = result.unresolved_operands
    assert operand.is_floating
    assert operand.operand_shape == (8, 8)


def test_activations_are_not_reported_unresolved():
    result = walk_model(
        _meta(GroupedEinsumToy), _toy_inputs(), strict=False)
    assert not result.unresolved_operands


# ------------------------------------------------- bias and embedding rules


class BiasedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8, bias=True)

    def forward(self, x):
        return self.lin(x)


def test_bias_is_recorded_additive_and_never_a_failure():
    result = walk_model(
        _meta(BiasedLinear), _toy_inputs(features=8),
        claim_rules=[LINEAR_DECIDE])
    assert result.ok
    assert "lin.bias" in result.unclaimed  # visible, not fatal
    (edge,) = result.edges_for("lin.bias")
    assert edge.role == "additive"


class EmbeddingOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 8)
        self.head = nn.Linear(8, 10, bias=False)

    def forward(self, ids):
        return self.head(self.emb(ids))


def test_embedding_weight_requires_a_claim_but_produces_no_edge():
    model = _meta(EmbeddingOnly)
    ids = torch.zeros(1, 2, dtype=torch.long, device="meta")
    with pytest.raises(WalkError) as excinfo:
        walk_model(model, (ids,), claim_rules=[LINEAR_DECIDE])
    assert "emb.weight" in str(excinfo.value)
    result = excinfo.value.result
    assert not result.edges_for("emb.weight")
    (use,) = result.embedding_uses
    assert use.param == "emb.weight"

    rules = [
        ClaimRule("exclude", "row-gather, not a GEMM",
                  module_class="Embedding"),
        LINEAR_DECIDE,
    ]
    assert walk_model(model, (ids,), claim_rules=rules).ok


# ----------------------------------------------------------- trace coverage


class PartiallyExecuted(nn.Module):
    def __init__(self):
        super().__init__()
        self.used = nn.Linear(8, 8, bias=False)
        self.dormant = nn.Linear(8, 8, bias=False)  # never called

    def forward(self, x):
        return self.used(x)


def test_trace_coverage_records_the_unexecuted_module():
    result = walk_model(
        _meta(PartiallyExecuted), _toy_inputs(features=8), strict=False)
    assert "used" in result.trace_coverage.executed
    assert "dormant" in result.trace_coverage.not_executed
    # The dormant weight is still discovered by root A and can be claimed.
    assert result.node("dormant.weight").shape == (8, 8)


# ------------------------------------------- interceptor is mode-agnostic


def test_real_execution_discovers_the_same_edges():
    fake = walk_model(_meta(GroupedEinsumToy), _toy_inputs(), strict=False)
    real_model = GroupedEinsumToy().eval()
    for p in real_model.parameters():
        p.data.zero_()
    real = walk_model(
        real_model, _toy_inputs(device="cpu"),
        execution="real", strict=False)
    strip = lambda r: [
        (e.param, e.op, e.equation, e.role, e.operand_shape)
        for e in r.edges]
    assert strip(fake) == strip(real)


def test_result_is_json_serializable():
    result = walk_model(
        _meta(GroupedEinsumToy), _toy_inputs(), strict=False)
    payload = json.dumps(result.to_json_dict(), sort_keys=True)
    assert "wo_a" in payload


# ------------------------------------------------------------ acceptance b


def _tiny_qwen3():
    import prismaquant  # noqa: F401  (import-time register_qwen3)
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "qwen3",
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=64, rope_theta=10000.0,
        attention_dropout=0.0, tie_word_embeddings=False,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    return model.eval()


@pytest.fixture(scope="module")
def qwen3_walk():
    from prismaquant.model_profiles.qwen3 import Qwen3Profile

    model = _tiny_qwen3()
    rules = Qwen3Profile().walk_claim_rules()
    return model, rules, walk_model(model, claim_rules=rules)


def test_qwen3_walk_passes_with_profile_claims(qwen3_walk):
    _, _, result = qwen3_walk
    assert result.ok


def test_qwen3_every_linear_weight_is_an_edge(qwen3_walk):
    model, _, result = qwen3_walk
    linear_weights = {
        f"{name}.weight" for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    }
    assert len(linear_weights) == 2 * 7 + 1  # q/k/v/o + gate/up/down, lm_head
    edged = {e.param for e in result.edges} | {
        a for e in result.edges for a in e.param_aliases}
    missing = linear_weights - edged
    assert not missing, f"nn.Linear weights with no discovered edge: {missing}"


def test_qwen3_profile_claims_cover_every_node(qwen3_walk):
    _, _, result = qwen3_walk
    assert not result.unclaimed
    assert result.claims["lm_head.weight"].disposition == "pin"
    assert result.claims["model.embed_tokens.weight"].disposition == "exclude"
    assert result.claims[
        "model.layers.0.self_attn.q_proj.weight"].disposition == "decide"
    assert result.claims[
        "model.layers.0.input_layernorm.weight"].disposition == "exclude"


def test_qwen3_walk_is_stable_across_two_runs(qwen3_walk):
    model, rules, first = qwen3_walk
    second = walk_model(model, claim_rules=rules)
    assert json.dumps(first.to_json_dict(), sort_keys=True) == \
        json.dumps(second.to_json_dict(), sort_keys=True)


# ------------------------------------------------------------ acceptance c
#
# The DSv4 meta-load itself works on the host venv; what blocks the FAKE
# trace is a data-dependent scalar in the vendored forward
# (`modeling_deepseek_v4.py` DSA attention: `int(position_ids[0, 0])` ->
# `aten._local_scalar_dense` -> DataDependentOutputException, measured
# 2026-08-21 on torch 2.11). The contract's fallback applies: root B runs a
# real tiny-tensor CPU forward instead. The fake block is pinned as a
# ratchet so a torch/vendored change that lifts it turns the test red with
# an instruction to promote the fake path.

DSV4_CONFIG = "/home/rob/dq-runs/dsv4-flash-0731/source/config.json"


def test_dsv4_profile_rules_decide_the_grouped_linear_weight():
    """The wo_a claim after the grouped Fisher accumulator landed: the
    DSv4 spec no longer declares `DeepseekV4GroupedLinear` in
    `probe_skip_module_class_names` (it moved to
    `probe_grouped_module_class_names`, and the probe prices it through
    the grouped accumulator), so the base rules claim its weight as an
    ordinary allocator decision. The pin mechanism itself stays covered
    by the generic base-rule tests below."""
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile
    from prismaquant.model_walk import WalkNode, apply_claim_rules

    rules = DeepseekV4Profile().walk_claim_rules()
    wo_a = WalkNode(
        name="model.layers.3.self_attn.wo_a.weight", kind="parameter",
        persistent=True,
        shape=(2048, 64), dtype="torch.bfloat16", stored_bytes=2048 * 64 * 2,
        owner_module="model.layers.3.self_attn.wo_a",
        module_class="DeepseekV4GroupedLinear",
        module_class_mro=("DeepseekV4GroupedLinear", "Linear", "Module"),
        aliases=(),
    )
    claim = apply_claim_rules([wo_a], rules)[wo_a.name]
    assert claim.disposition == "decide"


def _shrunken_dsv4():
    """The real vendored DSv4 modeling code at toy dimensions: same classes,
    same forward, ~1.3M parameters, real CPU init. No checkpoint needed."""
    import prismaquant  # noqa: F401
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile
    from transformers import AutoConfig, AutoModelForCausalLM

    profile = DeepseekV4Profile()
    profile.register_vendored_modeling()
    cfg = AutoConfig.for_model(
        "deepseek_v4",
        vocab_size=512, hidden_size=128, intermediate_size=256,
        moe_intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=1, head_dim=32,
        qk_rope_head_dim=16, q_lora_rank=32, o_lora_rank=32, o_groups=4,
        index_n_heads=4, index_head_dim=16, index_topk=8,
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
        num_hash_layers=1, sliding_window=8, max_position_embeddings=256,
        compress_ratios=[0, 0, 4, 128], rope_theta=10000,
        compress_rope_theta=160000, first_k_dense_replace=1,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return profile, AutoModelForCausalLM.from_config(cfg).eval()


@pytest.mark.slow
def test_dsv4_real_cpu_walk_discovers_and_decides_wo_a():
    """Acceptance c on the contract's root-B fallback: the real DSv4
    modeling code, real tiny tensors, CPU. Discovers the `wo_a` bmm edge;
    since the grouped Fisher accumulator landed, its claim is `decide` —
    priced, not pinned-with-debt. The OTHER matmul-fed bare-Parameter
    families (router gates, mHC mixers) stay pinned."""
    profile, model = _shrunken_dsv4()
    result = walk_model(
        model, execution="real", seq_len=16,
        claim_rules=profile.walk_claim_rules())
    assert result.ok

    wo_a_edges = [e for e in result.edges if ".wo_a." in e.param]
    assert len(wo_a_edges) == 4  # one per layer
    for edge in wo_a_edges:
        assert edge.op == "bmm"
        assert result.claims[edge.param].disposition == "decide"

    # The families the walk discovered beyond wo_a — each was a silent
    # omission candidate until claimed.
    for name in ("model.layers.1.mlp.gate.weight",
                 "model.layers.1.attn_hc.fn",
                 "model.hc_head.hc_fn"):
        assert result.claims[name].disposition == "pin", name


@pytest.mark.slow
def test_dsv4_walk_fails_without_the_profile_rules():
    """The wo_a defect, reproduced: with only the generic Linear rule, the
    walk refuses with wo_a (and its bmm) named — instead of shipping it."""
    from prismaquant.model_walk import ClaimRule

    _, model = _shrunken_dsv4()
    rules = [
        ClaimRule("pin", "test", predicate=lambda n: len(n.shape) < 2
                  or "emb" in n.name or "lm_head" in n.name),
        ClaimRule("pin", "router/mixer test claim",
                  module_class="DeepseekV4TopKRouter"),
        ClaimRule("pin", "router/mixer test claim",
                  module_class="DeepseekV4HashRouter"),
        ClaimRule("pin", "router/mixer test claim",
                  module_class="DeepseekV4HyperConnection"),
        ClaimRule("pin", "router/mixer test claim",
                  module_class="DeepseekV4HyperHead"),
        ClaimRule("exclude", "not a weight", floating=False),
        LINEAR_DECIDE,
    ]
    # LINEAR_DECIDE claims wo_a too (GroupedLinear subclasses nn.Linear),
    # so drop to a rule set that skips GroupedLinear the way the probe does.
    rules[-1] = ClaimRule(
        "decide", "nn.Linear weight", leaf="weight", module_class="Linear",
        predicate=lambda n: "GroupedLinear" not in n.module_class)
    with pytest.raises(WalkError) as excinfo:
        walk_model(model, execution="real", seq_len=16, claim_rules=rules)
    message = str(excinfo.value)
    assert "wo_a" in message
    assert "bmm" in message


@pytest.mark.slow
def test_dsv4_fake_trace_block_is_still_real():
    """Ratchet on the documented block: the fake trace of the real source
    config stops at `int(position_ids[0, 0])` (DataDependentOutputException
    on `aten._local_scalar_dense`). If torch or the vendored code ever
    lifts it, this turns red: switch acceptance c to the fake path and
    delete the real-CPU fallback note from the module docstring."""
    import pathlib

    from torch._subclasses.fake_tensor import DataDependentOutputException

    if not pathlib.Path(DSV4_CONFIG).is_file():
        pytest.skip(f"DSv4 source config not on this host: {DSV4_CONFIG}")
    import prismaquant  # noqa: F401
    from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile
    from transformers import AutoConfig, AutoModelForCausalLM

    profile = DeepseekV4Profile()
    profile.register_vendored_modeling()
    cfg = AutoConfig.from_pretrained(str(pathlib.Path(DSV4_CONFIG).parent))
    cfg.num_hidden_layers = 4  # walk the repeating cell, not 43 copies
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    model.eval()

    with pytest.raises(DataDependentOutputException):
        walk_model(model, claim_rules=profile.walk_claim_rules(),
                   strict=False)
