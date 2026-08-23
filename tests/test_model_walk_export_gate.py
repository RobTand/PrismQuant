"""The discovery-walker export gate + walk-artifact contract (R5).

Proves the pieces `docs/design/model_coverage_ledgers.md` line 11-12 calls
the open scope, minus the consumer migration:

- the gate REFUSES on an unclaimed matmul-fed node (name + op cited,
  structured fields only — never prose), on an unresolved floating
  multiplicand, on an unknown future failure kind (the Tensor-Parallel
  catch-all), and on a decided-but-unpriced contradiction (the gemma4
  router polarity class);
- the override excuses trace incompleteness ONLY — never a claim failure;
- WalkResult round-trips through save_walk/load_walk fail-closed, with
  trace-time provenance and the applied claim-rule list in the envelope;
- every registered profile's claim rules carry the universal router-pin and
  packed-expert-decide families the R5 sweep found missing;
- pipeline.py declares the gate stage before export, run-pipeline.sh runs it
  there, and both wirings are load-bearing (removal turns these red).
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

import pytest
import torch
import torch.nn as nn

from prismaquant.model_walk import (
    BYTE_POLICIES,
    SCHEMA,
    WALK_GATE_OVERRIDE_ENV,
    WALK_GATE_SCHEMA,
    ClaimRule,
    LoadedWalk,
    WalkFailure,
    WalkGateRefusal,
    capture_walk_provenance,
    claim_rules_to_json,
    evaluate_walk_gate,
    find_decided_but_unpriced,
    load_walk,
    per_device_bytes,
    require_walk_coverage,
    save_walk,
    walk_model,
)
from prismaquant.model_walk import _default_example_inputs
from test_model_walk import (
    LINEAR_DECIDE,
    GroupedEinsumToy,
    WO_A_PIN_REASON,
    _meta,
    _toy_inputs,
)

FULLY_CLAIMED = [
    ClaimRule("pin", WO_A_PIN_REASON, name_regex=r"^wo_a$"),
    LINEAR_DECIDE,
]


def _run_pipeline_script() -> str:
    root = pathlib.Path(__file__).resolve().parents[1]
    return (root / "prismaquant" / "run-pipeline.sh").read_text()


def _walked(**kwargs):
    model = kwargs.pop("model", None) or _meta(GroupedEinsumToy)
    return walk_model(
        model, kwargs.pop("example_inputs", _toy_inputs()),
        claim_rules=kwargs.pop("claim_rules", FULLY_CLAIMED),
        strict=False, **kwargs)


# ---------------------------------------------------------------- gate


def test_gate_refuses_unclaimed_matmul_fed_node_with_name_and_op():
    """THE acceptance test: no claim rule for wo_a -> the gate refuses, with
    the node named and the op cited, as STRUCTURED fields. Removing the gate
    function (or either wiring) turns this file red."""
    result = _walked(claim_rules=[LINEAR_DECIDE])
    assert result.failures, "walk must report the unclaimed einsum operand"
    verdict = evaluate_walk_gate(result)
    assert verdict.refused
    assert "unclaimed_node" in verdict.refusal_kinds
    entry = {
        "node": "wo_a",
        "op": "einsum",
        "equation": "bsgd,grd->bsgr",
        "module": "",
    }
    assert entry in verdict.provenance["unclaimed_matmul_fed_nodes"]
    with pytest.raises(WalkGateRefusal) as excinfo:
        require_walk_coverage(result)
    # The human-facing message names what the structured fields already say.
    assert "wo_a" in str(excinfo.value)
    assert "einsum" in str(excinfo.value)


def test_fully_claimed_model_passes_the_gate():
    result = _walked()
    assert result.ok
    verdict = evaluate_walk_gate(result)
    assert not verdict.refused
    assert verdict.provenance["refusal_kinds"] == []
    counts = verdict.provenance["claims_by_disposition"]
    assert counts == {"decide": 1, "pin": 1, "exclude": 0}
    require_walk_coverage(result)  # does not raise


def test_unknown_failure_kind_refuses_fail_closed():
    """The TP catch-all: a failure kind this gate version does not know must
    refuse rather than pass. A future 'tp_group_boundary_misaligned' category
    makes even an UNUPGRADED gate loud."""
    result = dataclasses.replace(_walked(), failures=(
        WalkFailure(
            kind="tp_group_boundary_misaligned",
            node="model.layers.0.mlp.experts.gate_up_proj",
            op="linear", equation=None,
            module="model.layers.0.mlp.experts",
            detail="group 64 crosses shard boundary at tp_degree=4"),
    ))
    verdict = evaluate_walk_gate(result)
    assert verdict.refused
    assert "unknown_walk_failure_kind" in verdict.refusal_kinds
    assert "tp_group_boundary_misaligned" in \
        verdict.provenance["failure_kinds_seen"]
    assert "tp_group_boundary_misaligned" in verdict.refusal_reason


def test_override_excuses_trace_incompleteness_only():
    aborted = evaluate_walk_gate(None, trace_status="incomplete")
    assert aborted.refused
    assert "incomplete_trace" in aborted.refusal_kinds

    excused = evaluate_walk_gate(
        None, trace_status="incomplete",
        trace_error_class="DataDependentOutputException",
        override_reason="DSA position scalar aborts the fake trace")
    assert not excused.refused
    assert excused.provenance["override"] == {
        "env": WALK_GATE_OVERRIDE_ENV,
        "reason": "DSA position scalar aborts the fake trace",
    }

    # ...but an override can NEVER excuse a claim failure.
    unclaimed = _walked(claim_rules=[LINEAR_DECIDE])
    still_refused = evaluate_walk_gate(unclaimed, override_reason="try me")
    assert still_refused.refused
    assert "unclaimed_node" in still_refused.refusal_kinds
    assert still_refused.provenance.get("override_excused_trace_only") is None


def test_verdict_is_independent_of_prose_detail_strings():
    """House rule: the gate branches on structured fields, never prose."""
    base = _walked(claim_rules=[LINEAR_DECIDE])
    shouted = dataclasses.replace(base, failures=tuple(
        dataclasses.replace(f, detail=f.detail.upper() + "!!!")
        for f in base.failures))
    assert evaluate_walk_gate(base).provenance == \
        evaluate_walk_gate(shouted).provenance


def test_byte_accounting_declares_total_logical_convention():
    verdict = evaluate_walk_gate(_walked())
    accounting = verdict.provenance["byte_accounting"]
    assert accounting["convention"] == "total_logical_tensor_bytes"
    assert "shard_policy" in accounting  # the named, reserved TP seam
    assert verdict.provenance["decision_unit"] == "whole_logical_tensor"


def test_scope_is_structured_and_stamped():
    verdict = evaluate_walk_gate(_walked(), scope={
        "profile": "qwen3", "rules_source": "profile"})
    assert verdict.provenance["scope"]["profile"] == "qwen3"


# ------------------------------------------------------------- byte seam


def test_per_device_bytes_seam():
    assert per_device_bytes(100, 1, "replicated") == 100   # identity at tp=1
    assert per_device_bytes(100, 1, "sharded_evenly") == 100
    assert per_device_bytes(100, 4, "replicated") == 100
    assert per_device_bytes(100, 4, "sharded_evenly") == 25
    with pytest.raises(ValueError):  # non-dividing shard boundary is loud
        per_device_bytes(100, 3, "sharded_evenly")
    for bad_call in (
        lambda: per_device_bytes(100, 0, "sharded_evenly"),
        lambda: per_device_bytes(100, 2, "row"),       # no silent policies
        lambda: per_device_bytes(100, 2, ""),          # ...and no default
    ):
        with pytest.raises(ValueError):
            bad_call()
    assert set(BYTE_POLICIES) == {"replicated", "sharded_evenly"}


# --------------------------------------------------- artifact extensions


def test_save_load_roundtrip_preserves_the_result_payload(tmp_path):
    result = _walked()
    provenance = result.provenance
    assert provenance is not None
    path = save_walk(result, tmp_path / "walk.json", provenance=provenance)
    loaded = load_walk(path, expect_claim_rules=FULLY_CLAIMED)
    assert isinstance(loaded, LoadedWalk)
    assert json.dumps(loaded.result.to_json_dict(), sort_keys=True) == \
        json.dumps(result.to_json_dict(), sort_keys=True)
    assert loaded.provenance.execution == result.execution
    assert loaded.provenance.rules_digest_matches(FULLY_CLAIMED)


def test_load_walk_refuses_foreign_schema(tmp_path):
    path = save_walk(_walked(), tmp_path / "walk.json",
                     provenance=_walked().provenance)
    payload = json.loads(path.read_text())
    payload["schema"] = "prismaquant.model_walk.v999"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported model-walk schema"):
        load_walk(path)


def test_load_walk_refuses_execution_mismatch(tmp_path):
    result = _walked()
    path = save_walk(result, tmp_path / "walk.json",
                     provenance=result.provenance)
    payload = json.loads(path.read_text())
    payload["provenance"]["execution"] = "real"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="disagrees"):
        load_walk(path)


def test_load_walk_refuses_different_claim_rules_digest(tmp_path):
    result = _walked()
    path = save_walk(result, tmp_path / "walk.json",
                     provenance=result.provenance)
    other_rules = [ClaimRule("pin", "different policy", name_regex=".*")]
    with pytest.raises(ValueError, match="different claim rules"):
        load_walk(path, expect_claim_rules=other_rules)
    # And the happy direction: matching rules load clean.
    load_walk(path, expect_claim_rules=FULLY_CLAIMED)


def test_provenance_is_captured_at_trace_time():
    result = _walked()
    prov = result.provenance
    assert prov is not None
    assert prov.execution == "fake"
    # Explicit inputs are recorded by digest of their shapes/dtypes...
    assert prov.example_inputs_spec.startswith("provided:")
    assert prov.seq_len is None
    # ...and the synthesized default records its contract verbatim (this is
    # load-bearing: two walks under different input contracts can differ in
    # trace_coverage when data-dependent control flow executes shape-wise).
    default_prov = capture_walk_provenance(
        _meta(GroupedEinsumToy), execution="fake",
        example_inputs=_default_example_inputs(
            _meta(GroupedEinsumToy), 8, __import__("torch").device("meta")),
        seq_len=8, claim_rules=FULLY_CLAIMED, used_default_inputs=True)
    assert default_prov.example_inputs_spec.startswith(
        "default:input_ids(1,8)")
    assert default_prov.seq_len == 8
    assert prov.torch_version == torch.__version__
    assert prov.model_identity  # architecture/config digest present
    assert prov.claim_rules_digest
    # The applied rule list survives serialization: index, reason, and
    # predicates recorded by identity (None only where there is none).
    serialized = claim_rules_to_json(FULLY_CLAIMED)
    assert [r["index"] for r in serialized] == [0, 1]
    assert serialized[0]["reason"] == WO_A_PIN_REASON
    assert all(r["predicate"] is None for r in serialized)
    with_predicate = claim_rules_to_json([ClaimRule(
        "pin", "pinned by name predicate",
        predicate=lambda node: node.name == "wo_a")])
    identity = with_predicate[0]["predicate"]
    assert identity["name"] == "<lambda>"
    assert "test_model_walk_export_gate.py:" in identity["location"]


def test_to_json_dict_stays_deterministic_with_provenance_attached():
    first = _walked()
    second = walk_model(
        _meta(GroupedEinsumToy), _toy_inputs(),
        claim_rules=FULLY_CLAIMED, strict=False)
    assert first.provenance is not None
    assert second.provenance is not None
    assert json.dumps(first.to_json_dict(), sort_keys=True) == \
        json.dumps(second.to_json_dict(), sort_keys=True)
    # The envelope carries the timestamps; the payload above never does.
    blob = json.dumps(first.to_json_dict())
    assert first.provenance.created_utc[:4] not in blob


# ------------------------------------------- R5 sweep: claim-rule families


class SweepTopKRouter(nn.Module):
    """The sweep's Finding 1 shape: a bare Parameter fed to F.linear."""

    def __init__(self, experts=8, hidden=8):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(experts, hidden))

    def forward(self, x):
        return torch.nn.functional.linear(x.float(), self.weight.float())


class SweepExperts(nn.Module):
    """Finding 2: one 3-D packed stack consumed by an expert-slice matmul."""

    def __init__(self, rows=8, hidden=8):
        super().__init__()
        self.gate_proj = nn.Parameter(torch.empty(rows, hidden, hidden))

    def forward(self, x):
        return x @ self.gate_proj[0]


class SweepModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = SweepTopKRouter()
        self.experts = SweepExperts()

    def forward(self, x):
        return self.router(x) + self.experts(x)


def test_base_rules_claim_router_gates_and_packed_expert_stacks():
    """End-to-end over the two universal families the R5 sweep found
    unclaimed on six/seven profiles: with the base rules alone, the router
    pin and the packed-expert decide cover them and the walk passes."""
    from prismaquant.model_profiles.default import DefaultProfile

    rules = DefaultProfile().walk_claim_rules()
    model = _meta(SweepModel)
    result = walk_model(model, _toy_inputs(features=8), claim_rules=rules)
    assert result.ok
    assert result.claims["router.weight"].disposition == "pin"
    assert "never" in result.claims["router.weight"].reason
    assert result.claims["experts.gate_proj"].disposition == "decide"
    assert "packed-expert Fisher path" in \
        result.claims["experts.gate_proj"].reason
    # And the discovered edges exist for both (they are genuinely fed).
    assert any(e.op == "linear" for e in result.edges_for("router.weight"))
    assert any(e.param == "experts.gate_proj"
               for e in result.edges_for("experts.gate_proj"))


@pytest.mark.parametrize("profile_cls_name", [
    "Qwen3_5DenseProfile", "Qwen3_5Profile", "Qwen3Profile", "Gemma4Profile",
    "Lfm2MoeProfile", "MiniMaxM2Profile", "DeepseekV4Profile",
    "HyV3Profile", "LagunaProfile",
])
def test_every_profile_carries_the_sweep_rule_families(profile_cls_name):
    """Per-profile enablement, provable without instantiating topologies:
    each profile's applied rule list contains a router pin and a packed-
    expert decide (its own explicit rule or the base family rule)."""
    import prismaquant.model_profiles.registry as registry

    cls = getattr(registry, profile_cls_name)
    profile = cls()
    rules = claim_rules_to_json(profile.walk_claim_rules())
    router_pins = [
        r for r in rules
        if r["disposition"] == "pin"
        and "router" in json.dumps(r["reason"] + str(r)).lower()
    ]
    expert_decides = [
        r for r in rules
        if r["disposition"] == "decide"
        and "expert" in json.dumps(r["reason"] + str(r)).lower()
    ]
    assert router_pins, f"{profile_cls_name} lost its router-pin rule"
    assert expert_decides, f"{profile_cls_name} lost its expert-decide rule"


# ------------------------------------- gemma4 polarity: decided-but-unpriced


class RouterBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.router(x)


class RouterLinearToy(nn.Module):
    """gemma4's polarity bug shape: the router IS an nn.Linear (so a
    Linear-decide rule claims it decide) while the probe's baseline name
    exclusion (\\.router.) removes it from pricing inventory."""

    def __init__(self):
        super().__init__()
        self.blk = RouterBlock()

    def forward(self, x):
        return self.blk(x)


def test_decided_but_unpriced_contradiction_refuses():
    from prismaquant.model_profiles.default import DefaultProfile

    model = _meta(RouterLinearToy)
    inputs = (_toy_inputs(features=8)[0],)
    linear_rule = ClaimRule(
        "decide", "nn.Linear weight", leaf="weight", module_class="Linear")
    result = walk_model(model, inputs, claim_rules=[linear_rule],
                        strict=False)
    assert result.ok  # the CLAIM table alone sees no problem...
    entries = find_decided_but_unpriced(result, model, DefaultProfile())
    assert [e["reason_code"] for e in entries] == ["probe_linear_excluded"]
    assert entries[0]["node"] == "blk.router.weight"

    verdict = evaluate_walk_gate(result, unpriced_decides=entries)
    assert verdict.refused
    assert "decided_but_unpriced_node" in verdict.refusal_kinds
    assert "blk.router.weight" in verdict.refusal_reason

    # The fix is correcting the CLAIM to a pin (as the base router rule now
    # does), which clears the contradiction without weakening anything:
    pinned = walk_model(model, inputs, claim_rules=[
        ClaimRule("pin", "routing logits; never priced",
                  predicate=lambda n: "router" in n.name.lower()),
        linear_rule,
    ], strict=False)
    assert pinned.ok
    assert pinned.claims["blk.router.weight"].disposition == "pin"
    assert find_decided_but_unpriced(pinned, model, DefaultProfile()) == ()
    require_walk_coverage(pinned, unpriced_decides=())


# ---------------------------------------------------------------- CLI


@pytest.mark.slow
def test_dsv4_real_topology_passes_the_gate_end_to_end():
    """The motivating model, through the whole gate: shrunken real-DSv4
    topology, real CPU forward, profile rules -> gate PASSES with wo_a a
    REAL DECISION (the grouped accumulator prices it, so the
    decided-but-unpriced checker accepts it), and the other pinned
    families stay out of its way."""
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from test_model_walk import _shrunken_dsv4

    profile, model = _shrunken_dsv4()
    result = walk_model(
        model, execution="real", seq_len=16,
        claim_rules=profile.walk_claim_rules())
    assert result.ok
    assert all(
        c.disposition == "decide" for n, c in result.claims.items()
        if ".wo_a." in n)
    unpriced = find_decided_but_unpriced(result, model, profile)
    assert unpriced == ()
    verdict = evaluate_walk_gate(
        result,
        unpriced_decides=unpriced,
        scope={"profile": "deepseek_v4", "rules_source": "profile"},
    )
    assert not verdict.refused
    assert verdict.provenance["scope"]["profile"] == "deepseek_v4"
    require_walk_coverage(result, unpriced_decides=find_decided_but_unpriced(
        result, model, profile))


# ---------------------------------------------------------------- CLI


@pytest.fixture(scope="module")
def tiny_qwen3_dir(tmp_path_factory):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "qwen3", vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=64, rope_theta=10000.0,
        attention_dropout=0.0, tie_word_embeddings=False,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    d = tmp_path_factory.mktemp("cli") / "tiny-qwen3"
    AutoModelForCausalLM.from_config(cfg).save_pretrained(d)
    return d


def test_cli_passes_clean_checkpoint_and_writes_structured_report(
        tiny_qwen3_dir):
    from prismaquant.model_walk import main

    out = tiny_qwen3_dir / "model_walk.json"
    rc = main(["--model", str(tiny_qwen3_dir),
               "--output", str(out), "--rules", "profile"])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["schema"] == WALK_GATE_SCHEMA
    assert report["walk_artifact_schema"] == SCHEMA
    assert report["gate"]["refused"] is False
    assert report["gate"]["scope"]["profile"] == "qwen3"
    assert report["gate"]["scope"]["router_pin_rules"] >= 1
    assert report["provenance"]["execution"] == "fake"
    assert report["provenance"]["seq_len"] == 8
    assert any(
        r["predicate"] for r in report["provenance"]["claim_rules"])


def test_cli_refuses_when_no_rules_claim_the_model(tiny_qwen3_dir):
    from prismaquant.model_walk import main

    out = tiny_qwen3_dir / "refuse.json"
    rc = main(["--model", str(tiny_qwen3_dir),
               "--output", str(out), "--rules", "none"])
    assert rc == 2
    report = json.loads(out.read_text())
    assert report["gate"]["refused"] is True
    assert "unclaimed_node" in report["gate"]["refusal_kinds"]
    first = report["gate"]["unclaimed_matmul_fed_nodes"][0]
    assert {"node", "op", "equation", "module"} <= set(first)


# ------------------------------------------------------------ wiring


def test_pipeline_contract_places_the_gate_before_export():
    from prismaquant.pipeline import default_production_pipeline_spec

    spec = default_production_pipeline_spec()
    names = [stage.name for stage in spec.stages]
    assert "export.walk_coverage_gate" in names
    assert names.index("export.walk_coverage_gate") < \
        names.index("export.native_compressed")
    assert spec.validate().ok
    stage = {s.name: s for s in spec.stages}["export.walk_coverage_gate"]
    assert stage.metadata["schema"] == WALK_GATE_SCHEMA
    assert stage.metadata["decision_unit"] == "whole_logical_tensor"
    assert "never_claims" in stage.metadata["override_scope"]


def test_run_pipeline_invokes_the_gate_before_every_export_lane():
    script = _run_pipeline_script()
    gate_pos = script.find('"${WALK_GATE_ARGS[@]}"')
    assert gate_pos > 0, "run-pipeline.sh no longer executes the walk gate"
    for exec_line in (
        '"${GGUF_EXPORT_ARGS[@]}"',
        '"${CB_EXPORT_ARGS[@]}"',
        '"${EXPORT_ARGS[@]}"',
    ):
        pos = script.find(exec_line)
        assert pos > 0, f"export invocation vanished: {exec_line}"
        assert gate_pos < pos, \
            f"the walk gate must precede the export lane ({exec_line})"
    # Real wiring, not a comment:
    assert "python3 -m prismaquant.model_walk" in script
