from __future__ import annotations

import pytest
import torch
import copy
import pickle
import hashlib

import prismaquant.aura_cost as aura
from prismaquant.joint_aura import validate_joint_aura_entry, paired_candidate_difference
from prismaquant.kl_fisher import fisher_probe_scalar
from prismaquant.perturbed_x_cache import _activation_qdq
from prismaquant import format_registry as fr
from prismaquant.production_weight_cache import ProductionWeightCache
from test_streamed_cost_checkpoints import (
    _DenseTinyLM, _dense_runner, _model_identity,
)


def _fixture(seed=85):
    torch.manual_seed(seed)
    state = _DenseTinyLM().eval().state_dict()
    model, context, runner = _dense_runner(state)
    weights = {
        (name, fmt): module.weight.detach().clone() + 0.03125
        for name, module in model.named_modules() if name.endswith(".proj")
        for fmt in ("FP8_E4M3", "NVFP4A16")
    }
    cache = ProductionWeightCache(weights=weights, levers={}, activation_max_abs={
        name: 1.0 for name, _ in weights
    })
    return model, context, runner, cache


def _run(runner, cache, **kwargs):
    return aura.compute_aura_cost_streamed(
        runner, torch.tensor([[1, 2, 3, 4]]),
        ["FP8_DYNAMIC", "NVFP4A16", "BF16"], n_probes=3,
        min_free_gib=0, production_cache=cache, joint_activation=True,
        model_identity=_model_identity("joint-source"), **kwargs,
    )


def test_joint_streamed_emits_complete_aligned_rows_and_zero_passthrough():
    model, context, runner, cache = _fixture()
    payload = _run(runner, cache)
    assert payload["provenance"]["joint_activation"] is True
    assert payload["provenance"]["cost_mode"] == "aura"
    assert context.active == set()
    assert context.max_active == 1
    for name, rows in payload["costs"].items():
        for fmt, row in rows.items():
            assert validate_joint_aura_entry(row)
            assert row["probe_ids"] == [7000, 7001, 7002]
            assert row["joint_operator_identity"]["qname"] == name
            assert row["joint_operator_identity"]["format"] == fmt
        assert rows["BF16"]["signed_per_probe"] == [0.0] * 3
        assert rows["NVFP4A16"]["joint_operator_identity"]["activation"]["quantizes_input"] is False


def test_joint_checkpoint_resume_and_refuses_changed_actual_render(tmp_path, monkeypatch):
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    _, _, runner, cache = _fixture()
    first = _run(runner, cache, checkpoint_dir=tmp_path)
    _, context, runner, cache = _fixture()
    second = _run(runner, cache, checkpoint_dir=tmp_path, resume=True)
    assert second["costs"] == first["costs"]
    assert context.install_calls == 0
    _, context, runner, cache = _fixture()
    next(iter(cache.weights.values())).add_(0.01)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _run(runner, cache, checkpoint_dir=tmp_path, resume=True)
    assert context.install_calls == 0


def test_joint_streamed_matches_full_model_output_residual_oracle(monkeypatch):
    from test_streamed_cost_checkpoints import _DenseLayer
    # Repeated calls to the same weight are a real autograd accumulation case.
    monkeypatch.setattr(_DenseLayer, "forward", lambda self, hidden_states, **kwargs:
                        torch.tanh(self.proj(hidden_states) + 0.5 * self.proj(-hidden_states)))
    model, _, runner, cache = _fixture()
    state = copy.deepcopy(model.state_dict())
    payload = _run(runner, cache)
    oracle_model = _DenseTinyLM(state).eval()
    captures = {}
    handles = []
    for name, module in oracle_model.named_modules():
        if not name.endswith(".proj"):
            continue
        def record(mod, args, output, name=name):
            output.retain_grad()
            captures.setdefault(name, []).append((args[0].detach(), output))
        handles.append(module.register_forward_hook(record))
    try:
        for probe_index in range(3):
            captures.clear()
            oracle_model.zero_grad(set_to_none=True)
            logits = oracle_model(torch.tensor([[1, 2, 3, 4]])).logits
            fisher_probe_scalar(logits, seed=7000 + probe_index, token_scope="all", temperature=1.0, distribution="rademacher").backward()
            for name, rows in payload["costs"].items():
                weight = oracle_model.get_submodule(name).weight.detach().double()
                for fmt in ("FP8_E4M3", "NVFP4A16"):
                    rendered = cache.get(name, fmt).double()
                    total = 0.0
                    for x, out in captures[name]:
                        qx = (_activation_qdq(x, fr.get_format(fmt), cache.activation_max_abs, name)
                              if fr.get_format(fmt).act_quant_changes_input else x)
                        residual = qx.double() @ rendered.T - x.double() @ weight.T
                        total += float((out.grad.double() * residual).sum())
                    assert rows[fmt]["signed_per_probe"][probe_index] == pytest.approx(total, rel=2e-5, abs=2e-9)
    finally:
        for handle in handles:
            handle.remove()


def test_joint_activation_identity_reduces_to_weight_only_cost():
    _, _, runner, cache = _fixture()
    joint = _run(runner, cache)
    _, _, runner, cache = _fixture()
    weight_only = aura.compute_aura_cost_streamed(
        runner, torch.tensor([[1, 2, 3, 4]]), ["NVFP4A16"], n_probes=3,
        min_free_gib=0, production_cache=cache, dw_dtype="float32",
    )
    for name in joint["costs"]:
        assert joint["costs"][name]["NVFP4A16"]["x2_per_probe"] == pytest.approx(weight_only["costs"][name]["NVFP4A16"]["x2_per_probe"], rel=1e-5, abs=1e-12)


def test_joint_interrupted_resume_preserves_signed_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    _, _, runner, cache = _fixture()
    expected = _run(runner, cache)
    writer = aura._write_aura_unit_checkpoint
    def interrupt(*args, **kwargs):
        writer(*args, **kwargs)
        raise RuntimeError("fixture interruption")
    monkeypatch.setattr(aura, "_write_aura_unit_checkpoint", interrupt)
    _, _, runner, cache = _fixture()
    with pytest.raises(RuntimeError, match="fixture interruption"):
        _run(runner, cache, checkpoint_dir=tmp_path)
    monkeypatch.setattr(aura, "_write_aura_unit_checkpoint", writer)
    _, _, runner, cache = _fixture()
    actual = _run(runner, cache, checkpoint_dir=tmp_path, resume=True)
    assert actual["costs"] == expected["costs"]


def test_joint_checkpoint_refuses_probe_alignment_even_valid_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "1" * 40)
    _, _, runner, cache = _fixture()
    _run(runner, cache, checkpoint_dir=tmp_path)
    path = next((tmp_path / "units").glob("*.pkl"))
    envelope = pickle.loads(path.read_bytes())
    state = pickle.loads(envelope["payload"])
    next(iter(state["joint_aura_rows"].values()))["probe_ids"].reverse()
    envelope["payload"] = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    envelope["payload_sha256"] = hashlib.sha256(envelope["payload"]).hexdigest()
    path.write_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
    _, context, runner, cache = _fixture()
    with pytest.raises(RuntimeError, match="probe alignment"):
        _run(runner, cache, checkpoint_dir=tmp_path, resume=True)
    assert context.install_calls == 0


def test_paired_difference_keeps_common_probe_covariance_and_refuses_reordering():
    _, _, runner, cache = _fixture()
    row = next(iter(_run(runner, cache)["costs"].values()))["FP8_E4M3"]
    same = paired_candidate_difference(row, row)
    assert same["paired_standard_error"] == 0
    assert same["mean_difference"] == 0
    assert row["predicted_dloss_stderr"] > 0
    wrong = copy.deepcopy(row)
    wrong["probe_ids"].reverse()
    with pytest.raises(ValueError, match="probe alignment"):
        paired_candidate_difference(row, wrong)


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(fisher_application_count=True),
    lambda row: row.update(predicted_dloss_stderr=float("nan")),
    lambda row: row.update(activation_pricing_applied=True),
    lambda row: row.update(output_mse=1.0),
    lambda row: row.update(act_dloss=1.0),
])
def test_joint_row_refuses_second_scalar_or_activation_application(mutation):
    _, _, runner, cache = _fixture()
    row = next(iter(_run(runner, cache)["costs"].values()))["FP8_E4M3"]
    mutation(row)
    with pytest.raises(ValueError, match="joint AURA"):
        validate_joint_aura_entry(row)
