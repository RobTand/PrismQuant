"""A campaign must account for the live packed expert population."""
from types import ModuleType, SimpleNamespace
import sys

import pytest

torch = pytest.importorskip("torch")


class Lfm2MoeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.arange(64.).reshape(2, 8, 4) / 64)
        self.down_proj = torch.nn.Parameter(torch.arange(32.).reshape(2, 4, 4) / 32)
        self.num_experts = 2
        self.act_fn = torch.nn.functional.silu
        self.observed = {0: {"gate_up": [], "down": []},
                         1: {"gate_up": [], "down": []}}

    def forward(self, hidden, indices, weights):
        output = torch.zeros_like(hidden)
        for expert in range(self.num_experts):
            tokens, positions = torch.where(indices == expert)
            x = hidden[tokens]
            gate, up = torch.nn.functional.linear(
                x, self.gate_up_proj[expert]).chunk(2, dim=-1)
            down = self.act_fn(gate) * up
            self.observed[expert]["gate_up"].append(x.detach().clone())
            self.observed[expert]["down"].append(down.detach().clone())
            output.index_add_(0, tokens, torch.nn.functional.linear(
                down, self.down_proj[expert]) * weights[tokens, positions, None])
        return output


class _Router(torch.nn.Module):
    def forward(self, hidden):
        return torch.stack((hidden[:, 0], -hidden[:, 0]), dim=-1)


class _RoutedBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _Router()
        self.experts = Lfm2MoeExperts()

    def route_tokens_to_experts(self, logits):
        values, indices = torch.softmax(logits, dim=-1).topk(1, dim=-1)
        return indices, values / values.sum(dim=-1, keepdim=True)

    def forward(self, hidden):
        indices, weights = self.route_tokens_to_experts(self.gate(hidden))
        return self.experts(hidden, indices, weights)


class _RoutedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="lfm2_moe", architectures=[])
        self.model = _mixed_model().model
        self.model.layers[2].feed_forward = _RoutedBlock()

    def forward(self, hidden):
        hidden = hidden.reshape(-1, hidden.shape[-1])
        self.model.layers[0].attention(hidden)
        return self.model.layers[2].feed_forward(hidden)


EXPERT_PREFIX = "model.layers.2.feed_forward.experts"


def _routed_tokens():
    return [torch.tensor([[[1., 2., 0., 1.], [-1., 3., 1., 0.],
                           [2., 1., 1., 0.], [-2., 1., 0., 2.]]]),
            torch.tensor([[[3., 2., 0., 1.], [-3., 3., 1., 0.],
                           [4., 1., 1., 0.], [-4., 1., 0., 2.]]])]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_capture_packed_projections_uses_every_actual_routed_row(dtype):
    from prismaquant.tessera_campaign import _collect_activations

    model = _RoutedModel().to(dtype=dtype)
    targets = [f"{EXPERT_PREFIX}.{expert}.{projection}"
               for expert in range(2) for projection in ("w1", "w3", "w2")]
    dense = "model.layers.0.attention"
    tokens = [batch.to(dtype=dtype) for batch in _routed_tokens()]
    rows, hessians, counts = _collect_activations(
        model, [dense, *targets], tokens, 1, "cpu", want_hessian=True)
    from prismaquant.routed_experts import profile_declared_packed_expert_projections

    observed = model.model.layers[2].feed_forward.experts.observed
    units = {unit.qname: unit for unit in profile_declared_packed_expert_projections(model)}
    assert set(units) == set(targets)
    for expert in range(2):
        for projection, input_kind in (("w1", "gate_up"),
                                       ("w3", "gate_up"), ("w2", "down")):
            name = f"{EXPERT_PREFIX}.{expert}.{projection}"
            actual = torch.cat(observed[expert][input_kind]).float()
            assert actual.shape[0] == 4  # The score cap must be binding.
            assert counts[name] == actual.shape[0]
            torch.testing.assert_close(rows[name], actual[:1])
            torch.testing.assert_close(hessians[name], actual.T @ actual)
            unit = units[name]
            packed = getattr(unit.module, unit.param_name)
            expected = (packed[expert] if projection == "w2" else
                        packed[expert, :4] if projection == "w1" else packed[expert, 4:])
            torch.testing.assert_close(unit.weight, expected)
            assert unit.weight.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    flat = torch.cat(tokens).reshape(-1, 4).float()
    assert counts[dense] == 8
    torch.testing.assert_close(hessians[dense], flat.T @ flat)
    assert not model.model.layers[2].feed_forward.experts._forward_pre_hooks
    assert not model.model.layers[0].attention._forward_pre_hooks


def test_packed_capture_subset_keeps_the_same_routed_rows_and_hessian():
    from prismaquant.tessera_campaign import _collect_activations

    all_targets = [f"{EXPERT_PREFIX}.{expert}.{projection}"
                   for expert in range(2) for projection in ("w1", "w3", "w2")]
    selected = f"{EXPERT_PREFIX}.1.w2"
    whole = _collect_activations(
        _RoutedModel(), all_targets, _routed_tokens(), 2, "cpu", want_hessian=True)
    subset = _collect_activations(
        _RoutedModel(), [selected], _routed_tokens(), 2, "cpu", want_hessian=True)
    for index in (0, 1):
        torch.testing.assert_close(subset[index][selected], whole[index][selected])
    assert subset[2][selected] == whole[2][selected] == 4


@pytest.mark.parametrize("want_hessian", [False, True])
def test_packed_capture_refuses_an_unobserved_expert(want_hessian):
    from prismaquant.tessera_campaign import _collect_activations

    model = _RoutedModel()
    missing = f"{EXPERT_PREFIX}.1.w1"
    with pytest.raises(RuntimeError, match="no routed calibration rows") as error:
        _collect_activations(model, [missing], [torch.ones(1, 3, 4)], 1,
                             "cpu", want_hessian=want_hessian)
    assert missing in str(error.value)
    assert not model.model.layers[2].feed_forward.experts._forward_pre_hooks


def _mixed_model():
    net = torch.nn.Module()
    net.model = torch.nn.Module()
    net.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(3)])
    net.model.layers[0].attention = torch.nn.Linear(4, 4, bias=False)
    net.model.layers[2].feed_forward = torch.nn.Module()
    net.model.layers[2].feed_forward.experts = Lfm2MoeExperts()
    return net


@pytest.mark.parametrize("stride", [1, 2])
def test_main_refuses_missing_packed_population_before_calibration(monkeypatch, tmp_path, stride):
    from prismaquant import model_profiles, tessera_campaign, tessera_render
    from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile

    model = _mixed_model()
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: model)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _path: Lfm2MoeProfile())
    monkeypatch.setattr(tessera_render, "tessera_encoder_hessian_status",
                        lambda: {"accepted": True})

    def calibration_would_hide_the_omission(*_args, **_kwargs):
        raise AssertionError("started calibration with a dense-only campaign population")

    monkeypatch.setattr(tessera_campaign, "_calibration_tokens",
                        calibration_would_hide_the_omission)
    with pytest.raises(RuntimeError, match="packed expert population") as error:
        tessera_campaign.main([
            "--model", "synthetic-lfm", "--out", str(tmp_path / "cost.pkl"),
            "--cache-dir", str(tmp_path / "cache"), "--hessian", "off",
            "--menu-mode", "research", "--layer-stride", str(stride),
        ])
    for parameter in ("gate_up_proj", "down_proj"):
        assert f"model.layers.2.feed_forward.experts.{parameter}" in str(error.value)
    assert not (tmp_path / "cost.pkl").exists()
