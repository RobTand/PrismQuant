"""A campaign must account for the live packed expert population."""
from types import ModuleType, SimpleNamespace
import sys

import pytest

torch = pytest.importorskip("torch")


class Lfm2MoeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.ones(2, 8, 4))
        self.down_proj = torch.nn.Parameter(torch.ones(2, 4, 4))


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
