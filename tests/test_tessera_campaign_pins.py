"""The campaign's priced population must obey the profile's immutable floor."""
from types import ModuleType, SimpleNamespace
import sys

import pytest

torch = pytest.importorskip("torch")


class _CaptureBoundaryReached(Exception):
    pass


@pytest.mark.parametrize("custom_pin", [False, True])
def test_main_excludes_exactly_the_existing_profile_pins(monkeypatch, tmp_path, custom_pin):
    from prismaquant import model_profiles, tessera_campaign, tessera_render
    from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([layer])
    layer.conv = torch.nn.Module()
    layer.conv.in_proj = torch.nn.Linear(4, 4)
    layer.conv.out_proj = torch.nn.Linear(4, 4)
    layer.feed_forward = torch.nn.Module()
    layer.feed_forward.gate = torch.nn.Linear(4, 4)
    layer.feed_forward.w1 = torch.nn.Linear(4, 4)
    layer.self_attn = torch.nn.Module()
    layer.self_attn.q_proj = torch.nn.Linear(4, 4)
    profile = Lfm2MoeProfile()
    if custom_pin:
        monkeypatch.setattr(profile, "pinned_names", lambda: ("self_attn.q_proj.weight",))
    expected = {name for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and not profile.is_pinned_name(name)}
    assert len(expected) == (4 if custom_pin else 2)

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: model)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _path: profile)
    monkeypatch.setattr(tessera_render, "tessera_encoder_hessian_status",
                        lambda: {"accepted": True})
    monkeypatch.setattr(tessera_campaign, "_calibration_tokens",
                        lambda *_args: ([], "synthetic calibration"))

    def capture(_model, targets, *_args, **_kwargs):
        assert set(targets) == expected, "campaign attempted to price profile-pinned Linears"
        raise _CaptureBoundaryReached

    monkeypatch.setattr(tessera_campaign, "_collect_activations", capture)
    with pytest.raises(_CaptureBoundaryReached):
        tessera_campaign.main([
            "--model", "synthetic-profile", "--out", str(tmp_path / "cost.pkl"),
            "--cache-dir", str(tmp_path / "cache"), "--hessian", "off",
            "--menu-mode", "research",
        ])
