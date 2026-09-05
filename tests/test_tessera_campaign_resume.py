"""A resumed cost must belong to this run's actual encoding inputs."""
import json
import pickle
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

UNIT = "model.layers.0.proj"


def _main_fixture(monkeypatch, tmp_path, *, priced=False):
    from prismaquant import model_profiles, tessera_campaign, tessera_render
    from prismaquant.model_profiles import DefaultProfile

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.model.layers[0].proj = torch.nn.Linear(256, 32, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.model.layers[0].proj.weight.copy_(
            torch.randn(32, 256, generator=torch.Generator().manual_seed(186)))
    inputs = {
        "tokens": [torch.ones(1, 256, dtype=torch.long)],
        "text": "one draw",
        "rows": torch.randn(4, 256, generator=torch.Generator().manual_seed(183)),
        "hessian": torch.eye(256),
        # The calibration maximum behind the unit's static input_global_scale;
        # a scoring input of every W4A4 row, bound like the rows themselves.
        "max_abs": 3.0,
        "menu": ([SimpleNamespace(
            format_name="TESSERA_E4M3_K1_R1024", family="TESSERA_E4M3_K1",
            body_rate_q256=1024, bpp=4.0)] if priced else []),
    }
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: model)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    # The fresh run prices under the default static-scale policy so a test
    # can change the policy afterwards and see the identity refuse it.
    monkeypatch.delenv("PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE", raising=False)
    monkeypatch.setattr(model_profiles, "detect_profile", lambda _path: DefaultProfile())
    monkeypatch.setattr(tessera_render, "tessera_encoder_hessian_status", lambda: {
        "accepted": True, "reason": "CPU test fixture", "kwargs": [], "recipe": {},
    })
    monkeypatch.setattr(tessera_campaign, "_calibration_tokens",
                        lambda *_args: (inputs["tokens"], inputs["text"]))
    monkeypatch.setattr(tessera_campaign, "_collect_activations", lambda *_args, **kwargs: (
        {UNIT: inputs["rows"]},
        {UNIT: inputs["hessian"]} if kwargs["want_hessian"] else {},
        {UNIT: len(inputs["rows"]) if kwargs["want_hessian"] else 0},
        {UNIT: float(inputs["max_abs"])},
    ))
    monkeypatch.setattr(tessera_campaign, "expand_menus_for_targets",
                        lambda _weights, targets, **_kwargs: {
                            name: inputs["menu"] for name in targets})

    def unverified_anchors_reached_payload(anchors, *_args, **_kwargs):
        assert not anchors, "unverified checkpoint anchors reached the current cost payload"
        return {"costs": {}, "formats": []}

    if not priced:
        monkeypatch.setattr(tessera_campaign, "campaign_cost_payload",
                            unverified_anchors_reached_payload)
    checkpoint = tmp_path / "campaign.anchors.json"
    argv = ["--model", "synthetic-current-model", "--out", str(tmp_path / "cost.pkl"),
            "--cache-dir", str(tmp_path / "cache"), "--checkpoint", str(checkpoint),
            "--hessian", "off", "--menu-mode", "research", "--max-rounds", "1"]
    return tessera_campaign, checkpoint, argv, model, inputs


@pytest.mark.parametrize("checkpoint_unit", [UNIT, "model.layers.8.other_model"])
def test_main_refuses_unbound_checkpoint_before_accepting_anchors(
    monkeypatch, tmp_path, checkpoint_unit,
):
    campaign, checkpoint, argv, _model, _inputs = _main_fixture(monkeypatch, tmp_path)
    anchor = campaign.CampaignAnchor(
        qname=checkpoint_unit, family="TESSERA_E4M3_K1",
        format_name="TESSERA_E4M3_K1_R1024", body_rate_q256=1024,
        dloss=0.25, dloss_stderr=0.0, memory_bytes=8, bits_per_param=4.0,
        activation_contract="a8", activation_quantized=True, wire_bytes=8,
        seconds=1.0, hessian_applied=False,
    )
    checkpoint.write_text(json.dumps({"schema": campaign.SCHEMA, "anchors": [vars(anchor)]}))
    with pytest.raises(RuntimeError, match="checkpoint.*identity|resume.*identity"):
        campaign.main(argv)


def _fresh_priced_campaign(monkeypatch, tmp_path, *, hessian=False):
    fixture = _main_fixture(monkeypatch, tmp_path, priced=True)
    campaign, checkpoint, argv, _model, _inputs = fixture
    if hessian:
        argv[argv.index("--hessian") + 1] = "require"
    assert campaign.main(argv) == 0
    with (tmp_path / "cost.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["costs"][UNIT]["TESSERA_E4M3_K1_R1024"]["output_mse_measured"]
    return fixture, payload


def _forbid_reencode(monkeypatch, campaign):
    def forbidden(**_kwargs):
        pytest.fail("resume attempted another encode instead of validating the priced bytes")
    monkeypatch.setattr(campaign, "_measure_anchor", forbidden)


def test_main_resumes_identical_cost_and_wire_without_reencoding(monkeypatch, tmp_path):
    from prismaquant.cost_stage_checkpoint import MANIFEST_SCHEMA

    (campaign, checkpoint, argv, _model, _inputs), initial = _fresh_priced_campaign(
        monkeypatch, tmp_path)
    manifest = json.loads(checkpoint.read_text())
    assert manifest["schema"] == MANIFEST_SCHEMA
    original_manifest = checkpoint.read_bytes()
    wire = next((tmp_path / "cache" / "wire").glob("*.tessera"))
    original_wire = wire.read_bytes()
    _forbid_reencode(monkeypatch, campaign)
    # Output location and interruption limit are not encoding/scoring inputs.
    argv[argv.index("--out") + 1] = str(tmp_path / "resumed.pkl")
    assert campaign.main([*argv, "--deadline-seconds", "1"]) == 0
    with (tmp_path / "resumed.pkl").open("rb") as handle:
        resumed = pickle.load(handle)
    assert resumed["costs"] == initial["costs"]
    assert checkpoint.read_bytes() == original_manifest
    assert wire.read_bytes() == original_wire


def test_main_refuses_changed_hessian_values_under_same_draw(monkeypatch, tmp_path):
    (campaign, checkpoint, argv, _model, inputs), _payload = _fresh_priced_campaign(
        monkeypatch, tmp_path, hessian=True)
    original_manifest = checkpoint.read_bytes()
    _forbid_reencode(monkeypatch, campaign)
    inputs["hessian"][0, 0] += 1
    with pytest.raises(RuntimeError, match="checkpoint identity mismatch"):
        campaign.main(argv)
    assert checkpoint.read_bytes() == original_manifest


@pytest.mark.parametrize("changed", [
    "weight", "scoring_rows", "input_scale", "scale_policy", "corpus", "tokens",
    "hessian_mode", "menu", "recipe", "encoder_source", "prismaquant_source", "scope",
])
def test_main_refuses_changed_encoding_or_scoring_inputs(monkeypatch, tmp_path, changed):
    (campaign, checkpoint, argv, model, inputs), _payload = _fresh_priced_campaign(
        monkeypatch, tmp_path)
    original_manifest = checkpoint.read_bytes()
    _forbid_reencode(monkeypatch, campaign)
    if changed == "weight":
        with torch.no_grad():
            model.model.layers[0].proj.weight[0, 0] += 1
    elif changed == "scoring_rows":
        inputs["rows"][0, 0] += 1
    elif changed == "input_scale":
        # The served A-side contract is a scoring input: another calibration
        # maximum is another static input_global_scale for the unit.
        inputs["max_abs"] *= 2
    elif changed == "scale_policy":
        # ...and so is the env-resolved policy that turns the maximum into it.
        monkeypatch.setenv("PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE", "1")
    elif changed == "corpus":
        inputs["text"] = "different corpus, same selected tokens"
    elif changed == "tokens":
        inputs["tokens"][0][0, 0] += 1
    elif changed == "hessian_mode":
        argv[argv.index("--hessian") + 1] = "require"
    elif changed == "menu":
        inputs["menu"] = []
    elif changed == "recipe":
        recipe = campaign.th.encoder_recipe()
        monkeypatch.setattr(campaign.th, "encoder_recipe", lambda: {**recipe, "changed": True})
    elif changed == "encoder_source":
        from tessera import cached_unit
        monkeypatch.setattr(cached_unit, "encoder_source_sha256", lambda: "0" * 64)
    elif changed == "prismaquant_source":
        from prismaquant import production_weight_cache
        monkeypatch.setattr(production_weight_cache, "_production_cache_source_sha256",
                            lambda: "0" * 64)
    elif changed == "scope":
        argv.extend(["--tp-degree", "2"])
    with pytest.raises(RuntimeError, match="checkpoint identity mismatch"):
        campaign.main(argv)
    assert checkpoint.read_bytes() == original_manifest


@pytest.mark.parametrize("damage", ["missing", "bytes", "symlink"])
def test_main_refuses_missing_or_changed_priced_wire(monkeypatch, tmp_path, damage):
    (campaign, checkpoint, argv, _model, _inputs), _payload = _fresh_priced_campaign(
        monkeypatch, tmp_path)
    _forbid_reencode(monkeypatch, campaign)
    original_manifest = checkpoint.read_bytes()
    wire = next((tmp_path / "cache" / "wire").glob("*.tessera"))
    if damage == "missing":
        wire.unlink()
    elif damage == "bytes":
        blob = bytearray(wire.read_bytes())
        blob[-1] ^= 1
        wire.write_bytes(blob)
    else:
        moved = wire.with_suffix(".elsewhere")
        wire.rename(moved)
        wire.symlink_to(moved.name)
    with pytest.raises(RuntimeError, match="checkpoint cached wire"):
        campaign.main(argv)
    assert checkpoint.read_bytes() == original_manifest
