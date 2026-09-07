"""Exercise buffer preservation through the actual streaming exporter."""
import json
from types import SimpleNamespace

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from prismaquant import export_native_compressed as export
from prismaquant import layer_streaming, sensitivity_probe, streaming_model


def _export_fixture(tmp_path, monkeypatch):
    """Replace architecture construction only; source reads and emission are real."""
    expected = torch.tensor([0.5, 0.5 + 2.0 ** -12], dtype=torch.float32)
    source = tmp_path / "source"
    source.mkdir()
    save_file({
        "model.layers.0.expert_bias": expected,
        "model.layers.0.proj.weight": torch.eye(2, dtype=torch.bfloat16),
    }, str(source / "model.safetensors"))

    class Skeleton:
        @staticmethod
        def _from_config(config):
            model = nn.Module()
            model.model = nn.Module()
            layer = nn.Module()
            layer.proj = nn.Linear(2, 2, bias=False, device="meta")
            layer.register_buffer("expert_bias", torch.empty(2, device="meta", dtype=torch.float32))
            model.model.layers = nn.ModuleList([layer])
            return model

    profile = SimpleNamespace(
        requires_multimodal_skeleton=lambda: False,
        concat_merge_groups=lambda: (),
        live_to_recipe_name=lambda name: name,
        packed_expert_param_names=lambda: (),
    )
    from transformers import AutoConfig
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(sensitivity_probe, "stage_text_only", lambda path: path)
    monkeypatch.setattr(streaming_model, "_skeleton_config_and_class",
                        lambda config, **kw: (config, Skeleton))
    monkeypatch.setattr(streaming_model, "_init_rotary_inplace", lambda *a: None)
    monkeypatch.setattr(export, "_PRODUCTION_WEIGHT_CACHE", None)

    def run(cache=None):
        emitted = {}
        def sink(tensors):
            emitted.update(tensors)
        export.materialize_tensors_streaming(
            str(source), {}, profile=profile, bf16_passthrough=set(),
            device=torch.device("cpu"), tensor_sink=sink,
            export_cache_dir=None if cache is None else str(cache))
        path = tmp_path / "emitted.safetensors"
        save_file(emitted, str(path))
        return load_file(str(path))
    return run, expected


def test_streaming_export_serializes_original_fp32_buffer_bytes(tmp_path, monkeypatch):
    run, expected = _export_fixture(tmp_path, monkeypatch)
    emitted = run()
    actual = emitted["model.layers.0.expert_bias"]
    assert actual.dtype == torch.float32
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert emitted["model.layers.0.proj.weight"].dtype == torch.bfloat16


def test_streaming_export_invalidates_resume_from_old_buffer_policy(tmp_path, monkeypatch):
    run, expected = _export_fixture(tmp_path, monkeypatch)
    cache = tmp_path / "resume"
    real_read = layer_streaming._read_layer_to_device
    def old_read(*args, **kwargs):
        kwargs.pop("buffer_dtypes", None)
        return real_read(*args, **kwargs)
    # Produce the prior implementation's layer payload through the real exporter.
    with monkeypatch.context() as old:
        old.setattr(layer_streaming, "_read_layer_to_device", old_read)
        narrowed = run(cache)["model.layers.0.expert_bias"]
    assert not torch.equal(narrowed, expected)
    manifest = cache / "manifest.json"
    legacy = json.loads(manifest.read_text())
    legacy.pop("persistent_buffer_read_policy", None)
    manifest.write_text(json.dumps(legacy))

    # Same source/assignment/levers, but the corrected reader must execute.
    emitted = run(cache)["model.layers.0.expert_bias"]
    assert torch.equal(emitted.view(torch.uint8), expected.view(torch.uint8))
    assert json.loads(manifest.read_text())["persistent_buffer_read_policy"]
