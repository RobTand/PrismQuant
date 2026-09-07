"""Explicit streaming attention selection survives both HF construction routes."""
import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

import prismaquant.streaming_model as streaming


@pytest.mark.parametrize('resolved_class', [False, True])
@pytest.mark.parametrize('backend', [None, 'eager'])
def test_streaming_skeleton_preserves_requested_or_default_backend(tmp_path, monkeypatch, resolved_class, backend):
    config = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64)
    config.architectures = ['Qwen2ForCausalLM']
    source = tmp_path / 'source'
    config.save_pretrained(source)
    stage = tmp_path / 'stage'
    stage.mkdir()
    monkeypatch.setenv('PRISMAQUANT_TMPDIR', str(stage))
    expected = AutoModelForCausalLM.from_config(config).config._attn_implementation
    if resolved_class:
        monkeypatch.setattr(streaming, '_skeleton_config_and_class',
            lambda cfg, **kwargs: (cfg, Qwen2ForCausalLM))
    class BuiltSkeleton(Exception):
        pass
    observed = {}
    def stop_after_build(model):
        observed['backend'] = model.config._attn_implementation
        raise BuiltSkeleton()
    monkeypatch.setattr(streaming, '_get_layer_list', stop_after_build)
    kwargs = {} if backend is None else {'attn_implementation': backend}
    with pytest.raises(BuiltSkeleton):
        streaming._build_streaming_context(str(source), device=torch.device('cpu'),
            dtype=torch.bfloat16, offload_folder=str(tmp_path / 'offload'), **kwargs)
    assert observed['backend'] == (backend if backend is not None else expected)
