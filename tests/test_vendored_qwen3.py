import pytest
import torch
import transformers
from packaging.version import Version


def _tiny_qwen3_config():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    return Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rope_theta=10000.0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


@pytest.mark.xfail(
    Version(transformers.__version__) >= Version("5.7"),
    reason="AutoModelForCausalLM.register() no longer overrides a natively "
           "supported model_type; register_qwen3() becomes a silent no-op. "
           "Known good on 5.6.0, known broken on 5.13.1 -- exact boundary "
           "unbisected. Tracked in issue #19; the silent no-op is the real "
           "defect, not the version.",
    strict=False,
)
def test_qwen3_auto_model_uses_vendored_causal_lm():
    import prismaquant  # noqa: F401
    from prismaquant.vendored.transformers_qwen3 import Qwen3ForCausalLM
    from transformers import AutoModelForCausalLM

    cfg = _tiny_qwen3_config()
    model = AutoModelForCausalLM.from_config(cfg)

    assert isinstance(model, Qwen3ForCausalLM)


def test_vendored_qwen3_rope_matches_upstream_on_cpu():
    from prismaquant.vendored.transformers_qwen3 import (
        Qwen3RotaryEmbedding as VendoredQwen3RotaryEmbedding,
    )
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3RotaryEmbedding as UpstreamQwen3RotaryEmbedding,
    )

    cfg = _tiny_qwen3_config()
    x = torch.zeros((2, 5, cfg.hidden_size), dtype=torch.float32)
    position_ids = torch.tensor(
        [
            [0, 1, 2, 7, 15],
            [3, 4, 8, 16, 31],
        ],
        dtype=torch.long,
    )

    upstream = UpstreamQwen3RotaryEmbedding(cfg)
    vendored = VendoredQwen3RotaryEmbedding(cfg)

    upstream_cos, upstream_sin = upstream(x, position_ids)
    vendored_cos, vendored_sin = vendored(x, position_ids)

    assert vendored.cos_cached.shape == (cfg.max_position_embeddings, cfg.head_dim)
    torch.testing.assert_close(vendored_cos, upstream_cos, rtol=0, atol=0)
    torch.testing.assert_close(vendored_sin, upstream_sin, rtol=0, atol=0)


def test_vendored_qwen3_model_refreshes_invalid_rope_cache():
    from prismaquant.vendored.transformers_qwen3 import Qwen3ForCausalLM

    cfg = _tiny_qwen3_config()
    model = Qwen3ForCausalLM(cfg)
    rope = model.model.rotary_emb

    rope.inv_freq.fill_(float("nan"))
    rope.cos_cached.fill_(float("nan"))
    rope.sin_cached.fill_(float("nan"))

    model._prismaquant_reset_rope_caches()

    assert torch.isfinite(rope.inv_freq).all()
    assert torch.isfinite(rope.cos_cached).all()
    assert torch.isfinite(rope.sin_cached).all()
