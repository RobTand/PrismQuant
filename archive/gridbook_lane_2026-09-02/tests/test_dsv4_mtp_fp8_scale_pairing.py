"""Physical DSpark MTP FP8/MXFP4 source-pairing regression tests."""
from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from prismaquant.export_nvfp4_cb_streaming import _LazySkeleton
from prismaquant.layer_streaming import _build_fp8_scale_inv_map
from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile


def _e8m0_ones(shape: tuple[int, ...]) -> torch.Tensor:
    # E8M0 byte code 127 represents 2**0.  Construct through the byte view so
    # the test exercises the serialized dtype DSv4 actually uses.
    return torch.full(shape, 127, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )


def _write_source(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "architectures": ["DeepseekV4ForCausalLM"],
                "expert_dtype": "fp4",
                "quantization_config": {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                    "scale_fmt": "ue8m0",
                    "weight_block_size": [128, 128],
                },
            }
        )
    )
    generator = torch.Generator().manual_seed(31)
    tensors = {
        # Existing body behavior: checkpoint spelling maps into the live
        # model namespace.
        "layers.0.attn.wq_a.weight": torch.randn(
            128, 128, generator=generator
        ).to(torch.float8_e4m3fn),
        "layers.0.attn.wq_a.scale": _e8m0_ones((1, 1)),
        # DSpark source behavior: these stay in their canonical physical
        # ``mtp.*`` checkpoint namespace because MTP remains probe-excluded.
        "mtp.0.attn.wq_a.weight": torch.randn(
            128, 128, generator=generator
        ).to(torch.float8_e4m3fn),
        "mtp.0.attn.wq_a.scale": _e8m0_ones((1, 1)),
        "mtp.0.ffn.experts.0.w1.weight": torch.randint(
            -128,
            128,
            (4, 32),
            dtype=torch.int8,
            generator=generator,
        ),
        "mtp.0.ffn.experts.0.w1.scale": _e8m0_ones((4, 2)),
        # Body packed-expert mapping remains live-namespaced and decoded too.
        "layers.0.ffn.experts.0.w1.weight": torch.randint(
            -128,
            128,
            (4, 32),
            dtype=torch.int8,
            generator=generator,
        ),
        "layers.0.ffn.experts.0.w1.scale": _e8m0_ones((4, 2)),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))


def test_mtp_pairs_use_physical_names_while_body_mapping_is_unchanged(
    tmp_path,
):
    _write_source(tmp_path)
    profile = DeepseekV4Profile()

    # This invariant keeps DSv4 MTP out of the body probe/cost path.
    assert profile.has_mtp() is False
    assert profile.checkpoint_to_live_name("mtp.0.attn.wq_a.weight") is None

    pairs = profile.fp8_scale_pairs(str(tmp_path))
    assert pairs is not None
    assert (
        pairs["mtp.0.attn.wq_a.weight"][1]
        == "mtp.0.attn.wq_a.scale"
    )
    assert (
        pairs["mtp.0.ffn.experts.0.w1.weight"][1]
        == "mtp.0.ffn.experts.0.w1.scale"
    )

    # The established body key remains the live qname; this fix must not
    # turn the whole map into checkpoint spelling.
    body = "model.layers.0.self_attn.wq_a.weight"
    assert pairs[body][1] == "layers.0.attn.wq_a.scale"
    assert "layers.0.attn.wq_a.weight" not in pairs


def test_lazy_skeleton_decodes_mtp_attention_and_expands_mxfp4_shape(
    tmp_path,
):
    _write_source(tmp_path)
    skeleton = _LazySkeleton(tmp_path)

    attn = "mtp.0.attn.wq_a.weight"
    expert = "mtp.0.ffn.experts.0.w1.weight"
    body_expert = "layers.0.ffn.experts.0.w1.weight"

    scale_map = _build_fp8_scale_inv_map(str(tmp_path))
    assert attn in scale_map
    assert expert in scale_map.mxfp4_names
    assert skeleton.logical_shape(attn) == (128, 128)
    assert skeleton.logical_shape(expert) == (4, 64)
    assert skeleton.logical_shape(body_expert) == (4, 64)

    # Exercise the same decode call the streaming CB exporter uses.  Before
    # the pairing fix, attention refused as unscaled FP8 and the packed expert
    # retained its physical half-width metadata.
    assert skeleton.dequant_weight(attn).shape == (128, 128)
    assert skeleton.dequant_weight(expert).shape == (4, 64)


def test_orphan_mtp_scale_is_rejected_at_pair_construction(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "quantization_config": {
                    "weight_block_size": [128, 128],
                },
            }
        )
    )
    save_file(
        {"mtp.0.attn.wq_a.scale": _e8m0_ones((1, 1))},
        str(tmp_path / "model.safetensors"),
    )

    with pytest.raises(RuntimeError, match="has no serialized weight sibling"):
        DeepseekV4Profile().fp8_scale_pairs(str(tmp_path))
