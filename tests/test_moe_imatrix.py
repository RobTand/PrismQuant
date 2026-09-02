"""Checkpoint-based packed-expert imatrix synthesis (moe_imatrix):
gate_up = module-input pool, down_proj = routed per-expert replay — the
entries the raw act-cache harvest can never contain, required by the CB
exporter (no silent RTN) and the local packed-expert cost (lockstep)."""
from __future__ import annotations

import json
import hashlib

import pytest
import torch
from safetensors.torch import load_file, save_file

from prismaquant.moe_imatrix import (
    RoutedActivationSamples,
    synthesize_packed_expert_col_weights,
)

HID, INTER, E = 16, 8, 2


class _IdentityProfile:
    def source_tensor_name(self, name: str) -> str:
        return name


@pytest.fixture()
def ckpt(tmp_path):
    torch.manual_seed(3)
    model_dir = tmp_path / "model"
    act_dir = tmp_path / "act"
    model_dir.mkdir()
    act_dir.mkdir()
    tensors = {
        "model.layers.0.mlp.gate.weight": torch.randn(E, HID),
    }
    for e in range(E):
        tensors[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = \
            torch.randn(INTER, HID)
        tensors[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = \
            torch.randn(INTER, HID)
    save_file(tensors, str(model_dir / "model.safetensors"))
    (model_dir / "config.json").write_text(json.dumps(
        {"num_experts_per_tok": 1, "norm_topk_prob": True}))
    torch.save({"inputs": torch.randn(64, HID),
                "name": "model.layers.0.mlp.experts"},
               act_dir / "model__layers__0__mlp__experts.pt")
    # A dense Linear act entry that must be ignored.
    torch.save({"inputs": torch.randn(64, HID),
                "name": "model.layers.0.self_attn.q_proj"},
               act_dir / "model__layers__0__self_attn__q_proj.pt")
    return model_dir, act_dir


def test_routed_down_samples_preserve_route_and_token_metadata(ckpt):
    model_dir, act_dir = ckpt
    entry = act_dir / "model__layers__0__mlp__experts.pt"
    blob = torch.load(entry, map_location="cpu", weights_only=False)
    blob["row_indices"] = torch.arange(1000, 1064)
    torch.save(blob, entry)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["num_experts_per_tok"] = 2
    config["model_type"] = "laguna"
    config["moe_router_logit_softcapping"] = 1.5
    config["moe_routed_scaling_factor"] = 2.5
    config_path.write_text(json.dumps(config))
    weights_with_bias = load_file(str(model_dir / "model.safetensors"))
    weights_with_bias[
        "model.layers.0.mlp.experts.e_score_correction_bias"
    ] = torch.tensor([0.35, -0.2])
    save_file(weights_with_bias, str(model_dir / "model.safetensors"))

    samples: dict = {}
    down_name = "model.layers.0.mlp.experts.down_proj"
    synthesize_packed_expert_col_weights(
        model_dir,
        act_dir,
        {},
        profile=_IdentityProfile(),
        device="cpu",
        max_rows=64,
        activation_samples=samples,
        target_names={down_name},
    )
    routed = samples[down_name]
    assert isinstance(routed, RoutedActivationSamples)
    routed.validate()
    assert routed.values.shape == (64, INTER)

    weights = load_file(str(model_dir / "model.safetensors"))
    x = blob["inputs"].float()
    logits = x @ weights["model.layers.0.mlp.gate.weight"].float().t()
    logits = torch.tanh(logits / 1.5) * 1.5
    scores = torch.sigmoid(logits)
    selection_scores = scores + weights[
        "model.layers.0.mlp.experts.e_score_correction_bias"
    ]
    _, topi = torch.topk(selection_scores, 2, dim=-1)
    topv = torch.gather(scores, -1, topi)
    topv = topv / topv.sum(dim=-1, keepdim=True) * 2.5
    seed = int.from_bytes(
        hashlib.sha256("model.layers.0.mlp.experts".encode()).digest()[:8],
        "little",
    ) & ((1 << 63) - 1)
    generator = torch.Generator().manual_seed(seed)
    sampled_flat = torch.randperm(128, generator=generator)[:64].sort().values
    sampled_rows = torch.div(sampled_flat, 2, rounding_mode="floor")
    sampled_slots = sampled_flat.remainder(2)
    expected_experts = topi[sampled_rows, sampled_slots]
    assert torch.equal(routed.cache_row_indices, sampled_rows)
    assert torch.equal(routed.source_row_indices, sampled_rows + 1000)
    assert torch.equal(routed.route_slots, sampled_slots)
    assert torch.equal(routed.expert_indices, expected_experts)
    assert torch.equal(routed.route_weights, topv[sampled_rows, sampled_slots])

    expected_values = []
    for row, expert in zip(sampled_rows, routed.expert_indices, strict=True):
        expert_id = int(expert.item())
        gate = x[row].double() @ weights[
            f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"
        ].double().t()
        up = x[row].double() @ weights[
            f"model.layers.0.mlp.experts.{expert_id}.up_proj.weight"
        ].double().t()
        expected_values.append(torch.nn.functional.silu(gate) * up)
    # Compare the production FP32 expert-batched replay against an independent
    # float64 route oracle. This validates the computed values without binding
    # the test to one CPU backend's FP32 GEMM/GEMV accumulation order.
    torch.testing.assert_close(
        routed.values.double(),
        torch.stack(expected_values),
        rtol=1e-5,
        atol=1e-6,
    )


def test_synthesizes_gateup_and_down(ckpt):
    model_dir, act_dir = ckpt
    cw: dict = {}
    added = synthesize_packed_expert_col_weights(
        model_dir, act_dir, cw, profile=_IdentityProfile(), device="cpu")
    assert set(added) == {"model.layers.0.mlp.experts.gate_up_proj",
                         "model.layers.0.mlp.experts.down_proj"}
    gu = cw["model.layers.0.mlp.experts.gate_up_proj"]
    dn = cw["model.layers.0.mlp.experts.down_proj"]
    assert gu.shape == (1, 1, HID) and bool((gu > 0).all())
    assert dn.shape == (E, 1, INTER) and bool((dn > 0).all())
    # No entry for the dense Linear (not a per-expert module).
    assert "model.layers.0.self_attn.q_proj.down_proj" not in cw


def test_physically_packed_gateup_matches_split_checkpoint(ckpt, tmp_path):
    """Qwen3.5/3.6 stores every expert in one gate_up rank-3 tensor."""
    model_dir, act_dir = ckpt
    split_cw: dict = {}
    synthesize_packed_expert_col_weights(
        model_dir,
        act_dir,
        split_cw,
        profile=_IdentityProfile(),
        device="cpu",
    )

    split = load_file(str(model_dir / "model.safetensors"))
    packed_dir = tmp_path / "packed-model"
    packed_dir.mkdir()
    gate_up = torch.stack([
        torch.cat([
            split[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"],
            split[f"model.layers.0.mlp.experts.{e}.up_proj.weight"],
        ])
        for e in range(E)
    ])
    save_file({
        "model.layers.0.mlp.gate.weight": split[
            "model.layers.0.mlp.gate.weight"
        ],
        # Qwen's checkpoint parameter name deliberately has no `.weight`.
        "model.layers.0.mlp.experts.gate_up_proj": gate_up,
    }, str(packed_dir / "model.safetensors"))
    (packed_dir / "config.json").write_text(
        (model_dir / "config.json").read_text()
    )

    packed_cw: dict = {}
    added = synthesize_packed_expert_col_weights(
        packed_dir,
        act_dir,
        packed_cw,
        profile=_IdentityProfile(),
        device="cpu",
    )
    assert set(added) == {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    }
    torch.testing.assert_close(
        packed_cw["model.layers.0.mlp.experts.gate_up_proj"],
        split_cw["model.layers.0.mlp.experts.gate_up_proj"],
    )
    torch.testing.assert_close(
        packed_cw["model.layers.0.mlp.experts.down_proj"],
        split_cw["model.layers.0.mlp.experts.down_proj"],
    )


def test_respects_existing_entries(ckpt):
    model_dir, act_dir = ckpt
    pre = torch.ones(E, 1, INTER)
    cw = {"model.layers.0.mlp.experts.gate_up_proj": torch.ones(1, 1, HID),
          "model.layers.0.mlp.experts.down_proj": pre}
    added = synthesize_packed_expert_col_weights(
        model_dir, act_dir, cw, profile=_IdentityProfile(), device="cpu")
    assert added == []
    assert torch.equal(cw["model.layers.0.mlp.experts.down_proj"], pre)


def test_missing_router_is_loud(ckpt, tmp_path):
    model_dir, act_dir = ckpt
    # Rebuild the checkpoint without the router weight.
    tensors = {}
    for e in range(E):
        tensors[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = \
            torch.randn(INTER, HID)
        tensors[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = \
            torch.randn(INTER, HID)
    m2 = tmp_path / "model2"
    m2.mkdir()
    save_file(tensors, str(m2 / "model.safetensors"))
    (m2 / "config.json").write_text(json.dumps({"num_experts_per_tok": 1}))
    with pytest.raises(ValueError, match="router weight"):
        synthesize_packed_expert_col_weights(
            m2, act_dir, {}, profile=_IdentityProfile(), device="cpu")


# Two tests were deleted here on 2026-09-02, with the Gridbook codebook lane
# (archive/gridbook_lane_2026-09-02/):
#
# `test_export_skeleton_expert_packing` pinned
# `export_nvfp4_cb._pack_skeleton_experts`, the CB exporter's per-expert ->
# packed skeleton bridge. That exporter module is archived. The equivalent
# bridge for the surviving lanes lives in `layer_streaming.py` and is covered
# by its own tests.
#
# `test_nvfp4_cb_profile_denies_stock_formats_on_packed_experts` pinned the
# `nvfp4_cb` serving profile's dense-only stock-CT delegation: stock NVFP4/FP8
# denied on rank-3 packed expert stacks, CB rungs allowed on both. That
# serving profile is archived and it was the only spec in the repo that ever
# admitted a CB rung, so the assertion has no profile to make it against.
# Re-pointing it at `vllm_packed_moe` -- which denies CB rungs outright --
# would have turned a capability loss into a green assertion.


def test_load_tensors_dequantizes_fp8_with_serialized_scale_contract(tmp_path):
    """An FP8-source MoE checkpoint carries block-wise `weight_scale_inv`
    companions; `_load_tensors` must fulfill that contract (exact block
    dequant) rather than refusing every float8 tensor -- and the refusal
    must stand for a tensor whose companion is genuinely absent.

    First FP8-source MoE through the packed-expert replay path
    (Qwen3.6-35B-A3B-FP8, fmt e4m3, 128x128 block scales); verified
    block-exact against a manual dequant before the fix shipped a build."""
    import json

    import pytest
    import torch
    from safetensors.torch import save_file

    from prismaquant.moe_imatrix import _WEIGHT_BLOCK_CACHE, _load_tensors

    torch.manual_seed(0)
    w = torch.randn(256, 384)
    scale = torch.rand(2, 3) + 0.5              # 128x128 blocks
    q = (w / scale.repeat_interleave(128, 0)[:256]
         .repeat_interleave(128, 1)[:384]).to(torch.float8_e4m3fn)
    good = "model.layers.0.mlp.experts.0.gate_proj.weight"
    bare = "model.layers.0.mlp.experts.1.gate_proj.weight"
    shard = "model.safetensors"
    save_file({good: q, good + "_scale_inv": scale,
               bare: q.clone()}, str(tmp_path / shard))
    wm = {good: shard, good + "_scale_inv": shard, bare: shard}
    (tmp_path / "config.json").write_text(json.dumps(
        {"quantization_config": {"quant_method": "fp8",
                                 "weight_block_size": [128, 128]}}))
    _WEIGHT_BLOCK_CACHE.clear()

    out = _load_tensors(tmp_path, wm, [good], dtype=torch.float32)
    expect = (q.to(torch.float32)
              * scale.repeat_interleave(128, 0)[:256]
              .repeat_interleave(128, 1)[:384])
    assert torch.equal(out[good], expect)

    with pytest.raises(ValueError, match="serialized scale contract"):
        _load_tensors(tmp_path, wm, [bare])


def test_load_tensors_partial_trailing_block_with_declared_size(tmp_path):
    """With the checkpoint's declared weight_block_size, a partial trailing
    block dequantizes exactly (128-blocks over 200 rows: rows 128-199 get
    scale row 1, not a uniformly mis-inferred 100-row tiling).

    Undeclared, the replay REFUSES -- it does not fall back to inferring the
    grid by division. 200 rows over a 2-row scale plane divides exactly at
    100 and is equally a 128-block tiling with a partial trailing block, and
    the two dequants differ on every row from 128 up, so an inferred grid is
    a silent calibration corruption. The refusal is the streaming loader's
    `_declared_weight_block_size` -- one contract for one checkpoint, shared
    with the layer-streaming load path."""
    import json

    import pytest
    import torch
    from safetensors.torch import save_file

    from prismaquant.moe_imatrix import _WEIGHT_BLOCK_CACHE, _load_tensors

    torch.manual_seed(1)
    q = torch.randn(200, 384).to(torch.float8_e4m3fn)
    scale = torch.rand(2, 3) + 0.5
    k = "model.layers.0.mlp.experts.0.up_proj.weight"
    save_file({k: q, k + "_scale_inv": scale}, str(tmp_path / "m.safetensors"))
    wm = {k: "m.safetensors", k + "_scale_inv": "m.safetensors"}

    (tmp_path / "config.json").write_text(json.dumps(
        {"quantization_config": {"quant_method": "fp8",
                                 "weight_block_size": [128, 128]}}))
    _WEIGHT_BLOCK_CACHE.clear()
    out = _load_tensors(tmp_path, wm, [k], dtype=torch.float32)
    expect = (q.to(torch.float32)
              * scale.repeat_interleave(128, 0)[:200]
              .repeat_interleave(128, 1)[:384])
    assert torch.equal(out[k], expect)

    # Same tensors, no declaration: exactly the ambiguous case above.
    (tmp_path / "config.json").write_text(json.dumps(
        {"quantization_config": {"quant_method": "fp8"}}))
    _WEIGHT_BLOCK_CACHE.clear()
    with pytest.raises(RuntimeError, match="weight_block_size"):
        _load_tensors(tmp_path, wm, [k])
    _WEIGHT_BLOCK_CACHE.clear()


def test_load_tensors_refuses_scale_plane_that_does_not_tile(tmp_path):
    """A declared grid that does not tile the weight is a transposed or
    mismatched scale plane; it must refuse rather than broadcast."""
    import json

    import pytest
    import torch
    from safetensors.torch import save_file

    from prismaquant.moe_imatrix import _WEIGHT_BLOCK_CACHE, _load_tensors

    q = torch.randn(256, 384).to(torch.float8_e4m3fn)
    scale = torch.rand(3, 2) + 0.5               # transposed grid
    k = "model.layers.0.mlp.experts.0.down_proj.weight"
    save_file({k: q, k + "_scale_inv": scale}, str(tmp_path / "m.safetensors"))
    wm = {k: "m.safetensors", k + "_scale_inv": "m.safetensors"}
    (tmp_path / "config.json").write_text(json.dumps(
        {"quantization_config": {"quant_method": "fp8",
                                 "weight_block_size": [128, 128]}}))
    _WEIGHT_BLOCK_CACHE.clear()
    with pytest.raises(ValueError, match="does not tile"):
        _load_tensors(tmp_path, wm, [k])
    _WEIGHT_BLOCK_CACHE.clear()
