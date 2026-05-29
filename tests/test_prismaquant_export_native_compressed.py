"""Tests for the native compressed-tensors exporter.

Covers the math (NVFP4 / FP8 round-trip) and the wire-format
plumbing (`_to_vllm_internal_name`, `build_quantization_config`)
that has to stay in sync with vLLM's compressed-tensors loader.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import prismaquant.export_native_compressed as enc

from prismaquant.allocator import promote_fused
from prismaquant.export_native_compressed import (
    DEFAULT_INPUT_GLOBAL_SCALE,
    FLOAT_TO_E2M1,
    FP8_E4M3_MAX,
    NVFP4_MAX,
    PER_EXPERT_MOE_REGEX,
    _bf16_upgrade_audit,
    _compute_layer_joint_nvfp4,
    _coerce_runtime_legal_assignment,
    _passthrough_dtype,
    _passthrough_tensor,
    _quantize_2d,
    _quantize_3d_packed,
    _resolve_perturbed_x_export_inputs,
    _round_to_codebook,
    _to_vllm_internal_name,
    compute_extra_ignore,
    validate_mtp_assignment_coverage,
    build_quantization_config,
    canonicalize_format,
    compute_nvfp4_global_real,
    pack_fp4_indices,
    quantize_dequantize_fp8_dynamic,
    quantize_dequantize_fp8_dynamic_packed,
    quantize_dequantize_mxfp4,
    quantize_dequantize_mxfp4_packed,
    quantize_dequantize_mxfp8,
    quantize_dequantize_mxfp8_packed,
    quantize_dequantize_nvfp4,
    quantize_dequantize_nvfp4_packed,
)
from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
from prismaquant.model_profiles.gemma4 import Gemma4Profile
from prismaquant import format_registry as fr
from prismaquant.layer_streaming import _dequant_fp8_block_weight


def _fpquant_gptq_factors_for_test(
    W: torch.Tensor,
    X: torch.Tensor,
    *,
    damp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    W_work = W.to(torch.float32).clone()
    H = X.to(torch.float32).t() @ X.to(torch.float32)
    diag_mean = torch.diagonal(H).mean().clamp_min(1e-12)
    H.diagonal().add_(float(damp) * diag_mean)
    dead = torch.diagonal(H) <= 0
    if dead.any():
        H[dead, dead] = 1.0
        W_work[:, dead] = 0.0
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    return W_work, torch.linalg.cholesky(Hinv, upper=True)


def _fpquant_columnwise_reference_for_test(
    W: torch.Tensor,
    U: torch.Tensor,
    *,
    block_size: int,
    quantize_column,
) -> torch.Tensor:
    W_ref = W.to(torch.float32).clone()
    cols = W_ref.shape[1]
    for c1 in range(0, cols, block_size):
        c2 = min(c1 + block_size, cols)
        ncols = c2 - c1
        w_blk = W_ref[:, c1:c2].clone()
        errs = torch.zeros_like(w_blk)
        U_blk = U[c1:c2, c1:c2]
        for i in range(ncols):
            w_ci = w_blk[:, i]
            w_q = quantize_column(w_ci, c1 + i).to(torch.float32)
            W_ref[:, c1 + i] = w_q
            err = (w_ci - w_q) / U_blk[i, i]
            w_blk[:, i:].addr_(err, U_blk[i, i:], alpha=-1)
            errs[:, i] = err
        if c2 < cols:
            W_ref[:, c2:].addmm_(errs, U[c1:c2, c2:], alpha=-1)
    return W_ref


class _IdentityProfile:
    """Minimal profile stub for tests that only need `live_to_recipe_name`
    to be identity. Avoids pulling in the full ModelProfile ABC and its
    abstract methods."""

    def live_to_recipe_name(self, live_qname: str) -> str:
        return live_qname


class _CustomPackedProfile:
    """Profile stub with a non-Qwen packed expert decomposition."""

    def live_to_recipe_name(self, live_qname: str) -> str:
        return live_qname

    def to_vllm_internal_name(self, checkpoint_name: str) -> str:
        return checkpoint_name

    def per_expert_moe_regex(self) -> str | None:
        return None

    def per_expert_mtp_regex(self) -> str | None:
        return None

    def packed_expert_param_names(self) -> frozenset[str]:
        return frozenset({"w13", "w2"})

    def packed_expert_projection_names(self, param_name: str) -> tuple[str, ...]:
        if param_name == "w13":
            return ("w1_proj", "w3_proj")
        return (param_name,)

    def packed_expert_parent_for_projection(
        self,
        projection_name: str,
    ) -> str | None:
        if projection_name in {"w1_proj", "w3_proj"}:
            return "w13"
        if projection_name == "w2":
            return "w2"
        return None

    def packed_expert_format_group(self, qname: str) -> str | None:
        if ".experts." not in qname:
            return None
        parent, leaf = qname.rsplit(".", 1)
        if leaf in {"w13", "w2"}:
            return f"{parent}::__packed_format__:w13,w2"
        return None


class _CustomPackedRegexProfile(_CustomPackedProfile):
    """Custom packed profile with a serving-side per-expert regex."""

    def to_vllm_internal_name(self, checkpoint_name: str) -> str:
        return checkpoint_name.replace(
            "model.layers.",
            "serving.layers.",
        ).replace(
            ".mlp.experts.",
            ".moe.experts.",
        )

    def per_expert_moe_regex(self) -> str | None:
        return (
            "re:^serving[.]layers[.][0-9]+[.]moe[.]experts"
            "[.][0-9]+[.](w1_proj|w3_proj|w2)$"
        )

    def per_expert_mtp_regex(self) -> str | None:
        return (
            "re:^mtp[.]layers[.][0-9]+[.]moe[.]experts"
            "[.][0-9]+[.](w1_proj|w3_proj|w2)$"
        )


class TestPassthroughDtype(unittest.TestCase):
    def test_passthrough_preserves_source_precision_policy(self):
        self.assertEqual(
            _passthrough_dtype(
                "model.layers.0.input_layernorm.weight",
                torch.bfloat16,
            ),
            torch.bfloat16,
        )
        self.assertEqual(
            _passthrough_dtype(
                "mtp.layers.0.self_attn.q_norm.weight",
                torch.float16,
            ),
            torch.float16,
        )
        self.assertEqual(
            _passthrough_dtype(
                "model.layers.0.self_attn.q_proj.weight",
                torch.float8_e4m3fn,
            ),
            torch.float8_e4m3fn,
        )

    def test_passthrough_uses_current_dtype_only_as_fallback(self):
        value, label = _passthrough_tensor(
            "model.norm.weight",
            torch.ones(4, dtype=torch.float32),
        )
        self.assertEqual(value.dtype, torch.float32)
        self.assertEqual(label, "FP32")

    def test_passthrough_rejects_missing_dtype_without_fallback(self):
        with self.assertRaises(ValueError):
            _passthrough_dtype("model.norm.weight")


class TestLazyActivationCache(unittest.TestCase):
    def test_get_loads_existing_tensor_on_demand(self):
        from prismaquant.export_native_compressed import _LazyActivationCache

        class FakeIndex:
            def __init__(self):
                self.load_count = 0
                self.values = {"layer.q_proj": torch.ones(2, 3, dtype=torch.bfloat16)}

            def __contains__(self, name):
                return name in self.values

            def load(self, name):
                self.load_count += 1
                return self.values[name]

        index = FakeIndex()
        cache = _LazyActivationCache(index)

        self.assertEqual(index.load_count, 0)
        self.assertIsNone(cache.get("missing"))
        self.assertEqual(index.load_count, 0)

        value = cache.get("layer.q_proj")
        self.assertEqual(index.load_count, 1)
        self.assertEqual(cache.loads, 1)
        self.assertEqual(value.dtype, torch.float32)
        self.assertTrue(torch.equal(value, torch.ones(2, 3, dtype=torch.float32)))


class TestPerturbedXExportInputs(unittest.TestCase):
    def test_resolves_summary_layer_config_and_final_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            layer_config = root / "final_layer_config.json"
            layer_config.write_text("{}")
            cache = root / "activation_cache_iter_02"
            cache.mkdir()
            with open(root / "summary.json", "w") as f:
                json.dump(
                    {
                        "final_layer_config": str(layer_config),
                        "iterations": [
                            {"cache": {"cache_dir": str(root / "activation_cache_iter_01")}},
                            {"cache": {"cache_dir": str(cache)}},
                        ],
                    },
                    f,
                )

            got_layer_config, got_cache = _resolve_perturbed_x_export_inputs(root)

            self.assertEqual(got_layer_config, layer_config)
            self.assertEqual(got_cache, cache)

    def test_resolves_latest_cache_without_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            layer_config = root / "final_layer_config.json"
            layer_config.write_text("{}")
            (root / "activation_cache_iter_01").mkdir()
            latest = root / "activation_cache_iter_03"
            latest.mkdir()

            got_layer_config, got_cache = _resolve_perturbed_x_export_inputs(root)

            self.assertEqual(got_layer_config, layer_config)
            self.assertEqual(got_cache, latest)


class TestIncrementalSafetensorsWriter(unittest.TestCase):
    def test_finalizes_multi_shard_index_without_temp_files(self):
        from safetensors.torch import load_file

        from prismaquant.export_native_compressed import (
            IncrementalSafetensorsWriter,
        )

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            writer = IncrementalSafetensorsWriter(out_dir, shard_bytes=32)
            writer.add_tensors({
                "b.weight": torch.ones(4, dtype=torch.float32),
                "a.weight": torch.arange(8, dtype=torch.float32),
            })
            writer.add_tensors({
                "c.weight": torch.arange(16, dtype=torch.int8),
            })
            writer.finalize()

            self.assertFalse(list(out_dir.glob("*.tmp")))
            idx_path = out_dir / "model.safetensors.index.json"
            self.assertTrue(idx_path.exists())
            with open(idx_path) as f:
                index = json.load(f)
            self.assertEqual(
                set(index["weight_map"]),
                {"a.weight", "b.weight", "c.weight"},
            )
            self.assertEqual(index["metadata"]["total_size"], 64)

            loaded = {}
            for shard_name in set(index["weight_map"].values()):
                loaded.update(load_file(str(out_dir / shard_name)))
            self.assertTrue(torch.equal(
                loaded["a.weight"], torch.arange(8, dtype=torch.float32)
            ))
            self.assertTrue(torch.equal(
                loaded["b.weight"], torch.ones(4, dtype=torch.float32)
            ))
            self.assertTrue(torch.equal(
                loaded["c.weight"], torch.arange(16, dtype=torch.int8)
            ))


class TestGroupedExportQuantization(unittest.TestCase):
    def test_grouped_rtn_formats_match_scalar_export(self):
        from prismaquant.export_native_compressed import (
            _quantize_2d,
            _quantize_2d_group_same_shape,
        )

        torch.manual_seed(0)
        weights = torch.randn(3, 4, 32)
        for fmt in ("MXFP8",):
            grouped = _quantize_2d_group_same_shape(weights, fmt)
            for i in range(weights.shape[0]):
                scalar = _quantize_2d(weights[i], fmt)
                for key, scalar_tensor in scalar.items():
                    grouped_tensor = grouped[key][i]
                    if key == "weight_global_scale":
                        grouped_tensor = grouped_tensor.reshape(1)
                    if scalar_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                        self.assertTrue(
                            torch.allclose(
                                grouped_tensor.to(torch.float32),
                                scalar_tensor.to(torch.float32),
                            ),
                            msg=f"{fmt} {key}[{i}]",
                        )
                    elif scalar_tensor.dtype.is_floating_point:
                        self.assertTrue(
                            torch.allclose(grouped_tensor, scalar_tensor),
                            msg=f"{fmt} {key}[{i}]",
                        )
                    else:
                        self.assertTrue(
                            torch.equal(grouped_tensor, scalar_tensor),
                            msg=f"{fmt} {key}[{i}]",
                        )

    def test_low_bit_custom_kernel_formats_are_rejected(self):
        from prismaquant.export_native_compressed import (
            _quantize_2d,
            _quantize_2d_group_same_shape,
            canonicalize_format,
        )

        weights = torch.randn(2, 4, 16)
        with self.assertRaises(ValueError):
            canonicalize_format("nvint2")
        with self.assertRaises(ValueError):
            canonicalize_format({"data_type": "int", "bits": 3})
        with self.assertRaises(ValueError):
            _quantize_2d(weights[0], "NVINT2")
        with self.assertRaises(ValueError):
            _quantize_2d_group_same_shape(weights, "INT3")


class _TinyQwenPackedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(2, 128, 32))
        self.down_proj = nn.Parameter(torch.randn(2, 32, 64))


class TestPackedExpertExport(unittest.TestCase):
    def test_mxfp8_split_experts_emit_weight_suffix_for_vllm_loader(self):
        """Qwen3.5's split expert loader matches
        `experts.<id>.<proj>.weight`; without the `.weight` suffix raw
        MXFP8 expert weights fall through to the generic loader and are
        skipped with "not found in params_dict" warnings.
        """

        root = nn.Module()
        root.model = nn.Module()
        root.model.language_model = nn.Module()
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.experts = _TinyQwenPackedExperts()
        root.model.language_model.layers = nn.ModuleList([layer])

        assignment = {
            "model.layers.0.mlp.experts.gate_up_proj": "MXFP8_E4M3",
            "model.layers.0.mlp.experts.down_proj": "MXFP8_E4M3",
        }

        tensors, hist = enc._materialize_tensors_inmemory(
            root,
            assignment,
            bf16_passthrough=set(),
            profile=Qwen3_5Profile(),
        )

        prefix = "model.language_model.layers.0.mlp.experts"
        expected = {
            f"{prefix}.0.gate_proj.weight",
            f"{prefix}.0.gate_proj.weight_scale",
            f"{prefix}.0.up_proj.weight",
            f"{prefix}.0.up_proj.weight_scale",
            f"{prefix}.0.down_proj.weight",
            f"{prefix}.0.down_proj.weight_scale",
        }
        self.assertTrue(expected.issubset(tensors.keys()))
        self.assertNotIn(f"{prefix}.0.gate_proj", tensors)
        self.assertNotIn(f"{prefix}.0.up_proj", tensors)
        self.assertNotIn(f"{prefix}.0.down_proj", tensors)
        self.assertEqual(
            hist.get(("packed_moe_per_expert", "MXFP8_E4M3")),
            2,
        )


def _nvfp4_dequantize(weight_packed, weight_scale_fp8, weight_global_scale_divisor):
    """Reproduce vLLM's NVFP4 dequant convention to verify round-trip.
    The on-disk `weight_global_scale` is `1/global_real`; vLLM inverts
    on load. Per-element dequant: `codebook[idx] * fp8_scale * global_real`.
    """
    rows = weight_packed.shape[0]
    cols = weight_packed.shape[1] * 2
    cb = torch.tensor(FLOAT_TO_E2M1, dtype=torch.float32)
    lo = (weight_packed & 0xF).long()
    hi = ((weight_packed >> 4) & 0xF).long()
    idx = torch.stack([lo, hi], dim=-1).reshape(rows, cols)
    abs_idx = idx & 0x7
    sign = -((idx >> 3).to(torch.float32) * 2 - 1)
    vals = sign * cb[abs_idx]
    fp8_per_col = (
        weight_scale_fp8.float()
        .unsqueeze(-1)
        .expand(-1, -1, cols // weight_scale_fp8.shape[1])
        .reshape(rows, cols)
    )
    global_real = 1.0 / weight_global_scale_divisor.item()
    return vals * fp8_per_col * global_real


def _mxfp4_served_dequantize(weight_packed, weight_scale_e8m0, group_size=32):
    """Reconstruct MXFP4 as the compressed-tensors/vLLM loader serves it."""
    rows = weight_packed.shape[0]
    cols = weight_packed.shape[1] * 2
    cb = torch.tensor(FLOAT_TO_E2M1, dtype=torch.float32)
    lo = (weight_packed & 0xF).long()
    hi = ((weight_packed >> 4) & 0xF).long()
    idx = torch.stack([lo, hi], dim=-1).reshape(rows, cols)
    abs_idx = idx & 0x7
    sign = -((idx >> 3).to(torch.float32) * 2 - 1)
    vals = sign * cb[abs_idx]
    scale = torch.pow(2.0, weight_scale_e8m0.to(torch.float32) - 127.0)
    return vals * scale.repeat_interleave(group_size, dim=1)


def _mxfp8_served_dequantize(weight_fp8, weight_scale_e8m0, group_size=32):
    scale = torch.pow(2.0, weight_scale_e8m0.to(torch.float32) - 127.0)
    return weight_fp8.float() * scale.repeat_interleave(group_size, dim=1)


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_nvfp4_2d_roundtrip_mse_small(self):
        W = torch.randn(64, 128) * 0.1
        wp, ws, wg = quantize_dequantize_nvfp4(W)
        self.assertEqual(wp.dtype, torch.uint8)
        self.assertEqual(ws.dtype, torch.float8_e4m3fn)
        self.assertEqual(wg.dtype, torch.float32)
        self.assertEqual(tuple(wp.shape), (64, 64))
        self.assertEqual(tuple(ws.shape), (64, 8))
        self.assertEqual(tuple(wg.shape), (1,))
        # fp8 scale must use the FP8 representable range, not be
        # squashed into [0, 1] (the latter loses precision).
        self.assertGreater(ws.float().max().item(), 32.0,
                           "fp8 scale appears to be normalized to [0,1]; "
                           "vLLM's NVFP4 path expects the full FP8 range")

        dequant = _nvfp4_dequantize(wp, ws, wg)
        mse = (W - dequant).pow(2).mean().item()
        self.assertLess(mse, 1e-3,
                        f"NVFP4 round-trip MSE {mse:.3e} too large")
        # max-abs preserved (NVFP4 has explicit ±6 codes covering the peak)
        self.assertAlmostEqual(
            dequant.abs().max().item(),
            W.abs().max().item(),
            places=3,
        )

    def test_nvfp4_rtn_dequant_matches_exported_scale_metadata(self):
        from prismaquant import export_native_compressed as enc

        W = torch.randn(16, 64) * 2.0
        prev = enc._NVFP4_SCALE_RULE
        try:
            enc._NVFP4_SCALE_RULE = enc.NVFP4_SCALE_RULE_STATIC_6
            wp, ws, wg = quantize_dequantize_nvfp4(W)
            packed_dequant = _nvfp4_dequantize(wp, ws, wg)
            helper_dequant = enc._rtn_dequant_nvfp4(W, group_size=16)
        finally:
            enc._NVFP4_SCALE_RULE = prev

        torch.testing.assert_close(helper_dequant, packed_dequant)

    def test_registry_served_metadata_reconciliation_covers_all_registered_formats(self):
        reconciled = {
            "NVFP4",
            "NVFP4A16",
            "MXFP4",
            "MXFP8_E4M3",
            "MXFP8_E5M2",
            "MXFP8A16",
            "FP8_E4M3",
            "FP8_E5M2",
            "BF16",
            "FP8_SOURCE",
        }
        explicit_gaps = {
            "MXFP6_E3M2": "no vLLM/compressed-tensors served export path is wired yet",
            "MXFP6_E2M3": "no vLLM/compressed-tensors served export path is wired yet",
            "INT8_W8A16": "registered allocator research format; no native exporter metadata path",
            "INT4_W4A16_g128": "registered allocator research format; no native exporter metadata path",
        }
        self.assertEqual(set(fr.REGISTRY), reconciled | set(explicit_gaps))

    def test_registry_render_dequant_matches_served_metadata(self):
        W = torch.randn(8, 128) * 1.75
        prev = enc._NVFP4_SCALE_RULE
        try:
            enc._NVFP4_SCALE_RULE = enc.NVFP4_SCALE_RULE_STATIC_6
            for fmt in sorted(fr.REGISTRY):
                with self.subTest(fmt=fmt):
                    if fmt in {"MXFP6_E3M2", "MXFP6_E2M3", "INT8_W8A16", "INT4_W4A16_g128"}:
                        continue

                    if fmt in {"NVFP4", "NVFP4A16"}:
                        wp, ws, wg = quantize_dequantize_nvfp4(W)
                        served = _nvfp4_dequantize(wp, ws, wg)
                        rendered = enc._rtn_dequant_nvfp4(W, group_size=16)
                        torch.testing.assert_close(rendered, served)
                        continue

                    if fmt == "MXFP4":
                        out = _quantize_2d(W, "MXFP4")
                        served = _mxfp4_served_dequantize(
                            out["weight_packed"],
                            out["weight_scale"],
                        )
                        rendered = enc._mxfp4_grouped_codec(
                            W.reshape(W.shape[0], W.shape[1] // 32, 32)
                        ).dequant.reshape_as(W)
                        torch.testing.assert_close(rendered, served)
                        continue

                    if fmt in {"MXFP8_E4M3", "MXFP8_E5M2", "MXFP8A16"}:
                        codec_fmt = "MXFP8_E4M3" if fmt == "MXFP8A16" else fmt
                        dtype, max_value = enc._fp8_element_dtype_and_max(codec_fmt)
                        q, s = quantize_dequantize_mxfp8(
                            W,
                            element_dtype=dtype,
                            element_max=max_value,
                        )
                        served = _mxfp8_served_dequantize(q, s)
                        rendered = enc._rtn_dequant_mxfp8(
                            W,
                            element_dtype=dtype,
                            element_max=max_value,
                        )
                        torch.testing.assert_close(rendered, served)
                        continue

                    if fmt in {"FP8_E4M3", "FP8_E5M2"}:
                        dtype, max_value = enc._fp8_element_dtype_and_max(fmt)
                        q, s = quantize_dequantize_fp8_dynamic(
                            W,
                            element_dtype=dtype,
                            element_max=max_value,
                        )
                        served = q.float() * s
                        rendered = enc._rtn_dequant_fp8_dynamic(
                            W,
                            element_dtype=dtype,
                            element_max=max_value,
                        )
                        torch.testing.assert_close(rendered, served)
                        continue

                    if fmt == "BF16":
                        out = _quantize_2d(W, "BF16")
                        torch.testing.assert_close(
                            out["weight"].float(),
                            W.bfloat16().float(),
                        )
                        continue

                    if fmt == "FP8_SOURCE":
                        source_scale = torch.full((1, 1), 0.125, dtype=torch.float32)
                        source_codes = (W[:128, :128] / source_scale).to(
                            torch.float8_e4m3fn
                        )
                        live_bf16 = _dequant_fp8_block_weight(source_codes, source_scale)
                        served = source_codes.float() * source_scale
                        torch.testing.assert_close(
                            live_bf16.float(),
                            served.bfloat16().float(),
                        )
                        continue

                    self.fail(f"unhandled registered format {fmt}")
        finally:
            enc._NVFP4_SCALE_RULE = prev

    def test_nvfp4_four_over_six_picks_lower_mse_block_scale(self):
        W = torch.tensor(
            [[10.0, 20.0, 30.0, 40.0] * 4],
            dtype=torch.float32,
        )
        prev = enc._NVFP4_SCALE_RULE
        try:
            enc._NVFP4_SCALE_RULE = "static_6"
            wp6, ws6, wg6 = quantize_dequantize_nvfp4(W)
            dq6 = _nvfp4_dequantize(wp6, ws6, wg6)
            mse6 = (W - dq6).pow(2).mean().item()

            enc._NVFP4_SCALE_RULE = "four_over_six_mse"
            wp4, ws4, wg4 = quantize_dequantize_nvfp4(W)
            dq4 = _nvfp4_dequantize(wp4, ws4, wg4)
            mse4 = (W - dq4).pow(2).mean().item()
        finally:
            enc._NVFP4_SCALE_RULE = prev

        self.assertLess(mse4, mse6)
        self.assertLess(mse4, 1e-5)
        self.assertAlmostEqual(
            ws4.float()[0, 0].item() / wg4.item(),
            10.0,
            places=4,
        )

    def test_nvfp4_four_over_six_global_real_matches_chosen_scales(self):
        W = torch.tensor(
            [
                [10.0, 20.0, 30.0, 40.0] * 4,
                [1.0, 2.0, 3.0, 6.0] * 4,
            ],
            dtype=torch.float32,
        )
        prev = enc._NVFP4_SCALE_RULE
        try:
            enc._NVFP4_SCALE_RULE = "static_6"
            g6 = compute_nvfp4_global_real(W).item()
            enc._NVFP4_SCALE_RULE = "four_over_six_mse"
            g4 = compute_nvfp4_global_real(W).item()
        finally:
            enc._NVFP4_SCALE_RULE = prev

        self.assertGreater(g4, g6)
        self.assertAlmostEqual(g4 / g6, 1.5, places=4)

    def test_nvfp4_packed_per_expert_global_scale(self):
        # Each expert's global_scale is independent.
        E, M, N = 4, 32, 64
        P = torch.randn(E, M, N) * 0.05
        wp, ws, wg = quantize_dequantize_nvfp4_packed(P)
        self.assertEqual(tuple(wp.shape), (E, M, N // 2))
        self.assertEqual(tuple(ws.shape), (E, M, N // 16))
        self.assertEqual(tuple(wg.shape), (E,))
        # Distinct experts → distinct per-tensor scales.
        self.assertGreater(wg.unique().numel(), 1)

    def test_fp8_dynamic_2d_per_channel_scale(self):
        W = torch.randn(64, 128) * 0.1
        w, s = quantize_dequantize_fp8_dynamic(W)
        self.assertEqual(w.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(s.shape), (64, 1))
        self.assertEqual(s.dtype, torch.float32)
        self.assertFalse(torch.isnan(w.float()).any().item(),
                         "fp8 cast NaN — likely overflow in scale")
        # Round-trip MSE
        dequant = w.float() * s
        mse = (W - dequant).pow(2).mean().item()
        self.assertLess(mse, 1e-4)

    def test_fp8_dynamic_packed_3d(self):
        E, M, N = 4, 32, 64
        P = torch.randn(E, M, N) * 0.1
        w, s = quantize_dequantize_fp8_dynamic_packed(P)
        self.assertEqual(tuple(w.shape), (E, M, N))
        self.assertEqual(tuple(s.shape), (E, M, 1))

    def test_mxfp8_2d_grouped_scale(self):
        W = torch.randn(32, 64) * 0.1
        w, s = quantize_dequantize_mxfp8(W)
        self.assertEqual(w.dtype, torch.float8_e4m3fn)
        self.assertEqual(s.dtype, torch.uint8)
        self.assertEqual(tuple(s.shape), (32, 2))
        scales = torch.pow(2.0, s.to(torch.float32) - 127.0)
        dequant = w.float() * scales.repeat_interleave(32, dim=1)
        mse = (W - dequant).pow(2).mean().item()
        self.assertLess(mse, 2e-4)

    def test_mx_e8m0_scale_encoding_matches_compressed_tensors_reference(self):
        try:
            from compressed_tensors.quantization.utils.mxfp_utils import (
                generate_mx_scales,
            )
        except ModuleNotFoundError:
            self.skipTest("compressed_tensors is not installed")

        W = torch.randn(8, 64) * 13.0
        grouped = W.reshape(8, 2, 32)

        mxfp8_scale = enc._mxfp8_grouped_codec(grouped).scale
        mxfp8_ref = generate_mx_scales(
            grouped.abs().amax(dim=-1),
            num_bits=8,
        ).to(torch.uint8)
        torch.testing.assert_close(mxfp8_scale, mxfp8_ref, atol=0, rtol=0)

        mxfp4_scale = enc._mxfp4_grouped_codec(grouped).scale
        mxfp4_ref = generate_mx_scales(
            grouped.abs().amax(dim=-1),
            num_bits=4,
        ).to(torch.uint8)
        torch.testing.assert_close(mxfp4_scale, mxfp4_ref, atol=0, rtol=0)

    def test_mxfp8_packed_3d(self):
        E, M, N = 4, 32, 64
        P = torch.randn(E, M, N) * 0.1
        w, s = quantize_dequantize_mxfp8_packed(P)
        self.assertEqual(tuple(w.shape), (E, M, N))
        self.assertEqual(tuple(s.shape), (E, M, 2))
        self.assertEqual(s.dtype, torch.uint8)

    def test_fp8_e4m3_gptq_returns_exportable_tensors(self):
        torch.manual_seed(0)
        W = torch.randn(4, 32) * 0.1
        X = torch.randn(12, 32)
        q, s, dq = enc._gptq_obs_rounding_fp8_like(
            W,
            X,
            fmt="FP8_E4M3",
            damp=0.01,
        )
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(q.shape), tuple(W.shape))
        self.assertEqual(tuple(s.shape), (4, 1))
        self.assertEqual(tuple(dq.shape), tuple(W.shape))
        self.assertTrue(torch.allclose(dq, q.float() * s))

    def test_mxfp8_e5m2_gptq_ignores_joint_scale_opt(self):
        torch.manual_seed(0)
        W = torch.randn(4, 32) * 0.1
        X = torch.randn(12, 32)
        q, s, dq = enc._gptq_obs_rounding_fp8_like(
            W,
            X,
            fmt="MXFP8_E5M2",
            damp=0.01,
            joint_scale_opt=True,
        )
        q_ref, s_ref, dq_ref = enc._gptq_obs_rounding_fp8_like(
            W,
            X,
            fmt="MXFP8_E5M2",
            damp=0.01,
            joint_scale_opt=False,
        )
        self.assertEqual(q.dtype, torch.float8_e5m2)
        self.assertEqual(tuple(q.shape), tuple(W.shape))
        self.assertEqual(tuple(s.shape), (4, 1))
        self.assertEqual(s.dtype, torch.uint8)
        self.assertTrue(torch.equal(q, q_ref))
        self.assertTrue(torch.equal(s, s_ref))
        self.assertTrue(torch.equal(dq, dq_ref))
        scales = torch.pow(2.0, s.to(torch.float32) - 127.0)
        self.assertTrue(torch.allclose(dq, q.float() * scales.repeat_interleave(32, dim=1)))

    def test_nvfp4_gptq_matches_fpquant_column_update(self):
        torch.manual_seed(11)
        W = torch.randn(3, 16) * 0.1
        X = torch.randn(20, 16)
        group_size = 4
        damp = 0.01
        block_size = 7

        X_gptq = enc._activation_matrix_for_gptq(X, W.shape[1], device=W.device)
        W_work, U = _fpquant_gptq_factors_for_test(W, X_gptq, damp=damp)
        grouped = W_work.reshape(W.shape[0], W.shape[1] // group_size, group_size)
        s_g_real = enc._select_nvfp4_group_scales(grouped)
        global_real = (s_g_real.amax() / enc.FP8_E4M3_MAX).clamp_min(1e-12)
        scale_by_col = enc._nvfp4_effective_scale_from_real(
            s_g_real,
            global_real,
            quantize_fp8=True,
        ).repeat_interleave(group_size, dim=1)

        def quantize_column(col, col_idx):
            _idx, dq = enc._nvfp4_quantize_dequantize_with_eff_scale(
                col.unsqueeze(1),
                scale_by_col[:, col_idx:col_idx + 1],
            )
            return dq.squeeze(1)

        expected = _fpquant_columnwise_reference_for_test(
            W_work,
            U,
            block_size=block_size,
            quantize_column=quantize_column,
        )
        with patch.dict("os.environ", {"PRISMAQUANT_GPTQ_BLOCK_SIZE": str(block_size)}):
            actual = enc._gptq_obs_rounding_nvfp4(
                W,
                X,
                group_size=group_size,
                damp=damp,
                static_act_order=False,
                joint_scale_opt=False,
            )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_fp8_gptq_matches_fpquant_column_update(self):
        torch.manual_seed(12)
        W = torch.randn(3, 16) * 0.1
        X = torch.randn(20, 16)
        damp = 0.01
        block_size = 7

        X_gptq = enc._activation_matrix_for_gptq(X, W.shape[1], device=W.device)
        W_work, U = _fpquant_gptq_factors_for_test(W, X_gptq, damp=damp)
        scale = enc._fp8_dynamic_codec(W).scale
        q_ref = torch.empty_like(W, dtype=torch.float8_e4m3fn)

        def quantize_column(col, col_idx):
            q_col, dq = enc._fp8_quantize_dequantize_with_scale(
                col.unsqueeze(1),
                scale,
                element_dtype=torch.float8_e4m3fn,
                element_max=enc.FP8_E4M3_MAX,
            )
            q_ref[:, col_idx] = q_col.squeeze(1)
            return dq.squeeze(1)

        expected = _fpquant_columnwise_reference_for_test(
            W_work,
            U,
            block_size=block_size,
            quantize_column=quantize_column,
        )
        with patch.dict("os.environ", {"PRISMAQUANT_GPTQ_BLOCK_SIZE": str(block_size)}):
            q, s, dq = enc._gptq_obs_rounding_fp8_like(
                W,
                X,
                fmt="FP8_E4M3",
                damp=damp,
            )
        self.assertTrue(torch.equal(q, q_ref))
        torch.testing.assert_close(s, scale, atol=0, rtol=0)
        torch.testing.assert_close(dq, expected, atol=0, rtol=0)

    def test_mxfp8_gptq_matches_fpquant_column_update(self):
        torch.manual_seed(13)
        W = torch.randn(3, 16) * 0.1
        X = torch.randn(20, 16)
        group_size = 4
        damp = 0.01
        block_size = 7

        X_gptq = enc._activation_matrix_for_gptq(X, W.shape[1], device=W.device)
        W_work, U = _fpquant_gptq_factors_for_test(W, X_gptq, damp=damp)
        scale = torch.empty((W.shape[0], W.shape[1] // group_size), dtype=torch.uint8)
        for group_idx, block_start in enumerate(range(0, W.shape[1], group_size)):
            _q_block, scale_block, _dq = enc._mxfp8_quantize_dequantize_block(
                W[:, block_start:block_start + group_size],
                col_importance=None,
                joint_scale_opt=False,
                element_dtype=torch.float8_e4m3fn,
                element_max=enc.MXFP8_E4M3_MAX,
            )
            scale[:, group_idx] = scale_block
        scale_by_col = enc.e8m0_to_scale(scale).repeat_interleave(
            group_size,
            dim=1,
        )
        q_ref = torch.empty_like(W, dtype=torch.float8_e4m3fn)

        def quantize_column(col, col_idx):
            q_col, dq = enc._fp8_quantize_dequantize_with_scale(
                col.unsqueeze(1),
                scale_by_col[:, col_idx:col_idx + 1],
                element_dtype=torch.float8_e4m3fn,
                element_max=enc.MXFP8_E4M3_MAX,
            )
            q_ref[:, col_idx] = q_col.squeeze(1)
            return dq.squeeze(1)

        _expected_w = _fpquant_columnwise_reference_for_test(
            W_work,
            U,
            block_size=block_size,
            quantize_column=quantize_column,
        )
        expected_dq = enc._mxfp8_dequantize_2d(q_ref, scale, group_size=group_size)
        with patch.dict("os.environ", {"PRISMAQUANT_GPTQ_BLOCK_SIZE": str(block_size)}):
            q, s, dq = enc._gptq_obs_rounding_fp8_like(
                W,
                X,
                fmt="MXFP8_E4M3",
                group_size=group_size,
                damp=damp,
            )
        self.assertTrue(torch.equal(q, q_ref))
        self.assertTrue(torch.equal(s, scale))
        torch.testing.assert_close(_expected_w, expected_dq, atol=0, rtol=0)
        torch.testing.assert_close(dq, expected_dq, atol=0, rtol=0)
        scales = torch.pow(2.0, s.to(torch.float32) - 127.0)
        self.assertTrue(torch.allclose(dq, q.float() * scales.repeat_interleave(group_size, dim=1)))

    def test_mxfp8_gptq_static_act_order_restores_export_order(self):
        torch.manual_seed(14)
        W = torch.randn(3, 16) * 0.1
        X = torch.randn(20, 16)
        group_size = 4
        damp = 0.01
        block_size = 7

        X_gptq = enc._activation_matrix_for_gptq(X, W.shape[1], device=W.device)
        W_work = W.to(torch.float32).clone()
        H = X_gptq.to(torch.float32).t() @ X_gptq.to(torch.float32)
        diag_mean = torch.diagonal(H).mean().clamp_min(1e-12)
        H.diagonal().add_(float(damp) * diag_mean)
        dead = torch.diagonal(H) <= 0
        if dead.any():
            H[dead, dead] = 1.0
            W_work[:, dead] = 0.0
        col_importance = torch.diagonal(H).detach().clone().clamp_min(1e-12)

        scale = torch.empty((W.shape[0], W.shape[1] // group_size), dtype=torch.uint8)
        for group_idx, block_start in enumerate(range(0, W.shape[1], group_size)):
            _q_block, scale_block, _dq = enc._mxfp8_quantize_dequantize_block(
                W_work[:, block_start:block_start + group_size],
                col_importance=col_importance[block_start:block_start + group_size],
                joint_scale_opt=False,
                element_dtype=torch.float8_e4m3fn,
                element_max=enc.MXFP8_E4M3_MAX,
            )
            scale[:, group_idx] = scale_block
        scale_by_col = enc.e8m0_to_scale(scale).repeat_interleave(
            group_size,
            dim=1,
        )

        perm = torch.argsort(col_importance, descending=True)
        inverse_perm = torch.empty_like(perm)
        inverse_perm[perm] = torch.arange(W.shape[1], device=W.device)
        W_perm = W_work.index_select(1, perm).contiguous()
        H_perm = H.index_select(0, perm).index_select(1, perm).contiguous()
        L = torch.linalg.cholesky(H_perm)
        Hinv = torch.cholesky_inverse(L)
        U = torch.linalg.cholesky(Hinv, upper=True)
        scale_by_perm_col = scale_by_col.index_select(1, perm).contiguous()
        q_perm = torch.empty_like(W, dtype=torch.float8_e4m3fn)

        def quantize_column(col, col_idx):
            q_col, dq = enc._fp8_quantize_dequantize_with_scale(
                col.unsqueeze(1),
                scale_by_perm_col[:, col_idx:col_idx + 1],
                element_dtype=torch.float8_e4m3fn,
                element_max=enc.MXFP8_E4M3_MAX,
            )
            q_perm[:, col_idx] = q_col.squeeze(1)
            return dq.squeeze(1)

        _expected_perm_w = _fpquant_columnwise_reference_for_test(
            W_perm,
            U,
            block_size=block_size,
            quantize_column=quantize_column,
        )
        expected_q = q_perm.index_select(1, inverse_perm).contiguous()
        expected_dq = enc._mxfp8_dequantize_2d(
            expected_q,
            scale,
            group_size=group_size,
        )
        with patch.dict("os.environ", {"PRISMAQUANT_GPTQ_BLOCK_SIZE": str(block_size)}):
            q, s, dq = enc._gptq_obs_rounding_fp8_like(
                W,
                X,
                fmt="MXFP8_E4M3",
                group_size=group_size,
                damp=damp,
                static_act_order=True,
            )
        self.assertTrue(torch.equal(q, expected_q))
        self.assertTrue(torch.equal(s, scale))
        torch.testing.assert_close(dq, expected_dq, atol=0, rtol=0)

    def test_mxfp4_gptq_matches_fpquant_column_update(self):
        torch.manual_seed(16)
        W = torch.randn(3, 16) * 0.1
        X = torch.randn(20, 16)
        group_size = 4
        damp = 0.01
        block_size = 7

        X_gptq = enc._activation_matrix_for_gptq(X, W.shape[1], device=W.device)
        W_work, U = _fpquant_gptq_factors_for_test(W, X_gptq, damp=damp)
        scale = torch.empty((W.shape[0], W.shape[1] // group_size), dtype=torch.uint8)
        for group_idx, block_start in enumerate(range(0, W.shape[1], group_size)):
            codec = enc._mxfp4_grouped_codec(
                W_work[:, block_start:block_start + group_size]
            )
            scale[:, group_idx] = codec.scale
        scale_by_col = enc.e8m0_to_scale(scale).repeat_interleave(
            group_size,
            dim=1,
        )
        idx_ref = torch.empty_like(W, dtype=torch.uint8)

        def quantize_column(col, col_idx):
            idx_col, dq = enc._nvfp4_quantize_dequantize_with_eff_scale(
                col.unsqueeze(1),
                scale_by_col[:, col_idx:col_idx + 1],
            )
            idx_ref[:, col_idx] = idx_col.squeeze(1)
            return dq.squeeze(1)

        _expected_w = _fpquant_columnwise_reference_for_test(
            W_work,
            U,
            block_size=block_size,
            quantize_column=quantize_column,
        )
        expected_q = pack_fp4_indices(idx_ref, W.shape[1])
        expected_dq = enc._mxfp4_dequantize_2d(
            expected_q,
            scale,
            group_size=group_size,
        )
        with patch.dict("os.environ", {"PRISMAQUANT_GPTQ_BLOCK_SIZE": str(block_size)}):
            q, s, dq = enc._gptq_obs_rounding_mxfp4(
                W,
                X,
                group_size=group_size,
                damp=damp,
            )
        self.assertTrue(torch.equal(q, expected_q))
        self.assertTrue(torch.equal(s, scale))
        torch.testing.assert_close(_expected_w, expected_dq, atol=0, rtol=0)
        torch.testing.assert_close(dq, expected_dq, atol=0, rtol=0)

    def test_mxfp4_gptq_static_act_order_restores_export_order(self):
        torch.manual_seed(17)
        W = torch.randn(3, 16) * 0.1
        X = torch.randn(20, 16)
        group_size = 4
        damp = 0.01
        block_size = 7

        X_gptq = enc._activation_matrix_for_gptq(X, W.shape[1], device=W.device)
        W_work = W.to(torch.float32).clone()
        H = X_gptq.to(torch.float32).t() @ X_gptq.to(torch.float32)
        diag_mean = torch.diagonal(H).mean().clamp_min(1e-12)
        H.diagonal().add_(float(damp) * diag_mean)
        dead = torch.diagonal(H) <= 0
        if dead.any():
            H[dead, dead] = 1.0
            W_work[:, dead] = 0.0
        col_importance = torch.diagonal(H).detach().clone().clamp_min(1e-12)

        scale = torch.empty((W.shape[0], W.shape[1] // group_size), dtype=torch.uint8)
        for group_idx, block_start in enumerate(range(0, W.shape[1], group_size)):
            codec = enc._mxfp4_grouped_codec(
                W_work[:, block_start:block_start + group_size]
            )
            scale[:, group_idx] = codec.scale
        scale_by_col = enc.e8m0_to_scale(scale).repeat_interleave(
            group_size,
            dim=1,
        )

        perm = torch.argsort(col_importance, descending=True)
        inverse_perm = torch.empty_like(perm)
        inverse_perm[perm] = torch.arange(W.shape[1], device=W.device)
        W_perm = W_work.index_select(1, perm).contiguous()
        H_perm = H.index_select(0, perm).index_select(1, perm).contiguous()
        L = torch.linalg.cholesky(H_perm)
        Hinv = torch.cholesky_inverse(L)
        U = torch.linalg.cholesky(Hinv, upper=True)
        scale_by_perm_col = scale_by_col.index_select(1, perm).contiguous()
        idx_perm = torch.empty_like(W, dtype=torch.uint8)

        def quantize_column(col, col_idx):
            idx_col, dq = enc._nvfp4_quantize_dequantize_with_eff_scale(
                col.unsqueeze(1),
                scale_by_perm_col[:, col_idx:col_idx + 1],
            )
            idx_perm[:, col_idx] = idx_col.squeeze(1)
            return dq.squeeze(1)

        _expected_perm_w = _fpquant_columnwise_reference_for_test(
            W_perm,
            U,
            block_size=block_size,
            quantize_column=quantize_column,
        )
        expected_idx = idx_perm.index_select(1, inverse_perm).contiguous()
        expected_q = pack_fp4_indices(expected_idx, W.shape[1])
        expected_dq = enc._mxfp4_dequantize_2d(
            expected_q,
            scale,
            group_size=group_size,
        )
        expected_w = _expected_perm_w.index_select(1, inverse_perm).contiguous()
        with patch.dict("os.environ", {"PRISMAQUANT_GPTQ_BLOCK_SIZE": str(block_size)}):
            q, s, dq = enc._gptq_obs_rounding_mxfp4(
                W,
                X,
                group_size=group_size,
                damp=damp,
                static_act_order=True,
            )
        self.assertTrue(torch.equal(q, expected_q))
        self.assertTrue(torch.equal(s, scale))
        torch.testing.assert_close(expected_w, expected_dq, atol=0, rtol=0)
        torch.testing.assert_close(dq, expected_dq, atol=0, rtol=0)

    def test_plain_fp8_gptq_ignores_static_act_order(self):
        torch.manual_seed(15)
        W = torch.randn(3, 16) * 0.1
        X = torch.randn(20, 16)
        with patch.dict("os.environ", {"PRISMAQUANT_GPTQ_BLOCK_SIZE": "7"}):
            q_ref, s_ref, dq_ref = enc._gptq_obs_rounding_fp8_like(
                W,
                X,
                fmt="FP8_E4M3",
                static_act_order=False,
            )
            q, s, dq = enc._gptq_obs_rounding_fp8_like(
                W,
                X,
                fmt="FP8_E4M3",
                static_act_order=True,
            )
        self.assertTrue(torch.equal(q, q_ref))
        torch.testing.assert_close(s, s_ref, atol=0, rtol=0)
        torch.testing.assert_close(dq, dq_ref, atol=0, rtol=0)


class TestPackBits(unittest.TestCase):
    def test_round_to_codebook_signed(self):
        # Known mapping: 0→0, 0.5→1, 1.0→2, 6.0→7, -6.0→15
        v = torch.tensor([0.0, 0.5, 1.0, 6.0, -6.0])
        idx = _round_to_codebook(v)
        self.assertEqual(idx.tolist(), [0, 1, 2, 7, 15])

    def test_pack_fp4_two_per_byte(self):
        # Indices 1, 2 packed as low=1, high=2 → byte 0x21 = 33
        idx = torch.tensor([[1, 2, 3, 4]])
        packed = pack_fp4_indices(idx, 4)
        self.assertEqual(packed.shape, torch.Size([1, 2]))
        self.assertEqual(packed[0, 0].item(), (1 | (2 << 4)))
        self.assertEqual(packed[0, 1].item(), (3 | (4 << 4)))


class TestRecipeParsing(unittest.TestCase):
    def test_canonicalize_autoround_dict(self):
        nv = {"bits": 4, "data_type": "nv_fp"}
        mx8 = {"bits": 8, "data_type": "mx_fp"}
        bf = {"bits": 16, "data_type": "float"}
        self.assertEqual(canonicalize_format(nv), "NVFP4")
        self.assertEqual(canonicalize_format(mx8), "MXFP8_E4M3")
        self.assertEqual(canonicalize_format(bf), "BF16")
        self.assertEqual(canonicalize_format({"bits": 4, "data_type": "mx_fp"}), "MXFP4")


class TestVLLMInternalNaming(unittest.TestCase):
    """vLLM's qwen3_5 hf_to_vllm_mapper transforms source HF names to
    internal module names. The exporter's `quantization_config` targets
    + ignore must match the INTERNAL form so `find_matched_target`
    succeeds."""

    def test_text_only_recipe_naming_remap(self):
        self.assertEqual(
            _to_vllm_internal_name("model.layers.0.linear_attn.in_proj_qkv"),
            "language_model.model.layers.0.linear_attn.in_proj_qkv",
        )
        self.assertEqual(
            _to_vllm_internal_name("model.embed_tokens"),
            "language_model.model.embed_tokens",
        )

    def test_lm_head_remap(self):
        self.assertEqual(
            _to_vllm_internal_name("lm_head"),
            "language_model.lm_head",
        )

    def test_multimodal_source_naming_remap(self):
        # Source on-disk uses `model.language_model.X`; vLLM internal
        # is `language_model.model.X` (the prefix swap).
        self.assertEqual(
            _to_vllm_internal_name(
                "model.language_model.layers.5.mlp.shared_expert_gate"),
            "language_model.model.layers.5.mlp.shared_expert_gate",
        )

    def test_visual_remap(self):
        self.assertEqual(
            _to_vllm_internal_name("model.visual.blocks.0.attn.proj"),
            "visual.blocks.0.attn.proj",
        )


class TestBuildQuantizationConfig(unittest.TestCase):
    def test_minimal_two_format_assignment(self):
        profile = Qwen3_5Profile()
        # Lots of NVFP4, fewer MXFP8 → NVFP4 becomes the catch-all
        # bucket (largest count) and gets the per-expert pattern.
        assignment = {
            f"model.layers.{i}.self_attn.k_proj": "MXFP8"
            for i in range(2)  # 2 MXFP8 entries
        }
        for i in range(5):  # 5 NVFP4 entries
            assignment[f"model.layers.{i}.mlp.experts.down_proj"] = "NVFP4"
        qc = build_quantization_config(
            assignment, bf16_passthrough={"lm_head"}, profile=profile,
        )
        self.assertEqual(qc["quant_method"], "compressed-tensors")
        self.assertEqual(qc["format"], "mixed-precision")
        self.assertEqual(len(qc["config_groups"]), 2)
        # Find each group by num_bits — order isn't part of the contract
        groups_by_bits = {
            g["weights"]["num_bits"]: g
            for g in qc["config_groups"].values()
        }
        mxfp8 = groups_by_bits[8]
        nvfp4 = groups_by_bits[4]
        # MXFP8 group: explicit per-name regex targets only
        self.assertTrue(all(t.startswith("re:^language_model[.]")
                            for t in mxfp8["targets"]))
        self.assertNotIn(PER_EXPERT_MOE_REGEX, mxfp8["targets"])
        # NVFP4 catch-all: explicit + the per-expert pattern
        self.assertEqual(nvfp4["weights"]["strategy"], "tensor_group")
        self.assertEqual(nvfp4["weights"]["group_size"], 16)
        self.assertIn(PER_EXPERT_MOE_REGEX, nvfp4["targets"])
        # NVFP4 group must declare its per-group format so vLLM's
        # is_activation_quantization_format check enables W4A4 dispatch.
        self.assertEqual(nvfp4["format"], "nvfp4-pack-quantized")

    def test_lfm2_experts_use_canonical_vllm_scheme_names(self):
        # LFM2.5 names experts w1/w3/w2 on disk, but vLLM's FusedMoE scheme
        # detection (get_moe_method) and ignore matching probe the canonical
        # gate_proj/up_proj/down_proj. The exported config_groups targets AND
        # ignore regexes for packed experts must therefore use the canonical
        # names (weights still ship as w1/w2/w3), or vLLM mis-resolves the
        # scheme (weight-only NVFP4A16 / BF16 experts left un-ignored) and the
        # artifact fails to load.
        from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile
        profile = Lfm2MoeProfile()
        assignment = {
            "model.layers.2.feed_forward.experts.gate_up_proj": "NVFP4",
            "model.layers.2.feed_forward.experts.down_proj": "NVFP4",
        }
        qc = build_quantization_config(
            assignment,
            bf16_passthrough={
                "model.layers.3.feed_forward.experts.gate_up_proj",
                "model.layers.3.feed_forward.experts.down_proj",
            },
            profile=profile,
        )
        expert_targets = [t for g in qc["config_groups"].values()
                          for t in g["targets"] if "experts" in t]
        expert_ignores = [x for x in qc["ignore"]
                          if "experts" in x and x.startswith("re:")]
        all_expert = expert_targets + expert_ignores
        self.assertTrue(all_expert, "expected packed-expert regexes")
        self.assertTrue(any("gate_proj" in t for t in all_expert),
                        "no canonical gate_proj target/ignore emitted")
        # On-disk projection names must NOT leak into vLLM scheme regexes.
        for t in all_expert:
            self.assertNotIn("(w1", t, f"on-disk name leaked: {t}")
            self.assertNotIn("w3|", t, f"on-disk name leaked: {t}")
            self.assertNotIn("|w2", t, f"on-disk name leaked: {t}")

    def test_ignore_uses_vllm_internal_naming(self):
        profile = Qwen3_5Profile()
        assignment = {
            "model.layers.0.mlp.gate_proj": "NVFP4",
            "model.layers.0.mlp.shared_expert_gate": "BF16",
        }
        qc = build_quantization_config(
            assignment, bf16_passthrough={"lm_head"},
            extra_ignore=["model.layers.0.mlp.gate"],
            profile=profile,
        )
        ignore = qc["ignore"]
        self.assertIn("language_model.lm_head", ignore)
        self.assertIn(
            "language_model.model.layers.0.mlp.shared_expert_gate", ignore)
        self.assertIn(
            "language_model.model.layers.0.mlp.gate", ignore)

    def test_packed_moe_collapses_to_per_expert_regex(self):
        """Qwen3.5/3.6 packed-3D MoE loads as FusedMoE; vLLM's
        `get_moe_method` dispatches by building synthetic per-expert-0
        layer names (``<moe_prefix>.0.gate_proj`` / .up_proj / .down_proj)
        and calling `find_matched_target` on each. The packed-tensor
        qnames we emit don't match that form — we must emit a regex
        pinned to this layer's FusedMoE that covers the per-expert
        projection forms so scheme dispatch fires."""
        profile = Qwen3_5Profile()
        assignment = {
            "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
            "model.layers.0.mlp.experts.down_proj":    "NVFP4",
            # Padding with non-experts so by_fmt isn't empty and the
            # catch-all path runs normally.
            "model.layers.0.self_attn.q_proj":         "NVFP4",
        }
        qc = build_quantization_config(
            assignment, bf16_passthrough=set(), profile=profile,
        )
        # Find the NVFP4 group
        nvfp4 = next(g for g in qc["config_groups"].values()
                     if g["weights"]["num_bits"] == 4)
        targets = nvfp4["targets"]
        # Per-expert regex for layer 0's FusedMoE — matches the
        # "unfused" per-expert layer_name form vLLM builds at
        # scheme-dispatch time.
        expected = (
            r"re:^language_model\[\.\]model\[\.\]layers\[\.\]0\[\.\]mlp\[\.\]"
            r"experts\[\.\]\[0\-9\]\+\[\.\]\(gate_proj\|up_proj\|down_proj\)\$"
        )
        # Looser match: just check the shape rather than exact string
        # (the re.escape() inside _per_expert_regex_for adds backslashes).
        has_per_expert = any(
            t.startswith("re:^")
            and "mlp[.]experts[.][0-9]+[.]" in t
            and "(gate_proj|up_proj|down_proj)$" in t
            for t in targets
        )
        self.assertTrue(has_per_expert,
                        f"missing per-expert MoE target; got {targets}")
        # No packed-tensor-name target should leak in.
        for t in targets:
            self.assertFalse(
                t.endswith("mlp[.]experts[.]gate_up_proj$"),
                f"packed tensor name leaked: {t}")
            self.assertFalse(
                t.endswith("mlp[.]experts[.]down_proj$"),
                f"packed tensor name leaked: {t}")

    def test_packed_moe_regex_uses_profile_projection_splits(self):
        profile = _CustomPackedProfile()
        assignment = {
            "model.layers.0.mlp.experts.w13": "NVFP4",
            "model.layers.0.mlp.experts.w2": "NVFP4",
            "model.layers.0.self_attn.q_proj": "MXFP8",
        }
        qc = build_quantization_config(
            assignment, bf16_passthrough=set(), profile=profile,
        )
        targets = [
            target
            for group in qc["config_groups"].values()
            for target in group["targets"]
        ]

        packed_targets = [
            target for target in targets
            if "mlp[.]experts[.][0-9]+[.]" in target
        ]
        self.assertEqual(
            packed_targets,
            [
                "re:^model[.]layers[.]0[.]mlp[.]experts"
                "[.][0-9]+[.](w1_proj|w3_proj|w2)$"
            ],
        )
        self.assertNotIn(
            "(gate_proj|up_proj|down_proj)",
            packed_targets[0],
        )

    def test_bf16_packed_ignore_uses_profile_projection_regex(self):
        profile = _CustomPackedRegexProfile()
        regexes = enc._bf16_packed_expert_ignore_regex(
            "model.layers.3.mlp.experts.w13",
            profile,
        )

        self.assertEqual(
            regexes,
            [
                "re:^serving[.]layers[.]3[.]moe[.]experts"
                "[.][0-9]+[.](w1_proj|w3_proj)$"
            ],
        )
        self.assertNotIn("w2", regexes[0])
        self.assertNotIn("gate|up|down", regexes[0])

    def test_bf16_mtp_ignore_does_not_taint_body_layer(self):
        """A BF16 MTP `mtp.layers.N.mlp.experts.*` assignment must emit
        an `mtp.*`-prefixed ignore regex, NOT a body `language_model.
        model.layers.N.*` regex. Otherwise the body's NVFP4 MoE at
        layer N is accidentally ignored → scheme dispatch fails →
        load_weights KeyErrors on `w2_input_global_scale`."""
        import re as _re
        profile = Qwen3_5Profile()
        assignment = {
            "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
            "model.layers.0.mlp.experts.down_proj":    "NVFP4",
            "mtp.layers.0.mlp.experts.gate_up_proj":   "BF16",
            "mtp.layers.0.mlp.experts.down_proj":      "BF16",
        }
        qc = build_quantization_config(
            assignment, bf16_passthrough=set(), profile=profile,
        )
        # Body layer 0 per-expert form MUST NOT match any ignore regex.
        body_ln = "language_model.model.layers.0.mlp.experts.0.gate_proj"
        hits = [
            i for i in qc["ignore"]
            if i.startswith("re:") and _re.match(i[3:], body_ln)
        ]
        self.assertEqual(hits, [],
                         f"BF16 MTP leaked into body-layer ignore: {hits}")
        # MTP layer 0 per-expert form SHOULD match an mtp-prefixed regex.
        mtp_ln = "mtp.layers.0.mlp.experts.0.gate_proj"
        mtp_hits = [
            i for i in qc["ignore"]
            if i.startswith("re:^mtp[.]") and _re.match(i[3:], mtp_ln)
        ]
        self.assertGreater(len(mtp_hits), 0,
                           f"missing MTP-prefixed ignore regex for MTP layer")

    def test_materialized_mtp_packed_parent_filters_source_expert_children(self):
        """BF16 MTP packed experts are synthesized as vLLM aggregate tensors.

        The source checkpoint stores the same weights as per-expert children.
        Copying those children as passthrough creates duplicate MTP expert keys
        and vLLM warns that `layers.0.mlp.experts.N.*.weight` was not found.
        """
        profile = Qwen3_5Profile()
        materialized = {
            "mtp.layers.0.mlp.experts.gate_up_proj": torch.empty(1),
            "mtp.layers.0.mlp.experts.down_proj": torch.empty(1),
        }
        src_extra = {
            "mtp.layers.0.mlp.experts.0.gate_proj.weight": torch.empty(1),
            "mtp.layers.0.mlp.experts.0.up_proj.weight": torch.empty(1),
            "mtp.layers.0.mlp.experts.0.down_proj.weight": torch.empty(1),
            "mtp.layers.0.input_layernorm.weight": torch.empty(1),
            "model.visual.blocks.0.norm.weight": torch.empty(1),
        }

        filtered = enc._filter_source_passthrough_against_materialized(
            src_extra,
            materialized,
            profile=profile,
        )

        self.assertNotIn(
            "mtp.layers.0.mlp.experts.0.gate_proj.weight", filtered)
        self.assertNotIn(
            "mtp.layers.0.mlp.experts.0.up_proj.weight", filtered)
        self.assertNotIn(
            "mtp.layers.0.mlp.experts.0.down_proj.weight", filtered)
        self.assertIn("mtp.layers.0.input_layernorm.weight", filtered)
        self.assertIn("model.visual.blocks.0.norm.weight", filtered)

    def test_packed_moe_mixed_format_rejected(self):
        """Different formats on gate_up_proj and down_proj of the same
        FusedMoE is a promote_moe_pair bug — we loud-crash rather than
        emit a malformed config."""
        profile = Qwen3_5Profile()
        assignment = {
            "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
            "model.layers.0.mlp.experts.down_proj":    "MXFP8",
        }
        with self.assertRaises(RuntimeError):
            build_quantization_config(
                assignment, bf16_passthrough=set(), profile=profile,
            )

    def test_no_class_name_catchall_target(self):
        # The class-name catch-all "Linear" short-circuits vLLM's
        # fused-layer match path and was the bug that produced wrong
        # scheme allocation. Make sure we don't reintroduce it.
        assignment = {"model.layers.0.mlp.gate_proj": "NVFP4"}
        qc = build_quantization_config(
            assignment, bf16_passthrough=set(), profile=Qwen3_5Profile()
        )
        for group in qc["config_groups"].values():
            for t in group["targets"]:
                self.assertNotEqual(t, "Linear",
                                    "do not use a 'Linear' class-name catch-all; "
                                    "it short-circuits fused-layer match")

    def test_fused_targets_are_emitted_from_structure_spec(self):
        profile = Qwen3_5Profile()
        assignment = {
            "model.layers.0.mlp.gate_proj": "NVFP4",
            "model.layers.0.mlp.up_proj": "NVFP4",
        }
        qc = build_quantization_config(
            assignment, bf16_passthrough=set(), profile=profile,
        )
        targets = {
            target
            for group in qc["config_groups"].values()
            for target in group["targets"]
        }

        self.assertIn(
            "re:^language_model[.]model[.]layers[.]0[.]mlp[.]gate_up_proj$",
            targets,
        )

    def test_missing_bf16_fused_siblings_are_ignored_from_structure_spec(self):
        profile = Qwen3_5Profile()
        assignment = {
            "model.layers.0.self_attn.q_proj": "BF16",
            "model.layers.0.self_attn.k_proj": "BF16",
            "model.layers.0.mlp.down_proj": "NVFP4",
        }
        qc = build_quantization_config(
            assignment, bf16_passthrough=set(), profile=profile,
        )

        self.assertIn(
            "language_model.model.layers.0.self_attn.v_proj",
            qc["ignore"],
        )
        self.assertIn(
            "language_model.model.layers.0.self_attn.qkv_proj",
            qc["ignore"],
        )


class TestQuantize2DDispatch(unittest.TestCase):
    def test_nvfp4_emits_input_global_scale(self):
        """vLLM's CompressedTensorsW4A4Nvfp4 process_weights_after_loading
        does `1 / input_global_scale.max()`. Without an emitted value,
        the param defaults to zeros and vLLM produces 1/0 = inf →
        degenerate output. Make sure we always emit it."""
        W = torch.randn(8, 16) * 0.1
        out = _quantize_2d(W, "NVFP4")
        self.assertIn("weight_packed", out)
        self.assertIn("weight_scale", out)
        self.assertIn("weight_global_scale", out)
        self.assertIn("input_global_scale", out)
        self.assertEqual(out["input_global_scale"].dtype, torch.float32)
        self.assertEqual(out["input_global_scale"].numel(), 1)
        self.assertAlmostEqual(
            out["input_global_scale"].item(), DEFAULT_INPUT_GLOBAL_SCALE)

    def test_mxfp8_emits_grouped_dense(self):
        W = torch.randn(8, 32) * 0.1
        out = _quantize_2d(W, "MXFP8")
        self.assertIn("weight", out)
        self.assertEqual(out["weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(out["weight_scale"].dtype, torch.uint8)
        self.assertEqual(tuple(out["weight_scale"].shape), (8, 1))

    def test_mxfp4_emits_packed_grouped_dense(self):
        W = torch.randn(8, 32) * 0.1
        out = _quantize_2d(W, "MXFP4")
        self.assertIn("weight_packed", out)
        self.assertEqual(out["weight_packed"].dtype, torch.uint8)
        self.assertEqual(tuple(out["weight_packed"].shape), (8, 16))
        self.assertEqual(out["weight_scale"].dtype, torch.uint8)
        self.assertEqual(tuple(out["weight_scale"].shape), (8, 1))

    def test_mxfp8_export_static_act_order_do_no_harm_selects_lower_candidate(self):
        import os
        import prismaquant.export_native_compressed as m

        W = torch.randn(4, 32) * 0.1
        X = torch.randn(12, 32)
        calls = []

        def fake_gptq(weight, activations, **kwargs):
            del activations
            use_static = bool(kwargs["static_act_order"])
            calls.append(use_static)
            marker = 2.0 if use_static else 1.0
            q = torch.full(
                tuple(weight.shape),
                marker,
                dtype=torch.float32,
                device=weight.device,
            ).to(torch.float8_e4m3fn)
            s = torch.zeros(
                (weight.shape[0], weight.shape[1] // 32),
                dtype=torch.uint8,
                device=weight.device,
            )
            dq = weight.to(torch.float32) + (1.0 if use_static else 0.0)
            return q, s, dq

        saved = m._gptq_obs_rounding_fp8_like
        saved_sweep = os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP")
        try:
            os.environ["PRISMAQUANT_GPTQ_DAMP_SWEEP"] = "0"
            m._gptq_obs_rounding_fp8_like = fake_gptq
            out = m._quantize_2d(
                W,
                "MXFP8_E4M3",
                gptq_enabled=True,
                static_act_order_enabled=True,
                cached_activations=X,
            )
        finally:
            m._gptq_obs_rounding_fp8_like = saved
            if saved_sweep is None:
                os.environ.pop("PRISMAQUANT_GPTQ_DAMP_SWEEP", None)
            else:
                os.environ["PRISMAQUANT_GPTQ_DAMP_SWEEP"] = saved_sweep

        self.assertEqual(calls, [False, True])
        self.assertTrue(torch.equal(
            out["weight"].float(),
            torch.ones_like(W),
        ))

    def test_mxfp8_export_do_no_harm_reverts_bad_gptq_to_rtn(self):
        import os
        import prismaquant.export_native_compressed as m

        W = torch.randn(4, 32) * 0.1
        X = torch.randn(12, 32)

        def fake_gptq(weight, activations, **kwargs):
            del activations, kwargs
            q = torch.full(
                tuple(weight.shape),
                7.0,
                dtype=torch.float32,
                device=weight.device,
            ).to(torch.float8_e4m3fn)
            s = torch.zeros(
                (weight.shape[0], weight.shape[1] // 32),
                dtype=torch.uint8,
                device=weight.device,
            )
            return q, s, weight.to(torch.float32) + 100.0

        saved = m._gptq_obs_rounding_fp8_like
        saved_sweep = os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP")
        saved_dnh = os.environ.get("PRISMAQUANT_DO_NO_HARM")
        try:
            os.environ["PRISMAQUANT_GPTQ_DAMP_SWEEP"] = "0"
            os.environ["PRISMAQUANT_DO_NO_HARM"] = "1"
            m._gptq_obs_rounding_fp8_like = fake_gptq
            out = m._quantize_2d(
                W,
                "MXFP8_E4M3",
                gptq_enabled=True,
                cached_activations=X,
            )
        finally:
            m._gptq_obs_rounding_fp8_like = saved
            if saved_sweep is None:
                os.environ.pop("PRISMAQUANT_GPTQ_DAMP_SWEEP", None)
            else:
                os.environ["PRISMAQUANT_GPTQ_DAMP_SWEEP"] = saved_sweep
            if saved_dnh is None:
                os.environ.pop("PRISMAQUANT_DO_NO_HARM", None)
            else:
                os.environ["PRISMAQUANT_DO_NO_HARM"] = saved_dnh

        q_rtn, s_rtn = quantize_dequantize_mxfp8(W)
        self.assertTrue(torch.equal(out["weight"], q_rtn))
        self.assertTrue(torch.equal(out["weight_scale"], s_rtn))

    def test_mxfp4_packed_expert_shapes(self):
        W = torch.randn(2, 6, 32) * 0.1
        wp, ws = quantize_dequantize_mxfp4_packed(W)
        self.assertEqual(wp.dtype, torch.uint8)
        self.assertEqual(tuple(wp.shape), (2, 6, 16))
        self.assertEqual(ws.dtype, torch.uint8)
        self.assertEqual(tuple(ws.shape), (2, 6, 1))


class TestProductionCacheExportPath(unittest.TestCase):
    def test_packs_cached_nvfp4_weight_with_cached_input_scale(self):
        import prismaquant.export_native_compressed as m
        from prismaquant.production_weight_cache import ProductionWeightCache

        W = torch.randn(8, 16) * 0.1
        cache = ProductionWeightCache(
            weights={("model.layers.0.mlp.down_proj", "NVFP4"): W},
            levers={"gptq": True, "scale_sweep": True},
            activation_max_abs={"model.layers.0.mlp.down_proj": 3.0},
        )
        saved_cache = m._PRODUCTION_WEIGHT_CACHE
        saved_scales = m._INPUT_GLOBAL_SCALES
        try:
            m._PRODUCTION_WEIGHT_CACHE = cache
            m._INPUT_GLOBAL_SCALES = m._production_cache_scales(cache)
            out = m._pack_production_cached_2d(
                "model.layers.0.mlp.down_proj",
                "NVFP4",
                device=torch.device("cpu"),
            )
            self.assertIsNotNone(out)
            self.assertIn("weight_packed", out)
            self.assertIn("input_global_scale", out)
            self.assertAlmostEqual(
                float(out["input_global_scale"].item()), 2.0, places=5)
        finally:
            m._PRODUCTION_WEIGHT_CACHE = saved_cache
            m._INPUT_GLOBAL_SCALES = saved_scales

    def test_mxfp8_alias_hits_e4m3_cache_key(self):
        import prismaquant.export_native_compressed as m
        from prismaquant.production_weight_cache import ProductionWeightCache

        W = torch.randn(8, 32) * 0.1
        cache = ProductionWeightCache(
            weights={("model.layers.0.self_attn.q_proj", "MXFP8_E4M3"): W},
            levers={},
        )
        saved_cache = m._PRODUCTION_WEIGHT_CACHE
        try:
            m._PRODUCTION_WEIGHT_CACHE = cache
            out = m._pack_production_cached_2d(
                "model.layers.0.self_attn.q_proj",
                "MXFP8",
                device=torch.device("cpu"),
            )
            self.assertIsNotNone(out)
            self.assertEqual(out["weight"].dtype, torch.float8_e4m3fn)
            self.assertEqual(out["weight_scale"].dtype, torch.uint8)
        finally:
            m._PRODUCTION_WEIGHT_CACHE = saved_cache

    def test_mxfp8_scale_sweep_cache_defers_to_export_recompute(self):
        import prismaquant.export_native_compressed as m
        from prismaquant.production_weight_cache import ProductionWeightCache

        W = torch.randn(8, 32) * 0.1
        cache = ProductionWeightCache(
            weights={("model.layers.0.self_attn.q_proj", "MXFP8_E4M3"): W},
            levers={"scale_sweep": True},
        )
        saved_cache = m._PRODUCTION_WEIGHT_CACHE
        saved_acts = m._CACHED_ACTIVATIONS
        try:
            m._PRODUCTION_WEIGHT_CACHE = cache
            m._CACHED_ACTIVATIONS = object()
            out = m._pack_production_cached_2d(
                "model.layers.0.self_attn.q_proj",
                "MXFP8_E4M3",
                device=torch.device("cpu"),
            )
            self.assertIsNone(out)
        finally:
            m._PRODUCTION_WEIGHT_CACHE = saved_cache
            m._CACHED_ACTIVATIONS = saved_acts

    def test_fp8_scale_sweep_cache_defers_to_export_recompute(self):
        import prismaquant.export_native_compressed as m
        from prismaquant.production_weight_cache import ProductionWeightCache

        W = torch.randn(8, 32) * 0.1
        cache = ProductionWeightCache(
            weights={("model.layers.0.self_attn.q_proj", "FP8_E4M3"): W},
            levers={"scale_sweep": True},
        )
        saved_cache = m._PRODUCTION_WEIGHT_CACHE
        saved_acts = m._CACHED_ACTIVATIONS
        try:
            m._PRODUCTION_WEIGHT_CACHE = cache
            m._CACHED_ACTIVATIONS = object()
            out = m._pack_production_cached_2d(
                "model.layers.0.self_attn.q_proj",
                "FP8_E4M3",
                device=torch.device("cpu"),
            )
            self.assertIsNone(out)
        finally:
            m._PRODUCTION_WEIGHT_CACHE = saved_cache
            m._CACHED_ACTIVATIONS = saved_acts

    def test_fp8_gptq_cache_defers_to_export_recompute(self):
        import prismaquant.export_native_compressed as m
        from prismaquant.production_weight_cache import ProductionWeightCache

        W = torch.randn(8, 32) * 0.1
        cache = ProductionWeightCache(
            weights={("model.layers.0.self_attn.q_proj", "FP8_E4M3"): W},
            levers={"gptq": True},
        )
        saved_cache = m._PRODUCTION_WEIGHT_CACHE
        saved_acts = m._CACHED_ACTIVATIONS
        try:
            m._PRODUCTION_WEIGHT_CACHE = cache
            m._CACHED_ACTIVATIONS = object()
            out = m._pack_production_cached_2d(
                "model.layers.0.self_attn.q_proj",
                "FP8_E4M3",
                device=torch.device("cpu"),
            )
            self.assertIsNone(out)
        finally:
            m._PRODUCTION_WEIGHT_CACHE = saved_cache
            m._CACHED_ACTIVATIONS = saved_acts

    def test_mxfp8_e4m3_gptq_cache_defers_to_export_recompute(self):
        import prismaquant.export_native_compressed as m
        from prismaquant.production_weight_cache import ProductionWeightCache

        W = torch.randn(8, 32) * 0.1
        cache = ProductionWeightCache(
            weights={("model.layers.0.self_attn.q_proj", "MXFP8_E4M3"): W},
            levers={"gptq": True, "joint_scale_opt": True},
        )
        saved_cache = m._PRODUCTION_WEIGHT_CACHE
        saved_acts = m._CACHED_ACTIVATIONS
        try:
            m._PRODUCTION_WEIGHT_CACHE = cache
            m._CACHED_ACTIVATIONS = object()
            out = m._pack_production_cached_2d(
                "model.layers.0.self_attn.q_proj",
                "MXFP8_E4M3",
                device=torch.device("cpu"),
            )
            self.assertIsNone(out)
        finally:
            m._PRODUCTION_WEIGHT_CACHE = saved_cache
            m._CACHED_ACTIVATIONS = saved_acts

class TestFusedSiblingJointGlobalScale(unittest.TestCase):
    """vLLM warns when q/k/v/gate/up have different weight_global_scale.
    The exporter pre-computes a joint per-tensor scale across each
    fused-sibling group so the warning goes away (and the per-tensor
    scale on disk is correct under vLLM's fused-loader rules)."""

    def test_fused_dense_group_self_attn(self):
        from prismaquant.export_native_compressed import _fused_dense_group
        g = _fused_dense_group("model.layers.5.self_attn.q_proj")
        self.assertIsNotNone(g)
        pre, members = g
        self.assertEqual(pre, "model.layers.5")
        self.assertIn("k_proj", members)

    def test_fused_dense_group_mlp_gate_up(self):
        from prismaquant.export_native_compressed import _fused_dense_group
        g = _fused_dense_group("model.layers.0.mlp.shared_expert.up_proj")
        self.assertIsNotNone(g)
        self.assertEqual(set(g[1]), {"gate_proj", "up_proj"})

    def test_fused_dense_group_qwen36_linear_attn(self):
        from prismaquant.export_native_compressed import _fused_dense_group
        for sib in ("in_proj_qkv", "in_proj_z"):
            g = _fused_dense_group(f"model.layers.7.linear_attn.{sib}")
            self.assertIsNotNone(g, f"missing fused-group pattern for {sib}")
            self.assertEqual(set(g[1]), {"in_proj_qkv", "in_proj_z"})

    def test_compute_nvfp4_joint_global_picks_max(self):
        from prismaquant.export_native_compressed import (
            _compute_nvfp4_joint_global, compute_nvfp4_global_real,
        )

        # Build a tiny model with two fused-sibling Linears (different
        # max-abs values). The joint scale must be the max of their
        # natural per-tensor scales.
        class TinyAttn(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.q_proj = torch.nn.Linear(32, 32, bias=False)
                s.k_proj = torch.nn.Linear(32, 32, bias=False)
                s.v_proj = torch.nn.Linear(32, 32, bias=False)

        class TinyLayer(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.self_attn = TinyAttn()

        class TinyModel(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.model = torch.nn.Module()
                s.model.layers = torch.nn.ModuleList([TinyLayer()])

        torch.manual_seed(0)
        m = TinyModel()
        # Force k_proj to have the largest max-abs.
        with torch.no_grad():
            m.model.layers[0].self_attn.k_proj.weight.mul_(10.0)

        assignment = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.self_attn.k_proj": "NVFP4",
            "model.layers.0.self_attn.v_proj": "NVFP4",
        }
        joint = _compute_nvfp4_joint_global(m, assignment)
        self.assertEqual(len(joint), 3)
        joint_value = next(iter(joint.values())).item()
        # All three must point to the SAME scalar.
        for v in joint.values():
            self.assertAlmostEqual(v.item(), joint_value)
        # And it must be at least the natural scale of the max sibling.
        natural = compute_nvfp4_global_real(
            m.model.layers[0].self_attn.k_proj.weight.float()).item()
        self.assertAlmostEqual(joint_value, natural, places=5)

    def test_profile_fused_group_drives_joint_global_without_baked_pattern(self):
        from prismaquant.export_native_compressed import _compute_nvfp4_joint_global

        class TinyBlock(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.a_proj = torch.nn.Linear(32, 32, bias=False)
                s.b_proj = torch.nn.Linear(32, 32, bias=False)

        class TinyModel(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.model = torch.nn.Module()
                s.model.layers = torch.nn.ModuleList([TinyBlock()])

        class CustomProfile:
            def fused_sibling_group(self, qname: str) -> str | None:
                if qname.endswith(".a_proj") or qname.endswith(".b_proj"):
                    return qname.rsplit(".", 1)[0] + ".ab_proj"
                return None

        torch.manual_seed(0)
        m = TinyModel()
        with torch.no_grad():
            m.model.layers[0].b_proj.weight.mul_(10.0)
        assignment = {
            "model.layers.0.a_proj": "NVFP4",
            "model.layers.0.b_proj": "NVFP4",
        }

        joint = _compute_nvfp4_joint_global(
            m,
            assignment,
            profile=CustomProfile(),
        )

        self.assertEqual(set(joint), set(assignment))
        self.assertEqual(
            joint["model.layers.0.a_proj"].item(),
            joint["model.layers.0.b_proj"].item(),
        )

    def test_profile_fused_group_drives_input_scale_unification(self):
        from prismaquant.export_native_compressed import (
            _unify_input_global_scales_across_fused_siblings,
        )

        class CustomProfile:
            def fused_sibling_group(self, qname: str) -> str | None:
                if qname.endswith(".a_proj") or qname.endswith(".b_proj"):
                    return qname.rsplit(".", 1)[0] + ".ab_proj"
                return None

        out = _unify_input_global_scales_across_fused_siblings(
            {
                "model.layers.0.a_proj": 0.50,
                "model.layers.0.b_proj": 0.25,
            },
            profile=CustomProfile(),
        )

        self.assertEqual(out["model.layers.0.a_proj"], 0.25)
        self.assertEqual(out["model.layers.0.b_proj"], 0.25)


class TestPackedExpertSplit(unittest.TestCase):
    def test_quantize_3d_packed_nvfp4_returns_per_expert_dim(self):
        # 3D packed `[E, M, N]` produces tensors with leading expert
        # dim preserved. Splitting into per-expert-per-projection is
        # done in materialize_tensors, not _quantize_3d_packed.
        E, M, N = 4, 32, 64
        P = torch.randn(E, M, N) * 0.05
        out = _quantize_3d_packed(P, "NVFP4")
        self.assertEqual(out["weight_packed"].shape[0], E)
        self.assertEqual(out["weight_global_scale"].shape, torch.Size([E]))

    def test_quantize_3d_packed_fp8_returns_per_expert_channel_scales(self):
        E, M, N = 4, 32, 64
        P = torch.randn(E, M, N) * 0.05
        out = _quantize_3d_packed(P, "FP8_E4M3")
        self.assertEqual(out["weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(out["weight"].shape, torch.Size([E, M, N]))
        self.assertEqual(out["weight_scale"].dtype, torch.float32)
        self.assertEqual(out["weight_scale"].shape, torch.Size([E, M, 1]))


class TestQwen35ProfileFallback(unittest.TestCase):
    def _cpu_only_profile(self):
        profile = Qwen3_5Profile()
        profile._vllm_cls = None
        profile._vllm_cls_loaded = True
        profile._fused_matcher = None
        return profile

    def test_fused_sibling_group_has_cpu_only_fallback(self):
        profile = self._cpu_only_profile()

        self.assertEqual(
            profile.fused_sibling_group(
                "model.layers.25.linear_attn.in_proj_qkv"
            ),
            "model.layers.25.linear_attn.in_proj_qkvz",
        )
        self.assertEqual(
            profile.fused_sibling_group(
                "model.layers.25.linear_attn.in_proj_z"
            ),
            "model.layers.25.linear_attn.in_proj_qkvz",
        )
        self.assertEqual(
            profile.fused_sibling_group(
                "model.layers.25.linear_attn.in_proj_a"
            ),
            "model.layers.25.linear_attn.in_proj_ba",
        )
        self.assertEqual(
            profile.fused_sibling_group(
                "model.layers.25.self_attn.q_proj"
            ),
            "model.layers.25.self_attn.qkv_proj",
        )
        self.assertEqual(
            profile.fused_sibling_group(
                "model.layers.25.mlp.shared_expert.gate_proj"
            ),
            "model.layers.25.mlp.shared_expert.gate_up_proj",
        )
        self.assertEqual(
            profile.fused_sibling_group(
                "model.layers.25.mlp.shared_expert.up_proj"
            ),
            "model.layers.25.mlp.shared_expert.gate_up_proj",
        )

    def test_promote_fused_keeps_linear_attn_qkvz_coherent_without_vllm(self):
        profile = self._cpu_only_profile()
        assignment = {
            "model.layers.25.linear_attn.in_proj_qkv": "MXFP8",
            "model.layers.25.linear_attn.in_proj_z": "NVFP4",
            "model.layers.25.linear_attn.in_proj_a": "NVFP4",
            "model.layers.25.linear_attn.in_proj_b": "NVFP4",
        }

        promoted = promote_fused(
            assignment,
            {"BF16": 0, "NVFP4": 1, "MXFP8": 2},
            profile=profile,
        )

        self.assertEqual(promoted["model.layers.25.linear_attn.in_proj_qkv"], "MXFP8")
        self.assertEqual(promoted["model.layers.25.linear_attn.in_proj_z"], "MXFP8")
        self.assertEqual(promoted["model.layers.25.linear_attn.in_proj_a"], "NVFP4")
        self.assertEqual(promoted["model.layers.25.linear_attn.in_proj_b"], "NVFP4")


class TestMtpCoverageValidation(unittest.TestCase):
    class _Profile:
        def has_mtp(self):
            return True

    def test_validate_mtp_assignment_coverage_raises_when_recipe_omits_mtp(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with open(td / "model.safetensors.index.json", "w") as f:
                json.dump({"weight_map": {"mtp.fc.weight": "model-00001.safetensors"}}, f)

            with self.assertRaisesRegex(RuntimeError, "contains no mtp"):
                validate_mtp_assignment_coverage(
                    str(td),
                    {"model.layers.0.self_attn.q_proj": "NVFP4"},
                    self._Profile(),
                )

    def test_validate_mtp_assignment_coverage_accepts_recipe_with_mtp(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with open(td / "model.safetensors.index.json", "w") as f:
                json.dump({"weight_map": {"mtp.fc.weight": "model-00001.safetensors"}}, f)

            validate_mtp_assignment_coverage(
                str(td),
                {"mtp.fc": "BF16"},
                self._Profile(),
            )


class TestRuntimeLegalAssignment(unittest.TestCase):
    def test_coerces_runtime_illegal_mxfp8_shape_to_bf16(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            shard = td / "model-00001-of-00001.safetensors"
            save_file({
                "model.layers.0.linear_attn.in_proj_a.weight": torch.zeros(
                    48, 5120, dtype=torch.bfloat16
                ),
                "model.layers.0.self_attn.o_proj.weight": torch.zeros(
                    128, 5120, dtype=torch.bfloat16
                ),
            }, str(shard))
            with open(td / "model.safetensors.index.json", "w") as f:
                json.dump({
                    "weight_map": {
                        "model.layers.0.linear_attn.in_proj_a.weight": shard.name,
                        "model.layers.0.self_attn.o_proj.weight": shard.name,
                    }
                }, f)

            assignment, coerced = _coerce_runtime_legal_assignment(str(td), {
                "model.layers.0.linear_attn.in_proj_a": "MXFP8_E4M3",
                "model.layers.0.self_attn.o_proj": "MXFP8",
            })

        self.assertEqual(assignment["model.layers.0.linear_attn.in_proj_a"], "BF16")
        self.assertEqual(
            assignment["model.layers.0.self_attn.o_proj"],
            "MXFP8_E4M3",
        )
        self.assertEqual(coerced, [
            ("model.layers.0.linear_attn.in_proj_a", [48, 5120], "MXFP8_E4M3")
        ])

    def test_coerces_profile_illegal_dense_format_to_bf16(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            shard = td / "model-00001-of-00001.safetensors"
            save_file({
                "model.language_model.layers.0.self_attn.o_proj.weight": (
                    torch.zeros(128, 5120, dtype=torch.bfloat16)
                ),
            }, str(shard))
            with open(td / "model.safetensors.index.json", "w") as f:
                json.dump({
                    "weight_map": {
                        "model.language_model.layers.0.self_attn.o_proj.weight": (
                            shard.name
                        ),
                    }
                }, f)

            assignment, coerced = _coerce_runtime_legal_assignment(
                str(td),
                {"model.layers.0.self_attn.o_proj": "MXFP4"},
                Qwen3_5Profile(),
            )

        self.assertEqual(assignment["model.layers.0.self_attn.o_proj"], "BF16")
        self.assertEqual(coerced, [
            ("model.layers.0.self_attn.o_proj", [128, 5120], "MXFP4")
        ])

    def test_coerces_parsed_research_format_without_export_scheme_to_bf16(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            shard = td / "model-00001-of-00001.safetensors"
            save_file({
                "model.language_model.layers.0.self_attn.o_proj.weight": (
                    torch.zeros(128, 5120, dtype=torch.bfloat16)
                ),
            }, str(shard))
            with open(td / "model.safetensors.index.json", "w") as f:
                json.dump({
                    "weight_map": {
                        "model.language_model.layers.0.self_attn.o_proj.weight": (
                            shard.name
                        ),
                    }
                }, f)

            assignment, coerced = _coerce_runtime_legal_assignment(
                str(td),
                {"model.layers.0.self_attn.o_proj": "FP8_E5M2"},
                Qwen3_5Profile(),
            )

        self.assertEqual(assignment["model.layers.0.self_attn.o_proj"], "BF16")
        self.assertEqual(coerced, [
            ("model.layers.0.self_attn.o_proj", [128, 5120], "FP8_E5M2")
        ])

    def test_bf16_audit_classifies_allocator_bf16_candidates(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            shard = td / "model-00001-of-00001.safetensors"
            save_file({
                "model.layers.0.self_attn.o_proj.weight": torch.zeros(
                    128, 5120, dtype=torch.bfloat16
                ),
                "model.layers.0.linear_attn.in_proj_a.weight": torch.zeros(
                    48, 5120, dtype=torch.bfloat16
                ),
            }, str(shard))
            with open(td / "model.safetensors.index.json", "w") as f:
                json.dump({
                    "weight_map": {
                        "model.layers.0.self_attn.o_proj.weight": shard.name,
                        "model.layers.0.linear_attn.in_proj_a.weight": shard.name,
                    }
                }, f)

            audit = _bf16_upgrade_audit(
                str(td),
                {
                    "model.layers.0.self_attn.o_proj": "BF16",
                    "model.layers.0.linear_attn.in_proj_a": "BF16",
                },
                set(),
                [("model.layers.0.linear_attn.in_proj_a", [48, 5120])],
                Qwen3_5Profile(),
            )

        reasons = {entry["name"]: entry["reason"] for entry in audit["entries"]}
        self.assertEqual(
            reasons["model.layers.0.linear_attn.in_proj_a"],
            "runtime_coerced_from_mxfp8_e4m3",
        )
        self.assertEqual(
            reasons["model.layers.0.self_attn.o_proj"],
            "allocator_selected_bf16_mxfp8_legal",
        )


class TestDeltaNetFusedSiblingJointScale(unittest.TestCase):
    """Regression for commit e2e0091: Qwen3.6 DeltaNet linear-attention
    fuses `in_proj_qkv + in_proj_z → in_proj_qkvz` (and `in_proj_b +
    in_proj_a → in_proj_ba`) at vLLM load time. The fused packed
    Linear needs a SHARED NVFP4 `weight_global_scale` across those
    siblings. `_compute_layer_joint_nvfp4` is the per-layer helper
    that computes it; if it ever drifts back to per-Linear scales,
    vLLM warns about reduced accuracy from mismatched parallel-layer
    scales."""

    def _build_hybrid_layer(self) -> torch.nn.Module:
        """Two DeltaNet siblings inside a `linear_attn` module stub."""
        class TinyLinearAttn(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.in_proj_qkv = torch.nn.Linear(64, 48, bias=False)
                s.in_proj_z = torch.nn.Linear(64, 16, bias=False)

        class TinyLayer(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.linear_attn = TinyLinearAttn()

        return TinyLayer()

    def test_deltanet_siblings_share_single_joint_scale(self):
        torch.manual_seed(0)
        layer = self._build_hybrid_layer()
        # Give `in_proj_qkv` a larger max-abs so the joint scale is
        # determined by it, not by `in_proj_z`.
        with torch.no_grad():
            layer.linear_attn.in_proj_qkv.weight.mul_(10.0)

        assignment = {
            "model.layers.0.linear_attn.in_proj_qkv": "NVFP4",
            "model.layers.0.linear_attn.in_proj_z": "NVFP4",
        }
        joint = _compute_layer_joint_nvfp4(
            layer, "model.layers.0", assignment, _IdentityProfile())

        # Both siblings must map to NVFP4 and share ONE scale tensor.
        self.assertEqual(
            set(joint),
            {
                "model.layers.0.linear_attn.in_proj_qkv",
                "model.layers.0.linear_attn.in_proj_z",
            },
        )
        scale_qkv = joint["model.layers.0.linear_attn.in_proj_qkv"]
        scale_z = joint["model.layers.0.linear_attn.in_proj_z"]
        # Exact equality — the helper reuses one tensor across the
        # fused group.
        self.assertEqual(scale_qkv.item(), scale_z.item())

        # The shared scale must equal the max of the per-sibling
        # natural scales (commit e2e0091 regression).
        from prismaquant.export_native_compressed import (
            compute_nvfp4_global_real,
        )
        nat_qkv = compute_nvfp4_global_real(
            layer.linear_attn.in_proj_qkv.weight.float(), group_size=16)
        nat_z = compute_nvfp4_global_real(
            layer.linear_attn.in_proj_z.weight.float(), group_size=16)
        self.assertAlmostEqual(
            scale_qkv.item(), max(nat_qkv.item(), nat_z.item()),
            places=5,
        )

    def test_mixed_format_siblings_do_not_emit_joint_scale(self):
        """If only one sibling is NVFP4 (and the other MXFP8/BF16),
        there's no fused packed Linear to share a scale across — the
        helper must skip the group."""
        torch.manual_seed(0)
        layer = self._build_hybrid_layer()
        assignment = {
            "model.layers.0.linear_attn.in_proj_qkv": "NVFP4",
            "model.layers.0.linear_attn.in_proj_z": "MXFP8",
        }
        joint = _compute_layer_joint_nvfp4(
            layer, "model.layers.0", assignment, _IdentityProfile())
        self.assertEqual(joint, {},
                         "mixed-format sibling group must not emit a joint scale")


class TestComputeExtraIgnorePerExpertSiblings(unittest.TestCase):
    """Regression for commit dab2473: per-expert MoE source tensors
    (e.g. `model.layers.0.mlp.experts.3.gate_proj`) are covered by the
    packed parent (`...mlp.experts.gate_up_proj`) at compressed-tensors
    load time. If the helper accidentally adds them to `extra_ignore`,
    vLLM marks the FusedMoE layer as un-quantized, the NVFP4 scale
    params never get registered, and load crashes."""

    def test_per_expert_siblings_excluded_when_parent_quantized(self):
        # Assignment includes the packed parent — both per-expert
        # source keys must be omitted from extra_ignore.
        assignment = {
            "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
            "model.layers.0.mlp.experts.down_proj": "NVFP4",
        }
        source_iter = [
            # Per-expert source tensors (2D) — must NOT appear in extra_ignore.
            ("model.layers.0.mlp.experts.0.gate_proj.weight", [512, 1024]),
            ("model.layers.0.mlp.experts.0.up_proj.weight", [512, 1024]),
            ("model.layers.0.mlp.experts.0.down_proj.weight", [1024, 512]),
            ("model.layers.0.mlp.experts.3.gate_proj.weight", [512, 1024]),
            ("model.layers.0.mlp.experts.3.up_proj.weight", [512, 1024]),
            ("model.layers.0.mlp.experts.3.down_proj.weight", [1024, 512]),
            # An unrelated 2D Linear the recipe doesn't cover — this
            # SHOULD end up in extra_ignore.
            ("model.visual.merger.weight", [768, 768]),
            # A non-2D tensor — always skipped regardless of coverage.
            ("model.layers.0.mlp.gate.weight", [128]),
        ]
        extra = compute_extra_ignore(source_iter, assignment)

        for name in [
            "model.layers.0.mlp.experts.0.gate_proj",
            "model.layers.0.mlp.experts.0.up_proj",
            "model.layers.0.mlp.experts.0.down_proj",
            "model.layers.0.mlp.experts.3.gate_proj",
            "model.layers.0.mlp.experts.3.up_proj",
            "model.layers.0.mlp.experts.3.down_proj",
        ]:
            self.assertNotIn(
                name, extra,
                f"per-expert sibling {name} must not be in extra_ignore "
                "when the packed parent is in the assignment "
                "(regression for commit dab2473)",
            )
        self.assertIn("model.visual.merger", extra)

    def test_per_expert_siblings_included_when_parent_missing(self):
        """Sanity: without the parent in the assignment, per-expert
        tensors DO end up in extra_ignore (they would be un-quantized
        on the vLLM side, so compressed-tensors needs to skip them)."""
        assignment: dict[str, str] = {
            # intentionally missing the packed parents
        }
        source_iter = [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", [512, 1024]),
            ("model.layers.0.mlp.experts.0.down_proj.weight", [1024, 512]),
        ]
        extra = compute_extra_ignore(source_iter, assignment)
        self.assertIn("model.layers.0.mlp.experts.0.gate_proj", extra)
        self.assertIn("model.layers.0.mlp.experts.0.down_proj", extra)

    def test_language_model_prefix_remap(self):
        """Multimodal checkpoints prefix body tensors with
        `model.language_model.*` on disk but the recipe uses
        `model.*` — the helper must remap before the coverage check."""
        assignment = {
            # recipe-side name (no language_model. infix)
            "model.layers.0.self_attn.q_proj": "NVFP4",
        }
        source_iter = [
            # disk-side name with language_model. infix
            ("model.language_model.layers.0.self_attn.q_proj.weight",
             [1024, 1024]),
            # unrelated 2D the recipe doesn't cover
            ("model.language_model.layers.0.mlp.shared_expert_gate.weight",
             [32, 1024]),
        ]
        extra = compute_extra_ignore(source_iter, assignment)
        self.assertNotIn(
            "model.language_model.layers.0.self_attn.q_proj", extra)
        self.assertIn(
            "model.language_model.layers.0.mlp.shared_expert_gate", extra)

    def test_profile_remap_excludes_gemma_per_expert_siblings(self):
        """Gemma inserts `.moe.experts` in multimodal source names while
        the recipe uses `.experts`; profile naming must drive coverage."""
        assignment = {
            "model.layers.0.experts.gate_up_proj": "NVFP4",
            "model.layers.0.experts.down_proj": "NVFP4",
        }
        source_iter = [
            (
                "model.language_model.layers.0.moe.experts.0.gate_proj.weight",
                [512, 1024],
            ),
            (
                "model.language_model.layers.0.moe.experts.0.down_proj.weight",
                [1024, 512],
            ),
            (
                "model.language_model.layers.0.moe.shared_gate.weight",
                [32, 1024],
            ),
        ]
        extra = compute_extra_ignore(source_iter, assignment, Gemma4Profile())

        self.assertNotIn(
            "model.language_model.layers.0.moe.experts.0.gate_proj",
            extra,
        )
        self.assertNotIn(
            "model.language_model.layers.0.moe.experts.0.down_proj",
            extra,
        )
        self.assertIn("model.language_model.layers.0.moe.shared_gate", extra)

    def test_profile_decomposition_excludes_custom_per_expert_siblings(self):
        assignment = {
            "model.layers.0.mlp.experts.w13": "NVFP4",
            "model.layers.0.mlp.experts.w2": "NVFP4",
        }
        source_iter = [
            ("model.layers.0.mlp.experts.0.w1_proj.weight", [512, 1024]),
            ("model.layers.0.mlp.experts.0.w3_proj.weight", [512, 1024]),
            ("model.layers.0.mlp.experts.0.w2.weight", [1024, 512]),
            ("model.layers.0.mlp.experts.0.side.weight", [512, 1024]),
        ]
        extra = compute_extra_ignore(
            source_iter,
            assignment,
            _CustomPackedProfile(),
        )

        self.assertNotIn("model.layers.0.mlp.experts.0.w1_proj", extra)
        self.assertNotIn("model.layers.0.mlp.experts.0.w3_proj", extra)
        self.assertNotIn("model.layers.0.mlp.experts.0.w2", extra)
        self.assertIn("model.layers.0.mlp.experts.0.side", extra)


if __name__ == "__main__":
    unittest.main()


class TestNvfp4InputGlobalScale(unittest.TestCase):
    """Per-layer input_global_scale calibration from cached activations.
    
    `compute_nvfp4_input_global_scale(activations)` returns FP4_MAX/max_abs
    so scaled activations fit [-6, 6]. Zero/negative max-abs falls back to
    the default."""

    def test_max_abs_scales_to_fp4_range(self):
        import torch
        from prismaquant.export_native_compressed import (
            compute_nvfp4_input_global_scale, _FP4_E2M1_MAX,
        )
        acts = torch.tensor([0.0, 1.5, -3.0, 2.0])
        s = compute_nvfp4_input_global_scale(acts)
        # max_abs=3.0, scale=6/3=2.0 → scaled activations in [-6, 6]
        self.assertAlmostEqual(s, _FP4_E2M1_MAX / 3.0, places=5)

    def test_degenerate_all_zero_falls_back(self):
        import torch
        from prismaquant.export_native_compressed import (
            compute_nvfp4_input_global_scale, DEFAULT_INPUT_GLOBAL_SCALE,
        )
        acts = torch.zeros(100)
        s = compute_nvfp4_input_global_scale(acts)
        self.assertEqual(s, DEFAULT_INPUT_GLOBAL_SCALE)

    def test_quantize_2d_reads_override(self):
        import torch
        from prismaquant.export_native_compressed import _quantize_2d
        weight = torch.randn(32, 32)
        out = _quantize_2d(weight, "NVFP4",
                           input_global_scale_override=2.5)
        self.assertAlmostEqual(
            float(out["input_global_scale"].item()), 2.5, places=4)

    def test_quantize_2d_uses_global_cache_when_named(self):
        import torch
        import prismaquant.export_native_compressed as m
        weight = torch.randn(32, 32)
        # Save/restore the module-level cache
        saved = m._INPUT_GLOBAL_SCALES
        try:
            m._INPUT_GLOBAL_SCALES = {"foo.bar.q_proj": 3.14}
            out = _quantize_2d = m._quantize_2d(
                weight, "NVFP4", linear_name="foo.bar.q_proj"
            )
            self.assertAlmostEqual(
                float(out["input_global_scale"].item()), 3.14, places=4)
        finally:
            m._INPUT_GLOBAL_SCALES = saved


class TestActivationAwarePasses(unittest.TestCase):
    """GPTQ OBS, scale sweep, and activation-weighted rounding are the
    calibration-aware passes wired into
    `_quantize_2d`'s NVFP4 path. Each has a per-pass unit test plus a
    composed integration test on a synthetic [out, in] linear with a
    heavily imbalanced activation distribution."""

    def setUp(self):
        import torch
        torch.manual_seed(42)

    def test_activation_matrix_explicit_threshold_overrides_quantile(self):
        import os
        import torch
        import prismaquant.export_native_compressed as m

        saved = os.environ.get("PRISMAQUANT_ACT_CLIP_QUANTILE")
        os.environ["PRISMAQUANT_ACT_CLIP_QUANTILE"] = "0.5"
        try:
            x = torch.tensor([[1.0, -2.0, 100.0, -100.0]])
            out = m._activation_matrix_for_gptq(
                x,
                4,
                clip_threshold=10.0,
            )
        finally:
            if saved is None:
                os.environ.pop("PRISMAQUANT_ACT_CLIP_QUANTILE", None)
            else:
                os.environ["PRISMAQUANT_ACT_CLIP_QUANTILE"] = saved

        expected = torch.tensor([[1.0, -2.0, 10.0, -10.0]])
        self.assertTrue(torch.equal(out, expected))

    def test_activation_matrix_applies_fisher_row_weights(self):
        import torch
        import prismaquant.export_native_compressed as m

        x = torch.ones(2, 2)
        out = m._activation_matrix_for_gptq(
            x,
            2,
            clip_quantile=0.0,
            row_weights=torch.tensor([0.0, 2.0]),
        )

        self.assertTrue(torch.allclose(out[0], torch.zeros(2)))
        self.assertTrue(torch.allclose(out[1], torch.full((2,), 2 ** 0.5)))

    def test_mxfp8_scale_sweep_is_no_worse_than_baseline(self):
        import os
        import torch
        import prismaquant.export_native_compressed as m

        torch.manual_seed(7)
        W = torch.randn(16, 64) * 0.2
        X = torch.randn(32, 64)
        q, s = m.quantize_dequantize_mxfp8(W, group_size=32)
        baseline = m._mxfp8_dequantize_grouped(
            q.reshape(16, 2, 32),
            s,
        ).reshape_as(W)
        saved = os.environ.pop("PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS", None)
        try:
            _, _, default = m._mxfp8_scale_sweep_quantize(W, X, group_size=32)
            self.assertTrue(torch.equal(default, baseline))
            os.environ["PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS"] = "-2,-1,0,1,2"
            _, _, swept = m._mxfp8_scale_sweep_quantize(W, X, group_size=32)
        finally:
            if saved is None:
                os.environ.pop("PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS", None)
            else:
                os.environ["PRISMAQUANT_MXFP8_SCALE_SWEEP_SHIFTS"] = saved

        imp = X.pow(2).mean(dim=0).reshape(1, 2, 32)
        base_err = ((W.reshape(16, 2, 32) - baseline.reshape(16, 2, 32)).pow(2) * imp).sum()
        swept_err = ((W.reshape(16, 2, 32) - swept.reshape(16, 2, 32)).pow(2) * imp).sum()
        self.assertLessEqual(float(swept_err), float(base_err) + 1e-6)

    def test_fp8_scale_sweep_is_no_worse_than_baseline(self):
        import torch
        import prismaquant.export_native_compressed as m

        torch.manual_seed(11)
        W = torch.randn(16, 64) * 0.2
        X = torch.randn(32, 64)
        q, s = m.quantize_dequantize_fp8_dynamic(W)
        baseline = q.to(torch.float32) * s.to(torch.float32)
        _, _, swept = m._fp8_dynamic_scale_sweep_quantize(W, X)

        imp = X.pow(2).mean(dim=0).reshape(1, 64)
        base_err = ((W - baseline).pow(2) * imp).sum()
        swept_err = ((W - swept).pow(2) * imp).sum()
        self.assertLessEqual(float(swept_err), float(base_err) + 1e-6)

    def _decode_nvfp4(self, wp, ws, wg):
        import torch
        from prismaquant.export_native_compressed import (
            FLOAT_TO_E2M1,
        )
        rows = wp.shape[0]
        cols = wp.shape[1] * 2
        cb = torch.tensor(FLOAT_TO_E2M1, dtype=torch.float32)
        lo = (wp & 0xF).long()
        hi = ((wp >> 4) & 0xF).long()
        idx = torch.stack([lo, hi], dim=-1).reshape(rows, cols)
        abs_idx = idx & 0x7
        sign = -((idx >> 3).to(torch.float32) * 2 - 1)
        vals = sign * cb[abs_idx]
        fp8_per_col = (
            ws.float().unsqueeze(-1)
            .expand(-1, -1, cols // ws.shape[1])
            .reshape(rows, cols)
        )
        global_real = 1.0 / wg.item()
        return vals * fp8_per_col * global_real

    def test_gptq_obs_rounding_returns_grid_aligned(self):
        """After GPTQ, every weight should round to some point on the
        NVFP4 grid — repacking should not change the dequantized value
        by more than one grid step (allowing for global-scale adjustments)."""
        import torch
        from prismaquant.export_native_compressed import (
            _gptq_obs_rounding_nvfp4, quantize_dequantize_nvfp4,
        )
        W = torch.randn(16, 32) * 0.2
        X = torch.randn(200, 32) * 0.5
        W_gptq = _gptq_obs_rounding_nvfp4(W, X, group_size=16)
        self.assertEqual(W_gptq.shape, W.shape)
        # Re-pack the GPTQ output — it should round-trip (each weight
        # already sits on the grid, so quant+dequant is approximately
        # idempotent up to the per-group outer scale math).
        wp, ws, wg = quantize_dequantize_nvfp4(W_gptq)
        dq = self._decode_nvfp4(wp, ws, wg)
        # GPTQ output packing re-quant MSE must be O(grid step²).
        mse = (W_gptq - dq).pow(2).mean().item()
        self.assertLess(mse, 1e-2,
                        f"GPTQ output not grid-aligned, mse={mse:.3e}")

    def test_joint_mse_scale_rule_subsumes_four_over_six(self):
        import torch
        import prismaquant.export_native_compressed as m

        torch.manual_seed(23)
        grouped = torch.randn(8, 4, 16, dtype=torch.float32) * 0.3
        scale_f6 = m._select_nvfp4_group_scales(
            grouped,
            scale_rule=m.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
        )
        scale_joint = m._select_nvfp4_group_scales(
            grouped,
            scale_rule=m.NVFP4_SCALE_RULE_JOINT_MSE,
        )
        mse_f6 = m._nvfp4_mse_for_group_scale(grouped, scale_f6)
        mse_joint = m._nvfp4_mse_for_group_scale(grouped, scale_joint)

        self.assertTrue(torch.all(mse_joint <= mse_f6 + 1e-7))

    def test_gptq_lift_static_order_and_joint_scale_opt_grid_aligned(self):
        import torch
        import prismaquant.export_native_compressed as m

        torch.manual_seed(29)
        W = torch.randn(16, 32) * 0.2
        X = torch.randn(160, 32) * 0.5
        X[:, :4] *= 8.0
        prev = m._NVFP4_SCALE_RULE
        try:
            m._NVFP4_SCALE_RULE = m.NVFP4_SCALE_RULE_JOINT_MSE
            W_gptq = m._gptq_obs_rounding_nvfp4(
                W,
                X,
                group_size=16,
                static_act_order=True,
                joint_scale_opt=True,
            )
            wp, ws, wg = m.quantize_dequantize_nvfp4(W_gptq)
            dq = self._decode_nvfp4(wp, ws, wg)
        finally:
            m._NVFP4_SCALE_RULE = prev

        self.assertEqual(W_gptq.shape, W.shape)
        self.assertLess((W_gptq - dq).pow(2).mean().item(), 1e-2)

    def test_gptq_cholesky_failure_falls_back_to_rtn_not_original(self):
        """A failed GPTQ solve must still return a valid NVFP4 render.

        Returning the original BF16 weight makes local output-MSE gates see
        impossible zero error in compute_only/production-cache paths.
        """
        import torch
        from unittest import mock
        from prismaquant.export_native_compressed import (
            _gptq_obs_rounding_nvfp4,
            _gptq_obs_rounding_nvfp4_swept,
            _rtn_dequant_nvfp4,
        )

        torch.manual_seed(911)
        W = torch.randn(16, 32) * 0.3
        X = torch.randn(24, 32)
        W_rtn = _rtn_dequant_nvfp4(W, group_size=16)
        with mock.patch("torch.linalg.cholesky", side_effect=RuntimeError("boom")):
            W_failed = _gptq_obs_rounding_nvfp4(W, X, group_size=16)
            W_swept = _gptq_obs_rounding_nvfp4_swept(W, X, group_size=16)

        torch.testing.assert_close(W_failed, W_rtn)
        torch.testing.assert_close(W_swept, W_rtn)
        self.assertGreater(float((W - W_failed).pow(2).mean().item()), 0.0)

    def test_composed_passes_reduce_output_space_error_vs_rtn(self):
        """Integration test: synthetic linear + imbalanced activations.
        Running `_quantize_2d` with GPTQ enabled should give no worse
        activation-weighted output-space MSE than pure RTN."""
        import torch
        from prismaquant.export_native_compressed import (
            _quantize_2d,
        )
        torch.manual_seed(7)
        out_f, in_f = 64, 128
        # Weight with some high-magnitude rows to stress quantization.
        W = torch.randn(out_f, in_f) * 0.15
        W[:, :8] *= 5.0                                  # bigger weights in first 8 cols
        # Heavily imbalanced activations: first 8 columns are huge,
        # rest are small.
        X = torch.randn(512, in_f) * 0.1
        X[:, :8] *= 20.0
        # Reference BF16 output.
        ref = (W @ X.t()).float()

        # Pure RTN.
        out_rtn = _quantize_2d(W, "NVFP4", linear_name=None)
        W_rtn = self._decode_nvfp4(
            out_rtn["weight_packed"], out_rtn["weight_scale"],
            out_rtn["weight_global_scale"],
        )
        # GPTQ on, activations passed explicitly.
        out_aa = _quantize_2d(
            W, "NVFP4",
            gptq_enabled=True,
            cached_activations=X,
        )
        W_aa = self._decode_nvfp4(
            out_aa["weight_packed"], out_aa["weight_scale"],
            out_aa["weight_global_scale"],
        )
        err_rtn = (ref - (W_rtn @ X.t())).pow(2).mean().item()
        err_aa = (ref - (W_aa @ X.t())).pow(2).mean().item()
        # The do-no-harm gate (PRISMAQUANT_DO_NO_HARM=1, default on)
        # reverts to RTN when act-aware passes don't improve, so the
        # invariant is `err_aa <= err_rtn`, not strictly less.
        self.assertLessEqual(
            err_aa, err_rtn,
            f"act-aware passes increased output error: "
            f"rtn={err_rtn:.4e} aa={err_aa:.4e}",
        )

    def test_act_aware_flags_module_default_off(self):
        """The module-level `_ACT_AWARE_FLAGS` defaults to all False so
        callers that don't touch main() get vanilla RTN behavior."""
        from prismaquant.export_native_compressed import (
            _ACT_AWARE_FLAGS,
        )
        self.assertFalse(_ACT_AWARE_FLAGS["gptq"])
        self.assertFalse(_ACT_AWARE_FLAGS["static_act_order"])
        self.assertFalse(_ACT_AWARE_FLAGS["joint_scale_opt"])

    def test_quantize_2d_picks_up_module_flags(self):
        """When `_ACT_AWARE_FLAGS` is set, GPTQ is selected by `_quantize_2d`
        based on the module-level flag bundle."""
        import torch
        import prismaquant.export_native_compressed as m
        torch.manual_seed(11)
        W = torch.randn(32, 64) * 0.2
        # Imbalanced activations so GPTQ's block-wise error propagation
        # has something to work with — uniform X yields the same per-
        # block scales across blocks and GPTQ's update becomes a near-
        # no-op vs RTN.
        X = torch.randn(256, 64) * 0.1
        X[:, :16] *= 10.0
        saved_flags = dict(m._ACT_AWARE_FLAGS)
        saved_cache = m._CACHED_ACTIVATIONS
        # Disable do-no-harm gate for this test: we're verifying the
        # ACT-AWARE PASS dispatch, not the gate. The gate can revert
        # to RTN when the random fixture happens not to benefit from
        # GPTQ — masking the dispatch test.
        import os
        saved_dnh = os.environ.get("PRISMAQUANT_DO_NO_HARM")
        os.environ["PRISMAQUANT_DO_NO_HARM"] = "0"
        try:
            m._ACT_AWARE_FLAGS.update({
                "gptq": True,
                "static_act_order": False, "joint_scale_opt": False,
            })
            m._CACHED_ACTIVATIONS = {"demo.linear": X}
            out_with = m._quantize_2d(
                W, "NVFP4", linear_name="demo.linear",
            )
            m._ACT_AWARE_FLAGS.update({
                "gptq": False,
                "static_act_order": False, "joint_scale_opt": False,
            })
            out_without = m._quantize_2d(
                W, "NVFP4", linear_name="demo.linear",
            )
        finally:
            m._ACT_AWARE_FLAGS.clear()
            m._ACT_AWARE_FLAGS.update(saved_flags)
            m._CACHED_ACTIVATIONS = saved_cache
            if saved_dnh is None:
                os.environ.pop("PRISMAQUANT_DO_NO_HARM", None)
            else:
                os.environ["PRISMAQUANT_DO_NO_HARM"] = saved_dnh
        # The weight_packed should differ because GPTQ reshapes the
        # weight via block-wise error propagation.
        self.assertFalse(
            torch.equal(out_with["weight_packed"],
                        out_without["weight_packed"]),
            "act-aware flags had no effect on output",
        )

    def test_quantize_2d_threads_lift_gptq_flags(self):
        import os
        import torch
        import prismaquant.export_native_compressed as m

        seen = {}

        def fake_gptq(weight, activations, **kwargs):
            seen.update(kwargs)
            return weight.to(torch.float32)

        W = torch.randn(32, 32)
        X = torch.randn(16, 32)
        saved_gptq = m._gptq_obs_rounding_nvfp4_swept
        saved_dnh = os.environ.get("PRISMAQUANT_DO_NO_HARM")
        os.environ["PRISMAQUANT_DO_NO_HARM"] = "0"
        try:
            m._gptq_obs_rounding_nvfp4_swept = fake_gptq
            m._quantize_2d(
                W,
                "NVFP4",
                gptq_enabled=True,
                static_act_order_enabled=True,
                joint_scale_opt_enabled=True,
                cached_activations=X,
                compute_only=True,
            )
        finally:
            m._gptq_obs_rounding_nvfp4_swept = saved_gptq
            if saved_dnh is None:
                os.environ.pop("PRISMAQUANT_DO_NO_HARM", None)
            else:
                os.environ["PRISMAQUANT_DO_NO_HARM"] = saved_dnh

        self.assertIs(seen.get("static_act_order"), True)
        self.assertIs(seen.get("joint_scale_opt"), True)

    def test_post_nonlinearity_names_do_not_skip_gptq_or_scale_sweep(self):
        """GPTQ and scale_sweep are still valid on post-nonlinearity
        readers such as down_proj/o_proj."""
        import os
        import torch
        import prismaquant.export_native_compressed as m

        calls = []

        def fake_gptq(weight, activations, **kwargs):
            calls.append("gptq")
            return weight.to(torch.float32)

        def fake_scale_sweep(weight, activations, **kwargs):
            calls.append("scale_sweep")
            return weight.to(torch.float32)

        W = torch.randn(32, 32)
        X = torch.randn(16, 32)
        saved_gptq = m._gptq_obs_rounding_nvfp4_swept
        saved_scale_sweep = m._scale_sweep_nvfp4
        saved_dnh = os.environ.get("PRISMAQUANT_DO_NO_HARM")
        os.environ["PRISMAQUANT_DO_NO_HARM"] = "0"
        try:
            m._gptq_obs_rounding_nvfp4_swept = fake_gptq
            m._scale_sweep_nvfp4 = fake_scale_sweep
            m._quantize_2d(
                W,
                "NVFP4",
                linear_name="model.layers.0.mlp.down_proj",
                gptq_enabled=True,
                scale_sweep_enabled=True,
                cached_activations=X,
                compute_only=True,
            )
        finally:
            m._gptq_obs_rounding_nvfp4_swept = saved_gptq
            m._scale_sweep_nvfp4 = saved_scale_sweep
            if saved_dnh is None:
                os.environ.pop("PRISMAQUANT_DO_NO_HARM", None)
            else:
                os.environ["PRISMAQUANT_DO_NO_HARM"] = saved_dnh

        self.assertEqual(calls, ["gptq", "scale_sweep"])
