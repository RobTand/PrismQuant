import json
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    SOURCE_BPP_EXCEEDED_REASON,
    _scan_source_dtype_manifest,
    cost_entry_source,
)
from prismaquant.allocator import (
    Candidate,
    build_candidates,
    filter_candidates_for_profile,
)
from prismaquant.model_profiles.qwen3_5 import Qwen3_5Profile
from prismaquant.sensitivity_probe import discover_moe_structure


class TestPrismaQuantFormatRegistry(unittest.TestCase):
    def test_block_formats_have_expected_shape_aware_bits(self):
        shape = (128, 128)
        self.assertAlmostEqual(fr.get_format("NVFP4").effective_bits_for_shape(shape), 4.5)
        self.assertAlmostEqual(fr.get_format("MXFP4").effective_bits_for_shape(shape), 4.25)
        self.assertAlmostEqual(fr.get_format("MXFP8").effective_bits_for_shape(shape), 8.25)
        self.assertAlmostEqual(fr.get_format("FP8_SOURCE").effective_bits_for_shape(shape), 8.001953125)
        self.assertAlmostEqual(fr.get_format("BF16").effective_bits_for_shape(shape), 16.0)

    def test_mxfp8_short_name_is_input_alias_only(self):
        self.assertEqual(fr.canonical_format_name("MXFP8"), "MXFP8_E4M3")
        self.assertEqual(fr.get_format("MXFP8").name, "MXFP8_E4M3")
        self.assertIn("MXFP8", fr.aliases_for("MXFP8_E4M3"))

    def test_fp8_dynamic_alias_targets_vllm_dynamic_fp8(self):
        self.assertEqual(fr.canonical_format_name("FP8_DYNAMIC"), "FP8_E4M3")
        self.assertEqual(fr.canonical_format_name("FP8"), "FP8_E4M3")
        self.assertEqual(fr.canonical_format_name("fp8_dynamic"), "FP8_E4M3")
        self.assertEqual(fr.get_format("FP8_DYNAMIC").name, "FP8_E4M3")
        self.assertEqual(fr.get_format("fp8_dynamic").name, "FP8_E4M3")
        self.assertIn("FP8_DYNAMIC", fr.aliases_for("FP8_E4M3"))

    def test_low_bit_custom_kernel_formats_are_not_registered(self):
        for name in ("INT2", "INT3", "NVFP3", "NOT_A_REAL_FORMAT"):
            with self.assertRaises(KeyError):
                fr.get_format(name)

    def test_source_fp8_uses_2d_scale_blocks(self):
        spec = fr.get_format("FP8_SOURCE")
        shape = (256, 256)
        expected_bytes = 256 * 256 + 4 * 4  # fp8 weights + four fp32 128x128 scales

        self.assertAlmostEqual(spec.effective_bits, 8.001953125)
        self.assertEqual(spec.scale_count_for_shape(shape), 4)
        self.assertEqual(spec.memory_bytes_for_shape(shape), expected_bytes)
        self.assertAlmostEqual(
            spec.effective_bits_for_shape(shape),
            8.0 * expected_bytes / (256 * 256),
        )

    def test_per_channel_formats_use_row_scale_count(self):
        shape = (5, 7)
        spec = fr.get_format("FP8_E4M3")
        expected_bytes = 5 * 7 + 5 * 4  # one byte per weight, fp32 row scales
        self.assertEqual(spec.scale_count_for_shape(shape), 5)
        self.assertEqual(spec.memory_bytes_for_shape(shape), expected_bytes)
        self.assertAlmostEqual(spec.effective_bits_for_shape(shape), 8.0 * expected_bytes / 35.0)

    def test_activation_quantization_changes_native_a4_a8_formats(self):
        x = torch.tensor([[0.13, -0.51, 1.77, -3.25]], dtype=torch.float32)
        for fmt in ("NVFP4", "MXFP8", "FP8_E4M3"):
            y = fr.get_format(fmt).activation_quantize_dequantize(x.clone())
            self.assertFalse(torch.equal(x, y), msg=fmt)

    def test_activation_quantization_skips_a16_formats(self):
        x = torch.tensor([[0.13, -0.51, 1.77, -3.25]], dtype=torch.float32)
        for fmt in ("MXFP8A16", "NVFP4A16", "BF16"):
            y = fr.get_format(fmt).activation_quantize_dequantize(x.clone())
            self.assertTrue(torch.equal(x, y), msg=fmt)


class TestPrismaQuantAllocatorMath(unittest.TestCase):
    def test_build_candidates_uses_shape_aware_bits(self):
        # Predicted Δloss = 0.5 · h_trace · output_mse  (joint W·X
        # perturbation under diagonal-Fisher curvature; output_mse is
        # the cost step's W*A*-aware error metric).
        stats = {
            "layer.weight": {
                "h_trace": 2.0,
                "out_features": 5,
                "in_features": 7,
                "n_params": 35,
            }
        }
        costs = {
            "layer.weight": {
                "FP8_E4M3": {"weight_mse": 0.10, "output_mse": 0.25},
                "BF16": {"weight_mse": 0.0, "output_mse": 0.0},
            }
        }
        cands = build_candidates(stats, costs, [fr.get_format("FP8_E4M3"), fr.get_format("BF16")])
        by_fmt = {cand.fmt: cand for cand in cands["layer.weight"]}
        self.assertAlmostEqual(
            by_fmt["FP8_E4M3"].bits_per_param,
            fr.get_format("FP8_E4M3").effective_bits_for_shape((5, 7)),
        )
        self.assertEqual(
            by_fmt["FP8_E4M3"].memory_bytes,
            fr.get_format("FP8_E4M3").memory_bytes_for_shape((5, 7)),
        )
        self.assertAlmostEqual(by_fmt["FP8_E4M3"].predicted_dloss, 0.5 * 2.0 * 0.25)
        self.assertAlmostEqual(by_fmt["BF16"].predicted_dloss, 0.0)

    def test_build_candidates_can_use_fisher_output_mse(self):
        stats = {
            "layer.weight": {
                "h_trace": 2.0,
                "out_features": 5,
                "in_features": 7,
                "n_params": 35,
            }
        }
        costs = {
            "layer.weight": {
                "FP8_E4M3": {
                    "weight_mse": 0.10,
                    "output_mse": 0.25,
                    "fisher_output_mse": 0.05,
                },
            }
        }

        with mock.patch.dict("os.environ", {}, clear=True):
            cands = build_candidates(stats, costs, [fr.get_format("FP8_E4M3")])
        self.assertAlmostEqual(cands["layer.weight"][0].predicted_dloss, 0.25)

        with mock.patch.dict(
            "os.environ",
            {"PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR": "1"},
            clear=True,
        ):
            cands = build_candidates(stats, costs, [fr.get_format("FP8_E4M3")])
            self.assertAlmostEqual(cands["layer.weight"][0].predicted_dloss, 0.05)

    def test_cost_entry_source_reports_authoritative_fallback(self):
        stats = {"h_trace": 2.0, "num_experts": 1}

        self.assertEqual(
            cost_entry_source(stats, {"output_mse": 0.25}),
            "output_mse",
        )
        self.assertEqual(
            cost_entry_source(
                stats,
                {
                    "output_mse": 0.0,
                    "output_mse_measured": False,
                    "predicted_dloss": 7.0,
                },
            ),
            "predicted_dloss",
        )
        self.assertEqual(
            cost_entry_source(
                stats,
                {"predicted_dloss": 3.0, "cost_source": "grouped_kl_share"},
            ),
            "grouped_kl_share",
        )

    def test_build_candidates_rejects_mxfp8_above_source_fp8(self):
        stats = {
            "layer.weight": {
                "h_trace": 2.0,
                "out_features": 128,
                "in_features": 128,
                "n_params": 128 * 128,
            }
        }
        costs = {
            "layer.weight": {
                "FP8_SOURCE": {"weight_mse": 0.0},
                "MXFP8": {"weight_mse": 0.01},
            }
        }
        masks = []
        cands = build_candidates(
            stats,
            costs,
            [fr.get_format("FP8_SOURCE"), fr.get_format("MXFP8")],
            source_manifest={"layer.weight": "fp8"},
            mask_records=masks,
        )
        by_fmt = {cand.fmt: cand for cand in cands["layer.weight"]}

        self.assertNotIn("MXFP8_E4M3", by_fmt)
        self.assertEqual(by_fmt["FP8_SOURCE"].memory_bytes, 128 * 128 + 4)
        self.assertAlmostEqual(by_fmt["FP8_SOURCE"].bits_per_param, 8.001953125)
        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0]["reason"], SOURCE_BPP_EXCEEDED_REASON)
        self.assertAlmostEqual(masks[0]["source_bpp"], 8.001953125)
        self.assertAlmostEqual(masks[0]["candidate_bpp"], 8.25)

    def test_source_dtype_manifest_uses_profile_name_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            payload = {
                "weight_map": {
                    "model.language_model.layers.0.mlp.gate_proj.weight": "s0",
                    "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv": "s0",
                    "model.visual.blocks.0.attn.proj.weight": "s0",
                }
            }
            with open(f"{td}/model.safetensors.index.json", "w") as f:
                json.dump(payload, f)

            manifest = _scan_source_dtype_manifest(td, Qwen3_5Profile())

        self.assertEqual(
            manifest,
            {"model.layers.0.mlp.gate_proj": "unknown"},
        )

    def test_source_dtype_manifest_reads_tensor_dtype_and_scale_sidecars(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as td:
            save_file(
                {
                    "bf.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
                    "fp16.weight": torch.zeros(2, 2, dtype=torch.float16),
                    "fp32.weight": torch.zeros(2, 2, dtype=torch.float32),
                    "ct.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
                    "ct.weight_scale": torch.ones(2, 1, dtype=torch.float32),
                },
                f"{td}/model.safetensors",
            )

            manifest = _scan_source_dtype_manifest(td)

        self.assertEqual(manifest["bf"], "bf16")
        self.assertEqual(manifest["fp16"], "f16")
        self.assertEqual(manifest["fp32"], "f32")
        self.assertEqual(manifest["ct"], "fp8")

    def test_build_candidates_rejects_missing_source_rate_owner(self):
        stats = {
            "layer": {
                "h_trace": 2.0,
                "out_features": 128,
                "in_features": 128,
                "n_params": 128 * 128,
            }
        }
        costs = {
            "layer": {
                "BF16": {"weight_mse": 0.0, "predicted_dloss": 0.0},
                "NVFP4": {"weight_mse": 0.01, "predicted_dloss": 0.01},
            }
        }

        with self.assertRaisesRegex(
            ValueError, "source-bpp legality cannot be established"
        ):
            build_candidates(
                stats,
                costs,
                [fr.get_format("BF16"), fr.get_format("NVFP4")],
                source_manifest={},
            )

    def test_source_dtype_manifest_uses_profile_fp8_scale_pairs(self):
        class _ScaleProfile:
            def checkpoint_to_live_name(self, key, *, multimodal=False):
                del multimodal
                if key == "ck.foo.weight":
                    return "live.foo.weight"
                return None

            def live_to_recipe_name(self, qname):
                return qname.replace("live.", "recipe.", 1)

            def fp8_scale_pairs(self, model_path):
                del model_path
                return {"live.foo.weight": ("s0", "ck.foo.scale")}

        with tempfile.TemporaryDirectory() as td:
            manifest = _scan_source_dtype_manifest(td, _ScaleProfile())

        self.assertEqual(manifest, {"recipe.foo": "fp8"})

    def test_build_candidates_applies_calibrated_gains(self):
        stats = {
            "layer.weight": {
                "h_trace": 2.0,
                "out_features": 128,
                "in_features": 128,
                "n_params": 128 * 128,
            }
        }
        costs = {
            "layer.weight": {
                "NVFP4": {"weight_mse": 0.10},
                "MXFP8": {"weight_mse": 0.02},
            }
        }
        # Without calibration: NVFP4 = 0.10, MXFP8 = 0.02 (per-element MSE).
        # With α_NVFP4=2 the NVFP4 candidate's predicted Δloss should double.
        cands = build_candidates(
            stats, costs,
            [fr.get_format("NVFP4"), fr.get_format("MXFP8")],
            calibrated_gains={"NVFP4": 2.0, "MXFP8": 1.0},
        )
        by_fmt = {c.fmt: c for c in cands["layer.weight"]}
        self.assertAlmostEqual(by_fmt["NVFP4"].predicted_dloss, 0.5 * 2.0 * 0.10 * 2.0)
        self.assertAlmostEqual(
            by_fmt["MXFP8_E4M3"].predicted_dloss,
            0.5 * 2.0 * 0.02 * 1.0,
        )

    def test_build_candidates_ignores_unmeasured_packed_output_mse(self):
        stats = {
            "model.layers.0.mlp.experts.gate_up_proj": {
                "h_trace": 3.0,
                "out_features": 8,
                "in_features": 16,
                "n_params": 2 * 8 * 16,
                "num_experts": 2,
                "_packed_experts_module": "model.layers.0.mlp.experts",
                "_packed_param": "gate_up_proj",
            }
        }
        costs = {
            "model.layers.0.mlp.experts.gate_up_proj": {
                "NVFP4": {
                    "weight_mse": 0.20,
                    "output_mse": 0.0,
                    "output_mse_measured": False,
                    "predicted_dloss": 7.0,
                },
                # Defensive path for old packed artifacts that wrote the
                # placeholder zero but did not carry the explicit measured flag.
                "MXFP4": {
                    "weight_mse": 0.10,
                    "output_mse": 0.0,
                    "predicted_dloss": 2.0,
                },
                "BF16": {
                    "weight_mse": 0.25,
                    "output_mse": 0.0,
                    "output_mse_measured": False,
                },
            }
        }
        cands = build_candidates(
            stats,
            costs,
            [fr.get_format("NVFP4"), fr.get_format("MXFP4"), fr.get_format("BF16")],
        )
        by_fmt = {
            c.fmt: c
            for c in cands["model.layers.0.mlp.experts.gate_up_proj"]
        }
        self.assertAlmostEqual(by_fmt["NVFP4"].predicted_dloss, 7.0)
        self.assertAlmostEqual(by_fmt["MXFP4"].predicted_dloss, 2.0)
        self.assertAlmostEqual(by_fmt["BF16"].predicted_dloss, 0.5 * 3.0 * 0.25)
        expected_nvfp4_bytes = fr.get_format("NVFP4").memory_bytes_for_shape(
            (2, 8, 16)
        )
        self.assertEqual(by_fmt["NVFP4"].memory_bytes, expected_nvfp4_bytes)

    def test_build_candidates_prices_packed_expert_memory_with_expert_dim(self):
        name = "model.layers.0.mlp.experts.down_proj"
        stats = {
            name: {
                "h_trace": 1.0,
                "out_features": 8,
                "in_features": 16,
                "num_experts": 4,
                "n_params": 4 * 8 * 16,
                "_packed_experts_module": "model.layers.0.mlp.experts",
                "_packed_param": "down_proj",
            }
        }
        costs = {
            name: {
                "NVFP4": {"weight_mse": 0.1, "predicted_dloss": 0.1},
                "BF16": {"weight_mse": 0.0, "predicted_dloss": 0.0},
            }
        }

        cands = build_candidates(
            stats, costs, [fr.get_format("NVFP4"), fr.get_format("BF16")]
        )
        by_fmt = {c.fmt: c for c in cands[name]}

        self.assertEqual(
            by_fmt["NVFP4"].memory_bytes,
            fr.get_format("NVFP4").memory_bytes_for_shape((4, 8, 16)),
        )
        self.assertEqual(
            by_fmt["BF16"].memory_bytes,
            fr.get_format("BF16").memory_bytes_for_shape((4, 8, 16)),
        )

    def test_qwen_packed_moe_profile_allows_only_exportable_expert_formats(self):
        name = "model.layers.0.mlp.experts.gate_up_proj"
        candidates = {
            name: [
                Candidate("NVFP4", 4.5, 100, 1.0),
                Candidate("MXFP8", 8.25, 180, 0.2),
                Candidate("MXFP8_E4M3", 8.25, 180, 0.2),
                Candidate("MXFP4", 4.25, 96, 0.9),
                Candidate("FP8_E4M3", 8.5, 190, 0.1),
                Candidate("BF16", 16.0, 512, 0.0),
            ],
            "model.layers.0.self_attn.q_proj": [
                Candidate("MXFP4", 4.25, 96, 0.9),
            ],
        }

        filtered = filter_candidates_for_profile(
            candidates, "vllm_packed_moe"
        )

        self.assertEqual(
            [c.fmt for c in filtered[name]],
            ["NVFP4", "MXFP8", "MXFP8_E4M3", "MXFP4", "FP8_E4M3", "BF16"],
        )
        self.assertNotIn("model.layers.0.self_attn.q_proj", filtered)


class _ToyExpertsLinearLoop(nn.Module):
    def __init__(self, num_experts=3, hidden=4, intermediate=6):
        super().__init__()
        self.gate_up_proj = nn.ModuleList(
            [nn.Linear(hidden, 2 * intermediate, bias=False) for _ in range(num_experts)]
        )
        self.down_proj = nn.ModuleList(
            [nn.Linear(intermediate, hidden, bias=False) for _ in range(num_experts)]
        )


class _ToyCustomExpertsLinearLoop(nn.Module):
    def __init__(self, num_experts=3, hidden=4, intermediate=6):
        super().__init__()
        self.w13 = nn.ModuleList(
            [nn.Linear(hidden, 2 * intermediate, bias=False) for _ in range(num_experts)]
        )
        self.w2 = nn.ModuleList(
            [nn.Linear(intermediate, hidden, bias=False) for _ in range(num_experts)]
        )


class _CustomPackedProfile:
    def packed_expert_param_names(self) -> frozenset[str]:
        return frozenset({"w13", "w2"})

    def packed_expert_projection_names(self, param_name: str) -> tuple[str, ...]:
        if param_name == "w13":
            return ("w1_proj", "w3_proj")
        return (param_name,)


class _ToyMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 3, bias=False)
        self.experts = _ToyExpertsLinearLoop()


class _ToyRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3, 4))


class _ToyMoeBlockCustomRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _ToyRouter()
        self.experts = _ToyExpertsLinearLoop()


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = _ToyMoeBlock()


class _ToyCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.gate = nn.Linear(4, 3, bias=False)
        self.model.layers[0].mlp.experts = _ToyCustomExpertsLinearLoop()


class TestMoeDiscovery(unittest.TestCase):
    def test_discover_moe_structure_handles_linear_loop_projection_lists(self):
        toy = _ToyModel()
        info = discover_moe_structure(toy)
        self.assertEqual(info["model.layers.0.mlp.experts.gate_up_proj.0"], ("model.layers.0.mlp.gate", "0"))
        self.assertEqual(info["model.layers.0.mlp.experts.down_proj.2"], ("model.layers.0.mlp.gate", "2"))
        self.assertEqual(len(info), 6)

    def test_discover_moe_structure_handles_router_modules_with_weight(self):
        toy = _ToyModel()
        toy.model.layers[0].mlp = _ToyMoeBlockCustomRouter()
        info = discover_moe_structure(toy)
        self.assertEqual(info["model.layers.0.mlp.experts.gate_up_proj.1"], ("model.layers.0.mlp.gate", "1"))
        self.assertEqual(len(info), 6)

    def test_discover_moe_structure_uses_profile_projection_names(self):
        toy = _ToyCustomModel()
        info = discover_moe_structure(toy, profile=_CustomPackedProfile())

        self.assertEqual(
            info["model.layers.0.mlp.experts.w13.0"],
            ("model.layers.0.mlp.gate", "0"),
        )
        self.assertEqual(
            info["model.layers.0.mlp.experts.w2.2"],
            ("model.layers.0.mlp.gate", "2"),
        )
        self.assertEqual(len(info), 6)


if __name__ == "__main__":
    unittest.main()
