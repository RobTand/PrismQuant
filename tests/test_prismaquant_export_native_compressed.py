"""Tests for the native compressed-tensors exporter.

Covers the math (NVFP4 / FP8 round-trip) and the wire-format
plumbing (`_to_vllm_internal_name`, `build_quantization_config`)
that has to stay in sync with vLLM's compressed-tensors loader.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
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
    _compressed_tensor_key,
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
    def test_rejects_symlink_output_directory(self):
        from prismaquant.export_native_compressed import (
            IncrementalSafetensorsWriter,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / "target"
            target.mkdir()
            output = root / "output"
            output.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(RuntimeError, "real directory"):
                IncrementalSafetensorsWriter(output, shard_bytes=32)

            self.assertTrue(output.is_symlink())
            self.assertEqual(list(target.iterdir()), [])

    def test_rejects_preexisting_high_index_temp_shard(self):
        from prismaquant.export_native_compressed import (
            IncrementalSafetensorsWriter,
        )

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            stale = out_dir / ".model-99999.safetensors.tmp"
            stale.write_bytes(b"partial-old-export")

            with self.assertRaisesRegex(
                RuntimeError,
                "preexisting native temporary shard",
            ):
                IncrementalSafetensorsWriter(out_dir, shard_bytes=32)

            self.assertEqual(stale.read_bytes(), b"partial-old-export")

    def test_finalizes_multi_shard_index_without_temp_files(self):
        from safetensors.torch import load_file

        from prismaquant.export_native_compressed import (
            IncrementalSafetensorsWriter,
        )

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            (out_dir / "model.safetensors").write_bytes(b"stale-single")
            (out_dir / "model-99999-of-99999.safetensors").write_bytes(
                b"stale-shard"
            )
            (out_dir / "tokenizer.json").write_text("keep")
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
            self.assertFalse((out_dir / "model.safetensors").exists())
            self.assertFalse(
                (out_dir / "model-99999-of-99999.safetensors").exists()
            )
            self.assertEqual((out_dir / "tokenizer.json").read_text(), "keep")

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

    def test_single_shard_replaces_stale_shards_and_index_only(self):
        from prismaquant.export_native_compressed import (
            IncrementalSafetensorsWriter,
        )

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            (out_dir / "model-00001-of-00002.safetensors").write_bytes(b"old")
            (out_dir / "model-00002-of-00002.safetensors").write_bytes(b"old")
            (out_dir / "model.safetensors.index.json").write_text("{}")
            (out_dir / "config.json").write_text('{"keep": true}')

            writer = IncrementalSafetensorsWriter(out_dir, shard_bytes=1024)
            writer.add_tensors({"a.weight": torch.ones(4)})
            writer.finalize()

            self.assertTrue((out_dir / "model.safetensors").exists())
            self.assertFalse(
                (out_dir / "model.safetensors.index.json").exists()
            )
            self.assertFalse(list(out_dir.glob("model-*-of-*.safetensors")))
            self.assertEqual(
                (out_dir / "config.json").read_text(), '{"keep": true}'
            )


class TestNativeExportOutputSafety(unittest.TestCase):
    @staticmethod
    def _argv(model: Path, output: Path, layer_config: Path) -> list[str]:
        return [
            "export_native_compressed",
            "--model",
            str(model),
            "--layer-config",
            str(layer_config),
            "--output",
            str(output),
        ]

    def test_main_rejects_in_place_and_symlink_alias_before_model_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            source = model / "model.safetensors"
            source.write_bytes(b"source-must-survive")
            layer_config = root / "assignment.json"
            layer_config.write_text("{}")

            for output in (model, root / "model-alias"):
                if output != model:
                    output.symlink_to(model, target_is_directory=True)
                with patch.object(
                    sys,
                    "argv",
                    self._argv(model, output, layer_config),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "resolve to the same path",
                    ):
                        enc.main()

            self.assertEqual(source.read_bytes(), b"source-must-survive")
            self.assertTrue((root / "model-alias").is_symlink())

    def test_main_rejects_stale_aux_before_model_load(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"source")
            layer_config = root / "assignment.json"
            layer_config.write_text("{}")
            output = root / "output"
            output.mkdir()
            stale = output / "modeling_old_remote_code.py"
            stale.write_text("STALE = True\n")

            with patch.object(
                sys,
                "argv",
                self._argv(model, output, layer_config),
            ):
                with self.assertRaisesRegex(RuntimeError, "is not empty"):
                    enc.main()

            self.assertEqual(stale.read_text(), "STALE = True\n")
            self.assertEqual(set(output.iterdir()), {stale})

    def test_main_rejects_ancestor_descendant_output_trees(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            layer_config = root / "assignment.json"
            layer_config.write_text("{}")

            model = root / "model"
            model.mkdir()
            payload = model / "model.safetensors"
            payload.write_bytes(b"source-one")
            nested_output = model / "exported"
            with patch.object(
                sys,
                "argv",
                self._argv(model, nested_output, layer_config),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "ancestor/descendant",
                ):
                    enc.main()
            self.assertFalse(nested_output.exists())
            self.assertEqual(payload.read_bytes(), b"source-one")

            outer_output = root / "outer-output"
            nested_model = outer_output / "model"
            nested_model.mkdir(parents=True)
            nested_payload = nested_model / "model.safetensors"
            nested_payload.write_bytes(b"source-two")
            with patch.object(
                sys,
                "argv",
                self._argv(nested_model, outer_output, layer_config),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "ancestor/descendant",
                ):
                    enc.main()
            self.assertEqual(nested_payload.read_bytes(), b"source-two")
            self.assertEqual(set(outer_output.iterdir()), {nested_model})

    def test_main_transaction_cleans_post_model_budget_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"source")
            layer_config = root / "assignment.json"
            layer_config.write_text("{}")
            output = root / "output"
            output.mkdir()
            before = output.stat()
            export_cache = root / "export-cache"
            export_cache.mkdir()
            (export_cache / "layer_000.pt").write_bytes(b"resume")

            def fail_after_model(argv):
                index = argv.index("--output")
                staged = Path(argv[index + 1])
                self.assertNotEqual(staged, output)
                (staged / "model.safetensors").write_bytes(b"over-budget")
                raise RuntimeError("hard whole-artifact budget exceeded")

            with patch.object(
                sys,
                "argv",
                self._argv(model, output, layer_config)
                + ["--export-cache-dir", str(export_cache)],
            ), patch.object(enc, "_main_impl", side_effect=fail_after_model):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "hard whole-artifact budget",
                ):
                    enc.main()

            after = output.stat()
            self.assertEqual(
                (after.st_dev, after.st_ino),
                (before.st_dev, before.st_ino),
            )
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(list(root.glob(".output.tmp-*")), [])
            self.assertEqual(
                (export_cache / "layer_000.pt").read_bytes(),
                b"resume",
            )

    def test_main_transaction_publishes_then_prints_final_serve_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"source")
            layer_config = root / "assignment.json"
            layer_config.write_text("{}")
            output = root / "output"
            export_cache = root / "export-cache"
            export_cache.mkdir()
            (export_cache / "layer_000.pt").write_bytes(b"resume")

            def succeed(argv):
                index = argv.index("--output")
                staged = Path(argv[index + 1])
                self.assertFalse(output.exists())
                (staged / "model.safetensors").write_bytes(b"complete")
                return "finished"

            with patch.object(
                sys,
                "argv",
                self._argv(model, output, layer_config)
                + ["--export-cache-dir", str(export_cache)],
            ), patch.object(
                enc,
                "_main_impl",
                side_effect=succeed,
            ), patch("builtins.print") as printed:
                self.assertEqual(enc.main(), "finished")

            self.assertEqual(
                (output / "model.safetensors").read_bytes(),
                b"complete",
            )
            self.assertEqual(list(root.glob(".output.tmp-*")), [])
            self.assertFalse(export_cache.exists())
            rendered = "\n".join(
                " ".join(str(arg) for arg in call.args)
                for call in printed.call_args_list
            )
            self.assertIn(str(output.resolve()), rendered)
            self.assertNotIn(".output.tmp-", rendered)


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


def _tiny_qwen_packed_root():
    root = nn.Module()
    root.model = nn.Module()
    root.model.language_model = nn.Module()
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = _TinyQwenPackedExperts()
    root.model.language_model.layers = nn.ModuleList([layer])
    return root


class TestPackedExpertExport(unittest.TestCase):
    def test_mxfp8_split_experts_emit_weight_suffix_for_vllm_loader(self):
        """Qwen3.5's split expert loader matches
        `experts.<id>.<proj>.weight`; without the `.weight` suffix raw
        MXFP8 expert weights fall through to the generic loader and are
        skipped with "not found in params_dict" warnings.
        """

        root = _tiny_qwen_packed_root()

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
            hist.get(("packed_moe_per_expert", "MXFP8_E4M3+rtn")),
            2,
        )

    def test_packed_expert_missing_production_cache_raises_by_default(self):
        from prismaquant.production_weight_cache import ProductionWeightCache

        assignment = {
            "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
            "model.layers.0.mlp.experts.down_proj": "NVFP4",
        }
        old_cache = enc._PRODUCTION_WEIGHT_CACHE
        old_escape = enc._ALLOW_PACKED_EXPERT_RTN
        try:
            enc._PRODUCTION_WEIGHT_CACHE = ProductionWeightCache(
                weights={}, levers={"gptq": True})
            enc._ALLOW_PACKED_EXPERT_RTN = False
            with self.assertRaisesRegex(
                RuntimeError,
                "has no production-cache render",
            ):
                enc._materialize_tensors_inmemory(
                    _tiny_qwen_packed_root(),
                    assignment,
                    bf16_passthrough=set(),
                    profile=Qwen3_5Profile(),
                )
        finally:
            enc._PRODUCTION_WEIGHT_CACHE = old_cache
            enc._ALLOW_PACKED_EXPERT_RTN = old_escape

    def test_packed_expert_rtn_escape_allows_cache_miss(self):
        from prismaquant.production_weight_cache import ProductionWeightCache

        assignment = {
            "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
            "model.layers.0.mlp.experts.down_proj": "NVFP4",
        }
        old_cache = enc._PRODUCTION_WEIGHT_CACHE
        old_escape = enc._ALLOW_PACKED_EXPERT_RTN
        try:
            enc._PRODUCTION_WEIGHT_CACHE = ProductionWeightCache(
                weights={}, levers={"gptq": True})
            enc._ALLOW_PACKED_EXPERT_RTN = True
            tensors, hist = enc._materialize_tensors_inmemory(
                _tiny_qwen_packed_root(),
                assignment,
                bf16_passthrough=set(),
                profile=Qwen3_5Profile(),
            )
        finally:
            enc._PRODUCTION_WEIGHT_CACHE = old_cache
            enc._ALLOW_PACKED_EXPERT_RTN = old_escape

        self.assertTrue(any(k.endswith(".weight_packed") for k in tensors))
        self.assertEqual(
            hist.get(("packed_moe_per_expert", "NVFP4+rtn")),
            2,
        )

    def test_packed_expert_no_cache_path_warns_before_rtn(self):
        assignment = {
            "model.layers.0.mlp.experts.gate_up_proj": "MXFP8_E4M3",
            "model.layers.0.mlp.experts.down_proj": "MXFP8_E4M3",
        }
        old_cache = enc._PRODUCTION_WEIGHT_CACHE
        old_escape = enc._ALLOW_PACKED_EXPERT_RTN
        stdout = io.StringIO()
        try:
            enc._PRODUCTION_WEIGHT_CACHE = None
            enc._ALLOW_PACKED_EXPERT_RTN = False
            with redirect_stdout(stdout):
                _tensors, hist = enc._materialize_tensors_inmemory(
                    _tiny_qwen_packed_root(),
                    assignment,
                    bf16_passthrough=set(),
                    profile=Qwen3_5Profile(),
                )
        finally:
            enc._PRODUCTION_WEIGHT_CACHE = old_cache
            enc._ALLOW_PACKED_EXPERT_RTN = old_escape

        self.assertIn("WARNING: RTN-rendering packed expert", stdout.getvalue())
        self.assertIn("no production cache", stdout.getvalue())
        self.assertEqual(
            hist.get(("packed_moe_per_expert", "MXFP8_E4M3+rtn")),
            2,
        )

    def test_packed_expert_hist_label_distinguishes_cached_and_rtn(self):
        self.assertEqual(
            enc._packed_expert_render_hist_label(
                "NVFP4",
                is_bf16=False,
                source_label="bf16",
                cached_3d=torch.ones(1, 1, 1),
            ),
            "NVFP4+cached",
        )
        self.assertEqual(
            enc._packed_expert_render_hist_label(
                "NVFP4",
                is_bf16=False,
                source_label="bf16",
                cached_3d=None,
            ),
            "NVFP4+rtn",
        )
        self.assertEqual(
            enc._packed_expert_render_hist_label(
                "BF16",
                is_bf16=True,
                source_label="bf16",
                cached_3d=None,
            ),
            "bf16",
        )

    def test_packed_expert_export_provenance_records_escape_and_coverage(self):
        class _Cache:
            metadata = {
                "packed_expert_coverage": {
                    "layer.experts.gate_up_proj": {
                        "rtn_fallbacks": 2,
                        "gptq_experts": 0,
                    }
                }
            }

        old_cache = enc._PRODUCTION_WEIGHT_CACHE
        old_escape = enc._ALLOW_PACKED_EXPERT_RTN
        try:
            enc._PRODUCTION_WEIGHT_CACHE = _Cache()
            enc._ALLOW_PACKED_EXPERT_RTN = True
            prov = enc._packed_expert_export_provenance()
        finally:
            enc._PRODUCTION_WEIGHT_CACHE = old_cache
            enc._ALLOW_PACKED_EXPERT_RTN = old_escape

        self.assertTrue(prov["rtn_escape_enabled"])
        self.assertTrue(prov["cache_has_packed_expert_coverage"])
        self.assertEqual(
            prov["cache_packed_expert_coverage"][
                "layer.experts.gate_up_proj"
            ]["rtn_fallbacks"],
            2,
        )

    def test_expected_cache_keys_require_packed_experts_by_default(self):
        from prismaquant.production_weight_cache import ProductionWeightCache

        assignment = {
            "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
            "model.layers.0.self_attn.q_proj": "NVFP4",
        }
        old_cache = enc._PRODUCTION_WEIGHT_CACHE
        old_escape = enc._ALLOW_PACKED_EXPERT_RTN
        try:
            enc._PRODUCTION_WEIGHT_CACHE = ProductionWeightCache(
                weights={}, levers={"gptq": True})
            enc._ALLOW_PACKED_EXPERT_RTN = False
            keys, missing = enc._production_cache_expected_keys(assignment)
        finally:
            enc._PRODUCTION_WEIGHT_CACHE = old_cache
            enc._ALLOW_PACKED_EXPERT_RTN = old_escape

        self.assertEqual(keys, [])
        self.assertIn(
            ("model.layers.0.mlp.experts.gate_up_proj", "NVFP4"),
            missing,
        )
        self.assertIn(
            ("model.layers.0.self_attn.q_proj", "NVFP4"),
            missing,
        )

    def test_expected_cache_keys_skip_passthrough_formats(self):
        """Issue #29 side effect, pinned: FP8_SOURCE entries now survive the
        runtime-legality guard (before, every one was rewritten to BF16
        before this check ran), and their emit path copies source bytes
        without ever consulting the cache -- so demanding a render for them
        would fail a valid FP8-source export over an entry it cannot use."""
        from prismaquant.production_weight_cache import ProductionWeightCache

        old_cache = enc._PRODUCTION_WEIGHT_CACHE
        try:
            enc._PRODUCTION_WEIGHT_CACHE = ProductionWeightCache(
                weights={}, levers={"gptq": True})
            keys, missing = enc._production_cache_expected_keys({
                "model.layers.0.self_attn.o_proj": "FP8_SOURCE",
                "model.layers.0.self_attn.k_proj": "BF16",
                "model.layers.0.self_attn.q_proj": "NVFP4",
            })
        finally:
            enc._PRODUCTION_WEIGHT_CACHE = old_cache

        self.assertEqual(keys, [])
        self.assertEqual(
            missing, [("model.layers.0.self_attn.q_proj", "NVFP4")]
        )

    def test_expected_cache_keys_escape_skips_only_packed_experts(self):
        from prismaquant.production_weight_cache import ProductionWeightCache

        assignment = {
            "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
            "model.layers.0.self_attn.q_proj": "NVFP4",
        }
        old_cache = enc._PRODUCTION_WEIGHT_CACHE
        old_escape = enc._ALLOW_PACKED_EXPERT_RTN
        try:
            enc._PRODUCTION_WEIGHT_CACHE = ProductionWeightCache(
                weights={}, levers={"gptq": True})
            enc._ALLOW_PACKED_EXPERT_RTN = True
            keys, missing = enc._production_cache_expected_keys(assignment)
        finally:
            enc._PRODUCTION_WEIGHT_CACHE = old_cache
            enc._ALLOW_PACKED_EXPERT_RTN = old_escape

        self.assertEqual(keys, [])
        self.assertNotIn(
            ("model.layers.0.mlp.experts.gate_up_proj", "NVFP4"),
            missing,
        )
        self.assertEqual(
            missing,
            [("model.layers.0.self_attn.q_proj", "NVFP4")],
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
        gguf_lane = {
            # Served via the GGUF container (export_gguf), not
            # compressed-tensors; rendered==served equivalence is pinned
            # bit-exact against gguf-py in tests/test_gguf_formats.py and
            # tests/test_gguf_iq_formats.py (the IQ family).
            "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0",
            "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL",
        }
        nvfp4_cb_lane = {
            # Served via the custom NVFP4-CB vLLM plugin container, not
            # compressed-tensors (its scheme vocabulary cannot express
            # codebooks); rendered==served is pinned in
            # tests/test_nvfp4_cb_formats.py (docs/lanes/nvfp4-cb).
            s.name for s in fr.REGISTRY.values()
            if s.family in ("nvfp4_cb", "fp8_cb")
        }
        nvfp4_cb_container_passthroughs = {
            # SOURCE-PASSTHROUGH carriers of the nvfp4_cb container. They have
            # no compressed-tensors served-metadata path because they have no
            # RENDER at all: the exporter copies the checkpoint's own bytes
            # (packed MXFP4 + E8M0 group scales; E4M3 + UE8M0 block scales).
            # "rendered == served" is trivially true and is pinned as the
            # identity of both the weight and activation paths in
            # test_registry_render_dequant_matches_served_metadata, and
            # byte-for-byte against a real source slice in
            # tests/test_nvfp4_cb_streaming.py. BF16 and FP8_SOURCE stay in
            # `reconciled` — those two ARE compressed-tensors passthroughs.
            "MXFP4_SOURCE",
            "FP8_BLOCK_UE8M0_SOURCE",
        }
        nvfp4_cb_container_requant = {
            # RE-QUANTIZED carriers of the nvfp4_cb container: unlike the
            # passthroughs above these DO have a render, but it is not a
            # compressed-tensors one — the exporter writes the element and
            # scale planes itself under a Gridbook wire id. "rendered ==
            # served" is therefore a real claim with real content, and it is
            # checked directly below (against the same E8M0 served formula the
            # stock MXFP8 rungs use) rather than deferred to another file.
            "MXFP8_UE8M0_G32",
        }
        self.assertEqual(
            set(fr.REGISTRY),
            reconciled | set(explicit_gaps) | gguf_lane | nvfp4_cb_lane
            | nvfp4_cb_container_passthroughs | nvfp4_cb_container_requant,
        )

    def test_registry_render_dequant_matches_served_metadata(self):
        W = torch.randn(8, 128) * 1.75
        prev = enc._NVFP4_SCALE_RULE
        try:
            enc._NVFP4_SCALE_RULE = enc.NVFP4_SCALE_RULE_STATIC_6
            for fmt in sorted(fr.REGISTRY):
                with self.subTest(fmt=fmt):
                    if fmt in {"MXFP6_E3M2", "MXFP6_E2M3", "INT8_W8A16", "INT4_W4A16_g128"}:
                        continue
                    if fr.get_format(fmt).family == "gguf":
                        # GGUF-container formats: rendered==served is pinned
                        # bit-exact in tests/test_gguf_formats.py.
                        continue
                    if fr.get_format(fmt).family in ("nvfp4_cb", "fp8_cb"):
                        # NVFP4-CB plugin-container formats: rendered==served
                        # is pinned in tests/test_nvfp4_cb_formats.py.
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

                    if fmt == "MXFP8_UE8M0_G32":
                        # Not a compressed-tensors rung: the STREAMING
                        # exporter writes its planes itself. Pack with the
                        # exporter's own packer, then decode with the same
                        # served E8M0 formula the stock MXFP8 rungs use, and
                        # require the registry render to equal it exactly.
                        from prismaquant.export_nvfp4_cb_streaming import (
                            _requant_pack,
                        )

                        packed = _requant_pack(fmt, W)
                        served = _mxfp8_served_dequantize(
                            packed["weight"],
                            packed["weight_scale"].view(torch.uint8),
                        )
                        rendered = fr.get_format(fmt).quantize_dequantize(
                            W.clone())
                        torch.testing.assert_close(rendered, served)
                        self.assertTrue(torch.equal(rendered, served))
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

                    from prismaquant.allocator_candidates import (
                        PASSTHROUGH_SOURCE_REQUIREMENTS,
                    )
                    if fmt in PASSTHROUGH_SOURCE_REQUIREMENTS:
                        # The rest of the SOURCE-PASSTHROUGH family
                        # (FP8_BLOCK_UE8M0_SOURCE, MXFP4_SOURCE, ...). There
                        # is no render to compare against a serve: the
                        # exporter copies the checkpoint's own bytes, so the
                        # served values ARE the source values. The invariant
                        # that must hold — and that would break if someone
                        # gave one of these a real codec — is that BOTH sides
                        # of the format are the identity, which is what makes
                        # its Δloss exactly 0 by construction rather than by
                        # measurement.
                        spec = fr.get_format(fmt)
                        torch.testing.assert_close(
                            spec.quantize_dequantize(W), W)
                        torch.testing.assert_close(
                            spec.activation_quantize_dequantize(W), W)
                        self.assertFalse(spec.act_quant_changes_input, fmt)
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

    def test_nvfp4_scale_selection_scores_fp8_snapped_scales(self):
        # Pins the RESEARCH path (snapped-scale scoring); default-off
        # pending its served A/B (QC M21).
        __import__('os').environ[
            'PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING'] = '1'
        self.addCleanup(lambda: __import__('os').environ.pop(
            'PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING', None))
        grouped = torch.tensor(
            [[[
                -0.02800447, 0.20611508, -0.25149462, 0.00026755,
                0.25256824, -0.12001037, 0.31183860, 0.10744593,
                -0.07380029, 0.69075495, -0.56450677, -0.01491811,
                -0.31349361, -0.28695017, 0.01005956, 0.21302599,
            ]]],
            dtype=torch.float32,
        )
        global_real = torch.tensor(0.0003, dtype=torch.float32)
        max_abs = grouped.abs().amax(dim=-1).clamp_min(1e-12)
        scale_6 = max_abs / 6.0
        scale_4 = max_abs / 4.0

        real_scale = enc._select_nvfp4_group_scales(
            grouped,
            scale_rule=enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
        )
        snapped_scale = enc._select_nvfp4_group_scales(
            grouped,
            scale_rule=enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
            global_real=global_real,
        )
        mse_6 = enc._nvfp4_mse_for_group_scale(
            grouped,
            scale_6,
            global_real=global_real,
        )
        mse_4 = enc._nvfp4_mse_for_group_scale(
            grouped,
            scale_4,
            global_real=global_real,
        )
        expected = torch.where(mse_4 < mse_6, scale_4, scale_6)

        self.assertFalse(torch.equal(real_scale, snapped_scale))
        torch.testing.assert_close(snapped_scale, expected)

        pack_scale, pack_global = enc._select_nvfp4_pack_scales_and_global(
            grouped,
            global_real_override=global_real,
            scale_rule=enc.NVFP4_SCALE_RULE_FOUR_OVER_SIX_MSE,
        )
        torch.testing.assert_close(pack_global, global_real)
        torch.testing.assert_close(pack_scale, expected)

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

    def test_quantized_head_weight_suffix_keeps_weight_key(self):
        self.assertEqual(
            _compressed_tensor_key("lm_head", "weight"),
            "lm_head.weight",
        )
        self.assertEqual(
            _compressed_tensor_key("lm_head", "weight_scale"),
            "lm_head.weight_scale",
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
    def test_batched_nvfp4_export_comment_matches_default_on(self):
        text = Path(enc.__file__).read_text()

        self.assertNotIn("disabled by default while", text)
        self.assertIn("PRISMAQUANT_BATCHED_NVFP4_EXPORT=0", text)

    def test_build_target_list_documents_sparse_expert_wildcard(self):
        doc = enc._build_target_list.__doc__ or ""

        self.assertIn("always emit a `[0-9]+`", doc)
        targets = enc._build_target_list([
            "model.layers.0.mlp.experts.2.gate_proj",
        ])
        self.assertEqual(targets, [
            "re:^model[.]layers[.]0[.]mlp[.]experts[.][0-9]+[.]gate_proj$",
        ])

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

    def test_legacy_native_config_does_not_claim_versioned_fused_contract(self):
        qc = build_quantization_config(
            {"model.layers.0.mlp.down_proj": "NVFP4"},
            bf16_passthrough=set(),
        )

        self.assertEqual(qc["quant_method"], "compressed-tensors")
        self.assertNotIn("execution_contracts", qc)
        self.assertNotIn(
            enc._nvfp4_activation_contract.NVFP4_ACTIVATION_CONTRACT_KEY,
            qc,
        )

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

    def test_dense_fused_sibling_mixed_quantized_formats_rejected(self):
        profile = Qwen3_5Profile()
        assignment = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.self_attn.k_proj": "MXFP8",
            "model.layers.0.self_attn.v_proj": "NVFP4",
        }

        with self.assertRaisesRegex(RuntimeError, "crash@load"):
            build_quantization_config(
                assignment, bf16_passthrough=set(), profile=profile,
            )

    def test_dense_fused_sibling_quantized_bf16_mix_rejected(self):
        profile = Qwen3_5Profile()
        assignment = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.self_attn.k_proj": "BF16",
            "model.layers.0.self_attn.v_proj": "BF16",
        }

        with self.assertRaisesRegex(RuntimeError, "silent-corruption"):
            build_quantization_config(
                assignment, bf16_passthrough=set(), profile=profile,
            )

    def test_incomplete_fused_sibling_mixed_present_states_rejected(self):
        profile = Qwen3_5Profile()
        assignment = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.self_attn.k_proj": "BF16",
        }

        with self.assertRaisesRegex(RuntimeError, "silent-corruption"):
            build_quantization_config(
                assignment,
                bf16_passthrough=set(),
                profile=profile,
            )

    def test_quantization_config_preflight_rejects_before_render(self):
        profile = Qwen3_5Profile()
        assignment = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.self_attn.k_proj": "BF16",
            "model.layers.0.self_attn.v_proj": "BF16",
        }

        with self.assertRaisesRegex(RuntimeError, "before rendering"):
            enc._preflight_quantization_config(
                assignment,
                set(),
                profile=profile,
            )

    def test_fp8_source_overlay_keeps_config_matched_to_emitted_bytes(self):
        profile = Qwen3_5Profile()
        assignment = {
            "model.layers.0.self_attn.o_proj": "BF16",
        }
        bf16_passthrough = {
            "model.layers.0.self_attn.o_proj",
            "lm_head",
        }
        fp8_map = {
            "model.layers.0.self_attn.o_proj": ("shard0", "o.scale"),
            "model.layers.0.mlp.down_proj": ("shard0", "down.scale"),
        }
        source_dtypes = {
            "model.layers.0.self_attn.o_proj.weight": torch.float8_e4m3fn,
            "model.layers.0.mlp.down_proj.weight": torch.float8_e4m3fn,
        }

        with (
            patch.object(enc, "_build_fp8_source_map", return_value=fp8_map),
            patch(
                "prismaquant.layer_streaming._build_weight_map",
                return_value=({}, {}),
            ),
            patch.object(enc, "_build_source_dtype_map", return_value=source_dtypes),
        ):
            config_assignment, config_bf16, overrides = (
                enc._fp8_source_config_overlay(
                    "/model",
                    assignment,
                    bf16_passthrough,
                    profile,
                )
            )

        self.assertEqual(
            config_assignment["model.layers.0.self_attn.o_proj"],
            "FP8_SOURCE",
        )
        self.assertEqual(
            config_assignment["model.layers.0.mlp.down_proj"],
            "FP8_SOURCE",
        )
        self.assertEqual(
            overrides,
            {
                "model.layers.0.self_attn.o_proj",
                "model.layers.0.mlp.down_proj",
            },
        )
        self.assertNotIn("model.layers.0.self_attn.o_proj", config_bf16)
        self.assertIn("lm_head", config_bf16)

        source_iter = [
            ("model.layers.0.self_attn.o_proj.weight", [128, 128]),
            ("model.layers.0.mlp.down_proj.weight", [128, 128]),
        ]
        self.assertEqual(
            compute_extra_ignore(source_iter, config_assignment, profile),
            [],
        )
        qc = build_quantization_config(
            config_assignment,
            config_bf16,
            profile=profile,
        )
        targets = {
            target
            for group in qc["config_groups"].values()
            for target in group["targets"]
        }
        self.assertIn(
            "re:^language_model[.]model[.]layers[.]0[.]self_attn[.]o_proj$",
            targets,
        )
        self.assertIn(
            "re:^language_model[.]model[.]layers[.]0[.]mlp[.]down_proj$",
            targets,
        )
        self.assertNotIn(
            "language_model.model.layers.0.self_attn.o_proj",
            qc["ignore"],
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
            # legacy default convention: 6 / max_abs = 6/3 = 2.0.
            self.assertAlmostEqual(
                float(out["input_global_scale"].item()), 2.0, places=5)
        finally:
            m._PRODUCTION_WEIGHT_CACHE = saved_cache
            m._INPUT_GLOBAL_SCALES = saved_scales

    def test_production_cache_scales_use_profile_fused_groups(self):
        import prismaquant.export_native_compressed as m
        from prismaquant.production_weight_cache import ProductionWeightCache

        class CustomProfile:
            def fused_sibling_group(self, qname: str) -> str | None:
                if qname.endswith(".a_proj") or qname.endswith(".b_proj"):
                    return qname.rsplit(".", 1)[0] + ".ab_proj"
                return None

        cache = ProductionWeightCache(
            weights={},
            levers={},
            activation_max_abs={
                "model.layers.0.a_proj": 12.0,
                "model.layers.0.b_proj": 24.0,
            },
        )

        scales = m._production_cache_scales(cache, profile=CustomProfile())

        # legacy default: 6/max_abs; the fused join takes min
        # (largest max_abs=24 wins) -> 6/24 = 0.25.
        self.assertEqual(scales["model.layers.0.a_proj"], 0.25)
        self.assertEqual(scales["model.layers.0.b_proj"], 0.25)

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


    def test_inmemory_mtp_linear_uses_production_cache(self):
        import prismaquant.export_native_compressed as m
        from prismaquant.production_weight_cache import ProductionWeightCache

        wrapper = nn.Module()
        wrapper.add_module("mtp", nn.Module())
        wrapper.mtp.add_module("proj", nn.Linear(16, 8, bias=False))
        W = torch.randn(8, 16) * 0.1
        cache = ProductionWeightCache(
            weights={("mtp.proj", "NVFP4"): W},
            levers={"gptq": True},
            activation_max_abs={"mtp.proj": 3.0},
        )
        saved_cache = m._PRODUCTION_WEIGHT_CACHE
        saved_scales = m._INPUT_GLOBAL_SCALES
        try:
            m._PRODUCTION_WEIGHT_CACHE = cache
            m._INPUT_GLOBAL_SCALES = m._production_cache_scales(cache)
            out, hist = m._materialize_tensors_inmemory(
                wrapper,
                {"mtp.proj": "NVFP4"},
                bf16_passthrough=set(),
                profile=_IdentityProfile(),
            )
        finally:
            m._PRODUCTION_WEIGHT_CACHE = saved_cache
            m._INPUT_GLOBAL_SCALES = saved_scales

        self.assertIn("mtp.proj.weight_packed", out)
        self.assertIn("mtp.proj.input_global_scale", out)
        self.assertEqual(hist[("linear", "NVFP4_PRODUCTION_CACHE")], 1)

    def test_inmemory_mtp_linear_missing_production_cache_raises(self):
        import prismaquant.export_native_compressed as m
        from prismaquant.production_weight_cache import ProductionWeightCache

        wrapper = nn.Module()
        wrapper.add_module("mtp", nn.Module())
        wrapper.mtp.add_module("proj", nn.Linear(16, 8, bias=False))
        cache = ProductionWeightCache(weights={}, levers={"gptq": True})
        saved_cache = m._PRODUCTION_WEIGHT_CACHE
        try:
            m._PRODUCTION_WEIGHT_CACHE = cache
            with self.assertRaisesRegex(RuntimeError, "auxiliary Linear mtp.proj"):
                m._materialize_tensors_inmemory(
                    wrapper,
                    {"mtp.proj": "NVFP4"},
                    bf16_passthrough=set(),
                    profile=_IdentityProfile(),
                )
        finally:
            m._PRODUCTION_WEIGHT_CACHE = saved_cache


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

    def test_native_fusion_compatibility_apis_delegate_to_contract_owner(self):
        shared = enc._nvfp4_activation_contract
        profile = object()
        sentinel_group = ("shared", ("a", "b"))
        with patch.object(
            shared,
            "fused_dense_group",
            return_value=sentinel_group,
        ) as dense:
            self.assertIs(enc._fused_dense_group("layer.a"), sentinel_group)
            dense.assert_called_once_with("layer.a")

        with patch.object(
            shared,
            "fused_sibling_group_key",
            return_value="shared.ab",
        ) as key:
            self.assertEqual(
                enc._fused_group_key_for_name("layer.a", profile),
                "shared.ab",
            )
            key.assert_called_once_with(
                "layer.a",
                profile=profile,
                tolerate_profile_errors=True,
            )

        expected = {"layer.a": 0.25, "layer.b": 0.25}
        with patch.object(
            shared,
            "unify_fused_sibling_input_global_scales",
            return_value=expected,
        ) as unify:
            actual = enc._unify_input_global_scales_across_fused_siblings(
                {"layer.a": 0.5, "layer.b": 0.25},
                profile=profile,
            )
            self.assertIs(actual, expected)
            unify.assert_called_once_with(
                {"layer.a": 0.5, "layer.b": 0.25},
                profile=profile,
                tolerate_profile_errors=True,
                diagnostic_prefix="[export-stream]",
            )

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

        def mtp_source_prefix(self):
            # Where the MTP tensors live in the SOURCE checkpoint. Recipe
            # names are always `mtp.*` regardless (see R12's
            # `build_mtp_module` naming contract), which is why the two
            # halves of the coverage check read different prefixes.
            return "mtp."

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


class TestExportableFormats(unittest.TestCase):
    """`EXPORTABLE_FORMATS` is what the serving profile's export lane
    bounds the allocator's menu by (issue #27), so it has to agree with
    what the export path really does -- not with a hand-maintained list.
    """

    def test_declared_set_matches_what_the_export_path_accepts(self):
        """Behavioural cross-check. For every registered format: if the
        exporter declares it emittable, some emit path takes it; if not,
        `_quantize_2d` refuses it.

        The two passthroughs are the reason this cannot be derived from the
        packer branches, so they are checked against their real emit path
        instead: BF16 through the plain-bf16 branch, FP8_SOURCE through the
        verbatim source copy (which has no `_quantize_2d` branch at all).
        """
        w = torch.randn(64, 256, dtype=torch.bfloat16)
        for fmt in sorted(fr.REGISTRY):
            emittable = fmt in enc.EXPORTABLE_FORMATS
            with self.subTest(fmt=fmt, emittable=emittable):
                if fmt == "FP8_SOURCE":
                    # Scheme + verbatim copy, no weight codec: the packer
                    # refuses it while the exporter still ships it.
                    self.assertTrue(emittable)
                    with self.assertRaises(ValueError):
                        _quantize_2d(w, fmt)
                    continue
                if fmt == "BF16":
                    self.assertTrue(emittable)
                    self.assertEqual(
                        _quantize_2d(w, fmt)["weight"].dtype, torch.bfloat16)
                    continue
                if emittable:
                    self.assertTrue(_quantize_2d(w, fmt))
                else:
                    with self.assertRaises(ValueError):
                        _quantize_2d(w, fmt)

    def test_every_declared_format_has_config_groups_metadata(self):
        """Gate #9's first clause: emittable means vLLM can dispatch it, so
        every non-passthrough entry must resolve to a `config_groups`
        scheme. BF16 is the sole exception -- it is named on `ignore`."""
        for fmt in sorted(enc.EXPORTABLE_FORMATS):
            if fmt in enc.CONTAINER_PASSTHROUGH_FORMATS:
                self.assertNotIn(fmt, enc.FORMAT_SCHEME, fmt)
                continue
            self.assertIn("config_groups", build_quantization_config(
                {"model.layers.0.self_attn.o_proj": fmt}, set()), fmt)

    def test_declaration_is_derived_and_canonical(self):
        """Derived, not hand-listed: FORMAT_SCHEME's legacy `MXFP8` alias
        must not leak into the declaration as a distinct rung, and adding a
        scheme must not need a second edit here. Canonicalized here through
        the registry (what the profile side uses), not the exporter's own
        alias map, so the two cannot drift apart."""
        self.assertEqual(
            enc.EXPORTABLE_FORMATS,
            frozenset(
                {fr.canonical_format_name(f) for f in enc.FORMAT_SCHEME}
                | set(enc.CONTAINER_PASSTHROUGH_FORMATS)
            ),
        )
        self.assertNotIn("MXFP8", enc.EXPORTABLE_FORMATS)
        self.assertIn("MXFP8_E4M3", enc.EXPORTABLE_FORMATS)
        # The asymmetry the constant exists to record.
        self.assertIn("FP8_SOURCE", enc.EXPORTABLE_FORMATS)
        self.assertNotIn("FP8_E5M2", enc.EXPORTABLE_FORMATS)


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
        # Rows stay positionally `(name, shape, from_fmt)` for the manifest
        # and `_bf16_upgrade_audit`; a lone Linear carries no serving group.
        self.assertEqual(len(coerced), 1)
        self.assertEqual(
            tuple(coerced[0][:3]),
            ("model.layers.0.linear_attn.in_proj_a", [48, 5120], "MXFP8_E4M3"),
        )
        self.assertIsNone(coerced[0].serving_group)
        self.assertEqual(coerced[0].serving_group_members, ())

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
        self.assertEqual(len(coerced), 1)
        self.assertEqual(
            tuple(coerced[0][:3]),
            ("model.layers.0.self_attn.o_proj", [128, 5120], "MXFP4"),
        )
        self.assertIsNone(coerced[0].serving_group)

    def _single_linear_source(self, td: Path) -> None:
        from safetensors.torch import save_file

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

    def test_unexportable_format_hard_fails_instead_of_bf16_coercion(self):
        """Issue #27. A format with no `config_groups` scheme cannot be
        emitted at all, so rewriting it to BF16 would ship that Linear at
        16 bpp -- blowing the byte budget the allocation was selected under
        and leaving the artifact's real bpp disagreeing with its own
        layer_config.json, with nothing recorded in the selection. The
        serving profile's export lane bounds the allocator's menu by
        EXPORTABLE_FORMATS, so reaching here is a regression in that bound
        and must be loud (CLAUDE.md §4.1: no post-allocator rewrites).

        FP8_E5M2 is the sharp case: it HAS a `_quantize_2d` byte-packer
        branch, so "has a packer" is not the same question as "is
        emittable"."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self._single_linear_source(td)

            for fmt in ("FP8_E5M2", "NVFP4A16", "MXFP6_E3M2", "INT8_W8A16"):
                with self.subTest(fmt=fmt):
                    with self.assertRaises(ValueError) as ctx:
                        _coerce_runtime_legal_assignment(
                            str(td),
                            {"model.layers.0.self_attn.o_proj": fmt},
                            Qwen3_5Profile(),
                        )
                    msg = str(ctx.exception)
                    # Names the Linear, the format, and the profile that
                    # admitted it -- an operator has to be able to act on it.
                    self.assertIn("model.layers.0.self_attn.o_proj", msg)
                    self.assertIn(fmt, msg)
                    self.assertIn("vllm_packed_moe", msg)
                    self.assertIn("EXPORTABLE_FORMATS", msg)

    def test_unexportable_format_hard_fails_without_a_profile(self):
        """The `profile=None` path (target_profile falls back to
        `research`, which is deliberately unbounded) must fail the same
        way: an unemittable format is a container fact, not a policy one."""
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            self._single_linear_source(td)

            with self.assertRaisesRegex(ValueError, "no compressed-tensors"):
                _coerce_runtime_legal_assignment(
                    str(td),
                    {"model.layers.0.self_attn.o_proj": "NVFP4A16"},
                    None,
                )

    def test_gguf_formats_hard_fail_instead_of_bf16_coercion(self):
        """A GGUF assignment reaching the compressed-tensors exporter is a
        wrong-container invocation (EXPORT_CONTAINER=gguf was not set), not
        a research format: silent BF16 coercion would ship a ~16 bpp
        artifact unrelated to the allocated budget."""
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

            with self.assertRaisesRegex(ValueError, "GGUF"):
                _coerce_runtime_legal_assignment(
                    str(td),
                    {"model.layers.0.self_attn.o_proj": "Q2_K"},
                    Qwen3_5Profile(),
                )

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


class TestServingGroupRuntimeCoercion(unittest.TestCase):
    """Issue #28: the shape/policy branch of the runtime-legality guard is
    serving-group aware.

    Packed-MoE expert projections and fused siblings (q/k/v, gate/up) are
    ATOMIC at serve time -- vLLM's `CompressedTensorsMoEMethod` selects one
    scheme per FusedMoE layer, and merged-column Linears carry one scheme
    for the whole packed weight. Their members do NOT share a shape, so a
    format legal for one member can be illegal for another; rewriting only
    the offending member produces a quantized + BF16 mix inside one serving
    unit, which is either a crash at load or (worse) a silent corruption.

    The guard therefore resolves the whole unit: it refuses when a
    quantized format legal for EVERY member exists (the allocation is
    repairable upstream at a quantized bit rate, so a 16 bpp rewrite of the
    unit is not what a shape-aware allocator would produce), and coerces
    the whole unit to BF16 when nothing else is representable.
    """

    def _source(self, td: Path, tensors: dict[str, tuple[int, int]]) -> str:
        from safetensors.torch import save_file

        shard = td / "model-00001-of-00001.safetensors"
        save_file(
            {
                f"{name}.weight": torch.zeros(*shape, dtype=torch.bfloat16)
                for name, shape in tensors.items()
            },
            str(shard),
        )
        with open(td / "model.safetensors.index.json", "w") as f:
            json.dump(
                {"weight_map": {f"{n}.weight": shard.name for n in tensors}}, f
            )
        return str(td)

    # `moe_intermediate_size = 40` is the trap the issue names: it divides
    # nothing, so NVFP4's group of 16 is illegal on `down_proj` (reduce dim
    # = intermediate) while `gate_proj`/`up_proj` (reduce dim = hidden) are
    # perfectly legal. One FusedMoE, two verdicts.
    _EXPERT_SHAPES = {"gate_proj": (40, 2048), "up_proj": (40, 2048),
                      "down_proj": (2048, 40)}
    _QKV_SHAPES = {"q_proj": (128, 5120), "k_proj": (48, 5120),
                   "v_proj": (48, 5120)}

    def _packed_moe_source(self, td: Path, n_experts: int = 2,
                           dense_neighbour: bool = False):
        tensors, assignment = {}, {}
        for expert in range(n_experts):
            for proj, shape in self._EXPERT_SHAPES.items():
                name = f"model.layers.0.mlp.experts.{expert}.{proj}"
                tensors[name] = shape
                assignment[name] = "NVFP4"
        if dense_neighbour:
            tensors["model.layers.0.self_attn.o_proj"] = (128, 5120)
            assignment["model.layers.0.self_attn.o_proj"] = "NVFP4"
        return self._source(td, tensors), assignment

    def _qkv_source(self, td: Path, fmt: str = "MXFP8_E4M3"):
        tensors = {
            f"model.layers.0.self_attn.{proj}": shape
            for proj, shape in self._QKV_SHAPES.items()
        }
        tensors["model.layers.0.self_attn.o_proj"] = (128, 5120)
        assignment = {name: fmt for name in tensors}
        return self._source(td, tensors), assignment

    def test_packed_moe_group_refuses_rather_than_ship_16bpp_experts(self):
        """A packed-expert unit whose `down_proj` is NVFP4-illegal is NOT
        rewritten: FP8_E4M3 is legal for every member, so a re-solve lands
        the unit on ~8 bpp, while coercing it here would ship the whole
        FusedMoE at 16 bpp -- num_experts x the per-Linear cost this branch
        is justified by, on every layer, because the dimension that made
        one member illegal is model-wide."""
        with tempfile.TemporaryDirectory() as td:
            src, assignment = self._packed_moe_source(Path(td))
            with patch.dict(
                enc.os.environ,
                {"PRISMAQUANT_TARGET_PROFILE": "vllm_packed_moe"},
            ):
                with self.assertRaises(ValueError) as ctx:
                    _coerce_runtime_legal_assignment(
                        src, assignment, Qwen3_5Profile()
                    )
        msg = str(ctx.exception)
        self.assertIn("packed_moe_experts", msg)
        self.assertIn("model.layers.0.mlp.experts", msg)
        self.assertIn("model.layers.0.mlp.experts.0.down_proj", msg)
        self.assertIn("group_divisibility", msg)
        # names the rung a re-solve can use, and what BF16 would have cost
        self.assertIn("FP8_E4M3", msg)
        self.assertIn("KiB", msg)
        self.assertIn("layer_config.json", msg)

    def test_fused_sibling_group_refuses_rather_than_ship_16bpp_qkv(self):
        """Same rule for merged columns: `k_proj`/`v_proj` at out_features
        48 are rejected by the MXFP8 kernel validator while `q_proj` at 128
        is fine. NVFP4 and FP8_E4M3 are legal for all three."""
        with tempfile.TemporaryDirectory() as td:
            src, assignment = self._qkv_source(Path(td))
            with patch.dict(
                enc.os.environ,
                {"PRISMAQUANT_TARGET_PROFILE": "vllm_packed_moe"},
            ):
                with self.assertRaises(ValueError) as ctx:
                    _coerce_runtime_legal_assignment(
                        src, assignment, Qwen3_5Profile()
                    )
        msg = str(ctx.exception)
        self.assertIn("fused_siblings", msg)
        self.assertIn("model.layers.0.self_attn.qkv_proj", msg)
        self.assertIn("model.layers.0.self_attn.k_proj", msg)
        self.assertIn("kernel_shape", msg)
        self.assertIn("NVFP4", msg)

    def _only_bf16_is_legal(self, illegal_for: set[str]):
        """Applicability stub: every emittable format is illegal on
        `illegal_for`, so BF16 is the only thing left for that unit and the
        guard has to coerce rather than refuse. Everything else stays
        legal, so the surrounding Linears keep their allocated format and
        `build_quantization_config` gets a REAL mixed config to gate."""
        real = enc.check_format_applicability

        def stub(shape, fmt, *, qname=None, target_profile=None, **kw):
            if qname in illegal_for:
                return type(real(shape, "BF16", qname=qname))(
                    False, "kernel_shape", f"stub: {fmt} illegal on {qname}"
                )
            return real(
                shape, fmt, qname=qname, target_profile=target_profile, **kw
            )

        return patch.object(enc, "check_format_applicability", stub)

    def test_whole_fused_group_coerced_when_bf16_is_the_only_option(self):
        with tempfile.TemporaryDirectory() as td:
            src, assignment = self._qkv_source(Path(td), fmt="NVFP4")
            with self._only_bf16_is_legal({"model.layers.0.self_attn.k_proj"}):
                out, coerced = _coerce_runtime_legal_assignment(
                    src, assignment, Qwen3_5Profile()
                )

        # ONE format across the whole merged-column group, and the
        # untouched Linear keeps what the allocator picked.
        for proj in self._QKV_SHAPES:
            self.assertEqual(out[f"model.layers.0.self_attn.{proj}"], "BF16")
        self.assertEqual(out["model.layers.0.self_attn.o_proj"], "NVFP4")

        rows = {row.name: row for row in coerced}
        self.assertEqual(
            set(rows),
            {f"model.layers.0.self_attn.{p}" for p in self._QKV_SHAPES},
        )
        for name, row in rows.items():
            self.assertEqual(row.serving_group,
                             "model.layers.0.self_attn.qkv_proj")
            self.assertEqual(row.serving_group_kind, "fused_siblings")
            self.assertEqual(
                set(row.serving_group_members),
                {f"model.layers.0.self_attn.{p}" for p in self._QKV_SHAPES},
            )
            self.assertEqual(row.trigger, "model.layers.0.self_attn.k_proj")
            self.assertGreater(row.delta_bytes, 0)
        # the trigger keeps its own verdict; the siblings say why they moved
        self.assertEqual(rows["model.layers.0.self_attn.k_proj"].reason,
                         "kernel_shape")
        self.assertEqual(rows["model.layers.0.self_attn.q_proj"].reason,
                         "serving_group_coherence")
        self.assertIn("k_proj",
                      rows["model.layers.0.self_attn.q_proj"].detail)

        # The invariant this exists to protect: the coerced assignment
        # passes the fused-coherence gate, and the fused module lands in
        # `ignore` as one unquantized unit.
        profile = Qwen3_5Profile()
        config = build_quantization_config(out, set(), profile=profile)
        vllm = profile.to_vllm_internal_name
        self.assertIn(
            vllm("model.layers.0.self_attn.qkv_proj"), config["ignore"]
        )
        # targets are anchored regexes; the quantized neighbour is still
        # nominated, and no member of the coerced unit is.
        targets = " ".join(
            target
            for group in config["config_groups"].values()
            for target in group["targets"]
        )
        for proj in self._QKV_SHAPES:
            self.assertNotIn(
                vllm(f"model.layers.0.self_attn.{proj}").replace(".", "[.]"),
                targets,
            )
        self.assertIn(
            vllm("model.layers.0.self_attn.o_proj").replace(".", "[.]"),
            targets,
        )

    def test_whole_packed_moe_group_coerced_when_bf16_is_the_only_option(self):
        with tempfile.TemporaryDirectory() as td:
            src, assignment = self._packed_moe_source(
                Path(td), dense_neighbour=True
            )
            experts = [n for n in assignment if ".experts." in n]
            trigger = "model.layers.0.mlp.experts.0.down_proj"
            with self._only_bf16_is_legal({trigger}):
                out, coerced = _coerce_runtime_legal_assignment(
                    src, assignment, Qwen3_5Profile()
                )
            # every projection of every expert in the unit, not just the
            # offending one -- vLLM picks one scheme for the whole FusedMoE
            self.assertEqual({out[name] for name in experts}, {"BF16"})
            self.assertEqual(out["model.layers.0.self_attn.o_proj"], "NVFP4")
            self.assertEqual({row.name for row in coerced}, set(experts))
            kinds = {row.serving_group_kind for row in coerced}
            self.assertEqual(kinds, {"fused_siblings+packed_moe_experts"})
            for row in coerced:
                self.assertIn("__packed_format__", row.serving_group)
                self.assertEqual(len(row.serving_group_members), len(experts))
                self.assertEqual(row.trigger, trigger)

            # the packed-MoE coherence gate ("FusedMoE ... has mixed states
            # across packed expert projections") is satisfied: one state for
            # the whole layer, ignored as a per-expert regex.
            config = build_quantization_config(
                out, set(), profile=Qwen3_5Profile()
            )
            self.assertTrue(
                any(
                    entry.startswith("re:")
                    and "mlp[.]experts" in entry
                    for entry in config["ignore"]
                ),
                config["ignore"],
            )

            # audit says group-level coercion, not per-Linear
            audit = _bf16_upgrade_audit(
                src, out, set(), coerced, Qwen3_5Profile()
            )
        self.assertEqual(
            audit["counts"],
            {"runtime_coerced_serving_group_from_nvfp4": len(experts)},
        )
        group_entry = audit["entries"][0]["serving_group"]
        self.assertEqual(group_entry["kind"], "fused_siblings+packed_moe_experts")
        self.assertEqual(group_entry["trigger"], trigger)
        self.assertEqual(len(group_entry["members"]), len(experts))

    def test_manifest_records_the_group_coercion(self):
        with tempfile.TemporaryDirectory() as td:
            src, assignment = self._qkv_source(Path(td), fmt="NVFP4")
            with self._only_bf16_is_legal({"model.layers.0.self_attn.k_proj"}):
                _out, coerced = _coerce_runtime_legal_assignment(
                    src, assignment, Qwen3_5Profile()
                )
        rows = {row["name"]: row for row in
                enc._runtime_coercion_manifest_rows(coerced)}
        self.assertEqual(len(rows), 3)
        row = rows["model.layers.0.self_attn.q_proj"]
        self.assertEqual(row["from"], "NVFP4")
        self.assertEqual(row["to"], "BF16")
        self.assertEqual(row["reason"], "serving_group_coherence")
        self.assertEqual(
            row["serving_group"]["key"], "model.layers.0.self_attn.qkv_proj"
        )
        self.assertEqual(
            row["serving_group"]["trigger"], "model.layers.0.self_attn.k_proj"
        )
        self.assertEqual(len(row["serving_group"]["members"]), 3)
        self.assertGreater(row["delta_bytes"], 0)
        # and the operator-facing report is impossible to miss
        report = enc._runtime_coercion_report(coerced)
        self.assertIn("SERVING-ATOMIC UNIT(S) COERCED TO BF16 IN FULL", report)
        self.assertIn("UPSTREAM REGRESSION", report)
        self.assertIn("model.layers.0.self_attn.qkv_proj", report)

    def test_single_member_coercion_is_what_the_gate_rejects(self):
        """The counterfactual, pinned: the per-Linear rewrite this change
        replaced produces exactly the mix `build_quantization_config`
        refuses. Without the group expansion the export dies here (with a
        wrong-model-profile diagnosis), which is why the coercion has to be
        group-aware rather than trust the gate to catch it."""
        mixed = {
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "model.layers.0.self_attn.k_proj": "BF16",
            "model.layers.0.self_attn.v_proj": "NVFP4",
        }
        with self.assertRaisesRegex(
            RuntimeError, "fused-sibling coherence violation"
        ):
            build_quantization_config(mixed, set(), profile=Qwen3_5Profile())

    def test_ungrouped_linear_still_coerces_alone(self):
        """No-op for the dense case: a Linear that is in no serving unit
        coerces exactly as before, and its neighbours are untouched."""
        with tempfile.TemporaryDirectory() as td:
            src = self._source(Path(td), {
                "model.layers.0.self_attn.o_proj": (128, 5120),
                "model.layers.1.self_attn.o_proj": (128, 5120),
            })
            with patch.dict(
                enc.os.environ,
                {"PRISMAQUANT_TARGET_PROFILE": "vllm_packed_moe"},
            ):
                out, coerced = _coerce_runtime_legal_assignment(
                    src,
                    {
                        "model.layers.0.self_attn.o_proj": "MXFP4",
                        "model.layers.1.self_attn.o_proj": "NVFP4",
                    },
                    Qwen3_5Profile(),
                )
            self.assertEqual(out, {
                "model.layers.0.self_attn.o_proj": "BF16",
                "model.layers.1.self_attn.o_proj": "NVFP4",
            })
            self.assertEqual(len(coerced), 1)
            self.assertEqual(
                tuple(coerced[0][:3]),
                ("model.layers.0.self_attn.o_proj", [128, 5120], "MXFP4"),
            )
            self.assertIsNone(coerced[0].serving_group)
            audit = _bf16_upgrade_audit(
                src, out, set(), coerced, Qwen3_5Profile()
            )
        self.assertEqual(audit["counts"], {"runtime_coerced_from_mxfp4": 1})
        self.assertNotIn("serving_group", audit["entries"][0])

    def test_unemittable_format_still_hard_fails_inside_a_serving_group(self):
        """Issue #27's hard error is not weakened by the group logic: a
        format with no `config_groups` scheme raises before any legality or
        grouping work, whether or not the Linear is serving-atomic."""
        with tempfile.TemporaryDirectory() as td:
            src, _ = self._qkv_source(Path(td))
            assignment = {
                "model.layers.0.self_attn.q_proj": "NVFP4",
                "model.layers.0.self_attn.k_proj": "NVFP4A16",
                "model.layers.0.self_attn.v_proj": "NVFP4",
            }
            with self.assertRaisesRegex(ValueError, "no compressed-tensors"):
                _coerce_runtime_legal_assignment(
                    src, assignment, Qwen3_5Profile()
                )

    def test_undeclared_packed_expert_role_is_fail_closed(self):
        """A profile that cannot name a packed expert's serving unit is a
        declaration gap. The raising name is absent from every unit, so
        "coerce the whole unit" would silently leave it behind and ship a
        mixed FusedMoE -- so the guard refuses instead of guessing."""
        class _RaisingProfile(Qwen3_5Profile):
            def packed_expert_format_group(self, qname):
                if qname.endswith("down_proj"):
                    raise KeyError("no role declared for down_proj")
                return super().packed_expert_format_group(qname)

        with tempfile.TemporaryDirectory() as td:
            src, assignment = self._packed_moe_source(Path(td))
            with self._only_bf16_is_legal(
                {"model.layers.0.mlp.experts.0.gate_proj"}
            ):
                with self.assertRaises(ValueError) as ctx:
                    _coerce_runtime_legal_assignment(
                        src, assignment, _RaisingProfile()
                    )
        msg = str(ctx.exception)
        self.assertIn("cannot verify serving-atomic coherence", msg)
        self.assertIn("could not name the serving unit", msg)
        self.assertIn("down_proj", msg)
        self.assertIn("projection_splits", msg)

    def test_passthrough_mismatch_escalates_like_any_other_illegality(self):
        """Issue #29 removed the passthrough exemption from group
        escalation. It existed only because the FP8_SOURCE verdict was an
        artifact of the missing `source_kind` (every FP8_SOURCE Linear read
        as illegal), so escalating it would have coerced whole units to
        BF16 on every FP8-source model. Now that the verdict is real, a
        passthrough mismatch inside a serving unit is escalated: FP8_SOURCE
        on a BF16-source q/k/v is refused, naming the quantized rungs a
        re-solve can use -- exactly the #28 policy."""
        with tempfile.TemporaryDirectory() as td:
            src, _ = self._qkv_source(Path(td))
            assignment = {
                "model.layers.0.self_attn.q_proj": "FP8_SOURCE",
                "model.layers.0.self_attn.k_proj": "FP8_SOURCE",
                "model.layers.0.self_attn.v_proj": "FP8_SOURCE",
            }
            with self.assertRaises(ValueError) as ctx:
                _coerce_runtime_legal_assignment(
                    src, assignment, Qwen3_5Profile()
                )
        msg = str(ctx.exception)
        self.assertIn("fused_siblings", msg)
        self.assertIn("model.layers.0.self_attn.qkv_proj", msg)
        self.assertIn("source_dtype_mismatch", msg)
        self.assertIn("source_kind='fp8'", msg)
        # names the rungs a re-solve can use, and says the fault is the
        # source precision (not a dimension)
        self.assertIn("FP8_E4M3", msg)
        self.assertIn("source precision", msg)


class TestPassthroughSourceIntegrityCoercion(unittest.TestCase):
    """Issue #29: the runtime-legality guard supplies `source_kind`.

    `PASSTHROUGH_SOURCE_REQUIREMENTS` makes FP8_SOURCE legal only where
    the source tensor is ALREADY fp8, so `check_format_applicability`
    needs the source dtype to judge it. The guard used to omit it, which
    made EVERY FP8_SOURCE Linear read `source_dtype_mismatch` and get
    rewritten to BF16 -- inert in the bytes (materialization copies the
    source fp8 verbatim; `_fp8_source_config_overlay` restores the
    config), but it filled every DSv4 / Hy3 / MiniMax manifest's
    `runtime_coercions` with demotions that never happened, so a real one
    was invisible in the crowd.
    """

    _FP8_OK_SHAPE = (512, 5120)  # out/in both divide 128 (fp8 block scale)

    def _source(
        self,
        td: Path,
        fp8: dict[str, tuple[int, int]] | None = None,
        bf16: dict[str, tuple[int, int]] | None = None,
    ) -> str:
        """A checkpoint with genuinely-fp8 and genuinely-bf16 Linears.

        fp8 entries get the `.weight_scale_inv` sibling that marks the
        128x128 block-scaled FP8 convention (`_build_fp8_source_map` and
        `_scan_source_dtype_manifest` both key off it), so the source
        dtype on disk is the real thing rather than a mock.
        """
        from safetensors.torch import save_file

        tensors: dict[str, torch.Tensor] = {}
        for base, (out_f, in_f) in (fp8 or {}).items():
            tensors[f"{base}.weight"] = torch.zeros(
                out_f, in_f, dtype=torch.bfloat16
            ).to(torch.float8_e4m3fn)
            tensors[f"{base}.weight_scale_inv"] = torch.ones(
                max(1, out_f // 128), max(1, in_f // 128), dtype=torch.float32
            )
        for base, (out_f, in_f) in (bf16 or {}).items():
            tensors[f"{base}.weight"] = torch.zeros(
                out_f, in_f, dtype=torch.bfloat16
            )
        shard = td / "model-00001-of-00001.safetensors"
        save_file(tensors, str(shard))
        with open(td / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {k: shard.name for k in tensors}}, f)
        return str(td)

    def _fp8_qkv(self, td: Path) -> tuple[str, dict[str, str]]:
        names = {
            f"model.layers.0.self_attn.{proj}": self._FP8_OK_SHAPE
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj")
        }
        src = self._source(td, fp8=names)
        return src, {name: "FP8_SOURCE" for name in names}

    def test_fp8_source_on_an_fp8_checkpoint_produces_no_coercion_row(self):
        """THE regression. A source-FP8 Linear allocated FP8_SOURCE is
        legal, so the assignment is untouched and `runtime_coercions` stays
        empty -- the manifest records what the exporter did, not what a
        missing argument made it look like."""
        with tempfile.TemporaryDirectory() as td:
            src, assignment = self._fp8_qkv(Path(td))
            out, coerced = _coerce_runtime_legal_assignment(
                src, dict(assignment), Qwen3_5Profile()
            )
            self.assertEqual(coerced, [])
            self.assertEqual(out, assignment)
            self.assertEqual(set(out.values()), {"FP8_SOURCE"})
            self.assertEqual(enc._runtime_coercion_report(coerced), "")
            self.assertEqual(enc._runtime_coercion_manifest_rows(coerced), [])
            # and the BF16 audit no longer counts them as coerced-to-BF16
            audit = _bf16_upgrade_audit(
                src, out, set(), coerced, Qwen3_5Profile()
            )
            self.assertEqual(audit["counts"], {})

            # The overlay's repair is now unnecessary for these names (it
            # stays for the BF16-assigned / pinned source-FP8 Linears whose
            # bytes materialization emits verbatim -- covered separately).
            _cfg, _bf16, overrides = enc._fp8_source_config_overlay(
                src, out, set(), Qwen3_5Profile()
            )
            self.assertEqual(overrides, set())

    def test_non_fp8_source_assigned_fp8_source_still_coerces_loudly(self):
        """The verdict this guard is FOR. A BF16-source Linear allocated
        FP8_SOURCE has no fp8 bytes to copy, so BF16 is the only
        representable answer -- as before #29, but now with the true byte
        delta (the emitted bytes really do change, 8.002 -> 16 bpp) and a
        report that says which invariant broke.

        Deliberately NOT a hard raise: this is the same shape of fault as
        every other per-Linear illegality the guard coerces, an artifact
        that ships today would newly fail export, and the sharp case (a
        serving unit with a legal quantized rung) already refuses via the
        #28 path."""
        dense = "model.layers.0.mlp.down_proj"
        with tempfile.TemporaryDirectory() as td:
            src = self._source(Path(td), bf16={dense: (5120, 512)})
            out, coerced = _coerce_runtime_legal_assignment(
                src, {dense: "FP8_SOURCE"}, Qwen3_5Profile()
            )
        self.assertEqual(out, {dense: "BF16"})
        self.assertEqual(len(coerced), 1)
        row = coerced[0]
        self.assertEqual(row.reason, "source_dtype_mismatch")
        self.assertIn("source_kind='fp8'", row.detail)
        self.assertIn("'bf16'", row.detail)
        self.assertIsNone(row.serving_group)
        # priced, where the exempted row used to carry delta_bytes=None
        self.assertEqual(
            row.delta_bytes,
            enc._bf16_coercion_delta_bytes([5120, 512], "FP8_SOURCE"),
        )
        self.assertGreater(row.delta_bytes, 0)
        report = enc._runtime_coercion_report(coerced)
        self.assertIn("PASSTHROUGH SOURCE MISMATCH", report)
        self.assertIn("PASSTHROUGH_SOURCE_REQUIREMENTS", report)
        manifest = enc._runtime_coercion_manifest_rows(coerced)
        self.assertEqual(manifest[0]["reason"], "source_dtype_mismatch")
        self.assertEqual(manifest[0]["from"], "FP8_SOURCE")
        self.assertEqual(manifest[0]["to"], "BF16")

    def test_passthrough_mismatch_coerces_the_whole_unit_when_bf16_is_all(self):
        """The other half of the escalation, with the exemption gone: when
        NO quantized format is legal for every member, the passthrough
        mismatch coerces the WHOLE unit (never one member -- that is the
        unservable mixed-scheme artifact) and the trigger row keeps its own
        source-dtype verdict."""
        real = enc.check_format_applicability

        def no_quantized_rung(shape, fmt, *, qname=None, target_profile=None,
                              **kw):
            """Every QUANTIZED format is illegal on k_proj, so the unit has
            no alternative rung; passthrough verdicts stay real."""
            if (qname == "model.layers.0.self_attn.k_proj"
                    and fmt not in enc.PASSTHROUGH_SOURCE_REQUIREMENTS):
                return type(real(shape, "BF16", qname=qname))(
                    False, "kernel_shape", f"stub: {fmt} illegal on {qname}"
                )
            return real(
                shape, fmt, qname=qname, target_profile=target_profile, **kw
            )

        qkv = [f"model.layers.0.self_attn.{p}"
               for p in ("q_proj", "k_proj", "v_proj")]
        with tempfile.TemporaryDirectory() as td:
            src = self._source(
                Path(td), bf16={name: self._FP8_OK_SHAPE for name in qkv}
            )
            with patch.object(
                enc, "check_format_applicability", no_quantized_rung
            ):
                out, coerced = _coerce_runtime_legal_assignment(
                    src, {name: "FP8_SOURCE" for name in qkv},
                    Qwen3_5Profile(),
                )
        self.assertEqual({out[name] for name in qkv}, {"BF16"})
        rows = {row.name: row for row in coerced}
        self.assertEqual(set(rows), set(qkv))
        for row in rows.values():
            self.assertEqual(
                row.serving_group, "model.layers.0.self_attn.qkv_proj"
            )
            self.assertEqual(row.serving_group_kind, "fused_siblings")
            self.assertGreater(row.delta_bytes, 0)
        self.assertEqual(
            {row.reason for row in rows.values()}, {"source_dtype_mismatch"}
        )

    def test_real_coercion_on_an_fp8_source_model_is_legible(self):
        """What the bogus rows were drowning out: on an FP8-source model a
        genuine shape coercion is now the ONLY row, next to an untouched
        FP8_SOURCE Linear."""
        fp8_ok = "model.layers.0.self_attn.o_proj"
        # 48 out_features is rejected by the MXFP8 kernel validator
        illegal = "model.layers.0.linear_attn.in_proj_a"
        with tempfile.TemporaryDirectory() as td:
            src = self._source(
                Path(td),
                fp8={fp8_ok: (5120, 512), illegal: (48, 5120)},
            )
            out, coerced = _coerce_runtime_legal_assignment(
                src,
                {fp8_ok: "FP8_SOURCE", illegal: "MXFP8_E4M3"},
                Qwen3_5Profile(),
            )
            self.assertEqual(out[fp8_ok], "FP8_SOURCE")
            self.assertEqual(out[illegal], "BF16")
            self.assertEqual(len(coerced), 1)
            self.assertEqual(coerced[0].name, illegal)
            self.assertEqual(coerced[0].reason, "kernel_shape")
            audit = _bf16_upgrade_audit(
                src, out, set(), coerced, Qwen3_5Profile()
            )
        self.assertEqual(
            audit["counts"], {"runtime_coerced_from_mxfp8_e4m3": 1}
        )

    def test_unemittable_format_still_hard_fails_on_an_fp8_source(self):
        """Issue #27's hard error is not weakened by the source-aware
        verdict: a format with no `config_groups` scheme raises before any
        legality work, on an FP8-source checkpoint whose other Linears are
        legitimately FP8_SOURCE."""
        with tempfile.TemporaryDirectory() as td:
            src = self._source(
                Path(td),
                fp8={
                    "model.layers.0.self_attn.o_proj": (5120, 512),
                    "model.layers.0.mlp.down_proj": (5120, 512),
                },
            )
            with self.assertRaisesRegex(ValueError, "no compressed-tensors"):
                _coerce_runtime_legal_assignment(
                    src,
                    {
                        "model.layers.0.self_attn.o_proj": "FP8_SOURCE",
                        "model.layers.0.mlp.down_proj": "NVFP4A16",
                    },
                    Qwen3_5Profile(),
                )

    def test_fp8_source_on_packed_experts_is_refused_by_the_profile(self):
        """Documented consequence of removing the exemption. FP8_SOURCE is
        absent from `vllm_packed_moe`'s packed-expert allow-list (only
        NVFP4/MXFP4/MXFP8_E4M3/FP8_E4M3/BF16 serve there), so on an
        FP8-source MoE an explicit FP8_SOURCE expert assignment is
        `profile_mismatch` and now escalates to the unit instead of being
        exempted into an inert BF16 rewrite. The allocator cannot produce
        this (`build_candidates` masks on the same profile), so reaching it
        means the allocation was solved under a different profile than it
        is being exported for -- which is exactly the case that must not
        ship, since vLLM's packed-MoE path has no FP8_SOURCE scheme."""
        experts = {
            f"model.layers.0.mlp.experts.{e}.{proj}": (512, 5120)
            for e in (0, 1)
            for proj in ("gate_proj", "up_proj", "down_proj")
        }
        with tempfile.TemporaryDirectory() as td:
            src = self._source(Path(td), fp8=experts)
            with self.assertRaises(ValueError) as ctx:
                _coerce_runtime_legal_assignment(
                    src, {name: "FP8_SOURCE" for name in experts},
                    Qwen3_5Profile(),
                )
        msg = str(ctx.exception)
        self.assertIn("packed_moe_experts", msg)
        self.assertIn("profile_mismatch", msg)
        self.assertIn("FP8_E4M3", msg)


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

    Default: legacy ``6/max_abs`` bytes (backwards-compatible — the
    generate_gparam convention is strongly artifact-dependent on served
    KL: 35B MoE -14.1%, 27B dense +37.5%, thin-calib LFM +5.8%; see the
    2026-07-02 audit C1 addendum). PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE=1
    opts into the compressed-tensors ``448*6/max_abs`` convention behind
    a per-artifact served A/B."""

    def test_default_is_legacy_6_over_amax(self):
        import torch
        from prismaquant.export_native_compressed import (
            compute_nvfp4_input_global_scale, _FP4_E2M1_MAX,
        )
        acts = torch.tensor([0.0, 1.5, -3.0, 2.0])
        s = compute_nvfp4_input_global_scale(acts)
        self.assertAlmostEqual(s, _FP4_E2M1_MAX / 3.0, places=5)

    def test_env_one_opts_into_fp8_range_convention(self):
        import os
        import torch
        from prismaquant.export_native_compressed import (
            compute_nvfp4_input_global_scale, _FP4_E2M1_MAX, _FP8_E4M3_MAX,
        )
        acts = torch.tensor([0.0, 1.5, -3.0, 2.0])
        key = "PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE"
        saved = os.environ.get(key)
        try:
            os.environ[key] = "1"
            s = compute_nvfp4_input_global_scale(acts)
        finally:
            if saved is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved
        # max_abs=3.0, scale = 448*6/3 = 896.0 (generate_gparam convention)
        self.assertAlmostEqual(
            s, _FP8_E4M3_MAX * _FP4_E2M1_MAX / 3.0, places=3)

    def test_matches_compressed_tensors_generate_gparam(self):
        """Oracle test: our input_global_scale must equal the installed
        compressed-tensors `generate_gparam` (the convention vLLM's
        CompressedTensorsW4A4Fp4 loads) to fp32 tolerance."""
        import torch
        try:
            from compressed_tensors.quantization.utils import generate_gparam
        except Exception as e:  # pragma: no cover - env without the lib
            self.skipTest(f"compressed_tensors not importable: {e}")
        from prismaquant.export_native_compressed import (
            compute_nvfp4_input_global_scale,
        )
        import os
        torch.manual_seed(0)
        acts = torch.randn(64, 128) * 3.7
        expected = generate_gparam(
            updated_min_val=acts.amin(),
            updated_max_val=acts.amax(),
        )
        key = "PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE"
        saved = os.environ.get(key)
        try:
            os.environ[key] = "1"
            ours = compute_nvfp4_input_global_scale(acts)
        finally:
            if saved is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = saved
        self.assertAlmostEqual(
            ours, float(expected.item()),
            delta=abs(float(expected.item())) * 1e-6,
        )

    def test_degenerate_all_zero_falls_back(self):
        import torch
        from prismaquant.export_native_compressed import (
            compute_nvfp4_input_global_scale, DEFAULT_INPUT_GLOBAL_SCALE,
        )
        acts = torch.zeros(100)
        s = compute_nvfp4_input_global_scale(acts)
        self.assertEqual(s, DEFAULT_INPUT_GLOBAL_SCALE)

    def test_legacy_formula_delegates_to_versioned_policy_owner(self):
        shared = enc._nvfp4_activation_contract
        policy = shared.LEGACY_INPUT_GLOBAL_SCALE_POLICY
        with patch.object(
            shared,
            "resolve_input_global_scale_policy",
            return_value=policy,
        ) as resolve, patch.object(
            shared,
            "input_global_scale_from_max_abs",
            return_value=7.25,
        ) as calibrate:
            self.assertEqual(
                enc._nvfp4_input_global_scale_from_max_abs(3.0),
                7.25,
            )
            resolve.assert_called_once_with()
            calibrate.assert_called_once_with(
                3.0,
                policy=policy,
                nonpositive_fallback=(
                    shared.UNCALIBRATED_INPUT_GLOBAL_SCALE
                ),
            )

    def test_legacy_default_resolution_delegates_with_explicit_opt_in(self):
        shared = enc._nvfp4_activation_contract
        saved = enc._INPUT_GLOBAL_SCALES
        enc._INPUT_GLOBAL_SCALES = {"layer.q_proj": 3.0}
        try:
            with patch.object(
                shared,
                "resolve_input_global_scale_value",
                return_value=2.5,
            ) as resolve:
                self.assertEqual(
                    enc._resolve_nvfp4_input_global_scale(
                        2.0,
                        target="layer.q_proj",
                    ),
                    2.5,
                )
                resolve.assert_called_once_with(
                    2.0,
                    target="layer.q_proj",
                    calibrated_scales={"layer.q_proj": 3.0},
                    allow_uncalibrated_fallback=True,
                )
        finally:
            enc._INPUT_GLOBAL_SCALES = saved

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

    def test_override_precedence_preserves_serialized_f32_value(self):
        weight = torch.randn(32, 32)
        saved = enc._INPUT_GLOBAL_SCALES
        try:
            enc._INPUT_GLOBAL_SCALES = {"foo.bar.q_proj": 3.14}
            out = enc._quantize_2d(
                weight,
                "NVFP4",
                linear_name="foo.bar.q_proj",
                input_global_scale_override=2.5,
            )
        finally:
            enc._INPUT_GLOBAL_SCALES = saved

        expected = torch.tensor([2.5], dtype=torch.float32)
        self.assertTrue(torch.equal(out["input_global_scale"], expected))
        self.assertTrue(torch.equal(
            out["input_global_scale"].view(torch.uint8),
            expected.view(torch.uint8),
        ))


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

    def test_do_no_harm_gate_failure_warns_and_counts(self):
        import os
        import torch
        from unittest import mock
        import prismaquant.export_native_compressed as m

        torch.manual_seed(912)
        W = torch.randn(8, 16) * 0.3
        X = torch.randn(16, 16)
        saved_stats = m._DO_NO_HARM_STATS.copy()
        saved_dnh = os.environ.get("PRISMAQUANT_DO_NO_HARM")
        saved_sweep = os.environ.get("PRISMAQUANT_GPTQ_DAMP_SWEEP")
        failure_count = None
        try:
            m._DO_NO_HARM_STATS.clear()
            os.environ["PRISMAQUANT_DO_NO_HARM"] = "1"
            os.environ["PRISMAQUANT_GPTQ_DAMP_SWEEP"] = "0"
            with (
                mock.patch.object(
                    m,
                    "_activation_col_importance_for_gptq",
                    side_effect=RuntimeError("boom"),
                ),
                mock.patch("builtins.print") as printed,
            ):
                out = m._quantize_2d(
                    W,
                    "NVFP4",
                    gptq_enabled=True,
                    cached_activations=X,
                    linear_name="demo.linear",
                )
                failure_count = m._DO_NO_HARM_STATS["NVFP4_failures"]
        finally:
            m._DO_NO_HARM_STATS.clear()
            m._DO_NO_HARM_STATS.update(saved_stats)
            if saved_dnh is None:
                os.environ.pop("PRISMAQUANT_DO_NO_HARM", None)
            else:
                os.environ["PRISMAQUANT_DO_NO_HARM"] = saved_dnh
            if saved_sweep is None:
                os.environ.pop("PRISMAQUANT_GPTQ_DAMP_SWEEP", None)
            else:
                os.environ["PRISMAQUANT_GPTQ_DAMP_SWEEP"] = saved_sweep

        self.assertIn("weight_packed", out)
        self.assertTrue(any(
            "[do-no-harm] WARN demo.linear NVFP4 gate failed" in str(call)
            for call in printed.call_args_list
        ))
        self.assertEqual(failure_count, 1)

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
        # These tests verify flag threading through the SWEPT
        # path, which is env-gated (default off since 2026-06-12).
        __import__('os').environ['PRISMAQUANT_GPTQ_DAMP_SWEEP'] = '1'
        self.addCleanup(lambda: __import__('os').environ.pop(
            'PRISMAQUANT_GPTQ_DAMP_SWEEP', None))
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
        # These tests verify flag threading through the SWEPT
        # path, which is env-gated (default off since 2026-06-12).
        __import__('os').environ['PRISMAQUANT_GPTQ_DAMP_SWEEP'] = '1'
        self.addCleanup(lambda: __import__('os').environ.pop(
            'PRISMAQUANT_GPTQ_DAMP_SWEEP', None))
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


class TestPerRoleGptqDamp(unittest.TestCase):
    """Per-role GPTQ damp research lever (PRISMAQUANT_GPTQ_DAMP_ROLES)."""

    def setUp(self):
        import os
        import prismaquant.export_native_compressed as m
        self.m = m
        self._saved = {
            k: os.environ.get(k)
            for k in ("PRISMAQUANT_GPTQ_DAMP_ROLES", "PRISMAQUANT_GPTQ_DAMP")
        }
        for k in self._saved:
            os.environ.pop(k, None)
        m._GPTQ_DAMP_ROLE_CACHE.clear()

    def tearDown(self):
        import os
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.m._GPTQ_DAMP_ROLE_CACHE.clear()

    def test_role_of_maps_known_linears(self):
        role = self.m._gptq_role_of
        self.assertEqual(role("model.layers.3.mlp.gate_proj"), "gate_up")
        self.assertEqual(role("model.layers.3.mlp.up_proj"), "gate_up")
        self.assertEqual(role("model.layers.3.mlp.down_proj"), "down")
        self.assertEqual(role("model.layers.0.self_attn.o_proj"), "o_proj")
        self.assertEqual(role("model.layers.0.self_attn.q_proj"), "qkv")
        self.assertEqual(role("model.layers.0.self_attn.k_proj"), "qkv")
        self.assertEqual(role("model.layers.0.self_attn.v_proj"), "qkv")
        self.assertEqual(role("model.embed_tokens"), "other")

    def test_unset_is_exact_noop(self):
        # No env => per-role resolver == the global fixed default (1.0), so the
        # production render is preserved bit-for-bit.
        self.assertEqual(self.m._resolve_gptq_fixed_damp(), 1.0)
        for q in ("model.layers.1.mlp.gate_proj",
                  "model.layers.1.mlp.down_proj",
                  "model.layers.1.self_attn.q_proj"):
            self.assertEqual(self.m._resolve_gptq_damp_for_role(q), 1.0)

    def test_role_table_applied_with_fallback(self):
        import os
        os.environ["PRISMAQUANT_GPTQ_DAMP_ROLES"] = \
            "qkv=1.0,o_proj=1.0,gate_up=0.3,down=3.0"
        r = self.m._resolve_gptq_damp_for_role
        self.assertEqual(r("model.layers.2.mlp.gate_proj"), 0.3)
        self.assertEqual(r("model.layers.2.mlp.up_proj"), 0.3)
        self.assertEqual(r("model.layers.2.mlp.down_proj"), 3.0)
        self.assertEqual(r("model.layers.2.self_attn.o_proj"), 1.0)
        self.assertEqual(r("model.layers.2.self_attn.q_proj"), 1.0)
        # 'other' role is unlisted => falls back to the global default (1.0).
        self.assertEqual(r("model.embed_tokens"), 1.0)

    def test_unlisted_role_falls_back_to_global_override(self):
        # Listed roles win; unlisted roles take the PRISMAQUANT_GPTQ_DAMP base.
        import os
        os.environ["PRISMAQUANT_GPTQ_DAMP"] = "0.5"
        os.environ["PRISMAQUANT_GPTQ_DAMP_ROLES"] = "gate_up=0.3"
        r = self.m._resolve_gptq_damp_for_role
        self.assertEqual(r("model.layers.2.mlp.gate_proj"), 0.3)
        self.assertEqual(r("model.layers.2.mlp.down_proj"), 0.5)
        self.assertEqual(r("model.layers.2.self_attn.q_proj"), 0.5)

    def test_malformed_entries_ignored(self):
        import os
        os.environ["PRISMAQUANT_GPTQ_DAMP_ROLES"] = \
            "gate_up=0.3,down=oops,,=1.0,qkv=-2,o_proj=2.0"
        table = self.m._parse_gptq_damp_roles(
            os.environ["PRISMAQUANT_GPTQ_DAMP_ROLES"])
        self.assertEqual(table, {"gate_up": 0.3, "o_proj": 2.0})


class TestExportMatchRenderScaleRuleM19(unittest.TestCase):
    """M19: NVFP4 export re-derive honors the render's recorded scale rule."""

    def setUp(self):
        import os
        import prismaquant.export_native_compressed as m
        self.m = m
        self._saved = os.environ.get(
            "PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE")
        os.environ.pop("PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE", None)

    def tearDown(self):
        import os
        if self._saved is None:
            os.environ.pop("PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE", None)
        else:
            os.environ["PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE"] = self._saved

    def test_match_helper_returns_recorded_rule_by_default(self):
        class _C:
            levers = {"nvfp4_scale_rule": "joint_mse"}
        self.assertEqual(
            self.m._export_match_render_scale_rule(_C()), "joint_mse")

    def test_match_helper_gate_off_is_none(self):
        import os
        os.environ["PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE"] = "0"

        class _C:
            levers = {"nvfp4_scale_rule": "joint_mse"}
        self.assertIsNone(self.m._export_match_render_scale_rule(_C()))

    def test_match_helper_no_recorded_rule_is_none(self):
        class _C:
            levers = {}
        self.assertIsNone(self.m._export_match_render_scale_rule(_C()))
        self.assertIsNone(self.m._export_match_render_scale_rule(object()))

    def test_temporary_scale_rule_sets_and_restores(self):
        m = self.m
        prev = m._NVFP4_SCALE_RULE
        with m._temporary_export_nvfp4_scale_rule("joint_mse"):
            self.assertEqual(
                m._NVFP4_SCALE_RULE, m.resolve_nvfp4_scale_rule("joint_mse"))
        self.assertEqual(m._NVFP4_SCALE_RULE, prev)
        with m._temporary_export_nvfp4_scale_rule(None):  # falsy = no-op
            self.assertEqual(m._NVFP4_SCALE_RULE, prev)

    def test_rule_actually_changes_packed_scales(self):
        # joint_mse explores extra per-group scale levels ({6,4,...}) and picks
        # the min-MSE one, so on a generic Gaussian weight its chosen group
        # scales differ from static_6's fixed max->6 — proving the rule plumbs
        # through the export re-derive (not just the helper).
        m = self.m
        torch.manual_seed(0)
        w = torch.randn(8, 16)
        with m._temporary_export_nvfp4_scale_rule("static_6"):
            _, s6, _ = m.quantize_dequantize_nvfp4(w, group_size=16)
        with m._temporary_export_nvfp4_scale_rule("joint_mse"):
            _, sj, _ = m.quantize_dequantize_nvfp4(w, group_size=16)
        self.assertFalse(
            torch.equal(s6, sj),
            "joint_mse should select different group scales than static_6")


class TestMtpCacheCoveragePreflight(unittest.TestCase):
    def test_missing_mtp_entries_diagnosed_at_attach_time(self):
        # QC M17: non-BF16 mtp.* with an attached cache must fail with the
        # producer-absence contract named, not a generic missing-keys error
        # (and never reach the late in-memory materialization gate).
        from prismaquant import export_native_compressed as enc

        keys, missing = enc._production_cache_expected_keys({
            "model.layers.0.self_attn.q_proj": "NVFP4",
            "mtp.layers.0.mlp.gate_proj": "NVFP4",
        })
        mtp_missing = [k for k in missing if str(k[0]).startswith("mtp.")]
        self.assertTrue(
            mtp_missing,
            "mtp.* entries must surface in the attach-time coverage check",
        )


class TestPackedExpertMatchRenderScaleRule(unittest.TestCase):
    """M2 (2026-07-02 audit): the packed-expert re-pack honors the render's
    RECORDED NVFP4 scale rule (the dense M19 wrap, lifted to packed experts).

    Cache levers record joint_mse; the export-entry env default is static_6.
    The re-derived packed-expert bytes must match a direct joint_mse
    re-quantization, not static_6."""

    def test_packed_expert_repack_uses_recorded_joint_mse_rule(self):
        from prismaquant.production_weight_cache import ProductionWeightCache

        torch.manual_seed(1234)
        root = _tiny_qwen_packed_root()
        experts = root.model.language_model.layers[0].mlp.experts
        live_prefix = "model.language_model.layers.0.mlp.experts"

        # "Cached renders": arbitrary bf16-storable tensors (values don't
        # need to be grid-valued for the rule pin — the codes just have to
        # follow the recorded rule on re-derive).
        cached_gup = (torch.randn_like(experts.gate_up_proj) * 0.2).float()
        cached_down = (torch.randn_like(experts.down_proj) * 0.2).float()

        cache = ProductionWeightCache(
            weights={
                (f"{live_prefix}.gate_up_proj", "NVFP4"): cached_gup,
                (f"{live_prefix}.down_proj", "NVFP4"): cached_down,
            },
            levers={"gptq": True, "nvfp4_scale_rule": "joint_mse"},
            activation_max_abs={
                f"{live_prefix}.gate_up_proj": 3.0,
                f"{live_prefix}.down_proj": 3.0,
            },
        )

        assignment = {
            "model.layers.0.mlp.experts.gate_up_proj": "NVFP4",
            "model.layers.0.mlp.experts.down_proj": "NVFP4",
        }

        saved_cache = enc._PRODUCTION_WEIGHT_CACHE
        saved_rule = enc._NVFP4_SCALE_RULE
        saved_flags = dict(enc._ACT_AWARE_FLAGS)
        try:
            enc._PRODUCTION_WEIGHT_CACHE = cache
            # Export-entry default rule: static_6 (the M2 trigger).
            enc._NVFP4_SCALE_RULE = enc.NVFP4_SCALE_RULE_STATIC_6
            for k in enc._ACT_AWARE_FLAGS:
                enc._ACT_AWARE_FLAGS[k] = False
            tensors, hist = enc._materialize_tensors_inmemory(
                root,
                assignment,
                bf16_passthrough=set(),
                profile=Qwen3_5Profile(),
            )
        finally:
            enc._PRODUCTION_WEIGHT_CACHE = saved_cache
            enc._NVFP4_SCALE_RULE = saved_rule
            enc._ACT_AWARE_FLAGS.clear()
            enc._ACT_AWARE_FLAGS.update(saved_flags)

        # down_proj expert 0: single projection -> per-Linear global.
        got = tensors[f"{live_prefix}.0.down_proj.weight_packed"]
        with enc._temporary_export_nvfp4_scale_rule("joint_mse"):
            wp_joint, _ws, _wg = enc.quantize_dequantize_nvfp4(
                cached_down[0], group_size=16)
        with enc._temporary_export_nvfp4_scale_rule("static_6"):
            wp_static, _ws6, _wg6 = enc.quantize_dequantize_nvfp4(
                cached_down[0], group_size=16)
        # The pin must be discriminating: the two rules disagree here.
        self.assertFalse(torch.equal(wp_joint, wp_static))
        self.assertTrue(torch.equal(got, wp_joint))

        # gate_up expert 0: gate/up halves share a joint global. The
        # export computes it from the SOURCE packed param (pre-existing
        # contract) — but now under the SAME recorded rule as the
        # re-derive of the cached dequant.
        src_gup = experts.gate_up_proj.detach().float()
        rows = src_gup.shape[1] // 2
        with enc._temporary_export_nvfp4_scale_rule("joint_mse"):
            joint = torch.stack([
                enc.compute_nvfp4_global_real(
                    src_gup[0][:rows], group_size=16),
                enc.compute_nvfp4_global_real(
                    src_gup[0][rows:], group_size=16),
            ]).max()
            wp_gate_joint, _s, _g = enc.quantize_dequantize_nvfp4(
                cached_gup[0][:rows], group_size=16,
                global_real_override=joint)
        got_gate = tensors[f"{live_prefix}.0.gate_proj.weight_packed"]
        self.assertTrue(torch.equal(got_gate, wp_gate_joint))


class TestGptqDeadColumnHandling(unittest.TestCase):
    """§3.16 (2026-07-02 audit): dead columns (diag(H)<=0) are detected
    BEFORE damping — the old check ran after damping and never fired —
    and their weights are NOT zeroed (serving-safe deviation from
    reference GPTQ)."""

    def test_dead_activation_column_survives_nvfp4_gptq(self):
        from prismaquant.export_native_compressed import (
            _gptq_obs_rounding_nvfp4,
        )
        torch.manual_seed(21)
        W = torch.randn(16, 32) * 0.3
        X = torch.randn(64, 32)
        dead_col = 7
        X[:, dead_col] = 0.0
        out = _gptq_obs_rounding_nvfp4(W, X, group_size=16)
        # The unexercised column is quantized, not destroyed.
        self.assertGreater(float(out[:, dead_col].abs().max()), 0.0)
        self.assertTrue(torch.isfinite(out).all())

    def test_dead_activation_column_survives_fp8_and_mxfp4_gptq(self):
        from prismaquant.export_native_compressed import (
            _gptq_obs_rounding_fp8_like,
            _gptq_obs_rounding_mxfp4,
        )
        torch.manual_seed(22)
        W = torch.randn(16, 64) * 0.3
        X = torch.randn(96, 64)
        dead_col = 11
        X[:, dead_col] = 0.0
        _q, _s, dq_fp8 = _gptq_obs_rounding_fp8_like(
            W, X, fmt="FP8_E4M3")
        self.assertGreater(float(dq_fp8[:, dead_col].abs().max()), 0.0)
        _q4, _s4, dq_mx4 = _gptq_obs_rounding_mxfp4(W, X, group_size=32)
        self.assertGreater(float(dq_mx4[:, dead_col].abs().max()), 0.0)


class TestCholeskyFallbackKeepsJointGlobal(unittest.TestCase):
    """§3.18 (2026-07-02 audit): when the GPTQ Cholesky fails under
    joint_scale_opt, the RTN fallback must carry the JSO-optimized
    tensor global instead of silently recomputing a per-tensor default."""

    def test_fallback_uses_jso_optimized_global(self):
        from unittest import mock
        torch.manual_seed(911)
        W = torch.randn(16, 32) * 0.3
        X = torch.randn(48, 32)

        # Expected: replicate the pre-Cholesky global computation of
        # _gptq_obs_rounding_nvfp4 under joint_scale_opt.
        W32 = W.to(torch.float32)
        grouped = W32.reshape(16, 32 // 16, 16)
        s_g = enc._select_nvfp4_group_scales(
            grouped, scale_rule=enc.NVFP4_SCALE_RULE_JOINT_MSE)
        base = (s_g.amax() / enc.FP8_E4M3_MAX).clamp_min(1e-12)
        opt = enc._optimize_nvfp4_joint_global_real(
            W32, group_size=16, base_global_real=base)
        expected = enc._rtn_dequant_nvfp4(
            W, group_size=16, global_real_override=opt)

        with mock.patch(
                "torch.linalg.cholesky", side_effect=RuntimeError("boom")):
            got = enc._gptq_obs_rounding_nvfp4(
                W, X, group_size=16, joint_scale_opt=True)

        torch.testing.assert_close(got, expected)
