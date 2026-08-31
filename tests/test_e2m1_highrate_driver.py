from pathlib import Path
import subprocess
import sys
import textwrap


DRIVER = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/e2m1_highrate.py"
)


def test_active_corpus_loader_is_isolated_from_locked_hull_package():
    source = DRIVER.read_text()

    assert "load_active_glm_corpus_bound(" in source
    assert "read_bound_json(PUBLISHED)" in source
    assert "from prismaquant.trellis_bf16_corpus import" not in source


def test_glm_high_rate_plan_is_explicit_and_does_not_replace_default():
    source = DRIVER.read_text()

    assert '"scaffold": BF16_RATES' in source
    assert '"high": NEW_RATES' in source
    assert 'default="scaffold"' in source
    assert 'rate_plan = GLM_RATE_PLANS[args.glm_rate_plan]' in source
    assert '"glm_rate_plan": args.glm_rate_plan' in source


def test_final_receipt_is_published_only_after_complete_result():
    source = DRIVER.read_text()

    assert 'partial_path = args.out.with_name(args.out.name + ".partial")' in source
    assert 'if args.out.exists()' in source
    assert 'expected_tensors=expected_tensors' in source
    assert source.count("_atomic_json(") >= 2
    assert 'publish_file_no_replace(partial_path, args.out)' in source
    assert "args.out.write_text" not in source
    assert "saved_cell = resumed_out.get(name)" in source
    assert "_require_e2_replay_match(name, saved_cell, cell)" in source


def test_future_result_binds_active_and_frozen_source_closures():
    source = DRIVER.read_text()

    assert '"schema": "trellis.e2m1_highrate.v3"' in source
    assert '"corpus_binding": corpus_binding' in source
    assert '"manifest_sha256"' in source
    assert '"importance_value_sha256"' in source
    assert '"driver_sha256"' in source
    assert '"isolated_loader_sha256"' in source
    assert '"active_corpus_reader_sha256"' in source
    assert '"snapshot_tree_sha256": H.snapshot_tree_sha256()' in source
    assert '"source_sha256": H.source_hashes()' in source
    assert '"bf16_ladder_sha256"' in source
    assert 'Path(module.MANIFEST)' in source
    assert 'Path(module.INPUT)' in source
    assert '"input_sha256"' in source
    assert '"control_sha256"' in source
    assert '_claim_identity(args, prepared["corpus_binding"])' in source
    assert '_corpus_binding(args) != binding' in source
    assert 'required published control' in source
    assert 'expected_tensors[name]' in source
    assert 'expected_controls[name]' in source
    assert "NVFP4_COMPARATOR._rtn_dequant_nvfp4" in source
    assert 'from prismaquant import format_registry' not in source
    assert '"nvfp4_scalar_comparator"' in source
    assert '"nvfp4_activation_contract"' in source


def test_dirty_frozen_nvfp4_comparator_source_is_rejected():
    program = textwrap.dedent(
        f"""
        import importlib.util, sys
        from pathlib import Path
        path = Path({str(DRIVER)!r})
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("e2m1_dirty_source", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = module.file_sha256
        for claim_source in (
            module.NVFP4_COMPARATOR, module.NVFP4_ACTIVATION_CONTRACT,
        ):
            dirty = Path(claim_source.__file__).resolve()
            def dirty_digest(candidate):
                if Path(candidate).resolve() == dirty:
                    return "0" * 64
                return original(candidate)
            module.file_sha256 = dirty_digest
            try:
                module._active_source_identity()
            except SystemExit as exc:
                assert "comparator source drifted" in str(exc)
            else:
                raise AssertionError(
                    f"dirty frozen NVFP4 source {{dirty}} was accepted"
                )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], text=True, capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_frozen_direct_nvfp4_comparator_matches_prior_registry_route():
    program = textwrap.dedent(
        f"""
        import importlib.util, sys, torch
        from pathlib import Path
        path = Path({str(DRIVER)!r})
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("e2m1_nvfp4_parity", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from prismaquant import format_registry
        weight = torch.linspace(-3.0, 3.0, 512).reshape(2, 256).to(torch.bfloat16)
        prior = format_registry.get_format("NVFP4").quantize_dequantize(weight)
        direct = module.NVFP4_COMPARATOR._rtn_dequant_nvfp4(
            weight, group_size=16
        ).to(weight.dtype)
        assert torch.equal(prior, direct)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], text=True, capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_self_bound_v3_partial_rejects_fake_full_prefix_in_isolated_process(tmp_path):
    program = textwrap.dedent(
        f"""
        import importlib.util, json, sys
        from pathlib import Path
        path = Path({str(DRIVER)!r})
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("e2m1_resume_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source_identity = module._active_source_identity()
        assert len(source_identity["frozen_hull"]["snapshot_tree_sha256"]) == 64
        assert source_identity["frozen_hull"]["source_sha256"]
        assert set(source_identity["frozen_hull"]["imported_codec_modules"]) == {{
            "hull_sweep", "weight_codec", "common", "plane", "schedule",
            "trellis_formats", "nvfp4_scalar_comparator",
            "nvfp4_activation_contract",
        }}
        receipt = {{
            "schema": "trellis.e2m1_highrate.v3",
            "started_at_unix_s": 1.0,
            "publication_identity_sha256": "a" * 64,
            "rate_plan": [3.25],
        }}
        cell = {{
            "shape": [2, 3], "numel": 6, "population": "dense",
            "weighted_energy": 1.0, "plain_energy": 1.0,
            "two_tier_plane_sha256": "b" * 64, "arms": {{}},
            "unreachable_rungs": [], "control": {{"status": "uncontrolled"}},
        }}
        partial = Path({str(tmp_path / 'run.partial')!r})
        module._atomic_json(
            partial,
            module._checkpoint_document(receipt, {{"tensor-a": cell}}, partial=True),
        )
        try:
            module._resume_partial(
                partial, receipt=receipt,
                expected_tensors={{
                    "tensor-a": {{"shape": [2, 3], "population": "dense"}},
                    "tensor-b": {{"shape": [2, 3], "population": "dense"}},
                }},
                expected_controls={{"tensor-a": {{}}, "tensor-b": {{}}}},
            )
        except SystemExit as exc:
            assert "contract differs" in str(exc)
        else:
            raise AssertionError("fake empty-arm prefix was accepted")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_bf16_missing_published_control_refuses_before_gpu(tmp_path):
    program = textwrap.dedent(
        f"""
        import importlib.util, sys
        from pathlib import Path
        from types import SimpleNamespace
        path = Path({str(DRIVER)!r})
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("e2m1_preflight_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = Path({str(tmp_path / 'bf16-manifest.json')!r})
        manifest.write_text("{{}}")
        class FakeLadder:
            MANIFEST = manifest
            @staticmethod
            def load_corpus():
                entry = {{
                    "source_weight_shape": [2, 256],
                    "importance_shape": [256],
                }}
                return {{}}, ["tensor-a"], {{"tensor-a": entry}}
        module._bf16_ladder_module = lambda: FakeLadder
        module.BF16_PUBLISHED = Path({str(tmp_path / 'missing-control.json')!r})
        args = SimpleNamespace(
            corpus="bf16", limit=None, glm_manifest=None,
            glm_rate_plan="scaffold",
        )
        try:
            module._prepare_campaign(args)
        except SystemExit as exc:
            assert "required published control" in str(exc)
        else:
            raise AssertionError("missing BF16 control reached campaign setup")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], text=True, capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_e2_resume_requires_exact_regenerated_claims_except_timing(tmp_path):
    program = textwrap.dedent(
        f"""
        import copy, importlib.util, sys
        from pathlib import Path
        path = Path({str(DRIVER)!r})
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("e2m1_replay_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        regenerated = {{
            "two_tier_plane_sha256": "a" * 64,
            "weighted_energy": 10.0,
            "plain_energy": 20.0,
            "arms": {{
                "tcq_v1@2.0": {{
                    "encode_seconds": 2.0,
                    "weighted_sse": 1.0,
                    "weighted_nsse": 0.1,
                    "weighted_snr_db": 10.0,
                    "subset_split": {{"2": {{"trellis_wsse": 1.0}}}},
                }}
            }},
        }}
        timing_only = copy.deepcopy(regenerated)
        timing_only["arms"]["tcq_v1@2.0"]["encode_seconds"] = 999.0
        module._require_e2_replay_match("tensor-a", timing_only, regenerated)

        invented = copy.deepcopy(regenerated)
        arm = invented["arms"]["tcq_v1@2.0"]
        arm.update({{
            "weighted_sse": 2.0,
            "weighted_nsse": 0.2,
            "weighted_snr_db": 6.9897000433601875,
        }})
        arm["subset_split"]["2"]["trellis_wsse"] = 2.0
        try:
            module._require_e2_replay_match("tensor-a", invented, regenerated)
        except SystemExit as exc:
            assert "deterministic replay" in str(exc)
        else:
            raise AssertionError("coherently invented E2 metrics were reused")

        false_plane = copy.deepcopy(regenerated)
        false_plane["two_tier_plane_sha256"] = "b" * 64
        try:
            module._require_e2_replay_match("tensor-a", false_plane, regenerated)
        except SystemExit as exc:
            assert "deterministic replay" in str(exc)
        else:
            raise AssertionError("arbitrary E2 plane hash was reused")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], text=True, capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_bf16_nonfinite_published_control_refuses_in_prepare(tmp_path):
    program = textwrap.dedent(
        f"""
        import importlib.util, json, math, sys
        from pathlib import Path
        from types import SimpleNamespace
        path = Path({str(DRIVER)!r})
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("e2m1_control_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = Path({str(tmp_path / 'bf16-manifest.json')!r})
        manifest.write_text("{{}}")
        class FakeLadder:
            MANIFEST = manifest
            @staticmethod
            def load_corpus():
                entry = {{
                    "source_weight_shape": [2, 256],
                    "importance_shape": [256],
                }}
                return {{}}, ["tensor-a"], {{"tensor-a": entry}}
        module._bf16_ladder_module = lambda: FakeLadder
        control = Path({str(tmp_path / 'corrupt-control.json')!r})
        control.write_text(json.dumps({{
            "cells": {{
                "tensor-a": {{
                    "shape": [2, 256], "numel": 512,
                    "weighted_energy": 1.0, "plain_energy": 1.0,
                    "arms": {{"tcq_two_tier@2.0": {{
                        "weighted_sse": float("nan")
                    }}}},
                }}
            }}
        }}))
        module.BF16_PUBLISHED = control
        called = False
        def reject(arm, **kwargs):
            global called
            called = True
            assert math.isnan(arm["weighted_sse"])
            raise module.CheckpointContractError("weighted_sse must be finite")
        module.validate_e2_published_control_arm = reject
        args = SimpleNamespace(
            corpus="bf16", limit=None, glm_manifest=None,
            glm_rate_plan="scaffold",
        )
        try:
            module._prepare_campaign(args)
        except ValueError as exc:
            assert not called
            assert "non-finite JSON constant" in str(exc)
        except SystemExit as exc:
            assert called
            assert "invalid published control" in str(exc)
            assert "refusing before GPU work" in str(exc)
        else:
            raise AssertionError("nonfinite BF16 control passed preflight")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program], text=True, capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
