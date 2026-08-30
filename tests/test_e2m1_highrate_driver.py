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

    assert "load_active_glm_corpus(REPO_ROOT, args.glm_manifest)" in source
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
    assert '_resume_partial(partial_path, receipt=receipt, names=names)' in source
    assert source.count("_atomic_json(") >= 2
    assert 'publish_file_no_replace(partial_path, args.out)' in source
    assert "args.out.write_text" not in source


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


def test_self_bound_v3_partial_resumes_exact_prefix_in_isolated_process(tmp_path):
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
            "trellis_formats",
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
        resumed, started = module._resume_partial(
            partial, receipt=receipt, names=["tensor-a", "tensor-b"]
        )
        assert resumed == {{"tensor-a": cell}}
        assert started == 1.0
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
