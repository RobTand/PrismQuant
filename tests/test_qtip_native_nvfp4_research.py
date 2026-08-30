import importlib
import json
import os
from pathlib import Path
import threading
import pytest
import torch

from prismaquant import export_native_compressed as enc
from prismaquant import format_registry as fr

M = importlib.import_module("research.qtip_native_nvfp4_2026-08-30.native_nvfp4_ldlq")


def fixture():
    g = torch.Generator().manual_seed(20260830)
    w = torch.randn(12, 32, generator=g) * 0.1
    x = torch.randn(96, 32, generator=g); x[:, 0] *= 4; x[:, 17] *= 0.2
    return w, x


def direct(w, x=None):
    with M.fixed_contract():
        return enc._quantize_2d(
            w, "NVFP4", input_global_scale_override=1.0,
            gptq_enabled=x is not None, scale_sweep_enabled=False,
            static_act_order_enabled=x is not None, joint_scale_opt_enabled=x is not None,
            cached_activations=x, act_clip_threshold=None,
            linear_name="research.qtip_native_nvfp4.one_linear" if x is not None else None)


def test_controls_reproduce_existing_codec():
    w, x = fixture(); a, b = M.rtn_arm(w), M.gptq_jso_arm(w, x)
    da, db = direct(w), direct(w, x)
    for name in M.FIELDS:
        assert torch.equal(a.fields[name], da[name]); assert torch.equal(b.fields[name], db[name])


def test_qtip_arm_is_deterministic_and_every_terminal_is_native():
    w, x = fixture(); a, b = M.qtip_native_arm(w, x), M.qtip_native_arm(w, x)
    assert len(a.terminal_blocks) == 2
    assert all(v["legal_native_nvfp4"] for v in a.terminal_blocks)
    for name in M.FIELDS: assert torch.equal(a.fields[name], b.fields[name])
    assert torch.equal(a.reconstruction, M.decode_fields(a.fields))


def test_reverse_recurrence_matches_pinned_qtip_buffered_orientation():
    generator = torch.Generator().manual_seed(20260831)
    weight = torch.randn(12, 64, generator=generator) * 0.1
    activations = torch.randn(96, 64, generator=generator)
    arm = M.qtip_native_arm(weight, activations)
    source = weight.float()
    _x, hessian, _damp = M.damped_hessian(activations, 64, source.device)
    lower = M.qtip_block_unit_lower(hessian)
    with M.fixed_contract():
        global_real = enc.nvfp4_global_real(
            source, group_size=M.GROUP, scale_rule=M.SCALE_RULE,
            snapped_scale_scoring=False, joint_scale_levels=M.SCALE_LEVELS,
        )

    # Independent row-transposed transcription of pinned QTIP LDLQ's
    # two-level loop, including prod_cache feedback across 32-column buffers.
    wr_t = source.T.contiguous()
    hat_t = torch.zeros_like(wr_t)
    prod_cache = torch.zeros_like(wr_t)
    block, buffer_columns = M.GROUP, 32
    buffer_blocks = buffer_columns // block
    for current in range(64 // block, 0, -buffer_blocks):
        start = block * (current - buffer_blocks)
        stop = block * current
        b_wr = wr_t[start:stop]
        b_hat = hat_t[start:stop]
        b_lower = lower[start:stop].contiguous()
        b_prod = prod_cache[start:stop]
        for index in reversed(range(buffer_blocks)):
            first, last = block * index, block * (index + 1)
            target_t = (
                b_wr[first:last]
                + b_lower[last:, start + first:start + last].T
                @ (b_wr[last:] - b_hat[last:])
                + b_prod[first:last]
            )
            terminal = M._native_rtn(target_t.T, global_real)
            b_hat[first:last] = terminal.reconstruction.T
        prod_cache += b_lower.T @ (b_wr - b_hat)
        hat_t[start:stop] = b_hat
    assert torch.equal(arm.reconstruction, hat_t.T)


def test_standard_fields_and_exact_payload_bpw():
    w, x = fixture(); arm = M.qtip_native_arm(w, x)
    assert M.validate_fields(arm.fields) == tuple(w.shape)
    expected = w.numel() // 2 + w.numel() // 16 + 8
    payload = M.payload_accounting(arm.fields)
    assert payload["bytes"] == expected
    assert payload["bits_per_weight"] == pytest.approx(expected * 8 / w.numel(), abs=0)


def test_comparison_registers_no_format_and_has_matched_bpw():
    w, x = fixture(); before = set(fr.REGISTRY); report, arms = M.compare_one_linear(w, x)
    assert set(fr.REGISTRY) == before and not any("QTIP" in k for k in fr.REGISTRY)
    assert report["scope"] == "research_only_one_linear_no_production_registration"
    assert set(report["arms"]) == set(arms)
    assert len({v["serialized"]["bits_per_weight"] for v in report["arms"].values()}) == 1
    assert report["native_nvfp4_contract"]["C2_is_exhaustive_e4m3_scale_byte_search"] is False
    assert "seven_level_scale_heuristic" in M.ARM_NAMES[-1]


def test_validation_rejects_qtip_or_nonstandard_fields():
    w, _ = fixture(); fields = dict(M.rtn_arm(w).fields)
    fields["trellis"] = torch.zeros(1, dtype=torch.int16)
    with pytest.raises(ValueError, match="exactly"): M.validate_fields(fields)
    fields.pop("trellis"); fields["weight_scale"] = fields["weight_scale"].float()
    with pytest.raises(ValueError, match="float8_e4m3fn"): M.validate_fields(fields)


def test_block_factor_matches_pinned_qtip_formula():
    _, x = fixture(); _xp, h, _d = M.damped_hessian(x, 32, torch.device("cpu"))
    lower = M.qtip_block_unit_lower(h)
    # Independent transcription of pinned QTIP math_utils.block_LDL: Cholesky
    # block-column right-multiplied by the inverse diagonal Cholesky block.
    chol = torch.linalg.cholesky(h)
    expected = chol.clone()
    for first in range(0, 32, 16):
        last = first + 16
        expected[:, first:last] = chol[:, first:last] @ torch.linalg.inv(
            chol[first:last, first:last])
        expected[first:last, first:last] = 0
    assert torch.allclose(lower, expected, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(lower[:16, :16]) == 0
    assert torch.count_nonzero(lower[16:, 16:]) == 0
    assert torch.count_nonzero(lower[:16, 16:]) == 0


def test_checkout_head_parser_needs_no_git_executable(tmp_path: Path):
    dotgit = tmp_path / ".git"; dotgit.mkdir()
    (dotgit / "HEAD").write_text(M.QTIP_PINNED_COMMIT + "\n")
    assert M._checkout_head_without_git(tmp_path) == M.QTIP_PINNED_COMMIT


def test_checkout_head_parser_resolves_worktree_common_refs(tmp_path: Path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    common = tmp_path / "repo.git"
    gitdir = common / "worktrees" / "checkout"
    gitdir.mkdir(parents=True)
    (checkout / ".git").write_text(f"gitdir: {gitdir}\n")
    (gitdir / "HEAD").write_text("ref: refs/heads/research\n")
    (gitdir / "commondir").write_text("../..\n")
    loose = common / "refs" / "heads" / "research"
    loose.parent.mkdir(parents=True)
    loose.write_text(M.QTIP_PINNED_COMMIT + "\n")
    assert M._checkout_head_without_git(checkout) == M.QTIP_PINNED_COMMIT


def test_atomic_publication_refuses_clobber(tmp_path: Path):
    output = tmp_path / "receipt.json"
    M._publish_no_clobber(output, lambda temp: temp.write_text("first\n"))
    with pytest.raises(FileExistsError):
        M._publish_no_clobber(output, lambda temp: temp.write_text("second\n"))
    assert output.read_text() == "first\n"
    assert output.stat().st_mode & 0o777 == 0o644
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_atomic_publication_uses_collision_safe_temporaries(tmp_path: Path):
    output = tmp_path / "receipt.json"
    barrier = threading.Barrier(2)
    errors = []

    def publish(value):
        try:
            def writer(path):
                path.write_text(value)
                barrier.wait(timeout=5)
            M._publish_no_clobber(output, writer)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=publish, args=(value,)) for value in ("A", "B")]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert output.read_text() in {"A", "B"}
    assert len(errors) == 1 and isinstance(errors[0], FileExistsError)
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_publication_plan_rejects_duplicates_and_escapes(tmp_path: Path):
    root = tmp_path / "result"
    output = root / "receipt.json"
    with pytest.raises(ValueError, match="unique"):
        M.validate_publication_plan(root, [output, output])
    with pytest.raises(ValueError, match="escapes"):
        M.validate_publication_plan(root, [tmp_path / "outside.json"])
    resolved, relative = M.validate_publication_plan(root, [output])
    assert resolved == root.resolve()
    assert relative[output.resolve()] == "receipt.json"


def test_calibration_manifest_is_canonically_bound(tmp_path: Path):
    manifest = {
        "schema": M.CALIBRATION_SCHEMA,
        "dataset": "unit-test",
        "capture_precision": "BF16",
        "calibration_hash": "1" * 32,
        "nsamples": 8,
        "seqlen": 512,
        "seed": 42,
    }
    manifest["identity_sha256"] = M._canonical_sha256(manifest)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(manifest))
    bound = M.validate_calibration_manifest(path)
    assert bound["contract"] == manifest
    changed = dict(manifest, seed=43)
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity_sha256 mismatch"):
        M.validate_calibration_manifest(path)


def test_fixed_contract_pins_and_restores_all_quality_state(monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_GPTQ_BLOCK_SIZE", "7")
    old_flags = dict(enc._ACT_AWARE_FLAGS)
    enc._ACT_AWARE_FLAGS.update({key: True for key in enc._ACT_AWARE_FLAGS})
    try:
        with M.fixed_contract():
            assert os.environ["PRISMAQUANT_GPTQ_BLOCK_SIZE"] == "128"
            assert os.environ["PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_GRID"] == "5"
            assert not any(enc._ACT_AWARE_FLAGS.values())
            assert torch.get_float32_matmul_precision() == "highest"
            assert torch.are_deterministic_algorithms_enabled()
        assert os.environ["PRISMAQUANT_GPTQ_BLOCK_SIZE"] == "7"
        assert enc._ACT_AWARE_FLAGS == {key: True for key in old_flags}
    finally:
        enc._ACT_AWARE_FLAGS.clear()
        enc._ACT_AWARE_FLAGS.update(old_flags)


def test_container_identity_requires_immutable_digest():
    digest = "sha256:" + "a" * 64
    assert M.validate_container_identity(digest) == digest
    assert M.validate_container_identity("repo/name@" + digest).endswith(digest)
    with pytest.raises(ValueError, match="sha256"):
        M.validate_container_identity("repo/name:mutable-tag")


def test_v2_cli_publishes_receipt_last_with_bound_members(tmp_path: Path, monkeypatch):
    weight, activations = fixture()
    weight_path, activation_path = tmp_path / "weight.pt", tmp_path / "activations.pt"
    torch.save(weight, weight_path)
    torch.save(activations, activation_path)
    calibration = {
        "schema": M.CALIBRATION_SCHEMA,
        "dataset": "unit-test",
        "capture_precision": "FP32",
        "calibration_hash": "2" * 32,
        "nsamples": 8,
        "seqlen": 512,
        "seed": 42,
    }
    calibration["identity_sha256"] = M._canonical_sha256(calibration)
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    monkeypatch.setattr(
        M, "validate_prismaquant_checkout",
        lambda path, commit: {"commit": commit, "source_sha256": {"unit": "3" * 64}},
    )
    monkeypatch.setattr(
        M, "validate_qtip_checkout",
        lambda path: {"commit": M.QTIP_PINNED_COMMIT, "source_sha256": {}},
    )
    root = tmp_path / "v2-result"
    output, fields, profile = root / "receipt.v2.json", root / "fields", root / "profile"
    argv = [
        "--weight", str(weight_path), "--activations", str(activation_path),
        "--device", "cpu", "--output", str(output),
        "--artifacts-dir", str(fields), "--profile-dir", str(profile),
        "--publication-root", str(root),
        "--durable-root-uri", "sparky:/durable/v2-result",
        "--host", "sparky", "--container-identity", "sha256:" + "a" * 64,
        "--model-id", "unit/model", "--calibration-manifest", str(calibration_path),
        "--prismaquant-checkout", str(tmp_path),
        "--prismaquant-commit", "4" * 40,
        "--qtip-checkout", str(tmp_path),
    ]
    assert M.main(argv) == 0
    receipt = json.loads(output.read_text())
    assert receipt["schema"] == M.SCHEMA
    assert receipt["publication"]["semantics"].endswith("receipt_is_commit_marker")
    assert len(receipt["publication"]["members_published_before_commit_marker"]) == 5
    assert receipt["publication"]["commit_marker"]["durable_uri"].endswith(
        "/receipt.v2.json"
    )
    assert output.stat().st_mode & 0o777 == 0o644
    assert all(path.stat().st_mode & 0o777 == 0o644 for path in fields.glob("*.safetensors"))
    with pytest.raises(FileExistsError, match="fresh empty"):
        M.main(argv)
