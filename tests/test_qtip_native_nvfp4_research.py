import importlib
from pathlib import Path
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


def test_atomic_publication_refuses_clobber(tmp_path: Path):
    output = tmp_path / "receipt.json"
    M._publish_no_clobber(output, lambda temp: temp.write_text("first\n"))
    with pytest.raises(FileExistsError):
        M._publish_no_clobber(output, lambda temp: temp.write_text("second\n"))
    assert output.read_text() == "first\n"
    assert not list(tmp_path.glob(".*.tmp.*"))
