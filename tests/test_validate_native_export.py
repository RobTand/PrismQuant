import json
import sys
import types

import pytest

from prismaquant.validate_native_export import (
    _flashinfer_runtime_package,
    _resolve_validation_target_profile,
    _speculative_config_uses_embedded_mtp,
    maybe_upgrade_flashinfer,
    require_native_lane_artifact,
)


def test_speculative_config_embedded_mtp_detection():
    assert _speculative_config_uses_embedded_mtp({"method": "mtp"})
    assert _speculative_config_uses_embedded_mtp({"method": "qwen3_5_mtp"})
    assert _speculative_config_uses_embedded_mtp({"method": "qwen3_next_mtp"})

    assert not _speculative_config_uses_embedded_mtp({"method": "ngram"})
    assert not _speculative_config_uses_embedded_mtp({"method": "draft_model"})
    assert not _speculative_config_uses_embedded_mtp({})


def test_flashinfer_runtime_package_comes_from_serving_profile():
    version, packages, env = _flashinfer_runtime_package("vllm_packed_moe")

    assert version == "0.6.8.post1"
    assert packages == ("flashinfer-python", "flashinfer-cubin")
    assert env["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"


@pytest.fixture
def fake_flashinfer(monkeypatch):
    """Install a stub `flashinfer` module and record any pip call."""
    calls = []
    monkeypatch.setattr(
        "prismaquant.validate_native_export.subprocess.check_call",
        lambda cmd, *a, **k: calls.append(cmd),
    )

    def install(version):
        mod = types.ModuleType("flashinfer")
        mod.__version__ = version
        monkeypatch.setitem(sys.modules, "flashinfer", mod)
        return calls

    return install


@pytest.mark.parametrize(
    "installed, pinned, should_install",
    [
        # The bug: the serving container ships NEWER than the profile pin, and
        # `== version` made that look wrong. It downgraded a container that had
        # just served the artifact cleanly and vLLM 0.26 died on the missing
        # `set_autotune_process_group`. Measured 2026-08-14, Qwen3.8-27B gate.
        ("0.6.18", "0.6.8.post1", False),
        # Same version, both spellings.
        ("0.6.8.post1", "0.6.8.post1", False),
        ("0.6.8", "0.6.8.post1", False),
        # A post-release must not read as older than its own base version.
        ("0.6.9.post2", "0.6.9", False),
        # Genuinely too old: the pin's original purpose (an image that cannot
        # dispatch the NVFP4 MoE backend on Blackwell) still upgrades.
        ("0.6.7", "0.6.8.post1", True),
        ("0.5.20", "0.6.0", True),
    ],
)
def test_flashinfer_pin_is_a_floor_and_never_downgrades(
    fake_flashinfer, installed, pinned, should_install
):
    calls = fake_flashinfer(installed)
    maybe_upgrade_flashinfer(pinned)

    assert bool(calls) is should_install, (
        f"installed={installed} pinned={pinned}: "
        f"{'expected an upgrade' if should_install else 'must not touch it'}"
    )
    if should_install:
        assert any(f"flashinfer-python=={pinned}" in part for part in calls[0])


def test_flashinfer_absent_still_installs(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "prismaquant.validate_native_export.subprocess.check_call",
        lambda cmd, *a, **k: calls.append(cmd),
    )
    monkeypatch.setitem(sys.modules, "flashinfer", None)  # import -> ImportError

    maybe_upgrade_flashinfer("0.6.8.post1")

    assert calls, "no flashinfer at all must still install the pinned version"


def test_validation_target_profile_defaults_from_model_config(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}'
    )

    assert _resolve_validation_target_profile(tmp_path, None) == "vllm_packed_moe"
    assert _resolve_validation_target_profile(tmp_path, "research") == "research"


# ---------------------------------------------------------------------------
# The gate's lane (RobTand/prismaquant#119 half 1)
# ---------------------------------------------------------------------------
def _write_config(model_dir, quant_method=None, nested=False):
    cfg = {"model_type": "qwen3"}
    if quant_method is not None:
        qc = {"quant_method": quant_method, "config_groups": {}}
        if nested:
            cfg["text_config"] = {"quantization_config": qc}
        else:
            cfg["quantization_config"] = qc
    (model_dir / "config.json").write_text(json.dumps(cfg))


def test_a_tessera_artifact_is_refused_before_vllm_loads(tmp_path):
    """`validate_native_export` is the NATIVE lane's load gate: it builds
    `LLM(..., quantization="compressed-tensors")`, so pointing it at a
    Tessera artifact must refuse with the lane named -- not die inside vLLM
    on bytes it was never going to dispatch."""
    _write_config(tmp_path, "tessera")
    with pytest.raises(SystemExit, match="tessera"):
        require_native_lane_artifact(tmp_path)


def test_a_nested_tessera_quant_config_is_refused_too(tmp_path):
    """Multimodal checkpoints nest `quantization_config` under `text_config`;
    the lane check must read where the loader reads."""
    _write_config(tmp_path, "tessera", nested=True)
    with pytest.raises(SystemExit, match="tessera"):
        require_native_lane_artifact(tmp_path)


def test_a_compressed_tensors_artifact_passes_the_lane_check(tmp_path):
    _write_config(tmp_path, "compressed-tensors")
    assert require_native_lane_artifact(tmp_path) == "compressed-tensors"


def test_an_unquantized_checkpoint_keeps_its_legacy_behavior(tmp_path):
    """No declared `quant_method` is not a foreign lane: refusing here would
    change what a dense smoke does today, which is not this issue's bargain."""
    _write_config(tmp_path, None)
    assert require_native_lane_artifact(tmp_path) is None
