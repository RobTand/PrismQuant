"""Source- and hardware-bound serve facts used by the RTX 4090 gate."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.serve_fingerprint as fingerprint
from prismaquant.validate_rtx4090_fp8_cb import (
    RTX4090FP8CBValidationError,
    create_rtx4090_artifact_content_receipt,
    rtx4090_serve_environment,
    rtx4090_serve_environment_allowlist,
    validate_rtx4090_artifact_content_receipt,
    validate_rtx4090_artifact_content_receipt_portable,
)


_GOOD_WRAPPER = """
import torch

def build(compiled_ptr, backend, options):
    return torch.compile(
        compiled_ptr,
        fullgraph=True,
        dynamic=False,
        backend=backend,
        options=options,
    )
"""
_VLLM_COMMIT = "b" * 40


def _one_tensor_artifact(root: Path) -> tuple[Path, bytes]:
    payload = b"\x01\x02\x03\x04"
    header = json.dumps(
        {
            "a.weight": {
                "dtype": "U8",
                "shape": [len(payload)],
                "data_offsets": [0, len(payload)],
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = struct.pack("<Q", len(header)) + header + payload
    weight = root / "model.safetensors"
    weight.write_bytes(raw)
    (root / "quant_config.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "weight_content_manifest": {
                        "schema": "prismaquant.weight_content_manifest/1",
                        "algorithm": "sha256",
                        "files": {
                            weight.name: {
                                "bytes": len(raw),
                                "sha256": hashlib.sha256(raw).hexdigest(),
                            }
                        },
                    },
                    "tensor_payload_identity": {
                        "tensor_sha256": {
                            "a.weight": hashlib.sha256(payload).hexdigest(),
                        }
                    },
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return weight, raw


def _direct_url(*, repository=fingerprint.VLLM_REPOSITORY, commit=_VLLM_COMMIT):
    return {
        "url": repository,
        "vcs_info": {
            "vcs": "git",
            "requested_revision": commit,
            "commit_id": commit,
        },
    }


def test_rtx4090_environment_projection_inherits_candidate_registry_inputs():
    candidates = {
        "PRISMAQUANT_CB_BF16_SWIZZLE",
        "PRISMAQUANT_CB_FP4V2_DENSE_R2",
    }
    assert candidates <= set(fingerprint.SERVER_ENV_ALLOWLIST)
    assert rtx4090_serve_environment_allowlist() == (
        fingerprint.SERVER_ENV_ALLOWLIST
    )
    expected = rtx4090_serve_environment({
        "commit": "a" * 40,
        "version": "0.9.1",
        "wheel_sha256": "b" * 64,
    })
    assert expected["PRISMAQUANT_CB_FP4V2_DENSE_R2"] == "0"
    assert "PRISMAQUANT_CB_BF16_SWIZZLE" not in expected


def test_wrapper_contract_requires_one_literal_fullgraph_call():
    assert fingerprint._torch_compile_wrapper_contract(_GOOD_WRAPPER) == {
        "direct_torch_compile_calls": 1,
        "fullgraph": True,
        "dynamic": False,
        "backend_explicit": True,
    }

    for bad in (
        _GOOD_WRAPPER.replace("fullgraph=True", "fullgraph=False"),
        _GOOD_WRAPPER.replace("dynamic=False", "dynamic=True"),
        _GOOD_WRAPPER + "\ntorch.compile(lambda: None, fullgraph=True, "
        "dynamic=False, backend='inductor')\n",
    ):
        with pytest.raises(ValueError):
            fingerprint._torch_compile_wrapper_contract(bad)


class _Distribution:
    def __init__(self, root: Path):
        self.root = root
        self.metadata = {"Name": "vllm"}
        self.version = "0.test"
        self.files = (
            Path("vllm/__init__.py"),
            Path("vllm/compilation/wrapper.py"),
            Path("vllm-0.test.dist-info/direct_url.json"),
            Path("vllm-0.test.dist-info/RECORD"),
        )

    def locate_file(self, item: Path) -> Path:
        return self.root / item


def _record_digest(raw: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode(
        "ascii"
    ).rstrip("=")


def _fake_distribution(
    tmp_path: Path,
    source: str = _GOOD_WRAPPER,
    *,
    direct_url=None,
):
    root = tmp_path / "site"
    package = root / "vllm"
    wrapper = package / "compilation" / "wrapper.py"
    wrapper.parent.mkdir(parents=True)
    init = package / "__init__.py"
    init.write_text("__version__ = '0.test'\n", encoding="utf-8")
    wrapper.write_text(source, encoding="utf-8")
    raw = wrapper.read_bytes()
    digest = _record_digest(raw)
    dist_info = root / "vllm-0.test.dist-info"
    dist_info.mkdir()
    direct_path = dist_info / "direct_url.json"
    direct_path.write_text(
        json.dumps(
            direct_url if direct_url is not None else _direct_url(),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    direct_raw = direct_path.read_bytes()
    (dist_info / "RECORD").write_text(
        "vllm/compilation/wrapper.py,sha256="
        f"{digest},{len(raw)}\n"
        "vllm-0.test.dist-info/direct_url.json,sha256="
        f"{_record_digest(direct_raw)},{len(direct_raw)}\n",
        encoding="utf-8",
    )
    return _Distribution(root), init, package


def _strict_pin(distribution: _Distribution) -> dict[str, str]:
    record = distribution.root / "vllm-0.test.dist-info/RECORD"
    return {
        "schema": fingerprint.VLLM_RUNTIME_PIN_SCHEMA,
        "repository": fingerprint.VLLM_REPOSITORY,
        "commit": _VLLM_COMMIT,
        "version": "0.test",
        "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
    }


def _install_distribution(monkeypatch, distribution, init, package):
    monkeypatch.setattr(
        fingerprint.importlib_metadata,
        "distribution",
        lambda name: distribution,
    )
    monkeypatch.setattr(
        fingerprint.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            origin=str(init), submodule_search_locations=[str(package)]
        ),
    )


def test_vllm_provenance_binds_record_origin_and_ast(tmp_path, monkeypatch):
    distribution, init, package = _fake_distribution(tmp_path)
    _install_distribution(monkeypatch, distribution, init, package)

    receipt = fingerprint.vllm_compilation_provenance()

    assert receipt["schema"] == (
        "prismaquant.vllm_compilation_provenance/1"
    )
    assert receipt["version"] == "0.test"
    assert receipt["compile_contract"]["fullgraph"] is True
    assert len(receipt["wrapper_identity"]["sha256"]) == 64
    assert len(receipt["identity_sha256"]) == 64
    assert "runtime_pin" not in receipt


def test_default_manifest_does_not_enforce_strict_vllm_wrapper(monkeypatch):
    """Historical/default evidence must accept its existing vLLM package."""

    pid = 123
    monkeypatch.setattr(
        fingerprint,
        "vllm_compilation_provenance",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("strict vLLM provenance reached the default profile")
        ),
    )
    monkeypatch.setattr(
        fingerprint, "residency_scan", lambda pids: ([], [pid], [])
    )
    monkeypatch.setattr(
        fingerprint,
        "host_identity",
        lambda: {"hostname": "legacy", "boot_id": "boot"},
    )
    monkeypatch.setattr(
        fingerprint,
        "process_identities",
        lambda pids, boot_id: [],
    )
    monkeypatch.setattr(
        fingerprint,
        "server_environment_snapshot",
        lambda pids, names: {"values": {}, "processes": []},
    )
    monkeypatch.setattr(
        fingerprint, "process_tcp_listeners", lambda pids: []
    )
    monkeypatch.setattr(
        fingerprint, "listener_binding", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        fingerprint,
        "gpu_identity",
        lambda: {
            "gpu_name": None,
            "gpu_uuid": None,
            "driver_version": None,
            "compute_capability": None,
            "gpu_compute_capabilities": [],
            "gpu_count": 0,
        },
    )
    monkeypatch.setattr(
        fingerprint, "package_versions", lambda: {"vllm": "0.6.0"}
    )
    monkeypatch.setattr(fingerprint, "gridbook_runtime_pin", lambda: None)

    manifest = fingerprint.collect_manifest(
        pids=[pid],
        launch_argv=["vllm", "serve", "/legacy"],
        source="measure",
    )

    assert manifest["package_versions"]["vllm"] == "0.6.0"
    assert "vllm_compilation_provenance" not in manifest


def test_strict_vllm_provenance_binds_official_vcs_and_record(
    tmp_path, monkeypatch,
):
    distribution, init, package = _fake_distribution(tmp_path)
    _install_distribution(monkeypatch, distribution, init, package)
    pin = _strict_pin(distribution)

    receipt = fingerprint.vllm_compilation_provenance(pin)

    assert receipt["runtime_pin"] == pin
    assert receipt["direct_url"] == _direct_url()
    assert receipt["record_identity"]["sha256"] == pin["record_sha256"]
    assert Path(receipt["direct_url_path"]).name == "direct_url.json"
    assert Path(receipt["record_path"]).name == "RECORD"


@pytest.mark.parametrize(
    "direct_url",
    (
        _direct_url(repository="https://github.com/example/vllm.git"),
        _direct_url(commit="c" * 40),
    ),
)
def test_strict_vllm_provenance_rejects_fork_or_wrong_commit(
    tmp_path, monkeypatch, direct_url,
):
    distribution, init, package = _fake_distribution(
        tmp_path, direct_url=direct_url
    )
    _install_distribution(monkeypatch, distribution, init, package)

    with pytest.raises(ValueError, match="exact official pinned VCS commit"):
        fingerprint.vllm_compilation_provenance(_strict_pin(distribution))


def test_strict_vllm_provenance_rejects_missing_direct_url(
    tmp_path, monkeypatch,
):
    distribution, init, package = _fake_distribution(tmp_path)
    pin = _strict_pin(distribution)
    distribution.files = tuple(
        item for item in distribution.files if item.name != "direct_url.json"
    )
    (distribution.root / "vllm-0.test.dist-info/direct_url.json").unlink()
    _install_distribution(monkeypatch, distribution, init, package)

    with pytest.raises(ValueError, match="exactly one direct_url.json"):
        fingerprint.vllm_compilation_provenance(pin)


def test_strict_vllm_provenance_rejects_modified_record(
    tmp_path, monkeypatch,
):
    distribution, init, package = _fake_distribution(tmp_path)
    pin = _strict_pin(distribution)
    record = distribution.root / "vllm-0.test.dist-info/RECORD"
    with record.open("a", encoding="utf-8") as handle:
        handle.write("unrelated.py,,\n")
    _install_distribution(monkeypatch, distribution, init, package)

    with pytest.raises(ValueError, match="differs from the exact runtime pin"):
        fingerprint.vllm_compilation_provenance(pin)


def test_vllm_provenance_rejects_record_or_import_shadow(tmp_path, monkeypatch):
    distribution, init, package = _fake_distribution(tmp_path)
    monkeypatch.setattr(
        fingerprint.importlib_metadata,
        "distribution",
        lambda name: distribution,
    )
    monkeypatch.setattr(
        fingerprint.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            origin=str(tmp_path / "shadow" / "vllm" / "__init__.py"),
            submodule_search_locations=[str(tmp_path / "shadow" / "vllm")],
        ),
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        fingerprint.vllm_compilation_provenance()

    monkeypatch.setattr(
        fingerprint.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            origin=str(init), submodule_search_locations=[str(package)]
        ),
    )
    wrapper = distribution.root / "vllm/compilation/wrapper.py"
    wrapper.write_text(_GOOD_WRAPPER + "# post-install mutation\n")
    with pytest.raises(ValueError, match="differs from its RECORD"):
        fingerprint.vllm_compilation_provenance()


def test_gpu_identity_records_compute_capability_without_cuda_context(
    monkeypatch,
):
    monkeypatch.setattr(
        fingerprint.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="NVIDIA GeForce RTX 4090, GPU-one, 590.00, 8.9\n"
        ),
    )

    identity = fingerprint.gpu_identity()

    assert identity["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert identity["compute_capability"] == [8, 9]
    assert identity["gpu_compute_capabilities"] == [[8, 9]]
    assert identity["gpu_count"] == 1


def test_rtx4090_receipt_reads_weight_once_then_replays_only_stats(
    tmp_path, monkeypatch,
):
    weight, raw = _one_tensor_artifact(tmp_path)

    receipt = create_rtx4090_artifact_content_receipt(tmp_path)

    assert receipt["source"] == "verified_read"
    assert receipt["content_read_passes"] == 1
    assert receipt["content_bytes_read"] == len(raw)
    assert receipt["files"][weight.name]["sha256"] == hashlib.sha256(
        raw
    ).hexdigest()

    import prismaquant.shipcard as shipcard

    monkeypatch.setattr(
        shipcard.os,
        "read",
        lambda *_args, **_kwargs: pytest.fail(
            "receipt replay must not reread weight bytes"
        ),
    )
    assert validate_rtx4090_artifact_content_receipt(
        tmp_path, receipt
    ) == receipt
    assert validate_rtx4090_artifact_content_receipt_portable(
        tmp_path, receipt
    ) == receipt


def test_rtx4090_receipt_is_stat_bound_and_closed(tmp_path):
    weight, _raw = _one_tensor_artifact(tmp_path)
    receipt = create_rtx4090_artifact_content_receipt(tmp_path)

    changed = weight.stat()
    os.utime(
        weight,
        ns=(changed.st_atime_ns, changed.st_mtime_ns + 1_000_000),
    )
    with pytest.raises(
        RTX4090FP8CBValidationError,
        match="changed after content receipt",
    ):
        validate_rtx4090_artifact_content_receipt(tmp_path, receipt)

    malformed = json.loads(json.dumps(receipt))
    malformed["content_bytes_read"] -= 1
    with pytest.raises(
        RTX4090FP8CBValidationError,
        match="did not read every container byte once",
    ):
        validate_rtx4090_artifact_content_receipt_portable(
            tmp_path, malformed
        )


def test_rtx4090_in_container_preflight_is_one_pass_and_no_clobber(
    tmp_path, monkeypatch,
):
    _weight, raw = _one_tensor_artifact(tmp_path)
    contract = tmp_path / "runtime-contract.json"
    contract.write_text("{}", encoding="utf-8")
    out = tmp_path / "run" / "receipt.json"
    out.parent.mkdir()

    import prismaquant.validate_rtx4090_fp8_cb as validator

    metadata_calls = []
    monkeypatch.setattr(
        validator,
        "validate_rtx4090_artifact_metadata",
        lambda model_dir, *, runtime_contract: metadata_calls.append(
            (Path(model_dir), runtime_contract)
        ),
    )
    args = SimpleNamespace(
        model_dir=str(tmp_path),
        runtime_contract=str(contract),
        out=str(out),
    )

    assert fingerprint._cmd_rtx4090_artifact_preflight(args) == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert metadata_calls == [(tmp_path, {})]
    assert receipt["content_read_passes"] == 1
    assert receipt["content_bytes_read"] == len(raw)
    assert out.stat().st_mode & 0o777 == 0o600

    with pytest.raises(ValueError, match="receipt must not exist"):
        fingerprint._cmd_rtx4090_artifact_preflight(args)
