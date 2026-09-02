import json
import os
import struct
from copy import deepcopy

import pytest
import torch
from safetensors.torch import save_file

import prismaquant.shipcard as shipcard
from prismaquant.export_nvfp4_cb_streaming import _StreamWriter
from prismaquant.nvfp4_cb_footprint import (
    _safetensors_tensor_payload_sha256,
)
from prismaquant.shipcard import (
    build_weight_content_manifest,
    capture_attested_safetensors_write_receipt,
    safetensors_content_receipt_manifest,
    validate_safetensors_content_receipt,
    verify_safetensors_content_once,
)


def _single_artifact(root):
    path = root / "model.safetensors"
    save_file(
        {
            "a.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
            "b.weight": torch.arange(8, dtype=torch.uint8),
        },
        path,
        metadata={"format": "pt"},
    )
    ledger = _safetensors_tensor_payload_sha256(
        path, ["a.weight", "b.weight"]
    )
    return (
        build_weight_content_manifest(root),
        ledger,
        {"a.weight": path.name, "b.weight": path.name},
    )


def _verify(root, manifest, ledger, tensor_to_file):
    return verify_safetensors_content_once(
        root,
        expected_weight_manifest=manifest,
        expected_tensor_sha256=ledger,
        expected_tensor_to_file=tensor_to_file,
    )


def _flip_payload_byte(path, tensor_name):
    with path.open("r+b") as handle:
        (header_length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_length))
        data_start = 8 + header_length
        tensor_start = int(header[tensor_name]["data_offsets"][0])
        handle.seek(data_start + tensor_start)
        original = handle.read(1)
        handle.seek(data_start + tensor_start)
        handle.write(bytes([original[0] ^ 0x01]))
        handle.flush()
        os.fsync(handle.fileno())


def _write_raw_safetensors(path, header, data):
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + data)


def _scan_raw_safetensors(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        return shipcard._verify_open_safetensors_fd(
            fd,
            name=path.name,
            initial_stat=os.fstat(fd),
        )
    finally:
        os.close(fd)


def test_one_pass_receipt_binds_container_and_tensor_content(tmp_path, monkeypatch):
    manifest, ledger, tensor_to_file = _single_artifact(tmp_path)
    calls = []
    original = shipcard._verify_open_safetensors_fd

    def counted(*args, **kwargs):
        calls.append(kwargs["name"])
        return original(*args, **kwargs)

    monkeypatch.setattr(shipcard, "_verify_open_safetensors_fd", counted)
    receipt = _verify(tmp_path, manifest, ledger, tensor_to_file)
    assert calls == ["model.safetensors"]
    assert receipt["source"] == "verified_read"
    assert receipt["content_read_passes"] == 1
    assert receipt["content_bytes_read"] == (tmp_path / "model.safetensors").stat().st_size
    assert safetensors_content_receipt_manifest(receipt) == manifest

    monkeypatch.setattr(
        shipcard.os,
        "read",
        lambda *_args, **_kwargs: pytest.fail(
            "receipt validation must not reread content"
        ),
    )
    validate_safetensors_content_receipt(
        tmp_path,
        receipt,
        expected_weight_manifest=manifest,
        expected_tensor_sha256=ledger,
        expected_tensor_to_file=tensor_to_file,
    )


def test_attested_writer_receipt_is_zero_reread_and_stat_bound(tmp_path):
    manifest, ledger, tensor_to_file = _single_artifact(tmp_path)
    receipt = capture_attested_safetensors_write_receipt(
        tmp_path,
        weight_manifest=manifest,
        tensor_sha256=ledger,
        tensor_to_file=tensor_to_file,
    )
    assert receipt["source"] == "attested_write"
    assert receipt["content_read_passes"] == 0
    assert receipt["content_bytes_read"] == 0
    validate_safetensors_content_receipt(
        tmp_path,
        receipt,
        expected_weight_manifest=manifest,
        expected_tensor_sha256=ledger,
        expected_tensor_to_file=tensor_to_file,
    )

    _flip_payload_byte(tmp_path / "model.safetensors", "a.weight")
    with pytest.raises(ValueError, match="changed after content receipt"):
        validate_safetensors_content_receipt(
            tmp_path,
            receipt,
            expected_weight_manifest=manifest,
            expected_tensor_sha256=ledger,
            expected_tensor_to_file=tensor_to_file,
        )


def test_streaming_writer_attestation_captures_zero_post_write_pass(tmp_path):
    values = {
        "a.weight": torch.arange(16, dtype=torch.float32),
        "b.weight": torch.arange(16, dtype=torch.float32) + 1,
    }
    writer = _StreamWriter()
    for name, tensor in values.items():
        writer.add(
            name,
            tensor.dtype,
            tensor.shape,
            lambda value=tensor: value.clone(),
        )
    writer.write(tmp_path / "model.safetensors", shard_bytes=64)
    manifest = {
        "schema": shipcard.WEIGHT_CONTENT_MANIFEST_SCHEMA,
        "algorithm": "sha256",
        "files": writer.last_weight_manifest_files,
    }
    receipt = capture_attested_safetensors_write_receipt(
        tmp_path,
        weight_manifest=manifest,
        tensor_sha256=writer.last_tensor_content_sha256,
        tensor_to_file=writer.last_tensor_to_file,
    )
    assert receipt["content_read_passes"] == 0
    assert set(receipt["files"]) == set(writer.last_weight_manifest_files)
    validate_safetensors_content_receipt(
        tmp_path,
        receipt,
        expected_weight_manifest=manifest,
        expected_tensor_sha256=writer.last_tensor_content_sha256,
        expected_tensor_to_file=writer.last_tensor_to_file,
    )


def test_one_pass_rejects_payload_and_declared_container_digest_mutation(tmp_path):
    manifest, ledger, tensor_to_file = _single_artifact(tmp_path)
    _flip_payload_byte(tmp_path / "model.safetensors", "a.weight")
    with pytest.raises(ValueError, match="weight content manifest differs"):
        _verify(tmp_path, manifest, ledger, tensor_to_file)

    manifest, ledger, tensor_to_file = _single_artifact(tmp_path)
    false_manifest = deepcopy(manifest)
    false_manifest["files"]["model.safetensors"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="weight content manifest differs"):
        _verify(tmp_path, false_manifest, ledger, tensor_to_file)


def test_one_pass_rejects_header_mutation(tmp_path):
    manifest, ledger, tensor_to_file = _single_artifact(tmp_path)
    path = tmp_path / "model.safetensors"
    with path.open("r+b") as handle:
        (header_length,) = struct.unpack("<Q", handle.read(8))
        raw = handle.read(header_length)
        assert b"a.weight" in raw
        mutated = raw.replace(b"a.weight", b"z.weight", 1)
        assert len(mutated) == len(raw)
        handle.seek(8)
        handle.write(mutated)
        handle.flush()
        os.fsync(handle.fileno())
    with pytest.raises(
        ValueError,
        match="(?:weight content manifest|tensor digest ledger) differs",
    ):
        _verify(tmp_path, manifest, ledger, tensor_to_file)


def test_one_pass_rejects_stat_race_and_symlink(tmp_path, monkeypatch):
    manifest, ledger, tensor_to_file = _single_artifact(tmp_path)
    path = tmp_path / "model.safetensors"
    original = shipcard._verify_open_safetensors_fd

    def raced(*args, **kwargs):
        result = original(*args, **kwargs)
        current = path.stat()
        os.utime(
            path,
            ns=(current.st_atime_ns, current.st_mtime_ns + 1_000_000),
        )
        return result

    monkeypatch.setattr(shipcard, "_verify_open_safetensors_fd", raced)
    with pytest.raises(ValueError, match="path changed during"):
        _verify(tmp_path, manifest, ledger, tensor_to_file)

    monkeypatch.setattr(shipcard, "_verify_open_safetensors_fd", original)
    path.unlink()
    backing = tmp_path / "backing.bin"
    backing.write_bytes(b"not a safetensors container")
    path.symlink_to(backing)
    with pytest.raises(ValueError, match="safely open"):
        _verify(tmp_path, manifest, ledger, tensor_to_file)


def test_one_pass_rejects_safetensors_namespace_race(tmp_path, monkeypatch):
    manifest, ledger, tensor_to_file = _single_artifact(tmp_path)
    original = shipcard._current_safetensors_names
    calls = 0

    def raced(directory_fd):
        nonlocal calls
        calls += 1
        names = original(directory_fd)
        return names if calls == 1 else [*names, "inserted.safetensors"]

    monkeypatch.setattr(shipcard, "_current_safetensors_names", raced)
    with pytest.raises(ValueError, match="namespace changed during"):
        _verify(tmp_path, manifest, ledger, tensor_to_file)


def test_one_pass_rejects_symlink_artifact_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    manifest, ledger, tensor_to_file = _single_artifact(real)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        _verify(alias, manifest, ledger, tensor_to_file)


def test_scanner_rejects_oversized_header_and_nonfinite_json(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", shipcard._MAX_SAFETENSORS_HEADER_BYTES + 1))
    fd = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(ValueError, match="invalid safetensors header length"):
            shipcard._verify_open_safetensors_fd(
                fd,
                name=path.name,
                initial_stat=os.fstat(fd),
            )
    finally:
        os.close(fd)
    with pytest.raises(ValueError, match="nonfinite JSON constant"):
        shipcard._strict_json_object(b'{"value":NaN}', where="test header")


def test_scanner_rejects_gap_trailing_data_and_shape_span_mismatch(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_raw_safetensors(
        path,
        {
            "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
            "b": {"dtype": "U8", "shape": [1], "data_offsets": [2, 3]},
        },
        b"abc",
    )
    with pytest.raises(ValueError, match="leaves a gap"):
        _scan_raw_safetensors(path)

    _write_raw_safetensors(
        path,
        {"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}},
        b"ab",
    )
    with pytest.raises(ValueError, match="data area is 2B"):
        _scan_raw_safetensors(path)

    _write_raw_safetensors(
        path,
        {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
        b"abcd",
    )
    with pytest.raises(ValueError, match="requires 8B"):
        _scan_raw_safetensors(path)


def test_scanner_accepts_and_hashes_empty_tensor(tmp_path):
    path = tmp_path / "model.safetensors"
    _write_raw_safetensors(
        path,
        {
            "empty": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
            "value": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        },
        b"x",
    )
    record, bytes_read, _calls = _scan_raw_safetensors(path)
    assert bytes_read == path.stat().st_size
    assert record["tensor_sha256"]["empty"] == (
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )


def test_one_pass_scans_each_shard_once(tmp_path, monkeypatch):
    shard_1 = tmp_path / "model-00001-of-00002.safetensors"
    shard_2 = tmp_path / "model-00002-of-00002.safetensors"
    save_file({"a.weight": torch.arange(16, dtype=torch.float32)}, shard_1)
    save_file({"b.weight": torch.arange(8, dtype=torch.uint8)}, shard_2)
    ledger = {
        **_safetensors_tensor_payload_sha256(shard_1, ["a.weight"]),
        **_safetensors_tensor_payload_sha256(shard_2, ["b.weight"]),
    }
    tensor_to_file = {
        "a.weight": shard_1.name,
        "b.weight": shard_2.name,
    }
    manifest = build_weight_content_manifest(tmp_path)
    calls = []
    original = shipcard._verify_open_safetensors_fd

    def counted(*args, **kwargs):
        calls.append(kwargs["name"])
        return original(*args, **kwargs)

    monkeypatch.setattr(shipcard, "_verify_open_safetensors_fd", counted)
    receipt = _verify(tmp_path, manifest, ledger, tensor_to_file)
    assert calls == [shard_1.name, shard_2.name]
    assert receipt["content_read_passes"] == 1
    assert receipt["content_bytes_read"] == (
        shard_1.stat().st_size + shard_2.stat().st_size
    )
    assert set(receipt["files"][shard_1.name]["tensor_sha256"]) == {"a.weight"}
    assert set(receipt["files"][shard_2.name]["tensor_sha256"]) == {"b.weight"}
