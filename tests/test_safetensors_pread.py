"""Contract tests for the explicit mmap-free streamed-weight backend."""
from __future__ import annotations

import errno
import json
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prismaquant import safetensors_pread as pread
from prismaquant.layer_streaming import _materialize, _read_layer_to_device
from prismaquant.streaming_model import _estimate_layer_cache_bytes


def _raw_file(path, header, payload=b""):
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
    return path


def _one_tensor_header(*, dtype="U8", shape=(1,), offsets=(0, 1)):
    return {
        "model.layers.0.weight": {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": list(offsets),
        }
    }


def test_pread_matches_safe_open_metadata_and_values(tmp_path):
    path = tmp_path / "model.safetensors"
    tensors = {
        "model.layers.0.bf16": torch.tensor(
            [[-1.5, 0.0], [2.25, 9.0]], dtype=torch.bfloat16
        ),
        "model.layers.0.f32": torch.tensor(
            [1.25, -7.5, 3.0], dtype=torch.float32
        ),
        "model.layers.0.f16": torch.tensor(
            [0.5, -2.0], dtype=torch.float16
        ),
        "model.layers.0.i64": torch.tensor([-(2**40), 2**40], dtype=torch.int64),
        "model.layers.0.i32": torch.tensor([-17, 42], dtype=torch.int32),
        "model.layers.0.i16": torch.tensor([-3, 8], dtype=torch.int16),
        "model.layers.0.i8": torch.tensor([-128, 127], dtype=torch.int8),
        "model.layers.0.u8": torch.tensor([0, 255], dtype=torch.uint8),
        "model.layers.0.bool": torch.tensor([True, False, True]),
        "model.layers.0.scalar": torch.tensor(11.0),
        "model.layers.0.empty": torch.empty((2, 0), dtype=torch.float32),
    }
    save_file(tensors, str(path), metadata={"fixture": "pread"})

    with safe_open(str(path), framework="pt") as reference:
        with pread.PreadSafetensors(path) as reader:
            assert set(reader.keys()) == set(reference.keys())
            for name in reference.keys():
                info = reader.tensor_info(name)
                ref_slice = reference.get_slice(name)
                assert info.dtype == ref_slice.get_dtype()
                assert info.shape == tuple(ref_slice.get_shape())
                torch.testing.assert_close(
                    reader.get_tensor(name),
                    reference.get_tensor(name),
                    rtol=0,
                    atol=0,
                )


def test_tensor_storage_survives_reader_close(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"x": torch.arange(9, dtype=torch.float32)}, str(path))
    with pread.PreadSafetensors(path) as reader:
        fd = reader.fileno
        tensor = reader.get_tensor("x")
    with pytest.raises(OSError) as exc:
        os.fstat(fd)
    assert exc.value.errno == errno.EBADF
    torch.testing.assert_close(tensor, torch.arange(9, dtype=torch.float32))


@pytest.mark.parametrize(
    "dtype_name",
    ["float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu"],
)
def test_float8_payload_bits_match_safe_open(tmp_path, dtype_name):
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        pytest.skip(f"torch lacks {dtype_name}")
    path = tmp_path / "model.safetensors"
    # Compare bytes rather than float8 arithmetic (which torch does not expose
    # uniformly across releases).  This includes E8M0 scale planes used by the
    # declared MXFP4/FP8 source contracts.
    source = torch.tensor([0x00, 0x3C, 0x7E, 0xFF], dtype=torch.uint8).view(dtype)
    save_file({"scale_or_weight": source}, str(path))
    with safe_open(str(path), framework="pt") as reference:
        with pread.PreadSafetensors(path) as reader:
            got = reader.get_tensor("scale_or_weight")
            assert got.dtype == source.dtype
            assert torch.equal(
                got.view(torch.uint8),
                reference.get_tensor("scale_or_weight").view(torch.uint8),
            )


def test_layer_reader_pread_matches_default_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "1")
    path = tmp_path / "model.safetensors"
    values = {
        "model.layers.0.a.weight": torch.arange(
            12, dtype=torch.float32
        ).reshape(3, 4),
        "model.layers.0.b.weight": torch.tensor(
            [[-3, 7], [11, -13]], dtype=torch.int32
        ),
        "model.layers.1.weight": torch.ones(2, dtype=torch.float32),
    }
    save_file(values, str(path))
    selected = {name: str(path) for name in values}
    checkpoint = {name: name for name in values}

    baseline = _read_layer_to_device(
        "model.layers.0.",
        selected,
        checkpoint,
        torch.bfloat16,
        torch.device("cpu"),
        safetensors_backend="safe_open",
    )
    candidate = _read_layer_to_device(
        "model.layers.0.",
        selected,
        checkpoint,
        torch.bfloat16,
        torch.device("cpu"),
        safetensors_backend="pread",
    )
    assert baseline.keys() == candidate.keys()
    for name in baseline:
        assert baseline[name].dtype == candidate[name].dtype
        assert baseline[name].is_contiguous() and candidate[name].is_contiguous()
        torch.testing.assert_close(candidate[name], baseline[name], rtol=0, atol=0)


def test_layer_reader_env_selects_pread_without_touching_safe_open(
    tmp_path, monkeypatch,
):
    from prismaquant import layer_streaming

    path = tmp_path / "model.safetensors"
    name = "model.layers.0.weight"
    save_file({name: torch.arange(6, dtype=torch.bfloat16)}, str(path))
    monkeypatch.setenv(pread.SAFETENSORS_BACKEND_ENV, "pread")
    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "1")

    def forbidden_safe_open(*_args, **_kwargs):
        raise AssertionError("pread selection must not call safe_open")

    monkeypatch.setattr(layer_streaming, "safe_open", forbidden_safe_open)
    got = _read_layer_to_device(
        "model.layers.0.", {name: str(path)}, {name: name},
        torch.bfloat16, torch.device("cpu"),
    )
    torch.testing.assert_close(got[name], torch.arange(6, dtype=torch.bfloat16))


def test_resident_materializer_pread_does_not_touch_safe_open(
    tmp_path, monkeypatch,
):
    from prismaquant import layer_streaming

    path = tmp_path / "model.safetensors"
    expected = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    save_file({"weight": expected}, str(path))
    model = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)

    def forbidden_safe_open(*_args, **_kwargs):
        raise AssertionError("pread resident load must not call safe_open")

    monkeypatch.setattr(layer_streaming, "safe_open", forbidden_safe_open)
    loaded = _materialize(
        model,
        ["weight"],
        {"weight": str(path)},
        {"weight": "weight"},
        torch.device("cpu"),
        torch.bfloat16,
        safetensors_backend="pread",
    )
    assert loaded == 1
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)


def test_pread_reads_fp8_scale_payload_without_safe_open(tmp_path, monkeypatch):
    from prismaquant import layer_streaming

    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("torch lacks float8_e4m3fn")
    path = tmp_path / "model.safetensors"
    name = "model.layers.0.weight"
    scale_name = "model.layers.0.weight_scale_inv"
    save_file({
        name: torch.ones((128, 128), dtype=torch.float32).to(dtype),
        scale_name: torch.full((1, 1), 2.0, dtype=torch.float32),
    }, str(path))

    def forbidden_safe_open(*_args, **_kwargs):
        raise AssertionError("pread FP8 scale load must not call safe_open")

    monkeypatch.setattr(layer_streaming, "safe_open", forbidden_safe_open)
    got = _read_layer_to_device(
        "model.layers.0.",
        {name: str(path)},
        {name: name},
        torch.bfloat16,
        torch.device("cpu"),
        fp8_scale_inv_map={name: (str(path), scale_name)},
        safetensors_backend="pread",
    )
    assert got[name].dtype == torch.bfloat16
    torch.testing.assert_close(
        got[name], torch.full((128, 128), 2.0, dtype=torch.bfloat16),
        rtol=0, atol=0,
    )


def test_threaded_layer_pread_matches_serial_safe_open(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    values = {
        f"model.layers.0.experts.{index}.weight": torch.full(
            (3, 5), index + 0.25, dtype=torch.float32
        )
        for index in range(24)
    }
    save_file(values, str(path))
    selected = {name: str(path) for name in values}
    checkpoint = {name: name for name in values}
    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "1")
    baseline = _read_layer_to_device(
        "model.layers.0.", selected, checkpoint, torch.bfloat16,
        torch.device("cpu"), safetensors_backend="safe_open",
    )
    monkeypatch.setenv("PRISMAQUANT_LAYER_READ_THREADS", "4")
    candidate = _read_layer_to_device(
        "model.layers.0.", selected, checkpoint, torch.bfloat16,
        torch.device("cpu"), safetensors_backend="pread",
    )
    assert list(candidate) == list(baseline)
    for name in baseline:
        torch.testing.assert_close(candidate[name], baseline[name], rtol=0, atol=0)


def test_pread_metadata_estimator_matches_safe_open(tmp_path):
    path = tmp_path / "model.safetensors"
    tensors = {
        "model.layers.0.a.weight": torch.zeros(5, 7, dtype=torch.bfloat16),
        "model.layers.0.b.weight": torch.zeros(3, dtype=torch.int64),
        "model.layers.1.a.weight": torch.zeros(2, 11, dtype=torch.float32),
    }
    save_file(tensors, str(path))
    kwargs = dict(
        weight_shard={name: str(path) for name in tensors},
        weight_ckpt={name: name for name in tensors},
        layers_prefix="model.layers.",
        num_layers=2,
        target_dtype=torch.bfloat16,
    )
    expected = _estimate_layer_cache_bytes(
        **kwargs, safetensors_backend="safe_open"
    )
    actual = _estimate_layer_cache_bytes(
        **kwargs, safetensors_backend="pread"
    )
    assert actual == expected


def test_backend_selection_defaults_env_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv(pread.SAFETENSORS_BACKEND_ENV, raising=False)
    assert pread.resolve_safetensors_backend() == "safe_open"
    monkeypatch.setenv(pread.SAFETENSORS_BACKEND_ENV, " PREAD ")
    assert pread.resolve_safetensors_backend() == "pread"
    assert pread.resolve_safetensors_backend("safe_open") == "safe_open"
    with pytest.raises(ValueError, match="unsupported safetensors backend"):
        pread.resolve_safetensors_backend("mmap-ish")


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        (b"", "length prefix"),
        (b"1234567", "length prefix"),
        ((0).to_bytes(8, "little"), "header length 0"),
        ((7).to_bytes(8, "little") + b"{\"x\":1", "not 8-byte aligned"),
        ((16).to_bytes(8, "little") + b"{}", "truncated safetensors header"),
        (
            (pread.MAX_HEADER_BYTES + 8).to_bytes(8, "little"),
            "exceeds the",
        ),
        ((8).to_bytes(8, "little") + b"not-json", "does not start"),
        ((8).to_bytes(8, "little") + b"{\xff     }", "invalid UTF-8"),
    ],
)
def test_rejects_broken_length_and_header_bytes(tmp_path, contents, match):
    path = tmp_path / "broken.safetensors"
    path.write_bytes(contents)
    with pytest.raises(pread.SafetensorsPreadError, match=match):
        pread.PreadSafetensors(path)


def test_rejects_duplicate_json_keys(tmp_path):
    raw = (
        b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},'
        b'"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
    )
    raw += b" " * ((-len(raw)) % 8)
    path = tmp_path / "duplicate.safetensors"
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + b"x")
    with pytest.raises(pread.SafetensorsPreadError, match="duplicate JSON"):
        pread.PreadSafetensors(path)


@pytest.mark.parametrize(
    ("header", "payload", "match"),
    [
        (_one_tensor_header(dtype="WAT"), b"x", "unknown safetensors dtype"),
        (_one_tensor_header(shape=(-1,)), b"", "invalid shape"),
        (_one_tensor_header(offsets=(-1, 0)), b"", "out-of-range"),
        (_one_tensor_header(offsets=(0, 2)), b"xx", "requires 1"),
        (
            {
                "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
                "b": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
            },
            b"x",
            "overlap",
        ),
        (
            {
                "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
                "b": {"dtype": "U8", "shape": [1], "data_offsets": [2, 3]},
            },
            b"x-y",
            "gap",
        ),
        (_one_tensor_header(), b"xy", "unbound trailing payload"),
        (
            {"x": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1],
                   "future": 1}},
            b"x",
            "must contain exactly",
        ),
        (
            {"__metadata__": {"bad": 4}, **_one_tensor_header()},
            b"x",
            "must map strings to strings",
        ),
    ],
)
def test_rejects_invalid_tensor_geometry(tmp_path, header, payload, match):
    path = _raw_file(tmp_path / "broken.safetensors", header, payload)
    with pytest.raises(pread.SafetensorsPreadError, match=match):
        pread.PreadSafetensors(path)


def test_detects_short_payload_read_after_open(tmp_path):
    path = _raw_file(
        tmp_path / "model.safetensors",
        _one_tensor_header(shape=(16,), offsets=(0, 16)),
        bytes(range(16)),
    )
    with pread.PreadSafetensors(path) as reader:
        os.truncate(path, path.stat().st_size - 5)
        with pytest.raises(pread.SafetensorsPreadError, match="short pread"):
            reader.get_tensor("model.layers.0.weight")


def test_every_os_pread_is_bounded(tmp_path, monkeypatch):
    path = _raw_file(
        tmp_path / "model.safetensors",
        _one_tensor_header(shape=(29,), offsets=(0, 29)),
        bytes(range(29)),
    )
    real_pread = os.pread
    requests = []

    def recording_pread(fd, size, offset):
        requests.append(size)
        return real_pread(fd, size, offset)

    monkeypatch.setattr(pread, "PREAD_CHUNK_BYTES", 7)
    monkeypatch.setattr(pread.os, "pread", recording_pread)
    with pread.PreadSafetensors(path) as reader:
        tensor = reader.get_tensor("model.layers.0.weight")
    assert tensor.tolist() == list(range(29))
    assert requests and max(requests) <= 7


def test_parse_failure_closes_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "broken.safetensors"
    path.write_bytes(b"short")
    real_close = os.close
    closed = []

    def recording_close(fd):
        closed.append(fd)
        return real_close(fd)

    monkeypatch.setattr(pread.os, "close", recording_close)
    with pytest.raises(pread.SafetensorsPreadError):
        pread.PreadSafetensors(path)
    assert len(closed) == 1


def test_pread_estimator_fails_closed_on_corrupt_header(tmp_path):
    path = tmp_path / "broken.safetensors"
    path.write_bytes(b"bad")
    kwargs = dict(
        weight_shard={"model.layers.0.weight": str(path)},
        weight_ckpt={"model.layers.0.weight": "model.layers.0.weight"},
        layers_prefix="model.layers.",
        num_layers=1,
        target_dtype=torch.bfloat16,
    )
    with pytest.raises(pread.SafetensorsPreadError):
        _estimate_layer_cache_bytes(**kwargs, safetensors_backend="pread")
