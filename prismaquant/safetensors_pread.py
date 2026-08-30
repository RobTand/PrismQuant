"""Validated, mmap-free safetensors reads backed by :func:`os.pread`.

This is a source reader for the existing streamed-layer residency path, not a
cache or scheduler.  It is intentionally opt-in: normal checkpoints continue
to use ``safetensors.safe_open`` unless the caller selects ``"pread"``.
"""
from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SAFE_OPEN_BACKEND = "safe_open"
PREAD_BACKEND = "pread"
SAFETENSORS_BACKENDS = frozenset({SAFE_OPEN_BACKEND, PREAD_BACKEND})
SAFETENSORS_BACKEND_ENV = "PRISMAQUANT_SAFETENSORS_BACKEND"

# Match safetensors' defensive header ceiling.  A corrupt length prefix must
# never turn into an unbounded allocation or read.
MAX_HEADER_BYTES = 100_000_000
PREAD_CHUNK_BYTES = 64 * 1024 * 1024


class SafetensorsPreadError(RuntimeError):
    """The safetensors container or a bounded pread failed validation."""


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int

    @property
    def nbytes(self) -> int:
        return self.data_end - self.data_start


_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E4M3FN": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
    # These formats are header-valid but cannot currently be materialized as
    # torch tensors by this reader.  ``get_tensor`` rejects them explicitly.
    "F4": 4,
    "F4_E2M1": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
}


def _torch_dtype(dtype: str) -> torch.dtype:
    mapping: dict[str, torch.dtype | None] = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
        "F8_E4M3FN": getattr(torch, "float8_e4m3fn", None),
        "F8_E4M3FNUZ": getattr(torch, "float8_e4m3fnuz", None),
        "F8_E5M2": getattr(torch, "float8_e5m2", None),
        "F8_E5M2FNUZ": getattr(torch, "float8_e5m2fnuz", None),
        "F8_E8M0": getattr(torch, "float8_e8m0fnu", None),
        "U16": getattr(torch, "uint16", None),
        "I16": torch.int16,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "U32": getattr(torch, "uint32", None),
        "I32": torch.int32,
        "F32": torch.float32,
        "U64": getattr(torch, "uint64", None),
        "I64": torch.int64,
        "F64": torch.float64,
    }
    result = mapping.get(dtype)
    if result is None:
        raise SafetensorsPreadError(
            f"safetensors dtype {dtype!r} cannot be materialized by "
            f"torch {torch.__version__} through the pread backend"
        )
    return result


def resolve_safetensors_backend(requested: str | None = None) -> str:
    """Resolve and validate the source backend once per streaming context."""
    raw = requested
    if raw is None:
        raw = os.environ.get(SAFETENSORS_BACKEND_ENV, SAFE_OPEN_BACKEND)
    backend = str(raw).strip().lower()
    if backend not in SAFETENSORS_BACKENDS:
        choices = ", ".join(sorted(SAFETENSORS_BACKENDS))
        raise ValueError(
            f"unsupported safetensors backend {raw!r}; expected one of "
            f"{choices} (from argument or {SAFETENSORS_BACKEND_ENV})"
        )
    return backend


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SafetensorsPreadError(
                f"duplicate JSON object key {key!r} in safetensors header"
            )
        result[key] = value
    return result


class PreadSafetensors:
    """A validated safetensors file whose payload is read without mmap.

    The descriptor is owned by this object and is closed both on context exit
    and when header parsing fails.  Tensor offsets are absolute after parsing,
    so every payload access is one bounded, positional read sequence.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = str(Path(path))
        self._fd = -1
        self._closed = False
        self._size = 0
        self._payload_start = 0
        self._tensors: dict[str, TensorInfo] = {}
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        try:
            self._fd = os.open(self.path, flags)
            st = os.fstat(self._fd)
            if not stat.S_ISREG(st.st_mode):
                raise SafetensorsPreadError(
                    f"safetensors path is not a regular file: {self.path}"
                )
            self._size = int(st.st_size)
            self._parse_header()
        except Exception:
            self.close()
            raise

    @property
    def fileno(self) -> int:
        return self._fd

    def __enter__(self) -> "PreadSafetensors":
        if self._closed:
            raise SafetensorsPreadError(
                f"cannot reopen closed safetensors reader: {self.path}"
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            fd, self._fd = self._fd, -1
            if fd >= 0:
                os.close(fd)

    def keys(self) -> tuple[str, ...]:
        self._require_open()
        return tuple(self._tensors)

    def tensor_info(self, name: str) -> TensorInfo:
        self._require_open()
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise SafetensorsPreadError(
                f"tensor {name!r} is absent from {self.path}"
            ) from exc

    def get_tensor(self, name: str) -> torch.Tensor:
        info = self.tensor_info(name)
        dtype = _torch_dtype(info.dtype)
        if info.nbytes == 0:
            return torch.empty(info.shape, dtype=dtype)
        payload = self._pread_exact(
            info.nbytes,
            info.data_start,
            purpose=f"tensor {name!r}",
        )
        try:
            tensor = torch.frombuffer(payload, dtype=dtype)
            return tensor.reshape(info.shape)
        except Exception as exc:
            raise SafetensorsPreadError(
                f"cannot materialize tensor {name!r} ({info.dtype}, "
                f"shape={list(info.shape)}, bytes={info.nbytes}) from "
                f"{self.path}: {exc}"
            ) from exc

    def _require_open(self) -> None:
        if self._closed or self._fd < 0:
            raise SafetensorsPreadError(
                f"safetensors reader is closed: {self.path}"
            )

    def _pread_exact(self, size: int, offset: int, *, purpose: str) -> bytearray:
        self._require_open()
        if size < 0 or offset < 0 or offset + size > self._size:
            raise SafetensorsPreadError(
                f"invalid read range for {purpose} in {self.path}: "
                f"offset={offset}, size={size}, file_size={self._size}"
            )
        out = bytearray(size)
        done = 0
        while done < size:
            request = min(PREAD_CHUNK_BYTES, size - done)
            try:
                chunk = os.pread(self._fd, request, offset + done)
            except InterruptedError:
                continue
            if not chunk:
                raise SafetensorsPreadError(
                    f"short pread for {purpose} in {self.path}: expected "
                    f"{size} bytes at offset {offset}, received {done}"
                )
            out[done:done + len(chunk)] = chunk
            done += len(chunk)
        return out

    def _parse_header(self) -> None:
        if self._size < 8:
            raise SafetensorsPreadError(
                f"truncated safetensors length prefix in {self.path}: "
                f"file has {self._size} bytes"
            )
        header_len = int.from_bytes(
            self._pread_exact(8, 0, purpose="header length"), "little"
        )
        if header_len <= 0:
            raise SafetensorsPreadError(
                f"invalid safetensors header length {header_len} in {self.path}"
            )
        if header_len > MAX_HEADER_BYTES:
            raise SafetensorsPreadError(
                f"safetensors header length {header_len} exceeds the "
                f"{MAX_HEADER_BYTES}-byte limit in {self.path}"
            )
        if header_len % 8 != 0:
            raise SafetensorsPreadError(
                f"safetensors header length {header_len} is not 8-byte "
                f"aligned in {self.path}"
            )
        if header_len > self._size - 8:
            raise SafetensorsPreadError(
                f"truncated safetensors header in {self.path}: declared "
                f"{header_len} bytes but file has only {self._size - 8} "
                "bytes after its length prefix"
            )
        raw_header = bytes(
            self._pread_exact(header_len, 8, purpose="header")
        )
        if not raw_header.startswith(b"{"):
            raise SafetensorsPreadError(
                f"safetensors header does not start with '{{' in {self.path}"
            )
        try:
            header = json.loads(
                raw_header.decode("utf-8"),
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except SafetensorsPreadError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SafetensorsPreadError(
                f"invalid UTF-8 JSON safetensors header in {self.path}: {exc}"
            ) from exc
        if not isinstance(header, dict):
            raise SafetensorsPreadError(
                f"safetensors header must be a JSON object in {self.path}"
            )

        metadata = header.pop("__metadata__", None)
        if metadata is not None and (
            not isinstance(metadata, dict)
            or any(not isinstance(k, str) or not isinstance(v, str)
                   for k, v in metadata.items())
        ):
            raise SafetensorsPreadError(
                f"__metadata__ must map strings to strings in {self.path}"
            )

        payload_start = 8 + header_len
        payload_bytes = self._size - payload_start
        relative: list[tuple[int, int, str, str, tuple[int, ...]]] = []
        for name, entry in header.items():
            if not isinstance(name, str) or not name:
                raise SafetensorsPreadError(
                    f"safetensors tensor names must be non-empty strings in "
                    f"{self.path}"
                )
            if not isinstance(entry, dict) or set(entry) != {
                "dtype", "shape", "data_offsets"
            }:
                raise SafetensorsPreadError(
                    f"tensor {name!r} in {self.path} must contain exactly "
                    "dtype, shape, and data_offsets"
                )
            dtype = entry["dtype"]
            if not isinstance(dtype, str) or dtype not in _DTYPE_BITS:
                raise SafetensorsPreadError(
                    f"unknown safetensors dtype {dtype!r} for tensor "
                    f"{name!r} in {self.path}"
                )
            raw_shape = entry["shape"]
            if not isinstance(raw_shape, list) or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
                for dim in raw_shape
            ):
                raise SafetensorsPreadError(
                    f"invalid shape for tensor {name!r} in {self.path}: "
                    f"{raw_shape!r}"
                )
            shape = tuple(raw_shape)
            offsets = entry["data_offsets"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in offsets)
            ):
                raise SafetensorsPreadError(
                    f"invalid data_offsets for tensor {name!r} in "
                    f"{self.path}: {offsets!r}"
                )
            start, end = offsets
            if start < 0 or end < start or end > payload_bytes:
                raise SafetensorsPreadError(
                    f"out-of-range data_offsets {offsets!r} for tensor "
                    f"{name!r} in {self.path}; payload has {payload_bytes} "
                    "bytes"
                )
            expected = (math.prod(shape) * _DTYPE_BITS[dtype] + 7) // 8
            if end - start != expected:
                raise SafetensorsPreadError(
                    f"byte range for tensor {name!r} in {self.path} has "
                    f"{end - start} bytes; dtype={dtype}, shape={list(shape)} "
                    f"requires {expected}"
                )
            relative.append((start, end, name, dtype, shape))

        # A safetensors payload is a contiguous, fully described buffer.  This
        # rejects overlaps, gaps and unbound trailing bytes before any tensor
        # can be returned to the streaming cache.
        cursor = 0
        tensors: dict[str, TensorInfo] = {}
        for start, end, name, dtype, shape in sorted(relative):
            if start != cursor:
                relation = "overlap" if start < cursor else "gap"
                raise SafetensorsPreadError(
                    f"{relation} in safetensors payload before tensor "
                    f"{name!r} in {self.path}: expected offset {cursor}, "
                    f"found {start}"
                )
            cursor = end
            tensors[name] = TensorInfo(
                dtype=dtype,
                shape=shape,
                data_start=payload_start + start,
                data_end=payload_start + end,
            )
        if cursor != payload_bytes:
            raise SafetensorsPreadError(
                f"unbound trailing payload in {self.path}: described "
                f"{cursor} of {payload_bytes} bytes"
            )
        self._payload_start = payload_start
        # Preserve header order for deterministic parity with safe_open keys.
        self._tensors = {
            name: tensors[name] for name in header
        }


def read_safetensors_metadata(
    path: str | os.PathLike[str],
) -> dict[str, TensorInfo]:
    """Return validated tensor metadata and close the descriptor promptly."""
    with PreadSafetensors(path) as reader:
        return {name: reader.tensor_info(name) for name in reader.keys()}
