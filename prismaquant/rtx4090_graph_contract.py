"""Fail-closed compile and CUDA-graph contract for the RTX 4090 FP8-CB lane.

The historical Gridbook endpoint gate intentionally proves a mode-0
``FULL_DECODE_ONLY`` configuration.  The context-first Ada artifact has a
stronger and independent contract: vLLM must run its real Inductor compilation
mode and capture both full and piecewise graphs at every advertised size.

This module is deliberately runtime-free.  It validates the resolved vLLM log
written by the separately pinned serving container; PrismaQuant never imports
Gridbook or vLLM to infer what that process did.
"""
from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Sequence


RTX4090_GRAPH_CONTRACT_SCHEMA = "prismaquant.rtx4090_graph_contract.v1"
RTX4090_COMPILE_CACHE_PREFLIGHT_SCHEMA = (
    "prismaquant.rtx4090_compile_cache_preflight.v1"
)
RTX4090_COMPILATION_MODE = 3
RTX4090_COMPILATION_BACKEND = "inductor"
RTX4090_CUDAGRAPH_MODE = "FULL_AND_PIECEWISE"
RTX4090_CUDAGRAPH_CAPTURE_SIZES = (1, 2, 4, 8, 16, 32, 64)
# vLLM's FULL-decode dispatcher only admits capture sizes at or below the
# scheduler's sequence ceiling. This is graph-shape capacity, not a claim that
# 64 simultaneous 32K contexts fit in the fixed KV allocation; endpoint smoke
# remains one request with n=1.
RTX4090_MAX_NUM_SEQS = max(RTX4090_CUDAGRAPH_CAPTURE_SIZES)
RTX4090_MAX_MODEL_LEN = 32768


def compilation_config_json(*, cache_dir: str | None = None) -> str:
    """Return the canonical vLLM CLI value for the Ada production gate."""

    payload: dict[str, Any] = {
        "mode": RTX4090_COMPILATION_MODE,
        "backend": RTX4090_COMPILATION_BACKEND,
        "cudagraph_mode": RTX4090_CUDAGRAPH_MODE,
        "cudagraph_capture_sizes": list(RTX4090_CUDAGRAPH_CAPTURE_SIZES),
    }
    if cache_dir is not None:
        cache_path = PurePosixPath(str(cache_dir))
        if (
            not cache_path.is_absolute()
            or ".." in cache_path.parts
            or str(cache_path) in {"/", "."}
        ):
            raise ValueError(
                "RTX4090 compile cache must be a dedicated absolute path"
            )
        payload["cache_dir"] = str(cache_path)
    return json.dumps(payload, separators=(",", ":"))


RTX4090_GRAPH_COMPILATION_CONFIG = compilation_config_json()


class RTX4090GraphContractError(RuntimeError):
    """The serving log does not prove the mandatory compilation contract."""


_SESSION_NONCE_RE = re.compile(r"[0-9a-f]{32}")


def _dedicated_absolute_path(value: str | Path, *, where: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) in {"/", "."}:
        raise RTX4090GraphContractError(
            f"{where} must be a dedicated absolute path"
        )
    return path


def _dedicated_container_path(value: str, *, where: str) -> PurePosixPath:
    path = PurePosixPath(str(value))
    if not path.is_absolute() or ".." in path.parts or str(path) in {"/", "."}:
        raise RTX4090GraphContractError(
            f"{where} must be a dedicated absolute path"
        )
    return path


def create_compile_cache_preflight(
    cache_root: str | Path,
    receipt_path: str | Path,
    *,
    configured_container_root: str,
    session_nonce: str,
) -> dict[str, Any]:
    """Create one no-clobber empty cache root and its pre-launch receipt."""

    root = _dedicated_absolute_path(cache_root, where="compile cache root")
    receipt = _dedicated_absolute_path(
        receipt_path, where="compile cache preflight receipt"
    )
    container = _dedicated_container_path(
        configured_container_root,
        where="configured container cache root",
    )
    if _SESSION_NONCE_RE.fullmatch(str(session_nonce)) is None:
        raise RTX4090GraphContractError(
            "compile cache session nonce must be 128-bit lowercase hex"
        )
    if root.exists() or root.is_symlink():
        raise RTX4090GraphContractError(
            f"compile cache root already exists: {root}"
        )
    if receipt.exists() or receipt.is_symlink():
        raise RTX4090GraphContractError(
            f"compile cache preflight receipt already exists: {receipt}"
        )
    if receipt == root or root in receipt.parents:
        raise RTX4090GraphContractError(
            "compile cache preflight receipt must be outside the cache root"
        )
    root.mkdir(mode=0o700)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or root.is_symlink() or any(root.iterdir()):
        raise RTX4090GraphContractError(
            "new compile cache root is not one empty ordinary directory"
        )
    payload = {
        "schema": RTX4090_COMPILE_CACHE_PREFLIGHT_SCHEMA,
        "session_nonce": str(session_nonce),
        "host_root": str(root.resolve(strict=True)),
        "configured_container_root": str(container),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "ctime_ns": int(info.st_ctime_ns),
        "prelaunch_empty": True,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    return payload


def validate_compile_cache_preflight(
    cache_root: str | Path,
    receipt_path: str | Path,
    *,
    configured_container_root: str,
    session_nonce: str,
) -> dict[str, Any]:
    """Bind the post-compile tree to the exact empty root created pre-launch."""

    root = _dedicated_absolute_path(cache_root, where="compile cache root")
    receipt = _dedicated_absolute_path(
        receipt_path, where="compile cache preflight receipt"
    )
    container = _dedicated_container_path(
        configured_container_root,
        where="configured container cache root",
    )
    if receipt == root or root in receipt.parents:
        raise RTX4090GraphContractError(
            "compile cache preflight receipt must be outside the cache root"
        )
    try:
        receipt_info = receipt.lstat()
    except OSError as exc:
        raise RTX4090GraphContractError(
            f"compile cache preflight receipt cannot be reopened: {exc}"
        ) from exc
    if not stat.S_ISREG(receipt_info.st_mode) or receipt.is_symlink():
        raise RTX4090GraphContractError(
            "compile cache preflight receipt is not one ordinary file"
        )
    try:
        raw = receipt.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RTX4090GraphContractError(
            f"compile cache preflight receipt cannot be read: {exc}"
        ) from exc
    required = {
        "schema", "session_nonce", "host_root", "configured_container_root",
        "device", "inode", "ctime_ns", "prelaunch_empty",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise RTX4090GraphContractError(
            "compile cache preflight receipt is not the closed v1 schema"
        )
    try:
        resolved_root = root.resolve(strict=True)
        info = root.lstat()
    except OSError as exc:
        raise RTX4090GraphContractError(
            f"compile cache root cannot be reopened: {exc}"
        ) from exc
    if (
        payload.get("schema") != RTX4090_COMPILE_CACHE_PREFLIGHT_SCHEMA
        or payload.get("session_nonce") != session_nonce
        or payload.get("configured_container_root")
        != str(container)
        or payload.get("host_root") != str(resolved_root)
        or payload.get("prelaunch_empty") is not True
        or type(payload.get("device")) is not int
        or type(payload.get("inode")) is not int
        or type(payload.get("ctime_ns")) is not int
        or payload.get("ctime_ns", 0) <= 0
        or payload.get("device") != int(info.st_dev)
        or payload.get("inode") != int(info.st_ino)
        or not stat.S_ISDIR(info.st_mode)
        or root.is_symlink()
    ):
        raise RTX4090GraphContractError(
            "compile cache post-run root differs from its empty pre-launch identity"
        )

    tree = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        entry = path.lstat()
        if stat.S_ISLNK(entry.st_mode):
            raise RTX4090GraphContractError(
                f"compile cache contains a symlink: {path}"
            )
        if stat.S_ISDIR(entry.st_mode):
            continue
        if not stat.S_ISREG(entry.st_mode):
            raise RTX4090GraphContractError(
                f"compile cache contains a non-regular member: {path}"
            )
        rel = path.relative_to(root).as_posix().encode("utf-8")
        content_digest = hashlib.sha256()
        content_bytes = 0
        try:
            with path.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    content_digest.update(block)
                    content_bytes += len(block)
        except OSError as exc:
            raise RTX4090GraphContractError(
                f"compile cache member cannot be read: {path}: {exc}"
            ) from exc
        if content_bytes != int(entry.st_size):
            raise RTX4090GraphContractError(
                f"compile cache member changed while hashing: {path}"
            )
        tree.update(len(rel).to_bytes(8, "little"))
        tree.update(rel)
        tree.update(content_bytes.to_bytes(8, "little"))
        tree.update(content_digest.digest())
        file_count += 1
        total_bytes += content_bytes
    if file_count == 0:
        raise RTX4090GraphContractError(
            "compile cache stayed empty; no post-run compiler artifact exists"
        )
    return {
        "schema": RTX4090_COMPILE_CACHE_PREFLIGHT_SCHEMA,
        "session_nonce": str(session_nonce),
        "configured_container_root": str(container),
        "preflight_sha256": hashlib.sha256(raw).hexdigest(),
        "directory_device": int(info.st_dev),
        "directory_inode": int(info.st_ino),
        "post_file_count": file_count,
        "post_total_bytes": total_bytes,
        "post_tree_sha256": tree.hexdigest(),
    }


_ENGINE_LINE_RE = re.compile(
    r"^.*Initializing a V1 LLM engine .*compilation_config=.*$", re.MULTILINE
)
_CAPTURE_SIZES_RE = re.compile(r"'cudagraph_capture_sizes': \[([^]]*)\]")
_MAX_CAPTURE_RE = re.compile(r"'max_cudagraph_capture_size':\s*(\d+)")
_MAX_SEQ_LEN_RE = re.compile(r"\bmax_seq_len=(\d+)")
_CAPTURE_FINISHED_RE = re.compile(
    r"Graph capturing finished in [0-9]+ secs, took -?[0-9.]+ GiB"
)
_PIECEWISE_CAPTURE_PROGRESS_RE = re.compile(
    r"Capturing CUDA graphs \(mixed prefill-decode, PIECEWISE\):"
    r"[^\r\n]*?([0-9]+)/([0-9]+)"
)
_FULL_CAPTURE_PROGRESS_RE = re.compile(
    r"Capturing CUDA graphs \(decode, FULL\):"
    r"[^\r\n]*?([0-9]+)/([0-9]+)"
)
_COMPILE_RANGE_RE = re.compile(
    r"Compiling a graph for compile range .*? takes [0-9.]+ s"
)
_DYNAMO_RE = re.compile(r"Dynamo bytecode transform time: [0-9.]+ s")
_COMPILE_TOTAL_RE = re.compile(r"torch\.compile took [0-9.]+ s in total")

_FORBIDDEN_MARKERS = (
    "CompilationMode.NONE",
    "Inductor compilation was disabled",
    "Skipping CUDA graph capture",
    "Overriding cudagraph_mode to PIECEWISE",
    "Overriding cudagraph_mode from FULL_AND_PIECEWISE",
    "setting cudagraph_mode=PIECEWISE",
    "setting cudagraph_mode=NONE",
    "torch._dynamo.exc.Unsupported",
    "torch._inductor.exc.InductorError",
    "BackendCompilerFailed",
    "recompile_limit",
    "recompile limit",
    "WON'T CONVERT",
)


def _parse_capture_sizes(engine_line: str) -> tuple[int, ...]:
    match = _CAPTURE_SIZES_RE.search(engine_line)
    if match is None:
        raise RTX4090GraphContractError(
            "resolved vLLM engine config has no cudagraph_capture_sizes"
        )
    raw = match.group(1).strip()
    if not raw:
        return ()
    try:
        values = tuple(int(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise RTX4090GraphContractError(
            "resolved vLLM capture sizes are not integers"
        ) from exc
    return values


def validate_rtx4090_graph_log(
    path: str | Path,
    *,
    expected_compile_cache: str | None = None,
    expected_compile_cache_root: str | None = None,
    expected_capture_sizes: Sequence[int] = RTX4090_CUDAGRAPH_CAPTURE_SIZES,
    expected_max_model_len: int = RTX4090_MAX_MODEL_LEN,
) -> dict[str, Any]:
    """Validate positive compile/capture evidence and return a bound receipt.

    ``expected_compile_cache_root`` should be the fresh run-specific root
    passed to vLLM.  vLLM writes the compiled graph into a fingerprinted
    descendant of that root, so the observed log path is intentionally kept
    separate from the configured launch path.  Positive Dynamo/compile timing
    prevents a pre-existing binary cache hit from masquerading as a compiler
    compatibility test.
    """

    log_path = Path(path)
    try:
        raw = log_path.read_bytes()
    except OSError as exc:
        raise RTX4090GraphContractError(
            f"serve log cannot be read: {log_path}"
        ) from exc
    text = raw.decode("utf-8", "replace")

    for marker in _FORBIDDEN_MARKERS:
        if marker.casefold() in text.casefold():
            raise RTX4090GraphContractError(
                f"vLLM log reports forbidden compile/graph marker: {marker}"
            )

    engine_lines = _ENGINE_LINE_RE.findall(text)
    if len(engine_lines) != 1:
        raise RTX4090GraphContractError(
            "serve log must contain exactly one resolved V1 engine "
            "compilation_config; concatenated/stale engine sessions are refused"
        )
    qualified = [
        line
        for line in engine_lines
        if "<CompilationMode.VLLM_COMPILE: 3>" in line
        and "<CUDAGraphMode.FULL_AND_PIECEWISE:" in line
    ]
    if len(qualified) != 1:
        raise RTX4090GraphContractError(
            "serve log must attest exactly one resolved mode-3 "
            "FULL_AND_PIECEWISE engine"
        )
    engine_line = qualified[0]
    if f"'backend': '{RTX4090_COMPILATION_BACKEND}'" not in engine_line:
        raise RTX4090GraphContractError(
            "resolved compilation backend is not explicit Inductor"
        )
    max_seq_len = _MAX_SEQ_LEN_RE.search(engine_line)
    if (
        max_seq_len is None
        or int(max_seq_len.group(1)) != int(expected_max_model_len)
    ):
        raise RTX4090GraphContractError(
            "resolved max_seq_len differs from the 32K serving contract"
        )

    expected_sizes = tuple(int(value) for value in expected_capture_sizes)
    observed_sizes = _parse_capture_sizes(engine_line)
    if observed_sizes != expected_sizes:
        raise RTX4090GraphContractError(
            "resolved capture sizes differ: "
            f"expected {list(expected_sizes)}, observed {list(observed_sizes)}"
        )
    maximum = _MAX_CAPTURE_RE.search(engine_line)
    if maximum is None or int(maximum.group(1)) != max(expected_sizes):
        raise RTX4090GraphContractError(
            "resolved max_cudagraph_capture_size does not match the profile"
        )

    dynamo = _DYNAMO_RE.search(text)
    compile_range = _COMPILE_RANGE_RE.search(text)
    compile_total = _COMPILE_TOTAL_RE.search(text)
    capture = _CAPTURE_FINISHED_RE.search(text)
    if dynamo is None or compile_range is None or compile_total is None:
        raise RTX4090GraphContractError(
            "serve log lacks positive fresh Dynamo/Inductor compilation evidence"
        )
    if capture is None:
        raise RTX4090GraphContractError(
            "serve log lacks positive 'Graph capturing finished' evidence"
        )
    piecewise_progress = tuple(
        (int(done), int(total))
        for done, total in _PIECEWISE_CAPTURE_PROGRESS_RE.findall(text)
    )
    piecewise_total = len(expected_sizes)
    if not any(
        done == total == piecewise_total
        for done, total in piecewise_progress
    ):
        raise RTX4090GraphContractError(
            "serve log does not prove completion of every requested PIECEWISE "
            "capture size"
        )
    full_progress = tuple(
        (int(done), int(total))
        for done, total in _FULL_CAPTURE_PROGRESS_RE.findall(text)
    )
    completed_full = [
        total
        for done, total in full_progress
        if done == total and total > 0
    ]
    if piecewise_total not in completed_full:
        raise RTX4090GraphContractError(
            "serve log does not prove completion of every requested FULL "
            "decode capture size"
        )

    if (
        expected_compile_cache is not None
        and expected_compile_cache_root is not None
    ):
        raise RTX4090GraphContractError(
            "choose an exact compile cache or a configured cache root, not both"
        )
    cache_matches = re.findall(
        r"Using cache directory: (\S+) for vLLM's torch\.compile", text
    )
    if len(cache_matches) != 1:
        raise RTX4090GraphContractError(
            "serve log must name exactly one torch.compile cache directory"
        )
    observed_cache = cache_matches[0]
    configured_cache_root: str | None = None
    if expected_compile_cache is not None:
        exact = PurePosixPath(str(expected_compile_cache))
        if (
            not exact.is_absolute()
            or ".." in exact.parts
            or str(exact) in {"/", "."}
        ):
            raise RTX4090GraphContractError(
                "expected torch.compile cache is not a dedicated absolute path"
            )
        configured_cache_root = str(exact)
        if observed_cache != configured_cache_root:
            raise RTX4090GraphContractError(
                "serve log does not bind compilation to the fresh expected cache"
            )

    if expected_compile_cache_root is not None:
        root = PurePosixPath(str(expected_compile_cache_root))
        observed = PurePosixPath(str(observed_cache))
        if (
            not root.is_absolute()
            or ".." in root.parts
            or str(root) in {"/", "."}
            or not observed.is_absolute()
            or ".." in observed.parts
            or (observed != root and root not in observed.parents)
        ):
            raise RTX4090GraphContractError(
                "resolved torch.compile cache is outside the fresh expected root"
            )
        configured_cache_root = str(root)

    if configured_cache_root is None:
        raise RTX4090GraphContractError(
            "one expected torch.compile cache path/root is required"
        )

    return {
        "schema": RTX4090_GRAPH_CONTRACT_SCHEMA,
        "compilation_mode": RTX4090_COMPILATION_MODE,
        "compilation_backend": RTX4090_COMPILATION_BACKEND,
        "cudagraph_mode": RTX4090_CUDAGRAPH_MODE,
        "capture_sizes": list(observed_sizes),
        "max_model_len": int(expected_max_model_len),
        "configured_compile_cache_root": configured_cache_root,
        "compile_cache": observed_cache,
        "dynamo_marker": dynamo.group(0),
        "compile_marker": compile_range.group(0),
        "compile_total_marker": compile_total.group(0),
        "piecewise_capture_count": piecewise_total,
        "full_capture_count": piecewise_total,
        "capture_marker": capture.group(0),
        "serve_log": str(log_path.resolve()),
        "serve_log_sha256": hashlib.sha256(raw).hexdigest(),
    }
