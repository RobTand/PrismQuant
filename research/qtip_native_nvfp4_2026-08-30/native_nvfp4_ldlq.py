"""QTIP-informed one-Linear optimizer that emits only stock NVFP4 fields.

Only QTIP's offline BlockLDLQ error-feedback schedule transfers.  Every
terminal decision is serialized immediately by PrismaQuant's existing native
group-16 E2M1 / FP8-E4M3 scale codec. QTIP trellis bytes, ``SU``/``SV``,
Hadamards, kernels, containers, and runtime reconstruction are absent.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import torch

from prismaquant import export_native_compressed as enc


SCHEMA = "prismaquant.research.qtip_native_nvfp4_one_linear.v2"
CALIBRATION_SCHEMA = "prismaquant.calibration_identity.v1"
QTIP_REPOSITORY = "https://github.com/Cornell-RelaxML/qtip"
QTIP_PINNED_COMMIT = "e90c6688c8dfae326a3a81b5eb032db7c6680ec0"
QTIP_SOURCE_FILES = {
    "lib/utils/math_utils.py": "65d50936e87b2c266806de201dea89b2d74a2ed38e33ef462bd8c3aafb333844",
    "lib/algo/ldlq.py": "793e364fbe91e5b28740d0fc81a6e8618daa6a6a8ce5adbf9b877ba2e46e5bbe",
    "lib/algo/finetune.py": "0a1021d9bffa3e6a1a86f537096a072779c759417b87624f1eef669a1df2c1a4",
    "lib/codebook/bitshift.py": "a299ae97d2ccc80a142095c3c16ed619b435b68736fd52702ab396bc37218531",
}
GROUP = 16
SCALE_RULE = enc.NVFP4_SCALE_RULE_JOINT_MSE
SCALE_LEVELS = (6.0, 4.0)
FULL_SCALE_LEVELS = (6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5)
CLIP_QUANTILE = 0.999
DAMP = 1.0
FIELDS = ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")
ARM_NAMES = (
    "A_native_nvfp4_rtn_jso",
    "B_prismaquant_gptq_static_order_jso",
    "C_qtip_block_ldl_native_nvfp4",
    "C2_qtip_block_ldl_native_nvfp4_seven_level_scale_heuristic",
)
_LOCK = threading.RLock()
QUALITY_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PRISMAQUANT_ACT_CLIP_QUANTILE": str(CLIP_QUANTILE),
    "PRISMAQUANT_GPTQ_DAMP": str(DAMP),
    "PRISMAQUANT_GPTQ_DAMP_SWEEP": "0",
    "PRISMAQUANT_GPTQ_DAMP_ROLES": "",
    "PRISMAQUANT_DO_NO_HARM": "1",
    "PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING": "0",
    "PRISMAQUANT_GPTQ_BLOCK_SIZE": "128",
    "PRISMAQUANT_FP8_GPTQ_BLOCK_SIZE": "128",
    "PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_GRID": "5",
    "PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_LO": "0.75",
    "PRISMAQUANT_NVFP4_JOINT_SCALE_GLOBAL_SPAN_HI": "1.25",
}


@dataclass(frozen=True)
class Arm:
    fields: Mapping[str, torch.Tensor]
    reconstruction: torch.Tensor
    terminal_blocks: tuple[dict[str, object], ...] = ()


@contextlib.contextmanager
def fixed_contract(scale_levels: tuple[float, ...] = SCALE_LEVELS) -> Iterator[None]:
    """Pin legacy env-backed render choices and restore the caller exactly."""
    with _LOCK:
        old_env = {k: os.environ.get(k) for k in QUALITY_ENV}
        old_levels = enc._NVFP4_JOINT_SCALE_LEVELS
        old_flags = dict(enc._ACT_AWARE_FLAGS)
        old_precision = torch.get_float32_matmul_precision()
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        old_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            os.environ.update(QUALITY_ENV)
            enc._NVFP4_JOINT_SCALE_LEVELS = scale_levels
            enc._ACT_AWARE_FLAGS.update(
                {key: False for key in enc._ACT_AWARE_FLAGS}
            )
            torch.set_float32_matmul_precision("highest")
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with enc._temporary_export_nvfp4_scale_rule(SCALE_RULE):
                yield
        finally:
            enc._NVFP4_JOINT_SCALE_LEVELS = old_levels
            enc._ACT_AWARE_FLAGS.clear()
            enc._ACT_AWARE_FLAGS.update(old_flags)
            torch.set_float32_matmul_precision(old_precision)
            torch.use_deterministic_algorithms(
                old_deterministic, warn_only=old_warn_only
            )
            torch.backends.cuda.matmul.allow_tf32 = old_cuda_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _raw(t: torch.Tensor) -> bytes:
    return t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_sha256(t: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(str(t.dtype).encode())
    h.update(json.dumps(list(t.shape), separators=(",", ":")).encode())
    h.update(_raw(t))
    return h.hexdigest()


def fields_sha256(fields: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in FIELDS:
        value = fields[name]
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        h.update(_raw(value))
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_calibration_manifest(path: str | Path) -> dict[str, object]:
    """Load a self-consistent calibration identity and bind its exact bytes."""
    source = Path(path).resolve()
    try:
        value = json.loads(source.read_text())
    except Exception as exc:
        raise ValueError(f"invalid calibration manifest {source}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"calibration manifest schema must be {CALIBRATION_SCHEMA!r}")
    for key in ("dataset", "capture_precision", "calibration_hash"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"calibration manifest {key!r} must be a non-empty string")
    if not re.fullmatch(r"[0-9a-f]{32}|[0-9a-f]{64}", value["calibration_hash"]):
        raise ValueError("calibration_hash must be lowercase 128- or 256-bit hex")
    for key in ("nsamples", "seqlen"):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int) or value[key] <= 0:
            raise ValueError(f"calibration manifest {key!r} must be a positive integer")
    if isinstance(value.get("seed"), bool) or not isinstance(value.get("seed"), int):
        raise ValueError("calibration manifest 'seed' must be an integer")
    claimed = value.get("identity_sha256")
    identity_payload = {key: item for key, item in value.items() if key != "identity_sha256"}
    expected = _canonical_sha256(identity_payload)
    if claimed != expected:
        raise ValueError(
            f"calibration identity_sha256 mismatch: expected {expected}, got {claimed!r}"
        )
    return {
        "path": str(source),
        "file_sha256": file_sha256(source),
        "identity_sha256": expected,
        "contract": value,
    }


def validate_prismaquant_checkout(
    path: str | Path, expected_commit: str
) -> dict[str, object]:
    """Bind the actual clean checkout and exact implementation sources."""
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise ValueError("--prismaquant-commit must be 40 lowercase hex characters")
    root = Path(path).resolve()
    git = shutil.which("git")
    if git:
        result = subprocess.run(
            [git, "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        commit = result.stdout.strip()
        dirty = subprocess.run(
            [git, "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if dirty:
            raise ValueError("PrismaQuant checkout has tracked modifications")
        tree_state = "tracked_clean_git_checked"
    else:
        commit = _checkout_head_without_git(root)
        tree_state = "git_unavailable_source_digests_bound"
    if commit != expected_commit:
        raise ValueError(
            f"PrismaQuant checkout mismatch: expected {expected_commit}, got {commit}"
        )
    sources = {
        "prismaquant/export_native_compressed.py": Path(enc.__file__).resolve(),
        "research/qtip_native_nvfp4_2026-08-30/native_nvfp4_ldlq.py": Path(__file__).resolve(),
    }
    digests: dict[str, str] = {}
    for relative, imported in sources.items():
        expected_path = (root / relative).resolve()
        if imported != expected_path or not expected_path.is_file():
            raise ValueError(
                f"imported source does not belong to pinned checkout: {relative}={imported}"
            )
        digests[relative] = file_sha256(expected_path)
    return {
        "commit": commit,
        "checkout": str(root),
        "tree_verification": tree_state,
        "source_sha256": digests,
    }


def quality_contract() -> dict[str, object]:
    return {
        "environment": dict(sorted(QUALITY_ENV.items())),
        "nvfp4_scale_rule": SCALE_RULE,
        "production_scale_levels": list(SCALE_LEVELS),
        "seven_level_heuristic_scale_levels": list(FULL_SCALE_LEVELS),
        "group_size": GROUP,
        "activation_aware_module_flags_for_native_terminals": {
            key: False for key in sorted(enc._ACT_AWARE_FLAGS)
        },
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "deterministic_algorithms": True,
    }


def device_identity(device: torch.device) -> dict[str, object]:
    identity: dict[str, object] = {
        "requested": str(device),
        "platform": platform.platform(),
        "container_hostname": socket.gethostname(),
    }
    if device.type != "cuda":
        identity["type"] = device.type
        return identity
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    props = torch.cuda.get_device_properties(index)
    identity.update({
        "type": "cuda", "index": index, "name": props.name,
        "compute_capability": [int(props.major), int(props.minor)],
        "total_memory_bytes": int(props.total_memory),
    })
    smi = shutil.which("nvidia-smi")
    if smi:
        probe = subprocess.run(
            [smi, "--query-gpu=driver_version,pci.bus_id,uuid",
             "--format=csv,noheader,nounits", "-i", str(index)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        parts = [part.strip() for part in probe.split(",")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"invalid nvidia-smi device identity: {probe!r}")
        identity.update({"driver_version": parts[0], "pci_bus_id": parts[1], "uuid": parts[2]})
    else:
        driver = Path("/proc/driver/nvidia/version")
        if not driver.is_file():
            raise ValueError("CUDA receipt requires a readable NVIDIA driver identity")
        identity["driver_version_text"] = driver.read_text().strip()
    return identity


def _publish_no_clobber(destination: Path, writer) -> None:
    """Publish one complete file atomically and refuse an existing result."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    )
    try:
        writer(temporary)
        if not temporary.is_file() or temporary.is_symlink():
            raise ValueError(f"publisher did not create one regular file: {temporary}")
        temporary.chmod(0o644)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def validate_publication_plan(
    root: str | Path, destinations: list[Path]
) -> tuple[Path, dict[Path, str]]:
    """Require a unique, one-root result set before any publication starts."""
    publication_root = Path(root).resolve()
    resolved = [path.resolve() for path in destinations]
    if len(set(resolved)) != len(resolved):
        raise ValueError("publication destinations must be unique")
    relative: dict[Path, str] = {}
    for destination in resolved:
        try:
            rel = destination.relative_to(publication_root)
        except ValueError as exc:
            raise ValueError(
                f"publication destination escapes root {publication_root}: {destination}"
            ) from exc
        if not rel.parts:
            raise ValueError("publication root itself cannot be an output file")
        relative[destination] = rel.as_posix()
    return publication_root, relative


def durable_uri(root: str, relative: str) -> str:
    base = root.strip().rstrip("/")
    if not base or any(ch in base for ch in ("\n", "\r", "\0")):
        raise ValueError("--durable-root-uri must be a non-empty single-line URI/path")
    return f"{base}/{relative}"


def validate_container_identity(value: str) -> str:
    identity = value.strip()
    if not re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", identity):
        raise ValueError(
            "--container-identity must be an image ID or repo digest ending in sha256:<64 hex>"
        )
    return identity


def validate_fields(fields: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    """Reject anything other than compressed-tensors native NVFP4 fields."""
    if set(fields) != set(FIELDS):
        raise ValueError(f"NVFP4 fields must be exactly {FIELDS}, got {sorted(fields)}")
    wp, ws = fields["weight_packed"], fields["weight_scale"]
    wg, xs = fields["weight_global_scale"], fields["input_global_scale"]
    if wp.ndim != 2 or wp.dtype != torch.uint8:
        raise ValueError("weight_packed must be rank-2 uint8")
    rows, cols = int(wp.shape[0]), int(wp.shape[1]) * 2
    if cols % GROUP:
        raise ValueError("packed width must be group-16 aligned")
    if ws.shape != (rows, cols // GROUP) or ws.dtype != torch.float8_e4m3fn:
        raise ValueError("weight_scale must be [out,in/16] float8_e4m3fn")
    if wg.dtype != torch.float32 or wg.numel() != 1:
        raise ValueError("weight_global_scale must be one float32 divisor")
    if xs.dtype != torch.float32 or xs.numel() != 1:
        raise ValueError("input_global_scale must be one float32 scalar")
    if not bool(torch.isfinite(ws.float()).all()) or bool((ws.float() < 0).any()):
        raise ValueError("weight scales must be finite and nonnegative")
    if not bool(torch.isfinite(wg).all()) or bool((wg <= 0).any()):
        raise ValueError("weight global divisor must be finite and positive")
    if not bool(torch.isfinite(xs).all()) or bool((xs <= 0).any()):
        raise ValueError("input global scale must be finite and positive")
    return rows, cols


def decode_fields(fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Decode using the exact E2M1 nibble and divisor convention vLLM reads."""
    rows, cols = validate_fields(fields)
    wp = fields["weight_packed"]
    lo, hi = (wp & 15).long(), ((wp >> 4) & 15).long()
    idx = torch.stack((lo, hi), -1).reshape(rows, cols)
    cb = enc._nvfp4_codebook(wp.device, torch.float32)
    value = cb[idx & 7]
    value = torch.where((idx & 8) != 0, -value, value)
    scale = fields["weight_scale"].float().unsqueeze(-1).expand(
        rows, cols // GROUP, GROUP).reshape(rows, cols)
    global_real = fields["weight_global_scale"].reshape(()).reciprocal()
    return value * scale * global_real


def payload_accounting(fields: Mapping[str, torch.Tensor]) -> dict[str, object]:
    rows, cols = validate_fields(fields)
    by_field = {k: int(v.numel() * v.element_size()) for k, v in fields.items()}
    total = sum(by_field.values())
    return {
        "bytes": total,
        "bytes_by_field": dict(sorted(by_field.items())),
        "n_weights": rows * cols,
        "bits_per_weight": 8.0 * total / (rows * cols),
    }


def _native_rtn(
    weight: torch.Tensor,
    global_real: torch.Tensor | None = None,
    scale_levels: tuple[float, ...] = SCALE_LEVELS,
) -> Arm:
    with fixed_contract(scale_levels):
        fields = enc._quantize_2d(
            weight.float(), "NVFP4",
            nvfp4_global_real_override=global_real,
            input_global_scale_override=1.0,
            gptq_enabled=False, scale_sweep_enabled=False,
            static_act_order_enabled=False, joint_scale_opt_enabled=False,
        )
    return Arm(fields, decode_fields(fields))


def rtn_arm(weight: torch.Tensor) -> Arm:
    """Arm A: native NVFP4 RTN using the production JSO scale rule."""
    return _native_rtn(weight)


def gptq_jso_arm(weight: torch.Tensor, activations: torch.Tensor) -> Arm:
    """Arm B: current PrismaQuant GPTQ + static act order + JSO."""
    with fixed_contract():
        fields = enc._quantize_2d(
            weight.float(), "NVFP4", input_global_scale_override=1.0,
            gptq_enabled=True, scale_sweep_enabled=False,
            static_act_order_enabled=True, joint_scale_opt_enabled=True,
            cached_activations=activations, act_clip_threshold=None,
            linear_name="research.qtip_native_nvfp4.one_linear",
        )
    return Arm(fields, decode_fields(fields))


def damped_hessian(activations: torch.Tensor, cols: int, device: torch.device):
    """Build the same preprocessed X and damped H used by arm B."""
    with fixed_contract():
        x = enc._activation_matrix_for_gptq(activations, cols, device=device)
    h = x.T @ x
    diag = h.diagonal()
    dead = diag <= 0
    alive = ~dead
    mean = (diag[alive].mean() if bool(alive.any()) else diag.new_ones(())).clamp_min(1e-12)
    if bool(dead.any()):
        h[dead, dead] = 1.0
    realized = DAMP * float(mean)
    h.diagonal().add_(realized)
    if not bool(torch.isfinite(h).all()):
        raise ValueError("non-finite Hessian")
    return x, h, realized


def qtip_block_unit_lower(hessian: torch.Tensor, block_size: int = GROUP) -> torch.Tensor:
    """Mirror pinned QTIP ``block_LDL`` and remove its identity blocks."""
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Hessian must be square")
    n = int(hessian.shape[0])
    if not n or n % block_size:
        raise ValueError("Hessian width must be block aligned")
    chol = torch.linalg.cholesky(hessian)
    lower = chol.clone()
    for first in range(0, n, block_size):
        last = first + block_size
        d = chol[first:last, first:last]
        lower[:, first:last] = torch.linalg.solve_triangular(
            d.T, chol[:, first:last].T, upper=True).T
        lower[first:last, first:last] = 0
    return lower


def qtip_native_arm(
    weight: torch.Tensor,
    activations: torch.Tensor,
    scale_levels: tuple[float, ...] = SCALE_LEVELS,
) -> Arm:
    """Arm C: QTIP reverse BlockLDLQ; standard codec at every terminal."""
    if weight.ndim != 2 or int(weight.shape[1]) % GROUP:
        raise ValueError("weight must be [out,in] with group-16 input width")
    source = weight.float()
    _x, h, _damp = damped_hessian(
        activations, int(source.shape[1]), source.device
    )
    return qtip_native_arm_from_hessian(source, h, scale_levels)


def qtip_native_arm_from_hessian(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    scale_levels: tuple[float, ...] = SCALE_LEVELS,
) -> Arm:
    """Arm C from an already constructed, matched-objective Hessian.

    This is the same stock-native terminal and reverse recurrence used by
    :func:`qtip_native_arm`.  It exists so activation-free corpora can supply
    their explicit Hessian contract without fabricating activation rows.
    """
    if weight.ndim != 2 or int(weight.shape[1]) % GROUP:
        raise ValueError("weight must be [out,in] with group-16 input width")
    source = weight.float()
    rows, cols = map(int, source.shape)
    h = hessian.to(device=source.device, dtype=torch.float32)
    if h.shape != (cols, cols):
        raise ValueError("hessian must be square over the weight input width")
    if not bool(torch.isfinite(h).all()):
        raise ValueError("hessian must be finite")
    if not torch.allclose(h, h.T, rtol=0.0, atol=1.0e-6):
        raise ValueError("hessian must be symmetric")
    lower = qtip_block_unit_lower(h)
    with fixed_contract(scale_levels):
        global_real = enc.nvfp4_global_real(
            source, group_size=GROUP, scale_rule=SCALE_RULE,
            snapped_scale_scoring=False, joint_scale_levels=scale_levels)

    q = torch.zeros_like(source)
    pblocks, sblocks, receipts = [], [], []
    wg = xs = None
    for first in range(cols - GROUP, -1, -GROUP):
        last = first + GROUP
        target = source[:, first:last]
        if last < cols:
            target = target + (source[:, last:] - q[:, last:]) @ lower[last:, first:last]
        terminal = _native_rtn(target, global_real, scale_levels)
        # Its decoded bytes, not an unprojected float candidate, feed the next step.
        q[:, first:last] = terminal.reconstruction
        pblocks.append(terminal.fields["weight_packed"])
        sblocks.append(terminal.fields["weight_scale"])
        if wg is None:
            wg, xs = terminal.fields["weight_global_scale"], terminal.fields["input_global_scale"]
        elif not torch.equal(wg, terminal.fields["weight_global_scale"]):
            raise AssertionError("terminal global-scale drift")
        receipts.append({
            "first_column": first, "last_column_exclusive": last,
            "fields_sha256": fields_sha256(terminal.fields),
            "legal_native_nvfp4": True,
        })
    assert wg is not None and xs is not None
    fields = {
        "weight_packed": torch.cat(tuple(reversed(pblocks)), 1),
        "weight_scale": torch.cat(tuple(reversed(sblocks)), 1),
        "weight_global_scale": wg, "input_global_scale": xs,
    }
    decoded = decode_fields(fields)
    if not torch.equal(decoded, q):
        raise AssertionError("assembled bytes differ from terminal recurrence")
    return Arm(fields, decoded, tuple(reversed(receipts)))


def qtip_native_arm_from_diagonal_hessian(
    weight: torch.Tensor,
    hessian_diagonal: torch.Tensor,
) -> Arm:
    """Exact Arm C specialization for a positive diagonal Hessian.

    With QTIP's group-16 BlockLDL geometry, a diagonal Hessian has an exactly
    zero strictly-lower feedback factor.  The stock-native terminals are thus
    independent RTN groups under the same tensor-global scale, exactly the
    ordinary native RTN field construction.  Retaining the diagonal avoids a
    pointless dense K-by-K allocation for activation-free GLM corpora.
    """

    if weight.ndim != 2 or int(weight.shape[1]) % GROUP:
        raise ValueError("weight must be [out,in] with group-16 input width")
    columns = int(weight.shape[1])
    # Validate the cheap structural contract before either input can trigger
    # a dtype conversion.  A malformed K-by-K BF16 Hessian is not allowed to
    # allocate the global FP32 matrix this specialization is meant to avoid.
    if hessian_diagonal.ndim != 1 or int(hessian_diagonal.numel()) != columns:
        raise ValueError("hessian_diagonal must be rank one over the input width")
    source = weight.float()
    diagonal = hessian_diagonal.detach().float()
    if not bool(torch.isfinite(diagonal).all()) or bool((diagonal <= 0).any()):
        raise ValueError("hessian_diagonal must be finite and strictly positive")
    arm = rtn_arm(source)
    receipts = tuple({
        "first_column": first,
        "last_column_exclusive": first + GROUP,
        "fields_sha256": fields_sha256({
            "weight_packed": arm.fields["weight_packed"][:, first // 2:(first + GROUP) // 2],
            "weight_scale": arm.fields["weight_scale"][:, first // GROUP:(first + GROUP) // GROUP],
            "weight_global_scale": arm.fields["weight_global_scale"],
            "input_global_scale": arm.fields["input_global_scale"],
        }),
        "legal_native_nvfp4": True,
        "blockldl_feedback_nonzero_count": 0,
        "diagonal_hessian_specialization": True,
    } for first in range(0, columns, GROUP))
    return Arm(arm.fields, arm.reconstruction, receipts)


def qtip_native_seven_level_scale_arm(
    weight: torch.Tensor, activations: torch.Tensor
) -> Arm:
    """QTIP BlockLDLQ plus the exporter's seven max-to-level heuristics."""
    return qtip_native_arm(weight, activations, FULL_SCALE_LEVELS)


def _metrics(
    weight: torch.Tensor,
    x: torch.Tensor,
    hessian: torch.Tensor,
    arm: Arm,
) -> dict[str, object]:
    w, q = weight.float(), arm.reconstruction.float()
    error = w - q
    wsse, wenergy = float(error.square().sum()), float(w.square().sum())
    out, eout = x @ w.T, x @ error.T
    osse, oenergy = float(eout.square().sum()), float(out.square().sum())
    wn, on = wsse / max(wenergy, 1e-30), osse / max(oenergy, 1e-30)
    hnum = float(((error @ hessian) * error).sum())
    hden = float(((w @ hessian) * w).sum())
    hproxy = hnum / max(hden, 1e-30)
    return {
        "fields_sha256": fields_sha256(arm.fields),
        "serialized": payload_accounting(arm.fields),
        "weight_mse": wsse / w.numel(),
        "weight_nsse": wn, "weight_snr_db": -10 * math.log10(max(wn, 1e-300)),
        "activation_output_mse": osse / eout.numel(),
        "activation_output_nsse": on,
        "activation_output_snr_db": -10 * math.log10(max(on, 1e-300)),
        "regularized_untransformed_hessian_proxy_nsse": hproxy,
        "regularized_untransformed_hessian_proxy_snr_db": -10 * math.log10(max(hproxy, 1e-300)),
        "regularized_untransformed_hessian_proxy_numerator": hnum,
        "regularized_untransformed_hessian_proxy_denominator": hden,
        "hessian_proxy_space": "untransformed_original_linear",
        "terminal_blocks": list(arm.terminal_blocks),
    }


def compare_one_linear(weight: torch.Tensor, activations: torch.Tensor):
    """Run matched-input, matched-bpw A/B/C/C2 arms and return report + fields."""
    if weight.ndim != 2 or activations.ndim < 2:
        raise ValueError("expected weight [out,in] and activations [...,in]")
    rows, cols = map(int, weight.shape)
    if rows <= 0 or cols % GROUP or int(activations.shape[-1]) != cols:
        raise ValueError("empty, misaligned, or width-mismatched input")
    if not bool(torch.isfinite(weight.float()).all()) or not bool(torch.isfinite(activations.float()).all()):
        raise ValueError("inputs must be finite")
    with torch.inference_mode():
        a = rtn_arm(weight)
        gate_before = dict(enc._DO_NO_HARM_STATS)
        b = gptq_jso_arm(weight, activations)
        gate_after = dict(enc._DO_NO_HARM_STATS)
        c = qtip_native_arm(weight, activations)
        d = qtip_native_seven_level_scale_arm(weight, activations)
        x, hessian, realized = damped_hessian(activations, cols, weight.device)
    arms = {
        ARM_NAMES[0]: a,
        ARM_NAMES[1]: b,
        ARM_NAMES[2]: c,
        ARM_NAMES[3]: d,
    }
    report = {
        "schema": SCHEMA, "status": "ok",
        "scope": "research_only_one_linear_no_production_registration",
        "weight": {"shape": list(weight.shape), "dtype": str(weight.dtype), "sha256": tensor_sha256(weight)},
        "activations": {"shape": list(activations.shape), "dtype": str(activations.dtype),
                        "sha256": tensor_sha256(activations), "clip_quantile": CLIP_QUANTILE,
                        "preprocessed_rows": int(x.shape[0]),
                        "preprocessed_sha256": tensor_sha256(x)},
        "hessian": {"construction": "X.T@X with production dead-channel convention",
                    "damp_fraction": DAMP, "realized_diagonal_damping": realized,
                    "block_size": GROUP, "sha256": tensor_sha256(hessian)},
        "quality_contract": quality_contract(),
        "native_nvfp4_contract": {"group_size": GROUP, "element_grid": "E2M1",
            "group_scale_dtype": "torch.float8_e4m3fn", "tensor_global": "float32_divisor",
            "scale_rule": SCALE_RULE,
            "scale_levels_by_arm": {"A_B_C": list(SCALE_LEVELS), "C2": list(FULL_SCALE_LEVELS)},
            "C2_is_exhaustive_e4m3_scale_byte_search": False,
            "scale_byte_is_semantic_not_side_channel": True,
            "fields": list(FIELDS)},
        "arm_contracts": {
            ARM_NAMES[0]: "RTN optimizer; JSO names the final native group/tensor scale search only",
            ARM_NAMES[1]: "GPTQ static activation order with joint_scale_opt, then final native JSO packing",
            ARM_NAMES[2]: "QTIP BlockLDLQ recurrence with production {6,4} native terminal JSO",
            ARM_NAMES[3]: "same QTIP recurrence with seven max-to-E2M1-level scale heuristics; not exhaustive E4M3 search",
        },
        "transferred": ["activation Hessian", "block-unit-lower Cholesky",
                        "reverse block schedule", "later-block error feedback"],
        "excluded": {
            "tail_biting_trellis": "QTIP stateful codebook/wire/runtime, not native NVFP4",
            "signs_hadamards": "need runtime inverse or separately proved model-wide fold",
            "SU_SV": "QTIP sidecars have no stock NVFP4 representation",
            "allocation": "whole-model mixed-rate allocation is outside this one-Linear isolate",
        },
        "control_observations": {
            "arm_B_do_no_harm_stat_delta": {
                key: int(gate_after.get(key, 0) - gate_before.get(key, 0))
                for key in sorted(set(gate_before) | set(gate_after))
            },
            "causal_interpretation_requires_a_recorded_pre_gate_candidate": True,
        },
        "arms": {name: _metrics(weight, x, hessian, arm) for name, arm in arms.items()},
    }
    return report, arms


def _checkout_head_without_git(root: Path) -> str:
    """Read a detached/ref HEAD without making the CUDA image ship Git."""
    dotgit = root / ".git"
    if dotgit.is_file():
        marker = dotgit.read_text().strip()
        if not marker.startswith("gitdir: "):
            raise ValueError(f"invalid gitdir marker in {dotgit}")
        gitdir = Path(marker.removeprefix("gitdir: "))
        if not gitdir.is_absolute():
            gitdir = (root / gitdir).resolve()
    else:
        gitdir = dotgit
    head_path = gitdir / "HEAD"
    if not head_path.is_file():
        raise ValueError(f"missing Git HEAD metadata in {gitdir}")
    head = head_path.read_text().strip()
    if re.fullmatch(r"[0-9a-f]{40}", head):
        return head
    if not head.startswith("ref: "):
        raise ValueError(f"invalid Git HEAD value {head!r}")
    ref = head.removeprefix("ref: ")
    if not re.fullmatch(r"refs/[A-Za-z0-9._/-]+", ref) or ".." in ref.split("/"):
        raise ValueError(f"invalid Git ref {ref!r}")
    roots = [gitdir]
    commondir = gitdir / "commondir"
    if commondir.is_file():
        common = Path(commondir.read_text().strip())
        if not common.is_absolute():
            common = (gitdir / common).resolve()
        roots.append(common)
    for metadata_root in roots:
        loose = metadata_root / ref
        if loose.is_file():
            commit = loose.read_text().strip()
            if re.fullmatch(r"[0-9a-f]{40}", commit):
                return commit
        packed = metadata_root / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line and not line.startswith(("#", "^")):
                    commit, name = line.split(" ", 1)
                    if name == ref and re.fullmatch(r"[0-9a-f]{40}", commit):
                        return commit
    raise ValueError(f"cannot resolve Git ref {ref!r}")


def validate_qtip_checkout(path: str | Path) -> dict[str, object]:
    root = Path(path).resolve()
    if shutil.which("git"):
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
    else:
        commit = _checkout_head_without_git(root)
    digests = {}
    for relpath, expected in QTIP_SOURCE_FILES.items():
        source = root / relpath
        if not source.is_file():
            raise ValueError(f"missing QTIP source {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        digests[relpath] = digest
        if digest != expected:
            raise ValueError(f"QTIP source mismatch: {relpath} sha256={digest}")
    if commit != QTIP_PINNED_COMMIT:
        raise ValueError(f"QTIP source mismatch: commit={commit}")
    return {"repository": QTIP_REPOSITORY, "commit": commit,
            "source_sha256": digests,
            "seams": {"regularize_and_incoherence": "lib/algo/finetune.py:108-188",
                      "block_LDL": "lib/utils/math_utils.py:14-42",
                      "LDLQ": "lib/algo/ldlq.py:16-80",
                      "tail_biting": "lib/codebook/bitshift.py:261-327"}}


def _load(path: str, key: str | None) -> torch.Tensor:
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(p), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if key is None:
                if len(keys) != 1:
                    raise ValueError("explicit tensor key required")
                key = keys[0]
            if key not in keys:
                raise KeyError(f"{key!r} is not present in {p}")
            # Pull only the named Linear.  Loading a whole sharded or monolithic
            # checkpoint to run a one-Linear isolate would violate the same
            # streaming discipline this repository enforces in production.
            return handle.get_tensor(key)
    else:
        value = torch.load(p, map_location="cpu", weights_only=True)
    if isinstance(value, torch.Tensor):
        if key: raise ValueError("key supplied for a single tensor")
        return value
    if key is None:
        if len(value) != 1: raise ValueError("explicit tensor key required")
        return next(iter(value.values()))
    return value[key]


def main(argv=None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weight", required=True); ap.add_argument("--weight-key")
    ap.add_argument("--activations", required=True); ap.add_argument("--activations-key")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", required=True)
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--profile-dir", required=True)
    ap.add_argument("--publication-root", required=True)
    ap.add_argument("--durable-root-uri", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--container-identity", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--calibration-manifest", required=True)
    ap.add_argument("--prismaquant-checkout", required=True)
    ap.add_argument("--prismaquant-commit", required=True)
    ap.add_argument("--qtip-checkout", default="/home/rob/dq-runs/qtip-reference-20260830")
    args = ap.parse_args(actual_argv)
    output = Path(args.output).resolve()
    outdir = Path(args.artifacts_dir).resolve()
    trace = Path(args.profile_dir).resolve() / "one_linear_trace.json"
    planned = [output, trace]
    planned.extend(outdir / f"{name}.safetensors" for name in ARM_NAMES)
    publication_root, relative_paths = validate_publication_plan(
        args.publication_root, planned
    )
    if publication_root.exists() and any(publication_root.iterdir()):
        raise FileExistsError(
            f"v2 requires a fresh empty publication root: {publication_root}"
        )
    durable_base = args.durable_root_uri.strip().rstrip("/")
    for relative in relative_paths.values():
        durable_uri(durable_base, relative)
    occupied = [str(path) for path in planned if path.exists()]
    if occupied:
        raise FileExistsError(f"refusing to overwrite research outputs: {occupied}")
    if not args.host.strip():
        raise ValueError("--host must name the physical execution host")
    if not args.model_id.strip():
        raise ValueError("--model-id must be non-empty")
    container = validate_container_identity(args.container_identity)
    prismaquant_source = validate_prismaquant_checkout(
        args.prismaquant_checkout, args.prismaquant_commit
    )
    qtip_source = validate_qtip_checkout(args.qtip_checkout)
    calibration = validate_calibration_manifest(args.calibration_manifest)
    device = torch.device(args.device)
    device_provenance = device_identity(device)
    weight_path = Path(args.weight).resolve()
    activation_path = Path(args.activations).resolve()
    input_sources = {
        "weight": {
            "path": str(weight_path), "key": args.weight_key,
            "file_sha256": file_sha256(weight_path),
        },
        "activations": {
            "path": str(activation_path), "key": args.activations_key,
            "file_sha256": file_sha256(activation_path),
        },
    }
    w = _load(str(weight_path), args.weight_key).to(device)
    x = _load(str(activation_path), args.activations_key).to(device)
    Path(args.profile_dir).mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    )
    profiler.__enter__()
    try:
        report, arms = compare_one_linear(w, x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        profiler.__exit__(None, None, None)
    report["qtip_source"] = qtip_source
    report["prismaquant_source"] = prismaquant_source
    report["calibration"] = calibration
    report["input_sources"] = input_sources
    report["weight"]["model_id"] = args.model_id
    report["execution"] = {
        "physical_host": args.host.strip(),
        "container_identity": container,
        "device": device_provenance,
        "torch_version": torch.__version__,
        "cuda_toolkit_version": torch.version.cuda,
        "command": [str(Path(__file__).resolve()), *actual_argv],
        "command_sha256": _canonical_sha256(actual_argv),
        "working_directory": str(Path.cwd().resolve()),
        "activation_precision": "source tensor recorded exactly; FP32 Hessian isolate; not a served W4A4 claim",
    }
    published: list[dict[str, object]] = []

    def record_member(path: Path, kind: str, **extra: object) -> dict[str, object]:
        resolved = path.resolve()
        relative = relative_paths[resolved]
        item: dict[str, object] = {
            "kind": kind,
            "relative_path": relative,
            "durable_uri": durable_uri(durable_base, relative),
            "file_sha256": file_sha256(resolved),
            "bytes": resolved.stat().st_size,
            "mode": oct(resolved.stat().st_mode & 0o777),
        }
        item.update(extra)
        published.append(item)
        return item

    _publish_no_clobber(
        trace, lambda temp: profiler.export_chrome_trace(str(temp))
    )
    report["execution"]["torch_profiler_trace"] = record_member(
        trace, "torch_profiler_trace"
    )
    from safetensors.torch import save_file
    outdir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, dict[str, object]] = {}
    for name, arm in arms.items():
        path = outdir / f"{name}.safetensors"
        tensors = {key: value.detach().contiguous().cpu() for key, value in arm.fields.items()}
        _publish_no_clobber(
            path,
            lambda temp, tensors=tensors, name=name: save_file(
                tensors, str(temp),
                metadata={"schema": SCHEMA, "arm": name, "format": "native NVFP4"}),
        )
        saved[name] = record_member(
            path, "native_nvfp4_fields", arm=name,
            fields_sha256=fields_sha256(tensors),
        )
    report["native_field_artifacts"] = saved
    receipt_relative = relative_paths[output]
    report["publication"] = {
        "semantics": "per_file_no_clobber_receipt_is_commit_marker",
        "publication_root": str(publication_root),
        "durable_root_uri": durable_base,
        "members_published_before_commit_marker": published,
        "commit_marker": {
            "relative_path": receipt_relative,
            "durable_uri": durable_uri(durable_base, receipt_relative),
        },
        "incomplete_rule": "any member set without the receipt commit marker is invalid and must not be resumed in place",
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _publish_no_clobber(output, lambda temp: temp.write_text(payload))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
