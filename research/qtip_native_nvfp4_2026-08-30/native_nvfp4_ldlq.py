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
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import torch

from prismaquant import export_native_compressed as enc


SCHEMA = "prismaquant.research.qtip_native_nvfp4_one_linear.v1"
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
    "C2_qtip_block_ldl_native_nvfp4_full_jso",
)
_LOCK = threading.RLock()


@dataclass(frozen=True)
class Arm:
    fields: Mapping[str, torch.Tensor]
    reconstruction: torch.Tensor
    terminal_blocks: tuple[dict[str, object], ...] = ()


@contextlib.contextmanager
def fixed_contract(scale_levels: tuple[float, ...] = SCALE_LEVELS) -> Iterator[None]:
    """Pin legacy env-backed render choices and restore the caller exactly."""
    values = {
        "PRISMAQUANT_ACT_CLIP_QUANTILE": str(CLIP_QUANTILE),
        "PRISMAQUANT_GPTQ_DAMP": str(DAMP),
        "PRISMAQUANT_GPTQ_DAMP_SWEEP": "0",
        "PRISMAQUANT_GPTQ_DAMP_ROLES": "",
        "PRISMAQUANT_DO_NO_HARM": "1",
        "PRISMAQUANT_NVFP4_SNAPPED_SCALE_SCORING": "0",
    }
    with _LOCK:
        old_env = {k: os.environ.get(k) for k in values}
        old_levels = enc._NVFP4_JOINT_SCALE_LEVELS
        try:
            os.environ.update(values)
            enc._NVFP4_JOINT_SCALE_LEVELS = scale_levels
            with enc._temporary_export_nvfp4_scale_rule(SCALE_RULE):
                yield
        finally:
            enc._NVFP4_JOINT_SCALE_LEVELS = old_levels
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


def _publish_no_clobber(destination: Path, writer) -> None:
    """Publish one complete file atomically and refuse an existing result."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        if temporary.exists():
            raise FileExistsError(f"stale temporary output exists: {temporary}")
        writer(temporary)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


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
    rows, cols = map(int, source.shape)
    _x, h, _damp = damped_hessian(activations, cols, source.device)
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


def qtip_native_full_scale_arm(weight: torch.Tensor, activations: torch.Tensor) -> Arm:
    """QTIP BlockLDLQ plus the exporter's existing full legal JSO candidate grid."""
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
        b = gptq_jso_arm(weight, activations)
        c = qtip_native_arm(weight, activations)
        d = qtip_native_full_scale_arm(weight, activations)
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
                        "preprocessed_rows": int(x.shape[0])},
        "hessian": {"construction": "X.T@X with production dead-channel convention",
                    "damp_fraction": DAMP, "realized_diagonal_damping": realized,
                    "block_size": GROUP},
        "native_nvfp4_contract": {"group_size": GROUP, "element_grid": "E2M1",
            "group_scale_dtype": "torch.float8_e4m3fn", "tensor_global": "float32_divisor",
            "scale_rule": SCALE_RULE,
            "scale_levels_by_arm": {"A_B_C": list(SCALE_LEVELS), "C2": list(FULL_SCALE_LEVELS)},
            "scale_byte_is_semantic_not_side_channel": True,
            "fields": list(FIELDS)},
        "arm_contracts": {
            ARM_NAMES[0]: "RTN optimizer; JSO names the final native group/tensor scale search only",
            ARM_NAMES[1]: "GPTQ static activation order with joint_scale_opt, then final native JSO packing",
            ARM_NAMES[2]: "QTIP BlockLDLQ recurrence with production {6,4} native terminal JSO",
            ARM_NAMES[3]: "same QTIP recurrence with existing opt-in full native terminal JSO grid",
        },
        "transferred": ["activation Hessian", "block-unit-lower Cholesky",
                        "reverse block schedule", "later-block error feedback"],
        "excluded": {
            "tail_biting_trellis": "QTIP stateful codebook/wire/runtime, not native NVFP4",
            "signs_hadamards": "need runtime inverse or separately proved model-wide fold",
            "SU_SV": "QTIP sidecars have no stock NVFP4 representation",
            "allocation": "whole-model mixed-rate allocation is outside this one-Linear isolate",
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
    loose = gitdir / ref
    if loose.is_file():
        return loose.read_text().strip()
    packed = gitdir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == ref:
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weight", required=True); ap.add_argument("--weight-key")
    ap.add_argument("--activations", required=True); ap.add_argument("--activations-key")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", required=True)
    ap.add_argument("--artifacts-dir")
    ap.add_argument("--profile-dir")
    ap.add_argument("--qtip-checkout", default="/home/rob/dq-runs/qtip-reference-20260830")
    args = ap.parse_args(argv)
    output = Path(args.output).resolve()
    planned = [output]
    if args.artifacts_dir:
        outdir = Path(args.artifacts_dir).resolve()
        planned.extend(outdir / f"{name}.safetensors" for name in ARM_NAMES)
    if args.profile_dir:
        trace = Path(args.profile_dir).resolve() / "one_linear_trace.json"
        planned.append(trace)
    occupied = [str(path) for path in planned if path.exists()]
    if occupied:
        raise FileExistsError(f"refusing to overwrite research outputs: {occupied}")
    device = torch.device(args.device)
    w = _load(args.weight, args.weight_key).to(device)
    x = _load(args.activations, args.activations_key).to(device)
    source = validate_qtip_checkout(args.qtip_checkout)
    profiler = None
    if args.profile_dir:
        Path(args.profile_dir).mkdir(parents=True, exist_ok=True)
        acts = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda": acts.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=acts, record_shapes=True, profile_memory=True)
        profiler.__enter__()
    try:
        report, arms = compare_one_linear(w, x)
        if device.type == "cuda": torch.cuda.synchronize(device)
    finally:
        if profiler: profiler.__exit__(None, None, None)
    report["qtip_source"] = source
    report["execution"] = {"device": str(device), "torch_version": torch.__version__,
                           "cuda_version": torch.version.cuda,
                           "activation_precision": "BF16/FP32 Hessian isolate; not a served W4A4 claim"}
    if profiler:
        trace = Path(args.profile_dir).resolve() / "one_linear_trace.json"
        _publish_no_clobber(trace, lambda temp: profiler.export_chrome_trace(str(temp)))
        report["execution"]["torch_profiler_trace"] = str(trace)
    if args.artifacts_dir:
        from safetensors.torch import save_file
        outdir = Path(args.artifacts_dir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
        saved = {}
        for name, arm in arms.items():
            p = outdir / f"{name}.safetensors"
            tensors = {k: v.detach().contiguous().cpu() for k, v in arm.fields.items()}
            _publish_no_clobber(
                p,
                lambda temp, tensors=tensors, name=name: save_file(
                    tensors, str(temp),
                    metadata={"schema": SCHEMA, "arm": name, "format": "native NVFP4"}),
            )
            saved[name] = str(p)
        report["native_field_artifacts"] = saved
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _publish_no_clobber(output, lambda temp: temp.write_text(payload))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
