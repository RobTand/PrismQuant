"""format_registry.py — extensible quantization format catalog.

A FormatSpec describes everything the pipeline needs to treat a format
uniformly:

  - name                    canonical identifier
  - weight_bits             weight element width
  - group_size              per-group scale granularity; 0 = per-channel
  - weight_element_dtype    torch dtype used on disk
  - scale_bits              bits consumed by per-group scales
  - scale_dtype_name        human-readable scale dtype
  - effective_bits          total bits/param (weight + scale amortized)
  - autoround_config()      dict AutoRound consumes via --layer_config
  - quantize_dequantize(w)  apply RTN in-place (for closed-loop MSE)
  - min_capability_sm       minimum SM arch (useful for hardware filter)

Users register new formats with @register_format. New hardware formats
(e.g. a future MXFP6 variant or Ada W4A8) can be added without touching
core code.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Callable

import torch


@dataclass
class FormatSpec:
    name: str
    weight_bits: int
    group_size: int          # 0 means per-channel; >=1 = block size
    scale_bits: int          # bits per scale element
    scale_dtype_name: str    # "fp8_e4m3", "uint8_e8m0", "fp32", ...
    weight_element_dtype: str   # "fp4_e2m1", "fp8_e4m3", "fp6_e3m2", "int4", ...
    scale_block_shape: tuple[int, int] | None = None
    act_bits: int | None = None   # None = no activation quant (W8A16)
    act_dtype_name: str | None = None
    act_group_size: int | None = None
    family: str = "generic"       # "nv", "mx", "int", "fp"
    min_capability_sm: int = 80   # minimum CUDA compute capability
    # The autoround layer_config dict. Leave as callable for lazy config.
    autoround_config: Callable[[], dict] = field(default=lambda: {})
    # RTN quantize+dequantize, returns the rounded tensor (same shape+dtype).
    quantize_dequantize: Callable[[torch.Tensor], torch.Tensor] = field(default=lambda x: x)
    # Optional activation RTN path. Formats with A16 / BF16 activations should
    # leave this as identity; W4A4/W8A8 style formats should provide the
    # matching activation-side quantizer so functional-cost measurement reflects
    # the actual serving bucket rather than weight-only error.
    activation_quantize_dequantize: Callable[[torch.Tensor], torch.Tensor] = field(
        default=lambda x: x
    )

    @property
    def effective_bits(self) -> float:
        """Average bits per parameter accounting for scales."""
        # Backward-compatible fallback when no layer shape is available.
        if self.group_size == 0:
            if self.scale_bits == 0:
                return float(self.weight_bits)
            # True overhead depends on the layer shape. Keep a small, explicit
            # fallback here so older code doesn't crash, but new allocation code
            # should call effective_bits_for_shape().
            return float(self.weight_bits) + 0.02
        if self.scale_block_shape is not None:
            rows, cols = self.scale_block_shape
            return float(self.weight_bits) + float(self.scale_bits) / (rows * cols)
        return float(self.weight_bits) + float(self.scale_bits) / self.group_size

    def scale_count_for_shape(self, shape: tuple[int, ...]) -> int:
        """Return the number of scale values needed for a tensor shape.

        Assumptions:
          - For block/group quantization (group_size > 0), groups are taken
            along the innermost dimension and repeated for every outer row.
          - For per-channel formats (group_size == 0), one scale is used per
            output channel / row, which for Linear weights is shape[0].
        """
        if len(shape) == 0:
            return 0
        if self.scale_bits == 0:
            return 0
        if self.scale_block_shape is not None:
            if len(shape) < 2:
                n_params = int(shape[0])
                rows, cols = self.scale_block_shape
                return math.ceil(n_params / (rows * cols))
            rows, cols = self.scale_block_shape
            outer = int(math.prod(shape[:-2])) if len(shape) > 2 else 1
            return (
                outer
                * math.ceil(int(shape[-2]) / rows)
                * math.ceil(int(shape[-1]) / cols)
            )
        if self.group_size == 0:
            return int(shape[0]) if len(shape) >= 1 else 1
        if len(shape) == 1:
            n_params = int(shape[0])
            return math.ceil(n_params / self.group_size)
        outer = int(math.prod(shape[:-1]))
        inner = int(shape[-1])
        return outer * math.ceil(inner / self.group_size)

    def memory_bytes_for_shape(self, shape: tuple[int, ...]) -> int:
        """Exact-ish serialized size for a tensor in this format."""
        n_params = int(math.prod(shape)) if len(shape) else 1
        weight_bytes = math.ceil(n_params * self.weight_bits / 8.0)
        scale_bytes = math.ceil(self.scale_count_for_shape(shape) * self.scale_bits / 8.0)
        return weight_bytes + scale_bytes

    def effective_bits_for_shape(self, shape: tuple[int, ...]) -> float:
        n_params = int(math.prod(shape)) if len(shape) else 1
        return 8.0 * self.memory_bytes_for_shape(shape) / max(n_params, 1)


REGISTRY: dict[str, FormatSpec] = {}
FORMAT_ALIASES: dict[str, str] = {
    # Historical artifacts and launchers used the short OCP-MX default name.
    # Keep it accepted at every input boundary, but normalize persisted solver
    # and measurement output to the explicit FP8 variant.
    "MXFP8": "MXFP8_E4M3",
}

FORMAT_DISPLAY_NAMES: dict[str, str] = {
    # Keep the short public-facing name in logs, reports, and test-visible
    # outputs even though internal persistence canonicalizes to MXFP8_E4M3.
    "MXFP8_E4M3": "MXFP8",
}


def register_format(spec: FormatSpec) -> FormatSpec:
    REGISTRY[spec.name] = spec
    return spec


def canonical_format_name(name: str) -> str:
    return FORMAT_ALIASES.get(name, name)


def display_format_name(name: str) -> str:
    return FORMAT_DISPLAY_NAMES.get(name, name)


def aliases_for(name: str) -> tuple[str, ...]:
    canonical = canonical_format_name(name)
    aliases = [alias for alias, target in FORMAT_ALIASES.items()
               if target == canonical]
    return (canonical, *aliases)


# -----------------------------------------------------------------------
# Reference format implementations
# -----------------------------------------------------------------------
# Helpers for RTN quantization reference impls.

def _rtn_uniform_int(w: torch.Tensor, bits: int, group_size: int,
                     symmetric: bool = True) -> torch.Tensor:
    """Round-to-nearest uniform-integer quantizer with optional group scaling."""
    orig_shape = w.shape
    out_f, in_f = w.shape[-2], w.shape[-1]
    w2 = w.reshape(-1, in_f).float()
    if group_size > 0 and group_size < in_f:
        w2 = w2.reshape(-1, in_f // group_size, group_size)
    else:
        w2 = w2.unsqueeze(1)
    # Per-group max for scale
    max_abs = w2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    if symmetric:
        levels = (1 << (bits - 1)) - 1
        scale = max_abs / levels
        q = torch.round(w2 / scale).clamp(-levels - 1, levels)
        w_rec = q * scale
    else:
        levels = (1 << bits) - 1
        w_min = w2.amin(dim=-1, keepdim=True)
        w_max = w2.amax(dim=-1, keepdim=True)
        scale = (w_max - w_min) / levels
        zp = torch.round(-w_min / scale.clamp_min(1e-8))
        q = torch.round(w2 / scale.clamp_min(1e-8) + zp).clamp(0, levels)
        w_rec = (q - zp) * scale
    return w_rec.reshape(orig_shape).to(w.dtype)


def _snap_scale_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Snap a real-valued per-group scale to the nearest power of two.

    The OCP MX spec encodes the per-block scale as an 8-bit E8M0 value:
    unsigned, exponent-only, range 2^(-127) to 2^127. Representable
    values are exactly the powers of two. Using a real-valued scale
    (the previous behavior) under-estimates RTN error because the actual
    serving path will round-trip through the E8M0 grid, introducing
    extra error proportional to the scale's distance from a power of
    two.

    For NV (non-MX) formats, scales are FP8 and effectively continuous;
    no snapping is applied.
    """
    log2_s = torch.log2(scale.clamp_min(2.0 ** -127))
    snapped_exp = torch.round(log2_s).clamp(-127.0, 127.0)
    return torch.pow(2.0, snapped_exp)


def _rtn_fp_codebook(w: torch.Tensor, codebook: torch.Tensor,
                     group_size: int, mx_scale: bool = False) -> torch.Tensor:
    """Round to nearest value in a small FP codebook, with per-group scaling.

    Vectorized via torch.bucketize on the sorted codebook. For each scaled
    weight value x, we binary-search the codebook to find the two bracketing
    entries and pick the closer one. O(N log K) instead of the O(N * K)
    pairwise-distance approach, with 0 extra-dim allocations.

    When `mx_scale=True`, the per-group scale is snapped to the nearest
    power of two (E8M0). This matches the OCP MX serving path; without
    it, RTN error for MX formats is slightly under-estimated.
    """
    orig_shape = w.shape
    in_f = w.shape[-1]
    w2 = w.reshape(-1, in_f).float()
    if group_size > 0 and group_size < in_f:
        w2 = w2.reshape(-1, in_f // group_size, group_size)
    else:
        w2 = w2.unsqueeze(1)

    cb = _codebook_on_device(
        codebook,
        device=w2.device,
        dtype=torch.float32,
    )
    cmax = cb.abs().max()
    max_abs = w2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = max_abs / cmax
    if mx_scale:
        scale = _snap_scale_e8m0(scale)
    x = w2 / scale                                    # shape (..., group)

    # Bucketize returns the insertion index: cb[idx-1] <= x < cb[idx].
    idx = torch.bucketize(x.contiguous(), cb)
    idx_lo = (idx - 1).clamp_min(0)
    idx_hi = idx.clamp_max(cb.numel() - 1)
    lo = cb[idx_lo]
    hi = cb[idx_hi]
    choose_hi = (hi - x).abs() < (x - lo).abs()
    q = torch.where(choose_hi, hi, lo)

    w_rec = q * scale
    return w_rec.reshape(orig_shape).to(w.dtype)


# FP codebooks
def _e2m1_codebook() -> torch.Tensor:
    # 4-bit: 1 sign + 2 exp + 1 mantissa.  Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    signed = [0.0] + [+v for v in vals[1:]] + [-v for v in vals[1:]]
    return torch.tensor(sorted(set(signed)), dtype=torch.float32)


def _e3m2_codebook() -> torch.Tensor:
    # 6-bit FP e3m2: 1s + 3 exp + 2 mantissa. 64 codes.
    codes = set([0.0])
    for exp in range(8):
        for m in range(4):
            val = (1 + m / 4) * (2 ** (exp - 3))
            codes.add(+val); codes.add(-val)
    return torch.tensor(sorted(codes), dtype=torch.float32)


def _e2m3_codebook() -> torch.Tensor:
    # 6-bit FP e2m3: 1s + 2 exp + 3 mantissa. 64 codes.
    codes = set([0.0])
    for exp in range(4):
        for m in range(8):
            val = (1 + m / 8) * (2 ** (exp - 1))
            codes.add(+val); codes.add(-val)
    return torch.tensor(sorted(codes), dtype=torch.float32)


def _e4m3_codebook() -> torch.Tensor:
    # 8-bit FP e4m3 (no inf). 256 codes, covering ±448.
    # We compute it analytically from the OCP FP8 spec (nan reserved).
    codes = set([0.0])
    for exp in range(16):
        for m in range(8):
            if exp == 0:
                val = (m / 8) * (2 ** -6)  # subnormals
            else:
                val = (1 + m / 8) * (2 ** (exp - 7))
            codes.add(+val); codes.add(-val)
    # Clip to ±448 per spec (remove overflowing exponents)
    return torch.tensor(sorted(c for c in codes if abs(c) <= 448.0),
                        dtype=torch.float32)


def _e5m2_codebook() -> torch.Tensor:
    # 8-bit FP e5m2. Wider range, less mantissa precision.
    codes = set([0.0])
    for exp in range(32):
        for m in range(4):
            if exp == 0:
                val = (m / 4) * (2 ** -14)
            else:
                val = (1 + m / 4) * (2 ** (exp - 15))
            codes.add(+val); codes.add(-val)
    return torch.tensor(sorted(codes), dtype=torch.float32)


_CODEBOOKS = {
    "fp4_e2m1": _e2m1_codebook(),
    "fp6_e3m2": _e3m2_codebook(),
    "fp6_e2m3": _e2m3_codebook(),
    "fp8_e4m3": _e4m3_codebook(),
    "fp8_e5m2": _e5m2_codebook(),
}

_CODEBOOK_DEVICE_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _codebook_on_device(
    codebook: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if codebook.device == device and codebook.dtype == dtype:
        return codebook.contiguous()
    key = (id(codebook), str(device), dtype)
    cached = _CODEBOOK_DEVICE_CACHE.get(key)
    if cached is not None and cached.device == device and cached.dtype == dtype:
        return cached
    source = codebook
    if (
        torch.device(device).type == "cuda"
        and codebook.device.type == "cpu"
        and torch.cuda.is_available()
        and not codebook.is_pinned()
    ):
        try:
            source = codebook.pin_memory()
        except RuntimeError:
            source = codebook
    cached = source.to(
        device=device,
        dtype=dtype,
        non_blocking=bool(source.device.type == "cpu" and source.is_pinned()),
    ).contiguous()
    _CODEBOOK_DEVICE_CACHE[key] = cached
    return cached


def _make_rtn(codebook_name: str, group_size: int, mx_scale: bool = False):
    """Return a closure that runs RTN through ``codebook`` at ``group_size``.

    The hot path is wrapped in ``torch.compile`` (mode='reduce-overhead',
    dynamic=False) by default — micro-benchmark on Blackwell + cu130 +
    torch 2.11 shows ~10x speedup on per-Linear activation RTN
    (12 ms eager → 1.2 ms compiled, max numerical diff 5e-7).  This
    matters most on the polish hot path where the closure is called
    once per Linear per forward, ~497 calls per measurement on 27B,
    several hundred measurements per polish pass.

    Set ``PRISMAQUANT_DISABLE_RTN_COMPILE=1`` to fall back to eager —
    only useful if torch.compile fails on a particular tensor shape
    or kernel mismatch, which has not been observed but might surface
    on older torch / non-Blackwell hardware.

    The codebook device-resolution is kept *outside* the compiled
    function: dynamo cannot trace ``codebook.pin_memory()`` (NYI in
    the Inductor backend), and compiling around it would force a
    graph break on every call.  Instead the closure resolves the
    device-resident codebook eagerly, then passes it as a positional
    argument to the compiled inner.
    """
    cb_cpu = _CODEBOOKS[codebook_name]

    # Inner function takes a pre-resolved on-device codebook so the
    # compile can trace cleanly.  Functionally equivalent to
    # _rtn_fp_codebook except we skip the device-resolution call
    # (caller already did it).
    def _inner_eager(
        w: torch.Tensor, cb: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = w.shape
        in_f = w.shape[-1]
        w2 = w.reshape(-1, in_f).float()
        if group_size > 0 and group_size < in_f:
            w2 = w2.reshape(-1, in_f // group_size, group_size)
        else:
            w2 = w2.unsqueeze(1)
        cmax = cb.abs().max()
        max_abs = w2.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = max_abs / cmax
        if mx_scale:
            scale = _snap_scale_e8m0(scale)
        x = w2 / scale
        idx = torch.bucketize(x.contiguous(), cb)
        idx_lo = (idx - 1).clamp_min(0)
        idx_hi = idx.clamp_max(cb.numel() - 1)
        lo = cb[idx_lo]
        hi = cb[idx_hi]
        choose_hi = (hi - x).abs() < (x - lo).abs()
        q = torch.where(choose_hi, hi, lo)
        w_rec = q * scale
        return w_rec.reshape(orig_shape).to(w.dtype)

    import os as _os
    if _os.environ.get(
        "PRISMAQUANT_DISABLE_RTN_COMPILE", "",
    ).strip().lower() in {"1", "true", "yes", "on"}:
        _inner = _inner_eager
    else:
        # ``dynamic=True`` produces one compiled kernel that handles
        # every (batch, hidden) shape via symbolic shape specialization.
        # ``dynamic=False`` would per-shape-recompile and quickly hit
        # dynamo's default recompile_limit=8, after which every new
        # shape silently falls back to eager.  At polish time we see
        # 30+ unique Linear input shapes on a single 27B model, so the
        # symbolic-shape kernel is the right trade-off (slightly less
        # aggressive optimization in exchange for no recompile thrash).
        # Raise the recompile limit defensively in case dynamic
        # specialization still triggers a recompile path.
        try:
            torch._dynamo.config.recompile_limit = max(
                int(getattr(torch._dynamo.config, "recompile_limit", 8)),
                256,
            )
        except Exception:
            pass
        _inner = torch.compile(_inner_eager, dynamic=True)

    def f(w: torch.Tensor) -> torch.Tensor:
        cb = _codebook_on_device(
            cb_cpu, device=w.device, dtype=torch.float32,
        )
        return _inner(w, cb)

    return f


# -----------------------------------------------------------------------
# Built-in format registrations
# -----------------------------------------------------------------------
# AutoRound layer_config entries match what AutoRound expects for its
# internal QuantizationScheme.  See auto_round.compressors.utils for the
# canonical fields.  Feel free to extend as new formats are added.

def _nv_autoround(bits=4, gsize=16, act_bits=4):
    return dict(
        bits=bits, group_size=gsize, sym=True, data_type="nv_fp",
        act_bits=act_bits, act_group_size=gsize, act_sym=True,
        act_data_type="nv_fp4_with_static_gs" if bits == 4 else "nv_fp",
        act_dynamic=True,
    )


def _mx_autoround(bits=8, gsize=32, act_bits=8, elt="fp8_e4m3"):
    return dict(
        bits=bits, group_size=gsize, sym=True, data_type="mx_fp",
        act_bits=act_bits, act_group_size=gsize, act_sym=True,
        act_data_type="mx_fp", act_dynamic=True,
    )


def _int_autoround(bits, gsize, act_bits=16):
    return dict(
        bits=bits, group_size=gsize, sym=True, data_type="int",
        act_bits=act_bits, act_group_size=gsize if act_bits <= 8 else 0,
        act_sym=True, act_data_type="int" if act_bits <= 8 else "float",
        act_dynamic=True,
    )


def _plain_fp8_autoround(elt="fp8_e4m3", act_bits=8):
    # Plain per-channel FP8 (no microscaling).  AutoRound's "fp8_e4m3" /
    # "fp8_e5m2" dtypes use group_size=-1 i.e. per-tensor scale on weights,
    # and per-token dynamic scale on activations (matches vLLM's native
    # FP8 serving path and compressed-tensors FP8 scheme).
    return dict(
        bits=8, group_size=-1, sym=True, data_type=elt,
        act_bits=act_bits, act_group_size=-1, act_sym=True,
        act_data_type=elt if act_bits == 8 else "float",
        act_dynamic=True,
    )


# NVFP4 / NVFP4A16  (NVIDIA, group_size=16, FP8 scales)
register_format(FormatSpec(
    name="NVFP4",
    weight_bits=4, group_size=16, scale_bits=8, scale_dtype_name="fp8_e4m3",
    weight_element_dtype="fp4_e2m1", act_bits=4, act_dtype_name="fp4_e2m1",
    act_group_size=16, family="nv", min_capability_sm=100,
    autoround_config=lambda: _nv_autoround(4, 16, 4),
    quantize_dequantize=_make_rtn("fp4_e2m1", 16),
    activation_quantize_dequantize=_make_rtn("fp4_e2m1", 16),
))
register_format(FormatSpec(
    name="NVFP4A16",
    weight_bits=4, group_size=16, scale_bits=8, scale_dtype_name="fp8_e4m3",
    weight_element_dtype="fp4_e2m1", act_bits=None,
    family="nv", min_capability_sm=100,
    autoround_config=lambda: _nv_autoround(4, 16, 16),
    quantize_dequantize=_make_rtn("fp4_e2m1", 16),
    activation_quantize_dequantize=lambda x: x,
))

# MXFP4 / MXFP8 / MXFP6 variants  (OCP MX, group_size=32, E8M0 scales)
# All MX formats use mx_scale=True so RTN models the actual E8M0 power-of-two
# per-block scale used by the serving path. Without this the measured RTN
# error would be slightly optimistic vs what the kernel actually produces.
register_format(FormatSpec(
    name="MXFP4",
    weight_bits=4, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp4_e2m1", act_bits=4, act_dtype_name="fp4_e2m1",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(4, 32, 4, "fp4_e2m1"),
    quantize_dequantize=_make_rtn("fp4_e2m1", 32, mx_scale=True),
    activation_quantize_dequantize=_make_rtn("fp4_e2m1", 32, mx_scale=True),
))
register_format(FormatSpec(
    name="MXFP6_E3M2",
    weight_bits=6, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp6_e3m2", act_bits=6, act_dtype_name="fp6_e3m2",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(6, 32, 6, "fp6_e3m2"),
    quantize_dequantize=_make_rtn("fp6_e3m2", 32, mx_scale=True),
    activation_quantize_dequantize=_make_rtn("fp6_e3m2", 32, mx_scale=True),
))
register_format(FormatSpec(
    name="MXFP6_E2M3",
    weight_bits=6, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp6_e2m3", act_bits=6, act_dtype_name="fp6_e2m3",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(6, 32, 6, "fp6_e2m3"),
    quantize_dequantize=_make_rtn("fp6_e2m3", 32, mx_scale=True),
    activation_quantize_dequantize=_make_rtn("fp6_e2m3", 32, mx_scale=True),
))
register_format(FormatSpec(
    name="MXFP8",  # alias for MXFP8_E4M3 (OCP MX canonical default)
    weight_bits=8, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp8_e4m3", act_bits=8, act_dtype_name="fp8_e4m3",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(8, 32, 8, "fp8_e4m3"),
    quantize_dequantize=_make_rtn("fp8_e4m3", 32, mx_scale=True),
    activation_quantize_dequantize=_make_rtn("fp8_e4m3", 32, mx_scale=True),
))
register_format(FormatSpec(
    name="MXFP8_E4M3",  # explicit name for the canonical variant
    weight_bits=8, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp8_e4m3", act_bits=8, act_dtype_name="fp8_e4m3",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(8, 32, 8, "fp8_e4m3"),
    quantize_dequantize=_make_rtn("fp8_e4m3", 32, mx_scale=True),
    activation_quantize_dequantize=_make_rtn("fp8_e4m3", 32, mx_scale=True),
))
register_format(FormatSpec(
    name="MXFP8_E5M2",  # wider dynamic range, less mantissa precision
    weight_bits=8, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp8_e5m2", act_bits=8, act_dtype_name="fp8_e5m2",
    act_group_size=32, family="mx", min_capability_sm=100,
    autoround_config=lambda: _mx_autoround(8, 32, 8, "fp8_e5m2"),
    quantize_dequantize=_make_rtn("fp8_e5m2", 32, mx_scale=True),
    activation_quantize_dequantize=_make_rtn("fp8_e5m2", 32, mx_scale=True),
))
register_format(FormatSpec(
    name="MXFP8A16",
    weight_bits=8, group_size=32, scale_bits=8, scale_dtype_name="uint8_e8m0",
    weight_element_dtype="fp8_e4m3", act_bits=None,
    family="mx", min_capability_sm=80,  # W8A16 works on Marlin
    autoround_config=lambda: _mx_autoround(8, 32, 16, "fp8_e4m3"),
    quantize_dequantize=_make_rtn("fp8_e4m3", 32, mx_scale=True),
    activation_quantize_dequantize=lambda x: x,
))

# Plain FP8 (per-tensor scale on weights, no microscaling).  vLLM-native
# serving path; works on Hopper (sm_90) and Blackwell (sm_100+).
register_format(FormatSpec(
    name="FP8_E4M3",
    weight_bits=8, group_size=0, scale_bits=32, scale_dtype_name="fp32",
    weight_element_dtype="fp8_e4m3", act_bits=8, act_dtype_name="fp8_e4m3",
    act_group_size=0, family="fp", min_capability_sm=90,
    autoround_config=lambda: _plain_fp8_autoround("fp8_e4m3", 8),
    quantize_dequantize=_make_rtn("fp8_e4m3", 0),
    activation_quantize_dequantize=_make_rtn("fp8_e4m3", 0),
))
register_format(FormatSpec(
    name="FP8_E5M2",
    weight_bits=8, group_size=0, scale_bits=32, scale_dtype_name="fp32",
    weight_element_dtype="fp8_e5m2", act_bits=8, act_dtype_name="fp8_e5m2",
    act_group_size=0, family="fp", min_capability_sm=90,
    autoround_config=lambda: _plain_fp8_autoround("fp8_e5m2", 8),
    quantize_dequantize=_make_rtn("fp8_e5m2", 0),
    activation_quantize_dequantize=_make_rtn("fp8_e5m2", 0),
))

# INT8 per-channel / INT4 per-group
register_format(FormatSpec(
    name="INT8_W8A16",
    weight_bits=8, group_size=0, scale_bits=16, scale_dtype_name="fp16",
    weight_element_dtype="int8",
    family="int", min_capability_sm=70,
    autoround_config=lambda: _int_autoround(8, -1, 16),
    quantize_dequantize=lambda w: _rtn_uniform_int(w, 8, 0),
    activation_quantize_dequantize=lambda x: x,
))
register_format(FormatSpec(
    name="INT4_W4A16_g128",
    weight_bits=4, group_size=128, scale_bits=16, scale_dtype_name="fp16",
    weight_element_dtype="int4",
    family="int", min_capability_sm=70,
    autoround_config=lambda: _int_autoround(4, 128, 16),
    quantize_dequantize=lambda w: _rtn_uniform_int(w, 4, 128),
    activation_quantize_dequantize=lambda x: x,
))

# Passthrough for highest-precision layer when budget is loose
register_format(FormatSpec(
    name="BF16",
    weight_bits=16, group_size=0, scale_bits=0, scale_dtype_name="none",
    weight_element_dtype="bfloat16",
    family="fp", min_capability_sm=75,
    autoround_config=lambda: dict(bits=16, group_size=0,
                                   data_type="float", act_bits=16,
                                   act_data_type="float"),
    quantize_dequantize=lambda w: w.clone(),
    activation_quantize_dequantize=lambda x: x.clone(),
))

# Source-FP8 passthrough — for models that ship FP8-quantized (MiniMax
# M2/M2.7, DeepSeek V3, several NVIDIA releases). When the allocator
# picks this format for a Linear, the exporter copies the SOURCE FP8
# tensor (+ its weight_scale_inv block-scale tensor) verbatim into the
# output checkpoint — no dequant/requant round-trip. That preserves
# bit-exact the value the probe's BF16 view was dequanted from.
#
# quantize_dequantize is identity: given the probe's BF16 weight (which
# IS the lossless dequant of the source FP8 — every E4M3 code is
# exactly representable in bfloat16), re-quantizing to FP8_SOURCE gives
# back the SAME BF16 view. Cost is zero Δloss, as it should be.
#
# effective_bits = 8 + 32 / (128*128) ≈ 8.002 bpp (scale_inv is fp32
# at the 128×128 block granularity MiniMax ships; smaller than MXFP8's
# 8.25 because the block is 128×128 not group-of-32).
register_format(FormatSpec(
    name="FP8_SOURCE",
    weight_bits=8, group_size=128, scale_bits=32, scale_dtype_name="fp32",
    weight_element_dtype="fp8_e4m3",
    scale_block_shape=(128, 128),
    act_bits=None,
    family="fp", min_capability_sm=89,
    autoround_config=lambda: dict(bits=8, group_size=128,
                                   data_type="fp8_e4m3", sym=True,
                                   act_bits=16, act_data_type="float"),
    quantize_dequantize=lambda w: w.clone(),
    activation_quantize_dequantize=lambda x: x.clone(),
))


def list_formats(family: str | None = None) -> list[FormatSpec]:
    if family is None:
        return sorted(REGISTRY.values(), key=lambda s: s.effective_bits)
    return sorted((s for s in REGISTRY.values() if s.family == family),
                  key=lambda s: s.effective_bits)


def get_format(name: str) -> FormatSpec:
    canonical = canonical_format_name(name)
    if canonical not in REGISTRY:
        raise KeyError(f"Unknown format '{name}'. Available: "
                       f"{sorted((*REGISTRY.keys(), *FORMAT_ALIASES.keys()))}")
    spec = REGISTRY[canonical]
    if name in FORMAT_ALIASES:
        return replace(spec, name=name)
    return spec
