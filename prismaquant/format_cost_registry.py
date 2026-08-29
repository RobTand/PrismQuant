"""Reference :class:`FormatCostPlugin` implementations backed by the real registry.

These are not a parallel model of what a format does -- they call
``FormatSpec.quantize_dequantize`` and ``FormatSpec.activation_quantize_dequantize``
directly, so the error a plugin reports is produced by the same code that
renders. That is the same rendering-identity requirement that makes the
surrogate, the KL validation and the exported bytes comparable; a plugin that
re-implemented a format's rounding would reintroduce exactly the confound the
"one cache mechanism" rule exists to prevent.

Because the registry already declares ``act_bits`` and
``act_quant_changes_input`` per format, AQUA-AURA needs no new format metadata:
``NVFP4`` (W4A4) and ``NVFP4A16`` (W4A16) are already distinct entries with
identical weight bits, which is the cleanest possible demonstration that the
activation term is doing the work.

The activation error is *measured* through the format's own activation
quantizer, applied to a synthetic activation whose per-channel scale is taken
from the card, rather than assumed to be a uniform grid. Caveat, stated plainly:
real activation quantizers are often per-token or per-group, so driving them
with a per-channel synthetic sample approximates the grouping. This is a
screening surrogate, not a served measurement.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from .format_registry import FormatSpec, get_format
from .format_cost_protocol import FormatDescriptor, expert_act_sigma
from .sensitivity_card import SensitivityUnit

# Formats that copy an already-matching source tensor verbatim. The allocator
# may pick these only when the source is already that precision; synthesizing
# BF16 from a dequantized FP8 would waste 8 bpp.
PASSTHROUGH_SOURCE_DTYPE = {
    "BF16": "bfloat16",
    "FP8_SOURCE": "float8_e4m3fn",
    "FP8_BLOCK_UE8M0_SOURCE": "float8_e4m3fn",
    "MXFP4_SOURCE": "float4_e2m1fn",
}

#: The production menu. Deliberately small: these are the formats vLLM serves
#: natively today. An author targeting another platform passes their own list.
PRODUCTION_MENU = ("NVFP4", "FP8_E4M3", "BF16", "FP8_SOURCE")

#: A menu that isolates the activation axis: same weight format, different
#: activation handling. This is the AQUA-AURA A/B in menu form.
ACTIVATION_AXIS_MENU = ("NVFP4", "NVFP4A16", "FP8_E4M3", "MXFP8A16")


def descriptor_for(spec: FormatSpec, *, shape: tuple[int, int],
                   speed_index: float | None = None) -> FormatDescriptor:
    """Adapt a registry ``FormatSpec`` into the costing seam's descriptor.

    ``effective_bits_for_shape`` is used rather than ``weight_bits`` because the
    byte budget spends *stored* bits including scale/codebook overhead, and
    because codebook formats report ``weight_bits == 0``.
    """
    # act_quant_changes_input is THE predicate (format_registry defines it);
    # it is carried across as explicit data rather than re-derived downstream.
    quantizes = bool(spec.act_quant_changes_input)
    return FormatDescriptor(
        name=spec.name,
        weight_bits=float(spec.effective_bits_for_shape(shape)),
        act_bits=spec.act_bits if quantizes else None,
        quantizes_activations=quantizes,
        group_size=spec.group_size or None,
        # Carried across for the same reason as act_bits: the A-side error model
        # needs the A-side grouping. Every activation-quantizing format in the
        # shipped menu is block scaled (NVFP4 16, MX 32), so dropping this made
        # the analytic fallback wrong for 100% of the formats it could fire on.
        act_group_size=(spec.act_group_size or None) if quantizes else None,
        passthrough=spec.name in PASSTHROUGH_SOURCE_DTYPE,
        requires_source_dtype=PASSTHROUGH_SOURCE_DTYPE.get(spec.name),
        speed_index=speed_index,
    )


@dataclasses.dataclass
class RegistryFormatPlugin:
    """Price a registry format using its own quantizers.

    ``device`` matters: this is a GPU-first codebase and the RTN kernels are
    written for it. Costing a full model on CPU is a bug, not a fallback.
    """

    descriptor: FormatDescriptor
    spec: FormatSpec
    device: str = "cuda"
    act_samples: int = 256
    seed: int = 0

    @classmethod
    def build(cls, name: str, *, shape: tuple[int, int], device: str = "cuda",
              speed_index: float | None = None) -> "RegistryFormatPlugin":
        spec = get_format(name)
        return cls(descriptor=descriptor_for(spec, shape=shape,
                                             speed_index=speed_index),
                   spec=spec, device=device)

    # ------------------------------------------------------------ weight side

    def weight_error(self, unit: SensitivityUnit,
                     weight: np.ndarray) -> np.ndarray:
        """Elementwise squared weight error under the format's own RTN render.

        Passthrough formats are lossless by construction, so their error is
        exactly zero -- not "small", zero. Returning a measured epsilon there
        would let float noise decide a passthrough-vs-quantized comparison.
        """
        if self.descriptor.passthrough:
            return np.zeros((unit.out_features, unit.in_features),
                            dtype=np.float32)
        w = torch.as_tensor(np.asarray(weight), dtype=torch.bfloat16,
                            device=self.device)
        with torch.no_grad():
            q = self.spec.quantize_dequantize(w)
            err = (w.float() - q.float()) ** 2
        return err.cpu().numpy()

    def render(self, weight) -> "torch.Tensor":
        """``W_hat = QDQ(W)`` -- the tensor this format would actually install.

        Same rendering as :meth:`weight_error` computes its error against; this
        just hands back the rendered tensor instead of the squared difference,
        because the render-conditioned A-side needs ``W_hat`` itself (it is
        ``sum_j W_hat[o,j]^2 * nu[j]``, not a difference).

        RENDER BASIS. This is the registry's own RTN render -- ``FormatSpec``'s
        ``quantize_dequantize``, chunked through :meth:`_row_chunked_qdq` when
        that is proven bitwise-identical to the unchunked call and unchunked
        otherwise. It is emphatically NOT the GPTQ+JSO production render; a
        caller that wants that reads ``ProductionWeightCache``. Mixing the two
        silently would be the rendering confound the one-cache rule forbids,
        which is why the basis is a caller's explicit choice, never a fallback
        this method takes on its own.

        A passthrough format renders exactly its input (lossless by
        construction), so it returns the weight unchanged rather than a
        round-tripped copy that would carry float noise into a
        passthrough-vs-quantized comparison.

        Accepts a device tensor and keeps it on the device: the caller uploads
        a unit's weight once and prices every candidate format off it, and an
        ``np.asarray`` here would drag it back to the host per format.
        """
        w = (weight if torch.is_tensor(weight)
             else torch.as_tensor(np.asarray(weight)))
        w = w.to(device=self.device, dtype=torch.bfloat16)
        if self.descriptor.passthrough or self.spec.name == "BF16":
            return w
        qdq = self._row_chunked_qdq(w)
        with torch.no_grad():
            if qdq is None:
                # No proven-safe chunked render for this format: render whole,
                # which is the reference behaviour rather than a degradation.
                return self.spec.quantize_dequantize(w)
            rows_per_chunk = max(1, int((256 << 20) // max(1, int(w.shape[1]) * 4)))
            out = torch.empty_like(w)
            for lo in range(0, int(w.shape[0]), rows_per_chunk):
                hi = min(lo + rows_per_chunk, int(w.shape[0]))
                out[lo:hi] = qdq(w[lo:hi]).to(w.dtype)
            return out

    # ------------------------------------------------- reduced (chunked) cost

    def weight_cost_reduced(self, unit: SensitivityUnit,
                            weight: np.ndarray,
                            *, chunk_bytes: int = 256 << 20,
                            ) -> tuple[float, float] | None:
        """``(mean(dw^2), row @ dw^2 @ col)`` without ever materializing dw^2.

        WHY THIS EXISTS. Both quantities the coster needs are REDUCTIONS of the
        elementwise squared error, and :meth:`weight_error` returns that error as
        a dense ``[out, in]`` fp32 array on the HOST. For a body Linear that is a
        356 MB round trip; for a large-vocab ``lm_head`` (248320 x 5120) it is
        5.08 GB, and ``price`` then builds several more arrays of that shape in
        float64 to reduce it. On GB10, where the GPU and host share one physical
        pool, those transients are what actually exhaust the box -- and they do
        it INVISIBLY to host RSS, because CUDA device memory is not charged to
        the process. Measured: an 89M-param unit settles at 9.8 GiB reserved, so
        `lm_head` at 14x that scale has no chance.

        The fix is to keep the reduction on the device and stream it in row
        chunks. Peak device memory becomes ``w + q`` (bf16, so 2 bytes each) plus
        one chunk, instead of five dense fp32 copies. It is also markedly faster:
        the dense path was bus- and host-bound, which is why the GPU sat at low
        utilization during costing -- exactly the "launch-overhead-bound, 96% at
        11 W" signature this codebase warns about.

        CRITICALLY, ``quantize_dequantize`` still runs on the WHOLE tensor. Only
        the reduction is chunked. Formats whose scales are computed per tensor
        (rather than per group along the input axis) would otherwise see a
        different scale per chunk and silently render something the exporter
        never ships -- a rendering confound, which is the one thing the "one
        cache mechanism" rule exists to prevent.

        Accumulation is float64 for both outputs, so the only difference from the
        dense path is summation ORDER, at fp64 precision. Returns ``None`` for a
        passthrough format (exactly zero error by construction) and when the card
        carries no marginals, letting the caller keep its existing fallbacks.
        """
        if self.descriptor.passthrough:
            return (0.0, 0.0)
        if not unit.has_vectors or unit.h_trace_raw <= 0.0:
            return None

        w = torch.as_tensor(np.asarray(weight), dtype=torch.bfloat16,
                            device=self.device)
        out_features, in_features = int(w.shape[0]), int(w.shape[1])
        qdq = self._row_chunked_qdq(w)
        if qdq is None:
            return None
        row = torch.as_tensor(np.asarray(unit.fisher_row), dtype=torch.float64,
                              device=self.device)
        col = torch.as_tensor(np.asarray(unit.fisher_col), dtype=torch.float64,
                              device=self.device)

        # One chunk holds `rows x in_features` in fp32 several times over; size
        # it from a byte budget so the peak is bounded by configuration rather
        # than by whatever shape the model happens to have.
        rows_per_chunk = max(1, int(chunk_bytes // max(1, in_features * 4)))

        total_sq = torch.zeros((), dtype=torch.float64, device=self.device)
        quad = torch.zeros((), dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for lo in range(0, out_features, rows_per_chunk):
                hi = min(lo + rows_per_chunk, out_features)
                blk = w[lo:hi]
                dd = ((blk.float() - qdq(blk).float()) ** 2).double()
                total_sq += dd.sum()
                # row-block @ (dw^2 block @ col) -- the same bilinear form the
                # dense path computes, accumulated block by block.
                quad += row[lo:hi] @ (dd @ col)
                del blk, dd
            del w
        n = float(out_features) * float(in_features)
        return (float(total_sq.item()) / n, float(quad.item()))

    # ------------------------------------------------------ chunking the QDQ

    #: Formats whose render is row-separable: every scale they derive is local
    #: to a group along the INPUT axis, so quantizing a row block gives bitwise
    #: the same result as quantizing the whole tensor. Verified by test, not
    #: assumed -- ``tests/test_row_chunked_qdq.py`` asserts bitwise equality.
    ROW_SEPARABLE_FAMILIES = ("mx",)

    def _row_chunked_qdq(self, w: "torch.Tensor"):
        """A callable rendering ONE row block exactly as the full tensor would.

        This is the piece that actually bounds memory. Chunking the reduction
        alone was not enough: ``quantize_dequantize`` still ran on the whole
        tensor, and for a 248320 x 5120 ``lm_head`` its fp32 internals reserved
        **102 GiB** on a box with 121 -- measured, and the direct cause of an
        OOM that RSS never showed, since CUDA device memory is not charged to
        the process on GB10's unified pool.

        Row-chunking is NOT universally safe, which is why this is a method and
        not a one-liner. NVFP4 derives a per-TENSOR global scale, so naively
        rendering a block picks a different scale and produces different bytes
        (measured max|diff| 2.4e-2 -- not rounding noise, a different render,
        i.e. exactly the rendering confound the one-cache rule forbids). The
        export codec anticipates this and takes ``global_real_override``, so the
        scale is computed ONCE over the whole tensor and pinned for every block,
        which is bitwise identical to the unchunked render by construction.

        Returns ``None`` when this format has no proven-safe chunked render, so
        the caller falls back to the dense path rather than silently shipping a
        different rendering.
        """
        import torch

        spec = self.spec
        name = spec.name
        if self.descriptor.passthrough or name == "BF16":
            return lambda blk: blk

        # Per-row / per-group only: separable by construction.
        if name.startswith("FP8_E4M3") or spec.family in self.ROW_SEPARABLE_FAMILIES:
            return lambda blk: spec.quantize_dequantize(blk.contiguous())

        if name in ("NVFP4", "NVFP4A16"):
            from . import export_native_compressed as enc

            group = 16
            in_f = int(w.shape[1])
            if in_f % group != 0:
                return None
            # Pass 1: the tensor-global scale, accumulated block by block so the
            # full [rows, groups, 16] view never exists at once.
            rows_scan = max(1, int((256 << 20) // max(1, in_f * 4)))
            gmax = torch.zeros((), dtype=torch.float32, device=w.device)
            for lo in range(0, int(w.shape[0]), rows_scan):
                blk = w[lo:lo + rows_scan].float()
                grouped = blk.reshape(blk.shape[0], in_f // group, group)
                s = enc._select_nvfp4_group_scales(grouped)
                gmax = torch.maximum(gmax, s.amax())
                del blk, grouped, s
            global_real = (gmax / enc.FP8_E4M3_MAX).clamp_min(1e-12)

            def _qdq(blk):
                out = enc._rtn_dequant_nvfp4(
                    blk.float(), group_size=group,
                    global_real_override=global_real)
                return out.to(blk.dtype)

            return _qdq

        return None

    # -------------------------------------------------------- activation side

    def activation_error_variance(self, unit: SensitivityUnit,
                                  ) -> np.ndarray | None:
        """Per-input-channel variance of this format's activation-quant error.

        Measured by pushing a synthetic activation through the format's own
        ``activation_quantize_dequantize``. The synthetic sample is scaled per
        channel to the card's measured second moment and clipped to its measured
        absmax, so the quantizer sees the dynamic range the channel actually
        spans -- which is what drives activation error.

        Returns None when the format does not quantize activations, or when the
        card carries no activation statistics. None is not zero: an unmeasured
        activation cost must never read as a free one.
        """
        if not self.descriptor.quantizes_activations:
            return None
        if unit.act_sq_sum is None:
            return None

        sigma = np.sqrt(np.asarray(unit.act_sq_sum, dtype=np.float64)
                        / max(1, unit.n_tokens))
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        base = torch.randn(self.act_samples, unit.in_features,
                           generator=gen, dtype=torch.float32)
        x = base * torch.as_tensor(sigma, dtype=torch.float32)
        if unit.act_absmax is not None:
            cap = torch.as_tensor(np.asarray(unit.act_absmax),
                                  dtype=torch.float32)
            x = torch.clamp(x, -cap, cap)

        x = x.to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            xq = self.spec.activation_quantize_dequantize(x)
            err = (x.float() - xq.float()) ** 2
            per_channel = err.mean(dim=0)
        out = per_channel.cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(out)):
            return None
        return out

    def expert_activation_error_variance(self, unit: SensitivityUnit,
                                         ) -> np.ndarray | None:
        """Per-expert per-channel activation-quant error variance, [E, N].

        Same estimator as :meth:`activation_error_variance` -- synthetic rows
        scaled to the measured second moment, clipped to the measured absmax,
        pushed through the format's own quantizer -- but fitted per expert,
        because on a routed MoE the activation distribution IS a function of
        the expert: the router sends systematically different tokens to
        different experts, which is the entire premise of routing.

        Batched: activation quantization is per-token (NVFP4's block scale is
        set within one row), so stacking every expert's synthetic rows into
        one tensor and reducing per expert is arithmetically identical to E
        separate calls, at a fraction of the launch overhead. Chunked over
        experts so the temporary stays bounded regardless of E.
        """
        if not self.descriptor.quantizes_activations:
            return None
        sigma = expert_act_sigma(unit)
        if sigma is None:
            return None
        n_e, n_in = sigma.shape
        dev = torch.device(self.device)
        # ONE standard-normal draw, shared by every expert (common random
        # numbers), generated on the CPU with the same seed the dense path
        # uses so the estimate stays deterministic and machine-independent.
        #
        # Drawing independently per expert instead costs ~65 s of CPU randn on
        # a 40-layer 256-expert model -- more than half this stage's wall clock
        # -- to buy nothing: the experts differ by sigma and by the clip bound,
        # not by the underlying sample. Sharing the draw is also the LOWER
        # variance estimator for what this feeds, which is a comparison BETWEEN
        # experts: the common component cancels out of the difference.
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        base = torch.randn(self.act_samples, n_in,
                           generator=gen, dtype=torch.float32).to(dev)
        sig_t = torch.as_tensor(sigma, dtype=torch.float32, device=dev)
        cap = (torch.as_tensor(np.asarray(unit.expert_act_absmax),
                               dtype=torch.float32, device=dev)
               if unit.expert_act_absmax is not None else None)
        out = np.zeros((n_e, n_in), dtype=np.float64)
        # ~64 MiB of bf16 per chunk before the quantizer's own temporaries.
        per_chunk = max(1, int((32 << 20) // max(1, self.act_samples * n_in)))
        for lo in range(0, n_e, per_chunk):
            hi = min(lo + per_chunk, n_e)
            k = hi - lo
            x = base.unsqueeze(0) * sig_t[lo:hi].unsqueeze(1)   # [k, S, N]
            if cap is not None:
                c = cap[lo:hi].unsqueeze(1)
                x = torch.clamp(x, -c, c)
            x = x.reshape(k * self.act_samples, n_in).to(torch.bfloat16)
            with torch.no_grad():
                xq = self.spec.activation_quantize_dequantize(x)
                err = (x.float() - xq.float()) ** 2
                per = err.view(k, self.act_samples, n_in).mean(dim=1)
            out[lo:hi] = per.double().cpu().numpy()
            del x, xq, err, per
        if not np.all(np.isfinite(out)):
            return None
        # An expert that saw no calibration tokens has sigma == 0, so the
        # quantizer returns exact zeros and its A-side is 0.0. That is the
        # honest answer for "no evidence", and it is visible in the census
        # rather than hidden: `expert_tokens` records the zero.
        return out


def build_menu(names, *, shape: tuple[int, int], device: str = "cuda"):
    """Build plugins for a named menu. This is the whole 'arbitrary menu' story."""
    return [RegistryFormatPlugin.build(n, shape=shape, device=device)
            for n in names]
