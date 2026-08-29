"""Row-chunked costing must render EXACTLY what the full-tensor path renders.

WHY THIS TEST EXISTS
--------------------
``RegistryFormatPlugin.weight_cost_reduced`` streams a unit's cost in row blocks
so a large-vocab ``lm_head`` can be priced at all: rendering one 248320 x 5120
tensor whole reserved 102 GiB of a 121 GiB shared pool on GB10 and took the
machine down. Chunking bounds that, but only if a block renders identically to
the whole tensor -- otherwise the coster is scoring bytes the exporter will
never ship, which is the rendering confound the one-cache rule exists to
prevent.

Separability is NOT a safe assumption and must never be added to
``ROW_SEPARABLE_FAMILIES`` on inspection. NVFP4 derives a per-TENSOR global
scale, so naive row-chunking changes the render: measured max|diff| 2.4e-2, four
orders of magnitude above rounding. It is handled by pinning the global scale
computed once over the full tensor, and that is exactly what this test guards.

The poisoned row (``w[7] *= 50``) is the load-bearing part of the fixture: it
makes one block's local statistics wildly unrepresentative of the tensor's, so
any per-tensor reduction that leaked into a per-block computation shows up as a
large difference rather than as noise.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from prismaquant.format_cost_protocol import weight_dloss_marginal  # noqa: E402
from prismaquant.format_cost_registry import RegistryFormatPlugin  # noqa: E402
from prismaquant.sensitivity_card import SensitivityUnit, UnitTopology  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="costing is a GPU path by design")

# Formats the coster must be able to chunk. NVFP4/NVFP4A16 are the interesting
# ones (per-tensor global scale); the rest confirm the separable branch.
CHUNKABLE = ["NVFP4", "NVFP4A16", "FP8_E4M3", "MXFP8_E4M3", "BF16"]


def _unit(o: int, i: int, rng) -> tuple[SensitivityUnit, np.ndarray]:
    w = (rng.standard_normal((o, i)) * 0.02).astype(np.float32)
    w[7] *= 50.0  # see module docstring: this is what makes the test bite
    row = np.abs(rng.standard_normal(o)) + 0.01
    col = np.abs(rng.standard_normal(i)) + 0.01
    unit = SensitivityUnit(
        topology=UnitTopology(name="t", source_dtype="bfloat16"),
        out_features=o, in_features=i, n_params=o * i, n_tokens=4096,
        h_trace_raw=float(row.sum() * col.sum()),
        h_w2_sum_raw=0.0, w_norm_sq=0.0, w_max_abs=float(np.abs(w).max()),
        fisher_row=row, fisher_col=col)
    return unit, w


@pytest.mark.parametrize("fmt", CHUNKABLE)
def test_chunked_cost_matches_dense_reference(fmt):
    """The two reductions must agree with the dense path to fp64 round-off."""
    rng = np.random.default_rng(11)
    unit, w = _unit(2048, 5120, rng)
    plugin = RegistryFormatPlugin.build(fmt, shape=w.shape, device="cuda")

    dense = plugin.weight_error(unit, w)
    mse_ref = float(dense.astype(np.float64).mean())
    dloss_ref = weight_dloss_marginal(unit, dense, 1.0)

    # A deliberately tiny budget so the loop runs many blocks; a single-block
    # "chunked" run would pass trivially and prove nothing.
    reduced = plugin.weight_cost_reduced(unit, w, chunk_bytes=1 << 18)
    assert reduced is not None, f"{fmt} lost its chunked path"
    mse_chunked, quad = reduced
    dloss_chunked = 0.5 * (quad / unit.h_trace_raw / max(1, unit.n_tokens))

    assert mse_chunked == pytest.approx(mse_ref, rel=1e-12)
    assert dloss_chunked == pytest.approx(dloss_ref, rel=1e-12)


@pytest.mark.parametrize("fmt", CHUNKABLE)
def test_chunked_path_is_at_least_as_accurate_as_dense(fmt):
    """Chunking must not cost precision -- it buys some.

    The dense path reduces a float32 array with ``np.mean``; the chunked path
    accumulates in float64. Against a float64 reference the chunked result is
    the closer one, so this change is an accuracy improvement and a memory fix
    at once. Asserted so a future "optimization" back to float32 accumulation
    cannot land quietly.
    """
    rng = np.random.default_rng(3)
    unit, w = _unit(1024, 5120, rng)
    plugin = RegistryFormatPlugin.build(fmt, shape=w.shape, device="cuda")

    dense = plugin.weight_error(unit, w)
    truth = float(dense.astype(np.float64).mean())
    err_dense = abs(float(np.mean(dense)) - truth)
    reduced = plugin.weight_cost_reduced(unit, w, chunk_bytes=1 << 18)
    assert reduced is not None
    err_chunked = abs(reduced[0] - truth)
    assert err_chunked <= err_dense + 1e-30


def test_nvfp4_is_not_naively_row_separable():
    """Pin the reason NVFP4 needs the pinned-global-scale treatment.

    If this ever starts passing trivially -- i.e. NVFP4 becomes separable -- the
    special case in ``_row_chunked_qdq`` can be simplified. Until then this
    documents, executably, why it cannot just call ``quantize_dequantize`` on a
    block like the separable formats do.
    """
    from prismaquant import format_registry as fr

    spec = fr.get_format("NVFP4")
    w = (torch.randn(2048, 5120, device="cuda", dtype=torch.bfloat16) * 0.02)
    w[7] *= 50.0
    with torch.no_grad():
        full = spec.quantize_dequantize(w)
        blocks = torch.cat(
            [spec.quantize_dequantize(w[lo:lo + 256].contiguous())
             for lo in range(0, w.shape[0], 256)], dim=0)
    assert not torch.equal(full, blocks), (
        "NVFP4 appears row-separable now; revisit _row_chunked_qdq")


def test_unknown_format_falls_back_rather_than_guessing():
    """No chunked path must mean None, never a differently-rendered answer."""
    rng = np.random.default_rng(5)
    unit, w = _unit(256, 512, rng)
    plugin = RegistryFormatPlugin.build("NVFP4", shape=w.shape, device="cuda")
    # A format the chunker does not know about must decline, so `price` uses the
    # dense reference path instead of silently rendering something else. Build
    # a private descriptor: RegistryFormatPlugin.build returns the registry's
    # shared FormatSpec object, so mutating it would poison every later test in
    # a GPU-enabled full-suite process.
    plugin.spec = replace(
        plugin.spec, name="TOTALLY_MADE_UP_FORMAT", family="nonesuch",
    )
    assert plugin._row_chunked_qdq(
        torch.zeros(4, 4, device="cuda", dtype=torch.bfloat16)) is None
