"""NVFP4-CB / FP8-CB codebook format tests (Milestone A emulation +
Milestone B byte packers / exporter)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from prismaquant import format_registry as fr
from prismaquant import layer_config as lc
from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_layout import (
    ACCEPTED_CB_FORMAT_NAMES,
    FP8_ACCEPTED_RUNGS,
    FP8_PRODUCT_RUNGS,
    PRODUCT_CB_FORMAT_NAMES,
    bit_split,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_serialization_context_stamp,
)

_NVFP4_KS = list(range(1, 26))
_FP8_KS = list(FP8_PRODUCT_RUNGS)
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# This module builds synthetic CB bodies on CPU and never serves them.
# Gridbook 0.9.1's v12 table names no CB cell on sm_121, so the route gate
# refuses these exports unless the artifact declares what it is.  See
# tests/cb_synthetic_target.py; the real sm_121 refusal stays asserted in
# tests/test_cb_route_status_gate.py.
pytestmark = pytest.mark.usefixtures("synthetic_cb_target")



def _production_cb_stamp(formats):
    return cb_serialization_context_stamp(
        CBSerializationContext.production(), formats=formats
    )


def _wmse(w, r, cw=None):
    e = (w - r).float().pow(2)
    if cw is not None:
        e = e * cw
    return float(e.mean())


# (a) effective-bits accounting, exact, for every rung.
@pytest.mark.parametrize("k", _NVFP4_KS)
def test_nvfp4_cb_effective_bits_exact(k):
    spec = fr.get_format(f"NVFP4_CB_K{k}")
    assert spec.effective_bits == pytest.approx(k / 8 + 0.5, abs=1e-9)
    assert spec.effective_bits_for_shape((64, 2048)) == pytest.approx(
        k / 8 + 0.5, abs=1e-9)
    assert spec.memory_bytes_for_shape((64, 2048)) == 64 * (2048 // 256) * (
        4 * k + 16)


@pytest.mark.parametrize("k", _FP8_KS)
def test_fp8_cb_effective_bits_exact(k):
    spec = fr.get_format(f"FP8_CB_K{k}")
    # Registry body = index stream only, k/8 bpw exact (no group scale plane).
    # The per-output-channel fp32 scale is the authoritative footprint's
    # concern (nvfp4_cb_footprint), not the single-scale FormatSpec.
    assert spec.effective_bits == pytest.approx(k / 8, abs=1e-9)
    assert spec.effective_bits_for_shape((64, 2048)) == pytest.approx(
        k / 8, abs=1e-9)
    assert spec.memory_bytes_for_shape((128, 256)) == 128 * (256 // 256) * (
        4 * k)


# (b) decode validity: every reconstructed value == a grid point * group scale.
@pytest.mark.parametrize("mode", ["full", "product"])
def test_decode_on_grid_times_scale(mode):
    torch.manual_seed(0)
    w = torch.randn(64, 512)
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode=mode)
    recon = cb.nvfp4_cb_reconstruct(fields, 12, grid="fp4", mode=mode)
    pes = cb._per_element_scale(fields["scales"], "fp4", 512)
    q = recon / pes
    grid = cb._e2m1_grid("cpu")
    dist = (q.unsqueeze(-1) - grid).abs().min(dim=-1).values
    assert float(dist.max()) < 1e-5


# (c) determinism: bit-identical, eager, per device.
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mode", ["full", "product"])
def test_determinism_per_device(device, mode):
    torch.manual_seed(3)
    w = torch.randn(48, 512, device=device)
    qdq = cb.make_nvfp4_cb_qdq(12, "fp4", mode)
    a, b = qdq(w), qdq(w)
    assert torch.equal(a, b)


# (d) col_weights changes the assignment and reduces weighted MSE.
def test_col_weights_reduces_weighted_mse():
    torch.manual_seed(1)
    w = torch.randn(64, 512)
    cw = torch.rand(512) + 0.05
    f0 = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="full")
    fw = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="full", col_weights=cw)
    assert not torch.equal(f0["indices"], fw["indices"])
    r0 = cb.nvfp4_cb_reconstruct(f0, 12, grid="fp4", mode="full")
    rw = cb.nvfp4_cb_reconstruct(fw, 12, grid="fp4", mode="full")
    assert _wmse(w, rw, cw) <= _wmse(w, r0, cw) + 1e-9


# (e) learned codebook (k=12, full) beats-or-ties the fixed lattice.
def test_learned_codebook_beats_fixed():
    torch.manual_seed(2)
    w = torch.randn(96, 512)
    cw = torch.rand(512) + 0.05
    vecs, _, _ = cb._scale_and_vectorize(w, "fp4")
    learned = cb.learn_codebook(vecs, 12, grid="fp4", iters=8)
    f_fix = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="full", col_weights=cw)
    f_lrn = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="full",
                              col_weights=cw, codebook=learned)
    r_fix = cb.nvfp4_cb_reconstruct(f_fix, 12, grid="fp4", mode="full")
    r_lrn = cb.nvfp4_cb_reconstruct(f_lrn, 12, grid="fp4", mode="full",
                                    codebook=learned)
    assert _wmse(w, r_lrn, cw) <= _wmse(w, r_fix, cw) + 1e-9
    # learned codebook is grid-valued (E2M1) so a decoded tile stays NVFP4.
    grid = cb._e2m1_grid("cpu")
    dist = (learned.unsqueeze(-1) - grid).abs().min(dim=-1).values
    assert float(dist.max()) < 1e-5


# scale sweep: joint per-group scale search (IQ-rendering parity).
def _sweep_total_err(w, cw, grid, mode, k, scale):
    cw2d = torch.broadcast_to(cw, w.shape).contiguous()
    wq = cb._col_weight_vectors(cw2d)
    C = cb._resolve_codebook(k, grid, mode, None, w.device)
    err, _, _ = cb._eval_candidate(w, wq, scale, grid, mode, C)
    return float(err.sum())


@pytest.mark.parametrize("grid,mode,k", [
    ("fp4", "full", 13), ("fp4", "product", 14),
    ("fp8", "product", 40),
])
def test_scale_sweep_never_worse_than_one_shot(grid, mode, k):
    torch.manual_seed(0)
    w = torch.randn(64, 512) * 0.3
    cw = torch.rand(512) + 0.05
    C = cb._resolve_codebook(k, grid, mode, None, w.device)
    amax = cb._group_amax(w, grid)
    cands = cb._candidate_scales(amax, grid, cb._SCALE_SWEEP_CANDIDATES)
    # candidate 0 is the amax/grid-max one-shot; it is IN the sweep set.
    one_shot = _sweep_total_err(w, cw, grid, mode, k, cands[0])
    fields = cb.nvfp4_cb_fields(w, k, grid=grid, mode=mode, col_weights=cw,
                                scale_sweep=True)
    swept = _sweep_total_err(w, cw, grid, mode, k, fields["scales"])
    assert swept <= one_shot + 1e-4


@pytest.mark.parametrize("mode,k", [
    ("full", 13), ("product", 14)])
def test_scale_sweep_fp4_scales_are_e4m3_legal(mode, k):
    torch.manual_seed(1)
    w = torch.randn(48, 512) * 0.3
    cw = torch.rand(512) + 0.05
    fields = cb.nvfp4_cb_fields(w, k, grid="fp4", mode=mode, col_weights=cw,
                                scale_sweep=True)
    s = fields["scales"]
    assert torch.equal(s, s.to(torch.float8_e4m3fn).to(torch.float32))
    assert bool((s > 0).all())


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("mode", ["full", "product"])
def test_scale_sweep_determinism(device, mode):
    torch.manual_seed(2)
    w = (torch.randn(48, 512, device=device) * 0.3)
    qdq = cb.make_nvfp4_cb_qdq(14, "fp4", mode, scale_sweep=True)
    assert torch.equal(qdq(w), qdq(w))


def test_scale_sweep_toggle_changes_output_and_default_on():
    torch.manual_seed(3)
    w = torch.randn(64, 512) * 0.3
    swept = cb.make_nvfp4_cb_qdq(14, "fp4", "product", scale_sweep=True)(w)
    one_shot = cb.make_nvfp4_cb_qdq(14, "fp4", "product", scale_sweep=False)(w)
    default = cb.make_nvfp4_cb_qdq(14, "fp4", "product")(w)
    assert torch.equal(default, swept)          # default is scale_sweep=True
    assert not torch.equal(swept, one_shot)     # the sweep actually moved


def test_scale_sweep_decode_validity_holds():
    # swept scales are still one E4M3 value per group-16, so decode == grid*scale
    torch.manual_seed(4)
    w = torch.randn(64, 512) * 0.3
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product",
                                scale_sweep=True)
    recon = cb.nvfp4_cb_reconstruct(fields, 12, grid="fp4", mode="product")
    pes = cb._per_element_scale(fields["scales"], "fp4", 512)
    grid = cb._e2m1_grid("cpu")
    dist = ((recon / pes).unsqueeze(-1) - grid).abs().min(dim=-1).values
    assert float(dist.max()) < 1e-5


def test_scale_sweep_does_not_change_effective_bits():
    for k in (12, 14):
        spec = fr.get_format(f"NVFP4_CB_K{k}")
        assert spec.effective_bits == pytest.approx(k / 8 + 0.5, abs=1e-9)


# FP8_CB: every registered rung is functional through the qdq closure —
# product mode splits into four 2-dim sub-vectors (9..12-bit sub-tables).
@pytest.mark.parametrize("k", _FP8_KS)
def test_fp8_cb_qdq_roundtrip_valid(k):
    torch.manual_seed(6)
    w = torch.randn(32, 512) * 0.3
    qdq = cb.make_nvfp4_cb_qdq(k, "fp8", "product")
    a, b = qdq(w), qdq(w)
    assert torch.equal(a, b)
    fields = cb.nvfp4_cb_fields(w, k, grid="fp8", mode="product")
    assert fields["indices"].shape[-1] == 4
    for table in fields["codebook"]:
        assert torch.equal(cb._snap_to_grid(table, "fp8"), table)
        assert table.shape == (1 << (k // 4), 2)
    recon = cb.nvfp4_cb_reconstruct(fields, k, grid="fp8", mode="product")
    # decode validity: recon / per-row scale recovers an E4M3 grid value
    # (up to the 1-ulp fp32 (c*s)/s roundtrip).
    pes = cb._per_element_scale(fields["scales"], "fp8", 512)
    q = recon / pes
    snap = cb._snap_to_grid(q, "fp8")
    rel = (q - snap).abs() / snap.abs().clamp_min(1e-12)
    assert float(rel.max()) < 1e-6


def test_product_n_sub4_determinism_pin():
    torch.manual_seed(7)
    w = torch.randn(24, 256) * 0.5
    f1 = cb.nvfp4_cb_fields(w, 40, grid="fp8", mode="product")
    f2 = cb.nvfp4_cb_fields(w, 40, grid="fp8", mode="product")
    assert torch.equal(f1["indices"], f2["indices"])
    assert torch.equal(f1["scales"], f2["scales"])


def test_bit_split_even_and_ceil_first():
    assert bit_split(13, 2) == (7, 6)
    assert bit_split(12, 2) == (6, 6)
    assert bit_split(36, 4) == (9, 9, 9, 9)
    assert bit_split(48, 4) == (12, 12, 12, 12)


# Lloyd at scale: the old dense one-hot path materialized (m, K) fp32 —
# 2M x 4096 = 32 GB — and would OOM here; index_add accumulation must not.
def test_lloyd_scale_no_dense_onehot():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device="cpu").manual_seed(11)
    vectors = torch.randn(2_000_000, 8, generator=gen).to(device)
    learned = cb.learn_codebook(vectors, 12, grid="fp4", iters=1)
    assert learned.shape == (4096, 8)
    grid = cb._e2m1_grid(device)
    dist = (learned.unsqueeze(-1) - grid).abs().min(dim=-1).values
    assert float(dist.max()) < 1e-5


# (f) product and full both reconstruct valid values at k=12.
def test_product_and_full_valid():
    torch.manual_seed(4)
    w = torch.randn(32, 768)
    for mode in ("full", "product"):
        r = cb.make_nvfp4_cb_qdq(12, "fp4", mode)(w)
        assert r.shape == w.shape
        assert torch.isfinite(r).all()


# (g) 3-D stacked experts round-trip with per-expert col_weights.
def test_stacked_experts_roundtrip():
    torch.manual_seed(5)
    w = torch.randn(3, 64, 256)
    cw = torch.rand(3, 1, 256) + 0.05
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product",
                                col_weights=cw)
    recon = cb.nvfp4_cb_reconstruct(fields, 12, grid="fp4", mode="product")
    assert recon.shape == w.shape
    assert fields["indices"].shape == (3 * 64, 256 // cb.VEC_DIM, 2)
    # each expert uses its own scale plane -> per-expert reconstruction differs.
    assert not torch.equal(recon[0], recon[1])


# (h) in_features % 256 != 0 raises.
def test_superblock_constraint():
    with pytest.raises(ValueError, match="multiple of 256"):
        cb.nvfp4_cb_fields(torch.randn(8, 300), 12)


def test_flat_k_ceiling_raises():
    with pytest.raises(ValueError, match="infeasible"):
        cb.fixed_lattice(15, "fp4", 8)


# (i) menu: all rungs register, resolve, sort by effective_bits.
#
# Registry/layer-config parsing is the full READER domain: it retains every
# historical K28..K48 wire id and adds the low K%4 rungs. New producer menus
# are narrower and are pinned separately below.
_MENU_LADDERS = (
    ("NVFP4_CB_K", tuple(range(1, 26)), lambda k: k / 8 + 0.5),
    ("FP8_CB_K", FP8_ACCEPTED_RUNGS, lambda k: k / 8),
)



def test_menu_registers_and_resolves():
    names = [f"{prefix}{k}" for prefix, ks, _ in _MENU_LADDERS for k in ks]
    for name in names:
        spec = fr.get_format(name)
        assert spec is not None
        assert lc.canonicalize_format(name.lower()) == name
    # dict-form canonicalization (custom quant-config JSON shape).
    assert lc.canonicalize_format(
        {"data_type": "nvfp4_cb", "cb_k": 20}) == "NVFP4_CB_K20"
    assert lc.canonicalize_format(
        {"data_type": "fp8_cb", "cb_k": 44}) == "FP8_CB_K44"
    # Registry and layer-config parsing are compatibility readers. The
    # explicit producer API is the narrower artifact-writing surface.
    fam = [s for s in fr.list_formats() if s.family in ("nvfp4_cb", "fp8_cb")]
    assert {s.name for s in fam} == ACCEPTED_CB_FORMAT_NAMES == set(names)
    assert set(lc._NVFP4_CB_FORMAT_NAMES) == ACCEPTED_CB_FORMAT_NAMES
    assert {
        s.name for s in fr.list_producer_formats()
        if s.family in ("nvfp4_cb", "fp8_cb")
    } == PRODUCT_CB_FORMAT_NAMES
    # Per family: bpw strictly increasing in k, and exactly the accounting
    # formula (index stream + the fp4 family's group-16 scale plane).
    for prefix, ks, bpw in _MENU_LADDERS:
        got = [fr.get_format(f"{prefix}{k}").effective_bits for k in ks]
        assert got == sorted(got) and len(set(got)) == len(got)
        for k, g in zip(ks, got):
            assert g == pytest.approx(bpw(k), abs=1e-9), f"{prefix}{k}"
    # list_formats() orders the whole menu by effective_bits.
    bpps = [s.effective_bits for s in fam]
    assert bpps == sorted(bpps)


# ===========================================================================
# Milestone B — byte packers (format-pipeline.md §1 / LAYOUT.md contract).
# ===========================================================================

# (grid, mode, k): full/product × both grids, the required matrix.
_PACK_CASES = [
    ("fp4", "product", 12), ("fp4", "product", 14), ("fp4", "product", 16),
    ("fp4", "full", 12), ("fp4", "full", 14), ("fp4", "full", 16),
    ("fp8", "product", 36), ("fp8", "product", 44),
]


def _codebook_for(w, grid, mode, k):
    """Fixed lattice by default; an explicit grid-valued table where the flat
    lattice is infeasible (full mode, k>MAX_FLAT_K) so k=16-full is still
    covered. The pack/unpack round-trip is codebook-quality-agnostic, so a
    cheap snapped random table suffices (no need for a full 2^16 Lloyd)."""
    if mode == "full" and k > cb.MAX_FLAT_K:
        g = torch.Generator(device=w.device).manual_seed(k)
        raw = torch.randn(1 << k, cb.VEC_DIM, generator=g, device=w.device)
        return cb._snap_to_grid(raw, grid)
    return None


@pytest.mark.parametrize("grid,mode,k", _PACK_CASES)
def test_nvfp4_cb_type_size_and_packed_shape(grid, mode, k):
    ts = cb.nvfp4_cb_type_size(k, grid)
    assert ts == 4 * k + (16 if grid == "fp4" else 0)
    w = torch.randn(48, 512) * 0.3
    C = _codebook_for(w, grid, mode, k)
    fields = cb.nvfp4_cb_fields(w, k, grid=grid, mode=mode, codebook=C)
    packed = cb.nvfp4_cb_assemble_bytes(fields, k, grid, mode)
    assert packed.dtype == torch.uint8
    assert packed.ndim == 2                      # (rows, bytes) — never flat
    assert packed.shape == (48, (512 // 256) * ts)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("grid,mode,k", _PACK_CASES)
def test_nvfp4_cb_pack_unpack_matches_emulation(device, grid, mode, k):
    """THE contract: reconstruct(unpack(assemble(fields))) is BIT-IDENTICAL to
    the emulation qdq output the cost measurement scored — with scale_sweep on,
    on CPU and CUDA, for every mode×grid×k."""
    torch.manual_seed(0)
    w = torch.randn(48, 512, device=device) * 0.3
    cw = torch.rand(512, device=device) + 0.05
    C = _codebook_for(w, grid, mode, k)
    fields = cb.nvfp4_cb_fields(w, k, grid=grid, mode=mode, col_weights=cw,
                                codebook=C, scale_sweep=True)
    packed = cb.nvfp4_cb_assemble_bytes(fields, k, grid, mode)
    assert packed.device == w.device
    scales = fields["scales"] if grid == "fp8" else None
    up = cb.nvfp4_cb_unpack(packed, k, grid, mode, tuple(w.shape),
                            codebook=fields["codebook"], scales=scales)
    rec = cb.nvfp4_cb_reconstruct(up, k, grid=grid, mode=mode).to(w.dtype)
    emu = cb.nvfp4_cb_reconstruct(fields, k, grid=grid, mode=mode).to(w.dtype)
    assert torch.equal(rec, emu)


@pytest.mark.parametrize("k", [1, 32])
def test_nvfp4_endpoint_bitstream_roundtrip_without_quantizer_cost(k):
    """Pin K1 and the research-only uint32 endpoint in the direct codec."""

    bits = cb.subtable_bit_widths(k, "product", 2)
    indices = torch.zeros(1, 32, 2, dtype=torch.int64)
    indices[..., 0] = torch.arange(32).reshape(1, 32) & ((1 << bits[0]) - 1)
    if bits[1]:
        indices[..., 1] = (
            torch.arange(31, -1, -1).reshape(1, 32)
            & ((1 << bits[1]) - 1)
        )
    fields = {
        "indices": indices,
        "scales": torch.ones(1, 16),
        "shape": (1, 256),
    }
    packed = cb.nvfp4_cb_assemble_bytes(
        fields, k, grid="fp4", mode="product"
    )
    unpacked = cb.nvfp4_cb_unpack(
        packed, k, "fp4", "product", (1, 256)
    )
    assert torch.equal(unpacked["indices"], indices)
    if k == 1:
        assert torch.count_nonzero(unpacked["indices"][..., 1]) == 0
    else:
        codes = cb._vector_codes(fields, k, "fp4", "product")
        codes[0, 0] = (1 << 32) - 1
        raw = cb._pack_codes_to_bytes(codes, k)
        assert torch.equal(cb._unpack_bytes_to_codes(raw, k), codes)


def test_nvfp4_cb_assemble_asserts_type_size():
    # A tampered fields dict whose scale plane is the wrong width must trip the
    # type_size assert rather than silently emit off-layout bytes.
    w = torch.randn(16, 512) * 0.3
    fields = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="product")
    fields["scales"] = fields["scales"][:, :-1]      # drop one group scale
    with pytest.raises(Exception):
        cb.nvfp4_cb_assemble_bytes(fields, 14, "fp4", "product")


def test_nvfp4_cb_unpack_fp8_requires_scales():
    w = torch.randn(16, 512) * 0.3
    fields = cb.nvfp4_cb_fields(w, 40, grid="fp8", mode="product")
    packed = cb.nvfp4_cb_assemble_bytes(fields, 40, "fp8", "product")
    with pytest.raises(ValueError, match="no on-disk scale plane"):
        cb.nvfp4_cb_unpack(packed, 40, "fp8", "product", tuple(w.shape),
                           codebook=fields["codebook"])


def test_nvfp4_cb_pack_stacked_experts():
    torch.manual_seed(1)
    w = torch.randn(3, 32, 256) * 0.3
    cw = torch.rand(3, 1, 256) + 0.05
    fields = cb.nvfp4_cb_fields(w, 12, grid="fp4", mode="product",
                                col_weights=cw)
    packed = cb.nvfp4_cb_assemble_bytes(fields, 12, "fp4", "product")
    assert packed.shape == (3 * 32, (256 // 256) * cb.nvfp4_cb_type_size(
        12, "fp4"))
    up = cb.nvfp4_cb_unpack(packed, 12, "fp4", "product", tuple(w.shape),
                            codebook=fields["codebook"])
    rec = cb.nvfp4_cb_reconstruct(up, 12, grid="fp4", mode="product").to(
        w.dtype)
    emu = cb.nvfp4_cb_reconstruct(fields, 12, grid="fp4",
                                  mode="product").to(w.dtype)
    assert torch.equal(rec, emu)


# ===========================================================================
# Milestone B — exporter (prismaquant.export_nvfp4_cb). CPU-only for
# bit-exactness (learned Lloyd / VQ argmin ties are device-dependent).
# ===========================================================================

@pytest.fixture
def export_dir(tmp_path: Path):
    """Use pytest's per-run root; CI must not depend on a developer home."""
    return tmp_path


def _tiny_model(mdl: Path, in_f: int = 256):
    """2-layer synthetic HF dir: two 256-in Linears + a norm sidecar."""
    from safetensors.torch import save_file

    mdl.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    tens = {
        "model.layers.0.mlp.gate_proj.weight":
            (torch.randn(128, in_f) * 0.3).to(torch.bfloat16),
        "model.layers.1.mlp.gate_proj.weight":
            (torch.randn(128, in_f) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(in_f, dtype=torch.bfloat16),
    }
    save_file(tens, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(
        json.dumps({"architectures": ["Tiny"], "hidden_size": in_f}))
    return tens


def _write_assignment(path: Path, mapping: dict):
    path.write_text(json.dumps(mapping))


@pytest.mark.parametrize("source", ["lattice", "learned"])
def test_exporter_roundtrip_equals_emulation(export_dir, source):
    from safetensors.torch import load_file

    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl, out = export_dir / "model", export_dir / "out"
    tens = _tiny_model(mdl)
    assign = {
        "model.layers.0.mlp.gate_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
        "model.layers.1.mlp.gate_proj": {"data_type": "fp8_cb", "cb_k": 40},
    }
    apath = export_dir / "assign.json"
    _write_assignment(apath, assign)
    cw = {q: torch.rand(256) + 0.05 for q in assign}
    spec = ({"source": "learned", "train": True}
            if source == "learned" else {"source": "lattice"})

    counts = export_nvfp4_cb(mdl, apath, out, cw,
                             shared_codebook_spec=spec, device="cpu",
                             allow_unstamped_research=True)
    assert counts["NVFP4_CB_K16"] == 1 and counts["FP8_CB_K40"] == 1

    qc = json.loads((out / "quant_config.json").read_text())
    # "gridbook" is the registry key the plugin registers under (32a02e7);
    # "prismaquant" survives only as a READ alias for artifacts exported
    # before the rename (the external Gridbook runtime's legacy alias and
    # config.py override_quantization_method), so the exporter stamps the
    # canonical name and never the legacy one.
    assert qc["quant_method"] == "gridbook"
    assert qc["provenance"]["render_identity_verified"] is False
    assert qc["layout_version"] == 2
    assert qc["provenance"]["serialized_payload"]["context"][
        "scale_coding"
    ] == "two_tier"
    # non-target norm copied verbatim; config.json copied + pointer injected.
    ot = load_file(str(out / "model.safetensors"))
    assert "model.norm.weight" in ot
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["quantization_config"]["quant_method"] == "gridbook"

    # Codebooks ship in the non-globbed .pqcb sidecar (sidecar-only), not in
    # model.safetensors — the plugin reads them there (LAYOUT.md §3).
    cbf = load_file(str(out / qc["codebook_file"]))
    for g in qc["config_groups"].values():
        s = g["scheme"]
        grid, mode, k = s["grid"], s["mode"], s["k"]
        coding = (cb.SCALE_CODING_TWO_TIER
                  if "scale_coding" in s else cb.SCALE_CODING_V1)
        ref = s["codebook_ref"]
        codebook = (tuple(cbf[r].float() for r in ref)
                    if isinstance(ref, list) else cbf[ref].float())
        for q in g["targets"]:
            w = tens[q + ".weight"].float()
            packed = ot[q + ".cb_qweight"]
            assert packed.dtype == torch.uint8
            scales = ot.get(q + ".weight_scale")
            if grid == "fp8":
                assert scales is not None and scales.numel() == 128
                scales = scales.reshape(-1, 1)
            else:
                assert (q + ".weight_scale") not in ot   # fp4: scales in bytes
            up = cb.nvfp4_cb_unpack(packed, k, grid, mode, tuple(w.shape),
                                    codebook=codebook, scales=scales,
                                    scale_coding=coding)
            rec = cb.nvfp4_cb_reconstruct(up, k, grid=grid, mode=mode).float()
            emu_f = cb.nvfp4_cb_fields(w, k, grid=grid, mode=mode,
                                       col_weights=cw[q], codebook=codebook,
                                       scale_coding=coding)
            emu = cb.nvfp4_cb_reconstruct(emu_f, k, grid=grid,
                                          mode=mode).float()
            assert torch.equal(rec, emu)


def test_exporter_rejects_unstamped_cb_recipe_by_default(export_dir):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl = export_dir / "model"
    _tiny_model(mdl)
    qname = "model.layers.0.mlp.gate_proj"
    apath = export_dir / "assign.json"
    _write_assignment(apath, {qname: "NVFP4_CB_K16"})
    with pytest.raises(ValueError, match="value-bearing render identity"):
        export_nvfp4_cb(
            mdl,
            apath,
            export_dir / "out",
            {qname: torch.ones(256)},
            device="cpu",
        )


def test_exporter_rejects_unknown_format(export_dir):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl = export_dir / "model"
    _tiny_model(mdl)
    apath = export_dir / "assign.json"
    _write_assignment(apath, {
        "model.layers.0.mlp.gate_proj": "MXFP4",
    })
    cw = {"model.layers.0.mlp.gate_proj": torch.rand(256) + 0.05}
    with pytest.raises(ValueError, match="cannot carry"):
        export_nvfp4_cb(mdl, apath, export_dir / "out", cw, device="cpu")


def test_exporter_rejects_non_multiple_of_256(export_dir):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl = export_dir / "model"
    _tiny_model(mdl, in_f=300)          # 300 % 256 != 0
    apath = export_dir / "assign.json"
    _write_assignment(apath, {
        "model.layers.0.mlp.gate_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    })
    cw = {"model.layers.0.mlp.gate_proj": torch.rand(300) + 0.05}
    with pytest.raises(ValueError, match="multiple of 256"):
        export_nvfp4_cb(mdl, apath, export_dir / "out", cw, device="cpu")


def test_exporter_rejects_missing_col_weights(export_dir):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl = export_dir / "model"
    _tiny_model(mdl)
    apath = export_dir / "assign.json"
    _write_assignment(apath, {
        "model.layers.0.mlp.gate_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    })
    with pytest.raises(ValueError, match="no col_weights"):
        export_nvfp4_cb(mdl, apath, export_dir / "out", {}, device="cpu")


def test_exporter_rejects_missing_learned_sidecar(export_dir):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl = export_dir / "model"
    _tiny_model(mdl)
    apath = export_dir / "assign.json"
    _write_assignment(apath, {
        "model.layers.0.mlp.gate_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    })
    cw = {"model.layers.0.mlp.gate_proj": torch.rand(256) + 0.05}
    # learned source, no training, no supplied codebooks -> missing sidecar.
    with pytest.raises(ValueError, match="missing learned sidecar"):
        export_nvfp4_cb(mdl, apath, export_dir / "out", cw,
                        shared_codebook_spec={"source": "learned",
                                              "codebooks": {}}, device="cpu")


def test_export_native_compressed_hard_fails_on_cb():
    """A CB assignment reaching the compressed-tensors exporter must raise,
    not silently coerce to BF16 (mirrors the GGUF wrong-container guard)."""
    from prismaquant.export_native_compressed import (
        _coerce_runtime_legal_assignment,
    )
    with pytest.raises(ValueError, match="nvfp4_cb container"):
        _coerce_runtime_legal_assignment(
            "unused-model",
            {"model.layers.0.mlp.gate_proj": "NVFP4_CB_K16"},
        )


# ===========================================================================
# Layout v2 — two-tier scale coding (docs/lanes/nvfp4-cb/two-tier-scale-spec.md).
# ===========================================================================

def _real_magnitude_w(rows=64, in_f=512, seed=0):
    """0.6B-magnitude weights: group scales land in e4m3's subnormal band
    (the regime where the v1 candidate sweep collapses)."""
    torch.manual_seed(seed)
    return torch.randn(rows, in_f) * 0.02


# T1 — compose exactness, exhaustive over all (E, c) pairs.
def test_two_tier_compose_exact_exhaustive():
    table, compose, legal = cb._two_tier_tables("cpu")
    assert table.shape == (16,) and compose.shape == (256, 16)
    assert torch.equal(table.to(torch.float8_e4m3fn).to(torch.float32), table)
    lv = compose[legal]
    assert int(legal.sum()) > 0
    # every legal pair round-trips e4m3 bit-exactly and lies in (0, 448].
    assert torch.equal(lv.to(torch.float8_e4m3fn).to(torch.float32), lv)
    assert bool((lv > 0).all()) and bool((lv <= 448.0).all())
    # every ILLEGAL finite positive pair fails the round-trip or the range.
    ill = ~legal & torch.isfinite(compose) & (compose > 0)
    iv = compose[ill]
    rt = iv.to(torch.float8_e4m3fn).to(torch.float32)
    assert bool(((rt != iv) | (iv > 448.0)).all())
    # union of legal compositions covers every positive e4m3 value (spec §1.2)
    e4m3_pos = sorted({float(torch.tensor(b, dtype=torch.uint8).view(
        torch.float8_e4m3fn).to(torch.float32)) for b in range(256)
        if 0 < float(torch.tensor(b, dtype=torch.uint8).view(
            torch.float8_e4m3fn).to(torch.float32)) <= 448.0})
    reachable = set(lv.tolist())
    assert set(e4m3_pos) <= reachable


# T1b — encoder fuzz: emitted (super, sub) pairs are always legal and the
# stored plane equals the composition.
@pytest.mark.parametrize("mode", ["full", "product"])
def test_two_tier_encoder_emits_only_legal_pairs(mode):
    k = 13 if mode == "full" else 14
    w = _real_magnitude_w(seed=1)
    fields = cb.nvfp4_cb_fields(w, k, grid="fp4", mode=mode,
                                scale_coding="two_tier")
    _, compose, legal = cb._two_tier_tables("cpu")
    e = fields["scale_super"].to(torch.int64)
    c = fields["scale_sub"]
    e_g = e.unsqueeze(-1).expand(*e.shape, 16).reshape(e.shape[0], -1)
    assert bool(legal[e_g, c].all())
    assert torch.equal(compose[e_g, c], fields["scales"])
    s = fields["scales"]
    assert torch.equal(s.to(torch.float8_e4m3fn).to(torch.float32), s)


# T2 — pack -> unpack -> reconstruct == emulation, bit-exact, all modes.
@pytest.mark.parametrize("mode", ["full", "product"])
def test_two_tier_pack_unpack_matches_emulation(mode):
    k = 13 if mode == "full" else 14
    w = _real_magnitude_w(seed=2)
    cw = torch.rand(512) + 0.05
    packed, fields = cb.nvfp4_cb_pack(w, k, grid="fp4", mode=mode,
                                      col_weights=cw,
                                      scale_coding="two_tier")
    up = cb.nvfp4_cb_unpack(packed, k, "fp4", mode, tuple(w.shape),
                            codebook=fields["codebook"],
                            scale_coding="two_tier")
    rec = cb.nvfp4_cb_reconstruct(up, k, grid="fp4", mode=mode)
    emu = cb.nvfp4_cb_reconstruct(fields, k, grid="fp4", mode=mode)
    assert torch.equal(rec, emu)
    assert torch.equal(up["scales"], fields["scales"])
    assert torch.equal(up["scale_super"], fields["scale_super"])
    assert torch.equal(up["scale_sub"], fields["scale_sub"])


# T3 — byte accounting: type_size 4k+9, packed nbytes, §2.1 bpw ladder.
def test_two_tier_type_size_and_effective_bits():
    for k, bpw in ((12, 1.78125), (13, 1.90625), (14, 2.03125),
                   (16, 2.28125), (18, 2.53125), (20, 2.78125),
                   (24, 3.28125)):
        assert cb.nvfp4_cb_type_size(k, "fp4", "two_tier") == 4 * k + 9
        assert cb.nvfp4_cb_effective_bits(
            k, "fp4", "two_tier") == pytest.approx(bpw, abs=1e-12)
        assert cb.nvfp4_cb_effective_bits(
            k, "fp4", "v1") == pytest.approx(k / 8 + 0.5, abs=1e-12)
    w = _real_magnitude_w(seed=3)
    packed, _ = cb.nvfp4_cb_pack(w, 14, grid="fp4", mode="product",
                                 scale_coding="two_tier")
    assert packed.shape == (64, (512 // 256) * (4 * 14 + 9))
    # Registered FormatSpecs retain their legacy nominal rate for API/read
    # compatibility; producer allocation uses nvfp4_cb_footprint instead.
    assert fr.get_format("NVFP4_CB_K14").effective_bits == pytest.approx(
        2.25, abs=1e-12)


# T4 — v1 regression: default decode path is v1 and unchanged.
def test_two_tier_v1_fixture_still_decodes():
    w = _real_magnitude_w(seed=4)
    packed, fields = cb.nvfp4_cb_pack(w, 14, grid="fp4", mode="product")
    assert packed.shape[-1] == (512 // 256) * (4 * 14 + 16)   # v1 type_size
    up = cb.nvfp4_cb_unpack(packed, 14, "fp4", "product", tuple(w.shape),
                            codebook=fields["codebook"])      # no scale_coding
    rec = cb.nvfp4_cb_reconstruct(up, 14, grid="fp4", mode="product")
    emu = cb.nvfp4_cb_reconstruct(fields, 14, grid="fp4", mode="product")
    assert torch.equal(rec, emu)
    assert "scale_super" not in up


# T5 — determinism: encode twice => identical bytes (CPU and CUDA).
@pytest.mark.parametrize("device", _DEVICES)
def test_two_tier_determinism(device):
    w = _real_magnitude_w(seed=5).to(device)
    cw = (torch.rand(512) + 0.05).to(device)
    p1, _ = cb.nvfp4_cb_pack(w, 14, grid="fp4", mode="product",
                             col_weights=cw, scale_coding="two_tier")
    p2, _ = cb.nvfp4_cb_pack(w, 14, grid="fp4", mode="product",
                             col_weights=cw, scale_coding="two_tier")
    assert torch.equal(p1, p2)


# T6 — edges: all-zero group / superblock, 448 top, subnormal snap-up.
def test_two_tier_edges():
    torch.manual_seed(6)
    w = torch.randn(4, 512) * 0.02
    w[0, :16] = 0.0                                  # all-zero group
    w[1, :256] = 0.0                                 # all-zero superblock
    w[2, :16] = 448.0 * 6.0                          # amax at the 448 scale top
    fields = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="product",
                                scale_coding="two_tier")
    s = fields["scales"]
    assert bool((s > 0).all())                       # T has no zero (spec)
    assert torch.equal(s.to(torch.float8_e4m3fn).to(torch.float32), s)
    assert float(s[2, 0]) == 448.0                   # top reachable: 1.75*2^8
    # zero regions: scale is the deterministic first-legal candidate (bytes
    # pinned below); recon is bounded by grid*scale (the lattice need not
    # contain an exact zero codeword — same as v1).
    recon = cb.nvfp4_cb_reconstruct(fields, 14, grid="fp4", mode="product")
    assert float(recon[0, :16].abs().max()) <= 6.0 * float(s[0, 0])
    assert float(recon[1, :256].abs().max()) <= 6.0 * float(s[1, :16].max())
    # determinism of the degenerate bytes
    p1 = cb.nvfp4_cb_assemble_bytes(fields, 14, "fp4", "product")
    f2 = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="product",
                            scale_coding="two_tier")
    p2 = cb.nvfp4_cb_assemble_bytes(f2, 14, "fp4", "product")
    assert torch.equal(p1, p2)


def test_two_tier_snap_up_no_clip():
    # One tiny-amax group among 15 big ones: its ideal scale sits below the
    # superblock's reachable floor at the chosen E -> snaps UP (>= ideal), the
    # no-clip direction: |w/s| <= 6 everywhere in that group.
    torch.manual_seed(7)
    w = torch.randn(1, 256) * 0.05
    w[0, :16] *= 1e-4                                # tiny group 0
    fields = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="product",
                                scale_coding="two_tier")
    s0 = float(fields["scales"][0, 0])
    ideal0 = float(w[0, :16].abs().amax() / 6.0)
    assert s0 >= ideal0
    assert float((w[0, :16].abs() / s0).max()) <= 6.0 + 1e-6


# Candidate diversity: the un-collapse the two-tier coding buys. On a
# subnormal-band tensor the v1 clip sweep collapses to a handful of distinct
# e4m3 candidates per group; the two-tier window offers >= 16 distinct legal
# reachable values per superblock.
def test_two_tier_candidate_diversity_vs_v1():
    w = _real_magnitude_w(seed=8)
    amax = cb._group_amax(w, "fp4")
    v1 = cb._candidate_scales(amax, "fp4", cb._SCALE_SWEEP_CANDIDATES)
    v1_distinct = torch.tensor([
        len(set(v1[:, r, g].tolist()))
        for r in range(0, 64, 16) for g in range(0, 32, 8)])
    assert float(v1_distinct.float().mean()) <= 8.0   # the collapse (v1)
    _, compose, legal = cb._two_tier_tables("cpu")
    e_lo, e_hi, W = cb._two_tier_window(amax)
    for r in range(0, 64, 16):
        for sb in range(2):
            vals = set()
            for i in range(W):
                E = min(int(e_lo[r, sb]) + i, int(e_hi[r, sb]))
                vals |= set(compose[E][legal[E]].tolist())
            assert len(vals) >= 16


# v2 sweep quality: never worse than the v1 one-shot; ties the v1 free sweep
# on subnormal-band tensors (v2's real win is bytes — 0.28 vs 0.5 bpw scale).
# The v1 balanced encoder now scans an EXHAUSTIVE scale grid (>= the old
# greedy hill's reach), so v1 error ties v2 here rather than trailing it.
@pytest.mark.parametrize("mode", ["product"])
def test_two_tier_beats_one_shot_and_v1_sweep(mode):
    w = _real_magnitude_w(seed=9)
    cw = torch.rand(512) + 0.05
    cw2d = torch.broadcast_to(cw, w.shape).contiguous()
    wq = cb._col_weight_vectors(cw2d)
    C = cb._resolve_codebook(14, "fp4", mode, None, w.device)

    def err_of(scales):
        e, _, _ = cb._eval_candidate(w, wq, scales, "fp4", mode, C)
        return float(e.sum())

    one_shot = cb._candidate_scales(cb._group_amax(w, "fp4"), "fp4", 16)[0]
    f_v1 = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode=mode, col_weights=cw)
    f_v2 = cb.nvfp4_cb_fields(w, 14, grid="fp4", mode=mode, col_weights=cw,
                              scale_coding="two_tier")
    e_one, e_v1, e_v2 = (err_of(one_shot), err_of(f_v1["scales"]),
                         err_of(f_v2["scales"]))
    assert e_v2 <= e_one + 1e-6
    # v2 ties v1's exhaustive-grid sweep on error (within fp/basin noise).
    assert e_v2 <= e_v1 * 1.002


def test_two_tier_rejects_fp8_and_no_sweep():
    w = _real_magnitude_w(seed=10)
    with pytest.raises(ValueError, match="fp4-family only"):
        cb.nvfp4_cb_fields(w, 40, grid="fp8", mode="product",
                           scale_coding="two_tier")
    with pytest.raises(ValueError, match="IS the sweep"):
        cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="product",
                           scale_sweep=False, scale_coding="two_tier")
    with pytest.raises(ValueError, match="scale_coding"):
        cb.nvfp4_cb_fields(w, 14, grid="fp4", mode="product",
                           scale_coding="v3")


def test_two_tier_exporter_writes_layout_version(export_dir):
    from safetensors.torch import load_file

    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl, out = export_dir / "model", export_dir / "out"
    tens = _tiny_model(mdl)
    assign = {
        "model.layers.0.mlp.gate_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
        "model.layers.1.mlp.gate_proj": {"data_type": "fp8_cb", "cb_k": 40},
    }
    apath = export_dir / "assign.json"
    _write_assignment(apath, assign)
    cw = {q: torch.rand(256) + 0.05 for q in assign}
    export_nvfp4_cb(mdl, apath, out, cw, device="cpu",
                    scale_coding="two_tier",
                    allow_unstamped_research=True)
    qc = json.loads((out / "quant_config.json").read_text())
    assert qc["layout_version"] == 2
    assert qc["provenance"]["scale_coding"] == "two_tier"
    ot = load_file(str(out / "model.safetensors"))
    for g in qc["config_groups"].values():
        s = g["scheme"]
        if s["grid"] == "fp4":
            sc = s["scale_coding"]
            assert sc["kind"] == "two_tier" and sc["sub_bits"] == 4
            assert sc["super_bias"] == 127 and len(sc["table"]) == 16
            tb = torch.tensor(sc["table"], dtype=torch.float32)
            assert torch.equal(
                tb.to(torch.float8_e4m3fn).to(torch.float32), tb)
            assert s["type_size"] == 4 * s["k"] + 9
            # round-trip through the v2 scheme (codebooks in the .pqcb sidecar)
            q = g["targets"][0]
            ref = s["codebook_ref"]
            cbf = load_file(str(out / qc["codebook_file"]))
            codebook = (tuple(cbf[r].float() for r in ref)
                        if isinstance(ref, list) else cbf[ref].float())
            w = tens[q + ".weight"].float()
            up = cb.nvfp4_cb_unpack(ot[q + ".cb_qweight"], s["k"], "fp4",
                                    s["mode"], tuple(w.shape),
                                    codebook=codebook,
                                    scale_coding="two_tier")
            emu_f = cb.nvfp4_cb_fields(w, s["k"], grid="fp4", mode=s["mode"],
                                       col_weights=cw[q], codebook=codebook,
                                       scale_coding="two_tier")
            assert torch.equal(
                cb.nvfp4_cb_reconstruct(up, s["k"], grid="fp4",
                                        mode=s["mode"]),
                cb.nvfp4_cb_reconstruct(emu_f, s["k"], grid="fp4",
                                        mode=s["mode"]))
        else:
            assert "scale_coding" not in s          # fp8: no scale plane
            assert s["type_size"] == 4 * s["k"]


def test_exporter_explicit_legacy_v1_remains_readable(export_dir):
    from safetensors.torch import load_file

    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl, out = export_dir / "model", export_dir / "out"
    _tiny_model(mdl)
    qname = "model.layers.0.mlp.gate_proj"
    apath = export_dir / "assign.json"
    _write_assignment(apath, {qname: "NVFP4_CB_K16"})
    export_nvfp4_cb(
        mdl,
        apath,
        out,
        {qname: torch.rand(256) + 0.05},
        device="cpu",
        scale_coding="v1",
        allow_unstamped_research=True,
    )
    qc = json.loads((out / "quant_config.json").read_text())
    assert "layout_version" not in qc
    legacy_stamp = cb_serialization_context_stamp(
        CBSerializationContext.legacy_v1(),
        formats=["NVFP4_CB_K16"],
    )
    assert qc["provenance"]["serialized_payload"]["context"] == {
        key: legacy_stamp[key]
        for key in (
            "scale_coding",
            "layout_version",
            "codebook_source",
            "scale_sweep",
            "ldlq",
            "encode_tier",
            "renderer_abi",
        )
    }
    group = next(iter(qc["config_groups"].values()))
    assert "scale_coding" not in group["scheme"]
    assert group["scheme"]["type_size"] == 4 * 16 + 16
    tensors = load_file(str(out / "model.safetensors"))
    packed = tensors[qname + ".cb_qweight"]
    assert packed.shape == (128, 4 * 16 + 16)
    sidecar = load_file(str(out / qc["codebook_file"]))
    refs = group["scheme"]["codebook_ref"]
    codebook = tuple(sidecar[ref].float() for ref in refs)
    fields = cb.nvfp4_cb_unpack(
        packed,
        16,
        "fp4",
        "product",
        (128, 256),
        codebook=codebook,
    )
    assert "scale_coding" not in fields  # absence is the legacy-v1 read tag


def test_exporter_rejects_allocator_serialization_context_drift(export_dir):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl, out = export_dir / "model", export_dir / "out"
    _tiny_model(mdl)
    qname = "model.layers.0.mlp.gate_proj"
    apath = export_dir / "assign.json"
    _write_assignment(apath, {
        qname: "NVFP4_CB_K16",
        "__prismaquant__": {
            "schema": "prismaquant.layer_config_meta.v1",
            "cb_serialized_payload": _production_cb_stamp(
                ["NVFP4_CB_K16"]
            ),
        },
    })
    with pytest.raises(ValueError, match="differs from allocator recipe"):
        export_nvfp4_cb(
            mdl,
            apath,
            out,
            {qname: torch.rand(256) + 0.05},
            device="cpu",
            scale_coding="v1",
        )


# ===========================================================================
# Tiered encoder (docs/lanes/nvfp4-cb/encode_tiers.md).
# ===========================================================================

def test_encode_tier_resolution_env(monkeypatch):
    assert cb._resolve_encode_tier("max") == "max"
    assert cb._resolve_encode_tier(None) == cb._ENCODE_TIER_DEFAULT
    monkeypatch.setenv(cb._ENCODE_TIER_ENV, "fast")
    assert cb._resolve_encode_tier(None) == "fast"
    with pytest.raises(ValueError, match="encode tier"):
        cb._resolve_encode_tier("turbo")


# max tier == the pre-tier encoder, bit-identical (CPU checksum pin).
def test_max_tier_bit_identity_regression():
    import hashlib
    torch.manual_seed(42)
    w = torch.randn(32, 512) * 0.02
    cw = torch.rand(512) + 0.05
    digests = {}
    for coding in ("v1", "two_tier"):
        packed, _ = cb.nvfp4_cb_pack(w, 14, grid="fp4", mode="product",
                                     col_weights=cw, scale_coding=coding,
                                     encode_tier="max")
        digests[coding] = hashlib.sha256(
            packed.cpu().numpy().tobytes()).hexdigest()[:16]
    w8 = torch.randn(32, 512) * 0.3
    packed, _ = cb.nvfp4_cb_pack(w8, 40, grid="fp8", mode="product",
                                 col_weights=cw, encode_tier="max")
    digests["fp8"] = hashlib.sha256(
        packed.cpu().numpy().tobytes()).hexdigest()[:16]
    # Pinned from the pre-tier encoder (commit ab1ccc5 behavior). If this
    # fails, the max tier is no longer bit-identical to the original sweep.
    assert digests == {
        "v1": "231729a4bea30e0a",
        "two_tier": "b51df7ffe4c30f2e",
        "fp8": "fdabaadd1ed75eee",
    }


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("tier", ["fast", "balanced"])
def test_tier_determinism(device, tier):
    torch.manual_seed(13)
    w = (torch.randn(64, 512) * 0.02).to(device)
    cw = (torch.rand(512) + 0.05).to(device)
    qdq = cb.make_nvfp4_cb_qdq(16, "fp4", "product", encode_tier=tier)
    assert torch.equal(qdq(w, cw), qdq(w, cw))
    p1, _ = cb.nvfp4_cb_pack(w, 16, grid="fp4", mode="product",
                             col_weights=cw, scale_coding="two_tier",
                             encode_tier=tier)
    p2, _ = cb.nvfp4_cb_pack(w, 16, grid="fp4", mode="product",
                             col_weights=cw, scale_coding="two_tier",
                             encode_tier=tier)
    assert torch.equal(p1, p2)


# fast-tier quality bound, asserted from the measured table with slack:
# measured fast wrecon deltas vs max were <= +0.9% (fp8) / better than max
# (fp4 v1) / +0.0% (v2); assert a 5% ceiling so noise cannot flake.
@pytest.mark.parametrize("grid,k,coding,std", [
    ("fp8", 44, "v1", 0.3),
    ("fp4", 16, "v1", 0.02),
    ("fp4", 16, "two_tier", 0.02),
])
def test_fast_tier_quality_bound(grid, k, coding, std):
    torch.manual_seed(14)
    w = torch.randn(128, 1024) * std
    cw = torch.rand(1024) + 0.05

    def wrecon(tier):
        r = cb.make_nvfp4_cb_qdq(k, grid, "product", scale_coding=coding,
                                 encode_tier=tier)(w, cw)
        return float((torch.broadcast_to(cw, w.shape)
                      * (w - r).pow(2)).sum())

    assert wrecon("fast") <= wrecon("max") * 1.05
    assert wrecon("balanced") <= wrecon("max") * 1.03


# tiers still produce decode-valid, pack-parity bytes (the layout contract
# is tier-independent).
@pytest.mark.parametrize("tier", ["fast", "balanced"])
def test_tier_pack_parity_and_validity(tier):
    torch.manual_seed(15)
    w = torch.randn(32, 512) * 0.02
    cw = torch.rand(512) + 0.05
    packed, fields = cb.nvfp4_cb_pack(w, 16, grid="fp4", mode="product",
                                      col_weights=cw,
                                      scale_coding="two_tier",
                                      encode_tier=tier)
    up = cb.nvfp4_cb_unpack(packed, 16, "fp4", "product", tuple(w.shape),
                            codebook=fields["codebook"],
                            scale_coding="two_tier")
    rec = cb.nvfp4_cb_reconstruct(up, 16, grid="fp4", mode="product")
    emu = cb.nvfp4_cb_reconstruct(fields, 16, grid="fp4", mode="product")
    assert torch.equal(rec, emu)
    s = fields["scales"]
    assert torch.equal(s.to(torch.float8_e4m3fn).to(torch.float32), s)


def test_predict_cb_ladder_costs_shape_and_holdout():
    torch.manual_seed(16)
    w = torch.randn(64, 512) * 0.02
    cw = torch.rand(512) + 0.05
    res = cb.predict_cb_ladder_costs(
        w, tuple(range(12, 25)), grid="fp4", mode="product",
        col_weights=cw, anchors=(12, 18, 24), holdout=15,
        scale_coding="two_tier", encode_tier="fast")
    assert set(res["measured"]) == {12, 18, 24}
    assert set(res["predicted"]) == set(range(12, 25))
    # monotone: more index bits -> lower predicted distortion
    pv = [res["predicted"][k] for k in range(12, 25)]
    assert all(a > b for a, b in zip(pv, pv[1:]))
    hold = res["holdout"]
    assert hold["k"] == 15 and hold["rel_error"] >= 0.0


# ===========================================================================
# MoE readiness: packed-expert export chain + serving-unit uniformity.
# ===========================================================================

class _CBMoEProfile:
    """Per-layer packed-expert serving unit (the vLLM FusedMoE constraint:
    experts uniform per layer; mix across layers, never within)."""

    def fused_sibling_group(self, name: str) -> str | None:
        return None

    def packed_expert_format_group(self, name: str) -> str | None:
        parts = name.split(".")
        if "experts" in parts:
            layer = ".".join(parts[:parts.index("mlp")])
            return f"{layer}.mlp.experts"
        return None


def test_moe_expert_uniformity_across_cb_rungs():
    from prismaquant.allocator_solver import promote_serving_units

    names_l0 = [f"model.layers.0.mlp.experts.{p}"
                for p in ("gate_up_proj", "down_proj")]
    names_l1 = [f"model.layers.1.mlp.experts.{p}"
                for p in ("gate_up_proj", "down_proj")]
    assignment = {
        names_l0[0]: "NVFP4_CB_K14",       # mixed WITHIN layer 0's unit
        names_l0[1]: "NVFP4_CB_K16",
        names_l1[0]: "NVFP4_CB_K12",       # uniform in layer 1
        names_l1[1]: "NVFP4_CB_K12",
        "model.layers.0.self_attn.q_proj": "NVFP4_CB_K20",   # non-expert
    }
    fam = [s for s in fr.list_formats()
           if s.family in ("nvfp4_cb", "fp8_cb")] + [fr.get_format("BF16")]
    rank = {s.name: i for i, s in enumerate(
        sorted(fam, key=lambda s: s.effective_bits))}
    promoted = promote_serving_units(assignment, rank,
                                     profile=_CBMoEProfile())
    # layer 0: promoted to ONE rung = the max-rank member (K16 = 2.5 bpw)
    assert promoted[names_l0[0]] == promoted[names_l0[1]] == "NVFP4_CB_K16"
    # layer 1: untouched (mix across layers is legal)
    assert promoted[names_l1[0]] == promoted[names_l1[1]] == "NVFP4_CB_K12"
    # non-expert Linear not dragged into the unit
    assert promoted["model.layers.0.self_attn.q_proj"] == "NVFP4_CB_K20"


def test_cb_legality_rejects_odd_shapes_falls_back():
    from prismaquant.allocator_candidates import check_format_applicability

    # in_features % 256 != 0: every CB rung is illegal (the 256-superblock
    # legality doubles as the vector-tiling gate); BF16 stays legal.
    for fmt in ("NVFP4_CB_K14", "NVFP4_CB_K16", "FP8_CB_K44"):
        v = check_format_applicability((64, 320), fmt)
        assert not v.legal and v.reason == "group_divisibility"
        assert check_format_applicability((64, 512), fmt).legal
    assert check_format_applicability((64, 320), "BF16").legal


def _tiny_moe_model(mdl: Path, n_exp: int = 3, in_f: int = 256):
    from safetensors.torch import save_file

    mdl.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    tens = {
        "model.layers.0.mlp.experts.gate_up_proj.weight":
            (torch.randn(n_exp, 96, in_f) * 0.3).to(torch.bfloat16),
        "model.layers.0.mlp.experts.down_proj.weight":
            (torch.randn(n_exp, 48, in_f) * 0.3).to(torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight":
            (torch.randn(128, in_f) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(in_f, dtype=torch.bfloat16),
    }
    save_file(tens, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(
        json.dumps({"architectures": ["TinyMoE"], "hidden_size": in_f}))
    return tens


@pytest.mark.parametrize("source", ["lattice", "learned"])
def test_exporter_packed_experts_roundtrip(export_dir, source, monkeypatch):
    from safetensors.torch import load_file

    from prismaquant import cb_learned_bundle
    from prismaquant import gridbook_runtime_pin as runtime_pin
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    # The learned arm exports routed-MoE learned refs, which are gated on
    # GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION. Supply the
    # capability explicitly rather than letting the external production pin
    # decide whether this test runs its subject.
    monkeypatch.setattr(
        cb_learned_bundle, "load_gridbook_runtime_pin",
        lambda: runtime_pin.parse_gridbook_runtime_pin({
            "schema": runtime_pin.GRIDBOOK_RUNTIME_PIN_SCHEMA,
            "repository": "https://github.com/RobTand/gridbook.git",
            "commit": "a" * 40,
            "version": (runtime_pin
                        .GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION),
            "version_is_release": False,
            "runtime_contract_schema": (
                runtime_pin.GRIDBOOK_RUNTIME_CONTRACT_SCHEMA
            ),
            "required_abi_features": dict(
                runtime_pin.GRIDBOOK_REQUIRED_ABI_FEATURES
            ),
        }))

    mdl, out = export_dir / "model", export_dir / "out"
    tens = _tiny_moe_model(mdl)
    assign = {
        "model.layers.0.mlp.experts.gate_up_proj":
            {"data_type": "fp8_cb", "cb_k": 40},
        "model.layers.0.mlp.experts.down_proj":
            {"data_type": "nvfp4_cb", "cb_k": 16},
        "model.layers.0.self_attn.q_proj":
            {"data_type": "nvfp4_cb", "cb_k": 16},
    }
    apath = export_dir / "assign.json"
    _write_assignment(apath, assign)
    torch.manual_seed(1)
    cw = {
        # per-expert (E, 1, in) — the gguf _qw_blocks convention
        "model.layers.0.mlp.experts.gate_up_proj": torch.rand(3, 1, 256) + .05,
        "model.layers.0.mlp.experts.down_proj": torch.rand(3, 1, 256) + 0.05,
        "model.layers.0.self_attn.q_proj": torch.rand(256) + 0.05,
    }
    spec = ({"source": "learned", "train": True}
            if source == "learned" else {"source": "lattice"})
    counts = export_nvfp4_cb(mdl, apath, out, cw,
                             shared_codebook_spec=spec, device="cpu",
                             allow_unstamped_research=True)
    assert counts["FP8_CB_K40"] == 1 and counts["NVFP4_CB_K16"] == 2

    ot = load_file(str(out / "model.safetensors"))
    qc = json.loads((out / "quant_config.json").read_text())
    cbf = load_file(str(out / qc["codebook_file"]))
    # stacked layout: (E, out, bytes); fp8 scales (E, out)
    gu = ot["model.layers.0.mlp.experts.gate_up_proj.cb_qweight"]
    assert gu.shape == (3, 96, (256 // 256) * cb.nvfp4_cb_type_size(40, "fp8"))
    ws = ot["model.layers.0.mlp.experts.gate_up_proj.weight_scale"]
    assert ws.shape == (3, 96)
    dn = ot["model.layers.0.mlp.experts.down_proj.cb_qweight"]
    assert dn.shape == (
        3,
        48,
        cb.nvfp4_cb_type_size(16, "fp4", cb.SCALE_CODING_TWO_TIER),
    )
    assert ("model.layers.0.mlp.experts.down_proj.weight_scale") not in ot

    # per-expert round-trip == whole-stack emulation with per-expert weights
    for g in qc["config_groups"].values():
        s = g["scheme"]
        coding = (cb.SCALE_CODING_TWO_TIER
                  if "scale_coding" in s else cb.SCALE_CODING_V1)
        for q in g["targets"]:
            if "experts" not in q:
                continue
            w = tens[q + ".weight"].float()
            ref = s["codebook_ref"]
            codebook = (tuple(cbf[r].float() for r in ref)
                        if isinstance(ref, list) else cbf[ref].float())
            packed = ot[q + ".cb_qweight"]
            scales = ot.get(q + ".weight_scale")
            E = w.shape[0]
            up = cb.nvfp4_cb_unpack(
                packed.reshape(E * w.shape[1], -1), s["k"], s["grid"],
                s["mode"], (E * w.shape[1], w.shape[2]), codebook=codebook,
                scales=(scales.reshape(-1, 1) if scales is not None
                        else None),
                scale_coding=coding)
            rec = cb.nvfp4_cb_reconstruct(
                up, s["k"], grid=s["grid"], mode=s["mode"]).reshape(w.shape)
            emu_f = cb.nvfp4_cb_fields(
                w, s["k"], grid=s["grid"], mode=s["mode"],
                col_weights=cw[q], codebook=codebook,
                scale_coding=coding)
            emu = cb.nvfp4_cb_reconstruct(
                emu_f, s["k"], grid=s["grid"], mode=s["mode"])
            assert torch.equal(rec, emu.reshape(w.shape))


def test_exporter_rejects_wrong_expert_col_weights(export_dir):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl, out = export_dir / "model", export_dir / "out"
    _tiny_moe_model(mdl)
    assign = {"model.layers.0.mlp.experts.down_proj":
              {"data_type": "nvfp4_cb", "cb_k": 16}}
    apath = export_dir / "assign.json"
    _write_assignment(apath, assign)
    cw = {"model.layers.0.mlp.experts.down_proj": torch.rand(2, 1, 256)}
    with pytest.raises(ValueError, match="col_weights"):
        export_nvfp4_cb(mdl, apath, out, cw, device="cpu")


# ===========================================================================
# M4-hybrid empirical expert cost for the CB lane (moe_cb_design.md §3).
# ===========================================================================

def _mk_cost_row(fmt_costs):
    return {f: {"predicted_dloss": v, "cost_source": "local"}
            for f, v in fmt_costs.items()}


def test_expert_empirical_merge_replaces_expert_rows():
    from prismaquant.expert_empirical_cost import merge_cost_payloads

    exp_name = "model.layers.0.mlp.experts.gate_up_proj"
    dense_name = "model.layers.0.self_attn.q_proj"
    base = {
        "stats": {exp_name: {"h_trace": 1.0}, dense_name: {"h_trace": 2.0}},
        "costs": {exp_name: _mk_cost_row({"NVFP4_CB_K16": 9.9}),
                  dense_name: _mk_cost_row({"NVFP4_CB_K16": 0.5})},
        "provenance": {
            "origin": "local",
            "cb_serialized_payload": _production_cb_stamp(
                ["NVFP4_CB_K16"]
            ),
        },
    }
    e_stats = {exp_name: {"h_trace": 0.0, "n_params": 10}}
    e_costs = {exp_name: {"NVFP4_CB_K16": {
        "predicted_dloss": 0.123, "cost_source": "empirical_unit_kl"}}}

    # CB lane: replace semantics — expert row swapped, dense untouched,
    # provenance records the replacement, allocator payload keys intact.
    merged = merge_cost_payloads(base, e_stats, e_costs,
                                 formats=["NVFP4_CB_K16", "BF16"],
                                 replace_experts=True)
    assert merged["costs"][exp_name]["NVFP4_CB_K16"][
        "cost_source"] == "empirical_unit_kl"
    assert merged["costs"][dense_name]["NVFP4_CB_K16"][
        "predicted_dloss"] == 0.5
    assert merged["stats"][exp_name]["h_trace"] == 0.0
    assert merged["provenance"]["replaced_smooth_expert_rows"] == [exp_name]
    assert merged["provenance"]["origin"] == "local"
    assert merged["formats"] == ["NVFP4_CB_K16", "BF16"]
    # AURA lane: the same collision is an error without the flag.
    with pytest.raises(RuntimeError, match="collision"):
        merge_cost_payloads(base, e_stats, e_costs,
                            formats=["NVFP4_CB_K16"])


class _UnitHolder(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(w, requires_grad=False)


def test_expert_empirical_cb_weighted_render_inplace():
    from prismaquant.expert_empirical_cost import _quantize_unit_inplace

    torch.manual_seed(20)
    w = (torch.randn(3, 16, 256) * 0.02).to(torch.bfloat16)
    cw = torch.rand(3, 1, 256) + 0.05
    qn = "model.layers.0.mlp.experts"
    full = f"{qn}.gate_up_proj"

    mod = _UnitHolder(w.clone())
    _quantize_unit_inplace(mod, ["gate_up_proj"], "NVFP4_CB_K16",
                           col_weights={full: cw}, unit_qname=qn)
    direct = cb.make_nvfp4_cb_qdq(
        16, "fp4", "product", scale_coding="two_tier"
    )(
        w.float(), cw).to(torch.bfloat16)
    assert torch.equal(mod.gate_up_proj.data, direct)

    # missing col_weights for a CB format is the rendering confound: raise.
    with pytest.raises(ValueError, match="col_weights"):
        _quantize_unit_inplace(_UnitHolder(w.clone()), ["gate_up_proj"],
                               "NVFP4_CB_K16", col_weights={},
                               unit_qname=qn)
    # scalar-family formats keep the chunked unweighted path.
    mod2 = _UnitHolder(w.clone())
    _quantize_unit_inplace(mod2, ["gate_up_proj"], "FP8_E4M3",
                           col_weights=None, unit_qname=qn)
    assert not torch.equal(mod2.gate_up_proj.data, w)


def test_expert_empirical_tier_inheritance(monkeypatch):
    from prismaquant.expert_empirical_cost import _quantize_unit_inplace

    torch.manual_seed(21)
    w = (torch.randn(2, 16, 256) * 0.02).to(torch.bfloat16)
    cw = torch.rand(256) + 0.05
    qn = "u"
    resolved = []
    original_resolve = cb._resolve_encode_tier

    def recording_resolve(tier):
        value = original_resolve(tier)
        resolved.append(value)
        return value

    monkeypatch.setattr(cb, "_resolve_encode_tier", recording_resolve)
    for tier in ("max", "fast"):
        monkeypatch.setenv(cb._ENCODE_TIER_ENV, tier)
        mod = _UnitHolder(w.clone())
        _quantize_unit_inplace(mod, ["gate_up_proj"], "NVFP4_CB_K16",
                               col_weights={f"{qn}.gate_up_proj": cw},
                               unit_qname=qn)
    # The production-context closure resolves the env per call. Layout-v2's
    # reachable scale set can make these tiers bit-identical on a small input,
    # so environment inheritance—not a forced output difference—is the
    # contract this test needs to pin.
    assert resolved == ["max", "fast"]


def test_cb_ladder_split_and_fit():
    from prismaquant.expert_empirical_cost import (
        _cb_ladder_fit,
        _cb_ladder_split,
    )

    fmts = [f"NVFP4_CB_K{k}" for k in (12, 14, 16, 20, 24)] + ["FP8_CB_K44"]
    ladders = _cb_ladder_split(fmts)
    # Per-family ladders: the lone FP8 rung can never join the NVFP4 fit
    # (different grid/law); with <4 FP8 rungs only the NVFP4 ladder pays.
    assert ladders is not None and len(ladders) == 1
    kmap, anchors, holdout, predicted = ladders[0]
    assert set(anchors) == {"NVFP4_CB_K12", "NVFP4_CB_K16", "NVFP4_CB_K24"}
    assert holdout in predicted or holdout not in anchors
    assert "FP8_CB_K44" not in kmap          # different family, separate map
    # exact RD law -> holdout accepted, predictions exact.
    # (R20: _cb_ladder_fit returns (pred, rel, tol_used) — the gate derives
    # its own tolerance where a noise datum exists, so the tolerance it
    # actually applied is part of the answer. Every k here is even, so the
    # ceil-first split is even and R(k) is exactly proportional to
    # 2^(-k/4): this exact RD law IS the shared law's own model.)
    kls = {f: 2.0 ** (3.0 - kmap[f] / 4.0) for f in kmap}
    pred, rel, tol = _cb_ladder_fit(
        kls, kmap, anchors, holdout, predicted, 0.10)
    assert rel < 1e-9 and tol == 0.10
    for f, v in pred.items():
        assert v == pytest.approx(kls[f], rel=1e-9)
    # corrupted holdout -> gate FAILS -> caller measures everything.
    bad = dict(kls)
    bad[holdout] *= 1.5
    pred2, rel2, _ = _cb_ladder_fit(
        bad, kmap, anchors, holdout, predicted, 0.10)
    assert pred2 is None and rel2 > 0.10
    # short ladders never interpolate
    assert cb_ladder_none() is None


def cb_ladder_none():
    from prismaquant.expert_empirical_cost import _cb_ladder_split
    return _cb_ladder_split(["NVFP4_CB_K12", "NVFP4_CB_K16", "BF16"])


def test_expert_empirical_parser_args():
    from prismaquant.expert_empirical_cost import _build_parser

    args = _build_parser().parse_args([
        "--model", "m", "--output", "o", "--formats",
        "NVFP4_CB_K12,NVFP4_CB_K16,BF16", "--merge-base", "b.pkl",
        "--replace-experts", "--col-weights", "cw.pkl",
        "--cb-ladder-interp", "--ladder-holdout-tol", "0.05",
    ])
    assert args.replace_experts and args.cb_ladder_interp
    assert args.col_weights == "cw.pkl"
    assert args.ladder_holdout_tol == 0.05
    d = _build_parser().parse_args(["--model", "m", "--output", "o"])
    assert not d.replace_experts and not d.cb_ladder_interp


# ===========================================================================
# FP8_SOURCE passthrough in the mixed CB container (fp8_source_passthrough).
# ===========================================================================

def _tiny_fp8_source_model(mdl: Path, in_f: int = 256):
    """Synthetic native-FP8 dir: one fp8_e4m3fn Linear + 128x128
    weight_scale_inv (MiniMax/DeepSeek convention) + one bf16 CB-target
    Linear + a norm sidecar."""
    from safetensors.torch import save_file

    mdl.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3)
    w_fp8 = (torch.randn(128, in_f) * 0.3).to(torch.float8_e4m3fn)
    # 128x128 block scale_inv, ceil-div blocks along each dim.
    si = (torch.rand((128 + 127) // 128, (in_f + 127) // 128) + 0.1).float()
    tens = {
        "model.layers.0.self_attn.q_proj.weight": w_fp8,
        "model.layers.0.self_attn.q_proj.weight_scale_inv": si,
        "model.layers.0.mlp.gate_proj.weight":
            (torch.randn(128, in_f) * 0.3).to(torch.bfloat16),
        "model.norm.weight": torch.ones(in_f, dtype=torch.bfloat16),
    }
    save_file(tens, str(mdl / "model.safetensors"))
    (mdl / "config.json").write_text(
        json.dumps({
            "architectures": ["TinyFP8"],
            "hidden_size": in_f,
            "quantization_config": {"weight_block_size": [128, 128]},
        }))
    return tens


def test_fp8_source_passthrough_verbatim(export_dir):
    from safetensors.torch import load_file

    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    mdl, out = export_dir / "model", export_dir / "out"
    tens = _tiny_fp8_source_model(mdl)
    assign = {
        "model.layers.0.self_attn.q_proj": {
            "data_type": "fp8_e4m3", "bits": 8, "group_size": 128},  # FP8_SOURCE
        "model.layers.0.mlp.gate_proj": {"data_type": "nvfp4_cb", "cb_k": 16},
    }
    apath = export_dir / "assign.json"
    _write_assignment(apath, assign)
    cw = {"model.layers.0.mlp.gate_proj": torch.rand(256) + 0.05}
    counts = export_nvfp4_cb(
        mdl,
        apath,
        out,
        cw,
        device="cpu",
        allow_unstamped_research=True,
    )
    assert counts["FP8_SOURCE"] == 1 and counts["NVFP4_CB_K16"] == 1

    ot = load_file(str(out / "model.safetensors"))
    q = "model.layers.0.self_attn.q_proj"
    # verbatim: fp8 weight bytes identical, scale_inv -> weight_scale (fp32)
    assert ot[q + ".weight"].dtype == torch.float8_e4m3fn
    assert torch.equal(
        ot[q + ".weight"].view(torch.uint8),
        tens[q + ".weight"].view(torch.uint8))
    assert torch.equal(ot[q + ".weight_scale"],
                       tens[q + ".weight_scale_inv"].float())
    assert (q + ".weight_scale_inv") not in ot   # renamed, not duplicated

    qc = json.loads((out / "quant_config.json").read_text())
    # FP8_SOURCE is a quantized scheme, NOT ignored.
    assert q not in qc["ignore"]
    assert qc["provenance"]["fp8_source_targets"] == 1
    # its config group is the stock CT block-fp8 scheme (no "scheme" key ->
    # plugin delegates to compressed-tensors), targeting the fp8 Linear.
    src_groups = [g for g in qc["config_groups"].values()
                  if g.get("format") == "float-quantized"
                  and "scheme" not in g]
    assert len(src_groups) == 1
    g = src_groups[0]
    assert g["weights"]["strategy"] == "block"
    assert g["weights"]["block_structure"] == [128, 128]
    # targets are compressed-tensors regex form (re:^...q_proj$)
    import re as _re
    assert any(t.startswith("re:") and _re.match(t[3:], q)
               for t in g["targets"])


def test_fp8_source_rejects_non_fp8_source(export_dir):
    from prismaquant.export_nvfp4_cb import export_nvfp4_cb

    # A bf16-source Linear assigned FP8_SOURCE must hard-fail (passthrough-only,
    # never synthesized) — the exporter guard mirrors the allocator's
    # passthrough-integrity filter.
    mdl, out = export_dir / "model", export_dir / "out"
    _tiny_model(mdl)   # all bf16, no weight_scale_inv anywhere
    assign = {"model.layers.0.mlp.gate_proj": {
        "data_type": "fp8_e4m3", "bits": 8, "group_size": 128}}
    apath = export_dir / "assign.json"
    _write_assignment(apath, assign)
    with pytest.raises(ValueError, match="passthrough-only"):
        export_nvfp4_cb(mdl, apath, out, {}, device="cpu")


def test_fp8_source_in_serving_allowlist():
    from prismaquant.serving_profiles import load_serving_profile

    profile = load_serving_profile("nvfp4_cb")
    allow = profile.format_rules[0].allow_formats
    assert "FP8_SOURCE" in allow
    # FP8_SOURCE is passthrough (no 256-superblock shape rule).
    assert "FP8_SOURCE" not in profile.shape_rules[0].formats
