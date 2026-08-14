"""Pure leaf-local NVFP4-CB core adversarial tests — independent oracle, v4 ABI.

Oracle is built from low-level raw encoder + fixed-codebook LDLQ
+ held-out activation_output MSE + explicit expert-major stitching.
Never calls encode_packed_parent_leaf_local or canonical identity under test
except via explicit validated evidence.
"""
import gc
import hashlib
import json
import struct
import weakref
import torch
import pytest
from unittest.mock import patch
from prismaquant.nvfp4_cb_formats import (
    nvfp4_cb_fields,
    nvfp4_cb_assemble_bytes,
    nvfp4_cb_reconstruct,
    prepare_ldlq_gate_evidence,
    ValidatedPackedLDLQEvidence,
    PreparedLDLQEvidence,
    ldlq_reassign_cb_fields,
    canonical_nvfp4_cb_single_output_mse,
    encode_packed_parent_leaf_local,
    canonical_nvfp4_cb_payload_identity,
    verify_canonical_payload_identity,
    clear_ldlq_factor_cache,
    _PAYLOAD_IDENTITY_VERSION,
    SCALE_CODING_TWO_TIER,
    _SPLIT_POLICY,
    _SPLIT_VERSION,
    LDLQ_GATE_EPSILON,
    fixed_lattice,
    _canonical_json_bytes,
    _codebook_blob_bytes,
    _canonical_split_infos_digest,
    _partition_multiset_digest,
    _row_content_hash,
)
from prismaquant import format_registry as fr
from prismaquant.cb_layout import (
    SCALE_CODING_TWO_TIER as SC2,
    family_for,
    codebook_subtable_shapes,
    subtable_bit_widths,
    type_size as serialized_type_size,
    SUPERBLOCK,
    VEC_DIM,
)
torch.manual_seed(0)
def _explicit_codebook(k: int, grid: str, mode: str):
    fam = family_for(grid, mode)
    exp_shapes = codebook_subtable_shapes(k, mode, fam.n_sub)
    bits = subtable_bit_widths(k, mode, fam.n_sub)
    if mode == "product":
        tables = tuple(fixed_lattice(b, grid, exp_shapes[i][1]) for i, b in enumerate(bits))
        return tables
    elif mode == "signed":
        return fixed_lattice(bits[0], grid, VEC_DIM, positive=True)
    else:
        raise ValueError(f"unsupported mode {mode!r} for explicit codebook")
def _make_weight(E, R_total, C, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn((E, R_total, C), generator=g, dtype=torch.bfloat16)
def _make_acts(E, C, N=16, seed=1, cold_idx=None):
    acts = []
    for e in range(E):
        if cold_idx is not None and e == cold_idx:
            acts.append(torch.empty((0, C), dtype=torch.float32))
        else:
            g = torch.Generator()
            g.manual_seed(seed + e * 1337)
            acts.append(torch.randn((N, C), generator=g, dtype=torch.float32))
    return tuple(acts)
def _leaf_col_weights(E, C, seed=0, extreme=False):
    if extreme:
        g = torch.Generator()
        g.manual_seed(seed)
        w_gate = torch.randn((E, 1, C), generator=g).abs() * 5.0 + 0.5
        g.manual_seed(seed + 1)
        w_up = torch.randn((E, 1, C), generator=g).abs() * 0.1 + 0.01
        return {"gate_proj": w_gate, "up_proj": w_up}
    else:
        out = {}
        seeds = {"gate_proj": 11, "up_proj": 22, "down_proj": 33}
        for leaf in ("gate_proj", "up_proj"):
            g = torch.Generator()
            g.manual_seed(seed + seeds[leaf])
            out[leaf] = torch.rand((E, 1, C), generator=g) + 0.05
        return out
def _manual_oracle(weight, k, grid, mode, codebook, scale_sweep, scale_coding, encode_tier, member_order, slice_boundaries, leaf_col_weights, validated):
    prepared = validated.prepared
    E, R_total, C = weight.shape
    holdout = tuple(prepared.holdout_rows)
    fit = tuple(prepared.fit_rows)
    cold = set(prepared.cold_experts) | set(prepared.insufficient_experts)
    eligible = tuple(prepared.eligible_experts)
    fmt = fr.get_format(f"NVFP4_CB_K{k}" if mode != "signed" else f"NVFP4_CB_S{k}")
    raw_by_leaf = {}
    ldlq_by_leaf = {}
    raw_mse = {}
    ldlq_mse = {}
    kept = {}
    for leaf in member_order:
        s, e = slice_boundaries[leaf]
        wl = weight[:, s:e, :].contiguous()
        cw = leaf_col_weights[leaf]
        raw = nvfp4_cb_fields(wl, k, grid=grid, mode=mode, col_weights=cw, codebook=codebook, scale_sweep=scale_sweep, scale_coding=scale_coding, encode_tier=encode_tier)
        raw_by_leaf[leaf] = raw
        if not eligible:
            ldlq = raw
        elif len(eligible) == E:
            fit_all = tuple(fit[i] for i in eligible)
            ldlq = ldlq_reassign_cb_fields(wl, raw, cw, fit_all, grid=grid, mode=mode)
        else:
            cand = {}
            for kk, v in raw.items():
                if kk in ("indices", "signs") and isinstance(v, torch.Tensor):
                    cand[kk] = v.clone()
                else:
                    cand[kk] = v
            for ei in eligible:
                w_e = wl[ei:ei+1].contiguous()
                cw_e = cw[ei:ei+1]
                Rl = e - s
                fields_e = {}
                for kk, v in raw.items():
                    if kk in ("indices", "signs", "scales", "scale_super", "scale_sub") and isinstance(v, torch.Tensor):
                        fields_e[kk] = v[ei*Rl:(ei+1)*Rl].clone()
                    else:
                        fields_e[kk] = v
                fields_e["shape"] = (1, Rl, C)
                fit_e = (fit[ei],)
                ldlq_e = ldlq_reassign_cb_fields(w_e, fields_e, cw_e, fit_e, grid=grid, mode=mode)
                for kk in ("indices", "signs"):
                    if kk in ldlq_e and isinstance(ldlq_e[kk], torch.Tensor):
                        cand[kk][ei*Rl:(ei+1)*Rl] = ldlq_e[kk]
            ldlq = cand
        ldlq_by_leaf[leaf] = ldlq
        cur_raw = []
        cur_ldlq = []
        for e_idx in range(E):
            w_leaf = weight[e_idx, s:e, :].contiguous().to(torch.bfloat16)
            leaf_shape = (E, e - s, C)
            raw_fields = {**raw, "shape": leaf_shape}
            ldlq_fields = {**ldlq, "shape": leaf_shape}
            raw_recon_all = nvfp4_cb_reconstruct(raw_fields, k, grid=grid, mode=mode)
            ldlq_recon_all = nvfp4_cb_reconstruct(ldlq_fields, k, grid=grid, mode=mode)
            raw_recon = raw_recon_all[e_idx]
            ldlq_recon = ldlq_recon_all[e_idx]
            act = holdout[e_idx]
            if act.numel() == 0:
                cur_raw.append(float("inf")); cur_ldlq.append(float("inf"))
            else:
                r_m = canonical_nvfp4_cb_single_output_mse(w_leaf, raw_recon.to(torch.bfloat16), act, fmt)
                l_m = canonical_nvfp4_cb_single_output_mse(w_leaf, ldlq_recon.to(torch.bfloat16), act, fmt)
                cur_raw.append(float(r_m)); cur_ldlq.append(float(l_m))
        raw_mse[leaf] = cur_raw
        ldlq_mse[leaf] = cur_ldlq
        mk = []
        for e_idx in range(E):
            if e_idx in cold:
                mk.append(False)
            else:
                re, le = cur_raw[e_idx], cur_ldlq[e_idx]
                if re == float("inf") or le == float("inf") or re != re or le != le:
                    mk.append(False)
                else:
                    mk.append(bool(le <= re + LDLQ_GATE_EPSILON * max(abs(re), abs(le), 1.0)))
        kept[leaf] = mk
    first = raw_by_leaf[member_order[0]]
    row_keys = [kk for kk in first if kk in ("indices", "signs", "scales", "scale_super", "scale_sub")]
    parent = {}
    for gk in ("codebook", "scale_coding"):
        if gk in first:
            parent[gk] = first[gk]
    for rk in row_keys:
        trail = tuple(first[rk].shape[1:])
        parent[rk] = torch.empty((E * R_total,) + trail, dtype=first[rk].dtype, device=first[rk].device)
    for e_idx in range(E):
        dst_off = e_idx * R_total
        cur = 0
        for leaf in member_order:
            Rl = slice_boundaries[leaf][1] - slice_boundaries[leaf][0]
            src = ldlq_by_leaf[leaf] if kept[leaf][e_idx] else raw_by_leaf[leaf]
            for rk in row_keys:
                parent[rk][dst_off + cur:dst_off + cur + Rl] = src[rk][e_idx * Rl:(e_idx + 1) * Rl]
            cur += Rl
    parent["shape"] = (E, R_total, C)
    return parent, kept, raw_mse, ldlq_mse
# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("E,K,mode", [
    (2, 12, "product"), (2, 13, "product"), (3, 12, "product"), (3, 13, "product"),
    (2, 13, "signed"), (3, 14, "signed"),
])
def test_leaf_local_bit_equality_vs_oracle(E, K, mode):
    assert mode in ("product", "signed")
    if mode == "signed":
        assert K in (13, 14, 15, 16)
    C = 256
    R_total = 8
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=K + E)
    leaf_cw = {}
    for idx, leaf in enumerate(member_order):
        g = torch.Generator(); g.manual_seed(K + E + idx * 100 + 7)
        leaf_cw[leaf] = torch.rand((E, 1, C), generator=g) + 0.05
    acts = _make_acts(E, C, N=16, seed=42)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.oracle")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    codebook = _explicit_codebook(K, "fp4", mode)
    scale_sweep = True
    scale_coding = SC2
    encode_tier = "balanced"
    grid = "fp4"
    parent, gate_info = encode_packed_parent_leaf_local(
        weight, K, grid=grid, mode=mode, codebook=codebook, scale_sweep=scale_sweep,
        scale_coding=scale_coding, encode_tier=encode_tier,
        member_order=member_order, slice_boundaries=slice_boundaries,
        leaf_col_weights=leaf_cw, prepared=validated,
    )
    o_parent, o_kept, o_raw, o_ldlq = _manual_oracle(weight, K, grid, mode, codebook, scale_sweep, scale_coding, encode_tier,
                                                      member_order, slice_boundaries, leaf_cw, validated)
    for kk in ("indices", "signs", "scales", "scale_super", "scale_sub"):
        if kk in o_parent or kk in parent:
            assert kk in parent and kk in o_parent
            assert torch.equal(parent[kk], o_parent[kk]), f"field {kk} mismatch E={E} K={K} mode={mode}"
    packed_helper = nvfp4_cb_assemble_bytes(parent, K, grid=grid, mode=mode)
    packed_oracle = nvfp4_cb_assemble_bytes(o_parent, K, grid=grid, mode=mode)
    assert torch.equal(packed_helper, packed_oracle)
    assert gate_info["per_leaf_kept"] == o_kept

def test_unequal_column_weights_proof():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=10)
    leaf_cw_extreme = _leaf_col_weights(E, C, seed=11, extreme=True)
    pooled = torch.stack([leaf_cw_extreme["gate_proj"], leaf_cw_extreme["up_proj"]], dim=0).mean(dim=0)
    leaf_cw_pooled_eq = {"gate_proj": pooled, "up_proj": pooled}
    k = 13; grid = "fp4"; mode = "product"
    acts = _make_acts(E, C, N=16, seed=20)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.pooled")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cb = _explicit_codebook(k, grid, mode)
    parent_extreme, _ = encode_packed_parent_leaf_local(weight, k, grid=grid, mode=mode, codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw_extreme, prepared=validated)
    parent_pooled, _ = encode_packed_parent_leaf_local(weight, k, grid=grid, mode=mode, codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw_pooled_eq, prepared=validated)
    packed_extreme = nvfp4_cb_assemble_bytes(parent_extreme, k, grid=grid, mode=mode)
    packed_pooled = nvfp4_cb_assemble_bytes(parent_pooled, k, grid=grid, mode=mode)
    assert not torch.equal(packed_extreme, packed_pooled)

def test_opposing_masks_and_mixed():
    torch.manual_seed(911)
    E, C = 2, 256
    R_total = 8
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    w = (torch.randn(E, 8, C) * 0.6).bfloat16()
    acts = []
    for e in range(E):
        x = torch.randn(10, C)
        x[:, :32] *= 1 + 3 * e
        acts.append(x)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.opposing")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cw_gate = torch.stack([(fit.square().mean(0) + 1e-5).reshape(1, C) for fit in prepared.fit_rows])
    cw_up = torch.stack([((fit.square().mean(0) + 1e-5) * torch.linspace(0.2, 4.0, C)).reshape(1, C) for fit in prepared.fit_rows])
    leaf_cw = {"gate_proj": cw_gate, "up_proj": cw_up}
    cb12 = _explicit_codebook(12, "fp4", "product")
    parent12, gi12 = encode_packed_parent_leaf_local(w, 12, grid="fp4", mode="product", codebook=cb12, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    assert gi12["per_leaf_kept"]["gate_proj"] == [False, False]
    assert gi12["per_leaf_kept"]["up_proj"] == [False, True]
    o_parent12, o_kept12, _, _ = _manual_oracle(w, 12, "fp4", "product", cb12, True, SC2, "balanced", member_order, slice_boundaries, leaf_cw, validated)
    assert gi12["per_leaf_kept"] == o_kept12
    for kk in ("indices", "signs", "scales", "scale_super", "scale_sub"):
        if kk in parent12 or kk in o_parent12:
            assert torch.equal(parent12[kk], o_parent12[kk])
    assert torch.equal(nvfp4_cb_assemble_bytes(parent12, 12, grid="fp4", mode="product"), nvfp4_cb_assemble_bytes(o_parent12, 12, grid="fp4", mode="product"))
    cb13s = _explicit_codebook(13, "fp4", "signed")
    parent13s, gi13s = encode_packed_parent_leaf_local(w, 13, grid="fp4", mode="signed", codebook=cb13s, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    assert gi13s["per_leaf_kept"]["gate_proj"] == [False, False]
    assert gi13s["per_leaf_kept"]["up_proj"] == [True, False]
    o_parent13s, o_kept13s, _, _ = _manual_oracle(w, 13, "fp4", "signed", cb13s, True, SC2, "balanced", member_order, slice_boundaries, leaf_cw, validated)
    assert gi13s["per_leaf_kept"] == o_kept13s
    for kk in ("indices", "signs", "scales", "scale_super", "scale_sub"):
        if kk in parent13s or kk in o_parent13s:
            assert torch.equal(parent13s[kk], o_parent13s[kk])
    assert torch.equal(nvfp4_cb_assemble_bytes(parent13s, 13, grid="fp4", mode="signed"), nvfp4_cb_assemble_bytes(o_parent13s, 13, grid="fp4", mode="signed"))
def test_all_cold_raw():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=40)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = tuple(torch.empty((0, C), dtype=torch.float32) for _ in range(E))
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.cold")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    k = 12
    cb = _explicit_codebook(k, "fp4", "product")
    parent, gi = encode_packed_parent_leaf_local(weight, k, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    for leaf in member_order:
        assert gi["per_leaf_kept"][leaf] == [False] * E
    first = None
    # Compare every field via oracle
    o_parent, o_kept, _, _ = _manual_oracle(weight, k, "fp4", "product", cb, True, SC2, "balanced", member_order, slice_boundaries, leaf_cw, validated)
    for kk in ("indices", "signs", "scales", "scale_super", "scale_sub"):
        if kk in o_parent or kk in parent:
            assert kk in parent and kk in o_parent
            assert torch.equal(parent[kk], o_parent[kk]), f"all-cold {kk} mismatch"
    assert torch.equal(nvfp4_cb_assemble_bytes(parent, k, grid="fp4", mode="product"), nvfp4_cb_assemble_bytes(o_parent, k, grid="fp4", mode="product"))

def test_zero_candidate_error():
    E, R_total, C = 1, 8, 256
    member_order = ["down_proj"]
    slice_boundaries = {"down_proj": (0, 8)}
    weight = _make_weight(E, R_total, C, seed=50)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=51)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.zero")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    # Metric side effect: raw >0 then LDLQ exactly 0
    with patch("prismaquant.nvfp4_cb_formats.canonical_nvfp4_cb_single_output_mse", side_effect=[1.5, 0.0]):
        parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
        assert gi["raw_mse_per_expert_per_leaf"]["down_proj"][0] == 1.5
        assert gi["ldlq_mse_per_expert_per_leaf"]["down_proj"][0] == 0.0
        assert gi["per_leaf_kept"]["down_proj"][0] is True
    parent2, gi2 = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    for leaf in member_order:
        for v in gi2["per_leaf_kept"][leaf]:
            assert isinstance(v, bool)

def test_expert_major_ordering():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=60)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=61)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.order")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    o_parent, o_kept, _, _ = _manual_oracle(weight, 12, "fp4", "product", cb, True, SC2, "balanced", member_order, slice_boundaries, leaf_cw, validated)
    for kk in ("indices", "scales", "scale_super", "scale_sub"):
        if kk in parent:
            assert torch.equal(parent[kk], o_parent[kk])
    flat_indices = torch.cat([o_parent["indices"][0:4], o_parent["indices"][8:12], o_parent["indices"][4:8], o_parent["indices"][12:16]])
    assert not torch.equal(parent["indices"], flat_indices)

def test_single_leaf_regression():
    E, R_total, C = 2, 8, 256
    member_order = ["down_proj"]
    slice_boundaries = {"down_proj": (0, 8)}
    weight = _make_weight(E, R_total, C, seed=70)
    leaf_cw = {"down_proj": torch.rand((E, 1, C)) + 0.05}
    acts = _make_acts(E, C, N=16, seed=71)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.single")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    o_parent, o_kept, _, _ = _manual_oracle(weight, 12, "fp4", "product", cb, True, SC2, "balanced", member_order, slice_boundaries, leaf_cw, validated)
    for kk in ("indices", "scales", "scale_super", "scale_sub"):
        if kk in parent:
            assert torch.equal(parent[kk], o_parent[kk])

def test_factor_reuse_across_rungs():
    E, R_total, C = 3, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=80)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=81)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.reuse")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    import prismaquant.rotation_ldlq_pilot as pilot
    cnt = {"n": 0}
    orig = pilot.inverse_hessian_cholesky
    def counting(*a, **kw):
        cnt["n"] += 1
        return orig(*a, **kw)
    with patch.object(pilot, "inverse_hessian_cholesky", side_effect=counting):
        clear_ldlq_factor_cache()
        cb12 = _explicit_codebook(12, "fp4", "product")
        cb13 = _explicit_codebook(13, "fp4", "product")
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb12, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
        encode_packed_parent_leaf_local(weight, 13, grid="fp4", mode="product", codebook=cb13, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
        n = cnt["n"]
    clear_ldlq_factor_cache()
    assert n == E

@pytest.mark.parametrize("K,mode", [(12, "product"), (13, "signed")])
def test_peak_liveness_weakref(K, mode):
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=90)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=91)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.weak")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cb = _explicit_codebook(K, "fp4", mode)
    raw_call_count = {"n": 0}
    ldlq_call_count = {"n": 0}
    leaf1_refs = []
    seen_ids = set()
    orig_fields = nvfp4_cb_fields
    orig_ldlq = ldlq_reassign_cb_fields
    row_fields = ("indices", "signs", "scales", "scale_super", "scale_sub")
    def fields_wrapper(*args, **kw):
        idx = raw_call_count["n"]
        raw_call_count["n"] += 1
        if idx == 1:
            gc.collect()
            assert leaf1_refs, "weakref set empty before leaf-2 entry (no leaf-1 row tensors recorded)"
            alive = sum(1 for r in leaf1_refs if r() is not None)
            assert alive == 0, f"peak liveness violation: {alive} leaf-1 row tensors still alive at entry to leaf-2 raw encode (expected 0)"
        res = orig_fields(*args, **kw)
        if idx == 0:
            for kk in row_fields:
                if kk in res and isinstance(res[kk], torch.Tensor):
                    t = res[kk]
                    tid = id(t)
                    if tid not in seen_ids:
                        seen_ids.add(tid)
                        leaf1_refs.append(weakref.ref(t))
        return res
    def ldlq_wrapper(*args, **kw):
        res = orig_ldlq(*args, **kw)
        idx = ldlq_call_count["n"]
        ldlq_call_count["n"] += 1
        if idx == 0:
            for kk in row_fields:
                if kk in res and isinstance(res[kk], torch.Tensor):
                    t = res[kk]
                    tid = id(t)
                    if tid not in seen_ids:
                        seen_ids.add(tid)
                        leaf1_refs.append(weakref.ref(t))
        return res
    with patch("prismaquant.nvfp4_cb_formats.nvfp4_cb_fields", new=fields_wrapper):
        with patch("prismaquant.nvfp4_cb_formats.ldlq_reassign_cb_fields", new=ldlq_wrapper):
            parent, _ = encode_packed_parent_leaf_local(weight, K, grid="fp4", mode=mode, codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    assert raw_call_count["n"] == 2, f"raw leaf 2 checkpoint did not run exactly once: {raw_call_count['n']}"
    assert ldlq_call_count["n"] >= 1, f"LDLQ candidate wrapper did not run for leaf 1: {ldlq_call_count['n']}"
    assert leaf1_refs, "weakref set empty after leaf-1 (no unique row tensors recorded)"
    gc.collect()
    alive2 = sum(1 for r in leaf1_refs if r() is not None)
    assert alive2 == 0, f"leaf-1 tensors survived after helper return/GC: {alive2}"

def test_fp8_rejection_and_raw_regression():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=100)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=101)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.fp8")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    from prismaquant.cb_layout import family_for as _ff
    cb_fp8_dummy = tuple(fixed_lattice(7, "fp8", 2) for _ in range(4))
    with pytest.raises(ValueError, match="NVFP4 only|grid.*fp4"):
        encode_packed_parent_leaf_local(weight, 12, grid="fp8", mode="product", codebook=cb_fp8_dummy, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    cb12 = _explicit_codebook(12, "fp4", "product")
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb12, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    with pytest.raises(ValueError, match="NVFP4 only|grid.*fp4"):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp8", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0, 1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    w = ((((torch.arange(8 * 256, dtype=torch.int64) * 37 + 11) % 257) - 128).float() / 64).reshape(8, 256)
    fields = nvfp4_cb_fields(w, 36, grid="fp8", mode="product", codebook=None, scale_sweep=True, scale_coding="v1", encode_tier="balanced")
    p = nvfp4_cb_assemble_bytes(fields, 36, grid="fp8", mode="product")
    sha = hashlib.sha256(p.cpu().numpy().tobytes()).hexdigest()
    expected_sha = "4a348e61b9cd6456722d0cf5b13c76734010992905ce3503fd79c97f573b1a53"
    assert sha == expected_sha, f"FP8 raw regression SHA mismatch {sha} != {expected_sha}"
    f2 = nvfp4_cb_fields(w, 36, grid="fp8", mode="product", codebook=None, scale_sweep=True, scale_coding="v1", encode_tier="balanced")
    assert torch.equal(fields["indices"], f2["indices"])
    assert torch.equal(fields["scales"], f2["scales"])

def test_attestation_mutations():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    expert_ids = [0, 1]
    weight = _make_weight(E, R_total, C, seed=110)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=111)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.attest")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    qnames = {"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}
    base = canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=base)
    cb13 = _explicit_codebook(13, "fp4", "product")
    parent13, gi13 = encode_packed_parent_leaf_local(weight, 13, grid="fp4", mode="product", codebook=cb13, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    mutated = canonical_nvfp4_cb_payload_identity(parent13, 13, grid="fp4", mode="product", format_name="NVFP4_CB_K13", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi13["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    assert mutated["full_hash"] != base["full_hash"]
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=mutated)
    with pytest.raises(ValueError, match="gate_up_proj|physical_target"):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=base)
    bad_acts = _make_acts(E, C, N=16, seed=999)
    bad_prep = prepare_ldlq_gate_evidence(bad_acts, qname="test.bad")
    bad_validated = ValidatedPackedLDLQEvidence(bad_prep, E, C)
    mutated_ev = canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=bad_validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    assert mutated_ev["full_hash"] != base["full_hash"]
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=mutated_ev)
    bad_mask = {"gate_proj": [1, 0], "up_proj": [True, False]}
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=bad_mask, scale_sweep=True, encode_tier="balanced")
    bad_len = {"gate_proj": [True], "up_proj": [True, False]}
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=bad_len, scale_sweep=True, encode_tier="balanced")
    flipped = {k: list(v) for k, v in gi["per_leaf_kept"].items()}
    flipped["gate_proj"][0] = not flipped["gate_proj"][0]
    mutated_mask = canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=flipped, scale_sweep=True, encode_tier="balanced")
    assert mutated_mask["full_hash"] != base["full_hash"]
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=mutated_mask)
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept={"gate_proj": ["True", "False"], "up_proj": [True, False]}, scale_sweep=True, encode_tier="balanced")
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries={"gate_proj": (4, 8), "up_proj": (0, 4)}, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    bad_fields = dict(parent)
    bad_fields["indices"] = parent["indices"].clone()
    bad_fields["indices"][0, 0] += 1
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(bad_fields, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=base)
    bad_exp = dict(base); del bad_exp["full_hash"]
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=bad_exp)
    bad_exp2 = dict(base); bad_exp2["extra"] = 1
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=bad_exp2)
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=base["full_hash"])
    legacy = dict(base); legacy["abi"] = "canonical_nvfp4_cb_payload_identity.v1"
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=legacy)
    legacy2 = dict(base); legacy2["abi"] = "canonical_nvfp4_cb_payload_identity.v2"
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=legacy2)
    legacy3 = dict(base); legacy3["abi"] = "canonical_nvfp4_cb_payload_identity.v3"
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=legacy3)

def test_noncontiguous_expert_ids():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    expert_ids = [17, 42]
    weight = _make_weight(E, R_total, C, seed=120)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=121)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.noncontig")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    qnames = {"gate_proj": [f"model.layers.0.mlp.experts.{eid}.gate_proj" for eid in expert_ids], "up_proj": [f"model.layers.0.mlp.experts.{eid}.up_proj" for eid in expert_ids]}
    ident = canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=expert_ids, qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # Verify row hashes are bound to expert_id but storage is local ordinal
    swapped_ids = [42, 17]
    swapped_qnames = {"gate_proj": [f"model.layers.0.mlp.experts.{eid}.gate_proj" for eid in swapped_ids], "up_proj": [f"model.layers.0.mlp.experts.{eid}.up_proj" for eid in swapped_ids]}
    swapped_ident = canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=swapped_ids, qnames_per_leaf=swapped_qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    assert swapped_ident["full_hash"] != ident["full_hash"]
    assert swapped_ident["packed_digest"] == ident["packed_digest"]

def test_rung_mismatch_and_field_validation():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=130)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=131)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.rung")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cb12 = _explicit_codebook(12, "fp4", "product")
    parent12, gi12 = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb12, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent12, 13, grid="fp4", mode="product", format_name="NVFP4_CB_K13", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0, 1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi12["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    cb_signed = _explicit_codebook(13, "fp4", "signed")
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent12, 13, grid="fp4", mode="signed", format_name="NVFP4_CB_S13", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0, 1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi12["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    with pytest.raises(ValueError):
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb12, scale_sweep=True, scale_coding="v1", encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    base = canonical_nvfp4_cb_payload_identity(parent12, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0, 1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi12["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    parent_mut = dict(parent12)
    cb_mut = tuple(t.clone() for t in cb12)
    cb_mut[0][0, 0] += 1.0
    parent_mut["codebook"] = cb_mut
    ident_mut = canonical_nvfp4_cb_payload_identity(parent_mut, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0, 1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi12["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    assert ident_mut["full_hash"] != base["full_hash"]
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent_mut, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0, 1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi12["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=base)
    bad = dict(parent12); bad["indices"] = parent12["indices"].clone(); bad["indices"][0, 0, 0] += 1
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(bad, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0, 1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi12["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=base)

def test_full_mode_rejection():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=140)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=141)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.full")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    dummy_cb = torch.randn((4096, 8))
    with pytest.raises(ValueError, match="full rejected|product.*signed"):
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="full", codebook=dummy_cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)

def test_exact_types_and_coercion_rejection():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=150)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=151)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.coerce")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    with pytest.raises(ValueError):
        encode_packed_parent_leaf_local(weight, "12", grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    with pytest.raises(ValueError):
        encode_packed_parent_leaf_local(weight, 12.0, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    with pytest.raises(ValueError):
        encode_packed_parent_leaf_local(weight, True, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    with pytest.raises(ValueError):
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=None, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    with pytest.raises(ValueError):
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated, warm_state={"scales": torch.zeros(1)})

def test_validation_cost_behavior():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=160)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=161)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.cost")
    import prismaquant.nvfp4_cb_formats as fmt_mod
    orig_digest = fmt_mod._partition_multiset_digest
    cnt = {"n": 0}
    def counting(t):
        cnt["n"] += 1
        return orig_digest(t)
    with patch.object(fmt_mod, "_partition_multiset_digest", side_effect=counting):
        validated = ValidatedPackedLDLQEvidence(prepared, E, C)
        n_after_construct = cnt["n"]
        cb12 = _explicit_codebook(12, "fp4", "product")
        cb13 = _explicit_codebook(13, "fp4", "product")
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb12, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
        encode_packed_parent_leaf_local(weight, 13, grid="fp4", mode="product", codebook=cb13, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
        parent12, gi12 = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb12, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
        qnames = {"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}
        ident = canonical_nvfp4_cb_payload_identity(parent12, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0, 1], qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi12["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
        verify_canonical_payload_identity(parent12, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0, 1], qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi12["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=ident)
        n_after_ops = cnt["n"]
    assert n_after_ops == n_after_construct
    # In-place mutation should fail cheaply via fingerprint
    priv = object.__getattribute__(validated, "_priv_prepared")
    # Mutate private FIT tensor via ordinary in-place
    t = priv.fit_rows[0]
    t[0, 0] = t[0, 0] + 1.0
    with pytest.raises(ValueError, match="fingerprint"):
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb12, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)

def test_adversarial_probes_immutability_and_validation():
    E, C = 2, 256
    R_total = 8
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    acts = _make_acts(E, C, N=16, seed=200)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.adv")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    # 1: mutable after construction should fail
    with pytest.raises(AttributeError):
        validated._policy = "attacker_policy"
    with pytest.raises((TypeError, AttributeError, ValueError)):
        validated.split_infos[0]["attacker_injected"] = "yes"
    # Mutating source tensor via .data should be caught, but mutating original caller source should not affect validated
    orig_fit = prepared.fit_rows[0].clone()
    # Mutate original source (caller-owned) - should not affect validated private
    prepared.fit_rows[0].data[0, 0] = 9999.0
    # Validated should still pass fingerprint (since it cloned)
    validated._check_fingerprints()
    # Restore for next check
    prepared.fit_rows[0].data.copy_(orig_fit.data)
    # 2: topology validation fail-open checks
    import dataclasses
    # eligible contains 999
    bad_prep = dataclasses.replace(prepared, eligible_experts=(999,))
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad_prep, E, C)
    bad_prep2 = dataclasses.replace(prepared, eligible_experts=(0,), cold_experts=(0,))
    # Actually need to create a case where 0 in both eligible and cold
    # Use a prepared where we manually craft overlapping
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(dataclasses.replace(prepared, eligible_experts=(0,), cold_experts=(0,), insufficient_experts=()), E, C)
    # 3: physical validation fail-open branches
    weight = _make_weight(E, R_total, C, seed=210)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    cb12 = _explicit_codebook(12, "fp4", "product")
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb12, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    # NaN codebook
    bad_cb = tuple(t.clone() for t in cb12)
    bad_cb[0][0, 0] = float("nan")
    bad_parent = dict(parent); bad_parent["codebook"] = bad_cb
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(bad_parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj","model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj","model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # int32 codebook
    bad_cb2 = tuple(t.clone().to(torch.int32) for t in cb12)
    bad_parent2 = dict(parent); bad_parent2["codebook"] = bad_cb2
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(bad_parent2, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj","model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj","model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # uint8 indices
    bad_parent3 = dict(parent); bad_parent3["indices"] = parent["indices"].to(torch.uint8)
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(bad_parent3, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj","model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj","model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # scales reshaped
    bad_parent4 = dict(parent); bad_parent4["scales"] = parent["scales"].reshape(E*R_total, C//32, 2)
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(bad_parent4, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj","model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj","model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # float16 scale_sub
    bad_parent5 = dict(parent); bad_parent5["scale_sub"] = parent["scale_sub"].to(torch.float16)
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(bad_parent5, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj","model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj","model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # 4: ABI v3 collision - view as int32
    cb_view = tuple(t.clone().view(torch.int32) for t in cb12)
    # Physical validation should reject int32
    bad_parent6 = dict(parent); bad_parent6["codebook"] = cb_view
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(bad_parent6, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj","model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj","model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # Also check digest differs if it were to pass (framing)
    # 5: verifier type confusable
    base = canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj","model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj","model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # bool/int confusion: per_leaf_kept with 0/1 ints should be rejected at construction, verifier should also reject if somehow passed
    bad_expected = json.loads(json.dumps(base))
    bad_expected["per_leaf_kept"] = {"gate_proj": [0, 1], "up_proj": [1, 0]}
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj","model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj","model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=bad_expected)
    bad_expected2 = json.loads(json.dumps(base))
    bad_expected2["expert_ids"] = [False, True]
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj","model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj","model.layers.0.mlp.experts.1.up_proj"]}, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=bad_expected2)

def test_row_hash_composability():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=200)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=201)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.row")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    qnames = {"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}
    base = canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # Build E=1 parent solely by slicing every row field from rows [R_total:2*R_total], retaining globals
    parent1 = {}
    for kk, vv in parent.items():
        if kk in ("indices", "signs", "scales", "scale_super", "scale_sub") and isinstance(vv, torch.Tensor):
            parent1[kk] = vv[R_total:2*R_total].clone()
        elif kk in ("codebook", "scale_coding"):
            parent1[kk] = vv
        elif kk == "shape":
            continue
        else:
            parent1[kk] = vv
    parent1["shape"] = (1, R_total, C)
    # Build E=1 Prepared by slicing expert-1 evidence and remapping split-info local expert to 0
    orig_priv = object.__getattribute__(validated, "_priv_prepared")
    info1_dict = dict(orig_priv.split_infos[1])
    info1_dict["expert"] = 0
    info1_tuple = (info1_dict,)
    digest1 = _canonical_split_infos_digest(info1_tuple)
    # map sets: if original had 1 in cold/insufficient/eligible, E1 has 0; else empty
    cold1 = (0,) if 1 in orig_priv.cold_experts else tuple()
    insuf1 = (0,) if 1 in orig_priv.insufficient_experts else tuple()
    elig1 = (0,) if 1 in orig_priv.eligible_experts else tuple()
    fit_missing1 = (0,) if 1 in orig_priv.fit_missing_pooled else tuple()
    has_obs1 = bool(elig1)
    prep1 = PreparedLDLQEvidence(
        original_rows=(orig_priv.original_rows[1].clone(),),
        fit_rows=(orig_priv.fit_rows[1].clone(),),
        holdout_rows=(orig_priv.holdout_rows[1].clone(),),
        fit_filled=(orig_priv.fit_filled[1].clone(),),
        split_infos=info1_tuple,
        cold_experts=cold1,
        insufficient_experts=insuf1,
        eligible_experts=elig1,
        fit_missing_pooled=fit_missing1,
        has_observed_fit=has_obs1,
        policy=orig_priv.policy,
        version=orig_priv.version,
        original_digests=(orig_priv.original_digests[1],),
        fit_digests=(orig_priv.fit_digests[1],),
        holdout_digests=(orig_priv.holdout_digests[1],),
        split_infos_digest=digest1,
    )
    validated1 = ValidatedPackedLDLQEvidence(prep1, 1, C)
    qnames1 = {"gate_proj": ["model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.1.up_proj"]}
    per_leaf_kept1 = {leaf: [gi["per_leaf_kept"][leaf][1]] for leaf in member_order}
    ident1 = canonical_nvfp4_cb_payload_identity(parent1, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[1], qnames_per_leaf=qnames1, prepared=validated1, per_leaf_kept=per_leaf_kept1, scale_sweep=True, encode_tier="balanced")
    # Assembled bytes equality
    packed = nvfp4_cb_assemble_bytes(parent, 12, grid="fp4", mode="product")
    packed1 = nvfp4_cb_assemble_bytes(parent1, 12, grid="fp4", mode="product")
    assert torch.equal(packed1, packed[R_total:2*R_total])
    # Row hashes composability
    assert ident1["row_hashes"]["gate_proj"][0] == base["row_hashes"]["gate_proj"][1]
    assert ident1["row_hashes"]["up_proj"][0] == base["row_hashes"]["up_proj"][1]
    # Prove row identity changes/rejects when decode authorities change
    # codebook change
    cb_mut = tuple(t.clone() for t in cb)
    cb_mut[0][0,0] += 1.0
    parent_mut = dict(parent1); parent_mut["codebook"] = cb_mut
    ident_cb = canonical_nvfp4_cb_payload_identity(parent_mut, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[1], qnames_per_leaf=qnames1, prepared=validated1, per_leaf_kept=per_leaf_kept1, scale_sweep=True, encode_tier="balanced")
    assert ident_cb["row_hashes"]["gate_proj"][0] != base["row_hashes"]["gate_proj"][1]
    # scale_sweep change
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent1, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[1], qnames_per_leaf=qnames1, prepared=validated1, per_leaf_kept=per_leaf_kept1, scale_sweep=False, encode_tier="balanced")
    # encode_tier change
    ident_tier = canonical_nvfp4_cb_payload_identity(parent1, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[1], qnames_per_leaf=qnames1, prepared=validated1, per_leaf_kept=per_leaf_kept1, scale_sweep=True, encode_tier="fast")
    assert ident_tier["row_hashes"]["gate_proj"][0] != base["row_hashes"]["gate_proj"][1]
    # qname leaf swap
    qnames_swap = {"gate_proj": ["model.layers.0.mlp.experts.1.up_proj"], "up_proj": ["model.layers.0.mlp.experts.1.gate_proj"]}
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent1, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[1], qnames_per_leaf=qnames_swap, prepared=validated1, per_leaf_kept=per_leaf_kept1, scale_sweep=True, encode_tier="balanced")
    # physical expert ID change
    ident_eid = canonical_nvfp4_cb_payload_identity(parent1, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[99], qnames_per_leaf={"gate_proj": ["model.layers.0.mlp.experts.99.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.99.up_proj"]}, prepared=validated1, per_leaf_kept=per_leaf_kept1, scale_sweep=True, encode_tier="balanced")
    assert ident_eid["row_hashes"]["gate_proj"][0] != base["row_hashes"]["gate_proj"][1]
    # physical target change must be rejected (strict gate_up_proj)
    with pytest.raises(ValueError, match="gate_up_proj|physical_target"):
        canonical_nvfp4_cb_payload_identity(parent1, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj_changed", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[1], qnames_per_leaf=qnames1, prepared=validated1, per_leaf_kept=per_leaf_kept1, scale_sweep=True, encode_tier="balanced")

def test_no_implicit_authority_and_tier_alias_rejection():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=300)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=301)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.authority")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    qnames = {"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}
    # Missing scale_sweep raises TypeError
    with pytest.raises(TypeError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], encode_tier="balanced")
    with pytest.raises(TypeError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True)
    with pytest.raises(TypeError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], expected=canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced"), encode_tier="balanced")
    with pytest.raises(TypeError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], expected=canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced"), scale_sweep=True)
    # Tier aliases must be rejected, not normalized
    for alias in [" balanced", "balanced ", "BALANCED"]:
        with pytest.raises(ValueError):
            canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier=alias)
        with pytest.raises(ValueError):
            encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier=alias, member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    # scale_sweep must be exact True
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=False, encode_tier="balanced")

def test_mixed_cold_eligible_topology():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=310)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    # Cold expert 0 empty, eligible expert 1 with sufficient varied rows
    acts_cold = torch.empty((0, C), dtype=torch.float32)
    g = torch.Generator(); g.manual_seed(311)
    acts_elig = torch.randn((16, C), generator=g, dtype=torch.float32)
    # Ensure eligible has at least 2 unique row contents
    acts_elig[0] = acts_elig[1] + 1.0
    acts = (acts_cold, acts_elig)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.mixed")
    assert set(prepared.cold_experts) == {0}
    assert set(prepared.eligible_experts) == {1}
    assert set(prepared.insufficient_experts) == set()
    assert prepared.cold_experts + prepared.insufficient_experts + prepared.eligible_experts == tuple(sorted(set(prepared.cold_experts) | set(prepared.insufficient_experts) | set(prepared.eligible_experts)))
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    for leaf in member_order:
        assert gi["per_leaf_kept"][leaf][0] is False
    # Exactly one factor for eligible expert
    import prismaquant.nvfp4_cb_formats as fmt
    cnt = {"n": 0}
    orig = fmt._partition_multiset_digest
    # Use factor cache count instead
    import prismaquant.rotation_ldlq_pilot as pilot
    fcnt = {"n": 0}
    orig_chol = pilot.inverse_hessian_cholesky
    def counting(*a, **kw):
        fcnt["n"] += 1
        return orig_chol(*a, **kw)
    from unittest.mock import patch as _patch
    with _patch.object(pilot, "inverse_hessian_cholesky", side_effect=counting):
        clear_ldlq_factor_cache()
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
        assert fcnt["n"] == 1
    clear_ldlq_factor_cache()
    # Every field and byte vs oracle
    o_parent, o_kept, _, _ = _manual_oracle(weight, 12, "fp4", "product", cb, True, SC2, "balanced", member_order, slice_boundaries, leaf_cw, validated)
    for kk in ("indices", "scales", "scale_super", "scale_sub"):
        if kk in o_parent:
            assert torch.equal(parent[kk], o_parent[kk])
    assert torch.equal(nvfp4_cb_assemble_bytes(parent, 12, grid="fp4", mode="product"), nvfp4_cb_assemble_bytes(o_parent, 12, grid="fp4", mode="product"))
    assert gi["per_leaf_kept"] == o_kept

def test_constructor_heavy_adversaries():
    E, C = 2, 256
    R_total = 8
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    import dataclasses
    # Helper to make valid prepared
    def _valid():
        acts = _make_acts(E, C, N=16, seed=400)
        prep = prepare_ldlq_gate_evidence(acts, qname="test.adv2")
        return prep
    # integer evidence including empty integer tensors
    prep = _valid()
    int_rows = tuple(t.to(torch.int64) if t.numel()>0 else torch.empty((0, C), dtype=torch.int64) for t in prep.original_rows)
    bad = dataclasses.replace(prep, original_rows=int_rows, original_digests=tuple(_partition_multiset_digest(t) for t in int_rows))
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # digest != fit_digest
    acts_insuf = (torch.empty((0, C), dtype=torch.float32), torch.randn((16, C), generator=torch.Generator().manual_seed(401)))
    prep_insuf = prepare_ldlq_gate_evidence(acts_insuf, qname="test.adv2.insuf")
    bad_infos = [dict(s) for s in prep_insuf.split_infos]
    for s in bad_infos:
        if s["insufficient"]:
            s["digest"] = "0"*64
            break
    bad = dataclasses.replace(prep_insuf, split_infos=tuple(bad_infos), split_infos_digest=_canonical_split_infos_digest(tuple(bad_infos)))
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # split insufficient contradicting category
    prep = _valid()
    bad_infos = [dict(s) for s in prep.split_infos]
    # flip insufficient for first expert
    bad_infos[0]["insufficient"] = not bad_infos[0]["insufficient"]
    if bad_infos[0]["insufficient"]:
        bad_infos[0].pop("unique_hashes", None)
        bad_infos[0]["digest"] = bad_infos[0]["fit_digest"]
    else:
        bad_infos[0].pop("digest", None)
        bad_infos[0]["unique_hashes"] = 2
    bad = dataclasses.replace(prep, split_infos=tuple(bad_infos), split_infos_digest=_canonical_split_infos_digest(tuple(bad_infos)))
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # eligible n_fit+n_holdout != n_full — tensor counts and metadata internally consistent so it reaches sum gate
    prep = _valid()
    # Find eligible index deterministically
    elig_idx = None
    for i, s in enumerate(prep.split_infos):
        if not s["insufficient"]:
            elig_idx = i
            break
    assert elig_idx is not None, "eligible expert must exist for n_fit sum gate proof (N=16 guarantees)"
    new_fit = torch.cat([prep.fit_rows[elig_idx], prep.fit_rows[elig_idx][:1].clone()], dim=0)
    new_fit_digest = _partition_multiset_digest(new_fit)
    new_fit_rows = list(prep.fit_rows)
    new_fit_rows[elig_idx] = new_fit
    new_fit_rows_t = tuple(new_fit_rows)
    new_fit_filled = list(prep.fit_filled)
    new_fit_filled[elig_idx] = new_fit
    new_infos = [dict(s) for s in prep.split_infos]
    new_infos[elig_idx]["n_fit"] = int(new_fit.shape[0])
    new_infos[elig_idx]["fit_digest"] = new_fit_digest
    bad = dataclasses.replace(prep, fit_rows=new_fit_rows_t, fit_filled=tuple(new_fit_filled), fit_digests=tuple(_partition_multiset_digest(t) for t in new_fit_rows_t), split_infos=tuple(new_infos), split_infos_digest=_canonical_split_infos_digest(tuple(new_infos)))
    with pytest.raises(ValueError, match="n_fit.*n_holdout.*n_full"):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # altered FIT/HOLDOUT whose union no longer equals original
    prep = _valid()
    assert elig_idx is not None, "eligible expert must exist for union mismatch proof"
    new_fit = prep.fit_rows[elig_idx].clone()
    if new_fit.numel()>0:
        new_fit[0,0] += 1.0
    new_fit_digest = _partition_multiset_digest(new_fit)
    new_fit_rows = list(prep.fit_rows)
    new_fit_rows[elig_idx] = new_fit
    new_fit_rows_t = tuple(new_fit_rows)
    new_infos = [dict(s) for s in prep.split_infos]
    new_infos[elig_idx]["fit_digest"] = new_fit_digest
    new_infos[elig_idx]["n_fit"] = int(new_fit.shape[0])
    bad = dataclasses.replace(prep, fit_rows=new_fit_rows_t, fit_digests=tuple(_partition_multiset_digest(t) for t in new_fit_rows_t), fit_filled=new_fit_rows_t, split_infos=tuple(new_infos), split_infos_digest=_canonical_split_infos_digest(tuple(new_infos)))
    with pytest.raises(ValueError, match="FIT.*HOLDOUT.*original|multiset"):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # duplicate row content crossing FIT/HOLDOUT — isolated adversary that passes preceding invariants
    # Isolated duplicate-crossing adversary that passes every preceding invariant (counts balance, multiset union == original)
    assert elig_idx is not None, "eligible expert must exist for duplicate crossing proof"
    row_a = torch.full((1, C), 1.0, dtype=torch.float32)
    row_b = torch.full((1, C), 2.0, dtype=torch.float32)
    row_c = torch.full((1, C), 3.0, dtype=torch.float32)
    new_fit = torch.cat([row_a, row_b], dim=0)
    new_hold = torch.cat([row_b, row_c], dim=0)
    new_orig = torch.cat([new_fit, new_hold], dim=0)
    new_orig_digest = _partition_multiset_digest(new_orig)
    new_fit_digest = _partition_multiset_digest(new_fit)
    new_hold_digest = _partition_multiset_digest(new_hold)
    new_orig_rows = list(prep.original_rows); new_orig_rows[elig_idx] = new_orig
    new_fit_rows = list(prep.fit_rows); new_fit_rows[elig_idx] = new_fit
    new_hold_rows = list(prep.holdout_rows); new_hold_rows[elig_idx] = new_hold
    new_filled = list(prep.fit_filled); new_filled[elig_idx] = new_fit
    new_infos = [dict(s) for s in prep.split_infos]
    new_infos[elig_idx]["n_full"] = int(new_orig.shape[0])
    new_infos[elig_idx]["n_fit"] = int(new_fit.shape[0])
    new_infos[elig_idx]["n_holdout"] = int(new_hold.shape[0])
    new_infos[elig_idx]["full_digest"] = new_orig_digest
    new_infos[elig_idx]["fit_digest"] = new_fit_digest
    new_infos[elig_idx]["holdout_digest"] = new_hold_digest
    uniq = len(set(_row_content_hash(new_orig[i]) for i in range(new_orig.shape[0])))
    new_infos[elig_idx]["unique_hashes"] = uniq
    bad = dataclasses.replace(prep, original_rows=tuple(new_orig_rows), fit_rows=tuple(new_fit_rows), holdout_rows=tuple(new_hold_rows), fit_filled=tuple(new_filled), original_digests=tuple(_partition_multiset_digest(t) for t in new_orig_rows), fit_digests=tuple(_partition_multiset_digest(t) for t in new_fit_rows), holdout_digests=tuple(_partition_multiset_digest(t) for t in new_hold_rows), split_infos=tuple(new_infos), split_infos_digest=_canonical_split_infos_digest(tuple(new_infos)))
    with pytest.raises(ValueError, match="duplicate row content crosses FIT/HOLDOUT"):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # wrong unique_hashes
    prep = _valid()
    assert elig_idx is not None
    bad_infos = [dict(s) for s in prep.split_infos]
    bad_infos[elig_idx]["unique_hashes"] = 999
    bad = dataclasses.replace(prep, split_infos=tuple(bad_infos), split_infos_digest=_canonical_split_infos_digest(tuple(bad_infos)))
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # rank-1 empty
    prep = _valid()
    bad_rows = list(prep.original_rows); bad_rows[0] = torch.empty((0,), dtype=torch.float32)
    bad = dataclasses.replace(prep, original_rows=tuple(bad_rows))
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # (0,wrong_C)
    prep = _valid()
    bad_rows = list(prep.original_rows); bad_rows[0] = torch.empty((0, 123), dtype=torch.float32)
    bad = dataclasses.replace(prep, original_rows=tuple(bad_rows))
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # bool/string/missing expert — truly missing key by removal, not None
    for bad_val in [True, "0"]:
        prep = _valid()
        bad_infos = [dict(s) for s in prep.split_infos]
        bad_infos[0]["expert"] = bad_val
        bad = dataclasses.replace(prep, split_infos=tuple(bad_infos), split_infos_digest=_canonical_split_infos_digest(tuple(bad_infos)))
        with pytest.raises(ValueError, match="expert.*int|exact int"):
            ValidatedPackedLDLQEvidence(bad, E, C)
    prep = _valid()
    bad_infos = [dict(s) for s in prep.split_infos]
    del bad_infos[0]["expert"]
    bad = dataclasses.replace(prep, split_infos=tuple(bad_infos), split_infos_digest=_canonical_split_infos_digest(tuple(bad_infos)))
    with pytest.raises(ValueError, match="expert.*missing|expert"):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # wrong counts
    prep = _valid()
    bad_infos = [dict(s) for s in prep.split_infos]
    bad_infos[0]["n_full"] = 999
    bad = dataclasses.replace(prep, split_infos=tuple(bad_infos), split_infos_digest=_canonical_split_infos_digest(tuple(bad_infos)))
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # nested/extra split key
    prep = _valid()
    bad_infos = [dict(s) for s in prep.split_infos]
    bad_infos[0]["extra"] = 1
    bad = dataclasses.replace(prep, split_infos=tuple(bad_infos), split_infos_digest=_canonical_split_infos_digest(tuple(bad_infos)))
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad, E, C)
    prep = _valid()
    bad_infos = [dict(s) for s in prep.split_infos]
    bad_infos[0]["nested"] = {"a":1}
    bad = dataclasses.replace(prep, split_infos=tuple(bad_infos), split_infos_digest=_canonical_split_infos_digest(tuple(bad_infos)))
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # inconsistent has_observed_fit/fit_missing
    prep = _valid()
    bad = dataclasses.replace(prep, has_observed_fit=not prep.has_observed_fit)
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad, E, C)
    prep = _valid()
    bad = dataclasses.replace(prep, fit_missing_pooled=tuple(sorted(set(prep.fit_missing_pooled) | {0})) if 0 not in prep.fit_missing_pooled else tuple(sorted(set(prep.fit_missing_pooled) - {0})))
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # eligible out of range/overlap
    prep = _valid()
    bad = dataclasses.replace(prep, eligible_experts=(999,))
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad, E, C)
    prep = _valid()
    bad = dataclasses.replace(prep, eligible_experts=(0,), cold_experts=(0,))
    with pytest.raises(ValueError):
        ValidatedPackedLDLQEvidence(bad, E, C)

def test_authority_topology_physical_verifier_adversaries():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=500)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=501)
    prepared = prepare_ldlq_gate_evidence(acts, qname="test.auth")
    validated = ValidatedPackedLDLQEvidence(prepared, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    # non-string/empty/duplicate member names
    with pytest.raises(ValueError):
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=["", "up_proj"], slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    with pytest.raises(ValueError):
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=[123, "up_proj"], slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    with pytest.raises(ValueError):
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=["gate_proj", "gate_proj"], slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=validated)
    # qname gate/up swap and false expert token
    qnames_ok = {"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}
    qnames_swap = {"gate_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"], "up_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"]}
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_swap, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    qnames_false = {"gate_proj": ["model.layers.0.mlp.experts10.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_false, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # tuple/list schema substitution in expected identity
    base = canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    bad_exp = json.loads(json.dumps(base))
    bad_exp["shape"] = tuple(bad_exp["shape"])
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=bad_exp)
    # bool/int confusables
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[True, False], qnames_per_leaf=qnames_ok, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=validated, per_leaf_kept={"gate_proj": [1, 0], "up_proj": [1, 0]}, scale_sweep=True, encode_tier="balanced")
    # legacy ABI
    legacy = dict(base); legacy["abi"] = "canonical_nvfp4_cb_payload_identity.v3"
    with pytest.raises(ValueError):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=legacy)
    # illegal super/sub — deterministic via _two_tier_tables legality authority
    from prismaquant.nvfp4_cb_formats import _two_tier_tables
    _, _, legal = _two_tier_tables(str(parent["scale_super"].device))
    illegal_super = illegal_sub = None
    for s in range(256):
        for c in range(16):
            if not bool(legal[s, c]):
                illegal_super, illegal_sub = s, c
                break
        if illegal_super is not None:
            break
    assert illegal_super is not None, "no illegal super/sub pair found in legality table"
    assert legal[illegal_super, illegal_sub] == False, f"chosen pair ({illegal_super},{illegal_sub}) must be illegal"
    bad = dict(parent)
    bad["scale_super"] = parent["scale_super"].clone()
    bad["scale_sub"] = parent["scale_sub"].clone()
    bad["scale_super"][0, 0] = illegal_super
    bad["scale_sub"][0, 0] = illegal_sub
    with pytest.raises(ValueError, match="illegal.*super.*sub|super.*sub.*illegal|illegal.*pair"):
        canonical_nvfp4_cb_payload_identity(bad, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # perturbed scales
    bad2 = dict(parent); bad2["scales"] = parent["scales"] + 0.1
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(bad2, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # bool/uint8/out-of-range indices, invalid signs, wrong ranks/dtypes, NaN/int32-view, K12 mislabeled K13
    bad3 = dict(parent); bad3["indices"] = parent["indices"].to(torch.bool)
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(bad3, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    bad4 = dict(parent); bad4["indices"] = parent["indices"].clone(); bad4["indices"][0,0,0] = 9999
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(bad4, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K13", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=validated, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")

def test_shuffled_category_tuples_rejected():
    E, C = 2, 256
    acts = _make_acts(E, C, N=16, seed=601)
    prep = prepare_ldlq_gate_evidence(acts, qname="test.shuffle")
    import dataclasses
    # All-eligible (1,0) shuffled must be rejected
    bad = dataclasses.replace(prep, eligible_experts=(1, 0), cold_experts=(), insufficient_experts=(), fit_missing_pooled=(), has_observed_fit=True)
    with pytest.raises(ValueError, match="canonical increasing order|shuffled"):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # Cold shuffled
    prep2 = prepare_ldlq_gate_evidence((torch.empty((0, C), dtype=torch.float32), torch.randn((16, C), generator=torch.Generator().manual_seed(602))), qname="test.coldshuffle")
    # prep2 should have cold (0) eligible (1)
    assert 0 in prep2.cold_experts and 1 in prep2.eligible_experts
    # Craft shuffled cold+insufficient etc — for this case insufficient empty, cold is (0,) already sorted, but shuffle eligible (1,0) already tested; try shuffled fit_missing
    bad2 = dataclasses.replace(prep, fit_missing_pooled=(1, 0) if len(prep.fit_missing_pooled)==2 else (0,))
    # Only test if original had 2 missing; otherwise skip
    if len(prep.fit_missing_pooled) == 2:
        with pytest.raises(ValueError, match="canonical increasing order"):
            ValidatedPackedLDLQEvidence(bad2, E, C)

def test_exact_json_native_rejects_subclasses():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=610)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=611)
    prep = prepare_ldlq_gate_evidence(acts, qname="test.json")
    val = ValidatedPackedLDLQEvidence(prep, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=val)
    qnames_ok = {"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}
    base = canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=val, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # dict subclass at top level
    class MyDict(dict): pass
    bad = MyDict(base)
    with pytest.raises(ValueError, match="exact dict"):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=val, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=bad)
    # nested list subclass
    class MyList(list): pass
    bad2 = json.loads(json.dumps(base))
    bad2["expert_ids"] = MyList(bad2["expert_ids"])
    with pytest.raises(ValueError, match="list|exact"):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=val, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=bad2)
    # custom Mapping
    from collections.abc import Mapping
    class CustomMapping(Mapping):
        def __init__(self, d): self._d = d
        def __getitem__(self, k): return self._d[k]
        def __iter__(self): return iter(self._d)
        def __len__(self): return len(self._d)
    bad3 = CustomMapping(base)
    with pytest.raises(ValueError, match="exact dict|Mapping"):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=val, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=bad3)
    # tuple instead of list
    bad4 = json.loads(json.dumps(base))
    bad4["expert_ids"] = tuple(bad4["expert_ids"])
    with pytest.raises(ValueError, match="tuple"):
        verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=val, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=bad4)
    # real json round trip still verifies
    rt = json.loads(json.dumps(base))
    verify_canonical_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_ok, prepared=val, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced", expected=rt)

def test_qname_one_experts_and_duplicate_rejected():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=620)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=621)
    prep = prepare_ldlq_gate_evidence(acts, qname="test.qname")
    val = ValidatedPackedLDLQEvidence(prep, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=val)
    # multiple experts component in one qname
    qnames_multi = {"gate_proj": ["model.layers.0.mlp.experts.0.experts.1.gate_proj", "model.layers.0.mlp.experts.1.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}
    with pytest.raises(ValueError, match="exactly one.*experts"):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_multi, prepared=val, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    # duplicate qname within leaf
    qnames_dup = {"gate_proj": ["model.layers.0.mlp.experts.0.gate_proj", "model.layers.0.mlp.experts.0.gate_proj"], "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}
    with pytest.raises(ValueError, match="unique"):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_dup, prepared=val, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")

def test_bf16_required():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=630)
    prep = prepare_ldlq_gate_evidence(acts, qname="test.bf16")
    val = ValidatedPackedLDLQEvidence(prep, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    # BF16 success
    w_bf16 = _make_weight(E, R_total, C, seed=630)
    assert w_bf16.dtype == torch.bfloat16
    parent, _ = encode_packed_parent_leaf_local(w_bf16, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=val)
    # float32 rejected
    w_f32 = w_bf16.to(torch.float32)
    with pytest.raises(ValueError, match="bfloat16"):
        encode_packed_parent_leaf_local(w_f32, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=val)
    w_f16 = w_bf16.to(torch.float16)
    with pytest.raises(ValueError, match="bfloat16"):
        encode_packed_parent_leaf_local(w_f16, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=val)
    # generic low-level research codec still FP8-capable (nvfp4_cb_fields with float32)
    w = torch.randn(8, 256, dtype=torch.float32)
    fields = nvfp4_cb_fields(w, 36, grid="fp8", mode="product", codebook=None, scale_sweep=True, scale_coding="v1", encode_tier="balanced")
    assert fields is not None

def test_product_codebook_list_rejected_and_qname_sequence_rejected():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=640)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=641)
    prep = prepare_ldlq_gate_evidence(acts, qname="test.listcb")
    val = ValidatedPackedLDLQEvidence(prep, E, C)
    cb_tuple = _explicit_codebook(12, "fp4", "product")
    cb_list = list(cb_tuple)
    with pytest.raises(ValueError, match="exact tuple"):
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb_list, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=val)
    # tuple subclass rejected
    class MyTuple(tuple): pass
    cb_sub = MyTuple(cb_tuple)
    with pytest.raises(ValueError, match="exact tuple"):
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb_sub, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=val)
    # qname non-sequence and string-as-sequence rejected
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb_tuple, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=val)
    qnames_bad_seq = {"gate_proj": "model.layers.0.mlp.experts.0.gate_proj", "up_proj": ["model.layers.0.mlp.experts.0.up_proj", "model.layers.0.mlp.experts.1.up_proj"]}
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_bad_seq, prepared=val, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")
    qnames_str_seq = {"gate_proj": "model.layers.0.mlp.experts.0.gate_proj", "up_proj": "model.layers.0.mlp.experts.1.up_proj"}
    # string value will be treated as non-list and fail length check
    with pytest.raises(ValueError):
        canonical_nvfp4_cb_payload_identity(parent, 12, grid="fp4", mode="product", format_name="NVFP4_CB_K12", physical_target="model.layers.0.mlp.experts.gate_up_proj", member_order=member_order, slice_boundaries=slice_boundaries, expert_ids=[0,1], qnames_per_leaf=qnames_str_seq, prepared=val, per_leaf_kept=gi["per_leaf_kept"], scale_sweep=True, encode_tier="balanced")

def test_fp8_ldlq_pack_guard():
    w = torch.randn(8, 256, dtype=torch.bfloat16)
    cb_dummy = _explicit_codebook(12, "fp4", "product")
    # raw FP8 pack must remain bit-for-bit
    from unittest.mock import patch
    with patch("prismaquant.nvfp4_cb_formats.ldlq_reassign_cb_fields") as mock_ldlq:
        mock_ldlq.side_effect = AssertionError("FP8+LDLQ should reject before ldlq_reassign_cb_fields")
        import prismaquant.nvfp4_cb_formats as fmt
        with pytest.raises(ValueError, match="grid.*fp4|FP8\\+LDLQ"):
            fmt.nvfp4_cb_pack(w, 12, grid="fp8", mode="product", col_weights=torch.rand(8, 256)+0.05, codebook=None, ldlq=True, activation_rows=torch.randn(16, 256))
        assert mock_ldlq.call_count == 0
    # FP8 raw still works
    import hashlib
    w2 = ((((torch.arange(8 * 256, dtype=torch.int64) * 37 + 11) % 257) - 128).float() / 64).reshape(8, 256)
    fields = nvfp4_cb_fields(w2, 36, grid="fp8", mode="product", codebook=None, scale_sweep=True, scale_coding="v1", encode_tier="balanced")
    p = nvfp4_cb_assemble_bytes(fields, 36, grid="fp8", mode="product")
    sha = hashlib.sha256(p.cpu().numpy().tobytes()).hexdigest()
    assert sha == "4a348e61b9cd6456722d0cf5b13c76734010992905ce3503fd79c97f573b1a53"
    # direct LDLQ FP8 research capability preserved
    from prismaquant.nvfp4_cb_formats import ldlq_reassign_cb_fields
    raw = nvfp4_cb_fields(w2, 12, grid="fp8", mode="product", codebook=None, col_weights=torch.rand(8,256)+0.05)
    # should not raise for FP8 direct call
    out = ldlq_reassign_cb_fields(w2, raw, torch.rand(8,256)+0.05, torch.randn(16,256), grid="fp8", mode="product")
    assert out is not None

def test_outer_evidence_exact_types():
    E, C = 2, 256
    acts = _make_acts(E, C, N=16, seed=650)
    prep = prepare_ldlq_gate_evidence(acts, qname="test.outer")
    import dataclasses
    # version as float must be rejected
    bad = dataclasses.replace(prep, version=2.0)
    with pytest.raises(ValueError, match="exact int"):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # policy subclass
    class MyStr(str): pass
    bad2 = dataclasses.replace(prep, policy=MyStr(prep.policy))
    with pytest.raises(ValueError, match="exact str"):
        ValidatedPackedLDLQEvidence(bad2, E, C)
    # digest subclass
    class MyDigest(str): pass
    bad3_digests = tuple(MyDigest(d) for d in prep.original_digests)
    bad3 = dataclasses.replace(prep, original_digests=bad3_digests)
    with pytest.raises(ValueError, match="lowercase SHA"):
        ValidatedPackedLDLQEvidence(bad3, E, C)
    # malformed digest
    bad4 = dataclasses.replace(prep, original_digests=("ZZZ",) + prep.original_digests[1:])
    with pytest.raises(ValueError, match="lowercase SHA"):
        ValidatedPackedLDLQEvidence(bad4, E, C)

def test_canonical_split_swap_and_relabel_rejected():
    E, C = 2, 256
    acts = _make_acts(E, C, N=16, seed=660)
    prep = prepare_ldlq_gate_evidence(acts, qname="test.swap")
    import dataclasses
    # Find eligible expert with split
    for i, info in enumerate(prep.split_infos):
        if not info["insufficient"]:
            elig = i
            break
    else:
        assert False, "no eligible expert for swap proof (N=16 guarantees eligible)"
    # Whole FIT/HOLDOUT swap with every count/digest refreshed
    fit = prep.fit_rows[elig]
    hold = prep.holdout_rows[elig]
    orig = prep.original_rows[elig]
    # Swap
    new_fit = hold.clone()
    new_hold = fit.clone()
    new_fit_d = _partition_multiset_digest(new_fit)
    new_hold_d = _partition_multiset_digest(new_hold)
    new_orig_d = _partition_multiset_digest(orig)  # original unchanged (multiset union same)
    new_infos = [dict(s) for s in prep.split_infos]
    new_infos[elig]["n_fit"] = int(new_fit.shape[0])
    new_infos[elig]["n_holdout"] = int(new_hold.shape[0])
    new_infos[elig]["fit_digest"] = new_fit_d
    new_infos[elig]["holdout_digest"] = new_hold_d
    # Keep n_full and full_digest same (swap preserves multiset, but FIT/HOLDOUT membership swapped)
    new_fit_rows = list(prep.fit_rows); new_fit_rows[elig] = new_fit
    new_hold_rows = list(prep.holdout_rows); new_hold_rows[elig] = new_hold
    bad = dataclasses.replace(prep, fit_rows=tuple(new_fit_rows), holdout_rows=tuple(new_hold_rows), fit_digests=tuple(_partition_multiset_digest(t) for t in new_fit_rows), holdout_digests=tuple(_partition_multiset_digest(t) for t in new_hold_rows), fit_filled=tuple(new_fit_rows), split_infos=tuple(new_infos), split_infos_digest=_canonical_split_infos_digest(tuple(new_infos)))
    with pytest.raises(ValueError, match="canonical.*multiset|canonical"):
        ValidatedPackedLDLQEvidence(bad, E, C)
    # eligible-to-insufficient relabel with empty FIT/HOLDOUT and all metadata refreshed (false insufficient)
    prep2 = prepare_ldlq_gate_evidence(acts, qname="test.relabel")
    for i, info in enumerate(prep2.split_infos):
        if not info["insufficient"]:
            elig2 = i
            break
    else:
        assert False, "no eligible for relabel (N=16 guarantees eligible)"
    empty = torch.empty((0, C), dtype=torch.float32)
    empty_d = _partition_multiset_digest(empty)
    new_orig2 = prep2.original_rows[elig2]
    full_d = _partition_multiset_digest(new_orig2)
    new_infos2 = [dict(s) for s in prep2.split_infos]
    new_infos2[elig2] = {"expert": elig2, "n_full": int(new_orig2.shape[0]), "n_fit": 0, "n_holdout": 0, "insufficient": True, "policy": new_infos2[elig2]["policy"], "version": new_infos2[elig2]["version"], "digest": empty_d, "full_digest": full_d, "fit_digest": empty_d, "holdout_digest": empty_d}
    new_fit2 = list(prep2.fit_rows); new_fit2[elig2] = empty
    new_hold2 = list(prep2.holdout_rows); new_hold2[elig2] = empty
    new_filled2 = list(prep2.fit_filled); new_filled2[elig2] = empty
    # recompute derived categories for this adversary: cold stays, eligible loses one, insufficient gains one, fit_missing etc must be wrong vs canonical
    # For simplicity, keep original cold/eligible/insufficient tuples as attacker would refresh them to be self-consistent but false vs canonical
    # Build attacker-refreshed categories that are self-consistent but not canonical (eligible removed)
    orig_cold = set(prep2.cold_experts)
    orig_insuf = set(prep2.insufficient_experts)
    orig_elig = set(prep2.eligible_experts)
    # attacker moves elig2 from eligible to insufficient
    new_insuf = tuple(sorted(orig_insuf | {elig2}))
    new_elig = tuple(sorted(orig_elig - {elig2}))
    new_missing = tuple(sorted(orig_cold | set(new_insuf)))
    has_obs = bool(new_elig)
    bad2 = dataclasses.replace(prep2, fit_rows=tuple(new_fit2), holdout_rows=tuple(new_hold2), fit_filled=tuple(new_filled2), fit_digests=tuple(_partition_multiset_digest(t) for t in new_fit2), holdout_digests=tuple(_partition_multiset_digest(t) for t in new_hold2), split_infos=tuple(new_infos2), split_infos_digest=_canonical_split_infos_digest(tuple(new_infos2)), insufficient_experts=new_insuf, eligible_experts=new_elig, fit_missing_pooled=new_missing, has_observed_fit=has_obs)
    with pytest.raises(ValueError, match="canonical"):
        ValidatedPackedLDLQEvidence(bad2, E, C)

def test_insufficient_not_cold_telemetry():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    # Create acts where expert 0 insufficient (identical rows) and expert 1 eligible
    acts0 = torch.ones((16, C), dtype=torch.float32)  # all identical => insufficient
    g = torch.Generator(); g.manual_seed(670)
    acts1 = torch.randn((16, C), generator=g, dtype=torch.float32)
    acts = (acts0, acts1)
    prep = prepare_ldlq_gate_evidence(acts, qname="test.insuf")
    # Validate we have one insufficient and one eligible
    assert 0 in prep.insufficient_experts or 0 in prep.cold_experts
    # If acts0 is considered cold? No, it's nonempty, so should be insufficient not cold
    # Ensure we have at least one insufficient and one eligible for proof
    assert prep.insufficient_experts, "acts0 must be insufficient for this mixed proof (deterministic identical rows guarantee)"
    val = ValidatedPackedLDLQEvidence(prep, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    weight = _make_weight(E, R_total, C, seed=670)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    parent, gi = encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=val)
    # Telemetry must expose actual cold vs insufficient separately
    assert "insufficient_experts" in gi or "ineligible_experts" in gi or "cold_experts" in gi
    # missing_experts must be only cold, not insufficient
    missing = gi.get("missing_experts", [])
    insuf = set(val.insufficient_experts)
    for eid in insuf:
        assert eid not in missing, f"insufficient expert {eid} reported as missing/cold"
    # Oracle equality
    o_parent, o_kept, _, _ = _manual_oracle(weight, 12, "fp4", "product", cb, True, SC2, "balanced", member_order, slice_boundaries, leaf_cw, val)
    for kk in ("indices", "scales", "scale_super", "scale_sub"):
        if kk in o_parent:
            assert torch.equal(parent[kk], o_parent[kk])
    # One factor for eligible expert only
    import prismaquant.rotation_ldlq_pilot as pilot
    from unittest.mock import patch
    cnt = {"n": 0}
    orig = pilot.inverse_hessian_cholesky
    def counting(*a, **kw):
        cnt["n"] += 1
        return orig(*a, **kw)
    with patch.object(pilot, "inverse_hessian_cholesky", side_effect=counting):
        clear_ldlq_factor_cache()
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=val)
        assert cnt["n"] == len(val.eligible_experts)
    clear_ldlq_factor_cache()

def test_codebook_stride_view_and_version_rejected():
    E, R_total, C = 2, 8, 256
    member_order = ["gate_proj", "up_proj"]
    slice_boundaries = {"gate_proj": (0, 4), "up_proj": (4, 8)}
    weight = _make_weight(E, R_total, C, seed=680)
    leaf_cw = {leaf: torch.rand((E, 1, C)) + 0.05 for leaf in member_order}
    acts = _make_acts(E, C, N=16, seed=681)
    prep = prepare_ldlq_gate_evidence(acts, qname="test.cbview")
    val = ValidatedPackedLDLQEvidence(prep, E, C)
    cb = _explicit_codebook(12, "fp4", "product")
    # In-place version increment between leaves: patch nvfp4_cb_fields to mutate codebook after first leaf snapshot
    orig_fields = nvfp4_cb_fields
    call_n = {"n": 0}
    def fields_mutating(*args, **kw):
        # Codebook is passed via kw or args; mutate the global cb tuple's first table after first leaf
        if call_n["n"] == 1:
            # Mutate in place to bump _version; this changes fingerprint for second leaf
            cb[0].add_(1.0)
        call_n["n"] += 1
        return orig_fields(*args, **kw)
    from unittest.mock import patch
    with patch("prismaquant.nvfp4_cb_formats.nvfp4_cb_fields", new=fields_mutating):
        with pytest.raises(ValueError, match="fingerprint|version|codebook"):
            encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=val)
    # Restore mutated value
    cb[0].sub_(1.0)
    # Ordinary private in-place mutation of validated evidence must be caught via fingerprint
    priv = object.__getattribute__(val, "_priv_prepared")
    t = priv.fit_rows[0]
    ver_before = getattr(t, "_version", None)
    t[0, 0] = t[0, 0] + 1.0
    ver_after = getattr(t, "_version", None)
    if ver_before is not None and ver_after is not None:
        assert ver_after != ver_before
    with pytest.raises(ValueError, match="fingerprint"):
        encode_packed_parent_leaf_local(weight, 12, grid="fp4", mode="product", codebook=cb, scale_sweep=True, scale_coding=SC2, encode_tier="balanced", member_order=member_order, slice_boundaries=slice_boundaries, leaf_col_weights=leaf_cw, prepared=val)
    t[0, 0] = t[0, 0] - 1.0

