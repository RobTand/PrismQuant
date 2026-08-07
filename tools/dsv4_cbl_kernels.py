"""CBL measurement kernels for the DSV4 A-FAST burn (chain-min-v1 semantics).

Provenance: assignment core and pooled-Lloyd trainer are the FABLE-certified
kernels from the F1/F2 studies (transfer-study-fable-verify/: certification
rel-diff <= 1.24e-08 vs prismaquant.nvfp4_cb_formats._eval_candidate at every
rung, GPU, TF32 off; cross-device identical CPU vs GB10). Design contract:
cost-ldlq/transfer-study-fable-verify/BURN_INTEGRATION_CBL.md and
gates_amended.json (ALL PASS, 2026-08-05).

Semantics registered here:
  * adopted rungs {28,33,38,43}: pooled learned product book per
    (layer, projection, rung), one-shot cand-0 scales, weighted assignment,
    NO scale sweep, NO LDLQ ("cbl_poolb").
  * all other rungs (K48 incl. audit draws): incumbent sweep WITHOUT LDLQ
    ("incumbent_sweep_noldlq") — operator decision 2026-08-05: LDLQ leaves
    the measurement loop (export-time A/B decides "if at all").
  * books are ARTIFACTS: content-addressed under RUN_ROOT/bucket-books/,
    fp16 sub-tables (grid values exact), sha256 folded into cell identity;
    resume loads by hash, never retrains.
  * audit: content-keyed n=16 sample per projection, incumbent-recipe
    re-encode at adopted rungs, per-expert claim verification.

NVFP4-CB extension (study/nvfp4-cbl branch):
  Same construction generalised to the NVFP4-CB family (grid="fp4",
  mode="product", n_sub=2, VEC_DIM=8 so sub_dim=4). The FP8 path is
  preserved bit-identically (default grid="fp8", mode="product" hits the
  legacy code path with identical hashes and tables). NVFP4 books live
  under /home/rob/dq-runs/nvfp4-cbl/books/ and are keyed by an
  enlarged book-key that includes grid/mode. Centroids are snapped to
  the family grid (E4M3 for fp8, E2M1 for fp4) — the format code
  already does this; the generalised learn_pool simply forwards the
  correct grid. Byte accounting uses the exact serialised type_size:
  fp4 v1 index K/8 + 0.5 bpw (16 B scale plane), fp4 two_tier 0.28125,
  fp8 K/8.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.nvfp4_cb_formats import nvfp4_cb_fields, nvfp4_cb_reconstruct
from tools import dsv4_afast_campaign as _campaign
from tools import dsv4_ldlq_cost_campaign as _ldlq_campaign
from tools.dsv4_ldlq_cost_campaign import RUN_ROOT

CBL_ADOPTED_RUNGS = frozenset({28, 33, 38})  # the anchor set; books are
# always banked here.  Encoder dispatch no longer keys on membership —
# see cbl_eligible (semantics v2).
CBL_ELIGIBLE_MAX_RUNG = 44  # registered 2048 rule: every product sub-table
# holds <= 2048 entries through K44; K45+ stays incumbent everywhere.
CBL_SEMANTICS_SCHEMA = "prismaquant.dsv4_cbl_measurement_semantics.v2"
BOOK_ROOT = RUN_ROOT / "bucket-books"
LLOYD_ROW_SAMPLE = 64
LLOYD_ROW_SEED = 4321
LLOYD_CAP = 2_000_000
LLOYD_ITERS = 4
LLOYD_SEED = 0
AUDIT_N = 16
AUDIT_SALT = "cbl-audit-v1"
AUDIT_SAMPLE_FALLBACK_MAX = 1          # >1/16 sampled incumbent wins -> abort
MICROCHECK_K43_MEDIAN_MAX = -0.10      # registered first-layer drift gate

# ---------------------------------------------------------------------------
# NVFP4-CB extension constants (study/nvfp4-cbl). FP8 values above remain
# authoritative for the production burn; NVFP4 values are additive.
# ---------------------------------------------------------------------------
NVFP4_CBL_ELIGIBLE_MAX_RUNG = 22  # 2048 rule for n_sub=2: ceil/floor split
# K23 (12,11) already has a 4096-entry sub-table, so K22 is the ceiling.
NVFP4_CBL_SEMANTICS_SCHEMA = "prismaquant.dsv4_cbl_measurement_semantics.nvfp4_v1"
NVFP4_BOOK_ROOT = Path("/home/rob/dq-runs/nvfp4-cbl/books")
# Keep NVFP4 stamp separate so FP8 stamp stays bit-identical.
NVFP4_CBL_ADOPTED_RUNGS = frozenset()  # study phase: no adoption yet


def cbl_eligible(rung, grid: str = "fp8", mode: str = "product") -> bool:  # type: ignore[override]
    """Semantics v2 dispatch rule: CBL wherever the 2048 rule admits it.

    Every measured rung <= K44 is encoded with the pooled learned book, so
    anchors, the audit draw, fallback interiors, and demand extensions all
    share one basis; the menu can never mix encoder families below K45.

    NVFP4 extension: NVFP4_CB_K12-K22 eligible (2048 rule with n_sub=2);
    K23+ stays incumbent. The grid/mode overload is backward compatible:
    cbl_eligible(rung) without args remains the FP8 rule.
    """
    r = int(rung)
    g = str(grid).lower()
    m = str(mode).lower()
    if g == "fp4" and m == "product":
        return r <= NVFP4_CBL_ELIGIBLE_MAX_RUNG
    if g == "fp4" and m == "signed":
        # Signed family not part of CBL study; treat as ineligible.
        return False
    # FP8 product (and any unknown combination defaults to FP8 ceiling for
    # backward compat — callers that don't name a family are FP8 call sites).
    return r <= CBL_ELIGIBLE_MAX_RUNG


SEMANTICS_STAMP = {
    "schema": CBL_SEMANTICS_SCHEMA,
    "adopted_rungs": sorted(CBL_ADOPTED_RUNGS),
    "priced_domain": [28, 38],
    "above_ceiling_policy": "demand_extension_or_native_fp8_fallback",
    "cbl_dispatch": "eligible_le_k44",
    "eligible_max_rung": CBL_ELIGIBLE_MAX_RUNG,
    "audit_basis": "cbl",
    "interp_coordinate": "hierarchical_law_v1_two_probe",
    "interp_anchors": [29, 35],
    "interp_law": {
        "level1": "logy=a0+a1K+phi_{K%4} per projection, fitted L14 medians",
        "level2": "per-expert affine log correction from anchors; post-gate "
                  "refit includes the audit draw",
        "output_metric": "per-expert coupling O=c*W^gamma to law weight",
        "detection": {"a_abs": 0.50, "b_abs": 0.015},
    },
    "tier3_group_mse": {
        "group_counts": [2, 4, 8], "basis": "free_arm",
        "grade": "study_sideband_cells_only",
    },
    "adopted_encoder": "cbl_poolb",
    "other_encoder": "incumbent_sweep_noldlq",
    "ldlq_in_measurement": False,
    "selection": "chain-min-v1",
    "book_train": {
        "row_sample": LLOYD_ROW_SAMPLE, "row_seed": LLOYD_ROW_SEED,
        "cap": LLOYD_CAP, "iters": LLOYD_ITERS, "seed": LLOYD_SEED,
        "init": "fixed_lattice", "normalization": "cand0_v1",
    },
    "audit": {"n": AUDIT_N, "salt": AUDIT_SALT,
              "sample_fallback_max": AUDIT_SAMPLE_FALLBACK_MAX,
              "microcheck_k43_median_max": MICROCHECK_K43_MEDIAN_MAX},
}

NVFP4_SEMANTICS_STAMP = {
    "schema": NVFP4_CBL_SEMANTICS_SCHEMA,
    "adopted_rungs": sorted(NVFP4_CBL_ADOPTED_RUNGS),
    "priced_domain": [12, 22],
    "above_ceiling_policy": "demand_extension_or_incumbent",
    "cbl_dispatch": "eligible_le_k22",
    "eligible_max_rung": NVFP4_CBL_ELIGIBLE_MAX_RUNG,
    "audit_basis": "cbl",
    "interp_coordinate": "hierarchical_law_v1_two_probe",
    "interp_anchors": [],  # study-only; no interpolation yet
    "adopted_encoder": "cbl_poolb",
    "other_encoder": "incumbent_sweep_noldlq",
    "ldlq_in_measurement": False,
    "selection": "chain-min-v1",
    "book_train": {
        "row_sample": LLOYD_ROW_SAMPLE, "row_seed": LLOYD_ROW_SEED,
        "cap": LLOYD_CAP, "iters": LLOYD_ITERS, "seed": LLOYD_SEED,
        "init": "fixed_lattice", "normalization": "cand0_v1",
    },
    "audit": {"n": AUDIT_N, "salt": AUDIT_SALT,
              "sample_fallback_max": AUDIT_SAMPLE_FALLBACK_MAX},
}

_NOLDLQ_CONTEXT = None


def _noldlq_context():
    """The incumbent context with LDLQ removed (sweep + balanced tier kept)."""
    global _NOLDLQ_CONTEXT
    if _NOLDLQ_CONTEXT is None:
        _NOLDLQ_CONTEXT = type(_ldlq_campaign.CONTEXT).production(
            scale_sweep=True, ldlq=False, encode_tier="balanced",
        )
    return _NOLDLQ_CONTEXT


def encode_free_noldlq(**kwargs):
    """Incumbent free-encode with LDLQ removed from the measurement row.

    Implementation: temporary module-global CONTEXT swap around the frozen
    `dsv4_afast_campaign._encode_free` (single-threaded burn; try/finally
    restores). Chosen over a body copy so the encoder logic stays literally
    the reviewed production function; the swapped context flows consistently
    into warm-state keying (new context -> cold once, warm thereafter).
    """
    saved_campaign = _campaign.CONTEXT
    saved_ldlq = _ldlq_campaign.CONTEXT
    ctx = _noldlq_context()
    try:
        _campaign.CONTEXT = ctx
        _ldlq_campaign.CONTEXT = ctx
        fields, recon, errors, timing, warm_path = _campaign._encode_free(**kwargs)
    finally:
        _campaign.CONTEXT = saved_campaign
        _ldlq_campaign.CONTEXT = saved_ldlq
    timing = dict(timing)
    timing["measurement_semantics"] = {
        "schema": CBL_SEMANTICS_SCHEMA, "encoder": "incumbent_sweep_noldlq",
        "ldlq": False,
    }
    return fields, recon, errors, timing, warm_path


def _wq_pattern(cw_e: torch.Tensor) -> torch.Tensor:
    """(1, in) col weights -> (in//8, 8) production vector-weight pattern
    (row-periodic exactness: the production imatrix broadcasts over rows)."""
    return cb._col_weight_vectors(cw_e.reshape(-1, cb.VEC_DIM))


def learn_pool(weight: torch.Tensor, col_weights: torch.Tensor, rung: int,
               grid: str = "fp8", mode: str = "product"):
    """FABLE-certified pooled weighted-Lloyd product book (cand0 domain).

    Generalised to both families. Default grid="fp8", mode="product" is the
    legacy FP8 path bit-identical to the production burn. For NVFP4 pass
    grid="fp4", mode="product" (n_sub=2, sub_dim=4, E2M1 snapping).
    """
    grid = str(grid).lower()
    mode = str(mode).lower()
    E, R, IN = weight.shape
    dev = weight.device
    g = torch.Generator().manual_seed(LLOYD_ROW_SEED)
    rows_tr = torch.randperm(R, generator=g)[:LLOYD_ROW_SAMPLE]
    vecs, wqs = [], []
    for e in range(E):
        v, _, _ = cb._scale_and_vectorize(weight[e, rows_tr].to(torch.float32), grid)
        vecs.append(v)
        wqs.append(_wq_pattern(col_weights[e].to(dev)).unsqueeze(0).expand(
            len(rows_tr), IN // cb.VEC_DIM, cb.VEC_DIM).reshape(-1, cb.VEC_DIM))
    V = torch.cat(vecs)
    Q = torch.cat(wqs)
    sel = torch.randperm(V.shape[0], generator=g)[:LLOYD_CAP].to(dev)
    V, Q = V[sel], Q[sel]
    n_sub = cb.family_for(grid, mode).n_sub
    bits = cb.subtable_bit_widths(int(rung), mode, n_sub)
    sub = cb.VEC_DIM // n_sub
    return tuple(
        cb.learn_codebook(
            V[:, i * sub:(i + 1) * sub], b, grid=grid,
            col_weights=Q[:, i * sub:(i + 1) * sub],
            init=cb.fixed_lattice(b, grid, sub).to(dev),
            iters=LLOYD_ITERS, seed=LLOYD_SEED,
        )
        for i, b in enumerate(bits)
    )


def book_sha256(pool: Sequence[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for t in pool:
        h.update(t.detach().cpu().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def _book_key(*, layer: int, projection: str, rung: int,
              source_digest: str, col_weights_digest: str,
              grid: str = "fp8", mode: str = "product") -> str:
    grid = str(grid).lower()
    mode = str(mode).lower()
    # Backward-compat: FP8 product keeps the legacy payload without grid/mode
    # so existing bucket-books hashes remain valid and resumes are bit-identical.
    if grid == "fp8" and mode == "product":
        payload = json.dumps({
            "schema": CBL_SEMANTICS_SCHEMA, "layer": int(layer),
            "projection": str(projection), "rung": int(rung),
            "source_digest": source_digest,
            "col_weights_digest": col_weights_digest,
            "train": SEMANTICS_STAMP["book_train"],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()
    # NVFP4 (or any non-FP8) key includes family to avoid cross-family collisions
    # (FP8 K28 and NVFP4 K12 would otherwise share the integer 28 vs 12 - distinct
    # anyway, but grid/mode makes the provenance explicit and future-proof).
    payload = json.dumps({
        "schema": NVFP4_CBL_SEMANTICS_SCHEMA, "layer": int(layer),
        "projection": str(projection), "rung": int(rung),
        "grid": grid, "mode": mode,
        "source_digest": source_digest,
        "col_weights_digest": col_weights_digest,
        "train": NVFP4_SEMANTICS_STAMP["book_train"],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _book_root_for(grid: str, mode: str) -> Path:
    g = str(grid).lower()
    m = str(mode).lower()
    if g == "fp4" and m == "product":
        return NVFP4_BOOK_ROOT
    # FP8 stays in the production bank; any unknown family defaults there
    # (legacy call sites without grid arg land here).
    return BOOK_ROOT


def _load_or_train_book(*, layer, projection, rung, weight, col_weights,
                        source_digest, col_weights_digest,
                        grid: str = "fp8", mode: str = "product"):
    """Content-addressed book: resume loads by key, never retrains."""
    grid = str(grid).lower()
    mode = str(mode).lower()
    key = _book_key(layer=layer, projection=projection, rung=rung,
                    source_digest=source_digest,
                    col_weights_digest=col_weights_digest,
                    grid=grid, mode=mode)
    book_root = _book_root_for(grid, mode)
    path = book_root / key[:2] / f"{key}.safetensors"
    if path.is_file():
        with safe_open(str(path), framework="pt") as f:
            meta = json.loads(f.metadata()["dsv4_cbl_book"])
            pool = tuple(f.get_tensor(f"sub{i}").to(torch.float32)
                         for i in range(int(meta["n_sub"])))
        return tuple(t.to(weight.device) for t in pool), meta["book_sha256"], \
            str(path), "book_restored_content_addressed", 0.0
    started = time.perf_counter()
    pool = learn_pool(weight, col_weights, rung, grid=grid, mode=mode)
    lloyd_seconds = time.perf_counter() - started
    sha = book_sha256(pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {f"sub{i}": t.detach().cpu().to(torch.float16).contiguous()
               for i, t in enumerate(pool)}
    # Stash the family in the sidecar so audits can verify grid legality
    stamp = SEMANTICS_STAMP if (grid == "fp8" and mode == "product") else NVFP4_SEMANTICS_STAMP
    schema = "prismaquant.dsv4_cbl_book.v1"
    book_schema = CBL_SEMANTICS_SCHEMA if (grid == "fp8" and mode == "product") else NVFP4_CBL_SEMANTICS_SCHEMA
    meta = {"schema": schema, "book_schema": book_schema,
            "book_key": key,
            "book_sha256": sha, "n_sub": len(pool),
            "layer": int(layer), "projection": str(projection),
            "rung": int(rung), "grid": grid, "mode": mode,
            "source_digest": source_digest,
            "col_weights_digest": col_weights_digest,
            "train": stamp["book_train"],
            "device_class": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu"}
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata={"dsv4_cbl_book": json.dumps(meta)})
    os.replace(tmp, path)
    return pool, sha, str(path), "book_trained_and_banked", lloyd_seconds


def encode_cbl(*, layer, projection, rung, data, expert_ids,
               source_identity=None, col_weights_identity=None,
               use_warm=True, expected_free_errors=None,
               grid: str = "fp8", mode: str = "product",
               scale_coding: str = "v1"):
    """CBL free arm: pooled book + one-shot cand0 scales, weighted assignment,
    no sweep, no LDLQ. Returns the `_encode_free` 5-tuple contract; the
    content-addressed book artifact path plays the reusable-state role.

    Family-parameterized: grid/mode select the family (fp8/product vs
    fp4/product). Default remains fp8/product for production-burn
    bit-identity; NVFP4 callers pass grid="fp4".
    """
    from prismaquant.cb_warm_state import tensor_value_identity
    grid = str(grid).lower()
    mode = str(mode).lower()
    weight = _campaign._select_experts(data["weight"], expert_ids)
    col_weights = _campaign._select_experts(data["col_weights"], expert_ids)
    if source_identity is None:
        source_identity = tensor_value_identity(weight)
    if col_weights_identity is None:
        col_weights_identity = tensor_value_identity(col_weights)
    torch.cuda.synchronize()
    started = time.perf_counter()
    pool, sha, book_path, book_outcome, lloyd_seconds = _load_or_train_book(
        layer=layer, projection=projection, rung=int(rung), weight=weight,
        col_weights=col_weights, source_digest=source_identity[1],
        col_weights_digest=col_weights_identity[1],
        grid=grid, mode=mode,
    )
    # NVFP4 scale_coding is carried in the fields for byte accounting but does
    # not affect the CBL book itself (the book is in the normalized vector
    # domain; scales are per-group and one-shot cand0 either way).
    fields = nvfp4_cb_fields(
        weight, int(rung), grid=grid, mode=mode,
        col_weights=col_weights, codebook=pool,
        scale_sweep=False, encode_tier="balanced",
        scale_coding=scale_coding if grid == "fp4" else "v1",
    )
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - started
    score_started = time.perf_counter()
    reconstruction = nvfp4_cb_reconstruct(
        fields, int(rung), grid=grid, mode=mode,
    ).to(weight.dtype)
    errors = _campaign._weight_mse(weight, reconstruction)
    torch.cuda.synchronize()
    if expected_free_errors is not None and \
            list(map(float, expected_free_errors)) != errors:
        raise AssertionError(
            f"CBL re-derivation mismatch L{layer} {projection} K{rung} [{grid}/{mode}]: "
            "content-addressed book must reproduce banked errors bit-exactly"
        )
    schema = CBL_SEMANTICS_SCHEMA if (grid == "fp8" and mode == "product") else NVFP4_CBL_SEMANTICS_SCHEMA
    timing = {
        "free_encode_seconds": encode_seconds,
        "free_reconstruct_and_weight_mse_seconds":
            time.perf_counter() - score_started,
        "lloyd_seconds": lloyd_seconds,
        "warm_state_write_seconds": 0.0,
        "warm_state_outcome": book_outcome,
        "measurement_semantics": {
            "schema": schema, "encoder": "cbl_poolb",
            "ldlq": False, "scale_policy": "one_shot_cand0",
            "grid": grid, "mode": mode,
            "book_sha256": sha, "book_path": book_path,
        },
    }
    return fields, reconstruction, errors, timing, book_path


# ---------------------------------------------------------------------------
# Byte accounting — exact serialised size, never an approximation.
# ---------------------------------------------------------------------------

def cb_effective_bpw(k: int, grid: str = "fp4", scale_coding: str = "v1") -> float:
    """Exact bpw for a CB rung (index K/8 + scale plane if any).

    Uses the serialised type_size so the number matches what the exporter
    writes. fp4 v1: K/8 + 0.5, fp4 two_tier: K/8 + 0.28125, fp8: K/8.
    """
    from prismaquant.nvfp4_cb_formats import nvfp4_cb_effective_bits
    return float(nvfp4_cb_effective_bits(int(k), str(grid), str(scale_coding)))


def cb_bytes_per_superblock(k: int, grid: str = "fp4", scale_coding: str = "v1") -> int:
    """Exact serialised bytes per 256-weight superblock."""
    from prismaquant.nvfp4_cb_formats import nvfp4_cb_type_size
    return int(nvfp4_cb_type_size(int(k), str(grid), str(scale_coding)))


def verify_grid_snap(pool: Sequence[torch.Tensor], grid: str) -> bool:
    """Return True iff every centroid coordinate is on the element grid."""
    for t in pool:
        snapped = cb._snap_to_grid(t.to(torch.float32), str(grid))
        if not torch.equal(snapped, t.to(torch.float32)):
            return False
    return True


def encode_incumbent_nvfp4(*, weight: torch.Tensor, rung: int,
                           col_weights: torch.Tensor | None = None,
                           scale_coding: str = "v1",
                           encode_tier: str = "balanced"):
    """Incumbent NVFP4-CB encoder at one rung (fixed lattice, sweep, no CBL).

    Used by the F1-style study to compare CBL vs incumbent at matched K.
    Returns (fields, reconstruction, per-row weight MSE list).
    """
    from prismaquant.cb_layout import SUPERBLOCK
    if int(weight.shape[-1]) % SUPERBLOCK != 0:
        raise ValueError(f"in_features={weight.shape[-1]} must be multiple of 256")
    fields = nvfp4_cb_fields(
        weight, int(rung), grid="fp4", mode="product",
        col_weights=col_weights, codebook=None,
        scale_sweep=True, encode_tier=encode_tier,
        scale_coding=scale_coding,
    )
    recon = nvfp4_cb_reconstruct(fields, int(rung), grid="fp4", mode="product").to(weight.dtype)
    # per-row (per-expert slice) weighted MSE not needed here; caller scores
    # per-expert via _weight_mse on the full expert stack.
    return fields, recon


def audit_sample(digests: Sequence[str], n: int = AUDIT_N) -> list[int]:
    keyed = sorted(range(len(digests)), key=lambda e: hashlib.sha256(
        (digests[e] + AUDIT_SALT).encode()).hexdigest())
    return keyed[:n]


def audit_projection(*, layer, projection, data, measured_rungs,
                     selected_by_rung, winning_by_rung, epsilon_le, rel_epsilon,
                     micro_check_k43, out_root,
                     grid: str = "fp8", mode: str = "product"):
    """n=16 content-keyed claim-verification audit (+ first-layer K43 gate).

    Re-encodes the sample with the incumbent no-LDLQ recipe at every adopted
    rung measured in this projection and verifies the recorded CBL/chain rows
    beat the incumbent per expert (audit fallback set bounded), recording a
    durable report. Raises on gate failure — mandatory, like G1/G2.

    Family-aware: when grid="fp4", adoption uses the NVFP4 2048 rule (K<=22)
    and the incumbent re-encode is the fp4 fixed-lattice sweep; for fp8 the
    behaviour is bit-identical to the legacy audit.
    """
    grid = str(grid).lower()
    mode = str(mode).lower()
    weight = data["weight"]
    E = int(weight.shape[0])
    digests = [hashlib.sha256(
        weight[e].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest() for e in range(E)]
    sample = audit_sample(digests)
    report: dict[str, Any] = {
        "schema": "prismaquant.dsv4_cbl_projection_audit.v1",
        "layer": int(layer), "projection": str(projection),
        "sample": sample, "salt": AUDIT_SALT, "rungs": {}, "pass": True,
        "grid": grid, "mode": mode,
    }
    adopted = [r for r in measured_rungs if cbl_eligible(r, grid=grid, mode=mode)]
    for rung in adopted:
        if grid == "fp4" and mode == "product":
            # NVFP4 incumbent: fixed lattice sweep, balanced tier, v1 scales
            # (the study writer's contract; matches Phase B's comparison basis).
            sample_weight = _campaign._select_experts(data["weight"], tuple(sample))
            sample_col = _campaign._select_experts(data["col_weights"], tuple(sample))
            fields = nvfp4_cb_fields(
                sample_weight, int(rung), grid="fp4", mode="product",
                col_weights=sample_col, codebook=None,
                scale_sweep=True, encode_tier="balanced",
                scale_coding="v1",
            )
            recon = nvfp4_cb_reconstruct(fields, int(rung), grid="fp4", mode="product").to(sample_weight.dtype)
            own_errors = _campaign._weight_mse(sample_weight, recon)
        else:
            _, _, own_errors, _, _ = encode_free_noldlq(
                layer=layer, projection=projection, rung=int(rung), data=data,
                expert_ids=tuple(sample), use_warm=True,
            )
        fallback = []
        uplifts = []
        for local, expert in enumerate(sample):
            sel = float(selected_by_rung[int(rung)][expert])
            own = float(own_errors[local])
            uplifts.append((sel - own) / own if own > 0 else 0.0)
            if not epsilon_le(sel, own, rtol=rel_epsilon):
                fallback.append(int(expert))
        med = float(torch.tensor(uplifts).median())
        row = {"sampled_median_uplift_vs_incumbent": med,
               "audit_fallback_experts": fallback,
               "winning_arms_sample": [
                   winning_by_rung[int(rung)][e] for e in sample],
               "pass": len(fallback) <= AUDIT_SAMPLE_FALLBACK_MAX}
        if micro_check_k43 and int(rung) == 43:
            row["microcheck_k43"] = {
                "median_max": MICROCHECK_K43_MEDIAN_MAX,
                "measured_median": med,
                "pass": med <= MICROCHECK_K43_MEDIAN_MAX,
            }
            row["pass"] = row["pass"] and row["microcheck_k43"]["pass"]
        report["rungs"][f"K{int(rung)}"] = row
        report["pass"] = report["pass"] and row["pass"]
    path = Path(out_root) / f"CBL_AUDIT_L{int(layer):02d}_{projection}.json"
    # Disambiguate NVFP4 audits so FP8 production audits are untouched
    if grid != "fp8" or mode != "product":
        path = Path(out_root) / f"CBL_AUDIT_L{int(layer):02d}_{projection}_{grid}_{mode}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True))
    os.replace(tmp, path)
    if not report["pass"]:
        raise AssertionError(
            f"CBL audit gate failed L{layer} {projection} [{grid}/{mode}]: {path}"
        )
    return report
