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


def cbl_eligible(rung) -> bool:
    """Semantics v2 dispatch rule: CBL wherever the 2048 rule admits it.

    Every measured rung <= K44 is encoded with the pooled learned book, so
    anchors, the audit draw, fallback interiors, and demand extensions all
    share one basis; the menu can never mix encoder families below K45."""
    return int(rung) <= CBL_ELIGIBLE_MAX_RUNG


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


def learn_pool(weight: torch.Tensor, col_weights: torch.Tensor, rung: int):
    """FABLE-certified pooled weighted-Lloyd product book (cand0 domain)."""
    E, R, IN = weight.shape
    dev = weight.device
    g = torch.Generator().manual_seed(LLOYD_ROW_SEED)
    rows_tr = torch.randperm(R, generator=g)[:LLOYD_ROW_SAMPLE]
    vecs, wqs = [], []
    for e in range(E):
        v, _, _ = cb._scale_and_vectorize(weight[e, rows_tr].to(torch.float32), "fp8")
        vecs.append(v)
        wqs.append(_wq_pattern(col_weights[e].to(dev)).unsqueeze(0).expand(
            len(rows_tr), IN // cb.VEC_DIM, cb.VEC_DIM).reshape(-1, cb.VEC_DIM))
    V = torch.cat(vecs)
    Q = torch.cat(wqs)
    sel = torch.randperm(V.shape[0], generator=g)[:LLOYD_CAP].to(dev)
    V, Q = V[sel], Q[sel]
    n_sub = cb.family_for("fp8", "product").n_sub
    bits = cb.subtable_bit_widths(int(rung), "product", n_sub)
    sub = cb.VEC_DIM // n_sub
    return tuple(
        cb.learn_codebook(
            V[:, i * sub:(i + 1) * sub], b, grid="fp8",
            col_weights=Q[:, i * sub:(i + 1) * sub],
            init=cb.fixed_lattice(b, "fp8", sub).to(dev),
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
              source_digest: str, col_weights_digest: str) -> str:
    payload = json.dumps({
        "schema": CBL_SEMANTICS_SCHEMA, "layer": int(layer),
        "projection": str(projection), "rung": int(rung),
        "source_digest": source_digest,
        "col_weights_digest": col_weights_digest,
        "train": SEMANTICS_STAMP["book_train"],
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_or_train_book(*, layer, projection, rung, weight, col_weights,
                        source_digest, col_weights_digest):
    """Content-addressed book: resume loads by key, never retrains."""
    key = _book_key(layer=layer, projection=projection, rung=rung,
                    source_digest=source_digest,
                    col_weights_digest=col_weights_digest)
    path = BOOK_ROOT / key[:2] / f"{key}.safetensors"
    if path.is_file():
        with safe_open(str(path), framework="pt") as f:
            meta = json.loads(f.metadata()["dsv4_cbl_book"])
            pool = tuple(f.get_tensor(f"sub{i}").to(torch.float32)
                         for i in range(int(meta["n_sub"])))
        return tuple(t.to(weight.device) for t in pool), meta["book_sha256"], \
            str(path), "book_restored_content_addressed", 0.0
    started = time.perf_counter()
    pool = learn_pool(weight, col_weights, rung)
    lloyd_seconds = time.perf_counter() - started
    sha = book_sha256(pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {f"sub{i}": t.detach().cpu().to(torch.float16).contiguous()
               for i, t in enumerate(pool)}
    meta = {"schema": "prismaquant.dsv4_cbl_book.v1", "book_key": key,
            "book_sha256": sha, "n_sub": len(pool),
            "layer": int(layer), "projection": str(projection),
            "rung": int(rung), "source_digest": source_digest,
            "col_weights_digest": col_weights_digest,
            "train": SEMANTICS_STAMP["book_train"],
            "device_class": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu"}
    tmp = path.with_suffix(".tmp")
    save_file(tensors, str(tmp), metadata={"dsv4_cbl_book": json.dumps(meta)})
    os.replace(tmp, path)
    return pool, sha, str(path), "book_trained_and_banked", lloyd_seconds


def encode_cbl(*, layer, projection, rung, data, expert_ids,
               source_identity=None, col_weights_identity=None,
               use_warm=True, expected_free_errors=None):
    """CBL free arm: pooled book + one-shot cand0 scales, weighted assignment,
    no sweep, no LDLQ. Returns the `_encode_free` 5-tuple contract; the
    content-addressed book artifact path plays the reusable-state role."""
    from prismaquant.cb_warm_state import tensor_value_identity
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
    )
    fields = nvfp4_cb_fields(
        weight, int(rung), grid="fp8", mode="product",
        col_weights=col_weights, codebook=pool,
        scale_sweep=False, encode_tier="balanced",
    )
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - started
    score_started = time.perf_counter()
    reconstruction = nvfp4_cb_reconstruct(
        fields, int(rung), grid="fp8", mode="product",
    ).to(weight.dtype)
    errors = _campaign._weight_mse(weight, reconstruction)
    torch.cuda.synchronize()
    if expected_free_errors is not None and \
            list(map(float, expected_free_errors)) != errors:
        raise AssertionError(
            f"CBL re-derivation mismatch L{layer} {projection} K{rung}: "
            "content-addressed book must reproduce banked errors bit-exactly"
        )
    timing = {
        "free_encode_seconds": encode_seconds,
        "free_reconstruct_and_weight_mse_seconds":
            time.perf_counter() - score_started,
        "lloyd_seconds": lloyd_seconds,
        "warm_state_write_seconds": 0.0,
        "warm_state_outcome": book_outcome,
        "measurement_semantics": {
            "schema": CBL_SEMANTICS_SCHEMA, "encoder": "cbl_poolb",
            "ldlq": False, "scale_policy": "one_shot_cand0",
            "book_sha256": sha, "book_path": book_path,
        },
    }
    return fields, reconstruction, errors, timing, book_path


def audit_sample(digests: Sequence[str], n: int = AUDIT_N) -> list[int]:
    keyed = sorted(range(len(digests)), key=lambda e: hashlib.sha256(
        (digests[e] + AUDIT_SALT).encode()).hexdigest())
    return keyed[:n]


def audit_projection(*, layer, projection, data, measured_rungs,
                     selected_by_rung, winning_by_rung, epsilon_le, rel_epsilon,
                     micro_check_k43, out_root):
    """n=16 content-keyed claim-verification audit (+ first-layer K43 gate).

    Re-encodes the sample with the incumbent no-LDLQ recipe at every adopted
    rung measured in this projection and verifies the recorded CBL/chain rows
    beat the incumbent per expert (audit fallback set bounded), recording a
    durable report. Raises on gate failure — mandatory, like G1/G2."""
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
    }
    adopted = [r for r in measured_rungs if cbl_eligible(r)]
    for rung in adopted:
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
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True))
    os.replace(tmp, path)
    if not report["pass"]:
        raise AssertionError(
            f"CBL audit gate failed L{layer} {projection}: {path}"
        )
    return report
