# 35B MoE served verdict — NVFP4-CB lane (first CB-on-MoE artifact)

**Model:** Ornith-1.0-35B (Qwen3.5-MoE-class, E=256 top-8, hybrid DeltaNet),
BF16 source `/home/rob/dq-runs/ornith-35b-base` (per-expert-on-disk, nested
`model.language_model.*` naming).
**OURS:** `EXPORT_CONTAINER=nvfp4_cb`, TARGET_BITS=4.75, widened 6-rung menu
(FP8_CB_K28..K48 + NVFP4/FP8_DYNAMIC/BF16), **corrected cost recipe**
(`CB_EXPERT_EMPIRICAL=0`: local weighted-MSE expert costs with per-expert
imatrix incl. the down_proj routed replay, `PRISMAQUANT_EXPERT_COST_SAMPLE=16`,
`CB_LADDER_INTERP=1` floor-law), lattice codebooks, 8×512 calib.
Served 2026-07-19 (`scripts/measure_35b_ab.sh`, `--enforce-eager`,
`--max-num-batched-tokens 2096`).

## Pipeline wall-clock (the encode-cost thesis at production scale)
- Cost stage: **~39 min total** (3 body shards ~12.7 min each + MTP sidecar)
  vs ~14 h projected for the killed 07-18 recipe (5¼ h local double-work +
  ~9 h empirical). Floor-law ladder holdout rejection 5–20% per chunk
  (log-linear had rejected 60–90%).
- Allocation @4.75 (achieved 4.758): **all-CB body across all six rungs** —
  K44 146, K36 26, K32 24, K48 21, K40 16, K28 2 — zero stock formats on
  experts (profile-scoped: the container's stock-CT delegation is dense-only),
  BF16 floor 265.
- Streaming export: **22 GB artifact**, 1,261 tensors, 235 CB targets
  (80 stacked expert groups (E,out,bytes)).

## Gold metric (vs the SAME BF16 session, wiki.test 8176 positions, top-20)

| | conf-KL | ALL-KL | conf top1 | ALL top1 | PPL |
|---|---|---|---|---|---|
| BF16 (ref) | — | — | — | — | 9.437 |
| AURA-4.75 (shipped, native serve) | 0.03625 | 0.0492 | 98.44% | 89.9% | 9.587 |
| **OURS (CB @4.75)** | **0.01706** | **0.0278** | **99.05%** | **92.6%** | **9.542 (+1.1%)** |

**CB wins decisively at matched bpp on MoE: conf-KL −53%, ALL-KL −43%,
top-1 better on both slices, PPL gap to BF16 −30%** (AURA arm measured
2026-07-19 against the SAME BF16 dump, same harness — the earlier
"0.0143" memory figure was a different protocol and is not comparable).
Also beats the arm-E-era served CB result (ALL-KL 0.0292) on a fully-
automated recipe. The 27B thesis reproduces on a 256-expert MoE: the same
bytes, spent on codebook rungs, buy materially more quality.

## Speed — correctness-tier serving (the known kernel gap)

| | TTFT(1400) | decode tok/s |
|---|---|---|
| BF16 (A3B) | 0.484 s | 28.43 |
| AURA-4.75 (native) | 0.325 s | ~35.9 (steady reps) |
| OURS (CB, per-expert transient loop) | 3.53 s | 3.52 |
| **OURS (CB, grouped decode kernels 2026-07-19)** | 3.46 s | **~33 (32.6–33.3 steady)** |

**Decode SOLVED: 9.4× via the grouped (token, expert)-pair GEMV — one
launch per projection replaces ~10k host syncs/launches per token. The CB
artifact now decodes FASTER than BF16 (33 vs 28.4) at 3× smaller, −43%
ALL-KL vs AURA at matched bpp; 8% shy of AURA's decode.** KL re-verified
with the fast path live (ALL 0.0278 identical, conf 0.01673, PPL 9.501 —
arithmetic-drift class). TTFT unmoved (3.46 s): the prefill wall is the
per-expert launch storm (~71k launches / 1400-tok prefill), not syncs —
batched-expert expand + grouped GEMM is the remaining piece.

(The correctness-first per-expert loop remains as the prefill/fallback
path; parity between the two is pinned by
`test_moe_grouped_gemv_matches_loop_numerics`.)

## First-contact bug ledger (all fixed + committed this session)
1. Packed-expert down_proj imatrix unobtainable from the act cache → routed
   replay synthesis (model-loaded + checkpoint-based twins), persisted for
   exporter lockstep.
2. Zero-expert calibration bug in the empirical CLI (packed params load
   zero-initialized) → fill wired + loud all-zero guard.
3. `fill_packed_experts_from_source` silently no-ops against staged text-only
   dirs → pass the ORIGINAL model dir.
4. Resident exporter: no per-expert→packed skeleton bridge → added (per-group
   invocation, bounded transient); then superseded by streaming for 35B.
5. Resident export OOM-killed (66 GB source + outputs + GPU workspace on the
   121 GB unified pool) → `EXPORT_STREAMING=1`.
6. Streaming exporter expert-group join missed nested checkpoint prefixes →
   canonical-prefix mapping.
7. Streaming copy pass double-shipped every per-expert bf16 tensor next to
   its packed stack (82 GB artifact) → canonical-prefix consumption test.
8. Serving: double `set_weight_attrs` assert + RoutedExperts loader
   substring-mapping derives dotted attrs / transpose-corrupts byte tensors →
   one-attrs-call + instance-level `load_weights` wrap (CB names direct,
   rest delegated).
