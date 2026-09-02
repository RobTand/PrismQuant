# NVFP4-CB — Format + Pipeline Integration Design (PLAN ONLY)

> **Historical Phase-0 design.** The implemented/shipped byte contract is
> [LAYOUT.md](LAYOUT.md), not the early estimates below. Production fp4 uses
> layout-v2 `4k+9`; explicit v1 remains readable at `4k+16`; FP8 adds one fp32
> scale per output row; and both fixed lattice and learned codebooks are real
> FP16 sidecar tensors shared once per `(codebook_ref, format)`. There is no CB
> global-scale scalar. Exact producer accounting lives in
> `prismaquant.nvfp4_cb_footprint` and requires an explicit serialization
> context.

Working name: **NVFP4-CB**. Vector-quantized codebook format. Codewords are
8-dim vectors of FP4 (E2M1) codes `{0,±0.5,±1,±1.5,±2,±3,±4,±6}`; per-group-of-16
E4M3 scales **identical to NVFP4** (a decoded 16-tile is bit-compatible NVFP4 and
feeds CUTLASS unchanged). Storage per 8-weight vector = one `k`-bit index into a
codebook of FP4-grid-valued 8-vectors. Effective bpw = `k/8 + 0.5`. k-ladder
k=12..24 → 2.0..3.5 bpw in 0.125 steps (13 rungs). Plain NVFP4 = the degenerate
`d=1, k=4` member (index *is* the E2M1 code; 4 + 0.5 = 4.5 bpw).

Codebook is either a **fixed structured lattice** (IQ-style generator, no sidecar)
or a **learned per-tensor** codebook (weighted Lloyd on the FP4 grid, shipped as a
sidecar tensor). Design supports **both** via a per-tensor `codebook_source`
attribute; Phase 0 measurement decides which rungs use which.

Serving is a **custom out-of-tree vLLM plugin** (separate workstream). Container is
safetensors + a compressed-tensors-**style** custom quant-config JSON with custom
scheme names only our plugin understands — **explicitly not stock
compressed-tensors** (its scheme vocabulary cannot express codebooks; do not try).

All file:line citations verified against the tree at the time of writing.

---

## 0. Reused precedent (the family this rides on)

The GGUF/IQ family is the exact structural precedent: one field-quantizer feeds
BOTH emulation QDQ and the byte packer; imatrix/col_weights weighting; exhaustive
weighted argmin over a fixed lattice; `>256M`-element stacked-expert slicing;
`scale_bits` carrying **all** non-element bytes so `effective_bits` is byte-exact.

- Field-quantizer pattern (single source → emulation + packer): `gguf_formats.py:485`
  `compute_fields`, `:388` `gguf_quantize_dequantize`, `:507` `assemble_bytes`,
  `:543` `gguf_pack`. IQ dispatch trio: `gguf_iq_formats.py:533/562/572`
  `iq_fields`/`iq_reconstruct`/`iq_assemble_bytes`.
- **The tightest precedent for VQ**: `gguf_iq_formats.py:230` `_grid_fields` — VQ
  over a *fixed vector codebook* with two-tier scale: weighted moment matrices
  `_moments` (`:263`), fused scale sweep `_sweep_errs` (`:184/199`), exact per-entry
  argmin `_pick_min` (`:208/221`), full-grid pick `pick_full` (`:267`), WLS refit
  `refit` (`:284`), 3-iter fixed point (`:318-326`). Grids are **fixed lattices**
  from `data/iq_grids.pt` (`_tables` `:70`) — **there is no learned/Lloyd path in
  the repo**; we add that.
- col_weights (Fisher/imatrix) entry: `_qw_blocks` `gguf_formats.py:373` (single
  `(in,)` or stacked `(N,1,in)`); imatrix composition `_imatrix_weights`
  `gguf_formats.py:180`, IQ `_weights` `gguf_iq_formats.py:128`; dead-group guard
  `_guard_dead_subblocks` `gguf_formats.py:169`.
- Stacked-expert / big-tensor slicing: `gguf_slice_max_elems` `gguf_formats.py:419`
  (256M k-quant / 64M IQ), pack loop `gguf_formats.py:563-574`, IQ chunking
  `iq_fields` `gguf_iq_formats.py:533-549` (`_CHUNK_ELEMS=128M`).
- FormatSpec byte-exactness: `format_registry.py:70-128` (`effective_bits`,
  `scale_count_for_shape`, `memory_bytes_for_shape`, `effective_bits_for_shape`);
  `_make_gguf_spec` factory `:826-843`; GGUF/IQ registrations `:846-864`.
- NVFP4 weight RTN routed through the export codec (resident==served):
  `_nvfp4_export_aligned_rtn` `format_registry.py:598-625`; NVFP4 spec `:629-639`;
  serve-faithful activation path `nvfp4_activation_qdq_served` `:882-915`.
- Sibling exporter template: `export_gguf.py` (512 LoC) — skeleton rewrite,
  `_map_assignment_to_gguf` (`:112`), provenance KVs (`:411-445`), strict coverage
  gates (`:218,273,307,387,398,359`). Provenance `git_commit` via
  `aura_cost._git_commit` (`:57`).
- Hard-fail-at-wrong-exporter precedent: `export_native_compressed.py`
  `_coerce_runtime_legal_assignment:1229-1241` raises when a GGUF format reaches
  the compressed-tensors exporter.

---

## 1. Bit-exact on-disk layout

### 1.1 Superblock = 256 weights (uniform across all k)

Mirror the GGUF k-quant superblock. Per 256-weight superblock along the input dim:

- **32 VQ indices** (256/8 vectors), `k` bits each → `32k` bits.
- **16 E4M3 group scales** (256/16 groups, identical byte layout to NVFP4's FP8
  block scales) → `16 × 8 = 128` bits.
- Total `type_size = (32k + 128)/8 = 4k + 16` **bytes — integer for every integer
  k** (this is why a uniform 256-superblock is chosen over a 16-weight block, which
  is byte-exact only for `k ≡ 0 mod 4`). Constraint: `in_features % 256 == 0`.

Index stream is packed LSB-first per superblock (32 k-bit fields → `4k` bytes),
then the 16 FP8 scales (`16` bytes), matching NVFP4's per-group FP8 scale bytes so
the decoder can hand the scale plane straight to CUTLASS. A **block-32 fallback
rung is deliberately out of Phase-0 scope** (unlike IQ4_NL); Linears with
`in_features % 256 != 0` fall back to a coarser legal rung or BF16 (§5).

### 1.2 Effective-bits table (registry-exact, sidecar-exclusive)

`effective_bits = type_size·8 / 256 = (4k+16)·8/256 = k/8 + 0.5`, reproduced
**exactly** by `memory_bytes_for_shape` when `256 | n_params`.

| k  | bpw (`k/8+0.5`) | type_size (B/256) | index bits/superblock (32k) | scale bits/superblock |
|----|-----------------|-------------------|-----------------------------|-----------------------|
| 12 | 2.000 | 64  | 384 | 128 |
| 13 | 2.125 | 68  | 416 | 128 |
| 14 | 2.250 | 72  | 448 | 128 |
| 15 | 2.375 | 76  | 480 | 128 |
| 16 | 2.500 | 80  | 512 | 128 |
| 17 | 2.625 | 84  | 544 | 128 |
| 18 | 2.750 | 88  | 576 | 128 |
| 19 | 2.875 | 92  | 608 | 128 |
| 20 | 3.000 | 96  | 640 | 128 |
| 21 | 3.125 | 100 | 672 | 128 |
| 22 | 3.250 | 104 | 704 | 128 |
| 23 | 3.375 | 108 | 736 | 128 |
| 24 | 3.500 | 112 | 768 | 128 |

Reference (degenerate): NVFP4 = `d=1,k=4` → `4/1 · (1/... )` collapses to 4 bpw
index + 0.5 scale = 4.5 bpw (already shipped as `NVFP4`).

### 1.3 Scale layout

Identical to NVFP4: one E4M3 (`torch.float8_e4m3fn`) block scale per 16 weights,
derived as `amax/6` snapped to FP8 (reuse the exact derivation in
`_nvfp4_export_aligned_rtn` → `enc._rtn_dequant_nvfp4`, `format_registry.py:614-622`).
Optional per-tensor global scale (the "+ negligible" term): one FP32 → 4 bytes/tensor.

### 1.4 Codebook sidecar (learned variant only)

Per-tensor blob. Entry = 8 FP4 (E2M1) 4-bit codes = **4 bytes/entry**; codebook of
`2^k` entries → `2^k · 4` bytes:

| k | learned-codebook sidecar size |
|---|-------------------------------|
| 12 | 16 KB |
| 16 | 256 KB |
| 20 | 4 MB |
| 24 | **64 MB** |

**⚠ CRITICAL OPEN QUESTION.** A per-tensor learned codebook is only viable at low
k. At k≥20 the sidecar dwarfs the weight it encodes (a 4096×4096 Linear at k=20 is
~6 MB of indices + 4 MB codebook — a 66% overhead the `k/8+0.5` bpw label hides).
Three mitigations, must pick before Phase 0 for high-k learned rungs:
(a) restrict learned codebooks to k≤~16; (b) **one shared codebook per (model,
role)** amortized to ~0 bpw (my recommendation — makes the sidecar negligible and
keeps `effective_bits` honest); (c) high-k rungs use the fixed lattice only.
The learned per-tensor path stays exact-accountable **only** if a per-tensor
footprint function adds the sidecar bytes; the registry `effective_bits`
**excludes** it (see §1.5).

Sidecar tensor: name `<linear>.cb_codebook`, dtype uint8 packed (8 codes×4 bits =
4 bytes/row), shape `(2^k, 4)`; a `codebook_source` field on the config entry is
either `{"lattice": "<generator_id>"}` (no sidecar) or `{"learned":
"<tensor_name>", "sha256": ...}`.

### 1.5 Per-tensor metadata (in the custom quant-config JSON)

Per Linear: `scheme = "nvfp4_cb"`, `k`, `superblock=256`, `group_size=16`,
`codebook_source` (lattice id or sidecar ref+hash), optional
`input_global_scale` (static activation scale, as NVFP4). Fused siblings / packed
experts share one `k` and one `codebook_source` (§5).

> **House-rule caveat, stated plainly:** the registry `effective_bits` is
> byte-exact for the **fixed-lattice** variant (no sidecar, exactly like GGUF).
> For the **learned** variant it *understates* true bytes by the sidecar. Byte
> accounting for shipped artifacts must go through a per-tensor footprint function
> (mirror `footprint.py`) that adds sidecar + per-tensor global scale — do **not**
> report the registry bpw as the shipped bpp for learned rungs.

---

## 2. FormatSpec registration (`_make_nvfp4_cb_spec` factory)

Mirror `_make_gguf_spec` (`format_registry.py:826-843`). One factory, 13 calls:

```
def _make_nvfp4_cb_spec(k: int) -> FormatSpec:
    return FormatSpec(
        name=f"NVFP4_CB_K{k}",
        weight_bits=0,                 # VQ: no per-element weight; index stream
        group_size=256,                #   lives in scale_bits, exactly as GGUF
        scale_bits=32*k + 128,         # 32 k-bit indices + 16 E4M3 scales / 256
        scale_dtype_name="nvfp4_cb_vq",
        weight_element_dtype=f"nvfp4_cb_k{k}",
        act_bits=4, act_dtype_name="fp4_e2m1", act_group_size=16,
        family="nvfp4_cb", min_capability_sm=100,   # Blackwell CUTLASS NVFP4
        quantize_dequantize=make_nvfp4_cb_qdq(k),        # §3, gguf-style closure
        activation_quantize_dequantize=_make_rtn("fp4_e2m1", 16),  # = NVFP4's path
    )
for k in range(12, 25):
    register_format(_make_nvfp4_cb_spec(k))
```

- `effective_bits` property = `0 + (32k+128)/256 = k/8 + 0.5` — exact; and
  `memory_bytes_for_shape` (`:119-124`) = `(n/256)·(4k+16)` exact when `256|n`.
  Verified against the §1.2 table.
- **Activation emulation = NVFP4's W4A4 path.** A decoded CB tile *is* NVFP4, so
  activations are byte-identical to NVFP4. Reuse `_make_rtn("fp4_e2m1", 16)` (the
  same object NVFP4 uses at `format_registry.py:638`). The serve-faithful static
  path is `nvfp4_activation_qdq_served` (`:882-915`) — wire CB to it wherever
  NVFP4 already routes there (parity, no new activation code).
- `weight_bits=0` is intentional and safe: the DP baseline is per-Linear
  min-bits (`allocator_solver.py:230`) and `memory_bytes` weight term is
  `ceil(n·0/8)=0`. (This differs from GGUF, which keeps a nominal `weight_bits`
  because k-quants *do* store per-element bits; VQ does not.)

---

## 3. Emulation `quantize_dequantize`

`make_nvfp4_cb_qdq(k)` returns a closure `(w, col_weights=None) -> w_hat` that is
the **single source** used by BOTH emulation and the packer (the gguf contract at
`gguf_formats.py:485/507`). Steps:

1. **Scales** exactly as NVFP4: group-16 `amax/6` → FP8 snap, via the export codec
   (`_nvfp4_export_aligned_rtn` path, `format_registry.py:614-622`) so
   resident==served. Scale the weight: `x = w / scale`.
2. **Vectorize** into 8-dim vectors along the input dim (256-superblock → 32
   vectors, aligned to the two group-16 scales they span).
3. **Codeword search** (weighted VQ nearest), reusing the IQ machinery:
   - *Fixed lattice:* the codebook is a fixed `(2^k, 8)` FP4-grid tensor; assign
     each vector by weighted argmin `Σ_j wq_j (x_j − cb[c]_j)²`. Reuse
     `_grid_fields`'s weighted moment/argmin kernels (`gguf_iq_formats.py:230,
     _moments:263, _pick_min:208/221, _best_kv:410` for the scalar-grid analogue).
     `col_weights` enters exactly as imatrix does in `_moments`.
   - *Learned:* **weighted Lloyd/k-means on the FP4 grid** (NEW — no repo
     precedent). Iterate: (assign) weighted argmin as above; (update) centroid =
     `col_weights`-weighted mean of assigned scaled-vectors, then **project each
     centroid coord onto the E2M1 grid** via `_rtn_fp_codebook`/`torch.bucketize`
     (`format_registry.py:247-295`) so every codeword stays FP4-valued (this is
     what keeps the decode NVFP4-bit-compatible). ~3–5 Lloyd iters; init from the
     fixed lattice for determinism.
4. **Decode** `w_hat = cb[idx] · scale`; reshape. Bit-identical to the packer's
   decode (pinned by tests, §7).
5. **col_weights source:** Fisher/imatrix per-column importance, plumbed exactly
   like the GGUF lane (`_qw_blocks` `gguf_formats.py:373`, single `(in,)` or
   stacked `(N,1,in)`); dead-group guard `_guard_dead_subblocks` (`:169`).
6. **Batched/stacked experts:** reuse the slicing precedent — `gguf_slice_max_elems`
   (`gguf_formats.py:419`) + the ≥3-D pack loop (`:563-574`) carrying per-expert
   `col_weights` slices; IQ per-chunk pattern (`gguf_iq_formats.py:533-549`).
   Blackwell UMA swap-kill (2026-07-11) motivated the 64M IQ threshold — pick a
   CB threshold from measurement, not guessed.

> **One-cache / no-confound note.** `FormatSpec.quantize_dequantize` is called
> **unweighted** by `production-render-score` (same reason the GGUF lane forbids
> it — `run-pipeline.sh:92-96`). So the *weighted* CB render must reach cost the
> same way GGUF does: **`COST_MODE=local`**, where the local cost path calls the
> weighted closure. Do not let production-render-score score an unweighted CB
> render while the exporter ships a weighted one — that is the exact rendering
> confound the house rules forbid (§6 gates enforce it).

---

## 4. Encoder: fixed vs learned; GPTQ interaction

- **Fixed-lattice path:** deterministic, no sidecar, generator id → codebook
  tensor. Encode = one weighted-argmin pass (+ the 3-iter scale fixed point from
  `_grid_fields:318-326`). This is the low-risk Phase-0 default.
- **Learned path:** weighted Lloyd (§3). Non-deterministic across BLAS orderings →
  pin a seed + fixed init (fixed lattice) for reproducibility; ship the resulting
  codebook + its sha256 (provenance, §6). Gated behind Phase-0 measurement showing
  it beats the lattice enough to justify the sidecar.
- **GPTQ / error feedback — DEFERRED, explicitly.** GGUF has GPTQ-under-frozen-
  scales (`test_gguf_gptq.py`, `render_score`'s `gptq` mechanism). OBQ's per-column
  closed-form update assumes **scalar** column quantization; VQ quantizes 8-column
  vectors **jointly**, so the per-column residual-propagation step does not map
  directly. Phase 0 ships **RTN-VQ + imatrix-weighted search only**. A block-OBQ
  extension (propagate residual across 8-col vector blocks, re-run the codeword
  search per block under the frozen NVFP4 scales) is plausible future work — do
  **not** attempt it before the emulation-only milestone lands.
- **"Everything through PrismaQuant" compliance:** the imatrix-weighted VQ search
  **is** the deliberate measured render for this format (analogous to GPTQ+JSO for
  NVFP4). CB Linears are not silently RTN'd-by-omission; `_format_supports_render_
  mechanism` (`production_weight_cache.py:1349-1376`) must return the CB-specific
  mechanism set (weighted VQ search; **not** gptq/jso/scale_sweep, which are
  scalar-column mechanisms and do not apply). Add an `NVFP4_CB` branch there.

---

## 5. Allocator / menu integration (13 rungs)

- **Menu parse:** rungs are registered names, resolved by `fr.get_format` at
  `allocator.py:1336-1341` (sorted by `effective_bits`). No change needed to
  parse; add the 13 names to the menu / serving allow-list.
- **⚠ Family-coherence gate (`allocator.py:1371-1389`):** buckets formats by
  `round(effective_bits*4)/4` (0.25-bit tiers); a 0.125-step ladder puts **2 CB
  rungs in every tier → collision**, which hard-fails under
  `--enforce-family-coherence`. **VERIFY how the existing GGUF/IQ ladder (which
  also has sub-0.25 spacing, e.g. IQ2_S 2.5625 vs Q2_K 2.625) avoids this today**
  — either the gate exempts same-`family` members or it is off by default. Fix:
  bucket per-`family` (or exempt intra-family ladders) so an intentional
  multi-rung family is legal. This gate **will** need a code change or confirmed
  exemption; do not assume it is free.
- **Candidate build (`allocator_candidates.py:305-393`):** per (Linear, k-rung)
  emits `Candidate(bits_per_param=effective_bits_for_shape, memory_bytes=
  memory_bytes_for_shape, predicted_dloss)`; cost precedence unchanged
  (`:230-294`). CB rungs need cost entries under `output_mse`/`predicted_dloss`/
  `weight_mse` keys — produced by the same weighted render used at export.
- **Legality gate (`allocator_candidates.py:70-153`):** group-16 divisibility and
  the FP8-scale rules apply as for NVFP4. **NEW constraint to add:** the
  256-superblock + 8-wide vector tiling ⇒ `in_features % 256 == 0`; no existing
  field models vector width. Cleanest: the `group_size=256` field already forces
  `in_features % 256` via the existing divisibility check (`:119-129`) — so **the
  256-superblock does double duty** (byte-exactness *and* the tiling legality),
  exactly like GGUF's 256 block. Linears failing it fall back to a coarser legal
  rung / BF16 (no block-32 CB rung in Phase 0).
- **DP (`allocator_solver.py:219-312`):** cost=`predicted_dloss`; 8→13+ options/
  Linear scales the inner loop ~1.6× — fine (`bit_precision=0.001`,
  `allocator.py:56`). Caveat: at 0.125-bpp spacing, small Linears' avg-bit deltas
  can collapse adjacent rungs into the same `dbins` (`_charged_bins:201-216`) — the
  DP keeps the higher-gain rung (correct) but fine granularity is partly cosmetic
  at the margin. Not a bug; note it.
- **Fused-sibling + packed-MoE uniformity (`allocator_solver.py:68-130,162-198`;
  `allocator_candidates.py:459-604`):** union-find promotes a mixed group to its
  **max-`format_rank`** member. With 13 CB ranks this promotes siblings **up the CB
  ladder** (to the highest-k present) — verify that is the intended coherence
  resolution (risk: over-spend). Fused aggregation intersects member format sets
  (`:556-563`), so a rung is offered to a group only if legal for **all** members.
  The `AssertionError` mixed-format post-check (`:187-197`) still guards export.
- **Passthrough integrity:** CB is **synthesized**, not passthrough → **do not** add
  it to `PASSTHROUGH_SOURCE_REQUIREMENTS` (`allocator_candidates.py:18-36`).
- **Serving profile:** new `serving_profile_specs/nvfp4_cb.json` (mirror
  `gguf.json`): `allow_formats = [NVFP4_CB_K12..K24, BF16]`, `runtime:
  "vllm_nvfp4_cb_plugin"`, a `shape_rules` entry `in_features_multiple_of: 256`
  for all CB rungs. Enforced via `serving_profiles.check_format` (`:301-434`).
  **Open question:** whether the container may *also* carry plain NVFP4/FP8/BF16
  (true mixed-precision — the whole point of PQ) or is CB-only in Phase 1. CB-only
  is simpler for the first plugin; flag mixed as a fast follow (plugin must then
  also decode stock NVFP4/FP8 schemes).

---

## 6. Container + exporter

**New sibling `export_nvfp4_cb.py`** (mirror `export_gguf.py`, ~512 LoC template —
do NOT extend the 8395-line `export_native_compressed.py`; it builds *stock*
compressed-tensors config and hard-fails on non-scheme formats).

- Reads the bf16 skeleton, maps assignment→tensor (mirror `_map_assignment_to_gguf`
  `export_gguf.py:112-145`), VQ-packs each Linear via the **same** weighted closure
  used for cost (`make_nvfp4_cb_qdq(k)` → `compute_fields`/`assemble_bytes`
  analogues), writes safetensors + a **custom** `quant_config.json` (custom scheme
  vocabulary; NOT stock compressed-tensors — its schemes can't express codebooks).
- **Custom config emitter** (bypass `build_quantization_config`
  `export_native_compressed.py:6845-7261`): emit `config_groups` keyed by CB scheme
  name + per-Linear `k`/`codebook_source`, a BF16 `ignore` list, and packed-MoE
  canonical `gate_proj/up_proj/down_proj` scheme names (same canonicalization the
  stock path does at `:7231-7252`).
- **Provenance KVs** (mirror `export_gguf.py:411-445`, under `prismaquant.*`):
  `git_commit` (`aura_cost._git_commit:57`), `assignment_sha256`, calibration hash,
  `imatrix_sha256`, **`codebook_sha256`** (per-tensor or shared), `tensor_formats`
  JSON, `imatrix_weighted`/`imatrix_fallback` counts.
- **Strict coverage gates** (mirror `export_gguf.py:218,273,307,387,398,359`):
  fail on any format not in the CB set; fail on `in_features % 256 != 0`; fail on
  any quantized tensor lacking its imatrix/col_weights vector (no silent RTN); fail
  on missing learned-codebook sidecar when `codebook_source=="learned"`.
- **Hard-fail-at-wrong-exporter:** add an `NVFP4_CB_FORMATS` membership check in
  `export_native_compressed._coerce_runtime_legal_assignment` (`:1229-1241`,
  exactly mirroring the GGUF branch) so a CB assignment reaching the
  compressed-tensors exporter raises "ships via nvfp4_cb container".
- **`layer_config.canonicalize_format`** (`layer_config.py:31-45`): add a
  `dt == "nvfp4_cb"` branch reading a `k` field → `f"NVFP4_CB_K{k}"`, mirroring the
  gguf branch (`:41-45`). Add the 13 names to `_GGUF_FORMAT_NAMES`'s CB analogue.

**`run-pipeline.sh` gates** (mirror the GGUF lane fail-fasts at `:72-105`). Under
`EXPORT_CONTAINER=nvfp4_cb`:
- ~~require `COST_MODE=local`~~ — **superseded 2026-07-30 (re-vet R3).** The
  invariant is that the cost render must be the *weighted* render the exporter
  ships; that is a property of the render, keyed off the format family, not of
  the objective. A cached-menu render now supplies `--col-weights` (Milestone C),
  so the gate is a render-faithfulness assertion instead;
- require `TARGET_PROFILE=nvfp4_cb` (exporter hard-fails on non-CB formats);
- require `PRODUCTION_CACHE=0 PRODUCTION_RECACHE=0` (exporter requantizes the bf16
  skeleton; no production cache is read — same as GGUF `:97-100`);
- default `ACTIVATION_ROWS_LIMIT=1024` (VQ codeword search wants a
  higher-rank imatrix than 256 rows, same rationale as the GGUF lane `:67-76` —
  **confirm by measurement**, don't assume).

> **Milestone C — ADOPTED and LANDED 2026-07-30 (re-vet R3).** This was filed as
> the "alternative (larger, better) path — flag not adopt", deferred past Phase 0.
> It is now implemented: `render_production_weight` and `build_production_cache`
> take `col_weights`, applied to the weighted-render families only
> (`production_weight_cache.WEIGHTED_RENDER_FAMILIES`) through the single shared
> definition `emu_forward_kl.weighted_quantize_dequantize`, so the standard
> ProductionWeightCache identity (cost == KL == bytes) holds on this lane and the
> `COST_MODE=local` restriction is **gone** — replaced by a render-faithfulness
> assertion (ARCHITECTURE.md §4.7). Native/NVFP4/FP8 renders are pinned
> **bit-identical** with and without the argument
> (`tests/test_col_weights_render_identity.py`). Phase 0's skeleton-requantize
> model still satisfies the one-cache rule the trivial way and remains what the
> exporter does; what changed is that the *cost* side is no longer forced to it.
> **Not promoted:** render-score / AURA objectives on CB are reachable, opt-in and
> non-default until a served CB objective A/B exists.

---

## 7. Test plan

Mirror `tests/test_gguf_iq_formats.py` (the load-bearing template) and
`test_format_registry.py` / `test_format_menu_expansion.py` /
`test_allocator_*`:

1. **Effective-bits accounting** (`test_effective_bits_exact` analogue,
   `test_gguf_iq_formats.py:46-52`): for each k, assert
   `spec.effective_bits_for_shape((64, 2048)) == k/8 + 0.5` (abs 1e-9) and
   `memory_bytes_for_shape == n/256 · (4k+16)`. Assert the §1.2 table.
2. **Bit-exact pack/decode pinning** (the contract): `pack(w) → decode == emulation
   qdq(w)`, element-for-element, for both fixed-lattice and learned codebooks, on
   several shapes and k. (GGUF pins against gguf-py; CB has no external reference,
   so pin **emulation-decode == packer-decode** — the two must be the same
   `compute_fields` output, like `gguf_iq_formats.py:509-514`.)
3. **col_weights plumbing:** weighted vs unweighted search change the assignment;
   assert weighting reduces weighted-MSE; assert stacked `(N,1,in)` per-expert
   weights slice correctly (`gguf_formats.py:373`).
4. **Allocator-menu tests** (`test_format_menu_expansion.py` style): 13 rungs
   register, resolve, sort by bpp; family-coherence gate does **not** false-fail on
   the intra-family ladder; a mixed fused group promotes coherently (one k);
   `in_features % 256 != 0` Linear is excluded / falls back.
5. **Exporter coverage gates:** unsupported format, `%256` violation, missing
   imatrix vector, missing learned sidecar → each raises.
6. **Per-device determinism:** fixed-lattice qdq is bit-identical CPU vs CUDA
   (note the `torch.compile` tie-flip caveat at `format_registry.py:411-414` —
   pin the eager path in bit-repro tests via `PRISMAQUANT_DISABLE_RTN_COMPILE=1`).
   Learned-Lloyd is **only** seed+init-deterministic within a device; document
   that cross-device learned codebooks may differ at tie boundaries → ship the
   codebook (don't regenerate at serve).

---

## 8. File-by-file change list + sequence

### Milestone A — EMULATION ONLY (everything through allocator+cost in pure
emulation; **no exporter, no plugin**). Ship this first and validate cost/KL on it.

| File | Change | ~LoC |
|---|---|---|
| `prismaquant/nvfp4_cb_formats.py` (NEW) | `make_nvfp4_cb_qdq(k)`; fixed-lattice generator + `data/nvfp4_cb_lattices.pt`; weighted Lloyd; `compute_fields`/`reconstruct` (gguf-style, no packer yet); slicing | 350–500 |
| `prismaquant/format_registry.py` | `_make_nvfp4_cb_spec` factory + 13 registrations; import the qdq | 30 |
| `prismaquant/layer_config.py` | `nvfp4_cb` canonicalize branch + name set | 15 |
| `prismaquant/production_weight_cache.py` | `_format_supports_render_mechanism` `NVFP4_CB` branch (weighted-VQ mechanism) | 15 |
| `prismaquant/allocator.py` | family-coherence gate: per-family bucketing / intra-family exemption (`:1371-1389`) | 20–40 |
| `prismaquant/allocator_candidates.py` | (verify only) 256-divisibility legality via existing `group_size` check; no new field if 256-superblock suffices | 0–20 |
| `prismaquant/serving_profile_specs/nvfp4_cb.json` (NEW) | allow_formats + 256 shape rule | 25 |
| `tests/test_nvfp4_cb_formats.py` (NEW) | effective-bits, emulation-decode determinism, col_weights, menu, family-coherence | 200 |

### Milestone B — EXPORTER + CONTAINER (after A's cost/KL is trusted).

| File | Change | ~LoC |
|---|---|---|
| `prismaquant/nvfp4_cb_formats.py` | `assemble_bytes` / `pack` byte packers (index bitstream + FP8 scales) | 150–250 |
| `prismaquant/export_nvfp4_cb.py` (NEW) | sibling exporter (skeleton rewrite, custom config emitter, provenance KVs, coverage gates) | 450–550 |
| `prismaquant/export_native_compressed.py` | hard-fail CB at `_coerce_runtime_legal_assignment:1229` | 10 |
| `prismaquant/run-pipeline.sh` | `EXPORT_CONTAINER=nvfp4_cb` gates (COST_MODE/TARGET_PROFILE/PRODUCTION_CACHE/ACTIVATION_ROWS_LIMIT) | 25 |
| `prismaquant/footprint.py` (or equivalent) | per-tensor exact bytes incl. codebook sidecar + global scale | 30 |
| `tests/test_nvfp4_cb_formats.py` | bit-exact **pack→decode == emulation**; exporter coverage gates; footprint | +150 |

### Milestone C — SERVING (separate workstream, out of scope here): custom vLLM
plugin decoding CB scheme → NVFP4 buffers → CUTLASS. Gate every rung on the serving
metric (exact full-vocab vLLM KL-vs-BF16 + WikiText PPL) before any rung is a
default (§9 promotion ladder).

**Ordered sequence:** A1 registry+factory+lattice → A2 emulation qdq + tests
(effective-bits + determinism GREEN) → A3 allocator menu/family-coherence + serving
profile → **A4 run allocator+cost+KL end-to-end in emulation on Qwen3-0.6B then 4B
(the emulation-only milestone gate)** → B1 packers + bit-exact pack==emulation test
→ B2 exporter + pipeline gates + footprint → C plugin.

---

## Open questions (flagged, not papered over)

1. **Learned-codebook sidecar cost at high k** (§1.4): `2^k·4` bytes is prohibitive
   at k≥20 per-tensor. Recommend a **shared per-(model,role) codebook** or
   restrict learned to low k; decide before Phase 0. The `k/8+0.5` label is
   dishonest for per-tensor learned high-k rungs.
2. **Family-coherence gate** (§5): RESOLVED by review (2026-07-15, verified
   allocator.py:1371-1389): the gate is a WARNING by default and hard-fails only
   under `--enforce-family-coherence` — that is how the GGUF ladder passes today.
   The per-family bucketing fix is still worth doing so the warning stays
   meaningful on an intentional intra-family ladder, but nothing blocks.
3. **Fused-group promotion up the CB ladder** (§5): union-find → max-rank may
   over-spend across 13 ranks; confirm intended.
4. **Mixed container** (§5): CB-only Phase 1, or CB + NVFP4/FP8/BF16 in one
   artifact (needs the plugin to decode stock schemes too). The PQ thesis wants
   mixing; the first plugin is simpler CB-only.
5. **GPTQ/error-feedback for VQ** (§4): deferred; block-OBQ is future work, not a
   Phase-0 promise.
6. **`ACTIVATION_ROWS_LIMIT` for VQ** (§6): assume 1024 by analogy to GGUF but
   confirm the VQ codeword search's imatrix rank needs by measurement.
7. ~~**One-cache via production-render-score**~~ (§6): **CLOSED 2026-07-30
   (re-vet R3, Milestone C).** The cache-integrated path is implemented —
   `col_weights` on `render_production_weight` / `build_production_cache` — so the
   one-cache identity holds on this lane without CB becoming the default. The
   exporter still requantizes the skeleton; what lifted is the restriction on
   which cost render is admissible.
8. **Does the 8-dim VQ + CUTLASS actually round-trip bit-compatible NVFP4?**
   Assumed (decoded codewords are on the E2M1 grid + NVFP4 scale layout). The
   serving workstream must confirm the plugin can materialize standard NVFP4
   packed buffers from CB indices at a *performant* kernel — else the whole
   "feeds CUTLASS unchanged" premise fails and CB is research-only.
