# NVFP4-CB two-tier scale — layout v2 scale coding (IMPLEMENTED, SHIPPED)

> **STATUS 2026-07-30: implemented and shipped; this page is the normative
> spec of the format, not a proposal.** Encoder + tables:
> `prismaquant/nvfp4_cb_formats.py` (`SCALE_CODING_TWO_TIER`,
> `TWO_TIER_SUB_TABLE`, `TWO_TIER_SUPER_BIAS`, `_two_tier_tables`). Export:
> `prismaquant/export_nvfp4_cb.py:670-676,755` (`layout_version: 2`, scheme
> carries `scale_coding.kind = "two_tier"` + the table). Serving: composition
> table in [Gridbook's `codec.py`](https://github.com/RobTand/gridbook/blob/master/gridbook/codec.py), in-kernel compose in the
> fp4-v2 dense/grouped GEMVs and `expand_fp4_v2_to_weight`. Shipped in the Hy3
> 295B and Laguna-S-2.1 artifacts (`CB_SCALE_CODING=two_tier`,
> `scripts/run_hy3_prod_nvfp4cb.sh:53`, `scripts/run_hy3_prod_joint.sh:56`,
> `scripts/run_laguna_s21_prod.sh:49`). Per `STANDARDS.md:13,22-23` v2 is the
> production fp4 scale coding and **v1 is legacy read-compat only** — and since
> 2026-07-30 (re-vet R28 / §12 D15) `run-pipeline.sh` **defaults
> `CB_SCALE_CODING=two_tier`**, so the shell default finally matches the shipped
> value. Production drivers still set it explicitly, so no shipped run changes. §6's implementation gates are
> historical; §1–§5 remain the contract.
>
> **Original motivation (as drafted).** Mitigation for the structural
> scale-packaging tax identified in `rd_ceiling_study.md` (+ its reviewer
> correction) and priced by exp-1b: NVFP4-CB's mandatory group-16 E4M3 scale
> costs **0.500 bpw** where IQ amortises a two-tier scale at **0.3125 bpw** —
> the ~**0.19 bpw** structural reason CB trails IQ at matched bytes
> (exp-1b native-FP4 premium ≈ +0.15 bpw at the IQ2_S crossing). This spec
> ships a two-tier scale (per-256 E8M0 super + per-16 4-bit sub) whose
> composition lands **exactly on E4M3 by construction**, so the tensor-core
> path still consumes a bona-fide E4M3 plane. CPU-only empirical risk check:
> `scripts/two_tier_scale_check.py` (§3).

---

## 1. Composition & legality (the core contract)

### 1.1 Stored form (per 256-weight superblock, fp4 family only)

| field | width | meaning |
|---|---|---|
| `super` | 1 × uint8 `E` | E8M0-style power of two: `2^(E-127)` (MX convention, bias 127) |
| `sub`   | 16 × 4-bit codes `c_g` | index into a fixed 16-entry table `T` of e4m3-exact multipliers |

**Reconstruction:** `scale_g = T[c_g] × 2^(E-127)` — the per-16 E4M3 scale the
CUTLASS block-scaled path consumes. FP8_CB is out of scope (no per-superblock
scale plane; per-output-channel fp32 lives in a separate tensor).

### 1.2 Exactness by construction (no rounding anywhere)

- Every table entry has the form `t = (8+j)/8 × 2^i` (an e4m3 significand
  `8+j ∈ [8,15]` times a power of two), so `t × 2^(E-127) = (8+j) × 2^(E-127+i-3)`
  — an e4m3 value whenever the composed exponent is in range.
- **Legality mask** `L[E, c]` (a precomputed 256×16 bool, trivial): pair is
  legal iff the composed value round-trips `float8_e4m3fn` bit-exactly and lies
  in `(0, 448]`. The encoder only ever emits legal pairs, therefore **the
  reconstructed plane is exact E4M3 by construction** — no cast, no rounding,
  and the emulation↔packer↔kernel parity contract (K1 discipline) holds with a
  plain fp32 multiply (`(8+j) ≤ 15` is fp32-exact; × power of two is exact).
- **Range ends.** Top: entries with `t×2^(E-127) > 448` are masked (448 itself
  is reachable: `1.75×2^8`). Bottom: composed values below `2^-6` are e4m3
  *subnormals* `m×2^-9, m∈1..7`; a pair is legal only when the product is
  exactly one of those (mantissa bits must not truncate — e.g. at
  `E-127+i-3 = -10`, only even `8+j` survive). The union over `E` of legal
  compositions still covers **every** e4m3 value (the default table carries all
  8 mantissa patterns), so nothing representable today becomes unreachable —
  the only real constraint is that one superblock's 16 scales must fit a
  ~2-octave window (measured in §3).
- **Zero / degenerate rules (deterministic bytes):** `T` contains no zero, so a
  composed scale is never 0 (matches today's `_E4M3_MIN_POS` floor — the
  current sweep also never emits 0). An all-zero *group* has zero error at any
  scale → argmin takes the first legal candidate (deterministic). An all-zero
  *superblock* stores `E = E_floor` (the smallest `E` with any legal entry) and
  all-zero sub codes. Groups whose ideal scale sits below the superblock's
  reachable set snap **up** to the smallest legal reachable — the no-clip
  direction (weights round toward zero; error bounded), same failure direction
  as today's floor.
- **Why an E8M0 super and not fp16 (the reviewer's sketch):** an fp16 super ×
  multiplier needs a *cast* to e4m3 at compose time — hand-rolled RN-to-even
  inside the Triton/CUTLASS consumers to stay bit-exact with the emulation.
  A power-of-two super makes compose a LUT read + exponent add (fp32-exact),
  costs half the bytes (1 B vs 2 B per 256), and loses nothing: the mantissa
  freedom the fp16 super would carry is exactly what the sub-table already
  provides. The IQ trick is kept; the container is made e4m3-native.

### 1.3 Default sub-table (a spec constant, shipped in the config)

`T4_2oct8m = {1.0, 1.125, …, 1.875, 2.0, 2.25, …, 3.75}` — all 8 e4m3 mantissa
steps × 2 octaves. Rationale: within a superblock the sub carries full e4m3
granularity (~6–12% steps) over a 3.75× span; the super absorbs placement.
Alternative `T4_4oct4m` (4 mantissas × 4 octaves, span 14×) trades granularity
for span; the 5-bit fallback `T5_4oct8m` (32 entries, span 15×, `type_size
4k+11`) buys both at +0.0625 bpw. **§3 measured all three on real tensors:
`T4_2oct8m` is the confirmed default; 5-bit is not needed (§3.2).** The table
ships as 16 floats in `quant_config` (§5), so it is
self-describing and per-artifact tunable without a layout change; entries are
asserted e4m3-exact at pack time.

### 1.4 Encoder = the existing sweep with a restricted candidate set

`_sweep_encode` today: 16 E4M3-snapped clip-level candidates per group
(`amax/L, L∈[6,4]`) → argmin weighted real-domain error → 2 WLS refits snapped
to free e4m3, accepted per group iff strictly better. The two-tier encoder is
the same machinery with the candidate set = the **reachable set**:

1. Per superblock, sweep `E` over the window `[E_lo, E_hi]` derived from the
   min/max ideal group scales (`amax_g/6`). Outside the window every group's
   nearest reachable moves monotonically away from every ideal, so error is
   non-decreasing — the windowed sweep is exhaustive-equivalent (no heuristic).
   Observed window size ~4–6.
2. Per `(E, group)`, argmin over the ≤16 legal `T[c]×2^(E-127)` via the
   existing `_eval_candidate` (weighted, original-domain, VQ-in-the-loop —
   unchanged).
3. Pick `E` per superblock by total weighted error; WLS refits snap to the
   frozen-`E` reachable set, accept-iff-strictly-better (unchanged contract).

Cost: `|window|×16 ≈ 64–96` candidate evals vs ~20 today → **~3–5× the sweep
stage**. Signed-S16 encode 0.3 s/Linear → ~1–1.5 s/Linear; acceptable for the
product/signed lanes (full-k16 is footprint-only anyway). The 4B encode-cost
measurement (`serve_prototype_4b.md` §5, candidate-histogram) may later justify
pruning the window (unimodal early-exit), gated on measured equivalence.

---

## 2. Byte accounting (exact)

Scale plane per 256-weight superblock: **16 B → 9 B** (1 super + 8 sub bytes);
`type_size(v2) = 4k + 9` (5-bit sub: `4k + 11`). Integer bytes for every k;
superblocks stay byte-aligned.

- scale bpw: `0.500 → 9·8/256 = 0.28125` (**Δ −0.21875 bpw**); 5-bit: 0.34375.
- vs IQ2_S's two-tier 0.3125 bpw: CB becomes **0.03125 bpw cheaper than IQ**
  on scales (1 B E8M0 super vs 2 B fp16 `d`).
- `effective_bits(fp4, v2) = k/8 + 0.28125`.

### 2.1 Ladder (body bpw) next to the IQ anchors

| k | v1 (=k/8+0.5) | **v2 4-bit** | v2 5-bit | type_size v2 (B) | IQ twin (bpw, Δ vs v2-4b) |
|---|---|---|---|---|---|
| 12 | 2.000 | **1.78125** | 1.84375 | 57 | — |
| 13 | 2.125 | **1.90625** | 1.96875 | 61 | — |
| 14 | 2.250 | **2.03125** | 2.09375 | 65 | IQ2_XXS 2.0625 (−0.03125) |
| 16 | 2.500 | **2.28125** | 2.34375 | 73 | IQ2_XS 2.3125 (−0.03125) |
| 18 | 2.750 | **2.53125** | 2.59375 | 81 | IQ2_S 2.5625 (−0.03125) |
| 20 | 3.000 | **2.78125** | 2.84375 | 89 | — |
| 24 | 3.500 | **3.28125** | 3.34375 | 105 | IQ3_XXS 3.0625 (no twin) |

**The whole IQ2 ladder acquires exact twins**, each 0.03125 bpw *cheaper*:
K14↔IQ2_XXS, K16↔IQ2_XS, K18/S18↔IQ2_S — and S18's magnitude table (m=10,
1024 entries) equals IQ2_S's grid size, making it a matched-bytes AND
matched-codebook-size twin (the RD study's matched-size result, B_mag/IQ ≈ +5%
mean, becomes directly load-bearing there).

### 2.2 What matched-bytes CB-vs-IQ becomes if quality holds

exp-1b (0.6B, conservative product-mode curve): CB reaches IQ2_S KL at
**2.71 bpw** (premium +0.15). Two-tier moves every CB point **left by
0.21875 bpw at unchanged KL** (same indices, cheaper scales) *iff* the scale
tax ≈ 0 (§3): crossing → **≈2.49 bpw < IQ2_S's 2.5625** — the premium flips to
**≈ −0.07 bpw** (CB matches IQ2_S quality *below* IQ2_S bytes while decoding
native FP4). Sensitivity: locally d(lnKL)/d(bpw) ≈ −2.2/bpw on the exp-1b
curve, so a scale-coding KL tax of (1+x) shifts the crossing right by
ln(1+x)/2.2 bpw: the flip to a *negative* premium survives x ≲ **+17% KL**;
even at +40% KL the premium is ≈ +0.08 bpw, still ~half of v1's +0.15. §3
measured the tax **negative** (−3.4% to −5.2% weighted recon error), so if
anything the crossing moves further left.
Honest limits: the IQ3_XXS crossing only improves 3.45 → ≈3.23 vs 3.062
(premium **+0.17 bpw survives** — two-tier does NOT fix the 3-bpw band, where
the deficit is index-rate, not scales); and the shift-the-curve argument
assumes KL responds to scales like MSE does — the 4B/served rerun is the
arbiter (§6 G1).

### 2.3 GB at artifact scale

Δ0.21875 bpw = 27.3 MB per 1e9 weights: 0.6B ≈ 16 MB, 4B-class ≈ 0.1 GB,
27B-class ≈ 0.7 GB, 295B (Hy3) ≈ **8.1 GB** — and vs IQ the scale overhead
goes from +0.19 bpw (≈ +7 GB on 295B, the reviewer's number) to **−0.03 bpw
(≈ −1.2 GB, cheaper than IQ)**.

---

## 3. Quality risk, bounded on CPU (real Qwen3-0.6B tensors)

`scripts/two_tier_scale_check.py` — CPU-only (`CUDA_VISIBLE_DEVICES=""`
asserted; the GPU belongs to the serving benchmark). Instrument: product-k16,
FIXED lattice codebook identical in every arm (isolates scale coding from the
codebook axis), real exp-1 imatrix (E[x²], seed 0) as col_weights, 512-row
slices of layer-6 `q_proj` / `gate_proj` / `down_proj`, shipping free-sweep
(16 candidates + 2 WLS refits) as baseline, two-tier arms via the same
`_eval_candidate` machinery with the §1.4 windowed-exhaustive selection +
snap-refit. Reachability asserted (`e4m3_exact(best_s).all()`).

### 3.1 Results (weighted recon error, lower is better; "tax" = two-tier vs free-sweep)

| tensor (512-row slice) | one-shot | free sweep (ships) | **T4_2oct8m** | T4_4oct4m | T5_4oct8m (5-bit) |
|---|---|---|---|---|---|
| L6 q_proj (512×1024)    | 7.847  | 6.967  | **6.607 (−5.2%)** | 6.625 (−4.9%) | 6.565 (−5.8%) |
| L6 gate_proj (512×1024) | 26.712 | 23.245 | **22.343 (−3.9%)** | 22.434 (−3.5%) | 22.187 (−4.6%) |
| L6 down_proj (512×3072) | 3.232  | 2.917  | **2.819 (−3.4%)** | 2.799 (−4.0%) | 2.795 (−4.2%) |

Unweighted deltas: −1.7% to −2.5% (same sign everywhere). Reachability assert
passed (every chosen scale is exactly e4m3 and of the form `T[c]×2^E`).

**The "tax" is NEGATIVE — two-tier scale coding is error-neutral-to-better
than the shipping encoder while saving 0.219 bpw.** The mechanism (verified in
the scale statistics, not conjectured): **89–98% of real ideal group scales
(`amax/6`) sit in e4m3's SUBNORMAL band** (< 2^-6; `subnormal_frac`
0.936/0.891/0.984 per tensor; the *chosen* v1 scales are 82–95% subnormal),
where bare e4m3 has 1–3 significant bits (~14–100% relative steps). There the
v1 free sweep's 16 clip candidates (`amax/L, L∈[6,4]`, a 1.5× span) collapse
after snapping to ~1–2 distinct values — the shipping "free" plane is
candidate-starved exactly where real LLM weights live. The two-tier windowed-E
sweep explores every reachable value across ~4 octaves per superblock
(weighted-VQ-validated, so aggressive clip levels win only when they truly
lower error) — the E8M0 super restores the per-block renormalization that the
CB container's bare-e4m3 plane lost when it dropped NVFP4's per-tensor global
scale. Within-superblock ideal-scale spread (p50 ≈ 1.2–1.3, p99 ≈ 2.1–2.8
octaves) sits inside the table-span + E-window coverage, so the 2-octave
default table suffices.

### 3.2 Verdict for the gates

- **G2 PASSES with margin inverted:** default 4-bit `T4_2oct8m` is −3.4% to
  −5.2% (improvement), far inside the ≤ +3% gate. **5-bit is NOT needed**
  (extra ~−1% is not worth +0.0625 bpw). `T4_2oct8m` confirmed default (wins
  2 of 3 tensors vs `T4_4oct4m`; near-tie on down_proj).
- Conservatism note: best-E landed on the sweep-window edge for 9–16% of
  superblocks (T4 tables) — a wider window could only improve two-tier
  further; the reported numbers are lower bounds on its advantage.
- Honest limits: (a) 0.6B tensors, one imatrix seed, product-k16 fixed-lattice
  instrument (same codebook both arms — deltas isolate scale coding);
  (b) the negative tax is measured vs the SHIPPING encoder, not vs a
  hypothetical free-e4m3 encoder given the same 4-octave window (unbuilt
  anywhere; by the subset argument it would be ≥ two-tier, but in the
  subnormal band the free set barely exceeds the reachable set, so the gap is
  small); (c) MSE proxy — the 4B served rerun stays the arbiter (G1).
- Side finding worth recording: the subnormal statistics expose that the v1
  bare-e4m3 plane (no global/super normalizer) is granularity-starved on real
  weights — two-tier is not merely a compression of the v1 plane, it is a
  better-conditioned encoding of it. This also predicts the v1→v2 KL delta at
  matched indices should be ≤ 0 (v2 dominates pointwise on recon error).

---

## 4. The resident-vs-disk trap (owned explicitly)

The retired low-bit lane's lesson (`v2_serving_memory_footprint.md`: 92.9 GB disk → 115.7 GiB
resident, OOM) replayed on scales: **reconstructing the e4m3 plane at LOAD
saves disk only** — the resident plane returns to 0.5 bpw and the allocator's
fit-the-card budget (resident bytes) gains nothing. Worse, the current
prototype already demonstrates the anti-pattern in miniature:
`plugins/.../linear.py` pre-decodes the plane to a resident **fp32** tensor
(`layer._cb_scale`, 32 b/16 w = 2.0 bpw of resident scales *on top of* the
packed 0.5 still inside `cb_qweight`). Harmless at 0.6B bring-up; disallowed
for v2 (§6 G4).

Serving modes, specced:

- **M0 — load-time recompose (disk-only stepping stone).** Loader expands
  9 B → 16 B per superblock (or worse, fp32). Honest use-case: **download/HF
  footprint only** (−8.75% artifact at K16). `effective_bits` and every
  allocator/footprint/bpw claim MUST stay at v1 numbers in this mode — no
  2.28-bpw label on a 2.5-bpw-resident serve. Verdict: **not worth shipping
  alone**; specced as an env-gated bring-up fallback
  (`PRISMAQUANT_CB_SCALE_RECOMPOSE=1`), never a ship config.
  **NOT IMPLEMENTED (verified 2026-07-30, R28): `PRISMAQUANT_CB_SCALE_RECOMPOSE`
  has zero readers anywhere in the tree — this bullet is a spec for unbuilt
  work, not a description of a live flag. Nothing sets it and nothing reads it;
  it is deliberately absent from `docs/design/runtime_flags.md`.** (Building the
  packer/layout is still sequenced first — it is required infrastructure for
  M1 — but v2 artifacts do not ship until M1 serves them.)
- **M1 — kernel-native (the actual win, required for v2 artifacts).**
  - **(a) Triton decode GEMV** (M ≤ 16 path): today the tuned kernel loads the
    16 scale bytes once per superblock (hoisted; `serve_prototype_4b.md` §2).
    v2 loads **9 B instead of 16** and composes: nibble → 16-entry LUT
    (registers/const) → `× 2^(E-127)` (fp32-exact; `tl.exp2` on the int-derived
    exponent or ldexp-style bit math) — ~3–5 ALU per 16 weights on a
    **memory-bound** kernel whose recent wins were byte-load reductions. Scale
    bytes −44%; total weight stream at K16 −8.75%. **Prediction:
    neutral-to-faster decode**; gate G3 requires ≥ v1 − 2%.
  - **(b) Transient-expansion prefill** (the K2-validated pattern): the
    per-layer expansion pass additionally composes this layer's e4m3 plane
    into the transient buffer (9 B → 16 B per superblock, 1/8 of the index
    stream — trivial vs the [N,K] value tile; freed per forward; the 9.4 MiB /
    0.0-growth INV-1 verification pattern applies unchanged). This is also
    where the **swizzled SF layout** for a future CUTLASS block-scaled prefill
    gets written — compose-during-expansion means **zero CUTLASS surgery**.
  - **(c) CUTLASS SF iterator (endgame, deferred).** A custom scale-factor
    iterator LUT-composing inline is possible but is deep sm_121 CUTLASS
    surgery; unnecessary while (b) covers prefill and (a) covers decode.
    Revisit only if a no-transient fused-expand prefill (prototype iii) lands.

Resident contract (asserted by a load gate): v2 resident scale bytes ==
packed 9 B/superblock; no fp32 plane, no e4m3 recompose outside a transient.

---

## 5. Migration & compatibility

### 5.1 LAYOUT.md delta (to apply on implementation)

- §1 superblock (fp4, v2): `[ INDEX 4k B | SUPER 1 B (E8M0, bias 127) |
  SUB 8 B (16×4-bit, group g in byte g/2, even g = low nibble — LSB-first,
  consistent with the index stream) ]`; `type_size = 4k + 9` (5-bit sub:
  10 sub bytes, `4k + 11`).
- §1.2 becomes "scale coding v1 (e4m3-direct, 16 B)" vs "v2 (two-tier, 9 B)";
  reconstruction formula + legality mask reference (§1 here).
- §1.3 type_size table gains the v2 column (57/61/65/73/81/89/105 for
  k=12/13/14/16/18/20/24).
- §4 scheme gains `"scale_coding": {"kind": "two_tier", "sub_bits": 4,
  "super_bias": 127, "table": [16 e4m3-exact floats]}` and the top-level
  config gains `"layout_version": 2`. **Absence of `scale_coding` ⇒ v1** —
  old artifacts parse unchanged, forever. The packer asserts
  type_size-vs-version consistency, so a mis-labeled artifact fails loudly at
  load, not silently.
- `effective_bits` becomes version-keyed: v1 `k/8+0.5`, v2 `k/8+0.28125` —
  and a v2 artifact may only ship when served M1-native (§4), so the
  allocator's resident-byte assumption is true by construction.

### 5.2 Change list (estimates)

| # | file | change | LoC |
|---|---|---|---|
| 1 | `prismaquant/nvfp4_cb_formats.py` | table + legality mask, two-tier candidate set, per-sb E window in `_sweep_encode`, fields carry `(super, sub)` | ~90–120 |
| 2 | same (packers) | `_scale_plane_bytes` v2 + unpack v2 + versioned `_type_size` | ~50–70 |
| 3 | `prismaquant/export_nvfp4_cb.py` | scheme fields, `layout_version`, table shipping + e4m3 assert | ~25 |
| 4 | `plugins/.../linear.py` | v2 dispatch; delete the fp32 `_cb_scale` pre-decode for v2 (pass packed) | ~30 |
| 5 | `plugins/.../kernels.py` | 9 B scale read + LUT/exp2 compose in the decode kernel | ~40–60 |
| 6 | `plugins/.../expand.py` | compose plane during (future fp4) transient expansion | ~15 |
| 7 | tests (formats + plugin) | T1–T6 below | ~130–160 |

Total ≈ **380–470 LoC**, no new dependencies, no CUTLASS changes.

### 5.3 Test plan

- **T1 compose-exactness (exhaustive):** all 256×|T| `(E, c)` pairs — legal
  pairs round-trip e4m3 bit-exactly; encoder fuzz never emits an illegal pair.
- **T2 parity:** pack→unpack→reconstruct == emulation reconstruct bit-exact
  for v2 (extends the existing pinned parity test), all three modes.
- **T3 byte accounting:** v2 `type_size` / `effective_bits` asserted; packed
  nbytes match on real shapes (incl. `in_features` edge multiples).
- **T4 v1 regression:** a pinned v1 fixture (no `scale_coding` key) loads and
  reconstructs unchanged.
- **T5 determinism:** CPU encode twice ⇒ identical bytes (scale path has no
  atomics — must be exactly deterministic, unlike the documented CUDA-Lloyd
  tie caveat).
- **T6 edges:** all-zero group / all-zero superblock; amax at the 448 edge;
  a subnormal-band tensor (weights ~1e-3) asserting the snap-up-no-clip rule
  (chosen scale ≥ ideal ⇒ |codes| ≤ 6, no clipping).

---

## 6. Recommendation & go/no-go

**Recommendation: GO to implementation, sequenced AFTER the pending 4B fills**
(`serve_prototype_4b.md` §4), because two-tier is only worth building if the 4B
gate keeps the sub-3-bpw lane alive — and it flips the exp-1b verdict from
"CB matches IQ2_S at +0.15 bpw" to "CB matches IQ2_S at −0.07 bpw while
decoding native FP4" for ~400 LoC and no new kernel *architecture*.

Numbered gates (ALL required for v2 default-on; any failure → recorded in
PLAN.md and v1 stays):

1. **G1 (4B served gate, pending):** the corrected 4B emu+served results keep
   the sub-3-bpw NVFP4_CB lane GO with a premium vs IQ2_S ≤ ~0.25 bpw. If the
   4B premium blows past the 0.219 bpw the mitigation buys, two-tier cannot
   pay for itself — defer.
2. **G2 (scale-tax gate, THIS spec §3): MEASURED — PASSED.** Default-table
   two-tier "tax" came back **−3.4% to −5.2%** (an improvement; gate was
   ≤ +3%) on real 0.6B tensors with the real imatrix. The 5-bit fallback
   (bytes in §2, saving drops to 0.15625 bpw) is specced but NOT needed.
   Residual risk moves entirely to G1/G5 (does the MSE-neutral coding stay
   KL-neutral served — expected, since v2's chosen scales dominate v1's
   pointwise on weighted recon error).
3. **G3 (kernel gate, serving-bench owner):** v2 decode tok/s ≥ v1 − 2% on the
   0.6B/4B suite (prediction: ≥ v1, the stream shrinks 8.75%); prefill via
   transient-compose shows no TTFT regression beyond noise.
4. **G4 (INV-1 resident audit):** load-gate assertion that v2 resident scale
   state is the packed 9 B/superblock — no fp32 plane, no load-time e4m3
   recompose in a ship config (M0 stays env-gated bring-up only).
5. **G5 (parity):** T1–T6 green, incl. bit-exact emulation↔unpack↔kernel
   parity on served metal (the K1/K2 discipline that caught the fused-kernel
   drift is reused as-is).

Non-goals (explicitly NOT fixed here): the 3-bpw-band premium vs IQ3_XXS
(+0.17 bpw survives — index-rate, not scales); FP8_CB (no scale plane); the
k>14 smem/computed-codebook constraint; codebook quality (orthogonal axis);
and the KL-vs-MSE mapping (4B/served is the arbiter, per the house rule).
