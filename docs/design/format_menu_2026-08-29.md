# The definitive format menu — 4-bit and 8-bit

**Status:** measured verdict for the 4-bit **and** 8-bit menus, joined on one
corpus under one metric and reported only where two price brackets agree.
**Date:** 2026-08-29.
**Scope warning up front:** every quality number below is measured on a corpus
of **24 GLM MoE expert tensors** (w1/w2/w3 × 8 layers) under the **routed
production imatrix**. Dense and attention tensors are not in that corpus. A
corpus hull is corpus-scoped; see §6.

---

## 1. The verdicts, in one place

| Question | Answer | Evidence |
|---|---|---|
| Can the fp4 lattice CB be retired? | **15 of 18 rungs: yes**, under *both* price brackets, now against all four families over 1.30–6.50 bpw. K2 and K5–K18 are never selected. | §6.3 |
| What survives? | **K1 and K4**, plus **contested K3**. All three sit *below* the 1.283 bpw trellis floor — the fp4 lattice CB has **no surviving rung anywhere the trellis can reach**. | §6.3 |
| Can the fp8 lattice CB be retired? | **No — it owns the top of the menu.** K48 is selected under both brackets; K28/K32/K36 retire under both; K40/K44 contested. | §6 |
| Why do they survive? | Not quality — position. They are the only rungs that exist below the 1.283 bpw trellis floor. Under an unweighted metric **zero** CB rungs are ever selected. | §3.2 |
| Can the trellis go below 1.283 bpw? | **No — structurally.** The v1 wire requires ≥1 coded bit per weight. | §2 |
| Does the trellis have a runtime kernel? | **Yes, and it is landed** (gridbook `1a57b31`, ABI 3). In-tree: **5.37×** the scan-free decoder on E2M1, **2.46×** on E4M3, **5.80×/3.18× work per joule**; 24/24 external golden vectors through the landed path; **rate-insensitive**. Landed as *research ops* — no runtime-contract cell, so every trellis rung is still **unbacked** until a release attests a route (principle 14). | §5 |
| Is the low-rate FP8 trellis viable for AMD / pre-Blackwell? | **Yes.** Decode + `_scaled_mm` is 2.22× faster than the BF16 matmul it replaces, decode cost included. | §4 |
| Is the 8-bit trellis competitive with the FP8 books on quality? | **Measured. Crossover ≈5.0 bpw** — trellis +2.3 dB at 4.0, book +7.8 dB at 6.0. Books measured are *fixed lattice*, not learned. | §6 |
| Does the 8-bit book beat raw FP8? | **Yes.** `fp8_cb@48` = 34.15 dB at 6.008 bpw vs a plain e4m3 cast at 31.57 dB and 8.008 bpw — **+2.57 dB at 75% of the bytes**. | §6.1 |
| Are the high E4M3 trellis rungs usable? | **Still no, and the cause is now known.** The clipping was a *Lloyd local optimum*, not E4M3: an exact grid DP gains +9.05 dB as a scalar book. But it does **not** transfer to the trellis — +2.45 dB at R6 on the shipping bracket, *negative on every rung of the penalty bracket*, and negative at R2 on both. R6 still sits ~2.7 dB below its own rate-8 bypass. Retirement **weakened, not overturned**; `exact_dp` stays research. | §6.2 |

---

## 2. The trellis floor is 1.283 bpw, and it is structural

`gridbook.trellis.wire.v1` cannot encode below **1.0 body bit per weight**:

- `validate_schedule` rejects any column whose rate is outside `1 <= rate <= native_bits(family)` — `gridbook/trellis.py:264`.
- `derived_schedule` refuses `q256 < SUPERBLOCK`, i.e. fewer than 256 body bits per 256 weights — `gridbook/trellis.py:232`.

This is not a tuning limit. Each coded position stores exactly one input bit
`u`, and the 256-state register is built from the `u` bits of the previous 8
coded columns (`MEMORY_ORDER = 8`, `gridbook/trellis.py:42`). Remove `u` and the
state machine has no input. **Rate ≥ 1 is intrinsic to a rate-1/1 tail-biting
convolutional code with per-position input.**

With production `two_tier` scale coding the cheapest realisable rung is
**1.28302 bpw** (rate 1.0 + 0.28125 bpw scale plane).

Going below would require a *different construction* — e.g. a rate-1/2 code
where one input bit drives two weights. That is a new wire family (v2), not a
parameter of this one. §3 quantifies exactly what such a family would buy.

---

## 3. The 4-bit menu — definitive

### 3.1 Reachability is not selection

The corpus ladder answers *reachability*: is a rung on a lower convex hull. It
cannot answer *selection*, because the cheapest rung is the leftmost hull point
**by construction** — `cb_two_tier@1` is a vertex on 24/24 tensors whether or
not any budget ever visits its λ band.

So the question was asked directly. For each of 24 tensors, solve the
per-tensor Lagrangian the allocator actually solves —
`cost_i(K) + λ · bits_i(K)`, with `cost` the absolute weighted SSE — and bisect
λ to hit a real byte budget. Sweep budgets 1.30 → 2.50 bpw in 0.05 steps.
Ties break to the *cheaper* rung, which is the conservative choice for a
retirement argument: it hands every tie to CB, so a CB rung that still fails to
appear is genuinely unused.

`/home/rob/dq-runs/trellis-hull-20260828/verdict/budget_selection_sweep.py`

**Result — never selected at any budget in 1.30–2.50 bpw:**

> `cb_two_tier@` 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18

**Selected somewhere:** `cb_two_tier@` 1, 3, 4 — and all six trellis rungs.

At 2.30 bpw and above the selection is **pure trellis**; no CB rung appears at
all. The GLM5.3 single-Spark target (≤2.7 bpw) sits entirely in that region.

### 3.2 The survivors survive on position, not merit

K1 / K3 / K4 are at 0.40625 / 0.65625 / 0.78125 bpw — **all three below the
1.283 bpw trellis floor**. They are picked for tensors the routed imatrix
judges nearly irrelevant, where the allocator wants to spend almost nothing.
K1's own quality is 0.38 dB SNR: it removes 8.4% of the energy.

The sensitivity check makes this explicit. Re-run the identical sweep against
the **unweighted** SSE and **not one CB rung is ever selected** — 0 of 18. The
three survivors exist *only* because the routed imatrix concentrates importance
hard enough that starving the unimportant tensors to sub-0.8 bpw pays.

`budget_selection_sweep_PLAIN.json` holds that arm.

### 3.3 The shipping 4-bit menu

| Rung | bpw | Role | Status |
|---|---|---|---|
| `NVFP4_CB_K1` | 0.40625 | sub-floor filler | **KEEP** — sole rung below trellis floor |
| `NVFP4_CB_K3` | 0.65625 | sub-floor | **CONTESTED** — kept here (4-bit axis), but dropped by the production bracket once the fp8 lanes join the menu (§6.3) |
| `NVFP4_CB_K4` | 0.78125 | sub-floor | **KEEP** — selected, 9/24 tensors |
| `NVFP4_CB_K2`, `K5`–`K18` | 0.53125, 0.90625–2.53125 | — | **RETIRE** — never selected |
| `TCQ_E2M1_R256` @1.0 | 1.28302 | low-bit body | KEEP |
| @1.25 | 1.53305 | | KEEP |
| @1.5 | 1.78305 | | **CONTESTED** — kept here (4-bit axis); bracket-dependent on the joint menu (§6.3) |
| @1.75 | 2.03305 | | **CONTESTED** — kept here (4-bit axis); bracket-dependent on the joint menu (§6.3) |
| @2.0 | 2.28305 | | **RETIRED under both brackets** once the fp8 lanes join the menu (§6.3) |
| @2.25 | 2.53305 | | KEEP |

On the 4-bit-only axis the trellis owns the entire 1.283–2.533 bpw band
outright; on the joint menu it keeps @1.0/@1.25/@2.25 under both brackets and
cedes @2.0 (§6.3). The fp4 lattice CB
degenerates from an 18-rung ladder to a **3-rung sub-floor tail** — and to a
two-rung tail once the fp8 lanes are on the same axis (§6.3).

If a v2 sub-1.0-bit trellis family were built, it would displace those last
three and the fp4 lattice codebook could be retired completely.

> **This subsection is the 4-bit-only axis, kept as the reproduction of the
> published study.** The joint four-family verdict is §6.3, and it is the one to
> cite: `fp8_ladder.py`'s E2M1 control reproduces this table's arm C
> **bit-identically** (`SELFCHECK PASS: 96/96`, rel = 0.000e+00), which is what
> licenses joining the two files at all.

---

## 4. The portability lane — low-rate E4M3 trellis on W8A8

Requirement: *"for AMD and pre-Blackwell compatibility I'd like low bit rate fp8
trellis to be supported."* **Proven today.**

`TCQ_E4M3_R256` carries exactly one positive fp32 scale per output row, which is
a structural match for the `scale_b` of a rowwise scaled FP8 GEMM. So the lane
is a *consumer swap*, not a new mainloop: decode to a contiguous
`[N,K] float8_e4m3fn` tile, hand it to the scaled GEMM.

The probe deliberately calls **`torch._scaled_mm`**, not vLLM's Blackwell-only
`cutlass_scaled_mm` — the whole point is a path that also exists on Ada, Hopper
and AMD via hipBLASLt.

**Measured, lina (GB10), N=K=4096, M=512 tokens, q256=512 (2.0 body bpw):**

| arm | ms/iter | mean W | envelope | MW/J |
|---|---|---|---|---|
| decode tile only | 0.19650 | 75.9 | 54.2% | 1125 |
| scaled_mm only | 0.10175 | 91.4 | 65.3% | 1804 |
| **decode + scaled_mm** | **0.31271** | 86.3 | 61.7% | 621 |
| BF16 reference matmul | 0.69526 | 45.4 | 32.4% | 531 |

**2.22× faster than the BF16 matmul it replaces, decode included**, at
1.17× the work per joule. The physics is coherent rather than coincidental: the
BF16 arm sits at 32% of the ~140 W envelope because it is memory-bound reading
32 MB of weights; the trellis lane reads an 8×-smaller wire and reaches 62%.

Numerical agreement with the fp32-expanded reference is 3.2% max relative
error, which is **activation** quantization only — the weights are identical in
both arms.

ABI note worth recording: rowwise `_scaled_mm` requires `scale_a` of shape
`(M,1)` and `scale_b` of shape `(1,N)`, both contiguous; mixing a tensorwise
`scale_a` with a rowwise `scale_b` is refused. Per-token activation
quantization is what a real W8A8 lane does anyway, so this is the faithful
contract, not a concession.

**Honest label:** this is a **lane-feasibility probe on synthetic wires at
M=512**, not a serving result. It is prefill-shaped; decode-shaped traffic
rides the fused GEMV instead (§5).

---

## 5. The runtime kernel — the trellis has one now

The trellis had never had a production decoder. The research kernel
(`gridbook/csrc/trellis_r256.cu`) re-read an int64 plan per row and walked bits
byte-serially — roughly 150 B of memory traffic per 2-bit weight, and a plan
footprint (300,032 B) 10× the size of the payload it described.

Two replacements were written. **v2 is warp-resident**: one warp owns one
(row, superblock), 32 lanes × 8 columns, and the u-bit history mask is built
with 8 `__ballot_sync()` calls — no CTA barrier and no shared-memory atomics on
the all-coded path. Body words are staged bit-aligned to the block start, so no
byte shift survives into the decode. It also exposes a fused GEMV that never
materialises a weight.

**Correctness.** Both kernels reproduce all **24 external golden vectors** —
`native_codes_sha256` and `native_packed_sha256` digests generated 2026-08-25 by
independent Stage-3/Stage-5 references, *not* by `gridbook.trellis`
(`tests/test_trellis_stage_reference_vectors.py`). Coverage includes the
minimum tail-biting block (T′=8), short blocks at 17/31/63/127/129/255/264/265/
300/511 columns, both wire layouts, and E4M3 shaped rates 3–6 with rate-8
bypass. `24 cases, 0 failures`. Expand is **bit-identical** to the reference in
both fp32 and bf16.

**Measured, lina (GB10), 1024×4096, E2M1 q256=512, 20 s sustained arms:**

| arm | ms/iter | speedup | mean W | envelope | MW/J |
|---|---|---|---|---|---|
| research_native_packed | 0.45060 | 1.00× | 54.9 | 39.2% | 169 |
| fast v1 native | 0.09852 | 4.57× | 55.4 | 39.6% | 769 |
| **v2 native** | **0.06692** | **6.73×** | 71.9 | 51.3% | **872** |
| v2 expand bf16 | 0.07084 | 6.36× | 74.0 | 52.8% | 800 |
| v2 fused gemv | 0.07100 | 6.35× | 75.4 | 53.9% | 783 |
| research expand fp32 | 4.89032 | 0.09× | 18.1 | 13.0% | 47 |
| research dequant gemv | 9.22118 | 0.05× | 18.2 | 13.0% | 25 |

**5.15× the work per joule** of the research decoder. Per principle 15 the
utilization column is omitted deliberately: it read 96% on *both* sides of this
change and would have read 96% on a decoder ten times worse. Power against the
envelope is the signal, and it moved 39% → 51%.

**Matched-bytes comparison, same box, same semantics:** gridbook's own bench
gives `matched_codebook_expand` (NVFP4_CB_K18) at **0.07139 ms**. v2 expand
bf16 is **0.07084 ms** emitting the same scaled-BF16 output — at or below CB
parity, while the trellis wire is **22,414 bytes smaller** than the CB rung it
is matched against.

**E4M3 on the same kernel, same box** (lina, 1024×4096, q256=1152 → 4.512 bpw,
parity `all_green`):

| arm | ms/iter | speedup | mean W | env% | MW/J |
|---|---|---|---|---|---|
| research_native_packed | 0.18367 | 1.00× | 79.03 | 56.5% | 289.0 |
| fast v1 native | 0.09744 | 1.89× | 60.38 | 43.1% | 712.9 |
| **v2 native** | **0.06254** | **2.94×** | 72.46 | 51.8% | **925.6** |
| v2 expand bf16 | 0.06655 | 2.76× | 79.55 | 56.8% | 792.3 |
| v2 fused gemv | 0.06850 | 2.68× | 77.12 | 55.1% | 794.0 |

Power is the pqteld series on lina (`gx10-6b77`), attributed per arm over its own
window (n=36 samples/arm, `logs/ab_v2_e4m3_q1152.power.json`). Ranked by work per
joule — the GB10 metric, since `gpu_utilization` is non-diagnostic there
(principle 15) — **v2 native is 3.20× the research decoder at 92% of its power**,
so the speedup is real work, not a higher duty cycle. Both arms sit near half the
~140 W envelope, which says the decoder still has headroom and is not
power-limited.

**Landed in gridbook** (`1a57b31`, branch `claude/trellis-v2-decoder`). The
in-tree numbers are lower than the standalone A/B above because the landed path
also carries the prepared-wire fingerprint check and the custom-op dispatch;
they are the ones to cite. Median of 200 iterations at 1024×4096, this tree vs
`2dcd538` built from a worktree:

| family | path | 2dcd538 | v2 | speedup |
|---|---|---|---|---|
| E2M1 | native packed decode | 0.43722 ms | 0.08138 ms | **5.37×** |
| E4M3 | native packed decode | 0.18989 ms | 0.07734 ms | **2.46×** |
| both | `decode_codes`, `dequant_gemv` | unchanged | unchanged | 1.00× |

Second instrument: 30 s decode-only loops attributed against the pqteld series
(`power_draw_w`, 60 samples per window), both families, same tree pair:

| family | 2dcd538 | v2 | work/joule | power | envelope |
|---|---|---|---|---|---|
| E2M1 | 32.0 calls/J @ 69.19 W | 185.7 calls/J @ 82.77 W | **5.80×** | 120% | 59% |
| E4M3 | 59.5 calls/J @ 94.51 W | 189.0 calls/J @ 86.85 W | **3.18×** | 92% | 62% |

Both families land near 187 calls/J — the same rate-insensitivity the in-process
timings show, now confirmed at the box. E2M1 buys its 5.80× by drawing *more*
power (69 → 83 W) and getting far more work for it; E4M3 buys its 3.18× while
drawing *less*. Neither exceeds 62% of the ~140 W envelope, so headroom remains
on both. The loop shape (1024×4096, no dispatch amortization) differs from the
in-tree bench above, so its raw throughput ratios (6.96× E2M1, 2.94× E4M3) run
higher than the 5.37×/2.46× that are the numbers to cite.

Two scope limits travel with this. The **fused GEMV deliberately did not land**:
the warp-resident form reduces per-superblock partials with `atomicAdd`, ~2.7×
faster but not run-to-run deterministic, and this repo quarantines irreproducible
numbers — a deterministic fixed-order reduction is the way to claim it. And the
kernel landed as **research ops only**: no quantization config, producer,
chooser, or runtime-contract cell, so the rungs are *in-tree*, not *backed*.

The E4M3 multiple is smaller than E2M1's 6.73× only because the *research*
baseline was already 2.5× faster on E4M3 (0.1837 vs 0.4506 ms). The number that
matters is the absolute one: **v2 decodes E4M3 at 4.512 bpw in 0.06254 ms and
E2M1 at 2.0 bpw in 0.06692 ms** — within 7% of each other across a 2.3× spread in
rate. The decoder is essentially **rate-insensitive**, which is what makes a
low-rate portability lane (§4) and a high-rate lane the same kernel.

The recorded negative in `docs/TRELLIS-R256-RESEARCH.md` (E4M3 R1152 at 12.1×
slower than FP8_CB_K36) was measured against the research decoder, and its two
halves ran on *different boxes* — E4M3 on lina, E2M1 on sparky, never one box.
**Half of the re-measurement is now done** (the table above, one box). The
matched `FP8_CB_K36` half on that same box is still owed; until it lands the
negative is neither confirmed nor withdrawn — do not cite it either way.

---

## 6. The 8-bit menu — measured

The gap named in the previous revision of this section is closed. `fp8_ladder.py`
put the fp8 lane on the *same* 24-tensor corpus under the *same* `weighted_nsse`,
and its E2M1 control reproduces the published 4-bit study **bit-identically** —
`SELFCHECK PASS: 96/96` control rungs, `rel = 0.000e+00`, `weighted_energy` equal.
That is what makes the two files joinable rather than merely comparable.

**Terminology correction.** The previous revision called the `FP8_CB_K` rungs
"the learned Lloyd books of the intended regime". They are not learned — every
arm here stamps `"codebook": "fixed_lattice"`. The **learned** book is still
unmeasured on both axes (§7). This matters in one direction only: a learned book
can only be ≥ a fixed one, so every result below where CB *wins* is safe
*a fortiori*, and every result where the trellis wins is the one exposed.

### 6.1 The measured ladder

Corpus median weighted SNR, 24 tensors, production bracket. **Both lanes carry the
same scale contract** — one fp32 per output row — and both are charged for it here
(see the accounting note below), so the bpw column is a range across the corpus's
two shapes rather than a single number:

| rung | bpw | median wSNR | | rung | bpw | median wSNR |
|---|---|---|---|---|---|---|
| `tcq_e4m3@2` | 2.0099–2.0170 | 11.21 dB | | `fp8_cb@28` | 3.5078–3.5156 | 17.91 dB |
| `tcq_e4m3@3` | 3.0100–3.0172 | 16.23 dB | | `fp8_cb@32` | 4.0078–4.0156 | 20.71 dB |
| `tcq_e4m3@4` | 4.0100–4.0172 | 23.00 dB | | `fp8_cb@36` | 4.5078–4.5156 | 23.37 dB |
| `tcq_e4m3@5` | 5.0101–5.0172 | 25.92 dB | | `fp8_cb@40` | 5.0078–5.0156 | 26.42 dB |
| `tcq_e4m3@6` | 6.0104–6.0172 | 26.33 dB | | `fp8_cb@44` | 5.5078–5.5156 | 30.42 dB |
| | | | | `fp8_cb@48` | 6.0078–6.0156 | 34.15 dB |

**Accounting note — a live pro-CB confound, corrected.** The fp8 CB grid *does*
build a `(rows, 1)` fp32 scale plane and cannot decode without it
(`nvfp4_cb_formats._scale_and_vectorize` :796-798; reconstruction is `x · pes`).
But `cb_layout.SCALE_PLANE_BYTES[("fp8","v1")]` is **0**, because that table is
denominated *per 256-weight superblock* and a per-**row** plane has no expression
in it — so `exact_bpw` came out as exactly `k/8`. The E4M3 trellis pays the
identical contract (`trellis_formats.py:333`, `per_output_row_fp32`) as an
explicit `rows·4`. Charging one lane and not the other is worth 0.0078 bpw at
in_features 4096 and 0.0156 at 2048 — small, one-directional, and exactly the
confound class this study exists to remove. It is charged symmetrically above and
in §6.3. The correction is applied post-hoc in
`verdict/menu_selection_4families.py::cb_scale_correction_bpw` rather than in the
driver, because the driver was mid-flight across two brackets when it was found
and editing it would have left the brackets on different accounting; the driver
fix is staged in `apply_cb_scale_fix.py` and is **owed**.

**The crossover is ~5.0 bpw.** Below it the trellis wins (+2.3 dB at 4.0 bpw);
above it the book wins, and decisively (+7.8 dB at 6.0). This is the low-trellis /
high-book regime, measured rather than assumed.

**One result worth stating on its own:** `fp8_cb@48` reaches **34.15 dB at 6.008
bpw**, while a plain RTN cast of the same weights onto the full e4m3 grid — the
8.008 bpw reference the whole family degenerates to at bypass — reaches only
**31.57 dB** (min 30.90, max 31.68, all 24). *The fixed-lattice book beats raw FP8
at 75% of the bytes.* A learned book can only widen that.

### 6.2 The E4M3 trellis rungs above ~4 bpw are not menu-eligible

`tcq_e4m3@5` and `tcq_e4m3@6` are **never selected at any budget** (§6.3), and the
reason is a harness defect, not a property of E4M3. R5→R6 buys **+0.36 dB for a
full bit**, and both land *below* the 31.57 dB their own rate-8 bypass reaches.

The cause is measured (`E4M3_ALPHABET_SPAN_DEFECT.md`): **the shared Lloyd
alphabet does not span the data.** `x` covers [−448, 448] by construction of the
row plane; the alphabets stop at −320 (rates 5–7) or −240 (rates 3–4). **0.11–0.22%
of elements carry 45–85% of the weighted SSE at every rung.** The residual is a
*clipping* residual, not a *resolution* one — rate 6→7 doubles the level count
(64→128 distinct e4m3 values, all verified unique; not a saturated codebook) for
**+0.001 dB**. The trellis inherits it structurally: the TCQ candidate set at rate
r *is* the scalar alphabet at rate r+1, so its reachable set carries the same
truncation one rung up.

Suboptimality is **proven at rate 7** — same level count, endpoints extended to the
data range: 25.069 → 26.226 dB (+1.16). At rate 6 that same edit *loses* 2.80 dB.

**Resolved 2026-08-29: the truncation was the selector, not E4M3 — but fixing it
does not rescue the rungs.** `optimize_e4m3_alphabet_hierarchy` is Lloyd, hence
locally optimal and init-dependent. Choosing k points from a finite ordered grid
to minimise weighted SSE under nearest-point assignment is *exactly* solvable in
1-D, so principle 2 says use the explicit: `e4m3_alphabet_dp.exact_alphabet`, a
pair-state DP over grid indices, verified against exhaustive search (24/24 to
1e-9, including the codebook re-scored under true nearest assignment). As a
**scalar** codebook the exact optimum is much better — median **+9.05 dB at
k=16**, never worse at any count on any of the 24 tensors, spanning the full
±448 from k=16 up and landing within **0.04 dB of the whole 254-value grid** at
k=128 (`E4M3_SPAN_DEFECT_RESOLVED.md`).

**On the ladder it does not transfer, and the two brackets disagree in sign:**

| rung | production Δ | penalty Δ |
|---|---|---|
| `tcq_e4m3@2` | −0.13 | −0.43 |
| `tcq_e4m3@3` | **+0.39** | −0.38 |
| `tcq_e4m3@4` | **+0.79** | −2.52 |
| `tcq_e4m3@5` | **+1.86** | −5.06 |
| `tcq_e4m3@6` | **+2.45** | −1.62 |

All six `fp8_cb` arms and all four E2M1 controls move by **exactly 0.00** and
selfcheck stays 96/96, so this is the alphabet and nothing else. A change that
improves one bracket and regresses the other is **not conclusive**, so
`E4M3_ALPHABET=exact_dp` stays **research and opt-in**; `lloyd` remains the
default and reproduces every arm in this document.

**Why the penalty bracket goes the other way — diagnosed, same day.** The
alphabets are stamped in the ladder JSONs, so pairing each rung's Δ against the
*span* of the set the DP chose costs no GPU
(`alphabet_span_pathology_probe.py` → `alphabet_span_pathology.json`):

| rung | penalty Δ (median) | DP span blow-up | span lloyd→dp | production Δ (median) |
|---|---|---|---|---|
| R2 | −0.41 | 0/24 | 5.0 → 6.0 | −0.06 |
| R3 | −0.55 | 0/24 | 6.0 → 6.0 | +0.29 |
| **R4** | **−2.56** | **24/24** | **6.5 → 448** | +1.03 |
| **R5** | **−4.80** | **24/24** | **8.0 → 448** | +1.88 |
| R6 | −1.10 | 0/24 | 80 → 448 | +2.40 |

*The two tables aggregate differently and neither is wrong: the first is `median(exact_dp) − median(lloyd)` (what `compare_alphabet_modes.py` prints), the second is `median(exact_dp − lloyd)` per tensor. They agree in sign at every rung; the paired form is the right one for the span pairing below.*

The two worst penalty rungs are **exactly** the two where the DP's span explodes
~55–70×. On the eff plane the bulk of `x` sits inside ±8; the DP is an exact
scalar-SSE minimiser, so it correctly buys the tail outliers at ±448 and
under-resolves the bulk — the dead-row/shared-alphabet shape, and a property of a
plane **we do not ship**. The production plane never blows up. So "the brackets
disagree" is too flat a reading: on the wire's own contract the exact alphabet is
*positive at R3–R6*. It still does not become the default, for three reasons in
order: this is corpus SSE, a **screen**; **R2 regresses in both brackets with no
span change at all** (240→240), so that residual is a genuine partition effect on
the rung the portability lane cares most about; and diagnosing away the bracket
that disagrees, before re-measuring, is the mirror-confound trap. The settling
run is a tail-matched DP sample on the penalty bracket.

The reason is instructive and is the real open lever: **scalar SSE is the wrong
objective for a TCQ candidate set.** A trellis's coding gain comes from subset
distance under Ungerboeck set partitioning, not from the scalar fidelity of the
union. At R2 the candidate set is `hierarchy[8]`, where the exact DP gains
**+0.012 dB** scalar — nothing — yet the rung moves −0.13 dB, which is only
possible if the DP chose *different grid points of near-equal scalar cost* and
thereby changed the partition. The gain grows with rate because the high rungs
had scalar headroom to convert; the low ones had none.

**Consequences.** (1) The span diagnosis stands, and its cause is now known, but
**no 8-bit E4M3 rung is re-ranked on this** — `@6` gains +2.45 dB in the
shipping bracket and still sits ~2.7 dB below its own rate-8 bypass, so §6.3's
retirement of `tcq_e4m3@6.0` is **weakened, not overturned**. (2) The hoped-for
lift of the **low** rungs did not materialise: `tcq_e4m3@2` regresses in both
brackets. The lower-bound caveat on the low rungs therefore stands, but the
alphabet is no longer the reason to expect them to rise — a partition-aware
selector is.

### 6.3 The joint four-family menu — the definitive list

All four families on one corpus, one metric, swept by per-tensor Lagrangian over
byte budgets 1.30–6.50 bpw in 0.10 steps, ties broken to **fewer** bits (the
conservative direction for a retirement argument). Joinability is *gated*, not
assumed — the run refuses unless per-tensor `weighted_energy` matches to 1e-9,
`source_weight_sha256` matches, and the fp8 ladder's `selfcheck.status` is PASS
(`verdict/menu_selection_4families.py`). The corpus is rectangular: 24 tensors ×
15 fp8 arms and 24 × 24 fp4 arms, no missing cells.

**Two brackets, and only their agreement counts.** Production charges each wire
its own declared scale contract; the penalty bracket charges the E4M3 trellis the
CB group-16 plane instead (~36× its real price). A rung is *retired* only where
both brackets never select it; anything else is an open question.

**Retired under BOTH brackets — 20 rungs:**

| family | retired |
|---|---|
| fp4 lattice CB | **K2, K5, K6, K7, K8, K9, K10, K11, K12, K13, K14, K15, K16, K17, K18** (15 of 18) |
| fp8 lattice CB | **K28, K32, K36** |
| E4M3 trellis | **@6.0** |
| E2M1 trellis | **@2.0** |

**Contested (bracket-dependent — NOT retired, NOT confirmed):** `cb_two_tier@3`,
`fp8_cb@40`, `fp8_cb@44`, `tcq_e4m3@5.0`, `tcq_two_tier@1.5`, `tcq_two_tier@1.75`.

**Selected under both brackets:** `cb_two_tier@1`, `cb_two_tier@4`,
`tcq_two_tier@1.0`, `tcq_two_tier@1.25`, `tcq_two_tier@2.25`, `tcq_e4m3@2.0`,
`tcq_e4m3@3.0`, `tcq_e4m3@4.0`, `fp8_cb@48`.

**The answer to the retirement question.** The fp4 lattice codebook keeps **two
confirmed rungs (K1, K4) plus contested K3 — and all three lie below the trellis
wire's 1.283 bpw structural floor (§2).** It has no surviving rung anywhere the
trellis can reach. Its entire remaining function is sub-floor filler, and a v2
sub-1.0-bit wire family would end it outright. The **fp8** lattice codebook is
the opposite case: it owns the top of the menu and is not a retirement candidate.

**Isolation — what the fp8 lanes actually changed.** Re-running the identical
sweep over the identical 1.30–2.50 budget band, varying only the menu
(`verdict/isolate_k3.py`, using the real solver rather than a re-implementation):

| menu | budgets | fp4 CB survivors |
|---|---|---|
| 4-bit families only | 1.30–2.50 | K1, K3, K4 — *reproduces the published study* |
| all four families | 1.30–2.50 | **K1, K4** |
| all four families | 1.30–6.50 | **K1, K4** |

So `cb_two_tier@3` is displaced specifically by the arrival of the fp8 trellis
lane, not by widening the budget range — which is why it lands in the contested
column rather than the retired one.


---

## 7. What this does not establish

- **Nothing here is a serving result.** No vLLM KL-vs-BF16, no WikiText PPL on a
  served artifact. Every quality number is corpus SSE under the routed imatrix;
  every speed number is a microbenchmark. Per principle 3 these are screens.
- **Corpus scope.** 24 GLM MoE expert tensors. Dense and attention tensors are
  uncovered, and the retirement verdict does not extend to them without a sweep
  that includes them.
- **Backing.** `FP8_CB_K` rungs are backed on sm120 with a cited preflight row.
  The trellis rungs have a *research-proven kernel as of today* and remain
  **unbacked in gridbook** until it is landed and attested per principle 14.
- **The learned-CB question is open on BOTH axes.** Every CB arm measured here —
  4-bit and 8-bit alike — stamps `"codebook": "fixed_lattice"`. The 4-bit sweep
  never ran `--with-learned-cb` and the 8-bit ladder has no learned arm at all, so
  "would *learning* the book change the menu?" has no answer. Direction of the
  exposure: a learned book can only be ≥ a fixed one, so every **CB win** above
  stands *a fortiori*, and every **trellis win** (the ≤4 bpw band, and the fp4
  retirements) is the part a learned book could move.
- **The E4M3 trellis rungs were measured under a known-suboptimal alphabet**
  (§6.2), and that exposure is now **measured, with a split verdict**. The scalar
  alphabet is solved exactly (`e4m3_alphabet_dp`, +9.05 dB median at k=16, full
  ±448 span), and the re-measured ladder splits by rung and by bracket:
  **positive at R3–R6 on the production plane** (+0.29/+1.03/+1.88/+2.40 dB
  median), **negative at R2 in both brackets**, and negative throughout the
  penalty bracket. The penalty negatives are *diagnosed*, not mysterious: at
  R4/R5 the DP's candidate span explodes 6.5→448 on **24/24** tensors — the exact
  scalar minimiser spending candidates on eff-plane tail outliers while the bulk
  lives inside ±8, the shape of the dead-row/shared-alphabet trap — and those are
  exactly the two worst rungs (−2.56, −4.80). The production plane never blows up.
  **`lloyd` stays the default anyway**: this is corpus SSE (a screen), R2
  regresses with *no* span change at all (240→240) so that residual is a genuine
  partition effect, and explaining away the disagreeing bracket before re-measuring
  is the mirror-confound trap. Settle it by tail-matching the DP sample and
  re-running the penalty bracket. The standing lever is still a
  **partition-aware** candidate selector (subset distance under Ungerboeck
  partitioning); nobody has built one.
  Evidence: `dq-runs/trellis-hull-20260828/alphabet_span_pathology.json`.
- **The fp8-CB per-output-row scale charge is now in the driver**, so
  `fp8_ladder.py` and `verdict/menu_selection_4families.py` agree natively
  instead of via a post-hoc correction; the selected menu is unchanged, which is
  the evidence the two paths were already consistent.
- **JSO is closed** (§8), not open: R≥3 ON, R=2 OFF.

---

## 8. Open items

1. **Make the trellis servable** — re-scoped 2026-08-29 by measurement, see
   `trellis_serving_gap_2026-08-29.md`. v2 *is* landed in gridbook
   (`1a57b31`, `c9bcf40`), but landing a decoder is not a serving path: `trellis`
   appears in three gridbook modules and **none is a `LinearMethod`**, so no
   config scheme recognises the wire and no loader accepts it. The contract
   (v11) has no trellis cells, which is why the producer gate correctly resolves
   every trellis unit to `unattested`. Order: (a) `TrellisLinearMethod`;
   (b) a deterministic fixed-order GEMV, since the fused one is withheld for
   `atomicAdd` nondeterminism; (c) routed-MoE evidence, all of the above being
   dense; (d) *then* the release-keyed contract bump — which is two steps, not
   one: **today the pinned serving release (0.8.11) publishes no
   `lane_eligibility` table at all**, so every unit resolves `unattested`, CB
   included, and the consumer hard-requires schema `v1` while the tree publishes
   `v2` — a v2 parser is owed before any cell can be attested.
   **And first, §4 of that doc:** every route reachable from `torch._scaled_mm`
   on GB10 is A=W (fp4×fp4, fp8×fp8 — bf16×fp4, fp8×fp4 and bf16×fp8 all
   refused), so the fast *batch* lanes are **W4A4/W8A8** while this entire menu
   prices **W*A16**. That fork must be settled before the menu can be called
   final for the native lane. Bounded, first-order: it does **not** reopen the
   within-grid CB retirements (activation format is common to both arms); it
   does reopen the cross-grid ~5.0 bpw crossover. Decode-lane work (b) is
   fork-independent — GEMV is already W*A16.
2. ~~Run the 8-bit hull sweep~~ — **DONE** (§6); both brackets `SELFCHECK PASS`,
   `rc=0`.
3. ~~Fix the E4M3 alphabet's tail span and re-measure rungs 3–6~~ — **DONE, and
   it did not pay** (§6.2). The truncation was a Lloyd local optimum, the exact
   1-D DP recovers it entirely, and the re-measured ladder went the wrong way in
   one bracket and at R2 in both. `lloyd` stays the default; `exact_dp` is
   research and opt-in via `E4M3_ALPHABET`. Successor lever: a partition-aware
   selector (§7).
4. ~~Charge fp8-CB its per-row scale plane in the driver~~ — **DONE**.
   `apply_cb_scale_fix.py` applied 2026-08-29 (`row_scale_bytes = rows * 4`); the
   sibling bpw identity assertion was updated to include the row plane. Both
   brackets re-run, `SELFCHECK PASS 96/96`. The verdict re-solves to the same
   13 selected / 20 retired-both / 6 contested as the post-hoc correction gave —
   which is the evidence the two paths were already consistent.
5. Re-measure the E4M3-vs-FP8_CB_K36 negative — **half done**: v2 E4M3 at
   q256=1152 (4.512 bpw, 1024×4096, lina) is `0.06254 ms`, **2.94×** the research
   decoder, parity `all_green`. The matched `FP8_CB_K36` half on the same box is
   still owed before the negative can be cited or withdrawn.
6. ~~Harvest `tcqjso-postpin-20260829`~~ — **DONE**. R=2 JSO loses 6 of 8 cells
   (median +8.02%); R=3 JSO wins 8 of 8 (median −3.35%). The 2026-08-28 SSE screen
   inverted on the KL gold lane. Standing recommendation: **R≥3 JSO ON, R=2 OFF**;
   quarantine lifted (`RETRACTION_tcq_r2_jso_20260828.md`, closed section).
7. Decide whether a v2 sub-1.0-bit wire family is worth designing — it is the only
   thing that would retire the last fp4 lattice CB rungs (§6.3).
8. Run the learned-CB arms on **both** axes (§7) — the one open question that
   could move the trellis-side verdicts.

---

## Addendum, 2026-08-29 — both families now have a native lane

The 8-bit and 4-bit trellis families each have a Gridbook `LinearMethod`:

| family | lane | native shape | measured identity | portability |
|---|---|---|---|---|
| E4M3 | `trellis_e4m3_lane.py` | W8A8 `fp8×fp8` | `torch.equal`, 4/4 shapes | Ada / Hopper / AMD hipBLASLt |
| E2M1 | `trellis_e2m1_lane.py` | W4A4 `fp4×fp4` | max abs err **0**, 8/8 shapes | **Blackwell only** |

Neither needs a scale bridge; the E2M1 "missing bridge" recorded on
2026-08-29 was **retracted the same day** (it described the E4M3 family's
contract). E2M1 needs only the cuBLAS 128×4 blocking of the plane it already
carries.

**This does not finalize the menu for the native lane.** Both lanes execute an
activation contract (`fp8_per_token_dynamic`, `e2m1_group16_ue4m3_static`) that
the entire ladder — weight-only corpus SSE, which prices W\*A16 — has never
priced. Within-grid verdicts hold the activation format fixed on both arms and
plausibly survive; the **cross-grid ~5.0 bpw crossover is a W4A4-vs-W8A8
question** once activations are priced, and it is open. See
`trellis_e2m1_lane_2026-08-29.md` and `trellis_serving_gap_2026-08-29.md`.
