# What "make the trellis servable" actually requires

**Status:** measured 2026-08-29 on sparky (GB10, sm_121, torch 2.11.0+cu130).
Evidence: `dq-runs/trellis-kernel-20260829/route_probe.py` →
`logs/route_probe.json`. Companion to `format_menu_2026-08-29.md`, which settles
*which* formats; this settles *what stands between them and a served artifact*.

> **This document corrects an earlier framing of mine.** I had recorded the
> remaining blocker as "route attestation is release-keyed, and Rob's call."
> That is true but it is not the *first* blocker, and naming it first made the
> gap look like a decision when it is mostly engineering. The real ordering is
> below.

---

## 1. The finding, in one line

**Gridbook can decode the trellis wire but cannot serve it.** `trellis` appears
in exactly three modules — `gridbook/trellis.py`, `gridbook/trellis_ops.py`,
`gridbook/cuda_ext.py` — and **none of them is a serving path**. There is no
`TrellisLinearMethod`, no `config.py` scheme that recognises a trellis payload,
and no weight loader that accepts the wire. The four existing linear methods
(`PrismaQuantCBLinearMethod`, `Fp8SourceW8A16LinearMethod`,
`Mxfp8DenseLinearMethod`, `MixedFusedLinearMethod`) all cover other families.

So the v2 decoder is a **kernel without a caller** in the serving stack.

## 2. What IS proven, and it is the load-bearing half

The reason this is an unblocked engineering task rather than a research question
is that the GEMMs the trellis would hand its tiles to are **measured native on
this hardware**. Kernel identity captured by profiler, classified on the MMA atom
and mainloop tag rather than a loose digit match (principle 14 — derived, not
asserted; principle 9 — native for the *declared target*, not an older schedule):

| route | kernel symbol (abbrev.) | generation | verdict | µs/call |
|---|---|---|---|---|
| bf16 reference (what we replace) | `nvjet_sm121_tst_mma_128x176x64` | **121** | native | 211.9 |
| **E4M3 → `_scaled_mm`** | `MainloopSm120TmaWarpSpecialized` + **`SM120_16x8x32_TN`**, `float_e4m3_t` | **120** | **native** | 105.5 |
| E4M3 GEMM alone (no decode) | same | 120 | native | 92.8 |
| **nvfp4 blockwise GEMM** | **`cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1`** | **120** | **native** | 50.9 |

`M=512, N=K=4096`. Both consumer GEMMs are Blackwell tensor-op kernels — **not**
the `arch::Sm80` fallback that rode 73.7% of the 92 GB body on 2026-08-17. The
fp4 GEMM is **4.2× faster than the bf16 matmul** it would replace and 1.8× faster
than the fp8 one, before any weight-bandwidth saving is counted.

*Measurement hygiene, since it bit twice here:* `key_averages()` returns the
**per-call average** in `device_time` (`device_time_total` is the sum), and
without `acc_events=True` the profiler silently drops whole cycles. Dividing the
average by `count` read 10.2 µs for the bf16 matmul — 1.6 PFLOP/s, physically
impossible on GB10. The corrected 211.9 µs is 85.4 TFLOP/s and agrees with
wall-clock (201.1 µs). Any future probe here must cross-check against wall-clock.

## 3. The gap, ordered — engineering first, release last

**(a) `TrellisLinearMethod` in gridbook — the actual blocker.** Config scheme
recognition, a weight loader for the wire, and `apply()` = decode → native GEMM.
Two lanes, at different readiness:

- **E4M3 / W8A8 lane — clear.** Decode to a contiguous `[N,K] float8_e4m3fn`
  tile, hand to rowwise `_scaled_mm`. The wire's `per_output_row_fp32` scale is a
  structural match for that GEMM's `scale_b`, so this is a *consumer swap*, not a
  new mainloop. This is also the **AMD / pre-Blackwell portability lane** — the
  probe deliberately uses `torch._scaled_mm`, which also exists on Ada, Hopper
  and (via hipBLASLt) AMD, rather than a Blackwell-only entry point.
- **E2M1 / fp4 lane — RETRACTED 2026-08-29, there was no missing bridge.**
  This bullet previously read: *"the decoded tile carries the right codes but
  the wrong scale plane … the trellis carries one fp32 scale per output row."*
  **That described the E4M3 family's contract, not E2M1's.** The E2M1 wire
  contracts `W[r][c] = e2m1_value(code) * e4m3fn_value(scale_blob[r][c//16])
  * global_scale_real` — a group-16 ue4m3 block plane, which is precisely the
  operand the `block_scaled_ue4m3xe2m1` mainloop demands
  (`trellis.py:716`, `trellis.py:544-553`). Measured with varied scales on
  **both** operands, at aligned and unaligned `N` and arbitrary `M`:
  decode → `_scaled_mm` reproduces the wire's contracted product at
  **max absolute error exactly 0, 8/8 shapes** (`_q6_swizzle.py`). The only
  transform needed is the cuBLAS 128×4 blocking of that plane. The lane is
  `gridbook/trellis_e2m1_lane.py`; the write-up is
  `trellis_e2m1_lane_2026-08-29.md`.

  *How the error survived:* one probe's prose asserted a family's contract
  from memory instead of reading `trellis.py`, and four documents inherited it.
  The receipt `logs/route_probe.json` is left unmodified with a sibling
  annotation, per the phase-6 rule that a receipt is never rewritten.

**(b) The decode regime has no route at all.** The warp-resident fused GEMV is
**withheld**: its reduction uses `atomicAdd` and is not bit-reproducible. Until a
fixed-order reduction lands, decode-regime traffic must either expand first or be
declared unbacked. This is the honest reason a trellis artifact could not yet
claim a decode cell even with (a) done.

**(c) Routed-MoE is entirely unexercised.** Every measurement above is dense.
`structure: routed_moe` cells cannot be proposed on dense evidence.

**(d) Only then: the contract cells** — and this step is further away than it
looks, for a reason I got wrong on the first pass and then measured.

*What I asserted:* that the producer gate resolves a trellis unit to
`unattested` because trellis has no cells. *What actually resolves* (synthetic
`UnitStructuralFacts` for `TCQ_E4M3_R256@2.0`, dense and routed, against
`load_eligibility_table()` in this tree):

```
present: False   rules: 0   ->  dense: unattested   routed_moe: unattested
```

The verdict is right; my reason was not, and the true reason is a **three-way
skew** worth recording:

1. **PrismaQuant's serving pin is 0.8.11**, whose materialized contract is
   `gridbook.runtime-contract.v4` and carries **no `lane_eligibility` key at
   all** (`keys: abi_features, contract_version, formats, layout, packing,
   producer_profiles, quant_method, schema`). So `present=False` and the loud
   `absent_reason` fires — *"a REFUSAL TO CLAIM, not a clean bill"*. **Today
   every unit resolves `unattested`, CB included.** Trellis is not specially
   unattested; nothing is attested.
2. The 12-cell table I read is in the **gridbook working tree** at
   `contract_version: 11` / `gridbook.lane-eligibility.v2` — *not* at the pin
   PrismaQuant consumes. This is the owed pin bump already on the books
   ([[tp2_cluster_campaign]]).
3. When that bump lands the parser **refuses**: the consumer hard-requires
   `LANE_ELIGIBILITY_SCHEMA == "gridbook.lane-eligibility.v1"`
   (`gridbook_lane_eligibility.py:623-626`) and the tree publishes **v2**. Loud
   and correct, but it means a v2 parser is owed before any cell — trellis or CB
   — can be attested.

And the schema decision is now concrete rather than hypothetical. The consumer's
`UnitStructuralFacts` keys on `payload_family`, `k`, `n_sub` and shapes; the
published v2 cell keys on `family` + `rungs` + `platform`/`regime`/`structure` +
`predicates`, and carries no `payload_family` at all. So the two vocabularies
have *already* diverged for CB, before trellis adds a **rate** (which is neither
a `k` nor an `n_sub`). Whoever writes the v2 parser is making the trellis
decision whether they mean to or not: give the facts a rate field, or let
`rungs` carry it.

Contract v11 → v12 remains **release-keyed** (the version is duplicated across
five files) and therefore Rob's call — and it is the *last* step, behind both
(a) and the v2 parser.

---

## 4. The finding that changes the serving plan: every route reachable *from
`torch._scaled_mm` in this venv* is A=W

**Scope first, because this claim will be inherited.** What was measured is the
set of GEMMs `torch._scaled_mm` will dispatch on GB10/sm121 under
`prismaquant-cu130` (torch 2.11+cu130) at one shape. It is *not* a statement
that no W*A16 tensor-core kernel can exist — CUTLASS mixed-input mainloops do —
only that none is reachable through the API the serving path would call, and
none is packaged in the pinned runtime. Within that scope,
`torch._scaled_mm` **refuses every mixed-precision activation**:

| A operand | B operand | result |
|---|---|---|
| bf16 | fp4 (e2m1) | **REFUSED** (operands must share the packed fp4 layout) |
| fp8_e4m3 | fp4 (e2m1) | **REFUSED** |
| fp4 | fp4 | **OK** → `cutlass3x_sm120_bstensorop_..._block_scaled_ue4m3xe2m1` |
| bf16 | fp8_e4m3 | **REFUSED** (`Invalid scaling configuration` — both operands must be float8) |
| fp8_e4m3 | fp8_e4m3 | **OK** → `MainloopSm120TmaWarpSpecialized` + `SM120_16x8x32_TN` |

So the native tensor-core routes are **W4A4** and **W8A8**. There is no native
W4A16 or W8A16 route; the only W*A16 shape is dequantise-to-bf16 then bf16 GEMM,
which is exactly the `expand_bf16` shape the contract already classifies as
**`fallback`** for both CB families.

**Why this matters more than the kernel work.** Every trellis quality number we
have — the whole 4-bit and 8-bit ladder, the 24-tensor sweep, the four-family
menu — is **weight-only corpus SSE**. That prices the W*A16 shape. The route
that is *fast and native* is W4A4 / W8A8, whose activation quantization the
ladder never measured and which the AURA cost is structurally blind to
(`aura_is_activation_quant_blind`).

That is the **2026-08-17 NVFP4_CB defect in its exact original shape**: rendering
identity without execution identity, a real A-side priced at zero. The
difference is that this time it is caught *before* an artifact is built, which is
the whole point of principle 8's second clause.

**The fork this creates — a real decision, not a default:**

- **(A) Serve the native contract and re-price.** Target W8A8 (and W4A4), and
  price the activation side with the machinery that already exists for exactly
  this: the AQUA lane (`aqua_aura_activation_awareness`,
  `aqua_on_cb_via_anchored_lane`). This is also the right answer for the
  portability lane Rob asked for — `_scaled_mm` fp8×fp8 is precisely what Ada,
  Hopper and AMD/hipBLASLt expose, so **low-rate E4M3 trellis on W8A8 is the
  portable shape**, not an approximation of one.
- **(B) Serve W*A16 through the bf16 expand.** Matches every number already
  measured, needs no re-pricing — but it is a `fallback` route by the contract's
  own vocabulary, and it forfeits the tensor-core win that motivated the format
  (fp4 GEMM is **4.2×** the bf16 matmul; fp8 is 2.0×). *Those two ratios are
  single-instrument* — `torch.profiler` device time at one shape, no power
  series — so per principle 15 they are identity-probe context, **not** a speed
  claim. A speed claim owes a pqtel window on a re-run.

- **(C) Build a fused-dequant W4A16 / W8A16 tensor-core kernel** — decode the
  trellis in the mainloop's prologue and feed bf16 MMA, the machete/Marlin shape.
  This preserves every number already measured *and* keeps tensor cores. It is
  disfavoured only because it is a new mainloop rather than a consumer swap, and
  because it is a custom kernel on a project whose stock-vLLM lane forbids one —
  but it is already on the books as the **low-bit kernel lane** direction
  (`CLAUDE.md` §8, `references/lowbit-kernels/`), so it is Rob's third option and
  not a strawman.

**Which regime the fork actually bites.** A=W constrains the **batch/prefill**
regime, where the GEMM is the route. The **decode** regime is GEMV, and the
withheld fused warp-resident GEMV is *already* the W*A16 shape — it decodes to
registers and multiplies a bf16 activation vector. So (b) above is not merely a
determinism chore: it is the one piece of trellis serving work that is
**fork-independent**, because decode never had an A=W constraint to resolve.

**No branch is chosen here.** What is settled is that the menu as it stands
prices lane (B) while the kernel work has been aimed at lane (A), and the two
cannot both be finalised on the current evidence.

**What this does and does not reopen in the menu** (first-order reasoning,
labelled as such, not measured). A *within-grid* comparison — fp4 trellis vs fp4
CB, which is the retirement Rob actually asked about — holds the activation
format fixed on both arms, so at fixed calibration activations the A-side error
is common-mode to first order and those verdicts plausibly survive an A-side
re-price. What genuinely moves is the *cross-grid* comparison: the ~5.0 bpw
4-bit/8-bit crossover is exactly a W4A4-vs-W8A8 question once activations are
priced, and it is the number to re-derive first under branch (A). So the caveat
is targeted, not blanket: **the CB retirements are not reopened by this finding;
the crossover is.**

---

## 5. Honest scope

Everything here is **kernel identity and reachability on one box at one shape**.
Nothing in this document is a serving result: no model has been loaded, no
artifact exported, no KL or PPL measured on a trellis-served checkpoint. Per
principle 3 a screen is not a result, and per principle 9 eligibility is judged
*per artifact, at export* — so even after (a)–(d), the first trellis artifact
still owes the full gold lane.
