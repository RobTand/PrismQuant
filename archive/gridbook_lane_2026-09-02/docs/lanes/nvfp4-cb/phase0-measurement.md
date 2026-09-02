# NVFP4-CB — Phase 0 Measurement Plan (PLAN ONLY)

> Historical pre-production plan. Its nominal-rate formulas predate the
> versioned serialized-payload accountant: production FP4-CB v2 writes
> `4k+9` bytes per 256-weight superblock, FP8-CB also writes one FP32 scale per
> output row, and every emitted FP16 codebook sidecar is charged once by exact
> physical/content identity. Do not use the formulas below for allocation or
> same-rate claims; final exported whole-artifact bytes are authoritative.

Working name **NVFP4-CB**: a vector-quantized codebook format. Codewords are
8-dim vectors of FP4 (E2M1) codes `{0,±0.5,±1,±1.5,±2,±3,±4,±6}`; group-of-16
E4M3 scales exactly as NVFP4, so a decoded tile is bit-compatible NVFP4 and
would feed the CUTLASS W4A4 path. Per 8-weight vector: a `k`-bit index into a
codebook of FP4-grid-valued vectors. Effective weight rate = `k/8` bpw; the
group-16 E4M3 scale adds `8/16 = 0.5` bpw ⇒ **nominal `k/8 + 0.5` bpw** (before
any learned-codebook sidecar — see §"Matched bytes").

Phase 0 runs **entirely in emulation + existing harnesses (no new kernels)**.
Its four experiments gate all downstream implementation.

---

## Cross-cutting: the Phase-0 gold metric (read first — it constrains everything)

**There is no NVFP4-CB kernel in Phase 0, so NVFP4-CB cannot be served.** The
strongest metric available is **whole-model emulated forward KL-vs-BF16** on a
held-out split:

- Swap each target Linear's weight for its format's *emulation* reconstruction,
  run the real model forward in the build venv (`prismaquant-cu130`), compute
  full-vocab KL vs the BF16 model on held-out WikiText.
- This is **weight-level bit-faithful** for NVFP4-CB: the format decodes to
  *exactly* the NVFP4 grid × E4M3 scale, i.e. the same numbers a CUTLASS kernel
  would consume. Emulate the W4A4 activation path with
  `format_registry.nvfp4_activation_qdq_served` (fr.py:882) so the emulated
  bucket matches the served bucket, not a weight-only screen.
- **Kill the cross-family confound:** measure the IQ baselines (exp 1) through
  the *same* emulated-forward harness, **not** the llama.cpp KL harness. The IQ
  emulation `iq_reconstruct` (gguf_iq_formats.py:562) is pinned bit-exact to
  gguf-py-decoded bytes (`tests/test_gguf_iq_formats.py`), so emulation-IQ ==
  served-IQ weights. Mixing llama.cpp-KL (IQ) with emulation-KL (CB) would be a
  rendering confound and is forbidden by house rule.

**House-rule caveat baked into every Phase-0 conclusion:** emulated forward KL
is an *emulation* gate, not the served gold metric. A downstream kernel phase
**must** re-confirm the winning format on true served vLLM/llama.cpp KL before
any promotion past Candidate. Phase 0 decides *direction*, not ship.

Cheap triage tier (for ranking / entropy, not gates): per-Linear
weighted-MSE and output-MSE from the batched cost path
(`measure_quant_cost.measure_batched_gpu`, mqc.py:1254), which already carries
the imatrix `col_weights` lockstep (mqc.py:1357).

### The encoder — reuse, don't rebuild
The exhaustive GPU codeword search already exists: `gguf_iq_formats._grid_fields`
(gguf_iq_formats.py:230) does an exhaustive weighted grid-argmin over
vector codewords with a two-tier scale, imatrix weighting
(`_weights`, l.128), sign handling (`_sign_fields`, l.148), and a fused
27-candidate scale sweep + WLS refit (l.294–341). NVFP4-CB reuses this
verbatim with three swaps: (a) the grid table becomes an FP4-grid-valued 8-dim
codebook; (b) group size 8→8 (already), scale group = NVFP4 group-16 E4M3 rather
than the ggml two-tier fp16+4-bit; (c) for the *learned* variant, the codebook
is produced by weighted k-means (below) instead of loaded from `iq_grids.pt`.
The scalar `_rtn_fp_codebook` (fr.py:247, `torch.bucketize`) is **not** reusable
— it is per-element RTN and cannot do 8-dim vector NN search.

### Two hard search-cost walls (surface these before spending GPU-hours)
1. **Sidecar (learned only).** A learned per-tensor codebook of `2^k` 8-dim FP4
   vectors costs `2^k × 8 × 4 bits = 2^k × 4 bytes`. Amortized over an
   `N`-param tensor that is `2^k × 32 / N` bpw. For a 1M-param 0.6B Linear:
   k=12 ⇒ +0.13 bpw; k=16 ⇒ +2.1 bpw (**dead on arrival**); k=20 ⇒ +33 bpw.
   **⚠ 2026-07-15 review: this penalty is a deterministic function of N and
   shrinks ~50× at production scale** — a 25M-param 27B-class Linear pays only
   +0.084 bpw at k=16, and a Hy3-class packed expert stack far less. A
   "learned loses match-bytes at 0.6B" verdict therefore MUST NOT kill the
   learned path for big-model use: exp 1 must report match-bytes **as an
   analytic curve over N** (no extra GPU work — quality from the test model,
   sidecar cost projected at 0.6B/4B/27B/300B tensor sizes) and gate per
   deployment scale. A *shared* (per-model or per-role) codebook amortizes it
   away but is a different format. This sidecar is the crux of fixed-vs-learned
   and MUST be in "matched bytes".
2. **Encode compute.** Exhaustive weighted argmin / k-means is
   O(vectors × 2^k × 8). The IQ machinery caps grids at `ngrid ≤ 1024` for
   exactly this reason. Tractable to roughly **k ≤ 14** (16384 codewords) on a
   single GB10; k=16 is heavy, k=20–24 is infeasible exhaustively (16M
   centroids). **The prompt's k=12–24 ladder is not uniformly searchable.**
   High-k requires structured-lattice fast closest-point (Conway–Sloane, the
   reason IQ uses structured grids) or product/residual VQ (split the 8-dim into
   sub-blocks). Phase 0 therefore runs its head-to-heads on the **exhaustively
   reachable rungs (k∈{12,14})** and treats "can a fixed lattice or PQ reach
   k=20–24 in budget at acceptable quality?" as an explicit gating unknown fed to
   exp 1's decision.

---

## Matched-bytes accounting (precise, per comparison)

All byte accounting uses `FormatSpec.memory_bytes_for_shape` /
`effective_bits_for_shape` (fr.py:119–128), which already amortizes the scale
sidecar exactly (GGUF proves the pattern). For NVFP4-CB add two terms a stock
FormatSpec does not model:

- **weight**: `k/8` bpw (k-bit index per 8 weights).
- **NVFP4 scale**: E4M3 per group-16 = `8/16 = 0.5` bpw.
- **codebook sidecar** (learned only): `2^k × 32 / N_tensor` bpw, or
  `2^k × 32 / N_model` if a codebook is shared model-wide.

Total shipped bytes per arm = Σ_tensor(weight + scale bytes) + Σ(codebook
sidecar bytes). Write a `nvfp4cb_footprint(assignment, model)` accountant
(~60 LoC, mirrors `footprint.py` / GGUF's exact-bpw derivation) so no arm can
hide sidecar cost.

- **Exp 1** reports both: **match-k** (equal k across fixed/learned/IQ —
  isolates codebook *quality*, learned carries a visible byte penalty) and
  **match-bytes** (learned gets a smaller k so total bytes incl. sidecar equal
  the fixed/IQ arm — the *shippable* question).
- **Exp 2** is entropy of the index stream only; no byte match needed.
- **Exp 3** matches **total model file bytes** via the allocator's byte-budget
  selection (`allocator.py --target-disk-gb`, l.1049) — fine-menu vs coarse-menu
  at one identical file size.
- **Exp 4** is byte-identical by construction (same format, same k, only the
  per-column weighting vector changes) — the cleanest isolation in the plan.

---

## Seed / repeat protocol (applies to every experiment)

- `--calib-repeats ≥ 4`: 4 independent calibration draws (distinct seeds) from
  `diverse-v1.jsonl` feed cost/imatrix/Fisher; single-seed n=8 KL is known to
  flip sign (memory: between-seed spread ~40% of mean at 0.6B).
- **Held-out disjoint**: selection/KL uses WikiText *test* text the calibration
  draws never saw (`/home/rob/dq-runs/gguf-smoke/wiki.test.raw`, held out from
  `diverse-v1.jsonl`).
- Report **mean ± between-seed std** on the forward-KL gold metric. A lever
  advances only when the effect **exceeds the between-seed std** on both models.
- Bake git commit + calibration hash + assignment hash into every output JSON
  (house reproducibility gate); an irreproducible number is quarantined.
- All arms of a comparison share ONE calibration draw per seed (paired), so
  arm deltas are within-draw.

Models (both, every experiment): **Qwen3-0.6B** (`/home/rob/models/Qwen3-0.6B`,
fast) then **Qwen3-4B** (HF cache `models--Qwen--Qwen3-4B`, scale check). 0.6B is
the fast triage; a lever only advances if it survives the 4B check — the GGUF
lane repeatedly saw 0.6B wins fail to transfer to 4B (memory:
gguf-lowbit-serving-lane, "0.6B win did NOT transfer").

---

## Experiment 1 — fixed FP4-grid lattice vs learned-on-grid codebook vs IQ

**Question.** At matched rate, does a per-tensor *learned* (Fisher/imatrix-
weighted k-means over FP4-grid-valued 8-dim vectors) codebook beat a *fixed*
structured FP4-grid lattice — and how do both compare to the existing IQ formats
(IQ2_S 2.56, IQ3_XXS 3.06 bpw) at matched bpw? Decomposes into (a) the
learned-codebook *quality* gain and (b) the cost of constraining codewords to
the FP4 grid vs IQ's free/structured grids.

**Hypotheses.**
- H1a: learned beats fixed on match-k weighted-MSE and forward KL (more of the
  codebook mass lands where the weight distribution is).
- H1b: on **match-bytes**, the learned sidecar erases much/all of H1a at
  k≥14 on small tensors ⇒ fixed lattice wins the shippable comparison unless the
  codebook is shared.
- H1c: NVFP4-CB (either variant) ≈ IQ at matched bpw on KL, but with the
  structural upside that it decodes to native FP4 (the whole point) — IQ decodes
  away from tensor cores.

**Reuse.**
- Encoder: `_grid_fields` (gguf_iq_formats.py:230) with an FP4-grid codebook
  table (fixed) or a k-means codebook (learned).
- Fixed lattice generator: adapt `scripts/gen_iq_grids.py` to enumerate a
  structured FP4-grid sub-lattice at each k (analogous to the IQ grid tables).
- IQ baselines: `iq_fields`/`iq_reconstruct` as-is, imatrix-weighted.
- Gold metric: emulated-forward-KL harness (below).

**Write (new).**
- `nvfp4cb_codebook.py` (~180 LoC): fixed FP4-lattice table builder + weighted
  k-means on the FP4 grid (init k-means++, Lloyd iterations, each centroid
  snapped to nearest FP4-grid vector after every update; weights = imatrix or
  Fisher per §exp4). GPU-first, chunked like `_grid_fields`.
- `emu_forward_kl.py` (~150 LoC): whole-model QDQ-swap forward KL-vs-BF16 on
  held-out WikiText. Takes a per-Linear format→qdq map; renders via registry
  `quantize_dequantize` (+ `col_weights` for weighted formats) + NVFP4 activation
  emulation; reuses the eval-loop/KL reduction from `validation_harness.py` /
  `kl_measurement.py` (do not re-implement KL). This harness is shared by exps
  1, 3, 4.
- `nvfp4cb_footprint.py` (~60 LoC): the sidecar-aware byte accountant (§Matched
  bytes).

**Arms (uniform-format, per model, per seed).** For k∈{12,14} (exhaustively
reachable): {fixed-lattice, learned-per-tensor} × {match-k, match-bytes}, plus
IQ2_S and IQ3_XXS at their native bpw as the cross-family reference. Weighting =
imatrix (Fisher deferred to exp 4).

**Metrics.** Primary: emulated-forward KL-vs-BF16 (mean ± between-seed std),
top-1 agreement. Secondary (triage/ranking): per-tensor imatrix-weighted MSE
ratio vs the fixed arm.

**Decision gates.**
- Learned beats fixed by **> between-seed std on match-bytes at BOTH models**
  ⇒ learned-codebook path advances to the kernel phase; else the fixed
  structured lattice is the default carrier (also the only high-k-tractable one)
  and learned is shelved as research.
- NVFP4-CB (better of the two) within **noise (±1 between-seed std)** of IQ at
  matched bpw ⇒ the "IQ-class compression at native-FP4 compute" thesis holds ⇒
  kernel phase is justified. If NVFP4-CB **loses IQ by > ~15% KL** at matched
  bpw at BOTH models, the FP4-grid constraint costs too much and the whole family
  is **killed** (fall back to serving IQ via MMVQ/Triton).

**GPU-hours.** Encode+eval per (arm,model,seed): 0.6B ~3–6 min (k≤14 exhaustive
+ 1 forward-KL pass), 4B ~15–30 min. ~8 arms × 4 seeds × 2 models ≈ **10–14
GPU-h**.

---

## Experiment 2 — index entropy H(indices) vs k

**Question.** On real NVFP4-CB encodings, how far below `k` is the empirical
entropy `H(indices)`? Decides whether entropy coding (a fractional/variable rate)
could ever pay.

**Hypothesis.** k-means/lattice cells are near-equiprobable by construction ⇒
`H ≳ k − ε` (ε small) ⇒ entropy coding buys < ~0.1 bpw and is **not worth** the
decode complexity (and would break the fixed-rate CUTLASS tile). Expected
answer: no.

**Reuse.** The exp-1 encoder output (index tensors) — no new encoding runs;
piggyback on exp-1 artifacts.

**Write (new).** `index_entropy.py` (~40 LoC): per-tensor empirical
`H = −Σ p log2 p` over the `2^k` index histogram, plus conditional/pairwise
`H(idx_t | idx_{t-1})` to check for exploitable serial correlation. Report
`k − H` (redundancy, bpw recoverable) per tensor and model-weighted mean.

**Metric.** Redundancy `k − H` in bpw, and best-case entropy-coded rate.

**Decision gate.** Model-weighted mean recoverable rate **> 0.25 bpw at k∈{12,14}
on both models** ⇒ open a (separate, later) entropy-coding investigation; else
**close the question** — fixed-rate indexing is optimal and this is the expected
result. Serial-correlation `H(idx_t|idx_{t-1}) < H(idx_t) − 0.1 bpw` would be a
surprise worth a note but does not, alone, advance anything (variable-rate
breaks the tile).

**GPU-hours.** Negligible (histogramming exp-1 outputs). **< 0.5 GPU-h.**

---

## Experiment 3 — fine k-ladder vs coarse menu through the real allocator

**Question.** Does a fine 0.125-bpw NVFP4-CB menu (k=12..24 ⇒ 2.0..3.5 bpw in
0.125 steps) buy measurable end-quality over a coarse {2.0, 2.5, 3.0} bpw menu
at matched total model bytes, when run through the real allocator?

**Hypothesis.** Finer granularity lets the knapsack place each Linear nearer its
ideal rate ⇒ lower KL at matched bytes; but the gain is likely small (memory:
most pipeline improvements are < 5% and the surrogate is mis-ranked at the
margin) and may be swamped by the exp-1 finding that high-k rungs (k>14) aren't
cleanly encodable — so the *usable* fine ladder may be short.

**Reuse.**
- `allocator.py` + `allocator_solver.py` multi-choice knapsack DP, byte-budget
  selection (`--target-disk-gb`, l.1049) for exact matched-bytes.
- Per-(Linear,format) cost from `measure_quant_cost` batched path
  (emulated NVFP4-CB qdq registered as `FormatSpec`s — see below), M6 objective
  (`h_trace × weight_mse`) which won the GGUF lane (docs/lanes/gguf.md:66).
- Gold check: `emu_forward_kl.py` on the two chosen assignments.

**Write (new).**
- Register the k-ladder as emulated `FormatSpec`s (~40 LoC in a Phase-0-only
  registration block, family `"nvcb"`, `effective_bits_for_shape` forced to
  `k/8 + 0.5 (+ sidecar)` the way `_make_gguf_spec` forces GGUF bpw, fr.py:826).
  `quantize_dequantize` = the exp-1 encoder's qdq. Only the exhaustively-
  reachable rungs get a *learned/exhaustive* encoder; higher rungs, if included,
  use the fixed lattice (flagged).
- No allocator changes expected (it already ingests arbitrary FormatSpecs +
  byte budget). If a family branch is needed in the cost path, ~20 LoC mirroring
  the `family == "gguf"` branch (mqc.py:1202).

**Arms.** Fine menu (all reachable k rungs) vs coarse menu ({2.0,2.5,3.0}),
each allocated to the **same total file bytes** (pick 2–3 budget points spanning
2.2–3.2 bpw body), rendered and scored by forward KL. Per model, per seed.

**Metrics.** Emulated-forward KL-vs-BF16 at matched total bytes (mean ±
between-seed std); allocation composition (rungs chosen) for interpretability.

**Decision gate.** Fine menu beats coarse by **> between-seed std at matched
bytes on BOTH models** ⇒ ship the fine ladder (justifies building encode/pack
for all reachable rungs). Gain **< noise** ⇒ ship only the coarse {2.0,2.5,3.0}
rungs (far less kernel/packer surface). This experiment is **gated on exp 1**
(needs a validated encoder + fixed-vs-learned decision first).

**GPU-hours.** Cost measurement is cheap batched (0.6B ~2 min, 4B ~10 min per
seed). Allocation is seconds. Forward-KL on ~4 assignments × 2 budgets × 4 seeds
× 2 models. ~**6–9 GPU-h**.

---

## Experiment 4 — Fisher per-column weights vs imatrix (E[x²]) weights

**Question.** Does weighting the codeword/scale search by **Fisher-derived
per-column weights** (which llama.cpp structurally cannot compute — no backward
pass) beat the imatrix `E[x²]` weighting, as the objective weight `w_i` in the
codebook/scale search?

**Dual purpose.** This is *also* the upgrade A/B for the existing GGUF lane,
where our imatrix-RTN arm currently **loses llama.cpp's imatrix arm by ~16% KL
at 0.6B** (docs/lanes/gguf.md:97; memory: gguf-lowbit-serving-lane). Fisher
weighting is a lever llama.cpp cannot match — a credible way to reclaim and
pass that gap on our own harness.

**Hypothesis.** Per-column Fisher `Σ_out (∂L/∂W_ij)²` (KL-Gauss-Newton) targets
the columns whose error most moves the model KL, which imatrix `E[x²]` only
proxies. Expect Fisher ≥ imatrix on forward KL, most visibly at deep bpw.

**Reuse.**
- Backward machinery: `aura_cost.compute_aura_cost` (aura_cost.py:340) already
  harvests KL-Fisher weight-grad energy per Linear (`g_trace`, l.452/518) via
  `kl_fisher.fisher_probe_scalar`. It currently reduces to a **scalar `h_trace`**
  per Linear (l.650). Experiment 4 needs the **per-column** reduction:
  `w_col[j] = Σ_out (∂L/∂W)[:,j]²` — same gradient, reduced over output rows
  instead of fully summed.
- Weight injection: the encoder already accepts per-column `col_weights`
  (`gguf_quantize_dequantize(w, fmt, col_weights=...)`, gguf_formats.py:388;
  `_grid_fields(... qw)`, gguf_iq_formats.py:230; `_weights`, l.128). Fisher
  weights drop straight into the `qw` slot in place of the imatrix vector.
- Both the NVFP4-CB encoder and the existing IQ/k-quant encoders take the same
  `col_weights`, so the A/B runs on both lanes with one code path.

**Write (new).**
- `fisher_col_weights.py` (~90 LoC): extend the `aura_cost` grad harvest to emit
  a per-Linear length-`in_features` Fisher vector (add a per-column
  accumulation next to the scalar `g_trace` sum, l.518), normalized to the same
  scale/composition the imatrix uses (`qw · sqrt(σ² + x²)`, so the two weightings
  are swapped cleanly — do **not** double-count activation energy; test
  Fisher-only and Fisher×activation compositions as two sub-arms).
- Reuse `emu_forward_kl.py` for the gold metric.

**Arms (byte-identical, per model, per seed).** On NVFP4-CB (best variant from
exp 1, k∈{12,14}) and on the GGUF IQ lane (IQ2_S/IQ3_XXS): weighting ∈
{imatrix (baseline), Fisher-only, Fisher×activation}. Also include the
llama.cpp imatrix arm as the external reference on the IQ lane (this arm *only*
runs through the llama.cpp KL harness — it is the one place llama.cpp appears,
and it is a reference bar, not an arm we render).

**Metrics.** Emulated-forward KL-vs-BF16 (mean ± between-seed std); for the GGUF
lane cross-check against the published llama.cpp-harness numbers to confirm we
close the ~16% gap.

**Decision gates.**
- Fisher (either composition) beats imatrix by **> between-seed std on BOTH
  models** ⇒ Fisher weighting becomes the default codeword/scale weighting for
  NVFP4-CB **and** is promoted as an opt-in GGUF-lane lever (default-off until
  4B + served re-confirm). If it also **closes the ~16% GGUF gap** on our
  harness, that is an independent, shippable GGUF-lane win.
- Fisher ≈ imatrix (within noise) ⇒ keep imatrix (cheaper, no backward pass) and
  record the null as a durable result (Fisher weighting doesn't earn its keep at
  ship bpw).
- Fisher **worse** ⇒ record and stop; imatrix stays.

**GPU-hours.** Adds a KL-Fisher backward per seed (aura_cost is ~single
calibration backward pass): 0.6B ~2 min, 4B ~10 min, plus encode+forward-KL per
arm. ~6 arms × 4 seeds × 2 models. ~**8–11 GPU-h**.

---

## Ordering / dependencies

1. **Build shared harnesses first** (blocks everything): `emu_forward_kl.py`,
   `nvfp4cb_codebook.py`, `nvfp4cb_footprint.py`. Validate `emu_forward_kl` by
   reproducing a *known* IQ/k-quant number: render an IQ2_S/Q2_K uniform
   assignment through emulation-forward-KL and confirm it tracks the published
   llama.cpp-harness KL ordering on the same 0.6B artifact (sanity that the
   emulation gate is faithful).
2. **Exp 1** (fixed vs learned vs IQ) — its fixed-vs-learned and family-viability
   gates feed exps 3 and 4. Can kill the whole family.
3. **Exp 2** — piggybacks on exp-1 encodings; run immediately after exp 1.
4. **Exp 4** (weighting) — needs the exp-1 encoder + best variant; independent of
   exp 3, can run in parallel with it.
5. **Exp 3** (allocator granularity) — needs a validated encoder (exp 1) and the
   chosen weighting (ideally exp 4's outcome, but can run on imatrix and re-check
   if Fisher wins).

**Sequential critical path:** harnesses → exp1 → (exp2 ∥ exp4 ∥ exp3).

## Total wall-clock budget

Compute: ~10–14 (e1) + <0.5 (e2) + 6–9 (e3) + 8–11 (e4) ≈ **25–35 GPU-hours** on
the single GB10, plus harness/encoder build (~2–3 dev-days for
`emu_forward_kl` + `nvfp4cb_codebook` + `fisher_col_weights` + footprint +
FormatSpec registration + the exp-2/entropy script; ~600 new LoC total).
Realistic elapsed with build + debug + the 4B-doesn't-transfer re-runs the GGUF
lane taught us to expect: **~1 week**.

## Skeptical bottom line

The two make-or-break unknowns are structural, not tuning: **(a)** the learned
codebook's per-tensor sidecar may erase its quality edge at shippable bytes
(exp 1 match-bytes), and **(b)** the k=12–24 ladder is **not** exhaustively
encodable above ~k=14, so the headline 2.0–3.5 bpw range depends on a
fast-lattice or product-VQ search that Phase 0 does *not* build — exp 1/3 must
report whether the *reachable* rungs alone clear the IQ bar. If NVFP4-CB only
matches IQ on KL (expected), the entire justification rests on the **native-FP4
compute** upside (IQ's 42 tok/s prefill tax), which Phase 0 **cannot measure**
— that is a kernel-phase gate, and no ship claim is defensible until it lands on
the served metric.
