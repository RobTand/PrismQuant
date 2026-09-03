# PrismaQuant — Working Agreement & Project Brain

This file is the durable, code-validated brief for working on PrismaQuant with
Robert Tand.

> **Prime directive — read every other doc incredulously.** This repo accretes
> handovers, results docs, READMEs, and a paper. They go stale *fast* and
> sometimes invert. A claim is true only if (a) it lives in current code/tests,
> or (b) it held on the **serving metric** (exact vLLM KL-vs-BF16 + WikiText PPL
> on the served artifact). Everything else is a lead to verify, not a fact to
> repeat. When a doc and the code disagree, the code wins and the doc is stale.
> This file itself is a snapshot — trust `git`, `run-pipeline.sh`, the tests,
> and the auto-memory over any prose, including this prose.

---

## 1. What PrismaQuant is

**Mixed-precision LLM quantization that chooses the right format for each
Linear, selected on real end-to-end KL — and ships as a stock `compressed-tensors`
checkpoint that vanilla vLLM serves with no forked runtime and no custom kernels.**

The intellectual move is to split quantization into two orthogonal axes:

- **Local question (well-studied):** *given a fixed format, how do you round this
  one Linear best?* — GPTQ, AutoRound, scale sweeps, rotations. This is the
  per-tensor toolkit, and it runs *under* whatever format is chosen.
- **Global question (PrismaQuant's contribution):** *how many bits should each
  Linear get, and in which hardware format?* — a per-Linear allocation of a
  total bit budget across `{BF16, FP8, NVFP4, MXFP8, FP8_SOURCE}`.

A heterogeneous per-Linear assignment extracts quality that no single-format
method structurally can. The headline proof: against RedHatAI's uniform
`Qwen3.6-35B-A3B-NVFP4` (342 hand-picked BF16 ignores), PrismaQuant ships
**2 GB smaller, with ~90 fewer Linears in BF16, and wins 8 of 9 zero-shot
metrics**. That gap is what end-to-end measurement buys over guessing.

**Author / coordinates.** Robert Tand, independent researcher. GitHub
`RobTand/prismaquant`; HF artifacts under `rdtand/`. Paper: `paper/main.tex`,
now AURA-spined (*"AURA: Production-Faithful KL–Fisher Allocation"*; the
PrismaSCOUT spine was retired 2026-06-08). The flagship artifact (Qwen3.6-27B
PrismaSCOUT) has DOI `10.57967/hf/8656`. ~60k HF downloads across the family in
the first two weeks — that traction is treated as hard evidence and a hard bar.

**Attribution rule:** public/paper/model-card attribution uses
**robert.tand@icloud.com**. `tenari@gmail.com` is dev/system only — never
publish it.

---

## 2. Robert, and how we work together

- **He is paper-author-level on this material.** He weighs HALO vs ParoQuant
  himself, derives the allocator's Fisher expansion, and judges rotation
  tradeoffs.
- **Terse, demanding, allergic to slowness and to band-aids.** *"this is
  unreasonably slow"* / *"what has changed so wildly to make this perform like
  such dogshit?"* If a hot path crawls, that is a bug to fix, not a cost to accept.
- **Multi-model deliberation is his method.** On hard design questions he convenes
  Claude (facilitate/read/synthesize), **Codex** (implement + run production,
  burns freely), and **Gemini** (sparse, high-leverage math). Disagreement is
  expected and wanted; he resolves ties by deferring to whoever holds production
  evidence. The `scratch/deliberation/` and `.claude/codex-*` files are the
  archive of this.
- **Once a path is authorized, don't pause.** *"Wish you hadn't waited for me,
  we've lost an hour."* Surface a question only at a real fork with no default,
  or before something destructive/irreversible/outward-facing.
- **He runs long autonomous mandates** (*"iterate and make as much progress as
  possible,"* *"make changes yourself or delegate to codex"*) and wants an honest
  morning summary: what worked, what didn't, what's next.
- **Honest accounting is non-negotiable and is itself a deliverable.** He
  retracts his own overstated claims the moment a comparison is found
  non-rigorous (the grouped-KL "−3.52% win", the "4× lower KL" framing, the
  "17 promotions / 0.0056 KL" polish headline — all retracted by him). Negative
  results are recorded *with the durable lesson*. The retired PrismaSCOUT
  paper's rejected-methods catalog lives in
  `paper/archive/prismascout_paper_2026-06-05.tex`; the current AURA paper keeps
  the limits and caveats in its own spine. **Never sell a screen as a result.**
- **Verify "done."** Agent/wrapper "completed" signals fire before child
  processes exit. Check `git log`, read the log file, confirm the artifact.

**Why he's doing this:** to beat bigger labs with smarter *allocation* rather
than brute uniform precision; to ship public, paper-grade, vLLM-servable
artifacts that are smaller *and* better than the last; and to do it with
intellectual honesty that would survive a reviewer. The L3/cross-layer redesign
is the part he thinks *"could radically change all quantization if done right."*

---

## 3. The methodological spine

> **Surrogates generate, real KL selects.** *"An allocator does not need a
> perfect cost model if every candidate it proposes can be cheaply re-scored
> end-to-end on a held-out split."* Cross-layer interactions stop being
> quantities you must **model** and become quantities you **observe**.

Everything downstream follows from this. A cheap, biased surrogate *proposes*
candidates; expensive, faithful end-to-end KL on a held-out split *decides* what
ships.

**There is one cost level, not three.** The surrogate is a *single faithful
unary cost* over `(Linear, format)`, solved by a multi-choice knapsack DP:

- **The cost.** **`COST_MODE=aura` by default since 2026-07-30** (re-vet R2):
  `predicted_dloss` from a **KL-adjoint** probe × the production-rendered `dW`
  (`aura_cost.py`), worth **−38%/−39.5% KL @4B** across two calibrations and
  **−17.9% @27B** against the `h_trace × output_mse` baseline, and the recipe
  behind both flagships. `production-render-score` is the explicit/legacy
  spelling that reproduces every pre-flip artifact — per-Linear error measured
  on the **production-rendered** weights the export will actually ship (score
  field `weight_mse` since audit M6), so the surrogate, the KL validation and
  the exported bytes are the same rendering (principle 8). Either way the
  rendering identity is the point; what the flip changed is the *objective*
  applied to it. `COST_MODE` is a spelling over `COST_RENDER × COST_OBJECTIVE`
  (re-vet R3). On MoE the smooth
  cost is route-flip-blind for routed experts, so packed experts get **measured**
  empirical unit-KL instead (`expert_empirical_cost.py`), merged into one hybrid
  payload. Fisher normalization detail: every `h_trace` row — dense or MoE
  expert — is normalized by the same **global** calib token count, since tokens
  never routed to an expert contribute zero gradient to the mean-Δloss
  objective. Two superseded conventions were reversed on purpose: an explicit
  ÷route_prob (removed in audit M4) and the per-routed-token division M4 kept as
  "the implicit ÷token-fraction" (removed in PR #14 — it inflated rarely-routed
  unpacked-expert rows by global/routed, i.e. inverted importance weighting; see
  `finalize_fisher_stats`).
- **The selection.** Render the allocator's Pareto candidates, **measure real
  KL** on a **held-out** split, take the empirical Pareto frontier under
  η-dominance, and pick the shipping point. The bit-rate target is a **byte
  budget** ("fit the card", `footprint.py` exact) plus the saturation point B*;
  **kneedle is demoted to a diagnostic** — it is axis-dependent and
  LOO-unstable (fp32 4B: elbow at 5.00 in only 454/1000 bootstraps).
- **The (retired) guarantee.** Coordinate-descent polish *accepted* only
  single-unit flips that *strictly* reduced measured real KL — *provably no
  worse* than the chosen frontier point under the fixed polish-time evaluator
  (explicitly **not** an optimality claim, and re-validated end-to-end after
  export). That mechanism was **archived on 2026-05-15**
  (`archive/polish_2026-05-15/`); `run-pipeline.sh` does not invoke it, so the
  shipping pipeline ends at frontier selection and this paragraph is history,
  not a stage to budget.

### History: the retired cascade

PrismaQuant was described for most of its life as a three-level cost cascade —
**L1** additive Fisher → **L2** perturbed-X fixed point (re-measure each
Linear's output MSE under the activation distribution the current assignment
induces, re-solve, iterate) → **L3** propagated end-KL for a bounded
neighborhood, paired against a frozen L2 baseline. **That framing was retired on
2026-07-30** and the code walled at `archive/l3_propagated_2026-07-30/`
(re-vet R4). Read the banner there before proposing any cross-layer revival.

Neither L2 nor L3 was ever executed by the shipping pipeline, and three
measurements closed the question:

- **L2 beat additive L1 by −1.5%. AURA beat L1 by −38.5%** on the same baseline
  (`aura_cascade_headtohead`). A better *single* cost was worth 25× more than a
  second level.
- **The pairwise residual is +5–12% and diffuse, 3 of 1180 pairs significant**
  (`xlayer_sensitivity_2026_06_09`), and the apparent non-additivity was a
  **bf16 differencing artifact** — per-Linear KLs *do* add in fp32
  (`cross_layer_additivity_fp32`). Measure cost in fp32.
- **L3-polish-of-many does not compose:** per-Linear L3 costs measured under an
  L2 context don't sum to true end-KL when many flip at once (§9). Measured
  one-at-a-time coordinate descent is the safe form, and it is what survives.

The real-KL gate is what makes all of this safe rather than lucky: on the
Qwen3.6-4B 4.5-bpp microbenchmark L3-polish-DP *regressed* (0.371→0.461) and was
rolled back **by the gate**; coord-descent then recovered and surpassed
(→0.245, 6 of 101 flips accepted) — and none of those flips would have been
chosen by the additive surrogate.

**Naming:** the current paper's production-faithful allocator is **AURA**.
**PrismaSCOUT** is the retired prior cascade (Surrogate-Cascaded Optimization
Under Tradeoff); an earlier internal name for the L3-polish piece was
*PrismaClade*.

**Why CLADO/QUBO/HAWQ aren't the answer here:** the literature *models* the
cross-layer bias (pairwise IQP, 2nd-order ILP, Shapley games). Robert *observes*
it by gating every shippable candidate on real KL — and the measurements above
say there is nothing there to model. The full integer-quadratic program was
rejected: O(N²) per-pair measurement, for an optimum the O(N) path recovers to
within 1–2%. The decision-unit *framing* from CLADO is kept
(`decision_units.py`); the solver is archived.

---

## 4. Core design principles (non-negotiable — with the *why*)

1. **The platform measures and optimizes; it does not band-aid.** If the
   allocator picks something that breaks at runtime, **the measurement is wrong,
   not the optimizer.** Fix the cost model so it sees the real cost and converges
   on its own. Hardcoded format bans, streak limits, "demote because it looks
   dangerous", post-allocator rewrites — all **vetoed**, debug-only. *Mixed
   quantization is fine; every individual format is fine; errors must be bounded
   by the platform, not by constraints on what the allocator may choose.* *One
   carve-out, and it is principle 9's, not an exception:* a **measured platform
   fact** — the pinned runtime has no native route for these bytes on the
   declared target — may fail export closed (`allocator_candidates.py`
   route-status doctrine, `:133-137`). It never removes an honestly priced rung
   from the menu: an allocator that wants an unbacked route is *reporting a
   serving gap*, and that signal is the point. The line sits between "the
   runtime measurably does not serve this natively" (legitimate, P9) and "this
   format looks risky/ugly/slow to me" (vetoed). When stuck, ask: **measurement
   gap or optimizer gap? It is almost always a measurement gap.**
2. **No heuristics when an explicit exists.** Derive thresholds and decisions
   from the objective, never from intuition. The only acceptable constants come
   from the numerical precision of a dtype.
3. **Promote on the serving metric, not the screen.** Local-allocator / HF-PPL /
   last-token-KL screens are for triage. A win counts only when it holds on
   **exact full-vocab vLLM KL-vs-BF16 + direct WikiText PPL on the served
   artifact at matched bpp**. (grouped-KL's "−3.52% PPL win" *inverted* on the
   vLLM A/B and was archived. The staged-render last-token-KL "win" regressed
   direct PPL. The `current_only` extrapolation won the hook screen and lost
   full-vocab KL. This keeps happening.)
4. **KL is a *screening* metric, not a standalone promotion metric.** A candidate
   that improves calibration KL but regresses held-out PPL/NLL or a downstream
   task stays research-only unless Robert explicitly accepts the tradeoff. Lower
   *mean* KL can hide a heavier *tail* (the shipped 27B PrismaSCOUT has a worse
   max-prompt NLL and a stable adversarial tool-call regression vs the older 5.5).
5. **Format-first over GPTQ compensation.** *"I'd rather use a format designed for
   this paradigm than gptq."* GPTQ is calibration-fragile post-hoc error
   compensation. Prefer choosing the right `(format, transform)` per Linear;
   evaluate transforms in RTN-only contexts to isolate them from the GPTQ
   confound. Future allocator candidates are `(format, transform_package)` tuples.
6. **Provably-non-regressive bias; defaults stay backwards-compatible.** The
   allocator code has no install base (downloads are of the *static artifacts*),
   so it can be rewritten wholesale — but **(a) the serialization formats vLLM
   reads and (b) the quality of future shipped artifacts** are hard constraints.
7. **GPU-first / GPU-or-bust.** Every production hot path (probe, cost, cache
   fill, recache, export, validation — polish was on this list while it was
   live; it is archived at `archive/polish_2026-05-15/`) must be GPU-bound. CPU/disk/NVMe
   pressure on a hot path is a **bug** — the fix is to use/repair/extend the
   prefetch path so resident data is ready, never to tolerate the slow path.
   `run-pipeline.sh` and `gpu_guard.require_cuda_hot_path` refuse to run on CPU.
8. **One cache mechanism — and one execution contract.** Rendered weights flow
   *only* through `ProductionWeightCache`; activations through
   `PerturbedActivationCache` / the streaming activation path. No parallel
   stores. `pipeline.py` `APPROVED_RESOURCE_OWNERS` is the declarative contract
   and validation layer; runtime enforcement lives in the stage code and
   fail-fast cache/prefetch gates. **Why it matters:** the surrogate, the KL
   validation, the exported bytes, *and the activation/kernel contract the
   serving runtime executes on them* must be *identical* — a mismatch on any
   link is a confound (a "rendering confound" is the exact reason the JSO
   wall-off was reverted). Rendering identity **without** execution identity
   priced a real A-side at zero: NVFP4_CB, 2026-08-17 (measured on the Gridbook
   codebook lane, retired 2026-09-02 — the lesson is about rendering-vs-execution
   identity and outlives the lane). The executed contract is
   a fact about the pinned runtime and is established per principle 14, never
   asserted locally.
9. **Serving reality gates every format — per lane, and per artifact.** A
   format is production-eligible only when its lane's **pinned runtime**
   (vanilla vLLM for `compressed-tensors`, llama.cpp/vLLM-GGUF for GGUF, and
   the pinned Tessera release for the Tessera wire, served by Tessera's own vLLM
   plugin `tessera.serving` and admitted only through
   `prismaquant/tessera_runtime/tessera_serving_runtime_pin.json`, which is
   PENDING, so that lane admits nothing yet — the sanctioned **three**, per
   `AGENTS.md`)
   loads it on real shapes, generates correctly eager **and** graph-mode,
   doesn't break MTP/spec-decode (or is gated away), and routes it to a
   **native** kernel for the declared target platform — tensor cores on the
   hardware the artifact names, not an older-architecture schedule or a slow
   fallback. Eligibility is judged **per artifact, at export**: the
   serving-route provenance (`selection_serving_lane_provenance`) is a *gate
   input*, not a log — export fails closed when any selected unit's activation
   contract resolves to an unbacked route, unless the artifact explicitly
   declares a non-native target platform or carries an explicit per-run
   override, and either is stamped on the shipcard. Route status lives in a
   **structured** `route_status` field on every serving-profile lane
   (`backed | backed_with_serve_flag | unbacked`, plus `requires_serve_flags`),
   never in prose a gate cannot read. Registry support alone is not enough; a
   format may live in the research menu without being exportable.
   *(2026-08-17: 73.7% of a built 92 GB body rode a CUTLASS `arch::Sm80`
   fallback on sm121 Blackwell, recorded in its own `selection.json`, refused
   by nothing. Provenance that nothing consumes is not a gate; it is a
   confession log.)*
10. **CUDA graphs everywhere applicable** — by default, not on request. Eval paths
    are launch-overhead-bound (96% util at 11 W = launching tiny kernels). Capture
    fixed-shape forwards; env-gate the fallback; bit-exactness with capture *off*
    must always hold. (Exception: the one-shot exploratory loops — coord-descent
    flips — default to `auto`, which skips capture when warmup dominates.)
11. **BF16 and FP8_SOURCE are passthrough-only.** The allocator may pick them only
    when the source tensor is *already* that precision; never synthesize them
    (synthesizing BF16 from dequant'd FP8 wastes 8 bpp — *"shame!!"*). Enforced by
    `PASSTHROUGH_SOURCE_REQUIREMENTS` + a post-allocation assertion.
12. **Report bpp over *quantizable* parameters only — and report the serving
    route with the same honesty.** Exclude `lm_head`, profile-pinned Linears,
    MTP/visual sidecars. Published comparisons against uniform NVFP4 must use
    the same convention. (Beware: bpp labels are *not* comparable across
    accounting eras — the public "5.31" artifact's body bpp is ~4.76 under
    current accounting.) Every published size or quality claim carries the
    artifact's **activation-contract / route histogram** (from
    `selection_serving_lane_provenance`): a bpp or KL win on units riding a
    fallback route says so, on the card, in the same table as the bpp. A screen
    is not a result, and **a win on a non-native kernel is not a win on the
    named hardware** until the route is native or the card says otherwise.
13. **`docs/ARCHITECTURE.md` stays current — same commit, not later.** Any
    change to pipeline defaults, the stage graph, the format menu, the plugin
    contract, serving-lane defaults, or ship gates updates `docs/ARCHITECTURE.md`
    (and its diagrams if topology changed) in the same commit, and re-stamps its
    provenance block. Handovers and results docs are append-only history and
    never substitute for this. `tests/test_docs_staleness.py` +
    `tests/test_architecture_doc.py` enforce the mechanical half; the
    judgment half is on you. **Currency is not truth:** a doc updated in
    lockstep with a false belief is current *and wrong* (both `a24fce2` and its
    parent updated this file while encoding one). Claims about runtime
    behaviour are grounded per principle 14, not by doc discipline.
14. **A claim about another runtime is attested, never asserted.** Any
    producer-side spec field that states what a serving runtime *does* —
    executed activation contracts, kernel routes, backed rungs — is either
    **derived from** a machine-readable table the pinned runtime publishes, or
    **refused**. Prose `detail`/`rationale` fields explain; they are never the
    value a gate or a cost model reads. Concretely: the pinned serving runtime
    publishes its executed-activation-contract table inside its packaged
    `runtime_contract.json` — Tessera does this today, and the retired Gridbook
    lane (2026-09-02) did it before, computing it from
    `nvfp4_activation_contract.bridge_contract` and its per-scheme
    `activation_contract` strings;
    the producer preflight asserts every lane spec's `executes` list equals the
    pinned contract's table and refuses on mismatch, exactly as the R6 profile
    preflight already refuses producer/consumer profile drift. Second leg,
    serve-side: `validate_native_export` compares the artifact's priced
    `activation_contracts` histogram against the routes the serve actually
    emitted (`emit_route` telemetry) and refuses on disagreement — the priced
    contract and the served contract must be the same object. The test may not
    import the serving runtime (`AGENTS.md` principle 5 forbids vendoring it); the
    attestation travels in the contract file. **Corollary for prose:** a
    recorded blocker or capability claim inherits the scope of the artifact it
    was measured on — record the scope or do not record the claim. *Two of our
    own spec files disagreed about one runtime; the runtime was never
    ambiguous.*

15. **Measurement is first-class, and telemetry counts as measurement.**
    Every claim about speed, cost, residency, or *where the time went* is
    carried by a measurement, never by log-line reasoning. **Profile before
    the change and after it — the delta is the claim**, and a bench number
    without a profile does not establish where the time went. Two instruments,
    both required, because they answer different questions: an **in-process
    profiler** (`torch.profiler`, py-spy, `/proc/PID/io`, `nsys`) says where
    time goes *inside* a run; the **Netdata series on both boxes** says whether
    the box was actually loaded, which no in-process tool can see. Perf
    delegations put profiler evidence in the acceptance criteria.
    *Why it matters:* on 2026-08-28 the trellis encoder ran at 96% GPU
    utilization and 47 W of a ~140 W envelope. Utilization said "saturated";
    power said "one-third loaded". Power was right — fusing the step gave
    **5.83× throughput, bit-exact**, while utilization read 96% on both sides
    of the change and would have read 96% on an encoder ten times worse.
    On GB10, **`gpu_utilization` is non-diagnostic under load** (it means "a
    kernel is resident", not "the SMs are working") and `utilization.memory`
    returns a fake hard 0. **Read power against the envelope; rank by work per
    joule.** The envelope fraction also estimates remaining headroom, which
    wall-clock cannot. This is the measurement half of principle 7: GPU-first
    is a claim, and an unmeasured claim about a hot path is not evidence.


---

## 5. What counts as a win (measures of success)

**Metric authority, highest first:**
1. **Exact full-vocab vLLM KL-vs-BF16** on WikiText, served-artifact, matched bpp
   (the canonical contract is n=8 × seqlen=512). The gold metric.
2. **Direct WikiText PPL** on the served artifact (8192 tokens, seqlen 512) —
   can veto a candidate even when a narrow KL screen improves.
3. **Mean NLL** alongside PPL; for IT/BOS-sensitive models use **KL-vs-BF16**
   (`/home/rob/dq-runs/kl_tool.py`) — raw PPL is garbage on instruct models.
4. **Downstream task suite** for materialized artifacts: GSM8K, IFEval, MMLU,
   and **ToolEvalBench** (sequential, `--no-think --hardmode --parallel 1`).
   Tool-use fidelity is a *deep* reason for choosing KL: a small probability shift
   at a decision point flips a tool call.
5. **Cheap last-token "hook KL" screens** — triage only, *never* final selection.

**Gates & discipline:**
- **Held-out split is disjoint from cost generation.** A prior audit found the
  "validation" KL was in-sample; selection KL must use text the surrogates never
   saw. Kneedle and artifact metadata use the held-out split (archived
   coord-descent did too, while it was live).
- **Reproducibility/provenance is a gate.** KL is bit-identical *within* a docker
  session but can drift 4–8× *across* sessions (stale live-model state). Bake
  git commit, calibration hash, assignment hash, cache hit/miss/RTN-fallback
  counts into output JSON. An irreproducible number is *quarantined*, not trusted.
- **Validate at small scale first:** every claimed lever on Qwen3-**0.6B** (fast)
  **and 4B** (scale check), with `--calib-repeats ≥ 4` (single-seed n=8/T=512 is
  dangerously noisy: +10% can flip to −5.2% across reps; between-seed std ~0.02).
- **Pre-ship gate** (`validate_quantized_model.py`): vLLM actually serves;
  generation is coherent; PPL < threshold **and p99 per-prompt NLL** < threshold
  (p99 added after a broken 27B passed on mean while 80% of prompts were broken).
- **No HF upload without `tools/publish_artifact.py`** — it refuses on an
  unverified shipcard (pipeline gates stay advisory; **publication** is the
  blocking point). `--force-unverified` needs the artifact dir's basename
  re-typed and stamps `forced_unverified` onto the card.
- **Promotion ladder:** Research (opt-in, documented, excluded from defaults) →
  Candidate (small-model GPU+vLLM smokes + a 27B measurement plan) → Production
  recipe (wins/preserves KL/bpp/runtime on the target stack + serving suite +
  tests) → Default-on (cleared on the target **and** one more representative
  model/shape). Regression or inconclusive → demote back to research.
- **Sobering calibration on expectations:** most pipeline "improvements" are
  <5% PPL deltas; the cost surrogate is even mis-ranked vs PPL *at the margin*
  (5.5 bpp beats 6.0 bpp on Qwen3-4B PPL). Over-prioritize correctness and
  principle over the illusion of rapid progress.

---

## 6. Architecture & code map (validated against the tree)

**`run-pipeline.sh` (at `prismaquant/run-pipeline.sh`, not the repo root) is
the real orchestrator.** `pipeline.py` is a *declarative
contract* layer (typed stages/gates/`APPROVED_RESOURCE_OWNERS`), **not** the
executor. Stages are file-artifact-coupled and skip-if-exists. `WORK_DIR/`:
`artifacts/` (probe.pkl, cost*.pkl, layer_config.json, pareto*, production cache),
`act/`, `work/` (shards), `logs/`, `exported/`.

```
incremental_probe ─► probe.pkl            (per-Linear empirical Fisher H_trace; streaming)
   │
incremental_measure_quant_cost            (per-(Linear,format) error — L1 baseline)
 OR production_render_cost                (default: derive allocator cost from a dedicated
   │                                        render-score cache; export/validated KL use
   │                                        their selected-assignment production cache)
allocator + allocator_solver ─► layer_config.json + pareto*    (knapsack DP + log-error kneedle)
   │                                        (fused-sibling & packed-MoE union-find promotion)
build_production_cache / production_recache ─► ProductionWeightCache
   │
[validated-surrogate, opt-in] validate_assignments_kl ─► select_validated_frontier
   │                                        (real held-out KL per Pareto point → kneedle)
export_native_compressed ─► exported/      (compressed-tensors checkpoint)
   │
validate_native_export   (vLLM eager+graph load + greedy smoke)
validate_quantized_model (PPL / p99-NLL / MMLU / MTP-acceptance ship gate)
```

**Defaults in `run-pipeline.sh` today:** `TARGET_BITS=4.75`,
`FORMATS=NVFP4,FP8_DYNAMIC,BF16` (note: **MXFP8 is de-menued for inference** —
exact-scale FP8 Pareto-dominates it), `NSAMPLES=32 SEQLEN=1024`,
`PRODUCTION_CACHE_LEVERS=gptq,static_act_order,joint_scale_opt`,
**`COST_MODE=aura`** (flipped from `production-render-score` on 2026-07-30,
re-vet R2 — both flagships were produced with it; `production-render-score` is
the explicit/legacy spelling that reproduces every pre-flip artifact, and a
stale `WORK_DIR` now **rebuilds loudly** on the mode change instead of silently
allocating on the other estimator), **`SELECTION_MODE=surrogate`** (set
`validated-surrogate` to opt into the real-KL frontier selection that produced
the shipped 27B), `TARGET_PROFILE=<unset, spec-resolved>`
(`TARGET_PROFILE_DEFAULT=vllm_packed_moe` is the fallback when an architecture
declares nothing — the shell itself pins nothing, re-vet R11), `PRODUCTION_CACHE=1`,
`PRODUCTION_RECACHE=1`, `VALIDATED_SOURCE_PREFETCH=require`,
`AURA_ADDITIVITY_GATE=measure`, `CB_EXPERT_EMPIRICAL=<unset>` and
`CB_SCALE_CODING=<unset>` (both lost their shell defaults with the Gridbook
lane on 2026-09-02 and survive only as `${VAR:-}` settings-hash entries).
`docs/ARCHITECTURE.md` §3.3 is the single source of truth for shell defaults.
The archived cost modes / levers (`grouped-kl`, fisher, hdq, multi-shot,
production-render-staged) **fail fast with `exit 2`** pointing at their archive.

`COST_MODE` is a **spelling** over two axes since 2026-07-30 (re-vet R3):
`COST_RENDER ∈ {inline, cached-menu}` × `COST_OBJECTIVE ∈ {weight-recon,
render-score, aura-adjoint}`. The CB/GGUF lanes are no longer pinned to
`COST_MODE=local` — that gate named a *render* property to block an *objective*;
it is now a render-faithfulness assertion, and `col_weights` on the production
render path (CB Milestone C) lets a cached-menu render be the exporter's
imatrix-weighted render. Non-`local` objectives on CB are reachable but
**opt-in and not recommended** until a served CB A/B exists.

**Subsystem owners (the files that matter):**

| Concern | Files |
|---|---|
| Orchestrate / contract | `run-pipeline.sh` (exec) · `pipeline.py` (declarative spec + owner validation, not executor) · `__init__.py` (transformers-5.x polyfills) |
| Allocate | `allocator.py` (CLI, Pareto sweep, log-error kneedle, `--bit-attribution-json/csv`) · `allocator_solver.py` (numpy multi-choice knapsack DP, union-find serving-unit promotion, `predicted_dloss`) · `allocator_candidates.py` (legality gate + cost-source precedence + passthrough integrity) · `decision_units.py` |
| Probe / cost | `incremental_probe.py` + `sensitivity_probe.py` + `kl_fisher.py` (Fisher probe) · `incremental_measure_quant_cost.py` + `measure_quant_cost.py` · `production_render_cost.py` (default cost) · `aura_cost.py` + `expert_empirical_cost.py` (`COST_MODE=aura`, MoE hybrid) · `kl_measurement.py` (whole-assignment KL + per-sequence tail) · `perturbed_x_cache.py` (activation cache / model loading — **not** a cost stage). The L3 propagated stack is walled at `archive/l3_propagated_2026-07-30/` |
| Formats | `format_registry.py` (FormatSpec + RTN `quantize_dequantize`/`activation_quantize_dequantize`, codebook bucketize, E8M0 snap, `torch.compile` hot path) · `mx_formats.py` · `fp8_dynamic.py` |
| The one cache + render | `production_weight_cache.py` (`ProductionWeightCache` + `render_production_weight`) · `build_production_cache.py` · `production_recache.py` · `render_score.py` (output-MSE render scorer + gate + mechanism registry/topo-order) · `layer_state_cache.py` |
| Streaming (huge models) | `layer_streaming.py` (LRU + pressure shrink + FP8 block dequant + per-expert→packed bridge) · `streaming_model.py` · `weight_session.py` (stage/revert/commit live-model format flips) · `source_prefetch.py` (fail-fast residency gate) · `autoscale.py` |
| Export | `export_native_compressed.py` (~7300 lines: NVFP4/MX/FP8 packing, unified codecs, `build_quantization_config` config_groups+ignore, packed-MoE split, FP8_SOURCE verbatim copy, BF16-upgrade audit) · `export_batched_gptq.py`. (`block_output_match.py` walled 2026-07-30 — `archive/block_output_match_2026-07-30/`; it was unreachable on the shipping recipe.) |
| Validate / select | `validate_assignments_kl.py` · `select_validated_frontier.py` (kneedle + surrogate-vs-KL Spearman + worst-rank-inversion) · `validation_harness.py` · `validate_native_export.py` · `validate_quantized_model.py` · `shipcard.py` (the refusal contract) · `lane_spec.py` + `lane_specs/*.json` (per-lane serve command / endpoint / gate set / KL evaluator — gates ADVISORY) · `gguf_kl_evaluator.py` (llama-perplexity behind the `validate_assignments_kl` interface) |
| Profiles (plug-in) | `model_profiles/`: `base.py` (auto-derives fused/packed/naming from the vLLM class), `structure.py` (declarative `ModelStructureSpec` JSON + `build_model_graph`), `registry.py` (`detect_profile`/`register_profile`), `vllm_registry.py`, + per-arch (`qwen3*`, `gemma4`, `lfm2_moe`, `minimax_m2`, `deepseek_v4`). Serving constraints live in the top-level `serving_profiles.py` + `serving_profile_specs/*.json` |
| Misc | MTP synthesis (transformers v5 drops it) now lives on the profile: `model_profiles/base.py` `mtp_source_prefix`/`read_mtp_source_state_dict`/`load_mtp_state_dict` + `model_profiles/qwen3_5.py::MtpModule` (top-level `mtp_module.py` deleted 2026-07-30) · `layer_config.py` (single canonical recipe parser) · `schemas.py` |

**Plug-in a new architecture = three registries, ~30–200 LoC:** model structure
(`model_profiles/specs/*.json` + a `ModelProfile`), serving constraints
(`serving_profile_specs/*.json`), pipeline contract (`pipeline.py`). `base.py`
auto-derives fused groups / packed-expert maps / vLLM-internal names from the
vLLM class's `packed_modules_mapping` + `hf_to_vllm_mapper`, so most of the work
is `matches()` + `vllm_architecture_class()`.

**Hard serving invariants** (these break vLLM loading if violated): fused
siblings (q/k/v, gate/up) and packed MoE experts must share **one** format
(union-find promotion enforces it); packed-expert `config_groups` must use vLLM
**canonical** scheme names (`gate_proj/up_proj/down_proj`) even when on-disk
leaves are `w1/w3/w2`; incomplete fused-sibling groups (e.g. Gemma4 `k_eq_v`
layers shipping no `v_proj`) are forced to BF16 + added to the ignore list;
experts must be uniform per-layer (mix across layers, not within).

---

## 7. Formats, hardware, environment

**Production format menu (what vLLM serves natively):** NVFP4 (W4A4, group 16,
FP8 block scale, CUTLASS on Blackwell) · FP8 dynamic / FP8_E4M3 (the 8-bit path)
· BF16 (passthrough) · FP8_SOURCE (verbatim copy of native-FP8 source + scale_inv,
lossless, ~8.002 bpp, **dominates MXFP8 on already-FP8 layers**). **MXFP8 is a
training-native format, excluded from the default `FORMATS` menu** — not denied by
the serving profile (`vllm_packed_moe` still allow-lists `MXFP8_E4M3`; only
`MXFP8_E5M2` is denied), but its E8M0 pow2 scale wastes ~√2 of a binade (+13.8%
output MSE over 410 Gemma Linears vs exact-scale FP8), so exact-scale FP8
Pareto-dominates it and the allocator never picks it when both are offered.

**Research / non-served:** NVFP4A16, MXFP4 (served but rarely chosen),
INT4/INT8 and MXFP8_E5M2 (registry entries, no served path). **MXFP6 was
removed from the registry entirely on 2026-09-01** — no hardware or runtime
supports it (`docs/design/mxfp6_cb_feasibility.md`: FP6 rides Blackwell's 8-bit
datapath, no separate rate, operands byte-padded).
**E5M2** is only a valid *kv-cache* dtype. `prismaquant/kernels/` ships only
`nvfp4_fused.py`.

**JSO (`joint_scale_opt`)** is the production NVFP4 implicit-clipping mechanism: it
maps to the `joint_mse` scale rule whose per-group levels default to **{6,4}**
(commit `cdb3022`), evaluated *inside* the GPTQ loop with activation-weighted MSE.
(The `PRISMAQUANT_NVFP4_SCALE_RULE` env-default `static_6` only governs non-JSO RTN
renders; `four_over_six_mse` is a separate, non-JSO rule — don't conflate the
three.) JSO subsumed the separate clip solvers (*"clipping is just another way of
asking what the right scale is, and JSO already answers it"*).
**The GPTQ `damp_sweep` is OFF by default since 2026-06-12; fixed damp 1.0.**
The sweep's evaluator was found in-sample (held-out basins invert 31/31);
the V1 served A/B had fixed-0.3 beating the sweep on every gold-lane readout
across two calibration draws at ~4.4× less render time, and the old
"+137.5% if disabled" claim was a tier-5 hook screen that inverted on the
gold lane. `PRISMAQUANT_GPTQ_DAMP_SWEEP=1` reproduces historical artifacts;
`PRISMAQUANT_GPTQ_DAMP` overrides the constant. Open research: derive the
per-Linear optimum from weights alone (docs/design/unified_render_theory.md).

**Hardware:** NVIDIA **GB10 / DGX Spark** ("sparky"), Blackwell sm_121, **128 GB
unified memory** (~121 GB usable serving budget — GPU and host share one physical
pool, so "move to CPU" is a no-op for memory pressure). The full speed stack
(fused kernel + 5 CUDA-graph paths + replay + caches) stacks past the budget and
OOM-kills; the shipping path runs eager/default-pool with the act cache + frozen
cache as the only large state. 1.8 TB NVMe.

**Toolchain:** build/probe/PPL venv `/home/rob/dq-runs/venvs/prismaquant-cu130`
(torch 2.11+cu130; host `.venv` lacks torch; `PYTHONPATH=.` for tests). Serving
uses separate vLLM venvs/containers: `vllm-fresh-b12x:latest` (transformers 5.5.4),
`vllm-node-tf5-cu132-lfm:latest` (LFM image with `causal-conv1d` +
`flash-linear-attention`). Qwen3.5/3.6 need `fla` (git source — PyPI wheels are
broken) + `causal-conv1d`, installed `--no-deps` so torch is never touched.

---

## 8. Current state (snapshot — verify against `git`/memory before relying)

Branch at last edit: **`claude/docs-consolidation`** (2026-07-30: docs
centralization + `docs/ARCHITECTURE.md` + cleanup; = NVFP4-CB lane +
origin/main merged, i.e. the allocator PR stack #13–#21, closures #27–#29, and
the v0.4.1 release stack). **`origin/main` is the integration branch — the
local `main` checkout had drifted 54 commits behind it; check
`git fetch && git log main..origin/main` before trusting local `main`.**

**Shipped public artifacts (`rdtand/`):**
- **Qwen3.6-27B PrismaSCOUT** — 5.31 bpp / held-out KL 0.0151, 20.17 GB; 11%
  smaller and 68% lower KL than the prior v1 5.5 (0.0475). DOI `10.57967/hf/8656`.
  *The flagship.*
- Qwen3.6-27B v1 5.5 (validator PPL 4.16, MTP 89.5%); 5.31 / Heretic-5.25 variants.
- Qwen3.6-35B-A3B 4.75 (predates 4 allocator/export fixes — **don't re-export
  without an orthogonal reason**).
- Qwen3.5-122B-A10B 4.75 · Mistral-Medium-3.5-128B 4.75 · MiniMax-M2.7 3.2.
- **LFM2.5-8B-A1B 6.5** (2026-05-29; repo-labelled 6.5bit, achieved ~6.58 bpp;
  ToolEvalBench = BF16 parity).
- **Gemma4-31B-IT 6bit** (2026-06-01; beats the shipped 5.5 by −24% *confident-position*
  KL-vs-BF16, +5.9pp top-1 agreement; the 5.5 repo is left untouched).

**Live recipe:** `gptq` (fixed damp 1.0, sweep OFF) + `static_act_order` +
**JSO** + the PrismaQuant solver; **AURA cost** (`COST_MODE=aura`, the default
since 2026-07-30) with the empirical packed-expert unit-KL hybrid on MoE and the
`[3c]` additivity report stamped into `cost.pkl` provenance; surrogate selection
by default, validated-surrogate for real-KL frontier selection (automatic under
a `TARGET_DISK_GB` card).

**In flight / open (and the stale claims to ignore):**
- **DSv4-Flash-Base** (284B FP8 source): vendored transformers (`PR #45643`) + 3
  monkey-patches, vLLM 0.20 native, **pruning disabled** (REAP under-counts
  misrouting cost). ~88 GB at 2.5 bpp. v2 SVD expert-factorization design deferred
  (prefer no new vLLM kernel).
- **Robust Fisher clip** (K×median per-role h_trace clip): a research **WIN**
  (~5% better at 6.0 on 4B). **Wired as an opt-in lever 2026-07-30 (R28)**:
  `PRISMAQUANT_FISHER_CAP_MULTIPLIER=K` → `allocator.clip_probe_fisher_outliers`,
  unset = byte-identical no-op. Still **research** — no served A/B; do not
  default it on. Reference tool: `/home/rob/dq-runs/robust_fisher_clip.py`.
- **Production-faithful polish** (5.39 bpp / polish-time KL 0.0054): **provisional**
  — that number is a polish-time signal on a 2×128 calibration, **not** a held-out
  8×512 claim. Do not cite −2.8× as a result until the 8×512 re-measurement lands.
- **Low-bit kernel lane** (`references/lowbit-kernels/`): direction agreed
  (MicroMix-style mixed MXFP4/MXFP8 per-channel kernel to fill the
  NVFP4→MXFP8 4× gap; the knee lands at 5.0–5.7 bpp across his models) but not
  implemented; *"chase the hardware, not old SoTA."*

> **⚠ Stale-doc corrections (as of 2026-06-04):**
> - **The "27B JSO isolation A/B" is CANCELLED — do NOT re-run it.** The
>   2026-05-28 handover still lists it as an open item; it is not. Robert decided
>   *"we don't need a JSO isolation test. We already know JSO works beautifully
>   when picking either 4/6."* JSO **ships**; the memory file named
>   `jso-archived-*` is now a misnomer.
> - **The 2026-05-28 handover's *other* open item is also closed:** MoE expert
>   projection-name pluggability (DSv4) is **DONE** — routed through
>   `model_profiles` accessors (`base.py`:
>   `packed_/unpacked_expert_projection_names`,
>   `vllm_fused_moe_scheme_projection_names`), tested in
>   `tests/test_moe_expert_projection_names.py` (5/5), on HEAD via `fe25ce5`/`396504e`.
> - **The public PrismaSCOUT HF README is partly historical.** Its narrative calls
>   L2 a "Lagrangian-relaxed QUBO" and lists HALO/AutoRound/SAO/scale_sweep as the
>   per-layer toolkit. The **shipped code** uses the perturbed-X fixed point for
>   L2 (QUBO/SMRF is archived, default-off, *rejected as a selector*), and
>   HALO/Fisher/SAO/scale_sweep/PrismaClip are all **archived**. Trust the code.
> - The 2026-05-20 handover's grouped-KL "−3.52% PPL win" is **superseded** — it
>   lost the vLLM A/B and is archived.

---

## 9. The graveyard (archived/rejected — and the durable lesson)

Robert treats this as institutional memory; the paper publishes it. Each lives
under a dated `archive/<name>_YYYY-MM-DD/` wall; the four production cost-mode /
lever ones (grouped-KL, Fisher, HDQ, multi-shot) **fail-fast with `exit 2`** from
`run-pipeline.sh`, while the rest are simply absent from the default recipe
(`scale_sweep` is still reachable via `--enable scale_sweep` for ablations).
**Do not revive any of these into production paths without an explicit ask.**

| Method | Why it lost (the lesson) |
|---|---|
| **The Gridbook codebook lane (NVFP4-CB / FP8-CB)** | Retired **by decision, not by defect**, 2026-09-02 (`archive/gridbook_lane_2026-09-02/`) — Rob: *"put Tessera in PrismaQuant and remove Gridbook."* It won where it was measured (27B: −45%/−53% confident-position KL at 19.93 GB vs a 23 GB native artifact; 35B: 0.01706 vs 0.03625; the 105.73 GB Hy3 295B artifact came off it) and its shipped artifacts are **not** withdrawn. What it cost was a second serving runtime to pin, attest and keep in step, for one non-native wire; Tessera replaces that wire with one plugin lane. *A lane is priced by the runtime it obliges you to maintain, not only by the KL it wins.* Two second-order lessons kept: an unbacked route that nothing consumes is a confession log, not a gate (principle 9, 2026-08-17 NVFP4_CB); and a lane's tendrils outlive it — the sample-parallel probe and DeepSeek-V4's role-composite exemption were both coupled to a *campaign* census inside it (debt D34). |
| **The three-level cost cascade (L1→L2→L3)** | Retired from the spine 2026-07-30 (`archive/l3_propagated_2026-07-30/`). L2 beat additive L1 by −1.5%; AURA beat L1 by −38.5%. Cross-layer residuals +5–12% diffuse, 3/1180 pairs significant, and the non-additivity was a bf16 artifact. *One faithful cost beats another level.* |
| **`COST_MODE=production-render-staged`** | Rendered NVFP4 first and promoted only the top-30% error tail, so the DP could not consider anything outside it. 27B: last-token-KL screen improved 0.0232 vs 0.0280, direct PPL **regressed 10.83 vs 8.33** — "Do not ship". The canonical screen-vs-gold inversion. `archive/production_render_staged_2026-07-30/`. |
| **`MSE_PROMOTION`** (post-frontier rewrite) | Re-ranked the already-KL-selected point by local `output_mse_per_bit`. On 35B: beat the strategic baseline, lost to both the shipped 4.75 and the 5.16 kneedle. *A post-allocator rewrite cannot beat a better cost inside the DP.* `archive/mse_promotion_2026-07-30/`. |
| **`PRODUCTION_CACHE_UNION`** (smart-union cache) | Offered an FP8 rung only above a percentile of the NVFP4 `output_mse` surrogate — a render-budget heuristic deciding the allocator's candidate set (principle 1). `archive/union_cache_2026-07-30/`. |
| **Block-output match** (lever #12) | **Unreachable**, not unmeasured: the production-cache pack `continue`s first, so it never ran on the shipping recipe (0 hits in two real export logs); had it run it would have discarded the render's `joint_mse` scales (M19). Subsumed by JSO. *Check the code executes before funding an A/B.* `archive/block_output_match_2026-07-30/`. |
| **grouped-KL** cost surrogate | "−3.52% PPL" was a local/HF screen; **lost the vLLM A/B**. *Promote on the serving metric.* |
| **CLADO** full IQP solver | O(N²) per-pair; the O(N) cascade matched it to 1–2%. Keep the decision-unit framing, drop the solver. |
| **L3-polish-of-many DP** | Non-additive: per-Linear L3 costs measured under L2 context don't sum to true end-KL when many flip at once. Coord-descent (one-at-a-time, measured) is the safe alternative. |
| **Multi-shot recalibration** | Double-negative: ΔKL=0 at production cal; one budget −153% on small cal. |
| **scale_sweep** (as default) | +77.5% KL on 4B (re-picks block scales *after* GPTQ, mis-calibrating its error compensation). JSO fixes scales *inside* the loop instead. |
| **Analytical/closed-form damp** | +100–161% KL vs the 5-candidate sweep; the fit's 2.4× per-Linear error compounds. Cheap discrete sweep wins. |
| **HALO / Hadamard-Fisher rotations** | Worked on Qwen3.5 dense after fixes, but cut in the 2026-05-15 consolidation. ParoQuant (`2511.10645`) is the tracked replacement. |
| **PrismaClip / PrismaFisherClip** | *"It's useless."* Subsumed by JSO's per-block scale grid. |
| **SAO** (column permutation) | Failed its own objective; redundant with GPTQ's full-Hessian propagation. |
| **ReSpinQuant / layer-wise rotations** | Need a residual-transition adapter (a custom kernel) at serve time — forbidden in vanilla-vLLM artifacts. |
| **REAP / expert pruning** | Cost model under-counts token redistribution / misrouting. Hit size via factorization/rotation/sub-NVFP4, not pruning. |
| **Surrogate-only knee** | On 27B the surrogate knee picks 5.857/0.056; validated picks 5.31/0.015. Outside the additive trust region, bpp order ≠ KL order. |
| **Lagrangian λ-bisection (as selector)** | The discrete frontier has non-convex pockets no λ selects; the surrogate inside it still overshoots 30–50%. Kept as a *candidate generator* only. |
| **Sparse pairwise QUBO/SMRF** | 8-of-~500-Linear coverage is "homeopathic" — too local to fix global non-additivity, expensive enough to dominate the budget. Default-off. |
| **Top-K Hessian covering** | Blind to the propagation graph; misses Linears with a small eigenvalue but a long downstream path. |
| **Top-down / ceiling-start polish** | Spends its whole budget on cheap ~12-bit flips, never reaches the knee bpp range. |

---

## 10. Operational landmines (lose-hours-or-an-artifact if forgotten)

- **Never write to `/tmp`.** It was cleared by an OOM (2026-04-23) and wiped the
  MiniMax artifacts — *"for the love of God."* Artifacts go under `/home/rob/` or
  `/models/`. Set `TMPDIR=` explicitly if a tool defaults to `mkdtemp()`.
- **Keep ≥10% disk free** (~180 GB of 1.8 TB). A 27B cache is ~90 GB; a multi-arm
  matrix is bounded by *peak*, not final state. `df -h /home/rob` before launching;
  build→PPL→delete-before-next.
- **Spec-decode poisons PPL:** `/v1/completions` echo+logprobs returns the *draft
  (MTP) model's* NLL when vLLM serves with `--speculative-config`. Run perplexity
  on a **no-spec-decode** serve. (`validate_quantized_model` now refuses a verdict
  if it sees `vllm:spec_decode_*` in `/metrics`.)
- **Gemma raw PPL is broken** (BOS dropped → ~ln(vocab) garbage). Use KL-vs-BF16.
  Raw PPL can't distinguish quantizations of instruct models anyway.
- **Activation device-residency landmine:** tensors from `_LazyActivationCache.get()`
  are CPU-resident; `.to(device, float32)` them explicitly in batched/sweep paths
  or the matmul silently runs on CPU (no speedup). Recurs across export work.
- **`transformers` pin is model-specific:** inside `vllm-fresh-b12x` do **not** pin
  `transformers==4.57.5` for Qwen3.5/3.6 (lacks the model types → KeyError; use the
  image default 5.5.4). MiniMax, conversely, *requires* 4.57.5.
- **Don't kill docker mid-build** — root-owned files on bind mounts need `sudo`.
- **GPU wedge:** HALO multimodal head materialization and some DSv4 loads can wedge
  the GPU into a D-state process `SIGKILL` won't reap — only `sudo reboot` recovers.
- **Don't change a kernel mid-A/B** — it invalidates the baseline arm. Install
  fast-path kernels *between* runs, torch-safe.
- **vLLM serve binding:** always `--host 0.0.0.0 --port 8000` + `docker -p 8000:8000`
  (Robert tests from opencode on another LAN host; a loopback bind is unreachable).

---

## 11. Where to read more — and how skeptically

Read these for *depth*; the prime directive applies.

- **`docs/ARCHITECTURE.md`** — **the master document. Start here.** Single
  current system map: pipeline stages + defaults, cost models, formats, export,
  validation, plugin architecture, the three serving lanes, hardware, history,
  and an honest debt register — every normative claim code-cited, three mermaid
  diagrams, provenance-stamped. Carries the maintenance contract (§0): any
  change to defaults, stages, formats, plugin contract, lanes, or ship gates
  updates it in the same commit. `docs/README.md` is the index of everything
  else, with per-doc status labels.
- **`docs/archive/prismaquant_design.md`** — the retired 2026-06 master design
  doc; superseded by `docs/ARCHITECTURE.md` (banner on the file). Historical
  depth on the cascade era only; several claims known-stale.
- **`AGENTS.md`** + **`docs/design/design_guidelines.md`** — the terse normative rules
  (the 9 core principles, promotion gates, progressive render-gate contract,
  rotation-transform rule, exception rule). Mandatory pre-read for new work.
- **`docs/design/runtime_flags.md`** — the env-flag vocabulary (`PRISMAQUANT_*`,
  `COST_MODE`, `SELECTION_MODE`, `PRODUCTION_CACHE_LEVERS`, CUDA-graph autos).
- **`docs/design/progressive_render_pipeline.md`** + **`docs/design/pluggable_refactor.md`**
  — render-gate ordering and the plugin contract. (`docs/archive/propagated_cost.md` is the
  L3 propagated-cost spec: **archived**, matching the code wall at
  `archive/l3_propagated_2026-07-30/`.)
- **`paper/main.tex`** — the current AURA spine: production-faithful KL--Fisher
  allocation, additivity/cancellation analysis, served KL/PPL evidence, and
  limitations. `paper/figures/fig_aura_rd_geometry.tex` is the key geometry
  figure. The retired PrismaSCOUT paper, including monotone-polish and the
  rejected-methods catalog, is archived at
  `paper/archive/prismascout_paper_2026-06-05.tex`.
- **`docs/handovers/prismaquant-handover-*.md`** (moved from `.claude/`
  2026-07-30; gitignored, local-only) — session history, newest first. Useful
  for narrative arc; **assume the "open items" are stale** (the 2026-05-28 one
  lists the cancelled JSO A/B). Codex/Gemini deliberations: `.claude/codex-*`,
  `scratch/deliberation/`.
- **Auto-memory** `/home/rob/.claude/projects/-home-rob-prismaquant/memory/` — the
  *freshest* cross-session truth; `MEMORY.md` is the index with current/superseded
  status per lever. When a handover and a memory note conflict, the memory note
  is usually newer.
- **`references/prismascout-prior-art/`** — the papers PrismaQuant positions
  against (CLADO, HAWQ-V3, AMQ, ImPQ, ParoQuant, CoopQ/Shapley, geometry/Babai).

When you find a doc that contradicts the code or a serving result, **fix the doc
(or flag it) — don't propagate it.** That is the house style here.
