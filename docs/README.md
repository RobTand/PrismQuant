# docs/ — index

**`docs/ARCHITECTURE.md` is the master document — start there.** Everything below
is either a rule set it points at, a lane record, or history.

**Maintenance rule:** a claim is true only if current code or a served measurement
backs it; every normative statement carries a `file:line` or commit hash, and when a
doc and the code disagree the doc is wrong — fix it or flag it, never propagate it.

As of 2026-07-30 (tree reorganised in `87c749e`). Status tags: **CURRENT** =
describes the live system · **HISTORICAL** = dated record, true when written, not
guidance · **ARCHIVED** = superseded narrative, kept for provenance.

Three things older docs get wrong and this index does not:
`COST_MODE` defaults to `production-render-score` (`prismaquant/run-pipeline.sh:187`)
and AURA is opt-in (`COST_MODE=aura`, `:314-336`, `:825-956`);
`SELECTION_MODE` defaults to `surrogate` (`:250`);
`run-pipeline.sh` lives at `prismaquant/run-pipeline.sh`, not the repo root.

---

## design/ — current normative

| Path | What it is | Status |
|---|---|---|
| `design/artifact_collections.md` | Content-addressed, format-agnostic control plane for probe-once/export-many collections: explicit candidate and target records, immutable stage receipts, shared-resource accounting, and a legacy Qwen artifact census. | CURRENT foundation — schemas and tests ship offline; ModelSnapshot/Probe/Cost/Solve/Export/Qualification adapters and pipeline wiring remain open |
| `design/model_coverage_ledgers.md` | Requirement discovery by traversal: one walk over the loaded model tree plus a traced forward discovers every parameter->op edge; each node is claimed at discovery (decide / pin+reason / exclude+reason) and an unclaimed node fails the walk. All consumers (probe, cost, footprint, read-bytes, routes) derive from the one enumeration; four stamped views reconcile against the checkpoint header. Adopted after the `wo_a` finding. The walker landed 2026-08-21 (`prismaquant/model_walk.py` + `ModelProfile.walk_claim_rules()`, ARCHITECTURE.md §8.8); intake walks are usable, but export-gate wiring and consumer migration onto the edge list are still open. | CURRENT |
| `design/gridbook_lane_eligibility_contract.md` | The `lane_eligibility` table Gridbook must package inside `runtime_contract.json` so CB serving-lane route status becomes attested rather than asserted (campaign R3, principles 9 and 14): the schema, the closed set of structural facts a predicate may name, and the four rules the DSv4 per-role case needs. Records what Gridbook 0.8.10 actually publishes and why none of it answers the lane question. | CURRENT — proposal to the Gridbook repository; the PrismaQuant consumption half ships (ARCHITECTURE.md §9.2.1), so every CB lane resolves `unattested` until the table lands |
| `design/design_guidelines.md` | The terse rule set: non-negotiables, measurement discipline, promotion ladder, rotation-transform rule, exception rule. Makes almost no code-line claims, so nothing has gone stale. | CURRENT |
| `design/constrained_pareto_allocation.md` | Serving SLOs as the allocator's second hard axis (ultraplan P5c): the measured dispatch-table schema and its mandatory per-row provenance, the additive layer-time aggregation model and its eight named assumptions, why the filter sits in the byte-budget ratchet rather than in `solve_allocation`'s DP, the shipped example table's row list and sources, and the D0.3 exact-rate harness (P5d). | CURRENT — normative policy is `lanes/nvfp4-cb/format-speed-policy.md` §1; everything the axis produces is proposal data pending the served NATIVE-PARITY protocol |
| `design/runtime_flags.md` | The `PRISMAQUANT_*` / `COST_MODE` / `SELECTION_MODE` / lever vocabulary and defaults; the most accurate defaults record in the tree. | CURRENT — re-verified 2026-08-02. The old drift note here was itself the stale claim: `PRISMAQUANT_CB_LADDER_INTERP` is **not** described as unwired — §5 documents it as live in both cost paths, and it is read at `measure_quant_cost.py:1737` (dense) and `expert_empirical_cost.py:1212` (expert). Same pass retired the L2/L3 knobs the 2026-07-30 wall orphaned (§9.1) and re-anchored every `file.py:NNN` that pointed past its file's EOF. Residual: it is a curated policy index, not an exhaustive inventory — §10 gives the mechanical sweep (re-run 2026-08-09: **166** distinct `PRISMAQUANT_*` tokens in live source under §10's own `rg` recipe, vs **126** carrying a table row here — 140 are mentioned anywhere in the file; the gap is refusals, ABI checks, constants and research switches) — and the CB lane's shell vars are prose in §5 rather than a table |
| `design/progressive_render_pipeline.md` | The local render-mechanism contract: baseline/candidate/score/accept loop, declared ordering, per-format mechanism matrix (`render_score.py`, `production_weight_cache._format_supports_render_mechanism`). | CURRENT — omits the `weighted_vq` mechanism, which since 2026-07-30 (re-vet R3) is a **registered** mechanism covering BOTH the CB families and gguf: those families' deliberate render is the imatrix-weighted search, driven by `col_weights` on `render_production_weight` |
| `design/pluggable_refactor.md` | The plugin contract — model-structure JSON, serving-profile JSON, pipeline spec, decision units; "a new architecture = three registries". | CURRENT — predates the `nvfp4_cb.json` and `gguf.json` serving profiles |
| `design/validation_harness.md` | CLI manual for `prismaquant.validation_harness` (validate-and-register / compare / list) and its registry gate (`artifact_registry.py:17,182`). | CURRENT — describes the registry validator, not the numeric ship gate (`validate_quantized_model.py:116-120`) |
| `design/mtp_rung_selection.md` | Throughput-optimal draft-rung selector for MTP/spec-decode: cost and acceptance curves, the 7-step selector, calibration procedure. Matches `prismaquant/mtp_rung_selection.py` section-for-section. | CURRENT — canon for the CB lane; the compressed-tensors pipeline still stamps `MTP_FORMAT=BF16` (`run-pipeline.sh:161`) |
| `design/calibration_diverse_v1.md` | The `diverse-v1.jsonl` calibration recipe (256 rows, 40/20/20/20 prose/code/math/multilingual) and its builder; pipeline default at `run-pipeline.sh:82`. | CURRENT — silent on the sibling cross-domain gate corpus `xdom-gate-v1.jsonl` (`run-pipeline.sh:88`), which has no doc at all |
| `design/unified_render_theory.md` | Theory paper treating every per-Linear render decision as one shrinkage question; derives a closed-form damp law and runs the V0→V1b validation ladder. | HISTORICAL — the ladder cleared and the thread closed 2026-06-22 (`a7500f5`): sweep OFF, fixed damp 1.0 (`export_native_compressed.py:1857,1860`). The header and §6.4 still say the sweep stays default; §8 and line 575 record the closure |
| `design/v20_memory_and_scheduling.md` | The v20 streaming cache/scheduling design note: mark-done eviction, pressure shrink, value-aware retention, predeclared shard schedule. | HISTORICAL — self-labelled "pre-implementation"; steps 1–5 shipped (`layer_streaming.py:1370,1206`, `incremental_probe.py:498-501`), the cheap-Gantt instrumentation and per-channel Fisher summaries never did. Its numbers are v19-era MiniMax |

---

## lanes/ — per-container serving lanes

### lanes/nvfp4-cb/ — the codebook (CB) format and the gridbook plugin

| Path | What it is | Status |
|---|---|---|
| `lanes/nvfp4-cb/STANDARDS.md` | The producer lane's normative format/kernel/cost contract: NVFP4 K1..K25 and FP8 step-four ladders, scale coding, kernel standards, the split-aware floored RD cost law, and the served-A/B rule. Runtime behavior is owned by the external Gridbook repository and its packaged contract. | CURRENT — production exports default to two-tier/v2; v1 is explicit legacy read compatibility; newly widened NVFP4 bands remain external-contract and measurement gated |
| `lanes/nvfp4-cb/LAYOUT.md` | The on-disk byte/container contract a third-party plugin implements against: production fp4 `type_size = 4k+9` (explicit legacy v1 `4k+16`), fp8 `4k` plus FP32 row scales, superblock 256, and exact FP16 product-codebook sidecars. The public spec. | CURRENT |
| `lanes/nvfp4-cb/two-tier-scale-spec.md` | The v2 scale-coding spec: per-256 E8M0 super + per-16 4-bit sub composing exactly onto E4M3; reduces fp4 scale cost 0.500 → 0.28125 bpw. | CURRENT and implemented |
| `lanes/nvfp4-cb/encode_tiers.md` | Measured speed/accuracy curve of the CB encoder tiers (`fast`/`balanced`/`max`, default `balanced`, `nvfp4_cb_formats.py:129-130`) plus the 27B-scale launch/volume fix. | CURRENT |
| `lanes/nvfp4-cb/format-speed-policy.md` | Normative constrained native-parity policy: exact whole-artifact bytes, phase-specific TTFT/ITL/TPS SLOs, execution-contract provenance, workload coverage, evidence boundaries, and the paused W4A16 support backlog. | CURRENT |
| `lanes/nvfp4-cb/PLAN.md` | The original master plan plus the lane's full running implementation log through the Hy3 ship. | HISTORICAL — its header now records the shipped lane; superseded as guidance by `STANDARDS.md`, retained as the provenance trail |
| `lanes/nvfp4-cb/moe_cb_design.md` | MoE CB design: route-flip cost argument, empirical-expert-cost integration, and the then-open serving gap. | HISTORICAL — §3 and §4 landed in the separately released Gridbook runtime and `run-pipeline.sh`; §2's route-flip argument survives |
| `lanes/nvfp4-cb/serving-kernel.md` | The original kernel/plugin plan: INV-1 (no resident dense `[N,K]`), INV-2 (tensor cores, not bf16 MMA), and the prototype ladder Triton → transient expand → fused. | HISTORICAL — the plan executed; the shipped runtime surface is the external Gridbook repository. Kept because INV-1/INV-2 are still live vocabulary in the kernel sources |
| `lanes/nvfp4-cb/format-pipeline.md` | The original format/allocator/exporter integration design. | ARCHIVED — superseded by `LAYOUT.md` (bytes) and the container gates (`run-pipeline.sh:119-131`) |
| `lanes/nvfp4-cb/phase0-measurement.md` | The four gating emulation experiments and the Phase-0 gold-metric definition. | ARCHIVED — all four ran (exp1/1b/1c, rd_ceiling); the lane has since moved to served metrics |
| `lanes/nvfp4-cb/persistent-n-prefill.md` | The large-M "decode B once per N-tile" endgame kernel — design, opportunity sizing, and the §7 build plan. | HISTORICAL — built and **measured negative**: parity-green but 2–5.7× slower than expand+fork at 27B shapes; quarantined behind `PRISMAQUANT_CB_PREFILL_DENSE=persistent` (`gridbook/linear.py:451`). Its redundancy analysis still explains why mid-M is the fused kernel's only niche |
| `lanes/nvfp4-cb/serving-tax-elimination.md` | Decomposition of the CB serving tax vs conventional AURA at matched size, plus a proposed lever sequence. | HISTORICAL — both headline levers were measured and rejected (decode contract v2 null, `d924d76`; persistent-N negative). §3's iso-size/iso-quality framing is the durable part |
| `lanes/nvfp4-cb/cutlass-kernel-notes.md` | CUTLASS grounding map plus the 2026-07-19 fused-prefill verdict (0.22× large-M structural, chunked-overlap negative, the `wait_stream` IMA lesson). | HISTORICAL — its "dispatch intentionally not wired" banner is stale; mid-M fused is default since `ac3e584` (`gridbook/linear.py:439`) |
| `lanes/nvfp4-cb/dsv4_readiness.md` | Read-only audit of what blocks a DeepSeek-V4-Flash run in the CB lane; 9 dependency-ordered gaps. | HISTORICAL — gaps 5–7 (MoE serve path) closed by the Hy3 work. Genuinely open: fp8-source ingestion into the CB encoder, and TP>1 (no TP handling anywhere in `gridbook/*.py`) |
| `lanes/nvfp4-cb/rd_ceiling_study.md` + `.json` | Rate-distortion ceiling study isolating the FP4-grid coding tax (+2–8%) from the structural scale-packaging tax (~0.19 bpw) — the finding that motivated two-tier v2. | HISTORICAL |
| `lanes/nvfp4-cb/exp1_0p6b_results.md` | Phase-0 exp-1 on Qwen3-0.6B: fixed vs learned vs product vs full codebooks, emulated whole-model KL. CB lost IQ2_S by +66% at matched bytes — the near-kill the RD study reinterpreted. | HISTORICAL |
| `lanes/nvfp4-cb/exp1b_0p6b_corrected.md` | Corrected CB-vs-IQ rerun reframed onto the native-FP4 bpw premium (≈ +0.15 bpw at the IQ2_S crossing under v1 coding); per-tensor sidecars ruled out, shared-per-role mandatory. | HISTORICAL |
| `lanes/nvfp4-cb/exp1c_v2_premium.md` | Re-measurement of that premium after two-tier v2 — eliminated at K18, not at K14/K16. The GO that put v2 into the Hy3 driver. | HISTORICAL |
| `lanes/nvfp4-cb/two_tier_scale_check.json` | CPU-only empirical risk check behind the two-tier spec §3 (per-tensor subnormal fraction, within-superblock scale spread). | HISTORICAL (data) |
| `lanes/nvfp4-cb/serve_prototype_0p6b.md` | The first *served* CB reading (Triton prototype). Emulation predicted served KL within ~1.1×, which is why emulated KL was allowed to gate Phase 0 at all. | HISTORICAL |
| `lanes/nvfp4-cb/serve_prototype_4b.md` | Transient-expansion prefill onto stock fp8 W8A8 at 4B — cut prefill 6.3×/8.5× without a CUTLASS mainloop. The decision that made the lane shippable. | HISTORICAL |
| `lanes/nvfp4-cb/prod_27b_results.md` | First production served A/B: Qwen3.6-27B CB @5.5 bpp vs shipped PrismaAURA-5.5bit — conf-KL −45/−53%, ALL-KL −56/−58% at 19.93 vs 23 GB; all 386 body Linears chose CB rungs. Also where the session-arithmetic KL drift was root-caused. | HISTORICAL — its CUDA-graph addendum ("keep `--enforce-eager`") and every absolute decode number are superseded by the M-branch-hoist dispatch |
| `lanes/nvfp4-cb/prod_35b_results.md` | First CB-on-MoE verdict: Ornith-1.0-35B @4.75 bpp, conf-KL 0.01706 vs AURA 0.03625 (−53%); grouped (token,expert) decode GEMV gave 9.4× decode. | HISTORICAL |
| `lanes/nvfp4-cb/prod_hy3_results.md` | Running log of the Hy3 295B-A21B @2.9 bpp build and single-Spark serve: 105.73 GB resident, prefill 109–115 vs the shipped GGUF-IQ's 42 tok/s, TEB 88. Carries an explicit no-quality-claims rule (a 295B cannot be KL-validated on this box). | HISTORICAL |

### lanes/gguf.md

| Path | What it is | Status |
|---|---|---|
| `lanes/gguf.md` | The GGUF lane: second export container, per-Linear k-quant/IQ menu, llama.cpp and vllm-gguf-plugin runtimes, five design invariants, measured 0.6B/4B tables. The `EXPORT_CONTAINER=gguf` gate (`run-pipeline.sh:97-110`) and its rendering-confound reason are accurately stated. | CURRENT — its "known limitations / open work" section is overtaken: the GPTQ-into-k-quant rounder exists (`gguf_gptq.py`, wired at `export_gguf.py:322`), MoE expert stacking exists (`export_gguf_direct.py`), imatrix weighting of packed experts exists (`moe_imatrix.py`). Invariant 5 (llama.cpp owns the container) now holds only for `export_gguf.py` |

---

## results/ — dated historical records

None of these is guidance. Numbers are point-in-time; check the superseded-by note
before citing anything.

| Path | What it is | Status / superseded by |
|---|---|---|
| `results/gridbook_0p8p5_w8a16_gate_2026-08-12.md` | Exact Gridbook 0.8.5 installed-wheel GB10/sm121 gate: immutable commit, wheel, image, command, 91-pass/0-skip JUnit, and raw evidence paths for block-FP8 W8A16. | CURRENT route-existence, residency, and operator evidence; not full-artifact serving, performance, KL, or PPL evidence |
| `results/qwen3_30b_a3b_profile_census_2026-08-03.md` | Contract/vLLM-qualified Qwen3-30B-A3B selection, complete safetensors census, unified `qwen3` producer bridge, and verified probe capture points. | CURRENT profile-onboarding evidence; not a quantization or serving result |
| `results/aura_4b_dense_frontier_2026-06-05.md` | Dense 4.5→8.0 bpp AURA RD sweep on Qwen3-4B, fp32 vs bf16; frontier clean and log-linear, kneedle unstable (fp32 5.00 in 454/1000 bootstraps). | HISTORICAL — its conclusion won: selection moved off kneedle to a byte-budget + saturation-B* rule |
| `results/fp8_gptq_mx_scale_27b_results_2026-05-21.md` | 27B tail A/B, FP8_E4M3 GPTQ vs MXFP8 with E8M0 joint-scale search: FP8 won all 149 tail Linears and all 90 fused units; the allocator then selected zero MXFP8. Explicit "do not ship". | HISTORICAL — the primary 27B evidence for de-menuing MXFP8; the joint-scale search it used has since been removed (self-bannered) |
| `results/production_render_staged_27b_results_2026-05-21.md` | Staged production-render allocator on 27B: improved the last-token-KL screen (0.0232 vs 0.0280) and regressed direct WikiText PPL (10.83 vs 8.33). "Do not ship". | HISTORICAL — the canonical citation for "a narrow KL screen can invert against direct PPL", and the measurement carried by the `COST_MODE=production-render-staged` `exit 2` gate (walled 2026-07-30, `archive/production_render_staged_2026-07-30/`, re-vet R17) |
| `results/milestone_qwen36_27b_fp8menu_2026-05-15.md` | Mid-flight snapshot of the 27B FP8-menu 5.05 run (allocator done, cache 25/487). Self-limiting: "not a completed shipping artifact". | HISTORICAL — superseded by the 5.31 PrismaSCOUT ship and then by the 2026-06-24 AURA regen |
| `results/qwen36_27b_current_vs_shipped_2026-05-25.md` | Served-vLLM A/B of then-current 27B allocator output vs the shipped 5.5/5.31 artifacts (KL 0.0344 vs 0.0475/0.0551). Carries the bpp-accounting caveat: the public "5.31" body bpp is ~4.76 under current accounting. | HISTORICAL |
| `results/qwen36_35b_mse_promotion_phase1_2026-05-25.md` | Phase-1 local-output-MSE promotion on 35B: removed 86% of stored local MSE, landed KL 0.0898 / PPL 9.81 — beat the strategic baseline, lost to both the shipped 4.75 and the 5.16 kneedle. | HISTORICAL — superseded by AURA-on-MoE; `MSE_PROMOTION` was **walled 2026-07-30** (`archive/mse_promotion_2026-07-30/`, re-vet R18) and now `exit 2`s. Ledger check before the wall: no shipped run carries `layer_config_before_mse_promotion.json` |
| `results/qwen36_35b_propagated_group_sensitivity_2026-05-25.md` | Paired propagated-KL over 75 promotion groups on 35B; `linear_attn.layer_9` ranks local-MSE 72 but propagated 4. The direct ancestor of AURA's KL-adjoint cost. | HISTORICAL — its selection role is superseded by AURA |
| `results/qwen36_35b_propagated_4p75_eval_2026-05-25.md` | Equal-budget test: propagated allocation vs the shipped 35B 4.75 — 0.0769 vs 0.0671 served KL. Honest negative. | HISTORICAL — superseded by AURA-on-MoE (served KL 0.0292 at 4.75 bpp) |
| `results/qwen36_35b_propagated_5p15_eval_2026-05-25.md` | The same signal at 5.15 bpp: KL 0.0488 / PPL 9.371, beating the shipped 4.75 but spending ~0.40 extra bpp. | HISTORICAL — dominated on both axes by AURA-on-MoE |
| `results/qwen36_35b_serving_unit_propagated_4p75_eval_2026-05-25.md` | Regrouping propagated sensitivity by **serving unit**: KL 0.0362 vs 0.0671 at equal bpp. Also the extrapolation A/B where `current_only` won the hook screen and lost full-vocab KL. | HISTORICAL — the serving-unit granularity conclusion survived into the shipped allocator (`run-pipeline.sh:216`) |
| `results/qwen36_35b_serving_unit_pareto_2026-05-25.md` | Full Pareto re-run on the serving-unit report; the hook kneedle picked 4.70 bpp, the best materialized artifact was 5.53 (KL 0.0327). Its operational notes (`TRITON_CACHE_DIR`, `FLASHINFER_DISABLE_VERSION_CHECK`, dataset cache dir) are still live landmines. | HISTORICAL — superseded by AURA-on-MoE at 4.66–4.75 bpp |
| `results/qwen35_0p8b_s_rung_headtohead_2026-07-22.md` | Reproducible product-K versus signed-S matched-rung screen: K won 609/776 weight-MSE comparisons; only six signed units survived the 2.6-bpp solve. | CURRENT evidence for keeping S13–S16 codec-compatible but out of production menus; not served KL/PPL |
| `results/qwen3_4b_low_bpp_results.md` | Four-term Block-CLADO + polish at 4B in the low-bpp regime; the surrogate kneedle over-estimated 2.3×. Earliest record of cross-process KL drift, later root-caused as extension-residency address shift. | HISTORICAL — Block-CLADO lane rejected; see `archive/block_clado/` |
| `results/union_cache_smoke_0p8b.md` | "Smart union cache": NVFP4 for every eligible Linear, FP8/MXFP8 fallbacks only above p50/p75 of the NVFP4 `output_mse` distribution — 258 vs 438 renders on Qwen3.5-0.8B. | HISTORICAL — the `PRODUCTION_CACHE_UNION` lever it proposed was **walled 2026-07-30** (`archive/union_cache_2026-07-30/`, re-vet R18): a render-budget percentile deciding which Linears may be offered an FP8 rung is a constraint on the allocator (principle 1). Now `exit 2` |
| `results/v1_milestone_validation.md` | The V1-era pre-tag release checklist: CPU suite, full pipeline from a fresh work dir, expected artifacts, eager + graph vLLM smokes, provenance to record at tag time. | HISTORICAL — **do not follow its env block**: it puts `MXFP8_E4M3` in `FORMATS`, omits `static_act_order` from the levers, names no `COST_MODE`, and expects an artifact (`format_applicability.json`) that nothing writes. The *shape* of the gate is still right |

---

## audits/ — the audit series

| Path | What it is | Status |
|---|---|---|
| `audits/cbl_export_diagnosis_2026-08-10.md` | Why learned codebooks could not reach a production artifact: the allocator's hard block, the absence of a value-bearing learned-book input to cost/cache, and a routed-MoE blocker where pinned Gridbook resolves one LUT for the fused `w13`/`w2` stacks while CBL learns distinct gate/up/down books. Establishes the safe producer scope — dense FP8-CB learned, NVFP4 lattice by default, fail closed on routed learned. | PARTLY SUPERSEDED — the allocator block it cites (`allocator.py:1980`) was lifted by the scoped-bundle stack in 0.11.0; the routed-MoE LUT-ABI blocker still stands |
| `audits/serve_env_census_setproctitle_2026-08-14.md` | Why the serve environment census refused every correct server: it reads `/proc/<pid>/environ`, and vLLM's EngineCore renames itself via `setproctitle`, which on Linux overwrites the argv+envp block and destroys that file while `os.environ` stays intact. Structural on every lane at every commit, and never once green. Fixed with `SPT_NOENV=1` in both runtime Docker vectors. | CURRENT — fix VERIFIED end-to-end on a live server (§7, `consistent: true` with EngineCore renamed). Read §5 before assuming it unblocks the DSv4-Flash 0731 bytes; it does not |
| `audits/math_reunderwrite_2026-08-21.md` | Full mathematical re-underwrite of the cost chain, encoders, selection and accounting: first-principles derivations cross-checked against implementations with independent numeric verification; verdicts per artifact plus two fixed defects (`solve_allocation` contract bound documented — overshoot ≤ `bit_precision·(n+3)/2`, proven; MTP Lambert-W overflow silently disabling scipy on ~41% of plausible range — log-space rescale + Newton continuation + loud solver status) and new proofs (two-tier scale-code 2-to-1 structure with empty exception set; rearrangement envelope for the ½·H·MSE collapse; charged-bin backtrack sufficiency). Paper consistency edits applied to `paper/main.tex` under the claims-test constraints. Gridbook-side companion on that repository's branch. | CURRENT
| `audits/serving_wheel_cache_poisoning_2026-08-14.md` | A wheel that failed the pinned-digest check was still moved into the digest-named cache, which the fast path then trusts forever — one `pip download` bricked the DSpark serving lane. Root cause: Bash disables `errexit` inside a command substitution in a `\|\|` list, so the pre-`mv` verify never aborted. Also records that the PyPI and served-image 0.8.6 wheels are content-identical but different archives. | CURRENT — fixed in v0.12.3; §5 has the operational prerequisite for any serve run |
| `audits/numerical_audit_2026-07-02.md` | Seven-domain line-by-line numerical audit: 2 criticals (NVFP4 `input_global_scale` 448× convention; `block_output_match` scale blow-up on negative max), majors M1–M19, plus a same-day fix-status addendum. It annotates its own closure and marks its own superseded section — the template for maintaining a results doc. | HISTORICAL — all findings closed; keep unedited as the reference for the residual open knobs |
| `audits/audit_findings_2026-05-22.md` | Five-finding quantization-correctness audit (MXFP4 E8M0 encoding, MXFP8 activation semantics, missing registry↔served-metadata reconciliation gate, missing cost-surrogate anchor fixture, duplicated codec math), all fixed in-patch with pinning tests that still exist. | HISTORICAL |
| `audits/audit_questions_2026-05-22.md` | The five product/serving decisions deliberately *not* filed as findings. Q1 — research-only registry entries need an explicit exportability predicate, not a test-side allowlist — is still open (`format_registry.py:666,693,742,751`). | HISTORICAL |
| `audits/codebase_audit_2026-05-10.md` | Post-cross-layer-archive surface reduction: what moved to `archive/tiny_bakeoff_2026-05-10/`, what consolidated, what stayed live. Its method — import graph + AST duplicate scan + reference search, tests before consolidating helpers — is the standing recipe. | HISTORICAL |
| `audits/kl_validation_inplace_replay_2026-05-12.md` | Infra smoke for in-place assignment materialization in `validate_assignments_kl` — destructive copy into the live CUDA model instead of preloading the whole cache; 27B n=16 replay, swap flat at ~240 MiB. | HISTORICAL — its KL numbers are last-token screen values; never quote them as quality results |

---

## research/ — literature surveys

External-evidence surveys: what the published literature and shipped open-source
practice actually establish, kept separate from this project's own measurements
so a citation is never mistaken for a result of ours.

| Path | What it is | Status |
|---|---|---|
| `research/calibration-data/SURVEY.md` | Calibration data for PTQ, with emphasis on sparse MoE: where the ~0.25M-token dense convention comes from (GPTQ 128×2048, AutoAWQ 128×512, SmoothQuant 512×512) and why it is a convention rather than a measured optimum; what the controlled size and composition studies do and do not establish. Every claim carries a source link and an evidence grade. | CURRENT — evidence checked through 2026-08-03; the MoE-specific gap it identifies (none of the dense size results transfer to rare experts) is still open |
| `research/rotation-codebooks/SURVEY.md` | Rotations, learned codebooks and second-order compensation: why a change of basis and a codebook solve different problems, and how much of incoherence a flexible per-tensor codebook can absorb. Mechanistic inference from QuIP / QuIP# / QTIP / AQLM, labelled as inference. | CURRENT — evidence checked through 2026-08-03; records an explicit NEGATIVE finding: no measured evidence was found that holds a per-tensor learned, FP-grid-constrained product codebook fixed in capacity and ablates rotation on/off at 2–3 bits, which is exactly the comparison the CB formats would need |

Each survey ships its `bibliography.json` beside it; the survey cites by key and
the bibliography is what those keys resolve to.

---

## archive/ — superseded narratives

| Path | What it is | Status |
|---|---|---|
| `archive/prismaquant_design.md` | The 58 KB master design doc (last touched 2026-06-12). Superseded wholesale by `ARCHITECTURE.md`. | ARCHIVED — never mentions AURA, the CB lane, or the GGUF lane; roughly half its `file:line` citations have drifted; §3.5 cites four modules that do not exist; §6.3/§12.2 contradict §10/§13.4 inside the same file. §11 (rejected alternatives) is the part worth reading |
| `archive/prismascout_overview.md` | One-page statement of PrismaSCOUT as "the current selection layer in PrismaQuant". | ARCHIVED — the spine was retired 2026-06-08 in favour of AURA; two of its five pipeline steps name symbols absent from the tree. Its contract sentence — surrogate scores may rank and prune, but nothing ships without a real KL gate — is the durable line |
| `archive/prismascout_handover_2026-05-03.md` | Codex-to-Codex handover on PrismaSCOUT/SMRF; origin of the 5.3117 bpp / KL 0.0151 point that became the shipped 27B flagship, against a surrogate-only knee of 5.857/0.0557. | ARCHIVED — self-bannered; its "new feature" section documents removed code |
| `archive/propagated_cost.md` | Tombstone for the L3 propagated-cost polish path. Correct about intent: the `kl_measurement.py` functions still exist, the allocation entry point does not. | ARCHIVED |
| `archive/vectorization_refactor.md` | 2026-04 plan to de-Python-ify the pipeline for MoE-heavy checkpoints. | ARCHIVED — a dated intent list, partly unexecuted: per-layer activation bundles were never built and inner-loop `empty_cache()` calls remain (`incremental_measure_quant_cost.py:531,672,694`) |
| `archive/block_clado/` (6 files) | The Block-CLADO lane: the pipeline design plus five results docs (smoke, iterate, output-Fisher, full-OF, final). Three contain runnable-looking "Recommended pipeline" blocks for modules absent from the tree. | ARCHIVED — lane rejected; the rejection record is `archive/cross_layer_2026-05-09/README.md:20-25`. Durable content: pair terms get the frontier *shape* right while ranking at ρ=0.23, and modelled pairwise interaction Ω_ij is uncorrelated with the four-term truth (ρ=−0.10) |
| `archive/codex-process/` (3 files) | The closed 2026-06-09 exhaustive review: `CODEX_QUEUE.md` (dispatch order), `CODEX_QUEUE_FINDINGS.md` (all 82 findings with evidence), `CODEX_PROGRESS.md` (per-finding disposition). | ARCHIVED — closed 2026-06-22; moved off the repo root because as root files they read as active instructions. Line references inside are pinned to June code. One entry is now inverted: MAJOR-M10's "no `COST_MODE=aura` in run-pipeline" is false on HEAD |

---

## Local-only, unpublished

Absent from a fresh clone. Do not cite outward; treat every claim in them as a lead.

| Path | What it holds | Why unpublished |
|---|---|---|
| `docs/handovers/` (17 files + `README.md`) | Dated session-handover records, 2026-04-29 → 2026-07-19. Narrative arc only; their "open items" sections are frequently superseded. | Gitignored (`.gitignore:22-23`, README excepted) — session history is local working state, not documentation |
| `.claude/codex-*`, `.claude/aura-review-brief-*` (30 files) | The Codex/Gemini multi-model deliberation archive: briefs, adversarial reviews, raw CLI transcripts, signoffs. Includes the AURA red-team blockers and the MoE cost-gate failure that produced the shipped empirical-expert hybrid. | Process history, not documentation; several positions were later inverted by production evidence |
| `scratch/deliberation/`, `scratch/review-2026-06-09/`, `scratch/doc-consolidation-2026-07-30/`, `scratch/hf_readmes/`, `scratch/gridbook-launch-post.md` | Working notes, the per-file doc censuses behind this index, HF card drafts, draft launch copy, one-off analysis scripts. | Gitignored (`.gitignore:15`) — working state; the launch post is marketing copy, not documentation |
| `references/` | Third-party prior art: ~12 PDFs (CLADO, HAWQ-V3, AMQ, ImPQ, ParoQuant, CoopQ), the vendored HTQ clone, and the low-bit-kernel conference thread. | Gitignored (`.gitignore:14`) — vendored third-party material, not ours to redistribute |

---

## Satellite docs

| Path | What it is |
|---|---|
| `paper/main.tex` | The canonical AURA narrative — *"AURA: Production-Faithful KL–Fisher Allocation"*. The derivations live here and nowhere else: `sec:additivity` (why cross-layer modelling was retired), `sec:aura` (the cost), `sec:rd` (frontier geometry), `sec:limits` (honest accounting). Built PDF alongside. **CURRENT** |
| `paper/archive/` | The retired PrismaSCOUT paper (source + PDF), an earlier legacy build, the explainer animation, and 13 retired figure sources. **ARCHIVED**, already walled |
| [Gridbook repository](https://github.com/RobTand/gridbook) | Canonical runtime, CUDA sources, tests, packaging, plugin documentation, and releases. PrismaQuant consumes one immutable pin plus the packaged runtime contract; no source copy lives here. |
| `prismaquant/README.md` | 17-line package note listing six `python -m` entrypoints. **STALE** — omits the AURA, GGUF and CB stage modules, and never says that `prismaquant/run-pipeline.sh` is the orchestrator, so a `pip install` reader has no route to the entry point that matters |
| `README.md` (repo root) | Public README: AURA framing, served results, quickstarts, format/architecture/artifact tables, and the external pinned Gridbook boundary. **CURRENT** — container-specific kernel ownership is stated explicitly |
| `AGENTS.md` | Normative agent rules: 9 core principles plus a before-editing / before-finishing checklist. **CURRENT** — rule 4 should read "the target serving profile's gate" now that two of three containers are not plain vLLM-native, and rule 6's MXFP8 example is a de-menued format |
| `CLAUDE.md` | The working agreement and project brain: how Robert works, the methodological spine, the graveyard, the operational landmines. **CURRENT** as intent; its snapshot sections (§6 file map, §8 current state) drift and are subordinate to code |
| `archive/` (repo root, 17 walls) | The graveyard: one dated directory per rejected method, each with a `README.md` banner stating why it lost. **Load-bearing — never move.** Four walls are cited by path in `run-pipeline.sh` `exit 2` messages (`grouped_kl_2026-05-28`, `hdq_2026-05-14`, `fisher_2026-05-15`, `multi_shot_2026-05-19`). Convention is dated directory name + top-level banner, and as of 2026-07-30 every wall satisfies both |
