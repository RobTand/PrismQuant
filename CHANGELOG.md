# Changelog

## 0.12.1 — 2026-08-13

### Fixed

- Serve fingerprints now attest both supported immutable Gridbook installation
  forms: the existing exact VCS commit and an exact release wheel. Wheel-backed
  images must provide `PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256`; PrismaQuant requires
  the matching PEP 610 archive SHA-256 and versioned wheel filename, then still
  verifies `direct_url.json`, `METADATA`, `RECORD`, every installed Python/CUDA
  source byte, and the actual import origin. Unpinned wheels, digest mismatches,
  editable/bare-directory installs, and same-version import shadows remain
  fail-closed. The v0.12.0 VCS evidence shape remains replay-compatible.
- Endpoint, performance, and shipcard replay accept the optional digest-bound
  wheel identity while continuing to require the tracked Gridbook commit and
  version. This fixes production images that correctly install a reviewed wheel
  instead of synthesizing VCS metadata inside the image.

## 0.12.0 — 2026-08-12

The DSv4Flash export-readiness release. It lands the activation-safe,
identity-bound AURA campaign and replay path merged in PR #81, then closes the
source block-FP8 serving mismatch discovered during final allocation review.
The approved 112.690 GB assignment is unchanged; this release changes the
legality and provenance gates used to reproduce and export it.

- Bind campaign completion, streamed checkpoints, allocator replay, exporter
  handoff, and child-process imports to one clean immutable PrismaQuant source
  snapshot. The CPU-only W8A16 readmission reconstructs the historical AURA
  rows and permits only the audited source-terminal correction; it refuses any
  assignment or byte-accounting drift before export.
- Preserve the production cache/prefetch contract and bounded one-Spark
  residency throughout the DSv4 path. No parallel activation or rendered-weight
  cache was introduced.

- Resolve the external Gridbook 0.8.5 runtime pin to immutable commit
  `e992e5980c96333a48149f96392d6cff56ae9e3f` and promote the raw block-128
  E4M3/UE8M0 source lane to its dedicated W8A16 route. The installed-wheel
  GB10/sm121 gate passed 91 tests with no skips; decode uses native source
  GEMV and prefill uses transient BF16 expansion plus Gridbook's owned grouped
  CUTLASS bridge. Newly exported artifacts no longer carry the obsolete
  route-pending acknowledgement. Full-artifact serving, performance, and
  quality remain independent shipcard gates. The exact command, image, wheel
  identity, logs, and JUnit record are in
  `docs/results/gridbook_0p8p5_w8a16_gate_2026-08-12.md`.
- Make every shipping boundary require that exact Gridbook version and commit,
  not merely a syntactically resolved release pin. Source block-FP8 remains
  W8A16; the distinct direct group-32 MXFP8 re-encode lane remains W8A8 and
  unbacked by default.

Activation-quantization-aware AURA and MTP/DSpark re-optimization are explicitly
post-ship work and are not part of 0.12.0.

## 0.11.0 — 2026-08-11

The learned-codebook release: value-bearing learned books reach the production
allocator and exporter, and the cost surrogate is refactored so that the
*evaluation of a model no longer knows what a codebook is*. A platform-agnostic
anchored-cost core owns anchor planning, shape fitting, hull pruning and
exposure reporting; a mapping plugin supplies the format vocabulary.

MINOR, not patch: the allocator now rejects candidates that price above the
source bit rate and replaces the structural CBL rung ceiling with a measured
per-rung policy, so a CB menu can select a different assignment than 0.10.0
would have. Published artifacts on disk are unaffected; the lattice default
render is byte-identical.

**Nothing here is a served result.** There is no vLLM KL-vs-BF16 or WikiText
PPL for learned codebooks, for LDLQ, or for anchored AURA in this release. The
routed learned-book runtime opt-in ships **default-off**. The DSv4 anchored-AURA
campaign has not been run — its driver's first production run will also be the
first end-to-end exercise of the seams below, every one of which is fail-closed.

### Learned codebooks reach production

- Lift the allocator's hard refusal of `--cb-codebook-source=learned`, replaced
  by scoped learned bundles: an immutable, value-bearing `.pqcb` read by cost,
  cache, KL, allocator and export from one identity, rather than a digest
  manifest plus export-time retraining.
- A bundle reports its basis **per rung** via `codebook_source_by_format()`
  (refusing a non-uniform map), which is what lets a single FP8_CB menu span
  learned K28–K46 alongside lattice K47/K48 instead of forcing one basis on the
  whole family. `load_bundle`'s policy stamp remains the primary guard.
- Wire learned bundles through the production exporter.
- Routed-MoE learned books, **producer side only and refusing by default**:
  version-gate routed learned refs on the Gridbook per-role LUT ABI
  (`>= 0.8.3`) so that old, prerelease, local or malformed pins refuse before
  export. Expert bundle cells are built only from explicitly selected,
  identity-verified K28–K33 burn shards, copying their FP16 books exactly with
  no export-time training, LDLQ, directory search or lattice fallback.
  Independent logical role refs are emitted for fused uniform and
  per-expert-format stacks while their physical `gate_up_proj`/`down_proj`
  payloads are retained.
- **The Gridbook pin stays on the released 0.8.2** (`9f915dd`). An earlier
  revision of this work advanced it to an "0.8.3 preparation commit"
  `032e8158…`; that commit exists nowhere — not on the Gridbook remote, whose
  newest tag is `v0.8.2`, and not in any checkout on the build box, where
  `gridbook.__version__` still reads `"0.8.2"`. CI caught it at
  `pip install gridbook @ git+…@032e8158…` and the pin was reverted before this
  release. Consequence: the routed learned path refuses under the shipped pin,
  which is the correct state while its ABI is unreleased, and the 0.8.2 fused
  mid-M rung table remains the attested backed set.

### Anchored AURA — a platform-agnostic cost core with a format plugin

- `anchored_cost` owns the generic mechanism and imports no format module. A
  plugin declares the candidate ladder, the shape-transfer equivalence
  partition, the renderer hook, the anchor-rung policy and provenance;
  `cb_anchored_cost` is the codebook instance.
- The equivalence partition is load-bearing: segment keys are
  `(family, role, equivalence_class)` with the class declared by the plugin, and
  the core refuses to fit or apply a shape across a declared boundary — so
  pricing a family as one segment when it spans two bases is impossible by
  construction rather than by convention.
- AURA `predicted_dloss` is the sole currency; `weight_mse`, `output_mse`,
  `h_trace` and `cw_m2` are refused as cost inputs so a sensitivity cannot be
  applied twice. Anchors are production-arm renders bound to a render receipt —
  a bare scalar cannot masquerade as one.
- The allocator admits anchored AURA on **provenance**, not on a claimed
  property: three independent stamps, with `fisher_application_count` read
  through `operator.index` so a string cannot forge it.
- **`activation-inclusive` is retired.** Calling anchored AURA an
  activation-inclusive supersurrogate was wrong. "Supersurrogate" is a statement
  about the *currency* — one projection replaced the two-factor
  `h_trace x output_mse` score — not an activation error model: `aura_cost`
  runs its adjoint on unquantized boundary activations and `dW` is a weight
  delta. AURA is activation-**weighted** and activation-quantization-**blind**.
  The activation path is constant across K within a CB family, so the blindness
  moves only the `nvfp4_cb`-vs-`fp8_cb` family-choice margin. Carried as a named
  limitation, reported and not gated; a served A/B arbitrates.
- `dsv4_aura_cb_reprice` is a thin acceptance driver (model profile, CB plugin,
  byte budget) behind the frozen `tools/run_aura_cb_reprice.sh`, defining none
  of the pricing math: one production anchor per legal
  `(unit, family, equivalence_class)` — 66,951 for DSv4, against ~198k cells and
  ~3.3 TB for a full-menu campaign.

### Allocation legality and provenance

- Derive exact source and candidate payloads from the footprint authorities,
  reject above-source candidates, and persist complete elimination provenance.
- Byte-verbatim terminals now speak the allocator's source-passthrough contract
  (`SOURCE_PASSTHROUGH_COST_SOURCE`) instead of a second spelling of it.
  Allocation was already numerically correct, but every terminal had been
  misclassified in provenance and on the activation branch. The two spellings
  are pinned together by a test.
- Correct the DSv4 routed-expert profile declarations.

### Identity-bound resume, checkpointing, and exact accounting

- `production_render_cost` gains `--format-plan` and validates the cache format
  set against it, so a cost run can no longer be scored against a menu the cache
  was not built for; `source_class_format_plan` derives the plan from source
  classes rather than a hardcoded family.
- Streamed cost with identity-bound atomic checkpoints, so a long cost stage
  resumes on the exact model/menu/arm identity instead of trusting file
  presence. Same treatment for CB pair shards and per-linear KL adjoints;
  resumed CB artifact state is verified rather than assumed.
- Producer identity hashes the complete producer package and binds git-less
  producer source bytes.
- Exact per-rung footprint rate accounting, plus an encode-identity regression
  test so a performance change cannot silently move bytes.

### Performance

- Fuse the LDLQ atom candidate search into one compiled region behind
  `PRISMAQUANT_CB_ATOM_COMPILE=1` (unset is a byte-identical no-op). Measured at
  production shapes: `reassign_product_2d` 223.6 ms → 70.0 ms (3.19x),
  `reassign_product_3d_batched` 1994.5 ms → 258.5 ms (7.72x), peak GPU memory
  unchanged at 1.21 GB. The eager route keeps `torch.linalg.solve_triangular` —
  substituting it unconditionally regressed the gate-OFF path to 0.46x.

## 0.10.0 — 2026-08-09

The DeepSeek-V4-Flash campaign release: LDLQ becomes a *certified* encoder path,
the vendored DSV4 forward becomes faithful to the reference, and four silent
correctness defects in the export path are closed.

MINOR, not patch: `layer_config.json` gains a required key, exported CB weight
bytes move for any artifact built with LDLQ on, and DSV4-class exports change
which units they declare quantized. Published artifacts already on disk are
unaffected — nothing here rewrites them — but they cannot be re-produced
byte-for-byte without the reproduction switches named below.

**The LDLQ quality claim is not yet a result.** Everything below is a screen:
local render/activation MSE, holdout-gated cost-table statistics, and block
parity against the reference implementation. There is no served vLLM
KL-vs-BF16 or WikiText-PPL number for LDLQ in this release. The served A/B is
pending.

### LDLQ certified out of sample — and the in-sample gate was anti-correlated

- **The do-no-harm gate was measuring the objective LDLQ had just been
  optimised against.** It built each expert's Hessian from the captured
  activation rows and then scored keep/revert on those same rows, so it could
  not fail — and its error was not a constant. On `L17 gate_proj, K12` the gate
  figure and the truth on held-out rows run in **opposite** directions: at full
  64-row support the gate posted 0.0325 against a true 0.6517 (20x
  overstatement); at 1–3 rows it posted its *best* figure of the study, 0.0196,
  where the true gain was 0.9504 — i.e. 5% — a 48.5x overstatement. Pricing
  from it would have **inverted** the allocator's ranking, not merely inflated
  it. Support is thin and layer-dependent: the median tensor has 21 activation
  rows against 2048–4096 columns, and 8.2% of tensors have exactly one.
- **The replacement certifies on rows no arm of the gate saw.** Scored with the
  gate handed only half of each expert's rows to keep the evaluation
  non-circular, degeneration falls **7/96 → 1/96**; on full-support `down_proj`
  the new gate rejected exactly the one regressing expert and nothing else.
  It is not literally never — at 4 rows the certificate splits 2/2 and has
  little power. Production hands the gate all rows, so 1/96 is an upper bound.
  Two honest limits: the *shipped* assignment is still the all-rows fit, which
  sees strictly more data than the arm that earned the certificate; and the
  authoritative check remains a model-level disjoint-corpus A/B, which is the
  pending served work.
- `PRISMAQUANT_CB_LDLQ_GATE` selects `holdout` (default), `in_sample`
  (reproduction of pre-2026-08-08 artifacts only, never for new ones), or `0`.
  The certifiability floor `LDLQ_GATE_MIN_ROWS` rises **2 → 16** — eight fit and
  eight decision rows, an evidence floor and explicitly *not* a claim that
  sixteen rows give a population-level guarantee. Tensors below it keep raw.
- **`PRISMAQUANT_CB_LDLQ_SCOPE`** (`none|nvfp4|all`, matching the allocator's
  `--cb-ldlq-scope`) replaces the legacy boolean as the authoritative
  per-family switch and is stamped into the serialization context; an
  inconsistent legacy/scope pair refuses. Per-tensor identities now record the
  LDLQ that actually applied to each format, which fixes mismatched identities
  on mixed NVFP4/FP8 assignments.
- **Both CB exporters resolve LDLQ per format rather than from one global
  boolean, and the activation loader is handed over per format with it.** In
  the non-streaming exporter each format resolves through `_ldlq_for_format`
  and an activation is loaded only when that format's family is LDLQ-eligible
  under the active scope. The streaming exporter had two further scope leaks:
  a single global boolean decided the warm path, the recorded identity and the
  loader requirement for every format alike. The actual per-format family scope
  now controls all four — encoding, warm path, identity and loader — so an
  FP8 tensor under `scope=nvfp4` is neither encoded as LDLQ nor charged an
  activation-loader requirement it cannot satisfy.
- **Product-atom E16 LDLQ** (`cb_ldlq_atoms.py`): exact FP8 atom-2 / FP4 atom-4
  exhaustive Mahalanobis assignment, with typed Hessian failures so dead
  channels refuse rather than fabricate an identity. The canonical packed route
  is ABI-fixed — serial, feeder-thread, non-16 expert-batch, shared-Hessian and
  non-divisible routes are now **refused**, where some were previously legal.
  Two concurrency defects are closed with them: a missing caller→worker
  `wait_stream` in both multi-stream arms, and a factor-cache cross-stream race
  (the cache is now event-backed, bounded at 8 GiB, and keyed on
  `Tensor._version`).
- **The gate no longer materializes the whole stack.** Each fp32 reconstruction
  at DSV4 shape is 16 GiB and the gate transiently held two of them; scoring is
  now chunked per expert slice, holding **64 MiB instead of 16 GiB**. This is
  bitwise identical, not approximately equal — reconstruction is row-local and
  the gate MSE is a within-expert reduction, so slicing commutes, and the tests
  assert exact float equality across both scale codings.
- **Fused siblings union their activation rows by global row index**, raising
  Hessian support rather than intersecting it: on DSV4 L0 expert 0, gate and up
  contribute 64 rows each with an intersection of 15 and a union of 113. Rows at
  shared indices must be bit-identical or it raises.
- Never-routed experts take the cold-expert prior on the direct per-expert path
  too — the DSV4 export reports 3,984 never-routed expert projections across 60
  stacks, so raising there would make LDLQ unusable on any MoE with cold
  experts. The substitution is logged, not silent.
- New additive artifact sidecar **`cb_ldlq_gate_telemetry.json`** (schema
  `prismaquant.cb_ldlq_gate_telemetry.v1`) with exact-qname coverage
  enforcement and a validated kernel stamp; and a post-allocation refinement
  contract `cb_ldlq_refinement.v1` recorded in `quant_config` provenance and in
  `layer_config`. Raw exports are unchanged.

### A no-LDLQ cost table from the LDLQ burn, without a second burn

- The CB fields encoded *before* the gated reassignment are the identical-env
  raw render — same encode, codebook, scale sweep and coding, same
  `col_weights`. A gated cost run therefore now also emits
  `weight_mse_raw_render`, `predicted_dloss_raw_render` and
  `weight_mse_per_expert_raw_render` under provenance
  `prismaquant.cb_ldlq_raw_render_sidecar.v1`, which is what makes the
  LDLQ-contribution A/B affordable at all: the isolate would otherwise cost a
  second multi-hour burn. Output-side metrics are **not** re-measured for the
  raw arm — the allocator prices `predicted_dloss`/`weight_mse`, and a raw
  output measurement would need the full-stack forward the sidecar exists to
  avoid. Ladder-rejected slices deliberately record no sidecar.
- **`tools/extract_raw_cost_table.py`** derives the no-LDLQ allocator cost table
  from it, re-stamping the CB context as `ldlq=false, scope=none` and recording
  the source stamp under `derived_from_ldlq_gated_cost`. It fail-closes on error
  rows, missing or partial sidecars, and already-raw input, and its output must
  pass `validate_cb_cost_provenance` before it is written.
- When LDLQ is off this is a strict no-op: no sidecar keys are emitted and cost
  pickles stay byte-identical, asserted key-for-key against the legacy row
  schema. Banked pickles are never rewritten.
- Never-routed experts have no calibration activations by construction (51 on
  the production capture), which crashed the weight-only row path under an LDLQ
  context. The fix is an explicit call-site opt-in used *only* by that path; the
  default still raises, so a broken activation loader can never silently produce
  an all-raw table stamped as LDLQ.

### Band-interpolated rungs are priced from their own `output_mse`

- A row stamped `band_interpolated`/`mixed` carries `output_mse_measured:
  false`, so it fell to the weight-only branch and was priced as `weight_mse ×`
  a per-family activation constant while its measured neighbours on the same
  ladder were priced from `output_mse` directly — one family, two bases. The
  true output/weight ratio is **not** family-wide: 187–320 on gate/up_proj
  against 9.4–22 on down_proj, a ~20x spread. The constant over-priced
  interpolated down_proj rungs ~12x and under-priced gate/up_proj ones ~1.6x,
  **breaking rung order inside the family**: the higher rung cost more than the
  one below it on ~85% of down_proj experts, so K13/K17 could never be selected.
- This retracts a design claim of `activation_fair_pricing`: "a per-family
  constant cannot reorder rungs inside a family" holds only while every rung of
  the family takes the same branch, and these did not. On the 32 banked layers
  down_proj violations fall from **84.8%/0%/84.6% to 0.014%/0.014%/0.043%**.
  `cost_source` and `output_mse_measured` keep their meanings and no banked
  pickle is rewritten; a new branch label `interpolated_output_mse` records in
  the shipped artifact which selected prices were predictions.
- Ladder interpolation was separately checked to survive the LDLQ identity, on
  the layer-9 pilot (18 exact rungs, 103 cells): gated `output_mse` is
  log-linear in K per family, LOO median 0.9–2.3% and p99 3–7.6%, and the gate
  branch is a cell property (~28% raw across all rungs) rather than a
  K-crossover — so the per-tensor anchors+holdout law applies unchanged.

### The vendored DeepSeek-V4 forward is now faithful to the reference

- The vendored forward carried a half-split `rotate_half` RoPE instead of the
  release's interleaved complex rotation, and an **amputated compressor/indexer
  branch**. Both are restored, with per-layer YaRN tables and the HCA/CSA path
  per the reference, and probe mode becomes an explicit flag with a loud
  one-time warning — the silent architectural degradation is gone.
- **Recorded for honesty: every prior DSV4 probe output — `h_trace`, the
  KL-adjoint, the activation cache, and every cost derived from them — was
  measured through the defective forward, and is scheduled for re-derivation
  before the next allocation.**
- **The first certification was itself wrong, and is retracted.** The "29/29
  parity, zero divergence" claim never executed the vendored module: its
  HyperConnection check was a source-text search and its block check compared
  the reference against itself. `HyperConnection.__getattr__` in fact raised for
  every non-alias name, so the `fn`/`base`/`scale` Parameters were unreachable
  and every `self.fn` access crashed.
- True block parity is now certified by executing the *vendored*
  `DecoderLayer.forward` against the vendor's own Block on real DSV4-Flash-Base
  weights, with forward-hook counters proving execution rather than source-text
  inspection: `max|diff|` **1.0133e-06** (sliding), **1.5348e-06** (CSA),
  **1.6689e-06** (HCA), every intermediate boundary ≤ 6.2e-06 and the mHC
  boundaries exactly 0.0. Six further defects fell out of it, two of them
  silent: the CSA top-k was used as a `torch.gather` index still carrying the
  reference's `-1` sentinel — which CUDA does not bounds-check, so it **read out
  of bounds on every CSA layer** — and pooled-entry mask columns were padded
  visible, a causality break on every compressed layer. The compressor was also
  skipped whenever no cache was passed, so the Fisher probe silently got
  window-only attention with no error.
- Profile fixes: the indexer lives on the CSA compressor, not on attention, and
  its checkpoint tensors sit flat on it — the corrected mapping resolves **all
  72,317 checkpoint keys** with zero misses. A new plugin accessor
  `ModelProfile.probe_linear_exclude_extra()` makes the probe's Linear-exclusion
  regex profile-owned; DSV4 excludes `self_attn.{compressor,indexer}`, restoring
  the 33,325-selectable-Linear inventory the byte accounting assumes.
  `transformers` signature drift between 5.6.0 and 5.12.1 is bound by keyword.

### Streaming

- Nested rotary instances are materialized on meta skeletons.
  `ModelProfile.init_rotaries` gains an optional `base_model` kwarg (the Gemma4
  override is updated in step) and the DSV4 override walks the skeleton, so no
  `inv_freq` buffer is left on meta.
- Eviction no longer meta-izes non-persistent buffers, which `_fast_install`
  deliberately skips and therefore could not restore. Harmless while all rotary
  state lived at the model root; fatal once the faithful forward puts
  compressor/indexer caches inside the layers.

### Export and allocator correctness

- **`layer_config.json` now records `cb_serialized_payload`.** The allocator
  wrote per-tensor CB identities but never the context they were written under,
  so on DSV4-Flash **24,851 of 24,851 CB tensors mismatched** at export. This is
  the root cause of four earlier "CB per-layer serialization identity mismatch"
  failures, and it **retracts** the earlier attribution of those to a
  packed-expert/LDLQ-scope mismatch. The assignment itself is unchanged. A
  `layer_config.json` produced before this release lacks the key and must be
  re-produced by the allocator.
- **21 `attn.indexer.wq_b` units were being shipped as unquantized floats.** The
  floor block-FP8 scan iterated a live-model-name map in which the indexer had
  0 of 33,368 entries, so those units fell through to the verbatim-copy loop and
  were declared `ignore`. A consumer honouring that allocates bf16, the element
  counts match, and the FP8 bytes are cast **with no scale applied** — every
  block wrong by its own power of two, and nothing raised. The scan now reads
  the checkpoint. Pre-existing and LDLQ-independent.
- A failed export **preserves** its partial `.tmp-<token>` root and prints
  resume/discard commands instead of deleting it; a 21-tensor mis-declaration
  threw away ~6 hours of tensor writes that the retry reproduced bit-for-bit.
  The destination is still never created on failure and never clobbered, and the
  preserved root carries no completeness stamp.
- New `dspark_source_metadata.py` bridges the released three-stage DSpark
  topology (4,705 tensors) into the exporter — MXFP4/FP8-block source unit
  classification, physical-to-construction MTP layer mapping, refusal of partial
  stages, and an atomic hardlink-sibling sidecar publisher that provably
  rewrites zero tensor bytes.
- Gridbook runtime pin advances **0.8.0 → 0.8.1 → 0.8.2** (`9011a19` →
  `c9c1265` → `9f915dd`), each with its matching backed-rung key
  `[28,32,36,40,44,48]`. The 0.8.1 bump had landed without its key, which
  emptied the fail-closed resolver's backed set and silently degraded every
  `k%4==0` rung to the expand+GEMM fallback.
  The advance to **0.8.2** is what makes the released pin describe the runtime
  that actually serves: the DSV4-Flash serving images are built from Gridbook
  0.8.2, so shipping a pin that declares `0.8.1` with `version_is_release: true`
  would have asserted a released runtime the artifact does not run on — the
  exact confusion `tests/test_gridbook_runtime_boundary.py` exists to prevent.
  The rung set carries over verbatim, verified more strongly than by reading the
  constant: `gridbook/codec.py` is **byte-unchanged** across the entire
  v0.8.1..v0.8.2 range, so `FP8_FUSED_KBITS` cannot have moved; 0.8.2's content
  is loader and dispatch work above the codec. **This release therefore depends
  on Gridbook v0.8.2 being tagged and its commit pushed** — CI installs the
  runtime from the pinned git commit and asserts the reported version matches.

### Campaign tooling

- Thirteen DSV4 A-FAST campaign tools under `tools/`, plus the burn runner. The
  CBL audit gate no longer escapes `_measure_projection` as an `AssertionError`
  before the layer-wide fallback can run — that turned a recoverable verdict
  into an aborted campaign at 42/43 layers. Per-expert uplifts are now recorded,
  because the aggregate hides the spread: one expert 8.51% worse and one 0.93%
  worse in a sample whose median *gained* 8.45%.
- A Pareto-point writer `KeyError` on every research-cost run emitting a CB
  format is fixed (it dereferenced `cb_render_identity` while guarding on a
  different field).

### Known state at release

- Suite: **3123 passed, 0 failed, 86 skipped, 3 xfailed, 151 subtests passed.**
- **One piece of disclosed debt.**
  `tests/test_dsv4_campaign_tools.py::test_dual_holdout_fit*` is
  `xfail(strict=True, raises=KeyError)`. These two tests were *introduced in
  this cycle* by 41b5c62 and have failed since — they did not exist at v0.9.0,
  so they are not pre-existing debt inherited from an earlier release. The
  cause is a live defect in the campaign tool rather than in the tests: the
  2026-08-06 demand-driven revision narrowed the priced domain to `RUNGS =
  K28–K38` without moving `ANCHORS = (28, 38, 48)` or `HOLDOUTS = (33, 43)`
  with it, so `_fit_slices` indexes its rung-name map with K48 and raises
  before evaluating anything. A real dual-holdout fit takes the same path.
  It is marked rather than repaired because choosing the replacement anchor and
  holdout rungs changes which rungs are measured and which are predicted — a
  campaign-design decision — and the 43-layer burn in flight was launched
  against these exact constants. The marker is conditional on the
  inconsistency and strict, so the suite fails the moment the domain is made
  consistent and the marker must then be deleted.
- `docs/ARCHITECTURE.md` §0 conformance for this range was checked and closed in
  the same release: the LDLQ certifiability floor (documented as 2, actually
  16), the raw-render sidecar and its extractor, three omitted defaults, and the
  two new profile-plugin accessors. `docs/design/runtime_flags.md` gains rows for
  `PRISMAQUANT_CB_LDLQ_SCOPE` and `PRISMAQUANT_CB_LDLQ_GATE`.

## 0.9.0 (2026-08-05)

- **Monotone Min-Chain encoder mode** (`PRISMAQUANT_CB_MINCHAIN=1`): per-rung min over the free LDLQ fit and the previous rung's embedded solution — error curves monotone non-increasing by construction at zero representational tax. Winning-arm/solution/predecessor digests enter the serialization context; mismatches refused at export. Validated: pilot-2 PASS on pre-declared DSV4 layer (zero violations, PCHIP held-out 2.7-2.9% median / <14% p95, 1.003x overhead). (#76)
- **Per-expert-slice ladder gating** with sliced measured fallback for packed-expert cost measurement; per-slice cost provenance (`cost_source_per_expert`). (#74)
- **Batched LDLQ encode** across identical-shape expert units (bit-identical to serial; ~5x on 256-expert stacks). (#73)
- **Pipelined export** (`PRISMAQUANT_EXPORT_PIPELINE=1`): read/encode/write three-stage overlap, byte-identical artifacts (GPU-verified). (#75)
- Qwen3.6-35B-A3B model profile + campaign machinery; amendment-v2 interpolation semantic (5-anchor monotone PCHIP, accept-all + CV outlier backstop + per-layer audit rung). (#74, #76)

## 0.8.0 — 2026-08-03

This release advances the immutable runtime boundary to Gridbook **0.8.0** at
exact commit `9011a19228ddb96b8a49e11a20ac75c99c83998e` — the released
`v0.8.0` tag commit — and adds the source-passthrough format family, its
measured serving verdicts, and the MXFP8 re-quantization rung.

MINOR, not patch: the allocator menu gains formats, `route_backed` is replaced
by a three-valued `route_status`, and the runtime pin file's schema moves to
`v2`. Persisted assignments and shipped artifacts are unaffected.

### Source-passthrough format family (#53)

- **`SOURCE_PASSTHROUGH_CONTRACTS` makes "keep the checkpoint's own bytes" a
  first-class rung** instead of two hand-written special cases. Two formats,
  both from a header-only census of the released checkpoint: `MXFP4_SOURCE`
  (routed experts, nibble-packed E2M1 + E8M0 group scales, **4.25 bpw**) and
  `FP8_BLOCK_UE8M0_SOURCE` (body, E4M3 + one-byte UE8M0 block exponents at
  128×128, **8.00049 bpw**). The census also corrected two live defects: the
  source-kind scan keyed on "has a scale sibling" and stamped 33,024 MXFP4
  experts as `FP8_SOURCE`-compatible — an 8.002 bpw format declared legal on a
  4.25 bpw unit — and the body is not `FP8_SOURCE` at all.
- **Zero-cost candidates, with bytes pinned to the checkpoint.** Shipping the
  source bytes is the identity transform on the reference, so `predicted_dloss`
  is `0.0` with provenance `cost_source="source_passthrough"`; the allocator
  *synthesizes* the candidate because no cost table will ever carry a column
  for a byte-copy contract. The claim cannot be forged — the provenance string
  is honoured only for a declared passthrough format whose activation path is
  the identity. `assignment_artifact_bytes` charges the unit's real header span
  and refuses an allocation where the span and the closed-form byte count
  disagree, rather than letting the artifact budget drift.
- **Measured route verdicts replace assumed ones, and both inverted.**
  `route_backed: bool` could not distinguish "nobody looked" from "we looked
  and it is dead", so it becomes `route_status` (backed / pending / blocked)
  plus `route_requirement` and `route_evidence`. Measured on GB10/sm121:
  `MXFP4_SOURCE` is **backed but only via vLLM's Marlin MoE backend**
  (requirement: `--moe-backend marlin`), and `FP8_BLOCK_UE8M0_SOURCE` was
  **blocked** — every stock route dead. The second finding is load-bearing: CB
  re-encoding of the body is the only way DSV4-Flash serves on this box by
  default, so the body's CB rungs are architectural, not merely economical.
- **Exporter verbatim stream-copy lane.** Passthrough tensors are stream-copied,
  never materialized, so a 3.4 GB expert layer moves through the 16 MiB chunked
  path without becoming a tensor. E8M0 scale planes are copied **verbatim**.
  `quant_config.json` gains `source_passthrough` (schema v1); absence of the key
  means "legacy all-CB artifact", so it is omitted rather than emitted empty.
  `build_quant_config` round-trips its own output through the consumer's parser
  before the file exists.
- **Fixes a pre-existing silent corruption.** `_StreamWriter.write` built its
  header as a dict, so two emit paths claiming one tensor name kept only the
  last span while both blobs were written — a file whose offsets are wrong from
  that point on, with no error. It now raises.

### MXFP8_UE8M0_G32 menu format (#54)

- **A second MX-FP8 format, because it is a different on-disk contract.**
  `MXFP8_E4M3` defers to compressed-tensors, which rounds the group amax to a
  power of two and can scale a group *up* — losing small values off the E4M3
  subnormal ladder. This rung picks the smallest non-clipping shared exponent
  and serializes `float8_e8m0fnu` rather than `uint8`. The `MXFP8` alias still
  points at `MXFP8_E4M3`: repointing it would reinterpret every persisted
  assignment that uses it.
- **W8A8, with the A side measured.** The exactness claim is `weight_mse == 0.0`,
  **not** `output_mse == 0.0`; declaring `act_bits=8` makes the cost stage apply
  the activation closure before measuring, and a row with no measured
  `output_mse` is masked off the menu rather than priced at the global minimum.
- **A latent codec divergence fixed on the way.** `_batched_quantize` dispatches
  on `weight_element_dtype`, which this rung shares with `MXFP8_E4M3` and
  `FP8_CB`, so the batched render would have priced it with the codebook
  replica's E8M0 snap — a different codec in the batched and unbatched paths.
  `_EXPORT_ALIGNED_BATCH_FORMATS` routes it to the registry closure and a test
  holds the two paths to each other.

### Gridbook runtime pin advanced to 0.8.0, and a ratchet so it cannot drift

- The pin now names Gridbook **0.8.0** at `9011a19228ddb96b8a49e11a20ac75c99c83998e`.
  It reached that value through two intermediate bumps during development
  (`7c0b527`, then `4d7292c`), and that is exactly what exposed the gap below.
- **Pin schema `v1` → `v2`: the new `version_is_release` boolean.** A gridbook
  feature merge does not bump `gridbook.__version__`, so a pin can advance to a
  post-release master commit while still self-reporting the same version —
  three distinct commits self-reported `0.7.0` during this work. The version
  string alone therefore cannot say whether a runtime was ever released; the
  commit is the identity and this flag records the rest.
- **`rungs_by_runtime_version` is ratcheted against it**
  (`tests/test_gridbook_runtime_boundary.py`): no key may name a runtime newer
  than the one actually pinned, and the pinned version may appear as a key only
  when its commit *is* that version's release. An unreleased pin backs nothing —
  the fail-closed direction the spec's own "when the pin advances, ADD the
  version key" rule already asked for, now enforced rather than trusted.
- `serving_profile_specs/nvfp4_cb.json` gains the `0.8.0` key, carrying the same
  fused mid-M rung set forward after verifying `FP8_FUSED_KBITS` is still
  `tuple(range(28, 49, 4))` at the v0.8.0 tag commit. The block-FP8 body lane
  moves from **BLOCKED** to **OPT-IN, NOT BACKED**: Gridbook 0.8.0 serves it
  through its own sm120 block-scaled MXFP8 collective, correctness-audited at
  worst rel-Frobenius 5.9e-5 over the seven real-checkpoint body shapes, but
  behind `GRIDBOOK_MXFP8_DENSE=1` with the served timing bench still pending.
  Available is not backed, so it prices no rung.

### Also

- Stage the dual-variant DP drivers for the mid-rung and body-route questions
  (#55): `run_dsv4_mxfp4_dual_alloc.sh` and `summarise_dual_alloc.py`, which
  produce the contingent allocation by *widening the menu* rather than editing
  the route table — pricing a future is fine, declaring it is not.
- Docs now cite the current pin rather than `746473c`; the `0.7.0` section below
  deliberately still names that commit, because it records what 0.7.0 pinned.
- `ci.yml`'s header claimed the suite was 967 tests in ~51 s (reference box,
  2026-07-28). It is ~2762 tests in 10.5–14 min on the runner; the note now says
  so, and names the real margin against the job's 30-minute timeout.

## 0.7.0 — 2026-08-02

This release advances the immutable runtime boundary to Gridbook 0.7.0 at exact
commit `746473c8459acd24c71e7602d1c982da2f8fa80e`, and carries the
DeepSeek-V4-Flash-0731 92 GB pipeline work, the cross-repository K0.2
stage-attestation interop test, and a verified docs-truth batch.

MINOR, not patch: the cost stage prices packed experts differently (per-Linear
activation rows, declared never-routed experts) and the CB encoder was rewritten
for speed, so a production run's numbers and provenance move even though the
producer ABI, format menu, allocation and export defaults, and
quality-promotion status are unchanged. This release makes no DeepSeek-V4
(DSV4) qualification or support claim; the DSv4 CB export lane remains
unbuilt (`docs/lanes/nvfp4-cb/dsv4_readiness.md`) and that work stays paused.

### The DSv4-Flash-0731 92 GB pipeline (merged from `dsv4/flash-0731-92gb`)

- Ported the 2026-08-01 92 GB study work onto the 0.6.0 line: the Stage-0
  format screen (`scripts/ab_nvfp4_vs_k36_dense.py`) now loads through
  `_LazySkeleton.dequant_weight` — the same profile-aware decoder the CB cost
  stage and the exporter use — instead of reading raw safetensors bytes, which
  compared *storage codes* against values for a checkpoint that ships packed
  I8-MXFP4 experts and F8_E4M3 dense tensors. It hard-fails on an unscaled
  float8 tensor rather than silently mis-measuring, resolves its checkout from
  `__file__` instead of a hard-coded `sys.path.insert`, and indexes through
  `detect_profile().checkpoint_to_live_name` so profile-rewritten names
  resolve. Pinned by `tests/test_format_choice_stage0_source.py`.
- Added the production calibration driver `scripts/run_dsv4_flash_92gb.sh`,
  runnable against the `gridbook:test` image, plus the exclusive-GPU
  old-vs-new validation harness (`tools/cb_encode_exclusive_bench.sh` and the
  `tools/cb_encode_*` / `tools/cost_*` probes behind it). The harness refuses
  to benchmark a contended GPU, stops orphaning driver containers, and
  distinguishes "no output" from "different output" rather than scoring a
  silent failure as a pass.
- **Cost-stage correctness.** Every Linear is now measured on its own
  activation rows and fails loud instead of borrowing a sibling's; declared
  never-routed routed experts get weight-only cost rows (Option A) under the
  new never-routed rule; and the CB cost encode is ~2x faster
  **bit-identically**, pinned by
  `tests/test_nvfp4_cb_encode_perf_identity.py` and the v1-vs-v2 K14-excess
  rank-stability cross-check.
- Corrected the DeepSeek-V4-Flash parameter count: **~285 B total** by
  checkpoint arithmetic (281,263,734,784 probe-measured quantizable
  parameters), not the 671 B DeepSeek-V3-family headline; and the 172 GB
  "below the floor" figure is the Pareto grid, not a fixture. The three
  dsv4-branch cost-stage flags are documented in
  `docs/design/runtime_flags.md`.

### K0.2 stage attestation executed across the repository boundary

- `tests/test_gridbook_attestation_interop.py` is the first place one process
  runs **both** halves of the producer/consumer stage attestation. It builds a
  routed-MoE record through the real emitter — a synthetic two-stage FusedMoE
  checkpoint through `synthesize_packed_expert_activation_samples`,
  `calibrated_input_global_scales_with_sources` and `build_execution_contract`
  — then parses it with Gridbook's own v2 parser and verifies it
  `attested_and_verified`, including the artifact-level K0.2 verdict read off a
  tmp `quant_config.json` + `model.safetensors` exactly as
  `k02_readiness_verdict` reads a real artifact.
- It closes a real trap rather than adding tidiness. The attestation was held
  together by 4 pinned digest hexes, 3 schema literals, and a Gridbook-side
  fixture that hand-mirrors this emitter; neither suite ever executed the
  other's code. Gridbook's parser requires a stage entry to declare *exactly*
  `_STAGE_ENTRY_FIELDS` (extra keys rejected), while every digest is framed
  over those same five fields **by name** — so an "additive, backwards-
  compatible" producer-side field moves no hex on either side, leaves every
  pinned-hex test green, and first surfaces at vLLM model load. The new tests
  demonstrate that mutation end to end and name the exact load-time error.
- Wired into the required `pinned Gridbook contract` CI job's file list, which
  already installs the exact VCS pin. Skip semantics match the three tests
  already there: gated on `PRISMAQUANT_REQUIRE_GRIDBOOK_CONTRACT` with
  `importorskip` behind it.

### Docs-truth batch

- Fixed eight verified defects from a corpus review — docs and one provenance
  data file, no code and no behaviour change. The load-bearing ones: §9.2's
  claim that the constrained Pareto formulation was "normative future work"
  had been false since P5c shipped; `docs/design/runtime_flags.md` carried 13
  ghost env vars left standing by the 2026-07-30 L2/L3 wall (retired into a new
  §9.1 ledger recording each token's last reader, rather than silently
  deleted); and 18 `file.py:NNN` citations in `docs/` pointed past EOF, 14 of
  them citing `kl_measurement.py` lines 2054-5516 against a 1,246-line file.
  Citations inside dated audit records were **annotated in place** per the D21
  ledger convention rather than rewritten.
- `docs/lanes/nvfp4-cb/STANDARDS.md` now states that its FP8-CB fused rung set
  is a hand-maintained mirror whose machine-readable form is this producer's
  own `nvfp4_cb.json`, and that PrismaQuant CI *cannot* catch it drifting,
  because Gridbook's packaged `runtime_contract.json` does not carry the fused
  rung set at all.
- The example serve-dispatch table's `provenance.source` strings were
  normalized to byte-match the Gridbook headings they cite (U+00D7 and U+2014
  had been ASCII stand-ins), so a `grep` against the source now resolves them.
  No claim changed — only the citations became resolvable.

### Gridbook runtime pin advanced to 0.7.0

- `prismaquant/gridbook_runtime/gridbook_runtime_pin.json` now names Gridbook
  **0.7.0** at exact commit `746473c8459acd24c71e7602d1c982da2f8fa80e` — the
  peeled `refs/tags/v0.7.0^{}` commit, not the annotated tag object
  (`c23393750fc53d0463892571a8854457a091ea0c`).
- **The producer/runtime lane gate is now directional.** Gridbook 0.7.0's D0.1
  registers `deepseek_v4` in the packaged `runtime_contract.json`, so for the
  first time the runtime *leads* this producer on an architecture: it declares
  it can serve DSV4, while the DSV4 CB export lane remains unlanded here
  (`docs/lanes/nvfp4-cb/dsv4_readiness.md` gaps 1-3 — the exporter still loads
  the whole model in `_load_skeleton`, has no fp8-block dequant-on-read, and
  does no per-expert to packed-expert stacking). The required contract job had
  asserted strict set EQUALITY between declared CB lanes and the runtime's
  `producer_profiles.supported_ids`, which made the intended landing order —
  serving contract first, exporter second — unrepresentable. It now asserts
  CONTAINMENT (`declared ⊆ supported`). The dangerous direction is unchanged
  and still fails closed: declaring a lane the runtime cannot serve does not
  crash, it ships an artifact that serves uninitialised memory, so a new
  negative-control test fabricates exactly that case and requires the check to
  fail and name the architecture. PrismaQuant makes no DSV4 export or
  qualification claim on the strength of the runtime's registration.
- The `nvfp4_cb` serving profile's FP8-CB fused mid-M lane now declares
  `"0.7.0": [28, 32, 36, 40, 44, 48]` — the SAME set as 0.5.0 and 0.6.0, and
  added as a NEW version key rather than by editing an existing one. An
  artifact produced under an older pin therefore stays resolvable at the route
  it actually shipped on, and a pin with no key here still backs nothing
  (fail-closed).
- The set did not move because it **cannot**: 0.6.0's K1.2 resolution proved
  `k % 4 == 0` is a format + TMA law, and `FP8_FUSED_KBITS` is still
  `range(28, 49, 4)` at 0.7.0. The five off-law rungs of the published 27B
  K36..K47 ladder stay permanently expand+GEMM-served and the allocator keeps
  pricing them on the fallback row.
- **No FP4-CB backed set was added.** Gridbook 0.7.0's contract-preserving
  FP4-CB v2 fused mid-M kernel is *still* opt-in behind
  `PRISMAQUANT_CB_FP4_FUSED_MIDM=1`, pending its served NATIVE-PARITY gate;
  with the flag unset the dispatch is byte-for-byte the BF16 bridge. Available
  is not backed, so the honest backed set for the DEFAULT contract stays empty.

## 0.6.0 — 2026-08-02

This release advances the immutable runtime boundary to Gridbook 0.6.0 at exact
commit `ca0f0f562d3f398e094bfa5356a9ce3fa47472f1`, and lands the producer-side
items **P5a**–**P5d** of the cross-repo performance ultraplan,
[gridbook
`docs/audits/ultraplan_perf_2026-08-01.md`](https://github.com/RobTand/gridbook/blob/master/docs/audits/ultraplan_perf_2026-08-01.md)
§6 ("Producer-side allocation: NVFP4 vs FP8-CB at matched bytes"), together with
the producer half of gridbook **K0.2**.

P5a and P5b change how candidates are **priced and described**;
`solve_allocation`'s DP semantics are untouched. P5c adds a **second hard
constraint axis** — served latency and device memory — at assignment level,
still without changing the DP and still without any λ: latency never enters
the objective. P5d adds the D0.3 exact-rate experiment harness.

The producer ABI, format menu, allocation and export defaults, and
quality-promotion status are unchanged by the runtime advance. This release
makes no DeepSeek-V4 (DSV4) qualification or support claim; that work remains
paused.

### Activation-fair pricing on the weight-only cost branches (P5a)

- Fixed the audit's first cost-model asymmetry: W4A4-vs-W8A8 activation cost
  was priced **only** on the measured `output_mse` branch, so packed experts
  under `PRISMAQUANT_EXPERT_COST_SAMPLE` and ladder-interpolated rungs under
  `PRISMAQUANT_CB_LADDER_INTERP` — most rows of a production run — were
  priced weight-only, crediting NVFP4-CB with its cheaper index stream and
  none of its A-side cost. The allocator now calibrates one per-format-family
  correction per run (geometric mean of the measured-over-weight-only Δloss
  ratio, over the rows that carry both estimators) and applies it to that
  family's weight-only-priced rows.
- The correction is multiplicative, so it cannot reorder rungs within a
  family (the holdout-gated ladder shape is untouched) and cannot lift an
  exactly-0.0 price off the DP's global minimum — the existing
  `activation_cost_unmeasured` candidate removal keeps full strength.
- Fail-closed: a run that would hand the DP a **mixed** scale (one family
  calibrated, another still uncorrected) refuses by name. A run with no
  measured activation rows anywhere corrects nothing, prints the verdict, and
  stamps it — no currently-legal run becomes illegal.
- Added `PRISMAQUANT_ACTIVATION_FAIR_PRICING` (default on; `0` reproduces
  prior pricing bit-for-bit), wired as the pipeline knob
  `ACTIVATION_FAIR_PRICING` and documented in `docs/design/runtime_flags.md`.
- Every candidate now records which estimator priced its activation contract,
  and the fit — sample, digest, residual band, per-rung dependence — is
  stamped into `format_applicability.json` and `selection.json`.

### Cross-family CB-ladder symmetry verdict (P5a)

- Fixed the audit's second asymmetry: the per-family RD-law ladders were never
  cross-calibrated. The expert cost stage now records each ladder's family and
  its **signed** holdout residual, and computes a family-symmetry verdict over
  held-out units with a tolerance derived the way `_cb_ladder_holdout_tol`
  derives its own — the sampling noise of the difference, floored at each
  family's declared resolution. No taste constant.
- A failure does not abort: it publishes
  `cross_family_comparison_publishable: false` with the numbers into the cost
  provenance, and the allocator republishes it in its diagnostics and
  selection provenance.

### Gridbook serving eligibility as candidate metadata (P5b)

- Fixed the audit's third asymmetry: the producer modelled exactly one
  gridbook kernel gate (`in_features % 256`). The `nvfp4_cb` serving profile
  now also declares the N-dimension load gates — `out_features % 8` for the
  fp4-CB families, `out_features % 16` for fp8-CB — per grid from the
  `cb_layout` family table.
- Added a declarative `serving_lanes` block: per CB format family, the served
  activation contract (`w8a8-dynamic-e4m3` vs `w4-bf16-bridge`), the fused
  mid-M rung set **as data keyed by the pinned Gridbook runtime version**, and
  the fallback route. Gridbook 0.5.0 backs FP8-CB fused mid-M for
  K ∈ {28,32,36,40,44,48}, and **Gridbook 0.6.0 — the version this release
  pins — backs the same set** (`nvfp4_cb.json` declares both keys; 0.6.0's
  K1.2 resolution proved `k % 4 == 0` is a format+TMA law, so the set is
  complete rather than pending); an undeclared runtime version backs nothing.
- Candidates carry the resolved route, and `selection.json` records which
  selected rungs ride a backed fused lane versus the expand+GEMM fallback —
  the producer-side mirror of gridbook K1.2, so neither repo can price an
  unbacked fast path.

### The constrained Pareto solver (P5c)

`docs/lanes/nvfp4-cb/format-speed-policy.md` §1 specified this solver and
deferred it ("not yet implemented"). It exists now; that paragraph has been
replaced with what it does and what still gates promotion.

- Added `prismaquant/serve_dispatch_table.py`: a torch-free declarative schema
  (`prismaquant.serve_dispatch_table.v1`) for measured per-(format-family,
  phase, M-regime, serving-lane) serving costs. **Provenance is mandatory on
  every row** — source document, date, GPU identity, measured quantity, units,
  and the derivation from the published number to the ratio — and a row
  without a source is a load error, not a defaulted field.
- Each `(phase, M-regime)` **arena** names exactly one reference route, so
  ratios measured against different denominators can never be silently
  composed (the 27B 1.44× is against a native artifact; the fused mid-M
  1.04×/1.26×/1.45× are against FP8-CB's own expand+GEMM route — multiplying
  them would manufacture a measurement). Isolated-operator (`operator_ms`)
  arenas and arenas with no published absolute are kept as evidence but are
  never SLO-eligible: policy §5, "raw standalone kernel timing is never served
  evidence".
- Shipped ONE example table,
  `prismaquant/serve_dispatch_tables/gridbook_gb10_2026-08-01.example.json`,
  populated **only** from measurements already published in Gridbook, each row
  citing its source. It is marked proposal data in both the file and the
  module docstring. It deliberately has **no whole-model NVFP4_CB row**: none
  is published, so an assignment containing NVFP4_CB cannot be certified
  against a latency SLO from it, and the evaluator refuses rather than
  interpolating.
- Added `prismaquant/serve_constraints.py`: policy §1's hard constraints
  (p95 TTFT, p95 ITL, p05 TPS, `resident + KV + peak_scratch`) evaluated on the
  exact expanded assignment. **No λ-blended objective anywhere.** Prefill and
  decode stay separate constraints. An assignment that misses an SLO is
  INFEASIBLE — removed from the candidate set, never re-ranked — and the
  objective and its tie-break (min predicted Δloss, ties toward the larger
  footprint) are unchanged among the survivors.
- Enforced at **assignment level**, in the byte-budget ratchet beside the
  exact byte filter, not inside `solve_allocation`. The DP is unchanged for
  the unconstrained case and that is pinned by test; the filter also sees the
  promoted, expanded assignment that actually ships, which the DP does not.
  The solver claims no global optimality it does not have: it stamps that
  every ACCEPTED assignment is feasible on both axes, not that the feasible
  set was enumerated.
- The aggregation model is explicit and stamped
  (`additive_layer_time__param_share_weighted__table_driven_proposal`) with
  **eight named assumptions** — additivity, parameter-share weighting, route
  locality, regime uniformity, baseline transfer, resident bytes, the
  single-stream `p05_TPS = 1000 / p95_ITL_ms` identity, and statistic
  transfer — carried in every artifact, along with policy §1's
  fastest-globally-feasible-assignment rule for any relative-tax denominator.
- Fail-closed: a unit with no dispatch row, an arena with no absolute
  reference, and an operator-microbenchmark arena all make the phase UNPRICED
  and therefore infeasible. "We could not price it" is never "it passed".
- Lane-aware pricing consumes P5b: a rung whose fused mid-M lane the pinned
  Gridbook version does not instantiate is priced with its **fallback** route's
  row, never the fused lane's. `FP8_CB_K36` (backed by 0.5.0) and
  `FP8_CB_K37` (not) therefore take different table rows despite sharing a
  family and a bpw class.
- `selection.json` records which constraints were active, which probed
  assignments the SLO axis rejected and the limit that rejected each, and
  which constraint binds at the shipped optimum. With no table and no SLOs
  supplied, every code path is byte-identical to the pre-P5c allocator apart
  from a stamp saying constraints were absent — pinned by an end-to-end test
  that compares `selection.json` and `layer_config.json` across a run with the
  feature absent and a run with it present-but-unused.
- New allocator flags: `--serve-dispatch-table`, `--serve-workload-mix`,
  `--slo-prefill-p95-ttft-ms`, `--slo-decode-p95-itl-ms`,
  `--slo-decode-p05-tps`, `--serve-device-budget-bytes`, `--serve-kv-bytes`,
  `--serve-peak-scratch-bytes`. Wired into `run-pipeline.sh` as
  `SERVE_DISPATCH_TABLE`, `SERVE_WORKLOAD_MIX`, `SLO_*`,
  `SERVE_DEVICE_BUDGET_BYTES`, `SERVE_KV_BYTES`, `SERVE_PEAK_SCRATCH_BYTES`
  and recorded in `STAGE_SETTINGS_ENV`. There is **no default workload mix**:
  policy §1 forbids one hidden in the allocator, and a latency SLO with no
  table or mix is refused by name.
- Design note: `docs/design/constrained_pareto_allocation.md`.

### D0.3 exact-rate experiment harness (P5d)

- Added `prismaquant/d03_exact_rate.py` and `scripts/run_d03_exact_rate.sh`:
  the two experiments gridbook ROADMAP **D0.3** names, run against a model's
  existing probe/cost artifacts. (i) `FP8_CB_K36` vs vanilla `NVFP4` on dense
  units at matched **exact whole-artifact bytes**, using the same non-additive
  accounting as the allocator's exact filter (shared CB codebook sidecars
  charged once per physical identity). (ii) Below 4.5 bpw, byte-neutral sweeps
  whose vanilla-NVFP4 promotions are **funded** by demoting other units down
  their own CB ladder, with a reclaim pass so each point sits at the baseline
  rate rather than under it.
- Each arm reports its assignment, exact bytes, predicted Δloss under the new
  activation-fair pricing, the P5c constraint verdict, and the serving-lane
  provenance (which selected rungs ride a backed fused lane).
- **Two refusals.** No cross-family verdict is printed when P5a's band check
  failed — suppressing it is that check's entire purpose, and printing it with
  a caveat would defeat it. No quality verdict follows when the two arms miss
  the ≤0.1% whole-artifact byte-match target policy §5 already names (the
  threshold the published 0.6B endpoint pair missed at +0.154%).
- **The harness prepares release-gate evidence; it does not claim it.** Every
  output is labelled proposal data pending the served NATIVE-PARITY protocol.
- Packed-expert vanilla NVFP4 is **excluded** from the contest and the
  exclusion is recorded explicitly in every report, citing gridbook **D0.2**:
  the producer profile denies stock NVFP4/FP8 on packed expert stacks because
  no stock-compressed-tensors packed-expert emit path exists, and building one
  is out of scope under the one-payload / no-new-packer rule.

### Routed-MoE stage attestation in the execution contract (gridbook K0.2)

- The NVFP4 W4A4 execution-contract record now carries a per-packed-FusedMoE
  stage section under `routed_moe_stages`
  (`prismaquant.nvfp4_w4a4_activation_stages.v1`). Each module attests BOTH
  stages — `w13` (the experts-module input) and `w2` (the routed intermediate)
  — with the stage label, the exact serialized physical target prefix, the
  input-global-scale policy, the calibration source that produced the scalar,
  and a per-stage value digest. A section digest covers the whole set. The
  scales were already stage-specific by construction (distinct physical
  targets; `unify_fused_sibling_input_global_scales` never joins across
  stages); what was missing was an attestation making that verifiable by a
  consumer.
- **Deliberate record-schema bump.** A record carrying the stage section
  declares `prismaquant.nvfp4_w4a4_activation.v2`; a dense-only record still
  declares `...v1` and is byte-identical to before. `target_values_sha256` is
  still framed with the **v1** literal, so the whole-model digest fields never
  move under the bump and an old reader verifies exactly what it always
  verified. The bump exists so a reader that cannot check stage attestation
  fails closed on a routed-MoE artifact instead of accepting a fused-readiness
  claim it cannot verify.
- `calibrated_input_global_scales_with_sources` reports which mechanism
  produced each scalar (target cache, parent experts-module cache, supplemental
  module-input sample, supplemental routed-intermediate replay, supplemental
  max-abs, packed-expert render max-abs). The stage attestation refuses an
  illegal pairing in either direction: `w2` can never be calibrated from the
  experts-module input, and `w13` can never be calibrated from a routed-
  intermediate replay.
- Fails closed exactly as before on a missing calibration input, and
  additionally makes it impossible to emit a routed-MoE artifact whose contract
  claims fused readiness with only one stage attested, or with no calibration
  source at all.
- All three emit paths build the section through one shared builder: the
  resident CB exporter, the streaming CB exporter (same inputs → byte-identical
  contract, as the existing resident-vs-streaming identity test requires), and
  the legacy native-compressed packed-expert path. The native container still
  publishes no `execution_contracts` record — its activation scalars remain
  optional/defaultable — but it now refuses to render a packed FusedMoE stage
  whose sibling stage has no calibrated max-abs.

### Gridbook runtime pin advanced to 0.6.0

- `prismaquant/gridbook_runtime/gridbook_runtime_pin.json` now names Gridbook
  **0.6.0** at exact commit `ca0f0f562d3f398e094bfa5356a9ce3fa47472f1` — the
  peeled `refs/tags/v0.6.0^{}` commit, not the annotated tag object. The
  required `pinned Gridbook contract` CI job installs that exact VCS revision
  and checks PEP 610 provenance, the packaged `runtime_contract.json`, producer
  profiles, rungs, layouts and emitted artifacts against it; all of it passes
  unchanged at 0.6.0, so no producer surface moved under this advance.
- The `nvfp4_cb` serving profile's FP8-CB fused mid-M lane now declares
  `"0.6.0": [28, 32, 36, 40, 44, 48]` — the SAME set as 0.5.0, and added as a
  NEW version key rather than by editing the old one. An artifact produced
  under the 0.5.0 pin therefore stays resolvable at the route it actually
  shipped on, and a pin with no key here still backs nothing (fail-closed).
- That set is now known to be **complete, not partial**. Gridbook 0.6.0
  resolved ROADMAP K1.2: `k % 4 == 0` is a format + TMA **law**, not a build
  option — `type_size = 4k` is the packed-B TMA box's contiguous extent and
  must be a 16-byte multiple, and the fused mainloop decodes with a single
  sub-table width `CbSubW = k/4` while the format splits `k` over `n_sub = 4`
  raggedly, so at k37 the true widths are `(10,9,9,9)` and a uniform decode
  would be *wrong*, not merely unaligned. The five off-law rungs of the
  published 27B K36..K47 ladder are therefore permanently expand+GEMM-served,
  and the allocator prices them on the fallback row for good rather than
  pending coverage that cannot arrive. The compiled set is queryable
  (`cb_fused_kbits()`), so the declaration is checkable against the runtime
  rather than transcribed from it.
- **No FP4-CB backed set was added.** Gridbook 0.6.0's contract-preserving
  FP4-CB v2 fused mid-M kernel exists only as an opt-in behind
  `PRISMAQUANT_CB_FP4_FUSED_MIDM=1`, pending its served NATIVE-PARITY gate;
  with the flag unset the dispatch is byte-for-byte the BF16 bridge. The honest
  backed set for the DEFAULT contract is therefore still empty, and the lane's
  `detail` records the available-versus-backed distinction and cites the flag.
  Pricing a rung on a lane the default serve never takes is exactly the P5b
  defect this data exists to prevent.

## 0.5.2 — 2026-08-01

This patch release advances the immutable runtime boundary to Gridbook 0.5.0
at exact commit `593f524e0a5d73b18e56d290a7b1355e66b2f9ce`.

Gridbook serving is now native CUDA/CUTLASS-only. Required native kernels are
attested at model load and missing or ineligible kernels fail closed instead of
falling back to Triton or another serving implementation.

The PrismaQuant producer ABI, format menu, allocation and export defaults, and
quality-promotion status are unchanged. This release makes no DeepSeek-V4
(DSV4) qualification or support claim; that work remains paused.

## 0.5.1 — 2026-08-01

This patch release makes fused NVFP4 W4A4 artifact eligibility explicit and
auditable while keeping fused serving opt-in. It does not claim default
enablement or a served-quality promotion.

### Versioned fused-activation contract

- Added one versioned NVFP4 W4A4 execution contract for production FP4-CB
  exports, including calibrated per-target `input_global_scale` tensors,
  fused-sibling scale unification, and a digest binding the serialized mapping.
- Added fail-closed coverage and provenance checks plus a serve-faithful
  activation-QDQ oracle. Legacy and unstamped research artifacts remain
  readable by their baseline paths but are not eligible for static fused
  dispatch; Gridbook's explicit rowwise fused research path remains available.

### Streaming and exact accounting

- Made resident and streaming exporters share the same activation-contract and
  served-target namespace rules, including packed-expert calibration synthesis.
- Accounted for FP4-CB activation-scale tensors and stock NVFP4 sidecars in
  whole-artifact bytes and bit totals, including weight-only W4A16 targets.

### One scale policy owner

- Consolidated activation-scale formulas, fused-unit grouping, calibration,
  and legacy compatibility behavior in one producer-owned module so native and
  CB exporters cannot silently drift into different contracts.

## 0.5.0 — 2026-08-01

This release establishes the production boundary between PrismaQuant and
Gridbook. PrismaQuant owns quantization, allocation, serialized-byte accounting,
and artifact export; Gridbook alone owns serving code, kernels, runtime flags,
tests, packaging, and releases.

### One runtime, one producer contract

- Deleted the complete vendored Gridbook runtime, CUDA/HIP sources, runtime
  tests, and source-sync machinery (38,216 lines removed).
- Added one immutable Gridbook commit pin, PEP 610 provenance checks, a packaged
  consumer contract, and tiny real-artifact compatibility tests.
- Consolidated producer-owned CB layout and export metadata in
  `cb_layout.py` and `cb_export_config.py`, shared by resident and streaming
  exporters.
- Moved the sole Gridbook pin and resolver into packaged assets under
  `prismaquant/gridbook_runtime/`. This is an intentional 0.x interface change:
  external scripts that sourced `scripts/lib/gridbook_runtime.sh` must source
  `prismaquant/gridbook_runtime/gridbook_runtime.sh` instead.

### Exact accounting and constrained selection

- Unified serialized-payload accounting across candidate construction,
  allocation, reporting, and exporter assertions, including FP4-CB layout-v2
  scale planes, FP8 per-row scales, and shared codebook sidecars.
- Replaced blended latency scoring with quality minimization under exact whole-
  artifact bytes, phase-specific serving SLOs, memory, backend, shape, TP, and
  serving-unit constraints.
- Excluded signed S13-S16 rungs from production menus while retaining research
  export and decoder compatibility.
- Fixed partial LFM packed-expert CB export layouts.

### Packaging correctness

- Wheels and sdists now include the canonical IQ grids and NVFP4/FP8-CB lattice
  tables. Earlier distributions omitted them: IQ failed at first use and CB
  could silently regenerate expensive lattices.
- The shipcard CLI is now installed as
  `python -m prismaquant.shipcard_cli`; the packaged pipeline no longer points
  at a checkout-only `tools/shipcard.py`.
- Distribution and clean installed-wheel gates now exercise every model,
  serving and lane spec, both tensor-table assets, the exact Gridbook pin and
  resolver, the pipeline, and the shipcard CLI.

### Fused NVFP4 safety decision

Gridbook's installed-wheel CUDA operator gate passed, but the teacher-backed
LFM2.5 A/B rejected promotion (exact full-vocabulary KL 0.247178, delta NLL
+0.054964, perplexity +5.65%). Dense and grouped fused-NVFP4 paths therefore
remain explicit opt-ins and default off in Gridbook 0.4.1.

## 0.4.1 — 2026-07-30

Tied-embedding models could not be quantized at all. Found by running the
pipeline on a real checkpoint rather than by reading code.

### A tied `lm_head` is structurally non-quantizable

On `google/gemma-4-31b-it` the cost stage cleared all 60 body layers, skipped the
vision-tower shards, then died on the `lm_head` shard with
`NotImplementedError: Cannot copy out of meta tensor`. Cause: the config declares
`tie_word_embeddings: True` and the checkpoint ships **no `lm_head` tensor at
all** — only `model.language_model.embed_tokens.weight` — so `lm_head.weight` is
a tied alias that nothing materialized. `tie_word_embeddings` appeared nowhere in
the streaming or cost path; there was no weight-tying support. Every
tied-embedding model hit this, which is most of the Gemma family; it went
unnoticed because every shipped artifact (Qwen3.6-27B, Qwen3.5-35B-A3B, Hy3) is
untied.

The head is now **materialized** (phase-2's CE backward runs through it, so meta
is never acceptable) via transformers' own `get_output_embeddings()` /
`get_input_embeddings()` accessors, so no embedding path is hardcoded and the
VL-prefixed name resolves like the plain one. Detection is from the config
declaration plus the index's absence of a head tensor — never a name guess. A
meta head with **no** declared tie now raises immediately instead of surfacing
thousands of lines later.

And a tied head is **excluded from probe, cost and the DP**, rather than
measured. Tying means one `Parameter`: quantizing the head would quantize the
embedding, and the surrogate cannot see that cost — probe and cost measure only
the head's output MSE, while the identical perturbation enters every token
embedding and thus layer 0's input for the whole forward, which no surrogate and
not even the L2 perturbed-X fixed point observes. There is also nothing to
re-encode: a tied source has no `lm_head.weight` bytes, so the footprint would
either fail to resolve the name or subtract the embedding from the floor while it
still ships verbatim. The codebase had already reached this conclusion in one
place — `aura_cost.py` hard-raises on a tied head with the same argument — so
this makes automatic what was an operator instruction, and extends it to the
L1/L2 path AURA does not cover. The exclusion deliberately ignores
`--allow-pinned lm_head`, because the tie is a property of the checkpoint rather
than of the serving profile.

Also removed: an ad-hoc repair in the probe that hardcoded three embedding names
inside a `try/except Exception` that only warned.

### Measured end to end

With this fix, Gemma4-31B completes **probe → cost → allocate → export** for the
first time. The probe was already passing (411 rows, all nonzero `h_trace`, 60
layers); cost now completes with zero errors; the allocator hits
`achieved_bits=6.000` with a genuinely heterogeneous 244 NVFP4 / 119 FP8 / 27
BF16 assignment; and the export writes a 27.18 GB compressed-tensors artifact
whose `config_groups` carry 4-bit `tensor_group` and 8-bit `channel` schemes,
with `tie_word_embeddings` preserved and **no `lm_head` tensor** — the embedding
ships once, so the tie is not silently materialized into duplicated bytes.

That run used a deliberately tiny calibration (2 samples, seqlen 512) to reach
failures fast. **It is an enablement result, not a quality claim** — the artifact
has not been served and no KL/PPL has been measured.

## 0.4.0 — 2026-07-30

Closes #29 and lands the KV-cotangent path, which removes the default-off guard
that was blocking KV-sharing architectures. Minor rather than patch because the
Fisher measurement for KV-sharing models changes (it was wrong), and because an
export that previously succeeded by silently demoting FP8_SOURCE now behaves
differently. Shipped allocations are unchanged (35B: 0 of 500).

### Fisher: the KV-cotangent path (part of #9, closes MINOR-M33)

Gemma4-style architectures share K/V across layers: a "storing" layer computes
K/V and later "sharing" layers consume them. Phase-3 forwards each layer in
isolation and handed the consumer a **detached** K/V, so its backward stopped
at that boundary and the storing layer's `k_proj`/`v_proj` Fisher never saw any
consumer's contribution — an under-count on precisely the layers that feed other
layers. That is why `num_kv_shared_layers > 0` was blocked behind
`PRISMAQUANT_ALLOW_KV_SHARED_FISHER`.

Consumers are now handed grad-enabled leaf clones; their `.grad` is the cotangent
each contributes, accumulated per storing layer and used to seed that layer's
backward alongside its own output cotangent. Phase-3 sweeps in reverse and
`kv_shared_layer_index` is derived from layers strictly below the sharing point,
so every consumer is harvested before its producer is forwarded — one pass, no
disk state. Both facts are pinned against the installed modeling source.

**Verified by exact equivalence, not plausibility:** on an fp64 synthetic model,
`h_trace` through the isolated protocol is bit-identical to a single end-to-end
autograd backward (relative error 0.00e+00), while the pre-fix protocol
under-counts `k_proj` by 85.1% and `v_proj` by 38.5%.

Three things the equivalence surfaced that the design did not predict:

- **The under-count was never confined to k/v_proj.** Phase-3 chains each layer's
  input gradient downward, so the producer's truncated input gradient was
  inherited by every layer *below* it — all of `layers.0.*` moves without the fix.
- **The Fisher hook must fire exactly once.** These hooks pop their saved forward
  input, so a backward hook firing once per root would silently drop half the
  Fisher; both roots go through one `torch.autograd.backward` so autograd
  accumulates at the shared node first. Pinned by counting hook invocations.
- **A borrowed leaf must never be seeded as a root.** In reverse order a consumer
  is handed a container keyed identically to the entry the previous consumer just
  filled; seeding it would inject one consumer's cotangent into another's harvest.

The guard is inverted rather than deleted: it now fires only when the cotangent
path is unavailable (`PRISMAQUANT_KV_COTANGENT=0`), and
`PRISMAQUANT_ALLOW_KV_SHARED_FISHER=1` still reproduces a pre-fix probe. Models
without KV sharing are bit-for-bit unaffected with the accumulator on or off.

**Honest limit:** no real `num_kv_shared_layers > 0` checkpoint has been probed.
The percentages above are a correctness demonstration on a toy; the real-model
magnitude is unmeasured. Three conditions would still make such a probe unsafe
and are documented in code: non-differentiable shared state (no cotangent
exists), cotangent left unclaimed at sweep end, and an architecture whose
consumer sits below its producer — the last two surface as diagnostics rather
than wrong-but-silent numbers.

### Export: passthrough source integrity (#29)

The runtime coercion never passed `source_kind`, so passthrough-integrity judged
**every** `FP8_SOURCE` Linear illegal and rewrote it to BF16. The bytes were fine
(the config overlay restored it, materialization copied verbatim) but every
FP8-source artifact's `runtime_coercions` was full of demotions that never
happened — making a real coercion invisible — and it forced a passthrough
exemption in 0.3.1's serving-group escalation.

The source dtype now comes from `_scan_source_dtype_manifest`, the same
recipe-keyed map `allocator.main` feeds `build_candidates` to gate passthrough
candidates, so the exporter judges legality against exactly the vocabulary the
gate that admitted the allocation used. It is scanned lazily, so a BF16-source
export does no extra header IO. Bogus rows went from 4-of-4 to 0 on a synthetic
fp8 checkpoint, and the exemption is gone, so a genuine passthrough mismatch
inside a serving unit now escalates like any other illegality.

No bespoke raise was added, on a measurement: a genuinely non-fp8 `FP8_SOURCE`
assignment is repaired by the coercion rather than the overlay, so it already
ships as BF16 today and a hard raise would be the only change turning a
succeeding export into a failure. The 0.3.1 policy decides instead — refuse
inside a serving unit (naming the legal rungs and the byte cost), coerce alone
when dense, now with a true `delta_bytes` and a passthrough-specific banner.

One required side-fix: with `FP8_SOURCE` surviving the guard it reached
`_production_cache_expected_keys` for the first time, and since its emit branch
returns before the packer, that check would have demanded a render entry nothing
reads — newly failing a valid FP8-source export. All passthrough formats are now
skipped there, not just BF16.

### Streaming: text-only skeletons for vision-language wrapper configs

`CALIBRATION_MODALITY=text-only` decided *what to calibrate on*, but it also
silently decided *how the skeleton is built*: the multimodal path instantiates
via the declared architecture specifically to bypass `AutoModelForCausalLM`'s
text-only downgrade, while the text-only path passed the top-level config
straight to `AutoModelForCausalLM` — which fails on any model whose wrapper
config is not in that mapping (reported on MiniMax-M3 in #12, where the error's
own accepted-class list contains only the model's *text* config).

The text-only path now falls back to the text sub-config **class** rebuilt from
the staged top-level keys. That distinction matters: `stage_text_only` pops the
nested `text_config` and lifts its keys up, so the sub-config object still
hanging off the wrapper is default-constructed and reading dimensions from it
would silently build a wrong-sized skeleton. Detection asks the same two
questions `from_config` asks itself (remote-code `auto_map`, then membership in
the mapping the call consults), so it is config-only and contains no model-type
or class name; a config the auto class can resolve returns the identical object
and takes the original path unchanged.

**This makes no architecture supported.** MiniMax-M3 still needs a
model-structure profile and a serving profile, and the mechanism was validated
against two unrelated real wrapper families since no M3 checkpoint exists here.
The next wall for any VL checkpoint is tensor-name matching (a text skeleton
expects `model.layers.*` where a VL checkpoint often ships
`model.language_model.layers.*`), which is per-architecture profile work.

## 0.3.1 — 2026-07-30

Closes #28: a serving-atomic group could end up with a quantized + BF16 mix
inside it, reported only by the fused-coherence gate at the very end of export.
Fixed at both the cause and the safety net. Allocations on the shipped
Qwen3.6-27B and Qwen3.5-35B-A3B are unchanged (0 of 614 and 0 of 500).

### Cause: promotion now picks a format the whole unit can run

`_promote_group_components` took the highest-**rank** format assigned to any
member of a serving-atomic component and wrote it to all of them, with no check
that the format was legal for the rest — it only received `assignment`,
`format_rank` and `groups`, so it had no way to know. Members of one unit do not
share a shape (gate_up vs down differ on the reduce dim; an odd
`moe_intermediate_size` makes one projection's group/scale-block divisibility
fail while the other's passes), so the promoted format could be illegal for a
subset.

Promotion now takes per-row legal-format sets (`legal_formats_from_candidates`)
derived from the candidate lists, which already encode source-passthrough
integrity, serving-profile rules, group/scale-block divisibility and kernel
shape rules. It picks the cheapest legal-for-all format at or above the max
rank — preserving promotion's non-degrading contract, which
`solve_with_promotion`'s tightening loop is built around — and only downgrades
to the highest legal-for-all when nothing above is common. In the illegal case
every member is written unconditionally, since a member on an equal-rank but
different format would otherwise survive and leave the unit mixed. No common
legal format raises, naming every member with its legal set and the three
upstream causes.

The argument is optional: omit it and the legacy max-rank path runs verbatim, so
callers that cannot supply legality (auxiliary MTP/visual pins, hand-built
assignments) keep today's behaviour rather than acquiring a new failure.

Two paths were genuinely reachable and are now covered: the un-aggregated
(`--no-packed-aggregation` / `--no-fused-aggregation`) solve path, where
promotion is the only coherence mechanism — there the pre-fix symptom was
actually an aborted run, since `compute_achieved` refuses to price an
unpriceable member — and the Pareto seed-JSON promotion, which is **not** priced
by `compute_achieved` and so could let an illegal member format escape silently.
The aggregated path already intersected member candidate sets and needed nothing;
that is now pinned by a test rather than assumed.

### Safety net: export coercion is group-aware

The per-Linear shape/policy coercion (deliberately preserved in 0.3.0) could
rewrite a single member of a unit to BF16. It now resolves whole serving-atomic
components, unioning overlapping units — on the split per-expert representation
a Linear can be both a fused sibling and a packed-expert member — using the same
profile accessors the fused-coherence gate uses, never by parsing names.

The resolution is deliberately asymmetric. If some emittable quantized format is
legal for every member, export **raises** and names it: coercing would ship the
whole unit at 16 bpp (for a packed-expert unit, `num_experts ×` the per-Linear
cost), the dimension that made one member illegal is model-wide so it recurs in
every layer, and a re-solve lands the unit on that legal format for free.
Export must not substitute a format itself — the format the allocator picked is
the one the production weight cache holds a deliberate render for, so a
substitute is a cache miss at best and an RTN render at worst. Only when no
quantized format is legal for every member is BF16 the sole representable
answer; then the whole unit is coerced, loudly, with every member and the byte
delta recorded in `runtime_coercions` and the BF16 audit.

Since the cause is fixed upstream, this path should be unreachable in normal
operation, and the report is written to make any firing look like the upstream
regression it would be. One case it must keep catching regardless: rank-1 legacy
probe stats carry no shape, so `check_stats_format_applicability` admits a
shape-illegal format legitimately and the exporter is the only gate.

## 0.3.0 — 2026-07-30

Closes the three open issues that were ours (#27, #19, #9 item 1). Minor rather
than patch because an export that previously "succeeded" can now raise, and
because a vendored-modelling override that cannot take effect now stops the run
instead of silently continuing on the wrong code.

### Export refuses what it cannot emit (#27)

`export_native_compressed` now declares `EXPORTABLE_FORMATS`, derived from
`FORMAT_SCHEME` plus the container passthrough rather than hand-listed, and the
vLLM serving lane reads its menu from that one place. A format with no
compressed-tensors emit path is a **hard error** naming the Linear, the format
and the resolved profile — it used to be silently rewritten to BF16 with only a
`print`, so a Linear allocated at ~4.25 bpp would ship at 16, blowing the byte
budget and leaving the artifact's real bpp disagreeing with its own
`layer_config.json`.

The *legitimate* coercion is unchanged: a format the exporter can emit but which
is shape-illegal or profile-denied still falls back to BF16 and is still audited
into `mixed_native_manifest.json`. Two facts corrected by reading the exporter:
`FP8_SOURCE` **is** emittable (verbatim-copy path, no packer branch) and
`FP8_E5M2` is **not** (packer branch, no scheme entry) — so the set cannot be
derived from the packer branches. Menu unchanged for every lane.

Consequence worth knowing: allocating under the `research` profile and then
exporting compressed-tensors now fails loudly instead of shipping ~16 bpp.

### Vendored modelling overrides verify or die (#19)

`register_qwen3()` returned cleanly, set its "registered" flag, and on
transformers ≥ 5.13.0 did nothing — after which a probe ran **upstream** Qwen3
modelling code, on the architecture family behind most shipped artifacts, with
no exception anywhere. Root cause is upstream: `_LazyAutoMapping.register`
returns early whenever the config key's `__module__` starts with
`transformers.`, so no override of a natively-supported `model_type` can land
through that call.

- The override now genuinely applies, via public API only: a PrismaQuant-owned
  subclass of the native config (same `__name__`, non-`transformers.`
  `__module__`, picklable) registered through `AutoConfig.register`, which
  applies no such filter. No transformers internals are patched, and the
  fallback engages only when the direct route is verified dead.
- Every registration is **verified** by resolving it config-only, and a failure
  raises with the transformers version, the resolved class, the upstream
  file/function and the remedy. The "registered" flag is set only after
  verification, so a failure stays retryable rather than caching as done.
- `register_deepseek_v4` had a second silent no-op of its own and was resolving
  correctly only by module-path hijack; it now gets the same verification plus a
  guard against a foreign module occupying its path.
- `detect_profile` no longer loses that verdict: it consults the recorded
  override failures and refuses to hand back a profile whose vendored path is
  known dead. The surrounding `except Exception: pass` is correct for keeping
  detection alive, but it cannot be allowed to re-hide a silent no-op.
- The version boundary is now measured, not guessed: healthy through 5.12.1,
  broken from 5.13.0. The old `xfail` threshold of 5.7 was six minor versions
  pessimistic, and the `xfail` is gone — the suite goes red on the wrong
  modelling path.

### Gemma4 KV-sharing pass state (#9, item 1)

The per-forward-pass state hook had already landed; what was missing is that
`_save_precompute_cache` never persisted it and the load path omitted the field
entirely. Since the precompute cache is the normal path for a sharded or
resumed probe, the first checkpoint with `num_kv_shared_layers > 0` would
capture the shared K/V in phase 1, silently drop it on save, and `KeyError`
inside attention for every sharing layer in phase 3 — after hours of phase-1
work, untested in either direction. Fixed both ways, an old cache now hits a
loud error rather than a `KeyError` deep in attention, and a sharing layer with
no captured source K/V raises naming the layer and the remedy instead of
handing back an empty dict. The merge of pass state into per-layer kwargs is now
one shallow-by-design function that raises on key collision instead of silently
overriding.

Item 2 of that issue is unchanged and still needs a GPU run. Two caveats worth
carrying: `google/gemma-4-31b-it` has `num_kv_shared_layers = 0`, so the sharing
path is covered only synthetically until a genuinely KV-sharing checkpoint is
probed; and KV-sharing probes remain default-off because phase-3's isolated
forward detaches the borrowed K/V, under-counting the storing layers'
`k_proj`/`v_proj` Fisher — a cost-model gap, not a flag.

### Also

`F8_E8M0` reporting, the verified DSv4 source layout, and the routed-only
expert-declaration scope all shipped in 0.2.1 and are unchanged here.

## 0.2.1 — 2026-07-30

Corrects one thing that shipped in 0.2.0 on an unverified assumption, settled by
pulling the real `deepseek-ai/DeepSeek-V4-Flash` config and safetensors headers
(a few hundred KB — no weights) plus the authors' `inference/convert.py`.

- **Declared-MXFP4 expert scope is routed-only again.** 0.2.0 widened it to
  `shared_experts.*` on the reasoning that `expert_dtype` describes all of a
  layer's experts. The headers refute that: routed-expert weights are `I8`
  nibble-packs (2304/2304 sampled) while shared-expert weights are `F8_E4M3`
  block-FP8 (9/9), and the authors' converter gates its fp4 path on
  `"experts" in name and dtype == torch.int8`. With the widening in place a real
  DSv4 load would have pushed block-FP8 into the nibble decode and hard-failed
  the packed-grid assertion. No other model is affected — the widening only ever
  applied to a checkpoint declaring `expert_dtype: fp4`.
- `F8_E8M0` added to the safetensors dtype table (it fell to the unknown-dtype
  default of 2 bytes in `dominant_source_bytes_per_param`; span-based accounting
  was already exact).
- The verified DSv4 source layout is recorded in
  `model_profiles/specs/deepseek_v4.json`, including the accounting trap that
  its scales are 1-byte E8M0 planes named `.scale` rather than fp32
  `.weight_scale_inv`, so DSv4 byte accounting must use the per-tensor manifest.

Everything else confirmed as the code already assumed: the `expert_dtype` key
and value, `scale_fmt: ue8m0`, the E8M0 exponent bias, the packed grid, and — the
question that previously could only be guessed — the nibble order and E2M1 table,
which match the authors' reference decode value-for-value.

## 0.2.0 — 2026-07-30

First published release. 0.1.0 existed only as a version string in
`pyproject.toml` and was never uploaded anywhere.

Everything below is allocator/solver/footprint/profile logic. **No shipped
artifact is affected:** re-solving the shipped Qwen3.6-27B and Qwen3.5-35B-A3B
probe/cost pairs at `TARGET_BITS` produces the same assignments as before (0 of
614 and 0 of 500 changed), and the byte-budget floors are byte-identical on both
real checkpoints (27B 6.012 GB, 35B 4.661 GB).

### Allocator

- **Fisher `h_trace` is normalized by the global calibration token count**, for
  every row. Per-routed-token normalization inflated a rarely-routed
  per-expert-`nn.Linear` row by `global/routed` (typically `n_experts/top_k`),
  i.e. inverted importance weighting. Tokens never routed to an expert
  contribute a genuine zero that belongs in the mean-Δloss average. Dense and
  packed-3D probes are numerically unchanged; existing probes are corrected at
  load time from their stored raw accumulators, and a probe carrying raw
  accumulators without token metadata now hard-fails
  (`--allow-legacy-fisher-norm` restores the old warning path). This reverses
  the convention audit M4 documented; the reversal is recorded in `CLAUDE.md`
  §3 and `docs/prismaquant_design.md` §2.2.
- **Packed serving groups are first-class DP units.** A packed-MoE serving
  group is atomic at serve time but the DP priced upgrades per row while
  promotion charged the whole group — a systematic mispricing that starved
  cheap dense rows. `aggregate_packed_serving_groups` collapses each group into
  one multi-choice item, so the DP and the serving constraint price identical
  moves and post-DP promotion is a validated no-op. `--no-packed-aggregation`
  restores per-row pricing.
- **Solver termination is feasible-only.** A rung either returns an iterate
  satisfying `achieved <= target + tolerance` or reports INFEASIBLE; deep
  undershoot is recovered by bisection rather than shipped. Among feasible
  iterates the solver keeps **minimum predicted Δloss** (ties to denser), which
  is its actual objective — denser is not monotonically better. `--target-bits`
  runs that previously emitted an over-budget config now exit, with the format
  floor, what the floor solve promotes to, and the closest achieved bits in the
  message.
- **Byte-budget "fit the card" selection ships minimum predicted Δloss among
  the rungs that fit** (ties to the larger footprint), matching the solver's
  objective. Filling the card is a proxy that can select a denser artifact with
  worse predicted loss than a sparser one that also fits. `selection.json` is
  self-describing (schema `…byte_budget_selection.v2`): objective, feasibility
  test, the tightened search ceiling, whether bisection ran and why not, the
  full ratchet trace, and the max-bytes pick for comparison.
- **Bit-exact re-encode pricing is gated on an identity activation path.** A
  measured `weight_mse == 0.0` proves `W' == W`, but for W·A· formats the
  measured `output_mse` is real activation-side error, so pricing such an entry
  at zero Δloss handed the DP an unbeatable global minimum for a W4A4
  assignment. The short-circuit now requires
  `FormatSpec.act_quant_changes_input` to be false — a dtype-level declaration,
  pinned registry-wide. Relatedly, a W·A· candidate whose activation cost was
  never measured and prices at exactly 0.0 is excluded from the menu with a
  counted, logged reason instead of winning every budget.
- **Fused-sibling and packed-group UCB hedges aggregate in quadrature**
  (`z·√Σ(stderr·gain)²`) instead of linearly, which over-hedged an N-member
  group by up to √N. Byte-for-byte identical at the default `COST_UCB_Z=0`.
- **`--packed-role-split` hard-errors unless the resolved serving profile
  declares `supports_per_role_expert_schemes`** (GGUF only). It could otherwise
  emit gate_up=NVFP4 with down=FP8 in one MoE layer — a checkpoint vLLM cannot
  load. Role grouping now comes from the model profile rather than a projection
  table inside the allocator.
- A fused group whose members have disjoint format menus, and an assignment
  row whose format has no candidate to price it, are now hard errors naming the
  group/row instead of silently vanishing from the DP or scoring as free.

### Footprint

- **Source bytes are priced from an exact per-tensor safetensors-span
  manifest.** The regime-wide accounting charged every re-encoded Linear at the
  FP8_SOURCE layout as soon as any fp8 dtype appeared, which on a mixed source
  removed more bytes than the checkpoint holds and drove the non-quantizable
  floor negative — letting an artifact twice the budget "fit". A negative floor
  is now always a hard error.
- Tensors whose live-name mapper declines them (MTP sidecars, visual towers)
  keep their source bytes: the mapper answers "is this in the live graph", not
  "does this have bytes on disk". Without this, every `--target-disk-gb` run on
  an MTP-carrying model failed.
- Two re-encoded names resolving to the **same** source span is rejected
  structurally (the manifest carries per-entry span provenance), not by
  docstring convention. Charging both a per-expert name and its packed parent
  would subtract the expert mass twice.
- One accounting path: the byte-budget selector calls
  `footprint.assignment_artifact_bytes` rather than reimplementing the identity,
  and `source_manifest` is a required keyword so the legacy regime
  approximation cannot be reached by omission.

### Serving profiles

- **A profile's format menu is bounded by its lane's exporter**, read from the
  exporter's own declaration (`export_native_compressed:FORMAT_SCHEME`,
  `gguf_formats:GGUF_BLOCK_BYTES`) rather than a duplicated list. Weight-only
  A16 rungs were legal for dense Linears on the vLLM lane while the exporter
  cannot emit them. No production format was narrowed; GGUF is unchanged.
  `research` declares `emulation_only` instead of a lane, deliberately, so
  unserved rungs stay measurable.

### Probe / streaming (DeepSeek-V4-Flash enablement)

- MXFP4-packed routed experts dequant on a dedicated vectorized path, triggered
  by the checkpoint's `expert_dtype` declaration rather than a tensor-shape
  heuristic, with the packed grid and the E8M0 scale-plane dtype as assertions.
  Shared experts are covered by the same declaration. E8M0 `0xFF` decodes to
  NaN. Bit-exactness is pinned against an independent scalar reference.
- Nibble-packed `I8`/`U8` expert tensors are sized at 2 logical elements per
  disk byte by both pre-load cache estimators; sizing them verbatim under-counts
  the resident tensor 4× and makes prefetch silently refuse layers.
- Compressed-sparse-attention layer types and the rope-axis `layer_types` dict
  are handled; phase-1 activations stream to host per layer
  (`PRISMAQUANT_PROBE_BATCHED_ACT_TRANSFER=1` restores the batched transfer).
- Per-expert cost rows resolve, and `PRISMAQUANT_EXPERT_COST_SAMPLE` works on
  the default `COST_MODE=production-render-score` path.
- h-detail blobs record their normalization denominator and a stale directory is
  refused rather than mixed with differently-normalized scalars.

### Packaging / CI

- CI runs the test suite on every push and pull request (Python 3.11 and 3.12,
  CPU torch) plus an import-surface job.
- Tag-driven release pipeline (`docs/RELEASING.md`): builds, asserts the tag
  matches the built version, asserts the runtime JSON specs and
  `run-pipeline.sh` are packaged in both wheel and sdist, verifies a
  non-editable install resolves those specs from site-packages, then publishes
  to PyPI via Trusted Publishing (no API token) and creates the GitHub Release.
- `prismaquant.__version__` is resolved from installed metadata, so
  `pyproject.toml` stays the single source of truth.
- Three tests that drive repo-root `tools/` scripts skip cleanly when run
  against an installed package instead of failing collection.
