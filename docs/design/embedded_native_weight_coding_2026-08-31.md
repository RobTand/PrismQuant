# Embedded-native weight coding (EN4/EN8): sub-4-bit formats that terminate on FP4 tensor cores

**Status:** proposal, revision 7. One normative text; no pointer
supersession. **The prohibited source model is barred everywhere, for any
reason** (owner directive, enforced by rounds 7–8): no prohibited model,
calibration,
tokenization, activation cache, probe, cost table, assignment, tensor
sample, corpus result, derived identity, or mixed-source handover may be a
Tessera input or load-bearing citation — relabeling or copying does not
change identity. Measurement-only work is permitted under §14's GLM-only
manifest; Tessera encoder, schema, allocator, menu, export, and serving
implementation are gated on the round-7 conditions. Experimental flags stay
default-off until the `design_guidelines.md` validation gate is passed.

**Provenance:** authored 2026-08-31 (session coordinator: GLM-5.3 via
opencode), hardened through eight review rounds — Claude rounds 1–7, Codex
rounds 3–4 — with every load-bearing citation verified in this checkout
before acceptance wherever the artifact was reachable. Evidence tiers used
throughout: `[in-tree]` (verified in this checkout — PrismaQuant or
gridbook, both repositories in this document's scope), `[branch]` (artifact
on
a named branch — `origin/claude/prismabuild-trellis-integration-20260831`
unless the row says otherwise — auditable there; that branch was fetched and
its six result artifacts verified present and content-checked at tip
`bd5de5c`, 2026-08-31 — **presence- and content-verified, not re-derived**;
re-derivation is the permitted reproduction work of §14 and is what promotes
these rows), `[branch-preliminary]` (microbenchmark-measured on
a named branch but not decision-quality under rule 13 — no in-process
profile, no two-box Netdata), `[review-reported]` (no artifact located; downgraded
per Codex P2-7), `[derived]`, `[arithmetic]`, `[conjecture]`, `[retracted]`.

**Revision history (detail lives in the review files, not here):**

- rev 0 (2026-08-31): original proposal.
- rev 1: round-1 fixes — rank-1 diagonals (§5 assumption (c) corrected),
  three-state pin picture, 4.5/4.25 arithmetic, floor accounting.
- rev 2: round-2 answers — `su` joint fit, segment 3a, clip tracking,
  bypass relabel, rotation factorial.
- rev 3: round-3 — Codex's seven findings (root-rate forest, CDNA4 vendor
  correction, partner completion, decode-deterministic orders, three artifact
  layers, contract status, guideline-bar plan) + Claude's measured
  falsifications (sidedness, residual whiteness).
- rev 4: round-4 — the streamed prior reclassified (unfused class); post-v2
  dating; decode-pool fix; Strix Halo.
- rev 5: round-5 — the measured handover: fused-decode pipelining negative;
  BlockLDLQ negative; scale-structure table; body rate cannot close the band.
- **rev 6 (this text): the fold.** Absorbs Codex round-4 (refinement grammar
  derived, branch DAG, honest terminal semantics, event order, encoder
  labeled heuristic, layer consolidation, P2-7 downgrades) and Claude round-6
  (the reopening bar is ~9.7×, not 4.7×; the alphabet refusal; the duplicate
  §21 absorbed and deleted). Companion review records:
  `docs/handovers/claude-findings-for-glm-2026-08-31.md`,
  `docs/design/embedded_native_weight_coding_2026-08-31_codex_round4_review.md`.
- **rev 6, round-7 amendments** (`claude-round7-review-2026-08-31.md`, all
  six items accepted after verification): the three former
  `[review-reported]` rows upgraded to `[branch]` — artifacts committed at
  branch tip `7670867` (ls-remote verified) and `qtip_rotation_weight_side_2026-08-31.md:20`;
  the EXL3 kernel facts promoted to `[in-tree]` (local reference checkout,
  all five line citations verified); the rank-1 row's "exact activations"
  corrected to "no activation model"; the 9.7× row carries its directional
  caveat; §17.6 de-circularized; build-order reversibility note added.
- **rev 6, post-round-7:** the whiteness rung replication landed (`bd5de5c`,
  ls-remote verified as branch tip) — six of six cells white across
  q256 512/768/896, both rotation states, s₁/s₂ ∈ [1.006, 1.163], rank-4
  energy 0.73–0.91%. Segment 3a's gate rests on that evidence (scope
  later extended by round 7: also q256 {256,384,640} and `R_in`-only;
  "white wherever this trellis operates" is superseded by the exact
  scope). Remaining scope: the extended rungs and states, GLM5.3-flash
  tensors, and a second seed pair.
- **rev 6, Codex round-5 amendments** (all accepted after verification;
  P2-6's absence finding predates the push — the fused-decode artifact is
  present and content-verified on the fetched branch tip): §9 rewritten as
  the typed-planes wire contract — an interleaved event stream ordered by a
  positional key is not uniquely decodable (P0-1); the 9.7× figure replaced
  by the bpp-dependent feasibility function with the regime-split caveat
  (P1-2); the scale-mantissa codec specified as §6b, with 2b and
  T-nvfp4-class conjectural until round-trip tests land (P1-3); two-sided
  rotation removed from serving branch identity — `R_in`-only is the only
  algebraically local state (P1-4); segment 3b and dual-accumulate removed
  from the normative grammar as a separately gated future proposal (P1-5);
  the 3.75 traffic ledger row deleted; the fused-decode reproduction added
  to the permitted set.
- **rev 6, Kimi K3 + Codex round-6 amendments** (all accepted; Kimi
  verified the §6/§13 arithmetic exactly and found two procedural defects;
  Codex round-6 revalidated every round-5 fix and raised one P0): terminals
  become **classes** — every encoder/allocator candidate gets a
  `terminal_id` manifest record with complete per-plane count arrays, clip
  code, exact bytes, and exact bpp (Codex P0-1); build item 1 splits into
  1a (reviewed byte-level schema and parse algorithm) and 1b (pure
  serializer/parser/footprint tests), with the schema moved ahead of 1b and
  no shipping wiring before 1b passes; §6b gains the legality-contract
  obligations and reuses the shipped two-tier scale abstraction, with
  "parity" renamed to **wire-rate parity** (P1-2); branch identity gains a
  content-addressed `encoder_profile_id`, and arm 8 splits into 8a/8b with
  a numeric regret threshold and only named runtime-contract lanes
  eligible (P1-3); the selected-prefix resident path gets an explicit
  `ProductionWeightCache` ownership contract before any lane extension
  (P1-4); the fused-decode row is relabeled `[branch-preliminary]` and
  "dead" becomes "blocked by a strong negative prior" (P1-5); the whiteness
  claim is scoped to exactly q256 {512,768,896} × {unrotated,
  two-sided-measurement}, with {256,384,640} and `R_in`-only added to the
  remaining work (P2-6); the projected po2 floor is corrected to ~1.25 bpp
  (the legacy per-256 two-tier plane's floor is re-derived in the pure
  calculator artifact before any quotation — round-8 P0-1.3); "never stores 4
  bits" becomes "never a dense scalar 4-bit plane — released positions
  append a 4-bit override"; the ARCHITECTURE.md carry-forward moves into
  build step 3 per rule 12 (Kimi D1); and §18 now carries resolvable
  pointers to every absorbed review round (Kimi D2). Kimi's follow-up
  (F1–F3): the footprint ledger row is aligned with the
  `[branch-preliminary]` relabel (F1); the ARCHITECTURE.md line list is
  completed to include 352 and 367, marked illustrative (F2); the
  whiteness scoping confirmed consistent across §5, §17.6, and the ledger
  (F3, clean).
- **rev 6, owner directive (fail-fast, 2026-08-31):** performance is
  tested first — "the cheapest decisive measurements run before the work
  they would invalidate" is now §16's ordering principle. The skeleton
  parity measurement is decoupled from the parse tests (a stub wire layout
  suffices for the verdict) and promoted to item 0's fail-first track
  alongside the fused-decode reproduction and the GLM5.3-flash campaign;
  arm 2's minimal measurement encoder becomes the first gated-work
  request, ahead of all plumbing.
- **Naming (owner grant, 2026-08-31):** the mechanism is named
  **Tessera**. The EN4/EN8 families become TESSERA-4/TESSERA-8 — wire
  spelling `TESSERA_E2M1_R{q256}` / `TESSERA_E4M3_R{q256}`, and the stream
  schema is born at build item 1a as `prismaquant.tessera.v1`. The name
  records the mechanism's facts: the E2M1 codes are uniform tiles; the TCQ
  trellis is the lattice they assemble on; every declared stage of the
  tessellation is a complete picture (the embedded property); the
  diagonals and scale planes are the grout — the cheapest binding
  material measured; and the terminal classes are the frames, the picture
  finished exactly when it fits them — the vendor-native FP4/MMFP4
  operands.   The document-wide sweep lands as one commit with 1a, where a
  **disjoint Tessera parser** and the collision tests land in the same
  commit — the legacy parser's accepted language and return type are
  unchanged (round-8 P2-8) — alongside a name-novelty check. Until
  that commit, EN4/EN8 remain this document's internal designations.
- **rev 7 (Codex round-7):** the proposal did not pass.
  **P0, owned by the author:** the measurement authority was contaminated
  with a prohibited-source calibration artifact, corpus, and statistic —
  written into the fold without checking corpus identity, the same
  read-the-number-not-the-verdict failure class this session's reviews
  had already caught three times, now at the corpus level, and it survived
  eight rounds because no reviewer checked provenance until this one
  (identities and figures live in the round-7 review file, an append-only
  record, not an active citation). Round-7's remedy — tombstone rows and
  "model-scoped priors" — was itself ruled a violation by round 8:
  retaining the numbers or citations in any form is still use. All
  GLM arms now run under content-bound measurement contracts with verified
  transitive input closure. Six P1s
  accepted: exact positive subnormals are legal in the scale predicate
  (P1-2); `encoder_profile_id` is input-only with post-encode receipts
  (P1-3); producer cache and Gridbook serving residency split at the
  artifact boundary   (P1-4); Gridbook 0.9.1 is labeled legacy-TCQ and an
  external Tessera release gate is added — GLM is dense-only until a
  routed cell exists (P1-5); the fail-first stub is left
  unauthorized — a mathematical lower bound is owed before any
  constructive kernel counts, the full-layout skeleton is a separate
  post-1b gate, and a post-release serving promotion arm added
  (P1-6); the Tessera name gets persisted-identity rules — human
  descriptor versus one normative content-bound record, disjoint parser
  (P1-7). P2-8 mechanics applied; remaining `en-stream.v1` references
  replaced by `tessera.v1`; the revision advances to 7.
- **rev 8 (Codex round-8, final):** the pedantry round, and it was right
  to be. P0-1: excision, not annotation — every prohibited-source row,
  number, rationale, and citation deleted from the active text and
  ledger (tombstones were still use); a CI check with content-addressed
  ancestry guards the document and manifest. P0-2, verified in this
  checkout: **no BF16 GLM source exists** — the identified checkpoint is
  FP8 on every quantizable parameter (305.9 B params, 100% F8_E4M3) and
  the driver hardcodes an absent path; a genuine BF16 checkpoint is owed
  (rule 11), all derived artifacts regenerate from it, and every GLM arm
  stays blocked until that closure passes a clean Docker preflight.
  §14 is now the three-record contract (`glm53_tessera_calibration.v1`,
  `tessera_arm_run.v1`, output receipt) with a fail-before-GPU validator
  (P1-3); the ~140 W envelope governs headroom with GPU-board and
  whole-system power as separate series and work per joule from
  integrated energy — an observed maximum is never an envelope (P1-4);
  pass-1 weights index only input `terminal_slot_id`s (P1-5); the
  checksum stub is not a proved bound and stays unauthorized (P1-6);
  the six-step promotion order makes a passing Arm 12 the sole gate for
  pin, menu, architecture, or shipping (P1-7); the P2 contradictions
  cleared. The review cycle is closed; disposition per §14 and §16.
- **rev 7, owner notes (2026-08-31):** the prohibited source is barred
  entirely — its rationale lives in the review records, not in this
  document (round-8 P0-2). The GLM campaign starts from **BF16**; a
  genuine BF16 checkpoint is owed (round-8 P0-2). Power: the **~140 W
  envelope** governs headroom and envelope-fraction; GPU-board and
  whole-system power are separate series, and work per joule uses actual
  integrated energy (round-8 P1-4). **GGUF is no longer supported** as a
  Tessera container. GLM5.3-Flash is the target model.

**Reader's contract:** the claim ledger (§15) is the only load-bearing claims
list. Everything else is specification or context.

---

## 1. Summary

One encode per unit, per branch of a small branch DAG (§7), produces a
typed-plane wire (§9) whose declared **candidate** terminals are
natively-servable weight representations on Blackwell (NVFP4) and — as
native-operand-representable, not yet served — AMD CDNA4 gfx950 (MXFP4);
the direct T-nvfp4-class terminal is conjectural pending the §6b legality
tests. Honest scope from §13: on this decoder generation, batch-1 decode
footprint is **blocked by a strong negative prior** — EN's near-term value
is quality-at-bpp, stored and distribution artifacts, and prefill or
large-batch traffic, not decode-regime footprint.
The format is **not** one stream and not a single forest: it is a stored
branch DAG over (root rate × rotation state × container-target class), with
the A-side handled by one declared multi-lane objective (§7.2). Sub-4-bit
bpp claims attach **only** to the selected serving artifact — never to the
canonical bundle, which stores every branch (§7.4).

The refinement grammar (§6) tops out at 3 bits of payload information per
column plus surgical release; EN never stores a **dense scalar 4-bit
payload plane** — a released position appends a 4-bit override to its
inherited embedded prefix — and the 4-bit-class quality claim rests on
completion + release + the scale planes —
forced by measurement: above the shaped cap the trellis buys +1.85 dB/bit
where a scalar bit buys ~6, and catching scalar NVFP4 needs ~4.2 body bits,
past the wire's structural ceiling of 3.96875 `[branch]`. Segments beyond
the body are the only route to the 3–4.5 band.

The allocator's role is unchanged: the DP selects branch and terminal per
unit from measured surfaces, failing closed on any priced-but-absent branch.

## 2. Problem and requirements

1. Sub-4-bit stored bpp on native 4-bit hardware paths: NVFP4 (E2M1
   payload, per-16 E4M3 scales) on Blackwell including GB10/sm_121; MXFP4
   (E2M1, per-32 E8M0) on AMD CDNA4 gfx950. RDNA4 has no FP4 matrix path
   (FP8 WMMA only) and is upconvert-only; Strix Halo gfx1151 has no FP8
   matrix either (IU8/IU4 only) — that tier points at the integer menu
   (`mxfp4_cb_feasibility.md:16,35,169-172` `[in-tree]`).
2. **Hypothesis, unmeasured on GLM5.3-flash:** the 3–4.25 bpp band lacks
   trellis-class encoding. It needs no prior value to motivate measurement;
   it is measured on GLM under the same source precision, calibration
   rows, sequence length, activation contract, per-Linear semantics, and
   production cache behavior as its comparison baseline.
3. Activations at 4 or 8 bits where measured cost permits, priced per unit
   by AQUA — and serving-contract-gated (§11): pricing is not shipping.
4. Ownership: no forked runtime, no off-grid decode kernels.
5. Later: the same architecture at E4M3 payload width (EN8), with
   per-channel scalar DNA instead of MXFP8's po2 blocks.

## 3. Prior art and the open lane

| System | Reconstruction grid | Serving path | Embedded | A-side |
|---|---|---|---|---|
| QTIP (Tseng et al. 2024) | Hadamard-mixed trellis lattice, off-grid | custom decode matmul | no | A16 |
| EXL3 (turboderp) | same lineage; per-channel `su`/`sv` diagonals | custom CUDA; A dtype-checked FP16 (`exl3_gemm.cu:26,132` `[in-tree]`) | per-layer mixed rates, separate encodes | A16 |
| AQLM (Egiazarian et al. 2024) | additive codebook sum, off-grid | custom kernels | no | A16 |
| VPTQ (Li et al. 2025) | residual VQ, off-grid | custom kernels | no | A16 |
| QuaRot / SpinQuant (2024) | on-grid weights, rotated basis | native A16 GEMM | no | A16 serving |
| SmoothQuant (Xiao et al. 2023) | per-channel migration factor — `su` on an A-quantized lane; on-grid | native W8A8 | no | A8 |
| AWQ (Lin et al. 2024) | activation-aware per-channel scales — `su` on an A16 lane | W4A16 dequant kernels | no | A16 |
| MatQuant (Nair et al., arXiv 2502.06786) | integer MSB-nested | dequant kernels | yes (inherited) | A16 |
| MatGPTQ (Kleinegger et al., arXiv 2602.03537) | integer MSB-nested, PTQ | dequant-on-the-fly FP16 kernels, vLLM plugin | yes (inherited) | A16 |
| Any-Precision LLM (Park et al. 2024); D2MoE (Wang et al. 2025) | nested int / MoE | dequant kernels | yes/partial | A16 |
| PrismaQuant CB (shipped) | on-grid E2M1/E4M3 + shared books | gridbook CUTLASS expansion, native MMA | no (discrete rungs) | W4A4/W8A8 allocated |
| PrismaQuant TCQ (gridbook v0.9.1) | on-grid E2M1/E4M3 trellis | gridbook lanes, contract-v12 cells (dense, sm_121, TP=1, R512/R1152) | no (discrete rungs) | W4A4/W8A8 lanes |

The open lane is the conjunction **embedded + native-terminal +
activation-measured + allocator-placed**; each row holds at most two.
Integer matryoshka inherits nesting from the datatype and serves through
dequant; the E2M1-plus-block-scales grid has no inherited nesting, so EN's
embedding is constructed by the encoder (§6, §9). EXL3's `su`/`sv` is prior
art for the diagonal mechanism; EN's use of it is the joint probe-fit inside
a segment structure. MatGPTQ is the closest competitor and the strongest
product validation — and its cross-width objective is borrowed in §10.

## 4. The reachable-set closure

Per weight position (i,j), one block-scaled FP4 GEMM pass contributes
`su[j]·sv[i]·s_b·q·x̃_j`, where `s_b` is the block-scale code, `q` an E2M1
code, and `su`/`sv` per-channel diagonals applied outside the MMA (`su`
composed into the A-side transform, `sv` into the epilogue). The
W-reachable set per position per pass is `{su[j]·sv[i]·s_b·q}` — n+k
diagonal DOF at ~0.01–0.03 bpp (measured value: +0.87 dB for +0.023 bpp at
3.0 body bits, ≈37 dB/bit, against ≈17.6 dB/bit for a po2→E4M3 block plane
and ≈5 dB/bit body rate `[branch]`). The K-accumulator adds across
positions; every scale-like factor multiplies; nothing adds. Consequences:

1. Single-pass reach is the multiplicative set; redundant scale-plane bits
   are storage, not code dimension.
2. k-fold Minkowski sums (accumulated passes) are the only mechanism for
   per-position values off the per-position multiplicative grids.
3. Five doors: selection within the enlarged set; summed passes; dense
   source transforms; A-side policy co-design; rank-1 channel rescaling
   outside the GEMM. No sixth door is known; round 2 proved assumption (c)
   of the original one-scalar framing false, and no review since has found
   multiplicative DOF beyond per-channel diagonals.

Not proven: encoder optimality within the set; the A-side's own reachable
set; block-level joint structure (door 1's richness).

## 5. Segments

| Segment | Content | Role |
|---|---|---|
| 0 | TCQ trellis body; root rate r₀ ∈ {1.0, 1.5, 2.0, 2.5, 3.0} (`q256` 256–768), per-column integer rates R ∈ {1,2,3} under an exact quota | the shaped base; coding gain is the extraction of structure (whiteness evidence, §15) |
| C | alphabet-completion bits, ≤ (3−R) per column (§6) | grow each anchor's reachable set toward joint-16 |
| 2a | rank-1 channel diagonals `su`/`sv` | cheapest bits measured; vendor-neutral |
| 2b | scale-mantissa plane (po2/32 → E4M3/16), fitted along the clip path | Blackwell-only top-up; clip-point tracking |
| B | constraint-release bits at selected positions (§6) | escape anchor and sequence constraints surgically |

There is no low-rank segment: the post-trellis additive residual is white
in the unrotated and two-sided-measurement rotation states, measured across
q256 {512, 768, 896} (rank-4 energy
0.73–0.91% `[branch]`), and the low-rank compensation literature is conditioned
on RTN-class residuals this encoder does not produce. A second coded residual
pass (former 3b) and dual-accumulate serving are **removed from the normative
grammar** (Codex round-5 P1-5): they were unspecified here, their decode-regime
route is blocked by the current negative prior and separately unspecified
(§13), and their prefill-only value belongs to a
separately gated future proposal, not to a single-authority specification.

## 6. Refinement grammar (decoder-level, exact)

**Objects.** A root r₀ fixes a per-column schedule over integer rates
R ∈ {1,2,3} (Bresenham exact quota; importance-placed arrangements legal if
every complete superblock keeps the quota). Each column's base alphabet
A_R has |A_R| = 2^(R+1) exhaustively optimized codes, stored in the
alphabet blob (a charged plane).

**Stage C — alphabet completion.** A stored descendant map per alphabet:
each anchor a ∈ A_R maps to descendants D(a) with |D(a)| = 2^c for a
completion level c ≤ 3−R. At c = 3−R the descendant sets **partition** the
16-code grid: every code is a descendant of exactly one anchor. Per-column
cost: c bits. Per-position reachable set after completion: D(anchor),
size 2^c.

- R=2, c=1: partner bijection; per-position 2 codes; joint 16.
- R=1, c=2: four-way map; per-position 4; joint 16.
- R=3, c=0: per-position 16 already.

**C-full** (c = 3−R on every column) costs R + (3−R) = **3 bits per column
from every root**; the roots differ only in anchor-tree depth, which is the
residual identity of the base constraint. Completion equalizes all roots at
C-full — a q256=384 (R=1/R=2 mix) root and a q256=768 root reach the same
3-bit joint-16 wire with different anchored structure.

**Stage B — constraint release.** At selected positions, replace the
current code with any of 16: **4 bits per released position** (a stored
code; deltas are not worth the complexity), placement free under the
canonical order (§9) or charged if masked. Release is partial by
construction: release-everywhere costs 3 + 4 = 7 bits/column, never
byte-competitive with scalar 4.5 — so **scalar rate-4 is not an EN
endpoint**. Sub-mode "B at r₀=3" unifies as R=3, c=0, release-only.

**The three readings of "full 16-code grid" are now distinct and named:**
(a) joint-16 coverage — C-full, 3 bits/column, per-position restricted to
descendants; (b) per-position-arbitrary at selected positions — C-full plus
release; (c) scalar rate-4 — release-everywhere — excluded. Every "16-grid"
claim in this document means (a) unless it says (b).

**Terminal payload rates** (information; scale planes added in §8):

| Terminal | Payload bits/column | Notes |
|---|---|---|
| T-po2 | r₀ + ε_C (ε_C ≤ 3−R per column) | ~1.25 bpp projected floor at r₀=1.0 (body 1.0 + 0.25 E8M0/32 + side; exact-byte-gated; the legacy two-tier plane's floor is re-derived, not quoted — round-8 P0-1.3) `[projected]` |
| T-C3 | 3.0 (C-full) | joint-16 anchored, from any root |
| T-nvfp4-class | 3.0 + 4·ε_B | C-full + release + 2b scales; Blackwell — **conjectural until the §6b codec's round-trip tests land** |

EN never stores a dense scalar 4-bit payload plane — a released position
appends a 4-bit override to its inherited embedded prefix. A materialized
compressed-tensors artifact pads the ≤3-bit information into legal 4-bit
nibbles (any code is legal), so CT bytes are 4.5-class while carrying EN
information — that is the derived-artifact story of §12, stated as such.
The old "payload 4.0 / ≤4.5 T-nvfp4" endpoint is superseded by this table.

**Owed before any terminal or traffic number is quoted (build item 1):**
tiny exhaustive tests per R ∈ {1,2,3} × c ∈ {0..3−R} proving nesting (every
completion prefix is a valid partial map), unique decode, legal truncation
at quota boundaries, and code cardinality at every prefix; then the
terminal and traffic tables re-derived through the exact-byte authority.
"Provisional arithmetic" is admissible for headers and padding, not for
body bits (Codex P0-1, accepted in full).

**The sub-3 rungs are currently unmeasurable in-tree:** the alphabet
campaign refuses shaped rates below 3
(`ValueError: the exact native-byte frontier left the canonical rate-3/4
campaign domain; shaped rates=[1, 2]` — `arm_e_quality_campaign.py`, on the
integration branch; symbol not present in this checkout, verified
`[branch]`). `canonical_highrate_alphabets` accepts shaped rates exactly
`{3}`. The reviewed rate-2 fixture `(15, 13, 11, 9, 8, 2, 4, 7)` — verified
in-tree at `gridbook/tests/test_trellis_wire.py:147` — sits at indices
0,2,4,6,8,10,12,15 of the value-ordered alphabet: stride-2 Ungerboeck
partitioning with the final slot snapped. One fixture is not a convention;
defining the rate-1/rate-2 convention is build item 2, and it gates the
sub-3 ladder, the matched-bpw fusing trade, and low-rate rotation behavior.

### 6b. Scale-mantissa codec (segment 2b)

**Candidate construction (Codex round-6 P1-2 obligations pending).** Per
32-weight group: one E8M0 base byte — stored-byte value `2^(E−127)`, bias
127 — plus, per 16-weight half, a 4-bit refinement word: one exponent-delta
bit `d ∈ {0,1}` and three mantissa bits `m ∈ [0,8)`, giving the half's
E4M3 scale as `2^(E−127+d) · (1 + m/8)` where the expanded byte is legal
E4M3FN. Total: 8 + 2×4 = 16 bits per 32 weights = **0.5 bpp — wire-rate
parity with E4M3/16** (not representational parity: both halves share one
base and `d ≤ 1`, so the two half-exponents lie within one octave and
**arbitrary legal E4M3 pairs are not representable**; equal-exponent pairs
carry duplicate encodings — `E,d=0` versus `E−1,d=1` — until a
canonicalization rule picks one). The coverage and quality cost of that
restriction is measurable and must be measured (arm 5 compares against two
**unrestricted** E4M3/16 bytes, not merely against legality of emitted
values).

Prefix semantics: refinement words apply to halves in canonical block
order, so a partial plane leaves earlier halves refined and later halves at
the po2 base. The terminal clip exponent composes multiplicatively — one
per-terminal scalar in the manifest (§9).

**Reuse, not parallel mechanism:** the shipped
[`two-tier-scale-spec.md`](../lanes/nvfp4-cb/two-tier-scale-spec.md)
defines E8M0 bias, a fixed E4M3-exact multiplier table, a 256×16 legality
mask, subnormal and range-end rules, deterministic zero handling, and
bit-exact composition `[in-tree]`. EN's codec is a per-32/two-subscale
**parameterization of that abstraction**, or documents why it differs.

**Test obligations (build item 1b):** the legality predicate — a
base/refinement tuple is legal **iff its exact real composition
round-trips bit-for-bit to a positive finite E4M3FN byte** under the
declared global/clip composition; **exact positive subnormals are legal**
(round-7 P1-2 — the previous reject-list wording would have rejected
them, contradicting the reused two-tier abstraction); zero, NaN,
overflow, and inexact underflow are illegal and fail closed (`0x7F`/`0xFF`
banned, `gridbook/trellis.py:57` `[in-tree]`). The 65,536-word
classification freezes the legal-set digest and separately tests normal,
subnormal, maximum-finite, duplicate/canonical, and invalid classes;
canonical encode/decode round-trips are proven; expanded E4M3 bytes are
compared bit-for-bit with the packer. The GLM scale arm reports the
subnormal population and its quality contribution.
**Until those tests and the matched-quality arm pass, segment 2b and the
non-requantized T-nvfp4-class terminal remain conjectural, not legal by
construction.** T-nvfp4-RQ remains coherent as an openly re-quantized
derived artifact with its own gate.

## 7. Branch DAG and artifact model

**Branch identity** (immutable, in the manifest): (unit, root r₀, rotation
state ρ ∈ {none, `R_in`-only} **for serving**, container-target class γ ∈
{CT-legal, gridbook, requant-derived}). `R_in`-only is the only
algebraically local rotation state under the current output contract;
**two-sided is a weight-space measurement state only** — its output basis
requires an `R_out^T` inverse or proved propagation through every consumer
(residual additions, norms, fused siblings, routed experts), which is a
model-level contract, not a per-unit branch (Codex round-5 P1-4, accepted).
Rotation changes the source tensor,
so base bytes are not shared across ρ; container-target classes differ in
diagonal legality (§12). All candidates the DP may select must be
physically present; the solver **fails closed** on a priced-but-absent
branch. Per-group rotation constraints for fused siblings and packed
experts couple branch choices within the group.

**A-side policy is not a stored branch.** One declared multi-lane objective
— the joint `su` fit under the campaign's expected lane mix — with per-lane
regret against lane-specific encodes measured (§14, arms 8a/8b). If the
regret exceeds the declared numeric threshold, the lane dimension becomes a
branch. Only **named runtime-contract lanes** may appear in the expected
mix or the DP; offline diagnostic scores are recorded but do not
participate in branch decisions.

**Provenance binding (round-7 P1-3 — the profile is input-only):** the
content-addressed `encoder_profile_id` declares a finite ordered set of
**terminal slots** — class, allowed planes, target budget or deterministic
λ/quota policy, clip policy, `w_p`, and all source/calibration and
algorithm/table versions — nothing that only the encode can produce. Each
slot deterministically yields one concrete `terminal_id` record (realized
count/offset planes, clip code, payload references, exact bytes, exact
bpp); the post-encode **artifact receipt** carries the profile ID, the
slot-to-terminal mapping, concrete terminal IDs, and payload digests.
Replay starts from the input profile; cache and runtime lookup use the
output receipt. Pass 1 scores declared slot targets; the optional rerun
may score realized terminals. Initialization, iteration limit, and
tie-breaks are frozen so the output is reproducible from the input profile
alone. Cache keys include the producer profile and payload digests; bytes
without a matching profile are not reused.

**Shared bytes:** none by default; side planes and diagonals are re-fit per
branch. Sharing is permitted only where measured byte-identical.

**Four byte quantities, named and charged separately (Codex P0-2):**

1. **Canonical-bundle bpp** — all branches of a unit. Not sub-4; never
   quoted as a checkpoint rate.
2. **Selected-prefix bpp** — the serving artifact the DP emits. The only
   quantity the sub-4 claim ever attaches to.
3. **Encoded-resident bpp** — wire bytes resident in the gridbook lane.
4. **Expanded-resident bpp** — materialized tiles: ~4.5 single-pass (a
   two-pass figure would belong to the removed future proposal, not this
   grammar).

The artifact model is three layers: the **canonical EN bundle** (source of
record), **derived serving artifacts** (CT: materialized byte-legal NVFP4 —
derived, not a truncation of a serving artifact; **GGUF is no longer
supported**, owner directive 2026-08-31), and the **gridbook lane** (the
only consumer of EN
bytes; resident set = the selected prefix). "Three containers as truncation
profiles of one artifact" was rev-0 language and is superseded.

## 8. Scalar DNA and scale structure

Three planes, priority order:

1. **po2 per-32 base** — the portable representable denominator (CDNA4's
   native contract; Blackwell materializes it as duplicated E4M3 po2 per
   16). Current wire accounting is **body + 0.500 flat** (group-16 E4M3 —
   re-derived in the pure calculator artifact before quotation, round-8
   P0-1.3); EN's po2/32 base projects
   body + 0.25, projecting a ~1.25 floor at the r₀=1.0 root before exact
   side overhead `[projected]`; the legacy two-tier plane's floor belongs
   to a different codec and is re-derived, not quoted (round-8 P0-1.3).
2. **Rank-1 diagonals `su`/`sv`** — segment 2a, vendor-neutral, the
   best-value bits measured (§4). `sv` folds into block scales (E4M3 range
   permitting; clipping charged); `su` does not fold and carries the
   SmoothQuant double duty (§11). Rotation and the fine block plane are
   **substitutes** — rotation is −0.51 dB with the fine plane and +4.23 dB
   without it `[branch]` — so on EN's coarser po2/32 base, rotation's
   W-side value sits between the flat and fine arms (bracketed, unmeasured).
   Rank-1 makes coarsening viable: fuse-16 + rank-1 is −0.80 dB while
   saving 0.445 bpw; net ≈ +1.4 dB at matched bpp as an indication (the
   5 dB/bit figure is recorded, not re-measured) `[branch]`. Block-32
   coarsening nets ≈ +0.7 dB under exact E4M3 — an upper bound, since real
   MXFP4's E8M0 po2 plane measured +13.8% worse output MSE in this repo
   `[branch]`.
3. **E4M3/16 scale-mantissa plane** — segment 2b, Blackwell-only, ≤0.25
   bpp, dual-purpose: radial precision and clip-point tracking. The clip
   optimum is a clean unimodal function (4.0σ 13.65 / 3.0σ 14.67 / **2.5σ
   14.92** / 2.25σ 14.77 / 2.0σ 14.12 / 1.75σ 12.98 / 1.5σ 11.36 `[branch]`)
   — JSO's `joint_mse` at the global scale — so the path-following fit
   walks a well-defined curve. Block/Hadamard alignment never helps
   (aligned cells never best; monotone damage in Hadamard size) `[branch]`.

Semantically coarser scales (64/128) remain allocator-selectable for cellar
units; the default defense of scale bits is entropy coding, charged with
restart/offset accounting for GPU segment-local random access.

## 9. Wire contract: typed planes and legal truncations

The wire is **separate typed planes** — body, completion, release,
diagonals, scale-mantissa — each with a plane-specific canonical placement
order that is a deterministic function of already-decoded base bytes
(position-level planes: descending decoded |value| within the superblock,
positional tie-break; the block-level scale plane: canonical block order).
Canonical placement removes the per-position mask, **not the per-plane
counts**: the manifest carries each plane's per-superblock count vector and
offset/restart table, stored and charged by the exact-byte authority.
(Codex round-5 P0-1, accepted: an interleaved single event stream ordered
by a positional key is not uniquely decodable — the key identifies a
position, not an event's type, width, family count, or restart offset, and
a block-level scale event has no positional key at all.)

**Terminal classes and terminal IDs (Codex round-6 P0-1).** {T-po2, T-C3,
T-nvfp4-class} are terminal **classes** — templates with free ε parameters,
not finite quota vectors. Every actual encoder or allocator candidate
carries a stable `terminal_id` whose manifest record contains the complete
per-plane count arrays, the clip-scalar code, the exact physical bytes, and
the exact bpp. The encoder's Σ_p w_p·D_p weights, the DP candidate rows,
the clip exponents, and the decoder's selected counts are all **indexed by
that ID** — two encodes with different count arrays are different
terminals, whatever their class. Per-superblock quota-boundary truncations
within a plane are legal and enumerate their own `terminal_id`s;
arbitrary interleaved byte-prefixes are not terminals. Pass-1 weights and
anticipated-completion targets are indexed **only by the stable input
`terminal_slot_id`**; the optional rerun may use realized `terminal_id`s;
the artifact receipt maps slot to terminal — no input profile or first
pass requires an output ID (round-8 P1-5). `TESSERA_E2M1_R{q256}`
is a **human-readable family/root descriptor only** (round-7 P1-7): the
persisted assignment carries **one normative representation** — the
structured record of schema, `encoder_profile_id`, `terminal_id`, branch
identity, and payload digest, with a digest computed over that record as
its hash domain (round-8 P1-5: no "or" alternative) — parsed by a
**disjoint Tessera record type**
that fails closed where a concrete terminal record is required. Legacy
`TCQ_*` names and their parser stay immutable; the legacy two-tuple parser
never silently accepts a Tessera artifact.

**Plane descriptors (one per plane, not one universal shape).** Each plane
declares its index domain (position, half-block, tensor-axis), count
granularity, integer widths, endianness, alignment and padding, offset and
restart encoding, and payload dtype. Count and offset arrays are **resident
binary side planes referenced by the manifest**, not host-parsed textual
arrays on the forward path. The alphabet and descendant-map blobs carry
byte layouts or content-addressed references. Marginal-value ordering
survives as **encoder-side budgeting only**: the λ-greedy pass chooses the
per-plane quota vectors, and the decoder never reconstructs that choice —
it reads the stored vectors.

**Clip exponents:** one per declared terminal, 2–3 bits in the manifest,
derived at encode time from that terminal's decoded state and composing
multiplicatively with the scale plane (§6b). Internal non-terminal
truncations carry no exponent, and their clip suboptimality is bounded by
the path-following fit plus measured drift (arm 10).

**Owed before the wire ungates** (the round-5/6 exit gate, restructured to
remove the circularity Codex round-6 found): build item **1a** — a reviewed
byte-level schema and parse algorithm; item **1b** — pure
serializer/parser/footprint code plus exhaustive tests that start **only
from serialized bytes** and prove unique parse, unique decode, truncation
at every declared quota boundary, and exact agreement between physical
bytes and the footprint accountant. The `prismaquant.tessera.v1` schema
moves **ahead of 1b** (formerly item 5). No menu, pipeline, or
shipping-code wiring precedes 1b passing.

## 10. Encoder

**Labeled honestly: an alternating heuristic, not an exact multi-prefix
Viterbi.** No additive recurrence is known — the canonical-order statistics
couple emissions across the trellis state, so a standard Viterbi cannot
optimize the prefix-weighted objective by swapping its metric (Codex P1-5,
accepted). The passes:

1. Base Viterbi with an **anticipated-completion metric**: anchor choices
   scored at the intended post-completion values, with terminal-weighted
   terms Σ_p w_p·D_p over the declared terminal set (MatGPTQ's cross-width
   objective, transplanted).
2. λ-greedy completion/release/scale placement under the canonical order.
3. At most one re-run of (1) given (2)'s placement.

Weights w_p come from the campaign's expected truncation distribution. The
encoder-side `su` fit is the joint SmoothQuant objective against captured
post-QDQ activations (the §11 metric), with an A-quantizer pre-inversion
term `[conjecture]`. Currency discipline is unchanged: anchors are
SSE-under-per-input-channel-second-moment; the per-Linear κ bridge to AURA
is licensed per-Linear only, unrotated-only until measured in the rotated
basis; no objective mixing. **The allocation story presupposes no
convexity** — the observed surfaces are explicitly non-convex
(`allocation_regret`'s own disclaimer), so the production gate is the real
DP's end-to-end selection plus held-out KL; measured convexity, if it ever
appears on GLM, only improves greedy placement efficiency and is never
assumed. Rotation is serving-space only; BlockLDLQ does
**not** rescue rotated weighting — `B == D`, `C == A` exactly, because the
LDL block equals the superblock and the dense structure lands inside `D_j`
where the 256-state Viterbi cannot reach `[branch]` (the rev-2 claim that
the transformed diagonal factors recover the +1.26 dB is **retracted**).
Recovery research — LDL block larger than the superblock, or a dense-`D_j`
Viterbi — is gated at ≥1 dB of the 1.95–2.29 dB rotation loss.

**Owed:** brute-force validation on tiny cases (the §6 exhaustive tests
double as encoder checks) and profiling at representative shapes before
any GPU-bound claim at 33k-expert scale.

## 11. Activation contract

The A-side is a property of the lane, priced per (unit, format) by
`aqua_activation_cost` reading the lane's `served_activation_quantization.executes`
globs. It is **pricing-only today**: the native routes are W4A4 and W8A8;
mixed FP4×FP8 and FP4×BF16 are refused; BF16 expansion is a fallback. Until
W4A8 and W4A16-bridge exist as named route variants with runtime-contract
cells, the DP's A-side column is advisory. `su` multiplies activations
pre-QDQ, so the diagonal fit and the A-side policy are one optimization
(the joint objective of §7.2); a W-only fit can worsen A4 dloss — arm 8 is
mandatory. Rotation is a **gridbook-lane-only lever** (Rᵀx is inexpressible
in CT, and materialization cannot un-rotate without requantizing),
per-group for packed units. Honest placement: sub-4 W-side traffic is the
decode win where a decode route exists (§13); A4 buys prefill, MTP, batch
throughput and joules, nothing at batch-1 decode.

## 12. Terminals and containers

- **Gridbook-CUDA** is the only consumer of EN bytes: fused expansion of
  the selected prefix, diagonals applied around the GEMM, resident set =
  prefix bytes per the truncation manifest. Serving-gated (§13).
- **Residency ownership — two owners, split at the artifact boundary
  (round-7 P1-4).** **Producer phase:** `ProductionWeightCache` (or the
  existing streamed renderer) is extended with a typed Tessera
  render/artifact record — probes, recache, export, exact-byte accounting,
  and validation reuse that one producer cache; required render inputs
  are resident-prefetched and failures close. `ProductionWeightCache`
  does **not** own serving residency — Gridbook is the sole owner of
  serving configuration, loader hooks, CUDA, dispatch, and runtime tests
  (`docs/ARCHITECTURE.md:6286-6293` `[in-tree]`), and compatibility
  crosses only through the immutable pin and packaged contract.
  **Artifact boundary:** content-addressed payload and side tensors plus
  the closed manifest; the pinned Gridbook runtime contract declares and
  validates the ABI and producer profile. **Serving phase:** the external
  Gridbook loader owns device placement, descriptors, residency
  accounting, prefetch, and fail-closed loading — no per-forward host
  parse or NVMe read: **Gridbook must perform neither per-forward host
  parsing nor per-forward NVMe reads**, enforced by the
  external gate. An installed-wheel producer/consumer interop test lands
  with the runtime gate. This contract is a build item **before** any
  Gridbook lane extension.
- **compressed-tensors** branches (GGUF is **no longer supported** — owner
  directive, 2026-08-31; no GGUF container, lane, or export path is a
  Tessera target) are explicitly **unrotated** and
  either diagonal-free, lawfully producer-folded, or **requantized**. `su`
  cannot be represented by CT's contract (one per-tensor
  `input_global_scale`, `export_native_compressed.py:980-988,2395-2413`
  `[in-tree]`) and cannot generally fold into shared producers for routed
  experts; `sv` folds into block scales. The requantized variant is a
  separate derived artifact — **T-nvfp4-RQ** — with its own render, exact
  bytes, probe/KL row, and serving gate. It is not the T-nvfp4-class
  terminal and never carries the diagonal-gain numbers.
- **CDNA4 gfx950**: "native-operand representable" until a released
  consumer contract and served-speed gate exist for EN bytes plus side
  operators — AITER's standard MXFP4 path consumes neither. vLLM-ROCm
  exposes native W4A4/W8A8 there (`mxfp4_cb_feasibility.md:191` `[in-tree]`),
  which is the lane an EN consumer would ride.
- **EXL3 comparisons** must state both activation contracts in the same
  table: EXL3 is W4A16 and pays no activation perturbation; a matched-bpp
  loss on A4-allocated units is non-diagnostic unless the A-side dloss is
  reported separately, and a win at A4 is larger than it looks. No sub-3.25
  W-MSE claim versus EXL3 is made.

## 13. Serving and kernels

Two established routes plus one preliminary decomposed microbenchmark on
the pinned lane (gridbook 0.9.1, `227420f`,
dist-ci wheel `cb4d7ad6…`):

- **Resident expansion** — ~4.5 bpp resident, 59.5 µs forward at 4096²,
  5.20× vs bf16, 164.4 calls/J `[branch]`. Fast, no footprint win.
- **Streamed (unfused)** — 1.10× vs bf16, 4.51× worse work/joule; its
  residency defect was fixed by the decode-scratch pool, leaving per-call
  decode cost `[branch]`.
- **Fused decode-in-mainloop** — measured as a decomposed benchmark
  (Claude's pipelining run, absorbed from the duplicate §21): decode via
  the real hot path is **236.9 µs** per 4096² tile at q256=512; MMA is
  50.6 µs at M=1 (launch-bound, flat 30–50 µs through M=256) to 889.4 at
  M=8192. decode/MMA: 4.68× at M=1, 7.68× at M=64, 0.58× at M=4096;
  **crossover M ≈ 2100–4096**, derivable in advance as M ≈ 2076
  `[branch-preliminary]`.

**The consequence is an inversion, not a threshold.** The traffic win
exists only where compute already dominates (prefill, large batch); in the
bandwidth-bound decode regime the decode compute exceeds the MMA it would
hide behind, and per-K-tile pipelining cannot beat max(Σdecode, ΣMMA)
`[derived]`. **The reopening question is bpp-dependent, not a single number** (Codex
round-5 P1-2, accepted — correcting round-6's framing): the required
decoder speedup is `T_decode / (((4.5 − b)·N·K/8) / B_eff)`, where `b` is
the exact selected-prefix bpp and `B_eff` the effective read bandwidth. At
the copy sensitivity point (214.5 GB/s, 4096², q256=512): b = 2.0
body-only → 24.4 µs saved → 9.69×; b = 2.25 projected body+po2 → 10.77×;
b = 2.5008 current exact wire → **12.12×**; b = 3.75 → 32.3× — the bar
**diverges as b approaches 4.5**, so no single number gates the 3–4.25
band `[derived on measured inputs]`. Round-6's caveat also had the algebra
backwards in the bandwidth-limited regime: if both paths are
bandwidth-limited at the same rate, a lower rate **increases** the saving
and lowers the bar; the bar rises only when launch or compute hides the
removed bytes, shrinking the saving toward zero. Copy bandwidth is a
**sensitivity input, not a gate**. The actual parity quantity is
`T_resident_path − T_fused_skeleton_with_encoded_reads_but_no_decode`,
which needs a measured skeleton on the target route; until it exists, the
decomposed result is a **strong negative prior for decode regimes, not a
precise reopening threshold**.

Therefore: every sub-4.5 **serve-time** footprint claim is blocked by a
strong negative prior in decode regimes on this decoder generation — not
killed by decision-quality evidence, which rule 13 says the microbenchmark
does not yet carry (no in-process profile, no two-box Netdata; promotion
requires the permitted reproduction's full evidence set: exact total
bytes, resident-path baseline, profiler output, interleaved raw logs,
synchronized power series). A second coded pass (former 3b) and
dual-accumulate are out of the normative grammar because they are
unspecified and separately gated — not because this artifact has completed
their retirement gate. The CT derived-artifact story is unaffected —
it never claimed serve-time footprint. The decoder-speedup feasibility
read (is the state walk irreducible, and against which point on the
bpp-dependent curve?) precedes any kernel work.

## 14. Measurement matrix (one table, guideline bar)

**GLM measurement contracts (round-8 P0-2/P1-3; binding).** Three
records, one validator:

1. **`glm53_tessera_calibration.v1`** (immutable base): full 64-hex
   digests for the BF16 checkpoint shard/index closure, model profile
   `glm5_next` and serving profile `vllm_glm5_next_packed_moe`, dataset
   `dq-runs/calibration/diverse-v1.jsonl`, tokenization receipt, probe
   census and payload, activation-cache manifest, checkpoint census,
   quantizable set, and calibration parameters.
2. **`tessera_arm_run.v1`** (one per arm, immutable): references the base
   ID; binds the encoder profile, terminal slots, thresholds, seeds,
   exact code commit and source closure, Docker image digest, hardware
   targets, and commands.
3. **Output receipt**: raw rows, logs, profiler traces, Netdata windows,
   payloads, and the pass/fail decision, bound to the arm-run ID.

A validator runs at every entry point: missing, truncated, mutable,
cross-model, wrong-source-dtype, dirty-code, or wrong-container
identities fail **before GPU allocation**, with a content-addressed
ancestry check — a text denylist alone is insufficient.

**The BF16 source does not exist yet (round-8 P0-2, verified in this
checkout):** the identified checkpoint at `/mnt/shared/models/GLM-5.3-Flash`
is **FP8 on every quantizable parameter**
(`prismaquant/model_profiles/specs/glm5_next.json`: 305,915,756,544
quantizable parameters, 100% F8_E4M3; checkpoint census 263 FP8 / 654
BF16 units), and `tools/run_glm53_stock_harvest.sh:45,60` hardcodes the
absent `/home/rob/models/GLM-5.3-Flash` path. Required before any GLM
arm: **acquire and content-bind a genuine BF16 GLM-5.3-Flash checkpoint**
(rule 11); regenerate the census, probe, activation cache, tokenization
receipt, and every baseline from it — an FP8-to-BF16 expansion is not a
BF16 start; make the driver consume the manifest's checkpoint URI and
digest with a full-closure preflight in the known-good Docker
environment. **Every GLM arm stays blocked until the BF16 closure exists
and a clean preflight succeeds.**

**Claim classes by source dtype:** a low-precision source supports claims
measured **relative to that exact source tensor**; it cannot support
fidelity claims to an unavailable higher-precision model. The GLM
campaign's source-dtype is bound in the base record when the BF16 closure
is authored.

bpp over quantizable parameters only; runtime claims carry in-process
profiler **and synchronized Netdata series on both boxes**, ranked by
**work per joule from actual integrated energy**; headroom and
envelope-fraction analysis uses the **~140 W envelope** (round-8 P1-4:
an observed maximum is not an envelope — measured GPU-board power and
whole-system power are reported as separate series, and a repeatable
~100 W GPU observation may be labeled and cited as an observation with a
receipt, never as the envelope); `gpu_utilization` remains non-diagnostic.
Numeric pass/fail thresholds, commands, log paths, eager/graph/MTP
checks, and downstream PPL/NLL/ToolEval gates per
`design_guidelines.md:104` are owed **before** any non-permitted arm
runs.

**Ledger invariant:** no prohibited-source artifact participates in
Tessera measurement, fitting, allocation, encoding, export, or
validation — in inputs or in citations. Retaining one as a "prior," a
"historical note," or a "removed row" is still use.

**Permitted now — GLM-dependent work launches when the manifest artifact
exists; synthetic work launches now:**

- Arm 1 — A-side dloss sweep (dense + routed stages, with and without
  probe-fit rotation; existing instruments); routed stages report
  separately.
- Arm 5 (scale-penalty sub-arm) — po2 vs E4M3/16 scale-plane dloss; the
  subnormal population and its quality contribution are reported (§6b).
- §18-class weight-space diagnostics at additional existing rungs —
  Qwen3-0.6B tensors now (model-scoped prior), GLM5.3-flash tensors when
  the manifest exists; existing encoder unchanged, unweighted diagnostics,
  with tensor/input digests, q256, rotation sidedness, seeds, commands,
  container identity, raw rows, and logs committed. No partner completion,
  multi-prefix encoding, schema, DP/menu wiring, export, or kernels.
- **Fused-decode reproduction** — independent re-run of the decomposed
  benchmark, existing released decoder and native MMA only: known-good
  container/pin, **exact total wire bytes (2.5008-style, never the 2.0
  body rate)**, in-process profile plus **synchronized Netdata series on
  both boxes**, interleaved repetitions, committed raw logs — a **clean
  synthetic artifact, independent of any calibration identity**. No
  fused-kernel implementation.
- **Stub lower-bound screen — not authorized** (round-8 P1-6: a
  checksum-bound read kernel is not a proved lower bound — checksum work
  is not a necessary operation of every fused decoder, so a constructive
  miss cannot bound the class). **Before authorization:** define a
  mathematical lower bound separate from any constructive benchmark —
  measured best-case device bandwidth, exact bytes, ideal overlap, and
  the target MMA floor — and report it alongside any concrete anti-elision
  kernel, which is evidence for its own construction only. Any kill
  decision is restricted to what the evidence actually bounds. The spec
  also owes: representative GLM shapes and regimes (routed gate/up/down,
  dense/shared MLP, very small per-expert decode M, reported separately)
  and a swept bpp/alignment envelope.

**Gated arms** (run after the fold passes review, speced to guideline bar):

- Arm 2 — embedded-vs-dedicated regret, matched-basis pairs across rotation
  states, placement ablation (canonical vs charged mask), brute-force tiny
  cases. **Its minimal measurement encoder — completion planes only, no
  schema, DP, menu, export, or serving plumbing — is the first gated-work
  request after the final review** (the fail-fast ordering: the
  load-bearing quality claim gets its code before any plumbing does).
  Production gate: the real DP's end-to-end selection under the full
  constraint set plus held-out KL at matched bpp against the shipped
  assignment — `allocation_regret` is a greedy diagnostic, not the gate
  (`trellis_rate_surface.py:801` `[in-tree]`).
- Arm 3 — release-vs-completion at matched bpp in the 3.5–4.0 band.
- Arm 4 — **two named gates** (round-7 P1-6): (a) the stub lower-bound
  screen — pending authorization, spec in the permitted list; (b) the
  **exact full-layout encoded-read skeleton, after 1b and before any
  Tessera reader, decoder, or kernel implementation** —
  completion/release/diagonal reader work cannot precede this gate.
  Dual-accumulate is off the arm list — removed from the grammar (§5).
- Arm 12 — **post-release serving promotion**: exact wheel, pin, and
  contract; bit-exact load; deterministic generation; eager and graph
  execution; MTP where applicable; dense and routed route attestations;
  exact resident bytes; latency and throughput by regime; in-process
  before/after profiles; synchronized Netdata on both boxes; work per
  joule — compared against the exact container displaced. A slower route
  remains research.
- Arm 5 (full race) — {po2/32; +diagonals; +mantissa; E4M3/16; +diagonals}
  × rotation states {none, `R_in`-only, two-sided}; the `su` W-side gain is
  expected to shrink rotated, `sv`'s at least as much (two-sided
  homogenizes rows 2.03× vs columns 1.75× `[branch]`). The 2b arm compares
  the constrained pair codec against two **unrestricted** E4M3/16 bytes
  (§6b) — legality of emitted values alone is not the comparison.
- Arm 8a — joint-A objective vs W-only on A4-allocated units (mandatory).
- Arm 8b — the shared multi-lane encode vs lane-specific optima for every
  declared A4/A8/A16 runtime-contract lane, with a **numeric regret
  threshold and estimator** declared in the arm spec — a "noise floor" is
  not a machine-readable gate (Codex round-6 P1-3).
- Arm 9 — residual-spectrum survey: extend from the measured
  q256 {512,768,896} × {unrotated, two-sided} to q256 {256,384,640} and
  `R_in`-only, plus GLM5.3-flash tensors and a second seed pair; mixed
  low-rate rows remain blocked on the alphabet convention.
- Arm 10 — clip-drift ablation (base-optimal vs terminal-optimal vs
  path-following ± manifest exponent).
- Arm 11 — release/bypass decay slopes, across the rotation states
  {none, `R_in`-only, two-sided-as-measurement}.

Per-expert truncation additionally requires the split-bank completeness
contract (released Gridbook consumers require packed-expert uniformity,
`docs/ARCHITECTURE.md:4278` `[in-tree]`).

## 15. Claim ledger (consolidated; the only load-bearing list)

| Claim | Status | Evidence |
|---|---|---|
| Trellis coding gain exhausts with the alphabet subset (`2^(R+1)`, `max_trellis_rate = native−1`) | measured | `trellis.py:112-113,182` `[in-tree]` |
| Above the shaped cap: +1.85 dB/bit vs ~6 scalar; 1.26 dB behind scalar NVFP4 at 4.0042 bpp; catching it needs ~4.2 body bits > the 3.96875 ceiling | measured | `trellis_shaped_rate_ceiling_2026-08-31.md` `[branch]` |
| v2 decoder landed (5.37×/2.46×; 5.80×/3.18× W/J); 08-29 lane numbers are post-v2 | measured | gridbook `1a57b31` release facts; git ancestry `1a57b31`→`ab80df3`, verified in `/home/rob/gridbook` `[in-tree]` |
| Gridbook v0.9.1 ships contract-v12 cells (dense, sm_121, R512/R1152); campaign worktrees pin it; canonical branch pins 0.8.11 | measured/status | verified in `/home/rob/gridbook` + pin files `[in-tree]` |
| E2M1 W4A4 lane exact on synthetic shapes (8/8, max abs err 0) | measured | `trellis_e2m1_lane_2026-08-29.md` (clean synthetic runtime source) `[in-tree]` |
| Rank-1 diagonals: +0.87 dB / +0.023 bpp (W-side reconstruction only; no activation model — the joint A-side optimum is arm 8); ≈37 vs ≈17.6 vs ≈5 dB/bit | measured | `rank1_scale_structure_2026-08-31.md` `[branch]` |
| Rotation/plane substitutability: −0.51 dB with fine plane, +4.23 without; bypass-fraction damage −0.27/−0.77/−0.72 | measured | `qtip_rotation_weight_side_2026-08-31.md` `[branch]` |
| Clip curve unimodal, 2.5σ optimum; block/Hadamard alignment never best; block-32 net ≈ +0.7 dB (E4M3-exact; E8M0 +13.8% prior) | measured | `rank1_scale_structure` `[branch]` |
| Fused decode: 236.9 µs real hot path; 4.68× at M=1, 7.68× at M=64, 0.58× at M=4096; crossover M≈2100; reopening bar bpp-dependent — 9.69× at 2.0 body-only, 12.12× at the 2.5008 exact wire, 32.3× at 3.75, diverging as b→4.5 | branch-preliminary | `trellis_fused_decode_pipelining_2026-08-31.md` — present and content-verified on the fetched branch tip `bd5de5c`; a measured microbenchmark, **not decision-quality under rule 13** (no in-process profile, no two-box Netdata) — promotion requires the permitted reproduction (§14) |
| Small-M fused is decode-bound regardless of pipelining (max(Σdecode, ΣMMA)) | derived | §13 |
| Sub-4.5 serve-time footprint in decode regimes: blocked by a strong negative prior on this decoder generation — not killed; promotion is gated on the permitted reproduction's rule-13 evidence set | status | §13; rides the `[branch-preliminary]` fused-decode row above |
| BlockLDLQ recovers nothing under rotation (B==D, C==A; LDL block = superblock; dense `D_j` unreachable) | measured | WO-F branch, verified present |
| Rev-2 claim "transformed diagonal factors recover the +1.26 dB" | retracted | §10 |
| Two-sided rotation homogenizes rows 2.03×, columns 1.75× | measured | `rotation_sidedness_and_residual_spectrum_2026-08-31.md` (branch tip `7670867`, ls-remote verified) |
| Post-trellis residual white — measured at q256 {512,768,896} × {unrotated, two-sided-measurement} (six cells; s₁/s₂ ∈ [1.006, 1.163]; rank-4 energy 0.73–0.91%) | measured | `rotation_sidedness_and_residual_spectrum_2026-08-31.md` + rung replication at `bd5de5c`; remaining scope: q256 {256,384,640}, `R_in`-only, GLM5.3-flash tensors, a second seed pair |
| Per-position weighting worth +1.26 dB at 3.0 bits (+0.60 dB at 3.5) | measured | `qtip_rotation_weight_side_2026-08-31.md:20` (round-7 correction: this row's P2-7 downgrade was factually wrong — the number was committed all along) |
| Alphabet campaign refuses shaped rates {1,2} (exact ValueError); rate-2 fixture verified in-tree at `gridbook/tests/test_trellis_wire.py:147` | measured | `arm_e_quality_campaign.py` `[branch]`; fixture `[in-tree]` |
| Native MXFP4 MMA: sm_100/120/121 + CDNA4 gfx950 only; RDNA4 FP8-WMMA-only; Strix Halo no FP8 matrix | measured | `mxfp4_cb_feasibility.md:16,35,169-172` `[in-tree]` |
| A-side separable pricing; A-term dominates post-render; per-Linear κ licensed, global refused | measured | `aqua_activation_cost.py`; currency campaign `[in-tree]` |
| `allocation_regret` is a diagnostic, not a gate — observed surfaces are explicitly non-convex | status | `trellis_rate_surface.py:801` `[in-tree]` |
| Mantissa/partner algebra; completion capacities (3−R); four byte quantities | arithmetic | §6, §7 |
| Refinement planes interpolate quality; path-following ≈ per-prefix refits; joint `su` retains the gain; `R_in`-only preserves `sv` at half cost; optimal release decay | conjecture | arms 2, 10, 8, 5, 11 |
| MatQuant/MatGPTQ integer nesting, dequant serving | literature | arXiv 2502.06786, 2602.03537 |
| EXL3 is W4A16 and ships `su`/`sv` channel diagonals around its GEMM | measured | `/home/rob/dq-runs/exl3-reference-20260830`: `exl3_gemm.cu:26,132` (A float16; `TORCH_CHECK_DTYPE(A, kHalf)`); `quantize.py:1208-1212` (`in_channel_scales`/`block_rms`, `weight /= su`); `exl3_gemm_kernel.cuh:25,66` (`suh`/`svh`) `[in-tree]` |
| No published float-grid embedded format with native terminals | literature-gap | search-scoped (one structured search, 2026-08-31) |

## 16. Build order

Items 1–3 are reversible (schema, tests, definitions, cache abstraction);
**item 4 is the first irreversible step** — it touches shipping code, and
its `docs/ARCHITECTURE.md` carry-forward travels **in the same commit**
(rule 12: ARCHITECTURE.md carries pin-version prose at lines 8, 322, 329,
337, 352, and 367 `[in-tree]` — the list is illustrative; re-grep the pin
version at bump time — so a pin bump without the update is a violation by
the letter; Kimi K3 D1 and F2).

**Ordering principle (owner directive, 2026-08-31): the cheapest decisive
measurements run before the work they would invalidate.** The performance
track (item 0) runs first and in parallel with the wire-spec track; arm 2's
minimal measurement encoder is the first gated ask, ahead of DP, menu,
export, or serving plumbing.

0. **Fail-first performance track:** the fused-decode reproduction (§14,
   permitted, synthetic); the **stub lower-bound screen** (§14 — pending
   explicit authorization and its anti-elision/shape/sweep spec; an
   optimistic necessary-condition screen, not a verdict — a
   proved-optimistic miss may kill a shape and regime, a pass validates
   nothing); and the GLM5.3-flash campaign — corpus, arm 1, the
   scale-penalty sub-arm, whiteness scope closure — the quality-side
   fail-fast, **launching when the GLM manifest artifact exists** (§14).
   The full-layout encoded-read skeleton is a separate post-1b gate (arm
   4b), before any Tessera reader, decoder, or kernel.

1. **1a — reviewed byte-level schema and parse algorithm**
   (`prismaquant.tessera.v1`, moved ahead of the tests that require it):
   plane descriptors, terminal-ID manifest records, and
   `encoder_profile_id` provenance (§7, §9). **1b — pure
   serializer/parser/footprint code plus the exhaustive
   serialized-bytes-only tests**: §6's grammar (nesting, unique decode,
   legal truncation, cardinality at every prefix), §9's parse/footprint
   agreement, and §6b's scale-codec classification of all 65,536
   base/refinement words with fail-closed rejection, canonical
   round-trips, and packer bit-exactness. No menu, pipeline, or
   shipping-code wiring precedes 1b passing (Codex round-6 P0-1).
2. Define the rate-1/rate-2 set-partitioning alphabet convention (fixture
   at `gridbook/tests/test_trellis_wire.py:147` is the starting point);
   unblocks the sub-3 ladder, the matched-bpw trade, and low-rate rotation.
3. **Producer cache and artifact handoff (§12)** — the
   `ProductionWeightCache` Tessera render/artifact record, content-
   addressed payload and side tensors, and the closed manifest at the
   artifact boundary. Reversible; precedes any lane work (round-7 P1-4).
4. **Legacy TCQ v0.9.1 pin/menu integration** (round-7 P1-5): land the pin
   bump and wire the eight `UNWIRED_LINKS` (`trellis_menu.py:124`
   `[in-tree]`) — the legacy `TCQ_*` allocation seam, independently
   useful; it neither creates Tessera bytes nor gives GLM a consumer. The
   `docs/ARCHITECTURE.md` pin-prose update and provenance re-stamp travel
   **in the same commit**. **First irreversible step.** A separate
   **external Gridbook Tessera release gate** follows: its packaged
   contract must declare `prismaquant.tessera.v1`, the root and terminal
   domain, plane layouts, descriptor ABI, residency modes, producer
   profiles (including `glm5_next`), and device-qualified route cells;
   PrismaQuant's pins advance only after that independent release, exact
   wheel digest, producer/consumer ABI test, and profile/cell checks — no
   fork, dirty checkout, or local alias. Until a routed-MoE cell exists,
   routed GLM units are absent from the Tessera DP and GLM experiments are
   explicitly dense-only.
5. Permitted measurements (§14) run throughout.
6. Guideline-bar arm specs; then gated arms in §14's order.
7. The **full-layout encoded-read skeleton** (arm 4b) precedes any
   Tessera reader, decoder, or kernel work; then the Gridbook lane
   extension (completion/release planes, diagonals) riding the item-3
   producer/artifact contract. Dual-accumulate is a future proposal, not
   on this order.
8. Split-bank completeness before per-expert truncation; W4A8/W4A16-bridge
   cells before any A-side shipping claim; T-nvfp4-RQ render and gate if
   the requant lane is wanted.
9. EN8 twin after the 4-bit ladder is anchored.
10. Every landing commit carries its own `docs/ARCHITECTURE.md` update and
    provenance re-stamp (rule 12) — there is no deferred documentation
    step.
11. A Tessera-document/manifest **CI check** rejects known
    prohibited-source identities and citations, with the content-addressed
    ancestry check (round-8 P0-1.6); the legacy-plane wire arithmetic is
    re-derived in a pure calculator/test artifact, and decoder/lane facts
    are re-published from clean synthetic Gridbook artifacts.

**Tessera promotion order (round-8 P1-7; one executable sequence):**
(1) the legacy TCQ 0.9.1 integration stays independent and never
advertises Tessera; (2) 1a/1b pass and the full-layout skeleton passes;
(3) the external Gridbook Tessera reader/kernels are implemented,
including a `glm5_next` routed-MoE cell — the GLM profile attributes
304.41 B quantizable parameters to routed experts, so a dense-only route
cannot support a representative GLM shipping claim; (4) Gridbook produces
a pinned release-candidate wheel and packaged contract; (5) Arm 12 runs
against that exact wheel and the exact displaced container; (6) only a
passing Arm 12 permits the PrismaQuant Tessera pin, menu admission,
architecture update, or shipping status. Dense-only results are reported
as dense-only research and never promote Tessera for the GLM target.

## 17. Attack surface for the next review (of this folded text)

1. The §6 grammar derivation — especially the partition claim at c = 3−R,
   the C-full equalization of roots, and the release accounting (4 bits,
   placement-free under canonical order).
2. The branch DAG's A-side resolution — one multi-lane objective vs a
   stored branch dimension; the regret measurement that decides it.
3. The bpp-dependent feasibility function and its copy-bandwidth
   sensitivity — the parity skeleton is the measurement that turns the
   negative prior into a threshold.
4. The 1a/1b restructure — the reviewed schema and parse algorithm, the
   serialized-bytes-only tests, and the §6b legality contract (two-tier
   parameterization, 65,536-word classification, canonical round-trips,
   packer bit-exactness, and the arm-5 comparison against unrestricted
   E4M3/16 pairs).
5. The heuristic encoder's declared prefix set and weights.
6. The whiteness evidence's remaining scope, now exact: measured at
   q256 {512,768,896} × {unrotated, two-sided-measurement}; owed are
   q256 {256,384,640}, `R_in`-only, GLM5.3-flash tensors, and a second
   seed pair (mixed low-rate rows blocked on the alphabet convention).
   Segment 3a's gate rests on this scope closing.
7. Whether the permitted-measurement boundary (§14) is drawn correctly.

## 18. References

- MatQuant: Nair, Datta, Dean, Jain, Kusupati. *Matryoshka Quantization.*
  ICML 2025; arXiv 2502.06786. MatGPTQ: Kleinegger, Crnčević, Alistarh,
  arXiv 2602.03537 (2026-02); code IST-DASLab/MatGPTQ.
- Any-Precision LLM: Park et al., 2024. D2MoE: Wang et al., 2025. QTIP:
  Tseng et al., 2024. EXL3: turboderp (GitHub); `su`/`sv` and A-dtype
  verified in the local reference checkout
  `/home/rob/dq-runs/exl3-reference-20260830` — `exl3_gemm.cu:26,132`,
  `quantize.py:1208-1212`, `exl3_gemm_kernel.cuh:25,66`. AQLM: Egiazarian
  et al., 2024. VPTQ: Li et al., 2025. QuaRot/SpinQuant: 2024.
  SmoothQuant: Xiao et al., 2023. AWQ: Lin et al., 2024. TCQ: Marcellin &
  Fischer, 1990. ZeroQuant-V2 / LQ-LoRA / CALDERA: the low-rank
  error-compensation line, name-level (read-level check still owed).
- Review records (every absorbed round has a resolvable pointer — Kimi K3
  D2): `docs/handovers/claude-findings-for-glm-2026-08-31.md` (Claude
  rounds 1–6);
  `docs/design/embedded_native_weight_coding_2026-08-31_codex_round4_review.md`
  (Codex rounds 3–4);
  `docs/design/embedded_native_weight_coding_2026-08-31_codex_round5_review.md`
  (Codex round 5);
  `docs/design/embedded_native_weight_coding_2026-08-31_codex_round6_review.md`
  (Codex round 6);
  `docs/design/embedded_native_weight_coding_2026-08-31_codex_round7_review.md`
  (Codex round 7);
  `docs/design/embedded_native_weight_coding_2026-08-31_codex_round8_review.md`
  (Codex round 8, final);
  `docs/handovers/claude-round7-review-2026-08-31.md` (Claude round 7).
  The Kimi K3 review was relayed in-session on 2026-08-31 and has no
  repository file; its two procedural defects are recorded in the revision
  history. Branch artifacts:
  `origin/claude/prismabuild-trellis-integration-20260831` and
  `muse/wo-f-trellis-ldlq-20260831`.
- Repo anchors cited inline by path; campaign artifacts under `dq-runs/`
  and worktrees are dated and immutable as cited.
