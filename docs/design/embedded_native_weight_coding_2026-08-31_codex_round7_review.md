# Codex round-7 review: Tessera and fail-first amendments

**Date:** 2026-08-31

**Reviewed document:** `embedded_native_weight_coding_2026-08-31.md`, SHA-256
`f075ba3a6059096e710298c4b2336b86020fe97cc5f16f8a15ba6df88a4cee60`,
885 lines.

**Evidence checked:** the PrismaQuant checkout; the fetched integration branch
`origin/claude/prismabuild-trellis-integration-20260831` at
`bd5de5c8da1be0dfffb8560e7d5463eaae0ac806`; Gridbook release commit
`227420f9821bab7089632ee914f0ba050f82b817` and packaged
`gridbook.runtime-contract.v12`; the existing GLM-5.3-Flash census and driver
under `/home/rob/dq-runs/glm53-flash` and
`tools/run_glm53_stock_harvest.sh`; the local two-tier scale specification; and
the current producer/runtime ownership rules in `docs/ARCHITECTURE.md`.

**Binding scope directive from the user:** **DSV4-Flash is not to be used
anywhere for any reason.** No DSV4 model, calibration data, tokenization,
activation cache, probe, cost table, assignment, tensor sample, corpus result,
derived identity, or mixed-source handover may be an input to Tessera or a
load-bearing Tessera citation. Relabeling or copying such an artifact does not
change its identity.

**Purpose:** provide findings and proposed amendments for GLM to accept or
reject. This review does not modify the normative proposal.

## Verdict

The fail-first principle is the right ordering principle, and the new human
name is workable. The proposal nevertheless **does not pass this review**.

One P0 blocks every GLM/data-dependent campaign: §14 names a DSV4 run as the
GLM calibration contract, the GLM menu premise cites a DSV4-only corpus, and
the claim ledger retains another DSV4-only statistic. These are prohibited
inputs, not merely weak evidence.

Six P1 findings remain:

1. the scale legality sentence can reject legal E4M3 subnormals;
2. the launch-time encoder profile depends on terminal records created only by
   the encode;
3. `ProductionWeightCache` is incorrectly assigned ownership of an external
   Gridbook serving runtime;
4. Gridbook 0.9.1 has neither the Tessera ABI nor the required GLM/routed route;
5. the new stub skeleton is inconsistently authorized and is not yet specified
   well enough to deliver the claimed verdict; and
6. the proposed Tessera format spelling aliases many distinct terminal
   artifacts and conflicts with both the stated schema name and legacy TCQ
   identity.

The correct immediate posture is therefore:

- no DSV4-derived input or citation, without exception;
- no GLM campaign until a GLM-only, content-bound measurement manifest exists;
- model-neutral serializer/parser work only after the scale, input/output
  identity, and Tessera naming contracts are corrected;
- a synthetic hardware reproduction may proceed only when it is explicitly
  separated from §14's bad calibration authority and its complete input closure
  is DSV4-free; and
- allocator/menu/export/serving work remains gated on the external Gridbook
  contract and the full served-performance promotion gate below.

## Revalidation of the round-6 findings

| Round-6 finding | Result in this revision |
|---|---|
| Terminal classes versus concrete quota vectors and circular schema gate | **Resolved at the design level.** Terminal classes and actual `terminal_id` records are distinct; schema 1a precedes parser/tests 1b. P1-2 and P1-3 below close remaining legality and identity holes. |
| Scale legality and false representational parity | **Mostly resolved.** The text now says wire-rate parity, states the shared-base restriction, requires canonicalization, and reuses the two-tier abstraction. The subnormal sentence remains unsafe. |
| Objective provenance and arm 8 | **Partly resolved.** Arms 8a/8b and a numeric threshold are present. The producer profile still binds outputs unavailable at launch. |
| Cache/prefetch ownership | **Intent accepted, owner still wrong.** PrismaQuant must reuse its producer cache; Gridbook must own serving residency. |
| Preliminary fused evidence used as retirement evidence | **Mostly resolved.** The primary conclusion and ledger now say negative prior rather than retirement. Several stale statements remain. |
| Whiteness scope | **Resolved in the normative measurement section and ledger.** An older revision-history sentence still overstates closure. |
| Projected floor and dense-four-bit wording | **Mostly resolved.** The terminal table is correct; §8 still restores the wrong 1.28 number. |
| New fail-first amendment | **Direction accepted, contract incomplete.** The stub is useful as an optimistic necessary-condition screen, not yet as the stated parity verdict. |
| New Tessera name | **Human-facing name accepted provisionally.** The persisted identity, legacy compatibility, and schema spelling are not coherent yet. |

## Findings and proposed changes

### P0-1 — The measurement authority and claim ledger still use prohibited DSV4 evidence

Section 14 calls `prod-cal-0p6-v2` the “GLM5.3-flash calibration contract”
(`embedded_native_weight_coding_2026-08-31.md:652-661`). In this repository,
that name resolves to
`/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2`. Direct references include
`calib-study/calibration_stability.py:931`, `research-alloc/MANIFEST.json`, and
the DSV4 campaign tools. It is not a GLM artifact. It may not be translated,
rekeyed, copied, or used as a warm start.

The current text has additional prohibited dependencies:

1. The asserted GLM menu gap cites `format_menu_2026-08-29.md`
   (`embedded_native_weight_coding_2026-08-31.md:171-173`), and the ledger uses
   it for the `0/24`, floor, and decoder rows (`:730`, `:734-735`). That source
   explicitly says its quality corpus is 24 DeepSeek-V4-Flash MoE expert
   tensors and records that it had previously mislabeled the corpus as GLM.
   Its quality/menu verdicts are not GLM evidence.
2. The `top-100 = 93.66%` ledger row (`:754`) cites `calib-study`.
   `calib-study/REPORT.md` is a DSV4 study and derives 93.6596% from its DSV4
   assignment. Delete the row; do not preserve the number under a new name.
3. `trellis_first_artifact_plan_2026-08-31.md`, used by ledger rows `:733` and
   `:737`, is a mixed handover that includes DSV4-specific source-closure work.
   Even where an individual byte or runtime fact is model-neutral, the user's
   directive requires it to be re-homed in a clean, narrowly scoped artifact.

The other principal result artifacts audited here are not DSV4 measurements:
the shaped-ceiling, rank-1, rotation, and residual-spectrum studies use
Qwen3-0.6B; the fused 4096² benchmark is synthetic. They may remain only as
explicitly model-scoped priors. They do not become GLM evidence by appearing in
a GLM proposal.

There is a GLM-native starting identity already present:

- model profile `glm5_next` and serving profile
  `vllm_glm5_next_packed_moe`;
- census `/home/rob/dq-runs/glm53-flash/stock_anchored/probe_census.json`,
  schema `prismaquant.stock_anchored.probe_census.v1`, file SHA-256
  `605ba1cbbf2dc77f71a28f18564d5edbdcbb5833cfaa3969cd04a747f3a3e235`;
- model `/home/rob/models/GLM-5.3-Flash`;
- dataset `/home/rob/dq-runs/calibration/diverse-v1.jsonl`, file SHA-256
  `e09a138a4903c4af66a3bf2f9367185f3432224391f1dfe8c94ccc29d99315ba`;
- tokenized calibration hash `2aaba2d6cc67bc147d668a8094cd4806`,
  eight samples, sequence length 512;
- referenced probe SHA-256
  `74ba18ca26b2aa652dfd33e92ec5990dd035a85179b24b7c9c692e0a62227278`;
  and
- fail-closed driver `tools/run_glm53_stock_harvest.sh`.

This identifies the correct model/corpus boundary; it does **not** authorize
reuse of the stock cost artifact as a Tessera result.

**Required correction:**

1. Remove `prod-cal-0p6-v2`, the hot-set row, every DSV4 quality/menu claim,
   and every mixed-source citation from the Tessera authority. Do not copy
   their values into GLM-named files.
2. Restate the 3–4.25-bpp GLM menu gap as a hypothesis until measured on GLM
   using the same source precision, calibration rows, sequence length,
   activation contract, per-Linear semantics, and production cache behavior as
   its comparison baseline.
3. Create a content-bound GLM Tessera measurement manifest. It must bind the
   checkpoint/model-profile digest, source-dtype policy, census and probe
   digests, tokenized calibration hash, activation-cache identity, exact row
   selection, calibration parameters, encoder profile, container, code commit,
   and seeds. Every GLM arm must allowlist this identity and fail closed on any
   other model or upstream digest.
4. Record and verify the transitive input closure. A path-name denylist is not
   enough: no input digest may descend from a DSV4 artifact, even if copied or
   renamed.
5. Re-derive model-neutral wire arithmetic in a pure calculator/test artifact,
   and publish hardware timings in synthetic artifacts with their exact input
   generator. Neither may cite a mixed DSV4 document.
6. Add a claim-ledger invariant that no DSV4 artifact participates in Tessera
   measurement, fitting, allocation, encoding, export, or validation.

Until these changes land, do not launch arm 1, arm 5, the GLM whiteness work,
arm 2, or any other data-dependent GLM campaign from this proposal.

### P1-2 — The scale predicate can reject legal positive E4M3 subnormals

The shared two-tier specification explicitly allows exact positive E4M3
subnormal compositions. Section 6b instead says to reject forbidden tuples and
places “E4M3FN subnormals at the low end” in the parenthetical list
(`embedded_native_weight_coding_2026-08-31.md:348-360`). An implementation may
therefore reject every subnormal, contradicting the reused abstraction and
materially changing the measured scale population.

**Proposed replacement predicate:**

> A base/refinement tuple is legal iff its exact real composition round-trips
> bit-for-bit to a positive finite E4M3FN byte under the declared global/clip
> composition. Exact positive subnormals are legal. Zero, NaN, overflow, and
> inexact underflow are illegal and fail closed.

The exhaustive 65,536-word classification should freeze the legal-set digest
and separately test normal, subnormal, maximum-finite, duplicate/canonical, and
invalid classes. The GLM scale arm should report the population and quality
contribution of the subnormal class.

### P1-3 — `encoder_profile_id` is not a closed launch-time input

Concrete `terminal_id` records contain the actual count arrays, clip code,
physical bytes, and exact bpp selected during encoding (`:470-480`). The
lambda-greedy second pass chooses those quotas (`:517-524`). Yet the
`encoder_profile_id` binds terminal IDs and weights before the encode
(`:391-398`), and pass 1 is said to index its objective by those same IDs.

Those output terminal IDs do not exist when the campaign is launched or pass 1
runs. The profile is therefore partly a post-encode receipt, while the actual
pre-encode targets and initialization remain implicit.

**Proposed correction:**

1. Make the content-addressed **encoder profile input-only**. It declares a
   finite ordered set of terminal slots: class, allowed planes, target budget
   or deterministic lambda/quota policy, clip policy, and `w_p`, plus all
   source/calibration and algorithm/table versions.
2. Deterministically produce one concrete terminal record per slot. Its
   `terminal_id` binds realized count/offset planes, clip code, payload
   references, exact bytes, and exact bpp.
3. Put `encoder_profile_id`, the slot-to-terminal mapping, concrete terminal
   IDs, and payload digests in an artifact receipt. Replay starts from the
   input profile; cache/runtime lookup uses the output receipt.
4. Pass 1 scores declared slot targets. The optional rerun may score realized
   terminals. Freeze initialization, iteration limit, and all tie-breaks so the
   output is reproducible from the input profile alone.

### P1-4 — PrismaQuant's producer cache cannot own Gridbook serving residency

Section 12 correctly says Gridbook is the only Tessera-byte consumer, but then
assigns the encoded payload, side planes, device descriptors, and kernel
measurements to `ProductionWeightCache` at serving time (`:560-577`). This
collapses two repository owners.

`ProductionWeightCache` currently stores production-faithful dequantized
tensors or disk paths (`prismaquant/production_weight_cache.py:198-242`). It is
a PrismaQuant producer/probe/recache/export abstraction. In contrast,
`docs/ARCHITECTURE.md:6286-6293` makes Gridbook the sole owner of serving
configuration, loader hooks, CUDA, dispatch, telemetry, and runtime tests;
`:6490-6506` permits compatibility to cross only through the immutable pin and
packaged contract. PrismaQuant does not import Gridbook while exporting, and
Gridbook cannot depend on a PrismaQuant cache to serve.

**Proposed correction:**

- **Producer phase:** extend `ProductionWeightCache` or the existing streamed
  renderer with a typed Tessera render/artifact record. Probes, recache,
  export, exact byte accounting, and validation reuse that one producer cache;
  required render inputs are resident-prefetched and failures close.
- **Artifact boundary:** emit content-addressed payload/side tensors and the
  closed manifest. The pinned Gridbook runtime contract declares and validates
  the ABI and producer profile.
- **Serving phase:** the external Gridbook loader owns device placement,
  descriptors, residency accounting, prefetch, and fail-closed loading. No
  per-forward host parse or NVMe read is allowed, but that is a Gridbook
  runtime obligation.

Rename build item 3 to “producer cache and artifact handoff.” Put the installed
Gridbook residency tests with the external runtime gate, and add one
installed-wheel producer/consumer interop test.

### P1-5 — Gridbook 0.9.1 is a legacy TCQ integration, not a Tessera or GLM runtime

The proposal's Tessera-4 body roots are q256
`{256,384,512,640,768}` (`:231-239`). Gridbook 0.9.1 contract v12 instead says:

- legacy `TCQ_E2M1_R256` candidate rungs are
  `{384,512,640,768,896}`; q256=256 is readable but not producer-candidate;
- only q256=512 has device-qualified E2M1 dense decode/batch cells on sm_121;
- there is no `prismaquant.tessera.v1` reader or ABI for completion, release,
  typed count/offset, diagonal, or scale-mantissa planes;
- `glm5_next` is absent from `producer_profiles.supported_ids`; and
- no routed-MoE trellis/Tessera cell exists.

The integration branch's `gridbook_trellis_dense_sm121` profile explicitly
denies TCQ on routed experts for that last reason. The checked-in
`glm5_next` model profile likewise deliberately declares no Gridbook CB lane
until a released runtime names the architecture. That fail-closed boundary is
correct and must not be widened locally.

Build item 4 nevertheless says that advancing to 0.9.1 and wiring the eight
legacy `UNWIRED_LINKS` makes “the EN menu” appear (`:812-817`). Those links are
for the existing `TCQ_*` allocation seam. Wiring them is independently useful,
but it neither creates Tessera bytes nor gives GLM a consumer.

**Proposed correction:**

1. Label item 4 “legacy TCQ v0.9.1 pin/menu integration.” Do not call its menu
   Tessera or EN.
2. Add a separate external Gridbook Tessera release gate. Its packaged contract
   must declare `prismaquant.tessera.v1`, the exact root and terminal domain,
   all plane layouts, descriptor ABI, residency modes, producer profiles, and
   device-qualified route cells.
3. For GLM require `glm5_next`, fail-closed load coverage, and explicit dense
   and routed-MoE eligibility. Until a routed cell exists, routed GLM units are
   absent from the Tessera DP; any experiment is explicitly dense-only.
4. Advance PrismaQuant's immutable Gridbook pins only after the independent
   release, exact wheel digest, producer/consumer ABI test, and profile/cell
   checks exist. No fork, dirty checkout, or local alias may fill the gap.
5. Keep the vanilla-vLLM `T-nvfp4-RQ` artifact separate. Its CT render and ship
   gate do not make Tessera bytes readable by Gridbook.

### P1-6 — The fail-first skeleton is not yet an authorized or valid “verdict”

Moving an optimistic feasibility screen before runtime work is correct. The
current amendment has four contract problems.

First, authorization contradicts itself. Arm 4 remains under “Gated arms”
(`:684-701`), while item 0 says the stub starts immediately and is merely
“submitted ... as a permitted-work extension” (`:778-788`). A submitted
extension is not permission. Move it into the permitted list and update the
status, or keep it gated.

Second, “wire-bpp bytes at provisional strides with no decode at all” is not a
defined lower bound. A native MMA cannot consume arbitrary compressed bytes
without some transformation. If the reads do not influence an observable
result, the compiler or cache may eliminate or unrealistically schedule them;
if a checksum or substitute transform is added, that extra work is not
automatically unavoidable. Provisional stride and alignment can also make a
stub slower than a future layout, causing a false kill.

Third, item 7 still orders “Gridbook lane extension ... then the skeleton”
while the next clause says feasibility precedes kernel work (`:820-823`). The
item-0 stub and the post-1b full-layout skeleton are different gates and must be
named separately. Completion/release/diagonal reader work cannot precede the
full-layout gate.

Fourth, no final promotion arm requires the completed released route to serve
at parity with the displaced container. A skeleton pass is only a necessary
condition.

**Proposed correction:**

1. Define the stub's exact input/output and anti-elision dependency. Use the
   same target MMA/store geometry, device-resident inputs, launch discipline,
   and an explicitly optimistic contiguous/aligned read envelope. Sweep the
   relevant bpp range and alignments rather than one provisional layout.
2. Run representative GLM shapes and regimes, not only 4096². The GLM target
   includes routed gate/up and down shapes, dense/shared MLP shapes, and very
   small per-expert decode M; report each separately.
3. Treat the stub as an **optimistic necessary-condition lower-bound screen**.
   If a proved-optimistic bound misses, it may kill the measured shape/regime
   within the declared kernel class. A pass does not validate the actual wire.
4. After 1b, run the exact full-layout encoded-read skeleton **before** any
   Tessera reader/decoder/kernel implementation. Only then extend Gridbook.
5. Add a post-release serving promotion arm: exact wheel/pin/contract, bit-exact
   load, deterministic generation, eager and graph execution, MTP where
   applicable, dense/routed route attestations, exact resident bytes,
   latency/throughput by regime, in-process before/after profiles, synchronized
   Netdata series on both boxes, and work per joule. Compare against the exact
   container displaced; a slower route remains research.

The existing fused-decode reproduction should also say “Netdata series on both
boxes,” not merely “Netdata/power evidence.” It must be a clean synthetic
artifact independent of the prohibited calibration identity.

### P1-7 — The Tessera name does not yet provide a unique persisted format identity

The naming amendment declares schema `prismaquant.tessera.v1`
(`:113-126`), but both the wire gate and build item 1a still name
`prismaquant.en-stream.v1` (`:499-507`, `:794-804`). That is a direct schema
conflict, not just an incomplete prose sweep.

More importantly, `TESSERA_E2M1_R{q256}` identifies only family and body root.
Section 9 says two encodes at the same root with different per-plane count
arrays are different terminals (`:470-480`). Thus many distinct byte payloads,
exact bpp values, clip codes, and decoder selections would all be called, for
example, `TESSERA_E2M1_R512`.

The current `parse_trellis_format_name` contract cannot repair this: it parses
only a legacy TCQ family and q256 rate, and `layer_config.json` currently stores
that name as the cross-module identity. Reusing that shape for Tessera would
alias terminal candidates and could make downstream code treat a Tessera
artifact as a legacy TCQ rung.

**Proposed correction:**

1. Keep every existing `TCQ_*` name and parser behavior immutable for old
   artifacts. Tessera is a new schema/family, not a rename of TCQ.
2. Define whether `TESSERA_E2M1_R512` is only a human-readable family/root
   descriptor. The persisted assignment must additionally carry at least
   `schema`, `encoder_profile_id`, `terminal_id`, branch identity, and payload
   digest, or use a canonical content-addressed suffix that uniquely binds
   those fields.
3. Add a disjoint Tessera parser/record type and fail closed if a caller offers
   a family/root string where a concrete terminal record is required. Do not
   make the legacy two-tuple parser silently accept Tessera.
4. Use `prismaquant.tessera.v1` consistently in item 1a and §9, and specify
   compatibility behavior before registry/menu changes.
5. Run the promised novelty and collision checks, but do not treat name
   novelty as wire-identity proof.

No objection is raised to “Tessera” as the human-facing name once these
identity rules are supplied.

### P2-8 — Stale statements contradict accepted amendments

These are mechanical, but several affect load-bearing interpretation:

- §5 still says the removed residual route is “measured shut” (`:241-249`).
  The fused evidence is preliminary and not a retirement gate. Say “blocked by
  the current negative prior and separately unspecified.”
- §8 still says `body + 0.25` restores a “~1.28 floor” (`:425-430`). The
  projection is approximately 1.25 before exact side overhead; 1.28302 belongs
  to a different codec.
- §13 opens with “Three measured routes” and marks the fused row `[branch]`
  (`:598-614`), despite later text and the ledger classifying it
  `[branch-preliminary]`. Say two established routes plus one preliminary
  decomposed microbenchmark.
- Revision-history lines `:56-61` say “both rotation states,” “white wherever,”
  and leave only GLM/seed replication, while the later accepted scope also owes
  q256 `{256,384,640}` and `R_in`-only (`:92-94`, `:713-716`). Mark the older
  sentence superseded or state the full scope there.
- Correct “wire un gates” at `:499` to “wire ungates.”
- After accepting the post-round-7 owner and naming amendments, advance the
  document revision/status rather than continuing to call the materially
  changed 885-line document “revision 6.”

## Accepted boundary and recommended decision

Subject to the required 1a/1b artifacts, this review raises no new objection to
the typed-plane topology, terminal class/record distinction, completion
cardinality, appended four-bit release semantics, local `R_in`-only serving
rotation, direct-versus-requantized terminal distinction, or the algebra of the
bpp-dependent sensitivity function.

The scale construction remains an acceptable **candidate**, not a legal
terminal. The finite legal set, canonical encoder, exact schema, and GLM-only
matched-quality result remain gates.

Recommended disposition:

- **Prohibited:** every DSV4-derived input, transitive artifact, statistic, and
  mixed-source citation; every GLM launch under the current §14 identity.
- **May proceed after textual correction:** schema 1a and pure
  model-independent parser/serializer/footprint tests; a clean synthetic
  reproduction of the existing decoder; and, if explicitly authorized and
  fully specified, the optimistic stub lower-bound screen.
- **First GLM work after a clean manifest:** the minimal arm-2 measurement
  encoder and the already-permitted quality diagnostics, all bound to the same
  GLM-only calibration/source contract and preregistered thresholds.
- **Still gated:** Tessera allocator/menu/export integration, producer-cache
  mutation, any Tessera Gridbook pin claim, runtime reader/kernels, and serving.
  These require the unique terminal identity, external runtime contract,
  full-layout skeleton pass, and final served-performance gate above.
