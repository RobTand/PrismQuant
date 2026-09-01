# Codex round-8 review: revision 7 source and execution audit

**Date:** 2026-08-31

**Reviewed document:** `embedded_native_weight_coding_2026-08-31.md`, SHA-256
`092ae974122e50ee6071fadecc508550f3fc020295aa28f44fffc15ee0dbb14f`,
1,020 lines.

**Evidence checked:** the current PrismaQuant checkout; Codex round-7 review;
the GLM model and serving profiles; the GLM probe and checkpoint censuses;
`tools/run_glm53_stock_harvest.sh`; the checkpoint visible under
`/mnt/shared/models/GLM-5.3-Flash` on both this host and `sparklina`; the
remote probe and activation-cache paths; the cited claim-ledger documents;
Gridbook release `227420f9821bab7089632ee914f0ba050f82b817`; and the mandatory
repository rules supplied with this review.

**Binding owner directive:** DSV4-Flash is not used anywhere for any reason.
Calling its evidence “removed,” “historical,” or a “model-scoped prior” while
retaining its numbers or citations in the active proposal is still use.

**Purpose:** provide a sibling review for GLM to accept or reject. The
normative proposal is not modified here.

## Verdict

Revision 7 correctly incorporates most of round 7's structural changes: the
scale predicate now admits exact positive subnormals; encoder inputs and output
receipts are mostly separated; producer and runtime residency owners are
split; Gridbook 0.9.1 is labeled legacy TCQ; the full-layout skeleton precedes
decoder work; a final serving arm exists; and the 1.25-bpp projection is fixed.

The proposal nevertheless **does not pass**. No GLM/data-dependent work is
authorized by this review.

Two P0 blockers remain:

1. the active proposal and its sole load-bearing ledger still use prohibited
   DSV4 evidence; and
2. the required BF16 GLM source does not exist in the declared execution
   closure. The identified GLM checkpoint is FP8 on every quantizable
   parameter, while the cited driver hardcodes a missing model path.

Five P1 findings affect the proposed manifest, the mandatory power envelope,
the encoder identity split, the stub lower-bound proof, and the external
runtime/promotion order. One P2 section collects remaining contradictions.

## Round-7 finding disposition

| Round-7 finding | Revision-7 result |
|---|---|
| P0-1, prohibited DSV4 evidence | **Not resolved.** The calibration path was removed, but DSV4 data and mixed-source citations remain active as “priors” and ledger rows. See P0-1. |
| P1-2, exact E4M3 subnormals | **Resolved.** §6b now states an exact positive-finite predicate, freezes a legal-set digest, and separates invalid classes. |
| P1-3, input profile versus output terminals | **Mostly resolved.** §7 defines input slots and output receipts. §9 still indexes pass-1 weights by output terminal IDs; see P1-5. |
| P1-4, PrismaQuant/Gridbook ownership | **Resolved at the design level.** The producer cache, artifact boundary, and external serving loader are distinct. One wording defect remains in P2-8. |
| P1-5, legacy TCQ versus Tessera/GLM runtime | **Resolved at the contract-description level.** A separate Tessera contract is required and routed GLM units fail closed. Build/promotion sequencing remains incomplete; see P1-7. |
| P1-6, stub and final serving gate | **Partly resolved.** Authorization remains pending and Arm 12 exists. The checksum stub is not proved optimistic, and Arm 12 is not placed in the build-order gate; see P1-6/P1-7. |
| P1-7, Tessera identity | **Mostly resolved.** The family/root name is explicitly non-unique and a disjoint record is required. The schema still offers two alternative persisted identities and the naming history contradicts the parser rule. |
| P2-8, stale mechanics | **Partly resolved.** The 1.25 floor, section heading, schema name, and “measured shut” wording are fixed. Several stale statements remain. |

## Findings and proposed changes

### P0-1 — DSV4 is still actively used by the proposal and its load-bearing ledger

The status says DSV4 is prohibited everywhere (`embedded_native_weight_coding_2026-08-31.md:3-12`),
and §14 says it may not appear even in citations (`:767-770`). The body then
does exactly that:

- revision history records the paths, corpus, statistic, and source rationale
  at `:132-164`;
- the GLM problem statement retains the DSV4 menu result as a “model-scoped
  prior” at `:210-217`;
- the source-precision rule calls DSV4 its rationale at `:735-754`;
- the sole load-bearing ledger preserves the removed hot-set number and the
  24-tensor menu result at `:860-861`;
- the ledger continues to cite the mixed DSV4 `format_menu` for the scale floor
  and decoder numbers at `:865-866`;
- `trellis_first_artifact_plan_2026-08-31.md`, cited at `:864` and `:868`, is a
  mixed handover containing DSV4-specific source-closure work;
- the combined lane row at `:869` joins clean synthetic exactness to the
  DSV4-corpus E4M3 crossover; and
- the MXFP4-source ceiling row at `:883` cites
  `MOE_LEARNED_CODEBOOK_SPEC.md:86`, whose stated rate gate is DSV4-specific.

The ledger is declared the **only** load-bearing claims list (`:167-168`). A
row labeled “removed” or “prior” is still in that authority, still supplies a
number, and still invites downstream reuse. This directly contradicts the
owner directive. Round 7 explicitly required clean re-homing; that work was
not done.

**Required correction:**

1. Delete every DSV4-derived row, number, comparison, rationale, and source
   citation from the active Tessera proposal. Do not retain them as priors,
   removed rows, historical context, or source-policy examples.
2. State the GLM menu gap only as an unmeasured GLM hypothesis. It needs no
   prior value to motivate measurement.
3. Re-derive the 1.28302 legacy-plane arithmetic in a pure calculator/test
   artifact if the comparison is still useful. Re-publish decoder and lane
   facts from clean synthetic Gridbook artifacts, with no provenance transit
   through a mixed document.
4. Split the `:869` row: retain synthetic E2M1 exactness only after giving it a
   clean source; delete the corpus crossover until measured on GLM.
5. Delete the `:883` source-ceiling row. It is both prohibited and irrelevant
   to a BF16-start GLM campaign.
6. Add a Tessera-document/manifest CI check that rejects the known prohibited
   source identities and citations. Keep the content-addressed ancestry check;
   a text denylist alone is insufficient.

Historical review files may remain append-only records, but they are not
Tessera inputs, active citations, baselines, or claim-ledger rows.

### P0-2 — The required BF16 GLM campaign has no BF16 checkpoint or executable driver

Revision 7 says the GLM campaign starts from BF16 (`:158-164`, `:744-754`).
The actual identified source contradicts that requirement:

- `/home/rob/dq-runs/glm53-flash/stock_anchored/checkpoint_census.json`, SHA-256
  `1830abd261c9e6ccc6687ce7668d22761fd0af825d64785fd27fa3f7e9aea417`,
  records 917 units: 263 FP8 and 654 BF16;
- more importantly, `prismaquant/model_profiles/specs/glm5_next.json:224-239`
  records all 305,915,756,544 **quantizable parameters as 100% F8_E4M3**;
- the checkpoint census and model profile identify the source as
  `/mnt/shared/models/GLM-5.3-Flash`, not the path in §14;
- `/home/rob/models/GLM-5.3-Flash` is absent on both this host and `sparklina`;
  and
- `tools/run_glm53_stock_harvest.sh:45,60` hardcodes that absent path, so its
  remote preflight would fail before GPU work.

The historical probe census is not a BF16-source receipt. It points at the
missing `/home/rob/models/...` path and at probe/activation paths that exist
only on `sparklina`; it is not content-bound to any available BF16 checkpoint,
and the stock driver combines it with the FP8 checkpoint census. Dequantizing
the FP8 checkpoint into a BF16 tensor does not restore the unavailable BF16
source and must not be called a BF16 start.

The prose is also self-contradictory: it requires BF16, then says sources are
unrestricted and permits four-bit-origin campaigns (`:748-754`). The latter
rule derives from the prohibited source rationale and should be removed. A
low-precision source can support claims relative to that exact source tensor;
it cannot support fidelity claims to an unavailable higher-precision model.

**Required correction:**

1. Acquire and content-bind a genuine BF16 GLM-5.3-Flash checkpoint. Do not
   substitute the existing FP8 checkpoint or an FP8-to-BF16 expansion.
2. Regenerate the checkpoint census, probe, activation cache, tokenization
   receipt, and every baseline/cost artifact from that BF16 checkpoint. Bind
   their complete digests to the new campaign; none of the current FP8-source
   identities may be inherited.
3. Make the driver consume the manifest's checkpoint URI and digest rather
   than a hardcoded host path. Its preflight must verify the full checkpoint
   closure before any capture or encode.
4. Keep every GLM arm blocked until the BF16 closure exists and a clean
   preflight succeeds in the known-good Docker environment.

### P1-3 — §14 describes a future manifest but does not define a closed machine-readable identity

Section 14 is an improved checklist, not yet a manifest contract. It provides
truncated hashes, defers the checkpoint digest, gives no schema or canonical
path, does not bind the checkpoint census or activation-cache contents, and
omits the arm-specific encoder profile, thresholds, seeds, code closure, and
container digest (`:735-765`). Therefore “every arm allowlists this identity”
has no concrete identity to validate.

One immutable manifest also cannot bind multiple arms with different encoder
profiles and seeds without being mutated. The ordinary calibration contract
must be stable across arms, while execution inputs must differ by arm.

The cited driver is not a Tessera fail-closed driver:

- it hardcodes a missing model path;
- it records a dirty count but still rsyncs and executes dirty workspace bytes;
- it consumes a host venv rather than the repository's known-good Docker
  environment; and
- it validates the historical stock-harvest census, not a Tessera manifest or
  encoder profile.

**Proposed contract:**

1. Define an immutable `glm53_tessera_calibration.v1` base record containing
   full 64-hex digests for the BF16 checkpoint shard/index closure, model and
   serving profiles, dataset, tokenization, probe census and payload,
   activation-cache manifest, checkpoint census, quantizable set, and
   calibration parameters.
2. Define one immutable `tessera_arm_run.v1` per arm. It references the base ID
   and binds the encoder profile, terminal slots, thresholds, seeds, exact code
   commit/source closure, Docker image digest, hardware targets, and commands.
3. Define an output receipt binding raw rows, logs, profiler traces, Netdata
   windows, payloads, and pass/fail decision back to the arm-run ID.
4. Add a validator used by every entry point. Missing, truncated, mutable,
   cross-model, wrong-source-dtype, dirty-code, or wrong-container identities
   fail before GPU allocation.

This separation preserves the “same calibration contract” requirement without
pretending different experiments are one immutable execution manifest.

### P1-4 — The ~100 W “GPU ceiling” contradicts the mandatory ~140 W envelope

The repository rule supplied for this review requires power to be evaluated
against the approximately **140 W envelope** and work ranked by work per joule.
Revision 7 instead changes headroom and envelope fraction to an asserted
~100 W GPU ceiling (`:158-164`, `:756-762`).

An observed maximum is not a hardware envelope. Treating “the GPU never
exceeded ~100 W” as the denominator makes a historically under-loaded run look
fully loaded and destroys the headroom signal the rule requires. The statement
also has no attached measurement receipt in this proposal.

**Required correction:**

- retain ~140 W as the envelope for headroom/envelope-fraction analysis;
- report measured GPU-board power and whole-system power as separate series;
- calculate work per joule from actual integrated energy, not either nominal
  ceiling; and
- if ~100 W is a repeatable observed plateau, label and cite it as an
  observation, never as the envelope. GPU utilization remains non-diagnostic.

### P1-5 — The output-terminal cycle survives in §9

Section 7 now correctly says the input profile contains ordered terminal slots
and concrete terminal IDs are produced after encoding (`:443-457`). Section 9
still says “the encoder's ... weights” are indexed by the concrete
`terminal_id` (`:532-541`), and pass 1 still speaks of a declared terminal set
(`:586-591`). That reintroduces the exact launch-time cycle revision 7 claims
to have removed.

**Proposed correction:** pass-1 weights and anticipated-completion targets are
indexed only by stable input `terminal_slot_id`. The optional rerun may use the
realized `terminal_id`. The artifact receipt maps slot to terminal; no input
profile or first pass may require an output ID.

Also choose one persisted Tessera identity in item 1a. The current “structured
fields **or** a content-addressed suffix” at `:542-546` is an unresolved wire
choice, not a canonical schema. A digest may summarize the structured record,
but there must be one normative representation and hash domain.

### P1-6 — A checksum-bound read kernel is not proved to be an optimistic lower bound

The stub is correctly left unauthorized and is now described only as a
necessary-condition screen (`:793-804`). Its proposed anti-elision mechanism,
however, requires checksum-bound reads (`:795-798`). Checksum work is not a
necessary operation of every possible fused decoder. It can make the stub
slower than a real implementation, so a miss cannot prove that all decoders in
the declared class miss. Calling it “proved-optimistic” does not supply the
proof.

**Required before authorization:** define a mathematical lower bound separately
from any constructive benchmark. The bound may use measured best-case device
bandwidth, exact bytes, ideal overlap, and the target MMA floor. A concrete
anti-elision kernel is evidence for that particular construction, not a
universal lower bound, unless the proposal proves every instruction on its
critical path is unavoidable. Report both and restrict any kill decision to
what the evidence actually bounds.

The post-1b full-layout skeleton remains useful, but a pass is still not a
decoder or serving result.

### P1-7 — Gridbook release, Arm 12, and pin/menu promotion are not in one executable order

Build item 4 says the external Tessera release gate “follows” the legacy TCQ
pin work (`:937-951`), while item 7 correctly says no Tessera reader or kernel
work precedes the full-layout skeleton (`:954-958`). Arm 12 exists in §14, but
the build order never makes its served-parity result a prerequisite for the
Tessera pin/menu promotion.

The target-model scope makes this more than editorial. The current GLM profile
attributes 304.41 billion quantizable parameters to routed experts. A
dense-only Gridbook route may support isolated research, but it cannot support
a representative GLM shipping claim.

**Proposed order:**

1. legacy TCQ 0.9.1 integration remains independent and must not advertise
   Tessera;
2. Tessera 1a/1b and the full-layout skeleton pass;
3. external Gridbook Tessera reader/kernels are implemented, including a
   `glm5_next` routed-MoE cell for any GLM production claim;
4. Gridbook produces a pinned release-candidate wheel and packaged contract;
5. Arm 12 runs against that exact wheel and the exact displaced container; and
6. only a passing Arm 12 permits the PrismaQuant Tessera pin, menu admission,
   architecture update, or shipping status.

A dense-only miss or pass is reported as dense-only research. It does not
promote Tessera for the GLM target.

### P2-8 — Remaining contradictions and provenance defects

- Revision-history lines `:110-115` still say the stub “suffices for the
  verdict,” contradicting the new necessary-condition language.
- Naming lines `:128-130` say `parse_trellis_format_name` must change, while
  §9 says legacy TCQ parsing is immutable and Tessera uses a disjoint record
  (`:542-549`). The naming commit should add a Tessera parser, not change the
  legacy parser's accepted language or return type.
- The fused-decode bullet still ends in `[branch]` at `:689-694`; its heading
  and ledger correctly call it preliminary. Use `[branch-preliminary]`
  consistently.
- Section 12 says “no per-forward host parse or NVMe read is a Gridbook runtime
  obligation” (`:649-653`), which can read as saying the prohibition is not an
  obligation. Say directly: “Gridbook must perform neither per-forward host
  parsing nor per-forward NVMe reads.”
- The provenance header still says the text was hardened through only the old
  six rounds (`:14-29`), and §18 claims every absorbed review is linked but
  omits `embedded_native_weight_coding_2026-08-31_codex_round7_review.md`
  (`:1004-1016`). Add the pointer and update the provenance statement.
- Revision 7 calls Codex round 7 “the final pass” (`:132`) even though this
  round was requested. Use neutral historical wording.
- Revision-history lines `:61-66` still preserve the superseded “white
  wherever”/incomplete remaining-scope claim. Mark it superseded by the exact
  scope at `:97-99`.

## Accepted boundary and recommended disposition

No new objection is raised to the typed-plane topology, completion/release
cardinality, exact subnormal predicate, producer/runtime ownership split,
legacy-TCQ separation, or the requirement for a post-1b full-layout skeleton
and a final serving arm.

Recommended disposition:

- **Prohibited:** every DSV4-derived datum, number, rationale, citation, and
  mixed-source handover; any attempt to retain one as a prior.
- **Blocked:** every GLM/data-dependent arm until a genuine BF16 checkpoint and
  newly derived content closure exist.
- **Not yet authorized:** the checksum stub, because it is not a proved
  optimistic lower bound.
- **May continue after textual correction:** pure model-independent schema,
  parser, serializer, footprint, and scale-legality work; and a clean synthetic
  reproduction of the released legacy decoder in the known-good environment.
- **Still gated:** Tessera allocator/menu/export work, producer-cache mutation,
  Gridbook implementation, pin advancement, and serving. These require the
  canonical identity, valid BF16 manifests, full-layout skeleton, routed GLM
  runtime cell, and passing Arm 12 in the order above.
