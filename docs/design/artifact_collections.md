# Artifact collections: probe once, solve and export many

Status: **CURRENT foundation, not pipeline-wired** (branch
`codex/rtx5060-fp4-gridbook-pilot-20260824`, based on `5cfa928`).  The live
implementation is `prismaquant/artifact_collection.py`; the read-only bridge
for existing exports is `prismaquant/artifact_collection_legacy.py`.  This
foundation changes no format menu, allocator default, export codec, runtime
contract, or ship gate.

## 1. Purpose

An expensive model probe should be reusable across any number of candidate
sets, byte targets, device profiles, solves, exports, and qualifications.  A
new artifact size should begin at solve; a newly implemented format should
begin at cost if the existing probe already contains its declared features.
Neither event should silently trigger a new probe.

The control plane therefore separates four things that legacy handoffs often
mix together:

1. scientific identity (what was measured or chosen);
2. physical content identity (the exact bytes carrying it);
3. location (where those bytes happen to be mounted or published); and
4. evidence (an immutable receipt that an operation occurred).

The first implementation slice deliberately stops before pipeline wiring.  It
provides strict, content-addressed records, candidate and target schemas,
stage receipts, collection manifests, exact shared-resource accounting, and a
legacy artifact census.

## 2. Record topology

```text
model snapshot ──┐
probe campaign ──┼── collection contract ── target A ── solve/export/qualify receipts
candidate catalog┤                         ├─ target B ── solve/export/qualify receipts
cost snapshot ───┘                         └─ target N ── solve/export/qualify receipts
                                                   │
                                      regenerable collection manifest
```

The model, probe, catalog, and cost objects are common references.  Variants
contain a target profile and export contract under one common accounting rule.
Every receipt binds the exact collection contract and either the common
measurement campaign or one declared variant; accepted receipts require an
output, and all receipts require inputs and evidence.  Stage completion can
therefore be derived by a future DAG evaluator rather than written into a
mutable `status` field (`make_stage_receipt`, `make_collection_manifest`).

The v1 foundation defines these semantic records:

| Record | Identity-bearing responsibility |
|---|---|
| candidate | Complete numerical, basis, render, scale, activation, container, and runtime semantics |
| candidate catalog | Explicit set of candidate references; gaps and cross-family overlaps are legal |
| target profile | Exact artifact-byte ceiling, usable VRAM, accounting rule, device/workload references, placement, exclusions |
| collection contract | One common measurement base plus a sorted set of named target variants |
| stage receipt | Immutable accepted/rejected evidence for measure, solve, validate, export, qualify, or publish |
| collection manifest | Regenerable index of one contract and its receipt set |
| legacy export audit | Separate assignment, physical-tensor, recorded-byte, and observed-byte censuses |

ModelSnapshot, ProbeCampaign, CostSnapshot, Assignment, Export, and Device
Qualification remain referenced opaque records in this slice.  Their strict
payload validators are the next control-plane layer; their references already
bind schema, semantic ID, content SHA-256, and byte size, so they can be added
without changing collection identity rules.

### 2.1 Complete planned DAG payloads

The opaque v1 references are not unspecified.  Their future typed payloads
have the following minimum identities and reconciliation rules:

| Entity | Required identity | Required reconciliation |
|---|---|---|
| ModelSnapshot | portable source binding; model-profile contract; immutable unit-ledger blob; assignment-required and excluded unit-set digests; producer source-tree digest | unit IDs/qnames unique; required and excluded sets disjoint and exhaustive; shapes and parameter totals reconcile to checkpoint headers |
| ProbeCampaign | ModelSnapshot ref; exact probe blob; calibration/token-content identity; measured feature set; covered/missing unit sets; merge receipt; producer | covered units resolve in ModelSnapshot; missing required units explicit; candidate reuse requires `required_probe_features` to be a subset of measured features |
| CostSnapshot | model/probe/catalog refs; observation blob; named metric contracts; cell coverage; measured/derived/unavailable counts; accounting-rule ref; producer | every cell resolves to an applicable unit/candidate pair; values finite under one currency; derived cells retain anchors; candidate-local bytes never masquerade as whole-artifact bytes |
| Assignment/Solve | model/probe/catalog/cost/target refs; exact unit→candidate records; optimized/fixed partitions; solver/search-space identity; predicted metrics; exact byte breakdown | every required unit assigned once; fixed and optimized sets disjoint/exhaustive; optimized choices have cost cells; all choices are applicable; accounting rule and target ceiling agree |
| Export | Solve ref; assignment digest; producer; exact file/tensor/codebook inventory; runtime artifact identity; named byte measurements | assignment agrees with Solve; inventories have no missing or extra members; content hashes verify; Target is checked against its exact named byte scope |
| DeviceQualification | Export and Target refs; runtime-contract ref; exact device/workload/placement; required check IDs; immutable evidence blobs; verdict | every check binds the same artifact; required set exact; device and placement satisfy Target; evidence from one device class cannot qualify another |
| MarketSnapshot | source-content refs for survey observations; observation date; normalized device IDs; counts/shares plus scope and collection method | raw observations remain separate by source; Hugging Face users are not presented as installed-base counts; Steam shares are not converted into absolute units |
| ReleaseDecision | MarketSnapshot ref; candidate Target/Export/Qualification refs; policy identity; included and rejected variants with reasons | no unqualified artifact is included; artifact aliases remain outside scientific identity; one release decision cannot mutate upstream evidence |

The release view is therefore a traversal over immutable records, not a ninth
mutable mega-manifest.  A target can be added by producing Target and Solve
records against the same shared probe/cost base.  A candidate can be added at
CostSnapshot only if the existing ProbeCampaign proves feature sufficiency;
otherwise it correctly starts a new probe lineage.

## 3. Identity and publication rules

Every record uses the repository's canonical JSON implementation
(`cost_stage_checkpoint.canonical_json`) and the envelope:

```json
{
  "schema": "prismaquant.artifact_collection.candidate.v1",
  "payload": {},
  "payload_sha256": "64 lowercase hexadecimal characters",
  "locators": {
    "subject payload SHA-256": ["advisory location"]
  }
}
```

`payload_sha256` is the semantic ID.  `locators` is an optional, sorted,
in-memory resolution overlay excluded from that ID.  `write_record` publishes
only the portable three-field semantic envelope, so the resulting file's
SHA-256 and size exactly match `reference_for_record`; locator overlays must be
distributed separately.  Moving identical content between a local worktree,
object store, and Hugging Face cannot rename it.  References contain the
subject schema and semantic ID plus the SHA-256 and size of its portable
content.  One `(subject schema, semantic ID)` is forbidden from resolving to
multiple contents.  Timestamps, aliases, release labels, and mutable gates do
not enter scientific identities.

All recognized payloads and the envelope itself are closed objects: unknown
identity-bearing fields fail validation.  JSON duplicate keys, non-finite
numbers, malformed hashes, Boolean-as-integer byte counts, tampering, and
duplicate references also fail.  `write_record` publishes via a no-clobber
hard-link operation; it will not replace a file or symlink that already
exists.

## 4. Candidates, including learned versus lattice bases

A format name is presentation, not candidate identity.  The candidate record
binds:

- format semantics and parameters (family and rung are data, not schema
  enums);
- basis kind, scope, and exact asset references;
- renderer, scale, and activation contracts;
- serialization and runtime contracts;
- required probe and runtime feature sets;
- applicability rules; and
- shared physical resources.

Consequently `NVFP4_CB_K12` backed by a lattice and the same label backed by a
learned book are distinct candidates.  Two containers for the same numerical
candidate are also distinct candidates, but share the computed `behavior_id`;
validation can therefore be reused without pretending the exported bytes are
the same.  Lattice assets are content-bound just like learned assets—"fixed"
does not mean provenance-free.

The schema encodes no minimum, maximum, or contiguous rung range.  K1, K32,
holes between them, and overlaps with FP8 are all representable as explicit
catalog members.  Promotion policy remains external: lattice-first is the
safe production default, while a learned candidate must earn its extra asset,
training, and runtime complexity under the same target and qualification
contract.

The current range-study recommendation is a **research catalog of K1–K32**, not
a blanket support claim.  With two four-value-vector product subtables, the
index split is `(ceil(K/2), floor(K/2))`; K1 is therefore the valid `(1, 0)`
degenerate product, not malformed metadata.  An E2M1 value has 16 possibilities
and a four-value tuple has `16^4 = 2^16` possibilities, so K32 gives each
subtable 16 bits and exhausts the information-bearing lattice.  K33 can only
duplicate tuples.

The physical boundary is earlier.  A BF16 product dictionary occupies
`2 * (4*2^ceil(K/2) + 4*2^floor(K/2))` bytes: 65,536 at K24, 98,304 at K25,
and 131,072 at K26.  The present GB10 full-dictionary staging budget is 101,376
bytes (`gridbook/csrc/cb_gemv_v2.cu`), making K25 the last rung compatible with
that expander architecture and K24 the current production/fast-LUT boundary.
K26–K32 require a new streamed, cached, or otherwise non-full-staging decoder;
they are not a constants-only extension.  Accordingly:

- enumerate K1–K32 as possible research candidates;
- qualify K1–K25 first, one explicit rung at a time;
- never advertise K26–K32 until a new decode route clears parity and speed
  gates; and
- keep lattice and learned versions as distinct candidates at every rung,
  with lattice the baseline and learned promotion driven by measured artifact
  quality after its shared asset cost is charged exactly once.

## 5. Byte accounting

An assignment maps a unit ID to a candidate ID and declares local bytes plus
shared-resource claims.  `assignment_byte_breakdown` sums local bytes and
deduplicates shared resources by physical content SHA-256.  It rejects a
reused digest with inconsistent sizes and duplicate unit IDs.

This primitive is intentionally smaller than the final whole-artifact
accountant.  The same `accounting_rule` reference must flow from target to
solve to export, where the implementation must add fixed model data,
container overhead, tokenizer/config files, and packaging files under one
named scope.  Candidate-local byte estimates must never substitute for the
final recursive package measurement.

### 5.1 Hardware tiers and export shoulders

As observed 2026-08-24, NVIDIA's desktop specifications group the relevant
cards into 8 GB (RTX 5050/5060 and one 5060 Ti SKU), 12 GB (5070), and 16 GB
(the other 5060 Ti SKU, 5070 Ti, and 5080).  Laptop parts are predominantly
8 GB through 5070, 12 GB at 5070 Ti, and 16 GB at 5080.  Sources:
[NVIDIA desktop comparison](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/)
and [NVIDIA laptop comparison](https://www.nvidia.com/en-us/geforce/laptops/50-series/).

Two demand proxies point at different but complementary audiences.  Valve's
July 2026 all-card survey reports desktop RTX 5060 at 3.03%, 5060 Ti at 2.40%,
5070 at 3.62%, 5070 Ti at 1.78%, and 5080 at 1.63%; 5060 Laptop adds 2.19%.
The [Steam survey](https://store.steampowered.com/hwsurvey/videocard?sort=chg)
is an installed-use share, but combines the 8/16 GB 5060 Ti SKUs.  Hugging
Face's opt-in AI-user census instead reports roughly 13k owners each for the
5060 Ti and 5070 Ti, 9k for 5080, 6k for 5070, and 4k for 5060.  The
[Hugging Face hardware page](https://huggingface.co/hardware) is an AI-demand
proxy, not a population estimate.

The solve grid should be dense from 5 through 14 decimal GB, but publication
should expose a small set of qualified shoulders:

| VRAM tier | Solve shoulders | Publication intent |
|---|---|---|
| 8 GiB | 5.0 and 5.5 GB | lab artifacts only until full GPU-only residency, context, eager/graph, KL, and PPL gates pass on real 5050/5060-class hardware |
| 12 GiB | 7.5, 9.5, and 10.0 GB | 7.5 GB headroom build plus the largest physically qualified quality build; 10 GB is a useful experiment, not a product definition |
| 16 GiB | 10.0, 13.0, and 14.0 GB | compact cross-tier build, 13 GB default, and 14 GB quality shoulder if runtime headroom remains green |

Because export is cheap, every shoulder may be materialized.  ReleaseDecision
should normally publish at most one headroom and one quality artifact per VRAM
tier, based on physical qualification and measured quality—not on a round
marketing size.  The 8 GiB tier is the reach objective; the 12 and 16 GiB
tiers are the credible initial shipping targets for a 27B model.

## 6. Legacy Qwen3.8-27B regression oracle

The current 13 GB Qwen artifact proves why named scopes are necessary.  The
legacy importer reads `quant_config.json` for assignment and recorded export
authority, a safetensors-header `shapes.json` projection for physical tensor
authority, and optionally stats the current artifact directory.  It does not
hash the 13 GB model and labels the directory observation
`stat_size_only_unsealed`.

The 2026-08-24 census is:

| Scope | Value |
|---|---:|
| explicit format assignments | 498 = 496 CB + 2 auxiliary |
| format assignments plus otherwise-fixed MTP matrix modules | 506 |
| physical model tensors | 1,365 |
| physical `mtp.` tensors | 15 = 8 matrices + 7 support tensors |
| physical `mtp.` bytes | 849,398,784 |
| CB tensor payload | 10,116,354,728 |
| shared CB sidecars | 103,424 |
| serialized CB body | 10,116,458,152 |
| producer-recorded recursive export | 12,982,409,320 |
| currently observed recursive package | 12,982,463,529 |
| post-inventory drift | 54,209 |

The drift is exactly `README.md` (18,796), `allocation-map.png` (14,624), and
`serve_manifest.json` (20,789).  The importer preserves both totals; it does
not overwrite producer provenance with a later observation.  Its recorded
ledger balances exactly as:

```text
12,982,409,320
  = 10,116,458,152 serialized CB body
  +    849,398,784 physical MTP namespace
  +  2,016,552,384 fixed recorded residual
```

This oracle also forbids an ambiguous `unit_count`: assignment units,
CB-only targets, matrix modules, and physical tensors are separately named.

## 7. Next implementation layers

The foundation is ready for the following work without deciding the NVFP4
rung range:

1. strict ModelSnapshot and UnitLedger payloads derived from `model_walk`;
2. a ProbeCampaign adapter that publishes feature coverage and a portable
   source binding;
3. a CostSnapshot adapter with measured/derived/unavailable cell provenance;
4. an Assignment/Solve record that maps every required unit to a candidate
   ID and invokes one whole-artifact accounting rule;
5. Export and DeviceQualification records that bind exact file hashes,
   artifact identity, runtime contract, workload, placement, and hardware;
6. a release-policy evaluator that derives publishability by traversing
   accepted receipts; and
7. only then, catalog population for the desired NVFP4 lattice and learned
   candidates and physical qualification on the target RTX 50-series tiers.

Until those adapters land, these records are an offline control-plane API,
not a claim that any new rung, artifact size, or device route is supported.
