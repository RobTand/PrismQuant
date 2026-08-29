# Artifact collections: probe once, solve and export many

Status: **CURRENT typed control plane, not pipeline-wired** (branch
`codex/rtx5060-fp4-gridbook-pilot-20260824`).  The envelope, candidate,
target, contract, and receipt layer is `prismaquant/artifact_collection.py`;
the value-bearing records and closed-graph verifier are
`prismaquant/artifact_collection_records.py`; the read-only bridge for
existing exports is `prismaquant/artifact_collection_legacy.py`.  Run
`python -m prismaquant.artifact_collection_cli verify RECORD.json ...` to
verify a complete record set offline.  This control plane changes no format
menu, allocator default, export codec, runtime contract, or ship gate.

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

The implementation deliberately stops before pipeline wiring.  It provides
strict, content-addressed records from source census through release decision,
closed-set graph reconciliation, exact shared-resource and whole-package byte
accounting, stage receipts, collection manifests, and a legacy artifact
census.  Producers still need adapters that turn live pipeline outputs into
these immutable records.

## 2. Record topology

```text
source + unit ledger ─ model snapshot ─ probe campaign ─ cost snapshot
                                                │              │
candidate catalog ──────────────────────────────┴──────────────┤
target A/B/N ───────────────── collection contract ────────────┤
                                                               v
                                      solve ─ export ─ evidence ─ qualification
                                                               │
market snapshot ───────────────────────── release decision <───┘
                         receipts ─ collection manifest
```

The model, probe, catalog, and cost objects are common references.  Variants
contain a target profile and export contract under one common accounting rule.
Every receipt binds the exact collection contract and either the common
measurement campaign or one declared variant; accepted receipts require an
output, and all receipts require inputs and evidence.  Stage completion can
therefore be derived by a future DAG evaluator rather than written into a
mutable `status` field (`make_stage_receipt`, `make_collection_manifest`).

The v1 control plane defines these semantic records:

| Record | Identity-bearing responsibility |
|---|---|
| candidate | Complete numerical, basis, render, scale, activation, container, and runtime semantics |
| candidate catalog | Explicit set of candidate references; gaps and cross-family overlaps are legal |
| target profile | Exact artifact-byte ceiling and scope, usable VRAM, fixed resources, accounting rule, device/workload/placement, exclusions, required qualification checks |
| collection contract | One common measurement base plus a sorted set of named target variants |
| unit ledger / model snapshot | Exhaustive assign-or-exclude source census, source-content digest and parameter total, model profile, and producer tree |
| probe campaign | Exact calibration/token/probe identities, measured features, and exhaustive covered/missing units |
| cost snapshot | Exhaustive unit-by-candidate measured/derived/unavailable matrix with local bytes and derivation anchors |
| solve | Exact exhaustive assignments, cost-derived local bytes, candidate-derived shared resources, target-derived fixed resources, and byte breakdown |
| export | Solve identity plus portable file/tensor/codebook inventory, runtime identity, exact byte scope, and whole-package bytes |
| qualification evidence | One check outcome bound to the exact export, runtime, device, workload, and placement |
| device qualification | Target-required nonempty check set, evidence references, and derived accepted/rejected verdict |
| market snapshot | Canonical UTC observation time and source-separated raw receipts, scope, method, and observations |
| release decision | Exact collection and market refs plus an exhaustive included/rejected partition of collection variants |
| stage receipt | Immutable accepted/rejected evidence for measure, solve, validate, export, qualify, or publish |
| collection manifest | Regenerable index of one contract and its receipt set |
| legacy export audit | Separate assignment, physical-tensor, recorded-byte, and observed-byte censuses |

Unknown artifact-collection schemas are refused rather than treated as opaque
validated data. External byte contracts may remain references, while every
reference to a control-plane-owned schema must resolve inside the graph's
closed record set.

### 2.1 Enforced DAG payloads

The typed payloads enforce the following minimum identities and reconciliation
rules:

| Entity | Required identity | Required reconciliation |
|---|---|---|
| ModelSnapshot | portable source binding; model-profile contract; immutable unit ledger; source parameter count; required/excluded digests; producer tree | nonempty unique unit IDs/qnames; every unit assign/exclude; shape products and totals reconcile; source content digest equals ledger model digest |
| ProbeCampaign | ModelSnapshot ref; exact probe blob; calibration/token-content identity; measured feature set; covered/missing unit sets; merge receipt; producer | covered units resolve in ModelSnapshot; missing required units explicit; candidate reuse requires `required_probe_features` to be a subset of measured features |
| CostSnapshot | model/probe/catalog refs; observation blob; metric contracts; exhaustive cell coverage; provenance counts; accounting rule | every required unit/candidate pair has one cell; usable cells require probe features and coverage, finite metrics, and `local_bytes`; derived cells retain unique anchors |
| Assignment/Solve | model/probe/catalog/cost/target refs; exact unit→candidate rows; solver identity; predicted metrics; exact byte breakdown | every required unit assigned once; local bytes equal its usable cost cell; shared resources equal its candidate; fixed resources equal Target; accounting and ceiling agree |
| Export | Solve ref; assignment digest; exact portable file/tensor/codebook inventory; runtime identity; named byte measurement | canonical relative paths only; tensors cover the whole ledger and primary qnames; codebooks equal selected basis assets; runtime equals selected candidate contracts; byte scope/ceiling equal Target |
| QualificationEvidence / DeviceQualification | exact export/runtime/device/workload/placement; check ID/outcome/measurement; Target; nonempty required checks | evidence cannot replay across any binding; required IDs equal Target; verdict derives only from exact evidence outcomes |
| MarketSnapshot | source-content refs for survey observations; observation date; normalized device IDs; counts/shares plus scope and collection method | raw observations remain separate by source; Hugging Face users are not presented as installed-base counts; Steam shares are not converted into absolute units |
| ReleaseDecision | CollectionContract and MarketSnapshot refs; policy identity; included/rejected rows | variants exactly partition the bound contract; included solve lineage matches it; no rejected qualification is included; aliases remain outside upstream scientific identity |

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
duplicate references also fail. Cross-record verification additionally
rejects semantic-reference equivocation, one physical digest with inconsistent
sizes, missing owned records, and duplicate records. `write_record` publishes via a no-clobber
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

The schema encodes no intrinsic minimum, maximum, or contiguous rung range.
K1, a future research K32, holes between them, and overlaps with FP8 are all
representable as explicit catalog members. That data-model capability is not
format authority. The current public NVFP4 reader and interim producer catalog
is exactly K12–K24; direct codec/lattice widths outside that interval have no
parser, registry, contract, chooser, assignment, bundle, or export identity.
Promotion policy remains external: lattice-first is the safe
production default, while a learned candidate must earn its extra asset,
training, and runtime complexity under the same target and qualification
contract. Learned NVFP4 remains receipt-gated and refused today.

The same separation applies to W8A16/source FP8. The generic schema may census
or reference a compatibility candidate for an already-published source-model
artifact without making it a maintained candidate for a new target. The
Qwen3.8 SM120/RTX50 collection is BF16-sourced and its exact serving profile
explicitly denies `W8A16_COMPAT_FORMAT_NAMES`; its catalog-to-assignment and
materialization steps must therefore intersect both source applicability and
that profile before recording an accepted Solve/Export/ReleaseDecision. The
current resident and streaming CB materializers already enforce the assignment
half when `layer_config.json` carries its allocator-stamped `target_profile`:
they refuse a denied format before opening an output transaction. Target v1 is
not pipeline-wired and does not yet perform the catalog intersection itself, so
free-form `exclusions` are not format authority and no collection record may
claim the earlier gate has run without its own receipt. Legacy readers remain
broad; maintained performance eligibility is target-specific and fail-closed.

The range study retains a **direct-codec/kernel research span through K32**,
including widths outside the public interval, but the evidence-backed public
artifact catalog is **K12–K24**. With two four-value-vector product subtables,
the index split is `(ceil(K/2), floor(K/2))`; K1 is therefore a valid `(1, 0)`
research geometry, not malformed metadata. An E2M1 value
has 15 distinct numeric possibilities (the two signed-zero encodings compare
equal), so a four-value tuple has `15^4 = 50,625` distinct numeric vectors. A
width-16 table can therefore fill 65,536 physical rows only by deterministic
duplicate fill. K32 is still useful for a low-level uint32 packing boundary
test, not as a shippable format.

The sidecar grows much earlier.  An FP16 sidecar product dictionary occupies
`2 * (4*2^ceil(K/2) + 4*2^floor(K/2))` bytes: 65,536 at K24, 98,304 at K25,
and 131,072 at K26. The present GB10 full-dictionary staging budget is 101,376
bytes (`gridbook/csrc/cb_gemv_v2.cu`), making K25 the last rung compatible with
that whole-dictionary staging architecture. That structural ceiling is not
format authority. Gridbook's generic global-LUT kernel can exercise larger
direct-codec words, but there is no legacy artifact reason to preserve widths
outside K12–K24 as public wire ids. A compatible external v11 contract must
therefore declare both NVFP4 `rungs` and `producer_rungs` as K12–K24, and every
eligibility cell must be a subset. The sealed untagged 0.9.1 candidate predates
this contraction and is deliberately rejected pending a newly committed
Gridbook package with a new reviewed identity. Its sm120 evidence is only
`compile_only`: dense and
routed decode are `backed`; routed batch is `backed` only for
`role_split=false` through persistent-B, with a generic expand-to-BF16
`fallback`; dense batch is likewise only the expand-to-BF16 `fallback`.
Accordingly:

- enumerate and export K12–K24 as explicit artifact candidates;
- keep widths outside K12–K24 confined to direct-codec, lattice-asset, and
  kernel research tests with no public format spelling;
- advertise every new band only as validation-only while its exact target,
  structure, regime, rung, and predicate winner remains `compile_only` or
  `fallback`;
- promote no rung to a shipping chooser until both decode and batch have exact
  `device_qualified` backed winners and physical parity/speed gates pass; and
- keep lattice and learned versions conceptually distinct at every public rung,
  with lattice the baseline and learned promotion driven by measured artifact
  quality after its shared asset cost is charged exactly once; the current
  NVFP4 producer refuses learned bases until that receipt exists.

## 5. Byte accounting

An assignment maps a unit ID to a candidate ID. Its `local_bytes` must equal
the selected usable CostSnapshot cell, its shared resources must equal the
Candidate, and Solve fixed resources must equal Target. The recomputed
`assignment_byte_breakdown` deduplicates shared resources by physical SHA-256.

Export then carries a complete portable file inventory. Its byte measurement
must equal the sum of every file, use Target's exact named scope, and stay
under Target's ceiling. Tensor units and primary qnames reconcile to the whole
source ledger, while codebooks and runtime identity reconcile to selected
candidates. This is the authoritative record model; a pipeline adapter still
must hash and stat the real filesystem to populate it. Candidate-local bytes
never substitute for the final recursive package measurement.

### 5.1 Hardware tiers and export shoulders

As observed 2026-08-24, NVIDIA's desktop specifications group the relevant
cards into 8 GB (RTX 5050/5060 and one 5060 Ti SKU), 12 GB (5070), 16 GB
(the other 5060 Ti SKU, 5070 Ti, and 5080), and 32 GB (5090). Laptop parts are
predominantly 8 GB through 5070, 12 GB at 5070 Ti, 16 GB at 5080, and 24 GB at
5090. Sources:
[NVIDIA desktop comparison](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/)
and [NVIDIA laptop comparison](https://www.nvidia.com/en-us/geforce/laptops/50-series/).

Two demand proxies point at different but complementary audiences. The live
[Hugging Face hardware page](https://huggingface.co/hardware) gave the following
per-SKU owner incidences; its
[methodology](https://huggingface.co/docs/hub/en/hardware) makes this opt-in,
self-reported public-profile telemetry rather than unique people, sales, or a
population estimate:

| reported SKU | VRAM variants | HF incidences |
|---|---:|---:|
| RTX 5050 desktop / laptop | 8 GB / 8 GB | 255 / 237 |
| RTX 5060 desktop / laptop | 8 GB / 8 GB | 3,625 / 1,010 |
| RTX 5060 Ti desktop | 8 or 16 GB | 13,470 |
| RTX 5070 desktop / laptop | 12 GB / 8 or 12 GB | 6,104 / 1,260 |
| RTX 5070 Ti desktop / laptop | 16 GB / 12 GB | 13,530 / 964 |
| RTX 5080 desktop / laptop | 16 GB / 16 GB | 9,264 / 1,046 |
| RTX 5090 desktop / laptop | 32 GB / 24 GB | 16,538 / 994 |
| RTX 5090 D desktop | 32 GB | 498 |

Those rows total 68,795 incidences, 20.78% of the 331,048 summed NVIDIA-SKU
incidences on the page. They are not a unique-user denominator. Valve's July
2026 displayed RTX 50 rows instead sum to 16.60% of surveyed video cards. The
[Steam survey](https://store.steampowered.com/hwsurvey/videocard/?sort=name)
is optional and anonymous, publishes no respondent count, and combines such
capacity variants as the 8/16 GB 5060 Ti. Allocating ambiguous identifiers to
either possible capacity gives the following honest ranges:

| capacity | HF share of RTX 50 incidences | Steam share of all displayed cards |
|---:|---:|---:|
| 8 GB | 7.5–28.9% | 5.79–8.77% |
| 12 GB | 10.3–12.1% | 3.99–4.57% |
| 16 GB | 34.7–54.2% | 3.41–5.81% |
| 24 GB | 1.4% | not separately listed |
| 32 GB | 24.8% | 0.43% |

These are a dated snapshot of mutable pages. A release campaign must store
the source URL, observation time, extraction method, and raw receipt in a
MarketSnapshot rather than silently treating these numbers as a current
installed base.

The solve grid should be dense from 5 through 25 decimal GB because one probe
can feed many cheap solves and exports. Publication should expose a small,
named collection of physically qualified shoulders:

| artifact alias | target tier | whole-package target | publication intent |
|---|---:|---:|---|
| `vram-8-stretch` | 8 GiB | 6.0–6.5 GB | Experimental small-context reach build; ship only if full GPU-only residency and eager/graph, KV, KL, PPL, and performance gates pass on real 5050/5060 hardware. Otherwise publish a smaller model, not a nominally fitting 27B. |
| `vram-12-compact` | 12 GiB | at most 10.0 GB | Primary compact 27B build, leaving about 2.7 GiB for runtime, activations, and a modest KV cache after exact resident/cold-load measurement. |
| `vram-16-balanced` | 16 GiB | at most 13.0 GB | Primary broad-reach quality build; the current 12.98 GB oracle is the first regression target, not automatic qualification. |
| `vram-24-quality` | 24 GiB | 19–20 GB | Higher-quality build with materially larger runtime/context headroom than a capacity-filling export. |
| `vram-32-max-quality` | 32 GiB | approximately 23–25 GB | Stop at measured quality saturation rather than filling VRAM for its own sake. |

A 10.0 GB decimal package is 9.31 GiB before runtime allocations, so it cannot
be a fully resident 8-GiB artifact. It is a conditional 12-GiB target and a
comfortable capacity target at 16 GiB or above. Because export is cheap, every
solve shoulder may be materialized, but ReleaseDecision should normally publish
at most one headroom and one quality artifact per tier based on physical
qualification and measured quality—not on a round marketing size. The 8-GiB
tier is the reach objective; 12 and 16 GiB are the credible initial shipping
tiers for a 27B model.

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

The schema and graph verifier now exist. Remaining work is integration:

1. derive UnitLedger/ModelSnapshot from `model_walk` and checkpoint headers;
2. publish ProbeCampaign and exhaustive CostSnapshot records from the live
   probe/reducer and cost pipeline;
3. make AQUA emit Solve records and exporters emit filesystem-derived Export
   records rather than reconstructing either after the fact;
4. turn physical device runs into typed QualificationEvidence and exact
   DeviceQualification records;
5. ingest dated market receipts without changing scientific identities; and
6. have release policy create an exhaustive ReleaseDecision and require the
   closed-graph verifier before publication.

Until those adapters and exact hardware qualifications land, this remains an
offline control-plane API, not a claim that any new rung, artifact size, or
device route is supported.
