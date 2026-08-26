# PrismaSnap additive source-preparation contract

Status: **candidate**, pending the Qwen3.8-27B text-only 20 GB served A/B.  The
older 0.6B/4B SnapQuant results motivate this gate; they are not claims about
27B or tomorrow's Qwen3.8-125B-A3B checkpoint.

## Boundary

PrismaSnap is an offline BF16 checkpoint-to-checkpoint transform.  It is not a
quantization format, allocator, cache, export layout, serving kernel, or
runtime adapter.  A missing `prismasnap_provenance.json` preserves the
historical pipeline.  A verified snapped checkpoint enters the ordinary
probe, AURA cost, per-Linear allocator, `ProductionWeightCache`, recache,
compressed-tensors export, and validation stages without another PrismaSnap
decision.

The nominal v1 search reproduces Fable's fast `stage,polish` treatment, not
the original sequential greedy search.  Its value-bearing receipt fixes:

- group size 16 and α candidates `0,.125,.25,.375,.5`;
- at most four fixed-global rounds, first-round top-half staging, and
  top-8/at-most-16 true-render polish;
- static-6 NVFP4, fp32 candidate folds, and one global per logical tensor;
- the prototype's sequential BF16 rounding order when a tensor carries two
  folds; and
- a true-render no-op upper bound.

Fused-sibling scoring and one-final-cast materialization are different,
unmeasured algorithms and are not aliases for that search.

### BF16-realized v2 execution

The first 27B served fold-fidelity run exposed a distinction the nominal v1
objective did not encode: Qwen's offset-one RMSNorm can round a small effective
gamma to zero when the nominal fold is cast to BF16.  Applying that nominal
scale to consumer columns is then not the inverse of the bytes the norm
actually executes.  Pre-cast fp64 algebra therefore remains a useful graph
gate, but is not an executed-BF16 fidelity proof.

Evidence is preserved under
`/home/rob/dq-runs/prismasnap-qwen38-27b-20gb-20260825/fold-fidelity/`:
the v1 snapped-BF16 arm measured all-position KL `0.002956950685081059`
(`student.json`) against the `5e-4` gate, while the identical source-vs-source
replay was exactly zero (`source_aa.json`).  This is a failed prerequisite,
not a quantized-quality result; no v1 checkpoint from that run is admissible.

`realize-bf16` is a deterministic derived-plan transition over one complete
merged v1 plan and its original BF16 source.  It does not search again.  For
each input/post-norm seam it stores three different, content-bound payloads:

- the parent nominal float64 scale (for audit, never silently relabelled as
  executed);
- the selected projected BF16 norm bytes, which materialization writes
  directly with `replace_bf16`; and
- a positive float64 consumer inverse derived from the quotient of projected
  and source effective gammas.

A channel executes identity when its source gamma is zero, its projected gamma
is zero, either value is non-finite, signs differ, or quotient reapplication
does not reproduce the selected BF16 norm bit-exactly.  The norm still uses the
direct projected payload so a quotient/product BF16 tie can never change it.
The transition then recomputes the full static-6 NVFP4 objective on the actual
`BF16(W / s_real)` consumer weights with importance multiplied by
`s_real**2`.  A seam with no strict improvement falls back entirely to the
source norm and identity inverse.  Dense up/down executes identity in v2; its
small nominal gain is retained only in the parent-search record.  Rejected
norm seams and all up/down seams emit no materialization transforms, so the
changed-tensor census describes intended executed byte changes rather than
identity rewrites.

The v2 plan schema/algorithm, projected-payload hashes, per-channel reason
counts, exact realized objective, parent plan/scales/producer identities, and
source identities are all self-hashed.  v1 plans remain readable for audit and
as the sole admitted parent of this transition; a v2 plan cannot enter the v1
worker-plan merge.  Materialized v2 provenance reports executed groups and
executed objective improvement; nominal parent movement remains only under the
explicitly labelled parent seam records.

## Dense exact seams

For each decoder layer, graph discovery uses equal activation-stat signatures
from the existing probe and profile mappings; it does not accept a Qwen name
allow-list as graph proof.

1. Input RMSNorm effective γ is multiplied by `d`; every direct attention or
   Gated-DeltaNet input projection column is divided by `d`.
2. Post-attention RMSNorm effective γ is multiplied by `d`; dense MLP gate/up
   columns are divided by `d`.
3. Dense up rows are multiplied by `d_ud`; down columns are divided by
   `d_ud`.  The gate is unchanged.

Qwen3.5/Qwen3.8 stores `p = γ - 1`, so its norm parameter transition is
`p' = (p + 1)d - 1`.  Profiles that do not explicitly declare an RMSNorm
parameter offset fail closed.  Q/K, V/O, DeltaNet output, final norm/head,
embeddings, vision, and MTP seams are outside v1.

Every transformed source tensor must be BF16.  FP8-source refolding is not an
additive exact-source operation and is refused.

## Deterministic state machine

`tools/prismasnap.py` exposes the application state machine:

```text
source identity + probe
  -> plan-dense (partial layer plans allowed)
  -> merge-plans (exact layer union)
  -> realize-bf16 (merged v1 + original BF16 -> executable v2 plan)
  -> materialize or materialize-part
  -> merge-checkpoint-parts (exact shard union)
  -> MATERIALIZED BF16 checkpoint
  -> served BF16 fold KL
  -> attest-fold-fidelity
  -> VERIFIED BF16 checkpoint
  -> ordinary PrismaQuant pipeline
```

Plans bind source config/index/shard content, safetensors header metadata,
probe bytes/calibration, profile code, search semantics, container rootfs, and
producer source bytes.  Part receipts bind source/output bytes, tensor census,
shape/dtype, and the exact transform count.  Merge operations require disjoint
exact covers.  Every long transition has no-clobber staging, fsynced receipts,
identical-argv `--resume`, bounded campaign retries, and committed-output
revalidation.

The executable v2 transition is:

```bash
python tools/prismasnap.py realize-bf16 \
  --source /original-bf16 \
  --plan /merged-v1-plan \
  --output /bf16-realized-v2-plan \
  --device cuda --resume
```

`prismaquant.cluster_campaign` is the non-agent coordinator.  A strict,
self-hashed manifest declares hosts, argv, dependencies, timeouts, retry
limits, and expected receipts.  Local and SSH workers receive explicit argv
without a shell; locks, PID ownership, atomic CAS state, watchdogs, and sealed
stage receipts drive recovery.  An operator or agent may author the manifest,
but no intelligence participates after launch.

## Admission and lane gates

Materialization proves tensor algebra and census and emits state
`MATERIALIZED`, never `VERIFIED`.  `attest-fold-fidelity` consumes the standard
all-position `measure_vllm_full_kl.py` result against the original BF16
teacher.  It replays both serve fingerprints, producer identity, teacher
payload tensor contract, calibration token/window/corpus identity, metric
coherence, checkpoint index, and every current shard hash.  Forward KL must be
at most `5e-4`.  Both `run-pipeline.sh` and the native exporter re-hash the
verified BF16 input before work.

Only the native compressed-tensors `{NVFP4, FP8_DYNAMIC, BF16}` lane is
admitted.  GGUF and Gridbook/codebook exporters fail closed on any PrismaSnap
source marker; the pilot found fixed-book interaction harmful.  The native
export carries the receipt as `source_prismasnap_provenance.json`, because its
hashes describe the BF16 input, not the compressed output tree.

## Promotion measurement

The first production gate is Qwen3.8-27B, text-only/no vision, strict decimal
20,000,000,000-byte target, using the same 16×512 diverse-v1 probe contract,
frozen per-Linear assignment semantics, unsnapped BF16 teacher, cache settings,
and gold 8×512 all-position KL/PPL protocol as the control.  The primary target
is at least 20% lower all-position and confident-position KL with no material
runtime cost and the same body-bpp/accounting contract.  Fold KL ≤`5e-4` is a
prerequisite, not the quality result.

The MoE adaptation is the second lane.  It must independently prove router and
shared-gate compensation, packed expert slice/axis semantics, per-expert
importance and batched codec equivalence, route stability, partial-source disk
lifecycle, and an actual release-layout census.  Dense results do not promote
MoE automatically.
