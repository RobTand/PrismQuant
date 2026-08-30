# FP8-CB learned-v2 quality contract

Status: opt-in research/production-candidate plumbing. The default bundle
trainer remains v1 for compatibility. Within learned-v2, every FP8-CB rung
defaults to the canonical lattice until a model-specific promotion receipt
passes validation.

## Calibration source

Learned-v2 uses the imatrix values already accumulated by the sensitivity
probe:

- dense Linear: `act_sq_sum / n_tokens_seen`;
- routed population: `expert_act_sq_sum / expert_tokens`, independently for
  each expert.

`prismaquant.cb_imatrix` converts those existing fields into the qname-to-tensor
mapping accepted by the renderer and records a canonical SHA-256 over sorted
qnames, shapes, and little-endian FP32 values. It creates no cache and reads no
external corpus. A zero-token routed expert must first go through the existing
neutral-prior synthesis; missing calibration is never silently turned into a
zero or unit weight.

The bundle builder accepts either the legacy `--col-weights` pickle or an
existing trusted `--imatrix-probe probe.pkl`. A learned-v2 promotion receipt
requires the probe form: its actual calibration hash and the canonical digest
of the derived values are checked against the receipt. An unpromoted all-
lattice v2 build and every v1 build retain the legacy input compatibility.

## Sampling and training

For one `(qname, rung)` cell, let `w` be the largest of its four product
subtable widths. Learned-v2 computes:

```text
entries         = 2**w
vectors_per_row = in_features / 8
target_rows     = ceil(64 * entries / vectors_per_row)
sample_rows     = min(output_rows, max(64, target_rows))
vector_cap      = 2,000,000 pooled vectors
```

Rows come from a stable SHA-256(qname)-seeded CPU permutation. All rungs for a
qname therefore take prefixes of the same permutation. Vector-cap selection is
separately keyed by qname and rung. Every trained cell records the selected-row
digest, selected-vector digest, requested and achieved density, and the exact
number of missing vectors when it cannot reach 64 vectors per largest-subtable
entry. A density shortfall fails that cell closed to lattice. It is not treated
as an assumed crossover for adjacent rungs or transferred to another model.

The v2 Lloyd update stably sorts assignments, preserves original vector order
inside each centroid, and uses contiguous segment reductions instead of
`index_add_` floating-point atomics. The v1 and universal-lattice paths retain
their original atomic implementation and identity.

Exact regeneration is required only for repeats on one fixed software build
and device. Ada and Blackwell can legitimately use different reduction
implementations or floating-point instruction sequences, so cross-architecture
training digest equality is not a ship gate. The canonical FP16 table digest in
the immutable `.pqcb` artifact remains the architecture-independent render
identity. Each trained cell stamps this repeat scope together with its PyTorch
version, CUDA build, device type, and CUDA compute capability when applicable.

Canonical producer lattices are loaded only from the committed, whole-file
SHA-256-pinned asset. A missing producer key fails closed. The maintenance
generator explicitly builds missing canonical candidates on CPU; a result is
not canonical until its asset digest pin is reviewed and updated. The
GPU-when-available cache-miss builder remains available only for non-production
research keys and is not a regeneration contract.

## Per-rung promotion receipt

`prismaquant.fp8_cbl_promotion_receipt.v1` contains all 12 rungs in the
K4, K8, ..., K48 ladder. It binds the result to one exact source model id and
content SHA-256, the v2 trainer, the actual probe calibration hash, the
training imatrix value digest, and exactly two distinct held-out calibration
sets. Holdout calibration identifiers are lowercase SHA-256 values, are
distinct from one another, and must also differ from training. The bundle
embeds the complete value-bearing streamed-model identity, including its
checkpoint tensor-to-shard map; a name stamp or decoder-only identity cannot
authorize promotion. The receipt does not contain a learned/lattice crossover
or require rung decisions to be monotone.

The receipt also closes the candidate surface. `candidate_codebooks` has
exactly the model-derived target qnames, exactly all 12 rungs per qname, and
exactly one SHA-256 per canonical product subtable. Before bundle publication,
every learned table is canonicalized to its shipped FP16 bytes and compared
with those digests. This applies equally to newly trained, pretrained, and
banked routed books, so an unmeasured replacement cannot ride on another
candidate's quality result. Bundle loading repeats the receipt-to-cell digest
check.

Each holdout must pass every threshold:

| Measurement (learned / lattice) | Promotion threshold |
| --- | ---: |
| Geometric-mean unit ratio | `<= 0.98` |
| 95% bootstrap upper bound | `< 1.00` |
| Every role aggregate ratio | `<= 1.01` |
| Unit-ratio p95 | `<= 1.05` |
| Worst unit ratio | `<= 1.10` |

Each holdout's role-ratio keys and positive coverage counts must exactly equal
the role census derived from the bundle's target qnames. A nonempty,
self-declared subset is insufficient.

The repeat delta must also be `<= 2.0` percentage points. Both holdouts and the
repeat check must pass, and `density_shortfall_cells` must be zero; a tie or
sampling shortfall stays lattice. The receipt's declared source must agree with
the metrics for every rung or the whole receipt is rejected. A receipt can
therefore promote K40, retain lattice at K44, and promote K48 if those
independent cells earn that result.

The receipt is embedded in v2 bundles, and its content digest is copied into
each compact rung-policy row. Manually listing learned formats cannot override
the receipt. With no receipt, learned-v2 writes an all-lattice bundle.

## Builder invocation

```bash
fp8_formats="FP8_CB_K4,FP8_CB_K8,FP8_CB_K12,FP8_CB_K16,FP8_CB_K20,FP8_CB_K24,"
fp8_formats+="FP8_CB_K28,FP8_CB_K32,FP8_CB_K36,FP8_CB_K40,FP8_CB_K44,FP8_CB_K48"
python3 -m prismaquant.build_cb_learned_bundle \
  --model-dir /path/to/model \
  --imatrix-probe /path/to/probe.pkl \
  --formats "$fp8_formats" \
  --trainer-version v2 \
  --promotion-receipt /path/to/promotion.json \
  --source-model-identity-cache /path/to/streamed_model_identity.json \
  --output /path/to/model.pqcb
```

This command remains on the existing streaming weight-provider path: one
decoded source Linear and its resident imatrix are consumed at a time, and only
the small canonical FP16 books accumulate. The identity cache must already
validate against the currently existing complete local safetensors checkpoint;
the builder checks file fingerprints and index coverage without creating a new
cache or rereading the checkpoint contents.
