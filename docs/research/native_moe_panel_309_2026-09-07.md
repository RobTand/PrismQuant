# Whole routed native panel boundary (#309)

This is the PQ side of one complete 32-expert LFM routed observation. It is not
an accepted runtime table or a native qualification result. All 96 logical
source Linear rows retain additive unary joint AURA costs; one existing
`RuntimeBinding` joins their operator identities to a single whole-stack native
invocation. No new scalar group cost or sum of leaf timing samples is used.

`prismaquant/native_moe_panel.py` owns source/PWC/wire and routing reference
bindings. Tessera's separate `experiments/bench_native_moe_operator.py` owns the
native preparation, execution, single-apply CUDA event samples, in-process
profiles and native memory collector. PQ never imports the serving package.

The reference applies the existing packed-expert operation and shared E4M3
activation QDQ at the input and intermediate GEMM boundaries. The scoped
protocol requires `PRISMAQUANT_PROD_ACT_SCALES=0`; this does not alter the global
default. The source sigmoid router's final weights are used as captured,
including BF16 rounding and selection-only FP32 expert bias. Int32 IDs and FP32
weights are supported as transport only after an actual lossless round trip.
The prefill input is the first complete calibration sequence; decode M=1 uses
its first row, an operator shape check rather than autoregressive generation.

Canonical checkpoint missing-state initialization and eager attention are
required in the raw PAC boundary and shared capture-v2 manifest. The old
captures with suppressed nonpersistent-buffer initialization are quarantined.
No metadata edit can qualify their old tensor values. Source files, source
configuration, auxiliary files, runtime config and exact calibration draw are
bound to the eventual member joint probes.

## Artifact adapter

`experiments/pq309_native_moe_panel.py` has three subcommands. Each existing
input path is paired with an independently supplied SHA256. Output files are
created exclusively and refuse replacement.

- `prepare --plan PATH --plan-sha256 SHA --out DIRECTORY` verifies the existing
  shared PAC/capture and existing ProductionWeightCache plus original wires.
  It prefetches the full selected Hessians, re-derives producer encoding
  identities independently from source/H, and verifies each original wire's
  decode against the actual PWC render before computing references. It writes
  `inputs.json`, `request.json`, `tensors.safetensors`, and a copy of the plan.
- `freeze --inputs PATH --inputs-sha256 SHA --preflight PATH
  --preflight-sha256 SHA --cost PATH --cost-sha256 SHA --out PATH` selects all
  96 actual member rows from the shared cost payload and binds the producer's
  untimed native facts into one immutable panel.
- `consume --receipt PATH --receipt-sha256 SHA --panel PATH --panel-sha256 SHA
  --memory-trace PATH --out PATH` validates the real producer observation.
  Without a complete bound, scratch stays unknown. Layer residency, persistent
  WorkspaceManager storage and incremental scratch remain distinct. Fixed
  model costs and cross-operator workspace composition always remain unknown.

The preparation plan schema is `prismaquant.native_moe_preparation.v1` and
requires these explicit fields:

| Field | Meaning |
| --- | --- |
| `unit` | Exact routed owner, initially `model.layers.2.feed_forward.experts` |
| `capture`, `capture_sha256`, `census` | Complete shared capture-v2 and its canonical census |
| `routing_boundary`, `routing_boundary_sha256` | Original PAC payload with `boundary_metadata` |
| `calibration_input`, `calibration_input_sha256` | Exact safetensors token draw |
| `production_cache`, `production_cache_sha256` | Existing PWC pickle used by the joint probe |
| `wires` | Map of all 96 names to `wire`, `wire_sha256`, `record`, `record_sha256` |
| `runtime_image` | Full immutable Docker RepoDigest |
| `serving_config`, `serving_config_sha256` | Versioned clean-runtime configuration bytes |
| `max_resident_bytes`, `max_temporary_bytes` | Explicit existing-PWC and reference-pack bounds |
| `numerics` | Predeclared `atol` and `rtol`, frozen before native outputs |
| `probe_request` | Source path/shards/config/auxiliary hashes plus `n_probes`, `seed_base`, `token_scope`, `temperature`, `distribution`, `normalization` |

All preparation, tests and batch validation run through PrismaBuild. Actual
native memory and CUDA timing use the producer's separate observer lifecycle;
performance claims require both in-process and both-host Netdata evidence.

## Validation and current limit

PB `bea7279fc72f1c8636ebe3a9cb3c38b1e898a366c50603004cd63773833029de`
ran the adapter compile check and 86 targeted checks on DL380g10, CPU-only, with
four pytest workers and native threads bounded to one. All passed, no skips,
exit 0. Receipt:
`aeb9167dcce979b0ba68b22e01b8570555b0cd0ad70c093c7a02538bdb6b3a01`.
The 1024-byte CAS result was independently rehashed to
`3c0ee7d1c554ca4328901c7100009c40113947416ebfe28cd95b608ad243246f`.

These are synthetic contract/regression checks, not measured native results.
Fresh canonical full capture, original 96-wire/PWC materialization, packed
per-Linear joint probe, and clean-runtime native whole-stack qualification are
integration dependencies still pending at this checkpoint. The source adapter
has not yet executed its actual GPU preparation. No quality, speed, resource
price, full-model allocation, export, or served claim follows from the CPU tests.


## First real capture failure and correction

The first fresh capture stopped before publishing a native boundary because
three helper uses called `profile.name()` although the real LFM profile exposes
`name` as a property. The synthetic protocol tests had not exercised that API.
The capture owner's failed PB action was
`35ab901ece41f8ef6a8f4c22672b383344e200e713f567c2afe357eb21a17c9a`.
A new regression resolves the actual registered `Lfm2MoeProfile` and invokes
the capture helper; it reproduced the same TypeError through PB
`3aa003fedc62` before the fix. Commit `73650a41` changes all three accesses.
PB `b04ac96ac1c3247ef9171eada4b6d4d24e87e13c31f4d330827ccca6d6e9fb0c`
then passed all 48 native-MoE/profile tests, CPU-only, no skips, exit 0;
receipt `a7f5047bc75c761e12ee4d16d87186e86482f0ce3352049c9eba985de4a40839`,
CAS result independently rehashed. The failed capture has no reusable native
boundary or complete canonical capture; the capture owner restarts it.

The adapter uses the campaign's projection-aware `unit_input_identity`,
independently derived from `carried_units` on the canonical census plus actual
source weights and full Hessians. Original campaign wire records are preserved
without rewrapping them as plain encoding-only records. The source weights are
read through the existing `source_unit_weight` bridge.


## Bounded first native integration screen

The coordinator froze the first joint/native integration screen at four
Rademacher probes (seeds 7000–7003), token scope `all`, temperature 1.0 and
`global_kl_fisher` normalization, using exactly sample zero of the full canonical
512 × 512 draw. This is a `[1,512]` subset screen, not full-draw quality currency.
The actual source activation/Hessian capture remains the complete canonical
draw and is never relabeled. Optional preparation fields
`probe_calibration_input` and `probe_calibration_input_sha256` must be paired.
The adapter loads both actual token tensors, verifies subset equality to full
row zero, and records the distinct parent/subset hashes and exact sample index
in `probe_scope`. This scope is carried through the immutable panel and final
observation; panel calibration identity is the subset's actual joint currency.
Tessera's producer validates the closed scope record but PQ owns the independent
parent/capture join. No native tensor or source weight is changed by this scope.

The subset binding, real profile regression, native consumers, and docs passed
87 checks plus adapter compilation through PB
`39323c408f7d01bfb22281e8a72f5fc542b8c5a7ad2c1ecc25bc6d038c6870b3`,
DL380g10, CPU-only, four workers, native threads one, no skips, exit 0.
Receipt `942023b516dd292f22403f929f22c4135c9593a64ee953858b94bcb52631a91e`;
CAS result independently rehashed. This does not qualify the pending real
subset joint/native run.


The first actual whole-reference preparation (PB
`9c50a7c9e47bb665abe0ff7a3f862342875901d44093f9fc947e09bae63cf73f`)
verified canonical boundary/subset/source inputs and prefetched all 96 full
Hessians, then stopped at the first rendered member's residency check. Existing
PWC prefetch intentionally loads disk shards onto CPU; the MoE helper had
incorrectly expected those entries already on CUDA. The fix uses the same
explicit device transfer as the dense native helper, preserving the PWC's
stored BF16 dtype and original wire bytes. No output panel was published by
the failed preparation, and no re-encoding is required.

## Actual preparation and source-qualified freeze

The device-transfer correction passed actual source-reference preparation through
PB `4aef85bed8318a002d461bede4aef940087fc859bc57489706b105fabac2deb7`
(exit 0, Sparklina). All 96 original source/PWC/wire members joined; the
1,415,632,912-byte tensor artifact was independently hashed as
`b05148dcd80f25a3a80484c02f140f751086f65ab0357719e426968c33c1ca64`.
The immutable input descriptor is
`552b6143972d3906f60109f974399ee1934add662e7f78041378e7a0975605a4`.
These artifacts are under
`/mnt/shared/tessera-measurements/first-model-20260907/native-moe-panel-r1024/prepare-01/`.
No performance improvement is claimed from this functional preparation.

Review found that semantic config serialization omits Transformers' private
attention/expert dispatch selectors. The old freeze accepted changed attention,
changed expert execution, and missing capture/probe selector identities: all four
regressions reproduced through PB `f205a17a15c2` (four expected failures).
Two earlier test launches failed collection from missing host dependencies
(`f36f28d5bc06`: PyTorch; `a6d5e00a1eaf`: compressed-tensors); the correct scoped
CPU environment is `/home/rob/venvs/pq-cpu312/bin/python` on the x86 fleet.

The fixed boundary compares the shared full module-local source execution
identity. New captures retain it directly. For the existing immutable capture,
a fresh independent source execution qualified all five original boundary
tensors (input, top-k IDs, top-k weights, FP32 bias, and sample/token coordinates)
bit-exactly with the original dtypes. Source04 also independently matched the
full-vocabulary BF16 forward, eight gradients, and actual grouped-MM down inputs.
Its 49-module source execution identity agrees with final cost02. The qualified
first sequence retains its bounded integration scope; this does not constitute
full-draw quality validation. The final panel seals the separate source04 file
hash without rewriting the original capture or prepared tensors.

The freeze also follows the actual shared streamed-model identity v1 contract:
`weight_map` maps logical names to original checkpoint tensor names, and the
shard list independently seals all checkpoint files. The full original name
roster and shard hashes must agree with capture provenance. The prior adapter
incorrectly required an additional `checkpoint_weight_map` field that the
current shared producer does not emit; an explicitly supplied map is still
checked.

After correction, 112 targeted native, calibration, capture-quarantine and docs
checks passed through PB
`76140931d27652701970c315d76c5ab31d6b0dd8ff30239a4c8f00c864b5881e`,
CPU-only on DL380g10, four pytest workers and one native thread each, no skips,
exit 0. Receipt `068788b3385db023c3e9fa89f1eec9a231cb6d08f01a7da5991eabc5f443e735`;
CAS result `e48ba03810aeb0d575ab2341d7ae647ae5c729f0901cdfe685ae4381a344efb4`
(701 bytes) independently rehashed. An intermediate green attempt and freeze
exposed set values passed to the existing JSON identity comparator; sorted
rosters corrected both, with no output from the failed freeze.

The actual CLI freeze completed through PB
`de6b7316eeae80d61f33e330c52ae581ce1f75a696833d8e64107548ca5a7a01`,
CPU-only on DL380g10, one worker/native thread, exit 0. Receipt
`1f37d2699206e55d150b80bbc54923d2cfa0b61f4cc5d78c77c457aceabada09`;
CAS result `febb6da29a0588df2f05f23fe20abe06ed17426af39bfde939821e44f02f69bc`
(106 bytes) independently rehashed. The command was
`python -m experiments.pq309_native_moe_panel freeze` with all four input-file
hashes explicitly supplied, including `--source-execution-qualification` and
its SHA-256. The PB request retains the full command.

| Frozen input/output | Independently checked file SHA-256 |
| --- | --- |
| Final `cost-02/joint-cost.json` | `fcf48abdb0e0b1d8bc324c5c938b9b68ebb821f1c5b0591aefffb4b7c3c45462` |
| Independent `source-04/results.json` | `f308e7f6fbf563884d9625721523f4247322e0bfa95876e04798438553ce8558` |
| Native `prepare-03/preflight.json` | `794829fc60af194514571ff90512e4e0ebe07f1f64d3adc3c273805d3d809188` |
| Final `native-moe-panel-r1024/panel-01.json` | `84b75f32ac47e8a77a0dfac3c3dc30a925150dcc3e3c2bf1c8878590276f1320` |

The panel uses the unchanged original E32/H2048/I1792 stack, with native runtime
identity `643a782b942c4436c354d4b1e4c34f2c79d734f6ae124c75d10ce96f7491f5b5`.
Native operator measurement and resource profiling are the next separate gate;
this frozen panel alone does not supply an allocator runtime table, full-model
resource bound, or serving qualification.
