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
