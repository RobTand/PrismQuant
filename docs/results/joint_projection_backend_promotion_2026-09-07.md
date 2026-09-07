# Qualified opt-in joint projection backend — 2026-09-07

The existing joint AURA path now has an explicit `fused_fp32_v1` backend. The
native `torch` expression remains the default. Promotion preserves the
[measured exact-tree kernel](joint_projection_fused_reduction_2026-09-07.md)
and adds runtime admission, prewarming, versioned arithmetic identity and
production lease wiring. No format menu, serving gate, calibration or allocator
objective changes.

## Selection and admission

The joint anchor plan's `execution` may include:

```json
"projection_backend": {
  "name": "fused_fp32_v1",
  "binary": {
    "path": "/mnt/shared/tessera-measurements/first-model-20260907/joint-fused-projection/actual-ab-01/pq_joint_projection_reduce_17d100e93d552b85.so",
    "sha256": "9305c183c5214dc5ff1f73382963f8275eb1b6197cb8d173c6a93bffd700c115"
  }
}
```

Omission or `{"name": "torch"}` selects the reference. Unknown selectors,
extra fields and unqualified binary hashes are rejected. `prewarm_projection_backend`
checks the packaged qualification's exact Torch/version/git/CUDA, critical
reduction headers, compiler identities, source/compiler flags, device name,
compute capability and SM count. The binary file and actually loaded extension
must match the qualified SHA256. Production loads this prebuilt artifact; it
never silently JIT-builds another binary or adopts a different runtime.
The qualified environment is the original pinned container, Torch
2.13.0+cu130 at `cf30153c4c131c8164ee7798e5022d810682e2cb`, CUDA 13.0 and
48-SM GB10 / compute capability 12.1. The package JSON contains complete fields.

Prewarm executes a transient zero-matrix exactness check on the selected device
before source/cotangent hot execution. Only immutable qualification metadata,
loaded code modules and device markers persist; no new tensor cache exists.
Repeated row admission uses that metadata without reopening its file. A lease
requires the prewarmed handle, refuses a different device, and uses the backend
at exactly the three weight, activation and mixed product/reduction sites.
Ineligible alignment/layout/dtype retains the reference expression, while an
unqualified matrix shape is refused. The observer's `no_grad`, GEMMs, QDQ,
invocation/format/probe order and signed accumulation remain unchanged.

The streamed API accepts `joint_projection_backend=<selector or prewarmed handle>`.
A direct `SignedJointProjectionLease` accepts only
`projection_backend=<prewarmed handle>` for the fused path. Runtime loading,
hashing and compilation cannot be triggered implicitly inside its projection
callback. All production source files, CUDA/C++ files and qualification JSON are
already included by the existing complete-package AURA implementation digest.

## Identity migration

Prepared completion/PWC metadata now binds the resolved backend identity, and
prepare/run compare it exactly. Probe arithmetic includes its qualified runtime
and loaded build identity. Prepared, probe, operator and joint-run schemas are
**v2**; legacy v1 signed rows and prepared records fail closed. Users must run
fresh preparation and recompute costs, even when choosing the reference backend.
The v1 plan remains readable with its reference default. There is no conversion
of old signed rows, prepared adoption, or checkpoint identity override.

Historical prepared artifacts used below are read-only inputs to bounded
numerical qualification. They are never passed through the production run's
prepared-cache admission as though produced by this implementation.

## Qualification

All six source matrix shapes in the original 2142-unit first-model census are
covered. The original actual-source expert-w1 capture retains its real Fisher
cotangents, four complete B1 source sequences, actual invocation row counts
137/102/74/80, and all seven original prepared rungs. The five added geometry
cases use actual source weights, canonical captured X and original seven-rung
renders, with explicitly seeded BF16 qualification cotangents; those cases do
not claim source-model Fisher costs or model-quality measurements.

| Representative unit | Matrix shape | Captured X | Exact reductions |
|---|---|---|---:|
| Layer 23 expert 0 w1, actual source cotangents | 1792×2048 | 16 actual invocations across four probes | 208 |
| Layer 0 dense w1 | 7168×2048 | 512×2048 | 52 |
| Layer 0 dense w2 | 2048×7168 | 512×7168 | 52 |
| Layer 23 expert 0 w2 | 2048×1792 | 512×1792 | 52 |
| Layer 10 attention k_proj | 512×2048 | 512×2048 | 52 |
| Layer 10 attention q_proj | 2048×2048 | 512×2048 | 52 |

Each case preserves all four probes × seven formats' weight/activation/mixed/
total signed components and forward/input-gradient/weight-gradient hashes.
Individual reductions compare FP32 integer bits; no tolerance was relaxed.
The actual-source gate was then repeated through two real production leases,
constructed with separately selected reference/fused prewarmed handles:
**208/208 exact reductions and all signed/forward/backward hashes match**.

The final CPU suite passes **297 tests, zero skips** across backend refusal,
legacy prepared/signed-record rejection before cache adoption, streamed/packed/
microbatch/lease/probe policies, downstream row consumers and architecture
checks. Touched production modules compile. This includes a regression ensuring
repeated serialized row admission performs no qualification-file I/O. The
initial 296-test pass preceded that final metadata-cache refinement; both
receipts are retained. Fifty-six existing Torch JIT deprecation warnings are
reported; no test is omitted. The original 22 CUDA reduction-tree tests remain
recorded in the preceding result and were not unnecessarily repeated.

## Production-path profile

A separate admitted profile, without another steady ABBA run, exercises three
cycles per arm through the production backend directly. The individual-bit
instrumentation is removed during profiling. All inputs and rendered weights
are resident through the existing prefetch owners.

| Three-cycle CUDA profile | Reference | Qualified production backend |
|---|---:|---:|
| Large product kernels | 624 / 110.846 ms | 0 |
| Reference sum kernels | 624 / 12.812 ms | 0 |
| Fused reductions | 0 | 624 / 71.066 ms |
| All kernels | 5280 / 158.165 ms | 4656 / 102.846 ms |
| `cudaLaunchKernel` calls | 4992 | 4368 |
| DtoH copies / stream synchronizations | 84 / 84 | 84 / 84 |

Both hosts' raw Netdata CPU, load, RAM, available-memory, host-I/O and GPU-power
series are retained for this action. The subsecond profile windows are shorter
than the ten-second power sampling period, so they support no new work/joule
claim. The earlier fixed-work candidate ABBA remains the attributable
**1.4731× local replay speed / 1.3751× work per GPU joule** result; production
full-model throughput and energy await the fresh full-cost run. This profile
shows that the admitted production path removes the same large intermediates;
it does not establish GPU saturation or a full-model speedup.

## Attributable artifacts and reproduction

All artifact paths below are relative to
`/mnt/shared/tessera-measurements/first-model-20260907/joint-fused-projection`.
Every cited PB terminal, actual CAS payload/hash, exit status and successful
resource cleanup was independently checked. All actions returned zero.

| Action | PB action key | Verified actual CAS payload SHA256 |
|---|---|---|
| Final CPU suite, `promotion-cpu-02` | `90f197b7ea1bb4fb97189dfe767ff6c367f693b2808c421a78615ef174f3b410` | `c32f93f5f8681a518ca46bbfe09aafb77cf862e86f975411d1813bb52b05ba0c` |
| Production actual-source gate, `promotion-actual-01` | `a95e957a99aa9c8ca98a324c2abc7c0ed8c8e27268e4819f0e5d49ba2d82a329` | `e55ffd75143ccf5073414eebcfcfc2041cabf6c6c5e9cbc996de08f13627bf59` |
| Production profile, `promotion-profile-01` | `87287a3175a7443e355eaef7d31d929b55cf24689f9965b6c0e22f923e3c4f37` | `8cb2c9ea41c1ce29cb985f8cd124ed65e91e747bed0b5c62c58803223321aabc` |

`geometry-01-campaign.json` records the five independently submitted geometry
commands. Per-unit `geometry-01/<unit>/verified-pb.json` binds each PB action,
terminal and CAS payload. The packaged qualification's `evidence` records
those action keys and explicitly named `artifact_receipt_sha256` fields for
the numerical receipt files, distinct from PB's own CAS receipts.

`promotion-actual-01/receipt.json` SHA256 is
`0810adeba9bc4598dbacff48eda029a6159eeb05ebe81604c068767c52b24064`.
`promotion-profile-01/receipt.json` SHA256 is
`ea62d63f9ec4ac88e8d796d197baccb45a437cd8a818d674c1af22928f80c64b`.
That profile directory also holds both CPU/CUDA tables, Chrome traces,
`profile-analysis.json`, `netdata-both-hosts.json`, the actual `.so`, build recipe,
and verified PB record. `promotion-cpu-02/pytest.xml` records every test and the
actual CPU-only mode. Source manifests preserve the actual sealed PB snapshots,
including final CPU snapshot `2ea0186de817b13292f63afb2d494460de2ae704`.

The actions use the original pinned container, native thread bounds of one and
four PB-assigned CPUs. CPU and GPU actions reserve 12 GiB total memory; GPU
qualification/profile actions additionally declare an 8 GiB GPU subset budget.
Docker retains PB's assigned affinity. CPU qualification uses four pytest
workers. The profiler action uses PB measurement isolation; numerical geometry
and production qualification remain portable between eligible GB10 workers.

To repeat only the production-path exact gate with an unused output directory:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --gpu --tag gb10 --cpus 4 --demand mem_gb=12 --gpu-memory-gb 8 \
  --timeout-s 600 --detach -- \
  bash experiments/joint_projection_reduce_run.sh production \
  --output /mnt/shared/tessera-measurements/first-model-20260907/joint-fused-projection/production-repeat \
  --source-receipt /mnt/shared/tessera-measurements/first-model-20260907/qdq-residency/source-capture-01/receipt.json \
  --source-sha256 bf5337381562e2b4ae1fbdb672e85f198679e2848487b291590d7810990872d3
```

Add `--profile-qualified` after `production` and use PB `--measurement` for the
short before/after profile. `policy_tests --output <unused-path>` with CPU-only
PB admission runs the recorded contract suite. Source-model assets, the original
prepared and capture artifacts, and the explicitly qualified binary are required;
missing or foreign artifacts fail closed.
