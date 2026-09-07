# Exact-tree fused joint projection reduction — 2026-09-07

Status: measured research candidate, unused by the production call path. The
retained actual-source replay takes **51.9208 → 35.2451 seconds** for the same
work, a **1.4731× speedup / 32.1176% elapsed reduction**. Measured work per GPU
joule improves **1.3751×**. Every captured individual reduction, signed component,
and forward/backward hash matches exactly. This is a local projection result,
not a full-model throughput, quality, export, or serving measurement.

## Mechanism and numerical boundary

The prior [QDQ residency measurement](nvfp4_codepoint_residency_2026-09-07.md)
identified large FP32 products as the dominant remaining CUDA work. The candidate
extends the existing kernels namespace and retains `SignedJointProjectionLease`
and its cache wiring. The replay changes only three `(left * right).sum()` sites:
weight, activation, and mixed projections. It retains the materialized FP32
operators, all GEMMs, source invocation boundaries/order, format order, and
probe accumulation order. QDQ codepoint reuse is enabled in both arms.

The CUDA candidate uses the **running PyTorch release's**
`gpu_reduce_kernel<float, float, 4, 4>` tree, matching the loaded reference sum
kernel. Each leaf performs separately rounded `__fmul_rn` then `__fadd_rn`;
partial reductions use `__fadd_rn` without another multiplication. This avoids
the large product tensor without changing the reduction association. Compilation
disables FMA and flush-to-zero. It binds the actual binary, source, flags, Torch
version/commit, and five reduction-related header hashes in the receipt.

The fast path requires same-device, same-shape, contiguous, nonempty FP32 CUDA
operands with 16-byte-aligned pointers and bounded 32-bit indexing. Other
layouts/dtypes, empty expert inputs, and CPU inputs retain `(left * right).sum()`.
CUDA device guard and PyTorch's current stream are used. Eligible inputs that
require autograd raise unless grad mode is disabled; the observer already runs
under `no_grad`. The module caches compiled code only; callers retain input
residency. Compilation is warmed before measured work.

This is qualified against Torch **2.13.0+cu130**, git
`cf30153c4c131c8164ee7798e5022d810682e2cb`, CUDA 13.0, GB10 SM 12.1. A different
Torch reduction implementation/compiler/device is not qualified by this result.
Production promotion needs explicit supported-runtime qualification and a
prewarm contract; the research helper is not a general bit-exact replacement
across releases. No production default, pipeline stage, format, export metadata,
serving gate, or architecture contract changes in this commit.

## Workload and exact gate

The source is the retained actual forward-input and backward-cotangent capture
for the same first model, original 512×512 calibration contract, sequence rows
0–3, Fisher probes 7000–7003, and `model.layers.23.feed_forward.experts.0.w1` projection used in the
preceding result. Its 16 invocations have actual row counts **137, 102, 74, 80**
per probe and FP32 projection operators of shape **1792×2048**. The capture keeps
the actual grouped-runtime row order. The previously established per-invocation
bijective mapping to canonical activation rows is checked, but no replay X/g
permutation is introduced.

Before timing, all **208 individual product/reductions** (13 per invocation)
match FP32 integer bits. All **4 probes × 7 formats** preserve weight,
activation, mixed, total signed components and forward, input-gradient, and
weight-gradient hashes. No tolerance relaxation is used. The original
`ProductionWeightCache` loads seven rendered entries (51,396,678 bytes), with
zero misses; calibration prefetch holds 20,971,520 bytes with zero misses. The
source weight, capture payload, prepared plan, and original calibration identity
remain hash-bound by the replay harness. No additional full-model capture ran.

The targeted GPU suite has **22 passed, 0 skipped**: exact finite reductions at
14 lengths including 3,670,016 elements, cancellation/subnormals/signed zeros,
strided/transposed/misaligned/empty fallbacks, FP16/BF16/FP64 and CPU fallback,
nondefault-stream execution, and the autograd rejection/no-grad boundary. One
pytest warning reports an already imported anyio module; no test is omitted.
Nonfinite NaN payload equivalence is not established.

## Measurements

One cycle runs all four probes and their four actual invocations through the
existing signed lease, seven formats per probe. Each steady arm runs 952 cycles
(26,656 completed format/probe results). ABBA order controls short-term drift;
profiles run separately for three cycles (48 invocations) per arm.

| Arm | Seconds | Mean GPU power (W) | Integrated GPU joules |
|---|---:|---:|---:|
| Before 0 | 51.948670 | 39.886557 | 2072.053570 |
| After 1 | 35.231539 | 41.999942 | 1479.722627 |
| After 2 | 35.258659 | 42.717809 | 1506.172666 |
| Before 3 | 51.892981 | 39.194949 | 2033.942750 |

Median joules are 2052.998160 → 1492.947646; replay cycles/GPU-joule are
0.463712 → 0.637665. Power is only approximately 28–31% of the ~140 W envelope;
this is not evidence of saturation. Energy integrates the raw 10-second Netdata
GPU-power samples with linear interpolation over each arm; the coarse sampling
limits precision. This is GPU energy, not whole-system energy. Sparklina's power
was 4.0/4.0/4.0/8.66 W during the arms and is excluded from that ratio.

Both hosts' raw CPU, load, RAM, available memory, host I/O, and GPU-power series
are retained. Sparky's mean user+system CPU ranges 7.81–8.93% of the host across
the arms; Sparklina ranges 0.88–6.98%. Host I/O includes unrelated activity; the
measured process reports **zero read_bytes/write_bytes and zero write syscalls**
in every steady arm. Its two read syscalls per arm read `/proc/self/io`.

| Three-cycle CUDA profile | Before | After |
|---|---:|---:|
| Large product kernels | 624 / 107.752 ms | 0 |
| Matching reference sum kernels | 624 / 12.226 ms | 0 |
| Fused product/sum kernels | 0 | 624 / 68.279 ms |
| All CUDA kernels | 5280 / 154.428 ms | 4656 / 99.907 ms |
| `cudaLaunchKernel` calls | 4992 | 4368 |
| DtoH copies / stream synchronizations | 84 / 84 | 84 / 84 |

The profiler attributes 8.61 GB of cumulative allocation to `aten::mul` before
(the table's displayed units); the 624 eliminated 1792×2048 FP32 products account
for exactly **9,160,359,936 bytes** of that cumulative allocation. This is not
peak resident memory. Fused reduction remains 68.17% of after-profile kernel
time, so the result does not establish that all remaining projection bottlenecks
are solved. CUDA high-water allocation is 345,611,776 bytes, reserved 385,875,968
bytes, measured across both arms together; peak RSS is 2,207,332 KiB. These shared
high-water figures cannot establish a per-arm peak-memory improvement.

## Reproduction and attributable artifacts

Artifact root:
`/mnt/shared/tessera-measurements/first-model-20260907/joint-fused-projection`.
All execution used PrismaBuild and the existing pinned container
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
Native threads are bounded to one, compile workers to four. The measured worker
was Sparky, CPU affinity 5–8, with 4 CPUs, 12 GiB memory and 8 GiB GPU memory
reserved. Its terminal resource cleanup is complete with release `ok: true`,
no OOM, and scope peak memory 3,548,872,704 bytes (includes extension build).

| Action | PB key | Verified CAS result SHA-256 |
|---|---|---|
| Portable CPU extension build, `build-01` | `0ade5ec62dcf57aba538badcabbc7087aafe8c9b9a538a0c96b2ee4a468fbd67` | `b86c51ef34261e772ce500a0d5c21957a3a2bd8d929a814ac1aa80dfda10c4a1` |
| GPU qualification, `qualify-01` | `9deeac16c469cbf286e5e366e839eeb516ac4fd8ea6d5fc100b795b38798073f` | `3da48d904031516aef03becddc538e3c5af5145f2efc43f5be74044bf2ba9c74` |
| Actual-source ABBA, `actual-ab-01` | `24c59324f3d263a30450fa3ffb88e0d5389317da40a162f5833813da5b2fcbae` | `048e0c474c834d61e0a06600b2585fe733016328a4e79d95bbbe2916f0d1873e` |

All three terminals and actual CAS bytes were independently read and rehashed;
all exit statuses are zero. Per-action `verified-pb.json` retains the terminal
and CAS receipt. `qualify-01/pytest.xml` records collection/results. Build and
measurement preserve their own loaded `.so` and `build.ninja`: recompilation
under distinct snapshots produced different binary hashes, so binaries are not
silently equated. The measured binary SHA is
`9305c183c5214dc5ff1f73382963f8275eb1b6197cb8d173c6a93bffd700c115`.
The compiler warns that `.minnctapersm` is ignored for the instantiated kernel;
the tested and timed binary includes that warning, retained in the CAS log.

`actual-ab-01` also contains `receipt.json`, `before.py`, `after.py`, both
`*-trace.json` traces and CPU/CUDA tables, `netdata-both-hosts.json`, and
`analysis.json`. `source-manifest.json` and `frozen-source/` were extracted from
the independently rehashed PB snapshot bundle for measured source commit
`783a2554ce1d4f44e2e83db79f461d3f2a2330f8`. `build-01/headers/` preserves the
five headers whose hashes match the measured build identity. Final receipt SHA:
`4c54fa604496fb06a2c275616506fd7d6d8ab6029e24080a15ef98d5917985b5`;
raw Netdata SHA:
`629d9a88c0f6e4d40913a2ee771f5e18594fa3b5f560ef1323ab3cd7d4d8f83e`.

From the repository, with an unused output directory and the retained input
artifacts available on eligible workers:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --gpu --cpus 4 --demand mem_gb=12 --gpu-memory-gb 8 \
  --measurement --timeout-s 1500 --detach -- \
  bash experiments/joint_projection_reduce_run.sh bench \
  --output /mnt/shared/tessera-measurements/first-model-20260907/joint-fused-projection/actual-ab-repeat \
  --source-receipt /mnt/shared/tessera-measurements/first-model-20260907/qdq-residency/source-capture-01/receipt.json \
  --source-sha256 bf5337381562e2b4ae1fbdb672e85f198679e2848487b291590d7810990872d3
```

Use `qualify --output <unused-path>` with the same wrapper for the targeted GPU
suite, omitting `--measurement`; `build --output <unused-path>` uses CPU-only
Docker under a portable PB reservation. The raw Netdata collector is retained at
`qdq-residency/frozen-source/collect_netdata.py`; run it read-only against the
finished benchmark receipt while the time series remains available.
