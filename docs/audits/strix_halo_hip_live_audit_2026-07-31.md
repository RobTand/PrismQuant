# Strix Halo HIP/Gridbook live audit — 2026-07-31

**Status:** current open audit. This is a dated evidence record, not a claim that
the ROCm lane is shippable. Close findings here with code, permanent tests and
served measurements; do not replace the record with a cleaner retrospective.

**Authority:** `docs/ARCHITECTURE.md` remains normative. This audit expands the
Strix Halo disposition in R7 of
`docs/audits/architecture_re-vet_2026-07-30.md`.

## Executive decision

The dense FP8-CB GEMV and BF16-WMMA GEMM are **good research/bring-up
kernels**. Their finite-input arithmetic is credible on `gfx1151`: the source
is unusually disciplined, the decoder is shared, odd rungs and ragged shapes
are covered, and every live correctness gate run for this audit passed.

They are **not production-ready**. Capability attestation, dtype and rounding
contracts, fallback safety, binding validation, build identity, permanent test
coverage, exact production-path benchmarking and vLLM serving evidence all
remain open. NVFP4-CB is decode-only; signed-S FP4, FP4 prefill and MoE are not
qualified.

Gridbook does **not** currently serve native integer INT8 or INT4. **Decision:
add W4A16 INT4 support to Gridbook**, using upstream vLLM's existing RDNA
W4A16 path as the delegated execution backend. The Gridbook work is the exact
export/profile/loader/dispatch contract and its validation, not a second custom
W4A16 kernel. Do not author an IU8 kernel until a served-faithful, smoothed W8A8
accuracy gate passes; do not start W4A4 or an integer-grid CB format yet.

## 1. Scope and provenance

| Item | Audited value |
|---|---|
| Host | Real Strix Halo box, `gfx1151`, 64 GB physical RAM / about 58 GiB visible to the GPU |
| OS | Fedora 44, kernel 7.1.5 |
| ROCm | 7.1.1 |
| PyTorch | 2.9.1 ROCm build |
| PrismaQuant tree | `e783fb3da22c38e184a2a7e16cc0cb09c8284a8e` |
| Standalone Gridbook tree | `348c689d2ae525c2728bdb0ce09c0b5ce299c4a7` |
| Remote source identity | `/home/rob/pq_hip_bringup/final` was byte-identical to the audited in-tree HIP implementation |
| Serving stack | vLLM was not installed on the box during this audit |

The hardware was reached by passwordless SSH and the tests below ran on the
real device. Because vLLM was absent, no statement in this document is a served
latency, throughput, graph-capture, memory-residency or quality result.

## 2. What “support” means

The word *support* has been used for four different states. This audit uses the
following ladder:

1. **Declared** — a registry/profile names the format.
2. **Serializable** — exporter and loader preserve its exact stored bytes and
   metadata.
3. **Kernel-capable** — a device kernel implements the declared arithmetic and
   exposes positive path attestation.
4. **Served** — vLLM routes real model layers through that path, including
   graph/eager and fallback behavior.
5. **Qualified** — served quality, performance, memory and soak gates pass and
   are recorded in a shipcard.

| Format/path | Declared | Serializable | Kernel-capable on Strix | Served | Qualified |
|---|---:|---:|---:|---:|---:|
| FP8-CB storage -> BF16 GEMV | yes | yes | yes | no | no |
| FP8-CB storage -> BF16-WMMA GEMM | yes | yes | yes | no | no |
| E2M1/NVFP4-CB -> BF16 GEMV | yes | yes | partial, product decode | no | no |
| E2M1/NVFP4-CB prefill | yes | yes | no | no | no |
| `INT8_W8A16` | research registry | no native served export | no | no | no |
| INT8 W8A8 / IU8 | scratch experiment | no | instruction probe only | no | no |
| `INT4_W4A16_g128` | research registry | no Gridbook export | no Gridbook kernel; upstream vLLM path exists | no Gridbook serve | no |
| INT4 W4A4 / IU4 | no product profile | no | instruction probe only | no | no |

`NVFP4_CB` is an E2M1 floating-point codebook format decoded to BF16 on Strix.
It is not uniform integer INT4 and does not execute the `iu4` WMMA builtin.

## 3. Correctness evidence

The following live gates passed:

- standalone compile, capability probe, parity and edge-case suite;
- 38/38 Torch extension tests;
- finite BF16 `fp8_act_qdq` comparison against
  `codec.fp8_dynamic_act_qdq` for `M={1,4,17,128}`, varied K, scales and finite
  extremes — bit-identical BF16 output;
- integrated `qdq_input=True` GEMV/GEMM comparison against explicit QDQ followed
  by `qdq_input=False` — bit-identical output.

The permanent suite still does not exercise the last two bullets. Both checked
in tests pass `qdq_input=False`
(`plugins/gridbook/tests/test_hip_decode_parity.py:176,229`). The exhaustive
E4M3-code test checks PyTorch conversion, not the extension (`:159`). Pytest
covers FP4 rungs `{12,13,16,18,20,24}`, not the entire advertised K12–K24
ladder (`:55`), and the standalone main invokes no FP4 case
(`csrc_hip/cb_hip_selftest.hip:675`). Tests also skip when the extension is
absent (`test_hip_decode_parity.py:46`); a release gate must fail on that skip.

Finite inputs are the currently supported arithmetic domain. NaN/Inf QDQ rows
did not match the Python reference in an exploratory check. Either reject
non-finite inputs explicitly or define and pin their behavior.

### What is already strong

- GEMV and GEMM share the format decoder, reducing drift risk
  (`csrc_hip/cb_decode_hip.h:1`).
- Odd-rung ceil-first splitting is explicit (`cb_decode_hip.h:145`).
- Padded-read and row/codebook boundary rules are documented and implemented
  (`cb_decode_hip.h:255`, `cb_gemv_hip.hip:150`).
- Kernels use the current PyTorch ROCm stream
  (`csrc_hip/cb_hip_torch.cpp:83-85`).
- The implementation avoids fast-math shortcuts and tests against an FP64
  reference at a one-BF16-ULP gate where exact WMMA rounding is unavailable.
- Weight expansion is transient; no resident dense `[N,K]` copy is introduced.

## 4. Production blockers

### P0 — contract and safe dispatch

1. **No exact device/build attestation.** `hip_enabled()` accepts any live ROCm
   device (`linear_hip.py:41-54`), the JIT target can be overridden arbitrarily
   (`hip_ext.py:104-117`), and the binding checks wave32 but not `gfx1151` or
   WMMA (`cb_hip_torch.cpp:87-101`). GEMM has a scalar compile fallback
   (`cb_gemm_hip.hip:58-75`); `cb_gemm_uses_wmma()` returns `true`, while the
   actual device capability kernel is not exposed (`:277-285`).

2. **Advertised dtype differs from the binding.** Gridbook advertises BF16 and
   FP16 (`gridbook/config.py:224`), but all HIP operations require BF16
   (`cb_hip_torch.cpp:134,150,228`). Choose and enforce one product contract;
   do not let FP8 and FP4 dispatch fail differently.

3. **Arithmetic changes at the GEMV/GEMM crossover.** GEMV rounds the decoded,
   scaled weight to BF16 before accumulation (`cb_gemv_hip.hip:196`); GEMM
   applies the scale in its FP32 epilogue (`cb_gemm_hip.hip:251-267`). Dispatch
   changes at flattened `M=16/17` (`linear_hip.py:119-133`). Define which
   contract Strix owns and add a crossover-continuity test.

4. **Binding validation is insufficient.** `check_common()` validates only part
   of the public raw-pointer contract (`cb_hip_torch.cpp:122-130`). Require all
   tensors to be device-resident, same-device and appropriately contiguous;
   validate packed row bytes, scale length, offsets and codebook bounds before
   launch.

5. **Fail-soft is unsafe.** Build failures return `None`
   (`hip_ext.py:196-260`), and `maybe_apply()` silently falls through
   (`linear_hip.py:67-78`). Large FP8 prefill can then reach a CUDA/CUTLASS path
   on ROCm (`gridbook/linear.py:557`), while ROCm model loading still probes the
   CUDA extension (`linear.py:295`). Add a required/fail-closed mode, explicit
   path counters and a bounded ROCm-valid BF16 fallback.

6. **The dispatch environment knob can violate the binding.**
   `PRISMAQUANT_CB_HIP_M_MAX` is unbounded (`linear_hip.py:59`); GEMV rejects
   `M>16` (`cb_hip_torch.cpp:167`). Clamp or validate it.

### P0 — build identity and permanent integration tests

- Key the JIT cache/module on content digest, format ABI, compiled architecture,
  PyTorch ABI and ROCm version. The current staged-copy test is size/mtime based
  (`hip_ext.py:168-190`), the module name is constant (`:243`), and
  `PYTORCH_ROCM_ARCH` uses `setdefault` (`:235`).
- Detect the current/tensor device rather than always device 0
  (`hip_ext.py:112`).
- Include the required shared header in install validation
  (`hip_ext.py:84-92`).
- Add permanent `qdq_input=True`, all-rung FP4, signed-S policy, malformed-input,
  non-default-stream, graph-capture, built-wheel and exact `maybe_apply()` tests.
- Run an actual vLLM custom-op serve in eager and graph modes with path
  attestation. No test today calls `maybe_apply()` or vLLM.

## 5. Performance evidence and immediate bug

The original standalone timing harness is not the production path:

- runtime prefers the BF16 codebook (`linear_hip.py:100-117`), while the harness
  always constructs E4M3 source (`cb_hip_selftest.hip:563-585`);
- runtime uses the GEMV-derived LUT selector for GEMM
  (`cb_hip_torch.cpp:247-254`), while the harness forces GEMM LDS
  (`cb_hip_selftest.hip:644-651`);
- the harness calls raw launchers without production activation QDQ
  (`cb_hip_selftest.hip:575`).

The following exploratory A/Bs were interleaved to limit Strix's dynamic-clock
bias. They isolate kernel policy (`qdq_input=False`); they are not served
numbers.

### Runtime codebook source, GEMV

`N=5120, K=4096, M=1`, exact runtime LDS/global resolver:

| Rung | BF16 codebook ms | E4M3 codebook ms | BF16 / E4M3 | Runtime LUT |
|---|---:|---:|---:|---|
| K36 | 0.0982 | 0.1020 | 0.963 | LDS |
| K40 | 0.1010 | 0.1096 | 0.922 | LDS |
| K42 | 0.1219 | 0.1684 | 0.724 | global |
| K44 | 0.1294 | 0.1692 | 0.765 | global |
| K48 | 0.1526 | 0.1834 | 0.832 | global |

The default BF16 source was 4–28% faster despite twice the LUT bytes because it
removes E4M3 conversion. At K44 GEMM `M=128`, the same isolated runtime-global
comparison was 0.6726 ms / 7.98 TFLOP/s for BF16 versus 1.1361 ms /
4.73 TFLOP/s for E4M3.

### GEMM LUT policy — P0 performance defect

K44, `N=5120, K=4096`, BF16 codebook:

| M | LDS | Global | Global slowdown |
|---:|---:|---:|---:|
| 128 | 0.3721 ms / 14.43 TFLOP/s | 0.6271 ms / 8.56 TFLOP/s | 1.685x |
| 512 | 1.4775 ms / 14.53 TFLOP/s | 4.8443 ms / 4.43 TFLOP/s | 3.279x |

Runtime currently selects the slower global arm for K44 while the benchmark
reports the LDS arm. Separate GEMV and GEMM LUT policies before making further
prefill claims.

Exploratory QDQ-only costs at `K=4096` were about 0.024 ms (`M=1`), 0.035 ms
(`M=128`) and 0.093 ms (`M=512`). At decode that is material relative to the
raw kernel, so all release comparisons must include the declared activation
contract. An explicitly named Strix `CB-A16` profile is worth measuring; it
must not silently change the existing A8 `FormatSpec`, because activation bits
participate in quality and allocator pricing.

### Enhancement order after P0

1. Benchmark the exact Python/custom-op dispatch with its real codebook source,
   QDQ setting, LUT policy and a same-process rocBLAS BF16 baseline.
2. Decode B once into LDS instead of redundantly in lanes/waves
   (`cb_gemm_hip.hip:229-240`).
3. Sweep a partial-LDS strategy for K42+ and learn a shape-aware selector.
4. Test cooperative A/B staging and K-loop double buffering.
5. Add FP4 prefill only after dense FP8-CB serving is qualified; add grouped
   MoE only after the dense contract is stable.

## 6. INT8/INT4 decision

### Evidence

The hardware probes are promising instruction ceilings, not product kernels:

- IU8 measured about 1.06x BF16 register-resident and 1.56x LDS-fed.
- IU4 measured about 2.06x BF16 register-resident and 2.86x LDS-fed.

The current GEMM is register-oriented, so the 1.56x IU8 LDS number is not a
credible application-speed forecast.

Naive W8A8 INT8 failed its quality gate. On the 4B confidence-KL screen it
scored 0.03659 versus 0.01042 for FP8 W8A8, a 3.51x regression. INT8 weights
with BF16 activations scored 0.00188; the activation grid caused the damage.
See `docs/design/int8_w8a8_accuracy_gate.md`.

Those screens are directional rather than serialized-serving proof. Registered
integer formats declare FP16 scales, while `_rtn_uniform_int()` and the scratch
gate used unsnapped FP32 scales (`prismaquant/format_registry.py:214-239`). Any
promotion experiment must pack, reload and dequantize the exact emitted scale
dtype before comparing quality.

Upstream vLLM merged `RDNAHybridW4A16LinearKernel` on 2026-07-14 (PR #40977).
It shares one packed weight tensor, routes skinny decode to HIP and prefill to a
fused-dequant Triton kernel, supports BF16/FP16 plus symmetric/asymmetric INT4
at group sizes 32/64/128, and rejects `g_idx`. This is the correct first control
because it already owns the generic RDNA W4A16 kernel surface.

### Interim use policy

**Product decision:** W4A16 is in scope for Gridbook. Implement standard
compressed-tensors-compatible INT4 packing and metadata, declare the narrow
Strix profile, delegate eligible layers to vLLM's RDNA hybrid kernel, and record
positive path telemetry. “Gridbook supports INT4” becomes true only when that
end-to-end route passes the release gates below; it does not require Gridbook
to own another matrix kernel.

| Candidate | Use now? | Promotion rule |
|---|---|---|
| FP8-CB A8 | Experiment only on Strix | Qualify exact vLLM path; use when its served quality/bytes/latency point is Pareto-efficient |
| Explicit FP8-CB A16 | Measure next | New profile and quality accounting; never mutate A8 semantics in place |
| NVFP4-CB | Decode research only | Require FP4 prefill, full-rung/product-mode tests and served evidence |
| Upstream W4A16 A16 | Yes, as control/first integer integration | Exact pack/load/dequant parity and matched-byte served A/B |
| W8A16 | Screen only | Keep only if it supplies a useful high-quality/capacity tier; do not call it IU8 acceleration |
| W8A8 IU8 | No | Folded SmoothQuant-style rescaling must pass serialized-faithful 4B and 27B quality gates first |
| W4A4 IU4 | No | Requires independent actual-scale kernel and served A4 quality gates after W8A8 succeeds |
| Integer-grid CB | No | Revisit only if it unlocks a validated integer-compute path; index width, not codebook dtype, sets CB bitrate |

The matched nominal-rate W4A16 experiment is:

| W4A16 | Nominal bpw | CB comparator |
|---|---:|---|
| symmetric g128 | 4.125 | FP8-CB K33 |
| symmetric g64 | 4.250 | FP8-CB K34 |
| symmetric g32 | 4.500 | FP8-CB K36 |

Use exact serialized bytes, including scales, sidecars, metadata and padding,
for the final match. Initial scope is symmetric, no `g_idx`, BF16, dense and
TP=1. Record two comparisons rather than confounding activation semantics:

1. **served-contract:** W4A16-A16 versus current FP8-CB-A8;
2. **weight-isolated:** W4A16-A16 versus an explicitly named FP8-CB-A16
   profile.

Promotion is quality-first: a candidate that fails the model-level KL/PPL veto
does not earn a place because it is fast. Among candidates that pass, retain
only exact-byte, served latency/memory Pareto points for the relevant workload
mix. Raw instruction throughput and standalone launcher timings cannot promote
a format.

## 7. Release gates

The Strix HIP lane remains on release hold until all of the following are
recorded:

1. exact compiled/live architecture, WMMA, format-ABI and source-digest
   attestation;
2. one explicit BF16/FP16 and GEMV/GEMM arithmetic contract;
3. permanent activation-QDQ, full advertised-rung, malformed-input, stream and
   crossover tests, with a fail-if-skipped hardware job;
4. wheel-install and vLLM custom-op tests in eager and graph modes;
5. required-mode failure tests and positive per-layer path telemetry;
6. exact-production-path decode, prefill, throughput, memory and cold/warm TTFT
   measurements against rocBLAS/upstream controls under sustained clocks;
7. model-level served quality at 4B, then the target 27B/MoE model;
8. a soak covering repeated load/unload, mixed M, graph replay and fallback
   faults.

The existing publication hold in `scripts/sync_gridbook.py:129` remains
correct.

## 8. Documentation drift found by this audit

These are audit findings, not alternate sources of truth:

- `docs/ARCHITECTURE.md:1786` still says nothing is built on Strix.
- `docs/ARCHITECTURE.md:2086,2116` carries older timings and says the integer
  gate is running.
- `docs/design/strix_halo_format_plan.md:3,54` says the plan implements nothing
  and recommends W8A8 before later recording its failed gate.
- `plugins/gridbook/gridbook/csrc_hip/README.md:426` says the completed integer
  gate is still running.
- `docs/design/format_kernel_inventory.md:353` calls the ROCm CB surface empty.

Reconcile those documents against this dated evidence and current code. Do not
promote any standalone number into the architecture's served-results table.
