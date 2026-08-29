# NVIDIA Ampere INT8 Gridbook feasibility (2026-08-24)

Status: research decision note only; no implementation or production-format change.

**Go/no-go: NO-GO for implementation or a dedicated RTX 3090 prototype. Native
feasibility is established, but possible is not the same as worth building.
Park this lane unless concrete demand and near-zero-opportunity-cost hardware
or kernel reuse materially change the product case.**

Scope: RTX 30-series / GA10x / compute capability 8.6, dense Linear first. The
local Gridbook audit used the clean v0.9.0-rebased tree at commit
`63e019f31b57c13b0fe9697441f5084eb194440a` (`v0.9.0-1-g63e019f`). Statements
below are labelled **Verified**, **Inference**, or **Conditional design**.
PrismaQuant's checked-in producer/runtime pin still names Gridbook 0.8.11 and
runtime-contract v4; the isolated v0.9.0 tree advertises runtime-contract v10
and only an SM89 platform. Neither tree currently declares an INT8-CB family or
SM86 serving cell. Any future prototype would therefore remain outside the
production menu until a released Gridbook contract is pinned.

## Decision

**Technical answer: yes, an INT8 Gridbook family is natively supportable on RTX
30-series, but it would have to be a distinct signed-INT8 numeric and wire-format
family, not an FP8-CB fallback. Product answer: do not build it now.** There is
no authorization here for kernels, a prototype artifact, format-registry or
runtime-contract changes, a calibration burn, or a physical-SM86 campaign.

If this lane is ever reopened, it could structurally retain the packed product-
code indices, K%4 ladder, row-scale concept, loader/route contracts, and graph-
safe custom-op pattern. It would still require new INT8 codebooks, activation
quantization, decode GEMV, expansion, and SM80/SM86 integer Tensor Core GEMM
kernels; that is precisely why it is not a low-investment extension.

The lowest-risk architecture **if the decision is explicitly reopened** would
be:

* decode (`M <= 8`): compare packed-code lookup plus signed `dp4a` against a
  W8A16 lookup route; do not assume the W8A8 arm wins at skinny M;
* prefill/batched (`M > 8`): transiently expand the codebook weights to `s8`,
  dynamically quantize activations to `s8`, then run SM80/SM86
  `s8 x s8 -> s32` Tensor Core GEMM with a custom scaling/output epilogue;
* leave direct decode-in-MMA and MoE grouped kernels for a second gate.

`dp4a` is a four-byte scalar/SIMD dot-product instruction, not a Tensor Core
instruction. The native Tensor Core route begins with the larger-M expanded
W8A8 GEMM. Both W8A8 routes also introduce activation quantization risk that a
signed-INT8 weight artifact using W8A16 decode does not.

This is a hardware-enablement lane, not a size reduction relative to FP8-CB.
At the same rung and layer assignment, the index stream, row scales, and raw
one-byte codebook values have essentially the same artifact cost.

## Hardware facts

**Verified.** NVIDIA lists desktop GeForce RTX 3050 through RTX 3090 Ti as
compute capability 8.6, with third-generation Tensor Cores. VRAM ranges from
6/8 GB at the low end to 24 GB only on RTX 3090/3090 Ti; the 3080 family is
10/12 GB and the 3060 is 8/12 GB. Therefore a nominal 20 GB model only
targets the 24 GB 3090 tier within this generation, and its remaining nominal
4 GB must cover runtime state, transient workspace, CUDA/vLLM overhead, and KV
cache rather than KV alone. Sources: [NVIDIA CUDA GPU compute-capability
table](https://developer.nvidia.com/cuda/gpus) and [official RTX 30-series
specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/).

**Verified.** PTX exposes `mma.sync.aligned.m16n8k16` and `m16n8k32` integer MMA
with `s8`/`u8` inputs on `sm_80` and later. The accumulator is `s32`; without
`.satfinite`, overflow wraps, while `.satfinite` clamps to the signed 32-bit
range. FP8 E4M3/E5M2 MMA requires `sm_89` and therefore is not native on
RTX 30-series. PTX also exposes `dp4a`, which performs four packed byte
multiplies plus a 32-bit accumulation and is available well before Ampere.
Source: [NVIDIA PTX ISA, matrix multiply-accumulate and video SIMD
instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).

**Verified.** CUTLASS lists an SM80 TensorOp path for
`s8 * s8 + s32 -> s32/s8`. Its GEMM design keeps accumulator fragments in
registers, double-buffers tiles through shared memory, and supplies an epilogue
stage for scaling/conversion. NVIDIA supplies an SM80 INT8 TensorOp test with
`int8` A/B and `int32` accumulation. Sources: [CUTLASS functionality
table](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/functionality.html),
[efficient GEMM design](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html),
and [NVIDIA CUTLASS SM80 INT8 TensorOp
test](https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/gemm_s8t_s8n_s32n_tensor_op_s32_sm80.cu).

**Inference.** Applying both broadcast scale vectors and producing BF16 in one
epilogue is a new Gridbook/CUTLASS engineering and qualification task; the
generic CUTLASS epilogue mechanism does not prove that exact kernel already
exists.

**Verified.** The Ampere tuning guide gives an SM86 budget of 48 resident warps,
64K 32-bit registers, at most 16 resident blocks, 100 KiB shared memory per SM,
and at most 99 KiB dynamic shared memory per block. Allocations above 48 KiB
per block require explicit opt-in. Ampere supports asynchronous global-to-shared
copy. Source: [NVIDIA Ampere tuning
guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#occupancy).

**Verified.** The GA102 whitepaper's dense peak figures give RTX 3080
59.5 TFLOP/s for BF16 Tensor operations with FP32 accumulation versus
238 TOP/s for INT8, and RTX 3090 71 versus 284: a 4x arithmetic-rate ratio.
The larger slash-separated numbers in the paper assume 2:4 sparsity and do not
apply to current codebooks. Peak TOP/s is not a serving-speed prediction.
Source: [NVIDIA GA102 architecture
whitepaper](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf).

## Proposed numeric contract

**Conditional design, if reopened.** Define a new family such as `INT8_CB_K*`
with:

```text
qW[n,k] = signed INT8 codebook lookup, preferably symmetric [-127, 127]
qA[m,k] = round/clamp(A[m,k] / sA[m]) as signed INT8
acc[m,n] = sum_k int32(qA[m,k]) * int32(qW[n,k])
Y[m,n] = BF16_RN(float(acc[m,n]) * sA[m] * sW[n] + optional_bias[n])
```

Use one FP32 weight scale per output row and one FP32 dynamic activation scale
per token as the baseline. A symmetric `s8 x s8` contract matches signed neural
weights and activations and avoids the zero-point correction sums required by
asymmetric `u8` quantization. PTX does support signed, unsigned, and mixed-sign
integer operands, but that capability is not a reason to introduce zero points.

**Verified.** TensorRT's documented INT8 scheme is signed two's-complement,
round-to-nearest-ties-to-even after clipping, with dequantization by a scale;
it documents per-axis weight and per-tensor activation quantization as supported
patterns. Source: [NVIDIA TensorRT quantized types and
schemes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html).

**Verified.** With magnitudes bounded by 127, an unscaled dot product cannot
overflow signed INT32 before `floor((2^31-1) / 127^2) = 133,144` terms. Every
target Linear's actual reduction dimension must nevertheless be enumerated and
checked at export/load; an unsupported dimension must fail closed. Using
`-128 * -128` lowers the simple bound to 131,071 terms, another reason to reserve
`-128` in the initial symmetric contract.

**Inference.** A per-token max-absolute activation scale is the simplest graph-
safe baseline but may be more sensitive to activation outliers than E4M3. A
clipped/calibrated arm and a scale-transfer arm should be measured. Per-group
activation scales are not the first design: they require separately scaled
partial `s32` sums and complicate both the GEMM mainloop and epilogue.

**Conditional design, if reopened.** Keep a small-M W8A16 comparison arm:
gather/convert signed
INT8 weights, accumulate against BF16 activations, then apply the row scale.
It gives up native integer dot products but avoids the per-token max reduction,
INT8 activation workspace, and activation quantization error. Decode is often
bandwidth- rather than arithmetic-bound, so W8A8 is not automatically superior
at `M=1`; this choice belongs in route qualification, not the artifact format.

**Inference.** A learned INT8 codebook may use the finite normalized range more
efficiently than E4M3 for well-scaled, bounded weight rows. Conversely, E4M3's
nonuniform exponent spacing may tolerate broad dynamic range better. Neither
quality ordering should be assumed; learned versus lattice and INT8 versus FP8
must be measured per rung on the same calibration contract.

## Why FP8-CB cannot simply be reinterpreted

**Verified from local sources.** PrismaQuant currently defines FP8 product
codebooks as four equal subtables over an 8-value codeword and the producer
ladder `K4,K8,...,K48`; the packed index stream costs exactly `K/8` bits per
quantizable weight, with no group-16 scale plane and a separate FP32 row-scale
cost. See `prismaquant/cb_layout.py` and `prismaquant/format_registry.py`.
The audited Gridbook reader validates BF16 sidecar values against the E4M3 grid
and losslessly re-encodes them to raw E4M3 bytes. Its native GEMV and expander
explicitly accept that E4M3-byte representation.

Consequently the base INT8-CB accounting is `K/8 + 32/in_features` bpw before
amortizing the shared codebook sidecar. Raw INT8 and raw FP8 tables both cost one
byte per entry in runtime memory. If the new wire stores signed bytes instead of
the current two-byte BF16/FP16 sidecar, it halves the sidecar only; that saving
is tiny relative to an 18--20 GB packed index stream.

**Verified.** E4M3 is not an affine signed-integer grid: its spacing changes with
the exponent. There is no single row/global scale that losslessly maps an
arbitrary E4M3 codebook to INT8. Requantizing an existing FP8 table therefore
adds error. Giving each product subtable a different scale would require
separate partial accumulations and scale applications because the four
subtables cover different positions in every 8-value codeword.

**Conditional design, if reopened.** Train/render INT8-grid codebooks directly.
A lossy
FP8-to-INT8 conversion can be retained only as an explicitly labelled bootstrap
or negative-control experiment, never as the production wire contract.

## What transfers from FP8 Gridbook

The following can be reused structurally:

* the 256-weight superblock packing, 8-value codeword, four-product-subtable
  layout, K%4 extraction law, padding, offsets, row scales, and qname metadata;
* format/reader fail-closed validation, artifact/runtime contract separation,
  load-time route qualification, tensor-parallel shard checks, and receipts;
* native-extension loading, resident weight/codebook state, and transient
  expansion/prefetch lifecycle;
* the `M <= 8` decode versus larger-M routing shape;
* `torch.library.custom_op` plus fake registrations, immutable load-time route
  selection, and graph/eager validation discipline.

The following must be new or explicitly requalified:

* INT8-grid quantization, learned/lattice candidate generation, QDQ and artifact
  identifiers (do not overload `FP8_CB_K*`);
* raw signed-INT8 codebook loading and validation;
* dynamic signed-INT8 activation quantization and scale production;
* `s8` lookup/decode GEMV with `s32` accumulation and scaled BF16 output;
* `s8` transient expansion and SM80/SM86 integer Tensor Core GEMM epilogue;
* runtime capability gates, compiled architectures, op ABI, workspace sizing,
  graph schemas/fakes, correctness oracles, and performance route thresholds;
* MoE grouped execution and every TP degree/shape cell.

**Verified from local sources.** Current dense FP8-CB serving uses a custom CUDA
GEMV for small M and expands weights to E4M3 before calling vLLM's direct
CUTLASS scaled-matmul route for larger M. The checked-in experimental
`cb_persistent_tc.cu` uses an SM89 FP8 MMA atom and is neither the right
instruction family nor a production Ampere base. The Blackwell fused codebook
kernel is also architecture-specific and not portable to SM86.

**Verified.** Current cuBLAS documentation permits `CUDA_R_8I` inputs with
32-bit integer compute/output, but the listed INT8 combinations do not support
the general epilogues needed here. Source: [NVIDIA cuBLAS data-type support
tables](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul).

**Verified.** cuBLAS requires at least 4-byte-aligned A/B and leading dimensions
that are multiples of four for `CUBLAS_COMPUTE_32I`; its performant regular IMMA
ordering is NT-only and recommends the documented 16-byte alignment conditions.
PTX integer Tensor Core atoms reduce in K quanta of 16 or 32. Gridbook's
256-weight superblock padding is therefore favorable for K alignment, but N
padding, tensor-parallel shard cuts, strides, pointer alignment, and output-tail
handling still require explicit load-time gates. A custom CUTLASS layout need
not exactly mirror cuBLAS, so these are constraints to qualify rather than
blindly copy.

**Inference.** cuBLASLt is useful as an expanded-W8A8 correctness/performance
baseline, but a CUTLASS/custom Gridbook kernel is likely required to apply
per-token `sA[m]` and per-output-row `sW[n]` in a single BF16 epilogue. Whether
the pinned vLLM native op has a compatible INT8 ABI is unverified and must be
audited; the existing FP8 wrapper is not evidence that it does.

## Kernel design and likely bottlenecks

### Decode, M <= 8

**Conditional design, if reopened.** Start with a new CUDA kernel that extracts
each packed
codeword, gathers eight signed bytes from the four product subtables, and uses
two `dp4a` operations per activation/weight codeword into `s32`. Apply
`sA[m] * sW[n]` once after reduction and convert to BF16. Reuse the existing
packed-load and single/double-buffer scheduling ideas, not its FP8 arithmetic.

The W8A8 route needs a graph-safe activation quantizer that emits signed bytes
and one FP32 scale per token before GEMV. Benchmark a fused/two-pass form and a
preallocated-workspace form against the W8A16 comparison arm above.

**Inference.** `dp4a` will probably beat a Tensor Core MMA decomposition at
M=1 and may remain best through the current M<=8 band because skinny-M MMA
utilization and packing overhead are poor. This is a benchmark hypothesis, not
a route rule.

### Prefill and larger batches

**Conditional design, if reopened.** The first candidate would expand `[N,K]`
to signed INT8,
dynamically quantize `[M,K]` to signed INT8, then use an SM80/SM86 CUTLASS
TensorOp mainloop with `s32` accumulation and a custom FP32 scaling/BF16
epilogue. Use preallocated/captured workspace or an out-variant where graph
capture requires it. A later fused kernel may decode codebook values directly
into B fragments and avoid the expanded-weight traffic.

**Inference.** The 4x peak INT8/BF16 arithmetic ratio will be substantially
discounted by bit extraction, LUT gathers, activation quantization, expansion,
and transient memory traffic. Expansion plus GEMM is attractive as the shortest
correct bridge, but it must not become a production default unless served
throughput reaches parity with the conventional lane it displaces.

### Shared memory, registers, and K ladder

For the current four-subtable layout, a raw one-byte INT8 (or FP8) LUT occupies:

| Rung | Raw LUT bytes | Rung | Raw LUT bytes |
|---:|---:|---:|---:|
| K4 | 16 B | K28 | 1 KiB |
| K8 | 32 B | K32 | 2 KiB |
| K12 | 64 B | K36 | 4 KiB |
| K16 | 128 B | K40 | 8 KiB |
| K20 | 256 B | K44 | 16 KiB |
| K24 | 512 B | K48 | 32 KiB |

The formula is `8 * 2^(K/4)` raw bytes. The existing FP16/BF16 sidecar is twice
these sizes, but the sidecar is negligible beside the packed model weights.

**Conditional design, if reopened.** Preserve all `K4,K8,...,K48` formats at
the storage and
expansion-route level. Do not impose an artifact-level K cutoff from shared
memory alone: expansion and GEMM do not require the LUT to coexist with the MMA
tiles.

**Inference.** A future fused LUT+MMA kernel is likely to hit its first practical
occupancy/tile-design cliff at K44/K48. A 16/32 KiB full LUT consumes meaningful
fractions of the 100 KiB/SM budget before double-buffered A/B tiles and epilogue
storage; `s32` accumulator fragments also compete for the fixed 64K-register
file. Initially qualify fused execution through K40, and enable K44/K48 only
per measured tile/shape. Global/L1-resident LUT access may move the crossover
and must be benchmarked rather than assumed.

## Comparison with alternative Ampere lanes

| Lane | Main benefit | Main drawback / required comparison |
|---|---|---|
| INT8-CB W8A8 | K/8 weight bpw (3.5 at K28, 6.0 at K48) with native Ampere integer MMA | Lookup/expansion and dynamic-activation tax; entirely new numeric kernels |
| Conventional W8A8 INT8 | Mature native MMA, simple dense storage and strong prefill baseline | 8 weight bpw, so a 27B dense model does not fit 24 GB as weights alone |
| Conventional INT4 weight-only | Much smaller weights, mature serving implementations, activations retain BF16/FP16 range | Decode may dominate; quality at matched bpw must be compared with codebook VQ |
| INT4 W4A4 MMA | Higher nominal integer throughput | Activation quality/scaling is much harder; TensorRT documents INT4 as weight-only in the referenced scheme |
| BF16/FP16 | Simple, reliable quality and kernels | Roughly 2 bytes/weight: infeasible for a dense 27B model on a single 24 GB card |
| FP8-CB | Existing Ada family and E4M3 dynamic range | FP8 MMA requires SM89, so it is not a native RTX 30-series lane |

**Inference.** INT8-CB's plausible differentiator is better empirical quality
than uniform INT4 at a similar 3.5--5-ish bpp while retaining an integer Tensor
Core prefill route. It is not guaranteed to outrun mature INT4 weight-only
decode, and should not ship merely because its dense integer GEMM is fast.

## Graph-compilation requirements

Graph compilation is a release gate, not a cleanup item:

* expose Gridbook-owned custom ops with fake/meta registrations;
* compile/load kernels and choose the route before capture;
* keep shape guards tensor-free and avoid `.item()`, host synchronization,
  algorithm discovery, lazy JIT, or allocations inside captured forwards;
* use stable out variants/preallocated workspace for expansion and quantization
  where the allocator would otherwise enter the graph;
* fail closed if SM86, rung, alignment, shape, TP, or compiled-kernel support is
  absent; never fall back to materialized BF16 or host/NVMe streaming;
* test eager and `torch.compile(fullgraph=True, dynamic=False)` outputs and
  route receipts on the same inputs.

## Dormant reopen checklist (not an implementation plan)

Do not execute the gates below under the current decision. They preserve the
minimum technically credible qualification sequence so a future demand signal
can be evaluated without repeating this research. Reopen only when all three
conditions hold: demonstrated 30-series user demand, access to physical SM86
hardware at negligible opportunity cost, and substantial reuse of an existing
qualified integer-kernel path. Otherwise this note is complete as a feasibility
answer.

All comparisons use the same calibration examples, sequence length, per-Linear
assignment semantics, resident cache policy, and quantizable-parameter bpp
accounting.

### Reopen gate 0: offline numeric feasibility

Render only three physical INT8-grid anchors initially: K28 (3.5 bpw), K40
(5.0 bpw), and K48 (6.0 bpw), with learned/lattice variants at the likely
crossover. Do not burn all 12 formats. Impute the remaining K%4 rungs only after
at least one held-out rung validates the interpolation law. Add the very low
rungs only if a sub-3.5-bpw product is in scope. Measure per-Linear weighted MSE,
held-out output error/KL, model NLL/perplexity, exact artifact bytes, and
determinism against:

* FP8-CB at the same rung and assignment (quality/size, on supported hardware);
* conventional symmetric W8A8 INT8;
* a mature conventional INT4 weight-only lane;
* BF16 reference.

Reject a direct FP8-to-INT8 table conversion unless it unexpectedly matches the
native INT8 render; retain it as a diagnostic arm.

### Reopen gate 1: physical RTX 3090 kernel qualification

A Spark or Ada machine can cross-compile `sm_86`, but cannot guarantee SM86
instruction selection, occupancy, launch behavior, graph capture, or speed.
Run on a physical RTX 3090/3090 Ti. Benchmark the target model's exact `(M,N,K)`
qname census. The first prototype matrix is deliberately small:

| Regime | M | Physical rungs | Candidate routes | Required baselines |
|---|---|---|---|---|
| Decode | 1, 4, 8 | K28, K40, K48 | W8A8 `dp4a`; W8A16 lookup | BF16; mature INT4 |
| Crossover | 16, 64 | K28, K40, K48 | `dp4a`; W8A16; expanded W8A8 Tensor Core | cuBLASLt INT8 oracle; BF16; INT4 |
| Prefill | 256, 1024 | K28, K40, K48 | expanded W8A8 SM80/86 CUTLASS | conventional W8A8; BF16; INT4 |

For W8A8, measure per-token max-absolute activation scaling first, then add one
clipped/calibrated arm only if saturation/outlier telemetry or held-out KL says
it is needed. Compare all W8A8 outputs with W8A16 to isolate activation error.
After a route survives this matrix, expand M coverage and qualify every producer
rung before release. Record:

* warm/cold route latency, tokens/s, effective bandwidth, expansion bytes and
  lifetime, Tensor Core utilization, registers/thread, spills, shared memory,
  achieved occupancy, and graph-replay latency.

Pass only with numerically correct `s32` accumulation/scaling, no overflow for
the enumerated dimensions, no fallback, no allocator/residency violation, and
served speed at least at parity with the production container being displaced.

### Reopen gate 2: graph and model correctness

On a small model and then the 27B target, compare eager and full-graph compiled
execution with the dequantized BF16 oracle across decode/prefill boundaries,
odd/padded shapes, all qualified rungs, TP degree 1 first, and deterministic
generation. Verify capture/replay, immutable route receipts, load failures for
wrong arch/dtype/rung/layout, and absence of persistent dense expansion.

### Reopen gate 3: 24 GB product gate

On a physical 3090, record exact weights, resident codebooks/scales, CUDA/vLLM
baseline, peak transient workspace, and KV capacity at each intended context and
concurrency. A “20 GB” filename is not evidence of fit. Require long-context
generation without OOM, the accepted NLL/task/ToolEval thresholds, eager and
compiled results, and representative served decode/prefill throughput.

MoE, tensor parallelism, K44/K48 fused execution, and direct decode-in-MMA each
remain separately gated. No INT8-CB format should enter the production menu
until Gridbook publishes the matching immutable runtime contract and physical
SM86 qualification cells.

## Review provenance

The factual audit used current NVIDIA PTX ISA 9.3, CUDA 13.3 Ampere tuning
documentation, current cuBLAS/cuBLASLt and TensorRT documentation, the current
NVIDIA CUTLASS tree, the GA102 whitepaper, and the two local source trees named
above. An independent Ox Alpha/OpenCode2 maximum-thinking pass rechecked the
PTX arch/type requirements, `dp4a`, saturation behavior, SM86 resource limits,
cuBLASLt epilogue restriction, CUTLASS SM80 INT8 path, local packing, and the
runtime-contract/pin split. Claude Fable 5 maximum-thinking was requested but
returned no review because the local account had reached its session limit; it
is not counted as a sign-off.

## Bottom line

Ampere supplies a real native W8A8 substrate, and Gridbook's index structure
could target it. Doing so is nevertheless a new quantization and kernel family,
with activation-outlier risk, decode/expand overhead, graph qualification, and
very tight 24 GB product headroom. That is too much investment for the present
demand signal. Record the feasibility, make no code or production-menu change,
and keep the 4090 FP8 lane on the critical path. If this is ever reopened, the
conditional design above remains the starting point; it is not current work.
