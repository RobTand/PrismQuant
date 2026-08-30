# QTIP-derived native NVFP4

**Status (2026-08-30): research contract, not a production format or serving
claim.** QTIP is the method lineage. EXL3 is a source-level comparator only.
Quartet II contributes scale-search hypotheses only where they preserve the
standard NVFP4 wire. None is a runtime dependency.

This note defines what “QTIP on native NVFP4” means in PrismaQuant. It keeps
three mechanisms separate so an improvement can be attributed rather than
named after a bundle:

1. QTIP-style Hessian/BlockLDLQ feedback with a native NVFP4 terminal;
2. randomized orthogonal incoherence around a native W4A4 GEMM; and
3. the existing stateful E2M1 trellis wire decoded by Gridbook into native FP4
   tensor-core operands.

The first mechanism can produce an ordinary compressed-tensors NVFP4 artifact
for stock vLLM. The latter two require the separately versioned Gridbook
runtime. PrismaQuant does not import or vendor that runtime.

## Pinned sources and claims

The source audit is against the official QTIP repository at commit
`e90c6688c8dfae326a3a81b5eb032db7c6680ec0`. In particular:

- `lib/utils/math_utils.py:block_LDL` constructs the block LDL factor;
- `lib/algo/ldlq.py:LDLQ` performs reverse block error feedback before each
  trellis decision;
- `lib/codebook/bitshift.py:bitshift_codebook` implements the tail-biting,
  stateful bitshift reconstruction; and
- `lib/algo/finetune.py` plus `lib/codebook/bitshift.py:BitshiftLinear.forward`
  show that randomized signs and normalized Hadamards are reversed online.

The method and reported model results belong to the QTIP paper and reference
implementation:

- https://arxiv.org/abs/2406.11235
- https://github.com/Cornell-RelaxML/qtip

The standard native endpoint is PrismaQuant's existing group-16 NVFP4
encoding: one E2M1 nibble per weight, one E4M3 scale byte per 16 weights, and
the existing scalar globals. The Gridbook E2M1 trellis lane already decodes its
wire into this E2M1/E4M3 operand pair and calls Blackwell `_scaled_mm`; this
design extends that existing lane rather than introducing another residency or
decode mechanism.

No EXL3 result, QTIP result, synthetic source-coding result, or weight-only SSE
number is a served PrismaQuant quality or speed result.

## Mathematical decomposition

For a Linear with column-vector convention

\[
y = W x, \qquad H = \mathbb E[x x^T],
\]

the usual isolated quadratic proxy is

\[
\mathcal L(Q) = \operatorname{tr}((W-Q)H(W-Q)^T).
\]

QTIP attacks distinct parts of this problem:

- **Incoherence.** Random signs and normalized Hadamards spread large or
  coherent coordinates before quantization.
- **Second-order feedback.** A block factorization of the transformed Hessian
  makes the quantization error of later blocks feed decisions for earlier
  blocks.
- **Source coding.** A stateful trellis makes a reconstruction depend on a
  code history. Its effective vector support is not a scalar lookup table.
- **Fine-tuning and allocation.** These are additional interventions and must
  not be credited to the three mechanisms above without their own arm.

For orthogonal input and output transforms `R_in` and `R_out`, define

\[
\widetilde W = R_{out} W R_{in}^T, \qquad
\widetilde H = R_{in} H R_{in}^T.
\]

Quantize `W_tilde` to `Q` and serve

\[
\widehat y = R_{out}^T Q R_{in}x.
\]

Because the transforms are orthogonal, the original-space output error is
exactly

\[
\mathbb E\lVert y-\widehat y\rVert_2^2
= \operatorname{tr}((\widetilde W-Q)\widetilde H
                     (\widetilde W-Q)^T).
\]

That identity is the contract for the Gridbook arm: rotate activations before
their native E2M1 quantizer, execute the existing W4A4 GEMM on a weight encoded
in the same basis, and apply the inverse output transform after the GEMM.
Rotating an already quantized activation, or using an untransformed Hessian to
optimize a transformed weight, is a different and invalid experiment.

## What a four-bit terminal can and cannot do

At a fixed group scale, ordinary NVFP4 exposes the 15 distinct numerical E2M1
values (the two signed-zero encodings are numerically identical). For a
separable weighted squared-error objective, the nearest allowed scalar value
minimizes every coordinate independently. Therefore a trellis whose complete
reconstruction support is restricted to that same product grid cannot beat
nearest scalar quantization on that fixed separable problem, regardless of its
state count. More states cannot change the set being minimized over.

This is a narrow boundary, not a no-go result for the project. It does not
cover:

- a non-diagonal Hessian objective, where the direction and correlation of
  errors matter;
- choosing the shared group scale jointly with codes;
- orthogonal incoherence, which changes how the source and Hessian meet the
  product partition;
- rate allocation across Linears or blocks; or
- a procedural decoder whose reconstruction values are not confined to the
  native E2M1 product grid.

The last case is QTIP's full source-coding mechanism. If such values are later
rounded independently to E2M1, the extra reconstruction support is lost. A
custom decoder may still expand a compact trellis stream into E2M1 tiles and
feed native FP4 tensor cores, as Gridbook already does, but that is a Gridbook
wire with native NVFP4 **compute**, not a stock compressed-tensors NVFP4 file.

This distinction is why a claim that “a four-bit trellis cannot beat EXL3” is
not acceptable without naming the exact support, objective, scale contract,
activation contract, and byte accounting. The scalar-support theorem proves
only the restricted case above.

## The NVFP4 scale byte is not spare storage

The per-group E4M3 byte contributes exactly 0.5 bits per quantizable weight and
is consumed by the hardware reconstruction

\[
w_i = \operatorname{E2M1}(c_i)\;s_{\lfloor i/16\rfloor}\;s_{global}.
\]

Changing it changes the weight unless the code nibbles are changed to an exact
alias. Signed zero and a possible “negate the group scale and all 16 codes”
symmetry are representation aliases, not additional reconstruction degrees of
freedom. They are data-dependent or decoder-specific, are not part of the
compressed-tensors contract, and give the stock kernel no new information it
can act on. Rotation seeds and other deterministic metadata are cheaper and
safer in an explicit versioned sidecar. PrismaQuant therefore assigns no
hidden-channel capacity to the native scale plane.

Quartet II's multi-scale search remains useful as an encoder search over legal
scale values. It does not turn the scale field into a second payload. The
existing PrismaQuant `{6,4}` JSO grid is the control; wider scale grids remain
opt-in until matched-calibration results justify them.

## Required experiment matrix

All arms use the same BF16 source tensor, activation rows, transformed or
untransformed Hessian as required by the equations above, group-16 E2M1/E4M3
terminal, quantizable-parameter denominator, and deterministic seeds.

| Arm | Offline optimizer | Online transform | Weight carrier | Purpose |
|---|---|---|---|---|
| A | RTN + existing JSO | none | stock NVFP4 | native scalar control |
| B | current GPTQ + static activation order + JSO | none | stock NVFP4 | current production optimizer control |
| C | QTIP-style BlockLDLQ, native terminal at every decision | none | stock NVFP4 | isolates transferable second-order feedback |
| D | same as C in the rotated basis | input + output sign/Hadamard | stock NVFP4 fields served by Gridbook | isolates incoherence with native W4A4 compute |
| E | same transformed Hessian contract | input + output sign/Hadamard | existing Gridbook E2M1 trellis wire | measures incremental stateful coding gain |

Arm D is not a new compressed-tensors scheme: it needs Gridbook metadata and
runtime transforms even though its matrix operand is ordinary native NVFP4.
Arm E retains the trellis and Gridbook's resident/streamed modes. Neither may
silently fall back to BF16 GEMM, materialize a parallel weight cache, or choose
a residency mode at runtime.

The first gate is a one-Linear isolate reporting serialized bpw, weight error,
the Hessian proxy above, and activation-output MSE. The next gate is a small
model all-position KL/PPL A/B. A production proposal requires exact served
quality, graph capture, load correctness, resident footprint, and prefill and
decode throughput against the displaced native/Gridbook lane.

For every performance gate, collect an in-process profile plus Netdata from
both Sparky and Sparklina. On GB10, report power against the approximately
140 W envelope and work per joule; GPU utilization is not diagnostic. Until
those measurements exist, the online-transform lanes remain explicitly
research-only and opt-in.

## Implementation boundaries

- PrismaQuant owns deterministic transform selection, transformed-Hessian
  calibration, encoder search, exact accounting, manifests, and quality gates.
- Gridbook owns transform execution, trellis decode, native W4A4 kernels,
  graph capture, residency, and its immutable runtime contract.
- The producer/runtime boundary is a versioned manifest plus the existing
  immutable Gridbook pin. Neither repository imports the other.
- Transform identities include algorithm, normalization, block geometry,
  input/output dimensions, seed, sign-vector digests, and padding. A mismatch
  refuses at load.
- Only graph-safe seams may cancel or fold transforms. Residual adds,
  nonlinearities, attention softmax, head reshapes, and routed-expert dispatch
  are barriers unless a separate algebraic proof and test covers that exact
  topology.
- Existing `ProductionWeightCache`, activation cache, and streaming prefetch
  paths remain the only cache/residency mechanisms.

The promotion question is empirical: how much of QTIP's advantage survives
the native E2M1 terminal, how much comes back from incoherence, and how much
additional value the stateful E2M1 wire supplies after both are present. The
matrix above answers those questions without importing EXL3 or attributing a
bundle's result to the wrong mechanism.
