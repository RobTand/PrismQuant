# QTIP-derived native NVFP4

**Status (2026-08-30): research contract, not a production format or serving
claim.** The stock-native Arm C BlockLDLQ isolate and a physical one-Linear
rotated-trellis producer are implemented. The producer has an exact
block-structured path for a strictly positive diagonal source Hessian under
the declared block-local input Hadamard; arbitrary dense Hessians retain the
reviewed whole-matrix boundary. An external Gridbook research
reference implements online sign/Hadamard transforms, but it is unpinned and
is not part of Gridbook 0.9.1's runtime contract. The producer emits the
existing canonical Gridbook trellis carrier, reparses and reference-decodes
the same bytes, and proves the original-basis serve algebra. QTIP is method
lineage; its serialized trellis wire is explicitly excluded and is distinct
from PrismaQuant/Gridbook's wire. Quartet II is literature only, and EXL3 was
read only as source context. Neither has an implementation, measurement arm,
or runtime dependency here.

This note defines what “QTIP on native NVFP4” means in PrismaQuant. It keeps
three mechanisms separate so an improvement can be attributed rather than
named after a bundle:

1. QTIP-style Hessian/BlockLDLQ feedback with a native NVFP4 terminal;
2. randomized orthogonal incoherence around a native W4A4 GEMM; and
3. the existing stateful E2M1 trellis wire decoded by Gridbook into native FP4
   tensor-core operands.

The first mechanism is implemented by the research Arm C isolate and produces
ordinary native NVFP4 fields; it is not registered as a production format. The
existing unrotated version of the third mechanism is served by the separately
versioned Gridbook runtime. An external, unpinned Gridbook research reference
implements the online-transform half. PrismaQuant's research producer now
combines that exact metadata contract with its existing physical trellis wire,
without importing or vendoring the Gridbook runtime.

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
the existing scalar globals. The pinned Gridbook E2M1 trellis lane already
decodes PrismaQuant/Gridbook's own `TCQ_E2M1_R256` wire into this E2M1/E4M3
operand pair and calls Blackwell `_scaled_mm`; it does not decode QTIP's
bitshift/tail-biting wire. The external transform reference is unpinned.
PrismaQuant's combined producer emits a research-only rotated artifact for the
existing trellis lane, but the current production pin neither declares nor
loads that transform sidecar.

No result reported by upstream QTIP, no EXL3 or Quartet II result, no synthetic
source-coding result, and no weight-only SSE number is a served PrismaQuant
quality or speed result.

### What the EXL3 source audit contributes—and does not

EXL3 was inspected at commit
`0c49587a7c235e6303a6bbedc8b665272ad3a2ea`; it is not vendored or invoked.
Its useful lesson is architectural, not a request to reproduce its container:

- the quantizer regularizes both matrix axes with sign/scale vectors and
  block Hadamards, then couples those coordinates to a procedural tail-biting
  codebook. Its reconstruction support is therefore much larger than a
  product of 15 scalar E2M1 values;
- the regular kernel path preserves FP16 activations. An optional integer-
  activation GEMV exists for a restricted codebook, but EXL3 is not generally
  the same W4A4 problem as Gridbook's native NVFP4 lane. A quality difference
  can include activation precision and cannot automatically be assigned to
  the weight code;
- EXL3 does require the transformed basis semantically, but not always as two
  standalone online launches. Small-row paths keep or fuse input/output
  Hadamards around the quantized kernel. For long prefill, its reconstruct
  path can emit the weight in the original basis by folding both Hadamards and
  sign vectors into reconstruction, so the following GEMM consumes raw input;
  and
- the transferable pattern is consequently “keep an exact basis contract and
  choose a shape-specific fused execution,” not “adopt EXL3 bytes.” Gridbook's
  research ABI implements the exact contract first; a production proposal
  still needs fused/graph-safe kernels and measured W4A4 quality.

This audit explains why a scalar E2M1 trellis need not match EXL3 and why that
observation is not a theorem against PrismaQuant's target. The target combines
QTIP-derived error shaping and incoherence with native NVFP4 compute while
retaining PrismaQuant/Gridbook's own physical trellis carrier.

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

That identity is the contract for the proposed combined Gridbook arm: rotate
activations before their native E2M1 quantizer, execute the existing W4A4 GEMM
on a weight encoded in the same basis, and apply the inverse output transform
after the GEMM.
Rotating an already quantized activation, or using an untransformed Hessian to
optimize a transformed weight, is a different and invalid experiment.

### Exact diagonal-source block reduction

The GLM corpus supplies a strictly positive diagonal source Hessian

\[
H_0 = \operatorname{diag}(d), \qquad d_i > 0.
\]

For an input transform composed of independent size-`B` normalized Sylvester
blocks and Rademacher signs, write each block as `R_b = H_b S_b`. Then

\[
R_b\operatorname{diag}(d_b)R_b^T
=H_bS_b\operatorname{diag}(d_b)S_bH_b^T
=H_b\operatorname{diag}(d_b)H_b^T.
\]

The signs cancel exactly and the full transformed Hessian is the ordered
block diagonal matrix with those `B`-by-`B` blocks. The producer may therefore
construct and factor one block at a time without constructing a `K`-by-`K`
matrix. This is algebraically identical to factoring the full transformed
matrix: cross-transform-block feedback is exactly zero, while all reverse
BlockLDL feedback within each transform block and every output row remains.
The contract requires `B` to be a power of two, a multiple of 256, and a
divisor of `K`; it binds each ordered block's offset, size, and source-diagonal
hash. A rank-two source, an off-block entry, a nonpositive diagonal, or a
reordered/rehashed block description fails closed.

This reduction is only a structure theorem. It does not prove that the
current 256-state terminal search minimizes a dense local `D` objective, and
it does not establish a quality advantage over EXL3 or any other method.
`dense_block_D` remains unsupported and refuses on both dense and structured
producer paths.

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

The last case describes QTIP's full source-coding mechanism. If such values are
later rounded independently to E2M1, the extra reconstruction support is lost.
QTIP's bitshift/tail-biting serialization is excluded from this project and is
not decoded by Gridbook. The proposed combined arm instead retains
PrismaQuant/Gridbook's existing `TCQ_E2M1_R256` trellis carrier while borrowing
QTIP-derived BlockLDLQ and rotation ideas. That remains a Gridbook wire with
native NVFP4 **compute**, not a stock compressed-tensors NVFP4 file and not a
QTIP artifact.

EXL3 is source-reading context only, so this design makes no claim that any arm
beats it. Any future comparison would have to name the exact support,
objective, scale contract, activation contract, and byte accounting. The
scalar-support theorem proves only the restricted case above.

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

Quartet II is literature motivation only. No Quartet II source, code, or result
is executed or bound here. The existing PrismaQuant `{6,4}` JSO grid is the
control, and Arm C2's seven max-to-level candidates are an existing
PrismaQuant heuristic, not a Quartet II implementation or reproduction. Wider
scale grids remain opt-in until matched-calibration results justify them; none
turns the scale field into a second payload. The former independent group-16
RTN selector is now retired for shaped trellis positions. The fail-closed
two-arm construction and exact zero-rate accounting are specified in
[`trellis_scale_grid.md`](trellis_scale_grid.md).

## Required experiment matrix

All arms use the same BF16 source tensor, activation rows, transformed or
untransformed Hessian as required by the equations above, group-16 E2M1/E4M3
terminal, quantizable-parameter denominator, and deterministic seeds.

| Arm | Offline optimizer | Online transform | Weight carrier | Purpose | Implementation state |
|---|---|---|---|---|---|
| A | RTN + existing JSO | none | stock NVFP4 | native scalar control | implemented in the isolate |
| B | current GPTQ + static activation order + JSO | none | stock NVFP4 | current optimizer control | implemented in the isolate |
| C | QTIP-style BlockLDLQ, native terminal at every decision | none | stock NVFP4 | isolates transferable second-order feedback | implemented in the isolate; not production-registered |
| D | same as C in the rotated basis | input + output sign/Hadamard | stock NVFP4 fields served by Gridbook | supporting incoherence ablation with native W4A4 compute | no stock-field producer and no pinned runtime contract |
| E | transformed-Hessian optimizer | input + output sign/Hadamard | existing PrismaQuant/Gridbook E2M1 trellis wire | target combined rotated-trellis arm | physical one-Linear producer implemented; external runtime reference unpinned |

Arm E's implemented optimizer consumes the complete 256-column block-LDL
cross-block feedback matrix in reverse order within every exact transform
block, and only the same-byte decoded trellis terminal feeds earlier blocks.
For the diagonal-source structured path, feedback between transform blocks is
provably zero rather than omitted. Its local terminal is honestly narrower:
`qtip_frobenius` uses the QTIP-style unweighted terminal and `diag_block_D`
uses the diagonal of the local dense LDL block. The residual cross terms
`2 D[s,t] e_s e_t` are not coordinate-additive and cannot be summarized by
the current 256-state Viterbi state, so `dense_block_D` fails closed. Thus the
producer implements all cross-block off-diagonal feedback, not an exact dense-
`D` trellis minimizer. Its receipt describes local use literally: the
`diag_block_D` mode records diagonal consumption, while both terminal modes
record that off-diagonal and full-matrix `D` consumption and an exact dense
objective are false. The same receipt binds the producer and trellis-encoder
sources plus the audited QTIP commit and source digests. Both local source
hashes are captured at module import and rechecked around final receipt
construction; a self-rehashed prepared receipt cannot change or extend its
fixed basis, wire, seam, scope, or eligibility semantics.

The structured receipt additionally binds the retained diagonal and its
ordered factor groups, including offsets, sizes, transformed-block hashes,
feedback hashes, and `D` hashes. Replay compares those semantics as well as
the canonical wire; matching final bytes alone cannot substitute a different
factor grouping.

Arm E also exposes an explicit research-only E4M3 scale-grid mode. It reruns
the entire feedback recurrence for identity and candidate arms and selects
only per `(output row, factor group)` using the realized fp64 Hessian proxy.
The same decision covers every terminal in the coupled group; finer
superblock/group splicing refuses. Both trajectories share the precommitted
global scale, canonical pack/reparse/decode is mandatory, ties keep identity,
and the selected result must prove exact `Cf=min(C0,C1)`, `Cf<=C0`, identical
wire length, and byte-identical no-win output. This is scaffolding, not a
measured Arm E promotion. A receipt self-digest is not artifact authority:
`require_blockldl_trellis_artifact_replay` revalidates the retained prepared
inputs and explicit recipe, reruns both trajectories, and requires exact wire,
decode, transform, and receipt identity.

Original weight/Hessian hashes remain explicitly preparation-time provenance;
the encode boundary reauthenticates the transformed tensors it actually owns.

Arm D would not be a new compressed-tensors scheme: it needs Gridbook metadata
and runtime transforms even though its matrix operand is ordinary native
NVFP4. Arm E is the actual project target and would retain the existing
PrismaQuant/Gridbook trellis and Gridbook's resident/streamed modes. The
external Gridbook transform reference demonstrates only the runtime-side seam;
it is not the current immutable pin. The Arm E producer closes the physical
wire and reference-algebra seam but not Gridbook loading or serving. Neither
arm may silently fall back to BF16 GEMM, materialize a
parallel weight cache, or choose a residency mode at runtime.

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
- Gridbook would own transform execution, trellis decode, native W4A4 kernels,
  graph capture, residency, and its immutable runtime contract.
- The intended producer/runtime boundary is a versioned manifest plus a future
  immutable Gridbook pin. The current 0.9.1 pin has no online-transform ABI;
  the available external implementation and producer pairing is research-only.
  Neither repository imports the other.
- Transform identities include algorithm, normalization, block geometry,
  input/output dimensions, seed, sign-vector digests, and padding. A mismatch
  refuses at load.
- Only graph-safe seams may cancel or fold transforms. Residual adds,
  nonlinearities, attention softmax, head reshapes, and routed-expert dispatch
  are barriers unless a separate algebraic proof and test covers that exact
  topology.
- Existing `ProductionWeightCache`, activation cache, and streaming prefetch
  paths remain the only cache/residency mechanisms.

The promotion question is empirical: how much of QTIP-derived BlockLDLQ's
advantage survives the native E2M1 terminal, how much comes back from
incoherence, and how much additional value PrismaQuant/Gridbook's stateful
E2M1 wire supplies after both are present. Arm C answers only the first
question. Arm E now has a physical producer and exact reference checks, but it
still lacks a matched GPU quality campaign and a pinned served runtime; E—not
D—is the intended combined target. EXL3 remains source-reading context only,
not an imported implementation or measured comparison arm.
