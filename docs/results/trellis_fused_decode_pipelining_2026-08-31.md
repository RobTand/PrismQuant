# Can a fused prologue hide trellis decode behind the MMA? Measured: no, below M≈2100

This answers the pipelining condition stated in
`docs/design/embedded_native_weight_coding_2026-08-31.md` §20.1, which that
document names as *"the single measurement that retires or confirms the
document's largest open risk"* (§20.7.1).

> *"Can per-K-tile decode compute in the prologue hide behind per-K-tile MMA?
> If yes, fused pays the wire's ~2.0 bpp of HBM traffic instead of the tile's
> ~4.5 and wins on both traffic and joules; if per-tile decode exceeds per-tile
> MMA, fusion loses regardless of round-trip cost."*

## Setup

GB10 sm_121, pinned gridbook **0.9.1 / `227420f`**, one 4096×4096 weight tile at
`q256=512`, CUDA-event timed, 20 reps after 5 warm-ups (10/3 for the MMA sweep).

- **decode** — `prepare_wire_cuda(...)` once, then `prepared.decode_native_packed()`
  per call: the lane's actual hot path, native v2 TCQ (ABI 3).
- **MMA** — `torch._scaled_mm` fp4×fp4, i.e. the
  `block_scaled_ue4m3xe2m1` mainloop a fused prologue would hide behind, with
  the group-16 scale planes in the cuBLAS 128×4 blocked layout.

Decode measured **236.9 µs**, against the lane document's **206.4 µs** for the
same shape (`trellis_e2m1_lane_2026-08-29.md:77`) — this harness re-validates
per call, so it runs slightly hot. The agreement is the sanity check.

## Result

| M | MMA µs | decode/MMA | verdict |
|---|---|---|---|
| 1 | 50.6 | **4.68×** | fusion loses |
| 16 | 47.4 | 4.99× | fusion loses |
| 64 | 30.8 | **7.68×** | fusion loses |
| 256 | 47.6 | 4.98× | fusion loses |
| 512 | 67.4 | 3.52× | fusion loses |
| 1024 | 118.9 | 1.99× | fusion loses |
| 2048 | 216.5 | 1.09× | fusion loses |
| **4096** | 410.4 | **0.58×** | fusion can hide decode |
| 8192 | 889.4 | 0.27× | fusion can hide decode |

**Crossover at M ≈ 2100–4096.** Decode is flat in M (as the lane document
measures); MMA is linear above M≈512. A derivation from the two numbers already
in the tree — 206.4 µs decode against the route probe's 50.9 µs MMA at M=512 —
predicted the crossover at M ≈ 2076, and the measurement brackets it between
2048 (1.09×) and 4096. The condition was decidable from existing measurements;
this confirms it.

## The consequence is an inversion

Fusion pays only in large-prefill regimes. **The traffic win exists for the
bandwidth-bound decode regime**, and at M=1 decode compute is **4.68× the MMA it
would need to hide behind**. The feature is decode-bound exactly where it was
supposed to help.

Two details that block the obvious escapes:

- **Below M≈512 the MMA is flat** (50.6 / 47.4 / 30.8 / 47.6 µs) — launch-bound,
  not math-bound. There is no small-M regime where the ratio improves; it gets
  *worse*, peaking at **7.68× at M=64**.
- **Nothing saturates the box**: peak 8.5 W of the ~140 W envelope. Both sides
  are latency/launch-bound rather than throughput-bound.

## Scope, and one failed attempt worth recording

**This is a decomposed benchmark, not a fused kernel.** A real prologue decodes
per K-tile and may pipeline better than one monolithic decode kernel. That is
the single way the verdict could soften, and it is why this should be
independently reproduced before it is used to kill a segment.

**The first version of this benchmark was wrong and reported 720 ms.** It called
`expand_wire_cuda(wire)` per iteration, which rebuilds the plan tensors from the
Python `TrellisWire` every call; it measured host-side list→tensor conversion
over 16.7M positions at **11 W — an idle GPU** — and labelled it decode. The
code comment predicted this would "overstate decode slightly"; it overstated by
3,500×. The tell was available at the time and was not read: **power**. A decode
kernel that leaves the GPU at 11 W of a 140 W envelope is not running
(principle 15 — read power against the envelope, not utilization). The prepared
owner exists precisely to avoid this and deliberately exposes no tensor getters;
the correct hot path is `prepare_wire_cuda` once, `decode_native_packed()` per
forward.
