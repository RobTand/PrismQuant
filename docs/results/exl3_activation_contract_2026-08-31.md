# What EXL3 actually executes, and what that makes the target

Read from the EXL3 source at `/home/rob/dq-runs/exl3-reference-20260830`
(reference material only; no product lane).

## The contract

```
exllamav3_ext/quant/exl3_gemm.cu:26
  "- A: row-major A tensor, shape (m, k), dtype float16, contiguous"
exllamav3_ext/quant/exl3_gemm.cu:132
  TORCH_CHECK_DTYPE(A, kHalf);
```

**EXL3's batch GEMM is W4A16.** Weights are trellis-coded 4-bit; activations
are fp16 and the kernel dequantizes into an fp16 multiply. A separate
`exl3_gemv_int8.cu` exists for the decode regime; the batch path takes `half`.

Our attested lane is **W4A4** (`e2m1_group16_ue4m3_static`): both operands
4-bit, multiplied by the Blackwell `block_scaled_ue4m3xe2m1` mainloop.

## What that does to the comparison

It splits the goal into two claims of opposite difficulty, and they should
never be stated as one.

**Speed — ours, structurally.** fp4 x fp4 is a native tensor-core route
measured at **4.2x the bf16 matmul** on this hardware; EXL3 runs an fp16 GEMM
and moves 4x the activation bytes. This should hold without any cleverness.
(Single-instrument profiler numbers; a speed *claim* still owes a power series
per principle 15.)

**Quality — harder for us than for EXL3, because of the same contract.** EXL3
pays no activation quantization error at all. Matching it at equal bpw means
absorbing a 4-bit activation perturbation it never carries. Beating EXL3 at
W4A4 is therefore a strictly harder bar than beating it at W4A16, and a
narrow loss on quality at equal bpw is **not** a clean defeat — it is the
handover's standing caveat in its exact form: *"W4A16 EXL3 results are not an
upper bound on a native W4A4 lane without matching activation and cost
contracts."*

**The honest goal statement** is therefore: *equal or better quality at matched
bpw while additionally quantizing activations to 4 bits, and substantially
faster.* That is the differentiated product. It is not "beating EXL3 on its own
contract", and describing it that way would overstate a real result.

## Corollary for the two-type ladder

The 8-bit type is the portability story as well as the overlap story:
`torch._scaled_mm` fp8 x fp8 exists on Ada, Hopper and AMD via hipBLASLt, while
fp4 x fp4 is Blackwell-only. An FP8 datatype housing low-rate trellis data is
what an AMD RDNA 3.5/4 build runs.

Measured side-information asymmetry, which decides who owns the overlap on the
weight side:

| at body 3.5 bits/weight | scale plane | wire bpw | shaping |
|---|---|---|---|
| 4-bit type `TCQ_E2M1_R256` | group-16 ue4m3, **+0.504 bpw** | **4.004** | 3.0 shaped + 0.5 bypass |
| 8-bit type `TCQ_E4M3_R256` | per-row fp32, ~**+0.03 bpw** | **~3.53** | fully shaped (cap 7) |

So in the region where the 4-bit type sags, the 8-bit type is *cheaper in bytes
and fully shaped*. AQUA arbitrating that overlap is not a workaround; it is the
8-bit type dominating on the weight side, against a W8A8 activation contract
that costs 2x activation traffic and drops the GEMM from 4.2x to 2.0x bf16.
Pricing exactly that trade is what AQUA is for.
