# PQ237 joint AURA: fixed-teacher numerical screen, 2026-09-05

The joint output-residual decomposition agrees with the independent GPU
oracle, but this bounded screen establishes **no selection-quality gain**.
Weight-only and joint AURA selected the same assignment. Joint pricing made
slightly fewer correct neighboring rankings, and the minimum-KL assignment
had worse heldout NLL than the three selected-unit uniform controls. The
feature remains research-only; PrismaQuant #237's numerical generalization
and actual serving-runtime qualification remain open.

## Measured scope

Source: PrismaQuant `5af973530baec22e96b5b8121a83c65f9ed7da49`, the validation
clone's cherry-pick of joint implementation `dc5337c266fddc6a820e5777352a08e85ef50bc5`
and its prerequisite. Joint helper SHA-256:
`2699cdee9e2b0112d33a65e9028926eefa2ba2d31587bff89708014fb357b944`.
The harness hash, complete model-file hashes, source closure, image identity,
activation policies, token IDs, blob/render hashes, and raw evidence hashes
are in the [receipt](../../experiments/results/pq237-joint-aura-20260905/receipt.json).

The model was local Qwen3-0.6B, running resident BF16 on NVIDIA GB10 with
Torch `2.13.0+cu130`, Transformers 5.15.1, and the known teacher Docker image
identified in the receipt. Tessera producer:
`ba582d476a3b6db9057ebd1385dc52926f171451`; every extracted source file was
verified against its exact Git archive. The container had four CPUs, 48 GiB
RAM, and native/Torch threads fixed to one. Direct Docker execution was
explicitly authorized by the session's user, who prohibited PrismaBuild.

Three down projections, layers 0/7/21, were quantizable in this screen:
9,437,184 parameters total. Other model parameters remained BF16. The nine
production cache renders covered E2M1 K1 R768 (A4), E4M3 K1 R896 (A8), and
BF16 K1 R896 (A16). Both pricing arms and every candidate forward consumed
the same rendered tensors and shared activation-QDQ implementation. The
existing activation cache captured calibration inputs; no separate cache or
rendering mechanism was introduced. All nine renders were resident
(56,623,104 bytes) before probing or assignment evaluation.

Two committed authored texts supplied 128 calibration tokens. Four distinct
committed authored texts supplied 256 heldout tokens. Each text was truncated
to 64 tokens, without padding; causal KL/NLL therefore covered 126 and 252
positions respectively. These texts constitute a reproducible numerical
screen, not a public language benchmark. Sixteen aligned Rademacher probes,
seeds 237000–237015, used the fixed teacher and full-vocabulary causal Fisher.
The teacher was never recentered on a candidate. The final selected source
weights were checked for exact restoration.

## Results

All 27 assignments were measured on the same heldout tokens. The declared
budget was 4,306,135 bytes: the midpoint between the smallest and largest
assignment sizes, computed from serialized blob lengths before reading
quality. Twenty assignments fit it. Bpp below counts only these three
quantizable units and includes their complete serialized blobs.

| Assignment on the three selected units | Bytes | Bpp | Heldout KL | Heldout mean NLL |
|---|---:|---:|---:|---:|
| Uniform A4, E2M1 K1 R768 | 4,426,017 | 3.75198 | 0.01212686 | 4.12355042 |
| Uniform A8, E4M3 K1 R896 | 4,186,254 | 3.54873 | 0.00785002 | 4.13192368 |
| Uniform A16, BF16 K1 R896 | 4,235,421 | 3.59041 | 0.00781205 | 4.13843536 |
| Weight-only selection: L0 A16, L7 A16, L21 A8 | 4,219,032 | 3.57652 | 0.00735440 | 4.14729166 |
| Joint selection: L0 A16, L7 A16, L21 A8 | 4,219,032 | 3.57652 | 0.00735440 | 4.14729166 |

Uniform A4 is an out-of-budget diagnostic. The unmodified teacher's heldout
mean NLL was 4.12383026. The selected assignment was also the retrospective
lowest-heldout-KL assignment among the 20 feasible choices, giving both
methods zero observed KL regret on this finite set. That oracle was never
used for selection. Identical selections give an exactly zero paired
assignment difference and zero probe standard error; this is not evidence
of generalization certainty.

There were 81 pairs differing in exactly one selected unit. Weight-only
pricing ordered 66 correctly by observed heldout KL; joint pricing ordered
65 correctly. The two methods disagreed on nine pair orderings. These 81
pairs are related comparisons, not independent trials. They demonstrate that
joint residual pricing can change decisions, but they do not establish a
better ranking model. The KL/NLL disagreement is a second reason the local
quadratic surrogate alone cannot authorize promotion.

For the first aligned probe, all nine signed decompositions were independently
checked against `sum(G * (Xhat @ What.T - X @ W.T))` in FP32. Maximum absolute
error was **2.685701474547386e-7**, below the declared
`2e-5 + 2e-4*abs(direct_projection)` tolerance. The lease recorded 96 activation
QDQ calls and 96 shared operator GEMMs across the 16 probes, with zero
persistent cache entries owned by the lease. Raw signed components and all
neighbor decisions are committed alongside the receipt.

## Execution evidence and limits

Final run `e` exited **0**, completing nine renders, 16 probes, all 27
assignments, the restoration assertion, and both Torch profiles in
144.785 seconds. Targeted compile checks for both experiment modules exited
0. Earlier driver failures and the superseded successful run are recorded
in the receipt, with their practical fixes; none were counted as a pass.

Raw evidence directory:
`/home/rob/tessera-runs/issue237-2026-09-05/validation/experiments/results/pq237-screen-20260905-e/`.
Terminal log: `/home/rob/tessera-runs/issue237-2026-09-05/validation-screen-e.log`.
Netdata: `/home/rob/tessera-runs/issue237-2026-09-05/validation-netdata-e.json`.
Tessera closure verification:
`/home/rob/tessera-runs/issue237-2026-09-05/validation-tessera-source-receipt.json`.
These artifacts are retained as bounded evidence; the predecessor runs are
superseded, and their failures are described in the receipt.

Both profiles cover three warmed QDQ model forwards; CPU/CUDA summaries are
committed, with full Chrome traces retained locally and hashed in the
receipt. Netdata supplies 147 host samples covering the run, including CPU,
RAM, I/O, and GPU power: 12–52 W, mean 20.868 W. This small experiment did
not saturate the GPU power envelope. Host CPU telemetry includes concurrent
agents' CPU work. No claim of GPU saturation or speed improvement is made
from utilization percentages or from these profiles. Both selected
assignments are identical, so there is no distinct serving-speed comparison.

**No actual served operator timing, TTFT, or decode price was measured or
supplied to the runtime-aware allocator.** QDQ execution and encoder time
cannot price an actual serving route. This screen also lacks a second model,
disjoint heldout units/rungs, a sparse-anchor fit, whole-model uniform
controls, end-to-end serving smokes, and downstream task evaluation. The
software can consume properly qualified runtime receipts, but this run does
not provide one or close #237's serving qualification.

The [experiment guide](../../experiments/pq237_joint_aura_screen.md) documents
the exact reusable command and source-manifest preparation.
