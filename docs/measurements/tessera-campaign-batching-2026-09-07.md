# Campaign batching: input and journal qualification

PrismaQuant #300 consumes Tessera #386 (`382a1a97`) for the #275 campaign.
`--anchor-batch-size` is opt-in; the scalar default remains one. The campaign
keeps its adaptive pending-anchor schedule and groups compatible expert units
inside each admitted PrismaBuild action. It uses the existing resident source
weights, captured scoring rows, ActivationSource memo, production scorer and
ProductionWeightCache. Tessera owns batch encoding. No serving release pin or
wire recipe changes.

The scalar and batch adapters share one input guard. Batch calls carry separate
per-unit Hessian mappings, retain the scalar artifact name, and decode each
result through the existing wire reader. The campaign scores each unit with its
own served activation scale, then writes its normal cache/wire entries and
journal receipt. Batch width is an execution setting excluded from checkpoint
input identity; batch timing is explicitly apportioned among its units. Missing
batch API support refuses before model load. A width-one run keeps the previous
ten-anchor journal flush cadence.

## Correctness evidence

All execution used PrismaBuild. The initial three adapter/grouping regressions
failed on main `70e5f7a8` because the batch seam did not exist. The final new suite
passed all six tests on CUDA in the known producer image
`prismaquant-qwen38-producer:20260827-tf516-hf128`, Python 3.12 / Torch 2.13+cu130:

- Action `a25b803d46b5b611292a00d724bd7c7353d374f1b6b98f6daf2f50069ced6599`,
  Sparky, exit 0, six passes, no skips, 21.63 seconds.
- CAS receipt `0b811bd4d82ed8ce2a800101d92c10a951a5d8f8ab71eed9ffb66b632bdcb10e`.
- Actual `campaign.main()` on a routed fixture encodes seven source units with
  each unit's own Hessian. Batches of four and two reach the real producer.
  Wire bytes and numerical cost rows equal the scalar run. Resuming with width
  two performs no encode, and a tampered wire refuses resume.
- Separate tests retain distinct static activation scales, refuse missing H,
  split incompatible shapes/rungs, and refuse an old producer before loading.

The relevant distinct coverage totals 150 tests: the existing campaign suite
(46 CPU plus five CUDA), new batch suite (six CUDA), packed campaign (16),
fanout (33), journal merge (10), architecture (13), document staleness (six),
and stack sample costs (15). CPU runs explicitly skipped CUDA cases; all those
cases were subsequently executed on CUDA. The first combined GPU run's sole
failure was a new fixture replacing every registry row; restricting the fixture
to its target format corrected it. The initial real CLI fixture used 64-wide
weights, which the normal pricing contract correctly refused; it was widened
to 256 before qualification. An expensive CPU encode rerun was withdrawn through
PB and moved to the GPU. None of those attempts is performance evidence.

Receipts/logs for the final GPU gate are retained under
`/mnt/shared/tessera-measurements/pq300-batch-20260907/`.
The CPU reports are `/home/rob/tmp/pq-batch-contracts.json` and
`/home/rob/tmp/pq-batch-finalcpu.json`; their CAS receipts remain authoritative.

## Calibration preparation and its limit

Retained input: `/mnt/shared/tessera-measurements/tessera385-2026-09-06/units/`,
WikiText-2 train, 32 samples of length 512, seed zero. Both exact corpus and token
hashes matched before the new forward:

- text: `aee724fa58bfbdeb3fc6803297fb6bab27b203d7c40b39ddef9b9770e5d52fe5`
- IDs: `809883cb91e1359d0b7e7829f4a7ab3ddeacd7eee75a66dd6866aa671a4f935d`

The current container forward routed no rows to layer-18 expert 6; the collector
refused, as it must (action `ba0a368758de697b`, exit 1). This is not a complete
campaign calibration. An explicitly bounded validation recapture excluded that
one unit and retained the other 31 expert w1 units plus dense layer-0 w1, using
the same collector and exact tokens. It saved 512 scoring rows per unit where
available, the existing export H/scales files, the token tensor and corpus text.

Action `539fa6315350c4182922d091f2faa33afca5a3b58f25973c87a0a99ee80d77d4`
completed on Sparklina with exit 0; receipt
`0134db4fd8b9574554164c8b84f0d3beef9f398896a991df4c9a7fe5b9ddc952`.
Files are in `/mnt/shared/tessera-measurements/pq300-batch-20260907/capture/`.
Dense H equals the retained H exactly. All 31 expert H tensors differ, despite
identical tokens and source weights; the largest absolute H-entry difference is
777.96484375. This establishes a forward difference, not its cause. Every count
and comparison is in `capture.json`. No old expert cost is relabelled as a
measurement on this new capture.

The real cost-row A/B and paired in-process/host profiles are pending. No
throughput or energy improvement is claimed by this correctness record.
