# Explicit Tessera serving targets through real endpoints

2026-09-04. Tessera #126 follow-up to merged PrismaQuant #181.
Implementation ends at `cbf44281`; merge `577bd89a` includes current main
`fc5a58c7` (only the architecture provenance stamp conflicted; both retained).
This is CPU correctness evidence, not a serve, quality, performance or energy
measurement. DEV contract `1221d2a`/v4 and release PENDING remain unchanged.

## What is connected

`ServingTarget` accepts exact platform/image/execution mode/residency; the
real campaign and allocator CLIs derive each unit's structure from model/probe
facts, not a global model-name guess. All candidate paths, fallback-route
caches and selected provenance retain those contexts. Global token intake is
the union of actual contexts; final per-unit admission remains authoritative.

Export rechecks selected assignments, recorded target/context and actual
source headers before the external producer translator. The existing shipcard
build interface receives exact allocation SHA-256 and validated scope. Plan
reuse separately binds the full allocation hash, refusing unbound older plans
instead of retroactively stamping them. The printed serve recipe carries the
same image/mode/residency and producer checkout with shell-safe quoting.

## Red evidence

PrismaBuild failed-action JSONs below retain the complete output under
`/mnt/shared/prismabuild-fleet/pb-queue/failed/<action>.json`.

| Action | Pre-fix observation |
| --- | --- |
| `f5170909c3af6734804ba13559943bdc8331040ae475c34abe89423b32a70e8b` | 16 failures: serving-target helper absent and actual CLIs reject the four target flags. |
| `430f151a0e393b44bfdd02ee0e1d040f0cb3d47f94552b8dc5ac84967d38b010` | Corrected-fixture replay: 3 failed, 18 passed. Actual scoped allocator still fails at context-free global literal/token intake before the per-unit gate. |
| `3a7d3cc57a4d6c00159d2c480be90b01f64e6db0033e15d8a084b96eba898d17` | 23 failures: selected export scope/assignment API, CLI forwarding and scope cache identity absent. |
| `9d1902afe7a8d1b51c51d797b284adea2e11d3b8a0acbaaace039c47d9eda036` | 3 failed, 30 passed: `--write-build-json` unrecognized and shell lacks `--build-json`. |
| `f02f32b21d6179c60bb3f662763a2960cb06580c8435ccc0a30a6774925d1892` | Inherited platform regression: expected `sm_121`, received `lm_head` at the scoped sixth policy line. |
| `71765dd3a4ba27d1ce4fa296fde5b92872c861a0b7dbe7f9b7b40aed8bf3e4bb` | 5 failed, 1 preservation passed: changed or unbound allocation still reaches `TEST_EXPORT_REACHED`; printed eager/compiled commands omit `IMAGE` and split paths containing spaces. |

Supplemental earlier export run `fd5e6b5f52f4` exposed five real new cases:
matching source-member predicates were admitted despite unavailable executed
fused projection, body/head policy used context-free admission, and a supplied
Tessera token reached `get_format` as a literal. Two other failures in that
run were test-fixture zero activation scores, corrected without production
changes; they are not claimed as regression evidence. The earlier allocator
fixture also used invalid 128-column shapes; the corrected replay above is
the valid red for global intake.

## Green populations

All actions were dispatched through PrismaBuild on Sparky, CPU-only, one
reserved CPU and 4 GiB, with `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`. No CUDA-gated population was exercised. Tests used
the existing PrismaQuant cu130 interpreter and immutable producer source
archive `tessera-1221d2a-src.tar`, SHA-256
`b4755a30d60974ec2758c2060fc4d3954f2e1b7c7bb11602a05f0b783ba60bc8`.
Snapshot sealing excluded only the three tracked external calibration
symlinks, immediately restored after queue; no test uses those datasets.

- Real allocator endpoint population: **21 passed, 0 failed, 0 skipped**,
  no collection errors. Action
  `371f7cf69f732c973c82dcdbab237c83527bb9de99b2d8be44516fbbf38049a2`;
  receipt `3cd84b4782faa0012f74bb2533d45a88da1ab791b44d217657672ec517625db5`.
- Main endpoint target population at `28e2d5c6`: **408 passed, 0 failed,
  5 skipped**, no collection errors, across 26 touched/reachable files.
  All five skip reasons verbatim: **`Tessera encodes need CUDA`**
  (`test_tessera_campaign.py` lines 194, 222, 248, 540 and 945 in that snapshot).
  Action `194a074fc1bb9d3c8f72c9a0212f2d0390737a67d7dcbf7f2737bed7672a077a`;
  receipt `c299be59a7b246e8f51c23f5156a88482cb469a15a19595eaf917862a99d919e`.
  Seven touched modules compiled and the shell passed `bash -n` in the same action.
- Review fixes at `cbf44281`: **112 passed, 0 failed, 0 skipped**, no
  collection errors, across `test_tessera_plan_input_binding`,
  `test_tessera_export_scope`, `test_tessera_export_lane`,
  `test_stage_settings_guard`, `test_pipeline_contracts`,
  `test_docs_staleness`, and `test_architecture_doc`; shell syntax also passed.
  Action `a77b1dc88916f55ada1941692bba808aed25a4bd9246b106695c40a1f71999ce`;
  receipt `99e107e7f946b4a83f04031fc10d35fb513a07df5265c8590a63351a665be570`.

## Boundaries and encountered fixes

This does not close #126 alone. Scoped raw-census/shipcard replay is a separate
worker's integration; final reviewed producer pin advancement follows actual
measured MoE publication. Packed/aggregate source projections and nonempty
executed-shape predicates refuse rather than guess; packed campaign/bridge
work is tracked and assigned in PrismaQuant #183. Supplied prepriced cost
input dispatch is separately tracked and assigned in #184. No shipped defaults,
wire bytes, thresholds or attestation cells change here.

Separate review fixes: `3b3fce05` binds cached plans to allocation content;
`cbf44281` preserves explicit runtime in printed serve instructions and fixes
contradictory current-tense architecture prose. The initial recipe-provenance
and legacy-explicit-scope findings were already isolated in merged #181.
