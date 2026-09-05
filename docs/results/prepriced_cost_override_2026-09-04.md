# Prepriced pipeline input dispatch (#184)

2026-09-04. This is intake/dispatch correctness evidence, not a performance,
quality, served-KL or GPU measurement. No objective, cost mode, format default,
wire or promotion threshold changes.

## Fix and owner boundaries

`COST_PATH_OVERRIDE` now passes the actual original supplied path to the
allocator and bypasses the entire incompatible local/render/AURA cost-builder
and finalization stage. The new read-only intake module delegates to the
existing schema, currency, research-acceptance and Tessera Hessian identity
owners. Explicit supplied payload mode must match requested `COST_MODE`.
Explicit producer model references must all exactly match the requested model
reference; this is not checkpoint-content attestation.

A local receipt records original lexical path, resolved path and exact file
SHA-256. Immediately before allocation, the driver rechecks resolution and
hash. Input bytes are never copied over or rewritten; report destinations
aliasing input by direct path, symlink or hardlink refuse. Ordinary no-override
cost generation remains unchanged. Allocator per-unit/renderer/coverage and
export gates remain authoritative, not bypassed by preflight.

Two defects encountered in owning validation rules were fixed separately:

- Cost schema now refuses nonfinite cost signals instead of accepting NaN or
  infinities that can enter allocator comparison/objective arithmetic.
- Cost currency discovery recognizes actual Tessera format rows through the
  existing grammar. Usable rows with absent or unknown campaign currency no
  longer evade the currency owner simply because their stamp is wrong.

The shared-owner guards were not copied into the intake module. Error rows
remain error rows; no unavailable measurement is fabricated.

## Red and green

All tests were dispatched through PrismaBuild, CPU-only Sparky, 1 reserved
CPU / 4 GiB, `CUDA_VISIBLE_DEVICES=''`, with both thread-count environment
knobs set to one. No CUDA-gated surface was covered. The existing cu130 venv
used the immutable reviewed Tessera `1221d2a` source archive (SHA-256
`b4755a30d60974ec2758c2060fc4d3954f2e1b7c7bb11602a05f0b783ba60bc8`).

Initial red action
`8d5697e15e8c6a593aa118ff70313e326e55dd285f8be499fb19264707cb4f02`:
**47 failed, 1 passed, 0 skipped**, no collection errors. Intake API was
absent; actual driver sections reached `incremental_measure_quant_cost` and
returned the intercepted builder status 77 instead of allocator status 88
or expected preflight refusal 2. The existing no-override builder path was
the positive preservation control.

Supplemental owner red action
`accd807d92a4361aa62c7bf40637add74391130587001166b74686d20ffb814c`:
**24 failed, 64 passed, 0 skipped**, no collection errors. Exact pre-fix
failure conditions were two symlink-retarget cases `DID NOT RAISE ValueError`,
six missing/unknown currency cases `DID NOT RAISE CostCurrencyError`, and
16 nonfinite cost cases `DID NOT RAISE SchemaValidationError`.

Combined green action
`086f848b54dea8793080e8d9823e16ba69b9128f91930863a475308f38c87108`:
**97 passed, 0 failed, 0 skipped**, no collection errors. Target files:
`test_prepriced_cost`, `test_run_pipeline_prepriced_override`,
`test_cost_currency_missing_stamp`, `test_cost_currency`,
`test_cost_schema_finite`, `test_schema_validation`. The three modified Python
sources compiled and the real pipeline passed `bash -n` in the same action.
Receipt SHA-256:
`36f3393ab02dd01f9f6f0f1a71d56c6adab9e02804281782666132da88add894`;
result blob:
`47f708f0c18b7c1e800015f2659a96fccc438e4f3409462b4b9a9704501df589`.

Failed output is retained in
`/mnt/shared/prismabuild-fleet/pb-queue/failed/<action>.json`; successful
receipts/results are in the PrismaBuild CAS. Only the three tracked external
calibration symlinks were excluded for sealing and immediately restored after
queue; source input semantics and measurement artifacts were preserved.

## Existing endpoint fixture integration

The stronger currency owner exposed three existing scope-test fixtures whose
synthetic Tessera rows omitted campaign currency/mode. The previously passing
file was shown red at that exact missing-stamp boundary (**3 failed, 18 passed**,
action prefix `13a7c8c2187c`), then corrected using the campaign's own constants
without relaxing source gates. The complete endpoint file then passed **21/0/0**,
no collection errors, under the same CPU-only reservation: action
`f305d657783e16e7dbc4615b5864323dc362281087870360b8ad5d6988f51e77`, receipt
`1d3e0bb685ebd2ba306a655fe37416bfe6b5ea46a3d30f11408adf3ecfee08a7`.

## Final main/endpoint integration

After merging main `10dd2395` (endpoint PR185) at `784f1674`, the expanded
15-file targeted population passed **234 tests, 0 failures, 0 skips**, no
collection errors. It includes all six owner/override files, real scoped
allocator endpoints, real shell gates/defaults, stage identity, pipeline
contracts, export scope, plan binding and both documentation checks. Shell
syntax also passed. The only merge conflict was the architecture provenance
stamp; shell source merged automatically. No production change followed this
measurement.

PrismaBuild action:
`49eae99259307543b4bf86b3b172400efffef5cfbf00eb37d52536291122e4cf`.
Receipt:
`5cdaefad68148b88e8d0022b39cdf591b0243b9fbf2ac54af5d6c3d014caf629`.
Result log:
`340529b91d351f0a5705e5d5e419661661ff8d86a7117796ae4e79aac83d5dbc`.
Population remained CPU-only Sparky, 1 core / 4 GiB, CUDA hidden and the same
immutable reviewed producer archive. This is not a full-suite or GPU claim.
