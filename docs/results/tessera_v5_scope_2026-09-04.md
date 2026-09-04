# Tessera v5 consumer capability — 2026-09-04

## Result and limits

The targeted consumer population passed **512 tests, failed 0, skipped 5**
at implementation `005b007a`. The same PrismaBuild action then compiled the
eight touched Python modules successfully. This is CPU-only evidence on
Sparky (aarch64), one reserved CPU and 4 GiB, with `CUDA_VISIBLE_DEVICES=''`.
There were no collection errors or uncollected test modules. The five skips
all say, verbatim, **`Tessera encodes need CUDA`**, in
`tests/test_tessera_campaign.py` at lines 194, 222, 248, 540 and 945 of that
source. No CUDA encode, kernel, physical serving or whole-suite result is
claimed here. Eighteen warnings were dependency deprecations.

The consumer now understands required per-cell image/execution scope and
retains a complete five-axis `ServingContext` through API-level admission,
menus, candidate masking, provenance and cache keys. Missing or mismatched
scope refuses. Every required regime must match the same target; modes or
images from different cells cannot be combined. Context-free v4 behavior is
preserved, while an explicit runtime query on an unscoped legacy table cannot
be reported as backed.

This is **capability, not Tessera #126 closure**. Production CLI/campaign/export
ingress is a separate follow-up. The reviewed development pin remains
`1221d2a4207a6baeffbe9726bce13125fc1649ae` (lane v4); the release pin remains
PENDING. Final producer review, measured MoE cells, cross-repository development
pin/admission integration and the actual release are not replaced by this
change. There is no new positive MoE attestation or runtime-image default.

## Immutable inputs

Ordinary producer imports use Tessera `1221d2a4207a6baeffbe9726bce13125fc1649ae`.
The explicit v5 interoperability test independently reads the actual publisher
contract from `a4927fe51219e809ab5c3138e4328473be1473f3` (contract 15, lane v5,
eight existing dense cells, no MoE cells). It verifies preserved dense admission
for both published modes and refusal for absent/mismatched scope. No current
working-tree producer bytes or fabricated development answer substitute for
these inputs.

The workers hash-check these source-only Git archives before extraction:

| Source archive under `/mnt/shared/pq-v4-source.o2Tc3O/` | SHA-256 |
|---|---|
| `tessera-1221d2a-src.tar` | `b4755a30d60974ec2758c2060fc4d3954f2e1b7c7bb11602a05f0b783ba60bc8` |
| `tessera-a4927fe-src.tar` | `941b979fe5a246b26a4bf298430bc2ac3727df35c3afd1c406c32bbecd77ffe9` |

`PYTHONPATH` selects the first archive's `src`; only
`PRISMAQUANT_TEST_TESSERA_V5_CONTRACT` selects the second archive's packaged
JSON. Thus the actual-v5 test does not silently advance the installed pin.

## Red before green

All actions below used PrismaBuild, CPU-only Sparky, one CPU and 4 GiB.
Failed actions retain their exact stdout in
`/mnt/shared/prismabuild-fleet/pb-queue/failed/<action>.json`, or
`withdrawn/<action>.json` when a repeated failed attempt was stopped after its
first complete result. Red is not inferred from a collection failure.

| Action | Population / observation |
|---|---|
| `3c040b638f4cb8a94977261fd51e9833e6e4360e939e61bd1ef2802ef4a1e734` | Refined old-code v5 regression set: 68 failed, 1 research preservation control passed, 11 setup errors. Setup errors are not counted as functional red evidence. |
| `4e4e3829c425898c281a68f9b5275d4ef47416f3f75a8df37b1597c309bd38ee` | Corrected-fixture renderer + actual-publisher red: 12 failed, 2 preservation controls passed; no setup/collection errors. Parser v5 was present, renderer/development admission still old. |
| `adf1c7274538d982f914c79414fc1f8a89be0a66b04023ce559280eedd346177` | Explicit legacy-runtime claims: 11 failed, 27 passed; no skips or setup/collection errors. |
| `22d031459c1a16f9777b2724ca4c0930ba0c38a41459f1e8ad01a0fdf41e4e9a` | Recipe completeness: both new tests failed because the result omitted trailing-objective and Gauss-Seidel settings. |
| `bd4b8e9ce69e5aa449e641fcc104e2beb8654970ad497d2291e2e5374771ba66` | Initial integrated v5 core: 114 passed, no skips/collection errors. |
| `6e2e245279dcb53e4dd8ac054c065d583e68f30fc05bc6f3b035d987643d472d` | Strengthened legacy-scope + v5/core checks: 154 passed, no skips/collection errors. |
| `1c9ff1e390476074b99244c1a244aa23bb6eb30d9af6eae7b4f0d9cfdda9a7e0` | Final 33-file targeted population: 512 passed, 0 failed, 5 named CUDA skips; compile checks passed. |

Representative pre-fix lines:

* Old v5 parsing failed at the unsupported schema; malformed runtime tests
  require a field-specific `cells.*runtime` diagnostic rather than accidentally
  matching the word `runtime` in the schema name.
* Corrected renderer red: unbound v5 `tessera_attesting_cells(...)` returned
  four cells instead of `()`; scoped calls raised unexpected
  `serving_context` keyword errors. Actual publisher development admission
  also returned native dense cells without a required context.
* Legacy-scope red: a supplied unsupported runtime context still produced a
  backed family-only admission; regression controls preserve the context-free
  v4 lookup.
* Recipe red at `test_tessera_encoder_recipe.py:31` and `:51`:
  `Right contains 2 more items: {'refit_gauss_seidel': False,
  'refit_objective_trailing': None}`; the second case exposes the same omission
  with normalized per-plane maps.

An earlier renderer attempt (`797412817e490465...`) did not collect because
the test constructed the new context API during parametrization. It is not
regression evidence. The later corrected-fixture red above replaces it.

## Findings fixed separately

* Explicit legacy scope could be serialized beside an unrelated backed claim:
  fixed across generic, development, released-pin, menu and candidate gates
  in `5f064e8b`, with the owner's shared refusal text.
* Two campaign tests pinned an obsolete encoder-key roster. The only failing
  file was rerun against pristine main `b6d6824e5bec0925d088104c8edc879aee240b0f`
  using the same immutable producer: action
  `1584de7ded4f375d3a75326f7c36f7888aaa45c8ed1de08ba97ffbfdc273bf44`
  produced 29 passed, the same 2 failed, and 5 `Tessera encodes need CUDA`
  skips. The failure was `'refit_gauss_seidel' != 'refit_metric'` at the
  hard-coded list's index 2. No whole-main baseline was run. Test expectations
  now derive from producer emissions (`877080af`).
* The packaged-contract test confused newest accepted schema with the actual
  reviewed producer schema; its assertion now reads the publisher field
  (`877080af`).
* `encoder_recipe()` omitted two actual settings. It now projects the producer
  object's `config_block()`, excluding only capture identity and explanatory
  prose (`005b007a`). No encoder input/default changes. Stale keyword-count and
  full-H-for-every-plane prose was corrected on sight.

## Reproduction and receipt

The final action's complete immutable argv, environment and checkout closure
are recorded in
`/mnt/shared/prismabuild-fleet/cas/requests/1c/1c9ff1e390476074b99244c1a244aa23bb6eb30d9af6eae7b4f0d9cfdda9a7e0.json`.
The submission used `tools/pbrun.py --tag sparky --demand cpu=1,mem_gb=4`,
no `--gpu`, `TMPDIR=/home/rob/tmp`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`; inside the reserved action it ran
`/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python -m pytest` with
`-q -ra --tb=short -p no:cacheprovider`, followed by `python -m py_compile`.

The targeted files cover the new context/parser/menu/candidate/renderer tests,
legacy v4 and predicates, lane/export/runtime-fingerprint paths, menu real
table, serving profiles/provenance, campaign, affected allocator legality and
aggregation paths, activation-fair pricing, source passthrough, format-menu
expansion, cost currency/byte pricing, and both documentation checks. The exact
33-file list is in the immutable action manifest rather than a mutable shell
history.

Successful result CAS blob:
`/mnt/shared/prismabuild-fleet/cas/blobs/99/998b27bbb34de2e4ab37462b2363aeaf2be7afdc7ce913fbc297392f83470535`.
Receipt SHA-256:
`a25a4b028f2160071809368eceb5e5d498f69081ed18cc3b9f3d259e69995168`.
Pytest elapsed time was 238.55 s; this is not a performance claim.

PB sealing excludes three external calibration symlinks in the isolated
submission index and restores them immediately after the immutable action is
queued. None of these tests uses the datasets; no symlink removal was committed.
No PrismaBuild source or fleet configuration was changed.
