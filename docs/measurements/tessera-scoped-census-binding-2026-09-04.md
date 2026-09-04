# Tessera scoped census binding — 2026-09-04

Source: `2d92542cdbfbe6cd2b4d30f7ab00631d8f16b488`, based on
`7a5f257a513da687e1d8dad960cba9b8d692dd37` plus the shared target helper
cherry-pick `7abd8100` (upstream `d4b7451f`). This is the receipt/replay leg
of approved Tessera #126, not a new runtime attestation or pin change.

The raw producer `tessera.serving.route_census/2` object is retained intact.
Exact allocation text is bound to the independent card build hash/scope;
config and manifest text are bound to the producer's in-process sidecar
seals and independently read artifact files. Current scoped cells are
replayed per selected source projection, owner and driven phase. Historical
unscoped rows retain their separate comparison without invented context.

## Final targeted population

PrismaBuild action:
`77101e9d68a957a139a0da47ddda66e92096d938faec1ae0acdd3cd765ac880b`.
Result: **148 passed, zero skipped, no collection errors**, 3.75 seconds
pytest time. Five explicitly selected files:

- `tests/test_tessera_scoped_route_receipt.py` (46 new cases)
- `tests/test_tessera_route_receipt.py`
- `tests/test_shipcard.py`
- `tests/test_shipcard_uniform_control.py`
- `tests/test_shipcard_git_provenance.py`

This is CPU-only Sparky/aarch64: one CPU, 4 GiB reservation, no GPU demand,
`CUDA_VISIBLE_DEVICES=''`, OMP/MKL/OpenBLAS threads each 1. The host's GB10
inventory in its worker attestation is not a CUDA test population. There is
no GPU, served-quality, performance or full-suite claim. Pytest reported
14 existing `torch.jit.script_method` deprecation warnings and no skip reasons.
No modules from the requested five-file population were uncollected; other
repository tests were not submitted.

CAS receipt:
`1ba576dedd54650f56210bdfdb1caa81767f7e1a22335512c3acc439f9ce8218`.
Result log SHA:
`2a29bb6162253d45ff951b24b038f3742bb7abe3993dce2dffca7a81c272e976`.
Sealed checkout archive SHA:
`f7f5d94fcc3b78f179ffee9d32760450f59168b814399a80fb9baa908cddcd9d`.
The final source commit differs from that tested tree only in docstrings/comments.

## Red before implementation and before review corrections

All runs used the same CPU-only PrismaBuild population. Pre-fix actions are
retained under `/mnt/shared/prismabuild-fleet/pb-queue/failed/<action>.json`.

| Action | Population and actual failure |
| --- | --- |
| `d9191c66049cb6f720d9a98d3eeeb10ca6e50d1d6e90cde2222aef64aa558910` | 36 initial cases failed on pristine `7abd8100`: new calls raised `TypeError: make_route_census_record() got an unexpected keyword argument 'binding'`; legacy parser regression reported `DID NOT RAISE` when runtime fields were silently discarded. |
| `9c2e6b96c97b7c8004c329c1b0ac2c672c0f74a51dd7515edb17f5103af81db4` | Four added CLI/JSON/schema cases failed on pristine `7abd8100`: legacy CLI raised `SystemExit: 2` demanding `--priced-route`; duplicate-key reader was absent; scoped binding argument was absent. |
| `c10b2204265666655ba47a5b5cc825943bc5c772a2c91ff8306095da61a60dbc` | Two review cases failed on the provisional adapter: artifact-free replay returned `[]` instead of problems; a missing package escaped as `ModuleNotFoundError: no packaged Tessera`. Both now produce explicit refusal. |
| `cbe90b3fd5895c568e077e6cfc4db99fc20dcc30e1c0f192128f56ef6f8a2152` | Passing member-dimension predicate case failed with `DID NOT RAISE`: source member dimensions had been used as executed-unit evidence. Relevant nonempty cell predicates now refuse before resolution. |
| `c84b6ec7a0f9c17224cd6782d26a39b294c5f369a55cfb88fb41a995a406197b` | Correct producer-shaped dense control failed with `structure/state/policy differs from written owner`: the provisional fixture/adapter used `kind=fp8`, while producer telemetry uses `kind=dense`. Both now use the actual wire spelling. |
| `c8de5ad510d505228a6bc01e1e6db1c25365c9452855fe54d7e134bc8629626c` | Three compiled-scope controls failed on pristine `7abd8100` at the absent binding argument. Final controls admit only scoped single-pair compiled MoE, rejecting combined launch symbols and dense compiled attribution. |

Exploratory runs are not substituted for these reds. The initial shorthand
recipe fixture was corrected to actual `data_type=tessera` dictionary rows,
then rerun against pristine source. A predicate fixture initially used
unsupported `eq`; that passing exploratory action `098360d5071d` is **not**
red evidence. The valid `equals` predicate produced the red above. Early
50/144-pass runs predate those corrections and are not the final population.

## Execution and scope boundaries

Every workload used deployed PrismaBuild generation
`aa6d3cfa2f77-1788542034-2b84265567ac` through `pbrun.py`. The worker checked
the immutable Tessera source archive
`/mnt/shared/pq-v4-source.o2Tc3O/tessera-1221d2a-src.tar` against SHA256
`b4755a30d60974ec2758c2060fc4d3954f2e1b7c7bb11602a05f0b783ba60bc8`,
extracted it into a fresh worker temporary directory, and set
`PYTHONPATH=.:<extracted>/src`. Interpreter:
`/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python`; pytest flags:
`-q -ra --tb=short -p no:cacheprovider`.
V5 claims in these tests are explicitly synthetic; the real reviewed pin is
unchanged. The three external calibration symlinks were excluded only while
sealing each snapshot and immediately restored; no removals were committed.

Dense compiled runtime eligibility is not dense compiled *census proof*:
producer telemetry combines launches in a symbolic trace and cannot attribute
them to each driven regime. This adapter preserves that known refusal.
Packed/aggregate source projections, nonidentity runtime owner mappings and
nonempty relevant cell predicates also remain explicit unsupported boundaries.
No adjacent defect was deferred or filed by this receipt implementation;
review findings were corrected in its bounded gate before handoff.
