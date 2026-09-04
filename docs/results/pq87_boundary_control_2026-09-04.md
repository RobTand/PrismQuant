# Boundary control preparation, 2026-09-04

Measured source: `24adebb9` on `codex/pq-87-control-relative`, based on the
endpoint/schema fix `483ff9ca` (PR177). This is CPU implementation evidence,
not a served result and not grounds to close #87 or promote a policy.

## Regressions and populations

All valid runs below used PrismaBuild on Sparky, one CPU, 3 GiB reservation,
`CUDA_VISIBLE_DEVICES=''`, and the existing `prismaquant-cu130` interpreter.
They exercised no GPU-gated tests. No tests were skipped or uncollected in
these explicit file populations; pytest emitted 14 existing Torch deprecation
warnings. No master/full suite was run.

- Existing sampling-input defect, pre-fix action
  `96cf3d26563033250f6d6daa4538bdea12d723a7e37732861926d8a1d47ebdb4`:
  8 failed, 22 deselected. Failure line: `AssertionError: all 0 sampled
  generations clean (5 prompts × 0 reps)`. Empty prompts likewise passed;
  NaN/infinite temperature and malformed bounds reached the HTTP stub.
- New paired instrument, absent-before-implementation action
  `59b306c17287174d1612a4440895e9f06571b266c94dac2d76bd75dac580e2ec`:
  13 failed with `ModuleNotFoundError: No module named
  'prismaquant.boundary_control'`. This proves the new API was absent; it is
  not presented as a numerical before/after measurement.
- Final action
  `4c8600c1dfd16f75ba35813b68f9d8699e47cdc023251e3b8fcb8437d5d1e1c1`,
  CAS receipt
  `4241901c5dcfc83aa2fb5e2ba0858906883778403161975882f86f2fcc33382b`:
  **140 passed, 0 failed, 0 skipped**, 17.39 seconds in pytest. Files:
  `test_ship_boundary_behavior.py`, `test_boundary_control.py`,
  `test_validate_quantized_model.py`, `test_shipcard.py`,
  `test_architecture_doc.py`, `test_docs_staleness.py`. Both new modules then
  passed `py_compile`, and the stdlib CLI's `--help` completed.

An earlier combined run had 139 passes and one failure because the historical
temperature-zero test pins the diagnostic phrase `not sampling`; retaining
that phrase corrected the regression. An initial x86 probe
`09eeba660c2add35b91f6da0c9d036c05fc6ef3565cc1048fcc0552b7d66aafc`
failed on missing `compressed_tensors` in that interpreter. It is environment
evidence only and is excluded from the pre-fix evidence above.

The repository carries three absolute `calibration/*.jsonl` symlinks, which
PrismaBuild correctly refuses to put in a hermetic snapshot. Only those links
were temporarily omitted from these targeted test snapshots; they were
restored to the source worktree immediately after submission. None of these
tests loads calibration data. The module source and test bytes were sealed by
PrismaBuild, and no scheduler source/runtime was changed.

## Scope and next measurement

Separate commits fix stale prose and the online empty/malformed-input defect.
The opt-in instrument records context-bounded control growth, exact paired
prompt/seeds/caps, raw outcomes and control-relative per-stratum counts. It
does not change the mandatory historical 64/zero gate or fill a shipcard.

The physical A/B recipe is `docs/experiments/pq87_boundary_ab.md`. It first
compares raw64 versus chat64 versus an uncensored context-bounded control on
one Qwen BF16 session, with an A/A repeat. Its GPU slot remains ordered after
Tessera #113/#5. No AQUA/Gridbook reproduction, candidate-model discrimination,
power delta, profiler delta or default-policy validation has been measured.

## Review follow-up

The bounded review found three new-instrument provenance gaps and one shared
process-discovery defect. All were fixed on this branch, in separate commits
from the initial implementation: actual token IDs and producer source now
participate in pairing; BF16 attestation reads live dtype/quantization arguments
and refuses aliases, repeated dtype and configuration overrides; shared serve
discovery no longer calls a client a server because its image argument contains
`vllm`.

- Pairing red `2673ed77fb4828b96f64a51690c7658d6d01defc6b3a88fd6090ef94b488de2b`:
  5 failed, 13 deselected. The equal-length/different-token and changed-source
  cases both failed with `DID NOT RAISE <class 'ValueError'>`; the new live-BF16
  guard was absent. The subsequent alias/repeated-argv API test, action
  `4466eeca0bfb`, failed 3 with the `launch_argv` input absent before the fix.
- Shared classifier red
  `384f6e3d064b59cb359adce584b1f257004a767137c81fe7e438c5eaedcbf3ca`:
  3 failed, 5 passed, 10 deselected on dl380 CPU. Failure line: `assert True is
  False` for the measurement client's image argument, a model-path argument,
  and a shell's quoted launch command. Legitimate server/worker cases passed.
- Expanded post-review run `c5bf7e89fbbb`: 225 passed, 9 failed. All nine were
  in `test_tessera_serve_fingerprint.py`: existing installed-Tessera versus
  PENDING-pin/schema drift. Only that failed file was run against pristine
  `483ff9ca`, action `860d5526cb14`: the identical 9 failures, 19 passes.
  The separate release consumer/v4 work owns those rows; no duplicate issue
  or pin change was made here.
- Final source `df44f6d2`, action
  `b5da2968616280679717305628147cec2fdba6fbe484a01e9d41106c30a8b0fd`,
  receipt `4a9b4d22ba9d92b0d096be0c51a516a177f187e1c22c372ba9034dd115100766`:
  **209 passed, 0 failed, 0 skipped** in 17.75 seconds, CPU-only Sparky.
  Population: the original six files plus `test_serve_fingerprint_descendants.py`,
  `test_kl_ab.py`, and `test_measure_served_gold.py`. The already-baselined
  fingerprint file remains red and is not represented as green by this subset.
  All three changed executable modules also passed compilation. No GPU result
  or changed shipping threshold is claimed.

## Exact native weight identity follow-up

Root review found that generic `shipcard.compute_model_sha` represents native
safetensors by names and sizes, so equal-size content changes could retain the
paired instrument's artifact ID. The instrument now combines that existing
metadata identity with `build_weight_content_manifest`, retains the composite
inside each measurement binding, and recomputes its digest during replay.
`weight_stat_attestation` brackets each full hash and the measurement interval;
both content and stats must stay unchanged before a receipt is written.
Generic shipcard identity and the mandatory boundary policy are unchanged.

Pre-fix action
`cf18a78f7fd2e1949d137ee200525960db79c61c4335748cebe0982745645604`
sealed tests-only snapshot `616f7d94db198e0e70fa7df1979cb4ff3a84504c`:
**6 failed, 0 passed, 0 skipped** in 3.01 seconds. Exact regression lines:

- Equal-size replacement: `AssertionError: assert
  '081c4cfa9808771bd1338272b6401b50b3b0058771c04f8304e3e5ab91b94aed'
  != '081c4cfa9808771bd1338272b6401b50b3b0058771c04f8304e3e5ab91b94aed'`.
- During-measurement replacement and change-and-restore: both
  `Failed: DID NOT RAISE <class 'ValueError'>`.
- Missing native weights: `Failed: DID NOT RAISE <class 'FileNotFoundError'>`.
- Stable-content receipt and replay-tamper checks: `KeyError: 'artifact_pre'`
  and `KeyError: 'artifact_content'` respectively, because those exact
  content records did not yet exist. These two failures establish missing
  evidence fields, not an already-existing replay arithmetic defect.

After merging main `b6d6824e` (merge `088264b4`), action
`bbca6ff0659d4a64bf365943c79df5b903b7d0a1ad405146c3fe5e1aa7b4c6bc`
sealed snapshot `79d2218cec02530ac9216e4898f15c9304b924f8`; CAS receipt
`1d99c6cace6e6e5cf2c0474d5771a39edfee9d4f949078c11407fd6e7409bc8f`.
**243 passed, 0 failed, 0 skipped**, 17.79 seconds. Population: CPU-only
Sparky, one CPU, 3 GiB, `CUDA_VISIBLE_DEVICES=''`, the eleven explicit files
from the prior nine-file population plus `test_tessera_serve_fingerprint.py`
and `test_measure_boundary_control_identity.py`. All selected modules
collected; skip reasons: none. No CUDA-gated surface or master suite was run.
The three changed executable modules also passed `py_compile`.

This run used the immutable Tessera `1221d2a` source archive
`/mnt/shared/pq-v4-source.o2Tc3O/tessera-1221d2a-src.tar`, verifying SHA-256
`b4755a30d60974ec2758c2060fc4d3954f2e1b7c7bb11602a05f0b783ba60bc8`
before extracting it on the worker. The nine previously baselined fingerprint
failures are now green with main's reviewed v4 consumer; no release sentinel
was promoted. As above, only the three unused calibration symlinks were
omitted from snapshots and immediately restored locally.

The physical campaign still has to capture the same exact content before
server startup and serve a frozen artifact. These client observations prove
source-content stability during the measurement, not what an already-running
server loaded earlier. No GPU A/B, speed claim, or new policy promotion is
implied by these CPU results.
