# #87: opt-in paired no-new-boundary-failures policy

The chosen rule is `prismaquant.no_new_boundary_failures/1`, implemented as
an opt-in decision over the existing paired instrument. This is a policy
implementation receipt, not a quantized-artifact measurement or a promotion
of the production gate.

`tools/measure_boundary_control.py candidate --decision-policy no-new-failures`
records the optional decision after measuring the candidate. Accepted exits
0; refused exits 2. Without the flag the original advisory comparison remains
unchanged. No shipcard slot or production threshold moves.

The decision first replays the exact control and candidate measurements:
prompt/seed order, source/content bindings, same runtime/campaign/host,
tokenization, the control's cap-growth stopping rule, raw outcomes and scores.
It then compares defect-kind sets on each matched pair. New defects on a
formerly clean pair refuse, as does any new kind on an already-broken pair.
Repairs cannot offset newly broken samples, even in the same stratum. Existing
control defects can persist or disappear. Candidate truncation at the valid
control cap is a measured defect; a censored control is inconclusive.

The offline decision replay recomputes every field, including the policy ID,
violation list and counts. Canonical JSON comparison also refuses coercions
such as integer `0` to boolean `false`; Python object equality alone cannot
distinguish them.

## PrismaBuild evidence

All actions used frozen deployed runtime `aa6d3cfa2f77`, Sparky with one CPU
and 4 GiB, no GPU reservation and `CUDA_VISIBLE_DEVICES=''`. These are serial
CPU fixtures, not CUDA coverage. Each named module collected; zero skips and
zero collection errors in the final population. The three unused absolute
calibration symlinks were excluded only while sealing each immutable snapshot
and restored immediately afterward; their removals were not committed.

| Evidence | Action | Observed result |
| --- | --- | --- |
| Policy absent, before implementation | `104fcc9e2cd6d09eaab27a8dff91a6b9f799120daa23ece339591079961ad806` | 18 failed: `AttributeError: module 'prismaquant.boundary_control' has no attribute 'decide_no_new_failures'` |
| Real CLI before optional argument | `acc7237eb6eedaf57f015468e10223296e2012ee89258b0de9e8cb926f477925` | 2 failed, 18 deselected: `unrecognized arguments: --decision-policy no-new-failures` |
| Strict receipt counter types before correction | `5c55d41d4d14bccf101a8577b01bebf32b59549f38940ca79f0ed794b2ef00b0` | 2 failed, 20 deselected: `Failed: DID NOT RAISE <class 'ValueError'>` |
| Final seven-module population and compile | `bfebe89b876b6bab4bb4b5fb27717df2483d112be549eb7a43179f0daf46042f` | 105 passed, 0 failed, 0 skipped; compilation succeeded |

The red wrappers returned success only when pytest exited 1, preventing the
pool from retrying an expected failure. They are red test evidence, not green
test runs. Final receipt SHA-256:
`f3cd6012edb901c2570f8d22756e9b9e2fbb6726fadf8e06e763facb2a5ea5d3`.
Final result blob:
`966227b799e5bdc102f3f7b0fea39fbaeac8c42a6d92f6b6e2e3c7b01b87604f`.

The selected modules were `test_boundary_policy.py`, `test_boundary_control.py`,
`test_measure_boundary_control_identity.py`, `test_pq87_physical_ab.py`,
`test_ship_boundary_behavior.py`, `test_docs_staleness.py`, and
`test_architecture_doc.py`. No master baseline or full suite was run.

## Remaining validation

The existing Qwen3-0.6B BF16 A/B establishes that the old 64-token cap
censored this population; it does not establish quantized discrimination.
A fresh matched BF16/quantized campaign must freeze its artifact identities
and disjoint held-out prompts before serving. The completed raw/chat64
instrument characterization need not be repeated. Held-out/downstream checks
and an additional representative population remain prerequisites for any
default promotion. A prior receipt from a different source closure or campaign
cannot be reused as if it were that campaign's BF16 control. #87 remains open.
