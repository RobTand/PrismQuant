# Sparse Tessera curve replay and the activation/prefill axis

Status: opt-in research replay, 2026-09-05. This capability emits a report and
measurement requests, not allocator costs or serving qualification. The normal
Tessera campaign retains its piecewise log-linear interpolation and adaptive
measurement policy.

## Recovered mechanism

The former Gridbook campaign used a shared projection-level error curve and a
per-expert correction. Two measurements fitted both a level and a slope;
separate audit measurements decided whether to use predictions or measure the
full declared grid. The retained implementation is in
`tools/dsv4_afast_campaign.py` and `tools/dsv4_afast_burn.py`, originally adopted
in `0b34145c17f76a2ec715af13e18715b25b0007af`. The earlier study survives at
`629cfb79958a4d4400471d8044b5df93b061908f`, under `interp-diagnosis/` and
`cost-ldlq/interp-diagnosis/`.

That history supports recovering the mechanism, not transferring its constants
or declaring all ranges predictable. The K28–38 median study recorded worst
rung errors of 3.50%, 5.87% and 6.34% on held-out L0 gate/up/down projections.
Broader ranges and heterogeneous experts produced larger misses. The archived
480-solve regret study used a weight-MSE proxy with uniform sensitivity, not
served KL or AURA. Its tolerances are not Tessera promotion gates.

`prismaquant.anchored_shape` now supplies the numerical mechanism shared with
the existing strict AURA fitter. Given a declared segment shape G and a
centered/scaled rate coordinate x:

```text
log10 D_i(x) = log10 G(x) + a_i + b_i*x
```

The pilot fit removes each pilot unit's own log-cost mean before solving its
feature coefficients. One anchor sets b to zero; two fit level and slope.
An explicit later fit can use more observations. Real measured points remain
exact. Invalid costs, unsupported keys and insufficient design rank refuse;
there is no fabricated epsilon loss or inherited Gridbook modulo-four term.
The strict `anchored_cost` AURA wrappers keep their currency, render-receipt
and segment checks.

## Replay inputs and output

Run from the repository with the same Python dependencies as the campaign:

```bash
python tools/tessera_surface_replay.py \
  --costs /run/cost.pkl \
  --checkpoint /run/anchors.json \
  --plan /run/replay-plan.json \
  --out /run/replay-report.json
```

The input is a trusted local campaign pickle plus the existing checkpoint
manifest, its `.parts` journal, and its recorded wire directory. The plan
binds the exact payload SHA-256 and checkpoint identity SHA-256. The importer
reuses the journal reader, checks measured rows against recorded anchors and
source/encoder/Hessian/static-scale identities, and verifies wire hashes and
sizes. It does not re-read model tensors or re-run producer wire validation.
Its report states that boundary explicitly; recorded identity replay is not
fresh source or serving attestation.

The plan schema is `prismaquant.tessera_anchored_replay.plan.v1`. It declares:

- `input.payload_sha256`, `input.checkpoint_identity_sha256`, and `currency`;
- a list of `segments`, each with an `id` and a `descriptor` identifying
  family, explicit profile role, activation contract, geometry, wire recipe
  without the rate, and Hessian applicability;
- `features_by_key` and integer rate `coordinates`, keyed by format name;
- `pilot`, mapping pilot unit names to measured format keys;
- `heldout`, mapping distinct unit names to one/two `anchors` and separate
  `audit` keys;
- `max_absolute_log10_error` and optional `refit_after_audit`.

The exact example construction and executable cases live in
`tests/test_tessera_anchored_surface.py`. Roles are explicit plan declarations;
the importer does not certify that a declared transfer class generalizes.
Actual family, activation, geometry, recipe and Hessian boundaries must agree.
The pilot covers the target rate envelope; held-out units are distinct from
pilot units, and audit rungs are distinct from their fit anchors.

Audit predictions are frozen before audit truth can enter an optional refit.
Reports retain the pre-refit errors, measured/predicted provenance and a
piecewise log-linear baseline where two anchors bracket the audit rung.
Missing evidence, failed audits or nonmonotone curves produce deterministic
measurement requests. Requests propagate to members of existing campaign
anchor groups. They are requests to the existing measurement machinery, not
newly encoded wires. Packed experts never become selectable from this report.

Replaying identical inputs is deterministic. A changed report cannot overwrite
an existing output pathname; choose another path. The plan hash binds the split,
features and policy. No speedup, full-model quality improvement, or production
qualification is implied by a successful replay or its numerical tests.

## AURA and overlapping activation families

The importer currently accepts only
`output_mse_under_route_activation_contract`. That is the actual campaign
currency. AURA import requires its own attested measurement path; MSE cannot
be relabeled as AURA. Interpolate within activation/recipe segments, then
compare candidates in a common validated downstream quality currency.

Current AURA propagates weight deltas through KL/Gauss–Newton adjoints. Current
Tessera output-MSE scoring includes the local joint weight/activation residual
but discards its direction before scalar sensitivity weighting. The proposed
bridge projects the joint residual through AURA's output cotangent G:

```text
deltaY = Xhat What.T - X W.T
       = X deltaW.T + deltaX W.T + deltaX deltaW.T
a[k] = <G[k], deltaY>
predicted_loss = 0.5 * mean_k(a[k]**2)
```

This must preserve signed cancellation across terms and repeated invocations
before squaring. The archived signed-operator helper at
`32f6b03fc063731b927ad33d143cac923c4950e9` contains an efficient equivalent:
compute `D = G.T @ deltaX` once per activation contract, then add its projections
against W and deltaW to the existing weight projection. Its archived harvester
and persistence wiring were unfinished. Reuse requires adapting today's
activation-scale contracts, completing signed sample persistence, and testing
fixed-teacher comparisons; enabling that archive is not an implementation.

Quality and runtime then form distinct axes. The proposed operator control is
best quality at an exact byte/device budget while meeting a specified prefill
latency target, with a separate decode guard. The existing prefill SLO flags
filter assignments after a bytes/quality search; they do not preserve faster
same-byte alternatives that search discarded. Candidate reduction and search
must retain the discrete bytes/quality/runtime frontier. A weighted penalty
sweep or convex hull cannot represent every nonconvex feasible choice.

Use measured kernel/workload costs to propose assignments and end-to-end
serving measurements to qualify them. Activation width, encode time and weight
bytes do not determine prefill time. Serialized weights, resident terminal
weights, activation scratch and KV memory also require distinct accounting.

Validate close cross-family choices using paired probe differences and shared
held-out calibration, then whole-assignment/close-swap checks against the fixed
teacher. Probe covariance is not layer-error additivity. Re-centering a Fisher
square on a quantized assignment alone changes its reference and omits the
fixed-teacher first-order term. Current historical AURA validation does not
establish these overlapping families' rankings or cross-layer additivity.

This joint AURA/runtime extension and its required current measurements are
tracked in [PrismaQuant #237](https://github.com/RobTand/prismaquant/issues/237).
The separate opt-in grouped-uncertainty defect is
[#236](https://github.com/RobTand/prismaquant/issues/236). Production defaults,
serving admission and the existing uniform-control/quality gates do not change
through research replay.
