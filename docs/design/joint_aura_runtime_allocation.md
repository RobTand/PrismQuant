# Joint AURA and runtime-constrained allocation

Status: research, opt-in. Tracked by PrismaQuant #237. This extends the
streamed AURA producer, production weight cache, allocator candidate builders,
and existing serving-constraint controls. It does not change the default
format menu, Tessera runtime pin, or serving admission.

## Measurement and objective

For one Linear, a route changes both its input and its rendered weight.
With `dX = Xhat - X` and `dW = What - W`, the complete local perturbation is

```
dY = X dW.T + dX W.T + dX dW.T
a[k] = <G_Y[k], dY>
cost = 0.5 mean_k(a[k] ** 2)
```

The streamed producer's `joint_activation=True` mode (CLI
`--joint-activation`) projects these terms through its downstream KL
cotangents. Terms and repeated invocations of the same Linear are summed
while signed, before squaring. The existing production cache supplies actual
rendered tensors. The shared activation QDQ owner supplies the route's
activation behavior, including calibrated static scales. The lease owns no
additional activation or weight cache. Resident FP32 projection products
reuse one activation-perturbation GEMM across matching activation policies.

Joint rows retain their signed components, aligned probe samples, source and
render tensor identities, activation policy/scales, projection arithmetic,
calibration and probe identities. A joint price already includes its
downstream Fisher and activation error; scalar sensitivity, calibrated gains
and AQUA activation transfer must not be applied again. Joint and weight-only
or output-MSE rows cannot share a joint allocation table. BF16 controls carry
complete measured zero rows rather than an unobserved zero placeholder.

Common probes enable paired differences. Their conditional sampling error is
not uncertainty over new calibration data or evidence of generalization.
The additive sum of local quadratic prices is still a model approximation;
it does not establish the quality of every joint assignment.

## Runtime input and search

`--measured-runtime-table` opts the allocator into measured-resource search.
`--measured-runtime-context` supplies the independently expected context. The
existing `--slo-prefill-p95-ttft-ms` sets its prefill proposal budget; optional
decode ITL and device-memory limits remain separate constraints. This mode
does not mix the legacy family-relative dispatch table with exact operator
measurements. A throughput floor cannot be inferred from operator medians.

The versioned table binds the actual cost-payload digest, source/calibration,
GPU and full runtime identity, prompt length, batch, TP, graph/residency mode,
and exact operator routes. `source_sha256` equals the streamed source model
identity's `content_sha256`; `calibration_sha256` equals the joint AURA probe
identity's calibration digest. The CLI compares these against the cost rows.
Every row binds actual joint operator identities,
member formats and shapes. Timings are repeated GPU measurements with raw
receipt hashes; the stored price is their median. Fused-group options need
whole-group timing. Encoder time, activation width and relative speed hints
cannot substitute for this input. No current runtime timing table ships as a
default.

Candidate reduction preserves all legal alternatives before runtime pricing,
including a faster, higher-loss route at the same serialized size. Fused
folding preserves every licensed coherent member recipe under an explicit
combination cap. The new solver enumerates the discrete nondominated
frontier in serialized bytes, quality cost and prefill time, optionally
including decode and device resources. It uses integer bytes, with no
rate-bin rounding or convex-hull assumption. State and transition limits
refuse excess work instead of silently truncating an answer.

Serialized weight bytes, terminal weight residency, peak activation memory,
peak scratch, and fixed KV/non-Linear resources remain separate. The solver
adds residency and tracks activation and scratch maxima independently.
Exactness describes the supplied additive resource model and finite menu.
Variable assignment-shared overhead needs explicit accounting; a final
feasibility filter cannot by itself establish global artifact optimality.

The allocator checks the expanded, promoted assignment and its fixed
auxiliary choices against the same measured resource contract. The sum of
operator medians is a proposal estimate, not an end-to-end p95 TTFT or decode
measurement. Existing serving and publication gates remain authoritative.

## Validation and promotion plan

The baseline is weight-only AURA over exactly the same production renders,
calibration tensors, sequence lengths and common probes. The candidate adds
the complete activation/weight residual. Numerical checks compare the
decomposition against an independently formed local residual, including
cancellation, mixed terms, repeated invocations, static QDQ and the identity
activation case. Cache/checkpoint tests verify resumption and changed-input
refusals. Solver checks compare exhaustive assignment enumeration on
nonconvex, same-byte and multiple-resource examples; CLI checks exercise
actual selection and final expanded-assignment feasibility.

The current-model numerical screen uses Qwen3-0.6B, a fixed unquantized
teacher, shared calibration and separate held-out tokens, and a bounded
measured subset of real Linears. It compares candidate and assignment KL/NLL
without re-centering the teacher. In-process profiles and host telemetry
record the workload and execution cost; outcomes belong in dated measurement
receipts, including negative or inconclusive results.

Production promotion additionally requires real served assignments on the
target runtime, matched-byte uniform controls, end-to-end prefill/decode and
residency measurements, and the existing held-out/downstream gates. The
fixed teacher remains the reference for bounded close swaps. Sparse-anchor
interpolation requires separate joint-currency pilot and held-out evidence;
the existing output-MSE replay and historical Gridbook coefficients do not
qualify this currency. Until those gates pass, this feature remains research.

## Explicit invocation

With an existing matching production cache, probe file and measured format
list, collect the joint table using the normal streamed producer:

```bash
python3 -m prismaquant.aura_cost --model "$MODEL" \
  --streaming --joint-activation --production-cache production.pkl \
  --formats "$MEASURED_FORMATS" --n-probes 16 \
  --checkpoint-dir joint-checkpoints --output joint.pkl

python3 -m prismaquant.allocator --probe probe.pkl --costs joint.pkl \
  --formats "$MEASURED_FORMATS" --target-bits "$TARGET_BITS" \
  --measured-runtime-table runtime.json --measured-runtime-context context.json \
  --slo-prefill-p95-ttft-ms "$PREFILL_BUDGET_MS" \
  --layer-config layer-config.json --pareto-csv pareto.csv
```

`runtime.json` must contain measurements of the exact supplied operator
recipes and workload, including fixed work. The numerical screen does not
produce those serving measurements. Menu eligibility and model-profile
requirements still apply to these direct CLI invocations. The pipeline
wrapper does not infer these experimental inputs or enable this mode.

The [current-model screen](../measurements/pq237-joint-aura-screen-2026-09-05.md)
verifies numerical decomposition but shows no selection-quality gain.
Qualification remains open in #237.
