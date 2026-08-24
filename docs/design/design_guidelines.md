# PrismaQuant Design Guidelines

These guidelines define how new PrismaQuant functionality should be designed,
implemented, and promoted into production recipes.

## Product Promise

PrismaQuant exists to choose the right quantization for the right layer. New
work should strengthen per-Linear empirical choice, production-faithful
measurement, and vLLM-serving correctness. A method that is interesting but
does not improve this loop belongs behind an experimental flag or in research
docs, not in the production path.

## Non-Negotiables

### GPU-First Execution

All production-relevant processes should be GPU-constrained:

- sensitivity probes
- production weight-cache fill
- production activation recache
- KL measurement
- polish
- export validation
- vLLM serving smokes

CPU, disk, or NVMe pressure on a hot path is treated as a failure mode. The
default response is not to tolerate the slow path; it is to use, repair, or
extend prefetching so the GPU has resident data ready when needed.

Required behavior:

- use existing pre-cache/prefetch systems before adding new scheduling code;
- report resident bytes, number of entries prefetched, and cache misses;
- make production runs fail fast when required data cannot fit resident;
- keep offline data-build steps clearly separated from production hot paths.

### Self-Contained, Non-Agentic Operation

PrismaQuant may be developed, diagnosed, and reviewed with agents, but an
agent is never part of its runtime architecture. Straight quantization and
scheduling must be a deterministic application function: a closed manifest
declares inputs, hosts, partitions, stage order, barriers, retry policy, and
outputs; machine-readable receipts determine the next state.

Required behavior:

- encode local and multi-host campaigns as explicit versioned state machines;
- assign work from stable data such as sample indices, layers, and format
  rungs, never from an agent's judgment during execution;
- make transfers and barriers content-addressed and independently verifiable;
- make retries idempotent, resumable, and deterministic from committed state;
- publish resource and GPU-utilization telemetry as part of the campaign
  receipt; and
- fail closed on missing, conflicting, or ambiguous state instead of asking an
  agent to infer how execution should continue.

An operator or agent may supply a newly reviewed manifest or fix a defective
implementation. Once launched, ordinary probe, quantization, allocation,
export, and validation must complete without intelligence in the loop.

### One Cache Mechanism

Rendered weights should flow through `ProductionWeightCache`. Activation
captures and perturbation replay should flow through `PerturbedActivationCache`
or the established streaming activation cache paths. New features must not
introduce parallel cache stores, independent preload systems, or duplicate
key-resolution logic.

If a call site needs different behavior, add a shared method to the existing
cache abstraction and reuse it everywhere.

### Per-Linear Decisions

Quantization choices should remain per-Linear unless there is a measured reason
to collapse them. New allocation, cost, or probe logic should preserve:

- per-Linear candidate formats;
- production-faithful rendered weights;
- explicit BF16 fallback;
- measured quality deltas;
- bpp accounting tied to the exported artifact.

Model-wide knobs are acceptable as defaults, caps, or ablation controls, but
they are not a substitute for per-Linear selection.

### vLLM and Kernel Reality

A format or transform is production-eligible only when it satisfies all of:

- represented correctly in compressed-tensors export metadata;
- accepted by vLLM on representative model shapes;
- routed to a performant kernel rather than a slow fallback;
- passes eager load/generation smoke;
- passes graph-mode load/generation smoke when graph mode is part of the
  serving target;
- does not break MTP/speculative decode when the artifact includes MTP, or is
  explicitly gated away from MTP.

Registry support alone is not enough. A format may remain in the research menu
without becoming part of the production default menu.

## Measurement Discipline

Every production candidate needs an apples-to-apples measurement plan before
it is implemented.

For a new numerical method, record:

- baseline artifact or recipe;
- candidate recipe;
- calibration set and sequence length;
- KL metric and validation command;
- bpp target and measured bpp;
- production-cache settings;
- recache/prefetch settings;
- vLLM smoke command and log path;
- downstream serving-suite commands and log paths, including PPL/mean NLL,
  log-likelihood task checks, and ToolEvalBench for materialized artifacts;
- observed GPU utilization and whether NVMe/CPU was idle during the hot path.

The default comparison is method A versus method A plus the new lever, with
all dependent sweeps re-run for both arms. Do not compare a well-tuned baseline
against an untuned candidate, or the reverse.

KL is a screening metric, not a standalone promotion metric. A candidate that
improves calibration KL but regresses held-out PPL/mean NLL or downstream
log-likelihood checks should remain research-only unless there is a documented
reason to prefer the KL tradeoff and the user explicitly accepts it.

## Promotion Gates

Use these states for new functionality:

- **Research:** code may exist, but it is opt-in, documented as experimental,
  and excluded from production defaults.
- **Candidate:** the feature has small-model GPU and vLLM smokes, plus a clear
  27B measurement plan.
- **Production recipe:** the feature improves or preserves KL/bpp/runtime on
  the target stack, passes vLLM serving checks and the downstream serving
  suite, and has tests covering the policy or metadata it changes.
- **Default-on:** the feature has cleared the production recipe gate on the
  current target and at least one additional representative model or shape
  class, unless the user explicitly accepts a narrower default.

Regression or inconclusive results should demote the feature back to research.

### Progressive Local Gates

Local render mechanisms must use the shared scorer in
`prismaquant/render_score.py` unless there is a documented exception. The
default gate is Fisher-weighted output MSE when h-detail exists, otherwise
activation output MSE on the same calibration rows. A candidate that regresses
the active score is skipped and the previous rendered baseline is kept.
For mechanisms that function as initializers, such as FourOverSix before
GPTQ/scale-sweep, the production cache may gate a candidate package so a
neutral standalone initializer can still be accepted when the composed package
improves the active score.

Mechanism ordering is declared by operation type, not by ad hoc flag order.
See `docs/design/progressive_render_pipeline.md` for the current order and extension
contract. Global basis transforms are full-recipe arms, not local progressive
steps.

Research-only transform families live under `archive/`. They must not be
reintroduced into production cache fill, allocator cost overrides, export
metadata, or pipeline flags unless the user explicitly asks to revive the
corresponding archive.

### Rotation Transforms

Rotation methods are production-eligible only when the exported graph remains
vanilla vLLM. A rotation that changes the residual-stream basis between
layers, such as full layer-wise ReSpinQuant, needs a residual transition
operator unless it collapses to a single global basis. That runtime adapter is
not allowed in production artifacts without an explicit vLLM/kernel support
decision. The 2026-05-13 ReSpinQuant attempt is archived under
`archive/respinquant_2026-05-13/`; do not reintroduce a layer-wise residual
basis path into cache fill or export unless that runtime-support decision is
made first.

## Design Review Checklist

Before implementation, answer these in the PR, commit notes, or working doc:

- What existing subsystem does this extend?
- What makes the hot path GPU-bound?
- What data must be resident, and which existing prefetch path ensures that?
- What happens if the data cannot be resident?
- Which formats, shapes, and vLLM kernels are affected?
- What is the baseline measurement?
- What is the pass/fail threshold?
- Which tests or smokes prove the implementation did not create a parallel
  mechanism or silent fallback?

## Exception Rule

Exceptions are allowed only when documented up front. An exception note must
state:

- which rule is being bypassed;
- why the existing mechanism cannot be used;
- how the exception is isolated behind a flag or research path;
- what result would justify promoting, deleting, or replacing it.
