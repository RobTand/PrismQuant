# PrismaQuant Agent Rules

These rules are mandatory for coding agents working in this repository.
Before implementing new functionality, read this file,
`docs/design/design_guidelines.md`, and `docs/ARCHITECTURE.md`.

## Core Principles

1. **GPU-bound by default.** Production probes, cache fills, recache,
   polish, export, and validation must be designed so the GPU is the
   bottleneck. A hot path that is CPU-bound, disk-bound, or NVMe-bound is a
   bug unless the user explicitly requested an offline data-prep step.
2. **Self-contained, deterministic operation.** Agents may design, diagnose,
   review, and improve PrismaQuant, but they must not be required to operate
   it. Ordinary quantization, distributed scheduling, barriers, retries,
   recovery, allocation, export, and validation must run from explicit
   versioned configuration and machine-readable state through deterministic
   code paths. If execution needs intelligence to decide what to do next, the
   application is missing a contract or state-machine transition. Ambiguous
   state must fail closed; an operator or agent may repair the implementation
   or supply a new explicit input, but may not serve as the runtime scheduler.
3. **Use the existing cache and prefetch system.** Do not create a parallel
   cache, preload, or residency mechanism for rendered weights or
   activations. Extend `ProductionWeightCache`, `PerturbedActivationCache`,
   the streaming model prefetch path, or the existing pipeline wiring.
   Production paths should fail fast when required resident data cannot be
   prefetched instead of silently streaming from NVMe.
4. **Right quantization for the right layer.** Preserve PrismaQuant's core
   contract: per-Linear empirical selection with measured quality/cost
   tradeoffs. Avoid model-wide defaults unless they are only a fallback or
   have been validated against the per-Linear path.
5. **Only ship formats the target engine serves performantly.** A format can
   appear in research menus before it is production-ready, but it must not
   become a production default until the engine for its container loads it,
   generates correctly, and uses a performant kernel on representative shapes.
   Three containers, three gates: `compressed-tensors` on vanilla vLLM (no
   PrismaQuant kernels); GGUF on llama.cpp and the vLLM GGUF plugin, gated
   additionally on bit-exactness against `gguf-py`; and the Tessera wire on
   **Tessera's own** vLLM plugin (package `tessera.serving`, entry point
   `tessera = "tessera.serving:register"` under `vllm.general_plugins`,
   registering `quant_method = "tessera"`), pinned by
   `prismaquant/tessera_runtime/tessera_serving_runtime_pin.json`. That lane
   has no enable flag — the checkpoint's `quant_method` selects the plugin, and
   the single operator knob is `TESSERA_SERVE_MODE=resident|streamed` — and it
   is gated on an unforked vLLM (the plugin is installed into the stock image,
   no core patches), on `device_qualified` native cells in Tessera's packaged
   `runtime_contract.json`, and on every such cell declaring
   `requires_plugin: "tessera"`, because stock vLLM has no reader for these
   bytes and the route is plugin-gated rather than merely flag-gated. Its
   admission is pinned to an **exact Tessera commit plus the SHA-256 of the
   `runtime_contract.json` that commit packages** (pin schema
   `prismaquant.tessera_serving_runtime_pin.v2`, 2026-09-04: Rob retired the
   release-tag requirement). `require_pinned_tessera_runtime` refuses unless
   the pin equals the reader's three pinned constants AND the installed
   contract hashes to the pinned digest, so any other Tessera on `PYTHONPATH`
   still answers False for every rung — by the pin, not by an edit.
   `version_is_release` is recorded and advisory; no gate reads it. It is
   dense-only at TP=1, and since contract v17 that is an EVIDENCE fact rather
   than an absence: the contract does publish two `routed_moe` cells, both
   carrying `evidence.smoke.status: "repetitive"`, and
   `lane_eligibility.cell_evidence_admits` refuses them on that measured
   generation failure. Promoting routed-MoE Tessera is Rob's decision under
   principle 9; never widen the gate to admit one. The non-vLLM-native lanes are
   sanctioned, not exceptions; what is forbidden is a forked runtime.
   PrismaQuant must never vendor or import the Tessera *serving* runtime;
   compatibility crosses that repository boundary only through the immutable
   pin and the packaged contract.

   **A fourth container was retired on 2026-09-02.** The codebook lane
   (NVFP4-CB / FP8-CB) served by the separately released
   [`gridbook`](https://github.com/RobTand/gridbook) plugin was removed from
   PrismaQuant by Robert's decision — *"put Tessera in PrismaQuant and remove
   Gridbook"* — and archived whole at `archive/gridbook_lane_2026-09-02/`. Its
   pin, exporter, serving profiles, ship gates and lane documents are gone from
   the live tree; the Tessera wire is its successor. Do not re-add a Gridbook
   pin, a `gridbook_runtime/` directory, or an `EXPORT_CONTAINER=nvfp4_cb`
   path: `run-pipeline.sh` refuses that container with `exit 2`.
6. **Measure on the same calibration contract.** New levers need apples-to-
   apples KL, bpp, and runtime measurements. Compare against the relevant
   shipped or current baseline using the same calibration set, sequence
   length, layer assignment semantics, and production cache behavior.
7. **Report bpp over quantizable parameters only.** Bits-per-parameter
   accounting must exclude immutable BF16 regions that the allocator is not
   allowed to quantize, including `lm_head` and any profile-pinned model
   components. Published uniform-NVFP4 and GGUF k-quant comparisons do not
   average in unquantizable parameters; PrismaQuant reports should follow the
   same convention. (MXFP8 is de-menued — exact-scale FP8 dominates it — so it
   is not the comparison to reach for.)
8. **Reuse local abstractions.** Prefer the existing format registry,
   allocator, production cache, recache, validation harness, and pipeline
   flags. If an abstraction is missing, add it at the shared layer rather
   than building a one-off call site.
9. **Keep cross-layer machinery archived unless explicitly requested.**
   The archived CLADO, propagated-cost, output-Fisher, PrismaSCOUT iteration,
   QUBO, and polish-of-many code is research context, not a production
   shipping lever.
10. **Use the known-good Docker environments.** PrismaQuant has Docker images
   with the required CUDA, PyTorch, Transformers, vLLM, and pipeline
   dependencies already installed. For GPU runs, validation, export, and
   large-model experiments, use those working containers first instead of
   assuming the host Python environment is sufficient or rebuilding ad hoc.
11. **Acquire required dependencies and artifacts.** Agents are authorized to
    download and install task-required files, model checkpoints, CLI tools,
    libraries, packages, and container images without asking solely for
    download/install permission. Prefer scoped, user-local, or containerized
    installs where practical, pin or record versions needed for reproducibility,
    and continue to use the known-good Docker environments for GPU work.
12. **Keep `docs/ARCHITECTURE.md` current in the same commit.** A change to
    pipeline defaults, the stage graph, the format menu, the plugin contract,
    serving-lane defaults, or ship gates is incomplete until
    `docs/ARCHITECTURE.md` (and its diagrams, if topology changed) reflects it
    and its provenance block is re-stamped. `tests/test_docs_staleness.py` and
    `tests/test_architecture_doc.py` enforce the mechanical subset. Dated
    results docs and handovers are append-only history, never a substitute.

13. **Measurement is first-class, and telemetry counts as measurement.** Any
    claim about speed, cost, residency, or "where the time went" is carried by
    a measurement, never by log-line reasoning. Profile BEFORE the change and
    AFTER it -- the delta is the claim, and a bench number without a profile
    does not establish where the time went. Two instruments, both required,
    because they answer different questions: an **in-process profiler**
    (`torch.profiler`, py-spy, `/proc/PID/io`, `nsys`) says where time goes
    *inside* a run; the **Netdata series on both boxes** says whether the box
    was actually loaded, which no in-process tool can see. Attach the evidence
    to the finding, and put it in the acceptance criteria of every perf
    delegation.

    On GB10, **`nvidia_smi.gpu_utilization` is non-diagnostic under load**: it
    means "at least one kernel is resident", not "the SMs are working", and it
    reads 96% for a memory-stalled kernel exactly as for a saturated one
    (measured 2026-08-28: 96% on both sides of a 5.83x throughput change).
    `utilization.memory` is worse -- a fake hard 0. Read **power against the
    ~140 W envelope** instead, and rank implementations by **work per joule**;
    the envelope fraction also estimates remaining headroom, which wall-clock
    cannot. Do not diagnose a GPU hot path from utilization.

14. **Fix the finding where you found it -- file only what you cannot.** A
    defect you noticed and neither fixed nor filed dies with your context. But a
    ticket is not the only record, and it is usually the worse one: whoever
    tripped over the defect understands it better than any fresh agent will, and
    that understanding evaporates the moment the task returns. A filed one-liner
    then costs a brief, a worktree, a fresh-context ramp-up, a report and a
    merge, to re-derive what somebody already knew. (Rob, 2026-09-03, watching a
    dozen tickets arrive in an hour: *"we're just proliferating issues right now
    that may make sense to fix in the context of the way in which they were
    discovered."*)

    **Default: fix it, on your branch, in a separate commit.** One commit does
    one thing so a reviewer can take it or drop it alone -- that constraint is
    satisfied by a second *commit*, and was never a reason to leave a defect
    unfixed. Only the mixed commit is forbidden.

    **File instead when the fix is not yours to make:**
    - it needs a decision only Rob prices -- a default moves, an artifact's bytes
      move, a format menu or serving lane changes, a ship gate is involved;
    - it needs a measurement you are not set up to take -- a served A/B, a
      second model, the other box;
    - it lives in another agent's live branch (read theirs, never edit theirs);
    - it is large enough to swamp the diff of the task you came for.

    Two bounds, so this is not a licence to widen scope: it covers what you
    **trip over** doing the task you came for, never a hunt for adjacent work;
    and every off-task fix is named in the report, one line each, so nothing
    lands unannounced.

    **When you do file, timeliness still binds.** Same working session, before
    starting the next task -- not at the end of the day, not in a handover, not
    in a summary. A finding held in context for "later" has already failed. The
    bar is **"I believe this is wrong"**, not "I have proved it is wrong and
    scoped the fix": over-filing beats losing a finding, and fixing beats both.

    **Say which it was.** An issue filed under this principle records whether the
    filer could have fixed it and chose not to, and why. Without that, a backlog
    stops describing the work and starts hiding it.

    What a filed issue otherwise owes is little: the **evidence at `file:line`**
    (read the line, do not repeat a claim), what breaks and under what inputs, a
    **severity** from the rubric below, and what would fix it -- or, when the fix
    is a judgment call, the options and who decides. Say plainly what you did
    *not* measure, and file an uncertain finding with its uncertainty stated.

    **Severity rubric.** `P0` -- can ship or serve a wrong artifact. `P1` -- a
    gate that cannot catch its own defect, or a wrong or underived number that
    a decision reads. `P2` -- provenance, observability, or a claim beyond its
    evidence. `P3` -- cleanup with no decision riding on it. Two orthogonal
    labels: `measurement-needed` when a GPU or served A/B decides it, and
    `needs-decision` when the answer is a trade only Rob prices.

    **One exception, narrower than it was.** A finding in prose -- a doc, a
    comment, a docstring -- is *fixed on sight* and never filed: reading the
    cited line IS the verification, so a stale sentence is a one-line commit.
    (The former second exception, that a delegated worker neither fixes nor
    files, is withdrawn: it described a constraint workers do not have.)

## Implementation Checklist

Before editing:

- Identify the existing mechanism this change should extend.
- Decide how the change stays GPU-bound and resident-prefetched.
- Decide what the before/after measurement is, and which instrument
  produces it (in-process profiler, Netdata series, or both).
- Define the vLLM compatibility gate if formats, export metadata, kernels,
  or compressed-tensors layout are touched.
- Define the KL/bpp/runtime comparison and calibration set.

Before finishing:

- Run targeted tests and compile checks for touched modules.
- Attach before/after profiler evidence for any hot-path or perf change;
  rank GPU work by work-per-joule, never by utilization.
- Add or update tests for new policies, format gates, cache residency, or
  validation behavior.
- Record measured results, commands, and log paths in docs when a claim is
  based on a run.
- Leave experimental methods opt-in until the validation gate in
  `docs/design/design_guidelines.md` is satisfied.
