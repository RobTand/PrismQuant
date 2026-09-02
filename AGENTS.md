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
   Four containers, four gates: `compressed-tensors` on vanilla vLLM (no
   PrismaQuant kernels); GGUF on llama.cpp and the vLLM GGUF plugin, gated
   additionally on bit-exactness against `gguf-py`; codebook (NVFP4-CB /
   FP8-CB) on the separately released
   [`gridbook`](https://github.com/RobTand/gridbook) plugin, pinned by
   `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`. It ships its own CUDA kernels and
   is gated on an unforked vLLM (no core patches), a producer profile declared
   in Gridbook's packaged `runtime_contract.json`, fail-closed expert loading,
   and served speed at least at parity
   with the container it displaces; and the Tessera wire on **Tessera's own**
   vLLM plugin (package `tessera.serving`, entry point
   `tessera = "tessera.serving:register"` under `vllm.general_plugins`,
   registering `quant_method = "tessera"`), pinned by
   `prismaquant/tessera_runtime/tessera_serving_runtime_pin.json`. That lane
   has no enable flag — the checkpoint's `quant_method` selects the plugin, and
   the single operator knob is `TESSERA_SERVE_MODE=resident|streamed` — and it
   is gated on the same unforked vLLM (the plugin is installed into the stock
   image, no core patches), on `device_qualified` native cells in Tessera's
   packaged `runtime_contract.json`, and on every such cell declaring
   `requires_plugin: "tessera"`, because stock vLLM has no reader for these
   bytes and the route is plugin-gated rather than merely flag-gated. Its
   admission is **fail-closed until a Tessera release tag exists**: the pin
   carries PENDING sentinels, `require_exact_tessera_runtime_release` refuses
   them, and `tessera_render.tessera_lane_attested` therefore answers False for
   every rung — by the pin, not by an edit — even though the contract publishes
   the cells and the `tessera` package is importable producer-side. It is
   dense-only at TP=1: no served measurement covers routed experts, so the
   contract carries no `routed_moe` cell. The non-vLLM-native lanes are
   sanctioned, not exceptions; what is forbidden is a forked runtime.
   PrismaQuant must never vendor or import the Gridbook or Tessera *serving*
   runtimes; compatibility crosses each repository boundary only through the
   immutable pin and the packaged contract. (Gridbook briefly carried a Tessera
   lane of its own; that contract version was never released and the lane is
   withdrawn, so the Gridbook pin does not govern Tessera admission.)
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
