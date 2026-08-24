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
   additionally on bit-exactness against `gguf-py`; codebook (NVFP4-CB /
   FP8-CB) on the separately released
   [`gridbook`](https://github.com/RobTand/gridbook) plugin, pinned by
   `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`. It ships its own CUDA kernels and
   is gated on an unforked vLLM (no core patches), a producer profile declared
   in Gridbook's packaged `runtime_contract.json`, fail-closed expert loading,
   and served speed at least at parity
   with the container it displaces. The non-vLLM-native lanes are sanctioned,
   not exceptions; what is forbidden is a forked runtime.
   PrismaQuant must never vendor or import the Gridbook runtime; compatibility
   crosses the repository boundary only through the immutable pin and contract.
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
11. **Keep `docs/ARCHITECTURE.md` current in the same commit.** A change to
    pipeline defaults, the stage graph, the format menu, the plugin contract,
    serving-lane defaults, or ship gates is incomplete until
    `docs/ARCHITECTURE.md` (and its diagrams, if topology changed) reflects it
    and its provenance block is re-stamped. `tests/test_docs_staleness.py` and
    `tests/test_architecture_doc.py` enforce the mechanical subset. Dated
    results docs and handovers are append-only history, never a substitute.

## Implementation Checklist

Before editing:

- Identify the existing mechanism this change should extend.
- Decide how the change stays GPU-bound and resident-prefetched.
- Define the vLLM compatibility gate if formats, export metadata, kernels,
  or compressed-tensors layout are touched.
- Define the KL/bpp/runtime comparison and calibration set.

Before finishing:

- Run targeted tests and compile checks for touched modules.
- Add or update tests for new policies, format gates, cache residency, or
  validation behavior.
- Record measured results, commands, and log paths in docs when a claim is
  based on a run.
- Leave experimental methods opt-in until the validation gate in
  `docs/design/design_guidelines.md` is satisfied.
