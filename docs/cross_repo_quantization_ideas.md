# Cross-Repo Quantization Ideas for PrismaQuant

Date: 2026-05-08

This note captures ideas worth lifting into `prismaquant` from sibling
repositories at the same directory level:

- `../llm-compressor`
- `../paroquant`
- `../jangq`

The focus here is narrow: ideas that could either:

- compress a model further at a similar quality level, or
- improve quality at a similar compressed size.

This is intentionally a working roadmap, not a literature review.

## Quick Wins

### 1. Protect attention and routing paths more aggressively

Source:

- `../jangq/README.md`

Idea:

- Encode architecture priors directly into PrismaScout / allocator
  decisions so attention, routers, and other coherence-critical paths
  have precision floors or protected budget classes.

Why it matters:

- JANG's reported results strongly suggest that low-bit failures on MoE
  models come from over-compressing attention and shared control paths,
  not from expert MLPs themselves.
- PrismaQuant already measures sensitivity per linear, but a hard or
  semi-hard architecture prior could prevent catastrophic allocations
  that a pure knapsack view still permits.

Likely PrismaQuant shape:

- Add optional per-role minimum formats, for example:
  `attention >= MXFP8`, `router >= BF16/MXFP8`, experts free to drop lower.
- Add role-aware budget buckets so "cheap but dangerous" downgrades are
  excluded before search.

Expected payoff:

- Better quality, especially on large MoE and reasoning models.

### 2. Add selective local refiners after global allocation

Source:

- `../llm-compressor/README.md`

Idea:

- After PrismaScout chooses a global mixed-precision assignment, run a
  second-stage local refinement on only the most sensitive layers using
  methods in the AWQ / GPTQ / AutoRound / SmoothQuant family.

Why it matters:

- PrismaQuant already spends effort choosing *where* bits go.
- `llm-compressor` suggests a complementary strategy for *how* to use
  those bits better on a small subset of critical layers.

Likely PrismaQuant shape:

- Keep PrismaScout as the global allocator.
- Add a post-allocation pass on the top-K most sensitivity-dense layers
  or on layers whose measured KL remains far above the local prediction.

Expected payoff:

- Better quality at the same size with bounded extra runtime.

### 3. Consider KV-cache quantization as a first-class runtime budget

Source:

- `../llm-compressor/README.md`

Idea:

- Extend PrismaQuant's artifact-selection and validation story to cover
  KV-cache precision, ideally including per-head scaling variants.

Why it matters:

- Weight compression is not the only memory bottleneck at inference.
- KV-cache savings can make a slightly less aggressive weight allocation
  viable while still fitting the target hardware budget.

Likely PrismaQuant shape:

- Add KV-cache settings to the artifact metadata and validation harness.
- Treat runtime memory as "weights + KV-cache" instead of weights alone
  for deployment-oriented profiles.

Expected payoff:

- Smaller deployable memory footprint without needing more aggressive
  weight loss everywhere.

## Medium Effort

### 4. Introduce pairwise rotation as a selective preconditioner

Source:

- `../paroquant/README.md`

Idea:

- Borrow ParoQuant's learned pairwise rotations to suppress outliers
  before low-bit quantization, but apply it selectively rather than
  globally.

Why it matters:

- This is the clearest path to pushing difficult layers lower without
  giving away as much quality.
- PrismaScout already identifies where the model is fragile; that makes
  it a good controller for where rotation is worth paying for.

Likely PrismaQuant shape:

- Add a new candidate family:
  `plain NVFP4`, `plain MXFP8`, `rotated NVFP4`, `rotated INT4`, etc.
- Limit rotation candidates to layers with strong outlier signatures or
  consistently high measured KL under standard formats.

Expected payoff:

- Better quality at low bits, especially near INT4/NVFP4 regimes.

### 5. Add activation quantization as part of mixed-precision search

Source:

- `../llm-compressor/README.md`

Idea:

- Expand the search space from weight format only to joint
  weight-and-activation format choices for a controlled subset of layers.

Why it matters:

- `llm-compressor` shows activation quantization is not just a serving
  detail; it is a real compression axis.
- PrismaQuant currently leaves some runtime memory savings on the table
  by not budgeting activation precision directly.

Likely PrismaQuant shape:

- Start with a constrained menu such as:
  `W4A16`, `W4AFP8`, `W8A8`, `MXFP8A16`.
- Restrict activation search to attention projections, MLP in/out, or
  only the layers selected by a sensitivity gate.

Expected payoff:

- Either smaller runtime memory, or higher-quality weights for the same
  total runtime memory budget.

### 6. Formalize non-uniform "profiles" by architecture family

Source:

- `../llm-compressor/README.md`
- `../jangq/README.md`

Idea:

- Add explicit architecture-aware allocation profiles instead of one
  generic policy for all transformers.

Why it matters:

- Both repos implicitly encode knowledge like:
  dense vs MoE vs multimodal vs hybrid-attention models need different
  compression behavior.
- PrismaQuant already has some model-profile logic; this could be pushed
  further into the allocator and validation contracts.

Likely PrismaQuant shape:

- Profiles such as:
  `dense_reasoning`, `packed_moe_reasoning`, `vlm`, `hybrid_attention`.
- Each profile controls candidate menus, format floors, exclusions, and
  validation thresholds.

Expected payoff:

- Better quality and fewer catastrophic misallocations on newer model
  families.

## Research-Heavy

### 7. Make rotation-aware candidates part of the core frontier

Source:

- `../paroquant/README.md`
- `../llm-compressor/README.md`

Idea:

- Treat preconditioned variants as full first-class states in the global
  search, rather than a simple post-pass.

Why it matters:

- The real opportunity is not just "rotate then quantize", but letting
  the allocator decide when the extra transformation cost is worth the
  quality saved.

Likely PrismaQuant shape:

- Multi-state per-layer search with transformed and untransformed
  variants.
- Separate storage/runtime cost accounting for the rotated artifacts.

Expected payoff:

- Best chance of reaching lower-bit frontiers without losing quality,
  but clearly a larger engineering project.

### 8. Add runtime-cache and hybrid-attention-aware compression

Source:

- `../jangq/README.md`

Idea:

- Extend PrismaQuant beyond "linear weight precision" into compression
  policies for hybrid attention runtimes, pooled state, and long-context
  cache components.

Why it matters:

- JANG's DSV4 notes make it clear that modern inference memory is not
  just model weights plus plain KV cache.
- For DeepSeek-class and future hybrid-attention models, the deployable
  budget may depend as much on cache/state policy as on weights.

Likely PrismaQuant shape:

- Deployment profiles that jointly consider:
  weights, KV cache, pooled attention state, and attention-type-specific
  runtime structures.

Expected payoff:

- Better compression decisions for new long-context models, but this is
  probably a separate phase after weight-side improvements land.

## Suggested Order

If the goal is near-term PrismaQuant improvement with reasonable
engineering effort, the order I would try is:

1. Attention/router protection rules from JANG.
2. Small local post-allocation refiners from LLM Compressor.
3. KV-cache quantization in deployment profiles.
4. Selective ParoQuant-style pairwise rotations on the worst layers.
5. Joint weight-and-activation candidate menus for a restricted subset.

## Notes for Future Investigation

- The JANG ideas are especially compelling for MoE and reasoning models.
- The ParoQuant ideas are especially compelling when PrismaQuant wants to
  push layers into true low-bit territory and current RTN-style formats
  are not holding quality.
- The LLM Compressor ideas are the broadest source of practical, staged
  PTQ components that can be bolted onto PrismaQuant without replacing
  its core search logic.
