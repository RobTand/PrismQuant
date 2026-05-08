# Cross-Repo Quantization Ideas for PrismaQuant

Date: 2026-05-08

This note captures ideas worth lifting into `prismaquant` from sibling
repositories at the same directory level:

- `../llm-compressor`
- `../paroquant`
- `../jangq`
- `../spark-vllm-docker`

The focus here is narrow: ideas that could either:

- compress a model further at a similar quality level, or
- improve quality at a similar compressed size.

This is intentionally a working roadmap, not a literature review.

## JANGQ / TurboQuant References

The JANGQ model cards add an important refinement to the earlier
cross-repo comparison: their low-bit story is not "quantize everything
equally." The cards describe a selective recipe where routed expert
weights are compressed very aggressively while attention and other
coherence-critical paths remain at much higher precision.

Reference links:

- JANGQ collection:
  <https://huggingface.co/collections/jangq/jang-turboquantized-models>
- Qwen3.6-35B-A3B-JANGTQ2:
  <https://huggingface.co/OsaurusAI/Qwen3.6-35B-A3B-JANGTQ2>
- MiniMax-M2.7-JANGTQ:
  <https://huggingface.co/OsaurusAI/MiniMax-M2.7-JANGTQ>
- MiniMax-M2.7-Small-JANGTQ:
  <https://huggingface.co/OsaurusAI/MiniMax-M2.7-Small-JANGTQ>

Key takeaways from those cards:

- JANGTQ uses ultra-low-bit TurboQuant on routed expert weights rather
  than uniformly quantizing the whole model.
- Attention, embeddings, shared experts, and `lm_head` are kept in
  higher-precision affine form.
- Some variants keep routers and norms even higher, which is a strong
  clue that coherence and control paths need hard protection.
- The MiniMax "Small" variant combines pruning with JANGTQ, which moves
  expert pruning higher in the long-term plan for large MoE models.

## Spark vLLM Deployment References

The Spark-focused vLLM deployment repo is useful because it narrows the
"what can we ship now?" question. Its recipes and patches are centered
around the quantization families that are practical today on Spark with
vLLM and FlashInfer, rather than around brand-new custom formats.

Reference links:

- Spark vLLM Docker:
  <https://github.com/eugr/spark-vllm-docker>

Local source pointers:

- `../spark-vllm-docker/README.md`
- `../spark-vllm-docker/recipes/README.md`
- `../spark-vllm-docker/recipes/openai-gpt-oss-120b.yaml`
- `../spark-vllm-docker/examples/README.md`

Key takeaways from that repo:

- The practical deployment menu today is built around `AWQ`,
  `INT4/AutoRound`, `FP8`, `NVFP4`, `MXFP4`, `MXFP8`, and `BF16`, plus
  runtime knobs like `--kv-cache-dtype fp8`.
- The repo includes explicit Spark-oriented paths for `MXFP4`, `NVFP4`,
  and `FP8`, which is a strong signal that PrismaQuant should prioritize
  those families first for near-term shipping work.
- The repo also highlights loader/runtime improvements like
  `--load-format instanttensor`, which matter for deployment readiness
  even when the quantized weights themselves are unchanged.
- Nothing in the current Spark vLLM path suggests stock support for
  JANGQ-style `2-bit` / `3-bit` TurboQuant expert codecs. That kind of
  support should be treated as a later custom-runtime effort rather than
  a quick follow-on to allocator changes.

## Quick Wins

### 1. Protect attention and routing paths more aggressively

Source:

- `../jangq/README.md`
- JANGQ / TurboQuant model cards listed above

Idea:

- Encode architecture priors directly into PrismaScout / allocator
  decisions so attention, routers, and other coherence-critical paths
  have precision floors or protected budget classes.

Why it matters:

- JANG's reported results strongly suggest that low-bit failures on MoE
  models come from over-compressing attention and shared control paths,
  not from expert MLPs themselves.
- The JANGQ cards make this more concrete: their recipe protects those
  paths explicitly while pushing routed experts much lower.
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

### 1b. Add an expert-only ultra-low-bit candidate family

Source:

- JANGQ / TurboQuant model cards listed above

Idea:

- Introduce a dedicated candidate family for routed expert weights that
  goes lower than the rest of the model, while the allocator keeps
  attention, routers, shared experts, and heads out of that bucket.

Why it matters:

- The JANGQ cards suggest the biggest compression wins come from
  separating "bulk expert memory" from "coherence-critical paths" rather
  than finding one globally fair low-bit format.

Likely PrismaQuant shape:

- Start with a vLLM-compatible expert-only low-bit family for MoE
  models, centered on formats like `NVFP4` and `MXFP4` where the Spark
  deployment path already exists.
- Keep the rest of the candidate menu unchanged at first so the new
  behavior can be isolated and measured cleanly.
- Treat expert-only `2-bit` / `3-bit` candidates as research-only until
  there is a credible export and runtime path.

Expected payoff:

- More compression on large MoE models with much lower quality risk than
  a model-wide low-bit push, while staying inside a serving stack that
  can actually be deployed today.

### 2. Add selective local refiners after global allocation

Source:

- `../llm-compressor/README.md`
- `../spark-vllm-docker/README.md`
- `../spark-vllm-docker/recipes/*.yaml`

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
- The Spark vLLM recipes reinforce that this is not theoretical:
  `--kv-cache-dtype fp8` shows up repeatedly in the practical serving
  configurations.

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
- `../jangq/jang-tools/jang_tools/turboquant/linear.py`
- `../jangq/jang-tools/jang_tools/loader.py`

Idea:

- Borrow ParoQuant's learned pairwise rotations to suppress outliers
  before low-bit quantization, but apply it selectively rather than
  globally.

Why it matters:

- This is the clearest path to pushing difficult layers lower without
  giving away as much quality.
- PrismaScout already identifies where the model is fragile; that makes
  it a good controller for where rotation is worth paying for.
- JANGQ provides the lighter-weight precursor: fixed/random Hadamard
  rotation plus codebook quantization. ParoQuant provides the stronger
  learned-rotation follow-on.

Likely PrismaQuant shape:

- Start with cheap Hadamard-rotated low-bit candidates for experts.
- Later add ParoQuant-style learned pairwise rotations for the worst
  layers once the basic rotated-candidate plumbing exists.
- Add a new candidate family:
  `plain NVFP4`, `plain MXFP8`, `rotated NVFP4`, `rotated INT4`, etc.
- Limit rotation candidates to layers with strong outlier signatures or
  consistently high measured KL under standard formats.

Expected payoff:

- Better quality at low bits, especially near INT4/NVFP4 regimes.

### 4b. Consider TurboQuant-style codebook candidates for experts

Source:

- JANGQ / TurboQuant model cards listed above
- `../jangq/jang-tools/jang_tools/turboquant/linear.py`
- `../jangq/jang-tools/jang_tools/build_jangtq_sidecar.py`

Idea:

- Add a non-affine, codebook-based expert candidate family to PrismaQuant
  experiments rather than assuming all very-low-bit candidates must be
  affine or RTN-like.

Why it matters:

- The JANGQ cards point to TurboQuant quality coming from three things in
  combination: protected critical paths, Hadamard rotation, and
  Lloyd-Max-style codebooks.
- If PrismaQuant wants to compete in the 2-bit / 3-bit expert regime, it
  likely needs a richer codec than plain affine quantization.

Likely PrismaQuant shape:

- Prototype codebook candidates for routed experts only.
- Keep export/runtime support out of scope initially; use them first for
  search and evaluation to see whether the frontier moves enough to
  justify deeper integration.

Expected payoff:

- Potentially the largest compression gain in the roadmap, but only on
  architectures where expert memory dominates.

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

### 5b. Add deployment-aware artifact profiles for Spark / vLLM

Source:

- `../spark-vllm-docker/README.md`
- `../spark-vllm-docker/recipes/README.md`
- `../spark-vllm-docker/examples/README.md`

Idea:

- Treat deployment compatibility as a first-class profile dimension, not
  just a post-hoc export filter.

Why it matters:

- The Spark repo shows that real deployment quality depends on more than
  weight format alone: `FlashInfer` backend behavior, `instanttensor`
  loading, `fp8` KV-cache choices, and Spark-specific recipe stability
  all influence what is practical.
- This is a good fit for PrismaQuant's existing profile concept because
  it lets the search stay focused on formats and settings that map to a
  proven serving stack.

Likely PrismaQuant shape:

- Add a `spark_vllm` deployment profile that prefers or requires
  exportable families such as `NVFP4`, `MXFP4`, `MXFP8`, `FP8`, and
  `BF16`.
- Include runtime assumptions like KV-cache dtype and load-format in the
  emitted artifact metadata and validation report.

Expected payoff:

- Better alignment between offline quantization search and the real
  container/runtime path used on Spark systems.

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
- JANGQ / TurboQuant model cards listed above

Idea:

- Treat preconditioned variants as full first-class states in the global
  search, rather than a simple post-pass.

Why it matters:

- The real opportunity is not just "rotate then quantize", but letting
  the allocator decide when the extra transformation cost is worth the
  quality saved.
- JANGQ strengthens this argument because it shows a practical deployed
  system where low-bit expert quality depends on a richer codec than
  naive affine quantization.
- The Spark vLLM deployment repo also sharpens the tradeoff: if a new
  candidate family cannot be served in the current vLLM path, it should
  not displace compatible `NVFP4` / `MXFP4` / `MXFP8` work in the near
  term.

Likely PrismaQuant shape:

- Multi-state per-layer search with transformed and untransformed
  variants.
- Separate storage/runtime cost accounting for the rotated artifacts.

Expected payoff:

- Best chance of reaching lower-bit frontiers without losing quality,
  but clearly a larger engineering project.

### 7b. Add custom vLLM type support only after the compatible path is strong

Source:

- JANGQ / TurboQuant model cards listed above
- `../jangq/jang-tools/jang_tools/turboquant/linear.py`
- `../spark-vllm-docker/README.md`

Idea:

- Defer custom vLLM support for brand-new expert codecs such as
  TurboQuant-style `2-bit` / `3-bit` codebook formats until after
  PrismaQuant has fully exploited the formats the current Spark vLLM
  stack already handles well.

Why it matters:

- New type support is not just an allocator feature. It implies export
  schema work, loader changes, kernel/backend work, testing, and ongoing
  maintenance against a fast-moving vLLM branch.
- The Spark repo is valuable because it shows how much engineering is
  already involved even for formats that are close to mainline support.

Likely PrismaQuant shape:

- Keep custom-codec exploration alive in evaluation mode.
- Postpone production export/runtime integration until the compatible
  `NVFP4` / `MXFP4` / `MXFP8` / `FP8` / `BF16` roadmap is mature.

Expected payoff:

- Better sequencing. We keep the long-term frontier in view without
  letting a large runtime project crowd out nearer deployment wins.

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
2. Expert-only vLLM-compatible low-bit candidate families for MoE models.
3. Spark / vLLM deployment-aware profiles and validation metadata.
4. Small local post-allocation refiners from LLM Compressor.
5. KV-cache quantization in deployment profiles.
6. JANGQ-style Hadamard-rotated codebook candidates for experts.
7. Selective ParoQuant-style pairwise rotations on the worst layers.
8. Joint weight-and-activation candidate menus for a restricted subset.
9. Custom vLLM type support for new expert codecs only after the above.

## Notes for Future Investigation

- The JANG ideas are especially compelling for MoE and reasoning models.
- The ParoQuant ideas are especially compelling when PrismaQuant wants to
  push layers into true low-bit territory and current RTN-style formats
  are not holding quality.
- The LLM Compressor ideas are the broadest source of practical, staged
  PTQ components that can be bolted onto PrismaQuant without replacing
  its core search logic.
- The Spark vLLM repo is the best current reality check for deployment:
  it should pull the roadmap toward compatible `NVFP4` / `MXFP4` /
  `MXFP8` / `FP8` work now, and push custom-codec runtime support later.
