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

This version is organized by what is actually realistic in vLLM today,
because deployment on NVIDIA Spark via vLLM is the target environment.

## Deployment Boundary

The strongest practical constraint comes from the current Spark vLLM
path. The `spark-vllm-docker` repo is centered around `AWQ`,
`INT4/AutoRound`, `FP8`, `NVFP4`, `MXFP4`, `MXFP8`, `BF16`, and runtime
knobs like `--kv-cache-dtype fp8` and `--load-format instanttensor`.
That is the best current reality check for what PrismaQuant should
prioritize first.

Separately, `paroquant` shows that vLLM can be extended by a repo-owned
plugin. It is not just a paper idea:

- `README.md` tells users to serve models with `vllm serve $MODEL`
- `pyproject.toml` registers a `vllm.general_plugins` entrypoint
- `paroquant/inference/backends/vllm/plugin.py` registers a custom
  `paroquant` quantization config for vLLM

That means there are really three categories for the roadmap:

1. works in the current Spark / vLLM path today
2. is reachable in vLLM with a known plugin-style integration path
3. needs new type support and should be treated as later work

## References

### JANGQ / TurboQuant

Reference links:

- JANGQ collection:
  <https://huggingface.co/collections/jangq/jang-turboquantized-models>
- Qwen3.6-35B-A3B-JANGTQ2:
  <https://huggingface.co/OsaurusAI/Qwen3.6-35B-A3B-JANGTQ2>
- MiniMax-M2.7-JANGTQ:
  <https://huggingface.co/OsaurusAI/MiniMax-M2.7-JANGTQ>
- MiniMax-M2.7-Small-JANGTQ:
  <https://huggingface.co/OsaurusAI/MiniMax-M2.7-Small-JANGTQ>

Key takeaways:

- JANGTQ pushes routed expert weights much lower than the rest of the
  model.
- Attention, embeddings, shared experts, and `lm_head` stay at much
  higher precision.
- Some variants keep routers and norms even higher, which strongly
  supports hard protection for coherence and control paths.
- The MiniMax "Small" variant combines pruning with quantization, which
  keeps expert pruning on the long-term radar.

### Spark vLLM Docker

Reference link:

- Spark vLLM Docker:
  <https://github.com/eugr/spark-vllm-docker>

Local source pointers:

- `../spark-vllm-docker/README.md`
- `../spark-vllm-docker/recipes/README.md`
- `../spark-vllm-docker/recipes/openai-gpt-oss-120b.yaml`
- `../spark-vllm-docker/examples/README.md`

Key takeaways:

- The practical deployment menu today is built around `AWQ`,
  `INT4/AutoRound`, `FP8`, `NVFP4`, `MXFP4`, `MXFP8`, and `BF16`.
- The repo has explicit Spark-oriented paths for `MXFP4`, `NVFP4`, and
  `FP8`.
- It repeatedly uses `--kv-cache-dtype fp8`, which means KV precision is
  part of the real deployment budget.
- It highlights loader/runtime improvements like `--load-format
  instanttensor`, which matter for shipping even when the model weights
  themselves do not change.

### ParoQuant

Local source pointers:

- `../paroquant/README.md`
- `../paroquant/pyproject.toml`
- `../paroquant/paroquant/inference/backends/vllm/plugin.py`
- `../paroquant/paroquant/cli/serve.py`

Key takeaways:

- `paroquant` claims direct vLLM support in its README.
- It registers a real vLLM plugin via `vllm.general_plugins`.
- Its vLLM backend is not "stock vLLM magically knows ParoQuant"; it is
  a concrete custom integration built around a repo-owned quantization
  config and kernels.
- That makes ParoQuant ideas materially easier to adopt than a
  completely new codec, but still more work than staying inside the
  current stock Spark / vLLM format families.

## Tier 1: Valid Today In Spark / vLLM

These are the ideas that fit the current serving stack without needing a
new vLLM type.

### 1. Protect attention and routing paths more aggressively

Source:

- `../jangq/README.md`
- JANGQ / TurboQuant model cards listed above

Idea:

- Encode architecture priors directly into PrismaScout / allocator
  decisions so attention, routers, and other coherence-critical paths
  have precision floors or protected budget classes.

Why it matters:

- The JANGQ cards suggest many MoE quality failures come from
  over-compressing attention and control paths, not from expert MLPs
  themselves.
- This is allocator work, not runtime work.

Likely PrismaQuant shape:

- Add optional per-role minimum formats such as:
  `attention >= MXFP8`, `router >= BF16/MXFP8`, experts free to drop
  lower within the supported family set.
- Add role-aware budget buckets so dangerous downgrades are excluded
  before search.

Expected payoff:

- Better quality, especially on large MoE and reasoning models.

### 2. Add expert-only vLLM-compatible low-bit candidate families

Source:

- JANGQ / TurboQuant model cards listed above
- `../spark-vllm-docker/README.md`
- `../spark-vllm-docker/recipes/openai-gpt-oss-120b.yaml`

Idea:

- Introduce a dedicated candidate family for routed expert weights that
  goes lower than the rest of the model, but keep it inside the formats
  that the current Spark / vLLM path already serves.

Why it matters:

- The JANGQ cards suggest the biggest wins come from separating bulk
  expert memory from coherence-critical paths.
- The Spark vLLM repo tells us what that can mean in practice today.

Likely PrismaQuant shape:

- Start with expert-only families centered on `NVFP4`, `MXFP4`,
  `MXFP8`, and `BF16` depending on role.
- Keep attention, router, shared expert, and head layers out of the
  lowest bucket.

Expected payoff:

- More compression on MoE models while staying inside a deployment path
  that already exists.

### 3. Add Spark / vLLM deployment-aware artifact profiles

Source:

- `../spark-vllm-docker/README.md`
- `../spark-vllm-docker/recipes/README.md`
- `../spark-vllm-docker/examples/README.md`

Idea:

- Treat deployment compatibility as a first-class profile dimension, not
  just a post-hoc export filter.

Why it matters:

- Real deployment quality depends on more than weight format alone:
  backend behavior, load format, KV precision, and Spark-oriented recipe
  stability all matter.

Likely PrismaQuant shape:

- Add a `spark_vllm` deployment profile that prefers or requires
  exportable families such as `NVFP4`, `MXFP4`, `MXFP8`, `FP8`, and
  `BF16`.
- Include runtime assumptions like KV-cache dtype and load-format in the
  emitted artifact metadata and validation report.

Expected payoff:

- Better alignment between offline quantization search and the real
  runtime path used on Spark systems.

### 4. Consider KV-cache quantization as a first-class runtime budget

Source:

- `../llm-compressor/README.md`
- `../spark-vllm-docker/README.md`
- `../spark-vllm-docker/recipes/*.yaml`

Idea:

- Extend PrismaQuant's artifact-selection and validation story to cover
  KV-cache precision.

Why it matters:

- The Spark recipes show that `--kv-cache-dtype fp8` is part of normal
  practice, not an edge case.
- Weight compression is not the only memory bottleneck at inference.

Likely PrismaQuant shape:

- Add KV-cache settings to artifact metadata and the validation harness.
- Treat runtime memory as "weights + KV-cache" instead of weights alone
  for deployment-oriented profiles.

Expected payoff:

- Smaller deployable memory footprint without forcing lower-quality
  weight choices.

### 5. Add selective local refiners after global allocation

Source:

- `../llm-compressor/README.md`

Idea:

- After PrismaScout chooses a global mixed-precision assignment, run a
  second-stage local refinement on only the most sensitive layers using
  methods in the AWQ / GPTQ / AutoRound / SmoothQuant family.

Why it matters:

- PrismaQuant already spends effort choosing where bits go.
- `llm-compressor` suggests a complementary strategy for using those
  bits better on a small subset of critical layers.

Likely PrismaQuant shape:

- Keep PrismaScout as the global allocator.
- Add a post-allocation pass on the top-K most sensitivity-dense layers
  or on layers whose measured KL remains far above the local prediction.

Expected payoff:

- Better quality at the same size with bounded extra runtime.

## Tier 2: Reachable In vLLM With A Known Integration Path

These ideas are not "stock Spark / vLLM today", but they are more
credible than a greenfield codec because another repo already shows the
basic integration pattern.

### 6. Add ParoQuant-style learned pairwise rotations

Source:

- `../paroquant/README.md`
- `../paroquant/pyproject.toml`
- `../paroquant/paroquant/inference/backends/vllm/plugin.py`

Idea:

- Borrow ParoQuant's learned pairwise rotations to suppress outliers
  before low-bit quantization.

Why it matters:

- This is one of the clearest paths to pushing difficult layers lower
  without giving away as much quality.
- Unlike JANGTQ's codebook path, there is already a repo-owned vLLM
  integration pattern for this family of ideas.

Likely PrismaQuant shape:

- Prototype a PrismaQuant export mode that emits a ParoQuant-like custom
  quantization config for selected layers.
- Prefer MoE expert projections and other high-error layers first.
- Treat this as a plugin-backed serving path rather than assuming stock
  vLLM support.

Expected payoff:

- Better quality at low bits, especially in INT4-like regimes, with a
  more credible path to serving than a completely novel codec.

### 7. Add rotation-aware candidates as optional frontier states

Source:

- `../paroquant/README.md`
- `../jangq/jang-tools/jang_tools/turboquant/linear.py`
- `../jangq/jang-tools/jang_tools/loader.py`

Idea:

- Let the allocator compare transformed and untransformed variants for a
  selected subset of layers.

Why it matters:

- The allocator should eventually be able to decide when the extra
  transformation cost is worth the quality it saves.
- ParoQuant provides the learned-rotation reference, and JANGQ provides
  the lighter-weight rotation intuition.

Likely PrismaQuant shape:

- Start with a very restricted candidate menu:
  `plain NVFP4`, `plain MXFP8`, `rotated INT4`, `rotated NVFP4`.
- Only enable these states in experimental or plugin-backed profiles.

Expected payoff:

- A better low-bit quality frontier without forcing every layer through a
  heavier transform.

## Tier 3: Later / Requires New vLLM Type Support

These ideas should stay in the document, but they should not drive the
near-term roadmap because they imply meaningful runtime, export, and
maintenance work.

### 8. Add TurboQuant-style 2-bit / 3-bit codebook candidates for experts

Source:

- JANGQ / TurboQuant model cards listed above
- `../jangq/jang-tools/jang_tools/turboquant/linear.py`
- `../jangq/jang-tools/jang_tools/build_jangtq_sidecar.py`

Idea:

- Add a non-affine, codebook-based expert candidate family to PrismaQuant
  experiments.

Why it matters:

- The JANGQ cards suggest that protected critical paths plus a richer
  low-bit expert codec can move the compression frontier.
- But this is not part of the current Spark / vLLM deployment path.

Likely PrismaQuant shape:

- Keep these candidates in research and offline evaluation first.
- Do not assume near-term serving support.

Expected payoff:

- Potentially very large compression gains on expert-heavy models, but
  only after deeper runtime work.

### 9. Add custom vLLM type support for new expert codecs

Source:

- JANGQ / TurboQuant model cards listed above
- `../jangq/jang-tools/jang_tools/turboquant/linear.py`
- `../spark-vllm-docker/README.md`

Idea:

- Add explicit vLLM support for brand-new expert codecs such as
  TurboQuant-style codebook formats.

Why it matters:

- This is not just an allocator feature. It implies export schema work,
  loader changes, kernel/backend work, tests, and ongoing maintenance
  against a fast-moving vLLM branch.
- The Spark repo is valuable because it shows how much engineering is
  already involved even for formats that are close to mainline support.

Likely PrismaQuant shape:

- Only pursue this after the compatible `NVFP4` / `MXFP4` / `MXFP8` /
  `FP8` / `BF16` roadmap is mature.

Expected payoff:

- Opens the door to deeper codec experiments, but at much higher
  engineering cost.

### 10. Add prune-then-quant MoE pipelines

Source:

- MiniMax-M2.7-Small-JANGTQ card

Idea:

- Combine expert pruning with quantization for very large MoE models.

Why it matters:

- JANGQ's MiniMax "Small" result suggests compression may eventually
  come from combining structural reduction with low-bit expert codecs.

Likely PrismaQuant shape:

- Revisit this only after the allocator and serving story are stable for
  non-pruning mixed-precision artifacts.

Expected payoff:

- Possibly large memory reduction on expert-heavy models, but not a
  near-term serving win.

## Suggested Order

If the goal is near-term PrismaQuant improvement with reasonable
engineering effort and a realistic Spark / vLLM deployment path, the
order I would try is:

1. Attention/router protection rules from JANGQ.
2. Expert-only vLLM-compatible low-bit families for MoE models.
3. Spark / vLLM deployment-aware profiles and validation metadata.
4. KV-cache quantization in deployment profiles.
5. Small local post-allocation refiners from LLM Compressor.
6. ParoQuant-style learned rotations via a plugin-backed serving path.
7. Rotation-aware candidate states in experimental profiles.
8. TurboQuant-style expert codecs in offline research mode.
9. Custom vLLM type support only after the compatible path is strong.
10. Prune-then-quant MoE pipelines after that.

## Bottom Line

The most important roadmap correction is:

- `NVFP4` / `MXFP4` / `MXFP8` / `FP8` / `BF16` work belongs early because
  it fits the current Spark / vLLM path.
- ParoQuant-style rotation belongs in the middle because there is a real
  vLLM plugin pattern in the repo, even though it is not stock support.
- JANGQ-style `2-bit` / `3-bit` TurboQuant and new vLLM type support
  belong later because they imply substantial runtime engineering.
