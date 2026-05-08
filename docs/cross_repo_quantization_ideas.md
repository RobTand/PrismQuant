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

The main near-term principle is:

- do not just add more formats everywhere
- add better formats where they are plausible
- remove dangerous formats where structure tells us they are likely to hurt

In other words, the highest-leverage short-term work is better
candidate-menu design, not merely a larger candidate set.

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
- `../spark-vllm-docker/recipes/qwen3.5-122b-int4-autoround.yaml`
- `../spark-vllm-docker/recipes/qwen3.5-397b-int4-autoround.yaml`

Idea:

- Introduce a dedicated candidate family for routed expert weights that
  goes lower than the rest of the model, but keep it inside the formats
  that the current Spark / vLLM path already serves.

Why it matters:

- The JANGQ cards suggest the biggest wins come from separating bulk
  expert memory from coherence-critical paths.
- The Spark vLLM repo tells us what that can mean in practice today.

Likely PrismaQuant shape:

- Start with expert-only families centered on `INT4`, `NVFP4`, `MXFP4`,
  `MXFP8`, and `BF16` depending on role.
- Treat `INT4` as a first-class family, not just a fallback, with room
  for `AWQ` and `AutoRound`-style variants where the serving stack
  already supports them.
- Keep attention, router, shared expert, and head layers out of the
  lowest bucket.

Expected payoff:

- More compression on MoE models while staying inside a deployment path
  that already exists.

### 2b. Make candidate menus role-aware instead of uniform

Source:

- JANGQ / TurboQuant model cards listed above
- `../spark-vllm-docker/README.md`

Idea:

- Give different layer roles different format menus instead of exposing
  the same candidate list everywhere.

Why it matters:

- This is the most practical form of the JANGQ lesson.
- The allocator does not just need more options; it needs better priors
  about which options are even worth considering for a given role.

Likely PrismaQuant shape:

- Attention / router / head / other coherence-critical paths:
  restrict to higher-trust menus such as `BF16`, `MXFP8`, and source
  passthrough where applicable.
- Experts and other bulk memory paths:
  allow lower-bit menus such as `INT4`, `NVFP4`, `MXFP4`, and `MXFP8`.
- Shared experts and other semi-critical paths:
  use an intermediate menu.

Expected payoff:

- Better allocations with very small engineering cost, because the
  search space gets both safer and more informative.

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
- TurboQuant-style KV-cache compression adds an important nuance here:
  the cache can be compressed asymmetrically, keeping `K` at a safer
  precision while compressing `V` more aggressively. That is attractive
  because `K` tends to be more quality-sensitive than `V`, so
  asymmetric K/V compression can save memory without paying the full
  quality cost of symmetric low-bit cache quantization.

Likely PrismaQuant shape:

- Add KV-cache settings to artifact metadata and the validation harness.
- Treat runtime memory as "weights + KV-cache" instead of weights alone
  for deployment-oriented profiles.
- Add room in deployment profiles for asymmetric KV-cache policies, not
  just one uniform cache dtype.

Expected payoff:

- Smaller deployable memory footprint without forcing lower-quality
  weight choices.
- A realistic path to memory savings that can arrive earlier than
  TurboQuant weight-codec integration, because the value comes from
  cache policy rather than from a new weight format.

### 4b. Improve calibration signal quality before adding exotic formats

Source:

- `../prismaquant/multi_chunk_probe.py`
- JANGQ / TurboQuant model cards listed above

Idea:

- Improve the data the allocator sees before adding many new candidate
  families.

Why it matters:

- A richer menu only helps if the measured signal is strong enough to
  separate the good choices from the bad ones.
- For MoE and reasoning models, better calibration coverage can matter
  as much as one more format family.

Likely PrismaQuant shape:

- Use multi-chunk calibration more deliberately for important model
  families.
- Add role-aware or domain-aware calibration presets for MoE and
  reasoning-heavy models.
- Keep held-out validation splits clearly separated from candidate
  generation.

Expected payoff:

- Better candidate ranking and more stable assignments without needing
  new runtime support.

### 5. Add selective local refiners after global allocation

Source:

- `../llm-compressor/README.md`
- `../spark-vllm-docker/recipes/qwen3.5-122b-int4-autoround.yaml`
- `../spark-vllm-docker/recipes/qwen3.5-397b-int4-autoround.yaml`

Idea:

- After PrismaScout chooses a global mixed-precision assignment, run a
  second-stage local refinement on only the most sensitive layers using
  methods in the AWQ / GPTQ / AutoRound / SmoothQuant family.

Why it matters:

- PrismaQuant already spends effort choosing where bits go.
- `llm-compressor` suggests a complementary strategy for using those
  bits better on a small subset of critical layers.
- The Spark repo reinforces that `INT4` refinement is practical today,
  not hypothetical, because `AutoRound` recipes are already part of the
  deployment toolbox.

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
- ParoQuant is specifically an `INT4` story, which makes it a very
  strong candidate if PrismaQuant wants better `INT4` quality rather
  than only better FP4-style quality.

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

## Method Placement Cheat Sheet

This section answers a practical question directly: for each major
quantization improvement discussed across the sibling repos, when would
it enter the PrismaQuant roadmap, how would PrismaQuant use it, and what
would we hope to gain from it?

| Method | Support status today | Planned stage | How PrismaQuant would use it | Potential benefit |
|---|---|---|---|---|
| `INT4` candidate family | Already supported | Early, Tier 1 | PrismaQuant already exposes `INT4_W4A16_g128` as a selectable format family; extend that existing usage by making `INT4` more central in role-aware menus for experts and other bulk layers | Stronger compression in a deployment path vLLM already serves |
| `NVFP4` / `MXFP4` candidate families | Already supported | Early, Tier 1 | PrismaQuant already uses these as first-class low-bit candidates in mixed-precision allocation; extend that by targeting them more deliberately at expert-heavy or memory-heavy paths | Better compression with a Spark/vLLM-compatible runtime path |
| `MXFP8` / `BF16` / source passthrough | Already supported | Early, Tier 1 | PrismaQuant already uses these as higher-trust formats and passthrough choices; extend that by making them the explicit protected menu for attention, router, head, and other coherence-critical paths | Quality protection and safer mixed-precision assignments |
| `AWQ` | Partially supported | Early-mid, Tier 1 | PrismaQuant already references AWQ in the broader per-tensor/export toolkit; extend that by turning AWQ into a clean local-refiner step after PrismaSCOUT chooses the global assignment, especially on sensitive `INT4` layers | Better `INT4` quality without changing the serving story |
| `AutoRound` | Partially supported | Early-mid, Tier 1 | PrismaQuant already positions itself as able to compose with AutoRound-style rounding; extend that by using AutoRound as a clear local-refiner or candidate-construction backend for selected low-bit layers | Better `INT4` / low-bit reconstruction on layers already chosen for compression |
| `GPTQ` | Already supported | Early-mid, Tier 1 | PrismaQuant already uses GPTQ-style reconstruction in its toolkit and polish/export path; extend that by focusing GPTQ refinement more explicitly on the highest-value layers under the chosen assignment | Better per-layer reconstruction where the surrogate says extra work is worth it |
| `SpinQuant` | Not yet supported | Mid, Tier 2 | Add as a rotation-aware transform option for selected fragile layers or experimental candidate states | Better low-bit quality, especially where outliers hurt plain `INT4` / `FP4` |
| `QuIP` | Not yet supported | Mid, Tier 2 | Treat as another transform-style experimental path similar to SpinQuant | Additional rotation-based quality improvements if the added transform cost is justified |
| `ParoQuant` | Not yet supported | Mid, Tier 2 | Use as the strongest plugin-backed `INT4` quality path, likely first on experts or other high-error layers | High-quality `INT4` with a more credible vLLM path than a brand-new codec |
| `TurboQuant 4-bit` / `JANGTQ4` | Not yet supported | Earlier, Tier 3 | Start as offline research for expert-only codebook candidates at a less aggressive compression point than `2/3-bit`; defer serving integration until there is a real export/runtime path | Potentially strong MoE compression with less quality risk than the lower-bit TurboQuant variants, but still requires new codec/runtime work |
| `TurboQuant 2/3-bit` / lower-bit `JANGTQ` | Not yet supported | Later, Tier 3 | Use only as deeper offline research after the 4-bit TurboQuant case is understood; defer serving integration until there is a mature export/runtime path | Potentially the largest MoE compression gain, but with both higher quality risk and deeper runtime/export work |
| custom vLLM codec support | Not yet supported | Latest, Tier 3 | Add explicit runtime support only after the current compatible path is mature | Unlocks new codecs, but at the highest engineering cost |

Interpretation of the support labels:

- `Already supported` means the relevant format or mechanism is already
  materially present in PrismaQuant / PrismaSCOUT today.
- `Partially supported` means the repo clearly has some of the needed
  machinery, but not yet as a clean first-class PrismaSCOUT workflow for
  the use described here.
- `Not yet supported` means it is a real roadmap item rather than
  something the current codebase already does.

Why these improvements are targeted instead of global:

- Most of the quality loss usually comes from a relatively small set of
  fragile Linears, not from every layer equally.
- Many improvement methods are not free: they cost calibration time,
  quantization time, export complexity, runtime complexity, or all four.
- Applying them everywhere can spend a lot of work on layers that are
  already good enough, while also making the search space noisier.
- PrismaQuant's mixed-precision philosophy applies here too: spend the
  expensive improvement where it materially changes the result.

## Suggested Order

If the goal is near-term PrismaQuant improvement with reasonable
engineering effort and a realistic Spark / vLLM deployment path, the
order I would try is:

1. Use already-supported `MXFP8` / `BF16` / source passthrough more explicitly as the protected menu for attention, router, head, and other coherence-critical paths.
2. Use already-supported `INT4`, `NVFP4`, and `MXFP4` more explicitly as low-bit menus for experts and other bulk-memory paths.
3. Formalize those two ideas into role-aware candidate menus and Spark / vLLM deployment-aware profiles.
4. Improve calibration signal quality for MoE and reasoning models before expanding into many heavier methods.
5. Treat KV-cache precision as part of the deployment budget in those same profiles.
6. Use already-supported `GPTQ` more explicitly as a targeted refiner on the highest-value layers under the chosen assignment.
7. Extend the partially supported `AWQ` and `AutoRound` paths into explicit targeted local refiners for selected low-bit layers.
8. Explore `SpinQuant` and `QuIP` as rotation-aware experimental transforms for selected fragile layers rather than as global defaults.
9. Explore `ParoQuant` as the strongest plugin-backed `INT4` quality path, likely first on experts or other high-error layers.
10. Explore `TurboQuant 4-bit` offline as an expert-codec research path after the compatible paths above are solid.
11. Explore `TurboQuant 2/3-bit` only after the 4-bit TurboQuant case is understood.
12. Add custom vLLM codec support only after the current compatible path is strong enough to justify the engineering cost.
13. Revisit prune-then-quant MoE pipelines after the non-pruning mixed-precision and codec work is stable.

Read another way, the expected order of arrival for the named methods is:

1. Already-supported candidate families: `INT4`, `NVFP4`, `MXFP4`, `MXFP8`, `BF16`
2. Already-supported and partially supported targeted refiners: `GPTQ`, then `AWQ` / `AutoRound`
3. Not-yet-supported rotation-aware transforms: `SpinQuant`, `QuIP`, `ParoQuant`
4. New-codec expert paths: `TurboQuant 4-bit`, then `TurboQuant 2/3-bit`
5. New vLLM runtime type support

## Bottom Line

The most important roadmap correction is:

- `INT4` work belongs early alongside `NVFP4` / `MXFP4` / `MXFP8` /
  `FP8` / `BF16`, because it also fits the current Spark / vLLM path and
  is already represented there via `AWQ` and `AutoRound`.
- The best immediate win is better candidate-menu design: add useful
  options where they are plausible, and remove risky ones where they are
  structurally unlikely to work.
- Better calibration signal quality is the other underappreciated cheap
  lever; better menus and better signal should come before exotic new
  codecs.
- ParoQuant-style rotation belongs in the middle because there is a real
  vLLM plugin pattern in the repo, even though it is not stock support,
  and because it offers a stronger path for high-quality `INT4`.
- `TurboQuant 4-bit` still belongs later because it is a new
  codec/runtime path, even though it is less aggressive than the lower
  bit variants.
- JANGQ-style `2-bit` / `3-bit` TurboQuant and new vLLM type support
  belong even later because they imply both higher quality risk and
  substantial runtime engineering.
