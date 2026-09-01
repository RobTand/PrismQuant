# Progressive Render Pipeline

PrismaQuant local numerical methods should be accepted by the same local
render score used for allocator-style output error:

1. Render the current baseline weight for a Linear or fused-sibling group.
2. Render the candidate after one mechanism, or a candidate package when a
   mechanism can only be useful as an initializer for downstream refinement.
3. Score both on the same activation rows.
4. If h-detail/Fisher rows are available, use Fisher-weighted output MSE.
   Otherwise use activation output MSE. Weight MSE is only a fallback when
   activations are unavailable.
5. Accept the candidate only when the score improves by the configured
   minimum gain. If it regresses or ties, keep the baseline and continue to
   the next mechanism.

The shared implementation lives in `prismaquant/render_score.py`.
Production-cache renders store decisions under
`cache.metadata["render_gates"]`; FourOverSix also has a compact
`cache.metadata["four_over_six"]` summary because it is a first-class plugin.

## Ordering

Mechanisms declare what kind of operation they perform. The production cache
resolves the order from those declarations rather than from the text order in
an environment variable.

Current V1 production order:

```text
format scale rule:         FourOverSix (NVFP4 only)
format scale optimizer:    joint_scale_opt (NVFP4 only)
rounding solver modifier:  static_act_order (NVFP4, MXFP4, MXFP8)
rounding solver:           GPTQ
codebook scale refine:     scale_sweep (explicit ablation only)
```

Historical research-only render paths live under `archive/`. V1 production
defaults use `gptq,static_act_order,joint_scale_opt`; static activation
ordering applies to production microscaling GPTQ formats (NVFP4, MXFP4, and
MXFP8), while joint scale optimization is NVFP4-only. `scale_sweep` remains
available for explicit ablations but is not part of the default recipe.

Global basis transforms are excluded from this local sequence; evaluate them
as separate full-recipe arms.

FourOverSix is a first-class NVFP4 scale-rule plugin. When enabled, the
production cache tests `static_6` against `four_over_six_mse` directly, then
also lets FourOverSix participate in downstream GPTQ packages. This catches
the case where FourOverSix alone is neutral or negative, but FourOverSix plus
rounding improves the active score.

MXFP8_E4M3 scale-sweep and the explicit FP8_E4M3/FP8_E5M2 plus
MXFP8_E4M3/MXFP8_E5M2 GPTQ paths use the same progressive gate: if the
activation-aware candidate regresses the active score, the baseline render is
kept.

The pipeline order is shared across formats. Support is format-gated:

- NVFP4 supports FourOverSix, GPTQ, joint_scale_opt, static_act_order, and
  scale_sweep.
- FP8_E4M3/FP8_E5M2 support GPTQ; FP8_E4M3 additionally supports dynamic
  per-row scale_sweep.
- MXFP4 supports GPTQ and static_act_order with the canonical E8M0 scale
  rule. (MXFP6 was removed from the registry on 2026-09-01; it had no served
  export/dequant path
  is wired yet.
- MXFP8_E4M3/MXFP8_E5M2 support GPTQ and static_act_order with the canonical
  E8M0 scale rule. When static_act_order is enabled, the production gate
  scores both ordinary GPTQ and static-order GPTQ and keeps the lower-score
  candidate. MXFP8_E4M3 additionally supports E8M0 scale_sweep.
- Unsupported mechanisms are skipped for that format rather than creating a
  separate pipeline.

The gate can be disabled only for debugging with
`PRISMAQUANT_RENDER_PROGRESSIVE_GATES=0`. The minimum relative gain is
`PRISMAQUANT_RENDER_GATE_MIN_GAIN` and defaults to `0.0`.

## Extension Contract

New local mechanisms should register a `RenderMechanismSpec` with:

- a stable `name`;
- an `operation` class;
- a `scope` such as `linear`, `fused_sibling_group`, or `nvfp4_block`;
- a numeric `phase`;
- the `gate_metric`;
- optional `after` / `before` dependencies;
- optional `exclusive_group`.

Then the mechanism's candidate renderer should call `score_render_error()` and
`gate_render_candidate()` for accept/reject. This keeps new numeric methods
easy to include, toss, or reorder without adding one-off scoring logic.
