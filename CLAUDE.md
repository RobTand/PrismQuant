# Claude Handover

Before implementing new functionality, read `AGENTS.md` and
`docs/design_guidelines.md`. Those files are the normative design rules for
GPU-first execution, cache reuse, vLLM gating, and measurement discipline.

Before working on PrismaQuant, read in order:

- `.claude/prismaquant-handover-2026-05-28.md` — **CURRENT STATE.**
  Shipped two CPU-only wins (allocator `--bit-attribution-json` budget
  report; revived the dead surrogate-vs-KL Spearman in
  `select_validated_frontier`), **walled off grouped-KL** under
  `archive/grouped_kl_2026-05-28/` (it LOST the shipped vLLM A/B on 27B —
  the −3.52% figure was an HF/local screen the serving contract reversed),
  and reconciled the stale docs. Open: pluggable MoE expert projection
  names (DSv4), and the queued 27B JSO isolation A/B (scale 4 and 6).
  Open issues + file map inside.
- `.claude/prismaquant-handover-2026-05-20.md` — earlier state. Note its
  grouped-KL "−3.52% PPL win" claim is **superseded**: that was a local /
  HF-PPL screen; grouped-KL lost the production vLLM A/B and is now
  archived. JSO wall-off was reverted pending the isolation A/B.
- `.claude/prismaquant-handover-2026-05-02.md` — earlier state.
  PrismaSCOUT L3-redesign landed end-to-end. v5 proved 34% KL improvement
  over L2 at 4.5 bpp on Qwen 4B.
- `.claude/prismaquant-handover-2026-05-01.md` — earlier state. Allocator
  Δloss bug + the design that became PrismaSCOUT.
- `.claude/prismaquant-handover-2026-04-29.md` — older session context.
