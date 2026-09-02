# The Gridbook codebook serving lane — archived 2026-09-02

**Kill order.** Robert, 2026-09-02: *"put Tessera in PrismaQuant and remove
Gridbook."* Not a measurement verdict and not a defect — a decision to carry
**one** non-vLLM-native wire instead of two. The lane's replacement is the
**Tessera** wire on Tessera's own vLLM plugin (`tessera.serving`, pinned by
`prismaquant/tessera_runtime/tessera_serving_runtime_pin.json`), which lands
separately. The sanctioned containers are now three: `compressed-tensors` on
vanilla vLLM, GGUF, and Tessera.

**Last commit where the lane was live:** `d263f54` (branch
`tessera/decouple-gridbook`). Everything in this directory is that commit's
content, moved with `git mv` so `git log --follow` still reaches it.

## What the lane was

A fourth export container, `EXPORT_CONTAINER=nvfp4_cb`, writing product-codebook
bytes (NVFP4-CB and FP8-CB) that no stock runtime can read, served by the
separately released [`gridbook`](https://github.com/RobTand/gridbook) vLLM
plugin. PrismaQuant never vendored it: the boundary was an immutable pin
(`gridbook_runtime/gridbook_runtime_pin.json`) plus the runtime's own packaged
`runtime_contract.json`, from which route status was **attested** rather than
asserted (principle 14). The producer side of that boundary is what lived here
and what has been removed.

Its parts, all now under this directory:

- **The pin and the contract boundary** — `prismaquant/gridbook_runtime/`
  (pin JSON, the materialized contracts for 0.8.10 / 0.8.11 / 0.9.1, the
  contract index, the shell helpers), `gridbook_runtime_pin.py`,
  `gridbook_serving_runtime_pin.py`, `gridbook_execution_contract.py`,
  `gridbook_format_contract.py`, `gridbook_environment.py`,
  `gridbook_assignment.py`, `cb_route_status_gate.py`.
- **The exporter** — `export_nvfp4_cb.py`, `export_nvfp4_cb_streaming.py`,
  `cb_export_config.py`, `build_cb_learned_bundle.py`.
- **The serving declarations** — `lane_specs/nvfp4_cb.json` and the four
  `serving_profile_specs/` that named the lane (`nvfp4_cb`, and the three
  RTX 4090 / sm120 FP8-CB validation profiles).
- **The validators and campaigns** — `validate_cb_endpoint.py`,
  `validate_cb_performance.py`, `native_baseline_feasibility.py`, the
  `dspark_*` matched-performance stack, the `rtx4090_*` census/compile-proof/
  graph-contract/policy/burn stack, the DSv4 CB reprice and W8A16 handoff, plus
  the `tools/` and `scripts/` drivers that ran them (40 scripts, 8 tools,
  including all of `scripts/pb_validation/`).
- **The tests** — 74 files, selected **by import dependency**, not by name:
  every test that stops collecting once the modules above are gone.
- **The lane documentation** — `docs/lanes/nvfp4-cb/` entire (27 files),
  including every served measurement the lane produced.

## What it was worth (kept, because it was measured)

These numbers are **not** retracted; the lane worked. They are archived because
the lane is, and they stay readable here:

- `docs/lanes/nvfp4-cb/prod_27b_results.md` — first production served A/B:
  Qwen3.6-27B CB @5.5 bpp vs the shipped PrismaAURA 5.5-bit: confident-position
  KL −45/−53%, ALL-KL −56/−58%, at 19.93 vs 23 GB. All 386 body Linears chose
  CB rungs.
- `docs/lanes/nvfp4-cb/prod_35b_results.md` — first CB-on-MoE verdict:
  Ornith-1.0-35B @4.75 bpp, confident-KL 0.01706 vs AURA 0.03625 (−53%).
- `docs/lanes/nvfp4-cb/prod_hy3_results.md` — Hy3 295B-A21B @2.9 bpp served on
  one Spark, 105.73 GB resident (carries its own no-quality-claims rule).
- `docs/lanes/nvfp4-cb/exp1c_v2_premium.md` and `rd_ceiling_study.md` — the
  rate-distortion work that separated the FP4-grid coding tax from the
  scale-packaging tax and motivated two-tier v2 scale coding.

Public artifacts already shipped from this lane (the AQUA 20 GB 27B, the DSv4
92 GB build, the RTX 4090 FP8-CB validation artifact) are **not** withdrawn.
They are, however, no longer re-verifiable or re-publishable from this
repository: `shipcard.open_cb_export_shipcard`, `CB_REQUIRED_SLOTS` and the
Gridbook distribution-identity / native-record / performance-record verifiers
went with the lane. Re-verifying one means reading it out of this archive.

## What went with it, that did not want to go

Recorded here because a removal that quietly drops a capability is a lie of
omission:

1. **`FP8_BLOCK_UE8M0_SOURCE` lost its only serve route.** Those bytes executed
   on the plugin's `Fp8SourceW8A16LinearMethod` and on nothing else. Its
   contract row is now `ROUTE_STATUS_BLOCKED` in
   `prismaquant/allocator_candidates.py`. Per principle 1 the rung is **still
   priced** — an allocator that wants it is reporting a serving gap, and that
   signal is the point — and per principle 9 the exporter fails closed on it
   without an explicit override. Its measured evidence is kept verbatim and
   scoped to the runtime it was measured on.
2. **`MXFP4_SOURCE` is orphaned.** Its route is genuinely stock (vLLM Marlin
   MoE, `--moe-backend marlin`) and stays `ROUTE_STATUS_BACKED` — that verdict
   was never Gridbook's to give. But the `nvfp4_cb` container was its only
   **writer** and the `nvfp4_cb` profile its only **offer**, so the producer can
   no longer reach it end to end. It fails closed at export
   (`_quantize_2d` raises). This is an exporter gap, not a route gap, and the
   two are deliberately not conflated: see
   `tests/test_source_passthrough_family.py` §3.
3. **The CB ship gate is gone rather than generalised.** `shipcard`'s
   `_verify_ship_gate_record` ran only behind the lane's `is_gridbook_cb` flag.
   Removing the lane leaves two readings of that branch and only one of them is
   a *removal*: running it on every lane instead would be a **new** refusal that
   no current native card passes. So the gate went with the lane it was written
   for. The comment at its former call site says so.
4. **`docs/design/constrained_pareto_allocation.md` is orphaned as policy.**
   The mechanism is live; the normative served-parity policy it defers to
   (`format-speed-policy.md` §1) is in here.

## What did NOT go, and why

- **`prismaquant/lane_eligibility.py`** (was `gridbook_lane_eligibility.py`).
  Despite the old name this is the **generic** closed-world lane-eligibility
  engine, and it is what admits the **Tessera** lane today. It was renamed, its
  error type generalised (`LaneEligibilityError`), the Gridbook schema string
  and asset-directory lookup removed, and `load_eligibility_table()` now
  requires an explicit `contract_path=` — with none, every unit resolves
  UNATTESTED and export fails closed. That is the fail-closed default the
  removal should have.
- **`prismaquant/prismasnap*`** — a source-checkpoint scale-fold prep for the
  *compressed-tensors* lane that merely refused the CB and GGUF lanes. Not
  Gridbook.
- **`tools/serve_fingerprint.py`** and the served-measurement tools — generic,
  used by the compressed-tensors and Tessera lanes.
- **`prismaquant/trellis_*.py`** — Tessera's own predecessor lineage, not this
  lane's.
- **The CB format / cost / render plumbing** (`cb_layout.py`,
  `nvfp4_cb_formats.py`, `nvfp4_cb_footprint.py`, `cb_ldlq*.py`,
  `cb_minchain.py`, `cb_warm_state.py`, `cb_banked_books.py`,
  `cb_learned_promotion.py`, `cb_anchored_cost.py`, `cb_ladder_cross_family.py`,
  `routed_moe_codebooks.py`, `mxfp4_widen.py`, `source_class_format_plan.py`,
  and the CB branches inside `production_weight_cache.py`, `allocator.py`,
  `format_registry.py`, `export_native_compressed.py`, `layer_config.py`,
  `lane_spec.py`, `serve_constraints.py`, `model_profiles/*`). This is the
  **remainder**, and it is deliberately not in this archive. Removing the lane
  made those rungs unexportable and unservable, which is the property that
  matters; excising the code is several hundred diffuse edits concentrated in
  exactly the files another branch is rewriting, and merging that against a
  live branch would be more dangerous than the debt. See
  `docs/measurements/gridbook-lane-retired-2026-09-02.md` for the full
  file-level remainder list.

## Do not revive

Not because the lane failed — it did not — but because the decision was to
carry one non-native wire. `run-pipeline.sh` refuses `EXPORT_CONTAINER=nvfp4_cb`
with `exit 2` pointing here, in the same shape as the archived cost modes.
Reviving it means re-litigating the decision with Robert first, not restoring a
file.
