# PrismaQuant Package Notes

This package README intentionally stays short. The user-facing overview lives in
the repository root `README.md`; the system map (stages, contracts, containers)
lives in `docs/ARCHITECTURE.md`.

`run-pipeline.sh` — the actual stage orchestrator — lives **in this package
directory**, not the repo root: `prismaquant/run-pipeline.sh`. `pipeline.py` is a
declarative contract layer, not the executor.

## CLI entrypoints (`python -m prismaquant.<module>`)

| Stage | Modules |
|---|---|
| Probe | `incremental_probe` |
| Cost | `incremental_measure_quant_cost`, `production_render_cost`, `aura_cost` (`COST_MODE=aura`), `expert_empirical_cost` (MoE hybrid), `aura_additivity_gate` (trust-region check) |
| Allocate | `allocator` |
| Walk gate | `model_walk` (R5 discovery walker: intake walk + fail-closed export gate, `python3 -m prismaquant.model_walk --model <dir>`) |
| Cache | `build_production_cache`, `production_recache` |
| Select | `validate_assignments_kl`, `select_validated_frontier` |
| Export | `export_native_compressed` (compressed-tensors), `export_gguf` / `export_gguf_direct` (GGUF), `export_nvfp4_cb` / `export_nvfp4_cb_streaming` (codebook, served by the separately released [Gridbook](https://github.com/RobTand/gridbook) package) |
| Validate | `validation_harness`, `validate_native_export`, `validate_quantized_model` |

## Library modules with no CLI (imported by the stages above)

- `footprint` — exact per-tensor byte accounting; the byte-budget ("fit the
  card") selection target.
- `saturation_select` — saturation-point bit-rate selection.
- `gguf_formats`, `gguf_iq_formats`, `gguf_gptq` — GGUF k-quant / IQ quantizers
  and the GPTQ-under-frozen-scales lever.
- `nvfp4_cb_formats`, `nvfp4_cb_footprint` — product-codebook codecs plus the
  versioned serialized-payload contract (production FP4 layout-v2, FP8 row
  scales, and shared FP16 codebook sidecars).

## Archive

Legacy additive, interaction, Block-CLADO, dense-cone, adjoint, and PrismaSCOUT
iteration/polish tools are archived for artifact replay and comparison under the
dated `archive/` walls.

The 2026-07-30 architecture re-vet added five: `l3_propagated_2026-07-30`
(the L2/L3 cascade — `kl_sensitivity_probe`, `propagated_sensitivity_costs`,
`sensitivity_response`, and the L3 half of `kl_measurement`),
`production_render_staged_2026-07-30`, `mse_promotion_2026-07-30`,
`union_cache_2026-07-30`, `block_output_match_2026-07-30`, plus
`orphans_2026-07-30`. Each wall's README carries the killing measurement and the
durable lesson; `docs/ARCHITECTURE.md` §11 indexes them.
