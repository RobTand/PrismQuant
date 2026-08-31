# Trellis serving-lane provenance, shipcard and served-route reconciliation (WO-D)

**Status:** implemented 2026-08-31, branch `muse/wo-d-trellis-provenance-20260831`. Pinned runtime Gridbook **0.9.1** commit `227420f`, wheel `cb4d7ad64c5a78d447f427a0aa98790406b6821d02c7f2f5d589d61890abdf9d` (dist-ci, not dist-final), contract `gridbook.runtime-contract.v12`, lane-eligibility schema `gridbook.lane-eligibility.v3` version 12.

This document records the cell table, the candidate-vs-attested gap, the missing routed-MoE cell as a declared serving gap, and the two attestation legs (principle 14). It is the D6 companion to `docs/ARCHITECTURE.md`'s re-stamp. No gating decision here is literal: every route verdict is a lookup into the pinned contract.

---

## 1. Cell table (four trellis cells, the whole trellis lane)

All cells are `platform: sm_121` (GB10/Spark, compute 12.1), `structure: dense`, `qualification: device_qualified`, `route_status: backed_with_serve_flag`. The table is the four trellis rows of the v12 contract's `lane_eligibility` block:

| cell id | family | regime | rungs_q256 | activation_contract | requires_serve_flags | qualification |
|---|---|---|---|---|---:|---|
| `trellis_e2m1_dense_sm121_decode_scaled_mm_w4a4` | `TCQ_E2M1_R256` | decode | `[512]` | `e2m1_group16_ue4m3_static` | `GRIDBOOK_TRELLIS_E2M1=1`, `GRIDBOOK_TRELLIS_E2M1_MODE=resident\|streamed` | `device_qualified` |
| `trellis_e2m1_dense_sm121_batch_scaled_mm_w4a4` | `TCQ_E2M1_R256` | batch | `[512]` | `e2m1_group16_ue4m3_static` | same | `device_qualified` |
| `trellis_e4m3_dense_sm121_decode_scaled_mm_w8a8` | `TCQ_E4M3_R256` | decode | `[1152]` | `fp8_per_token_dynamic` | `GRIDBOOK_TRELLIS_E4M3=1`, `GRIDBOOK_TRELLIS_E4M3_MODE=resident\|streamed` | `device_qualified` |
| `trellis_e4m3_dense_sm121_batch_scaled_mm_w8a8` | `TCQ_E4M3_R256` | batch | `[1152]` | `fp8_per_token_dynamic` | same | `device_qualified` |

Derived, not asserted: `prismaquant/gridbook_runtime/gridbook_runtime_contract.0.9.1.json` is the source; `prismaquant/gridbook_lane_eligibility.py` (`EligibilityCell`, `resolve_unit_route`) is the consumer parser; `prismaquant/allocator_candidates.py:selection_serving_lane_provenance` resolves per-unit by calling `unit_structural_facts` + `resolve_unit_route` with the target platform from the serving profile. A different pinned contract with different `rungs_q256` or `requires_serve_flags` changes the verdict — proven by `tests/test_trellis_serving_provenance.py::test_fixture_contract_changes_all_above`.

**Correction (WO-C, 2026-08-31):** Fused modules *are* trellis-eligible. Per `gridbook/config.py::_build_trellis_method` at the pinned commit, a fused serving unit (e.g., `qkv_proj`, `gate_up_proj` in `packed_modules_mapping` order) takes **one** trellis wire over the row-concatenated `[sum(out), in]` matrix, declared against the merged vLLM module name (e.g., `...self_attn.qkv_proj`). Per-role wires cannot be concatenated (each carries its own alphabets/schedule/row padding) and the smoke checkpoint's `ignore` for `q_proj/k_proj/v_proj` is a smoke, not the limit. The serving lane is **TP=1 only** (`max_world_size: 1` per `trellis_format_family` in the contract; `selection_serving_lane_provenance` records `trellis_tensor_parallel_world_size: 1` and `trellis_requires_tp1: true`; shipcard refuses `TP>1`).

`backed_with_serve_flag` is **not** `backed`. The artifact is only servable with those exact env flags set, so the flags are artifact metadata (stamped in `selection_serving_lane_provenance.trellis_requires_serve_flags` and `trellis_route_status.trellis_requires_serve_flags`, and carried to `prismaquant/lane_specs/gridbook_trellis.json`'s `served_activation_quantization` note and `scripts/serve_gridbook_trellis.sh`'s `-e` pass-through). The shipcard refuses a `backed_with_serve_flag` artifact that does not record them (WO-D D2, `prismaquant/shipcard.py:verify` gate 2).

---

## 2. Candidate vs attested rung gap (principle 1 carve-out)

The `formats` table publishes the producer menu:

- `TCQ_E2M1_R256`: `candidate_rungs_q256: [384, 512, 640, 768, 896]`, `reader_rate_range_q256: [256,1016]`
- `TCQ_E4M3_R256`: `candidate_rungs_q256: [1152]`, `reader_rate_range_q256: [256,2040]`

The lane-eligibility table attests **strict subsets**:

- E2M1: only `512` (one of five candidates)
- E4M3: `1152` (the single candidate, fully attested)

A producer candidate whose rate is not in any cell's `rungs_q256` (e.g., `TCQ_E2M1_R640`) carries a valid `rate_q256=640` fact but matches no cell in any regime. Under the closed-world v3 table, absence is the only negative signal, so it resolves `unattested` with `in_scope=True`, exposed as `route_status: "unbacked"` with `unattested_reason: "no lane cell covers regime(s) ['decode','batch'] for TCQ_E2M1_R256 rung 640 on sm_121"` — not silently omitted. The artifact-level `trellis_route_histogram` counts it as `TCQ_E2M1_R256::unbacked` (empty activation_contract). Export/shipcard fails closed on it unless a non-native target or per-run override is stamped (principle 9). Reporting the gap **is** the point (principle 1's one carve-out): the allocator that wants an unbacked rung is reporting a serving gap.

Evidence: `tests/test_trellis_serving_provenance.py::test_e2m1_640_resolves_unbacked_and_shipcard_refuses` and the `candidate_rungs_q256` vs `rungs_q256` inspection in `prismaquant/gridbook_runtime/gridbook_runtime_contract.0.9.1.json`.

---

## 3. Missing routed_moe trellis cell (declared serving gap)

No trellis cell has `structure: routed_moe`. Every trellis row in the contract is `dense`. A routed/packed-MoE trellis unit (`structure: routed_moe`, e.g., `model.layers.0.mlp.experts.*` → `TCQ_E2M1_R512`) therefore matches no cell in any regime and resolves `unbacked` with a reason naming the missing structure:

```
no lane cell covers regime(s) ['decode','batch'] for TCQ_E2M1_R256 rung 512 on sm_121
```

The refusal text names the cell gap (WO-D D1) and the shipcard gate treats it identically to the rung gap. This is a declared serving gap, not a code defect — the trellis kernel work to date is dense-only (`docs/design/trellis_serving_gap_2026-08-29.md` §3c).

Evidence: `tests/test_trellis_serving_provenance.py::test_routed_moe_trellis_resolves_unbacked_naming_missing_cell` and the fixture test that adds a `routed_moe` cell and sees the verdict flip to `backed_with_serve_flag`.

---

## 4. Two attestation legs (principle 14)

### Leg 1 — priced route (producer)

Per-unit resolution into the pinned contract (above) plus the artifact-level **route histogram** `trellis_route_histogram: (family, activation_contract, route_status) → count` (e.g., `{"TCQ_E2M1_R256:e2m1_group16_ue4m3_static:backed_with_serve_flag": 1}`) so a card can print `bpp` and `route` in the same table (principle 12). The histogram travels with the bpp/quality claim: `selection_serving_lane_provenance` writes it to `selection.json`; the exporter lifts it to `quant_config.json:provenance.trellis_route_status`; `build_shipcard` lifts it to `shipcard.json:trellis_route_status`. Shipcard's third gate refuses a bpp/quality claim without it.

Code: `prismaquant/allocator_candidates.py:selection_serving_lane_provenance` (trellis branch), `prismaquant/shipcard.py:build_shipcard` lift and `verify` gates 1–3.

### Leg 2 — served route must equal priced route (consumer)

`prismaquant/validate_native_export.py` compares the artifact's priced `activation_contracts` histogram against the routes the serve **actually emitted** (`emit_route` telemetry). For CB the seam is `gridbook/nvfp4_activation_contract.py:emit_route` / `read_route` (12 `_cb_route_*` scalars on the layer, read back by `scripts/pb_validation/dump_routes_probe.py`). For trellis lanes the pinned 0.9.1 tree sets `layer.gridbook_activation_contract` (`trellis_e2m1_lane.py:232`, `trellis_e4m3_lane.py:207`) but **does not call `emit_route`** — `grep -rn emit_route` on those two files is empty while `gridbook/linear.py` and `moe.py` call it at every CB dispatch.

**Finding (WO-D-FINDINGS.md):** Trellis served-route telemetry via the CB seam is absent. The comparison cannot be faked with an assumption. `validate_native_export.py:_collect_served_trellis_histograms` attempts to collect both `gridbook_activation_contract` and `read_route` telemetry; when `served is None` and the priced artifact contains trellis units, `verify_trellis_priced_vs_served` refuses for lack of evidence (fail-closed) rather than passing by default. A disagreement between priced and served activation_contract histograms also refuses.

Code: `prismaquant/validate_native_export.py:_load_priced_trellis_histograms`, `_collect_served_trellis_histograms`, `verify_trellis_priced_vs_served`, integrated into `_run_arm`'s post-generation check. Tests: `tests/test_trellis_serving_provenance.py::test_validate_native_export_refuses_on_histogram_disagreement`.

### Lane and serve

New lane `prismaquant/lane_specs/gridbook_trellis.json` (`runtime: vllm+gridbook_plugin`, endpoint `openai` on `127.0.0.1:8000`, serve script `scripts/serve_gridbook_trellis.sh` using `gridbook_runtime_prepare` / `GRIDBOOK_RUNTIME_DOCKER_ARGS` / in-container `install-container` and `vllm serve --host 0.0.0.0 --port 8000` with `-p 8000:8000` and `GRIDBOOK_TRELLIS_*` flags pass-through). KL evaluator is `validate_assignments_kl:main` (same interface as `gguf_kl_evaluator`).

Provenance stamp: `docs/ARCHITECTURE.md` re-stamped 2026-08-31 for `muse/wo-d-trellis-provenance-20260831`.

