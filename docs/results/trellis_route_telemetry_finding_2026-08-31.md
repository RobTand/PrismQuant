# WO-D Findings — 2026-08-31

## Trellis lane emit_route telemetry absent (D3 second leg)

**Checked:** Gridbook 0.9.1 (`227420f`, wheel `cb4d7ad64c5a78d447f427a0aa98790406b6821d02c7f2f5d589d61890abdf9d`, contract v12 `gridbook.lane-eligibility.v3`) trellis lanes `trellis_e2m1_lane.py` and `trellis_e4m3_lane.py` in the pinned serving release checkout (`/home/rob/.cache/prismaquant/gridbook-runtime/227420f9821bab7089632ee914f0ba050f82b817`).

**Finding:** Both trellis lanes set `layer.gridbook_activation_contract` (`e2m1_group16_ue4m3_static` / `fp8_per_token_dynamic`) and residency mode attributes at `create_weights`/`process_weights_after_loading`, but **do not call `nvfp4_activation_contract.emit_route`** anywhere. `grep -rn emit_route` on those two files returns zero hits, while `gridbook/linear.py` and `gridbook/moe.py` (CB lanes) call `emit_route` at every dispatch site.

For CB, `validate_native_export` and `scripts/pb_validation/dump_routes_probe.py` consume `read_route` (the 12 `_cb_route_*` scalars) as the served-route seam. For trellis, that seam is empty: `_collect_served_trellis_histograms` walking `named_modules()` finds `gridbook_activation_contract` attributes but no `_cb_route_*` records. Whether the `gridbook_activation_contract` attribute alone counts as telemetry is ambiguous, but it is **not** the `emit_route` mechanism WO-D names and the spec says to reuse. A gate that assumed equality between priced and served contracts without a per-forward telemetry record would be the "confession log" defect again.

**Disposition (fail-closed):** `prismaquant/validate_native_export.py:verify_trellis_priced_vs_served` now treats a priced trellis artifact with `served is None` (no emit_route records) as a refusal for lack of evidence, with a message naming the gap. `prismaquant/validate_native_export.py:_collect_served_trellis_histograms` attempts to collect both `gridbook_activation_contract` and `read_route` telemetry; when neither is present it returns `None` and the gate refuses rather than passes by default. This is recorded here per WO-D Rules ("Contradictions go in WO-D-FINDINGS.md") and is not a workaround.

**Principle reference:** 14 (both legs — priced claim derived from pinned contract's `lane_eligibility` table; served claim must be observed via `emit_route` telemetry) and 13 (measurement before/after — trellis throughput/contract correctness not re-proven here).

## Pinned contract verification

- Pinned runtime contract `prismaquant/gridbook_runtime/gridbook_runtime_contract.0.9.1.json` contains exactly four trellis cells as stated in WO-D, all `platform: sm_121`, `structure: dense`, `qualification: device_qualified`, `route_status: backed_with_serve_flag`, rungs `[512]` / `[1152]`, contracts `e2m1_group16_ue4m3_static` / `fp8_per_token_dynamic`, flags `GRIDBOOK_TRELLIS_E2M1=1, GRIDBOOK_TRELLIS_E2M1_MODE=resident|streamed` and `GRIDBOOK_TRELLIS_E4M3=1, GRIDBOOK_TRELLIS_E4M3_MODE=resident|streamed`. Verified via `grep` and `python -c` inspection; no `dist-final` wheel reference found (`grep -r dist-final` empty). The `dist-ci` wheel digest `cb4d7ad64c5a78d447f427a0aa98790406b6821d02c7f2f5d589d61890abdf9d` matches both `gridbook_runtime_pin.json` and `gridbook_serving_runtime_pin.json`.
- `formats` table candidate rungs `[384,512,640,768,896]` for E2M1 vs attested `[512]` confirms the strict-subset gap.
- No `routed_moe` trellis cell exists; verified `grep` shows only `dense` trellis cells.

## No other contradictions

No other WO-D Rules contradictions found. The lane spec `gridbook_trellis.json` and serve script `scripts/serve_gridbook_trellis.sh` follow the `serve_qwen27b_smoke.sh` pattern with required flags passed through and `--host 0.0.0.0 --port 8000` / `-p 8000:8000` binding.
