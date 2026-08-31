# Trellis export into the Gridbook container — 2026-08-31

**Branch:** `muse/wo-c-trellis-export-20260831` · **Worktree:** `/home/rob/wo-c-trellis-export` · **Pinned runtime:** Gridbook 0.9.1 (commit `227420f`, wheel `cb4d7ad64c5a78d447f427a0aa98790406b6821d02c7f2f5d589d61890abdf9d`, `dist-ci` not `dist-final`)

This is the exporter half of the trellis lane. The research surface (`trellis_formats`, `trellis_rate_surface`, `trellis_allocator`) and the render mechanism (`trellis_producer` / `trellis_encoder`, WO-B's ProductionWeightCache retention) exist elsewhere. This document records the carriage contract, the hard runtime facts, and the per-artifact gates that the exporter must enforce.

## 1. Carriage contract — the wire is the only carrier

Reference: `tools/make_trellis_smoke_checkpoint.py` in the pinned runtime. Docstring: *"A real exporter substitutes an encoder here and changes nothing else."* Its emitted shape is the contract this exporter implements.

Per trellis Linear `<target>` (e.g. `model.layers.3.self_attn.o_proj`):

| tensor | dtype | meaning |
|---|---|---|
| `<target>.wire_bytes` | `uint8` 1-D | the entire `TrellisWire.to_bytes()` blob (header, per-column schedule, tight block offsets, per-rate alphabets, scale plane, padded row bodies) |
| `<target>.trellis_input_global_scale` | `float32` `[1]` | **E2M1 only.** The A-side static activation scale for the `e2m1_group16_ue4m3_static` contract |

`config.json` → `quantization_config`:

```json
{
  "quant_method": "gridbook",
  "format": "mixed-precision",
  "config_groups": {
    "<group>": {
      "format": "TRELLIS",
      "targets": ["<target>"],
      "scheme": {
        "family": "TCQ_E2M1_R256",
        "body_rate_q256": 512,
        "rows": 256,
        "columns": 512,
        "wire_bytes": 1833
      }
    }
  },
  "ignore": ["lm_head", "model.layers.0.self_attn.q_proj", "..."]
}
```

Rules (each is stated in `gridbook/trellis_scheme.py`):

1. **The wire is the only carrier.** Schedule, alphabets, block offsets and scale plane exist nowhere else. Never emit them as separate tensors and never emit a `[rows, row_stride]` payload rectangle. The exporter writes one opaque `wire_bytes` blob per Linear; the reader derives everything from it.
2. **Every scale is derived from the blob, never loaded beside it** — except E2M1's `trellis_input_global_scale`, which is genuinely not a wire fact (it is the A-side activation quantization scale, not a weight scale). Emitting a redundant scale tensor creates state that can disagree with its own wire and nothing would notice.
3. **Fused modules cannot be trellis.** Reuse the union-find serving-unit promotion already in `allocator_solver.py` rather than writing a second grouping rule.
4. **Dense only.** The pinned contract publishes no `routed_moe` trellis cell; a routed/packed-MoE unit assigned a trellis rung must fail export closed naming the missing cell.

`quantization_config.format` is `"mixed-precision"` when the assignment mixes families (e.g. trellis + NVFP4), otherwise the existing single-family spellings are kept (the Gridbook container historically uses `"nvfp4_cb"`). The smoke checkpoint always writes `"mixed-precision"` because its 4 trellis units coexist with BF16 passthrough linears.

### Implementation

* **Exporter:** `prismaquant/export_nvfp4_cb.py` — the Gridbook container exporter (`quant_method: "gridbook"`, `format: "nvfp4_cb" | "fp8_cb" | `"mixed-precision"`). It is **not** `export_native_compressed.py`, which is the `compressed-tensors` container and correctly refuses trellis.
* **Wire bytes source:** `ProductionWeightCache` (WO-B). The exporter never re-encodes — principle 8. If the cache has no wire for a selected trellis unit, it fails closed. While WO-B is not merged, the exporter codes against the seam: `trellis_wire_cache: dict[(qname, fmt) → bytes]` passed explicitly, or `ProductionWeightCache.get_trellis_wire_bytes` when that method exists; missing is a hard error (message names the seam and cites WO-B).
* **Byte authority:** `prismaquant/trellis_footprint.py` — but the wire's own `to_bytes()` length is the ground truth for `scheme.wire_bytes`; the footprint is the pricing receipt, the wire is the shipped bytes.
* **Profile:** `prismaquant/serving_profile_specs/nvfp4_cb.json` now allows `ALL_LEGAL_TRELLIS_FORMAT_NAMES` (2546 addressable rungs) via `allow_formats_from`, so `check_serving_format` admits a trellis rung. The target platform stays `sm_121` — the only platform that publishes trellis cells (see §3).

## 2. Fused-module restriction — a hard runtime fact

vLLM merges `q/k/v` and `gate/up` into one module (`self_attn.qkv_proj`, `mlp.gate_up_proj`). Per-role trellis wires cannot be concatenated — each carries its own alphabets, rate schedule and row padding, and the kernel loads one wire per Linear. `gridbook/config.py` refuses such a target *by name* (`_build_trellis_method` checks `fused_role_owners` / `incomplete_fused_roles` and raises: *"per-role trellis wires cannot be concatenated"*). On a Qwen-shaped architecture the trellis-eligible Linears are the unfused ones: `o_proj` and `down_proj`. Fused siblings must take a non-trellis format (NVFP4/FP8/BF16) or go in `ignore`.

This is not a policy the producer may relax. The allocator already promotes fused siblings via union-find (`allocator_solver.promote_fused`, `allocator_solver.union`); the exporter mirrors that same grouping via `nvfp4_activation_contract.fused_sibling_group_key` rather than inventing a second rule. A trellis unit whose name resolves to a fused group fails closed at export with that exact message.

## 3. Rung-vs-cell distinction — per-artifact gating (principle 9)

The pinned contract (schema `gridbook.runtime-contract.v12`, `gridbook.lane-eligibility.v3`) publishes two different sets:

* `formats[]` (`kind: "tcq_trellis"`) → `candidate_rungs_q256`: the wire vocabulary the codec can decode (E2M1: `[384, 512, 640, 768, 896]`, E4M3: `[1152]`).
* `lane_eligibility.cells[]` → `rungs_q256`: the serving cells actually attested on `sm_121` (E2M1: `[512]` only, E4M3: `[1152]` only; each with `route_status: "backed_with_serve_flag"`, `qualification: "device_qualified"`, `requires_serve_flags: ["GRIDBOOK_TRELLIS_E2M1=1", "GRIDBOOK_TRELLIS_E2M1_MODE=resident|streamed"]` etc.).

A rung may be a **producer candidate** and still have **no attested serving cell**. Export must gate on the *cell*, not the candidate list. Absence is the runtime's only negative signal (the v3 table carries no `unbacked` cell), so a rung absent from every cell resolves `unattested` and the gate fails closed — exactly the `"units_on_fallback_route = 0"` defect wearing a newer schema, but now per artifact at export.

The gate is `prismaquant/gridbook_lane_eligibility.py` (attested, never asserted) + `prismaquant/cb_route_status_gate.py` (the unified entry `gate_cb_export_units` is shared between the in-memory and streaming exporters). Every selected trellis unit's `UnitStructuralFacts` (family, `rate_q256`, structure `dense`, platform `sm_121`) must resolve to a cell with `platform: sm_121`, matching `family`, `structure: dense`, and `rate_q256` present in that cell's `rungs_q256`. A rung with no cell refuses unless the artifact carries an explicit non-native-target declaration (`PQ_CB_NON_NATIVE_TARGET`) or per-run override (`PQ_CB_ROUTE_STATUS_OVERRIDE`) — stamped on the shipcard either way.

`route_status: "backed_with_serve_flag"` means the artifact must record its `requires_serve_flags` so the serve command can be reconstructed. For trellis:

* E2M1: `GRIDBOOK_TRELLIS_E2M1=1`, `GRIDBOOK_TRELLIS_E2M1_MODE=resident|streamed`
* E4M3: `GRIDBOOK_TRELLIS_E4M3=1`, `GRIDBOOK_TRELLIS_E4M3_MODE=resident|streamed`

These are stored in `quant_config.json:provenance.cb_route_status.requires_serve_flags`, in `provenance.selection_serving_lane_provenance` (the legacy key a gate reads), and in `shipcard.json` as structured fields — never prose.

## 4. A-side static scale — the one genuinely new quantity

E2M1's `trellis_input_global_scale` is the activation quantization scale for the `e2m1_group16_ue4m3_static` contract the pinned E2M1 cell declares. It is the **same execution contract** as stock NVFP4's `e2m1_group16_ue4m3_static` (`nvfp4_activation_contract.NVFP4_ACTIVATION_EXECUTION`), and is therefore derived identically:

```
input_global_scale = input_global_scale_from_max_abs(calibration_amax, policy)
```

* `calibration_amax` = `max_abs` of the calibration activations for that target (the exact `ActivationIndex` rows the probe cached, loaded via `load_activation_cache_samples` / `calibrated_input_global_scales_with_sources`).
* `policy` = `resolve_input_global_scale_policy(activation_scale_policy)` — the legacy `6 / amax` vs full-E4M3 `448*6 / amax` choice, resolved once at export startup and stamped in provenance. No new constant is invented.
* The artifact tensor is `float32` shape `[1]`, `struct.pack("<f", ...)`-rounded, `input_global_scale_tensor`.

Fused-sibling unification (`_unify_input_global_scales_across_fused_siblings`) does **not** apply: trellis units are unfused by rule 3. The exporter states this explicitly in a comment rather than leaving it silently unhandled.

Missing activations for a trellis E2M1 unit is **fail-closed** — there is no defensible default for an activation scale, and the export raises naming the missing target.

E4M3's activation contract is `fp8_per_token_dynamic` (per-token, no static scale) and carries no `trellis_input_global_scale`.

## 5. Files changed

* `prismaquant/export_nvfp4_cb.py` — trellis emission, wire-from-cache seam, E2M1 scale, route gating, mixed-precision format, `TRELLIS` config_groups.
* `prismaquant/layer_config.py` — `canonicalize_format` now recognizes `TCQ_E*M*_R*`.
* `prismaquant/serving_profile_specs/nvfp4_cb.json` — admits `ALL_LEGAL_TRELLIS_FORMAT_NAMES`.
* `prismaquant/trellis_wire.py`, `prismaquant/trellis_footprint.py` — unchanged (byte authority).
* `tests/fixtures/trellis_wire_golden/` — 10 golden wires (both layouts, both families, 4 E2M1 rungs + E4M3, non-multiple-256 cols, non-multiple-16 rows).
* `tests/test_trellis_wire_golden.py` — decodes with `prismaquant.trellis_wire`, asserts bit-exact.
* `tools/verify_trellis_golden_gridbook.py` — standalone verifier that decodes the same fixtures with `gridbook.trellis`.
* `tests/test_trellis_export_smoke.py` — 2-layer hidden-256 model, o_proj/down_proj on `TCQ_E2M1_R512`, rest BF16, asserts tensor names and config shape field-for-field vs the smoke checkpoint contract.

WO-B seam note: the wire bytes are retained by `ProductionWeightCache` (branch `muse/wo-b-trellis-render-20260831`). This branch codes against that seam; if WO-B is not merged, pass `trellis_wire_cache={qname: blob}` explicitly (the smoke test does) and note it in the commit message. Do not re-encode at export.

## 6. Validation

* `PYTHONPATH=. python -m pytest tests/test_trellis_wire_golden.py` — 2 passed
* `PYTHONPATH=. python -m pytest tests/test_trellis_export_smoke.py` — 1 passed
* `tools/verify_trellis_golden_gridbook.py --fixtures tests/fixtures/trellis_wire_golden` via `PYTHONPATH=/home/rob/gridbook` — 10 passed bit-exact
* Targeted suite `pytest -k "export or trellis or serving or architecture"` then full suite — see commit message for counts (principle 13).
