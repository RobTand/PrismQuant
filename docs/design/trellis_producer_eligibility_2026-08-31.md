# Trellis Producer Eligibility — 2026-08-31 (WO-A)

## What the pin now attests

The pinned Gridbook runtime is **0.9.1** (`227420f`, contract
`gridbook.runtime-contract.v12`, `gridbook.lane-eligibility.v3`) materialized at
`prismaquant/gridbook_runtime/gridbook_runtime_contract.0.9.1.json`.

It publishes, under `formats`:

```json
{"kind": "tcq_trellis", "family": "TCQ_E2M1_R256",
 "candidate_rungs_q256": [384, 512, 640, 768, 896],
 "reader_rate_range_q256": [256, 1016], "native_terminal_q256": 1024,
 "residency_modes": ["resident", "streamed"]}
{"kind": "tcq_trellis", "family": "TCQ_E4M3_R256",
 "candidate_rungs_q256": [1152],
 "reader_rate_range_q256": [256, 2040], "native_terminal_q256": 2048,
 "residency_modes": ["resident", "streamed"]}
```

and, under `lane_eligibility.cells`, four device-qualified trellis cells:

- `trellis_e2m1_dense_sm121_decode_scaled_mm_w4a4` — `sm_121`, `TCQ_E2M1_R256`,
  `dense`, `decode`, `rungs_q256: [512]`,
  `activation_contract: "e2m1_group16_ue4m3_static"`,
  `route_status: "backed_with_serve_flag"`, `qualification: "device_qualified"`,
  `requires_serve_flags: ["GRIDBOOK_TRELLIS_E2M1=1",
  "GRIDBOOK_TRELLIS_E2M1_MODE=resident|streamed"]`
- `trellis_e2m1_dense_sm121_batch_scaled_mm_w4a4` — same, `batch`
- `trellis_e4m3_dense_sm121_decode_scaled_mm_w8a8` — `TCQ_E4M3_R256`,
  `rungs_q256: [1152]`, `fp8_per_token_dynamic`
- `trellis_e4m3_dense_sm121_batch_scaled_mm_w8a8` — same, `batch`

This is the **executed-activation-contract table** that was missing when
`serving_profile_specs/trellis_research_sm121.json` was written (its
`_no_format_rules_rationale` explains why 2546 stubs were refused). The
panel now exists at the pin, which is the only thing that changes. Every
number, platform, route status, and activation contract below is **derived from
this file**, never retyped. If the file and the work order disagree, the file
wins.

## What is therefore newly legal

* **Five E2M1 rungs as producer-eligible FormatSpecs.** `prismaquant/format_registry.py`
  now registers exactly `candidate_rungs_q256` for `TCQ_E2M1_R256` — today
  `TCQ_E2M1_R384`, `R512`, `R640`, `R768`, `R896` — under the canonical names
  `trellis_formats.parse_trellis_format_name` accepts. The list is read at import
  time from the pinned contract via `format_registry.load_trellis_candidate_rungs`
  (which delegates to `gridbook_lane_eligibility.load_published_formats`), so a
  future pin that adds a rung adds a format and a pin that drops one drops a format
  with no source edit. A fixture test proves it.

  * `family` is the new `tcq_trellis`. Every `family` switch in the tree
    (`render_production_weight`, `WEIGHTED_RENDER_FAMILIES`, `check_format_applicability`,
    cost paths, export) now explicitly handles or explicitly refuses this value;
    a silent fall-through to a scalar default is impossible.

  * `quantize_dequantize` is the **unweighted** trellis encode:
    `trellis_producer.encode_trellis_one_linear` with `col_weights = ones(in_features)`
    and `return .decoded_weight`. It is expensive (full 256-state tail-biting Viterbi
    per superblock) and that cost is correct — a trellis rung has no cheap RTN;
    pretending otherwise would price a fiction. No cache is added here; WO-B owns
    the `ProductionWeightCache` wire.

  * `activation_quantize_dequantize` for the W4A4 lane is `e2m1_group16_ue4m3_static`.
    The tree has an exact implementation (`nvfp4_activation_contract.nvfp4_activation_qdq_served`)
    but it requires a calibrated `input_global_scale` per fused sibling and therefore
    has no unary `FormatSpec` signature. The trellis FormatSpec therefore
    **fails closed with a loud `NotImplementedError` naming the contract** rather
    than substituting a dynamic per-group RTN.

  * **Byte authority is `trellis_footprint.trellis_tensor_payload_breakdown`.**
    Body rate alone is never artifact bpp. The seam is
    `footprint.format_tensor_payload_breakdown`, the same seam CB uses. A trellis
    payload bundles the 88-byte wire header, schedule nibbles, offset table
    (`tight_offsets` vs `fixed_quota_per_256`), alphabet blob, row padding (16-byte
    alignment), and the family-specific scale plane (`group16_fp8_e4m3_0p5_bpw` for
    E2M1: one `float8_e4m3fn` byte per 16 weights plus the global `float32`). The
    canonical schedule and alphabets are deterministically derived from shape and
    `q256` via `trellis_formats.canonical_trellis_schedule` /
    `canonical_trellis_alphabets` (centered symmetric alphabets, low-then-high
    rate order to keep `decode_codes_torch` in bounds), so the same recipe prices
    and renders one wire. `FormatSpec.effective_bits == q256/256` is retained only
    as a nominal body-rate marker for ordering/display and is explicitly
    **nominal-and-unused** for budgeting.

* **A shippable serving profile.** New
  `prismaquant/serving_profile_specs/gridbook_trellis_dense_sm121.json`:

  * `target_platform: "sm_121"` (v3 cells are platform-scoped; a profile with no
    platform resolves every unit `unattested` with source `no_target_platform`).

  * Admits the five E2M1 rungs plus the honest scalar terminals an honest mixed
    menu needs (`BF16`, `FP8_E4M3`, `NVFP4` as existing profiles spell them).
    Other trellis rungs (e.g. `q256=1000`, legal to the reader range but not a
    candidate) are not registered and are refused by name.

  * Carries per-lane **structured** `route_status` and `requires_serve_flags`
    **derived at load time** from the pinned contract's cells via
    `ServingLaneSpec.route_status_for` (which reads
    `gridbook_runtime/gridbook_runtime_contract.0.9.1.json` through
    `gridbook_lane_eligibility.load_eligibility_table`). The method filters by
    `(platform, family, structure, rung)` and by `structure: ["dense"]`; a
    trellis cell with a predicate would resolve `unit_dependent` and defer to the
    export `cb_route_status_gate`, but the two E2M1 dense cells have none, so
    `R512` on `sm_121` resolves `backed_with_serve_flag` with
    `GRIDBOOK_TRELLIS_E2M1=1` and `GRIDBOOK_TRELLIS_E2M1_MODE=resident|streamed`,
    while `R384`/`R640`/`R768`/`R896` have no cell and resolve `unattested` with
    source `rung_not_listed`. Nothing is retyped; a fixture contract with
    `route_status: "fallback"` (or an invalid `"unbacked"` cell) is observed as
    such, proving derivation.

  * **Dense-only.** The contract publishes **no** `routed_moe` trellis cell.
    A routed-MoE unit (`...experts.gate_up_proj`, 3-D packed shape, or any
    `packed_expert: true`) under this profile fails closed via
    `format_rules.trellis_dense_only` / `allocator_candidates.check_format_applicability`
    with a message that says the pinned runtime publishes no routed trellis cell.
    `in_features % 256 != 0` also fails closed as `kernel_shape`.

* **Corrected export refusal.** `prismaquant/export_native_compressed.py:1658-1683`
  previously said the producer pin publishes no executed-activation-contract table
  for `TCQ_E2M1`/`TCQ_E4M3` — false at the 0.9.1 pin (see the four cells above).
  The message now says the remaining true blocker is `ProductionWeightCache`
  renders no trellis wire (owned by WO-B); the activation table is attested and
  the lane is declared. No export pack path is added; that is WO-C.

* **Menu ledger.** `prismaquant/trellis_menu.UNWIRED_LINKS` shrinks from eight to
  six: the loud `format_registry` and `footprint` byte-budget links are now wired;
  the six silent aggregation/currency/provenance links remain and the
  `TRELLIS_SURFACE` seam still refuses when the flag is set.

## What is still refused

* **Routed MoE.** No `TCQ_E*` cell with `structure: "routed_moe"` exists in the
  pinned contract on any platform. Every `experts.` / packed-MoE unit that asks
  for a trellis rung under `gridbook_trellis_dense_sm121` is denied, and the
  allocator's `check_format_applicability` refuses a 3-D shape for `tcq_trellis`.

* **Every rung outside `candidate_rungs_q256`.** A rung that is legal to the
  wire's `reader_rate_range_q256` (e.g. `TCQ_E2M1_R1000`, 256..1016) but not in
  the contract's candidate list is not in the registry and is refused by name
  (`KeyError: Unknown format` and `profile_mismatch`). The verbatim 2546-name
  vocabulary (`ALL_LEGAL_TRELLIS_FORMAT_NAMES`) remains addressable for a
  `allow_formats_from` rule but is not a promise of a kernel.

* **The E4M3 family.** The contract's `TCQ_E4M3_R256` candidate is `[1152]`
  and it has two `sm_121` dense cells (`fp8_per_token_dynamic`), but **WO-A does
  not register it**. Only `TCQ_E2M1_R256` is promoted. Registering E4M3 is a later
  phase with its own `per_output_row_fp32` scale contract and `w8a8` lane.

* **The remaining trellis seam.** The six `UNWIRED_LINKS` entries that survive
  WO-A are still the production seam's refusal text: the allocator's exact
  `_memory_bytes_by_format` filter, fused/packed aggregation by `FormatSpec`,
  `promote_serving_units` rank, `build_candidates` currency/provenance wiring,
  and the `trellis_rate_surface` weighted-SSE currency. Passing
  `PRISMAQUANT_TRELLIS_SURFACE` still raises `TrellisSeamUnwiredError`.

## Byte seam used

Exact bytes are priced through `footprint.format_tensor_payload_breakdown`,
which for a trellis format delegates to
`trellis_footprint.trellis_tensor_payload_breakdown` with the canonical
schedule/alphabets. The `FormatSpec` scalar (`weight_bits=0, group_size=256,
scale_bits=q256`, so `effective_bits == q256/256`) is the nominal body rate
and is documented as not authoritative.
