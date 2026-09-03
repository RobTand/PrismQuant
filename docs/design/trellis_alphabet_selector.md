# Trellis alphabet selection on the shipping export path

> **ARCHIVED 2026-09-02 (RobTand/prismaquant#118).** The modules this document
> describes — the five `prismaquant/trellis_*.py`,
> `serving_profile_specs/trellis_research_sm121.json` and the four
> `tests/test_trellis_*.py` — now live at `archive/trellis_wire_2026-09-02/`,
> whose README is their obituary. They priced `gridbook.trellis.wire.v1`, and
> Robert retired the Gridbook lane on 2026-09-02, so no sanctioned runtime
> reads those bytes. Tessera's wire is `prismaquant.tessera.v1`, a different
> plane set and deliberately not a port. `PRISMAQUANT_TRELLIS_SURFACE` is still
> refused, loudly, by `allocator_candidates.refuse_retired_trellis_surface`.
> Everything below is history. See `docs/ARCHITECTURE.md` §4.9.

**Verdict: the 4.0 and 5.0 bpw cells do not resolve.** PrismaQuant does not
currently ship a trellis render or export path, so neither `lloyd` nor
`exact_dp` is the selector of a shipping trellis artifact. `lloyd` is the
default only in the out-of-tree ladder driver. The production allocator refuses
the trellis seam when enabled, and all three permitted export containers reject
a manually injected `TCQ_E2M1_*` or `TCQ_E4M3_*` assignment before rendering.

This is stronger than finding that `exact_dp` is research-only: **`lloyd` is
also research-only.** Therefore the conditional ladder result (“fp8-CB wins if
the trellis uses Lloyd”) cannot be promoted to a present-tense shipping-menu
verdict. There is no shipping trellis arm whose selector can be identified.

Code baseline: PrismaQuant `8461ae2a3674c57d37d74a452d8d7f08e968ee43`
(`origin/main`, 2026-08-30). The external Gridbook comparison below distinguishes
the immutable shipping pin from an unreleased consumer branch.

## 1. Traced call path

### 1.1 A normal production run cannot create a trellis assignment

`allocator_candidates.build_candidates` always passes its candidate map through
`trellis_menu.augment_candidates` before returning
([`allocator_candidates.py:2023-2038`](../../prismaquant/allocator_candidates.py#L2023-L2038)).
That seam has only two executable branches:

1. Resolve `manifest_path` or `PRISMAQUANT_TRELLIS_SURFACE`. If neither is set,
   return the original candidates unchanged
   ([`trellis_menu.py:547-580`](../../prismaquant/trellis_menu.py#L547-L580)).
2. If either is set, raise `TrellisSeamUnwiredError`; it does not call
   `build_trellis_menu`
   ([`trellis_menu.py:582-590`](../../prismaquant/trellis_menu.py#L582-L590)).

`build_trellis_menu` remains directly callable for research, but the production
seam expressly does not render, export, or serve
([`trellis_menu.py:66-71`](../../prismaquant/trellis_menu.py#L66-L71)). Thus a
normal production solve cannot write a TCQ selection to `layer_config.json`.
The exporter refusals below are the second fail-closed wall for a hand-authored
or otherwise injected assignment.

### 1.2 The shipping export dispatch is exhaustive

The accepted `EXPORT_CONTAINER` vocabulary is exactly
`compressed-tensors`, `nvfp4_cb`, and `gguf`; unknown values fail canonicalization
([`model_profiles/structure.py:47-68`](../../prismaquant/model_profiles/structure.py#L47-L68)).
`compressed-tensors` is the unset default
([`run-pipeline.sh:92`](../../prismaquant/run-pipeline.sh#L92)). These are all
branches:

| Condition | Export entry point | What happens to a TCQ assignment |
|---|---|---|
| `EXPORT_CONTAINER == "gguf"` | `python3 -m prismaquant.export_gguf` ([`run-pipeline.sh:2554-2600`](../../prismaquant/run-pipeline.sh#L2554-L2600)) | Any format outside `GGUF_BLOCK_BYTES` and `BF16` raises before tensor packing ([`export_gguf.py:234-259`](../../prismaquant/export_gguf.py#L234-L259)). Both TCQ families are outside that set. |
| `EXPORT_CONTAINER == "nvfp4_cb"`, with `EXPORT_STREAMING` true | `prismaquant.export_nvfp4_cb_streaming` ([`run-pipeline.sh:2674-2679`](../../prismaquant/run-pipeline.sh#L2674-L2679)) | The streaming classifier admits CB, declared passthrough, stock-CT, declared native-requant, and BF16 branches; everything else is appended to `illegal` and raises ([`export_nvfp4_cb_streaming.py:3236-3279`](../../prismaquant/export_nvfp4_cb_streaming.py#L3236-L3279)). TCQ reaches `illegal`. |
| `EXPORT_CONTAINER == "nvfp4_cb"`, with `EXPORT_STREAMING` false | `prismaquant.export_nvfp4_cb` ([`run-pipeline.sh:2676-2680`](../../prismaquant/run-pipeline.sh#L2676-L2680)) | The in-memory classifier admits CB, `FP8_SOURCE`, delegated stock CT, and BF16; any other format raises ([`export_nvfp4_cb.py:853-880`](../../prismaquant/export_nvfp4_cb.py#L853-L880)). TCQ is not admitted. |
| `EXPORT_CONTAINER == "nvfp4_cb"`, with `EXPORT_STREAMING` `auto` or empty | In-memory unless source size is known and at least `EXPORT_STREAMING_THRESHOLD_GB` (default 80), then streaming ([`run-pipeline.sh:2673-2685`](../../prismaquant/run-pipeline.sh#L2673-L2685)) | Either selected exporter takes one of the two refusal paths above. An unknown `EXPORT_STREAMING` spelling itself fails ([`run-pipeline.sh:2687-2690`](../../prismaquant/run-pipeline.sh#L2687-L2690)). |
| Otherwise (therefore `compressed-tensors`) | `python3 -m prismaquant.export_native_compressed` ([`run-pipeline.sh:2791-2835`](../../prismaquant/run-pipeline.sh#L2791-L2835)) | The native preflight recognizes both TCQ families and raises the dedicated “no export path / no rendered wire” error ([`export_native_compressed.py:1658-1683`](../../prismaquant/export_native_compressed.py#L1658-L1683)). |

The native path is the most direct trace from shipping entry point to the point
where an alphabet-producing call would have had to occur:

```text
run-pipeline.sh
  -> python -m prismaquant.export_native_compressed
  -> main()
  -> _main_impl()
  -> _canonicalize_assignment(layer_config)
  -> _coerce_runtime_legal_assignment()
  -> parse_trellis_format_name(fmt) is not None
  -> ValueError (stop)
```

`main` opens the artifact transaction and calls `_main_impl`
([`export_native_compressed.py:8601-8634`](../../prismaquant/export_native_compressed.py#L8601-L8634)).
`_main_impl` loads and canonicalizes the assignment, then calls the legality
preflight
([`export_native_compressed.py:8829-8843`](../../prismaquant/export_native_compressed.py#L8829-L8843)).
Inside that preflight the complete non-exportable-format branch order is:

1. `BF16` -> continue.
2. An exportable native format -> proceed to ordinary shape/runtime legality.
3. A non-exportable CB format -> wrong-container error.
4. A non-exportable GGUF format -> wrong-container error.
5. A format for which `parse_trellis_format_name(...)` returns non-`None` ->
   dedicated TCQ no-render/no-export error.
6. Any other non-exportable format -> generic no-emit-path error.

Those conditions and their order are executable at
[`export_native_compressed.py:1627-1700`](../../prismaquant/export_native_compressed.py#L1627-L1700).
The TCQ error occurs before CUDA is required and before
`materialize_tensors_streaming` is called
([`export_native_compressed.py:9092-9098`](../../prismaquant/export_native_compressed.py#L9092-L9098),
[`export_native_compressed.py:9127-9134`](../../prismaquant/export_native_compressed.py#L9127-L9134)).
There is consequently no render, no alphabet construction, and no selector call
below the shipping entry point.

### 1.3 Both trellis families hit the same refusal

The parser used by the native refusal recognizes exactly
`TCQ_E2M1_R<q256>` and `TCQ_E4M3_R<q256>`
([`trellis_formats.py:35-47`](../../prismaquant/trellis_formats.py#L35-L47)) and
returns the canonical family plus validated rate for either spelling
([`trellis_formats.py:482-488`](../../prismaquant/trellis_formats.py#L482-L488)).
There is no E4M3-versus-E2M1 selector branch before the error.

The code that can account a research trellis recipe also takes `alphabets` as
an already-produced input and only validates it
([`trellis_footprint.py:438-482`](../../prismaquant/trellis_footprint.py#L438-L482)).
`validate_alphabets` checks rates, sizes, native-code range, ordering, and
duplicates; it does not choose levels
([`trellis_formats.py:596-662`](../../prismaquant/trellis_formats.py#L596-L662)).
That is an accounting/contract surface, not a hidden production selector.

## 2. Configuration and the unset default

There is no shipping alphabet-selector configuration.

- `PRISMAQUANT_TRELLIS_SURFACE` is an allocation-manifest switch, not an
  alphabet selector. Unset means “do not add TCQ candidates”; set means “refuse
  the unwired seam”
  ([`trellis_menu.py:99-101`](../../prismaquant/trellis_menu.py#L99-L101),
  [`trellis_menu.py:578-590`](../../prismaquant/trellis_menu.py#L578-L590)).
- `PRODUCTION_CACHE_LEVERS` defaults to
  `gptq,static_act_order,joint_scale_opt`
  ([`run-pipeline.sh:700-715`](../../prismaquant/run-pipeline.sh#L700-L715)). The
  production render resolver has GPTQ/damp, scale sweep, static activation
  order, joint scale optimization, Fisher-GPTQ, and the NVFP4 scale rule; it
  has no trellis or alphabet mechanism
  ([`production_weight_cache.py:4633-4695`](../../prismaquant/production_weight_cache.py#L4633-L4695),
  [`production_weight_cache.py:4698-4718`](../../prismaquant/production_weight_cache.py#L4698-L4718)).
- A recipe can supply research alphabets to the accounting API, but there is no
  recipe selector field and no production code that constructs a wire from
  those inputs.
- The pinned Gridbook producer is release 0.8.11 at commit `187c721...`
  ([`gridbook_runtime_pin.json:1-12`](../../prismaquant/gridbook_runtime/gridbook_runtime_pin.json#L1-L12)).
  Its complete format table contains only `NVFP4_CB_K`, `NVFP4_CB_S`, and
  `FP8_CB_K`; there is no TCQ family or selector default
  ([`gridbook_runtime_contract.0.8.11.json:58-153`](../../prismaquant/gridbook_runtime/gridbook_runtime_contract.0.8.11.json#L58-L153)).

Therefore the default with nothing set is **no trellis candidate and no trellis
artifact**, not `lloyd`.

For contrast only, the out-of-tree research driver at
`/home/rob/dq-runs/trellis-hull-20260828/fp8_ladder.py:294-373` reads
`E4M3_ALPHABET` with default `lloyd`. Its complete selector branch is:

1. `lloyd` -> `optimize_e4m3_alphabet_hierarchy`.
2. `exact_dp` -> take a deterministic 262,144-element sample and call
   `e4m3_alphabet_dp.exact_alphabet` on CUDA.
3. `canonical` -> read `E4M3_CANONICAL_RULE` (default `gaussian`) and build a
   data-independent hierarchy shape scaled by one measured RMS.
4. Anything else -> `SweepError`.

That environment variable does not occur in PrismaQuant production code. It is
a ladder-study control, not a `PRODUCTION_CACHE_LEVERS` member, recipe field, or
Gridbook default.

The unreleased Gridbook branch does not supply a selector either. At commit
`ab80df3`, its loader parses alphabets already serialized into `wire_bytes`
([`gridbook/trellis_scheme.py:123-152`](https://github.com/RobTand/gridbook/blob/ab80df3107913ebc715845cec9e83a0af208d6a9/gridbook/trellis_scheme.py#L123-L152)).
Its smoke-checkpoint tool says that Gridbook has no weight-to-wire encoder and
that the Viterbi encoder lives in research trees
([`tools/make_trellis_smoke_checkpoint.py:1-15`](https://github.com/RobTand/gridbook/blob/ab80df3107913ebc715845cec9e83a0af208d6a9/tools/make_trellis_smoke_checkpoint.py#L1-L15));
the tool fabricates canonical-prefix alphabets only to make a self-consistent
serving fixture
([`tools/make_trellis_smoke_checkpoint.py:40-77`](https://github.com/RobTand/gridbook/blob/ab80df3107913ebc715845cec9e83a0af208d6a9/tools/make_trellis_smoke_checkpoint.py#L40-L77)).
That branch is neither the PrismaQuant pin nor an exporter.

## 3. E4M3 (W8A8) versus E2M1 (W4A4)

The shipping answer does not differ: both families are recognized by the same
parser and refused before render.

The research generators do differ, which must not be mistaken for a shipping
policy:

| Family | Research alphabet path | Shipping path |
|---|---|---|
| E4M3 / W8A8 | The ladder-only `E4M3_ALPHABET` branch above selects `lloyd`, `exact_dp`, or `canonical`. | No encoder; TCQ assignment refused. |
| E2M1 / W4A4 | `/home/rob/dq-runs/trellis-hull-20260828/bf16_w4a4.py:143-158` calls `W.alphabets`; `/home/rob/dq-runs/trellis-stage0/stage6_worker.py:264-286` calls `stage3_mixed_rate.build_alphabets`. That function uses an exhaustive weighted-SSE subset search for R1/R2 and the fixed full 16-code E2M1 alphabet for R3 (`stage3_mixed_rate.py:128-136`; exhaustive solver at `tcq_pilot.py:192-228`). `E4M3_ALPHABET` does not reach this lane. | No encoder; TCQ assignment refused. |

Thus `lloyd` versus E4M3 `exact_dp` is not an E2M1 production-default
question. The E2M1 research path has its own small-grid exhaustive rule, but it
also has no route to shipped bytes.

## 4. Reachability verdict

`exact_dp` is reachable only from the out-of-tree research ladder. It is not
reachable from `run-pipeline.sh`, `ProductionWeightCache`, any of the three
shipping exporters, or the pinned Gridbook runtime.

Plainly: **this does not mean PrismaQuant ships the weaker Lloyd trellis. It
means PrismaQuant ships no trellis.** If a future producer deliberately chooses
Lloyd, the completed ladder says the 4.0/5.0 cells favor fp8-CB under that
conditional. Current code makes no such choice, so those two cells remain
not established.

## 5. Render-time cost

Making `exact_dp` a production default is not a flag flip. Before a selector
default can exist, the missing allocator aggregation/registry links must be
wired, the existing production cache must gain a trellis render, an exporter
must emit a self-describing wire, and the pinned Gridbook contract must admit
and attest the TCQ lane. Only then can a versioned selector field and default be
meaningful.

The existing ladder timing data answers a narrower question, with an important
timer boundary. In `bf16_ladder.py`, `e4m3_alphabets(...)` runs before the
per-rung timed call (`:179-195`); `encode_seconds` stores only the returned
per-rung Viterbi/trellis-arm interval (`:150-168`). **The JSON field therefore
excludes both Lloyd construction and exact-DP construction.** The exact-DP
branch's fixed 262,144-element sample and solver call are at
`fp8_ladder.py:325-334`.

From `cells[*].arms[*].encode_seconds` in
`bf16_ladder_lloyd.json` and `bf16_ladder_exact_dp.json`:

| Timed arms | Corpus | Lloyd sum | exact-DP sum | Lloyd median/arm | exact-DP median/arm |
|---|---:|---:|---:|---:|---:|
| `tcq_e4m3@4.0` | 24 tensors / 597,688,320 weights | 190.580 s | 103.278 s | 6.777 s | 1.717 s |
| `tcq_e4m3@5.0` | same | 173.272 s | 109.010 s | 5.483 s | 1.814 s |
| all R1-R7 trellis arms | 168 arms over the same tensors | 1,392.876 s | 751.322 s | 7.292 s | 2.162 s |

Those are the recorded values, not a claim that exact DP makes encoding faster.
Both JSONs stamp `contention_note = "shares the box with another sweep;
encode_seconds is not a timing claim"`, and the selector itself is outside the
timer. The exact-DP file's timed Viterbi work appears lower, consistent with a
different contention interval; 47 of 168 paired arms were instead slower. The
data establishes no selector-caused encode-time penalty or speedup.

A purely mechanical parameter-linear projection of the **timed Viterbi portion**
from 0.59768832B to 27B weights is 2.391 h versus 1.296 h at R4 and 2.174 h
versus 1.368 h at R5 (Lloyd versus exact-DP run). It is included only to expose
the scale of the recorded arm work. It is not a production forecast: the runs
were contended, a real assignment mixes rungs, and—most importantly—it omits
the selector whose cost is being asked about.

The selector's 27B cost is therefore **not measurable from
`arms[*].encode_seconds`**. Since the exact-DP driver samples a fixed 262,144
values per tensor, scaling its missing selector time by model parameter count
would also be the wrong extrapolation; it depends primarily on the number of
selected tensors and their sampling/launch overhead. Establishing that cost
would require a separate timer around `e4m3_alphabets`, which the completed
artifacts do not contain. No new GPU work was run for this analysis.

The per-rung aggregates were computed read-only with this query (run once per
JSON and requested arm; replace the equality filter with
`startswith("tcq_e4m3@")` for the all-rung total):

```bash
jq --arg arm 'tcq_e4m3@4.0' '
  def median:
    sort as $s | length as $n |
    if ($n % 2) == 1 then $s[($n / 2 | floor)]
    else (($s[$n / 2 - 1] + $s[$n / 2]) / 2) end;
  [.cells[].arms | to_entries[]
   | select(.key == $arm) | .value.encode_seconds]
  | {n: length, sum_seconds: add, median_seconds: median}
' /home/rob/dq-runs/trellis-bf16-20260829/bf16_ladder_lloyd.json
```

No file under `/home/rob/dq-runs/` was modified.
