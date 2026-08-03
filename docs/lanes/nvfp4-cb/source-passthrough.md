# Source passthrough in the nvfp4-cb container

Status: producer side implemented; the `delegated_native_mxfp4` serve route is
pending its cross-repo audit (see [Route status](#route-status)).

## The rule

> Any format the model natively uses belongs on the passthrough menu.

For every serving unit, the allocator's cheapest honest option is to keep the
bytes the checkpoint already has. A source-passthrough rung is that option
made explicit:

* **Δloss is exactly 0, by construction.** Every cost in this pipeline is
  measured against the *dequantized source*. Shipping the source bytes
  unchanged is the identity transform on that reference. There is nothing to
  measure and no measurement branch to take — the candidate is priced `0.0`
  with provenance `cost_source="source_passthrough"`, and no cost run can ever
  produce a more accurate number for it.
* **Bytes are the exact source slice** — the weight tensor plus every
  scale/aux tensor of the unit — pinned against the checkpoint's own
  safetensors header spans before an allocation ships.
* **Legality is a dtype fact, not a policy.** A passthrough format is legal
  exactly where the source already *is* that format. The same rule runs in
  both directions: it is what masks `BF16` on an MXFP4 expert and what masks
  `MXFP4_SOURCE` on a BF16 embedding.

This is a **family**, not a set of special cases.
`allocator_candidates.SOURCE_PASSTHROUGH_CONTRACTS` is the single table; adding
a newly encountered native format is a table entry, not a new code path.

## Census: DeepSeek-V4-Flash-0731

Taken from the safetensors index + shard headers + `config.json` only (no
tensor data read). Grouped by `(weight dtype, scale/aux pattern, rank)`.

| Weight dtype | Aux/scale pattern | Rank | Tensors | GB | Unit classes |
|---|---|---|---|---|---|
| `I8` | `.scale`: `F8_E8M0`, 32-value groups | 2 | 35,328 | 157.437 | routed experts (43 body layers + 3 MTP) |
| `F8_E4M3` | `.scale`: `F8_E8M0`, block 128×128 | 2 | 390 | 6.304 | attn projections, shared experts, indexer, MTP |
| `BF16` | none | 2 | 196 | 2.966 | embed/lm_head, MoE router, compressor, indexer |
| `F32` | none | 2 | 156 | 0.151 | compressor/indexer aux |
| `I64` | none | 2 | 3 | 0.019 | router `tid2eid` index table |
| `BF16` | none | 1 | 249 | 0.001 | norms, biases |
| `F32` | none | 1 | 234 | 0.000 | scalars |

Two findings changed the design:

1. **The routed experts are MXFP4, and the pre-existing manifest scan called
   them fp8.** The weights are nibble-packed E2M1 carried in `I8` (`w1`/`w3`
   stored `[2048, 2048]` for a logical `[2048, 4096]`), with `F8_E8M0` group
   scales. The old classifier keyed on "has a scale sibling" and stamped
   `fp8`, which would have made `FP8_SOURCE` — an 8.002 bpw format — legal on
   a 4.25 bpw unit.
2. **The body is *not* `FP8_SOURCE`.** It is block-FP8 with **one-byte UE8M0
   block exponents** (`config.json` → `quantization_config.scale_fmt ==
   "ue8m0"`), not the FP32 `weight_scale_inv` plane `FP8_SOURCE` models. Same
   element grid, different on-disk contract: 8.00049 bpw vs 8.00195, and
   `FP8_SOURCE`'s exporter widens its scale plane to FP32 on write, which here
   would quadruple the scale bytes and emit a tensor the model's own loader
   does not expect. It gets its own format, `FP8_BLOCK_UE8M0_SOURCE`.

Mapped onto the probe inventory the census is exact and complete:

```
probe inventory: 33,325 Linears
  mxfp4       33,024   routed experts        -> MXFP4_SOURCE
  fp8_ue8m0      301   body Linears          -> FP8_BLOCK_UE8M0_SOURCE
  (none without a source kind)
```

## The formats

| Format | Wire id | Source kind | bpw | Synth. | Route status on sm121 |
|---|---|---|---|---|---|
| `MXFP4_SOURCE` | `mxfp4_e2m1_ue8m0_g32` | `mxfp4` | 4.25 | yes | **backed — requires `--moe-backend marlin`** |
| `FP8_BLOCK_UE8M0_SOURCE` | `fp8_e4m3_ue8m0_block128` | `fp8_ue8m0` | 8.00049 | yes | **blocked (measured)** |
| `FP8_SOURCE` | — | `fp8` | 8.00195 | no | backed (other checkpoints) |
| `BF16` | — | `bf16` | 16 | no | backed |

The same routing record also carries the one **re-quantized** native rung,
which is not a passthrough and has no source kind (it is legal on any):

| Format | Wire id | bpw | Route status on sm121 |
|---|---|---|---|
| `MXFP8_UE8M0_G32` | `mxfp8_e4m3_e8m0_g32` | 8.25 | **unbacked — consumer lane exists but is opt-in (`GRIDBOOK_MXFP8_DENSE=1`), serve-parity bench pending** |

### The route verdicts came out the opposite way round

Both were measured on GB10/sm121, 2026-08-03. The intuition "the released
checkpoint already serves this way, so a route must exist" is **false**:

* **`MXFP4_SOURCE` is native-confirmed** — but only via vLLM's Marlin MoE
  backend. The auto-selected `DeepGEMM_MXFP4` asserts on the SF
  transformation, FlashInfer is gated to capability family 100, and the OAI
  Triton path is hard-excluded on SM12x (0/15 kernels). `--moe-backend marlin`
  is therefore **part of the serving contract, not a tuning hint**, and it
  travels with the artifact.
* **`FP8_BLOCK_UE8M0_SOURCE` is blocked** — every route measured dead:
  `deep_gemm` assert, cutlass `scaled_mm` rejects the block layout, triton
  `KeyError: float8_e8m0fnu`, flashinfer's gate is sm90-exact, marlin-linear
  tops out at sm89.

**The architectural consequence: CB re-encoding of the body is the only way
DSV4-Flash serves on this box at all.** The body's CB rungs are load-bearing,
not merely economical.

A blocked or pending rung still stays **on the menu** — if the DP wants it,
that is the serving gap surfacing in the allocation rather than at deploy
time — but the exporter refuses to ship a selection containing one without
`--allow-route-pending-passthrough`.

### Serving notes for the shipped artifact

    --moe-backend marlin        (MXFP4 experts; the default backend asserts)
    --kv-cache-dtype fp8        (the SM120 MLA backend asserts otherwise)
    VLLM_USE_DEEP_GEMM=0        (the hyper-connections path breaks otherwise)

"Synthesized candidate" means the allocator *manufactures* the cost row rather
than requiring a column in the cost table. No cost run will ever have a column
for a byte-copy contract, and asking the encoder to "measure" a copy would burn
GPU hours reproducing a zero. `FP8_SOURCE` and `BF16` are deliberately not
synthesized: the cost pipeline already emits real rows for them where they are
legal, and synthesizing over an existing row would hide a disagreement.

Because MXFP4 and block-FP8 are fixed-rate, the generic `FormatSpec`
arithmetic reproduces the checkpoint byte for byte, and that identity is
*checked* rather than trusted — `footprint.assignment_artifact_bytes` refuses
an allocation where a passthrough unit's header span and its closed-form byte
count disagree, because the floor subtracts the span while the body would add
the closed form and the artifact budget would silently drift:

```
expert w1  [2048, 4096]  4,194,304 + 262,144 = 4,456,448 B   (4.25 bpw)
attn.wo_a  [8192, 4096] 33,554,432 +   2,048 = 33,556,480 B  (8.00049 bpw)
expert layer (256 experts × 3 projections)   = 3,422,552,064 B
43 expert layers                             = 147,169,738,752 B
```

## Cost provenance

Three values a selected rung's price can carry, and they are distinguishable in
the shipped artifact:

| `cost_source` | Meaning |
|---|---|
| `output_mse` / `predicted_dloss` / `weight_mse` | measured this run |
| `band_interpolated` | fitted by the RD-ladder from measured anchors; the tensor's own law cleared a holdout gate, and a tensor whose law was rejected had its rungs measured instead |
| `source_passthrough` | never measured, because the exporter copies the bytes |
| `bit_exact` | a *measured* lossless re-encode (`weight_mse == 0`) |

`bit_exact` and `source_passthrough` are both free but are not the same claim:
one ran an encoder and found the output identical, the other never ran one.
`cost_entry_is_exact_by_construction` is the union, and it is what pricing and
the P5a branch label key off, so the two cannot drift apart.

## Serving lanes and route status

Source-passthrough rungs are declared as **distinct serving lanes**, so P5b
accounting never files them under a CB contract they do not have. A unit on
`delegated_native_mxfp4` has *no CB activation contract at all* — which is why
the K0.2 attestation scopes itself to CB layers and the artifact declares the
source-passthrough groups explicitly, rather than leaving them looking like a
missing attestation.

Neither lane declares a `fused_mid_m` block. A passthrough rung has no Gridbook
decode prologue to fuse, so an empty backed set is the honest state rather than
a data gap; it resolves with
`rungs_source = lane_declares_no_fused_mid_m_lane`.

### Route status policy

`route_status` is a **measurement**, not design intent: `backed` (measured
serving, possibly with a requirement), `pending` (unaudited), `blocked`
(measured, every known route dead). Anything other than `backed` keeps the
rung on the menu but makes the export fail closed without
`--allow-route-pending-passthrough`; the acknowledgement is recorded in
`quant_config["provenance"]["route_pending_passthrough_acknowledged"]`.

## The declaration: `source_passthrough`

The serving side must be able to tell, per unit and without inference, whether
the bytes are a Gridbook codebook payload or the checkpoint's own. The
consumer-side reader is implemented and shipped, so this is its exact spelling
— a top-level key in `quant_config.json`:

```json
"source_passthrough": {
  "version": 1,
  "units": {
    "model.layers.39.mlp.experts": "mxfp4_e2m1_ue8m0_g32",
    "model.layers.7.self_attn.wq_a": "fp8_e4m3_ue8m0_block128"
  }
}
```

* the format-id enum is **closed**: `mxfp4_e2m1_ue8m0_g32`,
  `fp8_e4m3_ue8m0_block128`, `mxfp8_e4m3_e8m0_g32`. Ids come from two producer
  tables — `PASSTHROUGH_WIRE_FORMAT_IDS` (byte-verbatim) and
  `REQUANT_WIRE_FORMAT_IDS` (re-encoded here) — whose union
  `WIRE_FORMAT_IDS` must stay 1:1, because the consumer resolves a unit by id
  alone (`gridbook.source_passthrough.FORMATS`);
* **what this record actually claims** is *delegated-native routing* — "served
  by a native Gridbook/model-owned route, not by a CB codebook decoder" —
  which is the question the consumer's dispatcher asks. It is NOT a claim that
  every listed unit's bytes are the checkpoint's own; a re-quantized rung is
  listed here too, and a unit omitted here reads to the consumer as CB, the
  one wrong answer that loads. The stronger byte-verbatim claim is per unit in
  its config group's `weights.source_passthrough` flag (`true` for the
  `*_SOURCE` family, `false` for a re-encode). The key keeps its historical
  name because it is a shipped cross-repo contract
  (`gridbook.source_passthrough.SCHEMA_KEY`);
* a unit may be an expert group **or** a dense body Linear — there is no
  expert-only framing and no exhaustiveness claim;
* **absence of the key means a legacy all-CB artifact**, so the key is emitted
  only when at least one unit is passthrough;
* load-time refusal on: unknown `version`, unknown format id, malformed
  `units`, or **a unit claimed by both the CB config groups and
  `source_passthrough`**. The producer asserts all four before writing.

The K0.2 reconciliation survives as a **producer-side invariant** rather than
a wire field: CB-attested routed-expert modules and passthrough units must be
disjoint and together cover the routed-expert set. That is what makes a
passthrough layer's absence from the K0.2 attestation a declaration rather
than a silent gap.

### Tensor naming

Passthrough groups emit the **checkpoint's own spelling**, because the point is
that the native loader consumes them unchanged:

```
layers.<L>.ffn.experts.<E>.{w1,w3,w2}.weight    I8       (nibble-packed E2M1)
layers.<L>.ffn.experts.<E>.{w1,w3,w2}.scale     F8_E8M0  (per-32 group scales)
```

They are **not** stacked into the packed `ffn.experts.{gate_up_proj,down_proj}`
form a CB layer uses, **not** re-encoded, and **not** added to
`quant_config["ignore"]` (an ignore entry declares a tensor unquantized, which
these are not).

## Where the code lives

| Concern | Location |
|---|---|
| Format specs | `prismaquant/format_registry.py` |
| Contract table, source-kind census, candidate synthesis | `prismaquant/allocator_candidates.py` |
| Byte pinning against the checkpoint | `prismaquant/footprint.py` |
| Legality, lanes, route-pending | `prismaquant/serving_profile_specs/nvfp4_cb.json` |
| Emit + declaration | `prismaquant/export_nvfp4_cb_streaming.py`, `prismaquant/cb_export_config.py` |
| K0.2 scoping | `prismaquant/nvfp4_activation_contract.py` |
