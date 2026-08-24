# NVFP4-CB / FP8-CB — On-disk Layout & Container Contract

**This document is the complete, self-contained contract the serving plugin
consumes.** A plugin author needs nothing else: it fully specifies the byte
layout of the packed weight stream, the safetensors tensor names, the
`quant_config.json` schema, and the codebook storage. It is pinned bit-for-bit
by `tests/test_nvfp4_cb_formats.py` (the pack→unpack→reconstruct contract) and
produced by `prismaquant/export_nvfp4_cb.py` and its streaming counterpart.

Producers of these bytes:
`prismaquant.nvfp4_cb_formats.nvfp4_cb_assemble_bytes`; the inverse the plugin
must implement is mirrored exactly by `nvfp4_cb_unpack`.

---

## 0. Format family at a glance

A codeword is a **d = 8** vector of grid values. A **k-bit index per 8 weights**
selects it. Two grids and one production index-encoding mode:

| family | grid | codeword values | act | scale plane | bpw (body) |
|---|---|---|---|---|---|
| `NVFP4_CB_K{k}` | fp4 / E2M1 | `{0,±.5,±1,±1.5,±2,±3,±4,±6}` | W4A4 | group-16 E4M3, **in the weight bytes** | production v2: `k/8 + 0.28125` |
| `FP8_CB_K{k}`   | fp8 / E4M3 | E4M3 grid (‖·‖ ≤ 448) | W8A8 | **none in weight bytes** — per-output-channel fp32, separate tensor | `k/8` |

`k` has separate producer and reader domains. New FP8-CB artifacts may use
exactly `FP8_CB_K4,K8,...,K48`; this K%4 rule is the format/TMA producer law.
Readers retain every historical `FP8_CB_K28..K48` wire id, including off-law
rungs such as K29 and K47, so old artifacts remain inspectable and loadable.
Reader acceptance is not menu authority: cost, allocation, learned-v2 bundle
construction, and export use the producer-only registry API and reject those
off-law ids. NVFP4-CB reader and producer domains are both exactly
`NVFP4_CB_K1..K25`: K1 is the smallest nonempty wire stream and K25 splits
`(13,12)`. K26..K32 have no public wire id. The direct codec retains a
research-only K32 uint32 boundary test, which is not reader or producer
authority.
A decoded fp4 tile is bit-compatible NVFP4 (E2M1 codes + NVFP4 group-16 E4M3
scale) and feeds the existing CUTLASS FP4 path unchanged.

Learned codebooks do **not** introduce `CBL_*` format names. The generic
sidecar refs, shapes, digests, and byte accountant can represent a learned
table for either product family without changing the wire format. Production
policy is narrower: NVFP4-CB defaults to lattice and the immutable bundle
builder/loader refuses learned NVFP4 cells without a new promotion receipt;
learned-v2 construction remains restricted to FP8 K4..K48 step 4 and records
its per-rung learned-versus-lattice decision. See `CBL_RUNG_POLICY`,
`require_cbl_rung_enabled`, and `prismaquant.cb_learned_promotion`.

The strict `qwen38_rtx4090_fp8_cb` producer narrows this generic wire contract
further: it currently permits canonical lattice references only, pending a
strict artifact attestation for raw learned-v2 results, and sets
`CB_ACTIVATION_SCOPE=none`. Thus its FP8-CB weights still execute dynamic W8A8
E4M3, but no NVFP4 static-activation scalar/metadata or FP4 codebook family may
appear in its config, sidecar, tensor census, or delegated terminals.

The separate `qwen38_sm120_cb_validation_only` profile closes candidate
registration to both public producer ladders plus native
NVFP4/FP8_E4M3/BF16 terminals and remains lattice-only. This changes no byte
layout. It has no producer policy and carries no release or device-qualified
claim: candidate Gridbook v11 is compile-only and unpinned, so an immutable
consumer release and exact device-qualified routes remain mandatory before
these structurally legal bytes can become a shippable SM120 artifact.
It also explicitly denies the shared registry's W8A16/source-FP8 compatibility
set. Those legacy wire/assignment readers remain available for published
source-model artifacts, but no such format may enter this profile's new RTX50
cost menu, assignment, or release decision. Both materializers read the
allocator's stamped target profile during outer preflight and refuse any denied
format before creating the destination transaction or a `.tmp-*` sibling.

**Hard constraint:** `in_features % 256 == 0` (the 256-weight superblock is both
byte-exact and the vector-tiling unit). Linears that fail it are shipped BF16.

---

## 1. Superblock byte layout (the weight stream)

The quant unit is a **256-weight superblock along the input dim**. Per row
(output channel) there are `in_features / 256` superblocks laid out
contiguously; the packed tensor is 2-D uint8 `(rows, (in_features/256) *
type_size)`.

Per superblock:

```
┌──────────────────────────────┬─────────────────────────┐
│ INDEX STREAM  (4k bytes)      │ SCALE PLANE (fp4)         │   type_size bytes
│ 32 k-bit codewords, LSB-first │ v2: 9 B; legacy v1: 16 B │   = 4k + 9 (fp4 v2)
│                               │ (fp8: absent)            │   = 4k      (fp8)
└──────────────────────────────┴─────────────────────────┘
```

- Production `type_size = 4k + 9` (fp4 layout-v2) / `4k` (fp8) bytes;
  explicit legacy-v1 fp4 is `4k + 16`. All are **integer for every k**.
- 32 codewords = 256 weights / 8 (VEC_DIM); 16 scales = 256 / 16 (FP4_GROUP).
- **fp8 has no per-superblock scale plane.** Its per-output-channel fp32 scales
  ship as a separate `<name>.weight_scale` tensor (§3).

### 1.1 Index stream — bit packing (LSB-first)

The 32 codewords of a superblock are concatenated into one bitstream, **LSB
first**, then emitted 8 bits per byte (bit 0 of the stream is bit 0 of byte 0):

```
stream bit index:   0            k           2k                   32k-1
                    ┌── cw0 ──┐  ┌── cw1 ──┐  ┌── cw2 ──┐   ...   ┌ cw31 ┐
                    │ b0…b(k-1)│  │ b0…b(k-1)│                     │      │
byte b = Σ_{j<8} stream_bit[8b+j] << j        (4k bytes total)
```

Codeword `c` occupies stream bits `[c·k, c·k + k)`, its own **LSB first**.
Because 32·k = 4k·8, superblock boundaries fall on byte boundaries; you may
unpack the whole row's index region as one contiguous stream.

Each **k-bit codeword** encodes one 8-dim vector. Its internal layout depends on
the mode:

**`full` mode** — the codeword *is* the codebook index (`0 ≤ idx < 2^k`):
```
 bit:  k-1 ................. 0
       [        idx         ]
```

**`product` mode** — the index is split into `n_sub` sub-indices packed
contiguously, **sub-index 0 in the low bits**. `n_sub = 2` (fp4, two 4-dim
halves) or `4` (fp8, four 2-dim quarters). Sub-index widths are
`bit_split(k, n_sub)` = as even as possible, **larger halves first**
(k=13,n=2 → (7,6); k=40,n=4 → (10,10,10,10); k=36,n=4 → (9,9,9,9)):
```
 fp4 (n_sub=2, widths b0≥b1, b0+b1=k):
 bit:  k-1 ........ b0 | b0-1 ...... 0
       [    sub1     ] | [   sub0    ]

 fp8 (n_sub=4, widths b0…b3):    high ─────────────────────► low
       [ sub3 ][ sub2 ][ sub1 ][ sub0 ]
```
Sub-index `i` decodes 8/n_sub coords via sub-codebook `i`; the 8-dim codeword is
the concatenation `[sub0 | sub1 | …]`.

At the public NVFP4 endpoints, `bit_split(1,2) = (1,0)`: sub1 has shape
`(1,4)` and its index is the empty/zero bit field. `bit_split(25,2) = (13,12)`.
The direct research codec also pins `bit_split(32,2) = (16,16)` and must split
and mask those two fields independently rather than computing `1u << 32`; no
serialized artifact may name that research endpoint.

### 1.2 Scale section (fp4 only) — two codings

**Scale coding v1 (e4m3-direct, 16 B — legacy read/write compatibility;
absence of the scheme's `scale_coding` key still means v1):** immediately
after the 4k index bytes, **16 E4M3
bytes**, one group-16 block scale per 16 consecutive weights (group `g` covers
weights `[16g, 16g+16)` of the superblock). Byte = the `torch.float8_e4m3fn`
value reinterpreted as uint8 (`scale.to(float8_e4m3fn).view(uint8)`). This is
**byte-identical to NVFP4's block-scale plane** — hand it to the block-scaled
MMA unchanged. Reconstruction:
`weight[i] = codeword_value[i] × e4m3_scale[group(i)]`.

**Scale coding v2 (two-tier, 9 B — production writer default;
`layout_version: 2`,
`docs/lanes/nvfp4-cb/two-tier-scale-spec.md`):** immediately after the 4k index
bytes:

```
[ SUPER 1 B (E8M0, bias 127) | SUB 8 B (16 × 4-bit codes) ]
```

- `SUPER` = uint8 `E`; the superblock's power-of-two super-scale `2^(E-127)`.
- `SUB` = 16 4-bit codes, group `g` in byte `g/2`, **even `g` = low nibble**
  (LSB-first, consistent with the index stream). Code `c_g` indexes the fixed
  16-entry multiplier table `T` shipped in the scheme
  (`scale_coding.table`; default `T4_2oct8m = {1.0, 1.125, …, 1.875, 2.0,
  2.25, …, 3.75}` — all 8 e4m3 mantissa steps × 2 octaves).
- **Reconstruction:** `scale_g = T[c_g] × 2^(E-127)` — exact E4M3 **by
  construction** (every table entry is `(8+j)/8 × 2^i`; the encoder only emits
  `(E, c)` pairs whose composition round-trips `float8_e4m3fn` bit-exactly and
  lies in `(0, 448]`), so the consumer still sees a bona-fide E4M3 plane with
  a plain fp32 multiply — no cast, no rounding.
- The packer asserts type_size-vs-version consistency, so a mis-labeled
  artifact fails loudly at load, not silently.

### 1.3 type_size table (asserted by the packer)

| grid | k | type_size v1 (B/256) | **type_size v2** | index bits (32k) | scale bytes v1 / v2 |
|---|---|---|---|---|---|
| fp4 | 1  | 20  | **13**  | 32  | 16 / 9 |
| fp4 | 12 | 64  | **57**  | 384 | 16 / 9 |
| fp4 | 13 | 68  | **61**  | 416 | 16 / 9 |
| fp4 | 14 | 72  | **65**  | 448 | 16 / 9 |
| fp4 | 16 | 80  | **73**  | 512 | 16 / 9 |
| fp4 | 18 | 88  | **81**  | 576 | 16 / 9 |
| fp4 | 20 | 96  | **89**  | 640 | 16 / 9 |
| fp4 | 24 | 112 | **105** | 768 | 16 / 9 |
| fp4 | 32 | 144 | **137** | 1024 | 16 / 9 |
| fp8 | 36 | 144 | —       | 1152 | 0 |
| fp8 | 40 | 160 | —       | 1280 | 0 |
| fp8 | 44 | 176 | —       | 1408 | 0 |

`effective_bits(fp4, v1) = (4k+16)·8/256 = k/8 + 0.5`;
`effective_bits(fp4, v2) = (4k+9)·8/256 = k/8 + 0.28125` (version-keyed —
`nvfp4_cb_formats.nvfp4_cb_effective_bits`). Registered `FormatSpec` values
remain a legacy nominal description for compatibility and must not price a
produced artifact; allocator/export/footprint paths use the versioned
`nvfp4_cb_footprint` payload API with an explicit serialization context.
`effective_bits(fp8 body) = 4k·8/256 = k/8` (+ the per-channel fp32 plane).
Two-tier is fp4-only (fp8 has no per-superblock scale plane).

---

## 2. Worked example (tiny tensor)

`NVFP4_CB_K12`, `product` mode, one row, `in_features = 256` (one superblock),
`n_sub = 2`, `bit_split(12,2) = (6,6)`, so each codeword = `sub0(6 bits) |
sub1(6 bits) << 6`.

Suppose vector 0 picks sub-indices `(sub0, sub1) = (5, 3)`:
codeword `c0 = 5 | (3 << 6) = 5 + 192 = 197 = 0b0011000101` (12 bits).

Stream bits `0..11` = LSB-first bits of 197 = `1,0,1,0,0,0,1,1,0,0,...`.
- byte 0 = bits[0..7] of the stream = `1+0+4+0+0+0+64+128 = 197`… but bits 8..11
  belong to `c0` and bits beyond come from `c1`, so byte 1's low 4 bits finish
  `c0` and its high 4 bits start `c1`. (Codewords are **not** byte-aligned; only
  the 4k-byte superblock is.)

Index region = `4·12 = 48` bytes (32 codewords × 12 bits = 384 bits). Production
v2 then appends its 9-byte scale plane → `type_size = 57` bytes for the single
superblock → `cb_qweight` shape `(1, 57)`. An explicitly requested legacy-v1
artifact appends 16 bytes instead and has shape `(1, 64)`.

To decode vector 0: read its 12 stream bits → `197`; `sub0 = 197 & 63 = 5`,
`sub1 = (197 >> 6) & 63 = 3`; codeword = `[sub_cb0[5] (4 coords) | sub_cb1[3] (4
coords)]`; multiply coords by their group-16 E4M3 scale.

---

## 3. safetensors tensor names

**Containers (2026-08-21).** CB weights are published in the standard HF shard
layout, at the repo-wide ~1 GiB budget (`--shard-bytes`, default
`prismaquant.shard_layout.DEFAULT_SHARD_BYTES`; `EXPORT_SHARD_BYTES` on the
pipeline). One resulting shard is `model.safetensors` with no index; more than
one is `model-XXXXX-of-YYYYY.safetensors` plus `model.safetensors.index.json`.
A budget at least as large as the artifact reproduces the pre-2026-08-21
single-container layout — there is no zero sentinel, in this lane or the
compressed-tensors one. The `.pqcb` codebook sidecar is **never** sharded and
never carries an index: it sits outside the `*.safetensors` glob on purpose
(§codebook contract), and its size is bounded by the codebook tables.

For each **CB target Linear** `<q>` (e.g. `model.layers.0.mlp.gate_proj`):

| tensor | dtype / shape | families | meaning |
|---|---|---|---|
| `<q>.cb_qweight` | uint8 `(rows, (in/256)·type_size)` | all | §1 superblock byte stream |
| `<q>.weight_scale` | fp32 `(rows,)` | **fp8 only** | per-output-channel scale (fp4 scales live inside `cb_qweight`) |

**Stacked packed experts** (a 3-D source weight `(E, out, in)`, e.g. Qwen3-MoE
`experts.gate_up_proj` / `experts.down_proj`): the expert axis stays explicit —
`<q>.cb_qweight` is uint8 `(E, out, (in/256)·type_size)` (each expert's rows
laid out exactly as the 2-D case; expert `e` = `cb_qweight[e]`), and the fp8
`<q>.weight_scale` is fp32 `(E, out)`. Encoding uses per-expert `col_weights`
`(E, 1, in)` when provided (a single `(in,)` vector is broadcast to all
experts). A uniform stack shares one format; the existing per-expert-format
declaration may instead partition its experts into physical rung sub-stacks.
From released Gridbook 0.8.5 on (the pin is 0.8.11), learned FP8-CB stacks may
carry one pooled book per logical gate/up/down role: `gate_up_proj` stays one physical
tensor, while ordinary logical `gate_proj` and `up_proj` config targets carry
distinct refs and select row-offset blocks. Split logical targets retain their
`format_group_*` discriminator and reuse the matching unsuffixed role/rung
bundle cell. **Served** by
`PrismaQuantCBMoEMethod` ([external Gridbook `moe.py`](https://github.com/RobTand/gridbook/blob/master/gridbook/moe.py)), which registers
w13/w2 buffers at these exact shapes so loading is a plain `copy_`; archs that
map experts at the top level additionally need a loader line in
Gridbook's packaged runtime contract and loader table (see `moe_cb_design.md` §4).
The producer version-gates this path: pins older than 0.8.4, malformed pins,
non-final versions, or a missing packaged ABI marker retain the routed/rank-3
refusal. The released capability does not waive artifact-specific device
validation. There is no lattice fallback under a learned identity;
the exact ABI and ship gate are recorded in `MOE_LEARNED_CODEBOOK_SPEC.md`.

**Codebooks — shipped once per `(ref, format)`, never per tensor:**

| tensor | dtype / shape | meaning |
|---|---|---|
| `cb_codebook.<ref>.<fmt>` | fp16 `(2^K, 8)` | research `full` codebook (`K = k`) |
| `cb_codebook.<ref>.<fmt>.sub{i}` | fp16 `(2^b_i, 8/n_sub)` | `product` sub-codebook `i` |

`<ref>` is `lattice` (the deterministic fixed lattice, shipped once per format)
or the exact dense target qname for a learned cell (for example
`model.layers.0.mlp.gate_proj`). Because `<fmt>` includes the rung, this gives
one distinct physical reference set per `(layer, role, rung)` even if two cells'
current values happen to match. `<fmt>` is the rung name (`NVFP4_CB_K16`, …).
Codebook values are grid-valued and **exact in fp16** for both grids; the plugin
may re-pack them to 4-bit (fp4) / 8-bit
(fp8) codes in `process_weights_after_loading` (a load-time transform of a tiny
table — not a resident weight expansion, so INV-1 is unaffected).

The public FP4 product ladder needs d4 widths 0..13. The digest-pinned
`nvfp4_cb_lattices.pt` additionally materializes research widths 14..16, and a
missing public production key refuses runtime synthesis.
`fp4-d4-nested-e2m1-v3` constructs widths 0..5
as nested progressive-farthest subsets of the immutable width-6 table, and
widths 13..16 as nested prefixes rooted in the exact width-12 table. Existing
width-6..12 tensor bytes are pinned unchanged. The high master appends a
deterministic well-distributed permutation of every missing E2M1 d4 vector,
then deterministic duplicates only after all numeric vectors are present.
Thus widening either constructed table cannot worsen nearest-codeword
distortion solely through a non-nested table change. E2M1 has 15 distinct
numeric values. A direct research K32 pair therefore has two `(65536,4)`
physical tables (1,048,576 FP16 bytes total), while each width-16 table has
50,625 distinct numeric rows; neither geometry creates a public K32 format.

### 3.1 Learned bundle, family scopes, and sidecar identity

Learned codebooks are selected per family by
`CB_CODEBOOK_SOURCE_SCOPE=none|fp8|all`:

| value | NVFP4-CB | FP8-CB | contract |
|---|---|---|---|
| `none` (default) | lattice | lattice | historical producer; the scope member is omitted from identity so the serialized stamp and rendered bytes stay byte-identical to baseline `76666bd` |
| `fp8` | lattice | learned | production CBL arm |
| `all` | learned | learned | research-only warning; the production bundle builder refuses learned NVFP4-CB because it is measured NO-GO |

The legacy `CB_CODEBOOK_SOURCE` scalar remains on the wire as an artifact-wide
ANY: it is `lattice` for `none` and `learned` otherwise. The mixed `fp8` value
is additionally stamped as `codebook_source_scope: "fp8"` and an explicit
`codebook_source_by_format` map; homogeneous `none`/`all` needs no new scope key
because the old scalar already identifies it unambiguously
(`prismaquant/nvfp4_cb_footprint.py:489-508,568-632`).

Scale search is independent and selected by
`CB_SCALE_SWEEP_SCOPE=none|nvfp4|fp8|all`. If it is unset, the legacy
`CB_SCALE_SWEEP` boolean still means none/all. A mixed one-shot-FP8 arm uses
`nvfp4`; the measured sweep-matched CBL arm uses `all` (or unset with
`CB_SCALE_SWEEP=1`). The production two-tier FP4 layout requires NVFP4 scale
search, so `none` or `fp8` is invalid when that layout is present. Only the
genuinely mixed values `nvfp4`/`fp8` add `scale_sweep_scope` to the stamp
(`nvfp4_cb_footprint.py:205-216,265-274,525-547,615-632`).

Any learned scope requires `CB_CODEBOOK_BUNDLE`, one immutable safetensors
`.pqcb` with schema `prismaquant.cb_learned_codebook_bundle.v1`. It is created
before cost/cache/KL, never during export. The bundle contains:

- the exact canonical FP16 lattice and learned subtables needed by its cells;
- one learned reference set per dense `(qname, format)` — equivalently
  `(layer, role, rung)`;
- the certified `learn_pool` trainer identity and the source-weight/imatrix
  value identity for every cell;
- for production banked expert cells, the exact selection/burn/book origin and
  payload digests, tied to the role input identity again on bundle reload;
- the measurement-backed rung-policy table; and
- `codebook_content_sha256`, one SHA-256 per contiguous little-endian FP16
  subtable payload, covering the bundle's tensor-name set exactly.

Training atomically publishes the bundle and refuses to overwrite an existing
path. Loading revalidates schema, trainer, names, shapes, cell ownership, full
reference coverage, and every digest; lookup of an absent or mismatched cell
raises instead of substituting a lattice book (see `train_and_save_bundle`,
`load_bundle`, and `CBLearnedBundle.codebook_for` in
`prismaquant.cb_learned_bundle`).

The exporter uses those exact selected tensors and writes them once to the
artifact's `cb_codebooks.pqcb`. Its `provenance.codebook_sha256` is a **complete
name map over that final sidecar**, including canonical lattice tensors and
learned tensors. This is stricter than a learned-only manifest because pinned
Gridbook compares the complete expected and observed name sets before checking
each value digest (`build_quant_config` in `prismaquant.cb_export_config`;
external Gridbook `gridbook/cb_digest.py`). Export-time retraining and silent lattice
fallback are both contract violations.

Learned FP8 eligibility is enforced from the data table
`CBL_RUNG_POLICY`, not inferred from subtable size:

| rungs | state | provenance |
|---|---|---|
| K28–K43 | enabled for production | K28/K33/K38/K43 are directly measured GO; each intermediate row says explicitly that it is admitted by the measured K43 boundary, not by a fabricated per-rung result |
| K44 | enabled for production | sweep-matched CBL/lattice holdout act-MSE ratio 0.6057 (`dq-runs/dsv4-quality-hybrid/sfd-analysis/cbl_k43_k47.log:31`) |
| K45 | enabled for production | sweep-matched ratio 0.6929 (`cbl_k43_k47.log:40`) |
| K46 | enabled for production | sweep-matched ratio 0.8312 (`cbl_k43_k47.log:51`) |
| K47 | rejected | sweep-matched ratio 1.0689 (`cbl_k43_k47.log:60`) |
| K48 | rejected | measured 54–98% worse than lattice |

Both bundle creation and bundle load call `require_cbl_rung_enabled`, so a stale
bundle cannot bypass a policy change (see `train_and_save_bundle` and
`load_bundle` in `prismaquant.cb_learned_bundle`).

### 3.2 Authoritative serialized-payload accounting

`prismaquant.nvfp4_cb_footprint` is the producer byte contract. Its persisted
production schema is `prismaquant.cb_serialized_payload.v3` (v4 only when
min-chain identity is enabled); v1 is legacy and rejected by current strict
producer rehydration. Exact calls require a `CBSerializationContext` carrying
scale coding/layout, family-scoped source and scale-search choices, physical
sidecar refs by `(qname, format)`, and materialized content digests for every
learned table. Omitting that context on producer paths is an error rather than
an implicit legacy estimate (`prismaquant/nvfp4_cb_footprint.py:56-89,118-152,
685-718`).

This contract is exact for **tensor data spans**, not for the byte size of the
finished export directory. Safetensors' 8-byte prefixes and JSON headers,
container metadata, `config.json`, `quant_config.json`, tokenizer assets, and
other copied files are not additive candidate costs. After either exporter has
written every file it parses the final safetensors headers, re-asserts the CB
data spans, and persists a second exact scope under
`provenance.artifact_inventory`: per-file sizes plus the total directory,
container, tensor-data, container-overhead, and non-container byte counts.

For `rows = product(shape[:-1])` and `n_sb = in_features / 256`:

- fp4-CB tensor payload = `rows · n_sb · (4k + 9)` bytes for production v2,
  or `rows · n_sb · (4k + 16)` for explicit legacy v1;
- fp8-CB tensor payload = `rows · n_sb · 4k + rows · 4` bytes (the second term
  is the separate fp32 output-row scale tensor);
- CB global-scale bytes are always zero; no such tensor exists;
- each product-codebook sidecar is the sum of its FP16 subtable payloads,
  `Σ_i 2^b_i · (8/n_sub) · 2` bytes. For NVFP4 this is
  `8·(2^ceil(k/2)+2^floor(k/2))`: 24 bytes at K1 and 98,304 bytes at the
  public K25 endpoint. The direct research K32 geometry is not accepted by
  this artifact accountant.

Assignment accounting deduplicates the sidecar by its full serialized identity:
format, source/sharing policy, physical refs, dtype, and subtable shapes. Both
exporters assert their actual/planned tensor bytes and FP16 sidecar tensors
against this breakdown, then persist a compact copy under
`provenance.serialized_payload`. The allocator also stamps scale coding,
layout, and sharing policy into `__prismaquant__.cb_serialized_payload`; an
export request that disagrees with that recipe fails before writing weights.

The additive allocator candidate cost includes the exact per-tensor payload but
not a globally shared sidecar fixed charge. Whole tensor-payload/fit-the-card
pricing adds each sidecar once. Enforcing that non-additive fixed charge
*inside* the knapsack would require a solver with shared binary activation
variables; it must not be approximated by charging every candidate a copy.

All **non-target tensors** (norms, embeddings, lm_head, BF16-assigned Linears)
are copied **verbatim** (bf16 passthrough). Their module names appear in the
config `ignore` list.

---

## 4. `quant_config.json` schema

Custom, compressed-tensors-**style** (its scheme vocabulary cannot express
codebooks — this is a distinct `quant_method`). Also mirrored into
`config.json["quantization_config"]` as a pointer so the loader auto-detects it.

```jsonc
{
  "quant_method": "gridbook",
  "format": "nvfp4_cb",
  "config_groups": {
    "group_0": {
      "targets": ["model.layers.0.mlp.gate_proj"],
      "format": "FP8_CB_K38",
      "scheme": {
        "grid": "fp8",            // "fp4" | "fp8"
        "mode": "product",
        "k": 38,
        "superblock": 256,
        "group_size": 0,          // fp4 group-16 scale; 0 for fp8
        "vec_dim": 8,
        "n_sub": 4,
        "type_size": 152,         // 4*k; no scale bytes in FP8 weight body
        "act_bits": 8,
        "codebook_source": "learned",
        "codebook_ref": [
          "cb_codebook.model.layers.0.mlp.gate_proj.FP8_CB_K38.sub0",
          "cb_codebook.model.layers.0.mlp.gate_proj.FP8_CB_K38.sub1",
          "cb_codebook.model.layers.0.mlp.gate_proj.FP8_CB_K38.sub2",
          "cb_codebook.model.layers.0.mlp.gate_proj.FP8_CB_K38.sub3"
        ],
        "codebook_group": "model.layers.0.mlp.gate_proj"
      }
    }
  },
  // top-level, v2 exports only; absence ⇒ layout v1:
  "layout_version": 2,
  "ignore": ["model.norm", "lm_head", ...],   // non-CB modules -> unquantized
  "provenance": {
    "git_commit": "...",
    "assignment_sha256": "...",
    "imatrix_sha256": "...",
    // Complete final cb_codebooks.pqcb name map: lattice AND learned tensors.
    "codebook_sha256": {
      "cb_codebook.lattice.NVFP4_CB_K16.sub0": "...",
      "cb_codebook.lattice.NVFP4_CB_K16.sub1": "...",
      "cb_codebook.model.layers.0.mlp.gate_proj.FP8_CB_K38.sub0": "..."
      // ... all remaining sidecar names, with no extras or omissions
    },
    "codebook_source": "learned",
    "codebook_source_scope": "fp8",
    "scale_sweep": true,
    // no scale_sweep_scope means the legacy/all-family sweep-matched arm;
    // a mixed one-shot-FP8 arm writes "scale_sweep_scope": "nvfp4".
    "ldlq": false,
    "encode_tier": "balanced",
    "renderer_abi": "prismaquant.nvfp4_cb_renderer.v1",
    "cb_targets": 128,
    "render_identity_verified": true,
    "serialized_payload": {
      "schema": "prismaquant.cb_serialized_payload.v3",
      "context": {"scale_coding": "two_tier", "layout_version": 2,
                  "codebook_source": "learned",
                  "codebook_source_scope": "fp8",
                  "scale_sweep": true, "ldlq": false,
                  "encode_tier": "balanced",
                  "renderer_abi": "prismaquant.nvfp4_cb_renderer.v1",
                  "activation_contract": "prismaquant.nvfp4_w4a4_activation.v1",
                  "activation_execution": "e2m1_group16_ue4m3_static"},
      "tensor_payload_bytes": 123456,
      "codebook_sidecar_bytes": 4096,
      "global_scale_bytes": 0,
      "total_bytes": 127552,
      "n_tensors": 128,
      "sidecars": [/* physical FP16 ref/shape/content-digest identities */]
    },
    "tensor_formats": {"model.layers.0.mlp.gate_proj": "FP8_CB_K38", ...}
  }
}
```

`codebook_ref` is a single tensor name (research `full`) or a list of sub-table
names (`product`, ordered sub0..sub{n_sub-1}). Grouping: targets sharing one
`(codebook_ref, format)` are one config group. Learned dense cells normally
have distinct refs and therefore distinct groups; canonical lattice cells at
one format share refs and may group together.

**Plugin dispatch:** a prefix matching a group's `targets` → the CB method
(decode via that scheme); a prefix in `ignore` → `UnquantizedLinearMethod`;
plain NVFP4/FP8 groups → the pinned runtime's delegated native route. Dense
fusion does **not** require one ref across sibling roles: Gridbook reads
each role's `codebook_ref`, interns distinct reference tuples, concatenates the
blocks, and supplies a per-row `cb_row_offset`, so gate≠up and q≠k≠v are valid
(external Gridbook `gridbook/linear.py:405-437`). Gridbook 0.8.4 carries that
mechanism for routed MoE. PrismaQuant emits three
ordinary logical config targets (`gate_proj`, `up_proj`, `down_proj`) with
independent refs while retaining the two physical `gate_up_proj`/`down_proj`
payloads. A pin without the released feature attestation refuses before those
refs can be emitted.

---

## 5. Decode recipe (reference)

```
for each row, each superblock s:
  idx_bytes = qweight[row, s*type_size : s*type_size + 4k]
  bits      = unpack LSB-first -> 32 codewords of k bits
  for v in 0..31:
    code = codewords[v]
    if full:    cw = codebook[code]
    if product: cw = concat(sub_cb[i][(code >> off_i) & ((1<<b_i)-1)] for i)
    for coord j in 0..7:
      w_idx  = s*256 + v*8 + j
      if fp4 v1: scale = e4m3(qweight[row, s*type_size + 4k + local_group16])
      if fp4 v2: scale = T[sub_code(local_group16)] * 2^(super_e-127)
      if fp8: scale = weight_scale[row]
      weight[row, w_idx] = cw[j] * scale
```

This reproduces the emulation render bit-for-bit; the two are pinned equal by
`test_nvfp4_cb_pack_unpack_matches_emulation`.
For a fused dense module, `codebook` above means the role block selected by that
output row's `cb_row_offset`. The 0.8.4 routed path applies the same rule to
each expert: the first `w13` row segment selects gate, the second selects up,
and `w2` selects down. The offset vector is resident and captured with the
decode route; it is not recomputed or read from host state per token.
