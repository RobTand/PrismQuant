# The first trellis artifact — verified state and critical path

**Status: current as of 2026-08-31.** Every claim below was re-derived from
code, the pinned contract file, or a measurement on this hardware during the
session that wrote it. It supersedes the *blocker list* in
`docs/handovers/prismaquant-handover-2026-08-31.md` §3 and §5, which was
written before the Gridbook 0.9.1 pin landed on this line and therefore
understates how far the serving half had already got.

---

## 0. The correction, in one line

**The Gridbook serving half is done, released, pinned and attested. Every
remaining blocker is producer-side.**

---

## 1. The attestation chain — verified, not inherited

`prismaquant/gridbook_runtime/gridbook_runtime_pin.json` and
`gridbook_serving_runtime_pin.json` both pin Gridbook **0.9.1**, commit
`227420f9821bab7089632ee914f0ba050f82b817`, contract schema
`gridbook.runtime-contract.v12`.

Checked, in order:

| link | evidence |
|---|---|
| the commit exists and is a release ancestor | `227420f` on `origin/master`; `ab80df3` (the lanes) is its ancestor via `30287aa` (contract attestation) and `698eb48` (release 0.9.1) |
| the commit carries the lanes | `gridbook/trellis_e2m1_lane.py`, `trellis_e4m3_lane.py`, `trellis_scheme.py`, `trellis_decode_pool.py`, `csrc/trellis_r256.cu` all present at `227420f` |
| the wheel matches the pin | `dq-runs/gridbook-release-0.9.1/**dist-ci**/gridbook-0.9.1-py3-none-any.whl` → sha256 `cb4d7ad64c5a78d447f427a0aa98790406b6821d02c7f2f5d589d61890abdf9d`, **equal to the pin** |
| the wheel publishes the table | its packaged `gridbook/runtime_contract.json` is contract_version 12, `lane_eligibility.schema == "gridbook.lane-eligibility.v3"`, carrying four trellis cells |

**Landmine:** `dist-final/` in that same release directory is a *later,
non-reproducible rebuild* with sha256 `7141acf9…`. It is content-equivalent but
byte-different and **does not match the pin**. Anything that supplies a serving
wheel must point at `dist-ci/`.

## 2. The four attested cells

All `platform: sm_121`, `structure: dense`, `qualification: device_qualified`,
`route_status: backed_with_serve_flag`.

| cell | family | regime | `rungs_q256` | `activation_contract` | `requires_serve_flags` |
|---|---|---|---|---|---|
| `trellis_e2m1_dense_sm121_decode_scaled_mm_w4a4` | TCQ_E2M1_R256 | decode | `[512]` | `e2m1_group16_ue4m3_static` | `GRIDBOOK_TRELLIS_E2M1=1`, `GRIDBOOK_TRELLIS_E2M1_MODE=resident\|streamed` |
| `trellis_e2m1_dense_sm121_batch_scaled_mm_w4a4` | TCQ_E2M1_R256 | batch | `[512]` | `e2m1_group16_ue4m3_static` | same |
| `trellis_e4m3_dense_sm121_decode_scaled_mm_w8a8` | TCQ_E4M3_R256 | decode | `[1152]` | `fp8_per_token_dynamic` | `GRIDBOOK_TRELLIS_E4M3=1`, `GRIDBOOK_TRELLIS_E4M3_MODE=resident\|streamed` |
| `trellis_e4m3_dense_sm121_batch_scaled_mm_w8a8` | TCQ_E4M3_R256 | batch | `[1152]` | `fp8_per_token_dynamic` | same |

The `formats` table separately lists `candidate_rungs_q256`
`[384, 512, 640, 768, 896]` for E2M1 and `[1152]` for E4M3.

## 3. Three constraints the first artifact must respect

1. **Attested ⊂ candidate.** Only `q256=512` (2.0 body bits/weight) has an E2M1
   serving cell. The other four candidates are *producer* rungs with no route.
   Export gates on the **cell**, never on the candidate list. A candidate with
   no cell is a reported serving gap, not a demotion (principle 1's carve-out).
2. **Fused modules cannot carry trellis.** vLLM merges `q/k/v` and `gate/up`;
   per-role wires cannot be concatenated because each carries its own
   alphabets, rate schedule and row padding. `gridbook/config.py` refuses such
   a target by name. So the trellis units on a Qwen-shaped architecture are the
   **unfused** `o_proj` and `down_proj`; fused siblings take NVFP4/FP8/BF16.
   The first artifact is therefore inherently **mixed** — which is the thesis,
   not a compromise.
3. **No `routed_moe` trellis cell exists.** GLM-5.3-Flash routed experts cannot
   take a trellis rung under this pin. The first artifact is **dense**.

## 4. Measured on this hardware

**Both trellis serving lanes pass on real sm_121.** `test_trellis_e2m1_lane.py`
+ `test_trellis_e4m3_lane.py` + `test_trellis_dispatch.py` at `227420f`:
**46 passed**, resident and streamed, exact against the wire contract, with the
native TCQ R256 CUDA extension JIT-built for capability `(12, 1)`.

*Environment note that cost time:* the extension build needs the `ninja`
**binary** on `PATH`. `ninja` is installed as a Python package in
`prismaquant-cu130` but invoking the venv's interpreter by absolute path does
not put the venv's `bin/` on `PATH`, so `torch.utils.cpp_extension.load` fails
with `Ninja is required to load C++ extensions`, and `cuda_ext` converts that
into a soft `NativeKernelUnavailableError` — a message that points at nvcc and
CUDA_HOME, neither of which is the cause. Put
`/home/rob/dq-runs/venvs/prismaquant-cu130/bin` on `PATH` and set
`PRISMAQUANT_CB_EXT_DIR` to a persistent directory.

## 4-bis. vLLM serves a trellis checkpoint — measured end to end

`gridbook/tools/make_trellis_smoke_checkpoint.py` at the pin, E2M1 `R512`,
2 layers, hidden 256, vocab 248320 (a real tokenizer's, so vLLM can load one),
served on this box with the **wheel-pinned** serving runtime
(`gridbook_serving_runtime.sh` + `GRIDBOOK_SERVING_RUNTIME_WHEEL` pointing at
the dist-ci wheel), `GRIDBOOK_TRELLIS_E2M1=1`,
`GRIDBOOK_TRELLIS_E2M1_MODE=resident`, image
`vllm/vllm-openai:qwen38-flash-next`:

```
[serve] READY after 130s
/v1/models -> {"id":"toy", ...}
/v1/completions -> 8 tokens, finite logprobs
```

So `quantization=gridbook` config parse, safetensors load, trellis lane
dispatch, load-time wire validation and the native TCQ R256 kernel all work
inside vLLM. The generated text is gibberish and that is correct: the smoke
checkpoint's weights are random by construction, and its own docstring says the
reference weight *is* the wire's decoded value. The lane's numerical exactness
is established by the 46 lane tests in §4, not by this serve; what this serve
establishes is **integration**.

Two environment facts this cost:

- **The producer runtime path cannot be used to serve.** `gridbook_runtime.sh`
  attests a *git checkout* and the vLLM images have no `git`
  (`gridbook-runtime: ERROR: git is required to attest a Gridbook checkout`).
  Serving must use `gridbook_serving_runtime.sh`, which binds the wheel and its
  published SHA-256 instead — which is the stronger attestation anyway.
- **The JIT build needs two headers the image lacks.** `trellis_r256.cu`
  includes `ATen/cuda/CUDAContext.h`, which pulls `cusparse.h` and
  `cusolverDn.h`; neither is in the image's `/usr/local/cuda/include`. They
  exist under `nvidia/cu13/include` in site-packages. Putting that directory on
  `CPATH` **does not work** — `crt/host_runtime.h` then also comes from the pip
  package and nvcc fails with *"macro `__cudaLaunch` passed 2 arguments, but
  takes just 1"*. Symlink **only the headers the toolkit include dir is
  missing**, so nvcc's own headers keep priority.

**A gap this exposed, and it is WO-D's:** the serve emitted **no `emit_route`
telemetry at all**. Principle 14's second leg — compare the artifact's priced
activation-contract histogram against the routes the serve actually emitted —
currently has nothing to consume on this lane. A gate with no evidence must
refuse for lack of evidence, not pass by default.

## 5. What is actually missing — all producer-side

`export_native_compressed.py` refuses every trellis rung today. Half of its
refusal is now **stale**: it says *"the producer Gridbook pin publishes no
executed-activation-contract table for TCQ_E2M1/TCQ_E4M3"*, which was true
before this pin and is false at 0.9.1. The other half is the real blocker and
is stated exactly right: **`ProductionWeightCache` renders no trellis wire.**

The producer work, in dependency order:

| piece | where | why it is required |
|---|---|---|
| **render mechanism** | `production_weight_cache.render_production_weight` | the wire must be produced *once*, and the decoded tensor the surrogate prices must be parsed from those same bytes (principle 8) |
| **format registration** | `format_registry.py` + a shippable serving profile | a trellis rung has no `FormatSpec`, so no cost path, menu or exporter can name it |
| **export writer** | `export_nvfp4_cb.py` (the Gridbook container — **not** compressed-tensors) | emits `<target>.wire_bytes`, E2M1's `<target>.trellis_input_global_scale`, and the `config_groups` TRELLIS scheme |
| **route provenance + gate** | `serving_profiles`, `shipcard.py`, `validate_native_export.py` | principle 9 gates per artifact at export; principle 14 leg 2 reconciles priced route against served route |

`gridbook/tools/make_trellis_smoke_checkpoint.py` is the authoritative
reference for the export shape — its own docstring says *"A real exporter
substitutes an encoder here and changes nothing else."*

## 5-bis. Two items that are Gridbook work, not producer work

Neither is on the critical path to the first *measurement*; both are on the
critical path to a first *ship*.

**(i) The trellis lanes emit no route telemetry.** Confirmed twice, from both
ends: a live serve produced no `emit_route` records, and source inspection of
`trellis_e2m1_lane.py` / `trellis_e4m3_lane.py` at the pin returns zero hits for
`emit_route`, while `gridbook/linear.py` and `gridbook/moe.py` call it at every
CB dispatch site. The lanes do set `layer.gridbook_activation_contract`, but
that is a *binding* recorded at `create_weights`, not a per-forward route
record, and it is not the mechanism `validate_native_export` consumes.

So principle 14's serve-side leg has nothing to compare against, and the
correct disposition is to **refuse for lack of evidence** rather than pass by
default — anything else recreates the 2026-08-17 defect, where provenance
nothing consumed was mistaken for a gate.

*The argument that could close this without new telemetry, recorded so it is
argued rather than assumed:* the trellis lanes have **no fallback at all**
(`NativeKernelUnavailableError`: *"There is no Triton or CB-symbol fallback"*),
so lane construction implies the native route in a way CB's dispatch does not.
If that no-fallback property is itself derived from the pinned runtime rather
than asserted here, `gridbook_activation_contract` becomes admissible served
evidence. Until someone makes that derivation explicit and testable, the
refusal stands.

**(ii) No `routed_moe` trellis cell, and the numbers make it decisive.** On
GLM-5.3-Flash (45 layers, 288 routed experts, `first_k_dense_replace=3`, one
shared expert) the routed experts are **98.5%** of the body. What a trellis can
reach today is `o_proj` + three dense `down_proj` + the shared-expert
`down_proj` — **0.4% of 309 B**. A trellis GLM is therefore not possible under
this pin no matter how good the producer becomes.

For contrast, on a dense architecture the whole body is reachable at TP=1, once
the fused rule below is applied.

## 5-ter. Fused modules ARE trellis-eligible — one wire, merged prefix

`gridbook/tools/make_trellis_smoke_checkpoint.py` parks `q/k/v` and `gate/up` in
`ignore`, which reads like a prohibition and is not one. `config.py::
_build_trellis_method` refuses only a fused target **with per-role owners**, and
its message ends: *"Export ONE wire covering the whole merged module and declare
it against this prefix."* `_trellis_scheme_for_prefix` resolves a scheme
declared against the merged name.

So the supported form is: concatenate the siblings along the output-row axis in
`packed_modules_mapping` order, encode **one** wire over the merged
`[sum(out_features), in_features]` matrix, and declare it against
`...qkv_proj` / `...gate_up_proj`. The allocator's union-find serving-unit
promotion already forces one format across a fused group, which is exactly the
precondition this needs.

This distinction is worth 69% of a dense model: on Qwen3-4B the unfused Linears
(`o_proj`, `down_proj`) are only **31.2%** of body parameters.

**The real constraint in this area is TP=1.** `_require_tp1_serving` refuses
every trellis target above one rank, because a sharded trellis needs per-rank
wires rather than a byte range into a shared schedule. Any artifact carrying a
trellis unit records `tp=1` in its serving-lane provenance.

## 5-quater. Encoder throughput — measured, and it is not the risk

`prismaquant/trellis_producer.encode_trellis_one_linear`, E2M1
`body_rate_q256=512`, `layout=fixed_quota_per_256`, `tailbite_candidates=4`,
`determinism_mode=on`, unweighted `col_weights`, random bf16 input, on this
GB10:

| shape | backend | wall | wSNR | wire bpw |
|---|---|---|---|---|
| 512x1024 | eager | 368 ms | 10.64 dB | 2.5093 |
| 512x1024 | triton | 625 ms | 10.64 dB | 2.5093 |
| 4096x4096 | triton | **0.84 s** | 10.65 dB | 2.5010 |

~20 Mparam/s batched, so Qwen3-4B's whole body is on the order of **3 minutes**.
The first artifact does not have to be 0.6B.

Two cautions attached to those rows. **`sb_chunk` is the whole story**: the
value in `tests/test_trellis_producer.py` is `1`, which runs one Viterbi row per
launch; at that setting a 256x512 tensor did not finish in two minutes at 48%
"utilization" and **14.3 W of a ~140 W envelope** — one-tenth loaded, the
launch-overhead signature from 2026-08-28. Batch it. And **triton loses to eager
at small shapes**, so the backend is a per-Linear decision, to be made on a
profile plus power rather than on wall-clock (principle 15).

Note the wire cost: `body_rate_q256=512` is 2.0 body bits and **2.51 bpw on the
wire**. The ~0.51 is the group-16 ue4m3 scale plane plus schedule, alphabets and
row padding. Any comparison against NVFP4's 4.5 uses the 2.51-style number, from
`trellis_footprint`, never the body rate.

## 5-quinquies. What the first artifact costs, and the region it opens

`trellis_tensor_payload_breakdown` over Qwen3-4B's real geometry (36 layers;
merged `qkv_proj` [6144, 2560], `o_proj` [2560, 4096], `gate_up_proj`
[19456, 2560], `down_proj` [2560, 9728]; 3.63 B body params):

| assignment | wire bpw | body |
|---|---|---|
| BF16 | 16 | 6.768 GB |
| uniform trellis E2M1 `q256=512` | **2.5008** | **1.058 GB** |

That is 6.4x smaller than BF16 — and the point worth drawing out is that
**2.5 bpw is a region the production menu cannot reach at all.** NVFP4 is
4.5 bpp per Linear; FP8 is 8; BF16 is 16. There is no assignment of
`{NVFP4, FP8, BF16}` that lands a dense body near 2.5 bpp without pushing
Linears out of the model entirely. Whatever the quality turns out to be, the
trellis is not competing for the same points on the rate axis — it extends the
axis. The honest comparison is therefore trellis-at-2.5 against *nothing we
currently ship*, and against EXL3/QTIP at matched bpw.

**A producer gap this exposed.** Only `q256=512` factors as a uniform rate-2
schedule (`512 = 2 x 256`). The other four candidate rungs — 384, 640, 768,
896 — need a **mixed** per-column schedule, which gridbook builds with
`trellis.build_q256_schedule(family, q256, 256)`. PrismaQuant has no
equivalent: `trellis_formats` exposes `validate_schedule` but no builder, and a
naive uniform schedule is refused (*"fixed-quota block 0 has 256 body bits;
expected 384"*). Since only 512 is serving-attested, this does not block the
first artifact — but any use of the other rungs owes a schedule builder that
agrees with gridbook's **byte for byte**, which is exactly a golden-vector
obligation, not a reimplementation to be eyeballed.

## 6. The A-side, stated honestly

Every trellis quality number in the tree — the whole 4- and 8-bit ladder, the
24-tensor sweeps, the four-family menu — is **weight-only corpus SSE**, which
prices the W\*A16 shape. The attested cells are **W4A4** and **W8A8**. Nothing
in the menu predicts the served quality of a W4A4 trellis Linear, and the AURA
cost is structurally activation-blind.

That is the 2026-08-17 NVFP4_CB defect in its exact original shape — rendering
identity without execution identity — and it is caught here *before* an
artifact exists, which is what principle 8's second clause is for. The first
artifact's job is therefore to **measure** the A-side end to end (real
held-out KL on the served checkpoint), not to predict it.

## 6-bis. Open at the time of writing — do not merge without these

1. **The frozen export-source closure is broken and must be re-frozen.**
   `prismaquant/production_weight_cache.py` is one of the 15 files whose exact
   bytes `dsv4_w8a16_export_handoff._FROZEN_EXPORT_SOURCE_SHA256` pins, so the
   WO-B render mechanism trips
   `test_tracked_frozen_export_sources_match_reviewed_bytes` and
   `test_exact_w8a16_handoff_returns_read_only_receipt`. That gate is doing its
   job. Clearing it is **not** a hash bump: the precedent in that file is a
   written `RE-FROZEN <date> for <commits>` block naming the file, what changed
   in it, and *why the DSv4 W8A16 path is unchanged*. Deferred only because
   WO-A and WO-C may also touch that file; re-stamping before they land would
   simply break again. **The branch is not clean until that block exists.**

2. **The trellis route record cannot distinguish `decode` from `batch`.**
   Found by WO-E1 while adding `emit_route` to the Gridbook lanes: the pinned
   contract publishes separate cells per regime, but each lane makes one
   `torch._scaled_mm` call and only the record's `shape` field carries `M`. So
   a served record cannot fully key the cell it is supposed to reconcile
   against. The telemetry is still worth having -- it turns "no evidence" into
   "evidence at family/contract granularity" -- but the regime half of
   principle 14's second leg stays open, and a gate must not pretend otherwise.

3. **`emit_route` needs a Gridbook release and a producer pin bump** before the
   ship gate can consume it. Until then a trellis artifact can be built and
   measured but not shipped, and that is the correct state rather than a
   blocker to route around.

## 7. Deliberately not on this path

The Stage-1 numeric campaign, the `pqwork` reserve-set v3 durability work, and
the two-host NFS capability harness guard a **preregistered scientific
receipt**, not the artifact path. Nothing in §5 depends on them. They stay
parked until the first artifact exists.
