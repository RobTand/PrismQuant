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

## 7. Deliberately not on this path

The Stage-1 numeric campaign, the `pqwork` reserve-set v3 durability work, and
the two-host NFS capability harness guard a **preregistered scientific
receipt**, not the artifact path. Nothing in §5 depends on them. They stay
parked until the first artifact exists.
