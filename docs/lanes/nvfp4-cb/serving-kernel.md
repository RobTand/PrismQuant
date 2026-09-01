# NVFP4-CB — Serving & Kernel Plan (PLAN ONLY)

> **HISTORICAL DESIGN RECORD — DO NOT USE AS CURRENT SERVING INSTRUCTIONS.**
> The runtime now lives in the external Gridbook repository and must be used at
> the exact commit recorded in `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`. The
> canonical quantization key is `gridbook`; any `prismaquant` compatibility
> alias, package name, import path, or CLI instruction below is retained only as
> historical evidence and must not be followed for a current deployment.

Scope: how we serve an NVFP4-CB (vector-quantized codebook over FP4/E2M1 codes,
NVFP4-identical group-16 E4M3 scales) checkpoint in vLLM on GB10/sm_121, mixed
per-Linear with plain NVFP4, FP8, BF16 in one artifact. Not an implementation.

## 0. The one design fact that changes everything vs the retired low-bit lane

The retired custom low-bit integer kernels died two ways (memory `gguf_lowbit_serving_lane.md`, session
`session_2026_04_24_minimax_kernel_attempts.md`): (a) load-time expansion to NVFP4
gave **zero** runtime-memory savings (92.9 GB artifact → 115.7 GiB resident, OOM on
the 121 GiB budget); (b) standalone MMVQ Triton kernels were memory-latency-bound at
a ~6 ms/call floor and were deleted. Codex's VQ verdict (`references/lowbit-kernels/
conference/002-codex-aqlm-vq-addendum.md`) was that AQLM/VQ "fights the hardware"
because the runtime primitive is lookup/decode, not native MX MMA.

NVFP4-CB is the escape hatch Codex-002 did not have: **a decoded CB tile is
bit-identical NVFP4** (same E2M1 codes, same group-16 E4M3 scales). So the runtime
primitive is *not* "decode a learned vector then run BF16 MMA". It is "expand k-bit
index → 8 FP4 codes into a staging buffer, then feed the *existing NVFP4 FP4-MMA
mainloop*". The codebook lookup replaces only the global→shared **load**; the scale
math and the tensor-core path are unchanged from plain NVFP4. Two hard invariants
follow, and every prototype below must be gated on them:

- **INV-1 (no expansion in HBM/resident state):** the resident weight is the k-bit
  index stream + codebook + E4M3 scales. Decoding to nibble-packed NVFP4 must happen
  in smem/registers per tile, never as a materialized `[N,K/2]` uint8 tensor. This is
  literally the trap that OOM-killed the retired low-bit lane. A resident-footprint assertion is a load
  gate, not a nicety.
- **INV-2 (FP4 tensor cores, not BF16 MMA):** the prefill win only exists if decoded
  codes go to the Blackwell FP4 MMA. A Triton kernel that dequants FP4→bf16 and runs
  `tl.dot` (exactly what `prismaquant/kernels/nvfp4_fused.py:250` does today) is the
  *slow* path — fine for correctness/decode, disqualified as the production prefill.

## 1. Kernel design for sm_121

### 1a. PREFILL / batched (M large): fused-expand NVFP4 GEMM

Pipeline per output tile:
1. Fetch the k-bit indices for the K-tile (byte-unaligned; pack as the retired
   3-stream/byte-triplet trick did — memory `session_2026_04_24_3stream_win.md` — so
   loads stay contiguous; a group of indices packs into whole bytes).
2. **Codebook lookup → 8 FP4 codes per index**, written into the smem staging buffer
   in the exact nibble layout the NVFP4 mainloop expects (`_pack_fp4_indices`,
   low/high nibble, `nvfp4_fused.py:32-36`).
3. Load the group-16 E4M3 scales — **unchanged from NVFP4**; the block-scaled MMA
   consumes them as-is. No new scale handling; this is the whole point of matching
   NVFP4's scale envelope.
4. Async-stage (Marlin/`cp.async`-style, double-buffered) so the lookup+expand of
   tile *t+1* overlaps the MMA of tile *t*. The Triton PoC is single-buffer
   (`num_stages=2` at `nvfp4_fused.py:347`); the production kernel needs explicit
   multi-stage on the expand→MMA boundary because the expand is now on the critical
   path where NVFP4 has a plain copy.

**The smem size wall (state honestly — MEASURED on the target box, 2026-07-15
review: GB10 sm_121 has 100 KB smem/SM, 99 KB opt-in per block — the 228 KB
figure is datacenter Blackwell sm_100 and does NOT apply here).** Codebook =
2^k × 8 codes × 0.5 B (FP4) = **2^k × 4 bytes**:
- k=12 → 16 KB — comfortable, full LUT resident + double-buffered staging.
- k=13 → 32 KB — fine; realistic flat-table ceiling for good occupancy.
- k=14 → 64 KB — *marginal*: fits the 99 KB opt-in but leaves ~35 KB for
  staging buffers + scales at 1 CTA/SM; treat as the boundary case, measure.
- k≥15 → 128 KB+ — **does not fit.** k=16/20/24 hopeless as a flat table.

So there are two codebook regimes, and they are different kernels:
- **Small-k (k≤13, k=14 marginal), learned/table:** flat LUT in smem, cooperative
  load at block start. Simple, fast, this is the first fused kernel to build.
- **Large-k (k≥14–15) MUST be structured/computed**, not a stored table: the index is
  decoded *arithmetically* into 8 FP4 codes with no smem table. Options: (i)
  lattice + sign/permutation decomposition (index = base-lattice-point ⊕ sign bits ⊕
  a small shared generator table that *does* fit); (ii) QTIP-style computed codewords
  (index run through a fixed hash/LFSR to synthesize codes on the fly). This is a
  real constraint on the *format design*, not just the kernel: **if the offline
  quantizer wants k>14, it must commit to a computed/structured codebook**, or the
  fused prefill kernel is impossible and we're stuck on the slow dequant path. Feed
  this back to the format/codebook-training plan.

**CUTLASS reuse — honest feasibility.** The attractive story is: reuse the CUTLASS 4
block-scaled NVFP4 GEMM collective mainloop, swap only the global→shared iterator for
one that does codebook lookup. Reality:
- CUTLASS 4 *does* ship block-scaled NVFP4/MXFP4 GEMM templates (Codex-001 confirms,
  `conference/001-codex-opening.md:40-41`), and the scale/MMA half is exactly what we
  want to inherit.
- But the CuTe collective mainloop does **not** cleanly expose a "custom
  global→shared transform" hook. Getting the LUT-expand into the prologue realistically
  means **forking the collective mainloop** (a custom `CollectiveMma` prologue /
  smem-tile producer in CuTe) or dropping to raw PTX for the inner loop.
  **2026-07-15 review correction:** `tcgen05` is the sm_100a *datacenter* Blackwell
  instruction family and does not exist on GB10 (sm_121, consumer-class); the FP4
  path here is the sm_120/121 block-scaled `mma` family. First kernel task is to
  disassemble what CUTLASS actually emits for the working sm_121 NVFP4 GEMM (we
  serve NVFP4 on this box today, so the instruction path exists) and target that.
  This is the genuinely hard, weeks-not-days piece.
- Triton **cannot** emit Blackwell FP4 (`tcgen05`) MMA today — a Triton kernel can do
  the smem lookup but only reaches BF16 MMA, violating INV-2. So Triton is a
  correctness/decode tool, **not** the prefill answer. Do not let a working Triton
  prefill masquerade as production-eligible; it will fail the perf gate.

### 1b. DECODE / batch-1 GEMV (M≤~8): bandwidth-bound, keep it simple

Measured facts argue *against* over-engineering here: IQ (codebook) decode was 17.8
vs 18.7 tok/s for k-quant on a 295B artifact — **codebook lookup overhead is small,
the path is bandwidth-bound** (memory `gguf_lowbit_serving_lane.md`). And decode is
where NVFP4-CB *should* win outright: fewer bytes/weight = less HBM traffic = faster
decode, the opposite of the retired lane's story once we honor INV-1.

Plan: a **Triton (or small CUDA) dequant-GEMV** for M≤threshold, mirroring GGUF's
own M-gated dispatch (`quantization/linear.py:34-57`: MMVQ for `x.shape[0] <= mmvq_safe`,
MMQ above). It streams k-bit indices, expands per-group in registers, multiplies by
the group scale, and accumulates against the (few) activation rows. It must obey
INV-1 (never materialize the full weight — the GGUF `DEQUANT_TYPES` fallback at
`linear.py:49-53` materializes `[N,K]` and is the anti-pattern for us at decode).
BF16-vs-FP4 MMA is irrelevant at M=1 (no tensor-core utilization anyway), so the
decode kernel does *not* need the CUTLASS FP4 path — this decouples decode
(ship-early, easy) from prefill (hard).

**GB10 latency-floor + CUDA-graph requirement.** The ~6 ms/call floor measured on
the retired low-bit kernels was
grouped-**MoE** dispatch + many tiny launches (`conference/001:24-25`), not an
intrinsic GEMV cost. Two mitigations, both mandatory per house rules (CLAUDE.md §4.10,
memory `feedback_cuda_graphs_everywhere.md`): (i) the GEMV must be a single fixed-shape
launch per Linear, **CUDA-graph-capturable** — no host-side branching on tensor
values, no data-dependent shapes inside the captured region; bit-exactness with
capture off must hold. (ii) fuse where the launch count is (RMSNorm+act-quant fusion,
Codex-001 tier-1) only after correctness. Verify capture compatibility on day one of
the decode prototype — GGUF's custom-op registration (`linear.py:68-74`
`direct_register_custom_op` + a `fake_impl`) is the pattern that keeps it
compile/capture-safe; we replicate it for our ops.

## 2. vLLM plugin architecture (`vllm-prismaquant-plugin`)

Out-of-tree package, GGUF plugin as the literal template.

**Registration** (`vllm_gguf_plugin/plugin.py:109-127`): `register()` calls
`register_quantization_config("prismaquant")(PrismaQuantConfig)`. Ship as an entry
point so `--quantization prismaquant` (or auto-detect from the config's
`quant_method`) activates it with zero vLLM-core edits. Unlike GGUF we do **not** need
the model-loader / config-parser / engine-arg monkeypatches (`plugin.py:51-97`) —
our artifact is already a standard HF dir + safetensors + `config.json`, so vLLM's
normal HF load path works; we only register the quant config (and, if needed, a
weight-loader shim for the codebook sidecar).

**One master config, per-layer dispatch + delegation.** Model on
`GGUFConfig.get_quant_method` (`quantization/config.py:67-90`). Our
`PrismaQuantConfig.get_quant_method(layer, prefix)`:
- prefix matches an **NVFP4-CB** target → return our `PrismaQuantCBLinearMethod`
  (or `...CBMoEMethod` for `RoutedExperts`).
- prefix matches **plain NVFP4 / FP8** → **delegate to stock vLLM
  `CompressedTensorsConfig.get_quant_method`**. Construct a `CompressedTensorsConfig`
  from the same `config.json` at our config's `from_config`, hold it as a member,
  and forward. This is the load-bearing "we are not reimplementing FP8/NVFP4" move —
  those layers hit vLLM's own CUTLASS block-scaled kernels, fully maintained upstream.
- **BF16 / unquantized** (matches the `ignore` list) → `UnquantizedLinearMethod()`
  (exactly `config.py:80`), delegated the same way.

**Config JSON encoding.** Extend the exporter's existing
`build_quantization_config` (`export_native_compressed.py:6845`, tail
`7256-7261`): keep `quant_method:"compressed-tensors"`, `format:"mixed-precision"`,
and the `config_groups`/`ignore` machinery **verbatim** — add NVFP4-CB scheme dicts
to `FORMAT_SCHEME` (`:6768`) with a custom scheme name (e.g. `"nvfp4_cb"`) carrying
`{num_bits, group_size:16, codebook_dim:8, index_bits:k, codebook_ref}`. Per-layer
targets are regex lists exactly as today (`_build_target_list`, `config.py`
group assembly `7208-7255`). vLLM's stock compressed-tensors parser will not know
`nvfp4_cb`; the delegation split above is what makes that fine — our config owns the
CB groups, delegates the rest. (Alternative if we want stock compressed-tensors to
parse everything: register a `compressed-tensors` **override** via
`override_quantization_method`, `config.py:58-65`. Prefer the clean separate
`quant_method` name to avoid coupling to compressed-tensors' schema churn.)

**Weight / sidecar loading.** Follow `GGUFLinearMethod.create_weights` /
`process_weights_after_loading` (`quantization/linear.py:86-146`): register the
packed **index** tensor (uint8, byte-shaped), the **E4M3 group scales**, and the
**codebook** as parameters; materialize in `process_weights_after_loading`. Learned
per-tensor codebook = a per-Linear safetensors tensor (≤256 KB, `linear.py`-style
extra param). A *shared* structured/lattice table = one model-level tensor referenced
by `codebook_ref` and loaded once (don't duplicate per Linear). INV-1: keep them
packed here; do not expand.

**Fused shards / MoE (`apply_vllm_mapper`, `packed_modules_mapping`).** The hard
serving invariants (CLAUDE.md §6) are already enforced *at export*: fused siblings
(q/k/v, gate/up) and packed MoE experts share **one** scheme via union-find promotion;
config_groups use canonical `gate_proj/up_proj/down_proj` names
(`_vllm_moe_scheme_projection_names`, `export_native_compressed.py:3793`, `6985`).
So at serve time a fused qkv Linear is a *single* NVFP4-CB (or NVFP4, or FP8) method —
no per-shard scheme mixing. GGUF's per-shard fallback (`linear.py:216-240`,
`fused_moe.py:111-120` "slow implementation") is a perf convenience we mostly won't
need because export guarantees uniformity; still worth keeping a per-shard slow path
for robustness. `apply_vllm_mapper` (`config.py:92-103`) must rewrite our target/ignore
lists through `hf_to_vllm_mapper` just as GGUF rewrites `unquantized_modules`.
MoE: mirror `GGUFMoEMethod` (`quantization/fused_moe.py`), one CB type for stacked
w13 and one for w2 per layer (experts-uniform-per-layer, memory
`session_2026_05_29_lfm25_ship.md`).

## 3. Risks & mitigations (each with a concrete test)

| Risk | Mitigation | Test |
|---|---|---|
| **Activation dtype.** GGUF forces fp16 activations on Blackwell (memory `gguf_lowbit_serving_lane.md`); we must NOT inherit that — CB layers are **W4A4**, activations RTN-quantized to NVFP4 exactly like stock NVFP4 (inline act-quant `nvfp4_fused.py:216-229`). | CB method advertises the same act path as vLLM's `CompressedTensorsW4A4Nvfp4` scheme; feed the kernel the NVFP4 activation scale. `get_supported_act_dtypes` returns bf16 (not fp16-forced). | Load a 2-layer CB model, assert activation tensor is NVFP4-quantized (not fp16 passthrough); compare a CB Linear's output vs a plain-NVFP4 Linear built from the same decoded tile — must match to NVFP4 RTN tolerance. |
| **CUDA-graph capture** breaks on data-dependent control flow. | Custom ops registered via `direct_register_custom_op` + `fake_impl` (`linear.py:68-74`, `ops.py:116-158`); fixed-shape, no host branching in captured region; env-gated eager fallback with bit-exactness. | Capture+replay a decode step; assert graph-on == graph-off logits bit-exact (CLAUDE.md §4.10). |
| **MoE fused-expert path** — grouped dispatch was the 6 ms floor on the retired lane. | Reuse `GGUFMoEMethod` structure + `moe_align_block_size` (`fused_moe.py:53-95`); single grouped kernel, not per-expert launches. | 8×-expert MoE layer decode tok/s ≥ plain-NVFP4-MoE within bandwidth ratio; no per-token launch storm (nsys). |
| **torch.compile interaction.** | Every kernel is a registered custom op with a `fake_impl` (matches `ops.py:110-158`); opaque to Dynamo, no graph break inside. | `vllm serve --enforce-eager` and default (compiled) both produce identical greedy output. |
| **vLLM API churn.** GGUF plugin already monkeypatches internal APIs (`plugin.py:51-97`); `LinearMethodBase`/`QuantizationConfig` signatures drift. | Pin a tested vLLM version per artifact (as with GGUF venvs, memory `container_transformers_pin`); keep our surface minimal (config + method + custom op), avoid loader patches. | CI smoke against the pinned vLLM; a canary test that imports the vLLM symbols we depend on and fails loudly on signature change. |
| **Maintenance burden — this makes us a CUDA kernel vendor.** | Upstreaming options, in order of preference: (1) **standalone `vllm-prismaquant-plugin`** now — clean ownership, GGUF's `setup.py` CUDAExtension harness (`setup.py:30-60`) copies directly; (2) eventually push the CB scheme into **vLLM compressed-tensors upstream** — the right long-term home since our scales *are* NVFP4 and much of the kernel inherits CUTLASS block-scaled NVFP4; (3) a PR into `vllm-gguf-plugin` is the **wrong** home (GGUF-typed csrc, semantic mismatch) — reject. | N/A (strategic). Decision recorded; revisit after prototype (iii). |

## 4. De-risking prototype sequence (effort: LoC + days)

Effort assumes one engineer familiar with the repo; days are focused-work days.

- **(i) Triton dequant-GEMV, correct end-to-end serve.** ~400–600 LoC (kernel +
  `PrismaQuantConfig` + `CBLinearMethod` + delegation to compressed-tensors + codebook
  loader). ~4–6 days. Slow but honors INV-1 (no full-weight materialization).
  **← first gold-metric gate:** a correct-but-slow serve is enough to measure exact
  vLLM KL-vs-BF16 + WikiText PPL on the served artifact — the *quality* of NVFP4-CB
  can be promoted/rejected here without any fast kernel. (Per CLAUDE.md §5 promotion
  ladder, kernel-perf is a *later* gate; quality is measured first.)
- **(ii) Decode-path perf to parity.** Tune (i) + register as CUDA-graph-capturable
  custom op; optional small-CUDA GEMV if Triton under-performs. ~200–400 LoC. ~3–5
  days. Target: decode tok/s ≥ plain-NVFP4 at the same layer count (should *beat* it
  on bandwidth). Not required for a KL number; required for a Candidate→Production
  promotion.
- **(iii) CUTLASS/CuTe fused-expand prefill (FP4 MMA).** The hard one. Custom CuTe
  smem-producer over the block-scaled NVFP4 collective (or the sm_121 block-scaled
  `mma` PTX inner loop — NOT tcgen05, see §1a correction);
  small-k LUT variant first, then structured/computed-codebook variant for k≥14.
  ~1000–2000 LoC + CUTLASS ramp. ~15–25 days, real risk it needs a mainloop fork.
  **Required for production-eligibility** (INV-2 / perf gate) but **not** for the
  first KL measurement.
- **(iv) MoE grouped fused-expand variant.** Port (iii) into a grouped-GEMM /
  `GGUFMoEMethod`-shaped path. ~600–1000 LoC. ~8–12 days. Needed for MoE artifacts
  (the 35B/122B/Hy3-class targets) to be production-eligible.

Minimum path to a **first shippable quality verdict**: (i) only. Minimum path to a
**production-eligible dense artifact**: (i)+(ii)+(iii). MoE artifact: + (iv).

## 5. Non-goals (what we do NOT build)

- **No llama.cpp / GGUF interop.** NVFP4-CB is a native compressed-tensors-style
  scheme; we do not emit or consume GGUF, and the GGUF plugin is a *template*, not a
  dependency.
- **No pre-Blackwell arches initially.** Kernels target sm_121 (GB10) FP4 tensor
  cores. No sm_80/90 fallback beyond the slow Triton dequant path (which exists only
  for correctness/CI, not as a shipped serving target).
- **No in-tree vLLM changes.** Everything via `register_quantization_config` +
  out-of-tree custom-op extension (GGUF-plugin model). Upstreaming to
  compressed-tensors is a *future* option, not this plan.
- **No revival of INT2/INT3 or BF16-MMA-after-dequant as a production path.** The FP4
  codes must reach FP4 tensor cores (INV-2); a bf16-MMA kernel is a correctness tool
  only.
- **No load-time expansion to a dense NVFP4 tensor** (INV-1) — that path is the
  documented OOM trap and is disqualified by construction.
- **No standalone learned codebook with k>14 stored as a flat table** — beyond the
  smem wall; such rungs must use a structured/computed codebook or they are
  unservable at fused-prefill speed.
