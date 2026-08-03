# gridbook / NVFP4-CB — FINAL kernel & format standards

Dated 2026-07-21 (Robert: "make a definitive determination about final kernel
and format standards"); runtime boundary updated 2026-08-02 for Gridbook
0.7.0. This page is the contract production runs build against. Changes to it
require a served A/B, not a preference.

The separately versioned, exact-commit-pinned Gridbook runtime follows this
contract. Dense and MoE fused-FP4 prefill are disabled unless their respective
opt-in environment variables are set. Kernel-only speedups do not promote a
path across activation buckets. Runtime aliases, rungs, layouts, and supported
producer profiles come from Gridbook's packaged `runtime_contract.json`; this
repository does not duplicate those tables.

## Format standard (what an artifact may contain)

**Production formats** — the complete set:

| Family | Rungs | Rate | Scale coding | Mode |
|---|---|---|---|---|
| NVFP4_CB (fp4 grid) | K12–K24, EVERY integer | 1.78125–3.28125 bpw serialized body | two-tier v2 (E8M0 super + 4-bit sub codes, 0.28125 bpw) | product, ceil-first uneven splits |
| FP8_CB (e4m3 grid) | K28–K48, EVERY integer | 3.5–6.0 index bpw + `32/in_features` scale bpw | per-output-row fp32 tensor | product, ceil-first uneven splits |
| NVFP4 (vanilla) | — | 4.5 bpw | group-16 E4M3 | menu member; Blackwell-only serving |
| FP8_DYNAMIC | — | 8 bpw | per-channel | menu member |
| BF16 / FP8_SOURCE | — | 16 / ~8 | — | passthrough-only (source dtype) |

- Codeword layout: 32 k-bit codewords per 256-weight superblock, LSB-first;
  sub-index bit split is **ceil-first** (`_bit_split`), sub-0 at the LSBs.
  Encoder-anchored tests pin this; it is frozen.
- fp4 scale coding v1 (bare E4M3 plane) is legacy-compat only: readable,
  never produced by new exports.
- **Signed S-rungs (S13–S16): CLOSED as research-only (measured,
  2026-07-22).** The K-vs-S head-to-head on Qwen3.5-0.8B (matched-rate menu,
  776 per-(Linear,k) direct cost comparisons): K wins 609/776 (78.48%), median S penalty
  +0.5–2.2%, allocator placed 6 S-units vs 147 K-units (only linear-attn
  in_proj_a/b/qkv/z ever preferred S). Serving propriety PROVEN: the signed
  chain (encoder → export → vLLM load → decode) is bit-exact on the real
  artifact (max |serve − reconstruct| = 0) plus the 18-test GPU battery.
  S-rungs stay OFF production menus — correct but not worth menu space; the
  spec keeps them for exotic weight geometries. Reproduction and immutable
  artifact identities: `../../results/qwen35_0p8b_s_rung_headtohead_2026-07-22.md`.
  Full mode: spec-reserved, unimplemented.
- MTP sidecars: CB-quantized, rung by the canon throughput selector
  (`mtp_rung_selection.py`). Vision towers (VLMs): vanilla NVFP4.
- Standard production menu = both product K-ladders (all integers) + NVFP4 +
  FP8_DYNAMIC + BF16 (+FP8_SOURCE where the source is fp8). Target hardware:
  Blackwell (GB10 sm_121 / RTX 5090 sm_120). Artifacts that happen to
  allocate zero vanilla-NVFP4 units remain Ada-servable as a bonus, never a
  constraint.

## Runtime/kernel standard (owned by Gridbook)

Runtime dispatch, kernel defaults, environment switches, supported shapes, and
operator evidence live only in the Gridbook repository and its packaged
contract. Consult `docs/PLUGIN.md`, `docs/KERNELS.md`, and the dated audits from
the exact commit in `prismaquant/gridbook_runtime/gridbook_runtime_pin.json`; this producer
document intentionally carries no parallel dispatch table.

**One honest exception, and it is load-bearing.** The fused mid-M rung set
below is a **hand-maintained mirror** of a Gridbook fact, not a derivation from
the pinned package. Its machine-readable form is this producer's own
`prismaquant/serving_profile_specs/nvfp4_cb.json`
(`serving_lanes[].fused_mid_m.rungs_by_runtime_version`), parsed at
`prismaquant/serving_profiles.py:460-489` and resolved against the pinned
runtime version into the per-candidate `fused_mid_m_backed` /
`fused_mid_m_rungs` the P5b router prices on; the prose here restates that data. The mirror exists because Gridbook's **packaged
`runtime_contract.json` does not carry the fused rung set** — verified against
the pinned commit `ca0f0f562d3f398e094bfa5356a9ce3fa47472f1`, whose contract
declares `quant_method`, `packing`, `layout`, `formats` and
`producer_profiles` and nothing about fused dispatch. So the only cross-repo
check on this mirror is Gridbook's own tests over `gridbook/codec.py`
(`FP8_FUSED_KBITS`, `cb_fused_kbits()`); PrismaQuant CI cannot catch it drifting,
because `tests/test_gridbook_runtime_contract.py` compares rungs, layouts and
`quant_method` against a contract that is silent on this axis. Treat any change
to the rung set as a two-repo change, and re-read `codec.py` at the pinned
commit before editing either the JSON or the paragraph below.

For PrismaQuant 0.8.0 that pin is Gridbook 0.8.0 at exact commit
`9011a19228ddb96b8a49e11a20ac75c99c83998e`. Every serving-reachable Gridbook
operation is native CUDA/CUTLASS and is resolved and attested at model load.
Gridbook has no Triton dependency, dispatch arm, or fallback; if an artifact,
shape, ABI, or device lacks its required native operation, serving fails closed.
This runtime advance does not change this producer's format/layout ABI,
menu, or quality-promotion status, and it makes no DSV4 qualification claim.

The FP8-CB fused mid-M rung surface is unchanged at 0.7.0 and is not pending
completion: Gridbook's K1.2 resolution proved `k % 4 == 0` is a format+TMA law
(`gridbook/codec.py` `FP8_FUSED_KBITS`, queryable as `cb_fused_kbits()`), so
`{28,32,36,40,44,48}` is the whole lane and the off-law rungs this producer may
still assign are permanently expand+GEMM-served. Gridbook 0.7.0's
contract-preserving FP4-CB v2 fused mid-M kernel is still **opt-in** behind
`PRISMAQUANT_CB_FP4_FUSED_MIDM` pending its served NATIVE-PARITY gate, so the
fp4-CB backed set in `prismaquant/serving_profile_specs/nvfp4_cb.json` stays
empty: available is not backed.

The producer-side decision needed here is fixed: both fused NVFP4 activation
contracts remain explicit opt-ins and default OFF. The 2026-08-01 LFM
teacher-backed gate rejected promotion despite green CUDA arithmetic/routing
tests. No PrismaQuant menu, launcher, or artifact metadata may imply those paths
are defaults. Reconsideration requires Gridbook's served quality and workload
gates to pass, followed by advancing the immutable external pin.

## Cost standard

- CB rung costs: measured anchors + the **split-aware floored RD law**
  `D(k) = F + C·R(k)` per (Linear, family) — F is the measured infinite-k
  grid floor (fp8: per-channel-fp8 RTN render; fp4: two-tier E2M1 RTN
  render), and `R(k) = Σᵢ 2^(−2·bᵢ/dᵢ)` is the EXACT rate factor over the
  rung's ceil-first sub-splits (bᵢ bits over dᵢ dims), so the k%n_sub
  sawtooth lives in the regressor instead of the residual; C by linear
  least squares over the anchors. Fit chain on rejection: split-aware →
  smooth floor law `F + C·2^(−α·k)` → legacy log-linear — each proposal
  **holdout-gated**: any tensor whose holdout error misses the bar falls
  back to full per-rung measurement. The gate is the contract; the laws
  are only ever proposals under it. (Measured on the 27B full-menu run:
  3.2% of tensors fall back.)
