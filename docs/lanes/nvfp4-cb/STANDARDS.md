# gridbook / NVFP4-CB — FINAL kernel & format standards

Dated 2026-07-21 (Robert: "make a definitive determination about final kernel
and format standards"); runtime boundary updated 2026-08-02 for Gridbook
0.7.0, refreshed 2026-08-12 for released Gridbook 0.8.5, and re-pointed
2026-08-21 to released Gridbook 0.8.11 when the producer pin advanced. The
strict Ada candidate boundary was refreshed 2026-08-24 without promoting the
tracked runtime. The NVFP4 producer scaffold was expanded the same day to
K1..K25; widths above the public ceiling remain direct-codec/kernel research
only. This page is the contract production runs build against.
Changes to it require a served A/B, not a preference.

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
| NVFP4_CB (fp4 grid) | K1–K25, EVERY integer | 0.40625–3.40625 bpw serialized body | two-tier v2 (E8M0 super + 4-bit sub codes, 0.28125 bpw) | product, ceil-first uneven splits |
| FP8_CB (e4m3 grid) | producer: K4–K48 in steps of 4; reader: every K28–K48 plus the low producer rungs | 0.5–6.0 index bpw + `32/in_features` scale bpw | per-output-row fp32 tensor | product, ceil-first uneven splits |
| NVFP4 (vanilla) | — | 4.5 bpw | group-16 E4M3 | menu member; Blackwell-only serving |
| FP8_DYNAMIC | — | 8 bpw | per-channel | menu member |
| BF16 / FP8_SOURCE | — | 16 / ~8 | — | passthrough-only (source dtype) |

- Codeword layout: 32 k-bit codewords per 256-weight superblock, LSB-first;
  sub-index bit split is **ceil-first** (`_bit_split`), sub-0 at the LSBs.
  K1 is `(1,0)`, with a one-row zero-bit second subtable; the public K25
  endpoint is `(13,12)`. A direct research-only K32 codec test pins `(16,16)`
  and an all-ones uint32 word, but K32 has no public format id.
- Canonical FP4 d4 lattices at widths 0..16 are versioned by
  `STRUCTURED_FP4_D4_LATTICE_VERSION`. Historical width-6..12 tables continue
  to resolve from the digest-pinned asset byte-for-byte. The materialized
  low-width tables are nested subsets of width 6; the materialized high-width
  tables are nested supersets of width 12. Production K1..K25 lookup is
  asset-only and refuses a missing key. Widths 14..16 remain explicitly
  research-only. Width 16 has 65,536 rows but 50,625 distinct numeric vectors
  because E2M1 has 15 numeric values.
- fp4 scale coding v1 (bare E4M3 plane) is legacy-compat only: readable,
  never produced by new exports.
- **Signed S-rungs (S13–S16): DELETED after measured NO-GO
  (measured,
  2026-07-22).** The K-vs-S head-to-head on Qwen3.5-0.8B (matched-rate menu,
  776 per-(Linear,k) direct cost comparisons): K wins 609/776 (78.48%), median S penalty
  +0.5–2.2%, allocator placed 6 S-units vs 147 K-units (only linear-attn
  in_proj_a/b/qkv/z ever preferred S). Serving propriety PROVEN: the signed
  chain (encoder → export → vLLM load → decode) is bit-exact on the real
  artifact (max |serve − reconstruct| = 0) plus the 18-test GPU battery.
  S-rungs stay OFF production menus and were deleted from the producer format
  vocabulary; no exotic-weight exception remains. Reproduction and immutable
  artifact identities: `../../results/qwen35_0p8b_s_rung_headtohead_2026-07-22.md`.
  Full mode: spec-reserved, unimplemented.
- MTP sidecars: CB-quantized, rung by the canon throughput selector
  (`mtp_rung_selection.py`). Vision towers (VLMs): vanilla NVFP4.
- Standard production menu = both product K-ladders (FP8 obeys K%4) + NVFP4 +
  FP8_DYNAMIC + BF16 (+FP8_SOURCE where the source is fp8). Target hardware:
  Blackwell (GB10 sm_121 / RTX 5090 sm_120). The separate strict
  `qwen38_rtx4090_fp8_cb` campaign removes both NVFP4 families and is closed
  to lattice FP8-CB (`CB_ACTIVATION_SCOPE=none`) plus delegated FP8/BF16. It
  remains closed until an exact `sm_89` Gridbook v11 contract device-qualifies
  dense decode and batch for **all twelve** K4..K48 step-4 producer rungs and a
  physical RTX 4090 receipt proves 7/7 FULL plus 7/7 PIECEWISE capture. The
  available explicit-sm89 SASS evidence is `compile_only`, not a device claim;
  Gridbook 0.9.0 TP/EP behavior remains preserved.

Target registration, not a hand-written family preference, controls AQUA's
choice. SM89/RTX40 registers no NVFP4 or NVFP4-CB activation contract and is
FP8-only. An exact attested sm120 target may register both NVFP4 K1..K25 and
FP8; AQUA then compares the registered candidates in its normal currency. It
must not bias one family manually, and K26..K32 are never registered.

The widened NVFP4 registry is producer scaffolding, not a runtime claim. A
new artifact using K1..K11 or K25 requires a Gridbook v11 contract whose
NVFP4 `rungs` and `producer_rungs` both attest K1..K25, plus device-qualified
route cells for the target structure and regime. K26..K32 are unsupported and
must not appear in a contract cell. Released Gridbook 0.8.11/v4
cannot provide producer-rung attestation, so release tooling must fail closed
until the external pin advances.

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
the pinned commit `187c7216b9d4882321c1923de0b4c49dc139743c`, whose contract
declares `schema`, `contract_version`, `quant_method`, `packing`, `layout`,
`formats`, `producer_profiles` and `abi_features` — and nothing about fused
dispatch. So the only cross-repo
check on this mirror is Gridbook's own tests over `gridbook/codec.py`
(`FP8_FUSED_KBITS`, `cb_fused_kbits()`); PrismaQuant CI cannot catch it drifting,
because `tests/test_gridbook_runtime_contract.py` compares rungs, layouts and
`quant_method` against a contract that is silent on this axis. Treat any change
to the rung set as a two-repo change, and re-read `codec.py` at the pinned
commit before editing either the JSON or the paragraph below.

The earlier PrismaQuant 0.8.0 boundary pinned Gridbook 0.8.0 at exact commit
`9011a19228ddb96b8a49e11a20ac75c99c83998e`. The current boundary pins released
Gridbook 0.8.11 at exact commit
`187c7216b9d4882321c1923de0b4c49dc139743c`. Every serving-reachable Gridbook
operation is native CUDA/CUTLASS and is resolved and attested at model load.
Gridbook has no Triton dependency, dispatch arm, or fallback; if an artifact,
shape, ABI, or device lacks its required native operation, serving fails closed.
The 0.8.11 contract declares the DSV4 body/MTP/DSpark loader, routed per-role
LUT ABI, raw-source block-FP8 W8A16 and the DSpark construction physical
bridge; that runtime capability does not itself qualify any DSV4 artifact or
change this producer's format/layout ABI, menu, or quality-promotion status.

The FP8-CB fused mid-M rung surface established by 0.7.0 remains unchanged
through 0.8.11 and is not pending completion: Gridbook's K1.2 resolution proved
`k % 4 == 0` is a format+TMA law
(`gridbook/codec.py` `FP8_FUSED_KBITS`, queryable as `cb_fused_kbits()`), so
`{28,32,36,40,44,48}` is the whole lane and the off-law rungs this producer may
still assign are permanently expand+GEMM-served. Gridbook 0.7.0 introduced the
contract-preserving FP4-CB v2 fused mid-M kernel; it remains **opt-in** in
0.8.11 behind `PRISMAQUANT_CB_FP4_FUSED_MIDM` pending its served NATIVE-PARITY
gate, so the fp4-CB backed set in
`prismaquant/serving_profile_specs/nvfp4_cb.json` stays empty: available is not
backed.

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
- The old dense K12..K24 rung/parity interpolation is not endpoint evidence.
  The widened dense campaign schema uses a below-K12 hinge and an above-K24
  endpoint shoulder, measures K1/K25 in its panel, and validates K25 transfer
  on held-out units. Because K25 is the only public rung above K24, this is not
  described as a fitted high-band slope. No cost payload produced under the
  former campaign schema may be restamped or extrapolated onto K1..K11 or K25.
