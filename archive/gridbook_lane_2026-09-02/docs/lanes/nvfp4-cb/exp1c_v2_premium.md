# NVFP4-CB exp-1c — v2 premium-flip re-measurement (Qwen3-0.6B)

> **EMULATION GATE — 0.6B, single model.** Whole-model emulated forward KL-vs-BF16 (fp32, held-out wiki.test.raw, seqlen 512 × 8192 tok; W4A4 act emulation for CB, weight-only for IQ — each format in its served bucket). 27B + served vLLM KL remain the promotion bar.

Two-tier v2 scale coding (two-tier-scale-spec.md) fixed the fp4 subnormal-collapse defect AND cut scale bytes 0.500→0.28125 bpw, making NVFP4_CB_K14/K16/K18 exact matched-bytes twins of IQ2_XXS/IQ2_XS/IQ2_S at −0.03125 bpw each. exp-1b measured the v1 native-FP4 premium at +0.15 bpw; this experiment re-measures it. Encoder: shared-per-role learned product codebooks, `balanced` tier (encode_tiers.md: ≈max quality, ~4× faster), scale sweep on, 4 paired calibration seeds (2 for context rungs), same imatrix draws as exp-1/1b.

- git `1dfa476e0e512ca23ee8b94450b5d98b283b15b5` · 196 target Linears · 7 roles.

## Per-arm results

| Arm | coding | seeds | total bpw | KL_conf mean±std | KL_all | top1 | n_swap |
|---|---|---|---|---|---|---|---|
| CB16_v2 | two_tier | 4 | 2.2814 | 2.2202±0.1470 | 2.2060 | 0.430 | 196 |
| IQ2XS | — | 4 | 2.3125 | 1.7870±0.0486 | 1.8782 | 0.523 | 196 |
| CB18_v2 | two_tier | 4 | 2.5315 | 1.1815±0.0436 | 1.2800 | 0.630 | 196 |
| IQ2S (exp-1b reuse) | — | 4 | 2.5625 | 1.5837±0.0751 | 1.7747 | 0.568 | 196 |
| CB14_v2 | two_tier | 2 | 2.0313 | 4.1630±0.2656 | 3.9673 | 0.235 | 196 |
| IQ2XXS | — | 2 | 2.0625 | 3.2062±0.0130 | 3.0971 | 0.330 | 196 |
| CB16_v1 | v1 | 2 | 2.5001 | 2.1651±0.0384 | 2.1225 | 0.443 | 196 |

## Verdicts

### (i) Premium eliminated per rung? (CB-v2 ≤ IQ at matched bytes within between-seed noise)

- CB16_v2 (2.2814 bpw) vs IQ2_XS (2.3125): KL_conf 2.2202 vs 1.7870 (Δ=+0.4332, +24.2%, σ=0.1470) → **NO** (CB worse by +24.2% > σ=0.1470).
- CB18_v2 (2.5315 bpw) vs IQ2_S (2.5625): KL_conf 1.1815 vs 1.5837 (Δ=-0.4022, -25.4%, σ=0.0751) → **YES — premium eliminated** (CB ≤ IQ within noise, at FEWER bytes).
- CB14_v2 (2.0313 bpw) vs IQ2_XXS (2.0625): KL_conf 4.1630 vs 3.2062 (Δ=+0.9568, +29.8%, σ=0.2656) → **NO** (CB worse by +29.8% > σ=0.2656).

### (ii) v1→v2 delta at k16 (scale fix + byte cut, tier held at balanced)

- k16 v1 2.1651 @ 2.500 bpw → v2 2.2202 @ 2.281 bpw: KL +2.5% (within between-seed noise, σ=0.147) at **−0.219 bpw** — the two-tier byte cut is KL-free within noise, i.e. the whole v1 curve shifts left as the spec predicted. (Reproduction check: exp-1b's PROD_shared_k16 = 2.2102±0.0923 with the pre-tier max encoder; the v1/balanced rerun reproduces it within noise.)

### (iii) GO/NO-GO for the 27B production run

**The spec-posed premium-flip test (§2.2): CONFIRMED.** The v2 curve (k16→k18 log-linear) reaches IQ2_S quality (1.5837) at ≈**2.42 bpw** vs IQ2_S's 2.5625 ⇒ native-FP4 premium ≈ **-0.15 bpw** (spec predicted ≈−0.07; exp-1b v1 measured +0.15). K18-v2 strictly dominates IQ2_S — better KL at fewer bytes — while decoding tensor-core-native FP4.

**GO.** Premium eliminated where the spec posed the test (the IQ2_S rung: 1/3 pairs flip outright, and the flipped one is the flagship twin). The k14/k16 rungs remain behind their IQ twins (+24–30%) — an INDEX-RATE deficit (CB's RD slope is steeper than IQ2's at the bottom of the band), exactly the limit §2.2 already flagged, NOT a scale-coding defect (v1→v2 cut 0.219 bpw at unchanged KL, (ii) above). This does not block the 27B run: IQ is not in the served vLLM menu; the CB rungs' 27B role is to open measured 2.0–3.3 bpw points below NVFP4's 4.5 floor, and the AURA allocator selects rungs on measured cost — dominated rungs price themselves out. Proceed to the 27B production run (AURA-allocated mixed menu incl. CB rungs vs shipped PrismaAURA-5.5 at matched bpw). Emulation gate, 0.6B, single model: the 27B artifact must win/preserve on exact served vLLM KL + PPL at matched bytes before any ship/promotion claim.

## Caveats

- Emulation gate, 0.6B, single model, 4 seeds (2 on context rungs). 27B + served vLLM KL is the promotion bar; the GGUF lane precedent says 0.6B wins can fail to transfer.
- IQ2_S row reused from exp-1b (identical harness/imatrix/seeds — paired). CB/IQ activation asymmetry (W4A4 vs weight-only) is deliberate: each format is measured in its served bucket; exp-1b bounded the W4A4 share at ~10% of CB KL.
- balanced tier throughout (encode_tiers.md: parity with max on K16-v2 spot checks); v1 arm also balanced so (ii) isolates scale coding, not tier.
