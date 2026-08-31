# Trellis E2M1 R512 on Qwen3-4B — weight-space screen

**A screen, not a result** (principle 3). Read §3 before quoting anything here.

Date 2026-08-31 · branch `claude/prismabuild-trellis-integration-20260831` ·
GB10 sm_121, `prismaquant-cu130` · driver
`scratchpad/qual_4b.py`, log
`dq-runs/trellis-serve-smoke-20260831/qual_4b_full.log`.

## 1. What was run

`prismaquant.trellis_producer.encode_trellis_one_linear` — the **production**
producer path, not the research encoder — over all 63 tensors of
`/home/rob/models/Qwen3-4B` for which
`dq-runs/trellis-currency-20260829/importance_qwen3-4b_n32_s512_inline.pt`
carries a per-input-column importance vector (9 layers x 7 roles). That vector
is passed as `col_weights`, so the Viterbi objective is importance-weighted, as
production would be.

Rung `TCQ_E2M1_R512` (`body_rate_q256=512` = 2.0 body bits/weight),
`layout=fixed_quota_per_256`, `scale_rule=static_6`, `tailbite_candidates=4`,
`determinism_mode=on`, `backend=triton`, `sb_chunk=rows`.

Baseline arm: `format_registry` `NVFP4` `quantize_dequantize` on the same
tensor, scored with the same importance weighting.

Metric: importance-weighted SNR, `10*log10( sum(w * W^2) / sum(w * (W-Ŵ)^2) )`.

## 2. Result

63 tensors, 46.4 s wall for the whole sweep. Medians by role:

| role | n | trellis dB | wire bpw | NVFP4-RTN dB @ 4.5 | gap |
|---|---|---|---|---|---|
| down_proj | 9 | 11.40 | 2.5016 | 20.45 | −9.05 |
| gate_proj | 9 | 11.50 | 2.5004 | 20.69 | −9.18 |
| k_proj | 9 | 11.49 | 2.5042 | 20.33 | −8.84 |
| o_proj | 9 | 10.84 | 2.5016 | 20.34 | −9.49 |
| q_proj | 9 | 11.28 | 2.5011 | 20.27 | −8.99 |
| up_proj | 9 | 10.85 | 2.5004 | 20.32 | −9.47 |
| v_proj | 9 | 10.77 | 2.5042 | 20.18 | −9.41 |
| **ALL** | **63** | **11.14** | **2.5016** | **20.36** | **−9.21** |

Role spread is only 0.73 dB — the format behaves uniformly across this model.

**The coding gain reproduces.** The 2026-08-30 ladder recorded scalar rate-2 on
Qwen3-4B dense at **9.462 dB**; the trellis at 2.0 body bits medians
**11.14 dB**, i.e. **+1.68 dB**, against the recorded "+1.880 dB at rate 2,
24/24 positive". This is the first confirmation that the *promoted production*
encoder delivers the research encoder's coding gain on real weights.

**63/63 completed with no abort**, which is itself the point: before the
reconstruction-association fix landed the same sweep died at
`layers.31.self_attn.v_proj` on the producer's own same-byte invariant.

## 3. Scope — what this does NOT say

- **Weight-space only.** wSNR prices the W\*A16 shape. The attested serving cell
  for this rung is **W4A4** (`e2m1_group16_ue4m3_static`), whose activation
  perturbation nothing here measures and the AURA cost is structurally blind to.
  No KL, no PPL, no served artifact.
- **The rates are not matched.** 2.50 bpw against 4.5 bpp. A −9.21 dB gap at
  44% of the size is not a defeat and not a win; it is a different point on a
  rate axis the production menu cannot otherwise reach (NVFP4's floor is 4.5 bpp
  per Linear).
- **Both arms are un-optimized, differently.** The NVFP4 arm is registry RTN with
  no GPTQ and no JSO, so it understates the production NVFP4 render. The trellis
  arm takes a one-shot `static_6` amax scale with no grid search, so it
  understates what the scale grid would give. Neither column is a production
  number.
- **Nine layers of 36**, chosen by which tensors the existing importance file
  covers, not by sampling design.
