# The E4M3 trellis W8A8 serving lane

**Status:** built and gated 2026-08-29. `gridbook/trellis_e4m3_lane.py` (225
lines) + `tests/test_trellis_e4m3_lane.py` (11 tests, 4 mutants falsified).
**Not committed, not default-on.** **Loaded by vLLM and serving exact
contracted values since the 2026-08-29 container sweep (§5b)** -- but that is a
load-and-value gate, not a device qualification: the route resolves
`unattested` by design. Companion to
`trellis_serving_gap_2026-08-29.md`, which is what this closes the first item
of.

---

## 1. Why this one, and why first

The serving gap listed four steps and one unresolved fork (A=W: native routes
are W4A4/W8A8 while the whole menu prices W\*A16). This lane is the piece that
is **fork-independent and already authorized**: Rob asked for *"low bit rate
fp8 trellis ... for amd and pre Blackwell compatibility"*, and `_scaled_mm`
fp8×fp8 is exactly what Ada, Hopper and AMD/hipBLASLt expose. W8A8 E4M3 is that
lane under **every** branch of the fork, so building it commits to nothing the
fork would have to undo.

It is also the cheap one: a *consumer swap*, not a new mainloop.

## 2. The consumer-swap claim, proven rather than asserted

`e4m3_scaled_mm_identity.py`, four shapes, all PASS:

| check | result |
|---|---|
| decoded uint8 plane == reference `decode_codes` | `torch.equal`, all 4 |
| `codes.view(fp8).to(fp32) * row_scale` == wire's contracted weight | `torch.equal`, all 4 (**exact**) |
| `_scaled_mm(..., scale_b=row_scale)` vs fp32 reference | max rel 1.7e-3 (bf16 output rounding) |

The E4M3 wire contracts `W[r][c] = e4m3fn_value(code) * row_scale[r]` with
`global_scale_real` pinned to 1.0 and one fp32 scale per row. `_scaled_mm`
computes `(A @ Bᵀ) · scale_a[:,None] · scale_b[None,:]`. **The trellis row scale
*is* `scale_b`** — no repack, no re-derivation, no bridge. (E2M1 is not so
lucky: its group-16 block-scale plane still needs one, unwritten.)

## 3. Residency is a declared mode, because the measurement says the choice is real

Decode costs a flat **~188 µs per 4096×4096 tile** — flat in `M`, because it is
per *weight*, not per token. So it never amortizes across a batch, and the two
residencies are genuinely different products:

| mode | resident bytes | forward |
|---|---|---|
| `resident` | full fp8 — **8 bpw** | GEMM + A-side quant |
| `streamed` | the wire — **2.0 bpw at R512**, per layer, plus **one** decode tile shared device-wide | decode + GEMM + A-side quant every call |

> **The streamed row was false when first written, and it is the row the mode
> exists for.** As built, a streamed layer held the loaded `trellis_payload`
> **and** `prepare_wire_cuda`'s private device clone of the same bytes **and**
> a full decoded tile *per layer* — so streamed occupied ≈ 2×wire + 8 bpw,
> **strictly more than resident's 8 bpw**, while the docs and the mode's own
> error message advertised "wire bpw in memory". Fixed by dropping the
> parameter in streamed too (the prepared wire owns the bytes) and pooling the
> decode scratch device-wide (`gridbook/trellis_decode_pool.py`); both halves
> are now falsified in the gate, and the per-layer test that existed could not
> see either, because each layer's buffer looked correct in isolation.

Picking between these by peeking at a shape would be principle 1's banned
heuristic *and* would silently decide the artifact's memory footprint. So the
operator declares `GRIDBOOK_TRELLIS_E4M3_MODE` and there is **no default** —
an unset mode raises and names both options. A test falsifies that rule.

## 4. Measured, both instruments (principle 15)

M=512, N=K=4096, sparky/GB10 sm_121, torch 2.11.0+cu130.

**In-process** — `torch.profiler`, `acc_events=True`, `device_time` read as the
per-call average it already is:

| component | µs/call |
|---|---|
| bf16 reference matmul (`nvjet_sm121`) | **276.9** |
| `_scaled_mm` fp8×fp8, the lane's GEMM | **94.1** |
| trellis decode (`decode_native_v2_kernel`) | **187.8** |
| A-side per-token quant — *reference impl, see caveat* | 176.6 |

⇒ `resident` GEMM-only is **2.94×** bf16; `streamed` (decode+GEMM) is
**281.9 µs = 0.98× bf16 at ⅛ the resident bytes.**

**Both of those ratios EXCLUDE the A-side quant, and they are therefore
ceilings, not results.** bf16 pays no activation quantization; this lane always
does.

**That open bound is now closed at one end — measured, and it is large.**
`_q10_aside_cost.py` (9 paired interleaved reps) decomposes the resident arm:

| component | µs/call | note |
|---|---|---|
| bf16 reference matmul | 264.6 | the denominator |
| lane GEMM alone (pre-quantized A) | **94.2** | ⇒ 2.81× — *this is the published ratio* |
| A-side per-token quant alone | **147.3** | **1.56× the GEMM it feeds; 54% of the arm** |
| **full resident forward** | **274.2** | ⇒ **0.96× bf16** |

and the whole-arm wall clock agrees independently: `e4m3_lane_bench.py` re-run
unchanged gives resident 274.0 µs = **1.03×** and streamed 458.0 µs =
**0.61×**, matching its own recorded JSON (276.9 / 278.7 / 455.9) all along.
**So the GEMM-only ratio overstates this lane by ~2.9×, and streamed is
decisively slower than bf16 rather than at parity.** The lane module's
docstring said "Forward is GEMM-only"; it is not, and that clause is corrected.

**The bracket, stated honestly in both directions.** The 147.3 µs quantizer is
a naive multi-kernel torch reference (amax → divide → clamp → cast), while
vLLM's fused per-token fp8 quant is one kernel over 4 MB and should cost far
less. So 147.3 µs is an **upper** bound on the A-side and 0.96× a **lower**
bound on the lane: the real standing is in **[0.96×, 2.81×]**, and only the
container run places it. What is settled is that the GEMM-only number is not
the lane's speedup. The two bounds are symmetric and
both are stated on purpose: the stub-quant walls below are worst-case floors,
these decode+GEMM ratios are best-case ceilings, and the honest lane sits
between them. **Do not quote "parity" without this sentence.**

**The other side of the same bound.** The A-side quantizer
measured is a *reference* torch implementation, not vLLM's fused
`dynamic_per_token_scaled_fp8_quant` (vLLM is not installed in the build venv).
At 176.6 µs it is comparable to the entire rest of the lane, so the end-to-end
walls it produces — resident 278.7 µs, streamed 455.9 µs — are **upper bounds
on time and lower bounds on speed**. The production A-side is one fused kernel;
I have not measured it and put no number on it.

**Box-level** — Netdata `nvidia_smi.gpu_power_draw`, two 45 s holds, against
the ~140 W envelope (`gpu_utilization` is non-diagnostic on GB10):

| mode | calls/s | power | work per joule | envelope |
|---|---|---|---|---|
| `streamed` | 2 167 | ~86 W | **25.2 calls/J** | 61% |
| `resident` | 3 622 | 78 W | **46.4 calls/J** | 56% |

`resident` does **1.84× the work per joule**. Both arms carry the same stub
quantizer, so the *ratio* is sound even though the absolute rates are not.
Neither mode passes 62% of envelope: **the lane does not saturate the box**,
which is consistent with decode being bandwidth/latency-bound rather than
compute-bound and says there is headroom left.

**And the number branch (C) is aimed at:** decode is 187.8 µs against a 94.1 µs
GEMM — **67% of the streamed lane is a decode that round-trips 16.7 MB through
HBM**. Fusing it into the mainloop prologue is exactly what removes that trip.
This is the first quantitative case for (C) rather than a preference.

## 5. What the gate does and does not establish

Exercised for real: wire decode, the fp8 reinterpretation, the row-scale
identity, `_scaled_mm`, both residency modes, all refusals. 11/11 pass.

**Falsified** (mutating the *lane*, not a fixture — a green suite proves a
check runs, not that it is load-bearing):

| mutant | caught |
|---|---|
| `scale_b := ones` (drop the row scale) | 2 failed |
| resident decodes a zero plane | 2 failed |
| streamed skips `decode_native_packed_out` | 2 failed |
| unset mode silently defaults to `resident` | 1 failed |

**Stubbed, so NOT established:** vLLM's `LinearMethodBase` /
`ModelWeightParameter` and its FP8 quantizer. **This lane has never been loaded
by vLLM.** That owes a container run. The tests *fail* rather than skip when
the CUDA ext cannot build, so a green run means it ran.

## 5b. The load path, and the carriage decision it forced (2026-08-29)

The lane as first built could not be loaded by anything: its factory wanted
`rows/columns/row_stride_bytes/row_body_bits` from an unnamed caller, and its
finalize hook *required* a caller to have already bound
`gridbook_trellis_prepared`. No such caller existed anywhere.

Closing that exposed a carriage error. The first draft declared
`trellis_payload` as a `[rows, row_stride_bytes]` tensor — **and a wire cannot
be rebuilt from that.** The per-column rate schedule, the tight block offsets,
the per-rate alphabets and the scale plane live in the wire *header* and exist
nowhere else. So the checkpoint carries **one opaque `wire_bytes` blob per
Linear** (`TrellisWire.to_bytes`), and the lane parses it at finalize.

That in turn made every scale **derived rather than loaded**: `decoded_scales`
and `wire.scale_blob` already *are* the `scale_b` operands, and
`global_scale_real` is a header field. A checkpoint that also carried them as
separate tensors could disagree with its own wire and nothing would notice —
deriving makes that state unrepresentable. The one exception is genuinely not
a wire fact: E2M1's `input_global_scale` is the **A-side** static scale, so it
stays a loaded parameter.

New pieces: `gridbook/trellis_scheme.py` (the `config_groups` vocabulary +
the scheme↔blob identity refusal), a dispatch arm in `config.get_quant_method`
resolved **ahead of** fused-role ownership, and
`tools/make_trellis_smoke_checkpoint.py` (the exporter's embryo).

Validation is against `trellis.py`'s own `FAMILIES`/`RUNG_POLICIES`, **not**
`runtime_contract.json`: the wires this reader accepts are an in-package fact,
while a `formats` row or an eligibility cell would be a *serving* claim, and
no attestation for these lanes exists. Absence resolves `unattested`, which is
the honest status.

**One check was removed for being unreachable.** The lane briefly refused an
E4M3 wire with `global_scale_real != 1.0`. `TrellisWire.validate` already
refuses to construct one and `from_bytes` validates, so it could never fire —
and an unreachable guard that reads like a gate is worse than none, because it
invites the belief that the lane enforces the rule. The wire format does;
`test_the_wire_format_is_what_pins_the_e4m3_global_scale` is the record.

Gate: 11 lane tests + 11 dispatch tests, **8/8 mutants falsified** (dropping
the discriminator → 11 failed; the scheme↔blob identity → 2; the blob-length
check → 1; the TP>1 gate → 1; the CT-ignore entry → 1; the rate-domain check →
1; the fused refusal → 1; the A-side positivity check → 1).

## 6. Still owed, in order

1. **A container run** — vLLM actually loading the method. Until then "servable"
   is a claim about code, not a measurement. *(In progress 2026-08-29: vLLM
   builds the layer and binds `wire_bytes` — the loader reached
   `process_weights_after_loading` — see §5b and the run log.)*
2. **A real exporter.** `tools/make_trellis_smoke_checkpoint.py` writes a
   valid checkpoint but is **self-consistent, not encoded**: it builds a wire
   and takes the reference weight to be that wire's own decoded value, because
   the Viterbi encoder lives in the research trees, not this package. That is
   the right scope for a serving smoke and says nothing about encoding
   quality. A real exporter substitutes an encoder and changes nothing else.
3. **The A-side price.** The lane stamps `fp8_per_token_dynamic` — a term
   `nvfp4_activation_contract.ROUTE_CONTRACTS` already defines, so a probe can
   read it — but **nothing has priced it**. Every trellis quality number is
   weight-only SSE. This is the fork, unresolved.
4. **Routed MoE.** Dense only, everywhere.
5. **Contract cells** — and note that today the pinned serving release
   publishes no `lane_eligibility` table at all, so *everything* resolves
   `unattested`. See the serving-gap doc §3(d).

Evidence: `dq-runs/trellis-kernel-20260829/{e4m3_scaled_mm_identity.py,
e4m3_lane_feasibility.py,e4m3_lane_bench.py}` + their JSONs.
