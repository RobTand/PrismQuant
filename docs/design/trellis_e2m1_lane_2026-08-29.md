# The E2M1 trellis W4A4 serving lane

**Status:** built, gated, benched. **Loaded by vLLM and serving exact
contracted values** since the 2026-08-29 container sweep (§3b). Still not a
serving *quality* result: the route resolves `unattested` by design and the
A-side quality price is unmeasured. 2026-08-29.

Code: `gridbook/gridbook/trellis_e2m1_lane.py` ·
Gate: `gridbook/tests/test_trellis_e2m1_lane.py` (15 tests) ·
Probes: `dq-runs/trellis-kernel-20260829/_q{1..8}*.py`, `e2m1_lane_bench.json`

This is the 4-bit half of the pair opened by
[`trellis_e4m3_lane_2026-08-29.md`](trellis_e4m3_lane_2026-08-29.md), and the
one the low-bpp mandate needs: the trellis ladder lives below 3 bpw on E2M1.

---

## 1. It needs no bridge — and that retracts a recorded blocker

`route_probe.py` recorded, and three documents inherited, that the E2M1 tile
"carries the right CODES but the wrong SCALE PLANE … one fp32 scale per output
row". **That is the E4M3 family's contract.** E2M1's is
(`gridbook/trellis.py:716`, validated at `:544-553`):

```
W[r][c] = e2m1_value(code[r][c]) · e4m3fn_value(scale_blob[r·G + c//16]) · global_scale_real
```

— a group-16 ue4m3 block plane, which is *exactly* the operand the Blackwell
`cutlass3x_sm120_bstensorop_..._block_scaled_ue4m3xe2m1` mainloop demands.
Nobody read `trellis.py`; one probe's prose asserted a contract from memory.

| question | measured answer |
|---|---|
| nibble order | low nibble = even column — payload is a pure `view`, no repack |
| `decode_native_packed` shape | `[rows, cols/2]`, already two codes per byte |
| scale plane | `[rows, G]` uint8 e4m3 — **is** `scale_b`, once blocked |
| identity, varied scales both operands | **max abs error exactly 0, 8/8 shapes** |

Shapes covered: `N ∈ {192, 256, 320, 384, 448}` (three of them `N%128 ≠ 0`),
`M ∈ {1, 8, 17, 32, 64, 100, 128}`.

## 2. Two things `_scaled_mm` accepts and silently gets wrong

Neither raises. Both were caught by comparing numbers to the wire contract.

1. **An unswizzled scale plane** is accepted and silently miscomputes. The
   plane must be in the cuBLAS 128×4 blocked layout, and `blocked_scales()` is
   that layout — pinned in the gate against the wire's contracted product
   (ground truth), not against another implementation.

   > **RETRACTED, and the retraction makes the trap worse.** This paragraph
   > previously read: *"It coincides byte-for-byte with row-major when
   > `N%128==0` and `G%4==0` — which is why a row-major plane appears to work
   > and lies everywhere else."* **Both halves are false**, measured in
   > `_q9_rowmajor_at_aligned.py`. The 128×4 tile is swizzled *internally* (the
   > 32×4×4 shuffle), so **no** shape makes the layout row-major, and a
   > row-major plane is **67–70% wrong at aligned shapes too** (N=128, 256, 384
   > all with `N%128==0`). It never worked anywhere — it only failed to raise;
   > the one thing that ever raised was a *shape* check on an unpadded plane.
   > There is no safe shape at which to skip the swizzle. Independent
   > cross-check from the same probe: `blocked_scales` is byte-for-byte equal
   > to torch's own `to_blocked` on every shape tested.
   >
   > This was my own inference from "the aligned case didn't raise", written
   > into a docstring as if measured — the same shape as the retracted E2M1
   > bridge blocker, one section up.
2. **`scale_result=`** is accepted and **not applied** (relative error 1.00).
   So `global_scale_real` — with the A-side `1/input_global_scale` correction
   the CB lane applies via `nvfp4_activation_contract.reciprocal_vector` —
   rides one scalar epilogue multiply instead.

**The rule this establishes: on this op, acceptance is not correctness.**

## 3. Residency, measured for fp4 rather than inherited

fp4 decode is **206.4 µs** per 4096×4096 tile (9 reps, 0.8% spread), flat in `M` (the E4M3 lane's
188 µs is a different family at a different rate — it was not inherited). So
the mode is again a required declaration with **no default**
(`GRIDBOOK_TRELLIS_E2M1_MODE`), and the no-default rule is falsified in the
gate. M=512, N=K=4096, **9 paired interleaved reps**
(`both_lanes_reps.py`; medians, spread in brackets):

| mode | resident bpw | forward | vs bf16 | calls/s | plateau W | calls/J |
|---|---|---|---|---|---|---|
| `resident` | ~4.5 | 59.5 µs [1.3%] | **5.20×** [5.17–5.39] | 15 549 | 94.6 | **164.4** |
| `streamed` | wire (2.0 @R512) | 282.1 µs [0.9%] | **1.10×** [1.09–1.14] | 3 337 | 91.5 | 36.5 |

> **The `streamed` bpw column was false when first written**, on this lane and
> on E4M3, and it is the column the mode exists for: a streamed layer held the
> loaded parameter **and** `prepare_wire_cuda`'s private device clone of the
> same bytes **and** a full decoded tile *per layer* — ≈ 2×wire + 4 bpw, more
> than resident's ~4.5. Fixed by dropping the parameter in streamed too and
> pooling the decode scratch device-wide (`gridbook/trellis_decode_pool.py`);
> both halves are falsified in the gate. The per-layer test that existed could
> not see it — each layer's buffer was correct in isolation, and only the
> *total* was wrong.

Resident is **4.51× the work per joule**. Neither saturates the box (67.6% /
65.4% of the ~140 W envelope). The two branches differ less than on E4M3
because a resident fp4 tile is only ~4.5 bpw, not 8.

> **Two corrections to the single-shot numbers this table replaces (5.12× /
> 1.06×), both about precision rather than direction.**
>
> 1. **The bf16 denominator is not stable across processes.** Same shapes, same
>    box, same session: 255.1 µs in one process and 309.5 µs in another (21%),
>    while every lane arm holds to <1.4% within a run. So the ratio is
>    **4.4–5.2× depending on the draw**, and any single-shot ratio against bf16
>    on this box is quoted with precision the instrument does not have. Ratios
>    are only meaningful measured in the same process, as these are.
> 2. **These still EXCLUDE the A-side quant the lane must pay and bf16 never
>    pays. They are CEILINGS, not results.** And on E4M3, where the A-side *can*
>    be measured, that exclusion is worth **2.9×**: the GEMM-only ratio is 2.81×
>    and the full forward is 0.96×, because the A-side quantize costs 1.56× the
>    GEMM it feeds (§ the E4M3 write-up). Expect an exclusion of the same order
>    here. **Do not quote 5.20× or 1.10× without this sentence.**
>
> The A-side cannot be measured for E2M1 outside the container at all:
> `native_fp4_quant` is one of vLLM's compiled ops and raises
> `NativeKernelUnavailableError` in the build venv. Power holds were
> **sequential** (46→65 °C), so absolute watts carry a thermal confound; the
> ratio is direction-robust.

## 3b. It is loaded by vLLM now — and the container run found a real defect

`gridbook/trellis_scheme.py` + a dispatch arm in `config.get_quant_method` +
`tools/make_trellis_smoke_checkpoint.py` close the load path (details in the
[E4M3 write-up §5b](trellis_e4m3_lane_2026-08-29.md), which owns the shared
carriage decision: one self-describing `wire_bytes` blob per Linear, every
scale derived from it, and E2M1's `input_global_scale` the sole loaded
exception because it is the only A-side fact).

**The first container run failed, and the CPU gate could not have caught it.**
vLLM's production `native_fp4_quant` returns the packed nibbles as plain
`uint8`; `_scaled_mm`'s block-scaled fp4 path requires **both** operands to be
`float4_e2m1fn_x2` and rejects the pair outright — `RuntimeError: Invalid
scaling configuration` — when only B is. The test stub returned
`.view(torch.float4_e2m1fn_x2)`, a difference that looks cosmetic and was
load-bearing: **the stub was easier than production, so the gate had a hole
exactly where the unmeasurable A-side is.** Both are fixed: the lane
reinterprets (never copies), and the stub now returns what production returns,
so the branch is exercised on CPU too.

This is the same lesson as the lane's own §2 in a new place: on this op,
acceptance is not correctness — and on this *family*, a stub is not the op.

With that fixed, both residency modes load and serve exactly: 2 arms x 4
trellis Linears, every row `codes EXACT, scale EXACT` against the wire
re-derived from the checkpoint blob
(`dq-runs/trellis-serve-20260829/all4.log`). The generated ids are a
degenerate `[483]*8` -- the smoke checkpoint has random weights, so nothing
about generation quality is claimed.

## 4. What this lane is not

- **The A side is unpriced for QUALITY, and it bites harder here than on fp8.**
  This is
  **W4A4** (`e2m1_group16_ue4m3_static`). Every trellis quality number that
  exists — both ladders, the 24-tensor sweep, the four-family menu — is
  weight-only corpus SSE, which prices W\*A16. **Nothing in the menu predicts
  this lane's quality.** A 4-bit activation is a far larger perturbation than
  the fp8 one.
- **Portability does not transfer from the E4M3 lane.** fp4×fp4 is
  Blackwell-only; the AMD / pre-Blackwell argument belongs to E4M3 W8A8.
- **It is branch (A) made concrete for 4-bit, not the A=W fork resolved.**
- Everything measured is **dense**. No routed-MoE evidence.

## 5. Falsification

Five mutants applied to the **lane**, not to fixtures; suite re-run each time.

| mutant | result |
|---|---|
| `blocked_scales` → plain zero padding | 5 failed |
| `scale_b := ones` | 4 failed |
| resident decodes a zero plane | 2 failed |
| streamed skips `decode_native_packed_out` | 4 failed |
| epilogue scalar dropped | 4 failed |

Restored: **15 passed**. Each check is individually load-bearing.
