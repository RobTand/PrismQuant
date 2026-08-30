# QTIP basis × E2M1 trellis producer status — 2026-08-30

**Status: deterministic physical one-Linear research producer; production
remains ineligible and fail-closed.** This is opt-in and unregistered. It
changes no pipeline default, format registry, runtime pin, Gridbook production
contract, or serving claim.

## What is now executable

`research/qtip_native_nvfp4_2026-08-30/trellis_online_hadamard_producer.py`
implements the producer-owned half of Gridbook's research ABI at Gridbook
commit `84a78c745e53676f87397e937456d7f2fc6ddd3f`:

- the exact `gridbook.qtip-online-hadamard.v1` closed field set;
- the SHA256-counter Rademacher generator, including its role domain, little-
  endian dimensions/seeds/counters, LSB-first bits, tail masking, and sign
  vector digests;
- the canonical transform digest over algorithm, normalization, padding,
  dimensions, block geometry, seeds, sign generator, and sign digests;
- the normalized block-Sylvester operations in the same row orientations as
  Gridbook: `x D_in H_in` before the W4A4 lane and `y H_out D_out` after it;
- the offline basis pair
  `W_tilde = R_out W R_in.T` and
  `H_tilde = R_in H R_in.T`; and
- a post-decode matrix gate proving that a supplied transformed-basis `Q`
  serves identically to its original-basis matrix
  `R_out.T Q R_in`, plus the corresponding Hessian-proxy invariance.

The cross-repository vectors are pinned rather than derived from a Gridbook
import. For `(rows, columns)=(8,16)`, blocks `(4,4)`, and seeds
`(0x1234,0x5678)`, the complete transform digest is
`3231ab8ed01b068864b261a369a9cfda3e3e59ffbabd5bd0b1bb854c1ec844b5`.
The independent 19-coordinate input sign vector remains
`c9850b2a7c2d365cdb23964ff33c6e934fd3dde47bedb07b35eb9ba8823f6368`.

Both producer entry points require the exact opt-in token
`qtip_trellis_online_hadamard_one_linear_v1`. Its receipt names only
`TCQ_E2M1_R256`, `gridbook.trellis.wire.v1`, and the existing group-16 E2M1 /
E4M3 scale terminal. It explicitly records `qtip_bitshift_wire_allowed=false`,
zero format registrations, and no production eligibility. Preparation has a
null wire identity because it has not encoded. The combined call returns
immutable physical bytes and a non-null SHA-256 only after same-byte parsing
and reference decoding succeed.

## Closed repository seam

The authoritative implementation was promoted from the repository-owned
`origin/codex/trellis-render-seam` branch at `57e2cc0` rather than copied from
a dated run or imported from Gridbook:

- `prismaquant/trellis_encoder.py` owns the exact 256-state tail-biting
  Viterbi encoder and emits coded-bit, point-index, bypass-code, and scale
  planes. Its default scale path is unchanged; the BlockLDL follow-up adds an
  explicit positive E2M1 global-scale override, with a test proving that the
  default one-block global supplied explicitly reproduces identical bytes;
- `prismaquant/trellis_wire.py` independently packs, validates, parses,
  decodes native codes, and reconstructs values for
  `gridbook.trellis.wire.v1`; and
- `prismaquant/trellis_producer.py` is the small versioned shared API. It
  encodes once, packs once, reparses and reserializes the exact bytes, decodes
  only from those bytes, and binds recipe/source/wire/code/value identities in
  `prismaquant.trellis_one_linear_producer.v1`.

The independent codec test retains Gridbook's frozen real Stage-5 vector:
307 exact wire bytes and decoded-code SHA-256
`289f28f80580fcd7565401b9b1f0d9c79451a31b7f46e76d98dbc4b1c5b78e61`.
It imports no Gridbook package. The producer additionally requires its
canonical parser to reserialize the same bytes and requires the decoded value
view to equal the encoder reconstruction at the BF16 validation boundary.
FP32 maximum absolute rounding delta is recorded because serialization of the
global scale can change the association of the final FP32 multiplication.

`require_combined_wire_round_trip()` now passes `W_tilde` to that API, derives
the encoder's explicit column-importance vector as `diag(H_tilde)`, binds the
complete online-transform object, decodes `Q_tilde` from the emitted bytes,
and verifies
`x R_in.T Q_tilde.T R_out == x (R_out.T Q_tilde R_in).T` by reference.
The combined receipt is
`prismaquant.research.qtip_trellis_online_hadamard_artifact.v1`.

## Full off-diagonal feedback follow-up

The research module also exposes
`require_blockldl_trellis_wire_round_trip()`. It factors the transformed
Hessian as `H_tilde = L D L.T` with unit 256-by-256 block-lower `L`, then
processes physical trellis superblocks in reverse:

```text
A_j = W_tilde_j + sum(k > j) (W_tilde_k - Q_k) L[k,j]
Q_j = trellis_terminal(A_j)
```

For `E = W_tilde - Q`, this gives `(E L)_j = A_j - Q_j` and therefore the
exact algebraic decomposition
`tr(E H_tilde E.T) = sum_j tr((A_j-Q_j) D_j (A_j-Q_j).T)`. The implementation
checks both the factorization and this objective identity, and compares the
QTIP-shaped buffered recurrence against an unbuffered oracle.

The atomic block is 256 input columns, not QTIP's original 16-output by
16-input tile. Gridbook's cyclic trellis state spans one output row by 256
input positions; committing 16-column fragments would cut that cycle and
produce a different codec. Schedule and alphabets are fixed before feedback.
One E2M1 tensor-global scale is selected from the complete transformed weight
and frozen across the reverse pass. Per-block `[rows,16]` E4M3 scale planes
are assembled in row-major `[rows,blocks*16]` order before the planes are
packed once into one wire. Each feedback step first packs and reference-decodes
its terminal bytes; that same-byte decoded tensor, never the encoder's
pre-serialization float reconstruction, feeds the earlier blocks. The final
assembled wire must decode FP32-exactly to those terminal decodes.
Temporary terminal wires use each block's local schedule sum and only its
locally used shaped-rate alphabets. The final wire retains the declared
tensor-wide rate and complete alphabet union; tests cover fixed-quota blocks
with differing rate sets and tight-offset blocks with local rates 368 and 656
that average to tensor rate 512.

Two honest additive terminal metrics are available:

- `qtip_frobenius` matches the pinned QTIP recurrence's unweighted terminal;
- `diag_block_D` uses the diagonal of the correct local LDL block `D_j`.

Neither mode claims to minimize the dense `D_j`. A general local cost contains
pairwise residual terms `2 D_j[s,t] e_s e_t`; those are not summarized by the
current convolutional eight-bit state or its coordinate-additive Viterbi
cost. `dense_block_D` therefore fails closed. Receipts record
`terminal_dense_D_consumed=false` and `terminal_dense_D_exact=false` while
separately binding the complete `L`, dense `D`, feedback targets, decoded
terminals, shared scale, encoder source, final wire, and transform metadata.

At width 256 there is no cross-block feedback. A full-feedback experiment
requires at least two physical superblocks; the API remains valid for one but
does not reinterpret its dense `D_0` as information consumed by the terminal.

## Remaining boundaries

- The simple non-BlockLDL entry point still consumes `diag(H_tilde)` and
  records `full_off_diagonal_blockldlq_applied=false`. The new BlockLDL entry
  point consumes all cross-block feedback in `L`, but its terminal remains an
  explicitly labelled additive approximation to each dense local `D_j`.
- The reference artifact is one dense Linear only. It is not connected to the
  production cache, allocator handoff, exporter, or Gridbook loader.
- The physical wire proves producer codec closure, not served native W4A4
  speed, graph capture, residency, quality, or work-per-joule.
- QTIP bitshift serialization remains forbidden. No EXL3 or Quartet format is
  implemented. PrismaQuant still does not import or vendor Gridbook.

No production registration is justified until the separate Gridbook release,
physical load, graph, quality, residency, and performance gates pass.

## Verification

CPU reference command:

```bash
/home/rob/venvs/pq-cu130/bin/python -m pytest -q \
  tests/test_trellis_formats.py tests/test_trellis_rate_surface.py \
  tests/test_trellis_allocator.py tests/test_trellis_menu.py \
  tests/test_trellis_wire_codec.py tests/test_trellis_producer.py \
  tests/test_qtip_trellis_online_hadamard_producer.py
```

Result after the BlockLDL follow-up: `111 passed`; the separate documentation
and architecture guards add `20 passed` (`131 passed` together). This is codec
and reference-algebra evidence only. No GPU performance, energy, graph, or
production-serving claim is made.
