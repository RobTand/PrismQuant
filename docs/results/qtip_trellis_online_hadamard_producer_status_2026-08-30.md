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
  planes;
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

## Remaining boundaries

- The existing trellis encoder consumes a nonnegative column-diagonal
  importance vector. This scaffold supplies `diag(H_tilde)` and explicitly
  records `full_off_diagonal_blockldlq_applied=false`; it does not fabricate a
  claim that the encoder consumes the complete transformed Hessian.
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

Result: `100 passed` in 5.89 seconds. This is codec and reference-algebra
evidence only. No GPU performance, energy, graph, or production-serving claim
is made.
