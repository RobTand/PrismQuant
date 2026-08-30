# QTIP basis × E2M1 trellis producer status — 2026-08-30

**Status: deterministic one-Linear contract scaffold; physical combined
producer is blocked and fail-closed.** This is research-only, opt-in, and
unregistered. It changes no pipeline default, format registry, runtime pin,
Gridbook production contract, or serving claim.

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

The preparation entry point requires the exact opt-in token
`qtip_trellis_online_hadamard_one_linear_v1`. Its receipt names only
`TCQ_E2M1_R256`, `gridbook.trellis.wire.v1`, and the existing group-16 E2M1 /
E4M3 scale terminal. It explicitly records `qtip_bitshift_wire_allowed=false`,
zero format registrations, and null physical-wire identities.

## Exact blocker

PrismaQuant's repository currently provides:

- E2M1 trellis family/rate, schedule, alphabet, and native-code validation in
  `prismaquant/trellis_formats.py`;
- pre-render wire footprint accounting in
  `prismaquant/trellis_footprint.py`; and
- the separate one-Linear QTIP-style BlockLDLQ/native-NVFP4 isolate in
  `native_nvfp4_ldlq.py`.

It does **not** provide one repository API that returns all of:

1. the tail-biting Viterbi coded-bit, point-index, and bypass-code planes;
2. physical `gridbook.trellis.wire.v1` bytes packed from those exact planes;
3. a reference parser/decoder of those same bytes; and
4. a content identity binding the encoder, packed wire, decoded native E2M1
   codes, group-16 E4M3 scale plane, and decoded transformed matrix.

The historical encoder and wire packer exist in dated run-directory sources,
and Gridbook owns its current reader/decoder, but neither is a supported
PrismaQuant repository interface. PrismaQuant may not import or vendor the
Gridbook runtime. Copying the run-directory implementation would create a
second unauthoritative wire owner. Substituting QTIP's bitshift codebook would
change the requested carrier. Accepting a caller-provided decoded tensor would
not prove that the tensor came from the claimed bytes.

Therefore `require_combined_wire_round_trip()` unconditionally raises
`MissingTrellisProducerSeam`. The post-decode algebra test is labelled
`wire_identity_verified=false`; it must not be reported as a physical wire
round trip.

## Seam required to unblock

The next implementation should expose the existing encoder through one
versioned, deterministic producer interface that returns immutable wire bytes
and a reference decode from those bytes. It must bind the existing generator
polynomials, state count, tail-biting proof, schedule, alphabets, native E2M1
codes, E4M3 scale bytes, global scale, and wire schema. The combined producer
can then pass `W_tilde` and `H_tilde` to that interface, insert the already
validated `online_transform` object into the Gridbook scheme, parse/decode the
emitted bytes, and run the existing post-decode serve-algebra gate.

No EXL3 or Quartet implementation is needed for that seam. No production
registration is justified until the separate Gridbook release, physical load,
graph, quality, residency, and performance gates pass.
