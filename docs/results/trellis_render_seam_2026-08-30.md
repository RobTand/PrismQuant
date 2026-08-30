# Trellis render seam evidence — 2026-08-30

Branch: `codex/trellis-render-seam`  
Starting point: skeleton `43278f2`, merged with current `origin/main` through
`d4e05b5` (PR #110 included).  
Design: `/home/rob/dq-runs/trellis-render-design.md`.

## Result

PrismaQuant can now encode and pack a dense TCQ rung from an explicit,
value-bearing plan, publish its exact wire through the one
`ProductionWeightCache`, reload the identical primary bytes, and derive the
matrix used by render scoring and KL validation from that wire. The producer
does not import Gridbook.

The allocation-to-plan handoff, packed MoE, Gridbook-container export and
serving remain refused. The pinned runtime publishes no trellis eligibility
attestation, so both research lanes carry structured
`route_status=unattested`; no code promotes that status to `backed`.

## Exact-byte evidence

The non-trellis vectors below were captured in the detached pre-implementation
worktree at `e9c30a6`, then frozen in
`test_every_existing_render_path_is_byte_identical_to_pre_change`. The current
renderer produces the same byte sequence both with its historical call shape
and with the new `trellis_wire_out=None` keyword:

| format | SHA-256 of rendered tensor bytes |
|---|---|
| BF16 | `6c7a110784ebed6d9b1ea72fb46d1494c5144d68306dfd0f77d4d39a1a931c3f` |
| FP8_E4M3 | `23ae708a2781234c09578a8beb09d6104dc153da2ae8dba2f991c4fdb1f2efb0` |
| MXFP4 | `f8e64c110e16a4c9e4b6c33f66cbd58b887c01ba8d83c353dc59f1267d8c53ef` |
| NVFP4 | `cdeda44496e21512642a2762a6b6055f4f35d2e39a990b898e4c27344a889aed` |

The independent wire-v1 agreement test reproduces Gridbook's frozen real
Stage-5 eager-Viterbi vector (seed 20260825): 307 wire bytes, parsed/repacked
byte-identically, with decoded-code SHA-256
`289f28f80580fcd7565401b9b1f0d9c79451a31b7f46e76d98dbc4b1c5b78e61`.

The central cache test renders once, stores a one-dimensional `uint8` shard,
compacts/reloads it, exercises the decoded LRU view, and checks
`get_wire_blob(qname, fmt) == sink.blob` at every step. A resume test replaces
the encoder with a function that raises and proves a complete admitted pair is
reused without re-encoding.

## Native A=W validation

The pair sidecar binds the exact native activation contract and, for E2M1, the
positive static activation global stored outside the wire. KL validation
authenticates that identity before its one-way model mutation and installs an
activation-only hook:

- E2M1: served group-16 E2M1 + UE4M3 static-scale QDQ (W4A4).
- E4M3: dynamic per-token E4M3 QDQ (W8A8).

The integration test constructs a teacher from the native W4A4 forward. The
validator returns KL below `1e-7`, records one trellis hook call, and the same
rendered weight with raw A16 has KL above `1e-6`. Missing wire identity, wrong
contract/shape/scale, ambiguous aliases and packed targets refuse before the
student forward.

## GPU agreement

Known-good producer image:
`prismaquant-qwen38-producer:20260827-tf516-hf128`.

Command shape:

```text
docker run --rm --gpus all --ipc=host --network none \
  --user "$(id -u):$(id -g)" \
  -e HOME=/worker-state -e TRITON_CACHE_DIR=/worker-state/triton \
  -v /home/rob/pq-wt-trellis-render-codex:/repo:ro \
  -v /home/rob/pq-trellis-docker-cache:/worker-state \
  --workdir /repo --entrypoint python3 \
  prismaquant-qwen38-producer:20260827-tf516-hf128 \
  -m pytest -q \
  tests/test_trellis_render_seam.py::test_triton_encoder_matches_eager_primary_wire_on_cuda
```

Result: `2 passed` (E2M1 and E4M3); each Triton primary wire equals the eager
reference byte-for-byte. This is a correctness gate, not a throughput claim.
No speed, energy or optimal-memoization claim is made, so no utilization or
work-per-joule conclusion is drawn.

## `sb_chunk` bounded experiment

With determinism off, rows=4 and otherwise identical recipe values, chunks 1
and 4 produced identical wire hashes on three seeds per family:

| family / encoded rung | seed | chunk 1 = chunk 4 SHA-256 |
|---|---:|---|
| E2M1 / R512 | 11 | `f481fbded871bd9edce4e6a68d83d9186f081d013ffe82fad8fb0aff0cc34f81` |
| E2M1 / R512 | 23 | `6a20a599fb0a9f450bf87bfd3fee3687e3f7a54b84ce7a4c5ddcf32f989292f6` |
| E2M1 / R512 | 47 | `1479532681e6caa0a3e50d7423c82c2c0d7dba1c5132c39137c62b37421dcddb` |
| E4M3 / R1024 | 11 | `853e94f3ca40a1f08e951c245d1cbccc6f754b0f4099299021eaad239da2f1f2` |
| E4M3 / R1024 | 23 | `9854e264ca11f9915450ce381d7795bbd5b10c421d6e2bb8588e882373f0128a` |
| E4M3 / R1024 | 47 | `1ef294f2ef937b3f8f7451582b736f6bf7fdee90760ca885685af7d9935b54af` |

This is a bounded null result, not a proof of invariance. `sb_chunk` remains in
the recipe/cache identity because chunk-local fp32 normalization and the fixed
`BIG` penalty still make a counterexample possible.

## Design §9 disposition

1. `sb_chunk`: bounded null result above; universally unresolved, retained in
   identity.
2. Packed MoE: unresolved and explicitly refused; no row-shared schedule or
   runtime attestation exists for `[E,out,in]`.
3. Decoded memoization: implemented as a derived LRU view and charged together
   with the primary wire in the resident budget. Whether decode-on-get is
   faster/cheaper remains unprofiled and no performance claim is made.
4. `col_weights`: the exact vector digest/shape is bound and a mismatch
   refuses. Whether a campaign should use the trellis importance vector or CB
   imatrix remains a campaign decision; the code does not substitute one.
5. Container id: unresolved. Native compressed-tensors now gives the accurate
   wrong-container + `unattested` refusal; no Gridbook exporter is invented.
6. E4M3 joint scaling: the qualified contract remains one fp32 scale per row
   with `global_scale_real=1`; `joint_global_real` is explicitly refused.
7. Uncommitted Gridbook contract v12: ignored as attestation. Only the pinned
   packaged contract is consulted; it has no trellis table, hence
   `unattested`.

Additional skeleton gaps resolved: hashes alone cannot construct a plan, so a
versioned exact plan set is mandatory; E2M1's A-side scale is bound outside the
wire; mutable plan mappings are revalidated at the render trust boundary; and
the independent codec is gated by the frozen Gridbook vector without importing
the runtime.

## Test receipts

- Affected surface: `418 passed, 4 skipped, 168 subtests passed`.
- Serving profile/lane follow-up: `67 passed, 1 skipped`.
- CUDA eager/Triton agreement: `2 passed`.
- DSv4 source closure: cache source SHA-256
  `dcf9fa3a1d08186ee37984929396637641e90c0c82ab36650fb49211f73be5f3`,
  with an explicit 2026-08-30 reviewer note in
  `dsv4_w8a16_export_handoff.py`; its targeted handoff tests pass.
- Mutation checks (detached worktree at `68cf1ef`):
  1. Changed the sink length predicate to `False and len(blob) != expected`.
     `test_sink_is_single_assignment_and_checks_priced_length` failed because
     the undersized blob did not raise.
  2. Changed the durable shard write from `torch.save(wire, ...)` to
     `torch.save(decoded_tensor, ...)`.
     `test_wire_blob_is_primary_across_store_load_and_lru` failed on the
     observed `bfloat16` matrix instead of a one-dimensional `uint8` wire.
  3. Changed the KL pre-hook from `quantize(value)` to `value`.
     `test_inplace_kl_executes_trellis_a_equals_w_hook` failed with KL
     `2.9145763619453646e-05` against the native teacher (required `<1e-7`).
  Each mutation was reverted and the detached worktree returned clean.
- Full suite before failure set (current PR #110 baseline): empty.
- Full suite after failure set: pending queued receipt.

No subagent or Fable consultation was used.
