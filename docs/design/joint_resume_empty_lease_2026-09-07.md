# Joint AURA checkpointed-layer lease correction — 2026-09-07

A partial streamed joint-AURA resume can have no pending units in the first
reverse layer while earlier layers remain incomplete. The caller previously
constructed `SignedJointProjectionLease` with an empty target map. Its fallback
device is CPU, which a CUDA-prewarmed fused projection backend correctly refuses.
The default Torch backend accepts that device, so the existing interrupted-resume
test did not expose this failure.

The caller now constructs a projection lease only when `pending` is nonempty.
Every reverse layer still runs the full input/shared-state cotangent traversal;
probe ordering, Fisher construction, GEMMs, QDQ, signed accumulation and squaring
are unchanged. The backend's device admission remains strict. Existing producer
source identity checks remain strict too: this commit does not authorize reuse
across source versions or rewrite any retained production checkpoint.

## Regression and integration evidence

The CPU regression interrupts the existing two-layer fixture after persisting
its last reverse layer, then resumes with the earlier layer still pending.
It uses the real fused backend device gate with logical CUDA targets for
populated leases and the existing CPU fallback for empty leases; numerical
projections execute through the Torch oracle. It does not load a native GPU
extension or claim CUDA numerical qualification.

Both ordinary and microbatched variants verify:

- exactly one reused unit and one newly written unit, with preserved checkpoint
  bytes unchanged;
- only the pending earlier layer acquires a projection lease;
- all six reverse input cotangents equal the uninterrupted run bit for bit;
- complete cost rows, signed terms, probe/arithmetic identity, gradient traces
  and Fisher column diagnostics equal the uninterrupted run exactly.

PB `c701712f3528` ran those two regression variants before the fix: both failed
at the fused device gate on the empty lease's CPU device. Exit 1 and completed,
released scope were verified. Failed actions publish no success CAS receipt;
actual terminal and stdout/stderr are retained.

PB `590484b22080` ran the following after the fix: **120 passed**, zero skips
or missing collection, with 56 upstream Torch/Python deprecation warnings.

```text
bash experiments/pq322_cpu_checks.sh \
  tests/test_joint_aura_streamed.py tests/test_joint_aura_microbatch.py \
  tests/test_joint_aura_packed.py tests/test_joint_aura_projection.py \
  tests/test_joint_projection_backend.py tests/test_streamed_cost_checkpoints.py \
  tests/test_architecture_doc.py tests/test_docs_staleness.py \
  -q -n 4 --dist worksteal -p no:cacheprovider
```

Both actions were portable CPU submissions at priority -10, admitted on dl380g10
with four CPUs/eight GiB and native threads bounded to one. Runtime: Python
3.14.4, Torch 2.11.0+cpu, Transformers 5.16.1, pytest 9.0.2. CUDA was disabled.
The green terminal exit, cleanup, actual CAS payload and canonical receipt hash
were independently verified. Payload:
`8e7033407cecca867cb3ae6d6533a59e022c3347869458f9c4761c7f3ccaa88c`;
receipt:
`05952743bfada4bbd9158717f8f1c5ad4973fc69c3123bd42bb7f3e5ec4a85ce`.

Evidence root:
`/mnt/shared/tessera-measurements/joint-resume-empty-lease-20260907/`.
`red-verified-01.json` and `green-verified-01.json` bind terminal records and
actual logs. No frozen production source/checkpoint bytes were changed and no
GPU work or performance measurement was repeated.

PB `a8aab10d7e8c` separately compiled both touched Python modules and verified
that inverting exactly the guard/comment hunk reproduces the original full
`aura_cost.py` bytes. It ran on dl380g10 with one CPU/one GiB, exit zero and
released scope. `compile-inverse-verified-01.json` records the checked CAS
payload and receipt. Final `aura_cost.py` SHA256:
`b615fa48377daca8b7264c96a5e5dd853cc963ba1d4c05c1519d3027264c0852`.
