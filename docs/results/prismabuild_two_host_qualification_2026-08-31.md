# PrismaBuild V4 two-host CPU qualification — 2026-08-31

## Result

The opt-in PrismaBuild initial-miss rendezvous passed its exact configured
two-host/shared-NFS CPU qualification. Both `gx10-6b77` and Sparky ran the same
immutable source snapshot, bound the same rendezvous manifest and configured
CAS root, published the exact closed arrival/ready sets, and returned matching
proof receipts. A fresh process subsequently reopened the frozen run root and
rebuilt the complete committed predicate with the same PASS result.

This is causal evidence: both exact `run-local` processes observed their
initial CAS miss before either in-protocol result publication. It does not
claim simultaneous task-argv execution and uses no cross-host clock ordering.

## Source and frozen harness

Qualified source:

- branch: `codex/prismabuild-initial-miss-rendezvous-20260831`
- commit: `452c6f68abead2bf1736000bd771e14ad31f76c2`
- tree: `e7b31d03219a3a4e0bd797e991f9ff755749b219`
- runtime closure:
  `266b33b0b7cd41465d02d7789ab7612514e3a3e825084513e7c93a97d706974f`
- closure entries: 1,223

Frozen harness root:
`/home/rob/dq-runs/prismabuild-two-host-qualification-harness-v4-452c6f6`

| Artifact | Mode | SHA-256 |
|---|---:|---|
| `ARTIFACT_MANIFEST.json` | `0444` | `52f2935c737ff852c01028bf69b3e65c2ad2b0dbd1ea9c85be16c446b87f9e3f` |
| V4 harness | `0555` | `281b8aab03feedb3bf84c4c895a099f3c9fdb45c6717a5d61cbaec7566021ff6` |
| hostile tests | `0444` | `2b0c4875205c7453c5f0a9731605a34cc248933cff10c249edf0be244a13cde9` |

The artifact root was `0555`, contained exactly those three single-link files,
and had entry-set digest
`044ace58632c8b0c65a20320c281d52584adfed88711efe14c0d3b85ff6f9288`.
Its suite passed 154 tests. A separate no-context review reproduced 426 passes
and one optional-dependency skip across the frozen harness, source, docs,
Slurm, and Dagster checks, then returned explicit GO for the live CPU command.

## Command and immutable evidence

```bash
/usr/bin/python3 -I -S -B \
  /home/rob/dq-runs/prismabuild-two-host-qualification-harness-v4-452c6f6/run_prismabuild_two_host_nfs_qualification_v4.py \
  --run-id run-v4-452c6f6-20260831-075539
```

Frozen run root:

`/mnt/shared/prismaquant-prismabuild-validation/452c6f6/run-v4-452c6f6-20260831-075539`

| Authority | SHA-256 |
|---|---|
| `COMMITTED.json` | `50691670b414b24815f7080a1daf38c12a86ea25c7b719aa457cd285e06644c9` |
| `qualification.json` | `645a4388e903d93238d84c8851e4b0190c6928dc23247a72b264bc011f1dfc8e` |
| protocol | `5a9e094eff09a201ada87767c81ce8ce0a13c892b69b3387439a3417e7d9ddee` |
| post-use runtime verification | `5aac9353cc7e40a0f455c7f9f417f8b7ad6427529942ecef0ac389f8e67be99b` |
| active-host manifest | `48d9ad163e3e9e4cc05718f8245a01d38b1029728c928115526acf0ce1924ec1` |
| Sparky manifest | `48d9ad163e3e9e4cc05718f8245a01d38b1029728c928115526acf0ce1924ec1` |
| manifest entry set | `c076cf5b8e894f2fd77d624ab646a83739c94df82f72008add2008dac12754f4` |

The run root is `0555`; `COMMITTED.json` and `qualification.json` are `0444`.
The post-run verifier reconstructed every nested authority, both exact host
manifests, and the root closure with only the committed marker self-excluded.

## Scope

This result qualifies the exact configured pair and shared-NFS path. It is not
a claim about other mounts or filesystem implementations. Unkeyed hashes prove
integrity under the cooperative namespace trust model, not authorship; deployed
ACL/WORM isolation remains required. Host power loss, an indefinitely blocked
filesystem syscall, live Slurm adoption, daemon operation, hostile task
containment, production payloads, GPU kernels, speed, and energy remain outside
this result. No GPU work was launched.
