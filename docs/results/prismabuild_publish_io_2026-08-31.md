# PrismaBuild winning-publication I/O profile — 2026-08-31

## Result

A fresh PrismaBuild CAS publisher no longer hashes its own newly published
payload two extra times. The worker hashes while copying to a private,
read-only, fsynced staging inode. If that inode wins the no-clobber hard link,
canonical readback proves exact inode and substantive metadata identity. A
pre-existing race winner, a different stochastic result, and every later
public lookup still receive a full content hash.

On the same active GB10 (`gx10-6b77`), the same warm 2 GiB sparse fixture on
the mounted NFS4 export produced:

| Instrument | Baseline `c897e07` | Change `76ebbb1` | Delta |
|---|---:|---:|---:|
| Publication wall time | 4.905971 s | 3.175261 s | 1.545x; -35.28% |
| `/proc/self/io` `rchar` | 6,442,574,016 B | 2,147,610,401 B | 3.000x; -66.67% |
| `/proc/self/io` read syscalls | 1,545 | 519 | 2.977x; -66.41% |
| cProfile SHA-256 update time | 2.608673 s | 0.866992 s | 3.009x; -66.77% |
| `/proc/self/io` `wchar` | 2,147,485,252 B | 2,147,485,250 B | unchanged |

The logical-read count is the clearest mechanism check: baseline consumed the
2 GiB payload three times inside `publish_result`; the change consumed it once.
The 256 MiB replication independently measured 805,429,439 B versus
268,562,208 B of `rchar`, with wall time 0.652904 s versus 0.395130 s.

## Instruments and retained evidence

The in-process instrument was `cProfile` plus `/proc/self/io`. Netdata raw JSON
and chart metadata were captured at one-second cadence over padded windows on
both `gx10-6b77` and Sparky for `system.cpu`, `system.io`, `system.net`, and
`nfs.rpc`. The exact 2 GiB measurement intervals have seven baseline and six
after samples on each host. The active box shows CPU, NFS-RPC, network, and
write activity; Sparky remains lightly loaded, which is expected because it
was an observer, not the NFS server or benchmark executor.

Evidence root:

`/home/rob/dq-runs/prismabuild-perf-2026-08-31`

| Evidence | SHA-256 |
|---|---|
| Benchmark source | `23920a1a52b6e7d578c86b7f5311598a0ea687f18dc3822c583643b343c6ee64` |
| Netdata capture source | `60b94165069be5a3a62e697876f65afe1d9d6e99fc9c7b739b01043a989b70bc` |
| Baseline 2 GiB report | `1880cda3193aa9a020c5c5d70bf67b9cf585c86c9804d56017f4a5447e769c61` |
| Baseline cProfile | `d422e5361b41d567a97c20ab2edd99060d2da53c0cdc02deae33a11fb01cc68c` |
| Baseline `/usr/bin/time -v` | `c7668809ba45d71cbec5d3abd8e55b8a00fb2c41f722884907b106dc1b9d1081` |
| Baseline two-box Netdata manifest file | `677b37bdbc782a909f4981bf85e8a26453ae76835712256836638f6edfccd38c` |
| After 2 GiB report | `7823aff98b93638be70f1ad01b956f4ecda788c9b76c03d58d69b7bb2d53c6e5` |
| After cProfile | `0022e777efff0ebff1aaaefa31e8f15e8d97d367f4be137b5a2dc1999075881c` |
| After `/usr/bin/time -v` | `8511521908b400037dfce7a6186535bc92aee48edebbabc173702e75cf0d2d56` |
| After two-box Netdata manifest file | `cd18ae4752b4bf8621287b347f33c83f9927987b74eaaae3a56e4336cd45a5f7` |

The profiled commands were:

```bash
/home/rob/venvs/pq-cu130/bin/python \
  /home/rob/dq-runs/prismabuild-perf-2026-08-31/bench_publish.py \
  --repository <exact-worktree> \
  --work-root <fresh-2GiB-NFS-fixture> \
  --profile <exact-profile-path> \
  --output <exact-report-path>
```

The baseline repository was clean commit
`c897e07cf9b140acbe4bbec142f97951e48e16c9`; the after repository was clean
commit `76ebbb1262c27f5bc73dd83c3acf0d96da477b14`. Both report the same payload
SHA-256, `a7c744c13cc101ed66c29f672f92455547889cc586ce6d44fe76ae824958ea51`.
The profiler source itself is identical across runs.

The post-change focused validation command covered the core, Slurm adapter,
and Dagster bridge: 180 passed, one skipped; the architecture/staleness gates
added 20 passes. `py_compile` and `git diff --check` also passed. The skip is
the existing environment-dependent Dagster case, not a publication test.

## Scope and remaining gates

This demonstrates a warm, fresh, winning publication—not a cold-NFS network
throughput claim. Linux reported `read_bytes=0` because the payload was cached;
therefore `rchar` establishes process-level logical consumption and Netdata
establishes box load, but neither is presented as measured NFS byte savings.
The avoided full reads should remove corresponding cold-storage traffic; that
is an inference from the code path, not a measured cold-cache result.

The fixture is 2 GiB, not a representative 90 GiB production artifact. No GPU
was involved, and GPU utilization is irrelevant. This does not qualify host or
power loss, production-scale NFS, malicious same-UID mutation, or ACL/WORM
retention. The winning-inode proof relies on the private staging inode being
read-only after its stable hash+fsync; public consumers do not rely on that
historical assumption and always hash canonical content. A real deployment
still needs the storage trust and durability gates in
`docs/design/prismabuild.md`.
