# Bounded shared-server cache warm-up — 2026-09-07

Rob asked whether known future inputs could be cached on dl380g10 before the
GPU stage needs them. The existing
`source_prefetch.prefetch_files_to_page_cache` can perform that anticipatory
read as an ordinary admitted PB action. It retains no tensors or additional
cache. `/mnt/shared` is backed by ZFS on this host: the read warms its ARC,
not a guaranteed pinned allocation. Future eviction and the Spark's own
NFS/application prefetch remain separate concerns.

At inspection, ARC held about 224 GiB with a configured 240 GiB maximum.
Linux reported roughly 54 GiB available and zero memory PSI. Reading the whole
roughly 598 GiB GLM checkpoint would exceed this cache; prospective warming
should use an explicit next-stage input set and a bounded memory budget.
There is no implemented PB speculative-cache policy inferred from this run.
The current pipeline's existing residency and integrity gates remain required.

A single admitted warm-up read the next LFM source checkpoint,
`/mnt/shared/models/LFM2.5-8B-A1B-BF16/model.safetensors`: **16,936,006,912
bytes in 6.6848 seconds**. Source file size/mtime were checked before and after.
PB reserved 1 CPU and 20 GiB, with native thread limits of one; the helper's
file budget was 18 GiB and an explicit preflight required 36 GiB available.
The source is one file, so the existing reader used one worker. No cache was
dropped, no model payload changed and no cold-cache benchmark ran.

During that window, global ARC demand-data counters increased by 16,660 hits
and 23 misses; prefetch-data misses increased by 258. This suggests the file
was largely warm already. These are host counters, not per-file attribution.
The result establishes a completed anticipatory read, not a downstream
speedup, permanent residency or an end-to-end bottleneck diagnosis.

PB action:
`a3d209e3a8bc438f4842e35fbcf81f471f30cb9e02c6590b8f0ebba856a12498`.
Actual exit 0, logs, source bundle, complete cleanup, CAS receipt and artifact
hashes were independently verified. CAS receipt SHA256:
`ea808b055d30eb32ab9fd7504d5df917ed6ab9bb097adc71188965d28e50ccd0`;
output SHA256:
`2c43d59f06dbf5bd64c627a9189a0de9248077b0c96cfa4a4ddbf7f4b253a734`.
The sealed PB request retains the exact invocation. Evidence under
`/mnt/shared/tessera-measurements/first-model-20260907/server-prewarm/lfm-source-01/`
contains `receipt.json`, `reader.pstats`, before/after ARC, memory, pressure,
block-I/O and process-I/O counters, plus `netdata-three-hosts.json` (17 raw
series covering dl380g10 and both Sparks). Raw Netdata SHA256:
`077a9fe12575bb52fa264cd032891ba97ed1635077576f7d50cf3e6e41f1f3f4`.

Inspection also found an unavailable configured L2ARC device. The four data
disks were online with no known data errors. The stale-name hypothesis and
exact GUID/path evidence are recorded in PrismaBuild issue 361; candidate
label inspection required unavailable administrator authentication, so no
storage configuration was changed. RAM-cache warming remains usable.

A separate code fix in this branch closes an existing exhausted-budget hole:
a zero automatically computed budget used to bypass the size check. Eight
bounded CPU regression cases reproduced it before the fix; after the fix,
67 prefetch/PWC tests passed. Auto mode now skips and require mode refuses
positive-sized input sets when no usable automatic budget exists. This
changes no positive-budget behavior and makes no performance claim.
