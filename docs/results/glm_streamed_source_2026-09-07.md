# GLM canonical source streaming qualification — 2026-09-07

The canonical census/capture entry point can use the existing layer streamer
without constructing the complete GLM BF16 model. This qualifies the source
forward and bounded capture semantics; it does not qualify a full GLM capture,
an export or a served artifact.

The fixture uses the real Transformers 5.16.1 GLM implementation, original
checkpoint names, split KDA convolution tensors, unpacked experts and mHC
renames. Native CUDA checks use GB10, PyTorch 2.13.0+cu130, BF16/eager execution,
two original B1 samples of 257 tokens and seven retained prefix rows. Two body
layers cover KDA and DSA, four experts/top two routing, and mHC expansion four;
hidden width is 256. The native gate calls actual CUDA kernels and does not
install the CPU fixture's convolution replacement.

The first valid native comparison failed: shared loading narrowed strict FP32
GLM state, changed routed counts and produced a maximum logit difference of
0.57421875. Source values alone did not expose the defect: several narrowed
values were exactly BF16 representable, but their arithmetic dtype changed.
The shared loader now applies HF's own dtype plan before materialization,
including propagating final convolution precision to its split sources.

| Check | Actual outcome | PrismaBuild action |
|---|---|---|
| Native source versus initial streamed route | Failed counts/maxima/logits | `b479d4…`, retained `gpu-native-02` |
| Diagnostic installed state | Identified strict FP32 narrowing | `381c9eb…`, retained `gpu-native-03` |
| Native source after dtype repair | Exact X/H/count/maxima/logits, 18 units | `07a919fa56ee8736a8a11f53dc37dfe5833f9609ed46de4c1ad4978f8cb3fdd5` |
| Native source with ephemeral metadata and direct packer, nonzero FP32 recurrence/routing controls | All installed state and X/H/count/maxima/logits exact, 18 units | `91bdf285f55b2e2304db79fe2da339c065cf716febecef3ca326356e8bbb04a2` |
| CLI census/capture and source-state tests, CPU | 38 passed | `ef9c6d4a2993155502bdb56e1a1f05b509f3b568007529d28a01aff05ffc11ed` |
| Metadata/source lifetime and pack-layout regressions, CPU | 15 passed | `6aa8b6e756c04f656a83689fe9c208bdd750828ccabdf62604e381985945fbcd` |
| Strict FP32 resident/cold/prefetched/cache replay plus canonical contracts, CPU | 33 passed | `f2a41f64241359cb78b40bf3138266e6e75d06cea028e3200a915959e85768ae` |
| Final architecture/docs, source streaming/prefetch, precision, packing and canonical contracts, CPU; touched-module compile | 84 passed, no skips; compile passed | `f0e450cb779a3822ff466b388b12a62743503d81964091de46dfd8649e1851d2` |

Native receipts, before/after CPU+CUDA traces, both-host Netdata series and
result JSON are retained under
`/mnt/shared/tessera-measurements/glm-streaming-source-20260907/`.
The final native payload SHA256 is
`87603320dafc2a3c28988df5f475525231bcb2f92a3ca18c9f8f944b4df846b4`;
CAS receipt SHA256 is
`c7e3a34d572782523660d803e1778ef24d6abb4cd3b288c995a6ce2bb492416c`.
Terminal exit status and the actual payload bytes were independently checked.
These small qualification timings are not a full-model speed or saturation
claim. Vision and MTP state are outside the text-forward witness and gate.

The representative production source is
`/mnt/shared/models/GLM-5.3-Flash-BF16`: 45 body layers, 42 MoE layers with
288 experts/top eight routing and 36,423 quantizable source Linears. The
proposed 512×512 draw remains unfrozen. A header-only upper bound has roughly
1,709 GiB full H and 237 GiB retained X, plus journal/replacement space; routed
counts need an actual census. Explicit workspace remains an allowance rather
than measured full-model allocation. No full-model census or capture was run
as part of this qualification.

Follow-up compile clarification: the 84-test wrapper compiles its pre-existing
module list; those tests also import the changed production modules. Dedicated
PB action `e6901251714f4d2819b57cef9c1027b05e798fc95806bb3026576336912c24f5`
subsequently compiled all 15 changed/new source, tool, experiment and test
modules, including the new real-layer workspace experiment, and exited zero.
Its verified payload SHA256 is
`5c6b74c22c93c9ca80c9a440a490e4ecac464fc06d6195bdb4e8e5d1e656b601`.

Real-source expert-packer ABBA qualification, 19:02–19:04 UTC, used one
complete paired action per model on Lina's GB10, driver 595.84, CUDA 13.0,
PyTorch 2.13.0+cu130, CPU affinity 5–8, native threads one and four shard
readers. The producer image ID was
`sha256:9f9b9f05b17531399ba66dc6415b054cf5d68c82270626d0e9150e75c808435f`.
Both measurement admissions recorded an empty preceding GPU membership.
Source reads and SHA256 byte audits were outside each profiled packing arm.

| Real source layer | Profiled before seconds (A1, A2) | Profiled after seconds (B1, B2) | Allocated peak GiB, before → after | Reserved peak GiB, before → after |
|---|---|---|---|---|
| GLM layer 3, 288 experts / 864 tensors | 6.1327, 0.7671 | 0.4629, 0.4564 | 36 → 22.5 | 36 → 27 |
| LFM layer 2, 32 experts / 96 tensors | 0.05859, 0.04131 | 0.02909, 0.02942 | 1.53125 → 1.09375 | 2.03125 → 1.59375 |

All original source and final packed bytes, shapes and BF16 dtypes matched in
all four arms for both models. GLM's 13.5 GiB allocated-peak reduction is
37.5%; the reserved-peak reduction is 9 GiB (25%). LFM reduced both peaks by
0.4375 GiB. The final baseline versus first after profiler shows GLM replacing
290 concatenation allocations (22.5 GiB cumulative) with two final allocations
(13.5 GiB cumulative) and 864 direct device copies. Corresponding CUDA work was
204.996 versus 121.088 ms; `cudaMalloc` CPU time was 617.201 versus 333.794 ms.
LFM replaced 34 concatenation allocations with two final allocations and 96
copies; CUDA work was 10.409 versus 5.973 ms. These are profiled packer-only
observations; the first GLM baseline includes a large cold/profiler effect, so
an aggregate wall-speed ratio would misrepresent the experiment.

Both-host Netdata series were recorded without errors. GLM had 86 power
samples per host across the complete action: Lina 4–21 W, Sparky 18–46 W.
No power sample landed within either subsecond GLM after arm; none landed in
any LFM packing arm (four samples per host over its complete action). These
series cannot attribute per-arm energy or rank work per joule. No GPU
saturation, energy-efficiency or full-pipeline throughput claim follows.
PB's GLM cgroup peak was only 14.61 GiB while torch's allocation peak was
36 GiB: cgroup memory alone does not bound GB10 CUDA physical allocation.
The subsequent workspace experiment therefore records separate cgroup,
CUDA-reserved and host-available planes and conservatively guards their sum.

GLM action `316e639634f55dd8be031e0783e80b8725d76eb1b350bb0291d12a4d0eb3d8a7`
completed with exit 0; verified CAS payload SHA256
`988de09f4ef7bc7b2fd3d27ef1d4609e619d466e9e2ff9890a74bf78989b3bff`,
receipt `7b1efc90cbf6f4640e610b58dd0999a9076db3592228fe282aaf819c11519e3e`.
LFM action `2b185c53a48e64fd061b6080dc38e48153e586d681a47e7ff0394ee509592e4c`
completed with exit 0; verified payload
`abbb2863a13cc597727d0fc9d6709a9f3da774fed2fe8c8ab85069cda182cd15`,
receipt `a52bb2d7e9280479ccf5aa2b86fd7e8917158c46b28f16b0f9946a3f0c92bf16`.
Full result JSON, four CPU/CUDA traces per model and telemetry are under the
qualification root's `packer-ab-01/` and `lfm-packer-ab-01/` directories.
The recorded baseline function SHA256 is
`d9eeacfe684472bc2ffa27e529ea2480f195f59fd8b621a52da9dcf238e4db6e`,
from commit `1c7492333`; the complete immutable fixture is checked in.

Final focused CPU checks on the committed packer/resource-plan state completed
on dl380g10: 34 passed, no skips, four xdist workers, Python 3.14.4,
PyTorch 2.11.0+cpu and Transformers 5.16.1. PB action
`4b227765ec04ff04951d29202a460ba803fa83856393da58257adbdf89ca1957`
exited 0; verified payload
`2d8b8946e3b9a2c72db28d98af0f99039a48fadb6316242ad8da73fe498031d1`,
receipt `24162937a96496c4e3d0500d3e9873eeb49fd1184a0206fdae6b8f6b5c9b84f6`.

The subsequent DSA-layer-4 workspace experiment **failed qualification**.
It ran all 512 original B1 samples of 512 tokens with real GLM source weights,
two source-cache slots, next-layer prefetch and 8 GiB of explicitly synthetic
current boundaries (seed 17092026). Experts 1, 48, 60, 98 and 254 received no
routed rows: 15 of 867 requested targets were unobserved. The existing
collector refused before the final full-H transfer to CPU. No routes were
forced and no missing H was fabricated. This is retained partial-workspace
evidence, not a completed capture or a proof that the proposed draw fits.

PB action `380caee08fabf78dbb070806744d7237cf3a6811ca06f97bb5491b5ca9b53c8a`
ran on Lina with CPU 6, physical reservation 104 GiB and GPU subset 92 GiB,
and exited 1. The experiment took 690.45 seconds. Peak CUDA allocated/reserved
was 85.8875/90.5137 GiB; the largest sampled conservative cgroup-plus-CUDA
reservation sum was 96.9208 GiB and minimum host MemAvailable was 18.3401 GiB.
There were zero CUDA allocator retries, CUDA OOMs or cgroup OOMs. Source/H/X
lifetimes after collector return remain unmeasured by this failed action.
Both-host Netdata recorded 523 power samples per host: Lina 4–39 W, mean
28.74 W; its first-two-batch profiler covers cold allocation/setup, with
132.61 seconds CPU and 1.55 seconds CUDA. Neither this cold profile nor the
failed capture establishes steady-state performance or work per joule.

The failed terminal, cleanup evidence and SHA256 inventory of the result,
512 per-batch records, continuous memory samples, CPU/CUDA trace and both-host
Netdata are recorded in `review-02/workspace-dsa-failed.json` under the shared
qualification root. PB did not publish a success CAS receipt for this nonzero
action. The practical lesson is that random residual boundaries do not ensure
real MoE routing coverage, even after 262,144 tokens; a source-prefix fixture
is needed to attempt the missing transfer and cleanup measurement.

Real-prefix input preparation exposed a separate dataset lookup defect. PB
`61405ca3b1c15ff45d24eecaa24edae16b3598598827e8cd0927a4d40344e148`
failed when the producer's installed Hub URI parser rejected the legacy
namespace-free `wikitext` repository alias. The canonical helper now requests
`Salesforce/wikitext`, preserving the `wikitext-2-raw-v1` train split and draw
algorithm. Green PB action
`8d091af6a523b2e514875411fa9ab88e8a675fcb555ee847983e51c2afa90168`
produced all 512×512 original B1 token IDs with seed 0. Its corpus SHA256 was
`aee724fa58bfbdeb3fc6803297fb6bab27b203d7c40b39ddef9b9770e5d52fe5`,
exactly the existing canonical corpus identity. Its verified CAS payload is
`98d8994c7e561e399ad1a565e0caca618a322d5462e2c9a190defb954b70c566`,
receipt `f7b35b3c7256c1f66997b7095d925da610f263624f6160fb2993d96a50cc9d67`.
This repairs repository resolution; the proposed production draw remains
unfrozen. The source/config byte binding and token IDs are immutable inputs
for the bounded workspace measurement only.

The bounded real-source prefix then ran canonical seed-0 B1 calibration
through layers 0–3 and attempted DSA layer 4, with layer 5 prefetch and the
same 104 GiB reservation, 102 GiB conservative guard, and 8 GiB minimum
host MemAvailable. Both attempts refused after the first target batch;
neither reached the full-H CPU transfer or established capture admission.
Action `8adc26ee625b34b2abe6331a19f1f0ed34792a06de116b9dbb5ca9d7e984ba3c`
used a harness missing the canonical per-layer `torch.cuda.empty_cache()`.
Action `30662666e4f95498b78d91c3f7d85a422789785c76fd0573aa5ce4637b1bba4a`
restored that call without changing the reader, draw, or memory guard. Both
exited 1 on Lina with completed PB cleanup and no OOMs; neither has a success
CAS receipt. Their shared input binding SHA256 is
`0d52fb4d17cd81e037eb66572a03616df3f53bd1848787fa1a9ff350a1045593`,
and source/config fingerprints remained unchanged at exit.

In the faithful-cleanup repeat, releasing the allocator at the layer-3
boundary reduced CUDA reserved bytes from 56,925,093,888 to 55,681,482,752;
13,749,179,904 inactive split bytes remained. At the guard, CUDA allocated
and reserved were 62,003,483,648 and 62,157,488,128 bytes; cgroup current
was 48,481,972,224 bytes. Of its file charge, 43,735,138,304 bytes were
ordinary file cache after excluding shmem. The conservative sum was
110,639,460,352 bytes, while host MemAvailable was 57,051,316,224 bytes.
These are separate accounting planes, not a deduplicated physical-memory
measurement. Canonical allocator cleanup alone did not satisfy the bound;
ordinary file-cache retention remains a material contributor. Global cache
state and cgroup page ownership can differ between runs, so this is not a
controlled speed or total-memory improvement claim.

SHA256 inventories, terminals, and cleanup are in
`review-02/source-prefix-first-refusal.json` and
`review-02/source-prefix-canonical-cleanup-refusal.json`. Each includes the
actual cold target-forward CPU/CUDA trace, main-thread Python stack samples,
continuous cgroup/CUDA/global memory observations and both-host Netdata.
The filenames say `forward-first-two-batches`, but only one target batch
completed before refusal; the requested later batches 32–33 were not reached.
The final focused integration with current main's shared-buffer precision fix
also passed 46 CPU checks without skips under PB action
`babb5f67986c755e3c58a0df6b918eeaa91de93786081da197e1806d95dd7490`;
its verified receipt inventory is `review-02/current-main-integration.json`.

The authorized opt-in source-page repair adds no cache or page ledger.
The shared reader retains CPU staging until per-chunk CUDA completion events
finish, drains all read futures on failure, and then advises only complete
pages wholly inside consumed tensor payloads. CPU-backed outputs, partial
edge/header/unread pages and non-regular files are excluded; source identity
must still match. Reader defaults and the workspace guard remain unchanged.
Its pre-fix PB regression `0d3175ddd7196e84008514ead709a635e498bfaff36d3c59fb744f3e664a8d7d`
failed seven checks and passed four. The final stream-scoped event and failure
lifetime implementation passed 66 CPU checks, including 20 source-page tests,
with no skips on dl380g10 (8 xdist workers, one native thread each).
PB action `f28bf31ca020e82a88112ef124ece5ed1796b9f048cb02acea2e6607e6e7f777`
exited 0 with completed cleanup; verified payload SHA256
`d2b79a9c6516ac04e10f7e9decef7389d8f88e7375b897e0ed27a449b73a1e71`,
receipt `0997c8fb3315ccc87181aed6a7aa15c26add40522ac963f8830f839a9f994fae`.
The immutable packet is `review-02/source-page-release-cpu.json`. Byte/range
checks use real CPU tensors and filesystem advice; CUDA copy/event ordering
and failure lifetimes are explicitly mocked in this CPU run. A real GPU
workspace repetition is still required; no throughput improvement is claimed.

The first real opt-in repeat, PB action
`2daabe0bf05a6fbc0bd810cc8979acaa1c329de1ac0780dd71d2b99de46fb71d`,
also refused after one completed target batch. Its original bound input stayed
unchanged. At the guard, cgroup current was 27,050,758,144 bytes and CUDA
reserved was 82,829,115,392 bytes; their conservative sum was 109,879,873,536
bytes, above the unchanged 102 GiB threshold. Ordinary file charge was
22,217,383,936 bytes after excluding shmem. The target roster is unknown:
the failed collector did not return its counters. This is not full-model
coverage or a frozen production calibration draw. The terminal exited 1,
cleanup completed, and no success CAS was published. The evidence inventory
is `review-02/source-page-release-workspace-refusal.json`.

The retained before/after traces show both target forwards started with the
prior layer's CUDA allocation already gone. Non-atomic phase sampling had
combined an earlier CUDA value with a later source-cache inventory. The newer
guard fired further into Hessian growth and next-layer prefetch, explaining
its higher CUDA reading. The old cgroup observer stopped at guard detection,
so its last sample cannot be added to a later CUDA peak to claim a physical
peak. The NFS module also changed between arms; no isolated causal speed claim
is made. Reproducible trace analysis is `review-02/memory-alignment.json` with
`analyze_memory.py`. Header-only page geometry is retained in
`review-02/range-coverage.json`; this screen does not establish OS residency.

A subsequent bounded repair moves staging ownership and one CUDA completion
event into each existing reader chunk. Successful chunks release their own
host views and advise the same payload-page ranges while sibling chunks can
still be reading. Failed chunks never advise, and the gather drains all
launched readers before refusing partial installation. The default and guard
are unchanged. PB regression
`b898bdc7dba2dd3f50a5d1a313835469f59b54ae24e3997cce6a3626b2591603`
failed six checks and passed 15 against the previous implementation, including
the finished-chunk-versus-blocked-sibling check. The repaired integration
passed 67 CPU checks without skips on Sparky (8 xdist workers, one native
thread each), under PB
`6972033ac33a686384dbc68f2eb244f2fd7355dfec91ee2dcc2aafbaf5c9fbfb`.
Exit was 0 and cleanup completed; payload SHA256
`6dd3b30d865d1f3be4f84091c188fb1424a737a5bcff4a4bea1ccb530e0096c3`,
receipt `06e86dd62c176fd8896003b8ae74509d6d5161f20c31df8352e689537f1feb08`.
The receipt, bundle and current source bytes were checked in
`review-02/chunk-source-page-release-cpu.json`. CUDA lifecycle remains mocked
in these CPU checks; the chunk repair has no successful GPU qualification yet.
