# LFM mixed-family sanity preparation (#253)

This opt-in preparation creates inputs for a PrismaBuild-owned export campaign.
It does not launch the export or select partitions. Its explicit fixed layout
quantizes all 2,178 eligible body matrices: 22 routed stacks / 2,112 E4M3 q1024
projections, first two dense FF blocks / six E2M1x2 q896 matrices, and remaining
60 dense BF16-grid q1792 matrices. Real BF16-grid trellis differs from plain
BF16 passthrough. The producer owns classification, source layout, expert
projection and 52 dense fusion owners; all retained source tensors are named.

The producer is sealed Tessera 6faa5ce314cadeee8a190cbeadcf6cde3a333efb.
Preparation uses the previously qualified immutable 47dd producer image and its
offline WikiText cache. Existing PQ `_calibration_tokens(32, 512, 0)` supplies the
same text and draws as issue 183. `_collect_activations` retains zero scoring
rows, computes no H, and observes every dense input row and its maximum.
`_static_input_scales` receives the producer fusion roster through its profile
interface, preserving one conservative scale per fused owner. Six F32 scales
travel with their policy, exact corpus/token/tokenizer identities and 16,384
observed rows per unit. No expert row coverage is claimed. The full model must
be GPU resident throughout this one coherent capture.

The preparation seals the plan, producer projection, source identity, retained
population, scale tensor and calibration receipt, host PB snapshot/image/source
identity and cProfile. Outputs must be fresh. The source manifest covers every
external Tessera file, including producer scripts; neither container Git nor an
unverified worker path supplies source authority. Host observation preserves
PB CPU affinity, image Config/RootFS identity, both-host Netdata, power, process
scope and exact owned-container cleanup. Full terminals remain private; any
publication copy must redact scope tokens before Git staging.

`python experiments/lfm_mixed_preparation.py host --mode preflight ...` runs
producer classification and source checks with GPU visibility disabled (4 GiB).
The same explicit arguments with `--mode calibrate` use a four-CPU, 64-GiB GPU
action with bounded native threads. Required arguments are `--model`, `--out`,
`--tessera-repo`, `--tessera-source-manifest`, its SHA via
`--tessera-source-manifest-sha256`, and two `--netdata-url` values. The CLI is
executed only inside a PB action. Final `preparation-seal.json` is the handoff;
a preflight seal cannot substitute for a `mode=calibrate` seal with scales.

The fixed weights-only plan is a sanity experiment, not per-Linear empirical
selection or an optimized production recipe. EUGR dense4/16 qualification and a
mixed dense/routed census gate are prerequisites owned separately by the
integration task. Existing official-image dense cells do not qualify EUGR.
No runtime pin or production menu changes are included here.

## Actual preparation result — 2026-09-06

The GPU calibration completed with exit zero and produced a sealed, readable
shared handoff at
`/mnt/shared/tessera-measurements/mixed-lfm-237-2026-09-06/calibrate-01`.
Its `preparation-seal.json` SHA256 is
`d302740baa39a484135a5de73abfcf9d5c3ec91eb419cb59a3d4af94c2704952`.
All eight sealed files were rehashed after archival; the six F32 values in
`input-scales.safetensors` were read back and exactly matched the JSON receipt.
The full source identity, native producer projection, binary scales, cProfile
and both-host telemetry remain in that archive. Compiled Triton cache files
are excluded. The original task-local output remains intact on Sparklina.
The [publication roster](artifacts/lfm-mixed-preparation-2026-09-06/publication-roster.json)
binds 19 bounded JSON records to their original archive hashes; scope tokens
were redacted before first Git staging. Full terminals remain in the mode-0700
shared `private-preparation-evidence` directory, with mode-0600 files.

This is successful input preparation, not a full-model encoding or serving
result. The fixed layout has 2,178 matrices and 74 serving owners: 22 expert
stacks and 52 dense fusion owners. The six E2M1x2 matrices occupy four dense
owners. All other eligible dense matrices use real BF16-grid trellis;
immutable source tensors and routers are explicitly retained/classified in
[population.json](artifacts/lfm-mixed-preparation-2026-09-06/population.json).
The [actual plan](artifacts/lfm-mixed-preparation-2026-09-06/plan.json) and full
native projection were derived from the pinned producer APIs. No manual
partitioning or full-model encode was submitted. Automatic distribution is
tracked by [PrismaBuild #201](https://github.com/RobTand/prismabuild/issues/201);
mixed artifact/serving validation is tracked by
[PrismaQuant #253](https://github.com/RobTand/prismaquant/issues/253).

The resolved existing static-scale policy was
`legacy_6_over_calibration_amax.v1`. Each of the following inputs actually
observed 16,384 rows during the same 32 sequences; these counts were not inferred
from the requested draw. The calibration kept zero scoring rows and no Hessian.

| Dense input | Observed maximum | F32 input scale |
| --- | ---: | ---: |
| layer 0, w1 | 2.515625 | 2.3850932121276855 |
| layer 0, w2 | 1.0234375 | 5.862595558166504 |
| layer 0, w3 | 2.515625 | 2.3850932121276855 |
| layer 1, w1 | 1.421875 | 4.219780445098877 |
| layer 1, w2 | 0.22265625 | 26.947368621826172 |
| layer 1, w3 | 1.421875 | 4.219780445098877 |

The complete names, counts, maxima, versions and draw identity are in
[calibration.json](artifacts/lfm-mixed-preparation-2026-09-06/calibration.json).
The existing offline WikiText `wikitext-2-raw-v1/train` draw used 32 samples,
512 tokens and seed 0. Corpus SHA256 was
`aee724fa58bfbdeb3fc6803297fb6bab27b203d7c40b39ddef9b9770e5d52fe5`;
token SHA256 was
`f79fd1b823c54b9f22d0cb814b9e496159086bec63cf5671bec13f1663375238`.
There is no claim of full expert calibration coverage, Hessian rank, quality,
optimality or serving performance.

### Source, execution and evidence

The actual preparation parent was
`a6aebbab93d1826b8b775f68db1c9c17956907fe`, with PB snapshot
`58c039dc35ff6b12cf2c29f5a6e513968e0e9e69` and 24,766,574-byte bundle SHA256
`77de08fd744170370ffd816e23cc89a854a6cf570a2df7af2b5707241dc7a662`.
The shared Tessera tree is `inputs/tessera-6faa5ce`, commit
`6faa5ce314cadeee8a190cbeadcf6cde3a333efb`. Its full 1,020-file source manifest
`inputs/tessera-source-manifest.json` has SHA256
`04965bbc921b8e7086b406d16e1d530bf6ed13b9e65531a164c28a1c01470c6f`.
Source model `model.safetensors` SHA256 is
`c9b9e3c4b3be50b576e6da8c02de1b4223614ffe131d812abf92bb84421f6217`.

The producer remained
`sha256:47dd0e9aaa4e7a6575d21cfc661d96a47c0e35e87c64e850631e210bdf04ebc0`,
with Config SHA256
`fda47b55fb7105c93e8a0bf99cd633191c198e4033719957734d065a635de31e`
and RootFS SHA256
`df0f8207331bd466df86322a178e14501f707f7b765e820a60e7ce9f28d51d71`.
The producer uses its pinned offline corpus cache, a read-only shared source
mount and a fresh task-local output. The separate EUGR serving target remains
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`;
this receipt does not qualify its dense runtime cells. Exact producer argv and
source/image bindings are in
[host-input.json](artifacts/lfm-mixed-preparation-2026-09-06/host-input.json).

The published PB client admitted calibration action
`d014fd59a345833af5492e4aa4eafe4a816650493307e5608dbc61174ae04ba3`
with four CPUs, 64 GiB, one exclusive physical GPU, 600 seconds plus 120 seconds
cleanup, and native threads bounded to one. The established qualified image
was the worker dependency; no model work was manually distributed. The actual
container retained the PB CPU set. Scope wall time was 36.355 seconds and CPU
time 35.272 seconds, with scope peak memory 17,528,045,568 bytes and CUDA peak
allocated memory 17,110,018,048 bytes. These describe this capture, not a speed
comparison or saturation finding. Both Netdata endpoints succeeded; power,
CPU, residency and cProfile observations are archived. Host and PB cleanup
completed, no OOM occurred, and the exact owned container was already absent.

The successful CAS receipt is
`5bd0543c3eede22e00b26c2945f06d2326190c725ec3219176888653aa641bb7`.
Its actual 177-byte payload was read and independently rehashed to
`9b706ffa017f740b81622076f877fe8ec741727d44c7e27ae482b6f3eab7c4dd`;
it names the prepared output and final seal. Submission alone was not treated
as success.

### Prerequisite checks and preserved negative attempts

Portable preparation guard tests passed 14 tests with zero skips through PB
`c63e3eb49178d455952eb446125c1ca8a22e509a7c533c690c29b5776fc4342f`.
They cover producer classification, full expert projection, fusion consistency,
retained population, scale policy/provenance and incomplete inputs. The actual
1,932-byte CAS payload SHA256 is
`94d9da84ebd4c1ae6e671b5ee5bcf45dc8ec8ce7aaa979e276abb24648089f8a`.
A separate real-torch regression exposed zero reported counts in no-H mode;
its red/green proof and shared production repair are recorded in
[the row-count results](tessera-capture-row-counts-2026-09-06.md).

CPU preflight 01, PB
`5bdd397e1f7c35a28f5d2fdee47c1aeccf55a21b31eebd16e8ee4e2399c5f3a9`,
failed before Python with Docker exit 126 while binding a newly staged shared
subdirectory. Preparation now binds the existing shared root read-only and
writes task-local output. Preflight 02, PB
`b403b43b35d955aa017b07ec335fbfb50d4fadf6040c05539f58c43b9b36e1b4`,
then failed before producer import with permission denial on the source
manifest: root-squashed container access could not traverse the mode-0700
staging ancestors. Only task/source directories were corrected to 0755 and
the manifest to 0644, preserving all file bytes. Both failed attempts retain
terminal/log/cleanup evidence; neither has a successful CAS or numerical result.

Fresh CPU preflight 03, PB
`893bada57f1d8504f3e866952ef7d7b47131115b9dd6b770769f0f450cba15c0`,
then passed actual producer source classification/projection with GPU hidden,
two CPUs and 4 GiB. Its shared archive is adjacent `preflight-03`; its seal is
`7d50c70421284d9cbf7a2b98f430c98a8d29e6e975a4550f696e1a0c28e5891e`.
The actual 177-byte CAS payload SHA256 is
`b07b8e4581031c0df4dd16b39d671f548110e8b098f3b9d5fb18b31b6f09ea7c`.
Only after this success did the single coherent GPU calibration execute.
