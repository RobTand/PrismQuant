# Issue 183 bounded LFM campaign: actual run evidence

This is the append-only observation record for the opt-in run described in
`pq183-lfm-bound-plan-2026-09-05.md`. Acceptance 4 requires a completed actual
campaign, export, attested census, paired serving smoke, and matched byte and
identity records. A dependency check or CPU regression does not satisfy it.

## Attempt 01: producer environment failure before calibration

PB action `bf8fe2dfbc3c836436077db638c22bf1435de314b303f77cd5bf4c2baccb839e`
ran on sparklina with four admitted CPUs (5–8), 64 GiB aggregate memory,
three exclusive GPU slots, and native numerical threads limited to one.
It exited 1 after 81.26 seconds. PB recorded 19,399,270,400 bytes peak
cgroup memory, no OOM, and completed resource cleanup.

The real LFM2.5-8B-A1B-BF16 model loaded and enumerated three dense Linear
targets plus 96 projected expert units from layer 12. The next operation,
the existing `_calibration_tokens` loader, raised
`ModuleNotFoundError: No module named 'datasets'`. PrismaQuant declares
`datasets>=3.0`; the producer image did not contain it. No calibration,
priced cost rows, exported artifact, census, or serving comparison resulted.
This attempt is inconclusive and does not establish acceptance 4.

The host harness recorded both Netdata endpoints (`http://sparky:19999` and
`http://sparklina:19999`) successfully, no telemetry-monitor errors, and safe
cleanup for every exact owned container. `campaign.pstats`, campaign and host
logs, image inspections, and `telemetry.jsonl` are retained. These are failure
diagnostics, not performance measurements.

Source provenance:

- Actual run parent: `1eb268f1825b85a0f0dff61423343ee13d59eb4e`.
- PB materialized snapshot: `f631c464cfa9f9fd740bf025f8178c581abb1af1`;
  bundle SHA-256 `545e5412da5222c5e46dd3072dc3f61d3e6a590de6fae35ac89f62676df2fc24`.
- Coverage correction subsequently merged in PR 243:
  `195f3ba754f4f4efdcbcfcc7a5aecd76d7b9acda`; the actual run already contained
  the reviewed production changes through their original commits.
- Tessera: `ba582d476a3b6db9057ebd1385dc52926f171451`, sealed full 1,017-file
  source manifest SHA-256
  `00710fbf9f15269ae0a579c02d6ad8fae22a476ece20293d73feea56c37f211a`.
- Producer local image:
  `sha256:79cb5c9a8cd696f30cb0d8b5803d67d65906de4df91741c9811f3de088a13846`;
  canonical Config SHA-256
  `83d0dcabcd3b6d259e9dea48bb67b5bf36108e22d03a7abb2209d73a2adc9e53`;
  RootFS SHA-256
  `d97ec6de925255c82642f99bc250a3e5a554002583b276aa8eacfd15166c7592`.
- Declared serving image, not reached in this attempt:
  `eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.

All run paths below are relative to
`/home/rob/tessera-runs/measurement-208-183-2026-09-05`:

- Original sparklina outputs: `run183-01/`; submission: `pq183-submit-01.json`.
- Preserved local diagnostic copy: `pq183-evidence/run183-01/`.
- PB terminal:
  `/mnt/shared/prismabuild-fleet/pb-queue/failed/bf8fe2dfbc3c836436077db638c22bf1435de314b303f77cd5bf4c2baccb839e.json`.

## Environment qualification, separate from the GPU measurement

CPU-only PB action
`4f58ad3a72d5fe8223e705ed3a1b697e3b3dab9a299c329abdfb9f749d93d2d3`
attempted `datasets==5.0.1` while constraining every existing image
distribution to its original version. Resolution refused because the base
contained `fsspec==2026.7.0`, while datasets requires `fsspec<=2026.6.0`.
No image was published; its exact owned container was removed. The negative
receipt and resolution log remain under `environment-repair-01/` on sparklina.

The scoped follow-up pins `fsspec==2026.6.0` explicitly and retains all other
existing distribution constraints. It must check the actual before/after
distribution inventories and the unchanged WikiText train calibration contract
(eight sequences, length 512, seed zero) before publishing a derivative image.

CPU action `3740e7795eefb460dafdb71c8232fdfb3d1a2f1109d40fc8f8eb0d3f21cc9845`
installed and imported datasets successfully, but the unchanged calibration
loader failed: base `huggingface-hub==1.28.0` rejected the `wikitext` alias
with `HfUriError` requiring a namespace. No image was published. Its evidence
and exact-container cleanup are retained in `environment-repair-02/`.
The next qualification also pins `huggingface-hub==1.10.2`, matching the
existing host environment's dataset client. This changes the transport client,
not the calibration dataset or token draw.

The base's package metadata already reports a Torch/NCCL requirement mismatch
(Torch 2.13.0+cu130 declares NCCL 2.29.7; the image contains NCCL 2.30.7).
Dependency repair retains that existing native stack. Base duplicate `six`
distribution metadata also exposed an inventory ambiguity; later qualifications
record the complete distribution roster and use effective `metadata.version`
lookups for constraints, rather than collapsing duplicate entries arbitrarily.

CPU action `acabfd790fe0bb8c00248272a963478c2b654640b119838b5b6314eaaebca884`
then completed installation, imports, unchanged numerical-package checks, and
the actual eight-by-512, seed-zero calibration draw in 24.8 seconds. Its package
delta was exactly fsspec 2026.7.0 → 2026.6.0 and huggingface-hub 1.28.0 → 1.10.2;
new packages were datasets 5.0.1, multiprocess 0.70.19, pandas 3.0.5,
pyarrow 25.0.1, and xxhash 4.0.1. Torch 2.13.0+cu130, Transformers 5.15.1,
NumPy 2.2.6, safetensors 0.8.0, accelerate 1.14.0, and native distributions
retained their existing effective versions. The CAS receipt is
`f6ae50e84d10cc88a36846fc88e8fd2886d191b4f66fe8c760a5ca194ea09560`;
its 35,280-byte payload SHA-256
`c40955c511e6cdf59bba80fa48f87e29c7d1fe105b8ae4ee44fac3102325f16e`
was independently verified. Original evidence is `environment-repair-03/`.

CPU action `432058644465fd5a9806ab46a71c603ae1e07a7360b8e68390ed89250ddf9875`
baked that verified Hugging Face cache into `/opt/pq183-hf-cache` and reloaded
the exact calibration with Docker networking disabled and both Hub/datasets
offline flags set. The corpus and all eight int64 token-buffer hashes matched
the online draw exactly. It exited zero in 11.3 seconds, used two admitted CPUs,
eight GiB memory, no GPU, and peaked at 2,340,651,008 bytes. Exact owned-container
cleanup and PB resource cleanup completed. The CAS receipt is
`f80c543ac3f891ecb5bf4f9ecb47f698a5dfb4b68ab1f5388178af5b87a79a5b`;
the 27,202-byte payload SHA-256
`8b2eb95fae60061875b7ba9de2ddb0099b80a6f23b77294cd23416b8c1d2e0de`
was independently verified.

Final qualified producer image:

- ID: `sha256:47dd0e9aaa4e7a6575d21cfc661d96a47c0e35e87c64e850631e210bdf04ebc0`.
- Canonical Config SHA-256:
  `fda47b55fb7105c93e8a0bf99cd633191c198e4033719957734d065a635de31e`.
- RootFS SHA-256:
  `df0f8207331bd466df86322a178e14501f707f7b765e820a60e7ce9f28d51d71`.
- Corpus SHA-256:
  `aee724fa58bfbdeb3fc6803297fb6bab27b203d7c40b39ddef9b9770e5d52fe5`.
- Cache manifest SHA-256:
  `130331b93cc509674cd06fed6e35fd3ecb1163257ad69a08be5efaf52176610d`.

`environment-repair-04/` contains the final image inspection, cache manifest,
offline calibration receipt with individual token hashes, exact command logs,
and cleanup evidence. The image fixes `HF_HOME=/opt/pq183-hf-cache`,
`HF_HUB_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`; the host producer command must
preserve those settings. The source model and Tessera source were read-only
mounts during qualification. Container creation/installation/qualification/
commit all ran through PB; no daemon build or host environment provisioning
was used. The bounded derivation scripts and equivalent Dockerfile are retained
in `environment-repair-source/`, with immutable PB snapshots in the receipts.
The intermediate image `sha256:500117f774d8d89f187107cfc7363481e361558febd0de0d082e69c15cbdacdd`
is retained as provenance; the final image above includes the offline cache.

These CPU qualifications repair and characterize the environment. Actual GPU
campaign/export/serving acceptance remains pending a fresh reviewed run.

## Attempt 02: real capture exposed a parent-owned router bias

PB action `b47707a2f754eb71a78eb34ecd7a7aaea81b890d4d03d4660e57479898b6018f`
did execute despite an initially stale placement indication. A later withdrawal
request found it already failed; it was not a cancelled, unexecuted attempt.
Source parent was `3ac5f3c69782b5430b4a5804cc20451a67038e37`, materialized
snapshot `936259b853512f5e81fffc53191d59c35109dc93`, bundle SHA-256
`6e99d1de8fb4dacee586346c51b0debcc2052100aa9cd658d54d95e865a08306`.
It used the final offline producer image above, four CPUs, 64 GiB, and the
then-submitted exclusive three-slot GPU request on sparklina. It exited 1
after 30.31 seconds with completed PB resource cleanup and safe exact-container
cleanup. Both-host telemetry, dependency preflight receipt, actual campaign
log, and cProfile output remain in `run183-02/` and the local mirror
`pq183-evidence/run183-02/`.

The producer dependency preflight and actual offline calibration load passed.
During real model forward capture, `derive_per_expert_activations` replayed
the packed router without its parent-owned `expert_bias`. Transformers 5.15.1
stores that buffer on `Lfm2MoeSparseMoeBlock` and passes it to
`Lfm2MoeTopKRouter.forward(hidden_states, expert_bias)`. PrismaQuant's existing
adapter recognized only `e_score_correction_bias`; the missing LFM argument
caused `Tensor + None` to fail. No priced costs, exported artifact, census, or
serving results were produced. Acceptance 4 remains unestablished.

The bounded repair extends the shared router adapter and its three callers:
activation capture, packed-cost replay, and empirical down-input statistics.
It passes the original parent buffer to the model router, refuses a missing
buffer when bias is enabled, and retains explicitly disabled-bias behavior.

CPU regression used the actual qualified image and Transformers 5.15.1 router,
with a nonzero parent bias that flips selected experts relative to uncorrected
routing. It checks actual model-forward indices/weights, derived gate/up and
down-input rows, and empirical column statistics. It also checks disabled bias,
clear missing-bias rejection, and existing correction-bias handling.

- Red PB `6e958854678fc4d3302a80846038bf47ff70814ca3d703f47da5fa1ca6b12c12`:
  two intended failures and one pass, no skips. The real-router capture
  reproduced the exact `Tensor + None` exception before the fix.
- Green PB `c6f5d1ff7694a576078497966c1538b14c579f16153adc5ef71fa2acaf089c00`:
  32 passed, no skips, including existing packed-cost and empirical suites.
  CPU-only, two admitted CPUs, eight GiB aggregate memory, native threads one,
  and networking disabled. CAS receipt
  `b0eec7fe8591638083b14d6b934977aa7cb5a423c6d41e6caafa07ed4bac44b8`;
  independently verified 546-byte payload SHA-256
  `c04707642fbca02bb534ad6879b09603eeb325ef56920d1a2135ae220a0a64f8`.

Both CPU actions' terminal records and the successful CAS receipt are retained
in `pq183-evidence/`; the sealed test-only source and driver are
`pq183-router-validation/`. These regressions establish the routing repair,
not completion of the actual GPU pipeline.

Required architecture/staleness checks and source compilation passed separately
through CPU PB
`d3e11c503da0f8fe1ba2e85acf61eb95b5d3ce692bdf89fe526b0c60fa8317b2`;
the actual log and verified CAS receipt are retained beside the routing tests.

## Attempt 03: insufficient routed calibration coverage

PB action `874fe72da13a00e3c6bd18f67e339235e655a5feb64c55cf5962c9b347c76608`
ran source parent `4cb2084134133b3d487a1d01112e0055b541b186`, including the
routing fix merged as PR 247 (`8b16aef3`). Its materialized snapshot was
`53b210b0fb7d6f336f2bbcc51629af4fda8302f2`, bundle SHA-256
`1ee0e9557776242c0ede3b00bd873160dd530f1b039ce659811065e916b5b309`.
The qualified offline image and eight-by-512, seed-zero calibration remained
unchanged. The resource request used the fleet's current one-slot representation
of one exclusive physical GPU, four CPUs and 64 GiB on sparklina.

The router fix worked. Real calibration completed its forwards, then the
existing no-routed-rows gate refused exactly three units:
`model.layers.12.feed_forward.experts.2.{w1,w2,w3}`. Expert 2 received no
calibration tokens in this draw, so those three units had no empirical Hessian.
The campaign correctly refused a shared-Hessian or weight-only fallback; it
did not emit `cost.pkl` or start export/serving. This negative result establishes
insufficient coverage for the declared eight-sample campaign, not a completed
priced observation or a defect in the no-row gate.

The action exited 1 after 30.64 seconds. Both Netdata endpoints were recorded,
monitor errors were empty, exact-container cleanup was safe, and PB resource
cleanup completed. Logs, cProfile, dependency receipt, host state and telemetry
remain under `run183-03/`, mirrored in `pq183-evidence/run183-03/`; the failed
terminal record is retained beside them. Acceptance 4 remains open.

The existing CLI supports changing the calibration sample count but has no
per-projected-unit exclusion flag. A later larger draw, if run, must be recorded
as an explicit revised calibration contract, preserving the corpus, sequence
length and seed and retaining this eight-sample negative observation.

## Attempt 04: the explicit 32-sample draw still misses layer-12 expert 2

PB action `e6e110874a033c26ed2ce1c3cf715f418758839a5cdefbad710ab358ee63cecc`
used source `47a31e87b71c6d39109c6c730a3ca0a0649f717d`, materialized snapshot
`4ed7bf06e93f9eebde705911b88ad1cee3708860`, bundle SHA-256
`652ea18f28c005034f36539bbacae93d15c8f8705ee7c81d229a1135281cabc0`.
This attempt explicitly increased the calibration draw to 32 sequences while
retaining WikiText train, length 512, seed zero, maximum stored activation rows
512, the qualified offline image, fixed candidate, and all 96 layer-12 units.

It exited 1 after 33.95 seconds: expert 2 still received zero routed calibration
rows, so its `w1`, `w2`, and `w3` projections had no Hessian. The unchanged gate
refused before pricing, export, census, or serving. This result is insufficient
layer-12 coverage under the 32-sample contract; it does not prove the expert is
permanently inactive. No further sample escalation or fabricated fallback was
used. A subsequent actual-forward histogram is a coverage diagnostic, not
quantization-quality-based layer selection.

Original artifacts are in `run183-04/`; the submission is `pq183-submit-04.json`.
Both are mirrored in `pq183-evidence/`, along with the failed terminal record.
Both-host telemetry succeeded with no monitor errors, all exact-container
cleanup records were safe, and PB resource cleanup completed. Acceptance 4
remains open.
