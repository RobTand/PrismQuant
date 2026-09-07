# Served NVFP4 codepoint residency — 2026-09-07

Reusing the format registry's existing device codebook removes a host-to-device
copy and synchronization from each served NVFP4 QDQ call. It preserves output
bytes, midpoint rounding, signed zero, underflow, and the joint projection's
signed components. On the actual routed calls below, the measured wall-time
reduction is **2.542%**. This does **not** resolve the stopped full joint run's
performance problem: elementwise multiplication still occupies about 70% of
the measured CUDA time. The sampled energy comparison establishes no gain.

## Change and numerical qualification

`nvfp4_activation_qdq_served` formerly constructed an eight-element FP32 tensor
from `_E2M1_POSITIVE` on every invocation. The registry's sorted E2M1 table
already contains precisely those eight positive encodings, including positive
zero, at its end. The owner now takes that view through the existing
`_codebook_on_device` cache. It adds no cache, quantization policy, arithmetic
reassociation, format, or pipeline default.

The CPU regression failed before the change because three warm calls rebuilt
the codepoints three times. The same test and related contract/projection
suites passed after the change: **81 passed, 6 CUDA-required skips**. In the
pinned GPU container, all **12 CPU/CUDA residency and numerical golden tests
passed**, including FP32/BF16/FP16 midpoint ties and signed zero, scale
underflow, and exact device-grid bytes.

Both GPU A/B comparisons also required exact forward, input-gradient and
weight-gradient hashes, plus all weight/activation/mixed/total signed
components for four probes and seven original measured formats. All passed.
No candidate bytes, calibration tokens, static scales, source weights or
source model batching changed.

## Matched measurements

Environment: Sparky, NVIDIA GB10, PyTorch `2.13.0+cu130`, CUDA 13.0, pinned
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
PB measurement admission reserved four CPU cores, 12 GiB total memory and an
8 GiB GPU subset. Native threads were bounded to one. CPU/CUDA
`torch.profiler` traces were collected separately from steady ABBA timings;
raw Netdata CPU, RAM, I/O and GPU-power series cover both hosts.

The unit is `model.layers.23.feed_forward.experts.0.w1`, BF16 weight shape
`[1792, 2048]`, with its seven original measured PWC renders. The harness
verified the prepared artifact, source tensor identity, original render-file
and tensor hashes, activation identities, canonical manifest/census, and
capture payload metadata. Existing PWC/capture prefetch reported zero misses;
the measured arms recorded zero process `read_bytes` and `write_bytes`.

| Workload | Old A1 | Reused B1 | Reused B2 | Old A2 | Median old → reused |
| --- | ---: | ---: | ---: | ---: | ---: |
| Canonical 512-row prefix, four fixed random BF16 cotangents; 1,696 cycles | 34.7277 s | 34.6955 s | 34.6856 s | 34.8122 s | 34.7700 → 34.6905 s; 0.228% less |
| Actual B1 routed calls, four actual Fisher probes; 632 cycles | 36.1002 s | 35.1915 s | 35.2405 s | 36.1689 s | 36.1345 → 35.2160 s; 2.542% less |

The first comparison is a useful negative measurement: the canonical scoring
prefix combines several invocations and is not the triggering per-call shape.
The second replay retains four actual invocation shapes, **137, 102, 74 and
80 rows**, from original complete calibration sequences 0–3. It accumulates
their signed terms in the actual invocation order before finishing each probe.
The replay exercises the real projection hook with retained X/g and an
isolated Linear; it is not a timing of the entire grouped model forward or
the complete 512-sequence joint cost.

The source-control capture used the existing streaming source path and
original prefetch contract, B1 full 512-token sequences, eager attention,
original source/dispatch identity, seeds 7000–7003 and global row offsets
0–3. Fisher normalization remained **262,144 global tokens**, as in the full
512-by-512 draw. It retained actual selected output cotangents, not generated
replacement gradients. Its peak CUDA allocation was 19.262 GB; the separate
capture was admitted for 36 GiB total memory and a 24 GiB GPU subset.

The actual-call profiles cover 48 invocations per arm. Reuse removes **48
pageable H→D copies**, reducing `cudaMemcpyAsync` and `cudaStreamSynchronize`
counts from 132 to 84 each; the remaining 84 copies are D→H. Device-kernel
count stays 5,280. `aten::mul` CUDA time is **111.211 → 111.142 ms**, about
70.1% of the CUDA total in each arm; GEMM is about 9.9%, and sum reduction
about 8.3%. A stack waiting at the old constructor therefore does not identify
the constructor as the owner of all preceding queued GPU work.

Actual-call mean GPU power by arm was 39.43, 41.26, 42.00 and 41.39 W, about
28–30% of the approximately 140 W envelope. Interpolated median GPU energy
was **1,460.24 → 1,466.05 J**; useful cycles/J were **0.43281 → 0.43109**
(ratio 0.9960). These are GPU-only estimates from ten-second samples, not
whole-host energy or evidence of an efficiency gain. The 512-row comparison
also showed no energy improvement (work/J ratio 0.9679). Neither comparison
establishes GPU saturation or full-model throughput. Actual replay peak CUDA
allocation was 345.6 MB; PB scope memory peak was 2.211 GB.

## Capture-order diagnosis and retained unsuccessful attempts

An initial check incorrectly assumed that actual grouped rows have the same
order as the canonical scoring prefix. The canonical collector re-derives
packed rows via `derive_per_expert_activations`, ordered by top-k position
then token; the grouped observer records the runtime's contiguous expert
slice. A PB CPU comparison proved that all **393 rows match byte-for-byte as
multisets**, including separately within each 137/102/74/80-row invocation.
It retained the complete permutation. The final harness verifies that
per-invocation bijection and exact mapped bytes while preserving actual X/g
order for projection. The apparent positional difference was not source
arithmetic drift.

Retained failures contain no timing claim: the first container lacked pytest
(fixed with a scoped pinned install); the next harness incorrectly treated a
canonical capture dictionary as a Tensor (fixed by using the existing
validated capture reader); and the first actual replay stopped on the invalid
row-order assumption above. The original CPU red is the intentional
three-allocation regression, not an environment failure.

## Attributable artifacts

All evidence is retained under
`/mnt/shared/tessera-measurements/first-model-20260907/qdq-residency/`.
`frozen-source/source-manifest.json` binds exact per-action source snapshots,
including unsuccessful harnesses; `run-02/` and `actual-replay-02/` contain
receipts, before/after source functions, CPU/CUDA tables, Chrome traces,
both-host raw Netdata and independently checked PB/CAS records.
`source-capture-01/` binds the actual X/g payload; its receipt SHA is
`bf5337381562e2b4ae1fbdb672e85f198679e2848487b291590d7810990872d3`.
`capture-order-diagnosis.json` retains the exact row permutation.

The checked-in experiment entry points are
`experiments/qdq_projection_source_capture.py`,
`experiments/qdq_capture_order_diagnose.py` and
`experiments/qdq_constant_residency.py`. Historical executed versions are
retained per action above. The final replay command, with a fresh output
directory substituted, is:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd CHECKOUT --gpu --measurement --cpus 4 \
  --demand mem_gb=12 --gpu-memory-gb 8 --timeout-s 1500 -- \
  bash experiments/qdq_constant_residency_run.sh --output NEW_OUTPUT \
  --source-receipt /mnt/shared/tessera-measurements/first-model-20260907/qdq-residency/source-capture-01/receipt.json \
  --source-sha256 bf5337381562e2b4ae1fbdb672e85f198679e2848487b291590d7810990872d3
```

| Qualification | PB action | Verified CAS stdout SHA-256 |
| --- | --- | --- |
| CPU regression red | `de66fa46439d203719c8a61d0b548d1c5aa87c261ce932e30bfb16b0bc7c3e3f` | Intentional failure; terminal retains assertion |
| CPU green | `d023dda6d25f6302600f722655ab2184f2966ebedbf1527d13b447cc6341954a` | `fe76df4904e347318babde2b75e7fb7be1d16926e424d889e77264d685e122ec` |
| CUDA goldens and M512 ABBA | `f2dddac6864b0ec49e18f6f722a27ee3dce1ce43957a471e719bbe768e42369a` | `74e2d9c575de865b6e1524947a822e2b43b5e0bcc1418b488f8cc4886ad08cca` |
| Actual source capture | `ac687314a937173c33a437242dd1a69631df30c89204d9b113bed537762f2191` | `85eabb6dd01613fc12790335048b3340209dcb9239167adaeb2526fdcaf8e1f5` |
| CPU row-order proof | `759ea5d8e839df82923b6dffc3a02e40043c360b8cb18e72b050ecdfe5a6f3a5` | `3da0caa74eae7964cea317fc1fc1bac877a1c7e1c3363565136822e6a87371d7` |
| Actual-call ABBA | `78cbb9c1fc0903c8d676e30bfded803d39fdf4aaef5bb958579800f3a5b3b460` | `cd51dc438bb8524b79cf1b8ea80fdb50cd22802a4819bda1cc22b811ddfeecd6` |

All successful actions above have checked exit status 0 and actual payloads;
the GPU scopes completed cleanup. The benchmark source base is
`d10b5bfcf38e52e03934764ea99e4395abff7518`; the final measured snapshot is
`acc3f0952e507a10937462db1797ad90759828e3`. This result qualifies the small
QDQ constant reuse only. Repeated product temporaries remain a separate
profiled optimization target requiring its own numerical and performance
qualification.
