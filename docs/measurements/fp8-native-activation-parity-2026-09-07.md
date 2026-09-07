# Native FP8 activation parity — 2026-09-07

The original complete routed-MoE measurement failed its input-QDQ gate even
though the whole-output tolerance passed. The shared PrismaQuant activation
adapter had two arithmetic differences from the target native kernel: CUDA
scalar division by 448 lowered to multiplication by a rounded FP32 reciprocal,
and the compressed-tensors quantizer added a zero point, erasing negative zero.
The fix divides by an on-device FP32 tensor and uses the existing direct
FP32 divide/clamp/cast helper for activations. Weight quantization is unchanged.
This repairs the existing served arithmetic contract; it does not promote a
format, change a tolerance, or establish full-model quality or speed.

## Actual failure and arithmetic diagnosis

The frozen native dataset is
`/mnt/shared/tessera-native376-resource/native-moe-original-r1024/activation-diagnostic-01/diagnostic.safetensors`,
SHA256 `8452ec28f9b325e52fcc6745ee4aa16d6828374e45650380b3e56ade770fffb2`.
It contains actual BF16 prefill `[512,2048]` and decode `[1,2048]` inputs,
native scales, native FP8 bytes, native QDQ and the old PQ references. Native
capture ran in official vLLM image
`vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33`,
upstream commit `1970f3ed4be7fa8620e4ddc4a12c36a8384cfc27`.
The exact native source/provenance is retained under that native evidence root
at `upstream1970/index.json`: scale division is in
`csrc/libtorch_stable/quantization/w8a8/fp8/common.cu:163`, and the non-inverted
conversion divides by scale in `csrc/quantization/w8a8/fp8/common.cuh:57`.

The old shared adapter exactly reproduced the old frozen reference. Against
native prefill, 256 of 512 scales differed by one FP32 ULP, 470 FP8 bytes differed
(25 were signed zeros), and 4,050 BF16 QDQ elements differed, with maximum
absolute difference 0.0703125. Decode was already exact. The original native
measurement `measure-02/receipt.json` (SHA256
`dffa1514d65612a936876da8398f1553ddaa2e7c7dc36d30dae4877582fc1215`)
therefore remains negative evidence; its normalized prefill input-QDQ error was
2.660508 against the unchanged `atol=rtol=0.015625` gate. No timing result from
that rejected panel is accepted.

One reproducible row has amax 0.96875 and native scale
0.0021623882930725813; the old adapter produced 0.002162388525903225.
Inputs `[-0.0908203125, 0.022705078125, 0.36328125]` quantize natively to
`[-44, 11, 176]`, versus old `[-40, 10, 160]`. These independently recorded
constants and negative-zero/floor behavior are CPU and CUDA regressions.
Tensor division and FP64 scale division cast back to FP32 both matched native.
Multiplying by the reciprocal of the native scale still produced 407 wrong
code bytes; idealized FP64 division through the entire quantization expression
also failed (135 wrong code bytes). The two FP32 divisions and their order matter.

After correction, all 512 prefill scales, all 1,048,576 prefill FP8 bytes and
BF16 QDQ values, and all decode results match native bit-for-bit. The diagnostic
JSON retains its historical `shared_compressed_tensors` variant label; that
label denotes the shared adapter loaded from the recorded source hash, which
now performs direct activation QDQ. It is not a claim that the corrected
activation route invokes compressed-tensors.

## Validation and retained receipts

Evidence root: `/mnt/shared/tessera-measurements/pq-fp8-native-parity-20260907`.
All execution below ran through PrismaBuild. Terminal JSONs live under
`/mnt/shared/prismabuild-fleet/pb-queue/{done,failed}/<action-key>.json` and carry
exact commands, checkout snapshots, worker identity, stdout, CAS receipt and
resource cleanup. The source-reference GPU environment is immutable image
`sha256:9f9b9f05b17531399ba66dc6415b054cf5d68c82270626d0e9150e75c808435f`
on Sparklina, PyTorch 2.13.0+cu130, CUDA 13.0, NVIDIA GB10, driver 595.84.
This deliberately differs from the target serving image: the diagnostic checks
PQ against independently captured target outputs. Native threads are bounded
to one and PB-assigned affinity is preserved.

| Check | PB action | Actual result |
| --- | --- | --- |
| Before arithmetic diagnostic | `be5ed565392b97e0850474728cf53d3f1242b6c10b20d950dcd5954a4c45ccdb` | rc 0; original failure reproduced |
| Regression before fix | `bd44408bac75641bc74c79963cf0ee2e98d4800c778e387ce5c5a331a5f3e26f` | Four CPU/CUDA cases failed as expected |
| Regression after fix | `e9b8fe8567c646308c86bb3628764e3030608e35b35065bc6f6ef8b7337f15cb4` | Four passed, no skips, rc 0 |
| After arithmetic diagnostic | `d1c9f2ba1b7934161fddc88319a81bebe3094e8861e03193f9c5ae23cbd39e4b` | Native bit parity, rc 0 |
| Native panel, packed/streamed joint and architecture tests | `600ecfcae8673cfabd10e38aa66448fa981ce0902ea5b90b75b0cee66c4c8b1c` | 107 passed, no skips; x86 CPU, four workers, rc 0 |
| Before profile | `6eabab316edeb1715fe4683df6f2a711ac714b10abe7d6ad3b2c72e330b5b17c` | rc 0, measurement admission |
| After profile | `cf062332e7a1fab076baaef3ce43521593a2bd44dfb0725364f75b2976198342` | rc 0, measurement admission |

Diagnostic artifact hashes: `diagnostic-01.json`
`fb22bc8e41f7794cd59ee9a44d7736a5eced60f3ddb10702b9104d71f771d243`;
`diagnostic-after-01.json`
`e955dfe266e5f704e38624549d9543019f2d8eec049fcd34ceed3e73ddbefeb6`.
Profile JSON hashes: `profile-before-01/profile.json`
`3f2b5bcc584da62b7be0f3700f375cd6fb83176fdaa75e7c1792bfaea7cad02f`;
`profile-after-01/profile.json`
`cf012fe7a2c30dddd19adc3f8c038131086169894af3d095c4374c5f92d40ec5`.
`verification-01.json` records independently rehashed profile/trace/table files,
CAS payload bytes and receipt hashes, and actual CUDA kernel counts.

## Scoped before/after profiling

The benchmark calls only `fp8_dynamic_activation_qdq_vllm(...).dequant.to(input.dtype)` on the
same captured GPU-resident inputs. It uses 20 warmup calls, one 20-second timed
CUDA-event window per phase with synchronization every 32 calls, then 32 calls
under `torch.profiler` CPU+CUDA. The measured event interval includes host
launch gaps. Source-before is commit `5a788c47`; source-after is `fa5d88e1`.
Each measurement requests one CPU, 8 GiB total memory and 4 GiB GPU memory with
PB measurement admission on Sparklina. The before arm precedes the after arm;
these are single windows without repeated/interleaved arms or error bars.

| QDQ-only metric | Prefill before | Prefill after | Decode before | Decode after |
| --- | ---: | ---: | ---: | ---: |
| ms/call, timed CUDA interval | 0.194001 | 0.123573 | 0.114600 | 0.079243 |
| CUDA kernels / profiled call | 15 | 13 | 15 | 13 |
| Actual kernel duration / profiled call, µs | 175.285 | 122.497 | 19.687 | 16.949 |
| Peak allocated bytes | 30,443,520 | 26,248,704 | 12,649,472 | 12,640,768 |
| Mean device power, W | 33.342 | 30.789 | 12.289 | 13.132 |
| Approximate QDQ calls / device joule | 154.6 | 262.8 | 710.1 | 961.0 |

`netdata-before-after-01.json` retains raw Netdata queries and samples for both
Sparky and Sparklina, covering each phase's integer-second interior (19 samples
per context): GPU power, CPU user/system/iowait, load, available memory and swap
I/O. Both boxes show zero swap I/O. Sparky stays at 4 W GPU power; its CPU
user/system means range 1.0–1.5% / 0.8–1.0%. Sparklina user/system means are
5.4–5.8% / 0.5–0.6%, with approximately 117,200 MiB available. Its low power
compared with the approximately 140 W envelope and the decode launch gaps do
not establish GPU saturation. This repair removes quantization operations but
does not resolve the helper's launch-bound decode behavior.

Calls/joule divides timed call throughput by the phase-interior Netdata mean
power; it is a coarse device-energy estimate, includes idle device power, and
is not whole-system energy. The profiler and power series support only this
local helper comparison. No full-model KL, bpp, serving throughput or winner
claim follows; corrected downstream costs and native panels must be regenerated
and gated with their own complete receipts.
