# Mixed LFM native arithmetic prerequisite (#258)

Ten real native calls passed on NVIDIA GB10 / SM121 in
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`,
with Torch 2.13.0+cu130 and vLLM 0.28.1rc1.dev397+gfd4a15126.d20260904.
The loaded Tessera source was exactly `6faa5ce314cadeee8a190cbeadcf6cde3a333efb`;
all 1,020 source-file hashes were checked before and after the call.

This is deterministic synthetic-operand ABI and shape evidence. It does not
encode the model, test quantized weight reconstruction, qualify a serving cell,
establish generated quality, or provide allocator runtime prices. Source shape
headers came from the actual LFM2.5-8B-A1B checkpoint. The native activation
quantizer and Torch GEMMs were real; no vLLM operators were stubbed.

| Family | Source owner geometry, N x K | M values |
| --- | --- | --- |
| FP4 | Fused dense FF w1/w3, 14336 x 2048 | 1, 64 |
| FP4 | Dense FF w2, 2048 x 7168 | 1, 64 |
| BF16 | Convolution input, 6144 x 2048 | 1, 64 |
| BF16 | Convolution output, 2048 x 2048 | 1, 64 |
| BF16 | Fused attention q/k/v, 3072 x 2048 | 1, 64 |

All outputs had the expected shape and finite entries. The FP4 calls used the
native FP4 quantizer and block-scaled GEMM; BF16 calls used FP32 accumulation
and the resident route's epilogue ordering. Peak allocated CUDA memory was
234,881,536 bytes. The profiler contains the actual block-scaled CUTLASS and
BF16 kernel activity. These cold, instrumented calls are not a speed benchmark.
Host Netdata observations for both fleet machines remain with the raw trace.

## Attempt and cleanup disposition

PrismaBuild selected Sparky for all three preflight attempts. The first stopped
at Docker exit 126 because newly staged NFS directories were not traversable
by the root-squashed daemon. The second used the established parent mount but
still could not import the unreadable source. Both stopped before arithmetic.
Only the task-owned source directories and required ancestors were made
traversable; the immutable source bytes did not change.

Attempt 03, action
`89d5ce963e622226c1816616b476a812980fb2e75c1fe7fe2f355a1c33ee49ef`,
ran all ten calls and the child exited zero. Its host action exited one while
removing root-owned temporary profiler files **after** copying the results.
The original failed action remains failed. Container and PB scope cleanup were
already complete, and both-host telemetry had no errors.

CPU-only recovery action
`4b2ffe85c77956057e0fe3419bdfd48e5f18523c7107943b3f6df81c2b963eae`
verified the three retained files against the exact owned temporary directory,
removed that directory, and exited zero with complete scope cleanup. Placement
was tied to that existing local directory, not GPU balancing. No GPU work was
repeated. Receipt:
`cf69b282225c58b63fd0a4fd8cbaad1ab1814b93dc6005e32bb83db16a14a94c`;
the actual 626-byte payload hashes to
`1c3cda1200fa26d57214a2c1032bf9b3872335b79d89324b706f04959920bb8b`.
The host wrapper now creates the result directory itself so it can remove
root-written files, and records artifact cleanup separately from container
cleanup.

The checked-in [attempt summary](artifacts/lfm-mixed-native-2026-09-06/attempt-summary.json)
binds every retained file and original terminal record. The successful native
[result](artifacts/lfm-mixed-native-2026-09-06/result.json), profiler table, host
identity and recovery result are alongside it. Full traces and telemetry remain
under `/mnt/shared/tessera-measurements/mixed-lfm-237-2026-09-06/native-preflight-03`.

The complete plan and actual static scales are a separate
[calibration result](lfm-mixed-preparation-2026-09-06.md). Full-model export,
actual owner census and paired serving remain obligations of #253.
