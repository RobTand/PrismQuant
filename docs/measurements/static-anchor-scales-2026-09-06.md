# Static activation scales in transient anchors — CPU regression evidence

Issue: #262. Source base: `3cb0d359`. Tessera development pin:
`ba582d476a3b6db9057ebd1385dc52926f171451` (unchanged).

A Tessera-only static A4 plan previously reached production scoring without a
calibrated activation maximum. The shared streamed renderer now asks the existing
resolved-format predicate whether static maxima are required, and gathers them
across the planned layer's fused siblings even for one-sibling transient requests.
Native NVFP4 weight-global computation keeps its independent native-format guard.

## Verified CPU evidence

PrismaBuild ran four pytest-xdist workers with `--dist worksteal --durations=15
-ra`, native math threads bounded to one, four CPUs and 12 GiB reserved. Worker:
`sparky`, aarch64, Python 3.12, cu130 environment with CUDA hidden by CPU admission.

Before the fix, action
`584abc9a2d6f755c48540d3d5ef901c9a319dec1036dc8dce55b661f760cc419`
exited 1: **2 failed, 2 skipped**. Both batched and one-sibling cases failed with:

```text
ActivationScaleContractError: production cache render score @ TESSERA_E2M1_K1_R768:
'model.layers.0.self_attn.q_proj' is served under the static
e2m1_group16_ue4m3_static activation contract but has no calibrated activation
maximum (got None); refusing to price it under a dynamic quantiser the runtime
never executes.
```

After the fix, action
`558d5bfb93ff975bd3324491dadc4aa912cb989e1da39af372ce8d0c490400e3`
exited 0: **19 passed, 2 skipped**, 8.38 seconds pytest time. Files:
`tests/test_streaming_static_activations.py` and
`tests/test_streamed_cost_checkpoints.py`. No collection errors were reported.
Both skips: `real Tessera anchor render needs CUDA`.
The CPU cases stub weight rendering only; static activation quantization,
production scoring and joint identity use their real implementations. Existing
checkpoint tests cover replay and identity refusal.

The output CAS payload was independently SHA-256 verified:
`37fbf2d78c375b02571b1c1bd51dcf8ef9239302e439be768cf761b13db4c678`.
Logs and receipts are retained under
`/mnt/shared/tessera-runs/issue-queue-receipts/<action-key>.{log,receipt.json}`.
The scoped test-plugin installation reported existing llmcompressor dependency
conflicts with host Torch 2.11.0+cu130 and Transformers 5.6.0; this result is the
bounded CPU regression population, not qualification of that host environment.

PB snapshots materialized the three tracked external calibration symlinks with
identical verified target bytes, then restored the original symlinks. None of
those materializations is committed. Their snapshot hashes are:

| Calibration file | SHA-256 |
| --- | --- |
| diverse-v1.jsonl | e09a138a4903c4af66a3bf2f9367185f3432224391f1dfe8c94ccc29d99315ba |
| xdom-fit-v1.jsonl | 8425e172ee00a85c92795c6350c862de5c0c6f0b9d16b323c1fd4551d8b11058 |
| xdom-gate-v1.jsonl | d44ce5d5f8f263cd37c3c028528c08f5ca8a6960aaa698e225d6e56f18ccbc05 |

## Pending GPU evidence

Corrected regression/checkpoint action:
`60172adc2c4d5f5efc126db42df1786aab4a8b2d61cdf626df7ddff0ceb51ddc`.
Existing streaming/resident parity action:
`367511b949336a23b9f239c5c683ae64be47d114b21236f2b7629a834fe69dc7`.
Both use the known producer container image
`sha256:cf3f7f83e6820fa75aae249393e8fa4840af4562192203a1aed3f2082f3ea2f9`,
with four CPUs, one GPU and 16 GiB reserved through PB; actual terminal results
must be checked separately. At this record's creation neither had completed.
No GPU correctness, served quality, performance or promotion result is claimed.
