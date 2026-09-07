# Persistent-buffer export and resume review — 2026-09-07

The streaming exporter now passes the skeleton's declared buffer dtypes to the
existing reader. A fresh export preserves FP32 buffer values that BF16 cannot
represent. The same fix alone did not protect resumes: old layer payloads were
replayed before the corrected reader ran. The existing render provenance and
cache fingerprint now include `persistent_buffer_read_policy=model_declared_v1`,
so those old payloads are invalidated through the existing manifest comparison.

Both regressions execute the real streaming exporter and safetensors emission:
PB `5b11586cec0e01dc9ab92be36e1bf9d835e485b99dc8ec84cb3f6c473133e8d0`
failed the fresh serialized-byte check before the reader wiring;
`ac627badd15c7967f26bdc2d2956077322049063714af1ae6d38006b5c197bef`
failed the resumed-byte check before the policy stamp (one other test passed).
Architecture construction is replaced by a small CPU skeleton; actual source
reads, cache payloads, tensor sink and serialized tensor bytes are exercised.

After both fixes, review PB `d710557982720e28af08d033a22f8ffe7fcec790d2620264a52aa34bc3b643f0`
and final-main integration PB
`5f82ab7762507ab06822a2ddecae7c05bd78ee12a95e5994639eb416c342202b`
each passed 206 tests, with no skips or uncollected modules. The final run took
10.32 seconds on DL380 CPU, four workers/12 GiB total, native threads one,
Python 3.14.4, Torch 2.11 CPU and Transformers 5.16.1. Dedicated compile action
`c328ca41f2eb978840f2c3bbf1389b77edba4115d1406bc9aea0ea62544d88bb`
also passed. Root independently checked actual exits, canonical CAS receipts,
payload/source bundle bytes, source equality and complete scope cleanup.

Command: `bash experiments/pq322_cpu_checks.sh -n4 tests/test_export_buffer_resume.py tests/test_export_buffer_precision.py tests/test_prismaquant_export_native_compressed.py tests/test_streaming_buffer_precision.py tests/test_pretrained_buffer_initialization.py tests/test_architecture_doc.py tests/test_docs_staleness.py`.
All checks were submitted through the published PrismaBuild client with
portable placement and priority -10.

No LFM compressed-tensors artifact or stock-vLLM load/generation gate was
measured. Issue #311 remains open for that validation. The separate standalone
resume-cache source/requested-dtype binding concern is tracked in #340; this
policy stamp does not claim to solve that provenance contract. Parameter and
format policy are unchanged.
