# Native MoE workspace receipt reader — 2026-09-07

The actual admitted original-wire native receipt failed PrismaQuant consumption
with `KeyError: 'workspace'` (PB `d47966d399202329ed0a1110495e491483d76a23cdacc0a7aea3b1a4fa515623`).
The reader and its fixture expected top-level workspace duplicates absent from
Tessera's published v1 shape. The corrected reader validates the workspace in
the already exact-matched frozen panel, then retains the independent observed
resource digest/byte checks. Measurement bytes remain unchanged; see #328.

The receipt-shaped regression reproduced two intended failures against unmodified
main through PB `48dd4734584f5aeb05e5485331a0354abffeec701eb0cf28eb84e7a3a9564e1d`.
The corrected native panel and architecture/staleness tests passed **84 tests,
zero skips or missing collection** through
`6dbb1c7966708b055e453704fdf57de4c32e5db57d359d4f187881be63e1d8af`.
These CPU tests used the existing `pq322_cpu_checks.sh` ARM path and immutable
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`
container on Sparklina with GPU visibility disabled, four CPU workers, a 12 GiB
reservation and native threads bounded to one. Warnings were existing Torch
JIT deprecations and pytest's read-only cache directory.

Actual unchanged receipt consumption and touched-module compilation passed PB
`5c5626d6e391486c1d5ebb174becb53651b808d429161814b47338c27776fe5f`,
using `experiments/pq329_native_receipt_consume.sh` in the same CPU-only container.
The output at `/mnt/shared/tessera-measurements/first-model-20260907/native-moe-panel-r1024/consumed-03.json`
has SHA256 `4ec5fe3f02048549ab3abe8a4926572aac91ab89d3333a45d74d8b89617b6ddb`.
It retains prefill/decode medians 1.906431973/0.263167992 ms, residency 353,042,432
bytes, persistent workspace 23,068,672 bytes, and scratch 7,367,680/14,848 bytes.
These values come from the original GPU receipt, not from the CPU reader run.
`runtime_table_admissible` remains false and `full_model_resources` remains null.

`workspace-reader-328-audit.json` beside that output, SHA256
`061b971bc210256174b29e14b1ce60cf75a29ce226c80ba0c7b4ea556f5bcda9`,
records actual exits, complete scope cleanup and rehashed successful CAS payloads.
Earlier unclaimed x86 validation submissions were withdrawn through PB while
that worker was unavailable and replaced with the existing qualified ARM path;
no GPU measurement was repeated for this repair.
