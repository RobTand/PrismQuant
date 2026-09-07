# Streamed joint AURA microbatch qualification, 2026-09-07

The opt-in streamed microbatch implementation in #318 is qualified against
small mathematical fixtures and the actual LFM full-vocabulary source on the
first three canonical calibration sequences. **The full 512-sequence cost run
has not yet executed.** It awaits the independently qualified shared activation
QDQ correction, so a full-cost artifact will price that policy once. These
controls change no production format, runtime pin, default or serving gate.

`probe_microbatch > 0` uses `prismaquant.kl_fisher.global_row.v1`: every global
sequence row has a fixed seed domain and the complete draw's token normalizer.
Local weight, activation and mixed residuals sum while signed across every
partition before squaring, separately for each source Linear. Diagnostics sum
FP32 gradients before taking their norm. GPU logits/recompute graphs are bounded
by batch size; the existing CPU boundaries and probe cotangents still cover the
full draw. The existing source/cache prefetch mechanism remains in use.

The RNG descriptor is independent of partition size. The complete probe identity
also describes source arithmetic: its existing `arithmetic` field now seals the
execution partition. Existing row, paired-candidate and assignment consumers
therefore reject mixed batch sizes, as does checkpoint resume. Legacy zero
retains the original streamed draws; it is not the positive full-size reference.

## Actual source and arithmetic controls

All GPU controls used the known-good image
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`,
Torch `2.13.0+cu130`, Transformers `5.16.1`, BF16 source weights, explicit eager
attention and the original grouped expert backend. Probes are Rademacher,
seeds 7000–7003, all tokens, temperature 1, global normalization. The original
checkpoint, resolved backend, full token artifact, actual subset tensor and
original rendered PWC bytes are sealed in the retained records. The canonical
full token artifact is int64 `[512,512]`, SHA256
`a38312e3b1eeecc2a4363d2a91739ba535388036d4910bffa815d451dcb9a940`.

The fixed-logit CUDA control `9410e1f58090a45151114d7c5b3012bd8be56f798b72a35a811813173fc8ef92`
passed all eight seed/partition comparisons exactly at `[3,512,128000]`:
batch sizes 1 and uneven 2 match size 3 bit for bit. It completed with exit 0,
aggregate reservation 16 GiB and GPU cap 14 GiB. This isolates Fisher probe
construction from model arithmetic.

The resident/streamed control
`2361a4cac58abbafbf18aec3c87365aca12117d95fd092bc0dc919c8cd3e0040`
completed with exit 0 on Sparky, aggregate reservation 64 GiB and GPU cap 48 GiB.
It compares the streamed path to ordinary resident HF execution at each
**same** batch size, using the same three canonical rows and four probes.
Every comparison below is dtype-identical and bit-exact:

| Batch rows | Decoder boundaries | Layer-2 route tensors | Full-vocabulary logits | Packed leaf gradients |
|---|---:|---:|---:|---:|
| 3 | 24 | 3 | 1 | 8 |
| 1 | 72 | 9 | 3 | 24 |

The two physical leaves are `gate_up_proj [32,3584,2048]` and
`down_proj [32,2048,1792]`. Route tensors are the original packed expert inputs,
expert IDs and route weights. Each gradient comparison records both hashes,
probe and global row offset. The qualification record is
`source-control02/results.json`, SHA256
`5acc61c565159e5e02e7480149d358ccb87eb55bd21ab2b8baa4ced352bb27ce`.

**Negative result: resident batch sizes 3 and 1 are not numerically equivalent.**
Decoder layers 0 and 1 match exactly. Layer 2 first differs by at most
0.00048828125 (relative L2 0.0009286), even though that layer's routed inputs,
IDs and weights match exactly. Subsequent layers amplify the difference:
full-vocabulary logits differ by relative L2 0.0758231 and maximum 9.25;
packed gradients differ by relative L2 0.4334–0.4514. The first difference is
inside the unchanged resident layer, not in streaming or probe RNG. This
control attributes the drift to batch-shaped source arithmetic without claiming
a particular low-level kernel instruction caused it. Canonical census/capture
forwards complete `[1,512]` sequences individually; batch size 1 matches that
arithmetic. CPU partition invariance alone would have missed this distinction.

## Three-row cost and memory controls

Before adding the final arithmetic compatibility seal, original rendered wires
were priced for the existing 96 w1/w3/w2 source Linears of layer 2, with
`TESSERA_E4M3_K1_R1024` and BF16. These are bounded diagnostic artifacts, not
full-calibration or native-panel costs. Their shared RNG identity permitted the
cross-size comparison that exposed the source arithmetic difference; final
consumers now refuse that mixed execution.

| PB action | Batch rows | Largest tail | Peak PyTorch GPU allocation |
|---|---:|---|---:|
| `ada12c11bda42cc6106d4d32a363176024304b1e40aeec1bab2cd80d44fc9a80` | 3 | `[3,512,128000]` | 23,714,409,472 bytes |
| `7bbddb3785e70ea6ed9f39418752d0fe3888679f9385e125fdc8d8631f33a18d` | 1 | `[1,512,128000]` | 20,415,026,688 bytes |

Both completed with exit 0 on Sparky under 40 GiB aggregate reservations and
32 GiB GPU caps. Each captured 157,286,400 bytes of host decoder boundaries.
The size-1 path made 12 bounded tail calls; size 3 made four. These measurements
establish bounded tail allocation on actual packed/full-vocabulary geometry.
They are not total physical-memory peaks and are not throughput claims.
Both arms retain CPU/CUDA `torch.profiler` memory/operator traces and both
hosts' Netdata series. Trace export overhead is included in the recorded
profile phase; those durations are not an isolated speed comparison.

The signed per-probe costs differ by normalized L2 0.42472 across batch sizes.
Summing the 96 unary predictions gives 0.0030473746 versus 0.0028390700; neither
sum is measured KL or a coherent whole-block quadratic. The same-size resident
controls above explain why these differences cannot be dismissed as a probe
partition bug, nor treated as comparable finite source arithmetic.

The first size-3 control's derived subset header retained its parent's sampling
counts. Its recorded actual `[3,512]` tensor digest and independently hashed
parent artifact establish the diagnostic input. That subset is not a native
panel input. The harness now adjusts subset counts and int32 draw identity;
size-1 and subsequent derived artifacts use the corrected header.

## Regression and retained evidence

Final targeted CPU validation
`fa5598de04a8f35cefa873b7e8bbd1845982f6b9517a831be0a57f00e4cf80db`
passed **121 tests, no skips**, through PB with four workers/native threads 1
and an 8 GiB aggregate reservation. It covers Fisher all/last/causal scopes,
uneven partitions, an independent packed row oracle with nonzero cross terms,
gradient diagnostics, checkpoint refusal, actual paired/assignment consumers,
legacy streamed/packed behavior and architecture/calibration documentation.
The initial missing-API regression `8bae3225...` failed five tests as expected;
`8549364a...` demonstrated that three existing consumers accepted mixed execution
sizes before the arithmetic seal. Earlier green checks are retained rather
than substituted for the final snapshot.

The first CUDA probe control `1cf4d668...` wrote eight exact comparisons but
ended with exit 137: PB stopped it after GPU-reported use 10,834,935,808 bytes
exceeded its 10,737,418,240-byte cap for two samples. Host OOM counts were zero
and cleanup completed. Its output is partial evidence, superseded by the
successful correctly reserved retry. Source control `25fefb27...` failed before
loading because its `device_map` required absent `accelerate`; it was corrected
to the already proven `from_pretrained(...).cuda()` load path. One intermediate
submission refused a changing checkout before any action executed. None of
these failures is reported as a pass.

[The manifest](lfm-joint-microbatch-2026-09-07.json) hashes the retained outputs,
exact PB requests, sanitized terminal summaries, logs, independently rehashed
CAS payloads/receipts and harnesses under
`/mnt/shared/tessera-measurements/pq318-joint-microbatch-20260907`.
Each PB checkout snapshot seals the historical code used by that action;
retained latest harnesses are convenient entry points, not replacements for
those snapshot identities. Full 512-row validation and final qualified-QDQ
cost production remain explicitly pending.
