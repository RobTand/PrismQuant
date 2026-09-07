# Original anchor verifier versus bound identities and a namespaced reader

On Sparklina's GB10, the equal-work fourteen-cell verifier took a median
3.994050 s with the original reader and 3.199550 s with per-unit source/H
binding and the separately loaded reader: 1.2483× throughput, or 19.89% less
wall time. All original verification records remained exactly equal. This
qualifies the reader and identity reuse on these original inputs; it does not
measure whole-model preparation, parallel intake hashing, or joint Aura.

## Workload and controls

The two original dense units are `model.layers.0.feed_forward.w1` and `w3`,
with their fourteen original campaign wires and PWC renders. Source weights
and the canonical 512×512 calibration capture are prefetched on CUDA; the
existing PWC owns resident rendered weights. Both arms call the same
`verify_anchor_render`, rehash the original render file, transfer the PWC
render to CUDA, independently derive the expected producer identity and
compare the decoded wire with the original render. Binding work is timed in
the optimized arm. No additional cache or worker dispatcher is introduced.

The immutable plan binds cost, checkpoint, census, capture and token files.
The original producer remains at its original package path. The optimized
consumer is separately loaded under a SHA-specific module namespace; its
source SHA is `14df443217e2a6a1bc4857532f0b5ead7fa7f5755dbd61e96605659940a8a1bc`.
Every original producer source file and all producer module objects loaded
before the consumer remained unchanged. Consumer provenance never substitutes
for the producer's expected encoding identity.

After retaining the first parse pair, the harness runs AB, BA, AB, with the
same `perf_counter` and CUDA synchronization in both arms. It then records a
separate torch CPU/CUDA profile for each arm. A 50 Hz py-spy sampler observes
the whole process. No deterministic per-call profiler runs during timing.
The container is
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
PB admitted four CPUs, 12 GiB total memory and an 8 GiB GPU subset under
measurement isolation on the submitting Sparklina host. Native threads were
bounded to one, and PB-assigned container affinity was preserved.

| Pair | Original seconds | Optimized seconds |
|---|---:|---:|
| First parse, retained separately | 8.938738 | 7.438363 |
| Measured 0, AB | 3.987748 | 3.199550 |
| Measured 1, BA | 3.994050 | 3.213332 |
| Measured 2, AB | 3.998542 | 3.170379 |

The first pair is not claimed to be disk-cold. Across all ten phases, all
fourteen records match exactly, including source tensor, rendered tensor,
encoding identity, wire SHA and render-file SHA. Both arms separately refuse
render, source, settings and wire corruption: eight refusals. No cases were
skipped. CUDA allocation peaked at 923,015,168 bytes; reserved memory peaked
at 968,884,224 bytes. The complete PB scope peaked at 2,628,898,816 bytes,
used 66.354 CPU seconds over 68.261 wall seconds, and had no OOM events.

## Profile and host evidence

The sampler captured 3,354 samples with zero errors. Aggregated stacks inside
`before_verify` and `after_verify` include the first-parse and explicit-profile
phases, so these counts are diagnostic rather than timing estimates for the
three measured pairs. `grid_digest` appears in 96 of 1,303 original-arm
samples and zero of 1,047 optimized-arm samples. `tensor_identity` appears in
202 original-arm samples versus 86 optimized-arm samples. Optimized decoding
still dominates: `read_unit_artifact` occurs in 591 optimized-arm samples,
with 236 leaf samples in NumPy `_sum` and 152 in wire `_from_bits`.
The remaining wire-unpack work was reported to the Tessera owner and the
full-run coordinator; it is outside this reader's measured change.

Raw Netdata series from both hosts and per-phase summaries are retained.
Sparklina's measured arms show about 6% aggregate CPU activity and 11–13 W
interpolated GPU power against the approximately 140 W envelope. This verifier
remains CPU limited. GPU power samples are ten seconds apart while measured
arms last three to four seconds: their interpolated energy integrals cannot
resolve an arm-level work-per-joule comparison. No energy improvement or GPU
saturation claim is made. This bounded result also does not support a 20×
whole-run speedup.

## Reproduction and attributable artifacts

Source commit: `1e017316`. Submit from Sparklina because this is a same-device
measurement, using a complete checkout of that commit:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd CHECKOUT --cpus 4 --demand mem_gb=12 --gpu --gpu-memory-gb 8 \
  --measurement --detach -- bash experiments/pq322_anchor_verify_ab.sh \
  SAMPLER_OUTPUT --plan PLAN --plan-sha256 PLAN_SHA256
```

The plan is
`/mnt/shared/tessera-measurements/first-model-20260907/full-model-joint-aura/qualify-reader-ab-plan-01.json`,
SHA `b6f11eb7af4229319dd48e048b76abe88d608ffce9ee54740de31bb96b1a7e94`.
Reruns require a new output directory in a newly hashed plan; the harness
refuses to overwrite prior results.

All paths below share
`/mnt/shared/tessera-measurements/first-model-20260907/full-model-joint-aura/`:

- `qualify-reader-ab-01/results.json`, SHA
  `5ab6c302dfd418c437fec1936168ecd86b0360726ad2e6205d29a545e8aac8f8`.
- `qualify-reader-ab-01/verified-evidence.json`, SHA
  `234bde783b2dde6d3a2b02ee09f955194fbd3592da655acf61d00af0ae61384a`,
  seals results, both torch traces/tables, both-host Netdata and sampler files.
- `qualify-reader-ab-01/profile-00-{before,after}.{trace.json,operators.txt}`.
- `qualify-reader-ab-01/netdata-both-hosts.json`, including raw host series.
- `qualify-reader-ab-sampler-01/{stacks.raw,profile-result.json,child-exit.json,sampler.log}`.
- `anchor-identity-reader-cpu-receipts.json`, retaining the prior 64-test
  combined CPU gate, 33-test and three-test narrower gates, and harness syntax
  check. These are CPU checks; GPU qualification is the separate action below.

PB action `c9ed62c27e35d4abc52dbbef32e110d5c3b60aa6075af7916796484904628997`
finished with exit 0 and complete resource-scope cleanup. The real child and
sampler both exited 0. The 1,899-byte CAS payload SHA is
`e609fa6266adf72d88631c93a88708667d115696e9fdee5426b682161bc4362a`;
receipt SHA is `94fe88e396a557c45010b824184a8601982414600094f828ae66c2324b00a4a8`.
Payload bytes, receipt body, sampler artifacts and result records were
independently checked after terminal completion.
