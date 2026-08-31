# QTIP-informed optimizer, native NVFP4 output

Status: **implemented and measured Arm C research-only one-Linear isolate,
plus a physical transformed-basis PrismaQuant/Gridbook trellis producer; no
production registration, QTIP runtime, or new serving lane.** Arm C tests
QTIP-derived BlockLDLQ with a stock native-NVFP4 terminal. The combined
producer includes an exact block-structured diagonal-Hessian path, emits
canonical `gridbook.trellis.wire.v1` bytes, and validates the
online sign/Hadamard algebra against an external, unpinned Gridbook research
reference outside the current Gridbook 0.9.1 contract. Schema v2 requires a
fresh no-clobber result root and a receipt that binds both source trees,
inputs, calibration, container, host, GPU/driver, command, and quality levers.

Pinned reference: official QTIP checkout
`/home/rob/dq-runs/qtip-reference-20260830` at
`e90c6688c8dfae326a3a81b5eb032db7c6680ec0`. The receipt pins SHA-256 digests
for `math_utils.py`, `ldlq.py`, `finetune.py`, and `bitshift.py` as well.

## What transfers

- QTIP's block-unit-lower factor, reverse BlockLDLQ schedule, and later-block
  error feedback are offline optimizer choices. The harness transfers that
  recurrence. It deliberately uses PrismaQuant's activation preprocessing and
  `damp=1.0`, not QTIP's default `sigma_reg=0.01`, so all controls share the
  same local objective.
- QTIP's random signs, input/output Hadamards, and `SU`/`SV` factors require
  inverse transforms during execution. They cannot decorate one native-NVFP4
  Linear without runtime sidecars. PrismaQuant now has a strict research
  producer for Gridbook's `gridbook.qtip-online-hadamard.v1` sidecar and its
  existing trellis wire; the stock-native Arm C isolate intentionally remains
  unrotated so its mechanism stays attributable.
- QTIP's tail-biting Viterbi codebook correlates choices across a 256-element
  tile and serializes trellis state. Native NVFP4 independently decodes each
  E2M1 nibble under one E4M3 scale per group of 16, so the QTIP trellis is not
  representable as standard native bytes. That QTIP wire is excluded and is
  not decoded by Gridbook. The combined research producer retains
  PrismaQuant/Gridbook's existing `TCQ_E2M1_R256` trellis wire while borrowing
  QTIP-derived optimizer and rotation ideas; it would not emit QTIP bytes.

Quartet II is literature motivation only: no Quartet II source, implementation,
or result is used here. EXL3 was source-reading context only and is neither a
dependency nor an implemented or measured arm.

## The NVFP4 scale byte

The group-scale byte is useful optimization capacity but not spare storage.
The stock decoder reconstructs every value as
`global_real * e4m3_group_scale * e2m1(nibble)`. Arbitrary payload bits in the
scale byte therefore alter all 16 reconstructed weights; there is no decoder-
invisible scalar channel. It already costs exactly 0.5 bits/weight. The legal
way to use more of that capacity is to search more E4M3 scale choices.

The harness reuses the exporter's existing search rather than creating a new
codec. Arm C uses the production FourOverSix candidates `{6, 4}`. Arm C2 uses
seven existing max-to-E2M1-level heuristics
`{6, 4, 3, 2, 1.5, 1, 0.5}`. This is **not** an exhaustive search over positive
E4M3 byte values, a Quartet II implementation, or a Quartet II reproduction,
and snapped-scale scoring remains disabled to match the production recipe. C2
therefore tests only whether those five extra existing PrismaQuant heuristic
candidates help the QTIP-derived recurrence at the same native payload rate.

## Matched arms

1. native NVFP4 RTN + production JSO, where JSO is only the final native
   group/tensor scale search (there is no GPTQ optimizer in this arm);
2. current PrismaQuant GPTQ + static activation order + JSO;
3. QTIP BlockLDLQ with a stock NVFP4 terminal at every reverse group-16 step;
4. C2, the same recurrence with the seven-level scale heuristic, not an
   exhaustive E4M3 scalar-channel search.

All arms emit only `weight_packed`, `weight_scale`, `weight_global_scale`, and
`input_global_scale`. Exact payload bpw includes both float32 scalars. The
metric is a BF16/FP32 activation-Hessian weight isolate, not a served W4A4 or
full QTIP model-quality claim.

The Arm-C receipt labels the ordinary activation-output proxy and the damped proxy
as **untransformed original-Linear space**. No transformed-Hessian number is
reported because this stock-native isolate has no online rotation. The
separate combined producer transforms both weight and Hessian consistently
and proves the post-decode serve algebra. Its runtime transform exists only in
an external, unpinned Gridbook research branch and is not a production
contract.

```bash
python -m research.qtip_native_nvfp4_2026-08-30.native_nvfp4_ldlq \
  --weight weight.safetensors --weight-key model.layers.0.self_attn.q_proj.weight \
  --activations q_proj_inputs.pt --activations-key inputs --device cuda \
  --output /run/qtip-native-v2/receipt.json \
  --artifacts-dir /run/qtip-native-v2/native-fields \
  --profile-dir /run/qtip-native-v2/profile \
  --publication-root /run/qtip-native-v2 \
  --durable-root-uri sparky:/absolute/host/path/qtip-native-v2 \
  --host sparky --container-identity image@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
  --model-id Qwen/Qwen3-0.6B --calibration-manifest calibration.json \
  --prismaquant-checkout /code --prismaquant-commit 0123456789abcdef0123456789abcdef01234567 \
  --qtip-checkout /path/to/qtip
```

The calibration manifest must use
`prismaquant.calibration_identity.v1`, carry non-empty `dataset`,
`capture_precision`, and `calibration_hash` fields, integer `nsamples`,
`seqlen`, and `seed`, and include `identity_sha256`, the canonical SHA-256 of
all other fields. The code and QTIP Git metadata must be readable inside the
container. Every output must resolve beneath `publication-root`. Files publish
individually without clobber; the receipt publishes last and is the commit
marker. A directory containing fields or a trace but no receipt is incomplete
and must be abandoned rather than resumed in place.

## Accepted schema-v2 one-Linear result (Sparky, 2026-08-30)

The accepted r3 run uses Qwen3-0.6B BF16
`model.layers.0.self_attn.q_proj.weight`, shape `[2048, 1024]`, and the same
256 FP32 UltraChat activation rows as the historical isolate. The publication
binds PrismaQuant commit `64679a7`, QTIP commit
`e90c6688c8dfae326a3a81b5eb032db7c6680ec0`, the calibration identity,
container, command, device/driver, exact quality environment, field hashes,
Torch trace, py-spy profile, and both-host Netdata series. Every arm is exactly
4.500030517578125 bpw over 2,097,152 quantizable weights.

| Arm | activation-output NSSE | regularized-H NSSE | weight SNR | bpw |
|---|---:|---:|---:|---:|
| A native RTN + JSO | 0.002579497 | 0.003893172 | 21.1268 dB | 4.5000305 |
| B current GPTQ + static order + JSO | 0.002579497 | 0.003893172 | 21.1268 dB | 4.5000305 |
| C QTIP BlockLDLQ + native terminal | **0.001043210** | **0.002936581** | 20.7342 dB | 4.5000305 |
| C2 C + seven-level scale heuristic | **0.001043210** | **0.002936581** | 20.7342 dB | 4.5000305 |

Against A/B, C reduces activation-output NSSE by **59.56%** (+3.9316 dB)
and the regularized-H proxy by **24.57%** (+1.2246 dB), despite losing 0.3925
dB of weight-only SNR. The result therefore isolates the value of error
direction under a non-diagonal activation metric. C2 is byte-identical to C:
the five additional max-to-level scale heuristics add no value on this
Linear. This is not an exhaustive legal E4M3 scale search.

The raw receipt is
`sparky:/home/rob/dq-runs/qtip-native-nvfp4-v2-r3-64679a7/receipt.v2.json`
(SHA-256
`77912b7def1a0951dbeb067c50d0d617bd2458945ffd8170938d480828d1a54b`).
The independent acceptance receipt is
`sparky:/home/rob/dq-runs/qtip-native-nvfp4-v2-r3-64679a7-telemetry/acceptance.v2.json`
(SHA-256
`98ad857d0addac04d22c318bbd0dffff3fb62cebacf83b35dda356b2208cd8ca`).
Its Netdata manifest SHA-256 is
`3026abf638434b34492c0ea55626959a74289f770b2de15a3a408bad4bc9aabb`.
The approximately 23-second isolate is quality evidence only; it is not a
throughput or work-per-joule benchmark.

## Gridbook online-transform reference

The separate Gridbook research branch at commit
`84a78c745e53676f87397e937456d7f2fc6ddd3f` implements strict
`gridbook.qtip-online-hadamard.v1` metadata and reference transforms without a
core-vLLM patch or PrismaQuant import. At M=32, K=N=4096, block size 256, and
20,000 iterations on Sparky, the input transform measured 0.150208 ms median
and the inverse output transform 0.153952 ms median; the pair adds 0.304160 ms
in this launch-heavy Torch reference. Pqteld measured 21.104 W mean, 26.33 W
maximum, 166.57 J, 240.13 transforms/J, and 31.475 million values/J. This is a
transform-only reference—not a fused kernel, graph, GEMM, or serving result.
The authoritative directory is
`/mnt/shared/dq-runs/gridbook-qtip-hadamard-ref-84a78c7-20260830T1747EDT-r5`.

The PrismaQuant combined producer uses that exact metadata contract, emits
canonical `gridbook.trellis.wire.v1` bytes, reparses and decodes those same
bytes, and proves original-basis serve algebra. It remains one-Linear,
unregistered, unpinned, and production-ineligible.

Prepared scaffold schemas v2 retain and re-transform the source weight at the
encode boundary; exact equality to the retained transformed weight is
mandatory. BlockLDL artifact schema v3 binds the producer/encoder/wire/format
source set and the declared callable closure (plus the scale-grid closure when
enabled). Module-level substitution of an execution or gateway callable fails
closed rather than publishing under the original source hashes.

## Executable Arm E quality campaign

`arm_e_quality_campaign.py` is the research-only matched runner for scalar
native NVFP4 (A), stock-native BlockLDL (C), and rotated BlockLDL plus the
canonical E2M1 trellis wire (E). Its only CLI inputs are a closed
`prismaquant.research.arm_e_quality_campaign_manifest.v1` and the optional
`--preflight-only` switch. Unknown fields, implicit seeds, noncanonical
alphabets, output paths outside the declared root, CPU quality execution, and
an unavailable CUDA/Triton encoder all fail closed.

The runner selects the largest legal q256 whose **complete** wire is no larger
than either native control. Headers, full schedules, offsets, alphabet bytes,
E4M3 group-scale bytes, and row padding are charged. Current exact preflight:

| shape | selected q256 | complete wire bpw |
|---|---:|---:|
| Qwen `[2048,1024]` | 992 | 4.377437592 |
| routed down `[4096,2048]` | 1008 | 4.438612938 |
| routed gate/up `[2048,4096]` | 1016 | 4.470870018 |
| dense gate/up `[12288,4096]` | 1016 | 4.469103336 |
| dense down `[4096,12288]` | 1016 | 4.469774723 |

Qwen q1000, q1008, and q1016 all occupy 4.502437592 bpw after 16-byte
row alignment and therefore exceed the 4.500030518-bpw native controls. The
q992 selection is deliberate.

The Qwen mode uses the exact preprocessed activation rows for both its Hessian
and activation-output metric. The GLM corpus has activation second moments but
no activation rows, so its receipt always marks activation-output unavailable
and reports only raw-importance-weighted, regularized-diagonal-H, and plain
weight metrics. A deterministic Rademacher matrix checks transform orientation
only and is explicitly excluded from quality.

Each E seed publishes one `.trellis` file, reopens it, canonical-reserializes
it, and decodes those same bytes before its tensor-result commit marker can
publish. The campaign receipt publishes last. A persistent identity claim and
deterministic GPU replay from hash-checked pinned inputs permit crash resume
only when the native controls, Hessian, BlockLDL receipt, canonical wire,
inverse transform, metrics, and verdicts reproduce exactly. Historical timing
values are noncomparative telemetry and are not replayed. This is a local
integrity/recomputation contract, not an external signature or independent
proof that the prior GPU execution occurred.

For GLM, preflight opens one finalized-corpus descriptor, hashes that entire
descriptor against the manifest once, retains the inode for every selected
pair `pread`, and rechecks the whole descriptor before receipt publication.
This closes pathname replacement without re-reading the multi-gigabyte corpus
once per tensor.

### Current GLM feasibility boundary

The GLM source contract is a retained strictly positive diagonal vector, not a
self-asserted dense matrix. If the declared input transform uses normalized
block-Hadamards `H_b` and signs `S_b`, then
`H_b S_b diag(d_b) S_b H_b^T = H_b diag(d_b) H_b^T`. The signs cancel and the
full transformed Hessian is exactly block diagonal. The producer constructs,
factors, and consumes one transform block at a time while preserving the full
reverse feedback recurrence within each block and across every output row.
It requires the block size to be a power of two, 256-aligned, and a divisor of
K. Ordered offsets, sizes, source-diagonal hashes, transformed-block hashes,
feedback hashes, and `D` hashes are receipt- and replay-bound.

Consequently, dense-down K=12288 with B=4096 uses three B-by-B factor groups
and never constructs a K-by-K Hessian. The avoided FP32 K-by-K matrix alone is
603,979,776 bytes. A 1,811,939,328-byte reference tensor quantity is recorded
for shape `[4096,12288]`, but it is explicitly neither a liveness-derived upper
bound nor a measured allocator peak: the current producer retains additional
W-shaped tensors and CUDA library workspaces. This makes the full 33-tensor GLM
census shape-contract supported; its memory gate remains unmeasured until a
CUDA pilot records the actual peak. It does not say that the campaign has run,
met its time/energy gates, or produced a quality result. The two-tensor pilot
remains the shortest safe first measured execution.

The boundary is deliberately narrow. Arbitrary dense or rank-two Hessians,
off-block entries, nonpositive diagonals, and forged/reordered group receipts
fail closed. The Qwen dense-H path still refuses K>4096. The exact block
reduction also does not make dense local `D` coordinate-additive: the current
256-state terminal does not prove dense-`D` optimality, `dense_block_D` remains
unsupported, and no EXL3-beating claim follows.

### Closed manifest and Sparky commands

The following creates the exact Qwen manifest from the accepted R3 inputs on
Sparky. Run it only from a clean committed checkout; the manifest binds the
commit and the runner rechecks every imported source at claim, completion, and
receipt publication.

```bash
PQ_TREE=/home/rob/pq-arm-e-quality-campaign
PQ_COMMIT=$(git -C "$PQ_TREE" rev-parse HEAD)
PQ_RUN=/home/rob/dq-runs/arm-e-qwen-${PQ_COMMIT:0:7}
mkdir -p "$PQ_RUN/contract" "$PQ_RUN/output" "$PQ_RUN/profile" "$PQ_RUN/netdata"
cp /home/rob/dq-runs/qtip-native-nvfp4-v2-r3-contract-64679a7/calibration_identity.v1.json "$PQ_RUN/contract/"
jq -n --arg commit "$PQ_COMMIT" --arg run "$PQ_RUN" '{
  schema:"prismaquant.research.arm_e_quality_campaign_manifest.v1",
  campaign_id:("arm-e-qwen-" + ($commit[0:7])),
  mode:"qwen_one_linear",
  research_opt_in:"qtip_trellis_online_hadamard_one_linear_v1",
  execution:{device:"cuda:0",host:"sparky",container_identity:("sha256:" + "cf3f7f83e6820fa75aae249393e8fa4840af4562192203a1aed3f2082f3ea2f9"),prismaquant_checkout:"/code",prismaquant_commit:$commit},
  output:{root:"/publication",durable_root_uri:("sparky:" + $run + "/output")},
  input:{kind:"qwen_one_linear",model_id:"Qwen/Qwen3-0.6B",weight_path:"/model/model.safetensors",weight_key:"model.layers.0.self_attn.q_proj.weight",activations_path:"/activation/model__layers__0__self_attn__q_proj.pt",activations_key:"inputs",calibration_manifest:"/contract/calibration_identity.v1.json"},
  recipe:{family:"TCQ_E2M1_R256",layout:"tight_offsets",rate_policy:"largest_q256_complete_wire_not_above_min_native_A_C_bytes_v1",schedule_policy:"uniform_column_schedule_bresenham_v1",alphabet_policy:"canonical_full_e2m1_value_order_rate3_v1",scale_rule:"static_6",terminal_metric_mode:"diag_block_D",input_block_policy:"largest_power_of_two_divisor_capped_v1",max_input_block_size:4096,output_block_size:256,sb_chunk:64,determinism_mode:"on",tailbite_candidates:4,backend:"triton",point_route:"full",buffer_blocks:2,damp_fraction:1.0,glm_algebra_witness_rows:2},
  seeds:[
    {label:"primary",input_seed:4660,output_seed:22136},
    {label:"holdout-1",input_seed:4661,output_seed:22137},
    {label:"holdout-2",input_seed:4662,output_seed:22138}
  ]
}' > "$PQ_RUN/contract/qwen.manifest.v1.json"
```

CPU-safe contract and exact-byte preflight:

```bash
docker run --rm --entrypoint /usr/bin/python3 \
  -e PYTHONPATH=/code -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PQ_TREE:/code:ro" \
  -v /home/rob/models/Qwen3-0.6B:/model:ro \
  -v /home/rob/dq-runs/aura-lmhead-fix-0p6b/act:/activation:ro \
  -v "$PQ_RUN/contract:/contract:ro" -v "$PQ_RUN/output:/publication" \
  -w /code sha256:cf3f7f83e6820fa75aae249393e8fa4840af4562192203a1aed3f2082f3ea2f9 \
  -m research.qtip_native_nvfp4_2026-08-30.arm_e_quality_campaign \
  --manifest /contract/qwen.manifest.v1.json --preflight-only
```

Gold quality execution, wrapped in the host's in-process CUDA/NVTX profiler:

```bash
ARM_E_START=$(date +%s)
nsys profile --trace=cuda,nvtx,osrt --sample=none \
  --force-overwrite=false -o "$PQ_RUN/profile/arm-e-qwen" \
  docker run --rm --gpus all --ipc=host --entrypoint /usr/bin/python3 \
  -e PYTHONPATH=/code -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PQ_TREE:/code:ro" \
  -v /home/rob/models/Qwen3-0.6B:/model:ro \
  -v /home/rob/dq-runs/aura-lmhead-fix-0p6b/act:/activation:ro \
  -v "$PQ_RUN/contract:/contract:ro" -v "$PQ_RUN/output:/publication" \
  -w /code sha256:cf3f7f83e6820fa75aae249393e8fa4840af4562192203a1aed3f2082f3ea2f9 \
  -m research.qtip_native_nvfp4_2026-08-30.arm_e_quality_campaign \
  --manifest /contract/qwen.manifest.v1.json
ARM_E_STOP=$(date +%s)
```

The trace does not clear the telemetry gate unless it contains CUDA events.
Capture both boxes over that exact epoch interval; GPU utilization is retained
only as context, while power is judged against the approximately 140 W GB10
envelope:

```bash
for host in sparky sparklina; do
  for chart in nvidia_smi.gpu_power_draw system.cpu disk.io system.net; do
    curl -fsS "http://${host}:19999/api/v1/data?chart=${chart}&after=${ARM_E_START}&before=${ARM_E_STOP}&format=json" \
      -o "$PQ_RUN/netdata/${host}-${chart}.json"
  done
done
sha256sum "$PQ_RUN/profile/arm-e-qwen.nsys-rep" "$PQ_RUN"/netdata/*.json \
  > "$PQ_RUN/telemetry.sha256"
```

The GLM pilot and full 33-tensor census subsequently completed. The largest
`[4096,12288]` tensor used three ordered 4096-column factor groups without a
global Hessian. Maximum allocated CUDA memory was `6,686,095,360` bytes; the
profiled census window was 652 seconds. The full receipt SHA-256 is
`84987783448da1db3aa6d0dcf8a409f7ab383de45fb2329f1bc667a58dd87b32`.
Thus the structured implementation is physically feasible on the measured
Sparky configuration.

Quality was negative under the preregistered separate population gates. Arm E
minus Arm C median raw-importance SNR was `-0.442704 dB` for dense tensors
(2/9 wins) and `-1.803932 dB` for routed tensors (3/24 wins); regularized-H
medians were `-1.350010 dB` and `-1.944914 dB`. Arm E therefore remains a
research ablation and must not replace Arm C. The full evidence, profile and
both-host telemetry identities are recorded in
`docs/results/qtip_native_nvfp4_arm_e_glm_2026-08-30.md`. This census contains
activation second moments rather than activation rows, so it establishes no
activation-output, KL/PPL, serving, throughput, or work-per-joule claim.

## Preliminary v1 one-Linear result (Sparky, 2026-08-30; superseded)

The following run discovered a promising seam, but its v1 receipt did not bind
the PrismaQuant source commit, full calibration identity, container/device
identity, or all quality-affecting configuration. It is retained as historical
evidence only and is **not an accepted v2 result**; r3 above supersedes it.

Source was Qwen3-0.6B BF16
`model.layers.0.self_attn.q_proj.weight`, shape `[2048, 1024]`. The 256 FP32
activation rows came from the existing UltraChat calibration capture
(`nsamples=8`, `seqlen=512`, BF16 capture, calibration hash
`41d25e036f3aaea2d3dafcf37831018a`). All four artifacts contain 2,097,152
quantized weights and exactly 1,179,656 payload bytes: 4.5000305 bpw including
the two float32 scalars.

| Arm | output NSSE on X | damped H proxy NSSE | weight SNR dB | payload bpw |
|---|---:|---:|---:|---:|
| A RTN + final JSO | 0.002579497 | 0.003893172 | 21.1268 | 4.5000305 |
| B GPTQ + static order + JSO | 0.002579497 | 0.003893172 | 21.1268 | 4.5000305 |
| C QTIP BlockLDLQ + native terminal | 0.001043210 | 0.002936581 | 20.7342 | 4.5000305 |
| C2 C + seven-level scale heuristic | 0.001043210 | 0.002936581 | 20.7342 | 4.5000305 |

On this calibration isolate, C reduced activation-output NSSE by **59.56%**
(+3.93 dB) and the exact regularized-H proxy by **24.57%** (+1.22 dB) versus
A/B, while weight-only SNR fell 0.39 dB. B's final bytes equal A's; v1 did not
record a pre-gate candidate or a gate counter, so it cannot prove why. C2's
final field digest equals C's, so the five additional max-to-level heuristics
found no gain here; this says nothing about exhaustive legal E4M3 scale-byte
search. This is preliminary evidence for the QTIP feedback seam, not held-out
KL/PPL or a served-quality claim.

Preliminary v1 receipt, superseded by the v2 evidence contract above:
`sparky:/home/rob/dq-runs/qtip-native-nvfp4-20260830/qwen3_0p6b_l0_qproj.json`
(SHA-256 `2cb266f637e6bcf2cd60cbdd63490a62c00d8d746bc59b1d08eac663464edba4`).
The original stock-field safetensors are beside it under `fields/`, but were
published by a root container as mode `0600`; v2 normalizes artifacts to
`0644` and records relative plus durable paths and file hashes. The original
in-process trace is `profile/one_linear_trace.json`. Queue log:
`/mnt/shared/pq-queue/logs/qtip-native-nvfp4-qwen06-l0q-v4.log`. No timing or
energy claim is made from this run.
