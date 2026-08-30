# QTIP-native NVFP4 and GLM numeric telemetry audit — 2026-08-30

**Status: accepted one-Linear QTIP result; GLM quality evidence remains
research-only; the immutable GLM E2M1-v2/learned-FP8-v1 results have
`execution_identity_attested=false`; no serving or comparative-performance
verdict.** All hashes in this note were recomputed from the files on `sparky`.
Power is reported because it was measured, but it is not converted into a
speed, saturation, or work-per-joule claim.

## Durable result wording

On the matched Qwen3-0.6B layer-0 `q_proj` isolate, QTIP-derived BlockLDLQ with
an ordinary native NVFP4 terminal reduced activation-output MSE from
`3.2870235736481845e-4` to `1.3293506344780326e-4` (59.5576%) and increased
activation-output SNR by 3.93163 dB at the same exact serialization rate,
4.500030517578125 bits per quantizable weight. This is a one-Linear,
activation-source/FP32-Hessian research result, not a served W4A4, model-wide,
KL/PPL, or runtime result. Weight MSE increased 9.45959%, so the evidence is
specifically Hessian/activation-aware error shaping, not uniformly better
weight reconstruction. Arms A and B were identical, and the production-scale
Arm C and seven-level-heuristic Arm C2 were identical; this run therefore does
not establish extra value from the wider scale heuristic.

On the GLM-5.3-Flash BF16 corpus, the production-plane E2M1 trellis approached
but did not beat the scalar NVFP4 reference below 4.5 bpw. At body rate 3.9375,
the median exact rates were 4.437853 bpw for dense tensors and 4.439620 bpw for
routed tensors; median weighted-SNR deficits to NVFP4 were 0.236197 dB and
0.298524 dB, with zero wins among 9 dense and 24 routed tensors. These are
quality-only observations. The near-four run overlapped QTIP R2 for about 18
seconds and has no attribution-clean performance window, so its encode times,
power, energy, throughput, and work per joule must not be cited.

The learned FP8 result is also measurement-only, with no serving verdict. At
its approximately four-bit rung it beat the fixed FP8 control on all 9 dense
and 24 routed tensors, by median 0.061383 dB and 0.083460 dB respectively, at
median learned rates 4.008464 and 4.011719 bpw. It is not an EXL3 comparison
and does not change any production format or runtime gate.

## Receipt and profiler evidence

| Measurement | Authoritative result | In-process profile | Valid scope |
|---|---|---|---|
| QTIP-native NVFP4 R3 | `/home/rob/dq-runs/qtip-native-nvfp4-v2-r3-64679a7-telemetry/acceptance.v2.json` — `98ad857d0addac04d22c318bbd0dffff3fb62cebacf83b35dda356b2208cd8ca`; raw receipt `/home/rob/dq-runs/qtip-native-nvfp4-v2-r3-64679a7/receipt.v2.json` — `77912b7def1a0951dbeb067c50d0d617bd2458945ffd8170938d480828d1a54b` | py-spy `/home/rob/dq-runs/qtip-native-nvfp4-v2-r3-64679a7-telemetry/py-spy.speedscope.json` — `f85c0cab950a38fb23a62c118b3c29622c23c690e15e8aa414c4e2b6a468403c`; Torch trace `/home/rob/dq-runs/qtip-native-nvfp4-v2-r3-64679a7/profile/one_linear_trace.json` — `38ad6e85a4289c6161983d0819b81672528e15f60307ae9f0570a7c9d8f6ec63` | Accepted schema-v2 one-Linear numerical result. No serving or performance claim. |
| GLM E2M1 high-rate | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/glm-e2m1-highrate.json` — `097b1863ff6c1ba27cecc0fb897e825bc360edb8d09e9b3596951c9bc5730bcd` | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/glm-e2m1-highrate-v4.speedscope.json` — `edc6bc486132703ca47e9da9f374fb82986144350623d291d9043addde165450` | Completed 33-tensor dense/routed-separated numerical sweep. Legacy schema v2; execution identity is not attested. |
| GLM E2M1 near-four | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/glm-e2m1-near4.json` — `d9356eee75f94c07fe11cfdab4f70a72357e2092a7e6846677aec0668211e3cd` | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/glm-e2m1-near4.speedscope.json` — `f4c879e1ba040adf364c9c0cfd03cecdf5793d9dcf8d89fa64bc487bb38ff645` | Quality only. Legacy schema v2 with unattested execution identity; QTIP R2 overlapped the beginning, so all performance attribution is forbidden. |
| GLM learned FP8 | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/glm-fp8-learned-balanced.json` — `3437be1473e1a880d5a6d338bc93f85902ea661be982d040681834426baf1a94` | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/glm-fp8-learned-balanced-v2.speedscope.json` — `f7ef868f00d55794b7f16f34d09880f682f5bd63991a3e6880764172820662ad` | Measurement complete, explicitly no serving verdict. Legacy schema v1 with unattested execution identity; telemetry has a short contaminated tail. |

The QTIP acceptance binds PrismaQuant commit
`64679a7c0d2ee29fd5d76b0db4f35ede0190989e`, exporter SHA-256
`4ec1173f3736fc8fe89342b888fd4855c582e00e5e6adc2599460fc3630ef9d5`,
QTIP-native source SHA-256
`81684bab3414443e67844d5e3e42e6341325d82583fb32912bfd19c21389cac4`,
the immutable container digest, model and activation hashes, calibration
identity, command, GB10 UUID/driver, and every published member. The receipt
is the publication commit marker; the external acceptance is the queue's only
completion marker.

The GLM rows deliberately have no retroactive acceptance receipt. Their exact
result hashes can be bound to the retained corpus and current source files
post hoc, but that cannot prove which source bytes executed. Future E2M1 v3
and learned-FP8 v2 results close this gap prospectively by binding the corpus,
active driver/loader/integration identity and transitive frozen-codec closure,
then replaying those bindings immediately before no-replace final publication.

## Box telemetry

Netdata energy below is the manifest's one-second sample-sum estimate. Its
`nvidia-smi` collector refreshes at roughly five seconds, so adjacent rows are
not independent. pqteld energy is trapezoidal integration of its retained 2 Hz
rows, whose power source refreshes at roughly one second. The two instruments
are deliberately reported separately.

| Window | Requested epoch interval | Sparky Netdata power | Sparky pqteld power | Sparklina Netdata power | Attribution |
|---|---:|---|---|---|---|
| GLM E2M1 high-rate | `[1788127110, 1788127386]` | 277 power samples; mean 37.8487 W, p95 60.0 W, max 63.0 W; 10,484.1 J; 27.0348% of 140 W | 552 rows; mean 39.8673 W, max 61.72 W; 11,000.9188 J; 28.4767% | mean 83.9812 W, p95 89.0 W, max 106.0 W | Clean with respect to the later QTIP/near4 collision. Sparklina load is independent box context, not energy attributable to the Sparky job. |
| GLM learned FP8 | `[1788127521, 1788128670]` | 1,150 power samples; mean 55.9330 W, p95 79.0 W, max 82.0 W; 64,323.0 J; 39.9522% | 2,298 rows; mean 57.2145 W, max 81.44 W; 65,729.1166 J; 40.8675% | mean 79.3848 W, p95 87.8 W, max 104.0 W | Result ended at `1788128660.124855`; the padded telemetry tail includes QTIP R2 from `1788128661.898201` and near4 from `1788128666.683847`. Full-window energy is not strictly FP8-attributable. |
| GLM E2M1 near-four, historical capture | `[1788128661, 1788128844]` | 184 power samples; mean 43.4348 W, p95 64.4 W, max 68.0 W; 7,992.0 J; 31.0248% | 366 rows; mean 38.3136 W, max 78.82 W; 6,987.0252 J; 27.3668% | mean 70.3370 W, p95 83.6 W, max 85.0 W | **Non-performance evidence.** QTIP R2 overlapped near4 for 17.9979 s, and the capture includes padding outside near4. Values establish overlap/load only. |
| QTIP-native R3 | `[1788128844, 1788128862]` | 18 power samples; mean 8.05556 W, p95/max 15.0 W; 145.0 J; 5.75397% | 36 rows; mean 11.3583 W, max 22.05 W; 195.8530 J; 8.11310% | 8 finite power samples; mean/p95/max 89.0 W | Starts after near4 completion and is the accepted QTIP window. It is too short and too lightly loaded for a serving or saturation claim; Sparklina was doing unrelated work. |

### Telemetry artifact hashes

| Window | Manifest | Sparky raw power | Sparklina raw power | pqteld CSV |
|---|---|---|---|---|
| GLM E2M1 high-rate | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/telemetry-e2m1-v2/netdata_manifest.json` — `060d4450d3283bb988fcfbeabecc46ade5951b5b50211487916c5d83472f7a0a` | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/telemetry-e2m1-v2/netdata_sparky_gpu_power.json` — `e6353715c0d0f4cfcfc8feca546038257cb60216fcf5245b8f94a34023b38b6f` | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/telemetry-e2m1-v2/netdata_sparklina_gpu_power.json` — `957daa4f7401d20669366a66db04a6f17bdef984f9edef2328e950047c1bca66` | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/telemetry-e2m1-v2/pqteld_sparky.csv` — `db083b5a8d11c5ebab0980fa7a9b333f67f0b0882790d992402563fc1dab8cb1` |
| GLM learned FP8 | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/telemetry-fp8-learned-v2/netdata_manifest.json` — `0054a9a3908a1225a05b129c15141342a0d477e9de152a0f1b30c62f77264f8e` | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/telemetry-fp8-learned-v2/netdata_sparky_gpu_power.json` — `324f5091c820bcb307b2e5db2acd7f95818556e2f7032f97db6009c78d01bf8b` | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/telemetry-fp8-learned-v2/netdata_sparklina_gpu_power.json` — `8f3df0b598abb49fd89a616b89a2e663b0b74764c3c49d0faa815773b5fadd07` | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/telemetry-fp8-learned-v2/pqteld_sparky.csv` — `82fbb25a9fc220f056540f23f81b33e6efbc204e6a9d256ed573162ec87ab192` |
| GLM E2M1 near-four, contaminated | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/telemetry-e2m1-near4-contaminated-r2/netdata_manifest.json` — `4266f54ef85f3f42ca8b6567713fa91e471953fc464feb563e2250130d888482` | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/telemetry-e2m1-near4-contaminated-r2/netdata_sparky_gpu_power.json` — `a521f0b2dc8869ae9eb78d375fb48f4398200b78e73d32291f283ef51c813714` | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/telemetry-e2m1-near4-contaminated-r2/netdata_sparklina_gpu_power.json` — `80e01915b0432d80875395a47907c2a92b4cab95f1a6073dda7bdbe0276dccbe` | `/home/rob/dq-runs/glm-corpus-20260830/final-bf16-pread-1469b9b-v2/telemetry-e2m1-near4-contaminated-r2/pqteld_sparky.csv` — `aebb4043c9a7ef2e132c7753d09cb465e1616be111e6d7069575e0e538ebed4d` |
| QTIP-native R3 | `/home/rob/dq-runs/qtip-native-nvfp4-v2-r3-64679a7-telemetry/netdata/netdata_manifest.json` — `3026abf638434b34492c0ea55626959a74289f770b2de15a3a408bad4bc9aabb` | `/home/rob/dq-runs/qtip-native-nvfp4-v2-r3-64679a7-telemetry/netdata/netdata_sparky_gpu_power.json` — `3abeae83212bd329f8d01fd23ab1eac7f34fcfe2b0d0eb0eb4fa96cfa8186dec` | `/home/rob/dq-runs/qtip-native-nvfp4-v2-r3-64679a7-telemetry/netdata/netdata_sparklina_gpu_power.json` — `aa890801a20be60dff808159e00e9d41cac96cc36067da9d62a35d55cbb9e68f` | `/home/rob/dq-runs/qtip-native-nvfp4-v2-r3-64679a7-telemetry/netdata/pqteld_sparky.csv` — `c51ecacd7a085f012e79d818b0c436f59650905d145197782269869cb490130a` |

The reconstructed near4 directory also contains
`measurement_scope.json`, SHA-256
`69d73ce35aa7e3a5ed008f268d107119a8388ef02aa3dcffdd8d7925c4cb5616`,
which marks the entire window `non_performance_evidence` and records the R2
queue/receipt identities plus the post-near4 R3 launch boundary. It was
reconstructed after the run from retained series with the same capture helper
and query set as the other GLM manifests; it is not a contemporaneous clean-run
attestation.

## Claim boundary

- The QTIP result supports further matched small-model KL/PPL work. It does
  not qualify a format, producer default, Gridbook runtime, or serving lane.
- The GLM E2M1 result answers only the measured scalar-NVFP4 quality question:
  the gap became small near the four-bit ceiling but did not reverse on this
  corpus. It is not evidence about EXL3.
- The FP8 learned gain is against its fixed FP8 control, not against QTIP,
  native NVFP4, or EXL3.
- No table above demonstrates GPU saturation. On GB10, utilization is not
  diagnostic, and all observed mean powers are far below the approximately
  140 W envelope.
- A performance or work-per-joule comparison requires a new non-overlapped,
  same-work, same-window run. In particular, near4 must be rerun alone with a
  contemporaneous in-process profile and dual-host Netdata/pqteld capture.

Audit provenance: report authored against integration worktree commit
`1f83b331a88d3cf512a858ee434d40ccde06f328`; the accepted QTIP producer run is
pinned independently to `64679a7c0d2ee29fd5d76b0db4f35ede0190989e`.
