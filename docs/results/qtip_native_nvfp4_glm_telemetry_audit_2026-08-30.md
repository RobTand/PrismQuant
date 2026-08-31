# QTIP-native NVFP4 and GLM numeric telemetry audit — 2026-08-30

**Status: accepted one-Linear QTIP result; the historical GLM rows below are
superseded by the authoritative 2026-08-31 addendum; GLM evidence remains
research-only; no serving or comparative-performance verdict.** All hashes in
the historical body were recomputed from the files on `sparky`.
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

## 2026-08-31 authoritative GLM addendum

This addendum supersedes the GLM result, profile, telemetry, and attribution
rows above. It does not change the accepted one-Linear QTIP result. The three
replacement baseline publications and the FP8-CB/TCQ publication all bind
PrismaQuant producer commit
`2f955fa7c073799e494110ff81029027955ee85d`, finalized manifest SHA-256
`a66f800827b92383985ce205004cd2d70b63bcc5e19cada6b05a8162401ee5b0`,
corpus artifact SHA-256
`0d3c08aed48e8d0b540d0705c305cc3197f77c250b07dd7a07e55345f5ddd94e`,
and importance SHA-256
`dad7818dd11ea8f853bd1869f41189ca3de4a2d10deda52cfef563f63496a9dd`.
Each is final, nonpartial, exit 0, and covers exactly 9 dense plus 24 routed
tensors.

### Replacement result and profiler receipts

| Measurement | Authoritative result | Derived receipt or summary | In-process profile | Scope |
|---|---|---|---|---|
| E2M1 same-grid | `/home/rob/dq-runs/numeric-authoritative-2f955fa/sparky/e2m1-scaffold-v1/result.json` — `d7dd0478d5e3ef3b55946f42e161f46ab0c62c1fff2fd57babb5cb11704ee332` | `analysis/coding-gain.json` — `e4a2af3d734fa7266cc8eb790148be5a3d86b80979a14d47bf0e0fbea78d7d60`; V2 receipt file `3fc1e6650367bdcb33a5170c431a815050c00d629080b00343d9ea0904c08bfc`, internal `4fbf36e4f4ab0636d99c668feaf6c3924c86a3e850bf131a5c490df3fb955f43` | py-spy `c12cedcc550011649c76c125005a41ebedc7b9f988c55a98768bc8bf1fcd9175` (621.24 s main sample; 601.905 s algorithm) | Same-rate E2M1 coding gain only. |
| E2M1 near-four | `/home/rob/dq-runs/numeric-authoritative-2f955fa/sparky/e2m1-high-v1/result.json` — `af0b0893401dc22c771ada019d181b2f4a09b64ea8797484543d45e877f21ac0` | `analysis/near-four-summary.json` — `08ae60d76d90a712cc8751ca76f13800a49834dee844a512fa7bf4eda85b0a48`; V2 receipt file `d17f8869dbeda7597148b8d60f2382edb4906ddb82444a2ee8a0af60d475a59e`, internal `bfe8214596bc685230f7ac657db400747614e657fe4bbf61b109611503632344` | py-spy `865e74d0e8a34d628acd0688ff3b737f58d7b336ea40ce3024716e95fc4fa065` (476.33 s main sample; 466.264 s algorithm) | Quality and offline-driver energy only. |
| Learned FP8-CB | `/home/rob/dq-runs/numeric-authoritative-2f955fa/sparklina/fp8-learned-v1/result.json` — `6f717f53dbc3401c72c6fac978b2f36a68368fe76ede13f26efb24b6e2971d6a` | Population summaries are closed in the final result | py-spy `e6521fd7466a0894476d9d02aa27697a322ce95069c3acd9aa79f84f437cd9a3` (1,168.14 s main sample; 1,156.735 s algorithm) | Fixed-vs-learned offline quality only. |
| FP8-CB vs E4M3-TCQ | `/home/rob/dq-runs/numeric-authoritative-2f955fa/sparky/fp8-cb-tcq-v1/result.json` — `3e37ebeb24f575d802461c14c8d844fe5ba0fd7029a0a3f0b2bfe4d2a9befe18` | `analysis/fp8-cb-tcq.analysis-receipt-v1.json` — file `f8d58799363bbb36f0b57d8c637a162931511015bf6604f87d352f6254890ec2`, internal `804bd192bdd41672c82fda3d772e88bbc480543cd4f8b545c0562b40cc530dc4` | py-spy `b13df57960f0042f2232f1c36ded7c5e6c2ba060be7d5d3bec674d8b07ef3c3f` (219,697 main-thread samples/2,196.97 s; 2,181.061 s algorithm) | Artifact-exact W\*A16 frontier closure only. |

The V1 same-grid and near-four derived receipts remain retained but are
superseded. Their V2 replacements bind analysis kind to plan and parse/hash
each source from one stable `O_NOFOLLOW` descriptor; numeric values did not
change.

### Replacement telemetry and quarantine

The following are the only admissible telemetry publications for the three
replacement baselines:

| Campaign | Manifest | Active-host Netdata power | Active-host pqteld power |
|---|---|---|---|
| E2M1 same-grid on Sparky | `telemetry-v3/` — `318505d2684bf64251207db8b9d80404f265e9dd662cca8f16e32898a1805772` (629 s) | 24.6024 W mean, 50 W max, 15.491 kJ; 17.57% of 140 W | 25.4434 W mean, 47.96 W max, 15.999 kJ |
| E2M1 near-four on Sparky | `telemetry-v1/` — `37fac51b2d7874393e8f73d85de5e66903a339003b3008636d8c78a7961bc89c` (498 s) | 22.9359 W mean, 55 W max, 11.436 kJ | 21.0886 W mean, 39.28 W max, 10.498 kJ |
| Learned FP8-CB on Sparklina | `telemetry-v1/` — `8f17d9523e45ce642cd1bfa6a8d0f74a7821b349a1a5f05101020697edb96933` (1,174 s) | 45.2677 W mean, 71 W max, 53.182 kJ; 32.33% of 140 W | 45.7926 W mean, 67.47 W max, 53.755 kJ |

For E2M1 same-grid, the retained `telemetry/` directory is forbidden: a
cross-date glob applied an older pqteld schema header to current rows and
shifted `power_draw_w` onto temperature. `telemetry-v2/` is also forbidden:
capture aborted on an older source containing nonnumeric rows and produced no
manifest. Only hash-checked `telemetry-v3/` supports the replacement figures.
The near-four replacement had no other Sparky GPU workload; the concurrent
learned-FP8 run was on Sparklina and is peer context only. The learned-FP8
replacement had no other Sparklina GPU workload; Sparky work is peer context.

The completed FP8-CB/TCQ telemetry envelope is exactly
`[1788148309, 1788150517]` (2,208 seconds), published at
`/home/rob/dq-runs/numeric-authoritative-2f955fa/sparky/fp8-cb-tcq-v1/telemetry-v1/`.
Its manifest SHA-256 is
`a9c1e01bd12c11bf990c46db9fcd8c392be1edcaffdf188652ae86f0438644aa`;
capture JSON SHA-256 is
`55b027d5731fca260503c616ed7002334175d24a31b87af3ab198cf8a3350592`.
This first CB/TCQ telemetry publication has no invalid or superseded sibling
directory.

| Host/instrument | Series SHA-256 | Samples | Mean / p95 / max | Energy |
|---|---|---:|---|---:|
| Sparky Netdata | `a13e96f4456c916e62284819935b9c515c14e6047d62caa2360152abcf922dec` | 2,209 | 23.0511 / 48.9 / 74 W | 50.914 kJ |
| Sparky pqteld | `d7dd7ae83ca0dfe64c674936e761e36e5a60914fdcae9450f33a42883a29fa13` | 4,416 | 22.8822 / 48.39 / 72.97 W | 50.519 kJ |
| Sparklina Netdata, peer context | `db8af9bbe3e4583f5f33101edbd2018a527f91f67c9babbecc0dfac1aa1b7fbf` | 2,209 | 6.4611 / 14.0 / 36 W | 14.268 kJ |
| Sparklina pqteld, peer context | `b9b060989707918c5c4d5b702950f35ca4dc520e481de150c90f0cf1fd2ec6b2` | 4,416 | 6.8662 / 14.27 / 36.67 W | 15.158 kJ |

The midnight-spanning pqteld extractor parsed each daily source under its own
42-column header and required exact schema equality before concatenation.
Sparky selected-row source hashes are `2ed0b04a927f708060455a471c6a58b049bce217c979bac56ae64a29e7c5eaa6`
and `8bb4d703e0e5a1947301c810c74cddeb3a19aba9eddc5a4e55fdad36fde7e252`;
Sparklina's are `df225a7fc0661a97e60e6159b52d0a70175471014f004e2b98c7b2239c74e66f`
and `fde1b1b102ad02f8a02667b3dbb106e3a5f07d2aaa6a937d322b95b04584e247`.
The capture helper SHA-256 is
`f1d4153f9414a4d5e54e37a5aabc47d870b392bfa3392803af11e7ea50c41131`.

### FP8-CB/TCQ numeric conclusion and execution identity

All 264 tensor × rate × scale-bracket × learned-book-price exact-byte
frontiers cross. All four dense/routed × 4/5-bit final cells are therefore
`NO_VERDICT`: **no objective-free family replacement on this artifact-exact
single draw**. FP8-CB supplies lower-byte points and TCQ higher-SNR points;
this is neither equality nor TCQ domination, and it does not rule out a
budget-specific preference. The independent theorem audit is
`/home/rob/dq-runs/deliberation/numeric_closure_fable_2026-08-31.md`, SHA-256
`36cffef74cea2a89f2784ba9bd94c8e1c37e502b6bc437500aad333491cce719`.
The report's contemporaneous campaign-unrun and branch-status prose is
superseded by the completed artifact and receipt above; its theorem-level
frontier analysis remains the cited result.

The final result has checkpoint self-digest
`c1d6699de67bcac88b084cd10b0dc3cc9b418aa0568febf8612e4c7c034a86ff`
and settings identity
`913e070ff6bd74506bc0156f9105ed453fa2af86cb0bdbc39d6efcd675fd5712`.
It ran in container
`8d5a2b79f98d24f10635bbe6fa7c339a8361377f18926f281ad4e2503d4bac89`;
the Docker client and container exited 0 and `OOMKilled=false`. Final Docker
inspect SHA-256 is
`bfa6e728e8805d2aa6efdef2e1710100b8dc85fa14db2d7b90507652e84dbc95`.
Launch attestation file SHA-256 is
`f495263965cba1a6b63d05c86fb328b8f488b3337d78a055090f6af9a84c6ee1`
with internal identity
`1cfad16122829dd0f3554a77d5b630a5005af78a57b706eca886bd756d8c510d`.
The independent analyzer is tracked at commit
`511385b5c916ee12f06771a68fb302d60e5e161a`, verifier SHA-256
`3b28992cdf91893cc6de54d73ac9492b3935c002297debf760ccb8fa59cece2f`.

All replacement telemetry is offline-driver box evidence. It establishes no
served-kernel throughput, saturation, process-exclusive energy, work per
joule, runtime parity, Gridbook readiness, or cross-implementation performance
claim. The numeric result is a W\*A16 weighted-SSE screen and is not W8A8,
KL/PPL, serving, or model-general evidence.

### 2026-08-31 frontier-geometry and durability addendum

The immutable result's `NO_VERDICT_exact_byte_frontiers_cross` string is a
legacy predicate label, not literal geometry. A from-scratch Fable audit of
the completed artifact found zero true crossings in 264 comparisons: TCQ is
strictly higher-SNR at every byte budget shared by the two families, while
fixed FP8-CB remains the cheapest point just below TCQ's minimum footprint.
That uncovered point is sufficient to refuse objective-free family
replacement, so the four `NO_VERDICT` cells remain correct. The median
minimum-footprint `CB−TCQ` offsets paired with the SNR table above are
`-0.000432/-0.273869` bpw (dense R4, production/two-tier),
`-0.000429/-0.273866` (dense R5), `-0.002217/-0.275655` (routed R4), and
`-0.002324/-0.275762` (routed R5). The production sliver is
0.0002–0.0023 bpw across tensors and narrows to 277 bytes; it must remain
attached to any summary of the dB advantage.

The audit is
`/home/rob/dq-runs/deliberation/fp8_cb_tcq_actual_fable_2026-08-31.md`,
SHA-256
`b4f4b83380dcfaa1fba51a280ad00091c09f771818b70010995c97acd4616d11`.
It independently reproduced all 264 verdicts, all integer footprints, and the
complete provenance chain. The Sparky evidence directory was subsequently
mirrored byte-for-byte onto Sparklina at the same absolute path. Its 31-file
sorted hash manifest is stored beside the mirror and has SHA-256
`4ac1df35ecccaa8cb203e92bf583b8cefec940a104df9df1d637231efa16525c`.
The result and receipt are no-replace evidence files rather than git-tracked
files; the analyzer/verifier source is what is tracked. No serving, runtime,
KL/PPL, work-per-joule, replication, or family-replacement claim follows.
