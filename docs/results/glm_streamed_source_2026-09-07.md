# GLM canonical source streaming qualification — 2026-09-07

The canonical census/capture entry point can use the existing layer streamer
without constructing the complete GLM BF16 model. This qualifies the source
forward and bounded capture semantics; it does not qualify a full GLM capture,
an export or a served artifact.

The fixture uses the real Transformers 5.16.1 GLM implementation, original
checkpoint names, split KDA convolution tensors, unpacked experts and mHC
renames. Native CUDA checks use GB10, PyTorch 2.13.0+cu130, BF16/eager execution,
two original B1 samples of 257 tokens and seven retained prefix rows. Two body
layers cover KDA and DSA, four experts/top two routing, and mHC expansion four;
hidden width is 256. The native gate calls actual CUDA kernels and does not
install the CPU fixture's convolution replacement.

The first valid native comparison failed: shared loading narrowed strict FP32
GLM state, changed routed counts and produced a maximum logit difference of
0.57421875. Source values alone did not expose the defect: several narrowed
values were exactly BF16 representable, but their arithmetic dtype changed.
The shared loader now applies HF's own dtype plan before materialization,
including propagating final convolution precision to its split sources.

| Check | Actual outcome | PrismaBuild action |
|---|---|---|
| Native source versus initial streamed route | Failed counts/maxima/logits | `b479d4…`, retained `gpu-native-02` |
| Diagnostic installed state | Identified strict FP32 narrowing | `381c9eb…`, retained `gpu-native-03` |
| Native source after dtype repair | Exact X/H/count/maxima/logits, 18 units | `07a919fa56ee8736a8a11f53dc37dfe5833f9609ed46de4c1ad4978f8cb3fdd5` |
| Native source with ephemeral metadata and direct packer, nonzero FP32 recurrence/routing controls | All installed state and X/H/count/maxima/logits exact, 18 units | `91bdf285f55b2e2304db79fe2da339c065cf716febecef3ca326356e8bbb04a2` |
| CLI census/capture and source-state tests, CPU | 38 passed | `ef9c6d4a2993155502bdb56e1a1f05b509f3b568007529d28a01aff05ffc11ed` |
| Metadata/source lifetime and pack-layout regressions, CPU | 15 passed | `6aa8b6e756c04f656a83689fe9c208bdd750828ccabdf62604e381985945fbcd` |
| Strict FP32 resident/cold/prefetched/cache replay plus canonical contracts, CPU | 33 passed | `f2a41f64241359cb78b40bf3138266e6e75d06cea028e3200a915959e85768ae` |
| Final architecture/docs, source streaming/prefetch, precision, packing and canonical contracts, CPU; touched-module compile | 84 passed, no skips; compile passed | `f0e450cb779a3822ff466b388b12a62743503d81964091de46dfd8649e1851d2` |

Native receipts, before/after CPU+CUDA traces, both-host Netdata series and
result JSON are retained under
`/mnt/shared/tessera-measurements/glm-streaming-source-20260907/`.
The final native payload SHA256 is
`87603320dafc2a3c28988df5f475525231bcb2f92a3ca18c9f8f944b4df846b4`;
CAS receipt SHA256 is
`c7e3a34d572782523660d803e1778ef24d6abb4cd3b288c995a6ce2bb492416c`.
Terminal exit status and the actual payload bytes were independently checked.
These small qualification timings are not a full-model speed or saturation
claim. Vision and MTP state are outside the text-forward witness and gate.

The representative production source is
`/mnt/shared/models/GLM-5.3-Flash-BF16`: 45 body layers, 42 MoE layers with
288 experts/top eight routing and 36,423 quantizable source Linears. The
proposed 512×512 draw remains unfrozen. A header-only upper bound has roughly
1,709 GiB full H and 237 GiB retained X, plus journal/replacement space; routed
counts need an actual census. Explicit workspace remains an allowance rather
than measured full-model allocation. No full-model census or capture was run
as part of this qualification.
