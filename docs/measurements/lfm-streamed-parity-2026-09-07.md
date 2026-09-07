# LFM source-forward correctness, 2026-09-07

Issue [#310](https://github.com/RobTand/prismaquant/issues/310). The corrected
normal resident and streamed paths produce bit-exact BF16 logits over the full
128,000-token vocabulary for one frozen 512-token LFM input. All 24 decoder
boundaries and all 24 isolated-layer outputs are also bit-exact. This is a
source-forward correctness result, not quantized quality or a performance claim.

## Three causal changes

1. Ordinary Transformers checkpoint loading must initialize missing state.
   Transformers 5.16.1 rematerializes nonpersistent buffers with `empty_like`;
   PrismaQuant's global initialization suppression left the subsequent LFM
   rotary initialization disabled. A poisoned-buffer regression deterministically
   returned -777 instead of 3 before the fix. The shared hook now enables the
   genuine initializer inside checkpoint missing-state finalization using a
   `ContextVar`, and preserves from-config skeleton suppression outside that
   phase. Tests cover loaded-weight preservation, exception cleanup and concurrent
   skeleton isolation. Successful finalization emits the validated
   `prismaquant.pretrained_initialization.v1` descriptor; it is evidence of that
   load phase, not a certificate for subsequent model mutations.
2. Shared materialization and streamed cold/prefetch/cache paths preserve
   model-declared buffer dtype independently of parameter compute dtype. LFM's
   FP32 expert bias remains FP32 when correcting BF16 router scores before top-k.
   A bias value that is itself exactly representable in BF16 does not make BF16
   addition equivalent to FP32 addition. Resident, cold and prefetched regressions
   all failed before this change and pass afterward. The existing layer cache is
   the sole residency mechanism.
3. AURA's streamed builder explicitly selects eager attention, matching its
   resident loader. The shared optional argument reaches both auto-model and
   resolved-class construction; omitted arguments retain Transformers' selection.
   Four CPU tests cover those routes and both explicit/default behavior.

The original failed preflight had maximum absolute logit error 15.78125. After
correct initialization and buffer precision, the diagnostic SDPA arm still
had maximum absolute error 5.5625 against eager. The final normal-path gate uses
no diagnostic initializer context, tensor substitutions or backend mutation.
The original tolerance remains `atol=rtol=2^-6`; the observed error is zero.

## Workload and immutable inputs

- Model: `/mnt/shared/models/LFM2.5-8B-A1B-BF16`, BF16, 24 layers, 32 routed
  experts per MoE layer and top-k 4. Source `model.safetensors` SHA256:
  `c9b9e3c4b3be50b576e6da8c02de1b4223614ffe131d812abf92bb84421f6217`.
- Input: the first row only of
  `/mnt/shared/tessera-measurements/pq300-batch-20260907/capture/calibration_tokens.safetensors`.
  File SHA256: `6b8fe5e0de9f509059f71d151f3e20c45a4fb57e391c8b06a502cefbcc94c98a`.
  That file holds the retained WT2 train 32-by-512 draw, seed 0. This gate does
  not certify every calibration sequence or expert coverage.
- Known image:
  `eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`;
  Transformers 5.16.1, PyTorch 2.13.0+cu130, GB10. PB preserves affinity;
  four reserved CPU cores and native thread limits of one; 40 GiB host demand,
  32 GiB GPU budget. Existing resident-prefetch cache is filled before comparison.

## Validation and retained evidence

All tests and GPU runs used PrismaBuild. Evidence root:
`/mnt/shared/tessera-measurements/lfm-streamed-parity-20260907/`.
The JSON captures contain per-layer comparisons, head/buffer comparisons, actual
model-load descriptor, actual backend, toolchain and CPU affinity. `receipts/`
retains terminal status, logs, cleanup and checked CAS result receipts. `harness/`
retains the exact diagnostic and normal-path scripts and Docker wrappers.

| Evidence | PB action | Result |
|---|---|---|
| Initial isolation | `0b7513a1ed51761af17d30e61db17b31a2425d7c0a8560e6123f56a6f1e458bd` | Diagnostic complete; dtype and attention boundary differences retained |
| Canonical HF/rotary isolation | `c999ff211ab72d3655a38e67dcdc1c88c34c8616f1ab18375123ae487dbc76bd` | Diagnostic complete; eager plus valid buffers exact |
| Five deterministic regressions | `fd39c08ea8929e7b81e8dd5fea553b4f598eed793841e1ffde6cff82fc255efd` | Five expected failures before changes |
| Initializer plus existing initialization tests | `f3c60eb5f869a214d289c17f61f0cacb1a4ac1e81e36c15f7903ed7199cf221d` | 8 passed |
| Buffer, scheduling and architecture gates | `734ca20bbc9f2c5083055fe88e990fddfe2d7958a77d000879fbdd4c7d4acfd4` | 34 passed |
| Backend construction | `0355a3e02bf8faa80509243256273c344953be9616d4496845e3c784de8a093f` | 4 passed |
| Normal paths on original integration base | `9a0c68830fbd23379d2988a2a2751afce36ff0fe756566ced878459acc06e319` | GPU rc0; all comparisons exact |
| Normal paths after rebase to main | `8081e7376fed063a1ec1ffb7b7be7804d857d25e12c66209ee3c58c007fd8222` | GPU rc0; all comparisons exact |
| Five touched production modules compiled | `e04b45dab4f121a135fefa72faa263712626a87fc2fe0080d2edc5a8eebbe2d1` | rc0, portable CPU worker |

The final main-based GPU receipt is
`4e7167cc18d171fb4a9cfbd5888374dbe954c1d49d9b7f6b7c34198b9333592c`.
The separate supported CPU fanout covered streamed joint AURA, streamed cost
checkpoints, calibration intake and wrapper construction: 52 passed, two
skipped because Accelerate is absent; `final-cpu-tests.json` records each action.
The final main-based CPU action
`16362aea7693580b859479388c087b954caa97f005a726f983db639848f5a914`
rechecked the changed contracts and architecture: 58 passed, the same two
Accelerate-guarded wrapper tests skipped. Actual HF backend-construction tests
and the real GPU streamed path still ran successfully. The calibration-intake seam
was present on the original diagnostic base and belongs to its separate PR;
it is not included in this parity delivery.

Two early CPU submissions lacked pytest in the image and are tooling failures,
not regression evidence. Scoped pytest 8.4.2 was then installed. One buffer test
run (`64f06f410fd152bea36d850e81ef95eea0b4cc06416dc8ef0cfc82a8f74922dc`)
reported 34 passing tests but failed CAS publication with `CASTamperError`;
its failed terminal is retained and was reported to the PB owner. The stored
blob subsequently matched its content hash. The replacement action above
completed and published a verified receipt.

Historical activation/Hessian captures produced under broken checkpoint
initialization cannot support model-quality claims. They remain bounded
performance evidence where both arms consumed identical inputs. New quality
captures must record actual successful model initialization and be regenerated;
this correction does not rewrite those historical files.

A separate artifact-writing caller in compressed-tensors export still omits
buffer precision metadata. It is recorded as
[#311](https://github.com/RobTand/prismaquant/issues/311), with the missing
artifact/stock-vLLM serving gate stated explicitly. It is outside this Tessera
source-parity delivery.

Evidence manifest: 26 retained files, SHA256
`6c9b7f7c4c049eed779f5b06c986965afaffd0c68c2d8873b2ca83486971aea9`.
