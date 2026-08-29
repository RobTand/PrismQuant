# Qwen3.8-27B-PrismaScout-AQUA-20GB release result — 2026-08-24

## Outcome

`rdtand/Qwen3.8-27B-PrismaScout-AQUA-20GB` shipped publicly as a text-only
mixed-precision quantization of `Qwen/Qwen3.8-27B`:

- Model: <https://huggingface.co/rdtand/Qwen3.8-27B-PrismaScout-AQUA-20GB>
- Final Hub commit: `649bcf9b2efda9f2568916ea7a8e034dce41f5ac`
- Producer commit: `5f4f4711a2d2da89da2f6f34f33b22b2b6720cb6`
- Exported model SHA-256:
  `3191a6a4ea85abcebd007be508d2208e95a65c48314e07f7b79ce439d9d38160`
- Publisher snapshot SHA-256:
  `f2985b7f16c7d499f2b022ec5e77c5cc211e6be09fce2c321808749487b015d4`
- Shipcard SHA-256:
  `0135e4dbc379f230faff936c6734a82d51c4efd1b0adcc55bc3b2a0ee871c461`

The exact post-publication Hub census is 32 files and `19,974,334,328`
bytes, including the Hub-managed 1,570-byte `.gitattributes`. This leaves
`25,665,672` bytes below the strict 20,000,000,000-byte whole-repository
limit. All local artifact files match the remote size and content hash, and
anonymous access was verified.

## Artifact and allocation

The source was pinned to Qwen revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The text-plus-MTP carve contains
866 tensors and no vision tensors. The export contains 19 safetensors shards,
1,981 tensors, and `19,950,598,312` tensor-payload bytes.

The exact assignment has 505 Linears:

| scope | format | Linear count |
|---|---|---:|
| language body | NVFP4 | 309 |
| language body | FP8 E4M3 | 179 |
| language body | BF16 | 8 |
| `lm_head` | FP8 E4M3 | 1 |
| MTP sidecar | FP8 E4M3 | 8 |

Body bpp is `5.1596801876` over 24,350,556,160 quantizable body parameters.
The reported body bpp excludes fixed auxiliary assignments, embeddings, norms,
and other non-allocatable tensors. Whole-artifact bytes include them.

Quantization was striped across Sparky and Sparklina, exact-unioned, and
recached before export. Tensor parallelism was not used for quantization. The
allocation map was rendered from the served checkpoint with the website code;
the keyed website renderer commit is
`65a5690899ed828e2fae7bbecabfb332cdb6b048`. The published 946×410 PNG has an
embedded key for NVFP4 W4A4, FP8 E4M3 W8A8, BF16, and blank/not-applicable
cells; its SHA-256 is
`5d77160f1001917ddb8e6b7281432af2f26f00865ca41248d54ac9ebb1172c21`.

## Publication gates

The canonical shipcard verifies 5/5 matching slots for the same model hash:

- native compressed-tensors eager generation;
- native compressed-tensors CUDA-graph generation;
- final no-spec HTTP quality gate;
- all-position gold KL;
- paired gold PPL.

The serving stack was vanilla vLLM
`0.26.1rc1.dev693+g7f7a32cfe.d20260812`, FlashInfer `0.6.18`, and
compressed-tensors `0.17.0`, with no Gridbook or other vLLM plugin loaded.
Native NVFP4 and FP8 kernels were exercised. A separate eager-plus-graph MTP
load/generation smoke passed.

Measured exported-byte quality:

| measurement | PrismaScout-AQUA | Qwen3.8 BF16 | delta |
|---|---:|---:|---:|
| all-position mean KL, 8×512 | 0.0402201669 | — | — |
| confident-position mean KL | 0.02395408 | — | — |
| KL p99 / max | 0.34028574 / 1.86000 | — | — |
| WikiText-2 test PPL | 9.7312222 | 9.3662130 | +3.90% |
| fixed prompt-suite PPL | 4.0743094 | 3.9813028 | +2.34% |
| generation coherence | PASS, 4/4 | — | — |

The MTP live acceptance audit measured 80.79% first-position acceptance and
70.94% total draft acceptance. In the matched 512-to-128 C1 acceptance run,
MTP-3 increased decode throughput from 12.49 to 23.85 output tok/s, or 1.91×.

## Qwen3.6 KL reconciliation

The predecessor card's public `0.0151` was not an all-position release
measurement. Archived records identify it as a 2×128 selection-time sanity
run, and the historical evaluator scored `[:, -1:, :]`: two terminal
predictions total. A later `0.0551` served audit likewise scored only eight
terminal predictions.

The exact released Qwen3.6 artifact was therefore rerun against its pinned BF16
source revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` under the current
8×512, 4,088-position, top-1,024-plus-tail contract:

| metric | PrismaScout-AQUA / Qwen3.8 BF16 | PrismaScout / Qwen3.6 BF16 |
|---|---:|---:|
| mean KL | 0.0402201669 | 0.0779318074 |
| confident-position mean KL | 0.02395408 | 0.05137074 |
| p99 KL | 0.34028574 | 0.81366166 |
| max KL | 1.86000 | 4.97122764 |

The new release has 48.39% lower mean KL under this matched current protocol.
This is a quantization-distortion comparison against each model's own BF16
source, not a base-model capability comparison or proof that AQUA's activation
price alone caused the improvement. The corpus, seed, estimator, context
length, and scored-position count match. The tokenizers differ, so seven of
eight token-ID windows differ. The AQUA measurement used CUDA graphs; the
restored Qwen3.6 rerun used eager execution.

The Qwen3.6 model card's KL record was corrected in Hub commit
`926416032dde5d7c3c655f3696d2593333797a7e`; its canonical successor link was
finalized in commit `849ffb1fc7e166376d231aa5a00ffa6415c1258a`. It explicitly
welcomes continued use of Qwen3.6 and removes the misleading `−68%` claim.

## Direct serving comparison

Both artifacts ran sequentially on the same DGX Spark and pinned container
with CUDA graphs, FP8 KV cache, prefix caching enabled, identical fixed token
lengths, two warmups, and paired seeds. Random prompts for both arms used the
Qwen3.6 tokenizer. Non-spec cells have three measured repeats; MTP cells have
six. C4 throughput is aggregate across four concurrent requests.

| workload | Qwen3.6 PrismaScout | PrismaScout-AQUA | AQUA / Qwen3.6 throughput |
|---|---:|---:|---:|
| prefill C1, input tok/s; TTFT | 3,047.77 ± 0.47; 671.8 ms | 3,035.94 ± 3.11; 674.4 ms | 0.9961× |
| prefill C4, input tok/s; TTFT | 11,529.79 ± 28.31; 675.4 ms | 11,297.28 ± 168.75; 686.0 ms | 0.9798× |
| no-spec decode C1, output tok/s; TPOT | 12.58136 ± 0.00009; 79.40 ms | 12.97836 ± 0.00390; 76.95 ms | 1.0316× |
| no-spec decode C4, output tok/s; TPOT | 48.90230 ± 0.01747; 81.28 ms | 48.99464 ± 0.03962; 81.09 ms | 1.0019× |
| MTP-3 decode C1, output tok/s; TPOT | 24.6393 ± 1.7580; 40.05 ms | 30.9371 ± 2.1886; 31.85 ms | 1.2556× |
| MTP-3 decode C4, output tok/s; TPOT | 83.0368 ± 7.0968; 39.93 ms | 95.9148 ± 12.4466; 35.24 ms | 1.1551× |

Throughput is mean ± sample standard deviation; TTFT and TPOT are means. All
48 JSON files pass identity, request, token, failure, arithmetic, and metric
validation, and all 12 groups are complete. No-spec primary metrics are stable.
MTP output-throughput CV ranges from 7.1% to 13.0%, reflecting seed-dependent
acceptance; the analyzer therefore correctly raises CV flags and exits
nonzero even though coverage and result integrity pass.

## Durable evidence

Primary release directory:
`/home/rob/dq-runs/qwen38-27b-scout-aqua-20gb`

Important evidence:

- Export:
  `/home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/exported-prismascout-aqua-20gb`
- Final local census:
  `/home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/publish/census-local.json`
- Hub verification:
  `/home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/publish/hub-verification.json`
- KL contract reconciliation:
  `/home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/logs/kl_contract_reconciliation.md`
- Qwen3.6 deep KL result:
  `/home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/comparison-gold-q36-fullkl-20260824/q36_prismascout_allpos_eager070.json`
- Direct comparison results:
  `/home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/comparison-throughput-q36-20260824`
- Direct comparison analyzer:
  `/home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/analyze_q36_direct_compare.py`
- Final endpoint ship gate:
  `/home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/ship-gate-nospec-20260824`
- Allocation assignment:
  `/home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/artifacts/layer_config.json`

Final census command:

```bash
python3 /home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/publish/census_artifact.py \
  /home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/exported-prismascout-aqua-20gb \
  --source-dir /home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/source-text-mtp \
  --layer-config /home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/artifacts/layer_config.json \
  --profile-spec /home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/code-5f4f471/prismaquant/model_profiles/specs/qwen3_5_dense.json
```

Frozen-byte publisher dry run:

```bash
python3 /home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/code-5f4f471/tools/publish_artifact.py \
  /home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/exported-prismascout-aqua-20gb \
  --repo-id rdtand/Qwen3.8-27B-PrismaScout-AQUA-20GB \
  --shipcard /home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/exported-prismascout-aqua-20gb/shipcard.json \
  --dry-run
```

Direct comparison audit:

```bash
python3 /home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/analyze_q36_direct_compare.py \
  --json \
  /home/rob/dq-runs/qwen38-27b-scout-aqua-20gb/comparison-throughput-q36-20260824
```
