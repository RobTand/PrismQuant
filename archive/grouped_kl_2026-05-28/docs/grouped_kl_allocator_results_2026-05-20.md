# Grouped-KL Allocator 27B Check (2026-05-20)

## Scope

Model: `/home/rob/.cache/huggingface/qwen36-27b-bf16`

Prior no-FP8 validation work dir:
`/home/rob/dq-runs/qwen36-27b-grouped-validation-20260519T192416-27b`

Fresh FP8-menu validation work dir:
`/home/rob/dq-runs/qwen36-27b-grouped-fp8-validation-20260520T2202`

The prior artifacts measured `NVFP4,MXFP8_E4M3,BF16`. A fresh CUDA run was
then performed with `NVFP4,MXFP8_E4M3,FP8_E4M3,BF16` to decide whether FP8
should enter the production default menu.

## Prior No-FP8 Reproduction

Regenerated allocator cost from the measured grouped-KL payload:

```bash
python3 - <<'PY'
from pathlib import Path
import pickle
from prismaquant.grouped_kl_cost import synthesize_grouped_cost_payload

base = Path('/home/rob/dq-runs/qwen36-27b-grouped-validation-20260519T192416-27b')
art = base / 'artifacts'
with open(art / 'kl_grouped.pkl', 'rb') as f:
    grouped = pickle.load(f)
with open(art / 'baseline_cost.pkl', 'rb') as f:
    baseline = pickle.load(f)
payload = synthesize_grouped_cost_payload(
    grouped,
    baseline,
    source_label=str(art / 'kl_grouped.pkl'),
)
with open(art / 'grouped_cost_intree.pkl', 'wb') as f:
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
PY
```

Result:

- `grouped_cost_intree.pkl` has 504 layer entries.
- 992 format entries use `grouped_kl_share`; 16 entries fall back to baseline
  cost; BF16 entries are zero.
- Max absolute `predicted_dloss` difference versus the prior
  `artifacts/grouped_cost.pkl` is `0.0`.

Allocator rerun:

```bash
BASE=/home/rob/dq-runs/qwen36-27b-grouped-validation-20260519T192416-27b
OUT=$BASE/intree_allocator_validation
mkdir -p "$OUT"
for T in 5.0 5.5 6.0; do
  python3 -m prismaquant.allocator \
    --probe "$BASE/artifacts/probe.pkl" \
    --costs "$BASE/artifacts/grouped_cost_intree.pkl" \
    --model-override /home/rob/.cache/huggingface/qwen36-27b-bf16 \
    --target-bits "$T" \
    --formats NVFP4,MXFP8_E4M3,BF16 \
    --target-profile vllm_packed_moe \
    --pareto-targets "$T" \
    --layer-config "$OUT/layer_config_grouped_intree_${T}.json" \
    --pareto-csv "$OUT/pareto_grouped_intree_${T}.csv" \
    --applicability-report "$OUT/format_applicability_${T}.json"
done
```

Allocator outputs:

| target bpp | achieved bpp | NVFP4 layers | BF16 layers | predicted dloss |
|---:|---:|---:|---:|---:|
| 5.0 | 4.996447 | 226 | 78 | 0.130167534 |
| 5.5 | 5.496609 | 198 | 106 | 0.115344731 |
| 6.0 | 5.994294 | 180 | 124 | 0.103391641 |

The regenerated configs are assignment-identical to the prior grouped configs
when omitted entries are treated as implicit BF16. The only structural
difference is that the current allocator emits 110 additional explicit BF16
visual-encoder entries.

## Recorded 27B PPL

Source logs:

- Baseline: `logs/ppl_baseline_5.0.log`, `logs/ppl_baseline_5.5.log`,
  `logs/ppl_baseline_6.0.log`
- Grouped-KL: `logs/ppl_grouped_5.0.log`, `logs/ppl_grouped_5.5.log`,
  `logs/ppl_grouped_6.0.log`

| target bpp | baseline achieved | baseline PPL | grouped achieved | grouped PPL | delta |
|---:|---:|---:|---:|---:|---:|
| 5.0 | 5.000548 | 7.235166956 | 4.996447 | 7.166811851 | -0.068355105 |
| 5.5 | 5.499085 | 7.137918660 | 5.496609 | 7.130786632 | -0.007132028 |
| 6.0 | 6.000910 | 7.237159819 | 5.994294 | 6.981816097 | -0.255343723 |

The local baseline allocator is non-monotone on this run: 6.0 bpp is worse
than 5.5 bpp by `+0.099241160` PPL. Grouped-KL is monotone across the tested
budgets and gives the largest gain at 6.0 bpp.

## Fresh FP8-Menu Run

Fresh run setup:

- CUDA venv: `/home/rob/dq-runs/venvs/prismaquant-cu130`
- Model: `/home/rob/.cache/huggingface/qwen36-27b-bf16`
- Dataset: `/home/rob/dq-runs/calibration/diverse-v1.jsonl`
- Calibration contract: 8 samples, sequence length 1024, seed 42
- Formats: `NVFP4,MXFP8_E4M3,FP8_E4M3,BF16`
- Target profile: `vllm_packed_moe`

Key logs:

- Baseline cost:
  `/home/rob/dq-runs/qwen36-27b-grouped-fp8-validation-20260520T2202/logs/baseline_cost_rerun_lps4.log`
- Grouped KL:
  `/home/rob/dq-runs/qwen36-27b-grouped-fp8-validation-20260520T2202/logs/grouped_kl_measure_lru8.log`
- PPL:
  `logs/ppl_grouped_fp8_5.0.log`, `logs/ppl_grouped_fp8_5.5.log`,
  `logs/ppl_grouped_fp8_6.0.log` under the fresh run dir

The production cache was reused for existing NVFP4/MXFP8 entries and extended
with FP8, then written as:
`artifacts/production_weight_cache.pkl`. The final cache manifest covered
1488 entries across `FP8_E4M3`, `MXFP8_E4M3`, and `NVFP4`.

Grouped KL completed with:

- `artifacts/grouped_kl.pkl`
- `artifacts/grouped_cost.pkl`
- `grouped_entries=1392`
- `fallback_entries=120`

Allocator outputs on the fresh grouped FP8 cost:

| target bpp | achieved bpp | NVFP4 core | FP8 core | MXFP8 core | BF16 core | predicted dloss |
|---:|---:|---:|---:|---:|---:|---:|
| 5.0 | 5.000 | 222 | 8 | 0 | 74 | 0.15753 |
| 5.5 | 5.499 | 188 | 10 | 0 | 106 | 0.13786 |
| 6.0 | 5.996 | 166 | 17 | 0 | 121 | 0.12421 |

Explicit full layer-config counts, including visual and MTP BF16 entries:

| target bpp | NVFP4 | FP8_E4M3 | BF16 |
|---:|---:|---:|---:|
| 5.0 | 353 | 21 | 240 |
| 5.5 | 317 | 25 | 272 |
| 6.0 | 272 | 40 | 302 |

Held-out WikiText PPL, 65,536 tokens:

| target bpp | fresh FP8 achieved | fresh FP8 PPL | prior grouped PPL | FP8 delta vs prior grouped |
|---:|---:|---:|---:|---:|
| 5.0 | 5.000 | 7.778006904 | 7.166811851 | +0.611195053 |
| 5.5 | 5.499 | 7.686406763 | 7.130786632 | +0.555620131 |
| 6.0 | 5.996 | 7.620808501 | 6.981816097 | +0.638992404 |

Interim 2026-05-20 decision was to ship grouped-KL over
`NVFP4,MXFP8_E4M3,BF16`, with `FP8_E4M3` remaining opt-in only. The shipped
5.5 comparison below supersedes that decision for replacing the published 5.5
artifact.

## Shipped 5.5 Comparison (2026-05-21)

Comparison work dir:
`/home/rob/dq-runs/qwen36-27b-grouped-5p5-vs-shipped-20260521T134058Z`

Shipped 5.5 artifact:
`/home/rob/.cache/huggingface/rdtand-Qwen3.6-27B-PrismaQuant-5.5bit-vllm`

Grouped 5.5 was exported from the measured production cache:

```bash
python3 -m prismaquant.export_native_compressed \
  --model /home/rob/.cache/huggingface/qwen36-27b-bf16 \
  --layer-config "$OUT/layer_config_grouped_intree_5.5.json" \
  --output "$OUT/exported" \
  --device cuda \
  --production-weight-cache /home/rob/dq-runs/qwen36-27b-grouped-validation-20260519T192416-27b/artifacts/production_weight_cache.pkl \
  --production-cache-dir-override /home/rob/dq-runs/qwen36-27b-grouped-validation-20260519T192416-27b/cache_dir \
  --production-cache-lru-gb 8 \
  --production-cache-prefetch-workers 4
```

Export log:
`logs/export_grouped_5p5.log`

Export result:

- 614 assignment entries: `NVFP4=331`, `BF16=283`
- 5 safetensors shards, 22 GiB
- Manifest histogram:
  `linear/NVFP4_PRODUCTION_CACHE=331`, `linear/BF16=165`,
  `layer_passthrough/BF16=352`, `head_passthrough/BF16=3`,
  `mtp_linear/BF16=8`, `mtp_passthrough/BF16=7`
- vLLM loaded the grouped export and selected
  `FlashInferCutlassNvFp4LinearKernel`.

Exact shipped-contract vLLM full-vocab next-token KL used the prior BF16
teacher payload:
`/home/rob/dq-runs/qwen3p6-27b-rerun/kl_shipped_5p5_20260503T012223Z/teacher_logprobs.pt`

Both rows use 8 WikiText train windows, sequence length 512, seed 42, exact
starts `[466956,104902,1153556,1027150,936213,585264,429895,2287433]`.

| artifact | KL mean | KL min | KL max | elapsed |
|---|---:|---:|---:|---:|
| shipped 5.5 | 0.047497515 | 0.001097764 | 0.148100182 | 136.58s |
| grouped 5.5 export | 0.087037444 | 0.000038506 | 0.170341790 | 128.43s |

Grouped regresses exact KL by `+0.039539929`, or `1.83x` shipped.

Direct vLLM WikiText test PPL was measured from the same pre-tokenized
8192-token slice for both artifacts. The shipped artifact was run with
`--language-model-only` and a combined FlashInfer AOT cache shim because its
MXFP8 path otherwise tried to write into a root-owned FlashInfer JIT cache.

| artifact | scored tokens | mean NLL | PPL | elapsed |
|---|---:|---:|---:|---:|
| shipped 5.5 | 8176 | 2.251110925 | 9.498281855 | 47.41s |
| grouped 5.5 export | 8176 | 2.276283775 | 9.740415492 | 56.42s |

Grouped regresses direct vLLM PPL by `+0.242133637` PPL, or `+2.55%`, and
mean NLL by `+0.025172850`.

Additional local in-tree KL screening, using `validate_assignments_kl` on the
same 8x512 train calibration contract but not the exact full-vocab vLLM KL
contract, also ranked the local baseline over grouped:

| assignment | bpp | last-token KL | format counts |
|---|---:|---:|---|
| local baseline 5.5 | 5.499084939 | 0.027974171 | `BF16=157`, `NVFP4=347` |
| grouped 5.5 | 5.496608892 | 0.074985422 | `BF16=283`, `NVFP4=331` |

Run outputs:

- Grouped exact KL:
  `vllm_kl_grouped_5p5_vs_bf16.json`,
  `logs/vllm_kl_grouped_5p5_retry_cache.log`
- Shipped exact KL:
  `/home/rob/dq-runs/qwen3p6-27b-rerun/kl_shipped_5p5_20260503T012223Z/shipped_5p5_kl.json`
- Grouped vLLM PPL:
  `vllm_ppl_grouped_5p5_wikitext_test_8192_from_ids.json`,
  `logs/vllm_ppl_grouped_5p5_from_ids.log`
- Shipped vLLM PPL:
  `vllm_ppl_shipped_5p5_wikitext_test_8192_from_ids.json`,
  `logs/vllm_ppl_shipped_5p5_from_ids_retry_combined_aot_lm_only.log`
- In-tree assignment KL:
  `kl_validate_assignments_grouped_5p5.json`,
  `kl_validate_assignments_baseline_5p5.json`

Decision after shipped comparison: do not ship grouped-KL 5.5 as a replacement
for the published 5.5 artifact. The grouped objective fixed a local allocator
non-monotonicity and beat the local allocator in the prior HF PPL harness, but
it loses to shipped 5.5 on exact vLLM KL and on a direct vLLM WikiText PPL
slice. `FP8_E4M3` remains no-ship for the default menu.

## Verification

```bash
python3 -m py_compile tools/measure_vllm_ppl_from_ids.py
python3 -m py_compile prismaquant/grouped_kl_cost.py
bash -n prismaquant/run-pipeline.sh
python3 -m pytest -q \
  tests/test_grouped_kl_cost.py \
  tests/test_serving_profiles.py \
  tests/test_allocator_shape_mask.py
git diff --check
```

Result: `23 passed, 15 warnings`; compile, shell syntax, and whitespace checks
passed.
