# #87: same-session BF16 boundary-instrument A/B

Measured on 2026-09-04, Sparky/GB10. **The 64-token ceiling rejected every
sample from this BF16 control under both endpoints.** The chat control reached
an uncensored schedule at 2,048 tokens, with zero scored defects; a same-session
BF16 A/A repeat at that cap also had zero defects. This is instrument evidence,
not quantized-artifact discrimination and not a shipping-policy promotion.

## Population and observed impact

Qwen3-0.6B BF16; the frozen five prompts in
`experiments/pq87_boundary_contract.json`, seeds 0–5, temperature 1, top-p 1:
30 generations per row, split into numeric 18, terse QA 6, short recall 6.
Every completed request/response and step is retained and re-scored.

| Endpoint / arm | Completion cap | Any scored defect | Cap hit | Missing closing tag |
| --- | ---: | ---: | ---: | ---: |
| Raw completions A | 64 | 30/30 | 30/30 | 30/30 |
| Chat completions B | 64 | 30/30 | 30/30 | 30/30 |
| Chat control | 128 | 25/30 | 25/30 | 24/30 |
| Chat control | 256 | 20/30 | 20/30 | 19/30 |
| Chat control | 512 | 10/30 | 10/30 | 4/30 |
| Chat control | 1,024 | 2/30 | 2/30 | 0/30 |
| Chat control | 2,048 | 0/30 | 0/30 | 0/30 |
| Same-session BF16 A/A repeat | 2,048 | 0/30 | 0/30 | 0/30 |

There was no think-stutter count in any row. Categories overlap; a cap-hit
sample can also lack its closing tag. At a matched cap of 64, switching the
endpoint changed the observed rejection count by **0/30**. Increasing the chat
budget to the measured uncensored point changed it by **−30/30**. This does not
show the endpoints are equivalent: their low-budget outcomes were both censored.

The live context was explicitly bounded to 4,096 tokens. Chat input counts
were `[14, 10, 16, 15, 14]`, with exact token IDs in both receipts, hence the
shared completion bound was 4,080. Doubling stopped at its measured fixed point
2,048, before that bound or the eleven-step backstop. This cap was selected on
these same prompts; it is not an unbiased held-out artifact-selection result
or a new universal token budget. All three strata had control-minus-repeat
defect delta zero. No quantized candidate, other model population, old DSV4
artifact reproduction, PPL/KL comparison, or acceptable regression threshold
was measured here.

## Identity, resources and receipts

- Driver commit: `4c43d9c6`; immutable PrismaBuild checkout:
  `feb67efa017c21140931cc2d1b20fa200cdcffb6`, based on merged main `6252ef70`.
- Image: `eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`;
  observed vLLM `0.28.1rc1.dev397+gfd4a15126.d20260904`.
- One eager server, one sequence, 512 batched tokens, explicit 1 GiB KV;
  PB `gpu=2,cpu=2,mem_gb=32`, Docker 28 GiB memory/swap ceiling. No percentage
  KV allocation. Physical work lasted 617.24 s (PB wall 622 s), below 1,200 s.
- Source `/home/rob/models/Qwen3-0.6B` was copied and hashed **before startup**;
  frozen weights: 1,503,300,328 bytes,
  SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
  Both clients matched the pre-launch compound artifact ID
  `24e9b66a00b4495068efc88c4450677912057b33b27caae40d6eaf909dca06bc`.
  Model/source were mounted read-only; pre/post exact content and weight-stat
  checks passed in both arms.
- Same serve-session ID:
  `5f105c2df75ca5d8d7523d81c423197116c4e9eb6fb86b74d4b788003c72a294`;
  same stack fingerprint:
  `1def1b6b551a87cab5b9c2d06231401d307427f2833aa81318ce4e8455c555c5`.
  Pre/post process, endpoint, token-ID and source-closure bindings replayed.
- Durable local evidence:
  `/home/rob/tmp/pq87-bf16-ab-20260904-r1/`: `control.json`, `candidate.json`,
  `prelaunch.json`, `*-http.jsonl`, `*-steps.jsonl`, `server.log`,
  `process-io.jsonl`, `telemetry.jsonl`, `report.json`,
  `telemetry-summary.json`, `cleanup-recheck.json`.
- Physical action
  `a184593f7e7bcdc7685e33134811e1d1c9534c6d58e00fbaa9fea96589f5d341`,
  CAS receipt `53e55b3b5be80bd2d7790b0f9f013b4ba83217e61e7e670918a64cc32080cf01`.
- Offline replay/independent cleanup action
  `8f2d9e2919f2a7c459bac574fcdf857a6aeb3f77d06ea956e0085b68a4bab25f`,
  CAS receipt `b3e416bdc669b327616e902a3953bb57b18169c7835922a1714a1beace19a67a`.

Telemetry has 62 successful Netdata snapshots **on each box**, with no HTTP
errors, plus process I/O/status observations and direct NVIDIA query output.
Netdata power samples were 4–42 W on Sparky (mean 33.84 W) and 16–60 W on
Sparklina (mean 54.76 W, running its separate MoE encode). These include startup
and shutdown; they are not steady-state throughput or work-per-joule claims.
The direct NVIDIA query reports memory as `[N/A]`; the initial report parser
therefore left its whole numeric tuple null. The preserved Netdata power
charts, with units `Watts`, were independently summarized in PB action
`289ba7235d99ccc92c550c10363011de6b81308b46c3a5cc7f14636d154a3d0e`.
No GPU-utilization interpretation or speed claim is made.

## Cleanup finding, tests and remaining decision

Both client return codes were zero and the server shut down gracefully at
20:11:59 UTC. Docker removed the exact container. Its lowercase
`error: no such object` diagnostic exposed a conservative case-sensitive
cleanup parser: the original outer `campaign.json` remains **inconclusive**
and is not rewritten. Independent PB inspection at 20:13:54 confirmed the
same uniquely named container absent (`safe=true`); that supplemental receipt
resolved cleanup before Sparky was handed back to the MoE owner. No GPU rerun
was needed. The diagnostic is fixed separately in `b6945ce4`.

Controller CPU evidence (all explicit selected modules collected, zero skips;
no master suite): initial API-absence red `c0d6a2c8a677` had three
`ModuleNotFoundError` failures; empty-health-body red `d6132e3c8693` had
`JSONDecodeError`, three controls passing; cleanup-helper red `a97c07cb71a4`
had two absent-API failures, four controls passing. The physical action's
preflight had 25 passing controller/docs tests inside its GPU-visible GB10
reservation; those files do not cover the CUDA-gated surface. The observed lowercase
cleanup regression was red in `1792656bc113`: `assert False is True`, one
failure/six deselected. Final CPU-only action
`554cc56a0e4fe39e605251a7d01de1059fbd4f17560c42dd7f445a6202e985d7`
had **26 passed, 0 failed, 0 skipped**, plus successful compilation, using
one CPU/3 GiB and `CUDA_VISIBLE_DEVICES=''`. The frozen-test snapshots omit
only the three unused absolute calibration symlinks, restored immediately
after sealing; physical prompts come from the committed contract above.

Professional reading: the historical 64/zero screen confounds an unfinished
BF16 response with an artifact defect on this population. Endpoint repair
alone does not remove that confound. Keep the new instrument opt-in while Rob
chooses the versioned control-relative promotion rule and the representative
candidate/second-population evidence it should require. #87 remains open with
`needs-decision`; these measurements do not authorize changing its defaults.
