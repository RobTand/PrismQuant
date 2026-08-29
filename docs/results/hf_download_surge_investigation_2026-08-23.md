# HF download surge investigation: PrismaSCOUT 27B, August 2026

Investigation date: 2026-08-23.
Question: why did `rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm`
gain roughly +97k counted downloads between Aug 14 and Aug 22, and can the
pattern be replicated on newer releases?

## Measured facts

From `prismaquant-site/site/stats_history.json` (nightly Hub API snapshots)
and the live Hub API on 2026-08-23:

| Date | SCOUT all-time | Family total | Notes |
|---|---|---|---|
| Aug 14 | 396,659 | 984,651 | calm ~4.8k/day family-wide |
| Aug 16 | 397,591 | — | |
| Aug 20 | 414,243 | 1,013,344 | surge underway (+16.7k in 4 days) |
| Aug 22 | 494,526 | 1,095,939 | +80.3k in two days |

The live API reports 114,791 downloads for the last 30 days. The repo was
last modified 2026-05-04, so none of this traffic is driven by a new upload.
Every other artifact in the `rdtand` family grew at its normal rate during
the same window; the acceleration concentrates on this one repo.

This matches the standing audit note in `prismaquant-site/README.md`
(2026-08-22): the site sums what the Hub API publishes, no cron or container
on sparky, the DL380, or the web VM touches Hugging Face, and HF counts every
HTTP request to a repo's query files — including `HEAD` requests — as a
download.

## What PrismaSCOUT is, and where it is embedded

PrismaSCOUT is the retired May 2026 allocator generation whose flagship
artifact is a mixed NVFP4/BF16 `compressed-tensors` export of
Qwen/Qwen3.6-27B for vanilla vLLM on Blackwell, Apache-2.0, 20.17 GB,
published 2026-05-04 with MTP heads included. AURA superseded it as the
allocator cost model on 2026-06-08; AQUA added activation-aware pricing and
shipped with Qwen3.8 releases in August. The artifact outlived the allocator.

Independent ecosystem embedding found via GitHub code search (119 matches),
the NVIDIA DGX Spark forum, and web search:

| Embedding | What it does | Dated evidence |
|---|---|---|
| NVIDIA DGX Spark forum threads | "Introducing PrismaQuant" (~7.6k views) and "Introducing PrismaScout" (~10k views, 138 posts); recipes posted inline by community members | launch thread May 4; still linked from active threads in August |
| bjk110/spark_vllm_docker (56 stars) | one-command presets `qwen3.6-27b-prismascout-nvfp4-tp2*.env` for eight stack variants | presets present since spring; repo pushed Aug 15 |
| technigmaai/dgx-spark (19 stars) | copy-paste recipe directories for PrismaQuant models | created May 14 |
| mcampa/sparkrun-recipes | three SCOUT serving recipes incl. MTP tuning | added May 27–28 |
| Spark Arena (sparkarena HF org + forum leaderboard) | community benchmark leaderboard born on the GB10 forums; multiple PrismaQuant submissions | submissions since May; forum post Jul 22 |
| totally-tim/tokenfeel | "Spark Arena importer" ingests 88 benchmark rows that include a SCOUT row (`data/models/qwen3.6-27b-prismascout-blackwell.json`) | merged Aug 12 |
| Neroued/ninfer (961 stars, pushed Aug 23) | single-GPU inference engine using the SCOUT-derived NVFP4 artifact as reference/model card; perf results refreshed Aug 10–12 | commits Jul 30 – Aug 12 |
| stateless/swamp-extensions | LLM catalog data listing SCOUT among Spark-served models | published Jul 7 |
| janhq/model-catalog | Jan desktop app ships a model catalog referencing SCOUT | catalog auto-updates ended Jul 2 in search index; current catalog still lists it |
| smol-ai ainews digest (Jul 21 issue) | quotes an r/LocalLLaMA commenter calling PrismaSCOUT a "preferred daily-driver quant" versus 5–6 bpw alternatives | Jul 21 |
| heavy-metal-cloud YouTube "bare metal AI" ep. 03 | SCOUT listed in published DGX benchmarks of the video | repo pushed Aug 23 |
| shrinedogg/biggs.dog + federationspace/biggs-sz | GitOps homelab repos deploying vLLM with the SCOUT repo pinned in Kubernetes manifests | SCOUT production config Jul 11; switched to unsloth Aug 15 then context suggests revert churn |
| cyburn/Qwopus3.6-35B-A3B derivative | third-party requant labeled "PrismaSCOUT" (554 downloads) | created May 7 |

Sentiment found along the way (all positive-to-neutral, none hostile):
forum users report 360 t/s aggregate at 16 concurrent threads on PDF
extraction, MTP acceptance rates 70–80% on code workflows, "35b too
unreliable, 122b too slow, now competitive with 122b"; complaints are about
Qwen base-model loops, vision TP=2 failures, and eval-replication friction —
not about the quantization itself. One user asked for a comparison against
`nvidia/Qwen3.6-27B-NVFP4`, unanswered after 17 days.

Reddit sentiment could not be read directly (403 walls on both HTML and
JSON endpoints, archive APIs rate-limited), but a r/LocalLLaMA comment
quoting SCOUT favorably was captured by the smol-ai AI-news digest on
Jul 21, which confirms organic Reddit mention exists independent of Rob's
own posts.

## Why the surge most plausibly happened

No single public event dated Aug 19–22 explains the spike: no HN story ever
mentions prismaquant or prismascout (Algolia search returns zero), no Wayback
Machine captures exist in the window, no Reddit front-page post was found,
no trending placement appears on the Hub today, no new base-model derivatives
reference SCOUT, and nothing on Rob's infrastructure pings HF.

The strongest remaining hypothesis is automated HEAD/GET pinging of the
single well-known repo id, triggered by something that started mid-August.
SCOUT's repo id appears in at least four shipping automation surfaces:
the bjk110 one-command presets, the Jan desktop app catalog, NInfer docs,
and tokenfeel's dataset. A health-check loop, a broken update check that
retries without backoff, a monitoring dashboard, or an LLM-agent workflow
that resolves the repo id nightly would each produce exactly this signature:
one repo, HEAD-heavy, tens of thousands per day, no matching file-download
bandwidth, starting abruptly and persisting daily. The alternative —
several thousand humans per day discovering a three-month-old 20 GB file —
would leave traces (forum posts, discussions, likes, derivatives) that are
absent.

A weaker but real human tailwind exists underneath: NInfer refreshed its
SCOUT-based performance docs Aug 10–12, tokenfeel merged Spark Arena import
data Aug 12, the "Lot of PrismaQuant news" thread ran Aug 17–22 keeping the
family visible, and the DGX Spark install base keeps growing. That mix can
move normal growth, not +80k in two days on one repo while siblings stay flat.

## How to verify definitively

1. Ask Hugging Face support whether request logs for the repo can be shared
   under the privacy policy, or what Publisher Analytics would show for
   Aug 19–22 (user agents, HEAD vs GET, IP classes). Publisher Analytics is
   a paid plan feature; a direct support query costs nothing to send.
2. Watch `downloadsAllTime` vs 30-day deltas and the likes/discussions
   ratio. If the counter keeps climbing +40k/day while discussions stay at
   zero per week, it is automation.
3. Check whether any tool the community runs (bjk110 preset scripts, llama
   swap dashboards, uptime monitors) gained a scheduled job around Aug 19.
   The bjk110 repo pushed DSv4 runtime changes Aug 11–15; a regression that
   makes every stack-health check hit the HF API would be invisible to its
   author.

## Replicating genuine SCOUT success on newer releases

The durable lessons from how SCOUT grew are independent of the bot spike:

1. Ship for the hardware community that actually talks. The DGX Spark / GB10
   forum is a dense, benchmark-hungry, recipe-sharing audience. SCOUT won
   because 20.17 GB fits a Spark with room for KV cache and MTP, and because
   Rob answered every technical question within hours, including uncomfy
   ones (infinite loops, TP=2 vision failures).
2. Publish turnkey artifacts others can embed: serve-ready compressed-tensors
   plus MTP tensors meant third parties (bjk110 presets, sparkrun recipes,
   Spark Arena, NInfer) adopted the exact repo id into their tooling. Every
   embedding is permanent distribution. Target the same for AQUA/AURA
   artifacts: get them into bjk110 presets and the Jan catalog the week they
   ship.
3. Make claims falsifiable and answer replication attempts. The honest
   limitations section on prismaquant.org and the public KL methodology
   generated trust; the one open eval-comparison discussion (#8) should be
   answered — it is currently the top open item a skeptical newcomer sees.
4. Keep MTP/speculative-decode support in the artifact. Multiple adopters
   cited acceptance rates and DFlash-vs-MTP throughput as reasons to pick
   these exports over NVIDIA's own NVFP4 builds.
5. Seed the comparison table proactively. The recurring ask is "same evals
   vs nvidia/Qwen3.6-27B-NVFP4 and unsloth quants." Publishing that table
   per release removes the main adoption objection before it is asked.
6. Treat the download counter as telemetry, not victory. The site already
   labels it request volume; keep that label until Publisher Analytics or
   HF support confirms otherwise.

## Provenance

- Live Hub API pulls, 2026-08-23: downloads=114791/30d, lastModified May 4.
- `prismaquant-site/site/stats_history.json`, nightly snapshots Aug 14–22.
- GitHub code search API for "PrismaSCOUT", 119 matches across 30+ repos.
- Discourse JSON of both NVIDIA forum threads (May 4 and Aug 17).
- HF community tab of the SCOUT repo (8 discussions read).
- Wayback CDX queries run 2026-08-23 (six captures total, none in window).
- HN Algolia API queries run 2026-08-23 (zero hits for both names).
