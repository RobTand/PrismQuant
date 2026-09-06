# #208 Qwen3-0.6B paired native validation — 2026-09-05

The complete campaign **refused native NVFP4 on the disjoint held-out
population**: four of 30 previously clean BF16 pairs became defective at the
BF16-derived cap. The screen population was accepted and both small PPL
screens passed. This is a measured policy refusal, retained without rerunning;
it does not promote a shipping default. Earlier infrastructure and preflight
failures are preserved below, separately from this complete observation.

## Complete paired result

PrismaBuild action
`0140467ada8390156fed4a27fc63d2d5796b52e88574d60a9d8bd95d0a90af32`
completed in 1119.9 seconds with worker rc0, `campaign.status: completed`,
all four policy arms and both PPL observations present. Exact owned-container
cleanup was safe and the subsequent action-owner Docker query was empty.

| Population | Matched pairs | BF16-derived cap | BF16 defects | Native defects | Newly broken pairs | New defect kind pairs | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Screen | 30 | 2048 | 0 | 0 | 0 | 0 | accepted |
| Held-out | 30 | 1024 | 0 | 4 | 4 | 0 | refused |

The held-out violations are `13×7` at seed 1 and `121÷11` at seeds 1, 3 and
4. All four responses hit the fixed token cap; three also lacked a closing
think tag. The `121÷11`, seed 1 response had only cap truncation. The policy
records these as newly broken previously clean pairs; its separate
`new_defect_kind_pairs` counter is zero. Neither prompts nor caps were
changed after seeing this result.

| Existing small PPL screen | BF16 | Native NVFP4 |
|---|---:|---:|
| PPL | 11.164634991734285 | 11.865439750163203 |
| Mean NLL/token | 2.4127511925709118 | 2.4736299520043907 |
| p99 NLL/token | 3.316376398656649 | 3.4148417506381565 |
| Max NLL/token | 3.3357807611838717 | 3.450937754757303 |
| Complete finite prompts / scored tokens | 12 / 583 | 12 / 583 |
| Existing threshold check | passed | passed |

Both PPL observations explicitly report non-speculative decoding. Passing this
small numerical screen did not predict acceptance on held-out boundary
behavior. These are native-serving treatment observations on one small model,
not isolated quantization-only effects, gold WikiText PPL, served KL,
downstream/tool quality, a speed comparison or work-per-joule improvement.

The final declaration was committed before the first candidate policy
observation. The raw full fingerprints remain different and preserved:
BF16 `1609474a8949c0b1b340c8f0aefb5aae4ef24821bfa260b7bf2670709229b220`;
native `03ea2d38644b95fbef30ab6eadb4c919ce7f9c5c6f2f22739e8d52843efdb8db`.
BF16 resident extensions were exactly `sampling.so`; native additionally
loaded `trtllm_utils.so`. Every other existing stack payload coordinate
matched canonically. The complete raw pre/post stack and content manifests,
policy v2 receipts, pair outcomes and decisions remain available for replay.

Final source parent `4888f033133bcb3f22416af37ef5400d8ccb29eb` was sealed as
snapshot `ec0805c93883c060708cc53485bc3f50cb72ae16`, bundle
`4ebe8cc4b4e87475c0f62ec73d343293c33446669ae95c675ce270373191426f`.
The three exact unused absolute calibration symlinks were excluded only from
snapshot sealing and immediately restored. The submitted command is retained
in `pq208-submit-05.json`. Deployed runtime `d8f82f3a48f6` offered one physical
GPU; `--gpu --exclusive` reserved it with two CPUs/32 GiB, assigning preferred
cores 5 and 6. All other numerical/resource settings remain those below.

Campaign SHA-256:
`45f8b530cc80de1db16264c6337a7adc1369b0d38c14ce6e3867929ce4c369ac`.
CAS receipt:
`6e77638429df07225ff7b27273cb581ded3fe10ad1b68bc7ee4d330310e53fa7`;
rehashed 23,350-byte result:
`39ab8f0aac858187a618ba2b877b50a1c106ba3a8dbad54f351a1041feb82c4e`.
All 49 top-level evidence files were byte/hash verified in the shared
`run208-04/` archive, including complete HTTP journals, step records, server
logs, process I/O and both-host Netdata. Frozen models remain archived;
transient socket and compiler-cache directories are excluded and retained
locally. Terminal copies redact broker capability values; the roster binds both the
archived originals and the published copies. Numerical receipts are unchanged.
The checked-in evidence subset and hash roster are under
[`pq208_paired_quantized_validation_2026-09-05/`](pq208_paired_quantized_validation_2026-09-05/).

The full campaign retained 111 usable Netdata samples from each host with no
missing/error samples, plus 111 GPU power readings. The observed power range
was 4.71–40.03 W (31.15 W mean across startup, work and shutdown); this is
host context for the fixed single-sequence workload, not a saturation or
energy-efficiency claim.

Independent offline replay through PB action
`0e88c1a7c7bbd30404f063e33a127e69b4f8a10f7120a47e6c7590038d28ab21`
on Sparklina (one CPU/4 GiB, disabled CUDA, native thread limits one) returned
rc0 and exactly reproduced both stored decisions from the archived controls
and candidates. It also required the complete finite PPL roster for each
arm. Its deployed runtime was `5afd512ef295`; receipt
`c946f1cc43cdd9e1ca1ab01d2e89a8636659eef6911262a170c9845c1b8cb932`
binds the rehashed 3,309-byte result
`42a9b344f22f76b805a8294a67ecc7f551ee407b9676c700eea938ef8c70c62f`.
The exact replay command is retained in `pq208-final-replay-submit.json`.

## Earlier model observations, incomplete candidate comparison

Action `d810b3abb35098dd85f3fc0a2da5cbf4d7b1184ff96e17f759b6f9ce5350c44c`
ran for 828.6 seconds. The screen reached its uncensored BF16 cap at 2048
tokens, with zero defective pairs; cap 64/128/256/512/1024 had
30/25/20/10/2 defective pairs. The disjoint held-out control reached its
uncensored cap at 1024 tokens with zero defective pairs; prior caps
64/128/256/512 had 30/30/26/12 defects. Each step contains all 30 pairs.
BF16 PPL passed at 11.164634991734285, mean NLL 2.4127511925709118,
p99 NLL 3.316376398656649 and max NLL 3.3357807611838717, with finite
per-prompt metrics for all 12 prompts and 583 scored tokens.

The native candidate loaded using
`FlashInferCuteDslNvFp4W4A16LinearKernel` and successfully served health,
raw/chat warmups and tokenization. Before candidate matched-pair generation,
the client raised `ValueError: paired serve_fingerprint differs` in
`boundary_control.py::_paired`. Candidate held-out and PPL were not reached.
This is not the paired policy's measured `refused` verdict. The overall
campaign is `inconclusive`; `cleanup.safe` is true, the exact owned
container was removed and the action-owner Docker query was empty.

Campaign SHA-256:
`7d7210a29bf2d8ae87fbd758c2079d8e81e68586d6f8b1f6a91373a14058e198`.
CAS receipt:
`34793950cf58a714c8a106bc4c85b058bff851f2e6da97a088e5f4603e444bc4`;
actual 21,149-byte result:
`62fde09d65217f8afe5a917a2247ac81c132e1e1f50c9b73e30dd798fa005e64`.
Source parent `a26cfcce06cb66472dd17fc7464c19a95dfcc43d` was sealed as
snapshot `3eb1c6efe66b01a25694c8a1d0adf8fc66091a28`, bundle SHA-256
`77469547b3e061747ec9e9932161aab34f15b0267cf64d866c48d7bcd430ebc0`.
The snapshot excluded only three unused absolute calibration symlinks,
immediately restored after sealing with exact targets recorded in submit
evidence. All 34 top-level campaign evidence files are byte-verified in
`/mnt/shared/tessera-runs/measurement-208-183-2026-09-05/run208-03/`;
compiler caches and transient socket directories remain local.

## Frozen scope and environment

The committed `experiments/pq87_validation_manifest.json` supplies the same
five screen prompts, five disjoint held-out prompts, seeds 0–5, temperature
1.0, initial cap 64, and at most 11 cap steps. Each population has 30 matched
pairs. The native BF16 server derives its own uncensored cap for each
population; native compressed-tensors NVFP4 uses those exact schedules.
The existing full-roster small PPL screen runs on both servers. No numerical,
prompt, seed, cap, PPL, or decision-policy setting was changed during launch
correction. A measured candidate refusal is retained without rerunning.

Sparky: GB10, 121.63 GiB unified device memory, driver 595.84, Linux aarch64.
The immutable image is
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`;
its server identifies as vLLM `0.28.1rc1.dev397+gfd4a15126.d20260904`.
The earlier GPU actions reserved the entire Sparky physical GPU (then two
deployed GPU slots), two CPUs and 32 GiB through PrismaBuild. The final
action used the current one-device offer, still exclusively. Containers retain the PB
Docker shim, assigned CPU affinity and scope parent, with 28 GiB memory plus
equal swap limit, two CPUs, one sequence, 4096-token context, 1 GiB KV and
eager execution. Both Netdata endpoints and GPU power are journalled; no
throughput, saturation or work-per-joule claim is made.

The frozen control artifact ID is
`24e9b66a00b4495068efc88c4450677912057b33b27caae40d6eaf909dca06bc`
(1,503,300,328 safetensors bytes); candidate artifact ID is
`0c87a90020e39d62fcf1fd560c22cc9d6ccf01d190ac9248a07825bd54f29a80`
(870,290,032 bytes). Source paths remain those in the manifest. The candidate
has no ship-certified status.

## Infrastructure observations before any model comparison

1. Action `dd4b28030d888df30fb9aed5440b448c7c34b2d8bb045d45702eb71f43f370ac`
   ended after 16.3 seconds. Docker could not traverse the task-created NFS
   parent directory, mode 0700, and returned `permission denied` while
   creating the output bind mount. No server started. The controller wrote
   `inconclusive`, verified its exact owned container and removed it safely.
   Campaign SHA-256:
   `b452539d916c610688af49ffbc7f623cf59a10078c62afe1d8268685e38451ec`.
   CAS receipt:
   `fa6e3049f834726ba2054a89b43add9e0e088198bb1060de8031dfe30a96328d`;
   actual 20,949-byte result blob:
   `1c9006723c336290f1d3f5af03997e7a2fbdd6df43ba5afdf389e3fa0df2abaf`.
   No shared permissions were changed; subsequent Docker outputs use a
   task-owned local ext4 directory, archived after cleanup.
2. A corrected output basename `pq208-run-02`, a sibling of checkout
   `pq208`, was rejected before submission because
   `pbrun.py::require_relocatable_checkout` used substring membership
   (`source in str(token)`, published line 1138), without directory
   boundaries. Renaming the sibling `run208-02` passed this guard. The
   incidental finding is tracked by PrismaBuild #187. This was not a worker
   attempt and produced no model observation.
3. Action `9c4547fbf25202ac2019f6f3def380ed6a63f7e4e84156933e156544838dd616`
   successfully launched the container, but the BF16 server refused during
   startup: free device memory was 111.87/121.63 GiB, below the inherited
   0.92 fraction's 111.9 GiB request. Explicit KV bytes did not bypass that
   check. No prompt or candidate ran. After 39.4 seconds the campaign was
   `inconclusive`, with safe exact-container cleanup and an empty
   action-owner Docker query. Campaign SHA-256:
   `d550bb6f795a34c054c147123d8200a000295791119ccd8ba39bfbc78612f755`.
   CAS receipt:
   `a4d02bb94c0a7246feda4d47d87579020ddcc0d101e39379889dc6acf94f90b3`;
   actual 19,940-byte result blob:
   `8521717f78c990a2106f3e26277c83345333cffc184438e464535f04b0eade25`.

Both executed attempts used main `cda074a8c063a9e9ebd59bb549271c6160a91796`.
Their wrapper exit status 0 is not a successful model observation. Original
campaigns, exact submit commands, terminal records, CAS receipts and logs are
retained under
`/mnt/shared/tessera-runs/measurement-208-183-2026-09-05/`.
Attempt 2's 13 top-level evidence files were byte-checked against its local
output. Unreadable root-owned ephemeral vLLM cache state and dead Unix-domain
sockets remain local and are excluded from the archival evidence roster.

## Scoped resource correction and verification

Commit `a26cfcce` adds `--gpu-memory-utilization 0.2` only to the paired
campaign's existing server-argument builder, identically for both arms.
Approximately 24.3 GiB on this GB10 fits the 28 GiB container; fixed 1 GiB KV
bytes remain unchanged. This corrects the launch resource contract, without
changing production shipping defaults or the historical BF16 instrument.

An initial CPU-only test action
`5f20e1b9601ba45b00ed22f4bbeb5382e16e6dbe9a336375768897d654b7d6ed`
on dl380g10 used `/home/rob/venvs/pb-cpu/bin/python` and had two setup errors
because `compressed_tensors` was missing; 27 tests were deselected. This
provides no regression evidence and caused no environment mutation.

The existing `/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python` supplied
the qualified CPU environment. Action
`e3bc80f1c78fc5680e62f173724683830c5f714fafa83da9101a1cbb4e6ab130`
on Sparky reproduced **2 failed, 27 deselected**, both naming the absent
startup memory flag, before the correction. Afterward action
`3347f6736ee3d2e31ae599b63ba57c3de1802773a7e345943273dda9674f2139`
on Sparklina observed **135 passed, 0 skipped**, with two xdist workers,
two CPUs/8 GiB and native thread limits of one. It collected all eight
established files: boundary policy/control, measurement identity, physical
instrument, paired controller, shipping boundary behavior, docs staleness
and architecture. All CPU actions had `CUDA_VISIBLE_DEVICES=''`.
Green CAS receipt:
`e4ee23974bbfb59d1aadc867eea6e409b793b8cc6dd809a22175aaaf0e874bf1`;
actual 732-byte output:
`e33eb1c5ab0625d1e98e84a1b973edb60257913f2524cc03f20e4690af747c1e`.

Action `4c49c61c29811803864b59e7933be112a424bd959f80576d2e3f4898c3d5b2f5`
compiled both touched Python files successfully on Sparky, one CPU/1 GiB.
Receipt:
`fec1734cebf38ed2e36c1a4a00df42b2d11b1f4f31c0987ffafd854b80196365`.
The exact red/green/compile commands and terminal records are retained beside
the campaign evidence. These tests verify the resource argument and existing
policy/safety contracts; they do not substitute for served validation.


## Diagnostic and explicit paired treatment

Bounded action `92740ca5b2d62f81bdae9de5c6e239f863e92863f83e661e99038f6f7b3cd71c`
used source `3a6719f35f693d8a02903545b3e447e8a81c5155` and completed after
124.7 seconds with `preflight_completed`, safe cleanup and an empty exact
owner container query. It ran server startup and warmups, without policy
populations or PPL. The full observed stack payloads differed only in
`resident_extensions`: BF16 had `sampling.so`; native NVFP4 had
`sampling.so` and `trtllm_utils.so`. Image, GPU, driver, packages, compilation
provenance, normalized argv, environment, listener and readability agreed.
Raw full fingerprints were respectively
`1609474a8949c0b1b340c8f0aefb5aae4ef24821bfa260b7bf2670709229b220` and
`03ea2d38644b95fbef30ab6eadb4c919ce7f9c5c6f2f22739e8d52843efdb8db`.
The raw manifests and 20 top-level evidence files were byte/hash verified in
the shared `stack208-01/` archive. CAS receipt
`1f8ea08a97a63611af13a4942626a484c32de6eca1c2824fa908458ae54d37ce`
binds the 20,613-byte result
`9ef5d10ff0f411957536d13259b4f9efafd3feb8fca4f37ed898ffbc5cdd95d7`.

Commit `4888f033133bcb3f22416af37ef5400d8ccb29eb` opts this experiment into
validation manifest v2 and a versioned paired-stack declaration, binding the
immutable image, exact content artifact IDs and exact role extension sets.
Boundary-control v2 receipts preserve and recompute the raw full fingerprint,
then compare canonical JSON digests of every other stack payload coordinate.
The declaration must be identical in both receipts, which retain same-campaign
identity, tokenization/producer/host checks and the exact control receipt hash.
Per-arm pre/post full residency checks remain strict; legacy receipts keep
full fingerprint equality and cannot silently replay new-schema evidence.
No numeric, prompt, seed, cap, PPL or candidate-policy contract changed.

PB red action `1b65696d241e88828c9153f995251540003a825ef976aa7d0919e1fd5d29c0c7`
observed 6 failed and 17 passed before implementation. Review identified that
ordinary Python equality would admit rehashed `gpu_count: true` versus `1`;
red `4b7629a98cb420831be4e00add176218a4948b99ceed9119f94b5dcff8faef9f`
confirmed 1 failed / 24 deselected before canonical digest equality fixed it.
Green `b7ae4af16c86c2c9bf403008e6c8b2b299f845b153968d22a0a4f3020aad063f`
ran all nine targeted files on Sparklina, two CPUs/8 GiB, pytest `-n 2`,
OMP/MKL/OpenBLAS limits one and disabled CUDA visibility: **161 passed,
0 skipped**, rc0. Receipt
`5d2fa2ee4c956bbd0b12a35a17d1bc7cf12bd2cf42f04c3b8ac452198833cb29`
binds the rehashed 812-byte output
`900b9439b7c7e462cf158ef7f4644c3148580ffb76b6f5fa335cc464782de436`.
Compile action `e33b8cd1d7ec6d474eb43b2297573b3de203b70f8735171e0baa76dbf9283639`
returned rc0, receipt
`b2deb0935fa16445d9a5afaa7772d6c9e8c11221f0743592814e884e3df61eb0`.
These checks used deployed runtime `d8f82f3a48f6`; no candidate policy
outcome had yet been observed when the declaration was fixed.
