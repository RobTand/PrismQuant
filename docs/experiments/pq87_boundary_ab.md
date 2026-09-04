# PrismaQuant #87: boundary instrument A/B

Status: prepared, opt-in; no physical measurement claimed. Run after the
Tessera #113/#5 GPU campaign, through one exclusive PrismaBuild action.

The first population is the local Qwen3-0.6B BF16 checkpoint. It is a small
non-DeepSeek thinking model, so it tests the contributor's claim about the
request contract without re-serving the retired 87 GB Gridbook artifact.
It cannot reproduce that artifact's reported 83/180 result.

Use the same immutable container image and live BF16 server for A/B:

1. A: raw `/v1/completions` at the historical 64-token cap.
2. B: chat `/v1/chat/completions` at the identical cap.
3. Grow the BF16 chat cap by doubling, stopping at an uncensored schedule or
   the exact live context remainder. Save every step, including an inconclusive
   context/backstop stop. The contract's 11-step backstop covers the initial
   64 through 65,536 budget; it does not define a successful result.
4. Repeat the BF16 schedule as candidate for a same-session A/A control at
   the selected cap. A quantized candidate can subsequently use the exact
   control receipt, provided its observed stack, boot, input tokenization and
   action/campaign identity match. Preserve distinct live serve identities.

The frozen input is `experiments/pq87_boundary_contract.json`: the existing
five terse prompts, six seeded repetitions, temperature 1, top-p 1. Every
outcome is retained and re-scored. No rate or zero-defect threshold decides
artifact eligibility. The fixed-point cap is selected on this schedule, so
these observations must not select calibration content or claim an unbiased
artifact-selection test.

Example client invocations inside the already running serving container,
with the exact read-only source at `/repo`, BF16 model at `/model`, persistent
run output at `/run`, and a unique nonce as the served model name:

```sh
python3 -P /repo/tools/measure_boundary_control.py control \
  --base-url http://127.0.0.1:8000 --model-name "$BOUNDARY_MODEL_NONCE" \
  --model-dir /model --image "$BOUNDARY_IMAGE_DIGEST" \
  --campaign-id "$BOUNDARY_CAMPAIGN_ID" \
  --contract /repo/experiments/pq87_boundary_contract.json \
  --legacy-ab --out /run/bf16-control.json

python3 -P /repo/tools/measure_boundary_control.py candidate \
  --base-url http://127.0.0.1:8000 --model-name "$BOUNDARY_MODEL_NONCE" \
  --model-dir /model --image "$BOUNDARY_IMAGE_DIGEST" \
  --campaign-id "$BOUNDARY_CAMPAIGN_ID" \
  --control /run/bf16-control.json --out /run/bf16-repeat.json
```

Warm up the live server before either pre-manifest is taken, so first-use
library loading is not confused with a stable-stack A/B. The client discovers
the actual serve processes, source model path, nonce, context and chat-token
counts and refuses a changed pre/post session. It uses the existing stdlib
source bootstrap and fingerprint collector without importing torch into the
measurement client. The numerical-stack fingerprint must agree across paired
arms; any mismatch is a refusal, not permission to weaken the pairing.

Capture the exact safetensors content manifest before server startup, serve
that frozen artifact through a read-only mount, and compare the client's
recorded content identity with the pre-launch identity. The client retains
full pre/post content manifests and hashes their combination with the existing
model metadata into `artifact_id`; weight-stat checks also refuse a write
restored to its original bytes during the interval. This client alone cannot
prove what an already-running server loaded before its first observation.
The outer launcher owns that pre-launch/frozen-source assertion.

The enclosing action must reserve the entire physical GPU token capacity,
use one server at a time with explicit KV bytes/max-sequences/context limits,
and collect the server log, profiler window, power and Netdata series from
both boxes. Report work per joule if making a speed claim; GPU utilization
alone is not diagnostic. No launch permission or capacity is implied by this
document, and this preparation has consumed no GPU slot.

After the physical result, the remaining decision is which versioned,
control-relative rule replaces the current mandatory 64/zero boundary gate.
That is separate from merging the endpoint/schema and empty-population fixes.
