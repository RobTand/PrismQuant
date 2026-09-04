# Tessera campaign checkpoint identity — 2026-09-04

PrismaQuant #186. Work is on `codex/pq186-campaign-identity`, based on the
#183 capture branch after integration onto main `10dd23957d6cf9ad784f8eac59fb914f66f6d89e`.
All tests below run through deployed PrismaBuild `aa6d3cfa2f77`, CPU-only on
Sparky, serial, 1 CPU / 4 GiB, with `CUDA_VISIBLE_DEVICES=''` and interpreter
`/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python`. These are not GPU,
quality, or performance measurements.

## Original defect

Action `88f2bf869834b1ddccce8a2a879ebee89b76b4668c3c98814dbef2c4b9ddfe05`
ran the real main-entry regressions in `tests/test_tessera_campaign_resume.py`
before the fix: 2 failed, no skips. Both same-qname legacy anchors and a unit
from another model reached `campaign_cost_payload` under the current run's
provenance. Failure line:

```text
tests/test_tessera_campaign_resume.py:38: AssertionError: unverified checkpoint anchors reached the current cost payload
```

## Existing journal, explicit manifest pathname

The compatibility adapter preserves the existing `--checkpoint` JSON-file
path and reuses the cost journal's existing manifest/unit schemas. Its two
new tests first failed with `TypeError: prepare_journal() got an unexpected
keyword argument 'manifest_path'` at lines 15 and 34 of
`tests/test_cost_stage_checkpoint_manifest.py` (PrismaBuild `dd23c1ba41a7`).

Green action `4b5265e28732daea67753713827aecaa3d2e192fa3705c0c76772e2291180d9e`
ran `tests/test_cost_stage_checkpoint_manifest.py` and
`tests/test_streamed_cost_checkpoints.py`: 19 passed, no skips or uncollected
module reported. Receipt:
`afda5c038374392dc7e1706bb4b32e105ac7c58226cdae59a1eb02213400b2ce`.
This verifies the adapter and old directory-oriented callers, not the pending
campaign integration or its producer byte receipts.
