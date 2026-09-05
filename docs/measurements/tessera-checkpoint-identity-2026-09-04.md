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

## Campaign and exact priced-wire intake

The campaign now consumes the producer's generic `encoding_input_identity`,
`make_unit_record` and `verify_cached_unit` APIs; it does not define another
producer receipt grammar or stamp new identities on old anchors. The existing
cost journal binds the whole current input population, including actual score
rows when Hessians are off. An independent read-only review by the producer
API author found no contract misuse in this adapter.

Tests use immutable Tessera producer source `6343137` from
`/mnt/shared/tessera-runs/pq183-producer-source-6343137.tar`, SHA-256
`50b2b29da4b7e0c818cdde442303ebe9e40fd0f4a04d8d8603a2da3450a0d14f`,
extracted into the worker's temporary directory and supplied through
`PYTHONPATH`. The installed producer, release pin and frozen serving candidate
are unchanged. This dependency must land before campaign intake is usable;
an older producer explicitly refuses for lack of its identity API.

Expanded red action
`935b09c3641fd0fcdfe8c2875ed36f2fca79fb5da668880b577a30993daadf62`
restored only `prismaquant/tessera_campaign.py` from the exact published
pre-fix source `2c22feaca1deff9dc0ca217b9a11d1e7cdf4bca8` in its isolated
PrismaBuild snapshot, retaining the new tests. Its first completed attempt
had **17 failures, 0 skips**, in 59.49 seconds. Repeated red attempts were
withdrawn after reading that completed result. Every fresh priced fixture is
a real 32x256 BF16-source unit encoded through the production adapter and
priced through the actual main entry point; no encode or scorer is stubbed.

Failure lines in `tests/test_tessera_campaign_resume.py`:

- `:54`: unverified legacy same-name and foreign-unit anchors reached the
  current payload (two cases).
- `:108`: the checkpoint schema was the old identity-free campaign schema,
  not the shared journal manifest schema (same-input resume regression).
- `:129`: `DID NOT RAISE <class 'RuntimeError'>` after changing actual Hessian
  values under the same draw.
- `:168`: the same refusal failure for changed source weights, score rows
  with H off, corpus text, token IDs, H mode, menu, recipe, producer source,
  PrismaQuant source, and TP scope (ten cases).
- `:190`: the same refusal failure for missing, byte-altered and symlinked
  priced wire (three cases).

Two earlier expanded-run attempts are not correctness evidence: `07c390a0a3f1`
failed before pytest because a PB snapshot has no `origin`; `bcafc601f210`
used a 32-column fixture outside the campaign accountant's 256-column
superblock contract. The corrected run above supplies the genuine failures.

Fixed action
`b62e9617fbfef24f65ae0b1491a2c74825ad6ab2b1538e1036d0cfe7e75b9ed9`
ran these eight targeted files:

- `tests/test_tessera_campaign_resume.py`
- `tests/test_cost_stage_checkpoint_manifest.py`
- `tests/test_streamed_cost_checkpoints.py`
- `tests/test_tessera_campaign.py`
- `tests/test_tessera_campaign_packed.py`
- `tests/test_tessera_campaign_pins.py`
- `tests/test_architecture_doc.py`
- `tests/test_docs_staleness.py`

**95 passed, 5 skipped**, no uncollected module reported; CPU-only serial
Sparky, 1 CPU / 4 GiB, 112.32 seconds. All five skip reasons, verbatim:
`Tessera encodes need CUDA`. Receipt:
`49a11ee95fc88145da43d1e5c34e47b87355e4015e16265fe4294191afd903e3`.
The positive resume test preserves the exact measured cost dictionary and
wire bytes, changes only the output path/interruption limit, and replaces
`_measure_anchor` with a failing sentinel on the resumed pass. The refusal
tests also require the original checkpoint file to remain unchanged.

This is correctness evidence for a CPU-tested checkpoint gate. It is not
GPU qualification, a served quality/performance A/B, or completion of the
packed campaign/export bridge in #183; that gate remains closed.
