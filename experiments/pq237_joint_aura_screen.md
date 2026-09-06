# PQ237 fixed-teacher numerical screen

`pq237_joint_aura_screen.py` is a bounded research experiment. It exercises
the production joint projection lease on a real resident causal LM and
compares weight-only and joint-AURA selections against the same BF16 teacher.
It does not supply a serving runtime price or satisfy a promotion gate.

The default population is Qwen3-0.6B down projections in layers 0, 7, and 21,
with Tessera `E2M1_K1_R768` (A4), `E4M3_K1_R896` (A8), and
`BF16_K1_R896` (A16). Each route uses its own production H-aware render where
the wire supports H. The exact render is shared by both pricing arms and
every candidate forward. `ProductionWeightCache` holds all nine resident
renders; `PerturbedActivationCache` owns calibration capture and assignment
weight/activation hooks. The independent oracle hook only observes inputs and
cotangents, and directly forms the full local output residual in FP32.

The two authored calibration texts and four disjoint authored heldout texts
are committed in the harness. Each contributes 64 unpadded tokens by default.
All 16 Rademacher probes use the same full-vocabulary, causal-token Fisher
normalization and fixed teacher. The three signed local terms are added
before squaring. Every one of the 27 assignments gets a full-model heldout
KL and next-token NLL measurement. Selection uses additive calibration costs
under `min_bytes + floor((max_bytes-min_bytes)/2)`, a budget derived solely
from emitted blob lengths. The retrospectively best heldout assignment is
reported only as an oracle diagnostic. No heldout labels enter selection.

`paired_candidate_minus_baseline` preserves common-probe covariance by
subtracting the two squared samples before estimating standard error. It
uses the signed sum across the three selected units and therefore is an
assignment-level diagnostic, distinct from the additive cost used for
selection. Its probe standard error does not measure text generalization.
The heldout per-text values remain available for that separate question.

The receipt records model-file hashes, exact token IDs, source-file hashes,
the rendered tensor and serialized blob hashes, actual blob lengths,
activation scales/contracts, precision, seeds, per-probe terms, and every
assignment. Bpp uses only the selected quantizable parameters; unchanged
embedding, head, and other BF16 weights are excluded. Tensor rendering times
and Torch profiles describe this numerical experiment and must never be
used as served operator timings or TTFT predictions.

Generate the source manifest on the host before entering the known image:

```bash
python3 experiments/pq237_source_manifest.py --out /path/to/source-manifest.json
```

Mount the source, that manifest, the exact Tessera producer source, and the
model into the known GPU Docker image. The September 5 run used
`tessera/lfm25-teacher:base-61fc8a-cconv-245e314e`, image ID
`sha256:337dae6b15313ff7a46aad56ec200119c6416555fd21c1085661f1c7cbd13b88`,
and Tessera `ba582d476a3b6db9057ebd1385dc52926f171451`.
Use a unique, previously nonexistent output directory:

```bash
PYTHONPATH=/workspace:/opt/tessera/src python3 experiments/pq237_joint_aura_screen.py \
  --model /model --out /results/new-run \
  --source-manifest /run/source-manifest.json \
  --tessera-commit ba582d476a3b6db9057ebd1385dc52926f171451
```

The run explicitly fixes native and Torch threads to one. Its container was
limited to four CPUs, 48 GiB RAM, and 8 GiB shared memory. The session's user
explicitly prohibited PrismaBuild, authorizing this direct Docker run.
For later sessions, follow their governing execution policy.

Retained limitations: three units on one model, one rung per family, no
sparse-anchor interpolation validation, no prefill/decode benchmark, no
served operator timings, no uniform whole-model control, and no downstream
task suite. Numerical screen success leaves these routes research-only.
