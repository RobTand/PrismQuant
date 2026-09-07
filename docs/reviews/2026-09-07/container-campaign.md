# Campaign container execution

The campaign dispatcher now carries its explicitly selected Docker runtime
through census, anchor and selected-wire materialization rows. The worker
mounts its sealed checkout, runs as its own UID/GID, forwards the declared
environment and data mounts, and executes the resolved immutable image ID.
PrismaBuild continues to own placement, resource reservations and containment.

Validation through the published PrismaBuild runtime `67a44bb72cf7`:

| Action | Mode and result |
| --- | --- |
| `68e0d76bfb7571bcbcf5d955eaff924400c92eb0b3c95e7afacb7aef5c97304c` | Before: base `70e5f7a8` plus new tests; Docker, CPU only; 6 failed, 1 passed. The old dispatcher ignored the container configuration. |
| `2c81ad059dec1226a14a2b30a5d15308385edb16f9ad5ba5dbc2b5003b986b15` | After: Docker, CPU only; 7 passed, no skips. |
| `3ca70ffd7bec71ca66004687b02848c05acb7cf23b239ad6db49c86fcf945143` | Container, existing fanout, architecture and documentation tests; qualified GB10 interpreter, CPU only, 4 pytest workers/native threads 1; 59 passed, no skips. |
| `6d4412cc3afb7c02752b643135255874b70e593d5132439e03656508b01978e6` | Python compilation of both tools and the new test module; exit 0. |

Terminal records, successful CAS payload hashes and logs were checked. Evidence:
`/mnt/shared/tessera-measurements/first-model-20260907/validation/container/receipts.json`.
The Docker tests used `prismaquant-qwen38-producer:20260827-tf516-hf128`;
the broader checks used `/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python`.

Two setup failures are excluded from those results: `b5a87d76997b` had no Torch
in the CPU test interpreter, and `1114fe5d737f` had no pytest-xdist in the
producer image. Both were replaced by the qualified environments above.

The initial full-model census attempt `5c9d294c707d` independently exposed
the shared-output problem: container root failed to create its cache directory
on the NFS mount. Using the worker UID/GID passed that point and loaded all
30 dense and 2,112 projected expert units. Calibration and final-model
quality/performance are separate work; these tests do not qualify a model.
