# Mixed LFM serving wrapper (#253)

`experiments/lfm_mixed_serving.py` is an opt-in, bounded host validation action.
It consumes the completed PB-assembled export, never a partition. It verifies
all PB result file hashes, the original six-scale calibration seal and the
immutable encoder source before constructing an artifact seal outside the model.
The native mixed gate validates all 74 owners and 2,178 matrices before load.

The sequential graph is raw prefill/decode census, observational census replay,
unchanged strict attestation replay, then matched BF16/student greedy smoke.
Only the exact absence of dense cell attestations is an expected strict refusal;
other raw/strict failures stop the action. Strict refusal never becomes
production admission. A non-recorded derived smoke status exits nonzero while
retaining the pair. All text/token receipts remain available for review; no
quality or speed certification is implied.

Each actual container receives native threads1, build parallelism4,64GiB memory
and the PB CPU affinity, an exclusive task label and a captured immutable ID.
Existing names are refused. CID polling waits for complete64hex contents; each
server's port must be unused before launch. Only a container matching both
captured ID and owner may reach the existing cleanup helper, including failure
paths. Inspection output is reduced to resource/image/ownership facts; arbitrary
daemon environment and scope tokens do not enter publishable host records.
Both-host Netdata plus local power,CPU,memory and container bounds are retained.
Fresh task-local output avoids the observed shared-subdirectory Docker mount
failure. The artifact and source remain under the existing shared root mounted
read-only. No rendered-weight or activation cache is introduced.

The original preparation remains Tessera6faa5ce/producer47dd, WikiText train
32×512 seed0, policy `legacy_6_over_calibration_amax.v1`, with no Hessians or
expert-row claims. Encoding and serving use actual Tessera
`7018fa2222925416b4c88cc8b6afab834dcac906` and immutable EUGR
`eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c`.
The original calibration seal is
`d302740baa39a484135a5de73abfcf9d5c3ec91eb419cb59a3d4af94c2704952`.
The actual assembled export_identity binds the new encoder and unchanged scale
SHA separately. No runtime pin or production default changes here.

Root supplies two new exact digests after PB completes: the immutable encoder's
full source manifest SHA and the actual assembly CAS result payload SHA. The
encoder manifest is an object with `commit` and `files` (relative name→SHA256)
covering every regular file; no Git files, symlinks or omitted producer scripts.
The assembly result is the actual PB `prismabuild.tessera-model.v1` result,
identical to the completed artifact's `pb-result.json`, with `index: null`.
Do not substitute a submission acknowledgement for that result.

Example admitted command (paths and digests supplied explicitly):

```bash
python3 experiments/lfm_mixed_serving.py \
  --out /home/rob/tessera-runs/mixed-lfm-237-2026-09-06/serve-01 \
  --encoder /path/to/immutable/encoder-7018fa \
  --encoder-manifest /path/to/encoder-source-manifest.json \
  --encoder-manifest-sha256 FULL_MANIFEST_SHA256 \
  --artifact /mnt/shared/tessera-measurements/mixed-lfm-237-2026-09-06/model-export-01 \
  --source /mnt/shared/models/LFM2.5-8B-A1B-BF16 \
  --plan /path/to/immutable/plan.json \
  --calibration /mnt/shared/tessera-measurements/mixed-lfm-237-2026-09-06/calibrate-01 \
  --assembly-result /path/to/actual-assembly-result.json \
  --assembly-result-sha256 ACTUAL_CAS_RESULT_SHA256 \
  --seconds 5400 --port 8198
```

Submit through published `pbrun.py` with four CPUs,64GiB,one exclusive physical
GPU and sufficient cleanup time beyond the bounded action. Native thread env
should also be1 on the host. PB owns placement; declare the actual immutable
image/source/model dependencies. The root chooses final execution. No GPU or
model serving run is part of this source delivery.

The existing Tessera smoke shell's prelaunch removal/cleanup/native-bound gap
is filed as [Tessera375](https://github.com/RobTand/tessera/issues/375). This
wrapper uses established plugin launch, request, compare, build-identity and
metrics helpers directly; frozen Tessera source stays unchanged. The old183
cached96/H campaign and old ts5 campaign-specific paths are not invoked.
