# PQ237 streamed fixed-teacher qualification

This is a predeclared diagnostic for [#261](https://github.com/RobTand/prismaquant/issues/261),
related to the still-open [#237](https://github.com/RobTand/prismaquant/issues/237).
It exercises the actual streamed producer, durable checkpoints, persisted
joint currency, and production candidate constructor. No model inference or
GPU qualification has been executed for this protocol as of preparation.

This preserves the earlier ranking panel's E2M1 K1 and calibrated FP8
preclipping contracts. Tessera 7018's serving panel uses different E2M1 K2
and unclipped FP8 operators. Neither the retained bytes nor these quality
measurements provide runtime prices or serving qualification for those
different operators; that panel must be declared separately.

The source-inspected prerequisite is the materialized `ProductionWeightCache`
path. It supplies the five actual tensors and captured static activation
maxima. The transient-anchor Tessera A4 scale omission is separately tracked
by [#262](https://github.com/RobTand/prismaquant/issues/262). This experiment
does not change that producer, the format menu, architecture, allocator,
runtime pin, or serving gates.

## Frozen inputs and questions

The immutable [protocol](protocols/pq237-streamed-20260906.json) has raw SHA-256
`f2268edac42bcaed2e0f4a0766357822bdf84c78db36b3488bd715d29f2e22f6`.
It records every input-file hash, full paragraph and its origin, sequence ID,
and exact token IDs. The driver requires that digest independently and checks
the actual input files. Container path mappings change no identity.

Model: the local BF16 Qwen3-0.6B checkpoint identified in the protocol. The
teacher is fixed throughout. Resident and streamed BF16 calibration logits
must be bit-exact before probing. A mismatch refuses the run; it does not
recenter the teacher or silently relax a tolerance. All forwards use eager
attention, no KV cache, BF16 model arithmetic, FP32 residual projection,
TF32 disabled, and one native/Torch thread. The streamed model uses its
existing prefetch owner with required prefetched residency.

Only these three dense down projections are quantizable in the experiment:

| Unit | Exact production renders |
|---|---|
| `model.layers.0.mlp.down_proj` | E4M3 K1 R896 (A8), BF16 K1 R896 (A16) |
| `model.layers.7.mlp.down_proj` | BF16 K1 R896 (A16) |
| `model.layers.21.mlp.down_proj` | E2M1 K1 R768 (A4), BF16 K1 R896 (A16) |

The four assignments cross L0 A8/A16 with L21 A4/A16; L7 is always A16.
Every other parameter remains source BF16. A16 names the Tessera activation
family and still has lossy compressed weights; it is not the unmodified
BF16 teacher. New render identities are declared. The earlier artifacts
contain hashes, not reusable rendered tensors, and do not authorize a claim
that the new tensors or blobs reproduce the old bytes.

The calibration is the original two authored texts, 64 unpadded tokens each.
The original four authored heldout texts, also 64 tokens each, are retained
as a diagnostic of the previously observed reversal. Both sets must match
the exact token arrays committed with the September 5 result.

The fresh heldout panel contains 32 distinct WikiText2 validation articles
from `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. In source order, take each
distinct top-level article's first non-heading paragraph of at least 1,024
characters; take the first 32 qualifying articles and the first 64 tokens of
each paragraph, with no padding or special tokens. The paragraph, heading
and paragraph row indices, title, raw Arrow hash, and tokens are frozen.
Duplicate full token sequences within or across splits refuse preparation.
No outcome was available when this panel was chosen.

Preparation initially interpreted the rule as requiring the literal first
paragraph of each article to qualify. The pinned corpus contains 60 distinct
articles but only four meet that rule, so preparation failed before writing
an output. The clarified first-qualifying-paragraph rule has 50 eligible
articles; the first 32 were sealed. That correction and its negative test are
recorded below; the panel size was not reduced after seeing a result.

Exactly 128 common causal Rademacher probes use seeds 237000–237127, full
vocabulary, temperature 1, and the producer's global KL Fisher normalization.
Probe count, calibration, source model, arithmetic, and producer source
identities are constructed from the frozen inputs before the producer runs.
The same cached renders and activation QDQ serve every diagnostic and actual
candidate forward. The Hessian text identity hashes newline-joined original
calibration paragraphs in frozen order, alongside actual fitting-token hashes.
The driver explicitly sets `PRISMAQUANT_PROD_ACT_SCALES=1`,
`PRISMAQUANT_NVFP4_ACT_EMULATE_SERVED_SCALES=0`, and
`PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE=0`, fixing the earlier diagnostic
policy instead of inheriting an ambient process setting.
Tessera A4 continues to use its mandatory static contract under those values.

For each L0 background, the declared contrast is L21 A4 **minus** L21 A16.
The background interaction subtracts the A16-background contrast from the
A8-background contrast. The report keeps these quantities separate:

- Additive joint AURA, `0.5 mean_k sum_i a_i[k]^2`, is the production candidate
  mean currency. The handoff requires UCB z=0 and exact candidate coverage.
- Joint quadratic, `0.5 mean_k (sum_i a_i[k])^2`, is a signed first-order
  assignment diagnostic that retains cross-unit terms. It does not update
  the background or become an allocator refinement.
- The additive weight-component diagnostic uses only each persisted signed
  weight term on those same probes/renders. It is explicitly not a second
  joint-currency cost table.
- Actual fixed-teacher forward KL and next-token NLL are measured separately
  on calibration, original heldout, and fresh heldout sequences.

Matched-probe differences retain common-probe covariance, including the
unchanged units. Their standard errors condition on the fixed calibration.
Sequence differences retain the same sequence ordering; the reported
`SD(difference)/sqrt(n)` is descriptive for this fixed article panel. It is
not a population confidence interval. No heldout value chooses an assignment,
corpus, probe count, favorable background, or stopping rule. All four outcomes
and raw per-probe/per-sequence values are retained, including inconclusive or
negative results. An additive unary ranking cannot represent both background
signs if the actual forward contrasts reverse; increasing probe count does
not remove that model limitation.

## Execution and refusal gates

`pq237_joint_aura_streamed.py` first captures original calibration inputs
through `PerturbedActivationCache` and renders exactly five production cache
entries. The small cache remains resident, is pickled and reloaded as the
actual `ProductionWeightCache`, and is consumed by
`compute_aura_cost_streamed(joint_activation=True, formats_by_qname=...)`.
It records source/render/blob identities and blob lengths. The producer's
normal per-layer prefetch/adjoint/checkpoint path owns the work. This dense
experiment does not claim packed 3D routed-expert support.

Original wire bytes are retained in the existing campaign's `wire/` naming
and atomic publication convention. Producer `cached_unit` records bind each
blob to the actual production `ActivationSource`, source weight and encoding
recipe. A second bytes-only decode must hash to the exact PWC tensor. These
are original encoded outputs beside the one rendered-weight cache; they do
not require re-encoding or a new cache mechanism for a later investigation.

The driver then replays the exact durable checkpoints, checks unchanged
signed rows, persists the payload, and validates it through
`require_run_currency` and `build_candidates(preserve_runtime_frontier=True)`.
The complete five-coordinate roster and unchanged mean costs are mandatory.
Any source/render/scale/arithmetic/probe/scope mismatch refuses. No synthetic
serving context, runtime price, or relaxed candidate mask is supplied.

An independent first-probe residual oracle forms
`Xhat @ What.T - X @ W.T` in FP32 with the shared activation QDQ, using a
resident full-model backward rather than the joint lease. Every streamed
projection must match within `2e-5 + 2e-4 * abs(direct)`. The source teacher
weights must be exactly restored after candidate evaluation.

Serialized blob bytes and allocator tensor-payload bytes are reported with
different labels. Bpp divides complete selected blob bits only by these
three quantizable units' parameters. No whole-model or matched-budget
allocation claim follows from this four-assignment diagnosis.

The first two calibration sequences of each assignment have an in-process
CPU/CUDA profile including weight materialization. These are cold diagnostic
profiles, not served timings or a speed comparison. Preserve a concurrent
host telemetry capture from both fleet machines with the eventual PB
receipt; GPU utilization on GB10 is not a saturation metric. No GPU-bound
or performance improvement claim is predeclared for this small experiment.

The current producer exposes one coupled forward/reverse invocation for the
common probes. Submit that dependency-preserving action through PrismaBuild;
PB owns placement and balancing. No application-side host partition or second
dispatcher is introduced. Any later fanout must use a supported PB contract
that preserves these calibration, probe and cache identities.

## Prepared launch

The shared immutable input root is
`/mnt/shared/tessera-measurements/mixed-lfm-237-2026-09-06/joint-inputs`.
The exact Tessera source root is
`joint-inputs/tessera-7018fa2222925416b4c88cc8b6afab834dcac906`, with `joint-inputs/tessera-7018fa2222925416b4c88cc8b6afab834dcac906-source-manifest.json`.
The full source manifest has SHA-256
`81dbf9c1ee123b5bdd8bdf3b09f2705b414e5070f70ee40e5779bbaa935f545c`
and binds all 1,029 source files. Its map matches the immutable Git archive
`07ee4eb51a7fa23051e0615313e6057bb016c6806ce0aefbb41cb61b56410b4a`.
This changes producer/render identity from the earlier prepared 6faa source;
the frozen corpus and probe manifest is unchanged.

Launch only after the coordinating task releases the mixed-LFM sanity gate.
Do not treat this prepared command as a GPU result.

The known image ID is
`sha256:337dae6b15313ff7a46aad56ec200119c6416555fd21c1085661f1c7cbd13b88`.
Read-only inventory found it on Sparky and absent on Sparklina, so the launch
declares the actual `sparky` image dependency. Remove that host restriction
only after the same image has been staged and qualified on another worker.
The admitted action first verifies and extracts the immutable source archive
to a temporary local directory, then mounts it read-only as `/tessera`. This
is code-input preparation; production weight and activation caching remain
with their existing owners. The local file-hash pass avoids the NFS metadata
stall observed when checking the materialized shared source one file at a time.
The PB Docker shim preserves its assigned CPU affinity and process ownership.
Native threads remain one; two prefetch workers share the four-CPU admission.
The 48 GiB reservation/container limit retains the prior small-model screen's
known working envelope; it is not a measured peak-memory claim.

After committing the reviewed source, run from this checkout. This Python
submission wrapper temporarily omits any remaining legacy external calibration
symlinks while PB seals the checkout, restoring them in `finally`. Main removed
these tracked links in #269; a checkout without them also passes this step.
The experiment reads only its own frozen inputs. The source manifest records
that omission; no deleted symlink belongs in the delivered commit. The manifest
now names untracked closure entries and in-tree bytecode caches in their own
fields rather than inside the opaque `status` text, and the reader refuses any
importable entry under a root that the manifest does not list.

Three later refusals extend that to the paths by which the interpreter can
still reach unhashed bytes. The reader loads its own closure policy from the
exact bytes it has just checked against the manifest, rather than by an import
a shadowing package could answer. `__pycache__` is walked like any other
directory, since redirecting cache lookups does not stop `__pycache__/x.py`
from importing; only the tagged cache files inside it stay unsealed. And a
symlink whose name or target can supply a module is admitted only when the code
behind it is itself sealed, so a declared link to code outside the root is
refused. A link to *data*, including the three `calibration/*.jsonl` links
above, is unaffected: it names nothing the import system can load, and the
manifest still pins its target string.

```bash
python3 - <<'PY'
from pathlib import Path
import subprocess

root = Path.cwd()
run = Path('/mnt/shared/tessera-measurements/mixed-lfm-237-2026-09-06')
links = {root / 'calibration' / (name + '.jsonl'): None
         for name in ('diverse-v1', 'xdom-fit-v1', 'xdom-gate-v1')}
links = {path: path.readlink() for path in links if path.is_symlink()}
try:
    for path in links:
        path.unlink()
    subprocess.run(['python3', 'experiments/pq237_source_manifest.py',
                    '--out', str(run / 'joint-inputs/source-manifest.json')], check=True)
    command = r'''from pathlib import Path
import hashlib, json, subprocess, tarfile, tempfile
from experiments.pq237_joint_aura_streamed import verify_source_manifest

inputs = Path('/mnt/shared/tessera-measurements/mixed-lfm-237-2026-09-06/joint-inputs')
prefix = inputs / 'tessera-7018fa2222925416b4c88cc8b6afab834dcac906-archive-input'
manifest = Path(str(prefix) + '.json')
raw = manifest.read_bytes()
if hashlib.sha256(raw).hexdigest() != '063aa02389cda1705a14e4f2736920a6cdc99ee56e96fe79b427776481743119':
    raise ValueError('Tessera archive manifest identity differs')
identity = json.loads(raw)
archive = Path(str(prefix) + '.tar')
if hashlib.sha256(archive.read_bytes()).hexdigest() != identity['archive_sha256']:
    raise ValueError('Tessera source archive identity differs')
with tempfile.TemporaryDirectory(prefix='pq237-streamed-source-') as source:
    with tarfile.open(archive) as bundle:
        bundle.extractall(source, filter='data')
    verify_source_manifest(source, manifest)
    subprocess.run([
      'docker', 'run', '--rm', '--gpus', 'all', '--network', 'none',
      '--memory', '48g', '--shm-size', '8g',
      '-v', str(Path.cwd()) + ':/workspace:ro', '-v', source + ':/tessera:ro',
      '-v', '/mnt/shared:/mnt/shared', '-w', '/workspace',
      '-e', 'OMP_NUM_THREADS=1', '-e', 'MKL_NUM_THREADS=1', '-e', 'OPENBLAS_NUM_THREADS=1',
      '-e', 'PYTHONPATH=/workspace:/tessera/src',
      # Bytecode caching is redirected out of both sealed roots. A
      # timestamp-validated .pyc under a root executes in place of the sealed
      # source that names it, so verify_source_manifest refuses an in-tree
      # __pycache__ unless the reader reads its cache from somewhere else.
      '-e', 'PYTHONPYCACHEPREFIX=/pq237-pycache',
      'sha256:337dae6b15313ff7a46aad56ec200119c6416555fd21c1085661f1c7cbd13b88',
      'python3', '-u', 'experiments/pq237_joint_aura_streamed.py',
      '--model', str(inputs / 'Qwen3-0.6B'),
      '--protocol', 'experiments/protocols/pq237-streamed-20260906.json',
      '--protocol-sha256', 'f2268edac42bcaed2e0f4a0766357822bdf84c78db36b3488bd715d29f2e22f6',
      '--corpus-arrow', str(inputs / 'b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-validation.arrow'),
      '--source-manifest', str(inputs / 'source-manifest.json'),
      '--tessera-root', '/tessera', '--tessera-source-manifest', str(manifest),
      '--image-id', 'sha256:337dae6b15313ff7a46aad56ec200119c6416555fd21c1085661f1c7cbd13b88',
      '--out', str(inputs.parent / 'joint-streamed-run-01')], check=True)
'''
    subprocess.run(['python3', '/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py',
                    '--cwd', str(root), '--gpu', '--tag', 'gb10', '--tag', 'sparky', '--cpus', '4',
                    '--demand', 'mem_gb=48', '--gpu-memory-gb', '32', '--exclusive',
                    '--timeout-s', '7200', '--detach', '--', 'python3', '-u', '-c', command], check=True)
finally:
    for path, target in links.items():
        path.symlink_to(target)
PY
```

Retain the action key and inspect its terminal record, log, CAS receipt, and
the actual output `receipt.json`/`identity.json`. A queued action or passing
CPU fixture is not numerical qualification. Use a new nonexistent output
directory for a reviewed rerun; do not overwrite failed evidence. The driver
exercises exact checkpoint replay within a run but does not automatically
resume a failed end-to-end experiment.

## CPU preparation evidence

All checks below were PB CPU actions with CUDA hidden. The helper tests used
four pytest workers and four reserved CPUs, native threads one; the streamed
fixture used one CPU. No skip was counted as a pass.

| Check | Actual result | Action / CAS receipt |
|---|---|---|
| New handoff before implementation | Import collection failure, expected red | `819a83a085898a615144b64281542c8857c2595d011751ea48a15bcbe11f206d` |
| First handoff attempt with driver | Missing Tessera dependency, not a behavioral pass | `28b4c70155d38eae9df9d65da71fdc463b486a849a7e86cfefd4010bd050f388` |
| Handoff + driver compile with shared Tessera source | 1 passed, zero skips; compile exit 0 | `49efd481a5971bf31b9404f5171ce93d92ddfed34bc62960d322c547f48977f6` / `939e0d1d6df6177cb2ada345941d730cc0297f1d7b43433c4da7fe620e6e549f` |
| Short-intro corpus parser regression | Failed strict parser, expected red | `fe64c3cf8b20249e231e475aeb0789179efaeffaf9b0f68739f9f5ad064c401b` |
| Revised protocol helper | 25 passed, zero skips | `c1da73c9b6dd572fda3978a4f4a974a54d3d1f7a36f9e60741cac9108fdb07da` / `a02deb1d7b7b8fdb15c6793eb5c24b6fed3f558d4bee10b05c93d5cf02aab3e6` |
| Helper compile and source-only article census | Exit 0 | `ce38012238519d642db1ef3c52b3809c980888772029a20c19eac7b043102c45` |
| Frozen corpus/token preparation | Exit 0, expected 2/4/32 sequences | `1a8f6778430934fb215957a718df85b4f764834ce60bb57b5e42a2cb356bf8c7` / `726272fb67ba2a19b57f884bdc8848e2bdeded5df3478b36fc87971a745e0086` |
| Rebased handoff and assignment diagnostics | 2 passed, zero skips | `06d3297981982f01c8234406616e73429cf8d38909f0f13b447c1b11076df78c` / `40cb0ef7a0000d398526c6759e378cd322787219dfe9a0a210155e7278eab453` |
| Rebased compile and frozen-protocol verification | Exit 0; 2/4/32 sequences and 128 probes | `16a3e849854421076f88e256e595f01c1f2d1b098545ae3dbc5aa102926d6a3f` / `b884602c58b5f53a9b66a1721f32ea17601f827ee6f5d974cd502583d58fac92` |
| Final wire/handoff attempt with per-file shared-source verification | Timeout at 302.8s; no tests counted | `2ea4384b5ab5abae48ec883bb6d38f030125142fb7561e730e3b5d51cea6bdb9` |
| Final wire/handoff with verified local Tessera 7018 source; driver/helper compile | 6 passed, zero skips; compile exit 0 | `57e3941848d22a509b7984ffb396d0e600a1860003cdfb2334e7a429c01435d9` / `47121d99d4e247d97ac9015bd2f5fb06f804511652500244a8f6e503fb84cc8c` |
| Prepared submission and nested admitted launch syntax | Exit 0; neither launch executed | `ce73b982efea7dc280e71df99abbc4e18dbd56c7a648f6a9655c3f1457e639f8` / `8dd22e47fd0e166ccfca45735a1d8e2e865ab196ea66f944dc82a69252c72df4` |

Terminal records live under `/mnt/shared/prismabuild-fleet/pb-queue/{done,failed}/ACTION.json`;
CAS payloads/receipts are linked by those records. The protocol test report is
`/home/rob/tessera-runs/mixed-lfm-237-2026-09-06/protocol-pbtest-final.json`.
Static review found and resolved two driver provenance defects before GPU
execution: self-derived expected probe digests and a placeholder Hessian text
hash. The rebased handoff checks include coherent full probe-identity rebinding,
bound source-shape/stat mismatch refusal, and all four assignment comparison
labels. PR #260 is merged at `fa229becf6a82752179a38189904209c0f5e8ad6`.

The per-file shared-source attempt timed out with 0.51 CPU-seconds, 45 MiB
peak memory, no test output and complete cleanup. It ran source verification
before importing Torch. Those observations are consistent with the NFS delay
tracked in [PB #205](https://github.com/RobTand/prismabuild/issues/205), rather
than a codec result; they do not identify the storage failure
causally. The retry uses the same immutable file map extracted locally from
the verified archive inside PB. This is the prepared launch path as well. The retry passed all six tests
in 8.25 seconds (37.4-second action), with 14 Torch deprecation warnings,
zero skips, one reserved CPU and 4 GiB RAM on Sparky. Four cases use the real
32×32 E4M3 weights-only encoder, record validator and decoder to check original
bytes/receipt reuse and refuse disk-byte, decoded-value and shape mismatches.
The other two exercise the actual streamed/persisted candidate handoff and
paired comparison labels. These CPU fixtures make no serving or GPU claim.

A separate CPU-only image preflight ran on Sparky using exact image 337dae
and verified Tessera 7018. Runtime imports, Qwen3 class import, Hessian API
acceptance and both CLI help checks passed: Torch 2.13.0+cu130, Transformers
5.15.1 and compressed-tensors 0.17.0. Action
`1673c5ca74cc5c08ed78502607edd4ff4c8ba44392f069ff9f8e258a08a83c53`
exited 1 because preparation-only `pyarrow` is absent. The frozen execution
path uses only standard-library protocol loading; it does not import
`pyarrow`, retokenize, or prepare another corpus. This preflight is retained
as a failed diagnostic, not a passing qualification. CUDA was uninitialized,
no GPU devices were exposed, and the PB/container CPU affinity matched.
Evidence is under
`/mnt/shared/tessera-measurements/mixed-lfm-237-2026-09-06/joint-cpu-preflight-01`;
`preflight.json` SHA-256 is
`2a0cf14d04592dc235de4400a1fdab763cad6ecff3e68b103340fcd2ca5cb626`.

This protocol leaves sparse-anchor validation, broader models/layers/rungs,
actual serving-route timings, whole-model uniform controls, downstream tasks,
and production promotion open. No result from it alone closes #237.
