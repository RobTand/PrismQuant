# Related native and full-engine runtime observations

Research boundary for PrismaQuant #323. No measured runtime table is admitted by
this change: the producer still lacks a qualified, recomputable full-engine fixed
resource and timing partition (Tessera #399). Native operator evidence and raw
engine ledgers retain their original scopes.

`prismaquant.runtime_provenance_relation.v1` has an independent canonical JSON
identity. A `prismaquant.measured_runtime_context.v2` binds that identity through
`runtime_sha256` and names its role with
`runtime_identity_kind=prismaquant.runtime_provenance_relation.v1`. It never
substitutes one original native or engine runtime digest for another.
`prismaquant.measured_runtime_prices.v2` supplies the relation artifact in
`runtime_provenance`, and an exact `native_receipt_bindings` roster. Parsing v2
alone cannot supply allocator resources. Loading must verify original receipts,
the relation, native intake, and the fixed-resource producer gate. V1 parsing
and its existing independent producer-admission requirement are unchanged.

## Relation envelope

Every artifact reference contains exactly `path` and its raw-byte `sha256`.
Relative paths in the relation resolve beside the relation file; references in
the table resolve beside the table. JSON artifacts reject duplicate keys and
nonfinite numbers. The relation contains:

| Field | Evidence and role |
| --- | --- |
| `schema` | `prismaquant.runtime_provenance_relation.v1` |
| `configuration` | Exact selected serving configuration file. Image, engine arguments, environment and launcher selection bind it. |
| `image_manifest` | Exact Docker v2 or OCI platform manifest bytes whose digest is the pinned image reference. Its `config.digest` explicitly relates Docker hosts reporting a config ID to hosts reporting the manifest ID. Original local IDs remain in each run. Unrelated IDs refuse. |
| `package_source` | Original source `archive` reference, relative package `prefix`, and explicit `excluded_files` roster. Installed files must match the remaining archive entries by name, byte length and SHA-256. The source-tree and installed-subset source seals are separately recomputed. |
| `runs` | Named observations, each with `scope`, `runtime`, `runtime_field`, `installation`, `post_core`, `post_package`, and `instrumentation`. |
| `full_engine_run_id` | Exactly one run has `scope=full_engine`; every other run has `scope=native_operator`. |
| `production_dependencies` | Every native production library gets exactly one `{native_run_id, native_path, full_engine_path, sha256}` relation to bytes actually mapped by the engine. Same-path mismatches refuse. A different path still needs exact byte equality. |
| `full_engine_extra_libraries` | Every remaining engine production library maps to `{sha256, scope: full_engine}`. No undeclared extra library or missing native dependency is accepted. |

The `runtime_field` is `null` for a standalone runtime observation and `runtime`
for an original native preflight/receipt envelope. Embedded intake verifies the
original `runtime_sha256`; it does not edit the envelope or discard its other
fields. Native dense and routed schemas remain distinct. Full-engine intake
reads `tessera.full_engine_runtime.v1`, including its actual execution and
embedded loaded-package observation.

The shared coordinates include pinned image and config identities, full GPU
identity, arithmetic and library versions, stock-core manifest and file count,
plugin archive/revision/entrypoints, installed file roster, and the independently
recomputed package seals. Installer and post-run core audits must agree. Every
loaded Tessera module must provide its canonical `__file__`, `__spec__.origin`
and actual byte digest, within the installed package roster; the package and
`cached_unit` modules must be observed. This is artifact intake: PrismaQuant
imports no Tessera serving runtime.

Instrumentation declares `libraries`, `python_sources`, and `artifacts`.
Resource collectors and optional BLAS workspace observers have separate loaded
paths, binary/source references and build receipts. Both the original single
collector build envelope and the newer `builds`/`files`/`source_files` envelope
bind source and output bytes. Installed production libraries cannot be excluded
by labeling them observers. All base/native/full-engine harness source digests
require exact source artifacts, including the full-engine worker source. An
observed native ownership rule needs its own exact artifact. Unknown observer
roles refuse. These differences explain measurement instrumentation; they do
not normalize or erase production dependencies.

## Native rows and incomplete full-model resources

Each native row binding names `unit`, `format`, `run_id`, `panel`, `receipt` and
`memory_trace`. The panel retains its original same-run runtime; existing dense
or routed intake rechecks frozen QDQ/numerical limits, actual route, tensors,
operator identity, and the original trace. Every wire member must retain the recomputed source-tree
encoder seal, never the installed-package seal. Table rows must match the observed
resource bounds, repeated prefill/decode samples, warmups, whole-unit member
binding, exact cost/source/calibration identities and token scope. Both phases
are required; the present native intake supports batch size one.

`admit_fixed_resources` deliberately refuses every current fixed-resource claim.
A `complete` status, opaque hashed proof, whole-engine peak, or raw ledger does
not prove a full partition. There is no positive measured-table fixture and no
SLO, serving-lane, release-pin or default promotion. Positive CPU fixtures test
only relation and native-intake contracts; their synthetic numbers are not
measurements.

## Retained producer evidence

The actual native `prepare-05` and engine `full-engine-startup-r4` artifacts under
`/mnt/shared/tessera-native376-resource/` use the same selected configuration
`f5064609...` and matching common-path native library bytes. They ran on different
GPU UUIDs, so that pair refuses the shared-device gate. A future same-device run
must supply its own fresh artifacts; neither observation can be relabeled.

The raw image witness is retained at
`/mnt/shared/tessera-native376-resource/runtime-provenance-323/image-manifest.json`.
Its bytes hash to `4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33`
and its config digest is
`b6801f819d2566b099a4e73801dc70eae9c8f5989a223f30064b3cc8dee428cc`.
This establishes two Docker ID representations of the same pinned manifest.

The original plugin archive is
`/mnt/shared/tessera-clean-runtime-20260907/tessera-382-source.tar`, SHA-256
`2703772a800c14ac1288df49e652913dc6d3678ec8a4c07ffd1f8ebe8372dbe1`.
The producer source-tree seal `57809bff...` and installed-subset seal
`8239d565...` are distinct: the installation omits five `_dev` Python files.
The original packaging diagnostic lists these files at
`native-moe-original-r1024/package-identity-diagnostic.json`. The relation checks
archive bytes and recomputes both roles instead of equating the two seals.

The proposed, intentionally unadmitted relation and its panel-03 context are
retained in `runtime-provenance-323/relation-prepare05-engine-r4.json` and
`context-panel03.json`. A second refusal is explicit in
`missing-production-dependencies.json`: native preparation mapped generated
Triton `cuda_utils` bytes `cea8029c...`, while the engine's temporary cache mapped
`72570889...`; no engine library matches the native bytes. The relation leaves
that dependency unbound and must refuse. Reusing a qualified immutable cache in
a future run requires fresh actual loaded-byte evidence; source or ABI
similarity does not satisfy this byte relation. The explicitly selected
`experiments/pq323_provenance_artifact_checks.py` checks each original runtime
individually and asserts the pair's GPU refusal through PrismaBuild.
