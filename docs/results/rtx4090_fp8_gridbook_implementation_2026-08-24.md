# RTX 4090 FP8 Gridbook implementation evidence — 2026-08-24

## Status

The strict RTX 4090 FP8-CB path is implemented in the PrismaQuant and Gridbook
working trees, but it is **not release-qualified**. Gridbook v10 is neither a
released nor a PrismaQuant-pinned runtime, the strict Qwen3.8-27B artifact has
not been built and served under this contract, and no physical RTX 4090 gate
has run. The only sm89 evidence in this report is a no-device cross-compile
whose receipt explicitly caps its claim at `compile_only`.

The PrismaQuant changes were developed in a working tree based on
`e370e1d7183c683c6ddab4bef82dc1dc1cf06200`. The Gridbook change was first
developed from `edfb3bca38aa1faae627dab82caef76e6b1743a2`, then rebased as one
reviewed commit, `63e019f31b57c13b0fe9697441f5084eb194440a`, whose exact parent is
the shipped Gridbook v0.9.0 commit
`d8cef3fba5fd99f12ba99efccbb52188d449ed6d`. That integration preserves the
v0.9.0 TP/EP contract subtrees and leaves the v10 runtime-contract bytes at
SHA-256 `4ecb851e50fa8c16d860969925b13cad31f4e1b979ba6fec21534035487efa6b`.
It is a SHA-pinned candidate, not another v0.9.0 release, wheel, tag, or
PrismaQuant runtime-pin advancement.

## Implemented contract

| Surface | Implemented behavior |
|---|---|
| FP8-CB format domain | Gridbook v10 readers accept K4/K8/K12/K16/K20/K24 plus every historical integer rung K28 through K48. New producers emit exactly K4/K8/K12/K16/K20/K24/K28/K32/K36/K40/K44/K48. Reader-only off-law rungs are never rounded into a new menu or export. |
| Strict artifact policy | Serving profile `qwen38_rtx4090_fp8_cb` selects producer policy `qwen38_27b_rtx4090_fp8_cb`. The only legal terminal formats are FP8-CB producer rungs, delegated `FP8_E4M3`, and BF16. Native NVFP4 and NVFP4-CB are refused, `lm_head` and MTP are fixed BF16, and `CB_ACTIVATION_SCOPE=none` prevents an NVFP4 static-activation scalar or contract from entering the FP8-only artifact. |
| Codebook source | The strict artifact is lattice-only: `CB_CODEBOOK_SOURCE_SCOPE=none`, `CB_CODEBOOK_SOURCE=lattice`. Generic learned-v2 defaults all twelve rungs to lattice and can promote a rung only from a complete two-holdout receipt, but strict serialization rejects learned groups until the raw promotion ledger, imatrix identity, and complete source closure become part of the strict artifact attestation. |
| Exact census and value provenance | The released official-wrapper layout is closed at exactly 1,199 source tensors and 615 Linears. Preflight requires a complete one-to-one assignment and binds the streamed source identity/tensor map. Final replay checks config-group ownership, exact `ignore`, every serialized tensor key/dtype/shape, and every referenced FP16 codebook-sidecar table. It additionally requires bool-true/source-complete render identity, binds the canonical render imatrix, replays codebook bytes, closes and reconciles the strict per-tensor digest ledger, validates the exact weight-container manifest, and binds the producer Git identity to the shipcard. One shared `O_NOFOLLOW` sequential scanner computes container and raw-tensor hashes together. Strict serving runs one fresh pass inside the actual container immediately before vLLM, pins every shard with an individual read-only bind, and reuses a no-clobber stat receipt for post-serve census plus host endpoint stat replay and signed shipcard evidence without rereading payload bytes. The streaming writer reuses its in-write hashes and performs zero post-write content passes. |
| Size and context envelope | The artifact validator recursively inventories regular files. Publication freezes the complete upload tree and reapplies the same non-forceable ceiling after documentation/evidence authoring: at most exactly 18,000,000,000 bytes. The serve target is one live request (`n=1`) at a 32,768-token model limit and an FP8 KV allocation of exactly 4,294,967,296 bytes (4 GiB). The scheduler ceiling is 64 solely so vLLM can admit the required FULL-decode capture ladder through 64; it does not promise 64 simultaneous 32K contexts. |
| Gridbook boundary | `gridbook.runtime-contract.v10` separates reader `rungs` from `producer_rungs`. `gridbook.lane-eligibility.v2` resolves exact platform/family/structure/regime/rung cells. Even when an artifact selects a legal subset, the candidate runtime must carry external flag-free `sm_89` `device_qualified`/`backed` dense decode and batch cells for the complete twelve-rung producer ladder. Strict route provenance is derived from that supplied candidate-v10 attestation, not the historical generic serving pin, and its schema cannot carry generic override/non-native/fallback dispositions. PrismaQuant does not vendor or import Gridbook. The v0.9.0 TP and EP contract subtrees and runtime behavior are preserved. |
| Graph gate | The graph arm requires vLLM compilation mode 3, explicit Inductor, `FULL_AND_PIECEWISE`, capture sizes `[1,2,4,8,16,32,64]`, **7/7 PIECEWISE and 7/7 FULL** completion, and one direct `torch.compile` wrapper with `fullgraph=True` and `dynamic=False`. Positive fresh compilation and all-capture markers are required; compiler, eager, graph, or partial-capture fallback refuses the receipt. |
| vLLM boundary | The strict launcher requires a separate closed vLLM pin naming the official upstream Git URL, one exact 40-hex commit, exact installed version, and installed `RECORD` SHA-256. Server-side collection requires matching PEP 610 VCS provenance and RECORD-bound `direct_url.json` plus fullgraph-wrapper bytes; manifest validation and shipcard replay require the same pin. No candidate vLLM pin is committed, so an immutable image alone cannot open release. Generic serving evidence is unchanged. |
| Build path | The operator launcher uses validated-surrogate selection with `PRODUCTION_CACHE=1`, `PRODUCTION_RECACHE=0`, resident prefetch required, and streamed BF16 AURA. The burn plan, checkpoint identity, and runner all bind exactly two existing source `LayerCache` slots, effective lookahead one, and explicit `require_prefetched_residency=true`. That opt-in policy fail-closes forward capture and reverse AURA to resident or completed-prefetch layers; reverse schedules the next lower layer before current render/backward work, and missing required residency refuses before synchronous source I/O. Generic streamed runners retain the false default, and no parallel cache is added. One absolute AURA checkpoint directory owns `PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE`; source preflight revalidates that identity. `CB_IMATRIX_SOURCE=probe` supplies one full-corpus calibration/value identity to cost, frontier KL, and export. The official wrapper is accepted only after the nested dense text model and complete source census validate, then uses the established flattened text-only staging path. |
| Two-host probe and burn | The sample-parallel probe is an exact map/reduce contract over disjoint complete-sample partitions. A global CE barrier precedes Fisher work; raw Fisher traces and both marginals merge before one global normalization. Activation selection is independently replayed over the full priority domain and enters the existing activation-cache path. Cross-host equality uses path-neutral checkpoint/config/shard content while each host still validates its own paths and live files. The merged bundle, execution attestation, and probe-derived imatrix are closed before CUDA. Allocate validates the complete bundle, passes the already captured probe bytes through a sealed inherited memfd, and rechecks them after the allocator exits. The burn then stripes whole layers 0--31 and 32--63 across the two hosts and renders four formats per Linear, not twelve. `PRISMAQUANT_CB_COMPILE_FAIL_CLOSED=1` upgrades live CB scoring to full-graph Inductor with compiler suppression disabled and exact one-dispatch-per-call proof. Receipts distinguish live CUDA calls from units restored through AURA's already checksummed, deserialized, strict-settings/producer/source-bound envelopes. They explicitly stamp LDLQ atom execution `not_applicable` because this lattice campaign has LDLQ off. Generic CB runs retain their compatibility fallback. |
| Learned-v2 | All twelve producer rungs default to their committed lattice. A rung promotes to a learned table only when one `prismaquant.fp8_cbl_promotion_receipt.v1` result certificate passes both holdouts and binds the complete source/tensor-map identity, training calibration, derived imatrix values, role census, exact candidate FP16 table digests, and all rung decisions. There is no inferred learned/lattice crossover. |
| Release evidence and publication | Strict artifacts gain a mandatory `rtx4090.fp8_cb` shipcard slot derived again from on-disk policy. A physical eager arm and a physical graph arm are independent from quality and matched-performance evidence; a Spark or cross-compile cannot fill the strict slot. Any mutable slot failure, frozen-snapshot replay failure, or whole-tree size failure is non-forceable: `--force-unverified` and `--confirm-name` cannot stamp or publish it. The strict publisher folds whole-container, upload-block, and per-tensor hashes into its existing held-FD freeze pass, then replays the frozen manifest/ledger/index from that receipt without a second weight scan; generic publication behavior and the post-upload held-FD replay remain unchanged. |

The implementation extends the existing format registry, activation/imatrix
path, `ProductionWeightCache`, allocator, exporter, serving-profile boundary,
shipcard, and publication snapshot. It does not add a parallel cache or a
forked vLLM runtime.

The build launcher no longer adds a second post-export content traversal. Both
strict exporters perform finalized census with their same-process receipt, and
the launcher then performs only the cheap policy/inventory metadata replay.
The strict serve path independently performs exactly one in-container
traversal immediately before vLLM and reuses its stat-bound receipt thereafter.

The serving handoff has one explicit fail-closed TOCTOU boundary. Individual
bind mounts pin shard inodes, so replacing a host pathname cannot retarget what
vLLM opens. A host process can still write an inode in place between the
preflight process and vLLM's open; the later receipt replay observes changed
ctime/mtime and refuses all endpoint/shipcard evidence. Thus corrupted bytes
cannot qualify a release, but the implementation does not claim vLLM's read is
atomic with preflight. Avoiding even the transient possibility would require
vLLM to load inherited verified descriptors or would require a second
copy/content pass, neither of which this unforked runtime supports.

## Test and proof record

| Evidence | Result | Provenance and limit |
|---|---|---|
| PrismaQuant CPU regression selection | **466 passed, 13 skipped** | Implementation-session output reported before the final documentation rerun. No standalone log or complete selector was retained, so this count is supporting evidence, not an immutable receipt. Skips were GPU-dependent. |
| Final PrismaQuant integration selector | **979 passed, 4 skipped; 11 subtests passed** | Post-audit selector covered every touched probe, measurement, anchored-cost, strict compile-proof, streamed-prefetch, burn, source-identity, allocator, DSV4 AURA compatibility, RTX 4090 artifact/policy/graph/shipcard/validation-only, Gridbook-boundary/execution, format-registry, and documentation suite. Durable log: `/home/rob/dq-runs/qwen38-27b-rtx4090-fp8cb-18gb-vonly-20260824/validation/cpu-final-postfix-precommit-v2.log`; SHA-256 `f8289d2d8bb90625fedaa90c7739f985857c3ad5b05fc9a56480e380f65b8f96`. No GPU was exposed. |
| Canonical extensionless-dataset launch fix | **52 passed, 1 skipped** | The first immutable Sparklina `prepare-calibration` smoke refused before model load because the host JSONL was correctly mounted at canonical `/dataset`, but the generic loader inferred local files only from suffix and attempted the absent Hugging Face `datasets` package. Failure log: `/home/rob/dq-runs/qwen38-27b-rtx4090-fp8cb-18gb-vonly-20260824/validation/canonical-dataset-first-smoke-failed.log`; SHA-256 `f8accef513f7dff9d521e7e978f64397fceb9f685b5ae5723ec2d9ab22639869`. The loader now content-sniffs only existing extensionless regular files while preserving known alternate-format behavior; focused loader/sample-runbook/docs log: `/home/rob/dq-runs/qwen38-27b-rtx4090-fp8cb-18gb-vonly-20260824/validation/canonical-dataset-fix.log`; SHA-256 `b8379191513758135aa2c426c085617a176c31efd9bd106bae9d92158d085a7b`. |
| Staged-text source-census launch fix | **165 passed, 1 skipped; live 866/866 header equality** | The next immutable `prepare-run-contract` preflight refused because the promoted text config declared the flattened namespace while its released physical index intentionally retained exact `model.language_model.*` body keys. Failure log: `/home/rob/dq-runs/qwen38-27b-rtx4090-fp8cb-18gb-vonly-20260824/validation/staged-text-census-first-smoke-failed.log`; SHA-256 `0956e55170302a6fd49e661c7cc2b4e2060b3a581b23d3146b7711aa3826b439`. Census comparison now applies only the established `model.language_model.* -> model.*` text-staging law, refuses mixed namespaces/collisions, and leaves the raw weight map in the value-bearing source identity. The live source's 866 observed headers then equal all 866 authoritative headers. Focused sample merge/artifact-census/policy log: `/home/rob/dq-runs/qwen38-27b-rtx4090-fp8cb-18gb-vonly-20260824/validation/staged-text-census-fix.log`; SHA-256 `13b70cb6ad38fad56398f88dc33927ca8927dfc10b20690c83f9ab12a3a8f795`. |
| Live two-host CE/Fisher barrier and MTP schema repair | **global CE closed; Fisher publication correctly refused; repair 165 passed, 1 skipped** | Both workers closed an exact 32,736-token CE cover with global mean `1.615080892165735`. Sparklina then completed the numerical body, reverse, MTP, and head passes but refused publication because all eight MTP rows lacked the five dense marginal fields required by the signed all-qname schema; no probe, merge, or burn was published. Retained failure log: `/home/rob/dq-runs/qwen38-27b-rtx4090-fp8cb-18gb-vonly-20260824/commit-0641aaf2/coordinator/failure-worker1-fisher.log`; SHA-256 `10949b3b6d52e98e3582d1e4eed76795e1dffd928b708e8aa56d2c5ccb400506`. Repair commit `b1db8321351728e64a2fbdb48c1ddac7339594bc` makes the generic Fisher accumulator emit those vectors only for sample-parallel MTP, accumulates them on device, drains once at finalize, and replays the same closed validator for direct and per-Linear-cache resume. The five vectors total 1.172 MiB per worker versus about 1.582 GiB of existing MTP Fisher matrices. Focused CPU log: `/home/rob/dq-runs/qwen38-27b-rtx4090-fp8cb-18gb-vonly-20260824/validation/mtp-marginal-schema-precommit.log`; SHA-256 `9325485c7e070cbb4c56758af581f822fb6e0aaff6b415bd713c87a91a0ec84c`. No GPU was exposed to the repair tests. |
| Combined FP8/NVFP4 packaging regression | **980 passed, 22 skipped** | The FP8 repair was rebased without conflict onto the completed RTX5060/NVFP4 implementation line, including its deterministic-operation policy and exact-cost allocation resume fix. The tested integration baseline is `b1db8321351728e64a2fbdb48c1ddac7339594bc`; later orchestration commits require their own gate. The CUDA-hidden selector covered the two format families, strict RTX4090 no-NVFP4 policy, sample-parallel merge/resume, compile and graph contracts, Gridbook boundaries, artifact census, serving profiles, and architecture/docs gates. Log: `/home/rob/dq-runs/qwen38-27b-rtx4090-fp8cb-18gb-vonly-20260824/validation/cpu-overlap-b1db832.log`; SHA-256 `d3d7c2d42d5ab1263a6dcf7defabaa5a2a0bed67d6e50f54ba086c5fb27252cb`. The 22 skips are CUDA, optional GGUF, or separately pinned Gridbook compatibility jobs; this row is structural evidence, not physical RTX4090 qualification. |
| Separate frozen DSv4 W8A16 diagnostic | **2 failed, 983 passed, 4 skipped; 11 subtests passed** | A deliberately broader diagnostic also selected `tests/test_dsv4_w8a16_export_handoff.py`. Its two closure tests refused because four reviewed source hashes are already stale at this branch's pre-change `HEAD`: three untouched files and `nvfp4_cb_formats.py` all disagree before this working-tree change. The RTX4090 work does not authorize re-freezing that independent exporter. Failure log: `/home/rob/dq-runs/qwen38-27b-rtx4090-fp8cb-18gb-vonly-20260824/validation/cpu-final-postfix-precommit.log`; SHA-256 `e2d25c4e3f631c234a81f69e1eb6449ef1994a1b1f0670df6e84c24f804b01d8`. No GPU was exposed. |
| Exact runbook audit | **201 passed, 1 skipped** | All concatenated Bash fences pass `bash -n`; all 12 documented sample/burn command argument sets parse against the live CLI; the incremental-probe options, snapshot/bootstrap parsers, canonical container paths, non-root writable caches, RepoDigest authority, barriers, transfers, compile flags, and FP8-only menu align. No GPU was exposed. |
| Gridbook CPU proof selection | **539 passed, 36 skipped** | Agent-reported proof output. No standalone log or complete selector was retained, so this count is supporting evidence only. It does not qualify a GPU route. |
| Gridbook v0.9.0 rebase proof | **410 passed, 35 skipped** | Focused tests in the preserved isolated worktree `/home/rob/dq-runs/gridbook-4090-rebase.LKc9XC`; ancestry proves candidate `63e019f` is exactly one commit over v0.9.0 `d8cef3f`, the 5060/NVFP4 branch was not changed, and `git diff --check` passed. This is CPU structural/contract evidence, not an installed-wheel or physical-device qualification. |
| Learned-v2 repeat on GB10 | **pass**, exact four-table digest repeat; peak CUDA allocation 33,676,288 bytes | Physical NVIDIA GB10, compute capability 12.1, PyTorch 2.13.0+cu130, CUDA 13.0, immutable image `eugr/spark-vllm@sha256:58862b388e0fab05a5c9b673f21d1d7b41a1123953a2d9ace49aae6c79319869`. This proves same-build/same-device learned-v2 repeatability only; it is not Ada or serving evidence. |
| Explicit sm89 Gridbook preflight | **pass at `compile_only`** | Container received no GPU devices. It compiled explicit `-gencode=arch=compute_89,code=sm_89` SASS, found the required Gridbook extension symbols, and found vLLM's `dynamic_per_token_scaled_fp8_quant` and `cutlass_scaled_mm` ABI without executing them. The receipt expressly excludes device correctness, device performance, `torch_compile`, and `vllm_cudagraph`. |

### Durable GB10 repeat evidence

Log:
`/home/rob/dq-runs/20260824_fp8_cbl_learned_v2_repeat_gb10/run.log`

SHA-256:
`19a9461977af618dc1007da45f25c13db97da738b65aefbd813313c16972ef9a`

The logged test body is
`tests/test_cb_learned_v2.py::test_learned_v2_repeat_digest_is_exact_on_one_cuda_device`.
Both runs produced, in the same order:

```text
5cbc0ad55142877d9c54dda4d77322d9933a074773e0de8ae9391465542f6025
fb732642c34ef2ecf4145e8ebea2a0bcd2b13ad3b9d737c6f32533a578aa5f3f
fa55e12a4fdb7ee453bd42f66cabd2904baaae80b18ef3f4653f7559ddf33600
a1300615a5ffd19b4e1fb7528d1c99aa96214baacb8a300f42944e70501f1710
```

This is the repeat scope promised by learned-v2: one fixed software build and
device. Cross-architecture equality between GB10 and Ada is not required; the
shipped canonical FP16 table digest is the architecture-independent render
identity.

### Durable sm89 compile-only evidence

Successful receipt:
`/home/rob/dq-runs/20260824_gridbook_sm89_v10_compile_only/receipt.json`

Receipt SHA-256:
`55a3679654db97617846f6addf46f8fbf617e16afbaeeebeea4ad2f454159bcd`

Successful log:
`/home/rob/dq-runs/20260824_gridbook_sm89_v10_compile_only/run_driverlib.log`

Successful-log SHA-256:
`ef488acd398809965992441d9297146077fc1e92c4e2edfa7107f4495345583d`

The successful command used the same immutable image as the GB10 repeat, mounted
the Gridbook source read-only, supplied no `--gpus`, limited Ninja to one job,
and mounted only the host's read-only `libcuda.so.1` so vLLM's precompiled
registration library could load. The receipt says `device_executed: false`,
`qualification_ceiling: compile_only`, and `present_not_executed` for the two
vLLM operations.

The first attempt compiled the Gridbook CUDA source but could not load vLLM's
registration library because `libcuda.so.1` was absent. It is retained rather
than hidden:

- log: `/home/rob/dq-runs/20260824_gridbook_sm89_v10_compile_only/run.log`
- SHA-256: `ec8412edcf006dd24823a38defb8f8bbd288a8c7b068b66931d2fef9690a7166`

The retry added only the read-only driver library; it did not expose a GPU,
allocate a tensor, invoke an operator, query a device capability, or run vLLM
graph compilation.

## Deferred AMD FP8-CB serving constraints

AMD implementation remains out of scope for this campaign, but the follow-up
must retain this dated design constraint.  With Gridbook's four-way FP8 product
codebook, the byte-resident LUT is `8 * 2**(K/4)` bytes: K40 is 8 KiB, K44 is
16 KiB, and K48 is 32 KiB.  Materialising the LUT as BF16 doubles those figures.
RDNA 3.5 and RDNA4 expose 128 KiB LDS per WGP but at most 64 KiB to one
workgroup; large allocations also reduce workgroup residency and divergent
gathers can suffer LDS bank conflicts.

The provisional producer/serving policy for the later AMD work is therefore:

- RDNA 3.5: K4--K40 are the default candidate range; K44 requires an FP8-byte
  LDS table with conversion after lookup or a measured equivalent; K48 is
  non-default.
- RDNA4: K4--K44 are the default candidate range; K48 requires a separately
  tuned and measured lane.
- The portable artifact format continues to admit K48.  These are performant
  serving-profile limits, not wire-format restrictions, and must be replaced
  by shape-specific LDS/occupancy/bank-conflict measurements on the owned AMD
  systems before any AMD production default ships.

This is consistent with the repository's historical Strix result, which saw a
performance cliff above K40 when the BF16-materialised LUT stopped being
comfortably LDS-resident (`docs/ARCHITECTURE.md`, Strix historical snapshot).

## Evidence boundary and remaining release gates

Nothing above establishes that FP8-CB is correct, fast, graph-compatible, or
memory-safe on an RTX 4090. In particular, the GB10 repeat and sm89 SASS receipt
must not be transformed into `device_qualified` lane-v2 cells.

Release still requires all of the following on a physical device identified as
`NVIDIA GeForce RTX 4090`, compute capability 8.9:

1. An immutable, released Gridbook v10 runtime and reviewed PrismaQuant pin
   whose lane-v2 contract marks all twelve FP8-CB producer rungs
   device-qualified for both exact-sm89 dense decode and batch regimes,
   regardless of which legal subset this assignment selects.
2. A reviewed vLLM runtime pin proving an installation from the exact official
   upstream VCS commit through matching PEP 610 metadata and installed
   `RECORD` bytes; no such candidate pin is supplied yet.
3. A strict artifact containing only FP8-CB/`FP8_E4M3`/BF16 assignments, a BF16
   `lm_head`, and no NVFP4 family anywhere in its assignment, quantization
   config, tensor inventory, or delegated terminals.
4. A complete frozen publication tree no larger than 18,000,000,000 bytes.
5. Fresh eager and graph serves at TP=1, 32K model length, one validation
   request with `n=1`, scheduler `max_num_seqs=64`, 32,768 maximum batched
   tokens, and exactly 4 GiB of FP8 KV cache. The scheduler ceiling exists to
   admit the graph capture ladder and is not a 64-concurrent-context claim.
6. Correct deterministic generation, with the graph arm proving mode 3,
   explicit Inductor, `FULL_AND_PIECEWISE`, 7/7 PIECEWISE and 7/7 FULL
   completion over all seven capture sizes,
   `fullgraph=True`, `dynamic=False`, and no fallback.
7. The repository's independent served quality, numeric ship, and matched
   performance gates. The 24 GiB resident-memory and workspace envelope must be
   observed, not inferred from artifact and KV byte arithmetic.

Until those gates close, the correct release verdict is **candidate implemented;
physical RTX 4090 qualification pending**.

The SM89 cross-compile receipt remains `compile_only` evidence and cannot be
promoted by inference. It demonstrates neither the physical route, graph
replay, memory envelope, nor throughput on a 4090.

## Joint-candidate integration update — 17:45 EDT

This append-only update supersedes the earlier references to the standalone
v10 FP8 candidate for subsequent integration and release work.  The intended
joint Gridbook FP8 + NVFP4 candidate is now
`c052d703fce04b1e27d7f9d5945f67e7dfd841c8` (tree
`0f3d784d921bf831dc162b1dc54ef511c533e4af`), a linear descendant of the
reviewed FP8 commit `63e019f31b57c13b0fe9697441f5084eb194440a` and the
shipped v0.9.0 base.  It packages `gridbook.runtime-contract.v11`, contract
version 11, and `gridbook.lane-eligibility.v2`.  NVFP4 readers and producers
admit K1--K25.  FP8 producers admit exactly K4 through K48 in steps of four;
readers additionally retain the historical irregular high rungs.  The strict
RTX4090 campaign remains unchanged: it physically renders only FP8-CB
K4/K16/K48 plus `FP8_E4M3`, treats BF16 as the unrendered terminal, and admits
no NVFP4 format.

The exact immutable pair PrismaQuant
`cb6730cc9049f74ae865ce14c60e1db43a3d8d17` (tree
`9330d41aa6ab86e8ab5d280d751fbfb8374bb334`) plus Gridbook `c052d703`
passes a CUDA-hidden, read-only-root cross-repository gate: **697 passed, 1
explicitly GPU-only skip**.  The gate checked packaged v11 byte/load identity,
both reader/producer domains, execution parsing, artifact conformance,
attestation interoperability, runtime boundaries, strict RTX4090 burn,
policy, graph, census, export, fingerprint, and shipcard behavior, TP/EP
surfaces, and SM89/SM120 structural preflights.  Its resolver accepted all 24
SM89 dense FP8-CB routes at the `compile_only` construction ceiling and
correctly refused them under the device-qualified release policy.

Durable log:
`/home/rob/dq-runs/qwen38-27b-rtx4090-fp8cb-18gb-vonly-20260824/validation/candidate-pair-cb6730c-c052d703-cpu-crossrepo-v3.log`;
SHA-256 `fa4f1e940d8a5c9f8e9ce6065f927e1e3eef70e54acd91eb4b679588eefd4008`.
The joint Gridbook candidate also passed its clean pinned-vLLM structural gate
with **406 passed, 1 GPU-only skip**; log
`/home/rob/dq-runs/qwen38-27b-rtx4090-fp8cb-18gb-vonly-20260824/validation/gridbook-joint-c052d703-cpu-vllm-structural.log`,
SHA-256 `e329eec4366b9842e97e21c3c49218f5b69dccfd4e9c0db8c3c57c439a279025`.

No tag, wheel, or PrismaQuant runtime-pin advancement follows from these CPU
receipts.  Every new SM89 and SM120 lane cell remains `compile_only`.  For the
current joint candidate, release-gate item 1 above must therefore be read as
requiring an immutable released Gridbook **v11-or-later contract-compatible**
runtime whose exact SM89 cells have been promoted only by physical device,
eager, graph, numeric, and performance evidence; the historical v10 wording is
not permission to release the superseded standalone candidate.
