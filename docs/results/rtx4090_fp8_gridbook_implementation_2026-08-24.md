# RTX 4090 FP8 Gridbook implementation evidence — 2026-08-24

## Status

The strict RTX 4090 FP8-CB path is implemented in the PrismaQuant and Gridbook
working trees, but it is **not release-qualified**. Gridbook v10 is neither a
released nor a PrismaQuant-pinned runtime, the strict Qwen3.8-27B artifact has
not been built and served under this contract, and no physical RTX 4090 gate
has run. The only sm89 evidence in this report is a no-device cross-compile
whose receipt explicitly caps its claim at `compile_only`.

The PrismaQuant changes were developed in a working tree based on
`e370e1d7183c683c6ddab4bef82dc1dc1cf06200`; the Gridbook changes were developed
in a separate working tree based on
`edfb3bca38aa1faae627dab82caef76e6b1743a2`. Those base revisions identify the
starting points, not integration commits or release pins.

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
| Build path | The operator launcher uses validated-surrogate selection with `PRODUCTION_CACHE=1`, `PRODUCTION_RECACHE=0`, resident prefetch required, and streamed BF16 AURA. One absolute AURA checkpoint directory owns `PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE`; source preflight revalidates that identity. `CB_IMATRIX_SOURCE=probe` supplies one full-corpus calibration/value identity to cost, frontier KL, and export. The official wrapper is accepted only after the nested dense text model and complete source census validate, then uses the established flattened text-only staging path. |
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
| Gridbook CPU proof selection | **539 passed, 36 skipped** | Agent-reported proof output. No standalone log or complete selector was retained, so this count is supporting evidence only. It does not qualify a GPU route. |
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
