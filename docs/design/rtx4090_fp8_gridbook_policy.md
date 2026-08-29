# RTX 4090 FP8-CB Gridbook producer policy

This document defines the implementation boundary for the context-first dense
Qwen3.8-27B RTX 4090 campaign. It records policy and gates, not a claim that
the current Gridbook release is Ada-qualified.

## Format domains

FP8-CB now has two deliberately different domains:

| Domain | Rungs | Authority | Intended use |
|---|---|---|---|
| producer | K4, K8, K12, K16, K20, K24, K28, K32, K36, K40, K44, K48 | `FP8_PRODUCT_RUNGS` / `list_producer_formats` | new cost menus, assignments, bundles, exports |
| reader | low producer rungs plus every K28 through K48 | `FP8_ACCEPTED_RUNGS` / `list_formats` | load, inspect, and report historical artifacts |

K29, K43, and K47 therefore remain valid historical wire ids but are never
legal inputs to a new producer. Explicit-menu APIs fail closed instead of
silently rounding an off-law rung. The K%4 rule is shared by Ada and
Blackwell; hardware qualification is a separate execution-contract question.

Each rung retains the existing exact FP8-CB footprint: `k/8` index bits per
quantizable parameter plus one FP32 scale per output row and the codebook
sidecar. Tests cover every producer rung and representative reader-only rungs.

## Strict 4090 campaign

The `qwen38_rtx4090_fp8_cb` serving profile inherits the historical
`nvfp4_cb` serializer lane but narrows production to:

- FP8-CB K4..K48 in steps of four;
- delegated W8A8 dynamic FP8 (`FP8_E4M3`); and
- BF16 terminals.

NVFP4, NVFP4-CB, source-passthrough formats, and reader-only FP8-CB rungs are
refused at profile-menu, assignment, export, and final-manifest boundaries.
The strict artifact advertises Gridbook top-level `format: "fp8_cb"`; generic
historical exports retain `format: "nvfp4_cb"`.

The strict campaign is currently **lattice-only**:
`CB_CODEBOOK_SOURCE_SCOPE=none` and `CB_CODEBOOK_SOURCE=lattice`. Generic
learned-v2 machinery remains independently useful, but the strict wire
validator rejects a learned group until the artifact contract itself carries
and replays its raw promotion ledger, probe-imatrix identity, and complete
source closure. There is no guessed learned/lattice crossover. The launcher
also sets `CB_ACTIVATION_SCOPE=none`: FP8-CB still executes dynamic W8A8 E4M3,
but it carries no NVFP4 static-activation scale or activation-contract
metadata. No native or codebook NVFP4 may enter through a terminal, sidecar,
assignment, or manifest.

The whole published directory has an exact ceiling of 18,000,000,000 bytes.
This is a context-first ceiling, not a best-effort target. On a 24 GiB card it
leaves an envelope for a 4 GiB KV-cache budget and runtime workspace. The
memory claim applies only to the pinned dense architecture: hidden size 5120,
64 layers, intermediate size 17408, vocabulary 248320, 24 attention heads,
four KV heads, head dimension 256, untied embeddings, at least 32K positions,
and the exact three-linear/one-full attention schedule (48 linear-attention
and 16 full-attention layers). A nearby Qwen model does not inherit the budget.

The architecture gate is a census, not a rank/dtype heuristic. For the exact
released official wrapper it derives 1,199 source tensors and 615 Linears,
including the vision and MTP namespaces, and requires the assignment to name
every Linear exactly once with `lm_head` explicitly BF16. The finalized gate
then re-derives the source layout from `config.json`, reconciles
`tensor_formats` with every `config_groups` target and the exact `ignore` set,
compares every safetensors key/dtype/shape with the assignment-derived artifact
manifest, and checks the complete codebook-sidecar reference census. Missing,
extra, renamed, or differently typed tensors fail closed. Final provenance is
also value-gated: the render marker is exactly bool true, the shared
source-complete render identity binds the imatrix, every declared codebook
digest is replayed against its finalized FP16 sidecar tensor, the strict
per-tensor payload ledger recomputes its aggregate over the exact header key
set, the weight-content manifest uses its closed schema and exact container
set/sizes, and the producer Git identity agrees with an exact `{commit, dirty}`
shipcard build record whose `dirty` value is bool false.

Container and tensor value replay share one sequential scanner. It opens the
artifact directory and every shard with `O_NOFOLLOW`, caps and strictly parses
the safetensors header, then computes full-container and per-tensor SHA-256 in
one forward traversal while checking the descriptor and namespace stats did
not change. Resident export performs that one traversal. The streaming writer
already feeds the same emitted bytes to both digest scopes, so it carries a
process-local receipt bound to device/inode/size/mtime/ctime into finalization
and performs no post-write content reread. Receipt reuse never crosses an
artifact copy. For strict serving, the fresh independent traversal runs inside
the actual serving-container namespace as the final pre-vLLM preflight. The
launcher overlays every weight shard with an individual read-only bind mount,
so the verified inode cannot be retargeted before the separately exec'd vLLM
process opens it. A no-clobber receipt under `/run` binds the mounted directory
and shard stats; post-serve census and the host endpoint gate recheck those
same underlying bind-source stats and the finalized ledgers without rereading
payload bytes, and shipcard replay carries the same signed content identity.
Any in-place mutation changes the bound stat and refuses. This is deliberately
fail-closed rather than a claim of an
atomic cross-process handoff: a host writer could modify a mounted inode after
preflight and before vLLM opens it, but cannot produce acceptable evidence
because post-serve ctime/mtime/inode replay then fails. Eliminating that last
host-write TOCTOU would require vLLM to inherit and consume the verified file
descriptors (which its model loader does not support) or a second copy/read.
The codebook sidecar remains a separate small exact-value replay.

The operator build uses streamed AURA with one absolute checkpoint directory.
`PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE` is stored in that directory and is
revalidated by the source census, so a resumed AURA stream and the eventual
artifact cannot silently name different source bytes or tensor maps.
`CB_IMATRIX_SOURCE=probe` derives full-corpus `act_sq_sum / n_tokens_seen`
weights from the existing probe; the resulting value digest and calibration
identity are shared by cost renders, measured frontier KL, and final weighted
export rather than recomputed from the row-capped activation replay cache.

## Cross-repository qualification

PrismaQuant never imports or vendors Gridbook. A strict export must receive an
explicit Gridbook runtime contract with both:

1. `gridbook.runtime-contract.v11`, whose `formats[].rungs` are reader domains
   and whose `formats[].producer_rungs` exactly match producer domains; and
2. `gridbook.lane-eligibility.v2`, resolved by exact platform, family, rung,
   structure, and both `decode` and `batch` regimes.

The lane table is closed-world. An exact `sm_89` runtime is producer-legal only
when **all twelve** FP8-CB producer rungs—not merely the subset selected by one
artifact—resolve to clean `backed`, flag-free, `device_qualified` routes in
both decode and batch regimes. The strict artifact stamps that supplied v11
attestation directly beside its selected-rung/unit counts; it neither consults
the repository's historical generic serving pin nor admits generic non-native,
override, or fallback dispositions. Capability is never
treated as a minimum: an sm90 or sm121 cell does not imply sm89 support. The
initial structural Ada cells are `compile_only`, so the strict campaign is
deliberately closed until physical RTX 4090 evidence advances Gridbook's
immutable release contract. Contract v11 otherwise preserves Gridbook 0.9.0's
tensor-parallel and expert-parallel declarations and enforcement; the new
dense Ada rows neither widen nor disable TP/EP behavior.

The producer stamp binds the canonical JSON SHA-256 of the complete supplied
runtime contract. It does not invent a Gridbook commit, wheel hash, or release
identity; those belong to the immutable external qualification receipt.

## Graph and ship gates

Graph evidence is owned by PrismaQuant and is not transcribed into Gridbook's
lane table. Shipping requires a real RTX 4090 receipt for all of:

- `torch.compile` with Inductor and `fullgraph=True`;
- vLLM compilation mode 3;
- `FULL_AND_PIECEWISE` CUDA graphs; and
- capture sizes 1, 2, 4, 8, 16, 32, and 64.

The log must positively finish **7/7 PIECEWISE and 7/7 FULL** captures, not
merely print a final generic capture marker. Compiler, eager, graph-mode, or
partial-capture fallback is a refusal.

The launcher also requires an explicit vLLM runtime pin. Its closed JSON
identity names only `https://github.com/vllm-project/vllm.git`, one exact
40-hex Git commit, the installed version, and the SHA-256 of that installed
distribution's `RECORD`. The in-container collector requires PEP 610
`direct_url.json` to name the same official VCS commit, checks both it and the
fullgraph wrapper against `RECORD`, and binds those bytes into the manifest.
Host validation and frozen shipcard replay require that exact pin again. An
immutable image name, a package merely named `vllm`, a fork URL, an unpinned
wheel, or missing PEP 610 provenance cannot qualify the strict lane. No
candidate pin is supplied by the repository, so release remains closed until
the reviewed serving installation provides one.

The scheduler ceiling is therefore `max_num_seqs=64`: vLLM's FULL-decode
dispatcher only admits capture sizes at or below that ceiling. This does not
change the context-first validation workload, which is one live request with
`n=1` at a 32,768-token model limit and a fixed 4 GiB FP8 KV allocation. It is
not a claim that 64 concurrent 32K sequences fit in that allocation.

The endpoint/shipcard gate replays that receipt, maps its GPU name and compute
capability to the contract's exact `sm_89` platform, and compares the runtime
contract SHA. Compile-only emulation on another GPU can find structural bugs,
but it cannot produce the device-qualified receipt.

## GB10 validation-only producer

`qwen38_rtx4090_fp8_cb_validation_only` is a separate producer identity for
finding serializer, census, and artifact-layout defects before Ada hardware is
available. The launcher requires exactly one DGX Spark GB10 (compute capability
12.1), then runs the ordinary GPU-bound pipeline and existing resident-prefetch
and `ProductionWeightCache` mechanisms. It consumes exact backed, flag-free
Gridbook v11 `compile_only` SM89 dense cells for the complete twelve-rung
producer ladder. It does not weaken or call the production device-qualified
resolver.

The resulting artifact uses top-level `format: fp8_cb` and the exact strict
FP8-CB/`FP8_E4M3`/BF16 assignment, group, tensor-format, BF16 head/MTP, lattice,
no-activation, source-census, and finalized-census rules. It additionally
carries `artifact_disposition: UNRELEASABLE_VALIDATION_ONLY`,
`runtime_qualification_ceiling: compile_only`, and
`build_host: dgx_spark_gb10`. Those fields are not a provisional ship claim:
shipcard verification always reports a categorical failure, publication
refuses before either force-override path, and the physical RTX4090 validator
refuses before interpreting the compile-only contract. Only the dedicated
structural validator may accept this stamp, and it emits no serving evidence.

For a campaign that already has a completed allocator `layer_config.json`, use
`scripts/export_qwen38_rtx4090_fp8_cb_validation_only.sh`. The direct handoff
requires the original source, the exact column-weight pickle bound by the
assignment's source-complete `cb_render_identity`, the compile-only v11
contract, and a fresh output directory. It requires the allocator's exact
assignment-bound whole-artifact budget stamp at a positive value no greater
than 18,000,000,000 bytes and forbids namespace exclusions. It calls the
existing streaming exporter once with the selected assignment; it does not
rerun probe, cost, retained format-menu cache construction, frontier selection,
or any other stock pipeline stage. The finalized artifact passes only the
dedicated structural validator and retains every categorical release refusal
described above.

Strict publication is also non-forceable. The publisher re-derives strictness
from the canonical policy/profile, strict slot topology, and on-disk
quantization manifest, freezes the complete upload tree, replays the shipcard,
and reapplies the 18,000,000,000-byte ceiling including documentation and
evidence. An unfilled/invalid `rtx4090.fp8_cb` slot, a frozen replay failure, or
an oversized tree cannot be waived, stamped, or published with
`--force-unverified` or `--confirm-name`; generic force behavior for unrelated
artifacts is unchanged. The early mutable-tree shipcard check is payload-cheap:
it closes obvious slot/stat refusals before the artifact scan and does not read
weight payloads. For a strict artifact, the subsequent `O_NOFOLLOW` frozen-tree
scan feeds each root safetensors shard's whole-file, 8-MiB-block, and
per-tensor hashes from the same bytes. Frozen replay consumes that process-local
receipt directly against the frozen weight manifest, canonical tensor ledger,
and shard index; it neither reopens mutable source paths nor performs a second
path-based payload traversal. The existing post-upload held-descriptor digest
replay remains the mutation check across the Hub commit window.

End-to-end release remains gated on the actual 18 GB-or-smaller Qwen3.8-27B
artifact loading in unforked vLLM with Gridbook, completing fullgraph and both
CUDA-graph modes, generating correctly, and meeting the repository's served
quality and native-parity performance standards.
