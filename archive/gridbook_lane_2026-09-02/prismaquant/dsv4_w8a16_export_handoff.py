"""Fail-closed pre-export gate for the fixed DSv4-Flash W8A16 release.

This module does not launch an exporter and never writes an artifact.  It
turns the reviewed readmission publication and every immutable exporter input
into one machine-readable handoff receipt immediately before the GPU job is
started.  The release driver may consume stdout only after this function
returns successfully.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from prismaquant.allocator_candidates import (
    ROUTE_GRIDBOOK_FP8_SOURCE_W8A16,
    ROUTE_PENDING_PASSTHROUGH_FORMATS,
    SOURCE_PASSTHROUGH_CONTRACTS,
)
from prismaquant.anchored_cost import AURA_CURRENCY
from prismaquant.cb_anchored_cost import (
    CB_ANCHORED_COST_SCHEMA,
    CB_ARTIFACT_PUBLISH_SCHEMA,
)
from prismaquant.cost_stage_checkpoint import canonical_json_sha256
from prismaquant.dsv4_aura_cb_reprice import (
    DSV4_W8A16_APPROVED_BUDGET_BYTES,
    DSV4_TOTAL_UNITS,
    DSV4_W8A16_APPROVED_ASSIGNMENT_SHA256,
    DSV4_W8A16_APPROVED_CB_COL_WEIGHTS_SHA256,
    DSV4_W8A16_APPROVED_LAYER_CONFIG_SHA256,
    DSV4_W8A16_APPROVED_SELECTION,
    DSV4_W8A16_APPROVED_SELECTION_SHA256,
    DSV4_W8A16_READMISSION_SCHEMA,
)
from prismaquant.format_registry import get_format
from prismaquant.gridbook_runtime_pin import (
    GRIDBOOK_RUNTIME_CONTRACT_SCHEMA,
    GRIDBOOK_RUNTIME_RELEASE_VERSION,
    load_gridbook_runtime_pin,
    require_exact_gridbook_runtime_release,
    supports_source_fp8_block128_w8a16,
)
from prismaquant.layer_config import load_assignment
from prismaquant.nvfp4_cb_footprint import (
    assignment_serialization_sha256,
    cb_serialization_metadata_from_assignment_payload,
)


DSV4_W8A16_EXPORT_HANDOFF_SCHEMA = (
    "prismaquant.dsv4_w8a16.export_handoff.v2"
)
DSV4_W8A16_EXPORT_SOURCE_CLOSURE_SCHEMA = (
    "prismaquant.dsv4_w8a16.export_source_closure.v1"
)
_PUBLISH_MANIFEST = ".anchored_publish.json"
_PUBLISHED_FILES = frozenset({
    "layer_config.json",
    "selection.json",
    "pareto.knees.json",
    "cb_col_weights.pkl",
})
# This is a deliberate release boundary, not an import-graph checksum.  It
# closes the reviewed streaming exporter plus the code that defines its CB
# wire/accounting contract, DSpark physical namespace, DeepSeek-v4 profile and
# decoded-source semantics, source-complete render identity, completeness, and
# output transaction.  Unrelated pipeline/probe modules remain outside this
# one-purpose pre-export handoff.
#
# RE-FROZEN 2026-08-15 for four Qwen3.8-27B CB changes, each reviewed against
# THIS handoff rather than merely re-hashed:
#   export_nvfp4_cb_streaming.py -- ports the `quantized_embedding` declaration
#     from export_nvfp4_cb (65bf9aa).  Every added branch is guarded by a
#     non-empty `embedding_stock`, which is populated only from recipe units
#     named `*.embed_tokens`.  A DSv4 W8A16 recipe assigns none, so
#     `embedding_stock` is empty and each branch is inert on this lane:
#     `sidecar_stock -= set(embedding_stock)` subtracts nothing and
#     `(qname in sidecar_stock or qname in embedding_stock)` is unchanged.
#   artifact_completeness.py -- also resolves a config-group target written in
#     vLLM's module namespace (the delegated-target spelling) back to its
#     checkpoint unit.  The change only ADDS spellings that can claim a unit,
#     so it can turn a false failure into a pass and never the reverse, and the
#     DeepSeek-v4 spec declares no `recipe_to_vllm` rewrite at all — on this
#     lane the added spelling is the name the gate already tested.
#   cb_export_config.py -- adds the `quantized_embedding` declaration builder
#     and its wire-id table (683b605), plus comment-only text (82c0b30). The
#     builder is called only from the embedding branch above, and its wire
#     table admits NVFP4 alone; nothing on the W8A16 path reaches it.
#   production_weight_cache.py -- NVFP4A16 now takes NVFP4's production render
#     (28152ba), which changes rendered bytes only for units ASSIGNED
#     NVFP4A16, a format this lane does not use; and a new
#     `release_resident_tensors` method (1cb5e1c) that drops re-readable
#     disk-backed copies while keeping every key resolvable — additive, and it
#     cannot alter a rendered weight.
#
# The drift was introduced by this session's own commits and went unnoticed
# because the gate reports only the FIRST mismatching file: refreshing one
# digest simply advanced the error to the next. Enumerate the whole closure
# when re-freezing.
#
# RE-FROZEN 2026-08-15 (second time, one Qwen3.5/3.6 dense namespace fix),
# reviewed against THIS handoff rather than re-hashed:
#   model_profiles/base.py + registry.py -- a profile is now handed the
#     `model_type`/`architectures` the checkpoint declares (`declare_config`),
#     and `structure_spec()` specializes the spec's naming block when that spec
#     declares `naming_variants`. `specs/qwen3_5_dense.json` is the ONLY spec
#     that declares any -- asserted over EVERY file in specs/ by
#     tests/test_qwen3_5_text_only_namespace.py::
#     test_qwen3_5_dense_is_the_only_spec_with_naming_variants, so the claim
#     covers specs added later and not just a hand-picked few --
#     so on the DeepSeek-v4 lane `for_config` is not reached and the spec this
#     handoff's profile returns is EQUAL to the unspecialized file spec; the
#     declaration is two otherwise-unread attributes. Verified directly:
#     `lm_head`/`model.embed_tokens`/expert names derive exactly as before on
#     both the vLLM-internal and checkpoint sides.
#
# RE-FROZEN 2026-08-16 (third time). Whole closure enumerated per the note
# above; THREE files drifted, reviewed against THIS handoff rather than
# re-hashed:
#   artifact_completeness.py -- two changes. (1) 1ccdf58 widened the
#     enumerator from "`.weight` in an FP8 dtype" to also read
#     `.cb_qweight`/`.weight_packed` planes, added `quantized_embedding` as a
#     claiming mechanism, and bridged a fused checkpoint unit claimed by its
#     unfused halves. The widening can only ADD units to classify, i.e. it can
#     only turn a pass into a failure, and on this lane it adds none: a DSv4
#     W8A16 artifact ships `FP8_BLOCK_UE8M0_SOURCE` blocks as `.weight` +
#     `.scale` and carries no coded or packed plane. The embedding branch needs
#     a `quantized_embedding` key this lane never writes (same argument as the
#     first re-freeze's `embedding_stock`). The fused bridge is the only
#     pass-ward change, and it is unreachable here: this lane's units resolve
#     through their `source_passthrough` declaration several branches earlier,
#     and it fires only when EVERY member is separately claimed. (2) The DSpark
#     construction bridge below, which is gated on a non-null
#     `provenance.dspark_cb_sidecar`; a W8A16 or target artifact declares none,
#     so the resolver is None and `_unit_variants` is byte-identical. Pinned by
#     tests/test_artifact_completeness_namespaces.py::
#     test_without_a_sidecar_declaration_the_construction_bridge_is_inert, and
#     confirmed on real bytes: artifact-aura-cb-112p69 reports the same 100
#     declared passthrough / 25 verbatim / COMPLETE before and after.
#     (3) `per_expert_format_groups` is now recognized as a claiming mechanism
#     in the classifier, closing the same omission 1ccdf58 left for split
#     expert banks: those tensors were already owned in both directions by
#     `_validate_per_expert_format_groups`, so the classifier was reporting
#     them a second time as claimed by nothing. Undeclared split tensors still
#     fail through that validator, pinned by a negative control. This lane
#     ships no split bank -- it writes no `per_expert_format_groups` key at
#     all, so the claimed set is empty and the branch is unreachable.
#   nvfp4_cb_footprint.py -- `whole_artifact_budget_stamp` gained an optional
#     `excluded_source_prefixes` that emits its field only when non-empty, its
#     reader validates that field, and two exclusion helpers were added. This
#     handoff imports `assignment_serialization_sha256` and
#     `cb_serialization_metadata_from_assignment_payload` and neither was
#     touched; a caller that passes no exclusions gets a byte-identical stamp,
#     pinned by tests/test_cb_serialization_contract.py::
#     test_budget_stamp_is_byte_identical_without_exclusions.
#   export_nvfp4_cb_streaming.py -- `_validate_namespace_exclusions` now
#     cross-checks the exclusion set against the budget stamp that priced the
#     allocation. It returns early when there is no stamp, and on a lane that
#     excludes nothing the check compares two empty sets, so it can refuse only
#     an export whose exclusions contradict its own price.
#
# RE-FROZEN 2026-08-16 (fourth time). Whole closure enumerated per the note
# above; exactly ONE file drifted, and unlike the previous three re-freezes the
# honest review is NOT "inert on this lane" -- it changes this lane's verdict,
# so it is written out in full rather than waved through:
#   artifact_completeness.py -- `_fused_member_units` gained a second fusion
#     source for ROUTED expert units only. Its first source is
#     `profile.fused_sibling_leaf_mapping()`, i.e. vLLM's
#     `packed_modules_mapping`, which describes DENSE fusions; DeepseekV4
#     exposes no vLLM architecture class, so that mapping is `{}` and
#     `specs/deepseek_v4.json` declares no `fused_groups`. The bridge the third
#     re-freeze called "unreachable here" was in fact unreachable EVERYWHERE on
#     this architecture, which is why the 8 expert stacks 1ccdf58 recorded as
#     task #14 were still failing. The fallback consults
#     `profile.packed_expert_projection_names`, the declarative
#     `packed_experts.projection_splits` the exporter itself used to emit the
#     halves, and the same table Gridbook keeps as `_FUSED_FALLBACK` for the
#     identical reason.
#
#     WHAT THIS CHANGES HERE, stated plainly: on `artifact-aura-cb-112p69` the
#     verdict moves from 8 undeclared / NOT complete to 0 undeclared / COMPLETE
#     (259->267 cb_units; passthrough 184, verbatim 25, fp8_in_ignore 0,
#     missing_scale 0 all unchanged). That is task #14 closing, not a gate being
#     widened to admit a defect: a per-role LEARNED codebook fits one book per
#     `(layer, projection)` and a packed `gate_up_proj` target binds exactly one
#     `codebook_ref`, so a per-role layer CANNOT name the packed stack -- the
#     halves are the only spelling the ABI permits. Gridbook 0.8.5 resolved it
#     (as read at that release: `config.py:1487-1503` `_moe_target_keys`
#     accepts the half leaves, `:1401-1454` builds `codebook_ref_by_role`,
#     `moe.py:512-527` consumes it); 0.8.11 carried the same three mechanisms
#     forward at shifted line numbers (`config.py:1729`, `:1694`,
#     `moe.py:574`), and so does the currently pinned 0.9.1 (`config.py:2161`
#     `_moe_target_keys`, `:2126` `codebook_ref_by_role`, `moe.py:673`
#     consuming it), and covers it with its tests
#     (`test_routed_per_role_codebooks.py`). Correspondingly, 1ccdf58's message
#     calls the dual spelling a "real inconsistency" -- that framing is wrong
#     and is retracted here; lattice layers share one book and legally name the
#     packed stack, so both spellings coexist in one correct artifact.
#
#     The pass-ward reach is bounded on three sides: the parent must be a
#     routed-expert container (dotted-boundary anchored, so `experts2` never
#     matches `experts`), the leaf must decompose to MORE than itself, and the
#     call site's EVERY-member rule is untouched, so a half-claimed stack still
#     fails -- Gridbook refuses that same partial. A W8A16 `.weight` + `.scale`
#     block still resolves through its `source_passthrough` declaration several
#     branches earlier and never reaches this code. Pinned by
#     tests/test_artifact_completeness_routed_per_role.py (6 tests: both-halves
#     claims, one-half still fails, packed spelling still works, the dense
#     fusion does NOT get the routed fallback, the vLLM mapping still covers
#     dense, and the `experts2` boundary).
#
# RE-FROZEN 2026-08-16 (fifth time). Whole closure enumerated per the note
# above; THREE files drifted, all for one fix, and the honest review is that
# this lane's verdict is UNCHANGED -- the drift is confined to the streamed
# FORWARD path, which no exporter calls:
#   layer_streaming.py -- `_compute_position_embeddings` gained an optional
#     `profile` argument and now re-keys a multi-rope dict from ROPE AXIS to
#     ATTENTION LAYER TYPE, and `_call_layer` lost its silent
#     `position_embeddings["main"]` fallback (an unresolved layer type raises).
#     DSv4-Flash's rotary is keyed `("main","compress")` while its layers
#     report `sliding_attention`/`compressed_sparse_attention`/
#     `heavily_compressed_attention`, so the lookup missed EVERY layer and the
#     fallback rotated 41 of 46 layers on base 10000 with YaRN off instead of
#     160000 with YaRN. That is the defect behind the perplexity-262 BF16
#     teacher, and it is a reintroduction of the bug PATCH 06 had already
#     fixed inside the vendored forward (modeling_deepseek_v4.py:1514-1521).
#   model_profiles/base.py -- new `rope_axis_for_layer_type` hook, default
#     None, which is exactly the pre-change behaviour for every other arch.
#   model_profiles/deepseek_v4.py -- overrides it by DELEGATING to
#     `DeepseekV4RotaryEmbedding.rope_axis_for_layer_type`, so the mapping has
#     one definition that `DeepseekV4Model.forward` resolves through too.
#
#     WHAT THIS CHANGES HERE, stated plainly: nothing. The two touched
#     functions have exactly five callers -- `cost_streaming.py:137`,
#     `incremental_probe.py:1542`/`:2706`, `sensitivity_probe.py:3219`/`:3274`
#     -- which are the teacher, the Fisher probe and the sensitivity probe.
#     No export path reaches them, the `LayerCache`/residency machinery this
#     handoff does depend on is untouched, and no already-written byte moves.
#     What DOES change is every FUTURE probe/cost pass on DSv4-Flash, whose
#     forward was previously wrong on 41 of 46 layers; the allocation behind
#     the current artifacts was produced through the defective path, and
#     whether to re-probe is being decided on gold KL against a valid teacher
#     rather than assumed here. Pinned by tests/test_multilayer_rope_forward.py
#     (6 new tests: the re-key, compressed layers receiving `compress` rope,
#     Gemma-style passthrough, a profile returning None, the removed fallback
#     now raising, and a profile naming an axis the rotary lacks) and
#     test_deepseek_v4_profile.py::test_rope_axis_mapping_matches_the_vendored_definition,
#     which asserts the model forward still resolves through the shared
#     definition so the two cannot drift apart again.
# RE-FROZEN 2026-08-18 (fourth time, the merge/proven-rescues line), reviewed
# against THIS handoff rather than re-hashed:
#   nvfp4_cb_formats.py + nvfp4_cb_footprint.py -- the signed CB family
#     (S13..S16, mode="signed") is deleted (c2c72a9). Lane-inert twice over:
#     no allocation in any campaign ever assigned a signed rung (the family
#     lost 78.48% of matched weight-MSE comparisons and was research-only),
#     and the W8A16 lane exports the FP8 block-source passthrough, which
#     never touches a CB codec. The footprint change removes the signed
#     branch of lattice_codebook_content_sha256 plus one docstring word;
#     every surviving branch's bytes are unchanged.
#   artifact_completeness.py -- three checker-read fixes (5d75fc4, 1ccdf58,
#     fcda875): the completeness gate learns to READ delegated-target
#     namespaces, per-expert split-format group tokens, and the DSpark
#     sidecar's physical->construction bijection (the fifth namespace,
#     resolved from the artifact's own published mapping, never inferred).
#     Post-export verifier only: it classifies claims over already-written
#     bytes and renders nothing; every change widens what a correctly
#     declared artifact can prove, and undeclared tensors still fail through
#     the same refusal paths.
#
# MERGE-FROZEN 2026-08-18: the proven-rescues and DSv4-release lines merged.
# Both lines above label themselves "third time" -- they were written in
# parallel on sibling branches that shared the completeness commits; both
# records are kept verbatim. The digests below are recomputed from the MERGED
# tree: footprint carries both lines' changes (signed-branch removal +
# excluded_source_prefixes), completeness carries both fifth-namespace
# mechanisms (the sidecar alias map AND the dspark-threaded _unit_variants
# bridge -- redundant claim paths, both fail-closed; unification is a
# candidate follow-up, not a correctness need). Per-file lane-inertness
# arguments are exactly the union of the two records above.
#
# RE-FROZEN 2026-08-21 (the CB lane joins the shard standard), reviewed
# against THIS handoff rather than re-hashed:
#   export_nvfp4_cb_streaming.py + nvfp4_cb_footprint.py -- the CB lane now
#     publishes ~1 GiB safetensors shards by default (`shard_bytes`,
#     `EXPORT_SHARD_BYTES`), which is the standing packaging default every
#     other lane already ships. The single-container layout it replaces is a
#     MEASURED user-hit defect, not a preference: the published 87 GB
#     `model.safetensors` stalls the default HF loader on a 128 GB
#     unified-memory GB10 and the reporter resharded it by hand
#     (RobTand/gridbook#47 setup notes). The footprint change is the matching
#     inventory rule -- it used to refuse any `model.safetensors.index.json`
#     and now derives from the published container set whether an index is
#     required, forbidden, or unrecognisable, failing closed on all three.
#
#     WHAT THIS CHANGES HERE, stated plainly, because this one is NOT inert:
#     a W8A16 export from this tree publishes ~90 shards plus an index rather
#     than one ~92 GB container, so its `model_sha` (a filename->size map,
#     shipcard.py:270-279) differs from what a pre-change run would have
#     produced, and the added index/per-shard-header bytes count against
#     `whole_artifact_budget_bytes` (kilobytes against 92 GB, but real and
#     measured by the same inventory). That is accepted deliberately: this
#     release is UNSHIPPED -- the handoff is a pre-export gate -- so no
#     published artifact's identity moves, and reproduction of the artifacts
#     that ARE published happens in era worktrees at their pinned commits,
#     where the pre-sharding exporter still lives, so replay identity there
#     is untouched. `--shard-bytes` at or above the finished artifact
#     reproduces the single-container layout if a specific release wants it;
#     there is deliberately no zero sentinel, because the native lane has
#     none. Recognition across a reshard is carried by the new layout-
#     INVARIANT `provenance.tensor_payload_identity`
#     (`shard_layout.tensor_payload_identity`), stamped by both CB exporters
#     from digests taken in the pass that already hashes the bytes. Pinned by
#     tests/test_shard_layout.py (12) and tests/test_cb_lane_sharding.py (21),
#     which include a cross-exporter test showing the two exporters agree on
#     the payload identity while their shard COUNTS differ.
# RE-FROZEN 2026-08-21 (second this date: the read-bytes ledger + discovery
# walker merges), reviewed against THIS handoff rather than re-hashed:
#   cb_export_config.py -- extracts the codebook sidecar name literal into
#     `CODEBOOK_TENSOR_PREFIX` and uses it in the same f-string (5b03e3e), so
#     a consumer (the read-traffic ledger) reads the producer's own spelling.
#     Serialized names are byte-identical; plus one `__all__` entry.
#   model_profiles/base.py + model_profiles/deepseek_v4.py -- the discovery
#     walker's claim rules (f5ce761): `walk_claim_rules()` on the profile and
#     five prepended DSv4 pins (routers, mHC mixers, hyper head,
#     compressor/indexer). Pure additions consumed only by `model_walk`; no
#     existing export or naming path is touched.
# RE-FROZEN 2026-08-21 (third this date: campaign rule R1 merged), reviewed
# against THIS handoff rather than re-hashed:
#   export_nvfp4_cb_streaming.py -- routed learned books are keyed per
#     (layer, stack, rung) when the bundle records that keying, emitting ONE
#     codebook per fused weight, and a fused weight whose scheme would name
#     more than one book fails closed unless --allow-per-role-books stamps the
#     shipcard. The W8A16 lane exports FP8 block-source passthrough and
#     assigns no CB rung, so neither branch is reachable on this lane; the
#     gate's predicate is structural (distinct refs a producer writes).
#   artifact_completeness.py -- learns to claim the pooled single-book routed
#     spelling alongside the per-role one; claim-widening only, and no CB
#     claims exist on this lane.
# RE-FROZEN 2026-08-21 (fourth this date: the R3 route-status merge), reviewed
# against THIS handoff rather than re-hashed:
#   export_nvfp4_cb_streaming.py -- the CB route-status gate (48618a6) runs
#     before any byte is written, resolving each unit's structural facts
#     against the pinned runtime's eligibility attestation; with the 0.8.10
#     attestation ABSENT every unit reports `unattested` and nothing refuses
#     without `--allow-unbacked-route`/`--non-native-target` being needed. The
#     W8A16 lane assigns no CB rung, so the gate's per-unit loop sees no CB
#     units on this lane; merge union with the R1 split-book gate reviewed
#     line-by-line (both are additive parameters + calls at the same anchors).
# RE-FROZEN 2026-08-24 (K1..K25 NVFP4 public producer scaffolding), after enumerating
# the whole closure and reviewing reachability against THIS handoff:
#   cb_export_config.py + nvfp4_cb_footprint.py -- the authoritative NVFP4
#     product domain and exact codebook-sidecar accounting now include K1..K25.
#     This handoff assigns only block-FP8 W8A16 source passthrough, no CB rung,
#     so neither a CB scheme nor its sidecar/accounting branch is reached.
#   nvfp4_cb_formats.py -- adds the digest-pinned nested d4 tables and uint32
#     K32 direct research codec surface while preserving every historical
#     lattice tensor hash. K26..K32 have no registry/export identity.
#     W8A16 copies its checkpoint element/scale planes verbatim and never calls
#     the CB field quantizer, lattice resolver, or bit packer.
#   export_nvfp4_cb_streaming.py -- updates only the strict external Gridbook
#     contract help text from v10 to v11. It cannot change exported bytes.
# RE-FROZEN 2026-08-24 (producer assignment preflight), after enumerating the
# whole closure and reviewing reachability against THIS handoff:
#   export_nvfp4_cb_streaming.py -- adds the shared assignment parser as an
#     outer decorator, before the transactional output wrapper. A valid W8A16
#     assignment is parsed once more and then follows the byte-identical export
#     body; an invalid/research-only CB spelling now refuses before creating a
#     destination or preserved `.tmp-*` tree. The decorator neither rewrites
#     the assignment nor touches tensor data, source discovery, or W8A16
#     passthrough emission.
# RE-FROZEN 2026-08-24 (strict compiled CB scoring from the refreshed RTX4090
# parent), after reviewing the only changed closure file against THIS legacy
# handoff:
#   nvfp4_cb_formats.py -- imports the closed compiled-helper contract and
#     routes CB VQ/scale-scoring reductions through it when strict campaign
#     compilation is requested. The W8A16 handoff copies published source-FP8
#     element/scale planes verbatim and never enters those CB scoring helpers;
#     no wire, tensor-name, source-discovery, or passthrough branch changed.
#     Keeping this frozen reproduction gate green is legacy compatibility, not
#     maintained target eligibility: hardware-scoped production profiles must
#     admit W8A16 separately, and the SM120 profile explicitly denies it.
#
# RE-FROZEN 2026-08-28 for three `93d340a` changes, each reviewed against THIS
# handoff rather than merely re-hashed.  All three are inert on the DSv4 W8A16
# lane; DeepSeek-v4 overrides none of the new accessors and inherits their
# fail-closed defaults.
#   export_nvfp4_cb_streaming.py -- adds the PrismaSnap lane refusals
#     (`@refuse_prismasnap_lane_before_output(lane="Gridbook/codebook")` and
#     the `main()` precheck).  Both fire only when the source directory carries
#     PrismaSnap provenance.  A DSv4 W8A16 source is never Snap-prepared
#     (PrismaSnap refuses non-BF16 sources outright), so both are inert here
#     and the emitted bytes are unchanged.
#   model_profiles/base.py -- adds `_declared_model_path` private intake plus
#     two accessors, `rms_norm_parameter_offset()` and
#     `prismasnap_moe_layer_contract()`, each defaulting to `None`.  `None` is
#     the fail-closed value: an offline source transform refuses rather than
#     inferring a gamma encoding or an MoE graph.  DeepSeek-v4 declares
#     neither, and this lane performs no offline transform.
#   model_profiles/registry.py -- attaches the resolved checkpoint root to the
#     profile as private intake evidence.  The sole consumer is
#     `qwen3_5.py:155`; DeepSeek-v4 never reads it.
#
# Every digest above is a HEAD blob, so the closure is exact at HEAD.  FOUR of
# the fifteen additionally differ in the working tree and are deliberately NOT
# stamped to their worktree bytes; the gate hashes the worktree, so it refuses
# until each lands and is reviewed:
#   base.py, registry.py -- re-stamped ABOVE for `93d340a` only.  The worktree
#     has since changed them again (base.py: `concat_merge_groups`,
#     `runtime_loads_source_fp8`, `requires_multimodal_skeleton`; registry.py:
#     two profile registrations).  Those newer edits are NOT covered by the
#     review recorded above and need their own before the next stamp.
#   layer_streaming.py, production_weight_cache.py -- unchanged since their
#     last freeze at HEAD; the worktree edits need real review rather than a
#     re-hash, because DSv4 W8A16 executes both paths: the default-ON threaded
#     gather in the streamed read, and `fill_packed_expert_cache_entries` for
#     packed experts.
# Sequencing: because these four span several commits, the re-stamp covering
# them belongs in the LAST commit that touches any of them, or the gate is red
# at every intermediate commit.
#
# RE-FROZEN 2026-08-29 for five changes -- the four the 2026-08-28 note left
# deliberately unstamped, plus the export file's post-merge delta -- each
# reviewed against THIS handoff rather than merely re-hashed.  This section
# closes that note's open item: every digest below is again a HEAD blob on a
# clean worktree.  For each file the reviewed delta is exactly
# `git diff 93d340a..HEAD -- <file>`, because `93d340a` is the blob every
# previous stamp named.  This commit touches no other closure file, so it IS
# the last commit touching any of them and the sequencing rule holds.
#   export_nvfp4_cb_streaming.py -- the origin/main side of merge `24366aa`,
#     i.e. PR #86 (`09bc72a`): three hunks, all written for a WRAPPED-MoE
#     (Qwen3.5-VLM) source whose skeleton speaks a live namespace.
#     (1) Expert-group keys in that live namespace are rekeyed to the recipe
#     spelling.  DSv4's per-expert regex
#     `^model[.]layers[.]N[.]mlp[.]experts[.]i[.](gate|up|down)_proj$` matches
#     ONLY the live-bridge spelling, so `_plan_expert_stacks` already keys
#     DSv4 groups by exactly that recipe prefix; `_prefix in (_recipe, _ck)`
#     holds and nothing is rekeyed.  The recipe->checkpoint map the hunk
#     builds is injective on this lane (`layers.N.ffn.experts` <-
#     `model.layers.N.mlp.experts`, N preserved), so neither ambiguity
#     `raise` can fire.
#     (2) A packed-stack export name is now taken from that map.  Verified
#     directly against `DeepseekV4Profile`: for both stacks the new
#     `f"{ckpt_prefix}.{leaf}"` and the old `_export_base_name(...,
#     assume_resolvable=True)` return the SAME string
#     (`layers.3.ffn.experts.gate_up_proj`, `...down_proj`).
#     (3) `_delegated_target_name` rewrites only names beginning
#     `language_model.`; DSv4's `to_vllm_internal_name` never emits that
#     prefix, and the hunk deliberately leaves the bare-`layers.`
#     DSv4-class case alone.  No emitted tensor name, config target or byte
#     changes on this lane.
#   model_profiles/base.py -- `b35ed53` adds three accessors:
#     `concat_merge_groups()` (empty unless the spec declares
#     `concat_merges`), `runtime_loads_source_fp8()` (False) and
#     `requires_multimodal_skeleton()` (False).  Each default is the
#     fail-closed value: no concat bridge, no runtime-side FP8 dequant
#     carve-out, no multimodal skeleton.  `Glm5NextProfile` is the ONLY
#     override of any of the three; `DeepseekV4Profile` subclasses
#     `ModelProfile` directly and all three resolve to the base
#     implementation (verified by attribute lookup on the MRO), and
#     `specs/glm5_next.json` is the only spec in the tree declaring
#     `concat_merges` -- so `concat_merge_groups()` returns `()` here and
#     every consumer (`layer_streaming._build_concat_merger`,
#     `export_native_compressed`) is a no-op on this lane.
#   model_profiles/registry.py -- `b35ed53` registers `Qwen4ExpProfile`
#     (priority 200) and `Glm5NextProfile` (priority 210), both AFTER
#     DeepSeek-v4's 170, and each `matches()` claims only `qwen4_exp*` /
#     `glm5_next*` model types and `Qwen4Exp*` / `Glm5Next*` architectures --
#     disjoint from `deepseek_v4`, so detection is unchanged.  The same
#     commit makes `detect_profile` refuse a `model_path` that is not an
#     existing directory.  That can only turn an absent-path run into a loud
#     refusal, and an absent path was never a valid DSv4 export (it would
#     have fallen through to `DefaultProfile` and mis-named every tensor).
#   layer_streaming.py -- `0c87d8d`.  This lane DOES execute the streamed
#     read, so each piece is argued, not dismissed as unreached.
#     * Threaded intra-layer gather (default ON).  The job list is built in
#       `by_shard` order and `_split_pairs` cuts each shard's pairs into
#       CONTIGUOUS chunks, and `out.update(fut.result())` consumes futures in
#       that same order -- so the assembled dict has identical contents AND
#       identical key order to the serial loop.  Each worker opens its own
#       `safe_open` handle and applies the same dtype cast and contiguity
#       fix; `.result()` re-raises, and a new count check refuses a partially
#       gathered layer rather than installing it.
#       `PRISMAQUANT_LAYER_READ_THREADS=1` restores the byte-identical serial
#       read.  Only the order pages fault in changes; no tensor value can.
#     * Post-gather view compaction: clones any tensor whose storage exceeds
#       2x its own bytes.  `detach().clone().contiguous()` preserves dtype,
#       shape and every element, and it runs AFTER the last in-place step
#       (the batched FP8 dequant, then the expert packer), so nothing
#       downstream loses a write-through it relied on.  Resident bytes
#       change; read bytes do not.
#     * `LayerCache` pressure-eviction rework (`_drop`, the new prefetch-pin
#       phase, the `_pinned_until_read` discard and `evicted_pinned`
#       counter).  It decides WHICH cached layers are dropped under host
#       memory pressure; a dropped layer is re-read from the same shards.
#       This moves cache hit/miss/eviction telemetry and wall-clock, never a
#       weight.
#     * `has_dsa` in `_compute_attention_mask` is doubly inert.
#       `deepseek_sparse_attention` is transformers' `glm5_next` layer type;
#       DSv4-Flash's own `DEEPSEEK_V4_LAYER_TYPES` is `{sliding_attention,
#       compressed_sparse_attention, heavily_compressed_attention}`, so
#       `has_dsa` is False -- and `sliding_attention` already made
#       `has_sliding` True, so the guarded early return was not taken before
#       the change either.  The added `masks` entry is unreachable here.
#   production_weight_cache.py -- `f7970e7`.  This lane DOES call
#     `fill_packed_expert_cache_entries`, so likewise argued.
#     * Packed experts now carry their own render-score and render-gate
#       records (`_packed_expert_render_score_record` /
#       `_packed_expert_render_gate_record`, summing the dense path's own
#       `_render_score_record` over experts).  Scoring is READ-ONLY: every
#       access is `detach().to(...)`, `.t()`, `.pow(2).mean()`; no in-place
#       write reaches `packed_param` or `rendered`, so the tensor stored is
#       the tensor that was rendered.  All four helpers it leans on
#       (`_render_score_record`, `_render_score_record_key`,
#       `_write_render_score_sidecar`, `_summarize_render_gate_records`)
#       already existed at the previous freeze, and the CB-pair admission's
#       `render_score=` kwarg was already the dense path's.
#     * `_needs_work` now also counts a missing render score.  On a resumed
#       build that runs an activation capture it previously skipped, but the
#       resume branch still `continue`s WITHOUT re-rendering: it scores the
#       shard bytes already on disk (a `torch.load(...,
#       weights_only=True)` read).  `activation_max_abs` keeps its "first
#       calibrated scale wins" `is None` guard, so the scale the shipped
#       rung calibrated cannot be clobbered by the extra pass either.
#     * `_finalize_packed_expert_cache_metadata` recomputes
#       `requested_entries` and the `render_scores` / `render_gates` /
#       `packed_expert_coverage` scopes from the cache.  It rewrites cache
#       METADATA only, and it can turn a stale-counter refusal of the exact
#       union into a pass, never the reverse -- the same direction the
#       2026-08-15 `artifact_completeness.py` bullet accepted.  Pruning is
#       bounded to `all_packed_fullnames`, so dense and MTP records are
#       outside its reach by construction.
#     * Device probe: the first NON-meta parameter instead of
#       `next(model.parameters())`.  Identical whenever the first parameter
#       is non-meta, which is every case a DSv4 export ever completed; where
#       it differs the old value was a `meta` device, which no successful
#       render used.
#     * `del src, overrides, render_acts, eval_acts, eval_gw` (and `w_rtn`
#       when `E > 0`) at the end of the batched branch.  Every one of those
#       names is dead at that point -- verified by scanning the remainder of
#       the function for a read -- and each is reassigned at the top of the
#       next iteration, so a mistake would be a loud `NameError`, never a
#       silent byte.  It frees GPU transients; `rendered` and `packed_param`,
#       the two tensors the score and the store consume, survive.
# RE-FROZEN 2026-09-02 (the R5 discovery-walker export gate, #99), reviewed
# against THIS handoff rather than re-hashed:
#   model_profiles/base.py -- `walk_claim_rules()` appends two ClaimRules
#     between the `ndim<=1` exclude and the final Linear-decide rule (a
#     router-class pin and an `*Experts` packed-stack decide), and renumbers
#     the docstring 9->11.  Verified by per-definition AST comparison with
#     docstrings elided (origin/main -> this branch): 79 definitions -> 79,
#     ZERO added, ZERO removed, and exactly ONE body changed --
#     `ModelProfile.walk_claim_rules`.  Nothing else in the file moved.
#     Why this lane cannot observe it: `walk_claim_rules()` is read at
#     exactly four sites -- `model_walk.py` (the walker and its CLI) and the
#     three profiles that extend it (`deepseek_v4.py`, `glm5_next.py`,
#     `qwen4_exp.py`).  A grep over this closure finds NO exporter,
#     completeness, decode-source, footprint, output-safety or namespace
#     consumer of the method.
#     On the DSv4 walk itself the two new rules are strictly shadowed or
#     equivalent, so this lane's claims are byte-identical:
#     * `DeepseekV4Profile.walk_claim_rules` returns `rules + super()...`,
#       i.e. it PREPENDS its own five pins, and `apply_claim_rules`
#       (`model_walk.py:675-688`) takes the FIRST match and `break`s.  Both
#       DSv4 routers (`DeepseekV4TopKRouter`, `DeepseekV4HashRouter`) are
#       pinned by those prepended rules, so they keep their original
#       disposition AND reason; the new base router pin never reaches them.
#     * The packed-stack rule matches `node.kind == "parameter"` on an owner
#       class containing "expert", and yields `decide` -- the same
#       disposition the final Linear-decide rule already gave those weights.
#       It changes `Claim.reason`/`rule_index`, never a disposition.
#     Off this lane the rules do two different things, and only one of them
#     is pure coverage.  On the R5 sweep's six/seven unclaimed profiles the
#     router rule claims nodes that previously had NO claim at all.  On
#     gemma4 it is a DISPOSITION CHANGE, and that is the point of it:
#     `Gemma4TextRouter` is a name-excluded Linear, `Gemma4Profile` does not
#     override `walk_claim_rules`, so the final Linear-decide rule used to
#     claim it `decide` while nothing ever priced it -- the wrong polarity
#     inside the claim table.  It becomes `pin`, like every other router.
#     That flip is real for the ALLOCATOR on gemma4 and inert for THIS lane,
#     which never instantiates a gemma4 profile.
#   The other 14 files in the closure are unchanged by #99: the gate names
#   exactly `model_profiles/base.py` and nothing else, and the 24 remaining
#   tests in `tests/test_dsv4_w8a16_export_handoff.py` pass unchanged.
_FROZEN_EXPORT_SOURCE_SHA256 = {
    "prismaquant/export_nvfp4_cb_streaming.py": (
        "dfffc634a7275e76a4c4b3bd0299e8b0775673dca23f6f5c56ca31f8b748b8a5"
    ),
    "prismaquant/cb_export_config.py": (
        "3aa767bba9e689d50234730846a1671088ec0b16278d18aa6fa2693815294412"
    ),
    "prismaquant/nvfp4_cb_formats.py": (
        "9f886165d4495f8e93615ac3804b41d87a69c8c4526833c196817366147d23d1"
    ),
    "prismaquant/dspark_source_metadata.py": (
        "94fac4b16922f381cffe989d7b9b1d00f211bb93d9479dfde30eb0c02ef167f7"
    ),
    "prismaquant/model_profiles/__init__.py": (
        "fb20303ed1b017a5a7f3a035d5ef43880822d775e252c28a08f32a67f8104c95"
    ),
    "prismaquant/model_profiles/base.py": (
        "7cff3a4af253777d831094838a35ec56b9cd5e4c1022654431449163be8a848e"
    ),
    "prismaquant/model_profiles/registry.py": (
        "5da03be05dafd7e804be9588854bfeabed2aec29b76ea2f6c6cbef2c6067188d"
    ),
    "prismaquant/model_profiles/deepseek_v4.py": (
        "6368f5657fbfb3b77a886e9bc0c589885d9240c49fdb77635d4bf2a74164b6f6"
    ),
    "prismaquant/model_profiles/specs/deepseek_v4.json": (
        "b8f3b22c16484a6859494d96ff052e5c5229c9a7c3afb7ae829e9cf5e26ecbf4"
    ),
    "prismaquant/cb_source_decode.py": (
        "d9a06483d008bf2361b0522bc258ab291db870d1c2432f9d4cd8d7a8cbacefbe"
    ),
    "prismaquant/layer_streaming.py": (
        "e2d947fe9ba98c612e13a9abe66dbb70aaadde83b0b0db394d100cbff82378c1"
    ),
    "prismaquant/production_weight_cache.py": (
        "69e29daf21ec8fcfb6ee86375dc6a9792750f38e32337c931b26be64bb0c9fc7"
    ),
    "prismaquant/nvfp4_cb_footprint.py": (
        "96bc38a7ab18c6d2401ed2b66141eef9809409c78468f8ceb16c0891b9701547"
    ),
    "prismaquant/artifact_completeness.py": (
        "7f0c6c74733c2503b1e9607383264479007f49ae04e41700327bf0e97ab59767"
    ),
    "prismaquant/export_output_safety.py": (
        "4af0a9d891313f1d9d031955e431e1e84c1ba0e11a9ce2605ea92de3bc3703b5"
    ),
}


class W8A16ExportHandoffError(RuntimeError):
    """The exact reviewed DSv4 W8A16 export handoff is not intact."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _real_file(path: Path, *, where: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise W8A16ExportHandoffError(f"{where} is not a regular file: {path}")
    return path


def _json_object(path: Path, *, where: str) -> dict[str, Any]:
    _real_file(path, where=where)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise W8A16ExportHandoffError(f"{where} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise W8A16ExportHandoffError(f"{where} is not a JSON object: {path}")
    return value


def _verify_publication(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    if root.is_symlink() or not root.is_dir():
        raise W8A16ExportHandoffError(
            f"readmission publication is not a real directory: {root}"
        )
    observed_names = {path.name for path in root.iterdir()}
    expected_names = _PUBLISHED_FILES | {_PUBLISH_MANIFEST}
    if observed_names != expected_names:
        raise W8A16ExportHandoffError(
            "readmission publication file set differs: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    manifest = _json_object(
        root / _PUBLISH_MANIFEST, where="readmission publication manifest"
    )
    identity = manifest.get("identity")
    outputs = manifest.get("outputs")
    if not isinstance(identity, Mapping) or not isinstance(outputs, Mapping):
        raise W8A16ExportHandoffError(
            "readmission publication lacks identity/output mappings"
        )
    try:
        identity_sha256 = canonical_json_sha256(
            identity, where="DSv4 W8A16 export publication identity"
        )
    except (TypeError, ValueError) as exc:
        raise W8A16ExportHandoffError(
            "readmission publication identity is non-canonical"
        ) from exc
    if (
        manifest.get("schema") != CB_ARTIFACT_PUBLISH_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("identity_sha256") != identity_sha256
        or identity.get("schema") != CB_ARTIFACT_PUBLISH_SCHEMA
        or outputs != identity.get("outputs")
        or set(map(str, outputs)) != _PUBLISHED_FILES
    ):
        raise W8A16ExportHandoffError(
            "readmission publication is incomplete, unbound, or has the "
            "wrong output set"
        )
    observed: dict[str, str] = {}
    for name in sorted(_PUBLISHED_FILES):
        descriptor = outputs.get(name)
        path = _real_file(root / name, where=f"published {name}")
        if not isinstance(descriptor, Mapping):
            raise W8A16ExportHandoffError(
                f"published {name} has no checksum descriptor"
            )
        digest = _sha256(path)
        actual = {"size_bytes": path.stat().st_size, "sha256": digest}
        if descriptor != actual:
            raise W8A16ExportHandoffError(
                f"published {name} differs from its atomic manifest"
            )
        observed[name] = digest
    return manifest, observed


def _selection_contract(selection: Mapping[str, object]) -> dict[str, object]:
    whole = selection.get("whole_artifact_budget")
    if not isinstance(whole, Mapping):
        raise W8A16ExportHandoffError(
            "readmitted selection lacks whole-artifact accounting"
        )
    observed = {
        "budget_bytes": selection.get("budget_bytes"),
        "chosen_achieved_bits": selection.get("chosen_achieved_bits"),
        "predicted_dloss": selection.get("predicted_dloss"),
        "selection_tensor_payload_bytes": whole.get(
            "selection_tensor_payload_bytes"
        ),
        "selection_whole_artifact_upper_bound_bytes": whole.get(
            "selection_whole_artifact_upper_bound_bytes"
        ),
    }
    if observed != DSV4_W8A16_APPROVED_SELECTION:
        raise W8A16ExportHandoffError(
            f"readmitted selection metrics differ from approval: {observed}"
        )
    return observed


def _verify_frozen_export_source_closure(
    repo_root: Path,
) -> dict[str, object]:
    if repo_root.is_symlink() or not repo_root.is_dir():
        raise W8A16ExportHandoffError(
            f"PrismaQuant root is not a real directory: {repo_root}"
        )
    observed: dict[str, str] = {}
    # Report the WHOLE drift, not the first file of it. Raising on the first
    # mismatch makes a re-freeze an N-round-trip guessing game: each refreshed
    # digest just advances the error to the next file, and the reviewer never
    # sees the size of what they are being asked to re-approve.
    drift: list[str] = []
    for relative, expected in _FROZEN_EXPORT_SOURCE_SHA256.items():
        path = _real_file(
            repo_root / relative,
            where=f"frozen exporter/source closure {relative}",
        )
        digest = _sha256(path)
        if digest != expected:
            drift.append(
                f"{relative}; observed={digest}, expected={expected}"
            )
        observed[relative] = digest
    if drift:
        raise W8A16ExportHandoffError(
            f"frozen exporter/source closure changed ({len(drift)} of "
            f"{len(_FROZEN_EXPORT_SOURCE_SHA256)} file(s)): "
            + "; ".join(drift)
        )
    closure: dict[str, object] = {
        "schema": DSV4_W8A16_EXPORT_SOURCE_CLOSURE_SCHEMA,
        "files_sha256": observed,
    }
    closure["identity_sha256"] = canonical_json_sha256(
        closure,
        where="DSv4 W8A16 exporter/source closure",
    )
    return closure


def _verify_runtime_contract() -> dict[str, object]:
    try:
        pin = load_gridbook_runtime_pin()
        require_exact_gridbook_runtime_release(pin)
    except Exception as exc:
        raise W8A16ExportHandoffError(
            f"Gridbook release pin is unresolved: {exc}"
        ) from exc
    contract = SOURCE_PASSTHROUGH_CONTRACTS["FP8_BLOCK_UE8M0_SOURCE"]
    if (
        pin.version != GRIDBOOK_RUNTIME_RELEASE_VERSION
        or pin.version_is_release is not True
        or pin.runtime_contract_schema != GRIDBOOK_RUNTIME_CONTRACT_SCHEMA
        or not supports_source_fp8_block128_w8a16(pin)
        or contract.serving_route != ROUTE_GRIDBOOK_FP8_SOURCE_W8A16
        or not contract.route_backed
        or "FP8_BLOCK_UE8M0_SOURCE" in ROUTE_PENDING_PASSTHROUGH_FORMATS
    ):
        raise W8A16ExportHandoffError(
            "FP8 block W8A16 is not backed by the exact released Gridbook "
            "runtime contract"
        )
    block = get_format("FP8_BLOCK_UE8M0_SOURCE")
    direct = get_format("MXFP8_UE8M0_G32")
    if (
        block.act_quant_changes_input
        or not direct.act_quant_changes_input
        or direct.act_bits != 8
    ):
        raise W8A16ExportHandoffError(
            "source W8A16 and direct group-32 W8A8 contracts have collapsed"
        )
    return {
        "schema": pin.schema,
        "repository": pin.repository,
        "commit": pin.commit,
        "version": pin.version,
        "version_is_release": pin.version_is_release,
        "runtime_contract_schema": pin.runtime_contract_schema,
        "required_abi_features": dict(pin.required_abi_features),
        "serving_route": contract.serving_route,
    }


def _verify_bundle(
    bundle_path: Path, layer_payload: Mapping[str, object]
) -> dict[str, object]:
    _real_file(bundle_path, where="immutable codebook bundle")
    context_stamp, _tensor_stamps = (
        cb_serialization_metadata_from_assignment_payload(layer_payload)
    )
    if not isinstance(context_stamp, Mapping):
        raise W8A16ExportHandoffError(
            "readmitted assignment lacks a CB serialization stamp"
        )
    try:
        from prismaquant.cb_learned_bundle import load_bundle

        bundle = load_bundle(bundle_path)
    except Exception as exc:
        raise W8A16ExportHandoffError(
            f"immutable codebook bundle is invalid: {bundle_path}"
        ) from exc
    if (
        context_stamp.get("codebook_content_sha256")
        != bundle.codebook_content_digests
        or context_stamp.get("codebook_source_by_format")
        != bundle.codebook_source_by_format
    ):
        raise W8A16ExportHandoffError(
            "codebook bundle bytes/source map differ from the assignment stamp"
        )
    return {
        "path": str(bundle_path.resolve(strict=True)),
        "file_sha256": _sha256(bundle_path),
        "bundle_content_sha256": bundle.bundle_content_sha256,
        "codebook_count": len(bundle.codebook_content_digests),
    }


def verify_dsv4_w8a16_export_handoff(
    *,
    publication_dir: str | Path,
    approved_raw_publication_dir: str | Path,
    source_model_dir: str | Path,
    source_identity_path: str | Path,
    codebook_bundle_path: str | Path,
    output_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Verify the fixed release handoff without mutating any input or output."""

    publication = Path(publication_dir)
    approved_raw = Path(approved_raw_publication_dir)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise W8A16ExportHandoffError(
            f"export output already exists; refusing clobber: {output}"
        )
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise W8A16ExportHandoffError(
            f"export output parent is not a real directory: {output.parent}"
        )

    manifest, published_sha256 = _verify_publication(publication)
    _raw_manifest, raw_sha256 = _verify_publication(approved_raw)
    expected_raw = {
        "layer_config.json": DSV4_W8A16_APPROVED_LAYER_CONFIG_SHA256,
        "selection.json": DSV4_W8A16_APPROVED_SELECTION_SHA256,
        "cb_col_weights.pkl": DSV4_W8A16_APPROVED_CB_COL_WEIGHTS_SHA256,
    }
    for name, expected in expected_raw.items():
        if raw_sha256[name] != expected:
            raise W8A16ExportHandoffError(
                f"approved raw publication changed at {name}"
            )

    layer_path = publication / "layer_config.json"
    selection_path = publication / "selection.json"
    layer_payload = _json_object(layer_path, where="readmitted layer config")
    selection = _json_object(selection_path, where="readmitted selection")
    try:
        raw_assignment = load_assignment(approved_raw / "layer_config.json")
        assignment = load_assignment(layer_path)
    except Exception as exc:
        raise W8A16ExportHandoffError(
            "approved/readmitted assignments are unreadable"
        ) from exc
    assignment_sha256 = assignment_serialization_sha256(assignment)
    if (
        assignment != raw_assignment
        or len(assignment) != DSV4_TOTAL_UNITS
        or assignment_sha256 != DSV4_W8A16_APPROVED_ASSIGNMENT_SHA256
        or sum(
            fmt == "FP8_BLOCK_UE8M0_SOURCE"
            for fmt in assignment.values()
        ) != 120
    ):
        raise W8A16ExportHandoffError(
            "readmitted full qname/format map differs from the approved "
            "33,325-unit assignment"
        )
    metrics = _selection_contract(selection)
    whole = selection["whole_artifact_budget"]
    if whole.get("selection_assignment_sha256") != assignment_sha256:
        raise W8A16ExportHandoffError(
            "readmitted whole-artifact accounting binds another assignment"
        )

    metadata = layer_payload.get("__prismaquant__")
    stamp = (
        metadata.get("aura_cb_reprice")
        if isinstance(metadata, Mapping) else None
    )
    readmission = stamp.get("cpu_replay") if isinstance(stamp, Mapping) else None
    attestation = (
        stamp.get("approved_raw_assignment_attestation")
        if isinstance(stamp, Mapping) else None
    )
    if (
        not isinstance(stamp, Mapping)
        or stamp.get("schema") != CB_ANCHORED_COST_SCHEMA
        or stamp.get("cost_currency") != AURA_CURRENCY
        or stamp.get("budget_bytes") != DSV4_W8A16_APPROVED_BUDGET_BYTES
        or selection.get("aura_cb_reprice") != stamp
        or selection.get("cost_currency") != AURA_CURRENCY
        or selection.get("feasible") is not True
        or not isinstance(readmission, Mapping)
        or readmission.get("schema") != DSV4_W8A16_READMISSION_SCHEMA
        or readmission.get("measurement_invoked") is not False
        or readmission.get("no_gpu_measurement_or_render") is not True
        or not isinstance(attestation, Mapping)
        or attestation.get("full_qname_format_map_equal") is not True
        or attestation.get("approved_assignment_sha256") != assignment_sha256
        or attestation.get("readmitted_assignment_sha256") != assignment_sha256
        or attestation.get("selection") != metrics
    ):
        raise W8A16ExportHandoffError(
            "publication lacks one matching CPU-only W8A16 readmission proof"
        )
    raw_stamp = readmission.get("approved_raw_publication")
    if (
        not isinstance(raw_stamp, Mapping)
        or Path(str(raw_stamp.get("publication", ""))).resolve(strict=False)
        != approved_raw.resolve(strict=True)
        or raw_stamp.get("assignment_sha256") != assignment_sha256
        or raw_stamp.get("selection") != metrics
        or raw_stamp.get("layer_config_sha256")
        != DSV4_W8A16_APPROVED_LAYER_CONFIG_SHA256
        or raw_stamp.get("selection_sha256")
        != DSV4_W8A16_APPROVED_SELECTION_SHA256
        or raw_stamp.get("cb_col_weights_sha256")
        != DSV4_W8A16_APPROVED_CB_COL_WEIGHTS_SHA256
    ):
        raise W8A16ExportHandoffError(
            "readmission provenance does not bind the exact approved raw "
            "publication"
        )

    runtime = _verify_runtime_contract()
    stamped_runtime = readmission.get("gridbook_runtime_pin")
    if stamped_runtime != {key: runtime[key] for key in (
        "schema", "repository", "commit", "version", "version_is_release",
        "runtime_contract_schema", "required_abi_features",
    )}:
        raise W8A16ExportHandoffError(
            "readmission was produced under a different Gridbook runtime pin"
        )

    from prismaquant.cost_streaming import (
        validate_cached_streamed_model_identity,
    )
    try:
        source_identity = validate_cached_streamed_model_identity(
            source_model_dir,
            source_identity_path,
            require_complete_checkpoint=True,
        )
    except Exception as exc:
        raise W8A16ExportHandoffError(
            "source checkpoint no longer matches its complete content identity"
        ) from exc

    from prismaquant.dspark_source_metadata import (
        discover_dspark_source_overlay_from_artifact,
    )
    try:
        overlay = discover_dspark_source_overlay_from_artifact(source_model_dir)
    except Exception as exc:
        raise W8A16ExportHandoffError(
            "DSpark source-header overlay is invalid"
        ) from exc
    routed_formats = set(assignment.values())
    if overlay is not None:
        routed_formats.update(overlay.construction_units.values())
    pending = sorted(routed_formats & ROUTE_PENDING_PASSTHROUGH_FORMATS)
    if pending:
        raise W8A16ExportHandoffError(
            f"release assignment still uses route-pending formats: {pending}"
        )

    bundle = _verify_bundle(Path(codebook_bundle_path), layer_payload)
    root = (
        Path(repo_root) if repo_root is not None
        else Path(__file__).resolve(strict=True).parent.parent
    )
    frozen = _verify_frozen_export_source_closure(root)
    return {
        "schema": DSV4_W8A16_EXPORT_HANDOFF_SCHEMA,
        "publication": str(publication.resolve(strict=True)),
        "publication_identity_sha256": manifest["identity_sha256"],
        "published_sha256": published_sha256,
        "approved_raw_publication": str(approved_raw.resolve(strict=True)),
        "assignment_sha256": assignment_sha256,
        "unit_count": len(assignment),
        "fp8_block_w8a16_count": 120,
        "selection": metrics,
        "source_checkpoint": {
            "identity_path": str(Path(source_identity_path).resolve(strict=True)),
            "content_sha256": source_identity["content_sha256"],
            "shard_count": len(source_identity["shards"]),
        },
        "codebook_bundle": bundle,
        "gridbook_runtime_pin": runtime,
        "frozen_export_source_closure": frozen,
        "output_path": str(output.resolve(strict=False)),
        "output_absent": True,
    }


__all__ = [
    "DSV4_W8A16_EXPORT_HANDOFF_SCHEMA",
    "DSV4_W8A16_EXPORT_SOURCE_CLOSURE_SCHEMA",
    "W8A16ExportHandoffError",
    "verify_dsv4_w8a16_export_handoff",
]
