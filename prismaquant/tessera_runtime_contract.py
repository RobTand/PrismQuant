"""Read the Tessera serving plugin's own packaged runtime contract.

Principle 14 in one file: every statement PrismaQuant makes about what a
Tessera artifact *serves* -- which family, at which rate, on which route,
behind which serve flags, at which tensor-parallel world size -- is read from
``tessera/serving/runtime_contract.json``, the table the plugin packages and
publishes, or it is refused.  Nothing here decides anything; it parses.

Since Tessera contract v7 that table also publishes what the plugin *LOADS*
(``native_extensions``: which CUDA library, under which filename pattern,
matched by which named rule, and what runs instead when it is absent), which
is the other §7.4 fact PrismaQuant used to maintain on this side by hand.
:func:`require_pin_native_extensions_match_contract` is the refusal that keeps
the serving pin a transcription of it.

The cell grammar has one home in ``lane_eligibility``. This reader adds the
development answer pin, format ranges, native-extension identity and fused
module licence; export uses the same parsed cells under the separate release
pin. Since lane schema v4, residency scopes each cell and ``executes`` names
its launches. Both readers retain those fields and reject malformed or
overlapping claims through the shared parser. Neither accepts a Gridbook
schema; that serving lane was retired on 2026-09-02.

**The dev pin.**  PrismaQuant's Tessera admission is fail-closed until a
Tessera RELEASE tag exists, and none has been cut.  A development override is
therefore explicit, named, and loud:

* ``PRISMAQUANT_TESSERA_DEV_PIN=<anything non-empty>`` opts in.  The value is
  recorded verbatim in provenance as what the operator asked for; it is not
  the gate.
* the installed contract's **answer** -- every value the ADMISSION gate
  reads, in the vocabulary of :func:`contract_answer` -- must equal
  :data:`TESSERA_DEV_PIN_ANSWER` or the read raises, with a field-level diff
  naming what moved.
* unset is production: no Tessera contract is read at all, and every rung
  stays ``unattested`` exactly as before.

There is no third state.  A mismatch never degrades to "unattested" -- that
would turn a stale pin into a silently empty menu, which is the failure mode
this whole file exists to prevent.

**Why the answer and not the bytes** (issue #38).  This pin used to compare
the environment's value against an exact commit and the file's sha256 against
a recorded one.  Both legs fired on *identity*, and the thing they name is an
editable checkout on the same box, so every Tessera commit that touched the
contract -- a prose ``detail``, a changelog paragraph, a ``contract_version``
bump -- turned PrismaQuant's attested path off and a fistful of tests red in a
repo nobody was editing, while the two rungs' meaning had not moved at all.
That is principle 14 read backwards: prose fields explain, they are never a
value a gate reads, so a prose edit is not a thing to re-review.  The gate now
reads the answer.  A commit that moves no answer passes silently; a commit
that moves one refuses and says which field, which is a review prompt rather
than a corruption warning.  :data:`TESSERA_DEV_PIN_COMMIT` and
:data:`TESSERA_DEV_PIN_CONTRACT_SHA256` survive as the *record of the review*
-- the build and the bytes a human read when the answer was accepted -- and
travel into provenance alongside the bytes this run actually read, so
prose-only drift is visible without being fatal.

This is deliberately weaker than a release pin and says so: it admits any
Tessera whose table answers identically, which is exactly the claim
PrismaQuant makes about it.  A Tessera RELEASE tag (issue #17) is still what
retires the override.

The contract's own identity (commit, sha, path, schema, contract_version)
travels into every allocation's provenance as ``tessera_dev_pin`` so a
shipcard records which table admitted its units.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .lane_eligibility import (
    CellEvidence,
    SCOPED_LANE_SCHEMAS,
    cell_evidence_admits,
    cell_lane_admits,
    LANE_ELIGIBILITY_SCHEMA_TESSERA,
    LANE_ELIGIBILITY_SCHEMA_TESSERA_V4,
    LaneClaim,
    LaneEligibilityError,
    ServingContext,
    QUALIFICATION_DEVICE_QUALIFIED,
    ROUTE_STATUS_BACKED,
    ROUTE_STATUS_BACKED_WITH_SERVE_FLAG,
    _parse_table,
    cell_matches_serving_context,
    parse_lane_claim,
)

__all__ = [
    "FUSED_MODULE_FIELD_LICENCES",
    "FUSED_MODULE_SCHEMA",
    "FusedModuleLicence",
    "MATCH_BASENAME_FNMATCH",
    "TESSERA_CONTRACT_SCHEMA",
    "TESSERA_DEV_PIN_COMMIT",
    "TESSERA_DEV_PIN_ANSWER",
    "TESSERA_DEV_PIN_CONTRACT_SHA256",
    "TESSERA_DEV_PIN_ENV",
    "TESSERA_LANE_SCHEMA",
    "TesseraContract",
    "TesseraContractError",
    "TesseraNativeExtension",
    "TesseraRouteCell",
    "cell_activation_projection",
    "contract_answer",
    "describe_dev_pin",
    "dev_pin_requested",
    "load_tessera_contract",
    "require_pin_native_extensions_match_contract",
]


class TesseraContractError(RuntimeError):
    """The Tessera runtime contract is absent, malformed, or off its pin."""


#: The schema ids this reader accepts.  Both are checked before any key is
#: read: an older table is not a subset of this one, and "missing field" is the
#: wrong error to hand someone whose contract predates the field.
#: ``TESSERA_LANE_SCHEMA`` is the CURRENT lane grammar and follows
#: ``lane_eligibility.LANE_ELIGIBILITY_SCHEMA_TESSERA`` (v8 since 2026-09-05)
#: so that the two readers cannot disagree about which schema is newest.
TESSERA_CONTRACT_SCHEMA = "tessera.runtime-contract.v1"
TESSERA_LANE_SCHEMA = LANE_ELIGIBILITY_SCHEMA_TESSERA
TESSERA_LANE_SCHEMAS = frozenset(
    {LANE_ELIGIBILITY_SCHEMA_TESSERA_V4} | SCOPED_LANE_SCHEMAS)
#: The ``fused_module`` block's own schema id, checked the same way.
FUSED_MODULE_SCHEMA = "tessera.fused-module.v1"

#: The only two licences a ``fused_module.fields`` entry may carry.  An
#: unknown token is REFUSED rather than mapped onto either: "shared" and
#: "per_member" are the two answers a group allocator can act on, and a third
#: word this reader does not know is a constraint it would silently drop.
FUSED_MODULE_FIELD_LICENCES = frozenset({"shared", "per_member"})

#: The development override.  See the module docstring; there is no default.
TESSERA_DEV_PIN_ENV = "PRISMAQUANT_TESSERA_DEV_PIN"

#: The Tessera commit this pin's answer was reviewed against.  Declared and
#: recorded; NOT compared to anything.  A moving ``master`` is not a review
#: event -- :data:`TESSERA_DEV_PIN_ANSWER` is what refuses.  Last re-read at
#: contract v21 / lane schema v8 (Tessera master 3efd690, the merge of its
#: #324; contract v21 landed eleven merges earlier in its #313, at b8b1cb38,
#: and the release e78959ed carried v20), which is the same commit
#: ``tessera_serving_runtime_pin`` binds: the two pins name ONE object, and
#: letting them drift is how two of this repository's own spec files came to
#: disagree about one runtime.  Between the v17 review and this one the
#: answer moved in exactly four places and nowhere else -- no family, rung,
#: route status, launch, image or version moved, and the ten cell ids are the
#: same ten in the same order: the lane schema (v6 -> v8); every cell's
#: evidence gained ``smoke.attribution`` / ``smoke.control`` (v7) and
#: ``artifact`` (v8); every native extension gained its ``lane`` predicate
#: (v20); and the two ``routed_moe`` cells' ``smoke.status`` moved from
#: ``repetitive`` to ``recorded`` (v21, Tessera #313, receipt
#: ``docs/measurements/moe-smoke-recorded-2026-09-05.md``), with the v18
#: control retired from them (``attribution: unattributed``, ``control:
#: null`` -- the shape the BF16 cells already used).
#:
#: The answer below TRANSCRIBES what the runtime publishes, including its two
#: ``routed_moe`` cells.  Transcribing them does not admit them: admission is
#: a separate predicate (``lane_eligibility.cell_evidence_admits``) that
#: refuses on ``smoke.status`` alone.  Under v17 and v20 it refused both
#: routed-MoE cells on the degenerate greedy smoke the runtime recorded
#: (``repetitive``); under v21 the runtime records a smoke through the
#: checkpoint's own chat template that does not degenerate on either arm, the
#: status reads ``recorded``, and the SAME status-only rule admits both cells
#: with no predicate change here (prismaquant #198, option C).  That is the
#: mechanism working as designed: the evidence block is part of the answer,
#: so the clean smoke re-staled the pin and this literal's diff is the review
#: a human reads before anything is admitted.  Whether routed-MoE Tessera is
#: PROMOTED past the menu is still Rob's decision under principle 9 (#198
#: stays open for it).
#:
#: Re-pinned to master's current tip on 2026-09-05 rather than to b8b1cb38,
#: the commit that first published this answer: 56 commits (11 PR merges,
#: Tessera #312 and #314-#324) later, ``src/tessera/serving/
#: runtime_contract.json`` is byte-identical, so the answer below is
#: unchanged and the digest is the same one.  That is the property this
#: design is for -- what re-stales a pin is a move in what the runtime
#: PUBLISHES, not a move in ``master``.  Verified by fetching
#: ``RobTand/tessera`` master and hashing the blob at the tip, not a working
#: tree (the command is in ``tessera_runtime/README.md``).
TESSERA_DEV_PIN_COMMIT = "3efd690dd4e4a1b71fb604a14bee521dc57badb3"

#: sha256 of ``tessera/serving/runtime_contract.json`` at that commit -- the
#: bytes a human read when the answer below was accepted.  Recorded, and
#: compared into provenance against the bytes this run read, so prose-only
#: drift is visible; it is not the refusal.
TESSERA_DEV_PIN_CONTRACT_SHA256 = (
    "34f1da7977f1aa155cd2ff18b584e5c35a6089b648bb4a51d449fc25082a2c3e"
)

#: The ANSWER this pin was reviewed against -- every value the ADMISSION
#: gate reads, in the vocabulary of :func:`contract_answer`.  This literal, not the file's
#: bytes, is what refuses: a Tessera commit that rewrites prose, reorders keys
#: or bumps ``contract_version`` publishes the same answer and does not
#: re-stale the pin, while any move in a family, a rate range, an attested
#: rung, a world-size ceiling, a cell, the canonical ``quant_method``, or a
#: published native extension (its prefix, its glob, the ``match`` rule, the
#: routes that need it, or what runs when it is absent) does -- with a
#: field-level diff naming it.  The git diff of this literal is the review.
TESSERA_DEV_PIN_ANSWER = {'schema': 'tessera.runtime-contract.v1',
 'lane_schema': 'tessera.lane-eligibility.v8',
 'required_regimes': ['batch', 'decode'],
 'quant_method': 'tessera',
 'fused_module': {'schema': 'tessera.fused-module.v1',
                  'fields': {'body': 'shared',
                             'columns': 'shared',
                             'family': 'shared',
                             'grid': 'shared',
                             'plane': 'shared',
                             'q256': 'per_member',
                             'rows': 'per_member',
                             'structure': 'shared'},
                  'sidecar_q256': 'int_or_per_role_list',
                  'mixed_rung_receipt': False},
 'native_extensions': [{'module_name_prefix': 'tessera_nvfp4_',
                        'filename_glob': 'tessera_nvfp4_*.so',
                        'match': 'basename_fnmatch',
                        'routes': ['TESSERA_NVFP4'],
                        'when_unavailable': {'resident': {'status': 'substituted',
                                                          'decoder': 'torch_materialize_stock'},
                                             'streamed': {'status': 'refused',
                                                          'decoder': None}},
                        'lane': {'decoder': 'native_span2', 'requires': None}},
                       {'module_name_prefix': 'tessera_window_gemv',
                        'filename_glob': 'tessera_window_gemv*.so',
                        'match': 'basename_fnmatch',
                        'routes': ['TESSERA_BF16', 'TESSERA_FP8'],
                        'when_unavailable': {'resident': {'status': 'substituted',
                                                          'decoder': 'torch_window'},
                                             'streamed': {'status': 'substituted',
                                                          'decoder': 'torch_window'}},
                        'lane': {'decoder': 'window_gemv',
                                 'requires': {'column_rates': [1, 2, 4],
                                              'window_bits': [14],
                                              'body': 'window',
                                              'plane': 'channel',
                                              'release_overrides': False,
                                              'diagonals': False,
                                              'rotation': ['none'],
                                              'start_state': False,
                                              'grid_arities': [1]}}}],
 'families': {'TESSERA_BF16_K1': {'reader_rate_range_q256': [256, 4096],
                                  'attested_rungs_q256': [1792],
                                  'max_world_size': 1},
              'TESSERA_E2M1_K2': {'reader_rate_range_q256': [896, 896],
                                  'attested_rungs_q256': [896],
                                  'max_world_size': 1},
              'TESSERA_E4M3_K1': {'reader_rate_range_q256': [256, 2048],
                                  'attested_rungs_q256': [1024],
                                  'max_world_size': 1}},
 'cells': [['tessera_bf16_k1_dense_sm121_batch',
            'sm_121',
            'TESSERA_BF16_K1',
            'dense',
            'batch',
            [1792],
            'bf16_unquantized',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident|streamed'],
            [['torch.mm', 'torch_window']],
            ['resident', 'streamed'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['compiled', 'eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['kl_lower_bound',
             'recorded',
             ['topk_intersection_lower_bound@1024'],
             'unattributed',
             None,
             None]],
           ['tessera_bf16_k1_dense_sm121_decode',
            'sm_121',
            'TESSERA_BF16_K1',
            'dense',
            'decode',
            [1792],
            'bf16_unquantized',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident|streamed'],
            [['torch.mm', 'torch_window']],
            ['resident', 'streamed'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['compiled', 'eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['route_only', 'recorded', [], 'unattributed', None, None]],
           ['tessera_e2m1_k2_dense_sm121_batch',
            'sm_121',
            'TESSERA_E2M1_K2',
            'dense',
            'batch',
            [896],
            'e2m1_group16_ue4m3_static',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident|streamed'],
            [['torch._scaled_mm', 'native_span2']],
            ['resident', 'streamed'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['compiled', 'eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['kl_lower_bound',
             'not_recorded',
             ['topk_intersection_lower_bound@1024'],
             'unattributed',
             None,
             None]],
           ['tessera_e2m1_k2_dense_sm121_decode',
            'sm_121',
            'TESSERA_E2M1_K2',
            'dense',
            'decode',
            [896],
            'e2m1_group16_ue4m3_static',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident|streamed'],
            [['torch._scaled_mm', 'native_span2']],
            ['resident', 'streamed'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['compiled', 'eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['route_only', 'not_recorded', [], 'unattributed', None, None]],
           ['tessera_e4m3_k1_dense_sm121_batch_resident',
            'sm_121',
            'TESSERA_E4M3_K1',
            'dense',
            'batch',
            [1024],
            'fp8_per_token_dynamic',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident'],
            [['torch._scaled_mm', 'torch_window']],
            ['resident'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['compiled', 'eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['kl_lower_bound',
             'not_recorded',
             ['topk_intersection_lower_bound@1024'],
             'unattributed',
             None,
             ['gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook',
              '8070ec6c4e0448826cda3f3f8d9401a125444e3b',
              '331703661baaa67ed73900c3d9e99f300fdc8415',
              'model.layers.0.mlp.down_proj',
              'different',
              'lower']]],
           ['tessera_e4m3_k1_dense_sm121_batch_streamed',
            'sm_121',
            'TESSERA_E4M3_K1',
            'dense',
            'batch',
            [1024],
            'fp8_per_token_dynamic',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=streamed'],
            [['tessera_window_gemv::gemv', 'window_gemv'],
             ['torch._scaled_mm', 'window_gemv']],
            ['streamed'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['compiled', 'eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['kl_lower_bound',
             'not_recorded',
             ['topk_intersection_lower_bound@1024'],
             'unattributed',
             None,
             ['gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook',
              '8070ec6c4e0448826cda3f3f8d9401a125444e3b',
              '331703661baaa67ed73900c3d9e99f300fdc8415',
              'model.layers.0.mlp.down_proj',
              'different',
              'lower']]],
           ['tessera_e4m3_k1_dense_sm121_decode_resident',
            'sm_121',
            'TESSERA_E4M3_K1',
            'dense',
            'decode',
            [1024],
            'fp8_per_token_dynamic',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident'],
            [['torch._scaled_mm', 'torch_window']],
            ['resident'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['compiled', 'eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['route_only',
             'not_recorded',
             [],
             'unattributed',
             None,
             ['gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook',
              '8070ec6c4e0448826cda3f3f8d9401a125444e3b',
              '331703661baaa67ed73900c3d9e99f300fdc8415',
              'model.layers.0.mlp.down_proj',
              'different',
              'lower']]],
           ['tessera_e4m3_k1_dense_sm121_decode_streamed',
            'sm_121',
            'TESSERA_E4M3_K1',
            'dense',
            'decode',
            [1024],
            'fp8_per_token_dynamic',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=streamed'],
            [['tessera_window_gemv::gemv', 'window_gemv']],
            ['streamed'],
            {'image': 'vllm/vllm-openai@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14',
             'execution_modes': ['compiled', 'eager']},
            '0.28.0',
            '2.13.0+cu130',
            ['kl_lower_bound',
             'not_recorded',
             ['topk_intersection_lower_bound@1024',
              'topk_intersection_lower_bound@1024'],
             'unattributed',
             None,
             ['gbfam/qwen3-0.6b-tessera-e4m3-reach-gridbook',
              '8070ec6c4e0448826cda3f3f8d9401a125444e3b',
              '331703661baaa67ed73900c3d9e99f300fdc8415',
              'model.layers.0.mlp.down_proj',
              'different',
              'lower']]],
           ['tessera_e4m3_k1_routed_moe_sm121_batch_resident',
            'sm_121',
            'TESSERA_E4M3_K1',
            'routed_moe',
            'batch',
            [1024],
            'fp8_per_token_dynamic',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident'],
            [['vllm.fused_moe.modular_kernel', 'torch_materialize_stock']],
            ['resident'],
            {'image': 'eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c',
             'execution_modes': ['eager']},
            '0.28.1rc1.dev397+gfd4a15126.d20260904',
            '2.13.0+cu130',
            ['kl_lower_bound',
             'recorded',
             ['topk_intersection_lower_bound@1024'],
             'unattributed',
             None,
             None]],
           ['tessera_e4m3_k1_routed_moe_sm121_decode_resident',
            'sm_121',
            'TESSERA_E4M3_K1',
            'routed_moe',
            'decode',
            [1024],
            'fp8_per_token_dynamic',
            'backed_with_serve_flag',
            'device_qualified',
            'tessera',
            ['TESSERA_SERVE_MODE=resident'],
            [['vllm.fused_moe.modular_kernel', 'torch_materialize_stock']],
            ['resident'],
            {'image': 'eugr/spark-vllm@sha256:0afec8d4f79f44685a1ddf758659d33aef3b0f3ec9068e5a7cd1108d30e5581c',
             'execution_modes': ['eager']},
            '0.28.1rc1.dev397+gfd4a15126.d20260904',
            '2.13.0+cu130',
            ['route_only', 'recorded', [], 'unattributed', None, None]]]}

#: Route statuses under which a cell says a native route EXECUTES.
_NATIVE_ROUTE_STATUSES = frozenset(
    {ROUTE_STATUS_BACKED, ROUTE_STATUS_BACKED_WITH_SERVE_FLAG}
)


#: What a published ``activation_contract`` string executes as, in the
#: vocabulary the allocator prices in: ``(act_bits, act_group_size)``.
#: A transcription of the runtime's vocabulary, in exactly one place, and
#: fail-closed: a contract string not listed here raises rather than
#: guessing, because guessing an A side is the currency error this table
#: exists to prevent (2026-08-17, 87 GB on NVFP4_CB).  Each leg is pinned
#: against the producer route for the family that publishes it -- the
#: ``tessera_serving_route`` contract quoted -- so the two halves of one
#: A side cannot drift apart without renaming one of them:
#:
#: * ``e2m1_group16_ue4m3_static``: 4-bit E2M1 activations in groups of 16
#:   with static scales -- ``w4a4-nvfp4-e2m1-group16-ue4m3``,
#:   ``(act_bits, act_group_size) == (4, 16)``.
#: * ``fp8_per_token_dynamic``: 8-bit FP8 activations with per-token dynamic
#:   scales -- ``w8a8-dynamic-e4m3-channel``, ``(8, 0)``: group 0 is the
#:   registry's spelling for ungrouped (the same one the ``FP8_E4M3`` row
#:   carries), and the dynamic half lives in the activation quantiser the
#:   route borrows by reference, not in the group size.
#: * ``bf16_unquantized``: 16-bit activations, no quantisation --
#:   ``w16a16-bf16-channel``, ``(16, 0)``.
_CELL_ACTIVATION_PROJECTION = {
    "e2m1_group16_ue4m3_static": (4, 16),
    "fp8_per_token_dynamic": (8, 0),
    "bf16_unquantized": (16, 0),
}


def cell_activation_projection(cell_contract: str) -> tuple[int, int]:
    """Project a cell's ``activation_contract`` onto ``(act_bits, act_group_size)``.

    The comparison :func:`prismaquant.tessera_menu.route_admission` runs:
    the producer prices ``tessera_serving_route``'s ``(act_bits,
    act_group_size)`` while the runtime executes the cell's string, and the
    two vocabularies do not match character for character.  Comparing the
    projection refuses a family whose priced A side is not the executed one.
    An unpublished string raises :class:`TesseraContractError` -- a new
    runtime vocabulary is a thing to transcribe in
    :data:`_CELL_ACTIVATION_PROJECTION`, never a thing to admit blind.
    """
    try:
        return _CELL_ACTIVATION_PROJECTION[str(cell_contract)]
    except KeyError:
        raise TesseraContractError(
            f"cell activation contract {cell_contract!r} is not a published "
            f"vocabulary this reader transcribes "
            f"({sorted(_CELL_ACTIVATION_PROJECTION)}); transcribing a new "
            "runtime vocabulary is a review, and admitting it blind would "
            "price an A side nothing executed"
        ) from None


@dataclass(frozen=True, slots=True)
class TesseraNativeExtension:
    """One ``native_extensions[]`` row, verbatim in the fields we read.

    The contract says what the plugin EXECUTES in ``formats`` /
    ``lane_eligibility``; this table says what it LOADS.  PrismaQuant needs it
    because §7.4 keys reproducibility on extension residency: a KL is
    bit-identical inside one container session and drifts 4-8x across them,
    keyed purely on whether a lane's ``.so`` was resident, so an A/B is
    comparable only across serves whose native-extension residency matches.
    A lane whose library no fingerprint pattern matches reports "nothing
    resident", and two serves the rule cannot tell apart get compared.

    ``match`` is the field that makes this principle 14 rather than a guess:
    the runtime names the RULE a consumer applies, so a consumer does not have
    to decide whether the published string is a stem, a prefix or a pattern.
    """

    #: The constant the runtime's JIT load path passes to
    #: ``cpp_extension.load``, trailing separator included.
    module_name_prefix: str
    #: The library name that produces.  A glob, because the module name
    #: carries a build-identity hash and no exact basename exists.
    filename_glob: str
    #: The rule that turns ``filename_glob`` into a decision.
    match: str
    #: Where the sources live in the runtime's tree -- identity, not answer.
    source: str
    #: The runtime module that loads it -- identity, not answer.
    loaded_by: str
    #: The routes that need it.
    routes: tuple[str, ...]
    #: Per residency mode, what a serve does when the library is absent:
    #: ``{"resident": {"status": "substituted", "decoder": "..."},
    #: "streamed": {"status": "refused", "decoder": None}}``.  This is what
    #: makes an absent ``.so`` readable: in one mode the serve keeps running on
    #: a NAMED substitute decoder and is a different numeric object, in the
    #: other there is no serve at all.
    when_unavailable: Mapping[str, Mapping[str, Any]]
    #: What the extension's kernel READS (contract v20): the decoder name it
    #: stamps and, when it publishes one, the predicate a unit's wire must
    #: satisfy for the lane to take it (``lane.requires``).  Parsed by
    #: ``lane_eligibility.parse_lane_claim`` -- the same reader the serving
    #: pin's table uses -- and decided by ``cell_lane_admits`` in
    #: :meth:`TesseraContract.native_cells`, so the development contract
    #: admits a rung on exactly the terms the pinned one does.
    lane: LaneClaim

    def as_contract_row(self) -> dict:
        """The four fields a residency predicate -- and the reading of an
        absent library -- is made of, as published."""
        return {
            "module_name_prefix": self.module_name_prefix,
            "filename_glob": self.filename_glob,
            "match": self.match,
            "when_unavailable": {
                mode: {"status": behaviour["status"],
                       "decoder": behaviour["decoder"]}
                for mode, behaviour in sorted(self.when_unavailable.items())
            },
        }


@dataclass(frozen=True, slots=True)
class TesseraRouteCell:
    """One ``lane_eligibility.cells[]`` row, verbatim in the fields we read."""

    cell_id: str
    platform: str
    family: str
    structure: str
    regime: str
    rungs_q256: frozenset[int]
    activation_contract: str
    route_status: str
    qualification: str
    requires_plugin: str
    requires_serve_flags: tuple[str, ...]
    executes: tuple[tuple[str, str], ...]
    residency_modes: tuple[str, ...]
    runtime_image: str
    execution_modes: tuple[str, ...]
    #: v6's per-cell runtime versions and evidence block; empty/``None`` under
    #: the pre-v6 grammars, which published neither.
    runtime_vllm: str = ""
    runtime_torch: str = ""
    evidence: "CellEvidence | None" = None

    @property
    def native(self) -> bool:
        """Does this cell attest a route the runtime executes natively?"""
        return (
            self.qualification == QUALIFICATION_DEVICE_QUALIFIED
            and self.route_status in _NATIVE_ROUTE_STATUSES
        )


@dataclass(frozen=True, slots=True)
class FusedModuleLicence:
    """``fused_module``: what one vLLM-fused module's roles must SHARE.

    The value a producer's group allocator reads instead of guessing.  vLLM
    merges q/k/v and gate/up into one Linear and builds ONE quant method per
    module, so everything that selects a method or a tile is a module fact;
    the RATE is not, because every decoder in the plugin is fed from each
    member's own parsed manifest.  Which of the two a field is is exactly what
    :attr:`fields` says, and it is checked on the Tessera side against
    ``tessera.serving.scheme.FUSED_MODULE_FIELDS`` -- the dict the loader
    itself gates on -- so the table cannot drift from the code.

    Read, never inferred.  ``prismaquant.allocator_candidates`` folds a fused
    group's menu over the fields this block marks ``per_member`` and holds
    every ``shared`` one fixed; a contract that re-tightens ``q256`` to
    ``shared`` therefore stops the fold rather than leaving it enumerating
    rungs the exporter refuses (prismaquant #132, RobTand/tessera#37).

    Every attribute here is a value a gate on this side decides on, which is
    what makes :meth:`answer` the block's whole projection into
    :func:`contract_answer`.  The block's ``container`` is deliberately not
    among them: nothing here reads the sidecar's magic, because there is no
    Tessera export leg to write one with.
    """

    #: The block's own schema id.  A gate: the reader refuses a block carrying
    #: another id rather than reading it as a subset of this one.
    schema: str
    #: ``field -> "shared" | "per_member"``, verbatim.
    fields: Mapping[str, str]
    #: How a mixed-rung module is SPELLED in the checkpoint's scheme.
    #: Recorded and carried into the fold's receipt; there is no writer yet.
    sidecar_q256: str
    #: Whether a container receipt covers a SERVED mixed-rung module.  False
    #: today: the relaxation is proven by a decode identity, not by a serve.
    #: A shipcard for an artifact that ships one has to say so.
    mixed_rung_receipt: bool

    def licence_for(self, field: str) -> str:
        """``"shared"``/``"per_member"`` for ``field``; an unpublished field raises.

        Absence is not permission.  A field the contract does not name is one
        this runtime has published nothing about, and guessing either way is
        the assertion principle 14 refuses.
        """
        try:
            return self.fields[str(field)]
        except KeyError:
            raise TesseraContractError(
                f"the Tessera contract's fused_module block publishes no "
                f"licence for {field!r}; it names "
                f"{sorted(self.fields)}. A field it does not name is not "
                "'probably per-member' -- ask Tessera to publish it."
            ) from None

    def is_per_member(self, field: str) -> bool:
        """Is ``field`` free to differ between one module's roles?"""
        return self.licence_for(field) == "per_member"

    def shared_fields(self) -> frozenset[str]:
        """Every field the module's roles must agree on."""
        return frozenset(
            name for name, licence in self.fields.items() if licence == "shared"
        )

    def per_member_fields(self) -> frozenset[str]:
        """Every field each role may hold on its own."""
        return frozenset(
            name for name, licence in self.fields.items()
            if licence == "per_member"
        )

    def answer(self) -> dict:
        """The gate-read projection, for :func:`contract_answer`.

        Derived from this object rather than from the JSON block, so the
        answer carries exactly what the reader kept and a field the reader
        does not parse cannot reach the pin.
        """
        return {
            "schema": str(self.schema),
            "fields": {str(k): str(v) for k, v in sorted(self.fields.items())},
            "sidecar_q256": str(self.sidecar_q256),
            "mixed_rung_receipt": bool(self.mixed_rung_receipt),
        }


@dataclass(frozen=True, slots=True)
class TesseraContract:
    """The plugin's packaged contract, parsed.  Every field is read, not typed."""

    #: ``family -> (lo, hi)`` inclusive q256 rate range the reader accepts.
    reader_rate_range: Mapping[str, tuple[int, int]]
    #: ``family -> the rungs a ``lane_eligibility`` cell attests``.
    attested_rungs: Mapping[str, frozenset[int]]
    cells: tuple[TesseraRouteCell, ...]
    #: The libraries the plugin loads into a serving process, in the table's
    #: order.  Non-empty by construction: :func:`_parse` refuses a contract
    #: that publishes no table rather than reading one as "loads nothing".
    native_extensions: tuple[TesseraNativeExtension, ...]
    #: ``family -> max tensor-parallel world size``, closed world.
    max_world_size: Mapping[str, int]
    #: What one vLLM-fused module's roles must share, and what is free.
    fused_module: FusedModuleLicence
    quant_method: str
    contract_version: int
    plugin_version: str
    #: ``versions.default_serve_image``: the ONE serve-image pin every harness
    #: reads. It replaced ``versions.attested_on`` at contract v17, which had
    #: made a single global claim about the runtime every cell was measured on
    #: -- false the moment one cell was measured on a dev wheel. The per-cell
    #: truth now lives on the cells (``runtime_image``/``runtime_vllm``/
    #: ``runtime_torch``); this is a default, and identity, not a gate input.
    default_serve_image: str
    #: Identity -- what travels into provenance.
    commit: str
    sha256: str
    path: str
    lane_schema: str
    regimes: tuple[str, ...]

    @property
    def requires_serving_context(self) -> bool:
        return self.lane_schema in SCOPED_LANE_SCHEMAS

    def governs(self, family: str) -> bool:
        """Does the contract publish this payload family at all?"""
        return str(family) in self.reader_rate_range

    def native_cells(self, family: str, rate_q256: int, *,
                     serving_context: ServingContext | None = None,
                     ) -> tuple[TesseraRouteCell, ...]:
        """V5 admits every required regime only under one explicit context.

        Context-free v4 keeps its historical family/rate projection. An
        explicit runtime query cannot borrow that unscoped claim. V5 never uses
        the first matching cell to infer an image, execution mode or structure.
        """
        if self.requires_serving_context and serving_context is None:
            return ()
        if not self.requires_serving_context and serving_context is not None:
            return ()
        selected = tuple(
            cell for cell in self.cells
            if cell.family == str(family)
            and int(rate_q256) in cell.rungs_q256
            and cell.native
            # The development menu reads the SAME evidence predicate the
            # export gate reads (``lane_eligibility.cell_evidence_admits``).
            # A rung the dev menu offers and the export refuses is the
            # split-brain principle 8 exists to stop.
            and cell_evidence_admits(cell)[0]
            # And the SAME lane predicate: the lane a cell launches through
            # must read what this producer plans at the rung
            # (``lane_eligibility.cell_lane_admits``, deciding
            # ``native_extensions[].lane.requires`` over
            # ``tessera_render.planned_wire_facts``).
            and cell_lane_admits(cell, int(rate_q256), self.lanes)[0]
            and (not self.requires_serving_context
                 or cell_matches_serving_context(cell, serving_context))
        )
        if self.requires_serving_context and {cell.regime for cell in selected} != set(self.regimes):
            return ()
        return selected

    @property
    def lanes(self) -> tuple[LaneClaim, ...]:
        """Every extension's lane claim, in table order, for the lane gate."""
        return tuple(ext.lane for ext in self.native_extensions)

    def identity(self) -> dict:
        """The ``tessera_dev_pin`` provenance block.

        Records the review *and* the read: which Tessera build and bytes a
        human accepted this answer against, which bytes this run actually
        consumed, and whether the two are the same file.  They can differ
        legitimately -- the gate is the answer, not the bytes -- and a
        shipcard that could not tell the two apart would be asserting a
        review it did not get.
        """
        return {
            "requested": dev_pin_requested(),
            "commit": self.commit,
            "contract_sha256": self.sha256,
            "reviewed_contract_sha256": TESSERA_DEV_PIN_CONTRACT_SHA256,
            "bytes_are_the_reviewed_bytes":
                self.sha256 == TESSERA_DEV_PIN_CONTRACT_SHA256,
            "contract_path": self.path,
            "schema": TESSERA_CONTRACT_SCHEMA,
            "contract_version": self.contract_version,
            "plugin_version": self.plugin_version,
            "quant_method": self.quant_method,
            "default_serve_image": self.default_serve_image,
            "native_extensions": [
                {
                    "module_name_prefix": ext.module_name_prefix,
                    "filename_glob": ext.filename_glob,
                    "match": ext.match,
                    "source": ext.source,
                    "loaded_by": ext.loaded_by,
                    "routes": list(ext.routes),
                    "when_unavailable": {
                        mode: dict(behaviour)
                        for mode, behaviour in sorted(
                            ext.when_unavailable.items())
                    },
                }
                for ext in self.native_extensions
            ],
            "note": (
                "development override: no Tessera RELEASE tag exists, so this "
                "allocation's Tessera routes were admitted by the packaged "
                "contract, whose ANSWER (every value the admission gate "
                "reads) equals the one reviewed at the named commit. Its "
                "scope is admission: the export lane's structures/platforms/"
                "regimes are the RELEASE pin's. The bytes need not be the "
                "reviewed bytes; bytes_are_the_reviewed_bytes says which"
            ),
        }


def describe_dev_pin(identity: "Mapping[str, Any]") -> str:
    """One human-readable line for a :meth:`TesseraContract.identity` block.

    Since the pin became an *answer* pin, bytes that differ from the reviewed
    ones are legal: a Tessera commit adding a block no gate reads moves the
    file and not the answer, and the answer is what admitted the routes.  But
    that makes ``bytes_are_the_reviewed_bytes`` the one field a reader needs
    and the one the old log line left out -- it printed a sha with nothing to
    compare it against, so "these are not the bytes a human reviewed" existed
    only inside a provenance blob nobody opens.  Name it here.
    """

    line = (f"commit={identity['commit'][:12]} "
            f"contract_sha={identity['contract_sha256'][:12]}")
    if not identity["bytes_are_the_reviewed_bytes"]:
        line += (f" (reviewed {identity['reviewed_contract_sha256'][:12]}, "
                 "answer equal)")
    return (f"{line} plugin={identity['plugin_version']} "
            f"contract_v{identity['contract_version']}")


def contract_answer(contract: "TesseraContract") -> dict:
    """Exactly the values the gates on THIS pin decide on, canonicalised, and no more.

    Two gates read the dev pin: admission -- which ``(family, rate)`` the
    allocator may put on the menu and under which route status -- and the
    serve fingerprint's native-extension residency (prismaquant #133).  The
    answer is their union and nothing else.  It is deliberately NOT every
    value any PrismaQuant gate reads out of this file: ``lane_eligibility.structures``,
    ``platforms`` and ``regimes`` are read by the EXPORT lane's own reader
    (``tessera_export_lane.require_declared_structure`` through
    ``lane_eligibility.load_eligibility_table``), which is gated by the
    RELEASE pin -- an exact commit and sha, fail-closed today on PENDING
    sentinels.  Pulling them in here would make an export-lane edit re-stale
    the allocator's menu, which is issue #38's own failure mode wearing a
    different hat.  Each pin covers the values its own gates read.

    Principle 14's line, made mechanical.  ``detail``, ``rationale``, the
    changelog and every other prose field explains; none of them is a value a
    gate reads, so none of them appears here.  Neither do ``contract_version``,
    ``plugin_version`` or ``default_serve_image``: those are the table's *identity*,
    which travels into provenance, and a version bump that moved no answer is
    not a thing to re-review.

    What is here is what a GATE decides on.  Most of it is the admission
    decision -- which families exist, the rate range the decoder accepts for
    each, the rungs a cell attests, the tensor-parallel ceiling, the canonical
    ``quant_method`` this producer writes into the checkpoint, and every cell
    field the route gate reads.  Two contracts with the same answer admit the
    same units.

    Schema v4's executed launches and residency scope are retained per cell.
    They identify which serving path the route claim covers; a changed
    decoder or narrowed residency therefore requires the same review as a
    changed rung, even when the cell's id and route status stay unchanged.

    ``native_extensions`` is here for a second gate, not the admission one:
    §7.4 says an A/B's arms must have identical native-extension residency,
    ``tools/serve_fingerprint.py`` decides residency by matching mapped
    libraries, and the prefix, the glob and the ``match`` rule are the values
    that decision is made of -- with ``when_unavailable`` the value that says
    what an ABSENT library means (a named substitute decoder, or no serve at
    all).  Those move the fingerprint's behaviour, so they are answer.
    ``source`` and ``loaded_by`` name files and modules in the runtime's own
    tree and move nothing on this side, so they are identity and stay out,
    exactly like ``plugin_version``.  ``lane`` (contract v20) IS answer: its
    ``decoder`` is what binds a cell's launch to the extension, and its
    ``requires`` is the predicate ``native_cells`` and the export gate
    decide this producer's plan against -- a lane that widens or narrows
    what it reads changes which rungs are admitted, and is a re-review.

    ``fused_module`` is here for a third (prismaquant #132): the group
    knapsack's fold reads it, so a contract that re-tightened ``q256`` to
    ``shared`` would change what this producer may allocate.  It carries
    exactly the block's values this reader parses -- see
    :class:`FusedModuleLicence` -- which is why ``container`` is NOT here.
    Nothing on this side reads the sidecar's magic (there is no Tessera export
    leg to write one with), and a value in the answer that no gate reads makes
    a re-review out of a field nobody uses.  The block's three ``*_note`` keys
    are prose and stay out for the same reason ``rationale`` does.
    """
    return {
        "schema": TESSERA_CONTRACT_SCHEMA,
        "lane_schema": contract.lane_schema,
        **({"required_regimes": sorted(contract.regimes)}
           if contract.requires_serving_context else {}),
        "quant_method": contract.quant_method,
        "fused_module": contract.fused_module.answer(),
        "native_extensions": [
            {
                "module_name_prefix": ext.module_name_prefix,
                "filename_glob": ext.filename_glob,
                "match": ext.match,
                "routes": sorted(ext.routes),
                "when_unavailable": {
                    mode: {"status": behaviour["status"],
                           "decoder": behaviour["decoder"]}
                    for mode, behaviour in sorted(ext.when_unavailable.items())
                },
                "lane": ext.lane.answer(),
            }
            for ext in sorted(contract.native_extensions,
                              key=lambda e: e.module_name_prefix)
        ],
        "families": {
            family: {
                "reader_rate_range_q256": [int(rng[0]), int(rng[1])],
                "attested_rungs_q256": sorted(
                    int(r) for r in contract.attested_rungs.get(family, ())),
                "max_world_size": int(contract.max_world_size.get(family, 0)),
            }
            for family, rng in sorted(contract.reader_rate_range.items())
        },
        "cells": sorted(
            [
                cell.cell_id,
                cell.platform,
                cell.family,
                cell.structure,
                cell.regime,
                sorted(int(r) for r in cell.rungs_q256),
                cell.activation_contract,
                cell.route_status,
                cell.qualification,
                cell.requires_plugin,
                sorted(cell.requires_serve_flags),
                [list(launch) for launch in sorted(cell.executes)],
                sorted(cell.residency_modes),
            ] + ([{"image": cell.runtime_image,
                   "execution_modes": sorted(cell.execution_modes)}]
                 if contract.requires_serving_context else [])
            # v6's per-cell runtime versions and evidence. Both are ANSWER,
            # not identity: ``cell_evidence_admits`` decides on the evidence
            # block, so a flipped smoke status or a new KL kind changes which
            # units this producer may put on the menu -- and a re-review is
            # exactly what should stand between Tessera recording a smoke and
            # PrismaQuant admitting the route it names. The versions are here
            # for the same reason the image is: they scope the claim.
            + ([cell.runtime_vllm, cell.runtime_torch, cell.evidence.answer()]
               if cell.evidence is not None else [])
            for cell in contract.cells
        ),
    }


def _answer_drift(reviewed: Mapping[str, Any], installed: Mapping[str, Any]
                  ) -> list[str]:
    """Field-level lines naming what moved, so the refusal is reviewable."""
    lines: list[str] = []
    for key in ("schema", "lane_schema", "quant_method", "required_regimes"):
        if reviewed.get(key) != installed.get(key):
            lines.append(
                f"  {key}: reviewed {reviewed.get(key)!r}, installed "
                f"{installed.get(key)!r}")
    r_fused = dict(reviewed.get("fused_module", {}))
    i_fused = dict(installed.get("fused_module", {}))
    r_lic = dict(r_fused.pop("fields", {}) or {})
    i_lic = dict(i_fused.pop("fields", {}) or {})
    for key in sorted(set(r_fused) | set(i_fused)):
        if r_fused.get(key) != i_fused.get(key):
            lines.append(
                f"  fused_module.{key}: reviewed {r_fused.get(key)!r}, "
                f"installed {i_fused.get(key)!r}")
    for field in sorted(set(r_lic) | set(i_lic)):
        if field not in r_lic:
            lines.append(
                f"  fused_module.fields[{field}]: NEW ({i_lic[field]!r}), not "
                "in the reviewed answer")
        elif field not in i_lic:
            lines.append(
                f"  fused_module.fields[{field}]: GONE from the installed "
                f"contract (reviewed {r_lic[field]!r})")
        elif r_lic[field] != i_lic[field]:
            lines.append(
                f"  fused_module.fields[{field}]: reviewed "
                f"{r_lic[field]!r}, installed {i_lic[field]!r}")
    r_fam, i_fam = reviewed.get("families", {}), installed.get("families", {})
    for family in sorted(set(r_fam) | set(i_fam)):
        if family not in r_fam:
            lines.append(f"  families[{family}]: NEW, not in the reviewed answer")
        elif family not in i_fam:
            lines.append(f"  families[{family}]: GONE from the installed contract")
        elif r_fam[family] != i_fam[family]:
            for k in sorted(set(r_fam[family]) | set(i_fam[family])):
                if r_fam[family].get(k) != i_fam[family].get(k):
                    lines.append(
                        f"  families[{family}].{k}: reviewed "
                        f"{r_fam[family].get(k)!r}, installed "
                        f"{i_fam[family].get(k)!r}")
    r_ext = {row["module_name_prefix"]: row
             for row in reviewed.get("native_extensions", ())}
    i_ext = {row["module_name_prefix"]: row
             for row in installed.get("native_extensions", ())}
    for prefix in sorted(set(r_ext) | set(i_ext)):
        if prefix not in r_ext:
            lines.append(
                f"  native_extensions[{prefix}]: NEW, not in the reviewed "
                "answer -- a library a serve can map that no fingerprint on "
                "this side was reviewed against")
        elif prefix not in i_ext:
            lines.append(
                f"  native_extensions[{prefix}]: GONE from the installed "
                "contract")
        elif r_ext[prefix] != i_ext[prefix]:
            for key in sorted(set(r_ext[prefix]) | set(i_ext[prefix])):
                if r_ext[prefix].get(key) != i_ext[prefix].get(key):
                    lines.append(
                        f"  native_extensions[{prefix}].{key}: reviewed "
                        f"{r_ext[prefix].get(key)!r}, installed "
                        f"{i_ext[prefix].get(key)!r}")
    r_cells = {tuple(c[:1])[0]: c for c in reviewed.get("cells", ())}
    i_cells = {tuple(c[:1])[0]: c for c in installed.get("cells", ())}
    for cell_id in sorted(set(r_cells) | set(i_cells)):
        if cell_id not in r_cells:
            lines.append(f"  cells[{cell_id}]: NEW, not in the reviewed answer")
        elif cell_id not in i_cells:
            lines.append(f"  cells[{cell_id}]: GONE from the installed contract")
        elif list(r_cells[cell_id]) != list(i_cells[cell_id]):
            lines.append(
                f"  cells[{cell_id}]: reviewed {r_cells[cell_id]!r}, installed "
                f"{i_cells[cell_id]!r}")
    return lines


def dev_pin_requested() -> str:
    """The commit the environment asks for, or ``""``."""
    return str(os.environ.get(TESSERA_DEV_PIN_ENV, "")).strip()


def contract_path():
    """Where the packaged contract lives: the actually-importable package's table.

    ``importlib.resources.files("tessera.serving")`` rather than path
    arithmetic on ``tessera.__file__``, so a wheel install, an editable
    install and an in-repo checkout all resolve identically -- and so this
    reads the table of the ``tessera.serving`` package that is actually
    importable, never a copy.  This is the one resolver both producer readers
    share: ``tessera_render.tessera_serving_contract_path`` delegates here
    rather than carrying a second policy for the same file.

    That call does import the ``tessera.serving`` package, but the import is
    cheap by the runtime's own design: its ``__init__`` defines ``register()``
    (which imports vLLM and registers the config) and calls nothing at module
    scope, importing neither torch nor vLLM, so locating the contract
    registers nothing and needs no GPU.
    ``tests/test_tessera_serving_contract_path.py`` pins that property in a
    subprocess rather than in prose, so drift in the runtime's import
    behaviour fails a test instead of silently aging this docstring.  What a
    producer must not need is the serving-side *code*:
    ``tessera.serving.contract``'s validator imports the plugin's dispatch
    tables, which is a serving-side import a producer must not need on a
    machine with no GPU -- so the JSON is read directly rather than through
    that module.
    """
    from importlib import resources

    return resources.files("tessera.serving").joinpath("runtime_contract.json")


def _require(block: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in block:
        raise TesseraContractError(f"{where} publishes no {key!r}")
    return block[key]


#: The one ``native_extensions[].match`` rule this reader implements.  A
#: contract naming another rule is REFUSED rather than read with this one:
#: the whole reason ``match`` is a value is that the predicate is not
#: guessable from the glob.
MATCH_BASENAME_FNMATCH = "basename_fnmatch"

_NATIVE_EXTENSION_MEMBERS = (
    "module_name_prefix", "filename_glob", "match", "source", "loaded_by",
    "routes", "when_unavailable", "lane",
)


def _parse_native_extensions(
    entries: Any, *, where: str
) -> tuple[TesseraNativeExtension, ...]:
    """Read ``native_extensions``, refusing anything a fingerprint can't use.

    Published since Tessera contract v7.  An older table does not carry it,
    and the honest answer there is a refusal: "this contract does not say what
    the plugin loads" is not the same statement as "the plugin loads nothing",
    and reading the second from the first is how a Tessera serve came to
    fingerprint as a stock serve in the first place.
    """
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TesseraContractError(f"{where} must be a JSON array")
    if not entries:
        raise TesseraContractError(
            f"{where} is empty. A contract that publishes no loadable library "
            "makes every serve fingerprint identical on the one axis §7.4 "
            "keys reproducibility on, so an empty table is refused rather "
            "than read as 'loads nothing'."
        )
    parsed: list[TesseraNativeExtension] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        at = f"{where}[{i}]"
        if not isinstance(entry, Mapping):
            raise TesseraContractError(f"{at} must be a JSON object")
        for member in _NATIVE_EXTENSION_MEMBERS:
            _require(entry, member, at)
        prefix = str(entry["module_name_prefix"])
        if not prefix:
            raise TesseraContractError(
                f"{at}.module_name_prefix must be a non-empty string")
        if prefix in seen:
            raise TesseraContractError(
                f"{at}.module_name_prefix {prefix!r} is declared twice")
        seen.add(prefix)
        rule = str(entry["match"])
        if rule != MATCH_BASENAME_FNMATCH:
            raise TesseraContractError(
                f"{at}.match is {rule!r}; this reader implements only "
                f"{MATCH_BASENAME_FNMATCH!r} (fnmatch the glob against the "
                "BASENAME of a mapped .so) and will not apply that predicate "
                "to a rule it does not know. The rule is a published value "
                "precisely because it is not guessable from the glob."
            )
        glob = str(entry["filename_glob"])
        # By MEANING, not by spelling, the same way the runtime's own
        # validator checks it: a library name this load path can produce must
        # match. A glob that matches nothing a serve maps is a fingerprint
        # that reports every serve identical.
        if not fnmatch.fnmatch(f"{prefix}0123456789abcdef.so", glob):
            raise TesseraContractError(
                f"{at}.filename_glob {glob!r} matches no library name the "
                f"load path can produce ({prefix}<build identity>.so)"
            )
        when = entry["when_unavailable"]
        if not isinstance(when, Mapping) or not when:
            raise TesseraContractError(
                f"{at}.when_unavailable must be a non-empty object keyed by "
                "residency mode"
            )
        behaviours: dict[str, Mapping[str, Any]] = {}
        for mode, behaviour in when.items():
            if not isinstance(behaviour, Mapping):
                raise TesseraContractError(
                    f"{at}.when_unavailable[{mode!r}] must be an object")
            _require(behaviour, "status", f"{at}.when_unavailable[{mode!r}]")
            _require(behaviour, "decoder", f"{at}.when_unavailable[{mode!r}]")
            behaviours[str(mode)] = {
                "status": str(behaviour["status"]),
                "decoder": (None if behaviour["decoder"] is None
                            else str(behaviour["decoder"])),
            }
        routes = entry["routes"]
        if (not isinstance(routes, Sequence)
                or isinstance(routes, (str, bytes)) or not routes):
            raise TesseraContractError(
                f"{at}.routes must name at least one route that needs it")
        # The lane block is read by the serving pin's reader, so the two
        # contracts a producer holds -- the pinned one and this development
        # one -- refuse the same block for the same reason, once.
        try:
            lane = parse_lane_claim(entry["lane"], f"{at}.lane", extension=prefix)
        except LaneEligibilityError as exc:
            raise TesseraContractError(str(exc)) from exc
        parsed.append(TesseraNativeExtension(
            module_name_prefix=prefix,
            filename_glob=glob,
            match=rule,
            source=str(entry["source"]),
            loaded_by=str(entry["loaded_by"]),
            routes=tuple(str(r) for r in routes),
            when_unavailable=behaviours,
            lane=lane,
        ))
    return tuple(parsed)


def require_pin_native_extensions_match_contract(
    contract: "TesseraContract",
    pin: "Any | None" = None,
) -> None:
    """Refuse a pin whose extension table is not the pinned contract's.

    The middle link of the §7.4 chain.  ``tools/serve_fingerprint.py`` runs
    inside a serving container from a bootstrap with no installed package, so
    it can read neither this contract nor the pin's reader module: it reads
    the transported pin JSON beside itself -- a member of its gold-producer
    source closure -- and ``tests/test_tessera_serve_fingerprint.py`` refuses
    a tool that does not read the pin.  That test was the ONLY refusal in the
    chain, and the link it checked was the one that was already sound -- the
    pin itself was a hand-written claim about another runtime, maintained one
    repository over, where nothing here could refuse it on drift.  Rename the
    extension in the plugin and a Tessera serve fingerprinted as "nothing
    resident", which is the hole that existed before 2026-09-03, when the
    pattern named no Tessera library at all.

    Compared over the fields a residency predicate -- and the reading of an
    absent library -- is MADE of -- prefix, glob, the match rule, and what
    runs when the library is absent (``when_unavailable``) -- keyed by prefix,
    in both directions: a library the contract publishes and the pin omits
    makes the fingerprint go quietly short, and a library the pin invents is
    a claim about a runtime that does not load it.  A substitute decoder the
    pin mistranscribes names the wrong fallback in the §7.4 refusal, so the
    block is compared value for value, not merely for presence.
    """
    from .tessera_serving_runtime_pin import (
        TesseraServingRuntimePinError,
        load_tessera_serving_runtime_pin,
    )

    if pin is None:
        pin = load_tessera_serving_runtime_pin()
    published = {row["module_name_prefix"]: row
                 for row in (ext.as_contract_row()
                             for ext in contract.native_extensions)}
    pinned = {row["module_name_prefix"]: row
              for row in pin.native_extension_rows()}
    lines: list[str] = []
    for prefix in sorted(set(published) | set(pinned)):
        if prefix not in pinned:
            lines.append(
                f"  {prefix!r}: the contract publishes it and the pin omits "
                "it -- a serve loading it would fingerprint as nothing "
                "resident")
        elif prefix not in published:
            lines.append(
                f"  {prefix!r}: the pin declares it and the contract "
                "publishes no such extension")
        elif published[prefix] != pinned[prefix]:
            for key in sorted(set(published[prefix]) | set(pinned[prefix])):
                if published[prefix].get(key) != pinned[prefix].get(key):
                    lines.append(
                        f"  {prefix!r}.{key}: contract "
                        f"{published[prefix].get(key)!r}, pin "
                        f"{pinned[prefix].get(key)!r}")
    if lines:
        raise TesseraServingRuntimePinError(
            "The Tessera serving pin's serving_native_extensions is not the "
            f"pinned contract's native_extensions table ({contract.path}):\n"
            + "\n".join(lines) + "\n"
            "The pin is a TRANSCRIPTION of that table (principle 14: a claim "
            "about another runtime is attested, never asserted), and "
            "tools/serve_fingerprint.py carries the same rows because it "
            "cannot read either file from inside a serving container. Fix the "
            "pin, and the tool's TESSERA_NATIVE_EXTENSIONS with it, in one "
            "commit."
        )

def _parse_fused_module(payload: Mapping[str, Any], path: str
                        ) -> FusedModuleLicence:
    """Read ``fused_module``, or refuse.

    Required, not defaulted.  A contract without this block has published
    nothing about what a fused module's roles may disagree on, and the two
    ways to default it are both assertions: "everything shared" would invent a
    constraint this runtime does not state, and "the rate is free" is the exact
    prose claim reading the block exists to replace.

    Only the values a gate on this side decides on are kept, which is what
    makes :meth:`FusedModuleLicence.answer` the whole block's projection.
    ``container`` is read past deliberately: there is no Tessera export leg to
    write a sidecar with, so nothing here consumes the magic, and parsing it
    would put a field nobody uses into the pin's answer and make a re-review
    out of it.  The day a writer reads it, it joins the dataclass and the
    answer in one commit.
    """
    where = f"{path}.fused_module"
    block = _require(payload, "fused_module", path)
    if not isinstance(block, Mapping):
        raise TesseraContractError(f"{where} must be a JSON object")
    schema = block.get("schema")
    if schema != FUSED_MODULE_SCHEMA:
        raise TesseraContractError(
            f"{where}.schema must be {FUSED_MODULE_SCHEMA!r}, got {schema!r}. "
            "A block with another id is not a subset of this one, so it is "
            "refused rather than partially read."
        )
    fields = _require(block, "fields", where)
    if not isinstance(fields, Mapping) or not fields:
        raise TesseraContractError(
            f"{where}.fields must be a non-empty object mapping a field name "
            "to its licence"
        )
    parsed: dict[str, str] = {}
    for name, licence in fields.items():
        if licence not in FUSED_MODULE_FIELD_LICENCES:
            raise TesseraContractError(
                f"{where}.fields[{name!r}] is {licence!r}; this reader knows "
                f"only {sorted(FUSED_MODULE_FIELD_LICENCES)} and will not "
                "guess at a third licence's meaning -- a token it mapped onto "
                "'per_member' would widen a group allocator's menu on a word "
                "it did not understand"
            )
        parsed[str(name)] = str(licence)
    receipt = _require(block, "mixed_rung_receipt", where)
    if not isinstance(receipt, bool):
        raise TesseraContractError(
            f"{where}.mixed_rung_receipt must be a JSON boolean, got "
            f"{receipt!r}. It is the difference between "
            "\"a serve has covered a mixed-rung module\" and \"a decode "
            "identity has\", and a truthy string answers neither question."
        )
    return FusedModuleLicence(
        schema=str(schema),
        fields=parsed,
        sidecar_q256=str(_require(block, "sidecar_q256", where)),
        mixed_rung_receipt=receipt,
    )



def _parse(payload: Mapping[str, Any], *, commit: str, sha: str, path: str
           ) -> TesseraContract:
    schema = payload.get("schema")
    if schema != TESSERA_CONTRACT_SCHEMA:
        raise TesseraContractError(
            f"{path}: schema must be {TESSERA_CONTRACT_SCHEMA!r}, got "
            f"{schema!r}. An older contract is not a subset of this one, so it "
            "is refused rather than partially read."
        )
    reader_range: dict[str, tuple[int, int]] = {}
    attested: dict[str, frozenset[int]] = {}
    formats = _require(payload, "formats", path)
    if not isinstance(formats, Sequence) or isinstance(formats, (str, bytes)):
        raise TesseraContractError(f"{path}.formats must be a JSON array")
    for i, entry in enumerate(formats):
        if not isinstance(entry, Mapping):
            raise TesseraContractError(f"{path}.formats[{i}] must be an object")
        where = f"{path}.formats[{i}]"
        kind = str(_require(entry, "kind", where))
        if kind != "tessera_wire":
            raise TesseraContractError(
                f"{where}.kind is {kind!r}; this reader knows only "
                "'tessera_wire' and will not guess at another kind's rung "
                "vocabulary"
            )
        family = str(_require(entry, "family", where))
        lo, hi = (int(v) for v in _require(entry, "reader_rate_range_q256", where))
        reader_range[family] = (lo, hi)
        # ``attested_rungs_q256`` is the field's name since Tessera contract
        # v2; ``candidate_rungs_q256`` is the deprecated alias it kept so the
        # rename stayed additive, and Tessera's own reader refuses the two if
        # they disagree.  Read the current name first so this reader survives
        # the alias being dropped, and accept the alias alone so it can still
        # read a v1 table.  Reading the alias *preferentially* is how the gap
        # the rename closed would reopen: the alias never was the decodable
        # set.
        rungs = entry.get("attested_rungs_q256", entry.get("candidate_rungs_q256"))
        if rungs is None:
            raise TesseraContractError(
                f"{where} publishes no 'attested_rungs_q256' (nor its "
                "deprecated alias 'candidate_rungs_q256')"
            )
        attested[family] = frozenset(int(r) for r in rungs)

    # The extension table is read BEFORE the lane table because the lane
    # table's cells launch through it: its own refusals (no table, an empty
    # one, a match rule this reader cannot apply, a lane block it cannot
    # decide) come first and by their own names.
    extensions = _parse_native_extensions(
        _require(payload, "native_extensions", path),
        where=f"{path}.native_extensions",
    )

    lane = _require(payload, "lane_eligibility", path)
    if not isinstance(lane, Mapping):
        raise TesseraContractError(f"{path}.lane_eligibility must be an object")
    lane_schema = lane.get("schema")
    if lane_schema not in TESSERA_LANE_SCHEMAS:
        raise TesseraContractError(
            f"{path}.lane_eligibility.schema must be one of {sorted(TESSERA_LANE_SCHEMAS)!r}, "
            f"got {lane_schema!r}"
        )
    try:
        # The lane table is read beside the extension table it launches
        # through (contract v20): the reader binds every extension launch a
        # cell names to that extension's lane. The predicate itself is NOT
        # decided here -- loading a contract imports no encoder and no
        # serving code; admission (``native_cells``) does.
        table = _parse_table(lane, formats, "", commit, sha,
                             native_extensions=payload["native_extensions"])
    except LaneEligibilityError as exc:
        raise TesseraContractError(f"{path}: {exc}") from exc
    cells: list[TesseraRouteCell] = []
    for cell in table.cells:
        if cell.predicates:
            raise TesseraContractError(
                f"{path}.lane_eligibility.cells[{cell.id!r}].predicates "
                "cannot be evaluated by shape-free development admission; "
                "a constrained route must not enter an unconditional menu. "
                "Pass structural facts through admission before accepting "
                "predicated cells."
            )
        cells.append(TesseraRouteCell(
            cell_id=cell.id,
            platform=cell.platform,
            family=cell.family,
            structure=cell.structure,
            regime=cell.regime,
            rungs_q256=frozenset(cell.rungs_q256),
            activation_contract=cell.activation_contract,
            route_status=cell.route_status,
            qualification=cell.qualification,
            requires_plugin=cell.requires_plugin,
            requires_serve_flags=cell.requires_serve_flags,
            executes=cell.executes,
            residency_modes=cell.residency_modes,
            runtime_image=cell.runtime_image,
            execution_modes=cell.execution_modes,
            runtime_vllm=cell.runtime_vllm,
            runtime_torch=cell.runtime_torch,
            evidence=cell.evidence,
        ))

    tp = _require(payload, "tensor_parallel", path)
    if str(tp.get("semantics")) != "closed_world":
        raise TesseraContractError(
            f"{path}.tensor_parallel.semantics is "
            f"{tp.get('semantics')!r}; this reader treats the block as a closed "
            "world (a family absent from it is not attested at any degree) and "
            "will not read an open-world table under that assumption"
        )
    world: dict[str, int] = {}
    for i, unit in enumerate(tp.get("units", ())):
        where = f"{path}.tensor_parallel.units[{i}]"
        world[str(_require(unit, "unit", where))] = int(
            _require(unit, "max_world_size", where))

    fused = _parse_fused_module(payload, path)

    versions = payload.get("versions", {})
    method = payload.get("quant_method", {})
    return TesseraContract(
        reader_rate_range=reader_range,
        attested_rungs=attested,
        cells=tuple(cells),
        native_extensions=extensions,
        max_world_size=world,
        fused_module=fused,
        quant_method=str(method.get("canonical", "")),
        contract_version=int(payload.get("contract_version", 0)),
        plugin_version=str(versions.get("tessera", "")),
        default_serve_image=str(versions.get("default_serve_image", "")),
        commit=commit,
        sha256=sha,
        path=path,
        lane_schema=table.schema,
        regimes=table.regimes,
    )


@lru_cache(maxsize=8)
def _load_at(path: str, sha: str, commit: str) -> TesseraContract:
    """Parse one contract file.  Keyed on the SHA, so a changed file re-reads.

    ``route_admission``'s docstring explains why a cache over a runtime
    contract is a defect when its key cannot state the contract's identity.
    This key is exactly that identity, so the cache is safe: edit the file and
    the sha changes, which is a different key.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _parse(payload, commit=commit, sha=sha, path=path)


def load_tessera_contract() -> "TesseraContract | None":
    """The packaged Tessera contract under the dev pin, or ``None``.

    ``None`` means the pin is not requested, which is production: no Tessera
    route is attested and the attested menu is empty.  Every other failure --
    a contract whose *answer* is not the reviewed one, a missing or malformed
    file -- raises.  A mismatch never degrades to "unattested"; that would turn
    a stale pin into a silently empty menu.
    """
    requested = dev_pin_requested()
    if not requested:
        return None
    from importlib.resources import as_file

    with as_file(contract_path()) as path:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise TesseraContractError(
                f"cannot read the packaged Tessera contract at {path}: {exc}"
            ) from exc
        sha = hashlib.sha256(raw).hexdigest()
        contract = _load_at(str(path), sha, TESSERA_DEV_PIN_COMMIT)
        drift = _answer_drift(TESSERA_DEV_PIN_ANSWER, contract_answer(contract))
        if drift:
            raise TesseraContractError(
                "Tessera moved and its answer moved with it -- re-review the pin.\n"
                f"The pin in {__name__} was reviewed against Tessera "
                f"{TESSERA_DEV_PIN_COMMIT} (contract sha256 "
                f"{TESSERA_DEV_PIN_CONTRACT_SHA256}); {path} hashes to {sha} and "
                "publishes a different answer:\n" + "\n".join(drift) + "\n"
                "This is not a corruption warning. Read what moved, decide whether "
                "PrismaQuant should admit it, and update TESSERA_DEV_PIN_ANSWER "
                "(with the commit and sha) in the same commit -- that diff is the "
                "review."
            )
        # The middle link of the §7.4 chain, checked wherever both objects are in
        # hand.  It is deliberately AFTER the answer check: a moved answer is the
        # more informative refusal, and the pin cannot be judged against a
        # contract this producer has not accepted.
        require_pin_native_extensions_match_contract(contract)
        return contract
