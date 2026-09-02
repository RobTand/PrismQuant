#!/usr/bin/env python
"""Bank accepted burn shards + books for the ON-LAW routed rungs (K28/K32).

WHY THIS EXISTS
---------------
The 112.69 GB DSv4-Flash allocation puts routed experts on FP8_CB_K28/K32.
Those are the only two rungs that are simultaneously

  * ON-LAW      -- k % 4 == 0, the gridbook K1.2 fused mid-M prefill law, whose
                   `type_size = 4k` packed-B TMA box needs a 16-byte multiple.
                   A uniform decode at an off-law rung is WRONG, not merely
                   unaligned.
  * BANKABLE    -- cb_banked_books.ROUTED_MOE_CBL_BANK_RUNGS caps routed books
                   at K28-K33 (the byte-exact source-payload ceiling).

Rendering a routed expert at rung K requires an ACCEPTED BURN SHARD whose
`content_guard` digests match the tensors the exporter is about to render:
`load_banked_cbl_book` never searches a directory, never trains, and never
falls back to a lattice.  A Lloyd book file on disk is NOT sufficient -- the
shard is the contract, the book is what the shard points at.

The a-fast burn banked full 129/129 coverage only at its two interpolation
anchors K29 and K35, and BOTH are off-law (29%4=1, 35%4=3).  Every other rung's
*cost* came from the law `logy = a0 + a1*K + phi_{K%4}`, and an interpolated
cost is not a book.  On-law coverage as measured 2026-08-11:

    K28: 27/129 accepted cells  ->  102 missing
    K32: 29/129 accepted cells  ->  100 missing

WHAT THIS TOOL DOES NOT DO
--------------------------
It does not measure allocator costs.  Costs come from the anchored-AURA
supersurrogate (probe of the underlying model, mapped onto formats through the
fitted error terms) -- this is deliberately NOT a measure-and-burn campaign.
This tool exists only to make the on-law rungs RENDERABLE.

WHY ONE RUNG PER CHAIN
----------------------
`_run_chain` over multiple rungs feeds each rung's reconstruction forward as the
next rung's `embed` (minchain) predecessor, and the per-expert winner is then
whichever arm scores lower weight_mse.  A cell whose winner is `embed` does not
carry the certified `cbl_poolb` measurement stamp, and `load_banked_cbl_book`
rejects it.  Running each rung as its OWN single-rung chain leaves no
predecessor, so the free (CBL) arm is the only arm and every banked cell is the
certified pooled-Lloyd product book.  The weights are loaded once per
(layer, projection) and reused across both rungs, so this costs I/O nothing.

CALIBRATION REBASE (--col-weights / --act-root)
-----------------------------------------------
A book is learned FROM an imatrix and an activation cache, so the calibration is
part of its identity, not a setting: `_book_key` hashes `col_weights_digest`,
and `load_banked_cbl_book` refuses a shard whose digest differs from the role
tensor the exporter is holding.

The a-fast burn ran under `prod-cal-0p6` (md5 fe1492f6).  This artifact's
campaign is pinned to `prod-cal-0p7` -- `run_aura_cb_reprice.sh` hard-codes
`--probe prod-cal-0p7/artifacts/probe.pkl` and defaults `CB_COL_WEIGHTS` to the
0p7 imatrix (md5 746f76fa), which is also what the dense learned bundle was
built from.  0p7 is the stronger calibration besides: 51 unrouted-expert
neutral-prior names against 0p6's thousands.  So the routed books are the one
thing on the wrong side of the campaign's calibration, and they get re-burned
rather than the campaign getting re-based around them.

Rebasing needs both halves and refuses either alone, because supplying one
silently learns books from one calibration's activations under another's
weighting:

  * `--col-weights`  the imatrix the exporter will hold;
  * `--act-root`     the activation cache it was harvested from.

The verified by-layer store stays the SOURCE-WEIGHT identity oracle -- those
digests are of the model's own FP8 tensors and no calibration moves them -- but
its imatrix fields are restated from the tensors actually in hand, and nothing
else in the identity is allowed to move.  The proof this was done right is a
check already in `load_projection`: it asserts `x.square().mean(0) == cw` per
qname, so a mismatched (imatrix, activations) pair aborts at layer 0 instead of
producing 258 quietly-wrong books.

RESUME
------
`_run_chain` validates and reuses any existing cell whose identity still
matches, so re-running is safe and skips completed work.  Cells are keyed on the
full 256-expert stack (`expert_ids=all_experts`, `full_encode_rungs=rung`),
which is exactly the digest the exporter computes.  Books are content-addressed
under `bucket-books/` by a key that includes the imatrix digest, so a rebased
burn writes new addresses and can never overwrite the 0p6 generation.

KEYING (--keying stack | role)
------------------------------
Campaign rule R1: routed learned books are burned per `(layer, STACK, rung)`,
so the fused `gate_up_proj` population gets ONE book covering gate and up, and
the exported fused weight names a single codebook.  That is the default here,
as it is for `build_cb_learned_bundle`.  `--keying role` reproduces the
pre-R1 per-`(layer, projection, rung)` form for the A/B arm.

A stack cell is the exact population the bundle builder's `provide_weight`
materializes: every expert's gate rows followed by its up rows, in the
profile's declared fused order, weighted by the packed target's OWN imatrix
entry reshaped `(E, 1, in)` -- read through the builder's `_stack_col_weights`
so burn and bundle cannot disagree about the reshape.  The per-expert-Linear
harvest carries no packed entry, so `tools/dsv4_packed_col_weights.py` writes
one with the export's own derivation first; a pickle without it refuses here,
exactly as the bundle build would.  `down_proj` is a one-projection stack:
its stack cell and its role cell are the same tensors under the same name, so
a role-keyed bank already satisfies the stack request for it.

The shard root is per CALIBRATION generation, not per keying: a stack cell is
named by its population (`layer_NNN_gate_up_proj_...`) and cannot collide
with a role cell, and the `down_proj` cells are shared between the two forms
by construction.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import torch

# The single source of truth for the canonical imatrix identity triple.  It is
# private, and imported anyway on purpose: re-deriving the byte framing here is
# how a producer and a consumer drift into disagreeing about the same tensors.
# The same reasoning imports the bundle builder's stack-imatrix reshape and the
# export's fused-projection order: the burn must hash exactly the tensors the
# consumers will hold.
from prismaquant.build_cb_learned_bundle import _stack_col_weights
from prismaquant.export_nvfp4_cb_streaming import (
    _packed_expert_col_weights,
    _packed_expert_projection_names,
)
from prismaquant.production_weight_cache import (
    _canonical_cb_col_weights_identity,
    validate_cb_render_identity_metadata,
)
from prismaquant.routed_moe_codebooks import (
    DEFAULT_ROUTED_BOOK_KEYING,
    ROUTED_BOOK_KEYINGS,
    ROUTED_BOOK_KEYING_STACK,
    ROUTED_ROLE_PROJECTIONS,
    ROUTED_STACK_KEYS,
    normalize_routed_book_keying,
)
from tools import dsv4_afast_burn as burn
from tools import dsv4_ldlq_cost_campaign as ldlq_campaign
from tools.dsv4_afast_campaign import load_layer_identity, load_projection

# The two on-law, bankable routed rungs.  Not configurable by accident: any
# other value is either off-law (breaks the fused mid-M decode) or outside
# ROUTED_MOE_CBL_BANK_RUNGS (unbankable).  --rungs exists to burn them one at a
# time, never to introduce a third.
ONLAW_ROUTED_RUNGS = (28, 32)
# The role-keyed population list; `populations()` is the keying-aware form.
PROJECTIONS = ROUTED_ROLE_PROJECTIONS
# The fused order the export's routed-expert ABI requires; the profile must
# declare exactly this (export_nvfp4_cb_streaming asserts the same thing).
_ABI_STACK_PROJECTIONS = {
    "gate_up_proj": ("gate_proj", "up_proj"),
    "down_proj": ("down_proj",),
}


def populations(keying: str) -> tuple[str, ...]:
    """The burn cells' ``projection`` spellings under *keying*."""

    keying = normalize_routed_book_keying(keying)
    if keying == ROUTED_BOOK_KEYING_STACK:
        return tuple(ROUTED_STACK_KEYS)
    return tuple(ROUTED_ROLE_PROJECTIONS)


def population_members(key: str, keying: str, profile) -> tuple[str, ...]:
    """The role projections one burn cell pools, in fused row order."""

    keying = normalize_routed_book_keying(keying)
    if keying != ROUTED_BOOK_KEYING_STACK:
        if key not in ROUTED_ROLE_PROJECTIONS:
            raise ValueError(f"{key!r} is not a routed role projection")
        return (key,)
    if key not in ROUTED_STACK_KEYS:
        raise ValueError(f"{key!r} is not a routed stack key")
    declared = tuple(_packed_expert_projection_names(profile, key))
    expected = _ABI_STACK_PROJECTIONS[key]
    if declared != expected:
        raise ValueError(
            f"{key}: Gridbook routed expert ABI expects {expected}, profile "
            f"declared {declared}"
        )
    return declared


def packed_qname(layer: int, key: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{key}"


def load_population(
    layer: int,
    key: str,
    *,
    keying: str,
    profile,
    device: torch.device,
    identity,
    all_col_weights,
    model_to_shard,
    model_to_ckpt,
    scale_map,
) -> dict:
    """`load_projection` for one burn population under *keying*.

    A one-projection population is `load_projection` unchanged.  A fused stack
    is every member loaded (and validated) through the same function, its
    weights concatenated along the output rows per expert in the declared
    fused order, and its imatrix the packed target's own entry through the
    bundle builder's reshape.  The entry must equal the export's per-expert
    mean derivation of the member vectors it sits beside: the pooled arm must
    weight the fused rows by the same per-expert statistics the per-role
    arm's export applies, or the A/B is confounded by its imatrix.
    """

    members = population_members(key, keying, profile)
    parts = [
        load_projection(
            layer, projection, device=device, identity=identity,
            all_col_weights=all_col_weights, model_to_shard=model_to_shard,
            model_to_ckpt=model_to_ckpt, scale_map=scale_map,
        )
        for projection in members
    ]
    if len(parts) == 1:
        return parts[0]

    qname = packed_qname(layer, key)
    experts = int(parts[0]["weight"].shape[0])
    in_features = int(parts[0]["weight"].shape[2])
    for projection, part in zip(members, parts):
        shape = tuple(int(dim) for dim in part["weight"].shape)
        if shape[0] != experts or shape[2] != in_features:
            raise ValueError(
                f"{qname}: {projection} stack {shape} does not fuse with "
                f"E={experts}, in={in_features}"
            )
    if qname not in all_col_weights:
        raise SystemExit(
            f"REFUSE: {qname} has no packed col_weights entry; a stack-keyed "
            "book is burned against the packed target's own imatrix (the "
            "bundle builder refuses the same pickle). Write it with "
            "tools/dsv4_packed_col_weights.py and pass that pickle as "
            "--col-weights."
        )
    col = _stack_col_weights(
        qname, experts=experts, in_features=in_features,
        col_weights=all_col_weights,
    ).to(torch.float32).contiguous()
    member_map = {
        (projection, expert): part["qnames"][expert]
        for projection, part in zip(members, parts)
        for expert in range(experts)
    }
    derived = _packed_expert_col_weights(
        {member: all_col_weights[member] for member in member_map.values()},
        {qname: member_map},
        profile,
    )[qname].to(torch.float32).contiguous()
    if tuple(derived.shape) != tuple(col.shape) or not torch.equal(derived, col):
        raise SystemExit(
            f"REFUSE: {qname}: packed col_weights entry (shape "
            f"{tuple(col.shape)}) is not the export's per-expert mean of its "
            f"{members} member vectors; the pooled book would be learned "
            "under a different imatrix spelling than the fused render"
        )
    weight = torch.cat([part["weight"] for part in parts], dim=1).contiguous()
    for part in parts:
        del part["weight"]
    observed = sum(int(part["observed_activation_files"]) for part in parts)
    cold = sorted({expert for part in parts for expert in part["cold_experts"]})
    return {
        "qnames": [name for part in parts for name in part["qnames"]],
        "weight": weight,
        "col_weights": col.to(device).contiguous(),
        # Books are learned from weights and the imatrix only (replay=False);
        # the per-member activation rows stay with their roles.
        "activation_rows": None,
        "observed_activation_files": observed,
        "cold_experts": cold,
    }

# Exactly the fields `_canonical_cb_col_weights_identity` produces.  A
# calibration rebase may restate these three and nothing else.
_IMATRIX_IDENTITY_FIELDS = [
    "col_weights_content_sha256",
    "col_weights_sha256",
    "col_weights_shapes",
]


def acceptable_cell(path: Path) -> bool:
    """True when this existing shard can back a routed book as-is.

    `load_banked_cbl_book` validates the pass-tag SCHEMA, not the pass-tag
    value, so a scout/primary/backstop cell is as good as a full-layer one
    provided it clears the two things the loader actually checks:

      * the content_guard is keyed on the FULL 256-expert stack -- the a-fast
        campaign full-encoded only its `full_encode_rungs`, so a cell at an
        off-schedule rung may have been written under a sampled expert slice;
      * the measurement is the certified `cbl_poolb` arm.  Cells carrying
        `incumbent_sweep_noldlq` (the disabled-rung lattice policy) or no
        semantics stamp at all exist at these rungs and must be re-burned.
    """
    if not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return False
    identity = payload.get("identity") or {}
    guard = identity.get("content_guard") or {}
    shape = guard.get("source_shape") or []
    if len(shape) != 3 or int(shape[0]) != burn.EXPERT_COUNT:
        return False
    if len(identity.get("encoded_expert_ids") or []) != burn.EXPERT_COUNT:
        return False
    semantics = (
        (payload.get("cell") or {}).get("timing") or {}
    ).get("measurement_semantics") or {}
    return (
        semantics.get("encoder") == "cbl_poolb"
        and semantics.get("ldlq") is False
        and semantics.get("scale_policy") == "one_shot_cand0"
    )


def accepted_shard(layer: int, projection: str, rung: int) -> Path | None:
    """The shard that can back this cell today, across every pass tag."""
    for tag in burn.BURN_PASS_TAGS.values():
        path = burn._burn_cell_path(layer, projection, tag, rung)
        if acceptable_cell(path):
            return path
    return None


def _missing_rungs(layer: int, projection: str, rungs) -> tuple[int, ...]:
    """Rungs with no acceptable cell yet for this (layer, projection)."""
    return tuple(
        rung for rung in rungs
        if accepted_shard(layer, projection, rung) is None
    )


def _rebased_identity(layer: int, all_col_weights) -> dict:
    """Layer identity whose imatrix fields describe the imatrix in hand.

    `load_layer_identity` runs FIRST and with the store's own activation root,
    because it validates the by-layer cost store as the artifact it is: its
    `meta` records the cache it was built against, and a store that fails its
    own provenance must not be trusted for source digests either.  Only after
    it has passed are the three imatrix fields restated.

    The restatement is bounded by an equality check on every other key, which
    is the `rekey_shards.py` discipline: restate exactly the fields whose
    inputs deliberately changed, and abort if anything else moved.  Re-running
    the full metadata validator against the new tensors is what makes the
    result an identity rather than an edit.
    """
    _, verified = load_layer_identity(layer)
    identity = verified["identity"]
    digest, shapes, content = _canonical_cb_col_weights_identity(
        all_col_weights, identity["col_weights_qnames"],
    )
    rebased = dict(identity)
    rebased["col_weights_sha256"] = digest
    rebased["col_weights_shapes"] = shapes
    rebased["col_weights_content_sha256"] = content
    # A SUBSET is expected, not the whole set: `col_weights_shapes` is a
    # property of the model, so two imatrices over the same tensors restate it
    # unchanged.  What must never happen is a key outside the set moving.
    escaped = sorted(
        key for key in set(identity) | set(rebased)
        if identity.get(key) != rebased.get(key)
        and key not in _IMATRIX_IDENTITY_FIELDS
    )
    if escaped:
        raise AssertionError(
            f"layer {layer}: calibration rebase moved {escaped}, which is "
            f"outside the imatrix fields {_IMATRIX_IDENTITY_FIELDS}"
        )
    validate_cb_render_identity_metadata(
        rebased,
        col_weights=all_col_weights,
        require_source_complete=True,
        where=f"DSV4 on-law book burn layer {layer} (rebased imatrix)",
    )
    return rebased


def resolve_identities(layers, all_col_weights, *, act_root=None) -> dict:
    """Rebased identities for `layers`, THEN install the burn activation root.

    The ordering is the whole function.  `load_layer_identity` asserts the
    by-layer store was built against the current `ACT_ROOT`, so every identity
    has to be resolved while that is still the store's own cache; the burn's
    root is installed once afterwards, one-way, for `load_projection`.  Callers
    that did this by hand would get it right until one of them didn't.
    """
    identities = {
        layer: _rebased_identity(layer, all_col_weights)
        for layer in sorted(set(layers))
    }
    if act_root is not None:
        ldlq_campaign.ACT_ROOT = Path(act_root)
    return identities


def _burn_cell(*, layer, projection, data, rung) -> dict:
    """One single-rung chain -> one accepted, full-stack-keyed CBL cell."""
    cells = burn._run_chain(
        layer=layer,
        projection=projection,
        pass_tag=burn.BURN_PASS_TAGS["full_layer"],
        data=data,
        rungs=(rung,),
        expert_ids=tuple(range(burn.EXPERT_COUNT)),
        replay=False,
        full_encode_rungs=(rung,),
    )
    return cells[rung]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rungs", default=",".join(map(str, ONLAW_ROUTED_RUNGS)),
        help="comma-separated subset of the on-law routed rungs",
    )
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=burn.LAYER_COUNT)
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report the work list and exit without touching the GPU",
    )
    # See CALIBRATION REBASE above.  Both halves or neither: an imatrix without
    # its activations weights one calibration's rows by another's second moment,
    # and `load_projection`'s x.square().mean(0) == cw assertion would be the
    # only thing standing between that and 258 wrong books.
    parser.add_argument(
        "--col-weights", default=None,
        help="imatrix pickle the exporter will hold; requires --act-root",
    )
    parser.add_argument(
        "--act-root", default=None,
        help="activation cache that imatrix was harvested from",
    )
    # A separate shard root per calibration generation. Without it the resume
    # scan would happily reuse cells burned under the other imatrix -- they are
    # structurally valid and differ only in a digest the scan does not check.
    parser.add_argument(
        "--shard-root", default=None,
        help="burn-cell directory; defaults to the campaign's burn-shards",
    )
    # See KEYING above.  The default is campaign rule R1, the same default
    # build_cb_learned_bundle applies; the per-role form is the A/B arm.
    parser.add_argument(
        "--keying", choices=ROUTED_BOOK_KEYINGS,
        default=DEFAULT_ROUTED_BOOK_KEYING,
        help="burn one book per (layer, stack, rung) [stack, R1] or per "
             "(layer, projection, rung) [role, the A/B arm]",
    )
    args = parser.parse_args()

    keying = normalize_routed_book_keying(args.keying)
    if bool(args.col_weights) != bool(args.act_root):
        raise SystemExit(
            "REFUSE: --col-weights and --act-root are one calibration and must "
            "be given together; supplying either alone learns books from one "
            "calibration's activations under another's weighting"
        )
    if args.shard_root:
        burn.BURN_CELL_ROOT = Path(args.shard_root)
        burn.BURN_CELL_ROOT.mkdir(parents=True, exist_ok=True)
    if args.col_weights:
        burn.COL_WEIGHTS = Path(args.col_weights)

    # Default the manifest INTO the shard root so a rebased generation cannot
    # append into the previous one's log and read back as one burn.
    manifest_path = Path(args.manifest) if args.manifest else (
        burn.BURN_CELL_ROOT / "ONLAW_BOOK_BURN.jsonl" if args.shard_root
        else burn.RUN_ROOT / "onlaw-books" / "ONLAW_BOOK_BURN.jsonl"
    )

    rungs = tuple(int(r) for r in args.rungs.split(",") if r.strip())
    illegal = [r for r in rungs if r not in ONLAW_ROUTED_RUNGS]
    if illegal:
        raise SystemExit(
            f"REFUSE: {illegal} is not an on-law bankable routed rung; "
            f"legal set is {list(ONLAW_ROUTED_RUNGS)} (k%4==0 AND <=K33)"
        )

    from tools import dsv4_cbl_kernels as cblk
    not_cbl = [r for r in rungs if not cblk.cbl_eligible(r)]
    if not_cbl:
        raise SystemExit(
            f"REFUSE: K{not_cbl} would dispatch the incumbent encoder, whose "
            "cells are not bankable (load_banked_cbl_book accepts cbl_poolb "
            "only)"
        )

    layers = range(args.start_layer, min(args.end_layer, burn.LAYER_COUNT))
    keys = populations(keying)
    worklist = [
        (layer, key, missing)
        for layer in layers
        for key in keys
        if (missing := _missing_rungs(layer, key, rungs))
    ]
    total_cells = sum(len(m) for _, _, m in worklist)
    print(
        f"[onlaw-books] keying={keying} populations={list(keys)}\n"
        f"[onlaw-books] {total_cells} cells to burn across "
        f"{len(worklist)} (layer, population) loads; rungs={list(rungs)}",
        flush=True,
    )
    if args.dry_run:
        for layer, projection, missing in worklist[:12]:
            print(f"    L{layer:02d} {projection}: K{list(missing)}")
        if len(worklist) > 12:
            print(f"    ... {len(worklist) - 12} more")
        return 0
    if not worklist:
        print("[onlaw-books] nothing to do; every on-law cell is banked")
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("REFUSE: on-law book burn requires CUDA")
    burn._configure()
    # `_configure()` is tuned for the a-fast COST burn, which runs the
    # throughput LDLQ route (16 feeder threads).  These books are canonical
    # packed ARTIFACTS -- the exporter renders through them -- and
    # `_validate_packed_ldlq_route_env` refuses every byte-changing packed
    # route, feeder threads included.  Overriding after `_configure()` so the
    # artifact ABI wins over the measurement tuning; the other three knobs
    # (batch-experts on, expert-batch 16, one stream) are already canonical.
    os.environ["PRISMAQUANT_CB_LDLQ_FEEDER_THREADS"] = "0"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with burn.COL_WEIGHTS.open("rb") as handle:
        all_col_weights = pickle.load(handle)

    identities = resolve_identities(
        {layer for layer, _, _ in worklist},
        all_col_weights,
        act_root=args.act_root,
    )
    print(
        f"[onlaw-books] imatrix={burn.COL_WEIGHTS}\n"
        f"[onlaw-books] activations={ldlq_campaign.ACT_ROOT}\n"
        f"[onlaw-books] shards={burn.BURN_CELL_ROOT}\n"
        f"[onlaw-books] manifest={manifest_path}",
        flush=True,
    )

    model_to_shard, model_to_ckpt = burn._build_weight_map(str(burn.SOURCE))
    scale_map = burn._build_fp8_scale_inv_map(str(burn.SOURCE))
    profile = None
    if keying == ROUTED_BOOK_KEYING_STACK:
        from prismaquant.model_profiles import detect_profile
        profile = detect_profile(str(burn.SOURCE))
        for key in keys:
            print(
                f"[onlaw-books] stack {key} pools "
                f"{list(population_members(key, keying, profile))}",
                flush=True,
            )

    started = time.time()
    done = 0
    with manifest_path.open("a") as manifest:
        for layer, projection, missing in worklist:
            data = load_population(
                layer, projection, keying=keying, profile=profile,
                device=torch.device("cuda:0"),
                identity=identities[layer], all_col_weights=all_col_weights,
                model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt,
                scale_map=scale_map,
            )
            try:
                for rung in missing:
                    cell_started = time.time()
                    cell = _burn_cell(
                        layer=layer, projection=projection,
                        data=data, rung=rung,
                    )
                    done += 1
                    semantics = (cell.get("timing") or {}).get(
                        "measurement_semantics"
                    ) or {}
                    row = {
                        "keying": keying,
                        "layer": layer, "projection": projection, "rung": rung,
                        "shard": str(burn._burn_cell_path(
                            layer, projection,
                            burn.BURN_PASS_TAGS["full_layer"], rung,
                        )),
                        "book_sha256": semantics.get("book_sha256"),
                        "book_path": semantics.get("book_path"),
                        "encoder": semantics.get("encoder"),
                        "ldlq": semantics.get("ldlq"),
                        "scale_policy": semantics.get("scale_policy"),
                        "warm_state_outcome": (
                            cell.get("timing") or {}
                        ).get("warm_state_outcome"),
                        "seconds": round(time.time() - cell_started, 2),
                    }
                    manifest.write(json.dumps(row, sort_keys=True) + "\n")
                    manifest.flush()
                    rate = (time.time() - started) / done
                    print(
                        f"[onlaw-books] {done}/{total_cells} "
                        f"L{layer:02d} {projection} K{rung} "
                        f"{row['seconds']}s enc={row['encoder']} "
                        f"eta={(total_cells - done) * rate / 3600:.2f}h",
                        flush=True,
                    )
            finally:
                del data
                torch.cuda.empty_cache()

    print(
        f"[onlaw-books] DONE {done} cells in "
        f"{(time.time() - started) / 3600:.2f}h -> {manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
