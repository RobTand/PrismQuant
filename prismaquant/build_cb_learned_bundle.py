"""Build the immutable learned-codebook bundle before any CB render stage.

This is orchestration, not a second model cache: source values are decoded by
the streaming export reader, one dense Linear at a time, and the certified
trainer retains only the tiny canonical-FP16 books.  Cost/cache/KL/export then
open this exact ``.pqcb`` through ``CB_CODEBOOK_BUNDLE``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import pickle
import re
from pathlib import Path
from typing import Mapping, Sequence

import torch

from . import format_registry as fr
from .cb_learned_bundle import (
    CBL_RUNG_POLICY,
    PretrainedCodebookCell,
    train_and_save_bundle_streaming,
)
from .cb_imatrix import canonical_imatrix_sha256, imatrix_from_probe_file
from .cb_learned_promotion import (
    ValidatedCBLPromotionReceipt,
    read_promotion_receipt_payload,
    role_census_for_qnames,
    validate_promotion_receipt,
)
from .cost_streaming import (
    validate_cached_streamed_model_identity,
    validate_streamed_model_identity,
)
from .cb_banked_books import (
    BankedCBLBookRequest,
    RoutedMoECBLSelection,
    RoutedMoECBLSelectionCell,
    banked_cbl_origin,
    load_banked_cbl_book,
    load_routed_moe_cbl_selection,
)
from .cb_warm_state import tensor_value_identity
from .export_nvfp4_cb import _try_resolve_skeleton
from .export_nvfp4_cb_streaming import (
    _LazySkeleton,
    _packed_expert_projection_names,
    _plan_expert_stacks,
)
from .model_profiles import detect_profile
from .nvfp4_cb_footprint import _cb_scope_family, is_cb_format
from .routed_moe_codebooks import (
    DEFAULT_ROUTED_BOOK_KEYING,
    ROUTED_BOOK_KEYINGS,
    ROUTED_BOOK_KEYING_STACK,
    bundle_book_qname,
    normalize_routed_book_keying,
    routed_book_key,
    stacked_role_col_weights,
)


_ROUTED_MOE_QNAME = re.compile(r"(?:^|[.])experts(?:[.]|$)")
_LAYER_QNAME = re.compile(r"(?:^|[.])layers[.]([0-9]+)(?:[.]|$)")


@dataclass(frozen=True)
class _RoutedBookPlan:
    """One burned book and the exact population it covers.

    Under stack keying ``projections`` holds both halves of a fused ``w13``
    stack and the plan's weight is the fused rank-3 tensor; under role keying
    it holds one projection and the plan is the pre-R1 per-role form.
    """

    layer: int
    keying: str
    key: str
    projections: tuple[str, ...]
    qname: str
    packed_qname: str
    expert_ids: tuple[int, ...]
    # [expert][projection] source ``.weight`` keys, in fused row order.
    source_weight_keys: tuple[tuple[str, ...], ...]
    rows_by_projection: tuple[int, ...]
    member_qnames: tuple[str, ...]
    col_weights: torch.Tensor
    col_weights_from_packed_entry: bool
    selected_cells: Mapping[str, RoutedMoECBLSelectionCell]

    @property
    def projection(self) -> str:
        """The single projection of a role-keyed plan."""

        if len(self.projections) != 1:
            raise ValueError(
                f"{self.qname}: pooled stack plan has no single projection"
            )
        return self.projections[0]


def _layer_from_qname(qname: str) -> int | None:
    match = _LAYER_QNAME.search(str(qname))
    return None if match is None else int(match.group(1))


def _packed_parent(profile: object, projection: str, *, where: str) -> str:
    try:
        packed_parent = profile.packed_expert_parent_for_projection(projection)
    except Exception as exc:
        raise ValueError(
            f"{where}: profile cannot resolve its packed expert parent"
        ) from exc
    if not packed_parent:
        raise ValueError(
            f"{where}: profile declares no packed expert parent"
        )
    return str(packed_parent)


def _stack_col_weights(
    cell_qname: str,
    *,
    experts: int,
    in_features: int,
    col_weights: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """The packed target's own imatrix, shaped for the pooled population.

    A pooled book is trained against the exact tensor the export renders the
    fused stack with, so this reads the packed entry rather than re-pooling the
    per-member vectors: two spellings of one imatrix would be a rendering
    confound.  A single broadcast row is materialized per expert, which is what
    the encoder does with it.
    """

    entry = col_weights.get(cell_qname)
    if entry is None:
        raise ValueError(
            f"{cell_qname}: pooled stack book needs the packed target's "
            "col_weights entry (moe_imatrix.synthesize_packed_expert_"
            "col_weights writes it); refusing to re-pool the per-member "
            "vectors into a second imatrix spelling"
        )
    value = torch.as_tensor(entry)
    # The same two shapes the export coverage gate accepts for a rank-3 CB
    # target: one vector per expert, or one broadcast vector for the stack.
    if value.numel() == experts * in_features:
        return value.reshape(experts, 1, in_features).contiguous()
    if value.numel() == in_features:
        return (
            value.reshape(1, 1, in_features)
            .expand(experts, 1, in_features)
            .contiguous()
        )
    raise ValueError(
        f"{cell_qname}: packed col_weights has {value.numel()} elements but "
        f"the pooled stack wants {in_features} or {experts}x{in_features}"
    )


def _discover_routed_book_plans(
    *,
    selection: RoutedMoECBLSelection,
    source: _LazySkeleton,
    profile: object,
    col_weights: Mapping[str, torch.Tensor],
    keying: str = DEFAULT_ROUTED_BOOK_KEYING,
) -> dict[str, _RoutedBookPlan]:
    """Bind each selected burn cell to one profile-declared expert population.

    Campaign rule R1: with ``keying="stack"`` (the default) a fused ``w13``
    stack yields ONE plan covering gate and up, so the burn and the export name
    one book per ``(layer, stack, rung)``.  ``keying="role"`` reproduces the
    pre-R1 per-``(layer, projection, rung)`` form for the A/B arm.
    """

    keying = normalize_routed_book_keying(keying)
    groups = _plan_expert_stacks(source, profile)
    candidates: dict[
        tuple[int, str], list[tuple[str, str, dict[str, Mapping[int, str]]]]
    ] = {}
    for prefix, projections in groups.items():
        layer = _layer_from_qname(prefix)
        if layer is None:
            continue
        for projection in ("gate_proj", "up_proj", "down_proj"):
            members = projections.get(projection)
            if not members:
                continue
            packed_parent = _packed_parent(
                profile, projection, where=f"L{layer} {projection}"
            )
            key = routed_book_key(packed_parent, projection, keying=keying)
            bucket = candidates.setdefault((layer, key), [])
            for entry in bucket:
                if entry[0] == str(prefix):
                    entry[2][projection] = members
                    break
            else:
                bucket.append(
                    (str(prefix), packed_parent, {projection: members})
                )

    selected_by_key: dict[
        tuple[int, str], dict[str, RoutedMoECBLSelectionCell]
    ] = {}
    for cell in selection.cells:
        selected_by_key.setdefault(
            (cell.layer, cell.projection), {}
        )[cell.format_name] = cell

    plans: dict[str, _RoutedBookPlan] = {}
    for (layer, key), selected_cells in sorted(selected_by_key.items()):
        matched = candidates.get((layer, key), [])
        if len(matched) != 1:
            raise ValueError(
                f"selection L{layer} {key}: expected exactly one "
                f"profile-declared routed expert population under {keying} "
                "keying, found "
                f"{[prefix for prefix, _parent, _members in matched]}. A "
                "selection burned under the other keying names cells this "
                "build cannot bind; pass --routed-book-keying to match the "
                "burn."
            )
        prefix, packed_parent, members_by_projection = matched[0]
        packed_qname = f"{prefix}.{packed_parent}"
        if keying == ROUTED_BOOK_KEYING_STACK:
            declared = _packed_expert_projection_names(profile, packed_parent)
            projections = tuple(
                projection for projection in declared
                if projection in members_by_projection
            )
            if len(projections) != len(members_by_projection):
                raise ValueError(
                    f"selection L{layer} {key}: profile declares projections "
                    f"{declared} but the source carries "
                    f"{sorted(members_by_projection)}"
                )
        else:
            projections = tuple(sorted(members_by_projection))
        qname = bundle_book_qname(
            packed_qname, projections[0], keying=keying
        )
        expert_id_sets = {
            projection: sorted(int(expert) for expert in members)
            for projection, members in members_by_projection.items()
        }
        expert_ids = expert_id_sets[projections[0]]
        if any(ids != expert_ids for ids in expert_id_sets.values()):
            raise ValueError(
                f"selection L{layer} {key}: fused projections cover different "
                "expert populations"
            )
        if expert_ids != list(range(len(expert_ids))):
            raise ValueError(
                f"selection L{layer} {key}: source expert ids are not "
                f"contiguous: {expert_ids[:8]}"
            )
        source_keys = tuple(
            tuple(
                str(members_by_projection[projection][expert]) + ".weight"
                for projection in projections
            )
            for expert in expert_ids
        )
        member_shapes = tuple(
            tuple(int(dim) for dim in source.logical_shape(
                str(members_by_projection[projection][expert_ids[0]])
                + ".weight"
            ))
            for projection in projections
        )
        if any(len(shape) != 2 for shape in member_shapes):
            raise ValueError(
                f"selection L{layer} {key}: routed expert members must be "
                f"rank 2, got {list(member_shapes)}"
            )
        rows_by_projection = tuple(shape[0] for shape in member_shapes)
        in_features = {shape[1] for shape in member_shapes}
        if len(in_features) != 1:
            raise ValueError(
                f"selection L{layer} {key}: fused projections disagree on "
                f"in_features: {sorted(in_features)}"
            )
        stack_in_features = next(iter(in_features))
        member_map = {
            (projection, expert): f"{prefix}.{expert}.{projection}"
            for projection in projections
            for expert in expert_ids
        }
        if keying == ROUTED_BOOK_KEYING_STACK:
            plan_col = _stack_col_weights(
                qname,
                experts=len(expert_ids),
                in_features=stack_in_features,
                col_weights=col_weights,
            )
            member_qnames = tuple(
                member_map[(projection, expert)]
                for projection in projections
                for expert in expert_ids
            )
            missing_members = [
                member for member in member_qnames if member not in col_weights
            ]
            if missing_members:
                raise ValueError(
                    f"{qname}: stack book member {missing_members[0]!r} has no "
                    "col_weights entry"
                )
            from_packed_entry = True
        else:
            plan_col, member_qnames = stacked_role_col_weights(
                packed_qname=packed_qname,
                projection=projections[0],
                member_qnames=member_map,
                col_weights=col_weights,
            )
            from_packed_entry = False
        if qname in plans:
            raise ValueError(f"duplicate routed learned book qname {qname}")
        plans[qname] = _RoutedBookPlan(
            layer=layer,
            keying=keying,
            key=key,
            projections=projections,
            qname=qname,
            packed_qname=packed_qname,
            expert_ids=tuple(expert_ids),
            source_weight_keys=source_keys,
            rows_by_projection=rows_by_projection,
            member_qnames=member_qnames,
            col_weights=plan_col,
            col_weights_from_packed_entry=from_packed_entry,
            selected_cells=dict(selected_cells),
        )
    return plans


def _canonical_cb_formats(formats: Sequence[str]) -> tuple[str, ...]:
    result = tuple(sorted({
        fr.get_format(str(name).strip()).name
        for name in formats
        if str(name).strip() and is_cb_format(str(name).strip())
    }))
    if not result:
        raise ValueError("learned bundle build was given no CB formats")
    return result


def build_bundle_from_model(
    *,
    model_dir: str | Path,
    col_weights: Mapping[str, torch.Tensor],
    formats: Sequence[str],
    output: str | Path,
    device: str | torch.device,
    trainer_version: str = "v1",
    promotion_receipt: (
        Mapping[str, object] | ValidatedCBLPromotionReceipt | None
    ) = None,
    source_model_identity: Mapping[str, object] | None = None,
    probe_calibration_hash: str | None = None,
    imatrix_value_sha256: str | None = None,
    routed_moe_book_selection: str | Path | None = None,
    routed_book_keying: str = DEFAULT_ROUTED_BOOK_KEYING,
) -> object:
    """Stream source weights and publish one value-bearing bundle.

    ``routed_book_keying`` selects the burn rule for routed cells.  The default
    ``"stack"`` is campaign rule R1: gate and up are pooled into one book per
    ``(layer, stack, rung)``, so the exported fused weight names a single
    codebook.  ``"role"`` reproduces the pre-R1 per-``(layer, projection,
    rung)`` form; that arm exists for the A/B and its artifact needs the
    exporter's explicit per-role override.  ``down_proj`` is a one-projection
    stack and is unaffected by the choice.

    Dense learned cells retain the certified trainer.  Routed rank-3 *learned*
    cells exist only when an explicit selection names an accepted burn shard,
    and their books are loaded after the current weight/imatrix identities
    are available.  No directory search, retraining, or lattice fallback is
    reachable for those cells.  Routed populations additionally carry a declaration
    for the supplied *lattice* formats of any family they hold no learned
    selection in, which needs no book and no training: the bundle is the
    authoritative per-(qname, format) cell map, so a legal lattice rung with no
    cell is simply unrenderable.  The family scoping matters -- a routed role's
    FP8 menu *is* its burn selection, so it must not pick up FP8 lattice rungs
    no allocation can legally reach.
    """

    routed_book_keying = normalize_routed_book_keying(routed_book_keying)
    canonical_formats = _canonical_cb_formats(formats)
    normalized_col = {
        str(name): torch.as_tensor(value)
        for name, value in col_weights.items()
    }
    raw_imatrix_sha256 = canonical_imatrix_sha256(normalized_col)
    if imatrix_value_sha256 is not None:
        declared_imatrix_sha256 = str(imatrix_value_sha256).strip().lower()
        if declared_imatrix_sha256 != raw_imatrix_sha256:
            raise ValueError(
                "declared probe imatrix value_sha256 differs from the exact "
                "input col_weights"
            )
    trainer_version = str(trainer_version).strip().lower().replace(
        "learned-", ""
    )
    raw_receipt: Mapping[str, object] | None = None
    if trainer_version == "v1":
        if promotion_receipt is not None:
            raise ValueError(
                "promotion_receipt is valid only with trainer_version='v2'"
            )
        learned_formats = tuple(
            name for name in canonical_formats
            if name.startswith("FP8_CB_")
            and CBL_RUNG_POLICY.get(
                int(name.rsplit("K", 1)[1]), {}
            ).get("enabled") is True
        )
    elif trainer_version == "v2":
        if promotion_receipt is not None:
            raw_receipt = (
                promotion_receipt.payload
                if isinstance(
                    promotion_receipt, ValidatedCBLPromotionReceipt
                )
                else promotion_receipt
            )
        learned_formats = ()
    else:
        raise ValueError(
            "trainer_version must be v1 or v2, got "
            f"{trainer_version!r}"
        )
    if not any(name.startswith("FP8_CB_") for name in canonical_formats):
        raise ValueError(
            "CB_CODEBOOK_SOURCE_SCOPE enables FP8 learned books, but the "
            "requested format menu contains no FP8_CB rung"
        )
    routed_col_qnames = {
        name for name in normalized_col if _ROUTED_MOE_QNAME.search(name)
    }
    dense_qnames = tuple(sorted(set(normalized_col) - routed_col_qnames))

    source = _LazySkeleton(model_dir)
    try:
        profile = detect_profile(str(model_dir))
    except Exception:
        profile = None
    selection = (
        None
        if routed_moe_book_selection is None
        or not str(routed_moe_book_selection).strip()
        else load_routed_moe_cbl_selection(routed_moe_book_selection)
    )
    routed_plans = (
        {}
        if selection is None
        else _discover_routed_book_plans(
            selection=selection,
            source=source,
            profile=profile,
            col_weights=normalized_col,
            keying=routed_book_keying,
        )
    )
    for qname, plan in routed_plans.items():
        if plan.col_weights_from_packed_entry:
            # A pooled stack book IS trained against the packed target's own
            # entry, reshaped to the population; there is no second spelling to
            # reconcile.
            normalized_col[qname] = plan.col_weights
            continue
        if qname in normalized_col:
            # Packed-expert imatrix augmentation legitimately synthesizes a
            # ``...experts.down_proj`` entry, which is also the logical CBL
            # role name.  The burn identity is defined by stacking the exact
            # per-expert rows, so accept the redundant spelling only when its
            # complete tensor identity agrees; never let it replace or mask
            # the bank-matched role rows.
            existing_identity = tensor_value_identity(normalized_col[qname])
            role_identity = tensor_value_identity(plan.col_weights)
            if existing_identity != role_identity:
                raise ValueError(
                    f"{qname}: redundant packed role col_weights differ from "
                    "the exact per-expert role stack"
                )
        normalized_col[qname] = plan.col_weights
    if not dense_qnames and not routed_plans:
        raise ValueError("learned bundle build found no target Linear qnames")

    receipt_target_qnames = tuple(sorted((*dense_qnames, *routed_plans)))
    actual_imatrix_sha256 = canonical_imatrix_sha256(normalized_col)
    validated_receipt: ValidatedCBLPromotionReceipt | None = None
    validated_source_identity: Mapping[str, object] | None = None
    bound_calibration_hash: str | None = None
    if raw_receipt is not None:
        validated_source_identity = validate_streamed_model_identity(
            source_model_identity,
            where="learned-v2 bundle source identity",
        )
        checkpoint_weight_map = validated_source_identity.get(
            "checkpoint_weight_map"
        )
        if (
            not isinstance(checkpoint_weight_map, dict)
            or not checkpoint_weight_map
        ):
            raise ValueError(
                "learned-v2 bundle source identity must cover the complete "
                "checkpoint tensor-to-shard map"
            )
        if (
            not isinstance(probe_calibration_hash, str)
            or not probe_calibration_hash.strip()
        ):
            raise ValueError(
                "learned-v2 promotion requires the actual probe calibration "
                "hash"
            )
        bound_calibration_hash = probe_calibration_hash.strip()
        role_census = role_census_for_qnames(receipt_target_qnames)
        validated_receipt = validate_promotion_receipt(
            raw_receipt,
            expected_model_id=str(validated_source_identity["source"]),
            expected_model_content_sha256=str(
                validated_source_identity["content_sha256"]
            ),
            expected_calibration_hash=bound_calibration_hash,
            expected_imatrix_sha256=actual_imatrix_sha256,
            expected_role_census=role_census,
            expected_qnames=receipt_target_qnames,
        )
        learned_formats = tuple(
            name for name in canonical_formats
            if name.startswith("FP8_CB_")
            and validated_receipt.source_for_rung(
                int(name.rsplit("K", 1)[1])
            ) == "learned"
        )
    elif any(
        value is not None
        for value in (
            source_model_identity,
            probe_calibration_hash,
            imatrix_value_sha256,
        )
    ):
        raise ValueError(
            "source/probe/imatrix promotion bindings require a learned-v2 "
            "promotion receipt"
        )

    if selection is not None:
        selected_formats = {cell.format_name for cell in selection.cells}
        absent = sorted(selected_formats - set(learned_formats))
        if absent:
            raise ValueError(
                "routed-MoE book selection names format(s) outside the "
                f"requested learned FP8-CB menu: {absent}"
            )

    resolved: dict[str, str] = {}
    for qname in dense_qnames:
        key = _try_resolve_skeleton(qname, source, profile)
        if key is None:
            raise KeyError(
                f"{qname}: no source weight for learned bundle training"
            )
        shape = source.logical_shape(key)
        if len(shape) != 2:
            raise ValueError(
                f"{qname}: dense learned bundle source must be rank 2, got "
                f"{shape}"
            )
        resolved[qname] = key

    target_device = torch.device(device)

    def provide_weight(qname: str) -> torch.Tensor:
        plan = routed_plans.get(qname)
        if plan is None:
            return source.dequant_weight(resolved[qname]).to(target_device)
        # One population entry per expert; its rows are the plan's projections
        # concatenated in the profile's declared fused order.  A role-keyed
        # plan has one projection, so this is the pre-R1 stack unchanged; a
        # stack-keyed plan is the fused w13 tensor the export renders, which is
        # exactly the union of both projections' data.
        rows = sum(plan.rows_by_projection)
        in_features = int(source.logical_shape(
            plan.source_weight_keys[0][0]
        )[1])
        first = source.dequant_weight(plan.source_weight_keys[0][0])
        weight = torch.empty(
            (len(plan.source_weight_keys), rows, in_features),
            dtype=first.dtype,
            device=target_device,
        )
        for expert_id, keys in enumerate(plan.source_weight_keys):
            offset = 0
            for index, key in enumerate(keys):
                member = (
                    first if (expert_id == 0 and index == 0)
                    else source.dequant_weight(key)
                )
                expected = (plan.rows_by_projection[index], in_features)
                if tuple(int(dim) for dim in member.shape) != expected:
                    raise ValueError(
                        f"{qname}: expert {expert_id} "
                        f"{plan.projections[index]} source shape "
                        f"{tuple(member.shape)} != {expected}"
                    )
                weight[expert_id, offset:offset + expected[0]].copy_(
                    member.to(target_device)
                )
                offset += expected[0]
                del member
        del first
        return weight

    def provide_banked_book(
        qname: str,
        format_name: str,
        weight: torch.Tensor,
        role_col_weights: torch.Tensor,
    ) -> object | None:
        plan = routed_plans.get(qname)
        if plan is None:
            return None
        selected_cell = plan.selected_cells.get(format_name)
        if selected_cell is None:
            raise ValueError(
                f"{qname}/{format_name}: no accepted routed burn shard; "
                "refusing retraining or lattice fallback"
            )
        source_shape, source_digest = tensor_value_identity(weight)
        col_shape, col_digest = tensor_value_identity(role_col_weights)
        assert selection is not None
        book = load_banked_cbl_book(
            BankedCBLBookRequest(
                burn_shard_path=selected_cell.burn_shard_path,
                layer=plan.layer,
                # Under stack keying the burn cell is named by the packed
                # parent, so a per-role shard can never satisfy a pooled
                # request and vice versa.
                projection=plan.key,
                rung=selected_cell.rung,
                source_digest=source_digest,
                col_weights_digest=col_digest,
                source_shape=tuple(source_shape),
                col_weights_shape=tuple(col_shape),
            ),
            book_root=selection.book_root,
        )
        expected_experts = tuple(range(int(weight.shape[0])))
        if book.encoded_expert_ids != expected_experts:
            raise ValueError(
                f"{qname}/{format_name}: banked encoded expert ids "
                f"{book.encoded_expert_ids[:8]} do not cover the current "
                "stack contiguously"
            )
        return PretrainedCodebookCell(
            codebook=book.subtables,
            origin=banked_cbl_origin(selection, book),
        )

    def provide_aliases(
        qname: str,
        weight: torch.Tensor,
        role_col_weights: torch.Tensor,
    ) -> Mapping[str, tuple[torch.Tensor, torch.Tensor]]:
        plan = routed_plans.get(qname)
        if plan is None:
            return {}
        # Each per-expert member keeps its own input identity: its rows of the
        # population entry, and its own imatrix vector.  A pooled book covers
        # every member of both projections, so every member aliases the one
        # cell.
        aliases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        offset = 0
        for index, projection in enumerate(plan.projections):
            rows = plan.rows_by_projection[index]
            for local_id in range(len(plan.expert_ids)):
                member_qname = plan.member_qnames[
                    index * len(plan.expert_ids) + local_id
                ]
                aliases[member_qname] = (
                    weight[local_id, offset:offset + rows],
                    normalized_col[member_qname],
                )
            offset += rows
        return aliases

    formats_by_qname: dict[str, tuple[str, ...]] = {
        qname: canonical_formats for qname in dense_qnames
    }
    # A routed role's LEARNED cells are exactly the accepted burn shards --
    # retraining a rank-3 population is forbidden, so a rung with no shard can
    # never be learned here.  Its LATTICE cells are a different question: a
    # lattice table is weight-independent and shared, so it costs no book and
    # no training, and the bundle is the authoritative per-(qname, format) cell
    # map that codebook_source_for_cell resolves every render through.  Giving
    # routed roles only the selection's formats therefore made every legal
    # lattice cell on a routed expert unrenderable -- the campaign prices the
    # NVFP4 lattice for all 33,325 units, so its first routed anchor died on
    # "immutable learned bundle has no NVFP4_CB_K15 cell; refusing lattice
    # fallback".  The per-rung basis map is unchanged either way; this makes
    # cell-level resolution agree with it.
    # Scope the addition to what a routed role can LEGALLY be rendered at, not
    # to everything the build was handed.  A routed role's FP8 menu *is* its
    # burn selection -- the on-law contract gives routed experts K28/K32 only,
    # and an FP8 rung outside the selection is either unlearnable here (no
    # accepted shard) or off-menu entirely.  So exclude any family the role
    # already has a learned selection in, and declare the lattice rungs only
    # for families where it has none.  Declaring an unreachable cell would be
    # inert today but would soften a fail-closed edge: a later path asking a
    # routed expert for, say, FP8_CB_K47 would get a lattice render instead of
    # a refusal.  ``--formats`` still carries K47/K48 for the DENSE qnames,
    # which is what witnesses the learned<=K46 / lattice>K46 rung-policy
    # boundary in the bundle-authoritative per-rung map.
    learned_families = {
        _cb_scope_family(name) for name in learned_formats
    }
    lattice_formats = tuple(
        name for name in canonical_formats
        if name not in set(learned_formats)
        and _cb_scope_family(name) not in learned_families
    )
    formats_by_qname.update({
        qname: tuple(sorted(set(plan.selected_cells) | set(lattice_formats)))
        for qname, plan in routed_plans.items()
    })
    target_qnames = tuple(sorted(formats_by_qname))

    return train_and_save_bundle_streaming(
        output,
        qnames=target_qnames,
        weight_provider=provide_weight,
        col_weights=normalized_col,
        formats=formats_by_qname,
        learned_formats=learned_formats,
        trainer_version=trainer_version,
        promotion_receipt=validated_receipt,
        source_model_identity=validated_source_identity,
        probe_calibration_hash=bound_calibration_hash,
        imatrix_value_sha256=(
            actual_imatrix_sha256
            if validated_receipt is not None
            else None
        ),
        routed_moe_qnames=routed_plans,
        routed_book_keying={
            qname: plan.keying for qname, plan in routed_plans.items()
        },
        pretrained_codebook_provider=provide_banked_book,
        input_alias_provider=provide_aliases,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    imatrix_group = parser.add_mutually_exclusive_group(required=True)
    imatrix_group.add_argument(
        "--col-weights",
        help="existing trusted qname -> imatrix tensor pickle",
    )
    imatrix_group.add_argument(
        "--imatrix-probe",
        help=(
            "existing trusted sensitivity probe.pkl; derive imatrix values "
            "from full-corpus act_sq_sum/n_tokens_seen"
        ),
    )
    parser.add_argument("--formats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--trainer-version",
        choices=("v1", "v2"),
        default="v1",
        help=(
            "v1 preserves legacy behavior; v2 uses density-aware deterministic "
            "sampling and defaults every rung to lattice"
        ),
    )
    parser.add_argument(
        "--promotion-receipt",
        default=None,
        help=(
            "strict learned-v2 two-holdout receipt; without it v2 emits only "
            "lattice cells"
        ),
    )
    parser.add_argument(
        "--source-model-identity-cache",
        default=os.environ.get(
            "PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE"
        ),
        help=(
            "existing complete streamed-model identity cache; required for "
            "learned-v2 promotion receipts"
        ),
    )
    parser.add_argument(
        "--routed-moe-book-selection",
        default=None,
        help=(
            "strict JSON naming the bank root and one absolute accepted burn "
            "shard per routed (layer, book key, K28-K33) cell; the book key is "
            "the packed parent under stack keying and one projection under "
            "role keying"
        ),
    )
    parser.add_argument(
        "--routed-book-keying",
        choices=list(ROUTED_BOOK_KEYINGS),
        default=DEFAULT_ROUTED_BOOK_KEYING,
        help=(
            "how routed learned books are keyed: 'stack' (default, campaign "
            "rule R1) pools gate and up into one book per (layer, stack, "
            "rung); 'role' reproduces the pre-R1 book per (layer, projection, "
            "rung), whose artifact needs the exporter's --allow-per-role-books"
        ),
    )
    args = parser.parse_args(argv)

    from .gpu_guard import require_cuda_hot_path

    require_cuda_hot_path("build_cb_learned_bundle", args.device)
    imatrix_provenance = None
    if args.imatrix_probe:
        raw_col_weights, imatrix_provenance = imatrix_from_probe_file(
            args.imatrix_probe
        )
    else:
        with open(args.col_weights, "rb") as handle:
            raw_col_weights = pickle.load(handle)
    if not isinstance(raw_col_weights, Mapping):
        raise ValueError("imatrix input must contain a qname -> tensor mapping")
    receipt = None
    source_model_identity = None
    probe_calibration_hash = None
    imatrix_value_sha256 = None
    if args.promotion_receipt is not None:
        if args.trainer_version != "v2":
            raise ValueError(
                "--promotion-receipt requires --trainer-version v2"
            )
        if not args.imatrix_probe or imatrix_provenance is None:
            raise ValueError(
                "learned-v2 promotion requires --imatrix-probe so the actual "
                "calibration and act_sq_sum provenance are bound"
            )
        probe_calibration_hash = imatrix_provenance.get(
            "calibration_hash"
        )
        if (
            not isinstance(probe_calibration_hash, str)
            or not probe_calibration_hash.strip()
        ):
            raise ValueError(
                "learned-v2 promotion probe has no calibration hash"
            )
        if not args.source_model_identity_cache:
            raise ValueError(
                "learned-v2 promotion requires "
                "--source-model-identity-cache"
            )
        source_model_identity = validate_cached_streamed_model_identity(
            args.model_dir,
            args.source_model_identity_cache,
            require_complete_checkpoint=True,
        )
        imatrix_value_sha256 = str(imatrix_provenance["value_sha256"])
        receipt = read_promotion_receipt_payload(
            args.promotion_receipt
        )
    bundle = build_bundle_from_model(
        model_dir=args.model_dir,
        col_weights=raw_col_weights,
        formats=[item for item in args.formats.split(",") if item.strip()],
        output=args.output,
        device=args.device,
        trainer_version=args.trainer_version,
        promotion_receipt=receipt,
        source_model_identity=source_model_identity,
        probe_calibration_hash=probe_calibration_hash,
        imatrix_value_sha256=imatrix_value_sha256,
        routed_moe_book_selection=args.routed_moe_book_selection,
        routed_book_keying=args.routed_book_keying,
    )
    print(
        f"[cbl-bundle] trainer: {args.trainer_version}",
        flush=True,
    )
    if imatrix_provenance is not None:
        print(
            "[cbl-bundle] imatrix: full-probe act_sq_sum/n_tokens_seen "
            f"sha256={imatrix_provenance['value_sha256']}",
            flush=True,
        )
    print(
        f"[cbl-bundle] routed book keying: {args.routed_book_keying}",
        flush=True,
    )
    print(
        f"[cbl-bundle] wrote {bundle.path}: "
        f"{len(bundle.sidecar_tensors)} canonical FP16 tables, "
        f"sha256={bundle.bundle_content_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
