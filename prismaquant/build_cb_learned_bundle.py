"""Build the immutable learned-codebook bundle before any CB render stage.

This is orchestration, not a second model cache: source values are decoded by
the streaming export reader, one dense Linear at a time, and the certified
trainer retains only the tiny canonical-FP16 books.  Cost/cache/KL/export then
open this exact ``.pqcb`` through ``CB_CODEBOOK_BUNDLE``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import pickle
import re
from pathlib import Path
from typing import Mapping, Sequence

import torch

from . import format_registry as fr
from .cb_learned_bundle import (
    PretrainedCodebookCell,
    train_and_save_bundle_streaming,
)
from .cb_banked_books import (
    BankedCBLBookRequest,
    RoutedMoECBLSelection,
    RoutedMoECBLSelectionCell,
    banked_cbl_origin,
    load_banked_cbl_book,
    load_routed_moe_cbl_selection,
)
from .cb_layout import parse_format_name
from .cb_warm_state import tensor_value_identity
from .export_nvfp4_cb import _try_resolve_skeleton
from .export_nvfp4_cb_streaming import (
    _LazySkeleton,
    _plan_expert_stacks,
)
from .model_profiles import detect_profile
from .nvfp4_cb_footprint import is_cb_format
from .routed_moe_codebooks import (
    logical_role_qname,
    stacked_role_col_weights,
)


_ROUTED_MOE_QNAME = re.compile(r"(?:^|[.])experts(?:[.]|$)")
_LAYER_QNAME = re.compile(r"(?:^|[.])layers[.]([0-9]+)(?:[.]|$)")


@dataclass(frozen=True)
class _RoutedRolePlan:
    layer: int
    projection: str
    qname: str
    source_weight_keys: tuple[str, ...]
    member_qnames: tuple[str, ...]
    col_weights: torch.Tensor
    selected_cells: Mapping[str, RoutedMoECBLSelectionCell]


def _layer_from_qname(qname: str) -> int | None:
    match = _LAYER_QNAME.search(str(qname))
    return None if match is None else int(match.group(1))


def _discover_routed_role_plans(
    *,
    selection: RoutedMoECBLSelection,
    source: _LazySkeleton,
    profile: object,
    col_weights: Mapping[str, torch.Tensor],
) -> dict[str, _RoutedRolePlan]:
    """Bind each selected burn cell to one profile-declared expert role."""

    groups = _plan_expert_stacks(source, profile)
    candidates: dict[tuple[int, str], list[tuple[str, Mapping[int, str]]]] = {}
    for prefix, projections in groups.items():
        layer = _layer_from_qname(prefix)
        if layer is None:
            continue
        for projection in ("gate_proj", "up_proj", "down_proj"):
            members = projections.get(projection)
            if members:
                candidates.setdefault((layer, projection), []).append(
                    (str(prefix), members)
                )

    selected_by_role: dict[
        tuple[int, str], dict[str, RoutedMoECBLSelectionCell]
    ] = {}
    for cell in selection.cells:
        selected_by_role.setdefault(
            (cell.layer, cell.projection), {}
        )[cell.format_name] = cell

    plans: dict[str, _RoutedRolePlan] = {}
    for (layer, projection), selected_cells in sorted(selected_by_role.items()):
        matched = candidates.get((layer, projection), [])
        if len(matched) != 1:
            raise ValueError(
                f"selection L{layer} {projection}: expected exactly one "
                "profile-declared routed expert stack, found "
                f"{[prefix for prefix, _members in matched]}"
            )
        prefix, source_members = matched[0]
        expert_ids = sorted(int(expert) for expert in source_members)
        if expert_ids != list(range(len(expert_ids))):
            raise ValueError(
                f"selection L{layer} {projection}: source expert ids are not "
                f"contiguous: {expert_ids[:8]}"
            )
        try:
            packed_parent = profile.packed_expert_parent_for_projection(
                projection
            )
        except Exception as exc:
            raise ValueError(
                f"selection L{layer} {projection}: profile cannot resolve its "
                "packed expert parent"
            ) from exc
        if not packed_parent:
            raise ValueError(
                f"selection L{layer} {projection}: profile declares no packed "
                "expert parent"
            )
        packed_qname = f"{prefix}.{packed_parent}"
        qname = logical_role_qname(packed_qname, projection)
        member_map = {
            (projection, expert): f"{prefix}.{expert}.{projection}"
            for expert in expert_ids
        }
        role_col, member_qnames = stacked_role_col_weights(
            packed_qname=packed_qname,
            projection=projection,
            member_qnames=member_map,
            col_weights=col_weights,
        )
        source_keys = tuple(
            str(source_members[expert]) + ".weight"
            for expert in expert_ids
        )
        if qname in plans:
            raise ValueError(f"duplicate routed learned role qname {qname}")
        plans[qname] = _RoutedRolePlan(
            layer=layer,
            projection=projection,
            qname=qname,
            source_weight_keys=source_keys,
            member_qnames=member_qnames,
            col_weights=role_col,
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
    routed_moe_book_selection: str | Path | None = None,
) -> object:
    """Stream source weights and publish one value-bearing bundle.

    Dense learned cells retain the certified trainer.  Routed rank-3 cells
    exist only when an explicit selection names an accepted burn shard, and
    their books are loaded after the current role weight/imatrix identities
    are available.  No directory search, retraining, or lattice fallback is
    reachable for those cells.
    """

    canonical_formats = _canonical_cb_formats(formats)
    learned_formats = tuple(
        name for name in canonical_formats
        if (
            (parsed := parse_format_name(name)) is not None
            and parsed[0].source == "learned"
        )
    )
    if not any(
        (parsed := parse_format_name(name)) is not None
        and parsed[0].grid == "fp8"
        for name in canonical_formats
    ):
        raise ValueError(
            "CB_CODEBOOK_SOURCE_SCOPE enables FP8 learned books, but the "
            "requested format menu contains no FP8 CB rung"
        )
    normalized_col = {
        str(name): torch.as_tensor(value)
        for name, value in col_weights.items()
    }
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
    if selection is not None:
        selected_formats = {cell.format_name for cell in selection.cells}
        absent = sorted(selected_formats - set(learned_formats))
        if absent:
            raise ValueError(
                "routed-MoE book selection names format(s) outside the "
                f"requested learned FP8-CB menu: {absent}"
            )
    routed_plans = (
        {}
        if selection is None
        else _discover_routed_role_plans(
            selection=selection,
            source=source,
            profile=profile,
            col_weights=normalized_col,
        )
    )
    for qname, plan in routed_plans.items():
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
        shapes = {
            tuple(int(dim) for dim in source.logical_shape(key))
            for key in plan.source_weight_keys
        }
        if len(shapes) != 1:
            raise ValueError(
                f"{qname}: routed expert source member shapes disagree: "
                f"{sorted(shapes)}"
            )
        member_shape = next(iter(shapes))
        first = source.dequant_weight(plan.source_weight_keys[0])
        weight = torch.empty(
            (len(plan.source_weight_keys), *member_shape),
            dtype=first.dtype,
            device=target_device,
        )
        weight[0].copy_(first.to(target_device))
        del first
        for expert_id, key in enumerate(
            plan.source_weight_keys[1:], start=1
        ):
            member = source.dequant_weight(key)
            if tuple(int(dim) for dim in member.shape) != member_shape:
                raise ValueError(
                    f"{qname}: expert {expert_id} source shape changed while "
                    "streaming the role stack"
                )
            weight[expert_id].copy_(member.to(target_device))
            del member
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
                projection=plan.projection,
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
        return {
            member_qname: (
                weight[expert_id],
                normalized_col[member_qname],
            )
            for expert_id, member_qname in enumerate(plan.member_qnames)
        }

    formats_by_qname: dict[str, tuple[str, ...]] = {
        qname: canonical_formats for qname in dense_qnames
    }
    formats_by_qname.update({
        qname: tuple(sorted(plan.selected_cells))
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
        routed_moe_qnames=routed_plans,
        pretrained_codebook_provider=provide_banked_book,
        input_alias_provider=provide_aliases,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--col-weights", required=True)
    parser.add_argument("--formats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--routed-moe-book-selection",
        default=None,
        help=(
            "strict JSON naming the bank root and one absolute accepted burn "
            "shard per routed (layer, projection, K28-K33) cell"
        ),
    )
    args = parser.parse_args(argv)

    from .gpu_guard import require_cuda_hot_path

    require_cuda_hot_path("build_cb_learned_bundle")
    with open(args.col_weights, "rb") as handle:
        raw_col_weights = pickle.load(handle)
    if not isinstance(raw_col_weights, Mapping):
        raise ValueError("--col-weights must contain a qname -> tensor mapping")
    bundle = build_bundle_from_model(
        model_dir=args.model_dir,
        col_weights=raw_col_weights,
        formats=[item for item in args.formats.split(",") if item.strip()],
        output=args.output,
        device=args.device,
        routed_moe_book_selection=args.routed_moe_book_selection,
    )
    print(
        f"[cbl-bundle] wrote {bundle.path}: "
        f"{len(bundle.sidecar_tensors)} canonical FP16 tables, "
        f"sha256={bundle.bundle_content_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
