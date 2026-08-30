"""Immutable, value-bearing learned-codebook bundles for the CB producer.

The cost, cache, KL, allocator, and export stages must render against the same
codebook *values*.  A digest-only manifest is deliberately insufficient: it can
identify values but cannot supply them to the encoder.  This module therefore
stores canonical FP16 tables and their strict manifest together in one
safetensors ``.pqcb`` file.

The training primitive is the FABLE-certified pooled weighted-Lloyd algorithm
formerly owned only by :mod:`tools.dsv4_cbl_kernels`.  It stays independent of
model loading and cache residency here: a production driver supplies already
decoded weights and the exact imatrix tensors, using the repository's existing
streaming/prefetch path.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_imatrix import canonical_imatrix_sha256
from prismaquant.cb_learned_promotion import (
    CBL_STEP4_RUNGS,
    CBL_V2_TRAINER_SCHEMA,
    ValidatedCBLPromotionReceipt,
    receipt_rung_policy,
    role_census_for_qnames,
    validate_promotion_receipt,
)
from prismaquant.cost_streaming import validate_streamed_model_identity
from prismaquant.cb_layout import (
    codebook_subtable_shapes,
    family_for,
    parse_format_name,
    subtable_bit_widths,
)
from prismaquant.gridbook_runtime_pin import (
    GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION,
    GridbookRuntimePinError,
    load_gridbook_runtime_pin,
    supports_routed_moe_per_role_codebook_lut,
)
from prismaquant.routed_moe_codebooks import (
    DEFAULT_ROUTED_BOOK_KEYING,
    ROUTED_BOOK_KEYINGS,
    ROUTED_BOOK_KEYING_ROLE,
    ROUTED_BOOK_KEYING_STACK,
    ROUTED_STACK_KEYS,
    normalize_routed_book_keying,
)


CB_LEARNED_BUNDLE_SCHEMA = "prismaquant.cb_learned_codebook_bundle.v1"
CB_LEARNED_BUNDLE_METADATA_KEY = "prismaquant_cb_learned_bundle"
CB_LEARNED_TRAINER_SCHEMA = "prismaquant.fp8_cbl_poolb.v1"
CB_LEARNED_TRAINER_V2_SCHEMA = CBL_V2_TRAINER_SCHEMA
CB_LEARNED_V2_SAMPLING_SCHEMA = "prismaquant.fp8_cbl_sampling.v2"

LLOYD_ROW_SAMPLE = 64
LLOYD_ROW_SEED = 4321
LLOYD_CAP = 2_000_000
LLOYD_ITERS = 4
LLOYD_SEED = 0


# This is a measurement policy table, not a structural bit-split rule.  K44,
# K45, and K46 are enabled by their own completed sweep-matched measurements;
# K47 is independently disabled by its completed NO-GO result. Rows can change
# one at a time without inventing a maximum-rung proxy.
CBL_RUNG_POLICY: dict[int, dict[str, object]] = {
    rung: {
        "enabled": True,
        "status": (
            "measured_go"
            if rung in {28, 33, 38, 43}
            else "enabled_inside_certified_k43_boundary"
        ),
        "provenance": (
            "transfer-study-fable-verify/F1_GENERALIZATION.md"
            if rung in {28, 33, 38, 43}
            else "F1 K43/K48 measured crossover boundary; this exact interior "
                 "rung has no separately cited quality cell"
        ),
        "note": (
            "FABLE-certified learned FP8 product-codebook arm"
            if rung in {28, 33, 38, 43}
            else "enabled by the measured <=K43 production boundary, not by "
                 "the 2048-entry structural rule"
        ),
    }
    for rung in range(28, 44)
}
CBL_RUNG_POLICY.update({
    44: {
        "enabled": True,
        "status": "measured_go_sweep_matched",
        "provenance": (
            "dq-runs/dsv4-quality-hybrid/sfd-analysis/"
            "cbl_k43_k47.log:31"
        ),
        "cbl_over_lattice_base_ratio": 0.6057,
        "note": "sweep-matched holdout activation-MSE: CBL is 39.43% better",
    },
    45: {
        "enabled": True,
        "status": "measured_go_sweep_matched",
        "provenance": (
            "dq-runs/dsv4-quality-hybrid/sfd-analysis/"
            "cbl_k43_k47.log:40"
        ),
        "cbl_over_lattice_base_ratio": 0.6929,
        "note": "sweep-matched holdout activation-MSE: CBL is 30.71% better",
    },
    46: {
        "enabled": True,
        "status": "measured_go_sweep_matched",
        "provenance": (
            "dq-runs/dsv4-quality-hybrid/sfd-analysis/"
            "cbl_k43_k47.log:51"
        ),
        "cbl_over_lattice_base_ratio": 0.8312,
        "note": "sweep-matched holdout activation-MSE: CBL is 16.88% better",
    },
    47: {
        "enabled": False,
        "status": "measured_no_go_sweep_matched",
        "provenance": (
            "dq-runs/dsv4-quality-hybrid/sfd-analysis/"
            "cbl_k43_k47.log:60"
        ),
        "cbl_over_lattice_base_ratio": 1.0689,
        "note": "sweep-matched holdout activation-MSE: CBL is 6.89% worse",
    },
})
CBL_RUNG_POLICY[43] = {
    "enabled": True,
    "status": "measured_strong_go_sweep_matched",
    "provenance": "2026-08-10 operator-reported sweep-matched holdout run",
    "note": "CBL/lattice holdout activation-MSE ratio 0.4897 (51.0% better)",
}
CBL_RUNG_POLICY[48] = {
    "enabled": False,
    "status": "measured_no_go",
    "provenance": "transfer-study-fable-verify/F1_GENERALIZATION.md",
    "note": "learned placement is measured 54--98% worse than lattice",
}

CB_LEARNED_TRAINER_STAMP: dict[str, object] = {
    "schema": CB_LEARNED_TRAINER_SCHEMA,
    "grid": "fp8",
    "mode": "product",
    "row_sample": LLOYD_ROW_SAMPLE,
    "row_seed": LLOYD_ROW_SEED,
    "vector_cap": LLOYD_CAP,
    "lloyd_iters": LLOYD_ITERS,
    "lloyd_seed": LLOYD_SEED,
    "initializer": "fixed_lattice",
    "normalization": "cand0_v1",
    "assignment": "imatrix_weighted",
    "materialization_dtype": "float16",
}

CB_LEARNED_TRAINER_V2_STAMP: dict[str, object] = {
    "schema": CB_LEARNED_TRAINER_V2_SCHEMA,
    "grid": "fp8",
    "mode": "product",
    "sampling_schema": CB_LEARNED_V2_SAMPLING_SCHEMA,
    "entries": "2**max(subtable_bit_widths)",
    "vectors_per_entry": 64,
    "target_rows": "ceil(64*entries/(in_features/8))",
    "sample_rows": "min(output_rows,max(64,target_rows))",
    "row_selection": "sha256(qname)-seeded_cpu_randperm_prefix",
    "vector_cap": LLOYD_CAP,
    "vector_cap_selection": "sha256(qname,rung)-seeded_cpu_randperm_prefix",
    "lloyd_iters": LLOYD_ITERS,
    "lloyd_seed": LLOYD_SEED,
    "initializer": "fixed_lattice",
    "normalization": "cand0_v1",
    "assignment": "imatrix_weighted",
    "centroid_accumulation": "stable_sort_segment_sum",
    "materialization_dtype": "float16",
}


def _default_v2_rung_policy() -> dict[int, dict[str, object]]:
    return {
        rung: {
            "enabled": False,
            "status": "default_lattice_unpromoted",
            "provenance": CB_LEARNED_TRAINER_V2_SCHEMA,
        }
        for rung in CBL_STEP4_RUNGS
    }


@dataclass(frozen=True)
class LearnedV2PoolResult:
    tables: tuple[torch.Tensor, ...]
    provenance: Mapping[str, object]


def _trainer_stamp(version: str) -> dict[str, object]:
    normalized = str(version).strip().lower().replace("learned-", "")
    if normalized == "v1":
        return dict(CB_LEARNED_TRAINER_STAMP)
    if normalized == "v2":
        return dict(CB_LEARNED_TRAINER_V2_STAMP)
    raise ValueError(f"learned bundle trainer_version must be v1 or v2, got {version!r}")


def _canonical_format(format_name: str) -> tuple[str, object, int]:
    parsed = parse_format_name(str(format_name).strip().upper())
    if parsed is None:
        raise ValueError(f"{format_name!r} is not a producer CB format")
    family, rung = parsed
    # Readers and the v1 compatibility trainer retain legacy off-law FP8
    # bundle compatibility.  Learned-v2 applies an explicit producer-rung
    # gate before selecting sources.
    return family.accepted_name(rung), family, int(rung)


def require_cbl_rung_enabled(rung_or_format: int | str) -> int:
    """Return the FP8-CB rung only when its measured policy enables CBL."""

    if isinstance(rung_or_format, str) and not str(rung_or_format).isdigit():
        canonical, family, rung = _canonical_format(rung_or_format)
        if family.grid != "fp8" or family.mode != "product":
            raise ValueError(
                f"{canonical}: learned production bundles support FP8 product "
                "codebooks only; NVFP4 CBL is measured NO-GO"
            )
    else:
        rung = int(rung_or_format)
    policy = CBL_RUNG_POLICY.get(rung)
    if policy is None:
        raise ValueError(
            f"FP8 CBL K{rung} has no measured production policy row"
        )
    if policy["enabled"] is not True:
        raise ValueError(
            f"FP8 CBL K{rung} is disabled by measured rung policy: "
            f"status={policy['status']}; provenance={policy['provenance']}; "
            f"{policy['note']}"
        )
    return rung


def refuse_routed_moe_learned(
    qname: str,
    *,
    routed_moe: bool = False,
    weight: torch.Tensor | None = None,
) -> None:
    """Fail closed unless the pinned runtime carries routed per-role LUTs.

    Rank-3 CB weights are routed stacks in the producer ABI and take this gate
    even if the caller omitted its explicit route flag.  Dense Linears never
    consult the gate because older Gridbook releases already support their
    per-row LUT offsets.
    """

    rank3 = weight is not None and torch.as_tensor(weight).ndim == 3
    routed_name = re.search(r"(?:^|[.])experts(?:[.]|$)", str(qname)) is not None
    if not (routed_moe or rank3 or routed_name):
        return
    try:
        pin = load_gridbook_runtime_pin()
    except GridbookRuntimePinError as exc:
        raise ValueError(
            f"{qname}: refusing learned codebook_ref on a routed-MoE stack: "
            "the immutable Gridbook runtime pin is invalid "
            f"({exc}); routed per-role LUTs require Gridbook "
            f">={GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION}."
        ) from exc
    if supports_routed_moe_per_role_codebook_lut(pin):
        return
    raise ValueError(
        f"{qname}: refusing learned codebook_ref on a routed-MoE stack: "
        f"pinned Gridbook {pin.version} ({pin.commit}) predates the "
        "per-row/per-role LUT offset ABI; routed learned codebooks require "
        "Gridbook "
        f">={GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION}. Dense "
        "Linear CBL remains supported."
    )


def _keying_implied_by_cell_qname(qname: str) -> str | None:
    """Keying a routed cell name proves on its own, or ``None`` if ambiguous.

    A cell named after the packed parent (``...experts.gate_up_proj``) pools
    both projections; one named after a half (``...experts.gate_proj``) does
    not.  ``down_proj`` is a one-projection stack, so its book is the same
    object under either keying and the name proves nothing.
    """

    leaf = str(qname).rsplit(".", 1)[-1]
    if leaf == "down_proj":
        return None
    if leaf in ROUTED_STACK_KEYS:
        return ROUTED_BOOK_KEYING_STACK
    if leaf in {"gate_proj", "up_proj"}:
        return ROUTED_BOOK_KEYING_ROLE
    return None


def resolve_routed_book_keying(
    qname: str,
    declared: object | None,
    *,
    fallback: str = DEFAULT_ROUTED_BOOK_KEYING,
) -> str:
    """Return the keying to record for one routed learned cell.

    A declared value must agree with whatever the cell name already proves;
    with nothing declared the name decides, and an ambiguous name (a
    one-projection ``down_proj`` stack) takes *fallback*.
    """

    implied = _keying_implied_by_cell_qname(qname)
    if declared is None:
        return implied or normalize_routed_book_keying(fallback)
    keying = normalize_routed_book_keying(declared)
    if implied is not None and keying != implied:
        raise ValueError(
            f"{qname}: cell name implies {implied} keying but the build "
            f"declared {keying}"
        )
    return keying


def _wq_pattern(col_weight: torch.Tensor) -> torch.Tensor:
    """Production vector-weight pattern for one imatrix row."""

    return cb._col_weight_vectors(col_weight.reshape(-1, cb.VEC_DIM))


def learn_pool(
    weight: torch.Tensor,
    col_weights: torch.Tensor,
    rung: int,
) -> tuple[torch.Tensor, ...]:
    """FABLE-certified pooled weighted-Lloyd FP8 product book.

    ``weight`` is ``[population, output_rows, input_features]`` and
    ``col_weights`` is ``[population, input_features]`` (a singleton middle
    dimension is accepted).  Dense Linears use a population of one.  The
    random-number call order intentionally matches the certified study kernel.
    """

    rung = int(rung)
    canonical, family, _ = _canonical_format(f"FP8_CB_K{rung}")
    if family.grid != "fp8" or family.mode != "product":
        raise ValueError(f"{canonical}: learn_pool requires an FP8 product rung")
    weight = torch.as_tensor(weight)
    col_weights = torch.as_tensor(col_weights)
    if weight.ndim != 3:
        raise ValueError(
            "learn_pool weight must have shape [population, rows, in_features], "
            f"got {tuple(weight.shape)}"
        )
    population, rows, in_features = (int(dim) for dim in weight.shape)
    if population <= 0 or rows <= 0 or in_features <= 0:
        raise ValueError(f"learn_pool weight shape must be positive, got {tuple(weight.shape)}")
    if in_features % cb.VEC_DIM:
        raise ValueError(
            f"learn_pool in_features={in_features} is not divisible by {cb.VEC_DIM}"
        )
    if col_weights.ndim == 3 and int(col_weights.shape[1]) == 1:
        col_weights = col_weights[:, 0, :]
    if tuple(col_weights.shape) != (population, in_features):
        raise ValueError(
            "learn_pool col_weights must have shape [population, in_features], "
            f"got {tuple(col_weights.shape)} for weight {tuple(weight.shape)}"
        )

    device = weight.device
    generator = torch.Generator().manual_seed(LLOYD_ROW_SEED)
    selected_rows = torch.randperm(rows, generator=generator)[:LLOYD_ROW_SAMPLE]
    vectors: list[torch.Tensor] = []
    vector_weights: list[torch.Tensor] = []
    for population_index in range(population):
        values, _, _ = cb._scale_and_vectorize(
            weight[population_index, selected_rows].to(torch.float32), "fp8"
        )
        vectors.append(values)
        pattern = _wq_pattern(
            col_weights[population_index].to(device)
        ).unsqueeze(0).expand(
            len(selected_rows), in_features // cb.VEC_DIM, cb.VEC_DIM
        ).reshape(-1, cb.VEC_DIM)
        vector_weights.append(pattern)
    pooled_values = torch.cat(vectors)
    pooled_weights = torch.cat(vector_weights)
    selected = torch.randperm(
        pooled_values.shape[0], generator=generator
    )[:LLOYD_CAP].to(device)
    pooled_values = pooled_values[selected]
    pooled_weights = pooled_weights[selected]

    n_sub = family_for("fp8", "product").n_sub
    widths = subtable_bit_widths(rung, "product", n_sub)
    sub_dim = cb.VEC_DIM // n_sub
    return tuple(
        cb.learn_codebook(
            pooled_values[:, index * sub_dim:(index + 1) * sub_dim],
            bits,
            grid="fp8",
            col_weights=pooled_weights[
                :, index * sub_dim:(index + 1) * sub_dim
            ],
            init=cb.fixed_lattice(bits, "fp8", sub_dim).to(device),
            iters=LLOYD_ITERS,
            seed=LLOYD_SEED,
        )
        for index, bits in enumerate(widths)
    )


def _stable_qname_generator(qname: str, *, purpose: str) -> torch.Generator:
    name = str(qname).strip()
    if not name:
        raise ValueError("learned-v2 row selection needs a nonempty qname")
    digest = hashlib.sha256(
        (CB_LEARNED_V2_SAMPLING_SCHEMA + "\0" + purpose + "\0" + name).encode(
            "utf-8"
        )
    ).digest()
    # ``manual_seed`` accepts signed 64-bit values on every supported torch.
    seed = int.from_bytes(digest[:8], byteorder="little") & ((1 << 63) - 1)
    return torch.Generator(device="cpu").manual_seed(seed)


def _index_sha256(value: torch.Tensor) -> str:
    raw = (
        torch.as_tensor(value)
        .detach()
        .to(device="cpu", dtype=torch.int64)
        .contiguous()
        .numpy()
        .astype("<i8", copy=False)
        .tobytes(order="C")
    )
    return hashlib.sha256(raw).hexdigest()


def learned_v2_sampling_plan(
    *,
    qname: str,
    output_rows: int,
    in_features: int,
    population: int,
    rung: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Return stable selected rows and the complete learned-v2 density record."""

    output_rows = int(output_rows)
    in_features = int(in_features)
    population = int(population)
    rung = int(rung)
    if output_rows <= 0 or in_features <= 0 or population <= 0:
        raise ValueError(
            "learned-v2 sampling dimensions must be positive, got "
            f"population={population}, output_rows={output_rows}, "
            f"in_features={in_features}"
        )
    if in_features % cb.VEC_DIM:
        raise ValueError(
            f"learned-v2 in_features={in_features} is not divisible by "
            f"{cb.VEC_DIM}"
        )
    canonical, family, _ = _canonical_format(f"FP8_CB_K{rung}")
    if family.grid != "fp8" or family.mode != "product":
        raise ValueError(f"{canonical}: learned-v2 requires an FP8 product rung")
    widths = subtable_bit_widths(rung, family.mode, family.n_sub)
    entries = 1 << max(widths)
    vectors_per_row = in_features // cb.VEC_DIM
    target_rows = (
        64 * entries + vectors_per_row - 1
    ) // vectors_per_row
    requested_rows = max(64, target_rows)
    sample_rows = min(output_rows, requested_rows)
    selected_rows = torch.randperm(
        output_rows,
        generator=_stable_qname_generator(qname, purpose="rows"),
    )[:sample_rows]
    available_vectors = population * sample_rows * vectors_per_row
    selected_vectors = min(available_vectors, LLOYD_CAP)
    target_vectors = 64 * entries
    density_shortfall_vectors = max(0, target_vectors - selected_vectors)
    provenance: dict[str, object] = {
        "schema": CB_LEARNED_V2_SAMPLING_SCHEMA,
        "source": "trained",
        "qname": str(qname),
        "rung": rung,
        "population": population,
        "output_rows": output_rows,
        "in_features": in_features,
        "subtable_bit_widths": list(widths),
        "entries": entries,
        "vectors_per_entry_target": 64,
        "vectors_per_row": vectors_per_row,
        "target_rows": target_rows,
        "requested_rows": requested_rows,
        "sample_rows": sample_rows,
        "row_selection_sha256": _index_sha256(selected_rows),
        "available_vectors": available_vectors,
        "vector_cap": LLOYD_CAP,
        "selected_vectors": selected_vectors,
        "target_vectors": target_vectors,
        "density_shortfall": density_shortfall_vectors > 0,
        "density_shortfall_vectors": density_shortfall_vectors,
        "achieved_vectors_per_entry": selected_vectors / entries,
        "centroid_accumulation": "stable_sort_segment_sum",
    }
    return selected_rows, provenance


def learn_pool_v2(
    weight: torch.Tensor,
    col_weights: torch.Tensor,
    rung: int,
    *,
    qname: str,
) -> LearnedV2PoolResult:
    """Density-aware, qname-stable learned FP8 product-codebook trainer.

    This is opt-in and receipt-gated at bundle construction.  The v1 trainer
    above remains untouched for old artifacts and legacy studies.
    """

    weight = torch.as_tensor(weight)
    col_weights = torch.as_tensor(col_weights)
    if weight.ndim != 3:
        raise ValueError(
            "learn_pool_v2 weight must have shape "
            "[population, rows, in_features], got "
            f"{tuple(weight.shape)}"
        )
    population, rows, in_features = (int(dim) for dim in weight.shape)
    if col_weights.ndim == 3 and int(col_weights.shape[1]) == 1:
        col_weights = col_weights[:, 0, :]
    if tuple(col_weights.shape) != (population, in_features):
        raise ValueError(
            "learn_pool_v2 col_weights must have shape "
            "[population, in_features], got "
            f"{tuple(col_weights.shape)} for weight {tuple(weight.shape)}"
        )
    selected_rows, provenance = learned_v2_sampling_plan(
        qname=qname,
        output_rows=rows,
        in_features=in_features,
        population=population,
        rung=rung,
    )
    device = weight.device
    cuda_capability = (
        list(torch.cuda.get_device_capability(device))
        if device.type == "cuda"
        else None
    )
    provenance["repeat_scope"] = {
        "policy": "exact_within_fixed_build_device",
        "torch_version": str(torch.__version__),
        "cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "device_type": device.type,
        "cuda_capability": cuda_capability,
    }
    device_rows = selected_rows.to(device=device)
    vectors: list[torch.Tensor] = []
    vector_weights: list[torch.Tensor] = []
    for population_index in range(population):
        values, _, _ = cb._scale_and_vectorize(
            weight[population_index].index_select(0, device_rows).to(
                torch.float32
            ),
            "fp8",
        )
        vectors.append(values)
        pattern = _wq_pattern(
            col_weights[population_index].to(device)
        ).unsqueeze(0).expand(
            len(selected_rows), in_features // cb.VEC_DIM, cb.VEC_DIM
        ).reshape(-1, cb.VEC_DIM)
        vector_weights.append(pattern)
    pooled_values = torch.cat(vectors)
    pooled_weights = torch.cat(vector_weights)
    if pooled_values.shape[0] > LLOYD_CAP:
        selected_vectors = torch.randperm(
            pooled_values.shape[0],
            generator=_stable_qname_generator(
                qname,
                purpose=f"vector-cap-K{int(rung)}",
            ),
        )[:LLOYD_CAP]
        provenance["vector_selection_sha256"] = _index_sha256(selected_vectors)
        device_vectors = selected_vectors.to(device=device)
        pooled_values = pooled_values.index_select(0, device_vectors)
        pooled_weights = pooled_weights.index_select(0, device_vectors)
    else:
        provenance["vector_selection_sha256"] = _index_sha256(
            torch.arange(pooled_values.shape[0], dtype=torch.int64)
        )

    family = family_for("fp8", "product")
    widths = subtable_bit_widths(int(rung), "product", family.n_sub)
    sub_dim = cb.VEC_DIM // family.n_sub
    tables = tuple(
        cb.learn_codebook(
            pooled_values[:, index * sub_dim:(index + 1) * sub_dim],
            bits,
            grid="fp8",
            col_weights=pooled_weights[
                :, index * sub_dim:(index + 1) * sub_dim
            ],
            init=cb.fixed_lattice(bits, "fp8", sub_dim).to(device),
            iters=LLOYD_ITERS,
            seed=LLOYD_SEED,
            accumulation="fixed_order",
        )
        for index, bits in enumerate(widths)
    )
    return LearnedV2PoolResult(tables=tables, provenance=provenance)


def _validate_v2_training_provenance(
    value: object,
    *,
    qname: str,
    rung: int,
    where: str,
) -> Mapping[str, object]:
    record = _require_mapping(value, where=where)
    common = {"schema", "source", "qname", "rung"}
    if record.get("schema") != CB_LEARNED_V2_SAMPLING_SCHEMA:
        raise ValueError(f"{where}: learned-v2 sampling schema differs")
    if record.get("qname") != qname or record.get("rung") != rung:
        raise ValueError(f"{where}: learned-v2 sampling coordinates differ")
    source = record.get("source")
    if source == "pretrained":
        if set(record) != common:
            raise ValueError(f"{where}: pretrained provenance members differ")
        return record
    trained_members = common | {
        "population",
        "output_rows",
        "in_features",
        "subtable_bit_widths",
        "entries",
        "vectors_per_entry_target",
        "vectors_per_row",
        "target_rows",
        "requested_rows",
        "sample_rows",
        "row_selection_sha256",
        "available_vectors",
        "vector_cap",
        "selected_vectors",
        "target_vectors",
        "density_shortfall",
        "density_shortfall_vectors",
        "achieved_vectors_per_entry",
        "centroid_accumulation",
        "vector_selection_sha256",
        "repeat_scope",
    }
    if source != "trained" or set(record) != trained_members:
        raise ValueError(f"{where}: trained provenance members differ")

    def positive_int(name: str) -> int:
        raw = record.get(name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"{where}.{name} must be a positive integer")
        return raw

    population = positive_int("population")
    rows = positive_int("output_rows")
    in_features = positive_int("in_features")
    entries = positive_int("entries")
    vectors_per_row = positive_int("vectors_per_row")
    target_rows = positive_int("target_rows")
    requested_rows = positive_int("requested_rows")
    sample_rows = positive_int("sample_rows")
    available_vectors = positive_int("available_vectors")
    vector_cap = positive_int("vector_cap")
    selected_vectors = positive_int("selected_vectors")
    target_vectors = positive_int("target_vectors")
    if in_features % cb.VEC_DIM:
        raise ValueError(f"{where}: in_features is not vector aligned")
    widths = subtable_bit_widths(rung, "product", 4)
    expected_entries = 1 << max(widths)
    expected_vectors_per_row = in_features // cb.VEC_DIM
    expected_target_rows = (
        64 * expected_entries + expected_vectors_per_row - 1
    ) // expected_vectors_per_row
    expected_sample_rows = min(rows, max(64, expected_target_rows))
    expected_available = population * expected_sample_rows * expected_vectors_per_row
    expected_selected = min(expected_available, LLOYD_CAP)
    expected_target_vectors = 64 * expected_entries
    expected_shortfall = max(0, expected_target_vectors - expected_selected)
    if (
        record.get("subtable_bit_widths") != list(widths)
        or entries != expected_entries
        or record.get("vectors_per_entry_target") != 64
        or vectors_per_row != expected_vectors_per_row
        or target_rows != expected_target_rows
        or requested_rows != max(64, expected_target_rows)
        or sample_rows != expected_sample_rows
        or available_vectors != expected_available
        or vector_cap != LLOYD_CAP
        or selected_vectors != expected_selected
        or target_vectors != expected_target_vectors
        or record.get("density_shortfall") != (expected_shortfall > 0)
        or record.get("density_shortfall_vectors") != expected_shortfall
        or record.get("centroid_accumulation") != "stable_sort_segment_sum"
    ):
        raise ValueError(f"{where}: learned-v2 sampling arithmetic differs")
    if expected_shortfall > 0:
        raise ValueError(
            f"{where}: promoted learned-v2 cell has a density shortfall; "
            "lattice wins"
        )
    achieved = record.get("achieved_vectors_per_entry")
    if isinstance(achieved, bool) or not isinstance(achieved, (int, float)):
        raise ValueError(f"{where}: achieved density is not numeric")
    if float(achieved) != selected_vectors / entries:
        raise ValueError(f"{where}: achieved density differs")
    for name in ("row_selection_sha256", "vector_selection_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get(name, ""))):
            raise ValueError(f"{where}.{name} is not a SHA-256")
    repeat_scope = _require_mapping(
        record.get("repeat_scope"), where=f"{where}.repeat_scope"
    )
    if set(repeat_scope) != {
        "policy",
        "torch_version",
        "cuda_version",
        "device_type",
        "cuda_capability",
    }:
        raise ValueError(f"{where}.repeat_scope members differ")
    if (
        repeat_scope.get("policy") != "exact_within_fixed_build_device"
        or not isinstance(repeat_scope.get("torch_version"), str)
        or not repeat_scope["torch_version"]
        or repeat_scope.get("device_type") not in {"cpu", "cuda"}
        or (
            repeat_scope.get("cuda_version") is not None
            and not isinstance(repeat_scope.get("cuda_version"), str)
        )
    ):
        raise ValueError(f"{where}.repeat_scope is malformed")
    capability = repeat_scope.get("cuda_capability")
    if repeat_scope["device_type"] == "cuda":
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in capability)
        ):
            raise ValueError(f"{where}.repeat_scope CUDA capability is malformed")
    elif capability is not None:
        raise ValueError(f"{where}.repeat_scope CPU capability must be null")
    return record


def canonical_codebook_refs(
    qname: str,
    format_name: str,
    *,
    source: str,
) -> tuple[str, ...]:
    """Physical sidecar names for one lattice format or learned cell."""

    canonical, family, rung = _canonical_format(format_name)
    source = str(source).strip().lower()
    if source not in {"lattice", "learned"}:
        raise ValueError(f"codebook source must be lattice/learned, got {source!r}")
    logical = "lattice" if source == "lattice" else str(qname).strip()
    if not logical:
        raise ValueError("learned codebook qname must be non-empty")
    count = len(codebook_subtable_shapes(rung, family.mode, family.n_sub))
    base = f"cb_codebook.{logical}.{canonical}"
    if count == 1:
        return (base,)
    return tuple(f"{base}.sub{index}" for index in range(count))


def canonical_fp16_table(tensor: torch.Tensor) -> torch.Tensor:
    """Return the exact CPU bytes the artifact sidecar will carry."""

    value = torch.as_tensor(tensor).detach().to(
        device="cpu", dtype=torch.float16
    ).contiguous()
    if value.ndim != 2:
        raise ValueError(f"codebook table must have rank 2, got {value.ndim}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("codebook table contains a non-finite FP16 value")
    return value


def codebook_table_sha256(tensor: torch.Tensor) -> str:
    """SHA-256 of one exact canonical FP16 subtable payload."""

    value = canonical_fp16_table(tensor)
    payload = value.numpy().astype("<f2", copy=False).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _tensor_identity(tensor: torch.Tensor) -> dict[str, object]:
    # This is the render-identity implementation already used by cache and warm
    # state.  Import lazily to keep bundle loading independent of cache setup.
    from prismaquant.production_weight_cache import _source_weight_value_identity

    shape, digest = _source_weight_value_identity(torch.as_tensor(tensor))
    return {"shape": shape, "sha256": digest}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _validated_complete_source_identity(
    identity: object,
    *,
    where: str,
) -> dict[str, object]:
    """Require the value-bearing, full indexed-checkpoint identity contract."""

    try:
        validated = validate_streamed_model_identity(identity, where=where)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    source = validated.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{where} requires a nonempty source model id")
    checkpoint_weight_map = validated.get("checkpoint_weight_map")
    if not isinstance(checkpoint_weight_map, dict) or not checkpoint_weight_map:
        raise ValueError(
            f"{where} requires a complete checkpoint_weight_map; a decoder-"
            "only or name-only identity cannot authorize learned-v2 promotion"
        )
    return validated


def _nonempty_binding(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a nonempty string")
    return value.strip()


def _strict_json_loads(raw: str, *, where: str) -> object:
    def reject_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"{where}: duplicate JSON member {key!r}")
            out[key] = value
        return out

    try:
        return json.loads(raw, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: malformed bundle metadata: {exc}") from exc


def _bundle_content_sha256(digests: Mapping[str, str]) -> str:
    payload = {
        "schema": CB_LEARNED_BUNDLE_SCHEMA,
        "codebook_content_sha256": dict(sorted(digests.items())),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_formats_by_qname(
    qnames: Iterable[str],
    formats: Sequence[str] | Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    normalized_qnames = {str(name) for name in qnames}
    if isinstance(formats, Mapping):
        raw = {str(name): tuple(values) for name, values in formats.items()}
    else:
        common = tuple(formats)
        raw = {name: common for name in normalized_qnames}
    if not raw:
        raise ValueError("learned bundle has no target cells")
    missing = sorted(set(raw) - normalized_qnames)
    if missing:
        raise ValueError(f"bundle formats name missing weight(s): {missing[:8]}")
    normalized: dict[str, tuple[str, ...]] = {}
    for qname, names in sorted(raw.items()):
        parsed_names = tuple(_canonical_format(name) for name in names)
        canonical = tuple(sorted({entry[0] for entry in parsed_names}))
        if not canonical:
            raise ValueError(f"{qname}: bundle has no CB formats")
        normalized[qname] = canonical
    return normalized


def _codebook_sequence(
    format_name: str,
    codebook: object,
) -> tuple[torch.Tensor, ...]:
    canonical, family, rung = _canonical_format(format_name)
    expected = codebook_subtable_shapes(rung, family.mode, family.n_sub)
    if isinstance(codebook, torch.Tensor):
        values = (codebook,)
    elif isinstance(codebook, (tuple, list)):
        values = tuple(codebook)
    else:
        raise TypeError(
            f"{canonical}: codebook must be a tensor or sequence, got "
            f"{type(codebook).__name__}"
        )
    if len(values) != len(expected):
        raise ValueError(
            f"{canonical}: expected {len(expected)} subtables, got {len(values)}"
        )
    result = tuple(canonical_fp16_table(value) for value in values)
    actual = tuple(tuple(int(dim) for dim in value.shape) for value in result)
    if actual != expected:
        raise ValueError(
            f"{canonical}: codebook shapes {actual} != canonical {expected}"
        )
    return result


def _lattice_codebook(format_name: str) -> tuple[torch.Tensor, ...]:
    canonical, family, rung = _canonical_format(format_name)
    resolved = cb._resolve_codebook(
        rung,
        family.grid,
        family.mode,
        None,
        torch.device("cpu"),
    )
    return _codebook_sequence(canonical, resolved)


@dataclass(frozen=True)
class PretrainedCodebookCell:
    """Value-bearing pretrained tables plus optional immutable provenance.

    The ordinary tensor/sequence spelling remains accepted and byte-identical.
    Production routed-MoE bank loading uses this wrapper so the bundle cell
    records which accepted burn shard and content-addressed file supplied the
    exact tables.
    """

    codebook: object
    origin: Mapping[str, object] | None = None


def _normalized_pretrained_origin(
    value: Mapping[str, object], *, where: str
) -> dict[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{where}: pretrained origin must be a nonempty object")
    try:
        normalized = json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{where}: pretrained origin is not strict JSON data: {exc}"
        ) from exc
    if not isinstance(normalized, dict) or not normalized:
        raise ValueError(f"{where}: pretrained origin must be a nonempty object")
    schema = normalized.get("schema")
    if not isinstance(schema, str) or not schema:
        raise ValueError(f"{where}: pretrained origin needs a schema")
    from prismaquant.cb_banked_books import (
        BANKED_CBL_ORIGIN_SCHEMA,
        BankedCBLBookError,
        validate_banked_cbl_origin,
    )

    if schema == BANKED_CBL_ORIGIN_SCHEMA:
        try:
            normalized = validate_banked_cbl_origin(
                normalized, where=where
            )
        except BankedCBLBookError as exc:
            raise ValueError(str(exc)) from exc
    return normalized


@dataclass(frozen=True)
class CBLearnedBundle:
    """A fully verified in-memory snapshot of one immutable ``.pqcb`` bundle."""

    path: Path
    manifest: Mapping[str, object]
    sidecar_tensors: Mapping[str, torch.Tensor]

    @property
    def codebook_content_digests(self) -> dict[str, str]:
        return dict(self.manifest["codebook_content_sha256"])

    @property
    def codebook_refs_by_cell(self) -> dict[str, dict[str, tuple[str, ...]]]:
        cells = self.manifest["cells"]
        resolved = {
            str(qname): {
                str(fmt): tuple(str(ref) for ref in cell["codebook_ref"])
                for fmt, cell in formats.items()
            }
            for qname, formats in cells.items()
        }
        for alias, entry in self.manifest.get("aliases", {}).items():
            resolved[str(alias)] = dict(resolved[str(entry["cell_qname"])])
        return resolved

    @property
    def codebook_source_by_format(self) -> dict[str, str]:
        """Return the bundle-authoritative per-rung source map.

        Bundle construction chooses learned formats once for the complete
        menu, so every qname carrying a given rung must agree.  Refuse a
        hand-edited bundle that tries to make source selection qname-local:
        cost/export provenance is deliberately a compact per-rung map, and a
        non-uniform map could not be represented without weakening identity.
        """

        resolved: dict[str, str] = {}
        owners: dict[str, str] = {}
        for raw_qname, raw_formats in self.manifest["cells"].items():
            qname = str(raw_qname)
            for raw_format, raw_cell in raw_formats.items():
                canonical, _family, _rung = _canonical_format(raw_format)
                source = str(raw_cell["source"])
                previous = resolved.setdefault(canonical, source)
                if previous != source:
                    raise ValueError(
                        f"{self.path}: bundle source for {canonical} differs "
                        f"between {owners[canonical]!r} ({previous}) and "
                        f"{qname!r} ({source}); per-rung render identity must "
                        "be uniform"
                    )
                owners.setdefault(canonical, qname)
        return dict(sorted(resolved.items()))

    @property
    def bundle_content_sha256(self) -> str:
        return str(self.manifest["bundle_content_sha256"])

    def cell(self, qname: str, format_name: str) -> Mapping[str, object]:
        canonical, _family, _rung = _canonical_format(format_name)
        requested = str(qname)
        alias = self.manifest.get("aliases", {}).get(requested)
        cell_qname = str(alias["cell_qname"]) if alias is not None else requested
        try:
            return self.manifest["cells"][cell_qname][canonical]
        except KeyError as exc:
            raise ValueError(
                f"{qname}: immutable learned bundle {self.path} has no "
                f"{canonical} cell; refusing lattice fallback"
            ) from exc

    def has_cell(self, qname: str, format_name: str) -> bool:
        """Whether this bundle carries one exact ``(qname, format)`` cell."""

        try:
            self.cell(qname, format_name)
        except ValueError:
            return False
        return True

    def routed_book_keying(self, qname: str, format_name: str) -> str:
        """Which rule burned one routed learned book.

        A cell written before campaign rule R1 records nothing, and those books
        are per role by construction, so an absent field reads as ``"role"``.
        """

        cell = self.cell(qname, format_name)
        return str(
            cell.get("routed_book_keying", ROUTED_BOOK_KEYING_ROLE)
        )

    def validate_inputs(
        self,
        qname: str,
        *,
        weight: torch.Tensor,
        col_weights: torch.Tensor,
    ) -> None:
        requested = str(qname)
        expected = self.manifest.get("aliases", {}).get(requested)
        if expected is None:
            try:
                expected = self.manifest["inputs"][requested]
            except KeyError as exc:
                raise ValueError(
                    f"{qname}: bundle has no source/imatrix identity"
                ) from exc
        actual_weight = _tensor_identity(weight)
        actual_col = _tensor_identity(col_weights)
        if actual_weight != expected["source_weight"]:
            raise ValueError(
                f"{qname}: source weight does not match learned bundle identity"
            )
        if actual_col != expected["col_weights"]:
            raise ValueError(
                f"{qname}: col_weights do not match learned bundle identity"
            )

    def codebook_for(
        self,
        qname: str,
        format_name: str,
        *,
        weight: torch.Tensor | None = None,
        col_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Resolve exact learned values, never a lattice fallback."""

        cell = self.cell(qname, format_name)
        if cell["source"] != "learned":
            raise ValueError(
                f"{qname}: {format_name} is {cell['source']} in bundle {self.path}, "
                "not learned"
            )
        if (weight is None) != (col_weights is None):
            raise ValueError(
                "learned bundle input validation needs both weight and col_weights"
            )
        if weight is not None:
            refuse_routed_moe_learned(qname, weight=weight)
            self.validate_inputs(
                qname, weight=weight, col_weights=col_weights
            )
        refs = tuple(str(ref) for ref in cell["codebook_ref"])
        try:
            # The sidecar bytes are FP16, but the reference encoder and
            # decoder perform lookup/multiply in FP32.  Promote only *after*
            # reload so every render is derived from the exact serialized
            # values, never from the trainer's pre-materialization tensors.
            return tuple(
                self.sidecar_tensors[ref].to(torch.float32) for ref in refs
            )
        except KeyError as exc:
            raise ValueError(
                f"{qname}: learned bundle is missing materialized ref {exc.args[0]!r}"
            ) from exc


def train_and_save_bundle(
    path: str | Path,
    *,
    weights: Mapping[str, torch.Tensor] | None = None,
    qnames: Iterable[str] | None = None,
    weight_provider: Callable[[str], torch.Tensor] | None = None,
    col_weights: Mapping[str, torch.Tensor],
    formats: Sequence[str] | Mapping[str, Sequence[str]],
    learned_formats: Iterable[str] | None = None,
    trainer_version: str = "v1",
    promotion_receipt: (
        Mapping[str, object] | ValidatedCBLPromotionReceipt | None
    ) = None,
    source_model_identity: Mapping[str, object] | None = None,
    probe_calibration_hash: str | None = None,
    imatrix_value_sha256: str | None = None,
    routed_moe_qnames: Iterable[str] = (),
    routed_book_keying: str | Mapping[str, str] | None = None,
    pretrained_codebooks: Mapping[tuple[str, str], object] | None = None,
    pretrained_codebook_provider: Callable[
        [str, str, torch.Tensor, torch.Tensor], object | None
    ] | None = None,
    input_alias_provider: Callable[
        [str, torch.Tensor, torch.Tensor],
        Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    ] | None = None,
) -> CBLearnedBundle:
    """Train cells, canonicalize FP16 values, and publish one immutable bundle.

    The default ``trainer_version="v1"`` preserves the historical policy and
    artifact identity.  Version 2 defaults every K4..K48 step-4 rung to
    lattice and promotes a rung only when a complete two-holdout
    ``promotion_receipt`` says learned.  With v2, an explicit
    ``learned_formats`` must exactly match the receipt-derived sources; it
    cannot override them.  The formats may be common to all qnames or supplied
    per qname.  For large
    models, pass ``qnames`` plus ``weight_provider`` instead of ``weights``:
    each decoded weight is requested once, all of its rung cells are trained,
    and it is released before the next qname.  This keeps residency to one
    source tensor plus the tiny accumulated FP16 books.

    ``pretrained_codebooks`` supplies immutable, value-bearing cells keyed by
    ``(qname, canonical_format)``.  The provider spelling is invoked only
    after that qname's current weight and imatrix tensors are resident, so a
    bank loader can verify their exact identities before returning bytes.  A
    supplied cell (static or provider-backed) is the only production route for
    a rank-3 routed-expert population: expert books measured during a burn must
    be copied byte-for-byte into the bundle, never retrained at export time.

    ``routed_book_keying`` records which rule burned each routed learned book —
    ``"stack"`` (one book per ``(layer, stack, rung)``, gate and up pooled) or
    ``"role"`` (one book per ``(layer, projection, rung)``).  A book's
    calibration is its identity, and so is the population it was pooled over,
    so the keying is written onto every routed learned cell.  Pass one value
    for the whole build or a per-qname mapping; omit it and each cell's name
    decides.
    """

    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable learned bundle {path}"
        )
    if weights is not None:
        if qnames is not None or weight_provider is not None:
            raise ValueError(
                "pass either weights or qnames+weight_provider, not both"
            )
        normalized_weights = {
            str(name): torch.as_tensor(value) for name, value in weights.items()
        }
        target_names = tuple(normalized_weights)

        def provide_weight(name: str) -> torch.Tensor:
            return normalized_weights[name]
    else:
        if qnames is None or weight_provider is None:
            raise ValueError(
                "bundle training requires weights or qnames+weight_provider"
            )
        target_names = tuple(dict.fromkeys(str(name) for name in qnames))
        if not target_names:
            raise ValueError("bundle training qnames must be non-empty")
        provide_weight = weight_provider
    selected_trainer = _trainer_stamp(trainer_version)
    trainer_schema = str(selected_trainer["schema"])
    is_v2 = trainer_schema == CB_LEARNED_TRAINER_V2_SCHEMA
    if not is_v2 and promotion_receipt is not None:
        raise ValueError("promotion_receipt is valid only with trainer_version='v2'")
    normalized_col = {
        str(name): torch.as_tensor(value)
        for name, value in col_weights.items()
    }
    formats_by_qname = _normalize_formats_by_qname(target_names, formats)
    target_qnames = set(formats_by_qname)
    missing_col = sorted(target_qnames - set(normalized_col))
    if missing_col:
        raise ValueError(f"bundle cells are missing col_weights: {missing_col[:8]}")
    validated_receipt: ValidatedCBLPromotionReceipt | None = None
    validated_source_identity: dict[str, object] | None = None
    bound_probe_calibration_hash: str | None = None
    bound_imatrix_sha256: str | None = None
    if promotion_receipt is not None:
        validated_source_identity = _validated_complete_source_identity(
            source_model_identity,
            where="learned-v2 promotion source model identity",
        )
        bound_probe_calibration_hash = _nonempty_binding(
            probe_calibration_hash,
            where="learned-v2 actual probe calibration hash",
        )
        observed_imatrix_sha256 = canonical_imatrix_sha256(normalized_col)
        bound_imatrix_sha256 = _nonempty_binding(
            imatrix_value_sha256,
            where="learned-v2 actual imatrix value_sha256",
        ).lower()
        if bound_imatrix_sha256 != observed_imatrix_sha256:
            raise ValueError(
                "learned-v2 supplied imatrix value_sha256 differs from the "
                "bundle col_weights"
            )
        raw_receipt = (
            promotion_receipt.payload
            if isinstance(promotion_receipt, ValidatedCBLPromotionReceipt)
            else promotion_receipt
        )
        qname_census = role_census_for_qnames(target_qnames)
        validated_receipt = validate_promotion_receipt(
            raw_receipt,
            expected_model_id=str(validated_source_identity["source"]),
            expected_model_content_sha256=str(
                validated_source_identity["content_sha256"]
            ),
            expected_calibration_hash=bound_probe_calibration_hash,
            expected_imatrix_sha256=bound_imatrix_sha256,
            expected_role_census=qname_census,
            expected_qnames=target_qnames,
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
            "learned-v2 promotion bindings are valid only with a promotion "
            "receipt"
        )
    active_rung_policy = (
        receipt_rung_policy(validated_receipt)
        if validated_receipt is not None
        else (_default_v2_rung_policy() if is_v2 else CBL_RUNG_POLICY)
    )
    routed = {str(name) for name in routed_moe_qnames}
    if routed_book_keying is None or isinstance(routed_book_keying, str):
        declared_keying: dict[str, object] = {
            qname: routed_book_keying for qname in routed
        }
    else:
        declared_keying = {
            str(name): value for name, value in routed_book_keying.items()
        }
        unknown_keying = sorted(set(declared_keying) - routed)
        if unknown_keying:
            raise ValueError(
                "routed_book_keying names non-routed qname(s): "
                f"{unknown_keying[:8]}"
            )
    # A ``down_proj`` cell is one projection either way, so its name proves no
    # keying.  Let the build's own unambiguous cells answer for it before
    # falling back to the campaign default: a bundle whose w13 books are per
    # role is a per-role bundle throughout.
    build_implied = {
        implied
        for qname in routed
        if (implied := _keying_implied_by_cell_qname(qname)) is not None
    }
    fallback_keying = (
        next(iter(build_implied)) if len(build_implied) == 1
        else DEFAULT_ROUTED_BOOK_KEYING
    )
    keying_by_qname = {
        qname: resolve_routed_book_keying(
            qname,
            declared_keying.get(qname),
            fallback=fallback_keying,
        )
        for qname in sorted(routed)
    }
    supplied_books: dict[tuple[str, str], object] = {}
    for raw_key, value in (pretrained_codebooks or {}).items():
        if (
            not isinstance(raw_key, tuple)
            or len(raw_key) != 2
            or not str(raw_key[0]).strip()
        ):
            raise ValueError(
                "pretrained_codebooks keys must be (qname, format) pairs"
            )
        key = (str(raw_key[0]), _canonical_format(str(raw_key[1]))[0])
        if key in supplied_books:
            raise ValueError(f"duplicate pretrained codebook cell {key}")
        supplied_books[key] = value
    expected_cells = {
        (qname, fmt)
        for qname, names in formats_by_qname.items()
        for fmt in names
    }
    unknown_books = sorted(set(supplied_books) - expected_cells)
    if unknown_books:
        raise ValueError(
            "pretrained_codebooks contains cells absent from the bundle: "
            f"{unknown_books[:8]}"
        )

    supplied_formats = {
        fmt for names in formats_by_qname.values() for fmt in names
    }
    if is_v2:
        off_ladder = sorted({
            fmt
            for fmt in supplied_formats
            if not _canonical_format(fmt)[1].is_producer_rung(
                _canonical_format(fmt)[2]
            )
        })
        if off_ladder:
            raise ValueError(
                "learned-v2 cannot produce legacy off-law rung(s): "
                f"{off_ladder}"
            )
    policy_learned = {
        fmt for fmt in supplied_formats
        if (
            _canonical_format(fmt)[1].grid == "fp8"
            and active_rung_policy.get(
                _canonical_format(fmt)[2], {}
            ).get("enabled") is True
        )
    }
    if learned_formats is None:
        learned = policy_learned
    else:
        learned = {_canonical_format(fmt)[0] for fmt in learned_formats}
        if is_v2 and learned != policy_learned:
            raise ValueError(
                "learned-v2 learned_formats must equal the promotion "
                "receipt-derived source set: "
                f"expected={sorted(policy_learned)}, got={sorted(learned)}"
            )
    unknown_learned = sorted(learned - supplied_formats)
    if unknown_learned:
        raise ValueError(
            f"learned_formats are absent from bundle cells: {unknown_learned}"
        )
    for fmt in sorted(learned):
        _canonical, family, rung = _canonical_format(fmt)
        if family.grid != "fp8" or family.mode != "product":
            raise ValueError(
                f"{fmt}: production learned bundle refuses NVFP4 CBL; it is "
                "measured NO-GO"
            )
        if is_v2:
            if rung not in CBL_STEP4_RUNGS:
                raise ValueError(
                    f"{fmt}: learned-v2 supports only the K4..K48 step-4 ladder"
                )
            if validated_receipt is None:
                raise ValueError(
                    f"{fmt}: learned-v2 requires a validated promotion receipt"
                )
            if active_rung_policy[rung]["enabled"] is not True:
                raise ValueError(
                    f"{fmt}: learned-v2 receipt did not promote this rung"
                )
        else:
            require_cbl_rung_enabled(rung)

    inputs: dict[str, dict[str, object]] = {}
    aliases: dict[str, dict[str, object]] = {}
    cells: dict[str, dict[str, dict[str, object]]] = {}
    tensors: dict[str, torch.Tensor] = {}
    owners: dict[str, tuple[str, str, str]] = {}

    for qname in sorted(formats_by_qname):
        weight = torch.as_tensor(provide_weight(qname))
        cw = normalized_col[qname]
        if not weight.is_floating_point():
            raise ValueError(
                f"{qname}: bundle training requires decoded floating-point weight values"
            )
        inputs[qname] = {
            "source_weight": _tensor_identity(weight),
            "col_weights": _tensor_identity(cw),
        }
        if input_alias_provider is not None:
            raw_aliases = input_alias_provider(qname, weight, cw)
            for raw_alias, pair in raw_aliases.items():
                alias = str(raw_alias)
                if not alias or alias in aliases or alias in formats_by_qname:
                    raise ValueError(
                        f"{qname}: duplicate or invalid bundle input alias "
                        f"{alias!r}"
                    )
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise ValueError(
                        f"{qname}: input alias {alias!r} must supply "
                        "(weight, col_weights)"
                    )
                alias_weight, alias_col = pair
                aliases[alias] = {
                    "cell_qname": qname,
                    "source_weight": _tensor_identity(alias_weight),
                    "col_weights": _tensor_identity(alias_col),
                }
        cells[qname] = {}
        for fmt in formats_by_qname[qname]:
            canonical, family, rung = _canonical_format(fmt)
            source = "learned" if canonical in learned else "lattice"
            supplied = supplied_books.get((qname, canonical))
            if source == "learned" and pretrained_codebook_provider is not None:
                provided = pretrained_codebook_provider(
                    qname, canonical, weight, cw
                )
                if supplied is not None and provided is not None:
                    raise ValueError(
                        f"{qname}/{canonical}: both static and provider-backed "
                        "pretrained books were supplied"
                    )
                if provided is not None:
                    supplied = provided
            pretrained_origin = None
            if isinstance(supplied, PretrainedCodebookCell):
                if supplied.origin is not None:
                    pretrained_origin = _normalized_pretrained_origin(
                        supplied.origin,
                        where=f"{qname}/{canonical}",
                    )
                supplied = supplied.codebook
            if supplied is not None and source != "learned":
                raise ValueError(
                    f"{qname}/{canonical}: a pretrained book was supplied for "
                    "a lattice cell"
                )
            training_provenance: Mapping[str, object] | None = None
            if source == "learned":
                refuse_routed_moe_learned(
                    qname,
                    routed_moe=qname in routed,
                    weight=weight,
                )
                if weight.ndim not in (2, 3):
                    raise ValueError(
                        f"{qname}: learned CBL expects a rank-2 Linear or a "
                        "rank-3 routed-expert population, "
                        f"got shape {tuple(weight.shape)}"
                    )
                in_features = int(weight.shape[-1])
                population = int(weight.shape[0]) if weight.ndim == 3 else 1
                if cw.numel() != population * in_features:
                    raise ValueError(
                        f"{qname}: learned CBL col_weights has {cw.numel()} "
                        f"values, expected {population}x{in_features}"
                    )
                if is_v2:
                    _rows, preview_provenance = learned_v2_sampling_plan(
                        qname=qname,
                        output_rows=int(weight.shape[-2]),
                        in_features=in_features,
                        population=population,
                        rung=rung,
                    )
                    if preview_provenance["density_shortfall"] is True:
                        raise ValueError(
                            f"{qname}/{canonical}: promotion receipt says "
                            "learned but the current source matrix has a "
                            "learned-v2 density shortfall; lattice wins"
                        )
                if supplied is not None:
                    tables = _codebook_sequence(canonical, supplied)
                    if is_v2:
                        training_provenance = {
                            "schema": CB_LEARNED_V2_SAMPLING_SCHEMA,
                            "source": "pretrained",
                            "qname": qname,
                            "rung": rung,
                        }
                elif weight.ndim == 3:
                    raise ValueError(
                        f"{qname}/{canonical}: routed-MoE learned CBL requires "
                        "an immutable banked pretrained_codebooks cell; "
                        "retraining is forbidden"
                    )
                else:
                    if is_v2:
                        trained_v2 = learn_pool_v2(
                            weight.unsqueeze(0),
                            cw.reshape(1, in_features),
                            rung,
                            qname=qname,
                        )
                        training_provenance = trained_v2.provenance
                        if training_provenance["density_shortfall"] is True:
                            raise ValueError(
                                f"{qname}/{canonical}: learned-v2 trainer "
                                "reported a density shortfall after preview; "
                                "lattice wins"
                            )
                        trained = trained_v2.tables
                    else:
                        trained = learn_pool(
                            weight.unsqueeze(0),
                            cw.reshape(1, in_features),
                            rung,
                        )
                    tables = _codebook_sequence(canonical, trained)
            else:
                tables = _lattice_codebook(canonical)
            refs = canonical_codebook_refs(qname, canonical, source=source)
            digests = [codebook_table_sha256(table) for table in tables]
            if is_v2 and source == "learned":
                assert validated_receipt is not None
                expected_candidate_digests = (
                    validated_receipt.candidate_digests(qname, rung)
                )
                if tuple(digests) != expected_candidate_digests:
                    raise ValueError(
                        f"{qname}/{canonical}: materialized learned table "
                        "digests differ from the exact promotion candidate; "
                        "refusing arbitrary retraining or bank substitution"
                    )
            for ref, table, digest in zip(
                refs, tables, digests, strict=True
            ):
                previous = tensors.get(ref)
                if previous is not None and not torch.equal(previous, table):
                    raise ValueError(
                        f"physical codebook ref {ref!r} has conflicting values"
                    )
                previous_owner = owners.get(ref)
                owner = (qname, canonical, source)
                if (
                    previous_owner is not None
                    and source == "learned"
                    and previous_owner[:2] != owner[:2]
                ):
                    raise ValueError(
                        f"learned cells {previous_owner[:2]} and {owner[:2]} "
                        f"share physical ref {ref!r}"
                    )
                tensors.setdefault(ref, table)
                owners.setdefault(ref, owner)
            cells[qname][canonical] = {
                "source": source,
                "codebook_ref": list(refs),
                "content_sha256": digests,
                **({
                    "rung_policy": dict(active_rung_policy[rung]),
                } if source == "learned" else {}),
                **({
                    "training_provenance": dict(training_provenance),
                } if training_provenance is not None else {}),
                **({
                    "routed_book_keying": keying_by_qname[qname],
                } if source == "learned" and qname in keying_by_qname else {}),
                **({
                    "pretrained_origin": pretrained_origin,
                } if pretrained_origin is not None else {}),
            }
        # The provider owns any external cache.  This function deliberately
        # retains no source weight after the qname's cells are materialized.
        del weight

    content_digests = {
        name: codebook_table_sha256(tensor)
        for name, tensor in sorted(tensors.items())
    }
    manifest: dict[str, object] = {
        "schema": CB_LEARNED_BUNDLE_SCHEMA,
        "trainer": selected_trainer,
        "rung_policy": {
            str(rung): dict(policy)
            for rung, policy in sorted(active_rung_policy.items())
        },
        **({
            "promotion_receipt": dict(validated_receipt.payload),
            "promotion_bindings": {
                "source_model_identity": validated_source_identity,
                "probe_calibration_hash": bound_probe_calibration_hash,
                "imatrix_value_sha256": bound_imatrix_sha256,
            },
        } if validated_receipt is not None else {}),
        "inputs": inputs,
        **({"aliases": aliases} if aliases else {}),
        "cells": cells,
        "tensor_shapes": {
            name: [int(dim) for dim in tensor.shape]
            for name, tensor in sorted(tensors.items())
        },
        "codebook_content_sha256": content_digests,
        "bundle_content_sha256": _bundle_content_sha256(content_digests),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
        save_file(
            tensors,
            temp_name,
            metadata={CB_LEARNED_BUNDLE_METADATA_KEY: _canonical_json(manifest)},
        )
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite immutable learned bundle {path}"
            )
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
    return load_bundle(path)


def train_and_save_bundle_streaming(
    path: str | Path,
    *,
    qnames: Iterable[str],
    weight_provider: Callable[[str], torch.Tensor],
    col_weights: Mapping[str, torch.Tensor],
    formats: Sequence[str] | Mapping[str, Sequence[str]],
    learned_formats: Iterable[str] | None = None,
    trainer_version: str = "v1",
    promotion_receipt: (
        Mapping[str, object] | ValidatedCBLPromotionReceipt | None
    ) = None,
    source_model_identity: Mapping[str, object] | None = None,
    probe_calibration_hash: str | None = None,
    imatrix_value_sha256: str | None = None,
    routed_moe_qnames: Iterable[str] = (),
    routed_book_keying: str | Mapping[str, str] | None = None,
    pretrained_codebooks: Mapping[tuple[str, str], object] | None = None,
    pretrained_codebook_provider: Callable[
        [str, str, torch.Tensor, torch.Tensor], object | None
    ] | None = None,
    input_alias_provider: Callable[
        [str, torch.Tensor, torch.Tensor],
        Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    ] | None = None,
) -> CBLearnedBundle:
    """Explicit spelling of the one-source-tensor residency API."""

    return train_and_save_bundle(
        path,
        qnames=qnames,
        weight_provider=weight_provider,
        col_weights=col_weights,
        formats=formats,
        learned_formats=learned_formats,
        trainer_version=trainer_version,
        promotion_receipt=promotion_receipt,
        source_model_identity=source_model_identity,
        probe_calibration_hash=probe_calibration_hash,
        imatrix_value_sha256=imatrix_value_sha256,
        routed_moe_qnames=routed_moe_qnames,
        routed_book_keying=routed_book_keying,
        pretrained_codebooks=pretrained_codebooks,
        pretrained_codebook_provider=pretrained_codebook_provider,
        input_alias_provider=input_alias_provider,
    )


def _require_mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    return value


def load_bundle(path: str | Path) -> CBLearnedBundle:
    """Load one stable snapshot and verify every name, shape, and FP16 digest."""

    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        raw_manifest = metadata.get(CB_LEARNED_BUNDLE_METADATA_KEY)
        if raw_manifest is None:
            raise ValueError(
                f"{path}: missing {CB_LEARNED_BUNDLE_METADATA_KEY!r} metadata"
            )
        manifest = _strict_json_loads(raw_manifest, where=str(path))
        names = tuple(handle.keys())
        tensors = {
            name: handle.get_tensor(name).detach().clone().contiguous()
            for name in names
        }
    manifest = _require_mapping(manifest, where=f"{path} manifest")
    if manifest.get("schema") != CB_LEARNED_BUNDLE_SCHEMA:
        raise ValueError(
            f"{path}: unsupported learned bundle schema {manifest.get('schema')!r}"
        )
    required_top_level = {
        "schema",
        "trainer",
        "rung_policy",
        "inputs",
        "cells",
        "tensor_shapes",
        "codebook_content_sha256",
        "bundle_content_sha256",
    }
    allowed_top_level = required_top_level | {
        "aliases",
        "promotion_receipt",
        "promotion_bindings",
    }
    if not required_top_level <= set(manifest) or not set(manifest) <= allowed_top_level:
        raise ValueError(
            f"{path}: learned bundle manifest members differ: "
            f"missing={sorted(required_top_level - set(manifest))}, "
            f"unknown={sorted(set(manifest) - allowed_top_level)}"
        )
    trainer = _require_mapping(manifest.get("trainer"), where=f"{path} trainer")
    receipt: ValidatedCBLPromotionReceipt | None = None
    if dict(trainer) == CB_LEARNED_TRAINER_STAMP:
        is_v2 = False
        if "promotion_receipt" in manifest or "promotion_bindings" in manifest:
            raise ValueError(
                f"{path}: v1 bundle cannot carry learned-v2 promotion data"
            )
        load_rung_policy = CBL_RUNG_POLICY
    elif dict(trainer) == CB_LEARNED_TRAINER_V2_STAMP:
        is_v2 = True
        raw_receipt = manifest.get("promotion_receipt")
        if raw_receipt is None:
            if "promotion_bindings" in manifest:
                raise ValueError(
                    f"{path}: learned-v2 lattice-default bundle cannot carry "
                    "orphaned promotion bindings"
                )
            load_rung_policy = _default_v2_rung_policy()
        else:
            bindings = _require_mapping(
                manifest.get("promotion_bindings"),
                where=f"{path} promotion_bindings",
            )
            expected_binding_members = {
                "source_model_identity",
                "probe_calibration_hash",
                "imatrix_value_sha256",
            }
            if set(bindings) != expected_binding_members:
                raise ValueError(
                    f"{path}: promotion binding members differ"
                )
            source_identity = _validated_complete_source_identity(
                bindings.get("source_model_identity"),
                where=f"{path} promotion source model identity",
            )
            calibration_hash = _nonempty_binding(
                bindings.get("probe_calibration_hash"),
                where=f"{path} promotion probe calibration hash",
            )
            imatrix_sha256 = _nonempty_binding(
                bindings.get("imatrix_value_sha256"),
                where=f"{path} promotion imatrix value_sha256",
            ).lower()
            if re.fullmatch(r"[0-9a-f]{64}", imatrix_sha256) is None:
                raise ValueError(
                    f"{path}: promotion imatrix value_sha256 is malformed"
                )
            raw_cells_for_census = _require_mapping(
                manifest.get("cells"), where=f"{path} cells"
            )
            cell_qnames = tuple(str(name) for name in raw_cells_for_census)
            role_census = role_census_for_qnames(cell_qnames)
            receipt = validate_promotion_receipt(
                _require_mapping(
                    raw_receipt,
                    where=f"{path} promotion_receipt",
                ),
                expected_model_id=str(source_identity["source"]),
                expected_model_content_sha256=str(
                    source_identity["content_sha256"]
                ),
                expected_calibration_hash=calibration_hash,
                expected_imatrix_sha256=imatrix_sha256,
                expected_role_census=role_census,
                expected_qnames=cell_qnames,
            )
            load_rung_policy = receipt_rung_policy(receipt)
    else:
        raise ValueError(f"{path}: learned bundle trainer identity differs")
    observed_policy = _require_mapping(
        manifest.get("rung_policy"), where=f"{path} rung_policy"
    )
    expected_policy = {
        str(rung): dict(policy)
        for rung, policy in sorted(load_rung_policy.items())
    }
    if dict(observed_policy) != expected_policy:
        raise ValueError(f"{path}: learned bundle rung policy differs")
    digests = _require_mapping(
        manifest.get("codebook_content_sha256"),
        where=f"{path} codebook_content_sha256",
    )
    normalized_digests: dict[str, str] = {}
    for raw_name, raw_digest in digests.items():
        name = str(raw_name)
        digest = str(raw_digest)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{path}: malformed SHA-256 for {name!r}")
        normalized_digests[name] = digest
    if set(tensors) != set(normalized_digests):
        raise ValueError(
            f"{path}: bundle digest map does not cover sidecar exactly: "
            f"missing={sorted(set(normalized_digests) - set(tensors))[:8]}, "
            f"unbound={sorted(set(tensors) - set(normalized_digests))[:8]}"
        )
    shapes = _require_mapping(
        manifest.get("tensor_shapes"), where=f"{path} tensor_shapes"
    )
    if set(map(str, shapes)) != set(tensors):
        raise ValueError(f"{path}: tensor_shapes does not cover sidecar exactly")
    for name, tensor in tensors.items():
        if tensor.dtype != torch.float16 or tensor.ndim != 2:
            raise ValueError(
                f"{path}: {name} must be rank-2 float16, got "
                f"{tuple(tensor.shape)} {tensor.dtype}"
            )
        expected_shape = tuple(int(dim) for dim in shapes[name])
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{path}: {name} shape {tuple(tensor.shape)} != {expected_shape}"
            )
        actual_digest = codebook_table_sha256(tensor)
        if actual_digest != normalized_digests[name]:
            raise ValueError(
                f"{path}: codebook digest mismatch for {name}: expected "
                f"{normalized_digests[name]}, got {actual_digest}"
            )
    expected_bundle_digest = _bundle_content_sha256(normalized_digests)
    if manifest.get("bundle_content_sha256") != expected_bundle_digest:
        raise ValueError(f"{path}: bundle content identity differs from table map")

    inputs = _require_mapping(manifest.get("inputs"), where=f"{path} inputs")
    cells = _require_mapping(manifest.get("cells"), where=f"{path} cells")
    if set(map(str, inputs)) != set(map(str, cells)):
        raise ValueError(f"{path}: inputs and cells qname coverage differs")

    def validate_identity_record(raw, *, where):
        identity = _require_mapping(raw, where=where)
        shape = identity.get("shape")
        digest = str(identity.get("sha256", ""))
        if (
            set(identity) != {"shape", "sha256"}
            or not isinstance(shape, list)
            or not shape
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                for dim in shape
            )
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(f"{where}: malformed tensor identity")

    raw_aliases = manifest.get("aliases", {})
    aliases = _require_mapping(raw_aliases, where=f"{path} aliases")
    for raw_alias, raw_entry in aliases.items():
        alias = str(raw_alias)
        if not alias or alias in cells:
            raise ValueError(f"{path}: invalid or colliding bundle alias {alias!r}")
        entry = _require_mapping(
            raw_entry, where=f"{path} aliases[{alias!r}]"
        )
        if set(entry) != {"cell_qname", "source_weight", "col_weights"}:
            raise ValueError(f"{path}: alias members differ for {alias!r}")
        cell_qname = str(entry.get("cell_qname", ""))
        if cell_qname not in cells:
            raise ValueError(
                f"{path}: alias {alias!r} names missing cell {cell_qname!r}"
            )
        for identity_name in ("source_weight", "col_weights"):
            validate_identity_record(
                entry.get(identity_name),
                where=f"{path} aliases[{alias!r}].{identity_name}",
            )
    referenced: set[str] = set()
    learned_owners: dict[str, tuple[str, str]] = {}
    for qname, raw_formats in cells.items():
        formats = _require_mapping(raw_formats, where=f"{path} cells[{qname!r}]")
        input_identity = _require_mapping(
            inputs[qname], where=f"{path} inputs[{qname!r}]"
        )
        for identity_name in ("source_weight", "col_weights"):
            validate_identity_record(
                input_identity.get(identity_name),
                where=f"{path} inputs[{qname!r}].{identity_name}",
            )
        for raw_fmt, raw_cell in formats.items():
            canonical, family, rung = _canonical_format(str(raw_fmt))
            if canonical != raw_fmt:
                raise ValueError(f"{path}: non-canonical format key {raw_fmt!r}")
            cell = _require_mapping(
                raw_cell, where=f"{path} cells[{qname!r}][{canonical!r}]"
            )
            source = str(cell.get("source", ""))
            if source not in {"lattice", "learned"}:
                raise ValueError(f"{path}: invalid source for {qname}/{canonical}")
            expected_cell_members = {
                "source", "codebook_ref", "content_sha256"
            } | ({"rung_policy"} if source == "learned" else set())
            if is_v2 and source == "learned":
                expected_cell_members.add("training_provenance")
            if "routed_book_keying" in cell:
                # Absent means a pre-R1 bundle, whose routed books ARE per
                # role; present, it must name a keying this producer knows.
                if source != "learned":
                    raise ValueError(
                        f"{path}: lattice cell records a routed book keying "
                        f"for {qname}/{canonical}"
                    )
                if cell.get("routed_book_keying") not in ROUTED_BOOK_KEYINGS:
                    raise ValueError(
                        f"{path}: invalid routed book keying "
                        f"{cell.get('routed_book_keying')!r} for "
                        f"{qname}/{canonical}"
                    )
                expected_cell_members.add("routed_book_keying")
            if "pretrained_origin" in cell:
                if source != "learned":
                    raise ValueError(
                        f"{path}: lattice cell has pretrained origin for "
                        f"{qname}/{canonical}"
                    )
                expected_cell_members.add("pretrained_origin")
            if set(cell) != expected_cell_members:
                raise ValueError(
                    f"{path}: cell members differ for {qname}/{canonical}"
                )
            if is_v2 and family.grid == "fp8":
                if rung not in CBL_STEP4_RUNGS:
                    raise ValueError(
                        f"{path}: learned-v2 has off-ladder cell "
                        f"{qname}/{canonical}"
                    )
                receipt_source = (
                    "learned" if load_rung_policy[rung]["enabled"] else "lattice"
                )
                if source != receipt_source:
                    raise ValueError(
                        f"{path}: learned-v2 source for {qname}/{canonical} "
                        f"is {source}, receipt policy requires {receipt_source}"
                    )
            if source == "learned":
                if family.grid != "fp8" or family.mode != "product":
                    raise ValueError(f"{path}: learned non-FP8 cell {qname}/{canonical}")
                if is_v2:
                    if load_rung_policy[rung]["enabled"] is not True:
                        raise ValueError(
                            f"{path}: learned-v2 rung K{rung} is not promoted"
                        )
                    _validate_v2_training_provenance(
                        cell.get("training_provenance"),
                        qname=str(qname),
                        rung=rung,
                        where=(
                            f"{path} cells[{qname!r}]"
                            f"[{canonical!r}].training_provenance"
                        ),
                    )
                else:
                    require_cbl_rung_enabled(rung)
                if cell.get("rung_policy") != load_rung_policy[rung]:
                    raise ValueError(
                        f"{path}: learned rung policy differs for {qname}/{canonical}"
                    )
                if "pretrained_origin" in cell:
                    _normalized_pretrained_origin(
                        cell["pretrained_origin"],
                        where=(
                            f"{path} cells[{qname!r}]"
                            f"[{canonical!r}].pretrained_origin"
                        ),
                    )
            refs = cell.get("codebook_ref")
            cell_digests = cell.get("content_sha256")
            if not isinstance(refs, list) or not isinstance(cell_digests, list):
                raise ValueError(f"{path}: malformed refs/digests for {qname}/{canonical}")
            origin = cell.get("pretrained_origin")
            if origin is not None:
                from prismaquant.cb_banked_books import (
                    BANKED_CBL_ORIGIN_SCHEMA,
                )

                if origin.get("schema") == BANKED_CBL_ORIGIN_SCHEMA:
                    layer_match = re.search(
                        r"(?:^|[.])layers[.]([0-9]+)(?:[.]|$)",
                        str(qname),
                    )
                    coordinates = (
                        None if layer_match is None else int(layer_match.group(1)),
                        str(qname).rsplit(".", 1)[-1],
                        rung,
                    )
                    if coordinates != (
                        origin["layer"],
                        origin["projection"],
                        origin["rung"],
                    ):
                        raise ValueError(
                            f"{path}: bank origin coordinates differ for "
                            f"{qname}/{canonical}"
                        )
                    if (
                        origin["source_digest"]
                        != input_identity["source_weight"]["sha256"]
                        or origin["col_weights_digest"]
                        != input_identity["col_weights"]["sha256"]
                    ):
                        raise ValueError(
                            f"{path}: bank origin input identity differs for "
                            f"{qname}/{canonical}"
                        )
                    if origin["subtable_content_sha256"] != cell_digests:
                        raise ValueError(
                            f"{path}: bank origin subtable digests differ for "
                            f"{qname}/{canonical}"
                        )
            expected_refs = canonical_codebook_refs(
                str(qname), canonical, source=source
            )
            if tuple(map(str, refs)) != expected_refs:
                raise ValueError(
                    f"{path}: physical refs differ for {qname}/{canonical}"
                )
            if len(cell_digests) != len(expected_refs):
                raise ValueError(f"{path}: digest count differs for {qname}/{canonical}")
            if is_v2 and source == "learned":
                if receipt is None:
                    raise ValueError(
                        f"{path}: learned-v2 cell lacks a validated promotion "
                        "receipt"
                    )
                candidate_digests = receipt.candidate_digests(
                    str(qname), rung
                )
                if tuple(map(str, cell_digests)) != candidate_digests:
                    raise ValueError(
                        f"{path}: learned table digests for {qname}/{canonical} "
                        "differ from the exact promotion candidate"
                    )
            expected_shapes = codebook_subtable_shapes(
                rung, family.mode, family.n_sub
            )
            canonical_lattice = (
                _lattice_codebook(canonical) if source == "lattice" else None
            )
            for table_index, (ref, digest, expected_shape) in enumerate(zip(
                expected_refs, cell_digests, expected_shapes, strict=True
            )):
                if ref not in tensors or tuple(tensors[ref].shape) != expected_shape:
                    raise ValueError(
                        f"{path}: missing or malformed physical table {ref!r}"
                    )
                # Every surviving family is an unsigned product family, so the
                # grid is always the full signed value grid. The signed
                # (magnitude-only, positive=True) family was deleted 2026-08-17.
                snapped = cb._snap_to_grid(
                    tensors[ref].to(torch.float32),
                    family.grid,
                    positive=False,
                )
                if not torch.equal(tensors[ref].to(torch.float32), snapped):
                    raise ValueError(
                        f"{path}: physical table {ref!r} contains a value "
                        f"outside the declared {family.grid} grid"
                    )
                if canonical_lattice is not None and not torch.equal(
                    tensors[ref], canonical_lattice[table_index]
                ):
                    raise ValueError(
                        f"{path}: physical table {ref!r} differs from the "
                        f"canonical {canonical} lattice bytes"
                    )
                if str(digest) != normalized_digests[ref]:
                    raise ValueError(
                        f"{path}: cell digest differs for physical table {ref!r}"
                    )
                if source == "learned":
                    owner = learned_owners.setdefault(ref, (str(qname), canonical))
                    if owner != (str(qname), canonical):
                        raise ValueError(
                            f"{path}: learned cells {owner} and "
                            f"{(str(qname), canonical)} share ref {ref!r}"
                        )
                referenced.add(ref)
    if referenced != set(tensors):
        raise ValueError(
            f"{path}: bundle cells do not cover sidecar exactly: "
            f"unreferenced={sorted(set(tensors) - referenced)[:8]}"
        )
    return CBLearnedBundle(path=path, manifest=dict(manifest), sidecar_tensors=tensors)


@lru_cache(maxsize=8)
def _load_bundle_cached_snapshot(
    resolved_path: str,
    size: int,
    mtime_ns: int,
) -> CBLearnedBundle:
    # size/mtime_ns are deliberate cache-key inputs.  The bundle is immutable,
    # but a same-path replacement by an operator must never leave a long-lived
    # cost process using stale values.
    del size, mtime_ns
    return load_bundle(resolved_path)


def load_bundle_cached(path: str | Path) -> CBLearnedBundle:
    """Cache a verified bundle by resolved path and current file identity."""

    resolved = Path(path).resolve(strict=True)
    for _attempt in range(2):
        before = resolved.stat()
        bundle = _load_bundle_cached_snapshot(
            str(resolved), int(before.st_size), int(before.st_mtime_ns)
        )
        after = resolved.stat()
        if (
            int(before.st_size), int(before.st_mtime_ns)
        ) == (
            int(after.st_size), int(after.st_mtime_ns)
        ):
            return bundle
    raise RuntimeError(f"learned bundle {resolved} changed repeatedly while loading")


__all__ = [
    "CBL_RUNG_POLICY",
    "CB_LEARNED_BUNDLE_SCHEMA",
    "CB_LEARNED_TRAINER_SCHEMA",
    "CB_LEARNED_TRAINER_STAMP",
    "CB_LEARNED_TRAINER_V2_SCHEMA",
    "CB_LEARNED_TRAINER_V2_STAMP",
    "CB_LEARNED_V2_SAMPLING_SCHEMA",
    "CBLearnedBundle",
    "GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION",
    "LLOYD_CAP",
    "LLOYD_ITERS",
    "LLOYD_ROW_SAMPLE",
    "LLOYD_ROW_SEED",
    "PretrainedCodebookCell",
    "LearnedV2PoolResult",
    "LLOYD_SEED",
    "canonical_codebook_refs",
    "canonical_fp16_table",
    "codebook_table_sha256",
    "learn_pool",
    "learn_pool_v2",
    "learned_v2_sampling_plan",
    "load_bundle",
    "load_bundle_cached",
    "refuse_routed_moe_learned",
    "require_cbl_rung_enabled",
    "train_and_save_bundle",
    "train_and_save_bundle_streaming",
]
