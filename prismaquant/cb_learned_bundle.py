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


CB_LEARNED_BUNDLE_SCHEMA = "prismaquant.cb_learned_codebook_bundle.v1"
CB_LEARNED_BUNDLE_METADATA_KEY = "prismaquant_cb_learned_bundle"
CB_LEARNED_TRAINER_SCHEMA = "prismaquant.fp8_cbl_poolb.v1"

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


def _canonical_format(format_name: str) -> tuple[str, object, int]:
    parsed = parse_format_name(str(format_name).strip().upper())
    if parsed is None:
        raise ValueError(f"{format_name!r} is not a producer CB format")
    family, rung = parsed
    return family.name(rung), family, int(rung)


def require_cbl_rung_enabled(rung_or_format: int | str) -> int:
    """Return the FP8-CB rung only when its measured policy enables CBL."""

    if isinstance(rung_or_format, str) and not str(rung_or_format).isdigit():
        canonical, family, rung = _canonical_format(rung_or_format)
        if (
            family.grid != "fp8"
            or family.mode != "product"
            or family.source != "learned"
        ):
            raise ValueError(
                f"{canonical}: learned production bundles support FP8 product "
                "codebooks with an explicit learned format name only"
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
    canonical, family, _ = _canonical_format(f"FP8_CBL_K{rung}")
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
        canonical = tuple(sorted({_canonical_format(name)[0] for name in names}))
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
        """Return the name-authoritative per-format source map.

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
                canonical, family, _rung = _canonical_format(raw_format)
                cell_source = str(raw_cell["source"])
                source = (
                    str(family.source)
                    if family.source is not None
                    else cell_source
                )
                if family.source is not None and cell_source != source:
                    raise ValueError(
                        f"{self.path}: bundle cell source {cell_source!r} "
                        f"contradicts format-name source {source!r} for "
                        f"{qname}/{canonical}"
                    )
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
    routed_moe_qnames: Iterable[str] = (),
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

    By default every policy-enabled supplied FP8-CB format is learned and all
    other supplied formats are canonical lattice.  ``learned_formats`` can
    explicitly narrow the learned FP8 set (an empty iterable produces an
    all-lattice bundle); an explicit disabled rung is still refused.  The
    formats may be common to all qnames or supplied per qname.  For large
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
    normalized_col = {str(name): torch.as_tensor(value) for name, value in col_weights.items()}
    formats_by_qname = _normalize_formats_by_qname(target_names, formats)
    target_qnames = set(formats_by_qname)
    missing_col = sorted(target_qnames - set(normalized_col))
    if missing_col:
        raise ValueError(f"bundle cells are missing col_weights: {missing_col[:8]}")
    routed = {str(name) for name in routed_moe_qnames}
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
    requested_learned = (
        None
        if learned_formats is None
        else {_canonical_format(fmt)[0] for fmt in learned_formats}
    )
    unknown_learned = sorted((requested_learned or set()) - supplied_formats)
    if unknown_learned:
        raise ValueError(
            f"learned_formats are absent from bundle cells: {unknown_learned}"
        )
    learned: set[str] = set()
    source_conflicts: list[str] = []
    for fmt in sorted(supplied_formats):
        _canonical, family, _rung = _canonical_format(fmt)
        if family.source is None:
            if requested_learned is not None and fmt in requested_learned:
                learned.add(fmt)
            continue
        name_is_learned = family.source == "learned"
        requested_is_learned = (
            name_is_learned
            if requested_learned is None
            else fmt in requested_learned
        )
        if requested_is_learned != name_is_learned:
            source_conflicts.append(fmt)
        if name_is_learned:
            learned.add(fmt)
    if source_conflicts:
        raise ValueError(
            "learned_formats contradict source-bearing format name(s): "
            f"{source_conflicts[:8]}"
        )
    for fmt in sorted(learned):
        _canonical, family, rung = _canonical_format(fmt)
        if (
            family.grid != "fp8"
            or family.mode != "product"
            or family.source != "learned"
        ):
            raise ValueError(
                f"{fmt}: production learned bundle refuses NVFP4 CBL; it is "
                "measured NO-GO"
            )
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
            source = (
                str(family.source)
                if family.source is not None
                else ("learned" if canonical in learned else "lattice")
            )
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
                if supplied is not None:
                    tables = _codebook_sequence(canonical, supplied)
                elif weight.ndim == 3:
                    raise ValueError(
                        f"{qname}/{canonical}: routed-MoE learned CBL requires "
                        "an immutable banked pretrained_codebooks cell; "
                        "retraining is forbidden"
                    )
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
            digests: list[str] = []
            for ref, table in zip(refs, tables, strict=True):
                digest = codebook_table_sha256(table)
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
                digests.append(digest)
            cells[qname][canonical] = {
                "source": source,
                "codebook_ref": list(refs),
                "content_sha256": digests,
                **({
                    "rung_policy": dict(CBL_RUNG_POLICY[rung]),
                } if source == "learned" else {}),
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
        "trainer": dict(CB_LEARNED_TRAINER_STAMP),
        "rung_policy": {
            str(rung): dict(policy)
            for rung, policy in sorted(CBL_RUNG_POLICY.items())
        },
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
    routed_moe_qnames: Iterable[str] = (),
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
        routed_moe_qnames=routed_moe_qnames,
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
    allowed_top_level = required_top_level | {"aliases"}
    if not required_top_level <= set(manifest) or not set(manifest) <= allowed_top_level:
        raise ValueError(
            f"{path}: learned bundle manifest members differ: "
            f"missing={sorted(required_top_level - set(manifest))}, "
            f"unknown={sorted(set(manifest) - allowed_top_level)}"
        )
    trainer = _require_mapping(manifest.get("trainer"), where=f"{path} trainer")
    if dict(trainer) != CB_LEARNED_TRAINER_STAMP:
        raise ValueError(f"{path}: learned bundle trainer identity differs")
    observed_policy = _require_mapping(
        manifest.get("rung_policy"), where=f"{path} rung_policy"
    )
    expected_policy = {
        str(rung): dict(policy)
        for rung, policy in sorted(CBL_RUNG_POLICY.items())
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
            if family.source is not None and source != family.source:
                raise ValueError(
                    f"{path}: source {source!r} contradicts source-bearing "
                    f"format name {canonical!r} ({family.source!r})"
                )
            expected_cell_members = {
                "source", "codebook_ref", "content_sha256"
            } | ({"rung_policy"} if source == "learned" else set())
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
            if source == "learned":
                if (
                    family.grid != "fp8"
                    or family.mode != "product"
                    or family.source != "learned"
                ):
                    raise ValueError(f"{path}: learned non-FP8 cell {qname}/{canonical}")
                require_cbl_rung_enabled(rung)
                if cell.get("rung_policy") != CBL_RUNG_POLICY[rung]:
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
                positive = family.mode == "signed"
                snapped = cb._snap_to_grid(
                    tensors[ref].to(torch.float32),
                    family.grid,
                    positive=positive,
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
    "CBLearnedBundle",
    "GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION",
    "LLOYD_CAP",
    "LLOYD_ITERS",
    "LLOYD_ROW_SAMPLE",
    "LLOYD_ROW_SEED",
    "PretrainedCodebookCell",
    "LLOYD_SEED",
    "canonical_codebook_refs",
    "canonical_fp16_table",
    "codebook_table_sha256",
    "learn_pool",
    "load_bundle",
    "load_bundle_cached",
    "refuse_routed_moe_learned",
    "require_cbl_rung_enabled",
    "train_and_save_bundle",
    "train_and_save_bundle_streaming",
]
