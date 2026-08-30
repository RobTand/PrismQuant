"""Result-gated learned/lattice policy for the step-4 FP8 Gridbook ladder.

This receipt is deliberately model- and calibration-specific.  It does not
encode a maximum learned rung or assume that source decisions are monotone in
K: every rung independently earns learned placement against lattice on two
held-out calibration sets.  A failed or tied rung remains lattice.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from .cb_imatrix import CB_IMATRIX_FROM_PROBE_SCHEMA
from .cb_layout import FP8_PRODUCT_RUNGS, codebook_subtable_shapes, family_for


CBL_PROMOTION_RECEIPT_SCHEMA = "prismaquant.fp8_cbl_promotion_receipt.v1"
CBL_V2_TRAINER_SCHEMA = "prismaquant.fp8_cbl_poolb.v2"
CBL_STEP4_RUNGS = FP8_PRODUCT_RUNGS
CBL_PROMOTION_THRESHOLDS: dict[str, float] = {
    "max_geomean_ratio": 0.98,
    "max_bootstrap_95_upper": 1.0,
    "max_role_aggregate_ratio": 1.01,
    "max_p95_unit_ratio": 1.05,
    "max_worst_unit_ratio": 1.10,
    "max_repeat_delta_percentage_points": 2.0,
}


class CBLPromotionReceiptError(ValueError):
    """A promotion receipt is incomplete, contradictory, or out of policy."""


@dataclass(frozen=True)
class ValidatedCBLPromotionReceipt:
    payload: Mapping[str, object]
    receipt_sha256: str
    source_by_rung: Mapping[int, str]
    candidate_digests_by_cell: Mapping[tuple[str, int], tuple[str, ...]]

    @property
    def learned_rungs(self) -> tuple[int, ...]:
        return tuple(
            rung for rung in CBL_STEP4_RUNGS
            if self.source_by_rung[rung] == "learned"
        )

    def source_for_rung(self, rung: int) -> str:
        try:
            return str(self.source_by_rung[int(rung)])
        except (KeyError, ValueError) as exc:
            raise CBLPromotionReceiptError(
                f"K{rung} is outside the K4..K48 step-4 promotion ladder"
            ) from exc

    def candidate_digests(
        self, qname: str, rung: int
    ) -> tuple[str, ...]:
        key = (str(qname), int(rung))
        try:
            return tuple(self.candidate_digests_by_cell[key])
        except KeyError as exc:
            raise CBLPromotionReceiptError(
                f"promotion receipt has no learned candidate digests for "
                f"{key[0]}/K{key[1]}"
            ) from exc


def _canonical_json(value: object, *, where: str) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CBLPromotionReceiptError(
            f"{where} is not strict canonical JSON data"
        ) from exc


def _mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CBLPromotionReceiptError(f"{where} must be an object")
    return value


def _exact_members(
    value: Mapping[str, object], expected: set[str], *, where: str
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise CBLPromotionReceiptError(f"{where} member names must be strings")
    actual = set(value)
    if actual != expected:
        raise CBLPromotionReceiptError(
            f"{where} members differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _nonempty_string(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CBLPromotionReceiptError(f"{where} must be a nonempty string")
    return value.strip()


def _sha256(value: object, *, where: str) -> str:
    text = _nonempty_string(value, where=where)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise CBLPromotionReceiptError(f"{where} must be a lowercase SHA-256")
    return text


def _metric(value: object, *, where: str) -> float:
    if isinstance(value, bool):
        raise CBLPromotionReceiptError(f"{where} must be a finite nonnegative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CBLPromotionReceiptError(
            f"{where} must be a finite nonnegative number"
        ) from exc
    if not math.isfinite(result) or result < 0:
        raise CBLPromotionReceiptError(
            f"{where} must be a finite nonnegative number"
        )
    return result


def role_census_for_qnames(qnames: Iterable[str]) -> dict[str, int]:
    """Count exact candidate cells by qname leaf role."""

    result: dict[str, int] = {}
    seen: set[str] = set()
    for raw_qname in qnames:
        qname = _nonempty_string(raw_qname, where="candidate qname")
        if qname in seen:
            raise CBLPromotionReceiptError(
                f"candidate qname {qname!r} is repeated"
            )
        seen.add(qname)
        role = qname.rsplit(".", 1)[-1]
        if not role:
            raise CBLPromotionReceiptError(
                f"candidate qname {qname!r} has no leaf role"
            )
        result[role] = result.get(role, 0) + 1
    if not result:
        raise CBLPromotionReceiptError("candidate qname census is empty")
    return dict(sorted(result.items()))


def _role_census(value: object, *, where: str) -> dict[str, int]:
    raw = _mapping(value, where=where)
    if not raw:
        raise CBLPromotionReceiptError(f"{where} must be nonempty")
    result: dict[str, int] = {}
    for raw_role, raw_count in raw.items():
        role = _nonempty_string(raw_role, where=f"{where} role")
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count <= 0
        ):
            raise CBLPromotionReceiptError(
                f"{where}[{role!r}] must be a positive integer"
            )
        result[role] = raw_count
    return dict(sorted(result.items()))


def _holdout_passes(
    metrics: Mapping[str, object],
    *,
    expected_role_census: Mapping[str, int],
    where: str,
) -> bool:
    _exact_members(
        metrics,
        {
            "geomean_ratio",
            "bootstrap_95_upper",
            "role_aggregate_ratios",
            "role_coverage_counts",
            "p95_unit_ratio",
            "worst_unit_ratio",
        },
        where=where,
    )
    roles = _mapping(
        metrics.get("role_aggregate_ratios"),
        where=f"{where}.role_aggregate_ratios",
    )
    if set(roles) != set(expected_role_census):
        raise CBLPromotionReceiptError(
            f"{where}.role_aggregate_ratios role coverage differs from the "
            "model-derived census"
        )
    role_values = []
    for raw_role, raw_ratio in roles.items():
        role = _nonempty_string(raw_role, where=f"{where} role")
        role_values.append(
            _metric(raw_ratio, where=f"{where}.role_aggregate_ratios[{role!r}]")
        )
    coverage = _role_census(
        metrics.get("role_coverage_counts"),
        where=f"{where}.role_coverage_counts",
    )
    if coverage != dict(expected_role_census):
        raise CBLPromotionReceiptError(
            f"{where}.role_coverage_counts differs from the model-derived "
            "census"
        )
    geomean = _metric(metrics.get("geomean_ratio"), where=f"{where}.geomean_ratio")
    upper = _metric(
        metrics.get("bootstrap_95_upper"),
        where=f"{where}.bootstrap_95_upper",
    )
    p95 = _metric(metrics.get("p95_unit_ratio"), where=f"{where}.p95_unit_ratio")
    worst = _metric(
        metrics.get("worst_unit_ratio"), where=f"{where}.worst_unit_ratio"
    )
    return (
        geomean <= CBL_PROMOTION_THRESHOLDS["max_geomean_ratio"]
        and upper < CBL_PROMOTION_THRESHOLDS["max_bootstrap_95_upper"]
        and max(role_values)
        <= CBL_PROMOTION_THRESHOLDS["max_role_aggregate_ratio"]
        and p95 <= CBL_PROMOTION_THRESHOLDS["max_p95_unit_ratio"]
        and worst <= CBL_PROMOTION_THRESHOLDS["max_worst_unit_ratio"]
    )


def validate_promotion_receipt(
    payload: Mapping[str, object],
    *,
    expected_model_id: str | None = None,
    expected_model_content_sha256: str | None = None,
    expected_calibration_hash: str | None = None,
    expected_imatrix_sha256: str | None = None,
    expected_role_census: Mapping[str, int] | None = None,
    expected_qnames: Iterable[str] | None = None,
) -> ValidatedCBLPromotionReceipt:
    """Validate a complete two-holdout receipt and derive every rung source."""

    missing_bindings = [
        name
        for name, value in (
            ("expected_model_id", expected_model_id),
            (
                "expected_model_content_sha256",
                expected_model_content_sha256,
            ),
            ("expected_calibration_hash", expected_calibration_hash),
            ("expected_imatrix_sha256", expected_imatrix_sha256),
            ("expected_role_census", expected_role_census),
            ("expected_qnames", expected_qnames),
        )
        if value is None
    ]
    if missing_bindings:
        raise CBLPromotionReceiptError(
            "promotion receipt validation requires actual external bindings: "
            f"{missing_bindings}"
        )
    bound_model_id = _nonempty_string(
        expected_model_id, where="expected model id"
    )
    bound_model_sha = _sha256(
        expected_model_content_sha256,
        where="expected model content_sha256",
    )
    bound_calibration_hash = _nonempty_string(
        expected_calibration_hash,
        where="expected probe calibration hash",
    )
    bound_imatrix_sha = _sha256(
        expected_imatrix_sha256,
        where="expected imatrix value_sha256",
    )
    if isinstance(expected_qnames, (str, bytes)):
        raise CBLPromotionReceiptError(
            "expected_qnames must be an iterable of qname strings"
        )
    bound_qnames = tuple(sorted(str(name) for name in expected_qnames))
    derived_role_census = role_census_for_qnames(bound_qnames)
    bound_role_census = _role_census(
        expected_role_census, where="expected model role census"
    )
    if bound_role_census != derived_role_census:
        raise CBLPromotionReceiptError(
            "expected role census differs from the model-derived qname census"
        )

    root = _mapping(payload, where="promotion receipt")
    _exact_members(
        root,
        {
            "schema",
            "model",
            "trainer",
            "imatrix",
            "holdouts",
            "thresholds",
            "rungs",
            "candidate_codebooks",
        },
        where="promotion receipt",
    )
    if root.get("schema") != CBL_PROMOTION_RECEIPT_SCHEMA:
        raise CBLPromotionReceiptError(
            f"unsupported promotion receipt schema {root.get('schema')!r}"
        )

    model = _mapping(root.get("model"), where="promotion receipt.model")
    _exact_members(
        model,
        {"model_id", "content_sha256", "role_census"},
        where="promotion receipt.model",
    )
    receipt_model_id = _nonempty_string(
        model.get("model_id"), where="promotion receipt.model.model_id"
    )
    receipt_model_sha = _sha256(
        model.get("content_sha256"),
        where="promotion receipt.model.content_sha256",
    )
    receipt_role_census = _role_census(
        model.get("role_census"),
        where="promotion receipt.model.role_census",
    )
    if (
        receipt_model_id != bound_model_id
        or receipt_model_sha != bound_model_sha
        or receipt_role_census != bound_role_census
    ):
        raise CBLPromotionReceiptError(
            "promotion receipt model binding differs from the validated "
            "complete source-model identity or role census"
        )

    trainer = _mapping(root.get("trainer"), where="promotion receipt.trainer")
    _exact_members(trainer, {"schema"}, where="promotion receipt.trainer")
    if trainer.get("schema") != CBL_V2_TRAINER_SCHEMA:
        raise CBLPromotionReceiptError(
            "promotion receipt must name the learned-v2 trainer schema"
        )

    imatrix = _mapping(root.get("imatrix"), where="promotion receipt.imatrix")
    _exact_members(
        imatrix,
        {"schema", "calibration_hash", "value_sha256"},
        where="promotion receipt.imatrix",
    )
    if imatrix.get("schema") != CB_IMATRIX_FROM_PROBE_SCHEMA:
        raise CBLPromotionReceiptError(
            "promotion receipt imatrix must come from full-probe act_sq_sum"
        )
    training_calibration_hash = _nonempty_string(
        imatrix.get("calibration_hash"),
        where="promotion receipt.imatrix.calibration_hash",
    )
    imatrix_sha = _sha256(
        imatrix.get("value_sha256"),
        where="promotion receipt.imatrix.value_sha256",
    )
    if training_calibration_hash != bound_calibration_hash:
        raise CBLPromotionReceiptError(
            "promotion receipt calibration identity differs from the actual "
            "probe calibration hash"
        )
    if imatrix_sha != bound_imatrix_sha:
        raise CBLPromotionReceiptError(
            "promotion receipt imatrix value identity differs from the "
            "bundle col_weights"
        )

    thresholds = _mapping(
        root.get("thresholds"), where="promotion receipt.thresholds"
    )
    _exact_members(
        thresholds,
        set(CBL_PROMOTION_THRESHOLDS),
        where="promotion receipt.thresholds",
    )
    observed_thresholds = {
        name: _metric(thresholds[name], where=f"promotion receipt.thresholds.{name}")
        for name in CBL_PROMOTION_THRESHOLDS
    }
    if observed_thresholds != CBL_PROMOTION_THRESHOLDS:
        raise CBLPromotionReceiptError(
            "promotion receipt thresholds differ from the two-holdout policy"
        )

    raw_holdouts = root.get("holdouts")
    if not isinstance(raw_holdouts, list) or len(raw_holdouts) != 2:
        raise CBLPromotionReceiptError(
            "promotion receipt must declare exactly two holdouts"
        )
    holdout_ids: list[str] = []
    holdout_hashes: list[str] = []
    for index, raw in enumerate(raw_holdouts):
        holdout = _mapping(raw, where=f"promotion receipt.holdouts[{index}]")
        _exact_members(
            holdout,
            {"id", "calibration_hash"},
            where=f"promotion receipt.holdouts[{index}]",
        )
        holdout_ids.append(
            _nonempty_string(
                holdout.get("id"),
                where=f"promotion receipt.holdouts[{index}].id",
            )
        )
        holdout_hashes.append(
            _sha256(
                holdout.get("calibration_hash"),
                where=f"promotion receipt.holdouts[{index}].calibration_hash",
            )
        )
    if len(set(holdout_ids)) != 2 or len(set(holdout_hashes)) != 2:
        raise CBLPromotionReceiptError(
            "promotion holdout ids and calibration hashes must be distinct"
        )
    if training_calibration_hash in holdout_hashes:
        raise CBLPromotionReceiptError(
            "promotion holdout calibration overlaps the training calibration"
        )

    raw_candidates = _mapping(
        root.get("candidate_codebooks"),
        where="promotion receipt.candidate_codebooks",
    )
    _exact_members(
        raw_candidates,
        set(bound_qnames),
        where="promotion receipt.candidate_codebooks",
    )
    product_family = family_for("fp8", "product")
    expected_digest_count = {
        rung: len(
            codebook_subtable_shapes(
                rung,
                product_family.mode,
                product_family.n_sub,
            )
        )
        for rung in CBL_STEP4_RUNGS
    }
    candidate_digests_by_cell: dict[
        tuple[str, int], tuple[str, ...]
    ] = {}
    expected_rung_keys = {str(rung) for rung in CBL_STEP4_RUNGS}
    for qname in bound_qnames:
        qname_candidates = _mapping(
            raw_candidates[qname],
            where=f"promotion receipt.candidate_codebooks[{qname!r}]",
        )
        _exact_members(
            qname_candidates,
            expected_rung_keys,
            where=f"promotion receipt.candidate_codebooks[{qname!r}]",
        )
        for rung in CBL_STEP4_RUNGS:
            where = (
                f"promotion receipt.candidate_codebooks[{qname!r}]"
                f"[{rung}]"
            )
            candidate = _mapping(qname_candidates[str(rung)], where=where)
            _exact_members(
                candidate,
                {"subtable_content_sha256"},
                where=where,
            )
            raw_digests = candidate.get("subtable_content_sha256")
            if (
                not isinstance(raw_digests, list)
                or len(raw_digests) != expected_digest_count[rung]
            ):
                raise CBLPromotionReceiptError(
                    f"{where}.subtable_content_sha256 must contain exactly "
                    f"{expected_digest_count[rung]} table digests"
                )
            candidate_digests_by_cell[(qname, rung)] = tuple(
                _sha256(
                    raw_digest,
                    where=(
                        f"{where}.subtable_content_sha256[{index}]"
                    ),
                )
                for index, raw_digest in enumerate(raw_digests)
            )

    rungs = _mapping(root.get("rungs"), where="promotion receipt.rungs")
    _exact_members(rungs, expected_rung_keys, where="promotion receipt.rungs")
    source_by_rung: dict[int, str] = {}
    for rung in CBL_STEP4_RUNGS:
        where = f"promotion receipt.rungs[{rung}]"
        record = _mapping(rungs[str(rung)], where=where)
        _exact_members(
            record,
            {
                "declared_source",
                "repeat_delta_percentage_points",
                "density_shortfall_cells",
                "holdouts",
            },
            where=where,
        )
        repeat_delta = _metric(
            record.get("repeat_delta_percentage_points"),
            where=f"{where}.repeat_delta_percentage_points",
        )
        shortfall = record.get("density_shortfall_cells")
        if isinstance(shortfall, bool) or not isinstance(shortfall, int) or shortfall < 0:
            raise CBLPromotionReceiptError(
                f"{where}.density_shortfall_cells must be a nonnegative integer"
            )
        metrics_by_holdout = _mapping(record.get("holdouts"), where=f"{where}.holdouts")
        _exact_members(metrics_by_holdout, set(holdout_ids), where=f"{where}.holdouts")
        passes = shortfall == 0 and all(
            _holdout_passes(
                _mapping(
                    metrics_by_holdout[holdout_id],
                    where=f"{where}.holdouts[{holdout_id!r}]",
                ),
                expected_role_census=bound_role_census,
                where=f"{where}.holdouts[{holdout_id!r}]",
            )
            for holdout_id in holdout_ids
        ) and repeat_delta <= CBL_PROMOTION_THRESHOLDS[
            "max_repeat_delta_percentage_points"
        ]
        expected_source = "learned" if passes else "lattice"
        declared_source = _nonempty_string(
            record.get("declared_source"), where=f"{where}.declared_source"
        )
        if declared_source not in {"learned", "lattice"}:
            raise CBLPromotionReceiptError(
                f"{where}.declared_source must be learned or lattice"
            )
        if declared_source != expected_source:
            raise CBLPromotionReceiptError(
                f"{where} declares {declared_source} but its two-holdout "
                f"metrics require {expected_source}"
            )
        source_by_rung[rung] = expected_source

    canonical = _canonical_json(root, where="promotion receipt")
    return ValidatedCBLPromotionReceipt(
        payload=json.loads(canonical),
        receipt_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        source_by_rung=source_by_rung,
        candidate_digests_by_cell=candidate_digests_by_cell,
    )


def read_promotion_receipt_payload(path: str | Path) -> Mapping[str, object]:
    """Read strict JSON without pretending the external bindings are known."""

    receipt_path = Path(path)

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise CBLPromotionReceiptError(
                    f"promotion receipt has duplicate JSON member {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            receipt_path.read_text(),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CBLPromotionReceiptError(
            f"cannot read promotion receipt {receipt_path}: {exc}"
        ) from exc
    return _mapping(payload, where="promotion receipt")


def load_promotion_receipt(
    path: str | Path,
    *,
    expected_model_id: str | None = None,
    expected_model_content_sha256: str | None = None,
    expected_calibration_hash: str | None = None,
    expected_imatrix_sha256: str | None = None,
    expected_role_census: Mapping[str, int] | None = None,
    expected_qnames: Iterable[str] | None = None,
) -> ValidatedCBLPromotionReceipt:
    """Read and validate a receipt against independently derived identities."""

    return validate_promotion_receipt(
        read_promotion_receipt_payload(path),
        expected_model_id=expected_model_id,
        expected_model_content_sha256=expected_model_content_sha256,
        expected_calibration_hash=expected_calibration_hash,
        expected_imatrix_sha256=expected_imatrix_sha256,
        expected_role_census=expected_role_census,
        expected_qnames=expected_qnames,
    )


def receipt_rung_policy(
    receipt: ValidatedCBLPromotionReceipt,
) -> dict[int, dict[str, object]]:
    """Compact bundle policy rows derived independently for every rung."""

    return {
        rung: {
            "enabled": receipt.source_by_rung[rung] == "learned",
            "status": (
                "two_holdout_promoted"
                if receipt.source_by_rung[rung] == "learned"
                else "two_holdout_lattice"
            ),
            "provenance": CBL_PROMOTION_RECEIPT_SCHEMA,
            "receipt_sha256": receipt.receipt_sha256,
            "density_shortfall_cells": int(
                receipt.payload["rungs"][str(rung)][
                    "density_shortfall_cells"
                ]
            ),
        }
        for rung in CBL_STEP4_RUNGS
    }


__all__ = [
    "CBL_PROMOTION_RECEIPT_SCHEMA",
    "CBL_PROMOTION_THRESHOLDS",
    "CBL_STEP4_RUNGS",
    "CBL_V2_TRAINER_SCHEMA",
    "CBLPromotionReceiptError",
    "ValidatedCBLPromotionReceipt",
    "load_promotion_receipt",
    "read_promotion_receipt_payload",
    "receipt_rung_policy",
    "role_census_for_qnames",
    "validate_promotion_receipt",
]
