from __future__ import annotations

import copy
import hashlib

import pytest

from prismaquant.cb_imatrix import CB_IMATRIX_FROM_PROBE_SCHEMA
from prismaquant.cb_layout import FP8_PRODUCT_RUNGS
from prismaquant.cb_learned_promotion import (
    CBL_PROMOTION_RECEIPT_SCHEMA,
    CBL_PROMOTION_THRESHOLDS,
    CBL_STEP4_RUNGS,
    CBL_V2_TRAINER_SCHEMA,
    CBLPromotionReceiptError,
    read_promotion_receipt_payload,
    receipt_rung_policy,
    role_census_for_qnames,
    validate_promotion_receipt,
)


MODEL_ID = "fixture/model"
MODEL_SHA256 = "1" * 64
CALIBRATION_HASH = "training-calibration"
IMATRIX_SHA256 = "a" * 64
QNAMES = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.mlp.up_proj",
)
ROLE_CENSUS = role_census_for_qnames(QNAMES)


def _candidate_digest(qname: str, rung: int, index: int) -> str:
    return hashlib.sha256(
        f"{qname}\0K{rung}\0sub{index}".encode("utf-8")
    ).hexdigest()


def promotion_receipt(
    imatrix_sha256: str = IMATRIX_SHA256,
    *,
    failed_rungs: set[int] = frozenset(),
) -> dict[str, object]:
    holdout_ids = ("holdout-a", "holdout-b")
    rungs = {}
    for rung in CBL_STEP4_RUNGS:
        passes = rung not in failed_rungs
        density_shortfall_cells = int(rung == 48)
        promoted = passes and density_shortfall_cells == 0
        metrics = {
            "geomean_ratio": 0.95 if passes else 1.0,
            "bootstrap_95_upper": 0.99,
            "role_aggregate_ratios": {
                role: 0.99 for role in ROLE_CENSUS
            },
            "role_coverage_counts": dict(ROLE_CENSUS),
            "p95_unit_ratio": 1.02,
            "worst_unit_ratio": 1.08,
        }
        rungs[str(rung)] = {
            "declared_source": "learned" if promoted else "lattice",
            "repeat_delta_percentage_points": 0.25,
            "density_shortfall_cells": density_shortfall_cells,
            "holdouts": {
                holdout_id: copy.deepcopy(metrics)
                for holdout_id in holdout_ids
            },
        }
    return {
        "schema": CBL_PROMOTION_RECEIPT_SCHEMA,
        "model": {
            "model_id": MODEL_ID,
            "content_sha256": MODEL_SHA256,
            "role_census": dict(ROLE_CENSUS),
        },
        "trainer": {"schema": CBL_V2_TRAINER_SCHEMA},
        "imatrix": {
            "schema": CB_IMATRIX_FROM_PROBE_SCHEMA,
            "calibration_hash": CALIBRATION_HASH,
            "value_sha256": imatrix_sha256,
        },
        "holdouts": [
            {"id": "holdout-a", "calibration_hash": "b" * 64},
            {"id": "holdout-b", "calibration_hash": "c" * 64},
        ],
        "thresholds": dict(CBL_PROMOTION_THRESHOLDS),
        "rungs": rungs,
        "candidate_codebooks": {
            qname: {
                str(rung): {
                    "subtable_content_sha256": [
                        _candidate_digest(qname, rung, index)
                        for index in range(4)
                    ],
                }
                for rung in CBL_STEP4_RUNGS
            }
            for qname in QNAMES
        },
    }


def _validate(payload: dict[str, object], **overrides):
    bindings = {
        "expected_model_id": MODEL_ID,
        "expected_model_content_sha256": MODEL_SHA256,
        "expected_calibration_hash": CALIBRATION_HASH,
        "expected_imatrix_sha256": IMATRIX_SHA256,
        "expected_role_census": ROLE_CENSUS,
        "expected_qnames": QNAMES,
    }
    bindings.update(overrides)
    return validate_promotion_receipt(payload, **bindings)


def test_receipt_promotes_every_rung_independently_without_a_crossover():
    assert CBL_STEP4_RUNGS == FP8_PRODUCT_RUNGS
    payload = promotion_receipt(failed_rungs={8, 44})
    validated = _validate(payload)
    assert validated.source_for_rung(4) == "learned"
    assert validated.source_for_rung(8) == "lattice"
    assert validated.source_for_rung(12) == "learned"
    assert validated.source_for_rung(44) == "lattice"
    assert validated.source_for_rung(48) == "lattice"
    assert validated.candidate_digests(QNAMES[0], 4) == tuple(
        payload["candidate_codebooks"][QNAMES[0]]["4"][
            "subtable_content_sha256"
        ]
    )
    policy = receipt_rung_policy(validated)
    assert policy[8]["enabled"] is False
    assert policy[12]["enabled"] is True
    assert policy[48]["density_shortfall_cells"] == 1
    assert {row["receipt_sha256"] for row in policy.values()} == {
        validated.receipt_sha256
    }


def test_tie_or_one_holdout_failure_stays_lattice():
    payload = promotion_receipt()
    rung = payload["rungs"]["28"]
    rung["holdouts"]["holdout-b"]["bootstrap_95_upper"] = 1.0
    rung["declared_source"] = "lattice"
    validated = _validate(payload)
    assert validated.source_for_rung(28) == "lattice"


def test_receipt_refuses_declared_source_that_metrics_do_not_earn():
    payload = promotion_receipt(failed_rungs={20})
    payload["rungs"]["20"]["declared_source"] = "learned"
    with pytest.raises(CBLPromotionReceiptError, match="metrics require lattice"):
        _validate(payload)


def test_receipt_requires_external_model_probe_and_imatrix_bindings():
    payload = promotion_receipt()
    with pytest.raises(CBLPromotionReceiptError, match="external bindings"):
        validate_promotion_receipt(payload)
    with pytest.raises(CBLPromotionReceiptError, match="model binding differs"):
        _validate(payload, expected_model_id="other/model")
    with pytest.raises(CBLPromotionReceiptError, match="model binding differs"):
        _validate(payload, expected_model_content_sha256="f" * 64)
    with pytest.raises(CBLPromotionReceiptError, match="calibration identity differs"):
        _validate(payload, expected_calibration_hash="other-calibration")
    with pytest.raises(CBLPromotionReceiptError, match="value identity differs"):
        _validate(payload, expected_imatrix_sha256="f" * 64)


def test_receipt_requires_exact_ladder_two_sha256_holdouts_and_imatrix_identity():
    payload = promotion_receipt()
    del payload["rungs"]["48"]
    with pytest.raises(CBLPromotionReceiptError, match="rungs members differ"):
        _validate(payload)

    payload = promotion_receipt()
    payload["holdouts"] = payload["holdouts"][:1]
    with pytest.raises(CBLPromotionReceiptError, match="exactly two holdouts"):
        _validate(payload)

    payload = promotion_receipt()
    payload["holdouts"][0]["calibration_hash"] = "not-a-sha256"
    with pytest.raises(CBLPromotionReceiptError, match="lowercase SHA-256"):
        _validate(payload)


def test_receipt_role_coverage_is_exactly_model_derived():
    payload = promotion_receipt()
    del payload["rungs"]["4"]["holdouts"]["holdout-a"][
        "role_aggregate_ratios"
    ]["up_proj"]
    with pytest.raises(CBLPromotionReceiptError, match="role coverage differs"):
        _validate(payload)

    payload = promotion_receipt()
    payload["rungs"]["4"]["holdouts"]["holdout-a"][
        "role_coverage_counts"
    ]["q_proj"] = 2
    with pytest.raises(CBLPromotionReceiptError, match="model-derived census"):
        _validate(payload)


@pytest.mark.parametrize("mutation", ("qname", "rung", "digest"))
def test_candidate_codebook_digest_coverage_is_closed_and_exact(mutation):
    payload = promotion_receipt()
    if mutation == "qname":
        del payload["candidate_codebooks"][QNAMES[0]]
        match = "candidate_codebooks members differ"
    elif mutation == "rung":
        del payload["candidate_codebooks"][QNAMES[0]]["48"]
        match = "members differ"
    else:
        payload["candidate_codebooks"][QNAMES[0]]["4"][
            "subtable_content_sha256"
        ][0] = "bad"
        match = "lowercase SHA-256"
    with pytest.raises(CBLPromotionReceiptError, match=match):
        _validate(payload)


def test_receipt_file_refuses_duplicate_json_members(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"first","schema":"second"}')
    with pytest.raises(CBLPromotionReceiptError, match="duplicate JSON member"):
        read_promotion_receipt_payload(path)
