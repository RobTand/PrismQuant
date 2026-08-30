from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/numeric_checkpoint_contract.py"
)
SPEC = importlib.util.spec_from_file_location("numeric_checkpoint_contract", PATH)
assert SPEC is not None and SPEC.loader is not None
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def _e2_footprint(lane: str, rate: float, shape: list[int], *, nested=False):
    keys = (
        C._E2_FOOTPRINT_BASE_KEYS if nested else
        (C._E2_TWO_TIER_FOOTPRINT_KEYS if lane == "tcq_two_tier"
         else C._E2_V1_FOOTPRINT_KEYS)
    )
    out = {key: 0 for key in keys}
    out.update({
        "family": "TCQ_E2M1_R256", "grid": "e2m1",
        "format": f"TCQ_E2M1_R{round(rate * 256)}", "shape": shape,
        "body_rate_q256": round(rate * 256), "body_bpw": rate,
        "exact_bpw": rate + 0.5, "total_bytes": 8,
        "identity_sha256": "a" * 64,
    })
    if not nested:
        out["scale_coding"] = "two_tier" if lane == "tcq_two_tier" else "v1"
    if lane == "tcq_two_tier" and not nested:
        out["production_payload_v1"] = _e2_footprint(
            "tcq_v1", rate, shape, nested=True
        )
    return out


def _e2_arm(lane: str, rate: float, shape: list[int]):
    if rate == 3.96875:
        assert shape[1] == 256
        counts = {"1": 0, "2": 0, "3": 8, "4": 248}
    else:
        integer_rate = int(rate)
        assert float(integer_rate) == rate
        counts = {str(value): shape[1] if value == integer_rate else 0
                  for value in range(1, 5)}
    subset = {}
    for rate_text, columns in counts.items():
        if not columns:
            continue
        local_rate = int(rate_text)
        n_levels = (1 << local_rate) if local_rate < 4 else 15
        scalar = {
            "coding_gain_db": 1.0, "db": 10.0,
            "levels": [float(index) for index in range(n_levels)],
            "n_levels": n_levels, "subset_fit_scope": "whole tensor",
            "wsse": 0.1,
        }
        subset[rate_text] = {
            "bits_per_weight_here": local_rate, "columns": columns,
            "energy": 1.0, "nvfp4_bits_per_weight": 4, "nvfp4_db": 9.0,
            "nvfp4_wsse": 0.2, "scalar_subgrid_oracle": copy.deepcopy(scalar),
            "scalar_subgrid_shared": copy.deepcopy(scalar), "trellis_db": 10.0,
            "trellis_minus_nvfp4_db": 1.0, "trellis_wsse": 0.1,
        }
    schedule = {key: 0 for key in C._E2_SCHEDULE_KEYS}
    schedule.update({
        "target_rate": rate, "achieved_rate": rate, "maximum_rate": 4,
        "invert": False, "fixed_quota_per_256": False,
        "schedule_sha256": "b" * 64, "counts": counts,
        "minimum_trellis_steps": 8,
    })
    return {
        "arm": lane, "encode_seconds": 1.0,
        "footprint": _e2_footprint(lane, rate, shape),
        "plain_nsse": 0.1, "plain_snr_db": 10.0, "plain_sse": 0.1,
        "reproduces_stage6": False, "rung": rate, "schedule": schedule,
        "subset_split": subset, "weighted_nsse": 0.1,
        "weighted_snr_db": 10.0, "weighted_sse": 0.1,
    }


def _e2_checkpoint():
    receipt = {key: None for key in C._E2_RECEIPT_KEYS - {"partial", "tensors_done"}}
    receipt.update({
        "schema": "trellis.e2m1_highrate.v3", "started_at_unix_s": 1.0,
        "publication_identity_sha256": "c" * 64, "rate_plan": [2.0],
        "mathematical_q256_bounds": [256, 1016],
    })
    shape = [2, 3]
    cell = {
        "shape": shape, "numel": 6, "population": "dense",
        "weighted_energy": 1.0, "plain_energy": 1.0,
        "two_tier_plane_sha256": "d" * 64,
        "arms": {
            "tcq_two_tier@2.0": _e2_arm("tcq_two_tier", 2.0, shape),
            "tcq_v1@2.0": _e2_arm("tcq_v1", 2.0, shape),
        },
        "unreachable_rungs": [],
        "control": {
            "status": "uncontrolled", "worst_relative": None,
            "footprint_equal": False, "checks": {},
        },
    }
    saved = {**receipt, "partial": True, "tensors_done": 1}
    checkpoint = {
        "receipt": saved, "per_tensor": {"tensor-a": cell},
        "checkpoint_sha256": "",
    }
    _seal_e2(checkpoint)
    return checkpoint, receipt


def _seal_e2(checkpoint):
    body = {key: checkpoint[key] for key in ("receipt", "per_tensor")}
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def test_e2_checkpoint_requires_closed_complete_semantics():
    checkpoint, receipt = _e2_checkpoint()
    C.validate_e2m1_checkpoint(
        checkpoint, current_receipt=receipt, names=["tensor-a", "tensor-b"]
    )

    for mutate, match in (
        (lambda value: value["receipt"].__setitem__("production_eligible", True), "receipt members"),
        (lambda value: value["per_tensor"]["tensor-a"].__setitem__("unrecognized_claim", True), "members differ"),
        (lambda value: value["per_tensor"]["tensor-a"].__setitem__("arms", {}), "cover every expected arm"),
        (lambda value: value["per_tensor"]["tensor-a"].__setitem__("weighted_energy", float("nan")), "canonical JSON"),
        (lambda value: value["per_tensor"]["tensor-a"]["arms"]["tcq_v1@2.0"].__setitem__("rung", 3.0), "rung"),
        (lambda value: value["per_tensor"]["tensor-a"]["arms"]["tcq_v1@2.0"]["footprint"].__setitem__("shipping_claim", True), "footprint members"),
    ):
        bad = copy.deepcopy(checkpoint)
        mutate(bad)
        _seal_e2(bad)
        with pytest.raises(C.CheckpointContractError, match=match):
            C.validate_e2m1_checkpoint(
                bad, current_receipt=receipt, names=["tensor-a", "tensor-b"]
            )

    drifted = copy.deepcopy(receipt)
    drifted["corpus_binding"] = {"input_sha256": "9" * 64}
    with pytest.raises(C.CheckpointContractError, match="identity differs"):
        C.validate_e2m1_checkpoint(
            checkpoint, current_receipt=drifted, names=["tensor-a", "tensor-b"]
        )

    stale = copy.deepcopy(checkpoint)
    stale["per_tensor"]["tensor-a"]["weighted_energy"] = 2.0
    with pytest.raises(C.CheckpointContractError, match="self-digest differs"):
        C.validate_e2m1_checkpoint(
            stale, current_receipt=receipt, names=["tensor-a", "tensor-b"]
        )


def test_e2_unreachable_is_only_paired_mathematical_ceiling_refusal():
    checkpoint, receipt = _e2_checkpoint()
    cell = checkpoint["per_tensor"]["tensor-a"]
    cell["shape"] = [2, 256]
    cell["numel"] = 512
    cell["arms"] = {
        "tcq_two_tier@2.0": _e2_arm("tcq_two_tier", 2.0, [2, 256]),
        "tcq_v1@2.0": _e2_arm("tcq_v1", 2.0, [2, 256]),
    }
    receipt["rate_plan"] = [2.0, 3.96875]
    checkpoint["receipt"]["rate_plan"] = [2.0, 3.96875]
    paired = [
        {"lane": lane, "rate": 3.96875,
         "reason": "cannot rebalance trellis-length guard"}
        for lane in ("tcq_two_tier", "tcq_v1")
    ]
    cell["unreachable_rungs"] = paired
    _seal_e2(checkpoint)
    C.validate_e2m1_checkpoint(
        checkpoint, current_receipt=receipt, names=["tensor-a"]
    )

    one_sided = copy.deepcopy(checkpoint)
    one_sided["per_tensor"]["tensor-a"]["unreachable_rungs"] = paired[:1]
    one_sided["per_tensor"]["tensor-a"]["arms"]["tcq_v1@3.96875"] = (
        _e2_arm("tcq_v1", 3.96875, [2, 256])
    )
    _seal_e2(one_sided)
    with pytest.raises(C.CheckpointContractError, match="one-sided"):
        C.validate_e2m1_checkpoint(
            one_sided, current_receipt=receipt, names=["tensor-a"]
        )

    all_unreachable = copy.deepcopy(checkpoint)
    all_unreachable["per_tensor"]["tensor-a"]["arms"] = {}
    all_unreachable["per_tensor"]["tensor-a"]["unreachable_rungs"] = [
        {"lane": lane, "rate": rate,
         "reason": "cannot rebalance trellis-length guard"}
        for rate in (2.0, 3.96875)
        for lane in ("tcq_two_tier", "tcq_v1")
    ]
    _seal_e2(all_unreachable)
    with pytest.raises(C.CheckpointContractError, match="cannot declare every arm unreachable"):
        C.validate_e2m1_checkpoint(
            all_unreachable, current_receipt=receipt, names=["tensor-a"]
        )


def _fp8_settings():
    settings = {
        "schema": "trellis.glm_fp8_learned_balanced.v2",
        "corpus_manifest": "/immutable/manifest.json",
        "corpus_manifest_sha256": "a" * 64,
        "corpus_file_sha256": "b" * 64,
        "importance_value_sha256": "c" * 64,
        "corpus_prismaquant_commit": "d" * 40,
        "population_counts": {"dense": 1, "routed": 0},
        "rungs": [32, 40, 48], "encode_tier": "balanced",
        "locked_sources": {}, "frozen_codec_closure": {},
        "active_source_identity": {},
        "aggregation_contract": "dense/routed population-separated; no pooled median",
    }
    settings["identity_sha256"] = hashlib.sha256(json.dumps(
        settings, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return settings


def _fp8_footprint(rung: int, learned: bool):
    keys = C._FP8_LEARNED_FOOTPRINT_KEYS if learned else C._FP8_FIXED_FOOTPRINT_KEYS
    result = {key: 0 for key in keys}
    result.update({
        "schema": "trellis.fp8_ladder.fp8_cb_accounting.v1",
        "format": f"FP8_CB_K{rung}",
        "codebook": "per_tensor_weighted_lloyd" if learned else "fixed_lattice",
        "body_bpw": rung / 8.0, "exact_bpw": rung / 8.0 + 0.5,
        "body_bits": 32, "total_bits": 64, "total_bytes": 8,
        "superblocks": 1, "scale_coding": "v1",
        "scale_contract": "per_output_row_fp32",
    })
    if learned:
        result.update({
            "learned_book_elements": 16, "learned_book_n_sub": 4,
            "learned_book_subtable_shapes": [[2, 2]] * 4,
        })
    return result


def _fp8_arm(rung: int, learned: bool):
    arm = {
        "encode_seconds_observation_not_perf_claim": 1.0,
        "encode_tier": "balanced", "footprint": _fp8_footprint(rung, learned),
        "reconstruction_sha256": "f" * 64, "weighted_nsse": 0.1,
        "weighted_snr_db": 10.0, "weighted_sse": 0.1,
    }
    if learned:
        arm["learned_book"] = {
            "elements": 16,
            "tables": [
                {"amax": 1.0, "distinct_levels": 4, "sha256": str(i) * 64,
                 "shape": [2, 2]}
                for i in range(1, 5)
            ],
        }
    return arm


def _fp8_checkpoint():
    settings = _fp8_settings()
    entry = SimpleNamespace(
        name="tensor-a", population="dense", source_weight_sha256="1" * 64,
        importance_sha256="2" * 64, source_weight_shape=(2, 3),
        importance_source_qname="q", importance_source_expert=None,
        importance_denominator_name="n_tokens_seen", importance_denominator=8,
    )
    arms = {}
    for rung in (32, 40, 48):
        arms[f"fp8_cb@{rung}"] = _fp8_arm(rung, False)
        arms[f"fp8_cb_learned@{rung}"] = _fp8_arm(rung, True)
    root = {
        "schema": settings["schema"], "settings": settings,
        "started_at_unix_s": 1.0,
        "per_tensor": {"tensor-a": {
            "population": "dense", "shape": [2, 3],
            "source_weight_sha256": "1" * 64, "importance_sha256": "2" * 64,
            "importance_source": {
                "qname": "q", "expert": None,
                "denominator_name": "n_tokens_seen", "denominator": 8,
            },
            "weighted_energy": 1.0, "arms": arms,
        }},
        "partial": True, "tensors_done": 1, "checkpoint_sha256": "",
    }
    _seal_fp8(root)
    return root, settings, [entry]


def _seal_fp8(root):
    body = {key: value for key, value in root.items()
            if key != "checkpoint_sha256"}
    root["checkpoint_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def test_fp8_checkpoint_rejects_unknown_claims_and_fake_metrics():
    root, settings, entries = _fp8_checkpoint()
    C.validate_fp8_checkpoint(root, settings=settings, entries=entries)
    for mutate, match in (
        (lambda value: value.__setitem__("production_eligible", True), "checkpoint members"),
        (lambda value: value["settings"].__setitem__("serving_ready", True), "settings members"),
        (lambda value: value["per_tensor"]["tensor-a"].__setitem__("claim", True), "members differ"),
        (lambda value: value["per_tensor"]["tensor-a"]["importance_source"].__setitem__("estimated", True), "importance_source members"),
        (lambda value: value["per_tensor"]["tensor-a"].__setitem__("weighted_energy", 0.0), "positive"),
        (lambda value: value["per_tensor"]["tensor-a"]["arms"].pop("fp8_cb@32"), "arms domain"),
        (lambda value: value["per_tensor"]["tensor-a"]["arms"]["fp8_cb@32"].__setitem__("production_claim", True), "members differ"),
        (lambda value: value["per_tensor"]["tensor-a"]["arms"]["fp8_cb@32"]["footprint"].__setitem__("production_claim", True), "members differ"),
        (lambda value: value["per_tensor"]["tensor-a"]["arms"]["fp8_cb_learned@32"]["learned_book"].__setitem__("quality_claim", True), "members differ"),
    ):
        bad = copy.deepcopy(root)
        mutate(bad)
        _seal_fp8(bad)
        with pytest.raises(C.CheckpointContractError, match=match):
            C.validate_fp8_checkpoint(bad, settings=settings, entries=entries)

    stale = copy.deepcopy(root)
    stale["per_tensor"]["tensor-a"]["weighted_energy"] = 2.0
    with pytest.raises(C.CheckpointContractError, match="self-digest differs"):
        C.validate_fp8_checkpoint(stale, settings=settings, entries=entries)
