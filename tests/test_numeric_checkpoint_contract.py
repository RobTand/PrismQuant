from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
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


def _e2_footprint(
    lane: str, rate: float, shape: list[int], counts, *, nested=False,
):
    keys = (
        C._E2_FOOTPRINT_BASE_KEYS if nested else
        (C._E2_TWO_TIER_FOOTPRINT_KEYS if lane == "tcq_two_tier"
         else C._E2_V1_FOOTPRINT_KEYS)
    )
    rows, columns = shape
    body_bits_per_row = sum(int(key) * value for key, value in counts.items())
    unpadded = (body_bits_per_row + 7) // 8
    stride = ((unpadded + 15) // 16) * 16
    body_bytes = rows * stride
    schedule_bytes = (columns * 4 + 7) // 8
    block_count = (columns + 255) // 256
    offset_bytes = (block_count + 1) * 4
    alphabet_by_rate = {
        key: 3 + (1 << (int(key) + 1))
        for key, value in counts.items() if value and int(key) < 4
    }
    alphabet_bytes = sum(alphabet_by_rate.values())
    side = 88 + schedule_bytes + offset_bytes + alphabet_bytes
    production_scale = rows * ((columns + 15) // 16)
    production_total = body_bytes + production_scale + side
    production = {key: 0 for key in C._E2_FOOTPRINT_BASE_KEYS}
    production.update({
        "schema": "prismaquant.trellis_tensor_payload.v1",
        "wire_schema": "gridbook.trellis.wire.v1",
        "family": "TCQ_E2M1_R256", "grid": "e2m1",
        "format": f"TCQ_E2M1_R{round(rate * 256)}", "shape": shape,
        "body_rate_q256": round(rate * 256),
        "body_bpw": body_bits_per_row / columns,
        "layout": "tight_offsets", "superblock_weights": 256,
        "block_count": block_count, "body_bits_per_row": body_bits_per_row,
        "unpadded_body_bytes_per_row": unpadded,
        "body_row_stride_bytes": stride,
        "body_padding_bytes": rows * (stride - unpadded),
        "body_bytes": body_bytes, "wire_header_bytes": 88,
        "scale_contract": "group16_fp8_e4m3_0p5_bpw",
        "scale_bytes": production_scale,
        "schedule_scope": "tensor_input_column_shared_across_rows",
        "schedule_bits_per_code": 4, "schedule_bytes": schedule_bytes,
        "block_offset_bits": 32, "block_offset_bytes": offset_bytes,
        "alphabet_bytes_by_rate": alphabet_by_rate,
        "alphabet_bytes": alphabet_bytes, "sidecar_header_bytes": 0,
        "side_information_bytes": side, "total_bytes": production_total,
        "exact_bpw": 8.0 * production_total / (rows * columns),
        "expanded_weight_resident_bytes": 0, "producer_eligible": False,
    })
    production["identity_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in production.items()
         if key != "identity_sha256"},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    if nested:
        return production
    if lane == "tcq_v1":
        return {**production, "scale_coding": "v1",
                "non_shipping_research": False}
    research_scale = rows * (columns // 256) * 9
    research_total = body_bytes + research_scale + side
    return {
        **production,
        "schema": "trellis.hull.tcq_two_tier_research_payload.v1",
        "scale_coding": "two_tier", "non_shipping_research": True,
        "scale_contract": "group16_two_tier_9B_per_superblock (RESEARCH)",
        "scale_bytes": research_scale,
        "scale_bytes_v1_production": production_scale,
        "scale_bytes_per_superblock": 9,
        "scale_bpw": research_scale * 8.0 / (rows * columns),
        "total_bytes": research_total,
        "exact_bpw": 8.0 * research_total / (rows * columns),
        "production_payload_v1": production,
        "research_pricing_note": "research repricing",
    }


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
        fraction = columns / shape[1]
        energy = fraction
        trellis_wsse = 0.1 * fraction
        nvfp4_wsse = 0.2 * fraction
        scalar_wsse = 0.125 * fraction
        scalar = {
            "coding_gain_db": 10.0 * math.log10(scalar_wsse / trellis_wsse),
            "db": 10.0 * math.log10(energy / scalar_wsse),
            "levels": sorted(C._E2M1_LEVELS)[:n_levels],
            "n_levels": n_levels, "subset_fit_scope": "",
            "wsse": scalar_wsse,
        }
        oracle = copy.deepcopy(scalar)
        oracle["subset_fit_scope"] = C._E2_SUBSET_SCOPES[
            "scalar_subgrid_oracle"
        ]
        shared = copy.deepcopy(scalar)
        shared["subset_fit_scope"] = C._E2_SUBSET_SCOPES[
            "scalar_subgrid_shared"
        ]
        subset[rate_text] = {
            "bits_per_weight_here": local_rate, "columns": columns,
            "energy": energy, "nvfp4_bits_per_weight": 4,
            "nvfp4_db": 10.0 * math.log10(energy / nvfp4_wsse),
            "nvfp4_wsse": nvfp4_wsse,
            "scalar_subgrid_oracle": oracle,
            "scalar_subgrid_shared": shared,
            "trellis_db": 10.0 * math.log10(energy / trellis_wsse),
            "trellis_minus_nvfp4_db": 10.0 * math.log10(
                nvfp4_wsse / trellis_wsse
            ),
            "trellis_wsse": trellis_wsse,
        }
    schedule = {key: 0 for key in C._E2_SCHEDULE_KEYS}
    schedule.update({
        "target_rate": rate, "achieved_rate": rate, "maximum_rate": 4,
        "invert": False, "fixed_quota_per_256": False,
        "schedule_sha256": "b" * 64, "counts": counts,
        "body_bits_per_block_min": round(rate * 256),
        "body_bits_per_block_max": round(rate * 256),
        "body_bits_per_block_std": 0.0,
        "transitions_per_block_mean": 0.0 if len(subset) == 1 else 1.0,
        "transitions_per_block_max": 0 if len(subset) == 1 else 1,
        "minimum_trellis_steps": 256 if counts["4"] == 0 else 8,
    })
    return {
        "arm": lane, "encode_seconds": 1.0,
        "footprint": _e2_footprint(lane, rate, shape, counts),
        "plain_nsse": 0.1, "plain_snr_db": 10.0, "plain_sse": 0.1,
        "reproduces_stage6": (
            lane == "tcq_v1" and rate in C._E2_REPRODUCTION_RATES
        ),
        "rung": rate, "schedule": schedule,
        "subset_split": subset, "weighted_nsse": 0.1,
        "weighted_snr_db": 10.0, "weighted_sse": 0.1,
    }


def _e2_checkpoint():
    receipt = {key: None for key in C._E2_RECEIPT_KEYS - {"partial", "tensors_done"}}
    receipt.update({
        "schema": "trellis.e2m1_highrate.v3", "started_at_unix_s": 1.0,
        "publication_identity_sha256": "c" * 64, "rate_plan": [2.0],
        "mathematical_q256_bounds": [256, 1016], "control_rungs": [],
    })
    shape = [2, 256]
    cell = {
        "shape": shape, "numel": 512, "population": "dense",
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


def _expected_e2(checkpoint, *, include_second=True):
    cell = checkpoint["per_tensor"]["tensor-a"]
    expected = {
        "tensor-a": {"shape": cell["shape"], "population": cell["population"]}
    }
    if include_second:
        expected["tensor-b"] = copy.deepcopy(expected["tensor-a"])
    return expected


def _expected_e2_controls(checkpoint, *, include_second=True):
    expected = {"tensor-a": {}}
    if include_second:
        expected["tensor-b"] = {}
    return expected


def test_e2_checkpoint_requires_closed_complete_semantics():
    checkpoint, receipt = _e2_checkpoint()
    C.validate_e2m1_checkpoint(
        checkpoint, current_receipt=receipt,
        expected_tensors=_expected_e2(checkpoint),
        expected_controls=_expected_e2_controls(checkpoint),
    )

    for mutate, match in (
        (lambda value: value["receipt"].__setitem__("production_eligible", True), "receipt members"),
        (lambda value: value["per_tensor"]["tensor-a"].__setitem__("unrecognized_claim", True), "members differ"),
        (lambda value: value["per_tensor"]["tensor-a"].__setitem__("arms", {}), "cover every expected arm"),
        (lambda value: value["per_tensor"]["tensor-a"].__setitem__("weighted_energy", float("nan")), "canonical JSON"),
        (lambda value: value["per_tensor"]["tensor-a"]["arms"]["tcq_v1@2.0"].__setitem__("rung", 3.0), "rung"),
        (lambda value: value["per_tensor"]["tensor-a"]["arms"]["tcq_v1@2.0"]["footprint"].__setitem__("shipping_claim", True), "footprint members"),
        (lambda value: value["per_tensor"]["tensor-a"]["arms"]["tcq_v1@2.0"]["footprint"].__setitem__("total_bytes", 999), "total_bytes accounting"),
        (lambda value: value["per_tensor"]["tensor-a"]["arms"]["tcq_v1@2.0"]["schedule"].__setitem__("stage4_guard_fixups", 1), "guard fixup"),
        (lambda value: value["per_tensor"]["tensor-a"]["arms"]["tcq_v1@2.0"]["subset_split"]["2"].__setitem__("trellis_db", 999.0), "trellis_db"),
    ):
        bad = copy.deepcopy(checkpoint)
        mutate(bad)
        _seal_e2(bad)
        with pytest.raises(C.CheckpointContractError, match=match):
            C.validate_e2m1_checkpoint(
                bad, current_receipt=receipt,
                expected_tensors=_expected_e2(checkpoint),
                expected_controls=_expected_e2_controls(checkpoint),
            )

    drifted = copy.deepcopy(receipt)
    drifted["corpus_binding"] = {"input_sha256": "9" * 64}
    with pytest.raises(C.CheckpointContractError, match="identity differs"):
        C.validate_e2m1_checkpoint(
            checkpoint, current_receipt=drifted,
            expected_tensors=_expected_e2(checkpoint),
            expected_controls=_expected_e2_controls(checkpoint),
        )

    stale = copy.deepcopy(checkpoint)
    stale["per_tensor"]["tensor-a"]["weighted_energy"] = 2.0
    with pytest.raises(C.CheckpointContractError, match="self-digest differs"):
        C.validate_e2m1_checkpoint(
            stale, current_receipt=receipt,
            expected_tensors=_expected_e2(checkpoint),
            expected_controls=_expected_e2_controls(checkpoint),
        )

    fake_shape = copy.deepcopy(checkpoint)
    fake_shape["per_tensor"]["tensor-a"]["shape"] = [1, 1]
    fake_shape["per_tensor"]["tensor-a"]["numel"] = 1
    _seal_e2(fake_shape)
    with pytest.raises(C.CheckpointContractError, match="logical identity"):
        C.validate_e2m1_checkpoint(
            fake_shape, current_receipt=receipt,
            expected_tensors=_expected_e2(checkpoint),
            expected_controls=_expected_e2_controls(checkpoint),
        )

    skipped_control = copy.deepcopy(checkpoint)
    skipped_receipt = copy.deepcopy(receipt)
    skipped_receipt["control_rungs"] = ["tcq_v1@2.0"]
    skipped_control["receipt"]["control_rungs"] = ["tcq_v1@2.0"]
    _seal_e2(skipped_control)
    arm = checkpoint["per_tensor"]["tensor-a"]["arms"]["tcq_v1@2.0"]
    published_control = {
        "metrics": {
            field: arm[field]
            for field in (
                "weighted_sse", "weighted_nsse", "weighted_snr_db",
                "plain_sse", "plain_nsse",
            )
        },
        "footprint": {
            field: arm["footprint"][field]
            for field in ("total_bytes", "body_rate_q256")
        },
    }
    expected_controls = {
        name: {"tcq_v1@2.0": copy.deepcopy(published_control)}
        for name in ("tensor-a", "tensor-b")
    }
    with pytest.raises(C.CheckpointContractError, match="checks domain"):
        C.validate_e2m1_checkpoint(
            skipped_control, current_receipt=skipped_receipt,
            expected_tensors=_expected_e2(checkpoint),
            expected_controls=expected_controls,
        )


def test_e2_control_domain_and_published_values_are_bound():
    checkpoint, receipt = _e2_checkpoint()
    rung = "tcq_v1@2.0"
    fields = (
        "weighted_sse", "weighted_nsse", "weighted_snr_db",
        "plain_sse", "plain_nsse",
    )
    receipt["control_rungs"] = [rung]
    receipt["control_rtol"] = 1e-9
    checkpoint["receipt"]["control_rungs"] = [rung]
    checkpoint["receipt"]["control_rtol"] = 1e-9
    arm = checkpoint["per_tensor"]["tensor-a"]["arms"][rung]
    checkpoint["per_tensor"]["tensor-a"]["control"] = {
        "status": "pass",
        "worst_relative": 0.0,
        "footprint_equal": True,
        "checks": {
            f"{rung}.{field}": {
                "mine": arm[field], "published": arm[field], "rel": 0.0,
            }
            for field in fields
        },
    }
    expected_controls = {
        "tensor-a": {
            rung: {
                "metrics": {field: arm[field] for field in fields},
                "footprint": {
                    field: arm["footprint"][field]
                    for field in ("total_bytes", "body_rate_q256")
                },
            }
        }
    }
    _seal_e2(checkpoint)
    C.validate_e2m1_checkpoint(
        checkpoint,
        current_receipt=receipt,
        expected_tensors=_expected_e2(checkpoint, include_second=False),
        expected_controls=expected_controls,
    )

    uncontrolled = copy.deepcopy(checkpoint)
    uncontrolled["per_tensor"]["tensor-a"]["control"]["status"] = (
        "uncontrolled"
    )
    _seal_e2(uncontrolled)
    with pytest.raises(C.CheckpointContractError, match="declared control"):
        C.validate_e2m1_checkpoint(
            uncontrolled,
            current_receipt=receipt,
            expected_tensors=_expected_e2(checkpoint, include_second=False),
            expected_controls=expected_controls,
        )

    unbound = copy.deepcopy(checkpoint)
    check = unbound["per_tensor"]["tensor-a"]["control"]["checks"][
        f"{rung}.weighted_sse"
    ]
    check["published"] += 1.0
    check["rel"] = abs(check["mine"] - check["published"]) / abs(
        check["published"]
    )
    unbound["per_tensor"]["tensor-a"]["control"]["worst_relative"] = (
        check["rel"]
    )
    unbound["per_tensor"]["tensor-a"]["control"]["status"] = "fail"
    _seal_e2(unbound)
    with pytest.raises(C.CheckpointContractError, match="bound control"):
        C.validate_e2m1_checkpoint(
            unbound,
            current_receipt=receipt,
            expected_tensors=_expected_e2(checkpoint, include_second=False),
            expected_controls=expected_controls,
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
        checkpoint, current_receipt=receipt,
        expected_tensors=_expected_e2(checkpoint, include_second=False),
        expected_controls=_expected_e2_controls(
            checkpoint, include_second=False
        ),
    )

    one_sided = copy.deepcopy(checkpoint)
    one_sided["per_tensor"]["tensor-a"]["unreachable_rungs"] = paired[:1]
    one_sided["per_tensor"]["tensor-a"]["arms"]["tcq_v1@3.96875"] = (
        _e2_arm("tcq_v1", 3.96875, [2, 256])
    )
    _seal_e2(one_sided)
    with pytest.raises(C.CheckpointContractError, match="one-sided"):
        C.validate_e2m1_checkpoint(
            one_sided, current_receipt=receipt,
            expected_tensors=_expected_e2(checkpoint, include_second=False),
            expected_controls=_expected_e2_controls(
                checkpoint, include_second=False
            ),
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
            all_unreachable, current_receipt=receipt,
            expected_tensors=_expected_e2(checkpoint, include_second=False),
            expected_controls=_expected_e2_controls(
                checkpoint, include_second=False
            ),
        )


def test_e2_final_control_and_population_receipts_are_derived():
    checkpoint, receipt = _e2_checkpoint()
    final_receipt = {
        **receipt,
        "completed_at_unix_s": 2.0,
        "tensors_done": 1,
        "status": "ok",
        "control_verdict": {"tensor-a": "uncontrolled"},
        "population_counts": {"dense": 1},
    }
    checkpoint["receipt"] = {**final_receipt, "partial": False}
    _seal_e2(checkpoint)
    C.validate_e2m1_checkpoint(
        checkpoint, current_receipt=final_receipt,
        expected_tensors=_expected_e2(checkpoint, include_second=False),
        expected_controls=_expected_e2_controls(
            checkpoint, include_second=False
        ),
        require_partial=False,
    )
    bad = copy.deepcopy(checkpoint)
    bad["receipt"]["population_counts"] = {"dense": 99}
    bad_receipt = {**final_receipt, "population_counts": {"dense": 99}}
    _seal_e2(bad)
    with pytest.raises(C.CheckpointContractError, match="population_counts"):
        C.validate_e2m1_checkpoint(
            bad, current_receipt=bad_receipt,
            expected_tensors=_expected_e2(checkpoint, include_second=False),
            expected_controls=_expected_e2_controls(
                checkpoint, include_second=False
            ),
            require_partial=False,
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


def _fp8_footprint(rung: int, learned: bool, shape=(2, 256)):
    keys = C._FP8_LEARNED_FOOTPRINT_KEYS if learned else C._FP8_FIXED_FOOTPRINT_KEYS
    rows, columns = shape
    numel = rows * columns
    superblocks = numel // 256
    index_bytes = rung * 4
    body_bits = 8 * index_bytes * superblocks
    row_scale_bytes = rows * 4
    scale_bits = row_scale_bytes * 8
    table_rows = 1 << (rung // 4)
    elements = table_rows * 8 if learned else 0
    book_bits = elements * 16
    total_bits = body_bits + scale_bits + book_bits
    result = {key: 0 for key in keys}
    result.update({
        "schema": "trellis.fp8_ladder.fp8_cb_accounting.v1",
        "format": f"FP8_CB_K{rung}",
        "codebook": "per_tensor_weighted_lloyd" if learned else "fixed_lattice",
        "body_bpw": body_bits / numel, "exact_bpw": total_bits / numel,
        "body_bits": body_bits, "total_bits": total_bits,
        "total_bytes": total_bits // 8, "superblocks": superblocks,
        "type_size_bytes_per_superblock": index_bytes,
        "index_bytes_per_superblock": index_bytes,
        "scale_bytes_per_superblock": 0,
        "row_scale_bytes": row_scale_bytes, "scale_bits": scale_bits,
        "scale_bpw": scale_bits / numel, "scale_coding": "v1",
        "scale_contract": "per_output_row_fp32",
        "backed_on_sm120": True,
        "sidecar_amortization": C._FP8_SIDECAR_AMORTIZATION,
    })
    if learned:
        result.update({
            "fixed_lattice_is_format_shared": False,
            "learned_book_elements": elements, "learned_book_n_sub": 4,
            "learned_book_subtable_shapes": [[table_rows, 2]] * 4,
            "learned_book_bits_per_element": 16,
            "codebook_side_bits": book_bits,
            "codebook_side_bpw": book_bits / numel,
            "codebook_side_bits_wire8": elements * 8,
            "codebook_side_bpw_wire8": elements * 8 / numel,
            "exact_bpw_book_wire8": (body_bits + scale_bits + elements * 8) / numel,
            "fp4_level_bits_charge_would_have_been_bits": elements * 4,
            "book_price_bracket_note": C._FP8_BOOK_PRICE_NOTE,
            "learned_book_is_per_tensor": C._FP8_PER_TENSOR_BOOK_NOTE,
        })
    return result


def _fp8_arm(rung: int, learned: bool, shape=(2, 256)):
    arm = {
        "encode_seconds_observation_not_perf_claim": 1.0,
        "encode_tier": "balanced",
        "footprint": _fp8_footprint(rung, learned, shape),
        "reconstruction_sha256": "f" * 64, "weighted_nsse": 0.1,
        "weighted_snr_db": 10.0, "weighted_sse": 0.1,
    }
    if learned:
        table_rows = 1 << (rung // 4)
        arm["learned_book"] = {
            "elements": table_rows * 8,
            "tables": [
                {"amax": 1.0, "distinct_levels": table_rows,
                 "sha256": str(i) * 64, "shape": [table_rows, 2]}
                for i in range(1, 5)
            ],
        }
    return arm


def _fp8_checkpoint():
    settings = _fp8_settings()
    entry = SimpleNamespace(
        name="tensor-a", population="dense", source_weight_sha256="1" * 64,
        importance_sha256="2" * 64, source_weight_shape=(2, 256),
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
            "population": "dense", "shape": [2, 256],
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
        (lambda value: (
            value["per_tensor"]["tensor-a"]["arms"]["fp8_cb@32"]["footprint"].__setitem__("total_bits", 9992),
            value["per_tensor"]["tensor-a"]["arms"]["fp8_cb@32"]["footprint"].__setitem__("total_bytes", 1249),
        ), "total accounting"),
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

    hashes = {
        "tensor-a": {
            arm_name: arm["reconstruction_sha256"]
            for arm_name, arm in root["per_tensor"]["tensor-a"]["arms"].items()
        }
    }
    books = {
        "tensor-a": {
            arm_name: arm["learned_book"]
            for arm_name, arm in root["per_tensor"]["tensor-a"]["arms"].items()
            if "learned_book" in arm
        }
    }
    C.validate_fp8_checkpoint(
        root, settings=settings, entries=entries,
        generated_hashes=hashes, generated_books=books,
    )
    wrong_hash = copy.deepcopy(hashes)
    wrong_hash["tensor-a"]["fp8_cb@32"] = "0" * 64
    with pytest.raises(C.CheckpointContractError, match="generated object"):
        C.validate_fp8_checkpoint(
            root, settings=settings, entries=entries,
            generated_hashes=wrong_hash, generated_books=books,
        )


def test_fp8_final_summary_and_performance_gate_are_derived():
    root, settings, entries = _fp8_checkpoint()
    root.update({
        "partial": False,
        "completed_at_unix_s": 2.0,
        "population_summaries": C.fp8_population_summaries(root["per_tensor"]),
        "status": "measurement_complete_no_serving_verdict",
        "performance_gate": C.FP8_PERFORMANCE_GATE,
    })
    _seal_fp8(root)
    C.validate_fp8_checkpoint(
        root, settings=settings, entries=entries, require_partial=False
    )

    bad_summary = copy.deepcopy(root)
    bad_summary["population_summaries"]["dense"]["rows"][0][
        "learned_better"
    ] = 999
    _seal_fp8(bad_summary)
    with pytest.raises(C.CheckpointContractError, match="population_summaries"):
        C.validate_fp8_checkpoint(
            bad_summary, settings=settings, entries=entries,
            require_partial=False,
        )

    bad_gate = copy.deepcopy(root)
    bad_gate["performance_gate"] = "fast enough for production"
    _seal_fp8(bad_gate)
    with pytest.raises(C.CheckpointContractError, match="performance_gate"):
        C.validate_fp8_checkpoint(
            bad_gate, settings=settings, entries=entries,
            require_partial=False,
        )
