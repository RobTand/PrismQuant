"""Closed semantic validators for resumable numeric campaign checkpoints.

The checkpoint digest detects accidental byte changes.  These validators are
the trust boundary: they reject extra claim-bearing members and require every
completed tensor to contain the full, internally coherent measurement domain.
This module is deliberately stdlib-only so importing it cannot escape the
frozen research-code hull.
"""
from __future__ import annotations

import math
import re
import hashlib
import json
import statistics
from typing import Mapping, Sequence


class CheckpointContractError(ValueError):
    pass


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_HOST = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,253}[A-Za-z0-9])?$")
_EXECUTION_KEYS = frozenset({
    "schema", "physical_host", "container_image_digest", "repo_git_commit",
    "repo_tree_clean", "python", "torch", "triton", "device",
})
_E2_LANES = ("tcq_two_tier", "tcq_v1")
_E2_ROOT_KEYS = frozenset({"receipt", "per_tensor", "checkpoint_sha256"})
_E2_RECEIPT_KEYS = frozenset({
    "schema", "started_at_unix_s", "question", "control_rungs",
    "arms_measured", "control_source", "control_rtol", "corpus",
    "corpus_label", "corpus_manifest", "corpus_binding",
    "active_source_identity", "publication_identity_sha256",
    "glm_rate_plan", "aggregation_contract", "rate_plan", "new_rates",
    "mathematical_q256_bounds", "arms", "pricing", "environment",
    "partial", "tensors_done",
})
_E2_FINAL_RECEIPT_KEYS = _E2_RECEIPT_KEYS | {
    "completed_at_unix_s", "status", "control_verdict", "population_counts",
}
_E2_CELL_KEYS = frozenset({
    "shape", "numel", "population", "weighted_energy", "plain_energy",
    "two_tier_plane_sha256", "arms", "unreachable_rungs", "control",
})
_E2_EXPECTED_TENSOR_KEYS = frozenset({"shape", "population"})
_E2_ARM_KEYS = frozenset({
    "arm", "encode_seconds", "footprint", "plain_nsse", "plain_snr_db",
    "plain_sse", "reproduces_stage6", "rung", "schedule", "subset_split",
    "weighted_nsse", "weighted_snr_db", "weighted_sse",
})
_E2_PUBLISHED_ARM_KEYS = _E2_ARM_KEYS - {"subset_split"}
_E2_PUBLISHED_BF16_ARM_KEYS = _E2_PUBLISHED_ARM_KEYS - {"plain_snr_db"}
_E2_SCHEDULE_KEYS = frozenset({
    "target_rate", "achieved_rate", "maximum_rate", "invert",
    "fixed_quota_per_256", "tailbite_guard_fixups", "schedule_sha256",
    "counts", "body_bits_per_block_min", "body_bits_per_block_max",
    "body_bits_per_block_std", "transitions_per_block_mean",
    "transitions_per_block_max", "minimum_trellis_steps",
    "stage4_guard_fixups",
})
_E2_CONTROL_KEYS = frozenset({
    "status", "worst_relative", "footprint_equal", "checks",
})
_E2_CONTROL_CHECK_KEYS = frozenset({"mine", "published", "rel"})
_E2_EXPECTED_CONTROL_KEYS = frozenset({"metrics", "footprint"})
_E2_EXPECTED_CONTROL_FOOTPRINT_KEYS = frozenset({
    "total_bytes", "body_rate_q256",
})
_E2_UNREACHABLE_KEYS = frozenset({"lane", "rate", "reason"})
_E2_CEILING_RATE = 3.96875
_E2_CEILING_REFUSAL = "cannot rebalance trellis-length guard"
_E2_REPRODUCTION_RATES = frozenset({1.5, 2.0, 2.5})
_E2M1_LEVELS = frozenset({
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
})
_E2_SUBSET_SCOPES = {
    "scalar_subgrid_oracle": (
        "these columns only (ORACLE: a tougher baseline than the trellis "
        "gets, whose alphabet is fit tensor-wide -- so a positive gain here "
        "is conservative)"
    ),
    "scalar_subgrid_shared": (
        "whole tensor (the SAME fitting scope the trellis alphabet gets)"
    ),
}
_E2_SUBSET_KEYS = frozenset({
    "bits_per_weight_here", "columns", "energy", "nvfp4_bits_per_weight",
    "nvfp4_db", "nvfp4_wsse", "scalar_subgrid_oracle",
    "scalar_subgrid_shared", "trellis_db", "trellis_minus_nvfp4_db",
    "trellis_wsse",
})
_E2_SCALAR_KEYS = frozenset({
    "coding_gain_db", "db", "levels", "n_levels", "subset_fit_scope", "wsse",
})
_E2_FOOTPRINT_BASE_KEYS = frozenset({
    "alphabet_bytes", "alphabet_bytes_by_rate", "block_count",
    "block_offset_bits", "block_offset_bytes", "body_bits_per_row",
    "body_bpw", "body_bytes", "body_padding_bytes", "body_rate_q256",
    "body_row_stride_bytes", "exact_bpw", "expanded_weight_resident_bytes",
    "family", "format", "grid", "identity_sha256", "layout",
    "producer_eligible", "scale_bytes", "scale_contract",
    "schedule_bits_per_code", "schedule_bytes", "schedule_scope", "schema",
    "shape", "side_information_bytes", "sidecar_header_bytes",
    "superblock_weights", "total_bytes", "unpadded_body_bytes_per_row",
    "wire_header_bytes", "wire_schema",
})
_E2_V1_FOOTPRINT_KEYS = _E2_FOOTPRINT_BASE_KEYS | {
    "non_shipping_research", "scale_coding",
}
_E2_TWO_TIER_FOOTPRINT_KEYS = _E2_V1_FOOTPRINT_KEYS | {
    "alphabet_bytes_by_rate", "production_payload_v1",
    "scale_bpw", "scale_bytes_per_superblock", "scale_bytes_v1_production",
    "research_pricing_note",
}

_FP8_ROOT_KEYS = frozenset({
    "schema", "settings", "started_at_unix_s", "per_tensor", "partial",
    "tensors_done", "checkpoint_sha256",
})
_FP8_FINAL_ROOT_KEYS = _FP8_ROOT_KEYS | {
    "completed_at_unix_s", "population_summaries", "status", "performance_gate",
}
_FP8_SETTINGS_KEYS = frozenset({
    "schema", "corpus_manifest", "corpus_manifest_sha256",
    "corpus_file_sha256", "importance_value_sha256",
    "corpus_prismaquant_commit", "population_counts", "rungs", "encode_tier",
    "locked_sources", "frozen_codec_closure", "active_source_identity",
    "environment", "aggregation_contract", "identity_sha256",
})
_FP8_CELL_KEYS = frozenset({
    "population", "shape", "source_weight_sha256", "importance_sha256",
    "importance_source", "weighted_energy", "arms",
})
_FP8_IMPORTANCE_KEYS = frozenset({
    "qname", "expert", "denominator_name", "denominator",
})
_FP8_FIXED_ARM_KEYS = frozenset({
    "encode_seconds_observation_not_perf_claim", "encode_tier", "footprint",
    "reconstruction_sha256", "weighted_nsse", "weighted_snr_db", "weighted_sse",
})
_FP8_LEARNED_ARM_KEYS = _FP8_FIXED_ARM_KEYS | {"learned_book"}
_FP8_FIXED_FOOTPRINT_KEYS = frozenset({
    "backed_on_sm120", "body_bits", "body_bpw", "codebook", "exact_bpw",
    "format", "index_bytes_per_superblock", "row_scale_bytes", "scale_bits",
    "scale_bpw", "scale_bytes_per_superblock", "scale_coding",
    "scale_contract", "schema", "sidecar_amortization", "superblocks",
    "total_bits", "total_bytes", "type_size_bytes_per_superblock",
})
_FP8_LEARNED_FOOTPRINT_KEYS = frozenset({
    "backed_on_sm120", "body_bits", "body_bpw", "book_price_bracket_note",
    "codebook", "codebook_side_bits", "codebook_side_bits_wire8",
    "codebook_side_bpw", "codebook_side_bpw_wire8", "exact_bpw",
    "exact_bpw_book_wire8", "fixed_lattice_is_format_shared", "format",
    "fp4_level_bits_charge_would_have_been_bits", "index_bytes_per_superblock",
    "learned_book_bits_per_element", "learned_book_elements",
    "learned_book_is_per_tensor", "learned_book_n_sub",
    "learned_book_subtable_shapes", "row_scale_bytes", "scale_bits",
    "scale_bpw", "scale_bytes_per_superblock", "scale_coding",
    "scale_contract", "schema", "sidecar_amortization", "superblocks",
    "total_bits", "total_bytes", "type_size_bytes_per_superblock",
})
_FP8_BOOK_KEYS = frozenset({"elements", "tables"})
_FP8_TABLE_KEYS = frozenset({"amax", "distinct_levels", "sha256", "shape"})
FP8_PERFORMANCE_GATE = (
    "encode timings are observations only; attach in-process profiler "
    "and both-host Netdata/power evidence before any performance claim"
)


def _validate_execution_environment(
    value: object, *, active_source_identity: object, where: str
) -> Mapping[str, object]:
    environment = _exact(value, _EXECUTION_KEYS, where=where)
    if environment.get("schema") != "trellis.numeric_execution.v1":
        raise CheckpointContractError(f"{where}.schema differs")
    host = environment.get("physical_host")
    if not isinstance(host, str) or _HOST.fullmatch(host) is None:
        raise CheckpointContractError(f"{where}.physical_host is invalid")
    image = environment.get("container_image_digest")
    if not isinstance(image, str) or _IMAGE_DIGEST.fullmatch(image) is None:
        raise CheckpointContractError(
            f"{where}.container_image_digest is invalid"
        )
    commit = environment.get("repo_git_commit")
    if not isinstance(commit, str) or _GIT_COMMIT.fullmatch(commit) is None:
        raise CheckpointContractError(f"{where}.repo_git_commit is invalid")
    if environment.get("repo_tree_clean") is not True:
        raise CheckpointContractError(f"{where}.repo_tree_clean must be true")
    for field in ("python", "torch", "triton", "device"):
        item = environment.get(field)
        if not isinstance(item, str) or not item:
            raise CheckpointContractError(f"{where}.{field} is unavailable")
    source = _mapping(active_source_identity, where=f"{where}.active_sources")
    if source.get("repo_git_commit") != commit:
        raise CheckpointContractError(
            f"{where}.repo_git_commit differs from active source identity"
        )
    return environment
_FP8_SIDECAR_AMORTIZATION = (
    "the fixed fp8 lattice sidecar is a format-shared asset, charged "
    "once per (rung, physical identity), NOT per tensor"
)
_FP8_BOOK_PRICE_NOTE = (
    "TWO PRICES, BOTH HONEST. wire8 = 8 bits/element, sufficient because "
    "every element is a legal e4m3 code (asserted per arm); production = "
    "16 bits/element, the FP16 sidecar cb_layout.codebook_subtable_shapes "
    "documents and what a shipped book costs today. exact_bpw carries the "
    "PRODUCTION charge so the bias runs against this arm; read the verdict "
    "under both and call it only where they agree."
)
_FP8_PER_TENSOR_BOOK_NOTE = (
    "a learned book is per-(tensor, format) side info and IS charged here; "
    "the fixed fp8 lattice is a format-shared asset and is not"
)


def _mapping(value: object, *, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CheckpointContractError(f"{where} must be an object")
    return value


def _exact(value: object, keys: frozenset[str], *, where: str) -> Mapping[str, object]:
    result = _mapping(value, where=where)
    actual = set(result)
    if actual != keys:
        raise CheckpointContractError(
            f"{where} members differ: missing={sorted(keys - actual)}, "
            f"unknown={sorted(actual - keys)}"
        )
    return result


def _finite(value: object, *, where: str, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckpointContractError(f"{where} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise CheckpointContractError(f"{where} must be finite")
    if positive and result <= 0:
        raise CheckpointContractError(f"{where} must be positive")
    if nonnegative and result < 0:
        raise CheckpointContractError(f"{where} must be nonnegative")
    return result


def _integer(value: object, *, where: str, positive: bool = False,
             nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CheckpointContractError(f"{where} must be an integer")
    if positive and value <= 0:
        raise CheckpointContractError(f"{where} must be positive")
    if nonnegative and value < 0:
        raise CheckpointContractError(f"{where} must be nonnegative")
    return value


def _sha(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CheckpointContractError(f"{where} must be a lowercase SHA-256")
    return value


def _shape(value: object, *, where: str) -> list[int]:
    if (not isinstance(value, list) or len(value) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0
                   for item in value)):
        raise CheckpointContractError(f"{where} must be a positive rank-2 shape")
    return value


def _text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointContractError(f"{where} must be a nonempty string")
    return value


def _close(a: float, b: float, *, where: str, rel: float = 1e-9) -> None:
    if not math.isclose(a, b, rel_tol=rel, abs_tol=1e-12):
        raise CheckpointContractError(f"{where} is internally inconsistent")


def _json_digest(value: object, *, newline: bool, where: str) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointContractError(f"{where} is not canonical JSON") from exc
    if newline:
        encoded += "\n"
    return hashlib.sha256(encoded.encode()).hexdigest()


def _validate_e2_footprint(value: object, *, lane: str, rate: float,
                           shape: list[int], counts: Mapping[str, int], where: str,
                           production_payload: bool = False) -> None:
    expected = (_E2_FOOTPRINT_BASE_KEYS if production_payload else
                (_E2_TWO_TIER_FOOTPRINT_KEYS if lane == "tcq_two_tier"
                 else _E2_V1_FOOTPRINT_KEYS))
    footprint = _exact(value, expected, where=where)
    if footprint.get("family") != "TCQ_E2M1_R256" or footprint.get("grid") != "e2m1":
        raise CheckpointContractError(f"{where} family/grid differs")
    if footprint.get("format") != f"TCQ_E2M1_R{round(rate * 256)}":
        raise CheckpointContractError(f"{where}.format differs from rung")
    if footprint.get("shape") != shape:
        raise CheckpointContractError(f"{where}.shape differs")
    if footprint.get("body_rate_q256") != round(rate * 256):
        raise CheckpointContractError(f"{where}.body_rate_q256 differs")
    rows, columns = shape
    if columns % 256:
        raise CheckpointContractError(f"{where}.shape is not 256-column aligned")
    if footprint.get("schema") != (
        "prismaquant.trellis_tensor_payload.v1"
        if lane == "tcq_v1" or production_payload
        else "trellis.hull.tcq_two_tier_research_payload.v1"
    ):
        raise CheckpointContractError(f"{where}.schema differs")
    if (footprint.get("wire_schema") != "gridbook.trellis.wire.v1"
            or footprint.get("layout") != "tight_offsets"
            or footprint.get("superblock_weights") != 256
            or footprint.get("schedule_scope")
            != "tensor_input_column_shared_across_rows"
            or footprint.get("schedule_bits_per_code") != 4
            or footprint.get("wire_header_bytes") != 88
            or footprint.get("sidecar_header_bytes") != 0
            or footprint.get("expanded_weight_resident_bytes") != 0
            or footprint.get("producer_eligible") is not False):
        raise CheckpointContractError(f"{where} fixed wire contract differs")
    body_bits_per_row = sum(int(key) * count for key, count in counts.items())
    expected_body_bpw = body_bits_per_row / columns
    _close(expected_body_bpw, rate, where=f"{where}.body rate")
    if footprint.get("body_bits_per_row") != body_bits_per_row:
        raise CheckpointContractError(f"{where}.body_bits_per_row differs")
    unpadded = (body_bits_per_row + 7) // 8
    stride = ((unpadded + 15) // 16) * 16
    body_padding = rows * (stride - unpadded)
    body_bytes = rows * stride
    block_count = (columns + 255) // 256
    schedule_bytes = (columns * 4 + 7) // 8
    offset_bits = 32 if body_bits_per_row <= 0xFFFFFFFF else 64
    offset_bytes = (block_count + 1) * (offset_bits // 8)
    alphabet_by_rate = {
        key: 3 + (1 << (int(key) + 1))
        for key, count in counts.items() if count and int(key) < 4
    }
    alphabet_bytes = sum(alphabet_by_rate.values())
    derived_integers = {
        "block_count": block_count,
        "unpadded_body_bytes_per_row": unpadded,
        "body_row_stride_bytes": stride,
        "body_padding_bytes": body_padding,
        "body_bytes": body_bytes,
        "schedule_bytes": schedule_bytes,
        "block_offset_bits": offset_bits,
        "block_offset_bytes": offset_bytes,
        "alphabet_bytes": alphabet_bytes,
    }
    for field, expected_value in derived_integers.items():
        if footprint.get(field) != expected_value:
            raise CheckpointContractError(f"{where}.{field} accounting differs")
    if footprint.get("alphabet_bytes_by_rate") != alphabet_by_rate:
        raise CheckpointContractError(f"{where}.alphabet_bytes_by_rate differs")
    _close(_finite(footprint.get("body_bpw"), where=f"{where}.body_bpw", positive=True),
           expected_body_bpw, where=f"{where}.body_bpw")
    exact_bpw = _finite(footprint.get("exact_bpw"), where=f"{where}.exact_bpw", positive=True)
    if exact_bpw <= rate:
        raise CheckpointContractError(f"{where}.exact_bpw must include side information")
    side_information = 88 + schedule_bytes + offset_bytes + alphabet_bytes
    if footprint.get("side_information_bytes") != side_information:
        raise CheckpointContractError(f"{where}.side_information_bytes differs")
    if production_payload or lane == "tcq_v1":
        scale_bytes = rows * ((columns + 15) // 16)
        scale_contract = "group16_fp8_e4m3_0p5_bpw"
    else:
        scale_bytes = rows * (columns // 256) * 9
        scale_contract = "group16_two_tier_9B_per_superblock (RESEARCH)"
    if (footprint.get("scale_bytes") != scale_bytes
            or footprint.get("scale_contract") != scale_contract):
        raise CheckpointContractError(f"{where} scale accounting differs")
    total_bytes = body_bytes + scale_bytes + side_information
    if footprint.get("total_bytes") != total_bytes:
        raise CheckpointContractError(f"{where}.total_bytes accounting differs")
    _close(exact_bpw, 8.0 * total_bytes / (rows * columns),
           where=f"{where}.exact_bpw")
    identity = _sha(footprint.get("identity_sha256"), where=f"{where}.identity_sha256")
    if (not production_payload
            and footprint.get("scale_coding") !=
            ("two_tier" if lane == "tcq_two_tier" else "v1")):
        raise CheckpointContractError(f"{where}.scale_coding differs from lane")
    if lane == "tcq_two_tier":
        nested = footprint.get("production_payload_v1")
        _validate_e2_footprint(nested, lane="tcq_v1", rate=rate,
                               shape=shape, counts=counts,
                               where=f"{where}.production_payload_v1",
                               production_payload=True)
        nested_mapping = _mapping(nested, where=f"{where}.production_payload_v1")
        if (footprint.get("scale_bytes_v1_production")
                != nested_mapping.get("scale_bytes")
                or footprint.get("scale_bytes_per_superblock") != 9
                or identity != nested_mapping.get("identity_sha256")):
            raise CheckpointContractError(f"{where} research repricing differs")
        _close(_finite(footprint.get("scale_bpw"), where=f"{where}.scale_bpw"),
               scale_bytes * 8.0 / (rows * columns), where=f"{where}.scale_bpw")
        allowed_differences = {
            "schema", "scale_bytes", "scale_contract", "total_bytes",
            "exact_bpw", "non_shipping_research", "scale_coding",
            "scale_bytes_v1_production", "scale_bytes_per_superblock",
            "scale_bpw", "production_payload_v1", "research_pricing_note",
        }
        for field in _E2_FOOTPRINT_BASE_KEYS - allowed_differences:
            if footprint.get(field) != nested_mapping.get(field):
                raise CheckpointContractError(
                    f"{where}.{field} changed during research repricing"
                )
    else:
        production_body = {
            key: footprint[key]
            for key in _E2_FOOTPRINT_BASE_KEYS if key != "identity_sha256"
        }
        if identity != _json_digest(
            production_body, newline=False, where=f"{where} identity"
        ):
            raise CheckpointContractError(f"{where}.identity_sha256 differs")


def _validate_e2_schedule(value: object, *, rate: float, columns: int,
                          where: str) -> dict[str, int]:
    schedule = _exact(value, _E2_SCHEDULE_KEYS, where=where)
    _close(_finite(schedule.get("target_rate"), where=f"{where}.target_rate"),
           rate, where=f"{where}.target_rate")
    _close(_finite(schedule.get("achieved_rate"), where=f"{where}.achieved_rate"),
           rate, where=f"{where}.achieved_rate")
    if schedule.get("maximum_rate") != 4:
        raise CheckpointContractError(f"{where}.maximum_rate differs")
    _sha(schedule.get("schedule_sha256"), where=f"{where}.schedule_sha256")
    counts = _exact(schedule.get("counts"), frozenset({"1", "2", "3", "4"}),
                    where=f"{where}.counts")
    parsed = {key: _integer(value, where=f"{where}.counts.{key}", nonnegative=True)
              for key, value in counts.items()}
    if sum(parsed.values()) != columns:
        raise CheckpointContractError(f"{where}.counts do not cover columns")
    achieved_from_counts = sum(int(key) * count for key, count in parsed.items()) / columns
    _close(achieved_from_counts, rate, where=f"{where}.counts achieved rate")
    for field in ("tailbite_guard_fixups", "stage4_guard_fixups",
                  "minimum_trellis_steps", "body_bits_per_block_min",
                  "body_bits_per_block_max", "transitions_per_block_max"):
        _integer(schedule.get(field), where=f"{where}.{field}", nonnegative=True)
    for field in ("body_bits_per_block_std", "transitions_per_block_mean"):
        _finite(schedule.get(field), where=f"{where}.{field}", nonnegative=True)
    if schedule.get("invert") is not False or schedule.get("fixed_quota_per_256") is not False:
        raise CheckpointContractError(f"{where} schedule mode differs")
    if schedule.get("tailbite_guard_fixups") != schedule.get("stage4_guard_fixups"):
        raise CheckpointContractError(f"{where} guard fixup counts differ")
    block_min = int(schedule["body_bits_per_block_min"])
    block_max = int(schedule["body_bits_per_block_max"])
    block_std = float(schedule["body_bits_per_block_std"])
    target_per_block = rate * 256
    if not (256 <= block_min <= target_per_block <= block_max <= 1024):
        raise CheckpointContractError(f"{where} block-bit bounds differ")
    if block_min == block_max and block_std != 0.0:
        raise CheckpointContractError(f"{where}.body_bits_per_block_std differs")
    transition_mean = float(schedule["transitions_per_block_mean"])
    transition_max = int(schedule["transitions_per_block_max"])
    if not (0 <= transition_mean <= transition_max <= 255):
        raise CheckpointContractError(f"{where} transition bounds differ")
    if transition_max == 0 and transition_mean != 0.0:
        raise CheckpointContractError(f"{where}.transitions_per_block_mean differs")
    minimum_steps = int(schedule["minimum_trellis_steps"])
    if not 0 <= minimum_steps <= 256:
        raise CheckpointContractError(f"{where}.minimum_trellis_steps differs")
    if parsed["4"] == 0 and minimum_steps != 256:
        raise CheckpointContractError(f"{where}.minimum_trellis_steps differs")
    return parsed


def _validate_e2_subset(
    value: object, *, counts: Mapping[str, int], weighted_energy: float,
    weighted_sse: float, where: str,
) -> None:
    subset = _mapping(value, where=where)
    expected_classes = {key for key, count in counts.items() if count}
    if set(subset) != expected_classes:
        raise CheckpointContractError(f"{where} rate classes differ from schedule")
    subset_energy = 0.0
    subset_trellis = 0.0
    for rate_key, raw in subset.items():
        item = _exact(raw, _E2_SUBSET_KEYS, where=f"{where}.{rate_key}")
        rate = int(rate_key)
        if item.get("bits_per_weight_here") != rate or item.get("nvfp4_bits_per_weight") != 4:
            raise CheckpointContractError(f"{where}.{rate_key} rate identity differs")
        if item.get("columns") != counts[rate_key]:
            raise CheckpointContractError(f"{where}.{rate_key}.columns differs")
        energy = _finite(item.get("energy"), where=f"{where}.{rate_key}.energy", positive=True)
        nvfp4_wsse = _finite(item.get("nvfp4_wsse"), where=f"{where}.{rate_key}.nvfp4_wsse", positive=True)
        trellis_wsse = _finite(item.get("trellis_wsse"), where=f"{where}.{rate_key}.trellis_wsse", positive=True)
        _close(_finite(item.get("nvfp4_db"), where=f"{where}.{rate_key}.nvfp4_db"),
               10.0 * math.log10(energy / nvfp4_wsse),
               where=f"{where}.{rate_key}.nvfp4_db", rel=1e-7)
        _close(_finite(item.get("trellis_db"), where=f"{where}.{rate_key}.trellis_db"),
               10.0 * math.log10(energy / trellis_wsse),
               where=f"{where}.{rate_key}.trellis_db", rel=1e-7)
        _close(_finite(item.get("trellis_minus_nvfp4_db"), where=f"{where}.{rate_key}.trellis_minus_nvfp4_db"),
               10.0 * math.log10(nvfp4_wsse / trellis_wsse),
               where=f"{where}.{rate_key}.trellis_minus_nvfp4_db", rel=1e-7)
        subset_energy += energy
        subset_trellis += trellis_wsse
        expected_levels = (1 << rate) if rate < 4 else 15
        for scope in ("scalar_subgrid_oracle", "scalar_subgrid_shared"):
            scalar = _exact(item.get(scope), _E2_SCALAR_KEYS,
                            where=f"{where}.{rate_key}.{scope}")
            if scalar.get("n_levels") != expected_levels:
                raise CheckpointContractError(f"{where}.{rate_key}.{scope}.n_levels differs")
            levels = scalar.get("levels")
            if not isinstance(levels, list) or len(levels) != expected_levels:
                raise CheckpointContractError(f"{where}.{rate_key}.{scope}.levels differs")
            for index, level in enumerate(levels):
                _finite(level, where=f"{where}.{rate_key}.{scope}.levels[{index}]")
            if len(set(float(level) for level in levels)) != expected_levels:
                raise CheckpointContractError(f"{where}.{rate_key}.{scope}.levels are not distinct")
            if (any(float(level) not in _E2M1_LEVELS for level in levels)
                    or [float(level) for level in levels]
                    != sorted(float(level) for level in levels)):
                raise CheckpointContractError(
                    f"{where}.{rate_key}.{scope}.levels are not an ordered "
                    "legal E2M1 subset"
                )
            if rate == 4 and set(float(level) for level in levels) != _E2M1_LEVELS:
                raise CheckpointContractError(
                    f"{where}.{rate_key}.{scope}.levels do not cover E2M1"
                )
            scalar_wsse = _finite(scalar.get("wsse"), where=f"{where}.{rate_key}.{scope}.wsse", positive=True)
            _close(_finite(scalar.get("db"), where=f"{where}.{rate_key}.{scope}.db"),
                   10.0 * math.log10(energy / scalar_wsse),
                   where=f"{where}.{rate_key}.{scope}.db", rel=1e-7)
            _close(_finite(scalar.get("coding_gain_db"), where=f"{where}.{rate_key}.{scope}.coding_gain_db"),
                   10.0 * math.log10(scalar_wsse / trellis_wsse),
                   where=f"{where}.{rate_key}.{scope}.coding_gain_db", rel=1e-7)
            if scalar.get("subset_fit_scope") != _E2_SUBSET_SCOPES[scope]:
                raise CheckpointContractError(
                    f"{where}.{rate_key}.{scope}.subset_fit_scope differs"
                )
    _close(subset_energy, weighted_energy, where=f"{where} energy sum", rel=1e-7)
    _close(subset_trellis, weighted_sse, where=f"{where} trellis SSE sum", rel=1e-7)


def _validate_e2_arm(value: object, *, lane: str, rate: float,
                     shape: list[int], weighted_energy: float,
                     plain_energy: float, where: str) -> None:
    arm = _exact(value, _E2_ARM_KEYS, where=where)
    if arm.get("arm") != lane:
        raise CheckpointContractError(f"{where}.arm differs from key")
    _close(_finite(arm.get("rung"), where=f"{where}.rung"), rate,
           where=f"{where}.rung")
    _finite(arm.get("encode_seconds"), where=f"{where}.encode_seconds", positive=True)
    for domain in ("plain", "weighted"):
        sse = _finite(arm.get(f"{domain}_sse"), where=f"{where}.{domain}_sse", positive=True)
        nsse = _finite(arm.get(f"{domain}_nsse"), where=f"{where}.{domain}_nsse", positive=True)
        snr = _finite(arm.get(f"{domain}_snr_db"), where=f"{where}.{domain}_snr_db")
        energy = weighted_energy if domain == "weighted" else plain_energy
        _close(nsse, sse / energy, where=f"{where}.{domain}_nsse", rel=1e-7)
        _close(snr, -10.0 * math.log10(nsse), where=f"{where}.{domain}_snr_db", rel=1e-7)
        del sse
    expected_reproduction = (
        lane == "tcq_v1" and rate in _E2_REPRODUCTION_RATES
    )
    if arm.get("reproduces_stage6") is not expected_reproduction:
        raise CheckpointContractError(
            f"{where}.reproduces_stage6 differs from the producer"
        )
    counts = _validate_e2_schedule(arm.get("schedule"), rate=rate,
                                   columns=shape[1], where=f"{where}.schedule")
    _validate_e2_footprint(arm.get("footprint"), lane=lane, rate=rate,
                           shape=shape, counts=counts,
                           where=f"{where}.footprint")
    _validate_e2_subset(arm.get("subset_split"), counts=counts,
                        weighted_energy=weighted_energy,
                        weighted_sse=float(arm["weighted_sse"]),
                        where=f"{where}.subset_split")


def validate_e2_published_control_arm(
    value: object, *, key: str, shape: Sequence[int],
    weighted_energy: object, plain_energy: object,
) -> None:
    """Validate a legacy published control before any campaign GPU work."""

    if not isinstance(key, str) or "@" not in key:
        raise CheckpointContractError("published control key is invalid")
    lane, rate_text = key.rsplit("@", 1)
    if lane not in _E2_LANES:
        raise CheckpointContractError("published control lane differs")
    try:
        rate = float(rate_text)
    except ValueError as exc:
        raise CheckpointContractError("published control rate is invalid") from exc
    logical_shape = _shape(list(shape), where="published control shape")
    mapping = _mapping(value, where=f"published control {key}")
    keys = frozenset(mapping)
    if keys not in {_E2_PUBLISHED_ARM_KEYS, _E2_PUBLISHED_BF16_ARM_KEYS}:
        raise CheckpointContractError(
            f"published control {key} members differ"
        )
    arm = mapping
    if arm.get("arm") != lane:
        raise CheckpointContractError(f"published control {key}.arm differs")
    _close(
        _finite(arm.get("rung"), where=f"published control {key}.rung"),
        rate,
        where=f"published control {key}.rung",
    )
    _finite(
        arm.get("encode_seconds"),
        where=f"published control {key}.encode_seconds",
        positive=True,
    )
    energies = {
        "weighted": _finite(
            weighted_energy, where="published control weighted_energy",
            positive=True,
        ),
        "plain": _finite(
            plain_energy, where="published control plain_energy", positive=True,
        ),
    }
    for domain, energy in energies.items():
        sse = _finite(
            arm.get(f"{domain}_sse"),
            where=f"published control {key}.{domain}_sse",
            positive=True,
        )
        nsse = _finite(
            arm.get(f"{domain}_nsse"),
            where=f"published control {key}.{domain}_nsse",
            positive=True,
        )
        _close(
            nsse, sse / energy,
            where=f"published control {key}.{domain}_nsse", rel=1e-7,
        )
        snr_key = f"{domain}_snr_db"
        if snr_key in arm:
            _close(
                _finite(
                    arm.get(snr_key),
                    where=f"published control {key}.{snr_key}",
                ),
                -10.0 * math.log10(nsse),
                where=f"published control {key}.{snr_key}", rel=1e-7,
            )
    expected_reproduction = (
        lane == "tcq_v1" and rate in _E2_REPRODUCTION_RATES
    )
    if arm.get("reproduces_stage6") is not expected_reproduction:
        raise CheckpointContractError(
            f"published control {key}.reproduces_stage6 differs"
        )
    counts = _validate_e2_schedule(
        arm.get("schedule"), rate=rate, columns=logical_shape[1],
        where=f"published control {key}.schedule",
    )
    _validate_e2_footprint(
        arm.get("footprint"), lane=lane, rate=rate, shape=logical_shape,
        counts=counts, where=f"published control {key}.footprint",
    )


def validate_e2m1_checkpoint(
    document: object, *, current_receipt: Mapping[str, object],
    expected_tensors: Mapping[str, Mapping[str, object]],
    expected_controls: Mapping[str, Mapping[str, object]],
    require_partial: bool = True,
) -> None:
    root = _exact(document, _E2_ROOT_KEYS, where="checkpoint")
    digest = _sha(root.get("checkpoint_sha256"), where="checkpoint.checkpoint_sha256")
    body = {key: root[key] for key in ("receipt", "per_tensor")}
    if digest != _json_digest(body, newline=True, where="checkpoint"):
        raise CheckpointContractError("checkpoint self-digest differs")
    receipt = _exact(
        root.get("receipt"),
        _E2_RECEIPT_KEYS if require_partial else _E2_FINAL_RECEIPT_KEYS,
        where="receipt",
    )
    if receipt.get("schema") != "trellis.e2m1_highrate.v3":
        raise CheckpointContractError("receipt schema differs")
    if receipt.get("partial") is not require_partial:
        raise CheckpointContractError(f"receipt.partial must be {require_partial}")
    started = _finite(receipt.get("started_at_unix_s"), where="receipt.started_at_unix_s")
    del started
    rates_raw = receipt.get("rate_plan")
    if not isinstance(rates_raw, list) or not rates_raw:
        raise CheckpointContractError("receipt.rate_plan must be nonempty")
    rates = [_finite(rate, where="receipt.rate_plan[]", positive=True) for rate in rates_raw]
    if len(set(rates)) != len(rates):
        raise CheckpointContractError("receipt.rate_plan contains duplicates")
    if receipt.get("mathematical_q256_bounds") != [256, 1016]:
        raise CheckpointContractError("receipt mathematical q256 bounds differ")
    control_rungs = receipt.get("control_rungs")
    if (not isinstance(control_rungs, list)
            or any(not isinstance(key, str) or not key for key in control_rungs)
            or len(set(control_rungs)) != len(control_rungs)):
        raise CheckpointContractError("receipt.control_rungs differs")
    _sha(receipt.get("publication_identity_sha256"), where="receipt.publication_identity_sha256")
    _validate_execution_environment(
        receipt.get("environment"),
        active_source_identity=receipt.get("active_source_identity"),
        where="receipt.environment",
    )
    comparable_saved = {key: value for key, value in receipt.items()
                        if key not in {"started_at_unix_s", "partial", "tensors_done"}}
    comparable_current = {key: value for key, value in current_receipt.items()
                          if key not in {"started_at_unix_s", "partial", "tensors_done"}}
    if comparable_saved != comparable_current:
        raise CheckpointContractError("receipt identity differs")
    per_tensor = _mapping(root.get("per_tensor"), where="per_tensor")
    names = list(expected_tensors)
    if set(expected_controls) != set(expected_tensors):
        raise CheckpointContractError("expected control tensor domain differs")
    if receipt.get("tensors_done") != len(per_tensor):
        raise CheckpointContractError("receipt tensor count differs")
    if not require_partial:
        if len(per_tensor) != len(names):
            raise CheckpointContractError("final checkpoint does not cover every tensor")
        _finite(receipt.get("completed_at_unix_s"), where="receipt.completed_at_unix_s")
        if receipt.get("status") != "ok":
            raise CheckpointContractError("receipt.status must be ok")
        if float(receipt["completed_at_unix_s"]) < float(receipt["started_at_unix_s"]):
            raise CheckpointContractError("receipt completion precedes start")
    if list(per_tensor) != list(names[:len(per_tensor)]):
        raise CheckpointContractError("per_tensor is not the ordered tensor prefix")
    expected_domain = {(lane, rate) for lane in _E2_LANES for rate in rates}
    for name, raw_cell in per_tensor.items():
        cell = _exact(raw_cell, _E2_CELL_KEYS, where=f"per_tensor.{name}")
        shape = _shape(cell.get("shape"), where=f"per_tensor.{name}.shape")
        expected_tensor = _exact(
            expected_tensors[name], _E2_EXPECTED_TENSOR_KEYS,
            where=f"expected_tensors.{name}",
        )
        expected_shape = _shape(
            expected_tensor.get("shape"), where=f"expected_tensors.{name}.shape"
        )
        if (shape != expected_shape
                or cell.get("population") != expected_tensor.get("population")):
            raise CheckpointContractError(
                f"per_tensor.{name} logical identity differs"
            )
        if cell.get("numel") != math.prod(shape):
            raise CheckpointContractError(f"per_tensor.{name}.numel differs")
        _text(cell.get("population"), where=f"per_tensor.{name}.population")
        weighted_energy = _finite(cell.get("weighted_energy"), where=f"per_tensor.{name}.weighted_energy", positive=True)
        plain_energy = _finite(cell.get("plain_energy"), where=f"per_tensor.{name}.plain_energy", positive=True)
        _sha(cell.get("two_tier_plane_sha256"), where=f"per_tensor.{name}.two_tier_plane_sha256")
        arms = _mapping(cell.get("arms"), where=f"per_tensor.{name}.arms")
        measured: set[tuple[str, float]] = set()
        for key, arm in arms.items():
            if not isinstance(key, str) or "@" not in key:
                raise CheckpointContractError(f"per_tensor.{name}.arms has invalid key")
            lane, rate_text = key.rsplit("@", 1)
            try:
                rate = float(rate_text)
            except ValueError as exc:
                raise CheckpointContractError(f"per_tensor.{name}.arms has invalid rate") from exc
            identity = (lane, rate)
            if identity not in expected_domain or identity in measured:
                raise CheckpointContractError(f"per_tensor.{name}.arms domain differs")
            measured.add(identity)
            _validate_e2_arm(arm, lane=lane, rate=rate, shape=shape,
                             weighted_energy=weighted_energy,
                             plain_energy=plain_energy,
                             where=f"per_tensor.{name}.arms.{key}")
        unreachable_raw = cell.get("unreachable_rungs")
        if not isinstance(unreachable_raw, list):
            raise CheckpointContractError(f"per_tensor.{name}.unreachable_rungs must be a list")
        unreachable: set[tuple[str, float]] = set()
        unreachable_reasons: dict[tuple[str, float], str] = {}
        for index, raw in enumerate(unreachable_raw):
            row = _exact(raw, _E2_UNREACHABLE_KEYS,
                         where=f"per_tensor.{name}.unreachable_rungs[{index}]")
            lane = row.get("lane")
            rate = _finite(row.get("rate"), where=f"per_tensor.{name}.unreachable_rungs[{index}].rate")
            identity = (lane, rate)
            if identity not in expected_domain or identity in unreachable:
                raise CheckpointContractError(f"per_tensor.{name} unreachable domain differs")
            _text(row.get("reason"), where=f"per_tensor.{name}.unreachable_rungs[{index}].reason")
            unreachable.add(identity)
            unreachable_reasons[identity] = str(row["reason"])
        if measured & unreachable or measured | unreachable != expected_domain:
            raise CheckpointContractError(f"per_tensor.{name} does not cover every expected arm exactly once")
        if not measured:
            raise CheckpointContractError(
                f"per_tensor.{name} cannot declare every arm unreachable"
            )
        unreachable_rates = {rate for _lane, rate in unreachable}
        if unreachable_rates:
            if unreachable_rates != {_E2_CEILING_RATE}:
                raise CheckpointContractError(
                    f"per_tensor.{name} declares a non-ceiling rung unreachable"
                )
            ceiling_lanes = {
                lane for lane, rate in unreachable if rate == _E2_CEILING_RATE
            }
            if ceiling_lanes != set(_E2_LANES):
                raise CheckpointContractError(
                    f"per_tensor.{name} has one-sided ceiling reachability"
                )
            if any(
                unreachable_reasons[(lane, _E2_CEILING_RATE)]
                != _E2_CEILING_REFUSAL
                for lane in _E2_LANES
            ):
                raise CheckpointContractError(
                    f"per_tensor.{name} has an unrecognized ceiling refusal"
                )
        control = _exact(cell.get("control"), _E2_CONTROL_KEYS,
                         where=f"per_tensor.{name}.control")
        if control.get("status") not in {"pass", "fail", "uncontrolled"}:
            raise CheckpointContractError(f"per_tensor.{name}.control.status differs")
        if not isinstance(control.get("footprint_equal"), bool):
            raise CheckpointContractError(f"per_tensor.{name}.control.footprint_equal must be boolean")
        if any(rung not in arms for rung in control_rungs):
            raise CheckpointContractError(
                f"per_tensor.{name} control rung was not measured"
            )
        checks = _mapping(control.get("checks"), where=f"per_tensor.{name}.control.checks")
        control_fields = (
            "weighted_sse", "weighted_nsse", "weighted_snr_db",
            "plain_sse", "plain_nsse",
        )
        expected_control_checks = {
            f"{rung}.{field}" for rung in control_rungs for field in control_fields
        }
        expected_control_rows = _mapping(
            expected_controls[name], where=f"expected_controls.{name}"
        )
        if set(expected_control_rows) != set(control_rungs):
            raise CheckpointContractError(
                f"expected_controls.{name} rung domain differs"
            )
        if set(checks) != expected_control_checks:
            raise CheckpointContractError(
                f"per_tensor.{name}.control.checks domain differs"
            )
        measured_checks = 0
        missing_checks = 0
        for key, raw in checks.items():
            _text(key, where=f"per_tensor.{name}.control.check key")
            check = _exact(raw, _E2_CONTROL_CHECK_KEYS,
                           where=f"per_tensor.{name}.control.checks.{key}")
            mine, published = check.get("mine"), check.get("published")
            if (mine is None) != (published is None):
                raise CheckpointContractError(f"per_tensor.{name}.control.checks.{key} null pairing differs")
            if mine is not None:
                mine_value = _finite(mine, where=f"per_tensor.{name}.control.checks.{key}.mine")
                published_value = _finite(published, where=f"per_tensor.{name}.control.checks.{key}.published")
                measured_checks += 1
            else:
                missing_checks += 1
            relative = _finite(
                check.get("rel"),
                where=f"per_tensor.{name}.control.checks.{key}.rel",
                nonnegative=True,
            )
            if mine is not None:
                control_rung, field = key.rsplit(".", 1)
                expected_control = _exact(
                    expected_control_rows[control_rung],
                    _E2_EXPECTED_CONTROL_KEYS,
                    where=f"expected_controls.{name}.{control_rung}",
                )
                expected_metrics = _exact(
                    expected_control.get("metrics"), frozenset(control_fields),
                    where=f"expected_controls.{name}.{control_rung}.metrics",
                )
                if mine_value != arms[control_rung][field]:
                    raise CheckpointContractError(
                        f"per_tensor.{name}.control.checks.{key}.mine differs "
                        "from measured arm"
                    )
                if published_value != expected_metrics[field]:
                    raise CheckpointContractError(
                        f"per_tensor.{name}.control.checks.{key}.published "
                        "differs from bound control"
                    )
                _close(
                    relative,
                    abs(mine_value - published_value)
                    / max(abs(published_value), 1e-300),
                    where=f"per_tensor.{name}.control.checks.{key}.rel",
                    rel=1e-7,
                )
        worst = control.get("worst_relative")
        if control_rungs:
            if control.get("status") not in {"pass", "fail"}:
                raise CheckpointContractError(
                    f"per_tensor.{name} declared control status differs"
                )
            if measured_checks != len(expected_control_checks) or missing_checks:
                raise CheckpointContractError(
                    f"per_tensor.{name} declared controls are not measured"
                )
            worst_value = _finite(worst, where=f"per_tensor.{name}.control.worst_relative", nonnegative=True)
            measured_relatives = [
                float(check["rel"]) for check in checks.values()
                if check["mine"] is not None
            ]
            _close(worst_value, max(measured_relatives),
                   where=f"per_tensor.{name}.control.worst_relative")
            tolerance = _finite(receipt.get("control_rtol"),
                                where="receipt.control_rtol", nonnegative=True)
            expected_footprint_equal = all(
                arms[rung]["footprint"][field]
                == _exact(
                    expected_control_rows[rung], _E2_EXPECTED_CONTROL_KEYS,
                    where=f"expected_controls.{name}.{rung}",
                )["footprint"][field]
                for rung in control_rungs
                for field in ("total_bytes", "body_rate_q256")
            )
            for rung in control_rungs:
                _exact(
                    expected_control_rows[rung]["footprint"],
                    _E2_EXPECTED_CONTROL_FOOTPRINT_KEYS,
                    where=f"expected_controls.{name}.{rung}.footprint",
                )
            if control.get("footprint_equal") is not expected_footprint_equal:
                raise CheckpointContractError(
                    f"per_tensor.{name}.control.footprint_equal differs"
                )
            should_pass = (not missing_checks and expected_footprint_equal
                           and worst_value <= tolerance)
            if (control.get("status") == "pass") != should_pass:
                raise CheckpointContractError(f"per_tensor.{name}.control.status is inconsistent")
        elif (measured_checks or missing_checks or worst is not None
              or control.get("status") != "uncontrolled"
              or control.get("footprint_equal") is not False):
            raise CheckpointContractError(
                f"per_tensor.{name}.control empty-check status differs"
            )
    if not require_partial:
        expected_verdict = {
            name: cell["control"]["status"] for name, cell in per_tensor.items()
        }
        if receipt.get("control_verdict") != expected_verdict:
            raise CheckpointContractError("receipt.control_verdict differs from cells")
        expected_populations = {
            population: sum(
                cell["population"] == population for cell in per_tensor.values()
            )
            for population in sorted({
                cell["population"] for cell in per_tensor.values()
            })
        }
        if receipt.get("population_counts") != expected_populations:
            raise CheckpointContractError("receipt.population_counts differs from cells")


def _validate_fp8_settings(settings: object, *, expected: Mapping[str, object]) -> Mapping[str, object]:
    value = _exact(settings, _FP8_SETTINGS_KEYS, where="settings")
    if value != expected:
        raise CheckpointContractError("settings identity differs")
    _validate_execution_environment(
        value.get("environment"),
        active_source_identity=value.get("active_source_identity"),
        where="settings.environment",
    )
    for field in ("corpus_manifest_sha256", "corpus_file_sha256", "importance_value_sha256", "identity_sha256"):
        _sha(value.get(field), where=f"settings.{field}")
    if value.get("schema") != "trellis.glm_fp8_learned_balanced.v2":
        raise CheckpointContractError("settings.schema differs")
    if value.get("rungs") != [32, 40, 48] or value.get("encode_tier") != "balanced":
        raise CheckpointContractError("settings rung/tier contract differs")
    identity_payload = {key: item for key, item in value.items()
                        if key != "identity_sha256"}
    expected_identity = hashlib.sha256(json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if value.get("identity_sha256") != expected_identity:
        raise CheckpointContractError("settings.identity_sha256 differs")
    return value


def _validate_fp8_footprint(value: object, *, rung: int, learned: bool,
                            shape: list[int],
                            where: str) -> Mapping[str, object]:
    keys = _FP8_LEARNED_FOOTPRINT_KEYS if learned else _FP8_FIXED_FOOTPRINT_KEYS
    footprint = _exact(value, keys, where=where)
    if footprint.get("schema") != "trellis.fp8_ladder.fp8_cb_accounting.v1":
        raise CheckpointContractError(f"{where}.schema differs")
    if footprint.get("format") != f"FP8_CB_K{rung}":
        raise CheckpointContractError(f"{where}.format differs from rung")
    if footprint.get("codebook") != ("per_tensor_weighted_lloyd" if learned else "fixed_lattice"):
        raise CheckpointContractError(f"{where}.codebook differs from arm")
    rows, columns = shape
    numel = rows * columns
    if numel % 256:
        raise CheckpointContractError(f"{where} shape is not superblock aligned")
    superblocks = numel // 256
    index_bytes = rung * 4
    row_scale_bytes = rows * 4
    body_bits = 8 * index_bytes * superblocks
    scale_bits = 8 * row_scale_bytes
    _close(_finite(footprint.get("body_bpw"), where=f"{where}.body_bpw", positive=True),
           body_bits / numel, where=f"{where}.body_bpw")
    exact_bpw = _finite(footprint.get("exact_bpw"), where=f"{where}.exact_bpw", positive=True)
    if exact_bpw <= rung / 8.0:
        raise CheckpointContractError(f"{where}.exact_bpw must include side information")
    for field, expected_value in {
        "superblocks": superblocks,
        "type_size_bytes_per_superblock": index_bytes,
        "index_bytes_per_superblock": index_bytes,
        "scale_bytes_per_superblock": 0,
        "row_scale_bytes": row_scale_bytes,
        "body_bits": body_bits,
        "scale_bits": scale_bits,
    }.items():
        if footprint.get(field) != expected_value:
            raise CheckpointContractError(f"{where}.{field} accounting differs")
    _close(_finite(footprint.get("scale_bpw"), where=f"{where}.scale_bpw"),
           scale_bits / numel, where=f"{where}.scale_bpw")
    if footprint.get("scale_coding") != "v1" or footprint.get("scale_contract") != "per_output_row_fp32":
        raise CheckpointContractError(f"{where} scale contract differs")
    if (footprint.get("backed_on_sm120") is not True
            or footprint.get("sidecar_amortization")
            != _FP8_SIDECAR_AMORTIZATION):
        raise CheckpointContractError(f"{where} fixed format claims differ")
    book_bits = 0
    book_bits_wire8 = 0
    if learned:
        table_rows = 1 << (rung // 4)
        expected_shapes = [[table_rows, 2]] * 4
        elements = table_rows * 8
        book_bits = elements * 16
        book_bits_wire8 = elements * 8
        expected_learned = {
            "fixed_lattice_is_format_shared": False,
            "learned_book_subtable_shapes": expected_shapes,
            "learned_book_n_sub": 4,
            "learned_book_elements": elements,
            "learned_book_bits_per_element": 16,
            "codebook_side_bits": book_bits,
            "codebook_side_bits_wire8": book_bits_wire8,
            "fp4_level_bits_charge_would_have_been_bits": elements * 4,
        }
        for field, expected_value in expected_learned.items():
            if footprint.get(field) != expected_value:
                raise CheckpointContractError(f"{where}.{field} accounting differs")
        _close(_finite(footprint.get("codebook_side_bpw"), where=f"{where}.codebook_side_bpw"),
               book_bits / numel, where=f"{where}.codebook_side_bpw")
        _close(_finite(footprint.get("codebook_side_bpw_wire8"), where=f"{where}.codebook_side_bpw_wire8"),
               book_bits_wire8 / numel, where=f"{where}.codebook_side_bpw_wire8")
        _close(_finite(footprint.get("exact_bpw_book_wire8"), where=f"{where}.exact_bpw_book_wire8"),
               (body_bits + scale_bits + book_bits_wire8) / numel,
               where=f"{where}.exact_bpw_book_wire8")
        if (footprint.get("book_price_bracket_note") != _FP8_BOOK_PRICE_NOTE
                or footprint.get("learned_book_is_per_tensor")
                != _FP8_PER_TENSOR_BOOK_NOTE):
            raise CheckpointContractError(
                f"{where} learned-book accounting notes differ"
            )
    total_bits = body_bits + scale_bits + book_bits
    if (footprint.get("total_bits") != total_bits
            or footprint.get("total_bytes") != total_bits // 8):
        raise CheckpointContractError(f"{where} total accounting differs")
    _close(exact_bpw, total_bits / numel, where=f"{where}.exact_bpw")
    return footprint


def _validate_fp8_book(value: object, *, footprint: Mapping[str, object], where: str) -> None:
    book = _exact(value, _FP8_BOOK_KEYS, where=where)
    tables = book.get("tables")
    if not isinstance(tables, list) or len(tables) != 4:
        raise CheckpointContractError(f"{where}.tables must contain exactly four subtables")
    elements = 0
    shapes = []
    for index, raw in enumerate(tables):
        table = _exact(raw, _FP8_TABLE_KEYS, where=f"{where}.tables[{index}]")
        shape = _shape(table.get("shape"), where=f"{where}.tables[{index}].shape")
        table_elements = math.prod(shape)
        distinct = _integer(table.get("distinct_levels"), where=f"{where}.tables[{index}].distinct_levels", positive=True)
        if distinct > table_elements:
            raise CheckpointContractError(f"{where}.tables[{index}].distinct_levels exceeds table")
        _finite(table.get("amax"), where=f"{where}.tables[{index}].amax", positive=True)
        _sha(table.get("sha256"), where=f"{where}.tables[{index}].sha256")
        elements += table_elements
        shapes.append(shape)
    if book.get("elements") != elements:
        raise CheckpointContractError(f"{where}.elements differs")
    if (footprint.get("learned_book_elements") != elements
            or footprint.get("learned_book_n_sub") != len(tables)
            or footprint.get("learned_book_subtable_shapes") != shapes):
        raise CheckpointContractError(f"{where} differs from footprint")


def _validate_fp8_arm(value: object, *, rung: int, learned: bool,
                      energy: float, shape: list[int], where: str) -> None:
    keys = _FP8_LEARNED_ARM_KEYS if learned else _FP8_FIXED_ARM_KEYS
    arm = _exact(value, keys, where=where)
    if arm.get("encode_tier") != "balanced":
        raise CheckpointContractError(f"{where}.encode_tier differs")
    _finite(arm.get("encode_seconds_observation_not_perf_claim"),
            where=f"{where}.encode_seconds", positive=True)
    error = _finite(arm.get("weighted_sse"), where=f"{where}.weighted_sse", positive=True)
    nsse = _finite(arm.get("weighted_nsse"), where=f"{where}.weighted_nsse", positive=True)
    snr = _finite(arm.get("weighted_snr_db"), where=f"{where}.weighted_snr_db")
    _close(nsse, error / energy, where=f"{where}.weighted_nsse", rel=1e-7)
    _close(snr, -10.0 * math.log10(nsse), where=f"{where}.weighted_snr_db", rel=1e-7)
    _sha(arm.get("reconstruction_sha256"), where=f"{where}.reconstruction_sha256")
    footprint = _validate_fp8_footprint(arm.get("footprint"), rung=rung,
                                        shape=shape,
                                        learned=learned, where=f"{where}.footprint")
    if learned:
        _validate_fp8_book(arm.get("learned_book"), footprint=footprint,
                           where=f"{where}.learned_book")


def validate_fp8_checkpoint(document: object, *, settings: Mapping[str, object],
                            entries: Sequence[object],
                            require_partial: bool = True,
                            generated_hashes: Mapping[str, Mapping[str, str]] | None = None,
                            generated_books: Mapping[str, Mapping[str, object]] | None = None,
                            ) -> None:
    root = _exact(
        document,
        _FP8_ROOT_KEYS if require_partial else _FP8_FINAL_ROOT_KEYS,
        where="checkpoint",
    )
    digest = _sha(root.get("checkpoint_sha256"), where="checkpoint.checkpoint_sha256")
    body = {key: value for key, value in root.items()
            if key != "checkpoint_sha256"}
    if digest != _json_digest(body, newline=False, where="checkpoint"):
        raise CheckpointContractError("checkpoint self-digest differs")
    if root.get("schema") != "trellis.glm_fp8_learned_balanced.v2":
        raise CheckpointContractError("checkpoint schema differs")
    _validate_fp8_settings(root.get("settings"), expected=settings)
    if root.get("partial") is not require_partial:
        raise CheckpointContractError(f"checkpoint.partial must be {require_partial}")
    _finite(root.get("started_at_unix_s"), where="checkpoint.started_at_unix_s")
    per_tensor = _mapping(root.get("per_tensor"), where="checkpoint.per_tensor")
    if root.get("tensors_done") != len(per_tensor):
        raise CheckpointContractError("checkpoint tensor count differs")
    if not require_partial:
        if len(per_tensor) != len(entries):
            raise CheckpointContractError("final checkpoint does not cover every tensor")
        _finite(root.get("completed_at_unix_s"), where="checkpoint.completed_at_unix_s")
        if root.get("status") != "measurement_complete_no_serving_verdict":
            raise CheckpointContractError("checkpoint.status differs")
        if root.get("performance_gate") != FP8_PERFORMANCE_GATE:
            raise CheckpointContractError("checkpoint.performance_gate differs")
        if float(root["completed_at_unix_s"]) < float(root["started_at_unix_s"]):
            raise CheckpointContractError("checkpoint completion precedes start")
    names = [entry.name for entry in entries]
    if generated_hashes is None or generated_books is None:
        raise CheckpointContractError(
            "generated FP8 evidence is required for every validation"
        )
    evidence_hashes = _mapping(
        generated_hashes, where="generated_hashes"
    )
    evidence_books = _mapping(
        generated_books, where="generated_books"
    )
    if set(evidence_hashes) != set(per_tensor):
        raise CheckpointContractError(
            "generated reconstruction tensor domain differs"
        )
    if set(evidence_books) != set(per_tensor):
        raise CheckpointContractError("generated book tensor domain differs")
    expected_population_counts = {
        population: sum(entry.population == population for entry in entries)
        for population in ("dense", "routed")
    }
    if settings.get("population_counts") != expected_population_counts:
        raise CheckpointContractError("settings.population_counts differs from entries")
    if list(per_tensor) != names[:len(per_tensor)]:
        raise CheckpointContractError("checkpoint is not the ordered tensor prefix")
    expected_arms = {f"{family}@{rung}" for rung in (32, 40, 48)
                     for family in ("fp8_cb", "fp8_cb_learned")}
    by_name = {entry.name: entry for entry in entries}
    for name, raw in per_tensor.items():
        entry = by_name[name]
        cell = _exact(raw, _FP8_CELL_KEYS, where=f"per_tensor.{name}")
        shape = _shape(cell.get("shape"), where=f"per_tensor.{name}.shape")
        if (cell.get("population") != entry.population
                or cell.get("source_weight_sha256") != entry.source_weight_sha256
                or cell.get("importance_sha256") != entry.importance_sha256
                or shape != list(entry.source_weight_shape)):
            raise CheckpointContractError(f"per_tensor.{name} identity differs")
        _sha(cell.get("source_weight_sha256"), where=f"per_tensor.{name}.source_weight_sha256")
        _sha(cell.get("importance_sha256"), where=f"per_tensor.{name}.importance_sha256")
        source = _exact(cell.get("importance_source"), _FP8_IMPORTANCE_KEYS,
                        where=f"per_tensor.{name}.importance_source")
        expected_source = {
            "qname": entry.importance_source_qname,
            "expert": entry.importance_source_expert,
            "denominator_name": entry.importance_denominator_name,
            "denominator": entry.importance_denominator,
        }
        if source != expected_source:
            raise CheckpointContractError(f"per_tensor.{name}.importance_source differs")
        energy = _finite(cell.get("weighted_energy"), where=f"per_tensor.{name}.weighted_energy", positive=True)
        arms = _mapping(cell.get("arms"), where=f"per_tensor.{name}.arms")
        if set(arms) != expected_arms:
            raise CheckpointContractError(f"per_tensor.{name}.arms domain differs")
        for rung in (32, 40, 48):
            _validate_fp8_arm(arms[f"fp8_cb@{rung}"], rung=rung, learned=False,
                              shape=shape,
                              energy=energy, where=f"per_tensor.{name}.arms.fp8_cb@{rung}")
            _validate_fp8_arm(arms[f"fp8_cb_learned@{rung}"], rung=rung, learned=True,
                              shape=shape,
                              energy=energy, where=f"per_tensor.{name}.arms.fp8_cb_learned@{rung}")
        expected_hashes = _mapping(
            evidence_hashes[name], where=f"generated_hashes.{name}"
        )
        if set(expected_hashes) != expected_arms:
            raise CheckpointContractError(
                f"per_tensor.{name} generated reconstruction hash domain differs"
            )
        for arm_name, expected_hash in expected_hashes.items():
            _sha(expected_hash, where=f"generated_hashes.{name}.{arm_name}")
            if arms[arm_name]["reconstruction_sha256"] != expected_hash:
                raise CheckpointContractError(
                    f"per_tensor.{name}.{arm_name} reconstruction hash differs "
                    "from generated object"
                )
        expected_books = _mapping(
            evidence_books[name], where=f"generated_books.{name}"
        )
        learned_domain = {
            f"fp8_cb_learned@{rung}" for rung in (32, 40, 48)
        }
        if set(expected_books) != learned_domain:
            raise CheckpointContractError(
                f"per_tensor.{name} generated book domain differs"
            )
        for arm_name, expected_book in expected_books.items():
            if arms[arm_name]["learned_book"] != expected_book:
                raise CheckpointContractError(
                    f"per_tensor.{name}.{arm_name} learned book differs "
                    "from generated tables"
                )
    if not require_partial:
        expected_summaries = fp8_population_summaries(per_tensor)
        if root.get("population_summaries") != expected_summaries:
            raise CheckpointContractError(
                "checkpoint.population_summaries differ from cells"
            )


def fp8_population_summaries(
    per_tensor: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Derive the only accepted final summaries from validated arm cells."""

    output: dict[str, object] = {}
    for population in ("dense", "routed"):
        names = sorted(
            name for name, cell in per_tensor.items()
            if cell.get("population") == population
        )
        if not names:
            continue
        rows = []
        for rung in (32, 40, 48):
            fixed_db = [float(per_tensor[name]["arms"][f"fp8_cb@{rung}"]["weighted_snr_db"])
                        for name in names]
            learned_db = [float(per_tensor[name]["arms"][f"fp8_cb_learned@{rung}"]["weighted_snr_db"])
                          for name in names]
            fixed_bpw = [float(per_tensor[name]["arms"][f"fp8_cb@{rung}"]["footprint"]["exact_bpw"])
                         for name in names]
            learned_bpw = [float(per_tensor[name]["arms"][f"fp8_cb_learned@{rung}"]["footprint"]["exact_bpw"])
                           for name in names]
            deltas = [learned - fixed for learned, fixed in zip(learned_db, fixed_db)]
            rows.append({
                "rung": rung, "tensors": len(names),
                "fixed_db_median": statistics.median(fixed_db),
                "learned_db_median": statistics.median(learned_db),
                "learned_minus_fixed_db_median": statistics.median(deltas),
                "learned_minus_fixed_db_min": min(deltas),
                "learned_minus_fixed_db_max": max(deltas),
                "learned_better": sum(delta > 0 for delta in deltas),
                "fixed_bpw_median": statistics.median(fixed_bpw),
                "learned_bpw_median": statistics.median(learned_bpw),
            })
        output[population] = {"tensors": len(names), "rows": rows}
    return output


def validate_fp8_replay_envelope(
    document: object, *, settings: Mapping[str, object],
    entries: Sequence[object],
) -> None:
    """Validate only an untrusted partial's envelope before GPU replay.

    Cell claims are deliberately not accepted here; the driver regenerates
    each prefix cell and compares it exactly before calling the full validator.
    """

    root = _exact(document, _FP8_ROOT_KEYS, where="checkpoint")
    digest = _sha(
        root.get("checkpoint_sha256"),
        where="checkpoint.checkpoint_sha256",
    )
    body = {
        key: value for key, value in root.items()
        if key != "checkpoint_sha256"
    }
    if digest != _json_digest(body, newline=False, where="checkpoint"):
        raise CheckpointContractError("checkpoint self-digest differs")
    if root.get("schema") != "trellis.glm_fp8_learned_balanced.v2":
        raise CheckpointContractError("checkpoint schema differs")
    _validate_fp8_settings(root.get("settings"), expected=settings)
    if root.get("partial") is not True:
        raise CheckpointContractError("checkpoint.partial must be True")
    _finite(root.get("started_at_unix_s"), where="checkpoint.started_at_unix_s")
    per_tensor = _mapping(root.get("per_tensor"), where="checkpoint.per_tensor")
    if root.get("tensors_done") != len(per_tensor):
        raise CheckpointContractError("checkpoint tensor count differs")
    names = [entry.name for entry in entries]
    if list(per_tensor) != names[:len(per_tensor)]:
        raise CheckpointContractError("checkpoint is not the ordered tensor prefix")


__all__ = [
    "CheckpointContractError", "FP8_PERFORMANCE_GATE",
    "fp8_population_summaries", "validate_e2m1_checkpoint",
    "validate_e2_published_control_arm", "validate_fp8_checkpoint",
    "validate_fp8_replay_envelope",
]
