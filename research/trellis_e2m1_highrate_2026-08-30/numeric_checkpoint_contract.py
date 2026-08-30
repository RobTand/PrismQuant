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
from typing import Mapping, Sequence


class CheckpointContractError(ValueError):
    pass


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
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
_E2_ARM_KEYS = frozenset({
    "arm", "encode_seconds", "footprint", "plain_nsse", "plain_snr_db",
    "plain_sse", "reproduces_stage6", "rung", "schedule", "subset_split",
    "weighted_nsse", "weighted_snr_db", "weighted_sse",
})
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
_E2_UNREACHABLE_KEYS = frozenset({"lane", "rate", "reason"})
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
    "aggregation_contract", "identity_sha256",
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


def _validate_e2_footprint(value: object, *, lane: str, rate: float,
                           shape: list[int], where: str,
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
    _close(_finite(footprint.get("body_bpw"), where=f"{where}.body_bpw", positive=True),
           rate, where=f"{where}.body_bpw")
    exact_bpw = _finite(footprint.get("exact_bpw"), where=f"{where}.exact_bpw", positive=True)
    if exact_bpw <= rate:
        raise CheckpointContractError(f"{where}.exact_bpw must include side information")
    _integer(footprint.get("total_bytes"), where=f"{where}.total_bytes", positive=True)
    _sha(footprint.get("identity_sha256"), where=f"{where}.identity_sha256")
    if (not production_payload
            and footprint.get("scale_coding") !=
            ("two_tier" if lane == "tcq_two_tier" else "v1")):
        raise CheckpointContractError(f"{where}.scale_coding differs from lane")
    if lane == "tcq_two_tier":
        nested = footprint.get("production_payload_v1")
        _validate_e2_footprint(nested, lane="tcq_v1", rate=rate,
                               shape=shape, where=f"{where}.production_payload_v1",
                               production_payload=True)


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
                  "minimum_trellis_steps"):
        _integer(schedule.get(field), where=f"{where}.{field}", nonnegative=True)
    for field in ("body_bits_per_block_min", "body_bits_per_block_max",
                  "body_bits_per_block_std", "transitions_per_block_mean",
                  "transitions_per_block_max"):
        _finite(schedule.get(field), where=f"{where}.{field}", nonnegative=True)
    for field in ("invert", "fixed_quota_per_256"):
        if not isinstance(schedule.get(field), bool):
            raise CheckpointContractError(f"{where}.{field} must be boolean")
    return parsed


def _validate_e2_subset(value: object, *, counts: Mapping[str, int], where: str) -> None:
    subset = _mapping(value, where=where)
    expected_classes = {key for key, count in counts.items() if count}
    if set(subset) != expected_classes:
        raise CheckpointContractError(f"{where} rate classes differ from schedule")
    for rate_key, raw in subset.items():
        item = _exact(raw, _E2_SUBSET_KEYS, where=f"{where}.{rate_key}")
        rate = int(rate_key)
        if item.get("bits_per_weight_here") != rate or item.get("nvfp4_bits_per_weight") != 4:
            raise CheckpointContractError(f"{where}.{rate_key} rate identity differs")
        if item.get("columns") != counts[rate_key]:
            raise CheckpointContractError(f"{where}.{rate_key}.columns differs")
        for field in ("energy", "nvfp4_wsse", "trellis_wsse"):
            _finite(item.get(field), where=f"{where}.{rate_key}.{field}", positive=True)
        for field in ("nvfp4_db", "trellis_db", "trellis_minus_nvfp4_db"):
            _finite(item.get(field), where=f"{where}.{rate_key}.{field}")
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
            _finite(scalar.get("wsse"), where=f"{where}.{rate_key}.{scope}.wsse", positive=True)
            _finite(scalar.get("db"), where=f"{where}.{rate_key}.{scope}.db")
            _finite(scalar.get("coding_gain_db"), where=f"{where}.{rate_key}.{scope}.coding_gain_db")
            _text(scalar.get("subset_fit_scope"), where=f"{where}.{rate_key}.{scope}.subset_fit_scope")


def _validate_e2_arm(value: object, *, lane: str, rate: float,
                     shape: list[int], weighted_energy: float,
                     plain_energy: float, where: str) -> None:
    arm = _exact(value, _E2_ARM_KEYS, where=where)
    if arm.get("arm") != lane:
        raise CheckpointContractError(f"{where}.arm differs from key")
    _close(_finite(arm.get("rung"), where=f"{where}.rung"), rate,
           where=f"{where}.rung")
    _finite(arm.get("encode_seconds"), where=f"{where}.encode_seconds", nonnegative=True)
    for domain in ("plain", "weighted"):
        sse = _finite(arm.get(f"{domain}_sse"), where=f"{where}.{domain}_sse", positive=True)
        nsse = _finite(arm.get(f"{domain}_nsse"), where=f"{where}.{domain}_nsse", positive=True)
        snr = _finite(arm.get(f"{domain}_snr_db"), where=f"{where}.{domain}_snr_db")
        energy = weighted_energy if domain == "weighted" else plain_energy
        _close(nsse, sse / energy, where=f"{where}.{domain}_nsse", rel=1e-7)
        _close(snr, -10.0 * math.log10(nsse), where=f"{where}.{domain}_snr_db", rel=1e-7)
        del sse
    if not isinstance(arm.get("reproduces_stage6"), bool):
        raise CheckpointContractError(f"{where}.reproduces_stage6 must be boolean")
    counts = _validate_e2_schedule(arm.get("schedule"), rate=rate,
                                   columns=shape[1], where=f"{where}.schedule")
    _validate_e2_footprint(arm.get("footprint"), lane=lane, rate=rate,
                           shape=shape, where=f"{where}.footprint")
    _validate_e2_subset(arm.get("subset_split"), counts=counts,
                        where=f"{where}.subset_split")


def validate_e2m1_checkpoint(document: object, *, current_receipt: Mapping[str, object],
                              names: Sequence[str], require_partial: bool = True) -> None:
    root = _exact(document, _E2_ROOT_KEYS, where="checkpoint")
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
    _sha(receipt.get("publication_identity_sha256"), where="receipt.publication_identity_sha256")
    comparable_saved = {key: value for key, value in receipt.items()
                        if key not in {"started_at_unix_s", "partial", "tensors_done"}}
    comparable_current = {key: value for key, value in current_receipt.items()
                          if key not in {"started_at_unix_s", "partial", "tensors_done"}}
    if comparable_saved != comparable_current:
        raise CheckpointContractError("receipt identity differs")
    per_tensor = _mapping(root.get("per_tensor"), where="per_tensor")
    if receipt.get("tensors_done") != len(per_tensor):
        raise CheckpointContractError("receipt tensor count differs")
    if not require_partial:
        if len(per_tensor) != len(names):
            raise CheckpointContractError("final checkpoint does not cover every tensor")
        _finite(receipt.get("completed_at_unix_s"), where="receipt.completed_at_unix_s")
        if receipt.get("status") != "ok":
            raise CheckpointContractError("receipt.status must be ok")
    if list(per_tensor) != list(names[:len(per_tensor)]):
        raise CheckpointContractError("per_tensor is not the ordered tensor prefix")
    expected_domain = {(lane, rate) for lane in _E2_LANES for rate in rates}
    for name, raw_cell in per_tensor.items():
        cell = _exact(raw_cell, _E2_CELL_KEYS, where=f"per_tensor.{name}")
        shape = _shape(cell.get("shape"), where=f"per_tensor.{name}.shape")
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
        if measured & unreachable or measured | unreachable != expected_domain:
            raise CheckpointContractError(f"per_tensor.{name} does not cover every expected arm exactly once")
        control = _exact(cell.get("control"), _E2_CONTROL_KEYS,
                         where=f"per_tensor.{name}.control")
        if control.get("status") not in {"pass", "fail", "uncontrolled"}:
            raise CheckpointContractError(f"per_tensor.{name}.control.status differs")
        if not isinstance(control.get("footprint_equal"), bool):
            raise CheckpointContractError(f"per_tensor.{name}.control.footprint_equal must be boolean")
        checks = _mapping(control.get("checks"), where=f"per_tensor.{name}.control.checks")
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
                _finite(mine, where=f"per_tensor.{name}.control.checks.{key}.mine")
                _finite(published, where=f"per_tensor.{name}.control.checks.{key}.published")
                measured_checks += 1
            else:
                missing_checks += 1
            _finite(
                check.get("rel"),
                where=f"per_tensor.{name}.control.checks.{key}.rel",
                nonnegative=True,
            )
        worst = control.get("worst_relative")
        if measured_checks:
            worst_value = _finite(worst, where=f"per_tensor.{name}.control.worst_relative", nonnegative=True)
            measured_relatives = [
                float(check["rel"]) for check in checks.values()
                if check["mine"] is not None
            ]
            _close(worst_value, max(measured_relatives),
                   where=f"per_tensor.{name}.control.worst_relative")
            tolerance = _finite(receipt.get("control_rtol"),
                                where="receipt.control_rtol", nonnegative=True)
            should_pass = (not missing_checks and control.get("footprint_equal")
                           and worst_value <= tolerance)
            if (control.get("status") == "pass") != should_pass:
                raise CheckpointContractError(f"per_tensor.{name}.control.status is inconsistent")
        elif worst is not None or control.get("status") != "uncontrolled":
            raise CheckpointContractError(f"per_tensor.{name}.control empty-check status differs")


def _validate_fp8_settings(settings: object, *, expected: Mapping[str, object]) -> Mapping[str, object]:
    value = _exact(settings, _FP8_SETTINGS_KEYS, where="settings")
    if value != expected:
        raise CheckpointContractError("settings identity differs")
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
                            where: str) -> Mapping[str, object]:
    keys = _FP8_LEARNED_FOOTPRINT_KEYS if learned else _FP8_FIXED_FOOTPRINT_KEYS
    footprint = _exact(value, keys, where=where)
    if footprint.get("schema") != "trellis.fp8_ladder.fp8_cb_accounting.v1":
        raise CheckpointContractError(f"{where}.schema differs")
    if footprint.get("format") != f"FP8_CB_K{rung}":
        raise CheckpointContractError(f"{where}.format differs from rung")
    if footprint.get("codebook") != ("per_tensor_weighted_lloyd" if learned else "fixed_lattice"):
        raise CheckpointContractError(f"{where}.codebook differs from arm")
    _close(_finite(footprint.get("body_bpw"), where=f"{where}.body_bpw", positive=True),
           rung / 8.0, where=f"{where}.body_bpw")
    exact_bpw = _finite(footprint.get("exact_bpw"), where=f"{where}.exact_bpw", positive=True)
    if exact_bpw <= rung / 8.0:
        raise CheckpointContractError(f"{where}.exact_bpw must include side information")
    for field in ("body_bits", "total_bits", "total_bytes", "superblocks"):
        _integer(footprint.get(field), where=f"{where}.{field}", positive=True)
    if footprint.get("scale_coding") != "v1" or footprint.get("scale_contract") != "per_output_row_fp32":
        raise CheckpointContractError(f"{where} scale contract differs")
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
                      energy: float, where: str) -> None:
    keys = _FP8_LEARNED_ARM_KEYS if learned else _FP8_FIXED_ARM_KEYS
    arm = _exact(value, keys, where=where)
    if arm.get("encode_tier") != "balanced":
        raise CheckpointContractError(f"{where}.encode_tier differs")
    _finite(arm.get("encode_seconds_observation_not_perf_claim"),
            where=f"{where}.encode_seconds", nonnegative=True)
    error = _finite(arm.get("weighted_sse"), where=f"{where}.weighted_sse", positive=True)
    nsse = _finite(arm.get("weighted_nsse"), where=f"{where}.weighted_nsse", positive=True)
    snr = _finite(arm.get("weighted_snr_db"), where=f"{where}.weighted_snr_db")
    _close(nsse, error / energy, where=f"{where}.weighted_nsse", rel=1e-7)
    _close(snr, -10.0 * math.log10(nsse), where=f"{where}.weighted_snr_db", rel=1e-7)
    _sha(arm.get("reconstruction_sha256"), where=f"{where}.reconstruction_sha256")
    footprint = _validate_fp8_footprint(arm.get("footprint"), rung=rung,
                                        learned=learned, where=f"{where}.footprint")
    if learned:
        _validate_fp8_book(arm.get("learned_book"), footprint=footprint,
                           where=f"{where}.learned_book")


def validate_fp8_checkpoint(document: object, *, settings: Mapping[str, object],
                            entries: Sequence[object],
                            require_partial: bool = True) -> None:
    root = _exact(
        document,
        _FP8_ROOT_KEYS if require_partial else _FP8_FINAL_ROOT_KEYS,
        where="checkpoint",
    )
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
    names = [entry.name for entry in entries]
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
                              energy=energy, where=f"per_tensor.{name}.arms.fp8_cb@{rung}")
            _validate_fp8_arm(arms[f"fp8_cb_learned@{rung}"], rung=rung, learned=True,
                              energy=energy, where=f"per_tensor.{name}.arms.fp8_cb_learned@{rung}")


__all__ = [
    "CheckpointContractError", "validate_e2m1_checkpoint",
    "validate_fp8_checkpoint",
]
