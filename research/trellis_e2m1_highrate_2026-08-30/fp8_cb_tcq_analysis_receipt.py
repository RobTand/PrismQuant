#!/usr/bin/env python3
"""Independently verify and seal a final GLM FP8-CB/TCQ result.

The campaign result remains the authority.  This read-only verifier checks its
closed receipt, recomputes every population-separated exact-byte frontier and
publishes one no-replace analysis receipt.  It also records descriptive
frontier-crossing diagnostics so a quality advantage is never mislabeled as
Pareto dominance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import stat
from types import SimpleNamespace
from typing import Mapping, Sequence

import atomic_publication as _ATOMIC_PUBLICATION
import fp8_cb_tcq_glm as _PRODUCER
import numeric_checkpoint_contract as _CHECKPOINT_CONTRACT
import numeric_execution_contract as _EXECUTION_CONTRACT
from atomic_publication import (
    PublicationError,
    atomic_checkpoint_json,
    canonical_json_bytes,
    exclusive_publication_claim,
    publish_file_no_replace,
)


SOURCE_SCHEMA = "trellis.glm_fp8_cb_tcq_two_bracket.v1"
SOURCE_STATUS = "measurement_complete_no_serving_verdict"
SCHEMA = "trellis.glm_fp8_cb_tcq_analysis_receipt.v1"
RATES = (4, 5)
CELL_MAP = {4: 32, 5: 40}
BRACKETS = ("production_row_fp32", "two_tier")
SELECTORS = ("lloyd", "exact_dp")
BOOK_PRICES = ("wire8", "fp16_production")
CLAIM_BOUNDARY = {
    "currency": "activation_importance_weighted_weight_sse",
    "activation_contract": "W*A16 screen; no activation quantizer executed",
    "population_aggregation": "dense and routed are separate; pooling forbidden",
    "serving_verdict": False,
    "runtime_claim": False,
    "performance_claim": False,
    "promotion_eligible": False,
}
EXPECTED_PRODUCER_COMMIT = "2f955fa7c073799e494110ff81029027955ee85d"
EXPECTED_MANIFEST_PATH = (
    "/home/rob/dq-runs/glm-corpus-20260830/"
    "final-bf16-pread-1469b9b-v2/manifest.json"
)
EXPECTED_MANIFEST_SHA256 = (
    "a66f800827b92383985ce205004cd2d70b63bcc5e19cada6b05a8162401ee5b0"
)
EXPECTED_CORPUS_FILE_SHA256 = (
    "0d3c08aed48e8d0b540d0705c305cc3197f77c250b07dd7a07e55345f5ddd94e"
)
EXPECTED_IMPORTANCE_VALUE_SHA256 = (
    "dad7818dd11ea8f853bd1869f41189ca3de4a2d10deda52cfef563f63496a9dd"
)
EXPECTED_CORPUS_PRODUCER_COMMIT = "4ddfb1e296997e834ea072db2f2f589950ce94ed"
EXPECTED_ATTESTATION_FILE_SHA256 = (
    "f495263965cba1a6b63d05c86fb328b8f488b3337d78a055090f6af9a84c6ee1"
)
EXPECTED_ATTESTATION_IDENTITY_SHA256 = (
    "1cfad16122829dd0f3554a77d5b630a5005af78a57b706eca886bd756d8c510d"
)
EXPECTED_CONTAINER_ID = "8d5a2b79f98d24f10635bbe6fa7c339a8361377f18926f281ad4e2503d4bac89"
_ACTIVE_SOURCE_SUFFIX_HASHES = {
    "/research/trellis_e2m1_highrate_2026-08-30/fp8_cb_tcq_glm.py":
        "aab789872431108761da43a3eb50fa556f1526024e0ba0cea9d2ed1106aa88b4",
    "/research/trellis_e2m1_highrate_2026-08-30/fp8_learned_glm.py":
        "1b55bdd134590bd0694ee28fef26b10f3fa7888c377d4116341a7fd25d221c51",
    "/research/trellis_e2m1_highrate_2026-08-30/atomic_publication.py":
        "741ce508d5675c1e5b3ffa409d392ac22f7710ba18a6a6115451ce078b1eb743",
    "/research/trellis_e2m1_highrate_2026-08-30/isolated_glm_corpus.py":
        "d5bd3df36cfc89aecbd71b7da933bcb8ef260820ab4c27a9d2016fac01b87b29",
    "/research/trellis_e2m1_highrate_2026-08-30/numeric_execution_contract.py":
        "4b06335da5cf9ffdb1dd5cb636ac5640dca1e95563b942270eb5b8dec06dd75a",
    "/prismaquant/trellis_bf16_corpus.py":
        "67dff31841179bb36005c73f46278ac1c6892cc2668fc9f8d05b9cd0be45abde",
}
EXPECTED_SNAPSHOT_TREE_SHA256 = (
    "072a11920ecbd5e16465f693fe29ebe76bbab91277ec1046208c0e61a39639e6"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_MAX_BOUND_BYTES = 64 << 20
_MAX_CORPUS_BYTES = 2 << 30

_MANIFEST_KEYS = {
    "schema", "status", "generated", "host", "corpus_label",
    "model_profile", "model", "model_config_sha256", "num_hidden_layers",
    "layers", "roles", "expert", "calibration", "importance_identity",
    "reader_contract", "prismaquant_commit", "file", "file_size_bytes",
    "file_sha256", "populations", "source_artifact", "entries",
}
_MANIFEST_ENTRY_KEYS = {
    "name", "population", "layer", "projection", "expert",
    "source_weight_dtype", "source_weight_shape", "source_weight_sha256",
    "importance_key", "importance_shape", "importance_dtype",
    "importance_sha256", "importance_source", "census",
}
_IMPORTANCE_IDENTITY_KEYS = {
    "schema", "probe_file_sha256", "probe_calibration_hash",
    "probe_imatrix_value_sha256", "value_sha256", "dense_normalization",
    "routed_normalization", "gate_up_mapping", "down_mapping",
}
_ACTIVE_SOURCE_LABELS = {
    "driver", "base_fp8_glm_driver", "atomic_publication",
    "isolated_glm_corpus", "numeric_execution_contract",
    "active_corpus_reader",
}
_ACTIVE_SOURCE_LABEL_SUFFIX = {
    "driver": "/research/trellis_e2m1_highrate_2026-08-30/fp8_cb_tcq_glm.py",
    "base_fp8_glm_driver": "/research/trellis_e2m1_highrate_2026-08-30/fp8_learned_glm.py",
    "atomic_publication": "/research/trellis_e2m1_highrate_2026-08-30/atomic_publication.py",
    "isolated_glm_corpus": "/research/trellis_e2m1_highrate_2026-08-30/isolated_glm_corpus.py",
    "numeric_execution_contract": "/research/trellis_e2m1_highrate_2026-08-30/numeric_execution_contract.py",
    "active_corpus_reader": "/prismaquant/trellis_bf16_corpus.py",
}
_LOCKED_SOURCE_KEYS = {
    "fp8_ladder_path", "fp8_ladder_sha256", "hull_sweep_path",
    "hull_sweep_sha256", "e4m3_alphabet_dp_path",
    "e4m3_alphabet_dp_sha256",
}
_LOCKED_EXPECTED_HASHES = {
    "fp8_ladder_sha256": (
        "f9c5167905b98fe98a3389a9471cb9bea06e6ced9a1288329ce1b0fb6a92d2a3"
    ),
    "hull_sweep_sha256": (
        "4420108cae7b024ae7effa75111a187efc0018220082ba724bf995c62b902a98"
    ),
    "e4m3_alphabet_dp_sha256": (
        "022cd576c052cf613eb856a8ad4fce94462e819cb23274815e297f0493491696"
    ),
}
_FROZEN_SOURCE_SUFFIXES = {
    "/trellis-hull-20260828/hull_sweep.py",
    "/trellis-stage0/stage3_mixed_rate.py",
    "/trellis-stage0/stage4_place.py",
    "/trellis-stage0/stage5_e4m3_codec.py",
    "/trellis-stage0/stage5_encoder.py",
    "/trellis-stage0/stage6_inputs_manifest.json",
    "/trellis-stage0/stage6_prismaquant_snapshot/prismaquant/cb_layout.py",
    "/trellis-stage0/stage6_prismaquant_snapshot/prismaquant/export_native_compressed.py",
    "/trellis-stage0/stage6_prismaquant_snapshot/prismaquant/nvfp4_cb_formats.py",
    "/trellis-stage0/stage6_prismaquant_snapshot/prismaquant/trellis_footprint.py",
    "/trellis-stage0/stage6_prismaquant_snapshot/prismaquant/trellis_formats.py",
    "/trellis-stage0/stage6_run_container.sh",
    "/trellis-stage0/stage6_worker.py",
    "/trellis-stage0/tcq_pilot.py",
}
_FROZEN_SOURCE_SUFFIX_HASHES = {
    "/trellis-hull-20260828/hull_sweep.py":
        "4420108cae7b024ae7effa75111a187efc0018220082ba724bf995c62b902a98",
    "/trellis-stage0/stage3_mixed_rate.py":
        "b137b282ca7d67828a26aa24470d5ef219eaca146255ea2043d7c0ee5fb85795",
    "/trellis-stage0/stage4_place.py":
        "f564bce76d4d35ce81e34428d3d5afb9bf27b670e16fc1a462150b72ff472f03",
    "/trellis-stage0/stage5_e4m3_codec.py":
        "eaf3221cf04049b3bf57e555ad2e0fb24ff81ae8eecf71e42558a25395eebb22",
    "/trellis-stage0/stage5_encoder.py":
        "16559405911eb6a397636cde4def0875b54830a47fcb883b07920732cb4240af",
    "/trellis-stage0/stage6_inputs_manifest.json":
        "900ef02851bf3cbe2aa938966e88a398c95fb5bd707a40f173b5ddee8ed3d75a",
    "/trellis-stage0/stage6_prismaquant_snapshot/prismaquant/cb_layout.py":
        "52d664a483625ee38c208c68dd489f83d5fb7922740998c80fd7ad4ef3c5a0e5",
    "/trellis-stage0/stage6_prismaquant_snapshot/prismaquant/export_native_compressed.py":
        "cec4e8f18d36f0c9b1f70cc69959bcf27449fe048fb8c612c488ea042326129a",
    "/trellis-stage0/stage6_prismaquant_snapshot/prismaquant/nvfp4_cb_formats.py":
        "9f886165d4495f8e93615ac3804b41d87a69c8c4526833c196817366147d23d1",
    "/trellis-stage0/stage6_prismaquant_snapshot/prismaquant/trellis_footprint.py":
        "0281aa50988f60680a55f4800e2b591304711d61d08860599ca4d248342562a4",
    "/trellis-stage0/stage6_prismaquant_snapshot/prismaquant/trellis_formats.py":
        "944a8de25345e0a10edc3e50161dc188a1ecdfbb03e22b8d16394af797804691",
    "/trellis-stage0/stage6_run_container.sh":
        "e97abe0e0797187da7b01ec4130377b3fdf2db231e5044d13f1da096c18b7fab",
    "/trellis-stage0/stage6_worker.py":
        "6d0985210c23d9b50c20c8443237f8dbd7aec836b0108b5c168a6c46185bc948",
    "/trellis-stage0/tcq_pilot.py":
        "13bd902641ec7385cf84d96f4c0d8192acdbf3c72f0976c42359c3b4b6faeb2d",
}
_ATTESTATION_KEYS = {
    "schema", "verification_scope", "physical_host", "uts_hostname",
    "gpu_uuid", "container_id", "container_hostname", "container_state",
    "container_rootfs_changes", "container_user", "image_reference",
    "image_digest", "image_id", "uts_mode", "network_mode", "ipc_mode",
    "gpu_request", "launch_attestation_container_path", "repo_root",
    "git_common_dir", "repo_mount_readonly", "git_mount_readonly",
    "storage_mount_readwrite", "rootfs_readonly", "runtime_isolation",
    "launch_environment", "launch_command", "launch_command_sha256",
    "attestation_sha256",
}
_TCQ_BASE_FOOTPRINT_KEYS = {
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
}
_TCQ_PRODUCTION_FOOTPRINT_KEYS = _TCQ_BASE_FOOTPRINT_KEYS | {
    "non_shipping_research", "scale_coding", "scale_bpw",
}
_TCQ_TWO_TIER_FOOTPRINT_KEYS = _TCQ_PRODUCTION_FOOTPRINT_KEYS | {
    "production_payload_v1", "research_pricing_note",
    "scale_bytes_v1_production",
}
_SCHEDULE_KEYS = {
    "achieved_rate", "body_bits_per_block_max", "body_bits_per_block_min",
    "body_bits_per_block_std", "counts", "fixed_quota_per_256", "invert",
    "maximum_rate", "minimum_trellis_steps", "schedule_sha256",
    "tailbite_guard_fixups", "target_rate", "transitions_per_block_max",
    "transitions_per_block_mean",
}
_ALPHABET_KEYS = {"alphabet_mode", "rule", "tcq_native_codes"}
_DENSE_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>[0-9]+)\.mlp\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
_ROUTED_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>[0-9]+)\.mlp\.experts\.0\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


class AnalysisReceiptError(RuntimeError):
    pass


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev), int(info.st_ino), int(info.st_mode),
        int(info.st_uid), int(info.st_gid), int(info.st_size),
        int(info.st_mtime_ns), int(info.st_ctime_ns), int(info.st_nlink),
    )


def _read_bound_file(path: Path) -> tuple[bytes, dict[str, object]]:
    """Read one stable regular-file inode and bind its exact bytes once."""

    candidate = path.absolute()
    if not hasattr(os, "O_NOFOLLOW"):
        raise AnalysisReceiptError("O_NOFOLLOW is required for bound inputs")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise AnalysisReceiptError(f"cannot open bound file {candidate}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AnalysisReceiptError(f"bound input is not regular: {candidate}")
        if before.st_size < 0 or before.st_size > _MAX_BOUND_BYTES:
            raise AnalysisReceiptError(
                f"bound input exceeds {_MAX_BOUND_BYTES} bytes: {candidate}"
            )
        chunks = []
        total = 0
        while chunk := os.read(descriptor, min(8 << 20, _MAX_BOUND_BYTES + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_BOUND_BYTES:
                raise AnalysisReceiptError(
                    f"bound input exceeds {_MAX_BOUND_BYTES} bytes: {candidate}"
                )
        after = os.fstat(descriptor)
        try:
            path_after = os.stat(candidate, follow_symlinks=False)
        except OSError as exc:
            raise AnalysisReceiptError(
                f"bound input path changed while read: {candidate}"
            ) from exc
        identity = _file_identity(before)
        if (
            _file_identity(after) != identity
            or _file_identity(path_after) != identity
            or not stat.S_ISREG(path_after.st_mode)
        ):
            raise AnalysisReceiptError(
                f"bound input identity changed while read: {candidate}"
            )
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            raise AnalysisReceiptError(f"bound input size changed while read: {candidate}")
        return raw, {
            "path": str(candidate),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    finally:
        os.close(descriptor)


def _stable_file_sha256(path: Path) -> str:
    return str(_read_bound_file(path)[1]["sha256"])


def _stream_bound_file(path: Path, *, expected_size: int) -> dict[str, object]:
    """Hash one large stable regular file without loading it into memory."""

    candidate = path.absolute()
    if not hasattr(os, "O_NOFOLLOW"):
        raise AnalysisReceiptError("O_NOFOLLOW is required for bound inputs")
    if not 0 < expected_size <= _MAX_CORPUS_BYTES:
        raise AnalysisReceiptError("corpus artifact size is outside the closed bound")
    try:
        descriptor = os.open(
            candidate,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise AnalysisReceiptError(f"cannot open corpus artifact {candidate}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise AnalysisReceiptError("corpus artifact type/size differs")
        digest = hashlib.sha256()
        total = 0
        while block := os.read(descriptor, 8 << 20):
            digest.update(block)
            total += len(block)
            if total > _MAX_CORPUS_BYTES:
                raise AnalysisReceiptError("corpus artifact exceeds the closed bound")
        after = os.fstat(descriptor)
        path_after = os.stat(candidate, follow_symlinks=False)
        identity = _file_identity(before)
        if (
            total != expected_size
            or _file_identity(after) != identity
            or _file_identity(path_after) != identity
            or not stat.S_ISREG(path_after.st_mode)
        ):
            raise AnalysisReceiptError("corpus artifact identity changed while read")
        return {
            "path": str(candidate), "sha256": digest.hexdigest(),
            "size_bytes": total,
        }
    finally:
        os.close(descriptor)


_VERIFIER_PATH = Path(__file__).absolute()
_DEPENDENCY_PATHS = {
    "atomic_publication": Path(_ATOMIC_PUBLICATION.__file__).absolute(),
    "producer_validator": Path(_PRODUCER.__file__).absolute(),
    "checkpoint_contract": Path(_CHECKPOINT_CONTRACT.__file__).absolute(),
    "execution_contract": Path(_EXECUTION_CONTRACT.__file__).absolute(),
}
_IMPORT_VERIFIER_BINDING = _read_bound_file(_VERIFIER_PATH)[1]
_IMPORT_DEPENDENCY_BINDINGS = {
    name: _read_bound_file(path)[1]
    for name, path in _DEPENDENCY_PATHS.items()
}


def _strict_json_object(
    path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    def object_from_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str):
        raise ValueError(f"non-finite JSON constant {token!r}")

    raw, binding = _read_bound_file(path)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError) as exc:
        raise AnalysisReceiptError(f"invalid JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisReceiptError(f"{path} is not one JSON object")
    return value, binding


def _identity_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _compact_identity_sha256(value: Mapping[str, object]) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _exact_keys(
    value: object, expected: set[str], *, where: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AnalysisReceiptError(f"{where} must be an object")
    actual = set(value)
    if actual != expected:
        raise AnalysisReceiptError(
            f"{where} members differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _integer(
    value: object, *, where: str, positive: bool = False,
) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise AnalysisReceiptError(f"{where} must be a {qualifier} integer")
    return value


def _sha(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AnalysisReceiptError(f"{where} must be lowercase SHA-256")
    return value


def _close(actual: object, expected: float, *, where: str) -> None:
    got = _finite(actual, where=where, nonnegative=True)
    if not math.isclose(got, expected, rel_tol=0, abs_tol=1e-12):
        raise AnalysisReceiptError(f"{where} differs from integer accounting")


def _finite(value: object, *, where: str, nonnegative: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AnalysisReceiptError(f"{where} must be finite")
    result = float(value)
    if nonnegative and result < 0:
        raise AnalysisReceiptError(f"{where} must be nonnegative")
    return result


def _arm_names() -> set[str]:
    cb = {
        f"fp8_cb_{kind}@{rung}"
        for kind in ("fixed", "learned")
        for rung in CELL_MAP.values()
    }
    tcq = {
        f"tcq_e4m3.{bracket}.{selector}@{rate}"
        for bracket in BRACKETS
        for selector in SELECTORS
        for rate in RATES
    }
    return cb | tcq


ARM_NAMES = _arm_names()


def _cost_bits(arm: Mapping[str, object], *, book_price: str) -> int:
    """Return the canonical serialized integer cost for one tensor arm."""

    footprint = arm.get("footprint")
    if not isinstance(footprint, Mapping):
        raise AnalysisReceiptError("arm footprint is absent")
    if arm.get("family") == "TCQ_E4M3_R256":
        return 8 * _integer(
            footprint.get("total_bytes"), where="footprint.total_bytes",
            positive=True,
        )
    total = _integer(
        footprint.get("total_bits"), where="footprint.total_bits", positive=True,
    )
    if arm.get("book_kind") == "per_tensor_weighted_lloyd" and book_price == "wire8":
        production = _integer(
            footprint.get("codebook_side_bits"),
            where="footprint.codebook_side_bits",
        )
        wire = _integer(
            footprint.get("codebook_side_bits_wire8"),
            where="footprint.codebook_side_bits_wire8",
        )
        if not 0 <= wire <= production <= total:
            raise AnalysisReceiptError("learned-book integer side-bit order differs")
        return total - production + wire
    return total


def _frontier(points: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for point in points:
        dominated = any(
            other is not point
            and other["bits"] <= point["bits"]
            and other["snr"] >= point["snr"]
            and (other["bits"] < point["bits"] or other["snr"] > point["snr"])
            for other in points
        )
        if not dominated:
            result.append(point)
    return sorted(result, key=lambda item: (item["bits"], -item["snr"], item["arm"]))


def _family_dominates(
    candidate: Sequence[dict[str, object]], other: Sequence[dict[str, object]]
) -> bool:
    return all(
        any(
            left["bits"] <= right["bits"]
            and left["snr"] >= right["snr"]
            and (left["bits"] < right["bits"] or left["snr"] > right["snr"])
            for left in candidate
        )
        for right in other
    )


def _tensor_verdict(
    cell: Mapping[str, object], *, rate: int, bracket: str, book_price: str
) -> dict[str, object]:
    arms = cell["arms"]
    rows, columns = cell["shape"]
    numel = int(rows) * int(columns)
    rung = CELL_MAP[rate]
    cb = _frontier([
        {
            "arm": name,
            "bits": _cost_bits(arms[name], book_price=book_price),
            "snr": float(arms[name]["weighted_snr_db"]),
        }
        for name in (f"fp8_cb_fixed@{rung}", f"fp8_cb_learned@{rung}")
    ])
    tcq = _frontier([
        {
            "arm": name,
            "bits": _cost_bits(arms[name], book_price=book_price),
            "snr": float(arms[name]["weighted_snr_db"]),
        }
        for name in (
            f"tcq_e4m3.{bracket}.lloyd@{rate}",
            f"tcq_e4m3.{bracket}.exact_dp@{rate}",
        )
    ])
    cb_dominates = _family_dominates(cb, tcq)
    tcq_dominates = _family_dominates(tcq, cb)
    if cb_dominates and tcq_dominates:
        raise AnalysisReceiptError("mutual strict family dominance is impossible")
    best_cb = max(cb, key=lambda point: point["snr"])
    best_tcq = max(tcq, key=lambda point: point["snr"])
    for point in (*cb, *tcq):
        point["bpw"] = point["bits"] / numel
    return {
        "verdict": (
            "FP8_CB" if cb_dominates else
            "TCQ_E4M3" if tcq_dominates else
            "NO_VERDICT_exact_byte_frontiers_cross"
        ),
        "cb_frontier": cb,
        "tcq_frontier": tcq,
        "best_quality_cb_minus_tcq_db": best_cb["snr"] - best_tcq["snr"],
        "best_quality_cb_bpw": best_cb["bpw"],
        "best_quality_tcq_bpw": best_tcq["bpw"],
        "minimum_cb_bpw": min(point["bits"] for point in cb) / numel,
        "minimum_tcq_bpw": min(point["bits"] for point in tcq) / numel,
    }


def _population_summaries(
    per_tensor: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    output: dict[str, object] = {}
    for population in ("dense", "routed"):
        cells = [
            cell for cell in per_tensor.values()
            if cell.get("population") == population
        ]
        bracket_rows = []
        final_cells = []
        for rate in RATES:
            combination_verdicts = []
            for bracket in BRACKETS:
                for book_price in BOOK_PRICES:
                    rows = [
                        _tensor_verdict(
                            cell, rate=rate, bracket=bracket,
                            book_price=book_price,
                        )
                        for cell in cells
                    ]
                    counts = {
                        label: sum(row["verdict"] == label for row in rows)
                        for label in (
                            "FP8_CB", "TCQ_E4M3",
                            "NO_VERDICT_exact_byte_frontiers_cross",
                        )
                    }
                    unanimous = (
                        "FP8_CB" if counts["FP8_CB"] == len(rows) else
                        "TCQ_E4M3" if counts["TCQ_E4M3"] == len(rows) else
                        "NO_VERDICT_mixed_or_crossing"
                    )
                    combination_verdicts.append(unanimous)
                    bracket_rows.append({
                        "nominal_body_bpw": rate,
                        "fp8_cb_rung": CELL_MAP[rate],
                        "trellis_scale_bracket": bracket,
                        "learned_book_price_bracket": book_price,
                        "tensors": len(rows),
                        "counts": counts,
                        "verdict": unanimous,
                        "best_quality_cb_minus_tcq_db_median": statistics.median(
                            row["best_quality_cb_minus_tcq_db"] for row in rows
                        ),
                        "best_quality_cb_bpw_median": statistics.median(
                            row["best_quality_cb_bpw"] for row in rows
                        ),
                        "best_quality_tcq_bpw_median": statistics.median(
                            row["best_quality_tcq_bpw"] for row in rows
                        ),
                    })
            final_cells.append({
                "nominal_body_bpw": rate,
                "fp8_cb_rung": CELL_MAP[rate],
                "verdict": (
                    combination_verdicts[0]
                    if len(set(combination_verdicts)) == 1
                    and combination_verdicts[0] in {"FP8_CB", "TCQ_E4M3"}
                    else "NO_VERDICT_brackets_disagree_or_frontiers_cross"
                ),
                "required_combination_verdicts": combination_verdicts,
            })
        output[population] = {
            "tensors": len(cells),
            "bracket_rows": bracket_rows,
            "cells": final_cells,
        }
    return output


def _frontier_diagnostics(
    per_tensor: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    output: dict[str, object] = {}
    for population in ("dense", "routed"):
        cells = [
            cell for cell in per_tensor.values()
            if cell.get("population") == population
        ]
        rows = []
        for rate in RATES:
            for bracket in BRACKETS:
                for book_price in BOOK_PRICES:
                    values = [
                        _tensor_verdict(
                            cell, rate=rate, bracket=bracket,
                            book_price=book_price,
                        )
                        for cell in cells
                    ]
                    rows.append({
                        "nominal_body_bpw": rate,
                        "trellis_scale_bracket": bracket,
                        "learned_book_price_bracket": book_price,
                        "tensors": len(values),
                        "tcq_best_quality_higher": sum(
                            value["best_quality_cb_minus_tcq_db"] < 0
                            for value in values
                        ),
                        "cb_minimum_bpw_lower": sum(
                            value["minimum_cb_bpw"] < value["minimum_tcq_bpw"]
                            for value in values
                        ),
                        "minimum_cb_minus_tcq_bpw_median": statistics.median(
                            value["minimum_cb_bpw"] - value["minimum_tcq_bpw"]
                            for value in values
                        ),
                    })
        output[population] = rows
    return output


def _classify_name(name: str) -> tuple[str, int, str, int | None]:
    routed = _ROUTED_RE.fullmatch(name)
    if routed is not None:
        return "routed", int(routed.group("layer")), routed.group("projection"), 0
    dense = _DENSE_RE.fullmatch(name)
    if dense is not None:
        return "dense", int(dense.group("layer")), dense.group("projection"), None
    raise AnalysisReceiptError(f"manifest tensor name is outside the GLM census: {name!r}")


def _expected_shape(population: str, projection: str) -> tuple[int, int]:
    if population == "dense":
        return (4096, 12288) if projection == "down_proj" else (12288, 4096)
    return (4096, 2048) if projection == "down_proj" else (2048, 4096)


def _manifest_entries(
    manifest: Mapping[str, object], *, binding: Mapping[str, object],
    settings: Mapping[str, object],
) -> tuple[tuple[SimpleNamespace, ...], dict[str, object]]:
    _exact_keys(manifest, _MANIFEST_KEYS, where="corpus manifest")
    if manifest.get("schema") != "trellis.bf16_corpus.v2" or manifest.get("status") != "finalized":
        raise AnalysisReceiptError("corpus manifest is not finalized schema v2")
    if (
        binding.get("path") != EXPECTED_MANIFEST_PATH
        or binding.get("sha256") != EXPECTED_MANIFEST_SHA256
        or settings.get("corpus_manifest") != EXPECTED_MANIFEST_PATH
        or settings.get("corpus_manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or settings.get("corpus_file_sha256") != EXPECTED_CORPUS_FILE_SHA256
        or manifest.get("file_sha256") != EXPECTED_CORPUS_FILE_SHA256
        or settings.get("importance_value_sha256")
        != EXPECTED_IMPORTANCE_VALUE_SHA256
        or settings.get("corpus_prismaquant_commit")
        != EXPECTED_CORPUS_PRODUCER_COMMIT
        or manifest.get("prismaquant_commit") != EXPECTED_CORPUS_PRODUCER_COMMIT
    ):
        raise AnalysisReceiptError("exact authoritative corpus identity differs")
    _sha(manifest.get("file_sha256"), where="manifest.file_sha256")
    artifact_name = manifest.get("file")
    if (
        not isinstance(artifact_name, str)
        or Path(artifact_name).name != artifact_name
    ):
        raise AnalysisReceiptError("manifest artifact name must be a basename")
    artifact_size = _integer(
        manifest.get("file_size_bytes"), where="manifest.file_size_bytes",
        positive=True,
    )
    artifact_binding = _stream_bound_file(
        Path(str(binding["path"])).parent / artifact_name,
        expected_size=artifact_size,
    )
    if artifact_binding["sha256"] != EXPECTED_CORPUS_FILE_SHA256:
        raise AnalysisReceiptError("corpus artifact SHA-256 differs")
    if not isinstance(manifest.get("prismaquant_commit"), str) or _COMMIT.fullmatch(
        str(manifest.get("prismaquant_commit"))
    ) is None:
        raise AnalysisReceiptError("manifest.prismaquant_commit is invalid")
    if (
        manifest.get("model_profile") != "glm5_next"
        or manifest.get("num_hidden_layers") != 45
        or manifest.get("layers") != [0, 1, 2, 3, 9, 15, 21, 26, 32, 38, 44]
        or manifest.get("roles") != ["gate_proj", "up_proj", "down_proj"]
        or manifest.get("expert") != 0
        or manifest.get("populations") != {
            "dense": {"count": 9, "layers": [0, 1, 2]},
            "routed": {"count": 24, "layers": [3, 9, 15, 21, 26, 32, 38, 44]},
        }
    ):
        raise AnalysisReceiptError("manifest GLM population census differs")
    importance_identity = _exact_keys(
        manifest.get("importance_identity"), _IMPORTANCE_IDENTITY_KEYS,
        where="manifest.importance_identity",
    )
    if (
        importance_identity.get("schema")
        != "prismaquant.glm_trellis_importance.probe_imatrix.v1"
        or settings.get("importance_value_sha256")
        != importance_identity.get("value_sha256")
    ):
        raise AnalysisReceiptError("manifest importance identity differs")
    for field in (
        "probe_file_sha256", "probe_imatrix_value_sha256", "value_sha256",
    ):
        _sha(importance_identity.get(field), where=f"importance_identity.{field}")
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != 33:
        raise AnalysisReceiptError("manifest must contain exactly 33 entries")
    dense_layers = {0, 1, 2}
    routed_layers = {3, 9, 15, 21, 26, 32, 38, 44}
    expected_names = sorted(
        [
            f"model.language_model.layers.{layer}.mlp.{projection}.weight"
            for layer in dense_layers
            for projection in ("down_proj", "gate_proj", "up_proj")
        ]
        + [
            f"model.language_model.layers.{layer}.mlp.experts.0.{projection}.weight"
            for layer in routed_layers
            for projection in ("down_proj", "gate_proj", "up_proj")
        ]
    )
    entries = []
    for index, raw in enumerate(raw_entries):
        entry = _exact_keys(raw, _MANIFEST_ENTRY_KEYS, where=f"manifest.entries[{index}]")
        name = entry.get("name")
        if not isinstance(name, str) or name != expected_names[index]:
            raise AnalysisReceiptError("manifest entry order/name census differs")
        population, layer, projection, expert = _classify_name(name)
        shape = _expected_shape(population, projection)
        if (
            entry.get("population") != population
            or entry.get("layer") != layer
            or entry.get("projection") != projection
            or entry.get("expert") != expert
            or entry.get("source_weight_dtype") != "torch.bfloat16"
            or entry.get("source_weight_shape") != list(shape)
            or entry.get("importance_key") != f"__bf16_importance__.{name}"
            or entry.get("importance_shape") != [shape[1]]
            or entry.get("importance_dtype") != "torch.float32"
        ):
            raise AnalysisReceiptError(f"{name}: manifest entry identity differs")
        source_sha = _sha(
            entry.get("source_weight_sha256"), where=f"{name}.source_weight_sha256",
        )
        importance_sha = _sha(
            entry.get("importance_sha256"), where=f"{name}.importance_sha256",
        )
        importance_source = _exact_keys(
            entry.get("importance_source"),
            {"qname", "expert", "denominator_name", "denominator"},
            where=f"{name}.importance_source",
        )
        expected_qname = (
            name.removesuffix(".weight")
            if population == "dense"
            else (
                name.split(".experts.0.", 1)[0]
                + ".experts."
                + ("down_proj" if projection == "down_proj" else "gate_up_proj")
            )
        )
        if (
            importance_source.get("qname") != expected_qname
            or importance_source.get("expert") != expert
            or importance_source.get("denominator_name")
            != ("expert_tokens" if population == "routed" else "n_tokens_seen")
        ):
            raise AnalysisReceiptError(f"{name}: importance provenance differs")
        denominator = _integer(
            importance_source.get("denominator"),
            where=f"{name}.importance_source.denominator", positive=True,
        )
        census = _exact_keys(
            entry.get("census"), {"distinct_source_values", "numel"},
            where=f"{name}.census",
        )
        if (
            _integer(census.get("numel"), where=f"{name}.census.numel", positive=True)
            != shape[0] * shape[1]
        ):
            raise AnalysisReceiptError(f"{name}: manifest census numel differs")
        _integer(
            census.get("distinct_source_values"),
            where=f"{name}.census.distinct_source_values", positive=True,
        )
        entries.append(SimpleNamespace(
            name=name, population=population, source_weight_shape=shape,
            source_weight_sha256=source_sha, importance_sha256=importance_sha,
            importance_source_qname=expected_qname,
            importance_source_expert=expert,
            importance_denominator_name=str(importance_source["denominator_name"]),
            importance_denominator=denominator,
        ))
    return tuple(entries), artifact_binding


def _validate_path_hash(path_value: object, sha_value: object, *, where: str) -> None:
    if not isinstance(path_value, str) or not path_value.startswith("/"):
        raise AnalysisReceiptError(f"{where}.path must be absolute")
    digest = _sha(sha_value, where=f"{where}.sha256")
    if _stable_file_sha256(Path(path_value)) != digest:
        raise AnalysisReceiptError(f"{where} bound bytes differ")


def _validate_source_closure(
    settings: Mapping[str, object], *, source_path: str,
) -> None:
    environment = settings["environment"]
    active = _exact_keys(
        settings.get("active_source_identity"),
        {"repo_root", "repo_git_commit", "files"},
        where="settings.active_source_identity",
    )
    if (
        active.get("repo_root") != environment.get("repo_root")
        or active.get("repo_git_commit") != EXPECTED_PRODUCER_COMMIT
        or environment.get("repo_git_commit") != EXPECTED_PRODUCER_COMMIT
    ):
        raise AnalysisReceiptError("active source exact commit/repository differs")
    files = _exact_keys(
        active.get("files"), _ACTIVE_SOURCE_LABELS,
        where="settings.active_source_identity.files",
    )
    for label, raw in files.items():
        item = _exact_keys(raw, {"path", "sha256"}, where=f"active source {label}")
        suffix = _ACTIVE_SOURCE_LABEL_SUFFIX[label]
        if (
            item.get("path") != str(active["repo_root"]) + suffix
            or item.get("sha256") != _ACTIVE_SOURCE_SUFFIX_HASHES[suffix]
        ):
            raise AnalysisReceiptError(f"active source {label} pinned identity differs")
        _validate_path_hash(item.get("path"), item.get("sha256"), where=f"active source {label}")
    command = settings.get("command")
    if not isinstance(command, list) or command != [
        files["driver"]["path"], "--manifest", settings["corpus_manifest"],
        "--out", source_path,
    ]:
        raise AnalysisReceiptError("settings.command is not the exact closed driver argv")

    locked = _exact_keys(
        settings.get("locked_sources"), _LOCKED_SOURCE_KEYS,
        where="settings.locked_sources",
    )
    for stem in ("fp8_ladder", "hull_sweep", "e4m3_alphabet_dp"):
        path_key, sha_key = f"{stem}_path", f"{stem}_sha256"
        if locked.get(sha_key) != _LOCKED_EXPECTED_HASHES[sha_key]:
            raise AnalysisReceiptError(f"locked {stem} expected digest differs")
        _validate_path_hash(locked.get(path_key), locked.get(sha_key), where=f"locked {stem}")

    closure = _exact_keys(
        settings.get("frozen_codec_closure"),
        {"snapshot_tree_sha256", "source_sha256", "imported_codec_modules"},
        where="settings.frozen_codec_closure",
    )
    if closure.get("snapshot_tree_sha256") != EXPECTED_SNAPSHOT_TREE_SHA256:
        raise AnalysisReceiptError("frozen snapshot tree digest differs")
    sources = closure.get("source_sha256")
    if not isinstance(sources, Mapping) or len(sources) != len(_FROZEN_SOURCE_SUFFIXES):
        raise AnalysisReceiptError("frozen source closure domain differs")
    matched_suffixes = set()
    for path_text, digest in sources.items():
        if not isinstance(path_text, str):
            raise AnalysisReceiptError("frozen source path is malformed")
        matches = {suffix for suffix in _FROZEN_SOURCE_SUFFIXES if path_text.endswith(suffix)}
        if len(matches) != 1:
            raise AnalysisReceiptError(f"frozen source path is outside closure: {path_text}")
        matched_suffixes.update(matches)
        suffix = next(iter(matches))
        if digest != _FROZEN_SOURCE_SUFFIX_HASHES[suffix]:
            raise AnalysisReceiptError(f"frozen source {suffix} pinned digest differs")
        _validate_path_hash(path_text, digest, where=f"frozen source {path_text}")
    if matched_suffixes != _FROZEN_SOURCE_SUFFIXES:
        raise AnalysisReceiptError("frozen source suffix census differs")
    imported = _exact_keys(
        closure.get("imported_codec_modules"), {"H", "C", "W", "P", "S4", "TF"},
        where="closure.imported_codec_modules",
    )
    for label, raw in imported.items():
        item = _exact_keys(raw, {"path", "sha256"}, where=f"imported codec {label}")
        if sources.get(item.get("path")) != item.get("sha256"):
            raise AnalysisReceiptError(f"imported codec {label} is outside frozen source map")


def _validate_attestation(
    segment: Mapping[str, object], *, settings: Mapping[str, object],
) -> dict[str, object]:
    path = Path(str(segment["launch_attestation_path"]))
    attestation, binding = _strict_json_object(path)
    _exact_keys(attestation, _ATTESTATION_KEYS, where="launch attestation")
    raw, rebound = _read_bound_file(path)
    if rebound != binding or raw != canonical_json_bytes(attestation):
        raise AnalysisReceiptError("launch attestation bytes are not stable canonical JSON")
    unsigned = {key: value for key, value in attestation.items() if key != "attestation_sha256"}
    if (
        binding.get("sha256") != EXPECTED_ATTESTATION_FILE_SHA256
        or attestation.get("attestation_sha256")
        != EXPECTED_ATTESTATION_IDENTITY_SHA256
        or attestation.get("attestation_sha256") != _compact_identity_sha256(unsigned)
        or segment.get("launch_attestation_sha256")
        != attestation.get("attestation_sha256")
        or attestation.get("schema") != "trellis.numeric_launch_attestation.v1"
        or attestation.get("verification_scope") != "host_docker_daemon_inspect_before_start"
    ):
        raise AnalysisReceiptError("launch attestation digest/schema differs")
    environment = settings["environment"]
    expected_links = {
        "physical_host": environment["physical_host"],
        "uts_hostname": environment["uts_hostname"],
        "gpu_uuid": environment["gpu_uuid"],
        "container_id": segment["container_id"],
        "container_user": environment["container_user"],
        "image_reference": environment["container_image_reference"],
        "image_digest": environment["container_image_digest"],
        "image_id": environment["container_image_id"],
        "ipc_mode": environment["ipc_mode"],
        "repo_root": environment["repo_root"],
        "launch_attestation_container_path": str(path),
    }
    for field, expected in expected_links.items():
        if attestation.get(field) != expected:
            raise AnalysisReceiptError(f"launch attestation {field} differs")
    if segment.get("container_id") != EXPECTED_CONTAINER_ID:
        raise AnalysisReceiptError("exact container identity differs")
    fixed = {
        "container_state": "created", "uts_mode": "host",
        "network_mode": "none", "repo_mount_readonly": True,
        "git_mount_readonly": True, "storage_mount_readwrite": True,
        "rootfs_readonly": True,
    }
    for field, expected in fixed.items():
        if attestation.get(field) != expected:
            raise AnalysisReceiptError(f"launch attestation {field} differs")
    launch = attestation.get("launch_command")
    command = settings["command"]
    repo_root = environment["repo_root"]
    if (
        not isinstance(launch, list)
        or len(launch) != 8 + len(command)
        or launch[:3] != [
            "/usr/bin/python3", "-B",
            f"{repo_root}/research/trellis_e2m1_highrate_2026-08-30/numeric_profiled_launcher.py",
        ]
        or launch[3] != "--profile"
        or not isinstance(launch[4], str)
        or launch[5:8] != ["--", "/usr/bin/python3", "-B"]
        or launch[8:] != command
    ):
        raise AnalysisReceiptError("launch attestation command differs")
    launch_sha = _compact_identity_sha256(launch)
    if (
        attestation.get("launch_command_sha256") != launch_sha
        or segment.get("launch_command_sha256") != launch_sha
    ):
        raise AnalysisReceiptError("launch command digest differs")
    launch_environment = attestation.get("launch_environment")
    if not isinstance(launch_environment, Mapping) or any(
        launch_environment.get(field) != expected
        for field, expected in {
            "HULL_PHYSICAL_HOST": environment["physical_host"],
            "HULL_REPO_ROOT": environment["repo_root"],
            "HULL_CONTAINER_IMAGE": environment["container_image_digest"],
            "HULL_LAUNCH_ATTESTATION": str(path),
        }.items()
    ):
        raise AnalysisReceiptError("launch environment linkage differs")
    return binding


def _validate_schedule(
    schedule: object, *, rate: int, columns: int, where: str,
) -> dict[str, int]:
    value = _exact_keys(schedule, _SCHEDULE_KEYS, where=where)
    _close(value.get("target_rate"), float(rate), where=f"{where}.target_rate")
    _close(value.get("achieved_rate"), float(rate), where=f"{where}.achieved_rate")
    if value.get("maximum_rate") != 8 or value.get("invert") is not False or value.get("fixed_quota_per_256") is not False:
        raise AnalysisReceiptError(f"{where} scheduling mode differs")
    _sha(value.get("schedule_sha256"), where=f"{where}.schedule_sha256")
    counts = _exact_keys(value.get("counts"), {str(i) for i in range(1, 9)}, where=f"{where}.counts")
    parsed = {
        key: _integer(raw, where=f"{where}.counts.{key}")
        for key, raw in counts.items()
    }
    if sum(parsed.values()) != columns or sum(int(key) * count for key, count in parsed.items()) != rate * columns:
        raise AnalysisReceiptError(f"{where}.counts do not encode the target rate")
    for field in (
        "body_bits_per_block_max", "body_bits_per_block_min",
        "minimum_trellis_steps", "tailbite_guard_fixups",
        "transitions_per_block_max",
    ):
        _integer(value.get(field), where=f"{where}.{field}")
    for field in ("body_bits_per_block_std", "transitions_per_block_mean"):
        _finite(value.get(field), where=f"{where}.{field}", nonnegative=True)
    return parsed


def _validate_alphabet(
    alphabet: object, *, selector: str, where: str,
) -> Mapping[str, list[int]]:
    value = _exact_keys(alphabet, _ALPHABET_KEYS, where=where)
    if value.get("alphabet_mode") != selector or not isinstance(value.get("rule"), str) or not value["rule"]:
        raise AnalysisReceiptError(f"{where} selector/rule differs")
    codes = _exact_keys(
        value.get("tcq_native_codes"), {str(i) for i in range(1, 8)},
        where=f"{where}.tcq_native_codes",
    )
    for rate_text, raw in codes.items():
        expected = 1 << (int(rate_text) + 1)
        if (
            not isinstance(raw, list) or len(raw) != expected
            or any(type(code) is not int or not 0 <= code <= 255 for code in raw)
        ):
            raise AnalysisReceiptError(f"{where}.tcq_native_codes.{rate_text} differs")
    return codes


def _validate_tcq_footprint(
    footprint: object, *, bracket: str, rate: int, shape: Sequence[int],
    counts: Mapping[str, int], codes: Mapping[str, Sequence[int]], where: str,
) -> int:
    expected_keys = (
        _TCQ_PRODUCTION_FOOTPRINT_KEYS
        if bracket == "production_row_fp32"
        else _TCQ_TWO_TIER_FOOTPRINT_KEYS
    )
    value = _exact_keys(footprint, expected_keys, where=where)
    rows, columns = map(int, shape)
    body_bits_per_row = rate * columns
    unpadded = (body_bits_per_row + 7) // 8
    stride = ((unpadded + 15) // 16) * 16
    body_bytes = rows * stride
    block_count = (columns + 255) // 256
    schedule_bytes = (columns * 4 + 7) // 8
    offset_bits = 32 if body_bits_per_row <= 0xFFFFFFFF else 64
    offset_bytes = (block_count + 1) * (offset_bits // 8)
    alphabet_by_rate = {
        key: 3 + len(codes[key])
        for key, count in counts.items() if count and key in codes
    }
    alphabet_bytes = sum(alphabet_by_rate.values())
    side = 88 + schedule_bytes + offset_bytes + alphabet_bytes
    production_scale = rows * 4
    scale = production_scale if bracket == "production_row_fp32" else rows * (columns // 256) * 9
    total = body_bytes + scale + side
    expected_integers = {
        "body_rate_q256": rate * 256, "body_bits_per_row": body_bits_per_row,
        "unpadded_body_bytes_per_row": unpadded, "body_row_stride_bytes": stride,
        "body_padding_bytes": rows * (stride - unpadded), "body_bytes": body_bytes,
        "block_count": block_count, "schedule_bytes": schedule_bytes,
        "block_offset_bits": offset_bits, "block_offset_bytes": offset_bytes,
        "alphabet_bytes": alphabet_bytes, "side_information_bytes": side,
        "scale_bytes": scale, "total_bytes": total,
    }
    for field, expected in expected_integers.items():
        if _integer(value.get(field), where=f"{where}.{field}") != expected:
            raise AnalysisReceiptError(f"{where}.{field} accounting differs")
    if value.get("alphabet_bytes_by_rate") != alphabet_by_rate:
        raise AnalysisReceiptError(f"{where}.alphabet_bytes_by_rate differs")
    fixed = {
        "family": "TCQ_E4M3_R256", "format": f"TCQ_E4M3_R{rate * 256}",
        "grid": "e4m3fn", "layout": "tight_offsets",
        "producer_eligible": False, "expanded_weight_resident_bytes": 0,
        "schedule_bits_per_code": 4,
        "schedule_scope": "tensor_input_column_shared_across_rows",
        "sidecar_header_bytes": 0, "superblock_weights": 256,
        "wire_header_bytes": 88, "wire_schema": "gridbook.trellis.wire.v1",
        "shape": list(shape), "scale_coding": bracket,
        "non_shipping_research": bracket == "two_tier",
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise AnalysisReceiptError(f"{where}.{field} differs")
    _close(value.get("body_bpw"), float(rate), where=f"{where}.body_bpw")
    _close(value.get("scale_bpw"), scale * 8 / (rows * columns), where=f"{where}.scale_bpw")
    _close(value.get("exact_bpw"), total * 8 / (rows * columns), where=f"{where}.exact_bpw")
    identity = _sha(value.get("identity_sha256"), where=f"{where}.identity_sha256")
    if bracket == "production_row_fp32":
        if (
            value.get("schema") != "prismaquant.trellis_tensor_payload.v1"
            or value.get("scale_contract") != "one_fp32_per_row (the E4M3 wire's own)"
        ):
            raise AnalysisReceiptError(f"{where} production contract differs")
        identity_body = {
            key: value[key] for key in _TCQ_BASE_FOOTPRINT_KEYS
            if key != "identity_sha256"
        }
        identity_body["scale_contract"] = "per_output_row_fp32"
        if identity != _compact_identity_sha256(identity_body):
            raise AnalysisReceiptError(f"{where}.identity_sha256 differs")
    else:
        nested = _exact_keys(
            value.get("production_payload_v1"), _TCQ_BASE_FOOTPRINT_KEYS,
            where=f"{where}.production_payload_v1",
        )
        nested_identity = _sha(
            nested.get("identity_sha256"),
            where=f"{where}.production_payload_v1.identity_sha256",
        )
        nested_body = {key: nested[key] for key in _TCQ_BASE_FOOTPRINT_KEYS if key != "identity_sha256"}
        if (
            nested_identity != _compact_identity_sha256(nested_body)
            or identity != nested_identity
            or value.get("scale_bytes_v1_production") != production_scale
            or value.get("schema")
            != "trellis.fp8_ladder.tcq_e4m3_two_tier_research_payload.v1"
            or value.get("scale_contract")
            != "group16_two_tier_9B_per_superblock (RESEARCH)"
            or not isinstance(value.get("research_pricing_note"), str)
            or not value["research_pricing_note"]
        ):
            raise AnalysisReceiptError(f"{where} two-tier repricing differs")
        allowed = {
            "schema", "scale_bytes", "scale_contract", "total_bytes",
            "exact_bpw",
        }
        for field in _TCQ_BASE_FOOTPRINT_KEYS - allowed:
            if value.get(field) != nested.get(field):
                raise AnalysisReceiptError(f"{where}.{field} changed during repricing")
    return total * 8


def _validate_nested_arms(per_tensor: Mapping[str, Mapping[str, object]]) -> None:
    for name, cell in per_tensor.items():
        shape = cell["shape"]
        planes: dict[str, set[str]] = {bracket: set() for bracket in BRACKETS}
        schedules: dict[tuple[str, int], list[Mapping[str, object]]] = {}
        alphabets: dict[tuple[str, str], list[Mapping[str, object]]] = {}
        for arm_name, arm in cell["arms"].items():
            if arm_name.startswith("fp8_cb_"):
                rung = int(arm_name.rsplit("@", 1)[1])
                learned = arm_name.startswith("fp8_cb_learned@")
                try:
                    footprint = _CHECKPOINT_CONTRACT._validate_fp8_footprint(
                        arm.get("footprint"), rung=rung, learned=learned,
                        shape=list(shape), where=f"{name}.{arm_name}.footprint",
                    )
                    if learned:
                        _CHECKPOINT_CONTRACT._validate_fp8_book(
                            arm.get("learned_book"), footprint=footprint,
                            where=f"{name}.{arm_name}.learned_book",
                        )
                except _CHECKPOINT_CONTRACT.CheckpointContractError as exc:
                    raise AnalysisReceiptError(str(exc)) from exc
                _cost_bits(arm, book_price="wire8")
                _cost_bits(arm, book_price="fp16_production")
                continue
            stem, rate_text = arm_name.rsplit("@", 1)
            _family, bracket, selector = stem.split(".")
            rate = int(rate_text)
            planes[bracket].add(_sha(
                arm.get("e4m3_plane_sha256"),
                where=f"{name}.{arm_name}.e4m3_plane_sha256",
            ))
            schedule = _validate_schedule(
                arm.get("schedule"), rate=rate, columns=int(shape[1]),
                where=f"{name}.{arm_name}.schedule",
            )
            codes = _validate_alphabet(
                arm.get("alphabet"), selector=selector,
                where=f"{name}.{arm_name}.alphabet",
            )
            schedules.setdefault((bracket, rate), []).append(arm["schedule"])
            alphabets.setdefault((bracket, selector), []).append(arm["alphabet"])
            bits = _validate_tcq_footprint(
                arm.get("footprint"), bracket=bracket, rate=rate,
                shape=shape, counts=schedule, codes=codes,
                where=f"{name}.{arm_name}.footprint",
            )
            if bits != _cost_bits(arm, book_price="wire8"):
                raise AnalysisReceiptError(f"{name}.{arm_name}: canonical integer cost differs")
        if any(len(values) != 1 for values in planes.values()):
            raise AnalysisReceiptError(f"{name}: E4M3 plane identity differs within bracket")
        if any(len(values) != 2 or values[0] != values[1] for values in schedules.values()):
            raise AnalysisReceiptError(f"{name}: schedule differs across alphabet selectors")
        if any(len(values) != 2 or values[0] != values[1] for values in alphabets.values()):
            raise AnalysisReceiptError(f"{name}: alphabet differs across nominal rates")


def _validate_source(
    source: Mapping[str, object], *, source_binding: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    settings = source.get("settings")
    per_tensor = source.get("per_tensor")
    if not isinstance(settings, Mapping) or not isinstance(per_tensor, Mapping):
        raise AnalysisReceiptError("source settings/per_tensor are malformed")
    manifest_path = settings.get("corpus_manifest")
    if not isinstance(manifest_path, str) or not manifest_path.startswith("/"):
        raise AnalysisReceiptError("settings.corpus_manifest must be absolute")
    manifest, manifest_binding = _strict_json_object(Path(manifest_path))
    entries, artifact_binding = _manifest_entries(
        manifest, binding=manifest_binding, settings=settings,
    )
    try:
        _PRODUCER.validate_report(
            source, settings=settings, entries=entries,
            require_complete=False, replay_envelope=True,
        )
    except (
        _PRODUCER.CampaignError,
        _EXECUTION_CONTRACT.NumericExecutionContractError,
    ) as exc:
        raise AnalysisReceiptError(f"producer contract: {exc}") from exc
    if (
        source.get("schema") != SOURCE_SCHEMA
        or source.get("status") != SOURCE_STATUS
        or source.get("partial") is not False
        or source.get("tensors_done") != 33
        or len(per_tensor) != 33
        or source.get("claim_boundary") != CLAIM_BOUNDARY
        or settings.get("population_counts") != {"dense": 9, "routed": 24}
    ):
        raise AnalysisReceiptError("source is not the closed 33-tensor campaign")
    environment = settings["environment"]
    if environment.get("physical_host") != "sparky":
        raise AnalysisReceiptError("source campaign did not execute on Sparky")
    _validate_source_closure(settings, source_path=str(source_binding["path"]))
    segments = source.get("execution_segments")
    if not isinstance(segments, list) or len(segments) != 1:
        raise AnalysisReceiptError("source must contain exactly one execution segment")
    _validate_attestation(segments[0], settings=settings)
    _validate_nested_arms(per_tensor)
    summaries = _population_summaries(per_tensor)
    if source.get("population_summaries") != summaries:
        raise AnalysisReceiptError(
            "source population summaries differ from canonical integer-bit recomputation"
        )
    return settings, {**manifest_binding, "artifact": artifact_binding}


def build_receipt(source_path: Path) -> dict[str, object]:
    verifier_binding = _read_bound_file(_VERIFIER_PATH)[1]
    dependency_bindings = {
        name: _read_bound_file(path)[1]
        for name, path in _DEPENDENCY_PATHS.items()
    }
    if verifier_binding != _IMPORT_VERIFIER_BINDING:
        raise AnalysisReceiptError("verifier bytes changed after module import")
    if dependency_bindings != _IMPORT_DEPENDENCY_BINDINGS:
        raise AnalysisReceiptError("validation dependency changed after module import")
    source, source_binding = _strict_json_object(source_path)
    settings, manifest_binding = _validate_source(
        source, source_binding=source_binding,
    )
    per_tensor = source["per_tensor"]
    body: dict[str, object] = {
        "schema": SCHEMA,
        "status": "verified_exact_recomputation",
        "source": {
            "path": source_binding["path"],
            "sha256": source_binding["sha256"],
            "size_bytes": source_binding["size_bytes"],
            "checkpoint_sha256": source["checkpoint_sha256"],
            "repo_git_commit": settings["environment"]["repo_git_commit"],
            "schema": source["schema"],
            "status": source["status"],
            "partial": source["partial"],
            "tensors_done": source["tensors_done"],
            "settings_identity_sha256": settings["identity_sha256"],
        },
        "population_counts": settings["population_counts"],
        "corpus_manifest": {
            "path": manifest_binding["path"],
            "sha256": manifest_binding["sha256"],
            "size_bytes": manifest_binding["size_bytes"],
            "corpus_file_sha256": settings["corpus_file_sha256"],
            "importance_value_sha256": settings["importance_value_sha256"],
            "corpus_prismaquant_commit": settings["corpus_prismaquant_commit"],
            "artifact": manifest_binding["artifact"],
        },
        "aggregation_contract": "dense/routed population-separated; no pooled field",
        "verdict_contract": (
            "strict exact-byte family coverage on every tensor under both scale "
            "and learned-book price brackets; crossing is NO_VERDICT"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "population_summaries": _population_summaries(per_tensor),
        "frontier_diagnostics": _frontier_diagnostics(per_tensor),
        "verifier": {
            "path": verifier_binding["path"],
            "sha256": verifier_binding["sha256"],
            "size_bytes": verifier_binding["size_bytes"],
            "dependencies": [
                {"name": name, **binding}
                for name, binding in sorted(dependency_bindings.items())
            ],
        },
    }
    return {**body, "receipt_sha256": _identity_sha256(body)}


def validate_receipt(receipt: Mapping[str, object]) -> None:
    expected_keys = {
        "schema", "status", "source", "population_counts",
        "corpus_manifest", "aggregation_contract", "verdict_contract",
        "claim_boundary", "population_summaries", "frontier_diagnostics",
        "verifier", "receipt_sha256",
    }
    _exact_keys(receipt, expected_keys, where="analysis receipt")
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "verified_exact_recomputation"
        or receipt.get("population_counts") != {"dense": 9, "routed": 24}
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise AnalysisReceiptError("analysis receipt semantic identity differs")
    digest = receipt.get("receipt_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise AnalysisReceiptError("analysis receipt digest is invalid")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if digest != _identity_sha256(body):
        raise AnalysisReceiptError("analysis receipt self-digest differs")
    source = _exact_keys(
        receipt.get("source"), {
            "path", "sha256", "size_bytes", "checkpoint_sha256",
            "repo_git_commit", "schema", "status", "partial", "tensors_done",
            "settings_identity_sha256",
        }, where="analysis receipt source",
    )
    source_path = source.get("path")
    if not isinstance(source_path, str) or not source_path.startswith("/"):
        raise AnalysisReceiptError("analysis receipt source path is invalid")
    expected = build_receipt(Path(source_path))
    if receipt != expected:
        raise AnalysisReceiptError("analysis receipt differs from exact recomputation")


def publish_receipt(output: Path, receipt: Mapping[str, object]) -> None:
    validate_receipt(receipt)
    output = output.absolute()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or output.is_symlink():
        raise AnalysisReceiptError(f"receipt output already exists: {output}")
    if partial.exists() or partial.is_symlink():
        raise AnalysisReceiptError(f"receipt partial already exists: {partial}")
    try:
        with exclusive_publication_claim(output, identity=receipt):
            atomic_checkpoint_json(partial, receipt)
            publish_file_no_replace(partial, output)
    except PublicationError as exc:
        raise AnalysisReceiptError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build_receipt(args.source)
    publish_receipt(args.out, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
