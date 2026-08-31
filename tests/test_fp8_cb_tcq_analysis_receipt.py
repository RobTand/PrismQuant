from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import pytest


_PATH = Path(__file__).resolve().parents[1] / (
    "research/trellis_e2m1_highrate_2026-08-30/"
    "fp8_cb_tcq_analysis_receipt.py"
)
_SPEC = importlib.util.spec_from_file_location("fp8_cb_tcq_analysis_receipt", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
M = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = M
sys.path.insert(0, str(_PATH.parent))
try:
    _SPEC.loader.exec_module(M)
finally:
    sys.path.pop(0)


def _names() -> list[str]:
    return sorted(
        [
            f"model.language_model.layers.{layer}.mlp.{projection}.weight"
            for layer in (0, 1, 2)
            for projection in ("down_proj", "gate_proj", "up_proj")
        ]
        + [
            f"model.language_model.layers.{layer}.mlp.experts.0.{projection}.weight"
            for layer in (3, 9, 15, 21, 26, 32, 38, 44)
            for projection in ("down_proj", "gate_proj", "up_proj")
        ]
    )


def _entry(name: str, index: int) -> dict:
    population, layer, projection, expert = M._classify_name(name)
    shape = M._expected_shape(population, projection)
    qname = (
        name.removesuffix(".weight")
        if population == "dense"
        else name.split(".experts.0.", 1)[0]
        + ".experts."
        + ("down_proj" if projection == "down_proj" else "gate_up_proj")
    )
    return {
        "name": name,
        "population": population,
        "layer": layer,
        "projection": projection,
        "expert": expert,
        "source_weight_dtype": "torch.bfloat16",
        "source_weight_shape": list(shape),
        "source_weight_sha256": f"{index + 1:064x}",
        "importance_key": f"__bf16_importance__.{name}",
        "importance_shape": [shape[1]],
        "importance_dtype": "torch.float32",
        "importance_sha256": f"{index + 101:064x}",
        "importance_source": {
            "qname": qname,
            "expert": expert,
            "denominator_name": (
                "expert_tokens" if population == "routed" else "n_tokens_seen"
            ),
            "denominator": 8,
        },
        "census": {"distinct_source_values": 16, "numel": math.prod(shape)},
    }


def _fp8_footprint(rung: int, learned: bool, shape: tuple[int, int]) -> dict:
    C = M._CHECKPOINT_CONTRACT
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
        "body_bpw": body_bits / numel,
        "exact_bpw": total_bits / numel,
        "body_bits": body_bits,
        "total_bits": total_bits,
        "total_bytes": total_bits // 8,
        "superblocks": superblocks,
        "type_size_bytes_per_superblock": index_bytes,
        "index_bytes_per_superblock": index_bytes,
        "scale_bytes_per_superblock": 0,
        "row_scale_bytes": row_scale_bytes,
        "scale_bits": scale_bits,
        "scale_bpw": scale_bits / numel,
        "scale_coding": "v1",
        "scale_contract": "per_output_row_fp32",
        "backed_on_sm120": True,
        "sidecar_amortization": C._FP8_SIDECAR_AMORTIZATION,
    })
    if learned:
        result.update({
            "fixed_lattice_is_format_shared": False,
            "learned_book_elements": elements,
            "learned_book_n_sub": 4,
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


def _book(rung: int) -> dict:
    rows = 1 << (rung // 4)
    return {
        "elements": rows * 8,
        "tables": [
            {
                "amax": 1.0,
                "distinct_levels": rows,
                "sha256": str(index) * 64,
                "shape": [rows, 2],
            }
            for index in range(1, 5)
        ],
    }


def _alphabet(selector: str) -> dict:
    return {
        "alphabet_mode": selector,
        "rule": f"fixture {selector}",
        "tcq_native_codes": {
            str(rate): [value % 256 for value in range(1 << (rate + 1))]
            for rate in range(1, 8)
        },
    }


def _schedule(rate: int, columns: int) -> dict:
    return {
        "achieved_rate": float(rate),
        "body_bits_per_block_max": rate * 256,
        "body_bits_per_block_min": rate * 256,
        "body_bits_per_block_std": 0.0,
        "counts": {
            str(value): columns if value == rate else 0 for value in range(1, 9)
        },
        "fixed_quota_per_256": False,
        "invert": False,
        "maximum_rate": 8,
        "minimum_trellis_steps": 256,
        "schedule_sha256": f"{rate:064x}",
        "tailbite_guard_fixups": 0,
        "target_rate": float(rate),
        "transitions_per_block_max": 1,
        "transitions_per_block_mean": 1.0,
    }


def _production_payload(rate: int, shape: tuple[int, int], alphabet: dict) -> dict:
    rows, columns = shape
    body_bits_per_row = rate * columns
    unpadded = (body_bits_per_row + 7) // 8
    stride = ((unpadded + 15) // 16) * 16
    body_bytes = rows * stride
    block_count = (columns + 255) // 256
    schedule_bytes = (columns * 4 + 7) // 8
    offset_bytes = (block_count + 1) * 4
    alphabet_by_rate = {
        str(rate): 3 + len(alphabet["tcq_native_codes"][str(rate)])
    }
    alphabet_bytes = sum(alphabet_by_rate.values())
    scale_bytes = rows * 4
    side = 88 + schedule_bytes + offset_bytes + alphabet_bytes
    total = body_bytes + scale_bytes + side
    body = {
        "schema": "prismaquant.trellis_tensor_payload.v1",
        "wire_schema": "gridbook.trellis.wire.v1",
        "family": "TCQ_E4M3_R256",
        "format": f"TCQ_E4M3_R{rate * 256}",
        "grid": "e4m3fn",
        "shape": list(shape),
        "body_rate_q256": rate * 256,
        "body_bpw": float(rate),
        "layout": "tight_offsets",
        "superblock_weights": 256,
        "block_count": block_count,
        "body_bits_per_row": body_bits_per_row,
        "unpadded_body_bytes_per_row": unpadded,
        "body_row_stride_bytes": stride,
        "body_padding_bytes": rows * (stride - unpadded),
        "body_bytes": body_bytes,
        "wire_header_bytes": 88,
        "scale_contract": "per_output_row_fp32",
        "scale_bytes": scale_bytes,
        "schedule_scope": "tensor_input_column_shared_across_rows",
        "schedule_bits_per_code": 4,
        "schedule_bytes": schedule_bytes,
        "block_offset_bits": 32,
        "block_offset_bytes": offset_bytes,
        "alphabet_bytes_by_rate": alphabet_by_rate,
        "alphabet_bytes": alphabet_bytes,
        "sidecar_header_bytes": 0,
        "side_information_bytes": side,
        "total_bytes": total,
        "exact_bpw": total * 8 / (rows * columns),
        "expanded_weight_resident_bytes": 0,
        "producer_eligible": False,
    }
    return {**body, "identity_sha256": M._compact_identity_sha256(body)}


def _tcq_footprint(
    rate: int, bracket: str, shape: tuple[int, int], alphabet: dict,
) -> dict:
    production = _production_payload(rate, shape, alphabet)
    rows, columns = shape
    if bracket == "production_row_fp32":
        result = copy.deepcopy(production)
        result.update({
            "scale_coding": bracket,
            "scale_contract": "one_fp32_per_row (the E4M3 wire's own)",
            "scale_bpw": rows * 4 * 8 / (rows * columns),
            "non_shipping_research": False,
        })
        return result
    scale = rows * (columns // 256) * 9
    total = production["total_bytes"] - production["scale_bytes"] + scale
    result = copy.deepcopy(production)
    result.update({
        "schema": "trellis.fp8_ladder.tcq_e4m3_two_tier_research_payload.v1",
        "non_shipping_research": True,
        "scale_coding": bracket,
        "scale_contract": "group16_two_tier_9B_per_superblock (RESEARCH)",
        "scale_bytes": scale,
        "scale_bytes_v1_production": production["scale_bytes"],
        "scale_bpw": scale * 8 / (rows * columns),
        "total_bytes": total,
        "exact_bpw": total * 8 / (rows * columns),
        "production_payload_v1": production,
        "research_pricing_note": "fixture penalty bracket",
    })
    return result


def _cell(entry: dict) -> dict:
    shape = tuple(entry["source_weight_shape"])
    arms = {}
    for rate, rung in M.CELL_MAP.items():
        for learned in (False, True):
            snr = 20.0 + rate + (0.1 if learned else 0.0)
            nsse = 10 ** (-snr / 10)
            name = f"fp8_cb_{'learned' if learned else 'fixed'}@{rung}"
            arm = {
                "encode_seconds_observation_not_perf_claim": 1.0,
                "weighted_sse": nsse,
                "weighted_nsse": nsse,
                "weighted_snr_db": -10 * math.log10(nsse),
                "reconstruction_sha256": "a" * 64,
                "footprint": _fp8_footprint(rung, learned, shape),
                "family": "FP8_CB_K",
                "rung": rung,
                "encode_tier": "balanced",
                "book_kind": (
                    "per_tensor_weighted_lloyd" if learned else "fixed_lattice"
                ),
            }
            if learned:
                arm["learned_book"] = _book(rung)
            arms[name] = arm
    for bracket in M.BRACKETS:
        plane = "b" * 64 if bracket == "production_row_fp32" else "c" * 64
        for selector in M.SELECTORS:
            alphabet = _alphabet(selector)
            for rate in M.RATES:
                snr = 21.0 + rate + (1.0 if selector == "exact_dp" else 0.0)
                nsse = 10 ** (-snr / 10)
                name = f"tcq_e4m3.{bracket}.{selector}@{rate}"
                arms[name] = {
                    "encode_seconds_observation_not_perf_claim": 1.0,
                    "weighted_sse": nsse,
                    "weighted_nsse": nsse,
                    "weighted_snr_db": -10 * math.log10(nsse),
                    "reconstruction_sha256": "d" * 64,
                    "footprint": _tcq_footprint(rate, bracket, shape, alphabet),
                    "family": "TCQ_E4M3_R256",
                    "rate": float(rate),
                    "trellis_scale_bracket": bracket,
                    "alphabet_selector": selector,
                    "e4m3_plane_sha256": plane,
                    "alphabet": alphabet,
                    "schedule": _schedule(rate, shape[1]),
                }
    return {
        "population": entry["population"],
        "shape": list(shape),
        "source_weight_sha256": entry["source_weight_sha256"],
        "importance_sha256": entry["importance_sha256"],
        "importance_source": copy.deepcopy(entry["importance_source"]),
        "metric_weight_sha256": "e" * 64,
        "weighted_energy": 1.0,
        "arms": arms,
    }


def _execution_environment(repo_root: Path) -> dict:
    E = M._EXECUTION_CONTRACT
    return {
        "schema": "trellis.numeric_execution.v2",
        "physical_host": "sparky",
        "uts_hostname": "sparky",
        "gpu_uuid": "GPU-e76c7efc-c157-b1f4-1348-83e4eb5092f4",
        "container_image_reference": E.CAMPAIGN_IMAGE_REFERENCE,
        "container_image_digest": E.CAMPAIGN_IMAGE_DIGEST,
        "container_image_id": E.CAMPAIGN_IMAGE_DIGEST,
        "container_image_evidence": "host_docker_daemon_inspect_before_start",
        "container_image_in_process_verification": "not_available",
        "container_user": "1000:1000",
        "ipc_mode": "private",
        "repo_root": str(repo_root),
        "source_mount_evidence": "host_docker_daemon_inspect_readonly_repo_and_git",
        "repo_git_commit": M.EXPECTED_PRODUCER_COMMIT,
        "repo_tree_clean": True,
        "python": "3.12.3",
        "torch": "2.13.0+cu130",
        "triton": "3.7.1",
        "device": "NVIDIA GB10",
    }


def _reseal_source(path: Path, *, summaries: bool = False) -> None:
    document = json.loads(path.read_text())
    if summaries:
        document["population_summaries"] = M._population_summaries(
            document["per_tensor"]
        )
    body = {key: value for key, value in document.items() if key != "checkpoint_sha256"}
    document["checkpoint_sha256"] = M._identity_sha256(body)
    path.write_text(json.dumps(document))


def _source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo_root = tmp_path / "repo"
    active_files, active_hashes = {}, {}
    for label, suffix in M._ACTIVE_SOURCE_LABEL_SUFFIX.items():
        path = Path(str(repo_root) + suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {label}\n")
        digest = M._stable_file_sha256(path)
        active_files[label] = {"path": str(path), "sha256": digest}
        active_hashes[suffix] = digest
    monkeypatch.setattr(M, "_ACTIVE_SOURCE_SUFFIX_HASHES", active_hashes)

    frozen, frozen_hashes = {}, {}
    for suffix in M._FROZEN_SOURCE_SUFFIXES:
        path = tmp_path / "sources" / suffix.removeprefix("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {suffix}\n")
        digest = M._stable_file_sha256(path)
        frozen[str(path)] = digest
        frozen_hashes[suffix] = digest
    monkeypatch.setattr(M, "_FROZEN_SOURCE_SUFFIX_HASHES", frozen_hashes)
    snapshot_member = next(
        Path(path) for path in frozen
        if path.endswith(
            "/stage6_prismaquant_snapshot/prismaquant/cb_layout.py"
        )
    )
    monkeypatch.setattr(
        M, "EXPECTED_SNAPSHOT_TREE_SHA256",
        M._snapshot_tree_sha256(snapshot_member.parents[1]),
    )

    locked, locked_hashes = {}, {}
    for stem in ("fp8_ladder", "hull_sweep", "e4m3_alphabet_dp"):
        path = tmp_path / "locked" / f"{stem}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {stem}\n")
        digest = M._stable_file_sha256(path)
        locked[f"{stem}_path"] = str(path)
        locked[f"{stem}_sha256"] = digest
        locked_hashes[f"{stem}_sha256"] = digest
    monkeypatch.setattr(M, "_LOCKED_EXPECTED_HASHES", locked_hashes)

    entries = [_entry(name, index) for index, name in enumerate(_names())]
    artifact = tmp_path / "corpus.bin"
    artifact.write_bytes(b"closed corpus fixture")
    artifact_sha = M._stable_file_sha256(artifact)
    importance_sha = "8" * 64
    manifest = {key: {} for key in M._MANIFEST_KEYS}
    manifest.update({
        "schema": "trellis.bf16_corpus.v2",
        "status": "finalized",
        "generated": "fixture",
        "host": "fixture",
        "corpus_label": "fixture",
        "model_profile": "glm5_next",
        "model": "fixture",
        "model_config_sha256": "7" * 64,
        "num_hidden_layers": 45,
        "layers": [0, 1, 2, 3, 9, 15, 21, 26, 32, 38, 44],
        "roles": ["gate_proj", "up_proj", "down_proj"],
        "expert": 0,
        "calibration": {"fixture": True},
        "importance_identity": {
            "schema": "prismaquant.glm_trellis_importance.probe_imatrix.v1",
            "probe_file_sha256": "1" * 64,
            "probe_calibration_hash": "fixture",
            "probe_imatrix_value_sha256": "2" * 64,
            "value_sha256": importance_sha,
            "dense_normalization": "fixture",
            "routed_normalization": "fixture",
            "gate_up_mapping": "fixture",
            "down_mapping": "fixture",
        },
        "reader_contract": {"fixture": True},
        "prismaquant_commit": M.EXPECTED_CORPUS_PRODUCER_COMMIT,
        "file": artifact.name,
        "file_size_bytes": artifact.stat().st_size,
        "file_sha256": artifact_sha,
        "populations": {
            "dense": {"count": 9, "layers": [0, 1, 2]},
            "routed": {
                "count": 24, "layers": [3, 9, 15, 21, 26, 32, 38, 44]
            },
        },
        "source_artifact": {"fixture": True},
        "entries": entries,
    })
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = M._stable_file_sha256(manifest_path)
    monkeypatch.setattr(M, "EXPECTED_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setattr(M, "EXPECTED_MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(M, "EXPECTED_CORPUS_FILE_SHA256", artifact_sha)
    monkeypatch.setattr(M, "EXPECTED_IMPORTANCE_VALUE_SHA256", importance_sha)

    result_path = tmp_path / "result.json"
    environment = _execution_environment(repo_root)
    command = [
        active_files["driver"]["path"], "--manifest", str(manifest_path),
        "--out", str(result_path),
    ]
    imported_suffixes = {
        "H": "/trellis-hull-20260828/hull_sweep.py",
        "C": "/trellis-stage0/stage5_e4m3_codec.py",
        "W": "/trellis-stage0/stage6_worker.py",
        "P": "/trellis-stage0/tcq_pilot.py",
        "S4": "/trellis-stage0/stage4_place.py",
        "TF": "/trellis-stage0/stage6_prismaquant_snapshot/prismaquant/trellis_formats.py",
    }
    imported = {}
    for label, suffix in imported_suffixes.items():
        path = next(path for path in frozen if path.endswith(suffix))
        imported[label] = {"path": path, "sha256": frozen[path]}
    settings = {
        "schema": M.SOURCE_SCHEMA,
        "corpus_manifest": str(manifest_path),
        "corpus_manifest_sha256": manifest_sha,
        "corpus_file_sha256": artifact_sha,
        "importance_value_sha256": importance_sha,
        "corpus_prismaquant_commit": M.EXPECTED_CORPUS_PRODUCER_COMMIT,
        "population_counts": {"dense": 9, "routed": 24},
        "rungs": [32, 40],
        "rates": [4.0, 5.0],
        "cell_map": {"4": 32, "5": 40},
        "trellis_scale_brackets": list(M.BRACKETS),
        "alphabet_selectors": list(M.SELECTORS),
        "book_price_brackets": list(M.BOOK_PRICES),
        "encode_tier": "balanced",
        "locked_sources": locked,
        "frozen_codec_closure": {
            "snapshot_tree_sha256": M.EXPECTED_SNAPSHOT_TREE_SHA256,
            "source_sha256": frozen,
            "imported_codec_modules": imported,
        },
        "active_source_identity": {
            "repo_root": str(repo_root),
            "repo_git_commit": M.EXPECTED_PRODUCER_COMMIT,
            "files": active_files,
        },
        "environment": environment,
        "command": command,
        "claim_boundary": M.CLAIM_BOUNDARY,
    }
    settings["identity_sha256"] = M._identity_sha256(settings)

    attestation_path = tmp_path / "run/attestation/launch-attestation.json"
    attestation_path.parent.mkdir(parents=True)
    monkeypatch.setattr(M._EXECUTION_CONTRACT, "CAMPAIGN_STORAGE_ROOT", str(tmp_path))
    container_id = "f" * 64
    launch = [
        "/usr/bin/python3", "-B",
        f"{repo_root}/research/trellis_e2m1_highrate_2026-08-30/numeric_profiled_launcher.py",
        "--profile", str(tmp_path / "run/profile/output.json"), "--",
        "/usr/bin/python3", "-B", *command,
    ]
    attestation_body = {
        "schema": "trellis.numeric_launch_attestation.v1",
        "verification_scope": "host_docker_daemon_inspect_before_start",
        "physical_host": "sparky",
        "uts_hostname": "sparky",
        "gpu_uuid": environment["gpu_uuid"],
        "container_id": container_id,
        "container_hostname": container_id[:12],
        "container_state": "created",
        "container_rootfs_changes": [],
        "container_user": "1000:1000",
        "image_reference": environment["container_image_reference"],
        "image_digest": environment["container_image_digest"],
        "image_id": environment["container_image_id"],
        "uts_mode": "host",
        "network_mode": "none",
        "ipc_mode": "private",
        "gpu_request": "one_or_all_gpu_device_request",
        "launch_attestation_container_path": str(attestation_path),
        "repo_root": str(repo_root),
        "git_common_dir": str(tmp_path / "git"),
        "repo_mount_readonly": True,
        "git_mount_readonly": True,
        "storage_mount_readwrite": True,
        "rootfs_readonly": True,
        "runtime_isolation": "fixture",
        "launch_environment": {
            "HULL_PHYSICAL_HOST": "sparky",
            "HULL_REPO_ROOT": str(repo_root),
            "HULL_CONTAINER_IMAGE": environment["container_image_digest"],
            "HULL_LAUNCH_ATTESTATION": str(attestation_path),
        },
        "launch_command": launch,
        "launch_command_sha256": M._compact_identity_sha256(launch),
    }
    attestation = {
        **attestation_body,
        "attestation_sha256": M._compact_identity_sha256(attestation_body),
    }
    attestation_path.write_bytes(M.canonical_json_bytes(attestation))
    monkeypatch.setattr(
        M, "EXPECTED_ATTESTATION_FILE_SHA256", M._stable_file_sha256(attestation_path),
    )
    monkeypatch.setattr(
        M, "EXPECTED_ATTESTATION_IDENTITY_SHA256", attestation["attestation_sha256"],
    )
    monkeypatch.setattr(M, "EXPECTED_CONTAINER_ID", container_id)
    segment_body = {
        "schema": "trellis.numeric_execution_segment.v1",
        "physical_host": "sparky",
        "container_id": container_id,
        "image_id": environment["container_image_id"],
        "gpu_uuid": environment["gpu_uuid"],
        "launch_attestation_path": str(attestation_path),
        "launch_attestation_sha256": attestation["attestation_sha256"],
        "launch_command_sha256": attestation["launch_command_sha256"],
    }
    segment = {
        **segment_body,
        "segment_sha256": M._compact_identity_sha256(segment_body),
    }
    per_tensor = {entry["name"]: _cell(entry) for entry in entries}
    body = {
        "schema": M.SOURCE_SCHEMA,
        "settings": settings,
        "started_at_unix_s": 1.0,
        "per_tensor": per_tensor,
        "partial": False,
        "tensors_done": 33,
        "execution_segments": [segment],
        "completed_at_unix_s": 2.0,
        "population_summaries": M._population_summaries(per_tensor),
        "status": M.SOURCE_STATUS,
        "claim_boundary": M.CLAIM_BOUNDARY,
    }
    document = {**body, "checkpoint_sha256": M._identity_sha256(body)}
    result_path.write_text(json.dumps(document))
    monkeypatch.setattr(M, "EXPECTED_RESULT_PATH", str(result_path))
    monkeypatch.setattr(
        M, "EXPECTED_RESULT_SHA256", M._stable_file_sha256(result_path),
    )
    monkeypatch.setattr(M, "EXPECTED_RESULT_SIZE_BYTES", result_path.stat().st_size)
    monkeypatch.setattr(
        M, "EXPECTED_CHECKPOINT_SHA256", document["checkpoint_sha256"],
    )
    monkeypatch.setattr(
        M, "EXPECTED_SETTINGS_IDENTITY_SHA256", settings["identity_sha256"],
    )
    return result_path


def test_exact_frontiers_are_recomputed_and_self_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    source = _source(tmp_path, monkeypatch)
    receipt = M.build_receipt(source)
    M.validate_receipt(receipt)
    assert receipt["population_counts"] == {"dense": 9, "routed": 24}
    assert receipt["source"]["sha256"] == M._stable_file_sha256(source)
    diagnostic = receipt["frontier_diagnostics"]["dense"][0]
    assert diagnostic["tcq_best_quality_higher"] == 9
    assert diagnostic["cb_minimum_bpw_lower"] == 9


@pytest.mark.parametrize(
    "attack", ["book_kind", "tensor_name", "negative_bits", "extra_footprint", "bpw"],
)
def test_hostile_resigned_source_contract_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str,
):
    source = _source(tmp_path, monkeypatch)
    document = json.loads(source.read_text())
    first_name = next(iter(document["per_tensor"]))
    cell = document["per_tensor"][first_name]
    if attack == "book_kind":
        cell["arms"]["fp8_cb_learned@32"]["book_kind"] = "fixed_lattice"
    elif attack == "tensor_name":
        document["per_tensor"] = {
            "forged-tensor": cell,
            **{
                key: value for key, value in document["per_tensor"].items()
                if key != first_name
            },
        }
    elif attack == "negative_bits":
        cell["arms"]["fp8_cb_fixed@32"]["footprint"]["total_bits"] = -1
    elif attack == "extra_footprint":
        cell["arms"]["fp8_cb_fixed@32"]["footprint"]["claim"] = True
    else:
        cell["arms"]["fp8_cb_fixed@32"]["footprint"]["exact_bpw"] += 0.01
    body = {key: value for key, value in document.items() if key != "checkpoint_sha256"}
    document["checkpoint_sha256"] = M._identity_sha256(body)
    source.write_text(json.dumps(document))
    with pytest.raises(M.AnalysisReceiptError):
        M.build_receipt(source)


@pytest.mark.parametrize("bad_total", [True, 1.5, "1", -1])
def test_canonical_integer_cost_types_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_total: object,
):
    source = _source(tmp_path, monkeypatch)
    document = json.loads(source.read_text())
    cell = next(iter(document["per_tensor"].values()))
    cell["arms"]["fp8_cb_fixed@32"]["footprint"]["total_bits"] = bad_total
    source.write_text(json.dumps(document))
    _reseal_source(source)
    with pytest.raises(M.AnalysisReceiptError):
        M.build_receipt(source)


def test_inconsistent_byte_and_learned_side_costs_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    for field in ("total_bytes", "codebook_side_bits_wire8"):
        source = _source(tmp_path / field, monkeypatch)
        document = json.loads(source.read_text())
        cell = next(iter(document["per_tensor"].values()))
        arm = (
            cell["arms"]["fp8_cb_fixed@32"]
            if field == "total_bytes"
            else cell["arms"]["fp8_cb_learned@32"]
        )
        arm["footprint"][field] = -1
        source.write_text(json.dumps(document))
        _reseal_source(source)
        with pytest.raises(M.AnalysisReceiptError):
            M.build_receipt(source)


def test_attestation_and_cross_arm_resigned_attacks_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    source = _source(tmp_path, monkeypatch)
    document = json.loads(source.read_text())
    attestation = Path(document["execution_segments"][0]["launch_attestation_path"])
    attestation.write_text("{}\n")
    with pytest.raises(M.AnalysisReceiptError):
        M.build_receipt(source)

    source = _source(tmp_path / "second", monkeypatch)
    document = json.loads(source.read_text())
    cell = next(iter(document["per_tensor"].values()))
    cell["arms"]["tcq_e4m3.production_row_fp32.exact_dp@4"][
        "e4m3_plane_sha256"
    ] = "0" * 64
    source.write_text(json.dumps(document))
    _reseal_source(source, summaries=True)
    with pytest.raises(M.AnalysisReceiptError, match="plane identity"):
        M.build_receipt(source)


def test_imported_codec_label_permutation_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    source = _source(tmp_path, monkeypatch)
    document = json.loads(source.read_text())
    imported = document["settings"]["frozen_codec_closure"][
        "imported_codec_modules"
    ]
    imported["H"], imported["C"] = imported["C"], imported["H"]
    unsigned = {
        key: value for key, value in document["settings"].items()
        if key != "identity_sha256"
    }
    document["settings"]["identity_sha256"] = M._identity_sha256(unsigned)
    source.write_text(json.dumps(document))
    _reseal_source(source)
    with pytest.raises(M.AnalysisReceiptError, match="imported codec"):
        M.build_receipt(source)


@pytest.mark.parametrize("forged_nsse", [-1e-16, 1e-16])
def test_resigned_metric_tolerance_verdict_flip_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forged_nsse: float,
):
    source = _source(tmp_path, monkeypatch)
    document = json.loads(source.read_text())
    baseline = M._population_summaries(document["per_tensor"])
    assert all(
        cell["verdict"] == "NO_VERDICT_brackets_disagree_or_frontiers_cross"
        for population in baseline.values()
        for cell in population["cells"]
    )
    for cell in document["per_tensor"].values():
        for rung in M.CELL_MAP.values():
            arm = cell["arms"][f"fp8_cb_fixed@{rung}"]
            arm["weighted_sse"] = 0.0
            arm["weighted_nsse"] = forged_nsse
            arm["weighted_snr_db"] = -10.0 * math.log10(
                max(forged_nsse, 1e-300)
            )
    attacked = M._population_summaries(document["per_tensor"])
    assert all(
        cell["verdict"] == "FP8_CB"
        for population in attacked.values()
        for cell in population["cells"]
    )
    document["population_summaries"] = attacked
    body = {
        key: value for key, value in document.items()
        if key != "checkpoint_sha256"
    }
    document["checkpoint_sha256"] = M._identity_sha256(body)
    source.write_text(json.dumps(document))
    with pytest.raises(M.AnalysisReceiptError, match="weighted_nsse"):
        M.build_receipt(source)


@pytest.mark.parametrize("field", ["python", "torch", "triton"])
def test_resigned_runtime_version_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
):
    source = _source(tmp_path, monkeypatch)
    document = json.loads(source.read_text())
    document["settings"]["environment"][field] = "forged"
    unsigned = {
        key: value for key, value in document["settings"].items()
        if key != "identity_sha256"
    }
    document["settings"]["identity_sha256"] = M._identity_sha256(unsigned)
    source.write_text(json.dumps(document))
    _reseal_source(source)
    with pytest.raises(M.AnalysisReceiptError, match="runtime versions"):
        M.build_receipt(source)


def test_algebraically_valid_alternative_metric_artifact_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    source = _source(tmp_path, monkeypatch)
    document = json.loads(source.read_text())
    for cell in document["per_tensor"].values():
        for rung in M.CELL_MAP.values():
            arm = cell["arms"][f"fp8_cb_fixed@{rung}"]
            arm["weighted_sse"] = 1e-10
            arm["weighted_nsse"] = 1e-10
            arm["weighted_snr_db"] = -10.0 * math.log10(1e-10)
    source.write_text(json.dumps(document))
    _reseal_source(source, summaries=True)
    with pytest.raises(M.AnalysisReceiptError, match="exact final result"):
        M.build_receipt(source)


def test_resigned_receipt_schema_and_summary_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    receipt = M.build_receipt(_source(tmp_path, monkeypatch))
    for field in ("schema", "population_summaries"):
        changed = copy.deepcopy(receipt)
        if field == "schema":
            changed[field] = "forged.schema.v999"
        else:
            changed[field]["dense"]["tensors"] = 8
        body = {key: value for key, value in changed.items() if key != "receipt_sha256"}
        changed["receipt_sha256"] = M._identity_sha256(body)
        with pytest.raises(M.AnalysisReceiptError):
            M.validate_receipt(changed)


def test_no_replace_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    receipt = M.build_receipt(_source(tmp_path, monkeypatch))
    output = tmp_path / "analysis-receipt.json"
    M.publish_receipt(output, receipt)
    assert json.loads(output.read_text()) == receipt
    with pytest.raises(M.AnalysisReceiptError, match="already exists"):
        M.publish_receipt(output, receipt)


def test_bound_reader_rejects_symlink_and_duplicate_json(tmp_path: Path):
    target = tmp_path / "target.json"
    target.write_text('{"x":1}\n')
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(M.AnalysisReceiptError, match="cannot open bound file"):
        M._strict_json_object(link)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}\n')
    with pytest.raises(M.AnalysisReceiptError, match="duplicate JSON member"):
        M._strict_json_object(duplicate)


def test_bound_reader_rejects_path_swap_during_one_fd_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source.json"
    source.write_text('{"x":1}\n')
    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"x":2}\n')
    displaced = tmp_path / "displaced.json"
    real_read = os.read
    swapped = False

    def swapping_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        block = real_read(descriptor, size)
        if block and not swapped:
            swapped = True
            source.rename(displaced)
            replacement.rename(source)
        return block

    monkeypatch.setattr(M.os, "read", swapping_read)
    with pytest.raises(M.AnalysisReceiptError, match="identity changed"):
        M._strict_json_object(source)


def test_bound_reader_requires_nofollow_and_enforces_size_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source.json"
    source.write_text("{}\n")
    monkeypatch.delattr(M.os, "O_NOFOLLOW")
    with pytest.raises(M.AnalysisReceiptError, match="O_NOFOLLOW is required"):
        M._read_bound_file(source)
    monkeypatch.undo()
    monkeypatch.setattr(M, "_MAX_BOUND_BYTES", 2)
    with pytest.raises(M.AnalysisReceiptError, match="exceeds 2 bytes"):
        M._read_bound_file(source)


def test_verifier_drift_after_import_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    source = _source(tmp_path, monkeypatch)
    changed = dict(M._IMPORT_VERIFIER_BINDING)
    changed["sha256"] = "f" * 64
    monkeypatch.setattr(M, "_IMPORT_VERIFIER_BINDING", changed)
    with pytest.raises(M.AnalysisReceiptError, match="changed after module import"):
        M.build_receipt(source)
