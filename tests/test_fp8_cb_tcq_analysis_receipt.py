from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import pytest


_PATH = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/"
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


def _arm(*, family: str, snr: float, bpw: float, numel: int) -> dict:
    nsse = 10 ** (-snr / 10)
    bits = round(bpw * numel)
    footprint = {"total_bits": bits, "exact_bpw": bits / numel}
    arm = {
        "family": family,
        "weighted_sse": nsse,
        "weighted_nsse": nsse,
        "weighted_snr_db": -10 * math.log10(nsse),
        "footprint": footprint,
    }
    return arm


def _cell(population: str) -> dict:
    numel = 1_000_000
    arms = {}
    for rate, rung in M.CELL_MAP.items():
        fixed = _arm(family="FP8_CB_K", snr=20 + rate, bpw=rate, numel=numel)
        fixed["book_kind"] = "fixed_lattice"
        learned = _arm(
            family="FP8_CB_K", snr=20.1 + rate, bpw=rate + 0.002,
            numel=numel,
        )
        learned["book_kind"] = "per_tensor_weighted_lloyd"
        learned["footprint"].update({
            "codebook_side_bits": 2_000,
            "codebook_side_bits_wire8": 1_000,
            "exact_bpw_book_wire8": (
                learned["footprint"]["total_bits"] - 1_000
            ) / numel,
        })
        arms[f"fp8_cb_fixed@{rung}"] = fixed
        arms[f"fp8_cb_learned@{rung}"] = learned
        for bracket in M.BRACKETS:
            for selector, bonus in (("lloyd", 0.0), ("exact_dp", 1.0)):
                arms[f"tcq_e4m3.{bracket}.{selector}@{rate}"] = _arm(
                    family="TCQ_E4M3_R256",
                    snr=21 + rate + bonus,
                    bpw=rate + (0.001 if bracket == "production_row_fp32" else 0.25),
                    numel=numel,
                )
    return {
        "population": population,
        "shape": [1000, 1000],
        "weighted_energy": 1.0,
        "arms": arms,
    }


def _source(tmp_path: Path) -> Path:
    per_tensor = {
        **{f"dense-{i}": _cell("dense") for i in range(9)},
        **{f"routed-{i}": _cell("routed") for i in range(24)},
    }
    summaries = M._population_summaries(per_tensor)
    active_file = tmp_path / "producer.py"
    active_file.write_text("# bound producer\n")
    attestation = tmp_path / "launch-attestation.json"
    attestation.write_text("{}\n")
    environment = {
        "repo_git_commit": "a" * 40,
        "repo_tree_clean": True,
        "physical_host": "sparky",
        "container_image_id": "sha256:image",
        "gpu_uuid": "GPU-test",
    }
    settings = {
        "schema": M.SOURCE_SCHEMA,
        "population_counts": {"dense": 9, "routed": 24},
        "rungs": [32, 40],
        "rates": [4.0, 5.0],
        "cell_map": {"4": 32, "5": 40},
        "trellis_scale_brackets": list(M.BRACKETS),
        "alphabet_selectors": list(M.SELECTORS),
        "book_price_brackets": list(M.BOOK_PRICES),
        "encode_tier": "balanced",
        "claim_boundary": M.CLAIM_BOUNDARY,
        "environment": environment,
        "active_source_identity": {
            "repo_git_commit": "a" * 40,
            "files": {
                "driver": {
                    "path": str(active_file.resolve()),
                    "sha256": M.file_sha256(active_file),
                }
            },
        },
    }
    settings["identity_sha256"] = M._identity_sha256(settings)
    segment_body = {
        "schema": "trellis.numeric_execution_segment.v1",
        "physical_host": "sparky",
        "container_id": "c" * 64,
        "image_id": environment["container_image_id"],
        "gpu_uuid": environment["gpu_uuid"],
        "launch_attestation_path": str(attestation.resolve()),
        "launch_attestation_sha256": M.file_sha256(attestation),
        "launch_command_sha256": "d" * 64,
    }
    segment = {**segment_body, "segment_sha256": M._identity_sha256(segment_body)}
    body = {
        "schema": M.SOURCE_SCHEMA,
        "settings": settings,
        "started_at_unix_s": 1.0,
        "per_tensor": per_tensor,
        "partial": False,
        "tensors_done": 33,
        "execution_segments": [segment],
        "completed_at_unix_s": 2.0,
        "population_summaries": summaries,
        "status": M.SOURCE_STATUS,
        "claim_boundary": M.CLAIM_BOUNDARY,
    }
    document = {**body, "checkpoint_sha256": M._identity_sha256(body)}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(document))
    return path


def test_exact_frontiers_are_recomputed_and_self_sealed(tmp_path: Path):
    source = _source(tmp_path)
    receipt = M.build_receipt(source)
    M.validate_receipt(receipt)
    assert receipt["population_counts"] == {"dense": 9, "routed": 24}
    assert receipt["source"]["sha256"] == M.file_sha256(source)
    assert receipt["population_summaries"]["dense"]["tensors"] == 9
    diagnostic = receipt["frontier_diagnostics"]["dense"][0]
    assert diagnostic["tcq_best_quality_higher"] == 9
    assert diagnostic["cb_minimum_bpw_lower"] == 9

    changed = copy.deepcopy(receipt)
    changed["population_summaries"]["dense"]["tensors"] = 8
    with pytest.raises(M.AnalysisReceiptError, match="self-digest differs"):
        M.validate_receipt(changed)


def test_source_or_summary_mutation_refuses(tmp_path: Path):
    source = _source(tmp_path)
    document = json.loads(source.read_text())
    document["population_summaries"]["dense"]["tensors"] = 8
    body = {key: value for key, value in document.items() if key != "checkpoint_sha256"}
    document["checkpoint_sha256"] = M._identity_sha256(body)
    source.write_text(json.dumps(document))
    with pytest.raises(M.AnalysisReceiptError, match="summaries differ"):
        M.build_receipt(source)


def test_no_replace_publication(tmp_path: Path):
    receipt = M.build_receipt(_source(tmp_path))
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
