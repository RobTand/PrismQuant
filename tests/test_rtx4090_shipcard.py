"""Shipcard refusal coverage for the strict RTX 4090 FP8-CB lane.

These tests stay at the CPU/record boundary.  Dispatch tests replace the
specialized validator module with an in-memory stand-in, while direct verifier
tests use synthetic records and never launch torch, vLLM, Gridbook, or inspect
a physical GPU.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import types

import pytest

import prismaquant.shipcard as shipcard_module
import prismaquant.validate_rtx4090_fp8_cb as rtx_verifier
from prismaquant.cb_layout import FP8_PRODUCT_RUNGS
from prismaquant.gridbook_validation_only_policy import (
    SM120_VALIDATION_CANDIDATE_CONTRACT_PATH,
    VALIDATION_ONLY_DISPOSITION,
    sm120_validation_only_policy_stamp,
    sm120_validation_only_route_status_stamp,
    validation_only_policy_build_fields,
)
from prismaquant.shipcard import (
    REQUIRED_SLOTS,
    RTX4090_REQUIRED_SLOTS,
    build_shipcard,
    required_slots,
    verify,
)


_STRICT_SLOT = RTX4090_REQUIRED_SLOTS[0]
_HEX64 = "a" * 64
_SESSION_NONCE = "e" * 32
_FAKE_RUNTIME_PIN = {
    "schema": rtx_verifier.RTX4090_SERVING_PIN_SCHEMA,
    "repository": rtx_verifier.RTX4090_GRIDBOOK_REPOSITORY,
    "commit": "b" * 40,
    "version": "1.2.3",
    "version_is_release": True,
    "wheel_sha256": "c" * 64,
    "runtime_contract_schema": "gridbook.runtime-contract.v11",
    "required_abi_features": {"sm89_fp8_cb": 1},
}
_FAKE_VLLM_PIN = {
    "schema": rtx_verifier.RTX4090_VLLM_RUNTIME_PIN_SCHEMA,
    "repository": rtx_verifier.RTX4090_VLLM_REPOSITORY,
    "commit": "f" * 40,
    "version": "9.8.7",
    "record_sha256": "1" * 64,
}
_FAKE_CONTENT_RECEIPT = {
    "schema": "prismaquant.safetensors_content_receipt/1",
    "source": "verified_read",
    "content_read_passes": 1,
    "content_bytes_read": 123,
    "read_calls": 2,
    "root": {"device": 1, "inode": 2},
    "files": {
        "model.safetensors": {
            "stat": {
                "device": 1,
                "inode": 3,
                "bytes": 123,
                "mtime_ns": 4,
                "ctime_ns": 5,
            },
            "sha256": "2" * 64,
            "tensor_sha256": {"model.layers.0.weight": "3" * 64},
        },
    },
}


def _artifact(tmp_path, *, artifact_format: str = "fp8_cb"):
    model_dir = tmp_path / "exported"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({
            "model_type": "qwen3_5_text",
            "architectures": ["Qwen3_5ForCausalLM"],
        }),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"strict-fp8-cb")
    (model_dir / "quant_config.json").write_text(
        json.dumps({
            "quant_method": "gridbook",
            "format": artifact_format,
        }),
        encoding="utf-8",
    )
    return model_dir


def _strict_card(model_dir):
    return build_shipcard(
        model_dir,
        build={
            "quant_method": "gridbook",
            "producer_policy": "qwen38_27b_rtx4090_fp8_cb",
            "serving_profile": "qwen38_rtx4090_fp8_cb",
        },
    )


def _passing_record(card, slot: str) -> dict[str, object]:
    return {
        "slot": slot,
        "tool": "test",
        "passed": True,
        "model_sha": card["model_sha"],
    }


def _install_specialized_verifier(monkeypatch, calls, *, result):
    module = types.ModuleType("prismaquant.validate_rtx4090_fp8_cb")

    def specialized(slot, record, *, model_dir=None):
        calls.append((slot, record, model_dir))
        return list(result)

    module.verify_rtx4090_shipcard_record = specialized
    monkeypatch.setitem(
        sys.modules, "prismaquant.validate_rtx4090_fp8_cb", module
    )


def _canonical_sha256(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _valid_compilation_provenance() -> dict[str, object]:
    provenance = {
        "schema": "prismaquant.vllm_compilation_provenance/1",
        "name": "vllm",
        "version": "9.8.7",
        "distribution_package_root": "/opt/site/vllm",
        "module_origin": "/opt/site/vllm/__init__.py",
        "wrapper_path": "/opt/site/vllm/compilation/wrapper.py",
        "wrapper_identity": {"bytes": 1234, "sha256": "d" * 64},
        "compile_contract": {
            "direct_torch_compile_calls": 1,
            "fullgraph": True,
            "dynamic": False,
            "backend_explicit": True,
        },
        "runtime_pin": copy.deepcopy(_FAKE_VLLM_PIN),
        "direct_url": {
            "url": rtx_verifier.RTX4090_VLLM_REPOSITORY,
            "vcs_info": {
                "vcs": "git",
                "requested_revision": _FAKE_VLLM_PIN["commit"],
                "commit_id": _FAKE_VLLM_PIN["commit"],
            },
        },
        "direct_url_path": "/opt/site/vllm-9.8.7.dist-info/direct_url.json",
        "direct_url_identity": {"bytes": 234, "sha256": "e" * 64},
        "record_path": "/opt/site/vllm-9.8.7.dist-info/RECORD",
        "record_identity": {
            "bytes": 3456,
            "sha256": _FAKE_VLLM_PIN["record_sha256"],
        },
    }
    provenance["identity_sha256"] = _canonical_sha256(provenance)
    return provenance


def _valid_specialized_record(slot: str) -> dict[str, object]:
    arm = "eager" if slot == "native_export.eager" else "graph"
    served_model = f"qwen38-rtx4090-{_HEX64[:32]}-{_SESSION_NONCE}"
    fingerprint = "f" * 64
    graph = None
    compile_cache_root = None
    if arm == "graph":
        compile_cache_root = "/var/cache/prismaquant/rtx4090/run-1"
        graph = {
            "schema": "prismaquant.rtx4090_graph_contract.v1",
            "compilation_mode": 3,
            "compilation_backend": "inductor",
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "capture_sizes": [1, 2, 4, 8, 16, 32, 64],
            "max_model_len": 32768,
            "configured_compile_cache_root": compile_cache_root,
            "compile_cache": (
                f"{compile_cache_root}/f00dbabe/rank_0_0/backbone"
            ),
            "piecewise_capture_count": 7,
            "full_capture_count": 7,
            "serve_log_sha256": "1" * 64,
            "compile_cache_freshness": {
                "schema": (
                    "prismaquant.rtx4090_compile_cache_preflight.v1"
                ),
                "session_nonce": _SESSION_NONCE,
                "configured_container_root": compile_cache_root,
                "preflight_sha256": "9" * 64,
                "directory_device": 1,
                "directory_inode": 2,
                "post_file_count": 3,
                "post_total_bytes": 4,
                "post_tree_sha256": "0" * 64,
            },
        }
    runtime_file = {"bytes": 4567, "sha256": "2" * 64}
    endpoint_body = {
        "response_object": "list",
        "model_count": 1,
        "model": {
            "id": served_model,
            "object": "model",
            "owned_by": "vllm",
            "root": "/model",
            "max_model_len": 32768,
        },
    }
    endpoint_identity = {
        "schema": "prismaquant.server_models_endpoint_binding/1",
        "canonical_identity_sha256": _canonical_sha256(endpoint_body),
        **endpoint_body,
    }
    smoke = {
        "served_model": served_model,
        "deterministic_repeats": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "n": 1,
        "stream": False,
        "generated_utf8_bytes": 23,
        "output_sha256": "3" * 64,
        "models_endpoint_identity": endpoint_identity,
    }
    contract = {
        "schema": rtx_verifier.RTX4090_FP8_CB_CONTRACT_SCHEMA,
        "arm": arm,
        "policy": {
            "schema": "prismaquant.rtx4090_qwen38_fp8_policy.v1",
            "id": "qwen38_27b_rtx4090_fp8_cb",
        },
        "artifact": {
            "schema": "prismaquant.served_artifact_binding/1",
            "model_sha": _HEX64,
            "artifact_bytes": 18_000_000_000,
            "artifact_inventory_sha256": "4" * 64,
        },
        "artifact_content_receipt": copy.deepcopy(_FAKE_CONTENT_RECEIPT),
        "image": "gridbook@sha256:" + "5" * 64,
        "serve_manifest": {
            "sha256": "6" * 64,
            "serve_fingerprint": fingerprint,
        },
        "gpu": {
            "name": "NVIDIA GeForce RTX 4090",
            "uuid": "GPU-" + "7" * 32,
            "count": 1,
            "compute_capability": [8, 9],
            "driver_version": "999.1",
        },
        "launch": {
            "model": "/model",
            "served_model_name": served_model,
            "options": dict(sorted(rtx_verifier.rtx4090_launch_options(
                arm=arm,
                served_model=served_model,
                compile_cache=compile_cache_root,
            ).items())),
            "switches": sorted(
                rtx_verifier.rtx4090_launch_switches(arm=arm)
            ),
            "requires_moe_backend_marlin": False,
        },
        "environment": {},
        "session": {
            "schema": "prismaquant.cb_endpoint_session.v1",
            "models_endpoint_binding": endpoint_identity,
        },
        "runtime_pin": copy.deepcopy(_FAKE_RUNTIME_PIN),
        "vllm_runtime_pin": copy.deepcopy(_FAKE_VLLM_PIN),
        "runtime_attestation": {
            "runtime_contract_schema": "gridbook.runtime-contract.v11",
            "runtime_contract_sha256": "8" * 64,
            "lane_eligibility_schema": "gridbook.lane-eligibility.v2",
            "platform": "sm_89",
            "device_capability": [8, 9],
            "family": "FP8_CB_K",
            "structure": "dense",
            "rungs": list(FP8_PRODUCT_RUNGS),
            "regime_routes": [
                {
                    "rung": rung,
                    "regime": regime,
                    "cell_id": f"fp8_cb_k{rung}_{regime}_sm89",
                    "route_status": "backed",
                    "qualification": "device_qualified",
                    "requires_serve_flags": [],
                }
                for rung in FP8_PRODUCT_RUNGS
                for regime in ("decode", "batch")
            ],
            "requires_serve_flags": [],
        },
        "runtime_contract_file_identity": runtime_file,
        "gridbook_distribution": {
            "source_files": {
                "gridbook/runtime_contract.json": dict(runtime_file),
            },
        },
        "resident_extensions": ["prismaquant_cb_ext.so"],
        "packages": {"gridbook": "1.2.3", "vllm": "9.8.7"},
        "vllm_compilation_provenance": _valid_compilation_provenance(),
        "endpoint_smoke": smoke,
        "graph": graph,
    }
    contract["identity_sha256"] = _canonical_sha256(contract)
    return {
        "slot": slot,
        "tool": rtx_verifier.RTX4090_FP8_CB_TOOL,
        "passed": True,
        "model_sha": _HEX64,
        "serve_fingerprint": fingerprint,
        "metrics": {"rtx4090_contract": contract},
    }


def _resign_specialized_record(record) -> None:
    contract = record["metrics"]["rtx4090_contract"]
    compilation = contract["vllm_compilation_provenance"]
    compilation["identity_sha256"] = _canonical_sha256({
        key: value
        for key, value in compilation.items()
        if key != "identity_sha256"
    })
    contract["identity_sha256"] = _canonical_sha256({
        key: value
        for key, value in contract.items()
        if key != "identity_sha256"
    })


def _set_path(payload, path, value) -> None:
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def _isolate_pure_record_validation(monkeypatch) -> None:
    """Replace release-file lookups that are orthogonal to record semantics."""
    monkeypatch.setattr(
        rtx_verifier,
        "_tracked_pin_dict",
        lambda: copy.deepcopy(_FAKE_RUNTIME_PIN),
    )
    monkeypatch.setattr(
        rtx_verifier,
        "_verify_gridbook_distribution_identity",
        lambda *args, **kwargs: [],
    )


def test_build_shipcard_opens_strict_slot_from_on_disk_fp8_cb_format(tmp_path):
    model_dir = _artifact(tmp_path)

    # Detection must not depend on the mutable build payload carrying the
    # policy/profile echoes.  The identity-bound top-level format is enough.
    card = build_shipcard(model_dir, build={"quant_method": "gridbook"})

    assert tuple(card["slots"]) == REQUIRED_SLOTS + RTX4090_REQUIRED_SLOTS
    assert card["slots"][_STRICT_SLOT] is None


def test_validation_only_artifact_is_categorically_unshippable(tmp_path):
    model_dir = _artifact(tmp_path)
    quant = json.loads((model_dir / "quant_config.json").read_text())
    quant["provenance"] = {
        "producer_policy": {
            "schema": (
                "prismaquant.rtx4090_qwen38_fp8_validation_only_policy.v1"
            ),
            "id": "qwen38_27b_rtx4090_fp8_cb_validation_only",
            "artifact_disposition": "UNRELEASABLE_VALIDATION_ONLY",
        }
    }
    (model_dir / "quant_config.json").write_text(json.dumps(quant))
    card = build_shipcard(model_dir, build={})

    problems = verify(card, model_dir=model_dir, required=())

    assert any("UNRELEASABLE_VALIDATION_ONLY" in item for item in problems)
    card["build"]["artifact_disposition"] = "UNRELEASABLE_VALIDATION_ONLY"
    assert any(
        "UNRELEASABLE_VALIDATION_ONLY" in item
        for item in verify(card, required=())
    )


def _stamp_sm120_validation_quant_config(model_dir) -> dict:
    assignment = {
        "model.layers.0.self_attn.q_proj": "NVFP4_CB_K1",
        "model.layers.0.self_attn.k_proj": "FP8_CB_K48",
        "model.norm": "BF16",
    }
    policy = sm120_validation_only_policy_stamp(
        SM120_VALIDATION_CANDIDATE_CONTRACT_PATH
    )
    quant = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "provenance": {
            "producer_policy": policy,
            "tensor_formats": assignment,
            "cb_route_status": sm120_validation_only_route_status_stamp(
                policy, assignment
            ),
        },
    }
    (model_dir / "quant_config.json").write_text(json.dumps(quant))
    return quant


def test_sm120_validation_only_shipcard_echoes_and_categorically_refuses(
    tmp_path,
):
    model_dir = _artifact(tmp_path, artifact_format="nvfp4_cb")
    quant = _stamp_sm120_validation_quant_config(model_dir)
    card = build_shipcard(
        model_dir,
        build=validation_only_policy_build_fields(
            quant["provenance"]["producer_policy"]
        ),
    )

    assert card["cb_route_status"]["serving_profile"] == (
        "qwen38_sm120_cb_validation_only"
    )
    assert card["cb_route_status"]["artifact_disposition"] == (
        VALIDATION_ONLY_DISPOSITION
    )
    assert card["cb_route_status"]["release_eligible"] is False
    problems = verify(card, model_dir=model_dir, required=())
    assert any(VALIDATION_ONLY_DISPOSITION in item for item in problems)
    assert any("exact stamp verified" in item for item in problems)


@pytest.mark.parametrize(
    "mutation",
    ("remove_stamp", "tamper_policy", "cross_policy_replay"),
)
def test_sm120_shipcard_refuses_removed_tampered_or_cross_policy_stamp(
    tmp_path,
    mutation,
):
    model_dir = _artifact(tmp_path, artifact_format="nvfp4_cb")
    quant = _stamp_sm120_validation_quant_config(model_dir)
    card = build_shipcard(
        model_dir,
        build=validation_only_policy_build_fields(
            quant["provenance"]["producer_policy"]
        ),
    )
    provenance = quant["provenance"]
    if mutation == "remove_stamp":
        del provenance["producer_policy"]
        del provenance["cb_route_status"]
    elif mutation == "tamper_policy":
        provenance["producer_policy"]["candidate_runtime"]["gridbook"][
            "tree"
        ] = "0" * 40
    else:
        provenance["producer_policy"] = {
            "schema": (
                "prismaquant.rtx4090_qwen38_fp8_validation_only_policy.v1"
            ),
            "id": "qwen38_27b_rtx4090_fp8_cb_validation_only",
            "artifact_disposition": VALIDATION_ONLY_DISPOSITION,
        }
    (model_dir / "quant_config.json").write_text(json.dumps(quant))

    problems = verify(card, model_dir=model_dir, required=())
    assert any(VALIDATION_ONLY_DISPOSITION in item for item in problems)
    assert any(
        "missing or malformed" in item or "lacks the required" in item
        for item in problems
    )


def test_required_slots_rederives_strict_obligation_after_card_erasure(tmp_path):
    model_dir = _artifact(tmp_path)
    card = _strict_card(model_dir)

    card["slots"].pop(_STRICT_SLOT)
    card["build"] = {}

    assert required_slots(card, model_dir=model_dir) == (
        REQUIRED_SLOTS + RTX4090_REQUIRED_SLOTS
    )
    assert f"{_STRICT_SLOT}: UNFILLED" in verify(card, model_dir=model_dir)


def test_generic_nvfp4_cb_card_does_not_inherit_rtx4090_gate(tmp_path):
    model_dir = _artifact(tmp_path, artifact_format="nvfp4_cb")
    card = build_shipcard(model_dir, build={"quant_method": "gridbook"})

    assert _STRICT_SLOT not in card["slots"]
    assert required_slots(card, model_dir=model_dir) == REQUIRED_SLOTS


def test_generic_nvfp4_cb_native_record_keeps_generic_dispatch(
    tmp_path, monkeypatch,
):
    model_dir = _artifact(tmp_path, artifact_format="nvfp4_cb")
    card = build_shipcard(model_dir, build={"quant_method": "gridbook"})
    slot = "native_export.graph"
    record = _passing_record(card, slot)
    card["slots"][slot] = record
    calls = []
    marker = f"{slot}: generic verifier marker"

    def generic(slot_arg, record_arg, *, model_dir=None):
        calls.append((slot_arg, record_arg, model_dir))
        return [marker]

    module = types.ModuleType("prismaquant.validate_rtx4090_fp8_cb")

    def strict_must_not_run(*args, **kwargs):
        raise AssertionError(
            "a generic nvfp4_cb artifact must not inherit the RTX4090 gate"
        )

    module.verify_rtx4090_shipcard_record = strict_must_not_run
    monkeypatch.setitem(
        sys.modules, "prismaquant.validate_rtx4090_fp8_cb", module
    )
    monkeypatch.setattr(
        shipcard_module, "_verify_gridbook_native_record", generic
    )

    problems = verify(card, model_dir=model_dir, required=(slot,))

    assert marker in problems
    assert calls == [(slot, record, model_dir)]


@pytest.mark.parametrize(
    "slot", ("native_export.eager", "native_export.graph")
)
def test_strict_native_slots_dispatch_only_to_specialized_verifier(
    tmp_path, monkeypatch, slot,
):
    model_dir = _artifact(tmp_path)
    card = _strict_card(model_dir)
    record = _passing_record(card, slot)
    card["slots"][slot] = record
    calls = []
    marker = f"{slot}: specialized verifier marker"
    _install_specialized_verifier(monkeypatch, calls, result=(marker,))

    def generic_must_not_run(*args, **kwargs):
        raise AssertionError(
            "strict RTX 4090 native records must bypass the generic "
            "Gridbook verifier"
        )

    monkeypatch.setattr(
        shipcard_module, "_verify_gridbook_native_record", generic_must_not_run
    )

    problems = verify(card, model_dir=model_dir, required=(slot,))

    assert marker in problems
    assert calls == [(slot, record, model_dir)]


def test_strict_hardware_slot_dispatches_to_specialized_verifier(
    tmp_path, monkeypatch,
):
    model_dir = _artifact(tmp_path)
    card = _strict_card(model_dir)
    record = _passing_record(card, _STRICT_SLOT)
    card["slots"][_STRICT_SLOT] = record
    calls = []
    marker = f"{_STRICT_SLOT}: specialized verifier marker"
    _install_specialized_verifier(monkeypatch, calls, result=(marker,))

    problems = verify(card, model_dir=model_dir, required=(_STRICT_SLOT,))

    assert marker in problems
    assert calls == [(_STRICT_SLOT, record, model_dir)]


@pytest.mark.parametrize("mutation", ("null", "remove"))
def test_strict_slot_null_or_removal_stays_unfilled_from_artifact_bytes(
    tmp_path, mutation,
):
    model_dir = _artifact(tmp_path)
    card = _strict_card(model_dir)
    card["build"] = {}
    if mutation == "null":
        card["slots"][_STRICT_SLOT] = None
    else:
        card["slots"].pop(_STRICT_SLOT)

    problems = verify(card, model_dir=model_dir)

    assert _STRICT_SLOT in required_slots(card, model_dir=model_dir)
    assert f"{_STRICT_SLOT}: UNFILLED" in problems


@pytest.mark.parametrize("bad_record", ("passed", 1, ["passed"]))
def test_malformed_strict_slot_record_fails_closed(tmp_path, bad_record):
    model_dir = _artifact(tmp_path)
    card = _strict_card(model_dir)
    card["slots"][_STRICT_SLOT] = bad_record

    problems = verify(
        card, model_dir=model_dir, required=(_STRICT_SLOT,)
    )

    assert any(
        problem.startswith(f"{_STRICT_SLOT}: malformed record")
        for problem in problems
    )


@pytest.mark.parametrize(
    "replacement",
    (
        {"quant_method": "gridbook", "format": "nvfp4_cb"},
        "{not-json",
    ),
)
def test_mutating_on_disk_policy_cannot_erase_gate_without_breaking_identity(
    tmp_path, replacement,
):
    model_dir = _artifact(tmp_path)
    card = _strict_card(model_dir)
    card["slots"].pop(_STRICT_SLOT)
    card["build"] = {}

    quant_path = model_dir / "quant_config.json"
    quant_path.write_text(
        replacement if isinstance(replacement, str) else json.dumps(replacement),
        encoding="utf-8",
    )

    problems = verify(card, model_dir=model_dir)

    assert any("artifact changed since the shipcard was opened" in p
               for p in problems)


@pytest.mark.parametrize(
    "slot",
    ("native_export.eager", "native_export.graph", _STRICT_SLOT),
)
def test_specialized_verifier_accepts_a_complete_self_bound_record(
    monkeypatch, slot,
):
    _isolate_pure_record_validation(monkeypatch)
    record = _valid_specialized_record(slot)

    assert rtx_verifier.verify_rtx4090_shipcard_record(
        slot, record, model_dir=None
    ) == []


def test_specialized_verifier_surfaces_gridbook_distribution_errors(
    monkeypatch,
):
    marker = f"{_STRICT_SLOT}: Gridbook distribution identity mismatch"
    monkeypatch.setattr(
        rtx_verifier,
        "_tracked_pin_dict",
        lambda: copy.deepcopy(_FAKE_RUNTIME_PIN),
    )
    monkeypatch.setattr(
        rtx_verifier,
        "_verify_gridbook_distribution_identity",
        lambda *args, **kwargs: [marker],
    )
    record = _valid_specialized_record(_STRICT_SLOT)

    problems = rtx_verifier.verify_rtx4090_shipcard_record(
        _STRICT_SLOT, record, model_dir=None
    )

    assert marker in problems


@pytest.mark.parametrize(
    ("path", "value", "expected_problem"),
    (
        (
            ("gpu", "name"),
            "NVIDIA RTX 4090",
            "exactly one physical RTX4090/SM89",
        ),
        (
            ("gpu", "compute_capability"),
            [9, 0],
            "exactly one physical RTX4090/SM89",
        ),
        (
            ("artifact", "artifact_bytes"),
            18_000_000_001,
            "artifact identity/18GB ceiling binding",
        ),
        (
            ("artifact_content_receipt", "content_read_passes"),
            2,
            "artifact one-pass content receipt is invalid",
        ),
        (
            ("launch", "options", "--max-model-len"),
            "32767",
            "exact 32K/4GiB/TP1 profile",
        ),
        (
            ("launch", "options", "--kv-cache-memory-bytes"),
            str(4 * 1024**3 - 1),
            "exact 32K/4GiB/TP1 profile",
        ),
        (
            ("launch", "options", "--tensor-parallel-size"),
            "2",
            "exact 32K/4GiB/TP1 profile",
        ),
        (
            ("graph", "compilation_mode"),
            0,
            "fullgraph mode-3 compile/CUDA-graph receipt",
        ),
        (
            ("graph", "compilation_backend"),
            "eager",
            "fullgraph mode-3 compile/CUDA-graph receipt",
        ),
        (
            ("graph", "cudagraph_mode"),
            "FULL_DECODE_ONLY",
            "fullgraph mode-3 compile/CUDA-graph receipt",
        ),
        (
            ("graph", "capture_sizes"),
            [1, 2, 4, 8, 16, 32],
            "fullgraph mode-3 compile/CUDA-graph receipt",
        ),
        (
            ("graph", "full_capture_count"),
            1,
            "fullgraph mode-3 compile/CUDA-graph receipt",
        ),
        (
            ("graph", "compile_cache_freshness", "session_nonce"),
            "0" * 32,
            "fullgraph mode-3 compile/CUDA-graph receipt",
        ),
        (
            ("graph", "compile_cache_freshness", "post_file_count"),
            0,
            "fullgraph mode-3 compile/CUDA-graph receipt",
        ),
        (
            ("endpoint_smoke", "served_model"),
            f"qwen38-rtx4090-{_HEX64[:32]}-short",
            "served model name must bind the artifact digest",
        ),
        (
            (
                "session",
                "models_endpoint_binding",
                "model",
                "id",
            ),
            f"qwen38-rtx4090-{_HEX64[:32]}-{'0' * 32}",
            "models endpoint canonical identity digest is stale",
        ),
        (
            (
                "vllm_compilation_provenance",
                "compile_contract",
                "fullgraph",
            ),
            False,
            "fullgraph=True, dynamic=False",
        ),
        (
            (
                "vllm_compilation_provenance",
                "direct_url",
                "url",
            ),
            "https://github.com/example/vllm.git",
            "exact official pinned VCS commit",
        ),
        (
            (
                "vllm_compilation_provenance",
                "direct_url",
                "vcs_info",
                "commit_id",
            ),
            "0" * 40,
            "exact official pinned VCS commit",
        ),
        (
            (
                "vllm_compilation_provenance",
                "record_identity",
                "sha256",
            ),
            "0" * 64,
            "direct_url/RECORD identities are incomplete",
        ),
        (
            ("vllm_runtime_pin", "commit"),
            "0" * 40,
            "installed vLLM identity differs from the exact runtime pin",
        ),
        (
            (
                "vllm_compilation_provenance",
                "compile_contract",
                "dynamic",
            ),
            True,
            "fullgraph=True, dynamic=False",
        ),
        (
            (
                "vllm_compilation_provenance",
                "compile_contract",
                "backend_explicit",
            ),
            False,
            "fullgraph=True, dynamic=False",
        ),
        (
            ("runtime_attestation", "platform"),
            "sm_90",
            "exact full K4..K48 step-4",
        ),
        (
            (
                "runtime_attestation",
                "regime_routes",
                0,
                "requires_serve_flags",
            ),
            ["--allow-fallback"],
            "route row is not backed, flag-free, and device-qualified",
        ),
        (
            ("runtime_attestation", "regime_routes", 1, "regime"),
            "decode",
            "runtime attestation repeats a rung/regime route",
        ),
    ),
)
def test_specialized_verifier_rejects_resigned_semantic_mutations(
    monkeypatch, path, value, expected_problem,
):
    """A fresh outer digest cannot bless a weaker physical/graph claim."""
    _isolate_pure_record_validation(monkeypatch)
    record = _valid_specialized_record(_STRICT_SLOT)
    contract = record["metrics"]["rtx4090_contract"]
    _set_path(contract, path, value)
    _resign_specialized_record(record)

    problems = rtx_verifier.verify_rtx4090_shipcard_record(
        _STRICT_SLOT, record, model_dir=None
    )

    assert any(expected_problem in problem for problem in problems), problems


def test_specialized_verifier_rejects_mutable_image_and_untracked_runtime(
    monkeypatch,
):
    _isolate_pure_record_validation(monkeypatch)
    record = _valid_specialized_record(_STRICT_SLOT)
    contract = record["metrics"]["rtx4090_contract"]
    contract["image"] = "gridbook:latest"
    contract["runtime_pin"]["commit"] = "9" * 40
    _resign_specialized_record(record)

    problems = rtx_verifier.verify_rtx4090_shipcard_record(
        _STRICT_SLOT, record, model_dir=None
    )

    assert any("serving image is not immutable" in p for p in problems)
    assert any("not the current immutable tracked pin" in p for p in problems)


def test_arbitrary_immutable_image_cannot_replace_vllm_provenance(monkeypatch):
    _isolate_pure_record_validation(monkeypatch)
    record = _valid_specialized_record(_STRICT_SLOT)
    contract = record["metrics"]["rtx4090_contract"]
    contract["image"] = "arbitrary@sha256:" + "9" * 64
    contract["vllm_compilation_provenance"].pop("direct_url")
    _resign_specialized_record(record)

    problems = rtx_verifier.verify_rtx4090_shipcard_record(
        _STRICT_SLOT, record, model_dir=None
    )

    assert any(
        "vLLM fullgraph provenance is missing or not closed" in problem
        for problem in problems
    ), problems


def test_specialized_verifier_rejects_graph_receipt_without_cache_freshness(
    monkeypatch,
):
    _isolate_pure_record_validation(monkeypatch)
    record = _valid_specialized_record(_STRICT_SLOT)
    contract = record["metrics"]["rtx4090_contract"]
    contract["graph"].pop("compile_cache_freshness")
    _resign_specialized_record(record)

    problems = rtx_verifier.verify_rtx4090_shipcard_record(
        _STRICT_SLOT, record, model_dir=None
    )

    assert any(
        "fullgraph mode-3 compile/CUDA-graph receipt is incomplete" in problem
        for problem in problems
    ), problems


@pytest.mark.parametrize(
    ("field", "value", "expected_problem"),
    (
        ("tool", "hand-written", "not filled by"),
        ("metrics", {}, "missing structured RTX4090 contract"),
    ),
)
def test_specialized_verifier_rejects_handmade_or_empty_records(
    monkeypatch, field, value, expected_problem,
):
    _isolate_pure_record_validation(monkeypatch)
    record = _valid_specialized_record(_STRICT_SLOT)
    record[field] = value

    problems = rtx_verifier.verify_rtx4090_shipcard_record(
        _STRICT_SLOT, record, model_dir=None
    )

    assert any(expected_problem in problem for problem in problems), problems
