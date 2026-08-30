from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import prismaquant.dspark_serving_profile as dsp
from prismaquant.gridbook_serving_runtime_pin import (
    GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
)
import prismaquant.validate_cb_endpoint as cbv
from prismaquant.gridbook_environment import CANONICAL_GOLD_ENVIRONMENT


_PIN = {
    "schema": "prismaquant.gridbook_serving_runtime_pin.v1",
    "repository": "https://github.com/RobTand/gridbook.git",
    "commit": "a" * 40,
    "version": GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
    "version_is_release": True,
    "wheel_sha256": "b" * 64,
    "runtime_contract_schema": "gridbook.runtime-contract.v4",
    "required_abi_features": {
        "routed_moe_per_role_codebook_lut": 1,
        "source_fp8_block128_w8a16": 1,
        "dspark_construction_physical_bridge": 1,
    },
}


def _sha(payload: object) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "gridbook-source"
    package = root / "gridbook"
    package.mkdir(parents=True)
    names = sorted(dsp.GRIDBOOK_087_EXPECTED_SOURCE_IDENTIFIERS)
    (package / "lane_select.py").write_text(
        "import os\n"
        "_CUDACXX = os.environ.get('CUDACXX')\n"
        "_CXX = os.environ.get('CXX')\n"
        + "\n".join(
            f"# {name}" for name in names if name not in {"CUDACXX", "CXX"}
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _cache_key(bucket: int) -> str:
    shapes = (
        (bucket, dsp.DSPARK_FLASHINFER_NUM_HEADS, dsp.DSPARK_FLASHINFER_HEAD_DIM),
        (bucket, dsp.DSPARK_FLASHINFER_MAIN_TOPK),
        (
            -1,
            dsp.DSPARK_FLASHINFER_NUM_HEADS,
            dsp.DSPARK_FLASHINFER_SPLIT_COUNT,
            dsp.DSPARK_FLASHINFER_HEAD_DIM,
        ),
        (-1, dsp.DSPARK_FLASHINFER_NUM_HEADS, dsp.DSPARK_FLASHINFER_SPLIT_COUNT),
        (-1, dsp.DSPARK_FLASHINFER_NUM_HEADS, dsp.DSPARK_FLASHINFER_HEAD_DIM),
        (-1, dsp.DSPARK_FLASHINFER_NUM_HEADS),
        (bucket,),
        (dsp.DSPARK_FLASHINFER_NUM_HEADS,),
        (bucket, dsp.DSPARK_FLASHINFER_EXTRA_TOPK),
        (bucket,),
    )
    extras = (True, True, dsp.DSPARK_FLASHINFER_EXTRA_TOPK, True)
    return repr(
        (
            dsp.DSPARK_FLASHINFER_CUSTOM_OP,
            dsp.DSPARK_FLASHINFER_RUNNER,
            shapes,
            extras,
        )
    )


def _write_cache(tmp_path: Path) -> Path:
    cache = {
        "_metadata": {"flashinfer_version": dsp.DSPARK_FLASHINFER_VERSION},
        **{
            _cache_key(bucket): [dsp.DSPARK_FLASHINFER_RUNNER, bucket + 10]
            for bucket in dsp.DSPARK_FLASHINFER_TOKEN_BUCKETS
        },
    }
    path = tmp_path / "flashinfer-cache.json"
    path.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
    return path


def _runtime_evidence(tmp_path: Path) -> dict:
    source_receipt = dsp.require_gridbook_087_source_compatible(
        _write_source_fixture(tmp_path)
    ).receipt()
    cache = dsp.inspect_flashinfer_tuned_cache(_write_cache(tmp_path))
    evidence = {
        "schema": dsp.DSPARK_RUNTIME_EVIDENCE_SCHEMA,
        "profile_receipt": dsp.serving_profile_receipt(_PIN),
        "packages": {
            "gridbook": GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
            "vllm": dsp.DSPARK_VLLM_VERSION,
            "torch": dsp.DSPARK_TORCH_VERSION,
            dsp.DSPARK_FLASHINFER_DISTRIBUTION: dsp.DSPARK_FLASHINFER_VERSION,
        },
        "module_versions": {
            "gridbook": GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION,
            "vllm": dsp.DSPARK_VLLM_VERSION,
            "torch": dsp.DSPARK_TORCH_VERSION,
            "flashinfer": dsp.DSPARK_FLASHINFER_VERSION,
        },
        "build_metadata": {
            "path": str((tmp_path / "build-metadata.yaml").resolve()),
            "sha256": "c" * 64,
            "vllm_commit": dsp.DSPARK_VLLM_COMMIT,
            "flashinfer_commit": dsp.DSPARK_FLASHINFER_COMMIT,
        },
        "capabilities": {
            "flashinfer_native_dsv4_dispatch": {
                "head_dim": 64,
                "topk": 256,
                "present": True,
            },
            "vllm_moe_skip_padding": True,
        },
        "max_context_tuned_cache": cache,
        "gridbook_source_environment": source_receipt,
    }
    evidence["evidence_sha256"] = _sha(evidence)
    return evidence


def _restamp_runtime(payload: dict) -> None:
    unstamped = dict(payload)
    unstamped.pop("evidence_sha256", None)
    payload["evidence_sha256"] = _sha(unstamped)


def _write_route_log(tmp_path: Path, *, suffix: str = "") -> Path:
    fp8_layers = {4, 9, 18, 23, 28, 33, 38, 41}
    rows = ["[prismaquant-cb] cb_gemv=v2"]
    for layer in range(dsp.DSPARK_TARGET_LAYER_COUNT + dsp.DSPARK_DRAFT_LAYER_COUNT):
        for stack, width in (("w13", 4096), ("w2", 2048)):
            if layer in fp8_layers:
                rows.append(
                    "[prismaquant-cb] cb_gemv_kernel "
                    f"model.layers.{layer}.ffn.experts.{stack} "
                    f"k=8 n_sub=4 type_size=32 K={width} "
                    "-> inherited (not fp4-CB two-tier v2)"
                )
            else:
                rows.append(
                    "[prismaquant-cb] cb_gemv_kernel "
                    f"model.layers.{layer}.ffn.experts.{stack} "
                    f"k=12 n_sub=2 type_size=57 K={width} -> v2 (mode=v2)"
                )
    path = tmp_path / "serve.log"
    path.write_text("\n".join(rows) + "\n" + suffix, encoding="utf-8")
    return path


def _restamp_routes(payload: dict) -> None:
    payload["routes_sha256"] = _sha(payload["routes"])
    unstamped = dict(payload)
    unstamped.pop("receipt_sha256", None)
    payload["receipt_sha256"] = _sha(unstamped)


def _leaf_paths(payload: object, prefix: tuple = ()) -> list[tuple]:
    if isinstance(payload, dict):
        paths: list[tuple] = []
        for key, value in payload.items():
            paths.extend(_leaf_paths(value, (*prefix, key)))
        return paths
    if isinstance(payload, list):
        paths = []
        for index, value in enumerate(payload):
            paths.extend(_leaf_paths(value, (*prefix, index)))
        return paths
    return [prefix]


def _mutated(value: object) -> object:
    if value is None:
        return "unexpected"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    if isinstance(value, str):
        return value + "-mutated"
    raise AssertionError(f"no mutation for {value!r}")


def _set_path(payload: object, path: tuple, value: object) -> None:
    parent = payload
    for member in path[:-1]:
        parent = parent[member]
    parent[path[-1]] = value


_PROFILE_PATHS = _leaf_paths(dsp.DSPARK_SERVING_PROFILE.as_dict())


# Added when the producer pin advanced 0.8.5 -> 0.8.11 on 2026-08-21.  Both
# names are real Gridbook environment reads that only exist from 0.8.9 on, so
# the closed namespace had to grow to stay closed.
_ADDED_BY_THE_0_8_11_PIN_ADVANCE = (
    "PRISMAQUANT_CB_FP8_GEMV_V2",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R",
)
# Registered 2026-08-22 ahead of any pin advance: gridbook d4df36e's opt-in
# dense FP4-v2 round-2 arm, a pq_env_bool01 strict boolean defaulting OFF.
_ADDED_BY_THE_R2_FLAG_REGISTRATION = ("PRISMAQUANT_CB_FP4V2_DENSE_R2",)


def test_gold_environment_grew_additively_over_the_historical_0_8_5_set():
    """The 0.8.5-era gold environment is unchanged; later growth is additive.

    2026-08-21: the producer pin advanced 0.8.5 -> 0.8.11 and this module's
    registry had to describe the newer namespace.  Freezing one digest would
    then either block the advance or hide it, so the freeze is stated on the
    scope it was written to protect: the 29-name HISTORICAL projection must
    still hash to its original literal, proving no pre-existing canonical
    value moved, while the full map carries its own digest.  A silent edit to
    any old entry still fails, which was the original point.  2026-08-22: the
    registry learned gridbook's opt-in PRISMAQUANT_CB_FP4V2_DENSE_R2 the same
    way, again without moving any pre-existing value.
    """
    historical = {
        name: value for name, value in CANONICAL_GOLD_ENVIRONMENT.items()
        if name not in _ADDED_BY_THE_0_8_11_PIN_ADVANCE
        and name not in _ADDED_BY_THE_R2_FLAG_REGISTRATION
    }
    assert len(historical) == 29
    assert _sha(historical) == (
        "41dd44c5365d961b58f1fb94db9af32243bdbe1a1863cbdea60618f42e88397e"
    )
    assert len(CANONICAL_GOLD_ENVIRONMENT) == 32
    assert _sha(dict(CANONICAL_GOLD_ENVIRONMENT)) == (
        "60af3a0728a08478b893c0360cfcb05edfa4aaa47839c39252162a925732c2d5"
    )
    # Both additions are dispatch kill switches, consistent with every other
    # selector in the table: the runtime's own default moved to "auto" in
    # 0.8.9, and gold stays pinned to the kernel its evidence was measured on.
    for name in _ADDED_BY_THE_0_8_11_PIN_ADVANCE:
        assert CANONICAL_GOLD_ENVIRONMENT[name] == "0"
    # The R2 arm is opt-in with the legacy kernel as its unset default, so
    # gold pins the explicit off value for the same reason.
    for name in _ADDED_BY_THE_R2_FLAG_REGISTRATION:
        assert CANONICAL_GOLD_ENVIRONMENT[name] == "0"
    assert CANONICAL_GOLD_ENVIRONMENT["PRISMAQUANT_CB_GEMV"] == "inherited"
    assert CANONICAL_GOLD_ENVIRONMENT["PRISMAQUANT_PRELOAD_FUSED"] == "0"
    assert "PYTORCH_ALLOC_CONF" not in CANONICAL_GOLD_ENVIRONMENT
    assert dsp.DSPARK_PROFILE_ENVIRONMENT["PRISMAQUANT_CB_GEMV"] == "v2"
    assert dsp.DSPARK_PROFILE_ENVIRONMENT["PRISMAQUANT_PRELOAD_FUSED"] == "1"


@pytest.mark.parametrize("name", sorted(dsp.DSPARK_PROFILE_ENVIRONMENT))
def test_every_selected_environment_value_is_fail_closed(name: str):
    environ = {
        key: value
        for key, value in dsp.DSPARK_PROFILE_ENVIRONMENT.items()
        if value is not None
    }
    expected = dsp.DSPARK_PROFILE_ENVIRONMENT[name]
    if expected is None:
        environ[name] = "unexpected"
    else:
        environ.pop(name)
    with pytest.raises(dsp.DSparkServingProfileError, match="environment differs"):
        dsp.attest_profile_environment(environ)


def test_apply_profile_environment_clears_w2_and_retired_overrides():
    environ = {name: "polluted" for name in dsp.DSPARK_PROFILE_ENVIRONMENT}
    environ["UNRELATED"] = "preserved"
    observed = dsp.apply_profile_environment(environ)
    assert observed == dict(dsp.DSPARK_PROFILE_ENVIRONMENT)
    assert environ["UNRELATED"] == "preserved"
    assert all(
        name not in environ
        for name in (
            "PRISMAQUANT_CB_W2_ROWS",
            "PRISMAQUANT_CB_W2_SCHED",
            "PRISMAQUANT_CB_W2_WARPS",
            "PRISMAQUANT_CB_DECODE",
            "PRISMAQUANT_CB_EXPAND",
        )
    )


@pytest.mark.parametrize("path", _PROFILE_PATHS, ids=lambda path: ".".join(map(str, path)))
def test_every_static_profile_leaf_rejects_a_restamped_mutation(path: tuple):
    receipt = dsp.serving_profile_receipt(_PIN)
    current = receipt["profile"]
    for member in path:
        current = current[member]
    _set_path(receipt["profile"], path, _mutated(current))
    receipt["profile_sha256"] = _sha(receipt["profile"])
    unstamped = dict(receipt)
    unstamped.pop("receipt_sha256")
    receipt["receipt_sha256"] = _sha(unstamped)
    with pytest.raises(dsp.DSparkServingProfileError, match="differs"):
        dsp.validate_serving_profile_receipt(receipt, expected_runtime_pin=_PIN)


def test_gridbook_0_8_6_source_namespace_is_exact_and_fail_closed(tmp_path: Path):
    root = _write_source_fixture(tmp_path)
    report = dsp.require_gridbook_087_source_compatible(root)
    assert report.unknown_identifiers == ()
    assert report.missing_expected_identifiers == ()

    source = root / "gridbook" / "lane_select.py"
    original = source.read_text(encoding="utf-8")
    source.write_text(original + "# PRISMAQUANT_CB_NEW_UNSCOPED\n", encoding="utf-8")
    with pytest.raises(dsp.DSparkServingProfileError, match="unknown=.*NEW_UNSCOPED"):
        dsp.require_gridbook_087_source_compatible(root)

    source.write_text(
        original.replace("# PRISMAQUANT_CB_GEMV\n", ""), encoding="utf-8"
    )
    with pytest.raises(dsp.DSparkServingProfileError, match="missing=.*CB_GEMV"):
        dsp.require_gridbook_087_source_compatible(root)


def test_tuned_cache_requires_every_exact_2048_bucket_and_positive_tactic(tmp_path: Path):
    cache_path = _write_cache(tmp_path)
    receipt = dsp.inspect_flashinfer_tuned_cache(cache_path)
    assert receipt["target_buckets"] == [1, 4, 8, 16, 32, 64]
    assert receipt["shape_contract"]["extra_topk"] == 2048

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload.pop(_cache_key(64))
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dsp.DSparkServingProfileError, match="lacks the exact"):
        dsp.inspect_flashinfer_tuned_cache(cache_path)

    payload[_cache_key(64)] = [dsp.DSPARK_FLASHINFER_RUNNER, -1]
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(dsp.DSparkServingProfileError, match="fallback tactic"):
        dsp.inspect_flashinfer_tuned_cache(cache_path)


_RUNTIME_STATIC_PATHS = [
    ("packages", name)
    for name in ("gridbook", "vllm", "torch", dsp.DSPARK_FLASHINFER_DISTRIBUTION)
] + [
    ("module_versions", name)
    for name in ("gridbook", "vllm", "torch", "flashinfer")
] + [
    ("build_metadata", "vllm_commit"),
    ("build_metadata", "flashinfer_commit"),
    ("capabilities", "flashinfer_native_dsv4_dispatch", "head_dim"),
    ("capabilities", "flashinfer_native_dsv4_dispatch", "topk"),
    ("capabilities", "flashinfer_native_dsv4_dispatch", "present"),
    ("capabilities", "vllm_moe_skip_padding"),
    ("max_context_tuned_cache", "target_entry_count"),
    ("max_context_tuned_cache", "target_buckets", 0),
    *[
        ("max_context_tuned_cache", "target_tactics", str(bucket))
        for bucket in dsp.DSPARK_FLASHINFER_TOKEN_BUCKETS
    ],
    *[
        ("max_context_tuned_cache", "shape_contract", name)
        for name in (
            "custom_op", "runner", "num_heads", "head_dim", "main_topk",
            "extra_topk", "split_count",
        )
    ],
]


@pytest.mark.parametrize(
    "path", _RUNTIME_STATIC_PATHS, ids=lambda path: ".".join(map(str, path))
)
def test_every_runtime_capability_leaf_rejects_restamped_mutation(
    tmp_path: Path, path: tuple
):
    evidence = _runtime_evidence(tmp_path)
    current = evidence
    for member in path:
        current = current[member]
    _set_path(
        evidence,
        path,
        -1 if "target_tactics" in path else _mutated(current),
    )
    _restamp_runtime(evidence)
    with pytest.raises(dsp.DSparkServingProfileError):
        dsp.validate_runtime_evidence(evidence, expected_runtime_pin=_PIN)


def test_runtime_evidence_accepts_exact_profile_and_capabilities(tmp_path: Path):
    evidence = _runtime_evidence(tmp_path)
    assert dsp.validate_runtime_evidence(
        evidence, expected_runtime_pin=_PIN
    ) == evidence


def test_exact_build_metadata_commits_are_bound(tmp_path: Path):
    metadata = tmp_path / "build-metadata.yaml"
    metadata.write_text(
        f"vllm_commit: {dsp.DSPARK_VLLM_COMMIT}\n"
        f"flashinfer_commit: {dsp.DSPARK_FLASHINFER_COMMIT}\n",
        encoding="utf-8",
    )
    assert dsp._build_metadata_receipt(metadata)["sha256"] == _file_sha(metadata)
    metadata.write_text(
        f"vllm_commit: {'0' * 40}\n"
        f"flashinfer_commit: {dsp.DSPARK_FLASHINFER_COMMIT}\n",
        encoding="utf-8",
    )
    with pytest.raises(dsp.DSparkServingProfileError, match="commits differ"):
        dsp._build_metadata_receipt(metadata)


_ROUTE_ROW_FIELDS = (
    "route_id", "artifact_role", "layer", "stack", "format_family", "k_bits",
    "n_sub", "type_size", "in_features", "kernel", "reason",
)


@pytest.mark.parametrize("field", _ROUTE_ROW_FIELDS)
def test_every_route_row_field_rejects_restamped_mutation(tmp_path: Path, field: str):
    receipt = dsp.collect_route_census(_write_route_log(tmp_path))
    receipt["routes"][0][field] = _mutated(receipt["routes"][0][field])
    _restamp_routes(receipt)
    with pytest.raises(dsp.DSparkServingProfileError):
        dsp.validate_route_census(receipt)


@pytest.mark.parametrize(
    "field",
    ("target_fp4_v2", "draft_fp4_v2", "fp8_inherited", "fallback", "unscoped", "total"),
)
def test_every_route_census_count_rejects_restamped_mutation(
    tmp_path: Path, field: str
):
    receipt = dsp.collect_route_census(_write_route_log(tmp_path))
    receipt["counts"][field] += 1
    _restamp_routes(receipt)
    with pytest.raises(dsp.DSparkServingProfileError, match="does not replay"):
        dsp.validate_route_census(receipt)


def test_route_census_is_exact_76_fp4_v2_16_fp8_and_zero_fallback(tmp_path: Path):
    receipt = dsp.collect_route_census(_write_route_log(tmp_path))
    assert receipt["counts"] == {
        "target_fp4_v2": 70,
        "draft_fp4_v2": 6,
        "fp8_inherited": 16,
        "fallback": 0,
        "unscoped": 0,
        "total": 92,
    }
    assert dsp.validate_route_census(receipt) == receipt

    with pytest.raises(dsp.DSparkServingProfileError, match="malformed/unscoped"):
        dsp.collect_route_census(
            _write_route_log(
                tmp_path, suffix="[prismaquant-cb] cb_gemv_kernel malformed\n"
            )
        )
    with pytest.raises(dsp.DSparkServingProfileError, match="fallback"):
        dsp.collect_route_census(
            _write_route_log(
                tmp_path, suffix="WARNING: CB-GEMV-v2 unavailable\n"
            )
        )


def test_cache_log_proves_exact_cache_hit_and_no_fallback(tmp_path: Path):
    runtime = _runtime_evidence(tmp_path)
    cache = runtime["max_context_tuned_cache"]
    log = tmp_path / "cache.log"
    log.write_text(
        "Autotuning FlashInfer SM120 sparse MLA DSv4 decode with cache: "
        f"{cache['path']}\n"
        "Config cache hit for sparse_mla_sm120_decode_dsv4 "
        "(runner=SparseMlaDecodeV3Runner, source=config file)\n"
        f"Using FlashInfer autotune cache file: {cache['path']}\n"
        "FlashInfer SM120 sparse MLA DSv4 decode autotune cache loaded on rank 0 "
        f"from {cache['path']}.\n",
        encoding="utf-8",
    )
    receipt = dsp.collect_cache_log_evidence(
        log, runtime, expected_runtime_pin=_PIN
    )
    assert receipt["cache_hit"] is True
    assert receipt["fallback_tactic_count"] == 0
    assert dsp.validate_cache_log_evidence(
        receipt, runtime, expected_runtime_pin=_PIN
    ) == receipt

    log.write_text(log.read_text(encoding="utf-8") + "tactic=-1\n", encoding="utf-8")
    with pytest.raises(dsp.DSparkServingProfileError, match="cache hit"):
        dsp.collect_cache_log_evidence(log, runtime, expected_runtime_pin=_PIN)


def _measurement_file(tmp_path: Path) -> tuple[str, str]:
    path = tmp_path / "baseline-measurement.json"
    path.write_text('{"measured":true}\n', encoding="utf-8")
    return str(path.resolve()), _file_sha(path)


def _baseline_evidence(tmp_path: Path) -> dict:
    evidence_path, evidence_sha = _measurement_file(tmp_path)
    local_model = dsp.stamp_baseline_unit(
        {
            "schema": dsp.DSPARK_BASELINE_LOCAL_MODEL_SCHEMA,
            "evidence_path": evidence_path,
            "evidence_sha256": evidence_sha,
            "target_model_sha256": "1" * 64,
            "draft_model_sha256": "2" * 64,
            "model_loaded_bytes": 100,
            "quant_profile": {
                "target_container": "gridbook",
                "target_quant_method": "gridbook",
                "target_format": "nvfp4_cb",
                "target_profile_id": "dsv4-release-profile",
                "target_quantizable_bpp": 4.25,
                "target_loaded_bytes": 80,
                "draft_container": "gridbook",
                "draft_quant_method": "gridbook",
                "draft_format": "NVFP4_CB_K12",
                "draft_quantizable_bpp": 4.5,
                "draft_loaded_bytes": 20,
                "total_loaded_bytes": 100,
                "kv_cache_dtype": "fp8",
            },
        },
        digest_field="identity_sha256",
    )
    prefill = []
    for tokens in (2048, 65536):
        internal_seconds = tokens / 1000.0
        prefill.append(dsp.stamp_baseline_unit(
            {
                "schema": dsp.DSPARK_BASELINE_PREFILL_ROW_SCHEMA,
                "name": f"uncached-context-{tokens}",
                "status": "collected",
                "evidence_path": evidence_path,
                "evidence_sha256": evidence_sha,
                "prompt_sha256": f"{tokens % 10}" * 64,
                "tokenizer_sha256": "3" * 64,
                "requested_uncached_prompt_tokens": tokens,
                "observed_uncached_prompt_tokens": tokens,
                "context_start_tokens": 0,
                "context_end_tokens": tokens,
                "max_model_len": dsp.DSPARK_MODEL_LEN,
                "max_num_batched_tokens": 512,
                "prefill_chunk_tokens": 512,
                "prefix_cache_enabled": False,
                "prefix_cache_hits_before": 0,
                "prefix_cache_hits_after": 0,
                "prefix_cache_hit_delta": 0,
                "concurrency": 1,
                "batch_size": 1,
                "wall_seconds": internal_seconds + 1.0,
                "internal_prefill_seconds": internal_seconds,
                "internal_prefill_tokens": tokens,
                "internal_prefill_tokens_per_second": 1000.0,
                "endpoint_prompt_tokens_before": 0,
                "endpoint_prompt_tokens_after": tokens,
                "endpoint_prompt_token_delta": tokens,
                "unavailable_reason": None,
            },
            digest_field="row_sha256",
        ))
    prefill.append(dsp.stamp_baseline_unit(
        {
            "schema": dsp.DSPARK_BASELINE_PREFILL_ROW_SCHEMA,
            "name": "uncached-context-517963",
            "status": "unavailable",
            "evidence_path": evidence_path,
            "evidence_sha256": evidence_sha,
            "prompt_sha256": "4" * 64,
            "tokenizer_sha256": "3" * 64,
            "requested_uncached_prompt_tokens": 517963,
            "observed_uncached_prompt_tokens": None,
            "context_start_tokens": 0,
            "context_end_tokens": None,
            "max_model_len": dsp.DSPARK_MODEL_LEN,
            "max_num_batched_tokens": 512,
            "prefill_chunk_tokens": 512,
            "prefix_cache_enabled": False,
            "prefix_cache_hits_before": None,
            "prefix_cache_hits_after": None,
            "prefix_cache_hit_delta": None,
            "concurrency": 1,
            "batch_size": 1,
            "wall_seconds": None,
            "internal_prefill_seconds": None,
            "internal_prefill_tokens": None,
            "internal_prefill_tokens_per_second": None,
            "endpoint_prompt_tokens_before": None,
            "endpoint_prompt_tokens_after": None,
            "endpoint_prompt_token_delta": None,
            "unavailable_reason": "requested_context_exceeds_selected_limit",
        },
        digest_field="row_sha256",
    ))
    decode = []
    for context, concurrency in ((12288, 1), (240000, 1), (12288, 12)):
        output_per_request = 192
        total = output_per_request * concurrency
        wall = total / 20.0
        internal = total / 25.0
        decode.append(dsp.stamp_baseline_unit(
            {
                "schema": dsp.DSPARK_BASELINE_DECODE_ROW_SCHEMA,
                "name": f"context-{context}-concurrency-{concurrency}",
                "evidence_path": evidence_path,
                "evidence_sha256": evidence_sha,
                "prompt_corpus_sha256": f"{context % 10}" * 64,
                "tokenizer_sha256": "3" * 64,
                "context_tokens": context,
                "max_model_len": dsp.DSPARK_MODEL_LEN,
                "max_num_batched_tokens": 512,
                "prefill_chunk_tokens": 512,
                "prefix_cache_enabled": False,
                "prefix_cache_hits_before": 0,
                "prefix_cache_hits_after": 0,
                "prefix_cache_hit_delta": 0,
                "concurrency": concurrency,
                "batch_size": concurrency,
                "request_count": concurrency,
                "output_tokens_per_request": output_per_request,
                "observed_output_tokens_total": total,
                "prefill_included": False,
                "wall_seconds": wall,
                "aggregate_output_tokens_per_second": 20.0,
                "internal_decode_seconds": internal,
                "internal_output_tokens": total,
                "internal_output_tokens_per_second": 25.0,
                "endpoint_generation_tokens_before": 0,
                "endpoint_generation_tokens_after": total,
                "endpoint_generation_token_delta": total,
            },
            digest_field="row_sha256",
        ))
    gaps = [
        "engine_runtime_and_container_differ",
        "model_weight_quantization_and_loaded_bytes_differ",
        "reference_prompt_and_tokenizer_inputs_are_not_published_exactly",
        "reference_prefill_claims_are_engine_side_while_local_rows_bind_wall_and_internal_metrics",
        "selected_release_context_limit_below_entrpi_517963_frontier",
    ]
    return dsp.build_baseline_comparison_evidence(
        local_model=local_model,
        prefill_frontier=prefill,
        decode_concurrency=decode,
        comparability_gaps=gaps,
    )


def _restamp_baseline(payload: dict) -> None:
    payload["local_model"] = dsp.stamp_baseline_unit(
        payload["local_model"], digest_field="identity_sha256"
    )
    payload["prefill_frontier"] = [
        dsp.stamp_baseline_unit(row, digest_field="row_sha256")
        for row in payload["prefill_frontier"]
    ]
    payload["decode_concurrency"] = [
        dsp.stamp_baseline_unit(row, digest_field="row_sha256")
        for row in payload["decode_concurrency"]
    ]
    unstamped = dict(payload)
    unstamped.pop("evidence_sha256", None)
    payload["evidence_sha256"] = _sha(unstamped)


def test_baseline_comparison_is_reference_only_and_honestly_incomplete(tmp_path: Path):
    evidence = _baseline_evidence(tmp_path)
    validated = dsp.validate_baseline_comparison_evidence(
        evidence, verify_files=True, require_complete=False
    )
    assert validated["reference"]["engine_source"]["tag"] == "v0.5.0"
    assert validated["reference"]["hardware"]["gpu"] == "NVIDIA GB10"
    assert validated["comparability"]["reference_only"] is True
    assert validated["comparability"]["derives_release_pass"] is False
    assert validated["comparability"]["thresholds"] == []
    assert validated["comparability"]["missing_measurements"] == [
        "uncached-context-517963"
    ]
    with pytest.raises(dsp.DSparkServingProfileError, match="incomplete"):
        dsp.validate_baseline_comparison_evidence(evidence, require_complete=True)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("reference", "engine_source", "commit"), "0" * 40),
        (("local_model", "model_loaded_bytes"), 101),
        (("local_model", "quant_profile", "target_format"), "mxfp4"),
        (("prefill_frontier", 0, "observed_uncached_prompt_tokens"), 2047),
        (("prefill_frontier", 0, "context_end_tokens"), 2047),
        (("prefill_frontier", 0, "max_num_batched_tokens"), 1024),
        (("prefill_frontier", 0, "prefix_cache_enabled"), True),
        (("prefill_frontier", 0, "prefix_cache_hits_after"), 1),
        (("prefill_frontier", 0, "internal_prefill_tokens_per_second"), 999.0),
        (("prefill_frontier", 2, "status"), "collected"),
        (("decode_concurrency", 0, "context_tokens"), 12289),
        (("decode_concurrency", 0, "prefix_cache_hit_delta"), 1),
        (("decode_concurrency", 0, "prefill_included"), True),
        (("decode_concurrency", 2, "concurrency"), 11),
        (("decode_concurrency", 2, "observed_output_tokens_total"), 1),
        (("decode_concurrency", 2, "aggregate_output_tokens_per_second"), 19.0),
        (("comparability", "thresholds"), [{"tps": 1.0}]),
        (("comparability", "derives_release_pass"), True),
    ),
)
def test_baseline_selected_fields_reject_semantic_restamped_mutations(
    tmp_path: Path, path: tuple, value: object
):
    evidence = _baseline_evidence(tmp_path)
    _set_path(evidence, path, value)
    _restamp_baseline(evidence)
    with pytest.raises(dsp.DSparkServingProfileError):
        dsp.validate_baseline_comparison_evidence(evidence)


def test_baseline_paths_are_file_digest_bound(tmp_path: Path):
    evidence = _baseline_evidence(tmp_path)
    Path(evidence["local_model"]["evidence_path"]).write_text(
        "changed\n", encoding="utf-8"
    )
    with pytest.raises(dsp.DSparkServingProfileError, match="bytes changed"):
        dsp.validate_baseline_comparison_evidence(evidence, verify_files=True)


def test_dspark_manifest_wrapper_passes_exact_environment_to_shared_validator(
    tmp_path: Path, monkeypatch
):
    runtime = _runtime_evidence(tmp_path)
    manifest = {
        "dspark_serving_profile": runtime["profile_receipt"],
        "dspark_runtime_evidence": runtime,
    }
    captured = {}

    def _fake_validate(payload, **kwargs):
        captured.update(kwargs)
        return "f" * 64

    monkeypatch.setattr(cbv, "validate_serve_manifest", _fake_validate)
    assert dsp.validate_dspark_serve_manifest(
        manifest,
        arm="mtp",
        expected_served_model="model",
        requires_moe_marlin=True,
        expected_runtime_pin=_PIN,
    ) == "f" * 64
    assert captured["expected_image"] == dsp.DSPARK_IMAGE
    assert captured["expected_vllm_version"] == dsp.DSPARK_VLLM_VERSION
    assert captured["expected_server_environment"] == (
        dsp.expected_server_environment(_PIN)
    )
    assert captured["expected_server_environment_allowlist"] == (
        dsp.DSPARK_SERVER_ENV_ALLOWLIST
    )

    mutated = deepcopy(manifest)
    mutated["dspark_runtime_evidence"]["capabilities"][
        "vllm_moe_skip_padding"
    ] = False
    _restamp_runtime(mutated["dspark_runtime_evidence"])
    with pytest.raises(dsp.DSparkServingProfileError, match="capability"):
        dsp.validate_dspark_serve_manifest(
            mutated,
            arm="mtp",
            expected_served_model="model",
            requires_moe_marlin=True,
            expected_runtime_pin=_PIN,
        )
