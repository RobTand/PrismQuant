from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import prismaquant.validate_cb_endpoint as cbv
from prismaquant.gridbook_environment import (
    CANONICAL_GOLD_ENVIRONMENT,
    CANONICAL_GOLD_SET_ENVIRONMENT,
)
from prismaquant.lane_spec import load_lane_spec
from prismaquant.shipcard import (
    _verify_gridbook_native_record,
    build_shipcard,
    build_weight_content_manifest,
    load_shipcard,
    write_shipcard,
)
from tools.serve_fingerprint import (
    MANIFEST_SCHEMA,
    SERVER_ENV_ALLOWLIST,
    collect_manifest,
    elide_argv_paths,
    fingerprint,
    models_endpoint_binding_from_bytes,
    models_endpoint_binding_identity,
    process_identity_sha256,
    serve_session_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "serve_dsv4_cb_validate.sh"
_RESOLVED_GRAPH_CONFIG = (
    "Initializing a V1 LLM engine (test) with config: model='/model', "
    "compilation_config={'mode': <CompilationMode.NONE: 0>, "
    "'cudagraph_mode': <CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>, "
    "'cudagraph_capture_sizes': [1], 'max_cudagraph_capture_size': 1}\n"
)
_SERVED_MODEL = cbv.DSV4_SERVED_MODEL_PREFIX + "a" * 32
_REAL_GRIDBOOK_RUNTIME_PIN = cbv._gridbook_runtime_pin


@pytest.fixture(autouse=True)
def _released_gridbook_pin(monkeypatch):
    """Exercise endpoint evidence behind the exact tracked release pin."""

    import prismaquant.shipcard as shipcard_module
    from prismaquant.gridbook_runtime_pin import load_gridbook_runtime_pin

    pin_path = (
        ROOT
        / "prismaquant"
        / "gridbook_runtime"
        / "gridbook_runtime_pin.json"
    )
    released = json.loads(pin_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(cbv, "_gridbook_runtime_pin", lambda: dict(released))
    released_pin = load_gridbook_runtime_pin()
    monkeypatch.setattr(
        shipcard_module, "load_gridbook_runtime_pin", lambda: released_pin
    )


def _gridbook_distribution(pin: dict) -> dict:
    package_root = "/usr/local/lib/python3.12/site-packages/gridbook"
    import_origin = {
        "schema": "prismaquant.gridbook_import_origin/1",
        "module_name": "gridbook",
        "imported_version": pin["version"],
        "distribution_package_root": package_root,
        "module_file": f"{package_root}/__init__.py",
        "module_search_locations": [package_root],
    }
    import_origin["identity_sha256"] = cbv._canonical_json_sha256(import_origin)
    source_files = {
        name: {"bytes": 123, "sha256": "c" * 64}
        for name in (
            "gridbook/__init__.py",
            "gridbook/cuda_ext.py",
            "gridbook/plugin.py",
            "gridbook/runtime_contract.json",
            "gridbook/source_passthrough.py",
            "gridbook/fp8_source_w8a16.py",
            "gridbook/csrc/cb_gemv.cu",
            "gridbook/csrc/fp8_source_w8a16.cu",
            "gridbook/csrc/mxfp8_dense_gemm.cu",
        )
    }
    return {
        "schema": "prismaquant.installed_gridbook_distribution/2",
        "name": "gridbook",
        "repository": pin["repository"],
        "version": pin["version"],
        "direct_url": {
            "url": f"file:///tmp/gridbook-runtime-{pin['commit'][:12]}",
            "vcs_info": {
                "vcs": "git",
                "requested_revision": pin["commit"],
                "commit_id": pin["commit"],
            },
        },
        "direct_url_path": f"gridbook-{pin['version']}.dist-info/direct_url.json",
        "direct_url_identity": {"bytes": 123, "sha256": "d" * 64},
        "metadata_path": f"gridbook-{pin['version']}.dist-info/METADATA",
        "metadata_identity": {"bytes": 123, "sha256": "e" * 64},
        "record_path": f"gridbook-{pin['version']}.dist-info/RECORD",
        "record_identity": {"bytes": 123, "sha256": "f" * 64},
        "source_files": source_files,
        "source_files_sha256": cbv._canonical_json_sha256(source_files),
        "import_origin": import_origin,
    }


def test_live_gridbook_runtime_pin_projects_immutable_features_to_plain_dict():
    pin = _REAL_GRIDBOOK_RUNTIME_PIN()

    assert pin["commit"] == cbv.load_gridbook_runtime_pin().commit
    assert type(pin["required_abi_features"]) is dict


def _models_payload(
    model_id: str = _SERVED_MODEL,
    *,
    created: int = 1_700_000_000,
    permission_id: str = "modelperm-dynamic",
    root: str = "/model",
) -> dict:
    return {
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": created,
            "owned_by": "vllm",
            "root": root,
            "max_model_len": 8192,
            "permission": [{"id": permission_id}],
        }],
    }


def _write_bound_manifest(path: Path, arm: str, artifact: Path) -> dict:
    manifest = _manifest(arm)
    manifest["artifact_binding"] = {
        "schema": "prismaquant.served_artifact_binding/1",
        "model_sha": load_shipcard(artifact / "shipcard.json")["model_sha"],
        "artifact_inventory_sha256": "9" * 64,
        "artifact_bytes": sum(
            item.stat().st_size for item in artifact.iterdir() if item.is_file()
        ),
    }
    manifest["serve_fingerprint"] = fingerprint(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _manifest(
    arm: str,
    *,
    requires_marlin: bool = False,
    model_sha: str = "c" * 64,
) -> dict:
    pin = cbv._gridbook_runtime_pin()
    argv = [
        "/usr/local/bin/vllm", "serve", "/model",
        "--served-model-name", _SERVED_MODEL,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--trust-remote-code",
        "--tokenizer-mode", "deepseek_v4",
        "--generation-config", "vllm",
        "--quantization", "gridbook",
        "--tensor-parallel-size", "1",
        "--kv-cache-dtype", "fp8",
        "--kv-cache-memory-bytes", "1073741824",
        "--max-model-len", "8192",
        "--max-num-seqs", "1",
        "--max-num-batched-tokens", "512",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization", "0.90",
    ]
    if arm == "eager":
        argv.append("--enforce-eager")
    else:
        argv.extend([
            "--compilation-config", cbv.DSV4_GRAPH_COMPILATION_CONFIG,
        ])
    if requires_marlin:
        argv.extend(["--moe-backend", "marlin"])
    environment = {
        **dict(CANONICAL_GOLD_SET_ENVIRONMENT),
        "PQ_GRIDBOOK_RUNTIME_COMMIT": pin["commit"],
        "PQ_GRIDBOOK_RUNTIME_VERSION": pin["version"],
        "PYTHONSAFEPATH": "1",
        "PRISMAQUANT_CB_EXT_DIR": "/opt/gridbook/ext-cache",
        "PRISMAQUANT_PRELOAD_FUSED": "1",
    }
    process_pids = [1001, 1002]
    environment_rows = [
        {
            "pid": pid,
            "values": dict(environment),
            "sha256": cbv._canonical_json_sha256(environment),
        }
        for pid in process_pids
    ]
    host_identity = {
        "hostname": "endpoint-test",
        "boot_id": "11111111-2222-3333-4444-555555555555",
        "machine_id_sha256": "8" * 64,
        "pid_namespace": "pid:[4026533000]",
    }
    processes = []
    for index, pid in enumerate(process_pids):
        process_argv = argv if index == 0 else ["VLLM::EngineCore"]
        process = {
            "pid": pid,
            "argv": process_argv,
            "cmdline": " ".join(process_argv),
            "start_time_ticks": 123456 + index,
            "pid_namespace": "pid:[4026533000]",
            "executable": "/usr/local/bin/python3.12",
        }
        process["identity_sha256"] = process_identity_sha256(
            process, boot_id=host_identity["boot_id"]
        )
        processes.append(process)
    listener_row = {
        "family": "ipv4",
        "address": "0.0.0.0",
        "port": 8000,
        "socket_inode": "98765",
        "pids": [process_pids[0]],
    }
    models_raw = json.dumps(_models_payload()).encode("utf-8")
    payload = {
        "schema": MANIFEST_SCHEMA,
        "hostname": host_identity["hostname"],
        "host_identity": host_identity,
        "source": "server",
        "image": cbv.DSV4_SPARK_VLLM_IMAGE,
        "model": "/model",
        "served_model_name": _SERVED_MODEL,
        "launch_argv": argv,
        "launch_flags": elide_argv_paths(argv),
        "enforce_eager": arm == "eager",
        "quantization": "gridbook",
        "kv_cache_dtype": "fp8",
        "speculative_config": None,
        "package_versions": {
            "gridbook": pin["version"],
            "vllm": cbv.DSV4_SPARK_VLLM_VERSION,
        },
        "gridbook_runtime_pin": {
            "commit": pin["commit"],
            "version": pin["version"],
        },
        "gridbook_distribution": _gridbook_distribution(pin),
        "resident_extensions": ["prismaquant_cb_ext.so"],
        "residency_readable": True,
        "processes": processes,
        "server_process_environment": {
            "schema": "prismaquant.server_process_environment/1",
            "allowlist": sorted(SERVER_ENV_ALLOWLIST),
            "readable_pids": process_pids,
            "unreadable_pids": [],
            "consistent": True,
            "values": dict(environment),
            "processes": environment_rows,
        },
        "gpu_name": cbv.DSV4_SPARK_GPU_NAME,
        "gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555",
        "gpu_count": 1,
        "pq_env": dict(environment),
        "artifact_binding": {
            "schema": "prismaquant.served_artifact_binding/1",
            "model_sha": model_sha,
            "artifact_inventory_sha256": "9" * 64,
            "artifact_bytes": 1234,
        },
        "listener_census": {
            "schema": "prismaquant.server_tcp_listeners/1",
            "tables_readable": True,
            "unreadable_pids": [],
            "listeners": [listener_row],
        },
        "listener_binding": {
            "schema": "prismaquant.server_listener_binding/1",
            "base_url": "http://127.0.0.1:8000",
            "launch_host": "0.0.0.0",
            "launch_port": 8000,
            "listeners": [listener_row],
        },
        "models_endpoint_binding": models_endpoint_binding_identity(
            models_endpoint_binding_from_bytes(
                models_raw,
                request_url="http://127.0.0.1:8000/v1/models",
                expected_served_model=_SERVED_MODEL,
            )
        ),
    }
    payload["serve_session_id"] = serve_session_fingerprint(payload)
    payload["serve_fingerprint"] = fingerprint(payload)
    return payload


def _artifact_decode_record(*, requires_marlin: bool = False) -> dict:
    overlay = {
        "schema": "prismaquant.dspark_source_overlay.v1",
        "physical_namespace": "mtp.{stage}",
        "construction_namespace": "model.layers.{num_hidden_layers+stage}",
        "num_hidden_layers": 43,
        "n_mtp_layers": 3,
        "physical_stage_ids": [0, 1, 2],
        "construction_layer_ids": [43, 44, 45],
        "physical_target_counts": {
            "FP8_BLOCK_UE8M0_SOURCE": 25,
            "MXFP4_SOURCE": 2304,
        },
        "construction_unit_count": 22,
        "tensor_bytes_rewritten": 0,
    }
    evidence = {
        "schema": cbv.ARTIFACT_DECODE_CONTRACT_SCHEMA,
        "complete": True,
        "completeness_sha256": "a" * 64,
        "declared_unit_count": 1,
        "cb_unit_count": 1,
        "passthrough_unit_count": 1,
        "verbatim_namespace_unit_count": 1,
        "classified_unit_count": 3,
        "route_pending_acknowledged": [],
        "excluded_namespaces": [],
        "dspark_overlay": overlay,
        "dspark_overlay_sha256": "b" * 64,
        "requires_moe_backend_marlin": requires_marlin,
    }
    evidence["evidence_sha256"] = cbv._canonical_json_sha256(evidence)
    return evidence


def _smoke_metrics() -> dict:
    generated = b" Paris."
    return {
        "served_model": _SERVED_MODEL,
        "generated_chars": len(generated.decode("utf-8")),
        "generated_utf8_bytes": len(generated),
        "output_sha256": hashlib.sha256(generated).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            cbv.DEFAULT_PROMPT.encode("utf-8")
        ).hexdigest(),
        "deterministic_repeats": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "n": 1,
        "stream": False,
        "max_tokens": cbv.DEFAULT_MAX_TOKENS,
        "models_endpoint_identity": cbv._validate_live_server_session(
            _manifest("eager"), expected_served_model=_SERVED_MODEL
        )["models_endpoint_binding"],
    }


@pytest.mark.parametrize("arm", ["eager", "graph"])
def test_manifest_validation_binds_each_arm_to_exact_stack(arm):
    manifest = _manifest(arm)
    assert cbv.validate_serve_manifest(
        manifest,
        arm=arm,
        expected_served_model=_SERVED_MODEL,
        requires_moe_marlin=False,
        expected_model_sha="c" * 64,
    ) == (
        manifest["serve_fingerprint"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda m: m.update(speculative_config="{}"), "speculative"),
        (lambda m: m.update(quantization="compressed-tensors"), "gridbook"),
        (lambda m: m.update(kv_cache_dtype="auto"), "fp8"),
        (lambda m: m.update(residency_readable=False), "address space"),
        (lambda m: m.update(image="eugr/spark-vllm:latest"), "exact DSv4 image"),
        (lambda m: m.update(gpu_count=2), "one NVIDIA GB10"),
        (lambda m: m.update(model="/different"), "exact /model"),
        (lambda m: m.update(resident_extensions=[]), "Gridbook-native CUDA extension"),
        (
            lambda m: m["pq_env"].__setitem__("GRIDBOOK_MXFP8_DENSE", "1"),
            "exact live-process Gridbook contract",
        ),
    ],
)
def test_manifest_validation_fails_closed(mutation, message):
    manifest = _manifest("eager")
    mutation(manifest)
    manifest["serve_fingerprint"] = fingerprint(manifest)
    with pytest.raises(cbv.CBEndpointValidationError, match=message):
        cbv.validate_serve_manifest(
            manifest,
            arm="eager",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
            expected_model_sha="c" * 64,
        )


def test_manifest_fingerprint_cannot_be_stale():
    manifest = _manifest("graph")
    manifest["package_versions"]["vllm"] = "different"
    with pytest.raises(cbv.CBEndpointValidationError, match="fingerprint"):
        cbv.validate_serve_manifest(
            manifest,
            arm="graph",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
        )


def test_manifest_accepts_digest_pinned_gridbook_release_wheel():
    manifest = _manifest("eager")
    wheel_sha256 = "9" * 64
    manifest["gridbook_runtime_pin"]["wheel_sha256"] = wheel_sha256
    manifest["gridbook_distribution"]["direct_url"] = {
        "url": "file:///opt/gridbook-install/gridbook-0.8.5-py3-none-any.whl",
        "archive_info": {
            "hash": f"sha256={wheel_sha256}",
            "hashes": {"sha256": wheel_sha256},
        },
    }
    manifest["serve_fingerprint"] = fingerprint(manifest)

    assert cbv.validate_serve_manifest(
        manifest,
        arm="eager",
        expected_served_model=_SERVED_MODEL,
        requires_moe_marlin=False,
        expected_model_sha="c" * 64,
    ) == manifest["serve_fingerprint"]


def test_manifest_rejects_release_wheel_digest_not_in_runtime_pin():
    manifest = _manifest("eager")
    manifest["gridbook_runtime_pin"]["wheel_sha256"] = "9" * 64
    manifest["gridbook_distribution"]["direct_url"] = {
        "url": "file:///opt/gridbook-install/gridbook-0.8.5-py3-none-any.whl",
        "archive_info": {"hashes": {"sha256": "8" * 64}},
    }
    manifest["serve_fingerprint"] = fingerprint(manifest)

    with pytest.raises(cbv.CBEndpointValidationError, match="PEP 610 identity"):
        cbv.validate_serve_manifest(
            manifest,
            arm="eager",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
            expected_model_sha="c" * 64,
        )


def test_manifest_rejects_resigned_gridbook_import_shadow():
    manifest = _manifest("eager")
    origin = manifest["gridbook_distribution"]["import_origin"]
    origin["module_file"] = "/tmp/stale/gridbook/__init__.py"
    origin["module_search_locations"] = ["/tmp/stale/gridbook"]
    origin["identity_sha256"] = cbv._canonical_json_sha256({
        key: value for key, value in origin.items()
        if key != "identity_sha256"
    })
    manifest["serve_fingerprint"] = fingerprint(manifest)
    with pytest.raises(cbv.CBEndpointValidationError, match="imported Gridbook"):
        cbv.validate_serve_manifest(
            manifest,
            arm="eager",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
            expected_model_sha="c" * 64,
        )


def test_manifest_rejects_explicit_server_pythonpath_even_when_resigned():
    manifest = _manifest("eager")
    process_environment = manifest["server_process_environment"]
    process_environment["values"]["PYTHONPATH"] = "/tmp/stale"
    manifest["pq_env"]["PYTHONPATH"] = "/tmp/stale"
    for row in process_environment["processes"]:
        row["values"]["PYTHONPATH"] = "/tmp/stale"
        row["sha256"] = cbv._canonical_json_sha256(row["values"])
    manifest["serve_fingerprint"] = fingerprint(manifest)
    with pytest.raises(
        cbv.CBEndpointValidationError,
        match="exact live-process Gridbook contract",
    ):
        cbv.validate_serve_manifest(
            manifest,
            arm="eager",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
            expected_model_sha="c" * 64,
        )


def test_manifest_requires_explicit_tp1_in_launch_argv():
    manifest = _manifest("graph")
    index = manifest["launch_argv"].index("--tensor-parallel-size")
    manifest["launch_argv"][index + 1] = "2"
    manifest["serve_fingerprint"] = fingerprint(manifest)
    with pytest.raises(cbv.CBEndpointValidationError, match="tensor-parallel-size"):
        cbv.validate_serve_manifest(
            manifest,
            arm="graph",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
        )


def test_manifest_requires_prefix_caching_off():
    manifest = _manifest("eager")
    manifest["launch_argv"].remove("--no-enable-prefix-caching")
    manifest["serve_fingerprint"] = fingerprint(manifest)
    with pytest.raises(cbv.CBEndpointValidationError, match="switch contract"):
        cbv.validate_serve_manifest(
            manifest,
            arm="eager",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
        )


def test_manifest_binds_memory_and_batch_launch_contract():
    manifest = _manifest("graph")
    index = manifest["launch_argv"].index("--max-num-batched-tokens")
    manifest["launch_argv"][index + 1] = "1024"
    manifest["serve_fingerprint"] = fingerprint(manifest)
    with pytest.raises(cbv.CBEndpointValidationError, match="max-num-batched-tokens"):
        cbv.validate_serve_manifest(
            manifest,
            arm="graph",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
        )


def test_manifest_rejects_unknown_or_duplicate_launch_switches():
    unknown = _manifest("eager")
    unknown["launch_argv"].append("--disable-custom-all-reduce")
    unknown["serve_fingerprint"] = fingerprint(unknown)
    with pytest.raises(cbv.CBEndpointValidationError, match="undeclared option"):
        cbv.validate_serve_manifest(
            unknown,
            arm="eager",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
        )

    duplicate = _manifest("eager")
    duplicate["launch_argv"].append("--trust-remote-code")
    duplicate["serve_fingerprint"] = fingerprint(duplicate)
    with pytest.raises(cbv.CBEndpointValidationError, match="duplicates"):
        cbv.validate_serve_manifest(
            duplicate,
            arm="eager",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
        )


def test_manifest_requires_canonical_launch_flags():
    manifest = _manifest("eager")
    manifest["launch_flags"] = list(manifest["launch_flags"]) + ["--invented"]
    manifest["serve_fingerprint"] = fingerprint(manifest)
    with pytest.raises(cbv.CBEndpointValidationError, match="launch_flags"):
        cbv.validate_serve_manifest(
            manifest,
            arm="eager",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
        )


def test_manifest_marlin_route_is_exactly_artifact_conditional():
    present = _manifest("graph", requires_marlin=True)
    cbv.validate_serve_manifest(
        present,
        arm="graph",
        expected_served_model=_SERVED_MODEL,
        requires_moe_marlin=True,
    )
    with pytest.raises(cbv.CBEndpointValidationError, match="moe-backend"):
        cbv.validate_serve_manifest(
            _manifest("graph"),
            arm="graph",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=True,
        )
    with pytest.raises(cbv.CBEndpointValidationError, match="moe-backend"):
        cbv.validate_serve_manifest(
            present,
            arm="graph",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
        )


def test_manifest_requires_deep_gemm_disabled():
    manifest = _manifest("eager")
    manifest["pq_env"].pop("VLLM_USE_DEEP_GEMM")
    manifest["serve_fingerprint"] = fingerprint(manifest)
    with pytest.raises(cbv.CBEndpointValidationError, match="VLLM_USE_DEEP_GEMM"):
        cbv.validate_serve_manifest(
            manifest,
            arm="eager",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
        )


def test_serve_fingerprint_captures_deep_gemm_environment(monkeypatch):
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
    manifest = collect_manifest(
        pids=[__import__("os").getpid()],
        launch_argv=["vllm", "serve", "/model"],
    )
    assert manifest["pq_env"]["VLLM_USE_DEEP_GEMM"] == "0"


def test_endpoint_smoke_is_greedy_repeated_and_nonempty():
    requests = []

    def request(method, url, payload, timeout):
        requests.append((method, url, payload, timeout))
        if method == "GET":
            return _models_payload()
        return {"choices": [{"text": " Paris."}]}

    metrics = cbv.run_endpoint_smoke(
        base_url="http://127.0.0.1:8000/v1",
        model_name=_SERVED_MODEL,
        requester=request,
    )
    assert [row[0] for row in requests] == ["GET", "POST", "POST"]
    assert requests[1][2] == requests[2][2]
    assert requests[1][2]["temperature"] == 0.0
    assert requests[1][2]["seed"] == 0
    assert metrics["deterministic_repeats"] == 2
    assert metrics["generated_chars"] > 0
    assert "perplexity" not in metrics and "throughput" not in metrics


def test_models_endpoint_binding_ignores_dynamic_fields_but_binds_identity():
    first = models_endpoint_binding_from_bytes(
        json.dumps(_models_payload(created=100, permission_id="one")).encode(),
        request_url="http://127.0.0.1:8000/v1/models",
        expected_served_model=_SERVED_MODEL,
    )
    second = models_endpoint_binding_from_bytes(
        json.dumps(_models_payload(created=200, permission_id="two")).encode(),
        request_url="http://127.0.0.1:8000/v1/models",
        expected_served_model=_SERVED_MODEL,
    )
    assert first["response_sha256"] != second["response_sha256"]
    assert first["canonical_identity_sha256"] == second[
        "canonical_identity_sha256"
    ]
    assert first["model"] == second["model"]


@pytest.mark.parametrize(
    "payload",
    [
        _models_payload(model_id=cbv.DSV4_SERVED_MODEL_PREFIX + "b" * 32),
        _models_payload(root="/different"),
        {
            **_models_payload(),
            "data": [{**_models_payload()["data"][0], "max_model_len": 4096}],
        },
    ],
)
def test_endpoint_smoke_refuses_a_different_attested_endpoint_identity(payload):
    expected = cbv._validate_live_server_session(
        _manifest("eager"), expected_served_model=_SERVED_MODEL
    )["models_endpoint_binding"]

    def request(method, url, body, timeout):
        if method == "GET":
            return payload
        return {"choices": [{"text": " Paris."}]}

    with pytest.raises(cbv.CBEndpointValidationError):
        cbv.run_endpoint_smoke(
            base_url="http://127.0.0.1:8000/v1",
            model_name=_SERVED_MODEL,
            requester=request,
            expected_models_identity=expected,
        )


def test_direct_endpoint_gate_refuses_a_non_session_model_name():
    manifest = _manifest("eager")
    with pytest.raises(cbv.CBEndpointValidationError, match="session nonce"):
        cbv.validate_serve_manifest(
            manifest,
            arm="eager",
            expected_served_model="dsv4-flash-gridbook",
            requires_moe_marlin=False,
        )


@pytest.mark.parametrize("mutation", ["process", "listener", "session"])
def test_manifest_session_and_listener_tampering_is_refused(mutation):
    manifest = _manifest("eager")
    if mutation == "process":
        manifest["processes"][0]["start_time_ticks"] += 1
    elif mutation == "listener":
        manifest["listener_binding"]["listeners"][0]["pids"] = [1002]
    else:
        manifest["serve_session_id"] = "f" * 64
    manifest["serve_fingerprint"] = fingerprint(manifest)
    with pytest.raises(cbv.CBEndpointValidationError):
        cbv.validate_serve_manifest(
            manifest,
            arm="eager",
            expected_served_model=_SERVED_MODEL,
            requires_moe_marlin=False,
        )


@pytest.mark.parametrize("memory", ["3.25", "-0.05"])
def test_graph_capture_log_requires_positive_marker_not_positive_delta(
    tmp_path, memory,
):
    log = tmp_path / "serve.log"
    log.write_text(
        _RESOLVED_GRAPH_CONFIG
        + f"Graph capturing finished in 1 secs, took {memory} GiB\n",
        encoding="utf-8",
    )
    result = cbv.validate_graph_capture_log(log)
    assert result["capture_marker"].startswith("Graph capturing finished")
    assert len(result["serve_log_sha256"]) == 64


@pytest.mark.parametrize(
    "body",
    [
        "server ready without a capture marker\n",
        _RESOLVED_GRAPH_CONFIG
        + "Skipping CUDA graph capture. To turn on CUDA graph capture...\n",
        _RESOLVED_GRAPH_CONFIG.replace(
            "FULL_DECODE_ONLY: (2, 0)", "PIECEWISE: 1"
        )
        + "Graph capturing finished in 1 secs, took 1.00 GiB\n",
        _RESOLVED_GRAPH_CONFIG
        + "Overriding cudagraph_mode to PIECEWISE.\n"
        + "Graph capturing finished in 1 secs, took 1.00 GiB\n",
    ],
)
def test_graph_capture_log_refuses_missing_or_skipped_capture(tmp_path, body):
    log = tmp_path / "serve.log"
    log.write_text(body, encoding="utf-8")
    with pytest.raises(cbv.CBEndpointValidationError):
        cbv.validate_graph_capture_log(log)


@pytest.mark.parametrize("texts", [("", ""), (" Paris", " Lyon")])
def test_endpoint_smoke_refuses_empty_or_nondeterministic_generation(texts):
    responses = iter(texts)

    def request(method, url, payload, timeout):
        if method == "GET":
            return _models_payload()
        return {"choices": [{"text": next(responses)}]}

    with pytest.raises(cbv.CBEndpointValidationError):
        cbv.run_endpoint_smoke(
            base_url="http://127.0.0.1:8000/v1",
            model_name=_SERVED_MODEL,
            requester=request,
        )


def _artifact_and_card(tmp_path: Path) -> tuple[Path, Path]:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        '{"model_type":"deepseek_v4"}\n', encoding="utf-8"
    )
    (artifact / "model.safetensors").write_bytes(b"weights")
    (artifact / "codebooks.pqcb").write_bytes(b"codebooks")
    quant_config = {
        "quant_method": "gridbook",
        "format": "nvfp4_cb",
        "config_groups": {
            "group_0": {"targets": ["model.layers.0"]},
            "group_1": {
                "format": "source-passthrough",
                "source_format": "FP8_BLOCK_UE8M0_SOURCE",
                "targets": ["model.main_proj"],
            },
        },
        "codebook_file": "codebooks.pqcb",
        "provenance": {},
    }
    quant_config["provenance"]["weight_content_manifest"] = (
        build_weight_content_manifest(artifact)
    )
    (artifact / "quant_config.json").write_text(
        json.dumps(quant_config), encoding="utf-8"
    )
    shipcard = artifact / "shipcard.json"
    write_shipcard(shipcard, build_shipcard(
        artifact, build={"quant_method": "gridbook"}
    ))
    return artifact, shipcard


def test_artifact_gate_accepts_backed_block_fp8_without_acknowledgement(tmp_path):
    artifact, _shipcard = _artifact_and_card(tmp_path)
    quant = cbv.validate_cb_artifact(artifact)
    assert "route_pending_passthrough_acknowledged" not in quant["provenance"]


def test_artifact_gate_still_requires_ack_for_a_pending_route(
    tmp_path, monkeypatch,
):
    artifact, _shipcard = _artifact_and_card(tmp_path)
    monkeypatch.setattr(
        cbv,
        "ROUTE_PENDING_PASSTHROUGH_FORMATS",
        frozenset({"FP8_BLOCK_UE8M0_SOURCE"}),
    )
    with pytest.raises(cbv.CBEndpointValidationError, match="route-pending"):
        cbv.validate_cb_artifact(artifact)


def test_artifact_decode_contract_binds_completeness_and_dspark_overlay(
    tmp_path, monkeypatch,
):
    from types import SimpleNamespace

    import prismaquant.artifact_completeness as completeness_module
    import prismaquant.dspark_source_metadata as dspark_module

    report = completeness_module.CompletenessReport(
        declared_units={"model.main_proj": "FP8_BLOCK_UE8M0_SOURCE"},
        cb_units=["model.layers.0.mlp"],
        passthrough_units=["model.main_proj"],
        verbatim_namespace_units=["mtp.0.attn.wq_a"],
        route_pending_acknowledged=[],
    )
    overlay_provenance = _artifact_decode_record()["dspark_overlay"]
    overlay = SimpleNamespace(
        provenance=lambda: overlay_provenance,
        physical_targets={"mtp.0.main_proj": "FP8_BLOCK_UE8M0_SOURCE"},
        construction_units={"model.main_proj": "FP8_BLOCK_UE8M0_SOURCE"},
        physical_to_construction_unit={
            "mtp.0.main_proj": "model.main_proj"
        },
    )
    monkeypatch.setattr(
        completeness_module,
        "assert_artifact_complete",
        lambda root: report,
    )
    monkeypatch.setattr(
        dspark_module,
        "discover_dspark_source_overlay_from_artifact",
        lambda root: overlay,
    )
    quant_config = {
        "provenance": {"dspark_source_overlay": overlay_provenance},
        "source_passthrough": {
            "version": 1,
            "units": {
                "model.layers.0.mlp.experts": cbv.DSV4_MXFP4_WIRE_ID,
            },
        },
    }

    evidence = cbv.validate_cb_artifact_decode_contract(
        tmp_path, quant_config
    )
    assert evidence["complete"] is True
    assert evidence["classified_unit_count"] == 3
    assert evidence["requires_moe_backend_marlin"] is True
    cbv._validate_artifact_decode_record(evidence)

    quant_config["provenance"]["dspark_source_overlay"] = {"schema": "wrong"}
    with pytest.raises(cbv.CBEndpointValidationError, match="does not match"):
        cbv.validate_cb_artifact_decode_contract(tmp_path, quant_config)


def test_endpoint_contract_is_self_hashing_and_replayable():
    model_sha = "c" * 64
    manifest = _manifest("eager", model_sha=model_sha)
    contract = cbv.build_endpoint_contract(
        arm="eager",
        model_sha=model_sha,
        manifest=manifest,
        manifest_sha256="d" * 64,
        artifact_decode=_artifact_decode_record(),
        endpoint_smoke=_smoke_metrics(),
        cuda_graph=None,
    )
    cbv.validate_endpoint_contract_record(
        contract,
        arm="eager",
        model_sha=model_sha,
        serve_fingerprint=manifest["serve_fingerprint"],
    )
    contract["environment"]["VLLM_USE_DEEP_GEMM"] = "1"
    with pytest.raises(cbv.CBEndpointValidationError, match="VLLM_USE_DEEP_GEMM"):
        cbv.validate_endpoint_contract_record(
            contract,
            arm="eager",
            model_sha=model_sha,
            serve_fingerprint=manifest["serve_fingerprint"],
        )


@pytest.mark.parametrize(
    "mutation",
    ["host", "gpu", "listener_owner", "session", "models"],
)
def test_endpoint_contract_refuses_rehashed_session_tampering(mutation):
    model_sha = "c" * 64
    manifest = _manifest("eager", model_sha=model_sha)
    contract = cbv.build_endpoint_contract(
        arm="eager",
        model_sha=model_sha,
        manifest=manifest,
        manifest_sha256="d" * 64,
        artifact_decode=_artifact_decode_record(),
        endpoint_smoke=_smoke_metrics(),
        cuda_graph=None,
    )
    session = contract["endpoint_session"]
    if mutation == "host":
        session["host_identity"]["boot_id"] = ""
    elif mutation == "gpu":
        session["gpu_uuid"] = "not-a-gpu"
    elif mutation == "listener_owner":
        session["listener_binding"]["listeners"][0]["pids"] = [9999]
    elif mutation == "session":
        session["serve_session_id"] = "f" * 64
    else:
        session["models_endpoint_binding"]["model"]["root"] = "/different"
        session["models_endpoint_binding"]["canonical_identity_sha256"] = (
            models_endpoint_binding_identity(
                models_endpoint_binding_from_bytes(
                    json.dumps(_models_payload(root="/different")).encode(),
                    request_url="http://127.0.0.1:8000/v1/models",
                    expected_served_model=_SERVED_MODEL,
                )
            )["canonical_identity_sha256"]
        )
    contract.pop("contract_sha256")
    contract["contract_sha256"] = cbv._canonical_json_sha256(contract)
    with pytest.raises(cbv.CBEndpointValidationError):
        cbv.validate_endpoint_contract_record(
            contract,
            arm="eager",
            model_sha=model_sha,
            serve_fingerprint=manifest["serve_fingerprint"],
        )


def test_cli_fills_only_the_requested_native_export_slot(tmp_path, monkeypatch):
    artifact, shipcard = _artifact_and_card(tmp_path)
    manifest_path = tmp_path / "serve_manifest.json"
    manifest = _write_bound_manifest(manifest_path, "graph", artifact)
    serve_log = tmp_path / "serve.log"
    serve_log.write_text(
        _RESOLVED_GRAPH_CONFIG
        + "Graph capturing finished in 12 secs, took 3.25 GiB\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cbv,
        "run_endpoint_smoke",
        lambda **kwargs: _smoke_metrics(),
    )
    monkeypatch.setattr(
        cbv,
        "validate_cb_artifact_decode_contract",
        lambda *args, **kwargs: _artifact_decode_record(),
    )

    assert cbv.main([
        "--arm", "graph",
        "--base-url", "http://127.0.0.1:8000/v1",
        "--model-dir", str(artifact),
        "--model-name", _SERVED_MODEL,
        "--serve-manifest", str(manifest_path),
        "--serve-log", str(serve_log),
        "--shipcard", str(shipcard),
    ]) == 0

    card = load_shipcard(shipcard)
    assert card["slots"]["native_export.eager"] is None
    record = card["slots"]["native_export.graph"]
    assert record["passed"] is True
    assert record["serve_fingerprint"] == manifest["serve_fingerprint"]
    assert record["model_sha"] == card["model_sha"]
    assert record["spec_decode_detected"] is False
    assert record["metrics"]["tensor_parallel_size"] == 1
    assert record["metrics"]["cuda_graph"]["serve_log_sha256"]
    assert _verify_gridbook_native_record(
        "native_export.graph", record
    ) == []

    record["metrics"]["endpoint_contract"]["environment"][
        "VLLM_USE_DEEP_GEMM"
    ] = "1"
    assert any(
        "invalid endpoint contract" in problem
        for problem in _verify_gridbook_native_record(
            "native_export.graph", record
        )
    )


def test_deferred_result_closes_slot_only_when_committed(tmp_path, monkeypatch):
    artifact, shipcard = _artifact_and_card(tmp_path)
    manifest_path = tmp_path / "serve_manifest.json"
    _write_bound_manifest(manifest_path, "eager", artifact)
    result_path = tmp_path / "result.json"
    monkeypatch.setattr(
        cbv,
        "run_endpoint_smoke",
        lambda **kwargs: _smoke_metrics(),
    )
    monkeypatch.setattr(
        cbv,
        "validate_cb_artifact_decode_contract",
        lambda *args, **kwargs: _artifact_decode_record(),
    )

    assert cbv.main([
        "--arm", "eager",
        "--base-url", "http://127.0.0.1:8000/v1",
        "--model-dir", str(artifact),
        "--model-name", _SERVED_MODEL,
        "--serve-manifest", str(manifest_path),
        "--shipcard", str(shipcard),
        "--output-json", str(result_path),
        "--defer-shipcard-fill",
    ]) == 0
    assert load_shipcard(shipcard)["slots"]["native_export.eager"] is None

    cbv.commit_deferred_result(result_path, shipcard, artifact)
    assert load_shipcard(shipcard)["slots"]["native_export.eager"]["passed"] is True


@pytest.mark.parametrize("arm", ["eager", "graph"])
def test_deferred_commit_refuses_mutated_hashed_evidence(
    tmp_path, monkeypatch, arm,
):
    artifact, shipcard = _artifact_and_card(tmp_path)
    manifest_path = tmp_path / "serve_manifest.json"
    _write_bound_manifest(manifest_path, arm, artifact)
    result_path = tmp_path / "result.json"
    serve_log = tmp_path / "serve.log"
    if arm == "graph":
        serve_log.write_text(
            _RESOLVED_GRAPH_CONFIG
            + "Graph capturing finished in 12 secs, took 3.25 GiB\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        cbv, "run_endpoint_smoke", lambda **kwargs: _smoke_metrics()
    )
    monkeypatch.setattr(
        cbv,
        "validate_cb_artifact_decode_contract",
        lambda *args, **kwargs: _artifact_decode_record(),
    )
    args = [
        "--arm", arm,
        "--base-url", "http://127.0.0.1:8000/v1",
        "--model-dir", str(artifact),
        "--model-name", _SERVED_MODEL,
        "--serve-manifest", str(manifest_path),
        "--shipcard", str(shipcard),
        "--output-json", str(result_path),
        "--defer-shipcard-fill",
    ]
    if arm == "graph":
        args.extend(["--serve-log", str(serve_log)])
    assert cbv.main(args) == 0

    evidence = manifest_path if arm == "eager" else serve_log
    evidence.write_bytes(evidence.read_bytes() + b"post-validation mutation\n")
    with pytest.raises(cbv.CBEndpointValidationError, match="differs"):
        cbv.commit_deferred_result(result_path, shipcard, artifact)
    assert load_shipcard(shipcard)["slots"][f"native_export.{arm}"] is None


def test_driver_and_lane_declare_two_isolated_arms_and_guards():
    text = DRIVER.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(DRIVER)], check=True)
    assert cbv.DSV4_SPARK_VLLM_IMAGE in text
    assert cbv.DSV4_SPARK_VLLM_VERSION in text
    assert cbv.DSV4_SPARK_VLLM_COMMIT in text
    assert 'install-container' in text
    assert '${VLLM_COMMIT}-${ARM}' in text
    assert 'NAME=${NAME:-pq-dsv4-cb-${ARM}}' in text
    assert '--quantization gridbook' in text
    assert '--tensor-parallel-size 1' in text
    assert '--kv-cache-dtype fp8' in text
    assert '--no-enable-prefix-caching' in text
    assert '--speculative-config' not in text
    assert cbv.DSV4_GRAPH_COMPILATION_CONFIG in text
    assert 'WATCHDOG_GIB' in text and 'READY_FLOOR_GIB' in text
    assert 'START_FLOOR_GIB < 110' in text
    assert 'serve-fingerprint write' in text
    assert '/repo/tools/serve_fingerprint.py write' not in text
    assert 'prismaquant.validate_cb_endpoint' in text
    assert '--defer-shipcard-fill' in text
    assert 'if [[ "$ARM" == eager ]]; then' in text
    assert 'pq_run_module prismaquant.validate_quantized_model' in text
    assert '--model-name "$SERVED_MODEL"' in text
    assert '--artifact-dir "$MODEL"' in text
    assert '--shipcard "$SHIPCARD"' in text
    assert '--max-ppl 25.0' in text
    assert '--max-mean-nll 3.0' in text
    assert '--max-p99-nll 6.0' in text
    assert '--min-gen-len 30' in text
    assert '--min-mtp-accept-p0 0.60' in text
    assert 'required=("ship_gate",)' in text
    assert 'record.get("served_model_name") != served_model' in text
    assert 'record.get("base_url") != base_url' in text
    assert 'record.get("model_sha_source") != model_dir' in text
    assert 'openssl rand -hex 16' in text
    assert 'SERVED_MODEL=dsv4-flash-gridbook-${SERVE_NONCE}' in text
    assert '--base-url http://127.0.0.1:8000' in text
    assert 'commit_deferred_result' in text
    assert 'docker stop -t 30 "$CID"' in text
    assert text.rindex('docker stop -t 30 "$CID"') < text.index(
        'commit_deferred_result'
    )
    endpoint_idx = text.index('pq_run_module prismaquant.validate_cb_endpoint')
    ship_gate_idx = text.index(
        'pq_run_module prismaquant.validate_quantized_model'
    )
    stop_idx = text.rindex('docker stop -t 30 "$CID"')
    assert endpoint_idx < ship_gate_idx < stop_idx
    assert 'contains_mxfp4_wire' in text
    assert '-e GRIDBOOK_MXFP8_DENSE=1' not in text
    assert '-e VLLM_USE_DEEP_GEMM=0' in text
    assert '-e PRISMAQUANT_PRELOAD_FUSED=1' in text
    assert '-e PRISMAQUANT_CB_DECODE=' not in text
    assert 'PRISMAQUANT_CB_DECODE_CONTRACT=v1' in text
    assert 'PRISMAQUANT_CB_FUSED_FP4' in text
    assert 'PRISMAQUANT_CB_FUSED_FP4_MOE' in text
    assert 'validate_cb_artifact_decode_contract' in text
    assert 'EVIDENCE must be outside the immutable MODEL tree' in text
    assert 'EXT_CACHE must be outside the immutable MODEL tree' in text

    endpoint_environment = dict(CANONICAL_GOLD_ENVIRONMENT)
    endpoint_environment.update({
        "PRISMAQUANT_CB_EXT_DIR": "/opt/gridbook/ext-cache",
        "PRISMAQUANT_PRELOAD_FUSED": "1",
    })
    unset_block = text.split("    unset ", 1)[1].split(
        "\n    bash ", 1
    )[0]
    for name, value in endpoint_environment.items():
        if value is None:
            assert f"-e {name}=" not in text
            assert name in unset_block
        else:
            assert f"-e {name}={value}" in text

    lane = load_lane_spec("nvfp4_cb")
    eager = lane.gate("load_generate.eager")
    graph = lane.gate("load_generate.graph")
    assert eager is not None and eager.shipcard_slot == "native_export.eager"
    assert graph is not None and graph.shipcard_slot == "native_export.graph"
    assert "serve_dsv4_cb_validate.sh eager" in eager.runner
    assert "serve_dsv4_cb_validate.sh graph" in graph.runner
    ship_gate = lane.gate("ship_gate.ppl_p99nll")
    assert ship_gate is not None and ship_gate.shipcard_slot == "ship_gate"
    assert "serve_dsv4_cb_validate.sh eager" in ship_gate.description
    assert "nonce-bound server" in ship_gate.description


@pytest.mark.parametrize(
    "extension",
    [
        "prismaquant_cb_ext.so",
        "prismaquant_cb_v2_ext.so",
        "pq_cb_fused_fp4_deadbeef.so",
        "pq_mxfp8_dense_deadbeef.so",
        "pq_fp8_source_w8a16_deadbeef.so",
        "pq_cb_bf16_grouped_deadbeef.so",
    ],
)
def test_manifest_accepts_reviewed_gridbook_native_extension_families(extension):
    manifest = _manifest("eager")
    manifest["resident_extensions"] = [extension]
    manifest["serve_fingerprint"] = fingerprint(manifest)
    cbv.validate_serve_manifest(
        manifest,
        arm="eager",
        expected_served_model=_SERVED_MODEL,
        requires_moe_marlin=False,
    )
