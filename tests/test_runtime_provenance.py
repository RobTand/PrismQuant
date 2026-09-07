"""Synthetic admission contracts, never runtime or resource measurements."""
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import tarfile

import pytest

from prismaquant.measured_runtime_prices import (
    RuntimePriceError, identity_sha256, load_measured_runtime_table,
    parse_runtime_context,
)


def context_payload():
    return {
        "schema": "prismaquant.measured_runtime_context.v2",
        "runtime_identity_kind": "prismaquant.runtime_provenance_relation.v1",
        "serving_context": {"platform": "sm_121", "structure": "dense",
            "residency": "resident", "runtime_image": "example.invalid/runtime@sha256:" + "a" * 64,
            "execution_mode": "eager"},
        "gpu_identity": "synthetic-gpu", "runtime_sha256": "b" * 64,
        "source_sha256": "c" * 64, "calibration_sha256": "d" * 64,
        "prompt_tokens": 512, "batch_size": 1, "tensor_parallel": 1,
        "graph_mode": "eager", "operator_routes": {"layer": {"FP8": "synthetic"}},
    }


def test_relation_context_keeps_a_distinct_derivation_identity():
    payload = context_payload()
    assert parse_runtime_context(payload).as_dict() == payload

from types import SimpleNamespace
from prismaquant.runtime_provenance import (
    ArtifactReader, admit_fixed_resources, admit_native_rows, load_runtime_relation,
)
from prismaquant.measured_runtime_prices import (
    RuntimeBinding, RuntimeResources, OperatorMeasurement, MeasuredRuntimeRow,
    build_runtime_resources, parse_measured_runtime_table,
)
from test_native_operator_panel import joined, receipt_fixture
from test_measured_runtime_prices import payload


class Evidence:
    def __init__(self, root):
        self.root = root

    def raw(self, name, raw):
        path = self.root / name
        path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    def put(self, name, value):
        return self.raw(name, json.dumps(value, sort_keys=True).encode())

    def get(self, reference):
        return json.loads((self.root / reference["path"]).read_bytes())

    def replace(self, reference, value):
        reference.update(self.put(reference["path"], value))


@pytest.fixture
def relation_fixture(tmp_path):
    evidence = Evidence(tmp_path)
    context = context_payload()
    manifest = evidence.put("image-manifest.json", {"schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {"digest": "sha256:" + "8" * 64}})
    image = "example.invalid/runtime@sha256:" + manifest["sha256"]
    context["serving_context"]["runtime_image"] = image
    image_id = "sha256:" + "8" * 64
    config = evidence.put("config.json", {"runtime_image": image, "engine_args": {}, "environment": {}})
    package_files = {name: {"sha256": hashlib.sha256(name.encode()).hexdigest(), "bytes": len(name)}
                     for name in ("__init__.py", "cached_unit.py", "serving/runtime_contract.json")}
    source_files = {name: name.encode() for name in package_files}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, raw in dict(source_files, **{"_dev/example.py": b"synthetic development source"}).items():
            info = tarfile.TarInfo("src/tessera/" + name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    archive_ref = evidence.raw("plugin.tar", buffer.getvalue())
    installed_digest = hashlib.sha256()
    for name, raw in sorted(source_files.items()):
        if name.endswith(".py"):
            installed_digest.update(name.encode() + b"\0" + raw + b"\0")
    installed_sha256 = installed_digest.hexdigest()
    installation = {"registry_base": image, "launcher_declared_image_id": image_id, "core_manifest_sha256": "1" * 64,
                    "core_files_unchanged": 42, "plugin_files": package_files, "plugin_source_commit": "2" * 40,
                    "plugin_archive_sha256": archive_ref["sha256"], "plugin_entrypoints": {"tessera": "tessera.serving:register"}}
    installer = evidence.put("installation.json", installation)
    package = {"schema": "tessera.loaded_package_identity.v1", "package_path": "/installed/tessera",
               "installer_evidence_sha256": installer["sha256"], "encoder_source_sha256": installed_sha256,
               "package_files": package_files, "package_files_unchanged_from_installer": True,
               "module_identity_errors": [], "loaded_tessera_modules": {
                   name: {"file": "/installed/tessera/" + filename, "origin": "/installed/tessera/" + filename,
                          "sha256": package_files[filename]["sha256"]}
                   for name, filename in (("tessera", "__init__.py"), ("tessera.cached_unit", "cached_unit.py"))}}
    post_package = evidence.put("package.json", package)
    runs = {}
    for name, scope in (("native", "native_operator"), ("engine", "full_engine")):
        binary = evidence.raw(name + "-observer.so", (name + " observer bytes").encode())
        source = evidence.raw(name + "-observer.cpp", b"synthetic collector source")
        build = evidence.put(name + "-build.json", {"source_sha256": source["sha256"], "output_sha256": binary["sha256"]})
        harness = evidence.raw(name + "-harness.py", b"synthetic harness source")
        analysis = evidence.raw(name + "-analysis.py", b"synthetic analysis source")
        loaded_path = "/measurement/" + name + "-observer.so"
        base = {"schema": "tessera.native_dense_runtime.v1", "image": image,
                "execution": {"mode": "resident", "execution_mode": "eager", "tensor_parallel": 1},
                "gpu": {"uuid": "synthetic-gpu", "capability": [12, 1]},
                "versions": {"torch": "synthetic"}, "arithmetic": {"tf32": False},
                "source": {"tessera_package_sha256": installed_sha256,
                           "runtime_contract_sha256": package_files["serving/runtime_contract.json"]["sha256"],
                           "harness_sha256": harness["sha256"]},
                "image_declaration": {"record": {"refused": False, "present": True, "gated": True,
                    "pinned": image, "resolved_reference": image, "requested": image,
                    "repo_digests": [image], "local_id": image_id, "selection": {"configuration_sha256": config["sha256"]}}},
                "native_libraries": {"/usr/lib/libtorch.so": "5" * 64, loaded_path: binary["sha256"]}}
        observer = {"library_sha256": binary["sha256"], "loaded_path": loaded_path, "analysis_source_sha256": analysis["sha256"]}
        sources = {"harness_sha256": harness}
        if scope == "native_operator":
            base["resource_collector"] = observer
            raw = base
            sources["resource_analysis_source_sha256"] = analysis
            audit = {"native_returncode": 0, "manifest_sha256": "1" * 64, "stock_files_unchanged": 42}
        else:
            raw = {"schema": "tessera.full_engine_runtime.v1", "base": base,
                   "actual_execution": base["execution"], "configuration_sha256": config["sha256"],
                   "loaded_package": package, "execution": {"engine_args": {}, "environment": {}},
                   "source": {"full_engine_worker_sha256": harness["sha256"]},
                   "instrumentation": {"resource_collector": observer}}
            sources["full_engine_worker_sha256"] = harness
            base["native_libraries"]["/usr/lib/engine-extra.so"] = "6" * 64
            audit = {"core_audit_" + phase: {"manifest_sha256": "1" * 64, "unchanged_files": 42}
                     for phase in ("before", "after")}
        runs[name] = {"scope": scope, "runtime_field": None, "runtime": evidence.put(name + "-runtime.json", raw),
                      "installation": copy.deepcopy(installer), "post_core": evidence.put(name + "-audit.json", audit),
                      "post_package": copy.deepcopy(post_package),
                      "instrumentation": {"artifacts": {}, "libraries": [{"role": "resource_collector", "loaded_path": loaded_path,
                          "artifact": binary, "source": source, "build_receipt": build}], "python_sources": sources}}
    relation = {"schema": "prismaquant.runtime_provenance_relation.v1", "configuration": config, "image_manifest": manifest, "runs": runs,
                "package_source": {"archive": archive_ref, "prefix": "src/tessera", "excluded_files": ["_dev/example.py"]},
                "full_engine_run_id": "engine", "production_dependencies": [{"native_run_id": "native",
                    "native_path": "/usr/lib/libtorch.so", "full_engine_path": "/usr/lib/libtorch.so", "sha256": "5" * 64}],
                "full_engine_extra_libraries": {"/usr/lib/engine-extra.so": {"sha256": "6" * 64, "scope": "full_engine"}}}
    return evidence, relation, context


def relation_load(fixture):
    evidence, relation, context = fixture
    context = dict(context, runtime_sha256=identity_sha256(relation))
    return load_runtime_relation(evidence.put("relation.json", relation), context=parse_runtime_context(context), root=evidence.root)


def test_relation_preserves_distinct_raw_manifests_and_exhaustive_dependencies(relation_fixture):
    evidence, original, _ = relation_fixture
    admitted = relation_load(relation_fixture)
    assert admitted["record"] == original
    native, engine = admitted["runs"]["native"], admitted["runs"]["engine"]
    assert native["sha256"] != engine["sha256"] != identity_sha256(original)
    assert native["sha256"] == identity_sha256(evidence.get(original["runs"]["native"]["runtime"]))


@pytest.mark.parametrize("mutation", ["common_bytes", "missing_dependency", "undeclared_extra", "missing_mapping",
    "different_core", "different_package", "foreign_config", "foreign_gpu", "boolean_tp", "missing_origin",
    "foreign_origin", "module_bytes", "missing_module", "installer_drift", "missing_source", "binary_drift",
    "installed_instrumentation", "runtime_hash", "arithmetic"])
def test_relation_refuses_unproved_or_changed_coordinates(relation_fixture, mutation):
    evidence, relation, _ = relation_fixture
    run = relation["runs"]["native"]
    raw = evidence.get(run["runtime"])
    if mutation == "common_bytes":
        raw["native_libraries"]["/usr/lib/libtorch.so"] = "7" * 64
    elif mutation == "missing_dependency":
        raw["native_libraries"]["/usr/lib/missing.so"] = "7" * 64
    elif mutation == "undeclared_extra":
        relation["full_engine_extra_libraries"] = {}
    elif mutation == "missing_mapping":
        relation["production_dependencies"] = []
    elif mutation in ("different_core", "different_package", "installer_drift"):
        install = evidence.get(run["installation"])
        install["core_manifest_sha256" if mutation == "different_core" else "plugin_archive_sha256"] = "7" * 64
        evidence.replace(run["installation"], install)
    elif mutation == "foreign_config":
        raw["image_declaration"]["record"]["selection"]["configuration_sha256"] = "7" * 64
    elif mutation == "foreign_gpu":
        raw["gpu"]["uuid"] = "foreign"
    elif mutation == "boolean_tp":
        raw["execution"]["tensor_parallel"] = True
    elif mutation in ("missing_origin", "foreign_origin", "module_bytes", "missing_module"):
        package = evidence.get(run["post_package"])
        module = package["loaded_tessera_modules"]["tessera.cached_unit"]
        if mutation == "missing_origin":
            module["origin"] = None
        elif mutation == "foreign_origin":
            module["origin"] = "/elsewhere/cached_unit.py"
        elif mutation == "module_bytes":
            module["sha256"] = "7" * 64
        else:
            del package["loaded_tessera_modules"]["tessera.cached_unit"]
        evidence.replace(run["post_package"], package)
    elif mutation == "missing_source":
        run["instrumentation"]["python_sources"] = {}
    elif mutation == "binary_drift":
        run["instrumentation"]["libraries"][0]["artifact"]["sha256"] = "7" * 64
    elif mutation == "installed_instrumentation":
        library = run["instrumentation"]["libraries"][0]
        raw["native_libraries"]["/usr/lib/observer.so"] = raw["native_libraries"].pop(library["loaded_path"])
        library["loaded_path"] = "/usr/lib/observer.so"
    elif mutation == "runtime_hash":
        run["runtime"]["sha256"] = "7" * 64
    elif mutation == "arithmetic":
        raw["arithmetic"]["tf32"] = True
    if mutation != "runtime_hash":
        evidence.replace(run["runtime"], raw)
    with pytest.raises(RuntimePriceError):
        relation_load(relation_fixture)


@pytest.mark.parametrize("claim", [None, {}, {"status": "complete", "resident_bytes": 0}])
def test_relation_never_admits_unproved_fixed_resources(tmp_path, claim):
    evidence = Evidence(tmp_path)
    ref = evidence.put("fixed.json", {"full_model_resources": claim, "full_model_fixed_resources_complete": True,
                                     "status": "qualified_complete", "closure": {"complete": True}})
    table = SimpleNamespace(source_path=str(tmp_path / "table.json"), fixed_resources_receipt_path=ref["path"],
                            fixed_resources_receipt_sha256=ref["sha256"])
    with pytest.raises(RuntimePriceError, match="incomplete|no qualified recomputable"):
        admit_fixed_resources(table, {})


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'[]'])
def test_artifacts_refuse_ambiguous_json(tmp_path, raw):
    ref = Evidence(tmp_path).raw("ambiguous.json", raw)
    with pytest.raises(RuntimePriceError):
        ArtifactReader(tmp_path).json(ref, "fixture")

@pytest.mark.parametrize("mutation", ["wrong_scope", "build_source", "build_output", "engine_config", "worker_source"])
def test_relation_refuses_unbound_observer_evidence(relation_fixture, mutation):
    evidence, relation, _ = relation_fixture
    run = relation["runs"]["engine"]
    if mutation == "wrong_scope":
        relation["full_engine_extra_libraries"]["/usr/lib/engine-extra.so"]["scope"] = "native_operator"
    elif mutation.startswith("build_"):
        ref = run["instrumentation"]["libraries"][0]["build_receipt"]
        build = evidence.get(ref)
        build["source_sha256" if mutation == "build_source" else "output_sha256"] = "7" * 64
        evidence.replace(ref, build)
    elif mutation == "worker_source":
        del run["instrumentation"]["python_sources"]["full_engine_worker_sha256"]
    else:
        raw = evidence.get(run["runtime"])
        raw["execution"]["engine_args"]["unexpected_override"] = True
        evidence.replace(run["runtime"], raw)
    with pytest.raises(RuntimePriceError):
        relation_load(relation_fixture)


def test_relation_keeps_original_preflight_envelope(relation_fixture):
    evidence, relation, _ = relation_fixture
    run = relation["runs"]["native"]
    raw = evidence.get(run["runtime"])
    envelope = {"schema": "tessera.native_dense_preflight.v1", "runtime": raw,
                "runtime_sha256": identity_sha256(raw), "original_other_field": "retained"}
    run["runtime"] = evidence.put("original-preflight.json", envelope)
    run["runtime_field"] = "runtime"
    result = relation_load(relation_fixture)
    assert result["runs"]["native"]["raw"] == raw
    assert evidence.get(run["runtime"]) == envelope


@pytest.fixture
def native_intake(relation_fixture, joined):
    evidence, relation, context = relation_fixture
    inputs, preflight, joint = joined
    raw = evidence.get(relation["runs"]["native"]["runtime"])
    inputs["runtime_image"] = raw["image"]
    source_tree_sha = relation_load(relation_fixture)["runs"]["native"]["common"]["producer_source_tree_sha256"]
    inputs["wire"]["record"]["identity"] = {"encoder_source_sha256": source_tree_sha}
    preflight["operator"]["wire_record_sha256"] = identity_sha256(inputs["wire"]["record"])
    raw["execution"] = copy.deepcopy(preflight["runtime"]["execution"])
    evidence.replace(relation["runs"]["native"]["runtime"], raw)
    preflight["runtime"] = raw
    preflight["runtime_sha256"] = identity_sha256(raw)
    panel, receipt, trace = receipt_fixture(joined, complete=True)
    trace["capture"]["collector_library_sha256"] = raw["resource_collector"]["library_sha256"]
    receipt["resources"]["trace_sha256"] = identity_sha256(trace)
    context.update(prompt_tokens=1, source_sha256=panel["source_sha256"],
                   calibration_sha256=panel["calibration_sha256"],
                   runtime_sha256=identity_sha256(relation),
                   operator_routes={panel["unit"]: {panel["format"]: "torch.mm"}})
    panel_ref, receipt_ref, trace_ref = (evidence.put(name, value) for name, value in (
        ("panel.json", panel), ("receipt.json", receipt), ("trace.json", trace)))
    measurement = OperatorMeasurement.from_dict({"method": "cuda_events", "samples_ms": [3., 1., 2.],
        "warmup_iterations": 4, "receipt_path": receipt_ref["path"], "receipt_sha256": receipt_ref["sha256"]})
    binding = RuntimeBinding.from_dict({"member_formats": {panel["unit"]: panel["format"]},
        "member_operator_identity_sha256": {panel["unit"]: panel["joint_operator_identity_sha256"]},
        "member_shapes": {panel["unit"]: panel["shape"]}, "operator_route": "torch.mm"})
    row = MeasuredRuntimeRow(panel["unit"], panel["format"], binding,
        RuntimeResources(prefill_ms=2., decode_ms=2., serialized_bytes=42, resident_bytes=64,
                         peak_scratch_bytes=128, activation_bytes=8, kv_bytes=0), measurement, measurement)
    table = SimpleNamespace(source_path=str(evidence.root / "table.json"), rows=(row,),
        context=parse_runtime_context(context), cost_sha256=panel["cost_sha256"],
        native_receipt_bindings=[{"unit": row.unit, "format": row.fmt, "run_id": "native",
            "panel": panel_ref, "receipt": receipt_ref, "memory_trace": trace_ref}])
    return evidence, table, relation_load(relation_fixture)


def test_native_rows_reuse_original_same_run_numerical_and_resource_gates(native_intake):
    _, table, relation = native_intake
    admit_native_rows(table, relation)


@pytest.mark.parametrize("mutation", ["foreign_runtime", "foreign_cost", "foreign_source", "foreign_calibration",
                                     "wrong_scope", "missing_row", "wrong_tokens", "foreign_trace", "wrong_samples", "installed_source_relabel"])
def test_native_rows_refuse_cross_run_relabel_or_scope_drift(native_intake, mutation):
    evidence, table, relation = native_intake
    binding = table.native_receipt_bindings[0]
    panel = evidence.get(binding["panel"])
    if mutation == "installed_source_relabel":
        panel["wire"]["record"]["identity"]["encoder_source_sha256"] = relation["runs"]["native"]["common"]["package_sha256"]
    elif mutation == "foreign_runtime":
        panel["runtime"] = relation["runs"]["engine"]["raw"]
    elif mutation in ("foreign_cost", "foreign_source", "foreign_calibration"):
        panel[mutation.removeprefix("foreign_") + "_sha256"] = "0" * 64
    elif mutation == "wrong_scope":
        binding["run_id"] = "engine"
    elif mutation == "missing_row":
        table.native_receipt_bindings = []
    elif mutation == "wrong_tokens":
        table.context = parse_runtime_context(dict(table.context.as_dict(), prompt_tokens=512))
    elif mutation == "foreign_trace":
        trace = evidence.get(binding["memory_trace"])
        trace["capture"]["collector_library_sha256"] = "0" * 64
        evidence.replace(binding["memory_trace"], trace)
    else:
        receipt = evidence.get(binding["receipt"])
        receipt["phases"]["decode"]["measurement"]["samples_ms"] = [4., 1., 2.]
        evidence.replace(binding["receipt"], receipt)
    evidence.replace(binding["panel"], panel)
    with pytest.raises(RuntimePriceError):
        admit_native_rows(table, relation)


def test_v2_parsing_cannot_supply_unadmitted_allocation_resources(payload):
    payload["schema"] = "prismaquant.measured_runtime_prices.v2"
    payload["context"].update(schema="prismaquant.measured_runtime_context.v2",
        runtime_identity_kind="prismaquant.runtime_provenance_relation.v1")
    payload.update(runtime_provenance={"path": "relation.json", "sha256": "1" * 64}, native_receipt_bindings=[])
    table = parse_measured_runtime_table(payload, expected_context=parse_runtime_context(payload["context"]),
        expected_cost_sha256=payload["cost_sha256"], now=datetime(2026, 9, 5, tzinfo=timezone.utc))
    assert table.as_dict()["schema"] == payload["schema"]
    assert table.as_dict()["context"] == payload["context"]
    with pytest.raises(RuntimePriceError, match="producer admission"):
        build_runtime_resources(table, {}, expected_bindings={})


def test_image_manifest_explicitly_relates_manifest_and_config_local_id_types(relation_fixture):
    evidence, relation, _ = relation_fixture
    run = relation["runs"]["native"]
    raw = evidence.get(run["runtime"])
    raw["image_declaration"]["record"]["local_id"] = "sha256:" + relation["image_manifest"]["sha256"]
    evidence.replace(run["runtime"], raw)
    installation = evidence.get(run["installation"])
    installation["launcher_declared_image_id"] = raw["image_declaration"]["record"]["local_id"]
    run["installation"] = evidence.put("native-installation.json", installation)
    package = evidence.get(run["post_package"])
    package["installer_evidence_sha256"] = run["installation"]["sha256"]
    run["post_package"] = evidence.put("native-package.json", package)
    admitted = relation_load(relation_fixture)
    assert admitted["runs"]["native"]["common"]["image_identity"] == {
        "manifest_digest": "sha256:" + relation["image_manifest"]["sha256"], "config_digest": "sha256:" + "8" * 64}
    assert admitted["runs"]["native"]["raw"] == raw


@pytest.mark.parametrize("mutation", ["unrelated_id", "changed_manifest", "changed_config"])
def test_image_relation_refuses_unproved_identity_types(relation_fixture, mutation):
    evidence, relation, _ = relation_fixture
    if mutation == "unrelated_id":
        run = relation["runs"]["native"]
        raw = evidence.get(run["runtime"])
        raw["image_declaration"]["record"]["local_id"] = "sha256:" + "7" * 64
        evidence.replace(run["runtime"], raw)
    else:
        ref = relation["image_manifest"]
        manifest = evidence.get(ref)
        manifest["config"]["digest"] = "sha256:" + "7" * 64
        if mutation == "changed_manifest":
            evidence.replace(ref, manifest)
        else:
            (evidence.root / ref["path"]).write_text(json.dumps(manifest))
    with pytest.raises(RuntimePriceError):
        relation_load(relation_fixture)


def test_nested_boolean_arithmetic_cannot_equal_numeric_flag(relation_fixture):
    evidence, relation, _ = relation_fixture
    run = relation["runs"]["native"]
    raw = evidence.get(run["runtime"])
    raw["arithmetic"]["tf32"] = 0
    evidence.replace(run["runtime"], raw)
    with pytest.raises(RuntimePriceError, match="arithmetic"):
        relation_load(relation_fixture)


def test_actual_multi_build_receipt_binds_observer_source_and_output(relation_fixture):
    evidence, relation, _ = relation_fixture
    library = relation["runs"]["native"]["instrumentation"]["libraries"][0]
    name = "native-observer.so"
    evidence.replace(library["build_receipt"], {"builds": [{"name": name, "returncode": 0}],
        "files": {name: {"sha256": library["artifact"]["sha256"]}},
        "source_files": {"native-observer.cpp": library["source"]["sha256"]}})
    relation_load(relation_fixture)


def test_source_tree_and_installed_package_have_distinct_recomputed_roles(relation_fixture):
    admitted = relation_load(relation_fixture)
    common = admitted["runs"]["native"]["common"]
    assert common["producer_source_tree_sha256"] != common["package_sha256"]


@pytest.mark.parametrize("mutation", ["undeclared_omission", "unknown_omission", "archive_drift", "source_relabel"])
def test_package_archive_refuses_unproved_installed_subset(relation_fixture, mutation):
    evidence, relation, _ = relation_fixture
    declaration = relation["package_source"]
    if mutation == "undeclared_omission":
        declaration["excluded_files"] = []
    elif mutation == "unknown_omission":
        declaration["excluded_files"].append("not_in_archive.py")
    elif mutation == "archive_drift":
        declaration["archive"]["sha256"] = "0" * 64
    else:
        run = relation["runs"]["native"]
        package = evidence.get(run["post_package"])
        package["encoder_source_sha256"] = "0" * 64
        evidence.replace(run["post_package"], package)
    with pytest.raises(RuntimePriceError):
        relation_load(relation_fixture)


@pytest.mark.parametrize("coordinate", ["platform", "graph_mode"])
def test_relation_checks_device_and_execution_against_context(relation_fixture, coordinate):
    _, _, context = relation_fixture
    if coordinate == "platform":
        context["serving_context"]["platform"] = "sm_100"
    else:
        context["graph_mode"] = "full"
    with pytest.raises(RuntimePriceError, match="actual GPU platform|actual graph mode"):
        relation_load(relation_fixture)
