"""A declared native treatment changes only its exact extension residency."""
import copy

import pytest

from prismaquant import boundary_control as bc
from tools import serve_fingerprint as sf
from test_boundary_control import _binding, _contract, _response


IMAGE = "fixture@sha256:" + "d" * 64


def _declared_pair():
    bindings = {"bf16_control": _binding(), "candidate": _binding("quant")}
    extensions = {"bf16_control": ["sampling.so"],
                  "candidate": ["sampling.so", "trtllm_utils.so"]}
    contract = {"schema": "prismaquant.boundary_stack_contract/1", "image": IMAGE,
                "roles": {role: {"artifact_id": bound["artifact_id"],
                                  "resident_extensions": extensions[role]}
                          for role, bound in bindings.items()}}
    for role, bound in bindings.items():
        manifest = {
            "image": IMAGE, "gpu_name": "GB10", "gpu_uuid": "gpu-1", "gpu_count": 1,
            "driver_version": "595.84", "compute_capability": "12.1",
            "gpu_compute_capabilities": ["12.1"], "package_versions": {"vllm": "pinned"},
            "vllm_compilation_provenance": {"source": "pinned"},
            "resident_extensions": extensions[role], "residency_readable": True,
            "normalized_performance_argv": ["vllm", "serve", "<arm>", "--enforce-eager"],
            "server_process_environment": {"values": {"OMP_NUM_THREADS": "2"}},
            "listener_binding": {"base_url": "http://server", "launch_host": "127.0.0.1",
                                 "launch_port": "8187"},
            "serve_session_id": bound["serve_session_id"],
            "host_identity": {"boot_id": bound["host_boot_id"]},
        }
        manifest["performance_stack_fingerprint"] = sf.performance_stack_fingerprint(manifest)
        bound["serve_fingerprint"] = manifest["performance_stack_fingerprint"]
        bound["paired_stack"] = {"schema": "prismaquant.boundary_stack_binding/1",
                                 "role": role, "contract": copy.deepcopy(contract),
                                 "manifest": manifest}
    return bindings


def _control(monkeypatch, bindings):
    monkeypatch.setattr(bc.vqm, "_post_json", lambda _url, body: _response(body["max_tokens"]))
    return bc.measure_control("http://server", "control", _contract(), bindings["bf16_control"])


def _rehash(binding):
    manifest = binding["paired_stack"]["manifest"]
    manifest["performance_stack_fingerprint"] = sf.performance_stack_fingerprint(manifest)
    binding["serve_fingerprint"] = manifest["performance_stack_fingerprint"]


def test_declared_native_treatment_pairs_with_raw_fingerprints_retained(monkeypatch):
    bindings = _declared_pair()
    assert bindings["bf16_control"]["serve_fingerprint"] != bindings["candidate"]["serve_fingerprint"]
    control = _control(monkeypatch, bindings)
    candidate = bc.measure_candidate("http://server", "candidate", control, bindings["candidate"])
    decision = bc.decide_no_new_failures(control, candidate)
    assert decision["verdict"] == "accepted"
    assert bc.replay_no_new_failures(decision, control, candidate) == decision
    assert control["schema"] == candidate["schema"] == "prismaquant.boundary_control/2"
    assert control["binding"]["serve_fingerprint"] != candidate["binding"]["serve_fingerprint"]


@pytest.mark.parametrize("field,value", [
    ("image", "other@sha256:" + "e" * 64), ("gpu_name", "other"),
    ("gpu_uuid", "gpu-2"), ("gpu_count", 2), ("driver_version", "other"),
    ("compute_capability", "9.0"), ("gpu_compute_capabilities", ["9.0"]),
    ("package_versions", {"vllm": "other"}),
    ("vllm_compilation_provenance", {"source": "other"}),
    ("normalized_performance_argv", ["vllm", "serve", "<arm>"]),
    ("server_process_environment", {"values": {"OMP_NUM_THREADS": "4"}}),
    ("listener_binding", {"base_url": "http://other", "launch_host": "127.0.0.1", "launch_port": "8187"}),
])
def test_declared_residency_never_waives_other_stack_fields(monkeypatch, field, value):
    bindings = _declared_pair()
    control = _control(monkeypatch, bindings)
    candidate = bindings["candidate"]
    candidate["paired_stack"]["manifest"][field] = value
    _rehash(candidate)
    monkeypatch.setattr(bc.vqm, "_post_json", lambda *_args: pytest.fail("unpaired stack reached server"))
    with pytest.raises(ValueError):
        bc.measure_candidate("http://server", "candidate", control, candidate)


@pytest.mark.parametrize("extensions", [["sampling.so"], ["sampling.so", "trtllm_utils.so", "extra.so"]])
def test_exact_role_residency_refuses_missing_and_extra_extensions(monkeypatch, extensions):
    bindings = _declared_pair()
    control = _control(monkeypatch, bindings)
    candidate = bindings["candidate"]
    candidate["paired_stack"]["manifest"]["resident_extensions"] = extensions
    _rehash(candidate)
    with pytest.raises(ValueError, match="resident_extensions"):
        bc.measure_candidate("http://server", "candidate", control, candidate)


def test_raw_fingerprint_is_recomputed_not_taken_from_binding(monkeypatch):
    bindings = _declared_pair()
    control = _control(monkeypatch, bindings)
    candidate = bindings["candidate"]
    candidate["serve_fingerprint"] = control["binding"]["serve_fingerprint"]
    with pytest.raises(ValueError, match="raw.*fingerprint"):
        bc.measure_candidate("http://server", "candidate", control, candidate)


@pytest.mark.parametrize("mutation", ["role", "artifact", "declaration", "missing", "unreadable", "campaign"])
def test_stack_proof_cannot_be_swapped_or_dropped(monkeypatch, mutation):
    bindings = _declared_pair()
    control = _control(monkeypatch, bindings)
    candidate = bindings["candidate"]
    proof = candidate["paired_stack"]
    if mutation == "role":
        proof["role"] = "bf16_control"
    elif mutation == "artifact":
        proof["contract"]["roles"]["candidate"]["artifact_id"] = "a" * 64
    elif mutation == "declaration":
        proof["contract"]["roles"]["bf16_control"]["resident_extensions"] = []
    elif mutation == "missing":
        del candidate["paired_stack"]
    elif mutation == "campaign":
        candidate["campaign_id"] = "another-campaign"
    else:
        proof["manifest"]["residency_readable"] = False
        _rehash(candidate)
    with pytest.raises(ValueError):
        bc.measure_candidate("http://server", "candidate", control, candidate)


def test_declared_receipt_cannot_be_downgraded_to_legacy_schema(monkeypatch):
    control = _control(monkeypatch, _declared_pair())
    control["schema"] = "prismaquant.boundary_control/1"
    with pytest.raises(ValueError, match="schema"):
        bc.replay_control(control)


def test_declared_control_cannot_erase_proof_under_new_schema(monkeypatch):
    control = _control(monkeypatch, _declared_pair())
    del control["binding"]["paired_stack"]
    with pytest.raises(ValueError, match="schema"):
        bc.replay_control(control)


def test_projected_stack_comparison_preserves_json_types(monkeypatch):
    bindings = _declared_pair()
    control = _control(monkeypatch, bindings)
    candidate = bindings["candidate"]
    candidate["paired_stack"]["manifest"]["gpu_count"] = True
    _rehash(candidate)
    with pytest.raises(ValueError, match="gpu_count"):
        bc.measure_candidate("http://server", "candidate", control, candidate)
