"""The serving attestations and comparator that refuses stack drift (R15).

§7.4's rule was prose: "A/B arms must have identical extension residency; deltas
under ~±20% across differing serving stacks are not evidence". These tests pin
the two halves of the mechanization: ``performance_stack_fingerprint`` ignores
which artifact/session was served so a legitimate A/B still compares, while
``serve_fingerprint`` remains a validated per-run artifact/session attestation.
The comparator refuses a cross-stack delta and downgrades it to an honest range
only when explicitly overridden.
"""
from __future__ import annotations

import json
import pathlib

import pytest

if not (pathlib.Path(__file__).resolve().parents[1] / "tools").is_dir():
    pytest.skip("requires a repo checkout (tools/ scripts)",
                allow_module_level=True)

from tools.kl_ab import CROSS_STACK_BAND, compare
from tools.kl_ab import main as kl_ab_cli
import tools.serve_fingerprint as serve_fingerprint
from tools.serve_fingerprint import (
    EXTENSION_PATTERN,
    collect_manifest,
    elide_argv_paths,
    fingerprint,
    find_in_process_server_pids,
    gridbook_runtime_pin,
    manifest_differences,
    performance_stack_fingerprint,
    resident_extensions,
    self_manifest,
)

BASE_MANIFEST = {
    "image": "vllm-node:latest",
    "gpu_name": "NVIDIA GB10",
    "driver_version": "595.42",
    "enforce_eager": True,
    "quantization": "compressed-tensors",
    "package_versions": {"vllm": "0.21.0", "torch": "2.11.0"},
    "resident_extensions": ["_gridbook_C.so"],
    "launch_flags": ["vllm", "serve", "<path>", "--enforce-eager"],
    # excluded from the fingerprint:
    "created": "2026-07-30T10:00:00",
    "launch_argv": ["vllm", "serve", "/dqruns/a/exported", "--enforce-eager"],
    "model": "/dqruns/a/exported",
    "processes": [{"pid": 1, "cmdline": "vllm serve"}],
}


def _result(value, *, manifest=None, metric="kl_confident_mean", model="/a"):
    payload = {"model": model, metric: value}
    if manifest is not None:
        signed = dict(manifest)
        signed["performance_stack_fingerprint"] = (
            performance_stack_fingerprint(signed)
        )
        signed["serve_fingerprint"] = fingerprint(signed)
        payload["serve_manifest"] = signed
        payload["serve_fingerprint"] = signed["serve_fingerprint"]
    return payload


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def test_argv_paths_are_elided_so_an_ab_shares_a_fingerprint():
    argv = ["vllm", "serve", "/dqruns/arm-a/exported", "--max-model-len", "8192"]
    assert elide_argv_paths(argv) == [
        "vllm", "serve", "<path>", "--max-model-len", "8192"]

    arm_a = dict(BASE_MANIFEST)
    arm_b = dict(BASE_MANIFEST,
                 model="/dqruns/b/exported",
                 launch_argv=["vllm", "serve", "/dqruns/b/exported"],
                 created="2026-07-31T11:00:00",
                 processes=[{"pid": 7, "cmdline": "vllm serve"}])
    assert fingerprint(arm_a) == fingerprint(arm_b), (
        "two artifacts served the same way must share a fingerprint, or every "
        "A/B would refuse itself")


@pytest.mark.parametrize("key, value", [
    ("resident_extensions", []),                       # the ±17% mechanism
    ("enforce_eager", False),
    ("image", "vllm-node:2026-07-01"),
    ("quantization", None),
    ("package_versions", {"vllm": "0.22.0", "torch": "2.11.0"}),
    ("launch_flags", ["vllm", "serve", "<path>"]),
    ("gpu_name", "NVIDIA H100"),
    ("gridbook_runtime_pin", {
        "commit": "f" * 40, "version": "0.4.1",
    }),
])
def test_stack_changes_move_the_fingerprint(key, value):
    changed = dict(BASE_MANIFEST, **{key: value})
    assert fingerprint(changed) != fingerprint(BASE_MANIFEST)
    assert key in manifest_differences(BASE_MANIFEST, changed)


def test_extension_pattern_matches_the_tracked_sos():
    for path in (
        "/gb_snap/gridbook/_gridbook_C.cpython-312-aarch64-linux-gnu.so",
        "/usr/lib/python3/site-packages/flashinfer/_kernels.so",
        "/usr/lib/python3/site-packages/causal_conv1d/_C.so",
        "/usr/lib/python3/site-packages/fla/ops/_triton.so",
        "/repo/prismaquant/kernels/nvfp4_fused.so",
    ):
        assert EXTENSION_PATTERN.search(path), path
    assert not EXTENSION_PATTERN.search("/usr/lib/libcudart.so.13")


def test_external_gridbook_pin_without_distribution_fails_closed(monkeypatch):
    monkeypatch.setenv("PQ_GRIDBOOK_RUNTIME_COMMIT", "a" * 40)
    monkeypatch.setenv("PQ_GRIDBOOK_RUNTIME_VERSION", "0.4.1")
    monkeypatch.setattr(
        serve_fingerprint.importlib_metadata,
        "distribution",
        lambda name: (_ for _ in ()).throw(
            serve_fingerprint.importlib_metadata.PackageNotFoundError(name)
        ),
    )
    assert gridbook_runtime_pin() == {
        "commit": "a" * 40,
        "version": "0.4.1",
    }
    with pytest.raises(ValueError, match="distribution is not installed"):
        collect_manifest(
            pids=[__import__("os").getpid()],
            launch_argv=["vllm", "serve", "/m"],
        )


def test_external_gridbook_wheel_pin_is_forwarded_to_attestation(monkeypatch):
    monkeypatch.setenv("PQ_GRIDBOOK_RUNTIME_COMMIT", "a" * 40)
    monkeypatch.setenv("PQ_GRIDBOOK_RUNTIME_VERSION", "0.8.5")
    monkeypatch.setenv("PQ_GRIDBOOK_RUNTIME_WHEEL_SHA256", "b" * 64)
    distribution = {
        "schema": "prismaquant.installed_gridbook_distribution/2",
        "import_origin": {"schema": "prismaquant.gridbook_import_origin/1"},
    }

    def attest(pin):
        assert pin == {
            "repository": serve_fingerprint.GRIDBOOK_REPOSITORY,
            "commit": "a" * 40,
            "version": "0.8.5",
            "wheel_sha256": "b" * 64,
        }
        return distribution

    monkeypatch.setattr(
        serve_fingerprint, "gridbook_distribution_provenance", attest
    )
    manifest = collect_manifest(
        pids=[__import__("os").getpid()],
        launch_argv=["vllm", "serve", "/m"],
    )
    assert manifest["gridbook_runtime_pin"]["wheel_sha256"] == "b" * 64
    assert manifest["gridbook_distribution"] == distribution


def test_external_gridbook_pin_and_distribution_are_recorded_in_stack(
    monkeypatch,
):
    monkeypatch.setenv("PQ_GRIDBOOK_RUNTIME_COMMIT", "a" * 40)
    monkeypatch.setenv("PQ_GRIDBOOK_RUNTIME_VERSION", "0.4.1")
    distribution = {
        "schema": "prismaquant.installed_gridbook_distribution/2",
        "import_origin": {
            "schema": "prismaquant.gridbook_import_origin/1",
        },
    }

    def attest(pin):
        assert pin == {
            "repository": serve_fingerprint.GRIDBOOK_REPOSITORY,
            "commit": "a" * 40,
            "version": "0.4.1",
        }
        return distribution

    monkeypatch.setattr(
        serve_fingerprint, "gridbook_distribution_provenance", attest
    )
    manifest = collect_manifest(
        pids=[__import__("os").getpid()], launch_argv=["vllm", "serve", "/m"])
    assert manifest["gridbook_runtime_pin"] == gridbook_runtime_pin()
    assert manifest["gridbook_distribution"] == distribution


def test_gridbook_mxfp8_lane_opt_in_is_fingerprinted(monkeypatch):
    monkeypatch.setenv("GRIDBOOK_MXFP8_DENSE", "1")
    manifest = collect_manifest(
        pids=[__import__("os").getpid()], launch_argv=["vllm", "serve", "/m"]
    )
    assert manifest["pq_env"]["GRIDBOOK_MXFP8_DENSE"] == "1"


def test_self_manifest_reads_this_process(tmp_path):
    manifest = self_manifest(image="test-image")
    assert manifest["source"] == "in_process"
    assert manifest["image"] == "test-image"
    assert len(manifest["serve_fingerprint"]) == 64
    assert isinstance(manifest["resident_extensions"], list)
    # recomputing over the written manifest is stable (the fingerprint field
    # itself is excluded from its own input)
    round_tripped = json.loads(json.dumps(manifest))
    assert fingerprint(round_tripped) == manifest["serve_fingerprint"]


def test_in_process_manifest_discovers_only_its_vllm_descendants(monkeypatch):
    parents = {10: 1, 11: 10, 12: 10, 13: 12, 14: 999}
    monkeypatch.setattr(
        serve_fingerprint.os,
        "listdir",
        lambda path: [str(pid) for pid in parents],
    )
    monkeypatch.setattr(
        serve_fingerprint,
        "_process_parent_pid",
        lambda pid: parents[int(pid)],
    )
    monkeypatch.setattr(
        serve_fingerprint,
        "_looks_like_vllm_process",
        lambda pid, pattern="vllm": pid in {11, 13, 14},
    )

    assert find_in_process_server_pids(10) == [10, 11, 13]


def test_unreadable_maps_never_masquerade_as_nothing_resident():
    """Reading a root-owned container process's maps from the host is denied,
    and that denial looks exactly like an empty extension list."""
    assert resident_extensions([999999999]) == []

    blind = collect_manifest(pids=[999999999])
    seeing = collect_manifest(pids=[__import__("os").getpid()])
    assert blind["resident_extensions"] == []
    assert blind["residency_readable"] is False
    assert seeing["residency_readable"] is True
    assert fingerprint(blind) != fingerprint(
        dict(blind, residency_readable=True)), (
        "an unverified scan must not fingerprint as a verified empty one")


def test_extra_annotations_cannot_overwrite_observed_manifest_fields():
    with pytest.raises(ValueError, match="non-reserved"):
        collect_manifest(
            pids=[__import__("os").getpid()],
            launch_argv=["vllm", "serve", "/m"],
            extra={"image": "forged-image"},
        )


def test_namespaced_measurement_annotation_remains_supported():
    manifest = collect_manifest(
        pids=[__import__("os").getpid()],
        launch_argv=["vllm", "serve", "/m"],
        extra={"measurement_tool": "test-tool"},
    )
    assert manifest["measurement_tool"] == "test-tool"


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------
def test_same_performance_stack_reports_a_delta():
    a = _result(0.0200, manifest=BASE_MANIFEST)
    b = _result(0.0100, manifest=BASE_MANIFEST)
    code, lines = compare(a, b, metric="kl_confident_mean")
    text = "\n".join(lines)
    assert code == 0
    assert "performance_stack_fingerprint" in text
    assert "(matched, validated)" in text
    assert "-50.00%" in text


def test_distinct_valid_serve_sessions_compare_on_the_performance_stack():
    arm_a = dict(
        BASE_MANIFEST,
        artifact_binding={"model_sha": "a" * 64},
        serve_session_id="1" * 64,
    )
    arm_b = dict(
        BASE_MANIFEST,
        artifact_binding={"model_sha": "b" * 64},
        serve_session_id="2" * 64,
    )
    a = _result(0.0200, manifest=arm_a, model="/arm-a")
    b = _result(0.0100, manifest=arm_b, model="/arm-b")
    assert (
        a["serve_manifest"]["performance_stack_fingerprint"]
        == b["serve_manifest"]["performance_stack_fingerprint"]
    )
    assert a["serve_fingerprint"] != b["serve_fingerprint"]

    code, lines = compare(a, b, metric="kl_confident_mean")
    text = "\n".join(lines)
    assert code == 0
    assert "(matched, validated)" in text
    assert text.count("validated per-run attestation") == 2
    assert "-50.00%" in text


def test_cross_fingerprint_refuses_without_the_flag():
    a = _result(0.01134, manifest=BASE_MANIFEST)
    b = _result(0.01328,
                manifest=dict(BASE_MANIFEST, resident_extensions=[]))
    code, lines = compare(a, b, metric="kl_confident_mean")
    text = "\n".join(lines)
    assert code == 3
    assert "REFUSED" in text
    assert "resident_extensions" in text
    assert "delta (" not in text, "a refusal must not quote a delta at all"
    assert "-14" not in text and "+17" not in text


def test_cross_fingerprint_downgrades_to_a_range_inside_the_band():
    """The dated 27B case: 0.01134 vs 0.01328 on residency alone."""
    a = _result(0.01134, manifest=BASE_MANIFEST)
    b = _result(0.01328,
                manifest=dict(BASE_MANIFEST, resident_extensions=[]))
    code, lines = compare(a, b, metric="kl_confident_mean",
                          allow_cross_fingerprint=True)
    text = "\n".join(lines)
    assert code == 0
    assert "CROSS-STACK RANGE (not a delta)" in text
    assert f"±{CROSS_STACK_BAND * 100:.0f}%" in text
    assert "NOT EVIDENCE" in text


def test_cross_fingerprint_outside_the_band_is_still_a_range():
    a = _result(0.0400, manifest=BASE_MANIFEST)
    b = _result(0.0100, manifest=dict(BASE_MANIFEST, image="other:latest"))
    code, lines = compare(a, b, metric="kl_confident_mean",
                          allow_cross_fingerprint=True)
    text = "\n".join(lines)
    assert code == 0
    assert "outside the ±20% band" in text
    assert "NOT EVIDENCE" not in text
    assert "image" in text


def test_legacy_json_without_a_fingerprint_compares_with_a_warning():
    a = _result(0.0200)
    b = _result(0.0100)
    code, lines = compare(a, b, metric="kl_confident_mean")
    text = "\n".join(lines)
    assert code == 0
    assert "legacy JSONs" in text
    assert "-50.00%" in text


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("performance_stack_fingerprint", "performance_stack_fingerprint is stale"),
        ("serve_fingerprint", "serve_fingerprint is stale"),
    ],
)
def test_stale_manifest_attestation_is_refused(field, message):
    a = _result(0.0200, manifest=BASE_MANIFEST)
    b = _result(0.0100, manifest=BASE_MANIFEST)
    a["serve_manifest"][field] = "0" * 64
    if field == "serve_fingerprint":
        a["serve_fingerprint"] = "0" * 64

    code, lines = compare(a, b, metric="kl_confident_mean")
    text = "\n".join(lines)
    assert code == 3
    assert "INVALID ATTESTATION" in text
    assert message in text
    assert "delta (" not in text


def test_attested_manifest_without_performance_fingerprint_is_refused():
    a = _result(0.0200, manifest=BASE_MANIFEST)
    b = _result(0.0100, manifest=BASE_MANIFEST)
    del a["serve_manifest"]["performance_stack_fingerprint"]

    code, lines = compare(a, b, metric="kl_confident_mean")
    text = "\n".join(lines)
    assert code == 3
    assert "no valid performance_stack_fingerprint" in text
    assert "delta (" not in text


def test_mixed_legacy_and_attested_results_are_refused():
    legacy = _result(0.0200)
    current = _result(0.0100, manifest=BASE_MANIFEST)
    code, lines = compare(legacy, current, metric="kl_confident_mean")
    text = "\n".join(lines)
    assert code == 3
    assert "one arm is an unattested legacy metric" in text
    assert "delta (" not in text


def test_spec_decode_taint_is_called_out():
    a = _result(0.02, manifest=BASE_MANIFEST)
    b = dict(_result(0.01, manifest=BASE_MANIFEST), spec_decode_detected=True)
    text = "\n".join(compare(a, b, metric="kl_confident_mean")[1])
    assert "DRAFT model's" in text


def test_differing_git_commit_is_noted_but_not_refused():
    a = dict(_result(0.02, manifest=BASE_MANIFEST), git_commit="a" * 40)
    b = dict(_result(0.01, manifest=BASE_MANIFEST), git_commit="b" * 40)
    code, lines = compare(a, b, metric="kl_confident_mean")
    assert code == 0
    assert any("different git_commit" in line for line in lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_metric_autoselect_and_exit_codes(tmp_path, capsys):
    same = tmp_path / "a.json"
    other = tmp_path / "b.json"
    same.write_text(json.dumps(_result(4.10, manifest=BASE_MANIFEST,
                                       metric="ppl")))
    other.write_text(json.dumps(
        _result(4.05, manifest=dict(BASE_MANIFEST, image="other:latest"),
                metric="ppl")))

    assert kl_ab_cli([str(same), str(other)]) == 3
    assert "REFUSED" in capsys.readouterr().out

    assert kl_ab_cli([str(same), str(other), "--allow-cross-fingerprint"]) == 0
    out = capsys.readouterr().out
    assert "metric: ppl" in out
    assert "CROSS-STACK RANGE" in out
