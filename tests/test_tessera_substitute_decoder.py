"""Issue #142: a substituted-decoder serve is a different hash with no name on it.

Tessera contract v7 publishes, per native extension, what a serve does when
the ``.so`` cannot build (``native_extensions[0].when_unavailable``):
resident mode keeps serving on a NAMED substitute decoder, streamed mode
refuses. ``tools/serve_fingerprint.py`` observes ``/proc/<pid>/maps``, so an
absent ``.so`` is absent from ``resident_extensions`` and the fingerprint
already moves -- but the manifest records only the basenames it *found*, never
which libraries the pinned runtime was *expected* to load, so "the Tessera
decoder was expected and is missing" and "this stack simply has no Tessera in
it" are the same manifest, and the §7.4 refusal can only imply a drift band.

Every expectation below is DERIVED from the installed Tessera contract's own
table (via ``tessera_runtime_contract``), never typed: a test that restates
today's decoder name would pass with a stale transcription, which is the
defect, not the check.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import tools.serve_fingerprint as serve_fingerprint
from tools.kl_ab import compare
from tools.serve_fingerprint import (
    fingerprint,
    performance_stack_fingerprint,
)


def _tool():
    """The tool by path, the way the container bootstrap loads it."""
    path = (Path(__file__).resolve().parents[1]
            / "tools" / "serve_fingerprint.py")
    spec = importlib.util.spec_from_file_location(
        "serve_fingerprint_by_path", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _published_rows():
    """The installed contract's ``native_extensions`` table, as JSON."""
    from prismaquant import tessera_runtime_contract as trc

    return json.loads(trc.contract_path().read_text(encoding="utf-8"))[
        "native_extensions"]


def _substitute_decoders():
    """Every non-null ``when_unavailable`` decoder the runtime names."""
    decoders = set()
    for row in _published_rows():
        for behaviour in row["when_unavailable"].values():
            if behaviour["decoder"] is not None:
                decoders.add(behaviour["decoder"])
    assert decoders, "the contract must name at least one substitute decoder"
    return sorted(decoders)


# --- the pin carries what runs when the library is absent --------------------

def test_the_pin_transcribes_when_unavailable_from_the_contract():
    """One more transcription under the same contract->pin refusal (#133).

    The pin carries prefix/glob/match today; without ``when_unavailable`` the
    manifest cannot say what an absent library *means*, only that it is
    absent.  Both halves are derived from the installed table, so a runtime
    that renames its substitute decoder fails here rather than passing with
    a stale name.
    """
    from prismaquant.tessera_serving_runtime_pin import (
        load_tessera_serving_runtime_pin,
    )

    pin = load_tessera_serving_runtime_pin()
    published = {
        row["module_name_prefix"]: {
            mode: {"status": behaviour["status"],
                   "decoder": behaviour["decoder"]}
            for mode, behaviour in sorted(row["when_unavailable"].items())
        }
        for row in _published_rows()
    }
    assert published, "the contract must publish at least one extension row"
    pinned = {
        row["module_name_prefix"]: dict(row["when_unavailable"])
        for row in pin.native_extension_rows()
    }
    assert pinned == published


def test_a_pin_row_without_when_unavailable_is_refused():
    """A transcription that drops the block is not a transcription of it."""
    import json as _json

    from prismaquant.tessera_serving_runtime_pin import (
        TesseraServingRuntimePinError,
        parse_tessera_serving_runtime_pin,
        tessera_serving_runtime_pin_path,
    )

    payload = _json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    row = dict(payload["serving_native_extensions"][0])
    row.pop("when_unavailable", None)
    payload["serving_native_extensions"] = [row]
    with pytest.raises(TesseraServingRuntimePinError,
                       match="when_unavailable"):
        parse_tessera_serving_runtime_pin(payload)


def test_the_tool_carries_the_pin_rows_including_when_unavailable():
    """The container bootstrap ships five tool files and no package data, so
    the tool carries the rows -- now with the block that says what an absent
    library means -- and any disagreement fails here."""
    from prismaquant.tessera_serving_runtime_pin import (
        load_tessera_serving_runtime_pin,
    )

    module = _tool()
    pin = load_tessera_serving_runtime_pin()
    assert (list(module.TESSERA_NATIVE_EXTENSIONS)
            == pin.native_extension_rows())
    for row in module.TESSERA_NATIVE_EXTENSIONS:
        assert set(row["when_unavailable"]), (
            "a carried row with no when_unavailable block says nothing "
            "about an absent library")


# --- the manifest records expected-vs-found per pinned row -------------------

def _status_manifest(*, resident: bool):
    """A minimal manifest pair whose only stack difference is the decoder."""
    row = _published_rows()[0]
    prefix = row["module_name_prefix"]
    stem = f"{prefix}9f2c"
    basename = f"{stem}.so"
    return {
        "image": "vllm-node:latest",
        "gpu_name": "NVIDIA GB10",
        "driver_version": "595.42",
        "enforce_eager": True,
        "quantization": "compressed-tensors",
        "package_versions": {"vllm": "0.21.0", "torch": "2.11.0"},
        "resident_extensions": ([basename] if resident else []),
        "residency_readable": True,
        "launch_flags": ["vllm", "serve", "<path>", "--enforce-eager"],
        "native_extension_status": [
            {
                "module_name_prefix": row["module_name_prefix"],
                "filename_glob": row["filename_glob"],
                "match": row["match"],
                "resident": bool(resident),
                "when_unavailable": {
                    mode: {"status": behaviour["status"],
                           "decoder": behaviour["decoder"]}
                    for mode, behaviour in sorted(
                        row["when_unavailable"].items())
                },
            }
            for row in _published_rows()
        ],
        "created": "2026-07-30T10:00:00",
        "launch_argv": ["vllm", "serve", "/dqruns/a/exported",
                        "--enforce-eager"],
        "model": "/dqruns/a/exported",
        "processes": [{"pid": 1, "cmdline": "vllm serve"}],
    }


def test_native_extension_status_is_derived_from_found_basenames():
    """The status block is a projection of ``resident_extensions`` through the
    carried rows -- not a second observation -- so it cannot disagree with
    the scan.  Derived from the contract's own glob: a basename the load path
    can produce reads resident, an unrelated one does not."""
    row = _published_rows()[0]
    prefix, stem = row["module_name_prefix"], row["module_name_prefix"] + "9f2c"
    status = serve_fingerprint.native_extension_status([f"{stem}.so"])
    assert status[0]["module_name_prefix"] == prefix
    assert status[0]["resident"] is True
    assert status[0]["when_unavailable"] == {
        mode: {"status": behaviour["status"], "decoder": behaviour["decoder"]}
        for mode, behaviour in sorted(row["when_unavailable"].items())
    }
    status = serve_fingerprint.native_extension_status([])
    assert status[0]["resident"] is False
    status = serve_fingerprint.native_extension_status(["libcudart.so.13"])
    assert status[0]["resident"] is False


def test_the_status_block_moves_neither_fingerprint():
    """The deliberate A/B-continuity call: the block is a deterministic
    projection of already-fingerprinted inputs (``resident_extensions`` plus
    the tool-carried rows), so recording it must not move any fingerprint --
    no manifest recorded before the change compares differently after."""
    manifest = _status_manifest(resident=True)
    without = {k: v for k, v in manifest.items()
               if k != "native_extension_status"}
    assert fingerprint(manifest) == fingerprint(without)
    assert (performance_stack_fingerprint(manifest)
            == performance_stack_fingerprint(without))
    # And the residency difference itself still moves both hashes: absence of
    # the .so is absence from resident_extensions, which is fingerprinted.
    other = _status_manifest(resident=False)
    assert fingerprint(manifest) != fingerprint(other)
    assert (performance_stack_fingerprint(manifest)
            != performance_stack_fingerprint(other))


# --- the §7.4 refusal names the substitute -----------------------------------

def _result(value, *, manifest, metric="kl_confident_mean", model="/a"):
    signed = dict(manifest)
    signed["performance_stack_fingerprint"] = (
        performance_stack_fingerprint(signed)
    )
    signed["serve_fingerprint"] = fingerprint(signed)
    return {"model": model, metric: value, "serve_manifest": signed,
            "serve_fingerprint": signed["serve_fingerprint"]}


def test_a_substituted_arm_is_named_not_band_implied():
    """One specific residency mismatch is categorically stronger than the
    drift band: the arm without the decoder measured nothing about the lane
    at any delta.  The refusal must name the substitute the pinned table
    publishes -- derived here from the installed contract, not typed."""
    native = _result(0.01134, manifest=_status_manifest(resident=True),
                     model="/a")
    substituted = _result(0.01328, manifest=_status_manifest(resident=False),
                          model="/b")
    code, lines = compare(native, substituted, metric="kl_confident_mean")
    assert code == 3, "\n".join(lines)
    text = "\n".join(lines)
    for decoder in _substitute_decoders():
        assert decoder in text, (
            f"the refusal names no substitute decoder ({decoder!r}); "
            "it implies a drift band instead:\n" + text)
    assert "measured nothing about the lane" in text
