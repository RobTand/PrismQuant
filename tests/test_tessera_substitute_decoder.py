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

import json
from pathlib import Path

import pytest

import tools.serve_fingerprint as serve_fingerprint
from tools.kl_ab import compare
from tools.serve_fingerprint import (
    fingerprint,
    performance_stack_fingerprint,
)


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
#
# NOTE (rebase onto #137/#138/#139): the pin-vs-contract transcription test
# that lived here was dropped -- the sibling's
# `test_the_pin_transcribes_the_contracts_native_extensions_table` already
# covers it, now including the `when_unavailable` projection, and two tests
# asserting one transcription would rot as one.  Likewise the tool-carries-
# the-pin-rows test below duplicates the sibling's
# `test_the_tools_rows_are_the_pins_and_not_this_files_opinion`, which the
# tool now satisfies by READING the pin JSON rather than carrying a
# constant: both were removed in favour of the sibling's, and what remains
# here is what the sibling does not cover -- the reader refusal, the status
# projection, its fingerprint exclusion, and the §7.4 naming.

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


def test_the_tool_reads_the_block_from_the_pin_file(tmp_path):
    """The tool carries no constant to go stale: its rows ARE the pin read.

    The sibling's `test_the_tools_rows_are_the_pins...` already asserts the
    tool rows equal the pin reader's on the tracked file; this asserts the
    fourth member rides that same read -- a pin without the block refuses in
    the container, and a well-formed one propagates the substitute naming.
    """
    import tools.serve_fingerprint as serve_fingerprint
    from prismaquant.tessera_serving_runtime_pin import (
        tessera_serving_runtime_pin_path,
    )

    tracked = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    stripped = json.loads(json.dumps(tracked))
    for row in stripped["serving_native_extensions"]:
        row.pop("when_unavailable", None)
    rejected = tmp_path / "pin_without_when_unavailable.json"
    rejected.write_text(json.dumps(stripped), encoding="utf-8")
    with pytest.raises(ValueError, match="when_unavailable"):
        serve_fingerprint._load_tessera_native_extensions_from_pin(rejected)
    loaded = serve_fingerprint._load_tessera_native_extensions_from_pin(
        tessera_serving_runtime_pin_path())
    assert [row["when_unavailable"] for row in loaded] == [
        row["when_unavailable"]
        for row in serve_fingerprint.TESSERA_NATIVE_EXTENSIONS]


# --- the manifest records expected-vs-found per pinned row -------------------

def _status_manifest(*, resident: bool):
    """A minimal manifest pair whose only stack difference is the decoder."""
    rows = _published_rows()
    basenames = [f"{row['module_name_prefix']}9f2c.so" for row in rows]
    return {
        "image": "vllm-node:latest",
        "gpu_name": "NVIDIA GB10",
        "driver_version": "595.42",
        "enforce_eager": True,
        "quantization": "compressed-tensors",
        "package_versions": {"vllm": "0.21.0", "torch": "2.11.0"},
        "resident_extensions": (basenames if resident else []),
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
            for row in rows
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
