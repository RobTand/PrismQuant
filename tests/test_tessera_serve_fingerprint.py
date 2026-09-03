"""§7.4: a Tessera serve must not fingerprint as "no lane extension resident".

KL is bit-identical *within* one docker session and drifts 4-8x *across* them,
keyed purely on whether a lane's CUDA `.so` was loaded into the serving process
(loading a CUDA extension shifts allocator addresses, activations get different
pointer alignments, and alignment-sensitive heuristics pick different kernels).
``tools/serve_fingerprint.py`` makes that stack an object so an A/B can be
refused when its arms differ.

Until 2026-09-02 the alternation named Gridbook's `.so` and, after the
Gridbook lane retired, named no Tessera one at all -- so a serve running
Tessera's OWN native span-2 decode produced the same fingerprint as a stock
serve, and the one lane whose entire point is a custom decoder was the one
lane §7.4 could not see.  The name arrived on 2026-09-03 as a hand-copy of a
hand-written pin, which is principle 14 read backwards: a claim about the
Tessera runtime, maintained one repository over, that nothing here could refuse
on drift.

**This module is the refusal (issue #133).**  Tessera contract v7 publishes
``native_extensions`` (RobTand/tessera#28), so the chain now reads

    runtime_contract.json  ->  tessera_serving_runtime_pin.json  ->  the tool

with a refusal at each link: ``require_serving_pin_matches_contract`` refuses a
drifted pin wherever the contract is read, ``contract_answer`` puts the table's
three decidable fields in the reviewed dev-pin answer so a Tessera rename must
be re-reviewed rather than silently un-matching a serve, and the tests below
refuse the last link -- the tool is stdlib-only and reads no package data
inside a serving container, so it carries the rows and its BEHAVIOUR, not only
its constant, is compared against the table.
"""
from __future__ import annotations

import fnmatch
import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

from prismaquant import tessera_runtime_contract as trc
from prismaquant.tessera_serving_runtime_pin import (
    TesseraServingRuntimePinError,
    load_tessera_serving_runtime_pin,
    parse_tessera_serving_runtime_pin,
    tessera_serving_runtime_pin_path,
)

#: A real JIT build identity is a sha256 hexdigest (``ext._build_identity``
#: over source, toolchain and compute capability), so the probe paths below use
#: one rather than a short invented suffix.
IDENTITY = "3f" * 32


def _serve_fingerprint():
    """Load the tool by path, the way the container bootstrap does.

    It is stdlib-only by construction and must not import torch or vllm, so it
    is not a package member and cannot be imported by name.
    """
    path = (Path(__file__).resolve().parents[1]
            / "tools" / "serve_fingerprint.py")
    spec = importlib.util.spec_from_file_location("serve_fingerprint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract():
    """The table the INSTALLED Tessera publishes, with no dev-pin gate.

    ``load_tessera_contract`` is the admission path and returns ``None`` unless
    the development override is requested; what these tests need is the runtime's
    own table, which is a plain read.
    """
    return trc.read_tessera_contract(trc.contract_path())


def _contract_payload() -> dict:
    return json.loads(trc.contract_path().read_text(encoding="utf-8"))


def _contract_at(tmp_path: Path, payload: dict, name: str = "runtime_contract.json"):
    location = tmp_path / name
    location.write_text(json.dumps(payload), encoding="utf-8")
    return location


# ---------------------------------------------------------------------------
# contract -> pin
# ---------------------------------------------------------------------------
def test_the_pin_is_derived_from_the_runtimes_own_table():
    """The middle link, and the one issue #133 was filed about.

    ``serving_native_extensions`` is produced by
    ``native_extension_pin_payload`` from the contract, so a hand edit on
    either side is a refusal rather than a silent divergence.
    """
    contract = _contract()
    pin = load_tessera_serving_runtime_pin()
    assert [row.as_payload() for row in pin.serving_native_extensions] == (
        trc.native_extension_pin_payload(contract))
    trc.require_serving_pin_matches_contract(contract)


@pytest.mark.parametrize("field,value", [
    # A rename, and a widened glob that still matches its own prefix (so it
    # passes the reader's coherence check and only the pin link can catch it).
    ("module_name_prefix", "tessera_nvfp5_"),
    ("filename_glob", "tessera_*.so"),
])
def test_a_renamed_extension_refuses_the_pin(tmp_path, field, value):
    """Rename the extension in the plugin and the pin must go red.

    This is the failure the issue named: before the derivation, a rename made a
    Tessera serve fingerprint as *nothing resident* and §7.4 then compared two
    serves it could not tell apart.
    """
    payload = _contract_payload()
    entry = payload["native_extensions"][0]
    entry[field] = value
    if field == "module_name_prefix":
        entry["filename_glob"] = value + "*.so"
    renamed = trc.read_tessera_contract(_contract_at(tmp_path, payload))
    with pytest.raises(TesseraServingRuntimePinError,
                       match="native-extension table is not the one the "
                             "runtime publishes"):
        trc.require_serving_pin_matches_contract(renamed)


def test_a_pin_edited_alone_is_refused(tmp_path):
    """The other direction: the pin cannot drift from the contract either."""
    payload = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    payload["serving_native_extensions"][0]["filename_glob"] = "tessera_*.so"
    edited = parse_tessera_serving_runtime_pin(payload)
    with pytest.raises(TesseraServingRuntimePinError, match="filename_glob"):
        trc.require_serving_pin_matches_contract(_contract(), edited)


def test_a_contract_without_the_table_is_refused(tmp_path):
    """A pre-v7 contract is refused, not read as "this runtime loads nothing"."""
    payload = _contract_payload()
    del payload["native_extensions"]
    with pytest.raises(trc.TesseraContractError,
                       match="publishes no 'native_extensions'"):
        trc.read_tessera_contract(_contract_at(tmp_path, payload))


def test_an_empty_table_is_refused(tmp_path):
    payload = _contract_payload()
    payload["native_extensions"] = []
    with pytest.raises(trc.TesseraContractError,
                       match="must be a non-empty JSON array"):
        trc.read_tessera_contract(_contract_at(tmp_path, payload))


def test_an_unknown_match_rule_is_refused_rather_than_approximated(tmp_path):
    """The point of publishing ``match`` is that a consumer never guesses."""
    payload = _contract_payload()
    payload["native_extensions"][0]["match"] = "substring"
    with pytest.raises(trc.TesseraContractError,
                       match="will not substitute a predicate of its own"):
        trc.read_tessera_contract(_contract_at(tmp_path, payload))


def test_a_glob_that_cannot_match_its_own_prefix_is_refused(tmp_path):
    payload = _contract_payload()
    payload["native_extensions"][0]["filename_glob"] = "tessera_other_*.so"
    with pytest.raises(trc.TesseraContractError,
                       match="describe different files"):
        trc.read_tessera_contract(_contract_at(tmp_path, payload))


def test_a_renamed_extension_moves_the_reviewed_dev_pin_answer(tmp_path):
    """The table is a value a gate reads, so it is in the reviewed ANSWER.

    ``load_tessera_contract`` refuses on ``_answer_drift`` lines, so this is
    what makes a Tessera rename a review event instead of a silent one.
    """
    payload = _contract_payload()
    payload["native_extensions"][0]["module_name_prefix"] = "tessera_nvfp5_"
    payload["native_extensions"][0]["filename_glob"] = "tessera_nvfp5_*.so"
    renamed = trc.read_tessera_contract(_contract_at(tmp_path, payload))
    drift = trc._answer_drift(trc.TESSERA_DEV_PIN_ANSWER,
                              trc.contract_answer(renamed))
    assert any("native_extensions[tessera_nvfp4_]: GONE" in line
               for line in drift), drift
    assert any("native_extensions[tessera_nvfp5_]: NEW" in line
               for line in drift), drift
    assert not trc._answer_drift(trc.TESSERA_DEV_PIN_ANSWER,
                                 trc.contract_answer(_contract()))


# ---------------------------------------------------------------------------
# pin -> the tool
# ---------------------------------------------------------------------------
def test_the_tool_carries_the_rows_the_runtime_publishes():
    """The last link.  The tool cannot read the pin inside a serving container
    (stdlib-only, five bootstrapped tool files, no package data), so this is
    what keeps its constant a projection of the runtime's table rather than an
    opinion about it."""
    module = _serve_fingerprint()
    assert [dict(row) for row in module.TESSERA_NATIVE_EXTENSIONS] == (
        trc.native_extension_pin_payload(_contract()))


def test_the_tool_implements_the_rule_the_table_names():
    contract = _contract()
    module = _serve_fingerprint()
    assert {ext.match for ext in contract.native_extensions} == {
        module.MATCH_BASENAME_FNMATCH}
    assert module.MATCH_BASENAME_FNMATCH == trc.MATCH_BASENAME_FNMATCH


def test_a_resident_tessera_decoder_is_matched():
    """Every library the table publishes, on both build roots in use."""
    module = _serve_fingerprint()
    for ext in _contract().native_extensions:
        stem = ext.module_name_prefix + IDENTITY
        for mapped in (
            f"/root/.cache/torch_extensions/py312_cu130/{stem}/{stem}.so",
            f"/dqruns/ext/{ext.module_name_prefix.rstrip('_')}/{IDENTITY}/{stem}.so",
        ):
            assert module.matches_tracked_extension(mapped), mapped
            assert ext.matches(mapped), mapped


def test_the_tool_agrees_with_the_table_on_every_probe_path():
    """Behaviour, not constants: the tool's answer IS the table's answer.

    The probe corpus deliberately contains the paths on which a basename
    ``fnmatch`` and the retired substring search DISAGREE, so a regression to a
    predicate of the tool's own fails here rather than passing on the happy
    path.
    """
    module = _serve_fingerprint()
    extensions = _contract().native_extensions
    stem = extensions[0].module_name_prefix + IDENTITY
    probes = [
        f"/root/.cache/torch_extensions/py312_cu130/{stem}/{stem}.so",
        # A build DIRECTORY named after the extension, holding somebody else's
        # library: a substring search over the path matches it, the runtime's
        # rule does not.
        f"/root/.cache/torch_extensions/{stem}/libtorch_python.so",
        # The name WITHOUT a build identity.  ``ext._load_locked`` always
        # appends one, so this file is not a library the plugin ever built.
        f"/x/{extensions[0].module_name_prefix.rstrip('_')}.so",
        f"/x/{stem}.so",
        "/usr/lib/x86_64-linux-gnu/libcudart.so.13",
    ]
    for path in probes:
        by_table = any(ext.matches(path) for ext in extensions)
        assert module.matches_tracked_extension(path) == by_table, path


def test_the_rule_is_not_the_substring_search_it_replaced():
    """Keeps the test above from being vacuous.

    If the two predicates agreed on every probe, comparing them would prove
    nothing.  These are the two paths the retired predicate got wrong.
    """
    module = _serve_fingerprint()
    retired = re.compile("|".join(
        re.escape(ext.module_name_prefix.rstrip("_"))
        for ext in _contract().native_extensions))
    stem = _contract().native_extensions[0].module_name_prefix + IDENTITY
    for path in (f"/root/.cache/torch_extensions/{stem}/libtorch_python.so",
                 f"/x/{_contract().native_extensions[0].module_name_prefix.rstrip('_')}.so"):
        assert retired.search(path), path
        assert not module.matches_tracked_extension(path), path


def test_an_unrelated_shared_object_is_still_not_matched():
    """Widening the predicate must not turn the fingerprint into a census of
    every mapped library."""
    module = _serve_fingerprint()
    for mapped in ("/usr/lib/x86_64-linux-gnu/libcudart.so.13",
                   "/usr/lib/python3.12/lib-dynload/_json.so"):
        assert not module.matches_tracked_extension(mapped)


def test_the_residency_scan_records_the_decoder(tmp_path, monkeypatch):
    """End to end through the function the manifest actually calls."""
    module = _serve_fingerprint()
    stem = _contract().native_extensions[0].module_name_prefix + IDENTITY
    maps = tmp_path / "1234" / "maps"
    maps.parent.mkdir()
    maps.write_text(
        "7f00-7f01 r-xp 00000000 00:00 1 "
        f"/root/.cache/torch_extensions/py312/{stem}/{stem}.so\n"
        "7f02-7f03 r-xp 00000000 00:00 2 /usr/lib/libcudart.so.13\n",
        encoding="utf-8")
    monkeypatch.setattr(module, "Path",
                        lambda p: Path(str(p).replace("/proc", str(tmp_path))))
    found, readable, unreadable = module.residency_scan(["1234"])
    assert found == [f"{stem}.so"]
    assert readable == [1234] and unreadable == []


# ---------------------------------------------------------------------------
# the pin's own shape
# ---------------------------------------------------------------------------
def test_the_pin_requires_at_least_one_extension_row():
    """An empty list would restore the silent failure with a field to point at."""
    payload = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    payload["serving_native_extensions"] = []
    # Matched on the refusal's own words and not merely on the field name: a
    # tree that does not know the member at all rejects this payload as an
    # unexpected member, and that message names the field too. This test must
    # fail there rather than pass for the wrong reason.
    with pytest.raises(TesseraServingRuntimePinError,
                       match="serving_native_extensions must be a non-empty "
                             "list"):
        parse_tessera_serving_runtime_pin(payload)


def test_the_pin_refuses_a_match_rule_nothing_here_applies():
    payload = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    payload["serving_native_extensions"][0]["match"] = "substring"
    with pytest.raises(TesseraServingRuntimePinError,
                       match="is not a rule PrismaQuant implements"):
        parse_tessera_serving_runtime_pin(payload)


def test_the_pin_refuses_a_hand_extended_row():
    payload = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    payload["serving_native_extensions"][0]["optional"] = True
    with pytest.raises(TesseraServingRuntimePinError,
                       match="expected exactly"):
        parse_tessera_serving_runtime_pin(payload)
