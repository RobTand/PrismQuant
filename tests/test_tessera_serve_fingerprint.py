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
lane §7.4 could not see.

**Three links, and this file refuses all three.**  What the fingerprint has to
know -- which library, under which filename pattern, matched by which rule --
is a claim about the TESSERA runtime, so principle 14 says it is read from
that runtime's own table or refused.  Since Tessera contract v7 the table
exists (``native_extensions``), and the chain is:

1. contract -> pin: ``tessera_runtime_contract`` refuses a serving pin whose
   ``serving_native_extensions`` is not the pinned contract's table.  Until
   this file's 2026-09-03 revision the pin was a HAND-WRITTEN claim about
   another runtime with nothing here to refuse it -- and it was already wrong
   by one character (``"tessera_nvfp4"`` where the load path's constant is
   ``"tessera_nvfp4_"``).
2. pin -> tool: the tool is stdlib-only and cannot read either file from
   inside a serving container, so it carries the rows and any disagreement
   fails here.
3. the tool's PREDICATE -> the contract's ``match`` rule: the old pattern was
   a bare substring search over the whole mapped path, and the contract says
   the runtime is matched by ``basename_fnmatch`` against
   ``tessera_nvfp4_*.so``.  Those are different predicates and only one of
   them is the runtime's.

The contract is read by ``json`` through ``tessera_runtime_contract`` -- located
through ``importlib.resources.files("tessera.serving")``, which imports only
that package's lazy ``__init__`` (it defines ``register()`` and calls nothing
at module scope, so it registers nothing and needs no GPU), and never through
``tessera.serving.contract``, whose validator imports the plugin's dispatch
tables -- and never by vendoring a copy.
"""
from __future__ import annotations

import fnmatch
import importlib.util
import json
from pathlib import Path

import pytest

from prismaquant import tessera_runtime_contract as trc
from prismaquant.tessera_serving_runtime_pin import (
    MATCH_BASENAME_FNMATCH,
    TesseraServingRuntimePinError,
    load_tessera_serving_runtime_pin,
    parse_tessera_serving_runtime_pin,
    tessera_serving_runtime_pin_path,
)


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


def _packaged_contract() -> dict:
    """The installed plugin's contract payload, as JSON.

    ``trc.contract_path()`` locates the table through
    ``importlib.resources.files("tessera.serving")``: that imports only the
    package's lazy ``__init__``, never ``tessera.serving.contract`` (whose
    validator imports the plugin's dispatch tables) and never vLLM.
    """
    return json.loads(trc.contract_path().read_text(encoding="utf-8"))


def _published_rows() -> list[dict]:
    return list(_packaged_contract()["native_extensions"])


# --- link 1: the pin is the contract's table ---------------------------------

def test_the_pin_transcribes_the_contracts_native_extensions_table():
    """Principle 14, at the link that had no refusal.

    The pin's rows must be the contract's rows, field for field, on the three
    fields a residency predicate is made of.  A prefix rename in the plugin
    used to make a Tessera serve fingerprint as "nothing resident" with
    nothing on this side able to notice.
    """
    pin = load_tessera_serving_runtime_pin()
    published = [
        {key: row[key] for key in
         ("module_name_prefix", "filename_glob", "match")}
        for row in _published_rows()
    ]
    assert pin.native_extension_rows() == published
    assert published, "the contract must publish at least one loadable library"


def test_the_contracts_own_rows_are_what_the_refusal_compares():
    """The refusal function passes on the tracked pin and the real contract."""
    contract = trc._parse(
        _packaged_contract(),
        commit="test", sha="test", path=str(trc.contract_path()))
    trc.require_pin_native_extensions_match_contract(contract)


@pytest.mark.parametrize("mutate,expected", [
    # The rename that started this: the diff is keyed by prefix, so it reads
    # as one library gone and one nobody transcribed -- which is what a rename
    # of a JIT module IS, and both halves are named.
    (lambda rows: rows.__setitem__(
        0, dict(rows[0], module_name_prefix="tessera_renamed_",
                filename_glob="tessera_renamed_*.so")),
     "publishes no such extension"),
    # A glob that still matches a library the load path can produce -- so the
    # parser accepts it -- and is a DIFFERENT predicate than the pinned one.
    (lambda rows: rows.__setitem__(
        0, dict(rows[0], filename_glob="tessera_nvfp4_*")),
     "filename_glob"),
    (lambda rows: rows.append(
        dict(rows[0], module_name_prefix="tessera_second_",
             filename_glob="tessera_second_*.so")),
     "the contract publishes it and the pin omits it"),
])
def test_a_contract_the_pin_does_not_transcribe_is_refused(mutate, expected):
    """Both directions: a moved field, and a library the pin never heard of.

    The second is the failure mode Tessera's own import-walk test guards from
    the other side: a table that goes quietly short reports a serve that maps
    a decoder as a serve that maps nothing.
    """
    payload = _packaged_contract()
    rows = list(payload["native_extensions"])
    mutate(rows)
    payload["native_extensions"] = rows
    contract = trc._parse(payload, commit="t", sha="t", path="<mutated>")
    with pytest.raises(TesseraServingRuntimePinError) as exc:
        trc.require_pin_native_extensions_match_contract(contract)
    assert expected in str(exc.value)


def test_a_contract_publishing_no_table_is_refused_not_read_as_empty():
    """"Does not say what it loads" is not "loads nothing"."""
    payload = _packaged_contract()
    del payload["native_extensions"]
    with pytest.raises(trc.TesseraContractError,
                       match="publishes no 'native_extensions'"):
        trc._parse(payload, commit="t", sha="t", path="<no-table>")

    payload = _packaged_contract()
    payload["native_extensions"] = []
    with pytest.raises(trc.TesseraContractError, match="is empty"):
        trc._parse(payload, commit="t", sha="t", path="<empty-table>")


def test_a_match_rule_the_reader_cannot_apply_is_refused():
    """A rule is a value because the predicate is not guessable from the glob."""
    payload = _packaged_contract()
    payload["native_extensions"] = [
        dict(payload["native_extensions"][0], match="whole_path_substring")
    ]
    with pytest.raises(trc.TesseraContractError, match="will not apply"):
        trc._parse(payload, commit="t", sha="t", path="<unknown-rule>")


def test_the_extension_table_is_part_of_the_reviewed_answer():
    """A moved extension table must re-stale the dev pin, with a named field.

    It is a value a GATE reads -- the §7.4 fingerprint's -- so it belongs in
    ``contract_answer`` beside the admission fields, and a Tessera commit that
    renames the library is a review event rather than a silent widening.
    """
    contract = trc._parse(
        _packaged_contract(), commit="t", sha="t",
        path=str(trc.contract_path()))
    answer = trc.contract_answer(contract)
    # DERIVED, not typed. ``TESSERA_DEV_PIN_ANSWER`` is already the literal and
    # ``test_the_reviewed_answer_is_the_installed_one`` already pins it to the
    # installed table; a second literal here would report a CORRECT second
    # extension as a defect, which is the one thing an anti-staleness test must
    # never do. What is pinned is the RULE: every published row appears, keyed
    # and ordered the way ``contract_answer`` canonicalises it, carrying every
    # field a gate reads and no field it does not.
    assert answer["native_extensions"] == [
        {"module_name_prefix": row["module_name_prefix"],
         "filename_glob": row["filename_glob"],
         "match": row["match"],
         "routes": sorted(row["routes"]),
         "when_unavailable": {
             mode: {"status": behaviour["status"],
                    "decoder": behaviour["decoder"]}
             for mode, behaviour in sorted(row["when_unavailable"].items())}}
        for row in sorted(_published_rows(),
                          key=lambda r: r["module_name_prefix"])
    ]
    assert all(row["match"] == MATCH_BASENAME_FNMATCH
               for row in _published_rows())
    # ``source``/``loaded_by`` name files in the runtime's own tree and move
    # nothing on this side, so they are identity and stay out of the answer --
    # and travel into provenance instead.
    flat = repr(answer)
    for row in _published_rows():
        assert row["source"] not in flat
        assert row["loaded_by"] not in flat
    assert [row["source"] for row in contract.identity()["native_extensions"]] \
        == [row["source"] for row in _published_rows()]

    moved = _packaged_contract()
    moved["native_extensions"] = [
        dict(moved["native_extensions"][0], filename_glob="tessera_nvfp4_*")
    ]
    drift = trc._answer_drift(
        trc.TESSERA_DEV_PIN_ANSWER,
        trc.contract_answer(
            trc._parse(moved, commit="t", sha="t", path="<moved>")))
    assert any("native_extensions[tessera_nvfp4_].filename_glob" in line
               for line in drift), drift


# --- link 2: the tool's table is the pin's -----------------------------------

def test_the_tools_rows_are_the_pins_and_not_this_files_opinion():
    """Principle 14's shape, as far as this seam can carry it.

    ``serve_fingerprint`` runs inside the serving container from a
    bootstrapped snapshot of five tool files and no package data, so it can
    read neither the contract nor the pin at runtime and carries the rows
    instead. This test is what makes them a transcription rather than a guess
    about the runtime: a pin edit that is not mirrored here fails, and vice
    versa.
    """
    module = _serve_fingerprint()
    pin = load_tessera_serving_runtime_pin()
    assert (list(module.TESSERA_NATIVE_EXTENSIONS)
            == pin.native_extension_rows())
    assert module.MATCH_BASENAME_FNMATCH == MATCH_BASENAME_FNMATCH


def test_the_pin_requires_at_least_one_extension():
    """An empty list would restore the silent failure with a field to point at."""
    payload = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    payload["serving_native_extensions"] = []
    # Matched on the refusal's own words and not merely on the field name: a
    # tree that does not know the member at all rejects this payload as an
    # unexpected member, and that message names the field too. This test must
    # fail there rather than pass for the wrong reason.
    with pytest.raises(TesseraServingRuntimePinError,
                       match="serving_native_extensions must be a non-empty"):
        parse_tessera_serving_runtime_pin(payload)


@pytest.mark.parametrize("row,expected", [
    ({"module_name_prefix": "tessera_nvfp4_",
      "filename_glob": "tessera_nvfp4_*.so",
      "match": "whole_path_substring"}, "will not guess"),
    ({"module_name_prefix": "tessera_nvfp4_",
      "filename_glob": "tessera_nvfp4.so",
      "match": MATCH_BASENAME_FNMATCH}, "matches no library name"),
    ({"module_name_prefix": "tessera_nvfp4_",
      "filename_glob": "tessera_nvfp4_*.so"}, "expected exactly"),
    ({"module_name_prefix": "Tessera_NVFP4_",
      "filename_glob": "Tessera_NVFP4_*.so",
      "match": MATCH_BASENAME_FNMATCH}, "must be a lowercase module-name"),
])
def test_the_pin_refuses_a_row_no_fingerprint_could_use(row, expected):
    payload = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    payload["serving_native_extensions"] = [row]
    with pytest.raises(TesseraServingRuntimePinError, match=expected):
        parse_tessera_serving_runtime_pin(payload)


# --- link 3: the predicate is the rule the contract names --------------------

def test_the_tool_applies_the_rule_the_contract_names():
    """Not "a defensible predicate": the published one, on the published glob.

    Cross-checked against ``fnmatch`` on the basename directly, over paths
    derived from the contract's own glob, so a tool that quietly went back to
    a substring search disagrees here.
    """
    module = _serve_fingerprint()
    for row in _published_rows():
        assert row["match"] == MATCH_BASENAME_FNMATCH
        glob = row["filename_glob"]
        prefix = row["module_name_prefix"]
        cases = [
            # what torch actually writes: prefix + build identity + LIB_EXT
            f"/root/.cache/torch_extensions/py312_cu130/{prefix}9f2c/"
            f"{prefix}9f2c.so",
            f"/dqruns/ext/{prefix}abcdef01/{prefix}abcdef01.so",
            f"/x/{prefix}deadbeef.so",
            # the prefix in a DIRECTORY and not in the basename: the substring
            # predicate this replaced answered yes, the published rule does
            # not, and only one of them is the runtime's.
            f"/root/.cache/torch_extensions/{prefix}9f2c/unrelated.so",
            f"/opt/{prefix.rstrip('_')}/libtorch_cpu.so",
            # a stem reading would match this and nothing a serve maps
            f"/x/{prefix.rstrip('_')}.so",
        ]
        for path in cases:
            assert module.matches_tracked_extension(path) is bool(
                fnmatch.fnmatch(Path(path).name, glob)), path


@pytest.mark.parametrize("mapped", [
    "/root/.cache/torch_extensions/py312_cu130/tessera_nvfp4_9f2c/"
    "tessera_nvfp4_9f2c.so",
    "/dqruns/ext/tessera_nvfp4_abcdef01/tessera_nvfp4_abcdef01.so",
])
def test_a_resident_tessera_decoder_is_matched(mapped):
    """The `.so` the plugin JIT-builds carries a build-identity suffix, so the
    rule has to be applied to the glob the contract publishes."""
    assert _serve_fingerprint().matches_tracked_extension(mapped) is True


def test_an_unrelated_shared_object_is_still_not_matched():
    """Widening the match must not turn the fingerprint into a census of every
    mapped library."""
    module = _serve_fingerprint()
    for mapped in ("/usr/lib/x86_64-linux-gnu/libcudart.so.13",
                   "/usr/lib/python3.12/lib-dynload/_json.so",
                   "/root/.cache/torch_extensions/tessera_nvfp4_9f2c/"
                   "build.ninja"):
        assert module.matches_tracked_extension(mapped) is False


def test_the_residency_scan_records_the_decoder(tmp_path, monkeypatch):
    """End to end through the function the manifest actually calls.

    The predicate tests above prove the rule; this proves ``residency_scan``
    applies it to what ``/proc/<pid>/maps`` says, which is the only place the
    fingerprint ever reads it from.  With the call to
    ``matches_tracked_extension`` removed the scan records nothing and this
    fails on ``found``; with the substring predicate restored it records the
    build directory's unrelated library too.
    """
    module = _serve_fingerprint()
    prefix = _published_rows()[0]["module_name_prefix"]
    stem = f"{prefix}9f2c"
    maps = tmp_path / "1234" / "maps"
    maps.parent.mkdir()
    maps.write_text(
        "7f00-7f01 r-xp 00000000 00:00 1 "
        f"/root/.cache/torch_extensions/py312/{stem}/{stem}.so\n"
        "7f02-7f03 r-xp 00000000 00:00 2 "
        f"/root/.cache/torch_extensions/py312/{stem}/libtorch_python.so\n"
        "7f04-7f05 r-xp 00000000 00:00 3 /usr/lib/libcudart.so.13\n",
        encoding="utf-8")
    monkeypatch.setattr(
        module, "Path",
        lambda p: Path(str(p).replace("/proc", str(tmp_path))))
    found, readable, unreadable = module.residency_scan(["1234", "4321"])
    assert found == [f"{stem}.so"]
    assert readable == [1234] and unreadable == [4321]


def test_an_unknown_match_rule_is_refused_rather_than_approximated():
    """The tool implements a named set of rules and refuses a name outside it.

    Falling back to a substring search on an unknown rule is how a fingerprint
    comes to report a predicate the runtime never published.
    """
    module = _serve_fingerprint()
    with pytest.raises(ValueError, match="does not implement"):
        module.extension_predicate(
            {"module_name_prefix": "tessera_nvfp4_",
             "filename_glob": "tessera_nvfp4_*.so",
             "match": "whole_path_substring"})
    for row in module.TESSERA_NATIVE_EXTENSIONS:
        assert module.extension_predicate(row) is not None


def test_the_other_lanes_are_still_matched_by_substring():
    """The published-table arm is additive; it must not narrow the rest."""
    module = _serve_fingerprint()
    for mapped in ("/repo/prismaquant/kernels/nvfp4_fused.so",
                   "/usr/lib/python3/site-packages/flashinfer/_kernels.so"):
        assert module.matches_tracked_extension(mapped) is True
