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

Two facts, and the second is what keeps the first honest: the pattern matches
the extension the plugin loads, and the names it matches are the PIN's, not
this repository's opinion about another runtime.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from prismaquant.tessera_serving_runtime_pin import (
    load_tessera_serving_runtime_pin,
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


@pytest.mark.parametrize("mapped", [
    "/root/.cache/torch_extensions/py312_cu130/tessera_nvfp4_9f2c/"
    "tessera_nvfp4_9f2c.so",
    "/dqruns/ext/tessera_nvfp4/abcdef01/tessera_nvfp4_abcdef01.so",
])
def test_a_resident_tessera_decoder_is_matched(mapped):
    """The `.so` the plugin JIT-builds carries a build-identity suffix, so the
    pattern has to match the PREFIX the pin declares."""
    assert _serve_fingerprint().EXTENSION_PATTERN.search(mapped) is not None


def test_an_unrelated_shared_object_is_still_not_matched():
    """Widening the alternation must not turn the fingerprint into a census of
    every mapped library."""
    module = _serve_fingerprint()
    for mapped in ("/usr/lib/x86_64-linux-gnu/libcudart.so.13",
                   "/usr/lib/python3.12/lib-dynload/_json.so"):
        assert module.EXTENSION_PATTERN.search(mapped) is None


def test_the_matched_names_are_the_pins_and_not_this_files_opinion():
    """Principle 14's shape, as far as this seam can carry it.

    ``serve_fingerprint`` runs inside the serving container from a bootstrapped
    snapshot of five tool files and no package data, so it cannot READ the pin
    at runtime and carries the tuple instead. This test is what makes the tuple
    a copy of the pin rather than a guess about the runtime: a pin edit that is
    not mirrored here fails, and vice versa.

    The remaining gap is Tessera-side and is recorded, not papered over: the
    packaged ``runtime_contract.json`` publishes the plugin's executed
    activation contracts but not the basenames of the CUDA extensions it
    loads, so the pin is the most machine-readable source that exists today.
    """
    module = _serve_fingerprint()
    pin = load_tessera_serving_runtime_pin()
    assert pin.serving_extension_basenames == ("tessera_nvfp4",)
    assert (tuple(module.TESSERA_EXTENSION_BASENAMES)
            == pin.serving_extension_basenames)
    for name in pin.serving_extension_basenames:
        assert module.EXTENSION_PATTERN.search(f"/x/{name}_deadbeef.so")


def test_the_pin_requires_at_least_one_extension_basename():
    """An empty list would restore the silent failure with a field to point at."""
    import json

    from prismaquant.tessera_serving_runtime_pin import (
        TesseraServingRuntimePinError,
        parse_tessera_serving_runtime_pin,
        tessera_serving_runtime_pin_path,
    )

    payload = json.loads(
        tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))
    payload["serving_extension_basenames"] = []
    # Matched on the refusal's own words and not merely on the field name: a
    # tree that does not know the member at all rejects this payload as an
    # unexpected member, and that message names the field too. This test must
    # fail there rather than pass for the wrong reason.
    with pytest.raises(TesseraServingRuntimePinError,
                       match="serving_extension_basenames must be a non-empty "
                             "list"):
        parse_tessera_serving_runtime_pin(payload)
