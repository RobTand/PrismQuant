"""The Tessera serving pin: an exact commit plus the contract's digest.

Until 2026-09-04 this file was ``test_tessera_release_pin_flip.py`` and it
guarded a flip that would happen when Rob cut a Tessera ``v0.1.0`` tag. Rob
retired that requirement -- *"can we just pin prismaquant to latest version of
tessera? then we won't have to keep cutting releases"* -- so the tag is gone
and the tests that waited for it are spent. What replaces them is a guard on
the thing the tag was standing in for: an exact, immutable, reviewed runtime.

"Latest" is read as **an exact commit plus the packaged contract's raw
SHA-256**, never as a floating ref, and of the two only the DIGEST is
enforceable from inside this process -- PrismaQuant cannot verify a sibling
checkout's git history, but it can hash the bytes it is about to read. So the
digest is the gate and the commit is recorded identity, and these tests hold
both to the "one reviewed change" rule the pin has always carried.

Note what this file does NOT assert: that the pinned version matches whatever
Tessera happens to be installed *next*. A test that read the installed
version and demanded the pin follow it would redden ``main`` on every Tessera
bump with a message saying "re-pin to 0.2.0" -- which is a moving ``master``
made into a review event, the exact anti-pattern
``TESSERA_DEV_PIN_COMMIT``'s docstring names. The pin moves when a human moves
it.
"""
import copy
import hashlib
import json
import re

import pytest
from importlib.resources import as_file

from prismaquant import tessera_runtime_contract as contract_module
from prismaquant import tessera_serving_runtime_pin as pin_module
from prismaquant.tessera_serving_runtime_pin import (
    TESSERA_SERVING_RUNTIME_COMMIT_PENDING,
    TESSERA_SERVING_RUNTIME_CONTRACT_SHA256_PENDING,
    TESSERA_SERVING_RUNTIME_PINNED_COMMIT,
    TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256,
    TESSERA_SERVING_RUNTIME_PINNED_VERSION,
    TESSERA_SERVING_RUNTIME_VERSION_PENDING,
    TesseraServingRuntimePinError,
    load_tessera_serving_runtime_pin,
    parse_tessera_serving_runtime_pin,
    require_exact_tessera_runtime_pin,
    require_pinned_tessera_runtime,
    tessera_serving_runtime_pin_path,
)


_FULL_SHA1 = re.compile(r"\A[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


def _pin_payload() -> dict:
    return json.loads(tessera_serving_runtime_pin_path().read_text(encoding="utf-8"))


def _installed_contract_bytes() -> bytes:
    with as_file(contract_module.contract_path()) as path:
        return path.read_bytes()


# ---------------------------------------------------------------------------
# One reviewed change
# ---------------------------------------------------------------------------
def test_the_pin_file_and_the_module_constants_are_one_change():
    pin = load_tessera_serving_runtime_pin()
    assert _FULL_SHA1.match(pin.commit), (
        "the pin must name an exact Tessera commit; a short sha, a tag or a "
        "branch name is the floating ref this pin exists to refuse")
    assert _SHA256.match(pin.contract_sha256)
    assert pin.commit == TESSERA_SERVING_RUNTIME_PINNED_COMMIT
    assert pin.version == TESSERA_SERVING_RUNTIME_PINNED_VERSION
    assert pin.contract_sha256 == TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256


@pytest.mark.parametrize("member,value", [
    ("commit", "0" * 40),
    ("version", "9.9.9"),
    ("contract_sha256", "0" * 64),
])
def test_a_pin_file_that_disagrees_with_the_constants_admits_nothing(member, value):
    payload = _pin_payload()
    payload[member] = value
    pin = parse_tessera_serving_runtime_pin(payload, where="fixture")
    with pytest.raises(TesseraServingRuntimePinError, match="ONE reviewed change"):
        require_exact_tessera_runtime_pin(
            pin, installed_contract_sha256=payload["contract_sha256"])


# ---------------------------------------------------------------------------
# The digest is the enforced binding
# ---------------------------------------------------------------------------
def test_the_pin_binds_the_bytes_the_installed_runtime_actually_packages():
    digest = hashlib.sha256(_installed_contract_bytes()).hexdigest()
    pin = load_tessera_serving_runtime_pin()
    assert digest == pin.contract_sha256, (
        "the installed Tessera is not the pinned one. Install Tessera at the "
        f"pinned commit ({TESSERA_SERVING_RUNTIME_PINNED_COMMIT}) or re-pin, "
        "which is a reviewed change to the pin JSON and the module constants "
        "together -- never a check that reads whatever is installed.\n"
        f"  installed: {digest}\n  pinned:    {pin.contract_sha256}")
    require_pinned_tessera_runtime()


def test_a_tessera_whose_contract_is_not_the_pinned_one_is_refused():
    """The stray-checkout property, kept without a tag.

    A Tessera source tree on ``PYTHONPATH`` used to be refused because the pin
    carried PENDING sentinels. It is refused now because its contract bytes
    are not the pinned bytes -- the same fail-closed answer, from a fact this
    process can actually check.
    """
    pin = load_tessera_serving_runtime_pin()
    with pytest.raises(TesseraServingRuntimePinError,
                       match="the installed Tessera is not the pinned Tessera"):
        require_exact_tessera_runtime_pin(pin, installed_contract_sha256="f" * 64)


def test_the_pinned_version_is_the_version_the_pinned_bytes_publish():
    """The version is a LABEL on the pinned bytes, checked against them.

    Read out of the contract this pin's digest already binds, so it cannot
    become a demand that the pin follow a moving installation: if the bytes
    are the pinned bytes, this is a statement about the pin alone.
    """
    published = str(json.loads(_installed_contract_bytes())["versions"]["tessera"])
    pin = load_tessera_serving_runtime_pin()
    if hashlib.sha256(_installed_contract_bytes()).hexdigest() != pin.contract_sha256:
        pytest.skip("the installed Tessera is not the pinned Tessera")
    assert pin.version == published


# ---------------------------------------------------------------------------
# version_is_release: advisory
# ---------------------------------------------------------------------------
def test_version_is_release_is_recorded_and_not_a_gate():
    """Rob retired the tag, so a gate that still demanded it would restore it."""
    pin = load_tessera_serving_runtime_pin()
    assert pin.version_is_release is False
    require_pinned_tessera_runtime(pin)  # admits anyway

    released = copy.deepcopy(_pin_payload())
    released["version_is_release"] = True
    require_exact_tessera_runtime_pin(
        parse_tessera_serving_runtime_pin(released, where="fixture"),
        installed_contract_sha256=released["contract_sha256"])


def test_a_pending_commit_still_cannot_be_marked_released():
    """The structural rule survives the demotion; it is what keeps the flag true."""
    payload = _pin_payload()
    payload["commit"] = TESSERA_SERVING_RUNTIME_COMMIT_PENDING
    payload["version_is_release"] = True
    with pytest.raises(TesseraServingRuntimePinError,
                       match="cannot be marked released"):
        parse_tessera_serving_runtime_pin(payload, where="fixture")


@pytest.mark.parametrize("member,sentinel", [
    ("commit", TESSERA_SERVING_RUNTIME_COMMIT_PENDING),
    ("version", TESSERA_SERVING_RUNTIME_VERSION_PENDING),
    ("contract_sha256", TESSERA_SERVING_RUNTIME_CONTRACT_SHA256_PENDING),
])
def test_a_pending_pin_admits_nothing(member, sentinel):
    payload = _pin_payload()
    payload[member] = sentinel
    payload["version_is_release"] = False
    pin = parse_tessera_serving_runtime_pin(payload, where="fixture")
    with pytest.raises(TesseraServingRuntimePinError, match="still PENDING"):
        require_exact_tessera_runtime_pin(
            pin, installed_contract_sha256=TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256)


def test_the_pin_schema_moved_with_its_member_set():
    payload = _pin_payload()
    assert payload["schema"] == "prismaquant.tessera_serving_runtime_pin.v2"
    stale = {k: v for k, v in payload.items() if k != "contract_sha256"}
    stale["schema"] = "prismaquant.tessera_serving_runtime_pin.v1"
    with pytest.raises(TesseraServingRuntimePinError):
        parse_tessera_serving_runtime_pin(stale, where="fixture")


# ---------------------------------------------------------------------------
# The two pins name one object
# ---------------------------------------------------------------------------
def test_the_development_pin_and_the_serving_pin_name_the_same_tessera():
    """Two of this repository's own spec files disagreed about one runtime once."""
    pin = load_tessera_serving_runtime_pin()
    assert contract_module.TESSERA_DEV_PIN_COMMIT == pin.commit
    assert contract_module.TESSERA_DEV_PIN_CONTRACT_SHA256 == pin.contract_sha256


def test_the_pin_transcribes_the_installed_contracts_native_extensions():
    """The §7.4 chain: contract -> pin -> fingerprint, refusing at each link."""
    with as_file(contract_module.contract_path()) as path:
        raw = path.read_bytes()
        contract = contract_module._load_at(
            str(path), hashlib.sha256(raw).hexdigest(), "unreviewed")
    contract_module.require_pin_native_extensions_match_contract(contract)
    pinned = {ext.module_name_prefix
              for ext in load_tessera_serving_runtime_pin().serving_native_extensions}
    assert pinned == {ext.module_name_prefix for ext in contract.native_extensions}
