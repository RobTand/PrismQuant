"""The merge guard for the Tessera ``v0.1.0`` release-pin flip.

This module exists so that the branch that *prepares* the flip cannot be
merged before the thing it pins exists.  Everything the flip needs except the
tag itself is on the branch; the three tag-dependent values in
``prismaquant/tessera_runtime/tessera_serving_runtime_pin.json`` (``commit``,
``version``, ``version_is_release``) and the two module constants that must
equal them (``TESSERA_SERVING_RUNTIME_RELEASE_{COMMIT,VERSION}``) are still the
conspicuous PENDING sentinels, because cutting a Tessera release tag is Rob's
decision and not an agent's.  While they are sentinels the first three tests
here FAIL, loudly and by name.  When the tag is cut and one reviewed commit
resolves both files together, they pass, and this same file is the command
that verifies the flip.

Why a test rather than a note in the PR body: ``tessera_lane_admission``
already refuses a PENDING pin at *runtime*, so nothing ships wrongly either
way.  What a runtime refusal cannot do is stop a green branch from being
merged.  A red test can, and it says which of the five values is still a
sentinel instead of leaving a reviewer to diff two files by eye.

Attestation, not assertion (principle 14).  The version this pin will carry is
not this repository's opinion: it is read from ``versions.tessera`` in the
``runtime_contract.json`` the Tessera plugin packages, through the same
``importlib.resources`` resolver every other producer reader uses.  The commit
cannot be attested from inside PrismaQuant -- Tessera publishes no wheel and
this repository must not shell into a sibling checkout -- so what is pinned
here is its *shape* (a full 40-hex sha, equal in both files), and the reviewed
identity travels in the pin's git diff.

These three tests are SPENT once they pass: **the flip commit deletes them**,
keeping only the fourth.  Blocking a premature merge is their whole purpose,
and one of them would go on to be actively wrong -- the version attestation
reads ``versions.tessera`` from whatever ``tessera.serving`` is importable,
which is a moving Tessera master, so leaving it in the suite would turn the
next ordinary version bump into a red ``main`` demanding a re-pin.  A moving
master is not a review event (the rule ``TESSERA_DEV_PIN_COMMIT`` states) and a
pin bump is a reviewed release, never a test's demand.  What the pin must go on
satisfying afterwards is already covered by
``tests/test_tessera_lane_admission.py`` and by the reader's own
JSON-equals-constants rule.

The fourth test is the half that is already ready, and it passes today: the
pin's ``serving_native_extensions`` table is the installed contract's
``native_extensions`` table, compared value for value by the production
refusal.  It deliberately does NOT go through ``load_tessera_contract``: that
reader additionally compares the whole contract answer against
``TESSERA_DEV_PIN_ANSWER``, which is a separate review axis (development
admission) on its own clock.  Coupling the transcription check to it would
make this guard red for reasons that have nothing to do with the release tag.
"""
from __future__ import annotations

import hashlib
import json
import re
from importlib.resources import as_file

import pytest

from prismaquant import tessera_runtime_contract as contract_module
from prismaquant import tessera_serving_runtime_pin as pin_module
from prismaquant.tessera_serving_runtime_pin import (
    TESSERA_SERVING_RUNTIME_COMMIT_PENDING,
    TESSERA_SERVING_RUNTIME_VERSION_PENDING,
    TesseraServingRuntimePinError,
    load_tessera_serving_runtime_pin,
    require_exact_tessera_runtime_release,
)

_FULL_SHA1 = re.compile(r"\A[0-9a-f]{40}\Z")


def _packaged_contract_bytes() -> bytes:
    """The bytes of the contract the importable ``tessera.serving`` packages."""
    with as_file(contract_module.contract_path()) as path:
        return path.read_bytes()


def _packaged_contract():
    """The packaged contract, parsed, WITHOUT the development-pin answer check.

    ``load_tessera_contract`` is the production reader and it does two things:
    parse, then refuse a contract whose *answer* is not
    ``TESSERA_DEV_PIN_ANSWER``.  This guard wants only the first.  The
    development answer is re-reviewed on its own schedule and its staleness is
    a different finding from "the release tag does not exist yet"; a merge
    guard that conflated them would send a reader to fix the wrong file.
    """
    with as_file(contract_module.contract_path()) as path:
        raw = path.read_bytes()
        return contract_module._load_at(
            str(path), hashlib.sha256(raw).hexdigest(), "unreviewed")


def test_the_tracked_pin_is_a_resolved_release_not_a_pending_sentinel():
    """FAILS until Rob cuts the Tessera tag.  That is this test's whole job.

    ``require_exact_tessera_runtime_release`` is the same function
    ``tessera_render._release_pin_satisfied`` calls, so passing here means the
    third admission conjunct is satisfied -- not that a fixture was arranged.
    """
    pin = load_tessera_serving_runtime_pin()
    try:
        require_exact_tessera_runtime_release(pin)
    except TesseraServingRuntimePinError as exc:
        pytest.fail(
            "The Tessera release pin is not resolved, so this branch must not "
            "merge.\n"
            f"  pin.commit             = {pin.commit!r}\n"
            f"  pin.version            = {pin.version!r}\n"
            f"  pin.version_is_release = {pin.version_is_release!r}\n"
            f"  module RELEASE_COMMIT  = "
            f"{pin_module.TESSERA_SERVING_RUNTIME_RELEASE_COMMIT!r}\n"
            f"  module RELEASE_VERSION = "
            f"{pin_module.TESSERA_SERVING_RUNTIME_RELEASE_VERSION!r}\n"
            f"refusal: {exc}\n"
            "Resolve the JSON pin and the two module constants in ONE commit; "
            "see the PR body's 'The edit, once the tag is cut'.")


def test_the_pinned_version_is_the_version_the_packaged_runtime_publishes():
    """The version is ATTESTED from the runtime's own table, never typed.

    Principle 14.  ``versions.tessera`` is what the plugin publishes about
    itself; a pin naming any other version is a claim about a runtime that
    does not exist.
    """
    published = str(json.loads(_packaged_contract_bytes())["versions"]["tessera"])
    pin = load_tessera_serving_runtime_pin()
    assert pin.version != TESSERA_SERVING_RUNTIME_VERSION_PENDING, (
        f"the pin still carries {TESSERA_SERVING_RUNTIME_VERSION_PENDING!r}; "
        f"the packaged Tessera contract publishes versions.tessera="
        f"{published!r}, which is the value the flip writes")
    assert pin.version == published, (
        f"pin version {pin.version!r} is not the version the packaged Tessera "
        f"runtime publishes ({published!r})")
    assert pin_module.TESSERA_SERVING_RUNTIME_RELEASE_VERSION == published, (
        "the module's reviewed-release constant and the JSON pin must be the "
        "same reviewed value")


def test_the_pinned_commit_is_a_full_sha_in_both_halves():
    """The commit's SHAPE is checkable here; its identity travels in the diff."""
    pin = load_tessera_serving_runtime_pin()
    assert pin.commit != TESSERA_SERVING_RUNTIME_COMMIT_PENDING, (
        f"the pin still carries {TESSERA_SERVING_RUNTIME_COMMIT_PENDING!r}; "
        "the flip writes the commit the Tessera v0.1.0 tag points at "
        "(git -C /home/rob/tessera rev-list -n 1 v0.1.0)")
    assert _FULL_SHA1.match(pin.commit), (
        f"pin commit {pin.commit!r} is not a full 40-hex sha; an abbreviated "
        "commit is not an immutable identity")
    assert pin_module.TESSERA_SERVING_RUNTIME_RELEASE_COMMIT == pin.commit, (
        "the module's reviewed-release constant and the JSON pin must name "
        "the same commit; neither half admits anything alone")


def test_the_pin_transcribes_the_installed_contracts_native_extensions():
    """Ready today, and independent of the tag.

    The production refusal, run against the contract the importable Tessera
    package actually ships, in both directions: a library the contract
    publishes and the pin omits would make a serve fingerprint as "nothing
    resident", and a library the pin invents is a claim about a runtime that
    does not load it.
    """
    contract = _packaged_contract()
    contract_module.require_pin_native_extensions_match_contract(contract)
    published = {ext.module_name_prefix for ext in contract.native_extensions}
    pinned = {row["module_name_prefix"]
              for row in load_tessera_serving_runtime_pin().native_extension_rows()}
    assert pinned == published, (published, pinned)
    assert "tessera_window_gemv" in pinned, (
        "the window GEMV extension is the second row the contract publishes; "
        "a pin without it fingerprints a streamed FP8/BF16 serve as carrying "
        "no native library")
