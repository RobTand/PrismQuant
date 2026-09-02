"""Strict producer-side interpretation of the immutable Gridbook pin."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from prismaquant import gridbook_runtime_pin as pinmod
from prismaquant import gridbook_serving_runtime_pin as serving_pinmod
from prismaquant.shipcard import _released_gridbook_runtime_pin


REPO = Path(__file__).resolve().parents[1]
PIN_PATH = (
    REPO / "prismaquant" / "gridbook_runtime" / "gridbook_runtime_pin.json"
)


def _payload(
    version=pinmod.GRIDBOOK_RUNTIME_RELEASE_VERSION,
    *,
    version_is_release=True,
    features=None,
):
    return {
        "schema": pinmod.GRIDBOOK_RUNTIME_PIN_SCHEMA,
        "repository": "https://github.com/RobTand/gridbook.git",
        "commit": "a" * 40,
        "version": version,
        "version_is_release": version_is_release,
        "runtime_contract_schema": pinmod.GRIDBOOK_RUNTIME_CONTRACT_SCHEMA,
        "required_abi_features": dict(
            pinmod.GRIDBOOK_REQUIRED_ABI_FEATURES
            if features is None else features
        ),
    }


def test_strict_reader_matches_the_tracked_pin():
    pinmod.load_gridbook_runtime_pin.cache_clear()
    parsed = pinmod.load_gridbook_runtime_pin()
    tracked = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    assert parsed == pinmod.GridbookRuntimePin(**tracked)


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("schema", "unknown", "unsupported schema"),
        ("repository", "file:///gridbook.git", "github.com/RobTand/gridbook"),
        ("commit", "ABC", "lowercase full 40-hex"),
        ("version", "not-a-version", "invalid package version"),
        ("version_is_release", 1, "JSON boolean"),
    ),
)
def test_strict_parser_rejects_malformed_pin_members(field, value, match):
    payload = _payload()
    payload[field] = value
    with pytest.raises(pinmod.GridbookRuntimePinError, match=match):
        pinmod.parse_gridbook_runtime_pin(payload)


def test_strict_parser_rejects_missing_and_unknown_members():
    missing = _payload()
    missing.pop("commit")
    with pytest.raises(pinmod.GridbookRuntimePinError, match="expected exactly"):
        pinmod.parse_gridbook_runtime_pin(missing)
    extra = {**_payload(), "abi_guess": True}
    with pytest.raises(pinmod.GridbookRuntimePinError, match="expected exactly"):
        pinmod.parse_gridbook_runtime_pin(extra)


def test_capabilities_are_feature_gated_not_version_inferred():
    pin = pinmod.parse_gridbook_runtime_pin(_payload("99.0.0"))
    assert pinmod.supports_routed_moe_per_role_codebook_lut(pin)
    assert pinmod.supports_source_fp8_block128_w8a16(pin)


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("runtime_contract_schema", "gridbook.runtime-contract.v2", "schema"),
        ("required_abi_features", {}, "contain exactly"),
        (
            # Right key SET, wrong value TYPE: the parser must reject a JSON
            # boolean even though ``True == 1`` in Python.  Keyed off the
            # required map so it keeps testing the type check, not the
            # membership check, when the ABI closure grows.
            "required_abi_features",
            {
                **dict(pinmod.GRIDBOOK_REQUIRED_ABI_FEATURES),
                "source_fp8_block128_w8a16": True,
            },
            "integer 1",
        ),
    ),
)
def test_strict_parser_rejects_runtime_contract_drift(field, value, match):
    payload = _payload()
    payload[field] = value
    with pytest.raises(pinmod.GridbookRuntimePinError, match=match):
        pinmod.parse_gridbook_runtime_pin(payload)


def test_tracked_release_commit_is_resolved_and_exact():
    pinmod.load_gridbook_runtime_pin.cache_clear()
    pin = pinmod.load_gridbook_runtime_pin()
    assert pin.commit == pinmod.GRIDBOOK_RUNTIME_RELEASE_COMMIT
    assert pin.version_is_release is True
    pinmod.require_resolved_gridbook_runtime_pin(pin)
    pinmod.require_exact_gridbook_runtime_release(pin)


def test_exact_release_gate_rejects_an_alternate_resolved_commit():
    pin = pinmod.parse_gridbook_runtime_pin(_payload())
    assert pin.commit_is_resolved
    with pytest.raises(pinmod.GridbookRuntimePinError, match="exact released"):
        pinmod.require_exact_gridbook_runtime_release(pin)


def test_tracked_serving_release_commit_is_resolved_and_exact():
    """The serving pin resolves to the reviewed v0.8.6 release.

    This replaces an earlier test that asserted the SHIPPED pin was still
    pending. That assertion was correct only while 0.8.6 was untagged; keeping
    it would have meant the suite went red exactly when the release landed,
    i.e. it would have been a test of the calendar rather than of the gate.
    The gate itself is not weakened: the pending path is still exercised below
    on a synthetic pin, and the alternate-commit path already has its own test.
    """
    serving_pinmod.load_gridbook_serving_runtime_pin.cache_clear()
    pin = serving_pinmod.load_gridbook_serving_runtime_pin()
    assert pin.commit == serving_pinmod.GRIDBOOK_SERVING_RUNTIME_RELEASE_COMMIT
    assert (pin.wheel_sha256
            == serving_pinmod.GRIDBOOK_SERVING_RUNTIME_RELEASE_WHEEL_SHA256)
    assert pin.version_is_release is True
    serving_pinmod.require_exact_gridbook_serving_runtime_release(pin)
    # And the shipcard gate accepts it, which is the thing that was blocked.
    _released_gridbook_runtime_pin()


def test_shipcard_gate_still_rejects_a_pending_serving_pin(monkeypatch):
    """The pending sentinel must remain rejected after the release lands."""
    import prismaquant.shipcard as shipcard_module

    pending = replace(
        serving_pinmod.load_gridbook_serving_runtime_pin(),
        commit=serving_pinmod.GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING,
        wheel_sha256=(
            serving_pinmod.GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING),
        version_is_release=False,
    )
    monkeypatch.setattr(
        shipcard_module,
        "load_gridbook_serving_runtime_pin",
        lambda: pending,
    )

    with pytest.raises(
        serving_pinmod.GridbookServingRuntimePinError,
        match="still pending",
    ):
        shipcard_module._released_gridbook_runtime_pin()


def test_shipcard_gate_rejects_an_alternate_resolved_serving_commit(monkeypatch):
    import prismaquant.shipcard as shipcard_module

    alternate = replace(
        serving_pinmod.load_gridbook_serving_runtime_pin(),
        commit="a" * 40,
        wheel_sha256="b" * 64,
        version_is_release=True,
    )
    monkeypatch.setattr(
        shipcard_module,
        "load_gridbook_serving_runtime_pin",
        lambda: alternate,
    )

    with pytest.raises(
        serving_pinmod.GridbookServingRuntimePinError,
        match="exact reviewed",
    ):
        shipcard_module._released_gridbook_runtime_pin()
