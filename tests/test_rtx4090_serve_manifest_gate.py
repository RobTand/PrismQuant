"""Failing-branch coverage for `validate_rtx4090_serve_manifest`.

This is the serve-side leg of principle 14: it is what forces the receipt the
producer priced and the session the server actually ran to be the same object.
It had zero test references, so the attestation gate was itself unattested.

The branches exercised here are the ones reachable with plain mappings, before
and at the serve-manifest schema check, so no live server or GPU is involved.
"""

from __future__ import annotations

import pytest

import prismaquant.validate_rtx4090_fp8_cb as rtx


MODEL_SHA = "a" * 64
NONCE = "b" * 32
SERVED_MODEL = f"qwen38-rtx4090-{MODEL_SHA[:32]}-{NONCE}"
IMAGE = f"registry.example/vllm@sha256:{'c' * 64}"

CONTENT_RECEIPT = {"schema": "test.content_receipt.v1", "artifact_bytes": 17}
ARTIFACT_BINDING = {
    "schema": "test.artifact_binding.v1",
    "launch_model": "/model",
    "model_sha": MODEL_SHA,
    "artifact_inventory_sha256": "d" * 64,
    "artifact_bytes": 17,
    "resolved_path": "/model",
}


def _kwargs(**overrides: object) -> dict[str, object]:
    payload = {
        "schema": "unsupported.serve_manifest.schema",
        "artifact_content_receipt": dict(CONTENT_RECEIPT),
    }
    base: dict[str, object] = {
        "payload": payload,
        "arm": "eager",
        "expected_image": IMAGE,
        "expected_served_model": SERVED_MODEL,
        "expected_model_sha": MODEL_SHA,
        "expected_artifact_binding": dict(ARTIFACT_BINDING),
        "expected_artifact_content_receipt": dict(CONTENT_RECEIPT),
        "runtime_pin": {},
        "vllm_runtime_pin": {},
        "runtime_contract": {},
        "expected_runtime_attestation": {},
        "runtime_contract_file_identity": {},
    }
    base.update(overrides)
    return base


def _call(**overrides: object) -> dict[str, object]:
    kwargs = _kwargs(**overrides)
    payload = kwargs.pop("payload")
    return rtx.validate_rtx4090_serve_manifest(payload, **kwargs)


def test_serve_manifest_requires_an_immutable_digest_pinned_image() -> None:
    with pytest.raises(
        rtx.RTX4090FP8CBValidationError, match="immutable name@sha256"
    ):
        _call(expected_image="registry.example/vllm:latest")


def test_serve_manifest_requires_a_served_name_bound_to_the_artifact_digest() -> None:
    with pytest.raises(
        rtx.RTX4090FP8CBValidationError, match="served model name must bind"
    ):
        _call(expected_served_model="qwen38-rtx4090-" + "0" * 32 + "-" + NONCE)


def test_serve_manifest_refuses_a_binding_that_is_not_the_opened_shipcard() -> None:
    drifted = dict(ARTIFACT_BINDING)
    drifted["model_sha"] = "e" * 64
    with pytest.raises(
        rtx.RTX4090FP8CBValidationError,
        match="validated artifact binding differs from the opened shipcard",
    ):
        _call(expected_artifact_binding=drifted)


def test_serve_manifest_refuses_a_receipt_the_one_pass_scan_did_not_produce() -> None:
    """The priced artifact and the served artifact must be one object."""

    payload = {
        "schema": "unsupported.serve_manifest.schema",
        "artifact_content_receipt": {
            "schema": "test.content_receipt.v1",
            "artifact_bytes": 18,
        },
    }
    with pytest.raises(
        rtx.RTX4090FP8CBValidationError,
        match="serve manifest differs from the one-pass artifact content receipt",
    ):
        _call(payload=payload)

    # A missing receipt is refused the same way as a contradicting one.
    with pytest.raises(
        rtx.RTX4090FP8CBValidationError,
        match="serve manifest differs from the one-pass artifact content receipt",
    ):
        _call(payload={"schema": "unsupported.serve_manifest.schema"})


def test_serve_manifest_refuses_an_unsupported_manifest_schema() -> None:
    """Past the receipt binding, an unknown manifest schema still fails closed."""

    with pytest.raises(
        rtx.RTX4090FP8CBValidationError, match="unsupported serve-manifest schema"
    ):
        _call()


def test_serve_manifest_refuses_a_stale_or_missing_fingerprint() -> None:
    from tools.serve_fingerprint import MANIFEST_SCHEMA

    payload = {
        "schema": MANIFEST_SCHEMA,
        "artifact_content_receipt": dict(CONTENT_RECEIPT),
        "serve_fingerprint": "f" * 64,
    }
    with pytest.raises(
        rtx.RTX4090FP8CBValidationError,
        match="serve manifest fingerprint is missing or stale",
    ):
        _call(payload=payload)
