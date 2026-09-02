"""Closed producer-compile evidence for the RTX 4090 FP8-CB burn.

Live units must execute the shared strict full-graph Inductor scorer contract.
Units restored by the existing AURA checkpoint mechanism are covered
transitively: its manifest identity contains the same campaign settings,
producer arm, and streamed source identity, and AURA has already opened,
checksummed, deserialized, and identity-validated each restored unit envelope
before it can contribute to the returned render counters.  This module
captures the manifest; it deliberately does not reopen unit paths and create a
second, weaker checkpoint-validation mechanism.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import stat

from prismaquant.cb_compile_contract import (
    CBCompileContractError,
    validate_cb_compile_execution_proof,
)


CAMPAIGN_CB_COMPILE_PROOF_SCHEMA = (
    "prismaquant.rtx4090_fp8_burn.cb_compile_execution.v1"
)
AURA_CHECKPOINT_BINDING_SCHEMA = (
    "prismaquant.rtx4090_fp8_burn.aura_checkpoint_compile_binding.v1"
)
AURA_CHECKPOINT_MANIFEST_SCHEMA = "prismaquant.aura_checkpoint.manifest.v1"
AURA_CHECKPOINT_IDENTITY_SCHEMA = "prismaquant.aura_checkpoint.identity.v1"

ATOM_NOT_APPLICABLE = {
    "status": "not_applicable",
    "reason": "campaign_cb_serialization_ldlq_false",
    "ldlq": False,
    "ldlq_scope": "none",
    "compiled_calls": 0,
}


class RTX4090CBCompileProofError(ValueError):
    """The producer-compile proof does not close the campaign render set."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _capture_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RTX4090CBCompileProofError(
            f"AURA checkpoint manifest is not a readable regular file: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 64 * 1024 * 1024:
            raise RTX4090CBCompileProofError(
                "AURA checkpoint manifest is not a bounded regular file"
            )
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    identity = lambda item: (
        int(item.st_dev), int(item.st_ino), int(item.st_size),
        int(item.st_mtime_ns), int(item.st_ctime_ns),
    )
    if remaining or identity(before) != identity(after) or len(payload) != before.st_size:
        raise RTX4090CBCompileProofError(
            "AURA checkpoint manifest changed while it was captured"
        )
    return payload


def capture_aura_checkpoint_compile_binding(
    checkpoint_dir: str | Path,
    *,
    expected_qnames: Sequence[str],
    expected_compile_settings: Mapping[str, str],
    expected_extra_fields: Mapping[str, object],
    expected_arm_identity: Mapping[str, object],
    expected_model_identity: Mapping[str, object],
) -> dict[str, object]:
    """Capture and validate the identity that authorizes restored AURA units."""
    root = Path(checkpoint_dir)
    manifest_bytes = _capture_regular_bytes(root / "manifest.json")
    try:
        manifest = json.loads(manifest_bytes)
    except Exception as exc:
        raise RTX4090CBCompileProofError(
            "AURA checkpoint manifest is not valid JSON"
        ) from exc
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema", "identity_sha256", "identity", "units",
    }:
        raise RTX4090CBCompileProofError(
            "AURA checkpoint manifest shape is not closed"
        )
    identity = manifest["identity"]
    if (
        manifest["schema"] != AURA_CHECKPOINT_MANIFEST_SCHEMA
        or not isinstance(identity, Mapping)
        or identity.get("schema") != AURA_CHECKPOINT_IDENTITY_SCHEMA
        or manifest["identity_sha256"] != _canonical_sha256(identity)
    ):
        raise RTX4090CBCompileProofError(
            "AURA checkpoint manifest identity is invalid"
        )
    extra = identity.get("extra")
    if not isinstance(extra, Mapping):
        raise RTX4090CBCompileProofError(
            "AURA checkpoint identity has no closed extra binding"
        )
    for name, expected in expected_extra_fields.items():
        if extra.get(name) != expected:
            raise RTX4090CBCompileProofError(
                f"AURA checkpoint identity differs at extra.{name}"
            )
    if extra.get("compile_settings") != dict(expected_compile_settings):
        raise RTX4090CBCompileProofError(
            "AURA checkpoint identity compile settings differ"
        )
    renderer = extra.get("production_anchor_renderer")
    if not isinstance(renderer, Mapping) or renderer.get(
        "arm_identity"
    ) != dict(expected_arm_identity):
        raise RTX4090CBCompileProofError(
            "AURA checkpoint identity producer arm differs"
        )
    if extra.get("streamed_model_identity") != dict(expected_model_identity):
        raise RTX4090CBCompileProofError(
            "AURA checkpoint identity streamed source differs"
        )
    qnames = tuple(str(name) for name in expected_qnames)
    if not qnames or len(qnames) != len(set(qnames)):
        raise RTX4090CBCompileProofError(
            "expected AURA checkpoint qname cover is invalid"
        )
    raw_units = manifest["units"]
    if not isinstance(raw_units, Sequence) or isinstance(raw_units, (str, bytes)):
        raise RTX4090CBCompileProofError("AURA checkpoint unit manifest is malformed")
    observed: list[str] = []
    for row in raw_units:
        if not isinstance(row, Mapping) or set(row) != {"qname", "file"}:
            raise RTX4090CBCompileProofError(
                "AURA checkpoint unit row shape differs"
            )
        qname = row["qname"]
        relative = row["file"]
        if not isinstance(qname, str) or not isinstance(relative, str):
            raise RTX4090CBCompileProofError(
                "AURA checkpoint unit row types differ"
            )
        expected_file = (
            "units/" + hashlib.sha256(qname.encode("utf-8")).hexdigest() + ".pkl"
        )
        if relative != expected_file:
            raise RTX4090CBCompileProofError(
                f"AURA checkpoint unit filename differs for {qname}"
            )
        observed.append(qname)
    if len(observed) != len(set(observed)) or set(observed) != set(qnames):
        raise RTX4090CBCompileProofError(
            "AURA checkpoint manifest is not the exact stripe qname cover"
        )
    return {
        "schema": AURA_CHECKPOINT_BINDING_SCHEMA,
        "manifest_schema": AURA_CHECKPOINT_MANIFEST_SCHEMA,
        "identity_schema": AURA_CHECKPOINT_IDENTITY_SCHEMA,
        "identity_sha256": str(manifest["identity_sha256"]),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "unit_count": len(qnames),
        "unit_qnames_sha256": _canonical_sha256(sorted(qnames)),
        "compile_settings_sha256": _canonical_sha256(
            dict(expected_compile_settings)
        ),
        "arm_identity_sha256": _canonical_sha256(dict(expected_arm_identity)),
        "streamed_model_identity_sha256": _canonical_sha256(
            dict(expected_model_identity)
        ),
    }


def _validate_checkpoint_binding(
    binding: object,
    *,
    expected_units: int,
    expected_qnames_sha256: str,
    compile_settings: Mapping[str, str],
) -> dict[str, object]:
    keys = {
        "schema", "manifest_schema", "identity_schema", "identity_sha256",
        "manifest_sha256", "unit_count", "unit_qnames_sha256",
        "compile_settings_sha256", "arm_identity_sha256",
        "streamed_model_identity_sha256",
    }
    if not isinstance(binding, Mapping) or set(binding) != keys:
        raise RTX4090CBCompileProofError(
            "AURA checkpoint compile binding shape is not closed"
        )
    normalized = dict(binding)
    if (
        normalized["schema"] != AURA_CHECKPOINT_BINDING_SCHEMA
        or normalized["manifest_schema"] != AURA_CHECKPOINT_MANIFEST_SCHEMA
        or normalized["identity_schema"] != AURA_CHECKPOINT_IDENTITY_SCHEMA
        or type(normalized["unit_count"]) is not int
        or normalized["unit_count"] != expected_units
        or normalized["unit_qnames_sha256"] != expected_qnames_sha256
        or normalized["compile_settings_sha256"]
        != _canonical_sha256(dict(compile_settings))
    ):
        raise RTX4090CBCompileProofError(
            "AURA checkpoint compile binding differs from the campaign"
        )
    for name in (
        "identity_sha256", "manifest_sha256", "unit_qnames_sha256",
        "compile_settings_sha256", "arm_identity_sha256",
        "streamed_model_identity_sha256",
    ):
        value = normalized[name]
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise RTX4090CBCompileProofError(
                f"AURA checkpoint compile binding {name} is invalid"
            )
    return normalized


def build_campaign_cb_compile_proof(
    compiler_proof: Mapping[str, object],
    *,
    compile_settings: Mapping[str, str],
    expected_qnames: Sequence[str],
    rendered_cells: int,
    restored_cells: int,
    formats_per_unit: int,
    cb_formats_per_unit: int,
    checkpoint_binding: Mapping[str, object],
) -> dict[str, object]:
    """Build and immediately replay one campaign compile-coverage proof."""
    expected_units = len(tuple(expected_qnames))
    expected_qnames_sha256 = _canonical_sha256(
        sorted(str(name) for name in expected_qnames)
    )
    live_units = rendered_cells // formats_per_unit if formats_per_unit else -1
    restored_units = restored_cells // formats_per_unit if formats_per_unit else -1
    if (
        expected_units < 1
        or formats_per_unit < 1
        or cb_formats_per_unit < 1
        or cb_formats_per_unit >= formats_per_unit
        or rendered_cells < 0
        or restored_cells < 0
        or rendered_cells % formats_per_unit
        or restored_cells % formats_per_unit
        or live_units + restored_units != expected_units
    ):
        raise RTX4090CBCompileProofError(
            "campaign compile coverage does not partition complete render units"
        )
    status = (
        "live_strict_fullgraph"
        if live_units and not restored_units
        else "restored_strict_checkpoint"
        if restored_units and not live_units
        else "mixed_live_and_restored"
    )
    coverage = {
        "status": status,
        "expected_units": expected_units,
        "formats_per_unit": formats_per_unit,
        "cb_formats_per_unit": cb_formats_per_unit,
        "expected_rendered_cells": expected_units * formats_per_unit,
        "live_rendered_cells": rendered_cells,
        "restored_rendered_cells": restored_cells,
        "live_units": live_units,
        "restored_units": restored_units,
        "live_cb_cells": live_units * cb_formats_per_unit,
        "restored_cb_cells": restored_units * cb_formats_per_unit,
    }
    body: dict[str, object] = {
        "schema": CAMPAIGN_CB_COMPILE_PROOF_SCHEMA,
        "compile_settings": dict(compile_settings),
        "compiler_proof": dict(compiler_proof),
        "coverage": coverage,
        "checkpoint_binding": dict(checkpoint_binding),
        "atom_route": dict(ATOM_NOT_APPLICABLE),
    }
    proof = {**body, "proof_sha256": _canonical_sha256(body)}
    return validate_campaign_cb_compile_proof(
        proof,
        expected_compile_settings=compile_settings,
        expected_qnames=expected_qnames,
        formats_per_unit=formats_per_unit,
        cb_formats_per_unit=cb_formats_per_unit,
    )


def validate_campaign_cb_compile_proof(
    proof: Mapping[str, object],
    *,
    expected_compile_settings: Mapping[str, str],
    expected_qnames: Sequence[str],
    formats_per_unit: int,
    cb_formats_per_unit: int,
) -> dict[str, object]:
    """Replay live/fullgraph and strict-checkpoint coverage from a receipt."""
    keys = {
        "schema", "compile_settings", "compiler_proof", "coverage",
        "checkpoint_binding", "atom_route", "proof_sha256",
    }
    if not isinstance(proof, Mapping) or set(proof) != keys:
        raise RTX4090CBCompileProofError(
            "campaign CB compile execution proof shape is not closed"
        )
    body = dict(proof)
    digest = body.pop("proof_sha256")
    if digest != _canonical_sha256(body):
        raise RTX4090CBCompileProofError(
            "campaign CB compile execution proof checksum differs"
        )
    if (
        body["schema"] != CAMPAIGN_CB_COMPILE_PROOF_SCHEMA
        or body["compile_settings"] != dict(expected_compile_settings)
        or body["atom_route"] != ATOM_NOT_APPLICABLE
    ):
        raise RTX4090CBCompileProofError(
            "campaign CB compile settings or inactive atom route differ"
        )
    qnames = tuple(str(name) for name in expected_qnames)
    if not qnames or len(qnames) != len(set(qnames)):
        raise RTX4090CBCompileProofError(
            "campaign CB compile expected qname cover is invalid"
        )
    coverage = body["coverage"]
    coverage_keys = {
        "status", "expected_units", "formats_per_unit",
        "cb_formats_per_unit", "expected_rendered_cells",
        "live_rendered_cells", "restored_rendered_cells", "live_units",
        "restored_units", "live_cb_cells", "restored_cb_cells",
    }
    if not isinstance(coverage, Mapping) or set(coverage) != coverage_keys:
        raise RTX4090CBCompileProofError(
            "campaign CB compile coverage shape is not closed"
        )
    integer_names = coverage_keys - {"status"}
    if any(type(coverage[name]) is not int or coverage[name] < 0 for name in integer_names):
        raise RTX4090CBCompileProofError(
            "campaign CB compile coverage counters are invalid"
        )
    live_units = int(coverage["live_units"])
    restored_units = int(coverage["restored_units"])
    expected_units = len(qnames)
    expected_status = (
        "live_strict_fullgraph"
        if live_units and not restored_units
        else "restored_strict_checkpoint"
        if restored_units and not live_units
        else "mixed_live_and_restored"
    )
    if (
        coverage["status"] != expected_status
        or coverage["expected_units"] != expected_units
        or coverage["formats_per_unit"] != formats_per_unit
        or coverage["cb_formats_per_unit"] != cb_formats_per_unit
        or coverage["expected_rendered_cells"] != expected_units * formats_per_unit
        or coverage["live_rendered_cells"] != live_units * formats_per_unit
        or coverage["restored_rendered_cells"] != restored_units * formats_per_unit
        or live_units + restored_units != expected_units
        or coverage["live_cb_cells"] != live_units * cb_formats_per_unit
        or coverage["restored_cb_cells"] != restored_units * cb_formats_per_unit
    ):
        raise RTX4090CBCompileProofError(
            "campaign CB compile coverage differs from the stripe plan"
        )
    try:
        compiler = validate_cb_compile_execution_proof(
            body["compiler_proof"],  # type: ignore[arg-type]
            require_live_calls=bool(live_units),
            require_cuda_calls=bool(live_units),
            allowed_helper_prefixes=("encode.",),
        )
    except CBCompileContractError as exc:
        raise RTX4090CBCompileProofError(str(exc)) from exc
    if not live_units and compiler["totals"]["attempted_calls"] != 0:  # type: ignore[index]
        raise RTX4090CBCompileProofError(
            "fully restored campaign proof unexpectedly contains live calls"
        )
    checkpoint = _validate_checkpoint_binding(
        body["checkpoint_binding"],
        expected_units=expected_units,
        expected_qnames_sha256=_canonical_sha256(sorted(qnames)),
        compile_settings=expected_compile_settings,
    )
    return {
        **body,
        "compiler_proof": compiler,
        "coverage": dict(coverage),
        "checkpoint_binding": checkpoint,
        "atom_route": dict(ATOM_NOT_APPLICABLE),
        "proof_sha256": str(digest),
    }
