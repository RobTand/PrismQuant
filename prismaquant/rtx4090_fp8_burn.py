"""Exact two-host transient AURA campaign for the strict RTX 4090 lane.

The ordinary format-menu production-cache path retains rendered weights.  This
driver instead partitions *qnames* by whole decoder layer, transiently renders
three lattice anchors plus fresh native FP8 through the existing streamed AURA
consumer, and merges the two receipt-bearing scalar shards before anchored
interpolation and one byte-budget solve.
Nothing in this module publishes or validates an exported artifact.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
from typing import Any

from prismaquant import format_registry as fr
from prismaquant.anchored_cost import (
    RenderRequest,
    candidates_by_segment,
    plan_anchor_requests,
    price_anchored_candidates,
    run_allocator_once,
)
from prismaquant.cb_anchored_cost import (
    CBUnitDeclaration,
    CodebookAnchoredFormatPlugin,
    LATTICE_BASIS,
    anchors_from_streamed_payload,
    build_cb_allocator_cost_payload,
    build_cb_units,
    fit_all_cb_segments,
    fitted_cb_hull_report,
    merge_streamed_cb_anchor_aura_shards,
    observations_from_streamed_payload,
    run_streamed_cb_anchor_aura,
)
from prismaquant.cb_compile_contract import (
    CB_COMPILE_FAIL_CLOSED_ENV,
    abort_cb_compile_execution_proof,
    begin_cb_compile_execution_proof,
    finish_cb_compile_execution_proof,
)
from prismaquant.cost_stage_checkpoint import (
    atomic_write_bytes,
    canonical_json_sha256,
)
from prismaquant.production_cache_stripes import plan_stripes
from prismaquant.rtx4090_cb_compile_proof import (
    RTX4090CBCompileProofError,
    build_campaign_cb_compile_proof,
    capture_aura_checkpoint_compile_binding,
    validate_campaign_cb_compile_proof,
)
from prismaquant.rtx4090_qwen38_policy import (
    RTX4090_CONTEXT_FIRST_TARGET_BYTES,
    RTX4090_QWEN38_FORMAT_MENU,
    RTX4090_QWEN38_SERVING_PROFILE,
    validate_rtx4090_format_menu,
)


PLAN_SCHEMA = "prismaquant.rtx4090_fp8_burn.plan.v4"
IMATRIX_CONTRACT_SCHEMA = "prismaquant.rtx4090_fp8_burn.probe_imatrix.v1"
EXECUTION_ATTESTATION_SCHEMA = (
    "prismaquant.rtx4090_fp8_burn.execution_attestation.v1"
)
SOURCE_IDENTITY_BINDING_SCHEMA = (
    "prismaquant.rtx4090_fp8_burn.source_model_identity_binding.v2"
)
SHARD_RECEIPT_SCHEMA = "prismaquant.rtx4090_fp8_burn.shard_receipt.v3"
SHARD_RECEIPT_KEY = "rtx4090_fp8_burn_shard_receipt"
CB_COMPILE_PROOF_KEY = "rtx4090_cb_compile_execution"
MERGED_SCHEMA = "prismaquant.rtx4090_fp8_burn.merged_aura.v1"
ALLOCATOR_COST_SCHEMA = "prismaquant.rtx4090_fp8_burn.allocator_cost.v1"
STRIPE_COUNT = 2
STREAMING_CACHE_MAX_SLOTS = 2
STREAMING_PREFETCH_LOOKAHEAD = 1
STREAMING_REQUIRE_PREFETCHED_RESIDENCY = True
ALLOCATOR_PROBE_FD = 198
ARTIFACT_OVERHEAD_RESERVE_BYTES = 268_435_456
CALIBRATION_NSAMPLES = 32
CALIBRATION_SEQLEN = 1024
CALIBRATION_SEED = 42
AURA_N_PROBES = 32
AURA_TOKEN_SCOPE = "all"
BF16_FORMAT = "BF16"
NATIVE_FP8_FORMAT = "FP8_E4M3"
CB_FORMATS = tuple(RTX4090_QWEN38_FORMAT_MENU[:-2])
MEASURED_CB_FORMATS = (CB_FORMATS[0], CB_FORMATS[3], CB_FORMATS[-1])
ANCHOR_CB_FORMAT = CB_FORMATS[3]
MEASURED_FORMATS = (*MEASURED_CB_FORMATS, NATIVE_FP8_FORMAT)
FULL_FORMATS = (*CB_FORMATS, NATIVE_FP8_FORMAT, BF16_FORMAT)
RENDER_FORMATS = (*MEASURED_FORMATS, BF16_FORMAT)
RENDER_LEVERS: Mapping[str, object] = {
    "gptq": True,
    "static_act_order": True,
    "joint_scale_opt": True,
    "weighted_vq": True,
}
CB_PRODUCER_SETTINGS: Mapping[str, object] = {
    "scale_coding": "v1",
    "codebook_source": "lattice",
    "codebook_source_scope": "none",
    "scale_sweep": True,
    "scale_sweep_scope": "fp8",
    "ldlq": False,
    "ldlq_scope": "none",
    "minchain": False,
    "encode_tier": "balanced",
    "activation_scope": "none",
    "encode_compile": True,
    "atom_compile": True,
    "compile_fail_closed": True,
}

_LAYER_RE = re.compile(r"(?:^|[.])layers[.](\d+)(?:[.]|$)")
_SHARD_RECEIPT_BODY_KEYS = frozenset({
    "schema", "campaign_schema", "global_plan_sha256", "stripe_index",
    "stripe_record_sha256", "stripe_qname_file_sha256",
    "stripe_qname_count", "n_probes", "token_scope",
    "fixed_bf16_census_sha256", "producer_snapshot_sha256",
    "common_execution_attestation_sha256", "compile_settings",
    "cb_compile_execution_proof",
    "container_image_digest",
    "arm_identity_sha256", "source_model_identity_binding",
    "live_streamed_model_portable_content_sha256", "renderer_identity_sha256",
    "measured_costs_sha256", "measured_stats_sha256",
})


class RTX4090FP8BurnError(ValueError):
    """An input or receipt differs from the immutable campaign contract."""


def _verify_sealed_allocator_probe(
    descriptor: int,
    *,
    expected_bytes: int,
    expected_sha256: str,
    required_seals: int,
) -> None:
    """Recheck the exact immutable probe fd handed to the allocator child."""
    try:
        observed = os.fstat(descriptor)
        observed_seals = int(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS))
        digest = hashlib.sha256()
        offset = 0
        while offset < int(observed.st_size):
            block = os.pread(
                descriptor,
                min(1024 * 1024, int(observed.st_size) - offset),
                offset,
            )
            if not block:
                break
            digest.update(block)
            offset += len(block)
    except (AttributeError, OSError) as exc:
        raise RTX4090FP8BurnError(
            "sealed allocator probe descriptor cannot be verified"
        ) from exc
    if (
        int(observed.st_size) != int(expected_bytes)
        or offset != int(expected_bytes)
        or digest.hexdigest() != str(expected_sha256)
        or observed_seals & int(required_seals) != int(required_seals)
    ):
        raise RTX4090FP8BurnError(
            "sealed allocator probe differs from the validated bundle bytes"
        )


@contextmanager
def _sealed_allocator_probe(
    payload: bytes,
    *,
    expected_sha256: str,
):
    """Expose exact validated probe bytes at one stable inherited memfd path.

    The fixed descriptor keeps allocator invocation identities stable across
    resume.  ``F_DUPFD_CLOEXEC`` acquires it without overwriting an unrelated
    descriptor; an occupied slot is a fail-closed launch error.  Linux seals
    make replacement, truncation, and in-place writes impossible, and the fd
    is rechecked after the child returns.
    """
    required_names = (
        "F_ADD_SEALS", "F_GET_SEALS", "F_SEAL_WRITE", "F_SEAL_GROW",
        "F_SEAL_SHRINK", "F_SEAL_SEAL", "F_DUPFD_CLOEXEC",
    )
    if not hasattr(os, "memfd_create") or not hasattr(
        os, "MFD_ALLOW_SEALING"
    ) or any(not hasattr(fcntl, name) for name in required_names):
        raise RTX4090FP8BurnError(
            "sealed allocator probe handoff requires Linux memfd sealing"
        )
    if not isinstance(payload, bytes) or hashlib.sha256(
        payload
    ).hexdigest() != str(expected_sha256):
        raise RTX4090FP8BurnError(
            "captured allocator probe bytes differ from their bundle digest"
        )
    flags = int(getattr(os, "MFD_CLOEXEC", 0)) | int(os.MFD_ALLOW_SEALING)
    descriptor = os.memfd_create("prismaquant-allocator-probe", flags)
    fixed_descriptor: int | None = None
    required_seals = (
        int(fcntl.F_SEAL_WRITE) | int(fcntl.F_SEAL_GROW)
        | int(fcntl.F_SEAL_SHRINK) | int(fcntl.F_SEAL_SEAL)
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count < 1:
                raise RTX4090FP8BurnError(
                    "sealed allocator probe write made no progress"
                )
            written += count
        os.fsync(descriptor)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, required_seals)
        if descriptor == ALLOCATOR_PROBE_FD:
            fixed_descriptor = descriptor
            descriptor = -1
        else:
            candidate = int(fcntl.fcntl(
                descriptor, fcntl.F_DUPFD_CLOEXEC, ALLOCATOR_PROBE_FD,
            ))
            if candidate != ALLOCATOR_PROBE_FD:
                os.close(candidate)
                raise RTX4090FP8BurnError(
                    f"allocator probe fd {ALLOCATOR_PROBE_FD} is occupied"
                )
            fixed_descriptor = candidate
        _verify_sealed_allocator_probe(
            fixed_descriptor,
            expected_bytes=len(payload), expected_sha256=expected_sha256,
            required_seals=required_seals,
        )
        try:
            yield f"/proc/self/fd/{ALLOCATOR_PROBE_FD}", fixed_descriptor
        finally:
            _verify_sealed_allocator_probe(
                fixed_descriptor,
                expected_bytes=len(payload), expected_sha256=expected_sha256,
                required_seals=required_seals,
            )
    finally:
        if fixed_descriptor is not None:
            os.close(fixed_descriptor)
        if descriptor >= 0:
            os.close(descriptor)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _is_calibration_hash(value: object) -> bool:
    """The existing calibration_data_hash contract is BLAKE2b-128 hex."""
    return re.fullmatch(r"[0-9a-f]{32}", str(value)) is not None


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _binding(path: str | Path, *, where: str) -> dict[str, object]:
    """Bind an input without copying path-dependent or unrelated contents."""
    input_path = Path(path)
    if not input_path.is_file():
        raise RTX4090FP8BurnError(f"{where} is not a file: {input_path}")
    result: dict[str, object] = {
        "sha256": _sha256_file(input_path),
        "bytes": input_path.stat().st_size,
    }
    if input_path.suffix.lower() == ".json":
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RTX4090FP8BurnError(
                f"{where} is not valid JSON: {input_path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise RTX4090FP8BurnError(f"{where} must be a JSON object")
        schema = payload.get("schema")
        if schema is not None:
            result["schema"] = str(schema)
        for key in (
            "content_sha256", "resolved_commit", "source_sha256",
            "tree_sha256", "closure_sha256", "identity_sha256",
            "cover_identity_sha256", "execution_identity_sha256",
            "manifest_identity_sha256",
            "activation_qname_manifest_sha256", "source_census_sha256",
            "container_image_digest", "commit", "tree",
        ):
            value = payload.get(key)
            if value is not None:
                result[key] = str(value)
    return result


def _sample_execution_projection(
    execution_identity: Mapping[str, object],
) -> dict[str, str]:
    """Project the sample producer's already-validated execution authority."""
    projection = {
        "sample_execution_identity_sha256": str(
            execution_identity.get("identity_sha256", "")
        ),
        "producer_snapshot_closure_sha256": str(
            execution_identity.get("producer_snapshot_sha256", "")
        ),
        "producer_snapshot_commit": str(
            execution_identity.get("producer_snapshot_commit", "")
        ),
        "producer_snapshot_tree": str(
            execution_identity.get("producer_snapshot_tree", "")
        ),
        "container_image_digest": str(
            execution_identity.get("container_image_digest", "")
        ),
    }
    if (
        not _is_sha256(projection["sample_execution_identity_sha256"])
        or not _is_sha256(
            projection["producer_snapshot_closure_sha256"]
        )
        or re.fullmatch(
            r"[0-9a-f]{40}", projection["producer_snapshot_commit"]
        ) is None
        or re.fullmatch(
            r"[0-9a-f]{40}", projection["producer_snapshot_tree"]
        ) is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}", projection["container_image_digest"]
        ) is None
    ):
        raise RTX4090FP8BurnError(
            "sample run-contract execution identity is malformed"
        )
    return projection


def build_execution_attestation(
    execution_identity: Mapping[str, object],
    *,
    producer_snapshot: Mapping[str, object],
    launcher_image_digest: str,
) -> dict[str, object]:
    """Build the closed launcher record from the sample execution identity.

    The image value is intentionally supplied by the host launcher.  Code in
    the container cannot authoritatively discover the immutable registry
    RepoDigest that launched it, so this function only proves that the
    supplied value agrees with the earlier sample run-contract authority.
    """
    projection = _sample_execution_projection(execution_identity)
    snapshot = {
        "producer_snapshot_closure_sha256": str(
            producer_snapshot.get("closure_sha256", "")
        ),
        "producer_snapshot_commit": str(producer_snapshot.get("commit", "")),
        "producer_snapshot_tree": str(producer_snapshot.get("tree", "")),
    }
    if any(snapshot[key] != projection[key] for key in snapshot):
        raise RTX4090FP8BurnError(
            "live runtime snapshot differs from the sample execution identity"
        )
    if str(launcher_image_digest) != projection["container_image_digest"]:
        raise RTX4090FP8BurnError(
            "trusted launcher image digest differs from the sample execution "
            "identity"
        )
    body: dict[str, object] = {
        "schema": EXECUTION_ATTESTATION_SCHEMA,
        **projection,
    }
    return {
        **body,
        "identity_sha256": canonical_json_sha256(
            body, where="RTX4090 launcher execution attestation",
        ),
    }


def _validate_execution_attestation(
    path: str | Path,
    *,
    execution_identity: Mapping[str, object],
    producer_snapshot: Mapping[str, object],
    launcher_image_digest: str,
) -> dict[str, object]:
    """Validate one closed launcher record against live pre-GPU inputs."""
    from prismaquant.sample_parallel_probe import _strict_json_loads

    attestation_path = Path(path)
    if attestation_path.is_symlink() or not attestation_path.is_file():
        raise RTX4090FP8BurnError(
            "execution attestation must be one regular JSON file"
        )
    try:
        raw = _strict_json_loads(
            attestation_path.read_text(encoding="utf-8"),
            where="RTX4090 launcher execution attestation",
        )
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"execution attestation is unreadable: {exc}"
        ) from exc
    keys = {
        "schema", "sample_execution_identity_sha256",
        "producer_snapshot_closure_sha256", "producer_snapshot_commit",
        "producer_snapshot_tree", "container_image_digest",
        "identity_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise RTX4090FP8BurnError(
            "execution attestation fields differ from the closed v1 contract"
        )
    body = dict(raw)
    identity_sha256 = body.pop("identity_sha256", None)
    if (
        body.get("schema") != EXECUTION_ATTESTATION_SCHEMA
        or identity_sha256 != canonical_json_sha256(
            body, where="RTX4090 launcher execution attestation",
        )
    ):
        raise RTX4090FP8BurnError(
            "execution attestation checksum differs"
        )
    expected = build_execution_attestation(
        execution_identity,
        producer_snapshot=producer_snapshot,
        launcher_image_digest=launcher_image_digest,
    )
    if dict(raw) != expected:
        raise RTX4090FP8BurnError(
            "execution attestation differs from the live sample/snapshot/image "
            "contract"
        )
    return expected


def _source_identity_binding(
    model: str | Path,
    cache_path: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate and bind the repository's nested streamed-identity cache."""
    from prismaquant.cost_streaming import (
        portable_streamed_model_content_identity,
        validate_cached_streamed_model_identity,
    )

    try:
        live_identity = validate_cached_streamed_model_identity(
            model, cache_path, require_complete_checkpoint=True,
        )
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"source model identity cache validation failed: {exc}"
        ) from exc
    try:
        portable_content = portable_streamed_model_content_identity(
            live_identity,
            where="RTX4090 source model portable content",
        )["portable_content_sha256"]
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"source model portable identity derivation failed: {exc}"
        ) from exc
    identity_schema = live_identity.get("schema")
    if (
        identity_schema != "prismaquant.streamed_model.identity.v1"
        or not _is_sha256(portable_content)
    ):
        raise RTX4090FP8BurnError(
            "validated source model identity lacks its schema/content digest"
        )
    # The cache envelope contains host-local paths, inode fingerprints, and
    # timestamps.  Binding its raw bytes makes a plan prepared on Sparky
    # impossible to consume with Sparklina's independently validated cache.
    # Bind only the value-bearing streamed identity; each host still validates
    # its own envelope and source fingerprints above before deriving this.
    body: dict[str, object] = {
        "schema": SOURCE_IDENTITY_BINDING_SCHEMA,
        "portable_content_sha256": str(portable_content),
    }
    identity_sha256 = canonical_json_sha256(
        body, where="RTX4090 source model identity binding",
    )
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    descriptor = {
        **body,
        "identity_sha256": identity_sha256,
        # Generic campaign bindings require a digest/length pair.  Here they
        # describe the canonical value-bearing body, never the host cache file.
        "sha256": identity_sha256,
        "bytes": len(encoded),
    }
    return descriptor, dict(live_identity)


def _verify_source_identity_binding(
    plan: Mapping[str, object],
    *,
    model: str | Path,
    cache_path: str | Path,
) -> dict[str, object]:
    bindings = plan.get("bindings")
    expected = (
        bindings.get("source_model_identity")
        if isinstance(bindings, Mapping) else None
    )
    if not isinstance(expected, Mapping):
        raise RTX4090FP8BurnError(
            "plan has no source_model_identity binding"
        )
    observed, live_identity = _source_identity_binding(model, cache_path)
    if observed != dict(expected):
        raise RTX4090FP8BurnError(
            "source_model_identity differs from the prepared plan"
        )
    return live_identity


def _normalized_probe_payload(
    payload: object,
) -> tuple[dict[str, dict], dict[str, Any]]:
    stats = payload.get("stats") if isinstance(payload, Mapping) else None
    meta = payload.get("meta") if isinstance(payload, Mapping) else None
    if not isinstance(stats, Mapping) or not isinstance(meta, Mapping):
        raise RTX4090FP8BurnError("probe lacks stats/meta mappings")
    return (
        {str(name): dict(row) for name, row in stats.items()
         if isinstance(row, Mapping)},
        dict(meta),
    )


def _probe_payload(path: str | Path) -> tuple[dict[str, dict], dict[str, Any]]:
    try:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)  # trusted local pipeline artifact
    except Exception as exc:
        raise RTX4090FP8BurnError(f"probe is unreadable: {path}") from exc
    return _normalized_probe_payload(payload)


def _probe_imatrix_contract(
    probe: str | Path,
    col_weights: str | Path,
    *,
    validated_probe_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Prove the supplied render weights are exactly derived from the probe."""
    import torch

    from prismaquant.cb_imatrix import (
        canonical_imatrix_sha256,
        imatrix_from_probe_file,
        imatrix_from_probe_stats,
    )

    try:
        if validated_probe_payload is None:
            expected, provenance = imatrix_from_probe_file(probe)
        else:
            probe_stats, probe_meta = _normalized_probe_payload(
                validated_probe_payload
            )
            expected, provenance = imatrix_from_probe_stats(probe_stats)
            calibration_hash = probe_meta.get("calib_hash")
            if calibration_hash is not None:
                provenance = {
                    **provenance,
                    "calibration_hash": str(calibration_hash),
                }
        with Path(col_weights).open("rb") as handle:
            raw = pickle.load(handle)
        if not isinstance(raw, Mapping):
            raise ValueError("column-weight pickle is not a mapping")
        observed = {
            str(name): torch.as_tensor(value).detach().to(
                device="cpu", dtype=torch.float32,
            ).contiguous()
            for name, value in raw.items()
        }
        expected_names = tuple(sorted(expected))
        if tuple(sorted(observed)) != expected_names:
            raise ValueError(
                "column-weight qnames differ from probe-derived imatrix"
            )
        expected_digest = canonical_imatrix_sha256(expected)
        observed_digest = canonical_imatrix_sha256(observed)
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"probe/column-weight imatrix contract differs: {exc}"
        ) from exc
    calibration_hash = provenance.get("calibration_hash")
    if not _is_calibration_hash(calibration_hash):
        raise RTX4090FP8BurnError(
            "probe-derived imatrix lacks the exact calibration identity"
        )
    if observed_digest != expected_digest or provenance.get(
        "value_sha256"
    ) != expected_digest:
        raise RTX4090FP8BurnError(
            "column-weight values differ from the bound probe marginals"
        )
    return {
        "schema": IMATRIX_CONTRACT_SCHEMA,
        "derivation_schema": provenance.get("schema"),
        "calibration_hash": str(calibration_hash),
        "qname_count": len(expected_names),
        "qname_census_sha256": canonical_json_sha256(
            list(expected_names), where="probe imatrix qname census",
        ),
        "value_sha256": expected_digest,
    }


def derive_col_weights(args: argparse.Namespace) -> Path:
    """Publish the raw CB imatrix map from one validated merge bundle.

    The bundle validator captures the exact probe object whose bytes and
    complete probe/activation topology it validated.  Derivation deliberately
    consumes that object rather than reopening ``probe.pkl`` after the trust
    boundary.  Publication uses the sample producer's durable hard-link
    primitive so an existing or concurrently created output wins unchanged.
    """
    from prismaquant.cb_imatrix import imatrix_from_probe_stats
    from prismaquant.sample_parallel_probe import (
        _atomic_write_bytes_no_clobber,
        validate_sample_parallel_merge_bundle,
    )

    bundle = Path(args.sample_merge_bundle)
    try:
        validated = validate_sample_parallel_merge_bundle(
            bundle, capture_consumables=True,
        )
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"sample-merge bundle validation failed: {exc}"
        ) from exc
    if not isinstance(validated, Mapping):
        raise RTX4090FP8BurnError(
            "sample-merge bundle validator returned no contract"
        )
    probe_payload = validated.get("_validated_probe_payload")
    if not isinstance(probe_payload, Mapping):
        raise RTX4090FP8BurnError(
            "sample-merge bundle validator did not capture its probe"
        )
    try:
        probe_stats, probe_meta = _normalized_probe_payload(probe_payload)
        calibration_hash = probe_meta.get("calib_hash")
        if not _is_calibration_hash(calibration_hash):
            raise ValueError(
                "validated probe lacks the exact BLAKE2b-128 calibration "
                "identity"
            )
        col_weights, provenance = imatrix_from_probe_stats(probe_stats)
        if provenance.get("value_sha256") is None:
            raise ValueError("probe-derived imatrix lacks its value identity")
        payload = pickle.dumps(
            col_weights, protocol=pickle.HIGHEST_PROTOCOL,
        )
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"validated sample-merge imatrix derivation failed: {exc}"
        ) from exc

    output = Path(args.output)
    try:
        _atomic_write_bytes_no_clobber(output, payload)
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"column-weight publication failed: {exc}"
        ) from exc
    return output


def _calibration_contract(
    meta: Mapping[str, object], *, nsamples: int, seqlen: int, seed: int,
) -> dict[str, object]:
    for name, expected in (("nsamples", nsamples), ("seqlen", seqlen)):
        if int(meta.get(name, -1)) != int(expected):
            raise RTX4090FP8BurnError(
                f"probe {name}={meta.get(name)!r}, expected {expected}"
            )
    digest = meta.get("calib_hash")
    if not isinstance(digest, str) or not _is_calibration_hash(digest):
        raise RTX4090FP8BurnError(
            "probe calib_hash is not the exact BLAKE2b-128 calibration identity"
        )
    return {
        "calib_hash": digest,
        "nsamples": int(nsamples),
        "seqlen": int(seqlen),
        "seed": int(seed),
    }


def _validate_stats(stats: Mapping[str, Mapping[str, object]]) -> None:
    if not stats:
        raise RTX4090FP8BurnError("body probe census is empty")
    for qname, row in stats.items():
        n_params = int(row.get("n_params", 0) or 0)
        in_features = int(row.get("in_features", 0) or 0)
        out_features = int(row.get("out_features", 0) or 0)
        if min(n_params, in_features, out_features) <= 0:
            raise RTX4090FP8BurnError(f"{qname}: incomplete positive shape")
        if n_params != in_features * out_features:
            raise RTX4090FP8BurnError(
                f"{qname}: n_params differs from in_features*out_features"
            )


def _qname_maps(qnames: Sequence[str]) -> dict[str, object]:
    ordered = tuple(sorted(str(name) for name in qnames))
    return {
        "formats_by_qname": {name: list(RENDER_FORMATS) for name in ordered},
        "purposes_by_qname": {
            name: {
                MEASURED_CB_FORMATS[0]: ["panel"],
                ANCHOR_CB_FORMAT: ["anchor", "panel"],
                MEASURED_CB_FORMATS[-1]: ["panel"],
                NATIVE_FP8_FORMAT: ["anchor"],
            }
            for name in ordered
        },
        "unmeasured_formats_by_qname": {
            name: [BF16_FORMAT] for name in ordered
        },
        "legal_cb_formats_by_qname": {
            name: list(CB_FORMATS) for name in ordered
        },
    }


def _stripe_metrics(
    qnames: Sequence[str], stats: Mapping[str, Mapping[str, object]],
) -> dict[str, int]:
    return {
        "qnames": len(qnames),
        "parameters": sum(int(stats[name]["n_params"]) for name in qnames),
        "estimated_work": sum(
            int(stats[name]["n_params"])
            * max(int(stats[name]["in_features"]), 1)
            for name in qnames
        ),
        "render_cells": len(qnames) * len(MEASURED_FORMATS),
    }


def _campaign_stripes(
    stats: Mapping[str, Mapping[str, object]], *, profile,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """Resolve two LPT bins, preferring an exactly tied contiguous split.

    For the periodic 64-layer Qwen body, contiguous halves are another
    optimal LPT tie resolution.  Prefer them only after proving that their
    qname, parameter, work, and render-cell totals are exactly equal and that
    those totals match the ordinary deterministic LPT solution.
    """
    lpt = plan_stripes(stats, profile=profile, n_stripes=STRIPE_COUNT)
    lpt_metrics = tuple(_stripe_metrics(stripe.qnames, stats) for stripe in lpt)
    selected_qnames = tuple(tuple(stripe.qnames) for stripe in lpt)
    selected_groups = tuple(tuple(stripe.groups) for stripe in lpt)
    strategy = "whole_layer_lpt"
    layer_ranges: tuple[list[int] | None, ...] = (None, None)

    by_layer: dict[int, list[str]] = {}
    non_layer: list[str] = []
    for qname in sorted(stats):
        match = _LAYER_RE.search(qname)
        if match is None:
            non_layer.append(qname)
        else:
            by_layer.setdefault(int(match.group(1)), []).append(qname)
    layers = tuple(sorted(by_layer))
    if not non_layer and len(layers) == 64 and layers == tuple(range(64)):
        halves = (layers[:32], layers[32:])
        contiguous = tuple(
            tuple(sorted(name for layer in half for name in by_layer[layer]))
            for half in halves
        )
        contiguous_metrics = tuple(_stripe_metrics(names, stats) for names in contiguous)
        metric_names = ("qnames", "parameters", "estimated_work", "render_cells")
        exactly_equal = all(
            contiguous_metrics[0][name] == contiguous_metrics[1][name]
            for name in metric_names
        )
        same_as_lpt = sorted(
            tuple(row[name] for name in metric_names) for row in contiguous_metrics
        ) == sorted(tuple(row[name] for name in metric_names) for row in lpt_metrics)
        if exactly_equal and same_as_lpt:
            selected_qnames = contiguous
            selected_groups = tuple(
                tuple(f"layer:{layer}" for layer in half) for half in halves
            )
            strategy = "whole_layer_lpt_contiguous_equal_tie"
            layer_ranges = ([0, 31], [32, 63])

    selected_metrics = tuple(_stripe_metrics(names, stats) for names in selected_qnames)
    records = tuple({
        "index": index,
        "qnames": list(selected_qnames[index]),
        "groups": list(selected_groups[index]),
        "estimated_work": selected_metrics[index]["estimated_work"],
        "parameters": selected_metrics[index]["parameters"],
        "render_cells": selected_metrics[index]["render_cells"],
        "layer_range": layer_ranges[index],
        "qname_file": f"stripe-{index:02d}.qnames.txt",
        "qname_file_sha256": hashlib.sha256("".join(
            f"{name}\n" for name in selected_qnames[index]
        ).encode("utf-8")).hexdigest(),
    } for index in range(STRIPE_COUNT))
    proof = {
        "strategy": strategy,
        "metric_names": ["qnames", "parameters", "estimated_work", "render_cells"],
        "selected": [dict(row) for row in selected_metrics],
        "ordinary_lpt": [dict(row) for row in lpt_metrics],
        "selected_metrics_exactly_equal": selected_metrics[0] == selected_metrics[1],
        "selected_matches_lpt_loads": sorted(selected_metrics, key=lambda row: tuple(row.values()))
        == sorted(lpt_metrics, key=lambda row: tuple(row.values())),
    }
    return records, proof


def _fixed_census_records(
    fixed_bf16: Mapping[str, Mapping[str, object] | str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for qname in sorted(fixed_bf16):
        raw = fixed_bf16[qname]
        if isinstance(raw, Mapping):
            reason = str(raw.get("reason", "fixed_bf16"))
            record = {
                "qname": qname,
                "format": BF16_FORMAT,
                "reason": reason,
                "source_dtype": str(raw.get("source_dtype", "bf16")),
                "n_params": int(raw.get("n_params", 0) or 0),
            }
        else:
            record = {
                "qname": qname,
                "format": BF16_FORMAT,
                "reason": str(raw),
                "source_dtype": "bf16",
                "n_params": 0,
            }
        if record["source_dtype"] != "bf16":
            raise RTX4090FP8BurnError(
                f"fixed unit {qname} is not a BF16 source"
            )
        records.append(record)
    return records


def build_campaign_plan(
    stats: Mapping[str, Mapping[str, object]],
    *,
    profile,
    fixed_bf16: Mapping[str, Mapping[str, object] | str],
    calibration: Mapping[str, object],
    bindings: Mapping[str, Mapping[str, object]],
    source_dtype_census_sha256: str,
    imatrix_contract: Mapping[str, object],
) -> dict[str, object]:
    """Build the path-independent global plan used by both GPU hosts."""
    validate_rtx4090_format_menu(FULL_FORMATS)
    body_stats = {str(name): dict(row) for name, row in stats.items()}
    _validate_stats(body_stats)
    fixed_records = _fixed_census_records(fixed_bf16)
    overlap = sorted(set(body_stats) & {str(row["qname"]) for row in fixed_records})
    if overlap:
        raise RTX4090FP8BurnError(
            f"body and fixed-BF16 censuses overlap: {overlap[:8]}"
        )
    stripe_records, balance_proof = _campaign_stripes(
        body_stats, profile=profile
    )
    maps = _qname_maps(tuple(sorted(body_stats)))
    plan: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "policy": {
            "serving_profile": RTX4090_QWEN38_SERVING_PROFILE,
            "target_bytes": RTX4090_CONTEXT_FIRST_TARGET_BYTES,
            "artifact_overhead_reserve_bytes": ARTIFACT_OVERHEAD_RESERVE_BYTES,
            "formats": list(FULL_FORMATS),
            "rendered_formats": list(RENDER_FORMATS),
            "measured_formats": list(MEASURED_FORMATS),
            "measured_cb_anchors": list(MEASURED_CB_FORMATS),
            "primary_cb_anchor": ANCHOR_CB_FORMAT,
            "unmeasured_terminal": BF16_FORMAT,
            "codebook_formats": list(CB_FORMATS),
            "lattice_only": True,
        },
        "producer": {
            "cb_serialization": dict(CB_PRODUCER_SETTINGS),
            "render_levers": dict(RENDER_LEVERS),
            "transient_renders": True,
            "purpose": "anchor",
            "n_probes": AURA_N_PROBES,
            "token_scope": AURA_TOKEN_SCOPE,
            "streamed_model_cache": {
                "max_cache_slots": STREAMING_CACHE_MAX_SLOTS,
                "effective_prefetch_lookahead": STREAMING_PREFETCH_LOOKAHEAD,
                "require_prefetched_residency": (
                    STREAMING_REQUIRE_PREFETCHED_RESIDENCY
                ),
            },
        },
        "calibration": dict(calibration),
        "imatrix": dict(imatrix_contract),
        "bindings": {str(k): dict(v) for k, v in sorted(bindings.items())},
        "source_dtype_census_sha256": str(source_dtype_census_sha256),
        "body": {
            "qnames": list(sorted(body_stats)),
            "qname_count": len(body_stats),
            "parameters": sum(
                int(row["n_params"]) for row in body_stats.values()
            ),
            "shapes": {
                name: [int(body_stats[name]["out_features"]),
                       int(body_stats[name]["in_features"])]
                for name in sorted(body_stats)
            },
        },
        "fixed_bf16_census": fixed_records,
        "fixed_bf16_census_sha256": canonical_json_sha256(
            fixed_records, where="RTX4090 fixed BF16 census"
        ),
        "stripes": stripe_records,
        "stripe_balance_proof": balance_proof,
        "maps": maps,
    }
    plan["plan_sha256"] = canonical_json_sha256(
        plan, where="RTX4090 FP8 burn plan"
    )
    validate_campaign_plan(plan)
    return plan


def validate_campaign_plan(plan: Mapping[str, object]) -> None:
    expected_top_keys = {
        "schema", "policy", "producer", "calibration", "bindings",
        "imatrix", "source_dtype_census_sha256", "body", "fixed_bf16_census",
        "fixed_bf16_census_sha256", "stripes", "stripe_balance_proof",
        "maps", "plan_sha256",
    }
    if set(plan) != expected_top_keys:
        raise RTX4090FP8BurnError("campaign plan top-level shape is not closed")
    if plan.get("schema") != PLAN_SCHEMA:
        raise RTX4090FP8BurnError("campaign plan schema mismatch")
    expected_digest = plan.get("plan_sha256")
    without_digest = dict(plan)
    without_digest.pop("plan_sha256", None)
    observed_digest = canonical_json_sha256(
        without_digest, where="RTX4090 FP8 burn plan"
    )
    if expected_digest != observed_digest:
        raise RTX4090FP8BurnError("campaign plan digest mismatch")
    policy = plan.get("policy")
    producer = plan.get("producer")
    calibration = plan.get("calibration")
    imatrix = plan.get("imatrix")
    bindings = plan.get("bindings")
    body = plan.get("body")
    maps = plan.get("maps")
    stripes = plan.get("stripes")
    if not all(isinstance(item, Mapping) for item in (
        policy, producer, calibration, imatrix, bindings, body, maps,
    )):
        raise RTX4090FP8BurnError(
            "campaign plan lacks closed policy/producer/calibration/"
            "bindings/body/maps"
        )
    if not isinstance(stripes, Sequence) or isinstance(stripes, (str, bytes)):
        raise RTX4090FP8BurnError("campaign plan lacks stripe records")
    assert isinstance(policy, Mapping) and isinstance(producer, Mapping)
    assert isinstance(calibration, Mapping) and isinstance(bindings, Mapping)
    assert isinstance(imatrix, Mapping)
    assert isinstance(body, Mapping) and isinstance(maps, Mapping)
    expected_policy = {
        "serving_profile": RTX4090_QWEN38_SERVING_PROFILE,
        "target_bytes": RTX4090_CONTEXT_FIRST_TARGET_BYTES,
        "artifact_overhead_reserve_bytes": ARTIFACT_OVERHEAD_RESERVE_BYTES,
        "formats": list(FULL_FORMATS),
        "rendered_formats": list(RENDER_FORMATS),
        "measured_formats": list(MEASURED_FORMATS),
        "measured_cb_anchors": list(MEASURED_CB_FORMATS),
        "primary_cb_anchor": ANCHOR_CB_FORMAT,
        "unmeasured_terminal": BF16_FORMAT,
        "codebook_formats": list(CB_FORMATS),
        "lattice_only": True,
    }
    if dict(policy) != expected_policy:
        raise RTX4090FP8BurnError("campaign policy constants are not exact")
    expected_producer = {
        "cb_serialization": dict(CB_PRODUCER_SETTINGS),
        "render_levers": dict(RENDER_LEVERS),
        "transient_renders": True,
        "purpose": "anchor",
        "n_probes": AURA_N_PROBES,
        "token_scope": AURA_TOKEN_SCOPE,
        "streamed_model_cache": {
            "max_cache_slots": STREAMING_CACHE_MAX_SLOTS,
            "effective_prefetch_lookahead": STREAMING_PREFETCH_LOOKAHEAD,
            "require_prefetched_residency": (
                STREAMING_REQUIRE_PREFETCHED_RESIDENCY
            ),
        },
    }
    if dict(producer) != expected_producer:
        raise RTX4090FP8BurnError("campaign producer constants are not exact")

    if set(calibration) != {"calib_hash", "nsamples", "seqlen", "seed"}:
        raise RTX4090FP8BurnError("campaign calibration shape is not closed")
    if (
        int(calibration.get("nsamples", -1)) != CALIBRATION_NSAMPLES
        or int(calibration.get("seqlen", -1)) != CALIBRATION_SEQLEN
        or int(calibration.get("seed", -1)) != CALIBRATION_SEED
        or not _is_calibration_hash(calibration.get("calib_hash"))
    ):
        raise RTX4090FP8BurnError("campaign calibration constants are not exact")
    if set(imatrix) != {
        "schema", "derivation_schema", "calibration_hash", "qname_count",
        "qname_census_sha256", "value_sha256",
    } or (
        imatrix.get("schema") != IMATRIX_CONTRACT_SCHEMA
        or not str(imatrix.get("derivation_schema", ""))
        or imatrix.get("calibration_hash") != calibration.get("calib_hash")
        or int(imatrix.get("qname_count", 0)) < int(
            body.get("qname_count", 0)
        )
        or not _is_sha256(imatrix.get("qname_census_sha256"))
        or not _is_sha256(imatrix.get("value_sha256"))
    ):
        raise RTX4090FP8BurnError(
            "campaign probe-derived imatrix contract is not exact"
        )

    required_bindings = {
        "probe", "col_weights", "source_model_identity",
        "producer_snapshot", "common_execution_attestation", "dataset",
        "sample_merge_commit", "activation_cache_manifest",
    }
    if set(bindings) != required_bindings:
        raise RTX4090FP8BurnError("campaign binding names are not exact")
    allowed_binding_fields = {
        "sha256", "bytes", "schema", "content_sha256", "resolved_commit",
        "portable_content_sha256",
        "source_sha256", "tree_sha256", "closure_sha256",
        "identity_sha256", "cover_identity_sha256",
        "execution_identity_sha256", "manifest_identity_sha256",
        "activation_qname_manifest_sha256", "source_census_sha256",
        "container_image_digest", "commit", "tree",
    }
    for name in sorted(required_bindings):
        descriptor = bindings.get(name)
        if not isinstance(descriptor, Mapping) or not {
            "sha256", "bytes",
        }.issubset(descriptor) or set(descriptor) - allowed_binding_fields:
            raise RTX4090FP8BurnError(
                f"campaign {name} binding shape is not closed"
            )
        if not _is_sha256(descriptor.get("sha256")) or int(
            descriptor.get("bytes", -1)
        ) < 1:
            raise RTX4090FP8BurnError(
                f"campaign {name} binding digest/size is invalid"
            )
        for field in (
            "content_sha256", "portable_content_sha256", "source_sha256",
            "tree_sha256",
            "closure_sha256", "identity_sha256", "cover_identity_sha256",
            "execution_identity_sha256", "manifest_identity_sha256",
            "activation_qname_manifest_sha256", "source_census_sha256",
        ):
            if field in descriptor and not _is_sha256(descriptor[field]):
                raise RTX4090FP8BurnError(
                    f"campaign {name} binding {field} is invalid"
                )
        for field in ("commit", "tree"):
            if field in descriptor and re.fullmatch(
                r"[0-9a-f]{40}", str(descriptor[field])
            ) is None:
                raise RTX4090FP8BurnError(
                    f"campaign {name} binding {field} is invalid"
                )
        if "resolved_commit" in descriptor:
            commit = str(descriptor["resolved_commit"])
            if len(commit) not in {40, 64} or any(
                character not in "0123456789abcdef" for character in commit
            ):
                raise RTX4090FP8BurnError(
                    f"campaign {name} resolved commit is invalid"
                )
        if "container_image_digest" in descriptor and re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(descriptor["container_image_digest"])
        ) is None:
            raise RTX4090FP8BurnError(
                f"campaign {name} container image digest is invalid"
            )
        if "schema" in descriptor and not str(descriptor["schema"]).strip():
            raise RTX4090FP8BurnError(
                f"campaign {name} binding schema is empty"
            )
    source_binding = bindings["source_model_identity"]
    assert isinstance(source_binding, Mapping)
    source_body = {
        key: source_binding[key]
        for key in ("schema", "portable_content_sha256")
        if key in source_binding
    }
    source_encoded = json.dumps(
        source_body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    source_identity_sha256 = canonical_json_sha256(
        source_body, where="campaign source model identity binding",
    )
    expected_source_keys = {
        "schema", "portable_content_sha256", "identity_sha256", "sha256",
        "bytes",
    }
    if (
        set(source_binding) != expected_source_keys
        or source_binding.get("schema") != SOURCE_IDENTITY_BINDING_SCHEMA
        or not _is_sha256(source_binding.get("portable_content_sha256"))
        or source_binding.get("identity_sha256") != source_identity_sha256
        or source_binding.get("sha256") != source_identity_sha256
        or source_binding.get("bytes") != len(source_encoded)
    ):
        raise RTX4090FP8BurnError(
            "campaign source-model binding is not the exact portable value "
            "identity"
        )
    execution_binding = bindings["common_execution_attestation"]
    assert isinstance(execution_binding, Mapping)
    if (
        execution_binding.get("schema") != EXECUTION_ATTESTATION_SCHEMA
        or not _is_sha256(execution_binding.get("identity_sha256"))
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(execution_binding.get("container_image_digest", "")),
        ) is None
    ):
        raise RTX4090FP8BurnError(
            "campaign execution-attestation binding is not closed"
        )
    if not _is_sha256(plan.get("source_dtype_census_sha256")):
        raise RTX4090FP8BurnError("campaign source dtype census digest is invalid")

    if set(body) != {"qnames", "qname_count", "parameters", "shapes"}:
        raise RTX4090FP8BurnError("campaign body shape is not closed")
    qnames = tuple(str(name) for name in body.get("qnames", ()))
    if (
        not qnames or qnames != tuple(sorted(qnames))
        or len(qnames) != len(set(qnames))
    ):
        raise RTX4090FP8BurnError("body qnames are not unique and sorted")
    shapes = body.get("shapes")
    if not isinstance(shapes, Mapping) or set(map(str, shapes)) != set(qnames):
        raise RTX4090FP8BurnError("campaign body shape census is not exact")
    body_parameters = 0
    body_stats: dict[str, dict[str, int]] = {}
    layers: set[int] = set()
    for qname in qnames:
        raw_shape = shapes[qname]
        if not isinstance(raw_shape, Sequence) or isinstance(
            raw_shape, (str, bytes)
        ) or len(raw_shape) != 2:
            raise RTX4090FP8BurnError(f"{qname}: body shape is malformed")
        out_features, in_features = map(int, raw_shape)
        if min(out_features, in_features) <= 0:
            raise RTX4090FP8BurnError(f"{qname}: body shape is not positive")
        n_params = out_features * in_features
        body_parameters += n_params
        body_stats[qname] = {
            "out_features": out_features,
            "in_features": in_features,
            "n_params": n_params,
        }
        match = _LAYER_RE.search(qname)
        if match is None:
            raise RTX4090FP8BurnError(
                f"{qname}: RTX4090 body qname has no decoder-layer owner"
            )
        layers.add(int(match.group(1)))
    if layers != set(range(64)):
        raise RTX4090FP8BurnError("campaign body is not the exact 64-layer lane")
    if int(body.get("qname_count", -1)) != len(qnames) or int(
        body.get("parameters", -1)
    ) != body_parameters:
        raise RTX4090FP8BurnError("campaign body counters differ from shapes")

    fixed = plan.get("fixed_bf16_census")
    if not isinstance(fixed, Sequence) or isinstance(fixed, (str, bytes)):
        raise RTX4090FP8BurnError("campaign fixed-BF16 census is malformed")
    normalized_fixed: list[dict[str, object]] = []
    fixed_names: list[str] = []
    for raw in fixed:
        if not isinstance(raw, Mapping) or set(raw) != {
            "qname", "format", "reason", "source_dtype", "n_params",
        }:
            raise RTX4090FP8BurnError(
                "campaign fixed-BF16 record shape is not closed"
            )
        record = dict(raw)
        qname = str(record["qname"])
        if (
            not qname or record["format"] != BF16_FORMAT
            or record["source_dtype"] != "bf16"
            or record["reason"] not in {
                "profile_pinned", "mtp_fixed", "visual_fixed",
            }
            or int(record["n_params"]) < 0
        ):
            raise RTX4090FP8BurnError(
                f"campaign fixed-BF16 record for {qname!r} is invalid"
            )
        record["n_params"] = int(record["n_params"])
        fixed_names.append(qname)
        normalized_fixed.append(record)
    if fixed_names != sorted(fixed_names) or len(fixed_names) != len(
        set(fixed_names)
    ) or set(fixed_names) & set(qnames):
        raise RTX4090FP8BurnError(
            "campaign fixed-BF16 census is unsorted, duplicate, or overlaps body"
        )
    fixed_digest = canonical_json_sha256(
        normalized_fixed, where="RTX4090 fixed BF16 census",
    )
    if plan.get("fixed_bf16_census_sha256") != fixed_digest:
        raise RTX4090FP8BurnError("campaign fixed-BF16 census digest differs")

    expected_maps = _qname_maps(qnames)
    if dict(maps) != expected_maps:
        raise RTX4090FP8BurnError("campaign qname maps are not the exact menu")
    if len(stripes) != STRIPE_COUNT:
        raise RTX4090FP8BurnError("campaign must contain exactly two stripes")
    flattened: list[str] = []
    owners: dict[str, int] = {}
    for expected_index, raw in enumerate(stripes):
        expected_stripe_keys = {
            "index", "qnames", "groups", "estimated_work", "parameters",
            "render_cells", "layer_range", "qname_file",
            "qname_file_sha256",
        }
        if not isinstance(raw, Mapping) or set(raw) != expected_stripe_keys or int(
            raw.get("index", -1)
        ) != expected_index:
            raise RTX4090FP8BurnError("stripe indices are not canonical")
        members = [str(name) for name in raw.get("qnames", ())]
        if members != sorted(members) or len(members) != len(set(members)):
            raise RTX4090FP8BurnError("stripe qnames are not unique and sorted")
        text = "".join(f"{name}\n" for name in members).encode("utf-8")
        if raw.get("qname_file_sha256") != hashlib.sha256(text).hexdigest():
            raise RTX4090FP8BurnError("stripe qname file binding mismatch")
        if raw.get("qname_file") != f"stripe-{expected_index:02d}.qnames.txt":
            raise RTX4090FP8BurnError("stripe qname filename is not canonical")
        expected_metrics = _stripe_metrics(members, body_stats)
        if any(int(raw.get(name, -1)) != expected_metrics[name] for name in (
            "parameters", "estimated_work", "render_cells",
        )):
            raise RTX4090FP8BurnError("stripe metrics differ from body shapes")
        member_layers = sorted({
            int(_LAYER_RE.search(name).group(1))  # type: ignore[union-attr]
            for name in members
        })
        expected_groups = [f"layer:{layer}" for layer in member_layers]
        if list(raw.get("groups", ())) != expected_groups:
            raise RTX4090FP8BurnError("stripe groups differ from qname owners")
        if raw.get("layer_range") != [member_layers[0], member_layers[-1]]:
            raise RTX4090FP8BurnError("stripe layer range differs from qnames")
        flattened.extend(members)
        for name in members:
            layer_groups = tuple(
                group for group in raw.get("groups", ())
                if str(group).startswith("layer:")
            )
            if layer_groups:
                owners[name] = expected_index
    if len(flattened) != len(set(flattened)) or set(flattened) != set(qnames):
        raise RTX4090FP8BurnError("stripes are not an exact disjoint body cover")
    # Every LPT decoder group is written once; this catches hand-edited plans
    # even when the flattened qname cover still looks complete.
    all_groups = [str(group) for raw in stripes if isinstance(raw, Mapping)
                  for group in raw.get("groups", ())]
    if len(all_groups) != len(set(all_groups)):
        raise RTX4090FP8BurnError("a whole-layer group spans stripes")
    proof = plan.get("stripe_balance_proof")
    if not isinstance(proof, Mapping) or set(proof) != {
        "strategy", "metric_names", "selected", "ordinary_lpt",
        "selected_metrics_exactly_equal", "selected_matches_lpt_loads",
    }:
        raise RTX4090FP8BurnError("campaign lacks stripe balance proof")
    selected = proof.get("selected")
    if not isinstance(selected, Sequence) or len(selected) != STRIPE_COUNT:
        raise RTX4090FP8BurnError("stripe balance proof is incomplete")
    for index, raw in enumerate(stripes):
        assert isinstance(raw, Mapping)
        metrics = selected[index]
        metric_keys = {
            "qnames", "parameters", "estimated_work", "render_cells",
        }
        if not isinstance(metrics, Mapping) or set(metrics) != metric_keys:
            raise RTX4090FP8BurnError("stripe balance metrics are malformed")
        if (
            int(metrics.get("qnames", -1)) != len(raw.get("qnames", ()))
            or int(metrics.get("parameters", -1)) != int(raw.get("parameters", -2))
            or int(metrics.get("estimated_work", -1)) != int(raw.get("estimated_work", -2))
            or int(metrics.get("render_cells", -1)) != int(raw.get("render_cells", -2))
        ):
            raise RTX4090FP8BurnError("stripe balance proof differs from stripe")
    if proof.get("metric_names") != [
        "qnames", "parameters", "estimated_work", "render_cells",
    ] or proof.get("strategy") != "whole_layer_lpt_contiguous_equal_tie":
        raise RTX4090FP8BurnError("campaign stripe strategy constants differ")
    if proof.get("selected_metrics_exactly_equal") is not True:
        raise RTX4090FP8BurnError("contiguous tie is not exactly balanced")
    if proof.get("selected_matches_lpt_loads") is not True:
        raise RTX4090FP8BurnError("contiguous tie differs from LPT loads")
    if [raw.get("layer_range") for raw in stripes if isinstance(raw, Mapping)] != [
        [0, 31], [32, 63]
    ]:
        raise RTX4090FP8BurnError("contiguous tie ranges are not 0-31/32-63")
    ordinary_lpt = proof.get("ordinary_lpt")
    if not isinstance(ordinary_lpt, Sequence) or len(ordinary_lpt) != (
        STRIPE_COUNT
    ) or any(
        not isinstance(row, Mapping) or set(row) != {
            "qnames", "parameters", "estimated_work", "render_cells",
        }
        for row in ordinary_lpt
    ) or sorted(
        (dict(row) for row in ordinary_lpt if isinstance(row, Mapping)),
        key=lambda row: tuple(row.get(name, -1) for name in (
            "qnames", "parameters", "estimated_work", "render_cells",
        )),
    ) != sorted(
        (dict(row) for row in selected if isinstance(row, Mapping)),
        key=lambda row: tuple(row.get(name, -1) for name in (
            "qnames", "parameters", "estimated_work", "render_cells",
        )),
    ):
        raise RTX4090FP8BurnError("ordinary LPT proof differs from selected tie")


def write_campaign_plan(plan: Mapping[str, object], output_dir: str | Path) -> Path:
    validate_campaign_plan(plan)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for raw in plan["stripes"]:  # type: ignore[index]
        assert isinstance(raw, Mapping)
        path = root / str(raw["qname_file"])
        data = "".join(f"{name}\n" for name in raw["qnames"]).encode("utf-8")
        if hashlib.sha256(data).hexdigest() != raw["qname_file_sha256"]:
            raise RTX4090FP8BurnError("refusing inconsistent stripe qname file")
        atomic_write_bytes(path, data)
    plan_path = root / "campaign-plan.json"
    atomic_write_bytes(plan_path, json.dumps(
        plan, indent=2, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8") + b"\n")
    return plan_path


def load_campaign_plan(path: str | Path) -> dict[str, object]:
    try:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RTX4090FP8BurnError(f"campaign plan is unreadable: {path}") from exc
    if not isinstance(plan, Mapping):
        raise RTX4090FP8BurnError("campaign plan is not an object")
    result = dict(plan)
    validate_campaign_plan(result)
    return result


def _classify_source_census(
    source_census: Mapping[str, str], *, profile,
) -> tuple[tuple[str, ...], dict[str, dict[str, object]]]:
    body: list[str] = []
    fixed: dict[str, dict[str, object]] = {}
    for qname in sorted(source_census):
        source_dtype = str(source_census[qname]).lower()
        if source_dtype != "bf16":
            raise RTX4090FP8BurnError(
                f"source Linear {qname} has dtype class {source_dtype!r}; "
                "this campaign requires one exact BF16 source class"
            )
        if profile.is_pinned_name(qname):
            reason = "profile_pinned"
        elif qname.startswith("mtp.") or ".mtp." in qname:
            reason = "mtp_fixed"
        elif qname.startswith("visual.") or ".visual." in qname:
            reason = "visual_fixed"
        else:
            body.append(qname)
            continue
        fixed[qname] = {"reason": reason, "source_dtype": source_dtype}
    return tuple(body), fixed


def _validate_sample_merge_bundle(
    *,
    probe: str | Path,
    activation_cache_dir: str | Path,
    commit_path: str | Path,
) -> dict[str, object]:
    """Validate the atomic sample probe/cache publication as one input."""
    from prismaquant.sample_parallel_probe import (
        MERGE_BUNDLE_ACTIVATIONS,
        MERGE_BUNDLE_COMMIT,
        MERGE_BUNDLE_PROBE,
        validate_sample_parallel_merge_bundle,
    )

    probe_path = Path(probe).resolve(strict=True)
    cache_dir = Path(activation_cache_dir).resolve(strict=True)
    commit_file = Path(commit_path).resolve(strict=True)
    bundle = commit_file.parent
    if (
        commit_file.name != MERGE_BUNDLE_COMMIT
        or probe_path != bundle / MERGE_BUNDLE_PROBE
        or cache_dir != bundle / MERGE_BUNDLE_ACTIVATIONS
    ):
        raise RTX4090FP8BurnError(
            "probe, activation cache, and commit are not one canonical "
            "sample-merge bundle"
        )
    try:
        validated = validate_sample_parallel_merge_bundle(
            bundle, capture_consumables=True,
        )
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"sample-merge bundle validation failed: {exc}"
        ) from exc
    probe_payload = validated.get("_validated_probe_payload")
    probe_bytes = validated.get("_validated_probe_bytes")
    activation_manifest = validated.get("_validated_activation_manifest")
    if (
        not isinstance(probe_payload, Mapping)
        or not isinstance(probe_bytes, bytes)
        or not isinstance(activation_manifest, Mapping)
    ):
        raise RTX4090FP8BurnError(
            "sample-merge bundle validator did not capture consumable members"
        )
    try:
        from prismaquant.sample_parallel_probe import (
            stable_source_census_projection,
        )

        probe_meta = probe_payload.get("meta")
        merge_meta = probe_meta.get("sample_parallel_merge") if isinstance(
            probe_meta, Mapping
        ) else None
        cover = merge_meta.get("cover") if isinstance(
            merge_meta, Mapping
        ) else None
        qname_census = cover.get("qname_census") if isinstance(
            cover, Mapping
        ) else None
        execution_identity = cover.get("execution_identity") if isinstance(
            cover, Mapping
        ) else None
        if not isinstance(qname_census, Mapping) or not isinstance(
            execution_identity, Mapping
        ):
            raise TypeError("validated bundle has no execution/qname census")
        if execution_identity.get("identity_sha256") != validated.get(
            "execution_identity_sha256"
        ):
            raise TypeError("validated bundle execution identity differs")
        source_projection = stable_source_census_projection(qname_census)
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"validated sample bundle source census is unavailable: {exc}"
        ) from exc
    return {
        "commit_identity_sha256": str(validated["identity_sha256"]),
        "cover_identity_sha256": str(validated["cover_identity_sha256"]),
        "execution_identity_sha256": str(
            validated["execution_identity_sha256"]
        ),
        "probe_sha256": str(validated["probe_sha256"]),
        "probe_bytes": int(validated["probe_bytes"]),
        "activation_manifest_identity_sha256": str(validated[
            "activation_manifest_identity_sha256"
        ]),
        "source_model_content_sha256": str(
            validated["source_model_content_sha256"]
        ),
        "source_model_upstream_content_sha256": validated.get(
            "source_model_upstream_content_sha256"
        ),
        "source_model_upstream_portable_content_sha256": validated.get(
            "source_model_upstream_portable_content_sha256"
        ),
        "source_census_identity_sha256": str(
            validated["source_census_identity_sha256"]
        ),
        "_validated_execution_identity": copy.deepcopy(
            dict(execution_identity)
        ),
        "_validated_source_census_projection": source_projection,
        "_validated_probe_bytes": probe_bytes,
        "_validated_probe_payload": probe_payload,
        "_validated_activation_manifest": activation_manifest,
    }


def _validate_sample_bundle_source_binding(
    sample_bundle: Mapping[str, object],
    source_binding: Mapping[str, object],
    *,
    live_source_identity: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Join the portable sample identity to the live streamed identity.

    ``source_model_content_sha256`` is the source-census digest built from raw
    checkpoint/config metadata.  The model-content record separately carries
    a portable projection derived from a locally validated streamed-model
    identity.  That projection strips staging/model paths while preserving the
    semantic config, both checkpoint maps, and shard bytes; it is the explicit
    bridge to each burn host's independently built cache.
    """
    portable = sample_bundle.get("source_model_content_sha256")
    upstream = sample_bundle.get(
        "source_model_upstream_portable_content_sha256"
    )
    bound = source_binding.get("portable_content_sha256")
    if live_source_identity is None:
        live = bound
    else:
        try:
            from prismaquant.cost_streaming import (
                portable_streamed_model_content_identity,
            )

            live = portable_streamed_model_content_identity(
                live_source_identity,
                where="live RTX4090 streamed-model portable content",
            )["portable_content_sha256"]
        except Exception as exc:
            raise RTX4090FP8BurnError(
                f"live streamed-model portable identity is invalid: {exc}"
            ) from exc
    if not _is_sha256(portable):
        raise RTX4090FP8BurnError(
            "sample bundle lacks its portable source-model content identity"
        )
    if (
        not _is_sha256(upstream)
        or not _is_sha256(bound)
        or not _is_sha256(live)
        or upstream != bound
        or live != bound
    ):
        raise RTX4090FP8BurnError(
            "sample bundle, source identity cache, and live model content differ"
        )
    return {
        "portable_content_sha256": str(portable),
        "upstream_portable_content_sha256": str(upstream),
    }


def _validate_live_sample_source_census(
    sample_bundle: Mapping[str, object],
    *,
    model: str | Path,
    source_identity_cache: str | Path,
) -> dict[str, object]:
    """Rebuild the bundle's portable source census from this live host."""
    from prismaquant.sample_parallel_probe import (
        build_rtx4090_qname_census,
        stable_source_census_projection,
    )

    expected_projection = sample_bundle.get(
        "_validated_source_census_projection"
    )
    if not isinstance(expected_projection, Mapping):
        raise RTX4090FP8BurnError(
            "sample bundle lacks its validated stable source projection"
        )
    try:
        live_census = build_rtx4090_qname_census(
            model, identity_cache_path=source_identity_cache,
        )
        live_projection = stable_source_census_projection(live_census)
        live_model_identity = live_census["source_census"][
            "source_model_identity"
        ]
        live_portable = live_model_identity["content_sha256"]
        live_upstream_portable = live_model_identity[
            "upstream_portable_content_sha256"
        ]
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"live portable source census validation failed: {exc}"
        ) from exc
    if (
        live_portable != sample_bundle.get("source_model_content_sha256")
        or live_upstream_portable != sample_bundle.get(
            "source_model_upstream_portable_content_sha256"
        )
        or live_projection != dict(expected_projection)
    ):
        raise RTX4090FP8BurnError(
            "live portable source census differs from the validated sample "
            "bundle"
        )
    return live_census


def _validate_burn_runtime_snapshot(
    snapshot_manifest: str | Path,
) -> dict[str, object]:
    """Prove this burn command executes from the bound immutable snapshot."""
    from prismaquant.sample_parallel_probe import (
        validate_local_producer_snapshot,
    )

    manifest_path = Path(snapshot_manifest)
    if (
        manifest_path.name != ".prismaquant-runtime-snapshot.json"
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise RTX4090FP8BurnError(
            "producer snapshot input must be its exact regular manifest"
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError("manifest is not a mapping")
        verified = validate_local_producer_snapshot(
            manifest_path.parent,
            expected_closure_sha256=str(raw["closure_sha256"]),
            expected_commit=str(raw["commit"]),
            expected_tree=str(raw["tree"]),
            require_current_module_inside=True,
        )
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"burn runtime snapshot verification failed: {exc}"
        ) from exc
    return verified


def _revalidate_live_campaign_census(
    *,
    model: str | Path,
    probe: str | Path,
    plan: Mapping[str, object],
    profile,
    validated_probe_payload: Mapping[str, object] | None = None,
) -> None:
    """Re-derive every source/probe census fact a self-hashed plan carries."""
    from prismaquant.allocator_candidates import _scan_source_dtype_manifest

    source_census = _scan_source_dtype_manifest(str(model), profile)
    body, fixed = _classify_source_census(source_census, profile=profile)
    probe_stats, _probe_meta = (
        _normalized_probe_payload(validated_probe_payload)
        if validated_probe_payload is not None
        else _probe_payload(probe)
    )
    missing_probe = sorted(set(body) - set(probe_stats))
    extra_probe = sorted(set(probe_stats) - set(body) - set(fixed))
    if missing_probe or extra_probe:
        raise RTX4090FP8BurnError(
            "live probe/source census mismatch: "
            f"missing body={missing_probe[:8]}, extra={extra_probe[:8]}"
        )
    for name in fixed:
        fixed[name]["n_params"] = int(
            probe_stats.get(name, {}).get("n_params", 0) or 0
        )
    expected_fixed = _fixed_census_records(fixed)
    expected_shapes = {
        name: [
            int(probe_stats[name].get("out_features", 0) or 0),
            int(probe_stats[name].get("in_features", 0) or 0),
        ]
        for name in sorted(body)
    }
    expected_parameters = sum(
        int(probe_stats[name].get("n_params", 0) or 0) for name in body
    )
    plan_body = plan.get("body")
    census_digest = canonical_json_sha256(
        dict(sorted((str(key), str(value)) for key, value in source_census.items())),
        where="live source dtype census",
    )
    if (
        not isinstance(plan_body, Mapping)
        or list(plan_body.get("qnames", ())) != list(sorted(body))
        or plan_body.get("shapes") != expected_shapes
        or int(plan_body.get("qname_count", -1)) != len(body)
        or int(plan_body.get("parameters", -1)) != expected_parameters
        or plan.get("fixed_bf16_census") != expected_fixed
        or plan.get("fixed_bf16_census_sha256") != canonical_json_sha256(
            expected_fixed, where="live RTX4090 fixed BF16 census",
        )
        or plan.get("source_dtype_census_sha256") != census_digest
    ):
        raise RTX4090FP8BurnError(
            "loaded campaign plan differs from the live source/probe census"
        )


def attest_execution(args: argparse.Namespace) -> Path:
    """Create one no-clobber burn attestation from the sample run-contract."""
    from prismaquant.sample_parallel_probe import (
        _atomic_write_bytes_no_clobber,
        _strict_json_loads,
        validate_run_contract,
    )

    producer_snapshot = _validate_burn_runtime_snapshot(
        args.producer_snapshot
    )
    run_contract_path = Path(args.sample_run_contract)
    if run_contract_path.is_symlink() or not run_contract_path.is_file():
        raise RTX4090FP8BurnError(
            "sample run-contract must be one regular JSON file"
        )
    try:
        raw = _strict_json_loads(
            run_contract_path.read_text(encoding="utf-8"),
            where="sample run-contract for RTX4090 burn",
        )
        if not isinstance(raw, Mapping):
            raise TypeError("run-contract is not a mapping")
        run_contract = validate_run_contract(raw)
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"sample run-contract is invalid: {exc}"
        ) from exc
    execution_identity = run_contract["execution_identity"]
    if not isinstance(execution_identity, Mapping):
        raise RTX4090FP8BurnError(
            "sample run-contract execution identity is malformed"
        )
    attestation = build_execution_attestation(
        execution_identity,
        producer_snapshot=producer_snapshot,
        launcher_image_digest=args.launcher_image_digest,
    )
    output = Path(args.output)
    try:
        _atomic_write_bytes_no_clobber(
            output,
            json.dumps(attestation, sort_keys=True, indent=2).encode("utf-8")
            + b"\n",
        )
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"execution attestation publication failed: {exc}"
        ) from exc
    return output


def prepare(args: argparse.Namespace) -> Path:
    from prismaquant.allocator_candidates import _scan_source_dtype_manifest
    from prismaquant.model_profiles import detect_profile

    producer_snapshot = _validate_burn_runtime_snapshot(
        args.producer_snapshot
    )
    profile = detect_profile(args.model)
    if profile.name not in {"qwen3_5_dense", "qwen3_5"}:
        raise RTX4090FP8BurnError(
            f"model profile {profile.name!r} is not the dense Qwen source"
        )
    sample_bundle = _validate_sample_merge_bundle(
        probe=args.probe,
        activation_cache_dir=args.activation_cache_dir,
        commit_path=args.sample_merge_commit,
    )
    execution_identity = sample_bundle["_validated_execution_identity"]
    if not isinstance(execution_identity, Mapping):
        raise RTX4090FP8BurnError(
            "sample-merge execution identity is malformed"
        )
    _validate_execution_attestation(
        args.execution_attestation,
        execution_identity=execution_identity,
        producer_snapshot=producer_snapshot,
        launcher_image_digest=args.launcher_image_digest,
    )
    source_identity_binding, live_source_identity = _source_identity_binding(
        args.model, args.source_identity,
    )
    _validate_sample_bundle_source_binding(
        sample_bundle,
        source_identity_binding,
        live_source_identity=live_source_identity,
    )
    _validate_live_sample_source_census(
        sample_bundle,
        model=args.model,
        source_identity_cache=args.source_identity,
    )
    validated_probe_payload = sample_bundle["_validated_probe_payload"]
    assert isinstance(validated_probe_payload, Mapping)
    probe_stats, probe_meta = _normalized_probe_payload(
        validated_probe_payload
    )
    source_census = _scan_source_dtype_manifest(args.model, profile)
    body, fixed = _classify_source_census(source_census, profile=profile)
    missing_probe = sorted(set(body) - set(probe_stats))
    extra_probe = sorted(set(probe_stats) - set(body) - set(fixed))
    if missing_probe or extra_probe:
        raise RTX4090FP8BurnError(
            "probe/source census mismatch: "
            f"missing body={missing_probe[:8]}, extra={extra_probe[:8]}"
        )
    body_stats = {name: probe_stats[name] for name in body}
    for name in fixed:
        if name in probe_stats:
            fixed[name]["n_params"] = int(probe_stats[name].get("n_params", 0) or 0)
        else:
            fixed[name]["n_params"] = 0
    calibration = _calibration_contract(
        probe_meta, nsamples=args.n_calib_samples,
        seqlen=args.calib_seqlen, seed=args.calib_seed,
    )
    imatrix_contract = _probe_imatrix_contract(
        args.probe, args.col_weights,
        validated_probe_payload=validated_probe_payload,
    )
    if imatrix_contract["calibration_hash"] != calibration["calib_hash"]:
        raise RTX4090FP8BurnError(
            "probe-derived imatrix calibration differs from burn calibration"
        )
    bindings = {
        "probe": _binding(args.probe, where="probe"),
        "col_weights": _binding(args.col_weights, where="column weights"),
        "source_model_identity": source_identity_binding,
        "producer_snapshot": _binding(
            args.producer_snapshot, where="producer source snapshot"
        ),
        "common_execution_attestation": _binding(
            args.execution_attestation, where="common execution attestation"
        ),
        "dataset": _binding(args.dataset, where="calibration dataset"),
        "sample_merge_commit": _binding(
            args.sample_merge_commit, where="sample merge bundle commit",
        ),
        "activation_cache_manifest": _binding(
            Path(args.activation_cache_dir) / "sample_parallel_merge.json",
            where="sample merge activation manifest",
        ),
    }
    if (
        bindings["probe"].get("sha256") != sample_bundle["probe_sha256"]
        or bindings["probe"].get("bytes") != sample_bundle["probe_bytes"]
        or bindings["activation_cache_manifest"].get("identity_sha256")
        != sample_bundle["activation_manifest_identity_sha256"]
    ):
        raise RTX4090FP8BurnError(
            "sample-merge members changed between validation and plan binding"
        )
    census_digest = canonical_json_sha256(
        dict(sorted((str(k), str(v)) for k, v in source_census.items())),
        where="source dtype census",
    )
    plan = build_campaign_plan(
        body_stats, profile=profile, fixed_bf16=fixed,
        calibration=calibration, bindings=bindings,
        source_dtype_census_sha256=census_digest,
        imatrix_contract=imatrix_contract,
    )
    return write_campaign_plan(plan, args.output_dir)


def _verify_binding(plan: Mapping[str, object], name: str, path: str | Path) -> None:
    bindings = plan.get("bindings")
    expected = bindings.get(name) if isinstance(bindings, Mapping) else None
    if not isinstance(expected, Mapping):
        raise RTX4090FP8BurnError(f"plan has no {name} binding")
    observed = _binding(path, where=name)
    if observed != dict(expected):
        raise RTX4090FP8BurnError(f"{name} differs from the prepared plan")


def _cb_context():
    from prismaquant.nvfp4_cb_footprint import CBSerializationContext

    return CBSerializationContext(
        scale_coding="v1", codebook_source="lattice",
        codebook_source_scope="none", scale_sweep=True,
        scale_sweep_scope="fp8", ldlq=False, ldlq_scope="none",
        minchain=False, encode_tier="balanced", activation_contract=None,
        activation_execution=None,
    )


def _arm_identity(plan: Mapping[str, object]) -> dict[str, object]:
    bindings = plan["bindings"]
    assert isinstance(bindings, Mapping)
    producer_snapshot = bindings["producer_snapshot"]
    execution = bindings["common_execution_attestation"]
    assert isinstance(producer_snapshot, Mapping)
    assert isinstance(execution, Mapping)
    return {
        "campaign_schema": PLAN_SCHEMA,
        "global_plan_sha256": plan["plan_sha256"],
        "producer_settings": dict(CB_PRODUCER_SETTINGS),
        "render_levers": dict(RENDER_LEVERS),
        "producer_snapshot_sha256": producer_snapshot["sha256"],
        "common_execution_attestation_sha256": execution["sha256"],
        "container_image_digest": execution["container_image_digest"],
        "compile_settings": {
            "PRISMAQUANT_CB_ENCODE_COMPILE": "1",
            "PRISMAQUANT_CB_ATOM_COMPILE": "1",
            CB_COMPILE_FAIL_CLOSED_ENV: "1",
        },
        "sparse_anchor_measurement": True,
    }


def _require_compile_settings() -> dict[str, str]:
    names = (
        "PRISMAQUANT_CB_ENCODE_COMPILE",
        "PRISMAQUANT_CB_ATOM_COMPILE",
        CB_COMPILE_FAIL_CLOSED_ENV,
    )
    resolved = {name: str(os.environ.get(name, "")).strip() for name in names}
    disabled = [name for name, value in resolved.items() if value != "1"]
    if disabled:
        raise RTX4090FP8BurnError(
            "strict CB compilation is not enabled: "
            f"set {', '.join(disabled)}=1 before measuring"
        )
    return resolved


def _campaign_shard_receipt_body(
    payload: Mapping[str, object],
    plan: Mapping[str, object],
    *,
    stripe_index: int,
    compile_settings: Mapping[str, str],
    model_identity: Mapping[str, object],
) -> dict[str, object]:
    """Recompute the immutable campaign/stripe binding for one raw shard."""
    validate_campaign_plan(plan)
    if stripe_index not in range(STRIPE_COUNT):
        raise RTX4090FP8BurnError("shard receipt stripe index is invalid")
    stripe = plan["stripes"][stripe_index]  # type: ignore[index]
    assert isinstance(stripe, Mapping)
    costs = payload.get("costs")
    stats = payload.get("stats")
    provenance = payload.get("provenance")
    if not isinstance(costs, Mapping) or not isinstance(
        stats, Mapping
    ) or not isinstance(provenance, Mapping):
        raise RTX4090FP8BurnError(
            "raw shard lacks costs/stats/provenance mappings"
        )
    if int(payload.get("n_probes", -1)) != AURA_N_PROBES:
        raise RTX4090FP8BurnError(
            "raw shard AURA probe count differs from campaign producer"
        )
    if payload.get("token_scope") != AURA_TOKEN_SCOPE:
        raise RTX4090FP8BurnError(
            "raw shard token scope differs from campaign producer"
        )
    expected_scope = tuple(str(name) for name in stripe["qnames"])
    if set(map(str, costs)) != set(expected_scope):
        raise RTX4090FP8BurnError(
            f"raw shard cost scope is not exact stripe {stripe_index} ownership"
        )
    if set(map(str, stats)) != set(expected_scope):
        raise RTX4090FP8BurnError(
            f"raw shard stats scope is not exact stripe {stripe_index} ownership"
        )
    renderer = provenance.get("production_anchor_renderer")
    if not isinstance(renderer, Mapping):
        raise RTX4090FP8BurnError("raw shard lacks production renderer identity")
    expected_arm = _arm_identity(plan)
    if renderer.get("arm_identity") != expected_arm:
        raise RTX4090FP8BurnError(
            "raw shard production arm differs from loaded campaign plan"
        )
    expected_compile = expected_arm["compile_settings"]
    if dict(compile_settings) != expected_compile:
        raise RTX4090FP8BurnError(
            "raw shard compile settings differ from loaded campaign plan"
        )
    compile_proof = provenance.get(CB_COMPILE_PROOF_KEY)
    try:
        validated_compile_proof = validate_campaign_cb_compile_proof(
            compile_proof,  # type: ignore[arg-type]
            expected_compile_settings=expected_compile,
            expected_qnames=expected_scope,
            formats_per_unit=len(MEASURED_FORMATS),
            cb_formats_per_unit=len(MEASURED_CB_FORMATS),
        )
    except (RTX4090CBCompileProofError, TypeError) as exc:
        raise RTX4090FP8BurnError(
            f"raw shard CB compile execution proof is invalid: {exc}"
        ) from exc
    proof_coverage = validated_compile_proof["coverage"]
    assert isinstance(proof_coverage, Mapping)
    if (
        int(provenance.get("production_anchor_expected_renders", -1))
        != proof_coverage["expected_rendered_cells"]
        or int(provenance.get(
            "production_anchor_rendered_this_invocation", -1
        )) != proof_coverage["live_rendered_cells"]
        or int(provenance.get("production_anchor_restored_renders", -1))
        != proof_coverage["restored_rendered_cells"]
    ):
        raise RTX4090FP8BurnError(
            "raw shard render counters differ from its CB compile proof"
        )
    renderer_model = renderer.get("source_model")
    if not isinstance(renderer_model, Mapping) or dict(renderer_model) != dict(
        model_identity
    ):
        raise RTX4090FP8BurnError(
            "raw shard live model identity differs from renderer identity"
        )
    bindings = plan["bindings"]
    assert isinstance(bindings, Mapping)
    source_binding = bindings["source_model_identity"]
    producer_binding = bindings["producer_snapshot"]
    execution_binding = bindings["common_execution_attestation"]
    assert isinstance(source_binding, Mapping)
    assert isinstance(producer_binding, Mapping)
    assert isinstance(execution_binding, Mapping)
    try:
        from prismaquant.cost_streaming import (
            portable_streamed_model_content_identity,
        )

        live_portable_content = portable_streamed_model_content_identity(
            model_identity,
            where=f"RTX4090 stripe {stripe_index} live portable content",
        )["portable_content_sha256"]
    except Exception as exc:
        raise RTX4090FP8BurnError(
            f"raw shard live model portable identity is invalid: {exc}"
        ) from exc
    if live_portable_content != source_binding.get(
        "portable_content_sha256"
    ):
        raise RTX4090FP8BurnError(
            "raw shard live model content differs from source-model binding"
        )
    return {
        "schema": SHARD_RECEIPT_SCHEMA,
        "campaign_schema": PLAN_SCHEMA,
        "global_plan_sha256": plan["plan_sha256"],
        "stripe_index": stripe_index,
        "stripe_record_sha256": canonical_json_sha256(
            stripe, where=f"RTX4090 stripe {stripe_index} plan record",
        ),
        "stripe_qname_file_sha256": stripe["qname_file_sha256"],
        "stripe_qname_count": len(expected_scope),
        "n_probes": AURA_N_PROBES,
        "token_scope": AURA_TOKEN_SCOPE,
        "fixed_bf16_census_sha256": plan["fixed_bf16_census_sha256"],
        "producer_snapshot_sha256": producer_binding["sha256"],
        "common_execution_attestation_sha256": execution_binding["sha256"],
        "container_image_digest": execution_binding[
            "container_image_digest"
        ],
        "compile_settings": dict(compile_settings),
        "cb_compile_execution_proof": copy.deepcopy(
            validated_compile_proof
        ),
        "arm_identity_sha256": canonical_json_sha256(
            expected_arm, where="RTX4090 shard production arm",
        ),
        "source_model_identity_binding": copy.deepcopy(dict(source_binding)),
        "live_streamed_model_portable_content_sha256": live_portable_content,
        "renderer_identity_sha256": canonical_json_sha256(
            renderer, where="RTX4090 shard renderer identity",
        ),
        "measured_costs_sha256": canonical_json_sha256(
            costs, where=f"RTX4090 stripe {stripe_index} measured costs",
        ),
        "measured_stats_sha256": canonical_json_sha256(
            stats, where=f"RTX4090 stripe {stripe_index} measured stats",
        ),
    }


def _attach_campaign_shard_receipt(
    payload: dict[str, object],
    plan: Mapping[str, object],
    *,
    stripe_index: int,
    compile_settings: Mapping[str, str],
    model_identity: Mapping[str, object],
) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise RTX4090FP8BurnError("raw shard provenance is not mutable")
    if SHARD_RECEIPT_KEY in provenance:
        raise RTX4090FP8BurnError("raw shard already carries a campaign receipt")
    body = _campaign_shard_receipt_body(
        payload, plan, stripe_index=stripe_index,
        compile_settings=compile_settings, model_identity=model_identity,
    )
    provenance[SHARD_RECEIPT_KEY] = {
        **body,
        "receipt_sha256": canonical_json_sha256(
            body, where="RTX4090 campaign shard receipt",
        ),
    }


def _validate_campaign_shard_receipt(
    payload: Mapping[str, object], plan: Mapping[str, object],
) -> int:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RTX4090FP8BurnError("raw shard lacks provenance")
    receipt = provenance.get(SHARD_RECEIPT_KEY)
    if not isinstance(receipt, Mapping):
        raise RTX4090FP8BurnError("raw shard lacks RTX4090 campaign receipt")
    try:
        stripe_index = int(receipt["stripe_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RTX4090FP8BurnError(
            "raw shard campaign receipt stripe index is malformed"
        ) from exc
    renderer = provenance.get("production_anchor_renderer")
    renderer_model = (
        renderer.get("source_model") if isinstance(renderer, Mapping) else None
    )
    if not isinstance(renderer_model, Mapping):
        raise RTX4090FP8BurnError(
            "raw shard renderer lacks live source-model identity"
        )
    expected_body = _campaign_shard_receipt_body(
        payload, plan, stripe_index=stripe_index,
        compile_settings=_arm_identity(plan)["compile_settings"],
        model_identity=renderer_model,
    )
    observed_body = dict(receipt)
    observed_checksum = observed_body.pop("receipt_sha256", None)
    if observed_body != expected_body or observed_checksum != (
        canonical_json_sha256(
            expected_body, where="RTX4090 campaign shard receipt",
        )
    ):
        raise RTX4090FP8BurnError(
            "raw shard campaign receipt differs from loaded plan/payload"
        )
    return stripe_index


def _validate_campaign_shards(
    payloads: Sequence[Mapping[str, object]], plan: Mapping[str, object],
) -> None:
    if len(payloads) != STRIPE_COUNT:
        raise RTX4090FP8BurnError("campaign requires exactly two raw shards")
    indices = [_validate_campaign_shard_receipt(payload, plan) for payload in payloads]
    if sorted(indices) != list(range(STRIPE_COUNT)):
        raise RTX4090FP8BurnError(
            "raw shards are not one exact receipt-bound cover of both stripes"
        )


def _validate_merged_input_receipts(
    receipts: object,
    *,
    costs: Mapping[str, object],
    stats: Mapping[str, object],
    plan: Mapping[str, object],
) -> list[dict[str, object]]:
    """Replay both raw receipts against the merged values they authorize."""
    if not isinstance(receipts, Sequence) or isinstance(
        receipts, (str, bytes)
    ) or len(receipts) != STRIPE_COUNT:
        raise RTX4090FP8BurnError(
            "merged AURA payload lacks both input shard receipts"
        )
    normalized: list[dict[str, object]] = []
    seen: set[int] = set()
    for raw in receipts:
        if not isinstance(raw, Mapping) or set(raw) != (
            _SHARD_RECEIPT_BODY_KEYS | {"receipt_sha256"}
        ):
            raise RTX4090FP8BurnError(
                "merged input shard receipt shape is not closed"
            )
        row = dict(raw)
        receipt_digest = row.pop("receipt_sha256")
        if receipt_digest != canonical_json_sha256(
            row, where="merged RTX4090 input shard receipt",
        ):
            raise RTX4090FP8BurnError(
                "merged input shard receipt checksum differs"
            )
        try:
            index = int(row["stripe_index"])
        except (TypeError, ValueError) as exc:
            raise RTX4090FP8BurnError(
                "merged input shard stripe index is malformed"
            ) from exc
        if type(row["stripe_index"]) is not int or index not in range(
            STRIPE_COUNT
        ) or index in seen:
            raise RTX4090FP8BurnError(
                "merged input shard receipt stripe cover differs"
            )
        seen.add(index)
        stripe = plan["stripes"][index]  # type: ignore[index]
        assert isinstance(stripe, Mapping)
        qnames = tuple(str(name) for name in stripe["qnames"])
        try:
            normalized_compile_proof = validate_campaign_cb_compile_proof(
                row["cb_compile_execution_proof"],  # type: ignore[arg-type]
                expected_compile_settings=_arm_identity(plan)[
                    "compile_settings"
                ],
                expected_qnames=qnames,
                formats_per_unit=len(MEASURED_FORMATS),
                cb_formats_per_unit=len(MEASURED_CB_FORMATS),
            )
        except (KeyError, RTX4090CBCompileProofError, TypeError) as exc:
            raise RTX4090FP8BurnError(
                f"merged input shard receipt {index} has invalid CB compile "
                f"execution proof: {exc}"
            ) from exc
        if row["cb_compile_execution_proof"] != normalized_compile_proof:
            raise RTX4090FP8BurnError(
                f"merged input shard receipt {index} CB compile proof is not "
                "canonical"
            )
        cost_subset = {name: costs[name] for name in qnames}
        stats_subset = {name: stats[name] for name in qnames}
        bindings = plan["bindings"]
        assert isinstance(bindings, Mapping)
        if (
            row["schema"] != SHARD_RECEIPT_SCHEMA
            or row["campaign_schema"] != PLAN_SCHEMA
            or row["global_plan_sha256"] != plan["plan_sha256"]
            or row["stripe_record_sha256"] != canonical_json_sha256(
                stripe, where=f"merged RTX4090 stripe {index} record",
            )
            or row["stripe_qname_file_sha256"]
            != stripe["qname_file_sha256"]
            or row["stripe_qname_count"] != len(qnames)
            or row["n_probes"] != AURA_N_PROBES
            or row["token_scope"] != AURA_TOKEN_SCOPE
            or row["fixed_bf16_census_sha256"]
            != plan["fixed_bf16_census_sha256"]
            or row["producer_snapshot_sha256"]
            != bindings["producer_snapshot"]["sha256"]
            or row["common_execution_attestation_sha256"]
            != bindings["common_execution_attestation"]["sha256"]
            or row["container_image_digest"]
            != bindings["common_execution_attestation"][
                "container_image_digest"
            ]
            or row["compile_settings"]
            != _arm_identity(plan)["compile_settings"]
            or row["source_model_identity_binding"]
            != bindings["source_model_identity"]
            or row["live_streamed_model_portable_content_sha256"]
            != bindings["source_model_identity"]["portable_content_sha256"]
            or row["measured_costs_sha256"] != canonical_json_sha256(
                cost_subset, where=f"merged RTX4090 stripe {index} costs",
            )
            or row["measured_stats_sha256"] != canonical_json_sha256(
                stats_subset, where=f"merged RTX4090 stripe {index} stats",
            )
        ):
            raise RTX4090FP8BurnError(
                f"merged input shard receipt {index} differs from payload/plan"
            )
        normalized.append({**row, "receipt_sha256": receipt_digest})
    if seen != set(range(STRIPE_COUNT)):
        raise RTX4090FP8BurnError(
            "merged input shard receipts are not an exact stripe cover"
        )
    return sorted(normalized, key=lambda item: int(item["stripe_index"]))


def _validate_merged_campaign_provenance(
    merged: Mapping[str, object], plan: Mapping[str, object],
) -> dict[str, object]:
    costs = merged.get("costs")
    stats = merged.get("stats")
    provenance = merged.get("provenance")
    burn = provenance.get("rtx4090_fp8_burn") if isinstance(
        provenance, Mapping
    ) else None
    if not isinstance(costs, Mapping) or not isinstance(
        stats, Mapping
    ) or not isinstance(burn, Mapping):
        raise RTX4090FP8BurnError(
            "merged AURA payload lacks receipt-bound costs/stats/provenance"
        )
    expected_keys = {
        "schema", "global_plan_sha256", "fixed_bf16_census_sha256",
        "producer_snapshot_sha256", "common_execution_attestation_sha256",
        "container_image_digest",
        "direct_measured_formats", "unmeasured_terminal",
        "input_shard_receipt_schema", "input_shard_receipts",
        "merged_costs_sha256", "merged_stats_sha256", "receipt_sha256",
    }
    if set(burn) != expected_keys:
        raise RTX4090FP8BurnError(
            "merged RTX4090 provenance shape is not closed"
        )
    body = dict(burn)
    receipt_digest = body.pop("receipt_sha256")
    if receipt_digest != canonical_json_sha256(
        body, where="merged RTX4090 provenance receipt",
    ):
        raise RTX4090FP8BurnError("merged RTX4090 receipt checksum differs")
    receipts = _validate_merged_input_receipts(
        body["input_shard_receipts"], costs=costs, stats=stats, plan=plan,
    )
    bindings = plan["bindings"]
    assert isinstance(bindings, Mapping)
    if (
        body["schema"] != MERGED_SCHEMA
        or body["global_plan_sha256"] != plan["plan_sha256"]
        or body["fixed_bf16_census_sha256"]
        != plan["fixed_bf16_census_sha256"]
        or body["producer_snapshot_sha256"]
        != bindings["producer_snapshot"]["sha256"]
        or body["common_execution_attestation_sha256"]
        != bindings["common_execution_attestation"]["sha256"]
        or body["container_image_digest"]
        != bindings["common_execution_attestation"][
            "container_image_digest"
        ]
        or body["direct_measured_formats"] != list(MEASURED_FORMATS)
        or body["unmeasured_terminal"] != BF16_FORMAT
        or body["input_shard_receipt_schema"] != SHARD_RECEIPT_SCHEMA
        or body["input_shard_receipts"] != receipts
        or body["merged_costs_sha256"] != canonical_json_sha256(
            costs, where="merged RTX4090 costs",
        )
        or body["merged_stats_sha256"] != canonical_json_sha256(
            stats, where="merged RTX4090 stats",
        )
    ):
        raise RTX4090FP8BurnError(
            "merged RTX4090 provenance differs from current payload/plan"
        )
    return {**body, "receipt_sha256": receipt_digest}


def measure(args: argparse.Namespace) -> Path:
    """Run exactly one stripe; this is the only GPU-bearing entry point."""
    plan = load_campaign_plan(args.plan)
    producer_snapshot = _validate_burn_runtime_snapshot(
        args.producer_snapshot
    )
    compile_settings = _require_compile_settings()
    stripe_index = int(args.stripe)
    if stripe_index not in range(STRIPE_COUNT):
        raise RTX4090FP8BurnError("stripe must be 0 or 1")
    if int(args.n_probes) != AURA_N_PROBES:
        raise RTX4090FP8BurnError(
            f"strict campaign requires exactly {AURA_N_PROBES} AURA probes"
        )
    for name, path in (
        ("probe", args.probe), ("col_weights", args.col_weights),
        ("producer_snapshot", args.producer_snapshot),
        ("common_execution_attestation", args.execution_attestation),
        ("dataset", args.dataset),
        ("sample_merge_commit", args.sample_merge_commit),
        (
            "activation_cache_manifest",
            Path(args.activation_cache_dir) / "sample_parallel_merge.json",
        ),
    ):
        _verify_binding(plan, name, path)

    live_source_identity = _verify_source_identity_binding(
        plan, model=args.model, cache_path=args.source_identity,
    )

    # Validate the complete sample bundle and the launcher-owned image value
    # before importing the GPU hot-path modules or requesting a CUDA device.
    sample_bundle = _validate_sample_merge_bundle(
        probe=args.probe,
        activation_cache_dir=args.activation_cache_dir,
        commit_path=args.sample_merge_commit,
    )
    execution_identity = sample_bundle["_validated_execution_identity"]
    if not isinstance(execution_identity, Mapping):
        raise RTX4090FP8BurnError(
            "sample-merge execution identity is malformed"
        )
    _validate_execution_attestation(
        args.execution_attestation,
        execution_identity=execution_identity,
        producer_snapshot=producer_snapshot,
        launcher_image_digest=args.launcher_image_digest,
    )

    from prismaquant.build_production_cache import _load_col_weights
    from prismaquant.cost_streaming import (
        build_streamed_causal_lm, build_streamed_model_identity,
    )
    from prismaquant.gpu_guard import require_cuda_hot_path
    from prismaquant.measure_quant_cost import ActivationIndex
    from prismaquant.model_profiles import detect_profile
    from prismaquant.perturbed_x_cache import calibration_data_hash
    from prismaquant.sensitivity_probe import load_calibration
    import torch
    from transformers import AutoTokenizer

    profile = detect_profile(args.model)
    expected_probe_binding = plan["bindings"]["probe"]
    expected_activation_binding = plan["bindings"][
        "activation_cache_manifest"
    ]
    if (
        not isinstance(expected_probe_binding, Mapping)
        or not isinstance(expected_activation_binding, Mapping)
        or sample_bundle["probe_sha256"]
        != expected_probe_binding.get("sha256")
        or sample_bundle["probe_bytes"]
        != expected_probe_binding.get("bytes")
        or sample_bundle["activation_manifest_identity_sha256"]
        != expected_activation_binding.get("identity_sha256")
    ):
        raise RTX4090FP8BurnError(
            "validated sample-merge members differ from the campaign plan"
        )
    validated_probe_payload = sample_bundle["_validated_probe_payload"]
    assert isinstance(validated_probe_payload, Mapping)
    probe_stats, probe_meta = _normalized_probe_payload(
        validated_probe_payload
    )
    source_binding = plan["bindings"]["source_model_identity"]
    if not isinstance(source_binding, Mapping):
        raise RTX4090FP8BurnError("campaign source binding is malformed")
    _validate_sample_bundle_source_binding(
        sample_bundle,
        source_binding,
        live_source_identity=live_source_identity,
    )
    _validate_live_sample_source_census(
        sample_bundle,
        model=args.model,
        source_identity_cache=args.source_identity,
    )
    _revalidate_live_campaign_census(
        model=args.model, probe=args.probe, plan=plan, profile=profile,
        validated_probe_payload=validated_probe_payload,
    )
    if _probe_imatrix_contract(
        args.probe,
        args.col_weights,
        validated_probe_payload=validated_probe_payload,
    ) != plan["imatrix"]:
        raise RTX4090FP8BurnError(
            "live probe-derived imatrix differs from campaign plan"
        )
    calibration_contract = plan["calibration"]
    assert isinstance(calibration_contract, Mapping)
    if _calibration_contract(
        probe_meta, nsamples=int(calibration_contract["nsamples"]),
        seqlen=int(calibration_contract["seqlen"]),
        seed=int(calibration_contract["seed"]),
    ) != dict(calibration_contract):
        raise RTX4090FP8BurnError("probe calibration identity changed")
    device = require_cuda_hot_path("rtx4090_fp8_burn", "cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
        local_files_only=Path(args.model).exists(),
    )
    calibration = load_calibration(
        tokenizer, args.dataset, int(calibration_contract["nsamples"]),
        int(calibration_contract["seqlen"]),
        calib_seed=int(calibration_contract["seed"]),
    ).to(device)
    observed_hash = calibration_data_hash(calibration)
    if observed_hash != calibration_contract["calib_hash"]:
        raise RTX4090FP8BurnError(
            "tokenized calibration differs from the prepared probe"
        )
    col_weights = _load_col_weights(args.col_weights, CB_FORMATS)
    if not isinstance(col_weights, Mapping):
        raise RTX4090FP8BurnError("column weights did not load")
    body_qnames = tuple(plan["body"]["qnames"])  # type: ignore[index]
    missing_col = sorted(set(body_qnames) - set(col_weights))
    if missing_col:
        raise RTX4090FP8BurnError(
            f"column weights miss body units: {missing_col[:8]}"
        )
    stripe = plan["stripes"][stripe_index]  # type: ignore[index]
    stripe_qnames = tuple(stripe["qnames"])
    stripe_stats = {name: probe_stats[name] for name in stripe_qnames}
    activation_manifest = sample_bundle["_validated_activation_manifest"]
    assert isinstance(activation_manifest, Mapping)
    activation_index = ActivationIndex(
        Path(args.activation_cache_dir),
        stripe_stats,
        verification_contract=activation_manifest,
    )
    missing_act = [name for name in stripe_qnames if name not in activation_index]
    if missing_act:
        raise RTX4090FP8BurnError(
            f"activation cache misses stripe units: {missing_act[:8]}"
        )
    maps = plan["maps"]
    formats = {name: tuple(maps["formats_by_qname"][name])
               for name in stripe_qnames}
    purposes = {name: maps["purposes_by_qname"][name]
                for name in stripe_qnames}
    legal = {name: tuple(maps["legal_cb_formats_by_qname"][name])
             for name in stripe_qnames}
    checkpoint_root = Path(args.checkpoint_dir)
    runner = build_streamed_causal_lm(
        args.model, device=device, dtype=torch.bfloat16,
        offload_folder=str(checkpoint_root / "streamed-model-offload"),
        profile=profile, max_cache_slots=STREAMING_CACHE_MAX_SLOTS,
        prefetch_lookahead=STREAMING_PREFETCH_LOOKAHEAD,
        require_prefetched_residency=(
            STREAMING_REQUIRE_PREFETCHED_RESIDENCY
        ),
    )
    if (
        runner.context.max_cache_slots != STREAMING_CACHE_MAX_SLOTS
        or runner.prefetch_lookahead != STREAMING_PREFETCH_LOOKAHEAD
        or runner.require_prefetched_residency
        is not STREAMING_REQUIRE_PREFETCHED_RESIDENCY
    ):
        runner.shutdown()
        raise RTX4090FP8BurnError(
            "streamed source cache execution differs from campaign contract"
        )
    try:
        model_identity = build_streamed_model_identity(
            runner, args.model,
            identity_cache_path=checkpoint_root / "streamed_model_identity.json",
        )
        source_binding = plan["bindings"]["source_model_identity"]  # type: ignore[index]
        from prismaquant.cost_streaming import (
            portable_streamed_model_content_identity,
        )

        live_portable_content = portable_streamed_model_content_identity(
            model_identity, where="live streamed burn model portable content",
        )["portable_content_sha256"]
        if live_portable_content != source_binding.get(
            "portable_content_sha256"
        ):
            raise RTX4090FP8BurnError(
                "live streamed source identity differs from prepared source"
            )
        arm_identity = _arm_identity(plan)
        if arm_identity["compile_settings"] != compile_settings:
            raise RTX4090FP8BurnError(
                "resolved compile switches differ from the campaign stamp"
            )
        checkpoint_identity_extra = {
            "campaign_schema": PLAN_SCHEMA,
            "global_plan_sha256": plan["plan_sha256"],
            "stripe_index": stripe_index,
            "stripe_qname_file_sha256": stripe["qname_file_sha256"],
            "fixed_bf16_census_sha256": plan["fixed_bf16_census_sha256"],
            "compile_settings": compile_settings,
            "streaming_source_cache": {
                "max_cache_slots": STREAMING_CACHE_MAX_SLOTS,
                "effective_prefetch_lookahead": (
                    STREAMING_PREFETCH_LOOKAHEAD
                ),
                "require_prefetched_residency": (
                    STREAMING_REQUIRE_PREFETCHED_RESIDENCY
                ),
            },
        }
        compile_proof_token = begin_cb_compile_execution_proof()
        try:
            payload = run_streamed_cb_anchor_aura(
                runner, calibration, formats_by_qname=formats,
                legal_formats_by_qname=legal, purposes_by_qname=purposes,
                activation_index=activation_index, render_levers=RENDER_LEVERS,
                col_weights=col_weights, cb_serialization_context=_cb_context(),
                calibration_hash=observed_hash, arm_identity=arm_identity,
                model_identity=model_identity,
                checkpoint_dir=checkpoint_root / "aura",
                resume=bool(args.resume),
                n_probes=int(args.n_probes), profile=profile,
                checkpoint_identity_extra=checkpoint_identity_extra,
            )
            live_compile_proof = finish_cb_compile_execution_proof(
                compile_proof_token
            )
        except Exception:
            abort_cb_compile_execution_proof(compile_proof_token)
            raise
    finally:
        runner.shutdown()
    if not isinstance(payload, dict):
        raise RTX4090FP8BurnError("streamed AURA returned a non-mutable payload")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise RTX4090FP8BurnError("streamed AURA returned no mutable provenance")
    checkpoint_binding = capture_aura_checkpoint_compile_binding(
        checkpoint_root / "aura",
        expected_qnames=stripe_qnames,
        expected_compile_settings=compile_settings,
        expected_extra_fields={
            name: value for name, value in checkpoint_identity_extra.items()
            if name != "compile_settings"
        },
        expected_arm_identity=arm_identity,
        expected_model_identity=model_identity,
    )
    try:
        provenance[CB_COMPILE_PROOF_KEY] = build_campaign_cb_compile_proof(
            live_compile_proof,
            compile_settings=compile_settings,
            expected_qnames=stripe_qnames,
            rendered_cells=int(provenance.get(
                "production_anchor_rendered_this_invocation", -1
            )),
            restored_cells=int(provenance.get(
                "production_anchor_restored_renders", -1
            )),
            formats_per_unit=len(MEASURED_FORMATS),
            cb_formats_per_unit=len(MEASURED_CB_FORMATS),
            checkpoint_binding=checkpoint_binding,
        )
    except (RTX4090CBCompileProofError, TypeError, ValueError) as exc:
        raise RTX4090FP8BurnError(
            f"streamed AURA compile execution could not be attested: {exc}"
        ) from exc
    _attach_campaign_shard_receipt(
        payload, plan, stripe_index=stripe_index,
        compile_settings=compile_settings, model_identity=model_identity,
    )
    if _validate_campaign_shard_receipt(payload, plan) != stripe_index:
        raise RTX4090FP8BurnError("fresh shard receipt index changed")
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise RTX4090FP8BurnError(f"stripe output exists: {output}")
    if output.exists() and args.resume:
        with output.open("rb") as handle:
            prior = pickle.load(handle)
        if not isinstance(prior, Mapping):
            raise RTX4090FP8BurnError("completed stripe output is malformed")
        if _validate_campaign_shard_receipt(prior, plan) != stripe_index:
            raise RTX4090FP8BurnError(
                "completed stripe output belongs to another stripe"
            )
        if canonical_json_sha256(
            prior.get("costs", {}), where="prior stripe costs"
        ) != canonical_json_sha256(payload.get("costs", {}), where="stripe costs"):
            raise RTX4090FP8BurnError("completed stripe output differs on resume")
        return output
    atomic_write_bytes(output, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    return output


def _load_pickle_mapping(path: str | Path, *, where: str) -> dict[str, object]:
    try:
        with Path(path).open("rb") as handle:
            value = pickle.load(handle)
    except Exception as exc:
        raise RTX4090FP8BurnError(f"{where} is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise RTX4090FP8BurnError(f"{where} is not a mapping")
    return dict(value)


def merge(args: argparse.Namespace) -> Path:
    plan = load_campaign_plan(args.plan)
    _verify_binding(plan, "producer_snapshot", args.producer_snapshot)
    _validate_burn_runtime_snapshot(args.producer_snapshot)
    _verify_binding(plan, "col_weights", args.col_weights)
    from prismaquant.build_production_cache import _load_col_weights

    col_weights = _load_col_weights(args.col_weights, CB_FORMATS)
    if not isinstance(col_weights, Mapping):
        raise RTX4090FP8BurnError("column weights did not load")
    payloads = [_load_pickle_mapping(path, where="AURA stripe")
                for path in args.shards]
    if len(payloads) != STRIPE_COUNT:
        raise RTX4090FP8BurnError("merge requires exactly two raw shards")
    _validate_campaign_shards(payloads, plan)
    body_qnames = tuple(plan["body"]["qnames"])  # type: ignore[index]
    maps = plan["maps"]
    merged = merge_streamed_cb_anchor_aura_shards(
        payloads, col_weights=col_weights, expected_qnames=body_qnames,
        expected_formats_by_qname=maps["formats_by_qname"],
        expected_purposes_by_qname=maps["purposes_by_qname"],
        expected_unmeasured_formats_by_qname=(
            maps["unmeasured_formats_by_qname"]
        ),
        expected_legal_cb_formats_by_qname=(
            maps["legal_cb_formats_by_qname"]
        ),
    )
    if int(merged.get("n_probes", -1)) != AURA_N_PROBES or merged.get(
        "token_scope"
    ) != AURA_TOKEN_SCOPE:
        raise RTX4090FP8BurnError(
            "merged payload probe/token contract differs from campaign"
        )
    costs = merged.get("costs")
    if not isinstance(costs, Mapping) or set(costs) != set(body_qnames):
        raise RTX4090FP8BurnError("merged cost qnames are not the full plan")
    for qname in body_qnames:
        rows = costs[qname]
        if not isinstance(rows, Mapping) or set(rows) != set(MEASURED_FORMATS):
            raise RTX4090FP8BurnError(f"{qname}: merged measured menu differs")
        native = rows[NATIVE_FP8_FORMAT]
        if (
            not isinstance(native, Mapping)
            or native.get("cost_source") != "aura"
            or native.get("production_anchor_measured") is not True
        ):
            raise RTX4090FP8BurnError(
                f"{qname}: native FP8 is not a fresh direct measured row"
            )
    provenance = merged.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise RTX4090FP8BurnError("merged payload provenance is not mutable")
    provenance.pop(SHARD_RECEIPT_KEY, None)
    input_receipts = sorted(
        (
            copy.deepcopy(payload["provenance"][SHARD_RECEIPT_KEY])
            for payload in payloads
        ),
        key=lambda receipt: int(receipt["stripe_index"]),
    )
    burn_receipt: dict[str, object] = {
        "schema": MERGED_SCHEMA,
        "global_plan_sha256": plan["plan_sha256"],
        "fixed_bf16_census_sha256": plan["fixed_bf16_census_sha256"],
        "producer_snapshot_sha256": plan["bindings"]["producer_snapshot"]["sha256"],  # type: ignore[index]
        "common_execution_attestation_sha256": plan["bindings"]["common_execution_attestation"]["sha256"],  # type: ignore[index]
        "container_image_digest": plan["bindings"]["common_execution_attestation"]["container_image_digest"],  # type: ignore[index]
        "direct_measured_formats": list(MEASURED_FORMATS),
        "unmeasured_terminal": BF16_FORMAT,
        "input_shard_receipt_schema": SHARD_RECEIPT_SCHEMA,
        "input_shard_receipts": input_receipts,
        "merged_costs_sha256": canonical_json_sha256(
            merged["costs"], where="merged RTX4090 costs",
        ),
        "merged_stats_sha256": canonical_json_sha256(
            merged["stats"], where="merged RTX4090 stats",
        ),
    }
    burn_receipt["receipt_sha256"] = canonical_json_sha256(
        burn_receipt, where="merged RTX4090 provenance receipt",
    )
    provenance["rtx4090_fp8_burn"] = burn_receipt
    _validate_merged_campaign_provenance(merged, plan)
    output = Path(args.output)
    if output.exists():
        raise RTX4090FP8BurnError(f"merged output exists: {output}")
    atomic_write_bytes(output, pickle.dumps(merged, protocol=pickle.HIGHEST_PROTOCOL))
    return output


def _allocator_cost(
    merged: Mapping[str, object], plan: Mapping[str, object],
) -> dict[str, object]:
    if int(merged.get("n_probes", -1)) != AURA_N_PROBES:
        raise RTX4090FP8BurnError(
            "merged AURA probe count is absent or differs from campaign"
        )
    if merged.get("token_scope") != AURA_TOKEN_SCOPE:
        raise RTX4090FP8BurnError(
            "merged AURA token scope is absent or differs from campaign"
        )
    raw_costs = merged.get("costs")
    stats = merged.get("stats")
    provenance = merged.get("provenance")
    if not isinstance(raw_costs, Mapping) or not isinstance(
        stats, Mapping
    ) or not isinstance(provenance, Mapping):
        raise RTX4090FP8BurnError(
            "merged payload lacks costs/stats/provenance mappings"
        )
    body_qnames = tuple(plan["body"]["qnames"])  # type: ignore[index]
    direct_rows: dict[str, dict[str, object]] = {}
    direct_row_bytes: dict[str, dict[str, bytes]] = {}
    for qname in body_qnames:
        rows = raw_costs.get(qname)
        if not isinstance(rows, Mapping) or set(rows) != set(
            MEASURED_FORMATS
        ):
            raise RTX4090FP8BurnError(f"{qname}: allocator input menu differs")
        direct_rows[qname] = {}
        direct_row_bytes[qname] = {}
        for format_name in MEASURED_FORMATS:
            direct = rows.get(format_name)
            if not isinstance(direct, dict) or (
                direct.get("dw_source") != "production_render"
                or direct.get("production_anchor_measured") is not True
            ):
                raise RTX4090FP8BurnError(
                    f"{qname}: {format_name} is not a direct production "
                    "render"
                )
            for scalar_name in (
                "predicted_dloss", "predicted_dloss_stderr",
            ):
                try:
                    scalar_value = float(direct[scalar_name])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RTX4090FP8BurnError(
                        f"{qname}: direct {format_name} {scalar_name} is "
                        "absent or nonnumeric"
                    ) from exc
                if not math.isfinite(scalar_value) or scalar_value < 0.0:
                    raise RTX4090FP8BurnError(
                        f"{qname}: direct {format_name} {scalar_name} must "
                        "be finite and nonnegative"
                    )
            direct_rows[qname][format_name] = copy.deepcopy(direct)
            direct_row_bytes[qname][format_name] = pickle.dumps(
                direct, protocol=pickle.HIGHEST_PROTOCOL,
            )

    context = _cb_context()
    source_map = {format_name: LATTICE_BASIS for format_name in CB_FORMATS}
    renderer = provenance.get("production_anchor_renderer")
    arm_identity = (
        renderer.get("arm_identity")
        if isinstance(renderer, Mapping) else None
    )
    if not isinstance(arm_identity, Mapping) or not arm_identity:
        raise RTX4090FP8BurnError(
            "merged production renderer lacks its exact arm identity"
        )
    plugin = CodebookAnchoredFormatPlugin(
        codebook_source_by_format=source_map,
        arm_identity=arm_identity,
        anchor_formats={("fp8_cb", LATTICE_BASIS): ANCHOR_CB_FORMAT},
    )

    from prismaquant.allocator_candidates import serialized_candidate_payload

    declarations: list[CBUnitDeclaration] = []
    for qname in body_qnames:
        row = stats.get(qname)
        if not isinstance(row, Mapping):
            raise RTX4090FP8BurnError(f"{qname}: merged stats row is absent")
        shape = (
            int(row.get("out_features", 0) or 0),
            int(row.get("in_features", 0) or 0),
        )
        n_params = int(row.get("n_params", 0) or 0)
        expected_shape = tuple(
            int(value) for value in plan["body"]["shapes"][qname]  # type: ignore[index]
        )
        if (
            min(shape) <= 0
            or math.prod(shape) != n_params
            or shape != expected_shape
        ):
            raise RTX4090FP8BurnError(
                f"{qname}: merged shape/n_params contract differs from plan"
            )
        payload_bytes: dict[str, int] = {}
        for format_name in (*CB_FORMATS, BF16_FORMAT):
            payload_bytes[format_name] = serialized_candidate_payload(
                fr.get_format(format_name), shape, qname=qname,
                cb_serialization_context=context,
            )[0]
        declarations.append(CBUnitDeclaration(
            qname=qname,
            role=qname.rsplit(".", 1)[-1],
            unit_class="nonexpert",
            n_params=n_params,
            payload_bytes_by_format=payload_bytes,
            terminal_format=BF16_FORMAT,
        ))

    try:
        units = build_cb_units(tuple(declarations), plugin)
        anchor_requests = plan_anchor_requests(units, plugin)
        panel_requests: list[RenderRequest] = []
        for unit in units:
            segments = candidates_by_segment(unit, plugin)
            if len(segments) != 1:
                raise RTX4090FP8BurnError(
                    f"{unit.qname}: expected one all-lattice FP8-CB segment"
                )
            (segment,) = segments
            for format_name in MEASURED_CB_FORMATS:
                panel_requests.append(RenderRequest(
                    unit.qname, segment, format_name, "panel",
                ))
        anchors = anchors_from_streamed_payload(anchor_requests, merged)
        observations = observations_from_streamed_payload(
            tuple(panel_requests), merged
        )
        fits = fit_all_cb_segments(
            observations, units, plugin, anchors=anchors
        )
        cells = price_anchored_candidates(units, plugin, anchors, fits)
        hull = fitted_cb_hull_report(units, plugin, fits)
        hull["cost_surface"] = (
            "fitted_imputation_law_before_direct_measurement_overlay"
        )
        hull["not_a_hull_over_final_overlaid_costs"] = True
        result = build_cb_allocator_cost_payload(
            cells, streamed_payload=merged, fits=fits, hull_report=hull,
        )
    except RTX4090FP8BurnError:
        raise
    except (TypeError, ValueError) as exc:
        raise RTX4090FP8BurnError(
            f"anchored FP8-CB fit/imputation refused: {exc}"
        ) from exc

    costs = result.get("costs")
    if not isinstance(costs, dict) or set(costs) != set(body_qnames):
        raise RTX4090FP8BurnError("fitted cost table qname census differs")
    for qname in body_qnames:
        rows = costs.get(qname)
        if not isinstance(rows, dict):
            raise RTX4090FP8BurnError(f"{qname}: fitted cost row is malformed")
        for format_name in CB_FORMATS:
            cb_row = rows.get(format_name)
            if not isinstance(cb_row, dict):
                raise RTX4090FP8BurnError(
                    f"{qname}: fitted row omitted {format_name}"
                )
            if format_name not in MEASURED_CB_FORMATS:
                cb_row["production_anchor_measured"] = False
                cb_row["extrapolated_not_rendered_measurement"] = True
        for format_name in MEASURED_FORMATS:
            pristine_bytes = direct_row_bytes[qname][format_name]
            if pickle.dumps(
                raw_costs[qname][format_name],
                protocol=pickle.HIGHEST_PROTOCOL,
            ) != pristine_bytes:
                raise RTX4090FP8BurnError(
                    f"{qname}: pristine raw {format_name} row changed before "
                    "final overlay"
                )
            rows[format_name] = copy.deepcopy(
                direct_rows[qname][format_name]
            )
            if pickle.dumps(
                rows[format_name], protocol=pickle.HIGHEST_PROTOCOL
            ) != pristine_bytes:
                raise RTX4090FP8BurnError(
                    f"{qname}: direct {format_name} row changed during "
                    "finalization"
                )
        if set(rows) != set(FULL_FORMATS):
            raise RTX4090FP8BurnError(
                f"{qname}: fitted allocator menu is not FULL_FORMATS"
            )
        terminal = rows.get(BF16_FORMAT)
        if not isinstance(terminal, Mapping) or terminal.get(
            "cost_source"
        ) != "source_passthrough":
            raise RTX4090FP8BurnError(
                f"{qname}: BF16 is not the exact source passthrough terminal"
            )
    result["formats"] = list(FULL_FORMATS)
    result["stats"] = copy.deepcopy(dict(stats))
    result["n_probes"] = AURA_N_PROBES
    result["token_scope"] = AURA_TOKEN_SCOPE
    meta = result.setdefault("meta", {})
    if not isinstance(meta, dict):
        raise RTX4090FP8BurnError("allocator cost meta is not mutable")
    imputed_formats = tuple(
        format_name for format_name in CB_FORMATS
        if format_name not in MEASURED_CB_FORMATS
    )
    meta.update({
        "unit_count": len(costs),
        "cell_count": sum(len(rows) for rows in costs.values()),
        "cost_currency": "aura_predicted_dloss",
        "cost_semantics": (
            "mixed final table: byte-preserved direct production-render "
            "AURA cells, anchored-AURA imputed CB cells, and an exact "
            "zero-loss BF16 source-passthrough terminal"
        ),
        "cell_semantics_counts": {
            "direct_measured": len(costs) * len(MEASURED_FORMATS),
            "anchored_cb_imputed": len(costs) * len(imputed_formats),
            "source_passthrough_terminal": len(costs),
        },
    })
    provenance = result.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise RTX4090FP8BurnError("allocator cost provenance is not mutable")
    provenance["rtx4090_fp8_burn_allocator_cost"] = {
        "schema": ALLOCATOR_COST_SCHEMA,
        "global_plan_sha256": plan["plan_sha256"],
        "target_bytes": RTX4090_CONTEXT_FIRST_TARGET_BYTES,
        "bf16_terminal_is_identity_by_construction": True,
        "cb_fit_panel_formats": list(MEASURED_CB_FORMATS),
        "cb_anchor_format": ANCHOR_CB_FORMAT,
        "direct_measured_formats": list(MEASURED_FORMATS),
        "imputed_formats": list(imputed_formats),
        "direct_rows_preserved_without_mutation": True,
        "fitted_hull_scope": (
            "imputation_law_not_final_overlaid_cost_table"
        ),
    }
    return result


def allocate(args: argparse.Namespace) -> Path:
    plan = load_campaign_plan(args.plan)
    _verify_binding(plan, "producer_snapshot", args.producer_snapshot)
    _validate_burn_runtime_snapshot(args.producer_snapshot)
    _verify_binding(plan, "probe", args.probe)
    _verify_binding(plan, "col_weights", args.col_weights)
    _verify_binding(plan, "sample_merge_commit", args.sample_merge_commit)
    _verify_binding(
        plan,
        "activation_cache_manifest",
        Path(args.activation_cache_dir) / "sample_parallel_merge.json",
    )
    from prismaquant.model_profiles import detect_profile

    sample_bundle = _validate_sample_merge_bundle(
        probe=args.probe,
        activation_cache_dir=args.activation_cache_dir,
        commit_path=args.sample_merge_commit,
    )
    validated_probe_payload = sample_bundle.get(
        "_validated_probe_payload"
    )
    validated_probe_bytes = sample_bundle.get("_validated_probe_bytes")
    if not isinstance(validated_probe_payload, Mapping) or not isinstance(
        validated_probe_bytes, bytes
    ):
        raise RTX4090FP8BurnError(
            "validated sample bundle lacks its exact captured probe"
        )
    bindings = plan.get("bindings")
    probe_binding = bindings.get("probe") if isinstance(
        bindings, Mapping
    ) else None
    activation_binding = bindings.get(
        "activation_cache_manifest"
    ) if isinstance(bindings, Mapping) else None
    source_binding = bindings.get("source_model_identity") if isinstance(
        bindings, Mapping
    ) else None
    if (
        not isinstance(probe_binding, Mapping)
        or not isinstance(activation_binding, Mapping)
        or not isinstance(source_binding, Mapping)
        or sample_bundle.get("probe_sha256")
        != probe_binding.get("sha256")
        or sample_bundle.get("probe_bytes") != probe_binding.get("bytes")
        or sample_bundle.get("activation_manifest_identity_sha256")
        != activation_binding.get("identity_sha256")
    ):
        raise RTX4090FP8BurnError(
            "validated sample-merge members differ from the campaign plan"
        )
    _validate_sample_bundle_source_binding(sample_bundle, source_binding)
    _revalidate_live_campaign_census(
        model=args.model,
        probe=args.probe,
        plan=plan,
        profile=detect_profile(args.model),
        validated_probe_payload=validated_probe_payload,
    )
    if _probe_imatrix_contract(
        args.probe,
        args.col_weights,
        validated_probe_payload=validated_probe_payload,
    ) != plan["imatrix"]:
        raise RTX4090FP8BurnError(
            "live probe-derived imatrix differs from campaign plan"
        )
    merged = _load_pickle_mapping(args.merged, where="merged AURA payload")
    _validate_merged_campaign_provenance(merged, plan)
    cost_payload = _allocator_cost(merged, plan)
    cost_path = Path(args.cost_output)
    if cost_path.exists():
        raise RTX4090FP8BurnError(f"allocator cost output exists: {cost_path}")
    atomic_write_bytes(
        cost_path, pickle.dumps(cost_payload, protocol=pickle.HIGHEST_PROTOCOL)
    )
    output_dir = Path(args.output_dir)
    with _sealed_allocator_probe(
        validated_probe_bytes,
        expected_sha256=str(sample_bundle["probe_sha256"]),
    ) as (sealed_probe_path, sealed_probe_fd):
        command = [
            sys.executable, "-m", "prismaquant.allocator",
            "--probe", sealed_probe_path, "--costs", str(cost_path),
            "--model-override", str(args.model),
            "--target-profile", RTX4090_QWEN38_SERVING_PROFILE,
            "--target-disk-gb", "18.000000000",
            "--artifact-overhead-reserve-bytes",
            str(ARTIFACT_OVERHEAD_RESERVE_BYTES),
            "--cb-scale-coding", "v1", "--cb-codebook-source", "lattice",
            "--cb-codebook-source-scope", "none", "--cb-scale-sweep", "1",
            "--cb-scale-sweep-scope", "fp8", "--cb-ldlq", "0",
            "--cb-ldlq-scope", "none", "--cb-encode-tier", "balanced",
            "--cb-col-weights", str(args.col_weights),
            "--formats", ",".join(FULL_FORMATS),
            "--lm-head-format", BF16_FORMAT, "--mtp-format", BF16_FORMAT,
            "--visual-format", BF16_FORMAT, "--threads", str(args.threads),
            "--layer-config", str(output_dir / "layer_config.json"),
            "--pareto-csv", str(output_dir / "pareto.csv"),
            "--pareto-output-dir", str(output_dir / "pareto-points"),
            "--applicability-report",
            str(output_dir / "format_applicability.json"),
            "--bit-attribution-json", str(output_dir / "bit_attribution.json"),
            "--bit-attribution-csv", str(output_dir / "bit_attribution.csv"),
        ]
        run_allocator_once(
            command=command, output_dir=output_dir, resume=bool(args.resume),
            pass_fds=(sealed_probe_fd,),
            invocation_provenance={
                "campaign_schema": PLAN_SCHEMA,
                "global_plan_sha256": plan["plan_sha256"],
                "target_bytes": RTX4090_CONTEXT_FIRST_TARGET_BYTES,
                "cost_payload_sha256": _sha256_file(cost_path),
                "sample_merge_commit_identity_sha256": sample_bundle[
                    "commit_identity_sha256"
                ],
                "validated_probe_sha256": sample_bundle["probe_sha256"],
                "validated_probe_bytes": sample_bundle["probe_bytes"],
                "direct_native_fp8_measurement": True,
                "bf16_unmeasured_terminal": True,
            },
        )
    return output_dir / "layer_config.json"


def _common_prepare(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--col-weights", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-identity", required=True)
    parser.add_argument("--producer-snapshot", required=True)
    parser.add_argument("--execution-attestation", required=True)
    parser.add_argument(
        "--launcher-image-digest",
        default=os.environ.get("PRISMAQUANT_PRODUCER_IMAGE_DIGEST"),
        required=os.environ.get("PRISMAQUANT_PRODUCER_IMAGE_DIGEST") is None,
        help=(
            "Host-launcher-verified immutable registry RepoDigest in "
            "sha256:<64-hex> form."
        ),
    )
    parser.add_argument("--sample-merge-commit", required=True)
    parser.add_argument("--activation-cache-dir", required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    derive_parser = sub.add_parser("derive-col-weights")
    derive_parser.add_argument("--sample-merge-bundle", required=True)
    derive_parser.add_argument("--output", required=True)
    derive_parser.set_defaults(handler=derive_col_weights)

    attest_parser = sub.add_parser("attest-execution")
    attest_parser.add_argument("--sample-run-contract", required=True)
    attest_parser.add_argument("--producer-snapshot", required=True)
    attest_parser.add_argument(
        "--launcher-image-digest",
        default=os.environ.get("PRISMAQUANT_PRODUCER_IMAGE_DIGEST"),
        required=os.environ.get("PRISMAQUANT_PRODUCER_IMAGE_DIGEST") is None,
        help=(
            "Host-launcher-verified immutable registry RepoDigest in "
            "sha256:<64-hex> form."
        ),
    )
    attest_parser.add_argument("--output", required=True)
    attest_parser.set_defaults(handler=attest_execution)

    prepare_parser = sub.add_parser("prepare")
    _common_prepare(prepare_parser)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument(
        "--n-calib-samples", type=int, default=CALIBRATION_NSAMPLES,
    )
    prepare_parser.add_argument(
        "--calib-seqlen", type=int, default=CALIBRATION_SEQLEN,
    )
    prepare_parser.add_argument(
        "--calib-seed", type=int, default=CALIBRATION_SEED,
    )
    prepare_parser.set_defaults(handler=prepare)

    measure_parser = sub.add_parser("measure")
    _common_prepare(measure_parser)
    measure_parser.add_argument("--plan", required=True)
    measure_parser.add_argument("--stripe", required=True, type=int)
    measure_parser.add_argument("--checkpoint-dir", required=True)
    measure_parser.add_argument("--output", required=True)
    measure_parser.add_argument(
        "--n-probes", type=int, default=AURA_N_PROBES,
    )
    measure_parser.add_argument("--resume", action="store_true")
    measure_parser.set_defaults(handler=measure)

    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--plan", required=True)
    merge_parser.add_argument("--producer-snapshot", required=True)
    merge_parser.add_argument("--col-weights", required=True)
    merge_parser.add_argument("--shards", nargs="+", required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.set_defaults(handler=merge)

    allocate_parser = sub.add_parser("allocate")
    allocate_parser.add_argument("--plan", required=True)
    allocate_parser.add_argument("--producer-snapshot", required=True)
    allocate_parser.add_argument("--model", required=True)
    allocate_parser.add_argument("--probe", required=True)
    allocate_parser.add_argument("--sample-merge-commit", required=True)
    allocate_parser.add_argument("--activation-cache-dir", required=True)
    allocate_parser.add_argument("--col-weights", required=True)
    allocate_parser.add_argument("--merged", required=True)
    allocate_parser.add_argument("--cost-output", required=True)
    allocate_parser.add_argument("--output-dir", required=True)
    allocate_parser.add_argument("--threads", type=int, default=16)
    allocate_parser.add_argument("--resume", action="store_true")
    allocate_parser.set_defaults(handler=allocate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = args.handler(args)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
