"""Persisted scale-search warm starts for the CB encoder.

Warm state is deliberately smaller and weaker than a rendered-weight cache:
it stores only the scale sweep's selected parameters.  Export always reruns
the lattice/codebook assignment and packer.  Matching is fail-closed over the
same decoded-source and imatrix value digests used by the CB render identity,
and sampled export verification compares both selected scales and rendered
bytes against a fresh full sweep.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from .cb_layout import (
    FP4_GROUP,
    SCALE_CODING_TWO_TIER,
    SCALE_CODING_V1,
    SUPERBLOCK,
    parse_format_name,
)
from .format_registry import canonical_format_name
from .nvfp4_cb_footprint import (
    CBSerializationContext,
    _ldlq_for_format,
    cb_serialization_context_stamp,
    lattice_codebook_content_sha256,
)


CB_WARM_STATE_SCHEMA = "prismaquant.cb_encoder_warm_state.v1"
CB_WARM_CONTEXT_SCHEMA = "prismaquant.cb_encoder_warm_context.v1"
CB_WARM_ENCODER_SCHEMA = "prismaquant.cb_encoder_initializer.v1"
_METADATA_KEY = "prismaquant_cb_warm_state"
_SCALE_KEYS = ("scales", "scale_super", "scale_sub")


class WarmStateVerificationError(RuntimeError):
    """A sampled warm encode differed from a full scale sweep."""


@dataclass(frozen=True)
class CBWarmStateRecord:
    metadata: Mapping[str, Any]
    scale_state: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class CBEncodedPayload:
    """Adapter result used by real and fake encoders in verification tests."""

    value: Any
    selected_scale: Mapping[str, Any]
    rendered: Mapping[str, Any]


def tensor_value_identity(tensor: torch.Tensor) -> tuple[list[int], str]:
    """Return the canonical decoded-value identity used by render stamps."""
    # Keep one implementation of source identity.  This helper is private in
    # production_weight_cache only because the cache originally had its sole
    # caller; warm records intentionally follow that exact established idiom.
    from .production_weight_cache import _source_weight_value_identity

    return _source_weight_value_identity(torch.as_tensor(tensor))


def tensor_value_sha256(tensor: torch.Tensor) -> str:
    return tensor_value_identity(tensor)[1]


def warm_serialization_context(
    context: CBSerializationContext,
    format_name: str,
) -> dict[str, Any]:
    """Project the full byte-affecting encoder context onto one CB format."""
    fmt = canonical_format_name(format_name)
    stamp = cb_serialization_context_stamp(context, formats=[fmt])
    # A warm record belongs to one format, whereas the producer context can
    # enable LDLQ for a family set.  Persist the decision that actually reaches
    # this encoder so an unrelated family's scope cannot cause a cold sweep.
    # Keep a canonical scope in the stamp so modern records still compare as a
    # complete dictionary and an explicit, conflicting legacy scope fails.
    format_ldlq = _ldlq_for_format(fmt, context)
    stamp["ldlq"] = format_ldlq
    stamp["ldlq_scope"] = "all" if format_ldlq else "none"
    if not format_ldlq:
        stamp.pop("ldlq_packed_kernel", None)
    # Lattice physical sidecar names do not affect assignment.  Export knows
    # those names while inline cost does not, so retain the canonical lattice
    # payload digests and discard artifact-wide naming details.  Learned
    # codebooks are not warm-startable until cost has the materialized table.
    if context.codebook_source == "lattice":
        stamp.pop("codebook_refs", None)
        stamp.pop("codebook_content_sha256", None)
    return {
        "schema": CB_WARM_CONTEXT_SCHEMA,
        "format": fmt,
        "serialization": stamp,
    }


def encoder_initializer_identity(
    context: CBSerializationContext,
    format_name: str,
) -> dict[str, Any]:
    """Identify the codebook initializer/seed behind a selected scale."""
    fmt = canonical_format_name(format_name)
    if context.codebook_source != "lattice":
        raise ValueError(
            "CB warm state currently requires codebook_source='lattice'; "
            "learned cost renders must bind their materialized initializer"
        )
    from . import nvfp4_cb_formats as cb

    return {
        "schema": CB_WARM_ENCODER_SCHEMA,
        "codebook_source": "lattice",
        "initializer": "fixed_lattice.grid_snapped_lloyd",
        "seed": int(cb._LATTICE_SEED),
        "sample_count": int(cb._LATTICE_SAMPLES),
        "lloyd_iters": int(cb._LATTICE_ITERS),
        "content_sha256": list(lattice_codebook_content_sha256(fmt)),
    }


def selected_scale_state(fields: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Extract only the scale-sweep argmin, never indices or codebooks."""
    if "scales" not in fields:
        raise ValueError("CB encoder fields have no selected scales")
    out = {
        "scales": torch.as_tensor(fields["scales"]).detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
    }
    for key in ("scale_super", "scale_sub"):
        if key in fields:
            out[key] = torch.as_tensor(fields[key]).detach().to(
                device="cpu", dtype=torch.uint8
            ).contiguous()
    return out


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _record_key(qname: str, format_name: str, source_digest: str) -> str:
    return hashlib.sha256(_canonical_json({
        "qname": str(qname),
        "format": canonical_format_name(format_name),
        "source_digest": str(source_digest).lower(),
    }).encode("utf-8")).hexdigest()


def _canonicalize_warm_metadata(
    metadata: Any,
    expected: Mapping[str, Any],
) -> Any:
    """Project validated historical aggregate LDLQ stamps onto one format.

    The result still has to equal ``expected`` as a complete dictionary.  This
    helper is intentionally not a subset matcher: it validates the historical
    global scope, bool, and packed-kernel identity before replacing only those
    fields with their effective per-format identity.  Every other difference
    remains a cold fallback.
    """
    if not isinstance(metadata, Mapping):
        return metadata
    candidate = dict(metadata)
    raw_context = candidate.get("serialization_context")
    expected_context = expected.get("serialization_context")
    if not isinstance(raw_context, Mapping) or not isinstance(
        expected_context, Mapping
    ):
        return candidate
    raw_serialization = raw_context.get("serialization")
    if not isinstance(raw_serialization, Mapping):
        return candidate

    serialization = dict(raw_serialization)
    raw_ldlq = serialization.get("ldlq")
    if type(raw_ldlq) is not bool:
        return candidate
    raw_scope = serialization.get("ldlq_scope")
    if "ldlq_scope" not in serialization:
        # Before per-family scope, the bool meant all/none.  Only the raw arm
        # is useful without today's packed-kernel stamp; an active old record
        # still fails the kernel check below.
        aggregate_scope = "all" if raw_ldlq else "none"
    elif raw_scope in {"none", "nvfp4", "all"}:
        aggregate_scope = raw_scope
        if raw_ldlq != (aggregate_scope != "none"):
            return candidate
    else:
        return candidate

    raw_kernel = serialization.get("ldlq_packed_kernel")
    if aggregate_scope == "none":
        if "ldlq_packed_kernel" in serialization:
            return candidate
    else:
        from .nvfp4_cb_formats import packed_ldlq_artifact_stamp

        if (
            not isinstance(raw_kernel, Mapping)
            or dict(raw_kernel) != packed_ldlq_artifact_stamp()
        ):
            return candidate

    fmt = expected.get("format")
    parsed = parse_format_name(str(fmt))
    if parsed is None:
        return candidate
    family, _k = parsed
    effective_ldlq = aggregate_scope == "all" or (
        aggregate_scope == "nvfp4" and family.grid == "fp4"
    )
    serialization["ldlq"] = effective_ldlq
    serialization["ldlq_scope"] = "all" if effective_ldlq else "none"
    if not effective_ldlq:
        # Validate an aggregate-scope kernel above before discarding it as
        # irrelevant to this format.
        serialization.pop("ldlq_packed_kernel", None)
    context = dict(raw_context)
    context["serialization"] = serialization
    candidate["serialization_context"] = context

    return candidate


def build_warm_record(
    *,
    qname: str,
    format_name: str,
    source_weight: torch.Tensor,
    col_weights: torch.Tensor,
    context: CBSerializationContext,
    fields: Mapping[str, Any],
    source_identity: tuple[list[int], str] | None = None,
    col_weights_identity: tuple[list[int], str] | None = None,
) -> CBWarmStateRecord:
    """Build a versioned record from a completed full cost encode."""
    source_shape, source_digest = (
        source_identity or tensor_value_identity(source_weight)
    )
    col_shape, col_digest = (
        col_weights_identity or tensor_value_identity(col_weights)
    )
    if list(source_shape) != list(source_weight.shape):
        raise ValueError("precomputed warm source identity has the wrong shape")
    if list(col_shape) != list(col_weights.shape):
        raise ValueError("precomputed warm imatrix identity has the wrong shape")
    fmt = canonical_format_name(format_name)
    metadata = {
        "schema": CB_WARM_STATE_SCHEMA,
        "qname": str(qname),
        "format": fmt,
        "source_shape": source_shape,
        "source_digest": source_digest,
        "col_weights_shape": col_shape,
        "col_weights_digest": col_digest,
        "serialization_context": warm_serialization_context(context, fmt),
        "encoder_initializer": encoder_initializer_identity(context, fmt),
    }
    return CBWarmStateRecord(
        metadata=metadata,
        scale_state=selected_scale_state(fields),
    )


def _validate_scale_state(
    scale_state: Mapping[str, torch.Tensor],
    *,
    format_name: str,
    source_shape: list[int],
    context: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    parsed = parse_format_name(canonical_format_name(format_name))
    if parsed is None:
        raise ValueError(f"{format_name!r} is not a CB format")
    family, _k = parsed
    rows = 1
    for dim in source_shape[:-1]:
        rows *= int(dim)
    in_features = int(source_shape[-1])
    groups = in_features // FP4_GROUP if family.grid == "fp4" else 1
    expected_scales = (rows, groups)
    scales = torch.as_tensor(scale_state.get("scales"))
    if tuple(scales.shape) != expected_scales:
        raise ValueError(
            f"warm scales shape {tuple(scales.shape)} != {expected_scales}"
        )
    scales = scales.to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError("warm scales must be finite and strictly positive")

    serialization = context.get("serialization")
    if not isinstance(serialization, Mapping):
        raise ValueError("warm serialization context is malformed")
    coding = (
        str(serialization.get("scale_coding"))
        if family.grid == "fp4"
        else SCALE_CODING_V1
    )
    out = {"scales": scales}
    extras = set(scale_state) - set(_SCALE_KEYS)
    if extras:
        raise ValueError(f"warm scale state has unknown tensors {sorted(extras)}")
    if coding == SCALE_CODING_TWO_TIER:
        n_sb = in_features // SUPERBLOCK
        super_e = torch.as_tensor(scale_state.get("scale_super"))
        sub_c = torch.as_tensor(scale_state.get("scale_sub"))
        if tuple(super_e.shape) != (rows, n_sb):
            raise ValueError("warm two-tier super-scale shape differs")
        if tuple(sub_c.shape) != (rows, groups):
            raise ValueError("warm two-tier sub-scale shape differs")
        super_e = super_e.to(device="cpu", dtype=torch.uint8).contiguous()
        sub_c = sub_c.to(device="cpu", dtype=torch.uint8).contiguous()
        if bool((sub_c > 15).any()):
            raise ValueError("warm two-tier sub-scale code is not 4-bit")
        from . import nvfp4_cb_formats as cb

        _, compose, legal = cb._two_tier_tables("cpu")
        e = super_e.to(torch.int64).unsqueeze(-1).expand(
            rows, n_sb, SUPERBLOCK // FP4_GROUP
        ).reshape(rows, groups)
        c = sub_c.to(torch.int64)
        if not bool(legal[e, c].all()):
            raise ValueError("warm two-tier scale contains an illegal pair")
        composed = compose[e, c]
        if not torch.equal(composed, scales):
            raise ValueError("warm composed two-tier scales differ from argmin")
        out.update(scale_super=super_e, scale_sub=sub_c)
    elif coding == SCALE_CODING_V1:
        if "scale_super" in scale_state or "scale_sub" in scale_state:
            raise ValueError("legacy/fp8 warm state carries two-tier tensors")
        if family.grid == "fp4":
            roundtrip = scales.to(torch.float8_e4m3fn).to(torch.float32)
            if not torch.equal(roundtrip, scales):
                raise ValueError("warm v1 FP4 scales are not E4M3-exact")
    else:
        raise ValueError(f"unknown warm scale coding {coding!r}")
    return out


class CBWarmStateStore:
    """One atomic safetensors record per content-keyed unit/rung."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, qname: str, format_name: str, source_digest: str) -> Path:
        key = _record_key(qname, format_name, source_digest)
        return self.root / key[:2] / f"{key}.safetensors"

    def write(self, record: CBWarmStateRecord) -> Path:
        metadata = dict(record.metadata)
        path = self.path_for(
            str(metadata["qname"]),
            str(metadata["format"]),
            str(metadata["source_digest"]),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            str(name): torch.as_tensor(value).detach().cpu().contiguous()
            for name, value in record.scale_state.items()
        }
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_name = handle.name
            save_file(tensors, temp_name, metadata={
                _METADATA_KEY: _canonical_json(metadata),
            })
            os.replace(temp_name, path)
        finally:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)
        return path

    def load_matching(
        self,
        *,
        qname: str,
        format_name: str,
        source_shape: list[int],
        source_digest: str,
        col_weights_shape: list[int],
        col_weights_digest: str,
        context: CBSerializationContext,
    ) -> CBWarmStateRecord | None:
        """Return a fully matching, structurally valid record or ``None``."""
        fmt = canonical_format_name(format_name)
        path = self.path_for(qname, fmt, source_digest)
        if not path.is_file():
            return None
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                raw = (handle.metadata() or {}).get(_METADATA_KEY)
                metadata = json.loads(raw) if raw is not None else None
                tensors = {name: handle.get_tensor(name) for name in handle.keys()}
            expected = {
                "schema": CB_WARM_STATE_SCHEMA,
                "qname": str(qname),
                "format": fmt,
                "source_shape": [int(dim) for dim in source_shape],
                "source_digest": str(source_digest).lower(),
                "col_weights_shape": [int(dim) for dim in col_weights_shape],
                "col_weights_digest": str(col_weights_digest).lower(),
                "serialization_context": warm_serialization_context(context, fmt),
                "encoder_initializer": encoder_initializer_identity(context, fmt),
            }
            metadata = _canonicalize_warm_metadata(metadata, expected)
            if metadata != expected:
                return None
            normalized = _validate_scale_state(
                tensors,
                format_name=fmt,
                source_shape=expected["source_shape"],
                context=expected["serialization_context"],
            )
            return CBWarmStateRecord(metadata=metadata, scale_state=normalized)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            RuntimeError,
            SafetensorError,
            json.JSONDecodeError,
        ):
            # A sidecar is an optimization only.  Corrupt, stale, partial, or
            # structurally impossible state is indistinguishable from absent
            # state and therefore triggers the ordinary full encode.
            return None


def _equal_value(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    return left == right


def _mapping_mismatches(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> list[str]:
    if set(left) != set(right):
        return [f"keys {sorted(left)} != {sorted(right)}"]
    return [key for key in sorted(left) if not _equal_value(left[key], right[key])]


def execute_warm_started_encode(
    *,
    qname: str,
    format_name: str,
    record: CBWarmStateRecord | None,
    verify: bool,
    full_encode: Callable[[], CBEncodedPayload],
    seeded_encode: Callable[[Mapping[str, torch.Tensor]], CBEncodedPayload],
) -> tuple[CBEncodedPayload, str]:
    """Run cold or warm encode; sampled disagreement always aborts."""
    if record is None:
        return full_encode(), "cold_fallback"
    warm = seeded_encode(record.scale_state)
    if verify:
        cold = full_encode()
        scale_diff = _mapping_mismatches(
            warm.selected_scale, cold.selected_scale
        )
        byte_diff = _mapping_mismatches(warm.rendered, cold.rendered)
        if scale_diff or byte_diff:
            raise WarmStateVerificationError(
                "CB warm-state verification failed for "
                f"{qname} ({canonical_format_name(format_name)}): "
                f"chosen scale mismatch={scale_diff or 'none'}, "
                f"rendered byte mismatch={byte_diff or 'none'}; "
                "aborting export because warm state is never trusted"
            )
        return warm, "verified"
    return warm, "warm_used"


class CBWarmStartSession:
    """Counted, deterministic random sample over matching warm records."""

    def __init__(
        self,
        records: Mapping[str, CBWarmStateRecord],
        *,
        all_qnames: list[str],
        verify_sample: int,
    ) -> None:
        if int(verify_sample) < 0:
            raise ValueError("warm_verify_sample must be >= 0")
        self.records = dict(records)
        candidates = sorted(self.records)
        sample_n = min(int(verify_sample), len(candidates))
        seed_material = _canonical_json([
            [name, self.records[name].metadata.get("source_digest")]
            for name in candidates
        ])
        seed = int.from_bytes(
            hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
            "little",
        )
        self.verify_qnames = set(
            random.Random(seed).sample(candidates, sample_n)
        )
        self._all_qnames = set(str(name) for name in all_qnames)
        self._seen: set[str] = set()
        self.warm_used = 0
        self.cold_fallback = 0
        self.verified_n = 0

    def encode(
        self,
        qname: str,
        format_name: str,
        *,
        full_encode: Callable[[], CBEncodedPayload],
        seeded_encode: Callable[[Mapping[str, torch.Tensor]], CBEncodedPayload],
    ) -> CBEncodedPayload:
        name = str(qname)
        if name in self._seen:
            raise RuntimeError(f"CB warm session encoded {name!r} twice")
        self._seen.add(name)
        record = self.records.get(name)
        payload, outcome = execute_warm_started_encode(
            qname=name,
            format_name=format_name,
            record=record,
            verify=name in self.verify_qnames,
            full_encode=full_encode,
            seeded_encode=seeded_encode,
        )
        if outcome == "cold_fallback":
            self.cold_fallback += 1
        else:
            self.warm_used += 1
            if outcome == "verified":
                self.verified_n += 1
        return payload

    def provenance(self) -> dict[str, int]:
        missing = sorted(self._all_qnames - self._seen)
        extra = sorted(self._seen - self._all_qnames)
        if missing or extra:
            raise RuntimeError(
                "CB warm session coverage differs from export targets: "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )
        return {
            "warm_used": int(self.warm_used),
            "cold_fallback": int(self.cold_fallback),
            "verified_n": int(self.verified_n),
        }
