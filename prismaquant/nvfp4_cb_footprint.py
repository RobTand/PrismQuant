"""Authoritative serialized-payload accounting for Gridbook CB formats.

The producer writes two kinds of payload:

* one per-Linear packed tensor (plus an FP32 row-scale tensor for FP8-CB, or
  one FP32 static activation scalar for contracted FP4-CB); and
* FP16 codebook subtables shared once per ``(codebook_ref, format)``.

This module describes those tensors, not an abstract nominal rate.  In
particular, production FP4-CB uses layout-v2 ``4k + 9`` byte superblocks,
FP8-CB carries ``4 * output_rows`` scale bytes, and neither CB family has an
NVFP4-style *weight* global-scale scalar.  The versioned static W4A4 execution
variant adds exactly one four-byte ``input_global_scale`` to each FP4-CB target;
historical/research v2 artifacts do not. Product codebook sizes are
derived from the exact subtable shapes emitted by
:mod:`prismaquant.export_nvfp4_cb`.

``cb_footprint`` is retained as the backwards-compatible Phase-0 entry point.
New producer code should use :func:`cb_tensor_payload_breakdown` and
:func:`cb_assignment_payload_breakdown` directly.  Every returned payload is
versioned so persisted reports remain interpretable after future layouts land.
These are tensor-data bytes, not safetensors container or export-directory
bytes; :func:`finalize_cb_export_artifact_inventory` measures and persists that
separate post-export scope.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import warnings
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import format_registry as fr
from .cb_layout import (
    INDEX_BYTES_PER_K,
    LAYOUT_FOR_SCALE_CODING,
    SCALE_CODINGS,
    SCALE_CODING_TWO_TIER,
    SCALE_CODING_V1,
    SUPERBLOCK,
    VEC_DIM,
    codebook_subtable_shapes as _layout_codebook_subtable_shapes,
    family_for,
    parse_format_name,
    subtable_bit_widths,
    type_size as cb_type_size,
)
from .routed_moe_codebooks import bundle_role_qname
from .shard_layout import SHARD_INDEX_NAME, describe_container_layout

CB_SERIALIZED_PAYLOAD_SCHEMA = "prismaquant.cb_serialized_payload.v3"
MINCHAIN_CB_SERIALIZED_PAYLOAD_SCHEMA = "prismaquant.cb_serialized_payload.v4"
PREVIOUS_CB_SERIALIZED_PAYLOAD_SCHEMA = "prismaquant.cb_serialized_payload.v2"
LEGACY_CB_SERIALIZED_PAYLOAD_SCHEMA = "prismaquant.cb_serialized_payload.v1"
CB_EXPORT_ARTIFACT_INVENTORY_SCHEMA = (
    "prismaquant.cb_export_artifact_inventory.v1"
)
WHOLE_ARTIFACT_BUDGET_SCHEMA = "prismaquant.whole_artifact_budget.v2"
WHOLE_ARTIFACT_BUDGET_FIELD = "whole_artifact_budget"
CB_TENSOR_IDENTITY_FIELD = "cb_serialized_identity"
CB_ASSIGNMENT_IDENTITIES_FIELD = "cb_serialized_identities"
PRODUCTION_FP4_SCALE_CODING = SCALE_CODING_TWO_TIER
LEGACY_FP4_SCALE_CODING = SCALE_CODING_V1
CB_RENDERER_ABI = "prismaquant.nvfp4_cb_renderer.v1"
CB_ENCODE_TIER_DEFAULT = "balanced"
CB_ENCODE_TIERS = frozenset({"fast", "balanced", "max"})
_SCALE_CODINGS = SCALE_CODINGS
_LAYOUT_FOR_SCALE_CODING = LAYOUT_FOR_SCALE_CODING
_FP16_BYTES = 2
_FP32_BYTES = 4
_SUPERBLOCK = SUPERBLOCK
_VEC_DIM = VEC_DIM


def _serialized_payload_schema(context: "CBSerializationContext") -> str:
    return (
        MINCHAIN_CB_SERIALIZED_PAYLOAD_SCHEMA
        if context.minchain
        else (
            CB_SERIALIZED_PAYLOAD_SCHEMA
            if context.activation_contract is not None
            else PREVIOUS_CB_SERIALIZED_PAYLOAD_SCHEMA
        )
    )


_SAFETENSORS_DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E4M3": 8,
    "F8_E4M3FN": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2": 8,
    "F8_E5M2FNUZ": 8,
    "F8_E8M0": 8,
    "F4": 4,
    "F4_E2M1": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U16": 16,
    "I16": 16,
    "F16": 16,
    "BF16": 16,
    "U32": 32,
    "I32": 32,
    "F32": 32,
    "U64": 64,
    "I64": 64,
    "F64": 64,
}

@dataclass(frozen=True)
class CBSerializationContext:
    """Artifact-wide producer choices needed for exact CB byte pricing.

    ``codebook_source_by_format`` is authoritative when present; the legacy
    ``codebook_source`` scalar is its artifact-wide ANY (``learned`` iff any
    rung is learned). Lattice rungs share one table set per format and learned
    rungs share one per projection role. A caller with an already-materialized
    config can pass exact physical tensor names through ``codebook_refs``
    (qname -> string/list). Omitting the context is an error on exact producer
    paths: otherwise an old caller would silently fall back to legacy-v1 bytes.
    """

    scale_coding: str
    codebook_source: str
    layout_version: int | None = None
    codebook_source_scope: str | None = None
    codebook_source_by_format: Mapping[str, str] | None = None
    scale_sweep: bool = True
    scale_sweep_scope: str | None = None
    ldlq: bool = False
    ldlq_scope: str | None = None
    minchain: bool = False
    minchain_version: str | None = None
    encode_tier: str = CB_ENCODE_TIER_DEFAULT
    renderer_abi: str = CB_RENDERER_ABI
    activation_contract: str | None = None
    activation_execution: str | None = None
    codebook_refs: Mapping[str, str | Sequence[str]] | None = None
    codebook_refs_by_qname_format: Mapping[
        str, Mapping[str, str | Sequence[str]]
    ] | None = None
    codebook_content_digests: Mapping[str, str] | None = None
    # Runtime locator only. Artifact identity is the complete physical ref ->
    # FP16 digest map above; paths are intentionally never serialized.
    codebook_bundle_path: str | None = None

    def __post_init__(self) -> None:
        coding = str(self.scale_coding).strip().lower()
        source = str(self.codebook_source).strip().lower()
        if coding not in _SCALE_CODINGS:
            raise ValueError(
                f"unknown CB scale_coding {self.scale_coding!r}; expected "
                f"{sorted(_SCALE_CODINGS)}"
            )
        if source not in {"lattice", "learned"}:
            raise ValueError(
                f"unknown CB codebook_source {self.codebook_source!r}; "
                "expected 'lattice' or 'learned'"
            )
        expected_layout = _LAYOUT_FOR_SCALE_CODING[coding]
        layout = expected_layout if self.layout_version is None else int(
            self.layout_version
        )
        if layout != expected_layout:
            raise ValueError(
                f"scale_coding={coding!r} requires layout_version="
                f"{expected_layout}, got {layout}"
            )
        object.__setattr__(self, "scale_coding", coding)
        object.__setattr__(self, "codebook_source", source)
        object.__setattr__(self, "layout_version", layout)
        raw_source_scope = self.codebook_source_scope
        if raw_source_scope is not None:
            source_scope = str(raw_source_scope).strip().lower()
            if source_scope not in {"none", "fp8", "all"}:
                raise ValueError(
                    "CB codebook_source_scope must be one of "
                    f"none/fp8/all, got {raw_source_scope!r}"
                )
            # The scope is authoritative, as with LDLQ below.  Keep the old
            # scalar normalized to the artifact-wide ANY so old readers still
            # see lattice for an all-lattice artifact and learned whenever any
            # family carries learned bytes.
            source = "lattice" if source_scope == "none" else "learned"
            object.__setattr__(self, "codebook_source", source)
            object.__setattr__(self, "codebook_source_scope", source_scope)
            if source_scope == "all":
                warnings.warn(
                    "codebook_source_scope='all' is research-only: learned "
                    "NVFP4 codebooks are measured NO-GO in the shipped band",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if not isinstance(self.scale_sweep, bool):
            raise TypeError(
                "CB scale_sweep identity must be an explicit bool, got "
                f"{self.scale_sweep!r}"
            )
        raw_sweep_scope = self.scale_sweep_scope
        if raw_sweep_scope is not None:
            sweep_scope = str(raw_sweep_scope).strip().lower()
            if sweep_scope not in {"none", "nvfp4", "fp8", "all"}:
                raise ValueError(
                    "CB scale_sweep_scope must be one of "
                    f"none/nvfp4/fp8/all, got {raw_sweep_scope!r}"
                )
            # The legacy bool records whether any family sweeps.  The scope is
            # the exact family selector for mixed recipes.
            object.__setattr__(self, "scale_sweep", sweep_scope != "none")
            object.__setattr__(self, "scale_sweep_scope", sweep_scope)
        if not isinstance(self.ldlq, bool):
            raise TypeError(
                "CB ldlq identity must be an explicit bool, got "
                f"{self.ldlq!r}"
            )
        # Per-family LDLQ scope: none (no family), nvfp4 (only NVFP4_CB), all (both families).
        # The legacy bool `ldlq` maps to all/none for backward compat; new code uses scope.
        allowed_scopes = {"none", "nvfp4", "all"}
        raw_scope = self.ldlq_scope
        if raw_scope is None:
            # Derive from legacy bool for old stamps / env that only set PRISMAQUANT_CB_LDLQ.
            derived = "all" if bool(self.ldlq) else "none"
            object.__setattr__(self, "ldlq_scope", derived)
        else:
            scope = str(raw_scope).strip().lower()
            if scope not in allowed_scopes:
                raise ValueError(f"CB ldlq_scope must be one of {sorted(allowed_scopes)}, got {raw_scope!r}")
            object.__setattr__(self, "ldlq_scope", scope)
            # Keep legacy bool consistent: true if any family uses LDLQ.
            expected_ldlq = scope != "none"
            if bool(self.ldlq) != expected_ldlq:
                # For mixed scope nvfp4, legacy bool is ambiguous; we normalize to True
                # if scope is nvfp4 (since at least one family is LDLQ). This keeps old
                # stamps that had ldlq=true for mixed assignments consistent.
                if scope == "nvfp4" and bool(self.ldlq) is False:
                    # Allow legacy false with scope nvfp4 only during rehydration from old stamp
                    # that had no scope; otherwise require consistency.
                    pass
                object.__setattr__(self, "ldlq", expected_ldlq)
        if not isinstance(self.minchain, bool):
            raise TypeError(
                "CB minchain identity must be an explicit bool, got "
                f"{self.minchain!r}"
            )
        if self.minchain:
            from .cb_minchain import MINCHAIN_CONTEXT_VERSION

            version = str(self.minchain_version or "").strip()
            if version != MINCHAIN_CONTEXT_VERSION:
                raise ValueError(
                    f"min-chain context requires minchain_version="
                    f"{MINCHAIN_CONTEXT_VERSION!r}, got {self.minchain_version!r}"
                )
            object.__setattr__(self, "minchain_version", version)
        elif self.minchain_version is not None:
            raise ValueError(
                "CB minchain_version cannot be set when minchain is disabled"
            )
        nvfp4_scale_sweep = (
            self.scale_sweep
            if self.scale_sweep_scope is None
            else self.scale_sweep_scope in {"nvfp4", "all"}
        )
        if coding == PRODUCTION_FP4_SCALE_CODING and not nvfp4_scale_sweep:
            raise ValueError(
                "CB layout-v2/two_tier requires scale_sweep=True; the "
                "two-tier scale encoder has no defined one-shot render"
            )
        tier = str(self.encode_tier).strip().lower()
        if tier not in CB_ENCODE_TIERS:
            raise ValueError(
                f"unknown CB encode_tier {self.encode_tier!r}; expected "
                f"{sorted(CB_ENCODE_TIERS)}"
            )
        renderer_abi = str(self.renderer_abi).strip()
        if renderer_abi != CB_RENDERER_ABI:
            raise ValueError(
                f"unsupported CB renderer_abi {renderer_abi!r}; rebuild "
                f"with {CB_RENDERER_ABI!r}"
            )
        object.__setattr__(self, "encode_tier", tier)
        object.__setattr__(self, "renderer_abi", renderer_abi)
        if self.activation_contract is not None:
            from .nvfp4_activation_contract import (
                NVFP4_ACTIVATION_CONTRACT_SCHEMA,
                NVFP4_ACTIVATION_EXECUTION,
            )

            activation_contract = str(self.activation_contract).strip()
            if activation_contract != NVFP4_ACTIVATION_CONTRACT_SCHEMA:
                raise ValueError(
                    "unsupported CB activation contract "
                    f"{activation_contract!r}; expected "
                    f"{NVFP4_ACTIVATION_CONTRACT_SCHEMA!r}"
                )
            object.__setattr__(
                self, "activation_contract", activation_contract
            )
            activation_execution = str(
                self.activation_execution or ""
            ).strip()
            if activation_execution != NVFP4_ACTIVATION_EXECUTION:
                raise ValueError(
                    "static CB activation contract requires "
                    f"activation_execution={NVFP4_ACTIVATION_EXECUTION!r}, "
                    f"got {self.activation_execution!r}"
                )
            object.__setattr__(
                self, "activation_execution", activation_execution
            )
        elif self.activation_execution is not None:
            raise ValueError(
                "CB activation_execution cannot be set without an "
                "activation_contract"
            )
        source_map = self.codebook_source_by_format
        if source_map is None and self.codebook_bundle_path is not None:
            bundle_path = str(self.codebook_bundle_path).strip()
            if bundle_path:
                # Exporters rebuild a context after selecting physical refs.
                # Loading the same immutable bundle here keeps that derived
                # context map-authoritative even when the call site predates
                # this optional field.
                from .cb_learned_bundle import load_bundle_cached

                source_map = load_bundle_cached(
                    bundle_path
                ).codebook_source_by_format
                object.__setattr__(
                    self, "codebook_source_by_format", source_map
                )
        if source_map is not None:
            normalized_sources: dict[str, str] = {}
            for raw_format, raw_source in (
                source_map.items()
            ):
                canonical = fr.get_format(str(raw_format)).name
                if not is_cb_format(canonical):
                    raise ValueError(
                        "CB codebook_source_by_format contains a non-CB "
                        f"format {raw_format!r}"
                    )
                cell_source = str(raw_source).strip().lower()
                if cell_source not in {"lattice", "learned"}:
                    raise ValueError(
                        f"CB source for {canonical} must be lattice/learned, "
                        f"got {raw_source!r}"
                    )
                if canonical in normalized_sources:
                    raise ValueError(
                        "CB codebook_source_by_format repeats canonical "
                        f"format {canonical!r}"
                    )
                normalized_sources[canonical] = cell_source
            if not normalized_sources:
                raise ValueError("CB codebook_source_by_format cannot be empty")
            normalized_sources = dict(sorted(normalized_sources.items()))
            object.__setattr__(
                self, "codebook_source_by_format", normalized_sources
            )
            learned_formats = [
                name
                for name, cell_source in normalized_sources.items()
                if cell_source == "learned"
            ]
            declared_source_scope = self.codebook_source_scope
            if declared_source_scope == "none" and learned_formats:
                raise ValueError(
                    "CB codebook_source_scope='none' cannot carry learned "
                    f"bundle cells: {learned_formats[:8]}"
                )
            if declared_source_scope == "fp8":
                learned_outside_scope = [
                    name
                    for name in learned_formats
                    if _cb_scope_family(name) != "fp8"
                ]
                if learned_outside_scope:
                    raise ValueError(
                        "CB codebook_source_scope='fp8' cannot carry learned "
                        "NVFP4 bundle cells: "
                        f"{learned_outside_scope[:8]}"
                    )
            # The old scalar is retained only as an artifact-wide ANY for
            # readers predating per-rung source identity.  A value-bearing
            # bundle map, not the build scope, decides it.
            object.__setattr__(
                self,
                "codebook_source",
                (
                    "learned"
                    if "learned" in normalized_sources.values()
                    else "lattice"
                ),
            )
        if self.codebook_refs is not None:
            normalized_refs: dict[str, str | tuple[str, ...]] = {}
            for qname, raw_refs in self.codebook_refs.items():
                normalized_refs[str(qname)] = (
                    str(raw_refs)
                    if isinstance(raw_refs, str)
                    else tuple(str(item) for item in raw_refs)
                )
            object.__setattr__(self, "codebook_refs", normalized_refs)
        if self.codebook_refs_by_qname_format is not None:
            normalized_refs_by_format: dict[
                str, dict[str, str | tuple[str, ...]]
            ] = {}
            for raw_qname, raw_formats in (
                self.codebook_refs_by_qname_format.items()
            ):
                qname = str(raw_qname)
                if not isinstance(raw_formats, Mapping):
                    raise TypeError(
                        "CB codebook_refs_by_qname_format entries must be "
                        f"format mappings, got {type(raw_formats).__name__} "
                        f"for {qname!r}"
                    )
                by_format: dict[str, str | tuple[str, ...]] = {}
                for raw_format, raw_refs in raw_formats.items():
                    canonical = fr.get_format(str(raw_format)).name
                    if parse_format_name(canonical) is None:
                        raise ValueError(
                            "CB codebook_refs_by_qname_format contains a "
                            f"non-CB format {raw_format!r} for {qname!r}"
                        )
                    if canonical in by_format:
                        raise ValueError(
                            "CB codebook_refs_by_qname_format repeats "
                            f"canonical format {canonical!r} for {qname!r}"
                        )
                    by_format[canonical] = (
                        str(raw_refs)
                        if isinstance(raw_refs, str)
                        else tuple(str(item) for item in raw_refs)
                    )
                normalized_refs_by_format[qname] = by_format
            object.__setattr__(
                self,
                "codebook_refs_by_qname_format",
                normalized_refs_by_format,
            )
        if self.codebook_content_digests is not None:
            normalized_digests: dict[str, str] = {}
            for raw_name, raw_digest in self.codebook_content_digests.items():
                name = str(raw_name)
                digest = str(raw_digest).strip().lower()
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError(
                        f"CB codebook digest for {name!r} is not a lowercase "
                        f"SHA-256 value: {raw_digest!r}"
                    )
                normalized_digests[name] = digest
            object.__setattr__(
                self, "codebook_content_digests", normalized_digests
            )
        if self.codebook_bundle_path is not None:
            bundle_path = str(self.codebook_bundle_path).strip()
            if not bundle_path:
                raise ValueError("CB codebook_bundle_path cannot be empty")
            object.__setattr__(self, "codebook_bundle_path", bundle_path)

    @classmethod
    def production(
        cls,
        *,
        scale_sweep: bool = True,
        ldlq: bool = False,
        ldlq_scope: str | None = None,
        minchain: bool = False,
        encode_tier: str = CB_ENCODE_TIER_DEFAULT,
        codebook_source: str = "lattice",
        codebook_source_scope: str | None = None,
        codebook_source_by_format: Mapping[str, str] | None = None,
        scale_sweep_scope: str | None = None,
        codebook_refs: Mapping[str, str | Sequence[str]] | None = None,
        codebook_refs_by_qname_format: Mapping[
            str, Mapping[str, str | Sequence[str]]
        ] | None = None,
        codebook_content_digests: Mapping[str, str] | None = None,
    ) -> CBSerializationContext:
        from .nvfp4_activation_contract import (
            NVFP4_ACTIVATION_CONTRACT_SCHEMA,
            NVFP4_ACTIVATION_EXECUTION,
        )
        from .cb_minchain import MINCHAIN_CONTEXT_VERSION

        # Scope is authoritative; ldlq bool is derived for backward compat.
        # Callers that pass ldlq_scope explicitly win; otherwise derive from ldlq.
        if ldlq_scope is not None:
            scope = str(ldlq_scope).strip().lower()
        else:
            scope = "all" if bool(ldlq) else "none"
        return cls(
            scale_coding=PRODUCTION_FP4_SCALE_CODING,
            layout_version=2,
            scale_sweep=scale_sweep,
            scale_sweep_scope=scale_sweep_scope,
            ldlq=bool(ldlq),
            ldlq_scope=scope,
            minchain=minchain,
            minchain_version=(MINCHAIN_CONTEXT_VERSION if minchain else None),
            encode_tier=encode_tier,
            codebook_source=codebook_source,
            codebook_source_scope=codebook_source_scope,
            codebook_source_by_format=codebook_source_by_format,
            activation_contract=NVFP4_ACTIVATION_CONTRACT_SCHEMA,
            activation_execution=NVFP4_ACTIVATION_EXECUTION,
            codebook_refs=codebook_refs,
            codebook_refs_by_qname_format=codebook_refs_by_qname_format,
            codebook_content_digests=codebook_content_digests,
        )

    @classmethod
    def legacy_v1(
        cls,
        *,
        scale_sweep: bool = True,
        ldlq: bool = False,
        ldlq_scope: str | None = None,
        minchain: bool = False,
        encode_tier: str = CB_ENCODE_TIER_DEFAULT,
        codebook_source: str = "lattice",
        codebook_source_scope: str | None = None,
        codebook_source_by_format: Mapping[str, str] | None = None,
        scale_sweep_scope: str | None = None,
        codebook_refs: Mapping[str, str | Sequence[str]] | None = None,
        codebook_refs_by_qname_format: Mapping[
            str, Mapping[str, str | Sequence[str]]
        ] | None = None,
        codebook_content_digests: Mapping[str, str] | None = None,
    ) -> CBSerializationContext:
        """Explicit legacy writer context; old artifacts remain readable."""
        if minchain:
            raise ValueError("min-chain has no legacy-v1 serialization contract")
        if ldlq_scope is not None:
            scope = str(ldlq_scope).strip().lower()
        else:
            scope = "all" if bool(ldlq) else "none"
        return cls(
            scale_coding=LEGACY_FP4_SCALE_CODING,
            layout_version=1,
            scale_sweep=scale_sweep,
            scale_sweep_scope=scale_sweep_scope,
            ldlq=bool(ldlq),
            ldlq_scope=scope,
            encode_tier=encode_tier,
            codebook_source=codebook_source,
            codebook_source_scope=codebook_source_scope,
            codebook_source_by_format=codebook_source_by_format,
            activation_contract=None,
            activation_execution=None,
            codebook_refs=codebook_refs,
            codebook_refs_by_qname_format=codebook_refs_by_qname_format,
            codebook_content_digests=codebook_content_digests,
        )


def _cb_scope_family(format_name: str) -> str:
    parsed = parse_format_name(str(format_name).strip().upper())
    if parsed is None:
        raise ValueError(f"{format_name!r} is not a CB format")
    family, _k = parsed
    return "nvfp4" if family.grid == "fp4" else "fp8"


def effective_codebook_source_scope(context: CBSerializationContext) -> str:
    """Return the canonical family scope carrying learned codebook bytes."""
    scope = context.codebook_source_scope
    if scope is None:
        return "all" if context.codebook_source == "learned" else "none"
    return str(scope)


def codebook_source_for_format(
    format_name: str,
    context: CBSerializationContext,
) -> str:
    """Resolve lattice/learned for one format under the artifact contract.

    A value-bearing bundle freezes this map when the render context is
    created.  The scope/policy path remains only for legacy logical contexts
    that have no bundle map; ``CB_CODEBOOK_SOURCE_SCOPE`` is a build-time
    training selector, not a render-time assertion about every FP8 rung.
    """

    canonical = fr.get_format(str(format_name)).name
    explicit_sources = context.codebook_source_by_format
    if explicit_sources is not None:
        try:
            return str(explicit_sources[canonical])
        except KeyError as exc:
            raise ValueError(
                f"{canonical}: value-bearing CB source map has no rung entry; "
                "refusing to infer one from current policy"
            ) from exc
    scope = effective_codebook_source_scope(context)
    if scope == "none":
        return "lattice"
    family = _cb_scope_family(canonical)
    if family == "fp8" and scope in {"fp8", "all"}:
        # CBL is a measured rendering policy, not a structural consequence of
        # the product-codebook split.  Disabled/pending rungs stay canonical
        # lattice even under an FP8 learned scope; this keeps K48 in the menu
        # without shipping the measured regression and lets K44--47 turn on
        # one row at a time as their sweep closes.
        from .cb_learned_bundle import CBL_RUNG_POLICY

        parsed = parse_format_name(canonical)
        assert parsed is not None
        _family, rung = parsed
        return (
            "learned"
            if CBL_RUNG_POLICY.get(int(rung), {}).get("enabled") is True
            else "lattice"
        )
    return "learned" if scope == "all" else "lattice"


def codebook_source_for_cell(
    qname: str,
    format_name: str,
    context: CBSerializationContext,
) -> str:
    """Resolve one render from its exact bundle cell when values are present."""

    canonical = fr.get_format(str(format_name)).name
    if context.codebook_bundle_path is None:
        return codebook_source_for_format(canonical, context)

    from .cb_learned_bundle import load_bundle_cached

    source = str(
        load_bundle_cached(context.codebook_bundle_path).cell(
            str(qname), canonical
        )["source"]
    )
    explicit_sources = context.codebook_source_by_format
    if explicit_sources is not None:
        expected = codebook_source_for_format(canonical, context)
        if source != expected:
            raise ValueError(
                f"{qname}/{canonical}: bundle cell source {source!r} differs "
                f"from the frozen per-rung source map {expected!r}"
            )
    return source


def effective_scale_sweep_scope(context: CBSerializationContext) -> str:
    """Return the canonical family scope on which scale search is enabled."""
    scope = context.scale_sweep_scope
    if scope is None:
        return "all" if context.scale_sweep else "none"
    return str(scope)


def scale_sweep_for_format(
    format_name: str,
    context: CBSerializationContext,
) -> bool:
    """Resolve the scale-search arm for one format under the scoped contract."""
    scope = effective_scale_sweep_scope(context)
    if scope == "none":
        return False
    if scope == "all":
        return True
    return _cb_scope_family(format_name) == scope


def _ldlq_for_format(format_name: str, context: CBSerializationContext) -> bool:
    """Per-family LDLQ decision for a single format under the scope contract.

    Scope ``none``: no family is LDLQ.
    Scope ``nvfp4``: only the fp4 family (NVFP4_CB) is LDLQ, fp8 stays raw.
    Scope ``all``: both families are LDLQ.
    """
    scope = str(getattr(context, "ldlq_scope", "none")).strip().lower()
    if scope == "none":
        return False
    if scope == "all":
        return True
    if scope == "nvfp4":
        info = _cb_info(str(format_name).strip().upper())
        if info is None:
            return False
        grid, _mode, _k = info
        return grid == "fp4"
    # Fallback to legacy bool for stamps that have no scope
    return bool(getattr(context, "ldlq", False))


def cb_serialization_context_stamp(
    context: CBSerializationContext,
    *,
    formats: Sequence[str | fr.FormatSpec] | None = None,
) -> dict:
    """Small identity stamp suitable for an allocator recipe's metadata."""
    if context is None:
        raise ValueError("CB serialization context stamp requires a context")
    source_scope = effective_codebook_source_scope(context)
    sweep_scope = effective_scale_sweep_scope(context)
    canonical_formats = (
        sorted({
            (
                item.name
                if isinstance(item, fr.FormatSpec)
                else fr.get_format(str(item)).name
            )
            for item in formats
            if is_cb_format(
                item.name if isinstance(item, fr.FormatSpec) else str(item)
            )
        })
        if formats is not None
        else None
    )
    source_formats = (
        sorted(context.codebook_source_by_format)
        if context.codebook_source_by_format is not None
        else canonical_formats
    )
    source_by_format = (
        {
            name: codebook_source_for_format(name, context)
            for name in source_formats
        }
        if source_formats
        and (
            context.codebook_source_by_format is not None
            or source_scope != "none"
        )
        else None
    )
    has_learned = (
        "learned" in source_by_format.values()
        if source_by_format is not None
        else context.codebook_source == "learned"
    )
    if has_learned and not context.codebook_content_digests:
        raise ValueError(
            "learned CB serialization identity requires materialized "
            "codebook_content_digests; a logical role/ref does not identify "
            "the bytes that cost, KL, and export must share"
        )
    lattice_formats = (
        [
            name for name in source_formats
            if codebook_source_for_format(name, context) == "lattice"
        ]
        if source_formats is not None
        else []
    )
    ldlq_scope = getattr(
        context, "ldlq_scope", "all" if context.ldlq else "none"
    )
    ldlq_kernel = None
    if ldlq_scope != "none":
        from .nvfp4_cb_formats import packed_ldlq_artifact_stamp

        ldlq_kernel = packed_ldlq_artifact_stamp()
    stamped_scalar_source = (
        "learned"
        if source_by_format is not None
        and "learned" in source_by_format.values()
        else (
            "lattice"
            if source_by_format is not None
            else context.codebook_source
        )
    )
    return {
        "schema": _serialized_payload_schema(context),
        "scale_coding": context.scale_coding,
        "layout_version": context.layout_version,
        "codebook_source": stamped_scalar_source,
        **({
            "codebook_source_scope": source_scope,
        } if (
            source_scope == "fp8"
            or (
                source_by_format is not None
                and source_scope != "none"
            )
        ) else {}),
        **({
            "codebook_source_by_format": source_by_format,
        } if source_by_format is not None else {}),
        "scale_sweep": context.scale_sweep,
        **({
            "scale_sweep_scope": sweep_scope,
        } if sweep_scope in {"nvfp4", "fp8"} else {}),
        "ldlq": context.ldlq,
        "ldlq_scope": ldlq_scope,
        **({"ldlq_packed_kernel": ldlq_kernel} if ldlq_kernel is not None else {}),
        **({
            "minchain": True,
            "minchain_version": context.minchain_version,
        } if context.minchain else {}),
        "encode_tier": context.encode_tier,
        "renderer_abi": context.renderer_abi,
        **({
            "activation_contract": context.activation_contract,
            "activation_execution": context.activation_execution,
        } if context.activation_contract is not None else {}),
        **({
            "lattice_codebook_sha256_by_format": {
                name: list(lattice_codebook_content_sha256(name))
                for name in lattice_formats
            },
        } if lattice_formats else {}),
        **({
            "codebook_refs": {
                str(name): (
                    str(refs)
                    if isinstance(refs, str)
                    else list(refs)
                )
                for name, refs in sorted(context.codebook_refs.items())
            },
        } if context.codebook_refs is not None else {}),
        **({
            "codebook_refs_by_qname_format": {
                str(name): {
                    str(format_name): (
                        str(refs)
                        if isinstance(refs, str)
                        else list(refs)
                    )
                    for format_name, refs in sorted(by_format.items())
                }
                for name, by_format in sorted(
                    context.codebook_refs_by_qname_format.items()
                )
            },
        } if context.codebook_refs_by_qname_format is not None else {}),
        **({
            "codebook_content_sha256": dict(sorted(
                context.codebook_content_digests.items()
            )),
        } if context.codebook_content_digests else {}),
    }


def cb_serialization_context_from_stamp(
    stamp: Mapping[str, object] | None,
    *,
    where: str,
) -> CBSerializationContext:
    """Rehydrate a context stamp, rejecting partial or stale identities.

    A CB assignment cannot be priced from a format label alone: FP4 layout
    version and codebook sharing both affect serialized bytes.  Consumers of
    persisted assignments therefore use this strict inverse instead of
    filling absent fields with whichever defaults happen to be current.
    """
    if not isinstance(stamp, Mapping):
        raise ValueError(
            f"{where}: CB assignment is missing its serialized-payload "
            "context stamp"
        )
    schema = stamp.get("schema")
    if schema == LEGACY_CB_SERIALIZED_PAYLOAD_SCHEMA:
        raise ValueError(
            f"{where}: legacy CB serialized-payload v1 stamp has no exact "
            "codebook-content identity; rebuild the cost/cache/assignment "
            "with the current producer contract"
        )
    if schema not in {
        PREVIOUS_CB_SERIALIZED_PAYLOAD_SCHEMA,
        CB_SERIALIZED_PAYLOAD_SCHEMA,
        MINCHAIN_CB_SERIALIZED_PAYLOAD_SCHEMA,
    }:
        raise ValueError(
            f"{where}: unsupported CB serialized-payload schema "
            f"{stamp.get('schema')!r}"
        )
    missing = [
        key for key in (
            "scale_coding",
            "layout_version",
            "codebook_source",
            "scale_sweep",
            "ldlq",
            "encode_tier",
            "renderer_abi",
        )
        if stamp.get(key) is None
    ]
    if missing:
        raise ValueError(
            f"{where}: CB serialized-payload context is missing {missing}"
        )
    activation_contract = stamp.get("activation_contract")
    activation_execution = stamp.get("activation_execution")
    current_schemas = {
        CB_SERIALIZED_PAYLOAD_SCHEMA,
        MINCHAIN_CB_SERIALIZED_PAYLOAD_SCHEMA,
    }
    if schema in current_schemas and activation_contract is None:
        raise ValueError(
            f"{where}: current CB serialized-payload stamp is missing its "
            "activation_contract"
        )
    if schema in current_schemas and activation_execution is None:
        raise ValueError(
            f"{where}: current CB serialized-payload stamp is missing its "
            "activation_execution"
        )
    if (
        schema == PREVIOUS_CB_SERIALIZED_PAYLOAD_SCHEMA
        and (activation_contract is not None or activation_execution is not None)
    ):
        raise ValueError(
            f"{where}: CB serialized-payload v2 stamp cannot claim an "
            "activation_contract"
        )
    raw_source_scope = stamp.get("codebook_source_scope")
    if raw_source_scope is not None:
        source_scope = str(raw_source_scope).strip().lower()
        if source_scope not in {"none", "fp8", "all"}:
            raise ValueError(
                f"{where}: CB codebook_source_scope must be one of "
                f"none/fp8/all, got {raw_source_scope!r}"
            )
    else:
        source_scope = None
    raw_source_by_format = stamp.get("codebook_source_by_format")
    if raw_source_by_format is not None and not isinstance(
        raw_source_by_format, Mapping
    ):
        raise ValueError(
            f"{where}: CB codebook_source_by_format stamp is not an object"
        )
    source_by_format: dict[str, str] | None = None
    if isinstance(raw_source_by_format, Mapping):
        source_by_format = {}
        for raw_format, raw_source in raw_source_by_format.items():
            canonical = fr.get_format(str(raw_format)).name
            if not is_cb_format(canonical):
                raise ValueError(
                    f"{where}: source map contains non-CB format "
                    f"{raw_format!r}"
                )
            source = str(raw_source).strip().lower()
            if source not in {"lattice", "learned"}:
                raise ValueError(
                    f"{where}: source map value for {canonical} must be "
                    f"lattice/learned, got {raw_source!r}"
                )
            if canonical in source_by_format:
                raise ValueError(
                    f"{where}: source map repeats canonical format "
                    f"{canonical!r}"
                )
            source_by_format[canonical] = source
        if not source_by_format:
            raise ValueError(f"{where}: CB source map cannot be empty")
        expected_scalar_source = (
            "learned"
            if "learned" in source_by_format.values()
            else "lattice"
        )
        observed_scalar_source = str(stamp["codebook_source"]).strip().lower()
        if observed_scalar_source != expected_scalar_source:
            raise ValueError(
                f"{where}: CB codebook_source scalar must be the artifact-wide "
                "ANY of codebook_source_by_format: "
                f"stamp={observed_scalar_source!r}, "
                f"map={expected_scalar_source!r}"
            )
    raw_sweep_scope = stamp.get("scale_sweep_scope")
    if raw_sweep_scope is not None:
        sweep_scope = str(raw_sweep_scope).strip().lower()
        if sweep_scope not in {"none", "nvfp4", "fp8", "all"}:
            raise ValueError(
                f"{where}: CB scale_sweep_scope must be one of "
                f"none/nvfp4/fp8/all, got {raw_sweep_scope!r}"
            )
    else:
        sweep_scope = None
    raw_digests = stamp.get("codebook_content_sha256")
    has_learned = (
        "learned" in source_by_format.values()
        if source_by_format is not None
        else (
            source_scope in {"fp8", "all"}
            if source_scope is not None
            else str(stamp["codebook_source"]).strip().lower() == "learned"
        )
    )
    if has_learned and not isinstance(raw_digests, Mapping):
        raise ValueError(
            f"{where}: learned CB serialized-payload context is missing "
            "codebook_content_sha256"
        )
    raw_lattice_digests = stamp.get("lattice_codebook_sha256_by_format")
    if raw_lattice_digests is not None and not isinstance(
        raw_lattice_digests, Mapping
    ):
        raise ValueError(
            f"{where}: lattice codebook digest stamp is not an object"
        )
    if isinstance(raw_lattice_digests, Mapping):
        for raw_format, raw_values in raw_lattice_digests.items():
            canonical = fr.get_format(str(raw_format)).name
            if not is_cb_format(canonical) or not isinstance(
                raw_values, Sequence
            ) or isinstance(raw_values, (str, bytes)):
                raise ValueError(
                    f"{where}: invalid lattice codebook digest entry for "
                    f"{raw_format!r}"
                )
            observed_values = tuple(str(value) for value in raw_values)
            expected_values = lattice_codebook_content_sha256(canonical)
            if observed_values != expected_values:
                raise ValueError(
                    f"{where}: lattice codebook digest entry for {canonical} "
                    "does not match canonical serialized bytes"
                )
    raw_refs = stamp.get("codebook_refs")
    if raw_refs is not None and not isinstance(raw_refs, Mapping):
        raise ValueError(f"{where}: CB codebook_refs stamp is not an object")
    raw_refs_by_format = stamp.get("codebook_refs_by_qname_format")
    if raw_refs_by_format is not None and not isinstance(
        raw_refs_by_format, Mapping
    ):
        raise ValueError(
            f"{where}: CB codebook_refs_by_qname_format stamp is not an object"
        )
    if isinstance(raw_refs_by_format, Mapping) and any(
        not isinstance(value, Mapping)
        for value in raw_refs_by_format.values()
    ):
        raise ValueError(
            f"{where}: CB codebook_refs_by_qname_format entries must be objects"
        )
    minchain = schema == MINCHAIN_CB_SERIALIZED_PAYLOAD_SCHEMA
    if minchain and stamp.get("minchain") is not True:
        raise ValueError(
            f"{where}: min-chain v4 stamp is missing minchain=true"
        )
    if minchain and stamp.get("minchain_version") is None:
        raise ValueError(
            f"{where}: min-chain v4 stamp is missing minchain_version"
        )
    if not minchain and (
        stamp.get("minchain") is not None
        or stamp.get("minchain_version") is not None
    ):
        raise ValueError(
            f"{where}: pre-min-chain stamp cannot claim min-chain fields"
        )
    # Per-family scope: new stamps carry ldlq_scope, old stamps only have ldlq bool.
    raw_scope = stamp.get("ldlq_scope")
    if raw_scope is None:
        ldlq_scope = "all" if bool(stamp["ldlq"]) else "none"
    else:
        ldlq_scope = str(raw_scope).strip().lower()
        if ldlq_scope not in {"none", "nvfp4", "all"}:
            raise ValueError(f"{where}: CB ldlq_scope must be one of none/nvfp4/all, got {raw_scope!r}")
    observed_ldlq_kernel = stamp.get("ldlq_packed_kernel")
    if ldlq_scope == "none":
        if observed_ldlq_kernel is not None:
            raise ValueError(
                f"{where}: non-LDLQ payload cannot claim an LDLQ packed kernel"
            )
    else:
        from .nvfp4_cb_formats import packed_ldlq_artifact_stamp

        expected_ldlq_kernel = packed_ldlq_artifact_stamp()
        if not isinstance(observed_ldlq_kernel, Mapping):
            raise ValueError(
                f"{where}: LDLQ payload is missing its packed-kernel ABI"
            )
        if dict(observed_ldlq_kernel) != expected_ldlq_kernel:
            raise ValueError(
                f"{where}: LDLQ packed-kernel ABI mismatch: "
                f"recipe={dict(observed_ldlq_kernel)}, "
                f"producer={expected_ldlq_kernel}"
            )
    return CBSerializationContext(
        scale_coding=str(stamp["scale_coding"]),
        layout_version=int(stamp["layout_version"]),
        codebook_source=str(stamp["codebook_source"]),
        codebook_source_scope=source_scope,
        codebook_source_by_format=source_by_format,
        scale_sweep=stamp["scale_sweep"],
        scale_sweep_scope=sweep_scope,
        ldlq=stamp["ldlq"],
        ldlq_scope=ldlq_scope,
        minchain=minchain,
        minchain_version=(str(stamp["minchain_version"]) if minchain else None),
        encode_tier=str(stamp["encode_tier"]),
        renderer_abi=str(stamp["renderer_abi"]),
        activation_contract=(
            str(activation_contract)
            if activation_contract is not None
            else None
        ),
        activation_execution=(
            str(activation_execution)
            if activation_execution is not None
            else None
        ),
        codebook_refs=(
            {str(name): value for name, value in raw_refs.items()}
            if isinstance(raw_refs, Mapping)
            else None
        ),
        codebook_refs_by_qname_format=(
            {
                str(name): {
                    str(format_name): refs
                    for format_name, refs in by_format.items()
                }
                for name, by_format in raw_refs_by_format.items()
            }
            if isinstance(raw_refs_by_format, Mapping)
            else None
        ),
        codebook_content_digests=(
            {str(name): str(value) for name, value in raw_digests.items()}
            if isinstance(raw_digests, Mapping)
            else None
        ),
    )


def cb_serialization_context_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    require_explicit: bool = False,
    where: str = "CB cost render",
) -> CBSerializationContext:
    """Resolve the producer identity used by a CB render.

    Production defaults are the v3 static-activation contract over the
    layout-v2/lattice weight payload. ``CB_ACTIVATION_SCOPE=none`` selects the
    existing v2 no-activation contract for a closed FP8-only producer; the
    historical mixed-family default remains ``nvfp4``. Pipeline stages pass
    the choices explicitly and stamp them into their cache provenance; the
    defaults keep direct library calls aligned with the production writer.
    Callers that consume an existing artifact use ``require_explicit=True`` so
    absence is never mistaken for proof that a cache used today's defaults.
    """
    if environ is None:
        import os

        environ = os.environ
    scale = environ.get("CB_SCALE_CODING")
    source = environ.get("CB_CODEBOOK_SOURCE")
    raw_source_scope = environ.get("CB_CODEBOOK_SOURCE_SCOPE")
    raw_sweep = environ.get("CB_SCALE_SWEEP")
    raw_sweep_scope = environ.get("CB_SCALE_SWEEP_SCOPE")
    raw_ldlq = environ.get("PRISMAQUANT_CB_LDLQ")
    raw_ldlq_scope = environ.get("PRISMAQUANT_CB_LDLQ_SCOPE")
    raw_minchain = environ.get("PRISMAQUANT_CB_MINCHAIN")
    raw_tier = environ.get("PRISMAQUANT_CB_ENCODE_TIER")
    raw_activation_scope = environ.get("CB_ACTIVATION_SCOPE")
    bundle_source = str(environ.get("CB_CODEBOOK_BUNDLE", "")).strip()
    if require_explicit and (
        not scale
        or (not source and raw_source_scope is None)
        or (raw_sweep is None and raw_sweep_scope is None)
        or (raw_ldlq is None and raw_ldlq_scope is None)
        or not raw_tier
    ):
        missing = [
            name for name, value in (
                ("CB_SCALE_CODING", scale),
                (
                    "CB_CODEBOOK_SOURCE_SCOPE/CB_CODEBOOK_SOURCE",
                    raw_source_scope or source,
                ),
                ("CB_SCALE_SWEEP_SCOPE/CB_SCALE_SWEEP", raw_sweep_scope or raw_sweep),
                ("PRISMAQUANT_CB_LDLQ_SCOPE/PRISMAQUANT_CB_LDLQ", raw_ldlq_scope or raw_ldlq),
                ("PRISMAQUANT_CB_ENCODE_TIER", raw_tier),
            ) if not value
        ]
        raise ValueError(
            f"{where}: missing explicit CB producer setting(s) {missing}"
        )
    digest_source = environ.get("CB_CODEBOOK_DIGESTS")
    digests = (
        load_cb_codebook_digest_manifest(digest_source, where=where)
        if digest_source
        else None
    )
    scale_sweep = _parse_bool_setting(
        raw_sweep,
        default=True,
        name="CB_SCALE_SWEEP",
        where=where,
    )
    source_scope = (
        str(raw_source_scope).strip().lower()
        if raw_source_scope is not None
        else None
    )
    if source_scope is not None:
        if source_scope not in {"none", "fp8", "all"}:
            raise ValueError(
                f"{where}: CB_CODEBOOK_SOURCE_SCOPE must be one of "
                f"none/fp8/all, got {raw_source_scope!r}"
            )
        # Without a bundle there is no per-rung artifact to consult, so keep
        # the legacy scalar/scope consistency error (and the missing-bundle
        # diagnostic below).  With a bundle, compare the scalar to the
        # bundle's artifact-wide ANY after loading its cell map instead.
        if not bundle_source:
            expected_source = (
                "lattice" if source_scope == "none" else "learned"
            )
            if (
                source is not None
                and str(source).strip().lower() != expected_source
            ):
                raise ValueError(
                    f"{where}: CB_CODEBOOK_SOURCE={source!r} is "
                    "inconsistent with "
                    f"CB_CODEBOOK_SOURCE_SCOPE={raw_source_scope!r}"
                )
            source = expected_source
    bundle_requested = (
        source_scope in {"fp8", "all"}
        if source_scope is not None
        else str(source or "lattice").strip().lower() == "learned"
    )
    refs_by_qname_format = None
    source_by_format = None
    if bundle_requested:
        if not bundle_source:
            raise ValueError(
                f"{where}: learned CB requires CB_CODEBOOK_BUNDLE pointing "
                "at the immutable value-bearing .pqcb; a digest manifest or "
                "logical ref cannot supply the values cost/cache/KL/export "
                "must share"
            )
        if digest_source:
            raise ValueError(
                f"{where}: set CB_CODEBOOK_BUNDLE, not the legacy "
                "digest-only CB_CODEBOOK_DIGESTS contract, for learned CB"
            )
        from .cb_learned_bundle import load_bundle_cached

        bundle = load_bundle_cached(bundle_source)
        digests = dict(bundle.codebook_content_digests)
        source_by_format = bundle.codebook_source_by_format
        bundle_scalar = (
            "learned"
            if "learned" in source_by_format.values()
            else "lattice"
        )
        if (
            source is not None
            and str(source).strip().lower() != bundle_scalar
        ):
            raise ValueError(
                f"{where}: CB_CODEBOOK_SOURCE={source!r} is inconsistent "
                "with the value-bearing bundle's per-rung source map "
                f"(artifact-wide source={bundle_scalar!r})"
            )
        source = bundle_scalar
        refs_by_qname_format = {
            str(qname): dict(by_format)
            for qname, by_format in bundle.codebook_refs_by_cell.items()
        }
    elif bundle_source:
        raise ValueError(
            f"{where}: CB_CODEBOOK_BUNDLE is set while the effective learned "
            "scope is none; refusing a stale value-bearing input that the "
            "stamp would otherwise ignore"
        )
    sweep_scope = (
        str(raw_sweep_scope).strip().lower()
        if raw_sweep_scope is not None
        else None
    )
    if sweep_scope is not None:
        if sweep_scope not in {"none", "nvfp4", "fp8", "all"}:
            raise ValueError(
                f"{where}: CB_SCALE_SWEEP_SCOPE must be one of "
                f"none/nvfp4/fp8/all, got {raw_sweep_scope!r}"
            )
        expected_sweep = sweep_scope != "none"
        if raw_sweep is not None and scale_sweep != expected_sweep:
            raise ValueError(
                f"{where}: CB_SCALE_SWEEP={raw_sweep!r} is inconsistent with "
                f"CB_SCALE_SWEEP_SCOPE={raw_sweep_scope!r}"
            )
        scale_sweep = expected_sweep
    # New per-family scope is authoritative; legacy bool maps to all/none.
    raw_ldlq_scope_clean = str(raw_ldlq_scope).strip().lower() if raw_ldlq_scope is not None else None
    if raw_ldlq_scope_clean is not None:
        if raw_ldlq_scope_clean not in {"none", "nvfp4", "all"}:
            raise ValueError(
                f"{where}: PRISMAQUANT_CB_LDLQ_SCOPE must be one of none/nvfp4/all, got {raw_ldlq_scope!r}"
            )
        ldlq_scope = raw_ldlq_scope_clean
        # Legacy bool, if also set, must be consistent with scope's ANY.
        if raw_ldlq is not None:
            legacy_ldlq = _parse_bool_setting(
                raw_ldlq, default=False, name="PRISMAQUANT_CB_LDLQ", where=where,
            )
            expected_legacy = ldlq_scope != "none"
            if legacy_ldlq != expected_legacy and not (
                # Allow legacy true with scope nvfp4 (mixed) for backward compat;
                # the legacy bool is ambiguous for mixed case.
                ldlq_scope == "nvfp4" and legacy_ldlq is True
            ):
                raise ValueError(
                    f"{where}: PRISMAQUANT_CB_LDLQ={raw_ldlq!r} inconsistent with "
                    f"PRISMAQUANT_CB_LDLQ_SCOPE={raw_ldlq_scope!r}"
                )
        ldlq = ldlq_scope != "none"
    else:
        ldlq = _parse_bool_setting(
            raw_ldlq,
            default=False,
            name="PRISMAQUANT_CB_LDLQ",
            where=where,
        )
        ldlq_scope = "all" if ldlq else "none"
    minchain = _parse_bool_setting(
        raw_minchain,
        default=False,
        name="PRISMAQUANT_CB_MINCHAIN",
        where=where,
    )
    from .cb_minchain import MINCHAIN_CONTEXT_VERSION
    from .nvfp4_activation_contract import (
        NVFP4_ACTIVATION_CONTRACT_SCHEMA,
        NVFP4_ACTIVATION_EXECUTION,
    )
    activation_scope = str(raw_activation_scope or "nvfp4").strip().lower()
    if activation_scope not in {"none", "nvfp4"}:
        raise ValueError(
            f"{where}: CB_ACTIVATION_SCOPE must be none or nvfp4, got "
            f"{raw_activation_scope!r}"
        )

    return CBSerializationContext(
        scale_coding=scale or PRODUCTION_FP4_SCALE_CODING,
        codebook_source=source or "lattice",
        codebook_source_scope=source_scope,
        codebook_source_by_format=source_by_format,
        scale_sweep=scale_sweep,
        scale_sweep_scope=sweep_scope,
        ldlq=ldlq,
        ldlq_scope=ldlq_scope,
        minchain=minchain,
        minchain_version=(MINCHAIN_CONTEXT_VERSION if minchain else None),
        encode_tier=resolve_cb_encode_tier(raw_tier, environ=environ),
        activation_contract=(
            None
            if activation_scope == "none"
            else NVFP4_ACTIVATION_CONTRACT_SCHEMA
        ),
        activation_execution=(
            None
            if activation_scope == "none"
            else NVFP4_ACTIVATION_EXECUTION
        ),
        codebook_refs_by_qname_format=refs_by_qname_format,
        codebook_content_digests=digests,
        codebook_bundle_path=(bundle_source if bundle_requested else None),
    )


def _parse_bool_setting(
    raw: object,
    *,
    default: bool,
    name: str,
    where: str,
) -> bool:
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{where}: {name} must be a boolean 0/1 setting, got {raw!r}"
    )


def resolve_cb_encode_tier(
    raw: object | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the encoder tier once so every render receives it explicitly."""
    if raw is None:
        if environ is None:
            import os

            environ = os.environ
        raw = environ.get("PRISMAQUANT_CB_ENCODE_TIER")
    tier = str(raw or CB_ENCODE_TIER_DEFAULT).strip().lower()
    if tier not in CB_ENCODE_TIERS:
        raise ValueError(
            f"unknown CB encode tier {tier!r}; expected "
            f"{sorted(CB_ENCODE_TIERS)}"
        )
    return tier


def load_cb_codebook_digest_manifest(
    source: str | Path,
    *,
    where: str,
) -> dict[str, str]:
    """Load a strict JSON digest object from a path or inline object text."""
    text = str(source)
    if text.lstrip().startswith("{"):
        raw_text = text
        source_label = "inline CB_CODEBOOK_DIGESTS"
    else:
        path = Path(text)
        raw_text = path.read_text()
        source_label = str(path)
    raw = _strict_json_loads(raw_text, where=source_label)
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"{where}: CB codebook digest manifest must be a JSON object"
        )
    return {str(name): str(value) for name, value in raw.items()}


def validate_cb_serialization_context_stamp(
    stamp: Mapping[str, object] | None,
    context: CBSerializationContext,
    *,
    where: str,
    formats: Sequence[str | fr.FormatSpec] | None = None,
) -> None:
    """Fail when export choices drift from an allocator-stamped recipe.

    Hand-written and historical recipes have no stamp and remain loadable.
    Once a producer stamps the identity, however, ignoring a mismatch would
    make allocation bytes describe a different artifact than the exporter.
    """
    if stamp is None:
        return
    if not isinstance(stamp, Mapping):
        raise TypeError(f"{where}: CB serialized-payload stamp is not an object")
    observed_context = cb_serialization_context_from_stamp(stamp, where=where)
    expected = cb_serialization_context_stamp(context, formats=formats)
    base_keys = (
        "schema",
        "scale_coding",
        "layout_version",
        "codebook_source",
        "scale_sweep",
        "ldlq",
        "encode_tier",
        "renderer_abi",
        "activation_contract",
        "activation_execution",
    )
    if context.minchain:
        base_keys += ("minchain", "minchain_version")
    observed = {key: stamp.get(key) for key in base_keys}
    expected_base = {key: expected.get(key) for key in base_keys}
    if observed != expected_base:
        raise ValueError(
            f"{where}: CB serialization context differs from allocator "
            f"recipe: recipe={observed}, exporter={expected_base}"
        )
    observed_source_scope = effective_codebook_source_scope(observed_context)
    expected_source_scope = effective_codebook_source_scope(context)
    if observed_source_scope != expected_source_scope:
        raise ValueError(
            f"{where}: CB codebook source scope differs from allocator recipe: "
            f"recipe={observed_source_scope!r}, "
            f"exporter={expected_source_scope!r}"
        )
    observed_source_by_format = stamp.get("codebook_source_by_format")
    expected_source_by_format = None
    if context.codebook_source_by_format is not None:
        expected_source_by_format = dict(context.codebook_source_by_format)
    elif formats is not None:
        expected_formats = sorted({
            (
                item.name
                if isinstance(item, fr.FormatSpec)
                else fr.get_format(str(item)).name
            )
            for item in formats
            if is_cb_format(
                item.name if isinstance(item, fr.FormatSpec) else str(item)
            )
        })
        if expected_source_scope != "none" and expected_formats:
            expected_source_by_format = {
                name: codebook_source_for_format(name, context)
                for name in expected_formats
            }
    if expected_source_by_format is not None:
        observed_source_map = (
            {
                fr.get_format(str(name)).name: str(source)
                for name, source in observed_source_by_format.items()
            }
            if isinstance(observed_source_by_format, Mapping)
            else None
        )
        if observed_source_map != expected_source_by_format:
            observed_keys = (
                set(observed_source_map) if observed_source_map is not None else set()
            )
            expected_keys = set(expected_source_by_format)
            changed = sorted(
                name for name in observed_keys & expected_keys
                if observed_source_map[name] != expected_source_by_format[name]
            )
            raise ValueError(
                f"{where}: CB per-rung codebook source map differs from the "
                "render: "
                f"missing={sorted(expected_keys - observed_keys)[:8]}, "
                f"extra={sorted(observed_keys - expected_keys)[:8]}, "
                f"changed={changed[:8]}"
            )
    elif (
        context.codebook_source_scope is not None
        and expected_source_scope != "none"
    ):
        if not isinstance(observed_source_by_format, Mapping):
            raise ValueError(
                f"{where}: scoped learned CB recipe has no checked "
                "codebook_source_by_format rung policy"
            )
        mismatched_source_formats = [
            str(format_name)
            for format_name, observed_source in observed_source_by_format.items()
            if str(observed_source) != codebook_source_for_format(
                str(format_name), context
            )
        ]
        if mismatched_source_formats:
            raise ValueError(
                f"{where}: CB per-format learned rung policy differs from the "
                f"producer: {mismatched_source_formats[:8]}"
            )
    observed_sweep_scope = effective_scale_sweep_scope(observed_context)
    expected_sweep_scope = effective_scale_sweep_scope(context)
    if observed_sweep_scope != expected_sweep_scope:
        raise ValueError(
            f"{where}: CB scale-sweep scope differs from allocator recipe: "
            f"recipe={observed_sweep_scope!r}, "
            f"exporter={expected_sweep_scope!r}"
        )
    # Per-family scope: new stamps carry ldlq_scope, old stamps only have ldlq bool.
    # For backward compat, an old stamp without scope is interpreted as all/none
    # from its ldlq bool. A scope mismatch is a real producer/consumer drift.
    expected_scope = getattr(context, "ldlq_scope", "all" if context.ldlq else "none")
    observed_scope = stamp.get("ldlq_scope")
    if observed_scope is None:
        # Legacy stamp: derive from its ldlq bool.
        observed_ldlq = stamp.get("ldlq")
        if isinstance(observed_ldlq, bool):
            observed_scope = "all" if observed_ldlq else "none"
        else:
            observed_scope = None
    if observed_scope is not None and observed_scope != expected_scope:
        raise ValueError(
            f"{where}: CB serialization scope differs from allocator recipe: "
            f"recipe ldlq_scope={observed_scope!r}, exporter ldlq_scope={expected_scope!r}"
        )
    observed_refs = stamp.get("codebook_refs")
    if context.codebook_refs is not None and isinstance(
        observed_refs, Mapping
    ):
        missing_refs = sorted(set(context.codebook_refs) - set(observed_refs))
        mismatched_refs = sorted(
            name
            for name, refs in context.codebook_refs.items()
            if name in observed_refs
            and (
                (str(observed_refs[name]),)
                if isinstance(observed_refs[name], str)
                else tuple(str(item) for item in observed_refs[name])
            )
            != (
                (str(refs),)
                if isinstance(refs, str)
                else tuple(str(item) for item in refs)
            )
        )
        if missing_refs or mismatched_refs:
            raise ValueError(
                f"{where}: CB physical codebook refs differ from allocator "
                f"recipe: missing={missing_refs[:8]}, "
                f"mismatched={mismatched_refs[:8]}"
            )
    observed_refs_by_format = stamp.get("codebook_refs_by_qname_format")
    if context.codebook_refs_by_qname_format is not None:
        if not isinstance(observed_refs_by_format, Mapping):
            raise ValueError(
                f"{where}: CB recipe has no per-qname/format codebook refs"
            )
        missing_by_format: list[str] = []
        mismatched_by_format: list[str] = []
        for qname, expected_formats in (
            context.codebook_refs_by_qname_format.items()
        ):
            observed_formats = observed_refs_by_format.get(qname)
            if not isinstance(observed_formats, Mapping):
                missing_by_format.append(qname)
                continue
            for format_name, refs in expected_formats.items():
                label = f"{qname}/{format_name}"
                if format_name not in observed_formats:
                    missing_by_format.append(label)
                    continue
                observed_value = observed_formats[format_name]
                observed_tuple = (
                    (str(observed_value),)
                    if isinstance(observed_value, str)
                    else tuple(str(item) for item in observed_value)
                )
                expected_tuple = (
                    (str(refs),)
                    if isinstance(refs, str)
                    else tuple(str(item) for item in refs)
                )
                if observed_tuple != expected_tuple:
                    mismatched_by_format.append(label)
        if missing_by_format or mismatched_by_format:
            raise ValueError(
                f"{where}: CB per-qname/format codebook refs differ from "
                "allocator recipe: "
                f"missing={missing_by_format[:8]}, "
                f"mismatched={mismatched_by_format[:8]}"
            )
    has_learned_codebook = (
        any(source == "learned" for source in expected_source_by_format.values())
        if expected_source_by_format is not None
        else effective_codebook_source_scope(context) != "none"
    )
    if has_learned_codebook:
        observed_digests = stamp.get("codebook_content_sha256")
        if not isinstance(observed_digests, Mapping):
            raise ValueError(
                f"{where}: learned CB recipe has no materialized codebook "
                "content digests"
            )
        expected_digests = context.codebook_content_digests or {}
        missing = sorted(set(expected_digests) - set(observed_digests))
        mismatched = sorted(
            name
            for name, digest in expected_digests.items()
            if str(observed_digests.get(name, "")).lower() != digest
        )
        if missing or mismatched:
            raise ValueError(
                f"{where}: learned CB serialization context differs from "
                "allocator recipe: relevant materialized digest "
                f"missing={missing[:8]}, mismatched={mismatched[:8]}"
            )


def cb_cost_provenance(
    formats: Sequence[str | fr.FormatSpec],
    *,
    context: CBSerializationContext | None = None,
) -> dict:
    """Return the CB identity fragment for a measured-cost payload."""
    names = [item.name if isinstance(item, fr.FormatSpec) else str(item)
             for item in formats]
    if not any(is_cb_format(name) for name in names):
        return {}
    ctx = context or cb_serialization_context_from_env()
    return {
        "cb_serialized_payload": cb_serialization_context_stamp(
            ctx,
            formats=names,
        ),
    }


def validate_cb_cost_provenance(
    payload: Mapping[str, object],
    formats: Sequence[str | fr.FormatSpec],
    *,
    context: CBSerializationContext,
    where: str,
) -> None:
    """Fail closed when a CB cost table lacks or mismatches render identity."""
    names = [item.name if isinstance(item, fr.FormatSpec) else str(item)
             for item in formats]
    if not any(is_cb_format(name) for name in names):
        return
    provenance = payload.get("provenance")
    stamp = (
        provenance.get("cb_serialized_payload")
        if isinstance(provenance, Mapping)
        else None
    )
    if stamp is None:
        raise ValueError(
            f"{where}: CB cost payload has no serialized-payload identity; "
            "refusing a cache whose scale layout/codebook source is unknown"
        )
    validate_cb_serialization_context_stamp(
        stamp,
        context,
        where=where,
        formats=names,
    )
    expected_lattice = {
        fr.get_format(name).name: list(
            lattice_codebook_content_sha256(fr.get_format(name).name)
        )
        for name in names
        if is_cb_format(name)
        and codebook_source_for_format(name, context) == "lattice"
    }
    if expected_lattice:
        observed_lattice = stamp.get("lattice_codebook_sha256_by_format")
        if not isinstance(observed_lattice, Mapping) or any(
            observed_lattice.get(name) != digests
            for name, digests in expected_lattice.items()
        ):
            raise ValueError(
                f"{where}: CB cost payload does not identify the exact "
                "canonical lattice bytes for its measured format menu"
            )


def _cb_info(format_name: str) -> tuple[str, str, int] | None:
    """Return ``(grid, mode, k)`` for a registered CB format."""
    canonical = str(format_name).strip().upper()
    parsed = parse_format_name(canonical)
    if parsed is None:
        return None
    try:
        registered = fr.get_format(canonical)
    except KeyError:
        return None
    if str(registered.name).strip().upper() != canonical:
        return None
    family, k = parsed
    return family.grid, family.mode, k


def is_cb_format(format_name: str) -> bool:
    return _cb_info(format_name) is not None


@lru_cache(maxsize=None)
def lattice_codebook_content_sha256(format_name: str) -> tuple[str, ...]:
    """Exact FP16 payload digests for the canonical lattice sidecar tables."""
    canonical = str(format_name).strip().upper()
    info = _cb_info(canonical)
    if info is None:
        raise ValueError(f"{format_name!r} is not a CB format")
    grid, mode, k = info
    import torch

    from . import nvfp4_cb_formats as cb

    if mode == "product":
        n_sub = family_for(grid, mode).n_sub
        sub_dim = _VEC_DIM // n_sub
        tables = tuple(
            cb.fixed_lattice(bits, grid, sub_dim)
            for bits in subtable_bit_widths(k, mode, n_sub)
        )
    else:
        tables = (cb.fixed_lattice(k, grid, _VEC_DIM),)
    return tuple(
        hashlib.sha256(
            tensor.to(torch.float16).cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for tensor in tables
    )


def cb_fields_for_context(
    spec: fr.FormatSpec,
    weight,
    *,
    context: CBSerializationContext,
    qname: str | None = None,
    col_weights=None,
    codebook=None,
    activation_rows=None,
    warm_scale_state=None,
    raw_fields_out: MutableMapping[str, object] | None = None,
    ldlq_missing_activation_ok: bool = False,
):
    """Encode CB fields under the exact artifact serialization context.

    The v1 and v2 FP4 layouts have different reachable scale sets, so this is
    a correctness contract, not merely a byte-pricing option.  FP8-CB has no
    FP4 scale plane; it still requires the context so the measured result is
    stamped with the same codebook-sharing identity as allocation/export.

    ``raw_fields_out``: optional sidecar capture.  When LDLQ applies to this
    format under ``context``, the mapping receives the pre-gate raw encode —
    the EXACT no-LDLQ render (same env, codebook, scale sweep/coding) that the
    gated reassignment starts from — under keys ``{"ldlq_applied": True,
    "fields", "grid", "mode", "k"}``.  When LDLQ does not apply the mapping is
    left untouched, so callers can key sidecar emission on its population.
    The captured fields are the same (never-mutated) dict the gate consumed;
    no extra encode or reconstruction happens here.
    """
    info = _cb_info(spec.name)
    if info is None:
        raise ValueError(f"{spec.name!r} is not a CB format")
    grid, mode, k = info
    from .nvfp4_cb_formats import SCALE_CODING_V1, nvfp4_cb_fields

    source = (
        codebook_source_for_cell(str(qname), spec.name, context)
        if qname
        else codebook_source_for_format(spec.name, context)
    )
    if source == "learned" and qname:
        from .cb_learned_bundle import refuse_routed_moe_learned

        refuse_routed_moe_learned(str(qname), weight=weight)
    if source == "learned" and codebook is None:
        if not qname:
            raise ValueError(
                f"{spec.name}: learned-codebook render requires its exact "
                "unit qname and materialized bundle cell; refusing to render "
                "the lattice and stamp it as learned"
            )
        if not context.codebook_bundle_path:
            raise ValueError(
                f"{qname} ({spec.name}): learned-codebook render has no "
                "value-bearing CB_CODEBOOK_BUNDLE"
            )
        from .cb_learned_bundle import load_bundle_cached

        codebook = load_bundle_cached(context.codebook_bundle_path).codebook_for(
            str(qname),
            spec.name,
            weight=weight,
            col_weights=col_weights,
        )
    coding = context.scale_coding if grid == "fp4" else SCALE_CODING_V1
    fields = nvfp4_cb_fields(
        weight,
        k,
        grid=grid,
        mode=mode,
        col_weights=col_weights,
        codebook=codebook,
        scale_sweep=scale_sweep_for_format(spec.name, context),
        scale_coding=coding,
        encode_tier=context.encode_tier,
        warm_scale_state=warm_scale_state,
    )
    if _ldlq_for_format(spec.name, context):
        if col_weights is None:
            raise ValueError(f"{spec.name}: LDLQ requires activation-weighted col_weights")
        if activation_rows is None:
            if ldlq_missing_activation_ok:
                # Explicit call-site opt-in for cells that PROVABLY have no
                # calibration activations (declared never-routed experts).
                # The holdout gate fail-closes such cells to the raw render
                # at export (raw_uncertifiable_too_few_rows), so the raw
                # fields ARE the shipping render — returning them here keeps
                # cost/export identity without weakening the guard for every
                # other caller (a broken activation loader must still raise,
                # never silently produce an all-raw table stamped as LDLQ).
                if raw_fields_out is not None:
                    raw_fields_out["ldlq_applied"] = True
                    raw_fields_out["fields"] = fields
                    raw_fields_out["grid"] = grid
                    raw_fields_out["mode"] = mode
                    raw_fields_out["k"] = k
                return fields
            raise ValueError(f"{spec.name}: LDLQ requires calibration activation rows")
        from .nvfp4_cb_formats import ldlq_reassign_cb_fields_gated

        # Post-allocation LDLQ refinement is byte-neutral but must be
        # do-no-harm on the declared gate metric — since 2026-08-08 that is
        # holdout_activation_output_mse: LDLQ is certified on rows its Hessian
        # never saw, because the previous in-sample scoring could not fail and
        # its error was measured ANTI-correlated with the true benefit.
        # The wrapper is also the canonical-route refusal boundary.  In
        # disabled gate mode it returns unconditional LDLQ on successful
        # experts while preserving typed Hessian failures as raw.
        info = _cb_info(spec.name)
        assert info is not None
        _grid, _mode, _k = info
        if raw_fields_out is not None:
            # The pre-gate fields ARE the identical-env no-LDLQ render; the
            # gate never mutates them (it builds new dicts), so keeping the
            # reference costs no extra encode and no extra residency.
            raw_fields_out["ldlq_applied"] = True
            raw_fields_out["fields"] = fields
            raw_fields_out["grid"] = grid
            raw_fields_out["mode"] = mode
            raw_fields_out["k"] = _k
        fields, _gate_info = ldlq_reassign_cb_fields_gated(
            weight,
            fields,
            col_weights,
            activation_rows,
            grid=grid,
            mode=mode,
            k=_k,
        )
    return fields


def cb_quantize_dequantize_for_context(
    spec: fr.FormatSpec,
    weight,
    *,
    context: CBSerializationContext,
    qname: str | None = None,
    col_weights=None,
    codebook=None,
    activation_rows=None,
    ldlq_missing_activation_ok: bool = False,
):
    """Render a CB weight under the exact artifact serialization context."""
    info = _cb_info(spec.name)
    if info is None:
        raise ValueError(f"{spec.name!r} is not a CB format")
    grid, mode, k = info
    from .nvfp4_cb_formats import nvfp4_cb_reconstruct

    fields = cb_fields_for_context(
        spec,
        weight,
        context=context,
        qname=qname,
        col_weights=col_weights,
        codebook=codebook,
        activation_rows=activation_rows,
        ldlq_missing_activation_ok=ldlq_missing_activation_ok,
    )
    return nvfp4_cb_reconstruct(
        fields,
        k,
        grid=grid,
        mode=mode,
    ).to(weight.dtype)


def cb_tensor_serialization_stamp(
    format_name: str,
    shape: tuple[int, ...] | Sequence[int],
    *,
    qname: str,
    context: CBSerializationContext,
) -> str:
    """Canonical per-tensor identity persisted beside an assignment."""
    return str(cb_tensor_payload_breakdown(
        format_name,
        shape,
        qname=qname,
        context=context,
    )["identity_key"])


def cb_assignment_serialization_stamps(
    assignment: Mapping[str, str],
    shapes: Mapping[str, tuple[int, ...] | Sequence[int]],
    *,
    context: CBSerializationContext,
) -> dict[str, str]:
    """Return exact per-CB-tensor identities for a concrete assignment."""
    return {
        str(qname): cb_tensor_serialization_stamp(
            format_name,
            shapes[qname],
            qname=str(qname),
            context=context,
        )
        for qname, format_name in assignment.items()
        if is_cb_format(format_name)
    }


def validate_cb_assignment_serialization_stamps(
    assignment: Mapping[str, str],
    shapes: Mapping[str, tuple[int, ...] | Sequence[int]],
    *,
    context: CBSerializationContext,
    stamps: Mapping[str, object] | None,
    where: str,
) -> dict[str, str]:
    """Require exact per-layer identities before consuming CB byte totals."""
    expected = cb_assignment_serialization_stamps(
        assignment, shapes, context=context
    )
    if not expected:
        return {}
    if not isinstance(stamps, Mapping):
        raise ValueError(
            f"{where}: CB assignment is missing per-layer serialization "
            "stamps"
        )
    observed = {
        str(name): str(value)
        for name, value in stamps.items()
        if str(name) in expected
    }
    missing = sorted(set(expected) - set(observed))
    # This mapping is dedicated to CB tensor identities. Any key outside the
    # exact CB assignment is stale metadata (including a tensor since changed
    # to a non-CB format), not an innocuous annotation to preserve.
    extra = sorted(str(name) for name in stamps if str(name) not in expected)
    mismatched = sorted(
        name for name in expected
        if name in observed and observed[name] != expected[name]
    )
    if missing or extra or mismatched:
        raise ValueError(
            f"{where}: CB per-layer serialization identity mismatch: "
            f"missing={missing[:8]}, extra={extra[:8]}, "
            f"mismatched={mismatched[:8]}"
        )
    return expected


def cb_serialization_metadata_from_assignment_payload(
    payload: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, dict[str, str]]:
    """Extract global/per-tensor CB stamps from layer or Pareto JSON."""
    raw_assignment = payload.get("assignment")
    if not isinstance(raw_assignment, Mapping):
        raw_assignment = payload
    meta = payload.get("__prismaquant__")
    context_stamp = (
        meta.get("cb_serialized_payload")
        if isinstance(meta, Mapping)
        else None
    )
    if context_stamp is None:
        context_stamp = payload.get("cb_serialized_payload")
    identities: dict[str, str] = {}
    meta_identities = (
        meta.get(CB_ASSIGNMENT_IDENTITIES_FIELD)
        if isinstance(meta, Mapping)
        else None
    )
    if isinstance(meta_identities, Mapping):
        identities.update({
            str(name): str(value)
            for name, value in meta_identities.items()
        })
    top_level = payload.get(CB_ASSIGNMENT_IDENTITIES_FIELD)
    if isinstance(top_level, Mapping):
        identities.update({str(name): str(value)
                           for name, value in top_level.items()})
    for name, entry in raw_assignment.items():
        if isinstance(entry, Mapping) and entry.get(CB_TENSOR_IDENTITY_FIELD):
            identities[str(name)] = str(entry[CB_TENSOR_IDENTITY_FIELD])
    return (
        context_stamp if isinstance(context_stamp, Mapping) else None,
        identities,
    )


def assignment_serialization_sha256(
    assignment: Mapping[str, str],
) -> str:
    """Canonical SHA-256 binding a byte budget to one exact assignment."""
    normalized = {
        str(name): fr.canonical_format_name(str(fmt).strip().upper())
        for name, fmt in assignment.items()
    }
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def whole_artifact_budget_stamp(
    *,
    budget_bytes: int,
    selection_tensor_payload_bytes: int,
    selection_non_tensor_reserve_bytes: int,
    selection_assignment: Mapping[str, str],
    excluded_source_prefixes: Iterable[str] = (),
) -> dict:
    """Persist the conservative selection contract consumed by exporters.

    ``excluded_source_prefixes`` records the source namespaces this price was
    computed WITHOUT. It travels because the price and the artifact are two
    halves of one statement: the allocator can only spend the excluded bytes
    on the body if the exporter actually omits them, and nothing else in the
    artifact records that they were meant to be absent. Omitted when empty, so
    a run that excludes nothing writes a byte-identical stamp.
    """
    values = {
        "budget_bytes": budget_bytes,
        "selection_tensor_payload_bytes": selection_tensor_payload_bytes,
        "selection_non_tensor_reserve_bytes": selection_non_tensor_reserve_bytes,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    upper_bound = (
        selection_tensor_payload_bytes + selection_non_tensor_reserve_bytes
    )
    if upper_bound > budget_bytes:
        raise ValueError(
            "selection whole-artifact upper bound exceeds its hard budget: "
            f"{upper_bound}B > {budget_bytes}B"
        )
    excluded = tuple(
        dict.fromkeys(
            str(prefix).strip()
            for prefix in (excluded_source_prefixes or ())
            if str(prefix).strip()
        )
    )
    return {
        "schema": WHOLE_ARTIFACT_BUDGET_SCHEMA,
        "scope": "all_regular_files_recursive",
        "budget_bytes": budget_bytes,
        **({"excluded_source_prefixes": list(excluded)} if excluded else {}),
        "selection_tensor_payload_bytes": selection_tensor_payload_bytes,
        "selection_non_tensor_reserve_bytes": selection_non_tensor_reserve_bytes,
        "selection_whole_artifact_upper_bound_bytes": upper_bound,
        "selection_assignment_sha256": assignment_serialization_sha256(
            selection_assignment
        ),
        "selection_contract": (
            "tensor_payload_plus_operator_supplied_non_tensor_reserve"
        ),
        "final_contract": "stat_all_regular_files_recursive_fail_closed",
    }


def whole_artifact_budget_from_assignment_payload(
    payload: Mapping[str, object],
    *,
    where: str,
    assignment: Mapping[str, str] | None = None,
) -> Mapping[str, object] | None:
    """Read and validate an optional hard export-directory budget stamp."""
    meta = payload.get("__prismaquant__")
    raw = (
        meta.get(WHOLE_ARTIFACT_BUDGET_FIELD)
        if isinstance(meta, Mapping)
        else None
    )
    if raw is None:
        raw = payload.get(WHOLE_ARTIFACT_BUDGET_FIELD)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: whole-artifact budget stamp is not an object")
    if raw.get("schema") != WHOLE_ARTIFACT_BUDGET_SCHEMA:
        raise ValueError(
            f"{where}: unsupported whole-artifact budget schema "
            f"{raw.get('schema')!r}"
        )
    required = (
        "budget_bytes",
        "selection_tensor_payload_bytes",
        "selection_non_tensor_reserve_bytes",
        "selection_whole_artifact_upper_bound_bytes",
    )
    parsed: dict[str, int] = {}
    for name in required:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{where}: whole-artifact budget field {name!r} must be a "
                "nonnegative integer"
            )
        parsed[name] = value
    expected_upper = (
        parsed["selection_tensor_payload_bytes"]
        + parsed["selection_non_tensor_reserve_bytes"]
    )
    if parsed["selection_whole_artifact_upper_bound_bytes"] != expected_upper:
        raise ValueError(
            f"{where}: whole-artifact upper bound does not reconcile: "
            f"stamp={parsed['selection_whole_artifact_upper_bound_bytes']}B, "
            f"payload+reserve={expected_upper}B"
        )
    if expected_upper > parsed["budget_bytes"]:
        raise ValueError(
            f"{where}: selected whole-artifact upper bound {expected_upper}B "
            f"exceeds budget {parsed['budget_bytes']}B"
        )
    assignment_digest = raw.get("selection_assignment_sha256")
    if not isinstance(assignment_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", assignment_digest
    ):
        raise ValueError(
            f"{where}: whole-artifact budget stamp has no valid exact "
            "selection_assignment_sha256"
        )
    if assignment is not None:
        actual_digest = assignment_serialization_sha256(assignment)
        if actual_digest != assignment_digest:
            raise ValueError(
                f"{where}: whole-artifact budget was priced for assignment "
                f"{assignment_digest}, but the assignment being consumed "
                f"hashes to {actual_digest}"
            )
    raw_excluded = raw.get("excluded_source_prefixes")
    if raw_excluded is not None:
        if not isinstance(raw_excluded, (list, tuple)) or not all(
            isinstance(p, str) and p.strip() for p in raw_excluded
        ):
            raise ValueError(
                f"{where}: whole-artifact budget field "
                "'excluded_source_prefixes' must be a list of non-empty "
                "strings"
            )
    return dict(raw)


def budget_stamp_excluded_prefixes(
    stamp: Mapping[str, object] | None,
) -> tuple[str, ...]:
    """The source namespaces a budget stamp was priced WITHOUT.

    Absent means "none excluded", which is both the pre-existing behaviour and
    the correct reading of every stamp written before the field existed: those
    prices charged the whole checkpoint.
    """

    if not stamp:
        return ()
    return tuple(
        dict.fromkeys(
            str(p).strip()
            for p in (stamp.get("excluded_source_prefixes") or ())
            if str(p).strip()
        )
    )


def assert_exclusions_match_budget_stamp(
    stamp: Mapping[str, object] | None,
    excluded_namespaces: Iterable[str],
    *,
    where: str,
) -> None:
    """Refuse an artifact whose contents contradict the price that justified it.

    A namespace exclusion is one statement made in two places: the allocator
    declines to CHARGE for the namespace, which hands those bytes to the body,
    and the exporter declines to WRITE it. Make only the first and the artifact
    overshoots its budget by the excluded mass -- caught, but only by the final
    recursive stat, hours later. Make only the second and the artifact comes in
    UNDER budget by that mass, having bought less quality than it paid for, and
    nothing catches it at all: the selection reconciles, the export passes, the
    shipcard agrees. That asymmetry is why this is an equality check and not a
    bound.

    No stamp means no claim: namespace exclusion is a legitimate operation on
    its own, and an unbudgeted export has nothing to contradict. The check
    binds only once a price has been asserted.
    """

    if not stamp:
        return
    priced = set(budget_stamp_excluded_prefixes(stamp))
    written = {
        str(p).strip() for p in (excluded_namespaces or ()) if str(p).strip()
    }
    if priced == written:
        return
    raise ValueError(
        f"{where}: namespace exclusions disagree with the budget stamp that "
        f"priced this assignment. The price was computed WITHOUT "
        f"{sorted(priced) or '[]'}; this export omits {sorted(written) or '[]'}. "
        f"Priced-but-written ({sorted(priced - written) or '[]'}) overshoots "
        f"the budget by those bytes; written-but-priced "
        f"({sorted(written - priced) or '[]'}) silently ships under budget, "
        f"having bought less quality than the budget paid for. Re-run the "
        f"allocation and the export with the same exclusion set."
    )


def recursive_regular_file_bytes(path: str | Path) -> int:
    """Measure a completed artifact using the budget stamp's final scope."""
    root = Path(path)
    if root.is_file():
        return int(root.stat().st_size)
    if not root.is_dir():
        raise FileNotFoundError(f"export artifact does not exist: {root}")
    return sum(
        int(item.stat().st_size)
        for item in root.rglob("*")
        if item.is_file()
    )


def enforce_whole_artifact_budget(
    artifact_path: str | Path,
    assignment_payload: Mapping[str, object],
    *,
    where: str,
    assignment: Mapping[str, str] | None = None,
) -> dict | None:
    """Hard-fail a completed file/directory against its persisted budget."""
    stamp = whole_artifact_budget_from_assignment_payload(
        assignment_payload,
        where=where,
        assignment=assignment,
    )
    if stamp is None:
        return None
    actual = recursive_regular_file_bytes(artifact_path)
    budget = int(stamp["budget_bytes"])
    attestation = {
        "scope": "all_regular_files_recursive",
        "artifact_path": str(artifact_path),
        "actual_bytes": actual,
        "budget_bytes": budget,
        "headroom_bytes": budget - actual,
        "within_budget": actual <= budget,
    }
    if actual > budget:
        raise RuntimeError(
            f"{where}: exact completed artifact size is {actual}B, exceeding "
            f"the hard whole-artifact budget of {budget}B by {actual - budget}B"
        )
    return attestation


def codebook_subtable_shapes(format_name: str) -> tuple[tuple[int, int], ...]:
    """Exact FP16 subtable shapes emitted for one CB format."""
    info = _cb_info(format_name)
    if info is None:
        raise ValueError(f"{format_name!r} is not a CB format")
    grid, mode, k = info
    n_sub = family_for(grid, mode).n_sub
    return _layout_codebook_subtable_shapes(k, mode, n_sub)


def codebook_sidecar_payload_bytes(format_name: str) -> int:
    """FP16 tensor payload bytes for one codebook table set."""
    return sum(rows * cols * _FP16_BYTES
               for rows, cols in codebook_subtable_shapes(format_name))


def _default_logical_ref(qname: str, source: str) -> str:
    return "lattice" if source == "lattice" else str(qname).rsplit(".", 1)[-1]


def _physical_codebook_refs(
    qname: str,
    format_name: str,
    context: CBSerializationContext,
) -> tuple[str, ...]:
    canonical = str(format_name).strip().upper()
    expected_count = len(codebook_subtable_shapes(canonical))
    supplied = None
    if context.codebook_refs_by_qname_format is not None:
        by_format = context.codebook_refs_by_qname_format.get(qname)
        if by_format is not None:
            supplied = by_format.get(canonical)
    if context.codebook_refs is not None:
        legacy_supplied = context.codebook_refs.get(qname)
        if supplied is not None and legacy_supplied is not None:
            raise ValueError(
                f"{qname}: {canonical} has both per-format and legacy "
                "codebook refs; refusing an ambiguous physical identity"
            )
        if supplied is None:
            supplied = legacy_supplied
    if supplied is not None:
        refs = (supplied,) if isinstance(supplied, str) else tuple(
            str(item) for item in supplied
        )
        if len(refs) != expected_count:
            raise ValueError(
                f"{qname}: {format_name} needs {expected_count} codebook "
                f"subtable ref(s), got {len(refs)}"
            )
        if len(set(refs)) != len(refs):
            raise ValueError(
                f"{qname}: {format_name} repeats a physical codebook ref "
                f"within one table set: {list(refs)}"
            )
        return tuple(str(item) for item in refs)

    source = codebook_source_for_format(canonical, context)
    logical = _default_logical_ref(qname, source)
    base = f"cb_codebook.{logical}.{canonical}"
    if expected_count == 1:
        return (base,)
    return tuple(f"{base}.sub{index}" for index in range(expected_count))


def _sidecar_identity(
    qname: str,
    format_name: str,
    context: CBSerializationContext,
    *,
    require_materialized: bool = True,
) -> dict:
    canonical = str(format_name).strip().upper()
    refs = _physical_codebook_refs(qname, canonical, context)
    shapes = codebook_subtable_shapes(canonical)
    source = codebook_source_for_format(canonical, context)
    content_sha256 = None
    deferred = False
    if source == "learned":
        digests = context.codebook_content_digests or {}
        missing = [ref for ref in refs if ref not in digests]
        if missing and require_materialized:
            raise ValueError(
                f"{qname}: learned {canonical} sidecar identity is missing "
                f"materialized SHA-256 digest(s) for {missing}; logical refs "
                "alone cannot prove render/export byte identity"
            )
        if missing:
            # Sizing-only caller (see require_materialized_codebook_identity
            # on cb_tensor_payload_breakdown).  Byte counts below are exact
            # regardless; only the identity is unproven, and it is marked as
            # such so it can never masquerade as a materialized one.
            deferred = True
        else:
            content_sha256 = [digests[ref] for ref in refs]
    else:
        content_sha256 = list(lattice_codebook_content_sha256(canonical))
        supplied_digests = context.codebook_content_digests or {}
        mismatched = [
            ref
            for ref, digest in zip(refs, content_sha256, strict=True)
            if ref in supplied_digests and supplied_digests[ref] != digest
        ]
        if mismatched:
            raise ValueError(
                f"{qname}: lattice {canonical} materialized codebook bytes "
                f"do not match the canonical lattice identity for "
                f"{mismatched}"
            )
    return {
        "format": canonical,
        "codebook_source": source,
        "codebook_ref": list(refs),
        "dtype": "float16",
        "subtable_shapes": [list(shape) for shape in shapes],
        "payload_bytes": codebook_sidecar_payload_bytes(canonical),
        "content_sha256": content_sha256,
        **({"materialized_identity": False} if deferred else {}),
    }


def _has_explicit_codebook_refs(
    qname: str,
    format_name: str,
    context: CBSerializationContext,
) -> bool:
    """Return whether ``qname`` explicitly supplies refs for this format."""
    canonical = str(format_name).strip().upper()
    by_format = context.codebook_refs_by_qname_format
    format_supplied = (
        by_format is not None
        and qname in by_format
        and canonical in by_format[qname]
    )
    legacy_supplied = (
        context.codebook_refs is not None
        and qname in context.codebook_refs
    )
    if format_supplied and legacy_supplied:
        raise ValueError(
            f"{qname}: {canonical} has both per-format and legacy "
            "codebook refs; refusing an ambiguous physical identity"
        )
    return bool(format_supplied or legacy_supplied)


def _tensor_sidecar_identities(
    qname: str,
    format_name: str,
    context: CBSerializationContext,
    *,
    require_materialized: bool = True,
) -> tuple[dict, ...]:
    """Resolve the physical codebook sets consumed by one packed tensor.

    Routed MoE ``gate_up_proj`` remains one packed weight tensor, but the
    per-role Gridbook ABI consumes a gate LUT for its first output-row half
    and an up LUT for its second half.  The producer represents those LUTs as
    ordinary logical ``gate_proj``/``up_proj`` cells.  Only that explicit,
    complete declaration changes accounting; legacy fused refs, lattice
    contexts, and dense ``gate_up_proj`` tensors retain their one-sidecar
    identity exactly.
    """
    canonical = str(format_name).strip().upper()
    routed_marker = ".experts.gate_up_proj"
    physical_parent = str(qname)
    discriminator = None
    if "." in physical_parent:
        candidate, discriminator = physical_parent.rsplit(".", 1)
        if discriminator.startswith("format_group_"):
            physical_parent = candidate
        else:
            discriminator = None
    if (
        codebook_source_for_format(canonical, context) == "learned"
        and discriminator is not None
        and physical_parent.endswith(".experts.down_proj")
    ):
        return (
            _sidecar_identity(
                bundle_role_qname(str(qname), "down_proj"),
                canonical,
                context,
                require_materialized=require_materialized,
            ),
        )
    if (
        codebook_source_for_format(canonical, context) != "learned"
        or not physical_parent.endswith(routed_marker)
    ):
        return (
            _sidecar_identity(
                qname, canonical, context,
                require_materialized=require_materialized,
            ),
        )

    gate_qname = bundle_role_qname(str(qname), "gate_proj")
    up_qname = bundle_role_qname(str(qname), "up_proj")
    physical_supplied = _has_explicit_codebook_refs(
        str(qname), canonical, context
    )
    gate_supplied = _has_explicit_codebook_refs(
        gate_qname, canonical, context
    )
    up_supplied = _has_explicit_codebook_refs(
        up_qname, canonical, context
    )

    if not gate_supplied and not up_supplied:
        return (
            _sidecar_identity(
                qname, canonical, context,
                require_materialized=require_materialized,
            ),
        )
    if physical_supplied:
        raise ValueError(
            f"{qname}: learned routed-MoE {canonical} supplies both the "
            "physical fused codebook refs and logical gate/up refs; use "
            "exactly one ABI"
        )
    if gate_supplied != up_supplied:
        missing = up_qname if gate_supplied else gate_qname
        raise ValueError(
            f"{qname}: learned routed-MoE {canonical} per-role sidecars are "
            f"incomplete; missing explicit refs for {missing!r}"
        )
    return (
        _sidecar_identity(
            gate_qname, canonical, context,
            require_materialized=require_materialized,
        ),
        _sidecar_identity(
            up_qname, canonical, context,
            require_materialized=require_materialized,
        ),
    )


def _identity_key(identity: Mapping) -> str:
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def cb_tensor_payload_breakdown(
    format_name: str,
    shape: tuple[int, ...] | Sequence[int],
    *,
    qname: str,
    context: CBSerializationContext,
    require_materialized_codebook_identity: bool = True,
) -> dict:
    """Versioned byte breakdown for one serialized CB Linear.

    ``tensor_payload_bytes`` excludes the shared codebook sidecar; the returned
    ``sidecar_identity`` is what assignment-level accounting deduplicates.

    ``require_materialized_codebook_identity=False`` answers the *rate*
    question alone: every byte count here is derived from the format and the
    shape (``codebook_sidecar_payload_bytes`` is a function of the rung), so a
    learned cell whose book has not been banked can still be priced.  It is
    for legality probes that must evaluate formats the unit will never be
    allowed to use -- a routed expert against a K48 rung, say, whose book was
    deliberately never learned because the source-payload ceiling already
    excludes it.  Making that verdict depend on which books happen to exist
    would make legality a function of build order.  Any identity produced this
    way carries ``materialized_identity: False`` and a null ``content_sha256``
    so it cannot be mistaken for a proof of render/export byte identity;
    callers that produce bytes must use the default.
    """
    if context is None:
        raise ValueError(
            "exact CB byte pricing requires CBSerializationContext; refusing "
            "to assume legacy-v1 bytes"
        )
    canonical = str(format_name).strip().upper()
    info = _cb_info(canonical)
    if info is None:
        raise ValueError(f"{format_name!r} is not a CB format")
    dims = tuple(int(dim) for dim in shape)
    if len(dims) < 2 or any(dim <= 0 for dim in dims):
        raise ValueError(
            f"{qname}: exact CB bytes need a positive rank>=2 Linear shape, "
            f"got {dims}"
        )
    in_features = dims[-1]
    if in_features % _SUPERBLOCK:
        raise ValueError(
            f"{qname}: CB in_features={in_features} is not divisible by "
            f"{_SUPERBLOCK}"
        )
    output_rows = int(math.prod(dims[:-1]))
    n_superblocks = in_features // _SUPERBLOCK
    grid, mode, k = info
    packed_type_size = cb_type_size(k, grid, context.scale_coding)
    index_bytes = output_rows * n_superblocks * (INDEX_BYTES_PER_K * k)
    fp4_scale_bytes = 0
    if grid == "fp4":
        scale_bytes_per_superblock = packed_type_size - (INDEX_BYTES_PER_K * k)
        fp4_scale_bytes = output_rows * n_superblocks * scale_bytes_per_superblock
    fp8_row_scale_bytes = _FP32_BYTES * output_rows if grid == "fp8" else 0
    input_global_scale_bytes = (
        _FP32_BYTES
        if grid == "fp4" and context.activation_execution is not None
        else 0
    )
    packed_weight_bytes = index_bytes + fp4_scale_bytes
    tensor_payload_bytes = (
        packed_weight_bytes
        + fp8_row_scale_bytes
        + input_global_scale_bytes
    )
    sidecars = _tensor_sidecar_identities(
        qname,
        canonical,
        context,
        require_materialized=require_materialized_codebook_identity,
    )
    if len(sidecars) == 1:
        sidecar = sidecars[0]
    else:
        sidecar = {
            "kind": "routed_moe_per_role_codebooks",
            "roles": [
                {"projection": projection, "sidecar": identity}
                for projection, identity in zip(
                    ("gate_proj", "up_proj"), sidecars, strict=True
                )
            ],
            "payload_bytes": sum(
                int(identity["payload_bytes"]) for identity in sidecars
            ),
        }
    payload_schema = _serialized_payload_schema(context)
    # Per-family LDLQ: the tensor's ldlq stamps the ACTUAL result, not the
    # global context's ANY. For scope nvfp4, NVFP4_CB tensors are ldlq:true
    # and FP8_CB tensors are ldlq:false.
    tensor_ldlq = _ldlq_for_format(canonical, context)
    identity = {
        "schema": payload_schema,
        "format": canonical,
        "grid": grid,
        "mode": mode,
        "k": k,
        "artifact_scale_coding": context.scale_coding,
        "layout_version": context.layout_version,
        "scale_sweep": scale_sweep_for_format(canonical, context),
        "ldlq": tensor_ldlq,
        "ldlq_scope": getattr(context, "ldlq_scope", "all" if context.ldlq else "none"),
        **({
            "minchain": True,
            "minchain_version": context.minchain_version,
        } if context.minchain else {}),
        "encode_tier": context.encode_tier,
        "renderer_abi": context.renderer_abi,
        "activation_contract": context.activation_contract,
        "activation_execution": context.activation_execution,
        "tensor_scale_coding": context.scale_coding if grid == "fp4" else "none",
        "type_size": packed_type_size,
        "shape": list(dims),
        "params": int(math.prod(dims)),
        "output_rows": output_rows,
        "in_features": in_features,
        "superblocks_per_row": n_superblocks,
        "index_bytes": int(index_bytes),
        "fp4_scale_bytes": int(fp4_scale_bytes),
        "fp8_row_scale_bytes": int(fp8_row_scale_bytes),
        "input_global_scale_bytes": int(input_global_scale_bytes),
        "global_scale_bytes": 0,
        "packed_weight_bytes": int(packed_weight_bytes),
        "tensor_payload_bytes": int(tensor_payload_bytes),
        "sidecar": sidecar,
    }
    result = {
        "schema": payload_schema,
        "identity": identity,
        "identity_key": _identity_key(identity),
        "qname": str(qname),
        "format": canonical,
        "shape": list(dims),
        "params": int(math.prod(dims)),
        "output_rows": output_rows,
        "superblocks_per_row": n_superblocks,
        "index_bytes": int(index_bytes),
        "fp4_scale_bytes": int(fp4_scale_bytes),
        "fp8_row_scale_bytes": int(fp8_row_scale_bytes),
        "input_global_scale_bytes": int(input_global_scale_bytes),
        "global_scale_bytes": 0,
        "packed_weight_bytes": int(packed_weight_bytes),
        "tensor_payload_bytes": int(tensor_payload_bytes),
        "sidecar_identity": sidecar,
        "sidecar_identity_key": _identity_key(sidecar),
        "sidecar_payload_bytes": int(sidecar["payload_bytes"]),
    }
    if len(sidecars) > 1:
        result["sidecar_identities"] = list(sidecars)
        result["sidecar_identity_keys"] = [
            _identity_key(identity) for identity in sidecars
        ]
    return result


def cb_breakdown_identity_is_materialized(breakdown: Mapping) -> bool:
    """False when a breakdown's codebook identity was left unproven.

    Only ``require_materialized_codebook_identity=False`` can produce that,
    and only for a learned cell whose book is not banked; a sizing-mode call
    on a banked cell is byte-identical to a strict one, so this is also how a
    caller cheaply asks "did the deferral actually apply here?".
    """
    identities = breakdown.get("sidecar_identities")
    if not identities:
        identities = [breakdown.get("sidecar_identity") or {}]
    return all(
        bool(dict(identity).get("materialized_identity", True))
        for identity in identities
    )


def cb_assignment_payload_breakdown(
    assignment: Mapping[str, str],
    shapes: Mapping[str, tuple[int, ...] | Sequence[int]],
    *,
    context: CBSerializationContext,
) -> dict:
    """Exact CB payload bytes for an assignment, deduplicating sidecars."""
    if context is None:
        raise ValueError(
            "exact CB assignment pricing requires CBSerializationContext; "
            "refusing to assume legacy-v1 bytes"
        )
    per_tensor: dict[str, dict] = {}
    sidecars: dict[str, dict] = {}
    sidecar_key_by_ref: dict[str, str] = {}
    totals = {
        "index_bytes": 0,
        "fp4_scale_bytes": 0,
        "fp8_row_scale_bytes": 0,
        "input_global_scale_bytes": 0,
        "global_scale_bytes": 0,
        "tensor_payload_bytes": 0,
    }
    for qname, format_name in assignment.items():
        if not is_cb_format(format_name):
            continue
        if qname not in shapes:
            raise KeyError(f"CB byte accounting has no shape for {qname!r}")
        item = cb_tensor_payload_breakdown(
            format_name, shapes[qname], qname=qname, context=context
        )
        per_tensor[qname] = item
        for key in totals:
            totals[key] += int(item[key])
        tensor_sidecars = item.get(
            "sidecar_identities", (item["sidecar_identity"],)
        )
        for sidecar_identity in tensor_sidecars:
            sidecar_key = _identity_key(sidecar_identity)
            refs = tuple(
                str(ref) for ref in sidecar_identity["codebook_ref"]
            )
            overlapping_keys = {
                sidecar_key_by_ref[ref]
                for ref in refs
                if ref in sidecar_key_by_ref
            }
            if overlapping_keys and overlapping_keys != {sidecar_key}:
                raise ValueError(
                    f"{qname}: physical CB codebook refs are partially shared "
                    "or reused with conflicting shape/content identity. "
                    "Explicit ref sets must be disjoint or completely "
                    f"identical; refs={list(refs)}, prior_identity_keys="
                    f"{sorted(overlapping_keys)}"
                )
            previous = sidecars.get(sidecar_key)
            if previous is None:
                if overlapping_keys:
                    raise ValueError(
                        f"{qname}: physical CB codebook refs overlap a prior "
                        "sidecar without an identical complete identity"
                    )
                sidecars[sidecar_key] = sidecar_identity
                sidecar_key_by_ref.update({ref: sidecar_key for ref in refs})
            elif previous != sidecar_identity:
                raise ValueError(
                    f"conflicting CB sidecar identity for {qname}: "
                    f"{previous} vs {sidecar_identity}"
                )
            elif any(
                sidecar_key_by_ref.get(ref) != sidecar_key for ref in refs
            ):
                raise ValueError(
                    f"{qname}: identical CB sidecar identity did not resolve "
                    "to the same complete physical ref set"
                )
    sidecar_bytes = sum(int(item["payload_bytes"]) for item in sidecars.values())
    total_bytes = totals["tensor_payload_bytes"] + sidecar_bytes
    serialized_formats = sorted({
        str(item["format"]) for item in per_tensor.values()
    })
    serialized_source_map = (
        dict(context.codebook_source_by_format)
        if context.codebook_source_by_format is not None
        else (
            {
                name: codebook_source_for_format(name, context)
                for name in serialized_formats
            }
            if serialized_formats
            and effective_codebook_source_scope(context) != "none"
            else None
        )
    )
    serialized_scalar_source = (
        "learned"
        if serialized_source_map is not None
        and "learned" in serialized_source_map.values()
        else (
            "lattice"
            if serialized_source_map is not None
            else context.codebook_source
        )
    )
    return {
        "schema": _serialized_payload_schema(context),
        "context": {
            "scale_coding": context.scale_coding,
            "layout_version": context.layout_version,
            "codebook_source": serialized_scalar_source,
            **({
                "codebook_source_scope": effective_codebook_source_scope(context),
            } if (
                effective_codebook_source_scope(context) == "fp8"
                or (
                    serialized_source_map is not None
                    and effective_codebook_source_scope(context) != "none"
                )
            ) else {}),
            **({
                "codebook_source_by_format": serialized_source_map,
            } if serialized_source_map is not None else {}),
            "scale_sweep": context.scale_sweep,
            **({
                "scale_sweep_scope": effective_scale_sweep_scope(context),
            } if effective_scale_sweep_scope(context) in {"nvfp4", "fp8"} else {}),
            "ldlq": context.ldlq,
            **({
                "minchain": True,
                "minchain_version": context.minchain_version,
            } if context.minchain else {}),
            "encode_tier": context.encode_tier,
            "renderer_abi": context.renderer_abi,
            **({
                "activation_contract": context.activation_contract,
                "activation_execution": context.activation_execution,
            } if context.activation_contract is not None else {}),
        },
        **{key: int(value) for key, value in totals.items()},
        "codebook_sidecar_bytes": int(sidecar_bytes),
        "total_bytes": int(total_bytes),
        "per_tensor": per_tensor,
        "sidecars": [sidecars[key] for key in sorted(sidecars)],
    }


def cb_payload_summary(breakdown: Mapping[str, object]) -> dict:
    """Compact persisted form of an assignment payload breakdown.

    Export provenance needs the version/layout and shared-sidecar identities,
    but duplicating every per-tensor record in ``quant_config.json`` would be
    needlessly large.  Keep the independently checkable totals and physical
    sidecar identities; config groups already map tensors to those refs.
    """
    byte_keys = (
        "index_bytes",
        "fp4_scale_bytes",
        "fp8_row_scale_bytes",
        "input_global_scale_bytes",
        "global_scale_bytes",
        "tensor_payload_bytes",
        "codebook_sidecar_bytes",
        "total_bytes",
    )
    per_tensor = breakdown.get("per_tensor", {})
    if not isinstance(per_tensor, Mapping):
        raise TypeError("CB payload breakdown has invalid per_tensor data")
    sidecars = breakdown.get("sidecars", [])
    if not isinstance(sidecars, list):
        raise TypeError("CB payload breakdown has invalid sidecars data")
    return {
        "schema": breakdown.get("schema", CB_SERIALIZED_PAYLOAD_SCHEMA),
        "context": breakdown.get("context"),
        **{key: int(breakdown.get(key, 0)) for key in byte_keys},
        "n_tensors": len(per_tensor),
        "sidecars": sidecars,
    }


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(raw: str, *, where: str):
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise AssertionError(f"{where}: invalid strict JSON: {exc}") from exc


def _safetensors_data_spans(path: Path) -> dict[str, int]:
    """Read exact tensor data-span bytes from a safetensors container.

    The serialized-payload API deliberately prices tensor data spans.  A real
    file is larger by its eight-byte prefix plus JSON header, so exporters use
    this parser for the separate final artifact inventory instead of treating
    payload bytes as filesystem bytes.
    """
    size = path.stat().st_size
    if size < 8:
        raise AssertionError(f"{path}: truncated safetensors prefix ({size}B)")
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        (header_length,) = struct.unpack("<Q", raw_length)
        if header_length > size - 8:
            raise AssertionError(
                f"{path}: safetensors header length {header_length} exceeds "
                f"the {size}B container"
            )
        raw_header = handle.read(header_length)
    try:
        header = _strict_json_loads(
            raw_header.decode("utf-8"), where=str(path)
        )
    except UnicodeDecodeError as exc:
        raise AssertionError(f"{path}: invalid safetensors JSON header") from exc
    if not isinstance(header, Mapping):
        raise TypeError(f"{path}: safetensors header is not an object")

    spans: dict[str, int] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, Mapping):
            raise TypeError(f"{path}: tensor {name!r} header is not an object")
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise AssertionError(f"{path}: tensor {name!r} has invalid offsets")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in offsets):
            raise AssertionError(
                f"{path}: tensor {name!r} offsets must be integers"
            )
        start, end = offsets
        if start < 0 or end < start:
            raise AssertionError(
                f"{path}: tensor {name!r} has invalid span [{start}, {end})"
            )
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        if dtype not in _SAFETENSORS_DTYPE_BITS:
            raise AssertionError(
                f"{path}: tensor {name!r} has unsupported dtype {dtype!r}"
            )
        if not isinstance(shape, list) or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
            for dim in shape
        ):
            raise AssertionError(
                f"{path}: tensor {name!r} has invalid shape {shape!r}"
            )
        expected_bits = int(math.prod(shape)) * _SAFETENSORS_DTYPE_BITS[dtype]
        expected_span = (expected_bits + 7) // 8
        if end - start != expected_span:
            raise AssertionError(
                f"{path}: tensor {name!r} span is {end - start}B but "
                f"{dtype}{tuple(shape)} requires {expected_span}B"
            )
        spans[str(name)] = end - start
        ranges.append((start, end, str(name)))

    previous_end = 0
    for start, end, name in sorted(ranges):
        if start != previous_end:
            relation = "overlaps" if start < previous_end else "leaves a gap after"
            raise AssertionError(
                f"{path}: tensor {name!r} {relation} the preceding data span "
                f"(expected offset {previous_end}, got {start})"
            )
        previous_end = end
    data_start = 8 + int(header_length)
    if data_start + previous_end != size:
        raise AssertionError(
            f"{path}: header plus tensor extent is {data_start + previous_end}B "
            f"but container is {size}B"
        )
    return spans


def _safetensors_tensor_payload_sha256(
    path: Path,
    tensor_names: Sequence[str],
) -> dict[str, str]:
    """Hash exact serialized data spans after structural validation."""
    spans = _safetensors_data_spans(path)
    requested = {str(name) for name in tensor_names}
    missing = sorted(requested - set(spans))
    if missing:
        raise AssertionError(
            f"{path}: cannot hash missing safetensors tensors {missing[:12]}"
        )
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        (header_length,) = struct.unpack("<Q", raw_length)
        header = _strict_json_loads(
            handle.read(header_length).decode("utf-8"), where=str(path)
        )
        data_start = 8 + int(header_length)
        out: dict[str, str] = {}
        for name in sorted(requested):
            start, end = header[name]["data_offsets"]
            handle.seek(data_start + int(start))
            raw = handle.read(int(end) - int(start))
            if len(raw) != int(end) - int(start):
                raise AssertionError(
                    f"{path}: truncated payload while hashing {name!r}"
                )
            out[name] = hashlib.sha256(raw).hexdigest()
    return out


def cb_export_artifact_inventory(
    out_dir: str | Path,
    *,
    serialized_payload: Mapping[str, object],
    cb_tensor_names: Sequence[str],
    codebook_file: str | None,
    expected_model_files: Sequence[str] | None = None,
    whole_artifact_budget_bytes: int | None = None,
) -> dict:
    """Inventory an already-written CB export and assert both byte scopes.

    ``serialized_payload`` is the analytic CB tensor-data contract.  The
    returned ``export_directory_bytes`` is a different, measured quantity: all
    regular files below ``out_dir``, including safetensors headers, the
    codebook container header, JSON configs, tokenizer files, and any other
    copied sidecars.  Keeping both fields prevents the allocator's payload
    budget from being misreported as an exact filesystem size.
    """
    root = Path(out_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"CB export directory does not exist: {root}")
    files = {
        path.relative_to(root).as_posix(): int(path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise AssertionError(f"{root}: CB export produced no files")

    if expected_model_files is not None:
        # The layout says whether an index belongs here, so nothing has to be
        # told: a single `model.safetensors` must NOT carry one (a leftover
        # index from an earlier sharded run would route a loader at containers
        # this export never wrote), and a `model-XXXXX-of-YYYYY` run MUST, and
        # that index must name exactly the planned containers.
        kind, _count = describe_container_layout(expected_model_files)
        index_present = SHARD_INDEX_NAME in files
        if kind == "single" and index_present:
            raise AssertionError(
                f"{root}: unexpected/stale {SHARD_INDEX_NAME} is "
                "present beside the fresh CB export plan"
            )
        if kind == "sharded":
            if not index_present:
                raise AssertionError(
                    f"{root}: sharded CB export is missing {SHARD_INDEX_NAME}"
                )
            index = json.loads((root / SHARD_INDEX_NAME).read_text())
            mapped = set((index.get("weight_map") or {}).values())
            if mapped != {str(name) for name in expected_model_files}:
                raise AssertionError(
                    f"{root}: {SHARD_INDEX_NAME} names containers outside the "
                    f"fresh export plan: index={sorted(mapped)}, "
                    f"expected={sorted(expected_model_files)}"
                )
        if kind == "other":
            raise AssertionError(
                f"{root}: CB export plan is neither a single container nor a "
                f"complete shard run: {sorted(expected_model_files)}"
            )

    stale_codebook_files = sorted(
        name
        for name in files
        if Path(name).suffix == ".pqcb" and name != codebook_file
    )
    if stale_codebook_files:
        raise AssertionError(
            f"{root}: unexpected/stale CB codebook sidecar files are present "
            f"outside the fresh export plan: {stale_codebook_files[:12]}"
        )

    if expected_model_files is not None:
        expected_containers = {str(name) for name in expected_model_files}
        actual_containers = {
            name for name in files if Path(name).suffix == ".safetensors"
        }
        if actual_containers != expected_containers:
            raise AssertionError(
                f"{root}: model safetensors files differ from the fresh export "
                f"plan: expected={sorted(expected_containers)}, "
                f"actual={sorted(actual_containers)}"
            )

    container_spans: dict[str, dict[str, int]] = {}
    for relative in files:
        path = root / relative
        if path.suffix == ".safetensors" or (
            codebook_file is not None and relative == codebook_file
        ):
            container_spans[relative] = _safetensors_data_spans(path)

    expected_names = {str(name) for name in cb_tensor_names}
    cb_bases = {
        name[: -len(".cb_qweight")]
        for name in expected_names
        if name.endswith(".cb_qweight")
    }
    unexpected_cb_tensors: list[str] = []
    reserved_suffixes = (
        ".cb_qweight",
        ".weight",
        ".weight_scale",
        ".weight_scale_inv",
        ".weight_global_scale",
        ".input_global_scale",
    )
    for relative, spans in container_spans.items():
        if codebook_file is not None and relative == codebook_file:
            continue
        for name in spans:
            belongs_to_expected_cb = any(
                name.startswith(f"{base}.") for base in cb_bases
            )
            if (
                (name.endswith(".cb_qweight") or belongs_to_expected_cb)
                and name.endswith(reserved_suffixes)
                and name not in expected_names
            ):
                unexpected_cb_tensors.append(f"{relative}:{name}")
    if unexpected_cb_tensors:
        raise AssertionError(
            f"{root}: unexpected/stale CB tensors are present outside the "
            f"export plan: {sorted(unexpected_cb_tensors)[:12]}"
        )
    found_names: dict[str, tuple[str, int]] = {}
    for relative, spans in container_spans.items():
        if codebook_file is not None and relative == codebook_file:
            continue
        for name, span in spans.items():
            if name not in expected_names:
                continue
            if name in found_names:
                raise AssertionError(
                    f"{root}: CB tensor {name!r} appears in both "
                    f"{found_names[name][0]!r} and {relative!r}"
                )
            found_names[name] = (relative, int(span))
    if set(found_names) != expected_names:
        raise AssertionError(
            f"{root}: final CB tensor inventory differs from the export plan: "
            f"missing={sorted(expected_names - set(found_names))}, "
            f"extra={sorted(set(found_names) - expected_names)}"
        )
    cb_tensor_bytes = sum(span for _relative, span in found_names.values())

    codebook_spans = (
        container_spans.get(codebook_file, {}) if codebook_file is not None else {}
    )
    expected_codebook_names = {
        str(ref)
        for sidecar in serialized_payload.get("sidecars", [])
        if isinstance(sidecar, Mapping)
        for ref in sidecar.get("codebook_ref", [])
    }
    if set(codebook_spans) != expected_codebook_names:
        raise AssertionError(
            f"{root}: final codebook tensors differ from the serialized "
            f"identity: expected={sorted(expected_codebook_names)}, "
            f"actual={sorted(codebook_spans)}"
        )
    expected_codebook_digests: dict[str, str] = {}
    for sidecar in serialized_payload.get("sidecars", []):
        if not isinstance(sidecar, Mapping):
            continue
        refs = sidecar.get("codebook_ref", [])
        digests = sidecar.get("content_sha256")
        if digests is None:
            continue
        if not isinstance(refs, list) or not isinstance(digests, list) or (
            len(refs) != len(digests)
        ):
            raise AssertionError(
                f"{root}: invalid codebook digest identity"
            )
        for ref, digest in zip(refs, digests):
            ref_name = str(ref)
            digest_value = str(digest).lower()
            previous = expected_codebook_digests.setdefault(
                ref_name, digest_value
            )
            if previous != digest_value:
                raise AssertionError(
                    f"{root}: conflicting codebook digests for "
                    f"{ref_name!r}"
                )
    actual_codebook_digests: dict[str, str] = {}
    if expected_codebook_digests:
        if codebook_file is None:
            raise AssertionError(
                f"{root}: codebook identity has no sidecar file"
            )
        actual_codebook_digests = _safetensors_tensor_payload_sha256(
            root / codebook_file,
            sorted(expected_codebook_digests),
        )
        mismatched = sorted(
            name
            for name, digest in expected_codebook_digests.items()
            if actual_codebook_digests.get(name) != digest
        )
        if mismatched:
            raise AssertionError(
                f"{root}: final codebook bytes differ from their "
                f"content identity: {mismatched[:12]}"
            )
    cb_codebook_bytes = sum(codebook_spans.values())
    expected_tensor_bytes = int(serialized_payload.get("tensor_payload_bytes", 0))
    expected_codebook_bytes = int(
        serialized_payload.get("codebook_sidecar_bytes", 0)
    )
    if cb_tensor_bytes != expected_tensor_bytes:
        raise AssertionError(
            f"{root}: final CB tensor data spans are {cb_tensor_bytes}B, "
            f"accounting expected {expected_tensor_bytes}B"
        )
    if cb_codebook_bytes != expected_codebook_bytes:
        raise AssertionError(
            f"{root}: final codebook data spans are {cb_codebook_bytes}B, "
            f"accounting expected {expected_codebook_bytes}B"
        )

    container_bytes = sum(files[name] for name in container_spans)
    tensor_data_bytes = sum(
        sum(spans.values()) for spans in container_spans.values()
    )
    directory_bytes = sum(files.values())
    cb_payload_bytes = cb_tensor_bytes + cb_codebook_bytes
    expected_total = int(serialized_payload.get("total_bytes", 0))
    if cb_payload_bytes != expected_total:
        raise AssertionError(
            f"{root}: final CB payload is {cb_payload_bytes}B, accounting "
            f"expected {expected_total}B"
        )
    if whole_artifact_budget_bytes is not None:
        if isinstance(whole_artifact_budget_bytes, bool) or int(
            whole_artifact_budget_bytes
        ) < 0:
            raise ValueError("whole_artifact_budget_bytes must be a nonnegative int")
        whole_artifact_budget_bytes = int(whole_artifact_budget_bytes)
    return {
        "schema": CB_EXPORT_ARTIFACT_INVENTORY_SCHEMA,
        "scope": "all_regular_files_recursive",
        "file_bytes": files,
        "export_directory_bytes": int(directory_bytes),
        "safetensors_container_bytes": int(container_bytes),
        "safetensors_tensor_data_bytes": int(tensor_data_bytes),
        "safetensors_container_overhead_bytes": int(
            container_bytes - tensor_data_bytes
        ),
        "non_safetensors_file_bytes": int(directory_bytes - container_bytes),
        "cb_serialized_payload_bytes": int(cb_payload_bytes),
        "cb_tensor_payload_bytes": int(cb_tensor_bytes),
        "cb_codebook_sidecar_bytes": int(cb_codebook_bytes),
        **({
            "cb_codebook_content_sha256": dict(sorted(
                actual_codebook_digests.items()
            )),
        } if actual_codebook_digests else {}),
        **({
            "whole_artifact_budget_bytes": whole_artifact_budget_bytes,
        } if whole_artifact_budget_bytes is not None else {}),
    }


def finalize_cb_export_artifact_inventory(
    out_dir: str | Path,
    quant_config: dict,
    *,
    serialized_payload: Mapping[str, object],
    cb_tensor_names: Sequence[str],
    codebook_file: str | None,
    expected_model_files: Sequence[str] | None = None,
    whole_artifact_budget_bytes: int | None = None,
) -> dict:
    """Write ``quant_config.json`` with a self-consistent final inventory.

    The inventory includes ``quant_config.json`` itself.  Its byte length can
    change when the measured totals are embedded, so write/measure iterations
    continue until the embedded inventory equals the bytes on disk.  The
    representation contains sizes rather than a self-hash and converges in a
    handful of iterations; failure to converge is a hard exporter error.

    Derived budget state (headroom and within-budget) is deliberately not
    persisted in the self-sized JSON.  Embedding decimal headroom can create a
    two-cycle at digit boundaries: changing 10 to 9 shrinks the JSON by one
    byte, which changes the headroom back to 10.  The final hard check derives
    those values from the stable directory size instead.
    """
    root = Path(out_dir)
    provenance = quant_config.setdefault("provenance", {})
    if not isinstance(provenance, dict):
        raise TypeError("quant_config provenance must be an object")
    provenance.setdefault(
        "artifact_inventory",
        {
            "schema": CB_EXPORT_ARTIFACT_INVENTORY_SCHEMA,
            "scope": "pending_final_write",
        },
    )
    config_path = root / "quant_config.json"
    for _attempt in range(16):
        config_path.write_text(json.dumps(quant_config, indent=2, sort_keys=True))
        inventory = cb_export_artifact_inventory(
            root,
            serialized_payload=serialized_payload,
            cb_tensor_names=cb_tensor_names,
            codebook_file=codebook_file,
            expected_model_files=expected_model_files,
            whole_artifact_budget_bytes=whole_artifact_budget_bytes,
        )
        if provenance.get("artifact_inventory") == inventory:
            if whole_artifact_budget_bytes is not None and (
                inventory["export_directory_bytes"]
                > whole_artifact_budget_bytes
            ):
                overage_bytes = int(
                    inventory["export_directory_bytes"]
                    - whole_artifact_budget_bytes
                )
                raise RuntimeError(
                    f"{root}: exact recursive export size is "
                    f"{inventory['export_directory_bytes']}B, exceeding the "
                    f"hard whole-artifact budget of "
                    f"{whole_artifact_budget_bytes}B by "
                    f"{overage_bytes}B"
                )
            return inventory
        provenance["artifact_inventory"] = inventory
    raise AssertionError(
        f"{root}: quant_config artifact inventory did not reach a byte-size "
        "fixed point after 16 writes"
    )


def validate_cb_sidecar_tensors(
    breakdown: Mapping[str, object],
    tensors: Mapping[str, object],
    *,
    where: str,
) -> int:
    """Assert that materialized sidecars match a payload identity exactly.

    Kept torch-free by using the small tensor protocol shared by torch tensors:
    ``dtype``, ``shape``, ``numel()``, and ``element_size()``. Returns the
    actual tensor payload bytes after validating refs, FP16 dtype, shapes, and
    the aggregate byte total.
    """
    raw_sidecars = breakdown.get("sidecars", [])
    if not isinstance(raw_sidecars, list):
        raise TypeError(f"{where}: CB payload sidecars are not a list")
    expected: dict[str, tuple[int, ...]] = {}
    for sidecar in raw_sidecars:
        if not isinstance(sidecar, Mapping):
            raise TypeError(f"{where}: CB sidecar identity is not an object")
        refs = sidecar.get("codebook_ref", [])
        shapes = sidecar.get("subtable_shapes", [])
        if not isinstance(refs, list) or not isinstance(shapes, list):
            raise TypeError(f"{where}: CB sidecar refs/shapes are not lists")
        if len(refs) != len(shapes):
            raise AssertionError(
                f"{where}: CB sidecar has {len(refs)} refs but "
                f"{len(shapes)} shapes"
            )
        for ref_name, ref_shape in zip(refs, shapes):
            name = str(ref_name)
            shape_tuple = tuple(int(dim) for dim in ref_shape)
            previous = expected.setdefault(name, shape_tuple)
            if previous != shape_tuple:
                raise AssertionError(
                    f"{where}: {name} has conflicting expected shapes "
                    f"{previous} and {shape_tuple}"
                )
    if set(tensors) != set(expected):
        raise AssertionError(
            f"{where}: emitted CB sidecars do not match accounting identity: "
            f"expected={sorted(expected)}, actual={sorted(tensors)}"
        )
    actual_bytes = 0
    for name, expected_shape in expected.items():
        tensor = tensors[name]
        dtype_name = str(getattr(tensor, "dtype", "")).removeprefix("torch.")
        shape = tuple(int(dim) for dim in getattr(tensor, "shape", ()))
        if dtype_name != "float16" or shape != expected_shape:
            raise AssertionError(
                f"{where}: {name} emitted {dtype_name}{shape}, accounting "
                f"identity requires float16{expected_shape}"
            )
        actual_bytes += int(tensor.numel()) * int(tensor.element_size())
    expected_bytes = int(breakdown.get("codebook_sidecar_bytes", 0))
    if actual_bytes != expected_bytes:
        raise AssertionError(
            f"{where}: emitted CB sidecars are {actual_bytes}B, accounting "
            f"expected {expected_bytes}B"
        )
    return actual_bytes


def _legacy_context_from_sources(
    assignment: Mapping[str, str],
    codebook_sources: Mapping[str, object] | None,
    *,
    scale_coding: str,
) -> CBSerializationContext:
    sources = codebook_sources or {}
    kinds: set[str] = set()
    refs: dict[str, str | Sequence[str]] = {}
    for qname, format_name in assignment.items():
        if not is_cb_format(format_name):
            continue
        raw = sources.get(qname)
        kind = "lattice"
        logical_ref = None
        if isinstance(raw, str):
            candidate = raw.strip().lower()
            if candidate in {"lattice", "learned"}:
                kind = candidate
        elif isinstance(raw, Mapping):
            if "learned" in raw:
                kind = "learned"
            elif "lattice" in raw:
                kind = "lattice"
            else:
                candidate = str(raw.get("kind", raw.get("source", "lattice"))).lower()
                if candidate in {"lattice", "learned"}:
                    kind = candidate
            logical_ref = raw.get("group") or raw.get("shared_group") or raw.get(
                "codebook_group"
            )
        kinds.add(kind)
        if logical_ref:
            count = len(codebook_subtable_shapes(format_name))
            base = f"cb_codebook.{logical_ref}.{str(format_name).strip().upper()}"
            refs[qname] = base if count == 1 else [
                f"{base}.sub{index}" for index in range(count)
            ]
    if len(kinds) > 1:
        raise ValueError(
            "cb_footprint compatibility wrapper cannot represent mixed lattice/"
            "learned sources in one artifact context; use "
            "cb_assignment_payload_breakdown with exact codebook refs"
        )
    source = next(iter(kinds), "lattice")
    from .nvfp4_activation_contract import (
        NVFP4_ACTIVATION_CONTRACT_SCHEMA,
        NVFP4_ACTIVATION_EXECUTION,
    )

    return CBSerializationContext(
        scale_coding=scale_coding,
        codebook_source=source,
        activation_contract=NVFP4_ACTIVATION_CONTRACT_SCHEMA,
        activation_execution=NVFP4_ACTIVATION_EXECUTION,
        codebook_refs=refs or None,
    )


def cb_footprint(
    assignment: Mapping[str, str],
    shapes: Mapping[str, tuple[int, ...]],
    *,
    codebook_sources: Mapping[str, object] | None = None,
    scale_coding: str = PRODUCTION_FP4_SCALE_CODING,
    context: CBSerializationContext | None = None,
) -> dict:
    """Backwards-compatible mixed-assignment footprint wrapper.

    Unlike the obsolete Phase-0 formula, lattice tables are real FP16
    sidecars and are charged, FP4 has no weight-global scalar, and the static
    production variant charges one activation scalar per FP4 target on top of
    layout-v2. Pass ``context=CBSerializationContext.legacy_v1(...)`` to
    reproduce a legacy layout-v1 artifact explicitly.
    """
    ctx = context or _legacy_context_from_sources(
        assignment, codebook_sources, scale_coding=scale_coding
    )
    cb_assignment = {
        qname: fmt for qname, fmt in assignment.items() if is_cb_format(fmt)
    }
    cb_breakdown = cb_assignment_payload_breakdown(
        cb_assignment, shapes, context=ctx
    )
    non_cb_bytes = 0
    n_params = 0
    per_tensor: dict[str, dict] = {}
    sidecar_first_owner: set[str] = set()
    for qname, format_name in assignment.items():
        if qname not in shapes:
            raise KeyError(f"cb_footprint: no shape for {qname!r}")
        shape = tuple(int(dim) for dim in shapes[qname])
        params = int(math.prod(shape)) if shape else 1
        n_params += params
        if is_cb_format(format_name):
            item = cb_breakdown["per_tensor"][qname]
            charged = 0
            tensor_sidecars = item.get(
                "sidecar_identities", (item["sidecar_identity"],)
            )
            for sidecar_identity in tensor_sidecars:
                sidecar_key = _identity_key(sidecar_identity)
                if sidecar_key not in sidecar_first_owner:
                    sidecar_first_owner.add(sidecar_key)
                    charged += int(sidecar_identity["payload_bytes"])
            grid, _mode, k = _cb_info(format_name)  # type: ignore[misc]
            per_tensor[qname] = {
                "format": str(format_name),
                "k": k,
                "cb_family": "nvfp4" if grid == "fp4" else "fp8",
                "params": params,
                "body_bytes": int(item["packed_weight_bytes"]),
                "global_scale_bytes": 0,
                "input_global_scale_bytes": int(
                    item["input_global_scale_bytes"]
                ),
                "channel_scale_bytes": int(item["fp8_row_scale_bytes"]),
                "sidecar_bytes": charged,
                "codebook_source": codebook_source_for_format(
                    format_name, ctx
                ),
                "body_bpw": 8.0 * int(item["packed_weight_bytes"]) / max(params, 1),
                "serialization_identity": item["identity"],
            }
        else:
            spec = fr.get_format(str(format_name))
            body = int(spec.memory_bytes_for_shape(shape))
            non_cb_bytes += body
            per_tensor[qname] = {
                "format": str(format_name),
                "k": None,
                "cb_family": None,
                "params": params,
                "body_bytes": body,
                "global_scale_bytes": 0,
                "channel_scale_bytes": 0,
                "sidecar_bytes": 0,
                "codebook_source": "none",
                "body_bpw": 8.0 * body / max(params, 1),
            }
    body_bytes = int(cb_breakdown["index_bytes"] +
                     cb_breakdown["fp4_scale_bytes"] + non_cb_bytes)
    channel_scale_bytes = int(cb_breakdown["fp8_row_scale_bytes"])
    input_global_scale_bytes = int(
        cb_breakdown["input_global_scale_bytes"]
    )
    sidecar_bytes = int(cb_breakdown["codebook_sidecar_bytes"])
    total_bytes = (
        body_bytes
        + channel_scale_bytes
        + input_global_scale_bytes
        + sidecar_bytes
    )
    return {
        "schema": _serialized_payload_schema(ctx),
        "serialization_context": cb_breakdown["context"],
        "total_bytes": total_bytes,
        "body_bytes": body_bytes,
        "sidecar_bytes": sidecar_bytes,
        "codebook_sidecar_bytes": sidecar_bytes,
        "global_scale_bytes": 0,
        "input_global_scale_bytes": input_global_scale_bytes,
        "channel_scale_bytes": channel_scale_bytes,
        "fp8_row_scale_bytes": channel_scale_bytes,
        "index_bytes": int(cb_breakdown["index_bytes"]),
        "fp4_scale_bytes": int(cb_breakdown["fp4_scale_bytes"]),
        "n_params": int(n_params),
        "body_bpw": 8.0 * body_bytes / max(n_params, 1),
        "total_bpw": 8.0 * total_bytes / max(n_params, 1),
        "per_tensor": per_tensor,
        "sidecars": cb_breakdown["sidecars"],
    }
