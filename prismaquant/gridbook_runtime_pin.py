"""Strict, torch-free reader for PrismaQuant's immutable Gridbook pin.

This module reads producer/consumer compatibility data only.  It never imports
the external Gridbook package: production compatibility continues to cross the
repository boundary through ``gridbook_runtime_pin.json``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any


GRIDBOOK_RUNTIME_PIN_SCHEMA = "prismaquant.gridbook_runtime_pin.v3"
GRIDBOOK_RUNTIME_REPOSITORY = "https://github.com/RobTand/gridbook.git"
GRIDBOOK_RUNTIME_RELEASE_VERSION = "0.9.1"
GRIDBOOK_RUNTIME_RELEASE_COMMIT = (
    "227420f9821bab7089632ee914f0ba050f82b817"
)
# Historical staging sentinel retained so parsers and third-party tooling can
# reject an unresolved future pin explicitly. The packaged v0.9.1 pin is a full
# immutable commit and does not use this value.
GRIDBOOK_RUNTIME_COMMIT_PENDING = "PENDING_GRIDBOOK_V0_8_11_RELEASE_COMMIT"
# The contract schema and this module's version/commit are ONE decision, not
# three: ``parse_gridbook_runtime_pin`` refuses any payload whose
# ``runtime_contract_schema`` differs from the constant below, so a release
# that moves the schema cannot be pinned by halves. 0.8.11 -> 0.9.1 crosses
# v4 -> v12 and lands 0.9.0's tensor/expert-parallel work and 0.9.1's trellis
# lanes plus the ladder retraction in one step; 0.9.0 was never pinned.
GRIDBOOK_RUNTIME_CONTRACT_SCHEMA = "gridbook.runtime-contract.v12"
# Historical diagnostic floor retained for error prose and third-party callers.
# Capability decisions no longer use it; they read the closed feature map.
GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION = "0.8.4"
GRIDBOOK_REQUIRED_ABI_FEATURES = {
    "routed_moe_per_role_codebook_lut": 1,
    "source_fp8_block128_w8a16": 1,
    "dspark_construction_physical_bridge": 1,
}
_REQUIRED_MEMBERS = {
    "schema",
    "repository",
    "commit",
    "version",
    "version_is_release",
    "runtime_contract_schema",
    "required_abi_features",
}
_FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_PIN_VERSION_RE = re.compile(
    r"[0-9]+(?:[.][0-9]+)*(?:[A-Za-z0-9.+-]*)?"
)
class GridbookRuntimePinError(ValueError):
    """The tracked Gridbook pin is missing or structurally invalid."""


@dataclass(frozen=True)
class GridbookRuntimePin:
    schema: str
    repository: str
    commit: str
    version: str
    version_is_release: bool
    runtime_contract_schema: str
    required_abi_features: Mapping[str, int]

    @property
    def commit_is_resolved(self) -> bool:
        return _FULL_COMMIT_RE.fullmatch(self.commit) is not None


def parse_gridbook_runtime_pin(
    payload: Mapping[str, Any],
    *,
    where: str = "gridbook_runtime_pin.json",
) -> GridbookRuntimePin:
    """Validate the complete v3 pin payload without permissive defaults."""

    if not isinstance(payload, Mapping):
        raise GridbookRuntimePinError(f"{where}: pin must be a JSON object")
    members = set(payload)
    if members != _REQUIRED_MEMBERS:
        raise GridbookRuntimePinError(
            f"{where}: expected exactly {sorted(_REQUIRED_MEMBERS)}, "
            f"got {sorted(members)}"
        )

    schema = payload["schema"]
    if schema != GRIDBOOK_RUNTIME_PIN_SCHEMA:
        raise GridbookRuntimePinError(
            f"{where}: unsupported schema {schema!r}"
        )
    repository = payload["repository"]
    if repository != GRIDBOOK_RUNTIME_REPOSITORY:
        raise GridbookRuntimePinError(
            f"{where}: repository must be {GRIDBOOK_RUNTIME_REPOSITORY!r}"
        )
    commit = payload["commit"]
    if not isinstance(commit, str) or (
        _FULL_COMMIT_RE.fullmatch(commit) is None
        and commit != GRIDBOOK_RUNTIME_COMMIT_PENDING
    ):
        raise GridbookRuntimePinError(
            f"{where}: commit must be a lowercase full 40-hex SHA or the "
            "one fail-closed release placeholder"
        )
    version = payload["version"]
    if not isinstance(version, str) or _PIN_VERSION_RE.fullmatch(version) is None:
        raise GridbookRuntimePinError(
            f"{where}: invalid package version {version!r}"
        )
    version_is_release = payload["version_is_release"]
    if not isinstance(version_is_release, bool):
        raise GridbookRuntimePinError(
            f"{where}: version_is_release must be a JSON boolean"
        )
    runtime_contract_schema = payload["runtime_contract_schema"]
    if runtime_contract_schema != GRIDBOOK_RUNTIME_CONTRACT_SCHEMA:
        raise GridbookRuntimePinError(
            f"{where}: runtime_contract_schema must be "
            f"{GRIDBOOK_RUNTIME_CONTRACT_SCHEMA!r}"
        )
    features = payload["required_abi_features"]
    if not isinstance(features, Mapping) or set(features) != set(
        GRIDBOOK_REQUIRED_ABI_FEATURES
    ):
        raise GridbookRuntimePinError(
            f"{where}: required_abi_features must contain exactly "
            f"{sorted(GRIDBOOK_REQUIRED_ABI_FEATURES)}"
        )
    normalized_features: dict[str, int] = {}
    for name, expected in GRIDBOOK_REQUIRED_ABI_FEATURES.items():
        value = features[name]
        if type(value) is not int or value != expected:
            raise GridbookRuntimePinError(
                f"{where}: required_abi_features.{name} must equal "
                f"integer {expected}"
            )
        normalized_features[name] = value
    if commit == GRIDBOOK_RUNTIME_COMMIT_PENDING and version_is_release:
        raise GridbookRuntimePinError(
            f"{where}: unresolved release commit cannot be marked released"
        )
    return GridbookRuntimePin(
        schema=schema,
        repository=repository,
        commit=commit,
        version=version,
        version_is_release=version_is_release,
        runtime_contract_schema=runtime_contract_schema,
        required_abi_features=MappingProxyType(normalized_features),
    )


def _reject_duplicate_members(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GridbookRuntimePinError(
                f"gridbook_runtime_pin.json: duplicate JSON member {key!r}"
            )
        result[key] = value
    return result


@lru_cache(maxsize=1)
def load_gridbook_runtime_pin() -> GridbookRuntimePin:
    """Read and validate the one packaged Gridbook runtime pin."""

    location = (
        Path(__file__).resolve().parent
        / "gridbook_runtime"
        / "gridbook_runtime_pin.json"
    )
    try:
        raw = location.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise GridbookRuntimePinError(
            f"{location}: cannot read Gridbook runtime pin: {exc}"
        ) from exc
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_members)
    except GridbookRuntimePinError:
        raise
    except json.JSONDecodeError as exc:
        raise GridbookRuntimePinError(
            f"{location}: malformed JSON: {exc}"
        ) from exc
    return parse_gridbook_runtime_pin(payload, where=str(location))


def supports_routed_moe_per_role_codebook_lut(
    pin: GridbookRuntimePin,
) -> bool:
    """Whether the consumer contract explicitly requires the routed LUT ABI."""

    return pin.required_abi_features.get(
        "routed_moe_per_role_codebook_lut"
    ) == 1


def supports_source_fp8_block128_w8a16(pin: GridbookRuntimePin) -> bool:
    """Whether the consumer contract explicitly requires raw-source W8A16."""

    return pin.required_abi_features.get("source_fp8_block128_w8a16") == 1


def supports_dspark_construction_physical_bridge(
    pin: GridbookRuntimePin,
) -> bool:
    """Whether the consumer contract requires the DSpark physical bridge ABI."""

    return pin.required_abi_features.get(
        "dspark_construction_physical_bridge"
    ) == 1


def require_resolved_gridbook_runtime_pin(pin: GridbookRuntimePin) -> None:
    """Refuse release work while the conspicuous commit placeholder remains."""

    if not pin.commit_is_resolved:
        raise GridbookRuntimePinError(
            f"Gridbook v{GRIDBOOK_RUNTIME_RELEASE_VERSION} exact release "
            "commit is still unresolved"
        )


def require_exact_gridbook_runtime_release(pin: GridbookRuntimePin) -> None:
    """Require the one reviewed release behind PrismaQuant's current ABI."""

    require_resolved_gridbook_runtime_pin(pin)
    if (
        pin.version != GRIDBOOK_RUNTIME_RELEASE_VERSION
        or pin.commit != GRIDBOOK_RUNTIME_RELEASE_COMMIT
        or pin.version_is_release is not True
    ):
        raise GridbookRuntimePinError(
            "Gridbook runtime must be exact released "
            f"v{GRIDBOOK_RUNTIME_RELEASE_VERSION} commit "
            f"{GRIDBOOK_RUNTIME_RELEASE_COMMIT}; observed "
            f"version={pin.version!r} commit={pin.commit!r} "
            f"version_is_release={pin.version_is_release!r}"
        )


__all__ = [
    "GRIDBOOK_REQUIRED_ABI_FEATURES",
    "GRIDBOOK_ROUTED_MOE_PER_ROLE_CODEBOOK_LUT_MIN_VERSION",
    "GRIDBOOK_RUNTIME_COMMIT_PENDING",
    "GRIDBOOK_RUNTIME_CONTRACT_SCHEMA",
    "GRIDBOOK_RUNTIME_RELEASE_COMMIT",
    "GRIDBOOK_RUNTIME_RELEASE_VERSION",
    "GRIDBOOK_RUNTIME_REPOSITORY",
    "GRIDBOOK_RUNTIME_PIN_SCHEMA",
    "GridbookRuntimePin",
    "GridbookRuntimePinError",
    "load_gridbook_runtime_pin",
    "parse_gridbook_runtime_pin",
    "require_exact_gridbook_runtime_release",
    "require_resolved_gridbook_runtime_pin",
    "supports_dspark_construction_physical_bridge",
    "supports_routed_moe_per_role_codebook_lut",
    "supports_source_fp8_block128_w8a16",
]
