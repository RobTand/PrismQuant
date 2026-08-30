"""Pinned Gridbook-0.8.11 serving-environment contract.

Gridbook resolves some dispatch choices at import/model-load time and reads a
few CUDA schedule selectors at launch time.  A gold measurement therefore
cannot treat the process environment as incidental provenance: it must clear
the complete known namespace, install one canonical state before importing
Gridbook, and attest that same state in the serving process.

This module is deliberately torch-free and never imports Gridbook.  The
producer/consumer boundary remains the immutable runtime pin plus the explicit
registry below.  ``scan_gridbook_source_environment`` is a compatibility
check for a separately checked-out Gridbook source tree; it is not a runtime
dependency.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
import os
from pathlib import Path
import re
from types import MappingProxyType

from .gridbook_runtime_pin import (
    GridbookRuntimePinError,
    load_gridbook_runtime_pin,
    require_exact_gridbook_runtime_release,
    supports_source_fp8_block128_w8a16,
)


GRIDBOOK_ENVIRONMENT_SCHEMA = "prismaquant.gridbook_environment/1"
PINNED_GRIDBOOK_VERSION = "0.8.11"
# A projection of the single packaged pin, not a second independently edited
# commit constant. Future unresolved pins still surface their fail-closed
# placeholder here; the current released pin is a full immutable commit.
PINNED_GRIDBOOK_COMMIT = load_gridbook_runtime_pin().commit

CATEGORY_EXECUTION = "execution"
CATEGORY_CORRECTNESS_BYPASS = "correctness_bypass"
CATEGORY_RESIDENCY_BUILD = "residency_build"
CATEGORY_RETIRED = "retired"
CATEGORY_DIAGNOSTIC = "diagnostic"


class GridbookEnvironmentError(ValueError):
    """The pinned runtime or observed serving environment is not exact."""


@dataclass(frozen=True)
class GridbookEnvironmentVariable:
    """One Gridbook-0.8.11 environment input and its gold-lane disposition."""

    name: str
    category: str
    canonical_gold_value: str | None
    gridbook_default: str
    accepted_domain: str


def _var(
    name: str,
    category: str,
    canonical_gold_value: str | None,
    gridbook_default: str,
    accepted_domain: str,
) -> GridbookEnvironmentVariable:
    return GridbookEnvironmentVariable(
        name=name,
        category=category,
        canonical_gold_value=canonical_gold_value,
        gridbook_default=gridbook_default,
        accepted_domain=accepted_domain,
    )


# The values and domains below carry forward the audited Gridbook 0.8.4 set
# through the 0.8.5 contract into the released 0.8.11 contract. ``None`` is a
# contract value: the
# variable must be
# absent.  This matters for FUSED_FP4/FUSED_FP4_MOE ("0" is invalid), for the
# expert-chunk override ("0" is invalid), and for CUDA switches whose source
# recognizes only a non-default sentinel rather than a canonical default word.
#
# 2026-08-21, pin advance 0.8.5 -> 0.8.11: two identifiers were added
# (PRISMAQUANT_CB_FP8_GEMV_V2, PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R) and no
# existing ``canonical_gold_value`` changed.  ``gridbook_default`` is
# documentation of what the PINNED runtime does when the name is unset, so the
# three selectors whose unset default moved to "auto" in 0.8.9 now say so.
# The canonical gold values deliberately do NOT follow those defaults: every
# dispatch selector in this table is pinned to the explicit kernel the gold
# evidence was measured on, precisely so a runtime-default change cannot move
# the gold lane's executed kernels without a reviewed decision.  Re-baselining
# gold onto the 0.8.9+ auto dispatch is such a decision; a pin bump is not.
#
# 2026-08-22: PRISMAQUANT_CB_FP4V2_DENSE_R2 was registered ahead of any pin
# advance so the source scan classifies gridbook d4df36e's opt-in dense
# round-2 arm; no existing canonical value moved.
GRIDBOOK_ENVIRONMENT_REGISTRY = (
    _var(
        "GRIDBOOK_MXFP8_DENSE", CATEGORY_EXECUTION, None, "disabled",
        "strict boolean: unset, 0, or 1",
    ),
    _var(
        "PRISMAQUANT_CB_GEMV", CATEGORY_EXECUTION, "inherited",
        "auto (since 0.8.9; 'inherited' is the kill switch)",
        "enum: inherited, auto, or v2",
    ),
    # The routed FP8-CB whole-row GEMV sibling, independent of the FP4
    # selector above (different bytes, kernels and evidence).  Gold pins the
    # kill switch: "0"/off reproduces the pre-0.8.9 inherited dispatch on
    # every routed FP8-CB stack.  "1"/require is never legal for a DSv4 gold
    # serve -- it fails the load on any stack outside the qualified
    # k=28/n_sub=4/type_size=112, K in {2048,4096} cell.
    _var(
        "PRISMAQUANT_CB_FP8_GEMV_V2", CATEGORY_EXECUTION, "0",
        "auto (since 0.8.9; unset means auto)",
        "enum: unset/auto, 1/require, or 0/off",
    ),
    _var(
        "PRISMAQUANT_CB_FUSED_FP4", CATEGORY_EXECUTION, None, "disabled",
        "unset or one of: 1, midm, static_lsq, static_lsq_midm, rowwise, "
        "rowwise_midm; literal 0 is invalid",
    ),
    _var(
        "PRISMAQUANT_CB_FUSED_FP4_MOE", CATEGORY_EXECUTION, None, "disabled",
        "unset or one of: 1, 128, 256, static_lsq, static_lsq128, "
        "static_lsq256, rowwise, rowwise128, rowwise256; literal 0 is invalid",
    ),
    _var(
        "PRISMAQUANT_CB_BF16_SM120", CATEGORY_EXECUTION, "0", "disabled",
        "strict boolean: unset, 0, or 1",
    ),
    _var(
        "PRISMAQUANT_CB_FP4_FUSED_MIDM", CATEGORY_EXECUTION, "0", "disabled",
        "strict boolean: unset, 0, or 1",
    ),
    _var(
        "PRISMAQUANT_CB_MOE_PERSISTENT_B", CATEGORY_EXECUTION, "0",
        "auto (since 0.8.9; 0 is the kill switch)",
        "enum: unset/auto, 1/require, or 0/off",
    ),
    _var(
        "PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG", CATEGORY_EXECUTION, "0", "auto",
        "integer 0..number of compiled persistent-B tile configurations",
    ),
    # A second switch nested UNDER persistent-B: Gridbook's model-load wiring
    # rejects it unless PRISMAQUANT_CB_MOE_PERSISTENT_B=1, so with the lane
    # pinned off above, "0" is the only self-consistent gold value.
    _var(
        "PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R", CATEGORY_EXECUTION, "0",
        "disabled", "strict boolean: unset, 0, or 1",
    ),
    _var(
        "PRISMAQUANT_CB_FUSED_MIDM", CATEGORY_EXECUTION, "1", "enabled",
        "strict boolean: unset, 0, or 1",
    ),
    _var(
        "PRISMAQUANT_CB_GROUPED_TRIM", CATEGORY_EXECUTION, "1", "enabled",
        "strict boolean: unset, 0, or 1",
    ),
    _var(
        "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK", CATEGORY_EXECUTION, None,
        "automatic chunking", "unset or integer >= 1; literal 0 is invalid",
    ),
    _var(
        "PRISMAQUANT_CB_PREFILL_CHUNK_BYTES", CATEGORY_EXECUTION, "1073741824",
        "1073741824 bytes", "integer >= 1",
    ),
    _var(
        "PRISMAQUANT_CB_DECODE_CONTRACT", CATEGORY_EXECUTION, "v1", "v1",
        "contract labels v1 or v2 (the CUDA launcher tests exactly for v2)",
    ),
    _var(
        "PRISMAQUANT_CB_FP8_SCHED", CATEGORY_EXECUTION, None, "double-buffer",
        "unset for double-buffer or legacy for the single-buffer A/B arm",
    ),
    _var(
        "PRISMAQUANT_CB_FP4V2_SCHED", CATEGORY_EXECUTION, None, "single-buffer",
        "unset for single-buffer or db for the double-buffer A/B arm",
    ),
    # An opt-in second instantiation of the dense FP4-v2 GEMV kernels
    # (gridbook d4df36e): pq_env_bool01 parses only unset/0/1 and unset runs
    # the legacy arm, so "0" is the self-consistent gold value until promotion.
    _var(
        "PRISMAQUANT_CB_FP4V2_DENSE_R2", CATEGORY_EXECUTION, "0",
        "disabled", "strict boolean: unset, 0, or 1",
    ),
    _var(
        "PRISMAQUANT_CB_W2_SCHED", CATEGORY_EXECUTION, None,
        "shape-tuned warp schedule",
        "unset for the tuned schedule, legacy, or rowpack",
    ),
    _var(
        "PRISMAQUANT_CB_W2_ROWS", CATEGORY_EXECUTION, None, "8 rows",
        "rowpack-only integer; 4, 8, or 16 take effect",
    ),
    _var(
        "PRISMAQUANT_CB_W2_WARPS", CATEGORY_EXECUTION, None, "no override",
        "integer 1..8 overrides; unset/0 means no override",
    ),
    _var(
        "VLLM_USE_DEEP_GEMM", CATEGORY_EXECUTION, "0",
        "owned by vLLM rather than Gridbook",
        "gold DSv4 contract requires the string 0",
    ),
    _var(
        "PRISMAQUANT_SKIP_CB_CAST_CHECK", CATEGORY_CORRECTNESS_BYPASS, "0",
        "cast check enforced", "1 bypasses correctness; any other value enforces",
    ),
    _var(
        "PRISMAQUANT_PRELOAD_FUSED", CATEGORY_RESIDENCY_BUILD, "0", "disabled",
        "strict boolean: unset, 0, or 1",
    ),
    _var(
        "PRISMAQUANT_CB_EXT_DIR", CATEGORY_RESIDENCY_BUILD, None,
        "Gridbook cache default", "unset or an extension build-directory path",
    ),
    _var(
        "PRISMAQUANT_CUTLASS_INCLUDE", CATEGORY_RESIDENCY_BUILD, None,
        "vLLM-bundled CUTLASS", "unset or a CUTLASS include-directory path",
    ),
    _var(
        "CUDACXX", CATEGORY_RESIDENCY_BUILD, None, "toolchain default",
        "unset or an nvcc executable path",
    ),
    _var(
        "CXX", CATEGORY_RESIDENCY_BUILD, None, "toolchain default",
        "unset or a host C++ compiler path",
    ),
    _var(
        "PRISMAQUANT_CB_DECODE", CATEGORY_RETIRED, None, "retired",
        "must be absent",
    ),
    _var(
        "PRISMAQUANT_CB_EXPAND", CATEGORY_RETIRED, None, "retired",
        "must be absent",
    ),
    _var(
        "PRISMAQUANT_CB_PREFILL", CATEGORY_RETIRED, None, "retired",
        "must be absent",
    ),
    _var(
        "PRISMAQUANT_DEBUG_PREFIXES", CATEGORY_DIAGNOSTIC, None, "disabled",
        "must be absent for the canonical non-debug serve",
    ),
)


def _names_for_category(category: str) -> tuple[str, ...]:
    return tuple(
        item.name for item in GRIDBOOK_ENVIRONMENT_REGISTRY
        if item.category == category
    )


GRIDBOOK_EXECUTION_ENVIRONMENT = _names_for_category(CATEGORY_EXECUTION)
GRIDBOOK_CORRECTNESS_BYPASS_ENVIRONMENT = _names_for_category(
    CATEGORY_CORRECTNESS_BYPASS
)
GRIDBOOK_RESIDENCY_BUILD_ENVIRONMENT = _names_for_category(
    CATEGORY_RESIDENCY_BUILD
)
GRIDBOOK_RETIRED_ENVIRONMENT = _names_for_category(CATEGORY_RETIRED)
GRIDBOOK_DIAGNOSTIC_ENVIRONMENT = _names_for_category(CATEGORY_DIAGNOSTIC)

GRIDBOOK_ENVIRONMENT_ALLOWLIST = tuple(sorted(
    item.name for item in GRIDBOOK_ENVIRONMENT_REGISTRY
))

_canonical_by_name = {
    item.name: item.canonical_gold_value
    for item in sorted(GRIDBOOK_ENVIRONMENT_REGISTRY, key=lambda item: item.name)
}
CANONICAL_GOLD_ENVIRONMENT = MappingProxyType(_canonical_by_name)
CANONICAL_GOLD_SET_ENVIRONMENT = MappingProxyType({
    name: value
    for name, value in _canonical_by_name.items()
    if value is not None
})
CANONICAL_GOLD_CLEARED_ENVIRONMENT = tuple(
    name for name, value in _canonical_by_name.items() if value is None
)


def require_pinned_gridbook_runtime() -> None:
    """Fail if PrismaQuant no longer pins the contract this registry describes."""

    pin = load_gridbook_runtime_pin()
    try:
        require_exact_gridbook_runtime_release(pin)
    except GridbookRuntimePinError as exc:
        raise GridbookEnvironmentError(
            f"Gridbook environment registry requires the exact release: {exc}"
        ) from exc
    if (
        pin.version != PINNED_GRIDBOOK_VERSION
        or not supports_source_fp8_block128_w8a16(pin)
    ):
        raise GridbookEnvironmentError(
            "Gridbook environment registry describes "
            f"{PINNED_GRIDBOOK_VERSION} with source-FP8 W8A16, but the "
            f"packaged pin is version={pin.version!r}, commit={pin.commit!r}, "
            f"version_is_release={pin.version_is_release!r}"
        )


def snapshot_gridbook_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Return the complete deterministic allowlist snapshot, including nulls."""

    source = os.environ if environ is None else environ
    return {name: source.get(name) for name in GRIDBOOK_ENVIRONMENT_ALLOWLIST}


def apply_canonical_gold_environment(
    environ: MutableMapping[str, str] | None = None,
    *,
    require_pin: bool = True,
) -> dict[str, str | None]:
    """Clear then install the exact gold state; call before importing Gridbook."""

    if require_pin:
        require_pinned_gridbook_runtime()
    target = os.environ if environ is None else environ
    for name in GRIDBOOK_ENVIRONMENT_ALLOWLIST:
        target.pop(name, None)
    for name, value in CANONICAL_GOLD_SET_ENVIRONMENT.items():
        target[name] = value
    return snapshot_gridbook_environment(target)


def attest_canonical_gold_environment(
    environ: Mapping[str, str] | None = None,
    *,
    require_pin: bool = True,
) -> dict[str, object]:
    """Return a receipt for the exact gold state or fail on every difference."""

    if require_pin:
        require_pinned_gridbook_runtime()
    observed = snapshot_gridbook_environment(environ)
    mismatches = []
    for name in GRIDBOOK_ENVIRONMENT_ALLOWLIST:
        expected = CANONICAL_GOLD_ENVIRONMENT[name]
        actual = observed[name]
        if actual != expected:
            mismatches.append(
                f"{name}: expected "
                f"{'<unset>' if expected is None else repr(expected)}, observed "
                f"{'<unset>' if actual is None else repr(actual)}"
            )
    if mismatches:
        raise GridbookEnvironmentError(
            "Gridbook gold environment mismatch: " + "; ".join(mismatches)
        )
    return {
        "schema": GRIDBOOK_ENVIRONMENT_SCHEMA,
        "gridbook_version": PINNED_GRIDBOOK_VERSION,
        "gridbook_commit": PINNED_GRIDBOOK_COMMIT,
        "environment": observed,
    }


# Source scanning deliberately sees more than environment reads.  Gridbook
# commonly passes a module constant to ``latched_bool`` and C++ uses getenv;
# scanning every Gridbook/PrismaQuant/vLLM-looking identifier catches a newly
# introduced flag even when the read is indirect.  The following identifiers
# are present in the audited source but are explicitly not runtime environment
# inputs.
#
# ``VLLM_MOE_SKIP_PADDING`` is a vLLM capability Gridbook NAMES but never
# reads: in 0.8.11 it occurs only in the ``_neutralize_moe_padding_sentinel``
# docstring (gridbook/ops.py), and that -1 sentinel normalization is
# unconditional.  It is therefore not an execution input of this gold lane --
# gold serves neither set it nor branch on it.  Where it IS an input (the
# paired DSpark serving profile) it is resolved and attested through
# ``vllm.envs``, not through this registry; see
# ``dspark_serving_profile.GRIDBOOK_087_SOURCE_NON_ENVIRONMENT_IDENTIFIERS``.
GRIDBOOK_SOURCE_NON_ENVIRONMENT_IDENTIFIERS = MappingProxyType({
    "PRISMAQUANT_ARTIFACT_INVENTORY_SCHEMA": "Python schema constant",
    "PRISMAQUANT_CB_W2_": "documentation wildcard for registered W2 knobs",
    "PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE": "retired no-op mentioned in a comment",
    "VLLM_COMPILE": "external vLLM setting mentioned in documentation",
    "VLLM_CUTLASS": "vLLM backend enum member, not an environment variable",
    "VLLM_MOE_SKIP_PADDING": "resolved vLLM capability, not a Gridbook env read",
    "VLLM_TEST_FORCE_FP8_MARLIN": "external vLLM test flag mentioned in prose",
})

_GRIDBOOK_EXPECTED_SOURCE_IDENTIFIERS = frozenset({
    "CUDACXX",
    "CXX",
    "GRIDBOOK_MXFP8_DENSE",
    "PRISMAQUANT_ARTIFACT_INVENTORY_SCHEMA",
    "PRISMAQUANT_CB_BF16_SM120",
    "PRISMAQUANT_CB_DECODE_CONTRACT",
    "PRISMAQUANT_CB_EXT_DIR",
    "PRISMAQUANT_CB_FP4V2_DENSE_R2",
    "PRISMAQUANT_CB_FP4V2_SCHED",
    "PRISMAQUANT_CB_FP4_FUSED_MIDM",
    "PRISMAQUANT_CB_FP8_GEMV_V2",
    "PRISMAQUANT_CB_FP8_SCHED",
    "PRISMAQUANT_CB_FUSED_FP4",
    "PRISMAQUANT_CB_FUSED_FP4_MOE",
    "PRISMAQUANT_CB_FUSED_MIDM",
    "PRISMAQUANT_CB_GEMV",
    "PRISMAQUANT_CB_GROUPED_TRIM",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B_CFG",
    "PRISMAQUANT_CB_MOE_PERSISTENT_B_D2R",
    "PRISMAQUANT_CB_PREFILL",
    "PRISMAQUANT_CB_PREFILL_CHUNK_BYTES",
    "PRISMAQUANT_CB_PREFILL_EXPERT_CHUNK",
    "PRISMAQUANT_CB_W2_",
    "PRISMAQUANT_CB_W2_ROWS",
    "PRISMAQUANT_CB_W2_SCHED",
    "PRISMAQUANT_CB_W2_WARPS",
    "PRISMAQUANT_CUTLASS_INCLUDE",
    "PRISMAQUANT_DEBUG_PREFIXES",
    "PRISMAQUANT_OPS_CUDAGRAPH_UNSAFE",
    "PRISMAQUANT_PRELOAD_FUSED",
    "PRISMAQUANT_SKIP_CB_CAST_CHECK",
    "VLLM_COMPILE",
    "VLLM_CUTLASS",
    "VLLM_MOE_SKIP_PADDING",
    "VLLM_TEST_FORCE_FP8_MARLIN",
    "VLLM_USE_DEEP_GEMM",
})

_SOURCE_SUFFIXES = frozenset({".py", ".cu", ".cuh", ".cc", ".cpp", ".h", ".hpp"})
_PREFIXED_IDENTIFIER_RE = re.compile(
    r"(?<![A-Z0-9_])((?:PRISMAQUANT|GRIDBOOK|VLLM)_[A-Z0-9_]+)"
    r"(?![A-Z0-9_])"
)
_DIRECT_ENVIRONMENT_RES = (
    re.compile(r"os[.]environ[.]get[(]\s*['\"]([A-Z][A-Z0-9_]*)['\"]"),
    re.compile(r"os[.]environ\[\s*['\"]([A-Z][A-Z0-9_]*)['\"]"),
    re.compile(r"(?:std::)?getenv[(]\s*['\"]([A-Z][A-Z0-9_]*)['\"]"),
)


@dataclass(frozen=True)
class GridbookSourceEnvironmentScan:
    """Deterministic result of scanning the separately released package."""

    source_root: str
    identifiers: tuple[str, ...]
    registered_environment: tuple[str, ...]
    classified_non_environment: tuple[str, ...]
    unknown_identifiers: tuple[str, ...]
    missing_expected_identifiers: tuple[str, ...]
    locations: tuple[tuple[str, tuple[str, ...]], ...]


def _gridbook_package_root(source_root: str | os.PathLike[str]) -> tuple[Path, Path]:
    root = Path(source_root).resolve()
    package = root / "gridbook"
    if package.is_dir():
        return root, package
    if root.name == "gridbook" and (root / "lane_select.py").is_file():
        return root.parent, root
    raise GridbookEnvironmentError(
        f"{root}: expected a Gridbook repository root containing gridbook/"
    )


def scan_gridbook_source_environment(
    source_root: str | os.PathLike[str],
) -> GridbookSourceEnvironmentScan:
    """Classify every environment-looking identifier in Gridbook source."""

    repo_root, package_root = _gridbook_package_root(source_root)
    locations: dict[str, set[str]] = {}
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise GridbookEnvironmentError(
                f"cannot scan Gridbook source {path}: {exc}"
            ) from exc
        names = set(_PREFIXED_IDENTIFIER_RE.findall(text))
        for pattern in _DIRECT_ENVIRONMENT_RES:
            names.update(pattern.findall(text))
        relative = str(path.relative_to(repo_root))
        for name in names:
            locations.setdefault(name, set()).add(relative)

    identifiers = tuple(sorted(locations))
    registered = tuple(
        name for name in identifiers if name in CANONICAL_GOLD_ENVIRONMENT
    )
    classified = tuple(
        name for name in identifiers
        if name in GRIDBOOK_SOURCE_NON_ENVIRONMENT_IDENTIFIERS
    )
    known = set(registered) | set(classified)
    unknown = tuple(name for name in identifiers if name not in known)
    missing = tuple(sorted(_GRIDBOOK_EXPECTED_SOURCE_IDENTIFIERS - set(identifiers)))
    location_rows = tuple(
        (name, tuple(sorted(paths))) for name, paths in sorted(locations.items())
    )
    return GridbookSourceEnvironmentScan(
        source_root=str(repo_root),
        identifiers=identifiers,
        registered_environment=registered,
        classified_non_environment=classified,
        unknown_identifiers=unknown,
        missing_expected_identifiers=missing,
        locations=location_rows,
    )


def require_gridbook_source_compatible(
    source_root: str | os.PathLike[str],
) -> GridbookSourceEnvironmentScan:
    """Fail closed when released source adds, removes, or renames an identifier."""

    report = scan_gridbook_source_environment(source_root)
    if report.unknown_identifiers or report.missing_expected_identifiers:
        raise GridbookEnvironmentError(
            "Gridbook environment source drift: unknown="
            f"{list(report.unknown_identifiers)}, missing="
            f"{list(report.missing_expected_identifiers)}"
        )
    return report


__all__ = [
    "CANONICAL_GOLD_CLEARED_ENVIRONMENT",
    "CANONICAL_GOLD_ENVIRONMENT",
    "CANONICAL_GOLD_SET_ENVIRONMENT",
    "GRIDBOOK_CORRECTNESS_BYPASS_ENVIRONMENT",
    "GRIDBOOK_DIAGNOSTIC_ENVIRONMENT",
    "GRIDBOOK_ENVIRONMENT_ALLOWLIST",
    "GRIDBOOK_ENVIRONMENT_REGISTRY",
    "GRIDBOOK_ENVIRONMENT_SCHEMA",
    "GRIDBOOK_EXECUTION_ENVIRONMENT",
    "GRIDBOOK_RESIDENCY_BUILD_ENVIRONMENT",
    "GRIDBOOK_RETIRED_ENVIRONMENT",
    "GRIDBOOK_SOURCE_NON_ENVIRONMENT_IDENTIFIERS",
    "GridbookEnvironmentError",
    "GridbookEnvironmentVariable",
    "GridbookSourceEnvironmentScan",
    "PINNED_GRIDBOOK_COMMIT",
    "PINNED_GRIDBOOK_VERSION",
    "apply_canonical_gold_environment",
    "attest_canonical_gold_environment",
    "require_gridbook_source_compatible",
    "require_pinned_gridbook_runtime",
    "scan_gridbook_source_environment",
    "snapshot_gridbook_environment",
]
