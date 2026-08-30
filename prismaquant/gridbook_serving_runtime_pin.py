"""Strict reader for the current Gridbook serving-runtime release pin.

``gridbook_runtime_pin.v3`` is the PRODUCER pin, which since 2026-08-21 names
the same release as this one (0.9.1 since 2026-08-30) and is held in lockstep
with it by
``tests/test_gridbook_runtime_boundary.py``.  Serving remains a distinct
consumer boundary: it additionally requires an exact reviewed wheel digest,
which the producer pin does not carry.
This module deliberately accepts conspicuous pending sentinels structurally
so a release patch can be reviewed before Gridbook is cut, while every live
serve/ship gate rejects those sentinels.
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


GRIDBOOK_SERVING_RUNTIME_PIN_SCHEMA = (
    "prismaquant.gridbook_serving_runtime_pin.v1"
)
GRIDBOOK_SERVING_RUNTIME_REPOSITORY = (
    "https://github.com/RobTand/gridbook.git"
)
GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION = "0.9.1"
GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING = (
    "PENDING_GRIDBOOK_V0_8_11_RELEASE_COMMIT"
)
GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING = (
    "PENDING_GRIDBOOK_V0_8_11_WHEEL_SHA256"
)
# Resolved 2026-08-30 against the v0.9.1 release.  Neither value is guessed,
# and neither is transcribed from the other's source:
#   commit  -- annotated tag v0.9.1, which the release workflow refuses to
#              build unless the commit is reachable from origin/master.  It is
#              also the build recorded in the image tag
#              `gridbook:0.9.1-clean-227420f`.
#   wheel   -- read out of that image's installed distribution, from the PEP
#              610 `direct_url.json` `archive_info.hashes.sha256` of
#              gridbook-0.9.1-py3-none-any.whl.  This is the digest of the
#              wheel that is actually importable at serve time, which is the
#              only digest a serving pin may assert; a locally rebuilt wheel is
#              a DIFFERENT archive and must not be substituted here without
#              re-reading it from the served image.  (Confirmed here: a local
#              `python -m build` of the tag produced 7141acf9..., a different
#              archive from the published one below.  Wheels are not
#              byte-reproducible and that is precisely why this field is read
#              from the image and not from a build.)
#
# As with 0.8.7 through 0.8.11, this digest IS the PyPI wheel's: the image was
# built by installing the published `gridbook==0.9.1` archive from a local
# file rather than rebuilding it, so `pip download gridbook==0.9.1` satisfies
# the pin instead of tripping the wheel-cache trap documented in
# docs/audits/serving_wheel_cache_poisoning_2026-08-14.md.  It is additionally
# the digest the release workflow's own build job recorded and re-checked at
# every later step, so the CI receipt, the PyPI archive, the GitHub Release
# asset and the served image all name one archive.
#
# WHY 0.9.1: it is the first release that packages a lane-eligibility table
# PrismaQuant can read as a gate input, which is the whole reason this pin
# moves.  Contract schema v4 -> v12; lane table
# `gridbook.lane-eligibility.v3`.  The bump crosses two releases because 0.9.0
# was never pinned: it brings 0.9.0's tensor/expert-parallel support and
# 0.9.1's two trellis serving lanes, the CB ladder retraction, and the table
# itself.
#
# What it changes for a CB serve, honestly: the CB codec, rung law and default
# dispatch are unchanged (gridbook/fp8_fused_lane.py still names the fused
# mid-M reader surface (28, 32, 36, 40, 44, 48); NVFP4-CB reads and produces
# K12..K24 as before).  What DOES change is that route status stops being
# UNATTESTED-by-absence and becomes a real resolution against a published
# table -- and that table names CB cells on sm_89 and sm_120 only, publishing
# trellis cells alone on sm_121.  A CB export declaring target_platform
# sm_121, which `serving_profile_specs/nvfp4_cb.json` now does, therefore
# REFUSES at `cb_route_status_gate` until Gridbook receipts an sm_121 CB cell
# or the artifact declares a non-native target.  That refusal is the point of
# the gate, not a regression to route around: both CB compile preflights fix
# their capability in code, so no CB receipt for compute 12.1 exists, and
# principle 14 forbids asserting one we have not been given.
#
# WHY 0.8.11 (history): two CUDA-graph capture fixes, no route/codec/default change for
# an eager or FULL_DECODE_ONLY serve.  (a) gridbook#46: the MXFP8 dense lane's
# swizzled-plane offset cache did an unpinned host->device copy on first use,
# which aborted FULL_DECODE_ONLY capture at load; the offsets are pre-warmed
# at load now.  (b) gridbook#47: the grouped routed lanes' padded-route
# helper read a routing-DEPENDENT trim count (and, for the BF16 bridge, the
# per-expert block offsets) on the host, which vLLM 0.27's default
# FULL_AND_PIECEWISE capture of prefill sizes > 16 tokens turned into an
# engine-start death ("operation not permitted when stream is capturing").
# Under capture the fused lanes now launch the static-capacity tile layout
# (the count every captured graph can be correct for -- identical to the
# PRISMAQUANT_CB_GROUPED_TRIM=0 arm); the one lane that chunks by host-read
# per-expert offsets (the OPT-IN sm12x bridge, PRISMAQUANT_CB_BF16_SM120=1)
# refuses capture naming the flag -- the default expand + grouped bridge and
# the persistent-B lane never host-read and capture as-is.  Eager and
# decode-band (<= 16 tokens) behaviour is byte-identical to 0.8.10, so the
# gold environment's recorded routes are unchanged under this wheel.
# MEASURED 2026-08-21 on the shipped 87 GB DSv4 body under this image
# (perf-b1-0811): the card command (FULL_DECODE_ONLY [1,2]) decodes
# 20.53-20.61 tok/s vs 20.54-20.63 on 0.8.10 (unchanged); vLLM's DEFAULT
# FULL_AND_PIECEWISE with capture sizes up to 64 now STARTS (11 piecewise +
# 7 full graphs) and decodes 20.56-20.64 single-stream -- the default
# command no longer needs --compilation-config to come up.
#
# WHY 0.8.10 (history): 0.8.9 is the release that defaults the qualified CB kernels on
# (three selectors tri-state with unset -> auto; every EXPLICIT spelling keeps
# its exact prior semantics, so the canonical gold environment -- which pins
# PRISMAQUANT_CB_MOE_PERSISTENT_B=0 and PRISMAQUANT_CB_GEMV=inherited --
# reproduces the recorded gold routes unchanged under this wheel.
# PRISMAQUANT_CB_FP8_GEMV_V2 was originally left OUT of that closed registry
# on the reasoning that the registry was 0.8.5 release evidence whose scan
# refuses namespace changes, so a gold replay served the qualified K28 GEMV
# cell in the routed decode band BY DEFAULT -- which the 0.8.9 default-state
# served leg measured end-to-end on the shipped clean 87 GB body: kl_mean
# +0.17 %, PPL -0.06 % vs the gold record, inside the +/-0.7 % cross-session
# KL envelope.  SUPERSEDED 2026-08-21: the PRODUCER pin advanced to 0.8.11, so
# gridbook_environment.py now describes the 0.8.11 namespace and the name is
# registered there with canonical gold value "0", like every other dispatch
# kill switch in that table.  A gold replay therefore now pins the sibling OFF
# and takes the inherited kernel on every routed FP8-CB stack.  That is a
# deliberate determinism choice, not a quality one -- the leg above says the
# two dispatches agree inside the cross-session envelope -- and it stops a
# future runtime default from silently moving the gold lane's kernels.
# Re-baselining gold ONTO the auto dispatch remains available, but it is a
# reviewed re-measurement, not a side effect of a pin bump.)  But 0.8.9
# also shipped a load regression its own suite could not see: the tri-state
# refactor renamed a moe_gemv_select symbol that gridbook/moe_mixed.py still
# imported, so any artifact declaring per_expert_format_groups (a split-bank
# mixed expert stack) died with an ImportError at config.py's dispatch.
# Uniform stacks -- every published artifact -- were unaffected.  0.8.10 is
# 0.8.9 plus that fix and its guards; it changes no route, no kernel, and no
# default for uniform-stack artifacts, so this pin supersedes 0.8.9 with zero
# serving-behaviour delta on everything we ship today.
GRIDBOOK_SERVING_RUNTIME_RELEASE_COMMIT = (
    "227420f9821bab7089632ee914f0ba050f82b817"
)
GRIDBOOK_SERVING_RUNTIME_RELEASE_WHEEL_SHA256 = (
    "cb4d7ad64c5a78d447f427a0aa98790406b6821d02c7f2f5d589d61890abdf9d"
)
GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA = "gridbook.runtime-contract.v12"
GRIDBOOK_SERVING_REQUIRED_ABI_FEATURES = {
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
    "wheel_sha256",
    "runtime_contract_schema",
    "required_abi_features",
}
_FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class GridbookServingRuntimePinError(ValueError):
    """The current serving pin is missing, pending, or malformed."""


@dataclass(frozen=True)
class GridbookServingRuntimePin:
    schema: str
    repository: str
    commit: str
    version: str
    version_is_release: bool
    wheel_sha256: str
    runtime_contract_schema: str
    required_abi_features: Mapping[str, int]

    @property
    def commit_is_resolved(self) -> bool:
        return _FULL_COMMIT_RE.fullmatch(self.commit) is not None

    @property
    def wheel_is_resolved(self) -> bool:
        return _SHA256_RE.fullmatch(self.wheel_sha256) is not None


def parse_gridbook_serving_runtime_pin(
    payload: Mapping[str, Any],
    *,
    where: str = "gridbook_serving_runtime_pin.json",
) -> GridbookServingRuntimePin:
    if not isinstance(payload, Mapping) or set(payload) != _REQUIRED_MEMBERS:
        observed = sorted(payload) if isinstance(payload, Mapping) else []
        raise GridbookServingRuntimePinError(
            f"{where}: expected exactly {sorted(_REQUIRED_MEMBERS)}, "
            f"got {observed}"
        )
    if payload["schema"] != GRIDBOOK_SERVING_RUNTIME_PIN_SCHEMA:
        raise GridbookServingRuntimePinError(
            f"{where}: unsupported schema {payload['schema']!r}"
        )
    if payload["repository"] != GRIDBOOK_SERVING_RUNTIME_REPOSITORY:
        raise GridbookServingRuntimePinError(
            f"{where}: repository differs from the reviewed Gridbook origin"
        )
    commit = payload["commit"]
    if not isinstance(commit, str) or (
        _FULL_COMMIT_RE.fullmatch(commit) is None
        and commit != GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING
    ):
        raise GridbookServingRuntimePinError(
            f"{where}: commit must be full lowercase SHA or the exact pending sentinel"
        )
    wheel = payload["wheel_sha256"]
    if not isinstance(wheel, str) or (
        _SHA256_RE.fullmatch(wheel) is None
        and wheel != GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING
    ):
        raise GridbookServingRuntimePinError(
            f"{where}: wheel_sha256 must be SHA-256 or the exact pending sentinel"
        )
    if payload["version"] != GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION:
        raise GridbookServingRuntimePinError(
            f"{where}: version must be {GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION!r}"
        )
    released = payload["version_is_release"]
    if not isinstance(released, bool):
        raise GridbookServingRuntimePinError(
            f"{where}: version_is_release must be a JSON boolean"
        )
    if (commit == GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING or
            wheel == GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING) and released:
        raise GridbookServingRuntimePinError(
            f"{where}: pending commit/wheel cannot be marked released"
        )
    if payload["runtime_contract_schema"] != (
        GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA
    ):
        raise GridbookServingRuntimePinError(
            f"{where}: serving runtime contract must be "
            f"{GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA}"
        )
    features = payload["required_abi_features"]
    if not isinstance(features, Mapping) or set(features) != set(
        GRIDBOOK_SERVING_REQUIRED_ABI_FEATURES
    ):
        raise GridbookServingRuntimePinError(
            f"{where}: ABI feature closure differs"
        )
    normalized: dict[str, int] = {}
    for name, expected in GRIDBOOK_SERVING_REQUIRED_ABI_FEATURES.items():
        value = features[name]
        if type(value) is not int or value != expected:
            raise GridbookServingRuntimePinError(
                f"{where}: required_abi_features.{name} must equal {expected}"
            )
        normalized[name] = value
    return GridbookServingRuntimePin(
        schema=str(payload["schema"]),
        repository=str(payload["repository"]),
        commit=commit,
        version=str(payload["version"]),
        version_is_release=released,
        wheel_sha256=wheel,
        runtime_contract_schema=str(payload["runtime_contract_schema"]),
        required_abi_features=MappingProxyType(normalized),
    )


def _reject_duplicate_members(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GridbookServingRuntimePinError(
                f"gridbook_serving_runtime_pin.json: duplicate member {key!r}"
            )
        result[key] = value
    return result


@lru_cache(maxsize=1)
def load_gridbook_serving_runtime_pin() -> GridbookServingRuntimePin:
    location = (
        Path(__file__).resolve().parent
        / "gridbook_runtime"
        / "gridbook_serving_runtime_pin.json"
    )
    try:
        payload = json.loads(
            location.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_members,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GridbookServingRuntimePinError(
            f"{location}: cannot read serving pin: {exc}"
        ) from exc
    return parse_gridbook_serving_runtime_pin(payload, where=str(location))


def require_exact_gridbook_serving_runtime_release(
    pin: GridbookServingRuntimePin,
) -> None:
    if not pin.commit_is_resolved or not pin.wheel_is_resolved:
        raise GridbookServingRuntimePinError(
            f"Gridbook {GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION} serving "
            "commit/wheel digest is still pending"
        )
    if (
        pin.commit != GRIDBOOK_SERVING_RUNTIME_RELEASE_COMMIT
        or pin.wheel_sha256 != GRIDBOOK_SERVING_RUNTIME_RELEASE_WHEEL_SHA256
        or pin.version != GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION
        or pin.version_is_release is not True
    ):
        raise GridbookServingRuntimePinError(
            "Gridbook serving runtime differs from the exact reviewed "
            f"{GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION} release"
        )


__all__ = [
    "GRIDBOOK_SERVING_REQUIRED_ABI_FEATURES",
    "GRIDBOOK_SERVING_RUNTIME_COMMIT_PENDING",
    "GRIDBOOK_SERVING_RUNTIME_CONTRACT_SCHEMA",
    "GRIDBOOK_SERVING_RUNTIME_PIN_SCHEMA",
    "GRIDBOOK_SERVING_RUNTIME_RELEASE_COMMIT",
    "GRIDBOOK_SERVING_RUNTIME_RELEASE_VERSION",
    "GRIDBOOK_SERVING_RUNTIME_RELEASE_WHEEL_SHA256",
    "GRIDBOOK_SERVING_RUNTIME_REPOSITORY",
    "GRIDBOOK_SERVING_RUNTIME_WHEEL_SHA256_PENDING",
    "GridbookServingRuntimePin",
    "GridbookServingRuntimePinError",
    "load_gridbook_serving_runtime_pin",
    "parse_gridbook_serving_runtime_pin",
    "require_exact_gridbook_serving_runtime_release",
]
