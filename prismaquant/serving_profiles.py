"""Config-backed serving/runtime constraint profiles.

Serving profiles capture backend-specific legality that should not live as
architecture branches in the allocator: format menus, kernel shape limits, and
other runtime constraints.  The allocator still performs cheap local checks,
but the policy comes from JSON specs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Collection, Mapping

from . import format_registry as fr


SCHEMA = "prismaquant.serving_profile.v1"
SERVING_LANE_SCHEMA = "prismaquant.serving_lane_route.v1"

#: Preference order when several unpredicated eligibility rules cover one lane.
#: ``backed`` wins over a flag-gated route, which wins over an announced
#: fallback, which wins over nothing -- and "nothing" is where an undeclared
#: rule lands, never a default pass.
_LANE_STATUS_RANK = {
    "unbacked": 0,
    "fallback": 1,
    "backed_with_serve_flag": 2,
    "backed": 3,
}

_ELIGIBILITY_TABLE: Any = None


def _cached_eligibility_table(loader):
    """One read of the pinned contract per process (it is immutable)."""
    global _ELIGIBILITY_TABLE
    if _ELIGIBILITY_TABLE is None:
        _ELIGIBILITY_TABLE = loader()
    return _ELIGIBILITY_TABLE


def _reset_eligibility_table_cache() -> None:
    """Test seam: the pinned contract is immutable, monkeypatched ones are not."""
    global _ELIGIBILITY_TABLE
    _ELIGIBILITY_TABLE = None


@dataclass(frozen=True)
class ServingFormatDecision:
    legal: bool
    reason: str | None = None
    detail: str = ""
    rule: str | None = None


@dataclass(frozen=True)
class NameCondition:
    contains: str | None = None
    not_contains: str | None = None
    prefix: str | None = None
    regex: str | None = None
    not_regex: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NameCondition":
        return cls(
            contains=_optional_str(payload.get("contains")),
            not_contains=_optional_str(payload.get("not_contains")),
            prefix=_optional_str(payload.get("prefix")),
            regex=_optional_str(payload.get("regex")),
            not_regex=_optional_str(payload.get("not_regex")),
        )

    def matches(self, name: str) -> bool:
        if self.contains is not None and self.contains not in name:
            return False
        if self.not_contains is not None and self.not_contains in name:
            return False
        if self.prefix is not None and not name.startswith(self.prefix):
            return False
        if self.regex is not None and re.search(self.regex, name) is None:
            return False
        if self.not_regex is not None and re.search(self.not_regex, name) is not None:
            return False
        return True


@dataclass(frozen=True)
class ServingFormatRule:
    id: str
    when: NameCondition = field(default_factory=NameCondition)
    allow_formats: tuple[str, ...] = ()
    deny_formats: tuple[str, ...] = ()
    reason: str = "profile_mismatch"
    detail: str = ""
    # Target-class scoping: "all" (default), "packed_experts" (rank-3 stacked
    # MoE tensors only), or "dense" (everything else). Lets a container
    # declare capabilities that differ between dense Linears and packed
    # expert stacks (e.g. nvfp4_cb carries no stock-CT packed-MoE emission).
    scope: str = "all"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServingFormatRule":
        scope = str(payload.get("scope", "all"))
        if scope not in ("all", "packed_experts", "dense"):
            raise ValueError(
                f"format rule {payload.get('id')!r}: unknown scope {scope!r} "
                f"(expected all|packed_experts|dense)")
        return cls(
            id=str(payload["id"]),
            when=NameCondition.from_dict(payload.get("when") or {}),
            allow_formats=_declared_formats(
                payload.get("allow_formats", ()),
                payload.get("allow_formats_from", ()),
                owner=f"format rule {payload['id']!r} allow-list",
            ),
            deny_formats=_declared_formats(
                payload.get("deny_formats", ()),
                payload.get("deny_formats_from", ()),
                owner=f"format rule {payload['id']!r} deny-list",
            ),
            reason=str(payload.get("reason", "profile_mismatch")),
            detail=str(payload.get("detail", "")),
            scope=scope,
        )

    def check(self, qname: str, fmt: str,
              packed_expert: bool | None = None) -> ServingFormatDecision | None:
        if self.scope == "packed_experts" and packed_expert is not True:
            return None
        if self.scope == "dense" and packed_expert is True:
            return None
        if not self.when.matches(qname):
            return None
        if self.allow_formats and not _format_in(fmt, self.allow_formats):
            return ServingFormatDecision(
                False,
                self.reason,
                self.detail,
                self.id,
            )
        if self.deny_formats and _format_in(fmt, self.deny_formats):
            return ServingFormatDecision(
                False,
                self.reason,
                self.detail,
                self.id,
            )
        return ServingFormatDecision(True, rule=self.id)


@dataclass(frozen=True)
class ShapeRule:
    id: str
    formats: tuple[str, ...]
    when: NameCondition = field(default_factory=NameCondition)
    min_in_features: int | None = None
    min_out_features: int | None = None
    in_features_multiple_of: int | None = None
    out_features_multiple_of: int | None = None
    reason: str = "kernel_shape"
    detail: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShapeRule":
        return cls(
            id=str(payload["id"]),
            formats=_declared_formats(
                payload.get("formats", ()),
                payload.get("formats_from", ()),
                owner=f"shape rule {payload['id']!r}",
            ),
            when=NameCondition.from_dict(payload.get("when") or {}),
            min_in_features=_optional_int(payload.get("min_in_features")),
            min_out_features=_optional_int(payload.get("min_out_features")),
            in_features_multiple_of=_optional_int(
                payload.get("in_features_multiple_of")
            ),
            out_features_multiple_of=_optional_int(
                payload.get("out_features_multiple_of")
            ),
            reason=str(payload.get("reason", "kernel_shape")),
            detail=str(payload.get("detail", "")),
        )

    def check(
        self,
        fmt: str,
        *,
        qname: str | None = None,
        in_features: int,
        out_features: int,
    ) -> ServingFormatDecision | None:
        if not _format_in(fmt, self.formats):
            return None
        if not self.when.matches(qname or ""):
            return None
        legal = True
        if self.min_in_features is not None and in_features < self.min_in_features:
            legal = False
        if self.min_out_features is not None and out_features < self.min_out_features:
            legal = False
        if (
            self.in_features_multiple_of is not None
            and in_features % self.in_features_multiple_of != 0
        ):
            legal = False
        if (
            self.out_features_multiple_of is not None
            and out_features % self.out_features_multiple_of != 0
        ):
            legal = False
        if legal:
            return ServingFormatDecision(True, rule=self.id)
        detail = self.detail or (
            f"{fmt} kernel does not support "
            f"(out_features={out_features}, in_features={in_features})"
        )
        if self.detail:
            detail = (
                f"{detail} "
                f"(out_features={out_features}, in_features={in_features})"
            )
        return ServingFormatDecision(False, self.reason, detail, self.id)


@dataclass(frozen=True)
class RuntimeShapeValidatorRule:
    id: str
    formats: tuple[str, ...]
    when: NameCondition = field(default_factory=NameCondition)
    callable_path: str | None = None
    optional: bool = True
    reason: str = "kernel_shape"
    detail: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeShapeValidatorRule":
        return cls(
            id=str(payload["id"]),
            formats=tuple(str(v) for v in payload.get("formats", ())),
            when=NameCondition.from_dict(payload.get("when") or {}),
            callable_path=_optional_str(
                payload.get("callable") or payload.get("callable_path")
            ),
            optional=bool(payload.get("optional", True)),
            reason=str(payload.get("reason", "kernel_shape")),
            detail=str(payload.get("detail", "")),
        )

    def check(
        self,
        fmt: str,
        *,
        qname: str | None = None,
        in_features: int,
        out_features: int,
    ) -> ServingFormatDecision | None:
        if not _format_in(fmt, self.formats):
            return None
        if not self.when.matches(qname or ""):
            return None
        verdict = _runtime_shape_validator_accepts(
            self.id,
            fmt,
            in_features=in_features,
            out_features=out_features,
            callable_path=self.callable_path,
        )
        if verdict is None:
            if self.optional:
                return ServingFormatDecision(True, rule=self.id)
            return ServingFormatDecision(
                False,
                self.reason,
                self.detail or f"runtime shape validator {self.id!r} unavailable",
                self.id,
            )
        if verdict:
            return ServingFormatDecision(True, rule=self.id)
        detail = self.detail or (
            f"{fmt} runtime validator {self.id} rejected "
            f"(out_features={out_features}, in_features={in_features})"
        )
        return ServingFormatDecision(False, self.reason, detail, self.id)


@dataclass(frozen=True)
class ExportLaneSpec:
    """The artifact container a serving lane ships through, plus the
    *exporter's own declaration* of what it can emit.

    A serving profile's format menu and its lane's exporter must not be
    able to disagree: a rung the exporter cannot emit is not "denied by
    policy", it is structurally unavailable, and the allocator must never
    be able to spend a bit budget on it (a recent bit-exact re-encode
    short-circuit prices weight-lossless A16 rungs at dloss 0.0 — the
    unbeatable global minimum — so an unexportable-but-legal rung is
    actively attractive to the DP).

    ``codec_formats_from`` is a tuple of ``module:ATTR`` paths whose
    attribute is an *iterable of format names the exporter itself
    declares it can emit* (a dict's keys or a set both count).  Nothing is
    duplicated here: the vLLM lane points at
    ``export_native_compressed.EXPORTABLE_FORMATS`` (that exporter's own
    declaration — its ``FORMAT_SCHEME`` metadata table, CLAUDE.md gate
    #9's "correctly represented in compressed-tensors metadata", already
    unioned with its container passthroughs), and the GGUF lane points at
    ``gguf_formats.GGUF_BLOCK_BYTES`` (the ggml type table
    ``export_gguf``/``export_gguf_direct`` gate on directly).

    ``passthrough_formats`` covers formats a container emits *without* a
    codec entry, for lanes whose declaration is a bare codec table that
    cannot contain them: BF16 is written as plain container floats
    (safetensors bf16 / GGUF F16-F32) and goes on the checkpoint's
    ``ignore`` list rather than into ``config_groups``.  It stays per-lane
    because passthrough is a container fact — FP8_SOURCE is a
    verbatim-copy passthrough on the compressed-tensors lane but has no
    ggml type at all — and it is empty for a lane like
    ``compressed_tensors`` whose exporter folds its own passthroughs into
    the constant it declares.
    """

    id: str
    exporter: str = ""
    codec_formats_from: tuple[str, ...] = ()
    passthrough_formats: tuple[str, ...] = ()
    reason: str = "exporter_cannot_emit"
    detail: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExportLaneSpec":
        return cls(
            id=str(payload["id"]),
            exporter=str(payload.get("exporter", "")),
            codec_formats_from=tuple(
                str(v) for v in payload.get("codec_formats_from", ())
            ),
            passthrough_formats=tuple(
                str(v) for v in payload.get("passthrough_formats", ())
            ),
            reason=str(payload.get("reason", "exporter_cannot_emit")),
            detail=str(payload.get("detail", "")),
        )

    def emittable_formats(self) -> frozenset[str]:
        """Canonical format names this lane's exporter can emit."""
        cached = _EMITTABLE_CACHE.get(self)
        if cached is None:
            names: set[str] = set()
            for path in self.codec_formats_from:
                names |= _declared_exporter_formats(path, self.id)
            if not names:
                raise RuntimeError(
                    f"serving profile export lane {self.id!r} declares no "
                    f"emittable formats (codec_formats_from="
                    f"{list(self.codec_formats_from)!r}). A lane with an "
                    f"empty menu would deny every format; declare the "
                    f"exporter's own format table instead."
                )
            names |= {
                fr.canonical_format_name(name)
                for name in self.passthrough_formats
            }
            cached = frozenset(names)
            _EMITTABLE_CACHE[self] = cached
        return cached

    def check(self, fmt: str) -> ServingFormatDecision:
        emittable = self.emittable_formats()
        if _format_in(fmt, emittable):
            return ServingFormatDecision(True, rule=self.id)
        detail = self.detail or (
            f"{fr.canonical_format_name(fmt)} has no emit path in this "
            f"lane's exporter"
        )
        return ServingFormatDecision(
            False,
            self.reason,
            f"{detail} (lane={self.id}"
            + (f", exporter={self.exporter}" if self.exporter else "")
            + f", emittable={sorted(emittable)})",
            self.id,
        )


@dataclass(frozen=True)
class ResolvedServingLane:
    """One format's concrete serving route, resolved against the pinned
    runtime version.

    ``fused_mid_m_backed`` is the P5b question the allocator could not ask
    before: does the consumer's fused mid-M kernel actually instantiate THIS
    rung, or does the rung fall to expand+GEMM? The retired codebook lane's
    K1.2 was the same defect seen from the runtime end — its published 27B
    artifact shipped an 8-rung K36..K47 ladder of which five rungs had no
    fused mid-M instantiation — so recording the answer per selected unit is
    what stops either repo from pricing an unbacked fast path.
    """

    lane_id: str
    format: str
    activation_contract: str
    fallback_route: str
    fused_mid_m_backed: bool
    fused_mid_m_rungs: tuple[int, ...]
    fused_mid_m_range: tuple[int, int] | None
    runtime_version: str
    rungs_source: str
    rung: int | None = None
    detail: str = ""
    # --- Structured route status (campaign rule R3, principle 9). ----------
    # Principle 9 requires route status in a STRUCTURED field a gate can read,
    # never in prose. These three carry it. Their values are RESOLVED from the
    # pinned serving release's packaged contract, never written into
    # a spec file: a hand-typed verdict is an assertion, and principle 14 takes
    # assertions about another runtime as refusals.
    #
    # ``route_status`` adds two values to principle 9's lane enum, both of
    # which say "this is not a verdict":
    #   ``unattested``     the pinned release publishes no eligibility table,
    #                      so no claim is made. NOT a zero and NOT a pass.
    #   ``unit_dependent`` the table's rules for this lane predicate on facts
    #                      that only exist per unit at export (role split,
    #                      out_features), so the verdict is the export gate's
    #                      the export gate's, not this lane's.
    route_status: str = "unattested"
    requires_serve_flags: tuple[str, ...] = ()
    route_status_source: str = ""

    def as_dict(self) -> dict:
        return {
            "lane_id": self.lane_id,
            "format": self.format,
            "rung": self.rung,
            "activation_contract": self.activation_contract,
            "route_status": self.route_status,
            "requires_serve_flags": list(self.requires_serve_flags),
            "route_status_source": self.route_status_source,
            "fused_mid_m_backed": bool(self.fused_mid_m_backed),
            "fused_mid_m_rungs": list(self.fused_mid_m_rungs),
            "fused_mid_m_range": (
                list(self.fused_mid_m_range)
                if self.fused_mid_m_range is not None else None
            ),
            "fallback_route": self.fallback_route,
            "runtime_version": self.runtime_version,
            "fused_mid_m_rungs_source": self.rungs_source,
            "detail": self.detail,
        }

    def route_key(self) -> str:
        """Stable one-line identity for candidate/provenance comparison."""
        return json.dumps(
            {
                "lane": self.lane_id,
                "act": self.activation_contract,
                "fused_mid_m": bool(self.fused_mid_m_backed),
                "runtime": self.runtime_version,
                "route_status": self.route_status,
            },
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class ServingLaneSpec:
    """Declarative per-format-family serving route (ultraplan P5b).

    The producer once modelled exactly ONE consumer kernel gate
    (``K % 256``) and nothing else — not the ``N % 8`` / ``N % 16`` load
    gates, not the fused mid-M rung set, not the activation contract. A
    format name alone is not an execution identity, so the concrete route is
    declared here as SPEC DATA and attached to every candidate.

    ``fused_mid_m_rungs_by_runtime_version`` is keyed by the pinned serving
    release, because the backed set is a property of the consumer release,
    not of the format. It was the codebook lane that needed it (that lane's
    0.5.0/0.6.0/0.7.0 all instantiated K ∈ {28,32,36,40,44,48} for FP8-CB
    while production permitted every K28..K48), and that lane was retired on
    2026-09-02; no live lane spec declares the key today, so every lane
    resolves the EMPTY backed set. That is the designed fail-closed answer:
    assuming a runtime backs what an older one did is exactly how an unbacked
    fast path gets priced.

    This is metadata only. It carries no latency term and imposes no
    constraint on the DP; the constrained Pareto solver is P5c.
    """

    id: str
    formats: tuple[str, ...] = ()
    activation_contract: str = ""
    fallback_route: str = ""
    fused_mid_m_range: tuple[int, int] | None = None
    fused_mid_m_rungs_by_runtime_version: tuple[
        tuple[str, tuple[int, ...]], ...
    ] = ()
    detail: str = ""
    #: Which structural class of the pinned runtime's eligibility table this
    #: lane's route status is resolved from (campaign rule R3). The spec
    #: declares the MAPPING -- which key to consult -- and never the verdict.
    #: A lane with no ``route_status_source`` resolves ``unattested``, which is
    #: the fail-closed direction: a lane that names no attestation has none.
    route_status_structures: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServingLaneSpec":
        fused = dict(payload.get("fused_mid_m") or {})
        m_range = fused.get("m_range")
        by_version = fused.get("rungs_by_runtime_version") or {}
        source = dict(payload.get("route_status_source") or {})
        return cls(
            route_status_structures=tuple(
                str(v) for v in source.get("structures", ())),
            id=str(payload["id"]),
            formats=_declared_formats(
                payload.get("formats", ()),
                payload.get("formats_from", ()),
                owner=f"serving lane {payload['id']!r}",
            ),
            activation_contract=str(payload.get("activation_contract", "")),
            fallback_route=str(payload.get("fallback_route", "")),
            fused_mid_m_range=(
                (int(m_range[0]), int(m_range[1]))
                if m_range is not None else None
            ),
            fused_mid_m_rungs_by_runtime_version=tuple(
                (str(version), tuple(sorted(int(k) for k in rungs)))
                for version, rungs in sorted(by_version.items())
            ),
            detail=str(payload.get("detail", "")),
        )

    def covers(self, fmt: str) -> bool:
        return _format_in(fmt, self.formats)

    def backed_rungs(self, runtime_version: str) -> tuple[
            tuple[int, ...], str]:
        """``(rungs, source)`` for one runtime version; fail-closed on miss."""
        for version, rungs in self.fused_mid_m_rungs_by_runtime_version:
            if version == runtime_version:
                return rungs, f"serving_profile_spec:{version}"
        if not self.fused_mid_m_rungs_by_runtime_version:
            return (), "lane_declares_no_fused_mid_m_lane"
        return (), "pinned_runtime_version_not_declared"

    def route_status_for(
        self, fmt: str, *, platform: str | None = None,
    ) -> tuple[str, tuple[str, ...], str]:
        """``(route_status, requires_serve_flags, source)`` from the pin (R3).

        Resolved, never declared. The spec names which structural classes of
        the runtime's eligibility table this lane consults; the verdict comes
        from the table the PINNED SERVING release packages.

        Under lane-eligibility v3 a cell is scoped to one platform, one payload
        family and an explicit rung list, so this lane-level answer is narrower
        than it looks: it filters by all three, and returns ``unit_dependent``
        the moment a surviving cell predicates on a fact only the export gate
        holds. It never widens -- an unmatched platform, family or rung is
        ``unattested``, because absence is the only way a v3 table says no.
        """
        from .lane_eligibility import (
            ROUTE_STATUS_UNATTESTED,
            load_eligibility_table,
            resolve_payload_rung,
        )

        table = _cached_eligibility_table(load_eligibility_table)
        version = table.runtime_version
        if not table.present:
            return (
                ROUTE_STATUS_UNATTESTED,
                (),
                f"serving_runtime_contract:{version}:absent",
            )
        if not self.route_status_structures:
            # A lane that names no attestation has none. Fail-closed.
            return (
                ROUTE_STATUS_UNATTESTED,
                (),
                "lane_declares_no_route_status_source",
            )
        if not platform:
            # v3 cells are platform-scoped. A lane resolved without one cannot
            # name a route; it must not fall through to a match-any.
            return (
                ROUTE_STATUS_UNATTESTED,
                (),
                f"serving_runtime_contract:{version}:no_target_platform",
            )
        canonical = fr.canonical_format_name(fmt)
        family, k, rate_q256 = resolve_payload_rung(canonical)
        cells = [
            cell for cell in table.cells
            if cell.structure in self.route_status_structures
            and cell.platform == platform
            and cell.family == family
        ]
        if not cells:
            return (
                ROUTE_STATUS_UNATTESTED,
                (),
                f"serving_runtime_contract:{version}:no_cell",
            )
        rung = rate_q256 if k is None else k
        covering = [
            cell for cell in cells
            if rung is not None
            and rung in (cell.rungs_q256 if cell.is_trellis else cell.rungs)
        ]
        if not covering:
            # The rung this lane would serve is not in any cell's list. A rung
            # the table does not name is unattested, never admitted.
            return (
                ROUTE_STATUS_UNATTESTED,
                (),
                f"serving_runtime_contract:{version}:rung_not_listed",
            )
        # Any cell carrying a predicate needs per-unit facts the lane does not
        # have (role split, out_features). The export gate settles those; this
        # lane says so rather than guessing a lane-wide verdict.
        if any(cell.predicates for cell in covering):
            return (
                "unit_dependent",
                tuple(sorted({
                    flag for cell in covering
                    for flag in cell.requires_serve_flags
                })),
                f"serving_runtime_contract:{version}"
                ":unit_dependent(cb_route_status_gate)",
            )
        best = max(covering, key=lambda cell: _LANE_STATUS_RANK.get(
            cell.route_status, 0))
        return (
            best.route_status,
            best.requires_serve_flags,
            f"serving_runtime_contract:{version}:{best.id}",
        )

    def resolve(self, fmt: str, *, runtime_version: str,
                rung: int | None,
                target_platform: str | None = None) -> ResolvedServingLane:
        rungs, source = self.backed_rungs(runtime_version)
        status, flags, status_source = self.route_status_for(
            fmt, platform=target_platform)
        return ResolvedServingLane(
            lane_id=self.id,
            format=fr.canonical_format_name(fmt),
            activation_contract=self.activation_contract,
            fallback_route=self.fallback_route,
            fused_mid_m_backed=bool(rung is not None and rung in rungs),
            fused_mid_m_rungs=rungs,
            fused_mid_m_range=self.fused_mid_m_range,
            runtime_version=runtime_version,
            rungs_source=source,
            rung=rung,
            detail=self.detail,
            route_status=status,
            requires_serve_flags=flags,
            route_status_source=status_source,
        )


@dataclass(frozen=True)
class RuntimePackageSpec:
    id: str
    module: str | None = None
    version: str | None = None
    pip_packages: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    optional: bool = True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimePackageSpec":
        return cls(
            id=str(payload["id"]),
            module=_optional_str(payload.get("module")),
            version=_optional_str(payload.get("version")),
            pip_packages=tuple(str(v) for v in payload.get("pip_packages", ())),
            env=tuple(
                (str(key), str(value))
                for key, value in (payload.get("env") or {}).items()
            ),
            optional=bool(payload.get("optional", True)),
        )

    def env_dict(self) -> dict[str, str]:
        return dict(self.env)


@dataclass(frozen=True)
class TensorParallelRule:
    """One qname pattern and the way tensor parallelism cuts it.

    ENUMERATED, never inferred.  "Fused implies column-parallel" is wrong for
    several units PrismaQuant already allocates -- MLA's ``q_a_proj`` and
    ``kv_a_proj_with_mqa`` are replicated, a router ``gate`` is replicated, and
    Qwen3.5's fused DeltaNet ``in_proj_qkvz`` is not head-sharded -- so the
    mapping is declared per profile and an unmatched qname is ``none``, which
    makes the gate a no-op rather than a guess.
    """

    id: str
    when: NameCondition
    kind: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorParallelRule":
        kind = str(payload.get("kind", "none"))
        if kind not in ("column", "row", "none"):
            raise ValueError(
                f"tensor_parallel rule {payload.get('id')!r}: kind must be "
                f"one of column|row|none, got {kind!r}")
        return cls(
            id=str(payload.get("id", "")),
            when=NameCondition.from_dict(payload.get("when", {}) or {}),
            kind=kind,
        )


@dataclass(frozen=True)
class TensorParallelSpec:
    """The TP degree an artifact built under this profile will be SERVED at.

    A declaration, and therefore a legality input: at ``world_size`` N a
    column-parallel Linear is N units of ``out_features/N`` rows and a
    row-parallel one is N units of ``in_features/N`` columns, and a format
    whose layout has a shard granularity (a trellis body period, a block scale
    plane) may be legal on the whole tensor and illegal on a shard of it.
    Default 1, which is what every profile in the tree served at before this
    field existed, so an undeclared profile keeps its behaviour exactly.

    This is the PRODUCER's target, not a claim about the runtime.  The runtime
    publishes its own per-format ceiling in the pinned contract's
    ``tensor_parallel`` table (``max_world_size``, ``shard_admission``); the two
    are compared, never conflated (principle 14).
    """

    world_size: int = 1
    rules: tuple[TensorParallelRule, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorParallelSpec":
        size = int(payload.get("world_size", 1))
        if size < 1:
            raise ValueError(f"tensor_parallel.world_size must be >= 1, got {size}")
        return cls(
            world_size=size,
            rules=tuple(
                TensorParallelRule.from_dict(entry)
                for entry in payload.get("rules", ())
            ),
        )

    def kind_for(self, qname: str | None) -> str:
        """How TP cuts this Linear.  First matching rule wins; default none."""
        name = qname or ""
        for rule in self.rules:
            if rule.when.matches(name):
                return rule.kind
        return "none"


@dataclass(frozen=True)
class ServingProfile:
    id: str
    runtime: str = ""
    extends: tuple[str, ...] = ()
    format_rules: tuple[ServingFormatRule, ...] = ()
    shape_rules: tuple[ShapeRule, ...] = ()
    runtime_shape_validators: tuple[RuntimeShapeValidatorRule, ...] = ()
    runtime_packages: tuple[RuntimePackageSpec, ...] = ()
    # Declarative per-format-family serving routes (ultraplan P5b). Empty for
    # profiles that describe no concrete lane (e.g. `research`).
    serving_lanes: tuple[ServingLaneSpec, ...] = ()
    description: str = ""
    # The artifact container this profile ships through. Bounds the format
    # menu by what the lane's exporter declares it can emit (see
    # ExportLaneSpec). Inherited from `extends` when not declared locally.
    export_lane: ExportLaneSpec | None = None
    # Declared exemption from the export-lane bound: this profile
    # constrains *emulation / kernel* legality only and does not
    # correspond to an artifact container, so no exporter bounds its
    # menu. Deliberately true for `research`, which exists so research
    # rungs with no served path (INT4_W4A16_g128, the A16 family)
    # stay measurable. False is the fail-closed default: a new serving
    # profile must name its export lane or declare itself emulation-only.
    emulation_only: bool = False
    # Capability: the serving lane can load DIFFERENT expert schemes for
    # different projection roles of one MoE layer (gate/up vs down). True
    # for GGUF (expert tensors are stacked PER projection, so each stacked
    # tensor carries its own ggml type); false for vLLM compressed-tensors
    # packed MoE, where CompressedTensorsMoEMethod selects ONE scheme per
    # FusedMoE layer. Gates --packed-role-split.
    supports_per_role_expert_schemes: bool = False
    # Exact runtime platform id for hardware-scoped producer profiles.  This
    # is an identity (e.g. ``sm_89``), never a minimum capability or a GPU-name
    # heuristic. Generic profiles leave it unset.
    target_platform: str | None = None
    # Optional producer-side policy id. The exporter remains the inherited
    # container serializer, while this narrower policy supplies additional
    # model/device/manifest gates.
    producer_policy: str | None = None
    # The tensor-parallel degree this profile's artifacts are served at, and
    # the enumerated per-qname split. Consumed as a legality input by
    # ``check_shape``: a shard is a different shape, and a format with a shard
    # granularity can be legal on a tensor and illegal on an Nth of it.
    tensor_parallel: TensorParallelSpec = field(default_factory=TensorParallelSpec)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServingProfile":
        schema = str(payload.get("schema", SCHEMA))
        if schema != SCHEMA:
            raise ValueError(f"unsupported serving-profile schema: {schema!r}")
        return cls(
            id=str(payload["id"]),
            runtime=str(payload.get("runtime", "")),
            extends=tuple(str(v) for v in payload.get("extends", ())),
            format_rules=tuple(
                ServingFormatRule.from_dict(entry)
                for entry in payload.get("format_rules", ())
            ),
            shape_rules=tuple(
                ShapeRule.from_dict(entry)
                for entry in payload.get("shape_rules", ())
            ),
            runtime_shape_validators=tuple(
                RuntimeShapeValidatorRule.from_dict(entry)
                for entry in payload.get("runtime_shape_validators", ())
            ),
            runtime_packages=tuple(
                RuntimePackageSpec.from_dict(entry)
                for entry in payload.get("runtime_packages", ())
            ),
            serving_lanes=tuple(
                ServingLaneSpec.from_dict(entry)
                for entry in payload.get("serving_lanes", ())
            ),
            description=str(payload.get("description", "")),
            export_lane=(
                ExportLaneSpec.from_dict(payload["export_lane"])
                if payload.get("export_lane")
                else None
            ),
            emulation_only=bool(payload.get("emulation_only", False)),
            supports_per_role_expert_schemes=bool(
                payload.get("supports_per_role_expert_schemes", False)
            ),
            target_platform=_optional_str(payload.get("target_platform")),
            producer_policy=_optional_str(payload.get("producer_policy")),
            tensor_parallel=TensorParallelSpec.from_dict(
                payload.get("tensor_parallel", {}) or {}
            ),
        )

    def check_format(self, qname: str | None, fmt: str,
                     packed_expert: bool | None = None
                     ) -> ServingFormatDecision:
        name = qname or ""
        for rule in self.format_rules:
            decision = rule.check(name, fmt, packed_expert=packed_expert)
            if decision is not None and not decision.legal:
                return decision
        # Structural bound, applied after the profile's own policy rules so
        # an explicitly-denied format keeps its policy attribution: the
        # lane's exporter has no emit path for this format, so no allow/deny
        # list may admit it. Fixing the menu disagreement at the root means
        # the profile *cannot* widen past the exporter, rather than a
        # hand-maintained deny list mirroring the exporter's branches.
        if self.export_lane is not None:
            decision = self.export_lane.check(fmt)
            if not decision.legal:
                return decision
        return ServingFormatDecision(True)

    def runtime_package(self, package_id: str) -> RuntimePackageSpec | None:
        for package in reversed(self.runtime_packages):
            if package.id == package_id:
                return package
        return None

    def serving_lane_for(
        self,
        fmt: str,
        *,
        runtime_version: str | None = None,
    ) -> ResolvedServingLane | None:
        """The concrete serving route for one format, or None.

        LAST declaration wins, mirroring how ``extends`` layers a derived
        profile's rules after its bases': a lane redeclared downstream is an
        override, not a second opinion.
        """
        chosen: ServingLaneSpec | None = None
        for lane in self.serving_lanes:
            if lane.covers(fmt):
                chosen = lane
        if chosen is None:
            return None
        return chosen.resolve(
            fmt,
            runtime_version=(
                runtime_version
                if runtime_version is not None
                else serving_runtime_version()
            ),
            rung=_cb_rung_of(fmt),
            target_platform=self.target_platform,
        )

    def check_shape(
        self,
        fmt: str,
        *,
        qname: str | None = None,
        in_features: int,
        out_features: int,
        packed_expert: bool | None = None,
    ) -> ServingFormatDecision:
        """Kernel-shape legality, on the shape the runtime will actually see.

        At ``tensor_parallel.world_size`` N the runtime is handed a SHARD, and
        every rule below is a kernel's, so they are asked about the shard
        rather than about the whole tensor.  The split is the profile's own
        ENUMERATED one (:meth:`TensorParallelSpec.kind_for`); an unmatched
        qname and a packed expert both resolve to ``none``, because expert
        parallelism cuts the stack and leaves each expert's 2-D unit whole.

        A dimension that does not divide the world size is refused rather than
        floored: that is not a format problem, it is a shape no format can be
        served at at this degree, and silently rounding it would price a
        tensor nobody will run.  ``world_size=1`` -- every profile in the tree
        before this field existed -- leaves the shape untouched.
        """
        world = int(self.tensor_parallel.world_size)
        kind = "none" if packed_expert else self.tensor_parallel.kind_for(qname)
        if world > 1 and kind != "none":
            axis = out_features if kind == "column" else in_features
            if axis % world:
                return ServingFormatDecision(
                    False,
                    "tensor_parallel_shard",
                    f"{kind}-parallel {qname or fmt}: {axis} features on the "
                    f"sharded axis is not divisible by the profile's "
                    f"tensor_parallel.world_size={world}",
                )
            if kind == "column":
                out_features = out_features // world
            else:
                in_features = in_features // world
        for rule in self.runtime_shape_validators:
            decision = rule.check(
                fmt,
                qname=qname,
                in_features=in_features,
                out_features=out_features,
            )
            if decision is not None and not decision.legal:
                return decision
        for rule in self.shape_rules:
            decision = rule.check(
                fmt,
                qname=qname,
                in_features=in_features,
                out_features=out_features,
            )
            if decision is not None and not decision.legal:
                return decision
        return ServingFormatDecision(True)


_CACHE: dict[str, ServingProfile] = {}
_EMITTABLE_CACHE: dict["ExportLaneSpec", frozenset[str]] = {}
_RUNTIME_VERSION: str | None = None


def _declared_format_source(path: str, owner: str) -> tuple[str, ...]:
    """Read one canonical format-name iterable at ``module:ATTR``."""

    if ":" in path:
        module_name, attr_name = path.split(":", 1)
    else:
        module_name, attr_name = path.rsplit(".", 1)
    try:
        module = import_module(module_name)
    except ImportError as exc:  # pragma: no cover - environment breakage
        raise RuntimeError(
            f"{owner} declares "
            f"{path!r} but {module_name!r} could not be imported "
            f"({exc!r}); its format set cannot be resolved."
        ) from exc
    try:
        declared = getattr(module, attr_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"{owner} declares "
            f"{path!r} but {module_name!r} has no attribute "
            f"{attr_name!r}; update the source declaration rather than "
            f"hand-listing its formats."
        ) from exc
    try:
        names = [str(name) for name in declared]
    except TypeError as exc:
        raise RuntimeError(
            f"{owner}: {path!r} is not "
            f"iterable ({type(declared).__name__}); expected a container of "
            f"format names (a dict keyed by format name counts)."
        ) from exc
    return tuple(fr.canonical_format_name(name) for name in names)


def _declared_formats(
    literal: Collection[object],
    sources: Collection[object],
    *,
    owner: str,
) -> tuple[str, ...]:
    """Combine explicit policy entries with named canonical format sets."""

    names = [fr.canonical_format_name(str(value)) for value in literal]
    for source in sources:
        names.extend(_declared_format_source(str(source), owner))
    return tuple(dict.fromkeys(names))


def _declared_exporter_formats(path: str, lane_id: str) -> set[str]:
    """Read an exporter's own format declaration at ``module:ATTR``.

    Imported lazily and cached by the caller: the compressed-tensors
    exporter imports this module, so a module-scope import would be
    circular, and the GGUF codec tables pull torch.
    """

    return set(_declared_format_source(
        path, f"serving profile export lane {lane_id!r}"
    ))


def serving_profile_names() -> tuple[str, ...]:
    root = Path(__file__).resolve().parent / "serving_profile_specs"
    names: list[str] = []
    try:
        for resource in root.iterdir():
            if resource.name.endswith(".json"):
                names.append(resource.name[:-5])
    except FileNotFoundError:
        pass
    return tuple(sorted(set(names) | {"research"}))


def load_serving_profile(profile_id: str | None) -> ServingProfile:
    profile_name = str(profile_id or "research")
    if profile_name in _CACHE:
        return _CACHE[profile_name]
    profile = _load_serving_profile_uncached(profile_name)
    if profile.extends:
        bases = tuple(load_serving_profile(base) for base in profile.extends)
        profile = ServingProfile(
            id=profile.id,
            runtime=profile.runtime,
            extends=profile.extends,
            format_rules=tuple(
                rule
                for base in bases
                for rule in base.format_rules
            ) + profile.format_rules,
            shape_rules=tuple(
                rule
                for base in bases
                for rule in base.shape_rules
            ) + profile.shape_rules,
            runtime_shape_validators=tuple(
                rule
                for base in bases
                for rule in base.runtime_shape_validators
            ) + profile.runtime_shape_validators,
            runtime_packages=tuple(
                package
                for base in bases
                for package in base.runtime_packages
            ) + profile.runtime_packages,
            serving_lanes=tuple(
                lane
                for base in bases
                for lane in base.serving_lanes
            ) + profile.serving_lanes,
            description=profile.description,
            export_lane=(
                profile.export_lane
                or next(
                    (base.export_lane for base in bases if base.export_lane),
                    None,
                )
            ),
            # Emulation-only is NOT inherited: a lane that extends the
            # research profile's kernel-shape rules is still a shipping
            # lane and must declare its exporter.
            emulation_only=profile.emulation_only,
            supports_per_role_expert_schemes=(
                profile.supports_per_role_expert_schemes
                or any(
                    base.supports_per_role_expert_schemes for base in bases
                )
            ),
            target_platform=(
                profile.target_platform
                or next(
                    (base.target_platform for base in bases
                     if base.target_platform),
                    None,
                )
            ),
            # A derived profile that declares no tensor_parallel inherits the
            # first base that does. A scalar omitted from this merge silently
            # drops to the default on every derived profile, which for a TP
            # degree would mean an artifact allocated for TP=1 legality and
            # served at TP=8.
            tensor_parallel=(
                profile.tensor_parallel
                if profile.tensor_parallel.world_size != 1
                or profile.tensor_parallel.rules
                else next(
                    (base.tensor_parallel for base in bases
                     if base.tensor_parallel.world_size != 1
                     or base.tensor_parallel.rules),
                    profile.tensor_parallel,
                )
            ),
            producer_policy=(
                profile.producer_policy
                or next(
                    (base.producer_policy for base in bases
                     if base.producer_policy),
                    None,
                )
            ),
        )
    _CACHE[profile_name] = profile
    return profile


def resolve_target_profile(
    profile=None,
    requested: str | None = None,
    *,
    default: str = "research",
) -> str:
    """Resolve the serving/backend constraint profile for a run.

    Explicit CLI/API input wins. Otherwise a model profile may declare its
    default serving profile in the structure spec. The fallback is the
    research profile, which only carries generic kernel-shape rules.
    """
    if requested:
        return str(requested)
    getter = getattr(profile, "serving_profile_id", None)
    if callable(getter):
        try:
            profile_id = getter()
            if profile_id:
                return str(profile_id)
        except Exception:
            pass
    return str(default)


def require_lane_supported(
    profile,
    export_container: str | None,
    *,
    flag: str = "EXPORT_CONTAINER",
):
    """Preflight: refuse an export lane the *architecture* has not declared.

    Lane eligibility is a model-profile property (`supported_export_lanes()`),
    not an operator preference. The GGUF lane needs a llama.cpp-side arch and
    the Tessera lane needs its plugin's reader; where that wiring is missing,
    nothing fails. The run completes, the
    exporter writes bytes, and the server loads uninitialised expert memory —
    the observed failure mode is *coherent-looking garbage generation*, not a
    crash (commit `9a79963`, Laguna, 93% of parameters). One quantization
    cycle on a 100 GB-class model is the cost of finding that out downstream,
    so it is refused up front against the declared set.

    Undeclared architectures support the native compressed-tensors lane only,
    which is what all of them have ever shipped through — so this is strictly
    additive: no run that is legal today becomes illegal.

    Returns the canonical lane id.
    """
    from .model_profiles.structure import (
        DEFAULT_EXPORT_LANE,
        canonical_export_lane,
    )

    requested = str(export_container or DEFAULT_EXPORT_LANE)
    try:
        lane = canonical_export_lane(requested)
    except ValueError as exc:
        raise SystemExit(f"[preflight] ERROR: {flag}: {exc}") from None

    getter = getattr(profile, "supported_export_lanes", None)
    if not callable(getter):
        return lane
    supported = tuple(getter())
    if lane in supported:
        return lane

    name = getattr(profile, "name", None) or type(profile).__name__
    preferred = getattr(profile, "preferred_export_lane", None)
    preferred_lane = preferred() if callable(preferred) else DEFAULT_EXPORT_LANE
    raise SystemExit(
        f"[preflight] ERROR: {flag}={lane!r} is not a declared lane for "
        f"architecture {name!r}. Declared: {list(supported)} "
        f"(preferred: {preferred_lane!r}). An undeclared lane does not fail "
        "loudly at serve time — the missing per-architecture loader means the "
        "runtime serves uninitialised weights and generates coherent-looking "
        "garbage. If this architecture really is wired for this lane, declare "
        f"it in model_profiles/specs/{name}.json (`supported_lanes`) together "
        "with the loader wiring that makes it true."
    )


def require_profile_export_lane(
    profile_id: str | None,
    export_container: str,
) -> str:
    """Require a profile's inherited exporter lane to match the container.

    This is intentionally keyed by ``export_lane.id`` rather than profile id:
    a narrow hardware policy may extend the historical CB profile while using
    the same serializer/container contract.
    """

    from .model_profiles.structure import canonical_export_lane

    requested = canonical_export_lane(export_container)
    profile = load_serving_profile(profile_id)
    if profile.export_lane is None:
        raise ValueError(
            f"serving profile {profile.id!r} declares no artifact export lane"
        )
    declared = canonical_export_lane(profile.export_lane.id)
    if declared != requested:
        raise ValueError(
            f"serving profile {profile.id!r} exports through {declared!r}, not "
            f"requested container {requested!r}"
        )
    return declared


def require_per_role_expert_scheme_support(
    profile_id: str | None,
    *,
    flag: str = "--packed-role-split",
) -> ServingProfile:
    """Hard gate: the resolved serving profile must DECLARE per-role
    expert scheme support before a per-role expert split is legal.

    A gate_up/down role split emits different formats for different
    projections of the SAME MoE layer. That is only loadable when the
    serving lane keys expert schemes per projection — GGUF does (expert
    tensors are stacked per projection, each stacked tensor carries its
    own ggml type). vLLM's compressed-tensors packed-MoE path does not:
    CompressedTensorsMoEMethod selects ONE scheme per FusedMoE layer, so
    a role-split checkpoint (e.g. gate_up=NVFP4 with down=FP8) cannot be
    loaded. Profiles opt in with ``supports_per_role_expert_schemes``.
    """
    resolved = str(profile_id or "research")
    try:
        profile = load_serving_profile(resolved)
    except FileNotFoundError:
        raise SystemExit(
            f"[alloc] ERROR: {flag} was requested, but the target profile "
            f"{resolved!r} is unknown."
        )
    if not profile.supports_per_role_expert_schemes:
        raise SystemExit(
            f"[alloc] ERROR: {flag} was requested, but the resolved "
            f"serving profile {resolved!r} does not declare "
            "supports_per_role_expert_schemes. A per-role split emits "
            "different expert formats for gate_up vs down projections of "
            "the SAME MoE layer; this profile's serving lane loads every "
            "projection of a layer's experts under ONE scheme (vLLM's "
            "CompressedTensorsMoEMethod selects one scheme per FusedMoE "
            "layer), so the checkpoint would be unservable. Use a profile "
            "whose lane keys expert schemes per projection (e.g. "
            "--target-profile gguf), or drop the flag."
        )
    return profile


def check_serving_format(
    profile_id: str | None,
    qname: str | None,
    fmt: str,
    packed_expert: bool | None = None,
) -> ServingFormatDecision:
    try:
        profile = load_serving_profile(profile_id)
    except FileNotFoundError:
        return ServingFormatDecision(
            False,
            "profile_mismatch",
            f"unknown target profile {profile_id!r}",
        )
    return profile.check_format(qname, fmt, packed_expert=packed_expert)


def lane_emittable_formats(profile_id: str | None) -> frozenset[str] | None:
    """Formats the profile's export lane can emit, or None when the
    profile declares no lane (emulation-only, e.g. ``research``)."""
    profile = load_serving_profile(profile_id)
    if profile.export_lane is None:
        return None
    return profile.export_lane.emittable_formats()


def serving_runtime_version() -> str:
    """The pinned serving release the ``fused_mid_m`` lane keys are read at.

    It resolved the Gridbook producer pin until 2026-09-02, when that lane was
    retired (``archive/gridbook_lane_2026-09-02/``) and its pin file went with
    it. There is no producer-side pin behind this key any more, so it answers
    ``""`` — which matches no declared version and therefore backs nothing,
    the same fail-closed answer an unreadable pin always produced.

    This is NOT the Tessera lane's version. Tessera's route is attested from
    the contract its own installed plugin packages and gated by
    ``tessera_serving_runtime_pin``; it never travels through a
    ``fused_mid_m_rungs_by_runtime_version`` table.
    """
    global _RUNTIME_VERSION
    if _RUNTIME_VERSION is None:
        _RUNTIME_VERSION = ""
    return _RUNTIME_VERSION


def _cb_rung_of(fmt: str) -> int | None:
    """The CB k-rung of a format name, or None for a non-CB format."""
    from .cb_layout import parse_format_name

    parsed = parse_format_name(fr.canonical_format_name(fmt))
    return None if parsed is None else int(parsed[1])


def serving_lane_route(
    profile_id: str | None,
    fmt: str,
    *,
    runtime_version: str | None = None,
) -> ResolvedServingLane | None:
    """Resolve one format's serving-lane route under a target profile."""
    try:
        profile = load_serving_profile(profile_id)
    except FileNotFoundError:
        profile = None
    lane = (
        profile.serving_lane_for(fmt, runtime_version=runtime_version)
        if profile is not None else None
    )
    if lane is not None:
        return lane
    # A Tessera rung is a point on a continuous rate axis: ~3000 names per
    # shape, so no profile spec can enumerate them as declared lanes and every
    # one of them lands here. Returning None would report "this profile
    # declares no lane" -- absence of a declaration -- when the truth is that
    # the Tessera admission seam DOES resolve a route and its verdict is
    # ``unattested``. Those are different facts, and principle 9 wants the
    # second on the card. Resolution goes through the same one seam that gates
    # the menu (``tessera_menu.route_admission``), so a unit's stamped lane and
    # the gate that admitted it cannot disagree.
    #
    # Deliberately AFTER the profile lookup, not before: a profile that one
    # day declares a real Tessera lane overrides this, exactly as it would for
    # any other format.
    if isinstance(fmt, str) and fmt.startswith("TESSERA_"):
        from .tessera_menu import tessera_resolved_serving_lane
        return tessera_resolved_serving_lane(
            fmt, runtime_version=(runtime_version or ""))
    return None


def serving_lane_catalog(profile_id: str | None) -> dict:
    """Every declared lane of a profile, resolved, for provenance reports."""
    try:
        profile = load_serving_profile(profile_id)
    except FileNotFoundError:
        return {}
    version = serving_runtime_version()
    lanes: dict[str, dict] = {}
    for lane in profile.serving_lanes:
        rungs, source = lane.backed_rungs(version)
        lanes[lane.id] = {
            "lane_id": lane.id,
            "formats": sorted(lane.formats),
            "activation_contract": lane.activation_contract,
            "fallback_route": lane.fallback_route,
            "fused_mid_m_rungs": list(rungs),
            "fused_mid_m_range": (
                list(lane.fused_mid_m_range)
                if lane.fused_mid_m_range is not None else None
            ),
            "fused_mid_m_rungs_source": source,
            "detail": lane.detail,
        }
    return {
        "schema": SERVING_LANE_SCHEMA,
        "target_profile": str(profile_id or "research"),
        "serving_runtime_version": version,
        "lanes": lanes,
    }


def check_serving_shape(
    profile_id: str | None,
    fmt: str,
    *,
    qname: str | None = None,
    in_features: int,
    out_features: int,
    packed_expert: bool | None = None,
) -> ServingFormatDecision:
    try:
        profile = load_serving_profile(profile_id)
    except FileNotFoundError:
        profile = load_serving_profile("research")
    return profile.check_shape(
        fmt,
        qname=qname,
        in_features=in_features,
        out_features=out_features,
        packed_expert=packed_expert,
    )


def _load_serving_profile_uncached(profile_id: str) -> ServingProfile:
    resource = (
        Path(__file__).resolve().parent
        / "serving_profile_specs"
        / f"{profile_id}.json"
    )
    text = resource.read_text(encoding="utf-8")
    return ServingProfile.from_dict(json.loads(text))


def _format_in(fmt: str, names: Collection[str]) -> bool:
    candidates = {fmt, fr.canonical_format_name(fmt), *fr.aliases_for(fmt)}
    return bool(candidates.intersection(names))


def _runtime_shape_validator_accepts(
    validator_id: str,
    fmt: str,
    *,
    in_features: int,
    out_features: int,
    callable_path: str | None = None,
) -> bool | None:
    path = callable_path or _LEGACY_RUNTIME_VALIDATORS.get(validator_id)
    if not path:
        return None
    validator = _load_runtime_validator(path)
    return validator(fmt, in_features=in_features, out_features=out_features)


_LEGACY_RUNTIME_VALIDATORS = {
    "flashinfer_mxfp8_problem_size": (
        "prismaquant.runtime_shape_validators:"
        "flashinfer_mxfp8_problem_size_accepts"
    ),
}


def _load_runtime_validator(callable_path: str):
    if ":" in callable_path:
        module_name, attr_name = callable_path.split(":", 1)
    else:
        module_name, attr_name = callable_path.rsplit(".", 1)
    module = import_module(module_name)
    validator = getattr(module, attr_name)
    if not callable(validator):
        raise TypeError(f"runtime validator {callable_path!r} is not callable")
    return validator


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
