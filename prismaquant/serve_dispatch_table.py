"""Measured per-(format-family, phase, M-regime) serving costs — canon.

The producer-side mirror of the dispatch table in gridbook
``docs/audits/ultraplan_perf_2026-08-01.md`` §2 ("Why NVFP4-CB trails FP8-CB —
the structural diagnosis"). §6, item **P5c** asks for it by name:

    Feed it the measured per-format x M-regime dispatch table from §2: until
    P1/P2 land, choosing FP8-CB over vanilla NVFP4 at ~4.5 bpw buys quality at
    a measured 1.44x dense-prefill cost, and the allocator should see that
    trade rather than discover it at the release gate.

This module is the *declarative input* only. It holds no policy, computes no
feasibility verdict, and knows nothing about assignments; the aggregation and
the hard-constraint check live in :mod:`prismaquant.serve_constraints`. It is
pure-Python (stdlib only) and never imports torch, so it stays importable in
the CPU allocator driver — the same discipline
:mod:`prismaquant.mtp_rung_selection` follows.

Shape of the data
-----------------

A table is a set of **arenas** and a set of **rows**.

An *arena* is one ``(phase, m_regime)`` cell. It names the **reference route**
every row in that cell is measured against, the metric that reference was
measured in, and — when one is published — the reference's **absolute** value.
Arenas exist because published serving numbers are ratios against different
denominators: the 27B dense-prefill 1.44x is against a native
compressed-tensors artifact, while the fused mid-M 1.04x/1.26x/1.45x are
against FP8-CB's *own* expand+GEMM route. Composing those two into "fused
mid-M vs native" would be arithmetic on incommensurable measurements dressed
up as a measurement, so this schema refuses to let one table cell hold both:
each cell has exactly one denominator, and the workload mix weights cells.

A *row* is one ``(format_family, phase, m_regime, lane)`` cost, expressed as
``relative_unit_cost`` — the arm's phase time divided by that arena's
reference route's phase time. ``1.0`` means "at the reference"; ``1.44`` means
"44% slower than the reference". Every row carries mandatory provenance:
source document, date, GPU identity, the measured quantity, its units, and the
**derivation** that turned the published measurement into this ratio. A row
without a source is a load-time error, not a defaulted field — the whole point
of the table is that the allocator can no longer discover a serving trade at
the release gate, and a fabricated row would replace one blind spot with a
worse one.

Statuses, and what a table may claim
------------------------------------

``docs/lanes/nvfp4-cb/format-speed-policy.md`` §1 is binding: "Per-layer or
per-operator timing tables may generate candidate assignments. They are not
final evidence." Every table therefore declares a ``status``; the only value
this module ships an example for is ``proposal_data``. Nothing here promotes
anything — the served NATIVE-PARITY protocol does.

Arenas whose metric is not a whole-phase metric (``operator_ms``: an isolated
GEMM or operator microbenchmark) are loaded, kept, and marked
``slo_eligible=False``. They are real evidence and belong in the record, but
turning a per-expert GEMM ratio into a served p95-TTFT claim is exactly the
substitution NATIVE-PARITY forbids, so the constraint evaluator refuses to
weight them.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "prismaquant.serve_dispatch_table.v1"

#: The two phases policy §1 constrains separately ("Prefill and decode are
#: separate constraints because a format can move them in opposite
#: directions"). There is deliberately no third "blended" phase.
PHASES = ("prefill", "decode")

#: The concrete serving route a row prices. Mirrors
#: ``serving_profiles.ResolvedServingLane``: a CB rung either rides the
#: consumer's fused mid-M kernel or it takes the fallback expand+GEMM route,
#: and which one is a property of the pinned Gridbook version (P5b), not of
#: the format name.
LANES = ("fused_mid_m", "fallback", "native")

#: Metrics an arena reference may be measured in. The first three are
#: whole-phase metrics an SLO can be stated against; ``operator_ms`` is an
#: isolated-operator microbenchmark and is never SLO-eligible ("Raw standalone
#: kernel timing is never served evidence" — policy §5).
#:
#: The names carry NO percentile. Policy §3 requires released timing to be
#: streaming TTFT/ITL/TPS percentiles, and most of the published record is
#: single-seed point measurements, so the statistic is a separate mandatory
#: arena field (``statistic``) rather than something smuggled into a metric
#: name. The evaluator compares it against the SLO's own statistic and stamps
#: the mismatch instead of silently letting a point measurement stand in for a
#: p95.
METRIC_UNITS = {
    "ttft_ms": "ms",
    "itl_ms": "ms",
    "decode_tok_s": "tok/s",
    "operator_ms": "ms",
}
_SLO_ELIGIBLE_METRICS = {"ttft_ms", "itl_ms", "decode_tok_s"}
_PHASE_METRICS = {
    "prefill": ("ttft_ms",),
    "decode": ("itl_ms", "decode_tok_s"),
}

#: The statistic an arena's absolute reference is. ``p95``/``p05`` are the
#: streaming percentiles policy §3 requires for release evidence; the others
#: are the weaker forms most of the published record actually holds.
STATISTICS = (
    "p95",
    "p05",
    "median_of_repeated_samples",
    "single_seed_point_measurement",
    "ratio_only_no_absolute",
)

_REQUIRED_PROVENANCE = (
    "source", "date", "gpu", "measured_quantity", "units", "derivation",
)


class DispatchTableError(ValueError):
    """A dispatch table failed schema or provenance validation.

    One exception type for both, deliberately: a row with a malformed cost and
    a row with no source are the same failure from the operator's side — the
    table cannot be trusted to price a serving decision — and both must stop
    the load rather than degrade it.
    """


@dataclass(frozen=True)
class RowProvenance:
    """Where one measurement came from. Every field is mandatory.

    ``derivation`` is the field that keeps this honest: published serving
    numbers are quoted as speedups, slowdowns, throughputs, or wall times, and
    ``relative_unit_cost`` is a slowdown ratio. The transform from one to the
    other (invert a speedup, divide two wall times, invert a tok/s pair) is a
    modelling step, so it is recorded in the artifact next to the number it
    produced instead of living in whoever wrote the JSON.
    """

    source: str
    date: str
    gpu: str
    measured_quantity: str
    units: str
    derivation: str
    detail: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *,
                  where: str) -> "RowProvenance":
        if not isinstance(payload, Mapping):
            raise DispatchTableError(
                f"{where}: 'provenance' must be an object naming the source "
                f"of the measurement, got {type(payload).__name__}. A serving "
                "cost with no provenance is a fabricated measurement."
            )
        missing = [
            field for field in _REQUIRED_PROVENANCE
            if not str(payload.get(field, "") or "").strip()
        ]
        if missing:
            raise DispatchTableError(
                f"{where}: provenance is missing required field(s) "
                f"{missing}. Every dispatch row must cite the document or "
                "session it was measured in, the date, the GPU identity, what "
                "was measured, its units, and how the published number became "
                "this relative cost. The allocator prices real serving "
                "decisions from these rows; an uncited row is worse than no "
                "row, because the missing row is at least visible."
            )
        return cls(
            source=str(payload["source"]).strip(),
            date=str(payload["date"]).strip(),
            gpu=str(payload["gpu"]).strip(),
            measured_quantity=str(payload["measured_quantity"]).strip(),
            units=str(payload["units"]).strip(),
            derivation=str(payload["derivation"]).strip(),
            detail=str(payload.get("detail", "") or ""),
        )

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "date": self.date,
            "gpu": self.gpu,
            "measured_quantity": self.measured_quantity,
            "units": self.units,
            "derivation": self.derivation,
            **({"detail": self.detail} if self.detail else {}),
        }


@dataclass(frozen=True)
class ArenaReference:
    """One ``(phase, m_regime)`` cell: its denominator and its scale.

    ``absolute_value`` is what converts a dimensionless assignment-level
    relative cost into the units an operator's SLO is written in. It may be
    ``None`` — the fused mid-M arena's denominator (FP8-CB's own expand+GEMM
    route at M = 32/64/128) has no published whole-model wall time — in which
    case the arena still carries evidence but cannot be weighted by a workload
    mix that has to satisfy an absolute SLO. That refusal is the point: a
    missing denominator is not a licence to invent one.
    """

    phase: str
    m_regime: str
    reference_route: str
    metric: str
    absolute_value: float | None
    statistic: str
    provenance: RowProvenance
    m: int | None = None
    detail: str = ""

    @property
    def units(self) -> str:
        return METRIC_UNITS[self.metric]

    @property
    def slo_eligible(self) -> bool:
        """Whether an absolute SLO may be evaluated in this arena.

        Requires BOTH a whole-phase metric (an isolated-operator ``ms`` is not
        a served latency) and a published absolute for the denominator.
        """
        return (
            self.metric in _SLO_ELIGIBLE_METRICS
            and self.absolute_value is not None
        )

    def reference_ms(self) -> float | None:
        """The denominator as milliseconds-per-phase-unit, or ``None``.

        Prefill: ms per request (TTFT). Decode: ms per output token (ITL).
        ``decode_tok_s`` is converted by the single documented identity
        ``itl_ms = 1000 / tok_s``, which is exact for the single-stream
        batch-1 measurements this metric is published from and is named as
        assumption A7 in :mod:`prismaquant.serve_constraints`.
        """
        if self.absolute_value is None:
            return None
        if self.metric == "decode_tok_s":
            return 1000.0 / float(self.absolute_value)
        return float(self.absolute_value)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArenaReference":
        where = (
            f"arena {payload.get('phase')!r}/{payload.get('m_regime')!r}"
            if isinstance(payload, Mapping) else "arena"
        )
        if not isinstance(payload, Mapping):
            raise DispatchTableError(f"{where}: arena entries must be objects")
        phase = str(payload.get("phase", ""))
        if phase not in PHASES:
            raise DispatchTableError(
                f"{where}: phase must be one of {list(PHASES)}, got "
                f"{phase!r}. Policy §1 constrains prefill and decode "
                "separately; there is no blended phase."
            )
        m_regime = str(payload.get("m_regime", "") or "").strip()
        if not m_regime:
            raise DispatchTableError(f"{where}: 'm_regime' must be non-empty")
        metric = str(payload.get("metric", ""))
        if metric not in METRIC_UNITS:
            raise DispatchTableError(
                f"{where}: metric must be one of "
                f"{sorted(METRIC_UNITS)}, got {metric!r}"
            )
        if metric != "operator_ms" and metric not in _PHASE_METRICS[phase]:
            raise DispatchTableError(
                f"{where}: metric {metric!r} is not a {phase} metric "
                f"(expected one of {list(_PHASE_METRICS[phase])})"
            )
        route = str(payload.get("reference_route", "") or "").strip()
        if not route:
            raise DispatchTableError(
                f"{where}: 'reference_route' must name the execution route "
                "every row in this arena is measured against. A ratio with no "
                "named denominator cannot be compared with any other ratio."
            )
        raw_absolute = payload.get("absolute_value", None)
        absolute: float | None = None
        if raw_absolute is not None:
            absolute = float(raw_absolute)
            if not math.isfinite(absolute) or absolute <= 0.0:
                raise DispatchTableError(
                    f"{where}: absolute_value must be finite and > 0, got "
                    f"{raw_absolute!r}"
                )
        statistic = str(payload.get("statistic", ""))
        if statistic not in STATISTICS:
            raise DispatchTableError(
                f"{where}: statistic must be one of {list(STATISTICS)}, got "
                f"{statistic!r}. Policy §3 requires streaming percentiles for "
                "release evidence; a table that does not say which statistic "
                "its reference is cannot be checked against a p95 SLO."
            )
        if absolute is None and statistic != "ratio_only_no_absolute":
            raise DispatchTableError(
                f"{where}: statistic {statistic!r} claims an absolute "
                "measurement but 'absolute_value' is null"
            )
        m = payload.get("m", None)
        return cls(
            phase=phase,
            m_regime=m_regime,
            reference_route=route,
            metric=metric,
            absolute_value=absolute,
            statistic=statistic,
            provenance=RowProvenance.from_dict(
                payload.get("provenance", {}), where=where),
            m=(int(m) if m is not None else None),
            detail=str(payload.get("detail", "") or ""),
        )

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "m_regime": self.m_regime,
            "m": self.m,
            "reference_route": self.reference_route,
            "metric": self.metric,
            "absolute_value": self.absolute_value,
            "units": self.units,
            "statistic": self.statistic,
            "slo_eligible": self.slo_eligible,
            "provenance": self.provenance.as_dict(),
            **({"detail": self.detail} if self.detail else {}),
        }


@dataclass(frozen=True)
class DispatchRow:
    """One measured serving cost, relative to its arena's reference route."""

    format_family: str
    phase: str
    m_regime: str
    lane: str
    relative_unit_cost: float
    provenance: RowProvenance
    detail: str = ""

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.format_family, self.phase, self.m_regime, self.lane)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DispatchRow":
        where = (
            f"row {payload.get('format_family')!r}/{payload.get('phase')!r}/"
            f"{payload.get('m_regime')!r}/{payload.get('lane')!r}"
            if isinstance(payload, Mapping) else "row"
        )
        if not isinstance(payload, Mapping):
            raise DispatchTableError(f"{where}: row entries must be objects")
        family = str(payload.get("format_family", "") or "").strip()
        if not family:
            raise DispatchTableError(
                f"{where}: 'format_family' must be non-empty")
        phase = str(payload.get("phase", ""))
        if phase not in PHASES:
            raise DispatchTableError(
                f"{where}: phase must be one of {list(PHASES)}, got "
                f"{phase!r}"
            )
        lane = str(payload.get("lane", ""))
        if lane not in LANES:
            raise DispatchTableError(
                f"{where}: lane must be one of {list(LANES)}, got {lane!r}. "
                "The lane is the P5b question — does the pinned consumer's "
                "fused mid-M kernel instantiate THIS rung, or does it take "
                "expand+GEMM — so a row must say which route it timed."
            )
        raw_cost = payload.get("relative_unit_cost", None)
        if raw_cost is None:
            raise DispatchTableError(
                f"{where}: 'relative_unit_cost' is required")
        cost = float(raw_cost)
        if not math.isfinite(cost) or cost <= 0.0:
            raise DispatchTableError(
                f"{where}: relative_unit_cost must be finite and > 0, got "
                f"{raw_cost!r}"
            )
        return cls(
            format_family=family,
            phase=phase,
            m_regime=str(payload.get("m_regime", "") or "").strip(),
            lane=lane,
            relative_unit_cost=cost,
            provenance=RowProvenance.from_dict(
                payload.get("provenance", {}), where=where),
            detail=str(payload.get("detail", "") or ""),
        )

    def as_dict(self) -> dict:
        return {
            "format_family": self.format_family,
            "phase": self.phase,
            "m_regime": self.m_regime,
            "lane": self.lane,
            "relative_unit_cost": self.relative_unit_cost,
            "provenance": self.provenance.as_dict(),
            **({"detail": self.detail} if self.detail else {}),
        }


@dataclass(frozen=True)
class ServeDispatchTable:
    """A validated dispatch table. Immutable, deterministically ordered."""

    table_id: str
    status: str
    description: str
    arenas: tuple[ArenaReference, ...]
    rows: tuple[DispatchRow, ...]
    notes: tuple[str, ...] = ()
    source_path: str = ""

    def arena(self, phase: str, m_regime: str) -> ArenaReference | None:
        for arena in self.arenas:
            if arena.phase == phase and arena.m_regime == m_regime:
                return arena
        return None

    def row(self, format_family: str, phase: str, m_regime: str,
            lane: str) -> DispatchRow | None:
        for row in self.rows:
            if row.key == (format_family, phase, m_regime, lane):
                return row
        return None

    def regimes_for_phase(self, phase: str) -> tuple[str, ...]:
        return tuple(
            arena.m_regime for arena in self.arenas if arena.phase == phase
        )

    def families(self) -> tuple[str, ...]:
        return tuple(sorted({row.format_family for row in self.rows}))

    def identity(self) -> dict:
        """Compact stamp for selection provenance (not the whole table)."""
        return {
            "schema": SCHEMA,
            "table_id": self.table_id,
            "status": self.status,
            "source_path": self.source_path,
            "n_arenas": len(self.arenas),
            "n_rows": len(self.rows),
            "families": list(self.families()),
            "arenas": [
                {
                    "phase": a.phase,
                    "m_regime": a.m_regime,
                    "m": a.m,
                    "reference_route": a.reference_route,
                    "metric": a.metric,
                    "absolute_value": a.absolute_value,
                    "units": a.units,
                    "statistic": a.statistic,
                    "slo_eligible": a.slo_eligible,
                }
                for a in self.arenas
            ],
            "row_sources": sorted({row.provenance.source for row in self.rows}),
        }

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "table_id": self.table_id,
            "status": self.status,
            "description": self.description,
            "notes": list(self.notes),
            "arenas": [a.as_dict() for a in self.arenas],
            "rows": [r.as_dict() for r in self.rows],
        }


def parse_dispatch_table(payload: Mapping[str, Any], *,
                         source_path: str = "") -> ServeDispatchTable:
    """Validate a decoded table payload. Raises :class:`DispatchTableError`."""
    if not isinstance(payload, Mapping):
        raise DispatchTableError(
            f"{source_path or 'dispatch table'}: top level must be an object")
    schema = str(payload.get("schema", ""))
    if schema != SCHEMA:
        raise DispatchTableError(
            f"{source_path or 'dispatch table'}: schema must be {SCHEMA!r}, "
            f"got {schema!r}"
        )
    table_id = str(payload.get("table_id", "") or "").strip()
    if not table_id:
        raise DispatchTableError(
            f"{source_path or 'dispatch table'}: 'table_id' must be non-empty")
    status = str(payload.get("status", "") or "").strip()
    if not status:
        raise DispatchTableError(
            f"{source_path or 'dispatch table'}: 'status' must be non-empty "
            "(e.g. 'proposal_data'). Policy §1: timing tables propose; only "
            "the served protocol promotes, so a table has to say what it is."
        )

    arenas = tuple(
        ArenaReference.from_dict(entry)
        for entry in payload.get("arenas", ())
    )
    seen_arenas: set[tuple[str, str]] = set()
    for arena in arenas:
        key = (arena.phase, arena.m_regime)
        if key in seen_arenas:
            raise DispatchTableError(
                f"{source_path or 'dispatch table'}: duplicate arena {key}. "
                "One cell, one denominator."
            )
        seen_arenas.add(key)

    rows = tuple(
        DispatchRow.from_dict(entry) for entry in payload.get("rows", ())
    )
    seen_rows: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if row.key in seen_rows:
            raise DispatchTableError(
                f"{source_path or 'dispatch table'}: duplicate row {row.key}"
            )
        seen_rows.add(row.key)
        if (row.phase, row.m_regime) not in seen_arenas:
            raise DispatchTableError(
                f"{source_path or 'dispatch table'}: row {row.key} names "
                f"arena ({row.phase!r}, {row.m_regime!r}), which the table "
                "does not declare. A relative cost with no declared "
                "denominator is uninterpretable."
            )

    # Deterministic order: the artifact must not depend on dict iteration or
    # on how the JSON happened to be written.
    arenas = tuple(sorted(arenas, key=lambda a: (a.phase, a.m_regime)))
    rows = tuple(sorted(rows, key=lambda r: r.key))
    return ServeDispatchTable(
        table_id=table_id,
        status=status,
        description=str(payload.get("description", "") or ""),
        arenas=arenas,
        rows=rows,
        notes=tuple(str(n) for n in payload.get("notes", ())),
        source_path=str(source_path or ""),
    )


def load_dispatch_table(path: str | Path) -> ServeDispatchTable:
    """Read and validate a dispatch table from disk."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise DispatchTableError(
            f"cannot read dispatch table {p}: {exc}") from None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DispatchTableError(
            f"{p}: not valid JSON ({exc})") from None
    return parse_dispatch_table(payload, source_path=str(p))


def example_table_path() -> Path | None:
    """The shipped example table, or None when the tree ships none.

    It shipped one until 2026-09-02: a proposal-data table built from the
    retired Gridbook codebook lane's published measurements, which went to
    ``archive/gridbook_lane_2026-09-02/prismaquant/serve_dispatch_tables/``
    with that lane. Returning None rather than a dead path is the honest
    answer, and it keeps the "a row without a source is refused at load" rule
    intact: there is no in-tree table to mistake for a qualified one.
    """
    return None


def dispatch_family_for_format(fmt: str) -> str:
    """Map a format name onto the table's ``format_family`` key.

    CB rungs collapse to their **grid family** (``FP8_CB`` / ``NVFP4_CB``)
    because the serving route is a property of the grid, not the rung: every
    fp8-CB rung takes the same expand+GEMM fallback and the same fused
    prologue when the pinned runtime backs it (the rung-level question is the
    *lane*, which is a separate axis). Everything else keys on its own
    canonical name, because ``format_registry``'s coarse ``family`` puts BF16
    and FP8_E4M3 in one bucket ('fp') and those two have nothing in common at
    serve time.
    """
    from . import format_registry as fr
    from .cb_layout import parse_format_name

    canonical = fr.canonical_format_name(fmt)
    parsed = parse_format_name(canonical)
    if parsed is not None:
        family, _rung = parsed
        return "FP8_CB" if family.grid == "fp8" else "NVFP4_CB"
    return canonical


def missing_family_report(
    families: Iterable[str],
    table: ServeDispatchTable,
    *,
    phase: str,
    m_regime: str,
) -> list[str]:
    """Families with no row in one arena, sorted. Diagnostic helper."""
    have = {
        row.format_family for row in table.rows
        if row.phase == phase and row.m_regime == m_regime
    }
    return sorted({str(f) for f in families} - have)


__all__ = [
    "ArenaReference",
    "DispatchRow",
    "DispatchTableError",
    "LANES",
    "METRIC_UNITS",
    "PHASES",
    "RowProvenance",
    "SCHEMA",
    "STATISTICS",
    "ServeDispatchTable",
    "dispatch_family_for_format",
    "example_table_path",
    "load_dispatch_table",
    "missing_family_report",
    "parse_dispatch_table",
]
