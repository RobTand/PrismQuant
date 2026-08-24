"""Codebook mapping plugin for the platform-agnostic anchored-cost core.

This module deliberately contains no model census, architecture role list, or
campaign byte budget.  Callers supply already source-gated candidate ladders
and profile-declared roles.  The plugin owns only CB facts: canonical format
families, the bundle-authoritative learned/lattice equivalence partition, the
fixed production anchor policy, the scalar render hook, and CB provenance.

AURA ``predicted_dloss`` is the only allocation currency.  Column weights are
carried as production-render identity/input and never enter this cost table.
Smooth AURA remains route-flip-blind on routed experts.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import SOURCE_PASSTHROUGH_CONTRACTS
from prismaquant.anchored_cost import (
    AURA_CURRENCY,
    AnchorScalar,
    AnchoredCostError,
    CandidateSpec,
    PluginDeclaration,
    PricedCell,
    RenderRequest,
    ScalarRenderResult,
    SegmentKey,
    ShapeFit,
    ShapeObservation,
    UnitSpec,
    assert_aura_only_cost_table,
    candidates_by_segment,
    fit_segment_shape,
    lower_convex_hull,
    make_production_render_receipt_from_hashes,
)
from prismaquant.cost_stage_checkpoint import (
    atomic_write_bytes,
    canonical_json,
    canonical_json_sha256,
)


CB_ANCHORED_PLUGIN_SCHEMA = "prismaquant.cb_anchored_cost.plugin.v1"
CB_ANCHORED_COST_SCHEMA = "prismaquant.cb_anchored_aura_cost.v1"
STREAMED_CB_SHARD_MERGE_SCHEMA = (
    "prismaquant.cb_anchored_aura_streamed_shard_merge.v1"
)
CB_ARTIFACT_PUBLISH_SCHEMA = (
    "prismaquant.cb_anchored_artifact_publish.v1"
)
_CB_ARTIFACT_PUBLISH_MANIFEST = ".anchored_publish.json"
_CB_ARTIFACT_OUTPUT_NAMES = (
    "layer_config.json", "selection.json", "pareto.knees.json",
    "cb_col_weights.pkl",
)
GENERIC_SEGMENT_FIELDS = ("family", "role", "equivalence_class")
CB_SEGMENT_FIELDS = ("family", "role", "basis")
LEARNED_BASIS = "learned"
LATTICE_BASIS = "lattice"
PASSTHROUGH_BASIS = "passthrough"
ROUTE_FLIP_LIMITATION = (
    "AURA is a smooth local projection and is route-flip-blind on routed "
    "experts; router decision changes remain unmodelled."
)

_CB_RUNG = re.compile(r"_(?:K|S)(?P<rung>[0-9]+)$")
_DEFAULT_ANCHORS = {
    ("nvfp4_cb", LATTICE_BASIS): "NVFP4_CB_K15",
    ("fp8_cb", LEARNED_BASIS): "FP8_CB_K33",
    ("fp8_cb", LATTICE_BASIS): "FP8_CB_K47",
}


def _canonical_cb_format(format_name: str) -> str:
    canonical = fr.canonical_format_name(str(format_name))
    if fr.get_format(canonical).family not in {"nvfp4_cb", "fp8_cb"}:
        raise AnchoredCostError(f"{format_name!r} is not a CB format")
    return canonical


def cb_rung(format_name: str) -> int:
    canonical = _canonical_cb_format(format_name)
    match = _CB_RUNG.search(canonical)
    if match is None:
        raise AnchoredCostError(f"CB format {canonical!r} has no rung")
    return int(match.group("rung"))


def basis_segment_dict(segment: SegmentKey) -> dict[str, str]:
    """Serialize the CB equivalence class under its load-bearing name."""
    return {
        "family": segment.family,
        "role": segment.role,
        "equivalence_class": segment.equivalence_class,
        "basis": segment.equivalence_class,
    }


def _normalize_source_map(
    source_map: Mapping[str, str],
) -> dict[str, str]:
    if not isinstance(source_map, Mapping) or not source_map:
        raise AnchoredCostError(
            "CB anchoring requires bundle/context codebook_source_by_format; "
            "a global source scalar or K-range inference is not authoritative"
        )
    normalized: dict[str, str] = {}
    for raw_format, raw_source in source_map.items():
        canonical = _canonical_cb_format(str(raw_format))
        source = str(raw_source).strip().lower()
        if source not in {LEARNED_BASIS, LATTICE_BASIS}:
            raise AnchoredCostError(
                f"{canonical}: unsupported codebook source {raw_source!r}"
            )
        previous = normalized.setdefault(canonical, source)
        if previous != source:
            raise AnchoredCostError(
                f"{canonical}: conflicting authoritative source declarations"
            )
    return dict(sorted(normalized.items()))


class CodebookAnchoredFormatPlugin:
    """CB ladder/equivalence declaration consumed by ``anchored_cost``.

    The source map must be taken directly from the loaded value-bearing bundle
    or its validated ``CBSerializationContext``.  Missing candidates refuse;
    neither FP8 K ranges nor a run-global ``codebook_source`` are consulted.
    """

    def __init__(
        self,
        *,
        codebook_source_by_format: Mapping[str, str],
        arm_identity: Mapping[str, object],
        renderer: Callable[
            [RenderRequest], ScalarRenderResult | Mapping[str, object]
        ] | None = None,
        anchor_formats: Mapping[tuple[str, str], str] | None = None,
    ) -> None:
        self.codebook_source_by_format = _normalize_source_map(
            codebook_source_by_format
        )
        self.arm_identity = dict(canonical_json(
            arm_identity, where="CB anchored production arm identity"
        ))
        if not self.arm_identity:
            raise AnchoredCostError("CB anchored production arm is empty")
        raw_anchors = anchor_formats or _DEFAULT_ANCHORS
        self.anchor_formats = {
            (str(family), str(basis)): _canonical_cb_format(fmt)
            for (family, basis), fmt in raw_anchors.items()
        }
        self._renderer = renderer

    def plugin_identity(self) -> PluginDeclaration:
        return PluginDeclaration(
            plugin_id="prismaquant.codebook",
            plugin_version="1",
            equivalence_contract=(
                "bundle-authoritative (family,role,basis); never transfer "
                "across family or learned/lattice basis"
            ),
        )

    def basis_for_format(self, format_name: str) -> str:
        canonical = _canonical_cb_format(format_name)
        if canonical not in self.codebook_source_by_format:
            raise AnchoredCostError(
                "bundle/context codebook_source_by_format lacks exact "
                f"candidate {canonical}; refusing inferred basis"
            )
        return self.codebook_source_by_format[canonical]

    def validate_candidate_coverage(
        self, formats: Sequence[str],
    ) -> None:
        for format_name in formats:
            canonical = _canonical_cb_format(format_name)
            basis = self.basis_for_format(canonical)
            family = fr.get_format(canonical).family
            if family == "nvfp4_cb" and basis != LATTICE_BASIS:
                raise AnchoredCostError(
                    f"{canonical}: NVFP4-CB production policy is lattice, "
                    f"authoritative map says {basis!r}"
                )

    def describe_candidate(
        self, unit: UnitSpec, format_name: str,
    ) -> CandidateSpec:
        canonical = fr.canonical_format_name(str(format_name))
        try:
            candidate = next(
                item for item in unit.candidates
                if item.format_name == canonical
            )
        except StopIteration as exc:
            raise AnchoredCostError(
                f"{unit.qname}: plugin was asked for undeclared {canonical}"
            ) from exc
        if candidate.terminal:
            return candidate
        expected_family = fr.get_format(canonical).family
        expected_basis = self.basis_for_format(canonical)
        if (
            candidate.family != expected_family
            or candidate.equivalence_class != expected_basis
        ):
            raise AnchoredCostError(
                f"{unit.qname}/{canonical}: candidate segment differs from "
                "the registry family or bundle-authoritative basis"
            )
        return candidate

    def select_anchor(
        self,
        unit: UnitSpec,
        segment: SegmentKey,
        candidates: Sequence[CandidateSpec],
    ) -> str:
        del unit
        key = (segment.family, segment.equivalence_class)
        try:
            anchor = self.anchor_formats[key]
        except KeyError as exc:
            raise AnchoredCostError(
                f"no production anchor policy for CB segment {key}"
            ) from exc
        legal = {candidate.format_name for candidate in candidates}
        if anchor not in legal:
            raise AnchoredCostError(
                f"segment {segment.stamp} does not contain required "
                f"production anchor {anchor}; legal={sorted(legal)}"
            )
        return anchor

    def render(
        self, request: RenderRequest,
    ) -> ScalarRenderResult | Mapping[str, object]:
        if self._renderer is None:
            raise AnchoredCostError(
                "CB scalar renderer is not installed; refusing RTN or "
                "render-free fallback"
            )
        return self._renderer(request)

    def provenance_identity_fields(self) -> Mapping[str, object]:
        return {
            "schema": CB_ANCHORED_PLUGIN_SCHEMA,
            "arm_identity": self.arm_identity,
            "segment_key_fields": list(GENERIC_SEGMENT_FIELDS),
            "cb_segment_alias_fields": list(CB_SEGMENT_FIELDS),
            "equivalence_vocabulary_name": "codebook_basis",
            "codebook_source_by_format": dict(
                self.codebook_source_by_format
            ),
            "source_map_sha256": canonical_json_sha256(
                self.codebook_source_by_format,
                where="CB authoritative source map",
            ),
            "anchor_formats_by_family_basis": {
                f"{family}|{basis}": fmt
                for (family, basis), fmt in sorted(
                    self.anchor_formats.items()
                )
            },
            "aura_is_only_cost_currency": True,
            "cb_col_weights_role": "production_render_input_only",
            "route_flip_limitation": ROUTE_FLIP_LIMITATION,
        }


@dataclass(frozen=True)
class CBUnitDeclaration:
    """Already source-gated unit data supplied by a model campaign."""

    qname: str
    role: str
    unit_class: str
    n_params: int
    payload_bytes_by_format: Mapping[str, int]
    terminal_format: str
    serving_group: str | None = None


# The registered CB shape design, widest first.  Which of these columns is
# actually *identifiable* is a property of the declared ladder, not of the
# family -- see _identifiable_cb_shape_columns.
_CB_SHAPE_COLUMNS: tuple[tuple[str, Callable[[int], float]], ...] = (
    ("rung", lambda rung: float(rung)),
    ("rung_parity", lambda rung: float(rung % 2)),
)


def _identifiable_cb_shape_columns(rungs: Sequence[int]) -> tuple[int, ...]:
    """Keep the rung coordinate plus the derived columns this ladder resolves.

    A derived feature that is constant across a segment's legal rungs carries
    no information: per-unit centering turns it into a zero column, the design
    drops to rank ``k-1`` of ``k``, and ``_fit_currency`` fails closed forever.
    The ladder decides which survive, not the family.  Two live ladders make
    that concrete -- the gridbook K1.2 fused mid-M kernel law admits only
    ``k % 4 == 0`` FP8-CB rungs, so parity is constant on every legal FP8-CB
    ladder, while NVFP4-CB (K12..K18) still spans both parities and keeps the
    full rung-plus-parity design.  The rung coordinate itself is always
    retained: ``CandidateSpec`` requires a nonempty basis, and the only ladder
    on which the rung is constant is a one-rung ladder, which is priced by its
    anchor and never fitted at all.
    """
    distinct = {int(rung) for rung in rungs}
    if not distinct:
        raise AnchoredCostError("cannot derive a shape basis from no rungs")
    return (0, *(
        index
        for index, (_name, projection) in enumerate(_CB_SHAPE_COLUMNS)
        if index > 0 and len({projection(rung) for rung in distinct}) > 1
    ))


def _cb_declaration_ladder(
    declaration: CBUnitDeclaration,
) -> tuple[str, dict[str, int], tuple[str, ...]]:
    """Canonicalize one declaration into (terminal, payloads, CB ladder)."""
    terminal = fr.canonical_format_name(declaration.terminal_format)
    payloads = {
        fr.canonical_format_name(str(name)): int(value)
        for name, value in declaration.payload_bytes_by_format.items()
    }
    if terminal not in payloads:
        raise AnchoredCostError(
            f"{declaration.qname}: terminal payload is absent"
        )
    cb_formats = sorted(
        (name for name in payloads if name != terminal),
        key=lambda name: (
            fr.get_format(name).family, cb_rung(name), name,
        ),
    )
    if not cb_formats:
        raise AnchoredCostError(
            f"{declaration.qname}: source-gated CB ladder is empty"
        )
    return terminal, payloads, tuple(cb_formats)


def build_cb_units(
    declarations: Sequence[CBUnitDeclaration],
    plugin: CodebookAnchoredFormatPlugin,
) -> tuple[UnitSpec, ...]:
    """Convert exact caller-owned ladders without redoing source legality."""
    # Pass 1: the shape basis is a *segment* property, so it must be derived
    # from the union of legal rungs across every unit in that segment.  Doing
    # it per unit would hand two units in one segment different widths the
    # moment source gating trims one ladder, and _fit_currency rejects a
    # segment whose candidates use mixed shape bases.
    ladder_rungs: dict[SegmentKey, set[int]] = defaultdict(set)
    for declaration in declarations:
        _terminal, _payloads, cb_formats = _cb_declaration_ladder(declaration)
        for format_name in cb_formats:
            ladder_rungs[SegmentKey(
                fr.get_format(format_name).family,
                declaration.role,
                plugin.basis_for_format(format_name),
            )].add(cb_rung(format_name))
    columns_by_segment = {
        segment: _identifiable_cb_shape_columns(sorted(rungs))
        for segment, rungs in ladder_rungs.items()
    }

    units: list[UnitSpec] = []
    seen: set[str] = set()
    for declaration in sorted(declarations, key=lambda item: item.qname):
        if declaration.qname in seen:
            raise AnchoredCostError(
                f"duplicate CB unit declaration {declaration.qname!r}"
            )
        seen.add(declaration.qname)
        terminal, payloads, cb_formats = _cb_declaration_ladder(declaration)
        plugin.validate_candidate_coverage(cb_formats)
        candidates: list[CandidateSpec] = []
        for format_name in cb_formats:
            payload = payloads[format_name]
            rung = cb_rung(format_name)
            family = fr.get_format(format_name).family
            basis = plugin.basis_for_format(format_name)
            shape_features = tuple(
                _CB_SHAPE_COLUMNS[index][1](rung)
                for index in columns_by_segment[
                    SegmentKey(family, declaration.role, basis)
                ]
            )
            candidates.append(CandidateSpec(
                format_name=format_name,
                bits=8.0 * payload / int(declaration.n_params),
                payload_bytes=payload,
                family=family,
                equivalence_class=basis,
                shape_features=shape_features,
                coordinate=float(rung),
            ))
        terminal_payload = payloads[terminal]
        try:
            terminal_contract = SOURCE_PASSTHROUGH_CONTRACTS[terminal]
        except KeyError as exc:
            raise AnchoredCostError(
                f"{declaration.qname}: terminal {terminal!r} has no exact "
                "source-passthrough contract"
            ) from exc
        terminal_spec = fr.get_format(terminal)
        # ``zero_cost_by_construction`` controls whether the generic cost
        # loader synthesizes a missing column; it is not the terminal's
        # end-to-end eligibility bit.  BF16 and FP8_SOURCE deliberately carry
        # measured columns in ordinary campaigns, yet an exact-source terminal
        # remains honest when its activation path is the identity.  The A-side
        # contract is the decisive gate here: it preserves the block-FP8 W8A16
        # source terminal, while activation-changing terminals remain excluded
        # until activation-side AURA exists.
        terminal_selectable = not terminal_spec.act_quant_changes_input
        candidates.append(CandidateSpec(
            format_name=terminal,
            bits=8.0 * terminal_payload / int(declaration.n_params),
            payload_bytes=terminal_payload,
            family="source_terminal",
            equivalence_class=PASSTHROUGH_BASIS,
            shape_features=(),
            coordinate=0.0,
            terminal=True,
            allocator_selectable=terminal_selectable,
        ))
        units.append(UnitSpec(
            qname=declaration.qname,
            role=declaration.role,
            unit_class=declaration.unit_class,
            candidates=tuple(candidates),
            n_params=declaration.n_params,
            serving_group=declaration.serving_group,
        ))
    return tuple(units)


@dataclass(frozen=True)
class CBPanelPolicy:
    # The role is load-bearing: a routed-expert FP8 learned ladder can end at
    # a different source-rate ceiling than a dense FP8 learned ladder.  A
    # family/basis-only policy could therefore request illegal fitting rungs
    # even though its transfer equivalence declaration includes the role.
    panel_rungs_by_segment: Mapping[
        tuple[str, str, str], tuple[str, ...]
    ]
    validation_rungs_by_segment: Mapping[
        tuple[str, str, str], tuple[str, ...]
    ]
    panel_units_per_role: int = 32
    validation_units_per_role: int = 4
    seed: int = 42


def _feature_rank(rows: Sequence[Sequence[float]]) -> int:
    matrix = [list(map(float, row)) for row in rows]
    if not matrix:
        return 0
    width = len(matrix[0])
    rank = 0
    for column in range(width):
        pivot = max(
            range(rank, len(matrix)),
            key=lambda index: abs(matrix[index][column]),
            default=rank,
        )
        if pivot >= len(matrix) or abs(matrix[pivot][column]) <= 1e-12:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        divisor = matrix[rank][column]
        matrix[rank] = [value / divisor for value in matrix[rank]]
        for index in range(len(matrix)):
            if index == rank:
                continue
            factor = matrix[index][column]
            matrix[index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(matrix[index], matrix[rank])
            ]
        rank += 1
    return rank


def plan_cb_panel_and_validation(
    units: Sequence[UnitSpec],
    plugin: CodebookAnchoredFormatPlugin,
    policy: CBPanelPolicy,
) -> tuple[tuple[RenderRequest, ...], tuple[RenderRequest, ...], dict[str, object]]:
    """Choose deterministic role cohorts, globally disjoint from validation."""
    by_role_segments: dict[str, set[SegmentKey]] = defaultdict(set)
    by_segment: dict[SegmentKey, list[UnitSpec]] = defaultdict(list)
    ladder_formats: dict[SegmentKey, set[str]] = defaultdict(set)
    for unit in units:
        for segment, candidates in candidates_by_segment(unit, plugin).items():
            by_role_segments[unit.role].add(segment)
            by_segment[segment].append(unit)
            ladder_formats[segment].update(
                candidate.format_name for candidate in candidates
            )

    cohorts: dict[str, tuple[tuple[UnitSpec, ...], tuple[UnitSpec, ...]]] = {}
    for role, segments in sorted(by_role_segments.items()):
        required = {
            fr.canonical_format_name(fmt)
            for segment in segments
            for fmt in (
                *policy.panel_rungs_by_segment.get(
                    (
                        segment.family,
                        segment.role,
                        segment.equivalence_class,
                    ),
                    (),
                ),
                *policy.validation_rungs_by_segment.get(
                    (
                        segment.family,
                        segment.role,
                        segment.equivalence_class,
                    ),
                    (),
                ),
            )
        }
        eligible = [
            unit for unit in units
            if unit.role == role and required.issubset({
                candidate.format_name for candidate in unit.candidates
            })
        ]
        eligible.sort(key=lambda unit: (
            hashlib.sha256(
                f"{policy.seed}|{role}|{unit.qname}".encode()
            ).hexdigest(),
            unit.qname,
        ))
        needed = policy.panel_units_per_role + policy.validation_units_per_role
        if len(eligible) < needed:
            raise AnchoredCostError(
                f"role {role!r} has {len(eligible)} all-segment units; "
                f"needs {needed}"
            )
        cohorts[role] = (
            tuple(eligible[: policy.panel_units_per_role]),
            tuple(eligible[
                policy.panel_units_per_role:needed
            ]),
        )

    panel: list[RenderRequest] = []
    validation: list[RenderRequest] = []
    accounting: dict[str, object] = {}
    for segment in sorted(by_segment):
        key = (
            segment.family, segment.role, segment.equivalence_class,
        )
        panel_formats = tuple(
            fr.canonical_format_name(fmt)
            for fmt in policy.panel_rungs_by_segment.get(key, ())
        )
        anchor = plugin.anchor_formats.get(
            (segment.family, segment.equivalence_class)
        )
        if len(ladder_formats[segment]) == 1:
            # A one-rung segment has no shape law to fit and no room to fit
            # one: the anchor is forced onto that rung, so pricing reduces to
            # ratio 1.0 and the cell carries its own production render.  This
            # is strictly more faithful than extrapolation, so renders spent
            # on a panel here would buy nothing -- refuse a policy that asks
            # for them rather than silently ignoring it.
            if panel_formats or policy.validation_rungs_by_segment.get(key, ()):
                raise AnchoredCostError(
                    f"{segment.stamp} declares one legal rung and is priced "
                    "by its anchor; it admits no panel or validation rungs"
                )
            accounting[segment.stamp] = {
                "segment": basis_segment_dict(segment),
                "panel_units": 0,
                "panel_rungs": [],
                "panel_render_cells": 0,
                "validation_units": 0,
                "validation_rungs": [],
                "validation_render_cells": 0,
                "validation_rungs_two_axis": [],
                "validation_rungs_unit_axis_only": [],
                "anchor_rung": anchor,
                "design_rank": 0,
                "design_rank_required": 0,
                "single_rung_measured": True,
                "sole_rung": sorted(ladder_formats[segment])[0],
            }
            continue
        if len(panel_formats) < 2:
            raise AnchoredCostError(
                f"panel policy lacks >=2 rungs for {segment.stamp}"
            )
        panel_units, heldout_units = cohorts[segment.role]
        declared = {
            candidate.format_name: candidate
            for candidate in panel_units[0].candidates
            if not candidate.terminal
        }
        if not set(panel_formats).issubset(declared):
            raise AnchoredCostError(
                f"panel policy crosses or leaves {segment.stamp}"
            )
        features = [declared[fmt].shape_features for fmt in panel_formats]
        means = [
            math.fsum(row[index] for row in features) / len(features)
            for index in range(len(features[0]))
        ]
        centered = [
            [value - means[index] for index, value in enumerate(row)]
            for row in features
        ]
        rank = _feature_rank(centered)
        required_rank = len(features[0])
        if rank != required_rank:
            raise AnchoredCostError(
                f"panel {segment.stamp} design rank is {rank} of "
                f"{required_rank}"
            )
        for unit in panel_units:
            panel.extend(
                RenderRequest(unit.qname, segment, fmt, "panel")
                for fmt in panel_formats
            )
        validation_formats = tuple(
            fr.canonical_format_name(fmt)
            for fmt in policy.validation_rungs_by_segment.get(key, ())
        )
        # Two hold-out axes, and only one of them is ever vacuous.  At the
        # anchor rung the prediction is anchor x ratio(anchor, anchor) =
        # anchor x 1.0, i.e. the measurement itself: the cell reports dex
        # 0.0000 by construction and dilutes the report with tautological
        # passes.  That is refused here for every driver -- it is the defect
        # that silently zeroed 48 of the first Qwen3.8-27B campaign's 192
        # validation cells.  A validation rung that the PANEL also contains is
        # a different thing and is NOT refused: the shape at that rung was
        # fitted from panel units, but this cell applies it to a held-out
        # unit's own anchor, so it genuinely tests cross-unit transfer.  On a
        # two-rung ladder whose second rung is the anchor (DSv4 routed
        # experts: K28, K32=anchor) it is also the ONLY validation that can
        # structurally exist, so refusing it would ban validating routed
        # experts at all.  It is classified and reported instead -- see
        # ``held_out_axes`` in ``heldout_validation_report``.
        if anchor is not None and anchor in validation_formats:
            raise AnchoredCostError(
                f"{segment.stamp}: {anchor} is this segment's anchor and "
                "cannot also be a validation rung -- the fit reproduces the "
                "anchor by construction, so every such cell reports dex "
                "0.0000 and the held-out report reads better than the fit "
                "is. Choose a rung the anchor is not."
            )
        if validation_formats and len(validation_formats) < 2:
            raise AnchoredCostError(
                f"validation policy needs >=2 rungs for {segment.stamp}"
            )
        for unit in heldout_units if validation_formats else ():
            validation.extend(
                RenderRequest(unit.qname, segment, fmt, "validation")
                for fmt in validation_formats
            )
        accounting[segment.stamp] = {
            "segment": basis_segment_dict(segment),
            "panel_units": len(panel_units),
            "panel_rungs": list(panel_formats),
            "panel_render_cells": len(panel_units) * len(panel_formats),
            "validation_units": (
                len(heldout_units) if validation_formats else 0
            ),
            "validation_rungs": list(validation_formats),
            "validation_render_cells": (
                len(heldout_units) * len(validation_formats)
            ),
            # Which axes each validation rung actually holds out.  A rung the
            # panel does not contain tests unit AND rung generalization; a
            # panel rung tests unit generalization only.  A segment whose
            # every validation rung is panel-trained has NO evidence that the
            # fitted shape law extends off the panel, and the report must say
            # so rather than let the reader assume otherwise.
            "validation_rungs_two_axis": [
                fmt for fmt in validation_formats
                if fmt not in set(panel_formats)
            ],
            "validation_rungs_unit_axis_only": [
                fmt for fmt in validation_formats
                if fmt in set(panel_formats)
            ],
            "anchor_rung": anchor,
            "design_rank": rank,
            "design_rank_required": required_rank,
        }

    panel_names = {request.qname for request in panel}
    validation_names = {request.qname for request in validation}
    overlap = sorted(panel_names & validation_names)
    if overlap:
        raise AnchoredCostError(
            f"CB panel and held-out validation overlap: {overlap[:8]}"
        )
    return tuple(panel), tuple(validation), {
        "segment_key_fields": list(GENERIC_SEGMENT_FIELDS),
        "cb_segment_alias_fields": list(CB_SEGMENT_FIELDS),
        "equivalence_vocabulary_name": "codebook_basis",
        "segment_count": len(accounting),
        "panel_render_cells": len(panel),
        "validation_render_cells": len(validation),
        "segments": accounting,
    }


def build_streamed_cb_render_plan(
    units: Sequence[UnitSpec],
    plugin: CodebookAnchoredFormatPlugin,
    anchors: Sequence[RenderRequest],
    panel: Sequence[RenderRequest],
    validation: Sequence[RenderRequest],
) -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, dict[str, list[str]]],
    dict[str, object],
]:
    """Union scalar cells; terminals are included but never synthesized."""
    by_name = {unit.qname: unit for unit in units}
    expected = {
        (unit.qname, segment)
        for unit in units
        for segment in candidates_by_segment(unit, plugin)
    }
    observed = [(request.qname, request.segment) for request in anchors]
    if set(observed) != expected or len(observed) != len(set(observed)):
        raise AnchoredCostError(
            "streamed plan is not exactly one anchor per unit/family/role/"
            "equivalence class"
        )
    formats: dict[str, list[str]] = {}
    purposes: dict[str, dict[str, list[str]]] = {}
    for unit in units:
        terminal = next(
            candidate.format_name for candidate in unit.candidates
            if candidate.terminal
        )
        formats[unit.qname] = [terminal]
        purposes[unit.qname] = {}
    for request in (*anchors, *panel, *validation):
        unit = by_name.get(request.qname)
        if unit is None:
            raise AnchoredCostError(
                f"render request has unknown unit {request.qname!r}"
            )
        candidate = next(
            (item for item in unit.candidates
             if item.format_name == request.format_name),
            None,
        )
        if candidate is None or candidate.terminal:
            raise AnchoredCostError(
                f"render request is not a legal CB candidate: "
                f"{request.qname}@{request.format_name}"
            )
        if unit.segment_for(candidate) != request.segment:
            raise AnchoredCostError(
                "render request crosses learned/lattice or family seam"
            )
        if request.format_name not in formats[request.qname]:
            formats[request.qname].append(request.format_name)
        bucket = purposes[request.qname].setdefault(
            request.format_name, []
        )
        if request.purpose not in bucket:
            bucket.append(request.purpose)
    logical = {
        "anchor": len(anchors),
        "panel": len(panel),
        "validation": len(validation),
    }
    physical = sum(len(row) for row in purposes.values())
    return (
        {name: tuple(values) for name, values in sorted(formats.items())},
        {
            name: dict(sorted(row.items()))
            for name, row in sorted(purposes.items())
        },
        {
            "logical_purpose_cells": logical,
            "logical_total": sum(logical.values()),
            "physical_union_render_cells": physical,
            "deduplicated_multi_purpose_cells": (
                sum(logical.values()) - physical
            ),
            "no_full_menu_materialization": True,
            "rendered_weight_persisted": False,
        },
    )


@dataclass(frozen=True)
class _StreamedReceiptBinding:
    costs: Mapping[str, object]
    formats_by_qname: Mapping[str, object]
    purposes_by_qname: Mapping[str, object]
    unmeasured_formats_by_qname: Mapping[str, object]
    arm_identity_sha256: str
    payload_identity_sha256: str


def _streamed_receipt_binding(
    payload: Mapping[str, object],
) -> _StreamedReceiptBinding:
    """Validate and hash global streamed provenance exactly once."""
    costs = payload.get("costs")
    if not isinstance(costs, Mapping):
        raise AnchoredCostError("streamed AURA payload has no costs")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise AnchoredCostError("streamed AURA payload has no provenance")
    renderer = provenance.get("production_anchor_renderer")
    purposes = provenance.get("production_anchor_render_purposes")
    unmeasured = provenance.get(
        "production_anchor_unmeasured_formats_by_qname"
    )
    cb_identity = provenance.get("cb_render_identity")
    if not all(isinstance(value, Mapping) for value in (
        renderer, purposes, unmeasured, cb_identity,
    )):
        raise AnchoredCostError(
            "streamed AURA lacks identity-bound production renderer/plan"
        )
    arm_identity = renderer.get("arm_identity")
    renderer_formats = renderer.get("formats_by_qname")
    if not isinstance(arm_identity, Mapping) or not isinstance(
        renderer_formats, Mapping
    ):
        raise AnchoredCostError(
            "streamed production renderer lacks arm or sparse format identity"
        )
    return _StreamedReceiptBinding(
        costs=costs,
        formats_by_qname=renderer_formats,
        purposes_by_qname=purposes,
        unmeasured_formats_by_qname=unmeasured,
        arm_identity_sha256=canonical_json_sha256(
            arm_identity, where="streamed production render arm identity",
        ),
        payload_identity_sha256=canonical_json_sha256(
            {
                "production_anchor_renderer": renderer,
                "production_anchor_render_purposes": purposes,
                "production_anchor_unmeasured_formats_by_qname": unmeasured,
                "cb_render_identity": cb_identity,
            },
            where="streamed production render payload identity",
        ),
    )


def _receipt_from_streamed_payload(
    request: RenderRequest,
    binding: _StreamedReceiptBinding,
):
    """Bind one streamed scalar to a prehashed renderer/plan identity."""
    try:
        planned_formats = tuple(binding.formats_by_qname[request.qname])
        planned_purposes = binding.purposes_by_qname[
            request.qname
        ][request.format_name]
        row = binding.costs[request.qname][request.format_name]
    except (KeyError, TypeError) as exc:
        raise AnchoredCostError(
            f"streamed AURA omitted identity for "
            f"{request.qname}@{request.format_name}"
        ) from exc
    if request.format_name not in planned_formats or (
        request.purpose not in tuple(planned_purposes)
    ):
        raise AnchoredCostError(
            f"{request.request_id}: streamed purpose/format plan differs"
        )
    if not isinstance(row, Mapping) or (
        row.get("dw_source") != "production_render"
        or row.get("production_anchor_measured") is not True
    ):
        raise AnchoredCostError(
            f"{request.qname}@{request.format_name} is not a real "
            "production-arm render"
        )
    scalar = ScalarRenderResult(
        float(row["predicted_dloss"]),
        (
            float(row["weight_mse_diagnostic"])
            if row.get("weight_mse_diagnostic") is not None else None
        ),
    )
    return make_production_render_receipt_from_hashes(
        request,
        scalar,
        arm_identity_sha256=binding.arm_identity_sha256,
        payload_identity_sha256=binding.payload_identity_sha256,
    )


def observations_from_streamed_payload(
    requests: Sequence[RenderRequest],
    payload: Mapping[str, object],
) -> tuple[ShapeObservation, ...]:
    observations: list[ShapeObservation] = []
    binding = _streamed_receipt_binding(payload)
    for request in requests:
        if request.purpose not in {"panel", "validation"}:
            raise AnchoredCostError(
                "shape observations require panel/validation requests"
            )
        receipt = _receipt_from_streamed_payload(request, binding)
        observations.append(ShapeObservation(
            request.qname,
            request.segment,
            request.format_name,
            receipt.scalar.predicted_dloss,
            receipt.scalar.weight_mse_diagnostic,
            receipt,
        ))
    return tuple(observations)


def anchors_from_streamed_payload(
    requests: Sequence[RenderRequest],
    payload: Mapping[str, object],
) -> dict[tuple[str, SegmentKey], AnchorScalar]:
    anchors: dict[tuple[str, SegmentKey], AnchorScalar] = {}
    binding = _streamed_receipt_binding(payload)
    for request in requests:
        if request.purpose != "anchor":
            raise AnchoredCostError("anchor adapter received a non-anchor")
        receipt = _receipt_from_streamed_payload(request, binding)
        key = (request.qname, request.segment)
        if key in anchors:
            raise AnchoredCostError(f"duplicate streamed anchor {key}")
        anchors[key] = AnchorScalar(
            request.qname,
            request.segment,
            request.format_name,
            receipt.scalar.predicted_dloss,
            receipt,
        )
    return anchors


def _single_rung_shape_fit(
    segment: SegmentKey,
    sole: CandidateSpec,
    anchors: Mapping[tuple[str, SegmentKey], AnchorScalar],
) -> ShapeFit:
    """Price a one-rung segment from its own anchors instead of a shape law.

    ``price_anchored_candidates`` evaluates ``anchor.predicted_dloss *
    fit.ratio(candidate, anchor)``.  When a segment declares exactly one legal
    rung the anchor is forced onto it, so the ratio is identically 1.0 and the
    reported cost is that unit's own production render -- measured, not
    extrapolated.  The identity fit below states that explicitly rather than
    laundering it through a regression: a one-coordinate design centers to the
    zero matrix, so fitting is impossible, not merely redundant.  Provenance
    comes from the anchors themselves, which is exactly the identity pricing
    re-checks per unit.
    """
    scoped = sorted(
        (anchor for (_qname, key), anchor in anchors.items() if key == segment),
        key=lambda anchor: anchor.qname,
    )
    if not scoped:
        raise AnchoredCostError(
            f"{segment.stamp} declares one legal rung but has no anchor"
        )
    off_ladder = sorted({
        anchor.format_name for anchor in scoped
        if anchor.format_name != sole.format_name
    })
    if off_ladder:
        raise AnchoredCostError(
            f"{segment.stamp} anchors at {off_ladder} outside its sole legal "
            f"rung {sole.format_name}"
        )
    receipts = [anchor.receipt for anchor in scoped]
    if len({receipt.arm_identity_sha256 for receipt in receipts}) != 1:
        raise AnchoredCostError("single-rung segment spans production arms")
    if len({receipt.payload_identity_sha256 for receipt in receipts}) != 1:
        raise AnchoredCostError(
            "single-rung segment spans render payload identities"
        )
    return ShapeFit(
        segment=segment,
        g_by_format={sole.format_name: 1.0},
        reference_format=sole.format_name,
        coefficients=(),
        design_rank=0,
        design_rank_required=0,
        n_units=len({anchor.qname for anchor in scoped}),
        n_observations=len(scoped),
        arm_identity_sha256=receipts[0].arm_identity_sha256,
        payload_identity_sha256=receipts[0].payload_identity_sha256,
        panel_receipts_sha256=canonical_json_sha256(
            sorted(receipt.receipt_sha256 for receipt in receipts),
            where="single-rung measured segment anchor receipts",
        ),
    )


def fit_all_cb_segments(
    observations: Sequence[ShapeObservation],
    units: Sequence[UnitSpec],
    plugin: CodebookAnchoredFormatPlugin,
    *,
    anchors: Mapping[tuple[str, SegmentKey], AnchorScalar],
) -> dict[SegmentKey, ShapeFit]:
    by_segment: dict[SegmentKey, list[ShapeObservation]] = defaultdict(list)
    ladders: dict[SegmentKey, dict[str, CandidateSpec]] = defaultdict(dict)
    for observation in observations:
        by_segment[observation.segment].append(observation)
    for unit in units:
        for segment, candidates in candidates_by_segment(unit, plugin).items():
            for candidate in candidates:
                ladders[segment][candidate.format_name] = candidate
    # A one-rung segment is priced by its anchor, so it is deliberately absent
    # from the panel; every other legal segment must still be covered.
    fitted = {
        segment for segment, ladder in ladders.items() if len(ladder) > 1
    }
    if set(by_segment) != fitted:
        raise AnchoredCostError(
            "panel does not cover every legal family/role/basis segment"
        )
    fits = {
        segment: fit_segment_shape(
            by_segment[segment],
            segment=segment,
            candidates=tuple(sorted(
                ladders[segment].values(),
                key=lambda item: (
                    item.bits, item.coordinate, item.format_name,
                ),
            )),
        )
        for segment in sorted(fitted)
    }
    for segment in sorted(set(ladders) - fitted):
        (sole,) = ladders[segment].values()
        fits[segment] = _single_rung_shape_fit(segment, sole, anchors)
    return fits


def heldout_validation_report(
    observations: Sequence[ShapeObservation],
    anchors: Mapping[tuple[str, SegmentKey], AnchorScalar],
    fits: Mapping[SegmentKey, ShapeFit],
    *,
    panel_requests: Sequence[RenderRequest],
    basis: str = LEARNED_BASIS,
    dex_bar: float = 0.05,
) -> dict[str, object]:
    """Strict-JSON, report-only factorisation validation.

    ``panel_requests`` is the panel the fit was actually trained on -- the
    rendered cells, not the policy literal -- and it is required rather than
    optional because it is what separates the report's two hold-out axes.  A
    validation cell at a rung the panel does not contain tests both cross-unit
    transfer and off-panel rung behaviour; a cell at a panel rung tests only
    the first.  Both are real evidence and both are reported, but a summary
    that cannot tell them apart lets a single-axis hold-out read as a
    two-axis one, which is precisely how a weak validation design survives
    review.  ``max_abs_dex`` and ``n_above_bar`` are therefore also broken out
    per axis.
    """
    panel_rungs_by_segment: dict[SegmentKey, set[str]] = defaultdict(set)
    for request in panel_requests:
        panel_rungs_by_segment[request.segment].add(request.format_name)
    by_unit: dict[tuple[str, SegmentKey], set[str]] = defaultdict(set)
    rows: list[dict[str, object]] = []
    for observation in observations:
        if observation.segment.equivalence_class != basis:
            continue
        by_unit[(observation.qname, observation.segment)].add(
            observation.format_name
        )
        anchor = anchors[(observation.qname, observation.segment)]
        fit = fits[observation.segment]
        if observation.receipt is None or (
            observation.receipt.arm_identity_sha256
            != fit.arm_identity_sha256
            or observation.receipt.payload_identity_sha256
            != fit.payload_identity_sha256
            or anchor.receipt.arm_identity_sha256
            != fit.arm_identity_sha256
            or anchor.receipt.payload_identity_sha256
            != fit.payload_identity_sha256
        ):
            raise AnchoredCostError(
                "validation, fit, and anchor do not share one production "
                "render identity"
            )
        predicted = anchor.predicted_dloss * fit.ratio(
            observation.format_name, anchor.format_name
        )
        measured = float(observation.predicted_dloss)
        if measured <= 0.0:
            dex_error = None
            reason = "measured_predicted_dloss_is_nonpositive"
        else:
            dex_error = abs(math.log10(max(predicted, 1e-300) / measured))
            reason = None
        two_axis = observation.format_name not in panel_rungs_by_segment[
            observation.segment
        ]
        rows.append({
            "qname": observation.qname,
            "segment": basis_segment_dict(observation.segment),
            "format": observation.format_name,
            "predicted_dloss": predicted,
            "measured_predicted_dloss": measured,
            "absolute_dex_error": dex_error,
            "dex_error_nonfinite_reason": reason,
            "within_dex_bar": (
                dex_error is not None and dex_error <= dex_bar
            ),
            "held_out_axes": ["unit", "rung"] if two_axis else ["unit"],
        })
    insufficient = [
        f"{qname}|{segment.stamp}"
        for (qname, segment), formats in by_unit.items()
        if len(formats) < 2
    ]
    if insufficient:
        raise AnchoredCostError(
            "held-out validation needs >=2 rungs per unit: "
            f"{insufficient[:8]}"
        )
    finite = [
        float(row["absolute_dex_error"])
        for row in rows if row["absolute_dex_error"] is not None
    ]
    nonfinite = sum(
        row["absolute_dex_error"] is None for row in rows
    )
    failures = nonfinite + sum(value > dex_bar for value in finite)

    def _axis_slice(axes: list[str]) -> dict[str, object]:
        subset = [row for row in rows if row["held_out_axes"] == axes]
        values = [
            float(row["absolute_dex_error"]) for row in subset
            if row["absolute_dex_error"] is not None
        ]
        blank = sum(row["absolute_dex_error"] is None for row in subset)
        return {
            "n_cells": len(subset),
            "n_above_bar": blank + sum(value > dex_bar for value in values),
            "n_nonfinite_dex": blank,
            "max_abs_dex": max(values, default=None),
        }

    two_axis = _axis_slice(["unit", "rung"])
    report = {
        "reported_not_gated": True,
        "basis": basis,
        "dex_bar": float(dex_bar),
        "n_cells": len(rows),
        "n_above_bar": failures,
        "n_nonfinite_dex": nonfinite,
        "max_abs_dex": max(finite, default=None),
        "status": (
            "BAD_FACTORISATION_SIGNAL" if failures else "within_bar"
        ),
        # The headline counts pool both hold-out axes.  These do not, because
        # they answer different questions and a pooled number can hide a
        # segment that never tested one of them at all.
        "by_held_out_axes": {
            "unit+rung": two_axis,
            "unit_only": _axis_slice(["unit"]),
        },
        "off_panel_rung_evidence": two_axis["n_cells"] > 0,
        "rows": rows,
    }
    # Enforce the portable JSON contract at construction time.
    json.dumps(report, allow_nan=False)
    return report


def fitted_cb_hull_report(
    units: Sequence[UnitSpec],
    plugin: CodebookAnchoredFormatPlugin,
    fits: Mapping[SegmentKey, ShapeFit],
) -> dict[str, object]:
    vertices: dict[SegmentKey, Counter[tuple[str, ...]]] = defaultdict(Counter)
    interiors: dict[SegmentKey, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for unit in units:
        for segment, candidates in candidates_by_segment(unit, plugin).items():
            hull = lower_convex_hull({
                candidate.format_name: (
                    candidate.bits,
                    float(fits[segment].g_by_format[candidate.format_name]),
                )
                for candidate in candidates
            })
            vertices[segment][hull.vertices] += 1
            interiors[segment][hull.interior] += 1
    return {
        "computed_not_inherited": True,
        "segment_key_fields": list(GENERIC_SEGMENT_FIELDS),
        "cb_segment_alias_fields": list(CB_SEGMENT_FIELDS),
        "equivalence_vocabulary_name": "codebook_basis",
        "segments": {
            segment.stamp: {
                "segment": basis_segment_dict(segment),
                "vertex_sets": [
                    {"formats": list(formats), "unit_count": count}
                    for formats, count in sorted(vertices[segment].items())
                ],
                "interior_sets": [
                    {"formats": list(formats), "unit_count": count}
                    for formats, count in sorted(interiors[segment].items())
                ],
            }
            for segment in sorted(vertices)
        },
    }


def build_cb_allocator_cost_payload(
    cells: Mapping[str, Sequence[PricedCell]],
    *,
    streamed_payload: Mapping[str, object],
    fits: Mapping[SegmentKey, ShapeFit],
    hull_report: Mapping[str, object],
    validation_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a truthful sparse-measurement/full-extrapolation AURA table."""
    assert_aura_only_cost_table(cells)
    provenance = streamed_payload.get("provenance")
    streamed_costs = streamed_payload.get("costs")
    if not isinstance(provenance, Mapping) or not isinstance(
        streamed_costs, Mapping
    ):
        raise AnchoredCostError(
            "streamed AURA payload lacks measured cost/provenance"
        )
    sparse_identity = provenance.get(
        "production_anchor_sparse_render_identity"
    )
    if not isinstance(sparse_identity, Mapping):
        raise AnchoredCostError(
            "streamed AURA payload lacks sparse production render identity"
        )
    extrapolation_identity = provenance.get("cb_render_identity")
    if not isinstance(extrapolation_identity, Mapping):
        raise AnchoredCostError(
            "streamed AURA payload lacks full extrapolation-input identity"
        )
    costs = {
        qname: {
            cell.candidate.format_name: cell.allocation_entry()
            for cell in row
        }
        for qname, row in sorted(cells.items())
    }
    measurements: dict[str, dict[str, object]] = {}
    legal_scope: dict[str, list[str]] = {}
    receipt_binding = _streamed_receipt_binding(streamed_payload)
    for qname, row in sorted(cells.items()):
        per_unit: dict[str, object] = {}
        legal_scope[qname] = []
        for cell in row:
            if cell.segment is None:
                continue
            legal_scope[qname].append(cell.candidate.format_name)
            stamp = cell.segment.stamp
            if stamp in per_unit:
                continue
            try:
                measured = streamed_costs[qname][cell.anchor_format]
            except (KeyError, TypeError) as exc:
                raise AnchoredCostError(
                    f"{qname}/{stamp}: measured anchor is absent"
                ) from exc
            if not isinstance(measured, Mapping) or (
                measured.get("dw_source") != "production_render"
                or measured.get("production_anchor_measured") is not True
            ):
                raise AnchoredCostError(
                    f"{qname}/{stamp}: anchor is not a production render"
                )
            observed = float(measured["predicted_dloss"])
            if observed != float(cell.anchor_predicted_dloss):
                raise AnchoredCostError(
                    f"{qname}/{stamp}: anchor level differs from measurement"
                )
            receipt = _receipt_from_streamed_payload(
                RenderRequest(
                    qname, cell.segment, str(cell.anchor_format), "anchor",
                ),
                receipt_binding,
            )
            per_unit[stamp] = {
                "segment": basis_segment_dict(cell.segment),
                "format": cell.anchor_format,
                "predicted_dloss": observed,
                "dw_source": "production_render",
                "production_anchor_measured": True,
                "production_render_receipt_sha256": receipt.receipt_sha256,
                "arm_identity_sha256": receipt.arm_identity_sha256,
                "payload_identity_sha256": receipt.payload_identity_sha256,
            }
        measurements[qname] = per_unit
    segment_fits = {
        segment.stamp: {
            "segment": basis_segment_dict(segment),
            "g_by_format": dict(fit.g_by_format),
            "reference_format": fit.reference_format,
            "coefficients": list(fit.coefficients),
            "design_rank": fit.design_rank,
            "design_rank_required": fit.design_rank_required,
            "arm_identity_sha256": fit.arm_identity_sha256,
            "payload_identity_sha256": fit.payload_identity_sha256,
            "panel_receipts_sha256": fit.panel_receipts_sha256,
            "shape_fit_currency": fit.shape_fit_currency,
            "aura_vs_weight_diagnostic": fit.aura_vs_weight_diagnostic,
        }
        for segment, fit in sorted(fits.items())
    }
    anchored_provenance = dict(provenance)
    anchored_provenance.update({
        "schema": CB_ANCHORED_COST_SCHEMA,
        "cost_currency": AURA_CURRENCY,
        "fisher_application_count": 1,
        "segment_key_fields": list(GENERIC_SEGMENT_FIELDS),
        "cb_segment_alias_fields": list(CB_SEGMENT_FIELDS),
        "equivalence_vocabulary_name": "codebook_basis",
        "segment_fits": segment_fits,
        "production_anchor_measurements": measurements,
        "full_legal_cb_formats_by_qname": legal_scope,
        # Keep measured outputs and extrapolated renderer inputs separate.
        # ``cb_render_identity`` is the full legal INPUT domain required so
        # allocator/export can reproduce a selected rung.  It does not claim
        # that those outputs were rendered; the sparse identity below is the
        # exact measured domain.
        "production_anchor_sparse_render_identity": dict(sparse_identity),
        "cb_render_identity": dict(extrapolation_identity),
        "extrapolation_input_domain": {
            "formats_by_qname": legal_scope,
            "source_identity": {
                "source_weights_complete": extrapolation_identity.get(
                    "source_weights_complete"
                ),
                "source_weights_shapes": extrapolation_identity.get(
                    "source_weights_shapes"
                ),
                "source_weights_content_sha256": extrapolation_identity.get(
                    "source_weights_content_sha256"
                ),
            },
            "cb_serialized_payload": extrapolation_identity.get(
                "cb_serialized_payload"
            ),
            "col_weights_shapes": extrapolation_identity.get(
                "col_weights_shapes"
            ),
            "col_weights_content_sha256": extrapolation_identity.get(
                "col_weights_content_sha256"
            ),
            "outputs_materialized": False,
            "measurement_scope": "sparse_anchor_panel_validation_only",
        },
        "extrapolated_cells_are_not_rendered_measurements": True,
        "lower_convex_hull": dict(hull_report),
        "cb_col_weights_role": "production_render_input_only",
        "route_flip_limitation": ROUTE_FLIP_LIMITATION,
        **(
            {"held_out_validation": dict(validation_report)}
            if validation_report is not None else {}
        ),
    })
    return {
        "schema": CB_ANCHORED_COST_SCHEMA,
        "formats": sorted({
            format_name for row in costs.values() for format_name in row
        }),
        "costs": costs,
        "provenance": anchored_provenance,
        "meta": {
            "cost_currency": AURA_CURRENCY,
            "cost_semantics": (
                "measured AURA anchor level times a within-(family,role,basis) "
                "g ratio; no h_trace, imatrix dispersion, weight-MSE, or "
                "parallel activation cost is applied"
            ),
            "unit_count": len(costs),
            "cell_count": sum(len(row) for row in costs.values()),
        },
    }


def build_cb_extrapolation_input_identity(
    sparse_identity: Mapping[str, object],
    *,
    legal_formats_by_qname: Mapping[str, Sequence[str]],
    col_weights: Mapping[str, object],
    cb_serialization_context,
) -> dict[str, object]:
    """Expand measured CB scope into a truthful renderer-input domain.

    The sparse identity proves which anchor/panel/validation cells actually
    produced ``dW``.  Allocation and export still need the exact source,
    imatrix, codebook, and production-arm identity for every legal rung they
    may select.  This helper expands only those immutable inputs; it never
    creates a score row or claims that an extrapolated output was rendered.
    """
    from prismaquant.production_weight_cache import (
        build_production_cache_cb_render_identity,
        validate_cb_render_identity_metadata,
    )

    try:
        validate_cb_render_identity_metadata(
            sparse_identity,
            expected_context=cb_serialization_context,
            col_weights=col_weights,
            where="CB sparse production-anchor render identity",
        )
    except ValueError as exc:
        raise AnchoredCostError(str(exc)) from None
    sparse_scope = sparse_identity["cb_formats_by_qname"]
    legal_cb_scope = {
        str(qname): tuple(
            fr.canonical_format_name(str(format_name))
            for format_name in formats
            if fr.get_format(
                fr.canonical_format_name(str(format_name))
            ).family in {"nvfp4_cb", "fp8_cb"}
        )
        for qname, formats in legal_formats_by_qname.items()
    }
    legal_cb_scope = {
        qname: formats for qname, formats in legal_cb_scope.items() if formats
    }
    if set(legal_cb_scope) != set(sparse_scope):
        raise AnchoredCostError(
            "full CB extrapolation domain and measured anchor qnames differ: "
            f"missing_measured={sorted(set(legal_cb_scope) - set(sparse_scope))[:8]} "
            f"extra_measured={sorted(set(sparse_scope) - set(legal_cb_scope))[:8]}"
        )
    if sparse_identity.get("source_weights_complete") is not True:
        raise AnchoredCostError(
            "sparse production anchors lack a source-complete identity"
        )
    sparse_formats = {
        (str(qname), str(format_name))
        for qname, formats in sparse_scope.items()
        for format_name in formats
    }
    legal_formats = {
        (qname, format_name)
        for qname, formats in legal_cb_scope.items()
        for format_name in formats
    }
    if getattr(cb_serialization_context, "minchain", False) and not (
        legal_formats <= sparse_formats
    ):
        raise AnchoredCostError(
            "CB min-chain carries per-rendered-cell chain identity and cannot "
            "truthfully expand sparse anchors to unrendered legal rungs"
        )
    contract = sparse_identity.get("render_contract")
    if not isinstance(contract, Mapping):
        raise AnchoredCostError("sparse CB render contract is absent")
    try:
        expanded = build_production_cache_cb_render_identity(
            legal_cb_scope,
            cb_serialization_context=cb_serialization_context,
            col_weights=col_weights,
            render_levers=contract.get("resolved_levers"),
            render_mechanism_plan=contract.get("mechanism_plan"),
        )
    except (TypeError, ValueError) as exc:
        raise AnchoredCostError(str(exc)) from None
    if not isinstance(expanded, dict):
        raise AnchoredCostError("full CB extrapolation domain is empty")
    if expanded.get("render_contract") != contract:
        raise AnchoredCostError(
            "expanding CB legal inputs changed the production render arm"
        )
    for field in (
        "source_weights_complete",
        "source_weights_sha256",
        "source_weights_shapes",
        "source_weights_content_sha256",
    ):
        expanded[field] = copy.deepcopy(sparse_identity.get(field))
    if "cb_minchain_cells" in sparse_identity:
        expanded["cb_minchain_cells"] = copy.deepcopy(
            sparse_identity["cb_minchain_cells"]
        )
    try:
        validate_cb_render_identity_metadata(
            expanded,
            expected_context=cb_serialization_context,
            expected_formats_by_qname=legal_cb_scope,
            col_weights=col_weights,
            where="CB full extrapolation-input identity",
        )
    except ValueError as exc:
        raise AnchoredCostError(str(exc)) from None
    return expanded


def write_cb_cost_payload(
    path: str | Path, payload: Mapping[str, object],
) -> Path:
    output = Path(path)
    atomic_write_bytes(
        output,
        pickle.dumps(dict(payload), protocol=pickle.HIGHEST_PROTOCOL),
    )
    return output


def run_streamed_cb_anchor_aura(
    runner,
    calibration_ids,
    *,
    formats_by_qname: Mapping[str, Sequence[str]],
    legal_formats_by_qname: Mapping[str, Sequence[str]],
    purposes_by_qname: Mapping[str, Mapping[str, Sequence[str]]],
    activation_index,
    render_levers: Mapping[str, object],
    col_weights: Mapping[str, object],
    cb_serialization_context,
    calibration_hash: str,
    arm_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
    checkpoint_dir: str | Path,
    resume: bool,
    n_probes: int,
    profile,
    checkpoint_identity_extra: Mapping[str, object],
    cold_expert_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Wire the exact bounded plan into the existing one-pass AURA runner."""
    from prismaquant.aura_cost import run_streamed_production_anchor_aura

    unmeasured_terminals: dict[str, tuple[str, ...]] = {}
    for raw_qname, raw_formats in formats_by_qname.items():
        qname = str(raw_qname)
        rendered = {
            fr.canonical_format_name(fmt)
            for fmt in purposes_by_qname.get(qname, {})
        }
        retained = tuple(
            fr.canonical_format_name(fmt)
            for fmt in raw_formats
            if fr.canonical_format_name(fmt) not in rendered
        )
        if len(retained) != 1:
            raise AnchoredCostError(
                f"{qname}: streamed CB plan must retain exactly one "
                f"unmeasured terminal, found {list(retained)}"
            )
        unmeasured_terminals[qname] = retained

    payload = run_streamed_production_anchor_aura(
        runner,
        calibration_ids,
        formats_by_qname=formats_by_qname,
        render_purposes_by_qname=purposes_by_qname,
        unmeasured_formats_by_qname=unmeasured_terminals,
        activation_index=activation_index,
        render_levers=render_levers,
        col_weights=col_weights,
        cb_serialization_context=cb_serialization_context,
        calibration_hash=calibration_hash,
        arm_identity=arm_identity,
        model_identity=model_identity,
        cold_expert_provenance=cold_expert_provenance,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        n_probes=n_probes,
        checkpoint_identity_extra=checkpoint_identity_extra,
        include_routed_experts=True,
        profile=profile,
    )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise AnchoredCostError("streamed AURA result has no provenance")
    sparse_identity = provenance.get("cb_render_identity")
    if not isinstance(sparse_identity, Mapping):
        raise AnchoredCostError(
            "streamed AURA result has no sparse CB render identity"
        )
    expanded_identity = build_cb_extrapolation_input_identity(
        sparse_identity,
        legal_formats_by_qname=legal_formats_by_qname,
        col_weights=col_weights,
        cb_serialization_context=cb_serialization_context,
    )
    provenance.update({
        "production_anchor_sparse_render_identity": copy.deepcopy(
            dict(sparse_identity)
        ),
        "production_anchor_sparse_serialized_payload": copy.deepcopy(
            sparse_identity.get("cb_serialized_payload")
        ),
        "cb_render_identity": expanded_identity,
        "cb_serialized_payload": copy.deepcopy(
            expanded_identity["cb_serialized_payload"]
        ),
        "cb_anchored_plugin": {
            "segment_key_fields": list(GENERIC_SEGMENT_FIELDS),
            "cb_segment_alias_fields": list(CB_SEGMENT_FIELDS),
            "equivalence_vocabulary_name": "codebook_basis",
            "aura_only_cost_currency": True,
            "cb_col_weights_role": "production_render_input_only",
            "route_flip_limitation": ROUTE_FLIP_LIMITATION,
        },
        "full_menu_materialized": False,
    })
    return payload


def merge_streamed_cb_anchor_aura_shards(
    payloads: Sequence[Mapping[str, object]],
    *,
    col_weights: Mapping[str, object],
    expected_qnames: Sequence[str],
    expected_formats_by_qname: Mapping[str, Sequence[str]],
    expected_purposes_by_qname: Mapping[
        str, Mapping[str, Sequence[str]]
    ],
    expected_unmeasured_formats_by_qname: Mapping[str, Sequence[str]],
    expected_legal_cb_formats_by_qname: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Reconstruct one global anchored-AURA payload from disjoint qnames.

    A generic cost-pickle union is insufficient here.  Per-cell production
    receipts bind the *global* renderer plan, purpose map, and CB render
    identity.  This merge therefore rebuilds those value-bearing objects and
    only then lets :func:`anchors_from_streamed_payload` mint receipts.  No
    receipt from an input shard is carried forward.

    The partition axis is deliberately qname, not format: every sparse cell
    for one Linear stays with one worker, source-weight identities form a
    disjoint cover, and one worker can install a decoder layer once for all of
    that layer's anchor/panel/validation renders.
    """
    from prismaquant.production_weight_cache import (
        merge_cb_render_identities,
        validate_cb_render_identity_metadata,
    )

    if not payloads:
        raise AnchoredCostError("streamed CB shard merge needs at least one shard")
    if not isinstance(col_weights, Mapping) or not col_weights:
        raise AnchoredCostError("streamed CB shard merge needs current col_weights")

    required_top_shared = ("schema", "n_probes", "token_scope")
    required_provenance_shared = (
        "seed_base",
        "temperature",
        "dw_dtype",
        "measurement_dtype",
        "include_lm_head",
        "calib_shape",
        "calib_sha256",
        "calib_hash",
        "calib_hashes",
        "omitted_packed_experts",
        "git_commit",
        "streaming",
        "streamed_gradient_harvest",
        "streamed_cotangent_rollover",
        "streamed_boundary_release",
        "cb_cost_provenance_schema",
        "cb_anchored_plugin",
        "full_menu_materialized",
        "production_anchor_no_full_menu_materialization",
        "production_anchor_cost_currency",
        "weight_mse_diagnostic_is_cost_input",
    )
    required_renderer_shared = (
        "schema",
        "calibration_hash",
        "max_act_rows",
        "arm_identity",
        "source_model",
        "source_weight_binding",
        "cold_expert_provenance",
        "producer_git_commit",
        "producer_source_sha256",
        "retention",
        "transient_consumer_identity",
    )

    def _mapping(value: object, *, where: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise AnchoredCostError(f"{where} is not a mapping")
        return value

    def _canonical_formats(
        value: object, *, where: str, nonempty: bool = True,
    ) -> tuple[str, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise AnchoredCostError(f"{where} is not a format sequence")
        result = tuple(
            fr.canonical_format_name(str(format_name))
            for format_name in value
        )
        if (nonempty and not result) or len(result) != len(set(result)):
            raise AnchoredCostError(
                f"{where} is empty or contains canonical duplicates"
            )
        return result

    expected_sequence = tuple(str(name) for name in expected_qnames)
    expected_qname_set = set(expected_sequence)
    if not expected_sequence or len(expected_sequence) != len(expected_qname_set):
        raise AnchoredCostError(
            "streamed CB shard merge expected qname plan is empty or has "
            "duplicates"
        )
    for label, plan in (
        ("full sparse formats", expected_formats_by_qname),
        ("render purposes", expected_purposes_by_qname),
        ("unmeasured terminals", expected_unmeasured_formats_by_qname),
        ("legal CB formats", expected_legal_cb_formats_by_qname),
    ):
        if not isinstance(plan, Mapping) or set(map(str, plan)) != (
            expected_qname_set
        ):
            raise AnchoredCostError(
                f"streamed CB shard merge expected {label} plan does not "
                "exactly cover expected qnames"
            )

    def _canonical_format_plan(
        value: Mapping[str, Sequence[str]],
    ) -> dict[str, tuple[str, ...]]:
        return {
            str(name): _canonical_formats(
                formats, where=f"declared format plan for {name}"
            )
            for name, formats in sorted(value.items())
        }

    def _canonical_cb_format_set_plan(
        value: Mapping[str, Sequence[str]],
    ) -> dict[str, tuple[str, ...]]:
        """Canonicalize an unordered legal-CB domain like its identity does."""
        return {
            str(name): tuple(sorted({
                fr.canonical_format_name(str(format_name))
                for format_name in formats
            }))
            for name, formats in sorted(value.items())
        }

    expected_full_formats = _canonical_format_plan(
        expected_formats_by_qname
    )
    expected_unmeasured = _canonical_format_plan(
        expected_unmeasured_formats_by_qname
    )
    expected_rendered: dict[str, tuple[str, ...]] = {}
    for qname in sorted(expected_qname_set):
        full = expected_full_formats[qname]
        terminal = expected_unmeasured[qname]
        if len(terminal) != 1 or not set(terminal).issubset(full):
            raise AnchoredCostError(
                f"streamed CB shard merge {qname} must declare exactly one "
                "unmeasured terminal inside its full sparse format plan"
            )
        terminal_set = set(terminal)
        rendered = tuple(
            format_name for format_name in full
            if format_name not in terminal_set
        )
        if not rendered or set(rendered) & terminal_set or (
            set(rendered) | terminal_set != set(full)
        ):
            raise AnchoredCostError(
                f"streamed CB shard merge {qname} full sparse plan is not "
                "the exact disjoint union of rendered formats and terminal"
            )
        expected_rendered[qname] = rendered

    def _canonical_purposes(value):
        if not isinstance(value, Mapping):
            raise AnchoredCostError("render-purpose plan is not a mapping")
        result: dict[str, dict[str, tuple[str, ...]]] = {}
        for raw_name, raw_rows in sorted(value.items()):
            if not isinstance(raw_rows, Mapping):
                raise AnchoredCostError(
                    f"render-purpose plan for {raw_name} is not a mapping"
                )
            rows: dict[str, tuple[str, ...]] = {}
            for raw_format, raw_purposes in sorted(raw_rows.items()):
                if not isinstance(raw_purposes, Sequence) or isinstance(
                    raw_purposes, (str, bytes)
                ):
                    raise AnchoredCostError(
                        f"render-purpose declaration for "
                        f"{raw_name}@{raw_format} is malformed"
                    )
                purposes = tuple(str(purpose) for purpose in raw_purposes)
                if (
                    not purposes or len(purposes) != len(set(purposes))
                    or not set(purposes).issubset({
                        "anchor", "panel", "validation",
                    })
                ):
                    raise AnchoredCostError(
                        f"render-purpose declaration for "
                        f"{raw_name}@{raw_format} is invalid"
                    )
                format_name = fr.canonical_format_name(str(raw_format))
                if format_name in rows:
                    raise AnchoredCostError(
                        f"render-purpose declaration for {raw_name} repeats "
                        f"canonical format {format_name}"
                    )
                rows[format_name] = tuple(sorted(purposes))
            result[str(raw_name)] = rows
        return result

    expected_purposes = _canonical_purposes(
        expected_purposes_by_qname
    )
    for qname, rows in expected_purposes.items():
        if set(rows) != set(expected_rendered[qname]):
            raise AnchoredCostError(
                f"streamed CB shard merge {qname} declared purposes do not "
                "exactly cover the rendered subset of the full sparse plan"
            )

    def _payload_sort_key(payload: Mapping[str, object]) -> tuple[str, ...]:
        costs = payload.get("costs")
        if not isinstance(costs, Mapping):
            return ()
        return tuple(sorted(str(name) for name in costs))

    # A canonical shard order makes the reconstructed provenance and its
    # receipt payload digest invariant to command-line/input arrival order.
    ordered_payloads = tuple(sorted(payloads, key=_payload_sort_key))
    first = ordered_payloads[0]
    first_provenance = _mapping(
        first.get("provenance"), where="streamed CB shard 0 provenance"
    )
    first_renderer = _mapping(
        first_provenance.get("production_anchor_renderer"),
        where="streamed CB shard 0 renderer",
    )
    top_reference = {key: copy.deepcopy(first.get(key)) for key in required_top_shared}
    provenance_reference = {
        key: copy.deepcopy(first_provenance.get(key))
        for key in required_provenance_shared
    }
    renderer_reference = {
        key: copy.deepcopy(first_renderer.get(key))
        for key in required_renderer_shared
    }
    if provenance_reference["full_menu_materialized"] is not False or (
        provenance_reference[
            "production_anchor_no_full_menu_materialization"
        ] is not True
    ):
        raise AnchoredCostError(
            "streamed CB shards do not declare sparse transient rendering"
        )
    if provenance_reference["production_anchor_cost_currency"] != (
        "aura_only"
    ) or provenance_reference["weight_mse_diagnostic_is_cost_input"] is not (
        False
    ):
        raise AnchoredCostError(
            "streamed CB shards do not declare AURA-only cost currency"
        )

    merged_costs: dict[str, object] = {}
    merged_stats: dict[str, object] = {}
    merged_renderer_formats: dict[str, object] = {}
    merged_purposes: dict[str, object] = {}
    merged_unmeasured: dict[str, object] = {}
    merged_source_records: dict[str, object] = {}
    sparse_identities: list[Mapping[str, object]] = []
    expanded_identities: list[Mapping[str, object]] = []
    shard_scopes: list[list[str]] = []
    format_union: set[str] = set()

    sum_fields = (
        "dw_rendered_rows",
        "dw_production_anchor_rows",
        "dw_rtn_fallback_rows",
        "weight_mse_diagnostic_rows",
        "production_anchor_expected_renders",
        "production_anchor_rendered_this_invocation",
        "production_anchor_restored_renders",
        "production_anchor_union_render_count",
    )
    sums = {key: 0 for key in sum_fields}
    shard_linear_chunks: list[int] = []
    maximum_live = 0
    purpose_counts: Counter[str] = Counter()

    for index, raw_payload in enumerate(ordered_payloads):
        payload = _mapping(raw_payload, where=f"streamed CB shard {index}")
        provenance = _mapping(
            payload.get("provenance"),
            where=f"streamed CB shard {index} provenance",
        )
        renderer = _mapping(
            provenance.get("production_anchor_renderer"),
            where=f"streamed CB shard {index} renderer",
        )
        for key, expected in top_reference.items():
            if payload.get(key) != expected:
                raise AnchoredCostError(
                    f"streamed CB shard {index} top-level {key!r} differs"
                )
        for key, expected in provenance_reference.items():
            if provenance.get(key) != expected:
                raise AnchoredCostError(
                    f"streamed CB shard {index} provenance {key!r} differs"
                )
        for key, expected in renderer_reference.items():
            if renderer.get(key) != expected:
                raise AnchoredCostError(
                    f"streamed CB shard {index} renderer {key!r} differs"
                )

        costs = _mapping(
            payload.get("costs"), where=f"streamed CB shard {index} costs"
        )
        stats = _mapping(
            payload.get("stats"), where=f"streamed CB shard {index} stats"
        )
        renderer_formats = _mapping(
            renderer.get("formats_by_qname"),
            where=f"streamed CB shard {index} renderer formats",
        )
        purposes = _mapping(
            provenance.get("production_anchor_render_purposes"),
            where=f"streamed CB shard {index} purposes",
        )
        unmeasured = _mapping(
            provenance.get("production_anchor_unmeasured_formats_by_qname"),
            where=f"streamed CB shard {index} unmeasured terminals",
        )
        sparse = _mapping(
            provenance.get("production_anchor_sparse_render_identity"),
            where=f"streamed CB shard {index} sparse CB identity",
        )
        expanded = _mapping(
            provenance.get("cb_render_identity"),
            where=f"streamed CB shard {index} expanded CB identity",
        )
        sparse_scope = _mapping(
            sparse.get("cb_formats_by_qname"),
            where=f"streamed CB shard {index} sparse CB scope",
        )
        expanded_scope = _mapping(
            expanded.get("cb_formats_by_qname"),
            where=f"streamed CB shard {index} expanded CB scope",
        )
        source_weights = _mapping(
            renderer.get("source_weights"),
            where=f"streamed CB shard {index} source weights",
        )
        source_records = _mapping(
            source_weights.get("records"),
            where=f"streamed CB shard {index} source records",
        )
        if source_weights.get("complete") is not True or (
            source_weights.get("scope") != "sparse_anchor_plan"
        ):
            raise AnchoredCostError(
                f"streamed CB shard {index} source-weight binding is not "
                "complete sparse_anchor_plan"
            )
        source_records_digest = canonical_json_sha256(
            source_records,
            where=f"streamed CB shard {index} source records",
        )
        if source_weights.get("identity_sha256") != source_records_digest:
            raise AnchoredCostError(
                f"streamed CB shard {index} source-weight identity differs"
            )
        if renderer.get("cb_render_identity") != sparse:
            raise AnchoredCostError(
                f"streamed CB shard {index} renderer/sparse CB identity differs"
            )
        if provenance.get(
            "production_anchor_sparse_serialized_payload"
        ) != sparse.get("cb_serialized_payload") or provenance.get(
            "cb_serialized_payload"
        ) != expanded.get("cb_serialized_payload"):
            raise AnchoredCostError(
                f"streamed CB shard {index} serialized CB identity differs"
            )
        try:
            sparse_context = validate_cb_render_identity_metadata(
                sparse,
                col_weights=col_weights,  # type: ignore[arg-type]
                where=f"streamed CB shard {index} sparse identity",
            )
            expanded_context = validate_cb_render_identity_metadata(
                expanded,
                col_weights=col_weights,  # type: ignore[arg-type]
                where=f"streamed CB shard {index} expanded identity",
            )
        except (TypeError, ValueError) as exc:
            raise AnchoredCostError(str(exc)) from None
        if sparse_context != expanded_context:
            raise AnchoredCostError(
                f"streamed CB shard {index} sparse/expanded context differs"
            )
        if getattr(sparse_context, "minchain", False) and sparse.get(
            "cb_minchain_cells"
        ) != expanded.get("cb_minchain_cells"):
            raise AnchoredCostError(
                f"streamed CB shard {index} sparse/expanded "
                "cb_minchain_cells differs"
            )
        for field in (
            "render_contract",
            "col_weights_schema",
            "col_weights_qnames",
            "col_weights_entries",
            "col_weights_shapes",
            "col_weights_content_sha256",
            "col_weights_sha256",
            "source_weights_schema",
            "source_weights_complete",
            "source_weights_shapes",
            "source_weights_content_sha256",
            "source_weights_sha256",
        ):
            if sparse.get(field) != expanded.get(field):
                raise AnchoredCostError(
                    f"streamed CB shard {index} sparse/expanded {field} differs"
                )
        expected_source_records = {
            str(name): {
                "shape": copy.deepcopy(sparse["source_weights_shapes"][name]),
                "sha256": str(
                    sparse["source_weights_content_sha256"][name]
                ),
            }
            for name in sparse_scope
        }
        if dict(source_records) != expected_source_records:
            raise AnchoredCostError(
                f"streamed CB shard {index} renderer/source CB identity differs"
            )
        scope = set(map(str, costs))
        scoped_sets = {
            "stats": set(map(str, stats)),
            "renderer formats": set(map(str, renderer_formats)),
            "purposes": set(map(str, purposes)),
            "unmeasured terminals": set(map(str, unmeasured)),
            "sparse CB identity": set(map(str, sparse_scope)),
            "expanded CB identity": set(map(str, expanded_scope)),
            "source records": set(map(str, source_records)),
        }
        for label, observed in scoped_sets.items():
            if observed != scope:
                raise AnchoredCostError(
                    f"streamed CB shard {index} {label} scope differs from "
                    f"costs: missing={sorted(scope - observed)[:8]} "
                    f"unexpected={sorted(observed - scope)[:8]}"
                )
        overlap = set(merged_costs) & scope
        if overlap:
            raise AnchoredCostError(
                "streamed CB shard qname scopes overlap; "
                f"sample={sorted(overlap)[:8]}"
            )
        if not scope:
            raise AnchoredCostError(f"streamed CB shard {index} is empty")

        shard_rendered_count = 0
        shard_diagnostic_rows = 0
        shard_purpose_counts: Counter[str] = Counter()
        shard_format_union: set[str] = set()
        for qname in sorted(scope):
            cost_row = _mapping(
                costs[qname],
                where=f"streamed CB shard {index} costs for {qname}",
            )
            purpose_row = _mapping(
                purposes[qname],
                where=f"streamed CB shard {index} purposes for {qname}",
            )
            rendered = _canonical_formats(
                renderer_formats[qname],
                where=(
                    f"streamed CB shard {index} rendered formats for {qname}"
                ),
            )
            unmeasured_row = _canonical_formats(
                unmeasured[qname],
                where=(
                    f"streamed CB shard {index} unmeasured terminal for {qname}"
                ),
            )
            if len(unmeasured_row) != 1:
                raise AnchoredCostError(
                    f"streamed CB shard {index} {qname} must have exactly one "
                    "unmeasured terminal"
                )
            cost_formats = tuple(
                fr.canonical_format_name(str(format_name))
                for format_name in cost_row
            )
            if len(cost_formats) != len(set(cost_formats)) or set(
                cost_formats
            ) != set(rendered):
                raise AnchoredCostError(
                    f"streamed CB shard {index} measured/rendered formats "
                    f"differ for {qname}"
                )
            purpose_formats = {
                fr.canonical_format_name(str(format_name))
                for format_name in purpose_row
            }
            if len(purpose_formats) != len(purpose_row) or purpose_formats != (
                set(rendered)
            ):
                raise AnchoredCostError(
                    f"streamed CB shard {index} measured/purpose formats "
                    f"differ for {qname}"
                )
            if set(rendered) & set(unmeasured_row):
                raise AnchoredCostError(
                    f"streamed CB shard {index} rendered/unmeasured formats "
                    f"overlap for {qname}"
                )
            sparse_formats = set(_canonical_formats(
                sparse_scope[qname],
                where=f"streamed CB shard {index} sparse formats for {qname}",
            ))
            expanded_formats = set(_canonical_formats(
                expanded_scope[qname],
                where=f"streamed CB shard {index} legal formats for {qname}",
            ))
            rendered_cb = {
                format_name for format_name in rendered
                if fr.get_format(format_name).family in {"nvfp4_cb", "fp8_cb"}
            }
            if sparse_formats != rendered_cb or not sparse_formats.issubset(
                expanded_formats
            ) or set(unmeasured_row) & expanded_formats:
                raise AnchoredCostError(
                    f"streamed CB shard {index} measured/sparse/expanded "
                    f"format surfaces differ for {qname}"
                )
            purpose_values_by_format: dict[str, tuple[str, ...]] = {}
            for raw_format, raw_purposes in purpose_row.items():
                format_name = fr.canonical_format_name(str(raw_format))
                if not isinstance(raw_purposes, Sequence) or isinstance(
                    raw_purposes, (str, bytes)
                ):
                    raise AnchoredCostError(
                        f"streamed CB shard {index} purposes are malformed "
                        f"for {qname}@{format_name}"
                    )
                purpose_values = tuple(str(value) for value in raw_purposes)
                if not purpose_values or len(purpose_values) != len(
                    set(purpose_values)
                ) or not set(purpose_values).issubset(
                    {"anchor", "panel", "validation"}
                ):
                    raise AnchoredCostError(
                        f"streamed CB shard {index} purposes are invalid "
                        f"for {qname}@{format_name}"
                    )
                purpose_values_by_format[format_name] = purpose_values
                shard_purpose_counts.update(purpose_values)
            for raw_format, raw_row in cost_row.items():
                format_name = fr.canonical_format_name(str(raw_format))
                if not isinstance(raw_row, Mapping) or (
                    raw_row.get("dw_source") != "production_render"
                    or raw_row.get("production_anchor_measured") is not True
                ):
                    raise AnchoredCostError(
                        f"streamed CB shard {index} has an unmeasured cost "
                        f"row for {qname}@{format_name}"
                    )
                for scalar_name in (
                    "predicted_dloss", "predicted_dloss_stderr",
                ):
                    try:
                        scalar_value = float(raw_row[scalar_name])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise AnchoredCostError(
                            f"streamed CB shard {index} "
                            f"{qname}@{format_name} {scalar_name} is absent "
                            "or nonnumeric"
                        ) from exc
                    if not math.isfinite(scalar_value) or scalar_value < 0.0:
                        raise AnchoredCostError(
                            f"streamed CB shard {index} "
                            f"{qname}@{format_name} {scalar_name} must be "
                            "finite and nonnegative"
                        )
                diagnostic = raw_row.get("weight_mse_diagnostic")
                has_diagnostic = "weight_mse_diagnostic" in raw_row
                is_panel = "panel" in purpose_values_by_format[format_name]
                if has_diagnostic != is_panel:
                    raise AnchoredCostError(
                        f"streamed CB shard {index} {qname}@{format_name} "
                        "must carry a weight-MSE diagnostic iff it is a "
                        "fitting-panel cell"
                    )
                if has_diagnostic:
                    try:
                        diagnostic_value = float(diagnostic)
                    except (TypeError, ValueError) as exc:
                        raise AnchoredCostError(
                            f"streamed CB shard {index} "
                            f"{qname}@{format_name} weight-MSE diagnostic "
                            "is not numeric"
                        ) from exc
                    if not math.isfinite(diagnostic_value) or (
                        diagnostic_value < 0.0
                    ):
                        raise AnchoredCostError(
                            f"streamed CB shard {index} "
                            f"{qname}@{format_name} weight-MSE diagnostic "
                            "must be finite and nonnegative"
                        )
                    if raw_row.get(
                        "weight_mse_diagnostic_normalization"
                    ) != "mean_per_weight" or raw_row.get(
                        "weight_mse_is_cost_input"
                    ) is not False:
                        raise AnchoredCostError(
                            f"streamed CB shard {index} "
                            f"{qname}@{format_name} weight-MSE diagnostic "
                            "contract differs"
                        )
                    shard_diagnostic_rows += 1
            shard_rendered_count += len(rendered)
            shard_format_union.update(rendered)
            shard_format_union.update(unmeasured_row)

        payload_formats = _canonical_formats(
            payload.get("formats", ()),
            where=f"streamed CB shard {index} top-level formats",
        )
        if set(payload_formats) != shard_format_union:
            raise AnchoredCostError(
                f"streamed CB shard {index} top-level format union differs"
            )
        if int(renderer.get("requested_entries", -1)) != shard_rendered_count:
            raise AnchoredCostError(
                f"streamed CB shard {index} renderer request count differs"
            )
        expected_renders = int(provenance.get(
            "production_anchor_expected_renders", -1
        ))
        union_renders = int(provenance.get(
            "production_anchor_union_render_count", -1
        ))
        rendered_now = int(provenance.get(
            "production_anchor_rendered_this_invocation", -1
        ))
        restored = int(provenance.get(
            "production_anchor_restored_renders", -1
        ))
        if (
            expected_renders != shard_rendered_count
            or union_renders != shard_rendered_count
            or rendered_now < 0
            or restored < 0
            or rendered_now + restored != shard_rendered_count
        ):
            raise AnchoredCostError(
                f"streamed CB shard {index} render counters differ from plan"
            )
        if (
            int(provenance.get("dw_rendered_rows", -1)) != 0
            or int(provenance.get("dw_rtn_fallback_rows", -1)) != 0
            or int(provenance.get("dw_production_anchor_rows", -1))
            != shard_rendered_count
            or int(provenance.get("weight_mse_diagnostic_rows", -1))
            != shard_diagnostic_rows
        ):
            raise AnchoredCostError(
                f"streamed CB shard {index} measured/diagnostic row counters "
                "differ from cost cells"
            )
        raw_counts = provenance.get("production_anchor_purpose_counts")
        purpose_names = ("anchor", "panel", "validation")
        observed_counts = (
            {
                name: int(raw_counts.get(name, 0))
                for name in purpose_names
            }
            if isinstance(raw_counts, Mapping)
            else None
        )
        expected_counts = {
            name: int(shard_purpose_counts.get(name, 0))
            for name in purpose_names
        }
        if (
            observed_counts != expected_counts
            or not isinstance(raw_counts, Mapping)
            or set(map(str, raw_counts)) - set(purpose_names)
        ):
            raise AnchoredCostError(
                f"streamed CB shard {index} purpose counts differ from plan"
            )
        chunks = int(provenance.get("n_linear_chunks", 0) or 0)
        if chunks < 1:
            raise AnchoredCostError(
                f"streamed CB shard {index} has no linear chunks"
            )
        shard_linear_chunks.append(chunks)

        merged_costs.update(copy.deepcopy(dict(costs)))
        merged_stats.update(copy.deepcopy(dict(stats)))
        merged_renderer_formats.update(copy.deepcopy(dict(renderer_formats)))
        merged_purposes.update(copy.deepcopy(dict(purposes)))
        merged_unmeasured.update(copy.deepcopy(dict(unmeasured)))
        merged_source_records.update(copy.deepcopy(dict(source_records)))
        sparse_identities.append(sparse)
        expanded_identities.append(expanded)
        shard_scopes.append(sorted(scope))
        format_union.update(
            fr.canonical_format_name(str(value))
            for value in payload.get("formats", ())
        )
        for key in sum_fields:
            sums[key] += int(provenance.get(key, 0) or 0)
        maximum_live = max(
            maximum_live,
            int(provenance.get("production_anchor_max_live_rendered", 0) or 0),
        )
        purpose_counts.update(shard_purpose_counts)

    observed_qnames = set(merged_costs)
    if observed_qnames != expected_qname_set:
        raise AnchoredCostError(
            "streamed CB shard merge is not the declared exact cover: "
            f"missing={sorted(expected_qname_set - observed_qnames)[:8]} "
            f"unexpected={sorted(observed_qnames - expected_qname_set)[:8]}"
        )

    observed_formats = _canonical_format_plan(merged_renderer_formats)
    observed_unmeasured = _canonical_format_plan(merged_unmeasured)
    # Validate the two non-CB plan surfaces here and the legal CB surface
    # immediately after the identity union below.
    if observed_formats != expected_rendered:
        raise AnchoredCostError(
            "streamed CB shard merge rendered format plan differs from the "
            "declared global plan"
        )
    if observed_unmeasured != expected_unmeasured:
        raise AnchoredCostError(
            "streamed CB shard merge unmeasured terminal plan differs from "
            "the declared global plan"
        )
    observed_purposes = _canonical_purposes(merged_purposes)
    if observed_purposes != expected_purposes:
        raise AnchoredCostError(
            "streamed CB shard merge render-purpose plan differs from "
            "the declared global plan"
        )
    canonical_merged_purposes = {
        qname: {
            format_name: list(purposes)
            for format_name, purposes in sorted(rows.items())
        }
        for qname, rows in sorted(observed_purposes.items())
    }

    try:
        merged_sparse = merge_cb_render_identities(
            sparse_identities,
            col_weights=col_weights,  # type: ignore[arg-type]
            where="streamed anchored-AURA sparse shard merge",
        )
        merged_expanded = merge_cb_render_identities(
            expanded_identities,
            col_weights=col_weights,  # type: ignore[arg-type]
            where="streamed anchored-AURA extrapolation shard merge",
        )
    except (TypeError, ValueError) as exc:
        raise AnchoredCostError(str(exc)) from None
    if merged_sparse is None or merged_expanded is None:
        raise AnchoredCostError("streamed CB shard merge produced no CB identity")
    observed_legal = _canonical_cb_format_set_plan(
        merged_expanded["cb_formats_by_qname"]
    )
    if observed_legal != _canonical_cb_format_set_plan(
        expected_legal_cb_formats_by_qname
    ):
        raise AnchoredCostError(
            "streamed CB shard merge legal CB plan differs from the "
            "declared global plan"
        )

    merged_renderer = copy.deepcopy(dict(first_renderer))
    merged_renderer.update({
        "formats_by_qname": dict(sorted(merged_renderer_formats.items())),
        "requested_entries": sum(
            len(tuple(formats))
            for formats in merged_renderer_formats.values()
        ),
        "cb_render_identity": merged_sparse,
        "source_weights": {
            "complete": True,
            "scope": "sparse_anchor_plan",
            "records": dict(sorted(merged_source_records.items())),
            "identity_sha256": canonical_json_sha256(
                dict(sorted(merged_source_records.items())),
                where="merged production anchor source-weight identity",
            ),
        },
    })

    merged_provenance = copy.deepcopy(dict(first_provenance))
    global_plan_identity = {
        "full_sparse_formats_by_qname": {
            name: list(formats)
            for name, formats in sorted(expected_full_formats.items())
        },
        "rendered_formats_by_qname": {
            name: list(formats)
            for name, formats in sorted(observed_formats.items())
        },
        "render_purposes_by_qname": {
            name: {
                format_name: list(purposes)
                for format_name, purposes in sorted(rows.items())
            }
            for name, rows in sorted(canonical_merged_purposes.items())
        },
        "unmeasured_formats_by_qname": {
            name: list(formats)
            for name, formats in sorted(observed_unmeasured.items())
        },
        "legal_cb_formats_by_qname": copy.deepcopy(
            merged_expanded["cb_formats_by_qname"]
        ),
    }
    global_plan_identity["identity_sha256"] = canonical_json_sha256(
        global_plan_identity,
        where="merged streamed anchored-AURA global cell plan",
    )
    merged_provenance.update({
        **sums,
        # Workers report this field, but this API has no authoritative stripe
        # chunk plan against which to validate it.  Preserve the legacy sum
        # while labeling it explicitly as unverified and non-executed.
        "n_linear_chunks": sum(shard_linear_chunks),
        "n_linear_chunks_semantics": (
            "sum_unverified_worker_reported_planned_chunks_"
            "not_execution_count"
        ),
        "production_anchor_max_live_rendered": maximum_live,
        "production_anchor_purpose_counts": {
            name: int(purpose_counts.get(name, 0))
            for name in ("anchor", "panel", "validation")
        },
        "production_anchor_renderer": merged_renderer,
        "production_anchor_render_purposes": canonical_merged_purposes,
        "production_anchor_unmeasured_formats_by_qname": dict(
            sorted(merged_unmeasured.items())
        ),
        "production_anchor_sparse_render_identity": merged_sparse,
        "production_anchor_sparse_serialized_payload": copy.deepcopy(
            merged_sparse.get("cb_serialized_payload")
        ),
        "cb_render_identity": merged_expanded,
        "cb_serialized_payload": copy.deepcopy(
            merged_expanded.get("cb_serialized_payload")
        ),
        "streamed_shard_merge": {
            "schema": STREAMED_CB_SHARD_MERGE_SCHEMA,
            "shards": len(ordered_payloads),
            "qnames": len(observed_qnames),
            "qname_scopes": shard_scopes,
            "exact_disjoint_cover": True,
            "receipt_identity_reconstructed_globally": True,
            "partition_axis": "whole_qnames",
            "shard_unverified_worker_reported_planned_chunks": list(
                shard_linear_chunks
            ),
            "n_linear_chunks_aggregation": (
                "sum_unverified_worker_reported_planned_chunks"
            ),
            "total_unverified_worker_reported_planned_chunks": sum(
                shard_linear_chunks
            ),
            "max_unverified_worker_reported_planned_chunks": max(
                shard_linear_chunks
            ),
            "global_cell_plan": global_plan_identity,
        },
    })
    merged_payload: dict[str, object] = {
        "schema": top_reference["schema"],
        "n_probes": top_reference["n_probes"],
        "formats": sorted(format_union),
        "token_scope": top_reference["token_scope"],
        "stats": dict(sorted(merged_stats.items())),
        "costs": dict(sorted(merged_costs.items())),
        "provenance": merged_provenance,
    }
    # This revalidates the exact global objects from which every later
    # per-cell receipt digest is derived.
    _streamed_receipt_binding(merged_payload)
    return merged_payload


def require_allocator_supersurrogate_support() -> None:
    """Fail before P0 until allocator admission explicitly understands AURA."""
    from prismaquant import allocator_candidates as candidates

    ready = (
        getattr(candidates, "AURA_SUPERSURROGATE_ALLOCATOR_SEMANTICS", False)
        and callable(getattr(
            candidates, "cost_entry_is_anchored_aura_supersurrogate", None
        ))
    )
    if not ready:
        raise AnchoredCostError(
            "allocator lacks explicit AURA supersurrogate activation-transfer "
            "and measured-zero admission semantics; refusing before P0"
        )


def _bytes_descriptor(payload: bytes) -> dict[str, object]:
    return {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _relocate_unbound_artifact_directory(root: Path) -> Path:
    base = root.with_name(f"{root.name}.incomplete-unbound")
    relocated = base
    suffix = 0
    while relocated.exists() or relocated.is_symlink():
        suffix += 1
        relocated = base.with_name(f"{base.name}-{suffix}")
    root.rename(relocated)
    parent_fd = os.open(root.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return relocated


def _load_artifact_publish_manifest(
    root: Path,
    *,
    expected_identity: Mapping[str, object],
    expected_identity_sha256: str,
    expected_outputs: Mapping[str, object],
) -> bool:
    path = root / _CB_ARTIFACT_PUBLISH_MANIFEST
    try:
        manifest = json.loads(path.read_text())
    except Exception as exc:
        raise AnchoredCostError(
            f"artifact publish identity at {path} is unreadable; refusing "
            "reuse or overwrite"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise AnchoredCostError(
            "artifact publish identity is not an object; refusing reuse or "
            "overwrite"
        )
    stored_identity = manifest.get("identity")
    try:
        stored_identity_sha256 = canonical_json_sha256(
            stored_identity, where="stored CB artifact publish identity",
        )
    except (TypeError, ValueError) as exc:
        raise AnchoredCostError(
            "artifact publish identity is corrupt; refusing reuse or "
            "overwrite"
        ) from exc
    if (
        manifest.get("schema") != CB_ARTIFACT_PUBLISH_SCHEMA
        or manifest.get("identity_sha256") != stored_identity_sha256
    ):
        raise AnchoredCostError(
            "artifact publish identity checksum differs; refusing reuse or "
            "overwrite"
        )
    if (
        stored_identity_sha256 != expected_identity_sha256
        or stored_identity != expected_identity
    ):
        raise AnchoredCostError(
            "artifact publish identity mismatch: "
            f"stored={stored_identity_sha256} "
            f"current={expected_identity_sha256}; refusing reuse or overwrite"
        )
    if manifest.get("outputs") != expected_outputs:
        raise AnchoredCostError(
            "artifact publish output identity differs; refusing reuse or "
            "overwrite"
        )
    complete = manifest.get("complete")
    if not isinstance(complete, bool):
        raise AnchoredCostError(
            "artifact publish completion state is invalid; refusing reuse "
            "or overwrite"
        )
    return complete


def _write_artifact_publish_manifest(
    root: Path,
    *,
    identity: Mapping[str, object],
    identity_sha256: str,
    outputs: Mapping[str, object],
    complete: bool,
) -> None:
    manifest = {
        "schema": CB_ARTIFACT_PUBLISH_SCHEMA,
        "identity_sha256": identity_sha256,
        "identity": dict(identity),
        "outputs": dict(outputs),
        "complete": bool(complete),
    }
    atomic_write_bytes(
        root / _CB_ARTIFACT_PUBLISH_MANIFEST,
        json.dumps(
            manifest, indent=2, sort_keys=True, allow_nan=False,
        ).encode(),
    )


def _validate_or_publish_artifact_files(
    root: Path,
    expected_payloads: Mapping[str, bytes],
    expected_outputs: Mapping[str, object],
    *,
    publish_missing: bool = True,
) -> None:
    for name in _CB_ARTIFACT_OUTPUT_NAMES:
        path = root / name
        descriptor = expected_outputs[name]
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise AnchoredCostError(
                    f"artifact output is not a real file: {path}; refusing "
                    "overwrite"
                )
            actual = {
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            if actual != descriptor:
                raise AnchoredCostError(
                    f"artifact completed output checksum differs for {path}; "
                    "refusing overwrite"
                )
            continue
        if not publish_missing:
            raise AnchoredCostError(
                f"completed artifact output is missing: {path}; refusing "
                "reuse or overwrite"
            )
        atomic_write_bytes(path, expected_payloads[name])


def write_exportable_artifacts(
    destination: str | Path,
    *,
    allocator_output_dir: str | Path,
    cb_col_weights_path: str | Path,
    provenance: Mapping[str, object],
    resume: bool = False,
) -> Path:
    """Publish or exactly resume one identity-bound exportable directory."""
    root = Path(destination)
    resolved_root = root.resolve(strict=False)
    baseline_artifacts = Path(
        "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p7/artifacts"
    ).resolve(strict=False)
    if (
        resolved_root == baseline_artifacts
        or baseline_artifacts in resolved_root.parents
    ):
        raise AnchoredCostError("refusing measured-table baseline overwrite")
    if (root.exists() or root.is_symlink()) and not resume:
        raise AnchoredCostError(
            f"artifact destination exists: {root}; refusing overwrite"
        )
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise AnchoredCostError(
            f"artifact destination is not a real directory: {root}"
        )
    allocator_root = Path(allocator_output_dir)
    try:
        allocator_layer_bytes = (
            allocator_root / "layer_config.json"
        ).read_bytes()
        allocator_selection_bytes = (
            allocator_root / "selection.json"
        ).read_bytes()
        allocator_pareto_bytes = (
            allocator_root / "pareto.knees.json"
        ).read_bytes()
        layer_config = json.loads(allocator_layer_bytes)
        selection = json.loads(allocator_selection_bytes)
        pareto_knees = json.loads(allocator_pareto_bytes)
    except Exception as exc:
        raise AnchoredCostError("allocator outputs are unreadable") from exc
    required = {
        "feasible", "chosen_achieved_bits", "predicted_dloss", "budget_bytes"
    }
    if selection.get("feasible") is not True or not required.issubset(selection):
        raise AnchoredCostError(
            "allocator selection is infeasible or lacks export contract keys"
        )
    primary_knee = (
        pareto_knees.get(pareto_knees.get("primary"))
        if isinstance(pareto_knees, Mapping)
        and isinstance(pareto_knees.get("primary"), str)
        else None
    )
    primary_achieved_bits = (
        primary_knee.get("achieved_bits")
        if isinstance(primary_knee, Mapping) else None
    )
    if (
        isinstance(primary_achieved_bits, bool)
        or not isinstance(primary_achieved_bits, (int, float))
        or not math.isfinite(float(primary_achieved_bits))
    ):
        raise AnchoredCostError(
            "allocator Pareto knees lack a primary achieved-bpp record"
        )
    col_path = Path(cb_col_weights_path)
    if not col_path.is_file():
        raise AnchoredCostError(f"CB render input is missing: {col_path}")
    col_payload = col_path.read_bytes()
    metadata = layer_config.setdefault("__prismaquant__", {})
    if not isinstance(metadata, dict):
        raise AnchoredCostError("layer-config metadata is not an object")
    stamp = dict(canonical_json(
        provenance, where="CB anchored artifact provenance"
    ))
    stamp.update({
        "schema": CB_ANCHORED_COST_SCHEMA,
        "cost_currency": AURA_CURRENCY,
        "segment_key_fields": list(GENERIC_SEGMENT_FIELDS),
        "cb_segment_alias_fields": list(CB_SEGMENT_FIELDS),
        "equivalence_vocabulary_name": "codebook_basis",
        "fisher_application_count": 1,
        "cb_col_weights_role": "production_render_input_only",
        "route_flip_limitation": ROUTE_FLIP_LIMITATION,
    })
    metadata["aura_cb_reprice"] = stamp
    selection["aura_cb_reprice"] = stamp
    selection["cost_currency"] = AURA_CURRENCY
    expected_payloads = {
        "layer_config.json": json.dumps(
            layer_config, indent=2, sort_keys=True, allow_nan=False
        ).encode(),
        "selection.json": json.dumps(
            selection, indent=2, sort_keys=True, allow_nan=False
        ).encode(),
        # Preserve the allocator's own bpp-accounting sidecar next to the
        # AURA-stamped recipe.  The shipcard deliberately reads this number
        # instead of recomputing it under a potentially different convention.
        "pareto.knees.json": allocator_pareto_bytes,
        "cb_col_weights.pkl": col_payload,
    }
    expected_outputs = {
        name: _bytes_descriptor(expected_payloads[name])
        for name in _CB_ARTIFACT_OUTPUT_NAMES
    }
    allocator_stage_identity = {}
    for name in (
        "anchored_allocator_identity.json",
        "anchored_allocator_invocation.json",
    ):
        stage_path = allocator_root / name
        allocator_stage_identity[name] = (
            hashlib.sha256(stage_path.read_bytes()).hexdigest()
            if stage_path.is_file() else None
        )
    identity = dict(canonical_json({
        "schema": CB_ARTIFACT_PUBLISH_SCHEMA,
        "allocator_layer_config_sha256": hashlib.sha256(
            allocator_layer_bytes
        ).hexdigest(),
        "allocator_selection_sha256": hashlib.sha256(
            allocator_selection_bytes
        ).hexdigest(),
        "allocator_pareto_knees_sha256": hashlib.sha256(
            allocator_pareto_bytes
        ).hexdigest(),
        "cb_col_weights_sha256": expected_outputs[
            "cb_col_weights.pkl"
        ]["sha256"],
        "provenance_sha256": canonical_json_sha256(
            provenance, where="CB anchored artifact provenance identity",
        ),
        "allocator_stage_identity_sha256": allocator_stage_identity,
        "outputs": expected_outputs,
    }, where="CB anchored artifact publish identity"))
    identity_sha256 = canonical_json_sha256(
        identity, where="CB anchored artifact publish identity",
    )
    if root.exists():
        manifest_path = root / _CB_ARTIFACT_PUBLISH_MANIFEST
        if not manifest_path.is_file():
            # Only state published without its first atomic identity marker is
            # considered unbound. Preserve it before creating a fresh output.
            _relocate_unbound_artifact_directory(root)
        else:
            complete = _load_artifact_publish_manifest(
                root,
                expected_identity=identity,
                expected_identity_sha256=identity_sha256,
                expected_outputs=expected_outputs,
            )
            allowed = {
                _CB_ARTIFACT_PUBLISH_MANIFEST,
                *_CB_ARTIFACT_OUTPUT_NAMES,
            }
            extras = sorted(path.name for path in root.iterdir()
                            if path.name not in allowed)
            if extras:
                if complete:
                    raise AnchoredCostError(
                        "completed artifact directory has unexpected files: "
                        f"{extras}; refusing reuse"
                    )
                _relocate_unbound_artifact_directory(root)
            else:
                _validate_or_publish_artifact_files(
                    root,
                    expected_payloads,
                    expected_outputs,
                    publish_missing=not complete,
                )
                if complete:
                    return root
                _write_artifact_publish_manifest(
                    root,
                    identity=identity,
                    identity_sha256=identity_sha256,
                    outputs=expected_outputs,
                    complete=True,
                )
                return root
    root.mkdir(parents=True, exist_ok=False)
    _write_artifact_publish_manifest(
        root,
        identity=identity,
        identity_sha256=identity_sha256,
        outputs=expected_outputs,
        complete=False,
    )
    _validate_or_publish_artifact_files(
        root, expected_payloads, expected_outputs,
    )
    _write_artifact_publish_manifest(
        root,
        identity=identity,
        identity_sha256=identity_sha256,
        outputs=expected_outputs,
        complete=True,
    )
    return root


__all__ = [
    "CB_ANCHORED_COST_SCHEMA",
    "CB_ANCHORED_PLUGIN_SCHEMA",
    "CBPanelPolicy",
    "CBUnitDeclaration",
    "CB_SEGMENT_FIELDS",
    "CodebookAnchoredFormatPlugin",
    "LATTICE_BASIS",
    "LEARNED_BASIS",
    "ROUTE_FLIP_LIMITATION",
    "STREAMED_CB_SHARD_MERGE_SCHEMA",
    "anchors_from_streamed_payload",
    "basis_segment_dict",
    "build_cb_allocator_cost_payload",
    "build_cb_extrapolation_input_identity",
    "build_cb_units",
    "build_streamed_cb_render_plan",
    "cb_rung",
    "fit_all_cb_segments",
    "fitted_cb_hull_report",
    "heldout_validation_report",
    "merge_streamed_cb_anchor_aura_shards",
    "observations_from_streamed_payload",
    "plan_cb_panel_and_validation",
    "require_allocator_supersurrogate_support",
    "run_streamed_cb_anchor_aura",
    "write_cb_cost_payload",
    "write_exportable_artifacts",
]
