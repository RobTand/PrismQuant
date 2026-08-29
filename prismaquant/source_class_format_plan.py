"""Exact source-payload format plans for split-menu cost campaigns.

The plan is deliberately independent of routed-expert discovery.  Each
Linear's menu is derived from the source representation recorded by the
checkpoint census and the allocator's existing exact-byte source-rate gate.
The ``expert`` / ``nonexpert`` names are only the campaign CLI's labels for
the two expected source classes; they are never predicates over a qname,
tensor rank, or architecture.

No candidate is silently filtered to make a declared menu fit.  The planner
first derives the complete legal family from the format registry, then
requires that set to equal one of the two declared menus.  A third source
class therefore fails before any render instead of doing illegal work or
truncating a legal candidate set.

``serving_backed_profile`` is the one declared narrowing of "complete
family", and it is a *serving-legality* restriction, never a demand or disk
one.  A caller that names a serving profile restricts the family to the
rungs that profile's fused mid-M lane actually instantiates under the pinned
Gridbook runtime (``serving_profile_specs/*.json`` ->
``fused_mid_m.rungs_by_runtime_version``, resolved through
``serving_profiles.serving_lane_route``).  Nothing about the caller's budget,
cache or preference enters: the set is read off the same immutable pin the
serving lane is declared against, an empty result is an error rather than a
silent truncation, and the resolved rungs are stamped into the plan body so
they land in ``identity_sha256`` and are re-verified on load.  Principle 9 --
a format is production-eligible only when it routes to a performant kernel --
is what licenses it; principle 2 is why it is read from an explicit table
instead of a ``k % 4`` literal here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from prismaquant import format_registry as fr
from prismaquant.allocator_candidates import (
    SOURCE_BPP_EXCEEDED_REASON,
    _source_bpp_applicability,
    source_footprint_owner_for_kind,
)
from prismaquant.allocator_solver import _shape_from_stats
from prismaquant.cost_stage_checkpoint import atomic_write_bytes, canonical_json
from prismaquant.nvfp4_cb_footprint import is_cb_format
from prismaquant.serving_profiles import serving_lane_route


FORMAT_PLAN_SCHEMA = "prismaquant.source_class_format_plan.v1"
FORMAT_PLAN_SELECTION_RULE = (
    "complete registered family filtered only by exact integer source payload; "
    "derived set must equal one declared menu"
)
FORMAT_PLAN_SELECTION_RULE_SERVING_BACKED = (
    "registered family restricted to the fused mid-M rungs the pinned runtime "
    "instantiates, then filtered only by exact integer source payload; "
    "derived set must equal one declared menu"
)
EXPERT_MENU = "expert"
NONEXPERT_MENU = "nonexpert"
_MENU_IDS = (EXPERT_MENU, NONEXPERT_MENU)


@dataclass(frozen=True)
class UnitFormatPlan:
    qname: str
    menu_id: str
    source_kind: str
    shape: tuple[int, ...]
    source_payload_bytes: int
    source_bpp_numerator_bits: int
    bpp_denominator_params: int

    @property
    def source_bpp(self) -> float:
        return (
            float(self.source_bpp_numerator_bits)
            / max(int(self.bpp_denominator_params), 1)
        )


@dataclass(frozen=True)
class SourceClassFormatPlan:
    menus: Mapping[str, tuple[str, ...]]
    units: Mapping[str, UnitFormatPlan]
    serving_groups: tuple[tuple[str, ...], ...]
    identity_sha256: str
    # Present only when the caller declared a serving-backed restriction.
    # Absent keeps the body byte-identical to an unrestricted plan, so the
    # DECLARATION -- not merely its effect -- is part of plan identity.
    serving_backed_restriction: Mapping[str, object] | None = None

    def formats_for(self, qname: str) -> tuple[str, ...]:
        try:
            unit = self.units[str(qname)]
        except KeyError as exc:
            raise KeyError(
                f"format plan has no unit {qname!r}; refusing an unplanned "
                "render"
            ) from exc
        return tuple(self.menus[unit.menu_id])

    def menu_id_for(self, qname: str) -> str:
        try:
            return self.units[str(qname)].menu_id
        except KeyError as exc:
            raise KeyError(
                f"format plan has no unit {qname!r}; refusing an unplanned "
                "render"
            ) from exc

    def formats_by_qname(self) -> dict[str, tuple[str, ...]]:
        return {
            qname: self.formats_for(qname)
            for qname in sorted(self.units)
        }

    def qnames_for_menu(self, menu_id: str) -> tuple[str, ...]:
        if menu_id not in self.menus:
            raise KeyError(f"unknown format-plan menu {menu_id!r}")
        return tuple(
            qname
            for qname, unit in sorted(self.units.items())
            if unit.menu_id == menu_id
        )

    def to_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema": FORMAT_PLAN_SCHEMA,
            "selection_rule": (
                FORMAT_PLAN_SELECTION_RULE
                if self.serving_backed_restriction is None
                else FORMAT_PLAN_SELECTION_RULE_SERVING_BACKED
            ),
            "menus": {
                menu_id: list(self.menus[menu_id]) for menu_id in _MENU_IDS
            },
            "units": {
                qname: {
                    "menu_id": unit.menu_id,
                    "source_kind": unit.source_kind,
                    "shape": list(unit.shape),
                    "source_payload_bytes": unit.source_payload_bytes,
                    "source_bpp_numerator_bits": (
                        unit.source_bpp_numerator_bits
                    ),
                    "bpp_denominator_params": unit.bpp_denominator_params,
                }
                for qname, unit in sorted(self.units.items())
            },
            "serving_groups": [
                list(component) for component in self.serving_groups
            ],
        }
        if self.serving_backed_restriction is not None:
            body["serving_backed_restriction"] = canonical_json(
                dict(self.serving_backed_restriction),
                where="serving-backed restriction",
            )
        body["identity_sha256"] = _plan_digest(body)
        return body


def _plan_digest(body: Mapping[str, object]) -> str:
    digest_body = {
        str(key): value
        for key, value in body.items()
        if str(key) != "identity_sha256"
    }
    canonical = canonical_json(digest_body, where="source-class format plan")
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_format_menu(raw: str | Sequence[str], *, where: str) -> tuple[str, ...]:
    values = raw.split(",") if isinstance(raw, str) else list(raw)
    canonical: list[str] = []
    for value in values:
        token = str(value).strip()
        if not token:
            continue
        try:
            name = fr.get_format(token).name
        except KeyError as exc:
            raise ValueError(f"{where} contains unknown format {token!r}") from exc
        if name in canonical:
            raise ValueError(
                f"{where} contains duplicate canonical format {name!r}"
            )
        canonical.append(name)
    if not canonical:
        raise ValueError(f"{where} is empty")
    return tuple(canonical)


def _serving_backed_family(
    family: str,
    registered: tuple[str, ...],
    profile_id: str,
) -> tuple[tuple[str, ...], dict[str, object]]:
    """The registered family restricted to fused-mid-M-backed rungs.

    Resolution goes through ``serving_lane_route`` -- the same entry point
    ``allocator_candidates`` already attaches routes with -- rather than
    reading ``serving_lanes[i]`` out of the spec JSON, so there is exactly one
    mechanism deciding what the pinned runtime backs and no lane index to get
    wrong.  Fail-closed throughout: an unroutable format, a disagreeing lane
    set and an empty backed set are all errors, because every one of them
    would otherwise silently shrink a candidate set.
    """
    backed: list[str] = []
    dropped: list[str] = []
    lanes = []
    for name in registered:
        lane = serving_lane_route(profile_id, name)
        if lane is None:
            raise ValueError(
                f"serving profile {profile_id!r} declares no lane for "
                f"{name!r}; a serving-backed restriction cannot be resolved "
                "for the whole family"
            )
        lanes.append(lane)
        (backed if lane.fused_mid_m_backed else dropped).append(name)
    lane_ids = sorted({lane.lane_id for lane in lanes})
    sources = sorted({lane.rungs_source for lane in lanes})
    versions = sorted({lane.runtime_version for lane in lanes})
    if len(lane_ids) != 1 or len(sources) != 1 or len(versions) != 1:
        raise ValueError(
            f"family {family!r} resolves to more than one serving lane under "
            f"{profile_id!r} (lanes={lane_ids} sources={sources} "
            f"runtimes={versions}); a serving-backed restriction must be one "
            "declaration"
        )
    if not backed:
        raise ValueError(
            f"serving profile {profile_id!r} backs no fused mid-M rung of "
            f"family {family!r} at pinned runtime {versions[0]!r} "
            f"(source={sources[0]!r}). Refusing an empty menu: declare no "
            "restriction, or advance the pin together with its backed-set key."
        )
    rungs = sorted({int(lane.rung) for lane in lanes
                    if lane.fused_mid_m_backed and lane.rung is not None})
    provenance: dict[str, object] = {
        "profile_id": str(profile_id),
        "family": str(family),
        "lane_id": lane_ids[0],
        "runtime_version": versions[0],
        "rungs_source": sources[0],
        "fused_mid_m_rungs": rungs,
        "backed_formats": list(backed),
        "restricted_out": list(dropped),
    }
    return tuple(backed), provenance


def _complete_family(
    expert_formats: tuple[str, ...],
    nonexpert_formats: tuple[str, ...],
    *,
    serving_backed_profile: str | None = None,
) -> tuple[tuple[str, ...], dict[str, object] | None]:
    declared = tuple(dict.fromkeys((*expert_formats, *nonexpert_formats)))
    families = {fr.get_format(name).family for name in declared}
    if len(families) != 1:
        raise ValueError(
            "source-class menus must describe one registered format family; "
            f"got {sorted(families)!r}"
        )
    family = next(iter(families))
    registered = tuple(spec.name for spec in fr.list_producer_formats(family))
    restriction: dict[str, object] | None = None
    if serving_backed_profile is not None:
        registered, restriction = _serving_backed_family(
            family, registered, serving_backed_profile
        )
    registered_set = set(registered)
    declared_set = set(declared)
    if declared_set != registered_set:
        missing = sorted(registered_set - declared_set)
        extra = sorted(declared_set - registered_set)
        scope = "the complete registered family"
        if restriction is not None:
            scope = (
                "the complete registered family backed by the pinned serving "
                f"runtime (profile={restriction['profile_id']!r} "
                f"runtime={restriction['runtime_version']!r} "
                f"rungs={restriction['fused_mid_m_rungs']})"
            )
        raise ValueError(
            f"source-class menus must cover {scope}; "
            f"family={family!r} missing={missing} extra={extra}. Refusing "
            "demand- or disk-driven candidate truncation."
        )
    # The registry sorts by effective rate.  Use it as the canonical ordering
    # so plan identity cannot depend on how two equivalent CLI strings happen
    # to interleave their shared formats.
    return registered, restriction


def _validate_declared_menus(
    expert_formats: tuple[str, ...],
    nonexpert_formats: tuple[str, ...],
    family_formats: tuple[str, ...],
) -> None:
    expert_set = set(expert_formats)
    nonexpert_set = set(nonexpert_formats)
    if not expert_set < nonexpert_set:
        raise ValueError(
            "expert-formats must be a strict subset of nonexpert-formats; "
            "the split represents lower- and higher-source-rate classes"
        )
    if nonexpert_set != set(family_formats):
        missing = sorted(set(family_formats) - nonexpert_set)
        extra = sorted(nonexpert_set - set(family_formats))
        raise ValueError(
            "nonexpert-formats must retain the complete registered family; "
            f"missing={missing} extra={extra}. Refusing truncation."
        )
    expected_expert_order = tuple(
        name for name in family_formats if name in expert_set
    )
    expected_nonexpert_order = tuple(
        name for name in family_formats if name in nonexpert_set
    )
    if expert_formats != expected_expert_order:
        raise ValueError(
            "expert-formats must follow the registry's increasing-rate order; "
            f"expected={list(expected_expert_order)}"
        )
    if nonexpert_formats != expected_nonexpert_order:
        raise ValueError(
            "nonexpert-formats must follow the registry's increasing-rate "
            f"order; expected={list(expected_nonexpert_order)}"
        )


def _serving_components(profile, qnames: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Union fused and packed groups exactly as serving promotion does."""
    names = tuple(sorted(dict.fromkeys(str(name) for name in qnames)))
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    groups: dict[tuple[str, str], list[str]] = {}
    for qname in names:
        for kind, accessor in (
            ("fused", "fused_sibling_group"),
            ("packed", "packed_expert_format_group"),
        ):
            method = getattr(profile, accessor, None)
            if not callable(method):
                raise RuntimeError(
                    f"profile {type(profile).__name__} is missing callable "
                    f"{accessor}(); source-class planning cannot validate "
                    "serving-atomic groups"
                )
            try:
                key = method(qname)
            except Exception as exc:
                raise RuntimeError(
                    f"profile {type(profile).__name__}.{accessor}({qname!r}) "
                    "failed during source-class planning"
                ) from exc
            if key is not None:
                groups.setdefault((kind, str(key)), []).append(qname)

    for members in groups.values():
        if len(members) < 2:
            continue
        first = members[0]
        for member in members[1:]:
            union(first, member)
    components: dict[str, list[str]] = {}
    for qname in names:
        components.setdefault(find(qname), []).append(qname)
    return tuple(
        tuple(sorted(members))
        for members in sorted(components.values(), key=lambda item: min(item))
        if len(members) > 1
    )


def build_source_class_format_plan(
    stats: Mapping[str, Mapping[str, object]],
    source_manifest: Mapping[str, str],
    profile,
    *,
    expert_formats: str | Sequence[str],
    nonexpert_formats: str | Sequence[str],
    cb_serialization_context,
    serving_backed_profile: str | None = None,
) -> SourceClassFormatPlan:
    """Derive one exact full-family menu per source-payload class.

    ``stats`` must carry exact Linear shapes and ``source_manifest`` must be a
    complete checkpoint census in the allocator's recipe namespace.  The
    existing allocator source-rate predicate is called for every registered
    family member; this module intentionally contains no bpp formula.

    ``serving_backed_profile`` names a serving profile whose fused mid-M lane
    bounds the family under the pinned Gridbook runtime (see the module
    docstring).  ``None`` -- the default -- keeps the unrestricted contract and
    an unchanged plan body.
    """
    expert_menu = parse_format_menu(expert_formats, where="expert-formats")
    nonexpert_menu = parse_format_menu(
        nonexpert_formats, where="nonexpert-formats"
    )
    family_formats, restriction = _complete_family(
        expert_menu,
        nonexpert_menu,
        serving_backed_profile=serving_backed_profile,
    )
    _validate_declared_menus(expert_menu, nonexpert_menu, family_formats)
    if any(is_cb_format(name) for name in family_formats):
        if cb_serialization_context is None:
            raise ValueError(
                "source-class CB planning requires an exact "
                "CBSerializationContext; the allocator deliberately defers "
                "CB source-rate checks without it"
            )

    units: dict[str, UnitFormatPlan] = {}
    for qname in sorted(str(name) for name in stats):
        row = stats[qname]
        if not isinstance(row, Mapping):
            raise ValueError(f"probe stats row {qname!r} is not an object")
        shape = tuple(int(dim) for dim in _shape_from_stats(dict(row)))
        if len(shape) < 2 or any(dim <= 0 for dim in shape):
            raise ValueError(
                f"{qname}: source-class planning needs an exact rank>=2 "
                f"Linear shape, got {shape}"
            )
        if qname not in source_manifest:
            raise ValueError(
                f"{qname}: source census is missing this planned unit; "
                "refusing an unverified source-rate class"
            )
        source_kind = str(source_manifest[qname])
        if source_footprint_owner_for_kind(source_kind) is None:
            raise ValueError(
                f"{qname}: source_kind={source_kind!r} has no exact source "
                "footprint owner; refusing source-class planning"
            )

        verdicts = {
            format_name: _source_bpp_applicability(
                shape,
                fr.get_format(format_name),
                qname=qname,
                source_kind=source_kind,
                cb_serialization_context=cb_serialization_context,
            )
            for format_name in family_formats
        }
        unexpected_reasons = sorted({
            verdict.reason
            for verdict in verdicts.values()
            if not verdict.legal
            and verdict.reason != SOURCE_BPP_EXCEEDED_REASON
        })
        if unexpected_reasons:
            sample = next(
                verdict.detail
                for verdict in verdicts.values()
                if not verdict.legal
                and verdict.reason != SOURCE_BPP_EXCEEDED_REASON
            )
            raise ValueError(
                f"{qname}: exact source-rate derivation failed with "
                f"{unexpected_reasons}: {sample}"
            )
        legal_formats = tuple(
            name for name in family_formats if verdicts[name].legal
        )
        if legal_formats == expert_menu:
            menu_id = EXPERT_MENU
        elif legal_formats == nonexpert_menu:
            menu_id = NONEXPERT_MENU
        else:
            raise ValueError(
                f"{qname}: source_kind={source_kind!r} derives legal family "
                f"{list(legal_formats)}, which matches neither declared menu. "
                "Refusing both illegal work and candidate truncation; declare "
                "the missing source class explicitly."
            )

        provenance = next(
            (
                verdict.provenance
                for verdict in verdicts.values()
                if isinstance(verdict.provenance, Mapping)
                and verdict.provenance.get("source_payload_bytes") is not None
            ),
            None,
        )
        if provenance is None:
            raise RuntimeError(
                f"{qname}: source-rate gate returned no exact payload "
                "provenance"
            )
        n_params = int(provenance["bpp_denominator_params"])
        if n_params != math.prod(shape):
            raise RuntimeError(
                f"{qname}: source-rate provenance parameter denominator "
                f"{n_params} disagrees with shape {shape}"
            )
        units[qname] = UnitFormatPlan(
            qname=qname,
            menu_id=menu_id,
            source_kind=source_kind,
            shape=shape,
            source_payload_bytes=int(provenance["source_payload_bytes"]),
            source_bpp_numerator_bits=int(
                provenance["source_bpp_numerator_bits"]
            ),
            bpp_denominator_params=n_params,
        )

    if not units:
        raise ValueError("source-class format plan has no units")
    unexpected_source_names = sorted(set(source_manifest) - set(units))
    # A source census normally includes only quantizable recipe qnames, but it
    # may legitimately carry pinned/passthrough entries.  They are not planned
    # and are intentionally ignored; completeness is checked in the other
    # direction above for every unit that will be rendered.
    del unexpected_source_names

    serving_groups = _serving_components(profile, tuple(units))
    for members in serving_groups:
        menu_ids = {units[qname].menu_id for qname in members}
        if len(menu_ids) != 1:
            detail = {
                qname: {
                    "menu_id": units[qname].menu_id,
                    "source_kind": units[qname].source_kind,
                }
                for qname in members
            }
            raise ValueError(
                "source-class format plan would split one fused/packed "
                f"serving component across menus: {detail}. Refusing to "
                "intersect or truncate the group."
            )

    provisional = SourceClassFormatPlan(
        menus={
            EXPERT_MENU: expert_menu,
            NONEXPERT_MENU: nonexpert_menu,
        },
        units=units,
        serving_groups=serving_groups,
        identity_sha256="",
        serving_backed_restriction=restriction,
    )
    body = provisional.to_dict()
    return SourceClassFormatPlan(
        menus=provisional.menus,
        units=provisional.units,
        serving_groups=provisional.serving_groups,
        identity_sha256=str(body["identity_sha256"]),
        serving_backed_restriction=provisional.serving_backed_restriction,
    )


def write_format_plan(plan: SourceClassFormatPlan, path: str | Path) -> None:
    body = plan.to_dict()
    expected = str(body["identity_sha256"])
    if plan.identity_sha256 and plan.identity_sha256 != expected:
        raise ValueError(
            "source-class format plan identity changed before publication: "
            f"stored={plan.identity_sha256} current={expected}"
        )
    encoded = json.dumps(
        body,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    atomic_write_bytes(Path(path), encoded)


def load_format_plan(
    path: str | Path,
    *,
    verify_current_serving_restriction: bool = True,
) -> SourceClassFormatPlan:
    plan_path = Path(path)
    try:
        raw = json.loads(plan_path.read_text())
    except Exception as exc:
        raise ValueError(f"format plan {plan_path} is unreadable") from exc
    if not isinstance(raw, Mapping):
        raise ValueError(f"format plan {plan_path} is not an object")
    if raw.get("schema") != FORMAT_PLAN_SCHEMA:
        raise ValueError(
            f"format plan {plan_path} has unsupported schema "
            f"{raw.get('schema')!r}"
        )
    expected_digest = _plan_digest(raw)
    if raw.get("identity_sha256") != expected_digest:
        raise ValueError(
            f"format plan {plan_path} identity mismatch: stored="
            f"{raw.get('identity_sha256')!r} current={expected_digest!r}"
        )
    raw_menus = raw.get("menus")
    if not isinstance(raw_menus, Mapping):
        raise ValueError(f"format plan {plan_path} has no menus object")
    menus = {
        menu_id: parse_format_menu(
            raw_menus.get(menu_id, ()), where=f"format plan {menu_id} menu"
        )
        for menu_id in _MENU_IDS
    }
    # A restricted plan carries its own restriction, so the load path
    # re-derives it from the CURRENT pin and refuses on drift: a plan written
    # under one Gridbook backed set must not be silently reused under another.
    stored_restriction = raw.get("serving_backed_restriction")
    if stored_restriction is not None and not isinstance(
        stored_restriction, Mapping
    ):
        raise ValueError(
            f"format plan {plan_path} has a non-object "
            "serving_backed_restriction"
        )
    if verify_current_serving_restriction:
        family_formats, restriction = _complete_family(
            menus[EXPERT_MENU],
            menus[NONEXPERT_MENU],
            serving_backed_profile=(
                None if stored_restriction is None
                else str(stored_restriction.get("profile_id", ""))
            ),
        )
        if restriction is not None and canonical_json(
            dict(restriction), where="serving-backed restriction"
        ) != canonical_json(
            dict(stored_restriction), where="stored serving-backed restriction"
        ):
            raise ValueError(
                f"format plan {plan_path} was written under a different "
                "serving-backed restriction than the current pin resolves: "
                f"stored={dict(stored_restriction)} current={dict(restriction)}"
            )
    else:
        # Explicit migration callers may need to verify an immutable historical
        # plan after the consumer pin advances. This bypasses only the CURRENT
        # pin comparison: the stored digest, closed menus, units, source bytes,
        # and serving groups are still parsed and checked below. The caller
        # must separately prove the allowed semantic delta to a fresh plan.
        family_formats = tuple(menus[NONEXPERT_MENU])
        restriction = (
            None
            if stored_restriction is None
            else dict(stored_restriction)
        )
    _validate_declared_menus(
        menus[EXPERT_MENU], menus[NONEXPERT_MENU], family_formats
    )

    raw_units = raw.get("units")
    if not isinstance(raw_units, Mapping) or not raw_units:
        raise ValueError(f"format plan {plan_path} has no units")
    units: dict[str, UnitFormatPlan] = {}
    for raw_qname, value in raw_units.items():
        qname = str(raw_qname)
        if not isinstance(value, Mapping):
            raise ValueError(f"format plan unit {qname!r} is not an object")
        menu_id = str(value.get("menu_id", ""))
        if menu_id not in menus:
            raise ValueError(
                f"format plan unit {qname!r} has unknown menu {menu_id!r}"
            )
        shape = tuple(int(dim) for dim in value.get("shape", ()))
        if len(shape) < 2 or any(dim <= 0 for dim in shape):
            raise ValueError(
                f"format plan unit {qname!r} has invalid shape {shape}"
            )
        denominator = int(value.get("bpp_denominator_params", 0))
        if denominator != math.prod(shape):
            raise ValueError(
                f"format plan unit {qname!r} has a parameter denominator "
                f"that disagrees with shape {shape}"
            )
        source_payload_bytes = int(value.get("source_payload_bytes", 0))
        source_bits = int(value.get("source_bpp_numerator_bits", 0))
        if source_payload_bytes <= 0 or source_bits != 8 * source_payload_bytes:
            raise ValueError(
                f"format plan unit {qname!r} has inconsistent source bytes"
            )
        units[qname] = UnitFormatPlan(
            qname=qname,
            menu_id=menu_id,
            source_kind=str(value.get("source_kind", "")),
            shape=shape,
            source_payload_bytes=source_payload_bytes,
            source_bpp_numerator_bits=source_bits,
            bpp_denominator_params=denominator,
        )

    raw_groups = raw.get("serving_groups", ())
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
        raise ValueError(f"format plan {plan_path} serving_groups is invalid")
    serving_groups: list[tuple[str, ...]] = []
    for index, value in enumerate(raw_groups):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(
                f"format plan serving group {index} is not a sequence"
            )
        members = tuple(str(name) for name in value)
        if len(members) < 2 or len(set(members)) != len(members):
            raise ValueError(
                f"format plan serving group {index} is malformed: {members}"
            )
        missing = sorted(set(members) - set(units))
        if missing:
            raise ValueError(
                f"format plan serving group {index} names missing units "
                f"{missing}"
            )
        menu_ids = {units[name].menu_id for name in members}
        if len(menu_ids) != 1:
            raise ValueError(
                f"format plan serving group {index} straddles menus "
                f"{sorted(menu_ids)}"
            )
        serving_groups.append(members)

    return SourceClassFormatPlan(
        menus=menus,
        units=units,
        serving_groups=tuple(serving_groups),
        identity_sha256=expected_digest,
        serving_backed_restriction=restriction,
    )


__all__ = [
    "EXPERT_MENU",
    "FORMAT_PLAN_SCHEMA",
    "NONEXPERT_MENU",
    "SourceClassFormatPlan",
    "UnitFormatPlan",
    "build_source_class_format_plan",
    "load_format_plan",
    "parse_format_menu",
    "write_format_plan",
]
