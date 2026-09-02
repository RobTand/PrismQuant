"""Exact no-codebook byte-budget feasibility certificates.

The certificate proves a lower bound, not a heuristic comparison: each
serving-atomic body unit is assigned its cheapest format that is legal in the
requested production profile after source-kind, shape, and container checks.
Source-only construction units (currently the released DSpark overlay) are
accounted from their exact safetensors spans.  Everything else remains in an
immutable source-payload floor.  Consequently, if the sum is over budget then
*every* codebook-free production assignment is over budget.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import pickle
import re
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from .allocator_candidates import (
    SOURCE_PASSTHROUGH_FORMATS,
    _scan_source_dtype_manifest,
    check_format_applicability,
)
from .allocator_solver import _shape_from_stats
from .artifact_completeness import read_artifact_header
from .cost_streaming import (
    _read_streamed_model_identity_cache,
    build_streamed_model_identity,
    validate_cached_streamed_model_identity,
)
from .dspark_source_metadata import discover_dspark_source_overlay_from_artifact
from .footprint import (
    format_tensor_payload_breakdown,
    resolve_reencoded_source_bytes,
    source_checkpoint_bytes,
    source_tensor_bytes_manifest,
    nvfp4_global_sidecar_bytes,
)
from .format_registry import canonical_format_name, list_producer_formats
from .gridbook_runtime_pin import load_gridbook_runtime_pin
from .lane_spec import load_lane_spec
from .model_profiles import detect_profile
from .nvfp4_cb_footprint import is_cb_format
from .serving_profiles import check_serving_format
from .source_class_format_plan import _serving_components


SCHEMA = "prismaquant.native_baseline_feasibility.v1"
PROBE_CENSUS_SCHEMA = "prismaquant.expected_probe_census.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PACKAGE_ROOT = Path(__file__).resolve().parent
_CANDIDATE_SELECTION = (
    "all registered non-CB formats legal under the exact production serving "
    "profile, source-kind ceiling, and shape rules; source-only construction "
    "units use their validated released overlay format"
)


@dataclass(frozen=True)
class NativeConstructionUnit:
    """One source-only unit constructed from one or more physical targets."""

    name: str
    format_name: str
    payload_bytes: int
    physical_targets: tuple[str, ...]
    source_span_ids: tuple[str, ...]
    physical_target_payload_bytes: tuple[int, ...]


@dataclass(frozen=True)
class ExpectedProbeCensus:
    """External authority for the complete quantizable probe inventory."""

    contract_id: str
    profile_id: str
    total_members: int
    routed_role_counts: Mapping[str, int]
    nonexpert_role_counts: Mapping[str, int]
    member_names_sha256: str
    authority_sha256: str


def _positive_int(value: object, *, where: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where} must be an exact integer, got {value!r}")
    if value < 0 or (value == 0 and not allow_zero):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{where} must be {relation}, got {value}")
    return value


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON representation."""

    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def certificate_sha256(certificate: Mapping[str, Any]) -> str:
    body = dict(certificate)
    body.pop("certificate_sha256", None)
    return canonical_sha256(body)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(
    value: object,
    expected: Iterable[str],
    *,
    where: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be an object")
    wanted = set(expected)
    observed = set(value)
    if observed != wanted:
        raise ValueError(
            f"{where} members differ: missing={sorted(wanted - observed)}, "
            f"extra={sorted(observed - wanted)}"
        )
    return value


def _require_string(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _require_sha256(value: object, *, where: str) -> str:
    text = _require_string(value, where=where)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{where} must be lowercase 64-hex")
    return text


def _require_sorted_unique_strings(value: object, *, where: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty JSON list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{where} must contain non-empty strings")
    if value != sorted(set(value)):
        raise ValueError(f"{where} must be sorted and duplicate-free")
    return value


def _role_counts(value: object, *, where: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{where} must be a non-empty role-count object")
    out: dict[str, int] = {}
    for raw_role, raw_count in value.items():
        role = _require_string(raw_role, where=f"{where} role")
        out[role] = _positive_int(raw_count, where=f"{where}.{role}")
    if list(value) != sorted(out):
        raise ValueError(f"{where} roles must be sorted")
    return out


def evaluate_expected_probe_census(
    qnames: Iterable[str],
    profile: object,
    contract: ExpectedProbeCensus,
) -> dict[str, Any]:
    """Classify and compare every probe member against an external census.

    The profile classifier is the membership authority; its declared direct
    per-expert regex is also embedded so a certificate consumer can reconstruct
    the same routed/nonexpert role counts from the unit ledger alone.
    """

    from .routed_experts import ProfileRoutedExpertClassifier

    names = tuple(sorted(str(name) for name in qnames))
    if len(names) != len(set(names)):
        raise ValueError("probe census contains duplicate qnames")
    profile_id = str(getattr(profile, "name", ""))
    if profile_id != contract.profile_id:
        raise ValueError(
            f"probe census profile differs: {profile_id!r} vs "
            f"{contract.profile_id!r}"
        )
    _require_sha256(contract.authority_sha256, where="probe census authority_sha256")
    expected_member_digest = _require_sha256(
        contract.member_names_sha256,
        where="probe census member_names_sha256",
    )
    expected_routed = dict(sorted(contract.routed_role_counts.items()))
    expected_nonexpert = dict(sorted(contract.nonexpert_role_counts.items()))
    for label, counts in (
        ("routed", expected_routed),
        ("nonexpert", expected_nonexpert),
    ):
        _role_counts(counts, where=f"expected {label} role counts")
    expected_total = _positive_int(
        contract.total_members, where="expected probe member count"
    )
    if sum(expected_routed.values()) + sum(expected_nonexpert.values()) != expected_total:
        raise ValueError("expected probe role counts do not sum to total_members")

    classifier = ProfileRoutedExpertClassifier(profile)
    pattern = classifier.per_expert_pattern
    if pattern is None:
        raise ValueError(
            f"profile {profile_id!r} has no direct per-expert census regex"
        )
    routed: Counter[str] = Counter()
    nonexpert: Counter[str] = Counter()
    for qname in names:
        classified = classifier.classify(qname)
        direct = pattern.fullmatch(qname) is not None
        if (classified is not None) != direct:
            raise ValueError(
                f"{qname}: profile routed classification cannot be reconstructed "
                "from its direct per-expert census regex"
            )
        if classified is not None:
            routed[classified.projection_name] += 1
        else:
            nonexpert[qname.rsplit(".", 1)[-1]] += 1
    observed_routed = dict(sorted(routed.items()))
    observed_nonexpert = dict(sorted(nonexpert.items()))
    if (
        len(names) != expected_total
        or observed_routed != expected_routed
        or observed_nonexpert != expected_nonexpert
        or canonical_sha256({"members": list(names)}) != expected_member_digest
    ):
        raise ValueError(
            "probe census is incomplete or role-shifted: "
            f"members={len(names)}/{expected_total}, "
            f"routed={observed_routed}/{expected_routed}, "
            f"nonexpert={observed_nonexpert}/{expected_nonexpert}"
        )
    counts = {
        "body_member_count": len(names),
        "routed_member_count": sum(observed_routed.values()),
        "nonexpert_member_count": sum(observed_nonexpert.values()),
        "routed_role_counts": observed_routed,
        "nonexpert_role_counts": observed_nonexpert,
    }
    return {
        "schema": PROBE_CENSUS_SCHEMA,
        "contract_id": _require_string(
            contract.contract_id, where="probe census contract_id"
        ),
        "authority_sha256": contract.authority_sha256,
        "profile_id": profile_id,
        "classifier": "profile_declared_routed_expert.v1",
        "routed_qname_regex": pattern.pattern,
        "expected_member_names_sha256": expected_member_digest,
        "observed_member_names_sha256": canonical_sha256(
            {"members": list(names)}
        ),
        "expected": counts,
        "observed": dict(counts),
        "complete": True,
    }


def _release_probe_census_contract(profile: object) -> ExpectedProbeCensus:
    """Return the repository-owned exact census for a release profile."""

    profile_id = str(getattr(profile, "name", ""))
    if profile_id != "deepseek_v4":
        raise ValueError(
            f"no release probe-census authority is registered for {profile_id!r}"
        )
    # These are the DSv4 campaign's executable release invariants.  Importing
    # them avoids a second dated table here; the owner file hash is carried in
    # the certificate so a later edit cannot silently change the authority.
    from . import dsv4_aura_cb_reprice as campaign

    total = int(campaign.DSV4_TOTAL_UNITS)
    routed_total = int(campaign.DSV4_EXPERT_UNITS)
    nonexpert_total = int(campaign.DSV4_NONEXPERT_UNITS)
    combined = dict(campaign._DSV4_ROLE_UNIT_COUNTS)
    nonexpert = dict(campaign._DSV4_NONEXPERT_ROLE_UNIT_COUNTS)
    routed = {
        role: int(count) - int(nonexpert.get(role, 0))
        for role, count in combined.items()
        if int(count) - int(nonexpert.get(role, 0)) > 0
    }
    if (
        sum(routed.values()) != routed_total
        or sum(nonexpert.values()) != nonexpert_total
        or routed_total + nonexpert_total != total
    ):
        raise ValueError("DSv4 campaign census constants are internally inconsistent")
    layer_count = next(iter(nonexpert.values()))
    if any(count != layer_count for count in nonexpert.values()):
        raise ValueError("DSv4 nonexpert role counts do not imply one layer count")
    routed_projection_counts = set(routed.values())
    if len(routed_projection_counts) != 1:
        raise ValueError("DSv4 routed roles do not imply one expert count")
    routed_per_role = next(iter(routed_projection_counts))
    if routed_per_role % layer_count:
        raise ValueError("DSv4 routed role count is not layer-divisible")
    expert_count = routed_per_role // layer_count
    expected_names = {
        *(
            f"model.layers.{layer}.mlp.experts.{expert}.{role}"
            for layer in range(layer_count)
            for expert in range(expert_count)
            for role in sorted(routed)
        ),
        *(
            (
                f"model.layers.{layer}.mlp.shared_experts.{role}"
                if role in {"gate_proj", "up_proj", "down_proj"}
                else f"model.layers.{layer}.self_attn.{role}"
            )
            for layer in range(layer_count)
            for role in sorted(nonexpert)
        ),
    }
    if len(expected_names) != total:
        raise ValueError("DSv4 exact expected qname set differs from campaign total")
    return ExpectedProbeCensus(
        contract_id="prismaquant.dsv4_aura_cb_reprice.census.v1",
        profile_id=profile_id,
        total_members=total,
        routed_role_counts=dict(sorted(routed.items())),
        nonexpert_role_counts=dict(sorted(nonexpert.items())),
        member_names_sha256=canonical_sha256(
            {"members": sorted(expected_names)}
        ),
        authority_sha256=_file_sha256(Path(inspect.getfile(campaign)).resolve()),
    )


def _span_map_for_names(
    manifest: Mapping[str, int], names: Iterable[str]
) -> dict[str, tuple[str, ...]]:
    spans = getattr(manifest, "spans", None)
    if not isinstance(spans, Mapping):
        raise ValueError("production source-byte manifest has no span provenance")
    out: dict[str, tuple[str, ...]] = {}
    for name in names:
        key = name if name in manifest else name.removesuffix(".weight")
        values = tuple(sorted(str(value) for value in spans.get(key, ())))
        if not values:
            raise ValueError(f"{name}: source-byte manifest has no physical spans")
        out[name] = values
    return out


def _claim_spans(
    span_ids_by_name: Mapping[str, Sequence[str]],
    construction_units: Sequence[NativeConstructionUnit],
) -> None:
    claimed: dict[str, str] = {}
    for name in sorted(span_ids_by_name):
        spans = span_ids_by_name[name]
        if not spans:
            raise ValueError(f"{name}: empty source-span coverage")
        for span in spans:
            first = claimed.setdefault(str(span), name)
            if first != name:
                raise ValueError(
                    f"source span {span!r} is charged by both {first!r} and {name!r}"
                )
    for unit in sorted(construction_units, key=lambda item: item.name):
        if not unit.source_span_ids:
            raise ValueError(f"{unit.name}: empty construction source-span coverage")
        for span in unit.source_span_ids:
            owner = f"construction:{unit.name}"
            first = claimed.setdefault(str(span), owner)
            if first != owner:
                raise ValueError(
                    f"source span {span!r} is charged by both {first!r} and {owner!r}"
                )


def _unit_components(profile: object, qnames: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    grouped = _serving_components(profile, qnames)
    members = {name for component in grouped for name in component}
    singles = ((name,) for name in sorted(qnames) if name not in members)
    components = tuple(grouped) + tuple(singles)
    flattened = [name for component in components for name in component]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(qnames):
        raise ValueError("serving-unit partition is incomplete or overlapping")
    return tuple(sorted(components, key=lambda item: item[0]))


def _component_is_packed(profile: object, component: Sequence[str]) -> bool:
    accessor = getattr(profile, "packed_expert_format_group", None)
    if not callable(accessor):
        raise ValueError("model profile has no packed_expert_format_group()")
    return any(accessor(name) is not None for name in component)


def _legal_no_cb_options(
    *,
    members: Sequence[str],
    member_evidence: Sequence[Mapping[str, Any]],
    packed: bool,
    target_profile: str,
) -> list[dict[str, Any]]:
    if len(members) != len(member_evidence) or not members:
        raise ValueError("serving members/evidence coverage differs")
    available: list[dict[str, Any]] = []
    for spec in list_producer_formats():
        fmt = canonical_format_name(spec.name)
        if is_cb_format(fmt):
            continue
        payload_bytes = 0
        legal = True
        for qname, evidence in zip(members, member_evidence, strict=True):
            shape = tuple(int(value) for value in evidence["shape"])
            source_kind = str(evidence["source_kind"])
            source_payload_bytes = int(evidence["source_payload_bytes"])
            decision = check_format_applicability(
                shape,
                spec,
                qname=qname,
                source_kind=source_kind,
                target_profile=target_profile,
            )
            if not decision.legal:
                legal = False
                break
            if fmt in SOURCE_PASSTHROUGH_FORMATS:
                closed_form = int(spec.memory_bytes_for_shape(shape))
                if closed_form != source_payload_bytes:
                    raise ValueError(
                        f"{qname}: {fmt} closed-form bytes {closed_form} disagree "
                        f"with exact source span bytes {source_payload_bytes}"
                    )
                payload_bytes += source_payload_bytes
            else:
                payload_bytes += int(
                    format_tensor_payload_breakdown(spec, shape, qname=qname)[
                        "tensor_payload_bytes"
                    ]
                )
                if fmt == "NVFP4":
                    payload_bytes += nvfp4_global_sidecar_bytes(
                        qname,
                        shape,
                        weight_only=bool(evidence["nvfp4_weight_only"]),
                    )
        if legal and packed:
            group_decision = check_serving_format(
                target_profile, members[0], fmt, packed_expert=True
            )
            legal = bool(group_decision.legal)
        if legal:
            available.append({"format": fmt, "payload_bytes": payload_bytes})
    available.sort(key=lambda row: (row["payload_bytes"], row["format"]))
    return available


def _body_unit(
    *,
    component: tuple[str, ...],
    stats: Mapping[str, Mapping[str, object]],
    source_kinds: Mapping[str, str],
    source_bytes: Mapping[str, int],
    source_span_ids: Mapping[str, Sequence[str]],
    profile: object,
    target_profile: str,
) -> dict[str, Any]:
    packed = _component_is_packed(profile, component)
    member_evidence = [
        {
            "shape": list(_shape_from_stats(dict(stats[qname]))),
            "source_kind": source_kinds[qname],
            "source_payload_bytes": int(source_bytes[qname]),
            "source_span_ids": sorted(str(span) for span in source_span_ids[qname]),
            "nvfp4_weight_only": bool(
                stats[qname].get("_nvfp4_weight_only", False)
            ),
        }
        for qname in component
    ]
    available = _legal_no_cb_options(
        members=component,
        member_evidence=member_evidence,
        packed=packed,
        target_profile=target_profile,
    )
    if not available:
        raise ValueError(
            f"serving unit {component[0]!r} has no legal no-codebook production format"
        )
    available.sort(key=lambda row: (row["payload_bytes"], row["format"]))
    best = available[0]
    return {
        "kind": "serving",
        "name": component[0] if len(component) == 1 else f"group:{component[0]}",
        "serving_atomic": len(component) > 1,
        "packed_expert": packed,
        "members": list(component),
        "member_evidence": member_evidence,
        "source_kinds": sorted({source_kinds[name] for name in component}),
        "source_payload_bytes": sum(source_bytes[name] for name in component),
        "source_span_ids": sorted(
            {span for name in component for span in source_span_ids[name]}
        ),
        "available_no_cb_formats": available,
        "lower_bound_format": best["format"],
        "lower_bound_payload_bytes": best["payload_bytes"],
    }


def build_native_baseline_certificate(
    *,
    stats: Mapping[str, Mapping[str, object]],
    source_kinds: Mapping[str, str],
    source_bytes: Mapping[str, int],
    source_span_ids: Mapping[str, Sequence[str]],
    model_profile: object,
    target_profile: str,
    lane_id: str,
    budget_bytes: int,
    source_checkpoint_payload_bytes: int,
    source_model_binding: Mapping[str, Any],
    contract_binding: Mapping[str, Any],
    probe_census: Mapping[str, Any],
    construction_units: Sequence[NativeConstructionUnit] = (),
) -> dict[str, Any]:
    """Build a deterministic exact-byte lower-bound certificate.

    The inputs are deliberately injectable so unit tests and future model
    families can exercise the proof without a real multi-shard checkpoint.
    Filesystem discovery and identity validation live in
    :func:`certify_native_baseline_from_model`.
    """

    budget = _positive_int(budget_bytes, where="budget_bytes")
    source_total = _positive_int(
        source_checkpoint_payload_bytes, where="source_checkpoint_payload_bytes"
    )
    qnames = tuple(sorted(str(name) for name in stats))
    if not qnames:
        raise ValueError("stats must enumerate at least one quantizable Linear")
    _validate_probe_census_binding(probe_census, qnames)
    expected = set(qnames)
    for label, table in (
        ("source_kinds", source_kinds),
        ("source_bytes", source_bytes),
        ("source_span_ids", source_span_ids),
    ):
        if set(table) != expected:
            raise ValueError(
                f"{label} coverage differs from stats: "
                f"missing={sorted(expected - set(table))[:8]}, "
                f"extra={sorted(set(table) - expected)[:8]}"
            )
    for name in qnames:
        shape = _shape_from_stats(dict(stats[name]))
        if len(shape) < 2 or any(int(dim) <= 0 for dim in shape):
            raise ValueError(f"{name}: exact Linear shape required, got {shape}")
        _positive_int(source_bytes[name], where=f"source_bytes[{name!r}]")
        if not isinstance(source_kinds[name], str) or not source_kinds[name]:
            raise ValueError(f"{name}: missing exact source kind")
    _claim_spans(source_span_ids, construction_units)

    body_units = [
        _body_unit(
            component=component,
            stats=stats,
            source_kinds=source_kinds,
            source_bytes=source_bytes,
            source_span_ids=source_span_ids,
            profile=model_profile,
            target_profile=target_profile,
        )
        for component in _unit_components(model_profile, qnames)
    ]
    construction_rows: list[dict[str, Any]] = []
    for unit in sorted(construction_units, key=lambda item: item.name):
        payload = _positive_int(
            unit.payload_bytes, where=f"construction {unit.name} payload_bytes"
        )
        fmt = canonical_format_name(unit.format_name)
        if is_cb_format(fmt):
            raise ValueError(f"construction {unit.name}: source format may not be CB")
        packed = ".ffn.experts" in unit.name
        decision = check_serving_format(
            target_profile, unit.name, fmt, packed_expert=packed
        )
        if not decision.legal:
            raise ValueError(
                f"construction {unit.name}: source format {fmt} is not legal in "
                f"{target_profile}: {decision.reason}: {decision.detail}"
            )
        if not unit.physical_targets:
            raise ValueError(f"construction {unit.name}: no physical targets")
        if (
            len(unit.physical_targets) != len(unit.physical_target_payload_bytes)
            or tuple(sorted(unit.physical_targets)) != unit.physical_targets
            or len(set(unit.physical_targets)) != len(unit.physical_targets)
        ):
            raise ValueError(
                f"construction {unit.name}: physical target byte coverage differs"
            )
        physical_payload_bytes = tuple(
            _positive_int(
                value,
                where=f"construction {unit.name} physical target payload",
            )
            for value in unit.physical_target_payload_bytes
        )
        if sum(physical_payload_bytes) != payload:
            raise ValueError(
                f"construction {unit.name}: physical target bytes do not sum "
                "to payload_bytes"
            )
        construction_rows.append(
            {
                "kind": "construction",
                "name": unit.name,
                "serving_atomic": True,
                "packed_expert": packed,
                "physical_targets": list(sorted(unit.physical_targets)),
                "physical_target_payload_bytes": list(physical_payload_bytes),
                "source_span_ids": list(sorted(unit.source_span_ids)),
                "source_payload_bytes": payload,
                "available_no_cb_formats": [
                    {"format": fmt, "payload_bytes": payload}
                ],
                "lower_bound_format": fmt,
                "lower_bound_payload_bytes": payload,
            }
        )

    body_source = sum(int(value) for value in source_bytes.values())
    construction_source = sum(row["source_payload_bytes"] for row in construction_rows)
    immutable = source_total - body_source - construction_source
    if immutable < 0:
        raise ValueError(
            "mutable source spans exceed complete checkpoint payload: "
            f"{body_source}+{construction_source}>{source_total}"
        )
    body_lower = sum(row["lower_bound_payload_bytes"] for row in body_units)
    construction_lower = sum(
        row["lower_bound_payload_bytes"] for row in construction_rows
    )
    lower_bound = immutable + body_lower + construction_lower
    excess = lower_bound - budget
    if excess <= 0:
        raise ValueError(
            "all-native infeasibility is not proven: no-CB lower bound "
            f"{lower_bound} does not exceed budget {budget}"
        )

    body_member_digest = canonical_sha256({"members": list(qnames)})
    construction_member_digest = canonical_sha256(
        {
            "units": [
                {"name": row["name"], "physical_targets": row["physical_targets"]}
                for row in construction_rows
            ]
        }
    )
    all_formats = sorted(
        {
            option["format"]
            for row in (*body_units, *construction_rows)
            for option in row["available_no_cb_formats"]
        }
    )
    certificate: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "infeasible",
        "source_model": dict(source_model_binding),
        "contract": dict(contract_binding),
        "byte_budget": {"budget_bytes": budget},
        "candidate_scope": {
            "codebook_formats_excluded": True,
            "selection": _CANDIDATE_SELECTION,
            "conservative_superset": True,
            "formats_with_at_least_one_legal_unit": all_formats,
        },
        "coverage": {
            "complete": True,
            "body_member_count": len(qnames),
            "body_serving_unit_count": len(body_units),
            "construction_unit_count": len(construction_rows),
            "body_members_sha256": body_member_digest,
            "construction_members_sha256": construction_member_digest,
            "source_spans_are_disjoint": True,
            "probe_census": dict(probe_census),
        },
        "accounting": {
            "source_checkpoint_payload_bytes": source_total,
            "body_source_payload_bytes": body_source,
            "construction_source_payload_bytes": construction_source,
            "immutable_floor_bytes": immutable,
            "body_no_cb_lower_bound_bytes": body_lower,
            "construction_no_cb_lower_bound_bytes": construction_lower,
            "all_native_lower_bound_bytes": lower_bound,
            "budget_bytes": budget,
            "excess_bytes": excess,
        },
        "proof": {
            "relation": "all_native_lower_bound_bytes > budget_bytes",
            "exact_integer_bytes": True,
            "excess_bytes": excess,
        },
        "units": body_units + construction_rows,
    }
    certificate["certificate_sha256"] = certificate_sha256(certificate)
    validate_native_baseline_certificate(certificate)
    return certificate


def _validate_probe_census_binding(
    raw: object,
    body_members: Sequence[str],
) -> str:
    census = _require_exact_keys(
        raw,
        {
            "schema",
            "contract_id",
            "authority_sha256",
            "profile_id",
            "classifier",
            "routed_qname_regex",
            "expected_member_names_sha256",
            "observed_member_names_sha256",
            "expected",
            "observed",
            "complete",
        },
        where="coverage.probe_census",
    )
    if census["schema"] != PROBE_CENSUS_SCHEMA:
        raise ValueError("coverage.probe_census has unsupported schema")
    _require_string(census["contract_id"], where="probe_census.contract_id")
    _require_sha256(
        census["authority_sha256"], where="probe_census.authority_sha256"
    )
    profile_id = _require_string(
        census["profile_id"], where="probe_census.profile_id"
    )
    if census["classifier"] != "profile_declared_routed_expert.v1":
        raise ValueError("probe_census classifier is unsupported")
    if census["complete"] is not True:
        raise ValueError("probe_census must assert complete=true")
    pattern_text = _require_string(
        census["routed_qname_regex"], where="probe_census.routed_qname_regex"
    )
    try:
        pattern = re.compile(pattern_text)
    except re.error as exc:
        raise ValueError("probe_census routed_qname_regex is invalid") from exc
    expected_member_digest = _require_sha256(
        census["expected_member_names_sha256"],
        where="probe_census.expected_member_names_sha256",
    )
    observed_member_digest = _require_sha256(
        census["observed_member_names_sha256"],
        where="probe_census.observed_member_names_sha256",
    )

    count_keys = {
        "body_member_count",
        "routed_member_count",
        "nonexpert_member_count",
        "routed_role_counts",
        "nonexpert_role_counts",
    }

    def counts(raw_counts: object, *, where: str) -> dict[str, Any]:
        value = _require_exact_keys(raw_counts, count_keys, where=where)
        routed_roles = _role_counts(
            value["routed_role_counts"], where=f"{where}.routed_role_counts"
        )
        nonexpert_roles = _role_counts(
            value["nonexpert_role_counts"],
            where=f"{where}.nonexpert_role_counts",
        )
        out = {
            "body_member_count": _positive_int(
                value["body_member_count"], where=f"{where}.body_member_count"
            ),
            "routed_member_count": _positive_int(
                value["routed_member_count"], where=f"{where}.routed_member_count"
            ),
            "nonexpert_member_count": _positive_int(
                value["nonexpert_member_count"],
                where=f"{where}.nonexpert_member_count",
            ),
            "routed_role_counts": routed_roles,
            "nonexpert_role_counts": nonexpert_roles,
        }
        if (
            sum(routed_roles.values()) != out["routed_member_count"]
            or sum(nonexpert_roles.values()) != out["nonexpert_member_count"]
            or out["routed_member_count"] + out["nonexpert_member_count"]
            != out["body_member_count"]
        ):
            raise ValueError(f"{where} role counts do not sum to member counts")
        return out

    expected = counts(census["expected"], where="probe_census.expected")
    observed = counts(census["observed"], where="probe_census.observed")
    routed: Counter[str] = Counter()
    nonexpert: Counter[str] = Counter()
    for qname in body_members:
        role = qname.rsplit(".", 1)[-1]
        if pattern.fullmatch(qname) is not None:
            routed[role] += 1
        else:
            nonexpert[role] += 1
    reconstructed = {
        "body_member_count": len(body_members),
        "routed_member_count": sum(routed.values()),
        "nonexpert_member_count": sum(nonexpert.values()),
        "routed_role_counts": dict(sorted(routed.items())),
        "nonexpert_role_counts": dict(sorted(nonexpert.items())),
    }
    ledger_digest = canonical_sha256({"members": sorted(body_members)})
    if (
        expected != observed
        or observed != reconstructed
        or expected_member_digest != observed_member_digest
        or observed_member_digest != ledger_digest
    ):
        raise ValueError(
            "probe_census expected/observed counts do not match the unit ledger"
        )
    if profile_id == "deepseek_v4":
        from .model_profiles.deepseek_v4 import DeepseekV4Profile

        profile = DeepseekV4Profile()
        authoritative = evaluate_expected_probe_census(
            body_members,
            profile,
            _release_probe_census_contract(profile),
        )
        if dict(census) != authoritative:
            raise ValueError(
                "probe_census differs from the repository-owned DSv4 authority"
            )
    return profile_id


def _validate_source_model_binding(raw: object) -> None:
    value = _require_exact_keys(
        raw,
        {"identity_schema", "content_sha256", "identity_sha256", "shard_count"},
        where="source_model",
    )
    if value["identity_schema"] != "prismaquant.streamed_model.identity.v1":
        raise ValueError("source_model identity_schema is unsupported")
    _require_sha256(value["content_sha256"], where="source_model.content_sha256")
    _require_sha256(value["identity_sha256"], where="source_model.identity_sha256")
    _positive_int(value["shard_count"], where="source_model.shard_count")


def _validate_contract_binding(raw: object) -> str:
    contract = _require_exact_keys(
        raw,
        {"model_profile", "serving_profile", "lane", "gridbook_runtime_pin"},
        where="contract",
    )
    model = _require_exact_keys(
        contract["model_profile"],
        {
            "id",
            "structure_spec_sha256",
            "implementation_sha256",
            "identity_sha256",
        },
        where="contract.model_profile",
    )
    model_id = _require_string(model["id"], where="contract.model_profile.id")
    structure_sha = _require_sha256(
        model["structure_spec_sha256"],
        where="contract.model_profile.structure_spec_sha256",
    )
    implementation_sha = _require_sha256(
        model["implementation_sha256"],
        where="contract.model_profile.implementation_sha256",
    )
    identity_sha = _require_sha256(
        model["identity_sha256"], where="contract.model_profile.identity_sha256"
    )
    expected_identity = canonical_sha256(
        {
            "id": model_id,
            "structure_spec_sha256": structure_sha,
            "implementation_sha256": implementation_sha,
        }
    )
    if identity_sha != expected_identity:
        raise ValueError("contract.model_profile identity_sha256 is inconsistent")

    serving = _require_exact_keys(
        contract["serving_profile"],
        {"id", "spec_sha256"},
        where="contract.serving_profile",
    )
    serving_id = _require_string(
        serving["id"], where="contract.serving_profile.id"
    )
    _require_sha256(
        serving["spec_sha256"], where="contract.serving_profile.spec_sha256"
    )
    lane = _require_exact_keys(
        contract["lane"],
        {"id", "export_container", "spec_sha256"},
        where="contract.lane",
    )
    lane_id = _require_string(lane["id"], where="contract.lane.id")
    _require_string(
        lane["export_container"], where="contract.lane.export_container"
    )
    _require_sha256(lane["spec_sha256"], where="contract.lane.spec_sha256")
    if lane_id == "nvfp4_cb" and serving_id != "nvfp4_cb":
        raise ValueError("nvfp4_cb lane must bind the nvfp4_cb serving profile")

    runtime = _require_exact_keys(
        contract["gridbook_runtime_pin"],
        {
            "schema",
            "repository",
            "commit",
            "version",
            "version_is_release",
            "runtime_contract_schema",
            "required_abi_features",
            "file_sha256",
        },
        where="contract.gridbook_runtime_pin",
    )
    _require_string(runtime["schema"], where="gridbook_runtime_pin.schema")
    _require_string(runtime["repository"], where="gridbook_runtime_pin.repository")
    commit = _require_string(runtime["commit"], where="gridbook_runtime_pin.commit")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("gridbook_runtime_pin.commit must be lowercase full SHA")
    _require_string(runtime["version"], where="gridbook_runtime_pin.version")
    if not isinstance(runtime["version_is_release"], bool):
        raise ValueError("gridbook_runtime_pin.version_is_release must be boolean")
    _require_string(
        runtime["runtime_contract_schema"],
        where="gridbook_runtime_pin.runtime_contract_schema",
    )
    features = runtime["required_abi_features"]
    if not isinstance(features, Mapping) or any(
        type(value) is not int for value in features.values()
    ):
        raise ValueError(
            "gridbook_runtime_pin.required_abi_features must be an integer map"
        )
    _require_sha256(
        runtime["file_sha256"], where="gridbook_runtime_pin.file_sha256"
    )
    if model_id == "deepseek_v4":
        from .model_profiles.deepseek_v4 import DeepseekV4Profile

        authoritative = _contract_binding(
            DeepseekV4Profile(), serving_id, lane_id
        )
        if dict(contract) != authoritative:
            raise ValueError(
                "contract hashes differ from the current DSv4 profile/lane/runtime"
            )
    return model_id


def _validate_format_options(raw: object, *, where: str) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where} must be a non-empty list")
    registered = {
        canonical_format_name(spec.name) for spec in list_producer_formats()
    }
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        row = _require_exact_keys(
            item, {"format", "payload_bytes"}, where=f"{where}[{index}]"
        )
        fmt = _require_string(row["format"], where=f"{where}[{index}].format")
        if fmt != canonical_format_name(fmt) or fmt not in registered:
            raise ValueError(f"{where}[{index}] is not a registered canonical format")
        if is_cb_format(fmt):
            raise ValueError(f"{where}[{index}] contains codebook format {fmt}")
        if fmt in seen:
            raise ValueError(f"{where} repeats format {fmt}")
        seen.add(fmt)
        out.append(
            {
                "format": fmt,
                "payload_bytes": _positive_int(
                    row["payload_bytes"], where=f"{where}[{index}].payload_bytes"
                ),
            }
        )
    if out != sorted(out, key=lambda row: (row["payload_bytes"], row["format"])):
        raise ValueError(f"{where} must be sorted by exact bytes then format")
    return out


def validate_native_baseline_certificate(certificate: Mapping[str, Any]) -> None:
    """Reconstruct and verify the complete infeasibility proof ledger."""

    top = _require_exact_keys(
        certificate,
        {
            "schema",
            "status",
            "source_model",
            "contract",
            "byte_budget",
            "candidate_scope",
            "coverage",
            "accounting",
            "proof",
            "units",
            "certificate_sha256",
        },
        where="certificate",
    )
    if top["schema"] != SCHEMA:
        raise ValueError(f"unsupported certificate schema {top['schema']!r}")
    digest = _require_sha256(
        top["certificate_sha256"], where="certificate_sha256"
    )
    if digest != certificate_sha256(certificate):
        raise ValueError("certificate_sha256 does not match canonical payload")
    if top["status"] != "infeasible":
        raise ValueError("certificate status must be infeasible")
    _validate_source_model_binding(top["source_model"])
    model_profile_id = _validate_contract_binding(top["contract"])
    target_profile_id = str(top["contract"]["serving_profile"]["id"])

    byte_budget = _require_exact_keys(
        top["byte_budget"], {"budget_bytes"}, where="byte_budget"
    )
    budget = _positive_int(
        byte_budget["budget_bytes"], where="byte_budget.budget_bytes"
    )
    scope = _require_exact_keys(
        top["candidate_scope"],
        {
            "codebook_formats_excluded",
            "selection",
            "conservative_superset",
            "formats_with_at_least_one_legal_unit",
        },
        where="candidate_scope",
    )
    if scope["codebook_formats_excluded"] is not True:
        raise ValueError("candidate_scope must exclude codebook formats")
    if scope["conservative_superset"] is not True:
        raise ValueError("candidate_scope must declare the conservative superset")
    if scope["selection"] != _CANDIDATE_SELECTION:
        raise ValueError("candidate_scope selection contract is unsupported")
    declared_formats = _require_sorted_unique_strings(
        scope["formats_with_at_least_one_legal_unit"],
        where="candidate_scope.formats_with_at_least_one_legal_unit",
    )

    units = top["units"]
    if not isinstance(units, list) or not units:
        raise ValueError("units must be a non-empty list")
    serving_rows: list[Mapping[str, Any]] = []
    construction_rows: list[Mapping[str, Any]] = []
    body_members: list[str] = []
    body_spans: set[str] = set()
    construction_spans: set[str] = set()
    physical_targets: set[str] = set()
    all_formats: set[str] = set()
    unit_names: set[str] = set()
    reached_construction = False
    previous_serving_member = ""
    previous_construction_name = ""

    for index, raw_unit in enumerate(units):
        if not isinstance(raw_unit, Mapping):
            raise ValueError(f"units[{index}] must be an object")
        kind = raw_unit.get("kind")
        common = {
            "kind",
            "name",
            "serving_atomic",
            "packed_expert",
            "source_span_ids",
            "source_payload_bytes",
            "available_no_cb_formats",
            "lower_bound_format",
            "lower_bound_payload_bytes",
        }
        if kind == "serving":
            row = _require_exact_keys(
                raw_unit,
                common | {"members", "member_evidence", "source_kinds"},
                where=f"units[{index}]",
            )
            if reached_construction:
                raise ValueError("serving units must precede construction units")
            members = _require_sorted_unique_strings(
                row["members"], where=f"units[{index}].members"
            )
            if members[0] <= previous_serving_member:
                raise ValueError("serving units are not in canonical member order")
            previous_serving_member = members[0]
            expected_name = members[0] if len(members) == 1 else f"group:{members[0]}"
            if row["name"] != expected_name:
                raise ValueError(f"units[{index}] serving name is not canonical")
            if row["serving_atomic"] is not (len(members) > 1):
                raise ValueError(f"units[{index}] serving_atomic is inconsistent")
            if not isinstance(row["packed_expert"], bool):
                raise ValueError(f"units[{index}].packed_expert must be boolean")
            declared_source_kinds = _require_sorted_unique_strings(
                row["source_kinds"], where=f"units[{index}].source_kinds"
            )
            raw_evidence = row["member_evidence"]
            if not isinstance(raw_evidence, list) or len(raw_evidence) != len(members):
                raise ValueError(f"units[{index}] member evidence coverage differs")
            evidence: list[dict[str, Any]] = []
            evidence_spans: set[str] = set()
            for member_index, raw_member in enumerate(raw_evidence):
                member = _require_exact_keys(
                    raw_member,
                    {
                        "shape",
                        "source_kind",
                        "source_payload_bytes",
                        "source_span_ids",
                        "nvfp4_weight_only",
                    },
                    where=f"units[{index}].member_evidence[{member_index}]",
                )
                shape = member["shape"]
                if (
                    not isinstance(shape, list)
                    or len(shape) < 2
                    or any(
                        isinstance(dim, bool)
                        or not isinstance(dim, int)
                        or dim <= 0
                        for dim in shape
                    )
                ):
                    raise ValueError(
                        f"units[{index}].member_evidence[{member_index}].shape "
                        "must be an exact positive Linear shape"
                    )
                source_kind = _require_string(
                    member["source_kind"],
                    where=(
                        f"units[{index}].member_evidence[{member_index}]"
                        ".source_kind"
                    ),
                )
                member_spans = _require_sorted_unique_strings(
                    member["source_span_ids"],
                    where=(
                        f"units[{index}].member_evidence[{member_index}]"
                        ".source_span_ids"
                    ),
                )
                overlap = evidence_spans.intersection(member_spans)
                if overlap:
                    raise ValueError(
                        f"units[{index}] member source spans overlap: "
                        f"{sorted(overlap)}"
                    )
                evidence_spans.update(member_spans)
                if not isinstance(member["nvfp4_weight_only"], bool):
                    raise ValueError(
                        f"units[{index}].member_evidence[{member_index}] "
                        "nvfp4_weight_only must be boolean"
                    )
                evidence.append(
                    {
                        "shape": list(shape),
                        "source_kind": source_kind,
                        "source_payload_bytes": _positive_int(
                            member["source_payload_bytes"],
                            where=(
                                f"units[{index}].member_evidence[{member_index}]"
                                ".source_payload_bytes"
                            ),
                        ),
                        "source_span_ids": member_spans,
                        "nvfp4_weight_only": member["nvfp4_weight_only"],
                    }
                )
            if declared_source_kinds != sorted(
                {member["source_kind"] for member in evidence}
            ):
                raise ValueError(f"units[{index}].source_kinds differs from evidence")
            body_members.extend(members)
            serving_rows.append(row)
        elif kind == "construction":
            row = _require_exact_keys(
                raw_unit,
                common | {"physical_targets", "physical_target_payload_bytes"},
                where=f"units[{index}]",
            )
            reached_construction = True
            name = _require_string(row["name"], where=f"units[{index}].name")
            if name <= previous_construction_name:
                raise ValueError("construction units are not in canonical name order")
            previous_construction_name = name
            if row["serving_atomic"] is not True:
                raise ValueError(f"units[{index}] construction must be serving-atomic")
            expected_packed = ".ffn.experts" in name
            if row["packed_expert"] is not expected_packed:
                raise ValueError(f"units[{index}].packed_expert is inconsistent")
            physical = _require_sorted_unique_strings(
                row["physical_targets"], where=f"units[{index}].physical_targets"
            )
            raw_physical_bytes = row["physical_target_payload_bytes"]
            if (
                not isinstance(raw_physical_bytes, list)
                or len(raw_physical_bytes) != len(physical)
            ):
                raise ValueError(
                    f"units[{index}] physical target byte coverage differs"
                )
            physical_bytes = [
                _positive_int(
                    value,
                    where=(
                        f"units[{index}].physical_target_payload_bytes[{item_index}]"
                    ),
                )
                for item_index, value in enumerate(raw_physical_bytes)
            ]
            overlap = physical_targets.intersection(physical)
            if overlap:
                raise ValueError(f"construction physical targets overlap: {sorted(overlap)}")
            physical_targets.update(physical)
            construction_rows.append(row)
        else:
            raise ValueError(f"units[{index}] has unsupported kind {kind!r}")

        name = _require_string(row["name"], where=f"units[{index}].name")
        if name in unit_names:
            raise ValueError(f"unit name {name!r} is repeated")
        unit_names.add(name)
        spans = _require_sorted_unique_strings(
            row["source_span_ids"], where=f"units[{index}].source_span_ids"
        )
        if kind == "serving" and spans != sorted(evidence_spans):
            raise ValueError(f"units[{index}] aggregate spans differ from evidence")
        claimed = body_spans if kind == "serving" else construction_spans
        other = construction_spans if kind == "serving" else body_spans
        overlap = (claimed | other).intersection(spans)
        if overlap:
            raise ValueError(f"source spans overlap across units: {sorted(overlap)}")
        claimed.update(spans)
        source_payload = _positive_int(
            row["source_payload_bytes"], where=f"units[{index}].source_payload_bytes"
        )
        options = _validate_format_options(
            row["available_no_cb_formats"],
            where=f"units[{index}].available_no_cb_formats",
        )
        if kind == "serving":
            evidence_source = sum(
                member["source_payload_bytes"] for member in evidence
            )
            if evidence_source != source_payload:
                raise ValueError(
                    f"units[{index}] aggregate source bytes differ from evidence"
                )
            recomputed_options = _legal_no_cb_options(
                members=members,
                member_evidence=evidence,
                packed=bool(row["packed_expert"]),
                target_profile=target_profile_id,
            )
            if options != recomputed_options:
                raise ValueError(
                    f"units[{index}] no-CB options differ from exact legality/bytes"
                )
        best = options[0]
        if (
            row["lower_bound_format"] != best["format"]
            or row["lower_bound_payload_bytes"] != best["payload_bytes"]
        ):
            raise ValueError(f"units[{index}] lower bound is not its exact minimum")
        if kind == "construction":
            if len(options) != 1 or options[0]["payload_bytes"] != source_payload:
                raise ValueError(
                    f"units[{index}] source-only construction option is inconsistent"
                )
            if row["physical_targets"] != row["source_span_ids"]:
                raise ValueError(
                    f"units[{index}] construction span coverage differs from targets"
                )
            if sum(physical_bytes) != source_payload:
                raise ValueError(
                    f"units[{index}] physical target bytes do not sum to source payload"
                )
        all_formats.update(option["format"] for option in options)

    if len(body_members) != len(set(body_members)):
        raise ValueError("body members overlap across serving units")
    body_members.sort()
    if model_profile_id == "deepseek_v4":
        from .model_profiles.deepseek_v4 import DeepseekV4Profile

        profile = DeepseekV4Profile()
        expected_components = _unit_components(profile, body_members)
        observed_components = tuple(tuple(row["members"]) for row in serving_rows)
        if observed_components != expected_components:
            raise ValueError(
                "DSv4 serving-unit partition differs from the bound model profile"
            )
        for row, component in zip(serving_rows, expected_components, strict=True):
            expected_packed = _component_is_packed(profile, component)
            if row["packed_expert"] is not expected_packed:
                raise ValueError(
                    "DSv4 packed-expert flags differ from the bound model profile"
                )
    if declared_formats != sorted(all_formats):
        raise ValueError("candidate_scope format union differs from unit ledger")

    coverage = _require_exact_keys(
        top["coverage"],
        {
            "complete",
            "body_member_count",
            "body_serving_unit_count",
            "construction_unit_count",
            "body_members_sha256",
            "construction_members_sha256",
            "source_spans_are_disjoint",
            "probe_census",
        },
        where="coverage",
    )
    if coverage["complete"] is not True or coverage["source_spans_are_disjoint"] is not True:
        raise ValueError("coverage must assert complete and disjoint source spans")
    if _positive_int(coverage["body_member_count"], where="coverage.body_member_count") != len(body_members):
        raise ValueError("coverage.body_member_count differs from unit ledger")
    if _positive_int(coverage["body_serving_unit_count"], where="coverage.body_serving_unit_count") != len(serving_rows):
        raise ValueError("coverage.body_serving_unit_count differs from unit ledger")
    construction_count = _positive_int(
        coverage["construction_unit_count"],
        where="coverage.construction_unit_count",
        allow_zero=True,
    )
    if construction_count != len(construction_rows):
        raise ValueError("coverage.construction_unit_count differs from unit ledger")
    expected_body_digest = canonical_sha256({"members": body_members})
    if _require_sha256(coverage["body_members_sha256"], where="coverage.body_members_sha256") != expected_body_digest:
        raise ValueError("coverage.body_members_sha256 differs from unit ledger")
    construction_digest = canonical_sha256(
        {
            "units": [
                {"name": row["name"], "physical_targets": row["physical_targets"]}
                for row in construction_rows
            ]
        }
    )
    if _require_sha256(
        coverage["construction_members_sha256"],
        where="coverage.construction_members_sha256",
    ) != construction_digest:
        raise ValueError("coverage.construction_members_sha256 differs from unit ledger")
    census_profile_id = _validate_probe_census_binding(
        coverage["probe_census"], body_members
    )
    if census_profile_id != model_profile_id:
        raise ValueError("probe census profile differs from contract model profile")

    accounting = _require_exact_keys(
        top["accounting"],
        {
            "source_checkpoint_payload_bytes",
            "body_source_payload_bytes",
            "construction_source_payload_bytes",
            "immutable_floor_bytes",
            "body_no_cb_lower_bound_bytes",
            "construction_no_cb_lower_bound_bytes",
            "all_native_lower_bound_bytes",
            "budget_bytes",
            "excess_bytes",
        },
        where="accounting",
    )
    source_total = _positive_int(
        accounting["source_checkpoint_payload_bytes"],
        where="accounting.source_checkpoint_payload_bytes",
    )
    body_source = sum(int(row["source_payload_bytes"]) for row in serving_rows)
    construction_source = sum(
        int(row["source_payload_bytes"]) for row in construction_rows
    )
    body_lower = sum(int(row["lower_bound_payload_bytes"]) for row in serving_rows)
    construction_lower = sum(
        int(row["lower_bound_payload_bytes"]) for row in construction_rows
    )
    immutable = _positive_int(
        accounting["immutable_floor_bytes"],
        where="accounting.immutable_floor_bytes",
        allow_zero=True,
    )
    declared_body_source = _positive_int(
        accounting["body_source_payload_bytes"],
        where="accounting.body_source_payload_bytes",
    )
    declared_construction_source = _positive_int(
        accounting["construction_source_payload_bytes"],
        where="accounting.construction_source_payload_bytes",
        allow_zero=True,
    )
    declared_body_lower = _positive_int(
        accounting["body_no_cb_lower_bound_bytes"],
        where="accounting.body_no_cb_lower_bound_bytes",
    )
    declared_construction_lower = _positive_int(
        accounting["construction_no_cb_lower_bound_bytes"],
        where="accounting.construction_no_cb_lower_bound_bytes",
        allow_zero=True,
    )
    lower = _positive_int(
        accounting["all_native_lower_bound_bytes"],
        where="accounting.all_native_lower_bound_bytes",
    )
    accounting_budget = _positive_int(
        accounting["budget_bytes"], where="accounting.budget_bytes"
    )
    excess = _positive_int(
        accounting["excess_bytes"], where="accounting.excess_bytes"
    )
    if (
        declared_body_source != body_source
        or declared_construction_source != construction_source
        or declared_body_lower != body_lower
        or declared_construction_lower != construction_lower
        or source_total != immutable + body_source + construction_source
        or lower != immutable + body_lower + construction_lower
        or accounting_budget != budget
        or lower - budget != excess
    ):
        raise ValueError("accounting cannot be reconstructed from the unit ledger")

    proof = _require_exact_keys(
        top["proof"],
        {"relation", "exact_integer_bytes", "excess_bytes"},
        where="proof",
    )
    if (
        proof["relation"] != "all_native_lower_bound_bytes > budget_bytes"
        or proof["exact_integer_bytes"] is not True
        or _positive_int(proof["excess_bytes"], where="proof.excess_bytes") != excess
    ):
        raise ValueError("proof statement differs from reconstructed accounting")


def _contract_binding(profile: object, target_profile: str, lane_id: str) -> dict[str, Any]:
    profile_id = str(getattr(profile, "name"))
    structure_path = _PACKAGE_ROOT / "model_profiles" / "specs" / f"{profile_id}.json"
    serving_path = _PACKAGE_ROOT / "serving_profile_specs" / f"{target_profile}.json"
    lane_path = _PACKAGE_ROOT / "lane_specs" / f"{lane_id}.json"
    runtime_path = _PACKAGE_ROOT / "gridbook_runtime" / "gridbook_runtime_pin.json"
    for path in (structure_path, serving_path, lane_path, runtime_path):
        if not path.is_file():
            raise FileNotFoundError(f"certificate identity input is missing: {path}")
    implementation_path = Path(inspect.getfile(type(profile))).resolve()
    profile_parts = {
        "id": profile_id,
        "structure_spec_sha256": _file_sha256(structure_path),
        "implementation_sha256": _file_sha256(implementation_path),
    }
    profile_parts["identity_sha256"] = canonical_sha256(profile_parts)
    lane = load_lane_spec(lane_id)
    if target_profile not in lane.serving_profiles:
        raise ValueError(
            f"lane {lane_id!r} does not declare serving profile {target_profile!r}"
        )
    pin = load_gridbook_runtime_pin()
    return {
        "model_profile": profile_parts,
        "serving_profile": {
            "id": target_profile,
            "spec_sha256": _file_sha256(serving_path),
        },
        "lane": {
            "id": lane.id,
            "export_container": lane.export_container,
            "spec_sha256": _file_sha256(lane_path),
        },
        "gridbook_runtime_pin": {
            "schema": pin.schema,
            "repository": pin.repository,
            "commit": pin.commit,
            "version": pin.version,
            "version_is_release": pin.version_is_release,
            "runtime_contract_schema": pin.runtime_contract_schema,
            "required_abi_features": dict(pin.required_abi_features),
            "file_sha256": _file_sha256(runtime_path),
        },
    }


def _source_model_binding(identity: Mapping[str, Any]) -> dict[str, Any]:
    content = identity.get("content_sha256")
    if not isinstance(content, str) or _SHA256_RE.fullmatch(content) is None:
        raise ValueError("validated streamed identity has no content_sha256")
    shards = identity.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("validated streamed identity has no source shards")
    return {
        "identity_schema": identity.get("schema"),
        "content_sha256": content,
        "identity_sha256": canonical_sha256(dict(identity)),
        "shard_count": len(shards),
    }


def upgrade_partial_model_identity_cache(
    source_model: str | Path,
    identity_cache_path: str | Path,
) -> dict[str, object]:
    """Upgrade an old decoder-only cache through the repository identity owner.

    ``build_streamed_model_identity`` already has the safe incremental law:
    reuse digests whose mutation-sensitive fingerprints still match and hash
    newly covered index shards only.  This lightweight adapter supplies its
    config/name-map inputs from the validated old cache; it never constructs
    or loads model weights.
    """

    source = str(Path(source_model).resolve())
    cache_path = Path(identity_cache_path)
    _cached, old_identity = _read_streamed_model_identity_cache(
        cache_path, source_model=source
    )
    config = old_identity.get("config")
    weight_map = old_identity.get("weight_map")
    if not isinstance(config, Mapping) or not isinstance(weight_map, Mapping):
        raise ValueError("partial streamed identity lacks config/weight_map inputs")

    class _CachedConfig:
        def to_dict(self) -> dict[str, Any]:
            return dict(config)

    runner = SimpleNamespace(
        model=SimpleNamespace(config=_CachedConfig()),
        context=SimpleNamespace(
            weight_ckpt={str(key): str(value) for key, value in weight_map.items()},
            weight_shard={},
        ),
    )
    return build_streamed_model_identity(
        runner,
        source,
        identity_cache_path=cache_path,
    )


def _dspark_construction_units(model_path: str) -> tuple[NativeConstructionUnit, ...]:
    overlay = discover_dspark_source_overlay_from_artifact(model_path)
    if overlay is None:
        return ()
    header = read_artifact_header(model_path)
    members: dict[str, list[str]] = defaultdict(list)
    for physical, unit in overlay.physical_to_construction_unit.items():
        members[str(unit)].append(str(physical))
    out: list[NativeConstructionUnit] = []
    for unit, physical_targets in sorted(members.items()):
        payload = 0
        physical_payload_bytes: list[int] = []
        span_ids: list[str] = []
        for base in sorted(physical_targets):
            physical_payload = 0
            for suffix in (".weight", ".scale"):
                name = base + suffix
                meta = header.get(name)
                if not isinstance(meta, Mapping):
                    raise ValueError(f"DSpark target span missing from header: {name}")
                offsets = meta.get("data_offsets")
                if not isinstance(offsets, list) or len(offsets) != 2:
                    raise ValueError(f"DSpark target has malformed data_offsets: {name}")
                span_bytes = int(offsets[1]) - int(offsets[0])
                payload += span_bytes
                physical_payload += span_bytes
            # Match SourceByteManifest's provenance namespace: one checkpoint
            # base identifies the folded weight+scale payload.  Using tensor
            # suffixes here would make a body/construction overlap invisible.
            span_ids.append(base)
            physical_payload_bytes.append(physical_payload)
        out.append(
            NativeConstructionUnit(
                name=unit,
                format_name=str(overlay.construction_units[unit]),
                payload_bytes=payload,
                physical_targets=tuple(sorted(physical_targets)),
                source_span_ids=tuple(sorted(span_ids)),
                physical_target_payload_bytes=tuple(physical_payload_bytes),
            )
        )
    return tuple(out)


def load_probe_stats(path: str | Path) -> dict[str, Mapping[str, object]]:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: probe must be a mapping")
    raw = payload.get("stats", payload)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: probe stats must be a mapping")
    stats = {
        str(name): value
        for name, value in raw.items()
        if name != "meta" and isinstance(value, Mapping)
    }
    if not stats:
        raise ValueError(f"{path}: probe contains no Linear stats")
    return stats


def certify_native_baseline_from_model(
    *,
    model_path: str | Path,
    probe_path: str | Path,
    model_identity_cache_path: str | Path,
    budget_bytes: int,
    target_profile: str = "nvfp4_cb",
    lane_id: str = "nvfp4_cb",
    upgrade_partial_identity_cache: bool = False,
) -> dict[str, Any]:
    """Discover a real checkpoint and produce its exact infeasibility proof."""

    model = str(Path(model_path).resolve())
    stats = load_probe_stats(probe_path)
    profile = detect_profile(model)
    probe_census = evaluate_expected_probe_census(
        stats,
        profile,
        _release_probe_census_contract(profile),
    )
    manifest = source_tensor_bytes_manifest(
        model,
        name_map=profile.checkpoint_to_live_name,
        expert_parent_for_projection=profile.packed_expert_parent_for_projection,
    )
    resolved = resolve_reencoded_source_bytes(
        manifest, stats, context="native-baseline feasibility certificate"
    )
    spans = _span_map_for_names(manifest, stats)
    source_census = _scan_source_dtype_manifest(model, profile)
    missing_kinds = sorted(set(stats) - set(source_census))
    if missing_kinds:
        raise ValueError(
            f"source-kind census misses {len(missing_kinds)} probe members: "
            f"{missing_kinds[:8]}"
        )
    if upgrade_partial_identity_cache:
        upgrade_partial_model_identity_cache(model, model_identity_cache_path)
    identity = validate_cached_streamed_model_identity(
        model, model_identity_cache_path, require_complete_checkpoint=True
    )
    source_total, _by_dtype = source_checkpoint_bytes(model)
    certificate = build_native_baseline_certificate(
        stats=stats,
        source_kinds={name: source_census[name] for name in stats},
        source_bytes=resolved,
        source_span_ids=spans,
        model_profile=profile,
        target_profile=target_profile,
        lane_id=lane_id,
        budget_bytes=budget_bytes,
        source_checkpoint_payload_bytes=source_total,
        source_model_binding=_source_model_binding(identity),
        contract_binding=_contract_binding(profile, target_profile, lane_id),
        probe_census=probe_census,
        construction_units=_dspark_construction_units(model),
    )
    validate_native_baseline_certificate(certificate)
    return certificate


def write_certificate(certificate: Mapping[str, Any], output_path: str | Path) -> None:
    validate_native_baseline_certificate(certificate)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")


__all__ = [
    "SCHEMA",
    "PROBE_CENSUS_SCHEMA",
    "ExpectedProbeCensus",
    "NativeConstructionUnit",
    "build_native_baseline_certificate",
    "canonical_sha256",
    "certificate_sha256",
    "certify_native_baseline_from_model",
    "evaluate_expected_probe_census",
    "load_probe_stats",
    "upgrade_partial_model_identity_cache",
    "validate_native_baseline_certificate",
    "write_certificate",
]
