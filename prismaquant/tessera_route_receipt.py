"""The priced-vs-served route gate for the Tessera lane (#136, Tessera #126).

Scoped census v2 retains the producer's complete observations and exact input
texts. Its replay binds price projections to the independent card build and
artifact sidecars, then compares each owner's actual launch to the current
scoped cells (``LANE_ELIGIBILITY_SCHEMA_TESSERA``) at the exact
image/mode/residency. A decoder used as a dense fallback
may be a legitimate routed-MoE launch only where that cell explicitly says so.
There is no global substitute veto for scoped rows. Legacy flat rows below
retain the historical v4 check and can never acquire fabricated scope.

Tessera's runtime contract publishes, per native extension, what a serve
does when the ``.so`` cannot build (``native_extensions[].when_unavailable``,
contract v7): resident mode keeps serving on a NAMED substitute decoder and
stamps it on every route record.  The field exists so "a receipt must never
claim the native decoder for a serve that took the" fallback -- but nothing
in PrismaQuant read it, so a shipcard could price ``TESSERA_NVFP4`` W4A4
while a serve produced every number on ``torch_materialize_stock`` with
nothing refusing.

The legacy path reads historical ``{route, decoder}`` row arrays. The
current producer tool remains Tessera-owned (``lane_specs/tessera.json``),
and its complete v2 object must use the scoped path above. The legacy
comparison refuses on:

* no records at all (an absent census is not a clean bill);
* any record without a decoder (a decoder-less row read as native is the
  hole that existed before this gate);
* any record whose decoder is a KNOWN substitute -- derived from the pinned
  contract's ``when_unavailable`` via
  :func:`substitute_decoders_from_contract_answer`, never hardcoded here;
* served-vs-priced route disagreement in either direction (a number whose
  units rode a route other than the one they were priced on is not a
  result).

Stdlib only, no torch: the shipcard replays this at publication through
``prismaquant.shipcard``, which is stdlib-only by contract.

What the legacy path does not do: its old row protocol carries no matched
cell launch, so that path cannot assert "this ran native" -- it
refuses every substitute the pin names and stamps the served decoder set it
observed, which is the positive claim the receipt carries.  Coverage
strictness is uncalibrated against a real serve (nothing has been served
yet on this side): both directions refuse, and relaxing either needs a
measured serve, not an argument.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


class TesseraRouteReceiptError(ValueError):
    """The route census is missing, malformed, or disagrees with the price."""


def substitute_decoders_from_contract_answer(
    answer: Mapping[str, Any],
) -> tuple[str, ...]:
    """Every substitute decoder the pinned answer names, sorted.

    Derived from ``native_extensions[].when_unavailable[].decoder`` (nulls
    dropped: a refused serve has no decoder to detect).  The gate reads this
    set rather than a hardcoded name, so a runtime that renames its fallback
    moves the gate instead of passing silently -- and the dev-pin's answer
    refusal is what keeps the transcription honest.
    """
    try:
        rows = answer["native_extensions"]
    except (KeyError, TypeError) as exc:
        raise TesseraRouteReceiptError(
            "the contract answer carries no native_extensions table"
        ) from exc
    decoders = set()
    for row in rows:
        behaviours = row.get("when_unavailable") if isinstance(
            row, Mapping) else None
        if not isinstance(behaviours, Mapping):
            continue
        for behaviour in behaviours.values():
            decoder = behaviour.get("decoder") if isinstance(
                behaviour, Mapping) else None
            if isinstance(decoder, str) and decoder:
                decoders.add(decoder)
    return tuple(sorted(decoders))


def parse_route_records(
    records: Any,
    *,
    where: str = "route records",
) -> list[dict[str, Any]]:
    """Fail-closed read of census rows into ``{route, decoder, count}``.

    Each row must carry a non-empty ``route`` and a non-empty ``decoder``;
    ``count`` is carried when present (a non-negative integer) and ``None``
    otherwise. Unrelated extra keys remain legacy-compatible, but explicit
    runtime/owner/phase fields refuse: flattening them would discard scope.
    """
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TesseraRouteReceiptError(
            f"{where} must be a sequence of route rows")
    parsed: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        at = f"{where}[{index}]"
        if not isinstance(row, Mapping):
            raise TesseraRouteReceiptError(f"{at} must be an object")
        if set(row) & {"runtime", "runtime_image", "execution_mode", "structure",
                       "residency", "phase", "owner", "record_owner"}:
            raise TesseraRouteReceiptError(
                f"{at}: scoped runtime observations require the complete v2 census; "
                "legacy flattening cannot retain or validate their scope")
        route, decoder = row.get("route"), row.get("decoder")
        if not isinstance(route, str) or not route:
            raise TesseraRouteReceiptError(
                f"{at} carries no route name; a row that cannot say which "
                "route it rode is not evidence for any of them")
        if not isinstance(decoder, str) or not decoder:
            raise TesseraRouteReceiptError(
                f"{at} carries no decoder for route {route!r}; a "
                "decoder-less row read as native is the hole this gate "
                "exists to close")
        count = row.get("count")
        if count is not None and (
                isinstance(count, bool) or not isinstance(count, int)
                or count < 0):
            raise TesseraRouteReceiptError(
                f"{at} carries count {count!r}, not a non-negative integer")
        parsed.append({"route": route, "decoder": decoder, "count": count})
    return parsed


def check_route_receipt(
    *,
    priced_routes: Sequence[str],
    route_records: Sequence[Mapping[str, Any]],
    substitute_decoders: Sequence[str],
) -> dict[str, Any]:
    """Priced-vs-served comparison.  Returns the verdict; never raises it.

    Malformed INPUTS (nothing priced, no known substitute, malformed rows)
    raise: a gate constructed so it cannot detect anything must not return a
    verdict.  A well-formed census that DISAGREES returns
    ``passed: False`` with the disagreement itemised.
    """
    priced = [str(r) for r in (priced_routes or ())]
    if not priced or any(not r for r in priced):
        raise TesseraRouteReceiptError(
            "the artifact priced no routes, so there is nothing to compare "
            "a serve against; name the priced routes explicitly")
    substitutes = [str(d) for d in (substitute_decoders or ())]
    if not substitutes or any(not d for d in substitutes):
        raise TesseraRouteReceiptError(
            "the gate knows no substitute decoder, so every serve would "
            "pass; derive the set from the pinned contract answer "
            "(substitute_decoders_from_contract_answer)")
    records = parse_route_records(list(route_records))

    verdict: dict[str, Any] = {
        "priced_routes": sorted(set(priced)),
        "served_routes": sorted({row["route"] for row in records}),
        "served_decoders": sorted({row["decoder"] for row in records}),
        "substitute_hits": sorted(
            {row["decoder"] for row in records
             if row["decoder"] in substitutes}),
        "unserved_priced": sorted(
            set(priced) - {row["route"] for row in records}),
        "unpriced_served": sorted(
            {row["route"] for row in records} - set(priced)),
        "n_records": len(records),
    }
    reasons: list[str] = []
    if not records:
        reasons.append("the serve emitted no route records")
    if verdict["substitute_hits"]:
        reasons.append(
            "served on substitute decoder(s) "
            f"{verdict['substitute_hits']}: the priced routes were not the "
            "routes that ran")
    if verdict["unserved_priced"]:
        reasons.append(
            f"priced route(s) never served: {verdict['unserved_priced']}")
    if verdict["unpriced_served"]:
        reasons.append(
            f"served route(s) never priced: {verdict['unpriced_served']}")
    verdict["passed"] = not reasons
    verdict["detail"] = ("route census agrees: "
                         f"{verdict['served_routes']} on "
                         f"{verdict['served_decoders']}"
                         if verdict["passed"]
                         else "route census REFUSED: " + "; ".join(reasons))
    return verdict


SCOPED_CENSUS_SCHEMA = "tessera.serving.route_census/2"
# The two forwards in this producer schema; future phases require review.
CENSUS_PHASE_REGIMES = {"decode": "decode", "prefill": "batch"}
_SIDECARS = {"config.json": "config_json", "tessera_serving_manifest.json": "manifest_json"}


def _require(condition, message):
    if not condition:
        raise TesseraRouteReceiptError(message)


def _object(value, where):
    _require(isinstance(value, Mapping), f"{where} must be an object")
    return value


def parse_census_json(text, *, where):
    """Preserve exact input bytes separately; refuse ambiguous duplicate keys."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"{where}: duplicate JSON key {key!r}")
            result[key] = value
        return result
    _require(isinstance(text, str), f"{where} must retain exact UTF-8 JSON text")
    try:
        return json.loads(text, object_pairs_hook=unique)
    except ValueError as exc:
        raise TesseraRouteReceiptError(f"{where}: {exc}") from exc


def _text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _current_scoped_contract():
    """Read the packaged contract through its existing dependency-free home."""
    from importlib.resources import as_file
    from . import tessera_runtime_contract as runtime
    from .lane_eligibility import load_eligibility_table, load_published_formats
    with as_file(runtime.contract_path()) as path:
        return (load_eligibility_table(contract_path=path),
                load_published_formats(contract_path=path))


def _declared_projections(scheme, where):
    """Decode the sidecar's explicit role rows, never guess a packed slice."""
    structure = scheme.get("structure", "dense")
    _require(structure in {"dense", "routed_moe"}, f"{where}: unsupported structure")
    if structure == "routed_moe":
        experts = scheme.get("experts")
        _require(type(experts) is int and experts > 0, f"{where}: invalid experts")
        groups = _object(scheme.get("groups"), where + ".groups")
        _require(bool(groups), f"{where}: empty groups")
    else:
        experts, groups = 1, {None: scheme}
    declarations = {}
    for group, spec in groups.items():
        _object(spec, f"{where}.{group}")
        roles, columns = spec.get("roles"), spec.get("columns")
        _require(isinstance(roles, list) and roles and type(columns) is int and columns > 0,
                 f"{where}: nonempty roles and positive columns required")
        rates = spec.get("q256")
        rates = rates if isinstance(rates, list) else [rates] * len(roles)
        _require(len(rates) == len(roles) and all(type(q) is int and q > 0 for q in rates),
                 f"{where}: role rates are missing or malformed")
        row_sum = 0
        for index, role in enumerate(roles):
            _require(isinstance(role, list) and len(role) == 2 and
                     isinstance(role[0], str) and role[0] and type(role[1]) is int and role[1] > 0,
                     f"{where}: invalid role declaration")
            row_sum += role[1]
            for expert in range(experts):
                key = (expert if structure == "routed_moe" else None, group, role[0])
                _require(key not in declarations, f"{where}: duplicate projection {key}")
                declarations[key] = {"rows": role[1], "cols": columns, "q256": rates[index],
                                     "grid": scheme.get("grid"), "family": scheme.get("family")}
        _require(spec.get("rows") == row_sum, f"{where}: role rows differ from group rows")
    return structure, declarations


def _priced_projection_population(binding, build, formats):
    from .layer_config import canonicalize_assignment, layer_config_metadata, strip_weight
    from .lane_eligibility import ServingContext, resolve_payload_rung
    from .tessera_serving_scope import ServingTarget
    recipe = _object(parse_census_json(binding.get("layer_config_json"), where="layer_config_json"), "recipe")
    _require(build.get("layer_config_sha") == _text_sha(binding["layer_config_json"]),
             "exact recipe text differs from independent card.build.layer_config_sha")
    names = [strip_weight(name) for name in recipe if name != "__prismaquant__"]
    _require(len(names) == len(set(names)), "recipe has duplicate normalized price units")
    assignment = canonicalize_assignment(recipe)
    selected = {name: fmt for name, fmt in assignment.items() if fmt.startswith("TESSERA_")}
    _require(bool(selected), "recipe has no Tessera-selected price units")
    scope = _object(layer_config_metadata(recipe).get("tessera_serving_scope"), "recipe serving scope")
    _require(set(scope) == {"target", "by_unit"} and scope == build.get("tessera_serving_scope"),
             "recipe serving scope differs from independent card.build.tessera_serving_scope")
    target = ServingTarget(**_object(scope["target"], "scope.target"))
    contexts = _object(scope["by_unit"], "scope.by_unit")
    config = _object(parse_census_json(binding.get("config_json"), where="config_json"), "config")
    manifest = _object(parse_census_json(binding.get("manifest_json"), where="manifest_json"), "manifest")
    _require("export_partition" not in manifest, "partition sidecars are not an assembled artifact")
    quant = _object(config.get("quantization_config"), "quantization_config")
    _require(quant.get("quant_method") == "tessera", "artifact is not Tessera")
    schemes = {}
    for key, group in _object(quant.get("config_groups"), "config_groups").items():
        _object(group, f"config_groups.{key}")
        targets = group.get("targets")
        _require(group.get("format") == "TESSERA" and isinstance(targets, list) and targets,
                 f"config_groups.{key}: exact Tessera targets required")
        for owner in targets:
            _require(isinstance(owner, str) and owner and owner not in schemes,
                     f"config_groups.{key}: duplicate or malformed owner")
            schemes[owner] = _object(group.get("scheme"), f"scheme.{owner}")
    modules = _object(manifest.get("modules"), "manifest.modules")
    _require(bool(schemes) and set(modules) == set(schemes), "config/manifest owner populations differ")
    by_unit, owners = {}, {}
    for owner, scheme in schemes.items():
        structure, expected = _declared_projections(scheme, owner)
        module = _object(modules[owner], f"manifest.{owner}")
        _require(module.get("structure", structure) == structure and
                 all(module.get(field) == scheme.get(field) for field in ("family", "grid")),
                 f"{owner}: manifest family/grid/structure differs from config")
        if structure == "routed_moe":
            _require(module.get("experts") == scheme["experts"], f"{owner}: expert population differs")
        roles = module.get("roles")
        _require(isinstance(roles, list) and roles, f"{owner}: manifest has no projections")
        seen, owner_units = set(), []
        for role in roles:
            _object(role, f"{owner} projection")
            key = (role.get("expert"), role.get("group"), role.get("role"))
            _require(key in expected and key not in seen, f"{owner}: extra or duplicate projection {key}")
            seen.add(key)
            _require(all(role.get(field) == value for field, value in expected[key].items()),
                     f"{owner}: projection {key} differs from declared shape/grid/rung")
            source = role.get("source_tensor", role.get("tensor"))
            _require(isinstance(source, str) and source.endswith(".weight"), f"{owner}: missing source tensor")
            _require(role.get("tensor") == source, f"{owner}: projected/packed source aliases are unsupported")
            if structure == "routed_moe":
                _require(role.get("source_layout") == "unpacked_per_expert" and
                         role.get("source_slice") == {"expert": role["expert"], "selector": "whole", "transpose": False},
                         f"{owner}: packed/aggregate source projection is unsupported")
            unit = strip_weight(source)
            _require(unit in selected and unit not in by_unit, f"{owner}: unpriced or duplicate source projection {unit}")
            context = ServingContext(**_object(contexts.get(unit), f"scope.by_unit.{unit}"))
            _require(context == target.context(structure), f"{unit}: price context differs from artifact structure/target")
            family, _k, rate = resolve_payload_rung(selected[unit], formats)
            _require(family in formats and rate == role["q256"] and formats[family].get("grid") == role["grid"],
                     f"{unit}: price family/grid/rung differs from written projection")
            by_unit[unit] = {"owner": owner, "format": selected[unit], "context": context.as_dict(),
                             "rows": role["rows"], "columns": role["cols"], "payload_family": family}
            owner_units.append(unit)
        _require(seen == set(expected), f"{owner}: missing declared projections")
        owners[owner] = {"structure": structure, "family": scheme["family"], "units": sorted(owner_units)}
    _require(set(by_unit) == set(selected), "not every Tessera-selected price unit has an artifact projection")
    return target, by_unit, owners


def check_scoped_route_receipt(census, binding, *, build, model_dir=None):
    """Replay v2 observations against current cells and independent artifact anchors.

    Nonidentity checkpoint/runtime mappings and packed source projections are
    explicit unsupported boundaries, never inferred from a similar name.
    """
    from . import lane_eligibility as lane
    from .shipcard import file_sha256
    try:
        _object(census, "route census")
        _object(binding, "census binding")
        _object(build, "card.build")
        _require(model_dir is not None, "scoped replay requires independent artifact files (model_dir)")
        _require(census.get("schema") == SCOPED_CENSUS_SCHEMA, "unknown scoped census schema")
        _require(set(binding) == {"layer_config_json", *_SIDECARS.values()}, "census binding fields differ from v2 contract")
        seals = {name: _text_sha(binding[field]) for name, field in _SIDECARS.items()
                 if isinstance(binding.get(field), str)}
        _require(len(seals) == len(_SIDECARS) and census.get("checkpoint_sidecars") == seals,
                 "census checkpoint sidecar seals differ from exact supplied file bytes")
        if model_dir is not None:
            _require(all(file_sha256(Path(model_dir) / name) == sha for name, sha in seals.items()),
                     "census sidecars differ from independently supplied artifact files")
        table, formats = _current_scoped_contract()
        _require(table.present and table.schema == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA,
                 f"scoped census needs a current {lane.LANE_ELIGIBILITY_SCHEMA_TESSERA} "
                 "eligibility table; legacy cells attest no image")
        target, units, owners = _priced_projection_population(binding, build, formats)
        _require(census.get("runtime") == {"image": target.runtime_image, "execution_mode": target.execution_mode}
                 and census.get("compiled") is (target.execution_mode == "compiled"),
                 "actual census runtime image/execution disagrees with price target")
        env = _object(census.get("env"), "census.env")
        _require(env.get("TESSERA_SERVE_MODE") == target.residency, "actual census residency differs from price target")
        capability = _object(census.get("device"), "census.device").get("capability")
        _require(isinstance(capability, list) and len(capability) == 2 and
                 all(type(n) is int and n >= 0 for n in capability) and
                 target.platform == f"sm_{capability[0]}{capability[1]}", "actual census platform differs from price target")
        _require(census.get("verdict") == "served" and census.get("problems") == [], "raw census did not pass its served checks")
        mapping = census.get("declared_name_mapping")
        _require((mapping is None and census.get("declared_names_mapped_to_module_space") is False) or
                 (mapping == {owner: owner for owner in owners} and
                  census.get("declared_names_mapped_to_module_space") is True),
                 "nonidentity or incomplete runtime owner mapping is unsupported; no mapper grammar is guessed")
        records = _object(census.get("records"), "census.records")
        owner_maps = _object(census.get("record_owner"), "census.record_owner")
        _require(set(records) == set(owner_maps) == set(CENSUS_PHASE_REGIMES) and
                 set(table.regimes) == set(CENSUS_PHASE_REGIMES.values()), "census must cover exactly both driven phases/all regimes")
        routes = {}
        for unit, row in units.items():
            facts = lane.unit_structural_facts(unit, row["format"], is_routed_moe=row["context"]["structure"] == "routed_moe",
                role_split=False, in_features=row["columns"], out_features=row["rows"], published_formats=formats)
            context = lane.ServingContext(**row["context"])
            # Source member dimensions are not necessarily the fused executed
            # shape. Match endpoint support: do not evaluate such predicates
            # until that projection has one shared, attested home.
            _require(not any(cell.predicates for cell in table.cells
                             if cell.family == facts.payload_family and cell.covers_rung(facts)
                             and lane.cell_matches_serving_context(cell, context)),
                     f"{unit}: executed-unit predicates are unsupported by this receipt projection")
            resolved = lane.resolve_unit_route(facts, table, **target.as_dict())
            _require(all(r.route_status in {lane.ROUTE_STATUS_BACKED, lane.ROUTE_STATUS_BACKED_WITH_SERVE_FLAG}
                         and r.qualification == lane.QUALIFICATION_DEVICE_QUALIFIED for r in resolved.regimes)
                     and len(resolved.regimes) == len(table.regimes), f"{unit}: current contract does not back every required regime")
            routes[unit] = {r.regime: r for r in resolved.regimes}
            row["regime_routes"] = {r.regime: r.as_dict() for r in resolved.regimes}
        served_routes, decoders, n_records = set(), set(), 0
        for phase, regime in CENSUS_PHASE_REGIMES.items():
            seen, replay_owners = set(), {}
            for name, record in _object(records[phase], f"records.{phase}").items():
                _object(record, f"{phase}.{name}")
                candidates = [name] if name in owners else [owner for owner in owners
                    if record.get("kind") == "moe" and name.startswith(owner + ".")]
                _require(len(candidates) == 1 and candidates[0] not in seen,
                         f"{phase}.{name}: extra, duplicate or ambiguous runtime owner")
                owner = candidates[0]
                seen.add(owner)
                replay_owners[name] = owner
                facts = owners[owner]
                structure = facts["structure"]
                kind = "moe" if structure == "routed_moe" else "dense"
                _require(record.get("kind") == kind and record.get("state") == "served" and
                         record.get("policy") == f"{facts['family']}:{target.residency}",
                         f"{phase}.{name}: structure/state/policy differs from written owner")
                if target.execution_mode == "eager":
                    shape = re.fullmatch(r"M([1-9][0-9]*):N([1-9][0-9]*):K([1-9][0-9]*)", str(record.get("shape", "")))
                    _require(shape is not None and (int(shape[1]) == 1) == (regime == "decode"),
                             f"{phase}.{name}: malformed shape or wrong driven regime")
                else:
                    _require(structure == "routed_moe" and str(record.get("shape", "")).startswith("M*:"),
                             f"{phase}.{name}: compiled dense/ambiguous trace agreement unsupported")
                symbol, decoder = record.get("symbol"), record.get("decoder")
                _require(isinstance(symbol, str) and symbol and isinstance(decoder, str) and decoder,
                         f"{phase}.{name}: missing observed launch pair")
                for unit in facts["units"]:
                    route = routes[unit][regime]
                    _require(record.get("contract") == route.activation_contract,
                             f"{phase}.{name}: activation contract mismatch")
                    _require(any(decoder == expected_decoder and (symbol == expected_symbol or
                                 (structure == "routed_moe" and symbol.startswith(expected_symbol + ":") and
                                  bool(symbol[len(expected_symbol) + 1:])))
                                 for expected_symbol, expected_decoder in route.executes),
                             f"{phase}.{name}: observed launch pair is not declared by its scoped cell")
                    for flag in route.requires_serve_flags:
                        key, _, values = flag.partition("=")
                        _require(env.get(key) in values.split("|"), f"{phase}.{name}: required serve flag {flag} not observed")
                served_routes.add(facts["family"])
                decoders.add(decoder)
                n_records += 1
            _require(seen == set(owners) and owner_maps[phase] == replay_owners,
                     f"{phase}: incomplete owner bijection or forged recorded owner map")
        return {"schema": "prismaquant.tessera-scoped-census.v1", "passed": True,
                "target": target.as_dict(), "by_unit": units, "owners": owners,
                "served_routes": sorted(served_routes), "served_decoders": sorted(decoders),
                "n_records": n_records, "contract": table.provenance(),
                "detail": "route census agrees with exact price/artifact/runtime scope in both driven phases"}
    except (TypeError, KeyError, ValueError, OSError, ImportError) as exc:
        if isinstance(exc, TesseraRouteReceiptError):
            raise
        raise TesseraRouteReceiptError(f"malformed scoped route census: {exc}") from exc


__all__ = [
    "TesseraRouteReceiptError",
    "check_route_receipt",
    "parse_route_records",
    "substitute_decoders_from_contract_answer",
]
