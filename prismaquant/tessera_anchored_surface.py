"""Research replay of receipt-bound Tessera costs through shared anchored shapes.

This reader verifies recorded campaign identities and blob hashes. It does not
recompute source tensors, validate producer wire grammar, qualify a serving lane,
or emit allocator input. The current campaign importer accepts output MSE only;
the neutral numerical core never turns that quantity into AURA.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections.abc import Mapping
from pathlib import Path

from .anchored_shape import (
    AnchoredShapeError, LogShapeObservation, audit_anchored,
    fit_anchor_correction, fit_centered_log_shape, predict_anchored,
)
from .cost_stage_checkpoint import (
    MANIFEST_SCHEMA, _load_unit, canonical_json_sha256, unit_path,
)

PLAN_SCHEMA = "prismaquant.tessera_anchored_replay.plan.v1"
REPORT_SCHEMA = "prismaquant.tessera_anchored_replay.report.v1"
CAMPAIGN_SCHEMA = "prismaquant.tessera_campaign_cost.v1"
CURRENCY = "output_mse_under_route_activation_contract"


class ReplayError(ValueError):
    """An explicit replay input or its recorded identity is inconsistent."""


def _require(condition, message):
    if not condition:
        raise ReplayError(message)


def _digest(value):
    return canonical_json_sha256(value, where="anchored replay identity")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive(value):
    return (type(value) in (float, int) and math.isfinite(value) and value > 0)


def load_campaign_measurements(cost_path, checkpoint_path, plan):
    """Read trusted local pickle artifacts, bound to the plan's exact hashes.

    Uses the existing journal reader and existing producer receipt fields.
    No journal, wire, source, or campaign artifact is created or rewritten.
    """
    cost_path, checkpoint_path = Path(cost_path), Path(checkpoint_path)
    binding = plan["input"]
    payload_bytes = cost_path.read_bytes()
    _require(hashlib.sha256(payload_bytes).hexdigest() == binding["payload_sha256"], "payload hash mismatch")
    manifest = json.loads(checkpoint_path.read_text())
    _require(manifest.get("schema") == MANIFEST_SCHEMA
             and manifest.get("stage") == "Tessera campaign", "unsupported checkpoint")
    identity = manifest["identity"]
    seal = _digest(identity)
    _require(seal == manifest.get("identity_sha256")
             == binding["checkpoint_identity_sha256"], "checkpoint identity mismatch")
    _require(identity.get("campaign_schema") == CAMPAIGN_SCHEMA
             and identity.get("currency") == CURRENCY, "unsupported campaign currency")
    for name in ("prismaquant_source_sha256", "encoder_source_sha256"):
        value = identity.get(name)
        _require(isinstance(value, str) and len(value) == 64
                 and all(c in "0123456789abcdef" for c in value), f"missing {name}")
    _require(isinstance(identity.get("calibration"), dict)
             and bool(identity["calibration"]), "missing calibration identity")
    payload = pickle.loads(payload_bytes)
    _require(payload.get("schema") == CAMPAIGN_SCHEMA
             and payload.get("currency") == CURRENCY == plan["currency"],
             "current campaign importer only accepts recorded output-MSE currency")
    provenance = payload["provenance"]
    _require(provenance.get("cost_mode") == "production-render-score",
             "campaign is not attested production-render-score")
    wire_dir = Path(provenance["wire_dir"])
    journal = checkpoint_path.with_name(checkpoint_path.name + ".parts")
    units = identity["units"]
    roster = manifest.get("units", [])
    _require({item["qname"] for item in roster} == set(units)
             and len(roster) == len(units), "checkpoint unit roster mismatch")
    measurements = {}
    receipts = {}
    for unit, rows in sorted(payload["costs"].items()):
        _require(unit in units, f"unknown source unit {unit}")
        state = _load_unit(unit_path(journal, unit), stage="Tessera campaign",
                           qname=unit, identity_sha256=seal)
        _require(set(state) == {"anchors", "wire_records"}, "invalid anchor state")
        anchors = {row["format_name"]: row for row in state["anchors"]}
        _require(len(anchors) == len(state["anchors"])
                 and set(anchors) == set(state["wire_records"]), "anchor receipt coverage mismatch")
        for key, row in sorted(rows.items()):
            if row.get("output_mse_measured") is not True:
                continue
            _require(key in anchors, f"unreceipted measured row {unit}/{key}")
            anchor = anchors[key]
            _require(anchor["qname"] == unit and key in units[unit]["menu"],
                     "anchor is outside checkpoint unit/menu")
            _require(row.get("currency") == CURRENCY
                     and row.get("cost_source") == "tessera_campaign_measured"
                     and row.get("tessera_provenance") == "measured", "measured provenance mismatch")
            for target, source in (("output_mse", "dloss"), ("tessera_family", "family"),
                                   ("tessera_body_rate_q256", "body_rate_q256"),
                                   ("activation_contract", "activation_contract"),
                                   ("activation_quantized", "activation_quantized"),
                                   ("wire_bytes", "wire_bytes")):
                _require(row.get(target) == anchor.get(source), f"anchor {target} mismatch")
            static_scale = units[unit].get("input_global_scale")
            _require(row.get("input_global_scale") == anchor.get("input_global_scale")
                     and (anchor.get("input_global_scale") is None
                          or anchor["input_global_scale"] == static_scale), "static-scale identity mismatch")
            hessian = row.get("hessian_identity", {})
            _require(hessian.get("applied") == anchor.get("hessian_applied"),
                     "Hessian applicability mismatch")
            for field in ("supplied", "capture_sha256", "text_sha256", "fit_ids_sha256", "fit_tokens"):
                _require(hessian.get(field) == provenance.get("hessian", {}).get(field),
                         f"Hessian {field} mismatch")
            record = state["wire_records"][key]
            recorded = record["identity"]
            _require(recorded.get("unit") == unit
                     and recorded.get("source") == units[unit]["weight"]
                     and recorded.get("encoder_source_sha256") == identity["encoder_source_sha256"],
                     "recorded source/encoder identity mismatch")
            calibration = recorded.get("calibration")
            if anchor.get("hessian_applied"):
                _require(isinstance(calibration, dict)
                         and calibration.get("hessian") == units[unit]["hessian"],
                         "recorded Hessian identity mismatch")
            else:
                _require(calibration is None, "H-free anchor has an H-aware receipt")
            recipe = dict(recorded["recipe"])
            _require(recipe.pop("q256", None) == anchor["body_rate_q256"], "wire rate mismatch")
            filename = record["file"]
            _require(isinstance(filename, str) and Path(filename).name == filename
                     and filename not in {".", ".."}, "wire filename escapes directory")
            wire = wire_dir / filename
            _require(not wire.is_symlink() and wire.resolve().parent == wire_dir.resolve(),
                     "wire path escapes directory")
            _require(wire.stat().st_size == record["blob_bytes"] == anchor["wire_bytes"]
                     and _sha(wire) == record["blob_sha256"], "wire blob hash/size mismatch")
            measurement = {
                "value": row["output_mse"], "currency": CURRENCY,
                "coordinate": anchor["body_rate_q256"],
                "family": row["tessera_family"], "activation_contract": row["activation_contract"],
                "geometry": units[unit]["weight"]["shape"], "wire_recipe": recipe,
                "hessian_applied": anchor["hessian_applied"],
            }
            measurements[(unit, key)] = measurement
            receipts[f"{unit}|{key}"] = _digest(record)
    return measurements, {
        "payload_sha256": binding["payload_sha256"], "checkpoint_identity_sha256": seal,
        "recorded_receipts_sha256": _digest(receipts),
        "recorded_menus": {unit: row["menu"] for unit, row in units.items()},
        "source_recomputed": False, "producer_wire_grammar_revalidated": False,
        "verification": "journal_identity_and_recorded_source_and_blob_hashes",
    }, provenance.get("anchor_groups", {})


def replay_measurements(measurements, plan, *, input_identity, groups=None):
    """Run a declared split; return research predictions and missing work only.

    ``measurements`` is the checked importer output. The explicit currency is
    carried unchanged; this function provides no mapping between currencies.
    """
    _require(plan.get("schema") == PLAN_SCHEMA, "unsupported replay plan")
    _require(isinstance(plan.get("currency"), str) and bool(plan["currency"]), "missing currency")
    reports = []
    requests = set()
    seen = set()
    for segment in plan["segments"]:
        label = segment["id"]
        _require(label not in seen, "duplicate segment id")
        seen.add(label)
        descriptor = segment["descriptor"]
        _require(isinstance(descriptor.get("role"), str) and bool(descriptor["role"]),
                 "segment needs an explicit profile role declaration")
        features = segment["features_by_key"]
        coordinates = segment["coordinates"]
        _require(set(features) == set(coordinates) and len(coordinates) >= 3, "invalid segment domain")
        _require(all(type(v) is int and v > 0 for v in coordinates.values())
                 and len(set(coordinates.values())) == len(coordinates), "invalid rate coordinates")
        domain = sorted(coordinates, key=lambda key: (coordinates[key], key))
        pilots, heldout = segment["pilot"], segment["heldout"]
        _require(bool(pilots) and bool(heldout) and not set(pilots) & set(heldout),
                 "pilot and held-out units must be nonempty and disjoint")
        for unit in set(pilots) | set(heldout):
            _require(set(domain) <= set(input_identity["recorded_menus"].get(unit, [])),
                     f"declared domain exceeds recorded source menu for {unit}")
        threshold = segment["max_absolute_log10_error"]
        _require(type(threshold) in (int, float) and math.isfinite(threshold) and threshold >= 0,
                 "invalid audit threshold")
        _require(type(segment.get("refit_after_audit", False)) is bool, "invalid refit policy")

        def read(unit, key):
            _require(key in coordinates, "requested rung outside declared domain")
            row = measurements.get((unit, key))
            if row is None:
                return None
            _require(row.get("currency") == plan["currency"], "measurement currency mismatch")
            for field in ("family", "activation_contract", "geometry", "wire_recipe", "hessian_applied"):
                _require(field in descriptor and row.get(field) == descriptor[field],
                         f"cross-segment {field}: {unit}/{key}")
            _require(row["coordinate"] == coordinates[key], "coordinate identity mismatch")
            return row["value"]

        observations = []
        pilot_missing = []
        for unit, keys in sorted(pilots.items()):
            _require(len(keys) >= 2 and len(set(keys)) == len(keys), "invalid pilot rungs")
            for key in keys:
                value = read(unit, key)
                if not _positive(value):
                    pilot_missing.append((unit, key))
                else:
                    observations.append(LogShapeObservation(unit, key, value))
        pilot_coordinates = [coordinates[key] for keys in pilots.values() for key in keys]
        _require(min(pilot_coordinates) == min(coordinates.values())
                 and max(pilot_coordinates) == max(coordinates.values()),
                 "pilot coordinate envelope must cover the declared domain")
        fit_reason = None
        shape = None
        if not pilot_missing:
            try:
                shape = fit_centered_log_shape(observations, features)
            except AnchoredShapeError as exc:
                fit_reason = str(exc)
        else:
            fit_reason = "missing_or_invalid_pilot_measurements"
            requests.update((label, unit, key) for unit, key in pilot_missing)
        units_report = {}
        for unit, split in sorted(heldout.items()):
            anchor_keys, audit_keys = split["anchors"], split["audit"]
            _require(1 <= len(anchor_keys) <= 2 and len(set(anchor_keys)) == len(anchor_keys),
                     "held-out unit needs one or two distinct anchors")
            _require(bool(audit_keys) and len(set(audit_keys)) == len(audit_keys)
                     and not set(anchor_keys) & set(audit_keys), "audit must be disjoint from anchors")
            needed = {key: read(unit, key) for key in anchor_keys + audit_keys}
            missing = [key for key, value in needed.items() if not _positive(value)]
            invalid_extra = [key for key in domain if (unit, key) in measurements
                             and not _positive(read(unit, key))]
            result = {"status": "measure_more", "anchors": list(anchor_keys), "audit": [],
                      "predictions": {}, "refit_after_audit": False,
                      "pwl_audit": []}
            for key in audit_keys:
                if len(anchor_keys) != 2 or any(not _positive(needed[k]) for k in anchor_keys):
                    result["pwl_audit"].append({"key": key, "unavailable": "needs_two_valid_anchors"})
                    continue
                left, right = sorted(anchor_keys, key=coordinates.get)
                if needed[right] >= needed[left]:
                    result["pwl_audit"].append({"key": key, "unavailable": "nonmonotone_anchors"})
                    continue
                if not coordinates[left] < coordinates[key] < coordinates[right]:
                    result["pwl_audit"].append({"key": key, "unavailable": "outside_anchor_envelope"})
                    continue
                fraction = (coordinates[key] - coordinates[left]) / (coordinates[right] - coordinates[left])
                predicted = 10 ** (math.log10(needed[left]) * (1 - fraction)
                                   + math.log10(needed[right]) * fraction)
                result["pwl_audit"].append({
                    "key": key, "predicted": predicted,
                    "measured": needed[key] if _positive(needed[key]) else None,
                    "absolute_log10_error": (abs(math.log10(predicted) - math.log10(needed[key]))
                                              if _positive(needed[key]) else None),
                })
            if missing or invalid_extra:
                requests.update((label, unit, key) for key in set(missing + invalid_extra))
                result["reason"] = "missing_or_invalid_anchor_or_audit"
            elif shape is None:
                result["reason"] = fit_reason
                requests.update((label, unit, key) for key in domain)
            else:
                try:
                    anchors = {key: needed[key] for key in anchor_keys}
                    correction = fit_anchor_correction(shape, anchors, coordinates)
                    audit = audit_anchored(correction, {key: needed[key] for key in audit_keys})
                    result["audit"] = [dict(key=row.key, predicted=row.predicted, measured=row.measured,
                                            absolute_log10_error=row.absolute_log10_error,
                                            held_out_axes=(["unit", "rung"] if row.key not in
                                                           {key for keys in pilots.values() for key in keys}
                                                           else ["unit"]))
                                       for row in audit.rows]
                    passed = audit.max_absolute_log10_error <= threshold
                    predictions = {key: predict_anchored(correction, key) for key in domain}
                    passed = passed and all(predictions[b] <= predictions[a]
                                            for a, b in zip(domain, domain[1:]))
                    if passed and segment.get("refit_after_audit", False):
                        correction = fit_anchor_correction(shape, needed, coordinates)
                        predictions = {key: predict_anchored(correction, key) for key in domain}
                        passed = all(predictions[b] <= predictions[a] for a, b in zip(domain, domain[1:]))
                        result["refit_after_audit"] = passed
                    if passed:
                        # Preserve every observed domain row; never relabel a prediction as measured.
                        result["predictions"] = {
                            key: {"value": value if _positive(value := read(unit, key)) else predictions[key],
                                  "source": "measured" if _positive(value) else "predicted"}
                            for key in domain
                        }
                        mixed = result["predictions"]
                        if all(mixed[b]["value"] <= mixed[a]["value"] for a, b in zip(domain, domain[1:])):
                            result["status"] = "audit_accepted_research_only"
                        else:
                            result["predictions"] = {}
                            result["refit_after_audit"] = False
                            result["reason"] = "measured_overlay_nonmonotone"
                            requests.update((label, unit, key) for key in domain)
                    else:
                        result["reason"] = "audit_or_monotonicity_failed"
                        requests.update((label, unit, key) for key in domain)
                except (AnchoredShapeError, OverflowError) as exc:
                    result["reason"] = str(exc)
                    requests.update((label, unit, key) for key in domain)
            units_report[unit] = result
        reports.append({"id": label, "descriptor": descriptor, "units": units_report,
                        "pilot_fit_error": fit_reason,
                        "pilot_fit": (None if shape is None else {
                            "coefficients_log10": list(shape.coefficients),
                            "reference_key": shape.reference_key, "design_rank": shape.design_rank,
                            "n_units": shape.n_units, "n_observations": shape.n_observations,
                        })})
    # Existing campaign group placement is retained even when only one sibling fails.
    for label, unit, key in sorted(requests.copy()):
        for members in (groups or {}).values():
            if unit in members:
                _require(all(key in input_identity["recorded_menus"].get(member, []) for member in members),
                         "fused-group measurement request exceeds a recorded member menu")
                requests.update((label, member, key) for member in members)
    return {
        "schema": REPORT_SCHEMA, "currency": plan["currency"], "plan_sha256": _digest(plan),
        "input_identity": {key: value for key, value in input_identity.items() if key != "recorded_menus"},
        "production_qualified": False,
        "allocator_payload": False, "latency_evidence": None,
        "role_partition": "explicit_plan_declaration", "segments": reports,
        "measurement_requests": [dict(segment=label, unit=unit, key=key)
                                 for label, unit, key in sorted(requests)],
    }


def replay_campaign(cost_path, checkpoint_path, plan):
    _require(plan.get("schema") == PLAN_SCHEMA, "unsupported replay plan")
    measurements, identity, groups = load_campaign_measurements(cost_path, checkpoint_path, plan)
    return replay_measurements(measurements, plan, input_identity=identity, groups=groups)
