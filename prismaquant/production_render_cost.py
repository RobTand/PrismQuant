#!/usr/bin/env python3
"""Build allocator costs from production-rendered reconstruction losses.

The local cost path estimates layer damage from a separate RTN-style cost
measurement and then, for measured output MSE, multiplies by
``0.5 * h_trace``.  Production cache fill already renders the weights the
export will ship and records the scorer objective used to accept GPTQ/JSO
and scale-sweep candidates.  This module turns those rendered scores into
allocator-compatible ``predicted_dloss`` entries.

The synthesized entries intentionally set ``output_mse_measured=False`` so
``allocator_candidates.cost_entry_predicted_dloss`` consumes
``predicted_dloss`` directly instead of applying the diagonal-Fisher proxy
again.
"""
from __future__ import annotations

import argparse
import math
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from prismaquant import format_registry as fr
from prismaquant.name_projection import strip_weight_leaf

if TYPE_CHECKING:
    from prismaquant.source_class_format_plan import SourceClassFormatPlan


SCHEMA = "prismaquant.production_render_score_cost.v1"


def canonical_cost_name(qname: str) -> str:
    """Normalize a producer-recorded qname to the recipe-unit spelling.

    The leaf half is the shared layer's one leaf function
    (`name_projection.strip_weight_leaf`). The umbrella-infix half mirrors
    the base profile's checkpoint→live rule (`model.language_model.` →
    `model.`, `model_profiles/base.py` `checkpoint_to_live_name`) and is
    deliberately NOT routed through a projection: this module never sees a
    profile, and a checkpoint/live projection may DECLINE a key (visual,
    scale siblings) while cost/render payloads legitimately carry rows this
    normalizer must keep — DSv4's MTP units are costed and keyed under their
    physical spelling (deepseek_v4 fp8_scale_pairs retains it for the same
    reason). Total by contract: every input comes back, spelled one way.
    """
    name = strip_weight_leaf(str(qname))
    prefix = "model.language_model."
    if name.startswith(prefix):
        name = "model." + name[len(prefix):]
    return name


def _record_key(qname: str, fmt: str) -> str:
    return f"{qname}|{fmt.upper()}"


def _load_pickle(path: str | Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _cache_render_score_records(cache: object) -> dict[tuple[str, str], dict]:
    meta = getattr(cache, "metadata", None)
    if not isinstance(meta, Mapping):
        return {}
    scores = meta.get("render_scores")
    if not isinstance(scores, Mapping):
        return {}
    records = scores.get("records")
    if not isinstance(records, Mapping):
        return {}

    out: dict[tuple[str, str], dict] = {}
    for raw_key, raw_record in records.items():
        if not isinstance(raw_record, Mapping):
            continue
        qname = canonical_cost_name(str(raw_record.get("qname", "")))
        fmt = fr.canonical_format_name(str(raw_record.get("format", "")))
        if not qname or not fmt:
            parts = str(raw_key).rsplit("|", 1)
            if len(parts) == 2:
                qname = canonical_cost_name(parts[0])
                fmt = fr.canonical_format_name(parts[1])
        if not qname or not fmt:
            continue
        out[(qname, fmt)] = dict(raw_record)
    return out


def _attested_transient_cb_pairs(cache: object) -> set[tuple[str, str]]:
    """Validate score-only CB artifacts and return their admitted pair scope.

    A bare ``render_scores`` row is deliberately insufficient.  The transient
    producer must bind it to the canonical rendered tensor, synchronous
    consumer result, exact pair identity, and the complete artifact-set digest.
    """
    meta = getattr(cache, "metadata", None)
    if not isinstance(meta, Mapping):
        return set()
    transient = meta.get("transient_render_artifacts")
    if not isinstance(transient, Mapping):
        return set()
    records = transient.get("records")
    consumer_identity = transient.get("consumer_identity")
    pair_set = meta.get("cb_cache_pair_identity")
    if (
        transient.get("schema")
        != "prismaquant.production_weight_cache.transient_render_artifacts.v1"
        or not isinstance(records, Mapping)
        or not isinstance(consumer_identity, Mapping)
        or not isinstance(pair_set, Mapping)
    ):
        raise ValueError("malformed transient CB render artifact manifest")

    from prismaquant.production_weight_cache import (
        CB_TRANSIENT_CONSUMER_RECEIPT_SCHEMA,
        _canonical_json_sha256,
        first_identity_difference,
    )

    if int(transient.get("entries", -1)) != len(records):
        raise ValueError("transient CB render artifact entry count differs")
    if int(pair_set.get("published_entries", -1)) != len(records):
        raise ValueError("transient CB pair publication count differs")
    observed_artifact_sha256 = _canonical_json_sha256(
        records,
        where="transient CB render artifact set",
    )
    if pair_set.get("artifact_sha256") != observed_artifact_sha256:
        raise ValueError("transient CB render artifact set digest differs")

    score_records = _cache_render_score_records(cache)
    admitted: set[tuple[str, str]] = set()
    for raw_key, artifact in records.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"malformed transient CB artifact {raw_key!r}")
        identity = artifact.get("identity")
        tensor = artifact.get("tensor")
        render_score = artifact.get("render_score")
        receipt = artifact.get("consumer_receipt")
        if not all(isinstance(value, Mapping) for value in (
            identity, tensor, render_score, receipt,
        )):
            raise ValueError(
                f"transient CB artifact {raw_key!r} is not value-bearing"
            )
        qname = canonical_cost_name(str(identity.get("qname", "")))
        fmt = fr.canonical_format_name(str(identity.get("format", "")))
        expected_binding = {
            "schema": CB_TRANSIENT_CONSUMER_RECEIPT_SCHEMA,
            "qname": str(identity.get("qname", "")),
            "format": fmt,
            "consumer_identity": dict(consumer_identity),
            "tensor": dict(tensor),
            "render_score_sha256": _canonical_json_sha256(
                render_score,
                where=f"transient CB render score {raw_key}",
            ),
        }
        observed_binding = {
            field: receipt.get(field) for field in expected_binding
        }
        if first_identity_difference(
            observed_binding,
            expected_binding,
            path="consumer_receipt",
        ) is not None:
            raise ValueError(
                f"transient CB consumer receipt differs for {raw_key}"
            )
        result = receipt.get("result")
        if not isinstance(result, Mapping) or receipt.get(
            "result_sha256"
        ) != _canonical_json_sha256(
            result,
            where=f"transient CB consumer result {raw_key}",
        ):
            raise ValueError(
                f"transient CB consumer result digest differs for {raw_key}"
            )
        manifest_score = score_records.get((qname, fmt))
        if (
            manifest_score is None
            or _canonical_json_sha256(
                manifest_score,
                where=f"cache render score {raw_key}",
            )
            != expected_binding["render_score_sha256"]
        ):
            raise ValueError(
                f"transient CB render score manifest differs for {raw_key}"
            )
        admitted.add((qname, fmt))
    return admitted


def _calibration_hashes(*sources: object) -> list[str]:
    """Union of R14 calibration identities carried by upstream artifacts.

    This module never sees calibration text — its cost comes from render
    scores the production cache already recorded — so the hash it stamps is
    inherited from the cache metadata (stamped by ``build_production_cache``)
    and from the baseline cost payload's meta/provenance. Returns ``[]`` when
    no upstream stamped one, which keeps the downstream disjointness check
    inert on pre-R14 artifacts rather than guessing.
    """
    found: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        single = source.get("calib_hash")
        if isinstance(single, str) and single:
            found.add(single)
        many = source.get("calib_hashes")
        if isinstance(many, Sequence) and not isinstance(many, (str, bytes)):
            found.update(str(item) for item in many if item)
    return sorted(found)


def _lookup_record(
    records: Mapping[tuple[str, str], Mapping],
    qname: str,
    fmt: str,
) -> Mapping | None:
    cname = canonical_cost_name(qname)
    for alias in fr.aliases_for(fmt):
        record = records.get((cname, fr.canonical_format_name(alias)))
        if record is not None:
            return record
    return None


def _score_value(record: Mapping, field: str) -> float | None:
    value = record.get(field)
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out) or out < 0.0:
        return None
    return out


def _production_cost_entry(
    record: Mapping,
    *,
    score_field: str,
) -> dict | None:
    metric = str(record.get("metric", "render_score"))
    if score_field in {"weight_mse_sum", "weight_mse"}:
        weight_mse = _score_value(record, "weight_mse")
        if weight_mse is None:
            return None
        # Hand the allocator the production-rendered weight_mse and leave
        # predicted_dloss/output_mse off the entry so its existing
        # h_trace * weight_mse fallback fires.
        return {
            "weight_mse": float(weight_mse),
            "output_mse_measured": False,
            "cost_source": "production_render_weight_mse",
            "render_score_metric": metric,
            "render_score": _score_value(record, "score"),
            "render_score_sum": _score_value(record, "score_sum"),
            "render_score_normalizer": _score_value(record, "normalizer"),
            "render_activation_rows": int(record.get("activation_rows", 0) or 0),
            "raw_render_metric": str(record.get("raw_render_metric", "")),
            "raw_render_score": _score_value(record, "raw_render_score"),
            "raw_render_score_sum": _score_value(record, "raw_render_score_sum"),
            "weight_mse_sum": _score_value(record, "weight_mse_sum"),
            "n_weights": int(record.get("n_weights", 0) or 0),
            "activation_quantized": bool(record.get("activation_quantized", False)),
            "activation_clipped": bool(record.get("activation_clipped", False)),
        }
    if score_field == "output_mse":
        # Use the production-rendered per-element output_mse (unweighted,
        # equals raw_render_score after Fisher row-weighting was dropped).
        # output_mse_measured=True triggers the allocator's path 1
        # (h_trace * output_mse), matching the original prismaquant cost
        # objective but on production-quality rendered weights instead of
        # naive RTN.
        output_mse = _score_value(record, "raw_render_score")
        if output_mse is None:
            output_mse = _score_value(record, "score")
        if output_mse is None:
            return None
        return {
            "output_mse": float(output_mse),
            "output_mse_measured": True,
            "weight_mse": float(_score_value(record, "weight_mse") or 0.0),
            "cost_source": "production_render_output_mse",
            "render_score_metric": metric,
            "render_score": _score_value(record, "score"),
            "render_score_sum": _score_value(record, "score_sum"),
            "render_score_normalizer": _score_value(record, "normalizer"),
            "render_activation_rows": int(record.get("activation_rows", 0) or 0),
            "raw_render_metric": str(record.get("raw_render_metric", "")),
            "raw_render_score": _score_value(record, "raw_render_score"),
            "raw_render_score_sum": _score_value(record, "raw_render_score_sum"),
            "weight_mse_sum": _score_value(record, "weight_mse_sum"),
            "n_weights": int(record.get("n_weights", 0) or 0),
            "activation_quantized": bool(record.get("activation_quantized", False)),
            "activation_clipped": bool(record.get("activation_clipped", False)),
        }
    score = _score_value(record, score_field)
    mean_score = _score_value(record, "score")
    if score is None:
        return None
    return {
        "predicted_dloss": float(score),
        "weight_mse": float(_score_value(record, "weight_mse") or 0.0),
        "output_mse": float(mean_score if mean_score is not None else 0.0),
        "rel_output_mse": 0.0,
        "output_mse_measured": False,
        "cost_source": "production_render_score",
        "render_score_metric": metric,
        "render_score": float(mean_score if mean_score is not None else score),
        "render_score_sum": _score_value(record, "score_sum"),
        "render_score_normalizer": _score_value(record, "normalizer"),
        "render_activation_rows": int(record.get("activation_rows", 0) or 0),
        "raw_render_metric": str(record.get("raw_render_metric", "")),
        "raw_render_score": _score_value(record, "raw_render_score"),
        "raw_render_score_sum": _score_value(record, "raw_render_score_sum"),
        "activation_quantized": bool(record.get("activation_quantized", False)),
        "activation_clipped": bool(record.get("activation_clipped", False)),
    }


def _validated_format_plan_scope(
    production_cache: object,
    baseline_costs: Mapping[str, object],
    output_formats: Sequence[str],
    format_plan: "SourceClassFormatPlan",
) -> tuple[dict[str, frozenset[str]], frozenset[str]]:
    """Bind pricing to the exact source-class plan used by the renderer.

    The format plan partitions only its declared family. Formats outside that
    family (for example BF16/source terminals or a separately declared family)
    remain global. Within the planned family, every baseline unit receives
    exactly its declared menu: no illegal higher-rate cell and no
    demand-truncated legal rung.
    """
    metadata = getattr(production_cache, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError(
            "--format-plan requires cache metadata with a bound plan identity"
        )
    expected_identity = str(format_plan.identity_sha256)
    observed_identity = metadata.get("format_plan_identity_sha256")
    if observed_identity != expected_identity:
        raise ValueError(
            "production cache format-plan identity mismatch: "
            f"expected={expected_identity!r} observed={observed_identity!r}"
        )

    scope: dict[str, frozenset[str]] = {}
    for raw_qname, raw_formats in format_plan.formats_by_qname().items():
        qname = canonical_cost_name(raw_qname)
        formats = frozenset(
            fr.canonical_format_name(str(fmt)) for fmt in raw_formats
        )
        previous = scope.setdefault(qname, formats)
        if previous != formats:
            raise ValueError(
                "format plan contains colliding canonical qnames with "
                f"different menus: {qname}"
            )
    planned_universe = frozenset(
        fmt for formats in scope.values() for fmt in formats
    )
    if not planned_universe:
        raise ValueError("format plan has an empty planned format universe")

    requested_planned = frozenset(
        fmt for fmt in output_formats if fmt in planned_universe
    )
    if requested_planned != planned_universe:
        raise ValueError(
            "requested formats truncate the source-class format plan: "
            f"missing={sorted(planned_universe - requested_planned)}"
        )

    baseline_names: dict[str, str] = {}
    for raw_qname in baseline_costs:
        qname = canonical_cost_name(str(raw_qname))
        previous = baseline_names.setdefault(qname, str(raw_qname))
        if previous != str(raw_qname):
            raise ValueError(
                "baseline costs contain colliding canonical qnames: "
                f"{previous!r} and {raw_qname!r}"
            )
    missing_baseline = sorted(set(scope) - set(baseline_names))
    if missing_baseline:
        raise ValueError(
            "baseline costs do not cover every format-plan unit; sample="
            f"{missing_baseline[:8]}"
        )
    unplanned_baseline = sorted(set(baseline_names) - set(scope))
    if unplanned_baseline:
        raise ValueError(
            "baseline costs contain units absent from the format plan; "
            f"sample={unplanned_baseline[:8]}"
        )

    # Refuse a cache that claims this plan identity but nevertheless recorded
    # an illegal planned-family render. Ignoring such a row would hide illegal
    # work and let a future consumer accidentally revive it.
    for (qname, fmt) in _cache_render_score_records(production_cache):
        if fmt not in planned_universe:
            continue
        allowed = scope.get(qname)
        if allowed is None or fmt not in allowed:
            raise ValueError(
                "production cache contains a render outside its source-class "
                f"format plan: {qname}@{fmt}"
            )
    return scope, planned_universe


def synthesize_production_render_cost_payload(
    production_cache: object,
    baseline_cost_payload: Mapping,
    *,
    formats: Sequence[str] | None = None,
    score_field: str = "score_sum",
    source_label: str | None = None,
    require_render_scores: bool = False,
    require_output_metric: bool = False,
    format_plan: "SourceClassFormatPlan | None" = None,
) -> dict:
    baseline_costs = dict(baseline_cost_payload["costs"])
    output_formats = [
        fr.canonical_format_name(str(fmt))
        for fmt in (
            formats
            if formats is not None
            else baseline_cost_payload.get("formats", [])
        )
    ]
    output_formats = list(dict.fromkeys(output_formats))

    planned_scope: dict[str, frozenset[str]] | None = None
    planned_universe: frozenset[str] = frozenset()
    if format_plan is not None:
        planned_scope, planned_universe = _validated_format_plan_scope(
            production_cache,
            baseline_costs,
            output_formats,
            format_plan,
        )

    cb_context = None
    cb_render_provenance: dict[str, object] = {}
    valid_cb_render_records: set[tuple[str, str]] = set()
    if any(
        fr.get_format(fmt).family in {"nvfp4_cb", "fp8_cb"}
        for fmt in output_formats
    ):
        from prismaquant.nvfp4_cb_footprint import validate_cb_cost_provenance
        from prismaquant.production_weight_cache import (
            production_cache_cb_render_provenance,
        )

        cb_render_provenance = production_cache_cb_render_provenance(
            production_cache,
            require_for_formats=output_formats,
            where="production render cost cache",
        )
        from prismaquant.nvfp4_cb_footprint import (
            cb_serialization_context_from_stamp,
        )

        cb_context = cb_serialization_context_from_stamp(
            cb_render_provenance["cb_serialized_payload"],
            where="production render cost cache",
        )
        # Fallback rows still come from the baseline table.  Both sources must
        # describe the same serialized CB artifact before their rows can be
        # combined under one provenance stamp.
        validate_cb_cost_provenance(
            baseline_cost_payload,
            output_formats,
            context=cb_context,
            where="production render baseline cost",
        )
        identity_scope = cb_render_provenance[
            "cb_render_identity"
        ]["cb_formats_by_qname"]
        identity_pairs = {
            (canonical_cost_name(qname), fr.canonical_format_name(fmt))
            for qname, formats_for_qname in identity_scope.items()
            for fmt in formats_for_qname
        }
        cache_pairs = {
            (canonical_cost_name(qname), fr.canonical_format_name(fmt))
            for qname, fmt in (getattr(production_cache, "weights", {}) or {})
            if fr.get_format(fr.canonical_format_name(fmt)).family
            in {"nvfp4_cb", "fp8_cb"}
        }
        transient_pairs = _attested_transient_cb_pairs(production_cache)
        # A score is usable only when both the value-bearing identity and an
        # actual admitted cache tensor cover the row.  This prevents an old
        # render_scores.json entry (including one left after a failed fresh
        # render) from being relabeled under today's identity.
        valid_cb_render_records = identity_pairs & (
            cache_pairs | transient_pairs
        )

    records = _cache_render_score_records(production_cache)

    output_costs: dict[str, dict[str, dict]] = {}
    render_entries = 0
    fallback_entries = 0
    missing: list[dict[str, str]] = []
    non_output_metric: list[dict[str, str]] = []
    cb_fallback_scope: dict[str, list[str]] = {}

    for qname, per_name_raw in baseline_costs.items():
        cname = canonical_cost_name(str(qname))
        per_name = dict(per_name_raw)
        synthesized: dict[str, dict] = {}
        per_qname_formats = output_formats
        if planned_scope is not None:
            allowed = planned_scope[cname]
            per_qname_formats = [
                fmt for fmt in output_formats
                if fmt not in planned_universe or fmt in allowed
            ]
        for fmt in per_qname_formats:
            fmt_c = fr.canonical_format_name(fmt)
            if fmt_c == "BF16":
                synthesized[fmt_c] = {
                    "predicted_dloss": 0.0,
                    "weight_mse": 0.0,
                    "output_mse": 0.0,
                    "rel_output_mse": 0.0,
                    "output_mse_measured": False,
                    "cost_source": "bf16_zero",
                }
                continue

            record = _lookup_record(records, qname, fmt_c)
            if (
                fr.get_format(fmt_c).family in {"nvfp4_cb", "fp8_cb"}
                and (cname, fmt_c) not in valid_cb_render_records
            ):
                record = None
            if record is not None:
                metric = str(record.get("metric", ""))
                if require_output_metric and metric not in {
                    "output_mse",
                    "fisher_output_mse",
                }:
                    non_output_metric.append({
                        "qname": str(qname),
                        "format": fmt_c,
                        "metric": metric,
                    })
                else:
                    entry = _production_cost_entry(
                        record,
                        score_field=score_field,
                    )
                    if entry is not None:
                        synthesized[fmt_c] = entry
                        render_entries += 1
                        continue

            missing.append({"qname": str(qname), "format": fmt_c})
            fallback = None
            for alias in fr.aliases_for(fmt_c):
                if alias in per_name:
                    fallback = dict(per_name[alias])
                    break
            if fallback is None and fmt_c in per_name:
                fallback = dict(per_name[fmt_c])
            if fallback is None:
                fallback = {"error": "missing production render score"}
            else:
                if fr.get_format(fmt_c).family in {"nvfp4_cb", "fp8_cb"}:
                    cb_fallback_scope.setdefault(str(qname), []).append(fmt_c)
                fallback["cost_source"] = fallback.get(
                    "cost_source",
                    "fallback_baseline",
                )
            synthesized[fmt_c] = fallback
            fallback_entries += 1
        output_costs[str(qname)] = synthesized

    if cb_fallback_scope:
        # A context-only match is insufficient: a baseline measured with
        # imatrix A cannot be relabeled as cache/imatrix B merely because both
        # used layout v2.  Require value-bearing provenance and compare every
        # CB row actually consumed as a fallback.
        from prismaquant.production_weight_cache import (
            validate_cb_render_provenance,
            validate_matching_cb_render_identities,
        )

        _baseline_context, baseline_identity = validate_cb_render_provenance(
            baseline_cost_payload,
            expected_context=cb_context,
            expected_formats_by_qname=cb_fallback_scope,
            where="production render baseline CB fallback",
        )
        validate_matching_cb_render_identities(
            cb_render_provenance["cb_render_identity"],
            baseline_identity,
            cb_fallback_scope,
            where="production render baseline CB fallback",
        )

    if require_render_scores and missing:
        sample = ", ".join(
            f"{row['qname']}@{row['format']}" for row in missing[:8]
        )
        raise ValueError(
            f"missing {len(missing)} production render scores; sample={sample}"
        )
    if require_output_metric and non_output_metric:
        sample = ", ".join(
            f"{row['qname']}@{row['format']}:{row['metric']}"
            for row in non_output_metric[:8]
        )
        raise ValueError(
            "production render scores fell back to non-output metrics for "
            f"{len(non_output_metric)} entries; sample={sample}"
        )

    calib_hashes = _calibration_hashes(
        getattr(production_cache, "metadata", None),
        baseline_cost_payload.get("meta"),
        baseline_cost_payload.get("provenance"),
    )
    baseline_provenance = baseline_cost_payload.get("provenance")
    inherited_provenance = (
        dict(baseline_provenance)
        if isinstance(baseline_provenance, Mapping)
        else {}
    )
    if cb_context is not None:
        # The rendered rows came from the cache. Carry the complete persisted
        # value-bearing identity; reconstructing a fresh stamp here would lose
        # the exact imatrix qname scope/content binding.
        inherited_provenance.update(cb_render_provenance)
    if format_plan is not None:
        inherited_provenance["source_format_plan_identity_sha256"] = (
            format_plan.identity_sha256
        )
    return {
        "schema": SCHEMA,
        "costs": output_costs,
        "formats": output_formats,
        # The synthesized table consumes the baseline render for every
        # fallback and the production cache was built under the same guarded
        # stage settings. Preserve the CB serialization identity so the
        # allocator can reject an unknown/stale v1-v2 cache.
        "provenance": inherited_provenance,
        "meta": {
            # R14: inherited calibration identity — see _calibration_hashes.
            "calib_hashes": calib_hashes,
            "calib_hash": calib_hashes[0] if len(calib_hashes) == 1 else None,
            "production_cache_source": source_label,
            "baseline_schema": baseline_cost_payload.get("schema"),
            "baseline_meta": baseline_cost_payload.get("meta"),
            "source_format_plan_identity_sha256": (
                format_plan.identity_sha256
                if format_plan is not None else None
            ),
            "score_field": score_field,
            "render_score_entries": int(render_entries),
            "fallback_entries": int(fallback_entries),
            "available_render_scores": int(len(records)),
            "missing_render_score_entries": int(len(missing)),
            "cost_semantics": (
                "predicted_dloss is copied directly from the production "
                "render score field; output_mse_measured is false so the "
                "allocator does not multiply by h_trace"
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize allocator costs from ProductionWeightCache render scores",
    )
    parser.add_argument("--production-cache", required=True)
    parser.add_argument("--baseline-cost", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cost-mode", default="",
                        help="Pipeline COST_MODE stamped into "
                             "provenance['cost_mode'] (re-vet R2).")
    parser.add_argument(
        "--formats",
        default=None,
        help="Comma-separated formats. Defaults to baseline cost formats.",
    )
    parser.add_argument(
        "--format-plan",
        default=None,
        help="Identity-bound source-class format plan used to build the "
        "production cache. Planned-family rows are priced only within each "
        "qname's exact legal menu; identity drift and truncation are fatal.",
    )
    parser.add_argument(
        "--score-field",
        choices=("output_mse", "weight_mse", "weight_mse_sum", "score_sum", "score"),
        default="output_mse",
        help="Cost source to feed the allocator. output_mse (default) hands "
        "the allocator the production-rendered per-element output_mse with "
        "output_mse_measured=True, so it computes h_trace * output_mse "
        "(the original prismaquant cost objective on production-quality "
        "rendered weights). weight_mse / weight_mse_sum emit weight_mse and "
        "let the allocator's h_trace * weight_mse fallback fire. score_sum / "
        "score emit the local render-gate score directly as predicted_dloss "
        "(legacy production-render-score behavior).",
    )
    parser.add_argument(
        "--require-render-scores",
        action="store_true",
        help="Fail instead of falling back to baseline costs when a non-BF16 "
        "format lacks a production render score.",
    )
    parser.add_argument(
        "--require-output-metric",
        action="store_true",
        help="Fail if any consumed render score is a weight_mse fallback "
        "instead of output_mse/fisher_output_mse.",
    )
    args = parser.parse_args(argv)
    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("production_render_cost")

    # The staged (two-pass NVFP4-then-promote) surface — --select-tail-*,
    # --promotion-qnames-file, --bf16-policy, --missing-render-score-policy —
    # was walled 2026-07-30 with COST_MODE=production-render-staged
    # (re-vet R17). See archive/production_render_staged_2026-07-30/.
    cache = _load_pickle(args.production_cache)
    if not args.baseline_cost:
        raise SystemExit("--baseline-cost is required")
    if not args.output:
        raise SystemExit("--output is required")
    baseline = _load_pickle(args.baseline_cost)
    formats = (
        [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
        if args.formats else None
    )
    format_plan = None
    if args.format_plan:
        from prismaquant.source_class_format_plan import load_format_plan

        format_plan = load_format_plan(args.format_plan)
    payload = synthesize_production_render_cost_payload(
        cache,
        baseline,
        formats=formats,
        score_field=args.score_field,
        source_label=str(args.production_cache),
        require_render_scores=bool(args.require_render_scores),
        require_output_metric=bool(args.require_output_metric),
        format_plan=format_plan,
    )
    # Stamp the pipeline COST_MODE (re-vet R2 precondition (i)): cost.pkl is
    # the same path under every mode, so reuse must be conditional on it.
    prov = dict(payload.get("provenance") or {})
    prov["cost_mode"] = str(args.cost_mode or "")
    payload["provenance"] = prov
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    meta = payload["meta"]
    print(
        f"[production-render-cost] wrote {output_path} "
        f"(render_entries={meta['render_score_entries']} "
        f"fallback_entries={meta['fallback_entries']})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
