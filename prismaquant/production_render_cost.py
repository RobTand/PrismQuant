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

from prismaquant import format_registry as fr
from prismaquant.render_score import persisted_cell_score_fields


SCHEMA = "prismaquant.production_render_score_cost.v1"


def canonical_cost_name(qname: str) -> str:
    name = str(qname)
    if name.endswith(".weight"):
        name = name[:-len(".weight")]
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
    persisted_scores = persisted_cell_score_fields(record)
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
            **persisted_scores,
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
            **persisted_scores,
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
        **persisted_scores,
    }


def synthesize_production_render_cost_payload(
    production_cache: object,
    baseline_cost_payload: Mapping,
    *,
    formats: Sequence[str] | None = None,
    score_field: str = "score_sum",
    source_label: str | None = None,
    require_render_scores: bool = False,
    require_output_metric: bool = False,
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
        # A score is usable only when both the value-bearing identity and an
        # actual admitted cache tensor cover the row.  This prevents an old
        # render_scores.json entry (including one left after a failed fresh
        # render) from being relabeled under today's identity.
        valid_cb_render_records = identity_pairs & cache_pairs

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
        for fmt in output_formats:
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
    payload = synthesize_production_render_cost_payload(
        cache,
        baseline,
        formats=formats,
        score_field=args.score_field,
        source_label=str(args.production_cache),
        require_render_scores=bool(args.require_render_scores),
        require_output_metric=bool(args.require_output_metric),
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
