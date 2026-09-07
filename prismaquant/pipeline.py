"""Typed pipeline, artifact, resource, and gate contracts.

This module is descriptive about stage *shape* and authoritative about one
thing: **which settings each build artifact's identity is keyed on** (re-vet
R5).  ``run-pipeline.sh`` executes; this module decides what a reuse of
``cost.pkl`` (or a 90 GB production cache) is allowed to mean.  Everything
else here — the artifact/stage/gate declarations — remains a documented view
of the production flow, not an executor (re-vet R23: no python port).
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


APPROVED_RESOURCE_OWNERS: dict[str, frozenset[str]] = {
    "rendered_weights": frozenset({"ProductionWeightCache"}),
    "perturbed_activations": frozenset({"PerturbedActivationCache"}),
    # layer_streaming.LayerCache is the real streaming-weight owner
    # (`class LayerCache` in layer_streaming.py; the cite carried a line
    # number that had already drifted). The former placeholder names
    # (StreamingActivationCache / StreamingModelPrefetch) were never
    # implemented anywhere in the tree and were deleted with re-vet R5/D10.
    "streaming_model_weights": frozenset({"LayerCache"}),
}


# ``validate_assignments_kl --assignment-materialization=hooks`` keeps the
# source model and every Pareto assignment's rendered weights in one process.
# That is fast on small dense checkpoints, but it is not a safe production
# plan once the checkpoint reaches the 35B class (and routed-MoE models hit the
# same limit earlier because their packed expert state is especially wide).
# Keep the threshold decimal, matching public model-size names.
FRONTIER_HOOKS_MAX_PARAMETERS = 35_000_000_000


def _model_config_for_frontier_policy(model_path: str | Path) -> Mapping[str, Any]:
    path = Path(model_path) / "config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read model config {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"model config {path} is not a JSON object")
    return payload


def _config_declares_moe(value: object) -> bool:
    """Conservatively identify routed-expert model configuration.

    Model families do not share one expert-count spelling.  A positive or
    otherwise non-empty configuration field containing ``expert`` is enough
    to require the memory-fit path; false positives only choose the safer
    materializer, while a false negative can OOM-kill a production host.
    Architecture/model-type names containing ``moe`` are an independent
    signal for configs whose expert details live in a nested text config.
    """

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).lower()
            if key in {"architectures", "model_type"}:
                names = item if isinstance(item, list) else [item]
                if any("moe" in str(name).lower() for name in names):
                    return True
            if "expert" in key:
                if isinstance(item, bool):
                    if item:
                        return True
                elif isinstance(item, (int, float)):
                    if item > 0:
                        return True
                elif isinstance(item, str):
                    if item.strip() and item.strip().lower() not in {
                        "none",
                        "null",
                        "false",
                        "0",
                    }:
                        return True
                elif item:
                    return True
            if _config_declares_moe(item):
                return True
        return False
    if isinstance(value, list):
        return any(_config_declares_moe(item) for item in value)
    return False


def _safe_model_member(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"unsafe safetensors shard path {relative!r}")
    base = root.resolve(strict=True)
    path = (base / relative).resolve(strict=True)
    if path != base and base not in path.parents:
        raise ValueError(f"safetensors shard escapes model root: {relative!r}")
    if not path.is_file():
        raise ValueError(f"safetensors shard is not a file: {path}")
    return path


def _safetensors_shards(model_path: str | Path) -> tuple[Path, ...]:
    root = Path(model_path)
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read safetensors index {index_path}: {exc}"
            ) from exc
        weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError(f"safetensors index has no weight_map: {index_path}")
        names = sorted({str(name) for name in weight_map.values()})
        return tuple(_safe_model_member(root, name) for name in names)

    shards = tuple(sorted(root.glob("*.safetensors")))
    if not shards:
        raise ValueError(f"model has no safetensors shards: {root}")
    return tuple(_safe_model_member(root, shard.name) for shard in shards)


def _safetensors_parameter_count(model_path: str | Path) -> int:
    """Count checkpoint parameters from headers without opening tensor data."""

    total = 0
    for shard in _safetensors_shards(model_path):
        try:
            with shard.open("rb") as handle:
                header_size = int.from_bytes(handle.read(8), "little")
                if not 0 < header_size <= 512 * 1024 * 1024:
                    raise ValueError(
                        f"implausible safetensors header size {header_size}"
                    )
                raw_header = handle.read(header_size)
            if len(raw_header) != header_size:
                raise ValueError("truncated safetensors header")
            header = json.loads(raw_header)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"cannot inspect safetensors shard {shard}: {exc}") from exc
        if not isinstance(header, Mapping):
            raise ValueError(f"safetensors header is not an object: {shard}")
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(metadata, Mapping):
                raise ValueError(f"malformed tensor metadata for {name!r} in {shard}")
            shape = metadata.get("shape")
            if (
                not isinstance(shape, list)
                or any(type(dim) is not int or dim < 0 for dim in shape)
            ):
                raise ValueError(f"malformed tensor shape for {name!r} in {shard}")
            total += math.prod(shape)
    if total <= 0:
        raise ValueError("safetensors headers contain no positive-size parameters")
    return total


def frontier_materialization_policy(model_path: str | Path) -> dict[str, Any]:
    """Return the fail-closed hooks/inplace policy for one source checkpoint."""

    config = _model_config_for_frontier_policy(model_path)
    is_moe = _config_declares_moe(config)
    parameters = _safetensors_parameter_count(model_path)
    reasons: list[str] = []
    if is_moe:
        reasons.append("model config declares routed experts")
    if parameters >= FRONTIER_HOOKS_MAX_PARAMETERS:
        reasons.append(
            f"checkpoint has {parameters:,} parameters "
            f"(threshold {FRONTIER_HOOKS_MAX_PARAMETERS:,})"
        )
    return {
        "model_path": str(Path(model_path).resolve(strict=False)),
        "parameters": parameters,
        "is_moe": is_moe,
        "requires_inplace": bool(reasons),
        "reasons": reasons,
    }


def check_frontier_materialization(model_path: str | Path, mode: str) -> tuple[int, str]:
    """Validate one requested frontier materializer without touching weights."""

    normalized = str(mode).strip().lower()
    if normalized not in {"hooks", "inplace"}:
        return 2, "VALIDATED_FRONTIER_MATERIALIZATION must be hooks or inplace"
    if normalized == "inplace":
        return 0, "validated-frontier materialization=inplace (memory-fit path)"
    try:
        policy = frontier_materialization_policy(model_path)
    except (OSError, ValueError) as exc:
        return 2, (
            "cannot prove that hooks materialization is safe; use "
            f"VALIDATED_FRONTIER_MATERIALIZATION=inplace: {exc}"
        )
    if policy["requires_inplace"]:
        return 2, (
            "hooks materialization is refused for this production model; use "
            "VALIDATED_FRONTIER_MATERIALIZATION=inplace: "
            + "; ".join(str(reason) for reason in policy["reasons"])
        )
    return 0, (
        "validated-frontier materialization=hooks admitted for proven dense "
        f"checkpoint below 35B ({int(policy['parameters']):,} parameters)"
    )


# ---------------------------------------------------------------------------
# Settings-hash authority (re-vet R5 / debt D6).
#
# `run-pipeline.sh`'s `require_stage_settings` used to hand-pass `k=v` pairs at
# each call site, which meant every new stage arrived with its own opinion of
# what keys its artifact depends on — and ten stages arrived with none at all.
# The key SET now lives here, in one reviewable table, and the shell only
# supplies values.
#
# Declaring a key means "a change to this setting makes the stored artifact a
# DIFFERENT artifact". Over-keying is the named risk: hashing a setting an
# artifact does not depend on forces a spurious rebuild, and some of these
# artifacts are 90 GB. The rule used below: key an artifact on the inputs that
# change its BYTES, and key expensive artifacts conservatively.
#
# Each entry is (manifest_key, settings_source). They differ where a stage
# historically recorded a short manifest key (`NS`) for a specific setting
# (`PRODUCTION_RENDER_COST_NSAMPLES`); keeping the historical manifest key
# means artifacts built before R5 stay valid instead of forcing a rebuild.
# ---------------------------------------------------------------------------

STAGE_SETTINGS_SCHEMA = "prismaquant.stage_settings/1"
STAGE_MANIFEST_SCHEMA = "prismaquant.stage_settings_manifest/2"

# Render-affecting environment, shared by every artifact that stores rendered
# weights. Mirrors RENDER_ENV_SETTINGS in run-pipeline.sh.
_RENDER_SETTINGS: tuple[str, ...] = (
    "PRISMAQUANT_NVFP4_SCALE_RULE",
    "PRISMAQUANT_GPTQ_DAMP_SWEEP",
    "PRISMAQUANT_GPTQ_DAMP",
    "PRISMAQUANT_ACT_CLIP_QUANTILE",
    "PRODUCTION_CACHE_LEVERS",
    "PRODUCTION_CACHE_DISABLE_LEVERS",
)

# Every CB producer choice below changes either the fitted/assigned values or
# their byte layout. Persisted cost/render artifacts must invalidate on any
# change, including assignment-only LDLQ even though serving stays unchanged.
_CB_SERIALIZATION_SETTINGS: tuple[str, ...] = (
    "CB_SCALE_CODING",
    "CB_CODEBOOK_SOURCE",
    "CB_CODEBOOK_SOURCE_SCOPE",
    "CB_CODEBOOK_BUNDLE",
    "CB_ROUTED_MOE_BOOK_SELECTION_SHA256",
    "CB_SCALE_SWEEP",
    "CB_SCALE_SWEEP_SCOPE",
    # The strict FP8-only lane selects the existing no-activation payload
    # schema.  Reusing a cache produced under the historical NVFP4 activation
    # contract would make its stamped bytes disagree with export.
    "CB_ACTIVATION_SCOPE",
    # Probe marginals cover the full calibration corpus; activation-cache
    # rows are intentionally capped.  They are different render inputs even
    # when MODEL_PATH/DATASET/NSAMPLES/SEQLEN are identical.
    "CB_IMATRIX_SOURCE",
    "PRISMAQUANT_CB_LDLQ",
    "PRISMAQUANT_CB_MINCHAIN",
    "PRISMAQUANT_CB_MINCHAIN_ANCHORS",
    "PRISMAQUANT_CB_MINCHAIN_HOLDBACKS",
    "PRISMAQUANT_CB_MINCHAIN_AUDIT_SEED",
    "PRISMAQUANT_CB_MINCHAIN_BACKSTOP",
    "PRISMAQUANT_CB_MINCHAIN_AUDIT_MEDIAN",
    "PRISMAQUANT_CB_MINCHAIN_AUDIT_P95",
    "PRISMAQUANT_CB_ENCODE_TIER",
)

# A head policy changes both the qname census rendered into a menu cache and
# whether AURA measures that row or the hybrid cost backfills it. Keep these
# axes on every persisted cost/cache stage that can contain lm_head.
_HEAD_SETTINGS: tuple[str, ...] = (
    "LM_HEAD_FORMAT",
    "LM_HEAD_RENDER_ACTIVE",
    "LM_HEAD_DP_UNPINNED",
)

# Scoped Tessera exports carry explicit runtime input in the plan identity.
# These fields are absent from legacy unscoped manifests, preserving reuse of
# existing plans. The endpoint, not this projection, refuses incomplete input.
_TESSERA_SCOPE_SETTINGS: tuple[str, ...] = (
    "TESSERA_PLATFORM", "TESSERA_RUNTIME_IMAGE", "TESSERA_EXECUTION_MODE",
    "TESSERA_RESIDENCY", "TESSERA_TARGET_PROFILE",
)


def _key_pairs(*specs: str) -> tuple[tuple[str, str], ...]:
    """``"NS<-PRODUCTION_RENDER_COST_NSAMPLES"`` -> ``("NS", "PRODUCTION_…")``."""
    out: list[tuple[str, str]] = []
    for spec in specs:
        if "<-" in spec:
            manifest_key, source = spec.split("<-", 1)
            out.append((manifest_key.strip(), source.strip()))
        else:
            out.append((spec.strip(), spec.strip()))
    return tuple(out)


STAGE_SETTINGS_KEYS: dict[str, tuple[tuple[str, str], ...]] = {
    # --- probe / cost ------------------------------------------------------
    # The Fisher trace is a function of (model, calibration corpus, window
    # count/length, modality). NOT of FORMATS: the probe is format-blind.
    "probe": _key_pairs(
        "MODEL_PATH", "DATASET", "NSAMPLES", "SEQLEN", "CALIBRATION_MODALITY",
    ),
    # Per-(Linear, format) baseline error. Adds FORMATS; drops modality
    # (the cost stage reads the probe's activation cache, whose modality the
    # probe guard already pins).
    "base-cost": _key_pairs(
        "MODEL_PATH", "DATASET", "NSAMPLES", "SEQLEN",
        "FORMATS<-COST_FORMATS", *_HEAD_SETTINGS,
        *_CB_SERIALIZATION_SETTINGS,
    ),
    # production-render-score: the rendered format-menu cache the score reads.
    "render-cost-cache": _key_pairs(
        "MODEL_PATH", "DATASET", "FORMATS<-COST_FORMATS",
        "NS<-PRODUCTION_RENDER_COST_NSAMPLES",
        "SL<-PRODUCTION_RENDER_COST_SEQLEN",
        "SEED<-PRODUCTION_RENDER_COST_SEED",
        *_HEAD_SETTINGS,
        *_RENDER_SETTINGS,
        *_CB_SERIALIZATION_SETTINGS,
    ),
    # …and the allocator cost table synthesized from it. Cheap to rebuild, so
    # it carries the score field and the require-flags too.
    "render-cost": _key_pairs(
        "MODEL_PATH", "FORMATS<-COST_FORMATS", "COST_MODE",
        "SCORE_FIELD<-PRODUCTION_RENDER_COST_SCORE_FIELD",
        "REQUIRE_SCORES<-PRODUCTION_RENDER_COST_REQUIRE_SCORES",
        "REQUIRE_OUTPUT<-PRODUCTION_RENDER_COST_REQUIRE_OUTPUT",
        *_HEAD_SETTINGS,
        *_CB_SERIALIZATION_SETTINGS,
    ),
    # AURA dW cache. FORMATS is the derived non-BF16 menu (AURA_CACHE_FORMATS);
    # SELECTION_MODE is keyed because validated-surrogate redirects this cache
    # to the frontier path and renders packed experts into it.
    "aura-dw-cache": _key_pairs(
        "MODEL_PATH", "DATASET",
        "FORMATS<-AURA_CACHE_FORMATS",
        "NS<-NSAMPLES", "SL<-SEQLEN", "SELECTION_MODE",
        *_HEAD_SETTINGS,
        *_RENDER_SETTINGS,
        *_CB_SERIALIZATION_SETTINGS,
    ),
    "aura-cost": _key_pairs(
        "MODEL_PATH", "DATASET", "FORMATS<-COST_FORMATS", "COST_MODE",
        "NPROBES<-AURA_COST_NPROBES",
        "NS<-AURA_COST_NSAMPLES",
        "SL<-AURA_COST_SEQLEN",
        "SEED<-AURA_COST_CALIB_SEED",
        "DTYPE<-AURA_COST_DTYPE",
        "AURA_COST_STREAMING",
        "AURA_COST_CHECKPOINT_DIR",
        *_HEAD_SETTINGS,
        *_CB_SERIALIZATION_SETTINGS,
    ),
    "aura-hybrid-cost": _key_pairs(
        "MODEL_PATH", "DATASET", "FORMATS<-COST_FORMATS", "COST_MODE",
        "EXPERT_NS<-AURA_EXPERT_NSAMPLES",
        "EXPERT_SL<-AURA_EXPERT_SEQLEN",
        # When streamed AURA is enabled, the empirical routed-expert tail is
        # streamed and resumed too.  Its checkpoint path is the deterministic
        # ``expert-empirical-cost`` child of AURA_COST_CHECKPOINT_DIR, so these
        # two inputs fully bind that resume identity without a second knob.
        "AURA_COST_STREAMING",
        "AURA_COST_CHECKPOINT_DIR",
        *_HEAD_SETTINGS,
        *_CB_SERIALIZATION_SETTINGS,
    ),
    # --- CB lane -----------------------------------------------------------
    # The imatrix harvest reads the probe's activation cache; key it on what
    # produced that cache. Minutes to rebuild, so keying is generous.
    "cb-col-weights": _key_pairs(
        "MODEL_PATH", "DATASET", "NSAMPLES", "SEQLEN", "ACTIVATION_ROWS_LIMIT",
        "CB_IMATRIX_SOURCE",
    ),
    "cb-learned-bundle": _key_pairs(
        "MODEL_PATH", "FORMATS", "CB_CODEBOOK_SOURCE_SCOPE",
        "CB_CODEBOOK_BUNDLE", "CB_COL_WEIGHTS_SHA256",
        "CB_ROUTED_MOE_BOOK_SELECTION_SHA256",
        "CB_ROUTED_BOOK_KEYING",
        "CB_LEARNED_TRAINER_VERSION",
        "CB_LEARNED_PROMOTION_RECEIPT_SHA256",
        "CB_LEARNED_SOURCE_MODEL_IDENTITY_SHA256",
    ),
    "cb-hybrid-cost": _key_pairs(
        "MODEL_PATH", "FORMATS", "COST_MODE",
        "EXPERT_NS<-CB_EXPERT_NSAMPLES",
        "EXPERT_SL<-CB_EXPERT_SEQLEN",
        "EXPERT_SAMPLE<-CB_EXPERT_SAMPLE",
        "LADDER_INTERP<-CB_LADDER_INTERP",
        *_CB_SERIALIZATION_SETTINGS,
    ),
    # --- production caches -------------------------------------------------
    "frontier-cache": _key_pairs(
        "MODEL_PATH", "DATASET", "NSAMPLES", "SEQLEN",
        "FORMATS<-CACHE_FORMATS",
        *_HEAD_SETTINGS,
        *_RENDER_SETTINGS,
        *_CB_SERIALIZATION_SETTINGS,
    ),
    "frontier-recache": _key_pairs(
        "MODEL_PATH", "DATASET", "NSAMPLES", "SEQLEN",
        *_HEAD_SETTINGS,
        *_RENDER_SETTINGS,
        *_CB_SERIALIZATION_SETTINGS,
    ),
    "production-cache-recached": _key_pairs(
        "MODEL_PATH", "DATASET", "NSAMPLES", "SEQLEN", "FORMATS", "TARGET_BITS",
        "ASSIGNMENT_DIGEST",
        *_HEAD_SETTINGS,
        *_RENDER_SETTINGS,
        *_CB_SERIALIZATION_SETTINGS,
    ),
    "production-cache-raw": _key_pairs(
        "MODEL_PATH", "DATASET", "NSAMPLES", "SEQLEN",
        "FORMATS<-CACHE_FORMATS", "ASSIGNMENT_DIGEST",
        "RENDER_SCOPE<-PRODUCTION_CACHE_RENDER_SCOPE",
        *_HEAD_SETTINGS,
        *_RENDER_SETTINGS,
        *_CB_SERIALIZATION_SETTINGS,
    ),
    # --- validated frontier ------------------------------------------------
    # One JSON per Pareto point; the point's identity is in the filename, so
    # the manifest keys the measurement conditions and the render env behind
    # the weights being measured.
    "frontier-kl-point": _key_pairs(
        "MODEL_PATH", "FORMATS",
        "DATASET<-VALIDATED_FRONTIER_DATASET",
        "NS<-VALIDATED_FRONTIER_NSAMPLES",
        "SL<-VALIDATED_FRONTIER_SEQLEN",
        "REPEATS<-VALIDATED_FRONTIER_CALIB_REPEATS",
        "SKIP_CALIB<-VALIDATED_FRONTIER_CALIB_SKIP_FIRST",
        "KL_SCOPE<-VALIDATED_FRONTIER_KL_SCOPE",
        *_RENDER_SETTINGS,
        *_CB_SERIALIZATION_SETTINGS,
    ),
    # --- GGUF lane ---------------------------------------------------------
    # llama.cpp's converter reads only the checkpoint.
    "gguf-skeleton": _key_pairs("MODEL_PATH"),
    # --- Tessera lane ------------------------------------------------------
    # The plan is a projection of the allocation onto the exporter's per-tensor
    # vocabulary, so its identity includes the exact allocation content,
    # checkpoint, coverage decision and explicitly supplied serving scope.
    # The allocator can rewrite layer_config.json on every invocation; its
    # path is not an identity. What is NOT recoverable from that file is the coverage mode, and
    # it is the one setting that changes the artifact without changing the
    # allocation: `broadcast-by-role` extrapolates a single-layer allocation to
    # every depth, `as-allocated` does not. A skip-if-exists plan built under
    # the other mode is a different artifact.
    "tessera-plan": _key_pairs("MODEL_PATH", "COVER<-TESSERA_PLAN_COVER", "ASSIGNMENT_DIGEST",
                               "PLAN_ASSIGNMENT_DIGEST",
                               *_TESSERA_SCOPE_SETTINGS),
}


def parse_settings(pairs: Iterable[str]) -> dict[str, str]:
    """Parse ``K=V`` strings into a settings mapping (later wins)."""
    out: dict[str, str] = {}
    for raw in pairs:
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(f"setting {raw!r} is not K=V")
        key, value = raw.split("=", 1)
        out[key.strip()] = value
    return out


def stage_settings_projection(
    stage: str,
    settings: Mapping[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Project ``settings`` onto ``stage``'s declared key set.

    Returns ``(manifest_key -> value, unresolved_source_names)``.
    """
    try:
        keys = STAGE_SETTINGS_KEYS[stage]
    except KeyError:
        raise KeyError(
            f"unknown settings-hash stage {stage!r}; declare it in "
            "pipeline.STAGE_SETTINGS_KEYS (known: "
            f"{sorted(STAGE_SETTINGS_KEYS)})"
        ) from None
    projection: dict[str, str] = {}
    unresolved: list[str] = []
    for manifest_key, source in keys:
        if (stage == "tessera-plan"
                and (source in _TESSERA_SCOPE_SETTINGS or source == "PLAN_ASSIGNMENT_DIGEST")
                and not settings.get(source)):
            continue
        if source in settings:
            projection[manifest_key] = str(settings[source])
        else:
            unresolved.append(source)
    return projection, unresolved


def stage_settings_document(settings: Mapping[str, str]) -> dict[str, Any]:
    """Emit the per-artifact key sets, already projected onto ``settings``."""
    artifacts: dict[str, dict[str, str]] = {}
    unresolved: dict[str, list[str]] = {}
    for stage in STAGE_SETTINGS_KEYS:
        projection, missing = stage_settings_projection(stage, settings)
        artifacts[stage] = projection
        if missing:
            unresolved[stage] = missing
    return {
        "schema": STAGE_SETTINGS_SCHEMA,
        "artifacts": artifacts,
        "unresolved": unresolved,
    }


def _load_stage_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: settings manifest is not a JSON object")
    if payload.get("schema") == STAGE_MANIFEST_SCHEMA:
        return payload
    # Pre-R5 manifests are a flat {manifest_key: value} dict written by one
    # anonymous stage. Keep them under "legacy" so upgrading the file never
    # drops the guard that was already there.
    return {
        "schema": STAGE_MANIFEST_SCHEMA,
        "stages": {},
        "legacy": {str(k): str(v) for k, v in payload.items()},
    }


def check_stage_settings(
    artifact: str | Path,
    stage: str,
    document: Mapping[str, Any],
    *,
    overrides: Mapping[str, str] | None = None,
) -> tuple[int, list[str]]:
    """Guard one skip-if-exists artifact. Returns ``(exit_code, messages)``.

    * artifact absent -> record this stage's projection, exit 0.
    * artifact present with a matching recorded projection -> exit 0.
    * artifact present, projection differs -> exit 2, naming every diff.
    * artifact present, this stage never recorded -> WARN and record
      (trust-on-first-use: artifacts predating a stage's guard are not
      invalidated, which is the pre-R5 contract for missing manifests).
      Tessera plans are the exception: an old plan cannot be bound to a new
      allocation by recording today's hash after translation already happened.
    """
    declared = dict((document.get("artifacts") or {}).get(stage) or {})
    unresolved = list((document.get("unresolved") or {}).get(stage) or [])
    if overrides:
        extra, still_missing = stage_settings_projection(stage, overrides)
        declared.update(extra)
        unresolved = [name for name in unresolved if name not in overrides]
    if unresolved:
        return 2, [
            f"[pipeline] ERROR: {stage}: settings-hash key(s) "
            f"{sorted(set(unresolved))} were declared in "
            "pipeline.STAGE_SETTINGS_KEYS but no value was supplied by "
            "run-pipeline.sh; the guard refuses to hash a partial key set.",
        ]

    artifact_path = Path(artifact)
    manifest_path = Path(f"{artifact_path}.settings.json")
    messages: list[str] = []

    if artifact_path.exists():
        if not manifest_path.exists():
            if stage == "tessera-plan":
                return 2, [
                    f"[pipeline] ERROR: {stage}: {artifact_path} has no recorded "
                    "allocation content binding; refusing silent reuse. "
                    "Rebuild the plan from the current allocation.",
                ]
            return 0, [
                f"[pipeline] WARNING: {stage}: reusing {artifact_path} which "
                "has no settings manifest (predates the settings-hash guard); "
                "cannot verify it matches the current settings",
            ]
        stored = _load_stage_manifest(manifest_path)
        prev = (stored.get("stages") or {}).get(stage)
        if prev is None:
            legacy = stored.get("legacy")
            if isinstance(legacy, Mapping) and set(legacy) == set(declared):
                prev = dict(legacy)
        if prev is None:
            if stage == "tessera-plan":
                return 2, [
                    f"[pipeline] ERROR: {stage}: {artifact_path} has no recorded "
                    "allocation content binding for this stage; refusing silent reuse. "
                    "Rebuild the plan from the current allocation.",
                ]
            messages.append(
                f"[pipeline] WARNING: {stage}: {artifact_path} predates this "
                "stage's settings guard; recording the current settings "
                "instead of invalidating it"
            )
            _record_stage_settings(manifest_path, stage, declared)
            return 0, messages
        diffs = {
            key: (prev.get(key), declared.get(key))
            for key in sorted(set(prev) | set(declared))
            if prev.get(key) != declared.get(key)
        }
        if diffs:
            messages.append(
                f"[pipeline] ERROR: {stage}: {artifact_path} was built under "
                "DIFFERENT settings; refusing silent reuse:"
            )
            for key, (was, now) in diffs.items():
                messages.append(f"    {key}: artifact={was!r}  current={now!r}")
            messages.append(
                f"    -> delete {artifact_path} (and its .settings.json) to "
                "rebuild, or restore the original settings"
            )
            return 2, messages
        return 0, messages

    _record_stage_settings(manifest_path, stage, declared)
    return 0, messages


def _record_stage_settings(
    manifest_path: Path,
    stage: str,
    projection: Mapping[str, str],
) -> None:
    if manifest_path.exists():
        payload = _load_stage_manifest(manifest_path)
    else:
        payload = {"schema": STAGE_MANIFEST_SCHEMA, "stages": {}}
    stages = dict(payload.get("stages") or {})
    stages[stage] = dict(projection)
    payload["stages"] = stages
    payload["schema"] = STAGE_MANIFEST_SCHEMA
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


@dataclass(frozen=True)
class ArtifactSpec:
    """One typed artifact that can enter or leave a pipeline stage."""

    name: str
    kind: str
    version: str = "v1"
    description: str = ""
    resident: bool = False
    provided: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("artifact name must be non-empty")
        if not self.kind:
            raise ValueError(f"{self.name}: artifact kind must be non-empty")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactSpec":
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            version=str(payload.get("version", "v1")),
            description=str(payload.get("description", "")),
            resident=bool(payload.get("resident", False)),
            provided=bool(payload.get("provided", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "version": self.version,
            "description": self.description,
            "resident": self.resident,
            "provided": self.provided,
        }


@dataclass(frozen=True)
class ResourceContract:
    """Cache/prefetch ownership required by a stage.

    ``resource`` is the data class being managed, for example
    ``rendered_weights``.  ``owner`` is the implementation that owns residency.
    Validation rejects unapproved owners for resources covered by PrismaQuant's
    one-cache rule.
    """

    resource: str
    owner: str
    residency: str = "none"
    required: bool = True
    gpu_bound: bool = True
    fail_fast: bool = True

    def __post_init__(self) -> None:
        if not self.resource:
            raise ValueError("resource contract requires a resource name")
        if not self.owner:
            raise ValueError(f"{self.resource}: resource owner must be non-empty")
        if self.residency not in {"none", "optional", "required"}:
            raise ValueError(
                f"{self.resource}: invalid residency {self.residency!r}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourceContract":
        return cls(
            resource=str(payload["resource"]),
            owner=str(payload["owner"]),
            residency=str(payload.get("residency", "none")),
            required=bool(payload.get("required", True)),
            gpu_bound=bool(payload.get("gpu_bound", True)),
            fail_fast=bool(payload.get("fail_fast", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource": self.resource,
            "owner": self.owner,
            "residency": self.residency,
            "required": self.required,
            "gpu_bound": self.gpu_bound,
            "fail_fast": self.fail_fast,
        }


@dataclass(frozen=True)
class MetricDecision:
    key: str
    accepted: bool
    baseline: float | None
    candidate: float | None
    delta: float | None
    relative_gain: float | None
    reason: str


@dataclass(frozen=True)
class GateEvaluation:
    gate_name: str
    passed: bool
    decisions: tuple[MetricDecision, ...]

    def accepted_keys(self) -> tuple[str, ...]:
        return tuple(decision.key for decision in self.decisions if decision.accepted)

    def rejected_keys(self) -> tuple[str, ...]:
        return tuple(
            decision.key for decision in self.decisions if not decision.accepted
        )


@dataclass(frozen=True)
class MetricGateSpec:
    """A configurable metric gate for global or per-item decisions.

    Examples:
      - global KL gate: candidate ``end_kl`` must be lower than baseline.
      - local render gate: per-Linear ``output_mse`` must improve; accepted
        keys are the Linears that should receive the candidate transform.
    """

    name: str
    metric: str
    direction: str = "lower_is_better"
    mode: str = "all"
    min_absolute_delta: float = 0.0
    min_relative_gain: float = 0.0
    require_improvement: bool = True
    max_absolute_regression: float = 0.0
    max_relative_regression: float = 0.0
    missing: str = "fail"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("gate name must be non-empty")
        if not self.metric:
            raise ValueError(f"{self.name}: metric must be non-empty")
        if self.direction not in {"lower_is_better", "higher_is_better"}:
            raise ValueError(f"{self.name}: invalid direction {self.direction!r}")
        if self.mode not in {"all", "any", "per_item"}:
            raise ValueError(f"{self.name}: invalid mode {self.mode!r}")
        if self.missing not in {"fail", "skip", "pass"}:
            raise ValueError(f"{self.name}: invalid missing policy {self.missing!r}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MetricGateSpec":
        return cls(
            name=str(payload["name"]),
            metric=str(payload["metric"]),
            direction=str(payload.get("direction", "lower_is_better")),
            mode=str(payload.get("mode", "all")),
            min_absolute_delta=float(payload.get("min_absolute_delta", 0.0)),
            min_relative_gain=float(payload.get("min_relative_gain", 0.0)),
            require_improvement=bool(payload.get("require_improvement", True)),
            max_absolute_regression=float(
                payload.get("max_absolute_regression", 0.0)
            ),
            max_relative_regression=float(
                payload.get("max_relative_regression", 0.0)
            ),
            missing=str(payload.get("missing", "fail")),
            description=str(payload.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "direction": self.direction,
            "mode": self.mode,
            "min_absolute_delta": self.min_absolute_delta,
            "min_relative_gain": self.min_relative_gain,
            "require_improvement": self.require_improvement,
            "max_absolute_regression": self.max_absolute_regression,
            "max_relative_regression": self.max_relative_regression,
            "missing": self.missing,
            "description": self.description,
        }

    def evaluate(
        self,
        *,
        baseline: Mapping[str, Any] | float | int,
        candidate: Mapping[str, Any] | float | int,
        keys: Iterable[str] | None = None,
    ) -> GateEvaluation:
        baseline_values = _metric_values(baseline, self.metric)
        candidate_values = _metric_values(candidate, self.metric)
        if keys is None:
            eval_keys = tuple(
                sorted(set(baseline_values) | set(candidate_values))
            )
        else:
            eval_keys = tuple(str(key) for key in keys)
        decisions = tuple(
            self._decision_for(
                key,
                baseline_values.get(key),
                candidate_values.get(key),
            )
            for key in eval_keys
        )
        if self.mode == "all":
            passed = bool(decisions) and all(d.accepted for d in decisions)
        elif self.mode == "any":
            passed = any(d.accepted for d in decisions)
        else:
            passed = True
        return GateEvaluation(
            gate_name=self.name,
            passed=bool(passed),
            decisions=decisions,
        )

    def _decision_for(
        self,
        key: str,
        baseline: float | None,
        candidate: float | None,
    ) -> MetricDecision:
        if baseline is None or candidate is None:
            accepted = self.missing == "pass"
            reason = "missing"
            return MetricDecision(
                key=key,
                accepted=accepted,
                baseline=baseline,
                candidate=candidate,
                delta=None,
                relative_gain=None,
                reason=reason,
            )
        if not math.isfinite(float(baseline)) or not math.isfinite(float(candidate)):
            return MetricDecision(
                key=key,
                accepted=False,
                baseline=float(baseline),
                candidate=float(candidate),
                delta=None,
                relative_gain=None,
                reason="non_finite",
            )
        if self.direction == "lower_is_better":
            delta = float(baseline) - float(candidate)
        else:
            delta = float(candidate) - float(baseline)
        denom = max(abs(float(baseline)), 1e-30)
        relative_gain = delta / denom
        accepted = (
            delta > 0.0
            and delta >= float(self.min_absolute_delta)
            and relative_gain >= float(self.min_relative_gain)
        )
        if not accepted and not self.require_improvement:
            regression = max(-delta, 0.0)
            relative_regression = regression / denom
            abs_budget = (
                float(self.max_absolute_regression)
                if self.max_absolute_regression > 0.0
                else float("inf")
            )
            accepted = (
                regression <= abs_budget
                and relative_regression <= float(self.max_relative_regression)
            )
        if accepted:
            reason = "improved" if delta > 0.0 else "within_regression_budget"
        elif delta <= 0.0:
            reason = "regressed_or_tied"
        elif delta < float(self.min_absolute_delta):
            reason = "below_min_absolute_delta"
        else:
            reason = "below_min_relative_gain"
        return MetricDecision(
            key=key,
            accepted=bool(accepted),
            baseline=float(baseline),
            candidate=float(candidate),
            delta=float(delta),
            relative_gain=float(relative_gain),
            reason=reason,
        )


@dataclass(frozen=True)
class PipelineStageSpec:
    """One pluggable stage in a PrismaQuant pipeline."""

    name: str
    component: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    gates: tuple[str, ...] = ()
    resources: tuple[ResourceContract, ...] = ()
    tags: tuple[str, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must be non-empty")
        if not self.component:
            raise ValueError(f"{self.name}: component must be non-empty")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineStageSpec":
        return cls(
            name=str(payload["name"]),
            component=str(payload["component"]),
            inputs=tuple(str(v) for v in payload.get("inputs", ())),
            outputs=tuple(str(v) for v in payload.get("outputs", ())),
            gates=tuple(str(v) for v in payload.get("gates", ())),
            resources=tuple(
                ResourceContract.from_dict(entry)
                for entry in payload.get("resources", ())
            ),
            tags=tuple(str(v) for v in payload.get("tags", ())),
            description=str(payload.get("description", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "component": self.component,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "gates": list(self.gates),
            "resources": [resource.to_dict() for resource in self.resources],
            "tags": list(self.tags),
            "description": self.description,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PipelineValidation:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class PipelineComponentSpec:
    """A named, opt-in pipeline extension.

    Components are contract fragments: they can declare artifacts, gates, and
    stages without taking over core execution.  This is the integration point
    for archived or experimental methods that need to be wired into the
    pluggable pipeline while remaining explicit and off by default.
    """

    id: str
    stages: tuple[PipelineStageSpec, ...]
    artifacts: tuple[ArtifactSpec, ...] = ()
    gates: tuple[MetricGateSpec, ...] = ()
    insert_after: str | None = None
    status: str = "research"
    default_enabled: bool = False
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("component id must be non-empty")
        if self.status not in {
            "research",
            "candidate",
            "production_recipe",
            "default_on",
        }:
            raise ValueError(f"{self.id}: invalid component status {self.status!r}")
        if self.default_enabled and self.status in {"research", "candidate"}:
            raise ValueError(
                f"{self.id}: {self.status} components must be opt-in"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineComponentSpec":
        return cls(
            id=str(payload["id"]),
            stages=tuple(
                PipelineStageSpec.from_dict(entry)
                for entry in payload.get("stages", ())
            ),
            artifacts=tuple(
                ArtifactSpec.from_dict(entry)
                for entry in payload.get("artifacts", ())
            ),
            gates=tuple(
                MetricGateSpec.from_dict(entry)
                for entry in payload.get("gates", ())
            ),
            insert_after=(
                None
                if payload.get("insert_after") is None
                else str(payload.get("insert_after"))
            ),
            status=str(payload.get("status", "research")),
            default_enabled=bool(payload.get("default_enabled", False)),
            description=str(payload.get("description", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "default_enabled": self.default_enabled,
            "insert_after": self.insert_after,
            "metadata": dict(self.metadata),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "gates": [gate.to_dict() for gate in self.gates],
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass(frozen=True)
class PipelineSpec:
    """A declarative PrismaQuant pipeline plan."""

    id: str
    stages: tuple[PipelineStageSpec, ...]
    artifacts: tuple[ArtifactSpec, ...] = ()
    gates: tuple[MetricGateSpec, ...] = ()
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineSpec":
        return cls(
            id=str(payload["id"]),
            stages=tuple(
                PipelineStageSpec.from_dict(entry)
                for entry in payload.get("stages", ())
            ),
            artifacts=tuple(
                ArtifactSpec.from_dict(entry)
                for entry in payload.get("artifacts", ())
            ),
            gates=tuple(
                MetricGateSpec.from_dict(entry)
                for entry in payload.get("gates", ())
            ),
            description=str(payload.get("description", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "metadata": dict(self.metadata),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "gates": [gate.to_dict() for gate in self.gates],
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def artifact_map(self) -> dict[str, ArtifactSpec]:
        return {artifact.name: artifact for artifact in self.artifacts}

    def gate_map(self) -> dict[str, MetricGateSpec]:
        return {gate.name: gate for gate in self.gates}

    def validate(self) -> PipelineValidation:
        errors: list[str] = []
        warnings: list[str] = []
        artifact_names = [artifact.name for artifact in self.artifacts]
        errors.extend(_duplicates("artifact", artifact_names))
        gate_names = [gate.name for gate in self.gates]
        errors.extend(_duplicates("gate", gate_names))
        stage_names = [stage.name for stage in self.stages]
        errors.extend(_duplicates("stage", stage_names))

        declared = set(artifact_names)
        available = {
            artifact.name
            for artifact in self.artifacts
            if artifact.provided
        }
        produced: set[str] = set()
        known_gates = set(gate_names)
        for stage in self.stages:
            for gate in stage.gates:
                if gate not in known_gates:
                    errors.append(f"{stage.name}: unknown gate {gate!r}")
            for input_name in stage.inputs:
                if input_name not in available:
                    errors.append(
                        f"{stage.name}: input {input_name!r} is not available"
                    )
                if input_name not in declared and input_name not in produced:
                    warnings.append(
                        f"{stage.name}: input {input_name!r} is not declared"
                    )
            for output_name in stage.outputs:
                if output_name in produced:
                    errors.append(
                        f"{stage.name}: output {output_name!r} is produced twice"
                    )
                produced.add(output_name)
                available.add(output_name)
            for resource in stage.resources:
                allowed = APPROVED_RESOURCE_OWNERS.get(resource.resource)
                if allowed is not None and resource.owner not in allowed:
                    errors.append(
                        f"{stage.name}: {resource.resource} must use one of "
                        f"{sorted(allowed)}, got {resource.owner!r}"
                    )
                if (
                    resource.required
                    and resource.residency == "required"
                    and not resource.fail_fast
                ):
                    errors.append(
                        f"{stage.name}: required resident {resource.resource} "
                        "must fail fast on miss"
                    )
                if resource.required and not resource.gpu_bound:
                    warnings.append(
                        f"{stage.name}: required resource {resource.resource} "
                        "is not marked GPU-bound"
                    )
        return PipelineValidation(
            errors=tuple(errors),
            warnings=tuple(warnings),
        )


_STAGES: dict[str, PipelineStageSpec] = {}
_COMPONENTS: dict[str, PipelineComponentSpec] = {}
_BUILTINS_REGISTERED = False
_BUILTIN_COMPONENTS_REGISTERED = False


def register_pipeline_stage(spec: PipelineStageSpec) -> None:
    _STAGES[spec.name] = spec


def register_pipeline_component(spec: PipelineComponentSpec) -> None:
    _COMPONENTS[spec.id] = spec


def pipeline_stage(name: str) -> PipelineStageSpec:
    _ensure_builtins_registered()
    return _STAGES[str(name)]


def pipeline_component(name: str) -> PipelineComponentSpec:
    _ensure_components_registered()
    return _COMPONENTS[str(name)]


def registered_pipeline_stages() -> Mapping[str, PipelineStageSpec]:
    _ensure_builtins_registered()
    return dict(_STAGES)


def registered_pipeline_components() -> Mapping[str, PipelineComponentSpec]:
    _ensure_components_registered()
    return dict(_COMPONENTS)


def load_pipeline_spec(path: str | Path) -> PipelineSpec:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: pipeline spec must be a JSON object")
    return PipelineSpec.from_dict(payload)


def write_pipeline_spec(spec: PipelineSpec, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        json.dump(spec.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")


def parse_render_mechanisms(
    enabled: str | Iterable[str] | None,
    *,
    disabled: str | Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Normalize comma-separated render mechanism config.

    Production paths historically spell render levers as env vars such as
    ``PRODUCTION_CACHE_LEVERS=gptq,static_act_order,joint_scale_opt``.  The
    pipeline contract stores the resolved mechanism list so the run artifact
    records the same plugins the cache fill will execute.
    """

    requested = _split_csv(enabled)
    blocked = set(_split_csv(disabled))
    if "none" in requested:
        requested = ()
    return tuple(name for name in requested if name != "none" and name not in blocked)


def production_pipeline_spec_from_config(
    *,
    render_mechanisms: str | Iterable[str] | None = None,
    disabled_render_mechanisms: str | Iterable[str] | None = None,
    model_path: str | None = None,
    work_dir: str | None = None,
    formats: str | None = None,
    target_bits: float | None = None,
    target_profile: str | None = None,
    calibration_modality: str | None = None,
    selection_mode: str | None = None,
    production_cache: str | bool | None = None,
    production_recache: str | bool | None = None,
    components: Iterable[str | PipelineComponentSpec] | None = None,
) -> PipelineSpec:
    """Build the production contract for one configured run."""

    mechanisms = parse_render_mechanisms(
        render_mechanisms,
        disabled=disabled_render_mechanisms,
    )
    spec = default_production_pipeline_spec(render_mechanisms=mechanisms)
    omitted_stages: list[str] = []
    if str(selection_mode or "").strip().lower() == "surrogate":
        omitted_stages.append("validate.kl")
    # run-pipeline.sh records the vLLM smoke command for manual execution; it
    # does not execute that stage as part of the default production run.
    omitted_stages.append("validate.vllm_smoke")
    if omitted_stages:
        omitted = set(omitted_stages)
        spec = PipelineSpec(
            id=spec.id,
            artifacts=spec.artifacts,
            gates=spec.gates,
            stages=tuple(stage for stage in spec.stages if stage.name not in omitted),
            description=spec.description,
            metadata=dict(spec.metadata),
        )
    component_specs = tuple(_resolve_pipeline_component(c) for c in (components or ()))
    if component_specs:
        spec = compose_pipeline_spec(spec, component_specs)
    ordered_mechanisms = tuple(
        str(stage.metadata["mechanism"])
        for stage in spec.stages
        if stage.name.startswith("render.") and "mechanism" in stage.metadata
    )
    metadata = {
        "render_mechanisms": list(ordered_mechanisms),
        "model_path": model_path,
        "work_dir": work_dir,
        "formats": formats,
        "target_bits": target_bits,
        "target_profile": target_profile,
        "calibration_modality": calibration_modality,
        "selection_mode": selection_mode,
        "production_cache": production_cache,
        "production_recache": production_recache,
    }
    if omitted_stages:
        metadata["omitted_unexecuted_stages"] = list(omitted_stages)
    if component_specs:
        metadata["components"] = list(spec.metadata.get("components", ()))
    return PipelineSpec(
        id=spec.id,
        artifacts=spec.artifacts,
        gates=spec.gates,
        stages=spec.stages,
        description=spec.description,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )


def compose_pipeline_spec(
    base: PipelineSpec,
    components: Iterable[str | PipelineComponentSpec],
) -> PipelineSpec:
    """Return ``base`` plus opt-in component contract fragments."""

    artifacts = list(base.artifacts)
    gates = list(base.gates)
    stages = list(base.stages)
    artifact_by_name = {artifact.name: artifact for artifact in artifacts}
    gate_by_name = {gate.name: gate for gate in gates}
    enabled_components: list[dict[str, Any]] = []

    for raw_component in components:
        component = _resolve_pipeline_component(raw_component)
        for artifact in component.artifacts:
            existing = artifact_by_name.get(artifact.name)
            if existing is not None:
                if existing.to_dict() != artifact.to_dict():
                    raise ValueError(
                        f"{component.id}: artifact {artifact.name!r} conflicts "
                        "with the base pipeline"
                    )
                continue
            artifacts.append(artifact)
            artifact_by_name[artifact.name] = artifact
        for gate in component.gates:
            existing = gate_by_name.get(gate.name)
            if existing is not None:
                if existing.to_dict() != gate.to_dict():
                    raise ValueError(
                        f"{component.id}: gate {gate.name!r} conflicts "
                        "with the base pipeline"
                    )
                continue
            gates.append(gate)
            gate_by_name[gate.name] = gate

        insert_at = len(stages)
        if component.insert_after:
            for idx, stage in enumerate(stages):
                if stage.name == component.insert_after:
                    insert_at = idx + 1
                    break
            else:
                raise ValueError(
                    f"{component.id}: insert_after stage "
                    f"{component.insert_after!r} was not found"
                )
        stages[insert_at:insert_at] = component.stages
        enabled_components.append({
            "id": component.id,
            "status": component.status,
            "default_enabled": component.default_enabled,
        })

    metadata = dict(base.metadata)
    metadata["components"] = list(metadata.get("components", ())) + enabled_components
    return PipelineSpec(
        id=base.id,
        artifacts=tuple(artifacts),
        gates=tuple(gates),
        stages=tuple(stages),
        description=base.description,
        metadata=metadata,
    )


def render_mechanism_stage_specs(enabled: Iterable[str]) -> tuple[PipelineStageSpec, ...]:
    """Expose registered render mechanisms as pipeline stages."""

    requested = tuple(enabled)
    if not requested:
        return ()

    from .render_score import resolve_render_mechanism_order

    plan = resolve_render_mechanism_order(requested)
    if plan.errors:
        raise ValueError("; ".join(plan.errors))
    stages: list[PipelineStageSpec] = []
    current_input = "render.baseline_weight"
    for spec in plan.ordered:
        output = f"render.after.{spec.name}"
        stages.append(PipelineStageSpec(
            name=f"render.{spec.name}",
            component=f"render_score:{spec.name}",
            inputs=(
                current_input,
                "render.reference_weight",
                "render.activation_rows",
            ),
            outputs=(output,),
            gates=(f"gate.render.{spec.gate_metric}",),
            resources=(ResourceContract(
                resource="rendered_weights",
                owner="ProductionWeightCache",
                residency="required",
            ),),
            tags=("render", spec.operation, spec.scope),
            metadata={
                "mechanism": spec.name,
                "operation": spec.operation,
                "scope": spec.scope,
                "phase": spec.phase,
                "gate_metric": spec.gate_metric,
            },
            description=spec.description,
        ))
        current_input = output
    return tuple(stages)


def default_production_pipeline_spec(
    *,
    render_mechanisms: Iterable[str] = (
        "four_over_six",
        "static_act_order",
        "joint_scale_opt",
        "gptq",
    ),
) -> PipelineSpec:
    """Return a declarative view of the current production pipeline."""

    artifacts = (
        ArtifactSpec(
            "source_model",
            "hf_checkpoint",
            description="Source HF checkpoint",
            provided=True,
        ),
        ArtifactSpec("model_graph", "model_structure_graph"),
        ArtifactSpec("calibration_batch", "calibration_rows", provided=True),
        ArtifactSpec("probe_stats", "probe_payload"),
        ArtifactSpec("quant_costs", "cost_payload"),
        ArtifactSpec("layer_assignment", "layer_config"),
        ArtifactSpec("production_weight_cache", "production_weight_cache"),
        ArtifactSpec(
            "resident_production_weight_cache",
            "production_weight_cache",
            resident=True,
        ),
        ArtifactSpec("kl_metrics", "validation_metrics"),
        # R5 discovery-walker export gate: the structured verdict the walk
        # gate refuses on (prismaquant.model_walk.WALK_GATE_SCHEMA).
        ArtifactSpec(
            "walk_coverage_report",
            "walk_coverage_report",
            description=(
                "Discovery-walker coverage ledger + fail-closed export-gate "
                "verdict over the source model"
            ),
        ),
        ArtifactSpec("compressed_artifact", "hf_checkpoint"),
        ArtifactSpec("vllm_smoke", "validation_metrics"),
        ArtifactSpec("render.baseline_weight", "tensor", provided=True),
        ArtifactSpec("render.reference_weight", "tensor", provided=True),
        ArtifactSpec("render.activation_rows", "tensor", provided=True),
    )
    gates = (
        MetricGateSpec(
            name="gate.render.output_mse",
            metric="output_mse",
            mode="per_item",
            direction="lower_is_better",
        ),
        MetricGateSpec(
            name="gate.render.fisher_output_mse",
            metric="fisher_output_mse",
            mode="per_item",
            direction="lower_is_better",
        ),
        MetricGateSpec(
            name="gate.validation.end_kl",
            metric="end_kl",
            mode="all",
            direction="lower_is_better",
        ),
    )
    stages = [
        PipelineStageSpec(
            name="model.structure_graph",
            component="model_profiles.structure:build_model_graph",
            inputs=("source_model",),
            outputs=("model_graph",),
            tags=("model_structure",),
        ),
        PipelineStageSpec(
            name="probe.sensitivity",
            component="incremental_probe",
            inputs=("source_model", "model_graph", "calibration_batch"),
            outputs=("probe_stats",),
            resources=(ResourceContract(
                resource="streaming_model_weights",
                owner="LayerCache",
                residency="required",
            ),),
            tags=("probe", "gpu_bound"),
        ),
        PipelineStageSpec(
            name="measure.quant_cost",
            component="incremental_measure_quant_cost",
            inputs=("source_model", "model_graph", "probe_stats"),
            outputs=("quant_costs",),
            resources=(ResourceContract(
                resource="streaming_model_weights",
                owner="LayerCache",
                residency="required",
            ),),
            tags=("cost", "gpu_bound"),
        ),
        PipelineStageSpec(
            name="allocate.assignment",
            component="allocator",
            inputs=("model_graph", "probe_stats", "quant_costs"),
            outputs=("layer_assignment",),
            tags=("allocator", "cpu_solver"),
        ),
        PipelineStageSpec(
            name="cache.fill_production_weights",
            component="production_weight_cache:fill_production_weight_cache",
            inputs=(
                "source_model",
                "model_graph",
                "calibration_batch",
                "layer_assignment",
            ),
            outputs=("production_weight_cache",),
            resources=(ResourceContract(
                resource="rendered_weights",
                owner="ProductionWeightCache",
                residency="optional",
            ),),
            tags=("cache", "render", "gpu_bound"),
        ),
        *render_mechanism_stage_specs(render_mechanisms),
        PipelineStageSpec(
            name="cache.prefetch_assignment",
            component="ProductionWeightCache.prefetch_assignment",
            inputs=("production_weight_cache", "layer_assignment"),
            outputs=("resident_production_weight_cache",),
            resources=(ResourceContract(
                resource="rendered_weights",
                owner="ProductionWeightCache",
                residency="required",
            ),),
            tags=("cache", "prefetch", "gpu_bound"),
        ),
        PipelineStageSpec(
            name="validate.kl",
            component="validate_assignments_kl",
            inputs=(
                "source_model",
                "calibration_batch",
                "layer_assignment",
                "resident_production_weight_cache",
            ),
            outputs=("kl_metrics",),
            gates=("gate.validation.end_kl",),
            resources=(
                ResourceContract(
                    resource="rendered_weights",
                    owner="ProductionWeightCache",
                    residency="required",
                ),
                ResourceContract(
                    resource="perturbed_activations",
                    owner="PerturbedActivationCache",
                    residency="optional",
                ),
            ),
            tags=("validation", "gpu_bound"),
        ),
        PipelineStageSpec(
            name="export.walk_coverage_gate",
            component="model_walk:walk_export_gate",
            inputs=("source_model",),
            outputs=("walk_coverage_report",),
            tags=("walk", "gate", "fail_closed", "meta_intake_cpu"),
            metadata={
                "schema": "prismaquant.model_walk_gate.v1",
                "policy": (
                    "refuse on an unclaimed matmul-fed node, an unresolved "
                    "floating multiplicand, or any unknown walk failure kind"
                ),
                # TP stance: identity and dispositions live on the whole
                # logical tensor; byte fields are totals with a reserved
                # additive shard_policy annotation.
                "decision_unit": "whole_logical_tensor",
                "byte_accounting": "total_logical_tensor_bytes",
                "override_env": "PRISMAQUANT_WALK_GATE_OVERRIDE",
                "override_scope": "trace_incompleteness_only_never_claims",
                # Deliberately NOT a MetricGateSpec: the refusal is
                # structural (a named node), not a metric comparison, and
                # runtime enforcement lives in the stage code
                # (run-pipeline.sh -> python3 -m prismaquant.model_walk).
                "gate_kind": "structural_refusal",
            },
            description=(
                "R5 discovery-walker export gate (§8.8): walks the source "
                "model — module tree plus one FakeTensorMode forward — "
                "against the profile's claim rules, immediately before every "
                "export lane. An unclaimed matmul-fed parameter refuses the "
                "export with the node named and the op cited. Meta-device "
                "intake: no GPU, no weight I/O, no cache residency."
            ),
        ),
        PipelineStageSpec(
            name="export.native_compressed",
            component="export_native_compressed",
            inputs=(
                "source_model",
                "layer_assignment",
                "resident_production_weight_cache",
            ),
            outputs=("compressed_artifact",),
            resources=(ResourceContract(
                resource="rendered_weights",
                owner="ProductionWeightCache",
                residency="required",
            ),),
            tags=("export", "gpu_bound"),
        ),
        PipelineStageSpec(
            name="validate.vllm_smoke",
            component="validation_harness:vllm_smoke",
            inputs=("compressed_artifact",),
            outputs=("vllm_smoke",),
            tags=("vllm", "validation", "gpu_bound"),
        ),
    ]
    return PipelineSpec(
        id="prismaquant.production.v1",
        artifacts=artifacts,
        gates=gates,
        stages=tuple(stages),
        description="Current production flow expressed as typed contracts.",
    )


def _metric_values(
    payload: Mapping[str, Any] | float | int,
    metric: str,
) -> dict[str, float]:
    if isinstance(payload, (float, int)):
        return {"__global__": float(payload)}
    out: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            if metric not in value:
                continue
            raw = value[metric]
        elif key == metric:
            raw = value
            key = "__global__"
        else:
            raw = value
        try:
            out[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def _duplicates(label: str, values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return [f"duplicate {label}: {value}" for value in sorted(dupes)]


def _split_csv(values: str | Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw_values: Iterable[str] = values.split(",")
    else:
        raw_values = values
    out: list[str] = []
    for raw in raw_values:
        for value in str(raw).split(","):
            name = value.strip()
            if name and name not in out:
                out.append(name)
    return tuple(out)


def _register_builtins() -> None:
    for stage in default_production_pipeline_spec(render_mechanisms=()).stages:
        register_pipeline_stage(stage)


def _register_builtin_components() -> None:
    # Research components live in archive until explicitly revived.  The
    # component registry remains available for programmatic opt-in specs, but
    # production imports do not load shelved cross-layer methods.
    return


def _ensure_builtins_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _register_builtins()
    _BUILTINS_REGISTERED = True


def _ensure_components_registered() -> None:
    global _BUILTIN_COMPONENTS_REGISTERED
    if _BUILTIN_COMPONENTS_REGISTERED:
        return
    _register_builtin_components()
    _BUILTIN_COMPONENTS_REGISTERED = True


def _resolve_pipeline_component(
    component: str | PipelineComponentSpec,
) -> PipelineComponentSpec:
    if not isinstance(component, str) and hasattr(component, "id"):
        return component
    return pipeline_component(str(component))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Write or validate PrismaQuant pipeline contracts."
    )
    ap.add_argument(
        "--write-default-production",
        metavar="PATH",
        help="Write the configured production PipelineSpec JSON to PATH.",
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Validate the generated or loaded PipelineSpec and fail on errors.",
    )
    ap.add_argument(
        "--input",
        metavar="PATH",
        help="Validate an existing PipelineSpec JSON instead of generating one.",
    )
    ap.add_argument("--render-mechanisms", default="")
    ap.add_argument("--disable-render-mechanisms", default="")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--formats", default=None)
    ap.add_argument("--target-bits", type=float, default=None)
    ap.add_argument("--target-profile", default=None)
    ap.add_argument("--calibration-modality", default=None)
    ap.add_argument("--selection-mode", default=None)
    ap.add_argument("--production-cache", default=None)
    ap.add_argument("--production-recache", default=None)
    ap.add_argument(
        "--include-component",
        action="append",
        default=[],
        help="Opt-in pipeline component id to compose into the contract.",
    )
    ap.add_argument(
        "--list-components",
        action="store_true",
        help="List registered opt-in pipeline components and exit.",
    )
    ap.add_argument(
        "--check-frontier-materialization",
        metavar="MODEL_PATH",
        help=(
            "Fail closed when hooks materialization is requested for a MoE, "
            "a checkpoint with at least 35B parameters, or a model whose "
            "header-only classification cannot be proven."
        ),
    )
    ap.add_argument(
        "--frontier-materialization",
        metavar="MODE",
        help="Requested validated-frontier materializer: hooks or inplace.",
    )
    ap.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="K=V",
        help="A pipeline setting value. Consumed by --write-stage-settings "
             "(projected onto each artifact's declared key set) and by "
             "--check-stage-settings (late-computed overrides).",
    )
    ap.add_argument(
        "--write-stage-settings",
        metavar="PATH",
        help="Write the per-artifact settings-hash key sets, already "
             "projected onto --setting values, to PATH (re-vet R5).",
    )
    ap.add_argument(
        "--check-stage-settings",
        action="store_true",
        help="Guard one skip-if-exists artifact against its recorded "
             "settings. Requires --stage-settings/--artifact/--stage.",
    )
    ap.add_argument("--stage-settings", metavar="PATH", default=None,
                    help="Path written by --write-stage-settings.")
    ap.add_argument("--artifact", metavar="PATH", default=None)
    ap.add_argument("--stage", metavar="ID", default=None)
    args = ap.parse_args(argv)

    if args.check_frontier_materialization:
        if args.frontier_materialization is None:
            print(
                "[pipeline] ERROR: --check-frontier-materialization needs "
                "--frontier-materialization"
            )
            return 2
        code, message = check_frontier_materialization(
            args.check_frontier_materialization,
            args.frontier_materialization,
        )
        prefix = "[pipeline]" if code == 0 else "[pipeline] ERROR:"
        print(f"{prefix} {message}")
        return code

    if args.check_stage_settings:
        if not (args.stage_settings and args.artifact and args.stage):
            print("[pipeline] ERROR: --check-stage-settings needs "
                  "--stage-settings, --artifact and --stage")
            return 2
        document = json.loads(Path(args.stage_settings).read_text())
        code, messages = check_stage_settings(
            args.artifact,
            args.stage,
            document,
            overrides=parse_settings(args.setting),
        )
        for message in messages:
            print(message, flush=True)
        return code

    if args.write_stage_settings:
        document = stage_settings_document(parse_settings(args.setting))
        out = Path(args.write_stage_settings)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        print(f"[pipeline-spec] wrote {out}")
        return 0

    if args.list_components:
        for component in registered_pipeline_components().values():
            print(
                f"{component.id}\t{component.status}\t"
                f"default_enabled={int(component.default_enabled)}"
            )
        return 0

    if args.input:
        spec = load_pipeline_spec(args.input)
    else:
        spec = production_pipeline_spec_from_config(
            render_mechanisms=args.render_mechanisms,
            disabled_render_mechanisms=args.disable_render_mechanisms,
            model_path=args.model_path,
            work_dir=args.work_dir,
            formats=args.formats,
            target_bits=args.target_bits,
            target_profile=args.target_profile,
            calibration_modality=args.calibration_modality,
            selection_mode=args.selection_mode,
            production_cache=args.production_cache,
            production_recache=args.production_recache,
            components=args.include_component,
        )

    validation = spec.validate()
    if args.validate and validation.errors:
        for error in validation.errors:
            print(f"[pipeline-spec] ERROR: {error}")
        return 2
    for warning in validation.warnings:
        print(f"[pipeline-spec] WARN: {warning}")

    if args.write_default_production:
        write_pipeline_spec(spec, args.write_default_production)
        print(f"[pipeline-spec] wrote {args.write_default_production}")
    elif not args.input:
        print(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
    elif validation.ok:
        print(f"[pipeline-spec] valid: {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
