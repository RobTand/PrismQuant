"""Exact source and finalized-artifact census for the strict Ada lane.

This module is deliberately architecture-specific.  Rank/dtype heuristics are
not sufficient to prove that every source Linear was assigned or that an
unclaimed verbatim tensor did not carry an FP4 payload into an FP8-only
artifact.  The qualified Qwen3.8-27B layout is closed here, then replayed
against source and finalized safetensors headers without loading tensor data.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from safetensors import safe_open

from .cb_layout import codebook_subtable_shapes
from .cost_streaming import (
    compact_streamed_model_identity,
    validate_cached_streamed_model_identity,
)


RTX4090_SOURCE_CENSUS_SCHEMA = (
    "prismaquant.rtx4090_qwen38_source_census.v1"
)
STREAMED_MODEL_IDENTITY_CACHE_ENV = (
    "PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE"
)


class RTX4090ArtifactCensusError(ValueError):
    """The source or finalized checkpoint differs from the closed layout."""


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _canonical_aura_configs_from_source_config(
    source_config: Mapping[str, Any],
    *,
    where: str,
    normalize_execution: bool = True,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Return AURA's stable staged JSON and normalized Transformers config.

    The streamed identity caches ``runner.model.config.to_dict()``.  Reusing
    its shard fingerprints alone is insufficient when ``config.json`` has
    changed, because that file is not a safetensors shard.  A config-only
    temporary source lets both preflight and final replay re-enter AURA's exact
    canonical staging seam.  The staged JSON digest is Transformers-version
    stable; the normalized config is compared to the cache only at producer
    preflight, under the same known-good environment that generated it.
    """

    staged: str | None = None
    try:
        from prismaquant.aura_cost import _stage_aura_model
        from prismaquant.cost_stage_checkpoint import canonical_json

        with tempfile.TemporaryDirectory(
            prefix="prismaquant_rtx4090_config_"
        ) as temporary:
            temporary_root = Path(temporary)
            (temporary_root / "config.json").write_text(
                json.dumps(
                    dict(source_config),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            staged = _stage_aura_model(str(temporary_root))
            staged_payload = json.loads(
                (Path(staged) / "config.json").read_text(encoding="utf-8")
            )
            staged_canonical = canonical_json(
                staged_payload, where=f"{where} staged JSON"
            )
            normalized_canonical = None
            if normalize_execution:
                from transformers import AutoConfig

                normalized = AutoConfig.from_pretrained(
                    staged,
                    trust_remote_code=True,
                    local_files_only=True,
                ).to_dict()
                normalized_canonical = canonical_json(
                    normalized, where=f"{where} normalized config"
                )
    except Exception as exc:
        raise RTX4090ArtifactCensusError(
            f"{where}: cannot rebuild AURA execution config: {exc}"
        ) from exc
    finally:
        if staged is not None:
            staged_path = Path(staged)
            # The canonical stager owns process-exit cleanup too.  This
            # config-only replay has no model using the tree after return, so
            # remove its tiny temporary projection eagerly; the registered
            # cleanup is deliberately tolerant of an already-absent path.
            if staged_path.name.startswith("prismaquant_stage_"):
                shutil.rmtree(staged_path, ignore_errors=True)
    if normalized_canonical is not None:
        normalized_canonical.pop("_name_or_path", None)
    return staged_canonical, normalized_canonical


def _tensor(dtype: str, *shape: int) -> dict[str, object]:
    return {"dtype": dtype, "shape": [int(dim) for dim in shape]}


def _add_tensor(
    manifest: dict[str, dict[str, object]],
    name: str,
    shape: tuple[int, ...],
) -> None:
    if name in manifest:
        raise AssertionError(f"duplicate qualified tensor name {name!r}")
    manifest[name] = _tensor("BF16", *shape)


def _add_linear(
    manifest: dict[str, dict[str, object]],
    linears: dict[str, str],
    *,
    qname: str,
    source_base: str,
    shape: tuple[int, int],
) -> None:
    if qname in linears:
        raise AssertionError(f"duplicate qualified Linear name {qname!r}")
    linears[qname] = source_base + ".weight"
    _add_tensor(manifest, source_base + ".weight", shape)


def _require_exact_ints(
    payload: Mapping[str, Any],
    expected: Mapping[str, int],
    *,
    where: str,
) -> None:
    for key, value in expected.items():
        if type(payload.get(key)) is not int or payload.get(key) != value:
            raise RTX4090ArtifactCensusError(
                f"{where}.{key} must equal {value}, got {payload.get(key)!r}"
            )


def expected_qwen38_source_layout(
    config: Mapping[str, Any],
    validated_config: Mapping[str, Any],
    *,
    where: str,
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Derive every source tensor and every Linear from the qualified config."""

    if "quantization_config" in config:
        raise RTX4090ArtifactCensusError(
            f"{where}: strict BF16 source must not carry quantization_config"
        )
    source_layout = str(validated_config.get("source_layout"))
    text = config.get("text_config") if source_layout == "official_wrapper" else config
    if not isinstance(text, Mapping):
        raise RTX4090ArtifactCensusError(f"{where}: text config is missing")
    _require_exact_ints(
        text,
        {
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "mtp_num_hidden_layers": 1,
        },
        where=f"{where}.text_config",
    )
    if text.get("attn_output_gate") is not True:
        raise RTX4090ArtifactCensusError(
            f"{where}.text_config.attn_output_gate must be true"
        )
    if text.get("attention_bias") is not False:
        raise RTX4090ArtifactCensusError(
            f"{where}.text_config.attention_bias must be false"
        )

    manifest: dict[str, dict[str, object]] = {}
    linears: dict[str, str] = {}
    hidden = 5120
    intermediate = 17408
    body_source = (
        "model.language_model" if source_layout == "official_wrapper" else "model"
    )
    _add_tensor(manifest, f"{body_source}.embed_tokens.weight", (248320, hidden))
    _add_tensor(manifest, f"{body_source}.norm.weight", (hidden,))

    for layer, layer_type in enumerate(validated_config["layer_types"]):
        recipe = f"model.layers.{layer}"
        source = f"{body_source}.layers.{layer}"
        _add_tensor(manifest, f"{source}.input_layernorm.weight", (hidden,))
        _add_tensor(
            manifest, f"{source}.post_attention_layernorm.weight", (hidden,)
        )
        for projection, shape in (
            ("gate_proj", (intermediate, hidden)),
            ("up_proj", (intermediate, hidden)),
            ("down_proj", (hidden, intermediate)),
        ):
            _add_linear(
                manifest,
                linears,
                qname=f"{recipe}.mlp.{projection}",
                source_base=f"{source}.mlp.{projection}",
                shape=shape,
            )
        if layer_type == "linear_attention":
            linear = f"{source}.linear_attn"
            _add_tensor(manifest, f"{linear}.A_log", (48,))
            _add_tensor(manifest, f"{linear}.dt_bias", (48,))
            _add_tensor(manifest, f"{linear}.conv1d.weight", (10240, 1, 4))
            _add_tensor(manifest, f"{linear}.norm.weight", (128,))
            for projection, shape in (
                ("in_proj_a", (48, hidden)),
                ("in_proj_b", (48, hidden)),
                ("in_proj_qkv", (10240, hidden)),
                ("in_proj_z", (6144, hidden)),
                ("out_proj", (hidden, 6144)),
            ):
                _add_linear(
                    manifest,
                    linears,
                    qname=f"{recipe}.linear_attn.{projection}",
                    source_base=f"{linear}.{projection}",
                    shape=shape,
                )
        elif layer_type == "full_attention":
            attention = f"{source}.self_attn"
            _add_tensor(manifest, f"{attention}.q_norm.weight", (256,))
            _add_tensor(manifest, f"{attention}.k_norm.weight", (256,))
            for projection, shape in (
                ("q_proj", (12288, hidden)),
                ("k_proj", (1024, hidden)),
                ("v_proj", (1024, hidden)),
                ("o_proj", (hidden, 6144)),
            ):
                _add_linear(
                    manifest,
                    linears,
                    qname=f"{recipe}.self_attn.{projection}",
                    source_base=f"{attention}.{projection}",
                    shape=shape,
                )
        else:  # The policy's config validator owns the exact schedule.
            raise RTX4090ArtifactCensusError(
                f"{where}: unsupported layer type {layer_type!r}"
            )

    # The released checkpoint carries one full-attention MTP layer.
    _add_linear(
        manifest,
        linears,
        qname="mtp.fc",
        source_base="mtp.fc",
        shape=(hidden, 2 * hidden),
    )
    for name in (
        "norm",
        "pre_fc_norm_embedding",
        "pre_fc_norm_hidden",
    ):
        _add_tensor(manifest, f"mtp.{name}.weight", (hidden,))
    mtp_source = "mtp.layers.0"
    _add_tensor(manifest, f"{mtp_source}.input_layernorm.weight", (hidden,))
    _add_tensor(
        manifest, f"{mtp_source}.post_attention_layernorm.weight", (hidden,)
    )
    for projection, shape in (
        ("gate_proj", (intermediate, hidden)),
        ("up_proj", (intermediate, hidden)),
        ("down_proj", (hidden, intermediate)),
    ):
        _add_linear(
            manifest,
            linears,
            qname=f"{mtp_source}.mlp.{projection}",
            source_base=f"{mtp_source}.mlp.{projection}",
            shape=shape,
        )
    _add_tensor(manifest, f"{mtp_source}.self_attn.q_norm.weight", (256,))
    _add_tensor(manifest, f"{mtp_source}.self_attn.k_norm.weight", (256,))
    for projection, shape in (
        ("q_proj", (12288, hidden)),
        ("k_proj", (1024, hidden)),
        ("v_proj", (1024, hidden)),
        ("o_proj", (hidden, 6144)),
    ):
        _add_linear(
            manifest,
            linears,
            qname=f"{mtp_source}.self_attn.{projection}",
            source_base=f"{mtp_source}.self_attn.{projection}",
            shape=shape,
        )

    if source_layout == "official_wrapper":
        vision = config.get("vision_config")
        if not isinstance(vision, Mapping):
            raise RTX4090ArtifactCensusError(
                f"{where}: official wrapper requires vision_config"
            )
        _require_exact_ints(
            vision,
            {
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "in_channels": 3,
                "num_position_embeddings": 2304,
                "out_hidden_size": hidden,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
            where=f"{where}.vision_config",
        )
        visual = "model.visual"
        for layer in range(27):
            block = f"{visual}.blocks.{layer}"
            for norm in ("norm1", "norm2"):
                _add_tensor(manifest, f"{block}.{norm}.weight", (1152,))
                _add_tensor(manifest, f"{block}.{norm}.bias", (1152,))
            for name, shape in (
                ("attn.qkv", (3456, 1152)),
                ("attn.proj", (1152, 1152)),
                ("mlp.linear_fc1", (4304, 1152)),
                ("mlp.linear_fc2", (1152, 4304)),
            ):
                _add_linear(
                    manifest,
                    linears,
                    qname=f"{block}.{name}",
                    source_base=f"{block}.{name}",
                    shape=shape,
                )
                _add_tensor(manifest, f"{block}.{name}.bias", (shape[0],))
        _add_tensor(manifest, f"{visual}.merger.norm.weight", (1152,))
        _add_tensor(manifest, f"{visual}.merger.norm.bias", (1152,))
        for name, shape in (
            ("linear_fc1", (4608, 4608)),
            ("linear_fc2", (hidden, 4608)),
        ):
            _add_linear(
                manifest,
                linears,
                qname=f"{visual}.merger.{name}",
                source_base=f"{visual}.merger.{name}",
                shape=shape,
            )
            _add_tensor(manifest, f"{visual}.merger.{name}.bias", (shape[0],))
        _add_tensor(
            manifest,
            f"{visual}.patch_embed.proj.weight",
            (1152, 3, 2, 16, 16),
        )
        _add_tensor(manifest, f"{visual}.patch_embed.proj.bias", (1152,))
        _add_tensor(manifest, f"{visual}.pos_embed.weight", (2304, 1152))
    elif "vision_config" in config:
        raise RTX4090ArtifactCensusError(
            f"{where}: flattened text layout must not carry vision_config"
        )

    _add_linear(
        manifest,
        linears,
        qname="lm_head",
        source_base="lm_head",
        shape=(248320, hidden),
    )
    return dict(sorted(manifest.items())), dict(sorted(linears.items()))


def scan_indexed_safetensors(
    root: str | Path,
    *,
    where: str,
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Read the exact indexed tensor header set and reject hidden shards."""

    directory = Path(root)
    nested_shards = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*.safetensors")
        if path.parent != directory
    )
    if nested_shards:
        raise RTX4090ArtifactCensusError(
            f"{where}: nested safetensors containers are forbidden: "
            f"{nested_shards[:8]}"
        )
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RTX4090ArtifactCensusError(
                f"{where}: cannot read {index_path}: {exc}"
            ) from exc
        weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise RTX4090ArtifactCensusError(
                f"{where}: safetensors index has no nonempty weight_map"
            )
        resolved_map: dict[str, str] = {}
        for name, shard in weight_map.items():
            if not isinstance(name, str) or not name:
                raise RTX4090ArtifactCensusError(
                    f"{where}: safetensors index has an invalid tensor name"
                )
            if (
                not isinstance(shard, str)
                or not shard.endswith(".safetensors")
                or Path(shard).name != shard
            ):
                raise RTX4090ArtifactCensusError(
                    f"{where}: unsafe indexed shard name {shard!r}"
                )
            resolved_map[name] = shard
        shard_names = sorted(set(resolved_map.values()))
    else:
        single = directory / "model.safetensors"
        if not single.is_file():
            raise RTX4090ArtifactCensusError(
                f"{where}: no model.safetensors or safetensors index"
            )
        shard_names = [single.name]
        resolved_map = {}

    actual_shards = sorted(path.name for path in directory.glob("*.safetensors"))
    if actual_shards != shard_names:
        raise RTX4090ArtifactCensusError(
            f"{where}: indexed safetensors set differs from files on disk: "
            f"indexed={shard_names[:8]}, actual={actual_shards[:8]}"
        )

    manifest: dict[str, dict[str, object]] = {}
    observed_map: dict[str, str] = {}
    for shard_name in shard_names:
        shard_path = directory / shard_name
        if not shard_path.is_file():
            raise RTX4090ArtifactCensusError(
                f"{where}: indexed shard is missing: {shard_path}"
            )
        try:
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                if resolved_map:
                    expected_keys = sorted(
                        name for name, shard in resolved_map.items()
                        if shard == shard_name
                    )
                    if sorted(keys) != expected_keys:
                        raise RTX4090ArtifactCensusError(
                            f"{where}: {shard_name} header keys differ from index"
                        )
                for name in keys:
                    if name in manifest:
                        raise RTX4090ArtifactCensusError(
                            f"{where}: tensor {name!r} appears in multiple shards"
                        )
                    tensor_slice = handle.get_slice(name)
                    manifest[name] = {
                        "dtype": str(tensor_slice.get_dtype()).upper(),
                        "shape": [int(dim) for dim in tensor_slice.get_shape()],
                    }
                    observed_map[name] = shard_name
        except RTX4090ArtifactCensusError:
            raise
        except Exception as exc:
            raise RTX4090ArtifactCensusError(
                f"{where}: cannot read safetensors header {shard_path}: {exc}"
            ) from exc
    if not resolved_map:
        resolved_map = observed_map
    if observed_map != resolved_map:
        raise RTX4090ArtifactCensusError(
            f"{where}: safetensors header/index tensor-to-shard maps disagree"
        )
    return dict(sorted(manifest.items())), dict(sorted(observed_map.items()))


def preflight_rtx4090_source_census(
    *,
    model_dir: str | Path,
    config: Mapping[str, Any],
    validated_config: Mapping[str, Any],
    assignment: Mapping[str, str],
    validate_config: Callable[..., Mapping[str, Any]],
    where: str,
) -> dict[str, object]:
    """Prove complete source layout and assignment before GPU encoding."""

    cache_path = os.environ.get(STREAMED_MODEL_IDENTITY_CACHE_ENV)
    if not cache_path:
        raise RTX4090ArtifactCensusError(
            f"{where}: strict producer requires "
            f"{STREAMED_MODEL_IDENTITY_CACHE_ENV} from the completed AURA "
            "streamed_model_identity.json"
        )
    try:
        identity = validate_cached_streamed_model_identity(
            model_dir, cache_path, require_complete_checkpoint=True
        )
        compact_identity = compact_streamed_model_identity(
            identity, where=f"{where} source identity"
        )
    except RuntimeError as exc:
        raise RTX4090ArtifactCensusError(f"{where}: {exc}") from exc
    identity_config = identity.get("config")
    current_aura_staged_config, current_aura_config = (
        _canonical_aura_configs_from_source_config(
            config, where=f"{where} current AURA execution config"
        )
    )
    assert current_aura_config is not None
    if not isinstance(identity_config, Mapping):
        raise RTX4090ArtifactCensusError(
            f"{where}: streamed identity has no execution config"
        )
    cached_aura_config = dict(identity_config)
    cached_aura_config.pop("_name_or_path", None)
    if cached_aura_config != current_aura_config:
        changed = sorted(
            key for key in set(cached_aura_config) & set(current_aura_config)
            if cached_aura_config[key] != current_aura_config[key]
        )
        raise RTX4090ArtifactCensusError(
            f"{where}: streamed identity execution config is stale: "
            f"missing={sorted(set(current_aura_config)-set(cached_aura_config))[:8]}, "
            f"extra={sorted(set(cached_aura_config)-set(current_aura_config))[:8]}, "
            f"changed={changed[:8]}"
        )
    try:
        identity_validated = validate_config(
            identity_config, where=f"{where} identity config"
        )
    except Exception as exc:
        raise RTX4090ArtifactCensusError(str(exc)) from exc
    # A streamed text runner may normalize the official multimodal wrapper to
    # its inner causal-LM config.  That is an execution-layout difference, not
    # a model-content difference: compare the complete qualified text shape
    # while the raw source config + checkpoint map below bind the wrapper,
    # visual tower, MTP, and exact serialized namespace independently.
    text_identity_keys = (
        "model_type",
        "architecture",
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "vocab_size",
        "head_dim",
        "num_key_value_heads",
        "num_attention_heads",
        "max_position_embeddings",
        "layer_types",
        "tie_word_embeddings",
    )
    if any(
        identity_validated.get(key) != validated_config.get(key)
        for key in text_identity_keys
    ):
        raise RTX4090ArtifactCensusError(
            f"{where}: streamed identity text config differs from source config"
        )

    expected, linears = expected_qwen38_source_layout(
        config, validated_config, where=f"{where} qualified source layout"
    )
    observed, observed_weight_map = scan_indexed_safetensors(
        model_dir, where=f"{where} source checkpoint"
    )
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(
            name for name in set(expected) & set(observed)
            if expected[name] != observed[name]
        )
        raise RTX4090ArtifactCensusError(
            f"{where}: source checkpoint differs from the exact qualified "
            f"layout: missing={missing[:8]}, extra={extra[:8]}, "
            f"dtype_or_shape={changed[:8]}"
        )
    if identity.get("checkpoint_weight_map") != observed_weight_map:
        raise RTX4090ArtifactCensusError(
            f"{where}: streamed identity checkpoint map differs from scanned "
            "source safetensors"
        )
    if set(assignment) != set(linears):
        missing = sorted(set(linears) - set(assignment))
        extra = sorted(set(assignment) - set(linears))
        raise RTX4090ArtifactCensusError(
            f"{where}: assignment is not the exact source Linear census: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    if assignment.get("lm_head") != "BF16":
        raise RTX4090ArtifactCensusError(
            f"{where}: lm_head must be explicitly assigned BF16"
        )
    source_config = dict(config)
    return {
        "schema": RTX4090_SOURCE_CENSUS_SCHEMA,
        "source_layout": validated_config["source_layout"],
        "source_config_sha256": _canonical_digest(source_config),
        "aura_staged_config_sha256": _canonical_digest(
            current_aura_staged_config
        ),
        "aura_execution_config_sha256": _canonical_digest(
            current_aura_config
        ),
        "source_tensor_manifest_sha256": _canonical_digest(expected),
        "source_tensor_count": len(expected),
        "source_linear_count": len(linears),
        "assignment_sha256": _canonical_digest(dict(sorted(assignment.items()))),
        "source_model_identity": compact_identity,
    }


def bind_rtx4090_source_provenance(
    quant_config: dict[str, Any],
    producer_policy: Mapping[str, Any],
) -> None:
    """Bind the already-validated compact source identity at the common key."""

    census = producer_policy.get("source_census")
    identity = census.get("source_model_identity") if isinstance(
        census, Mapping
    ) else None
    provenance = quant_config.get("provenance")
    if not isinstance(identity, Mapping) or not isinstance(provenance, dict):
        raise RTX4090ArtifactCensusError(
            "strict producer cannot bind source identity provenance"
        )
    previous = provenance.get("source_model_identity")
    if previous is not None and previous != identity:
        raise RTX4090ArtifactCensusError(
            "strict producer source identity disagrees with existing provenance"
        )
    provenance["source_model_identity"] = dict(identity)


def _vllm_target(qname: str, source_layout: str) -> str:
    if source_layout == "official_wrapper":
        if qname.startswith("model.visual."):
            return qname[len("model."):]
        if qname.startswith("model."):
            return "language_model." + qname
    return qname


def _decode_exact_regex(target: str, *, where: str) -> str:
    if not target.startswith("re:^") or not target.endswith("$"):
        raise RTX4090ArtifactCensusError(
            f"{where}: delegated FP8 target must be an exact anchored regex"
        )
    body = target[len("re:^"):-1]
    decoded = body.replace("[.]", ".")
    if decoded.replace(".", "[.]") != body:
        raise RTX4090ArtifactCensusError(
            f"{where}: delegated target contains regex/wildcard syntax"
        )
    return decoded


def _reconcile_routes_and_ignore(
    quant_config: Mapping[str, Any],
    *,
    assignment: Mapping[str, str],
    linears: Mapping[str, str],
    source_manifest: Mapping[str, Mapping[str, object]],
    source_layout: str,
    where: str,
) -> None:
    expected_claims: dict[str, str] = {}
    for qname, fmt in assignment.items():
        if fmt == "BF16":
            continue
        source_weight = linears[qname]
        source_base = source_weight[:-len(".weight")]
        target = source_base if fmt.startswith("FP8_CB_K") else _vllm_target(
            qname, source_layout
        )
        previous = expected_claims.setdefault(target, fmt)
        if previous != fmt:
            raise RTX4090ArtifactCensusError(
                f"{where}: two assignment entries collapse onto target {target!r}"
            )

    observed_claims: dict[str, str] = {}
    for group_name, group in quant_config["config_groups"].items():
        raw_format = str(group.get("format"))
        fmt = raw_format if raw_format.startswith("FP8_CB_K") else "FP8_E4M3"
        for index, raw_target in enumerate(group["targets"]):
            target = str(raw_target)
            if fmt == "FP8_E4M3":
                target = _decode_exact_regex(
                    target,
                    where=f"{where}.config_groups.{group_name}.targets[{index}]",
                )
            elif target.startswith("re:"):
                raise RTX4090ArtifactCensusError(
                    f"{where}.config_groups.{group_name}: FP8-CB targets must "
                    "be literal physical checkpoint bases"
                )
            if target in observed_claims:
                raise RTX4090ArtifactCensusError(
                    f"{where}: config groups claim target {target!r} more than once"
                )
            observed_claims[target] = fmt
    if observed_claims != expected_claims:
        missing = sorted(set(expected_claims) - set(observed_claims))
        extra = sorted(set(observed_claims) - set(expected_claims))
        changed = sorted(
            target for target in set(expected_claims) & set(observed_claims)
            if expected_claims[target] != observed_claims[target]
        )
        raise RTX4090ArtifactCensusError(
            f"{where}: config groups disagree with tensor_formats: "
            f"missing={missing[:8]}, extra={extra[:8]}, formats={changed[:8]}"
        )

    linear_by_weight = {weight: qname for qname, weight in linears.items()}
    expected_ignore = sorted(
        name[:-len(".weight")]
        for name, descriptor in source_manifest.items()
        if name.endswith(".weight")
        and len(descriptor["shape"]) >= 2
        and (
            name not in linear_by_weight
            or assignment[linear_by_weight[name]] == "BF16"
        )
    )
    ignore = quant_config.get("ignore")
    if (
        not isinstance(ignore, list)
        or any(not isinstance(name, str) or not name for name in ignore)
        or len(set(ignore)) != len(ignore)
        or sorted(ignore) != expected_ignore
    ):
        observed_ignore = sorted(ignore) if isinstance(ignore, list) else []
        raise RTX4090ArtifactCensusError(
            f"{where}: ignore differs from exact unquantized rank>=2 source "
            f"census: missing={sorted(set(expected_ignore)-set(observed_ignore))[:8]}, "
            f"extra={sorted(set(observed_ignore)-set(expected_ignore))[:8]}"
        )


def _expected_artifact_manifest(
    source_manifest: Mapping[str, Mapping[str, object]],
    linears: Mapping[str, str],
    assignment: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    expected = {name: dict(descriptor) for name, descriptor in source_manifest.items()}
    for qname, source_weight in linears.items():
        fmt = assignment[qname]
        if fmt == "BF16":
            continue
        descriptor = expected.pop(source_weight)
        shape = [int(dim) for dim in descriptor["shape"]]
        base = source_weight[:-len(".weight")]
        if fmt == "FP8_E4M3":
            expected[base + ".weight"] = {"dtype": "F8_E4M3", "shape": shape}
            expected[base + ".weight_scale"] = {
                "dtype": "F32",
                "shape": [shape[0], 1],
            }
            continue
        k = int(fmt.rsplit("K", 1)[1])
        if len(shape) != 2 or shape[1] % 256:
            raise RTX4090ArtifactCensusError(
                f"{qname}: FP8-CB requires a 2-D input width divisible by 256"
            )
        expected[base + ".cb_qweight"] = {
            "dtype": "U8",
            "shape": [shape[0], (shape[1] // 256) * 4 * k],
        }
        expected[base + ".weight_scale"] = {
            "dtype": "F32",
            "shape": [shape[0]],
        }
    return dict(sorted(expected.items()))


def validate_rtx4090_serialized_tensor_manifest(
    observed: Mapping[str, Mapping[str, object]],
    expected: Mapping[str, Mapping[str, object]],
    *,
    where: str,
) -> None:
    """Require exact finalized keys/dtypes/shapes, including verbatim tensors."""

    if dict(observed) == dict(expected):
        return
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    changed = sorted(
        name for name in set(expected) & set(observed)
        if expected[name] != observed[name]
    )
    raise RTX4090ArtifactCensusError(
        f"{where}: finalized safetensors differ from the exact assigned "
        f"source census: missing={missing[:8]}, extra={extra[:8]}, "
        f"dtype_or_shape={changed[:8]}"
    )


def _validate_codebook_sidecar(
    artifact_dir: Path,
    quant_config: Mapping[str, Any],
    *,
    where: str,
) -> dict[str, str]:
    expected: dict[str, dict[str, object]] = {}
    for group_name, group in quant_config["config_groups"].items():
        raw_format = str(group.get("format"))
        if not raw_format.startswith("FP8_CB_K"):
            continue
        scheme = group["scheme"]
        refs = scheme["codebook_ref"]
        shapes = codebook_subtable_shapes(
            int(scheme["k"]), str(scheme["mode"]), int(scheme["n_sub"])
        )
        for ref, shape in zip(refs, shapes, strict=True):
            descriptor = {"dtype": "F16", "shape": list(shape)}
            previous = expected.setdefault(str(ref), descriptor)
            if previous != descriptor:
                raise RTX4090ArtifactCensusError(
                    f"{where}.config_groups.{group_name}: codebook ref {ref!r} "
                    "is reused with a different shape"
                )
    codebook_file = quant_config.get("codebook_file")
    if not expected:
        raise RTX4090ArtifactCensusError(
            f"{where}: strict artifact selected no FP8-CB tensors"
        )
    if (
        not isinstance(codebook_file, str)
        or Path(codebook_file).name != codebook_file
        or codebook_file != "cb_codebooks.pqcb"
    ):
        raise RTX4090ArtifactCensusError(
            f"{where}: strict codebook_file must be cb_codebooks.pqcb"
        )
    path = artifact_dir / codebook_file
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            observed = {
                name: {
                    "dtype": str(handle.get_slice(name).get_dtype()).upper(),
                    "shape": [
                        int(dim) for dim in handle.get_slice(name).get_shape()
                    ],
                }
                for name in handle.keys()
            }
    except Exception as exc:
        raise RTX4090ArtifactCensusError(
            f"{where}: cannot read codebook sidecar {path}: {exc}"
        ) from exc
    if dict(sorted(observed.items())) != dict(sorted(expected.items())):
        raise RTX4090ArtifactCensusError(
            f"{where}: codebook sidecar keys/dtypes/shapes differ from exact "
            "config-group references"
        )
    from .nvfp4_cb_footprint import _safetensors_tensor_payload_sha256

    return _safetensors_tensor_payload_sha256(path, sorted(expected))


def _validate_final_value_provenance(
    root: Path,
    provenance: Mapping[str, Any],
    *,
    observed_artifact: Mapping[str, Mapping[str, object]],
    observed_shards: Mapping[str, str],
    codebook_digests: Mapping[str, str],
    artifact_content_receipt: Mapping[str, Any] | None,
    where: str,
) -> None:
    """Bind strict value claims to the finalized files and exact census."""

    declared_codebooks = provenance.get("codebook_sha256")
    if (
        not isinstance(declared_codebooks, Mapping)
        or dict(sorted(declared_codebooks.items()))
        != dict(sorted(codebook_digests.items()))
    ):
        raise RTX4090ArtifactCensusError(
            f"{where}: codebook_sha256 differs from finalized FP16 sidecar bytes"
        )

    from .shard_layout import tensor_payload_identity

    tensor_identity = provenance.get("tensor_payload_identity")
    tensor_ledger = tensor_identity.get("tensor_sha256") if isinstance(
        tensor_identity, Mapping
    ) else None
    if (
        not isinstance(tensor_identity, Mapping)
        or set(tensor_identity) != {
            "schema", "algorithm", "tensors", "payload_sha256", "tensor_sha256"
        }
        or not isinstance(tensor_ledger, Mapping)
        or set(tensor_ledger) != set(observed_artifact)
    ):
        raise RTX4090ArtifactCensusError(
            f"{where}: strict tensor_payload_identity is not the closed, exact "
            "finalized tensor census"
        )
    try:
        expected_tensor_identity = tensor_payload_identity(
            tensor_ledger, include_tensor_sha256=True
        )
    except ValueError as exc:
        raise RTX4090ArtifactCensusError(
            f"{where}: invalid tensor payload digest ledger: {exc}"
        ) from exc
    if dict(tensor_identity) != expected_tensor_identity:
        raise RTX4090ArtifactCensusError(
            f"{where}: tensor payload count/hash differs from its exact digest ledger"
        )
    from .shipcard import (
        SHIPCARD_FILENAME,
        WEIGHT_CONTENT_MANIFEST_SCHEMA,
        _validate_weight_content_manifest,
        validate_safetensors_content_receipt,
        verify_safetensors_content_once,
    )

    weight_manifest = provenance.get("weight_content_manifest")
    weights = {
        path.name: int(path.stat().st_size)
        for path in sorted(root.glob("*.safetensors"))
    }
    files = weight_manifest.get("files") if isinstance(
        weight_manifest, Mapping
    ) else None
    if (
        not isinstance(weight_manifest, Mapping)
        or set(weight_manifest) != {"schema", "algorithm", "files"}
        or weight_manifest.get("schema") != WEIGHT_CONTENT_MANIFEST_SCHEMA
        or weight_manifest.get("algorithm") != "sha256"
        or not isinstance(files, Mapping)
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"bytes", "sha256"}
            for row in files.values()
        )
    ):
        raise RTX4090ArtifactCensusError(
            f"{where}: weight_content_manifest differs from the closed schema"
        )
    try:
        _validate_weight_content_manifest(weight_manifest, weights, where=root)
    except ValueError as exc:
        raise RTX4090ArtifactCensusError(f"{where}: {exc}") from exc
    try:
        if artifact_content_receipt is None:
            verify_safetensors_content_once(
                root,
                expected_weight_manifest=weight_manifest,
                expected_tensor_sha256=tensor_ledger,
                expected_tensor_to_file=observed_shards,
            )
        else:
            validate_safetensors_content_receipt(
                root,
                artifact_content_receipt,
                expected_weight_manifest=weight_manifest,
                expected_tensor_sha256=tensor_ledger,
                expected_tensor_to_file=observed_shards,
            )
    except (OSError, ValueError) as exc:
        raise RTX4090ArtifactCensusError(
            f"{where}: finalized safetensors value verification failed: {exc}"
        ) from exc

    git_commit = provenance.get("git_commit")
    if not isinstance(git_commit, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", git_commit
    ) is None:
        raise RTX4090ArtifactCensusError(
            f"{where}: git_commit is not a full lowercase commit identity"
        )
    shipcard_path = root / SHIPCARD_FILENAME
    try:
        shipcard = json.loads(shipcard_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RTX4090ArtifactCensusError(
            f"{where}: cannot read shipcard for git binding: {exc}"
        ) from exc
    build = shipcard.get("build") if isinstance(shipcard, Mapping) else None
    git = build.get("git") if isinstance(build, Mapping) else None
    if (
        not isinstance(git, Mapping)
        or set(git) != {"commit", "dirty"}
        or git.get("commit") != git_commit
        or git.get("dirty") is not False
    ):
        raise RTX4090ArtifactCensusError(
            f"{where}: shipcard build Git identity must be the exact matching "
            "clean producer commit"
        )

    inventory = provenance.get("artifact_inventory")
    inventory_codebooks = inventory.get(
        "cb_codebook_content_sha256"
    ) if isinstance(inventory, Mapping) else None
    if inventory_codebooks != dict(sorted(codebook_digests.items())):
        raise RTX4090ArtifactCensusError(
            f"{where}: artifact inventory codebook digest ledger is missing or "
            "differs from finalized bytes"
        )


def validate_rtx4090_finalized_artifact_census(
    *,
    artifact_dir: str | Path,
    quant_config: Mapping[str, Any],
    assignment: Mapping[str, str],
    validate_config: Callable[..., Mapping[str, Any]],
    where: str,
    artifact_content_receipt: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Replay source, ownership, and serialized tensor closure from disk."""

    root = Path(artifact_dir)
    try:
        artifact_config = json.loads(
            (root / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RTX4090ArtifactCensusError(
            f"{where}: cannot read finalized config.json: {exc}"
        ) from exc
    if not isinstance(artifact_config, Mapping):
        raise RTX4090ArtifactCensusError(
            f"{where}: finalized config.json must be an object"
        )
    source_config = dict(artifact_config)
    source_config.pop("quantization_config", None)
    try:
        validated = validate_config(
            source_config, where=f"{where} finalized source config"
        )
    except Exception as exc:
        raise RTX4090ArtifactCensusError(str(exc)) from exc
    source_manifest, linears = expected_qwen38_source_layout(
        source_config, validated, where=f"{where} finalized source layout"
    )
    finalized_aura_staged_config, _ = (
        _canonical_aura_configs_from_source_config(
            source_config,
            where=f"{where} finalized AURA config",
            normalize_execution=False,
        )
    )

    provenance = quant_config.get("provenance")
    policy = provenance.get("producer_policy") if isinstance(
        provenance, Mapping
    ) else None
    census = policy.get("source_census") if isinstance(policy, Mapping) else None
    if not isinstance(census, Mapping) or census.get(
        "schema"
    ) != RTX4090_SOURCE_CENSUS_SCHEMA:
        raise RTX4090ArtifactCensusError(
            f"{where}: producer_policy.source_census is required"
        )
    expected_census_fields = {
        "source_layout": validated["source_layout"],
        "source_config_sha256": _canonical_digest(source_config),
        "aura_staged_config_sha256": _canonical_digest(
            finalized_aura_staged_config
        ),
        "source_tensor_manifest_sha256": _canonical_digest(source_manifest),
        "source_tensor_count": len(source_manifest),
        "source_linear_count": len(linears),
        "assignment_sha256": _canonical_digest(dict(sorted(assignment.items()))),
    }
    for key, value in expected_census_fields.items():
        if census.get(key) != value:
            raise RTX4090ArtifactCensusError(
                f"{where}: source_census.{key} must equal {value!r}"
            )
    aura_config_digest = str(
        census.get("aura_execution_config_sha256", "")
    ).lower()
    if (
        len(aura_config_digest) != 64
        or any(char not in "0123456789abcdef" for char in aura_config_digest)
    ):
        raise RTX4090ArtifactCensusError(
            f"{where}: source_census has no exact AURA execution-config digest"
        )
    compact_identity = census.get("source_model_identity")
    if (
        not isinstance(compact_identity, Mapping)
        or provenance.get("source_model_identity") != compact_identity
        or compact_identity.get("checkpoint_tensors") != len(source_manifest)
        or type(compact_identity.get("checkpoint_shards")) is not int
        or compact_identity.get("checkpoint_shards") <= 0
    ):
        raise RTX4090ArtifactCensusError(
            f"{where}: compact source identity is missing, partial, or unbound"
        )
    if set(assignment) != set(linears) or assignment.get("lm_head") != "BF16":
        raise RTX4090ArtifactCensusError(
            f"{where}: tensor_formats must cover every source Linear exactly "
            "and explicitly keep lm_head BF16"
        )

    _reconcile_routes_and_ignore(
        quant_config,
        assignment=assignment,
        linears=linears,
        source_manifest=source_manifest,
        source_layout=str(validated["source_layout"]),
        where=where,
    )
    expected_artifact = _expected_artifact_manifest(
        source_manifest, linears, assignment
    )
    observed_artifact, observed_shards = scan_indexed_safetensors(
        root, where=f"{where} finalized model checkpoint"
    )
    validate_rtx4090_serialized_tensor_manifest(
        observed_artifact, expected_artifact, where=where
    )
    codebook_digests = _validate_codebook_sidecar(
        root, quant_config, where=where
    )
    _validate_final_value_provenance(
        root,
        provenance,
        observed_artifact=observed_artifact,
        observed_shards=observed_shards,
        codebook_digests=codebook_digests,
        artifact_content_receipt=artifact_content_receipt,
        where=where,
    )
    return {
        "source_tensors": len(source_manifest),
        "source_linears": len(linears),
        "artifact_tensors": len(expected_artifact),
        "source_layout": validated["source_layout"],
    }


__all__ = [
    "RTX4090ArtifactCensusError",
    "RTX4090_SOURCE_CENSUS_SCHEMA",
    "STREAMED_MODEL_IDENTITY_CACHE_ENV",
    "bind_rtx4090_source_provenance",
    "expected_qwen38_source_layout",
    "preflight_rtx4090_source_census",
    "scan_indexed_safetensors",
    "validate_rtx4090_finalized_artifact_census",
    "validate_rtx4090_serialized_tensor_manifest",
]
