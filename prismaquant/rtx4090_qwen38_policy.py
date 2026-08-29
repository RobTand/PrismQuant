"""Strict producer policy for a context-first Qwen3.8-27B RTX 4090 build.

The registry is intentionally a compatibility superset.  This module is the
campaign boundary that turns that inventory into one narrow artifact contract:
FP8-CB K40/K44/K48 plus dynamic FP8 and BF16, dense Qwen3.8-27B only, with an
18 GB whole-directory ceiling. Runtime device qualification comes from
Gridbook v11; compile/CUDA-graph evidence is an independent PrismaQuant gate.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .cb_layout import FP8_CB_FORMAT_NAMES, FP8_PRODUCT_RUNGS
from .format_registry import canonical_format_name, format_is_producer_eligible
from .gridbook_execution_contract import (
    GridbookExecutionContractError,
    require_compile_only_gridbook_routes,
    require_device_qualified_gridbook_routes,
)
from .layer_config import canonicalize_assignment, canonicalize_format
from .nvfp4_cb_footprint import PREVIOUS_CB_SERIALIZED_PAYLOAD_SCHEMA
from .rtx4090_graph_contract import (
    RTX4090_COMPILATION_MODE,
    RTX4090_CUDAGRAPH_CAPTURE_SIZES,
    RTX4090_CUDAGRAPH_MODE,
    RTX4090_MAX_NUM_SEQS,
)


RTX4090_QWEN38_POLICY_SCHEMA = "prismaquant.rtx4090_qwen38_fp8_policy.v1"
RTX4090_VALIDATION_ONLY_POLICY_SCHEMA = (
    "prismaquant.rtx4090_qwen38_fp8_validation_only_policy.v1"
)
RTX4090_ROUTE_STATUS_SCHEMA = (
    "prismaquant.rtx4090_fp8_cb_route_status.v1"
)
RTX4090_QWEN38_POLICY_ID = "qwen38_27b_rtx4090_fp8_cb"
RTX4090_QWEN38_SERVING_PROFILE = "qwen38_rtx4090_fp8_cb"
RTX4090_VALIDATION_ONLY_POLICY_ID = (
    "qwen38_27b_rtx4090_fp8_cb_validation_only"
)
RTX4090_VALIDATION_ONLY_SERVING_PROFILE = (
    "qwen38_rtx4090_fp8_cb_validation_only"
)
RTX4090_VALIDATION_ONLY_DISPOSITION = "UNRELEASABLE_VALIDATION_ONLY"
RTX4090_TARGET_PLATFORM = "sm_89"
RTX4090_COMPUTE_CAPABILITY_SM = 89
RTX4090_CONTEXT_FIRST_TARGET_BYTES = 18_000_000_000
RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES = 18_000_000_000

RTX4090_QWEN38_TERMINAL_FORMATS = frozenset({"FP8_E4M3", "BF16"})
RTX4090_QWEN38_ALLOWED_FORMATS = (
    FP8_CB_FORMAT_NAMES | RTX4090_QWEN38_TERMINAL_FORMATS
)
RTX4090_QWEN38_FORMAT_MENU = (
    *(f"FP8_CB_K{k}" for k in FP8_PRODUCT_RUNGS),
    "FP8_E4M3",
    "BF16",
)

_DENSE_IDENTITIES = frozenset({
    ("qwen3_5_text", "Qwen3_5ForCausalLM"),
    ("qwen3_8", "Qwen3_8ForCausalLM"),
})
_DENSE_WRAPPER_IDENTITIES = frozenset({
    ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
})
_DENSE_27B_SHAPE = {
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "head_dim": 256,
    "num_key_value_heads": 4,
    "num_attention_heads": 24,
}

# Closed producer wire shapes for the two non-BF16 group kinds admitted by
# this campaign.  The generic Gridbook container intentionally accepts a much
# wider compatibility surface; the Ada artifact does not.  In particular,
# ``num_bits == 8`` alone is not an FP8 identity (an integer W8A8 scheme can
# carry the same widths), and an FP8-CB group must not smuggle an FP4
# activation/scale contract through otherwise plausible CB fields.
_FP8_CB_SCHEME_KEYS = frozenset({
    "grid",
    "mode",
    "k",
    "superblock",
    "group_size",
    "vec_dim",
    "n_sub",
    "type_size",
    "act_bits",
    "codebook_source",
    "codebook_ref",
    "codebook_group",
})
_FP8_E4M3_WEIGHTS = {
    "num_bits": 8,
    "type": "float",
    "strategy": "channel",
    "symmetric": True,
    "dynamic": False,
    "observer": "memoryless_minmax",
}
_FP8_E4M3_ACTIVATIONS = {
    "num_bits": 8,
    "type": "float",
    "strategy": "token",
    "symmetric": True,
    "dynamic": True,
}

# Closed strict artifact surface.  Gridbook reads top-level declarations to
# decide which loader/method owns a tensor, so an unknown key here is not
# harmless forward-compatible metadata.  Provenance is non-dispatching, but
# closing its producer-known field names prevents a second, unvalidated policy
# or source identity from being smuggled alongside the certified records.
_RTX4090_QUANT_CONFIG_KEYS = frozenset({
    "quant_method",
    "format",
    "config_groups",
    "ignore",
    "codebook_file",
    "provenance",
})
_RTX4090_PROVENANCE_KEYS = frozenset({
    "git_commit",
    "assignment_sha256",
    "imatrix_sha256",
    "codebook_sha256",
    "codebook_source",
    "scale_sweep",
    "scale_sweep_scope",
    "ldlq",
    "encode_tier",
    "renderer_abi",
    "scale_coding",
    "cb_targets",
    "stock_ct_targets",
    "fp8_source_targets",
    "source_passthrough_targets",
    "requant_native_targets",
    "serialized_payload",
    "render_identity_verified",
    "streaming",
    "cb_render_identity",
    "tensor_formats",
    "producer_policy",
    "source_model_identity",
    "cb_route_status",
    "tensor_payload_identity",
    "encoder_warm_start",
    "artifact_inventory",
    "weight_content_manifest",
})
_RTX4090_SOURCE_CENSUS_KEYS = frozenset({
    "schema",
    "source_layout",
    "source_config_sha256",
    "aura_staged_config_sha256",
    "aura_execution_config_sha256",
    "source_tensor_manifest_sha256",
    "source_tensor_count",
    "source_linear_count",
    "assignment_sha256",
    "source_model_identity",
})
_RTX4090_COMPACT_SOURCE_IDENTITY_KEYS = frozenset({
    "schema",
    "content_sha256",
    "resolved_commit",
    "checkpoint_shards",
    "checkpoint_tensors",
})
_RTX4090_RENDER_IDENTITY_KEYS = frozenset({
    "schema",
    "cb_serialized_payload",
    "render_contract",
    "cb_formats_by_qname",
    "col_weights_schema",
    "col_weights_sha256",
    "col_weights_entries",
    "col_weights_qnames",
    "col_weights_shapes",
    "col_weights_content_sha256",
    "source_weights_schema",
    "source_weights_complete",
    "source_weights_sha256",
    "source_weights_shapes",
    "source_weights_content_sha256",
})


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(
        r"[0-9a-f]{64}", value
    ) is not None


def _is_full_git_commit(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value
    ) is not None


def _expected_layer_types() -> tuple[str, ...]:
    return tuple(
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(64)
    )


RTX4090_QWEN38_LAYER_TYPES = _expected_layer_types()


def rtx4090_graph_requirement() -> dict[str, Any]:
    """The local graph contract; Gridbook's device table never claims this."""

    return {
        "torch_compile_backend": "inductor",
        "torch_compile_fullgraph": True,
        "vllm_compilation_mode": RTX4090_COMPILATION_MODE,
        "vllm_cudagraph_mode": RTX4090_CUDAGRAPH_MODE,
        "cudagraph_capture_sizes": list(RTX4090_CUDAGRAPH_CAPTURE_SIZES),
        # FULL decode capture eligibility is capped by this scheduler field;
        # one validation request with n=1 remains a separate workload fact.
        "scheduler_max_num_seqs": RTX4090_MAX_NUM_SEQS,
        "receipt_schema": "prismaquant.rtx4090_graph_contract.v1",
    }


class RTX4090Qwen38PolicyError(ValueError):
    """A menu, assignment, artifact, or manifest violates the Ada policy."""


def load_rtx4090_runtime_contract(
    source: Mapping[str, Any] | str | Path,
    *,
    where: str = "RTX 4090 Gridbook runtime contract",
) -> dict[str, Any]:
    """Load an explicitly supplied contract; never fall back to the old pin."""

    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RTX4090Qwen38PolicyError(
            f"{where}: cannot read {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RTX4090Qwen38PolicyError(f"{where}: contract must be an object")
    return dict(payload)


def prepare_rtx4090_export_policy(
    *,
    model_dir: str | Path,
    assignment: Mapping[str, Any],
    producer_policy: str | None,
    runtime_contract: Mapping[str, Any] | str | Path | None,
    where: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Preflight the opt-in strict producer lane before encoding any weights."""

    if producer_policy is None:
        if runtime_contract is not None:
            raise RTX4090Qwen38PolicyError(
                f"{where}: a runtime contract was supplied without an explicit "
                "producer policy"
            )
        return None
    if producer_policy not in {
        RTX4090_QWEN38_POLICY_ID,
        RTX4090_VALIDATION_ONLY_POLICY_ID,
    }:
        raise RTX4090Qwen38PolicyError(
            f"{where}: unsupported producer policy {producer_policy!r}"
        )
    if runtime_contract is None:
        raise RTX4090Qwen38PolicyError(
            f"{where}: {producer_policy} requires an explicit Gridbook v11 "
            "runtime contract; the materialized v4 pin cannot qualify SM89"
        )
    contract = load_rtx4090_runtime_contract(
        runtime_contract, where=f"{where} Gridbook contract"
    )
    config_path = Path(model_dir) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RTX4090Qwen38PolicyError(
            f"{where}: cannot read source config {config_path}: {exc}"
        ) from exc
    validated_config = validate_qwen38_dense_config(
        config, where=f"{where} source model"
    )
    canonical = validate_rtx4090_assignment(
        assignment, where=f"{where} assignment"
    )
    from .rtx4090_artifact_census import (
        RTX4090ArtifactCensusError,
        preflight_rtx4090_source_census,
    )

    try:
        source_census = preflight_rtx4090_source_census(
            model_dir=model_dir,
            config=config,
            validated_config=validated_config,
            assignment=canonical,
            validate_config=validate_qwen38_dense_config,
            where=where,
        )
    except RTX4090ArtifactCensusError as exc:
        raise RTX4090Qwen38PolicyError(str(exc)) from exc
    if producer_policy == RTX4090_QWEN38_POLICY_ID:
        stamp = producer_policy_stamp(contract, tuple(canonical.values()))
    else:
        stamp = validation_only_producer_policy_stamp(
            contract, tuple(canonical.values())
        )
    stamp["source_census"] = source_census
    return contract, stamp


def _canonical_policy_format(entry: Any, *, where: str) -> str:
    try:
        canonical = canonical_format_name(canonicalize_format(entry))
    except (KeyError, TypeError, ValueError) as exc:
        raise RTX4090Qwen38PolicyError(
            f"{where}: unsupported format {entry!r}"
        ) from exc
    if canonical not in RTX4090_QWEN38_ALLOWED_FORMATS:
        raise RTX4090Qwen38PolicyError(
            f"{where}: {canonical} is forbidden by {RTX4090_QWEN38_POLICY_ID}; "
            "the only legal formats are FP8_CB_K40,K44,K48, FP8_E4M3, "
            "and BF16 (NVFP4/NVFP4-CB and reader-only FP8-CB are refused)"
        )
    if canonical.startswith("FP8_CB_K") and not format_is_producer_eligible(
        canonical
    ):
        raise RTX4090Qwen38PolicyError(
            f"{where}: {canonical} is reader-compatible but not producer-eligible"
        )
    return canonical


def validate_rtx4090_format_menu(
    formats: Sequence[Any],
    *,
    where: str = "RTX 4090 format menu",
) -> tuple[str, ...]:
    """Validate a nonempty, duplicate-free subset of the strict menu."""

    if isinstance(formats, (str, bytes, bytearray)):
        formats = tuple(part.strip() for part in str(formats).split(","))
    if not isinstance(formats, Sequence) or not formats:
        raise RTX4090Qwen38PolicyError(f"{where}: menu must be nonempty")
    canonical = tuple(
        _canonical_policy_format(entry, where=f"{where}[{index}]")
        for index, entry in enumerate(formats)
    )
    if len(set(canonical)) != len(canonical):
        raise RTX4090Qwen38PolicyError(
            f"{where}: aliases/entries collapse to duplicate formats"
        )
    return canonical


def validate_rtx4090_assignment(
    assignment: Mapping[str, Any],
    *,
    where: str = "RTX 4090 assignment",
) -> dict[str, str]:
    """Canonicalize and hard-refuse any format outside the Ada FP8 policy."""

    if not isinstance(assignment, Mapping) or not assignment:
        raise RTX4090Qwen38PolicyError(
            f"{where}: assignment must be a nonempty object"
        )
    try:
        canonical = canonicalize_assignment(assignment)
    except (KeyError, TypeError, ValueError) as exc:
        raise RTX4090Qwen38PolicyError(
            f"{where}: assignment cannot be canonicalized: {exc}"
        ) from exc
    if not canonical:
        raise RTX4090Qwen38PolicyError(
            f"{where}: assignment contains no tensor entries"
        )
    resolved = {
        qname: _canonical_policy_format(fmt, where=f"{where}.{qname}")
        for qname, fmt in canonical.items()
    }
    for qname, fmt in resolved.items():
        if qname == "lm_head" or qname.endswith(".lm_head"):
            if fmt != "BF16":
                raise RTX4090Qwen38PolicyError(
                    f"{where}.{qname}: lm_head is an immutable BF16 auxiliary "
                    "region and is excluded from quantizable-parameter bpp"
                )
    return resolved


def validate_qwen38_dense_config(
    config: Mapping[str, Any],
    *,
    where: str = "Qwen3.8 model config",
) -> dict[str, Any]:
    """Require the released dense 27B structure, never its MoE sibling."""

    if not isinstance(config, Mapping):
        raise RTX4090Qwen38PolicyError(f"{where}: config must be an object")
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, Sequence)
        or isinstance(architectures, (str, bytes, bytearray))
        or len(architectures) != 1
    ):
        raise RTX4090Qwen38PolicyError(
            f"{where}: architectures must identify exactly one dense class"
        )
    outer_identity = (
        str(config.get("model_type", "")), str(architectures[0])
    )
    if outer_identity in _DENSE_IDENTITIES:
        text_config = config
        text_identity = outer_identity
        source_layout = "flattened_text"
    elif outer_identity in _DENSE_WRAPPER_IDENTITIES:
        nested = config.get("text_config")
        if not isinstance(nested, Mapping):
            raise RTX4090Qwen38PolicyError(
                f"{where}: official conditional-generation wrapper has no "
                "text_config object"
            )
        text_config = nested
        nested_architectures = text_config.get("architectures")
        if nested_architectures not in (None, [], ["Qwen3_5ForCausalLM"]):
            raise RTX4090Qwen38PolicyError(
                f"{where}: wrapper text_config carries an unexpected "
                f"architecture declaration {nested_architectures!r}"
            )
        text_identity = (
            str(text_config.get("model_type", "")),
            "Qwen3_5ForCausalLM",
        )
        if text_identity not in _DENSE_IDENTITIES:
            raise RTX4090Qwen38PolicyError(
                f"{where}: wrapper text_config is not the dense Qwen3.8 text "
                f"model, got {text_identity!r}"
            )
        source_layout = "official_wrapper"
    else:
        raise RTX4090Qwen38PolicyError(
            f"{where}: expected dense Qwen3.8-27B identity, got "
            f"{outer_identity!r}"
        )
    routed_markers = {
        key: holder[key]
        for holder in (config, text_config)
        for key in ("num_experts", "num_local_experts", "moe_intermediate_size")
        if key in holder and holder[key] not in (None, 0)
    }
    if routed_markers:
        raise RTX4090Qwen38PolicyError(
            f"{where}: routed-expert fields are forbidden: {routed_markers}"
        )
    for key, expected in _DENSE_27B_SHAPE.items():
        if int(text_config.get(key, -1)) != expected:
            raise RTX4090Qwen38PolicyError(
                f"{where}: {key} must equal {expected}, got "
                f"{text_config.get(key)!r}"
            )
    if int(text_config.get("max_position_embeddings", -1)) < 32768:
        raise RTX4090Qwen38PolicyError(
            f"{where}: max_position_embeddings must be at least 32768"
        )
    layer_types = text_config.get("layer_types")
    if (
        not isinstance(layer_types, Sequence)
        or isinstance(layer_types, (str, bytes, bytearray))
        or tuple(layer_types) != RTX4090_QWEN38_LAYER_TYPES
    ):
        raise RTX4090Qwen38PolicyError(
            f"{where}: layer_types must be the exact 64-layer hybrid schedule "
            "(three linear_attention layers then one full_attention, repeated)"
        )
    if text_config.get("tie_word_embeddings") is not False:
        raise RTX4090Qwen38PolicyError(
            f"{where}: the qualified 27B checkpoint has untied embeddings/head"
        )
    return {
        "source_layout": source_layout,
        "outer_model_type": outer_identity[0],
        "outer_architecture": outer_identity[1],
        "model_type": text_identity[0],
        "architecture": text_identity[1],
        **_DENSE_27B_SHAPE,
        "max_position_embeddings": int(text_config["max_position_embeddings"]),
        "layer_types": list(RTX4090_QWEN38_LAYER_TYPES),
        "tie_word_embeddings": False,
    }


def _validate_group_targets(group: Mapping[str, Any], *, where: str) -> None:
    targets = group.get("targets")
    if (
        not isinstance(targets, Sequence)
        or isinstance(targets, (str, bytes, bytearray))
        or not targets
        or any(not isinstance(target, str) or not target for target in targets)
        or len(set(targets)) != len(targets)
    ):
        raise RTX4090Qwen38PolicyError(
            f"{where}: targets must be a nonempty list of unique strings"
        )


def _validate_config_group(group: Mapping[str, Any], *, where: str) -> None:
    raw_format = str(group.get("format", ""))
    if raw_format.startswith(("FP8_CB_K", "NVFP4_CB_K")):
        if set(group) != {"format", "scheme", "targets"}:
            raise RTX4090Qwen38PolicyError(
                f"{where}: FP8-CB group has fields outside the closed wire schema"
            )
        _validate_group_targets(group, where=where)
        _canonical_policy_format(raw_format, where=where)
        scheme = group.get("scheme")
        if not isinstance(scheme, Mapping) or set(scheme) != _FP8_CB_SCHEME_KEYS:
            raise RTX4090Qwen38PolicyError(
                f"{where}: FP8-CB scheme fields differ from the closed FP8 wire schema"
            )
        if scheme.get("grid") != "fp8" or scheme.get("mode") != "product":
            raise RTX4090Qwen38PolicyError(
                f"{where}: FP8-CB group must carry the fp8/product scheme"
            )
        expected_k = int(raw_format.rsplit("K", 1)[1])
        exact_fields = {
            "k": expected_k,
            "superblock": 256,
            "group_size": 0,
            "vec_dim": 8,
            "n_sub": 4,
            "type_size": 4 * expected_k,
            "act_bits": 8,
        }
        if any(
            type(scheme.get(key)) is not int or scheme.get(key) != value
            for key, value in exact_fields.items()
        ):
            raise RTX4090Qwen38PolicyError(
                f"{where}: FP8-CB numeric layout differs from {exact_fields!r}"
            )
        source = scheme.get("codebook_source")
        group_name = scheme.get("codebook_group")
        # Learned-v2 remains independently result-gated.  Until its raw
        # promotion ledger, imatrix identity, and source closure are carried
        # into this artifact contract, the shipping profile is lattice-only.
        if source != "lattice" or group_name is not None:
            raise RTX4090Qwen38PolicyError(
                f"{where}: strict RTX4090 release groups must use canonical "
                "lattice codebooks; learned-v2 has not cleared its artifact "
                "attestation gate"
            )
        refs = scheme.get("codebook_ref")
        if (
            not isinstance(refs, Sequence)
            or isinstance(refs, (str, bytes, bytearray))
            or len(refs) != 4
            or any(not isinstance(ref, str) or not ref for ref in refs)
            or len(set(refs)) != len(refs)
        ):
            raise RTX4090Qwen38PolicyError(
                f"{where}: FP8-CB codebook_ref must name four unique subtables"
            )
        return
    if raw_format == "float-quantized":
        if set(group) != {
            "format", "weights", "input_activations", "targets"
        }:
            raise RTX4090Qwen38PolicyError(
                f"{where}: delegated FP8 group has fields outside the closed "
                "wire schema"
            )
        _validate_group_targets(group, where=where)
        if (
            group.get("weights") != _FP8_E4M3_WEIGHTS
            or group.get("input_activations") != _FP8_E4M3_ACTIVATIONS
        ):
            raise RTX4090Qwen38PolicyError(
                f"{where}: delegated float group must be the exact dynamic "
                "E4M3 W8A8 scheme, never INT8 or NVFP4"
            )
        return
    raise RTX4090Qwen38PolicyError(
        f"{where}: unsupported/undeclared config-group format {raw_format!r}"
    )


def rtx4090_route_status_stamp(
    producer_policy: Mapping[str, Any],
    assignment: Mapping[str, str],
) -> dict[str, Any]:
    """Bind strict route status to the supplied v11 device attestation.

    The generic CB route gate intentionally resolves the repository's current
    historical serving pin.  That is the wrong authority for this unreleased
    candidate: strict export is admitted by the external Gridbook v11 contract
    supplied to :func:`prepare_rtx4090_export_policy`.  Preserve the complete
    full-ladder attestation, then bind the selected artifact population beside
    it.  No generic override/non-native/fallback disposition is representable
    in this schema.
    """

    runtime = producer_policy.get("runtime_attestation")
    if not isinstance(runtime, Mapping):
        raise RTX4090Qwen38PolicyError(
            "strict route status requires producer_policy.runtime_attestation"
        )
    canonical = validate_rtx4090_assignment(
        assignment, where="RTX4090 strict route assignment"
    )
    return {
        "schema": RTX4090_ROUTE_STATUS_SCHEMA,
        "authority": "producer_policy.runtime_attestation",
        "selected_fp8_cb_units": sum(
            fmt.startswith("FP8_CB_K") for fmt in canonical.values()
        ),
        "selected_fp8_cb_rungs": sorted({
            int(fmt.rsplit("K", 1)[1])
            for fmt in canonical.values()
            if fmt.startswith("FP8_CB_K")
        }),
        "delegated_fp8_e4m3_units": sum(
            fmt == "FP8_E4M3" for fmt in canonical.values()
        ),
        "runtime_attestation": json.loads(json.dumps(runtime)),
    }


def is_rtx4090_validation_only_policy(
    payload: Mapping[str, Any] | None,
) -> bool:
    """Return true only for the immutable unreleasable policy identity."""

    return bool(
        isinstance(payload, Mapping)
        and payload.get("schema") == RTX4090_VALIDATION_ONLY_POLICY_SCHEMA
        and payload.get("id") == RTX4090_VALIDATION_ONLY_POLICY_ID
        and payload.get("artifact_disposition")
        == RTX4090_VALIDATION_ONLY_DISPOSITION
    )


def validate_rtx4090_route_status(
    payload: Mapping[str, Any],
    *,
    producer_policy: Mapping[str, Any],
    assignment: Mapping[str, str],
    where: str,
) -> dict[str, Any]:
    """Replay the closed strict route record and require clean native cells."""

    if not isinstance(payload, Mapping):
        raise RTX4090Qwen38PolicyError(
            f"{where}: strict cb_route_status must be an object"
        )
    expected = rtx4090_route_status_stamp(producer_policy, assignment)
    if dict(payload) != expected:
        raise RTX4090Qwen38PolicyError(
            f"{where}: cb_route_status differs from the exact supplied Gridbook "
            "v11 runtime attestation and selected assignment"
        )
    runtime = payload["runtime_attestation"]
    expected_runtime_keys = {
        "runtime_contract_schema",
        "runtime_contract_sha256",
        "lane_eligibility_schema",
        "platform",
        "device_capability",
        "family",
        "structure",
        "rungs",
        "regime_routes",
        "requires_serve_flags",
    }
    if set(runtime) != expected_runtime_keys:
        raise RTX4090Qwen38PolicyError(
            f"{where}: runtime attestation fields differ from the closed schema"
        )
    validation_only = is_rtx4090_validation_only_policy(producer_policy)
    required_qualification = (
        "compile_only" if validation_only else "device_qualified"
    )
    if (
        not _is_sha256(runtime.get("runtime_contract_sha256"))
        or runtime.get("lane_eligibility_schema")
        != "gridbook.lane-eligibility.v2"
        or runtime.get("platform") != RTX4090_TARGET_PLATFORM
        or runtime.get("device_capability") != [8, 9]
        or runtime.get("family") != "FP8_CB_K"
        or runtime.get("structure") != "dense"
        or runtime.get("rungs") != list(FP8_PRODUCT_RUNGS)
        or runtime.get("requires_serve_flags") != []
    ):
        raise RTX4090Qwen38PolicyError(
            f"{where}: runtime attestation is not the exact clean sm89 dense "
            "full-ladder FP8-CB route"
        )
    rows = runtime.get("regime_routes")
    expected_coordinates = {
        (rung, regime)
        for rung in FP8_PRODUCT_RUNGS
        for regime in ("decode", "batch")
    }
    observed_coordinates: set[tuple[int, str]] = set()
    if not isinstance(rows, list):
        raise RTX4090Qwen38PolicyError(
            f"{where}: runtime attestation regime_routes must be a list"
        )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "rung",
            "regime",
            "cell_id",
            "route_status",
            "qualification",
            "requires_serve_flags",
        }:
            raise RTX4090Qwen38PolicyError(
                f"{where}: regime_routes[{index}] differs from the closed schema"
            )
        coordinate = (row.get("rung"), row.get("regime"))
        if (
            type(coordinate[0]) is not int
            or coordinate[1] not in {"decode", "batch"}
            or not isinstance(row.get("cell_id"), str)
            or not row.get("cell_id")
            or row.get("route_status") != "backed"
            or row.get("qualification") != required_qualification
            or row.get("requires_serve_flags") != []
        ):
            raise RTX4090Qwen38PolicyError(
                f"{where}: regime_routes[{index}] is not a clean, native, "
                f"{required_qualification} route"
            )
        observed_coordinates.add(coordinate)
    if len(observed_coordinates) != len(rows) or (
        observed_coordinates != expected_coordinates
    ):
        raise RTX4090Qwen38PolicyError(
            f"{where}: runtime route coordinates do not cover each full-ladder "
            "decode/batch cell exactly once"
        )
    return expected


def rtx4090_route_status_summary(
    payload: Mapping[str, Any],
    *,
    producer_policy: Mapping[str, Any],
    assignment: Mapping[str, str],
    where: str,
) -> dict[str, Any]:
    """Compact, validated strict route evidence for the shipcard bpp row."""

    validated = validate_rtx4090_route_status(
        payload,
        producer_policy=producer_policy,
        assignment=assignment,
        where=where,
    )
    runtime = validated["runtime_attestation"]
    qualification = (
        "compile_only"
        if is_rtx4090_validation_only_policy(producer_policy)
        else "device_qualified"
    )
    return {
        "schema": RTX4090_ROUTE_STATUS_SCHEMA,
        "authority": validated["authority"],
        "platform": runtime["platform"],
        "device_capability": runtime["device_capability"],
        "family": runtime["family"],
        "structure": runtime["structure"],
        "qualification": qualification,
        "route_status": "backed",
        "regimes": ["decode", "batch"],
        "full_ladder_rungs": runtime["rungs"],
        "selected_fp8_cb_units": validated["selected_fp8_cb_units"],
        "selected_fp8_cb_rungs": validated["selected_fp8_cb_rungs"],
        "delegated_fp8_e4m3_units": validated[
            "delegated_fp8_e4m3_units"
        ],
        "runtime_contract_sha256": runtime["runtime_contract_sha256"],
    }


def validate_rtx4090_quant_config_manifest(
    quant_config: Mapping[str, Any],
    *,
    runtime_contract: Mapping[str, Any] | None = None,
    require_policy_stamp: bool = True,
    allow_unreleasable_validation_only: bool = False,
    artifact_dir: str | Path | None = None,
    artifact_content_receipt: Mapping[str, Any] | None = None,
    where: str = "RTX 4090 quant_config.json",
) -> dict[str, Any]:
    """Validate the emitted Gridbook manifest, not just its input assignment."""

    if not isinstance(quant_config, Mapping):
        raise RTX4090Qwen38PolicyError(f"{where}: manifest must be an object")
    if set(quant_config) != _RTX4090_QUANT_CONFIG_KEYS:
        missing = sorted(_RTX4090_QUANT_CONFIG_KEYS - set(quant_config))
        extra = sorted(set(quant_config) - _RTX4090_QUANT_CONFIG_KEYS)
        raise RTX4090Qwen38PolicyError(
            f"{where}: top-level keys differ from the closed strict schema: "
            f"missing={missing}, extra={extra}"
        )
    if quant_config.get("quant_method") != "gridbook":
        raise RTX4090Qwen38PolicyError(
            f"{where}: quant_method must be 'gridbook'"
        )
    if quant_config.get("format") != "fp8_cb":
        raise RTX4090Qwen38PolicyError(
            f"{where}: top-level format must be 'fp8_cb'; the historical "
            "'nvfp4_cb' token is forbidden on the strict FP8-only artifact"
        )
    groups = quant_config.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise RTX4090Qwen38PolicyError(
            f"{where}: config_groups must be a nonempty object"
        )
    for name, group in groups.items():
        if not isinstance(group, Mapping):
            raise RTX4090Qwen38PolicyError(
                f"{where}.config_groups.{name}: group must be an object"
            )
        _validate_config_group(group, where=f"{where}.config_groups.{name}")

    provenance = quant_config.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RTX4090Qwen38PolicyError(f"{where}: provenance is required")
    unknown_provenance = sorted(set(provenance) - _RTX4090_PROVENANCE_KEYS)
    if unknown_provenance:
        raise RTX4090Qwen38PolicyError(
            f"{where}: unknown provenance fields are forbidden by the closed "
            f"strict schema: {unknown_provenance}"
        )
    if artifact_dir is not None:
        required_final_provenance = {
            "git_commit",
            "assignment_sha256",
            "imatrix_sha256",
            "codebook_sha256",
            "codebook_source",
            "scale_sweep",
            "scale_sweep_scope",
            "ldlq",
            "encode_tier",
            "renderer_abi",
            "scale_coding",
            "cb_targets",
            "stock_ct_targets",
            "fp8_source_targets",
            "source_passthrough_targets",
            "requant_native_targets",
            "serialized_payload",
            "render_identity_verified",
            "cb_render_identity",
            "tensor_formats",
            "producer_policy",
            "source_model_identity",
            "cb_route_status",
            "tensor_payload_identity",
            "artifact_inventory",
            "weight_content_manifest",
        }
        missing_final_provenance = sorted(
            required_final_provenance - set(provenance)
        )
        if missing_final_provenance:
            raise RTX4090Qwen38PolicyError(
                f"{where}: finalized provenance is incomplete: "
                f"{missing_final_provenance}"
            )
    tensor_formats = provenance.get("tensor_formats")
    if not isinstance(tensor_formats, Mapping) or not tensor_formats:
        raise RTX4090Qwen38PolicyError(
            f"{where}: provenance.tensor_formats is required and nonempty"
        )
    canonical_tensor_formats = validate_rtx4090_assignment(
        tensor_formats, where=f"{where}.tensor_formats"
    )
    assignment_sha256 = hashlib.sha256(json.dumps(
        dict(sorted(canonical_tensor_formats.items())),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if (
        "assignment_sha256" in provenance
        and provenance.get("assignment_sha256") != assignment_sha256
    ):
        raise RTX4090Qwen38PolicyError(
            f"{where}: provenance.assignment_sha256 disagrees with "
            "tensor_formats"
        )
    strict_provenance_values = {
        "codebook_source": "lattice",
        "scale_sweep": True,
        "scale_sweep_scope": "fp8",
        "ldlq": False,
        "encode_tier": "balanced",
        "renderer_abi": "prismaquant.nvfp4_cb_renderer.v1",
        "scale_coding": "v1",
        "fp8_source_targets": 0,
        "source_passthrough_targets": {},
        "requant_native_targets": {},
    }
    for key, value in strict_provenance_values.items():
        if key in provenance and provenance.get(key) != value:
            raise RTX4090Qwen38PolicyError(
                f"{where}: provenance.{key} must equal {value!r}"
            )
    if "streaming" in provenance and provenance.get("streaming") is not True:
        raise RTX4090Qwen38PolicyError(
            f"{where}: provenance.streaming, when present, must be true"
        )
    expected_cb_targets = sum(
        fmt.startswith("FP8_CB_K")
        for fmt in canonical_tensor_formats.values()
    )
    expected_stock_targets = sum(
        fmt == "FP8_E4M3" for fmt in canonical_tensor_formats.values()
    )
    for key, value in (
        ("cb_targets", expected_cb_targets),
        ("stock_ct_targets", expected_stock_targets),
    ):
        if key in provenance and provenance.get(key) != value:
            raise RTX4090Qwen38PolicyError(
                f"{where}: provenance.{key} disagrees with tensor_formats"
            )

    if artifact_dir is not None:
        if not _is_full_git_commit(provenance.get("git_commit")):
            raise RTX4090Qwen38PolicyError(
                f"{where}: provenance.git_commit must be a full lowercase "
                "40- or 64-hex commit identity"
            )
        if provenance.get("render_identity_verified") is not True:
            raise RTX4090Qwen38PolicyError(
                f"{where}: provenance.render_identity_verified must be bool true"
            )
        render_identity = provenance.get("cb_render_identity")
        if (
            not isinstance(render_identity, Mapping)
            or set(render_identity) != _RTX4090_RENDER_IDENTITY_KEYS
        ):
            raise RTX4090Qwen38PolicyError(
                f"{where}: cb_render_identity fields differ from the closed "
                "strict lattice-render schema"
            )
        expected_cb_formats = {
            qname: fmt
            for qname, fmt in canonical_tensor_formats.items()
            if fmt.startswith("FP8_CB_K")
        }
        try:
            from .production_weight_cache import (
                validate_cb_render_identity_metadata,
            )

            validate_cb_render_identity_metadata(
                render_identity,
                expected_qnames=sorted(expected_cb_formats),
                expected_formats_by_qname=expected_cb_formats,
                require_source_complete=True,
                where=f"{where}.provenance.cb_render_identity",
            )
        except ValueError as exc:
            raise RTX4090Qwen38PolicyError(str(exc)) from exc
        render_scope = render_identity.get("cb_formats_by_qname")
        if not isinstance(render_scope, Mapping) or any(
            not isinstance(formats, list)
            or not formats
            or any(
                not isinstance(fmt, str) or not fmt.startswith("FP8_CB_K")
                for fmt in formats
            )
            for formats in render_scope.values()
        ):
            raise RTX4090Qwen38PolicyError(
                f"{where}: cb_render_identity contains a non-FP8-CB format"
            )
        imatrix_sha256 = provenance.get("imatrix_sha256")
        if (
            not _is_sha256(imatrix_sha256)
            or imatrix_sha256 != render_identity.get("col_weights_sha256")
        ):
            raise RTX4090Qwen38PolicyError(
                f"{where}: provenance.imatrix_sha256 must bind the canonical "
                "value-bearing CB render imatrix"
            )

        codebook_refs = {
            str(ref)
            for group in groups.values()
            if str(group.get("format", "")).startswith("FP8_CB_K")
            for ref in group["scheme"]["codebook_ref"]
        }
        codebook_sha256 = provenance.get("codebook_sha256")
        if (
            not isinstance(codebook_sha256, Mapping)
            or set(codebook_sha256) != codebook_refs
            or any(not _is_sha256(value) for value in codebook_sha256.values())
            or any("fp4" in ref.lower() for ref in codebook_refs)
        ):
            raise RTX4090Qwen38PolicyError(
                f"{where}: provenance.codebook_sha256 must bind exactly the "
                "referenced FP8 lattice subtables"
            )

        if not isinstance(provenance.get("weight_content_manifest"), Mapping):
            raise RTX4090Qwen38PolicyError(
                f"{where}: finalized weight_content_manifest is required"
            )

    serialized_payload = provenance.get("serialized_payload")
    if not isinstance(serialized_payload, Mapping):
        raise RTX4090Qwen38PolicyError(
            f"{where}: provenance.serialized_payload is required"
        )
    serialized_context = serialized_payload.get("context")
    expected_serialized_context = {
        "scale_coding": "v1",
        "layout_version": 1,
        "codebook_source": "lattice",
        "scale_sweep": True,
        "scale_sweep_scope": "fp8",
        "ldlq": False,
        "encode_tier": "balanced",
        # This is the truthful historical shared-codec ABI, not an NVFP4
        # activation or weight-format claim.  Pin it rather than accepting an
        # arbitrary renderer spelling.
        "renderer_abi": "prismaquant.nvfp4_cb_renderer.v1",
    }
    if (
        serialized_payload.get("schema")
        != PREVIOUS_CB_SERIALIZED_PAYLOAD_SCHEMA
        or not isinstance(serialized_context, Mapping)
        or dict(serialized_context) != expected_serialized_context
    ):
        raise RTX4090Qwen38PolicyError(
            f"{where}: serialized_payload must be the lattice FP8-only "
            "no-activation context"
        )
    for key in (
        "fp4_scale_bytes",
        "input_global_scale_bytes",
        "global_scale_bytes",
    ):
        if type(serialized_payload.get(key)) is not int or (
            serialized_payload.get(key) != 0
        ):
            raise RTX4090Qwen38PolicyError(
                f"{where}: serialized_payload.{key} must be exactly zero"
            )

    # Gridbook's current quantized-embedding declaration only names NVFP4;
    # source-passthrough declarations name separate wire contracts. Neither is
    # part of this dense FP8-CB policy, so their presence is a hard refusal.
    for forbidden_key in (
        "quantized_embedding",
        "source_passthrough",
        "execution_contracts",
    ):
        if forbidden_key in quant_config:
            raise RTX4090Qwen38PolicyError(
                f"{where}: {forbidden_key} is outside the FP8-CB-only policy"
            )

    stamp = provenance.get("producer_policy")
    if require_policy_stamp:
        if not isinstance(stamp, Mapping):
            raise RTX4090Qwen38PolicyError(
                f"{where}: provenance.producer_policy is required"
            )
        validation_only = is_rtx4090_validation_only_policy(stamp)
        if validation_only and not allow_unreleasable_validation_only:
            raise RTX4090Qwen38PolicyError(
                f"{where}: {RTX4090_VALIDATION_ONLY_DISPOSITION} artifacts are "
                "structural-validation outputs and are categorically ineligible "
                "for the strict RTX4090 release validator"
            )
        expected = {
            "schema": (
                RTX4090_VALIDATION_ONLY_POLICY_SCHEMA
                if validation_only else RTX4090_QWEN38_POLICY_SCHEMA
            ),
            "id": (
                RTX4090_VALIDATION_ONLY_POLICY_ID
                if validation_only else RTX4090_QWEN38_POLICY_ID
            ),
            "serving_profile": (
                RTX4090_VALIDATION_ONLY_SERVING_PROFILE
                if validation_only else RTX4090_QWEN38_SERVING_PROFILE
            ),
            "compute_capability_sm": RTX4090_COMPUTE_CAPABILITY_SM,
            "target_platform": RTX4090_TARGET_PLATFORM,
            "context_first_target_bytes": RTX4090_CONTEXT_FIRST_TARGET_BYTES,
            "artifact_ceiling_bytes": (
                RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES
            ),
            "format_menu": list(RTX4090_QWEN38_FORMAT_MENU),
            "graph_requirement": rtx4090_graph_requirement(),
        }
        if validation_only:
            expected.update({
                "artifact_disposition": RTX4090_VALIDATION_ONLY_DISPOSITION,
                "runtime_qualification_ceiling": "compile_only",
                "build_host": "dgx_spark_gb10",
            })
        required_stamp_keys = set(expected) | {"runtime_attestation"}
        allowed_stamp_keys = required_stamp_keys | {"source_census"}
        if not required_stamp_keys.issubset(stamp) or not set(stamp).issubset(
            allowed_stamp_keys
        ):
            raise RTX4090Qwen38PolicyError(
                f"{where}: producer_policy fields differ from the closed "
                "strict schema"
            )
        source_census = stamp.get("source_census")
        if artifact_dir is not None and not isinstance(source_census, Mapping):
            raise RTX4090Qwen38PolicyError(
                f"{where}: finalized producer_policy.source_census is required"
            )
        if source_census is not None:
            from .cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA
            from .rtx4090_artifact_census import (
                RTX4090_SOURCE_CENSUS_SCHEMA,
            )

            if (
                not isinstance(source_census, Mapping)
                or set(source_census) != _RTX4090_SOURCE_CENSUS_KEYS
            ):
                raise RTX4090Qwen38PolicyError(
                    f"{where}: source_census fields differ from the closed schema"
                )
            compact_identity = source_census.get("source_model_identity")
            if (
                not isinstance(compact_identity, Mapping)
                or set(compact_identity)
                != _RTX4090_COMPACT_SOURCE_IDENTITY_KEYS
            ):
                raise RTX4090Qwen38PolicyError(
                    f"{where}: compact source identity fields differ from the "
                    "closed schema"
                )
            if (
                source_census.get("schema") != RTX4090_SOURCE_CENSUS_SCHEMA
                or compact_identity.get("schema")
                != STREAMED_MODEL_IDENTITY_SCHEMA
            ):
                raise RTX4090Qwen38PolicyError(
                    f"{where}: source census/identity schema is invalid"
                )
            if source_census.get("assignment_sha256") != assignment_sha256:
                raise RTX4090Qwen38PolicyError(
                    f"{where}: source_census.assignment_sha256 disagrees with "
                    "tensor_formats"
                )
            if source_census.get("source_layout") not in {
                "official_wrapper", "flattened_text"
            }:
                raise RTX4090Qwen38PolicyError(
                    f"{where}: source_census.source_layout is invalid"
                )
            for digest_key in (
                "source_config_sha256",
                "aura_staged_config_sha256",
                "aura_execution_config_sha256",
                "source_tensor_manifest_sha256",
                "assignment_sha256",
            ):
                digest = str(source_census.get(digest_key, "")).lower()
                if len(digest) != 64 or any(
                    char not in "0123456789abcdef" for char in digest
                ):
                    raise RTX4090Qwen38PolicyError(
                        f"{where}: source_census.{digest_key} is not SHA-256"
                    )
            if (
                type(source_census.get("source_tensor_count")) is not int
                or source_census.get("source_tensor_count") <= 0
                or type(source_census.get("source_linear_count")) is not int
                or source_census.get("source_linear_count") <= 0
                or compact_identity.get("checkpoint_tensors")
                != source_census.get("source_tensor_count")
                or type(compact_identity.get("checkpoint_shards")) is not int
                or compact_identity.get("checkpoint_shards") <= 0
            ):
                raise RTX4090Qwen38PolicyError(
                    f"{where}: source census/identity coverage counts are invalid"
                )
            compact_digest = str(
                compact_identity.get("content_sha256", "")
            ).lower()
            if len(compact_digest) != 64 or any(
                char not in "0123456789abcdef" for char in compact_digest
            ):
                raise RTX4090Qwen38PolicyError(
                    f"{where}: compact source identity has no content SHA-256"
                )
        for key, value in expected.items():
            if stamp.get(key) != value:
                raise RTX4090Qwen38PolicyError(
                    f"{where}: producer_policy.{key} must equal {value!r}"
                )
        if runtime_contract is None:
            raise RTX4090Qwen38PolicyError(
                f"{where}: the Gridbook v11 runtime contract is required to "
                "verify producer_policy.runtime_attestation"
            )
        runtime_resolver = (
            require_rtx4090_compile_only_runtime_contract
            if validation_only else require_rtx4090_runtime_contract
        )
        expected_runtime = runtime_resolver(
            runtime_contract,
            canonical_tensor_formats.values(),
            where=f"{where} Gridbook contract",
        )
        if stamp.get("runtime_attestation") != expected_runtime:
            raise RTX4090Qwen38PolicyError(
                f"{where}: producer_policy.runtime_attestation differs from "
                "the supplied Gridbook contract"
            )
        if artifact_dir is not None:
            validate_rtx4090_route_status(
                provenance.get("cb_route_status"),
                producer_policy=stamp,
                assignment=canonical_tensor_formats,
                where=f"{where}.provenance.cb_route_status",
            )

    inventory = provenance.get("artifact_inventory")
    if not isinstance(inventory, Mapping):
        raise RTX4090Qwen38PolicyError(
            f"{where}: final artifact_inventory is required"
        )
    artifact_bytes = inventory.get("export_directory_bytes")
    if type(artifact_bytes) is not int or artifact_bytes < 0:
        raise RTX4090Qwen38PolicyError(
            f"{where}: artifact inventory has no exact directory byte count"
        )
    if artifact_bytes > RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES:
        raise RTX4090Qwen38PolicyError(
            f"{where}: artifact is {artifact_bytes} bytes, above the context-"
            f"first ceiling {RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES}"
        )
    result = {
        "policy_id": (
            str(stamp.get("id"))
            if isinstance(stamp, Mapping) else RTX4090_QWEN38_POLICY_ID
        ),
        "formats": sorted(set(canonical_tensor_formats.values())),
        "artifact_bytes": artifact_bytes,
    }
    if artifact_dir is not None:
        from .rtx4090_artifact_census import (
            RTX4090ArtifactCensusError,
            validate_rtx4090_finalized_artifact_census,
        )

        try:
            result["source_census"] = (
                validate_rtx4090_finalized_artifact_census(
                    artifact_dir=artifact_dir,
                    quant_config=quant_config,
                    assignment=canonical_tensor_formats,
                    validate_config=validate_qwen38_dense_config,
                    where=where,
                    artifact_content_receipt=artifact_content_receipt,
                )
            )
        except RTX4090ArtifactCensusError as exc:
            raise RTX4090Qwen38PolicyError(str(exc)) from exc
    return result


def require_rtx4090_runtime_contract(
    contract: Mapping[str, Any],
    formats: Sequence[Any],
    *,
    where: str = "RTX 4090 Gridbook runtime contract",
) -> dict[str, Any]:
    """Require exact device-qualified dense FP8-CB sm89 execution cells.

    Graph evidence is intentionally external to the Gridbook contract and is
    replayed by the release gate.
    """

    # ``formats`` may be an assignment population with hundreds of repeated
    # values. The strict menu validator still rejects duplicate MENU entries;
    # this contract resolver consumes the assignment's unique selected set.
    canonical = validate_rtx4090_format_menu(
        tuple(dict.fromkeys(formats)), where=f"{where} selected formats"
    )
    selected_rungs = tuple(sorted({
        int(name.rsplit("K", 1)[1])
        for name in canonical
        if name.startswith("FP8_CB_K")
    }))
    if not selected_rungs:
        raise RTX4090Qwen38PolicyError(
            f"{where}: the FP8-CB campaign selected no FP8-CB rungs"
        )
    # Artifact assignments are deliberately free to select any legal subset,
    # but the candidate runtime qualifies the producer family, not one
    # allocation outcome.  Requiring the complete current producer set here
    # prevents a sparse assignment (for example K40 only) from blessing a Gridbook
    # release whose other advertised producer rungs have no Ada route.
    qualified_rungs = FP8_PRODUCT_RUNGS
    try:
        attestation = require_device_qualified_gridbook_routes(
            contract,
            family="FP8_CB_K",
            device_capability=(8, 9),
            structure="dense",
            rungs=qualified_rungs,
            where=where,
        )
    except GridbookExecutionContractError as exc:
        raise RTX4090Qwen38PolicyError(str(exc)) from exc
    return {
        "runtime_contract_schema": str(contract.get("schema")),
        "runtime_contract_sha256": hashlib.sha256(
            json.dumps(
                contract,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        **attestation.as_dict(),
    }


def require_rtx4090_compile_only_runtime_contract(
    contract: Mapping[str, Any],
    formats: Sequence[Any],
    *,
    where: str = "RTX 4090 validation-only Gridbook runtime contract",
) -> dict[str, Any]:
    """Require the exact full-ladder compile-only SM89 structural contract.

    This resolver is intentionally separate from
    :func:`require_rtx4090_runtime_contract`.  It can authorize only an
    immutable ``UNRELEASABLE_VALIDATION_ONLY`` artifact stamp and is never a
    device or serving qualification.
    """

    canonical = validate_rtx4090_format_menu(
        tuple(dict.fromkeys(formats)), where=f"{where} selected formats"
    )
    if not any(name.startswith("FP8_CB_K") for name in canonical):
        raise RTX4090Qwen38PolicyError(
            f"{where}: the FP8-CB campaign selected no FP8-CB rungs"
        )
    try:
        attestation = require_compile_only_gridbook_routes(
            contract,
            family="FP8_CB_K",
            device_capability=(8, 9),
            structure="dense",
            rungs=FP8_PRODUCT_RUNGS,
            where=where,
        )
    except GridbookExecutionContractError as exc:
        raise RTX4090Qwen38PolicyError(str(exc)) from exc
    if attestation.requires_serve_flags:
        raise RTX4090Qwen38PolicyError(
            f"{where}: compile-only validation cells must be flag-free"
        )
    return {
        "runtime_contract_schema": str(contract.get("schema")),
        "runtime_contract_sha256": hashlib.sha256(
            json.dumps(
                contract,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        **attestation.as_dict(),
    }


def producer_policy_stamp(
    runtime_contract: Mapping[str, Any],
    formats: Sequence[Any],
) -> dict[str, Any]:
    """Producer intent plus full-ladder Gridbook route qualification.

    ``formats`` is the artifact's selected subset.  It is validated for the
    strict campaign, while the embedded runtime attestation always covers the
    complete FP8 producer ladder in both serving regimes.
    """

    return {
        "schema": RTX4090_QWEN38_POLICY_SCHEMA,
        "id": RTX4090_QWEN38_POLICY_ID,
        "serving_profile": RTX4090_QWEN38_SERVING_PROFILE,
        "compute_capability_sm": RTX4090_COMPUTE_CAPABILITY_SM,
        "target_platform": RTX4090_TARGET_PLATFORM,
        "context_first_target_bytes": RTX4090_CONTEXT_FIRST_TARGET_BYTES,
        "artifact_ceiling_bytes": RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES,
        "format_menu": list(RTX4090_QWEN38_FORMAT_MENU),
        "graph_requirement": rtx4090_graph_requirement(),
        "runtime_attestation": require_rtx4090_runtime_contract(
            runtime_contract, formats
        ),
    }


def validation_only_producer_policy_stamp(
    runtime_contract: Mapping[str, Any],
    formats: Sequence[Any],
) -> dict[str, Any]:
    """Stamp a GB10-built structural artifact as permanently unreleasable."""

    return {
        "schema": RTX4090_VALIDATION_ONLY_POLICY_SCHEMA,
        "id": RTX4090_VALIDATION_ONLY_POLICY_ID,
        "serving_profile": RTX4090_VALIDATION_ONLY_SERVING_PROFILE,
        "compute_capability_sm": RTX4090_COMPUTE_CAPABILITY_SM,
        "target_platform": RTX4090_TARGET_PLATFORM,
        "context_first_target_bytes": RTX4090_CONTEXT_FIRST_TARGET_BYTES,
        "artifact_ceiling_bytes": RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES,
        "format_menu": list(RTX4090_QWEN38_FORMAT_MENU),
        "graph_requirement": rtx4090_graph_requirement(),
        "artifact_disposition": RTX4090_VALIDATION_ONLY_DISPOSITION,
        "runtime_qualification_ceiling": "compile_only",
        "build_host": "dgx_spark_gb10",
        "runtime_attestation": require_rtx4090_compile_only_runtime_contract(
            runtime_contract, formats
        ),
    }


__all__ = [
    "RTX4090_COMPUTE_CAPABILITY_SM",
    "RTX4090_CONTEXT_FIRST_ARTIFACT_CEILING_BYTES",
    "RTX4090_CONTEXT_FIRST_TARGET_BYTES",
    "RTX4090_QWEN38_ALLOWED_FORMATS",
    "RTX4090_QWEN38_FORMAT_MENU",
    "RTX4090_QWEN38_POLICY_ID",
    "RTX4090_QWEN38_POLICY_SCHEMA",
    "RTX4090_VALIDATION_ONLY_DISPOSITION",
    "RTX4090_VALIDATION_ONLY_POLICY_ID",
    "RTX4090_VALIDATION_ONLY_POLICY_SCHEMA",
    "RTX4090_VALIDATION_ONLY_SERVING_PROFILE",
    "RTX4090_ROUTE_STATUS_SCHEMA",
    "RTX4090_QWEN38_LAYER_TYPES",
    "RTX4090_QWEN38_SERVING_PROFILE",
    "RTX4090_TARGET_PLATFORM",
    "RTX4090Qwen38PolicyError",
    "load_rtx4090_runtime_contract",
    "is_rtx4090_validation_only_policy",
    "prepare_rtx4090_export_policy",
    "producer_policy_stamp",
    "rtx4090_route_status_stamp",
    "rtx4090_route_status_summary",
    "rtx4090_graph_requirement",
    "require_rtx4090_runtime_contract",
    "require_rtx4090_compile_only_runtime_contract",
    "validate_qwen38_dense_config",
    "validate_rtx4090_assignment",
    "validate_rtx4090_format_menu",
    "validate_rtx4090_quant_config_manifest",
    "validate_rtx4090_route_status",
    "validation_only_producer_policy_stamp",
]
