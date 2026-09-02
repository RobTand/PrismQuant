#!/usr/bin/env python3
"""Build the ship-safe hybrid recipe for the DeepSeek-V4 DSpark sidecar.

This is intentionally a weight-only draft experiment: every decoder Linear in
the three physical ``mtp.{0,1,2}`` stages receives the requested CB format
except ``attn.wo_a``.  ``wo_a`` is a grouped BMM whose algebra is not provided
by Gridbook's generic CB Linear method, so all three bases remain exact source
FP8 W8A16 passthroughs.  ``main_proj``, router gates, Markov/confidence heads,
norms, sinks, and hyper-connection glue also retain their source
representation.  ``main_proj`` is added by the exporter as the fourth explicit
source W8A16 route; this builder never calls any of those four bases BF16.

The source checkpoint's closed 4,705-tensor layout is validated before either
output is written.  Routed experts remain expanded in allocator vocabulary
(``gate_proj/up_proj/down_proj``); the streaming exporter collapses them into
the two physical Gridbook stacks only after checking uniformity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch

# Make the documented direct invocation work from any current directory while
# still importing this checkout, not an older installed PrismaQuant wheel.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from prismaquant.cb_layout import parse_format_name
from prismaquant.dspark_source_metadata import (
    FP8_BLOCK_UE8M0_SOURCE_FORMAT,
    discover_dspark_source_overlay,
    dspark_cb_source_passthrough_mapping,
    dspark_cb_physical_source_for_recipe_target,
)
from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA
from prismaquant.export_nvfp4_cb_streaming import (
    _LazySkeleton,
    _source_model_identity_from_env,
)
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    CB_ASSIGNMENT_IDENTITIES_FIELD,
    cb_assignment_serialization_stamps,
    cb_serialization_context_from_env,
)
from prismaquant.production_weight_cache import (
    build_production_cache_cb_render_identity,
)
from prismaquant.nvfp4_cb_footprint import assignment_serialization_sha256


_EXPERT_SOURCE_RE = re.compile(
    r"^(mtp[.](?P<stage>\d+)[.]ffn[.]experts)[.]"
    r"(?P<expert>\d+)[.](?P<leaf>w1|w2|w3)$"
)
_RECIPE_PROJECTION = {
    "w1": "gate_proj",
    "w3": "up_proj",
    "w2": "down_proj",
}

_RELEASED_CB_DECODER_LINEAR_COUNT = 2_325
_RELEASED_CB_QUANTIZED_PARAMETERS = 19_623_051_264
_DSPARK_RENDER_RECIPE_SCHEMA = "prismaquant.dspark_cb_render_recipe.v1"
_DSPARK_RENDER_SOURCE_BINDING = "streamed_decoded_cb_source.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _validate_source_model_identity(
    value: Mapping[str, object] | None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(
            "production DSpark recipe requires the complete source-checkpoint "
            "identity cache via PRISMAQUANT_STREAMED_MODEL_IDENTITY_CACHE"
        )
    required = {
        "schema",
        "content_sha256",
        "resolved_commit",
        "checkpoint_shards",
        "checkpoint_tensors",
    }
    if set(value) != required:
        raise RuntimeError(
            "DSpark source identity must be the exact compact complete-"
            f"checkpoint projection; got keys={sorted(value)}"
        )
    if value.get("schema") != STREAMED_MODEL_IDENTITY_SCHEMA:
        raise RuntimeError("DSpark source identity schema is not supported")
    if re.fullmatch(
        r"[0-9a-f]{64}", str(value.get("content_sha256", ""))
    ) is None:
        raise RuntimeError("DSpark source identity has no exact content SHA-256")
    for name in ("checkpoint_shards", "checkpoint_tensors"):
        member = value.get(name)
        if isinstance(member, bool) or not isinstance(member, int) or member <= 0:
            raise RuntimeError(f"DSpark source identity {name} is not positive")
    resolved_commit = value.get("resolved_commit")
    if resolved_commit is not None and (
        not isinstance(resolved_commit, str) or not resolved_commit
    ):
        raise RuntimeError("DSpark source identity resolved_commit is malformed")
    return dict(value)


def _recipe_qname(physical_base: str) -> str:
    match = _EXPERT_SOURCE_RE.fullmatch(physical_base)
    if match is None:
        return physical_base
    return (
        f"{match.group(1)}.{int(match.group('expert'))}."
        f"{_RECIPE_PROJECTION[match.group('leaf')]}"
    )


def build(
    source: Path,
    out_dir: Path,
    format_name: str,
    *,
    source_model_identity: Mapping[str, object] | None = None,
    serialization_context: CBSerializationContext | None = None,
) -> dict:
    parsed = parse_format_name(format_name)
    if parsed is None:
        raise ValueError(f"unsupported CB format {format_name!r}")
    format_name = format_name.upper()
    if out_dir.exists():
        raise FileExistsError(
            f"refusing to replace existing DSpark input directory {out_dir}"
        )
    source_model_identity = _validate_source_model_identity(
        source_model_identity
        if source_model_identity is not None
        else _source_model_identity_from_env(source)
    )
    serialization_context = (
        serialization_context
        if serialization_context is not None
        else cb_serialization_context_from_env(
            require_explicit=True,
            where="DSpark production render recipe",
        )
    )
    # This release's DSpark sidecar is deliberately W4A16.  Preserve every
    # weight-render lever from the explicit producer environment while
    # removing the static-W4A4 execution claim that would require an MTP
    # activation calibration scan and physical/construction scale bridge.
    serialization_context = replace(
        serialization_context,
        activation_contract=None,
        activation_execution=None,
    )

    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    skeleton = _LazySkeleton(source)
    overlay = discover_dspark_source_overlay(skeleton, config)
    if overlay is None:
        raise ValueError(
            f"{source}: not the released DeepSeek-V4 three-stage DSpark source"
        )

    assignment: dict[str, str] = {}
    cb_targets: set[str] = set()
    col_weights: dict[str, torch.Tensor] = {}
    logical_shapes: dict[str, list[int]] = {}
    quantized_parameters = 0
    source_mapping = dspark_cb_source_passthrough_mapping(config)
    source_decoder_targets = set(source_mapping) - {"mtp.0.main_proj"}
    for physical_base in sorted(overlay.physical_targets):
        if physical_base == "mtp.0.main_proj":
            continue
        qname = _recipe_qname(physical_base)
        resolved = dspark_cb_physical_source_for_recipe_target(qname, config)
        if resolved != physical_base:
            raise AssertionError(
                f"{qname}: resolved {resolved!r}, expected {physical_base!r}"
            )
        weight_key = physical_base + ".weight"
        shape = tuple(int(value) for value in skeleton.logical_shape(weight_key))
        if len(shape) != 2:
            raise ValueError(f"{weight_key}: expected rank-2 Linear, got {shape}")
        if shape[1] % 256:
            raise ValueError(
                f"{qname}: in_features={shape[1]} is not CB-superblock aligned"
            )
        logical_shapes[qname] = list(shape)
        if physical_base in source_decoder_targets:
            source_format = overlay.physical_targets[physical_base]
            if source_format != FP8_BLOCK_UE8M0_SOURCE_FORMAT:
                raise ValueError(
                    f"{physical_base}: hybrid wo_a route requires "
                    f"{FP8_BLOCK_UE8M0_SOURCE_FORMAT}, got {source_format}"
                )
            assignment[qname] = source_format
        else:
            assignment[qname] = format_name
            cb_targets.add(qname)
            col_weights[qname] = torch.ones(shape[1], dtype=torch.float32)
            quantized_parameters += shape[0] * shape[1]

    expected_recipe_members = overlay.n_mtp_layers * (
        8 + int(config["n_routed_experts"]) * 3
    )
    expected_cb_members = overlay.n_mtp_layers * (
        7 + int(config["n_routed_experts"]) * 3
    )
    if len(assignment) != expected_recipe_members:
        raise AssertionError(
            f"DSpark recipe has {len(assignment)} decoder Linears, expected "
            f"{expected_recipe_members}"
        )
    if len(cb_targets) != expected_cb_members:
        raise AssertionError(
            f"DSpark recipe has {len(cb_targets)} CB decoder Linears, expected "
            f"{expected_cb_members}"
        )
    observed_source_decoder = {
        qname for qname, fmt in assignment.items()
        if fmt == FP8_BLOCK_UE8M0_SOURCE_FORMAT
    }
    if observed_source_decoder != source_decoder_targets:
        raise AssertionError(
            "DSpark recipe source-FP8 decoder targets differ from the exact "
            f"wo_a set: {sorted(observed_source_decoder)}"
        )
    observed_stages = {
        int(qname.split(".", 2)[1]) for qname in assignment
    }
    if observed_stages != set(range(overlay.n_mtp_layers)):
        raise AssertionError(
            f"DSpark recipe stages are incomplete: {sorted(observed_stages)}"
        )

    # The released 256-expert Flash checkpoint has a fixed, reviewed count and
    # CB parameter total.  Keep both as executable tripwires so a geometry or
    # role-vocabulary drift cannot quietly change the K12 draft's size.
    if (
        int(config.get("n_routed_experts", -1)) == 256
        and int(config.get("hidden_size", -1)) == 4096
        and int(config.get("num_attention_heads", -1)) == 64
        and int(config.get("head_dim", -1)) == 512
        and int(config.get("q_lora_rank", -1)) == 1024
        and int(config.get("o_groups", -1)) == 8
        and int(config.get("o_lora_rank", -1)) == 1024
        and int(config.get("moe_intermediate_size", -1)) == 2048
    ):
        if len(cb_targets) != _RELEASED_CB_DECODER_LINEAR_COUNT:
            raise AssertionError(
                "released DSv4 hybrid recipe must contain exactly "
                f"{_RELEASED_CB_DECODER_LINEAR_COUNT} CB decoder Linears"
            )
        if quantized_parameters != _RELEASED_CB_QUANTIZED_PARAMETERS:
            raise AssertionError(
                "released DSv4 hybrid recipe must contain exactly "
                f"{_RELEASED_CB_QUANTIZED_PARAMETERS} CB parameters, got "
                f"{quantized_parameters}"
            )

    header_identity = {
        name: {
            "dtype": str(skeleton.get_dtype(name)),
            "shape": list(skeleton.get_shape(name)),
        }
        for name in sorted(skeleton.keys())
        if str(name).startswith("mtp.")
    }
    header_sha256 = hashlib.sha256(json.dumps(
        header_identity, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()

    render_identity_seed = build_production_cache_cb_render_identity(
        {qname: (format_name,) for qname in sorted(cb_targets)},
        cb_serialization_context=serialization_context,
        col_weights=col_weights,
        render_levers={"weighted_vq": True},
        render_mechanism_plan=[],
    )
    if render_identity_seed is None:
        raise AssertionError("DSpark CB recipe produced no render identity")
    render_identity_seed_sha256 = _canonical_sha256(render_identity_seed)
    serialization_stamps = cb_assignment_serialization_stamps(
        assignment,
        logical_shapes,
        context=serialization_context,
    )
    render_recipe = {
        "schema": _DSPARK_RENDER_RECIPE_SCHEMA,
        "source_binding": _DSPARK_RENDER_SOURCE_BINDING,
        "source_model_identity": source_model_identity,
        "source_config_sha256": _sha256(config_path),
        "mtp_header_identity_sha256": header_sha256,
        "assignment_sha256": assignment_serialization_sha256(assignment),
        "col_weights_sha256": render_identity_seed["col_weights_sha256"],
        "render_identity_seed_sha256": render_identity_seed_sha256,
    }

    out_dir.mkdir(parents=True)
    assignment_path = out_dir / "dspark_layer_config.json"
    col_weights_path = out_dir / "dspark_col_weights.pkl"
    manifest_path = out_dir / "dspark_inputs_manifest.json"
    recipe_payload: dict[str, object] = dict(assignment)
    recipe_payload["__prismaquant__"] = {
        "cb_serialized_payload": render_identity_seed[
            "cb_serialized_payload"
        ],
        "cb_render_identity": render_identity_seed,
        CB_ASSIGNMENT_IDENTITIES_FIELD: serialization_stamps,
        "dspark_render_recipe": render_recipe,
    }
    assignment_path.write_text(
        json.dumps(recipe_payload, indent=2, sort_keys=True) + "\n"
    )
    with col_weights_path.open("wb") as handle:
        pickle.dump(col_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "schema": "prismaquant.dspark_cb_sidecar_inputs.v2",
        "source": str(source.resolve()),
        "source_config_sha256": _sha256(config_path),
        "mtp_header_identity_sha256": header_sha256,
        "mtp_tensor_count": len(header_identity),
        "num_hidden_layers": overlay.num_hidden_layers,
        "n_mtp_layers": overlay.n_mtp_layers,
        "n_routed_experts": int(config["n_routed_experts"]),
        "cb_format": format_name,
        "h_source": "uniform (no MTP Fisher or activation scan)",
        "activation_contract": None,
        "decoder_recipe_entry_count": len(assignment),
        "decoder_linear_count": len(cb_targets),
        "source_passthrough_decoder_linear_count": len(
            observed_source_decoder
        ),
        "quantized_parameters": quantized_parameters,
        "main_proj_route": "FP8_BLOCK_UE8M0_SOURCE/W8A16",
        "source_passthrough_targets": dict(sorted(source_mapping.items())),
        "assignment_file": assignment_path.name,
        "assignment_sha256": _sha256(assignment_path),
        "col_weights_file": col_weights_path.name,
        "col_weights_sha256": _sha256(col_weights_path),
        "logical_shapes": logical_shapes,
        "source_model_identity": source_model_identity,
        "dspark_render_recipe": render_recipe,
        "render_identity_seed_sha256": render_identity_seed_sha256,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="build DeepSeek-V4 DSpark CB sidecar inputs"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--format", default="NVFP4_CB_K12")
    args = parser.parse_args(argv)
    manifest = build(args.source, args.out_dir, args.format)
    print(json.dumps({
        key: manifest[key]
        for key in (
            "cb_format",
            "decoder_recipe_entry_count",
            "decoder_linear_count",
            "source_passthrough_decoder_linear_count",
            "quantized_parameters",
            "main_proj_route",
            "h_source",
        )
    }, indent=2))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
